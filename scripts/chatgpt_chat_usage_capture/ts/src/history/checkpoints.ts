import type {
  DiscoveryCheckpoint,
  HistoryCheckpointStore,
  HistoryScope,
  RevisitEntry,
} from "../contracts/history.js";
import { HISTORY_STATE_VERSION } from "../contracts/history.js";
import { collectorScopeKey } from "../ledger/identity.js";
import type { Ledger } from "../ledger/store.js";
import type { LedgerScope } from "../ledger/types.js";
import { assertNoSecrets } from "../security/sanitizer.js";

interface PersistedHistoryState {
  stateVersion: typeof HISTORY_STATE_VERSION;
  accountId: string;
  checkpoints: Partial<Record<HistoryScope, DiscoveryCheckpoint>>;
  revisits: RevisitEntry[];
}

export class MemoryCheckpointStore implements HistoryCheckpointStore {
  private readonly checkpoints = new Map<HistoryScope, DiscoveryCheckpoint>();
  private readonly revisits = new Map<string, RevisitEntry>();

  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null {
    const checkpoint = this.checkpoints.get(scope);
    return checkpoint ? clone(checkpoint) : null;
  }

  saveDiscovery(checkpoint: DiscoveryCheckpoint): void {
    this.checkpoints.set(checkpoint.scope, clone(checkpoint));
  }

  listRevisits(): RevisitEntry[] {
    return [...this.revisits.values()]
      .filter((entry) => entry.status === "pending")
      .map((entry) => clone(normalizeRevisit(entry)))
      .sort((left, right) =>
        left.nextEligibleAt.localeCompare(right.nextEligibleAt) ||
        left.conversationId.localeCompare(right.conversationId),
      );
  }

  upsertRevisit(entry: RevisitEntry): void {
    this.revisits.set(entry.conversationId, clone(normalizeRevisit(entry)));
  }

  completeRevisit(accountId: string, conversationId: string): void {
    const entry = this.revisits.get(conversationId);
    if (entry?.accountId === accountId) {
      this.revisits.delete(conversationId);
    }
  }
}

/**
 * Buffer acquisition state until its evidence can be committed in the same
 * SQLite transaction. No database write lock is held across browser requests.
 */
export class SqliteCheckpointStore extends MemoryCheckpointStore {
  constructor(
    private readonly ledger: Ledger,
    private readonly scope: LedgerScope,
  ) {
    super();
    const row = ledger.db
      .prepare("SELECT state_json FROM history_state WHERE scope_key=?")
      .get(collectorScopeKey(scope)) as { state_json: string } | undefined;
    if (!row) {
      return;
    }
    const state = JSON.parse(row.state_json) as PersistedHistoryState;
    if (
      state.stateVersion !== HISTORY_STATE_VERSION ||
      state.accountId !== scope.collectorAccountId ||
      !state.checkpoints ||
      !Array.isArray(state.revisits)
    ) {
      throw new Error("history checkpoint state is invalid");
    }
    for (const checkpoint of Object.values(state.checkpoints)) {
      if (checkpoint) {
        this.saveDiscovery(checkpoint);
      }
    }
    for (const revisit of state.revisits) {
      this.upsertRevisit(revisit);
    }
  }

  persist(): void {
    if (!this.ledger.db.inTransaction) {
      throw new Error("history checkpoints must commit with collected evidence");
    }
    const state: PersistedHistoryState = {
      stateVersion: HISTORY_STATE_VERSION,
      accountId: this.scope.collectorAccountId,
      checkpoints: {},
      revisits: this.listRevisits(),
    };
    for (const scope of ["active", "archived"] as const) {
      const checkpoint = this.loadDiscovery(scope);
      if (checkpoint) {
        state.checkpoints[scope] = checkpoint;
      }
    }
    assertNoSecrets(state);
    this.ledger.db
      .prepare(`
        INSERT INTO history_state(scope_key, collector_account_id, state_json)
        VALUES (?, ?, ?)
        ON CONFLICT(scope_key) DO UPDATE SET state_json=excluded.state_json
      `)
      .run(collectorScopeKey(this.scope), this.scope.collectorAccountId, JSON.stringify(state));
  }
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function normalizeRevisit(entry: RevisitEntry): RevisitEntry {
  return {
    ...entry,
    continuation:
      typeof entry.continuation === "string" && entry.continuation.trim()
        ? entry.continuation.trim()
        : null,
    detailPagesFetched:
      Number.isInteger(entry.detailPagesFetched) && entry.detailPagesFetched >= 0
        ? entry.detailPagesFetched
        : 0,
  };
}
