import type {
  DiscoveryCheckpoint,
  HistoryAccountState,
  HistoryCheckpointStore,
  HistoryScope,
  OutstandingGenerationState,
  RevisitPageIssue,
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
  accountState?: HistoryAccountState;
  checkpoints: Partial<Record<HistoryScope, DiscoveryCheckpoint>>;
  revisits: RevisitEntry[];
}

export class MemoryCheckpointStore implements HistoryCheckpointStore {
  private accountState: HistoryAccountState = readyAccountState();
  private readonly checkpoints = new Map<HistoryScope, DiscoveryCheckpoint>();
  private readonly revisits = new Map<string, RevisitEntry>();

  loadAccountState(): HistoryAccountState {
    return clone(this.accountState);
  }

  saveAccountState(state: HistoryAccountState): void {
    this.accountState = clone(normalizeAccountState(state));
  }

  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null {
    const checkpoint = this.checkpoints.get(scope);
    return checkpoint ? clone(checkpoint) : null;
  }

  saveDiscovery(checkpoint: DiscoveryCheckpoint): void {
    this.checkpoints.set(checkpoint.scope, clone(checkpoint));
  }

  acknowledgeCandidates(
    conversationId: string,
    scopes: readonly HistoryScope[],
  ): void {
    const acknowledgedScopes = new Set(scopes);
    for (const scope of acknowledgedScopes) {
      const checkpoint = this.checkpoints.get(scope);
      if (!checkpoint?.candidateQueue) {
        continue;
      }
      this.saveDiscovery({
        ...checkpoint,
        candidateQueue: checkpoint.candidateQueue.filter(
          (candidate) => candidate.summary.conversationId !== conversationId,
        ),
      });
    }
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
    this.saveAccountState(normalizeAccountState(state.accountState));
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
      accountState: this.loadAccountState(),
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
  const continuation =
    typeof entry.continuation === "string" && entry.continuation.trim()
      ? entry.continuation.trim()
      : null;
  return {
    ...entry,
    continuation,
    continuationRevision:
      continuation !== null &&
      typeof entry.continuationRevision === "string" &&
      entry.continuationRevision.trim()
        ? entry.continuationRevision.trim()
        : null,
    detailPagesFetched:
      Number.isInteger(entry.detailPagesFetched) && entry.detailPagesFetched >= 0
        ? entry.detailPagesFetched
        : 0,
    malformedPage: normalizePageIssue(entry.malformedPage),
    outstandingGeneration: normalizeOutstandingGeneration(
      entry.outstandingGeneration,
    ),
  };
}

function normalizePageIssue(
  issue: RevisitPageIssue | null | undefined,
): RevisitPageIssue | null {
  if (!issue || typeof issue.reason !== "string") {
    return null;
  }
  return {
    reason: issue.reason as RevisitPageIssue["reason"],
    warnings: Array.isArray(issue.warnings)
      ? issue.warnings.filter(
          (warning): warning is string => typeof warning === "string",
        )
      : [],
  };
}

function normalizeOutstandingGeneration(
  state: OutstandingGenerationState | null | undefined,
): OutstandingGenerationState | null {
  if (
    !state ||
    (state.state !== "nonterminal" && state.state !== "unknown") ||
    typeof state.since !== "string" ||
    !state.since.trim()
  ) {
    return null;
  }
  return {
    state: state.timedOut ? "unknown" : state.state,
    since: state.since,
    timedOut: state.timedOut === true,
    messageId:
      typeof state.messageId === "string" && state.messageId.trim()
        ? state.messageId.trim()
        : null,
    generationId:
      typeof state.generationId === "string" && state.generationId.trim()
        ? state.generationId.trim()
        : null,
    requestId:
      typeof state.requestId === "string" && state.requestId.trim()
        ? state.requestId.trim()
        : null,
  };
}

function readyAccountState(): HistoryAccountState {
  return {
    status: "ready",
    reason: null,
    pausedAt: null,
    cooldownUntil: null,
    lastError: null,
  };
}

function normalizeAccountState(
  state: HistoryAccountState | undefined,
): HistoryAccountState {
  if (!state) {
    return readyAccountState();
  }
  if (state.status === "ready") {
    return readyAccountState();
  }
  if (
    state.status !== "paused" ||
    (state.reason !== "authentication" && state.reason !== "cooldown")
  ) {
    throw new Error("history account state is invalid");
  }
  if (
    state.pausedAt !== null &&
    !Number.isFinite(new Date(state.pausedAt).getTime())
  ) {
    throw new Error("history account pause timestamp is invalid");
  }
  if (
    state.cooldownUntil !== null &&
    !Number.isFinite(new Date(state.cooldownUntil).getTime())
  ) {
    throw new Error("history account cooldown deadline is invalid");
  }
  return {
    status: "paused",
    reason: state.reason,
    pausedAt: state.pausedAt,
    cooldownUntil: state.cooldownUntil,
    lastError: state.lastError ?? null,
  };
}
