import type {
  DiscoveryCheckpoint,
  HistoryCheckpointStore,
  HistoryScope,
  RevisitEntry,
} from "../contracts/history.js";

export interface BridgeStateEnvelope {
  stateVersion: 1;
  collectorAccountId: string;
  stateVersionCounter: number;
  discovery: Partial<Record<HistoryScope, DiscoveryCheckpoint>>;
  revisits: RevisitEntry[];
}

export interface BridgePageMutation {
  pageCommitId: string;
  conversationId: string;
  scopes: readonly HistoryScope[];
  checkpoint?: DiscoveryCheckpoint;
  revisit?: RevisitEntry;
}

/**
 * Parent-owned hydration is an explicit operation, never implicit disk state.
 * The checkpoint store remains serial and mutable within one worker run.
 */
export class BridgeCheckpointStore implements HistoryCheckpointStore {
  private readonly discovery = new Map<HistoryScope, DiscoveryCheckpoint>();
  private readonly revisits = new Map<string, RevisitEntry>();

  hydrate(state: BridgeStateEnvelope | null): void {
    if (!state) {
      this.discovery.clear();
      this.revisits.clear();
      return;
    }
    validateEnvelope(state);
    const hydratedRevisits = new Map<string, RevisitEntry>();
    for (const revisit of state.revisits) {
      hydratedRevisits.set(revisit.conversationId, clone(normalizeRevisit(revisit)));
    }
    const hydratedDiscovery = new Map<HistoryScope, DiscoveryCheckpoint>();
    for (const scope of ["active", "archived"] as const) {
      const checkpoint = state.discovery?.[scope];
      if (checkpoint) {
        validateCheckpoint(checkpoint, state.collectorAccountId, scope);
        hydratedDiscovery.set(scope, clone(checkpoint));
      }
    }
    this.revisits.clear();
    this.discovery.clear();
    for (const [scope, checkpoint] of hydratedDiscovery) {
      this.discovery.set(scope, checkpoint);
    }
    for (const [conversationId, revisit] of hydratedRevisits) {
      this.revisits.set(conversationId, revisit);
    }
  }

  loadAccountState(): never {
    throw new Error("account state is owned by the collector bridge header");
  }

  saveAccountState(): never {
    throw new Error("account state is owned by the collector bridge header");
  }

  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null {
    const checkpoint = this.discovery.get(scope);
    return checkpoint ? clone(checkpoint) : null;
  }

  saveDiscovery(checkpoint: DiscoveryCheckpoint): void {
    validateCheckpoint(checkpoint);
    this.discovery.set(checkpoint.scope, clone(checkpoint));
  }

  acknowledgeCandidates(
    conversationId: string,
    scopes: readonly HistoryScope[],
  ): void {
    for (const scope of new Set(scopes)) {
      const checkpoint = this.discovery.get(scope);
      if (!checkpoint?.candidateQueue) {
        continue;
      }
      this.saveDiscovery({
        ...checkpoint,
      candidateQueue: checkpoint.candidateQueue.filter(
        (candidate: { summary: { conversationId: string } }) =>
          candidate.summary.conversationId !== conversationId,
      ),
      });
    }
  }

  listRevisits(): RevisitEntry[] {
    return [...this.revisits.values()]
      .filter((entry) => entry.status === "pending")
      .map((entry) => clone(normalizeRevisit(entry)))
      .sort(
        (left, right) =>
          left.nextEligibleAt.localeCompare(right.nextEligibleAt) ||
          left.conversationId.localeCompare(right.conversationId),
      );
  }

  upsertRevisit(entry: RevisitEntry): void {
    this.revisits.set(entry.conversationId, clone(normalizeRevisit(entry)));
  }

  completeRevisit(_accountId: string, conversationId: string): void {
    this.revisits.delete(conversationId);
  }

  snapshot(collectorAccountId: string, stateVersionCounter: number): BridgeStateEnvelope {
    const discovery: Partial<Record<HistoryScope, DiscoveryCheckpoint>> = {};
    for (const scope of ["active", "archived"] as const) {
      const checkpoint = this.discovery.get(scope);
      if (checkpoint) {
        discovery[scope] = clone(checkpoint);
      }
    }
    return {
      stateVersion: 1,
      collectorAccountId,
      stateVersionCounter,
      discovery,
      revisits: this.listRevisits(),
    };
  }
}

export function validateEnvelope(state: BridgeStateEnvelope): void {
  if (state.stateVersion !== 1 || typeof state.collectorAccountId !== "string") {
    throw new Error("collector state envelope is invalid");
  }
  if (!Number.isSafeInteger(state.stateVersionCounter) || state.stateVersionCounter < 0) {
    throw new Error("collector state version is invalid");
  }
  if (!isRecord(state.discovery) || !Array.isArray(state.revisits)) {
    throw new Error("collector state contents are invalid");
  }
  for (const revisit of state.revisits) {
    normalizeRevisit(revisit);
  }
}

function validateCheckpoint(
  checkpoint: DiscoveryCheckpoint,
  accountId?: string,
  scope?: HistoryScope,
): void {
  if (checkpoint.stateVersion !== 1 || typeof checkpoint.accountId !== "string") {
    throw new Error("discovery checkpoint is invalid");
  }
  if (accountId !== undefined && checkpoint.accountId !== accountId) {
    throw new Error("discovery checkpoint account differs from run envelope");
  }
  if (scope !== undefined && checkpoint.scope !== scope) {
    throw new Error("discovery checkpoint scope differs from envelope");
  }
  if (!validRange(checkpoint.range)) {
    throw new Error("discovery checkpoint range is invalid");
  }
}

function normalizeRevisit(entry: RevisitEntry): RevisitEntry {
  if (
    entry.stateVersion !== 1 ||
    typeof entry.accountId !== "string" ||
    typeof entry.conversationId !== "string" ||
    !entry.conversationId.trim()
  ) {
    throw new Error("revisit entry is invalid");
  }
  return {
    ...entry,
    continuation:
      typeof entry.continuation === "string" && entry.continuation.trim()
        ? entry.continuation.trim()
        : null,
    detailPagesFetched:
      Number.isSafeInteger(entry.detailPagesFetched) && entry.detailPagesFetched >= 0
        ? entry.detailPagesFetched
        : 0,
  };
}

function validRange(range: DiscoveryCheckpoint["range"]): boolean {
  const start = new Date(range.start).getTime();
  const end = new Date(range.end).getTime();
  return Number.isFinite(start) && Number.isFinite(end) && start < end;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function clone<T>(value: T): T {
  return structuredClone(value);
}
