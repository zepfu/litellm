import type {
  DiscoveryCheckpoint,
  HistoryCheckpointStore,
  HistoryDiscoveryPageCommit,
  HistoryPageCommit,
  HistoryScope,
  RevisitEntry,
} from "../contracts/history.js";
import type { HistoryAccountState } from "../contracts/history.js";
import type {
  IngestContext,
  ReconstructedAttempt,
} from "../ledger/types.js";

export interface BridgeStateEnvelope {
  stateVersion: 1;
  collectorAccountId: string;
  stateVersionCounter: number;
  discovery: Partial<Record<HistoryScope, DiscoveryCheckpoint>>;
  /**
   * Revisit rows are hydrated from the explicit candidate queue. They are
   * intentionally absent from the durable header/checkpoint payload.
   */
  revisits?: RevisitEntry[];
  accountState?: HistoryAccountState;
  queueCoverage?: "complete" | "partial";
  nextCursor?: string | null;
  hasMore?: boolean;
}

export type BridgeQueueMutationOperation = "replace" | "remove";

export interface BridgeCandidateMutation {
  candidateKey: string;
  operation: BridgeQueueMutationOperation;
  queueKind: "candidate" | "revisit";
  scope?: HistoryScope;
  payload?: Record<string, unknown>;
}

export interface BridgeCoverageMutation {
  sourceKind: string;
  sourceId: string;
  reason: string;
  state: "open" | "resolved";
  seenAt: string;
  details?: Record<string, unknown>;
}

export interface BridgeCheckpointMutation {
  kind: "history";
  value: BridgeStateEnvelope;
}

export interface BridgePageMutation {
  pageCommitId: string;
  conversationId: string;
  scopes: readonly HistoryScope[];
  expectedStateVersion?: number;
  source?: IngestContext;
  discovery?: HistoryDiscoveryPageCommit;
  page?: HistoryPageCommit;
  attempts?: readonly ReconstructedAttempt[];
  checkpointMutations?: BridgeCheckpointMutation;
  candidateMutations?: readonly BridgeCandidateMutation[];
  coverageMutations?: readonly BridgeCoverageMutation[];
  checkpoint?: DiscoveryCheckpoint;
  revisit?: RevisitEntry;
  accountState?: HistoryAccountState;
}

/**
 * Parent-owned hydration is an explicit operation, never implicit disk state.
 * The checkpoint store remains serial and mutable within one worker run.
 */
export class BridgeCheckpointStore implements HistoryCheckpointStore {
  private readonly discovery = new Map<HistoryScope, DiscoveryCheckpoint>();
  private readonly revisits = new Map<string, RevisitEntry>();
  private readonly pendingQueueMutations = new Map<
    string,
    BridgeCandidateMutation
  >();
  private accountState: HistoryAccountState = {
    status: "ready",
    reason: null,
    pausedAt: null,
    cooldownUntil: null,
    lastError: null,
  };

  hydrate(state: BridgeStateEnvelope | null): void {
    if (!state) {
      this.discovery.clear();
      this.revisits.clear();
      this.pendingQueueMutations.clear();
      this.accountState = {
        status: "ready",
        reason: null,
        pausedAt: null,
        cooldownUntil: null,
        lastError: null,
      };
      return;
    }
    validateEnvelope(state);
    if (state.accountState) {
      this.accountState = validateAccountState(state.accountState);
    }
    const hydratedRevisits = new Map<string, RevisitEntry>();
    for (const revisit of state.revisits ?? []) {
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
    this.pendingQueueMutations.clear();
    for (const [scope, checkpoint] of hydratedDiscovery) {
      this.discovery.set(scope, checkpoint);
    }
    for (const [conversationId, revisit] of hydratedRevisits) {
      this.revisits.set(conversationId, revisit);
    }
  }

  loadAccountState(): HistoryAccountState {
    return clone(this.accountState);
  }

  saveAccountState(state: HistoryAccountState): void {
    this.accountState = validateAccountState(state);
  }

  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null {
    const checkpoint = this.discovery.get(scope);
    return checkpoint ? clone(checkpoint) : null;
  }

  saveDiscovery(checkpoint: DiscoveryCheckpoint): void {
    validateCheckpoint(checkpoint);
    const previous = this.discovery.get(checkpoint.scope);
    this.recordCandidateMutations(
      checkpoint.scope,
      previous?.candidateQueue ?? [],
      checkpoint.candidateQueue ?? [],
    );
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
    const previous = this.revisits.get(entry.conversationId);
    const normalized = clone(normalizeRevisit(entry));
    if (!previous || !sameValue(previous, normalized)) {
      this.pendingQueueMutations.set(
        `revisit:${entry.conversationId}`,
        {
          candidateKey: `revisit:${entry.conversationId}`,
          operation: "replace",
          queueKind: "revisit",
          payload: { entry: normalized },
        },
      );
    }
    this.revisits.set(entry.conversationId, normalized);
  }

  completeRevisit(_accountId: string, conversationId: string): void {
    this.revisits.delete(conversationId);
    this.pendingQueueMutations.set(
      `revisit:${conversationId}`,
      {
        candidateKey: `revisit:${conversationId}`,
        operation: "remove",
        queueKind: "revisit",
      },
    );
  }

  pendingQueueMutationsSnapshot(): BridgeCandidateMutation[] {
    return [...this.pendingQueueMutations.values()].map((mutation) =>
      clone(mutation),
    );
  }

  acknowledgeQueueMutations(
    mutations: readonly BridgeCandidateMutation[],
  ): void {
    for (const mutation of mutations) {
      const current = this.pendingQueueMutations.get(mutation.candidateKey);
      if (current && sameValue(current, mutation)) {
        this.pendingQueueMutations.delete(mutation.candidateKey);
      }
    }
  }

  snapshot(
    collectorAccountId: string,
    stateVersionCounter: number,
    queueCoverage: "complete" | "partial" = "partial",
  ): BridgeStateEnvelope {
    const discovery: Partial<Record<HistoryScope, DiscoveryCheckpoint>> = {};
    for (const scope of ["active", "archived"] as const) {
      const checkpoint = this.discovery.get(scope);
      if (checkpoint) {
        const { candidateQueue: _candidateQueue, ...headerCheckpoint } =
          checkpoint;
        discovery[scope] = clone(headerCheckpoint);
      }
    }
    return {
      stateVersion: 1,
      collectorAccountId,
      stateVersionCounter,
      discovery,
      accountState: clone(this.accountState),
      queueCoverage,
    };
  }

  private recordCandidateMutations(
    scope: HistoryScope,
    previous: readonly {
      summary: { conversationId: string };
      missingUpdateTime: boolean;
    }[],
    next: readonly {
      summary: { conversationId: string };
      missingUpdateTime: boolean;
    }[],
  ): void {
    const previousById = new Map(
      previous.map((candidate) => [candidate.summary.conversationId, candidate]),
    );
    const nextById = new Map(
      next.map((candidate) => [candidate.summary.conversationId, candidate]),
    );
    for (const conversationId of previousById.keys()) {
      if (!nextById.has(conversationId)) {
        this.pendingQueueMutations.set(
          `candidate:${scope}:${conversationId}`,
          {
            candidateKey: `candidate:${scope}:${conversationId}`,
            operation: "remove",
            queueKind: "candidate",
            scope,
          },
        );
      }
    }
    for (const [conversationId, candidate] of nextById) {
      const prior = previousById.get(conversationId);
      if (prior && sameValue(prior, candidate)) {
        continue;
      }
      this.pendingQueueMutations.set(
        `candidate:${scope}:${conversationId}`,
        {
          candidateKey: `candidate:${scope}:${conversationId}`,
          operation: "replace",
          queueKind: "candidate",
          scope,
          payload: { candidate, scope },
        },
      );
    }
  }
}

function validateAccountState(state: HistoryAccountState): HistoryAccountState {
  if (
    (state.status !== "ready" && state.status !== "paused") ||
    state.pausedAt !== null && !isValidInstant(state.pausedAt) ||
    state.cooldownUntil !== null && !isValidInstant(state.cooldownUntil) ||
    state.lastError !== null && typeof state.lastError !== "string" ||
    (
      state.reason !== null &&
      state.reason !== "authentication" &&
      state.reason !== "cooldown"
    )
  ) {
    throw new Error("account state is invalid");
  }
  if (state.status === "ready" && state.reason !== null) {
    throw new Error("ready account state cannot have a pause reason");
  }
  if (state.status === "paused" && state.reason === null) {
    throw new Error("paused account state requires a reason");
  }
  return clone(state);
}

function isValidInstant(value: string): boolean {
  return Number.isFinite(new Date(value).getTime());
}

export function validateEnvelope(state: BridgeStateEnvelope): void {
  if (state.stateVersion !== 1 || typeof state.collectorAccountId !== "string") {
    throw new Error("collector state envelope is invalid");
  }
  if (!Number.isSafeInteger(state.stateVersionCounter) || state.stateVersionCounter < 0) {
    throw new Error("collector state version is invalid");
  }
  if (!isRecord(state.discovery) ||
    state.revisits !== undefined && !Array.isArray(state.revisits)
  ) {
    throw new Error("collector state contents are invalid");
  }
  for (const revisit of state.revisits ?? []) {
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

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}
