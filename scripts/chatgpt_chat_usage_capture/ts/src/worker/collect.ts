import { createHash } from "node:crypto";

import type {
  AcquiredConversation,
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryMetadataPage,
  HistoryReader,
} from "../contracts/history.js";
import type {
  ConversationSummary,
  MessageRecord,
} from "../contracts/records.js";
import type {
  IngestContext,
  LedgerScope,
  ModelMappingVersion,
  ReconstructedAttempt,
} from "../ledger/types.js";
import { reconstructAttempts } from "../normalize/reconstruct.js";
import type {
  BridgeCandidateMutation,
  BridgeCoverageMutation,
  BridgePageMutation,
} from "../history/bridge-checkpoints.js";
import { BridgeCheckpointStore } from "../history/bridge-checkpoints.js";
import { HistoryCollector } from "../history/collector.js";
import { resolveRequestedRange } from "../history/range.js";
import type {
  CollectorRunEnvelope,
  TypedReadResult,
  WorkerBridge,
  WorkerBounds,
} from "./contracts.js";
import { DEFAULT_WORKER_BOUNDS } from "./contracts.js";

const MAX_METADATA_PAGES = 100;

export interface CollectorRunContext {
  envelope: CollectorRunEnvelope;
  bridge: WorkerBridge;
  reader: HistoryReader;
  scope: LedgerScope;
  mapping: ModelMappingVersion;
  request: HistoryCollectionRequest;
  bounds?: Partial<WorkerBounds>;
  signal?: AbortSignal;
}

export async function runBoundedCollector(
  context: CollectorRunContext,
): Promise<{
  result: HistoryCollectionResult;
  requestCount: number;
  bytesRead: number;
  coverageIncomplete: boolean;
  warnings: string[];
}> {
  const bounds = validateBounds({ ...DEFAULT_WORKER_BOUNDS, ...context.bounds });
  const signal = context.signal ?? new AbortController().signal;
  let requestCount = 0;
  let bytesRead = 0;
  let bridgeStateVersion = 0;
  const account = context.envelope.collectorAccountId;
  const store = new BridgeCheckpointStore();

  const bridgeCall = async <T>(operation: () => Promise<T>): Promise<T> => {
    assertBounds(++requestCount, bytesRead, bounds, signal);
    const result = await withAbort(operation(), signal);
    bytesRead += serializedBytes(result);
    assertBounds(requestCount, bytesRead, bounds, signal);
    return result;
  };

  const header = await bridgeCall(() =>
    context.bridge.loadState({ kind: "header" }),
  );
  let initial = header;
  if (header !== null) {
    initial = await bridgeCall(() =>
      context.bridge.loadState({
        kind: "candidates",
        expectedStateVersion: header.stateVersionCounter,
      }),
    );
  }
  if (initial !== null && initial.queueCoverage !== "complete") {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
  store.hydrate(initial);
  bridgeStateVersion = initial?.stateVersionCounter ?? 0;

  const budgetedReader: HistoryReader = {
    capabilities: context.reader.capabilities,
    inspectSessionIdentity: async () =>
      bridgeCall(() => context.reader.inspectSessionIdentity()),
    listConversations: async (
      options: Parameters<HistoryReader["listConversations"]>[0],
    ) => bridgeCall(() => context.reader.listConversations(options)),
    fetchConversation: async (
      conversationId: string,
      options?: Parameters<HistoryReader["fetchConversation"]>[1],
    ) => bridgeCall(() => context.reader.fetchConversation(conversationId, options)),
    fetchMessages: async (
      conversationId: string,
      options?: Parameters<HistoryReader["fetchMessages"]>[1],
    ) => bridgeCall(() => context.reader.fetchMessages(conversationId, options)),
  };

  const commitMutation = async (
    mutation: BridgePageMutation,
    queueMutations: readonly BridgeCandidateMutation[],
  ): Promise<void> => {
    const expectedStateVersion = bridgeStateVersion;
    const source = mutation.source;
    if (!source) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    const kind = mutation.discovery
      ? "discovery"
      : mutation.page
        ? "detail"
        : "discovery";
    if (
      (kind === "discovery" && mutation.page !== undefined) ||
      (kind === "detail" && mutation.discovery !== undefined)
    ) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    const acknowledgment = await bridgeCall(() =>
      context.bridge.commitPage({
        pageCommitId: mutation.pageCommitId,
        expectedStateVersion,
        kind,
        source,
        ...(mutation.discovery ? { discovery: mutation.discovery } : {}),
        ...(mutation.page ? { page: mutation.page } : {}),
        ...(mutation.attempts ? { attempts: mutation.attempts } : {}),
        checkpointMutations: {
          kind: "history",
          value: store.snapshot(account, expectedStateVersion, "complete"),
        },
        candidateMutations: queueMutations,
        ...(mutation.coverageMutations
          ? { coverageMutations: mutation.coverageMutations }
          : {}),
      }),
    );
    if (
      acknowledgment.acknowledged !== true ||
      !Number.isSafeInteger(acknowledgment.stateVersion) ||
      acknowledgment.stateVersion <= expectedStateVersion ||
      acknowledgment.pageCommitId !== mutation.pageCommitId
    ) {
      throw new BoundedWorkerError("protocol_invalid", false, true);
    }
    bridgeStateVersion = acknowledgment.stateVersion;
    store.acknowledgeQueueMutations(queueMutations);
    assertBounds(requestCount, bytesRead, bounds, signal);
  };

  const collector = new HistoryCollector(budgetedReader, {
    accountId: account,
    store,
    ...context.request,
    scope: context.scope,
    mapping: context.mapping,
    loadConversationMetadata: async (
      conversationId,
    ): Promise<HistoryMetadataPage | null> => {
      let cursor: string | null = null;
      let snapshotId: string | null = null;
      let first: HistoryMetadataPage | null = null;
      const messages: MessageRecord[] = [];
      const attempts: ReconstructedAttempt[] = [];
      const items: Array<Record<string, unknown>> = [];
      const warnings: string[] = [];
      const seenCursors = new Set<string>();
      let coverage: HistoryMetadataPage["coverage"] = "complete";
      let truncated = false;
      let source: HistoryMetadataPage["source"];
      let schemaVersion: string | undefined;
      let coverageDetails: Record<string, unknown> | undefined;
      let expectedSnapshotId: string | null | undefined;
      let sourceInitialized = false;
      let schemaInitialized = false;
      let snapshotInitialized = false;

      for (let page = 0; page < MAX_METADATA_PAGES; page += 1) {
        const current = await bridgeCall(() =>
          context.bridge.loadConversationMetadata({
            conversationId,
            limit: 256,
            cursor,
            snapshotId,
          }),
        );
        first ??= current;
        if (!snapshotInitialized) {
          expectedSnapshotId = current.snapshotId;
          snapshotInitialized = true;
        } else if (current.snapshotId !== expectedSnapshotId) {
          warnings.push("retained_metadata_snapshot_changed");
          coverage = "partial";
          truncated = true;
          break;
        }
        if (current.snapshotId !== undefined) {
          snapshotId = current.snapshotId;
        }
        items.push(...(current.items ?? []));
        messages.push(...(current.messages ?? []));
        attempts.push(...(current.attempts ?? []));
        warnings.push(...(current.warnings ?? []));
        if (!sourceInitialized) {
          source = current.source;
          sourceInitialized = true;
        } else if (!sameValue(source, current.source)) {
          warnings.push("retained_metadata_source_changed");
          coverage = "partial";
        }
        if (!schemaInitialized) {
          schemaVersion = current.schemaVersion;
          schemaInitialized = true;
        } else if (schemaVersion !== current.schemaVersion) {
          warnings.push("retained_metadata_schema_changed");
          coverage = "partial";
        }
        coverageDetails ??= current.coverageDetails;
        if (coverageDetails && current.coverageDetails) {
          coverageDetails = {
            ...coverageDetails,
            ...current.coverageDetails,
          };
        }
        if (current.coverage === "unknown") {
          coverage = "unknown";
        } else if (current.coverage !== "complete" && coverage !== "unknown") {
          coverage = "partial";
        }
        truncated ||= current.truncated === true;
        if (current.hasMore === false) {
          if (current.nextCursor !== undefined && current.nextCursor !== null) {
            warnings.push("retained_metadata_contradictory_continuation");
            coverage = "partial";
            truncated = true;
          }
          break;
        }
        if (current.hasMore !== true) {
          warnings.push("retained_metadata_missing_continuation_state");
          coverage = "partial";
          truncated = true;
          break;
        }

        const nextCursor = current.nextCursor ?? null;
        const nextSnapshot: string | null =
          current.snapshotId ?? snapshotId;
        if (
          nextCursor === null ||
          nextSnapshot === null ||
          seenCursors.has(nextCursor)
        ) {
          warnings.push("retained_metadata_invalid_continuation");
          coverage = "partial";
          truncated = true;
          break;
        }
        seenCursors.add(nextCursor);
        cursor = nextCursor;
        snapshotId = nextSnapshot;
        if (page === MAX_METADATA_PAGES - 1) {
          warnings.push("retained_metadata_page_budget_exhausted");
          coverage = "partial";
          truncated = true;
        }
      }

      if (!first) {
        return null;
      }
      return {
        ...first,
        ...(items.length > 0 ? { items } : {}),
        ...(messages.length > 0 ? { messages } : {}),
        ...(attempts.length > 0 ? { attempts } : {}),
        ...(source ? { source } : {}),
        ...(schemaVersion ? { schemaVersion } : {}),
        ...(coverageDetails ? { coverageDetails } : {}),
        ...(warnings.length > 0
          ? { warnings: [...new Set(warnings)] }
          : {}),
        coverage,
        truncated,
        snapshotId,
        nextCursor: cursor,
        hasMore: truncated,
      };
    },
    onDiscoveryPageCommit: async (page) => {
      const source = sourceFor(
        context.envelope.runId,
        `discovery:${page.checkpoint.scope}:${page.checkpoint.pagesFetched}`,
        page.scanStartedAt,
      );
      const mutation = {
        pageCommitId: stablePageCommitId("discovery", context.envelope.runId, page),
        conversationId: `${page.checkpoint.accountId}:discovery:${page.checkpoint.scope}`,
        scopes: [page.checkpoint.scope],
        source,
        discovery: page,
      } satisfies BridgePageMutation;
      await commitMutation(mutation, store.pendingQueueMutationsSnapshot());
    },
    onPageCommit: async (page) => {
      store.acknowledgeCandidates(page.summary.conversationId, page.scopes);
      const source = sourceFor(
        context.envelope.runId,
        `${page.summary.conversationId}:${page.pageKind}:${page.pageNumber}`,
        page.scanStartedAt,
      );
      const attempts = mergeAttempts(
        reconstructAttempts(page.messages, {
          scope: context.scope,
          conversationId: page.summary.conversationId,
          mapping: context.mapping,
        }),
        page.retainedAttempts ?? [],
      );
      const mutation = {
        pageCommitId: stablePageCommitId("page", context.envelope.runId, page),
        conversationId: page.summary.conversationId,
        scopes: page.scopes,
        source,
        page,
        attempts,
        coverageMutations: [coverageMutationFor(page, source)],
      } satisfies BridgePageMutation;
      await commitMutation(mutation, store.pendingQueueMutationsSnapshot());
    },
    onAccountStateCommit: async () => {
      const expectedStateVersion = bridgeStateVersion;
      const acknowledgment = await bridgeCall(() =>
        context.bridge.compareAndSetState({
          expectedStateVersion,
          next: store.snapshot(account, expectedStateVersion, "complete"),
        }),
      );
      if (
        !Number.isSafeInteger(acknowledgment.stateVersion) ||
        acknowledgment.stateVersion <= expectedStateVersion
      ) {
        throw new BoundedWorkerError("protocol_invalid", false, true);
      }
      bridgeStateVersion = acknowledgment.stateVersion;
    },
  });

  try {
    const result = await withAbort(collector.collect(context.request), signal);
    return {
      result,
      requestCount,
      bytesRead,
      coverageIncomplete: result.status !== "complete",
      warnings: result.warnings,
    };
  } catch (error) {
    if (signal.aborted) {
      throw new BoundedWorkerError("cancelled", false, true);
    }
    if (error instanceof BoundedWorkerError) {
      throw error;
    }
    throw error;
  }
}

function sourceFor(
  runId: string,
  sourceId: string,
  observedAt: string,
): {
  runId: string;
  observedAt: string;
  sourceKind: string;
  sourceId: string;
  schemaVersion: string;
  provenance: Record<string, unknown>;
} {
  return {
    runId,
    observedAt,
    sourceKind: "chatgpt_history",
    sourceId,
    schemaVersion: "chatgpt-chat-history-v1",
    provenance: { collector: "typescript_worker" },
  };
}

function coverageMutationFor(
  page: {
    summary: { conversationId: string };
    pageKind: "detail" | "messages";
    pageNumber: number;
    coverage: "complete" | "partial" | "unknown";
    warnings: string[];
  },
  source: {
    sourceKind: string;
    sourceId: string;
    observedAt: string;
  },
): BridgeCoverageMutation {
  return {
    sourceKind: source.sourceKind,
    sourceId: source.sourceId,
    reason:
      page.coverage === "complete"
        ? `history_${page.pageKind}_complete`
        : `history_${page.pageKind}_partial`,
    state: page.coverage === "complete" ? "resolved" : "open",
    seenAt: source.observedAt,
    details: {
      conversationId: page.summary.conversationId,
      pageNumber: page.pageNumber,
      warnings: page.warnings,
    },
  };
}

function stablePageCommitId(
  kind: "discovery" | "page",
  runId: string,
  payload: unknown,
): string {
  return createHash("sha256")
    .update(`${kind}\0${runId}\0${JSON.stringify(canonicalize(payload))}`)
    .digest("hex");
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, canonicalize(item)]),
  );
}

function sameValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonicalize(left)) === JSON.stringify(canonicalize(right));
}

function mergeAttempts(
  current: readonly ReconstructedAttempt[],
  retained: readonly ReconstructedAttempt[],
): ReconstructedAttempt[] {
  const byId = new Map<string, ReconstructedAttempt>();
  for (const attempt of retained) {
    byId.set(attempt.attemptId, attempt);
  }
  for (const attempt of current) {
    byId.set(attempt.attemptId, attempt);
  }
  return [...byId.values()];
}

export class BoundedWorkerError extends Error {
  constructor(
    readonly code:
      | "protocol_invalid"
      | "fence_invalid"
      | "state_conflict"
      | "history_contract_unavailable"
      | "history_reader_failed"
      | "bounds_exceeded"
      | "conversation_not_found"
      | "operation_unsupported"
      | "cancelled",
    readonly retryable: boolean,
    readonly coverageIncomplete: boolean,
  ) {
    super(`bounded worker error: ${code}`);
    this.name = "BoundedWorkerError";
  }
}

export function typedReadResult(
  result:
    | Awaited<ReturnType<HistoryReader["listConversations"]>>
    | Awaited<ReturnType<HistoryReader["fetchMessages"]>>,
): TypedReadResult {
  const records = result.items as Array<ConversationSummary | MessageRecord>;
  const indexItems = records.filter(isConversationSummary);
  const messages = records.filter(isMessageRecord);
  return {
    ...(indexItems.length > 0 ? { indexItems } : {}),
    ...(messages.length > 0 ? { messages } : {}),
    records,
    continuation: result.continuation,
    exhausted: result.exhausted,
    paginationState: result.paginationState,
    coverage: result.coverage,
    warnings: [...result.warnings],
    schemaVersion: result.schemaVersion,
  };
}

export function resolvedRequestedRange(
  request: HistoryCollectionRequest,
  now: Date,
  defaultBackfillDays: number,
): ReturnType<typeof resolveRequestedRange> {
  const rangeOptions: Parameters<typeof resolveRequestedRange>[0] = {
    mode: request.mode,
    now,
    defaultBackfillDays,
  };
  if (request.range !== undefined) {
    rangeOptions.range = request.range;
  }
  return resolveRequestedRange(rangeOptions);
}

export function collectAcquisitionWarnings(
  acquisitions: readonly AcquiredConversation[],
): string[] {
  return [...new Set(acquisitions.flatMap((item) => item.warnings))];
}

function assertBounds(
  requestCount: number,
  bytesRead: number,
  bounds: WorkerBounds,
  signal: AbortSignal,
): void {
  if (signal.aborted) {
    throw new BoundedWorkerError("cancelled", false, true);
  }
  if (
    requestCount > bounds.maxRequests ||
    bytesRead > bounds.maxTotalBytes ||
    monotonicNow() >= bounds.deadlineAt
  ) {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
}

function validateBounds(bounds: WorkerBounds): WorkerBounds {
  if (
    !Number.isSafeInteger(bounds.maxRequests) ||
    bounds.maxRequests <= 0 ||
    bounds.maxRequests > DEFAULT_WORKER_BOUNDS.maxRequests
  ) {
    throw new Error("worker request bound is invalid");
  }
  if (
    !Number.isSafeInteger(bounds.maxFrameBytes) ||
    bounds.maxFrameBytes <= 0 ||
    bounds.maxFrameBytes > DEFAULT_WORKER_BOUNDS.maxFrameBytes
  ) {
    throw new Error("worker frame bound is invalid");
  }
  if (
    !Number.isSafeInteger(bounds.maxTotalBytes) ||
    bounds.maxTotalBytes <= 0 ||
    bounds.maxTotalBytes > DEFAULT_WORKER_BOUNDS.maxTotalBytes
  ) {
    throw new Error("worker total byte bound is invalid");
  }
  if (!Number.isFinite(bounds.deadlineAt) && bounds.deadlineAt !== Number.POSITIVE_INFINITY) {
    throw new Error("worker deadline is invalid");
  }
  return bounds;
}

function serializedBytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value ?? null), "utf8");
}

function monotonicNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

async function withAbort<T>(
  promise: Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  if (signal.aborted) {
    throw new BoundedWorkerError("cancelled", false, true);
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => {
      reject(new BoundedWorkerError("cancelled", false, true));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

function isConversationSummary(
  value: ConversationSummary | MessageRecord,
): value is ConversationSummary {
  return "conversationId" in value && "isArchived" in value;
}

function isMessageRecord(
  value: ConversationSummary | MessageRecord,
): value is MessageRecord {
  return "messageId" in value && "role" in value;
}
