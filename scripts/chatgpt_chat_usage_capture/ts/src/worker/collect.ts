import type {
  AcquiredConversation,
  HistoryCollectionRequest,
  HistoryCollectionResult,
  HistoryReader,
} from "../contracts/history.js";
import { HistoryCollector } from "../history/collector.js";
import { BridgeCheckpointStore } from "../history/bridge-checkpoints.js";
import { resolveRequestedRange } from "../history/range.js";
import type {
  CollectorRunEnvelope,
  TypedReadResult,
  WorkerBridge,
  WorkerBounds,
} from "./contracts.js";
import { DEFAULT_WORKER_BOUNDS } from "./contracts.js";

export interface CollectorRunContext {
  envelope: CollectorRunEnvelope;
  bridge: WorkerBridge;
  reader: HistoryReader;
  mapping: Parameters<HistoryCollector["collect"]>[0] extends never
    ? never
    : ConstructorParameters<typeof HistoryCollector>[1] extends never
      ? never
      : import("../ledger/types.js").ModelMappingVersion;
  scope: import("../ledger/types.js").LedgerScope;
  request: HistoryCollectionRequest;
  bounds?: Partial<WorkerBounds>;
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
  const store = new BridgeCheckpointStore();
  const initial = await context.bridge.loadState();
  store.hydrate(initial);
  const deadline = bounds.deadlineAt;
  let requestCount = 0;
  let bytesRead = 0;
  let cancelled = false;

  const budgetedReader: HistoryReader = {
    capabilities: context.reader.capabilities,
    inspectSessionIdentity: async () => {
      assertBounds(++requestCount, bytesRead, deadline, cancelled);
      return context.reader.inspectSessionIdentity();
    },
    listConversations: async (
      options: Parameters<HistoryReader["listConversations"]>[0],
    ) => {
      assertBounds(++requestCount, bytesRead, deadline, cancelled);
      const result = await context.reader.listConversations(options);
      bytesRead += result.items.length;
      return result;
    },
    fetchConversation: async (
      conversationId: string,
      options?: Parameters<HistoryReader["fetchConversation"]>[1],
    ) => {
      assertBounds(++requestCount, bytesRead, deadline, cancelled);
      const result = await context.reader.fetchConversation(conversationId, options);
      bytesRead += result.messages.length;
      return result;
    },
    fetchMessages: async (
      conversationId: string,
      options?: Parameters<HistoryReader["fetchMessages"]>[1],
    ) => {
      assertBounds(++requestCount, bytesRead, deadline, cancelled);
      const result = await context.reader.fetchMessages(conversationId, options);
      bytesRead += result.items.length;
      return result;
    },
  };

  const collector = new HistoryCollector(budgetedReader, {
    accountId: context.envelope.collectorAccountId,
    store,
    ...context.request,
  });
  try {
    const result = await collector.collect(context.request);
    const nextState = store.snapshot(
      context.envelope.collectorAccountId,
      (initial?.stateVersionCounter ?? 0) + 1,
    );
    await context.bridge.compareAndSetState({
      expectedStateVersion: initial?.stateVersionCounter ?? 0,
      next: nextState,
    });
    return {
      result,
      requestCount,
      bytesRead,
      coverageIncomplete: result.status !== "complete",
      warnings: result.warnings,
    };
  } catch (error) {
    const coverageIncomplete = true;
    if (cancelled) {
      throw new BoundedWorkerError("cancelled", false, coverageIncomplete);
    }
    if (requestCount > bounds.maxRequests) {
      throw new BoundedWorkerError("bounds_exceeded", true, coverageIncomplete);
    }
    throw error;
  }
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
  return {
    continuation: result.continuation,
    exhausted: result.exhausted,
    paginationState: result.paginationState,
    coverage: result.coverage,
    warnings: [...result.warnings],
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
  deadlineAt: number,
  cancelled: boolean,
): void {
  if (cancelled) {
    throw new BoundedWorkerError("cancelled", false, true);
  }
  if (Date.now() >= deadlineAt) {
    throw new BoundedWorkerError("bounds_exceeded", true, true);
  }
  if (requestCount > DEFAULT_WORKER_BOUNDS.maxRequests || bytesRead > Number.MAX_SAFE_INTEGER) {
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
