import type {
  ConversationDetailProjection,
  ConversationSummary,
  IdentityRecord,
  MessageRecord,
} from "../contracts/records.js";
import type { HistoryReader } from "../contracts/history.js";
import type { BridgePageMutation, BridgeStateEnvelope } from "../history/bridge-checkpoints.js";

export const WORKER_PROTOCOL_VERSION = 1;
export const MAX_REQUEST_ID_LENGTH = 128;

export type WorkerOperation =
  | "loadState"
  | "compareAndSetState"
  | "readHistory"
  | "loadConversationMetadata"
  | "commitPage"
  | "loadReportSnapshot"
  | "finishRun"
  | "cancel";

export interface WorkerRequest {
  protocolVersion: typeof WORKER_PROTOCOL_VERSION;
  requestId: string;
  runId: string;
  collectorAccountId: string;
  profileId: string;
  bindingGeneration: number;
  leaseFencingToken: number;
  operation: WorkerOperation;
  payload?: unknown;
}

export interface WorkerErrorValue {
  code: WorkerErrorCode;
  retryable: boolean;
  coverageIncomplete: boolean;
}

export type WorkerErrorCode =
  | "protocol_invalid"
  | "fence_invalid"
  | "state_conflict"
  | "history_contract_unavailable"
  | "history_reader_failed"
  | "bounds_exceeded"
  | "conversation_not_found"
  | "operation_unsupported"
  | "cancelled";

export type WorkerResponse =
  | {
      protocolVersion: typeof WORKER_PROTOCOL_VERSION;
      requestId: string;
      ok: true;
      result?: unknown;
    }
  | {
      protocolVersion: typeof WORKER_PROTOCOL_VERSION;
      requestId: string;
      ok: false;
      error: WorkerErrorValue;
    };

export interface CollectorRunEnvelope {
  runId: string;
  collectorAccountId: string;
  profileId: string;
  bindingGeneration: number;
  leaseFencingToken: number;
}

export interface StateMutationEnvelope {
  expectedStateVersion: number;
  next: BridgeStateEnvelope;
}

export interface OpaquePageRequest {
  kind: "index" | "detail" | "messages";
  archived?: boolean;
  conversationId?: string;
  offset?: number;
  before?: string | null;
  conversationSurface?: ConversationSummary["surface"];
  allowLegacyFallback?: boolean;
}

export interface TypedReadResult {
  summary?: ConversationSummary;
  detail?: ConversationDetailProjection;
  messages?: MessageRecord[];
  continuation: string | number | null;
  exhausted: boolean;
  paginationState: Parameters<HistoryReader["listConversations"]>[0] extends never
    ? never
    : "complete" | "continuation" | "contradictory" | "unknown" | "repeated_cursor" | "budget_exhausted";
  coverage: "validated_page" | "partial" | "unrecognized";
  warnings: string[];
}

export interface CommitPagePayload {
  pageCommitId: string;
  mutations: BridgePageMutation;
}

export interface WorkerBounds {
  maxFrameBytes: number;
  maxTotalBytes: number;
  maxRequests: number;
  deadlineAt: number;
}

export const DEFAULT_WORKER_BOUNDS: Readonly<WorkerBounds> = {
  maxFrameBytes: 1024 * 1024,
  maxTotalBytes: 16 * 1024 * 1024,
  maxRequests: 256,
  deadlineAt: Number.POSITIVE_INFINITY,
};

export type WorkerBridge = {
  loadState(): Promise<BridgeStateEnvelope | null>;
  compareAndSetState(
    envelope: StateMutationEnvelope,
  ): Promise<{ stateVersion: number }>;
  readHistory(request: OpaquePageRequest): Promise<TypedReadResult>;
  loadConversationMetadata(conversationId: string): Promise<{
    summary: ConversationSummary;
  }>;
  commitPage(payload: CommitPagePayload): Promise<{ acknowledged: true }>;
  loadReportSnapshot(): Promise<Record<string, unknown>>;
  finishRun(): Promise<{ finished: true }>;
};
