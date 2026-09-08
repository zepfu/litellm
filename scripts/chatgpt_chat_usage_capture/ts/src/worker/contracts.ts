import type {
  CapabilityRecord,
  ConversationDetailProjection,
  ConversationSummary,
  IdentityRecord,
  MessageRecord,
} from "../contracts/records.js";
import type {
  HistoryDiscoveryPageCommit,
  HistoryMetadataPage,
  HistoryPageCommit,
  HistoryReader,
} from "../contracts/history.js";
import type {
  BridgeCandidateMutation,
  BridgeCheckpointMutation,
  BridgeCoverageMutation,
  BridgeStateEnvelope,
} from "../history/bridge-checkpoints.js";
import type {
  IngestContext,
  LedgerScope,
  ModelMappingVersion,
  ReconstructedAttempt,
} from "../ledger/types.js";

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

export type WorkerControl = "startRun" | "prepareHistory" | "cancelRun";

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

export interface WorkerStartRun {
  protocolVersion: typeof WORKER_PROTOCOL_VERSION;
  control: "startRun";
  envelope: CollectorRunEnvelope;
  scheduleOptions: Record<string, unknown>;
  collectionRequest: Record<string, unknown>;
  mapping: ModelMappingVersion;
  reportRequest?: Record<string, unknown>;
  authenticationRecoveryRequested?: boolean;
  bounds: {
    maxFrameBytes: number;
    maxTotalBytes: number;
    maxRequests: number;
    remainingMs: number;
  };
}

export interface WorkerControlMessage {
  protocolVersion: typeof WORKER_PROTOCOL_VERSION;
  control: WorkerControl;
  requestId?: string;
  envelope?: CollectorRunEnvelope;
  runId?: string;
  collectorAccountId?: string;
  profileId?: string;
  bindingGeneration?: number;
  leaseFencingToken?: number;
  payload?: unknown;
  reason?: string;
}

export interface StateMutationEnvelope {
  expectedStateVersion: number;
  next: BridgeStateEnvelope;
}

export interface OpaquePageRequest {
  kind: "index" | "detail" | "messages";
  requiredCapability: "index" | "modern_detail" | "messages";
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
  indexItems?: ConversationSummary[];
  continuation: string | number | null;
  exhausted: boolean;
  paginationState: Parameters<HistoryReader["listConversations"]>[0] extends never
    ? never
    : "complete" | "continuation" | "contradictory" | "unknown" | "repeated_cursor" | "budget_exhausted";
  coverage: "validated_page" | "partial" | "unrecognized";
  warnings: string[];
  schemaVersion?: string;
  records: Array<ConversationSummary | MessageRecord>;
  source?: IngestContext;
  snapshotId?: string | null;
  nextCursor?: string | null;
  hasMore?: boolean;
  truncated?: boolean;
  coverageDetails?: Record<string, unknown>;
}

export interface CommitPagePayload {
  pageCommitId: string;
  expectedStateVersion: number;
  kind: "discovery" | "detail";
  source: IngestContext;
  discovery?: HistoryDiscoveryPageCommit;
  page?: HistoryPageCommit;
  attempts?: readonly ReconstructedAttempt[];
  checkpointMutations?: BridgeCheckpointMutation;
  candidateMutations?: readonly BridgeCandidateMutation[];
  coverageMutations?: readonly BridgeCoverageMutation[];
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
  loadState(
    request?: {
      kind?: "header" | "schedule" | "candidates";
      cursor?: string | null;
      limit?: number;
      expectedStateVersion?: number;
    },
  ): Promise<BridgeStateEnvelope | null>;
  compareAndSetState(
    envelope: StateMutationEnvelope,
  ): Promise<{ stateVersion: number }>;
  readHistory(request: OpaquePageRequest): Promise<TypedReadResult>;
  loadConversationMetadata(request: {
    conversationId: string;
    cursor?: string | null;
    limit?: number;
    snapshotId?: string | null;
  }): Promise<HistoryMetadataPage>;
  commitPage(payload: CommitPagePayload): Promise<{
    acknowledged: true;
    stateVersion: number;
    pageCommitId: string;
  }>;
  stateVersion?: number;
  loadReportSnapshot(request?: {
    cursor?: string | null;
    limit?: number;
    snapshotId?: string | null;
  }): Promise<Record<string, unknown>>;
  finishRun(payload: {
    expectedVersion: number;
    triggerId: string;
    outcome: "success" | "failure" | "authentication";
    summary?: Record<string, unknown>;
  }): Promise<{ finished: true; stateVersion: number }>;
  cancel(payload: {
    expectedVersion: number;
    triggerId: string;
    outcome?: string;
    summary?: Record<string, unknown>;
  }): Promise<{ cancelled: true; stateVersion: number }>;
};

export interface PreparedHistory {
  identity: IdentityRecord;
  capabilities: CapabilityRecord;
  capabilityManifest: Record<string, unknown>;
  scope: LedgerScope;
}
