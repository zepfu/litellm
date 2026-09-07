import type {
  AdaptedPage,
  CapabilityRecord,
  ConversationDetailProjection,
  ConversationSummary,
  IdentityRecord,
  MessageRecord,
  PaginationState,
} from "./records.js";

export const HISTORY_STATE_VERSION = 1;
export const DEFAULT_BACKFILL_DAYS = 14;
export const DEFAULT_OVERLAP_MS = 48 * 60 * 60 * 1000;
export const DEFAULT_INDEX_PAGE_SIZE = 100;
export const DEFAULT_MAX_INDEX_PAGES = 500;
export const DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION = 100;

export type HistoryScope = "active" | "archived";
export type HistoryCollectionMode =
  | "backfill"
  | "incremental"
  | "reconciliation";
export type CheckpointStatus =
  | "not_started"
  | "in_progress"
  | "complete"
  | "partial";
export type RevisitStatus = "pending" | "complete";
export type RevisitReason =
  | "incomplete_detail"
  | "repeated_cursor"
  | "page_budget"
  | "unknown_pagination"
  | "contradictory_pagination"
  | "detail_unavailable"
  | "missing_update_time";

export interface HistoryRange {
  /** Inclusive UTC instant. */
  start: string;
  /** Exclusive UTC instant. */
  end: string;
}

export interface DiscoveryCheckpoint {
  stateVersion: typeof HISTORY_STATE_VERSION;
  accountId: string;
  scope: HistoryScope;
  status: CheckpointStatus;
  mode: HistoryCollectionMode;
  range: HistoryRange;
  candidateCutoff: string;
  scanStartedAt: string | null;
  continuation: number | null;
  pagesFetched: number;
  pageBudget: number;
  lastCompleteDiscoveryStartedAt: string | null;
  lastPageAt: string | null;
  paginationState: PaginationState;
  warnings: string[];
  updatedAt: string;
}

export interface RevisitEntry {
  stateVersion: typeof HISTORY_STATE_VERSION;
  accountId: string;
  conversationId: string;
  scopes: HistoryScope[];
  status: RevisitStatus;
  reason: RevisitReason;
  firstSeenAt: string;
  lastSeenAt: string;
  attempts: number;
  nextEligibleAt: string;
  lastError: string | null;
  detailPagesFetched: number;
}

export interface HistoryCheckpointStore {
  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null;
  saveDiscovery(checkpoint: DiscoveryCheckpoint): void;
  listRevisits(): RevisitEntry[];
  upsertRevisit(entry: RevisitEntry): void;
  completeRevisit(accountId: string, conversationId: string): void;
}

export interface HistoryReader {
  readonly capabilities: CapabilityRecord;
  inspectSessionIdentity(): Promise<IdentityRecord>;
  listConversations(options: {
    archived: boolean;
    offset?: number;
    limit?: number;
    order?: string;
  }): Promise<AdaptedPage<ConversationSummary>>;
  fetchConversation(
    conversationId: string,
    options?: { allowLegacyFallback?: boolean },
  ): Promise<ConversationDetailProjection>;
  fetchMessages(
    conversationId: string,
    options?: {
      before?: string | null;
      numTurns?: number;
      conversationSurface?: ConversationSummary["surface"];
    },
  ): Promise<AdaptedPage<MessageRecord>>;
}

export interface HistoryCollectionRequest {
  mode: HistoryCollectionMode;
  range?: HistoryRange;
  now?: Date;
  overlapMs?: number;
  indexPageSize?: number;
  maxIndexPagesPerScope?: number;
  maxMessagePagesPerConversation?: number;
  legacyFallbackApproved?: boolean;
}

export interface HistoryCollectionOptions {
  accountId: string;
  store: HistoryCheckpointStore;
  clock?: { now(): Date };
  defaultBackfillDays?: number;
  overlapMs?: number;
  indexPageSize?: number;
  maxIndexPagesPerScope?: number;
  maxMessagePagesPerConversation?: number;
  legacyFallbackApproved?: boolean;
}

export interface ScopeCoverageResult {
  scope: HistoryScope;
  status: CheckpointStatus;
  coverage: "complete" | "partial" | "unknown";
  pagesFetched: number;
  candidates: number;
  continuation: number | null;
  paginationState: PaginationState;
  candidateCutoff: string;
  warnings: string[];
}

export interface HistoryCoverageResult {
  active: ScopeCoverageResult;
  archived: ScopeCoverageResult;
  projects: "validated_for_discovered_projects" | "unknown";
  branches: "version_metadata_observed" | "active_branch_only" | "unknown";
  overall: "complete" | "partial" | "unknown";
  gaps: string[];
}

export interface AcquiredConversation {
  summary: ConversationSummary;
  scopes: HistoryScope[];
  detail: ConversationDetailProjection | null;
  messages: MessageRecord[];
  coverage: "complete" | "partial" | "unknown";
  revisit: RevisitEntry | null;
  warnings: string[];
}

export interface HistoryCollectionResult {
  accountId: string;
  mode: HistoryCollectionMode;
  range: HistoryRange;
  scanStartedAt: string;
  status: "complete" | "partial" | "blocked";
  identity: IdentityRecord;
  scopes: ScopeCoverageResult[];
  conversations: AcquiredConversation[];
  revisits: RevisitEntry[];
  coverage: HistoryCoverageResult;
  pagesFetched: number;
  detailPagesFetched: number;
  warnings: string[];
}
