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
export const DEFAULT_MAX_OLDER_HISTORY_AUDIT_PAGES = 1;
export const OUTSTANDING_GENERATION_TIMEOUT_MS = 24 * 60 * 60 * 1000;
export const OUTSTANDING_GENERATION_REVISIT_BASE_DELAY_MS = 15 * 60 * 1000;
export const OUTSTANDING_GENERATION_REVISIT_MAX_DELAY_MS = 6 * 60 * 60 * 1000;

export type HistoryScope = "active" | "archived";
export type HistoryCollectionMode =
  | "backfill"
  | "incremental"
  | "reconciliation";
export type HistoryAccountPauseReason = "authentication" | "cooldown";
export interface HistoryAccountState {
  status: "ready" | "paused";
  reason: HistoryAccountPauseReason | null;
  pausedAt: string | null;
  cooldownUntil: string | null;
  lastError: string | null;
}
export type CheckpointStatus =
  | "not_started"
  | "in_progress"
  | "complete"
  | "partial";
export type RevisitStatus = "pending" | "complete";
export type RevisitReason =
  | "incomplete_detail"
  | "partial_detail"
  | "unrecognized_detail"
  | "repeated_cursor"
  | "bad_continuation"
  | "page_budget"
  | "unknown_pagination"
  | "contradictory_pagination"
  | "detail_unavailable"
  | "missing_update_time"
  | "nonterminal_generation";

export type OlderHistoryAuditStatus =
  | "disabled"
  | "in_progress"
  | "partial"
  | "complete";

export type GenerationCompletionState = "nonterminal" | "unknown";

export interface RevisitPageIssue {
  reason: RevisitReason;
  warnings: string[];
}

export interface OutstandingGenerationState {
  state: GenerationCompletionState;
  since: string;
  timedOut: boolean;
}

export interface OlderHistoryAuditRequest {
  enabled: boolean;
  maxPages?: number;
}

export interface OlderHistoryAuditState {
  enabled: boolean;
  status: OlderHistoryAuditStatus;
  continuation: number | null;
  pagesFetched: number;
  conversationsAudited: number;
  lastStartedAt: string | null;
  lastPageAt: string | null;
  lastCompletedAt: string | null;
}

export interface OlderHistoryAuditCoverage extends OlderHistoryAuditState {}

export interface HistoryRange {
  /** Inclusive UTC instant. */
  start: string;
  /** Exclusive UTC instant. */
  end: string;
}

export interface DiscoveryCandidate {
  summary: ConversationSummary;
  missingUpdateTime: boolean;
}

export interface DiscoveryCheckpoint {
  stateVersion: typeof HISTORY_STATE_VERSION;
  accountId: string;
  scope: HistoryScope;
  status: CheckpointStatus;
  /** Frozen identity for the discovery acquisition being resumed. */
  mode: HistoryCollectionMode;
  range: HistoryRange;
  /** Lower-bound discovery cutoff frozen with mode and range. */
  candidateCutoff: string;
  scanStartedAt: string | null;
  continuation: number | null;
  pagesFetched: number;
  pageBudget: number;
  /** Clean implicit-refresh watermark; historical scans never advance it. */
  lastCompleteDiscoveryStartedAt: string | null;
  lastPageAt: string | null;
  paginationState: PaginationState;
  warnings: string[];
  updatedAt: string;
  /** SHA-256 fingerprint of the frozen offset-zero index page. */
  headFingerprint?: string | null;
  /** Candidates discovered by this scope and not yet durably acknowledged. */
  candidateQueue?: DiscoveryCandidate[];
  olderHistoryAudit?: OlderHistoryAuditState;
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
  continuation: string | null;
  /** Revision of the conversation when the saved continuation was issued. */
  continuationRevision: string | null;
  /** A page-shape/pagination issue that remains unresolved independently of budget. */
  malformedPage: RevisitPageIssue | null;
  /** Nonterminal or unproven generation completion remains unknown. */
  outstandingGeneration: OutstandingGenerationState | null;
}

export interface HistoryCheckpointStore {
  loadAccountState(): HistoryAccountState;
  saveAccountState(state: HistoryAccountState): void;
  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null;
  saveDiscovery(checkpoint: DiscoveryCheckpoint): void;
  acknowledgeCandidates(
    conversationId: string,
    scopes: readonly HistoryScope[],
  ): void;
  listRevisits(): RevisitEntry[];
  upsertRevisit(entry: RevisitEntry): void;
  completeRevisit(accountId: string, conversationId: string): void;
}

export interface HistoryDiscoveryPageCommit {
  checkpoint: DiscoveryCheckpoint;
  identity: IdentityRecord;
  scanStartedAt: string;
}

export interface HistoryPageCommit {
  accountId: string;
  mode: HistoryCollectionMode;
  scanStartedAt: string;
  identity: IdentityRecord;
  summary: ConversationSummary;
  scopes: HistoryScope[];
  detail: ConversationDetailProjection | null;
  messages: MessageRecord[];
  coverage: AcquiredConversation["coverage"];
  warnings: string[];
  pageKind: "detail" | "messages";
  pageNumber: number;
  nextContinuation: string | null;
  revisit: RevisitEntry | null;
  accountState: HistoryAccountState;
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
  olderHistoryAudit?: OlderHistoryAuditRequest;
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
  onDiscoveryPageCommit?: (
    page: HistoryDiscoveryPageCommit,
  ) => Promise<void> | void;
  onPageCommit?: (
    page: HistoryPageCommit,
  ) => Promise<void> | void;
  onAccountStateCommit?: (
    state: HistoryAccountState,
  ) => Promise<void> | void;
  loadConversationMetadata?: (
    conversationId: string,
  ) => Promise<{ summary: ConversationSummary; messages?: MessageRecord[] } | null>;
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
  olderHistoryAudit: OlderHistoryAuditCoverage;
}

export interface HistoryCoverageResult {
  active: ScopeCoverageResult;
  archived: ScopeCoverageResult;
  projects: "validated_for_discovered_projects" | "unknown";
  branches: "version_metadata_observed" | "active_branch_only" | "unknown";
  olderHistoryAudit: {
    enabled: boolean;
    status: OlderHistoryAuditStatus;
    active: OlderHistoryAuditCoverage;
    archived: OlderHistoryAuditCoverage;
  };
  overall: "complete" | "partial" | "unknown";
  gaps: string[];
}

export interface AcquiredConversation {
  summary: ConversationSummary;
  scopes: HistoryScope[];
  detail: ConversationDetailProjection | null;
  /** Evidence is bounded by the collection range's exclusive end. */
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
  accountState: HistoryAccountState;
  identity: IdentityRecord;
  scopes: ScopeCoverageResult[];
  conversations: AcquiredConversation[];
  revisits: RevisitEntry[];
  coverage: HistoryCoverageResult;
  pagesFetched: number;
  detailPagesFetched: number;
  warnings: string[];
}
