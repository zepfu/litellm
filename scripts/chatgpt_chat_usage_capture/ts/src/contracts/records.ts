/**
 * Stage-1 normalized records for the ChatGPT ordinary Chat usage collector.
 *
 * These interfaces are the single source of truth for the evidence contract
 * shared by the browser boundary, the ChatGPT route adapter, and Stage-2+
 * consumers (ledger, accounting, scheduler). They intentionally carry no
 * prompt/answer content and no credential material.
 */

/** Versioned adapter contract identifier; bump on any breaking route change. */
export const ADAPTER_VERSION = "chatgpt-chat-history-v1";

/** Ordinary Chat surface label; unknown is never auto-classified as Chat. */
export const SURFACE_CHAT = "chat";

export type Surface =
  | typeof SURFACE_CHAT
  | "work"
  | "codex"
  | "deep_research"
  | "agent_mode"
  | "voice"
  | "image_generation"
  | "unknown";

export type Coverage = "validated_page" | "partial" | "unrecognized";

export type CoverageLevel = "complete" | "partial" | "unknown";

export type AuthState =
  | "unconfigured"
  | "ready"
  | "auth_required"
  | "identity_mismatch"
  | "browser_unavailable"
  | "schema_unavailable"
  | "paused";

/** A page of adapted items with provenance and coverage signals. */
export interface AdaptedPage<T> {
  items: T[];
  /** Opaque cursor or numeric offset for the next page; null when terminal. */
  continuation: string | number | null;
  /** True only when the adapter proved the index/detail is exhausted. */
  exhausted: boolean;
  schemaVersion: string;
  coverage: Coverage;
  warnings: string[];
}

export interface ConversationSummary {
  conversationId: string;
  createdAt: string | null;
  updatedAt: string | null;
  isArchived: boolean;
  workspaceId: string | null;
  projectId: string | null;
  surface: Surface;
  origin: string | null;
  hasVersions: boolean | null;
  currentNode: string | null;
  coverage: Coverage;
}

export interface ConversationDetailProjection {
  conversationId: string;
  createdAt: string | null;
  updatedAt: string | null;
  currentNode: string | null;
  surface: Surface;
  messages: MessageRecord[];
  coverage: Coverage;
  warnings: string[];
}

export interface MessageRecord {
  conversationId: string;
  messageId: string;
  nodeId: string | null;
  parentId: string | null;
  children: string[];
  role: string | null;
  channel: string | null;
  createdAt: string | null;
  status: string | null;
  endTurn: boolean | null;
  requestedModelRaw: string | null;
  requestedModeRaw: string | null;
  requestedReasoningEffortRaw: string | null;
  recordedFinalModelRaw: string | null;
  generationId: string | null;
  requestId: string | null;
  surface: Surface;
  origin: string | null;
  /** Sanitized metadata projection only; no raw nested metadata. */
  metadata: Record<string, unknown>;
}

export interface AttemptRecord {
  attemptId: string;
  conversationId: string;
  identityBasis: string;
  timeBasis: string;
  attemptTime: string | null;
  earliestPossibleAt: string | null;
  latestPossibleAt: string | null;
  requestedModelRaw: string | null;
  requestedModeRaw: string | null;
  requestedReasoningEffortRaw: string | null;
  recordedFinalModelRaw: string | null;
  resolvedModelRaw: string | null;
  requestedFamily: string | null;
  recordedFinalFamily: string | null;
  resolvedFamily: string | null;
  mappingVersion: string;
  outcome: string;
  completedAnswer: boolean;
  generationStarted: boolean;
  surface: Surface;
  origin: string | null;
  aliases: Array<[string, string]>;
  evidenceMessageIds: string[];
  revision: number;
  warnings: string[];
}

/** Verified local account binding; never guessed from partial browser state. */
export interface AccountBinding {
  collectorAccountId: string;
  providerUserId: string | null;
  workspaceId: string | null;
  quotaOwnerId: string | null;
  surface: Surface;
  authState: AuthState;
  planPolicyId: string;
  enabled: boolean;
}

/** Per-account, per-adapter-version capability and coverage inventory. */
export interface CapabilityRecord {
  adapterVersion: string;
  indexScopes: string[];
  archiveBehavior: string;
  projectCoverage: CoverageLevel;
  modernDetail: string;
  pagination: string;
  legacySupport: string;
  branchVisibility: string;
  modelMetadata: string;
  quotaMetadata: string;
  warnings: string[];
}

/** Normalized, sanitized identity projection from a session/auth response. */
export interface IdentityRecord {
  providerUserId: string | null;
  workspaceId: string | null;
  quotaOwnerId: string | null;
  surface: Surface;
  authState: AuthState;
  identityErrors: string[];
}
