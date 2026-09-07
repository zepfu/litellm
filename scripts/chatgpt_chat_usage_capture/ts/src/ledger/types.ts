import type {
  AttemptRecord,
  MessageRecord,
  Surface,
} from "../contracts/records.js";
import type { SanitizedProvenance } from "./identity.js";

export interface LedgerScope {
  collectorAccountId: string;
  provider: string;
  providerUserId: string | null;
  workspaceId: string | null;
  quotaOwnerId: string | null;
  surface: Surface;
}

export interface MappingRule {
  slug: string;
  mode?: string | null;
  reasoningEffort?: string | null;
  family: string;
  collectorAccountId?: string | null;
  reviewed?: boolean;
  source?: string;
}

export type MappingChangeKind = "prospective" | "historical_correction";

export interface ModelMappingVersion {
  version: string;
  canonicalFamilies: string[];
  rules: MappingRule[];
  reviewStatus: "draft" | "approved" | "retired";
  source: string;
  createdAt: string;
  reviewedAt?: string | null;
  reviewedBy?: string | null;
  changeKind?: MappingChangeKind;
  validFrom?: string | null;
  validUntil?: string | null;
  publishedAt?: string | null;
  supersedesVersion?: string | null;
  correctionOfVersion?: string | null;
  provenance?: Record<string, unknown>;
  warnings?: string[];
}

export interface MappingEvidence {
  slug: string | null;
  mode: string | null;
  reasoningEffort: string | null;
}

export interface MappingResolution {
  family: string | null;
  rule: MappingRule | null;
  warnings: string[];
  applied: boolean;
}

export interface StoredMessage {
  scope: LedgerScope;
  record: MessageRecord;
  revision: number;
  revisionFingerprint: string;
}

export interface IngestContext {
  runId: string;
  observedAt: string;
  sourceKind: string;
  sourceId: string;
  schemaVersion: string;
  provenance?: Record<string, unknown>;
}

export type LedgerProvenance = SanitizedProvenance;

export interface IngestResult {
  observationId: string;
  observationInserted: boolean;
  messageInserted: number;
  messageDeduplicated: number;
  attemptInserted: number;
  attemptUpdated: number;
  attemptDeduplicated: number;
  aliasConflicts: number;
  coverage: "validated_page" | "partial" | "unrecognized";
  warnings: string[];
}

export interface ReconstructedAttempt extends AttemptRecord {
  scope: LedgerScope;
}
