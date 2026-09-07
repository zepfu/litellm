import {
  WORKING_ESTIMATE_LABEL,
  type DirectServerQuotaObservation,
  type QuotaEstimate,
} from "../accounting/quota.js";
import type { RawModelReport } from "../accounting/raw-model.js";
import { assertNoSecrets } from "../security/sanitizer.js";

export const REPORT_EXPORT_SCHEMA_VERSION = "d1-752-report-export-v1";
export const REPORT_RELEVANCE_LABEL = "100% to requested interval";

export type ReportExportFormat = "json" | "csv" | "markdown";
export type ReportFreshnessStatus = "fresh" | "stale" | "unknown";
export type ReportCoverageStatus = "complete" | "partial" | "unknown";
export type ExportConfidence = "A" | "B" | "C" | "F" | "unknown";

export type QuotaObservationSource =
  | "provider"
  | "operator"
  | "manual"
  | "unknown";

export type QuotaObservationQualification =
  | "provider_verified"
  | "operator_reported"
  | "manual"
  | "unknown";

export interface ReportFreshnessContext {
  status: ReportFreshnessStatus;
  observedAt?: string | null;
  ageMs?: number | null;
  maxAgeMs?: number | null;
}

export interface ReportCoverageContext {
  status: ReportCoverageStatus;
  source?: string | null;
  observedAt?: string | null;
}

export interface ReportProvenance {
  source?: string | null;
  revisionId?: string | null;
  inputFingerprint?: string | null;
  mappingVersion?: string | null;
}

export interface PolicyProvenance {
  source?: string | null;
  policyId?: string | null;
  version?: string | null;
  provenance?: string | null;
}

export interface WindowProvenance {
  source?: string | null;
  provenance?: string | null;
  windowType?: string | null;
  start?: string | null;
  end?: string | null;
  timezone?: string | null;
  evidenceObservedAt?: string | null;
}

export interface ReportProvenanceContext {
  report?: ReportProvenance;
  policy?: PolicyProvenance;
  windows?: Readonly<Record<string, WindowProvenance>>;
}

/**
 * This context is intentionally separate from DirectServerQuotaObservation.
 * The quota engine's observation shape does not establish who supplied a
 * value, and an observation field alone must never receive provider/A status.
 */
export interface QuotaObservationContext {
  source?: QuotaObservationSource;
  qualification?: QuotaObservationQualification;
  provenance?: string | null;
  scoped?: boolean;
}

export interface ReportExportContext {
  evaluatedAt: string;
  timezone: string;
  freshness: ReportFreshnessStatus | ReportFreshnessContext;
  coverage: ReportCoverageStatus | ReportCoverageContext;
  provenance?: ReportProvenanceContext;
  quotaObservationContextByBucket?: Readonly<
    Record<string, QuotaObservationContext>
  >;
}

export interface ReportExportInput {
  report: RawModelReport;
  context: ReportExportContext;
  quotaEstimate?: QuotaEstimate | null;
}

export interface ExportFreshnessContext {
  status: ReportFreshnessStatus;
  observedAt: string | null;
  ageMs: number | null;
  maxAgeMs: number | null;
}

export interface ExportCoverageContext {
  status: ReportCoverageStatus;
  source: string | null;
  observedAt: string | null;
}

export interface ExportCoverageGap {
  gapId: string | null;
  sourceKind: string | null;
  sourceId: string | null;
  reason: string | null;
  state: "open" | "resolved" | "unknown";
  firstSeenAt: string | null;
  lastSeenAt: string | null;
  details: {
    coverage: string | null;
    warnings: string[];
  };
}

export interface ExportRawModelReport {
  accountId: string | null;
  scopeKey: string | null;
  surface: string | null;
  durationMs: number | null;
  start: string | null;
  end: string | null;
  observedAttemptsByRequestedModel: Record<string, number | null>;
  completedAnswersByRecordedFinalModel: Record<string, number | null>;
  observedAttemptsByResolvedModel: Record<string, number | null>;
  modelMismatches: number | null;
  rawSlugDifferences: number | null;
  includedAttempts: number | null;
  possibleAttemptIds: string[];
  possibleAttemptsByRequestedModel: Record<string, number | null>;
  possibleCompletedAnswersByRecordedFinalModel: Record<string, number | null>;
  possibleAttemptsByResolvedModel: Record<string, number | null>;
  ambiguousTimeAttempts: number | null;
  unknownTimeAttempts: number | null;
  unresolvedAttempts: number | null;
  unknownModelAttempts: number | null;
  excludedSurfaceAttempts: number | null;
  excludedOriginAttempts: number | null;
  unclassifiedAttempts: number | null;
  attemptIds: string[];
  coverageGaps: ExportCoverageGap[];
  label: string | null;
}

export interface ExportServerQuotaObservation {
  bucketId: string;
  windowId: string;
  remaining: number | null;
  observedAt: string | null;
  source: QuotaObservationSource;
  qualification: QuotaObservationQualification;
  provenance: string | null;
  scoped: boolean;
  confidence: ExportConfidence;
}

export interface ExportQuotaDiscrepancy {
  kind: string;
  excess: number | null;
  qualification: string | null;
}

export interface ExportQuotaBucket {
  bucketId: string | null;
  kind: string | null;
  eligibleFamilies: string[];
  capacity: number | null;
  windowId: string | null;
  workingUsageEstimate: number | null;
  workingRemainingEstimate: number | null;
  workingRemainingUnclamped: number | null;
  discrepancy: ExportQuotaDiscrepancy | null;
  serverObservation: ExportServerQuotaObservation | null;
  countedAttemptIds: string[];
  unknownDebitAttemptIds: string[];
  unknownMembershipAttemptIds: string[];
  windowProvenance: WindowProvenance | null;
}

export interface ExportQuotaAssessment {
  attemptId: string;
  disposition: string;
  selectedFamily: string | null;
  basis: string | null;
  applicableBucketIds: string[];
  contributedBucketIds: string[];
  membershipByBucket: Record<string, string>;
  uncertainDebitCategories: string[];
  unclassifiedReasons: string[];
}

export interface ExportQuotaEstimate {
  policyId: string | null;
  policyVersion: string | null;
  workingEstimator: string | null;
  uncertainAttemptMode: string | null;
  coverage: string | null;
  label: typeof WORKING_ESTIMATE_LABEL;
  buckets: Record<string, ExportQuotaBucket>;
  workingUsageEstimateByBucket: Record<string, number | null>;
  workingRemainingEstimateByBucket: Record<string, number | null>;
  workingRemainingUnclampedByBucket: Record<string, number | null>;
  serverReportedRemainingByBucket: Record<
    string,
    ExportServerQuotaObservation | null
  >;
  workingHeadroomByFamily: Record<string, number | null>;
  countedAttemptIds: string[];
  rejectedBeforeStartAttemptIds: string[];
  outOfWindowAttemptIds: string[];
  unknownDebitAttempts: number | null;
  unknownDebitAttemptIds: string[];
  uncertainDebitCategories: Record<string, string[]>;
  unclassifiedAttempts: number | null;
  unclassifiedAttemptIds: string[];
  unclassifiedReasons: Record<string, string[]>;
  windowMembershipUnknownAttemptIds: string[];
  assessments: ExportQuotaAssessment[];
}

export interface ExportProvenance {
  report: {
    source: string | null;
    revisionId: string | null;
    inputFingerprint: string | null;
    mappingVersion: string | null;
  };
  policy: {
    source: string | null;
    policyId: string | null;
    version: string | null;
    provenance: string | null;
  };
  windows: Record<string, WindowProvenance>;
}

export interface ReportExportDocument {
  schemaVersion: typeof REPORT_EXPORT_SCHEMA_VERSION;
  relevance: typeof REPORT_RELEVANCE_LABEL;
  context: {
    evaluatedAt: string;
    timezone: string;
    freshness: ExportFreshnessContext;
    coverage: ExportCoverageContext;
  };
  provenance: ExportProvenance;
  report: ExportRawModelReport;
  quotaEstimate: ExportQuotaEstimate | null;
}

export function buildReportExport(
  input: ReportExportInput,
): ReportExportDocument {
  const evaluatedAt = requiredContextString(
    input.context.evaluatedAt,
    "context.evaluatedAt",
  );
  const timezone = requiredContextString(
    input.context.timezone,
    "context.timezone",
  );
  const document: ReportExportDocument = {
    schemaVersion: REPORT_EXPORT_SCHEMA_VERSION,
    relevance: REPORT_RELEVANCE_LABEL,
    context: {
      evaluatedAt,
      timezone,
      freshness: normalizeFreshness(input.context.freshness),
      coverage: normalizeCoverage(input.context.coverage),
    },
    provenance: normalizeProvenance(input.context.provenance),
    report: projectRawModelReport(input.report),
    quotaEstimate:
      input.quotaEstimate === undefined || input.quotaEstimate === null
        ? null
        : projectQuotaEstimate(
            input.quotaEstimate,
            input.context.quotaObservationContextByBucket,
            input.context.provenance?.windows,
          ),
  };
  assertNoSecrets(document);
  return document;
}

export function renderJson(input: ReportExportInput): string {
  return `${JSON.stringify(buildReportExport(input), null, 2)}\n`;
}

export const renderReportJson = renderJson;

export function renderCsv(input: ReportExportInput): string {
  const document = buildReportExport(input);
  const rows = csvRows(document);
  const header = [
    "section",
    "key",
    "value",
    "secondary_value",
    "status",
    "observed_at",
    "first_seen_at",
    "last_seen_at",
    "source",
    "qualification",
    "scoped",
    "window_id",
    "discrepancy_kind",
    "discrepancy_excess",
    "relevance",
    "confidence",
    "provenance",
  ];
  const lines = [
    header.map((value) => csvCell(value)).join(","),
    ...rows.map((row) =>
      [
        row.section,
        row.key,
        row.value,
        row.secondaryValue,
        row.status,
        row.observedAt,
        row.firstSeenAt,
        row.lastSeenAt,
        row.source,
        row.qualification,
        row.scoped,
        row.windowId,
        row.discrepancyKind,
        row.discrepancyExcess,
        row.relevance,
        row.confidence,
        row.provenance,
      ]
        .map((value) => csvCell(value))
        .join(","),
    ),
  ];
  return `${lines.join("\r\n")}\r\n`;
}

export const renderReportCsv = renderCsv;

export function renderMarkdown(input: ReportExportInput): string {
  const document = buildReportExport(input);
  const lines: string[] = [
    "# Offline report export",
    "",
    `Relevance: ${REPORT_RELEVANCE_LABEL}`,
    "",
    "This export is bounded to the supplied local report. Working quota values "
      + "are estimates and are not official remaining quota.",
    "",
    "## Context",
    "",
    markdownTable(
      ["Field", "Value", "Confidence"],
      [
        ["Evaluated at", document.context.evaluatedAt, "B"],
        ["Timezone", document.context.timezone, "B"],
        ["Freshness", formatFreshness(document.context.freshness), "B"],
        ["Coverage", formatCoverage(document.context.coverage), "B"],
      ],
    ),
    "",
    "## Raw model activity",
    "",
    "Confidence: B (reconstructed activity; requested and recorded-final "
      + "models remain separate).",
    "",
    markdownTable(
      [
        "Model",
        "Observed requested",
        "Completed recorded-final",
        "Observed resolved",
      ],
      countRows(document.report),
    ),
    "",
    "### Mismatches and uncertainty",
    "",
    markdownTable(
      ["Metric", "Value", "Confidence"],
      reportMetricRows(document.report),
    ),
    "",
    "## Coverage",
    "",
    markdownTable(
      ["Gap", "Reason", "State", "Source", "First seen", "Last seen"],
      document.report.coverageGaps.length === 0
        ? [["None", "No open coverage gaps", "complete", "Unknown", "Unknown", "Unknown"]]
        : document.report.coverageGaps.map((gap) => [
            gap.gapId,
            gap.reason,
            gap.state,
            gap.sourceKind,
            gap.firstSeenAt,
            gap.lastSeenAt,
          ]),
    ),
    "",
    "## Quota projection",
    "",
    document.quotaEstimate === null
      ? "No quota estimate supplied."
      : [
          `Working estimate label: ${WORKING_ESTIMATE_LABEL}`,
          "",
          `Policy: ${mdEscape(display(document.quotaEstimate.policyId))} ` +
            `(version ${mdEscape(display(document.quotaEstimate.policyVersion))})`,
          "",
          `Estimate coverage: ${mdEscape(
            display(document.quotaEstimate.coverage),
          )}`,
          "",
          markdownTable(
            [
              "Bucket",
              "Window",
              "Capacity",
              "Working usage",
              "Working remaining",
              "Unclamped remaining",
              "Server observed",
              "Observed at",
              "Source",
              "Qualification",
              "Discrepancy",
              "Confidence",
            ],
            quotaBucketRows(document.quotaEstimate),
          ),
          "",
          "Working estimates are confidence C because they depend on local "
            + "model, attempt, reset, and coverage interpretation.",
          "",
          "### Working headroom",
          "",
          markdownTable(
            ["Family", "Headroom", "Confidence"],
            Object.entries(document.quotaEstimate.workingHeadroomByFamily)
              .sort(([left], [right]) => left.localeCompare(right))
              .map(([family, headroom]) => [family, headroom, "C"]),
          ),
          "",
          "### Quota uncertainty",
          "",
          markdownTable(
            ["Category", "Attempt IDs", "Confidence"],
            quotaUncertaintyRows(document.quotaEstimate),
          ),
        ].join("\n"),
    "",
    "## Provenance",
    "",
    markdownTable(
      ["Scope", "Field", "Value"],
      provenanceRows(document.provenance),
    ),
    "",
    "Provider confidence is A only when the export supplies explicit "
      + "provider, provider_verified, and scoped observation context. "
      + "Operator, manual, or absent source context is never upgraded to A.",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

export const renderReportMarkdown = renderMarkdown;

export function renderReport(
  input: ReportExportInput,
  format: ReportExportFormat,
): string {
  switch (format) {
    case "json":
      return renderJson(input);
    case "csv":
      return renderCsv(input);
    case "markdown":
      return renderMarkdown(input);
  }
}

interface CsvRow {
  section: string;
  key: string;
  value: string | number | null;
  secondaryValue: string | number | null;
  status: string | null;
  observedAt: string | null;
  firstSeenAt: string | null;
  lastSeenAt: string | null;
  source: string | null;
  qualification: string | null;
  scoped: boolean | null;
  windowId: string | null;
  discrepancyKind: string | null;
  discrepancyExcess: number | null;
  relevance: string | null;
  confidence: ExportConfidence | null;
  provenance: string | null;
}

function csvRows(document: ReportExportDocument): CsvRow[] {
  const rows: CsvRow[] = [];
  const add = (row: Partial<CsvRow> & Pick<CsvRow, "section" | "key">): void => {
    rows.push({
      section: row.section,
      key: row.key,
      value: row.value ?? null,
      secondaryValue: row.secondaryValue ?? null,
      status: row.status ?? null,
      observedAt: row.observedAt ?? null,
      firstSeenAt: row.firstSeenAt ?? null,
      lastSeenAt: row.lastSeenAt ?? null,
      source: row.source ?? null,
      qualification: row.qualification ?? null,
      scoped: row.scoped ?? null,
      windowId: row.windowId ?? null,
      discrepancyKind: row.discrepancyKind ?? null,
      discrepancyExcess: row.discrepancyExcess ?? null,
      relevance: row.relevance ?? null,
      confidence: row.confidence ?? null,
      provenance: row.provenance ?? null,
    });
  };
  const relevance = REPORT_RELEVANCE_LABEL;

  add({
    section: "context",
    key: "evaluated_at",
    value: document.context.evaluatedAt,
    status: "context",
    relevance,
    confidence: "B",
  });
  add({
    section: "context",
    key: "timezone",
    value: document.context.timezone,
    status: "context",
    relevance,
    confidence: "B",
  });
  add({
    section: "context",
    key: "freshness",
    value: document.context.freshness.status,
    secondaryValue: document.context.freshness.ageMs,
    observedAt: document.context.freshness.observedAt,
    status: "context",
    relevance,
    confidence: "B",
  });
  add({
    section: "context",
    key: "coverage",
    value: document.context.coverage.status,
    secondaryValue: document.context.coverage.source,
    observedAt: document.context.coverage.observedAt,
    status: "context",
    relevance,
    confidence: "B",
  });

  const report = document.report;
  for (const [key, value] of [
    ["account_id", report.accountId],
    ["scope_key", report.scopeKey],
    ["surface", report.surface],
    ["duration_ms", report.durationMs],
    ["start", report.start],
    ["end", report.end],
    ["label", report.label],
  ] as const) {
    add({
      section: "report",
      key,
      value,
      status: "context",
      relevance,
      confidence: "B",
    });
  }
  addCountMap(add, "observed_requested", report.observedAttemptsByRequestedModel);
  addCountMap(
    add,
    "completed_recorded_final",
    report.completedAnswersByRecordedFinalModel,
  );
  addCountMap(add, "observed_resolved", report.observedAttemptsByResolvedModel);
  addCountMap(
    add,
    "possible_requested",
    report.possibleAttemptsByRequestedModel,
    "possible",
  );
  addCountMap(
    add,
    "possible_recorded_final",
    report.possibleCompletedAnswersByRecordedFinalModel,
    "possible",
  );
  addCountMap(
    add,
    "possible_resolved",
    report.possibleAttemptsByResolvedModel,
    "possible",
  );

  for (const [key, value] of [
    ["model_mismatches", report.modelMismatches],
    ["raw_slug_differences", report.rawSlugDifferences],
    ["included_attempts", report.includedAttempts],
    ["ambiguous_time_attempts", report.ambiguousTimeAttempts],
    ["unknown_time_attempts", report.unknownTimeAttempts],
    ["unresolved_attempts", report.unresolvedAttempts],
    ["unknown_model_attempts", report.unknownModelAttempts],
    ["excluded_surface_attempts", report.excludedSurfaceAttempts],
    ["excluded_origin_attempts", report.excludedOriginAttempts],
    ["unclassified_attempts", report.unclassifiedAttempts],
  ] as const) {
    add({
      section: "report_metric",
      key,
      value,
      status: key.includes("mismatch") || key.includes("uncertain")
        ? "uncertainty"
        : "observed",
      relevance,
      confidence: "B",
    });
  }
  for (const attemptId of report.attemptIds) {
    add({
      section: "attempt_id",
      key: attemptId,
      value: "observed",
      status: "observed",
      relevance,
      confidence: "B",
    });
  }
  for (const attemptId of report.possibleAttemptIds) {
    add({
      section: "possible_attempt_id",
      key: attemptId,
      value: "possible",
      status: "possible",
      relevance,
      confidence: "B",
    });
  }
  for (const gap of report.coverageGaps) {
    add({
      section: "coverage_gap",
      key: gap.gapId ?? "unknown-gap",
      value: gap.reason,
      secondaryValue: gap.sourceId,
      status: gap.state,
      firstSeenAt: gap.firstSeenAt,
      lastSeenAt: gap.lastSeenAt,
      source: gap.sourceKind,
      relevance,
      confidence: "B",
      provenance: [
        gap.details.coverage,
        ...gap.details.warnings,
      ]
        .filter((item) => item !== null)
        .join("; ") || null,
    });
  }

  addProvenanceRows(add, document.provenance, relevance);

  const quota = document.quotaEstimate;
  if (quota === null) {
    add({
      section: "quota",
      key: "estimate",
      value: null,
      status: "not_supplied",
      relevance,
      confidence: "C",
    });
    return rows;
  }
  for (const [key, value] of [
    ["policy_id", quota.policyId],
    ["policy_version", quota.policyVersion],
    ["working_estimator", quota.workingEstimator],
    ["uncertain_attempt_mode", quota.uncertainAttemptMode],
    ["coverage", quota.coverage],
    ["label", quota.label],
    ["unknown_debit_attempts", quota.unknownDebitAttempts],
    ["unclassified_attempts", quota.unclassifiedAttempts],
  ] as const) {
    add({
      section: "quota",
      key,
      value,
      status: "working_estimate",
      relevance,
      confidence: "C",
    });
  }
  addCountMap(add, "working_usage", quota.workingUsageEstimateByBucket, "working_estimate", "C");
  addCountMap(add, "working_remaining", quota.workingRemainingEstimateByBucket, "working_estimate", "C");
  addCountMap(
    add,
    "working_remaining_unclamped",
    quota.workingRemainingUnclampedByBucket,
    "working_estimate",
    "C",
  );
  addCountMap(add, "working_headroom", quota.workingHeadroomByFamily, "working_estimate", "C");
  addStringMap(add, "uncertain_debit_category", quota.uncertainDebitCategories, relevance);
  addStringMap(add, "unclassified_reason", quota.unclassifiedReasons, relevance);
  addStringRows(add, "counted_attempt_id", quota.countedAttemptIds, "counted", relevance, "C");
  addStringRows(
    add,
    "rejected_before_start_attempt_id",
    quota.rejectedBeforeStartAttemptIds,
    "rejected_before_start",
    relevance,
    "C",
  );
  addStringRows(add, "out_of_window_attempt_id", quota.outOfWindowAttemptIds, "out_of_window", relevance, "C");
  addStringRows(add, "unknown_debit_attempt_id", quota.unknownDebitAttemptIds, "unknown_debit", relevance, "C");
  addStringRows(add, "unclassified_attempt_id", quota.unclassifiedAttemptIds, "unclassified", relevance, "C");
  addStringRows(
    add,
    "window_membership_unknown_attempt_id",
    quota.windowMembershipUnknownAttemptIds,
    "unknown_membership",
    relevance,
    "C",
  );
  for (const bucket of Object.values(quota.buckets).sort(compareBucket)) {
    const key = bucket.bucketId ?? "unknown-bucket";
    add({
      section: "quota_bucket",
      key: `${key}.kind`,
      value: bucket.kind,
      status: "working_estimate",
      windowId: bucket.windowId,
      relevance,
      confidence: "C",
    });
    add({
      section: "quota_bucket",
      key: `${key}.capacity`,
      value: bucket.capacity,
      status: "working_estimate",
      windowId: bucket.windowId,
      relevance,
      confidence: "C",
    });
    add({
      section: "quota_bucket",
      key: `${key}.eligible_families`,
      value: bucket.eligibleFamilies.join("|"),
      status: "working_estimate",
      windowId: bucket.windowId,
      relevance,
      confidence: "C",
    });
    add({
      section: "quota_bucket",
      key: `${key}.working_usage_estimate`,
      value: bucket.workingUsageEstimate,
      status: "working_estimate",
      windowId: bucket.windowId,
      relevance,
      confidence: "C",
    });
    add({
      section: "quota_bucket",
      key: `${key}.working_remaining_estimate`,
      value: bucket.workingRemainingEstimate,
      status: "working_estimate",
      windowId: bucket.windowId,
      relevance,
      confidence: "C",
    });
    add({
      section: "quota_bucket",
      key: `${key}.working_remaining_unclamped`,
      value: bucket.workingRemainingUnclamped,
      status: "working_estimate",
      windowId: bucket.windowId,
      relevance,
      confidence: "C",
    });
    if (bucket.discrepancy !== null) {
      add({
        section: "quota_discrepancy",
        key,
        value:
          bucket.discrepancy.excess === null
            ? null
            : -Math.abs(bucket.discrepancy.excess),
        secondaryValue: bucket.discrepancy.qualification,
        status: "negative_discrepancy",
        windowId: bucket.windowId,
        discrepancyKind: bucket.discrepancy.kind,
        discrepancyExcess: bucket.discrepancy.excess,
        relevance,
        confidence: "C",
      });
    }
    const observation = bucket.serverObservation;
    if (observation === null) {
      add({
        section: "server_observation",
        key,
        value: null,
        status: "not_observed",
        windowId: bucket.windowId,
        relevance,
        confidence: "unknown",
      });
    } else {
      add({
        section: "server_observation",
        key,
        value: observation.remaining,
        secondaryValue: observation.windowId,
        status: "observed",
        observedAt: observation.observedAt,
        source: observation.source,
        qualification: observation.qualification,
        scoped: observation.scoped,
        windowId: observation.windowId,
        relevance,
        confidence: observation.confidence,
        provenance: observation.provenance,
      });
    }
    addStringRows(add, `${key}.counted_attempt_id`, bucket.countedAttemptIds, "counted", relevance, "C");
    addStringRows(add, `${key}.unknown_debit_attempt_id`, bucket.unknownDebitAttemptIds, "unknown_debit", relevance, "C");
    addStringRows(
      add,
      `${key}.unknown_membership_attempt_id`,
      bucket.unknownMembershipAttemptIds,
      "unknown_membership",
      relevance,
      "C",
    );
  }
  for (const assessment of quota.assessments) {
    add({
      section: "quota_assessment",
      key: assessment.attemptId,
      value: assessment.disposition,
      secondaryValue: assessment.selectedFamily,
      status: "working_estimate",
      relevance,
      confidence: "C",
      provenance: [
        `basis=${assessment.basis ?? "Unknown"}`,
        `applicable=${assessment.applicableBucketIds.join("|") || "Unknown"}`,
        `contributed=${assessment.contributedBucketIds.join("|") || "Unknown"}`,
        `uncertain=${assessment.uncertainDebitCategories.join("|") || "Unknown"}`,
        `reasons=${assessment.unclassifiedReasons.join("|") || "Unknown"}`,
      ].join("; "),
    });
  }
  return rows;
}

function addCountMap(
  add: (row: Partial<CsvRow> & Pick<CsvRow, "section" | "key">) => void,
  section: string,
  map: Readonly<Record<string, number | null>>,
  status = "observed",
  confidence: ExportConfidence = "B",
): void {
  for (const [key, value] of Object.entries(map).sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    add({
      section,
      key,
      value,
      status,
      relevance: REPORT_RELEVANCE_LABEL,
      confidence,
    });
  }
}

function addStringMap(
  add: (row: Partial<CsvRow> & Pick<CsvRow, "section" | "key">) => void,
  section: string,
  map: Readonly<Record<string, string[]>>,
  relevance: string,
): void {
  for (const [key, values] of Object.entries(map).sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    add({
      section,
      key,
      value: values.join("|") || null,
      status: "uncertainty",
      relevance,
      confidence: "C",
    });
  }
}

function addStringRows(
  add: (row: Partial<CsvRow> & Pick<CsvRow, "section" | "key">) => void,
  section: string,
  values: readonly string[],
  status: string,
  relevance: string,
  confidence: ExportConfidence,
): void {
  for (const value of values) {
    add({
      section,
      key: value,
      value: status,
      status,
      relevance,
      confidence,
    });
  }
}

function addProvenanceRows(
  add: (row: Partial<CsvRow> & Pick<CsvRow, "section" | "key">) => void,
  provenance: ExportProvenance,
  relevance: string,
): void {
  for (const [key, value] of [
    ["source", provenance.report.source],
    ["revision_id", provenance.report.revisionId],
    ["input_fingerprint", provenance.report.inputFingerprint],
    ["mapping_version", provenance.report.mappingVersion],
  ] as const) {
    add({
      section: "report_provenance",
      key,
      value,
      status: "provenance",
      relevance,
      confidence: "B",
    });
  }
  for (const [key, value] of [
    ["source", provenance.policy.source],
    ["policy_id", provenance.policy.policyId],
    ["version", provenance.policy.version],
    ["provenance", provenance.policy.provenance],
  ] as const) {
    add({
      section: "policy_provenance",
      key,
      value,
      status: "provenance",
      relevance,
      confidence: "C",
    });
  }
  for (const [bucketId, window] of Object.entries(provenance.windows).sort(
    ([left], [right]) => left.localeCompare(right),
  )) {
    for (const [field, value] of [
      ["source", window.source],
      ["provenance", window.provenance],
      ["window_type", window.windowType],
      ["start", window.start],
      ["end", window.end],
      ["timezone", window.timezone],
      ["evidence_observed_at", window.evidenceObservedAt],
    ] as const) {
      add({
        section: "window_provenance",
        key: `${bucketId}.${field}`,
        value: stringOrNull(value),
        status: "provenance",
        relevance,
        confidence: "C",
      });
    }
  }
}

function csvCell(value: string | number | boolean | null): string {
  if (value === null) {
    return "\"Unknown\"";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "\"Unknown\"";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  const formulaSafe = /^[=+\-@]/.test(value) ? `'${value}` : value;
  return `"${formulaSafe.replaceAll("\"", "\"\"")}"`;
}

function projectRawModelReport(report: RawModelReport): ExportRawModelReport {
  return {
    accountId: stringOrNull(report.accountId),
    scopeKey: stringOrNull(report.scopeKey),
    surface: stringOrNull(report.surface),
    durationMs: finiteOrNull(report.durationMs),
    start: stringOrNull(report.start),
    end: stringOrNull(report.end),
    observedAttemptsByRequestedModel: projectNumberMap(
      report.observedAttemptsByRequestedModel,
    ),
    completedAnswersByRecordedFinalModel: projectNumberMap(
      report.completedAnswersByRecordedFinalModel,
    ),
    observedAttemptsByResolvedModel: projectNumberMap(
      report.observedAttemptsByResolvedModel,
    ),
    modelMismatches: finiteOrNull(report.modelMismatches),
    rawSlugDifferences: finiteOrNull(report.rawSlugDifferences),
    includedAttempts: finiteOrNull(report.includedAttempts),
    possibleAttemptIds: projectStringArray(report.possibleAttemptIds),
    possibleAttemptsByRequestedModel: projectNumberMap(
      report.possibleAttemptsByRequestedModel,
    ),
    possibleCompletedAnswersByRecordedFinalModel: projectNumberMap(
      report.possibleCompletedAnswersByRecordedFinalModel,
    ),
    possibleAttemptsByResolvedModel: projectNumberMap(
      report.possibleAttemptsByResolvedModel,
    ),
    ambiguousTimeAttempts: finiteOrNull(report.ambiguousTimeAttempts),
    unknownTimeAttempts: finiteOrNull(report.unknownTimeAttempts),
    unresolvedAttempts: finiteOrNull(report.unresolvedAttempts),
    unknownModelAttempts: finiteOrNull(report.unknownModelAttempts),
    excludedSurfaceAttempts: finiteOrNull(report.excludedSurfaceAttempts),
    excludedOriginAttempts: finiteOrNull(report.excludedOriginAttempts),
    unclassifiedAttempts: finiteOrNull(report.unclassifiedAttempts),
    attemptIds: projectStringArray(report.attemptIds),
    coverageGaps: Array.isArray(report.coverageGaps)
      ? report.coverageGaps
          .filter(isRecord)
          .map(projectCoverageGap)
          .sort(compareGap)
      : [],
    label: stringOrNull(report.label),
  };
}

function projectQuotaEstimate(
  quota: QuotaEstimate,
  observationContexts:
    | Readonly<Record<string, QuotaObservationContext>>
    | undefined,
  windowProvenance:
    | Readonly<Record<string, WindowProvenance>>
    | undefined,
): ExportQuotaEstimate {
  const buckets = Object.fromEntries(
    Object.values(quota.buckets)
      .map((bucket) => {
        const observation = bucket.serverReportedRemaining ??
          quota.serverReportedRemainingByBucket[bucket.bucketId] ??
          null;
        const bucketId = stringOrNull(bucket.bucketId);
        const observationContext =
          bucketId === null ? undefined : observationContexts?.[bucketId];
        return [
          bucketId ?? "unknown-bucket",
          {
            bucketId,
            kind: stringOrNull(bucket.kind),
            eligibleFamilies: projectStringArray(bucket.eligibleFamilies),
            capacity: finiteOrNull(bucket.capacity),
            windowId: stringOrNull(bucket.windowId),
            workingUsageEstimate: finiteOrNull(bucket.workingUsageEstimate),
            workingRemainingEstimate: finiteOrNull(
              bucket.workingRemainingEstimate,
            ),
            workingRemainingUnclamped: finiteOrNull(
              bucket.workingRemainingUnclamped,
            ),
            discrepancy:
              bucket.discrepancy === null
                ? null
                : {
                    kind: stringOrNull(bucket.discrepancy.kind) ?? "unknown",
                    excess: finiteOrNull(bucket.discrepancy.excess),
                    qualification: stringOrNull(bucket.discrepancy.qualification),
                  },
            serverObservation: observation === null
              ? null
              : projectServerObservation(
                  observation,
                  observationContext,
                ),
            countedAttemptIds: projectStringArray(bucket.countedAttemptIds),
            unknownDebitAttemptIds: projectStringArray(
              bucket.unknownDebitAttemptIds,
            ),
            unknownMembershipAttemptIds: projectStringArray(
              bucket.unknownMembershipAttemptIds,
            ),
            windowProvenance:
              bucketId === null
                ? null
                : projectWindowProvenance(windowProvenance?.[bucketId]),
          } satisfies ExportQuotaBucket,
        ] as const;
      })
      .sort(([left], [right]) => left.localeCompare(right)),
  ) as Record<string, ExportQuotaBucket>;

  const serverReportedRemainingByBucket = Object.fromEntries(
    Object.entries(buckets).map(([bucketId, bucket]) => [
      bucketId,
      bucket.serverObservation,
    ]),
  ) as Record<string, ExportServerQuotaObservation | null>;

  return {
    policyId: stringOrNull(quota.policyId),
    policyVersion: stringOrNull(quota.policyVersion),
    workingEstimator: stringOrNull(quota.workingEstimator),
    uncertainAttemptMode: stringOrNull(quota.uncertainAttemptMode),
    coverage: stringOrNull(quota.coverage),
    label: WORKING_ESTIMATE_LABEL,
    buckets,
    workingUsageEstimateByBucket: projectNumberMap(
      quota.workingUsageEstimateByBucket,
    ),
    workingRemainingEstimateByBucket: projectNumberMap(
      quota.workingRemainingEstimateByBucket,
    ),
    workingRemainingUnclampedByBucket: projectNumberMap(
      quota.workingRemainingUnclampedByBucket,
    ),
    serverReportedRemainingByBucket,
    workingHeadroomByFamily: projectNumberMap(quota.workingHeadroomByFamily),
    countedAttemptIds: projectStringArray(quota.countedAttemptIds),
    rejectedBeforeStartAttemptIds: projectStringArray(
      quota.rejectedBeforeStartAttemptIds,
    ),
    outOfWindowAttemptIds: projectStringArray(quota.outOfWindowAttemptIds),
    unknownDebitAttempts: finiteOrNull(quota.unknownDebitAttempts),
    unknownDebitAttemptIds: projectStringArray(quota.unknownDebitAttemptIds),
    uncertainDebitCategories: projectStringMap(
      quota.uncertainDebitCategories,
    ),
    unclassifiedAttempts: finiteOrNull(quota.unclassifiedAttempts),
    unclassifiedAttemptIds: projectStringArray(quota.unclassifiedAttemptIds),
    unclassifiedReasons: projectStringMap(quota.unclassifiedReasons),
    windowMembershipUnknownAttemptIds: projectStringArray(
      quota.windowMembershipUnknownAttemptIds,
    ),
    assessments: quota.assessments
      .map((assessment) => ({
        attemptId: stringOrNull(assessment.attemptId) ?? "unknown-attempt",
        disposition: stringOrNull(assessment.disposition) ?? "unknown",
        selectedFamily: stringOrNull(assessment.selectedFamily),
        basis: stringOrNull(assessment.basis),
        applicableBucketIds: projectStringArray(
          assessment.applicableBucketIds,
        ),
        contributedBucketIds: projectStringArray(
          assessment.contributedBucketIds,
        ),
        membershipByBucket: projectStringMapValues(
          assessment.membershipByBucket,
        ),
        uncertainDebitCategories: projectStringArray(
          assessment.uncertainDebitCategories,
        ),
        unclassifiedReasons: projectStringArray(
          assessment.unclassifiedReasons,
        ),
      }))
      .sort((left, right) => left.attemptId.localeCompare(right.attemptId)),
  };
}

function projectServerObservation(
  observation: DirectServerQuotaObservation,
  context: QuotaObservationContext | undefined,
): ExportServerQuotaObservation {
  const source = enumOrUnknown(
    context?.source,
    ["provider", "operator", "manual", "unknown"] as const,
  );
  const qualification = enumOrUnknown(
    context?.qualification,
    ["provider_verified", "operator_reported", "manual", "unknown"] as const,
  );
  const scoped = context?.scoped === true;
  const explicitContext =
    context !== undefined &&
    source === "provider" &&
    qualification === "provider_verified" &&
    scoped;
  return {
    bucketId: stringOrNull(observation.bucketId) ?? "unknown-bucket",
    windowId: stringOrNull(observation.windowId) ?? "unknown-window",
    remaining: finiteOrNull(observation.remaining),
    observedAt: stringOrNull(observation.observedAt),
    source,
    qualification,
    provenance: stringOrNull(context?.provenance),
    scoped,
    confidence: explicitContext ? "A" : context === undefined ? "unknown" : "F",
  };
}

function normalizeFreshness(
  input: ReportFreshnessStatus | ReportFreshnessContext,
): ExportFreshnessContext {
  if (typeof input === "string") {
    return {
      status: normalizeStatus(
        input,
        ["fresh", "stale", "unknown"] as const,
      ),
      observedAt: null,
      ageMs: null,
      maxAgeMs: null,
    };
  }
  return {
    status: normalizeStatus(
      input.status,
      ["fresh", "stale", "unknown"] as const,
    ),
    observedAt: stringOrNull(input.observedAt),
    ageMs: finiteOrNull(input.ageMs),
    maxAgeMs: finiteOrNull(input.maxAgeMs),
  };
}

function normalizeCoverage(
  input: ReportCoverageStatus | ReportCoverageContext,
): ExportCoverageContext {
  if (typeof input === "string") {
    return {
      status: normalizeStatus(
        input,
        ["complete", "partial", "unknown"] as const,
      ),
      source: null,
      observedAt: null,
    };
  }
  return {
    status: normalizeStatus(
      input.status,
      ["complete", "partial", "unknown"] as const,
    ),
    source: stringOrNull(input.source),
    observedAt: stringOrNull(input.observedAt),
  };
}

function normalizeProvenance(
  input: ReportProvenanceContext | undefined,
): ExportProvenance {
  const report = input?.report;
  const policy = input?.policy;
  return {
    report: {
      source: stringOrNull(report?.source),
      revisionId: stringOrNull(report?.revisionId),
      inputFingerprint: stringOrNull(report?.inputFingerprint),
      mappingVersion: stringOrNull(report?.mappingVersion),
    },
    policy: {
      source: stringOrNull(policy?.source),
      policyId: stringOrNull(policy?.policyId),
      version: stringOrNull(policy?.version),
      provenance: stringOrNull(policy?.provenance),
    },
    windows: Object.fromEntries(
      Object.entries(input?.windows ?? {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([bucketId, value]) => [
          bucketId,
          projectWindowProvenance(value) as WindowProvenance,
        ]),
    ),
  };
}

function projectWindowProvenance(
  value: WindowProvenance | undefined,
): WindowProvenance | null {
  if (value === undefined) {
    return null;
  }
  return {
    source: stringOrNull(value.source),
    provenance: stringOrNull(value.provenance),
    windowType: stringOrNull(value.windowType),
    start: stringOrNull(value.start),
    end: stringOrNull(value.end),
    timezone: stringOrNull(value.timezone),
    evidenceObservedAt: stringOrNull(value.evidenceObservedAt),
  };
}

function projectCoverageGap(value: Record<string, unknown>): ExportCoverageGap {
  const details = isRecord(value.details) ? value.details : {};
  return {
    gapId: firstString(value, "gap_id", "gapId"),
    sourceKind: firstString(value, "source_kind", "sourceKind"),
    sourceId: firstString(value, "source_id", "sourceId"),
    reason: firstString(value, "reason"),
    state: enumOrUnknown(value.state, ["open", "resolved", "unknown"] as const),
    firstSeenAt: firstString(value, "first_seen_at", "firstSeenAt"),
    lastSeenAt: firstString(value, "last_seen_at", "lastSeenAt"),
    details: {
      coverage: firstString(details, "coverage"),
      warnings: projectStringArray(details.warnings),
    },
  };
}

function projectNumberMap(value: unknown): Record<string, number | null> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, finiteOrNull(item)]),
  );
}

function projectStringMap(
  value: unknown,
): Record<string, string[]> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, projectStringArray(item)]),
  );
}

function projectStringMapValues(value: unknown): Record<string, string> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, stringOrNull(item) ?? "unknown"]),
  );
}

function projectStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string").sort()
    : [];
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function firstString(
  value: Record<string, unknown>,
  ...keys: string[]
): string | null {
  for (const key of keys) {
    const candidate = stringOrNull(value[key]);
    if (candidate !== null) {
      return candidate;
    }
  }
  return null;
}

function enumOrUnknown<T extends string>(
  value: unknown,
  allowed: readonly T[],
): T | "unknown" {
  return typeof value === "string" && allowed.includes(value as T)
    ? (value as T)
    : "unknown";
}

function normalizeStatus<T extends string>(
  value: unknown,
  allowed: readonly T[],
): T {
  return typeof value === "string" && allowed.includes(value as T)
    ? (value as T)
    : allowed[allowed.length - 1] as T;
}

function requiredContextString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function compareGap(left: ExportCoverageGap, right: ExportCoverageGap): number {
  return (left.gapId ?? "").localeCompare(right.gapId ?? "");
}

function compareBucket(
  left: ExportQuotaBucket,
  right: ExportQuotaBucket,
): number {
  return (left.bucketId ?? "").localeCompare(right.bucketId ?? "");
}

function display(value: string | number | null): string {
  if (value === null || (typeof value === "number" && !Number.isFinite(value))) {
    return "Unknown";
  }
  return String(value);
}

function formatFreshness(value: ExportFreshnessContext): string {
  const details = [
    value.status,
    value.observedAt === null ? null : `observed=${value.observedAt}`,
    value.ageMs === null ? null : `age_ms=${value.ageMs}`,
    value.maxAgeMs === null ? null : `max_age_ms=${value.maxAgeMs}`,
  ].filter((item): item is string => item !== null);
  return details.join("; ");
}

function formatCoverage(value: ExportCoverageContext): string {
  return [
    value.status,
    value.source === null ? null : `source=${value.source}`,
    value.observedAt === null ? null : `observed=${value.observedAt}`,
  ]
    .filter((item): item is string => item !== null)
    .join("; ");
}

function mdEscape(value: string): string {
  return value
    .replaceAll("\\", "\\\\")
    .replaceAll("|", "\\|")
    .replaceAll("`", "\\`")
    .replaceAll("*", "\\*")
    .replaceAll("_", "\\_")
    .replaceAll("[", "\\[")
    .replaceAll("]", "\\]")
    .replaceAll("<", "\\<")
    .replaceAll(">", "\\>")
    .replaceAll("\r", " ")
    .replaceAll("\n", " ");
}

function markdownTable(
  headers: readonly string[],
  rows: ReadonlyArray<ReadonlyArray<string | number | boolean | null>>,
): string {
  const header = `| ${headers.map(mdEscape).join(" | ")} |`;
  const separator = `| ${headers.map(() => "---").join(" | ")} |`;
  const body = rows.map(
    (row) => `| ${headers.map((_, index) => mdCell(row[index] ?? null)).join(" | ")} |`,
  );
  return [header, separator, ...body].join("\n");
}

function mdCell(value: string | number | boolean | null): string {
  if (value === null) {
    return "Unknown";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "Unknown";
  }
  return mdEscape(String(value));
}

function countRows(
  report: ExportRawModelReport,
): ReadonlyArray<ReadonlyArray<string | number | null>> {
  const keys = new Set([
    ...Object.keys(report.observedAttemptsByRequestedModel),
    ...Object.keys(report.completedAnswersByRecordedFinalModel),
    ...Object.keys(report.observedAttemptsByResolvedModel),
  ]);
  return [...keys]
    .sort()
    .map((key) => [
      key,
      report.observedAttemptsByRequestedModel[key] ?? null,
      report.completedAnswersByRecordedFinalModel[key] ?? null,
      report.observedAttemptsByResolvedModel[key] ?? null,
    ]);
}

function reportMetricRows(
  report: ExportRawModelReport,
): ReadonlyArray<ReadonlyArray<string | number | null>> {
  return [
    ["Model mismatches", report.modelMismatches, "B"],
    ["Raw slug differences", report.rawSlugDifferences, "B"],
    ["Included attempts", report.includedAttempts, "B"],
    ["Possible attempts", report.possibleAttemptIds.length, "B"],
    ["Ambiguous time attempts", report.ambiguousTimeAttempts, "B"],
    ["Unknown time attempts", report.unknownTimeAttempts, "B"],
    ["Unresolved attempts", report.unresolvedAttempts, "B"],
    ["Unknown model attempts", report.unknownModelAttempts, "B"],
    ["Excluded surface attempts", report.excludedSurfaceAttempts, "B"],
    ["Excluded origin attempts", report.excludedOriginAttempts, "B"],
    ["Unclassified attempts", report.unclassifiedAttempts, "B"],
  ];
}

function quotaBucketRows(
  quota: ExportQuotaEstimate,
): ReadonlyArray<ReadonlyArray<string | number | null>> {
  return Object.values(quota.buckets)
    .sort(compareBucket)
    .map((bucket) => {
      const observation = bucket.serverObservation;
      const discrepancy = bucket.discrepancy;
      const negativeDiscrepancy =
        discrepancy?.excess === null || discrepancy === null
          ? null
          : -Math.abs(discrepancy.excess);
      return [
        bucket.bucketId,
        bucket.windowId,
        bucket.capacity,
        bucket.workingUsageEstimate,
        bucket.workingRemainingEstimate,
        bucket.workingRemainingUnclamped,
        observation?.remaining ?? null,
        observation?.observedAt ?? null,
        observation?.source ?? null,
        observation?.qualification ?? null,
        negativeDiscrepancy === null
          ? null
          : `${negativeDiscrepancy} (${discrepancy?.kind ?? "unknown"}; ` +
            `${discrepancy?.qualification ?? "Unknown"})`,
        observation?.confidence ?? "C",
      ];
    });
}

function quotaUncertaintyRows(
  quota: ExportQuotaEstimate,
): ReadonlyArray<ReadonlyArray<string | number | null>> {
  const rows: Array<ReadonlyArray<string | number | null>> = [
    ["Unknown debit attempts", quota.unknownDebitAttempts, "C"],
    ["Unclassified attempts", quota.unclassifiedAttempts, "C"],
  ];
  for (const [category, attemptIds] of Object.entries(
    quota.uncertainDebitCategories,
  ).sort(([left], [right]) => left.localeCompare(right))) {
    rows.push([category, attemptIds.join(", ") || null, "C"]);
  }
  for (const [reason, attemptIds] of Object.entries(
    quota.unclassifiedReasons,
  ).sort(([left], [right]) => left.localeCompare(right))) {
    rows.push([reason, attemptIds.join(", ") || null, "C"]);
  }
  return rows;
}

function provenanceRows(
  provenance: ExportProvenance,
): ReadonlyArray<ReadonlyArray<string | null>> {
  const rows: Array<ReadonlyArray<string | null>> = [
    ["report", "source", provenance.report.source],
    ["report", "revision_id", provenance.report.revisionId],
    ["report", "input_fingerprint", provenance.report.inputFingerprint],
    ["report", "mapping_version", provenance.report.mappingVersion],
    ["policy", "source", provenance.policy.source],
    ["policy", "policy_id", provenance.policy.policyId],
    ["policy", "version", provenance.policy.version],
    ["policy", "provenance", provenance.policy.provenance],
  ];
  for (const [bucketId, window] of Object.entries(provenance.windows).sort(
    ([left], [right]) => left.localeCompare(right),
  )) {
    rows.push(
      ["window", `${bucketId}.source`, window.source ?? null],
      ["window", `${bucketId}.provenance`, window.provenance ?? null],
      ["window", `${bucketId}.type`, window.windowType ?? null],
      ["window", `${bucketId}.start`, window.start ?? null],
      ["window", `${bucketId}.end`, window.end ?? null],
      ["window", `${bucketId}.timezone`, window.timezone ?? null],
      [
        "window",
        `${bucketId}.evidence_observed_at`,
        window.evidenceObservedAt ?? null,
      ],
    );
  }
  return rows;
}
