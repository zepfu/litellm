import type { HistoryCoverageResult } from "../contracts/history.js";
import type { ReconstructedAttempt } from "../ledger/types.js";
import type {
  QuotaCoverageStatus,
  QuotaEstimate,
  QuotaEstimatorInput,
  QuotaOwnershipExpectation,
  QuotaPolicy,
  ResolvedQuotaWindow,
  UncertainAttemptMode,
} from "../accounting/quota.js";
import { estimateQuota } from "../accounting/quota.js";
import {
  evaluateResetWindow,
  intervalMembership,
  resolveResetWindow,
  type ResetWindowEvidence,
  type WindowMembership,
} from "../accounting/windows.js";
import type { CoverageLevel } from "../contracts/records.js";

export const USAGE_REPORT_SNAPSHOT_VERSION = 1;

export type UsageModelDimension = "requested" | "recordedFinal" | "resolved";

export interface UsageCoverage {
  history: CoverageLevel;
  overall: CoverageLevel;
  projects: CoverageLevel;
  branches: CoverageLevel;
  gaps: readonly string[];
}

export interface UsageReportSnapshot {
  snapshotVersion: typeof USAGE_REPORT_SNAPSHOT_VERSION;
  collectorAccountId: string;
  attempts: readonly ReconstructedAttempt[];
  coverage: UsageCoverage;
  historyCoverage?: HistoryCoverageResult;
}

export interface UsageReportRequest {
  account: string;
  asOf: string | Date;
  modelDimension: UsageModelDimension;
  calendarDay?: {
    timezone: string;
  };
  elapsedLookbackMs?: number;
  resetWindow?: {
    evidence: readonly ResetWindowEvidence[];
  };
}

export interface UsageWindow {
  kind: "calendar_day" | "elapsed_lookback" | "reset_window" | "unknown";
  start: string | null;
  end: string | null;
  known: boolean;
  timezone: string | null;
  resetEvidenceSource:
    | "provider_explicit"
    | "operator_explicit"
    | "reviewed_rule"
    | "provisional_assumption"
    | "unknown"
    | null;
}

export interface RawModelBreakdown {
  requested: Record<string, number>;
  recordedFinal: Record<string, number>;
  resolved: Record<string, number>;
}

export interface UsageUncertainty {
  unknownOwnership: number;
  unknownModel: number;
  unknownWindowMembership: number;
  ambiguousWindowMembership: number;
  quarantined: number;
  unresolvedIdentity: number;
  nonGenerationActivity: number;
}

export interface UsageExclusions {
  outOfWindow: number;
  nonChatSurface: number;
  excludedOrigin: number;
  rejectedBeforeStart: number;
}

export interface UsageActivityMetric {
  count: number;
  completed: number;
  uncertain: number;
  excluded: number;
}

export interface UsageActivity {
  last24h: UsageActivityMetric;
  last7d: UsageActivityMetric;
}

export interface UsageReport {
  snapshotVersion: typeof USAGE_REPORT_SNAPSHOT_VERSION;
  account: string;
  asOf: string;
  modelDimension: UsageModelDimension;
  window: UsageWindow;
  activityOnly: boolean;
  activity: UsageActivity;
  definite: {
    total: number;
    completed: number;
    byRequestedFamily: Record<string, number>;
    byRecordedFinalFamily: Record<string, number>;
    byResolvedFamily: Record<string, number>;
    byDimension: Record<string, number>;
    raw: RawModelBreakdown;
    modelMismatches: number;
  };
  uncertain: UsageUncertainty;
  excluded: UsageExclusions;
  quotaEstimate: QuotaEstimate | null;
  coverage: UsageCoverage;
}

const RESET_EVIDENCE_SOURCES = new Set([
  "provider_explicit",
  "operator_explicit",
  "reviewed_rule",
  "provisional_assumption",
  "unknown",
]);

export function buildUsageReport(
  snapshot: UsageReportSnapshot,
  request: UsageReportRequest,
): UsageReport {
  validateSnapshot(snapshot);
  validateRequest(snapshot, request);
  const asOfDate = new Date(request.asOf);
  const asOf = asOfDate.toISOString();
  const selected = selectedWindow(request, asOfDate);
  const quota = selected.window.known ? buildQuotaEstimate(snapshot) : null;

  const output: UsageReport = {
    snapshotVersion: USAGE_REPORT_SNAPSHOT_VERSION,
    account: request.account,
    asOf,
    modelDimension: request.modelDimension,
    window: selected.window,
    activityOnly: !selected.window.known,
    activity: { last24h: createActivityMetric(), last7d: createActivityMetric() },
    definite: {
      total: 0,
      completed: 0,
      byRequestedFamily: createCounts(),
      byRecordedFinalFamily: createCounts(),
      byResolvedFamily: createCounts(),
      byDimension: createCounts(),
      raw: {
        requested: createCounts(),
        recordedFinal: createCounts(),
        resolved: createCounts(),
      },
      modelMismatches: 0,
    },
    uncertain: {
      unknownOwnership: 0,
      unknownModel: 0,
      unknownWindowMembership: 0,
      ambiguousWindowMembership: 0,
      quarantined: 0,
      unresolvedIdentity: 0,
      nonGenerationActivity: 0,
    },
    excluded: {
      outOfWindow: 0,
      nonChatSurface: 0,
      excludedOrigin: 0,
      rejectedBeforeStart: 0,
    },
    quotaEstimate: quota,
    coverage: snapshot.coverage,
  };

  for (const attempt of snapshot.attempts) {
    validateAttemptScope(snapshot, attempt);
    const membership = attemptMembership(attempt, selected.window);
    if (membership === "out") {
      output.excluded.outOfWindow += 1;
      continue;
    }
    if (membership === "unknown") {
      output.uncertain.unknownWindowMembership += 1;
      continue;
    }
    if (membership === "ambiguous") {
      output.uncertain.ambiguousWindowMembership += 1;
      continue;
    }
    if (
      attempt.scope.providerUserId === null ||
      attempt.scope.workspaceId === null ||
      attempt.scope.quotaOwnerId === null
    ) {
      output.uncertain.unknownOwnership += 1;
      continue;
    }
    if (attempt.surface !== "chat") {
      output.excluded.nonChatSurface += 1;
      continue;
    }
    if (
      attempt.origin !== null &&
      ["shared", "imported", "copied"].includes(attempt.origin)
    ) {
      output.excluded.excludedOrigin += 1;
      continue;
    }
    if (attempt.quarantine?.state === "quarantined") {
      output.uncertain.quarantined += 1;
      continue;
    }
    if (
      attempt.quarantine === undefined ||
      attempt.identityBasis === "unresolved" ||
      attempt.warnings.includes("unresolved_linkage")
    ) {
      output.uncertain.unresolvedIdentity += 1;
      continue;
    }
    if (attempt.outcome === "rejected_before_start") {
      output.excluded.rejectedBeforeStart += 1;
      continue;
    }
    if (!attempt.generationStarted && !attempt.completedAnswer) {
      output.uncertain.nonGenerationActivity += 1;
      continue;
    }

    output.definite.total += 1;
    if (attempt.completedAnswer) {
      output.definite.completed += 1;
    }
    incrementKey(output.definite.byRequestedFamily, attempt.requestedFamily);
    incrementKey(
      output.definite.byRecordedFinalFamily,
      attempt.recordedFinalFamily,
    );
    incrementKey(output.definite.byResolvedFamily, attempt.resolvedFamily);
    incrementRaw(output.definite.raw.requested, attempt.requestedModelRaw);
    incrementRaw(
      output.definite.raw.recordedFinal,
      attempt.recordedFinalModelRaw,
    );
    incrementRaw(output.definite.raw.resolved, attempt.resolvedModelRaw);

    const primaryFamily = primaryFamilyFor(attempt, request.modelDimension);
    if (primaryFamily === null) {
      output.uncertain.unknownModel += 1;
    } else {
      incrementKey(output.definite.byDimension, primaryFamily);
    }
    if (
      attempt.requestedFamily !== null &&
      attempt.recordedFinalFamily !== null &&
      attempt.requestedFamily !== attempt.recordedFinalFamily
    ) {
      output.definite.modelMismatches += 1;
    }
  }

  addActivityMetrics(output.activity, snapshot.attempts, asOfDate);
  return output;
}

function createActivityMetric(): UsageActivityMetric {
  return { count: 0, completed: 0, uncertain: 0, excluded: 0 };
}

function addActivityMetrics(
  activity: UsageActivity,
  attempts: readonly ReconstructedAttempt[],
  asOf: Date,
): void {
  const asOfMs = asOf.getTime();
  const periods = [
    { metric: activity.last24h, durationMs: 24 * 60 * 60 * 1000 },
    { metric: activity.last7d, durationMs: 7 * 24 * 60 * 60 * 1000 },
  ];

  for (const attempt of attempts) {
    if (!attempt.generationStarted && !attempt.completedAnswer) {
      continue;
    }

    const attemptMs = new Date(attempt.attemptTime ?? "").getTime();
    if (!Number.isFinite(attemptMs)) {
      for (const { metric } of periods) {
        metric.uncertain += 1;
      }
      continue;
    }

    for (const { metric, durationMs } of periods) {
      if (attemptMs < asOfMs - durationMs || attemptMs > asOfMs) {
        metric.excluded += 1;
      } else {
        metric.count += 1;
        if (attempt.completedAnswer) {
          metric.completed += 1;
        }
      }
    }
  }
}

function validateSnapshot(snapshot: UsageReportSnapshot): void {
  if (!isRecord(snapshot)) {
    throw new TypeError("usage report snapshot must be an object");
  }
  if (snapshot.snapshotVersion !== USAGE_REPORT_SNAPSHOT_VERSION) {
    throw new TypeError("usage report snapshot version is unsupported");
  }
  if (
    typeof snapshot.collectorAccountId !== "string" ||
    !snapshot.collectorAccountId
  ) {
    throw new TypeError("usage report snapshot account is invalid");
  }
  if (!Array.isArray(snapshot.attempts)) {
    throw new TypeError("usage report snapshot attempts must be an array");
  }
  validateCoverage(snapshot.coverage);
}

function validateCoverage(coverage: UsageCoverage): void {
  if (!isRecord(coverage)) {
    throw new TypeError("usage report coverage must be an object");
  }
  for (const key of ["history", "overall", "projects", "branches"] as const) {
    if (!isCoverageLevel(coverage[key])) {
      throw new TypeError(`usage report coverage ${key} is invalid`);
    }
  }
  if (
    !Array.isArray(coverage.gaps) ||
    coverage.gaps.some((gap) => typeof gap !== "string")
  ) {
    throw new TypeError("usage report coverage gaps are invalid");
  }
}

function isCoverageLevel(value: unknown): value is CoverageLevel {
  return value === "complete" || value === "partial" || value === "unknown";
}

function validateRequest(
  snapshot: UsageReportSnapshot,
  request: UsageReportRequest,
): void {
  if (!isRecord(request)) {
    throw new TypeError("usage report request must be an object");
  }
  if (request.account !== snapshot.collectorAccountId) {
    throw new TypeError("usage report request account differs from snapshot");
  }
  const asOf = new Date(request.asOf);
  if (!Number.isFinite(asOf.getTime())) {
    throw new TypeError("usage report asOf must be a valid timestamp");
  }
  if (
    request.modelDimension !== "requested" &&
    request.modelDimension !== "recordedFinal" &&
    request.modelDimension !== "resolved"
  ) {
    throw new TypeError("usage report model dimension is invalid");
  }
  const selectors = [
    request.calendarDay !== undefined,
    request.elapsedLookbackMs !== undefined,
    request.resetWindow !== undefined,
  ];
  if (selectors.filter(Boolean).length !== 1) {
    throw new TypeError("usage report requires exactly one window selector");
  }
  if (request.calendarDay !== undefined) {
    const timezone = request.calendarDay.timezone;
    if (typeof timezone !== "string" || !validTimezone(timezone)) {
      throw new TypeError("usage report calendar timezone is invalid");
    }
  }
  if (
    request.elapsedLookbackMs !== undefined &&
    (!Number.isSafeInteger(request.elapsedLookbackMs) ||
      request.elapsedLookbackMs <= 0)
  ) {
    throw new TypeError("usage report elapsed lookback must be positive");
  }
  if (request.resetWindow !== undefined) {
    if (!Array.isArray(request.resetWindow.evidence)) {
      throw new TypeError("usage report reset evidence must be an array");
    }
    for (const evidence of request.resetWindow.evidence) {
      validateResetEvidence(evidence);
    }
  }
}

function selectedWindow(
  request: UsageReportRequest,
  asOf: Date,
): { window: UsageWindow } {
  if (request.calendarDay !== undefined) {
    const evaluated = evaluateResetWindow({
      rule: {
        type: "calendar",
        period: "day",
        timezone: request.calendarDay.timezone,
        windowId: "report_calendar_day",
      },
      asOf,
    });
    return {
      window: {
        kind: "calendar_day",
        start: evaluated.start,
        end: evaluated.end,
        known: evaluated.known,
        timezone: evaluated.timezone,
        resetEvidenceSource: null,
      },
    };
  }
  if (request.elapsedLookbackMs !== undefined) {
    const evaluated = evaluateResetWindow({
      rule: {
        type: "rolling_elapsed",
        durationMs: request.elapsedLookbackMs,
        windowId: "report_elapsed_lookback",
      },
      asOf,
    });
    return {
      window: {
        kind: "elapsed_lookback",
        start: evaluated.start,
        end: evaluated.end,
        known: evaluated.known,
        timezone: null,
        resetEvidenceSource: null,
      },
    };
  }

  const evidence = request.resetWindow?.evidence ?? [];
  const resolved = resolveResetWindow({ asOf, evidence });
  return {
    window: {
      kind: resolved.known ? "reset_window" : "unknown",
      start: resolved.start,
      end: resolved.end,
      known: resolved.known,
      timezone: resolved.timezone,
      resetEvidenceSource: RESET_EVIDENCE_SOURCES.has(resolved.evidenceSource)
        ? resolved.evidenceSource
        : "unknown",
    },
  };
}

function validateResetEvidence(evidence: ResetWindowEvidence): void {
  if (
    !isRecord(evidence) ||
    !isRecord(evidence.rule) ||
    typeof evidence.provenance !== "string"
  ) {
    throw new TypeError("usage report reset evidence is invalid");
  }
  if (!RESET_EVIDENCE_SOURCES.has(evidence.source)) {
    throw new TypeError("usage report reset evidence source is invalid");
  }
}

function attemptMembership(
  attempt: ReconstructedAttempt,
  window: UsageWindow,
): WindowMembership {
  if (!window.known || window.start === null || window.end === null) {
    return "unknown";
  }
  return intervalMembership({
    window: { start: window.start, end: window.end },
    time: {
      attemptTime: attempt.attemptTime,
      earliestPossibleAt: attempt.earliestPossibleAt,
      latestPossibleAt: attempt.latestPossibleAt,
    },
  });
}

function validateAttemptScope(
  snapshot: UsageReportSnapshot,
  attempt: ReconstructedAttempt,
): void {
  if (!isRecord(attempt) || !isRecord(attempt.scope)) {
    throw new TypeError("usage report attempt is invalid");
  }
  if (attempt.scope.collectorAccountId !== snapshot.collectorAccountId) {
    throw new TypeError("usage report snapshot contains mixed accounts");
  }
}

function buildQuotaEstimate(
  snapshot: UsageReportSnapshot,
): QuotaEstimate | null {
  const input = (
    snapshot as unknown as { quota?: QuotaEstimatorInput }
  ).quota;
  if (input === undefined) {
    return null;
  }
  const quotaInput = validateQuotaInput(input);
  const estimatorInput: QuotaEstimatorInput = {
    attempts: snapshot.attempts,
    ...(quotaInput.policy === undefined ? {} : { policy: quotaInput.policy }),
    windows: quotaInput.windows,
    coverage: quotaInput.coverage,
    ownership: quotaInput.ownership,
    ...(quotaInput.serverObservations === undefined
      ? {}
      : { serverObservations: quotaInput.serverObservations }),
    ...(quotaInput.uncertainAttemptMode === undefined
      ? {}
      : { uncertainAttemptMode: quotaInput.uncertainAttemptMode }),
  };
  return estimateQuota(estimatorInput);
}

function validateQuotaInput(
  input: QuotaEstimatorInput,
): QuotaEstimatorInput {
  if (!isRecord(input)) {
    throw new TypeError("usage report quota input is invalid");
  }
  if (!isRecord(input.policy)) {
    throw new TypeError("usage report quota policy is invalid");
  }
  if (
    !Array.isArray(input.windows) ||
    input.windows.some((window) => !isRecord(window))
  ) {
    throw new TypeError("usage report quota windows are invalid");
  }
  if (
    input.coverage !== "complete" &&
    input.coverage !== "partial" &&
    input.coverage !== "unknown"
  ) {
    throw new TypeError("usage report quota coverage is invalid");
  }
  if (!isRecord(input.ownership)) {
    throw new TypeError("usage report quota ownership is invalid");
  }
  return input;
}

function primaryFamilyFor(
  attempt: ReconstructedAttempt,
  dimension: UsageModelDimension,
): string | null {
  if (dimension === "requested") {
    return attempt.requestedFamily;
  }
  if (dimension === "recordedFinal") {
    return attempt.recordedFinalFamily;
  }
  return attempt.resolvedFamily;
}

function incrementKey(
  target: Record<string, number>,
  value: string | null,
): void {
  if (value === null) {
    return;
  }
  target[value] = (target[value] ?? 0) + 1;
}

function createCounts(): Record<string, number> {
  return Object.create(null);
}

function incrementRaw(
  target: Record<string, number>,
  value: string | null,
): void {
  incrementKey(target, value?.trim() || null);
}

function validTimezone(timezone: string): boolean {
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: timezone });
    return true;
  } catch {
    return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
