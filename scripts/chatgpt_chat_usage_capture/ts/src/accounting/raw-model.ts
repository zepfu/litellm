import type { Ledger } from "../ledger/store.js";
import { scopeKey } from "../ledger/identity.js";
import type { LedgerScope } from "../ledger/types.js";

export interface RawModelReport {
  accountId: string;
  scopeKey: string;
  surface: string;
  durationMs: number;
  start: string;
  end: string;
  observedAttemptsByRequestedModel: Record<string, number>;
  completedAnswersByRecordedFinalModel: Record<string, number>;
  observedAttemptsByResolvedModel: Record<string, number>;
  modelMismatches: number;
  rawSlugDifferences: number;
  includedAttempts: number;
  possibleAttemptIds: string[];
  possibleAttemptsByRequestedModel: Record<string, number>;
  possibleCompletedAnswersByRecordedFinalModel: Record<string, number>;
  possibleAttemptsByResolvedModel: Record<string, number>;
  ambiguousTimeAttempts: number;
  unknownTimeAttempts: number;
  unresolvedAttempts: number;
  unknownModelAttempts: number;
  excludedSurfaceAttempts: number;
  excludedOriginAttempts: number;
  unclassifiedAttempts: number;
  attemptIds: string[];
  coverageGaps: Array<Record<string, unknown>>;
  label: string;
}

export function buildRawModelReport(
  ledger: Ledger,
  scope: LedgerScope,
  options: { durationMs: number; now?: string },
): RawModelReport {
  if (!Number.isFinite(options.durationMs) || options.durationMs <= 0) {
    throw new Error("durationMs must be a positive finite number");
  }
  const endDate = options.now ? new Date(options.now) : new Date();
  if (!Number.isFinite(endDate.getTime())) {
    throw new Error("report now must be a valid ISO timestamp");
  }
  const startDate = new Date(endDate.getTime() - options.durationMs);
  const start = startDate.toISOString();
  const end = endDate.toISOString();
  const requested = new Map<string, number>();
  const recorded = new Map<string, number>();
  const resolved = new Map<string, number>();
  const possibleRequested = new Map<string, number>();
  const possibleRecorded = new Map<string, number>();
  const possibleResolved = new Map<string, number>();
  const attemptIds: string[] = [];
  const possibleAttemptIds: string[] = [];
  const unclassifiedAttemptIds = new Set<string>();
  let modelMismatches = 0;
  let rawSlugDifferences = 0;
  let ambiguousTimeAttempts = 0;
  let unknownTimeAttempts = 0;
  let unresolvedAttempts = 0;
  let unknownModelAttempts = 0;
  let excludedSurfaceAttempts = 0;
  let excludedOriginAttempts = 0;

  for (const attempt of ledger.listAttempts(scope)) {
    const attemptId = String(attempt.attemptId);
    const membership = membershipFor(attempt, startDate, endDate);
    if (membership === "out") {
      continue;
    }
    attemptIds.push(attemptId);

    const unresolved = isUnresolvedFragment(attempt);
    const hasRawModelEvidence = hasAnyRawModelEvidence(attempt);
    const excludedSurface = String(attempt.surface) !== "chat";
    const excludedOrigin = ["shared", "imported", "copied"].includes(String(attempt.origin));
    const possibleMembership = membership !== "in" || unresolved;

    if (membership === "ambiguous") {
      ambiguousTimeAttempts += 1;
      unclassifiedAttemptIds.add(attemptId);
    }
    if (membership === "unknown") {
      unknownTimeAttempts += 1;
      unclassifiedAttemptIds.add(attemptId);
    }
    if (unresolved) {
      unresolvedAttempts += 1;
      unclassifiedAttemptIds.add(attemptId);
    }
    if (!hasRawModelEvidence) {
      unknownModelAttempts += 1;
      unclassifiedAttemptIds.add(attemptId);
    }
    if (excludedSurface) {
      excludedSurfaceAttempts += 1;
      unclassifiedAttemptIds.add(attemptId);
    }
    if (excludedOrigin) {
      excludedOriginAttempts += 1;
      unclassifiedAttemptIds.add(attemptId);
    }

    if (possibleMembership) {
      possibleAttemptIds.push(attemptId);
      if (!excludedSurface && !excludedOrigin) {
        incrementIfPresent(possibleRequested, attempt.requestedModelRaw);
        if (attempt.completedAnswer) {
          incrementIfPresent(possibleRecorded, attempt.recordedFinalModelRaw);
        }
        incrementIfPresent(possibleResolved, attempt.resolvedModelRaw);
      }
    }

    if (
      membership !== "in" ||
      unresolved ||
      excludedSurface ||
      excludedOrigin ||
      !hasRawModelEvidence
    ) {
      continue;
    }

    incrementIfPresent(requested, attempt.requestedModelRaw);
    if (attempt.completedAnswer) {
      incrementIfPresent(recorded, attempt.recordedFinalModelRaw);
    }
    incrementIfPresent(resolved, attempt.resolvedModelRaw);

    const requestedFamily = nonEmptyString(attempt.requestedFamily);
    const recordedFinalFamily = nonEmptyString(attempt.recordedFinalFamily);
    if (
      requestedFamily !== null &&
      recordedFinalFamily !== null &&
      requestedFamily !== recordedFinalFamily
    ) {
      modelMismatches += 1;
    }
    if (
      nonEmptyString(attempt.requestedModelRaw) !== null &&
      nonEmptyString(attempt.recordedFinalModelRaw) !== null &&
      attempt.requestedModelRaw !== attempt.recordedFinalModelRaw
    ) {
      rawSlugDifferences += 1;
    }
  }

  attemptIds.sort();
  possibleAttemptIds.sort();
  return {
    accountId: scope.collectorAccountId,
    scopeKey: scopeKey(scope),
    surface: scope.surface,
    durationMs: options.durationMs,
    start,
    end,
    observedAttemptsByRequestedModel: sortedCounts(requested),
    completedAnswersByRecordedFinalModel: sortedCounts(recorded),
    observedAttemptsByResolvedModel: sortedCounts(resolved),
    modelMismatches,
    rawSlugDifferences,
    includedAttempts: attemptIds.length,
    possibleAttemptIds,
    possibleAttemptsByRequestedModel: sortedCounts(possibleRequested),
    possibleCompletedAnswersByRecordedFinalModel: sortedCounts(possibleRecorded),
    possibleAttemptsByResolvedModel: sortedCounts(possibleResolved),
    ambiguousTimeAttempts,
    unknownTimeAttempts,
    unresolvedAttempts,
    unknownModelAttempts,
    excludedSurfaceAttempts,
    excludedOriginAttempts,
    unclassifiedAttempts: unclassifiedAttemptIds.size,
    attemptIds,
    coverageGaps: ledger.coverageGaps(scope).filter((gap) => gap.state === "open"),
    label: "Observed raw-model activity; not an official provider quota balance",
  };
}

function membershipFor(
  attempt: Record<string, unknown>,
  start: Date,
  end: Date,
): "in" | "out" | "ambiguous" | "unknown" {
  const exact = dateOrNull(attempt.attemptTime);
  if (exact) {
    return exact >= start && exact < end ? "in" : "out";
  }
  const earliest = dateOrNull(attempt.earliestPossibleAt);
  const latest = dateOrNull(attempt.latestPossibleAt);
  if (!earliest && !latest) {
    return "unknown";
  }
  if (earliest && latest && earliest.getTime() === latest.getTime()) {
    return earliest >= start && earliest < end ? "in" : "out";
  }
  if (latest && latest < start) {
    return "out";
  }
  if (earliest && earliest >= end) {
    return "out";
  }
  if (earliest && latest && earliest >= start && latest < end) {
    return "in";
  }
  return "ambiguous";
}

function dateOrNull(value: unknown): Date | null {
  if (typeof value !== "string") {
    return null;
  }
  const parsed = new Date(value);
  return Number.isFinite(parsed.getTime()) ? parsed : null;
}

function isUnresolvedFragment(attempt: Record<string, unknown>): boolean {
  return (
    attempt.identityBasis === "unresolved" ||
    attempt.outcome === "unresolved"
  );
}

function hasAnyRawModelEvidence(attempt: Record<string, unknown>): boolean {
  return (
    nonEmptyString(attempt.requestedModelRaw) !== null ||
    nonEmptyString(attempt.recordedFinalModelRaw) !== null ||
    nonEmptyString(attempt.resolvedModelRaw) !== null
  );
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function incrementIfPresent(map: Map<string, number>, value: unknown): void {
  const key = nonEmptyString(value);
  if (key !== null) {
    increment(map, key);
  }
}

function increment(map: Map<string, number>, key: string): void {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function sortedCounts(map: Map<string, number>): Record<string, number> {
  return Object.fromEntries([...map.entries()].sort(([left], [right]) => left.localeCompare(right)));
}
