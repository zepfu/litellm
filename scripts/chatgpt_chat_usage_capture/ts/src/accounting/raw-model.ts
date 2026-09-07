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
  includedAttempts: number;
  ambiguousTimeAttempts: number;
  unknownTimeAttempts: number;
  excludedSurfaceAttempts: number;
  excludedOriginAttempts: number;
  unclassifiedAttempts: number;
  attemptIds: string[];
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
  const attemptIds: string[] = [];
  let modelMismatches = 0;
  let ambiguousTimeAttempts = 0;
  let unknownTimeAttempts = 0;
  let excludedSurfaceAttempts = 0;
  let excludedOriginAttempts = 0;
  let unclassifiedAttempts = 0;

  for (const attempt of ledger.listAttempts(scope)) {
    const membership = membershipFor(attempt, startDate, endDate);
    if (membership === "out") {
      continue;
    }
    if (membership === "ambiguous") {
      ambiguousTimeAttempts += 1;
      unclassifiedAttempts += 1;
    }
    if (membership === "unknown") {
      unknownTimeAttempts += 1;
      unclassifiedAttempts += 1;
    }
    attemptIds.push(String(attempt.attemptId));
    if (String(attempt.surface) !== "chat") {
      excludedSurfaceAttempts += 1;
      unclassifiedAttempts += 1;
      continue;
    }
    if (["shared", "imported", "copied"].includes(String(attempt.origin))) {
      excludedOriginAttempts += 1;
      unclassifiedAttempts += 1;
      continue;
    }

    increment(requested, rawOrUnknown(attempt.requestedModelRaw));
    if (attempt.completedAnswer) {
      increment(recorded, rawOrUnknown(attempt.recordedFinalModelRaw));
    }
    increment(resolved, rawOrUnknown(attempt.resolvedModelRaw));
    if (
      attempt.requestedModelRaw !== null &&
      attempt.recordedFinalModelRaw !== null &&
      attempt.requestedModelRaw !== attempt.recordedFinalModelRaw
    ) {
      modelMismatches += 1;
    }
    if (
      attempt.requestedModelRaw === null &&
      attempt.recordedFinalModelRaw === null &&
      attempt.resolvedModelRaw === null
    ) {
      unclassifiedAttempts += 1;
    }
  }

  attemptIds.sort();
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
    includedAttempts: attemptIds.length,
    ambiguousTimeAttempts,
    unknownTimeAttempts,
    excludedSurfaceAttempts,
    excludedOriginAttempts,
    unclassifiedAttempts,
    attemptIds,
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
  if (latest && latest <= start) {
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

function rawOrUnknown(value: unknown): string {
  return typeof value === "string" && value.length > 0 ? value : "unknown";
}

function increment(map: Map<string, number>, key: string): void {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function sortedCounts(map: Map<string, number>): Record<string, number> {
  return Object.fromEntries([...map.entries()].sort(([left], [right]) => left.localeCompare(right)));
}
