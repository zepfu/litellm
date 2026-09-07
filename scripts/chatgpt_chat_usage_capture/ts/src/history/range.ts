import {
  DEFAULT_BACKFILL_DAYS,
  type HistoryCollectionMode,
  type HistoryRange,
} from "../contracts/history.js";

const DURATION_RE = /^(\d+(?:\.\d+)?)([dhm])$/i;

export function defaultBackfillRange(
  now: Date = new Date(),
  days = DEFAULT_BACKFILL_DAYS,
): HistoryRange {
  if (!Number.isFinite(days) || days <= 0) {
    throw new Error("backfill days must be a positive finite number");
  }
  return makeRange(new Date(now.getTime() - days * 24 * 60 * 60 * 1000), now);
}

export function parseDuration(value: string): number {
  const match = DURATION_RE.exec(value.trim());
  if (!match) {
    throw new Error(`unsupported duration '${value}'; use values such as 14d or 48h`);
  }
  const amount = Number(match[1]);
  const unit = match[2]!.toLowerCase();
  const multiplier =
    unit === "d"
      ? 24 * 60 * 60 * 1000
      : unit === "h"
        ? 60 * 60 * 1000
        : 60 * 1000;
  const milliseconds = amount * multiplier;
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) {
    throw new Error(`duration must be positive: '${value}'`);
  }
  return milliseconds;
}

export function parseInstant(value: string | Date): Date {
  if (value instanceof Date) {
    if (!Number.isFinite(value.getTime())) {
      throw new Error("invalid date");
    }
    return new Date(value.getTime());
  }
  const duration = DURATION_RE.test(value.trim());
  if (duration) {
    throw new Error(`duration '${value}' needs an explicit reference time`);
  }
  const instant = new Date(value);
  if (!Number.isFinite(instant.getTime())) {
    throw new Error(`invalid date '${value}'`);
  }
  return instant;
}

export function makeRange(start: Date, end: Date): HistoryRange {
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime())) {
    throw new Error("history range contains an invalid date");
  }
  if (start.getTime() >= end.getTime()) {
    throw new Error("history range start must be before its exclusive end");
  }
  return {
    start: start.toISOString(),
    end: end.toISOString(),
  };
}

export function parseExplicitRange(
  start: string | Date,
  end: string | Date,
): HistoryRange {
  return makeRange(parseInstant(start), parseInstant(end));
}

export function resolveRequestedRange(options: {
  mode: HistoryCollectionMode;
  now: Date;
  range?: HistoryRange;
  defaultBackfillDays?: number;
}): HistoryRange {
  if (options.range) {
    return validateRange(options.range);
  }
  if (options.mode === "reconciliation") {
    throw new Error("reconciliation requires an explicit range");
  }
  return defaultBackfillRange(
    options.now,
    options.defaultBackfillDays ?? DEFAULT_BACKFILL_DAYS,
  );
}

export function validateRange(range: HistoryRange): HistoryRange {
  const start = parseInstant(range.start);
  const end = parseInstant(range.end);
  return makeRange(start, end);
}

export function clampStart(
  range: HistoryRange,
  candidateCutoff: string,
): HistoryRange {
  const rangeStart = parseInstant(range.start);
  const cutoff = parseInstant(candidateCutoff);
  if (cutoff.getTime() <= rangeStart.getTime()) {
    return range;
  }
  return makeRange(cutoff, parseInstant(range.end));
}
