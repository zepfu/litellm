/**
 * Pure reset-window resolution and membership helpers.
 *
 * This module deliberately owns no quota arithmetic or persistence. Callers
 * provide all evidence and evaluate a window at an explicit as-of instant.
 */

export const WINDOW_TYPES = [
  "provider_explicit",
  "operator_explicit",
  "anchored_elapsed",
  "calendar",
  "rolling_elapsed",
  "unknown",
] as const;

export type WindowType = (typeof WINDOW_TYPES)[number];
export type InstantInput = string | Date;
export type CalendarPeriod = "day" | "week";
export type Weekday = 0 | 1 | 2 | 3 | 4 | 5 | 6;

export type WindowEvidenceSource =
  | "provider_explicit"
  | "operator_explicit"
  | "reviewed_rule"
  | "provisional_assumption"
  | "unknown";

/**
 * A rule is intentionally structural so the integration owner can map it to
 * configuration, provider observations, or a later persistence model.
 */
export interface ResetWindowRule {
  readonly type: WindowType;
  /** Explicit boundary or the anchored rule's legacy anchor field. */
  readonly start?: InstantInput | null;
  readonly end?: InstantInput | null;
  readonly anchor?: InstantInput | null;
  readonly durationMs?: number | null;
  readonly timezone?: string | null;
  /** Alias accepted for JavaScript callers; `timezone` is canonical. */
  readonly timeZone?: string | null;
  readonly period?: CalendarPeriod | null;
  /** Compatibility alias for policy documents that call this a period hint. */
  readonly documentedPeriodHint?: string | null;
  /** JavaScript weekday numbering: Sunday=0, Monday=1, ... Saturday=6. */
  readonly weekStartsOn?: Weekday;
  readonly windowId?: string | null;
}

/**
 * Evidence is retained even when it is not selected. Eligibility fields make
 * the evidence-strength requirements explicit instead of silently inferring
 * them from a timestamp or a field name.
 */
export interface ResetWindowEvidence {
  readonly source: WindowEvidenceSource;
  readonly rule: ResetWindowRule;
  readonly provenance: string;
  readonly observedAt?: InstantInput | null;
  /** Provider evidence must set this to true before it can win precedence. */
  readonly validated?: boolean;
  readonly current?: boolean;
  readonly reviewed?: boolean;
  readonly supportedByObservations?: boolean;
  readonly provisional?: boolean;
}

export interface EvaluateResetWindowInput {
  readonly rule: ResetWindowRule;
  readonly asOf: InstantInput;
  readonly eventTimes?: readonly InstantInput[];
}

export interface ResolveResetWindowInput {
  readonly asOf: InstantInput;
  readonly evidence?: readonly ResetWindowEvidence[];
  readonly eventTimes?: readonly InstantInput[];
}

export interface WindowBounds {
  readonly start: string | null;
  readonly end: string | null;
}

export interface EvaluatedResetWindow extends WindowBounds {
  readonly type: WindowType;
  readonly asOf: string;
  readonly anchor: string | null;
  readonly durationMs: number | null;
  readonly timezone: string | null;
  readonly period: CalendarPeriod | null;
  readonly weekStartsOn: Weekday | null;
  readonly windowId: string | null;
  readonly known: boolean;
  /**
   * For rolling windows, the earliest known event expiry at or after `asOf`.
   * It is null when no event expiry is known.
   */
  readonly nextExpiryAt: string | null;
}

export interface ResolvedResetWindow extends EvaluatedResetWindow {
  readonly evidenceSource: WindowEvidenceSource;
  readonly provenance: string | null;
  readonly selectedEvidence: ResetWindowEvidence | null;
  readonly evidence: readonly ResetWindowEvidence[];
}

export type WindowMembership = "in" | "out" | "ambiguous" | "unknown";

export interface WindowMembershipInput {
  readonly window: WindowBounds;
  readonly instant: InstantInput;
}

export interface TimeMembershipEvidence {
  readonly attemptTime?: InstantInput | null;
  readonly earliestPossibleAt?: InstantInput | null;
  readonly latestPossibleAt?: InstantInput | null;
}

export interface IntervalMembershipInput {
  readonly window: WindowBounds;
  readonly time: TimeMembershipEvidence;
}

export interface RollingExpiryInput {
  readonly eventTimes: readonly InstantInput[];
  readonly durationMs: number;
  readonly asOf: InstantInput;
}

const EVIDENCE_PRIORITY: Record<WindowEvidenceSource, number> = {
  provider_explicit: 4,
  operator_explicit: 3,
  reviewed_rule: 2,
  provisional_assumption: 1,
  unknown: 0,
};

const RECURRING_WINDOW_TYPES = new Set<WindowType>([
  "anchored_elapsed",
  "calendar",
  "rolling_elapsed",
]);

interface LocalDateTime {
  readonly year: number;
  readonly month: number;
  readonly day: number;
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
}

interface LocalDate {
  readonly year: number;
  readonly month: number;
  readonly day: number;
}

const formatterCache = new Map<string, Intl.DateTimeFormat>();

/**
 * Select the strongest justified evidence. Same-priority evidence uses the
 * newest observedAt, with input order as the deterministic tie-breaker.
 */
export function resolveResetWindow(
  input: ResolveResetWindowInput,
): ResolvedResetWindow {
  const evidence = [...(input.evidence ?? [])];
  const candidates = evidence
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => isEligibleEvidence(item))
    .sort((left, right) => {
      const priority =
        EVIDENCE_PRIORITY[right.item.source] -
        EVIDENCE_PRIORITY[left.item.source];
      if (priority !== 0) {
        return priority;
      }
      const rightObserved = observedAtMs(right.item);
      const leftObserved = observedAtMs(left.item);
      if (rightObserved !== leftObserved) {
        return rightObserved - leftObserved;
      }
      return left.index - right.index;
    });
  const selected = candidates[0]?.item ?? null;
  const evaluated = evaluateResetWindow({
    rule: selected?.rule ?? { type: "unknown" },
    asOf: input.asOf,
    ...(input.eventTimes === undefined
      ? {}
      : { eventTimes: input.eventTimes }),
  });
  return {
    ...evaluated,
    evidenceSource: selected?.source ?? "unknown",
    provenance: selected?.provenance ?? null,
    selectedEvidence: selected,
    evidence,
  };
}

/** Short alias for integrations that use the generic window terminology. */
export function resolveWindow(
  input: ResolveResetWindowInput,
): ResolvedResetWindow {
  return resolveResetWindow(input);
}

export function evaluateResetWindow(
  input: EvaluateResetWindowInput,
): EvaluatedResetWindow {
  const asOfDate = parseInstant(input.asOf, "asOf");
  const asOf = asOfDate.toISOString();
  const rule = input.rule;
  if (!isWindowType(rule.type)) {
    throw new Error(`unsupported window type: ${String(rule.type)}`);
  }

  let start: Date | null = null;
  let end: Date | null = null;
  let anchor: Date | null = null;
  let durationMs: number | null = null;
  let timezone: string | null = null;
  let period: CalendarPeriod | null = null;
  let weekStartsOn: Weekday | null = null;
  const windowId = nonEmptyString(rule.windowId);

  switch (rule.type) {
    case "provider_explicit":
      start = optionalInstant(rule.start, "window start");
      end = optionalInstant(rule.end, "window end");
      validateBounds(start, end);
      timezone = ruleTimezone(rule);
      break;
    case "operator_explicit":
      start = requiredInstant(rule.start, "operator window start");
      end = requiredInstant(rule.end, "operator window end");
      validateBounds(start, end);
      timezone = ruleTimezone(rule);
      break;
    case "anchored_elapsed": {
      anchor = requiredInstant(
        rule.anchor ?? rule.start,
        "anchored window UTC anchor",
      );
      durationMs = requiredDuration(rule.durationMs);
      const steps = Math.floor(
        (asOfDate.getTime() - anchor.getTime()) / durationMs,
      );
      start = new Date(anchor.getTime() + steps * durationMs);
      end = new Date(start.getTime() + durationMs);
      break;
    }
    case "calendar": {
      timezone = requiredTimezone(rule);
      period = calendarPeriod(rule);
      weekStartsOn = period === "week" ? weekStart(rule) : null;
      const localAsOf = localDateTime(asOfDate, timezone);
      const asOfDateOnly: LocalDate = {
        year: localAsOf.year,
        month: localAsOf.month,
        day: localAsOf.day,
      };
      const startDate =
        period === "week"
          ? addLocalDays(
              asOfDateOnly,
              -((dayOfWeek(asOfDateOnly) - weekStartsOn! + 7) % 7),
            )
          : asOfDateOnly;
      const endDate = addLocalDays(startDate, period === "week" ? 7 : 1);
      start = localMidnightUtc(startDate, timezone);
      end = localMidnightUtc(endDate, timezone);
      validateBounds(start, end);
      break;
    }
    case "rolling_elapsed":
      durationMs = requiredDuration(rule.durationMs);
      end = asOfDate;
      start = new Date(asOfDate.getTime() - durationMs);
      validateBounds(start, end);
      break;
    case "unknown":
      break;
  }

  return {
    type: rule.type,
    asOf,
    start: start?.toISOString() ?? null,
    end: end?.toISOString() ?? null,
    anchor: anchor?.toISOString() ?? null,
    durationMs,
    timezone,
    period,
    weekStartsOn,
    windowId,
    known: start !== null && end !== null,
    nextExpiryAt:
      rule.type === "rolling_elapsed" && durationMs !== null
        ? nextRollingExpiry({
            eventTimes: input.eventTimes ?? [],
            durationMs,
            asOf: asOfDate,
          })
        : null,
  };
}

/** Short alias for integrations that use the generic window terminology. */
export function evaluateWindow(
  input: EvaluateResetWindowInput,
): EvaluatedResetWindow {
  return evaluateResetWindow(input);
}

/**
 * Classify a point against [start, end). A missing bound prevents a definite
 * in/out result unless the point is provably outside the known bound.
 */
export function windowMembership(
  input: WindowMembershipInput,
): Exclude<WindowMembership, "ambiguous"> {
  const instant = parseInstant(input.instant, "membership instant");
  const { start, end } = parseBounds(input.window);
  if (start !== null && instant.getTime() < start.getTime()) {
    return "out";
  }
  if (end !== null && instant.getTime() >= end.getTime()) {
    return "out";
  }
  if (start === null || end === null) {
    return "unknown";
  }
  return "in";
}

/**
 * Classify bounded time evidence without assigning a straddling interval to a
 * window. Exact attempt time takes precedence over possible bounds.
 */
export function intervalMembership(
  input: IntervalMembershipInput,
): WindowMembership {
  const { start, end } = parseBounds(input.window);
  const exact = input.time.attemptTime;
  if (exact !== null && exact !== undefined) {
    return windowMembership({
      window: {
        start: start?.toISOString() ?? null,
        end: end?.toISOString() ?? null,
      },
      instant: exact,
    });
  }

  const earliest = optionalInstant(
    input.time.earliestPossibleAt,
    "earliest possible time",
  );
  const latest = optionalInstant(
    input.time.latestPossibleAt,
    "latest possible time",
  );
  if (earliest === null || latest === null) {
    return "unknown";
  }
  if (latest.getTime() < earliest.getTime()) {
    return "unknown";
  }
  if (start !== null && latest.getTime() < start.getTime()) {
    return "out";
  }
  if (end !== null && earliest.getTime() >= end.getTime()) {
    return "out";
  }
  if (
    start !== null &&
    end !== null &&
    earliest.getTime() >= start.getTime() &&
    latest.getTime() < end.getTime()
  ) {
    return "in";
  }
  if (start !== null && end !== null) {
    return "ambiguous";
  }
  return "unknown";
}

/**
 * Return the next expiry of an event currently inside a rolling
 * [asOf-duration, asOf) window. An event exactly at the lower bound expires
 * at `asOf`, which is retained as the next boundary.
 */
export function nextRollingExpiry(input: RollingExpiryInput): string | null {
  const asOf = parseInstant(input.asOf, "rolling asOf");
  const durationMs = requiredDuration(input.durationMs);
  const windowStart = asOf.getTime() - durationMs;
  let nextExpiry: number | null = null;

  for (const eventTime of input.eventTimes) {
    const event = parseInstant(eventTime, "rolling event time");
    const eventMs = event.getTime();
    if (eventMs < windowStart || eventMs >= asOf.getTime()) {
      continue;
    }
    const expiry = eventMs + durationMs;
    if (expiry >= asOf.getTime() && (nextExpiry === null || expiry < nextExpiry)) {
      nextExpiry = expiry;
    }
  }
  return nextExpiry === null ? null : new Date(nextExpiry).toISOString();
}

function isEligibleEvidence(evidence: ResetWindowEvidence): boolean {
  const rule = evidence.rule;
  if (evidence.current === false || !isNonEmptyString(evidence.provenance)) {
    return false;
  }
  switch (evidence.source) {
    case "provider_explicit":
      return (
        evidence.validated === true &&
        rule.type === "provider_explicit" &&
        (rule.start != null ||
          rule.end != null ||
          isNonEmptyString(rule.windowId))
      );
    case "operator_explicit":
      return (
        evidence.validated !== false &&
        rule.type === "operator_explicit" &&
        rule.start != null &&
        rule.end != null
      );
    case "reviewed_rule":
      return (
        evidence.validated !== false &&
        evidence.reviewed === true &&
        evidence.supportedByObservations === true &&
        RECURRING_WINDOW_TYPES.has(rule.type)
      );
    case "provisional_assumption":
      return (
        evidence.validated !== false &&
        evidence.provisional !== false &&
        RECURRING_WINDOW_TYPES.has(rule.type)
      );
    case "unknown":
      return rule.type === "unknown";
  }
}

function observedAtMs(evidence: ResetWindowEvidence): number {
  if (evidence.observedAt === null || evidence.observedAt === undefined) {
    return Number.NEGATIVE_INFINITY;
  }
  return parseInstant(evidence.observedAt, "window evidence observedAt").getTime();
}

function isWindowType(value: unknown): value is WindowType {
  return (
    typeof value === "string" &&
    (WINDOW_TYPES as readonly string[]).includes(value)
  );
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function nonEmptyString(value: string | null | undefined): string | null {
  return isNonEmptyString(value) ? value.trim() : null;
}

function parseInstant(value: InstantInput, label: string): Date {
  const date =
    value instanceof Date ? new Date(value.getTime()) : new Date(value);
  if (!Number.isFinite(date.getTime())) {
    throw new Error(`${label} must be a valid ISO timestamp`);
  }
  return date;
}

function optionalInstant(
  value: InstantInput | null | undefined,
  label: string,
): Date | null {
  return value === null || value === undefined
    ? null
    : parseInstant(value, label);
}

function requiredInstant(
  value: InstantInput | null | undefined,
  label: string,
): Date {
  const parsed = optionalInstant(value, label);
  if (parsed === null) {
    throw new Error(`${label} is required`);
  }
  return parsed;
}

function requiredDuration(value: number | null | undefined): number {
  if (
    value === null ||
    value === undefined ||
    !Number.isSafeInteger(value) ||
    value <= 0
  ) {
    throw new Error("durationMs must be a positive safe integer");
  }
  return value;
}

function validateBounds(start: Date | null, end: Date | null): void {
  if (start !== null && end !== null && start.getTime() >= end.getTime()) {
    throw new Error("window start must be before its exclusive end");
  }
}

function parseBounds(window: WindowBounds): {
  readonly start: Date | null;
  readonly end: Date | null;
} {
  const start = optionalInstant(window.start, "window start");
  const end = optionalInstant(window.end, "window end");
  validateBounds(start, end);
  return { start, end };
}

function ruleTimezone(rule: ResetWindowRule): string | null {
  if (
    rule.timezone !== null &&
    rule.timezone !== undefined &&
    rule.timeZone !== null &&
    rule.timeZone !== undefined &&
    rule.timezone !== rule.timeZone
  ) {
    throw new Error("timezone and timeZone disagree");
  }
  const timezone = rule.timezone ?? rule.timeZone ?? null;
  if (timezone !== null && !isNonEmptyString(timezone)) {
    throw new Error("timezone must be a non-empty IANA timezone");
  }
  return timezone;
}

function requiredTimezone(rule: ResetWindowRule): string {
  const timezone = ruleTimezone(rule);
  if (timezone === null) {
    throw new Error("calendar windows require an explicit timezone");
  }
  getFormatter(timezone);
  return timezone;
}

function calendarPeriod(rule: ResetWindowRule): CalendarPeriod {
  const raw = rule.period ?? rule.documentedPeriodHint ?? "day";
  const normalized = String(raw).toLowerCase();
  if (normalized === "day" || normalized === "daily") {
    return "day";
  }
  if (normalized === "week" || normalized === "weekly") {
    return "week";
  }
  throw new Error(`unsupported calendar period: ${String(raw)}`);
}

function weekStart(rule: ResetWindowRule): Weekday {
  const value = rule.weekStartsOn ?? 1;
  if (!Number.isInteger(value) || value < 0 || value > 6) {
    throw new Error("weekStartsOn must be an integer from 0 through 6");
  }
  return value as Weekday;
}

function getFormatter(timezone: string): Intl.DateTimeFormat {
  const cached = formatterCache.get(timezone);
  if (cached !== undefined) {
    return cached;
  }
  let formatter: Intl.DateTimeFormat;
  try {
    formatter = new Intl.DateTimeFormat("en-US", {
      calendar: "gregory",
      day: "2-digit",
      hour: "2-digit",
      hourCycle: "h23",
      minute: "2-digit",
      month: "2-digit",
      numberingSystem: "latn",
      second: "2-digit",
      timeZone: timezone,
      year: "numeric",
    });
  } catch (error) {
    throw new Error(
      `invalid timezone '${timezone}': ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
  formatterCache.set(timezone, formatter);
  return formatter;
}

function localDateTime(date: Date, timezone: string): LocalDateTime {
  const parts = getFormatter(timezone).formatToParts(date);
  const values = new Map(parts.map((part) => [part.type, part.value]));
  const year = numericPart(values, "year", timezone);
  const month = numericPart(values, "month", timezone);
  const day = numericPart(values, "day", timezone);
  const hour = numericPart(values, "hour", timezone);
  const minute = numericPart(values, "minute", timezone);
  const second = numericPart(values, "second", timezone);
  return { year, month, day, hour, minute, second };
}

function numericPart(
  values: Map<string, string>,
  name: string,
  timezone: string,
): number {
  const raw = values.get(name);
  const value = raw === undefined ? Number.NaN : Number(raw);
  if (!Number.isInteger(value)) {
    throw new Error(`timezone '${timezone}' did not provide a valid ${name}`);
  }
  return value;
}

function localDateToUtcMilliseconds(
  value: LocalDateTime,
): number {
  const date = new Date(0);
  date.setUTCFullYear(value.year, value.month - 1, value.day);
  date.setUTCHours(value.hour, value.minute, value.second, 0);
  return date.getTime();
}

function localMidnightUtc(value: LocalDate, timezone: string): Date {
  const wallClock = localDateToUtcMilliseconds({
    ...value,
    hour: 0,
    minute: 0,
    second: 0,
  });
  let candidate = wallClock;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const offset = timezoneOffsetMs(new Date(candidate), timezone);
    const next = wallClock - offset;
    if (next === candidate) {
      break;
    }
    candidate = next;
  }
  const resolved = new Date(candidate);
  const actual = localDateTime(resolved, timezone);
  if (
    actual.year !== value.year ||
    actual.month !== value.month ||
    actual.day !== value.day ||
    actual.hour !== 0 ||
    actual.minute !== 0 ||
    actual.second !== 0
  ) {
    throw new Error(
      `timezone '${timezone}' has no representable local midnight for ` +
        `${String(value.year).padStart(4, "0")}-${String(value.month).padStart(
          2,
          "0",
        )}-${String(value.day).padStart(2, "0")}`,
    );
  }
  return resolved;
}

function timezoneOffsetMs(date: Date, timezone: string): number {
  const parts = localDateTime(date, timezone);
  const localAsUtc = localDateToUtcMilliseconds(parts) + date.getUTCMilliseconds();
  return localAsUtc - date.getTime();
}

function addLocalDays(value: LocalDate, days: number): LocalDate {
  const date = new Date(0);
  date.setUTCFullYear(value.year, value.month - 1, value.day);
  date.setUTCHours(0, 0, 0, 0);
  date.setUTCDate(date.getUTCDate() + days);
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
  };
}

function dayOfWeek(value: LocalDate): Weekday {
  const date = new Date(0);
  date.setUTCFullYear(value.year, value.month - 1, value.day);
  date.setUTCHours(0, 0, 0, 0);
  return date.getUTCDay() as Weekday;
}
