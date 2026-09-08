import { parseRefreshInterval } from "./interval.js";
import {
  DEFAULT_JITTER_SECONDS,
  DEFAULT_REFRESH_INTERVAL,
  SchedulerTransitionError,
  validateEpoch,
  validateScope,
} from "./types.js";
import type {
  ActiveTrigger,
  CompleteTriggerRequest,
  RefreshRequestResult,
  ScheduleContext,
  ScheduleMutationResult,
  ScheduleOptions,
  SchedulePending,
  ScheduleScope,
  ScheduleState,
  TriggerClaim,
  TriggerKind,
} from "./types.js";

const MAX_FAILURE_STREAK = 1_000_000;

export function createScheduleState(
  scope: ScheduleScope,
  options: ScheduleOptions,
  context: ScheduleContext,
  stateVersion: number,
): ScheduleState {
  validateScope(scope);
  validateEpoch(context.at, "at");
  const interval = parseRefreshInterval(options.interval ?? DEFAULT_REFRESH_INTERVAL);
  const jitterSeconds = validateJitterSeconds(options.jitterSeconds ?? DEFAULT_JITTER_SECONDS);
  const anchorAt = validateEpoch(options.anchorAt ?? context.at, "anchorAt");
  const jitterMs = sampleJitter(jitterSeconds);
  return {
    scope,
    stateVersion,
    interval: interval.spec,
    intervalMs: interval.milliseconds,
    anchorAt,
    jitterSeconds,
    jitterMs,
    nextTickIndex: 1,
    nextDueAt: scheduledDueAt(anchorAt, interval.milliseconds, 1, jitterMs),
    pending: null,
    active: null,
    retryNotBefore: null,
    failureStreak: 0,
    authPausedUntil: null,
    lastTriggerAt: null,
    lastCompletedAt: null,
  };
}

export function requestRefresh(
  state: ScheduleState | null,
  context: ScheduleContext,
): RefreshRequestResult {
  const current = state ?? createScheduleState(
    { collectorAccountId: "pending", profileId: "pending" },
    {},
    context,
    0,
  );
  if (current.active !== null) {
    return { queued: true, coalesced: true, state: current };
  }
  const pending = combinePending(current.pending, { kind: "manual", missedCount: 0 });
  return {
    queued: true,
    coalesced: state !== null && current.pending !== null,
    state: {
      ...current,
      pending: {
        kind: pending.kind,
        missedCount: pending.missedCount,
        requestedAt: current.pending?.requestedAt ?? context.at,
      },
    },
  };
}

export function claimTrigger(
  state: ScheduleState,
  context: ScheduleContext,
  fencingToken: number,
): TriggerClaim | null {
  validateScope(state.scope);
  validateEpoch(context.at, "at");
  validateNonNegativeInteger(fencingToken, "fencingToken");
  if ((state.authPausedUntil !== null && state.authPausedUntil > context.at) ||
      (state.retryNotBefore !== null && state.retryNotBefore > context.at)) {
    return null;
  }
  if (state.active !== null) {
    return { trigger: state.active, resumed: true, state };
  }

  const dueCount = countDueTicks(state, context.at);
  if (!state.pending && dueCount === 0) {
    return null;
  }
  const trigger: ActiveTrigger = {
    triggerId: `scheduled-${state.nextTickIndex}-${context.at}`,
    kind: triggerKindForClaim(state.pending, dueCount),
    missedCount: (state.pending?.missedCount ?? 0) + dueCount,
    dueAt: dueCount > 0 ? state.nextDueAt : null,
    jitterMs: state.jitterMs,
    claimedAt: context.at,
    fencingToken,
  };
  const nextTickIndex = dueCount > 0
    ? state.nextTickIndex + dueCount
    : state.nextTickIndex;
  return {
    trigger,
    resumed: false,
    state: {
      ...state,
      nextTickIndex,
      nextDueAt:
        dueCount > 0
          ? scheduledDueAt(state.anchorAt, state.intervalMs, nextTickIndex, state.jitterMs)
          : state.nextDueAt,
      pending: null,
      active: trigger,
      lastTriggerAt: context.at,
    },
  };
}

export function completeTrigger(
  state: ScheduleState | null,
  context: ScheduleContext,
  fencingToken: number,
  request: CompleteTriggerRequest,
): ScheduleMutationResult {
  validateEpoch(context.at, "at");
  validateNonNegativeInteger(fencingToken, "fencingToken");
  if (state === null || state.active === null || state.active.fencingToken !== fencingToken) {
    return { applied: false, state };
  }

  let failureStreak = state.failureStreak;
  let retryNotBefore = state.retryNotBefore;
  let authPausedUntil = state.authPausedUntil;
  if (request.outcome === "success") {
    retryNotBefore = null;
    failureStreak = 0;
    authPausedUntil = null;
  } else {
    failureStreak = Math.min(MAX_FAILURE_STREAK, failureStreak + 1);
    retryNotBefore = context.at + retryBackoffMs(failureStreak);
  }
  if (request.outcome === "authentication") {
    authPausedUntil = context.at + authPauseMs(request.retryAfterMs, failureStreak);
  }

  return {
    applied: true,
    state: {
      ...state,
      active: null,
      retryNotBefore,
      failureStreak,
      authPausedUntil,
      lastCompletedAt: context.at,
    },
  };
}

export function materializeDue(
  state: ScheduleState,
  context: ScheduleContext,
): ScheduleState {
  const dueCount = countDueTicks(state, context.at);
  if (dueCount === 0) {
    return state;
  }
  const pending = combinePending(state.pending, { kind: "scheduled", missedCount: dueCount });
  const nextTickIndex = state.nextTickIndex + dueCount;
  return {
    ...state,
    nextTickIndex,
    nextDueAt: scheduledDueAt(state.anchorAt, state.intervalMs, nextTickIndex, state.jitterMs),
    pending: {
      kind: pending.kind,
      missedCount: pending.missedCount,
      requestedAt: state.pending?.requestedAt ?? context.at,
    },
  };
}

function combinePending(
  current: SchedulePending | null,
  incoming: Pick<SchedulePending, "kind" | "missedCount">,
): SchedulePending {
  return {
    kind: current === null || current.kind === incoming.kind ? incoming.kind : "coalesced",
    missedCount: (current?.missedCount ?? 0) + incoming.missedCount,
    requestedAt: current?.requestedAt ?? 0,
  };
}

function triggerKindForClaim(
  pending: SchedulePending | null,
  dueCount: number,
): TriggerKind {
  if (!pending || (pending.missedCount === 0 && dueCount === 0)) {
    return pending?.kind ?? "scheduled";
  }
  return "coalesced";
}

function countDueTicks(state: ScheduleState, now: number): number {
  if (state.nextDueAt > now) {
    return 0;
  }
  return Math.floor((now - state.nextDueAt) / state.intervalMs) + 1;
}

function scheduledDueAt(
  anchorAt: number,
  intervalMs: number,
  tickIndex: number,
  jitterMs: number,
): number {
  const dueAt = anchorAt + intervalMs * tickIndex + jitterMs;
  if (!Number.isSafeInteger(dueAt)) {
    throw new SchedulerTransitionError(
      "invalid_schedule_state",
      "scheduler due time exceeds safe integer range",
    );
  }
  return dueAt;
}

export function sampleJitter(jitterSeconds: number, random: () => number = Math.random): number {
  const limitMs = jitterSeconds * 1_000;
  if (limitMs === 0) {
    return 0;
  }
  const bounded = Math.min(1, Math.max(0, random()));
  if (!Number.isFinite(bounded)) {
    throw new SchedulerTransitionError("invalid_request", "jitter source returned a non-finite value");
  }
  return Math.min(limitMs, Math.floor(bounded * (limitMs + 1)));
}

export function validateJitterSeconds(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 60) {
    throw new SchedulerTransitionError(
      "invalid_request",
      "jitterSeconds must be an integer from 0 through 60",
    );
  }
  return value;
}

function retryBackoffMs(failureStreak: number): number {
  return Math.min(300_000, 1_000 * 2 ** Math.min(6, failureStreak));
}

function authPauseMs(retryAfterMs: number | undefined, failureStreak: number): number {
  const fallback = Math.min(600_000, 30_000 * 2 ** Math.min(4, failureStreak));
  if (retryAfterMs === undefined) {
    return fallback;
  }
  return Math.max(1_000, Math.min(600_000, validateNonNegativeInteger(retryAfterMs, "retryAfterMs")));
}

function validateNonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SchedulerTransitionError(
      "invalid_request",
      `${label} must be a non-negative safe integer`,
    );
  }
  return value;
}
