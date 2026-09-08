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
    serverRetryNotBefore: null,
    failureStreak: 0,
    authPausedUntil: null,
    lastTriggerAt: null,
    lastCompletedAt: null,
  };
}

export function requestRefresh(
  state: ScheduleState,
  context: ScheduleContext,
): RefreshRequestResult {
  validateScope(state.scope);
  validateEpoch(context.at, "at");
  const pending = combinePending(state.pending, { kind: "manual", missedCount: 0 });
  return {
    queued: true,
    coalesced: state.active !== null || state.pending !== null,
    state: withPending(state, pending, state.pending?.requestedAt ?? context.at),
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
  if (state.authPausedUntil !== null ||
      (state.retryNotBefore !== null && state.retryNotBefore > context.at) ||
      (state.serverRetryNotBefore !== null && state.serverRetryNotBefore > context.at)) {
    return null;
  }
  if (state.active !== null) {
    if (state.active.fencingToken === fencingToken) {
      return { trigger: state.active, resumed: true, state };
    }
    const revalidated: ScheduleState = {
      ...state,
      active: {
        ...state.active,
        fencingToken,
      },
    };
    const revalidatedTrigger = revalidated.active;
    if (revalidatedTrigger === null) {
      throw new SchedulerTransitionError("invalid_schedule_state", "active trigger disappeared");
    }
    return {
      trigger: revalidatedTrigger,
      resumed: true,
      state: revalidated,
    };
  }

  const dueCount = countDueTicks(state, context.at);
  if (!state.pending && dueCount === 0) {
    return null;
  }
  const trigger: ActiveTrigger = {
    triggerId: state.pending?.triggerId ?? `scheduled-${state.nextTickIndex}-${context.at}`,
    kind: triggerKindForClaim(state.pending, dueCount),
    missedCount: (state.pending?.missedCount ?? 0) + dueCount,
    dueAt: claimedDueAt(state.pending, state.nextDueAt, dueCount),
    jitterMs: claimedJitterMs(state.pending, state.jitterMs, dueCount),
    claimedAt: context.at,
    fencingToken,
  };
  const nextTickIndex = dueCount > 0
    ? state.nextTickIndex + dueCount
    : state.nextTickIndex;
  return {
    trigger,
    resumed: state.pending?.triggerId !== undefined,
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

  const serverRetryDeadline = retryDeadline(context.at, request.retryAfterMs);
  let failureStreak = state.failureStreak;
  let retryNotBefore = state.retryNotBefore;
  let serverRetryNotBefore = state.serverRetryNotBefore;
  let authPausedUntil = state.authPausedUntil;
  let pending = state.pending;
  if (request.outcome === "success") {
    retryNotBefore = null;
    serverRetryNotBefore = null;
    failureStreak = 0;
    authPausedUntil = null;
  } else {
    failureStreak = Math.min(MAX_FAILURE_STREAK, failureStreak + 1);
    retryNotBefore = addDuration(context.at, retryBackoffMs(failureStreak), "retry backoff");
    pending = pendingWithRequestedAt(
      combinePending(state.pending, {
        kind: state.active.kind,
        missedCount: state.active.missedCount,
        triggerId: state.active.triggerId,
        dueAt: state.active.dueAt,
        jitterMs: state.active.jitterMs,
      }),
      state.pending?.requestedAt ?? context.at,
    );
    serverRetryNotBefore = laterDeadline(serverRetryNotBefore, serverRetryDeadline);
  }
  if (request.outcome === "authentication") {
    authPausedUntil = Number.MAX_SAFE_INTEGER;
  }

  return {
    applied: true,
    state: {
      ...state,
      active: null,
      retryNotBefore,
      serverRetryNotBefore,
      failureStreak,
      authPausedUntil,
      pending,
      lastCompletedAt: context.at,
    },
  };
}

export function recoverAuthentication(
  state: ScheduleState,
  context: ScheduleContext,
): ScheduleMutationResult {
  validateScope(state.scope);
  validateEpoch(context.at, "at");
  if (state.authPausedUntil === null) {
    return { applied: false, state };
  }
  return {
    applied: true,
    state: {
      ...state,
      authPausedUntil: null,
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
  return withPending(
    {
      ...state,
      nextTickIndex,
      nextDueAt: scheduledDueAt(
        state.anchorAt,
        state.intervalMs,
        nextTickIndex,
        state.jitterMs,
      ),
    },
    pending,
    state.pending?.requestedAt ?? context.at,
  );
}

export function reconcileSchedule(
  state: ScheduleState,
  options: ScheduleOptions,
  context: ScheduleContext,
): ScheduleState {
  validateScope(state.scope);
  validateEpoch(context.at, "at");
  const caughtUp = materializeDue(state, context);
  const interval = parseRefreshInterval(options.interval ?? state.interval);
  const jitterSeconds = validateJitterSeconds(options.jitterSeconds ?? state.jitterSeconds);
  const anchorAt = validateEpoch(options.anchorAt ?? state.anchorAt, "anchorAt");
  const intervalChanged = interval.milliseconds !== state.intervalMs;
  const anchorChanged = anchorAt !== state.anchorAt;
  const jitterChanged = jitterSeconds !== state.jitterSeconds;
  const configChanged = intervalChanged || anchorChanged || jitterChanged;

  if (!configChanged) {
    return caughtUp;
  }

  const jitterMs = jitterChanged ? sampleJitter(jitterSeconds) : state.jitterMs;
  const nextTickIndex = intervalChanged
    ? firstFutureTickIndex(
        anchorAt,
        interval.milliseconds,
        jitterMs,
        context.at,
      )
    : anchorChanged
      ? 1
      : caughtUp.nextTickIndex;
  const reconfigured = {
    ...caughtUp,
    interval: interval.spec,
    intervalMs: interval.milliseconds,
    anchorAt,
    jitterSeconds,
    jitterMs,
    nextTickIndex,
    nextDueAt: scheduledDueAt(
      anchorAt,
      interval.milliseconds,
      nextTickIndex,
      jitterMs,
    ),
  };
  return materializeDue(reconfigured, context);
}

function combinePending(
  current: SchedulePending | null,
  incoming: Pick<SchedulePending, "kind" | "missedCount"> &
    Partial<Pick<SchedulePending, "triggerId" | "dueAt" | "jitterMs">>,
): SchedulePending {
  const combined: SchedulePending = {
    kind: pendingKind(current, incoming),
    missedCount: (current?.missedCount ?? 0) + incoming.missedCount,
    requestedAt: current?.requestedAt ?? 0,
  };
  const triggerId = current?.triggerId ?? incoming.triggerId;
  const dueAt = mergedDueAt(current, incoming);
  const jitterMs = mergedJitterMs(current, incoming, dueAt);
  const hasRetryMetadata =
    triggerId !== undefined || dueAt !== undefined || jitterMs !== undefined;
  if (!hasRetryMetadata) {
    return combined;
  }
  return {
    ...combined,
    ...(triggerId === undefined ? {} : { triggerId }),
    ...(dueAt === undefined ? {} : { dueAt }),
    ...(jitterMs === undefined ? {} : { jitterMs }),
  };
}

function withPending(
  state: ScheduleState,
  pending: SchedulePending,
  requestedAt: number,
): ScheduleState {
  return {
    ...state,
    pending: pendingWithRequestedAt(pending, requestedAt),
  };
}

function pendingWithRequestedAt(
  pending: SchedulePending,
  requestedAt: number,
): SchedulePending {
  return { ...pending, requestedAt };
}

function triggerKindForClaim(
  pending: SchedulePending | null,
  dueCount: number,
): TriggerKind {
  if (dueCount === 0) {
    return pending?.kind ?? "scheduled";
  }
  if (!pending || (pending.kind === "scheduled" && pending.triggerId === undefined)) {
    return "scheduled";
  }
  return "coalesced";
}

function pendingKind(
  current: SchedulePending | null,
  incoming: Pick<SchedulePending, "kind" | "missedCount"> &
    Partial<Pick<SchedulePending, "triggerId" | "dueAt" | "jitterMs">>,
): TriggerKind {
  if (current === null) {
    return incoming.kind;
  }
  if (current.triggerId !== undefined || incoming.triggerId !== undefined) {
    return "coalesced";
  }
  return current.kind === incoming.kind ? incoming.kind : "coalesced";
}

function mergedDueAt(
  current: SchedulePending | null,
  incoming: Partial<Pick<SchedulePending, "dueAt">>,
): number | null | undefined {
  if (current === null) {
    return incoming.dueAt;
  }
  if (current.dueAt === undefined) {
    return undefined;
  }
  if (current.dueAt === null) {
    return incoming.dueAt === undefined ? null : incoming.dueAt;
  }
  if (incoming.dueAt === undefined || incoming.dueAt === null) {
    return current.dueAt;
  }
  return Math.min(current.dueAt, incoming.dueAt);
}

function mergedJitterMs(
  current: SchedulePending | null,
  incoming: Partial<Pick<SchedulePending, "dueAt" | "jitterMs">>,
  dueAt: number | null | undefined,
): number | undefined {
  if (dueAt === undefined || dueAt === null) {
    return undefined;
  }
  if (
    current?.dueAt !== undefined &&
    current.dueAt !== null &&
    current.dueAt <= dueAt
  ) {
    return current.jitterMs;
  }
  return incoming.jitterMs;
}

function claimedDueAt(
  pending: SchedulePending | null,
  nextDueAt: number,
  dueCount: number,
): number | null {
  if (pending?.dueAt === undefined) {
    return pending === null && dueCount > 0 ? nextDueAt : null;
  }
  if (pending.dueAt === null) {
    return dueCount > 0 ? nextDueAt : null;
  }
  return dueCount > 0 ? Math.min(pending.dueAt, nextDueAt) : pending.dueAt;
}

function claimedJitterMs(
  pending: SchedulePending | null,
  scheduleJitterMs: number,
  dueCount: number,
): number {
  if (pending?.dueAt === undefined || pending.dueAt === null) {
    return dueCount > 0 ? scheduleJitterMs : pending?.jitterMs ?? scheduleJitterMs;
  }
  return pending.jitterMs ?? scheduleJitterMs;
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

function retryDeadline(at: number, retryAfterMs: number | null): number | null {
  if (retryAfterMs === null) {
    return null;
  }
  return addDuration(at, retryAfterMs, "retryAfterMs");
}

function firstFutureTickIndex(
  anchorAt: number,
  intervalMs: number,
  jitterMs: number,
  now: number,
): number {
  return Math.max(1, Math.floor((now - anchorAt - jitterMs) / intervalMs) + 1);
}

function addDuration(at: number, duration: number, label: string): number {
  validateEpoch(at, "at");
  const validatedDuration = validateNonNegativeInteger(duration, label);
  if (at > Number.MAX_SAFE_INTEGER - validatedDuration) {
    throw new SchedulerTransitionError(
      "invalid_request",
      `${label} exceeds the safe epoch range`,
    );
  }
  return at + validatedDuration;
}

function laterDeadline(
  current: number | null,
  incoming: number | null,
): number | null {
  if (incoming === null) {
    return current;
  }
  return current === null || incoming > current ? incoming : current;
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
