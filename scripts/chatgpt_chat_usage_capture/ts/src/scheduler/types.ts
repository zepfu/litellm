export const DEFAULT_REFRESH_INTERVAL = "PT1H";
export const DEFAULT_JITTER_SECONDS = 60;

export type SchedulerTime = number;

export interface ScheduleScope {
  readonly collectorAccountId: string;
  readonly profileId: string;
}

export interface ScheduleOptions {
  readonly interval?: string;
  readonly anchorAt?: SchedulerTime;
  readonly jitterSeconds?: number;
}

export interface ScheduleContext {
  readonly at: number;
}

export interface SchedulePending {
  readonly kind: TriggerKind;
  readonly missedCount: number;
  readonly requestedAt: number;
  readonly triggerId?: string;
  readonly dueAt?: number | null;
  readonly jitterMs?: number;
}

export type TriggerKind = "scheduled" | "manual" | "coalesced";

export interface ActiveTrigger {
  readonly triggerId: string;
  readonly kind: TriggerKind;
  readonly missedCount: number;
  readonly dueAt: number | null;
  readonly jitterMs: number;
  readonly claimedAt: number;
  readonly fencingToken: number;
}

export interface ScheduleState {
  readonly scope: ScheduleScope;
  readonly stateVersion: number;
  readonly interval: string;
  readonly intervalMs: number;
  readonly anchorAt: number;
  readonly jitterSeconds: number;
  readonly jitterMs: number;
  readonly nextTickIndex: number;
  readonly nextDueAt: number;
  readonly pending: SchedulePending | null;
  readonly active: ActiveTrigger | null;
  readonly retryNotBefore: number | null;
  readonly serverRetryNotBefore: number | null;
  readonly failureStreak: number;
  readonly authPausedUntil: number | null;
  readonly lastTriggerAt: number | null;
  readonly lastCompletedAt: number | null;
}

export interface TriggerClaim {
  readonly trigger: ActiveTrigger;
  readonly resumed: boolean;
  readonly state: ScheduleState;
}

export interface ScheduleMutationResult {
  readonly applied: boolean;
  readonly state: ScheduleState | null;
}

export interface RefreshRequestResult {
  readonly queued: true;
  readonly coalesced: boolean;
  readonly state: ScheduleState;
}

export type RetryOutcome = "success" | "failure" | "authentication";

export interface CompleteTriggerRequest {
  readonly outcome: RetryOutcome;
  readonly retryAfterMs: number | null;
}

export type SchedulerErrorCode =
  | "state_version_conflict"
  | "stale_fence"
  | "invalid_schedule_state"
  | "invalid_request";

export class SchedulerTransitionError extends Error {
  readonly code: SchedulerErrorCode;

  constructor(code: SchedulerErrorCode, message: string) {
    super(message);
    this.name = "SchedulerTransitionError";
    this.code = code;
  }
}

export function validateScope(scope: ScheduleScope): void {
  validateNonEmpty(scope.collectorAccountId, "collectorAccountId");
  validateNonEmpty(scope.profileId, "profileId");
}

export function validateNonEmpty(value: string, label: string): void {
  if (typeof value !== "string" || value.trim() === "") {
    throw new SchedulerTransitionError("invalid_request", `${label} must be a non-empty string`);
  }
}

export function validateEpoch(value: number, label: string): number {
  if (!Number.isSafeInteger(value)) {
    throw new SchedulerTransitionError(
      "invalid_request",
      `${label} must be a safe integer epoch millisecond`,
    );
  }
  return value;
}
