export type SchedulerTime = number | Date;

export interface SchedulerScope {
  accountId: string;
  profileId: string;
}

export interface ScheduleOptions {
  interval?: string;
  anchorAt?: SchedulerTime;
  jitterSeconds?: number;
}

export interface AtOptions {
  at?: SchedulerTime;
}

export interface LeaseOptions extends AtOptions {
  leaseDurationMs?: number;
}

export interface SchedulerState {
  scope: SchedulerScope;
  anchorAt: number;
  interval: string;
  intervalMs: number;
  jitterSeconds: number;
  jitterMs: number;
  nextTickIndex: number;
  nextDueAt: number;
  pendingTrigger: boolean;
  pendingKind: TriggerKind | null;
  pendingMissedCount: number;
  pendingRequestedAt: number | null;
  activeTriggerId: string | null;
  activeKind: TriggerKind | null;
  activeMissedCount: number | null;
  activeDueAt: number | null;
  activeFencingToken: number | null;
  activeStartedAt: number | null;
  activeJitterMs: number | null;
  lastTriggerAt: number | null;
  lastCompletedAt: number | null;
  updatedAt: number;
}

export type TriggerKind = "scheduled" | "manual" | "coalesced";

export interface SchedulerTrigger {
  triggerId: string;
  kind: TriggerKind;
  missedCount: number;
  dueAt: number | null;
  jitterMs: number;
  claimedAt: number;
  fencingToken: number;
}

export interface TriggerClaim {
  trigger: SchedulerTrigger;
  resumed: boolean;
  state: SchedulerState;
}

export interface LeaseState {
  scope: SchedulerScope;
  ownerId: string | null;
  fencingToken: number;
  leaseUntilAt: number;
  heartbeatAt: number | null;
  updatedAt: number;
}

export interface LeaseClaim {
  acquired: boolean;
  lease: LeaseState;
}

export interface LeaseMutation {
  applied: boolean;
  lease: LeaseState | null;
}

export interface RefreshRequest {
  queued: true;
  coalesced: boolean;
  occupied: boolean;
  state: SchedulerState;
}

export const DEFAULT_REFRESH_INTERVAL = "PT1H";
export const DEFAULT_JITTER_SECONDS = 60;
export const DEFAULT_LEASE_DURATION_MS = 120_000;
