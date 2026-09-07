import Database from "better-sqlite3";
import { randomUUID } from "node:crypto";

import { parseRefreshInterval } from "./interval.js";
import {
  DEFAULT_JITTER_SECONDS,
  DEFAULT_LEASE_DURATION_MS,
  DEFAULT_REFRESH_INTERVAL,
} from "./types.js";
import type {
  AtOptions,
  LeaseClaim,
  LeaseMutation,
  LeaseOptions,
  LeaseState,
  ScheduleOptions,
  SchedulerScope,
  SchedulerState,
  SchedulerTime,
  RefreshRequest,
  SchedulerTrigger,
  TriggerClaim,
  TriggerKind,
} from "./types.js";

type SqlRow = Record<string, unknown>;
type SqliteDatabase = InstanceType<typeof Database>;

const CREATE_SCHEDULER_TABLES = `
  CREATE TABLE IF NOT EXISTS chatgpt_scheduler_state (
    account_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    anchor_at INTEGER NOT NULL,
    interval_spec TEXT NOT NULL,
    interval_ms INTEGER NOT NULL,
    jitter_limit_seconds INTEGER NOT NULL,
    jitter_ms INTEGER NOT NULL,
    next_tick_index INTEGER NOT NULL,
    next_due_at INTEGER NOT NULL,
    pending_trigger INTEGER NOT NULL DEFAULT 0,
    pending_kind TEXT,
    pending_missed_count INTEGER NOT NULL DEFAULT 0,
    pending_requested_at INTEGER,
    active_trigger_id TEXT,
    active_kind TEXT,
    active_missed_count INTEGER,
    active_due_at INTEGER,
    active_fencing_token INTEGER,
    active_started_at INTEGER,
    active_jitter_ms INTEGER,
    last_trigger_at INTEGER,
    last_completed_at INTEGER,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (account_id, profile_id),
    CHECK (interval_ms >= 300000),
    CHECK (jitter_limit_seconds BETWEEN 0 AND 60),
    CHECK (jitter_ms BETWEEN 0 AND jitter_limit_seconds * 1000),
    CHECK (next_tick_index >= 1),
    CHECK (pending_trigger IN (0, 1)),
    CHECK (pending_missed_count >= 0),
    CHECK (
      pending_kind IS NULL OR
      pending_kind IN ('scheduled', 'manual', 'coalesced')
    ),
    CHECK (
      active_kind IS NULL OR
      active_kind IN ('scheduled', 'manual', 'coalesced')
    )
  );

  CREATE TABLE IF NOT EXISTS chatgpt_scheduler_leases (
    account_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    owner_id TEXT,
    fencing_token INTEGER NOT NULL,
    lease_until_at INTEGER NOT NULL,
    heartbeat_at INTEGER,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (account_id, profile_id)
  );
`;

export class SchedulerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SchedulerError";
  }
}

export class SchedulerLeaseError extends SchedulerError {
  constructor(message: string) {
    super(message);
    this.name = "SchedulerLeaseError";
  }
}

export interface SchedulerStoreOptions {
  clock?: () => number;
  random?: () => number;
}

/**
 * Durable Stage-3 scheduler state. The injected database remains owned by the
 * caller; this class creates only its two prefixed scheduler tables.
 */
export class SchedulerStore {
  private savepointCounter = 0;
  private readonly clock: () => number;
  private readonly random: () => number;

  constructor(
    readonly db: SqliteDatabase,
    options: SchedulerStoreOptions = {},
  ) {
    this.clock = options.clock ?? (() => Date.now());
    this.random = options.random ?? Math.random;
    this.db.exec(CREATE_SCHEDULER_TABLES);
  }

  ensureSchedule(
    scope: SchedulerScope,
    options: ScheduleOptions = {},
  ): SchedulerState {
    validateScope(scope);
    const interval = parseRefreshInterval(options.interval ?? DEFAULT_REFRESH_INTERVAL);
    const jitterSeconds = validateJitterSeconds(
      options.jitterSeconds ?? DEFAULT_JITTER_SECONDS,
    );
    const now = normalizeTime(this.clock());
    const anchorAt = normalizeTime(options.anchorAt ?? now);

    return this.transaction(() => {
      const existing = this.readSchedule(scope);
      if (existing) {
        return existing;
      }
      this.insertSchedule(scope, {
        anchorAt,
        intervalSpec: interval.spec,
        intervalMs: interval.milliseconds,
        jitterSeconds,
        jitterMs: sampleJitter(jitterSeconds, this.random),
        updatedAt: now,
      });
      return this.requireSchedule(scope);
    });
  }

  getSchedule(scope: SchedulerScope): SchedulerState | null {
    validateScope(scope);
    return this.readSchedule(scope);
  }

  reconfigureSchedule(
    scope: SchedulerScope,
    intervalValue: string,
    options: AtOptions = {},
  ): SchedulerState {
    validateScope(scope);
    const interval = parseRefreshInterval(intervalValue);
    const now = resolveNow(this.clock, options.at);

    return this.transaction(() => {
      let state = this.requireSchedule(scope);
      state = this.materializeDue(state, now);
      const nextTickIndex = firstFutureTickIndex(
        state.anchorAt,
        interval.milliseconds,
        state.jitterMs,
        now,
      );
      const nextDueAt = scheduledDueAt(
        state.anchorAt,
        interval.milliseconds,
        nextTickIndex,
        state.jitterMs,
      );
      this.db
        .prepare(`
          UPDATE chatgpt_scheduler_state
          SET interval_spec=?,
              interval_ms=?,
              next_tick_index=?,
              next_due_at=?,
              updated_at=?
          WHERE account_id=? AND profile_id=?
        `)
        .run(
          interval.spec,
          interval.milliseconds,
          nextTickIndex,
          nextDueAt,
          now,
          scope.accountId,
          scope.profileId,
        );
      return this.requireSchedule(scope);
    });
  }

  requestRefresh(
    scope: SchedulerScope,
    options: AtOptions = {},
  ): RefreshRequest {
    validateScope(scope);
    const explicitAt = explicitTime(options.at);

    return this.transaction(() => {
      const now = explicitAt ?? normalizeTime(this.clock());
      let state = this.readSchedule(scope);
      if (!state) {
        const interval = parseRefreshInterval(DEFAULT_REFRESH_INTERVAL);
        this.insertSchedule(scope, {
          anchorAt: now,
          intervalSpec: interval.spec,
          intervalMs: interval.milliseconds,
          jitterSeconds: DEFAULT_JITTER_SECONDS,
          jitterMs: sampleJitter(DEFAULT_JITTER_SECONDS, this.random),
          updatedAt: now,
        });
        state = this.requireSchedule(scope);
      }

      const lease = this.readLease(scope);
      const occupied =
        lease !== null &&
        lease.ownerId !== null &&
        lease.leaseUntilAt > now;
      const wasPending = state.pendingTrigger;
      const pendingKind = combinePendingKind(
        state.pendingKind,
        wasPending ? null : "manual",
      );
      this.db
        .prepare(`
          UPDATE chatgpt_scheduler_state
          SET pending_trigger=1,
              pending_kind=?,
              pending_requested_at=COALESCE(pending_requested_at, ?),
              updated_at=?
          WHERE account_id=? AND profile_id=?
        `)
        .run(
          pendingKind,
          state.pendingRequestedAt ?? now,
          now,
          scope.accountId,
          scope.profileId,
        );

      return {
        queued: true,
        coalesced: occupied || wasPending,
        occupied,
        state: this.requireSchedule(scope),
      };
    });
  }

  claimLease(
    scope: SchedulerScope,
    ownerId: string,
    options: LeaseOptions = {},
  ): LeaseClaim {
    validateScope(scope);
    validateNonEmpty(ownerId, "ownerId");
    const explicitAt = explicitTime(options.at);
    const leaseDurationMs = validateLeaseDuration(
      options.leaseDurationMs ?? DEFAULT_LEASE_DURATION_MS,
    );

    return this.transaction(() => {
      const now = explicitAt ?? normalizeTime(this.clock());
      const current = this.readLease(scope);
      if (!current) {
        this.db
          .prepare(`
            INSERT INTO chatgpt_scheduler_leases(
              account_id, profile_id, owner_id, fencing_token,
              lease_until_at, heartbeat_at, updated_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?)
          `)
          .run(
            scope.accountId,
            scope.profileId,
            ownerId,
            now + leaseDurationMs,
            now,
            now,
          );
        return {
          acquired: true,
          lease: this.requireLease(scope),
        };
      }

      if (current.leaseUntilAt > now) {
        return {
          acquired: false,
          lease: current,
        };
      }

      this.db
        .prepare(`
          UPDATE chatgpt_scheduler_leases
          SET owner_id=?,
              fencing_token=fencing_token + 1,
              lease_until_at=?,
              heartbeat_at=?,
              updated_at=?
          WHERE account_id=? AND profile_id=?
        `)
        .run(
          ownerId,
          now + leaseDurationMs,
          now,
          now,
          scope.accountId,
          scope.profileId,
        );
      return {
        acquired: true,
        lease: this.requireLease(scope),
      };
    });
  }

  getLease(scope: SchedulerScope): LeaseState | null {
    validateScope(scope);
    return this.readLease(scope);
  }

  heartbeatLease(
    scope: SchedulerScope,
    ownerId: string,
    fencingToken: number,
    options: LeaseOptions = {},
  ): LeaseMutation {
    validateScope(scope);
    validateNonEmpty(ownerId, "ownerId");
    validateFencingToken(fencingToken);
    const explicitAt = explicitTime(options.at);
    const leaseDurationMs = validateLeaseDuration(
      options.leaseDurationMs ?? DEFAULT_LEASE_DURATION_MS,
    );

    return this.transaction(() => {
      const now = explicitAt ?? normalizeTime(this.clock());
      const result = this.db
        .prepare(`
          UPDATE chatgpt_scheduler_leases
          SET lease_until_at=?,
              heartbeat_at=?,
              updated_at=?
          WHERE account_id=? AND profile_id=?
            AND owner_id=?
            AND fencing_token=?
            AND lease_until_at > ?
        `)
        .run(
          now + leaseDurationMs,
          now,
          now,
          scope.accountId,
          scope.profileId,
          ownerId,
          fencingToken,
          now,
        );
      return {
        applied: result.changes === 1,
        lease: this.readLease(scope),
      };
    });
  }

  releaseLease(
    scope: SchedulerScope,
    ownerId: string,
    fencingToken: number,
    options: AtOptions = {},
  ): LeaseMutation {
    validateScope(scope);
    validateNonEmpty(ownerId, "ownerId");
    validateFencingToken(fencingToken);
    const explicitAt = explicitTime(options.at);

    return this.transaction(() => {
      const now = explicitAt ?? normalizeTime(this.clock());
      const result = this.db
        .prepare(`
          UPDATE chatgpt_scheduler_leases
          SET owner_id=NULL,
              lease_until_at=?,
              heartbeat_at=?,
              updated_at=?
          WHERE account_id=? AND profile_id=?
            AND owner_id=?
            AND fencing_token=?
            AND lease_until_at > ?
        `)
        .run(
          now,
          now,
          now,
          scope.accountId,
          scope.profileId,
          ownerId,
          fencingToken,
          now,
        );
      return {
        applied: result.changes === 1,
        lease: this.readLease(scope),
      };
    });
  }

  claimTrigger(
    scope: SchedulerScope,
    ownerId: string,
    fencingToken: number,
    options: AtOptions = {},
  ): TriggerClaim | null {
    validateScope(scope);
    validateNonEmpty(ownerId, "ownerId");
    validateFencingToken(fencingToken);
    const explicitAt = explicitTime(options.at);

    return this.transaction(() => {
      const now = explicitAt ?? normalizeTime(this.clock());
      this.assertLease(scope, ownerId, fencingToken, now);
      let state = this.requireSchedule(scope);

      if (state.activeTriggerId !== null) {
        state = this.materializeDue(state, now);
        if (state.activeFencingToken !== fencingToken) {
          this.db
            .prepare(`
              UPDATE chatgpt_scheduler_state
              SET active_fencing_token=?, updated_at=?
              WHERE account_id=? AND profile_id=?
            `)
            .run(fencingToken, now, scope.accountId, scope.profileId);
          state = this.requireSchedule(scope);
        }
        return {
          trigger: triggerFromState(state),
          resumed: true,
          state,
        };
      }

      const dueCount = countDueTicks(state, now);
      const missedCount = state.pendingMissedCount + dueCount;
      if (!state.pendingTrigger && dueCount === 0) {
        return null;
      }

      const kind = triggerKindForClaim(state, dueCount);
      const dueAt = dueCount > 0 ? state.nextDueAt : null;
      const nextTickIndex =
        dueCount > 0 ? state.nextTickIndex + dueCount : state.nextTickIndex;
      const nextDueAt =
        dueCount > 0
          ? scheduledDueAt(
              state.anchorAt,
              state.intervalMs,
              nextTickIndex,
              state.jitterMs,
            )
          : state.nextDueAt;
      const triggerId = randomUUID();

      this.db
        .prepare(`
          UPDATE chatgpt_scheduler_state
          SET next_tick_index=?,
              next_due_at=?,
              pending_trigger=0,
              pending_kind=NULL,
              pending_missed_count=0,
              pending_requested_at=NULL,
              active_trigger_id=?,
              active_kind=?,
              active_missed_count=?,
              active_due_at=?,
              active_fencing_token=?,
              active_started_at=?,
              active_jitter_ms=?,
              last_trigger_at=?,
              updated_at=?
          WHERE account_id=? AND profile_id=?
        `)
        .run(
          nextTickIndex,
          nextDueAt,
          triggerId,
          kind,
          missedCount,
          dueAt,
          fencingToken,
          now,
          state.jitterMs,
          now,
          now,
          scope.accountId,
          scope.profileId,
        );
      state = this.requireSchedule(scope);
      return {
        trigger: triggerFromState(state),
        resumed: false,
        state,
      };
    });
  }

  completeTrigger(
    scope: SchedulerScope,
    ownerId: string,
    fencingToken: number,
    options: AtOptions = {},
  ): LeaseMutation {
    validateScope(scope);
    validateNonEmpty(ownerId, "ownerId");
    validateFencingToken(fencingToken);
    const explicitAt = explicitTime(options.at);

    return this.transaction(() => {
      const now = explicitAt ?? normalizeTime(this.clock());
      if (!this.hasValidLease(scope, ownerId, fencingToken, now)) {
        return {
          applied: false,
          lease: this.readLease(scope),
        };
      }
      const state = this.requireSchedule(scope);
      if (
        state.activeTriggerId === null ||
        state.activeFencingToken !== fencingToken
      ) {
        return {
          applied: false,
          lease: this.readLease(scope),
        };
      }
      this.db
        .prepare(`
          UPDATE chatgpt_scheduler_state
          SET active_trigger_id=NULL,
              active_kind=NULL,
              active_missed_count=NULL,
              active_due_at=NULL,
              active_fencing_token=NULL,
              active_started_at=NULL,
              active_jitter_ms=NULL,
              last_completed_at=?,
              updated_at=?
          WHERE account_id=? AND profile_id=?
        `)
        .run(now, now, scope.accountId, scope.profileId);
      return {
        applied: true,
        lease: this.readLease(scope),
      };
    });
  }

  private insertSchedule(
    scope: SchedulerScope,
    values: {
      anchorAt: number;
      intervalSpec: string;
      intervalMs: number;
      jitterSeconds: number;
      jitterMs: number;
      updatedAt: number;
    },
  ): void {
    const nextTickIndex = 1;
    const nextDueAt = scheduledDueAt(
      values.anchorAt,
      values.intervalMs,
      nextTickIndex,
      values.jitterMs,
    );
    this.db
      .prepare(`
        INSERT INTO chatgpt_scheduler_state(
          account_id, profile_id, anchor_at, interval_spec, interval_ms,
          jitter_limit_seconds, jitter_ms, next_tick_index, next_due_at,
          pending_trigger, pending_kind, pending_missed_count,
          pending_requested_at, active_trigger_id, active_kind,
          active_missed_count, active_due_at, active_fencing_token,
          active_started_at, active_jitter_ms, last_trigger_at,
          last_completed_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, NULL, NULL, NULL,
                  NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
      `)
      .run(
        scope.accountId,
        scope.profileId,
        values.anchorAt,
        values.intervalSpec,
        values.intervalMs,
        values.jitterSeconds,
        values.jitterMs,
        nextTickIndex,
        nextDueAt,
        values.updatedAt,
      );
  }

  private materializeDue(
    state: SchedulerState,
    now: number,
  ): SchedulerState {
    const dueCount = countDueTicks(state, now);
    if (dueCount === 0) {
      return state;
    }
    const nextTickIndex = state.nextTickIndex + dueCount;
    const nextDueAt = scheduledDueAt(
      state.anchorAt,
      state.intervalMs,
      nextTickIndex,
      state.jitterMs,
    );
    const pendingKind = combinePendingKind(state.pendingKind, "scheduled");
    this.db
      .prepare(`
        UPDATE chatgpt_scheduler_state
        SET next_tick_index=?,
            next_due_at=?,
            pending_trigger=1,
            pending_kind=?,
            pending_missed_count=pending_missed_count + ?,
            pending_requested_at=COALESCE(pending_requested_at, ?),
            updated_at=?
        WHERE account_id=? AND profile_id=?
      `)
      .run(
        nextTickIndex,
        nextDueAt,
        pendingKind,
        dueCount,
        now,
        now,
        state.scope.accountId,
        state.scope.profileId,
      );
    return this.requireSchedule(state.scope);
  }

  private assertLease(
    scope: SchedulerScope,
    ownerId: string,
    fencingToken: number,
    now: number,
  ): void {
    if (!this.hasValidLease(scope, ownerId, fencingToken, now)) {
      throw new SchedulerLeaseError(
        `lease is not held by owner ${ownerId} for ${scope.accountId}/${scope.profileId}`,
      );
    }
  }

  private hasValidLease(
    scope: SchedulerScope,
    ownerId: string,
    fencingToken: number,
    now: number,
  ): boolean {
    const lease = this.readLease(scope);
    return (
      lease !== null &&
      lease.ownerId === ownerId &&
      lease.fencingToken === fencingToken &&
      lease.leaseUntilAt > now
    );
  }

  private requireSchedule(scope: SchedulerScope): SchedulerState {
    const state = this.readSchedule(scope);
    if (!state) {
      throw new SchedulerError(
        `schedule not found for ${scope.accountId}/${scope.profileId}`,
      );
    }
    return state;
  }

  private requireLease(scope: SchedulerScope): LeaseState {
    const lease = this.readLease(scope);
    if (!lease) {
      throw new SchedulerError(
        `lease not found for ${scope.accountId}/${scope.profileId}`,
      );
    }
    return lease;
  }

  private readSchedule(scope: SchedulerScope): SchedulerState | null {
    const row = this.db
      .prepare(`
        SELECT *
        FROM chatgpt_scheduler_state
        WHERE account_id=? AND profile_id=?
      `)
      .get(scope.accountId, scope.profileId) as SqlRow | undefined;
    return row ? scheduleFromRow(row, scope) : null;
  }

  private readLease(scope: SchedulerScope): LeaseState | null {
    const row = this.db
      .prepare(`
        SELECT *
        FROM chatgpt_scheduler_leases
        WHERE account_id=? AND profile_id=?
      `)
      .get(scope.accountId, scope.profileId) as SqlRow | undefined;
    return row ? leaseFromRow(row, scope) : null;
  }

  private transaction<T>(callback: () => T): T {
    if (!this.db.inTransaction) {
      this.db.exec("BEGIN IMMEDIATE");
      try {
        const result = callback();
        this.db.exec("COMMIT");
        return result;
      } catch (error) {
        if (this.db.inTransaction) {
          this.db.exec("ROLLBACK");
        }
        throw error;
      }
    }

    const savepoint = `scheduler_sp_${this.savepointCounter++}`;
    this.db.exec(`SAVEPOINT ${savepoint}`);
    try {
      const result = callback();
      this.db.exec(`RELEASE ${savepoint}`);
      return result;
    } catch (error) {
      this.db.exec(`ROLLBACK TO ${savepoint}`);
      this.db.exec(`RELEASE ${savepoint}`);
      throw error;
    }
  }
}

function scheduleFromRow(
  row: SqlRow,
  scope: SchedulerScope,
): SchedulerState {
  const pendingTrigger = integer(row.pending_trigger, "pending_trigger") === 1;
  const pendingKind = nullableTriggerKind(row.pending_kind, "pending_kind");
  const activeTriggerId = nullableString(row.active_trigger_id);
  const activeKind = nullableTriggerKind(row.active_kind, "active_kind");
  const activeMissedCount = nullableInteger(
    row.active_missed_count,
    "active_missed_count",
  );
  const activeFencingToken = nullableInteger(
    row.active_fencing_token,
    "active_fencing_token",
  );
  const activeStartedAt = nullableInteger(
    row.active_started_at,
    "active_started_at",
  );
  if (pendingTrigger && pendingKind === null) {
    throw new SchedulerError("scheduler state has pending work without a kind");
  }
  if (!pendingTrigger && pendingKind !== null) {
    throw new SchedulerError("scheduler state has a pending kind without work");
  }
  if (activeTriggerId === null && activeKind !== null) {
    throw new SchedulerError("scheduler state has an active kind without a trigger");
  }
  if (activeTriggerId !== null && (activeKind === null || activeMissedCount === null)) {
    throw new SchedulerError("scheduler state has an incomplete active trigger");
  }

  return {
    scope,
    anchorAt: integer(row.anchor_at, "anchor_at"),
    interval: stringValue(row.interval_spec, "interval_spec"),
    intervalMs: integer(row.interval_ms, "interval_ms"),
    jitterSeconds: integer(row.jitter_limit_seconds, "jitter_limit_seconds"),
    jitterMs: integer(row.jitter_ms, "jitter_ms"),
    nextTickIndex: integer(row.next_tick_index, "next_tick_index"),
    nextDueAt: integer(row.next_due_at, "next_due_at"),
    pendingTrigger,
    pendingKind,
    pendingMissedCount: integer(
      row.pending_missed_count,
      "pending_missed_count",
    ),
    pendingRequestedAt: nullableInteger(
      row.pending_requested_at,
      "pending_requested_at",
    ),
    activeTriggerId,
    activeKind,
    activeMissedCount,
    activeDueAt: nullableInteger(row.active_due_at, "active_due_at"),
    activeFencingToken,
    activeStartedAt,
    activeJitterMs: nullableInteger(row.active_jitter_ms, "active_jitter_ms"),
    lastTriggerAt: nullableInteger(row.last_trigger_at, "last_trigger_at"),
    lastCompletedAt: nullableInteger(row.last_completed_at, "last_completed_at"),
    updatedAt: integer(row.updated_at, "updated_at"),
  };
}

function leaseFromRow(row: SqlRow, scope: SchedulerScope): LeaseState {
  return {
    scope,
    ownerId: nullableString(row.owner_id),
    fencingToken: integer(row.fencing_token, "fencing_token"),
    leaseUntilAt: integer(row.lease_until_at, "lease_until_at"),
    heartbeatAt: nullableInteger(row.heartbeat_at, "heartbeat_at"),
    updatedAt: integer(row.updated_at, "updated_at"),
  };
}

function triggerFromState(state: SchedulerState): SchedulerTrigger {
  if (
    state.activeTriggerId === null ||
    state.activeKind === null ||
    state.activeMissedCount === null ||
    state.activeStartedAt === null ||
    state.activeFencingToken === null
  ) {
    throw new SchedulerError("scheduler state has no complete active trigger");
  }
  return {
    triggerId: state.activeTriggerId,
    kind: state.activeKind,
    missedCount: state.activeMissedCount,
    dueAt: state.activeDueAt,
    jitterMs: state.activeJitterMs ?? state.jitterMs,
    claimedAt: state.activeStartedAt,
    fencingToken: state.activeFencingToken,
  };
}

function triggerKindForClaim(
  state: SchedulerState,
  dueCount: number,
): TriggerKind {
  if (!state.pendingTrigger) {
    return "scheduled";
  }
  if (state.pendingKind === "manual" && dueCount === 0 && state.pendingMissedCount === 0) {
    return "manual";
  }
  if (state.pendingKind === "scheduled" && dueCount === 0) {
    return "scheduled";
  }
  return "coalesced";
}

function combinePendingKind(
  existing: TriggerKind | null,
  incoming: TriggerKind | null,
): TriggerKind | null {
  if (existing === "coalesced" || incoming === "coalesced") {
    return "coalesced";
  }
  if (existing === null) {
    return incoming;
  }
  if (incoming === null) {
    return existing;
  }
  if (existing === incoming) {
    return existing;
  }
  return "coalesced";
}

function countDueTicks(state: SchedulerState, now: number): number {
  if (state.nextDueAt > now) {
    return 0;
  }
  const elapsed = now - state.nextDueAt;
  const count = Math.floor(elapsed / state.intervalMs) + 1;
  if (!Number.isSafeInteger(count) || count < 1) {
    throw new SchedulerError("scheduler due-tick count exceeds safe integer range");
  }
  return count;
}

function scheduledDueAt(
  anchorAt: number,
  intervalMs: number,
  tickIndex: number,
  jitterMs: number,
): number {
  const dueAt = anchorAt + intervalMs * tickIndex + jitterMs;
  if (!Number.isSafeInteger(dueAt)) {
    throw new SchedulerError("scheduler due time exceeds safe integer range");
  }
  return dueAt;
}

function firstFutureTickIndex(
  anchorAt: number,
  intervalMs: number,
  jitterMs: number,
  now: number,
): number {
  const elapsed = now - anchorAt - jitterMs;
  let index = elapsed < 0 ? 1 : Math.floor(elapsed / intervalMs) + 1;
  if (index < 1) {
    index = 1;
  }
  while (scheduledDueAt(anchorAt, intervalMs, index, jitterMs) <= now) {
    index += 1;
  }
  if (!Number.isSafeInteger(index)) {
    throw new SchedulerError("scheduler tick index exceeds safe integer range");
  }
  return index;
}

function sampleJitter(jitterSeconds: number, random: () => number): number {
  const limitMs = jitterSeconds * 1000;
  if (limitMs === 0) {
    return 0;
  }
  const sample = random();
  if (!Number.isFinite(sample)) {
    throw new SchedulerError("scheduler jitter source returned a non-finite value");
  }
  const bounded = Math.min(1, Math.max(0, sample));
  return Math.min(limitMs, Math.floor(bounded * (limitMs + 1)));
}

function validateJitterSeconds(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 60) {
    throw new SchedulerError("jitterSeconds must be an integer from 0 through 60");
  }
  return value;
}

function validateLeaseDuration(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new SchedulerError("leaseDurationMs must be a positive integer");
  }
  return value;
}

function validateFencingToken(value: number): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new SchedulerError("fencingToken must be a positive integer");
  }
}

function validateScope(scope: SchedulerScope): void {
  validateNonEmpty(scope.accountId, "accountId");
  validateNonEmpty(scope.profileId, "profileId");
}

function validateNonEmpty(value: string, label: string): void {
  if (typeof value !== "string" || value.trim() === "") {
    throw new SchedulerError(`${label} must be a non-empty string`);
  }
}

function resolveNow(clock: () => number, value: SchedulerTime | undefined): number {
  return normalizeTime(value ?? clock());
}

function explicitTime(value: SchedulerTime | undefined): number | undefined {
  return value === undefined ? undefined : normalizeTime(value);
}

function normalizeTime(value: SchedulerTime): number {
  const time = value instanceof Date ? value.getTime() : value;
  if (!Number.isSafeInteger(time)) {
    throw new SchedulerError("scheduler times must be safe integer epoch milliseconds");
  }
  return time;
}

function integer(value: unknown, label: string): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) {
    throw new SchedulerError(`scheduler column ${label} is not a safe integer`);
  }
  return number;
}

function nullableInteger(value: unknown, label: string): number | null {
  return value === null || value === undefined ? null : integer(value, label);
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new SchedulerError(`scheduler column ${label} is not a string`);
  }
  return value;
}

function nullableString(value: unknown): string | null {
  return value === null || value === undefined ? null : String(value);
}

function nullableTriggerKind(
  value: unknown,
  label: string,
): TriggerKind | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (value === "scheduled" || value === "manual" || value === "coalesced") {
    return value;
  }
  throw new SchedulerError(`scheduler column ${label} has an invalid trigger kind`);
}
