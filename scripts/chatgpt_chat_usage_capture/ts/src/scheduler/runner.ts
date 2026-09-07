import type { AccountConfig } from "../config.js";
import type {
  HistoryCollectionRequest,
  HistoryRange,
  HistoryReader,
} from "../contracts/history.js";
import { collectIntoLedger } from "../history/ingest.js";
import type { CollectionExecutionHooks, PersistedHistoryResult } from "../history/ingest.js";
import { Ledger } from "../ledger/store.js";
import {
  DEFAULT_LEASE_DURATION_MS,
  type LeaseClaim,
  type ScheduleOptions,
  type SchedulerScope,
  type SchedulerTrigger,
  type TriggerClaim,
} from "./types.js";
import { SchedulerStore } from "./store.js";

const RECONCILIATION_INTERVAL_MS = 24 * 60 * 60 * 1000;
const DEFAULT_HEARTBEAT_INTERVAL_MS = Math.floor(DEFAULT_LEASE_DURATION_MS / 3);

const CREATE_EXECUTION_STATE_TABLE = `
  CREATE TABLE IF NOT EXISTS chatgpt_scheduler_execution (
    account_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    reconciliation_anchor_at INTEGER NOT NULL,
    reconciliation_next_tick_index INTEGER NOT NULL,
    reconciliation_next_due_at INTEGER NOT NULL,
    reconciliation_pending_missed_count INTEGER NOT NULL DEFAULT 0,
    pending_request_json TEXT,
    pending_requested_at INTEGER,
    active_trigger_id TEXT,
    active_fencing_token INTEGER,
    active_request_json TEXT,
    active_reconciliation_missed_count INTEGER,
    last_reconciliation_started_at INTEGER,
    last_reconciliation_finished_at INTEGER,
    last_reconciliation_status TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (account_id, profile_id),
    CHECK (reconciliation_next_tick_index >= 1),
    CHECK (reconciliation_pending_missed_count >= 0),
    CHECK (active_reconciliation_missed_count IS NULL OR active_reconciliation_missed_count >= 0),
    CHECK (
      last_reconciliation_status IS NULL OR
      last_reconciliation_status IN ('complete', 'partial', 'blocked')
    )
  );
`;

export type SchedulerExecutionMode =
  | "manual"
  | "backfill"
  | "reconciliation"
  | "scheduled";

export type SchedulerCollectionOptions = Omit<
  HistoryCollectionRequest,
  "mode" | "range" | "now"
>;

export interface SchedulerExecutionRequest {
  mode: SchedulerExecutionMode;
  range?: HistoryRange;
  mappingVersion?: string | null;
  collection?: SchedulerCollectionOptions;
  signal?: AbortSignal;
}

export interface SchedulerReaderHandle {
  reader: HistoryReader;
  close?: () => Promise<void> | void;
  cancel?: () => Promise<void> | void;
}

export interface SchedulerReaderFactoryContext {
  account: AccountConfig;
  scope: SchedulerScope;
  request: HistoryCollectionRequest;
  mode: SchedulerExecutionMode;
  signal: AbortSignal;
}

export type SchedulerReaderFactory = (
  context: SchedulerReaderFactoryContext,
) => Promise<HistoryReader | SchedulerReaderHandle>;

export interface SchedulerExecutionServiceOptions {
  ledger: Ledger;
  account: AccountConfig;
  scope: SchedulerScope;
  ownerId: string;
  readerFactory: SchedulerReaderFactory;
  scheduler?: SchedulerStore;
  schedule?: ScheduleOptions;
  clock?: () => number;
  leaseDurationMs?: number;
  heartbeatIntervalMs?: number;
}

export type SchedulerExecutionStatus =
  | "idle"
  | "queued"
  | "complete"
  | "partial"
  | "blocked";

export interface SchedulerExecutionResult {
  status: SchedulerExecutionStatus;
  mode: SchedulerExecutionMode | null;
  trigger: SchedulerTrigger | null;
  resumed: boolean;
  coalesced: boolean;
  missedCount: number;
  reconciliationMissedCount: number;
  collection: PersistedHistoryResult | null;
}

export interface SchedulerExecutionState {
  scope: SchedulerScope;
  reconciliationAnchorAt: number;
  reconciliationNextTickIndex: number;
  reconciliationNextDueAt: number;
  reconciliationPendingMissedCount: number;
  pendingMode: SchedulerExecutionMode | null;
  pendingRequestedAt: number | null;
  activeTriggerId: string | null;
  activeFencingToken: number | null;
  activeMode: SchedulerExecutionMode | null;
  activeReconciliationMissedCount: number | null;
  lastReconciliationStartedAt: number | null;
  lastReconciliationFinishedAt: number | null;
  lastReconciliationStatus: "complete" | "partial" | "blocked" | null;
  updatedAt: number;
}

export class SchedulerExecutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SchedulerExecutionError";
  }
}

export class SchedulerLeaseLostError extends SchedulerExecutionError {
  constructor(message = "scheduler execution lease was lost") {
    super(message);
    this.name = "SchedulerLeaseLostError";
  }
}

export class SchedulerExecutionCancelledError extends SchedulerExecutionError {
  constructor(message = "scheduler execution was cancelled") {
    super(message);
    this.name = "SchedulerExecutionCancelledError";
  }
}

interface StoredExecutionRequest {
  mode: SchedulerExecutionMode;
  range?: HistoryRange;
  mappingVersion: string | null;
  collection?: SchedulerCollectionOptions;
}

interface ExecutionRow {
  accountId: string;
  profileId: string;
  reconciliationAnchorAt: number;
  reconciliationNextTickIndex: number;
  reconciliationNextDueAt: number;
  reconciliationPendingMissedCount: number;
  pendingRequestJson: string | null;
  pendingRequestedAt: number | null;
  activeTriggerId: string | null;
  activeFencingToken: number | null;
  activeRequestJson: string | null;
  activeReconciliationMissedCount: number | null;
  lastReconciliationStartedAt: number | null;
  lastReconciliationFinishedAt: number | null;
  lastReconciliationStatus: "complete" | "partial" | "blocked" | null;
  updatedAt: number;
}

interface PreparedExecution {
  lease: LeaseClaim & { acquired: true };
  claim: TriggerClaim;
  request: StoredExecutionRequest;
  reconciliationMissedCount: number;
  coalesced: boolean;
}

interface PrepareResult {
  kind: "claimed" | "queued" | "idle";
  prepared?: PreparedExecution;
  mode: SchedulerExecutionMode | null;
  trigger: SchedulerTrigger | null;
  coalesced: boolean;
}

/**
 * One bounded collection attempt. This class owns coordination and durable
 * execution state; it does not start a daemon, create a browser, or implement
 * request retries and throttling.
 */
export class SchedulerExecutionService {
  readonly scheduler: SchedulerStore;

  private readonly ledger: Ledger;
  private readonly account: AccountConfig;
  private readonly scope: SchedulerScope;
  private readonly ownerId: string;
  private readonly readerFactory: SchedulerReaderFactory;
  private readonly schedule: ScheduleOptions;
  private readonly clock: () => number;
  private readonly leaseDurationMs: number;
  private readonly heartbeatIntervalMs: number;

  constructor(options: SchedulerExecutionServiceOptions) {
    validateNonEmpty(options.ownerId, "ownerId");
    validateNonEmpty(options.scope.accountId, "scope.accountId");
    validateNonEmpty(options.scope.profileId, "scope.profileId");
    if (options.account.id !== options.scope.accountId) {
      throw new SchedulerExecutionError(
        "scheduler scope accountId must match the configured account id",
      );
    }
    if (!options.readerFactory) {
      throw new SchedulerExecutionError("scheduler readerFactory is required");
    }

    this.ledger = options.ledger;
    this.account = options.account;
    this.scope = options.scope;
    this.ownerId = options.ownerId;
    this.readerFactory = options.readerFactory;
    this.schedule = options.schedule ?? {};
    this.clock = options.clock ?? (() => Date.now());
    this.leaseDurationMs = validatePositiveInteger(
      options.leaseDurationMs ?? DEFAULT_LEASE_DURATION_MS,
      "leaseDurationMs",
    );
    this.heartbeatIntervalMs = validatePositiveInteger(
      options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
      "heartbeatIntervalMs",
    );
    if (this.heartbeatIntervalMs >= this.leaseDurationMs) {
      throw new SchedulerExecutionError(
        "heartbeatIntervalMs must be shorter than leaseDurationMs",
      );
    }

    this.scheduler =
      options.scheduler ??
      new SchedulerStore(this.ledger.db, {
        clock: this.clock,
      });
    if (this.scheduler.db !== this.ledger.db) {
      throw new SchedulerExecutionError(
        "scheduler and ledger must share the same SQLite database",
      );
    }
    this.ledger.db.exec(CREATE_EXECUTION_STATE_TABLE);
  }

  async execute(request: SchedulerExecutionRequest): Promise<SchedulerExecutionResult> {
    const normalized = normalizeRequest(request);
    const prepared = this.prepare(normalized);
    if (prepared.kind !== "claimed" || !prepared.prepared) {
      return {
        status: prepared.kind === "queued" ? "queued" : "idle",
        mode: prepared.mode,
        trigger: prepared.trigger,
        resumed: false,
        coalesced: prepared.coalesced,
        missedCount: 0,
        reconciliationMissedCount: 0,
        collection: null,
      };
    }

    const { lease, claim, request: activeRequest } = prepared.prepared;
    const controller = new AbortController();
    let leaseLost: SchedulerLeaseLostError | null = null;
    let readerHandle: SchedulerReaderHandle | null = null;
    let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
    let cancelPromise: Promise<void> | null = null;
    const externalSignal = request.signal;

    const cancelReader = (): void => {
      if (cancelPromise !== null || !readerHandle?.cancel) {
        return;
      }
      cancelPromise = Promise.resolve()
        .then(() => readerHandle!.cancel!())
        .catch(() => {});
    };

    const abortForLeaseLoss = (): void => {
      if (leaseLost === null) {
        leaseLost = new SchedulerLeaseLostError();
      }
      if (!controller.signal.aborted) {
        controller.abort();
      }
      cancelReader();
    };

    const abortForCancellation = (): void => {
      if (!controller.signal.aborted) {
        controller.abort();
      }
      cancelReader();
    };

    const onExternalAbort = (): void => {
      abortForCancellation();
    };
    if (externalSignal?.aborted) {
      abortForCancellation();
    } else {
      externalSignal?.addEventListener("abort", onExternalAbort, { once: true });
    }

    const assertCanProceed = (): void => {
      if (leaseLost) {
        throw leaseLost;
      }
      if (externalSignal?.aborted || controller.signal.aborted) {
        throw new SchedulerExecutionCancelledError();
      }
      try {
        this.assertLease(lease.lease.fencingToken);
      } catch (error) {
        if (error instanceof SchedulerLeaseLostError) {
          abortForLeaseLoss();
        }
        throw error;
      }
    };

    const heartbeat = (): void => {
      if (controller.signal.aborted || leaseLost) {
        return;
      }
      try {
        const mutation = this.scheduler.heartbeatLease(
          this.scope,
          this.ownerId,
          lease.lease.fencingToken,
          { leaseDurationMs: this.leaseDurationMs },
        );
        if (!mutation.applied) {
          abortForLeaseLoss();
        }
      } catch {
        abortForLeaseLoss();
      }
    };

    heartbeatTimer = setInterval(heartbeat, this.heartbeatIntervalMs);
    try {
      assertCanProceed();
      const historyRequest = toHistoryRequest(activeRequest, this.clock());
      const created = await this.readerFactory({
        account: this.account,
        scope: this.scope,
        request: historyRequest,
        mode: activeRequest.mode,
        signal: controller.signal,
      });
      readerHandle = normalizeReaderHandle(created);
      assertCanProceed();

      const hooks: CollectionExecutionHooks = {
        beforeRead: assertCanProceed,
        beforeWrite: assertCanProceed,
      };
      const collection = await collectIntoLedger(
        readerHandle.reader,
        this.ledger,
        this.account,
        historyRequest,
        activeRequest.mappingVersion,
        hooks,
      );
      assertCanProceed();
      this.complete(
        lease.lease.fencingToken,
        claim,
        activeRequest,
        collection,
      );

      return {
        status: collection.status,
        mode: activeRequest.mode,
        trigger: claim.trigger,
        resumed: claim.resumed,
        coalesced: prepared.prepared.coalesced,
        missedCount: claim.trigger.missedCount,
        reconciliationMissedCount:
          prepared.prepared.reconciliationMissedCount,
        collection,
      };
    } finally {
      if (heartbeatTimer !== null) {
        clearInterval(heartbeatTimer);
      }
      externalSignal?.removeEventListener("abort", onExternalAbort);
      if (
        (leaseLost !== null ||
          externalSignal?.aborted ||
          controller.signal.aborted) &&
        readerHandle?.cancel &&
        cancelPromise === null
      ) {
        cancelReader();
      }
      try {
        if (cancelPromise !== null) {
          await cancelPromise;
        }
        if (readerHandle?.close) {
          await readerHandle.close();
        }
      } finally {
        this.scheduler.releaseLease(
          this.scope,
          this.ownerId,
          lease.lease.fencingToken,
        );
      }
    }
  }

  executeScheduled(
    options: Omit<SchedulerExecutionRequest, "mode"> = {},
  ): Promise<SchedulerExecutionResult> {
    return this.execute({ ...options, mode: "scheduled" });
  }

  executeManual(
    options: Omit<SchedulerExecutionRequest, "mode"> = {},
  ): Promise<SchedulerExecutionResult> {
    return this.execute({ ...options, mode: "manual" });
  }

  executeBackfill(
    options: Omit<SchedulerExecutionRequest, "mode"> = {},
  ): Promise<SchedulerExecutionResult> {
    return this.execute({ ...options, mode: "backfill" });
  }

  executeReconciliation(
    options: Omit<SchedulerExecutionRequest, "mode"> = {},
  ): Promise<SchedulerExecutionResult> {
    return this.execute({ ...options, mode: "reconciliation" });
  }

  getExecutionState(): SchedulerExecutionState | null {
    const row = this.readExecutionState();
    return row ? toPublicState(row) : null;
  }

  private prepare(request: StoredExecutionRequest): PrepareResult {
    let result: PrepareResult | undefined;
    this.ledger.transaction(() => {
      const now = this.clock();
      let state = this.ensureExecutionState(now);
      this.scheduler.ensureSchedule(this.scope, this.schedule);

      let coalesced = false;
      if (request.mode !== "scheduled") {
        const queued = this.queuePending(state, request, now);
        state = queued.state;
        coalesced = queued.coalesced;
        const refresh = this.scheduler.requestRefresh(this.scope);
        coalesced ||= refresh.coalesced;
      }

      const materialized = this.materializeReconciliation(state, now);
      state = materialized.state;
      if (
        state.pendingRequestJson !== null ||
        state.reconciliationPendingMissedCount > 0
      ) {
        if (state.pendingRequestJson === null) {
          state = this.setPendingRequest(
            state,
            {
              mode: "reconciliation",
              mappingVersion: null,
            },
            now,
          );
        }
        const refresh = this.scheduler.requestRefresh(this.scope);
        coalesced ||= refresh.coalesced;
      }

      const lease = this.scheduler.claimLease(this.scope, this.ownerId, {
        leaseDurationMs: this.leaseDurationMs,
      });
      if (!lease.acquired) {
        result = {
          kind: "queued",
          mode: request.mode,
          trigger: null,
          coalesced: true,
        };
        return;
      }

      const claim = this.scheduler.claimTrigger(
        this.scope,
        this.ownerId,
        lease.lease.fencingToken,
      );
      if (!claim) {
        this.scheduler.releaseLease(
          this.scope,
          this.ownerId,
          lease.lease.fencingToken,
        );
        result = {
          kind: "idle",
          mode: null,
          trigger: null,
          coalesced,
        };
        return;
      }

      state = this.readExecutionStateOrThrow();
      const activeRequest = claim.resumed
        ? parseStoredRequest(state.activeRequestJson) ??
          inferRequestFromTrigger(claim.trigger)
        : parseStoredRequest(state.pendingRequestJson) ??
          inferRequestFromTrigger(claim.trigger);
      const reconciliationMissedCount =
        activeRequest.mode === "reconciliation"
          ? claim.resumed
            ? Math.max(
                1,
                state.activeReconciliationMissedCount ??
                  state.reconciliationPendingMissedCount,
              )
            : Math.max(1, state.reconciliationPendingMissedCount)
          : 0;
      this.assertLease(lease.lease.fencingToken);
      if (!claim.resumed) {
        this.writeActiveRequest(
          state,
          activeRequest,
          reconciliationMissedCount,
          claim.trigger.triggerId,
          lease.lease.fencingToken,
          now,
        );
      } else if (state.activeRequestJson === null) {
        this.writeActiveRequest(
          state,
          activeRequest,
          reconciliationMissedCount,
          claim.trigger.triggerId,
          lease.lease.fencingToken,
          now,
        );
      } else {
        this.rebindActiveRequest(
          claim.trigger.triggerId,
          lease.lease.fencingToken,
          now,
        );
      }
      result = {
        kind: "claimed",
        prepared: {
          lease: lease as LeaseClaim & { acquired: true },
          claim,
          request: activeRequest,
          reconciliationMissedCount,
          coalesced,
        },
        mode: activeRequest.mode,
        trigger: claim.trigger,
        coalesced,
      };
    });
    if (!result) {
      throw new SchedulerExecutionError("scheduler execution was not prepared");
    }
    return result;
  }

  private complete(
    fencingToken: number,
    claim: TriggerClaim,
    request: StoredExecutionRequest,
    collection: PersistedHistoryResult,
  ): void {
    this.ledger.transaction(() => {
      this.assertLease(fencingToken);
      const completed = this.scheduler.completeTrigger(
        this.scope,
        this.ownerId,
        fencingToken,
      );
      if (!completed.applied) {
        throw new SchedulerLeaseLostError();
      }
      this.assertLease(fencingToken);
      const now = this.clock();
      const result = this.ledger.db
        .prepare(`
          UPDATE chatgpt_scheduler_execution
          SET active_trigger_id=NULL,
              active_fencing_token=NULL,
              active_request_json=NULL,
              active_reconciliation_missed_count=NULL,
              last_reconciliation_finished_at=CASE
                WHEN json_extract(active_request_json, '$.mode')='reconciliation'
                THEN ?
                ELSE last_reconciliation_finished_at
              END,
              last_reconciliation_status=CASE
                WHEN json_extract(active_request_json, '$.mode')='reconciliation'
                THEN ?
                ELSE last_reconciliation_status
              END,
              updated_at=?
          WHERE account_id=? AND profile_id=?
            AND active_trigger_id=?
            AND active_fencing_token=?
        `)
        .run(
          now,
          collection.status,
          now,
          this.scope.accountId,
          this.scope.profileId,
          claim.trigger.triggerId,
          fencingToken,
        );
      if (result.changes !== 1) {
        throw new SchedulerLeaseLostError(
          "active scheduler execution was fenced before completion",
        );
      }
      if (request.mode === "reconciliation") {
        this.assertLease(fencingToken);
      }
    });
  }

  private queuePending(
    state: ExecutionRow,
    request: StoredExecutionRequest,
    now: number,
  ): { state: ExecutionRow; coalesced: boolean } {
    const existing = parseStoredRequest(state.pendingRequestJson);
    const merged = mergePendingRequest(existing, request);
    const coalesced = existing !== null;
    const next = this.updateExecutionRow(
      `
        pending_request_json=?,
        pending_requested_at=COALESCE(pending_requested_at, ?),
        reconciliation_pending_missed_count=?,
        updated_at=?
      `,
      [
        JSON.stringify(merged),
        state.pendingRequestedAt ?? now,
        state.reconciliationPendingMissedCount,
        now,
      ],
    );
    return { state: next, coalesced };
  }

  private materializeReconciliation(
    state: ExecutionRow,
    now: number,
  ): { state: ExecutionRow; dueCount: number } {
    const dueCount =
      state.reconciliationNextDueAt > now
        ? 0
        : Math.floor(
            (now - state.reconciliationNextDueAt) /
              RECONCILIATION_INTERVAL_MS,
          ) + 1;
    if (!Number.isSafeInteger(dueCount) || dueCount < 0) {
      throw new SchedulerExecutionError(
        "reconciliation due count exceeds safe integer range",
      );
    }
    if (dueCount === 0) {
      return { state, dueCount };
    }
    const nextTickIndex =
      state.reconciliationNextTickIndex + dueCount;
    const nextDueAt =
      state.reconciliationAnchorAt +
      nextTickIndex * RECONCILIATION_INTERVAL_MS;
    if (!Number.isSafeInteger(nextDueAt)) {
      throw new SchedulerExecutionError(
        "reconciliation due time exceeds safe integer range",
      );
    }
    const next = this.updateExecutionRow(
      `
        reconciliation_next_tick_index=?,
        reconciliation_next_due_at=?,
        reconciliation_pending_missed_count=
          reconciliation_pending_missed_count + ?,
        updated_at=?
      `,
      [
        nextTickIndex,
        nextDueAt,
        dueCount,
        now,
      ],
    );
    return { state: next, dueCount };
  }

  private ensureExecutionState(now: number): ExecutionRow {
    const existing = this.readExecutionState();
    if (existing) {
      return existing;
    }
    this.ledger.db
      .prepare(`
        INSERT INTO chatgpt_scheduler_execution(
          account_id,
          profile_id,
          reconciliation_anchor_at,
          reconciliation_next_tick_index,
          reconciliation_next_due_at,
          reconciliation_pending_missed_count,
          updated_at
        ) VALUES (?, ?, ?, 1, ?, 0, ?)
      `)
      .run(
        this.scope.accountId,
        this.scope.profileId,
        now,
        now + RECONCILIATION_INTERVAL_MS,
        now,
      );
    return this.readExecutionStateOrThrow();
  }

  private setPendingRequest(
    state: ExecutionRow,
    request: StoredExecutionRequest,
    now: number,
  ): ExecutionRow {
    return this.updateExecutionRow(
      `
        pending_request_json=?,
        pending_requested_at=COALESCE(pending_requested_at, ?),
        updated_at=?
      `,
      [JSON.stringify(request), now, now],
    );
  }

  private writeActiveRequest(
    state: ExecutionRow,
    request: StoredExecutionRequest,
    reconciliationMissedCount: number,
    triggerId: string,
    fencingToken: number,
    now: number,
  ): void {
    const pendingReconciliationMissedCount =
      request.mode === "reconciliation"
        ? 0
        : state.reconciliationPendingMissedCount;
    const result = this.ledger.db
      .prepare(`
        UPDATE chatgpt_scheduler_execution
        SET pending_request_json=NULL,
            pending_requested_at=NULL,
            reconciliation_pending_missed_count=?,
            active_trigger_id=?,
            active_fencing_token=?,
            active_request_json=?,
            active_reconciliation_missed_count=?,
            last_reconciliation_started_at=CASE
              WHEN ?='reconciliation' THEN ?
              ELSE last_reconciliation_started_at
            END,
            updated_at=?
        WHERE account_id=? AND profile_id=?
          AND active_trigger_id IS NULL
      `)
      .run(
        pendingReconciliationMissedCount,
        triggerId,
        fencingToken,
        JSON.stringify(request),
        request.mode === "reconciliation" ? reconciliationMissedCount : null,
        request.mode,
        now,
        now,
        this.scope.accountId,
        this.scope.profileId,
      );
    if (result.changes !== 1) {
      throw new SchedulerExecutionError(
        "scheduler execution state changed before trigger activation",
      );
    }
  }

  private rebindActiveRequest(
    triggerId: string,
    fencingToken: number,
    now: number,
  ): void {
    const result = this.ledger.db
      .prepare(`
        UPDATE chatgpt_scheduler_execution
        SET active_fencing_token=?,
            updated_at=?
        WHERE account_id=? AND profile_id=?
          AND active_trigger_id=?
          AND active_request_json IS NOT NULL
      `)
      .run(
        fencingToken,
        now,
        this.scope.accountId,
        this.scope.profileId,
        triggerId,
      );
    if (result.changes !== 1) {
      throw new SchedulerExecutionError(
        "resumed scheduler execution state was not fenced to the current lease",
      );
    }
  }

  private updateExecutionRow(setClause: string, values: unknown[]): ExecutionRow {
    this.ledger.db
      .prepare(`
        UPDATE chatgpt_scheduler_execution
        SET ${setClause}
        WHERE account_id=? AND profile_id=?
      `)
      .run(...values, this.scope.accountId, this.scope.profileId);
    return this.readExecutionStateOrThrow();
  }

  private readExecutionState(): ExecutionRow | null {
    const row = this.ledger.db
      .prepare(`
        SELECT *
        FROM chatgpt_scheduler_execution
        WHERE account_id=? AND profile_id=?
      `)
      .get(this.scope.accountId, this.scope.profileId) as Record<string, unknown> | undefined;
    return row ? executionRowFromSql(row) : null;
  }

  private readExecutionStateOrThrow(): ExecutionRow {
    const state = this.readExecutionState();
    if (!state) {
      throw new SchedulerExecutionError(
        `execution state not found for ${this.scope.accountId}/${this.scope.profileId}`,
      );
    }
    return state;
  }

  private assertLease(fencingToken: number): void {
    const lease = this.scheduler.getLease(this.scope);
    const now = this.clock();
    if (
      !lease ||
      lease.ownerId !== this.ownerId ||
      lease.fencingToken !== fencingToken ||
      lease.leaseUntilAt <= now
    ) {
      throw new SchedulerLeaseLostError(
        `lease is not held by owner ${this.ownerId} for ` +
          `${this.scope.accountId}/${this.scope.profileId}`,
      );
    }
  }
}

export { SchedulerExecutionService as SchedulerRunner };

function normalizeRequest(
  request: SchedulerExecutionRequest,
): StoredExecutionRequest {
  if (
    request.mode === "reconciliation" &&
    request.range !== undefined
  ) {
    throw new SchedulerExecutionError(
      "reconciliation uses the collector-owned default range and outstanding work",
    );
  }
  const normalized: StoredExecutionRequest = {
    mode: request.mode,
    mappingVersion: request.mappingVersion ?? null,
  };
  if (request.range !== undefined) {
    normalized.range = request.range;
  }
  if (request.collection !== undefined) {
    normalized.collection = request.collection;
  }
  return normalized;
}

function toHistoryRequest(
  request: StoredExecutionRequest,
  now: number,
): HistoryCollectionRequest {
  const mode: HistoryCollectionRequest["mode"] =
    request.mode === "backfill"
      ? "backfill"
      : request.mode === "reconciliation"
        ? "reconciliation"
        : "incremental";
  const historyRequest: HistoryCollectionRequest = {
    ...(request.collection ?? {}),
    mode,
    now: new Date(now),
  };
  if (request.range !== undefined) {
    historyRequest.range = request.range;
  }
  return historyRequest;
}

function normalizeReaderHandle(
  created: HistoryReader | SchedulerReaderHandle,
): SchedulerReaderHandle {
  if (
    typeof created === "object" &&
    created !== null &&
    "reader" in created
  ) {
    return created;
  }
  const candidate = created as HistoryReader & {
    close?: () => Promise<void> | void;
    cancel?: () => Promise<void> | void;
  };
  const handle: SchedulerReaderHandle = { reader: candidate };
  if (typeof candidate.close === "function") {
    handle.close = () => candidate.close!();
  }
  if (typeof candidate.cancel === "function") {
    handle.cancel = () => candidate.cancel!();
  }
  return handle;
}

function mergePendingRequest(
  existing: StoredExecutionRequest | null,
  incoming: StoredExecutionRequest,
): StoredExecutionRequest {
  if (!existing) {
    return incoming;
  }
  if (existing.mode === "backfill" && incoming.mode === "backfill") {
    const merged: StoredExecutionRequest = {
      mode: "backfill",
      mappingVersion: incoming.mappingVersion ?? existing.mappingVersion,
    };
    const range = mergeRanges(existing.range, incoming.range);
    if (range !== undefined) {
      merged.range = range;
    }
    if (incoming.collection !== undefined) {
      merged.collection = incoming.collection;
    } else if (existing.collection !== undefined) {
      merged.collection = existing.collection;
    }
    return merged;
  }
  if (incoming.mode === "backfill") {
    return mergePendingRequest(incoming, existing.mode === "backfill" ? existing : incoming);
  }
  if (existing.mode === "backfill") {
    return existing;
  }
  if (existing.mode === "reconciliation" || incoming.mode === "reconciliation") {
    return {
      mode: "reconciliation",
      mappingVersion: incoming.mappingVersion ?? existing.mappingVersion,
      ...(incoming.collection ?? existing.collection
        ? { collection: incoming.collection ?? existing.collection }
        : {}),
    };
  }
  return {
    mode: "manual",
    mappingVersion: incoming.mappingVersion ?? existing.mappingVersion,
    ...(incoming.collection ?? existing.collection
      ? { collection: incoming.collection ?? existing.collection }
      : {}),
  };
}

function mergeRanges(
  left: HistoryRange | undefined,
  right: HistoryRange | undefined,
): HistoryRange | undefined {
  if (!left) {
    return right;
  }
  if (!right) {
    return left;
  }
  return {
    start:
      new Date(left.start).getTime() <= new Date(right.start).getTime()
        ? left.start
        : right.start,
    end:
      new Date(left.end).getTime() >= new Date(right.end).getTime()
        ? left.end
        : right.end,
  };
}

function inferRequestFromTrigger(trigger: SchedulerTrigger): StoredExecutionRequest {
  return {
    mode: trigger.kind === "scheduled" ? "scheduled" : "manual",
    mappingVersion: null,
  };
}

function parseStoredRequest(json: string | null): StoredExecutionRequest | null {
  if (json === null) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new SchedulerExecutionError("scheduler execution request is invalid JSON");
  }
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !("mode" in parsed) ||
    (parsed.mode !== "manual" &&
      parsed.mode !== "backfill" &&
      parsed.mode !== "reconciliation" &&
      parsed.mode !== "scheduled")
  ) {
    throw new SchedulerExecutionError("scheduler execution request has an invalid mode");
  }
  const request = parsed as {
    mode: SchedulerExecutionMode;
    range?: HistoryRange;
    mappingVersion?: string | null;
    collection?: SchedulerCollectionOptions;
  };
  const normalized: StoredExecutionRequest = {
    mode: request.mode,
    mappingVersion: request.mappingVersion ?? null,
  };
  if (request.range !== undefined) {
    normalized.range = request.range;
  }
  if (request.collection !== undefined) {
    normalized.collection = request.collection;
  }
  return normalized;
}

function executionRowFromSql(row: Record<string, unknown>): ExecutionRow {
  const mode = (value: unknown): SchedulerExecutionMode | null => {
    if (
      value === "manual" ||
      value === "backfill" ||
      value === "reconciliation" ||
      value === "scheduled"
    ) {
      return value;
    }
    return null;
  };
  const status = (value: unknown): "complete" | "partial" | "blocked" | null => {
    if (value === "complete" || value === "partial" || value === "blocked") {
      return value;
    }
    return null;
  };
  return {
    accountId: stringValue(row.account_id, "account_id"),
    profileId: stringValue(row.profile_id, "profile_id"),
    reconciliationAnchorAt: integer(row.reconciliation_anchor_at, "reconciliation_anchor_at"),
    reconciliationNextTickIndex: integer(
      row.reconciliation_next_tick_index,
      "reconciliation_next_tick_index",
    ),
    reconciliationNextDueAt: integer(
      row.reconciliation_next_due_at,
      "reconciliation_next_due_at",
    ),
    reconciliationPendingMissedCount: integer(
      row.reconciliation_pending_missed_count,
      "reconciliation_pending_missed_count",
    ),
    pendingRequestJson: nullableString(row.pending_request_json),
    pendingRequestedAt: nullableInteger(row.pending_requested_at, "pending_requested_at"),
    activeTriggerId: nullableString(row.active_trigger_id),
    activeFencingToken: nullableInteger(
      row.active_fencing_token,
      "active_fencing_token",
    ),
    activeRequestJson: nullableString(row.active_request_json),
    activeReconciliationMissedCount: nullableInteger(
      row.active_reconciliation_missed_count,
      "active_reconciliation_missed_count",
    ),
    lastReconciliationStartedAt: nullableInteger(
      row.last_reconciliation_started_at,
      "last_reconciliation_started_at",
    ),
    lastReconciliationFinishedAt: nullableInteger(
      row.last_reconciliation_finished_at,
      "last_reconciliation_finished_at",
    ),
    lastReconciliationStatus: status(row.last_reconciliation_status),
    updatedAt: integer(row.updated_at, "updated_at"),
  };
}

function toPublicState(row: ExecutionRow): SchedulerExecutionState {
  return {
    scope: {
      accountId: row.accountId,
      profileId: row.profileId,
    },
    reconciliationAnchorAt: row.reconciliationAnchorAt,
    reconciliationNextTickIndex: row.reconciliationNextTickIndex,
    reconciliationNextDueAt: row.reconciliationNextDueAt,
    reconciliationPendingMissedCount: row.reconciliationPendingMissedCount,
    pendingMode: parseStoredRequest(row.pendingRequestJson)?.mode ?? null,
    pendingRequestedAt: row.pendingRequestedAt,
    activeTriggerId: row.activeTriggerId,
    activeFencingToken: row.activeFencingToken,
    activeMode: parseStoredRequest(row.activeRequestJson)?.mode ?? null,
    activeReconciliationMissedCount: row.activeReconciliationMissedCount,
    lastReconciliationStartedAt: row.lastReconciliationStartedAt,
    lastReconciliationFinishedAt: row.lastReconciliationFinishedAt,
    lastReconciliationStatus: row.lastReconciliationStatus,
    updatedAt: row.updatedAt,
  };
}

function validateNonEmpty(value: string, label: string): void {
  if (typeof value !== "string" || value.trim() === "") {
    throw new SchedulerExecutionError(`${label} must be a non-empty string`);
  }
}

function validatePositiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new SchedulerExecutionError(`${label} must be a positive safe integer`);
  }
  return value;
}

function integer(value: unknown, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) {
    throw new SchedulerExecutionError(
      `scheduler execution column ${label} is not a safe integer`,
    );
  }
  return parsed;
}

function nullableInteger(value: unknown, label: string): number | null {
  return value === null || value === undefined ? null : integer(value, label);
}

function nullableString(value: unknown): string | null {
  return value === null || value === undefined ? null : String(value);
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new SchedulerExecutionError(
      `scheduler execution column ${label} is not a string`,
    );
  }
  return value;
}
