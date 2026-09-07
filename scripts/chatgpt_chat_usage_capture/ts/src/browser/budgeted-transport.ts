/**
 * Per-run request budgeting for the read-only history transport.
 *
 * This wrapper serializes all application-issued reads for one collection run.
 * It never changes response payloads, including authentication and rate-limit
 * responses; the adapter and collector retain ownership of those semantics.
 */

import {
  AdapterError,
  assertAllowedRequest,
  LEGACY_DETAIL,
  MODERN_DETAIL,
  MODERN_INDEX,
  MODERN_MESSAGES,
  SESSION_ROUTE,
  type HistoryTransport,
  type HistoryTransportRequestOptions,
} from "../adapters/chatgpt/adapter.js";

export const DEFAULT_CONCURRENT_READS = 1;
export const DEFAULT_MIN_GAP_MS = 1_000;
export const DEFAULT_MAX_ATTEMPTS_PER_RUN = 500;
export const DEFAULT_MAX_RUN_DURATION_MS = 1_200_000;
export const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
export const DEFAULT_TRANSIENT_RETRIES = 2;
export const DEFAULT_RETRY_BACKOFF_BASE_MS = 250;
export const DEFAULT_RETRY_BACKOFF_MAX_MS = 5_000;
export const DEFAULT_RETRY_JITTER_MS = 250;

export type BudgetedRequestClass = "identity" | "index" | "head" | "detail";

export type BudgetedPartialReason =
  | "attempt_budget_exhausted"
  | "run_deadline_exceeded"
  | "request_timeout"
  | "transient_retries_exhausted"
  | "cancelled";

export interface BudgetedTransportOptions {
  concurrentReads?: number;
  minGapMs?: number;
  maxAttemptsPerRun?: number;
  maxRunDurationMs?: number;
  requestTimeoutMs?: number;
  transientRetries?: number;
  retryBackoffBaseMs?: number;
  retryBackoffMaxMs?: number;
  retryJitterMs?: number;
  clock?: () => number;
  sleep?: (durationMs: number) => Promise<void>;
  random?: () => number;
}

export interface BudgetedTransportConfig {
  readonly concurrentReads: number;
  readonly minGapMs: number;
  readonly maxAttemptsPerRun: number;
  readonly maxRunDurationMs: number;
  readonly requestTimeoutMs: number;
  readonly transientRetries: number;
  readonly retryBackoffBaseMs: number;
  readonly retryBackoffMaxMs: number;
  readonly retryJitterMs: number;
}

export interface BudgetedReadCounts {
  identity: number;
  index: number;
  head: number;
  detail: number;
}

export interface BudgetedTransportDiagnostics {
  readonly runStartedAt: number;
  readonly runDeadlineAt: number;
  readonly attempts: number;
  readonly retries: number;
  readonly reads: number;
  readonly readCounts: BudgetedReadCounts;
  readonly attemptsByClass: BudgetedReadCounts;
  readonly retriesByClass: BudgetedReadCounts;
  readonly lastStatus: number | null;
  readonly blocked: boolean;
  readonly blockedReason: BudgetedPartialReason | null;
}

export class BudgetedTransportConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BudgetedTransportConfigError";
  }
}

export class BudgetedTransportPartialError extends Error {
  readonly reason: BudgetedPartialReason;
  readonly requestClass: BudgetedRequestClass | null;
  readonly attempts: number;
  readonly maxAttemptsPerRun: number;
  readonly runStartedAt: number;
  readonly runDeadlineAt: number;
  readonly blocksLaterReads: boolean;

  constructor(
    message: string,
    options: {
      reason: BudgetedPartialReason;
      requestClass?: BudgetedRequestClass | null;
      attempts: number;
      maxAttemptsPerRun: number;
      runStartedAt: number;
      runDeadlineAt: number;
      blocksLaterReads: boolean;
    },
  ) {
    super(message);
    this.name = "BudgetedTransportPartialError";
    this.reason = options.reason;
    this.requestClass = options.requestClass ?? null;
    this.attempts = options.attempts;
    this.maxAttemptsPerRun = options.maxAttemptsPerRun;
    this.runStartedAt = options.runStartedAt;
    this.runDeadlineAt = options.runDeadlineAt;
    this.blocksLaterReads = options.blocksLaterReads;
  }
}

class BudgetedAttemptTimeoutError extends Error {
  constructor(readonly deadline: boolean) {
    super(deadline ? "request reached the run deadline" : "request timed out");
    this.name = "BudgetedAttemptTimeoutError";
  }
}

interface ActiveOperation {
  readonly controller: AbortController;
  readonly requestClass: BudgetedRequestClass;
  settled: boolean;
}

const DEFAULT_OPTIONS: Required<
  Pick<
    BudgetedTransportOptions,
    | "concurrentReads"
    | "minGapMs"
    | "maxAttemptsPerRun"
    | "maxRunDurationMs"
    | "requestTimeoutMs"
    | "transientRetries"
    | "retryBackoffBaseMs"
    | "retryBackoffMaxMs"
    | "retryJitterMs"
  >
> = {
  concurrentReads: DEFAULT_CONCURRENT_READS,
  minGapMs: DEFAULT_MIN_GAP_MS,
  maxAttemptsPerRun: DEFAULT_MAX_ATTEMPTS_PER_RUN,
  maxRunDurationMs: DEFAULT_MAX_RUN_DURATION_MS,
  requestTimeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
  transientRetries: DEFAULT_TRANSIENT_RETRIES,
  retryBackoffBaseMs: DEFAULT_RETRY_BACKOFF_BASE_MS,
  retryBackoffMaxMs: DEFAULT_RETRY_BACKOFF_MAX_MS,
  retryJitterMs: DEFAULT_RETRY_JITTER_MS,
};

export class BudgetedTransport implements HistoryTransport {
  readonly config: BudgetedTransportConfig;

  private readonly clock: () => number;
  private readonly sleep: (durationMs: number) => Promise<void>;
  private readonly random: () => number;
  private readonly runStartedAt: number;
  private readonly runDeadlineAt: number;
  private readonly readCounts: BudgetedReadCounts = emptyCounts();
  private readonly attemptsByClass: BudgetedReadCounts = emptyCounts();
  private readonly retriesByClass: BudgetedReadCounts = emptyCounts();

  private queueTail: Promise<void> = Promise.resolve();
  private activeOperation: ActiveOperation | null = null;
  private blockedError: BudgetedTransportPartialError | null = null;
  private closePromise: Promise<void> | null = null;
  private pendingReads = 0;
  private closed = false;
  private attempts = 0;
  private retries = 0;
  private reads = 0;
  private lastAttemptStartedAt: number | null = null;
  private lastStatus: number | null = null;

  constructor(
    private readonly transport: HistoryTransport,
    options: BudgetedTransportOptions = {},
  ) {
    if (
      typeof transport.cancel !== "function" &&
      typeof transport.close !== "function"
    ) {
      throw new BudgetedTransportConfigError(
        "transport must expose cancel() or close() for safe request cancellation",
      );
    }
    this.config = validateConfig(options);
    this.clock = options.clock ?? (() => Date.now());
    this.sleep = options.sleep ?? defaultSleep;
    this.random = options.random ?? Math.random;
    this.runStartedAt = readClock(this.clock);
    this.runDeadlineAt = this.runStartedAt + this.config.maxRunDurationMs;
    if (!Number.isSafeInteger(this.runDeadlineAt)) {
      throw new BudgetedTransportConfigError(
        "run deadline must be a safe integer timestamp",
      );
    }
  }

  async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
    _options?: HistoryTransportRequestOptions,
  ): Promise<Record<string, unknown>> {
    assertAllowedRequest(method, path);
    const requestClass = classifyRequest(path, params);
    if (this.blockedError) {
      return Promise.reject(this.blockedErrorFor(requestClass));
    }
    if (this.closed) {
      return Promise.reject(
        this.makePartialError("cancelled", requestClass, true),
      );
    }
    this.pendingReads += 1;
    const operation = this.queueTail
      .then(() =>
        this.executeRead(method.toUpperCase(), path, params, requestClass),
      )
      .finally(() => {
        this.pendingReads -= 1;
      });
    this.queueTail = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }

  async cancel(): Promise<void> {
    if (this.blockedError === null) {
      this.blockedError = this.makePartialError(
        "cancelled",
        this.activeOperation?.requestClass ?? null,
        false,
      );
    }
    const active = this.activeOperation;
    let cancellationError: unknown = null;
    if (active) {
      active.controller.abort();
      try {
        await this.cancelUnderlying();
      } catch (error) {
        cancellationError = error;
      }
    }
    await this.queueTail;
    if (cancellationError) {
      throw cancellationError;
    }
  }

  async close(): Promise<void> {
    if (this.closePromise) {
      return this.closePromise;
    }
    this.closePromise = (async () => {
      let cancellationError: unknown = null;
      if (this.pendingReads > 0) {
        try {
          await this.cancel();
        } catch (error) {
          cancellationError = error;
        }
      } else {
        await this.queueTail;
      }
      let closeError: unknown = null;
      try {
        if (typeof this.transport.close === "function") {
          await this.transport.close();
        }
        this.closed = true;
      } catch (error) {
        closeError = error;
      }
      if (cancellationError) {
        throw cancellationError;
      }
      if (closeError) {
        throw closeError;
      }
    })();
    return this.closePromise;
  }

  getDiagnostics(): BudgetedTransportDiagnostics {
    return {
      runStartedAt: this.runStartedAt,
      runDeadlineAt: this.runDeadlineAt,
      attempts: this.attempts,
      retries: this.retries,
      reads: this.reads,
      readCounts: { ...this.readCounts },
      attemptsByClass: { ...this.attemptsByClass },
      retriesByClass: { ...this.retriesByClass },
      lastStatus: this.lastStatus,
      blocked: this.blockedError !== null,
      blockedReason: this.blockedError?.reason ?? null,
    };
  }

  private async executeRead(
    method: string,
    path: string,
    params: Record<string, unknown>,
    requestClass: BudgetedRequestClass,
  ): Promise<Record<string, unknown>> {
    this.reads += 1;
    this.readCounts[requestClass] += 1;
    let retryIndex = 0;
    let nextAttemptNotBefore: number | null = null;

    while (true) {
      await this.waitForAttemptSlot(requestClass, nextAttemptNotBefore);
      nextAttemptNotBefore = null;
      this.assertCanRead(requestClass);
    this.attempts += 1;
      this.attemptsByClass[requestClass] += 1;
      if (retryIndex > 0) {
        this.retries += 1;
        this.retriesByClass[requestClass] += 1;
      }
      this.lastAttemptStartedAt = readClock(this.clock);

      try {
        const response = await this.issueAttempt(
          method,
          path,
          params,
          requestClass,
        );
        this.lastStatus = readStatus(response);
        if (
          !isTransientStatus(this.lastStatus) ||
          retryIndex >= this.config.transientRetries
        ) {
          return response;
        }
        retryIndex += 1;
        nextAttemptNotBefore = this.retryNotBefore(retryIndex - 1);
      } catch (error) {
        if (this.blockedError) {
          throw this.blockedErrorFor(requestClass);
        }
        const status = statusFromError(error);
        if (
          status === 401 ||
          status === 403 ||
          status === 429 ||
          !isRetryableError(error)
        ) {
          throw error;
        }
        if (retryIndex >= this.config.transientRetries) {
          throw this.makePartialError(
            error instanceof BudgetedAttemptTimeoutError
              ? "request_timeout"
              : "transient_retries_exhausted",
            requestClass,
            false,
          );
        }
        retryIndex += 1;
        nextAttemptNotBefore = this.retryNotBefore(retryIndex - 1);
      }
    }
  }

  private async waitForAttemptSlot(
    requestClass: BudgetedRequestClass,
    nextAttemptNotBefore: number | null,
  ): Promise<void> {
    this.assertCanRead(requestClass);
    const now = readClock(this.clock);
    const gapNotBefore =
      this.lastAttemptStartedAt === null
        ? now
        : this.lastAttemptStartedAt + this.config.minGapMs;
    const target = Math.max(now, gapNotBefore, nextAttemptNotBefore ?? now);
    if (target <= now) {
      return;
    }
    const remaining = this.runDeadlineAt - now;
    if (remaining <= 0) {
      throw this.blockRun("run_deadline_exceeded", requestClass);
    }
    await this.sleep(Math.min(target - now, remaining));
    this.assertCanRead(requestClass);
    if (readClock(this.clock) < target) {
      return;
    }
    if (readClock(this.clock) >= this.runDeadlineAt) {
      throw this.blockRun("run_deadline_exceeded", requestClass);
    }
  }

  private async issueAttempt(
    method: string,
    path: string,
    params: Record<string, unknown>,
    requestClass: BudgetedRequestClass,
  ): Promise<Record<string, unknown>> {
    const now = readClock(this.clock);
    const remaining = this.runDeadlineAt - now;
    if (remaining <= 0) {
      throw this.blockRun("run_deadline_exceeded", requestClass);
    }
    const timeoutMs = Math.min(this.config.requestTimeoutMs, remaining);
    const controller = new AbortController();
    const operation: ActiveOperation = {
      controller,
      requestClass,
      settled: false,
    };
    this.activeOperation = operation;
    let timeoutTriggered = false;
    const deadlineTimeout = remaining <= this.config.requestTimeoutMs;
    let requestPromise: Promise<Record<string, unknown>>;
    try {
      requestPromise = this.transport.request(method, path, params, {
        signal: controller.signal,
      });
    } catch (error) {
      requestPromise = Promise.reject(error);
    }
    void this.sleep(timeoutMs).then(
      () => {
        if (operation.settled) {
          return;
        }
        timeoutTriggered = true;
        controller.abort();
        void this.cancelUnderlying().catch(() => undefined);
      },
      () => {
        if (operation.settled) {
          return;
        }
        timeoutTriggered = true;
        controller.abort();
        void this.cancelUnderlying().catch(() => undefined);
      },
    );
    try {
      const response = await requestPromise;
      if (timeoutTriggered) {
        throw new BudgetedAttemptTimeoutError(deadlineTimeout);
      }
      return response;
    } catch (error) {
      if (timeoutTriggered) {
        if (deadlineTimeout) {
          throw this.blockRun("run_deadline_exceeded", requestClass);
        }
        throw new BudgetedAttemptTimeoutError(false);
      }
      throw error;
    } finally {
      operation.settled = true;
      if (this.activeOperation === operation) {
        this.activeOperation = null;
      }
    }
  }

  private retryNotBefore(retryIndex: number): number {
    const exponential = Math.min(
      this.config.retryBackoffMaxMs,
      this.config.retryBackoffBaseMs * 2 ** retryIndex,
    );
    const randomValue = this.random();
    if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
      throw new BudgetedTransportConfigError(
        "random must return a finite number in the range [0, 1)",
      );
    }
    const jitter = Math.min(
      this.config.retryBackoffMaxMs - exponential,
      this.config.retryJitterMs * randomValue,
    );
    return readClock(this.clock) + exponential + Math.max(0, jitter);
  }

  private assertCanRead(requestClass: BudgetedRequestClass): void {
    if (this.blockedError) {
      throw this.blockedErrorFor(requestClass);
    }
    if (this.attempts >= this.config.maxAttemptsPerRun) {
      throw this.blockRun("attempt_budget_exhausted", requestClass);
    }
    if (readClock(this.clock) >= this.runDeadlineAt) {
      throw this.blockRun("run_deadline_exceeded", requestClass);
    }
  }

  private blockRun(
    reason: "attempt_budget_exhausted" | "run_deadline_exceeded",
    requestClass: BudgetedRequestClass,
  ): BudgetedTransportPartialError {
    if (this.blockedError === null) {
      this.blockedError = this.makePartialError(reason, requestClass, true);
    }
    return this.blockedErrorFor(requestClass);
  }

  private blockedErrorFor(
    requestClass: BudgetedRequestClass,
  ): BudgetedTransportPartialError {
    if (!this.blockedError) {
      throw new BudgetedTransportConfigError("transport is not blocked");
    }
    return new BudgetedTransportPartialError(
      "history collection is partially complete and further reads are blocked",
      {
        reason: this.blockedError.reason,
        requestClass,
        attempts: this.blockedError.attempts,
        maxAttemptsPerRun: this.blockedError.maxAttemptsPerRun,
        runStartedAt: this.blockedError.runStartedAt,
        runDeadlineAt: this.blockedError.runDeadlineAt,
        blocksLaterReads: true,
      },
    );
  }

  private makePartialError(
    reason: BudgetedPartialReason,
    requestClass: BudgetedRequestClass | null,
    blocksLaterReads: boolean,
  ): BudgetedTransportPartialError {
    return new BudgetedTransportPartialError(
      "history collection request budget produced partial coverage",
      {
        reason,
        requestClass,
        attempts: this.attempts,
        maxAttemptsPerRun: this.config.maxAttemptsPerRun,
        runStartedAt: this.runStartedAt,
        runDeadlineAt: this.runDeadlineAt,
        blocksLaterReads,
      },
    );
  }

  private async cancelUnderlying(): Promise<void> {
    if (typeof this.transport.cancel === "function") {
      await this.transport.cancel();
      return;
    }
    if (typeof this.transport.close === "function") {
      await this.transport.close();
      return;
    }
    throw new BudgetedTransportConfigError(
      "transport cancellation capability disappeared during a request",
    );
  }
}

function validateConfig(
  options: BudgetedTransportOptions,
): BudgetedTransportConfig {
  const config: BudgetedTransportConfig = {
    concurrentReads: positiveInteger(
      options.concurrentReads ?? DEFAULT_OPTIONS.concurrentReads,
      "concurrentReads",
    ),
    minGapMs: nonNegativeFinite(
      options.minGapMs ?? DEFAULT_OPTIONS.minGapMs,
      "minGapMs",
    ),
    maxAttemptsPerRun: positiveInteger(
      options.maxAttemptsPerRun ?? DEFAULT_OPTIONS.maxAttemptsPerRun,
      "maxAttemptsPerRun",
    ),
    maxRunDurationMs: positiveFinite(
      options.maxRunDurationMs ?? DEFAULT_OPTIONS.maxRunDurationMs,
      "maxRunDurationMs",
    ),
    requestTimeoutMs: positiveFinite(
      options.requestTimeoutMs ?? DEFAULT_OPTIONS.requestTimeoutMs,
      "requestTimeoutMs",
    ),
    transientRetries: nonNegativeInteger(
      options.transientRetries ?? DEFAULT_OPTIONS.transientRetries,
      "transientRetries",
    ),
    retryBackoffBaseMs: nonNegativeFinite(
      options.retryBackoffBaseMs ?? DEFAULT_OPTIONS.retryBackoffBaseMs,
      "retryBackoffBaseMs",
    ),
    retryBackoffMaxMs: nonNegativeFinite(
      options.retryBackoffMaxMs ?? DEFAULT_OPTIONS.retryBackoffMaxMs,
      "retryBackoffMaxMs",
    ),
    retryJitterMs: nonNegativeFinite(
      options.retryJitterMs ?? DEFAULT_OPTIONS.retryJitterMs,
      "retryJitterMs",
    ),
  };
  if (config.concurrentReads !== 1) {
    throw new BudgetedTransportConfigError(
      "concurrentReads must be exactly 1 for account-scoped history reads",
    );
  }
  if (config.retryBackoffMaxMs < config.retryBackoffBaseMs) {
    throw new BudgetedTransportConfigError(
      "retryBackoffMaxMs must be at least retryBackoffBaseMs",
    );
  }
  if (typeof options.clock !== "undefined" && typeof options.clock !== "function") {
    throw new BudgetedTransportConfigError("clock must be a function");
  }
  if (typeof options.sleep !== "undefined" && typeof options.sleep !== "function") {
    throw new BudgetedTransportConfigError("sleep must be a function");
  }
  if (typeof options.random !== "undefined" && typeof options.random !== "function") {
    throw new BudgetedTransportConfigError("random must be a function");
  }
  return config;
}

function classifyRequest(
  path: string,
  params: Record<string, unknown>,
): BudgetedRequestClass {
  if (path === SESSION_ROUTE) {
    return "identity";
  }
  if (path === MODERN_INDEX) {
    const offset = params.offset;
    return offset === undefined || Number(offset) === 0 ? "head" : "index";
  }
  if (
    path.startsWith(MODERN_DETAIL.slice(0, -"{conversation_id}".length)) ||
    path.startsWith(MODERN_MESSAGES.slice(0, -"{conversation_id}".length)) ||
    path.startsWith(LEGACY_DETAIL.slice(0, -"{conversation_id}".length))
  ) {
    return "detail";
  }
  throw new AdapterError(`request path was not classifiable: ${path}`);
}

function isTransientStatus(status: number | null): boolean {
  return status === 408 || (status !== null && status >= 500 && status <= 599);
}

function isRetryableError(error: unknown): boolean {
  return !(error instanceof BudgetedTransportPartialError) &&
    !(error instanceof BudgetedTransportConfigError) &&
    !(error instanceof AbortError);
}

function statusFromError(error: unknown): number | null {
  if (!error || typeof error !== "object") {
    return null;
  }
  const raw = (error as { status?: unknown }).status;
  return parseStatus(raw);
}

function readStatus(payload: Record<string, unknown>): number | null {
  return parseStatus(payload.http_status);
}

function parseStatus(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isInteger(value) && value >= 100 && value <= 599 ? value : null;
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) && parsed >= 100 && parsed <= 599
      ? parsed
      : null;
  }
  return null;
}

function emptyCounts(): BudgetedReadCounts {
  return { identity: 0, index: 0, head: 0, detail: 0 };
}

function readClock(clock: () => number): number {
  const value = clock();
  if (!Number.isFinite(value)) {
    throw new BudgetedTransportConfigError(
      "clock must return a finite timestamp",
    );
  }
  return value;
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new BudgetedTransportConfigError(`${name} must be a positive integer`);
  }
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new BudgetedTransportConfigError(
      `${name} must be a non-negative integer`,
    );
  }
  return value;
}

function positiveFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new BudgetedTransportConfigError(
      `${name} must be a positive finite number`,
    );
  }
  return value;
}

function nonNegativeFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new BudgetedTransportConfigError(
      `${name} must be a non-negative finite number`,
    );
  }
  return value;
}

function defaultSleep(durationMs: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, durationMs);
  });
}

class AbortError extends Error {
  constructor() {
    super("transport request was aborted");
    this.name = "AbortError";
  }
}
