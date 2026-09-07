import { describe, expect, it } from "vitest";

import {
  AdapterError,
  LEGACY_DETAIL,
  MODERN_DETAIL,
  MODERN_INDEX,
  MODERN_MESSAGES,
  SESSION_ROUTE,
  type HistoryTransport,
} from "../../src/adapters/chatgpt/adapter.js";
import {
  BudgetedTransport,
  BudgetedTransportConfigError,
  BudgetedTransportPartialError,
} from "../../src/browser/budgeted-transport.js";

interface PendingSleep {
  durationMs: number;
  resolve: () => void;
}

class ManualClock {
  nowMs = 0;
  readonly sleeps: number[] = [];
  private readonly pendingSleeps: PendingSleep[] = [];

  constructor(private readonly pendingAtOrAboveMs = 10_000) {}

  now = (): number => this.nowMs;

  sleep = (durationMs: number): Promise<void> => {
    this.sleeps.push(durationMs);
    if (durationMs >= this.pendingAtOrAboveMs) {
      return new Promise((resolve) => {
        this.pendingSleeps.push({ durationMs, resolve });
      });
    }
    this.nowMs += durationMs;
    return Promise.resolve();
  };

  releaseNextSleep(): void {
    const pending = this.pendingSleeps.shift();
    if (!pending) {
      throw new Error("no pending sleep");
    }
    this.nowMs += pending.durationMs;
    pending.resolve();
  }
}

type ResponsePlan =
  | Record<string, unknown>
  | Error
  | (() => Promise<Record<string, unknown>>);

class ScriptedTransport implements HistoryTransport {
  readonly calls: Array<{
    method: string;
    path: string;
    params: Record<string, unknown>;
  }> = [];
  inFlight = 0;
  maxInFlight = 0;
  cancelCalls = 0;

  constructor(private readonly plans: ResponsePlan[]) {}

  async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> {
    this.calls.push({ method, path, params: { ...params } });
    this.inFlight += 1;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    const plan = this.plans.shift() ?? { http_status: 200 };
    try {
      if (plan instanceof Error) {
        throw plan;
      }
      if (typeof plan === "function") {
        return await plan();
      }
      return plan;
    } finally {
      this.inFlight -= 1;
    }
  }

  async cancel(): Promise<void> {
    this.cancelCalls += 1;
  }

  async close(): Promise<void> {}
}

class DeferredTransport implements HistoryTransport {
  readonly calls: string[] = [];
  inFlight = 0;
  maxInFlight = 0;
  private resolveFirst: ((value: Record<string, unknown>) => void) | null = null;

  request(_method: string, path: string): Promise<Record<string, unknown>> {
    this.calls.push(path);
    this.inFlight += 1;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    if (this.calls.length > 1) {
      this.inFlight -= 1;
      return Promise.resolve({ http_status: 200 });
    }
    return new Promise((resolve) => {
      this.resolveFirst = (value) => {
        this.inFlight -= 1;
        resolve(value);
      };
    });
  }

  resolveFirstRequest(): void {
    this.resolveFirst?.({ http_status: 200 });
  }

  async cancel(): Promise<void> {
    this.resolveFirst?.({ http_status: 499 });
  }
}

class TimeoutTransport implements HistoryTransport {
  readonly calls: string[] = [];
  inFlight = 0;
  maxInFlight = 0;
  cancelCalls = 0;
  private pendingReject: ((error: Error) => void) | null = null;

  async request(_method: string, path: string): Promise<Record<string, unknown>> {
    this.calls.push(path);
    if (this.calls.length > 1) {
      return { http_status: 200 };
    }
    this.inFlight += 1;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    return new Promise((_resolve, reject) => {
      this.pendingReject = (error) => {
        this.inFlight -= 1;
        reject(error);
      };
    });
  }

  async cancel(): Promise<void> {
    this.cancelCalls += 1;
    this.pendingReject?.(new Error("aborted"));
    this.pendingReject = null;
  }
}

function makeTransport(
  transport: HistoryTransport,
  clock: ManualClock,
  options: ConstructorParameters<typeof BudgetedTransport>[1] = {},
): BudgetedTransport {
  return new BudgetedTransport(transport, {
    clock: clock.now,
    sleep: clock.sleep,
    random: () => 0,
    requestTimeoutMs: 10_000,
    ...options,
  });
}

async function flush(): Promise<void> {
  await new Promise<void>((resolve) => {
    setImmediate(resolve);
  });
}

describe("BudgetedTransport", () => {
  it("serializes concurrent reads and counts identity, head, index, and detail classes", async () => {
    const clock = new ManualClock();
    const transport = new DeferredTransport();
    const budgeted = makeTransport(transport, clock, { minGapMs: 0 });

    const first = budgeted.request("GET", SESSION_ROUTE);
    const second = budgeted.request("GET", MODERN_INDEX, { offset: 0 });
    const third = budgeted.request("GET", MODERN_INDEX, { offset: 100 });
    const fourth = budgeted.request(
      "GET",
      MODERN_MESSAGES.replace("{conversation_id}", "conv-001"),
    );

    await flush();
    expect(transport.calls).toEqual([SESSION_ROUTE]);
    expect(transport.maxInFlight).toBe(1);

    transport.resolveFirstRequest();
    await expect(first).resolves.toEqual({ http_status: 200 });
    await expect(second).resolves.toEqual({ http_status: 200 });
    await expect(third).resolves.toEqual({ http_status: 200 });
    await expect(fourth).resolves.toEqual({ http_status: 200 });

    expect(transport.maxInFlight).toBe(1);
    expect(budgeted.getDiagnostics()).toMatchObject({
      attempts: 4,
      retries: 0,
      reads: 4,
      readCounts: { identity: 1, index: 1, head: 1, detail: 1 },
      attemptsByClass: { identity: 1, index: 1, head: 1, detail: 1 },
    });
  });

  it("retries transient responses with bounded exponential backoff and jitter", async () => {
    const clock = new ManualClock();
    const transport = new ScriptedTransport([
      { http_status: 503 },
      { http_status: 502 },
      { http_status: 200, value: "kept" },
    ]);
    const budgeted = makeTransport(transport, clock, {
      minGapMs: 0,
      retryBackoffBaseMs: 10,
      retryBackoffMaxMs: 20,
      retryJitterMs: 10,
    });

    await expect(
      budgeted.request("GET", MODERN_INDEX, { offset: 100 }),
    ).resolves.toEqual({ http_status: 200, value: "kept" });

    expect(transport.calls).toHaveLength(3);
    expect(clock.sleeps.filter((durationMs) => durationMs < 10_000)).toEqual([
      10,
      20,
    ]);
    expect(budgeted.getDiagnostics()).toMatchObject({
      attempts: 3,
      retries: 2,
      retriesByClass: { identity: 0, index: 2, head: 0, detail: 0 },
    });
  });

  it.each([401, 403, 429])(
    "returns HTTP %s unchanged without retrying",
    async (status) => {
      const clock = new ManualClock();
      const response = {
        http_status: status,
        retry_after: "120",
        responseReceivedAt: 1_757_265_600_000,
      };
      const transport = new ScriptedTransport([response, { http_status: 200 }]);
      const budgeted = makeTransport(transport, clock, {
        minGapMs: 0,
      });

      await expect(
        budgeted.request("GET", SESSION_ROUTE),
      ).resolves.toEqual(response);
      expect(transport.calls).toHaveLength(1);
      expect(budgeted.getDiagnostics().retries).toBe(0);
    },
  );

  it("raises a typed partial error at the attempt budget and blocks later reads", async () => {
    const clock = new ManualClock();
    const transport = new ScriptedTransport([
      { http_status: 503 },
      { http_status: 503 },
      { http_status: 200 },
    ]);
    const budgeted = makeTransport(transport, clock, {
      minGapMs: 0,
      maxAttemptsPerRun: 2,
      retryBackoffBaseMs: 0,
      retryBackoffMaxMs: 0,
      retryJitterMs: 0,
    });

    const firstError = await budgeted
      .request("GET", MODERN_INDEX, { offset: 100 })
      .catch((error: unknown) => error);
    expect(firstError).toBeInstanceOf(BudgetedTransportPartialError);
    expect((firstError as BudgetedTransportPartialError).reason).toBe(
      "attempt_budget_exhausted",
    );
    expect(transport.calls).toHaveLength(2);

    const secondError = await budgeted
      .request("GET", SESSION_ROUTE)
      .catch((error: unknown) => error);
    expect(secondError).toBeInstanceOf(BudgetedTransportPartialError);
    expect(transport.calls).toHaveLength(2);
    expect(budgeted.getDiagnostics().blockedReason).toBe(
      "attempt_budget_exhausted",
    );
  });

  it("raises a typed partial error when the run deadline prevents another read", async () => {
    const clock = new ManualClock(100);
    const transport = new ScriptedTransport([
      { http_status: 200 },
      { http_status: 200 },
    ]);
    const budgeted = makeTransport(transport, clock, {
      minGapMs: 1_000,
      maxRunDurationMs: 100,
    });

    await expect(budgeted.request("GET", SESSION_ROUTE)).resolves.toEqual({
      http_status: 200,
    });
    clock.releaseNextSleep();
    const second = budgeted.request("GET", MODERN_INDEX, { offset: 100 });
    const error = await second.catch((value: unknown) => value);
    expect(error).toBeInstanceOf(BudgetedTransportPartialError);
    expect((error as BudgetedTransportPartialError).reason).toBe(
      "run_deadline_exceeded",
    );
    expect(transport.calls).toHaveLength(1);
  });

  it("waits for the inner request to settle after timeout before issuing another read", async () => {
    const clock = new ManualClock(10);
    const transport = new TimeoutTransport();
    const budgeted = makeTransport(transport, clock, {
      minGapMs: 0,
      requestTimeoutMs: 10,
      transientRetries: 0,
    });

    const first = budgeted.request("GET", SESSION_ROUTE);
    await flush();
    clock.releaseNextSleep();
    const firstError = await first.catch((error: unknown) => error);

    expect(firstError).toBeInstanceOf(BudgetedTransportPartialError);
    expect((firstError as BudgetedTransportPartialError).reason).toBe(
      "request_timeout",
    );
    await expect(
      budgeted.request("GET", MODERN_INDEX, { offset: 100 }),
    ).resolves.toEqual({ http_status: 200 });
    expect(transport.cancelCalls).toBe(1);
    expect(transport.maxInFlight).toBe(1);
    expect(transport.calls).toHaveLength(2);
  });

  it("cancels the active request and blocks queued reads", async () => {
    const clock = new ManualClock();
    const transport = new TimeoutTransport();
    const budgeted = makeTransport(transport, clock, {
      minGapMs: 0,
      requestTimeoutMs: 10_000,
    });

    const first = budgeted.request("GET", SESSION_ROUTE);
    await flush();
    const cancellation = budgeted.cancel();
    const firstError = await first.catch((error: unknown) => error);
    await cancellation;

    expect(firstError).toBeInstanceOf(BudgetedTransportPartialError);
    expect((firstError as BudgetedTransportPartialError).reason).toBe(
      "cancelled",
    );
    await expect(
      budgeted.request("GET", MODERN_INDEX, { offset: 100 }),
    ).rejects.toBeInstanceOf(BudgetedTransportPartialError);
    expect(transport.calls).toHaveLength(1);
    expect(transport.maxInFlight).toBe(1);
  });

  it("enforces the allowlist without exposing conversation identifiers in diagnostics", async () => {
    const clock = new ManualClock();
    const transport = new ScriptedTransport([{ http_status: 200 }]);
    const budgeted = makeTransport(transport, clock, { minGapMs: 0 });

    await expect(
      budgeted.request("GET", "/backend-api/me"),
    ).rejects.toBeInstanceOf(AdapterError);
    expect(transport.calls).toHaveLength(0);

    await budgeted.request(
      "GET",
      MODERN_DETAIL.replace("{conversation_id}", "conv-001"),
    );
    expect(JSON.stringify(budgeted.getDiagnostics())).not.toContain("conv-001");
  });

  it("validates all fixed safety controls", () => {
    const transport = new ScriptedTransport([]);
    const clock = new ManualClock();
    const invalid = [
      { concurrentReads: 2 },
      { minGapMs: -1 },
      { maxAttemptsPerRun: 0 },
      { maxRunDurationMs: 0 },
      { requestTimeoutMs: 0 },
      { transientRetries: -1 },
      { retryBackoffMaxMs: 1, retryBackoffBaseMs: 2 },
    ];

    for (const options of invalid) {
      expect(() => makeTransport(transport, clock, options)).toThrow(
        BudgetedTransportConfigError,
      );
    }
  });

  it("classifies legacy detail reads as detail reads", async () => {
    const clock = new ManualClock();
    const transport = new ScriptedTransport([{ http_status: 200 }]);
    const budgeted = makeTransport(transport, clock, { minGapMs: 0 });

    await budgeted.request(
      "GET",
      LEGACY_DETAIL.replace("{conversation_id}", "conv-001"),
    );

    expect(budgeted.getDiagnostics().readCounts.detail).toBe(1);
  });
});
