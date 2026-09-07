import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { defaultConfig, type AccountConfig } from "../../src/config.js";
import type {
  AdaptedPage,
  IdentityRecord,
} from "../../src/contracts/records.js";
import type {
  HistoryCollectionRequest,
  HistoryReader,
} from "../../src/contracts/history.js";
import { emptyCapabilities } from "../../src/normalize/identity.js";
import {
  SchedulerExecutionService,
  SchedulerLeaseLostError,
} from "../../src/scheduler/runner.js";
import { Ledger } from "../../src/ledger/store.js";

const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;
const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("scheduler execution service", () => {
  it("runs one reconciliation after its durable 24-hour due time", async () => {
    let now = 0;
    const ledger = openLedger();
    const account = makeAccount("runner-reconciliation");
    const scope = {
      accountId: account.id,
      profileId: "profile-one",
    };
    const reader = new EmptyReader();
    const requests: HistoryCollectionRequest[] = [];
    const service = new SchedulerExecutionService({
      ledger,
      account,
      scope,
      ownerId: "worker-a",
      schedule: {
        anchorAt: 0,
        interval: "PT1H",
        jitterSeconds: 0,
      },
      clock: () => now,
      leaseDurationMs: 10_000,
      heartbeatIntervalMs: 1_000,
      readerFactory: async ({ request }) => {
        requests.push(request);
        return reader;
      },
    });

    const initial = await service.executeScheduled();
    expect(initial.status).toBe("idle");
    expect(service.getExecutionState()?.reconciliationNextDueAt).toBe(DAY_MS);

    now = DAY_MS + 1;
    const result = await service.executeScheduled();

    expect(result.status).toBe("partial");
    expect(result.mode).toBe("reconciliation");
    expect(result.reconciliationMissedCount).toBe(1);
    expect(result.trigger?.missedCount).toBeGreaterThan(0);
    expect(requests).toHaveLength(1);
    expect(requests[0]?.mode).toBe("reconciliation");
    expect(service.getExecutionState()?.activeTriggerId).toBeNull();
    ledger.close();
  });

  it("rebinds a resumed execution to the replacement fencing token", async () => {
    let now = HOUR_MS + 1;
    const path = makeDatabasePath("scheduler-runner-restart-");
    const account = makeAccount("runner-restart");
    const scope = {
      accountId: account.id,
      profileId: "profile-one",
    };
    const firstLedger = new Ledger(path);
    const first = new SchedulerExecutionService({
      ledger: firstLedger,
      account,
      scope,
      ownerId: "worker-a",
      schedule: {
        anchorAt: 0,
        interval: "PT1H",
        jitterSeconds: 0,
      },
      clock: () => now,
      leaseDurationMs: 1_000,
      heartbeatIntervalMs: 100,
      readerFactory: async () => {
        throw new Error("reader bootstrap failed");
      },
    });

    await expect(first.executeScheduled()).rejects.toThrow(
      "reader bootstrap failed",
    );
    const activeTriggerId = first.getExecutionState()?.activeTriggerId;
    expect(activeTriggerId).not.toBeNull();
    firstLedger.close();

    now += 2_000;
    const secondLedger = new Ledger(path);
    const reader = new EmptyReader();
    const second = new SchedulerExecutionService({
      ledger: secondLedger,
      account,
      scope,
      ownerId: "worker-b",
      schedule: {
        anchorAt: 0,
        interval: "PT1H",
        jitterSeconds: 0,
      },
      clock: () => now,
      leaseDurationMs: 1_000,
      heartbeatIntervalMs: 100,
      readerFactory: async () => reader,
    });

    const result = await second.executeScheduled();

    expect(result.resumed).toBe(true);
    expect(result.trigger?.triggerId).toBe(activeTriggerId);
    expect(result.status).toBe("partial");
    expect(second.getExecutionState()?.activeTriggerId).toBeNull();
    secondLedger.close();
  });

  it("stops before the next browser read when the lease expires", async () => {
    let now = HOUR_MS + 1;
    const ledger = openLedger();
    const account = makeAccount("runner-fencing");
    const scope = {
      accountId: account.id,
      profileId: "profile-one",
    };
    const reader = new EmptyReader();
    let cancelled = 0;
    let closed = 0;
    const service = new SchedulerExecutionService({
      ledger,
      account,
      scope,
      ownerId: "worker-a",
      schedule: {
        anchorAt: 0,
        interval: "PT1H",
        jitterSeconds: 0,
      },
      clock: () => now,
      leaseDurationMs: 1_000,
      heartbeatIntervalMs: 100,
      readerFactory: async () => ({
        reader: {
          capabilities: reader.capabilities,
          inspectSessionIdentity: async () => {
            reader.calls.push("inspectSessionIdentity");
            now += 2_000;
            return reader.identity;
          },
          listConversations: reader.listConversations.bind(reader),
          fetchConversation: reader.fetchConversation.bind(reader),
          fetchMessages: reader.fetchMessages.bind(reader),
        },
        cancel: () => {
          cancelled += 1;
        },
        close: () => {
          closed += 1;
        },
      }),
    });

    await expect(service.executeScheduled()).rejects.toBeInstanceOf(
      SchedulerLeaseLostError,
    );

    expect(reader.calls).toEqual(["inspectSessionIdentity"]);
    expect(cancelled).toBe(1);
    expect(closed).toBe(1);
    expect(service.getExecutionState()?.activeTriggerId).not.toBeNull();
    ledger.close();
  });

  it("stops before the first ledger mutation when the lease expires after a read", async () => {
    let now = HOUR_MS + 1;
    const ledger = openLedger();
    const account = makeAccount("runner-write-fencing");
    const scope = {
      accountId: account.id,
      profileId: "profile-one",
    };
    const reader = new AdvanceOnListReader(() => {
      now += 2_000;
    });
    let cancelled = 0;
    let closed = 0;
    const service = new SchedulerExecutionService({
      ledger,
      account,
      scope,
      ownerId: "worker-a",
      schedule: {
        anchorAt: 0,
        interval: "PT1H",
        jitterSeconds: 0,
      },
      clock: () => now,
      leaseDurationMs: 1_000,
      heartbeatIntervalMs: 100,
      readerFactory: async () => ({
        reader,
        cancel: () => {
          cancelled += 1;
        },
        close: () => {
          closed += 1;
        },
      }),
    });

    await expect(service.executeScheduled()).rejects.toBeInstanceOf(
      SchedulerLeaseLostError,
    );

    expect(reader.calls).toEqual([
      "inspectSessionIdentity",
      "listConversations:active",
    ]);
    expect(cancelled).toBe(1);
    expect(closed).toBe(1);
    expect(ledger.listAccounts()).toHaveLength(1);
    expect(
      (
        ledger.db
          .prepare("SELECT ended_at FROM collector_runs")
          .get() as { ended_at: string | null } | undefined
      )?.ended_at,
    ).toBeNull();
    ledger.close();
  });

  it("persists one coalesced manual request while another execution holds the lease", async () => {
    let now = 0;
    const path = makeDatabasePath("scheduler-runner-pending-");
    const account = makeAccount("runner-pending");
    const scope = {
      accountId: account.id,
      profileId: "profile-one",
    };
    const firstLedger = new Ledger(path);
    const secondLedger = new Ledger(path);
    let releaseReader: (() => void) | undefined;
    let readerStarted: (() => void) | undefined;
    const readerReady = new Promise<void>((resolve) => {
      readerStarted = resolve;
    });
    const readerGate = new Promise<void>((resolve) => {
      releaseReader = resolve;
    });
    const first = new SchedulerExecutionService({
      ledger: firstLedger,
      account,
      scope,
      ownerId: "worker-a",
      clock: () => now,
      leaseDurationMs: 10_000,
      heartbeatIntervalMs: 1_000,
      readerFactory: async () => {
        readerStarted?.();
        await readerGate;
        return new EmptyReader();
      },
    });
    const second = new SchedulerExecutionService({
      ledger: secondLedger,
      account,
      scope,
      ownerId: "worker-b",
      clock: () => now,
      leaseDurationMs: 10_000,
      heartbeatIntervalMs: 1_000,
      readerFactory: async () => new EmptyReader(),
    });

    const firstRun = first.executeManual();
    await readerReady;
    const queued = await second.executeManual();

    expect(queued.status).toBe("queued");
    expect(queued.coalesced).toBe(true);
    expect(second.getExecutionState()?.pendingMode).toBe("manual");

    releaseReader?.();
    const firstResult = await firstRun;
    expect(firstResult.status).toBe("partial");

    firstLedger.close();
    secondLedger.close();
  });
});

class EmptyReader implements HistoryReader {
  readonly capabilities = emptyCapabilities();
  readonly calls: string[] = [];
  readonly identity: IdentityRecord = {
    providerUserId: "user-abc123",
    workspaceId: "workspace-abc123",
    quotaOwnerId: "user-abc123",
    surface: "chat",
    authState: "ready",
    identityErrors: [],
  };

  async inspectSessionIdentity(): Promise<IdentityRecord> {
    this.calls.push("inspectSessionIdentity");
    return this.identity;
  }

  async listConversations(options: {
    archived: boolean;
    offset?: number;
    limit?: number;
    order?: string;
  }): Promise<AdaptedPage<never>> {
    this.calls.push(
      `listConversations:${options.archived ? "archived" : "active"}`,
    );
    return completePage([]);
  }

  async fetchConversation(): Promise<never> {
    this.calls.push("fetchConversation");
    throw new Error("unexpected conversation read");
  }

  async fetchMessages(): Promise<AdaptedPage<never>> {
    this.calls.push("fetchMessages");
    throw new Error("unexpected message read");
  }
}

class AdvanceOnListReader extends EmptyReader {
  constructor(private readonly advance: () => void) {
    super();
  }

  override async listConversations(options: {
    archived: boolean;
    offset?: number;
    limit?: number;
    order?: string;
  }): Promise<AdaptedPage<never>> {
    this.calls.push(
      `listConversations:${options.archived ? "archived" : "active"}`,
    );
    this.advance();
    return completePage([]);
  }
}

function completePage<T>(items: T[]): AdaptedPage<T> {
  return {
    items,
    continuation: null,
    exhausted: true,
    paginationState: "complete",
    schemaVersion: "chatgpt-chat-history-v1",
    coverage: "validated_page",
    warnings: [],
  };
}

function makeAccount(id: string): AccountConfig {
  const account = defaultConfig().accounts[0];
  if (!account) {
    throw new Error("default account is missing");
  }
  return {
    ...account,
    id,
    expectedProviderUserId: "user-abc123",
    expectedWorkspaceId: "workspace-abc123",
    quotaOwnerId: "user-abc123",
  };
}

function openLedger(): Ledger {
  return new Ledger(makeDatabasePath("scheduler-runner-"));
}

function makeDatabasePath(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  temporaryDirectories.push(directory);
  return join(directory, "state.sqlite");
}
