import Database from "better-sqlite3";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { parseRefreshInterval } from "../../src/scheduler/interval.js";
import {
  SchedulerLeaseError,
  SchedulerStore,
} from "../../src/scheduler/store.js";
import type { SchedulerScope } from "../../src/scheduler/types.js";

const HOUR_MS = 60 * 60 * 1000;
const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("durable scheduler state", () => {
  it("validates minute/hour intervals and records one bounded stable jitter", () => {
    expect(parseRefreshInterval("PT5M").spec).toBe("PT5M");
    expect(parseRefreshInterval("PT1H30M").minutes).toBe(90);
    expect(parseRefreshInterval("PT60M").spec).toBe("PT1H");
    expect(() => parseRefreshInterval("PT4M")).toThrow();
    expect(() => parseRefreshInterval("PT30S")).toThrow();

    let now = 0;
    const db = new Database(":memory:");
    const scheduler = new SchedulerStore(db, {
      clock: () => now,
      random: () => 0.5,
    });
    const state = scheduler.ensureSchedule(scope(), {
      anchorAt: 0,
      interval: "PT1H",
    });

    expect(state.interval).toBe("PT1H");
    expect(state.jitterSeconds).toBe(60);
    expect(state.jitterMs).toBe(30_000);
    expect(state.nextDueAt).toBe(HOUR_MS + 30_000);
    now = 15 * 60 * 1000;
    expect(scheduler.getSchedule(scope())?.jitterMs).toBe(30_000);
    db.close();
  });

  it("coalesces six missed hourly ticks into one durable catch-up", () => {
    let now = 0;
    const db = new Database(":memory:");
    const scheduler = new SchedulerStore(db, {
      clock: () => now,
      random: () => 0,
    });
    const account = scope();
    scheduler.ensureSchedule(account, { anchorAt: 0, interval: "PT1H" });
    now = 6 * HOUR_MS + 1;

    const lease = scheduler.claimLease(account, "worker-a", {
      leaseDurationMs: HOUR_MS,
    });
    expect(lease.acquired).toBe(true);
    const claim = scheduler.claimTrigger(
      account,
      "worker-a",
      lease.lease.fencingToken,
    );

    expect(claim?.trigger.kind).toBe("scheduled");
    expect(claim?.trigger.missedCount).toBe(6);
    expect(claim?.trigger.dueAt).toBe(HOUR_MS);
    expect(claim?.state.nextDueAt).toBe(7 * HOUR_MS);
    expect(claim?.state.pendingTrigger).toBe(false);
    db.close();
  });

  it("recovers an active trigger after restart while fencing the stale worker", () => {
    const directory = mkdtempSync(join(tmpdir(), "scheduler-restart-"));
    temporaryDirectories.push(directory);
    const path = join(directory, "state.sqlite");
    let now = 0;
    const firstDb = new Database(path);
    const first = new SchedulerStore(firstDb, {
      clock: () => now,
      random: () => 0,
    });
    const account = scope();
    first.ensureSchedule(account, { anchorAt: 0, interval: "PT1H" });
    now = HOUR_MS + 1;
    const firstLease = first.claimLease(account, "worker-a", {
      leaseDurationMs: 1_000,
    });
    const firstClaim = first.claimTrigger(
      account,
      "worker-a",
      firstLease.lease.fencingToken,
    );
    expect(firstClaim).not.toBeNull();
    const triggerId = firstClaim?.trigger.triggerId;
    firstDb.close();

    now += 2_000;
    const secondDb = new Database(path);
    const second = new SchedulerStore(secondDb, {
      clock: () => now,
      random: () => 0.9,
    });
    expect(second.getSchedule(account)?.activeTriggerId).toBe(triggerId);
    const staleDb = new Database(path);
    const stale = new SchedulerStore(staleDb, {
      clock: () => now,
      random: () => 0.9,
    });
    const secondLease = second.claimLease(account, "worker-b", {
      leaseDurationMs: HOUR_MS,
    });
    expect(secondLease.acquired).toBe(true);
    expect(secondLease.lease.fencingToken).toBe(
      firstLease.lease.fencingToken + 1,
    );
    expect(() =>
      stale.claimTrigger(account, "worker-a", firstLease.lease.fencingToken, {
        at: now,
      }),
    ).toThrow(SchedulerLeaseError);
    expect(
      stale.heartbeatLease(account, "worker-a", firstLease.lease.fencingToken, {
        at: now,
      }).applied,
    ).toBe(false);
    expect(
      stale.completeTrigger(account, "worker-a", firstLease.lease.fencingToken, {
        at: now,
      }).applied,
    ).toBe(false);

    const resumed = second.claimTrigger(
      account,
      "worker-b",
      secondLease.lease.fencingToken,
    );
    expect(resumed?.resumed).toBe(true);
    expect(resumed?.trigger.triggerId).toBe(triggerId);
    expect(
      second.completeTrigger(account, "worker-b", secondLease.lease.fencingToken)
        .applied,
    ).toBe(true);
    staleDb.close();
    secondDb.close();
  });

  it("coalesces a refresh request while a lease is occupied", () => {
    let now = 0;
    const db = new Database(":memory:");
    const first = new SchedulerStore(db, {
      clock: () => now,
      random: () => 0,
    });
    const second = new SchedulerStore(db, {
      clock: () => now,
      random: () => 0,
    });
    const account = scope();
    first.ensureSchedule(account, { anchorAt: 0, interval: "PT1H" });
    now = HOUR_MS + 1;
    const firstLease = first.claimLease(account, "worker-a", {
      leaseDurationMs: HOUR_MS,
    });
    expect(
      second.claimLease(account, "worker-b", { leaseDurationMs: HOUR_MS })
        .acquired,
    ).toBe(false);
    const firstClaim = first.claimTrigger(
      account,
      "worker-a",
      firstLease.lease.fencingToken,
    );
    expect(firstClaim).not.toBeNull();

    const request = second.requestRefresh(account);
    expect(request.occupied).toBe(true);
    expect(request.coalesced).toBe(true);
    expect(request.state.pendingTrigger).toBe(true);

    expect(
      first.completeTrigger(account, "worker-a", firstLease.lease.fencingToken)
        .applied,
    ).toBe(true);
    const next = first.claimTrigger(
      account,
      "worker-a",
      firstLease.lease.fencingToken,
    );
    expect(next?.trigger.kind).toBe("manual");
    expect(next?.trigger.missedCount).toBe(0);
    db.close();
  });

  it("preserves a missed interval when changing the recurring interval", () => {
    let now = 0;
    const db = new Database(":memory:");
    const scheduler = new SchedulerStore(db, {
      clock: () => now,
      random: () => 0,
    });
    const account = scope();
    const initial = scheduler.ensureSchedule(account, {
      anchorAt: 0,
      interval: "PT1H",
    });
    now = 90 * 60 * 1000;
    const reconfigured = scheduler.reconfigureSchedule(account, "PT3H");

    expect(reconfigured.anchorAt).toBe(initial.anchorAt);
    expect(reconfigured.interval).toBe("PT3H");
    expect(reconfigured.nextDueAt).toBe(3 * HOUR_MS);
    expect(reconfigured.pendingTrigger).toBe(true);
    expect(reconfigured.pendingMissedCount).toBe(1);

    const lease = scheduler.claimLease(account, "worker-a", {
      leaseDurationMs: HOUR_MS,
    });
    const claim = scheduler.claimTrigger(
      account,
      "worker-a",
      lease.lease.fencingToken,
    );
    expect(claim?.trigger.missedCount).toBe(1);
    expect(claim?.trigger.kind).toBe("scheduled");
    expect(claim?.state.nextDueAt).toBe(3 * HOUR_MS);
    db.close();
  });
});

function scope(): SchedulerScope {
  return {
    accountId: "account-one",
    profileId: "profile-one",
  };
}
