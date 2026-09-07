# Durable Scheduler State

This document describes the bounded D1-752 Stage 3 scheduler state API. It is
storage and coordination state only; it does not start a process, execute a
collector job, enforce request budgets, or wire the CLI/configuration.

## Database boundary

`SchedulerStore` receives an already-open `better-sqlite3` database. The caller
still owns that connection and its lifecycle. Construction creates only these
prefixed tables:

- `chatgpt_scheduler_state`: one row per `(account_id, profile_id)`.
- `chatgpt_scheduler_leases`: one row per `(account_id, profile_id)`.

The scheduler does not alter the shared ledger migration table or ledger store.
Times are stored as UTC epoch milliseconds. `account_id` identifies the
configured collector account and `profile_id` identifies the dedicated browser
profile that must not be shared with another collector.

## Schedule semantics

`ensureSchedule(scope, options)` creates an hourly schedule by default:

```ts
const state = scheduler.ensureSchedule(
  { accountId: "personal-primary", profileId: "profile-primary" },
  { interval: "PT1H" },
);
```

Intervals use strict ISO hour/minute syntax such as `PT5M`, `PT1H`, or
`PT1H30M`. The minimum is five minutes. `PT1H` is the default, and `PT60M`
normalizes to `PT1H`. A seconds component, zero interval, fractional value, or
shorter interval is rejected.

The persisted `anchorAt` is the fixed schedule origin. The first normal tick is
`anchorAt + interval + jitter`; later ticks use the same fixed origin and tick
index, so task runtime cannot cause schedule drift. A single random jitter
sample is recorded when the row is first created. The default limit is 60
seconds, and the recorded `jitterMs` is reused after restart and for every tick
in that schedule. Jitter is never greater than the configured limit.

`nextDueAt` is the next unclaimed normal tick. When it is in the past,
`claimTrigger` advances the fixed tick index over all due ticks and returns one
trigger with `missedCount`; it does not return one job per missed interval.
The next due time is persisted before the trigger is returned.

`requestRefresh` stores one pending manual trigger. Repeated requests, or a
request while a valid account/profile lease is occupied, set no additional
queue depth and report `coalesced: true`. A later claim combines a pending
manual request with any due schedule ticks into one trigger.

Changing the interval through `reconfigureSchedule` preserves the anchor,
recorded jitter, pending work, active work, and all ledger data. Due ticks from
the old interval are first materialized into the pending missed count. The next
normal tick is then recalculated strictly after the reconfiguration time using
the same anchor and new interval. No old ledger records are deleted or reset.

## Lease and fencing API

The caller claims a lease before claiming a trigger:

```ts
const lease = scheduler.claimLease(scope, "worker-a", {
  leaseDurationMs: 120_000,
});
const claim = scheduler.claimTrigger(
  scope,
  "worker-a",
  lease.lease.fencingToken,
);
```

Lease acquisition is atomic per account/profile. An unexpired lease prevents a
second owner from acquiring the same scope. When a lease expires, a new owner
receives a monotonically higher `fencingToken`. Heartbeats and release are
conditional on owner, token, and unexpired lease. A stale worker therefore
cannot extend, release, complete, or claim newer state.

An active trigger remains durable across process restart. After the old lease
expires, the new fenced owner can claim the same trigger with `resumed: true`.
Completion clears only the active trigger associated with the current fencing
token. Pending work created while a trigger is active remains queued for the
next claim.

The intended later integration sequence is:

1. `claimLease`.
2. `claimTrigger`.
3. Periodic `heartbeatLease`.
4. Run the separately implemented collector job.
5. `completeTrigger`, then `releaseLease`.

The Stage 3 API deliberately leaves job execution, collector budgets,
retry/cooldown policy, configuration loading, and CLI wiring to later stages.
