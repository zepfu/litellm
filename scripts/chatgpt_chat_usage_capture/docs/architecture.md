# Architecture

Collection, normalization, storage, accounting, and presentation stay separate:

1. Adapter (collector lane) reads Chat history and emits sanitized observations.
2. Ledger persists accounts, attempts, aliases, windows, observations, runs, and scheduler state.
3. Scheduler owns due times, leases, catch-up, backoff, and jitter.
4. Accounting evaluates windows and rebuilds aggregates from the ledger only.
5. CLI / local API / dashboard present independent totals.

One durable lease per account coordinates scheduled refresh, manual refresh, backfill, and reconciliation. Overlapping requests coalesce into pending work instead of starting a second collector.

`next_due_at` is aligned to an explicit UTC `schedule_anchor_at`. Interval changes recompute the next slot without clearing history.
