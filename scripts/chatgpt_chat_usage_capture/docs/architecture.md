# Architecture

Collection, normalization, storage, accounting, and presentation stay separate:

1. Adapter (collector lane) reads Chat history and emits sanitized observations.
2. Ledger persists accounts, attempts, aliases, windows, observations, runs, and scheduler state.
3. Scheduler owns due times, leases, catch-up, backoff, and jitter.
4. Accounting evaluates windows and rebuilds aggregates from the ledger only.
5. CLI / local API / dashboard present independent totals.

One durable lease per account coordinates scheduled refresh, manual refresh, backfill, and reconciliation. Overlapping requests coalesce into pending work instead of starting a second collector.

`next_due_at` is aligned to an explicit UTC `schedule_anchor_at`. Interval changes recompute the next slot without clearing history.

The TypeScript history worker runs behind a bounded Python bridge. One
operation context carries the absolute deadline and cancellation state through
binding, preparation, native reads, and durable cleanup. Late lease, child, or
history acquisitions remain owned until their cleanup is proven.

Native history is enabled only after authenticated identity, typed
capabilities, and an explicit native-history manifest are verified. Missing
native callbacks are unavailable; they are never represented as an empty
successful history result. Browser cleanup uses the owner's registered
lifecycle capability and registration, and an unresolved worker process group
is reported as a cleanup failure rather than being silently forgotten.
