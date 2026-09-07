# Operating runbook

Initialize the ledger, inspect capabilities, then backfill or refresh through the scheduler so lease, backoff, and catch-up stay shared.

- 429: persist `Retry-After` (delta-seconds or HTTP-date) as `backoff_until`. This is a transport cooldown, not quota exhaustion.
- Missed hours: one catch-up collect plus queued incomplete work, not one poll per missed interval.
- Interval change: `schedule set --every PT3H` or `3h` recalculates `next_due_at` from the UTC anchor.
- Rebuild: `rebuild --dry-run` then `rebuild --apply`. Aggregates publish only after the transaction commits.
- Retention prune keeps attempt aliases/tombstones. If raw observations are gone, rebuild uses retained attempt projections and warns.

Dashboard cards show three policy buckets. Unknown remaining is the string Unknown, never zero.
