# Operating runbook

Initialize the ledger, inspect capabilities, then backfill or refresh through the scheduler so lease, backoff, and catch-up stay shared.

- 429: persist `Retry-After` (delta-seconds or HTTP-date) as `backoff_until`. This is a transport cooldown, not quota exhaustion.
- Missed hours: one catch-up collect plus queued incomplete work, not one poll per missed interval.
- Interval change: `schedule set --every PT3H` or `3h` recalculates `next_due_at` from the UTC anchor.
- Rebuild: `rebuild --dry-run` then `rebuild --apply`. Aggregates publish only after the transaction commits.
- Retention prune keeps attempt aliases/tombstones. If raw observations are gone, rebuild uses retained attempt projections and warns.
- Configure `expected_provider_user_id`, `expected_workspace_id`, and `quota_owner_id` for every account. Collection pauses as `unconfigured` or `identity_mismatch` until all three values are observed from the authenticated session and match; configured values are never used as observed evidence.
- Page observations are projected through the typed metadata allowlist before
  they are written. The `items`, `messages`, and `mapping` collections are
  recursively content-sanitized and bounded; unknown fields remain limited to
  structural counts and provenance. Projection truncation downgrades coverage
  and cannot establish page exhaustion. Upstream truncation reasons and
  incompleteness are preserved when local traversal adds no new truncation.
  Unsupported transfer versions are retained as evidence but marked
  unrecognized.
- Coverage-gap and alias-collision writes validate the complete scope/context
  envelope, including dynamic mapping keys, before any database mutation.
- Missing or contradictory detail pagination, including an unknown HTTP 200 shape, records an open coverage gap, keeps the conversation pending, and prevents the run watermark from advancing.

Dashboard cards show three policy buckets. Unknown remaining is the string Unknown, never zero.
