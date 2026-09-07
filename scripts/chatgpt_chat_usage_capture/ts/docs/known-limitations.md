# Known limitations

- Stage 2 provides manual acquisition into a shared SQLite ledger, raw-model
  reports, mapping suggestions, and rebuilds. Scheduler, leases, persisted
  cooldowns, startup catch-up, automatic nightly reconciliation, reset/provider
  quota accounting, API, UI, and exports are deferred.
- `inspect-capabilities` inspects only the first active/archived index pages.
  Backfill/refresh/reconcile perform bounded traversal. Neither can prove
  coverage of temporary, deleted, inaccessible, or unenumerated history.
- Project coverage is unknown unless identifiers are observed. Version
  metadata and active-branch visibility do not prove every historical branch.
- Pagination budgets, repeated cursors, contradictions, schema drift, missing
  timestamps, and provider failures retain explicit partial/unknown coverage.
  Capabilities are adapter declarations plus observations, not provider
  guarantees.
- Checkpoint/evidence commits are per acquired page. A failed page replays from
  the last durable continuation. Stage-2A JSON history checkpoints are not
  automatically imported; rerun backfill into SQLite.
- Incomplete details and nonterminal generations retain revisits and prevent
  the affected refresh watermark from advancing. These revisits and opt-in
  older-history audits still require manual collection; there is no scheduler.
- Changing indexes can skip records despite deduplication. Explicit
  reconciliation and overlapping manual refresh reduce this risk but do not
  provide a snapshot guarantee. Run collectors sequentially until Stage-3
  leases are available.
- Attempt reconstruction is deterministic over retained evidence. Hidden
  generations, branches, linkage, or statuses can leave provisional/unresolved
  attempts. Raw reports expose unknown/ambiguous time and open coverage gaps;
  their counts are not provider billing or official remaining quota.
- Observed shared/copied/imported origins and non-Chat surfaces are excluded
  from ordinary Chat model counts and remain excluded after rebuild. An origin
  not exposed by the provider cannot be inferred.
- Raw labels remain reportable without canonical mapping. `models` is a
  review/suggestion view; recording approvals uses the TypeScript ledger API,
  not a CLI approval editor. Latest stored mapping is the default unless an
  explicit stored version is selected.
- Activity is shared by verified owner. Collector-specific mapping overrides are
  accepted only when they resolve consistently for collectors sharing that
  owner; conflicting overrides are rejected. An inapplicable selected mapping
  falls back to an applicable stored mapping or leaves family fields
  explicitly unresolved.
- Bootstrap metadata JSON remains separate from SQLite; browser authentication
  remains only in its dedicated profile. No content or credential migration is
  performed.
- Live endpoint availability, authentication, and schema compatibility were
  not tested. Fixture acceptance is offline contract evidence only.
- Live operation requires an operator-authorized dedicated Playwright profile
  and installed Chromium. Login is human-driven and explicitly opt-in.
- The parent Python implementation and runtime services are outside this
  TypeScript package and were not imported or modified.
