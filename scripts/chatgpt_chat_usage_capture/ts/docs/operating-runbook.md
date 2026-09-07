# Stage-2 operating runbook

## First run

```bash
cd scripts/chatgpt_chat_usage_capture/ts
npm ci
npm run build
node bin/usage-capture.mjs init --config ./config.json
```

Bind the provider user, workspace, and quota owner. Choose a dedicated
user-owned browser profile and one `application.database_path`. Do not point
at a normal browser profile or copy authentication state.

Run `bootstrap`, then `inspect-capabilities`, then `backfill`.
Interactive login requires `bootstrap --interactive-login`. If collection
pauses for authentication, rerun that command with `--interactive-login`; only
explicit recovery followed by verified bound ready Chat identity clears the
persisted authentication pause. Read-only inspection and ordinary collection
never clear it. Recovery does not require opening login UI when the session
is already ready. Inspection covers the first index pages only; backfill
performs traversal. Fixture acceptance uses `fixture_history` plus
`--fixture-root`, without live authentication.

## Collection

```bash
node bin/usage-capture.mjs backfill --config ./config.json
node bin/usage-capture.mjs backfill --config ./config.json \
  --since 30d --until 2026-09-08T00:00:00Z
node bin/usage-capture.mjs refresh --config ./config.json
node bin/usage-capture.mjs reconcile --config ./config.json \
  --since 2026-09-01T00:00:00Z --until 2026-09-08T00:00:00Z
```

Backfill defaults to 14 elapsed days. `--since` accepts a duration or an
ISO-8601 instant; `--until` is exclusive. Reconcile requires an explicit start
and ignores newer incremental watermarks. Refresh is a single manual run, not
an hourly service.

All commands resolve the same database: explicit `--database`, otherwise
`--state-directory/usage.sqlite` when overridden, otherwise the configured
database path. Relative paths use the current working directory. Configure an
absolute path when invoking from different directories.

Collection returns its database path, run ID, committed flag, and
inserted/updated/deduplicated counts. A repeated backfill should add a run,
not duplicate observations or attempts. Exit `0` means collection finished
and committed, possibly with partial coverage; inspect `status`, `coverage`,
and `revisits`. A blocked identity returns exit `1` without ingesting evidence.

## Checkpoints, coverage, and recovery

SQLite `history_state` retains independent active/archived continuations,
budgets, pagination states, ranges, warnings, completed discovery starts, and
incomplete-detail revisits. It commits with the corresponding evidence and
attempts. No separate JSON history checkpoint needs copying.
Each acquired page commits with its corresponding continuation. Checkpoints
remain local to the collector account even when activity is shared by owner.

- Partial index/detail: rerun `refresh` to use committed continuations and
  revisits. An interrupted uncommitted run is replayed.
- Exact or older range: use `backfill` or `reconcile --since ... --until ...`.
- Authentication or identity failure: repair the dedicated session or binding;
  use `bootstrap --interactive-login` for explicit interactive recovery. The
  collector clears an authentication pause only after that command verifies the
  configured ready Chat identity. Do not rotate accounts or copy credentials.
  A different identity needs a distinct local account ID.
- `429`: transport throttling, not quota exhaustion. There is no automatic
  retry/cooldown scheduler in Stage 2. All account reads stop and a persisted
  Retry-After cooldown blocks subsequent manual collection until eligible.
- Schema drift, repeated cursors, or exhausted budgets: retain the database
  and review explicit pagination/coverage warnings.

Historical backfill/reconciliation does not advance refresh watermarks.
Refresh advances a scope after clean discovery durably queues its candidates;
detail revisits remain independent. Frozen acquisition metadata and a matching
reread head are required to resume a saved index continuation.

`collection.max_response_bytes` sets the response ceiling (default 33554432,
32 MiB). Declared and actual response sizes are checked before JSON parsing.
Sanitization remains bounded, with a 4096-node default and separate validation
headroom for generated provenance.

`complete` applies only to the declared available index/detail paths.
`partial` retains known gaps; `unknown` means controls or scope could not be
validated. Project IDs and `has_versions` do not prove all Projects or branches
were traversed. Inspect `coverage_gaps`, run details, and evidence revisions;
a partial history result is not proof of no usage.

Do not run concurrent collectors against an account/profile. Account leases
and unattended execution belong to Stage 3.

## Mapping and reporting

The first committed collection records `initial-unmapped`: canonical families
with no raw-slug rules. Reports work before approval:

```bash
node bin/usage-capture.mjs models --config ./config.json
node bin/usage-capture.mjs report --config ./config.json --last-hours 168
```

Review exact slug/mode/effort evidence. `Ledger.saveModelMapping` records
reviewed mappings in this same database; the CLI does not yet author approval
records. Never treat a suggestion as a billing or entitlement classification.
Use `--mapping-version <stored-version>` for an explicit selection in
collection, `models`, or `rebuild`; otherwise the latest stored version is used.
Published versions are immutable. Event-time validity controls applicability;
historical corrections require an explicit bounded interval. Inapplicable or
draft mappings never authorize a stale family to be retained after raw evidence
changes: the applicable stored mapping is used, or the attempt remains
explicitly unresolved. Collector-specific overrides are checked at canonical
owner scope and conflicting overrides are rejected.

## Rebuild

```bash
node bin/usage-capture.mjs rebuild --config ./config.json \
  --until 2026-09-08T00:00:00Z
node bin/usage-capture.mjs rebuild --config ./config.json \
  --until 2026-09-08T00:00:00Z --apply
```

Preview rolls back its transaction. Apply records an aggregate revision.
Keeping `--until` and the mapping version fixed makes evaluation repeatable.
Shared/imported/copied origins remain excluded after rebuild. These commands
use local evidence only and never fetch history. Their output is observed
activity, not remaining quota or provider-charged messages.

## Backup and restore

Stop writers before copying SQLite, or use SQLite's online backup facility.
Preserve `-wal` and `-shm` companions while a writer is active. Protect the
database directory and backups. Restore into an isolated local path and run a
report or rebuild preview before collection.

Do not include the browser profile in report or ledger artifacts: it contains
authentication material. Stage-2A-only JSON checkpoint files are not imported
automatically; repeat the bounded backfill into SQLite instead.

## Deferred operations

Scheduler, leases, durable cooldowns, automatic nightly reconciliation,
reset/provider-quota accounting, API, dashboard, and exports remain deferred.
No container deployment or live acceptance is implied by local fixture checks.
