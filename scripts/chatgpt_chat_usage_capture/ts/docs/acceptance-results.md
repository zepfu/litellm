# Integrated Stage-2 acceptance results

Acceptance date: September 7, 2026.

Scope: `scripts/chatgpt_chat_usage_capture/ts` only.
Exact integration base: `dbea78d4a15ce7953b570160d162fba6687da62d`.
Source lanes:

- Stage 2A: `e4df7d850ff64d8035c5d675fc623e19f3215de9`.
- Stage 2B: `ce0818c4375da4d96a815c93fd32108da65779b6`.

## Integrated contract

One config and one SQLite database connect GET-only history discovery,
message/detail acquisition, sanitized observations, reconstructed attempts,
coverage, checkpoint/revisit state, raw-model reports, and offline rebuilds.
Sparse message pages preserve detail linkage. Rebuild preserves origin
exclusions. An empty `initial-unmapped` seed makes reporting usable without
out-of-band database population or a guessed model-family mapping.

## Pins

Node requirement `>=24.0.0`; verification runtime `v24.18.0`.
Dependencies: better-sqlite3 `13.0.3`, Playwright `1.63.0`,
TypeScript `5.9.3`, typescript-eslint `8.69.0`, ESLint `10.10.0`,
`@eslint/js` `9.39.2`, `@types/node` `26.4.1`,
`@types/better-sqlite3` `9.6.0`, Vitest `3.2.7`.
Adapter `chatgpt-chat-history-v1`; fixture manifest `fixture-manifest-v1`;
history state `1`; SQLite migrations `1` through `3`.

## Gate

| Command/check | Result |
| --- | --- |
| `npm ci` | Passed; 148 packages installed, 149 audited, zero vulnerabilities |
| `npm audit --audit-level=critical` | Passed; zero vulnerabilities |
| `npm run typecheck` | Passed |
| `npm test -- --reporter=dot` | Passed; 65 tests across 11 unit/integration files |
| `npm run lint` | Passed |
| `npm run build` | Passed |
| `git diff --check` / staged diff check | Passed |
| Built CLI `inspect-capabilities` | Exit 0; ready, active 2, archived 0 |
| Built CLI `backfill` | Exit 0; default 14-day range, 2 conversations, 2 observations, 5 messages, 3 attempts |
| Repeated built CLI `backfill` | Exit 0; 0 new observations/messages/attempts, 0 updated attempts, 5 messages and 3 attempts deduplicated |
| Built CLI `report --last-hours 168` | Exit 0; 3 requested-model attempts, 2 completed recorded-final answers, 0 model mismatches |
| Built CLI `rebuild` preview/apply | Both exit 0; same attempt IDs, no attempt changes, identical input fingerprint and revision ID |
| Built CLI `refresh` / `reconcile` | Both exit 0; persisted 48-hour overlap / explicit range, no duplicate attempts |
| Built CLI `models` | Exit 0; retained raw slug, draft empty mapping, no invented family |
| SQLite integrity / foreign keys | `ok` / no violations |

The direct launcher smokes used the committed fixture config and fixture root,
with one state-directory override for every command. Exact disposable database:
`/home/zepfu/.workspace/d1-752-ts-stage2-smoke.C17y5R/usage.sqlite`.
No out-of-band account, mapping, message, or attempt seeding was performed.
Backfill/report/rebuild used exclusive end `2026-09-08T00:00:00Z`;
reconciliation explicitly started at `2026-09-01T00:00:00Z`.

Before rebuild apply, the database had zero aggregate revisions. After both
backfills, refresh, reconciliation, and rebuild apply it held 2 observations,
5 messages and message revisions, 3 attempts and attempt revisions, 1 history
state row, 4 collection runs, and 1 aggregate revision.
Rebuild preview/apply shared input fingerprint
`d1af27d923682930633d4179f9d946670d828a63bccdc8dd296dc30155b6c1c7`.

Fixture collection remains truthfully `partial` with
`project_coverage_unknown`; both index scans and detail traversals completed.
That gap is retained in SQLite and visible in the raw-model report.

Existing Stage-2A tests retain pagination contradictions, budgets, repeated
cursors, legacy fallback gates, late-visible history, overlap, revisits,
scope deduplication, Project/branch evidence, and GET-only coverage. Existing
Stage-2B tests retain scoped identities, whole-generation grouping,
regenerations, terminal timestamp ordering, immutable revisions, privacy,
mapping history, and elapsed raw-model reporting. Integrated checks add the
single-database CLI lifecycle, sparse-message graph preservation,
evidence/checkpoint rollback, durable SQLite state, common path precedence,
origin-preserving rebuild, and metadata-only malformed-node diagnostics.

Npm emitted lifecycle-script approval warnings for better-sqlite3 and esbuild.
No approval policy was changed; SQLite opened successfully and the installed
dependencies passed all test/build/launcher checks.

## Evidence boundary

All acquisition acceptance uses committed synthetic fixtures or injected
readers. Rebuild/report commands use only the fixture-populated local database.
No live ChatGPT request, credential, user profile, Python change, Anthropic
change, container operation, Harness v2 execution, or investigation file is
part of this gate.

## Remaining boundaries

- Live endpoint/profile acceptance remains unrun.
- Commit granularity is a bounded collection result; an interrupted
  uncommitted run replays from previous durable state.
- Mapping approvals use the existing TypeScript ledger API.
- Stage-3 scheduler/recovery policies and Stage-4 reset accounting/API/UI
  remain deferred, as described in `known-limitations.md`.
