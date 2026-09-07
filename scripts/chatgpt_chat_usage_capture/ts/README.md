# ChatGPT Chat usage capture: TypeScript Stage 2

An independent, read-only ordinary Chat history collector and SQLite evidence
ledger. Stage 2 integrates history acquisition with attempt reconstruction and
raw-model reporting. It is not an official ChatGPT quota meter.

## Commands

- `init`: write a starter JSON configuration.
- `bootstrap`: verify a dedicated browser profile against the bound provider
  user, workspace, and quota owner.
- `inspect-capabilities`: inspect the first active and archived index pages.
- `backfill`: acquire active/archived history into SQLite, using a 14-day
  default or an explicit range.
- `refresh`: manually collect with durable per-scope discovery watermarks,
  a 48-hour overlap, and incomplete-detail revisits.
- `reconcile`: manually reread an explicit range regardless of the watermark.
- `report`: report retained raw-model activity over the last N hours.
- `models`: show retained model/mode/effort evidence for mapping review.
- `rebuild`: preview or apply deterministic reaggregation from local evidence.

All history and ledger commands use one configuration and one SQLite database.
There is no scheduler, reset-window accounting, provider quota accounting,
local API, dashboard, or export implementation.

## Requirements and installation

- Node.js `>=24.0.0` and npm.
- For live operation only: a dedicated Playwright Chromium profile and a
  locally installed browser.
- Tests and fixture-backed commands are synthetic and offline.

```bash
cd scripts/chatgpt_chat_usage_capture/ts
npm ci
npm run build
```

Live browser installation, when separately needed:

```bash
npx playwright install chromium
```

Exact direct dependency pins in `package.json` and `package-lock.json`:

| Package | Version |
| --- | --- |
| `better-sqlite3` | `13.0.3` |
| `playwright` | `1.63.0` |
| `typescript` | `5.9.3` |
| `typescript-eslint` | `8.69.0` |
| `eslint` | `10.10.0` |
| `@eslint/js` | `9.39.2` |
| `@types/node` | `26.4.1` |
| `@types/better-sqlite3` | `9.6.0` |
| `vitest` | `3.2.7` |

Adapter contract: `chatgpt-chat-history-v1`. SQLite schema: `5`. History state
contract: `1`.

## Configuration

```bash
node bin/usage-capture.mjs init --config ./config.json
```

The config is schema-version-1 JSON. Bind `expected_provider_user_id`,
`expected_workspace_id`, and `quota_owner_id` before collection. Set
`browser.profile_path` to a dedicated user-owned profile, never a normal or
shared browser profile. Credentials, cookies, headers, and tokens do not belong
in config.

`application.database_path` defaults to `./state/usage.sqlite`. When that field
is omitted, the database is `usage.sqlite` within `application.state_directory`.
Every history/report/model/rebuild command uses the same resolution:

1. `--database <path>`, when supplied.
2. `<path>/usage.sqlite` when `--state-directory <path>` is supplied.
3. `application.database_path` from the loaded config.

Relative paths are relative to the command's working directory. Use consistent
overrides across commands, or configure an absolute database path once.
Bootstrap identity JSON remains under `<state-directory>/bootstrap/`; all
history checkpoints, revisits, observations, messages, and attempts are in
SQLite. Browser authentication stays in the dedicated profile.

## Live operation

```bash
node bin/usage-capture.mjs bootstrap --config ./config.json
node bin/usage-capture.mjs inspect-capabilities --config ./config.json
node bin/usage-capture.mjs backfill --config ./config.json
node bin/usage-capture.mjs refresh --config ./config.json
node bin/usage-capture.mjs reconcile --config ./config.json \
  --since 2026-09-01T00:00:00Z --until 2026-09-08T00:00:00Z
node bin/usage-capture.mjs report --config ./config.json --last-hours 168
node bin/usage-capture.mjs models --config ./config.json
node bin/usage-capture.mjs rebuild --config ./config.json
```

Interactive login is opt-in: `bootstrap --interactive-login`. When collection
is paused for authentication, that command is the explicit recovery entry:
only a verified interactive ready Chat identity clears the persisted pause.
Read-only inspection cannot clear it. Missing or mismatched identity blocks
collection. A stored collector account cannot be rebound to a different
identity/workspace/quota owner; use a distinct account ID instead. No command
submits prompts or mutates provider data.

`backfill --since` accepts an elapsed duration such as `30d` or an ISO-8601
instant. `--until` is exclusive. `reconcile` requires `--since`.
`report --until` and `rebuild --until` fix the evaluation instant for repeatable
historical results. Report defaults to 24 hours; rebuild's report covers seven
elapsed days.

The first committed collection seeds `initial-unmapped` with canonical
families and no raw-slug rules. Raw reports and rebuilds work without mapping
approval. `models` suggests classifications; recording reviewed rules uses
`Ledger.saveModelMapping`. `--mapping-version` selects a stored version.
`rebuild` previews by default; `--apply` writes the reviewed result. Neither
reporting nor rebuilding makes website requests.

## Offline fixture example

The committed `stage1-config.json` remains usable by all Stage-2 commands:

```bash
node bin/usage-capture.mjs inspect-capabilities \
  --config tests/fixtures/v1/stage1-config.json \
  --fixture-root tests/fixtures/v1 --state-directory ./state/fixture-stage2
node bin/usage-capture.mjs backfill \
  --config tests/fixtures/v1/stage1-config.json \
  --fixture-root tests/fixtures/v1 --state-directory ./state/fixture-stage2 \
  --until 2026-09-08T00:00:00Z
node bin/usage-capture.mjs report \
  --config tests/fixtures/v1/stage1-config.json \
  --state-directory ./state/fixture-stage2 --last-hours 168 \
  --until 2026-09-08T00:00:00Z
node bin/usage-capture.mjs rebuild \
  --config tests/fixtures/v1/stage1-config.json \
  --state-directory ./state/fixture-stage2 --until 2026-09-08T00:00:00Z
```

Fixture mode requires `browser.adapter: "fixture_history"` and an explicit
`--fixture-root` for acquisition. It never opens Playwright or contacts ChatGPT.
Repeating the same backfill keeps observations, messages, and attempts
deduplicated while recording another collection run.

## Verification and documentation

```bash
npm ci
npm audit --audit-level=critical
npm run typecheck
npm test
npm run lint
npm run build
git diff --check
```

See `docs/architecture.md`, `docs/endpoint-evidence.md`,
`docs/counting-semantics.md`, `docs/operating-runbook.md`,
`docs/privacy-threat-model.md`, `docs/known-limitations.md`, and
`docs/acceptance-results.md`.
