# ChatGPT Chat usage capture: TypeScript Stage 2B

This directory contains the bounded TypeScript Stage-2B implementation for
read-only ordinary Chat history evidence, a local SQLite ledger, whole-generation
attempt reconstruction, versioned model mapping, deterministic rebuilds, and
raw-model activity reporting. It is an independent collector boundary, not an
official ChatGPT quota meter.

Stage 1 browser and adapter behavior remains available:

- `init`: write a starter JSON configuration;
- `bootstrap`: verify a dedicated persistent browser profile and bind the
  provider user, workspace, and quota owner;
- `inspect-capabilities`: read the active and archived conversation indexes and
  report adapter coverage. It supports live Playwright inspection and an
  explicit fixture-backed offline acceptance mode.

Stage 2B adds:

- transactional SQLite migrations and a sanitized, revisioned evidence ledger;
- scope-aware message, attempt, alias, provenance, and coverage-gap storage;
- chained generation reconstruction across analysis, reasoning, tool, and final
  nodes, including retries, regenerations, and branches;
- reviewed versioned raw-model mapping and deterministic reaggregation;
- offline `report`, `models`, and `rebuild` commands.

Stage 2B does not implement browser discovery traversal, scheduling, reset-window
accounting, provider quota accounting, local API, dashboard UI, or exports.
Those commands fail closed.

## Requirements

- Node.js `>=24.0.0`
- npm
- A dedicated Playwright Chromium profile for live operation

The fixture suite is synthetic and runs offline. No credentials or
browser profile are required for tests or fixture-backed CLI inspection.

## Install

```bash
cd scripts/chatgpt_chat_usage_capture/ts
npm install
npm run build
```

For a live browser run, install the Playwright browser once in the same
environment:

```bash
npx playwright install chromium
```

The exact direct dependency versions are pinned in `package.json` and
`package-lock.json`:

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

The adapter contract version is `chatgpt-chat-history-v1`.

## Configure

Generate a starter config:

```bash
node bin/usage-capture.mjs init --config ./config.json
```

The file is JSON even if an operator gives it another filename. Before
bootstrap, set the three expected identity values:

- `expected_provider_user_id`
- `expected_workspace_id`
- `quota_owner_id`

Set `browser.profile_path` to a dedicated user-owned profile directory. Do not
point it at the normal browser home, a shared default profile, or a directory
containing another account. The config contains identity bindings and policy
metadata only; never put cookies, headers, tokens, or passwords in it.

The generated config enables explicit interactive login. Login is never
started unless `--interactive-login` is supplied.

## Run

Build before invoking the launcher:

```bash
npm run build
node bin/usage-capture.mjs bootstrap --config ./config.json
node bin/usage-capture.mjs inspect-capabilities --config ./config.json
```

For offline Stage-1 CLI acceptance, use the committed synthetic config and
fixtures:

```bash
node bin/usage-capture.mjs inspect-capabilities \
  --config ./tests/fixtures/v1/stage1-config.json \
  --fixture-root ./tests/fixtures/v1 \
  --state-directory /tmp/chatgpt-chat-usage-capture-stage1-state
```

The fixture command must use an account with
`browser.adapter: "fixture_history"` and never contacts ChatGPT. Live
`bootstrap` and live `inspect-capabilities` require
`browser.adapter: "playwright_persistent_context"` and a dedicated profile.

Stage-2B ledger commands operate only on retained local evidence:

```bash
node bin/usage-capture.mjs report \
  --config ./config.json \
  --account personal-primary \
  --database ./state/usage.sqlite \
  --last-hours 24

node bin/usage-capture.mjs models \
  --config ./config.json \
  --account personal-primary \
  --database ./state/usage.sqlite \
  --mapping-version mapping-v2

node bin/usage-capture.mjs rebuild \
  --config ./config.json \
  --account personal-primary \
  --database ./state/usage.sqlite \
  --mapping-version mapping-v2
```

These commands do not issue website requests. `rebuild` previews by default;
pass `--apply` only after reviewing the deterministic result.

The bootstrap command returns exit code `0` only for a verified `ready`
identity. Missing authentication, missing expected identity bindings, identity
mismatch, and browser unavailability return a non-zero status with a
machine-readable JSON result.

To authorize the browser login flow explicitly:

```bash
node bin/usage-capture.mjs bootstrap \
  --config ./config.json \
  --interactive-login
```

`--state-directory <path>` overrides the configured state directory. Persisted
Persisted bootstrap state is limited to sanitized identity metadata, adapter
version, state, and timestamp under `bootstrap/<account-id>.json`; browser
state stays inside the dedicated profile. Stage-2B ledger state is stored in
the configured SQLite database and never contains prompt or answer content.

## Verify

The focused offline acceptance commands are:

```bash
npm run typecheck
npm test
npm run lint
npm run build
git diff --check
```

The fixture-backed transport provides deterministic synthetic session,
conversation-index, detail, and message-page responses without network access.
It is available to the offline `inspect-capabilities` CLI path and is not live
endpoint evidence.

See:

- `docs/architecture.md`
- `docs/endpoint-evidence.md`
- `docs/privacy-threat-model.md`
- `docs/counting-semantics.md`
- `docs/operating-runbook.md`
- `docs/known-limitations.md`
- `docs/acceptance-results.md`
