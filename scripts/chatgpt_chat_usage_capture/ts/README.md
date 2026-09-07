# ChatGPT Chat usage capture: TypeScript Stage 2A

This directory contains the bounded TypeScript Stage-2A implementation for
read-only ordinary Chat history discovery and acquisition. It is an
independent collector boundary, not an official ChatGPT quota meter.

Stage 2A implements:

- `init`: write a starter JSON configuration;
- `bootstrap`: verify a dedicated persistent browser profile and bind the
  provider user, workspace, and quota owner;
- `inspect-capabilities`: read the active and archived conversation indexes and
  report adapter coverage;
- `backfill`: enumerate active and archived indexes, use a 14-day default or an
  arbitrary range, acquire modern detail/messages, and checkpoint each scope;
- `refresh`: use each scope's durable discovery watermark with a 48-hour
  discovery overlap and revisit incomplete conversations;
- `reconcile`: force acquisition for an explicitly supplied range, independent
  of the incremental watermark.

The adapter exposes explicit complete, continuation, contradictory, unknown,
repeated-cursor, and budget-exhausted pagination states. Legacy detail is used
only for a capability-approved modern `404` or `405`; authentication and
throttle responses never trigger fallback. Project IDs and branch/version
visibility are reported as coverage, never assumed complete.

Stage 2A does not implement SQLite storage/schema, attempt reconstruction,
accounting, quota windows, scheduling, local API, dashboard UI, exports, or
model reporting. Those commands remain deferred.

## Requirements

- Node.js `>=24.0.0`
- npm
- A dedicated Playwright Chromium profile for live operation

The fixture suite is synthetic and runs offline. No credentials or
browser profile are required for tests or fixture-backed CLI inspection.

## Install

```bash
cd scripts/chatgpt_chat_usage_capture/ts
npm ci
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
| `playwright` | `1.63.0` |
| `typescript` | `5.9.3` |
| `typescript-eslint` | `8.69.0` |
| `eslint` | `10.10.0` |
| `@eslint/js` | `9.39.2` |
| `@types/node` | `26.4.1` |
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
node bin/usage-capture.mjs backfill --config ./config.json
node bin/usage-capture.mjs refresh --config ./config.json
node bin/usage-capture.mjs reconcile --config ./config.json \
  --since 2026-09-01T00:00:00Z --until 2026-09-08T00:00:00Z
```

For offline CLI acceptance, use the committed synthetic config and fixtures:

```bash
node bin/usage-capture.mjs inspect-capabilities \
  --config ./tests/fixtures/v1/stage1-config.json \
  --fixture-root ./tests/fixtures/v1 \
  --state-directory /tmp/chatgpt-chat-usage-capture-stage1-state
```

The same fixture root can be used with `backfill`, `refresh`, or
`reconcile`. `backfill` defaults to the last 14 elapsed days. `--since` accepts
an elapsed duration such as `30d` or an ISO-8601 instant; `--until` is an
exclusive ISO-8601 instant. `reconcile` requires an explicit `--since` range.

The fixture command must use an account with
`browser.adapter: "fixture_history"` and never contacts ChatGPT. Live
`bootstrap`, `inspect-capabilities`, and history collection require
`browser.adapter: "playwright_persistent_context"` and a dedicated profile.

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
state is limited to sanitized identity metadata plus history checkpoints and
revisit IDs/timestamps under `bootstrap/<account-id>.json` and
`history/<encoded-account-id>.json`; browser state stays inside the dedicated
profile. No raw response body, message content, credential, cookie, or header
is written.

## Verify

The focused offline acceptance commands are:

```bash
npm ci
npm audit --audit-level=critical
npm run typecheck
npm test
npm run lint
npm run build
git diff --check
```

The fixture-backed transport provides deterministic synthetic session,
conversation-index, detail, and message-page responses without network access.
It is available to the offline capability and history collection CLI paths and
is not live endpoint evidence.

See:

- `docs/architecture.md`
- `docs/endpoint-evidence.md`
- `docs/privacy-threat-model.md`
- `docs/known-limitations.md`
- `docs/acceptance-results.md`
