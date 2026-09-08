# ChatGPT Chat usage capture: TypeScript Stage 1

This directory contains the bounded TypeScript Stage-1 implementation for
read-only ordinary Chat history bootstrap and capability inspection. It is an
independent collector boundary, not an official ChatGPT quota meter.

Stage 1 implements:

- `init`: write a starter JSON configuration;
- `bootstrap`: verify a dedicated persistent browser profile and bind the
  provider user, workspace, and quota owner;
- `inspect-capabilities`: read the active and archived conversation indexes and
  report adapter coverage. It supports live Playwright inspection and an
  explicit fixture-backed offline acceptance mode.
- `observe-native-history`: one bounded, attach-only authenticated history
  index observation through an explicit Oracle-owned CDP binding.

Stage 1 does not implement ledger storage, attempt reconstruction, accounting,
quota windows, scheduling, local API, dashboard UI, exports, or model
reporting. Those commands fail closed with an explicit Stage-2 error.

`observe-native-history` requires the live browser adapter, an existing
ChatGPT page in the attached context, an explicit CDP endpoint and page target
id, and the canonical12 inventory account hash. It creates one owned page,
issues one GET history-index request, and closes only that page before
disconnecting. It never launches a browser, retries, paginates, submits a
model request, or mutates account state. Authentication, throttle, challenge,
identity, shape, size, timeout, or boundary failures fail closed. Output is
structural only: fixed booleans, counters, and model-field presence; no
titles, IDs, payload content, headers, cookies, tokens, or storage values.

## Requirements

- Node.js `>=24.0.0`
- npm
- A dedicated Playwright Chromium profile for live operation

The Stage-1 fixture suite is synthetic and runs offline. No credentials or
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
| `playwright` | `1.63.0` |
| `typescript` | `5.9.3` |
| `typescript-eslint` | `8.69.0` |
| `eslint` | `10.10.0` |
| `@eslint/js` | `9.39.2` |
| `@types/node` | `26.4.1` |
| `vitest` | `3.2.4` |

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

For offline CLI acceptance, use the committed synthetic config and fixtures:

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
Stage-1 state is limited to sanitized identity metadata, adapter version,
state, and timestamp under `bootstrap/<account-id>.json`; browser state stays
inside the dedicated profile.

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
- `docs/known-limitations.md`
- `docs/acceptance-results.md`
