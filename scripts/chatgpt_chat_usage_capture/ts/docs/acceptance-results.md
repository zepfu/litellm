# Stage-1 acceptance results

Acceptance date: September 7, 2026.

The acceptance target is the TypeScript Stage-1 subtree only. The parent
LiteLLM repository and its runtime services are outside this gate.

## Dependency and contract pins

- Node engine: `>=24.0.0`
- Playwright: `1.63.0`
- TypeScript: `5.9.3`
- TypeScript ESLint: `8.69.0`
- ESLint: `10.10.0`
- `@eslint/js`: `9.39.2`
- `@types/node`: `26.4.1`
- Vitest: `3.2.4`
- Adapter version: `chatgpt-chat-history-v1`
- Fixture manifest version: `fixture-manifest-v1`

## Commands

| Command | Result |
| --- | --- |
| `npm install` | Passed; installed 147 packages from the committed lockfile |
| `npm run typecheck` | Passed |
| `npm test` | Passed; 38 tests across 8 focused unit/integration files |
| `npm run lint` | Passed |
| `npm run build` | Passed |
| `node bin/usage-capture.mjs init ...` | Passed; generated config round-tripped |
| Fixture CLI `inspect-capabilities` | Passed offline; exit `0`, `ready`, active `2`, archived `0` |
| Launcher smoke without a profile | Passed; inspect `browser_unavailable`, bootstrap `auth_required`, both exit `1` |
| `git diff --check` | Passed |

The npm install output reported one dependency audit finding. No audit
remediation or dependency upgrade beyond the pinned Stage-1 set was performed
in this bounded task.

## Evidence boundary

The tests use only synthetic fixtures and mocked browser transports. No live
ChatGPT account, credential, browser profile, protected container, Anthropic
route, Harness v2 path, or investigation file was accessed or modified.

## Remaining Stage-1 gaps

- Live endpoint acceptance with an operator-authorized dedicated profile has
  not been run in this checkout.
- Playwright Chromium installation and a real authenticated session remain
  environment prerequisites for live bootstrap.
- Complete conversation traversal, usage ledger/accounting, scheduler, API, and
  UI remain intentionally deferred to later stages.
