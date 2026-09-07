# Stage-2B acceptance results

Acceptance date: September 7, 2026.

The acceptance target is the TypeScript Stage-2B subtree only. The parent
LiteLLM repository and its runtime services are outside this gate.

## Dependency and contract pins

- Node engine: `>=24.0.0`
- better-sqlite3: `13.0.3`
- Playwright: `1.63.0`
- TypeScript: `5.9.3`
- TypeScript ESLint: `8.69.0`
- ESLint: `10.10.0`
- `@eslint/js`: `9.39.2`
- `@types/node`: `26.4.1`
- `@types/better-sqlite3`: `9.6.0`
- Vitest: `3.2.7`
- Adapter version: `chatgpt-chat-history-v1`
- Fixture manifest version: `fixture-manifest-v1`

## Commands

| Command | Result |
| --- | --- |
| `npm ci` | Passed; 148 packages installed, 0 vulnerabilities |
| `npm run typecheck` | Passed |
| `npm test` | Passed; 46 tests across 9 focused unit/integration files |
| `npm run lint` | Passed |
| `npm run build` | Passed |
| `npm audit --audit-level=critical` | Passed; 0 vulnerabilities |
| `git diff --check` | Passed |
| `node bin/usage-capture.mjs init ...` | Passed; generated config round-tripped |
| Fixture CLI `inspect-capabilities` | Passed offline; exit `0`, `ready`, active `2`, archived `0` |
| Launcher smoke without a profile | Passed; inspect `browser_unavailable`, bootstrap `auth_required`, both exit `1` |
| Stage-2B integration scenarios | Passed; scoped identity, chained generations, timestamp ordering, revisions, sanitized ingestion, mapping history, deterministic rebuild, and raw-model interval reporting |

The critical audit gate was clear for the committed dependency set. No
dependency upgrade outside this lane was performed.

## Evidence boundary

The tests use only synthetic fixtures and mocked browser transports. The
Stage-2B rebuild test verifies that reaggregation uses retained local messages
without website requests. No live ChatGPT account, credential, browser profile,
protected container, Anthropic route, Harness v2 path, or investigation file
was accessed or modified.

## Remaining Stage-2B gaps

- Live endpoint acceptance with an operator-authorized dedicated profile has
  not been run in this checkout.
- Playwright Chromium installation and a real authenticated session remain
  environment prerequisites for live bootstrap.
- Browser discovery traversal, scheduler, reset-window/provider quota
  accounting, API, UI, and exports remain intentionally deferred to later
  stages.
