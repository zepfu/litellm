# Stage-2A acceptance results

Acceptance date: September 7, 2026.

The acceptance target is the TypeScript Stage-2A subtree only, from exact base
`dbea78d4a1`. The parent LiteLLM repository and its runtime services are
outside this gate.

## Dependency and contract pins

- Node engine: `>=24.0.0`
- Playwright: `1.63.0`
- TypeScript: `5.9.3`
- TypeScript ESLint: `8.69.0`
- ESLint: `10.10.0`
- `@eslint/js`: `9.39.2`
- `@types/node`: `26.4.1`
- Vitest: `3.2.7`
- Adapter version: `chatgpt-chat-history-v1`
- Fixture manifest version: `fixture-manifest-v1`
- History state version: `1`

## Commands

| Command | Result |
| --- | --- |
| `npm ci` | Passed; 146 packages audited, no vulnerabilities reported |
| `npm audit --audit-level=critical` | Passed |
| `npm run typecheck` | Passed |
| `npm test` | Passed; 55 tests across 10 focused unit/integration files |
| `npm run lint` | Passed |
| `npm run build` | Passed |
| Fixture CLI `init` / `inspect-capabilities` | Passed; generated config round-tripped, exit `0`, `ready`, active `2`, archived `0` |
| Fixture CLI `backfill` | Passed; default 14-day range, both scopes, two acquired conversations, durable checkpoints |
| Fixture CLI `reconcile` | Passed; explicit `2026-09-01` through `2026-09-08` range honored |
| Stage-2A focused tests | Passed; short-page/total contradiction, fallback gate, repeated cursors, budgets, late-visible history, revisits, overlap, project/branch coverage, durable store, and GET-only acquisition |
| `git diff --check` | Passed |

The critical-severity npm audit gate passed against the committed lockfile. No
dependency upgrade beyond the pinned Stage-1 dependency set was performed in this bounded
task.

## Evidence boundary

The tests use only synthetic fixtures and injected/mock browser transports. No
live ChatGPT account, credential, browser profile, protected container,
Anthropic route, Harness v2 path, or investigation file was accessed or
modified. All collector requests in the acquisition test were `GET`.

## Remaining Stage-2A gaps

- Live endpoint acceptance with an operator-authorized dedicated profile has
  not been run in this checkout.
- Playwright Chromium installation and a real authenticated session remain
  environment prerequisites for live bootstrap.
- SQLite observation/attempt ledger, attempt reconstruction, accounting,
  scheduler, API, and UI remain intentionally deferred to later stages.
