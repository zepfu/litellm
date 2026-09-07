# Endpoint evidence

Evidence date: September 7, 2026.

The live origin configured by the Playwright transport is
`https://chatgpt.com`. Stage 1 records the route contract in source and
exercises it with reviewed synthetic fixtures. This acceptance run does not
use live credentials, a live browser session, or protected containers.

## Read routes

| Method | Route shape | Stage-1 use | Evidence |
| --- | --- | --- | --- |
| `GET` | `/api/auth/session` | Verify provider user, workspace, quota owner, and surface | `SESSION_ROUTE`; valid/auth/mismatch fixtures; identity tests |
| `GET` | `/backend-api/conversations` | Read active or archived index with `offset`, `limit`, `order`, and `is_archived` | `MODERN_INDEX`; active and archived fixtures; capability tests |
| `GET` | `/backend-api/conversations/{conversation_id}` | Modern conversation detail | `MODERN_DETAIL`; conversation fixtures; adapter tests |
| `GET` | `/backend-api/conversations/{conversation_id}/messages` | Message page and cursor coverage | `MODERN_MESSAGES`; message fixtures; adapter tests |
| `GET` | `/backend-api/conversation/{conversation_id}` | Legacy detail fallback after modern `404` or `405` | `LEGACY_DETAIL`; fallback branch in adapter |

`inspect-capabilities` currently requests the session route and the first
active and archived index pages. It does not claim that every conversation
detail or historical cursor has been live-validated.

## Blocked routes and methods

`ChatGPTHistoryAdapter` and both transports reject every method other than
`GET` before issuing a request. They also reject:

- `/backend-api/conversation/init` because Stage 1 does not collect quota
  metadata;
- nested or action paths such as
  `/backend-api/conversations/{id}/delete`;
- unknown provider routes such as `/backend-api/me`;
- query-bearing or unsafe conversation path tokens.

Modern detail falls back only on `404` or `405`. Authentication failures and
`429` responses do not fall back to another route. A `429` is reported as
transport rate limiting, not as a quota result.

## Fixture provenance

`tests/fixtures/v1/manifest.json` identifies the fixture schema as
`fixture-manifest-v1`, the adapter as `chatgpt-chat-history-v1`, and the data
as synthetic. Fixture files contain no real account, credential, title, or
message content. `tests/fixtures/v1/stage1-config.json` is the offline CLI
acceptance config; it binds the synthetic identity and selects
`fixture_history`.
