# Endpoint evidence

Evidence date: September 7, 2026.

The live origin configured by the Playwright transport is
`https://chatgpt.com`. Stage 2A records the route contract in source, traverses
both index scopes, and exercises acquisition with reviewed synthetic fixtures.
This acceptance run does not use live credentials, a live browser session, or
protected containers.

## Read routes

| Method | Route shape | Stage-2A use | Evidence |
| --- | --- | --- | --- |
| `GET` | `/api/auth/session` | Verify provider user, workspace, quota owner, and surface | `SESSION_ROUTE`; valid/auth/mismatch fixtures; identity tests |
| `GET` | `/backend-api/conversations` | Read active or archived index with `offset`, `limit`, `order`, and `is_archived` | `MODERN_INDEX`; active/archived fixtures; collector integration tests |
| `GET` | `/backend-api/conversations/{conversation_id}` | Modern conversation detail | `MODERN_DETAIL`; conversation fixtures; adapter and collector tests |
| `GET` | `/backend-api/conversations/{conversation_id}/messages` | Message page and cursor coverage | `MODERN_MESSAGES`; message fixtures; repeated-cursor and budget tests |
| `GET` | `/backend-api/conversation/{conversation_id}` | Legacy detail fallback only after a capability-approved modern `404` or `405` | `LEGACY_DETAIL`; fallback gate unit tests |

`inspect-capabilities` currently requests the session route and the first
active and archived index pages. Stage-2A collection paginates each scope,
acquires modern detail/messages, and retains incomplete conversations for later
revisit. Index and message pages expose explicit `complete`, `continuation`,
`contradictory`, `unknown`, `repeated_cursor`, or `budget_exhausted` states.
Collection bounds are explicit and finite: the index page size, per-scope index
page budget, and per-conversation message-page budget are validated from config.

## Blocked routes and methods

`ChatGPTHistoryAdapter` and both transports reject every method other than
`GET` before issuing a request. They also reject:

- `/backend-api/conversation/init` because Stage 2A does not collect quota
  metadata;
- nested or action paths such as
  `/backend-api/conversations/{id}/delete`;
- unknown provider routes such as `/backend-api/me`;
- query-bearing or unsafe conversation path tokens.

Modern detail falls back only when both conditions hold: the response status is
`404` or `405`, and the caller has approved the adapter's legacy capability.
Authentication failures, `429`, and other statuses do not fall back. A `429` is
reported as transport rate limiting, not as a quota result.

## Coverage and state evidence

Active and archived scopes are tracked independently. Conversations appearing
in both scopes are deduplicated by scoped identity while retaining both scope
labels. A Project ID observed in an index is reported as
`validated_for_discovered_projects`; absence of a validated Project enumeration
remains `unknown`. `has_versions` is exposed as version metadata or
active-branch-only evidence; it is not treated as proof that every branch was
returned.

In fixture-tested Stage 2 behavior, the checkpoint store is the `history_state`
table in the configured SQLite database; it contains no response body or
browser state. Each fetched page and its continuation/evidence are committed
atomically. Successful earlier pages survive a later failure, and a restart
resumes from the durable continuation. Explicit backfill and reconciliation
ranges are stored and used as requested; the incremental watermark is only
consulted by an implicit refresh.

## Fixture provenance

`tests/fixtures/v1/manifest.json` identifies the fixture schema as
`fixture-manifest-v1`, the adapter as `chatgpt-chat-history-v1`, and the data
as synthetic. Fixture files contain no real account, credential, title, or
message content. `tests/fixtures/v1/stage1-config.json` is the offline CLI
acceptance config; it binds the synthetic identity and selects
`fixture_history`.
