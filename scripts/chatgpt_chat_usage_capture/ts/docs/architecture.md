# Stage-2A architecture

The TypeScript slice keeps browser access, provider adaptation, identity
verification, privacy projection, history discovery, and durable collection
state separate.

```text
CLI
  |
  v
JSON config -> bootstrap state machine
                    |
                    v
        Playwright persistent context
                    |
                    v
          GET-only request boundary
                    |
                    v
          ChatGPT history adapter
             |                  |
             v                  v
      identity verifier   sanitized page records
             |                  |
             +----------+-------+
                        v
              Stage-2A history collector
              active | archived | revisits
                        |
                        v
             0600 JSON checkpoints
```

## Boundaries

### Configuration

`src/config.ts` reads schema version `1` JSON and maps snake_case file fields
to typed internal values. `init` writes the same snake_case wire shape, so a
generated config can be loaded without a manual format conversion.

### Dedicated browser session

`src/browser/session.ts` owns the Playwright persistent context. A live request
requires an existing dedicated profile. A missing profile is created only
after `--interactive-login` is explicitly supplied. Downloads are disabled.
The browser context and its cookies, storage, and tokens remain in the profile
directory and are never serialized into collector state.

For offline acceptance, the CLI can select `fixture_history` with an explicit
`--fixture-root`. That path uses the same adapter against synthetic JSON and
does not construct a Playwright context. Live bootstrap and live capability
inspection reject the fixture adapter.

### Adapter

`src/adapters/chatgpt/adapter.ts` accepts only `GET` and only the exact
history/session route shapes documented in `endpoint-evidence.md`. It treats
HTTP `401`, `403`, HTML login pages, and `429` as control signals. A `429` is a
transport rate limit and is never interpreted as a ChatGPT quota exhaustion
signal. `assertAllowedRequest` is applied by the adapter and by both the
Playwright and fixture transports before any request is issued.

The adapter exposes active and archived index coverage, modern detail support,
message-page pagination evidence, project/version visibility, and legacy detail
fallback behavior. Pagination controls are normalized into explicit states;
short pages contradicting a larger reported total remain partial and
resumable. The capability record is versioned with
`chatgpt-chat-history-v1`. A modern detail `404` or `405` reaches the legacy
route only when the caller has approved that capability; `401`, `403`, `429`,
and other statuses do not.

Stage 2A extends the committed Stage-1 contracts without removing existing
fields: `AdaptedPage` adds `paginationState`, and
`ConversationDetailProjection` adds `detailRoute` plus `paginationState`.
These fields preserve route provenance and distinguish validated completion
from continuation, contradiction, repetition, budget exhaustion, or unknown
pagination.

### History collection

`src/history/collector.ts` is the Stage-2A acquisition boundary. It scans both
active and archived scopes, deduplicates conversation IDs across scopes, uses
conversation update time for candidate discovery, and then fetches message
metadata. It never reconstructs attempts or writes accounting totals.

`backfill` uses a 14-day elapsed range by default and accepts arbitrary
half-open UTC ranges. `refresh` computes a per-scope discovery cutoff from the
last complete discovery start minus 48 hours. An explicit backfill or
reconciliation range always wins over a stored watermark. Discovery completion
and detail completion are separate: an incomplete detail remains in the
revisit queue even when the index scan watermark advances.

`src/history/checkpoints.ts` provides a small atomic JSON implementation of the
Stage-2A checkpoint interface. Each scope stores its continuation, page
budget, pagination state, range, warnings, and last complete discovery start.
The same state file stores outstanding conversation revisits. This is an
intermediate Stage-2A boundary; it is not the later SQLite ledger.

### Identity

`src/normalize/identity.ts` requires all three configured bindings:

1. provider user ID;
2. workspace ID;
3. quota owner ID.

An authenticated response with a missing expected binding is `unconfigured`.
An observed value that is absent or different is `identity_mismatch`.
Unauthenticated responses are `auth_required`. No email, page title, first
account, or default ID is used as an identity guess.

### Privacy

`src/security/sanitizer.ts` projects only allowlisted identifiers, typed
metadata, status, timestamps, model labels, and relationship IDs. Message
content, titles, prompt/answer fields, credentials, cookies, raw headers,
browser storage, and email addresses are stripped or rejected at the
persistence boundary. `assertNoSecrets` checks every persisted identity
projection before it is written.

### Stage boundary

The TypeScript implementation intentionally stops before SQLite
ledger/accounting, attempt reconstruction, quota windows, scheduler, local API,
dashboard, and exports. The Python implementation in the parent directory
remains the reference for those later stages; it is not imported by this
package.
