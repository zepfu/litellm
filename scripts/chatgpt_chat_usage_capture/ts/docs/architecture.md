# Stage-2B architecture

The TypeScript slice keeps browser access, provider adaptation, identity
verification, privacy projection, local bootstrap state, and the evidence ledger
separate. Stage 2B operates on adapted records and retained local evidence; it
does not add browser discovery traversal.

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
             |
             v
   0600 bootstrap identity state
             |
             v
       SQLite evidence ledger
       observations/messages
       attempts/aliases/revisions
             |
             v
      raw-model reports/rebuilds
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
directory and are never serialized into Stage-1 state.

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
message-page pagination evidence, and legacy detail fallback behavior. The
capability record is versioned with `chatgpt-chat-history-v1`.

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

### Ledger and accounting

`src/ledger/store.ts` applies ordered SQL migrations to a local SQLite database
configured with WAL, foreign keys, and a busy timeout. Ingestion is
transactional: sanitized observations are revisioned by source identity and
stable evidence fingerprint; messages and attempts retain immutable revisions;
and aliases are scoped by account/provider/user/workspace/quota owner.

`src/normalize/reconstruct.ts` follows the user-to-descendant graph and groups
analysis, reasoning, tool, and final nodes by generation, then by request and
branch evidence. Terminal answers are selected by timestamp, and unresolved or
ambiguous linkage is retained as an explicit attempt or coverage gap.

`src/normalize/model-mapping.ts` keeps requested, recorded-final, and resolved
raw labels separate. Reviewed mapping versions are stored independently from
raw evidence. `src/accounting/reaggregate.ts` rebuilds from local messages and
`src/accounting/raw-model.ts` reports raw-model activity over a half-open
last-N interval without interpreting it as official quota.

### Stage boundary

The TypeScript implementation intentionally stops before browser discovery
traversal, scheduler, reset-window accounting, provider quota accounting, local
API, dashboard, and exports. The Python implementation in the parent directory
remains outside this lane and is not imported or modified by this package.
