# Stage-1 architecture

The TypeScript slice keeps browser access, provider adaptation, identity
verification, privacy projection, and local bootstrap state separate.

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
          GET-only route transport
                    |
                    v
          ChatGPT history adapter
             |                  |
             v                  v
      identity verifier   sanitized page records
             |
             v
   0600 bootstrap identity state
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

### Adapter

`src/adapters/chatgpt/adapter.ts` accepts only `GET` and only the exact
history/session route shapes documented in `endpoint-evidence.md`. It treats
HTTP `401`, `403`, HTML login pages, and `429` as control signals. A `429` is a
transport rate limit and is never interpreted as a ChatGPT quota exhaustion
signal.

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

### Stage boundary

The TypeScript implementation intentionally stops before ledger/accounting,
attempt reconstruction, scheduler, local API, dashboard, and exports. The
Python implementation in the parent directory remains the reference for those
later stages; it is not imported by this package.
