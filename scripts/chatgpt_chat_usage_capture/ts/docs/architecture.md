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
message-page pagination evidence, and legacy detail fallback behavior. It
reconciles all nested pagination-control aliases before use, marks conflicting
controls as contradictory, and never treats a full page without terminal or
total evidence as exhaustion. HTTP 200 HTML authentication challenges are
classified as authentication-required before generic adapter handling. The
capability record is versioned with `chatgpt-chat-history-v1`.

Legacy mapping-only detail responses are an established terminal shape and are
complete only when reached through the capability-approved legacy route.
Modern mapping-only responses without `page_info` remain partial/unknown with
no continuation, so they cannot be mistaken for complete history. Summary and
message origin evidence is resolved together from root, mapping wrapper, and
metadata fields: strict `imported`, `from_copy`, and `from_shared` exclusion
flags take precedence over any benign `origin` label, and conflicting or
malformed labels resolve to unknown origin. These origins remain excluded from
ordinary Chat accounting.

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
persistence boundary. Allowlisted evidence is projected before unknown-field
diagnostics consume traversal budget, and incomplete projection is propagated
as a page warning rather than silently losing model or generation metadata.
`assertNoSecrets` checks every persisted identity projection before it is
written.

### Stage boundary

The TypeScript implementation intentionally stops before ledger/accounting,
attempt reconstruction, scheduler, local API, dashboard, and exports. The
Python implementation in the parent directory remains the reference for those
later stages; it is not imported by this package.
