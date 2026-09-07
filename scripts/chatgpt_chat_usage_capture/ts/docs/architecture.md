# Stage-2 architecture

```text
                    one JSON config
                           |
             +-------------+-------------------+
             |                                 |
     bootstrap / inspect             backfill / refresh / reconcile
             |                                 |
             +----- GET-only history adapter --+
                   Playwright or fixtures      |
                   verified Chat identity      v
                                      history collector
                                active / archived / revisits
                                               |
                                      history/ingest.ts
                                               |
                                   one SQLite transaction
                                   observations / messages
                                   attempts / aliases / revisions
                                   coverage / runs / checkpoints
                                               |
                                  report / models / rebuild
                                    local evidence only
```

## Configuration and identity

`src/config.ts` loads schema-version-1 JSON and round-trips it through `init`.
One database-path resolver serves every collection and ledger CLI command.
Bootstrap's metadata-only identity JSON is separate from history state.

`src/normalize/identity.ts` requires configured provider user, workspace, and
quota-owner bindings to match the observed session. Missing expected values
are `unconfigured`; absent or unequal observed values are `identity_mismatch`.
Unauthenticated sessions are `auth_required`. Collection additionally requires
explicit observed `surface=chat`. No email, title, default account, or first
returned account is an identity guess. Ledger scope includes the local account,
provider, user, workspace, quota owner, and surface; the CLI rejects rebinding
an existing local account.

## Browser and adapter

`src/browser/session.ts` owns the dedicated Playwright persistent context.
Live access requires an existing profile unless interactive login was explicitly
authorized. Downloads are disabled, and browser storage never leaves the
profile. Fixture mode uses the same adapter without constructing a browser.

`src/adapters/chatgpt/adapter.ts` and both transports enforce exact GET-only
route shapes and safe conversation IDs. `401`, `403`, HTML auth pages, and
`429` are control signals, not history or remaining-quota observations.
Legacy detail fallback requires an approved capability and a modern `404` or
`405`; authentication and throttling never trigger fallback.

The shared `AdaptedPage.paginationState` and detail `detailRoute` /
`paginationState` contracts retain the Stage-2A route and pagination evidence:
complete, continuation, contradictory, unknown, repeated cursor, and exhausted
budget. Project IDs and version metadata qualify coverage, not completeness of
all Projects or branches.

## Acquisition and durability

`src/history/collector.ts` scans active and archived scopes, deduplicates
conversation IDs, selects candidates by update time, and fetches detail plus
message pages. Sparse message pages retain already observed graph links and
model metadata for the same message ID. Attempt attribution uses message
evidence, not the conversation update time.

Backfill defaults to 14 elapsed days. Implicit refresh uses each scope's last
complete discovery start with a 48-hour overlap. Explicit ranges take
precedence. Incomplete discovery retains its continuation without advancing the
completed watermark. Incomplete detail/message traversal creates a revisit
independent of the discovery cutoff.

`src/history/checkpoints.ts` buffers the checkpoint contract in memory.
`SqliteCheckpointStore` loads/saves the account's per-scope state and revisits
in the `history_state` table. There is no JSON history-store CLI path.
`src/history/ingest.ts` joins collection to the Stage-2B ledger: account binding,
empty mapping seed, run, evidence, reconstructed attempts, coverage, and
checkpoint changes commit together after acquisition. No SQLite write lock
spans browser requests. A failed commit rolls everything back; an interrupted
acquisition replays from the previous durable state. Commit granularity is one
bounded collection result, not each network page.

Index completion and detail completion remain separate. A completed index may
advance its discovery watermark when an incomplete conversation is safely
retained in the same committed revisit state.

## Ledger, reconstruction, and reports

`src/ledger/store.ts` applies three ordered SQLite migrations with WAL, foreign
keys, and a busy timeout. Source identity and stable sanitized fingerprints
deduplicate observations while changed evidence appends revisions. Message,
attempt, alias, mapping, and aggregate history remains available.

`src/normalize/reconstruct.ts` groups linked user, analysis, reasoning, tool,
and final nodes into generations. Distinct generation evidence preserves
regenerations; request IDs remain scoped grouping evidence. Terminal answers
are selected by timestamp. Missing linkage and ambiguous collisions remain
explicit rather than being guessed.

Requested, recorded-final, and resolved model labels are independent. The
initial mapping has canonical families but no slug rules. Reviewed exact
mapping versions and their history are separate from raw evidence.
`src/accounting/raw-model.ts` reports elapsed half-open intervals with
mismatches, ambiguous/unknown times, surface/origin exclusions, and open
coverage gaps. `src/accounting/reaggregate.ts` rebuilds from the same messages
and conversation-origin metadata; preview rolls back, apply commits a report
revision. Neither needs website access.

## Privacy and stage boundary

Allowlisted identifiers, typed metadata, route/pagination evidence, timestamps,
and model labels cross persistence boundaries. Message contents, titles,
credentials, cookies, raw headers, browser storage, and email do not.

Stage 2 has manual collection only. Scheduling, leases, cooldown/catch-up,
automated reconciliation, reset/quota accounting, API, UI, and exports are
deferred. The parent Python implementation is neither imported nor modified.
