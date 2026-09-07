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
                                   per-page SQLite transactions
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
returned account is an identity guess. Activity scope includes provider, user,
workspace, quota owner, and surface. Collectors for the same verified owner
share activity while retaining collector provenance. Unverified scopes and
discovery/checkpoint state remain collector-local. The CLI rejects rebinding
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
clean implicit-refresh discovery start with a 48-hour overlap, without clamping
an old watermark to the default range. Explicit ranges take precedence;
historical scans do not advance refresh watermarks. Continuations resume only
when frozen acquisition metadata and the reread head match. Explicit discovery
includes conversations updated beyond the range end. Full message evidence is
retained; report membership is bounded by attempt evidence, not acquisition time.
Leading-page rereads leave changing indexes partial. Incomplete detail/message
traversal and nonterminal generations retain revisits independently of cutoff.

`src/history/checkpoints.ts` buffers the checkpoint contract in memory.
`SqliteCheckpointStore` loads/saves the account's per-scope state and revisits
in the `history_state` table. There is no JSON history-store CLI path.
`src/history/ingest.ts` joins collection to the Stage-2B ledger: account binding,
empty mapping seed, run, evidence, reconstructed attempts, coverage, and
checkpoint changes commit together for each acquired page. No SQLite write
lock spans browser requests. A failed page commit rolls back that page; prior
successful pages and their continuations remain durable. Bad saved message
continuations allow one bounded restart.

Index completion and detail completion remain separate. Clean implicit
discovery advances its watermark once candidates are durably queued, even when
detail work remains. Failed evidence commits retain candidates for restart.
Observed Project and branch metadata still leave
global visibility unproven. Older-history audit rotation is opt-in through the
collector request API; audit completion requires successful candidate acquisition.

## Ledger, reconstruction, and reports

`src/ledger/store.ts` applies six ordered SQLite migrations with WAL, foreign
keys, and a busy timeout, checking applied migration names and checksums.
Pending schemas and owner-scope conversion commit atomically; conversion runs
after all required columns exist. Quarantine provenance survives revision replay.
Migration 4 enables occurrence revisions and provenance; migration 5 adds
mapping lifecycle columns and transactionally migrates legacy activity scopes.
Legacy evidence IDs and revision occurrences remain available; duplicate owner
activity converges, and checkpoints remain collector-local. Consecutive
identical evidence from one collector deduplicates; A-B-A changes retain all
three occurrences. Older evidence does not replace newer current projections.

`src/normalize/reconstruct.ts` groups linked user, analysis, reasoning, tool,
and final nodes into generations. Distinct generation evidence preserves
regenerations; request IDs remain scoped grouping evidence. Terminal answers
are selected by timestamp. Missing linkage and ambiguous collisions remain
explicit rather than being guessed.

Requested, recorded-final, and resolved model labels are independent. The
initial mapping has canonical families but no slug rules. Published mappings
are immutable, use event-time validity, and distinguish prospective changes
from historical corrections. Changed raw evidence uses the applicable stored
mapping when the selected mapping is not yet applicable; without one, family
fields remain unresolved. Collector-specific overrides are rejected when they
would produce different results for collectors sharing one canonical owner.
Rebuild preserves mapping warnings, lifecycle bounds, linkage, and retired
duplicates. Mapping history is separate from raw evidence.
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
