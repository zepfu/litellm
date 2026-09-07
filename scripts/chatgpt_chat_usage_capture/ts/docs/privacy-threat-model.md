# Stage-2 privacy and threat model

## Assets

- Authentication cookies, tokens, and storage inside the dedicated profile.
- Provider user, workspace, quota-owner, conversation, and message identifiers.
- Model, mode, reasoning, timestamp, status, generation, request, branch, route,
  coverage, and revision metadata.
- Config, bootstrap identity state, SQLite history/ledger, and WAL companions.

## Controls

### Isolation and identity

Live access requires a dedicated non-default profile. A missing profile is
created only after explicit interactive-login authorization. Profile/state
directories are created with mode `0700`; bootstrap identity JSON uses mode
`0600` where supported. Browser state stays in the profile.

The SQLite parent directory is created with mode `0700`. Protect existing
directories and backups as well as the database and `-wal`/`-shm` files.
WAL, foreign keys, a busy timeout, and transactional migrations support local
ledger consistency, not host-level security.

Provider user, workspace, and quota owner must match before collection.
Acquisition requires explicit ordinary Chat surface. Account rebinding is
rejected by the integrated CLI. Attempts share verified-owner activity scope
with per-collector provenance; checkpoints remain collector-local.

### Read-only egress

The adapter and both transports enforce GET-only exact route shapes and safe
conversation ID tokens. They do not submit prompts, mutate conversations, call
the init route, or export browser storage. Legacy fallback requires an approved
capability and modern `404`/`405`, never auth or throttling responses.

### Data minimization

The adapter projects only allowlisted metadata. Titles, prompt/answer content,
tool arguments/results, credentials, cookies, raw headers, storage, and email
are stripped or rejected before persistence. Normalized messages retain
relationship evidence even when a later message page omits graph fields.

Sanitized observations, messages, mapping records, coverage details,
checkpoints, and reports pass secret checks before writes. Fingerprints provide
idempotence; changed observations/messages/attempts retain revisions. Route,
pagination, and coverage evidence accompanies the retained data.

The integration CLI prints metadata and per-conversation message counts, not
message bodies. Browser authentication is never copied into SQLite, checkpoint
JSON, fixtures, diagnostics, or reports.

### Atomic local state

History state is buffered during browser requests and commits with evidence,
attempts, and run metadata in a transaction per acquired page. A failed write
cannot advance a checkpoint past missing evidence. No write lock is held across
network requests. Interrupted uncommitted pages are replayed.

### Fixture isolation

Fixtures and offline CLI commands are synthetic. They do not open a user
profile or make website requests. Reports and rebuilds consume only local
ledger data.

## Threats and residual risk

| Threat | Response | Residual risk |
| --- | --- | --- |
| Mutating provider call | GET-only exact route allowlist | Future adapter changes must preserve it |
| Wrong account attribution | Verified binding and scoped ledger state | Provider identity schema can become unavailable |
| Content or credential persistence | Typed allowlists and secret checks | Raw upstream responses exist transiently in memory |
| Shared/default profile exposure | Dedicated profile and explicit login | OS permissions and host compromise remain external |
| Auth challenge mistaken for history | HTML, `401`, and `403` control signals | Live detection may require updates |
| Throttling mistaken for quota | `429` is a transport error | No retry scheduler or quota accounting exists |
| Schema/pagination drift | Coverage, warnings, and revisits | Unknown future shapes remain uninterpreted |
| Cross-account ID collision | Scope-keyed attempts and aliases | Upstream linkage may remain ambiguous |
| Interrupted ingestion | Evidence/checkpoint transaction | Uncommitted pages need replay |
| Rebuild changes origin attribution | Same conversation-origin evidence in ingestion and rebuild | Unexposed provider origin cannot be inferred |
| Mapping overclaim | Empty seed, raw labels, review history | Local mapping does not establish billing semantics |

This does not defend against a compromised host, Node process, dependency, or
user with access to the profile. No live credential/profile validation was part
of the integration gate.
