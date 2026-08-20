# AAWM Provider Status Observations

## MS-031: shared Kimi native contract

The authoritative capture source is the WSL
`/home/zepfu/.kimi-code/bin/kimi` installation at version `0.29.1`. The paused
Thoth updater in `MS-032` is not a prerequisite and must not be resumed or used
to publish a descriptor without separate operator authorization.

The Kimi usage poller is one consumer of the same sanitized native contract
descriptor used by managed chat, the Codex adapter, the Claude adapter, and
the raw gateway. The descriptor contains the exact endpoint
`https://api.kimi.com/coding/v1`, native client name/version, the native
`User-Agent` and header names, service-owned non-personal identity values,
issued/expiry timestamps, and an integrity digest. Managed model allowlisting
remains in existing `kimi_code` provider metadata/config and is not part of the
descriptor.

The descriptor contains no token, personal device, account, session, or caller
identity. Reject caller attempts to spoof its endpoint, native header/version,
freshness, digest, or identity fields. Replacement is validated and atomically
published without a LiteLLM restart. The configured host directory is mounted
read-only at `/app/kimi-descriptor`, and the consumer path is
`/app/kimi-descriptor/native-contract.json`.

The standalone image default
`LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED=false` is compatibility mode. Managed
dev Compose defaults the gate to `true`. Required mode requires a resolved
Kimi identity, not a fresh descriptor. An expired but structurally valid
descriptor remains usable as the last valid identity and records sanitized
stale-source telemetry. If the descriptor is missing, malformed,
digest-invalid, endpoint-mismatched, version-incoherent, or hostile, installed
Kimi remains available through the conservative built-in identity with
sanitized source telemetry. Neither fallback may use generic `moonshot/*`,
API-key authentication, another provider, or another endpoint.

The descriptor does not define usage-to-chat correlation. Any such join must
be proved by the polling and persistence runtime evidence; prompt text,
caller-supplied identity, and model-name guesses are not descriptor join keys.

The provider-status sidecar records non-inference front-door health signals for
configured providers. It probes DNS, TCP/TLS, and optional ICMP paths, then
writes rows to `public.provider_status_observations`.

## COHERE-002: direct native usage observations

Accepted direct Cohere terminal HTTP 200 `/v2/chat` calls are counted exactly
once by stable `litellm_call_id`. The immutable event ledger is
`public.locally_counted_accepted_calls`; snapshots still go to
`rate_limit_observations` as `source=locally_counted`, not provider-reported.
Monthly usage is shared per credential with a locally enforced monthly limit of
1000; numeric `quota_used` is counted from `accepted_at`. RPM usage is
exact-model and is compared with model metadata. Missing or unknown RPM
metadata, and missing or non-numeric usage, remain unknown and do not block.
Stale or reset observations are ignored.

This contract applies only to `provider=cohere`, `lane=cohere_native`,
`credential_scope=cohere_trial_default`, and the direct Codex Cohere route.
OpenRouter free daily remains on the `session_history` meter until a later
cutover. OpenCode Zen and NVIDIA NIM have no local numeric policy in this
commit. Anthropic adapter integration and Anthropic testing or acceptance are
not part of Cohere work. Migration, deployment, and authenticated acceptance
have not been performed; each remains separately authorized.

`scripts/run_provider_status_observations_loop.py` emits one
`provider_status_observations_cycle` JSON line per cycle. The aggregate fields
include:

- `row_count`
- `success_count`
- `failure_count`
- `inserted`
- `skipped`
- `skip_error_class`
- `skip_reason`

When `failure_count > 0`, the cycle log also includes bounded
`failure_summaries`. Each entry contains only:

- `provider`
- `endpoint_key`
- `probe_type`
- `error_class`
- redacted and truncated `error_message`

The summaries intentionally omit raw metadata, provider payloads, prompts,
resolved IP details, and credentials. If more failed rows exist than the
summary cap, `failure_summaries_omitted_count` records how many failed rows were
not included in the log payload. All-green cycles omit the summary fields so
normal logs stay compact.

## Passive OAuth Credential Health

The sidecar can inspect the existing Grok OIDC, Codex OAuth, xAI OAuth, and
Kimi OAuth credential files without refreshing or modifying them. This mode is
intended for read-only consumers such as Thoth, where another host remains the
single credential writer but `provider_auth_observations` and
`provider_auth_current` still need fresh health state.

The inspection performs local file reads only. It does not acquire credential
locks, repair metadata, write files, or make network calls. Each non-Codex
provider emits one sanitized event. Codex emits one event per enabled inventory
record plus `codex_oauth_health_aggregate`. Health is `healthy` when every
enabled record is fresh, `degraded` when some remain usable, and `terminal`
when none are usable. Events omit credential paths, raw account IDs, and all
token values. Persisted rows use `source_task=provider_auth_health_poll` and
metadata flags `passive_read_only=true`, `network_calls=false`, and
`credential_file_mutated=false`.

Relevant environment variables:

- `AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED`: enables the passive inspection.
  Defaults to disabled so existing credential-refresh ownership is unchanged.
- `AAWM_PROVIDER_AUTH_HEALTH_POLL_INTERVAL_SECONDS`: minimum seconds between
  inspections; defaults to `3600`.

## Deadline-Aware OAuth Refresh Scheduling

The scheduled Grok OIDC, managed xAI OAuth, Kimi OAuth, and Codex OAuth tasks
have two separate timing controls:

- The outer eligibility cadence is the provider-status loop cadence
  (`AAWM_PROVIDER_STATUS_INTERVAL_SECONDS`). On each eligible outer cycle, the
  sidecar performs a read-only credential inspection and derives
  `refresh_due_at` from the credential expiry and the provider refresh buffer.
  Kimi uses `max(300, expires_in * 0.5)` and reports that effective value as
  `refresh_threshold_seconds`. A `refresh_not_due` inspection does not consume
  the endpoint-attempt throttle. A local or pre-network helper failure also
  leaves that throttle untouched and may retry on the next outer cycle. Every
  outer cycle reinspects the current credential pathname, including an inode
  replaced by an external credential writer.
- The actual token-endpoint attempt throttle is per task, and for Codex is
  isolated per inventory label. The scheduler passes an explicit
  `on_token_endpoint_attempt` callback to the refresh helper. The helper invokes
  that callback immediately before the outbound token endpoint request, after
  local validation, lock acquisition, and request construction. Only that
  callback updates the monotonic last-attempt timestamp. Legacy `attempted`
  summary flags are descriptive evidence and are not throttle authority.

Refresh events include sanitized scheduler evidence such as
`eligibility_checked_at`, `expires_at`, `refresh_due_at`,
`next_refresh_check_at`, `last_actual_attempt_at`, `actual_attempted`,
`actual_attempt_count`, `eligibility_cadence_seconds`,
`refresh_attempt_interval_seconds`, `refresh_buffer_seconds` or
`refresh_threshold_seconds`, `credential_health`, `usable`, and redacted
error fields. The result classes are:

- `refresh_not_due`: the credential remains outside its refresh deadline. A
  forced operation failure can still retain this class when the credential
  remains not-due; the scheduler records the failure metadata while leaving the
  credential usable and healthy.
- `refresh_due`: the credential is due and no endpoint failure was recorded.
- `refresh_failed`: an eligible helper operation failed before or during the
  actual token-endpoint attempt while the credential was not already expired.
  An outer cycle skipped by the actual-attempt throttle retains this failure
  and degraded health until a later inspection or operation changes the result.
- `expired`: expiry overrides a refresh failure and the credential is not
  usable.

Missing or malformed expiry data never makes a credential permanently fresh.
An otherwise usable credential with missing or unparseable `expires_at` is
eligible for the safe refresh path and is reported as degraded. A malformed or
unreadable credential is reported as malformed and unusable until recovery.
Inspection and operation errors are bounded and redacted before they enter
events or persisted auth observations.

The pre-refresh inspection and post-refresh inspection are both retained in
the scheduler evidence. The post-refresh file is authoritative when a helper
replaces the credential with an earlier or later expiry. If the post-refresh
read is transiently unavailable, the scheduler retains the pre-refresh expiry
and deadline rather than discarding known state. The attempt throttle is
process-local monotonic state. After a restart, the current credential is
re-inspected from wall-clock expiry and no stale pre-restart timestamp is
reused.

Credential replacement is lock-protected and atomic: the refresh helper writes
the complete sanitized credential payload to a private same-directory
temporary file, applies the resolved private metadata, and atomically replaces
the target. Read-only LiteLLM consumers re-read the replaced inode and do not
need a restart. Runtime and persisted evidence may retain provider, scope,
result class, expiry, safe Codex label/hash, and bounded error metadata, but
must not contain access, refresh, or ID tokens, authorization headers, the full
credential payload, or a reversible account identity. Persisted auth records
use a hashed file identity rather than the raw credential path.

The managed xAI dev Compose contract is exact:

- `AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS=300`
- `AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS=900`
- `AAWM_XAI_OAUTH_FORCE_REFRESH=0`

These are checked-in development configuration defaults. They are not
deployment or runtime acceptance evidence.

## Codex OAuth Inventory Boundary

Managed Codex proxy consumers use only the explicit
`LITELLM_CODEX_OAUTH_INVENTORY`. Development Compose enrolls ordered
`account1` and `account2` records with separate auth files, separate lock
files, stable non-secret labels, explicit model eligibility, and
operator-supplied expected account hashes. It supplies the same inventory to
the provider-status sidecar so deployment configuration cannot silently drift.

The LiteLLM proxy mounts `/home/zepfu/.codex` read-only. The sidecar mounts that
directory read-write for lock ownership and atomic credential publication.
Neither mount enrolls files: only paths named in the inventory are eligible,
and there is no `~/.codex/auth.json`, glob, backup-file, or `api.openai.com`
fallback for proxy requests.

The sidecar loads this inventory whenever Codex refresh, passive health, or
quota polling is enabled. Every enabled record has independent refresh,
passive-health, and quota timers. One timeout, malformed/revoked credential,
identity mismatch, or lock conflict is recorded for that label and does not
suppress later records, including an otherwise idle `account2`. Refresh and
health aggregates are `healthy` when every enabled record is usable,
`degraded` when some are usable, and `terminal` when none are usable. Events
emit only the configured label and pinned safe hash for account identity.
Refresh scheduler evidence and actual-attempt throttles remain independent for
each label; one account's due check, failure, or endpoint attempt does not
consume the other account's timer.

OPENAI-004 request routing consumes the same explicit inventory plus the
sidecar's fresh per-account five-hour and weekly/seven-day quota observations.
The proxy selects only enabled, auth-healthy, model-eligible accounts;
treats only fresh confirmed exhaustion as terminal for an account; allows at
most one immediate pre-first-byte account move; pins continuations and fails
fast on the pinned account; returns a structured safe `429` when no account is
admissible; and publishes only label/hash/lane/failover metadata without raw
IDs or secrets. Managed Codex OAuth dispatch never falls back to
`api.openai.com` API keys or unlisted credential files. Full operator contract:
`docs/aawm-oauth-credential-maintenance.md` section
`Codex multi-account request routing (OPENAI-004)`.

Enrollment, removal, label/hash handling, permissions, rotation, and rollback
are defined in `docs/aawm-oauth-credential-maintenance.md`.

## Grok banked usage-limit resets (XAI-005 / XAI-006 / XAI-007)

The sidecar can poll grok.com `GetRemainingResets` for banked usage-limit reset
tokens. This is independent of the hourly Grok CLI billing poll. Auth reuses
the existing Grok OIDC file (`AAWM_GROK_OIDC_AUTH_FILE`); there is no separate
cookie file, `AAWM_GROK_WEB_AUTH_*` variable, or grok-web cookie mount.

The request is stdlib gRPC-Web: `POST` to
`https://grok.com/prod_mc_billing.ConsumerUiSvc/GetRemainingResets` with Bearer
OIDC only (never Cookie plus Bearer), an empty 5-byte gRPC-Web frame, and
`User-Agent: aawm-provider-status-observations`. The poller never calls
`RedeemReset` and never substitutes managed `oa_xai` OAuth on grpc-status 16.

Parser outcomes:

- empty data frame plus `grpc-status 0` is a known zero inventory
- missing tokens, truncated frames, leftover bytes, or nonzero grpc-status
  other than 16 are unknown and do not persist a synthetic empty inventory
- grpc-status 16 / HTTP 401/403 is `reauthentication_required`; last good
  credit state is retained and xAI OAuth is not invoked

Successful inventories persist hashed rows into the existing
`provider_credit_*` tables:

- `provider=xai`
- `credit_family=xai_usage_limit_reset`
- `credit_type=usage_reset_token`
- `source=xai_grok_web_remaining_resets`
- `parser_version=xai_grok_web_remaining_resets_v1`

Credit identity uses `derive_provider_credit_identity(..., hash_provider_credit_id=True)`.
Account identity is `probes.account_identity_hash(user_id)` and never the raw
user id. Persist only nonempty token ids with `validity_end` still in the
future. Missing before expiry synthesizes `used`; at or after expiry
synthesizes `expired`. Failed/unknown polls skip synthesis. A known zero
inventory synthesizes used/expired for previously available rows.

Relevant environment variables:

- `AAWM_XAI_RESET_POLL_ENABLED`: defaults to disabled (`0` / `false`).
- `AAWM_XAI_RESET_POLL_INTERVAL_SECONDS`: defaults to `3600`.
- `AAWM_XAI_RESET_POLL_HTTP_TIMEOUT_SECONDS`: defaults to `30`.
- `AAWM_XAI_RESET_POLL_URL`: defaults to the grok.com `GetRemainingResets`
  endpoint. Attempts/backoff match Grok billing (`3` / `0.5s`).

Each due attempt emits a sanitized `xai_reset_poll` event including
`last_good_state_retained`. Events must not contain access tokens,
Authorization headers, cookies, or raw token ids. A reset-poll failure does
not fail `grok_billing_poll`.

## Grok OIDC Refresh Task

The same sidecar can also own the scheduled Grok native OIDC credential refresh.
This is separate from the five-minute provider front-door probes. In shared
multi-provider compose layouts the sidecar may mount `/home/zepfu/.grok`
writable. On the operator WSL host, however, native Grok OIDC refresh is owned
by the dedicated WSL-local dual credential writer described in
[WSL-local dual credential writer (XAI-003 / XAI-004)](#wsl-local-dual-credential-writer-xai-003--xai-004)
so multi-provider sidecars and both LiteLLM proxies stay non-writers for both native Grok OIDC and managed `oa_xai/*` OAuth. LiteLLM
mounts that directory read-only and reads the credential directly for
`xai/grok-composer-2.5-fast`, `xai/grok-build`, and `xai/grok-build-0.1`.

Relevant environment variables:

- `AAWM_GROK_OIDC_REFRESH_ENABLED`: enables the scheduled task.
- `AAWM_GROK_OIDC_AUTH_FILE`: Grok CLI auth JSON path. When unset, the
  sidecar falls back through `LITELLM_XAI_GROK_AUTH_FILE`,
  `LITELLM_XAI_OAUTH_GROK_AUTH_FILE`, `GROK_AUTH_FILE`, `GROK_HOME/auth.json`,
  and finally `/home/zepfu/.grok/auth.json`.
- `AAWM_GROK_OIDC_LOCK_FILE`: file lock path used while writing the auth JSON.
- `AAWM_GROK_OIDC_AUTH_FILE_UID`: optional uid applied to the atomic auth-file
  replacement. Use this when the sidecar runs as a different container user
  than the host Grok CLI owner.
- `AAWM_GROK_OIDC_AUTH_FILE_GID`: optional gid applied to the atomic auth-file
  replacement.
- `AAWM_GROK_OIDC_AUTH_FILE_MODE`: optional private file mode applied to the
  atomic auth-file replacement. Group/other-readable or writable values are
  rejected and fall back to `0600`. Ownership/mode snapshot, resolve, and apply
  for this refresh path use the shared helpers in
  `litellm/secret_managers/credential_file_metadata.py` (same clamp/ownership
  safety as the Codex/xAI refresh scripts).
- `AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS`: minimum seconds between attempts.
- `AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS`: near-expiry window for non-forced
  refreshes.
- `AAWM_GROK_OIDC_FORCE_REFRESH`: when true, refreshes on every scheduled
  attempt even if the current token still appears valid.
- `AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS`: token endpoint timeout.

Credential lock acquisition is nonblocking and fail closed. A held lock,
unavailable `fcntl`, or failed lock open/acquisition aborts that refresh without
an unlocked write or an indefinite wait.

Each due attempt emits a separate `grok_oidc_refresh` JSON line with sanitized
status fields such as `attempted`, `refreshed`, `skipped`, `auth_file`,
`scope`, `expires_at`, `error_class`, and `error_message`. The event must not
contain access tokens, refresh tokens, id tokens, client secrets, raw auth
headers, or the full credential payload.

When Grok OIDC refresh is enabled, the sidecar also performs a metadata-only
repair on every provider-status cycle before billing polls or token refreshes.
It applies the configured auth-file uid/gid/mode without reading or rewriting
token values and emits `grok_oidc_metadata_repair` only when it repairs the file
or encounters an error. This bounds damage from another process recreating the
shared auth file with container-owned metadata between hourly refreshes.

## WSL-local dual credential writer (XAI-003 / XAI-004)

On the operator WSL host, native Grok OIDC and managed `oa_xai/*` OAuth refresh
are owned by one dedicated dual-writer unit, not by the multi-provider
`provider-status-observations` sidecar and not by either LiteLLM proxy.

- Compose: `docker-compose.wsl-grok-oidc.yml`
- Launcher: `scripts/ensure-wsl-grok-oidc-sidecar.sh` (`--status` default,
  `--apply`, `--stop`)
- Static smoke: `tests/wsl-grok-oidc-sidecar-smoke.sh`

### Ownership and consumers

- This WSL-local sidecar is the **only automatic writer** of both:
  - native Grok OIDC: `/home/zepfu/.grok/auth.json`
  - managed xAI OAuth (`oa_xai/*`): `/home/zepfu/.litellm/xai/oauth-auth.json`
- The two families stay independent: separate files, separate lock files,
  separate sanitized event names (`grok_oidc_*` vs `xai_oauth_*`), and separate
  metadata env vars. Native Grok OIDC must never populate managed xAI OAuth and
  the reverse must not happen either.
- `aawm-litellm` and `litellm-dev` remain **read-only consumers** of both files.
  They require **no restart** when either credential is refreshed or replaced
  under its own lock.
- Manual Grok CLI / Grok Build login is **break-glass only** for the native
  family. Use it when the native refresh token is revoked, the native credential
  file is missing/corrupt, or the automatic writer cannot recover. After a
  manual login, the sidecar resumes non-forced native refresh from the new file
  without proxy restarts.
- Managed `oa_xai/*` recovery remains on the managed OAuth path
  (`scripts/xai_oauth_refresh.py` / operator re-auth for that family). Do not
  copy native tokens into the managed file.

### Cadence and safety

The dedicated unit reuses image `aawm-provider-status-observations:prod` but
overrides the entrypoint to a dual-family loop around
`scripts/grok_oidc_refresh.py` and `scripts/xai_oauth_refresh.py`. Front-door
probes, DB apply/schema, anomaly scan, billing, Codex/Kimi/Alibaba writers, and
auth health poll stay disabled. Only the two credential directories are mounted
writable; Codex, Kimi, and Alibaba credential dirs are not mounted.

A single cycle can run both refresh tasks. Each family keeps independent
config variables even when both use the same defaults:

Rendered native defaults:

- `AAWM_GROK_OIDC_REFRESH_ENABLED=1`
- `AAWM_GROK_OIDC_AUTH_FILE=/home/zepfu/.grok/auth.json`
- `AAWM_GROK_OIDC_LOCK_FILE=/home/zepfu/.grok/auth.json.lock`
- `AAWM_GROK_OIDC_AUTH_FILE_UID=1000` / `GID=1000` / `MODE=0o600`
- `AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=300`
- `AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS=900`
- `AAWM_GROK_OIDC_FORCE_REFRESH=0`
- `AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS=30`

Rendered managed defaults:

- `AAWM_XAI_OAUTH_REFRESH_ENABLED=1`
- `AAWM_XAI_OAUTH_AUTH_FILE=/home/zepfu/.litellm/xai/oauth-auth.json`
- `AAWM_XAI_OAUTH_LOCK_FILE=/home/zepfu/.litellm/xai/oauth-auth.json.lock`
- `AAWM_XAI_OAUTH_AUTH_FILE_UID=0` / `GID=0` / `MODE=0o600`
- `AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS=300`
- `AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS=900`
- `AAWM_XAI_OAUTH_FORCE_REFRESH=0`
- `AAWM_XAI_OAUTH_HTTP_TIMEOUT_SECONDS=30`

Shared unit safety defaults include `AAWM_PROVIDER_STATUS_APPLY=0` (no DB writes
required). Sanitized events include independent
`grok_oidc_metadata_repair` / `grok_oidc_refresh` and
`xai_oauth_metadata_repair` / `xai_oauth_refresh` lines; token material is never
logged.

### Launcher, health, and proxy isolation

The launcher preflights the image and **both** credentials, captures exact
container ID / start timestamp / restart count for both LiteLLM proxies, starts
or stops only `wsl-grok-oidc-refresh` with
`docker compose -f docker-compose.wsl-grok-oidc.yml ... --no-deps --no-build`,
and fails if either proxy snapshot changes. Never run a broad compose
`up`/`down` for this unit, and never mention a proxy service in a mutating
compose command for this file. Status mode works before activation.

Native preflight validates host-readable mode `0600` and uid/gid `1000/1000`
plus issuer/client/access/refresh/expiry metadata. Managed preflight validates
mode `0600` and root-family ownership (`0/0`, accepting legacy `65534` until the
sidecar rewrites ownership). Because the host user often cannot read the
root/nobody-owned managed file, the launcher validates managed metadata from
`stat` and, when the file is unreadable, uses a disposable **read-only**
`docker run` of the existing prod image with a read-only mount to validate JSON
safely. No secrets are printed.

Combined credential/process health requires **both** credential records to have:

- a current access credential (`key` or `access_token`)
- a refresh token
- the expected client ID / scope
- mode `0600`
- remaining lifetime well above LiteLLM's 300-second near-expiry rejection
  boundary (`remaining > 600` seconds)

Native health also requires issuer `https://auth.x.ai`. Managed health may omit
issuer. The launcher waits for that combined healthcheck before reporting
`apply_ok`. Healthcheck and refresh logs emit no access tokens, refresh tokens,
id tokens, or raw credential payloads.

## Grok Billing Poll Task

The same sidecar can also run an explicit hourly Grok billing poll. This is
telemetry-only and separate from the five-minute provider front-door probes and
the Grok OIDC refresh task. The poll reads the current OIDC credential from
`AAWM_GROK_OIDC_AUTH_FILE`, derives the Grok account identity headers from the
scoped credential record, and calls
`https://cli-chat-proxy.grok.com/v1/billing?format=credits` with Grok CLI-style
headers. The request includes the OIDC bearer token plus `x-userid`,
`x-grok-user-id`, `x-teamid`, and `x-email` derived from the credential
`user_id`, `team_id`, and `email` fields. The poll persists the returned billing
snapshot as a sanitized `rate_limit_observations` row using the same stored field
shape and dedupe guard as the LiteLLM callback path.

### Native Grok client-version consumers

The billing poll resolves its Grok client version for every outbound request
attempt. An explicitly supplied `--grok-billing-client-version` is an
operator-only override. Otherwise, request-time precedence is:

1. `AAWM_GROK_BILLING_CLIENT_VERSION`
2. `LITELLM_XAI_GROK_CLIENT_VERSION`
3. `GROK_CLIENT_VERSION`
4. the shared native Grok client-version cache

There is no fixed-version fallback. A present override is authoritative, so an
empty or invalid value fails that request instead of falling through to a
lower-precedence source. The native Grok OIDC request path uses
`LITELLM_XAI_GROK_CLIENT_VERSION` as its explicit emergency override, then the
legacy `GROK_CLIENT_VERSION`, then the same shared cache. Normal operation
leaves the overrides unset.

The cache is part of the native Grok OIDC contract used by Grok requests and
billing. It is separate from managed xAI OAuth for `oa_xai/*`; managed OAuth
does not read the native version cache, and neither credential family may be
used as a substitute for the other.

For billing, the resolved version creates the actual outbound `user-agent` and
`x-grok-client-version` headers. Request events and persisted evidence are
derived from that same header object, rather than reconstructing an expected
header set after the request. Version provenance is limited to sanitized fields
such as `client_version_source`, `client_version_cache_source`, and
`client_version_cache_path_class`; it does not expose the configured host path
or cache contents.

### Native Grok client-version cache maintenance

Consumers read `AAWM_GROK_CLIENT_VERSION_CACHE_PATH`, which defaults to
`/run/aawm/grok/native-client-version.json`. The maximum accepted record age is
configured by `AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS` and defaults to
`172800` seconds. Missing, malformed, invalid, future-dated, or stale cache
records fail closed. The cache is re-read for every request attempt, so its
writer must publish a complete file by atomic replacement; the next request
observes the replacement without a LiteLLM or sidecar restart.

Local development and Thoth mount the host cache directory selected by
`AAWM_GROK_CLIENT_VERSION_CACHE_DIR` (default
`/home/zepfu/.cache/aawm/grok`) read-only at `/run/aawm/grok`. Mount the parent
directory, not the JSON file, so atomic inode replacement remains visible to
the containers. Cache discovery or refresh belongs in host maintenance outside
request serving. Do not add a host command, CLI invocation, package lookup, or
subprocess to the request path.

Grok billing has two distinct payload shapes and must not be conflated:

- `GET /v1/billing?format=credits` (credit-format): weekly Grok Build credit
  usage. When `config.currentPeriod.type` is `USAGE_PERIOD_TYPE_WEEKLY`, the
  snapshot uses quota key `xai_grok_build_weekly_credits:credits`,
  `quota_period=weekly`, and billing boundaries from `billingPeriodStart` /
  `billingPeriodEnd` or `currentPeriod.start` / `currentPeriod.end`.
  `creditUsagePercent` is **used** percent; `remaining_pct` stores
  `100 - creditUsagePercent`. When a fresh weekly period omits
  `creditUsagePercent` but includes weekly period bounds, the sidecar persists
  `0%` used / `100%` remaining with explicit `grok_billing_weekly_fresh_period`
  evidence (matching the TUI `Weekly limit: 0%` display).

- `GET /v1/billing` (no format): monthly request counters via `monthlyLimit.val`
  and `used.val`. These persist as `xai_grok_build_monthly_requests:requests`
  with `quota_period=monthly` and are **not** used as a fallback for weekly
  credit percent.

## Kimi Code Native Usage Observations

The optional native Kimi Code usage poll records
`source=kimi_code_usage`, `provider=kimi_code`, `client=kimi-code`,
`model=kimi-code`, and `quota_type=quota_units`. Its versioned parser records
the parser/source versions and parser path in safe evidence; a parsed reset is
stored in `expected_reset_at`, while an absent or malformed reset remains
unknown rather than guessed. Account identity is represented only by the
derived `account_hash`, never by the raw account fields.

Native quota windows use
`kimi_code_5h:quota_units`, `kimi_code_7d:quota_units`, and, when supplied,
`kimi_code_monthly:quota_units`; a supplied parallel limit is a separate
concurrency observation. The units are provider-native quota units, not token,
USD, output-length, or model-capability claims. Missing optional monthly,
parallel, reset, or usage values do not prove unlimited quota, zero usage, a
disabled model, or a billing outcome.

Malformed or missing Kimi usage telemetry is a degraded observation, not a
successful zero-quota snapshot. Account-wide quota/capacity evidence applies a
shared managed-account cooldown. Authenticated `/models` capability or effort
rejection is candidate-scoped, so it does not cool unrelated managed models.
An alias declaration does not enable Kimi Code: a configured route plus a valid
credential and accepted capability remain required.

The provider-status sidecar enables this poll with the existing Kimi Code
credential already mounted for OAuth maintenance. It sends an authenticated
`GET https://api.kimi.com/coding/v1/usages`; it does not create, copy, or
reauthorize a credential. The scheduled quota cadence is hourly and is
independent of the five-minute provider-status and Kimi OAuth-maintenance loop.

Relevant Kimi usage environment variables:

- `AAWM_KIMI_USAGE_POLL_ENABLED`: enables the scheduled native usage poll.
- `AAWM_KIMI_USAGE_POLL_INTERVAL_SECONDS`: minimum seconds between usage poll
  attempts; the managed sidecar default is `3600`.
- `AAWM_KIMI_USAGE_POLL_HTTP_TIMEOUT_SECONDS`: native usage endpoint timeout;
  the managed sidecar default is `30`.

Each due attempt emits one sanitized `kimi_usage_poll` JSON event. Successful
runtime verification requires `status_code=200`, `persisted=true`, and a
positive parsed `observation_count` without logging an access token or raw
credential payload.

This is an interpretation-only documentation update. It adds no
`session_history` or observation schema, view, or API change, so no
`dashboard-shell` handoff is needed.

## Z.AI Coding Plan quota polling

The optional Z.AI Coding Plan quota poll records account-scoped 5-hour and
weekly remaining windows from `GET https://api.z.ai/api/monitor/usage/quota/limit`
with `Authorization: Bearer <key>`. It does not scrape the z.ai SPA, does not
call bare `/api/monitor/usage`, and does not send inference traffic as a health
check. An optional `GET https://api.z.ai/api/biz/subscription/list` supplies a
hashed `customerId`; subscription failure must not blank a valid quota payload.

Persisted rows use `source=zai_coding_plan_quota_poll`,
`provider=zai_coding_plan`, `client=zai-coding-plan`, and `model=zai-coding-plan`.
The mapper accepts both live `CREDIT_LIMIT` absolute remaining windows and
OpenQuota `TOKENS_LIMIT` percent / `TIME_LIMIT` count windows. Account identity
is hashed; raw keys and customer ids are never persisted. Routing is not blocked
when the poll is off or unknown.

Relevant environment variables:

- `AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED`: enables the scheduled poll.
  Defaults to disabled (`0`).
- `AAWM_ZAI_CODING_PLAN_QUOTA_POLL_INTERVAL_SECONDS`: minimum seconds between
  attempts; default `3600`.
- `AAWM_ZAI_CODING_PLAN_QUOTA_POLL_HTTP_TIMEOUT_SECONDS`: quota HTTP timeout;
  default `30`.
- Credential order is `ZAI_KEY` then `ZAI_CODING_PLAN_API_KEY`. Ordinary
  `ZAI_API_KEY` / `ZHIPU_API_KEY` are not used.

## Cursor Agent monthly usage observations

The optional Cursor Agent usage poll records account-scoped monthly
included spend from Dashboard Connect
`POST /aiserver.v1.DashboardService/GetCurrentPeriodUsage` on
`https://api2.cursor.sh`. This is not Cloud Agents `GET /v0/me`, not the
`agentn` turn host, and not `public.provider_status_observations`.

Persisted rows use `source=cursor_agent_usage`, `provider=cursor_agent`,
`client=cursor-agent`, `model=cursor-agent`, `quota_key=cursor_agent_monthly:cents`,
`quota_period=monthly`, and `quota_type=cents`. Mapping from the 2026-08-12
camelCase dashboard dump (spend in USD cents):

- `quota_used` ← `planUsage.includedSpend`
- `quota_limit` ← `planUsage.limit`
- `quota_remaining` ← `planUsage.remaining`
- `quota_period` ← `monthly`

`totalPercentUsed` / `autoPercentUsed` / `apiPercentUsed` are **not**
`totalSpend / limit`. The trustworthy included fraction is
`includedSpend / limit`. Account identity is hashed; raw tokens and
account ids are never persisted. Failed refreshes emit a sanitized
`cursor_agent_usage_poll` event with `last_good_state_retained=true` and
do not write a replacement row, so the last valid observation stays.

Weekly Cursor Grok Bot used/limit/reset remains truthful unknown. There
is no weekly `quota_key`. Do not treat xAI Grok Build weekly credits or
BugBot license RPCs as Grok Bot. Reevaluate only when
`AAWM_CURSOR_AGENT_GROK_BOT_USAGE_SOURCE` names a verified Dashboard or
Connect RPC; that env is a checkpoint, not a quota source.

Relevant environment variables:

- `AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED`: enables the scheduled poll.
  Defaults to disabled so the sidecar does not send live dashboard traffic.
- `AAWM_CURSOR_AGENT_USAGE_POLL_INTERVAL_SECONDS`: minimum seconds between
  attempts; default `3600`.
- `AAWM_CURSOR_AGENT_USAGE_POLL_HTTP_TIMEOUT_SECONDS`: dashboard RPC
  timeout; default `30`.
- `AAWM_CURSOR_AGENT_USAGE_DASHBOARD_URL`: Dashboard host override;
  default `https://api2.cursor.sh`.
- `AAWM_CURSOR_AGENT_GROK_BOT_USAGE_SOURCE`: optional reevaluation
  checkpoint for a future weekly Grok Bot RPC. Empty keeps Grok Bot
  unknown.

Auth matches CURSOR-004: `CURSOR_AUTH_TOKEN` preferred, then
`CURSOR_API_KEY` as Bearer. `CURSOR_CLI_KEY` is ignored.

The provider-status sidecar image ships only the stdlib Cursor usage
helpers (`constants.py`, `dashboard.py`, and `usage.py`). It does not
package the full `cursor_agent` provider, `common_utils.py`, or `httpx`.
Dashboard polling uses `urllib` with those stdlib helpers.

## Alibaba Token Plan quota polling

The provider-status sidecar can poll the authenticated ModelStudio Token Plan
console contract without sending inference traffic. It records the provider's
5-hour and 7-day Credit usage windows as:

- `alibaba_token_plan_5h:credits`
- `alibaba_token_plan_7d:credits`

Alibaba reports each window as a consumed fraction. The sidecar converts that
fraction to `remaining_pct`, preserves the provider reset timestamp, and leaves
absolute limit/used/remaining columns null because this console response does
not provide authoritative absolute Credit limits. It does not invent daily or
monthly quota windows. Alibaba may omit a reset timestamp for a window whose
consumed fraction is exactly zero. That unused window is persisted with
`remaining_pct=100`, `expected_reset_at=NULL`, and explicit
`reset_at_state=absent_unused_window` evidence. A missing reset for a consumed
window remains malformed telemetry and is not persisted.

The same scheduled poll separately reads the manual reset-card inventory from
`zeldaHttp.apikeyMgr./tokenplan/personal/api/v2/reset-card/list`. These cards
are operator-consumable weekly quota resets, not the automatic rolling 5-hour
or 7-day quota-window reset timestamps above. Observation is read-only: it
lists cards but never consumes or applies a reset.

Manual reset-card rows reuse `public.provider_credit_observations` and
`public.provider_credit_current` with
`provider=alibaba_token_plan`,
`credit_family=alibaba_token_plan_manual_quota_reset`, and
`source=alibaba_token_plan_reset_card_list`. Each card has a stable SHA-256
`credit_identity` derived from the hashed account scope and `cardNo`; raw
`cardNo` is never stored. Sanitized fields retain `cardType`, `effectiveAt`,
and `expiresAt`, while normalized columns use `granted_at=effectiveAt`,
`expires_at`, per-row `available_count` (`1` or `0`), and lifecycle `status`.
Visible unexpired cards are `available`; visible expired cards are `expired`.
A previously visible `available` card that disappears before expiry becomes
`used`, while one that disappears at or after expiry becomes `expired`.
Unchanged inventories dedupe through the shared credit writer. An empty list is
valid and transitions any previously available cards without fabricating a
card row. The poll event exposes the aggregate as
`reset_card_available_count`.

Authentication uses international RAM access-key credentials, not a console
ticket file, cookie jar, or `sec_token`. On each due poll the sidecar reads
`ALIBABA_RAM_KEY` and `ALIBABA_RAM_SECRET` (optional
`ALIBABA_RAM_PRINCIPAL` contributes only to a hashed fingerprint). Those
credentials sign one ACS3-HMAC-SHA256 `GenerateCLIAccessToken` POST to
`modelstudio.ap-southeast-1.aliyuncs.com` `/modelstudio/cli/generateAccessToken`
(API version `2026-02-10`) with an empty body. The returned `cliAccessToken`
is cached in process memory and sent as `Authorization: Bearer` to the
Singapore ModelStudio CLI gateway
`https://bailian-singapore-cs.alibabacloud.com/cli/api.json`. Gateway form
bodies carry only `params` plus `region=ap-southeast-1`. China console hosts
are rejected.

The cached Bearer is reused across subscription, usage, and reset-card list
calls and across poll cycles. A known application-level authentication
envelope (`NotLogined`, `NoPermission`, `Team.NotAuthorised`,
`AuthorityPolicies.NoPermission`, or the narrow login/session allowlist) or
an HTTP 401/403 triggers at most one remint per endpoint fetch, followed by a
single replay. A still-failing replay fails closed as `auth` with no second
mint. Mint HTTP 401/403 or a `NoPermission` mint envelope is also `auth`.
Failed polls keep last-good subscription and reset-card current state.

Each `alibaba_quota_poll` event reports value-free mint telemetry:
`auth_source`, `token_cached`, `mint_attempted`, `mint_succeeded`,
`refresh_attempted`, `refresh_succeeded`, and `credential_reset`. No RAM
secret, principal, Bearer token, account identifier, cookie, or response body
is ever logged or persisted.

The sidecar does not copy the Token Plan inference API key, run browser
flows, launch `bl` or other secondary CLIs, ask for passwords, or automate
MFA. Credential rotation is a process environment change; a new RAM
fingerprint discards the cached Bearer and remints on the next due poll.

The usage and manual reset-card list requests run at startup and then on the
configured usage cadence. Subscription metadata is refreshed at startup and
independently on the configured subscription cadence. The subscription
response supplies the active-plan status, plan specification, period
boundaries, and a plan-instance identifier used only to derive `account_hash`;
the raw identifier is not persisted.

Relevant environment variables:

- `ALIBABA_RAM_KEY`: RAM access-key ID used to mint the console Bearer.
- `ALIBABA_RAM_SECRET`: RAM access-key secret used for ACS3 mint signing.
  Never log or persist this value.
- `ALIBABA_RAM_PRINCIPAL`: optional RAM principal. Used only to hash the
  in-memory credential fingerprint; never logged or persisted.
- `AAWM_ALIBABA_QUOTA_POLL_ENABLED`: enables the scheduled poll.
- `AAWM_ALIBABA_QUOTA_POLL_INTERVAL_SECONDS`: minimum seconds between usage
  polls; the managed sidecar default is `300`.
- `AAWM_ALIBABA_SUBSCRIPTION_POLL_INTERVAL_SECONDS`: minimum seconds between
  subscription refreshes; the managed sidecar default is `21600`.
- `AAWM_ALIBABA_QUOTA_POLL_HTTP_TIMEOUT_SECONDS`: request timeout; the managed
  sidecar default is `30`.
- `AAWM_ALIBABA_QUOTA_GATEWAY_URL`: override for the ModelStudio CLI gateway.
  Defaults to `https://bailian-singapore-cs.alibabacloud.com/cli/api.json`.
  Keep the default unless contract verification proves a provider change.
  China hosts are not valid overrides.
- `AAWM_ALIBABA_QUOTA_POLL_MAX_ATTEMPTS`: bounded attempts per endpoint call;
  the managed sidecar default is `2`.
- `AAWM_ALIBABA_QUOTA_POLL_RETRY_BACKOFF_SECONDS`: base exponential backoff for
  transient failures; the managed sidecar default is `0.5`.

Each due attempt emits one sanitized `alibaba_quota_poll` JSON event. Reset-card
telemetry includes status/attempt counts, visible and available card counts,
credit observation/insert counts, and whether the reset-card state was
persisted. Runtime success requires HTTP 200 responses, an active subscription,
valid recognized usage windows, a valid reset-card array (including an empty
array), successful configured persistence, and no credential, raw response,
raw card number, account identity, RAM secret, Bearer token, or traceback in
container logs. Authentication, transport, HTTP, envelope, or field-validation
failures are degraded and leave the last-known reset-card current state
unchanged. The reset-card poll is list-only and never calls `/reset-card/use`.

Anthropic unified response headers persist separate weekly buckets:

- `anthropic_unified_7d:7d` — baseline unified seven-day quota (`quota_period=seven_day`, `window_minutes=10080`).
- `anthropic_unified_7d_oi:7d_oi` — Fable / overage-included weekly bucket from
  `anthropic-ratelimit-unified-7d_oi-*` (and `llm_provider-anthropic-ratelimit-unified-7d_oi-*`)
  headers. Representative claim and overage status fields are stored in
  `raw_provider_fields` (for example `seven_day_overage_included`). Interval
  materialization maps this key to `quota_type = weekly_overage_included`, not
  `weekly_special`.
- `anthropic_unified_7d_sonnet:7d_sonnet` — retired Sonnet-specific weekly-special
  series kept for historical rows; new Fable traffic must not be labeled as Sonnet.

The `public.rate_limit_intervals` materialized view has historical migrations
(`scripts/apply_rate_limit_intervals_mview_2026_06_03_antigravity.sql` and the
older `scripts/apply_rate_limit_intervals_mview_2026_05_23.sql` without
Antigravity pool rows). Those migration names and historical rows are retained
as database history; they do not define a current Antigravity route. The view
explicitly allows quota key
`xai_grok_build_weekly_credits:credits` and maps it to `quota_type = 'weekly'`
in the final projection CASE. That mapping is deliberate: weekly Grok Build
credit intervals must roll up under the same `weekly` bucket as other weekly
quota keys, not under raw observation `quota_type` alone. Unlike most quota
keys, weekly Grok Build credits also materialize observations with
`remaining_pct = 100` (fresh weekly period / `grok_billing_weekly_fresh_period`)
so a new interval closes out the prior depleted window instead of leaving an
old 0% row open past reset. Monthly Grok request
rows (`xai_grok_build_monthly_requests:requests`) continue to enter via
`quota_type = 'requests'` when `remaining_pct < 100`.


### Concurrent Refresh and Unique Index Contract (D1-491)

Both rebuild scripts create a plain seven-column unique index
`rate_limit_intervals_unique_idx` over `(provider, model, quota_key,
quota_type, fromdate, expected_reset_at, remaining_pct)` with
`NULLS NOT DISTINCT`. No expression wrappers (such as `COALESCE(model, '')`)
appear in the index definition; the `COALESCE` is retained only inside the
window `PARTITION BY` clauses for correct interval-grouping semantics.
This column-only shape satisfies the PostgreSQL requirement for
`REFRESH MATERIALIZED VIEW CONCURRENTLY` eligibility.

For databases still carrying the older expression-based unique index, a
one-time repair script is provided:
`scripts/apply_rate_limit_intervals_concurrent_index_2026_07_25.sql`.
It acquires the same advisory lock used by dashboard-shell maintenance
(`hashtext('dashboard-shell'), hashtext('materialized-view-maintenance')`),
runs fail-closed preflight guards for raw column-key duplicates and
null-vs-empty model collisions, creates a temporary eligible unique index,
performs `REFRESH MATERIALIZED VIEW CONCURRENTLY` while both indexes coexist,
then atomically drops the old index and renames the new one to the canonical
name. The script does not drop or recreate the materialized view and does not
mutate pg_cron schedules.

When a non-weekly credit payload includes only billing-period boundaries with
no usage fields, the sidecar may still persist a period-only monthly credits
snapshot with null `remaining_pct`.

Relevant environment variables:

- `AAWM_GROK_BILLING_POLL_ENABLED`: enables the scheduled billing poll.
- `AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS`: minimum seconds between billing
  poll attempts.
- `AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS`: billing endpoint timeout.
- `AAWM_GROK_BILLING_URL`: billing endpoint URL.
- `AAWM_GROK_BILLING_CLIENT_VERSION`: explicit billing-only client-version
  override. It has no default and should remain unset outside emergency or
  diagnostic use.
- `LITELLM_XAI_GROK_CLIENT_VERSION`: explicit native Grok emergency override
  and second-precedence billing override. It has no default.
- `GROK_CLIENT_VERSION`: legacy native Grok and billing override. It has no
  default.
- `AAWM_GROK_CLIENT_VERSION_CACHE_DIR`: host directory mounted read-only at
  `/run/aawm/grok` for local development and Thoth. Defaults to
  `/home/zepfu/.cache/aawm/grok`.
- `AAWM_GROK_CLIENT_VERSION_CACHE_PATH`: in-container cache path. Defaults to
  `/run/aawm/grok/native-client-version.json`.
- `AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS`: maximum cache record age.
  Defaults to `172800`.
- `AAWM_GROK_BILLING_CLIENT_IDENTIFIER`: Grok CLI client identifier header.
- `AAWM_GROK_BILLING_XAI_TOKEN_AUTH`: `x-xai-token-auth` header value.
- `AAWM_GROK_BILLING_MODEL`: model label stored with the billing snapshot.
- `AAWM_GROK_BILLING_HTTP_METHOD`: HTTP method used for billing poll requests.
  Defaults to `GET`.
- `AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE`: when true, include
  `x-grok-model-override` using `AAWM_GROK_BILLING_MODEL` on billing poll
  requests. Defaults to true so the sidecar matches successful native Grok
  passthrough/manual billing calls. Set to false only when an operator
  explicitly wants the older minimal header shape. The sidecar still sends
  `content-type: application/json`; disabling model override only omits
  `x-grok-model-override`. The model label is still persisted on
  `rate_limit_observations.model` regardless of this setting.
- `AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS`: maximum billing poll attempts per
  scheduled run, including retries.
- `AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS`: base backoff seconds between
  retryable billing poll failures.

The billing poll retries only transient transport/capacity failures and the
known Grok `400` timeout/cancel response (`The operation was cancelled` /
`Timeout expired`). Auth failures and provider rate limits (`401`, `403`, and
`429`) are not retried; they are surfaced as a single degraded telemetry event
for that scheduled run.

Each due attempt emits a separate `grok_billing_poll` JSON line with sanitized
status fields such as `attempted`, `persisted`, `skipped`, `auth_file`,
`resolved_auth_file`, `auth_file_source`, `billing_url`, `client_version`,
`user_agent`, `client_version_source`, `client_version_cache_source`,
`client_version_cache_path_class`, `model`, `status_code`, `attempt_count`,
`retry_count`, `poll_max_attempts`, `observation_count`, `inserted_count`,
`error_class`, and `error_message`. For D1-304 debugging, the event also
includes compact request/transport diagnostics such as `http_client`,
`request_method`, `billing_host`, `billing_path`, `billing_query_keys`,
`billing_query_present`, `header_names`, `include_model_override`,
`model_override_configured`, `client_identifier`,
`x_xai_token_auth_configured`, and `request_contract_fingerprint`.

The fingerprint is derived from the non-secret request contract only: HTTP
method, billing host/path, query key names, the actual client version and
user-agent, client identifier, whether `x-xai-token-auth` is configured,
model-override flags, and header names. It must not include authorization
tokens, account identity values, raw auth payloads, resolved IP addresses, the
configured cache path, cache contents, or the configured `x-xai-token-auth`
value. The event must not emit dedicated identity fields or raw auth headers.
It must not contain access tokens, refresh tokens, id tokens, client secrets,
account identity values (`user_id`, `team_id`, `email`, or the derived
`x-userid`, `x-grok-user-id`, `x-teamid`, and `x-email` headers), or the full
billing credential payload. Billing poll failures are logged and do not raise
out of the sidecar loop.

Successful sidecar billing polls copy the same safe request-contract evidence
into `rate_limit_observations.evidence` with
`request_contract_source=grok_billing_sidecar_poll`. This lets later DB-only
investigations distinguish snapshots inserted by the scheduled sidecar from
snapshots extracted from Grok passthrough/manual traffic without storing auth
tokens or account identity values. The evidence records the client version,
user-agent, and sanitized version source from the headers used by the successful
request.

Successful native Grok billing passthrough calls also record comparable
request-contract metadata in Langfuse/session-history metadata and copy the
fingerprint into `rate_limit_observations.evidence` when a Grok billing payload
is extracted. Those fields are prefixed with
`grok_billing_passthrough_` and include the HTTP client, method, target
host/path, query key names, outbound header names, user-agent, whether
`x-xai-token-auth` was configured, and a non-secret request-contract
fingerprint. They intentionally omit authorization tokens, account identity
values, raw auth payloads, resolved IP addresses, and the configured
`x-xai-token-auth` value. Compare that passthrough fingerprint with the
sidecar `request_contract_fingerprint` when investigating sidecar billing poll
parity.

## Codex Reset-Credit Poll Task

The same sidecar can run an explicit hourly Codex reset-credit and quota poll.
This is telemetry-only and separate from the five-minute provider front-door
probes, Codex OAuth refresh, and Grok billing poll. The poll independently reads
every enabled `LITELLM_CODEX_OAUTH_INVENTORY` record, calls the native ChatGPT
reset-credit **detail** endpoint (default
`https://chatgpt.com/backend-api/wham/rate-limit-reset-credits`) with that
record's exact `Authorization: Bearer <token>` and `ChatGPT-Account-Id`
headers, and continues to the next record after an account-specific failure.
Reset-credit rows use `public.provider_credit_observations`; native five-hour
and seven-day quota windows are written synchronously to
`public.rate_limit_observations`. The quota writer uses
`AAWM_CODEX_QUOTA_DSN` when configured and otherwise falls back to the general
provider-status DSN. This lets deployments store Codex quota beside the
LiteLLM callback/session-history tables that hydrate routing while leaving
other provider-status and reset-credit writes on the sidecar's general
database. The provider-status script reuses the existing observation schema.

`AAWM_CODEX_USAGE_URL` remains the backward-compatible env name for the poll URL.
If it is still set to the legacy aggregate URL (`/backend-api/wham/usage`), the
sidecar maps it to the detail endpoint automatically. The aggregate `/wham/usage`
response and its rate-limit window reset fields are **not** credit expiry; only
per-credit `expires_at` from the detail `credits[]` entries (or explicit
credit-level fields) define banked credit expiry.

Relevant environment variables:

- `AAWM_CODEX_RESET_CREDIT_POLL_ENABLED`: enables the scheduled poll.
- `AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS`: minimum seconds between poll
  attempts (default `3600`).
- `AAWM_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS`: HTTP timeout.
- `AAWM_CODEX_USAGE_URL`: poll URL (defaults to rate-limit-reset-credits).
- `AAWM_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS`: max attempts per scheduled run.
- `AAWM_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS`: retry backoff base.
- `AAWM_CODEX_QUOTA_DSN`: optional Postgres DSN used only for direct Codex
  quota persistence; falls back to the general sidecar DSN when unset.

The detail parser reads `credits[]` with `status`, `reset_type`, `granted_at`,
`expires_at`, `redeem_started_at`, and `redeemed_at` when present. Each visible
credit becomes one observation row with stable `credit_identity` (provider `id`
when present, otherwise a derived hash from account, family, grant/expiry, and
reset type). Normalized columns include `granted_at`, `expires_at`, `status`,
`redeem_started_at`, `redeemed_at`, `operator_annotation`, and `source_url`.
`available_count` is `1` per available credit row (aggregate available count in
poll events is the number of visible available credits). Seed/backfill metadata
in the recorder applies operator annotations and source URLs for known historical
credits and emits absent historical credits as `used` or `expired` rows without
storing secrets or raw account ids.

Lifecycle: credits still returned by the detail endpoint keep provider-derived
status (typically `available`) until `expires_at`; a visible or stored
`available` credit past `expires_at` without provider redemption timestamps is
inferred as `expired`. A previously stored `available` credit that no longer
appears before `expires_at` is inferred as `used`. Provider `redeemed_at` /
`redeem_started_at` are preferred when present.

Account identity is hashed (stable short SHA-256 prefix) before storage. Rows
omit `client` and `client_version`. Inserts dedupe on the latest row per
`credit_identity` when status, counts, timestamps, and annotations are unchanged.

Each due attempt emits `codex_reset_credit_poll` with sanitized fields such as
`account_label`, `account_hash`, `attempted`, `persisted`, `status_code`,
`attempt_count`, `retry_count`, `available_count`, `inserted_count`,
`quota_accepted_count`, `quota_storage_status`, `quota_window_states`,
`quota_period_states`, `quota_health`, `poll_url`, `error_class`, and
`error_message`. `inserted_count` covers the synchronous reset-credit writer;
quota rows report `quota_storage_status=persisted` and
`quota_inserted_count` after the synchronous direct insert. The paired quota
windows are healthy only when both five-hour and seven-day observations are
fresh. Stale or unknown windows retain their reset/freshness evidence but store
no fabricated remaining or used percentage. Upstream scope and model are
stored only when the response actually supplies them; a model-like limit name
is not promoted to model specificity.

`codex_quota_poll_aggregate` is `healthy` when every enabled account has both
fresh windows, `degraded` when at least one account has usable fresh quota
state, and `terminal` when none do. Events must not contain access tokens,
refresh tokens, raw auth headers, account IDs, emails, or credential paths.

The detail endpoint is undocumented and provider-owned; shape may change without notice.

## One-Shot Exit Policy

With `--once`, enabled `grok_oidc_refresh`, per-account
`codex_oauth_refresh`, and `xai_oauth_refresh` events are required tasks.
A successful refresh or successful no-op/skipped refresh satisfies the task.
Any required failure returns a non-zero process status. Telemetry, metadata
repair, passive health, Kimi work, and aggregate events are optional; their
failures are reported as optional degradation without changing the required
exit status. Native Grok OIDC and managed xAI OAuth remain separate credential
families and tasks.


## Observability Anomaly Scan Task

The same sidecar can also run a scheduled session-history and rate-limit
telemetry anomaly scan. This is separate from the five-minute provider
front-door probes and the hourly credential or billing tasks. When enabled, the
scan reads recent `public.session_history` and `public.rate_limit_observations`
rows and looks for persistence or mapping inconsistencies such as missing
provider/model fields, alias metadata that was not promoted, token or tool
activity counters that do not match persisted activity, parsed git commit/push
activity counters that were not rolled up to `session_history`, or stale
rate-limit reset timestamps that still have matching recent traffic. Git
commit/push anomaly triggers use the parsed `session_history_tool_activity`
counter fields; broad command-text regex matches are retained only as bounded
example context because shell heredocs and generated notes can mention
`git commit` without executing it.

The `missing_repository_for_agent_context` class is limited to rows where a
repository should be derivable from trusted agent or AAWM alias context. Generic
native Grok shell/pager passthrough rows (`provider=xai`, `client_name=grok-build`,
`passthrough_route_family=grok_cli_chat_proxy`) are excluded when they are not an
AAWM alias and carry no trusted repository source.

Rows that have already been classified as Codex repository text with an
untrusted source (`metadata.tenant_id_source=repository_untrusted` and
`metadata.repository_tenant_fallback_skipped=true`) are not treated as
`missing_repository_for_agent_context`; those rows are unresolved attribution,
not proof that a trusted repository was dropped. The sidecar still surfaces
large groups of non-excluded null repositories through
`large_null_repository_cluster`, which lets operators investigate dashboard
`unknown` repository spikes without backfilling prompt-derived or file-like
repository guesses. Once a repair pass stamps
`metadata.session_history_repository_status=unresolved` and
`metadata.session_history_repository_unresolved=true`, those rows are no longer
treated as unclassified active anomaly intake; fresh unclassified null rows
remain actionable.

The `stale_rate_limit_reset_with_recent_traffic` class only considers rate-limit
observations whose `observed_at` falls inside the same recent lookback window
used for the scan (`AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS`). Older
reset rows are ignored so historical `DISTINCT ON` snapshots cannot be matched
against unrelated recent provider traffic. Rows with a non-null `account_hash`
are also skipped for this anomaly because `session_history` does not carry a
matching account identifier; joining only on provider/model would produce
cross-account false positives until same-account traffic can be proven.

Relevant environment variables:

- `AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED`: enables the scheduled scan task.
- `AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS`: minimum seconds between
  scan attempts. Defaults to `3600`.
- `AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS`: recent database window
  scanned for anomalies. Defaults to `4`.
- `AAWM_OBSERVABILITY_ANOMALY_SCAN_STATEMENT_TIMEOUT_MS`: bounded statement
  timeout for each analytical query. Defaults to `15000`.
- `AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR`: directory for
  `<environment>-error.jsonl` anomaly intake. Defaults to
  `LITELLM_AAWM_ERROR_LOG_DIR` when set, otherwise `/app/.analysis`.
- Shared size bound (same env var as the generic AAWM error sink):
  `LITELLM_AAWM_ERROR_LOG_MAX_BYTES` (default `10485760` / 10 MiB). The sidecar
  refuses appends that would exceed this projected size rather than rotating or
  rewriting the shared active file. `LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT` is not
  used by this writer (backup-count `0` must never delete unresolved intake here).

In managed dev compose the task is enabled by default on the same hourly cadence
as the other scheduled sidecar tasks. The scan uses the sidecar environment name
(`AAWM_LITELLM_ENVIRONMENT`) for the output filename, so a `dev` sidecar writes
`/app/.analysis/dev-error.jsonl` inside the container and `.analysis/dev-error.jsonl`
in this repo when `.analysis` is mounted.

Each due attempt emits a separate `observability_anomaly_scan` JSON line with
sanitized status fields such as `attempted`, `status`, `lookback_hours`,
`anomaly_count`, `anomaly_classes`, `error_log_record_count`, and
`error_log_path`. Healthy scans keep `status=healthy`. When one or more anomaly
classes match, the scan sets `status=anomalies_found` and appends only new or
materially changed anomaly intake rows into the environment error file.

Intake write behavior (RR-089):

- **Append-safe shared sink**: the sidecar only opens
  `<environment>-error.jsonl` with `O_APPEND` (create+write). It never rewrites,
  truncates, renames, rotates, or unlinks the active shared JSONL. Concurrent
  generic/terminal AAWM writers that append to the same path therefore cannot
  lose rows to a sidecar read/merge/`os.replace` race.
- **Dedupe / standing check**: records are keyed by
  `(environment, anomaly_source, anomaly_class)` against the latest prior row for
  that identity in the active file (read-only index). An identical standing
  anomaly is not re-appended every scan interval (`error_log_record_count=0`).
  Material changes (row_count, expected, lookback, or sample identities) append a
  fresh durable row with a new `observed_at`; prior unresolved rows remain until
  operator cleanup. Distinct anomaly classes remain separate rows.
- **Preserve unrelated intake**: non-`aawm_observability_anomaly` JSONL rows
  already present in the active file are left untouched because the file is never
  rewritten. Unresolved intake is never deleted merely because it is unresolved,
  including when `LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT=0`.
- **Bound growth / projected-size refusal**: if `current_size + pending_line_bytes`
  would exceed `LITELLM_AAWM_ERROR_LOG_MAX_BYTES`, the write is refused
  (`error_log_record_count=0`) and existing intake is left intact. Ownership/mode
  repair still runs after successful appends only.
- **Operator signal**: `error_log_record_count` is the number of anomaly rows
  newly appended on that scan, not the total lines in the file.

Each anomaly record uses `event=aawm_observability_anomaly` and should
include at least:

- `environment`
- `observed_at`
- `error_class`
- `error_message`
- `anomaly_class`
- `anomaly_source` (`provider_status_observations_sidecar`)
- `lookback_hours`
- `row_count`
- `expected`
- bounded `examples`
- `recommended_todo`
- `cleanup_requirement`

The examples are intentionally bounded samples for triage. They must not include
raw prompts, tool arguments, request or response bodies, auth headers, API keys,
or credential payloads.

Treat these JSONL rows as normal active error intake, not as a separate queue.
Convert each anomaly class into or update a matching `.analysis/todo.md` item,
investigate the underlying telemetry mapping or persistence path, verify healthy
data, and then delete or archive the source `<environment>-error.jsonl` file once
the anomaly is resolved and recorded in completed notes. Scan failures are logged
as `status=scan_failed` on the sidecar event and do not raise out of the sidecar
loop.

## Langfuse ClickHouse `default.observations` metadata (Map)

Langfuse stores observation metadata in ClickHouse as a **Map**, not a JSON
string column. On `aawm-clickhouse`, `default.observations.metadata` is
`Map(LowCardinality(String), String)` (values are strings). Related columns
used in provider audits include `provided_model_name` (`Nullable(String)`) and
`start_time` (`DateTime64(3)`).

**Do not** filter provider identity with `JSONExtractString(metadata, 'custom_llm_provider')`
or similar JSON extractors on `metadata`; ClickHouse returns Code 43
`ILLEGAL_TYPE_OF_ARGUMENT` because the first argument is not JSON text.

**Use Map key access instead**, for example:

```sql
metadata['custom_llm_provider'] = 'anthropic'
```

Example Anthropic/Claude observation count (July 2026 window), verified after
replacing the bad predicate:

```sql
SELECT count()
FROM default.observations
WHERE start_time >= toDateTime64('2026-07-01 00:00:00', 3)
  AND (
    provided_model_name ILIKE '%claude%'
    OR provided_model_name ILIKE '%anthropic%'
    OR metadata['custom_llm_provider'] = 'anthropic'
  );
```

This repo's backfill scripts already use `metadata[...]` patterns; ad-hoc probes
and sibling dashboards should follow the same shape.
