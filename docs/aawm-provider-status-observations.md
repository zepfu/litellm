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
`/app/kimi-descriptor/kimi-native-contract.json`.

The standalone image default
`LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED=false` is compatibility mode. Managed
dev Compose defaults the gate to `true`. In required mode, a missing, stale,
malformed, digest-invalid, endpoint-mismatched, version-incoherent, or hostile
descriptor fails closed and records a sanitized failure. It must not fall back
to generic `moonshot/*`, API-key authentication, or another endpoint.

The descriptor does not define usage-to-chat correlation. Any such join must
be proved by the polling and persistence runtime evidence; prompt text,
caller-supplied identity, and model-name guesses are not descriptor join keys.

The provider-status sidecar records non-inference front-door health signals for
configured providers. It probes DNS, TCP/TLS, and optional ICMP paths, then
writes rows to `public.provider_status_observations`.

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
locks, repair metadata, write files, or make network calls. Each provider emits
one sanitized event with a health status of `fresh`, `expired`, `degraded`, or
`malformed`; the event omits the credential path and all token values. Persisted
rows use `source_task=provider_auth_health_poll` and metadata flags
`passive_read_only=true`, `network_calls=false`, and
`credential_file_mutated=false`.

Relevant environment variables:

- `AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED`: enables the passive inspection.
  Defaults to disabled so existing credential-refresh ownership is unchanged.
- `AAWM_PROVIDER_AUTH_HEALTH_POLL_INTERVAL_SECONDS`: minimum seconds between
  inspections; defaults to `3600`.

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

OPENAI-001 does not add multi-account scheduling, quota polling, or routing.
The current sidecar refresh, passive health, and reset-credit tasks still use
one `AAWM_CODEX_AUTH_FILE` / `AAWM_CODEX_LOCK_FILE` pair; development Compose
points that pair at `account1`. `account2` is mounted and enrolled in the
read-only proxy inventory but is not scheduled by those tasks until OPENAI-002.
Per-account quota persistence remains OPENAI-003 and account routing remains
OPENAI-004.

Enrollment, removal, label/hash handling, permissions, rotation, and rollback
are defined in `docs/aawm-oauth-credential-maintenance.md`.

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

Authentication is file-driven and reloaded on each due poll. The sidecar reads
`AAWM_ALIBABA_WEB_AUTH_FILE` (default:
`/home/zepfu/.alibaba/token-plan-session.json`) and requires a JSON payload in
this exact shape:

```json
{"version":1,"login_ticket":"..."}
```

The auth file must be a regular file at the mounted path, not a symlink, and
mode `0600`. The file payload is treated as cookie-only session bootstrap
material for subscription and usage calls.

The sidecar attempts cookie-only subscription/usage calls first. If `sec_token`
is required, discovery is bounded and in-memory only: first through the
dashboard endpoint, then through `/tool/user/info.json`, with exactly one retry.
The discovered `sec_token` is never persisted and is discarded after the poll
cycle.

The sidecar does not copy the Token Plan inference API key, run browser
flows, launch secondary CLIs, ask for passwords, or automate MFA. Session
lifetime is finite: expired sessions are surfaced as degraded events until
`AAWM_ALIBABA_WEB_AUTH_FILE` is replaced.

The replacement procedure is atomic write-to-temp plus rename to the configured
path. A replaced file is picked up automatically on the next due poll without
container restart.

The usage request runs at startup and then on the configured usage cadence.
Subscription metadata is refreshed at startup and independently on the
configured subscription cadence. The subscription response supplies the
active-plan status, plan specification, period boundaries, and a plan-instance
identifier used only to derive `account_hash`; the raw identifier is not
persisted.

Relevant environment variables:

- `AAWM_ALIBABA_WEB_AUTH_FILE`: file-backed session bootstrap for quota polling.
  Defaults to `/home/zepfu/.alibaba/token-plan-session.json`.
- `ALIBABA_WEB_KEY`: legacy migration fallback only. Never log or persist its
  decoded values in database fields. Remove this fallback after dev proof.
- `AAWM_ALIBABA_QUOTA_POLL_ENABLED`: enables the scheduled poll.
- `AAWM_ALIBABA_QUOTA_POLL_INTERVAL_SECONDS`: minimum seconds between usage
  polls; the managed sidecar default is `300`.
- `AAWM_ALIBABA_SUBSCRIPTION_POLL_INTERVAL_SECONDS`: minimum seconds between
  subscription refreshes; the managed sidecar default is `21600`.
- `AAWM_ALIBABA_QUOTA_POLL_HTTP_TIMEOUT_SECONDS`: request timeout; the managed
  sidecar default is `30`.
- `AAWM_ALIBABA_QUOTA_GATEWAY_URL`: override for the undocumented ModelStudio
  gateway base URL. Keep the default unless contract verification proves a
  provider change.
- `AAWM_ALIBABA_QUOTA_POLL_MAX_ATTEMPTS`: bounded attempts per endpoint call;
  the managed sidecar default is `2`.
- `AAWM_ALIBABA_QUOTA_POLL_RETRY_BACKOFF_SECONDS`: base exponential backoff for
  transient failures; the managed sidecar default is `0.5`.

Each due attempt emits one sanitized `alibaba_quota_poll` JSON event. Runtime
success requires HTTP 200 responses, an active subscription, two parsed
observations, successful persistence, and no credential, raw response, account
identity, or traceback in container logs.

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

The same sidecar can run an explicit hourly Codex banked usage-limit reset-credit
poll. This is telemetry-only and separate from the five-minute provider front-door
probes, Codex OAuth refresh, and Grok billing poll. The poll reads the current
Codex OAuth credential from the compatibility `AAWM_CODEX_AUTH_FILE` record
(`account1` in development Compose), calls the native ChatGPT reset-credit
**detail** endpoint (default
`https://chatgpt.com/backend-api/wham/rate-limit-reset-credits`) with
`Authorization: Bearer <token>` and `ChatGPT-Account-Id` when the auth file
includes an account id, and persists sanitized rows to
`public.provider_credit_observations` (not `public.rate_limit_observations`).
It does not iterate the ordered inventory under OPENAI-001.

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
`attempted`, `persisted`, `status_code`, `attempt_count`, `retry_count`,
`available_count`, `inserted_count`, `poll_url`, `error_class`, and
`error_message`. Events must not contain access tokens, refresh tokens, raw auth
headers, account ids, or emails.

The detail endpoint is undocumented and provider-owned; shape may change without notice.


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
