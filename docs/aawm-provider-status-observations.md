# AAWM Provider Status Observations

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

## Grok OIDC Refresh Task

The same sidecar can also own the scheduled Grok native OIDC credential refresh.
This is separate from the five-minute provider front-door probes. In dev compose
the sidecar mounts `/home/zepfu/.grok` writable and refreshes
`/home/zepfu/.grok/auth.json` on a one-hour cadence. The LiteLLM container mounts
that directory read-only and reads the credential directly for
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
  rejected and fall back to `0600`.
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

When the billing response includes `billingPeriodEnd` but no `monthlyLimit`,
`used`, `creditUsagePercent`, or `productUsage`, the sidecar still persists a
period-only snapshot for the Grok credits billing series. That row updates
`expected_reset_at` and the billing-period boundary columns while leaving
`remaining_pct` and absolute quota fields null.

Relevant environment variables:

- `AAWM_GROK_BILLING_POLL_ENABLED`: enables the scheduled billing poll.
- `AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS`: minimum seconds between billing
  poll attempts.
- `AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS`: billing endpoint timeout.
- `AAWM_GROK_BILLING_URL`: billing endpoint URL.
- `AAWM_GROK_BILLING_CLIENT_VERSION`: Grok CLI client version header.
  Defaults to `AAWM_GROK_BILLING_CLIENT_VERSION`,
  `LITELLM_XAI_GROK_CLIENT_VERSION`, `GROK_CLIENT_VERSION`, then `0.2.55`.
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
`model`, `status_code`, `attempt_count`, `retry_count`, `poll_max_attempts`,
`observation_count`, `inserted_count`, `error_class`, and `error_message`. For
D1-304 debugging, the event also includes compact request/transport diagnostics
such as `http_client`, `request_method`, `billing_host`, `billing_path`,
`billing_query_keys`, `billing_query_present`, `header_names`,
`include_model_override`, `model_override_configured`, `client_identifier`,
`x_xai_token_auth_configured`, and `request_contract_fingerprint`.

The fingerprint is derived from the non-secret request contract only: HTTP
method, billing host/path, query key names, configured client version and
identifier, whether `x-xai-token-auth` is configured, model-override flags, and
header names. It must not include authorization tokens, account identity values,
raw auth payloads, resolved IP addresses, or the configured
`x-xai-token-auth` value. The event must not emit dedicated identity fields or
raw auth headers. It must not contain access tokens, refresh tokens, id tokens,
client secrets, account identity values (`user_id`, `team_id`, `email`, or the
derived `x-userid`, `x-grok-user-id`, `x-teamid`, and `x-email` headers), or the
full billing credential payload. Billing poll failures are logged and do not
raise out of the sidecar loop.

Successful sidecar billing polls copy the same safe request-contract evidence
into `rate_limit_observations.evidence` with
`request_contract_source=grok_billing_sidecar_poll`. This lets later DB-only
investigations distinguish snapshots inserted by the scheduled sidecar from
snapshots extracted from Grok passthrough/manual traffic without storing auth
tokens or account identity values.

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
Codex OAuth credential from `AAWM_CODEX_AUTH_FILE`, calls the native ChatGPT
reset-credit **detail** endpoint (default
`https://chatgpt.com/backend-api/wham/rate-limit-reset-credits`) with
`Authorization: Bearer <token>` and `ChatGPT-Account-Id` when the auth file
includes an account id, and persists sanitized rows to
`public.provider_credit_observations` (not `public.rate_limit_observations`).

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
provider/model fields, alias metadata that was not promoted, token or git/tool
activity counters that do not match persisted activity, or stale rate-limit
reset timestamps that still have matching recent traffic.

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
- `AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR`: directory where detected
  anomalies are appended as `<environment>-error.jsonl`. Defaults to
  `LITELLM_AAWM_ERROR_LOG_DIR` when set, otherwise `/app/.analysis`.

In managed dev compose the task is enabled by default on the same hourly cadence
as the other scheduled sidecar tasks. The scan uses the sidecar environment name
(`AAWM_LITELLM_ENVIRONMENT`) for the output filename, so a `dev` sidecar writes
`/app/.analysis/dev-error.jsonl` inside the container and `.analysis/dev-error.jsonl`
in this repo when `.analysis` is mounted.

Each due attempt emits a separate `observability_anomaly_scan` JSON line with
sanitized status fields such as `attempted`, `status`, `lookback_hours`,
`anomaly_count`, `anomaly_classes`, `error_log_record_count`, and
`error_log_path`. Healthy scans keep `status=healthy`. When one or more anomaly
classes match, the scan sets `status=anomalies_found` and appends one append-safe
JSONL record per anomaly class to the environment error file.

Each appended anomaly record uses `event=aawm_observability_anomaly` and should
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
