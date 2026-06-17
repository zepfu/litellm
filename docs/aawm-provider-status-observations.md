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
- `AAWM_GROK_OIDC_AUTH_FILE`: Grok CLI auth JSON path.
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

Relevant environment variables:

- `AAWM_GROK_BILLING_POLL_ENABLED`: enables the scheduled billing poll.
- `AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS`: minimum seconds between billing
  poll attempts.
- `AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS`: billing endpoint timeout.
- `AAWM_GROK_BILLING_URL`: billing endpoint URL.
- `AAWM_GROK_BILLING_CLIENT_VERSION`: Grok CLI client version header.
- `AAWM_GROK_BILLING_CLIENT_IDENTIFIER`: Grok CLI client identifier header.
- `AAWM_GROK_BILLING_XAI_TOKEN_AUTH`: `x-xai-token-auth` header value.
- `AAWM_GROK_BILLING_MODEL`: model label stored with the billing snapshot.
- `AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE`: when true, include native
  Grok billing request-shape headers on billing poll requests:
  `content-type: application/json` and `x-grok-model-override` using
  `AAWM_GROK_BILLING_MODEL`. Defaults to true so the sidecar matches successful
  native Grok passthrough/manual billing calls. Set to false only when an
  operator explicitly wants the older minimal header shape. The model label is
  still persisted on `rate_limit_observations.model` regardless of this
  setting.
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
`billing_url`, `client_version`, `model`, `status_code`, `attempt_count`,
`retry_count`, `observation_count`,
`inserted_count`, `error_class`, and `error_message`. The event must not emit
dedicated identity fields or raw auth headers. It must not contain access
tokens, refresh tokens, id tokens, client secrets, account identity values
(`user_id`, `team_id`, `email`, or the derived `x-userid`, `x-grok-user-id`,
`x-teamid`, and `x-email` headers), or the full billing credential payload.
Billing poll failures are logged and do not raise out of the sidecar loop.
