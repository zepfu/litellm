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

## Grok Billing Poll Task

The same sidecar can also run an explicit hourly Grok billing poll. This is
telemetry-only and separate from the five-minute provider front-door probes and
the Grok OIDC refresh task. The poll reads the current OIDC access token from
`AAWM_GROK_OIDC_AUTH_FILE`, calls
`https://cli-chat-proxy.grok.com/v1/billing?format=credits` with Grok CLI-style
headers, and persists the returned billing snapshot as a sanitized
`rate_limit_observations` row using the same stored field shape and dedupe guard
as the LiteLLM callback path.

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

Each due attempt emits a separate `grok_billing_poll` JSON line with sanitized
status fields such as `attempted`, `persisted`, `skipped`, `auth_file`,
`billing_url`, `client_version`, `model`, `status_code`, `observation_count`,
`inserted_count`, `error_class`, and `error_message`. The event must not contain
access tokens, refresh tokens, id tokens, client secrets, raw auth headers, or
the full billing credential payload. Billing poll failures are logged and do not
raise out of the sidecar loop.
