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
