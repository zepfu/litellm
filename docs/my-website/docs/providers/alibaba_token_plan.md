# Alibaba Token Plan

## Overview

| Property | Details |
| --- | --- |
| Provider route | `alibaba_token_plan/` |
| Protocol | OpenAI-compatible chat completions |
| Base URL | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| Credential | Existing `ALIBABA_KEY` environment variable or canonical Qwen settings file |

Alibaba Token Plan is a bounded provider integration for the Token Plan
subscription endpoint. It is distinct from the general `dashscope/` provider:
the endpoint, credential name, model catalog, billing provenance, and AAWM
alias routes remain separate.

## Supported models

- `alibaba_token_plan/qwen4-max`
- `alibaba_token_plan/qwen4-max-preview`
- `alibaba_token_plan/qwen3.8-max`
- `alibaba_token_plan/qwen3.7-plus`
- `alibaba_token_plan/qwen3.7-max`
- `alibaba_token_plan/qwen3.8-flash`
- `alibaba_token_plan/deepseek-v4.1-flash`
- `alibaba_token_plan/deepseek-v4-pro`
- `alibaba_token_plan/glm-5.3`

The configured DeepSeek Flash model is `deepseek-v4.1-flash`; the provider
receives that exact unprefixed ID.

`deepseek-v4.1-flash-1017` remains an unresolved provider identifier. Retained
evidence does not establish exact equivalence to `deepseek-v4.1-flash`, active
Token Plan entitlement, or a separate reference tariff. It is intentionally
absent from this provider's model metadata and reference catalog; do not apply
the current Flash reference rates to it until exact provider evidence is
available.

## Reasoning effort

Each Token Plan model carries a `reasoning_effort_wire` contract in
`model_prices_and_context_window.json`. The chat transformation reads that
contract through `get_model_info` and writes the provider-bound body from it.
Direct HTTP calls place these fields at the top level of the chat-completions
body. Source: the QwenCloud OpenAI chat API reference.

| Model | Configured effort | Serialized field |
| --- | --- | --- |
| `qwen4-max` | Provider default `high` | Native `reasoning_effort` accepts `low`, `high`, and `xhigh` |
| `qwen4-max-preview` | `max` on alpha | `reasoning_effort=xhigh`; native values are `low`, `medium`, and `xhigh` |
| `qwen3.8-max` | `max` | `reasoning_effort=xhigh` (`max` and `high` map to `xhigh`; `none` sets `enable_thinking=false`) |
| `qwen3.7-max`, `qwen3.7-plus` | `max` when set | `enable_thinking=true`. `reasoning_effort` is not a supported wire field |
| `deepseek-v4-pro`, `glm-5.3` | `max` | `reasoning_effort=max` (native `high` or `max`; `low`/`medium` map to `high`; `xhigh` maps to `max`) |
| `deepseek-v4.1-flash` | omitted | no effort field; provider default is `high` |
| `qwen3.8-flash` | omitted | no effort field; provider default is unchanged |
| `qwen3.8-max-preview` | — | reasoning effort is rejected |

`qwen3.8-flash` has no configured effort, so the request omits both fields and
leaves the provider thinking default in place.

The adapter accepts a bare model ID or `alibaba_token_plan/<model-id>` and
sends the bare model ID upstream. The catalog supplies model capabilities and
reference pricing. The Token Plan endpoint determines account access.

## Credential and endpoint contract

```python
import os

os.environ["ALIBABA_KEY"] = "..."
```

The integration first reads `ALIBABA_KEY` from the existing environment
reference on every request. That value is not cached, so replacing it takes
effect on the next request. A managed deployment may instead set
`LITELLM_ALIBABA_TOKEN_PLAN_SETTINGS_FILE` to the canonical Qwen
`settings.json`; LiteLLM resolves the one shared credential entry used by the
approved Token Plan endpoint and model allowlist directly from that file.
Repeated requests reuse that parsed file credential for 60 seconds and do not
read the file again while it is unchanged. After 60 seconds, the next request
checks file identity and reads a replacement. Unchanged files are not parsed
again. `LITELLM_ALIBABA_TOKEN_PLAN_SETTINGS_CACHE_TTL_SECONDS` may set a
shorter positive window, and values above 300 seconds are ignored so a
replaced file cannot stay cached indefinitely. The cache key is the settings
path, never the credential. Caller-supplied API-key and base-URL overrides
are ignored so requests cannot drift to another credential or endpoint.
Deployment must mount the canonical file in place; do not copy, synthesize,
refresh, or log the credential.

## Usage

```python
from litellm import completion

response = completion(
    model="alibaba_token_plan/qwen3.8-max",
    messages=[{"role": "user", "content": "Explain the result briefly."}],
)
print(response.choices[0].message.content)
```

Streaming uses the same provider route:

```python
from litellm import completion

response = completion(
    model="alibaba_token_plan/qwen3.7-max",
    messages=[{"role": "user", "content": "List three checks."}],
    stream=True,
)
for chunk in response:
    print(chunk)
```

## AAWM aliases and adapters

The AAWM proxy configuration may expose:

- `sota-alibaba`: Qwen 3.8 Max, then Qwen 3.7 Max.
- `sota-deepseek`: DeepSeek V4 Pro.
- `sota-zai`: last-resort Alibaba Token Plan GLM 5.3 after Z.AI Coding
  Plan `zai_coding_plan/glm-5.3`. Do not invent a second public name.

Alpha mounts overrides from `config/alpha-alias-overrides/` that place
`qwen4-max-preview` first in `sota-alibaba`, `provider-alibaba`, and
`expert-other`, with configured effort `max`. Qwen4 is admitted only during
the official preview availability window, from 10:00 on November 11 through
24:00 on November 25, 2026 (UTC+8); `provider-alibaba` and `sota-alibaba`
use an inclusive internal absolute end at the last representable microsecond
before the exclusive expiry. `expert-other` references a same-file Qwen4 leaf
bounded to that full preview window; its nightly helper schedule gates the
reference while the leaf's absolute schedule preserves ordinary priority and
the half-open expiry. Fallback candidates and schedules are preserved. `sota`
inherits this order for Qwen clients, and `expert` references `expert-other`.
Canonical alias files and dev/production deployments retain their current
order. Only `docker-compose.alpha.yml` mounts the three Qwen overrides.

The preview supports a 1M-token context and up to 393,216 output tokens.
Its `max_tokens` budget includes both thinking and the final answer. Alibaba
may report the canonical name `qwen4.0-max-preview` in responses. The announced
preview ends after November 25, 2026 at 24:00 (UTC+8), after which the model
is retired.

Stable `qwen4-max` has a 1M-token context. Without thinking it supports up to
991,808 input tokens and 131,072 output tokens. With thinking enabled the
limits are 983,616 input tokens and 262,144 output tokens.
[Qwen4-Max model reference](https://www.alibabacloud.com/help/en/model-studio/qwen4-max)

Codex Responses ingress uses
`codex_alibaba_token_plan_chat_completions_adapter`. Anthropic Messages ingress
uses `anthropic_alibaba_token_plan_chat_completions_adapter`. Both preserve the
inbound alias in observability metadata while sending only the resolved
provider model upstream.

### Auto-review structured output

For schema-bearing Codex auto-review requests routed to Alibaba Token Plan,
the adapter includes the requested JSON schema in the prompt instead of
sending an upstream `response_format`. This does not imply native structured
output support on the Token Plan endpoint.

Before returning a review, the adapter requires exactly one native assistant
text choice with `finish_reason="stop"`, without tool calls or refusals, and
validates its JSON against the requested schema. Streaming responses are
buffered until validation completes. Invalid output cannot become an approval.
This handling does not change ordinary request formats or alias ordering.

## Cost provenance

Token Plan generations record:

- `billing_mode=alibaba_token_plan_subscription`
- `actual_invoice_cost_known=false`
- `reference_cost_model` as the actual `alibaba_token_plan/...` route key
- `reference_cost_kind` and the source/version/equivalence provenance from the
  catalog reference contract

The catalog carries current undiscounted International and Singapore Alibaba
Cloud Model Studio direct list rates as non-invoice references. Qwen 3.8 Max
uses `$2/$0.25/$0.17/$2.50/$6` per million input, implicit-cache-read,
explicit-cache-read, explicit-cache-write, and output tokens. Qwen 3.7 Max uses
`$2.50/$0.25/$0.25/$3.125/$7.50`; Qwen 3.8 Flash uses
`$0.15/$0.016/$0.016/$0.20/$0.47`.

Retained Qwen 3.6 Flash reference tiers use
`$0.25/$0.025/$0.025/$0.3125/$1.50` through 256K whole-request input tokens
and `$1/$0.10/$0.10/$1.25/$4` through 1M. Output tokens do not select that
tier. DeepSeek V4.1 Flash uses `$0.30/$0.03/$1.20` for input,
implicit-cache-read, and output; GLM 5.2 and 5.3 use `$1.40/$0.28/$4.40`;
DeepSeek V4 Pro uses `$2.40/$0.20/$4.80`. The dated
`deepseek-v4-pro-0813` tariff is not assigned to the current alias.

These direct list rates are distinct from Token Plan subscription economics and
do not claim the subscription invoice cost. Off-peak PAYG and promotional
discounts are excluded. When a contract has separate implicit and explicit
cache rates, the resolver requires the usage record to identify the cache kind;
missing attribution remains unpriced. Reference totals are metadata only and do
not populate `response_cost` or `session_history.response_cost_usd`.
Sources: [Model pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
and [context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache).

The stable `qwen4-max` route uses its own Singapore International standard
rates for reference pricing. The Token Plan subscription invoice remains
unknown. The preview uses those same rates as a comparison with production.

Qwen4-Max-Preview is available only through Token Plan, with no published
pay-as-you-go tariff; the preview ends before November 25, 2026 at 16:00 UTC
and the model is retired afterward. It uses the production Qwen4-Max standard
Singapore international PAYG list rates as a comparison baseline: `$1.50` per
million input tokens, `$4.50` per million output tokens, `$0.075` per million
cache hits, `$1.875` per million 5-minute cache-creation tokens, and `$3` per
million 1-hour cache-creation tokens. This is labeled
`production_model_baseline`, with basis model `qwen4-max`; it claims neither
model equivalence nor the preview's invoice price, including for cache-bearing
usage. Promotional Credits discounts are excluded. When implicit and explicit
cache rates differ, missing cache-kind attribution remains unpriced. The
1-hour creation rate is reference metadata; the shared reference-cost contract
prices only the 5-minute creation rate. Sources:
[Qwen4-Max-Preview](https://www.alibabacloud.com/help/en/model-studio/qwen4-max-preview)
and [Model pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing).

## Quota observability

AAWM deployments may enable the provider-status sidecar's read-only
ModelStudio quota poll. It records reported Token Plan 5-hour, 7-day, or
monthly Credit windows without sending model traffic or using the plan-specific
inference API key. The sidecar mints a console Bearer from RAM
`ALIBABA_RAM_KEY` / `ALIBABA_RAM_SECRET` (optional `ALIBABA_RAM_PRINCIPAL`)
and calls the Singapore CLI gateway `/cli/api.json`. See
`docs/aawm-provider-status-observations.md` for the RAM mint contract,
polling cadence, stored quota keys, and degraded last-good behavior.

The same RAM-minted Bearer also observes the manual weekly reset-card
inventory through the console's read-only reset-card list contract. Manual
cards are separate from the automatic quota-window resets. The sidecar records
sanitized card type and validity timestamps plus a hashed card identity and
lifecycle state; it never stores the raw card number and never consumes or
applies a reset (`/reset-card/use` is out of scope).
Available-card totals and per-card current state are exposed through the
shared provider-credit observations.

LiteLLM does not invent a per-token price for this subscription. Consumers must
not interpret a null invoice cost as a free request.

Routing and recovery use only fresh, exact-environment, healthy Alibaba Token
Plan observations for the single configured account: either `5h` and `7d`, or
monthly alone. Missing, stale, unhealthy, malformed, mismatched, partial,
unavailable, or ambiguous evidence is unknown/fail-closed for recovery and
cannot clear `alibaba_token_plan:__account_quota__:alibaba_token_plan`.

When an explicit Alibaba Token Plan five-hour or weekly exhaustion is detected
in a response, LiteLLM publishes one shared account and lane cooldown covering
`qwen3.8-max` and `qwen3.7-max`, including their last-resort use. Generic or
ambiguous `429` responses do not trigger that cooldown. The cooldown lasts two
hours plus up to one hour of jitter. Any fresh confirmed exhausted quota window
blocks all Alibaba candidates, even if its reset time has passed or another
window is missing. Early recovery clears only
`alibaba_token_plan:__account_quota__:alibaba_token_plan`, and only when fresh
positive evidence for a complete window set (`5h` and `7d`, or monthly alone)
has finite reset times in the future and no fresh exhausted evidence exists.
Newer unavailable evidence invalidates older positive cached observations;
cross-account positive ambiguity neither blocks nor clears. Response-driven
cooldown behavior continues to work without sidecar data.

## Acceptance boundary

Instance-backed acceptance runs through the existing authenticated repository
harness against `litellm-dev` first. The required Codex and Claude cases prove
child-agent tool usage, parallel tool batches, exact Bash system-time stdout,
clean route/error logs, Langfuse correlation, and `aawm_tristore` persistence.
Production promotion is permitted only after the complete dev gate passes
against one candidate build and the current `PROD_RELEASE.md` has been read.
