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
the endpoint, credential name, model allowlist, billing provenance, and AAWM
alias routes remain separate.

## Supported models

- `alibaba_token_plan/qwen3.8-max`
- `alibaba_token_plan/qwen3.7-plus`
- `alibaba_token_plan/qwen3.7-max`
- `alibaba_token_plan/qwen3.6-flash`
- `alibaba_token_plan/deepseek-v4.1-flash`
- `alibaba_token_plan/deepseek-v4-pro`
- `alibaba_token_plan/glm-5.2`

The configured DeepSeek Flash model is `deepseek-v4.1-flash`; the provider
receives that exact unprefixed ID.

Other model IDs are rejected locally. The provider always resolves the public
LiteLLM model identity to the unprefixed provider model before egress. A public
AAWM alias such as `sota-alibaba` must never appear in an upstream request
body.

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
- `sota-zai`: last-resort Alibaba Token Plan GLM 5.2 after Z.AI Coding
  Plan `zai_coding_plan/glm-5.3`. Do not invent a second public name.

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

The catalog carries the current undiscounted international Alibaba Cloud Model
Studio direct list rates as non-invoice references for Qwen 3.6
(`$0.25/$1.50` per million input/output tokens through 256000 whole-request
input tokens, then `$1/$4`), Qwen 3.7 (`$2.50/$7.50`), and DeepSeek V4 Pro
(`$2.40/$4.80`). Output tokens do not select the Qwen 3.6 tier. These direct
list rates are distinct from Token Plan subscription economics and do not claim
the subscription invoice cost. Qwen 3.8 remains explicitly unpriced. Reference
totals are metadata only and do not populate `response_cost` or
`session_history.response_cost_usd`.

DeepSeek V4.1 Flash uses Singapore/International Model Studio direct reference
base rates of `$0.30/$1.20` per million input/output tokens and `$0.03` per
million cached input tokens. Off-peak and promotional discounts are excluded.
These are reference rates, not known Token Plan invoice costs.
Sources: [Model pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
and [context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache).

## Quota observability

AAWM deployments may enable the provider-status sidecar's read-only
ModelStudio quota poll. It records the Token Plan 5-hour and 7-day Credit
windows without sending model traffic or using the plan-specific inference API
key. The sidecar mints a console Bearer from RAM `ALIBABA_RAM_KEY` /
`ALIBABA_RAM_SECRET` (optional `ALIBABA_RAM_PRINCIPAL`) and calls the
Singapore CLI gateway `/cli/api.json`. See
`docs/aawm-provider-status-observations.md` for the RAM mint contract,
polling cadence, stored quota keys, and degraded last-good behavior.

The same RAM-minted Bearer also observes the manual weekly reset-card
inventory through the console's read-only reset-card list contract. Manual
cards are separate from the automatic rolling 5-hour and 7-day quota-window
resets. The sidecar records sanitized card type and validity timestamps plus a
hashed card identity and lifecycle state; it never stores the raw card number
and never consumes or applies a reset (`/reset-card/use` is out of scope).
Available-card totals and per-card current state are exposed through the
shared provider-credit observations.

LiteLLM does not invent a per-token price for this subscription. Consumers must
not interpret a null invoice cost as a free request.

Routing and recovery use only fresh, exact-environment, healthy Alibaba Token
Plan observations for the single configured account and the relevant `5h` and
`7d` windows. Missing, stale, unhealthy, malformed, mismatched, partial,
unavailable, or ambiguous evidence is unknown/fail-closed for recovery and
cannot clear `alibaba_token_plan:__account_quota__:alibaba_token_plan`.

When an explicit Alibaba Token Plan five-hour or weekly exhaustion is detected
in a response, LiteLLM publishes one shared account and lane cooldown covering
`qwen3.8-max` and `qwen3.7-max`, including their last-resort use. Generic or
ambiguous `429` responses do not trigger that cooldown. The cooldown lasts two
hours plus up to one hour of jitter. Any fresh confirmed exhausted `5h` or `7d`
window blocks all Alibaba candidates, even if its reset time has passed or the
other window is missing. Early recovery clears only
`alibaba_token_plan:__account_quota__:alibaba_token_plan`, and only when fresh
positive `5h` and `7d` evidence has finite reset times in the future and no
fresh exhausted evidence exists. Newer unavailable evidence invalidates older
positive cached observations; cross-account positive ambiguity neither blocks
nor clears. Response-driven cooldown behavior continues to work without
sidecar data.

## Acceptance boundary

Instance-backed acceptance runs through the existing authenticated repository
harness against `litellm-dev` first. The required Codex and Claude cases prove
child-agent tool usage, parallel tool batches, exact Bash system-time stdout,
clean route/error logs, Langfuse correlation, and `aawm_tristore` persistence.
Production promotion is permitted only after the complete dev gate passes
against one candidate build and the current `PROD_RELEASE.md` has been read.
