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
- `alibaba_token_plan/deepseek-v4-pro`
- `alibaba_token_plan/glm-5.2`

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
reference. A managed deployment may instead set
`LITELLM_ALIBABA_TOKEN_PLAN_SETTINGS_FILE` to the canonical Qwen
`settings.json`; LiteLLM resolves the one shared credential entry used by the
approved Token Plan endpoint and model allowlist directly from that file.
Caller-supplied API-key and base-URL overrides are ignored so requests cannot
drift to another credential or endpoint. Deployment must mount the canonical
file in place; do not copy, synthesize, refresh, or log the credential.

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
GLM 5.2 is not exposed through public `sota-zai`; that name is reserved for
the dedicated Z.AI Coding Plan route. Any Alibaba GLM route remains deferred
and fail-closed pending dedicated configuration.

Codex Responses ingress uses
`codex_alibaba_token_plan_chat_completions_adapter`. Anthropic Messages ingress
uses `anthropic_alibaba_token_plan_chat_completions_adapter`. Both preserve the
inbound alias in observability metadata while sending only the resolved
provider model upstream.

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

## Quota observability

AAWM deployments may enable the provider-status sidecar's read-only
ModelStudio quota poll. It records the Token Plan 5-hour and 7-day Credit
windows without sending model traffic or using the plan-specific inference API
key. See `docs/aawm-provider-status-observations.md` for the
`AAWM_ALIBABA_WEB_AUTH_FILE` session-file contract, polling cadence, stored
quota keys, and degraded-session behavior. During migration, `ALIBABA_WEB_KEY`
remains a fallback while `AAWM_ALIBABA_WEB_AUTH_FILE` is proven in dev; remove
that fallback after proof.

The same maintained web session also observes the manual weekly reset-card
inventory through the console's read-only reset-card list contract. Manual
cards are separate from the automatic rolling 5-hour and 7-day quota-window
resets. The sidecar records sanitized card type and validity timestamps plus a
hashed card identity and lifecycle state; it never stores the raw card number
and never consumes or applies a reset. Available-card totals and per-card
current state are exposed through the shared provider-credit observations.

LiteLLM does not invent a per-token price for this subscription. Consumers must
not interpret a null invoice cost as a free request.

## Acceptance boundary

Instance-backed acceptance runs through the existing authenticated repository
harness against `litellm-dev` first. The required Codex and Claude cases prove
child-agent tool usage, parallel tool batches, exact Bash system-time stdout,
clean route/error logs, Langfuse correlation, and `aawm_tristore` persistence.
Production promotion is permitted only after the complete dev gate passes
against one candidate build and the current `PROD_RELEASE.md` has been read.
