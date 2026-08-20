# Z.AI Coding Plan (`zai_coding_plan`)

Dedicated LiteLLM provider for the **Z.AI Coding Plan** OpenAI-compatible
chat endpoint. This is **not** ordinary [`zai`](/docs/providers/zai)
list-rate API.

| Property | Details |
| --- | --- |
| Provider route | `zai_coding_plan/` |
| Protocol | OpenAI-compatible chat completions |
| Base URL | `https://api.z.ai/api/coding/paas/v4` |
| Chat URL | `https://api.z.ai/api/coding/paas/v4/chat/completions` |
| Credential | `ZAI_KEY` (then `ZAI_CODING_PLAN_API_KEY`, then `ZHIPU_API_KEY`) |
| Not used | Ordinary `ZAI_API_KEY`, `https://api.z.ai/api/paas/v4`, `open.bigmodel.cn` |

Ordinary `zai/glm-*` stays on `/api/paas/v4` with `ZAI_API_KEY`. Sending a
Coding Plan request to that base (error 1113) is a routing/billing defect,
not a fallback.

## Supported models

- `zai_coding_plan/glm-5.3`
- `zai_coding_plan/glm-5-turbo`
- `zai_coding_plan/glm-4.7`

Admission is the documented Coding Plan set. GLM-5.2 / GLM-5.1 leftovers
and public alias names such as `sota-zai` are rejected as provider models.

## Credential and endpoint contract

```python
import os

os.environ["ZAI_KEY"] = "..."
```

Caller-supplied API-key and base-URL overrides are ignored unless the
base is exactly the canonical coding origin. Ordinary `ZAI_API_KEY` is
not reused.

## Usage

```python
from litellm import completion

response = completion(
    model="zai_coding_plan/glm-5.3",
    messages=[{"role": "user", "content": "Explain the result briefly."}],
)
print(response.choices[0].message.content)
```

Thinking is enabled by default (`thinking.type=enabled`,
`clear_thinking=false`). Inbound `reasoning.effort` / `reasoning_effort`
maps onto Z.AI `low` / `high` / `max`.

## AAWM alias

Public `sota-zai` prefers this provider first:

1. `zai_coding_plan/glm-5.3` on
   `codex_zai_coding_plan_chat_completions_adapter` (priority 110)
2. last-resort `alibaba_token_plan/glm-5.2` on
   `codex_alibaba_token_plan_chat_completions_adapter` (priority 100)

Usage is `POST /openai_passthrough/v1/responses` with `model=sota-zai` or
`model=zai_coding_plan/glm-5.3`. Generic `/v1/chat/completions` is not an
alias ingress. Coding Plan is Codex-only: there is no
`anthropic_zai_coding_plan_*` family. Do not invent `aawm-sota-zai` or
`sota-zcode`. Do not add `sota-zai` to `sota.yaml` TUI dispatch. Leave
`aawm-sota-glm`, ordinary `zai`, and every `aawm-code*` unchanged.

## Cost provenance

Coding Plan generations record:

- `billing_mode=zai_coding_plan_subscription`
- `actual_invoice_cost_known=false`
- catalog `aawm_reference_pricing` only

Do not copy ordinary `zai/glm-*` list rates. Quota windows are a
subscription credit package; there is currently no stable public JSON
quota API (`public_quota_endpoint_unavailable`).
