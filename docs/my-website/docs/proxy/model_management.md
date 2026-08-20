import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Management
Add new models + Get model info without restarting proxy.

## In Config.yaml 

```yaml
model_list:
  - model_name: text-davinci-003
    litellm_params: 
      model: "text-completion-openai/text-davinci-003"
    model_info: 
      metadata: "here's additional metadata on the model" # returned via GET /model/info
```

## Get Model Information - `/model/info`

Retrieve detailed information about each model listed in the `/model/info` endpoint, including descriptions from the `config.yaml` file, and additional model info (e.g. max tokens, cost per input token, etc.) pulled from the model_info you set and the [litellm model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). Sensitive details like API keys are excluded for security purposes.


## AAWM OpenAI gpt-5.6 catalog (D1-478)

LiteLLM's canonical cost map includes `gpt-5.6-sol`, `gpt-5.6-terra`, and
`gpt-5.6-luna` in both `model_prices_and_context_window.json` and
`litellm/bundled_model_prices_and_context_window_fallback.json`. Pricing and
cache billing fields are sourced from
[Previewing GPT-5.6 Sol](https://openai.com/index/previewing-gpt-5-6-sol/)
(per 1M tokens: Sol $5/$30, Terra $2.50/$15, Luna $1/$6; cache write 1.25× input;
cache read 10% of input). All three entries advertise reasoning efforts through
API value `max` (`none`, `low`, `medium`, `high`, `xhigh`, `max`) using the
config-driven `supports_max_reasoning_effort` capability. Context-window and
unrelated capability fields remain omitted until independently verified.

The user-level Codex model catalog exposes `max` for Sol, Terra, and Luna.
`ultra` is exposed only for Sol: it is a Codex product intelligence mode that
combines maximum reasoning with proactive subagent delegation, not an API
`reasoning.effort` value sent by LiteLLM.

Config-driven alias routing uses:

- `sota-openai`: `gpt-5.6-sol`
- `sota-xai`: `oa_xai/grok-4.6` (priority 100), then
  `cursor_agent/cursor-grok-4.6-high` (priority 90)
- Managed `sota-xai` has no candidate-level reasoning override: caller
  `reasoning.effort=xhigh` is sent unchanged to `oa_xai/grok-4.6`. Request
  metadata records the requested/native effort as `xhigh` and the
  provider-native field/provider; the route rollup is
  `grok-4.6(sota-xai):xhigh`. Cursor Grok is a lower-priority inherited
  fallback, distinct from `oa_xai/grok-4.6` and `xai/grok-4.6`.
- `sota`: selects the producer-family `sota-*` alias from TUI origin, defaulting to `sota-openai`.
- Grok 4.6 (`xai/grok-4.6`, `oa_xai/grok-4.6`): 500k context; created 1785974400 (2026-08-06T00:00:00Z), owned by xAI, no aliases, source `https://docs.x.ai/developers/models/grok-4.6`, verified 2026-08-12; input $2/M, output $6/M, cached input $0.50/M, image input $2/M image tokens; above 200k whole-request input tokens, input/cache/output are $4/$1/$12 per million; managed-route session history records this as a reference rate with invoice cost unknown
- Cursor Agent Composer 2.5 standard (`cursor_agent/composer-2.5`): CLI slug `composer-2.5`, not Fast and not xAI `grok-composer-2.5-fast`; public list $0.50/$0.20/$2.50 per million input/cache-read/output, reference-only with invoice cost unknown
- Cursor Grok 4.6 (`cursor_agent/cursor-grok-4.6-high`): CLI slug `cursor-grok-4.6-high`, distinct from `oa_xai/grok-4.6` and `xai/grok-4.6`; public list $2/$0.50/$6 per million input/cache-read/output, reference-only with invoice cost unknown; temporary launch discount is not baked in
- `sota-alibaba`: `alibaba_token_plan/qwen3.8-max` → `alibaba_token_plan/qwen3.7-max`
- `sota-zai`: `alibaba_token_plan/glm-5.2`
- `basic`: the config-driven low-cost alias; Cursor Composer 2.5 standard (`cursor_agent/composer-2.5`) sits at priority 42 after `alibaba_token_plan/deepseek-v4-flash-0731`
- `work`: `gpt-5.3-codex-spark` → nested `work-other` alias reference → Claude-only native Sonnet tail → `gpt-5.6-luna`
- `work-other`: ordinary configured alias and valid exact-name / `alias_reference` target; omitted from Codex and Claude TUI selection only by those clients' explicit model-definition inclusion lists. During `22:00-08:00 UTC+8` the order is `sota-deepseek` (`alibaba_token_plan/deepseek-v4-pro`), then `sota-moonshot`, then `sota-xai` (`oa_xai/grok-4.6` preferred, then inherited `cursor_agent/cursor-grok-4.6-high`). Outside that window DeepSeek is omitted from new selection. Qwen Max models are not `work-other` candidates.
- `expert`: during `22:00-08:00 UTC+8`, `alibaba_token_plan/qwen3.8-max` is first for every ingress. Claude-origin in-window order is Qwen, then native Anthropic `claude-opus-5`, then universal `gpt-5.6-terra`. Codex / non-Claude in-window order is Qwen, then Terra. Outside the window the previous Opus/Terra behavior is unchanged. Canonical Opus 5 is inherently 1M-context; there is no `claude-opus-5[1m]` selector. All three compiled candidates use authoritative `reasoning_effort: max` via the CFG-006 candidate pipeline. Opus egresses only through Anthropic-native credentials; Terra keeps its OpenAI/Codex credential domain on both ingresses.

Config-driven AAWM aliases and candidates come only from the compiled YAML
snapshot. Missing or failed config fails closed; there is no built-in candidate
table and no startup or no-snapshot fallback. Every configured YAML alias is an
ordinary exact-name route with no public/internal routing distinction.

Non-Codex-native `GET /openai_passthrough/v1/models` lists compiled YAML
aliases. Usage is `POST /openai_passthrough/v1/responses`. Generic
`/v1/chat/completions` is not an alias ingress. The public name is `sota-zai`.

### Codex reasoning effort follows the resolved model

For native OpenAI Codex Responses traffic, LiteLLM reconciles a recognized
reasoning effort after alias or adapter resolution and before provider egress.
The supported order is `none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
then `max`.

The mapping is capability-driven and downward-only. For example,
`reasoning.effort=max` becomes `xhigh` when `work` resolves to
`gpt-5.3-codex-spark`, while GPT-5.6 Sol, Terra, and Luna retain `max` because
their model entries advertise `supports_max_reasoning_effort=true`. Direct
concrete-model Codex passthrough uses the same rule. Alias fallback attempts
recalculate from the original request, so a later candidate with a higher
ceiling can retain the original effort.

### Candidate-level reasoning policy

An alias YAML candidate may optionally set an authoritative reasoning effort:

```yaml
candidates:
  - model: gpt-5.6-luna
    reasoning_effort: low
```

Accepted values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and
`max`. A configured candidate value replaces reasoning supplied by the caller
or TUI. If `reasoning_effort` is omitted, the candidate preserves the caller's
original intent. Provider/model translation or clamping follows the existing
capability policy and does not bypass provider constraints.

Each failover candidate reevaluates this policy from the original caller
intent, rather than reusing a previous candidate's transformed request.
Request and configuration sources are recorded in audit and route metadata
without including secrets.

LiteLLM does not guess for unknown effort strings, unknown model ceilings, or
providers without an explicit compatible capability contract. Those requests
retain provider-native validation. A deterministic invalid-effort HTTP 400 is
not retried as an alias capacity failure.

Operators can inspect `reasoning_effort_requested`,
`reasoning_effort_native_value`, `reasoning_effort_supported_ceiling`,
`reasoning_effort_resolved_model`, `reasoning_effort_resolved_provider`,
`reasoning_effort_mapping_reason`, and the alias candidate attempt in request
metadata and session history. Downward mappings also carry clamp metadata and
`reasoning-effort-map:<requested>-to-<emitted>` tags.

:::warning Anthropic models require Anthropic-native egress
Determine this boundary from the selected upstream provider/model, not from the
client wire format. Claude Code may send Anthropic-shaped requests to
non-Anthropic alias candidates through supported adapters, but a selected
Anthropic/Claude model must use an Anthropic-native route and provider
credential.

Never route an Anthropic/Claude model through Codex/ChatGPT OAuth,
`chatgpt.com/backend-api/codex/responses`, an OpenAI/Codex adapter, or another
provider's transport. This includes alias fallback, retries, cooldown recovery,
probes, smoke tests, acceptance harnesses, and manual diagnostics. If the
Anthropic-native route is unavailable, fail closed instead of rerouting; treat
cross-provider Anthropic-model egress as a potential terms-of-service
violation.
:::

:::tip Sync Model Data
Keep your model pricing data up to date by [syncing models from GitHub](sync_models_github.md).
:::

<Tabs
  defaultValue="curl"
  values={[
    { label: 'cURL', value: 'curl', },
  ]}>
  <TabItem value="curl">

```bash
curl -X GET "http://0.0.0.0:4000/model/info" \
     -H "accept: application/json" \
```
  </TabItem>
</Tabs>

## Add a New Model

Add a new model to the proxy via the `/model/new` API, to add models without restarting the proxy.


:::info Database required
`/model/new` persists deployments to the LiteLLM proxy database. The proxy must start with `DATABASE_URL` set so Prisma initializes successfully, and `store_model_in_db` must be enabled. Config-file-only or DB-less proxy deployments should not use dynamic model-management writes; use `model_list` in your config instead.
:::

<Tabs>
<TabItem value="API">

```bash
curl -X POST "http://0.0.0.0:4000/model/new" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{ "model_name": "azure-gpt-turbo", "litellm_params": {"model": "azure/gpt-3.5-turbo", "api_key": "os.environ/AZURE_API_KEY", "api_base": "my-azure-api-base"} }'
```
</TabItem>
<TabItem value="Yaml">

```yaml
model_list:
  - model_name: gpt-3.5-turbo ### RECEIVED MODEL NAME ### `openai.chat.completions.create(model="gpt-3.5-turbo",...)`
    litellm_params: # all params accepted by litellm.completion() - https://github.com/BerriAI/litellm/blob/9b46ec05b02d36d6e4fb5c32321e51e7f56e4a6e/litellm/types/router.py#L297
      model: azure/gpt-turbo-small-eu ### MODEL NAME sent to `litellm.completion()` ###
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_EU" # does os.getenv("AZURE_API_KEY_EU")
      rpm: 6      # [OPTIONAL] Rate limit for this deployment: in requests per minute (rpm)
    model_info: 
      my_custom_key: my_custom_value # additional model metadata
```

</TabItem>
</Tabs>


### Model Parameters Structure

When adding a new model, your JSON payload should conform to the following structure:

- `model_name`: The name of the new model (required).
- `litellm_params`: A dictionary containing parameters specific to the Litellm setup (required).
- `model_info`: An optional dictionary to provide additional information about the model.

Here's an example of how to structure your `ModelParams`:

```json
{
  "model_name": "my_awesome_model",
  "litellm_params": {
    "some_parameter": "some_value",
    "another_parameter": "another_value"
  },
  "model_info": {
    "author": "Your Name",
    "version": "1.0",
    "description": "A brief description of the model."
  }
}
```
---

Keep in mind that as both endpoints are in [BETA], you may need to visit the associated GitHub issues linked in the API descriptions to check for updates or provide feedback:

- Get Model Information: [Issue #933](https://github.com/BerriAI/litellm/issues/933)
- Add a New Model: [Issue #964](https://github.com/BerriAI/litellm/issues/964)

Feedback on the beta endpoints is valuable and helps improve the API for all users.


## Add Additional Model Information 

If you want the ability to add a display name, description, and labels for models, just use `model_info:` 

```yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "gpt-4"
      api_key: "os.environ/OPENAI_API_KEY"
    model_info: # 👈 KEY CHANGE
      my_custom_key: "my_custom_value"
```

### Usage

1. Add additional information to model 

```yaml
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "gpt-4"
      api_key: "os.environ/OPENAI_API_KEY"
    model_info: # 👈 KEY CHANGE
      my_custom_key: "my_custom_value"
```

2. Call with `/model/info` 

Use a key with access to the model `gpt-4`.

```bash
curl -L -X GET 'http://0.0.0.0:4000/v1/model/info' \
-H 'Authorization: Bearer LITELLM_KEY' \
```

3. **Expected Response**

Returned `model_info = Your custom model_info + (if exists) LITELLM MODEL INFO`


[**How LiteLLM Model Info is found**](https://github.com/BerriAI/litellm/blob/9b46ec05b02d36d6e4fb5c32321e51e7f56e4a6e/litellm/proxy/proxy_server.py#L7460) 

[Tell us how this can be improved!](https://github.com/BerriAI/litellm/issues)

```bash
{
    "data": [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4"
            },
            "model_info": {
                "id": "e889baacd17f591cce4c63639275ba5e8dc60765d6c553e6ee5a504b19e50ddc",
                "db_model": false,
                "my_custom_key": "my_custom_value", # 👈 CUSTOM INFO
                "key": "gpt-4", # 👈 KEY in LiteLLM MODEL INFO/COST MAP - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
                "max_tokens": 4096,
                "max_input_tokens": 8192,
                "max_output_tokens": 4096,
                "input_cost_per_token": 3e-05,
                "input_cost_per_character": null,
                "input_cost_per_token_above_128k_tokens": null,
                "output_cost_per_token": 6e-05,
                "output_cost_per_character": null,
                "output_cost_per_token_above_128k_tokens": null,
                "output_cost_per_character_above_128k_tokens": null,
                "output_vector_size": null,
                "litellm_provider": "openai",
                "mode": "chat"
            }
        },
    ]
}
```

## Direct Anthropic routes and provider credentials

For **direct Anthropic** model routes (for example `anthropic/claude-sonnet-4-6` or
`/anthropic/v1/messages`), LiteLLM must send a **real Anthropic provider credential**
to the upstream API. Acceptable sources include:

- the deployment `api_key` in `model_list` / stored model params
- `litellm.anthropic_key` (or equivalent library setting)
- the `ANTHROPIC_API_KEY` environment variable
- a valid Anthropic OAuth bearer token (when configured for that route)

**LiteLLM proxy authentication is not a provider credential.** Headers such as
`Authorization: Bearer <litellm-proxy-key>` or `x-litellm-api-key` authenticate
the client to the LiteLLM gateway only. They must not be promoted or reused as
Anthropic `api_key` / `x-api-key` for upstream calls.

**BYOK `x-api-key` promotion** applies only on **direct Anthropic** routes when
provider-header forwarding is enabled. Use `forward_llm_provider_auth_headers: true`
in general settings, or model-group forwarding via `forward_client_headers_to_llm_api`
for `anthropic/*` in model group settings. In that case, a
client-supplied Anthropic `x-api-key` that was **not** used for LiteLLM proxy auth
may be promoted to `api_key` for the upstream request. Proxy-only keys and
non-Anthropic routes do not use this path.
