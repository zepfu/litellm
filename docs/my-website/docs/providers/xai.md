import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# xAI

https://docs.x.ai/docs

:::tip

**We support ALL xAI models, just set `model=xai/<any-model-on-xai>` as a prefix when sending litellm requests**

:::

## Supported Models



**Latest Release** - Grok 4.1 Fast: Optimized for high-performance agentic tool calling with 2M context and prompt caching.

| Model | Context | Features |
|-------|---------|----------|
| `xai/grok-4-1-fast-reasoning` | 2M tokens | **Reasoning**, Function calling, Vision, Audio, Web search, Caching |
| `xai/grok-4-1-fast-non-reasoning` | 2M tokens | Function calling, Vision, Audio, Web search, Caching |

**When to use:**
- ✅ **Reasoning model**: Complex analysis, planning, multi-step reasoning problems
- ✅ **Non-reasoning model**: Simple queries, faster responses, lower token usage

**Example:**
```python
from litellm import completion

# With reasoning
response = completion(
    model="xai/grok-4-1-fast-reasoning",
    messages=[{"role": "user", "content": "Analyze this problem step by step..."}]
)

# Without reasoning
response = completion(
    model="xai/grok-4-1-fast-non-reasoning",
    messages=[{"role": "user", "content": "What's 2+2?"}]
)
```

---

### All Available Models

| Model Family | Model | Context | Features |
|--------------|-------|---------|----------|
| **Grok 4.1** | `xai/grok-4-1-fast-reasoning` | 2M | **Reasoning**, Tools, Vision, Audio, Web search, Caching |
| | `xai/grok-4-1-fast-non-reasoning` | 2M | Tools, Vision, Audio, Web search, Caching |
| **Grok 4** | `xai/grok-4` | 256K | Tools, Web search |
| | `xai/grok-4-0709` | 256K | Tools, Web search |
| | `xai/grok-4.6` | 500K | **Reasoning**, Tools, Vision, Web search |
| | `xai/grok-4-fast-reasoning` | 2M | **Reasoning**, Tools, Web search |
| | `xai/grok-4-fast-non-reasoning` | 2M | Tools, Web search |
| **Grok 3** | `xai/grok-3` | 131K | Tools, Web search |
| | `xai/grok-3-mini` | 131K | Tools, Web search |
| | `xai/grok-3-fast-beta` | 131K | Tools, Web search |
| **Grok Code** | `xai/grok-code-fast` | 256K | **Reasoning**, Tools, Code generation, Caching |
| **Grok 2** | `xai/grok-2` | 131K | Tools, **Vision** |
| | `xai/grok-2-vision-latest` | 32K | Tools, **Vision** |

**Features:**
- **Reasoning** = Chain-of-thought reasoning with reasoning tokens
- **Tools** = Function calling / Tool use
- **Web search** = Live internet search
- **Vision** = Image understanding
- **Audio** = Audio input support
- **Caching** = Prompt caching for cost savings
- **Code generation** = Optimized for code tasks

**Pricing:** See [xAI's pricing page](https://docs.x.ai/docs/models) for current rates.

## API Key
```python
# env variable
os.environ['XAI_API_KEY']
```

## Sample Usage

```python showLineNumbers title="LiteLLM python sdk usage - Non-streaming"
from litellm import completion
import os

os.environ['XAI_API_KEY'] = ""
response = completion(
    model="xai/grok-3-mini-beta",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)
print(response)
```

## Sample Usage - Streaming

```python showLineNumbers title="LiteLLM python sdk usage - Streaming"
from litellm import completion
import os

os.environ['XAI_API_KEY'] = ""
response = completion(
    model="xai/grok-3-mini-beta",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    stream=True,
    max_tokens=10,
    response_format={ "type": "json_object" },
    seed=123,
    stop=["\n\n"],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)

for chunk in response:
    print(chunk)
```

## Responses API Metadata

xAI documents top-level `metadata` on the Responses API as unsupported and
compatibility-only. LiteLLM strict mode rejects this parameter. To
permissively adapt the request, set `litellm.drop_params=True`; LiteLLM then
drops the unsupported field before sending the request.

When LiteLLM reports this adaptation, diagnostics identify the parameter
without including caller-provided metadata values. LiteLLM's internal
observability, routing, authentication, and session metadata remain in the
separate `litellm_metadata` structure and are never merged into caller
top-level `metadata`.

## Rate-limit handling

For an xAI provider `429`, LiteLLM uses a valid `Retry-After` value first.
Otherwise it uses the request or token reset header for the exhausted
dimension. If the response does not identify a usable dimension-specific
reset, `x-ratelimit-reset` and `x-rate-limit-reset` are bounded generic
fallbacks. When the response does not identify the exhausted dimension and
both dimension-specific values are valid, LiteLLM waits for the later reset.

Retry values may be bounded durations, timestamps, ISO timestamps, or
HTTP-date values. Malformed, expired, non-finite, and unreasonably future
values are ignored instead of creating a durable cooldown.

Native Grok OIDC and managed xAI OAuth headers are recorded as separate
observation sources. When xAI supplies quota values without reset evidence,
the observation leaves the reset time and quota period unknown; LiteLLM does
not estimate a monthly boundary. Explicit provider reset or billing-period
evidence is retained.

## Responses API Compatibility

xAI's Responses API does not accept OpenAI's top-level `instructions` field.
For supported LiteLLM proxy passthrough routes, LiteLLM preserves caller
instructions by lowering them into one `system` message in `input` before the
request is sent to xAI. The outbound request has no top-level `instructions`,
and the same behavior applies across fresh requests, continuations, retries,
and proxy redispatch. For direct SDK calls, put system guidance in the
`input` message sequence yourself; strict mode rejects the unsupported
parameter and permissive mode drops it.

For supported Codex auto-agent proxy routes, LiteLLM converts configured custom
and namespace tools into xAI-compatible function tools before egress. The route
applies required tool-description patches, removes unsupported hosted/request
fields, and restores the original tool and namespace identities in returned tool
calls. Send the original tool definitions; do not pre-flatten namespace tools.

### Image Inputs

When a Responses request contains an image item, image URL, or base64 image
data, LiteLLM forces `store=False` before sending it to xAI. This applies even
if the caller sets `store=True`. Requests without image input retain the
caller's `store` value or the provider default.

```python showLineNumbers title="xAI Responses with image input"
import litellm

response = litellm.responses(
    model="xai/grok-4.6",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Answer concisely. What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/image.png",
                },
            ],
        }
    ],
    store=True,  # LiteLLM sends store=False because the input contains an image.
)
```

## Proxy Retry and Quota Behavior

For proxy routes that use Grok, LiteLLM recognizes exact account-quota
responses and classifies them as `usage_limit_reached`:

- HTTP 402 with the xAI/Grok usage-balance exhaustion response.
- HTTP 403 with the recognized personal/team spending-limit response.

The proxy publishes the Grok account-quota lane cooldown once and proceeds
through the replay-safe fallback policy without sleeping and retrying the
exhausted account. Fresh requests can advance immediately; continuations keep
their account/session-affinity safety rule. Generic 403 responses, HTTP 429,
and transient 5xx responses retain their ordinary classifications and retry
behavior.

### Native Grok Continuations

Native `xai/grok-4.6` has an explicit
`native_grok_continuation_retry` capability. The managed OAuth
`oa_xai/grok-4.6` and Cursor
`cursor_agent/cursor-grok-4.6-high` lanes use separate route and credential
families and do not inherit native continuation recovery. Future native models
must declare this capability before they are eligible; unknown or unprofiled
models fail closed.

For a continuation with eligible `upstream_transient_internal` failure, the
native route retries the same candidate with a request-scoped total-attempt
budget of 8 by default. Set
`AAWM_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS` to tune the budget;
LiteLLM clamps it to 6-16. Retries use short exponential backoff with bounded
jitter, capped near one second, and the delay occurs outside routing locks.
There is no generic fixed ten-second sleep. Fresh requests and non-continuation
failures retain the generic proxy retry policy.

### Managed OAuth Credential Rotation

For a provider-returned managed `oa_xai/*` HTTP `401` before response bytes are
committed, LiteLLM can reread the configured credential and make one retry on
alias routes, direct async SDK routes, and OpenAI passthrough. The reread must
produce a new credential generation with the same derived non-secret account
identity. Client IDs and OAuth scope are not account proof by themselves.
An unchanged, expired, malformed, unproven-account, or different-account
credential is not retried, and native `xai/*` Grok OIDC traffic does not use
this managed OAuth recovery.

## Sample Usage - Vision

```python showLineNumbers title="LiteLLM python sdk usage - Vision"
import os 
from litellm import completion

os.environ["XAI_API_KEY"] = "your-api-key"

response = completion(
    model="xai/grok-2-vision-latest",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
            ],
        },
    ],
)
```

## Usage with LiteLLM Proxy Server

Here's how to call a XAI model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml showLineNumbers
  model_list:
    - model_name: my-model
      litellm_params:
        model: xai/<your-model-name>  # add xai/ prefix to route as XAI provider
        api_key: api-key                 # api key to send your model
  ```


2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python showLineNumbers
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="my-model",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>


## Reasoning Usage

LiteLLM supports reasoning usage for xAI models.

<Tabs>

<TabItem value="python" label="LiteLLM Python SDK">

```python showLineNumbers title="reasoning with xai/grok-3-mini-beta"
import litellm
response = litellm.completion(
    model="xai/grok-3-mini-beta",
    messages=[{"role": "user", "content": "What is 101*3?"}],
    reasoning_effort="low",
)

print("Reasoning Content:")
print(response.choices[0].message.reasoning_content)

print("\nFinal Response:")
print(completion.choices[0].message.content)

print("\nNumber of completion tokens (input):")
print(completion.usage.completion_tokens)

print("\nNumber of reasoning tokens (input):")
print(completion.usage.completion_tokens_details.reasoning_tokens)
```
</TabItem>

<TabItem value="curl" label="LiteLLM Proxy - OpenAI SDK Usage">

```python showLineNumbers title="reasoning with xai/grok-3-mini-beta"
import openai
client = openai.OpenAI(
    api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000" # litellm-proxy-base url
)

response = client.chat.completions.create(
    model="xai/grok-3-mini-beta",
    messages=[{"role": "user", "content": "What is 101*3?"}],
    reasoning_effort="low",
)

print("Reasoning Content:")
print(response.choices[0].message.reasoning_content)

print("\nFinal Response:")
print(completion.choices[0].message.content)

print("\nNumber of completion tokens (input):")
print(completion.usage.completion_tokens)

print("\nNumber of reasoning tokens (input):")
print(completion.usage.completion_tokens_details.reasoning_tokens)
```

</TabItem>
</Tabs>

**Example Response:**

```shell
Reasoning Content:
Let me calculate 101 multiplied by 3:
101 * 3 = 303.
I can double-check that: 100 * 3 is 300, and 1 * 3 is 3, so 300 + 3 = 303. Yes, that's correct.

Final Response:
The result of 101 multiplied by 3 is 303.

Number of completion tokens (input):
14

Number of reasoning tokens (input):
310
```
