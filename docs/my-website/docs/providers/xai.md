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

## Native Grok OIDC request snapshots

Native Grok routes use immutable, validated snapshots for the OIDC credential
and the installed client-version cache. File metadata checks, reads, and JSON
validation run off the request event loop, with one in-flight validation per
credential or version path. Atomic file replacement invalidates the matching
snapshot on the next request; missing, malformed, ambiguous-scope, expired,
or near-expiry records fail closed. LiteLLM does not refresh or write native
credential files during request handling.

## Managed OAuth Credential Snapshots

Managed `oa_xai/*` request preparation uses an immutable, generation-aware
snapshot of the configured credential file and exact scope. Configuration
resolution, file metadata checks, reads, and JSON parsing run off the request
event loop, and concurrent requests share one in-flight validation for the
same generation. Atomic replacement or route-safety expiry invalidates the
cached snapshot before a later request rebuilds it. Missing, malformed,
ambiguous-scope, expired, and near-expiry records fail closed; request handling
does not refresh or write the managed credential file.

For a provider-owned managed xAI `401` before response bytes are committed,
LiteLLM may reread the exact bound file and scope and retry once on alias,
direct async, OpenAI passthrough, or Anthropic compatibility routes. The
reread must produce a changed trusted generation with the same non-secret
account identity, and the retry reuses the original request body. Unchanged
generations, a second `401`, missing or unproven account evidence, and
different-account material are not retried. Native `xai/*` Grok OIDC traffic
does not use this managed OAuth recovery.

## Rate-limit handling

For an xAI provider `429`, LiteLLM uses a valid `Retry-After` value first.
Otherwise it uses the request or token reset header for the exhausted
dimension. If the response does not identify the exhausted dimension and both
dimension-specific values are valid, LiteLLM waits for the later reset.
Bounded generic reset headers are used only when no dimension-specific reset
is available.

Reset values may be bounded durations, epoch timestamps, ISO timestamps, or
HTTP-date values. Malformed, expired, non-finite, and unreasonably future
values are ignored instead of creating a durable cooldown.

Native Grok OIDC and managed xAI OAuth responses keep separate rate-limit
observation identities. Native headers are captured under
`xai_grok_oidc_response_headers` and labeled with the `xai_grok_oidc`
credential family and `grok-build` client family; managed headers use
`xai_oauth_response_headers` and the `xai_oauth` client family. A legacy
native observation under the managed key is read only when native metadata
proves ownership, and authorization or unrelated headers are excluded.

When xAI supplies quota limit or remaining values without provider reset or
billing-period evidence, LiteLLM leaves the reset time and quota period
unknown. It does not synthesize a monthly boundary that could drive cooldown,
availability, rollover, or forecasting. Explicit provider reset and billing
period evidence remains authoritative.

## OAuth Credential Scope Selection

Managed xAI OAuth and native Grok OIDC credential files must contain the exact
configured scope when they contain multiple records. LiteLLM rejects a missing
scope before provider or token I/O and never selects a record based on JSON key
order. An explicitly unambiguous legacy flat record remains supported; mixed
flat-and-nested documents must be migrated to an exact scope-keyed record.

Managed `oa_xai/*` request and provider-status paths share one file/scope
resolver. Auth-file precedence is `AAWM_XAI_OAUTH_AUTH_FILE`, an explicit
configured path, `LITELLM_XAI_OAUTH_AUTH_FILE`,
`LITELLM_XAI_OAUTH_MIGRATED_AUTH_FILE`, then the portable default. Scope
precedence is an explicit configured scope, `AAWM_XAI_OAUTH_SCOPE`,
`LITELLM_XAI_OAUTH_SCOPE`, then the default Grok subscription scope. Conflicting
configured values fail closed. Resolution metadata uses a nonsecret
`credential_identity` derived from the canonical file target and exact scope;
credential contents and raw paths are never included in that identity.
Refresh and passive-health metadata also carries a separate nonsecret
`credential_generation` digest for the published file generation; token values
are excluded from both identifiers.

Managed credential lifecycle is evaluated by one side-effect-free policy shared
by request readiness, refresh eligibility, and passive health. It reports
structural validity, access availability, expiry availability, refresh
possibility, refresh due state, route usability, and terminal unrefreshability.
The route-safety buffer controls whether an access token may be sent; the
proactive refresh threshold is a separate value derived from issued lifetime
and the configured minimum. A credential can therefore be `refresh_due` while
remaining `route_usable`, and refresh-only, access-only, malformed-expiry, and
expired records receive distinct lifecycle states. Missing or malformed expiry
never becomes permanently fresh.

`AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS` controls writer-side proactive refresh.
`LITELLM_XAI_OAUTH_REFRESH_BUFFER_SECONDS` controls the route-safety deadline
used by request handling and sidecar health/eligibility. These settings are
independent; an omitted writer buffer retains the compatibility fallback to
the consumer setting and then the 300-second default.

Managed xAI refreshes derive their default advisory lock from the canonical
resolved auth file, using the file's `.lock` sibling. Different custom auth
files use independent locks, while aliases for one file coordinate on one
lock. Set `AAWM_XAI_OAUTH_LOCK_FILE` or `--xai-oauth-lock-file` only to an
alias of that canonical sibling; arbitrary paths, lock symlinks, and auth-file
lock collisions fail closed.

## Proxy Retry and Quota Behavior

For proxy routes that use Grok, LiteLLM treats these exact upstream responses as
account quota exhaustion:

- HTTP 402 with the xAI/Grok usage-balance exhaustion response.
- HTTP 403 with the recognized personal/team spending-limit response.

The proxy publishes the Grok account-quota lane cooldown once, advances through
the replay-safe fallback policy without sleeping, and does not retry the
exhausted account. Fresh requests can advance immediately; continuations retain
their account/session-affinity safety rule. Generic 403 responses, HTTP 429,
and transient 5xx responses keep their existing classifications and retry
behavior.

## Responses API Image Retention

For xAI Responses requests containing `input_image` or `image_url` content,
LiteLLM sends `store=false` automatically. This applies to URL and base64
image forms on both native xAI and managed xAI OAuth routes. Text-only
requests preserve the caller's explicit `store` value or the provider default.

## Responses API Tool Compatibility

For supported Codex auto-agent native Grok routes, LiteLLM converts mixed
custom and namespace tools into xAI-compatible function tools before egress.
It applies the required description patches, removes unsupported hosted tools,
request fields, input items, and empty tool choices, then restores the
original custom and namespace identities in both streaming and non-streaming
responses. Name collisions use the established deterministic policy; retries
start from the caller's original tool definitions, so conversions are not
applied twice.

For stock `collaboration.wait_agent` and its supported namespace/tool aliases,
native Grok and managed xAI Codex routes narrow `timeout_ms` from `number` to
`integer` in derived build/restoration schemas to match the client's integer
parser. Finite integral response values become JSON integers in both JSON and
SSE output. Fractional values, unrelated numeric arguments, original tool names,
and the caller's replay definitions remain unchanged.

Native Grok and managed xAI Codex routes repair supported literal tool-call
text in JSON and fully buffered Responses output. Context notes explicitly
marked as historical, non-executable data are never converted into calls;
they remain unchanged for malformed-output validation. Once a stream is being
forwarded lazily, malformed tool-call text fails closed before its successful
terminal event and deferred session-owner promotion. LiteLLM does not replay
already forwarded text as executable calls or buffer an unbounded response
to repair it.

## Responses API Instructions

xAI Responses does not accept OpenAI's top-level `instructions` field. On
supported LiteLLM proxy routes, caller instructions and configured alias
guidance are lowered into one ordered `system` message in `input` before
egress. The outbound request omits top-level `instructions`, and repeated
preparation during continuations, retries, or redispatch does not duplicate the
message. Internal metadata and credentials remain outside prompt input.

## Native Grok Route Capabilities

Native `xai/grok-4.5` and `xai/grok-4.6` declare the
`native_grok_continuation_retry` capability in the canonical xAI model
metadata. Native recovery and cooldown handling require both this explicit
capability and a native Grok route family. Managed `oa_xai/*`, Cursor, Composer,
Grok Build, and unprofiled future models do not inherit the native policy.
Malformed native output remains request-local and does not create a durable
candidate cooldown.

Native `xai/grok-4.6` also declares `native_responses_tool_history`. Its native
OIDC request projection preserves typed `function_call` and
`function_call_output` pairs, their order, names, and `call_id` correlation.
Arguments remain JSON strings; object arguments and outputs are serialized
without replacing their contents. Output-only item IDs and status fields are
omitted. Canonical replay remains unchanged. This replaces synthesized
assistant history notes only on this capability-enabled native route;
managed OAuth and other models retain their existing compatibility policy.

## Native Grok Continuation Recovery

For a provider-owned continuation on a capability-enabled native Grok route,
bare `upstream_transient_internal` failures use the native request-scoped
continuation budget. The planner runs before generic pre-commit retry handling,
so transient recovery uses bounded short backoff instead of the generic
ten-second wait. Five transient failures can therefore be followed by a sixth
provider attempt, while exhaustion terminates at the configured request budget.

Fresh requests, managed `oa_xai/*` routes, and non-transient failures retain
their existing generic retry and cooldown policy.

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
