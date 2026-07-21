import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Moonshot AI

## Overview

| Property | Details |
|-------|-------|
| Description | Moonshot AI provides large language models including the moonshot-v1 series and kimi models. |
| Provider Route on LiteLLM | `moonshot/` |
| Link to Provider Doc | [Moonshot AI ↗](https://platform.moonshot.ai/) |
| Base URL | `https://api.moonshot.ai/` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://platform.moonshot.ai/

**We support ALL Moonshot AI models, just set `moonshot/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["MOONSHOT_API_KEY"] = ""  # your Moonshot AI API key
```

**ATTENTION:**

Moonshot AI offers two distinct API endpoints: a global one and a China-specific one.
- Global API Base URL: `https://api.moonshot.ai/v1` (This is the one currently implemented)
- China API Base URL: `https://api.moonshot.cn/v1`

You can overwrite the base url with:

```
os.environ["MOONSHOT_API_BASE"] = "https://api.moonshot.cn/v1"
```

## Managed Kimi Code OAuth (AAWM)

This is separate from the paid Moonshot API-key integration above. The
`moonshot/` route continues to use `MOONSHOT_API_KEY`; it is not a substitute
for the managed OAuth-backed `kimi_code` provider.

Managed Kimi Code accepts only these exact model IDs: `k3`,
`kimi-for-coding`, and `kimi-for-coding-highspeed`. The authenticated Kimi
Code `/models` response is the capability authority, including whether a
requested reasoning effort is supported. Missing or unrecognized capability
data does not justify inventing output ceilings, translating to another model,
or silently substituting an effort.

AAWM may use internal `k3-low`, `k3-high`, and `k3-max` labels while resolving
capabilities and public reference pricing. Those labels are observability and
routing identities only; they are normalized to `k3` for introspection and are
rejected if they reach provider request transformation.

Kimi subscription reference costs are not invoice spend. Callback observers
may use the deterministic reference total as generation cost only when they
also retain `actual_invoice_cost_known=false` and the source/model/kind
provenance documented in `docs/aawm-session-history.md`.

The raw Kimi CLI compatibility gateway is the trusted-local `/kimi/v1` route.
It accepts only a direct `127.0.0.1` or `::1` peer; forwarded identity headers
are ignored. Incoming authorization and forwarded/provider-specific identity
headers are stripped, and the gateway supplies the managed Kimi credential
instead of relaying caller authorization.

Codex and Anthropic adapter continuations preserve assistant tool calls and
their paired tool results. When replay contains an assistant tool-call message
whose text content is blank, the adapter omits the `content` field instead of
sending `null` or an empty value that Kimi rejects. A remaining Kimi 400/422
request-shape rejection is returned once as a bounded client error; it is not
retried as provider capacity and must not surface as HTTP 500 or a raw
traceback.

Codex collaboration may transport a `spawn_agent` or follow-up assignment as an
empty `NEW_TASK`/`MESSAGE` payload wrapper plus a dedicated
`encrypted_content` part. The Kimi Responses adapter restores that task text
only for the recognized empty collaboration envelope before translating it to
chat completions. Ordinary encrypted reasoning and continuation state is never
treated as task text or sent to Kimi as plaintext.

Kimi Code `0.27.0` cannot use a custom `KIMI_CODE_BASE_URL` while retaining
the official default credential slot. It derives a separate environment-scoped
OAuth slot for every custom base URL. Do not copy or symlink
`~/.kimi-code/credentials/kimi-code.json`, and do not enroll a second grant to
make the CLI use this gateway. Until the client supports an explicit
same-credential custom-base contract, keep the direct `kimi` command on the
official endpoint. The gateway remains available to trusted loopback clients
that explicitly present the current managed bearer.

For the shared credential and native lock contract, see
`docs/aawm-oauth-credential-maintenance.md` in the repository. Native quota
observation semantics are maintained in
`docs/aawm-provider-status-observations.md`; subscription reference-cost and
alias cooldown provenance are maintained in `docs/aawm-session-history.md`.

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Moonshot Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["MOONSHOT_API_KEY"] = ""  # your Moonshot AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Moonshot call
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Moonshot Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["MOONSHOT_API_KEY"] = ""  # your Moonshot AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Moonshot call with streaming
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: moonshot-v1-8k
    litellm_params:
      model: moonshot/moonshot-v1-8k
      api_key: os.environ/MOONSHOT_API_KEY

  - model_name: moonshot-v1-32k
    litellm_params:
      model: moonshot/moonshot-v1-32k
      api_key: os.environ/MOONSHOT_API_KEY

  - model_name: moonshot-v1-128k
    litellm_params:
      model: moonshot/moonshot-v1-128k
      api_key: os.environ/MOONSHOT_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Moonshot via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Moonshot via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Moonshot via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Moonshot via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key",
    stream=True
)

for chunk in response:
    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Moonshot via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

```bash showLineNumbers title="Moonshot via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "hello from litellm"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Image / Vision Support

Moonshot vision models (`kimi-k2.5`, `kimi-latest`, `moonshot-v1-*-vision-preview`, etc.) accept the standard OpenAI content array with `image_url` blocks.

LiteLLM automatically detects when your messages contain images and preserves the content array so the image payload reaches the Moonshot API. For text-only requests the content is flattened to a plain string, as required by Moonshot text models.

```python showLineNumbers title="Moonshot Vision Example"
import os
import litellm

os.environ["MOONSHOT_API_KEY"] = ""

response = litellm.completion(
    model="moonshot/kimi-k2.5",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

## Moonshot AI Limitations & LiteLLM Handling

LiteLLM automatically handles the following [Moonshot AI limitations](https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility) to provide seamless OpenAI compatibility:

### Temperature Range Limitation
**Limitation**: Moonshot AI only supports temperature range [0, 1] (vs OpenAI's [0, 2])  
**LiteLLM Handling**: Automatically clamps any temperature > 1 to 1

### Temperature + Multiple Outputs Limitation  
**Limitation**: If temperature < 0.3 and n > 1, Moonshot AI raises an exception  
**LiteLLM Handling**: Automatically sets temperature to 0.3 when this condition is detected

### Tool Choice "Required" Not Supported
**Limitation**: Moonshot AI doesn't support `tool_choice="required"`  
**LiteLLM Handling**: Converts this by:
- Adding message: "Please select a tool to handle the current issue."
- Removing the `tool_choice` parameter from the request
