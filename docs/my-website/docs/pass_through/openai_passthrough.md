# OpenAI Passthrough

Pass-through endpoints for direct OpenAI API access

## Overview

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ❌ | Not supported |
| Logging | ✅ | Works across all integrations |
| Streaming | ✅ | Fully supported |

## Available Endpoints

### `/openai_passthrough` - Recommended
Dedicated passthrough endpoint that guarantees direct routing to OpenAI without conflicts.

**Use this for:**
- OpenAI Responses API (`/v1/responses`)
- Any endpoint where you need guaranteed passthrough
- When `/openai` routes are conflicting with LiteLLM's native implementations

### `/openai` - Legacy
Standard passthrough endpoint that may conflict with LiteLLM's native implementations.

**Note:** Some endpoints like `/openai/v1/responses` will be routed to LiteLLM's native implementation instead of OpenAI.

## Responses streaming lifecycle

Native OpenAI Responses streaming requires an authoritative terminal event:
`response.completed`, `response.failed`, or `response.incomplete`. LiteLLM
forwards the first valid terminal, emits exactly one `data: [DONE]`, and
discards incomplete trailing frames. A stream that ends without a valid
terminal is completed on the wire as `response.incomplete`; it is not treated
as a successful response.

For managed alias routing, the session-owner reservation and legacy session
affinity remain request-scoped while the stream is active. A completed terminal
can promote the owner and commit affinity. Failed, incomplete, cancelled, or
disconnected streams release the reservation and do not write new affinity.

The final delivered disposition is also published for logging, rollup, and
session-transfer consumers. Consumers must use this disposition rather than
the upstream HTTP status: `completed` is the only successful outcome;
`failed`, `incomplete`, `cancelled`, and `disconnected` are non-success
outcomes. The disposition is selected once, after the terminal event and
`[DONE]` have been delivered for streams or after the response body has been
sent for non-streaming Responses.

The internal callback contract is
`on_disposition(disposition, trace)`. Consumers should read a copy from
`trace.snapshot()` containing the wire state, response/body commitment,
terminal and `[DONE]` markers, final disposition, and bounded duplicate,
partial-frame, and close-error fields. Detailed malformed-frame metadata
remains on `trace.metadata`; a malformed frame is converted to synthetic
`response.failed` before any later success can be selected. Consumers must
not promote ownership, write legacy affinity, or infer success independently
from the provider response.

## When to use this?

- For 90% of your use cases, you should use the [native LiteLLM OpenAI Integration](https://docs.litellm.ai/docs/providers/openai) (`/chat/completions`, `/embeddings`, `/completions`, `/images`, `/batches`, etc.)
- Use `/openai_passthrough` to call less popular or newer OpenAI endpoints that LiteLLM doesn't fully support yet, such as `/assistants`, `/threads`, `/vector_stores`, `/responses`

Simply replace `https://api.openai.com` with `LITELLM_PROXY_BASE_URL/openai_passthrough`

## Usage Examples

Requirements:
Set `OPENAI_API_KEY` in your environment variables.

> **Maintainer note:** Codex-native authenticated `GET /models` and `GET /v1/models` requests preserve inbound OpenAI auth and use `CHATGPT_API_BASE`. Generic or non-Codex requests retain the server-side `OPENAI_API_KEY` behavior.

### Assistants API

#### Create OpenAI Client

Make sure you do the following:
- Point `base_url` to your `LITELLM_PROXY_BASE_URL/openai`
- Use your `LITELLM_API_KEY` as the `api_key`

```python
import openai

client = openai.OpenAI(
    base_url="http://0.0.0.0:4000/openai_passthrough",  # <your-proxy-url>/openai_passthrough
    api_key="sk-anything"  # <your-proxy-api-key>
)
```

#### Create an Assistant

```python
# Create an assistant
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a math tutor. Help solve equations.",
    model="gpt-4o",
)
```

#### Create a Thread
```python
# Create a thread
thread = client.beta.threads.create()
```

#### Add a Message to the Thread
```python
# Add a message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Solve 3x + 11 = 14",
)
```

#### Run the Assistant
```python
# Create a run to get the assistant's response
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Check run status
run_status = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
)
```

#### Retrieve Messages
```python
# List messages after the run completes
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)
```

#### Delete the Assistant

```python
# Delete the assistant when done
client.beta.assistants.delete(assistant.id)
```

## Text watermark policy (optional)

`/openai_passthrough` can inspect visible Responses and Chat Completions text
for deterministic Unicode carriers (zero-width/format controls, noncharacters,
and configured exotic spaces). This is not a vendor-watermark detector and
does not rewrite prose with a second model.

Shipped default is off:

```yaml
general_settings:
  openai_passthrough_text_watermark:
    mode: off
    unicode:
      enabled: true
      policy: conservative
    removal:
      enabled: false
      stream_policy: audit_only
    statistical_detectors: []
```

Leave this disabled unless you explicitly opt in. `sanitize` / `enforce`
require `removal.enabled: true`. `enforce` plus streamed output also requires
`stream_policy: buffer_response`. Statistical detectors stay an empty disabled
registry; they report `unsupported` / `inconclusive` and never load
torch/transformers.
