# AAWM/LiteLLM Local LLM

Use the AAWM LiteLLM proxy surface for locally hosted LLM chat completions. Consumers authenticate to AAWM/LiteLLM with the LiteLLM virtual key issued to them; do not call the local llama.cpp service port directly from clients.

## Base URL and Auth

For local development, use the dev proxy:

```bash
export LITELLM_BASE_URL="http://127.0.0.1:4001"
export LITELLM_API_KEY="sk-..."
```

For deployed environments, replace `http://127.0.0.1:4001` with the AAWM LiteLLM proxy URL.

Every request should include:

```http
Authorization: Bearer $LITELLM_API_KEY
Content-Type: application/json
```

## Attribution Headers

Send attribution headers consistently so usage, Langfuse traces, and AAWM session history can be joined.

Required:

- `x-litellm-session-id`: stable session/conversation id. `session_history` rows require this value; if it is omitted, the request can still succeed but no session history row is created.

Recommended:

- `x-aawm-tenant-id`: tenant/account id for AAWM attribution.
- `x-litellm-end-user-id`: end-user id for LiteLLM spend and policy attribution.
- `x-litellm-trace-id`: caller trace/correlation id.
- `x-litellm-agent-id`: agent or application id.
- `langfuse_trace_user_id`: Langfuse trace user id, usually the same logical user as `x-litellm-end-user-id`.
- `langfuse_trace_name`: human-readable trace name, for example `local-llm-chat`.

## Available Model

Consumer-facing model names are LiteLLM aliases. The backing model and service port are operational details owned by AAWM/LiteLLM.

| Mode | LiteLLM model | Backing model | Local service | Notes |
|---|---|---|---|---|
| Chat | `qwen3-heretic-gguf` | `qwen3-4b-heretic-q8` | `qwen3-heretic-gguf` on `8093` | Local OpenAI-compatible llama.cpp chat/completion model |

This is a pure LLM route. Do not use `qwen3-heretic-gguf` with `/v1/embeddings`, `/rerank`, or `/v1/rerank`.

## Chat Completions

Endpoint:

- `POST /v1/chat/completions`

Model:

- `qwen3-heretic-gguf`

```bash
curl -sS "$LITELLM_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-litellm-session-id: sess_123" \
  -H "x-aawm-tenant-id: tenant_abc" \
  -H "x-litellm-end-user-id: user_456" \
  -H "x-litellm-trace-id: trace_789" \
  -H "x-litellm-agent-id: local-chat-client" \
  -H "langfuse_trace_user_id: user_456" \
  -H "langfuse_trace_name: local-llm-chat" \
  -d '{
    "model": "qwen3-heretic-gguf",
    "messages": [
      {
        "role": "system",
        "content": "Answer concisely."
      },
      {
        "role": "user",
        "content": "Explain what the local LLM route is for."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 256
  }'
```

## OpenAI SDK

Use the normal OpenAI SDK chat-completions client pointed at the LiteLLM proxy.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=f"{os.environ['LITELLM_BASE_URL']}/v1",
)

response = client.chat.completions.create(
    model="qwen3-heretic-gguf",
    messages=[
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Explain what the local LLM route is for."},
    ],
    temperature=0.2,
    max_tokens=256,
    extra_headers={
        "x-litellm-session-id": "sess_123",
        "x-aawm-tenant-id": "tenant_abc",
        "x-litellm-end-user-id": "user_456",
        "x-litellm-trace-id": "trace_789",
        "x-litellm-agent-id": "local-chat-client",
        "langfuse_trace_user_id": "user_456",
        "langfuse_trace_name": "local-llm-chat",
    },
)

print(response.choices[0].message.content)
```

## Operational Notes

- Keep the same `x-litellm-session-id` across related embedding, retrieval, rerank, and generation calls when they belong to one user workflow.
- Use unique `x-litellm-trace-id` values per request or logical trace segment when your application already has trace ids.
- Do not place local service ports in browser, mobile, or consumer-service requests. AAWM/LiteLLM owns local model routing and cost attribution.
- `session_history` records for this route should use the LiteLLM-facing model `qwen3-heretic-gguf`; the upstream local service model is `qwen3-4b-heretic-q8`.
- If a workflow needs local embeddings or rerank, use the separate models in [LOCAL_EMBED_RERANK_CONSUMER.md](LOCAL_EMBED_RERANK_CONSUMER.md).
