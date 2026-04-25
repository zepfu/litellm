# AAWM/LiteLLM OpenRouter Embedding and Rerank

Use the AAWM LiteLLM proxy surface for OpenRouter-hosted embeddings and rerank. Consumers authenticate to AAWM/LiteLLM with the LiteLLM virtual key issued to them; do not send OpenRouter API keys from clients.

## Base URL and Auth

Replace `https://litellm.example.com` with the AAWM LiteLLM proxy URL.

```bash
export LITELLM_BASE_URL="https://litellm.example.com"
export LITELLM_API_KEY="sk-..."
```

Every request must include:

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
- `langfuse_trace_name`: human-readable trace name, for example `search-embedding` or `document-rerank`.

## Embeddings

Endpoint:

- `POST /v1/embeddings`

Model:

- `openrouter/qwen/qwen3-embedding-8b`

Pin DeepInfra as the OpenRouter provider with the `provider` object. OpenRouter receives the upstream provider routing instruction through LiteLLM; clients still authenticate only to LiteLLM.

```bash
curl -sS "$LITELLM_BASE_URL/v1/embeddings" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-litellm-session-id: sess_123" \
  -H "x-aawm-tenant-id: tenant_abc" \
  -H "x-litellm-end-user-id: user_456" \
  -H "x-litellm-trace-id: trace_789" \
  -H "x-litellm-agent-id: retrieval-api" \
  -H "langfuse_trace_user_id: user_456" \
  -H "langfuse_trace_name: search-embedding" \
  -d '{
    "model": "openrouter/qwen/qwen3-embedding-8b",
    "input": [
      "AAWM routes embedding requests through LiteLLM.",
      "DeepInfra is pinned as the OpenRouter provider."
    ],
    "provider": {
      "order": ["DeepInfra"],
      "allow_fallbacks": false
    }
  }'
```

## Rerank

Endpoints:

- `POST /rerank`
- `POST /v1/rerank`

Model:

- `openrouter/cohere/rerank-4-pro`

`/rerank` is the preferred LiteLLM rerank route. The proxy also registers `/v1/rerank` for clients that use Cohere-style v1 routing.

```bash
curl -sS "$LITELLM_BASE_URL/rerank" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-litellm-session-id: sess_123" \
  -H "x-aawm-tenant-id: tenant_abc" \
  -H "x-litellm-end-user-id: user_456" \
  -H "x-litellm-trace-id: trace_790" \
  -H "x-litellm-agent-id: retrieval-api" \
  -H "langfuse_trace_user_id: user_456" \
  -H "langfuse_trace_name: document-rerank" \
  -d '{
    "model": "openrouter/cohere/rerank-4-pro",
    "query": "How do consumers call OpenRouter-hosted rerank through AAWM LiteLLM?",
    "documents": [
      "Consumers call OpenRouter directly with an OpenRouter key.",
      "Consumers call the AAWM LiteLLM proxy with their LiteLLM Authorization key.",
      "Session history rows are created only when x-litellm-session-id is present."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

The same payload can be sent to `/v1/rerank`:

```bash
curl -sS "$LITELLM_BASE_URL/v1/rerank" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-litellm-session-id: sess_123" \
  -d '{
    "model": "openrouter/cohere/rerank-4-pro",
    "query": "What key should clients use?",
    "documents": [
      "Use the AAWM LiteLLM Authorization key.",
      "Use an OpenRouter API key from the client."
    ],
    "top_n": 1
  }'
```

## Operational Notes

- Keep the same `x-litellm-session-id` across related embedding, retrieval, rerank, and generation calls when they belong to one user workflow.
- Use unique `x-litellm-trace-id` values per request or logical trace segment when your application already has trace ids.
- Do not place provider API keys in browser, mobile, or consumer-service requests. AAWM/LiteLLM owns upstream OpenRouter authentication and routing.
