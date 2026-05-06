# AAWM/LiteLLM Local Embedding and Rerank

Use the AAWM LiteLLM proxy surface for locally hosted embeddings and rerank. Consumers authenticate to AAWM/LiteLLM with the LiteLLM virtual key issued to them; do not call the local TEI, llama.cpp, or reranker service ports directly from clients.

For the local chat/completion LLM route, see [LOCAL_LLM_CONSUMER.md](LOCAL_LLM_CONSUMER.md).

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
- `langfuse_trace_name`: human-readable trace name, for example `local-code-embedding` or `local-document-rerank`.

## Available Models

Consumer-facing model names are LiteLLM aliases. The backing model and service ports are operational details owned by AAWM/LiteLLM.

| Mode | LiteLLM model | Backing model | Local service | Estimated input cost |
|---|---|---|---|---:|
| Embedding | `tei-medcpt-article` | `ncbi/MedCPT-Article-Encoder` | `tei-medcpt-article` on `8083` | `$0.0046 / M tokens` |
| Embedding | `tei-medcpt-query` | `ncbi/MedCPT-Query-Encoder` | `tei-medcpt-query` on `8084` | `$0.0028 / M tokens` |
| Embedding | `specter2-adapter` | `allenai/specter2_base` with SPECTER2 adapter | `specter2-adapter` on `8086` | `$0.0046 / M tokens` |
| Embedding | `tei-indus` | `nasa-impact/nasa-ibm-st.38m` | `tei-indus` on `8087` | `$0.0056 / M tokens` |
| Embedding | `tei-sapbert` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | `tei-sapbert` on `8088` | `$0.0046 / M tokens` |
| Embedding | `nomic-embed-code` | `nomic-embed-code.Q8_0.gguf` | `nomic-embed-code` on `8082` | `$0.15 / M tokens` |
| Rerank | `tei-reranker` | `BAAI/bge-reranker-v2-m3` | `tei-reranker` on `8090` | `$0.025 / M tokens` |

Costs are estimated commercial-equivalent token prices for local services so LiteLLM spend and AAWM `session_history.response_cost_usd` remain populated.

## Embeddings

Endpoint:

- `POST /v1/embeddings`

Choose the model alias that matches the retrieval domain:

- Use `tei-medcpt-query` for biomedical query text.
- Use `tei-medcpt-article` for biomedical article/document text.
- Use `specter2-adapter` for scientific paper/document similarity.
- Use `tei-indus` for NASA/earth-science document embeddings.
- Use `tei-sapbert` for biomedical entity and concept text.
- Use `nomic-embed-code` for code-oriented embeddings.

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
  -H "langfuse_trace_name: local-code-embedding" \
  -d '{
    "model": "nomic-embed-code",
    "input": [
      "def add(a, b): return a + b",
      "async function fetchJson(url) { return fetch(url).then(r => r.json()) }"
    ]
  }'
```

Biomedical query/document pairing example:

```bash
curl -sS "$LITELLM_BASE_URL/v1/embeddings" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-litellm-session-id: sess_123" \
  -d '{
    "model": "tei-medcpt-query",
    "input": "first-line therapy for EGFR mutated non-small cell lung cancer"
  }'
```

## Rerank

Endpoints:

- `POST /rerank`
- `POST /v1/rerank`

Model:

- `tei-reranker`

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
  -H "langfuse_trace_name: local-document-rerank" \
  -d '{
    "model": "tei-reranker",
    "query": "Which document explains local embedding and rerank usage through AAWM LiteLLM?",
    "documents": [
      "Consumers call local model service ports directly.",
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
    "model": "tei-reranker",
    "query": "What key should clients use?",
    "documents": [
      "Use the AAWM LiteLLM Authorization key.",
      "Use a direct local service port from the client."
    ],
    "top_n": 1
  }'
```

## Operational Notes

- Keep the same `x-litellm-session-id` across related embedding, retrieval, rerank, and generation calls when they belong to one user workflow.
- Use unique `x-litellm-trace-id` values per request or logical trace segment when your application already has trace ids.
- Do not place local service ports in browser, mobile, or consumer-service requests. AAWM/LiteLLM owns local model routing and cost attribution.
- `session_history` records use provider `local_embed` for embedding calls and `local_rerank` for rerank calls.
