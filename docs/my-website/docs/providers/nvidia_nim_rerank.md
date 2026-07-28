import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# NVIDIA NIM - Rerank

Use NVIDIA NIM rerank models through LiteLLM's Cohere-compatible `rerank`
interface.

| Property | Details |
|----------|---------|
| Description | NVIDIA NIM provides reranking models for semantic search and retrieval-augmented generation (RAG) |
| Provider Doc | [NVIDIA NIM API Reference ↗](https://docs.api.nvidia.com/nim/reference/) |
| LiteLLM Endpoint | `/rerank` |

## Overview

Nvidia NIM rerank models help you:
- Reorder search results by relevance to a query
- Improve RAG (Retrieval-Augmented Generation) accuracy
- Filter and rank large document sets efficiently

**Supported Models:**
- NVIDIA NIM rerank models with LiteLLM model metadata

:::tip

See the full list of LiteLLM supported NVIDIA NIM models on
[models.litellm.ai](https://models.litellm.ai).

:::

## Usage

### LiteLLM Python SDK

<Tabs>
<TabItem value="mistral-4b-catalog" label="Catalog Model">

```python
import litellm
import os

os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="What is the GPU memory bandwidth of H100 SXM?",
    documents=[
        "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth.",
        "A100 provides up to 20X higher performance over the prior generation.",
        "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of memory bandwidth per GPU."
    ],
    top_n=3,
)

print(response)
```

</TabItem>
<TabItem value="mistral-4b-body-alias" label="Catalog Body Model Alias">

```python
import litellm
import os

os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

response = litellm.rerank(
    model="nvidia_nim/nv-rerank-qa-mistral-4b:1",
    query="What is the GPU memory bandwidth of H100 SXM?",
    documents=[
        "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth.",
        "A100 provides up to 20X higher performance over the prior generation.",
        "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of memory bandwidth per GPU."
    ],
    top_n=3,
)

print(response)
```

</TabItem>
</Tabs>

:::caution

NVIDIA has deprecated `nv-rerankqa-mistral-4b-v3`. Existing deployments may
continue to use the compatibility alias
`nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3`, but new configurations should
use the catalog-supported `nvidia_nim/nvidia/rerank-qa-mistral-4b` model in the
first tab above.

:::

**Response:**
```json
{
    "results": [
        {
            "index": 2,
            "relevance_score": 6.828125,
            "document": {
                "text": "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of memory bandwidth per GPU."
            }
        },
        {
            "index": 0,
            "relevance_score": -1.564453125,
            "document": {
                "text": "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth."
            }
        }
    ]
}
```


## Usage with LiteLLM Proxy

### 1. Setup Config

Add Nvidia NIM rerank models to your proxy configuration:

```yaml
model_list:
  - model_name: nvidia-rerank
    litellm_params:
      model: nvidia_nim/nvidia/rerank-qa-mistral-4b
      api_key: os.environ/NVIDIA_NIM_API_KEY
```

### 2. Start Proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Make Rerank Requests

```bash
curl -X POST http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia-rerank",
    "query": "What is the GPU memory bandwidth of H100?",
    "documents": [
      "H100 delivers 3TB/s memory bandwidth",
      "A100 has 2TB/s memory bandwidth",
      "V100 offers 900GB/s memory bandwidth"
    ],
    "top_n": 2
  }'
```

## Endpoint families

LiteLLM supports both NVIDIA NIM rerank endpoint families:

| NVIDIA NIM endpoint | LiteLLM model form | Resolution |
|----------|----------|----------|
| `/v1/retrieval/{model}/reranking` | `nvidia_nim/<model>` | Catalog metadata may instead select a shared retrieval `endpoint_path` and `body_model`. Without rerank metadata, LiteLLM derives the model-in-path URL shown in this column. |
| `/v1/ranking` | `nvidia_nim/ranking/<model>` | The `ranking/` prefix selects the fixed endpoint, and the cleaned model name is sent in the JSON body. |

Use the `ranking/` segment after `nvidia_nim/` to select `/v1/ranking`.

:::caution

NVIDIA has retired the hosted `llama-3.2-nv-rerankqa-1b-v2` route. The
`/v1/ranking` examples below retain its catalog entry only to document
compatibility with an existing or self-hosted deployment; they are not a
recommendation for new hosted configurations. Only add a `ranking/` route for
a model that NVIDIA documents as available to your account or deployment.

:::

### LiteLLM Python SDK

```python showLineNumbers title="Compatibility only: select /v1/ranking"
import litellm
import os

os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

# Compatibility only: this retired hosted model may remain on existing or
# self-hosted deployments.
response = litellm.rerank(
    model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
    query="which way did the traveler go?",
    documents=[
        "two roads diverged in a yellow wood...",
        "then took the other, as just as fair...",
        "i shall be telling this with a sigh somewhere ages and ages hence..."
    ],
    top_n=3,
    truncate="END",  # Optional: truncate long text from the end
)

print(response)
```

### LiteLLM Proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: nvidia-ranking-compat
    litellm_params:
      # Compatibility only: verify this model exists on your deployment.
      model: nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2
      api_key: os.environ/NVIDIA_NIM_API_KEY
```

```bash title="Request to LiteLLM Proxy"
curl -X POST http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia-ranking-compat",
    "query": "which way did the traveler go?",
    "documents": [
      "two roads diverged in a yellow wood...",
      "then took the other, as just as fair..."
    ],
    "top_n": 2
  }'
```

### Understanding model resolution

**Catalog-backed shared retrieval endpoint:**

```
LiteLLM model: nvidia_nim/nvidia/rerank-qa-mistral-4b
Catalog endpoint_path: /v1/retrieval/nvidia/reranking
Catalog body_model: nv-rerank-qa-mistral-4b:1

Provider URL: https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking
Provider JSON model: nv-rerank-qa-mistral-4b:1
```

The typed catalog entry takes precedence over generic model-in-path derivation.
For a retrieval model without rerank metadata, LiteLLM strips the
`nvidia_nim/` prefix, derives
`/v1/retrieval/{clean_model}/reranking`, and converts underscores to periods in
the JSON body model. This fallback describes routing behavior; it does not
confirm that an uncataloged model is available.

**Endpoint flow:**

```
Client Request                LiteLLM                              Provider API
──────────────              ────────────                         ─────────────

# Catalog-backed shared retrieval endpoint
model: "nvidia_nim/nvidia/rerank-qa-mistral-4b"
                            1. Reads catalog endpoint_path ─────▶ POST /v1/retrieval/nvidia/reranking
                            2. Reads catalog body_model          Body: {"model": "nv-rerank-qa-mistral-4b:1", ...}


# Compatibility-only ranking endpoint
model: "nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2"
                            1. Detects "ranking/" prefix
                            2. Extracts model: nvidia/llama-3.2-nv-rerankqa-1b-v2
                            3. Routes to ranking endpoint ──────▶ POST /v1/ranking
                                                                  Body: {"model": "nvidia/llama-3.2-nv-rerankqa-1b-v2", ...}
```

**When to use each endpoint:**

| Endpoint | Model Prefix | Use Case |
|----------|--------------|----------|
| Catalog `endpoint_path` | `nvidia_nim/<catalog-model>` | Preferred for catalog-backed rerank models, including shared retrieval endpoints |
| Derived `/v1/retrieval/{model}/reranking` | `nvidia_nim/<model>` | Fallback for models without rerank metadata; availability must be verified separately |
| `/v1/ranking` | `nvidia_nim/ranking/<model>` | Compatibility or deployment-specific models that require the ranking endpoint |

:::tip

Check NVIDIA's model deployment page for your selected model to see which
endpoint it requires.

:::

### Exact provider request bodies

For the default retrieval family:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="What is GPU bandwidth?",
    documents=["H100 has 3TB/s", "A100 has 2TB/s"],
    top_n=2,
    truncate="END",
)
```

LiteLLM sends:

```json
{
  "model": "nv-rerank-qa-mistral-4b:1",
  "query": {"text": "What is GPU bandwidth?"},
  "passages": [{"text": "H100 has 3TB/s"}, {"text": "A100 has 2TB/s"}],
  "top_k": 2,
  "truncate": "END"
}
```

For `/v1/ranking`, the following exact body is compatibility-only for an
existing deployment of the retired catalog model:

```json
{
  "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
  "query": {"text": "Which GPU is faster?"},
  "passages": [{"text": "H100 is fast"}, {"text": "A100 is slower"}],
  "top_k": 1
}
```

The corresponding LiteLLM request uses
`model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2"`,
`query="Which GPU is faster?"`, `documents=["H100 is fast", "A100 is slower"]`,
and `top_n=1`.

## API parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | The Nvidia NIM rerank model name with `nvidia_nim/` prefix |
| `query` | string | The search query to rank documents against |
| `documents` | array | Documents to rank |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | integer | All documents | Mapped to NVIDIA NIM's `top_k` |

### Nvidia-Specific Parameters

**`truncate`** controls truncation when text exceeds the model's context
window. Valid values are `"NONE"` and `"END"`.

LiteLLM does not support Cohere's `rank_fields`, `max_chunks_per_doc`, or
`max_tokens_per_doc` for NVIDIA NIM. By default, providing one raises an
`UnsupportedParamsError`; with `drop_params=True`, LiteLLM drops it instead:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="GPU performance",
    documents=["High performance computing", "Fast GPU processing"],
    rank_fields=["text"],
    drop_params=True,
)
```

`top_n` is recorded as a provider rename to `top_k` in internal adaptation
metadata. Dropped unsupported parameters are recorded by name, action, and
reason; values are not recorded.

### Parameter adaptation metadata

When an adaptation occurs, LiteLLM writes two flat keys directly under
`response._hidden_params`: `provider_parameter_adaptations` and
`provider_parameter_adaptations_truncated_count`. The records contain exactly
`name`, `action`, and `reason`; parameter values are never included.

For a normal `top_n` to `top_k` rename, the relevant keys are:

```json
{
  "provider_parameter_adaptations": [
    {
      "name": "top_n",
      "action": "renamed",
      "reason": "provider_rename"
    }
  ],
  "provider_parameter_adaptations_truncated_count": 0
}
```

For an unsupported parameter dropped with `drop_params=True`, the corresponding
value-free record is:

```json
{
  "provider_parameter_adaptations": [
    {
      "name": "rank_fields",
      "action": "dropped",
      "reason": "unsupported_param"
    }
  ],
  "provider_parameter_adaptations_truncated_count": 0
}
```

### `return_documents`

`return_documents` controls local LiteLLM response shaping and is never sent
to NVIDIA NIM. It defaults to including returned document text. Set it to
`False` to return indexes and relevance scores without document text:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="GPU performance",
    documents=["High performance computing", "Fast GPU processing"],
    top_n=2,
    truncate="END",
    return_documents=False,
)
```

## Model-driven endpoint and body metadata

For typed catalog models with rerank metadata, LiteLLM uses `endpoint_path` to
select the provider URL and `body_model` for the JSON `model` field. This
supports shared endpoints such as `/v1/retrieval/nvidia/reranking`, where the
model is selected by the body rather than embedded in the URL.

Only models without rerank metadata use generic model-in-path derivation for
the retrieval family. The `nvidia_nim/` and optional `ranking/` prefixes are
LiteLLM routing syntax and are not sent as part of the provider model name.

## Authentication

Set your Nvidia NIM API key:

<Tabs>
<TabItem value="env" label="Environment Variable">

```bash
export NVIDIA_NIM_API_KEY="nvapi-..."
```

</TabItem>
<TabItem value="python" label="Python">

```python
import os
os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

# Or pass directly
response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="test",
    documents=["doc1"],
    api_key="nvapi-...",
)
```

</TabItem>
</Tabs>

## Custom API Base URL

LiteLLM resolves the NVIDIA NIM rerank API base in this order:

1. `api_base` passed in the request or proxy model configuration
2. `NVIDIA_NIM_RERANK_API_BASE`
3. `NVIDIA_NIM_API_BASE`, unless it points to `integrate.api.nvidia.com`
4. `https://ai.api.nvidia.com`

The `integrate.api.nvidia.com` base is for other NVIDIA NIM endpoint families,
so LiteLLM ignores it as a rerank fallback and uses the hosted rerank base.

**Option 1: Dedicated rerank environment variable**

```bash
export NVIDIA_NIM_RERANK_API_BASE="https://your-custom-endpoint.com"
```

**Option 2: Pass `api_base` directly**

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="test",
    documents=["doc1"],
    api_base="https://your-custom-endpoint.com",
)
```

**Option 3: Shared NVIDIA NIM environment fallback**

```bash
export NVIDIA_NIM_API_BASE="https://your-custom-endpoint.com"
```

Use this only when the same base is appropriate for rerank traffic. Prefer
`NVIDIA_NIM_RERANK_API_BASE` when rerank and other NVIDIA NIM APIs use different
hosts.

**Option 4: Full retrieval endpoint URL**

If `api_base` already contains a `/retrieval/` endpoint path, LiteLLM uses the
URL as-is:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
    query="test",
    documents=["doc1"],
    api_base="https://your-custom-endpoint.com/v1/retrieval/nvidia/reranking",
)
```

LiteLLM will detect the full URL (by checking for `/retrieval/` in the path) and use it as-is.

### How do I get an API key?

Get your Nvidia NIM API key from [Nvidia's website](https://developer.nvidia.com/nim/).

## Related Documentation

- [Nvidia NIM - Main Documentation](./nvidia_nim)
- [Nvidia NIM Chat Completions](./nvidia_nim#sample-usage)
- [LiteLLM Rerank Endpoint](../rerank)
- [Nvidia NIM Official Docs ↗](https://docs.api.nvidia.com/nim/reference/)
