# AAWM local embedding routes

The local embedding aliases use the existing `local_embed` provider and
OpenAI-compatible `/v1/embeddings` endpoints. Keep the MedCPT article and query
encoders separate and use the SPECTER-2 adapter service for SPECTER requests.

| LiteLLM alias | Backend | Served model | Dimensions |
| --- | --- | --- | --- |
| `specter2-adapter` | Mahaf, port 8086 | `allenai/specter2`, active `[PRX]` adapter | 768 |
| `tei-medcpt-article` | Mahaf, port 8083 | `tei-medcpt-article` (`ncbi/MedCPT-Article-Encoder`) | 768 |
| `tei-medcpt-query` | Mahaf, port 8084 | `tei-medcpt-query` (`ncbi/MedCPT-Query-Encoder`) | 768 |
| `nomic-embed-code` | Thoth, `aawm-tap-nomic-code-gguf-dev:8080` | `nomic-embed-code`, Q4_K_M | 3584 |
| `qwen3-embedding-8b` | Thoth, `aawm-tap-qwen3-embedding-8b-dev:8080` | `qwen3-embedding-8b`, Q4_K_M | 4096 |

Mahaf URLs retain `AAWM_TAP_MODEL_HOST`, normally
`mahaf.tailf1878c.ts.net`, and the `/v1` suffix. The Thoth URLs use the
container names above with `/v1`; their host ports 8082 and 8088 are published
only on `127.0.0.1`. A bridge-networked proxy cannot reach those loopback
publications merely by using a host-gateway address.

## Private Docker connectivity

The affected proxies share the existing `aawm-infrastructure_default` network.
Attach each Thoth embedding container to that network before loading these
routes:

```bash
docker network connect aawm-infrastructure_default aawm-tap-nomic-code-gguf-dev
docker network connect aawm-infrastructure_default aawm-tap-qwen3-embedding-8b-dev
```

Run each connection command only when the container is absent from the network;
inspect membership with `docker network inspect aawm-infrastructure_default`.
This gives the proxies stable Docker DNS names without changing host port
bindings. Connections survive container restart and host reboot. Reconnect
after recreating either backend container, before accepting its LiteLLM route.
Do not use transient container IP addresses in the proxy configuration.

## Configuration and activation

On Thoth, both `aawm-litellm` and `litellm-dev` load persistent repository-root
`litellm-dev-config.yaml`. Their container paths are respectively
`/app/litellm-config.yaml` and `/app/litellm-dev-config.yaml`. Use the existing
deployment workflow for the selected proxy and restart it to load the updated
configuration. A file change alone does not verify its loaded routing table.

Alpha has its own configuration and lifecycle; see
[LiteLLM Alpha](litellm-alpha.md). Inspect its actual bind source before
activation: an operator-selected persistent configuration may override the
default repository-root `litellm-alpha-config.yaml`. Preserve its unrelated
routes when synchronizing the selected embedding entries. Never bind a live
proxy to an implementation worktree.

## Acceptance evidence

From the affected proxy container, check Mahaf TEI `/info`, SPECTER `/health`,
and the two Thoth `/v1/models` catalogs. SPECTER's health response identifies
the active adapter; it does not implement TEI `/info` or `/v1/models`.
Then send embedding requests through all five LiteLLM aliases using the
deployment's supported authentication, and verify each vector is finite and
has the dimension above. Correlate the request window with the named backend
container's access logs and runtime model identity. Host-only probes, health
responses, and a successful vector from another model do not establish routed
acceptance. Keep API credentials out of retained evidence.
