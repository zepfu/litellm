# AAWM LiteLLM Fork — Patch Registry and Migration Guide

This fork of [BerriAI/litellm](https://github.com/BerriAI/litellm) tracks the
upstream `v1.81.13` release with AAWM-specific patches applied on top.

## Standalone Deployment

`litellm-config.yaml` and `docker-compose.yml` are now co-located in this repo,
enabling the fork to run independently without the AAWM repo.

**`litellm-config.yaml`** — copied from `~/projects/aawm/litellm-config.yaml`.
This is the canonical model routing config. Updates should be made here and
synced back to AAWM if needed (or AAWM can reference this repo as the source
of truth going forward).

**`docker-compose.yml`** — standalone compose file that brings up all required
services: LiteLLM (built from this fork), CLIProxyAPI (Gemini routing),
and the full Langfuse v3 observability stack (ClickHouse, Redis, MinIO,
langfuse-web, langfuse-worker, PostgreSQL).

**One prerequisite:** `cliproxyapi-config.yaml` must be present in this repo
root before `docker compose up`. Copy from the AAWM repo:

```bash
cp ~/projects/aawm/cliproxyapi-config.yaml ./cliproxyapi-config.yaml
```

**AAWM-specific references in `litellm-config.yaml`** (no changes needed —
these work as-is in the compose network):

- Gemini models use `api_base: "http://cliproxyapi:8317/v1"` — resolves to the
  `cliproxyapi` service defined in this compose file.
- Langfuse callback uses `LANGFUSE_HOST: "http://langfuse-web:3000"` — resolves
  to the `langfuse-web` service defined in this compose file.
- `os.environ/AAWM_XAI_API_KEY` and `os.environ/AAWM_OPENAI_API_KEY` — set
  these in a `.env` file at repo root (both optional; only needed for xAI/OpenAI
  models).

**Port allocation** (matches AAWM defaults to avoid host conflicts):

| Service | Host port |
|---------|-----------|
| LiteLLM proxy | 4000 |
| CLIProxyAPI | 8317 |
| Langfuse web | 3000 |
| PostgreSQL | 5435 |
| ClickHouse HTTP | 8123 |
| Redis | 6380 |
| MinIO S3 | 9010 |
| MinIO console | 9011 |

Note: PostgreSQL is mapped to host port **5435** (not 5434) to avoid conflicting
with the AAWM tristore if both stacks are running simultaneously on the same host.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Pinned to upstream tag `v1.81.13`. Receives upstream version bumps when AAWM is ready to upgrade. |
| `aawm/patches` | Work-in-progress patches. Rebased onto `main` before each release. |

**Versioning scheme:** `{upstream_version}-aawm.{patch_number}`
Current version: `1.81.13-aawm.2`

## Applied Patches

### aawm.1 — OAuth token preservation in clean_headers and header forwarding

**Commit:** `852cf63` on `aawm/patches`
**File:** `litellm/proxy/litellm_pre_call_utils.py`
**Upstream issue:** `clean_headers()` strips the `Authorization` header because
it appears in `_SPECIAL_HEADERS_CACHE`. When Claude Code authenticates via
device-code OAuth (token prefix `sk-ant-oat01-*`), the bearer token is silently
dropped before it reaches the Anthropic provider's `optionally_handle_anthropic_oauth()`
handler.

**Fix:**
1. In `clean_headers()`: preserve `Authorization` when the value starts with
   `"Bearer sk-ant-oat"` so `forward_client_headers_to_llm_api()` can see it.
2. In `_get_forwardable_headers()`: explicitly forward the `Authorization` header
   alongside `anthropic-beta` and `x-*` headers.

**Why not upstream:** Upstream requires `ANTHROPIC_API_KEY` for all Anthropic
calls. OAuth-only authentication (no API key) is a non-standard config.

---

### aawm.2 — Skip x-api-key in /anthropic/ pass-through when OAuth token present

**Commit:** `709e755` on `aawm/patches`
**File:** `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
**Upstream issue:** Claude Code uses the `/anthropic/v1/messages` pass-through
route, which bypasses the normal LiteLLM request pipeline. The pass-through
calls `get_credentials()` to build the `x-api-key` custom header. When
`ANTHROPIC_API_KEY` is not set, `get_credentials()` returns `None`, resulting
in `x-api-key: None` being forwarded to Anthropic, which rejects the request.

**Fix:** Detect `Bearer sk-ant-oat` in the request's `Authorization` header.
When found, set `custom_headers={}` so no `x-api-key` is injected and the OAuth
bearer token passes through unmodified via `_forward_headers=True`.

**Why not upstream:** Same as aawm.1 — upstream assumes API key auth.

---

## Migration Plan: Building from Fork Instead of Patching at Runtime

### Current approach (litellm.Dockerfile in aawm repo)

```dockerfile
FROM ghcr.io/berriai/litellm:main-latest
RUN pip install --no-cache-dir --upgrade "litellm>=1.81.13"
# ... runtime Python one-liner patches ...
```

**Problems:**
- Patches are fragile: they assert on exact upstream source strings. Any upstream
  refactor breaks the build with an `AssertionError`.
- The `pip install --upgrade litellm>=1.81.13` step pulls from PyPI and may
  change on every build (non-deterministic).
- One-liner patches are difficult to read, review, and test.

### Target approach (build from this fork)

```dockerfile
# Build the LiteLLM wheel from the fork at a pinned commit
FROM python:3.11-slim AS builder
WORKDIR /build
COPY . .
RUN pip install build && python -m build --wheel

# Runtime image matches upstream base
FROM ghcr.io/berriai/litellm:main-v1.81.13
# Replace the installed litellm package with our patched wheel
COPY --from=builder /build/dist/litellm-*.whl /tmp/
RUN pip install --no-cache-dir --force-reinstall /tmp/litellm-*.whl && rm /tmp/litellm-*.whl

ENV CONFIG_FILE_PATH=/app/config.yaml
EXPOSE 4000
```

**Alternatively (simpler, no wheel build needed):**

```dockerfile
FROM ghcr.io/berriai/litellm:main-v1.81.13
# Overlay patched source files directly from the fork checkout
COPY litellm/proxy/litellm_pre_call_utils.py \
     /usr/local/lib/python3.11/site-packages/litellm/proxy/litellm_pre_call_utils.py
COPY litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py \
     /usr/local/lib/python3.11/site-packages/litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py

ENV CONFIG_FILE_PATH=/app/config.yaml
EXPOSE 4000
```

This second approach is simpler for two patched files. It requires knowing the
Python site-packages path inside the base image (verify with
`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"`).

### Migration steps

1. Confirm fork builds and patches are stable across a full `docker compose up`.
2. Update `aawm/litellm.Dockerfile` to use one of the approaches above,
   referencing a specific commit SHA or tag from this fork (not `main-latest`).
3. Add a GitHub Actions workflow in this fork to build and push a Docker image
   to `ghcr.io/zepfu/litellm` on pushes to `aawm/patches`.
4. Update `aawm/docker-compose.yml` to pull the pre-built image from
   `ghcr.io/zepfu/litellm:1.81.13-aawm.2` instead of building locally.

### Upstream version upgrades

When upgrading to a new upstream LiteLLM version:

1. Fetch upstream tags: `git fetch upstream --tags`
2. Checkout new tag on `main`: `git checkout main && git reset --hard vX.Y.Z`
3. Rebase `aawm/patches` onto new `main`: `git checkout aawm/patches && git rebase main`
4. Resolve conflicts (most likely in the two patched files — check if upstream
   changed the target code blocks).
5. Push both branches, bump the AAWM version string in this file.
6. Update `aawm/litellm.Dockerfile` to reference the new AAWM image tag.
