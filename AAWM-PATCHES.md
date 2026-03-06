# AAWM LiteLLM Fork — Patch Registry and Migration Guide

This fork of [BerriAI/litellm](https://github.com/BerriAI/litellm) tracks
upstream releases with AAWM-specific patches applied on top.

**Base version:** v1.82.1 (commit `d07689d2d7`, rebased 2026-03-06)

**Note:** aawm.1 (OAuth token preservation in `clean_headers` and
`_get_forwardable_headers`) was absorbed by upstream in PR #19912 (v1.81.13)
and is no longer carried as a separate patch.

## Standalone Deployment

`litellm-config.yaml` and `docker-compose.yml` are co-located in this repo,
enabling the fork to run independently without the AAWM repo.

**`litellm-config.yaml`** — canonical model routing config. Updates should be
made here and synced back to AAWM if needed.

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
| `aawm/patches` | Production branch — latest stable patches on top of a pinned upstream release. |
| `dev` | Integration branch — worktrees merge here for testing before promotion to `aawm/patches`. |
| `main` | Mirrors upstream `main` at the latest stable release. |

**Versioning scheme:** `{upstream_version}-aawm.{patch_number}`
Current version: `1.82.1-aawm.4` (4 active patches)

## Applied Patches

### aawm.2 — Skip x-api-key in /anthropic/ pass-through when OAuth token present

**File:** `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
**Function:** `anthropic_proxy_route()`
**Upstream issue:** Claude Code uses the `/anthropic/v1/messages` pass-through
route, which bypasses the normal LiteLLM request pipeline. The pass-through
calls `get_credentials()` to build the `x-api-key` custom header. When
`ANTHROPIC_API_KEY` is not set, `get_credentials()` returns `None`, resulting
in `x-api-key: None` being forwarded to Anthropic, which rejects the request.

**Fix:** Detect `Bearer sk-ant-oat` in the request's `Authorization` header.
When found, set `custom_headers={}` so no `x-api-key` is injected and the OAuth
bearer token passes through unmodified via `_forward_headers=True`.

**Why not upstream:** Upstream assumes API key auth. OAuth-only authentication
(no API key) is a non-standard configuration not currently supported upstream.

**Upstream watch:** v1.82.2+ adds `forward_llm_provider_auth_headers` and
`authenticated_with_header` in `clean_headers()`. This BYOK flow may eventually
supersede Patch 2. Monitor upstream BYOK for pass-through route coverage.

---

### aawm.4 — Agent identity callback for Langfuse attribution

**File:** `litellm/integrations/aawm_agent_identity.py` (new file)
**Upstream issue:** LiteLLM has no built-in mechanism to extract agent/tenant
identity from the system prompt and map it to Langfuse tracing metadata.

**Fix:** Custom `AawmAgentIdentity(CustomLogger)` callback that:
- Extracts `LF_AGENT`, `LF_TENANT`, and `LF_TASK_IDS` markers from system
  prompts in `async_log_pre_api_call`.
- Maps them to Langfuse fields: `trace_name` ← `LF_AGENT`,
  `trace_user_id` ← `LF_TENANT`, `session_id` ← `LF_TASK_IDS`.
- Enables per-agent and per-task cost attribution in Langfuse without requiring
  Claude Code to support dynamic per-subagent headers.

**Why not upstream:** AAWM-specific identity injection scheme; not a general
upstream feature.

**Note:** Includes the unused-import fix (removal of `Union` import) that was
previously a separate `chore` commit — squashed into this patch.

**Upstream watch:** v1.82.2+ adds `x-litellm-agent-id` / `x-litellm-trace-id`
agent tracing headers. These are header-based (not system-prompt-based) and
require client support. Patch 4 remains needed because Claude Code cannot set
per-subagent headers dynamically.

---

### aawm.3 — Propagate request headers through pass-through logging handlers

**Files:**
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py`
- `litellm/types/passthrough_endpoints/pass_through_endpoints.py`

**Upstream issue:** [Langfuse discussion #4848](https://github.com/langfuse/langfuse/discussions/4848).
The `AnthropicPassthroughLoggingHandler` builds `proxy_server_request` with
only a `body` field, never populating `headers`. Callback integrations like
`add_metadata_from_header` receive an empty headers dict and silently skip all
header-based metadata extraction.

**Fix:**
1. Add `request_headers: Optional[dict]` to `PassthroughStandardLoggingPayload`.
2. In `pass_through_request()`, call `clean_headers(request.headers)` and store
   the result as `request_headers` in the payload.
3. In `AnthropicPassthroughLoggingHandler.anthropic_passthrough_logged_success_handler()`,
   populate `proxy_server_request["headers"]` from `request_headers` when present.

**Why not upstream:** Upstream bug not fixed as of v1.82.1 (March 2026).

---

### aawm.5 — Google OAuth tokens in Gemini pass-through route

**Files:**
- `litellm/proxy/litellm_pre_call_utils.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`

**Upstream issue:** The Gemini pass-through route requires `GEMINI_API_KEY` and
always injects it as a `key` query parameter. Google Cloud OAuth tokens
(`Bearer ya29.*`) cannot be used as an alternative authentication method.
Additionally, `clean_headers()` strips `Authorization` headers containing
Google OAuth tokens via the `_SPECIAL_HEADERS_CACHE` mechanism.

**Fix:**
1. In `clean_headers()`: extend the existing Anthropic OAuth preservation check
   to also preserve `Authorization` headers containing Google OAuth tokens
   (`Bearer ya29.*`). Change: extend `if is_anthropic_oauth_key(value)` to
   `if is_anthropic_oauth_key(value) or value.startswith("Bearer ya29.")`.
2. In `gemini_proxy_route()`: detect `Bearer ya29.*` in the Authorization header.
   When found, skip injecting the server-side `GEMINI_API_KEY` query param and
   enable `_forward_headers=True` so the OAuth token reaches Google's API.

**Why not upstream:** Upstream Gemini pass-through requires `GEMINI_API_KEY` and
does not support OAuth token-based authentication. GCP service-account tokens
and user OAuth credentials are an AAWM-specific use case.

**Upstream watch:** v1.82.2+ adds `authenticated_with_header` param to
`clean_headers()`. When we next rebase past v1.82.2, integrate the `ya29.*`
check with the new `authenticated_with_header` guard.

---

## Dropped Patches

### aawm.1 — OAuth token preservation in clean_headers and header forwarding (DROPPED)

**Upstream status:** Merged in PR #19912, included in v1.81.13.
**What it did:** Preserved `Authorization` headers containing Anthropic OAuth
tokens (`sk-ant-oat*`) in `clean_headers()` and `_get_forwardable_headers()`.
**Action:** Dropped since v1.81.14 rebase. Upstream now handles this via
`is_anthropic_oauth_key()` in both functions.

aawm.5 extends the upstream `clean_headers()` fix to also cover Google OAuth
tokens (`ya29.*`).

---

## Upstream Changelog: v1.81.14 → v1.82.1

Key changes in the 685 commits between our previous base and current base:

**Features:**
- Generic `llm_passthrough_factory_proxy_route()` — shared factory for LLM provider pass-through endpoints
- Cursor pass-through endpoint (`/cursor/{endpoint:path}`)
- GPT-5.4/5.4-pro model support, ChatGPT gpt-5.3 OAuth model aliases
- Anthropic top-level `cache_control` for automatic prompt caching
- MCP BYOK with OAuth 2.1 PKCE
- A2A agent custom headers (`static_headers`, `extra_headers`)
- Chat UI with MCP tools and streaming
- RBAC for Vector Stores and Agents

**Bug Fixes:**
- OAuth token handling in `count_tokens` endpoint
- Pass-through Azure 429/5xx streaming propagation
- Exception catching in pass-through streaming logging handler

**Security:**
- JWT plaintext leak in debug logs fixed

**Breaking Changes:**
- `StreamingChoices` removed from `ModelResponse` (use `ModelResponseStream`)
- `python-multipart` version bumped to `>=0.0.22`

**Post-v1.82.1 upstream (not included in this rebase):**
- Claude Code BYOK: `forward_llm_provider_auth_headers` config + `authenticated_with_header` param in `clean_headers()`. May eventually supersede Patches 2 and 5.
- Agent Tracing: `context_id`-based trace propagation + `x-litellm-agent-id` / `x-litellm-trace-id` headers.

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
FROM ghcr.io/berriai/litellm:main-v1.82.1
# Overlay patched source files directly from the fork checkout
COPY litellm/proxy/litellm_pre_call_utils.py \
     /usr/local/lib/python3.11/site-packages/litellm/proxy/litellm_pre_call_utils.py
COPY litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py \
     /usr/local/lib/python3.11/site-packages/litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py
COPY litellm/proxy/pass_through_endpoints/pass_through_endpoints.py \
     /usr/local/lib/python3.11/site-packages/litellm/proxy/pass_through_endpoints/pass_through_endpoints.py
COPY litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py \
     /usr/local/lib/python3.11/site-packages/litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py
COPY litellm/integrations/aawm_agent_identity.py \
     /usr/local/lib/python3.11/site-packages/litellm/integrations/aawm_agent_identity.py
COPY litellm/types/passthrough_endpoints/pass_through_endpoints.py \
     /usr/local/lib/python3.11/site-packages/litellm/types/passthrough_endpoints/pass_through_endpoints.py

ENV CONFIG_FILE_PATH=/app/config.yaml
EXPOSE 4000
```

### Migration steps

1. Confirm fork builds and patches are stable across a full `docker compose up`.
2. Update `aawm/litellm.Dockerfile` to use the overlay approach above,
   referencing `ghcr.io/berriai/litellm:main-v1.82.1` as the base image.
3. Add a GitHub Actions workflow in this fork to build and push a Docker image
   to `ghcr.io/zepfu/litellm` on pushes to `aawm/patches`.
4. Update `aawm/docker-compose.yml` to pull the pre-built image from
   `ghcr.io/zepfu/litellm:1.82.1-aawm.4` instead of building locally.

### Upstream version upgrades

When upgrading to a new upstream LiteLLM version:

1. Fetch upstream tags: `git fetch upstream --tags`
2. Create new rebase branch: `git checkout <target-commit> -b aawm/patches-vX.Y.Z`
3. Cherry-pick AAWM commits in order (excluding any that were absorbed upstream).
4. Resolve conflicts (most likely in the patched files — check if upstream
   changed the target code blocks).
5. Update this file: version string, patch entries, dropped patches.
6. Push branch, update `aawm/litellm.Dockerfile` to reference the new AAWM image tag.
