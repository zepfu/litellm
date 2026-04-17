# AAWM LiteLLM Fork — Patch Registry and Migration Guide

This fork of [BerriAI/litellm](https://github.com/BerriAI/litellm) tracks
upstream releases with AAWM-specific patches applied on top.

**Base version:** v1.82.3-stable.patch.4 (commit `1e10e5cc8f`, rebased 2026-04-10)

**Image format:** venv artifact (`FROM scratch` — not a runnable image). Consumed via `COPY --from=`.

**Callback wheel:** production callback changes are published separately from
`.wheel-build/` and consumed as a GitHub release wheel by the production image.
The in-repo callback at `litellm/integrations/aawm_agent_identity.py` and the
published wheel source under `.wheel-build/` must stay behaviorally aligned.
See `WHEEL.md` for the production wheel workflow and release path.

**Note:** aawm.1 (OAuth token preservation in `clean_headers` and
`_get_forwardable_headers`) was absorbed by upstream in PR #19912 (v1.81.13)
and is no longer carried as a separate patch.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `aawm/patches` | Legacy production branch name from the original fork workflow. |
| `develop` | Integration branch — validated subsystem work should land here first. |
| `main` | Promotion branch — advances from `develop` after the full validation matrix passes. |

**Versioning scheme:** `{upstream_version}+aawm.{patch_number}` (PEP 440 local version)
Git tags use `v{upstream_version}-aawm.{patch_number}` (hyphen, since git tags aren't PEP 440).
Current carried patch set: `aawm.2`, `aawm.3`, `aawm.4`, `aawm.5`, `aawm.6`, `aawm.7`, `aawm.8`, `aawm.9`, `aawm.10`, `aawm.11`, `aawm.12`, `aawm.13`, `aawm.14`, `aawm.15`, `aawm.16`, `aawm.17` (16 active patches)

**Version metadata note:** `pyproject.toml` should stay aligned to this
registry's carried patch set. `litellm/_version.py` now reflects the installed
distribution version directly. The current working tree now aligns on
`1.82.3+aawm.17`.

**Current rebased checkpoint:** branch `rebase/upstream-1.82.3-stable.patch.4`
passed the local acceptance suite with artifact
`.analysis/artifacts/local-acceptance-20260410T232900Z.json`.

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

### aawm.4 — Agent identity callback for Langfuse trace naming

**File:** `litellm/integrations/aawm_agent_identity.py` (new file)
**Upstream issue:** LiteLLM has no built-in mechanism to extract agent/tenant
identity from Claude Code prompt content and map it to Langfuse trace naming.

**Fix:** Custom `AawmAgentIdentity(CustomLogger)` callback that:
- Extracts Claude Code subagent identity from prompt text like
  `You are 'orchestrator'` / `You are 'engineer'`.
- Rewrites `langfuse_trace_name` / fallback `metadata["trace_name"]` so traces
  become `claude-code.orchestrator`, `claude-code.engineer`, etc.
- Preserves a workable attribution model without requiring Claude Code to emit
  dynamic per-subagent headers.

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

### aawm.6 — Codex-native passthrough with preserved client OAuth

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/pass_through_unit_tests/test_pass_through_unit_tests.py`
- `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`

**Upstream issue:** Redirecting Codex CLI to LiteLLM through the existing
OpenAI passthrough path does not work if LiteLLM forwards the Codex OAuth
bearer to the public `api.openai.com/v1/responses` API. The Codex bearer is
valid for Codex-native upstream traffic but does not carry the public
`api.responses.write` scope required by the OpenAI Responses API.

**Fix:**
1. Detect real Codex-native ingress on the OpenAI passthrough lane.
2. Preserve the inbound Codex OAuth bearer and native Codex headers.
3. Route the request to `https://chatgpt.com/backend-api/codex/responses`
   instead of the public OpenAI Responses API.
4. Keep passthrough observability intact by masking raw auth in logs and
   hashing the inbound bearer before it lands in metadata.

**Why not upstream:** Upstream OpenAI passthrough assumes generic OpenAI API
traffic. This fork needs Codex-native passthrough semantics so Codex CLI can
stay logged in normally while LiteLLM acts only as the routing and
observability layer.

**Validation status:** Real `codex exec -p litellm` traffic succeeded through
`litellm-dev`, and Langfuse persisted traces with `name = "codex"`.

---

### aawm.8 — Codex streamed usage normalization and cost preservation

**Files:**
- `litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
- `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`
- `tests/llm_translation/test_llm_response_utils/test_convert_dict_to_chat_completion.py`

**Upstream issue:** Native Codex passthrough responses carry Responses-API-style
usage (`input_tokens`, `output_tokens`, `input_tokens_details`, etc.), but the
chat-completion conversion and passthrough streaming logging paths were
normalizing them incorrectly. The result was a successful Codex response with
`prompt_tokens = 0`, `completion_tokens = 0`, and `response_cost = None` inside
LiteLLM even though Codex CLI itself reported non-zero usage.

**Fix:**
1. Normalize Responses-API-style usage dicts before constructing `Usage` in the
   completion-response conversion path.
2. Detect `/responses` streaming routes in the OpenAI passthrough logging
   handler and rebuild the final response from the actual `response.completed`
   event instead of the chat-completions stream builder.
3. Calculate passthrough streaming cost with `call_type="responses"` so Codex
   usage is costed on the correct path.
4. Add focused regression tests for both the direct conversion path and the
   streamed passthrough logging path.

**Why not upstream:** Upstream's generic OpenAI passthrough logging path does
not cover this Codex-native `/backend-api/codex/responses` streaming shape.
This fork needs accurate usage and pricing for native Codex CLI observability.

**Validation status:** Focused handler regression tests passed, and a fresh live
`codex exec -p litellm` run against rebuilt `litellm-dev` showed non-zero
`input_tokens`, `cached_input_tokens`, and `output_tokens` on the passthrough
response-completed path instead of zeroed usage.

---

### aawm.9 — Bridge Anthropic streaming payload into callback logging on the rebased base

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py`

**Upstream issue:** On the rebased `v1.82.3-stable.patch.4` base, the Anthropic
streaming passthrough logging path rebuilt the final callback payload without
the original `passthrough_logging_payload`. That meant downstream callback and
logging code lost access to the request-side passthrough context during
streaming Anthropic responses.

**Fix:**
1. Thread `passthrough_logging_payload` through
   `AnthropicPassthroughLoggingHandler.chunk_creator()`.
2. Reattach it when building the final kwargs passed into the Anthropic
   callback/logging payload creation path.

**Why not upstream:** This is a fork-carried compatibility bridge tied to the
current rebased base and our callback/logging expectations on Anthropic
streaming passthrough traffic. It is not part of the net-new observability span
or tagging work; it restores request-context continuity that our existing
callback flow depends on.

**Validation status:** Included in the local acceptance runs that exercise
Claude passthrough streaming through `litellm-dev`.

---

### aawm.10 — Expand Claude persisted-output wrappers before Anthropic passthrough send

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** Recent Claude Code CLI releases truncate large
`SubagentStart` / `SessionStart` additional-context payloads and replace them
with a wrapper like:

- `Output too large (...)`
- `Full output saved to: /home/zepfu/.claude/projects/.../tool-results/...-additionalContext.txt`
- `Preview (first 2KB): ...`

This is not acceptable for the AAWM fork because subagents need to receive the
full expanded context exactly as before, without being told to read a file
manually.

**Fix:**
1. Detect the Claude persisted-output wrapper pattern in Anthropic passthrough
   text blocks.
2. Resolve the referenced file only when it is under the allowlisted Claude
   projects root and matches the expected `tool-results/*-additionalContext.txt`
   shape.
3. Replace the truncated wrapper with the full file contents before the request
   is sent upstream.

**Why not upstream:** This is a Claude Code compatibility workaround for a
specific local/private-machine workflow where LiteLLM is running on the same
machine as Claude Code and can safely access the generated persisted-output
files. It is not generic provider functionality.

**Validation status:** Focused passthrough tests cover both `SubagentStart` and
`SessionStart`, and the local acceptance harness verifies that the logged
request text no longer contains the truncation wrapper and instead contains the
fully expanded file contents.

---

### aawm.7 — Gemini Code Assist native passthrough and logging fixes

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/proxy/pass_through_endpoints/streaming_handler.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/pass_through_unit_tests/test_gemini_streaming_handler.py`
- `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py`

**Upstream issue:** Gemini CLI `oauth-personal` traffic does not use the public
Gemini Developer API path for this workflow. It targets Google Code Assist on
`https://cloudcode-pa.googleapis.com/v1internal:...`. Upstream Gemini
pass-through behavior is API-key oriented and did not correctly support the
native Code Assist route, its streamed SSE chunk format, or its response
envelope shape for observability callbacks.

**Fix:**
1. Detect Gemini Code Assist ingress on `/gemini/v1internal:...`.
2. Preserve the inbound Google OAuth bearer and forward to
   `https://cloudcode-pa.googleapis.com/v1internal:...`.
3. Route streamed Gemini passthrough callbacks to the Gemini-specific logging
   handler instead of the generic Vertex path.
4. Parse SSE `data: {...}` chunks and unwrap Code Assist responses shaped like
   `{ "traceId": ..., "response": {...} }`.
5. Strip stale compression and transfer headers when reconstructing synthetic
   `httpx.Response` objects for logging transforms so the callback path does not
   trip `httpx.DecodingError`.

**Why not upstream:** Upstream Gemini pass-through is focused on API-key-based
Gemini API traffic. AAWM needs native Gemini CLI / Google Code Assist OAuth
passthrough with clean Langfuse logging and no OpenAI-compatible translation
layer.

**Validation status:** Real `gemini -p ... --output-format json` traffic
succeeded through `litellm-dev`, native `generateContent` and
`streamGenerateContent?alt=sse` both returned `200`, and Langfuse persisted
the Gemini traces.

---
### aawm.11 — AAWM dynamic directive injection for Claude Anthropic passthrough

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** AAWM-managed Claude prompt content needs a request-time
expansion mechanism for dynamic context blocks such as agent memory. The model
must see the expanded content, not a raw directive marker, and the resolution
needs to happen on the Anthropic passthrough path after Claude persisted-output
expansion but before the upstream request is sent.

**Fix:**
1. Add a generic `<!-- AAWM ... -->` directive parser on the Anthropic request
   rewrite path.
2. Extract trusted `agent` / `tenant` context from Claude's injected identity
   line in the rewritten Anthropic request body.
3. Resolve `p=get_agent_memories` against the configured AAWM Postgres stored
   procedure `get_agent_memories(agent, tenant)`.
4. Replace the directive inline with the returned markdown text, treat `NULL`
   results as empty injections, and replace execution/config/context failures
   with a short generic AAWM failure block.

**Why not upstream:** This is AAWM-specific prompt/context control-plane
functionality tied to our Anthropic passthrough workflow, local Claude Code
request shape, and AAWM-backed context store. It is not generic LiteLLM
provider behavior.

**Validation status:** Focused passthrough tests cover successful injection,
empty injection on `NULL`, failure replacement when required context is
missing, persisted-output compatibility, and the now-async Anthropic request
preparation path.

---

### aawm.12 — Claude Code system-prompt rewrites and file-backed prompt patching

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `context-replacement/claude-code/2.1.110/auto-memory-replacement.md`
- `context-replacement/claude-code/prompt-patches/roman01la-2026-04-02.json`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `scripts/local-ci/config.json`

**Upstream issue:** Claude Code currently injects its own large `# auto memory`
and other system instructions before Anthropic passthrough send. AAWM needs to
replace portions of that prompt with file-backed fork-specific instructions
without patching Claude Code itself, while keeping the rewrite observable and
version-aware.

**Fix:**
1. Add Claude Code version-aware system-prompt override support on the
   Anthropic passthrough path.
2. Replace the `# auto memory` section with a file-backed AAWM-specific prompt
   template while preserving selected upstream sections verbatim.
3. Apply exact-match prompt patches from a JSON manifest when Claude sends
   known upstream text fragments.
4. Record rewrite tags/metadata and preserve post-rewrite context-file
   detection for `MEMORY.md` / `CLAUDE.md`.

**Why not upstream:** This is AAWM-specific Claude Code control-plane behavior.
It depends on our local prompt-management direction, our context store, and our
decision to rewrite Claude Code system prompt fragments at the LiteLLM seam
rather than patching the client.

**Validation status:** Focused passthrough tests cover the auto-memory
replacement, prompt-patch manifest application, and post-rewrite context-file
detection. Live Claude runs through `litellm-dev` showed the rewrite tags and
replacement flow on current Claude Code builds.

---

### aawm.13 — Persist per-call session history to AAWM tristore

**Files:**
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `scripts/backfill_session_history.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`

**Upstream issue:** LiteLLM and Langfuse store call-level observability, but
AAWM also needs a first-class per-call session ledger in its own Postgres store
for session analytics, model-level token rollups, and historical repair/backfill
independent of Langfuse retention.

**Fix:**
1. Add lazy creation and persistence of `public.session_history` through the
   AAWM callback path.
2. Store per-call session/model/provider lineage, token/cost fields, tool-call
   counts, and reasoning-related metadata keyed by `litellm_call_id`.
3. Add a historical backfill script that reconstructs rows from historical
   LiteLLM/Langfuse sources and can also write derived Langfuse trace tags.

**Why not upstream:** The table shape, target database, and callback/backfill
workflow are specific to AAWM's tristore and session analytics model rather
than generic LiteLLM provider behavior.

**Validation status:** Focused callback tests cover record construction and DB
write behavior, and the backfill path has already been used to populate the
tristore session-history table.

---

### aawm.14 — Session-history upserts and Gemini/Codex usage breakout enrichment

**Files:**
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `scripts/backfill_session_history.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`

**Upstream issue:** The first session-history implementation wrote sparse rows
and could not enrich existing records when richer data became available later.
Gemini and Codex/OpenAI Responses usage also expose provider-specific cache,
reasoning, and tool-call fields that were not being normalized into either
session history or Langfuse metadata.

**Fix:**
1. Change session-history writes from insert-only to upsert-on-call-id so later
   runs can enrich existing rows.
2. Normalize Gemini `thoughtsTokenCount` / `cachedContentTokenCount` and Codex
   Responses `input_tokens_details.cached_tokens` /
   `output_tokens_details.reasoning_tokens`.
3. Fall back to Responses-style `output` items and metadata-carried usage/tool
   fields when reconstructing historical observations.
4. Emit provider usage-breakout metadata/tags/spans from the callback path so
   Langfuse and session history carry the same enriched token/tool view.

**Why not upstream:** This is AAWM-specific observability normalization tied to
our session-history table, Langfuse conventions, and the provider mix we are
running through passthrough.

**Validation status:** Focused callback tests cover the Gemini and Codex usage
breakout cases, and the historical backfill has been rerun against tristore
with the upsert/enrichment logic.

---

### aawm.15 — Treat forwarded-client-auth passthrough models as healthy without server API keys

**Files:**
- `litellm/proxy/health_check.py`
- `tests/litellm_utils_tests/test_health_check.py`

**Upstream issue:** `/health` marked passthrough routes unhealthy when the
proxy itself did not have a server-side provider key, even if those routes are
designed to rely on forwarded client OAuth/auth headers at request time. That
produced false unhealthy signals for Claude passthrough deployments.

**Fix:**
1. Detect models configured to forward client headers to the upstream LLM API.
2. Skip direct health probes for those models when no server-side credential is
   configured.
3. Mark the result as healthy with an explicit `health_check_skipped` reason so
   the deployment is not shown as unhealthy solely because it depends on client
   auth passthrough.

**Why not upstream:** This behavior is coupled to our passthrough/OAuth usage
pattern and current dev/prod routing expectations for client-authenticated
provider traffic.

**Validation status:** The live `/health` response on `litellm-dev` now reports
the Anthropic passthrough model healthy with
`health_check_skipped=forwarded_client_headers_required`.

---

### aawm.16 — Align image release workflow with current Docker build and publish a GitHub release object

**Files:**
- `.github/workflows/aawm-publish.yml`

**Upstream issue:** The fork image workflow still assumed an older `venv-builder`
target and `/opt/litellm-venv/bin/python` smoke-test path that no longer exist
in the current Dockerfile. It also pushed GHCR images without creating a GitHub
release object for the `v*-aawm.*` tags, which made the release state hard to
see in the GitHub UI.

**Fix:**
1. Change the smoke test to build the current runtime image shape and run the
   import test with `--entrypoint python3`.
2. Keep the existing GHCR publish path.
3. Add an explicit GitHub release creation step for `v*-aawm.*` tags so image
   releases are visible on the Releases page alongside the callback wheel line.

**Why not upstream:** This is part of the AAWM fork release workflow and is
specific to how this fork ships its container/image artifacts and GitHub
release metadata.

**Validation status:** The previous `v1.82.3-aawm.15` tag failed on the stale
smoke-test path. This patch updates the workflow to match the current Docker
build model before the next image release tag is cut.

---

### aawm.17 — Split AAWM non-core control-plane behavior into an independent wheel line

**Files:**
- `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `.control-plane-wheel-build/pyproject.toml`
- `.github/workflows/aawm-control-plane.yml`
- `WHEEL.md`

**Upstream issue:** AAWM-specific Claude control-plane behavior such as prompt
rewrites, dynamic memory injection, and post-rewrite context tagging was still
living only inside the forked LiteLLM release path. That made it impossible for
infrastructure to pin a specific LiteLLM fork release while independently
pulling newer AAWM enhancements as wheel artifacts.

**Fix:**
1. Extract the Claude control-plane enhancement path behind a dedicated module:
   `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`.
2. Route Anthropic request-prep through that module instead of binding the
   enhancement logic directly to the main passthrough file.
3. Add an independent `aawm-litellm-control-plane` wheel build and GitHub
   release workflow on `cp-v*` tags.
4. Update wheel documentation so callback vs control-plane overlays are treated
   as separate artifact lines from the base LiteLLM fork release.

**Why not upstream:** This is AAWM-specific control-plane packaging and release
strategy. The goal is a clean distinction between required LiteLLM fork
patches and AAWM-managed enhancements that should ship independently.

**Validation status:** Focused Anthropic passthrough tests still cover the live
rewrite/injection path after the helper-module split, and the control-plane
wheel build definition now exists as its own releaseable artifact line.

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

## Upstream Version Upgrades

When upgrading to a new upstream LiteLLM version:

1. Fetch upstream tags: `git fetch upstream --tags`
2. Create new rebase branch: `git checkout <target-commit> -b aawm/patches-vX.Y.Z`
3. Cherry-pick AAWM commits in order (excluding any that were absorbed upstream).
4. Resolve conflicts (most likely in the patched files — check if upstream
   changed the target code blocks).
5. Update this file: version string, patch entries, dropped patches.
6. Tag: `git tag v{version}-aawm.{patch_count}`
