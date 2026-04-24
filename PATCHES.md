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
Current carried patch set: `aawm.2`, `aawm.3`, `aawm.4`, `aawm.5`, `aawm.6`, `aawm.7`, `aawm.8`, `aawm.9`, `aawm.10`, `aawm.11`, `aawm.12`, `aawm.13`, `aawm.14`, `aawm.15`, `aawm.16`, `aawm.17`, `aawm.18`, `aawm.19`, `aawm.20`, `aawm.21`, `aawm.22`, `aawm.23`, `aawm.24`, `aawm.25`, `aawm.26`, `aawm.27`, `aawm.28`, `aawm.29`, `aawm.30`, `aawm.31`, `aawm.32`, `aawm.33` (32 active carried patches)

**Working-tree note:** `develop` is the integration branch for the current
carried patch set. Promotion to `main` should happen only after the full
adapter harness and focused regression tests pass against the intended target.

**Version metadata note:** `pyproject.toml` should stay aligned to the last
carried patch set. `litellm/_version.py` now reflects the installed
distribution version directly. The current promotion target is
`1.82.3+aawm.33`.

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
### aawm.18 — Anthropic-route native adapter lane for OpenAI/Codex, OpenRouter, and Google Code Assist

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py`
- `litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py`

**Upstream issue:** Claude Code only speaks the Anthropic Messages API. AAWM needs
selected non-Anthropic models to be reachable through `/anthropic/v1/messages`
without Claude suspecting it is talking to anything other than Anthropic.

**Fix:**
1. Add allowlisted Anthropic-route adapter handling for:
   - OpenAI/Codex Responses models (`gpt-5.4`, `gpt-5.3-codex-spark`)
   - Google Code Assist / Gemini 3.1 models (`gemini-3.1*`)
   - OpenRouter-backed free-model targets
2. Translate Anthropic request/stream shapes into the native target provider
   request and translate the response/stream back into Anthropic-compatible
   output for Claude Code.
3. Reuse the provider-native local auth sources instead of inventing new
   server-side auth flows:
   - local Codex OAuth for OpenAI/Codex
   - local Google OAuth / Code Assist project routing for Gemini 3.1
   - `AAWM_OPENROUTER_API_KEY` for OpenRouter
4. Preserve Anthropic-compatible access-log labeling, adapter trace tags, and
   backend `session_history` attribution for the adapted calls.

**Why not upstream:** This is AAWM-specific multi-provider routing at the
Anthropic seam, driven by Claude Code compatibility and local-provider auth
reuse. Upstream LiteLLM does not provide a generic “pretend Anthropic, route to
other providers” control plane.

**Validation status:** Real Claude CLI traffic on `litellm-dev` succeeded on the
OpenAI/Codex lane, the OpenRouter lane, and the Google Code Assist Gemini lane
(`gemini-3.1-pro-preview` as the current validated baseline).

---

### aawm.19 — Provider-family egress guard, adapted access-log labeling, and adapter harness hardening

**Files:**
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/_logging.py`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** Once Anthropic-route traffic is adapted to other providers,
AAWM needs a hard guarantee that local provider credentials never leak to the
wrong egress target. It also needs operator-visible logging and a real-Claude
acceptance suite that tests the actual CLI path instead of only synthetic curl
traffic.

**Fix:**
1. Add a provider-family egress guard that pairs:
   - credential family (`openai`, `google`, `openrouter`, `anthropic`)
   - expected target family
   and blocks cross-provider sends locally before egress.
2. Add sticky local alerting and clearer adapted access-log labeling so a
   request can read like:
   `POST /anthropic/v1/messages?beta=true -> target native endpoint ...`
3. Add the standalone Anthropic adapter harness that shells out to the real
   Claude CLI on `:4001`, validates adapter trace tags/metadata, and checks
   backend `session_history` rows.
4. Treat upstream-volatile cases as warning-only canaries instead of forcing
   false hard failures:
   - Google Code Assist `gemini-3.1-pro-preview` due to quota windows
   - unstable OpenRouter free-model candidates
5. Mirror OpenRouter free-model cost to the paid OpenRouter twin wherever one
   exists, while keeping true zero-cost exceptions (`openrouter/free`,
   `openrouter/elephant-alpha`) at zero.

**Why not upstream:** This is AAWM-specific safety and validation policy tied to
local-provider auth reuse, our Anthropic-route adapter model, and our operator
workflow.

**Validation status:** Focused guard tests pass, real Claude CLI adapter runs on
`litellm-dev` show adapted access-log labeling, and the current harness now uses
hard-gate vs warning-canary semantics that match the real provider behavior.

---



### aawm.20 — Reduce local callback/control-plane latency and add lightweight runtime instrumentation

**Files:**
- `litellm/integrations/aawm_payload_capture.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** AAWM-specific observability and Claude control-plane work added measurable local overhead on every request even when those debugging surfaces were not actively needed. The main offenders were always-on payload capture, inline `session_history` persistence, and repeated Claude dynamic-injection DB hits.

**Fix:**
1. Gate payload capture behind both `AAWM_CAPTURE=1` and `LITELLM_LOG=DEBUG` so normal work does not pay JSON-serialization and disk-write cost.
2. Move `session_history` persistence to a background batch writer with tunables for batch size and flush interval.
3. Collapse the Claude system-prompt override and prompt-patch phases into a single traversal on the Anthropic request-prep path.
4. Add a short TTL cache for AAWM dynamic injection results keyed by proc/session/agent/tenant.
5. Add low-overhead instrumentation for the new hot-path controls:
   - batched `session_history` flush timing in DEBUG logs
   - dynamic injection cache hit/miss metadata and Langfuse span fields

**Why not upstream:** This is AAWM-specific performance tuning around our callback wheel, Claude control-plane rewrite path, and local observability workflow. It is not generic LiteLLM provider behavior.

**Validation status:** Focused callback tests (`18 passed`) and Anthropic passthrough/control-plane tests (`13 passed`) cover the batched `session_history` path, cached dynamic injection reuse, and the preserved Claude rewrite behavior.

---

### aawm.21 — Stabilize Anthropic-route OpenRouter free-model pacing and completion-lane attribution

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `docker-compose.dev.yml`

**Upstream issue:** The Anthropic-route OpenRouter free-model path had two AAWM-specific operational gaps:
1. free-model `429` handling could consume too much local hidden-retry time on every fresh manual retest, even after it was clear the same model/backend was still throttled
2. the special `openrouter/elephant-alpha` completion lane was not passing `proxy_server_request` context into `litellm.acompletion()`, so callback attribution could fall back to generic orchestrator traces instead of preserving Claude subagent/session context like the passthrough lanes

**Fix:**
1. Keep OpenRouter free-model retry/cooldown strictly model-scoped.
2. Use a short hidden retry budget on `:4001` for sporadic recovery and then a longer per-model post-failure cooldown so repeated manual runs fail fast instead of re-spending the full retry window.
3. Pass `proxy_server_request.headers/body` into the `elephant-alpha` completion lane so `AawmAgentIdentity`, Langfuse trace naming, and `session_history` enrichment have the same Claude subagent/session context as the passthrough routes.
4. Add focused regression coverage for the failure-circuit fast-fail path and the completion-lane request-context propagation.

**Why not upstream:** This is specific to AAWM's Anthropic-route OpenRouter control plane, free-model operator workflow, and Claude subagent attribution model.

**Validation status:** Focused OpenRouter retry/route tests pass after the failure-circuit and completion-lane context work (`21 passed` in the targeted slice), and `litellm-dev` is running the updated dev pacing on `:4001`.

---

### aawm.22 — Search-exact prompt context expansion and identifier-list rewrites

**Files:**
- `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** AAWM needs Claude prompts and dispatched child-agent prompts
to receive tenant/agent-scoped reference context for known technical topics.
The upstream Anthropic passthrough path has no concept of AAWM reference-store
lookups, context-marker escaping, or dynamically injected technical identifier
lists.

**Fix:**
1. Add `:#name.ctx#:` replacement backed by `tristore_search_exact`, preserving
   only the visible `name` in the prompt and appending retrieved context once
   per unique name in mention order.
2. Add literal escaping with `\\:#name.ctx#\\:` so prompts can show marker
   syntax without triggering a lookup.
3. Apply the same search-exact context appendix to dispatched child-agent
   prompts for single-backticked topics and bare all-caps acronyms while
   keeping misses silent.
4. Replace Claude Code's generic CommonMark system-prompt line with the
   tenant/agent-scoped list of known technical identifiers.
5. Extend the adapter harness to validate the `:#port-allocation.ctx#:` path,
   escaped marker preservation, and child-agent topic lookup behavior.

**Why not upstream:** This is AAWM-specific prompt enrichment tied to the local
tristore schema, tenant/agent scoping, and Claude Code agent-dispatch
conventions.

**Validation status:** The default Anthropic adapter harness on `:4001` now
includes hard gates for the ctx marker path, escaped marker path, and associated
Langfuse/session-history metadata.

---

### aawm.23 — Normalize session-history cache, reasoning, tool, and git telemetry

**Files:**
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `scripts/backfill_session_history.py`
- `scripts/repair_session_history_provider_cache.py`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`

**Upstream issue:** AAWM's `public.session_history` table is fork-local
observability state. It needs normalized provider attribution, reasoning-token
source semantics, provider-cache miss tracking, tool activity rollups, and git
command counters across Anthropic, OpenAI/Codex, Google, OpenRouter, and NVIDIA
traffic. Upstream LiteLLM does not persist or repair this AAWM-specific table.

**Fix:**
1. Infer missing providers from model/route metadata before persistence.
2. Prevent invalid `provider_reported` reasoning-token rows with zero reported
   reasoning tokens from being written.
3. Populate provider-cache attempted/miss/status/reason/token/cost fields from
   observed cache read/write counters and target-provider metadata.
4. Broaden tool-activity extraction for Codex/OpenAI streams and parent rollups.
5. Parse git command activity, including nested payloads and global options, so
   `git_commit_count` and `git_push_count` are populated.
6. Add repair/backfill scripts for existing rows and harden the harness checks
   so regressions fail before promotion.

**Why not upstream:** The `session_history` schema, repair policy, and AAWM
cost/cache attribution rules are local operational requirements, not generic
LiteLLM proxy behavior.

**Validation status:** Local repair normalized existing rows, focused callback
tests cover the writer changes, and the default harness now rejects null
providers, invalid reasoning sources, missing provider-cache status, and missing
tool/git rollups where expected.

---

### aawm.24 — NVIDIA NIM Anthropic-route adapter lane and cost mapping

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
- `model_prices_and_context_window.json`
- `litellm/bundled_model_prices_and_context_window_fallback.json`
- `scripts/local-ci/anthropic_adapter_config.json`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/test_cost_calculator.py`

**Upstream issue:** NVIDIA-hosted agent models from `~/.claude/agents` need to
work through Claude Code's Anthropic Messages interface while preserving NVIDIA
egress isolation, Langfuse metadata, session-history rows, and cost accounting.
Upstream does not provide an Anthropic-compatible NVIDIA NIM adapter lane.

**Fix:**
1. Add `anthropic_nvidia_completion_adapter` for direct `nvidia/...` agent
   models targeting NVIDIA's OpenAI-compatible `/v1/chat/completions` API.
2. Enforce NVIDIA provider-family egress checks separately from OpenAI,
   Google, OpenRouter, and Anthropic lanes.
3. Normalize NVIDIA/NIM model names for logging, session history, and cost-map
   lookup.
4. Add NVIDIA model pricing entries and fallback pricing based on OpenRouter
   equivalents when NVIDIA does not expose usable non-free pricing.
5. Add focused harness cases for NVIDIA DeepSeek, GLM, and MiniMax models while
   keeping high-latency MiniMax out of the default suite unless explicitly
   requested.

**Why not upstream:** This is AAWM-specific Anthropic-route provider adaptation
for Claude Code agents using local NVIDIA API credentials and AAWM telemetry.

**Validation status:** Focused `:4001` harness cases cover the NVIDIA lane,
with MiniMax documented as a slower upstream path using non-stream upstream
completion plus Anthropic-compatible fake streaming.

---

### aawm.25 — Dev/prod adapter harness targeting and promotion runtime process

**Files:**
- `Dockerfile.dev`
- `Makefile`
- `docker-compose.dev.yml`
- `docker-compose.yml`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `scripts/local-ci/README.md`
- `TEST_HARNESS.md`
- `CLAUDE.md`
- `GEMINI.md`

**Upstream issue:** AAWM promotion needs repeatable dev/prod validation against
the same harness without editing environment variables by hand. The fork also
needs a consistent minimal-effort Docker rebuild/restart process for
`litellm-dev` and clear defaults that avoid unnecessary paid-provider test
traffic.

**Fix:**
1. Expose the dev proxy on `:4001` while reserving `:4000` for production.
2. Add Docker management make targets for minimal rebuild/restart/log workflows.
3. Add `--target dev` and `--target prod` harness profiles with port,
   container, and Langfuse environment validation.
4. Exclude unstable/high-cost optional canaries such as Gemma and MiniMax from
   the default suite while keeping opt-in coverage available.
5. Preserve trace context metadata (`session_id`, `trace_environment`) on
   rewritten passthrough requests so prod/dev harness checks can validate the
   expected Langfuse environment and parent-session linkage.

**Why not upstream:** These are AAWM local-operations and promotion-process
requirements for the forked proxy, not general LiteLLM release behavior.

**Validation status:** The default adapter harness on `:4001` passed after the
target-profile update, and focused regression tests cover trace context
preservation for rewritten passthrough requests.

---

### aawm.26 — OpenAI Responses hardening, GPT-5.5, and runtime identity telemetry

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `model_prices_and_context_window.json`
- `litellm/bundled_model_prices_and_context_window_fallback.json`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `tests/test_litellm/`

**Upstream issue:** AAWM promotion uncovered additional local-pass-through
gaps that upstream LiteLLM does not cover for this fork: OpenAI Responses
stream reconstruction can receive reasoning-only output and nested object tool
schemas, GPT-5.5 needs an AAWM pass-through/cost-map entry before official API
pricing is available, and `public.session_history` needs runtime/client
identity for dev/prod attribution.

**Fix:**
1. Rebuild OpenAI Responses logging payloads correctly for reasoning-only
   completed responses and preserve Codex local-shell output activity.
2. Normalize nested OpenAI Responses tool schemas before upstream dispatch.
3. Add GPT-5.5 / ChatGPT GPT-5.5 cost-map entries and default harness coverage.
4. Add first-class `session_history` columns for LiteLLM environment,
   fork/runtime/wheel versions, and initiating client name/version/user-agent.
5. Harden prod-cutover harness checks so runtime exceptions, stale transfer
   headers, upstream passthrough 429/5xx tracebacks, missing runtime identity,
   and missing client identity fail before promotion.

**Why not upstream:** These are AAWM-specific pass-through adaptation,
telemetry, local cost-map, and prod-promotion requirements layered on top of
upstream LiteLLM.

**Validation status:** Focused callback, OpenAI passthrough, Responses, and
adapter-harness hardening tests pass locally. Full prod `:4000` harness
validation is required after promoting this patch set into the prod container.

---

### aawm.27 — Release tag alignment for main-head image promotion

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `WHEEL.md`

**Upstream issue:** The initial `v1.82.3-aawm.26` fork tag was created before
`main` had advanced to the final promotion merge, and the image publisher
intentionally rejects tags whose commit is not the current `main` head.

**Fix:** Bump the fork-local version to `1.82.3+aawm.27` so the production
image release tag can be cut from the converged `develop` / `main` head without
force-moving the already-published `aawm.26` tag.

**Why not upstream:** This is AAWM release-line bookkeeping for our guarded
GHCR image publishing workflow.

**Validation status:** Version/docs-only follow-up on top of the already
validated `aawm.26` code patch set.

---

### aawm.28 — Prod trace-user harness hardening

**Files:**
- `scripts/local-ci/run_acceptance.py`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `scripts/local-ci/harness-version.txt`
- `TEST_HARNESS.md`
- `COMPLETED.md`
- `TODO.md`

**Upstream issue:** The prod adapter harness must validate the user/session
headers that the harness itself controls. Ambient Claude project/user settings
can otherwise override `ANTHROPIC_CUSTOM_HEADERS`, making trace-user checks
look like a product failure when the local test runner is the source of drift.

**Fix:** Add a temporary per-run Claude `--settings` overlay for harness runs
so `x-litellm-end-user-id` and `langfuse_trace_user_id` are injected by the
harness and cannot be overridden by local operator settings. Basic OpenAI
adapter smoke cases validate routing, usage/cost, Langfuse, runtime logs, and
session history without requiring exact model text.

**Why not upstream:** This is AAWM promotion-harness behavior for our local
Claude/LiteLLM/Langfuse integration.

**Validation status:** Full prod `:4000` default adapter harness passed on
`1.82.3+aawm.28` before this follow-up OpenRouter stream hardening.

---

### aawm.29 — OpenRouter Responses missing-completed stream fallback

**Files:**
- `litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `scripts/local-ci/harness-version.txt`
- `tests/test_litellm/`

**Upstream issue:** Some OpenRouter Responses streams can terminate with
`response.output_text.done` and `[DONE]` without a `response.completed` event.
The upstream OpenAI passthrough logging path then cannot rebuild the final
response, causing zero usage/cost in Langfuse and `public.session_history`.
The Anthropic adapter stream can also finish without a final usage-bearing
`message_delta`.

**Fix:** Synthesize a valid Anthropic `message_delta` / `message_stop` with
estimated usage when an upstream Responses stream ends without
`response.completed`. The passthrough logging handler now builds a fallback
`ModelResponse` from streamed text, estimates usage from the request/output,
and calculates cost from the checked-in/bundled model-price JSON if
`completion_cost()` cannot resolve a newly added OpenRouter free-model entry.
The peeromega fanout harness timeout was raised to avoid treating slow real
multi-agent provider fanout as an adapter failure.

**Why not upstream:** This is defensive handling for the AAWM Anthropic ->
OpenRouter Responses adapter and local cost-map release cadence.

**Validation status:** Focused stream-wrapper and OpenAI passthrough logging
tests pass locally. Isolated prod `gpt-oss-120b`, peeromega fanout, NVIDIA, and
OpenRouter cases plus a full prod `:4000` harness run are required after
promoting this patch into the prod container.

---

### aawm.30 — Release tag alignment after harness artifact autobump

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `TODO.md`
- `TEST_HARNESS.md`
- `scripts/local-ci/README.md`

**Upstream issue:** The `v1.82.3-aawm.29` image tag was created before the
artifact autobump workflow advanced `main` for `h-v0.0.10`. The guarded image
publisher correctly rejected a tag that was no longer the current `main` head.

**Fix:** Bump the fork-local version to `1.82.3+aawm.30` on top of the
autobumped `main` head so the production image tag can be cut without
force-moving the already-pushed `aawm.29` tag.

**Why not upstream:** This is AAWM release-line bookkeeping for the guarded
GHCR image publishing workflow.

**Validation status:** Superseded by `aawm.31` after the next harness artifact
autobump moved `main` again.

---

### aawm.31 — Release tag alignment after second harness artifact autobump

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `TODO.md`
- `TEST_HARNESS.md`

**Upstream issue:** Updating `scripts/local-ci/README.md` in the `aawm.30`
alignment commit correctly triggered another harness artifact autobump, moving
the main head to `h-v0.0.11`.

**Fix:** Bump the fork-local version to `1.82.3+aawm.31` on the new main head
without touching `scripts/local-ci/**`, so the image tag can be cut from the
guarded publisher's expected commit.

**Why not upstream:** AAWM release-line bookkeeping for the local harness
artifact and fork-image publishing workflow.

**Validation status:** Version/docs-only follow-up on top of the `aawm.29`
OpenRouter stream fallback and `h-v0.0.11` harness artifact. Full prod `:4000`
harness validation is required after publishing and promoting the `aawm.31`
image.

---

### aawm.32 — Prod adapter validation hardening

**Files:**
- `pyproject.toml`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `TEST_HARNESS.md`
- `PATCHES.md`
- `TODO.md`
- `COMPLETED.md`

**Upstream issue:** The `aawm.31` prod promotion exposed two AAWM-specific
validation gaps:

1. NVIDIA completion-adapter calls wrote `session_history.litellm_environment`
   correctly but left Langfuse trace `environment` as `default` because normal
   completion callbacks only promote trace fields from `metadata.trace_*`, not
   from `litellm_metadata`.
2. `claude_adapter_gpt_oss_120b` remained the OpenRouter hard gate, but
   OpenRouter/Provider returned repeated `503` / `provider=OpenInference` /
   `raw=no healthy upstream` during prod validation. That provider-unavailable
   condition should not be confused with the prior local missing
   `response.completed` adapter bug.

**Fix:** Mirror trace/session context from `litellm_metadata` into normal
completion-adapter `metadata` for NVIDIA/OpenRouter completion adapters so
Langfuse trace fields match session-history environment fields. Add a narrow
harness classifier that soft-fails only timeout cases whose overlapping runtime
logs contain the exact OpenRouter provider-unavailable signature; all other
timeouts remain hard failures.

**Why not upstream:** AAWM-specific Anthropic adapter routing, local prod/dev
harness policy, and provider-free-model operational semantics.

**Validation status:** Focused unit coverage passes for the NVIDIA metadata
bridge and OpenRouter provider-unavailable timeout classifier. Full prod
`:4000` harness validation is required after publishing and promoting the
`aawm.32` image.

---

### aawm.33 — Preserve completion-adapter metadata through the Anthropic transformer

**Files:**
- `pyproject.toml`
- `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `PATCHES.md`
- `TODO.md`
- `COMPLETED.md`

**Upstream issue:** Prod `aawm.32` validation proved OpenRouter trace-environment
propagation, but NVIDIA completion-adapter traces still landed in Langfuse with
`environment=default`. The NVIDIA route passed `trace_environment=prod` to the
completion adapter, but the Anthropic-to-completion transformer only converted
Anthropic `metadata.user_id` into OpenAI `user` and dropped the remaining
metadata before the inner `litellm.acompletion()` call.

**Fix:** Preserve Anthropic completion-adapter metadata as LiteLLM completion
`metadata`, and merge explicit `litellm_metadata` into it when present. This
keeps trace/session fields available to Langfuse while retaining the existing
`metadata.user_id` -> `user` conversion. The harness also now classifies exact
OpenRouter `provider=OpenInference` / `raw=no healthy upstream` command failures
as provider-unavailable soft-fails, not just subprocess timeouts.

**Why not upstream:** AAWM-specific Anthropic adapter observability semantics
for NVIDIA/OpenRouter lanes and AAWM prod/dev harness policy.

**Validation status:** Focused metadata-regression and harness-hardening tests
pass locally. Publish/promote `aawm.33`, rerun focused prod NVIDIA/OpenRouter
cases, then run the default prod `:4000` harness.

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
