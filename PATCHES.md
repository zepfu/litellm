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
This file records the patch line and historical patch entries. Do not use the
top-level orientation as the source of truth for the currently promoted image or
overlay artifacts; those exact versions belong in `COMPLETED.md` release
evidence and in GitHub Release/tag state. When preparing a release, derive the
current fork version from `pyproject.toml`, the image tag from the fork release
tag, and overlay versions from the published `cb-v*`, `cp-v*`, `h-v*`, and
`cfg-v*` GitHub Releases.

**Working-tree note:** `develop` is the integration branch for the current
carried patch set. Promotion to `main` should happen only after the full
adapter harness and focused regression tests pass against the intended target.

**Version metadata note:** `pyproject.toml` should stay aligned to the last
carried patch set. `litellm/_version.py` now reflects the installed
distribution version directly. Treat `pyproject.toml` as the source for the
active fork-local version during release prep; do not duplicate a current
promotion target here because it goes stale after each promotion turn.

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
- Preserves tenant-only Langfuse user ids (`userId=<tenant_id>`) while keeping
  child agent identity in the trace name (`claude-code.<agent>`), not in the
  user id.
- Rewrites stale inbound `langfuse_trace_name: claude-code.orchestrator`
  headers to the child `metadata.trace_name` when Claude Code dispatches a
  subagent, while preserving unrelated explicit caller trace names.
- Preserves a workable attribution model without requiring Claude Code to emit
  dynamic per-subagent headers.

**Why not upstream:** AAWM-specific identity injection scheme; not a general
upstream feature.

**Note:** Includes the unused-import fix (removal of `Union` import) that was
previously a separate `chore` commit — squashed into this patch.
Production callback behavior is shipped from the overlay wheel, so changes here
must stay in parity with `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
and the published `cb-v*` artifact.

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

**Why not upstream:** This pass-through logging path is still carried locally on
the current v1.82.3-based AAWM release line.

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

### aawm.34 — Release tag alignment after harness artifact autobump

**Files:**
- `pyproject.toml`
- `scripts/local-ci/harness-version.txt`
- `PATCHES.md`
- `TODO.md`
- `COMPLETED.md`

**Upstream issue:** Publishing `aawm.33` to `main` correctly triggered the
independent harness artifact autobump to `h-v0.0.13`. The guarded image
publisher requires fork image tags to point at the current `main` head, so the
pre-autobump `v1.82.3-aawm.33` tag was rejected.

**Fix:** Fast-forward `develop` through the `h-v0.0.13` artifact bump and bump
the fork-local version to `1.82.3+aawm.34` on the new main head. This preserves
the published `aawm.33` tag while creating a valid image-publish tag for the
same metadata-preservation fix plus the harness artifact update.

**Why not upstream:** AAWM release-management guardrail for fork image tags and
independent harness artifacts.

**Validation status:** Version/docs-only follow-up on top of `aawm.33`; publish
and promote `aawm.34`, then rerun focused prod NVIDIA/OpenRouter cases and the
default prod `:4000` harness.

---

### aawm.35 — OpenRouter embedding and rerank proxy surface

**Files:**
- `pyproject.toml`
- `litellm/__init__.py`
- `litellm/_lazy_imports_registry.py`
- `litellm/main.py`
- `litellm/utils.py`
- `litellm/rerank_api/main.py`
- `litellm/llms/openrouter/embedding/transformation.py`
- `litellm/llms/openrouter/rerank/transformation.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `model_prices_and_context_window.json`
- `litellm/bundled_model_prices_and_context_window_fallback.json`
- `litellm-dev-config.yaml`
- `docker-compose.dev.yml`
- `OPENROUTER_EMBED_RERANK_CONSUMER.md`
- focused OpenRouter, cost, and session-history tests

**Upstream issue:** LiteLLM did not expose OpenRouter-hosted embedding/rerank
models needed by AAWM consumers through the local proxy surface in a way that
kept upstream billing centralized, preserved OpenRouter provider/cost metadata,
and wrote complete `public.session_history` rows for rerank calls that bill by
search unit instead of provider-reported tokens.

**Fix:** Add OpenRouter rerank provider support for `/api/v1/rerank`, normalize
OpenRouter embedding base URLs, pass OpenRouter provider routing through
embeddings, and include `AAWM_OPENROUTER_API_KEY` in OpenRouter credential
fallbacks. Add model metadata for `openrouter/qwen/qwen3-embedding-8b` and
`openrouter/cohere/rerank-4-pro`. Preserve caller-facing `openrouter/...`
models, OpenRouter upstream provider/model details, response cost, search-unit
usage, and estimated rerank prompt/total tokens in AAWM `session_history`.

**Why not upstream:** The provider surface is generally useful, but the
session-history attribution, tenant/session requirements, callback wheel
packaging, and centralized OpenRouter billing policy are AAWM-specific.

**Validation status:** Focused unit coverage passed for OpenRouter embedding
and rerank transforms, AAWM session-history rows, and exact cost-map coverage.
Live dev validation on `:4001` passed for `openrouter/qwen/qwen3-embedding-8b`
and `openrouter/cohere/rerank-4-pro`, including tenant/session attribution,
DeepInfra/Cohere provider metadata, non-zero costs, and estimated rerank tokens.

---

### aawm.36 — Release metadata alignment for the OpenRouter release

**Files:**
- `pyproject.toml`
- `PATCHES.md`

**Upstream issue:** The previously cut `v1.82.3-aawm.35` tag pointed at an
artifact-bump commit and still reported `1.82.3+aawm.34` from installed package
metadata. The guarded image publisher requires new fork image tags to point at
current `main`, and published tags must not be force-moved.

**Fix:** Bump the fork-local version to `1.82.3+aawm.36` for the OpenRouter
embedding/rerank promotion and leave the already-published `aawm.35` tag
untouched.

**Why not upstream:** AAWM release-line bookkeeping for fork image tags,
runtime identity in `session_history`, and the local image publisher guard.

**Validation status:** Version/docs-only release alignment on top of `aawm.35`.
After `main` is converged and the artifact autobump has published callback and
model-config assets, tag and publish `v1.82.3-aawm.36`, then promote through
the normal prod `:4000` process.

---

### aawm.37 — Claude-dispatched parallel tool dispatch and harness closure

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `pyproject.toml`
- `TODO.md`
- `COMPLETED.md`
- `PATCHES.md`

**Upstream issue:** Claude Code child-agent dispatch through LiteLLM's
Anthropic-compatible adapter needed durable proof that OpenAI/GPT, Gemini,
OpenRouter, and NVIDIA routes preserve sequential and parallel tool-call
semantics plus child trace identity. GPT-5.5 and OpenRouter Responses requests
also needed compact function-calling instruction policy when
`parallel_tool_calls=true`, while completion adapters needed to prefer child
trace metadata over stale orchestrator metadata.

**Fix:** Add reusable Responses-adapter parallel instruction policy metadata,
apply OpenRouter-specific context compaction and parallel policy, preserve child
trace names/user ids through completion-adapter metadata merging, and extend the
local Anthropic adapter harness with default-excluded OpenRouter, NVIDIA, and
Gemini 3.1 Pro parallel read-tool proofs. The release metadata now advances to
`1.82.3+aawm.37`.

**Why not upstream:** This is AAWM-specific Claude Code dispatch behavior,
Langfuse/session-history identity policy, local harness coverage, and provider
selection for `/anthropic` routes backed by OpenAI Responses, Google Code
Assist, OpenRouter, and NVIDIA.

**Validation status:** Focused proxy tests passed for child trace context,
trace-name precedence, context compaction, and parallel instruction policy
(`11 passed`). Harness hardening passed (`41 passed`), JSON validation passed,
and Ruff `E9,F821,F823` passed for touched Python files. Live dev `:4001`
artifacts passed for OpenAI/GPT-5.5
(`/tmp/claude_adapter_gpt55_child_parallel_read_tools_parallel_instruction_policy.json`),
OpenRouter/Ling plus NVIDIA/DeepSeek
(`/tmp/claude_adapter_openrouter_ling_nvidia_parallel_read_tools.json`,
`/tmp/claude_adapter_nvidia_parallel_read_tools_trace_fix.json`), Gemini 3
Flash parallel read tools
(`/tmp/claude_adapter_gemini3_flash_child_parallel_read_tools_rerun.json`), and
Gemini 3.1 Pro sequential plus parallel gates after quota reset
(`/tmp/claude_adapter_gemini31_pro_quota_reset_seq_parallel.json`). Prod `:4000`
was promoted to `v1.82.3-aawm.37` with `cb-v0.0.12`, `cp-v0.0.6`, and
`h-v0.0.21`. Focused prod validation passed at
`/tmp/litellm-prod-aawm37-cb12-focused-no-openrouter.json`. The default prod
harness was not a clean pass because `claude_adapter_peeromega_fanout` timed
out on the OpenRouter/Ling child lane; that follow-up is tracked in
`TODO.md`, not as a blocker for the already-proven child trace-name fix.

---

### aawm.38 — Codex `spawn_agent` fanout policy and prompt-overhead telemetry

**Files:**
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `scripts/local-ci/anthropic_adapter_config.json`
- `scripts/local-ci/run_anthropic_adapter_acceptance.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
- `TODO.md`
- `COMPLETED.md`
- `PATCHES.md`
- `TEST_HARNESS.md`
- `scripts/local-ci/README.md`

**Upstream issue:** Codex CLI can inject a `spawn_agent` tool description that
tells the model to use subagents only when the user explicitly asks for
subagents, delegation, or parallel agent work. That conflicts with AAWM's
operator policy, where concrete independent work should fan out by default
while the main thread keeps ownership of the critical path.

**Fix:** On Codex-native Responses passthrough requests, rewrite only structured
`tools[]` descriptors named `spawn_agent` when they contain the restrictive
wording. Replace that block with AAWM's fanout policy: use independent
subagents, keep one local critical-path owner, avoid duplicate work, prefer the
latest frontier model for architecture/migration-risk/high-stakes DB reasoning,
prefer the latest Codex model for bounded implementation with disjoint write
ownership, and use mini-class agents for narrow scans and quick QA. The rewrite
records `codex-tool-description-patch` /
`codex-tool-description-patch:spawn-agent-fanout-policy` tags,
`codex_tool_description_patch_*` metadata, and a
`codex.tool_description_patch` span.

The same development patch line also adds D1-060 prompt-overhead telemetry to
`public.session_history`. The callback now stores estimated system /
provider-equivalent, tool-advertisement, conversation, residual/other, and
system-classifier token buckets, and the adapter/native harness summarizes
those rows under `summary.prompt_overhead_cost_share` for route/client/model
comparisons. Counts and proportional cost-share dollars are explicitly labeled
estimated unless the provider reports exact component-level values.

**Why not upstream:** This is AAWM-specific Codex CLI context policy. The goal
is to shape our local Codex-native passthrough request before upstream send,
not to patch Codex CLI or alter generic OpenAI passthrough traffic. The
prompt-overhead reporting is AAWM-specific billing/observability analysis for
CLI prompt overhead.

**Validation status:** Local focused tests pass for the structured rewrite,
non-`spawn_agent` no-op behavior, Codex route integration, generic OpenAI
Responses no-op behavior, harness config shape, `pytest-classifier` caller
attribution, and passthrough callback ordering. Live dev `:4001` validation
passed after refreshing `litellm-dev`:
`AAWM_OBSERVE_SERVICE_NAME=pytest-classifier-scan ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config scripts/local-ci/anthropic_adapter_config.json --target dev --cases native_openai_passthrough_responses_codex --write-artifact /tmp/native_codex_spawn_agent_tool_description_patch.json`.
The artifact recorded trace `092a1ac4-1cd5-4859-949b-9898b7ba3b1c`, Langfuse
user `pytest-classifier`, the patch tags, required fanout text, and no
restrictive `Only use spawn_agent if and only if...` text.

Prompt-overhead validation also passed on dev `:4001`: unit coverage proves the
native and translated/billed request shapes are counted for Anthropic, OpenAI /
Codex Responses, Gemini / Code Assist, NVIDIA chat completions, OpenRouter chat
completions, and OpenRouter Responses; the native Codex harness assertion
populated `input_system_tokens_estimated`,
`input_tool_advertisement_tokens_estimated`,
`input_conversation_tokens_estimated`, `input_other_tokens_estimated`, and
`prompt_overhead_counted_shape=openai_responses`; and
`/tmp/native_codex_4001_prompt_overhead_cost_share.json` plus
`/tmp/native_openai_prompt_overhead_cost_share_4001.json` populated the harness
`summary.prompt_overhead_cost_share` report.

---

### aawm.39 — Local embed/rerank routes and explicit OpenRouter adapter wildcard

**Files:**
- `pyproject.toml`
- `litellm-dev-config.yaml`
- `model_prices_and_context_window.json`
- `litellm/bundled_model_prices_and_context_window_fallback.json`
- `litellm/integrations/aawm_agent_identity.py`
- `litellm/llms/base_llm/rerank/transformation.py`
- `litellm/llms/huggingface/rerank/transformation.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `tests/test_litellm/test_cost_calculator.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `LOCAL_EMBED_RERANK_CONSUMER.md`

**Upstream issue:** AAWM now runs local TEI, Nomic code embedding, and BGE
rerank services that should be consumed through LiteLLM rather than direct
service ports, while still producing usable `session_history` provider/model
and cost rows. Claude adapter routing also needed a low-touch path for explicit
`openrouter/*` model names that are not hardcoded in the local allowlist.

**Fix:** Add local provider/cost-map entries under `local_embed/*` and
`local_rerank/*`, including MedCPT article/query, SPECTER2, Indus, SapBERT,
Nomic code embeddings, and `BAAI/bge-reranker-v2-m3`. Dev config exposes those
routes through local TEI/Nomic/rerank services and applies estimated
commercial-equivalent prices so `session_history.response_cost_usd` remains
populated. The local consumer doc records the proxy-facing aliases and
required attribution headers.

The same patch line allows explicitly prefixed `openrouter/*` Anthropic
adapter model requests to route through the OpenRouter Responses adapter after
normalization, without changing the separate OpenRouter embedding/rerank
process.

**Why not upstream:** The local service topology, pricing estimates,
consumer-facing aliases, and `session_history` attribution policy are
AAWM-specific. The OpenRouter wildcard is also scoped to AAWM's Claude adapter
routing policy.

**Validation status:** Focused local provider, cost, and session-history unit
tests passed, including local rerank token/cost attribution and Nomic
embedding metadata. The pass-through endpoint test module passed
(`267 passed`) after the explicit `openrouter/*` adapter routing change. Full
dev harness on `:4001` passed all non-Codex lanes in
`/tmp/litellm-dev-harness-2026-05-04-openrouter-wildcard.json`; the remaining
red cases were the known Codex `gpt-5.3-codex-spark` `usage_limit_reached`
path with reset at `2026-05-05T22:08:30Z`.

---

### aawm.40 — Explicit NVIDIA adapter wildcard for early model testing

**Files:**
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** Early NVIDIA NIM chat testing should not require a full
LiteLLM config/model allowlist buildout for every newly exposed NVIDIA model.
The existing NVIDIA adapter path only routed locally allowed bare model names,
while explicit `nvidia/*` operator intent was not enough by itself.

**Fix:** Treat literal `nvidia/*` Anthropic/Claude adapter model requests as an
explicit NVIDIA NIM target and forward the stripped upstream model after the
existing alias normalization. Unknown bare names, `nvidia_nim/*`, and
`openrouter/*` do not receive this wildcard behavior. Existing OpenRouter
namespace models such as `nvidia/nemotron-3-super-120b-a12b:free` remain on the
OpenRouter adapter instead of being hijacked by the NVIDIA resolver.

**Why not upstream:** This is an AAWM Claude adapter routing policy for local
operator-driven model trials.

**Validation status:** The full pass-through endpoint test module passed
(`273 passed`) after the resolver and route coverage were added. Focused live
Claude CLI validation on dev `:4001` used the current NVIDIA free endpoint
model `nvidia/qwen/qwen3-coder-480b-a35b-instruct` and passed with artifact
`/tmp/nvidia-qwen3-coder-wildcard-cli-4001.json`. The run dispatched a real
Claude Code child agent, recorded `provider=nvidia_nim` and
`model=qwen/qwen3-coder-480b-a35b-instruct` in `public.session_history`, and
proved parallel `Read` / `Glob` / `Grep`, `Bash`, and `Write` tool calls.

---

### aawm.41 — Post-autobump release candidate retag

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `PROD_RELEASE.md`
- `TODO.md`
- `COMPLETED.md`
- `TEST_HARNESS.md`
- `.analysis/todo.md`
- `.analysis/completed.md`

**Upstream issue:** The `v1.82.3-aawm.40` tag was cut before the GitHub artifact
autobump advanced `main` to `h-v0.0.26`. The fork image workflow requires the
tagged commit to be reachable from current `main`, and the release runbook says
to cut a new image tag from the new `main` head instead of force-moving an
already-pushed release tag.

**Fix:** Bump the fork metadata and release docs to `1.82.3+aawm.41`, preserve
`aawm.40` as a stale pre-publication candidate, and document `h-v0.0.26` as the
current harness overlay release for the pending prod promotion.

**Why not upstream:** This is AAWM release orchestration and artifact-version
bookkeeping.

**Validation status:** The missing `h-v0.0.26` GitHub Release asset was built
from `scripts/local-ci/harness-version.txt` and published as
`litellm-local-ci-harness-0.0.26.tar.gz` before the `aawm.41` source candidate
was cut. This candidate was superseded before image publication when the
release-doc commit's `scripts/local-ci/README.md` change triggered the next
harness autobump to `h-v0.0.27`.

---

### aawm.42 — Post-harness-0.0.27 release candidate retag

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `PROD_RELEASE.md`
- `TODO.md`
- `COMPLETED.md`
- `TEST_HARNESS.md`
- `.analysis/todo.md`
- `.analysis/completed.md`

**Upstream issue:** The `v1.82.3-aawm.41` tag was cut before the subsequent
`h-v0.0.27` harness autobump advanced `main`. The fork image workflow requires
the tagged commit to be reachable from current `main`.

**Fix:** Bump the fork metadata and release docs to `1.82.3+aawm.42`, preserve
`aawm.41` as a stale pre-publication candidate, and document `h-v0.0.27` as the
current harness overlay release for the pending prod promotion.

**Why not upstream:** This is AAWM release orchestration and artifact-version
bookkeeping.

**Validation status:** The missing `h-v0.0.27` GitHub Release asset was built
from `scripts/local-ci/harness-version.txt` and published as
`litellm-local-ci-harness-0.0.27.tar.gz` before the `aawm.42` source candidate
was cut.

---

### aawm.43 — Provider-originated rate-limit observations and local Qwen route

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `PROD_RELEASE.md`
- `LOCAL_LLM_CONSUMER.md`
- `LOCAL_EMBED_RERANK_CONSUMER.md`
- `litellm-dev-config.yaml`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `litellm/integrations/aawm_passthrough_shape_capture.py`
- `litellm/proxy/pass_through_endpoints/google_code_assist_quota.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/proxy/pass_through_endpoints/streaming_handler.py`
- `litellm/proxy/pass_through_endpoints/success_handler.py`
- `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
- `scripts/backfill_rate_limit_observations.py`
- `tests/test_scripts/test_backfill_rate_limit_observations.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`

**Upstream issue:** AAWM needs locally observed quota-window state for Codex,
Claude, and Gemini without relying on status-line side channels or fixed
provider reset schedules. The production proxy also needs to expose the new
local `qwen3-heretic-gguf` LLM route separately from local embedding/rerank
models.

**Fix:** Capture provider-originated quota observations from Codex/OpenAI
response headers, Anthropic response headers, Google Code Assist
`retrieveUserQuota` payloads, and provider error payloads into the processed
`public.rate_limit_observations` stream. Keep unchanged snapshots out of the
table, normalize provider/client/quota keys, and preserve transition evidence
for later reset-window inference. Codex/OpenAI response-header observations
parse `x-codex-*-used-percent` into stored `remaining_pct` when the provider
emits those percentage headers. Add sanitized pass-through shape capture for
debugging, a historical backfill utility, and a dev-only chat route for
`qwen3-heretic-gguf`.

**Why not upstream:** This is AAWM-specific observability, local routing, and
consumer documentation for the AAWM LiteLLM deployment.

**Validation status:** Focused unit and script validation covers the callback
extractors, processed insert guards, pass-through metadata plumbing, Google
quota observation-only logging, and historical backfill. Live dev `:4001`
validation proved Codex, Claude, and Gemini provider-originated rows in
`aawm_tristore.public.rate_limit_observations`; Sonnet-specific
`7d_sonnet` rows are only expected on Sonnet traffic because Anthropic only
emits that header family for Sonnet responses. The local
`qwen3-heretic-gguf` route was smoke-tested through the dev proxy on
`/v1/chat/completions` and returned `OK`. A follow-up focused unit check
verified Codex response-header `used-percent` values populate
`remaining_pct`; older saved shape captures only retained the used-percent
header names, not the values, so existing null OpenAI/Codex rows cannot be
accurately backfilled from those artifacts.

---

### aawm.44 — Codex Spark hosted-tool sanitizer and session response-gap telemetry

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `TODO.md`
- `WHEEL.md`
- `PROD_RELEASE.md`
- `docker-compose.dev.yml`
- `model_prices_and_context_window.json`
- `litellm/bundled_model_prices_and_context_window_fallback.json`
- `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/proxy/response_api_endpoints/endpoints.py`
- `litellm/types/utils.py`
- `litellm/utils.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `scripts/backfill_session_history_latency.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py`

**Upstream issue:** Native Codex traffic for `gpt-5.3-codex-spark` can advertise
the OpenAI hosted `image_generation` tool even though the Codex Spark upstream
rejects that tool. AAWM also needs a `public.session_history` metric that
captures the delay between a previous response finishing and the next request
entering processing for the same session.

**Fix:** Mark Codex Spark as not supporting `image_generation` hosted tools,
sanitize that hosted tool out of native Codex Responses requests on both the
generic passthrough and first-class Responses paths, preserve normal function
tools, and retain auditable metadata for removed hosted tools. Add
`previous_response_to_current_request_ms` to `public.session_history`, derive it
from the immediately previous same-session row when writes are persisted, and
extend the latency backfill script with a separate gap-metric pass for existing
rows.

**Why not upstream:** The Codex hosted-tool filter is tied to AAWM's native
Codex passthrough deployment shape and current model catalog policy. The
session-history response-gap field is AAWM tristore telemetry.

**Validation status:** Focused unit coverage verifies the hosted-tool sanitizer
for passthrough and first-class Responses routes, non-Spark behavior, preserved
function tools, and session-history metadata retention. Dev `:4001` live smoke
for `codex exec --profile litellm-dev -m gpt-5.3-codex-spark "just a test msg"`
completed without the upstream image-generation 400. Focused session-history
tests verify the widened payload and post-insert gap update; static checks
covered the callback sources and the latency backfill script. The current
release candidate should be treated as a new base fork candidate because the
Codex sanitizer touches core passthrough/Responses code and model metadata;
main-branch artifact automation is expected to advance the callback,
control-plane, and model-config overlay lines before the infrastructure rebuild.

---

### aawm.45-aawm.48 — Codex-native Gemini Code Assist adapter and child-agent hardening

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `TODO.md`
- `COMPLETED.md`
- `PROD_RELEASE.md`
- `docker-compose.dev.yml`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `litellm/integrations/langfuse/langfuse_prompt_management.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_litellm/integrations/langfuse/test_langfuse_prompt_management.py`

**Upstream issue:** AAWM needs normal Codex sessions to dispatch Gemini child
agents through the existing LiteLLM proxy using Google Code Assist OAuth,
quota observation, and session-history attribution. The initial prod release
also exposed two hardening gaps: Codex can request `reasoning_effort=xhigh`,
which Google Code Assist does not accept, and Gemini child agents can confuse
terminal tool results for the next tool-call arguments after a tool-result
turn.

**Fix:** Add a Codex-native Google Code Assist adapter path that translates
OpenAI Responses/chat-shaped Codex requests into native
`streamGenerateContent` requests while preserving the Gemini Code Assist OAuth
token path and quota preflight. Normalize Codex `xhigh` effort to Google
`high`, keep Langfuse prompt-management logging tolerant of missing dynamic
callback parameters, and append a Codex-specific tool contract for native
Gemini Code Assist requests. The tool contract instructs Gemini to treat tool
results as observations, construct later function-call arguments from the
declared schema, and never copy prior terminal transcript text such as
`Chunk ID`, `Wall time`, or `Output:` into executable tool arguments.

**Why not upstream:** This is AAWM-specific Codex/Gemini account routing,
Google Code Assist OAuth reuse, and child-agent operational hardening for the
AAWM proxy deployment.

**Validation status:** `aawm.47` is live on prod with the Codex-native Gemini
adapter, `xhigh` normalization, and Langfuse missing-dynamic-params guard.
`aawm.48` adds the tool-contract hardening and callback metadata parity. Dev
`:4001` validation proved `codex exec --profile litellm-dev -m
gemini-3.1-flash-lite-preview` can run a real shell tool through Gemini, writes
`provider=gemini` / `model=gemini-3.1-flash-lite-preview` rows with
`codex_google_code_assist_tool_contract_policy=append`, and records no
terminal-transcript-shaped malformed Gemini command rows after the smoke.

---

### aawm.49 — Session-history attribution/performance, local biomedical attribution, and Codex auto-agent fallback

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `WHEEL.md`
- `litellm-dev-config.yaml`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`

**Upstream issue:** AAWM needs native Codex/proxy sessions to preserve the
current workspace repository in `session_history`, keep the recurring
session-history/rate-limit usage report fast enough for pgAdmin use, record
local biomedical REST pass-through calls as local biomedical usage, and give
normal Codex child-agent sessions a single model alias that can move across the
Spark/Gemini preview lanes before falling back to `gpt-5.4-mini`.

**Fix:** Prioritize current `<environment_context><cwd>...</cwd>` values and
scan list-shaped request payloads newest-first when deriving repository
identity. Add a `session_history.created_at` index, rate-limit observation
indexes, and a durable `public.rate_limit_intervals` materialized view for the
no-temp usage report path. Add dev pass-through routes for local scispaCy and
TinyBERN2 REST services and classify those route writes in `session_history` as
`provider=local_biomed`. Add the `aawm-codex-agent-auto` OpenAI Responses alias
for Codex agent traffic; it selects Spark first, then the three Gemini preview
models through the Codex Google Code Assist adapter, and uses `gpt-5.4-mini` as
the final fallback only after the preferred lanes are exhausted. The alias
records selected-target/attempt metadata and applies bounded cooldowns for
`429`, `usage_limit_reached`, `RESOURCE_EXHAUSTED`, and
`MODEL_CAPACITY_EXHAUSTED` responses.

**Why not upstream:** These are AAWM deployment contracts: repository
attribution from Codex/Gemini workspace context, AAWM tristore reporting
objects, local biomedical sidecar routing, Google Code Assist OAuth reuse, and
the AAWM-specific Codex agent model policy.

**Validation status:** Dev validation on `:4001` proved the repository
attribution fix with an `aegis-dashboard` Codex session and a historical repair
of `2,726` semantically misattributed rows in exact database `aawm_tristore`.
The reusable `public.rate_limit_intervals` view and indexes reduced the
representative no-temp report from minute-scale interval joins to about four
seconds under `EXPLAIN ANALYZE`, and pg_cron jobs now refresh/analyze it every
30 minutes in the live database. Local scispaCy pass-through smoke wrote a
`provider=local_biomed`, `model=scispacy` row; TinyBERN2 health passes through
LiteLLM, but live annotation remains blocked by missing sidecar model-id
configuration outside LiteLLM. The Codex auto-agent alias dev smoke wrote
`provider=gemini`, `model=gemini-3-flash-preview` after Spark and flash-lite
reported quota exhaustion. Focused pass-through and session-history unit tests
cover the new selection, metadata, repository, local biomedical, and materialized
view behavior.

---

### aawm.50 — Anthropic rate-limit observation capture and backfill

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `litellm/proxy/pass_through_endpoints/success_handler.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `scripts/backfill_rate_limit_observations.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_scripts/test_backfill_rate_limit_observations.py`

**Upstream issue:** AAWM relies on `public.rate_limit_observations` for provider
quota reporting. Anthropic streaming passthrough responses already preserved
sanitized `anthropic-ratelimit-*` headers, but non-stream Anthropic passthrough
successes did not copy upstream rate-limit headers into callback metadata. Some
LiteLLM callback payloads also carried Anthropic headers under hidden response
fields as `llm_provider-anthropic-ratelimit-*`, which the AAWM extractor did
not recognize.

**Fix:** Record sanitized Anthropic non-stream response rate-limit headers in
callback metadata before passthrough response normalization. Expand the AAWM
rate-limit extractor to inspect hidden response/header containers, handle
mapping-like header objects, and accept both raw `anthropic-ratelimit-*` and
LiteLLM-prefixed `llm_provider-anthropic-ratelimit-*` names. Expand the
ClickHouse backfill marker scan so retained Anthropic header metadata can be
replayed into `public.rate_limit_observations`.

**Why not upstream:** This is AAWM-specific quota telemetry and historical
repair logic for the local `aawm_tristore` reporting schema.

**Validation status:** Focused unit coverage proves non-stream Anthropic
passthrough header capture, hidden/prefixed Anthropic header extraction, error
header extraction, and the expanded backfill markers. The live backfill was run
against exact database `aawm_tristore` for retained ClickHouse observations
from `2026-05-05T00:00:00Z` through `2026-05-14T14:35:00Z`, then
`public.rate_limit_intervals` was refreshed and analyzed.

---

### aawm.51 — Provider-health failure observations and active probe schema

**Files:**
- `litellm/integrations/aawm_agent_identity.py`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `scripts/record_provider_status_observations.py`
- `scripts/backfill_provider_error_observations_from_docker_logs.py`
- `scripts/aawm-provider-status-observations.service`
- `scripts/aawm-provider-status-observations.timer`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
- `tests/test_scripts/test_record_provider_status_observations.py`

**Upstream issue:** AAWM needs provider-health telemetry that separates passive
traffic failures from active network probe degradation. LiteLLM pass-through
failures were visible in process logs, but the AAWM callback only persisted
quota-oriented failure signals and prod Anthropic overloads such as `529`
could disappear on restart. The pass-through failure path could also hand
callbacks a recursive request payload, blocking failure observation writes.

**Fix:** Add `public.provider_error_observations` and
`public.provider_status_observations` bootstrap/persistence to the AAWM
callback. Failure callbacks now persist normalized provider, model, route,
status, error type/code/class, reset hint, session, trace, and call metadata.
The pass-through failure payload copies the parsed body before merging callback
kwargs so `passthrough_logging_payload.request_body` cannot point back to the
same request-data object. Add an active status collector for DNS, TCP, TLS,
ICMP, and control-baseline probes, plus a Docker-log preservation backfill for
restart windows where the running prod callback did not yet persist provider
errors.

**Why not upstream:** This is AAWM-specific observability and historical
preservation for the local `aawm_tristore` reporting schema.

**Validation status:** Focused unit coverage proves provider-error
classification, recursive pass-through failure handling, provider-error insert
payloads, and active probe row construction. Local/dev runtime smokes wrote
real `provider_error_observations` and `provider_status_observations` rows in
exact database `aawm_tristore`; retained prod Docker-log errors were preserved
before the prod restart.

---

### aawm.52 — OpenRouter direct routing and in-flight Codex auto-agent cooldowns

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
- `litellm/litellm_core_utils/get_llm_provider_logic.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `litellm-dev-config.yaml`
- `docker-compose.dev.yml`
- `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
- `tests/test_litellm/llms/openrouter/test_openrouter_provider_routing.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`

**Upstream issue:** AAWM needs OpenRouter-prefixed model IDs to pass through
without a second provider-prefix rewrite, and Codex auto-agent retries must not
switch providers once an agent attempt is already carrying stateful
continuation data. Mid-session provider switching can strand tool calls,
reasoning items, or `previous_response_id` continuations on a different
provider than the one that created the state.

**Fix:** Preserve regular OpenRouter model IDs when the caller already selected
`custom_llm_provider="openrouter"`, add direct/wildcard OpenRouter routes for
the Qwen flash models, and allow the Anthropic-to-OpenRouter Responses adapter
to target `qwen/qwen3.5-flash-02-23` and `qwen/qwen3.6-flash`. For
`aawm-codex-agent-auto`, detect stateful continuation markers and session
affinity before retry selection; if the selected provider hits quota/capacity
while an agent is in flight, set the usual cooldown and surface that provider
error instead of selecting a different provider. If a subsequent continuation
arrives while the affinity target is already cooling down, return HTTP `429`
with code `aawm_codex_auto_agent_in_flight_provider_cooling_down` so the
orchestrator can redispatch a fresh agent attempt.

OpenRouter session-history cost extraction now prefers provider-reported
`usage.cost` / `usage_openrouter_cost` before generic LiteLLM zero/catalog
costs for OpenRouter rows.

**Why not upstream:** These are AAWM-specific routing, callback, and Codex
agent orchestration contracts layered on top of LiteLLM's generic provider
routing and retry behavior.

**Validation status:** Focused unit coverage passes for OpenRouter model
preservation, the Anthropic OpenRouter adapter allowlist, OpenRouter reported
cost precedence, and Codex auto-agent in-flight terminal cooldown behavior.
Live dev validation on `litellm-dev` `:4001` proved the new Qwen OpenRouter
routes and callback writes before prod promotion.

---

### aawm.53 — Codex auto-agent continuation scanner crash guard

**Status:** AAWM local hotfix.

**What changed:** Harden `_codex_auto_agent_request_has_continuation_state()` so
it treats continuation `type` markers only when they are strings and skips
already-seen dict/list objects while walking request payloads.

**Why:** Prod showed `aawm-codex-agent-auto` could crash before candidate
fallback when a request contained a nested dict-valued `type` field, raising
`TypeError: unhashable type: 'dict'`. That prevented the alias from reaching
its normal first-call provider fallback behavior.

**Why not upstream:** This scanner is part of AAWM's local Codex auto-agent
alias and in-flight provider-stickiness policy.

**Validation status:** Live dev `litellm-dev` on `:4001` returned a normal
provider-shaped `400 invalid_type` for the formerly crashing payload shape, and
focused `codex_auto_agent` tests passed.

---

### aawm.54 — xAI/Grok provider-health capture

**Files:**
- `pyproject.toml`
- `PATCHES.md`
- `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
- `litellm/integrations/aawm_agent_identity.py`
- `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
- `scripts/record_provider_status_observations.py`
- `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
- `tests/test_litellm/integrations/test_aawm_agent_identity.py`
- `tests/test_scripts/test_record_provider_status_observations.py`

**Upstream issue:** AAWM's native Grok Build `/grok` passthrough route could
record successful `session_history` rows, but xAI/Grok upstream failures were
not guaranteed to reach `public.provider_error_observations`. The active
provider-status collector also did not probe xAI front doors, leaving the
provider-health overlay blind to xAI DNS/TCP/TLS/ICMP behavior.

**Fix:** Add `cli-chat-proxy.grok.com:443` and `api.x.ai:443` to the active
provider-status collector. Enrich xAI/Grok passthrough failure payloads with
provider, route-family, API base, and `x-grok-model-override` metadata, then
directly call the AAWM failure callback for xAI passthrough failures in
addition to the normal LiteLLM failure hook. Make provider-error persistence
idempotent for duplicate callbacks by deduping rows with the same
`litellm_call_id`, provider, route family, and status code.

**Why not upstream:** This is AAWM-specific native Grok Build routing and
provider-health telemetry for the local `aawm_tristore` reporting schema.

**Validation status:** Focused unit coverage proves xAI default active-probe
endpoints, Grok failure classification, direct xAI passthrough failure capture,
and idempotent provider-error SQL. Live dev validation on `litellm-dev` `:4001`
inserted an exact `aawm_tristore.public.provider_error_observations` row for a
controlled `/grok/v1/responses` upstream `401` with `provider=xai`,
`model=grok-build`, `error_class=auth_failed`, and
`route_family=grok_cli_chat_proxy`, with one row for the call id after dedupe.

---

### aawm.55 — xAI/Grok provider-health prod overlay import fallback

**Status:** AAWM local hotfix.

**What changed:** The xAI/Grok direct failure-capture path now resolves the
AAWM agent-identity callback from either the source-tree module
`litellm.integrations.aawm_agent_identity` or the production overlay module
`aawm_litellm_callbacks.agent_identity`.

**Why:** The `v1.82.3-aawm.54` fork image correctly included the direct xAI
capture path, but production packaging excludes the source-tree integration
module and installs the callback wheel as an overlay. Prod therefore logged
`No module named 'litellm.integrations.aawm_agent_identity'` when the direct
capture fallback ran for `/grok` failures. The normal callback registration
still uses the overlay wheel, so the direct path must use the same package when
the source-tree module is not present.

**Why not upstream:** This is specific to the AAWM split between fork image
core code and overlay callback wheel packaging.

**Validation status:** Focused unit coverage proves the direct xAI passthrough
failure path falls back to `aawm_litellm_callbacks.agent_identity` when the
source integration module is missing.

---

### aawm.57 — Codex auto-agent fresh-dispatch affinity fallback

**Status:** AAWM local hotfix.

**What changed:** `aawm-codex-agent-auto` no longer treats every
`session_affinity` retryable exhaustion as terminal. If a request has
Responses continuation state, provider `429`/quota/capacity exhaustion still
sets cooldown and surfaces the provider error so the orchestrator can
redispatch. If the request is a fresh dispatch, the alias can skip a cooled
session-affinity target and continue through the preferred candidates before
using the `gpt-5.4-mini` last-resort fallback.

**Why:** Recent child-agent dispatches were pinned to
`gpt-5.3-codex-spark` via session affinity. When Spark later returned
`usage_limit_reached`, the alias raised immediately because the selection
reason was `session_affinity`, even for fresh agent dispatches that could have
fallen through to Gemini or the final `gpt-5.4-mini` fallback. In-flight
sessions should still fail fast on provider exhaustion because their tool calls,
reasoning items, and `previous_response_id` state belong to the selected
provider family.

**Why not upstream:** This is an AAWM-specific Codex agent orchestration policy
layered on top of native Codex and Google Code Assist pass-through routing.

**Validation status:** Focused unit coverage proves a fresh dispatch with
stale Spark affinity reaches `gpt-5.4-mini` after Spark and all Gemini
candidates return retryable exhaustion, while existing in-flight affinity tests
still prove continuation failures are terminal redispatch signals. Live dev
validation on `litellm-dev` `:4001` showed Spark return
`usage_limit_reached`, the alias set Spark cooldown, and the same fresh request
completed on `gemini-3.1-flash-lite-preview` with
`requested_model_alias=aawm-codex-agent-auto` in `session_history`.

---

### aawm.58 — Claude auto-review telemetry attribution

**Status:** AAWM local hotfix.

**What changed:** Claude permission-check traffic is now persisted as the
logical model `claude-auto-review` with `agent_name=auto-reviewer`, while
preserving the provider source model for cost/reference metadata. Langfuse tag
collection now also reads metadata-level request/trace tags so permission-check
tags survive callback ordering, and the bundled model-cost map includes the
`claude-auto-review` alias for offline reporting. A bounded backfill script can
repair historical permission-check rows without recalculating already stored
cost.

**Why:** Claude Code auto-review/permission checks were being mixed into normal
Claude session history, often with ephemeral agent-style repository values.
That made session-level attribution and downstream review of real work harder,
especially when permission checks appeared inside active development sessions.

**Why not upstream:** This is AAWM-specific telemetry classification for the
local `session_history`, Langfuse tagging, and Claude Code auto-review reporting
model.

**Validation status:** Focused callback/backfill coverage passed locally
(`181 passed`) with the callback wheel source synced to the in-repo callback.
Langfuse tag unit coverage passed, model cost maps are synced, JSON checks
passed, and the dev runtime/backfill evidence is recorded in `COMPLETED.md`.

---

### aawm.60 — Codex auto-agent ordered fallback and empty-success rollover

**Status:** AAWM local hotfix.

**What changed:** `aawm-codex-agent-auto` now treats retryable fresh-dispatch
failures as per-candidate cooldowns in the explicit candidate order:
Codex Spark, each Gemini candidate individually, OpenRouter
`deepseek/deepseek-v4-flash:free`, then `gpt-5.4-mini` as the last resort.
Default retryable cooldowns are approximately three hours, with longer
provider reset hints still honored. OpenRouter DeepSeek Responses payloads
that complete successfully but contain no meaningful output and report
`output_tokens <= 1` are converted into retryable candidate failures so fresh
dispatches roll forward instead of returning an unusable child result.

**Why:** Fresh child dispatches should always start from the top of the alias
candidate list and use the first non-cooled working model. In-flight affinity
must remain scoped to the exact active session/thread, but active agents on
Gemini, DeepSeek, or `gpt-5.4-mini` must not pin unrelated future fresh
dispatches after Spark or an earlier Gemini candidate comes off cooldown.

**Why not upstream:** This is an AAWM-specific Codex agent orchestration policy
layered on top of native Codex, Google Code Assist, and OpenRouter
pass-through routing.

**Validation status:** Focused alias coverage proves the ordered fallback, 529
retryability, per-candidate Gemini cooldowns, DeepSeek empty-success rollover
to `gpt-5.4-mini`, and legitimate one-token text success behavior.

---

### aawm.61 — Demote routine LiteLLM operational log noise

**What changed:** Demoted routine success/bookkeeping logs to debug level:
Langfuse header overwrite notices, per-request shared-session attachment,
Router `200 OK` success lines, adapter upstream attempt counters, Google
post-tool cooldown bookkeeping, and Anthropic/Vertex managed-object/model-id
bookkeeping. Failure, retry/backoff, cooldown-active, telemetry-loss, and
fallback warnings remain visible at warning level.

**Why:** Dev and prod both run with `LITELLM_LOG=INFO`; normal request flow was
producing high-volume INFO/WARNING lines that made actionable errors harder to
see in Docker logs.

**Why not upstream:** The exact noisy paths are tied to AAWM pass-through,
adapter, and operational observability behavior in this fork.

**Validation status:** `py_compile` passed for all touched modules and
`litellm-dev` was rebuilt/recreated successfully with `/health` returning HTTP
200 before prod promotion.

---

### aawm.62 — DeepSeek chat-shape auto-agent routing and observability repair

**Status:** AAWM local hotfix.

**What changed:** The Codex and Anthropic auto-agent aliases now route
OpenRouter `deepseek/deepseek-v4-flash:free` through a chat/completions adapter
instead of the OpenRouter Responses adapter. The Codex alias still exposes a
Responses-shaped result to the caller, but the upstream request uses the
request shape accepted by the free DeepSeek target. Gemini Code Assist
empty-success responses with no meaningful output and no more than one output
token are also treated as retryable alias candidate failures.

The release also tightens session-history observability around the same traffic:
Langfuse and `session_history` preserve explicit `openrouter/...` model names
when the caller passed one, unmapped OpenRouter cost calculation preserves usage
instead of dropping the row, Gemini stream logging parses multiple SSE JSON
objects per chunk, noisy repository placeholders/transcript artifact names are
rejected, and Langfuse payload-size warnings now identify the fields pushing an
event near the SDK size limit without logging sensitive metadata values.

**Why:** OpenRouter DeepSeek free was returning empty one-token successes from
the Responses-shaped path. Treating that as a successful child agent left
dispatches with no useful work. At the same time, native and repaired
session-history rows exposed gaps in model attribution, Gemini stream usage
parsing, and repository cleanup that could hide real usage from reports.

**Why not upstream:** These changes are tied to AAWM's auto-agent alias policy,
session-history schema, Langfuse tagging conventions, and OpenRouter free-model
metering/repair workflows.

**Validation status:** Focused unit coverage covers the OpenRouter completion
adapter, auto-agent retry rollover, Gemini stream usage parsing, repository
repair rules, Langfuse size-audit helper, and OpenRouter model attribution.
Dev runtime was validated before prod infrastructure promotion.

---

### aawm.63 — Codex Responses stream logging repair

**Status:** AAWM local hotfix.

**What changed:** Codex Responses API passthrough stream logging now tolerates
`response.completed.response` payloads that omit `output`. The logging path
normalizes the strict response model payload with an empty output list, then
continues to rebuild streamed text, tool output items, usage, cache-read
tokens, and the `codex.usage_normalize` span from the collected SSE chunks.

**Why:** The ChatGPT Codex backend can stream valid `response.output_*` events
and finish with a `response.completed` payload that carries usage but no
top-level `output` field. The previous strict reconstruction failed validation
and stored Langfuse generation output beginning with `cannot parse chunks`
instead of the actual Codex answer and usage.

**Why not upstream:** This is tied to AAWM's direct Codex Responses passthrough
route family, Langfuse/session-history normalization, and Codex-specific
usage span behavior.

**Validation status:** Focused unit coverage rebuilds a Codex Responses stream
with missing completed-output payload, preserves usage/cache-read telemetry,
and asserts the standard logging object no longer contains the chunk-parse
fallback. Dev runtime was live-validated with Codex profile traffic before prod
promotion.

---

### aawm.64 — Native alternate-provider routing and quota/session telemetry expansion

**Status:** AAWM local release candidate.

**What changed:** The fork now carries the current native alternate-provider
route set for Grok/xAI, Google Antigravity Code Assist, and OpenCode Zen.
Grok/xAI changes add managed OAuth aliases and native Grok CLI/OIDC
passthrough models across OpenAI/Codex and Anthropic-style endpoints, Composer
and Grok Build pricing/side-channel handling, xAI quota/header extraction into
provider rate-limit observations, and corrected Responses-path routing for the
new Grok model family. Google Antigravity changes add Code Assist routing,
model-selection support for the exposed Gemini/Vertex-family pools, and
pool-level rate-limit attribution. OpenCode Zen changes add the saved-credential
provider route, current free-model allowlist, Codex/OpenAI and Claude/Anthropic
adapter paths, path-aware `/responses`, `/chat/completions`, and `/messages`
parsing, plus free-model pricing/catalog entries. Session-history changes add
tracked flags for changes touching pre-commit config, `.env*`, `pyproject.toml`,
and `.gitignore`, and the agent-quality ruleset is packaged with both the
source tree and callback overlay.

**Why:** AAWM routes local TUI clients through LiteLLM so Codex, Claude Code,
Gemini/Grok/OpenCode clients, and production reporting share a single model
usage, identity, quota, and session-history surface. The providers expose
different authentication artifacts, endpoint shapes, side channels, quota
families, model aliases, and unsupported parameters, so the fork needs
provider-specific adapters and telemetry normalization rather than generic
OpenAI-compatible handling.

**Why not upstream:** These integrations depend on local AAWM credential
locations, provider-specific TUI behavior, `aawm_tristore` session-history and
rate-limit schemas, AAWM callback enrichment, and local Codex/Claude/Gemini/Grok
acceptance harness conventions.

**Validation status:** Focused unit and harness coverage exists for the Grok/xAI
native/OAuth paths, Google Antigravity Code Assist quota pools, OpenCode Zen
passthrough and adapter behavior, session-history config-change flags,
rate-limit backfills, and agent-quality scoring. Dev runtime smoke evidence
exists for the added provider lanes on `:4001`; prod promotion still requires
the normal release runbook steps, image publication, infrastructure rebuild,
built-image inspection, and prod validation after the container restart.

---

### aawm.65 — OpenCode Zen Codex Responses adapter chat-completions bridge

**Status:** AAWM local release candidate.

**What changed:** Codex/OpenAI Responses traffic for OpenCode Zen free models,
including `opencode/big-pickle`, now adapts the inbound Responses request into a
chat-completions call against OpenCode Zen `/v1/chat/completions` through
`litellm.acompletion`. The adapter strips OpenCode's unsupported top-level
Responses `format` parameter, preserves supported function tools, removes
unsupported Codex tool advertisements, and converts the chat completion result
back into an OpenAI Responses response for Codex. Route metadata now records the
public model, adapter model, OpenCode provider family, and
`codex_adapter_target_endpoint=opencode_zen:/v1/chat/completions`.

**Why:** OpenCode Zen accepts `big-pickle` through its chat-completions shape but
rejects the prior direct `/v1/responses` passthrough for Codex/OpenAI-format
requests with `401 ModelError: Model big-pickle is not supported for format
openai`. The existing route reached OpenCode, but the endpoint/format pairing
was unusable for Codex CLI.

**Why not upstream:** This depends on the AAWM local OpenCode saved credential,
the AAWM OpenCode model aliases, Codex-native request markers, and
`aawm_tristore` session-history metadata used to prove adapter attribution.

**Validation status:** Focused unit coverage verifies saved OpenCode auth,
egress validation, format stripping, supported function-tool conversion,
Responses response reconstruction, streaming iterator wiring, and the
OpenAI-passthrough route selection. Dev runtime smoke
`d1-203-dev-opencode-bigpickle-20260604T2058` returned
`OPENCODE BIG PICKLE DEV OK`, and `public.session_history` row `1133308`
stored `provider=opencode_zen`, `model=big-pickle`, `client_name=codex_exec`,
token counts `16660/32/16692`, route family
`codex_opencode_zen_adapter`, and
`codex_adapter_target_endpoint=opencode_zen:/v1/chat/completions`. Prod
promotion still requires publishing the `v1.82.3-aawm.65` fork image,
rebuilding/restarting `aawm-litellm`, then running the prod Codex smoke and
session-history proof.

---

### aawm.66 — AAWM tiered auto-agent aliases for Codex and Anthropic clients

**Status:** AAWM local release candidate.

**What changed:** The auto-agent selectors now support alias-keyed policy maps
instead of one hard-coded candidate list per client family. New Codex/OpenAI
Responses aliases are `aawm-sota`, `aawm-code`, and `aawm-low`; new
Anthropic/Messages aliases are `aawm-sota-anthropic`,
`aawm-code-anthropic`, and `aawm-low-anthropic`. Legacy
`aawm-codex-agent-auto` and `aawm-anthropic-agent-auto` behavior is preserved.

The selector keeps cooldowns scoped to provider/model/lane, scopes new session
affinity keys by requested alias, and dispatches selected candidates through
existing native OpenAI, direct Anthropic, Antigravity Code Assist, Grok native
OAuth, OpenRouter, and OpenCode Zen adapter paths. Request metadata records the
actual requested alias, selected provider/model/route family, selection reason,
attempt history, skipped cooldown candidates, lane key, and last-resort status.
OpenCode Zen auth/config misses remain hard failures for direct OpenCode
requests, but are converted to candidate-unavailable cooldowns inside auto-agent
low-tier selection so later candidates can be tried.
The AAWM callback source and production wheel overlay allowlist the new
Anthropic auto-agent metadata and normalize session-history
provider/model/model-group to the selected target.

**Why:** New user-facing AAWM model aliases need the same ordered fallback,
cooldown, continuation, and observability contract as `aawm-codex-agent-auto`.
Static model labels would hide candidate selection and would not provide
bounded retry/fallback behavior for transient 429, capacity, or empty-success
failures.

**Why not upstream:** The aliases, candidate order, saved credentials, and
session-history metadata are AAWM-specific operational policy.

**Validation status:** Focused pass-through tests cover resolver gating,
candidate ordering, cooldown fallthrough, continuation terminal behavior,
dynamic alias metadata, and cross-provider adapter dispatch for the auto-agent
paths. Focused callback tests cover source/overlay parity plus Codex and
Anthropic auto-agent selected-provider/model attribution. Active
`~/.codex` catalog/cache entries and live prod/dev smokes for the six public
aliases remain separate acceptance gates. The direct Anthropic SOTA candidate
was updated to the verified direct Opus slug `claude-opus-4-8` in the D1-208
completion patch; see `aawm.70` for the live proof and session-history rows.

---

### aawm.67 — Sanitized passthrough tool-definition observability snapshots

**Status:** AAWM local release candidate.

**What changed:** OpenAI Responses/Codex passthrough observability now captures a
bounded, sanitized snapshot of advertised tool definitions in `litellm_metadata`.
The snapshot records source list, total and captured tool counts, tool names,
tool types, description text, callable parameter schema, a sanitized full
definition copy, a stable snapshot hash, and whether capture was truncated.

The capture path redacts secret-like keys and values before metadata is handed
to logging surfaces. It covers common credential shapes such as API keys,
authorization headers, bearer tokens, passwords, secrets, and `sk-...` /
`pk-...` values while preserving non-secret schema structure for auditability.
The AAWM session-history callback allowlists the new
`aawm_tool_definition_*` metadata fields so operators can prove which tool
definitions were advertised without scanning raw request bodies.

**Why:** Operators need to distinguish natural-language tool descriptions from
the actual callable JSON schema that reached LiteLLM, especially for subagent
tool dispatch and model-selection debugging. Prior evidence surfaces could show
tool use after the fact, but not the request-time advertised tool contract.

**Why not upstream:** The persisted metadata names, session-history allowlist,
and AAWM `session_history` proof workflow are fork-specific observability
policy.

**Validation status:** Focused tests cover passthrough metadata capture,
snapshot hashing/truncation fields, credential redaction, and preservation of
the new metadata through the AAWM session-history record builder. Live dev proof
on 2026-06-05 used exact session
`d1-206-spawn-tool-20260605T0112Z` and response id
`resp_0b38de761f6182ed006a2256b669b4819184f512bd46218574`.
`public.session_history` row `1150506` and ClickHouse observation
`time-04-55-18-144748_resp_0b38de761f6182ed006a2256b669b4819184f512bd46218574`
recorded `aawm_tool_definition_names=["spawn_agent"]`, the callable parameter
schema including lower-case `model`, the `multi_agent_v1.spawn_agent`
description text, snapshot hash
`5b2b552cacfb16cdc82286196bf62ce1e3472873f02ecb8e73d02cf1d521deb9`, and
`aawm_tool_definition_snapshot_truncated=false`.

---

### aawm.68 — Resilient AAWM session-history flush diagnostics and timeout

**Status:** AAWM local release candidate.

**What changed:** The AAWM session-history callback now logs flush failures with
`verbose_logger.exception()` and formats empty-string exceptions as
`TypeName: repr(...)`, making async persistence failures actionable in runtime
logs. The asyncpg command timeout used by both direct session-history
connections and pooled writer connections is now configurable through
`AAWM_SESSION_HISTORY_COMMAND_TIMEOUT_SECONDS` and defaults to 60 seconds.
The callback wheel overlay is synced to the source callback so production-style
installed-wheel runtimes use the same writer behavior.

**Why:** Large Codex/OpenAI passthrough records carrying full request context and
tool-definition snapshots can exceed the previous hard-coded 10 second asyncpg
command timeout. Before this patch, the failure could appear as an empty warning
message, leaving no useful root cause in dev/prod logs.

**Why not upstream:** The callback, environment variable, and
`aawm_tristore.public.session_history` persistence path are AAWM-specific
observability infrastructure.

**Validation status:** Focused tests cover the parsed/default command timeout,
pool wiring, empty-message overflow-thread warning formatting, and
empty-message batch flush exception logging. Live dev proof after restart
persisted exact D1-206 session-history row `1150506` for session
`d1-206-spawn-tool-20260605T0112Z`.

---

### aawm.69 — Tiered auto-agent alias fallback hardening

**Status:** AAWM local release candidate.

**What changed:** Codex/OpenAI and Anthropic auto-agent alias probes now treat
Antigravity OAuth refresh failures and Grok native/xAI OAuth credential-prep
failures as retryable candidate-unavailable conditions instead of terminal
500s. Codex/OpenAI OpenRouter Responses candidates now pre-validate streamed
Responses payloads before returning them to Codex; a stream with no
`response.completed` event or an empty successful payload raises the existing
`aawm_codex_auto_agent_empty_success` marker so `aawm-low` can cool that
candidate and continue to OpenCode or the final Codex fallback. Successful
OpenRouter streams are replayed unchanged after validation.

**Why:** D1-207 live smokes showed `aawm-code` could get past Antigravity only
to stop on stale Grok credentials, and `aawm-low` could receive an OpenRouter
stream without a usable `response.completed`, causing the client to fail before
the alias selector could try the next candidate. These are candidate
availability failures for fresh alias dispatches, not reasons to abandon the
ordered model-selection policy.

**Why not upstream:** The tiered aliases, candidate ordering, per-candidate
cooldown semantics, and local credential families are AAWM-specific
Codex/Claude routing policy.

**Validation status:** Focused tests cover Antigravity `invalid_client`
rollover, Grok native `invalid_grant` rollover for Codex and Anthropic alias
probes, OpenRouter streamed empty/no-completed rollover, and the broader
auto-agent/Grok/OpenCode/Antigravity selection surface. Live dev smokes on
2026-06-05 proved `aawm-sota`, `aawm-code`, and `aawm-low` return sentinels and
persist selected target/attempt metadata in `public.session_history`.

---

### aawm.70 — Anthropic tiered alias Opus 4.8 completion and NUL-safe callback payloads

**Status:** AAWM local release candidate.

**What changed:** `aawm-sota-anthropic` now targets direct Anthropic
`claude-opus-4-8`, with matching model-cost entries in the primary and bundled
fallback cost maps. The AAWM session-history callback strips PostgreSQL NUL
bytes from text and JSON payloads at the final DB payload boundary for both
`public.session_history` and `public.session_history_tool_activity`, and the
callback wheel overlay is synced to the source callback.

**Why:** D1-208 requested Opus 4.8 for the SOTA Anthropic tier. Live Claude
smoke validation also exposed asyncpg JSONB persistence failures when callback
payloads contained `\x00`; PostgreSQL cannot store NUL bytes in text/jsonb
values, so the writer must scrub them before insertion.

**Why not upstream:** The alias names, Opus 4.8 routing policy, local model-cost
entry, callback overlay, and `aawm_tristore.public.session_history`
observability contract are AAWM-specific.

**Validation status:** Focused route tests covered Anthropic alias ordering,
Grok/Antigravity rollover, OpenCode/OpenRouter adapter selection, and the SOTA
Opus 4.8 target. Focused callback tests covered source/overlay parity,
session-history selected-target attribution, builder-level NUL scrubbing, and
the async persistence flush path. Live dev proof on 2026-06-05 after restarting
`litellm-dev` used route hash
`f96d6f72395318ba39385aba679bc3f43f82d841f80394d60a5830b1de7097fa` and
callback hash
`138386c93f4d69ec7e3d8e61c1cabd5f43f3212b73c3bddc70e9e55f5c151d21`.
`aawm-low-anthropic` returned `D1 208 Low Anthropic routing is ready.` and
persisted row `1154404` selecting `openrouter|google/gemma-4-31b-it:free`.
`aawm-code-anthropic` returned `D1 208 Code Anthropic routing is ready.` and
persisted row `1154444` selecting `openai|gpt-5.3-codex-spark` after
Antigravity `invalid_client` cooled as candidate-unavailable. A credentialed
Claude CLI smoke for `aawm-sota-anthropic` returned model output and persisted
row `1154487` selecting `anthropic|claude-opus-4-8`. Rate-limit observations
were recorded for the three live rows: OpenRouter daily request meter `132049`,
Codex token pools `132061`-`132064`, and Anthropic response-header token pools
`132075`-`132076`. A post-smoke log scan after 2026-06-05 02:10 EDT found no
`unsupported Unicode`, `\u0000`, or `failed to flush` callback errors.

---

### aawm.71 — D1-215 production migration base release for persistence, provider status, alias failover, and PgBouncer pooling

**Status:** AAWM local release candidate.

**What changed:** The fork release line now includes the tested D1-211 through
D1-214 fixes needed for the D1-215 production migration. Session-history
persistence is retry-backed instead of dropping failed flush batches, side
writes such as rate-limit/provider-error observations are best-effort after the
primary `session_history` insert, and outage backfills align `created_at` to
the source observation time. Provider-status collection no longer performs DDL
on the steady-state insert path and has explicit lock/statement timeout
guardrails. AAWM model-alias routing now treats busy/high-demand/capacity
responses as candidate-unavailable conditions so aliases can cool the failed
candidate and continue to the next configured model. Runtime Postgres paths now
use the shared PgBouncer convention while backfill, repair, scoring, and schema
maintenance paths retain direct database access.

**Why:** Production had recovered through local runtime repair and older overlay
artifacts, but the durable container path still needed the tested fork code,
published overlay artifacts, and PgBouncer runtime configuration promoted
together. Without a new base image, prod could install the latest callback and
control-plane wheels while still missing base-code changes in passthrough
routing, stream logging, provider-status collection, and maintenance script
DSN handling.

**Why not upstream:** The persistence tables, provider-status sidecar, AAWM
alias ordering/fallback policy, and local PgBouncer topology are AAWM-specific
operational behavior.

**Validation status:** D1-211 through D1-214 focused tests and live checks
proved fresh `session_history`, `rate_limit_observations`, and
`provider_status_observations` rows; no session-history queue overflow/drop
logs; no provider-status lock waiters; alias failover on observed busy/capacity
messages; and a live PgBouncer transaction pool on `127.0.0.1:6432`. D1-215
promotion still requires publishing the `v1.82.3-aawm.71` base image,
rebuilding/restarting `aawm-litellm`, and running prod smokes including direct
`gpt5.5` concurrency.

---

### aawm.72 — Release-prep base for alias routing audits, session-history durability, passthrough hardening, and harness cleanup

**Status:** AAWM local release candidate.

**What changed:** The fork release line now includes the post-aawm.71 carried
patches on `develop`: AAWM read-only and coding aliases with routing audit
persistence, output-contract metadata, spawn-agent contract guidance, durable
session-history ingestion under pressure, provider-status sidecar guardrails,
alias high-demand and in-flight redispatch behavior, deployed adapter
session-history attribution hardening, Anthropic Fable 5 pricing, xAI/Grok
Responses passthrough session and tool handling, OpenCode Zen DeepSeek tool
adjacency handling, removal of Gemini-backed Anthropic harness cases, Ling
OpenRouter retry pacing, Antigravity CLI no-op token acceptance, local
Anthropic tool-result block validation, bounded Langfuse generation payloads,
Codex/OpenCode model-exposure limits, and deduplicated Langfuse tool-definition
snapshots.

**Why:** The production promotion needs a single base fork image tag that
matches the current `develop` head before the infrastructure image is rebuilt.
These changes touch the proxy routing path, callback/session-history
observability, local acceptance harness, model pricing metadata, and release
operator docs; shipping them behind the previous `1.82.3+aawm.71` fork version
would make prod image provenance ambiguous and would not distinguish the latest
alias, passthrough, Langfuse, and harness behavior from the older D1-215 base.

**Why not upstream:** The alias ordering, AAWM routing-audit table,
`aawm_tristore` session-history contract, local CLI/passthrough harness, and
OpenCode/Antigravity/Grok operational policy are AAWM-specific fork behavior.

**Additional release-prep fix:** During validation, OpenCode Zen returned an
account billing/authentication failure (`CreditsError` / `No payment method`) as
a 401 while `aawm-low` was probing alias candidates. Alias probe mode now treats
that OpenCode billing/auth state as candidate unavailable so the alias can
continue to `gpt-5.4-mini`; direct OpenCode routes still surface the upstream
failure.

**Validation status:** Pre-promotion validation is expected to run against
`litellm-dev` on `:4001` before main/tag promotion, then stop before the prod
container restart boundary. The final release evidence should include focused
tests for changed code paths, the default dev adapter harness artifact,
overlapping `litellm-dev` log inspection, a clean tracked worktree, and a
published fork image tag `v1.82.3-aawm.72` from current `main`.

---

### cb-v0.0.42 — Callback overlay parity for Antigravity/OpenCode attribution

**Status:** AAWM callback overlay release candidate.

**What changed:** The callback wheel source under
`.wheel-build/aawm_litellm_callbacks/agent_identity.py` is synced back to the
in-repo callback at `litellm/integrations/aawm_agent_identity.py`. This carries
the Antigravity session-history and provider-rate-limit attribution fixes from
the source tree into the production callback overlay, including
`provider=antigravity` preservation for Codex/OpenAI and Claude/Anthropic
adapter traffic, Antigravity pool-level quota rows with
`client_family=antigravity_code_assist`, and the OpenCode Zen attribution
metadata added for the aawm.65 Responses bridge. A focused regression test now
asserts source/overlay parity so future callback behavior cannot silently drift
between `litellm-dev` and the production-style installed wheel path.

**Why:** `litellm-dev` imports the source-tree callback, while the production
`aawm-litellm` image imports the installed
`aawm_litellm_callbacks.agent_identity` wheel module. Antigravity source fixes
had landed in the fork, but production reporting could still store
Antigravity traffic under upstream-shaped `openai`, `gemini`, or Google Code
Assist dimensions if the callback overlay wheel stayed on the older source.

**Why not upstream:** This is specific to AAWM's callback overlay release line,
`aawm_tristore` reporting schema, Antigravity/OpenCode pass-through metadata,
and production image packaging.

**Validation status:** Focused callback tests cover source/overlay parity,
Antigravity quota-pool attribution, Antigravity over Google API-base
normalization, exact Codex/OpenAI `provider=openai` masking, exact
Claude/Anthropic `provider=gemini` masking, and OpenCode Zen identity
preservation. The callback wheel builds successfully from `.wheel-build/`.
Promotion still requires merging to `main` so artifact autobump publishes the
next `cb-v*` release, rebuilding/restarting `aawm-litellm`, then running dev and
prod persisted `session_history` / `rate_limit_observations` proof.

---

### cb-v0.0.45 — Callback overlay parity for PgBouncer runtime pooling and persistence recovery

**Status:** AAWM callback overlay release candidate.

**What changed:** The callback wheel source under
`.wheel-build/aawm_litellm_callbacks/agent_identity.py` remains synced to the
in-repo callback at `litellm/integrations/aawm_agent_identity.py` and carries
the D1-211/D1-214 session-history persistence and runtime Postgres changes into
the production installed-wheel path. This release includes retry-backed
session-history flush behavior, side-write isolation, callback package fallback
imports for bundled quality rules, and the shared PgBouncer runtime DSN
convention while keeping migration/admin scripts on direct database paths.

**Why:** Production imports `aawm_litellm_callbacks.agent_identity` from the
installed callback wheel, not the in-repo source callback used by `litellm-dev`.
The previous production container reported `aawm-litellm-callbacks=0.0.42`,
which predates the D1-211 persistence repair and the D1-214 callback-side
PgBouncer pooling update. D1-215 therefore needs a new callback wheel before
the prod migration can be considered durable.

**Why not upstream:** This is specific to AAWM's callback overlay release line,
`aawm_tristore` session-history persistence, local PgBouncer topology, and
production image packaging.

**Validation status:** Focused D1-211 and D1-214 tests covered
session-history flush retry behavior, side-write isolation, source/overlay
parity, PgBouncer runtime DSN derivation, direct-DSN preservation for backfill
and repair scripts, and provider-status no-hot-path-DDL behavior. D1-215
promotion still requires publishing `cb-v0.0.45`, rebuilding/restarting
`aawm-litellm`, and proving fresh prod `session_history` /
`rate_limit_observations` rows with no queue overflow/drop logs.

---

### cp-v0.0.8 — Control-plane overlay parity for PgBouncer-aware passthrough runtime paths

**Status:** AAWM control-plane overlay release candidate.

**What changed:** The control-plane wheel source now packages the D1-214
changes to `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
so production installed-wheel runtimes use the same PgBouncer-aware runtime
database convention and metadata behavior as the source tree.

**Why:** Production installs the independent `aawm-litellm-control-plane` wheel
over the pinned LiteLLM base image. The previous production container reported
`aawm-litellm-control-plane=0.0.7`, while D1-214 changed the control-plane
source after that release. D1-215 needs `cp-v0.0.8` published and installed so
control-plane behavior does not drift from the tested source path.

**Why not upstream:** This is AAWM-specific Claude/Codex control-plane packaging
and local runtime database topology, not generic LiteLLM provider behavior.

**Validation status:** D1-214 focused tests covered the touched control-plane
and passthrough paths, including pooled runtime database configuration and
preserved direct database usage for maintenance paths. D1-215 promotion still
requires publishing `cp-v0.0.8`, rebuilding/restarting `aawm-litellm`, and
verifying the prod container reports `aawm-litellm-control-plane=0.0.8`.

---

### aawm.D1-244 — Normalize Claude `[1m]` selectors on native Anthropic passthrough

**What changed:** The native `/anthropic/v1/messages` passthrough now accepts
Claude Code-style 1M model selectors such as `claude-opus-4-8[1m]` at the
LiteLLM boundary. Before falling through to direct Anthropic egress, the proxy
strips the bracket suffix from the request body model and appends
`context-1m-2025-08-07` to the outbound `anthropic-beta` header, preserving and
deduping any beta values the client already supplied. If the client supplied a
case-variant beta header or `x-pass-anthropic-beta`, those values are folded
into the normalized beta header and the prefixed pass-through beta header is
blocked from overwriting the normalized value during final forwarding.

**Why:** A dev Claude Code request reached direct Anthropic with the literal
model `claude-opus-4-8[1m]` and received an upstream 404. Claude Code's intended
wire shape is the base model plus `anthropic-beta: context-1m-2025-08-07`; the
bracket suffix is a local selector, not an Anthropic model id. This fix makes
the passthrough tolerant of clients or relays that fail to strip the suffix
before LiteLLM.

**Why not upstream:** This is an AAWM operational compatibility shim for local
Claude Code model selectors. It is scoped to the native Anthropic passthrough
fall-through and does not select or alter Google/Antigravity/OpenRouter adapter
routes.

**Validation status:** Focused passthrough tests cover direct Anthropic target
preservation, model normalization, beta-header append/dedupe behavior,
mixed-case and `x-pass-anthropic-beta` forwarding, and ordinary native model
passthrough remaining unchanged.

---

### aawm.D1-246/D1-248 — Repair Claude tool_use ids in Antigravity/Vertex replay

**What changed:** Anthropic pass-through request preparation now repairs
assistant `tool_use` content blocks that arrive without a non-empty `id`, using
the paired `tool_result.tool_use_id` when present and a deterministic local id
otherwise. The Anthropic-to-OpenAI adapter applies the same repair before
Google/Antigravity request construction, and Claude-target Gemini function-call
parts now carry the repaired id as `functionCall.id` so the Antigravity/Vertex
Claude backend receives a complete multi-turn tool envelope. The reverse
Gemini/Vertex response transform also preserves upstream `functionCall.id`
when converting tool calls back into OpenAI/Anthropic-compatible responses, so
the first streamed tool-use turn gives Claude clients an id they can replay on
the follow-up request. The local dev compose file mounts
`litellm/types/llms/vertex_ai.py` to keep request/response shape types aligned
with live-mounted serializer code after restart-only validation. D1-248 extends
the repair to the Anthropic-native Google/Antigravity completion-adapter branch,
annotates Claude `functionResponse` parts with both top-level `id` and nested
`response.tool_use_id`, reconstructs missing native `model/functionCall` turns
from cached tool-call context when a compacted follow-up contains only a
`functionResponse`, and trims Code Assist content windows without orphaning
function-response turns from their paired function calls.

**Why:** Subagent canaries were dying on their second model request with
`messages.1.content.1.tool_use.id: Field required` because the replayed first
assistant turn reached the Vertex/Antigravity serializer without the required
Claude `tool_use.id`. The initial request-side repair did not cover the
response-parser leg that feeds ids back to streaming Anthropic clients. Later
D1-248 canaries showed the follow-up shape could also arrive as a compacted
native `functionResponse` without the paired prior `functionCall`; once the
response id was present, Vertex rejected that as an unpaired tool result until
the native pair was preserved or reconstructed.

**Why not upstream:** This protects AAWM's Anthropic-compatible
Antigravity/Code Assist adapter flow and its Claude-backed alias candidates.
Normal Gemini payloads remain unchanged except for pair-preserving content
window trimming; Claude-specific `functionCall.id` / `functionResponse.id`
repair is scoped to Claude-like adapter models.

**Validation status:** Focused tests cover proxy repair/preservation,
Anthropic-to-OpenAI adapter repair, Claude-target builder serialization, normal
Gemini no-id behavior, non-adjacent `tool_result` pairing, response-side
fallback id generation, and Vertex/Gemini response transform preservation of
upstream `functionCall.id`. D1-248 adds focused coverage for Anthropic-native
adapter replay repair, OpenAI-chat Anthropic replay normalization, cached
missing-tool-call reconstruction, Claude `functionResponse.id` annotation,
native missing-pair insertion, and content-window trimming that preserves
valid function-call/function-response pairs without exceeding the configured
window. A controlled `aawm-code` canary passed its second model request after
the final runtime recreate.

---

### aawm.D1-245 — Treat Antigravity silent-refresh failures as alias-candidate unavailable

**What changed:** Antigravity alias-probe fallback classification now treats
`AGY CLI silent auth refresh did not produce a valid token` as an
Antigravity credential-unavailable condition. In local dev,
`docker-compose.dev.yml` mounts `/home/zepfu/.gemini` read/write so Gemini and
Antigravity OAuth refreshes can persist updated token files instead of looping
on an expired read-only token.

**Why:** After the Vertex `tool_use.id` replay fix was deployed, controlled
subagent canaries stopped showing the old request-envelope error but repeatedly
hit Antigravity token refresh warnings before falling into high-demand
redispatch behavior. The dev container found the Antigravity token under
`/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token`, but the token was
expired and the `.gemini` mount was read-only, so silent refresh could not
produce/persist a usable replacement. Alias probes should continue to the next
candidate when that credential is unavailable, while direct Antigravity routes
still surface the auth failure.

**Why not upstream:** This is AAWM-specific local Antigravity/Codex alias
routing and dev-container credential handling.

**Validation status:** Focused tests cover both Anthropic-style
`aawm-code-anthropic` and Codex-style `aawm-code` fallback after the silent
refresh no-token error, while existing tests preserve terminal handling for
Vertex request-envelope 400s.

---

### aawm.D1-250 — Fail closed for tool-bearing Anthropic code-agent aliases

**What changed:** `aawm-code-anthropic` requests that declare tools or carry
stateful tool/reasoning continuation content now require the Antigravity Claude
compatibility route. If Antigravity is cooling down or unavailable, the alias
returns a structured 429 with `redispatch_model=aawm-code-anthropic` and marks
non-compatible candidates as `tool_schema_incompatible` instead of selecting
OpenAI, xAI, OpenRouter/OpenCode, or direct native Anthropic fallbacks.

**Why:** A dashboard-shell subagent (`ad13aecd1b0f72e0f`) showed the old policy
falling through from `aawm-code-anthropic` to `openai/gpt-5.3-codex-spark` via
`anthropic_openai_responses_adapter`. The Spark-backed session initially made
valid Bash calls, then degraded into `Bash` tool calls with empty inputs such as
`call_y67sWAwSpPGbqb8DLmIAx6mT`, causing repeated missing-`command` validation
errors and an empty final completion. Failing closed is safer than letting a
tool-driven engineering agent continue on a target that cannot reliably preserve
the Claude/Codex tool contract.

**Why not upstream:** This is AAWM-specific alias policy for local Codex/Claude
engineering workers and Antigravity-backed Claude access.

**Validation status:** Focused tests cover Antigravity quota/auth failure for
tool-bearing `aawm-code-anthropic` requests and prove neither the Spark adapter
nor direct native Anthropic passthrough is called.

---

### aawm.74 — AAWM alias routing, logging, and billing-detail promotion set

**What changed:** This release promotes the current AAWM LiteLLM work on
`develop` after the `v1.82.3-aawm.73` image. The promoted set includes
executable multi-provider failover behavior for AAWM aliases, route/access-log
identity enrichment, native access-log suppression for enriched route lines,
Grok/OIDC request-shape hardening, encrypted Grok compaction-state
preservation, local session-history outage spooling, managed error-log intake,
Langfuse event fitting, inbound model-alias persistence, and durable
rate-limit/billing detail storage. It also updates Codex native passthrough
tool-description patching for the current deferred multi-agent `tool_search`
surface, so the fanout guidance and patch metadata still apply when Codex no
longer exposes a direct `spawn_agent` tool definition.

**Why:** The prior prod line did not contain the completed follow-up work for
AAWM alias failover, Grok/native passthrough robustness, route observability,
session-history durability, and provider billing visibility. Promoting these
changes gives prod the same behavior already validated on the development
runtime and records a distinct fork image tag for the new core/runtime code.

**Why not upstream:** These changes are AAWM-specific operational behavior for
local alias routing, Claude/Codex/Grok/Antigravity provider adaptation,
session-history observability, and production diagnostics.

**Validation status:** Focused local tests cover the modified callback,
passthrough, backfill, harness, and quality-rule paths. Dev runtime validation
for the latest work verified `litellm-dev` readiness, in-container callback SQL
markers, and sanitized `rate_limit_observations` billing-detail persistence in
`aawm_tristore`. A focused live Codex passthrough harness on `litellm-dev`
verified the deferred-tool fanout guidance metadata after the dev container
restart. Production promotion must build and publish
`v1.82.3-aawm.74`, rebuild the infrastructure image from that pinned base, then
run the documented prod readiness, log, and harness gates.

---

### aawm.80 — Grok OIDC sidecar ownership and side-channel continuity hardening

**What changed:** Native Grok OIDC credential refresh ownership is hardened so
the provider-status sidecar remains the scheduled writer for
`~/.grok/auth.json` while LiteLLM stays a read-only consumer. Atomic refresh
writes now apply configured uid/gid/private mode before replacement, clamp
unsafe group/other modes back to `0600`, and include a metadata-only repair task
that can fix bad file ownership without reading or rewriting token values. The
Grok side-channel classifier also covers the current CLI endpoints
`/sessions/{id}/signals`, `/sessions/{id}/turn-deltas`, and `/traces` while
retaining historical `/sessions/register` and
`/sessions/{id}/replicas/update` coverage for redacted shape metadata and
retryable 5xx handling.

**Why:** Prod/dev showed stale Grok OIDC ownership could require repeated manual
login and could steer `aawm-code` away from Composer 2.5 Fast toward other Grok
targets. The sidecar must refresh and repair the shared credential even during
long idle windows, while model requests must not mutate credential files.

**Why not upstream:** This is AAWM-specific local Grok CLI/OIDC credential
ownership, provider-status sidecar behavior, and pass-through observability for
AAWM native Grok routes.

**Validation status:** Focused pytest coverage passes for the Grok OIDC refresh
script, provider-status sidecar tasks, native Grok OIDC harness config, and
Grok side-channel pass-through request handling. Dev runtime validation rebuilt
the provider-status sidecar, verified `auth.json` remains `zepfu:zepfu 0600`,
observed a successful Grok billing poll, restarted `litellm-dev`, and confirmed
the running container recognizes current and historical side-channel endpoint
types. Production promotion still requires publishing `v1.82.3-aawm.80`,
rebuilding/restarting `aawm-litellm`, verifying the prod Grok mount/env split,
and running the documented prod readiness/log gates.

---

### aawm.81 — Passthrough retry classification and Grok billing evidence hardening

**What changed:** The fork release line now includes the post-`aawm.80` fixes
for runtime error classification and Grok quota evidence. ChatGPT Codex
passthrough `503` connection-termination responses are covered by the
pre-first-byte hidden retry path and have a regression test that proves the
5/15/30/60/120 second retry schedule, final
`expected_upstream_capacity_or_internal` classification, and no old traceback
terminal log path. Known Grok billing timeout/cancel responses are downgraded
to degraded telemetry instead of active traceback intake. The provider-status
sidecar now records safe Grok billing request-contract diagnostics on
successful `rate_limit_observations` rows, including a stable fingerprint,
source `grok_billing_sidecar_poll`, method, host/path, query keys, client
identifier/version, model-override booleans, `x-xai-token-auth` configured
boolean, user-agent, and header names only. The sidecar dedupe SQL now includes
`evidence` so the first upgraded evidence-bearing snapshot can land even when
quota values are unchanged.

**Why:** Prod was still running an older passthrough path that emitted a
traceback-style active error log for an expected upstream Codex `503` transport
failure. Separately, successful Grok billing sidecar rows were ambiguous in the
database because the row evidence did not identify the sidecar request contract,
which made manual-versus-container billing comparisons repeat-prone.

**Why not upstream:** This is AAWM-specific provider routing, runtime
error-intake policy, Grok CLI/OIDC sidecar polling, and local
`aawm_tristore.rate_limit_observations` evidence semantics.

**Validation status:** Focused pytest coverage passed for the Codex passthrough
`503` hidden-retry regression, the broader passthrough hidden-retry subset, and
the Grok billing sidecar evidence path. Dev provider-status sidecar validation
rebuilt and recreated `aawm-provider-status-observations:dev`, confirmed the
running container contains the sidecar evidence and dedupe SQL patches, forced a
live Grok billing poll with HTTP `200`, and verified DB row id `184547` includes
`request_contract_source='grok_billing_sidecar_poll'` and fingerprint
`d3f3858fb59887901db82e1895900c3f5ced2cba736da7ef8a01231dfc3d5d60`.
The `v1.82.3-aawm.81` tag was cut from the pre-autobump release merge and
failed the image publisher's current-`main` reachability gate after the
artifact autobump advanced `main`; see `aawm.82` for the replacement tag.
Production promotion is superseded by `v1.82.3-aawm.83`, which carries this
hidden-retry behavior plus the later Grok billing sidecar drift fix.

---

### aawm.83 — Grok billing sidecar auth-path drift hardening

**What changed:** The fork metadata advances to `1.82.3+aawm.83` and includes
the D1-308 provider-status sidecar hardening after `aawm.82`. The Grok billing
sidecar now resolves the Grok auth file with sidecar override first and native
Grok env fallback (`LITELLM_XAI_GROK_AUTH_FILE`,
`LITELLM_XAI_OAUTH_GROK_AUTH_FILE`, `GROK_AUTH_FILE`, `GROK_HOME/auth.json`,
then `/home/zepfu/.grok/auth.json`). Billing client-version fallback now also
checks `GROK_CLIENT_VERSION`, the disabled model-override branch keeps
`content-type: application/json` while omitting only `x-grok-model-override`,
and billing poll logs include safe diagnostics for resolved auth path source and
attempt budget.

**Why:** After D1-307 proved the containerized billing path could succeed, D1-308
identified remaining request/auth drift that could make future manual,
passthrough, forced sidecar, and scheduled sidecar comparisons ambiguous. The
release also carries the `aawm.82` hidden-retry Codex passthrough behavior that
prod still needs for D1-305.

**Why not upstream:** This is AAWM-specific provider-status sidecar behavior and
local Grok OIDC credential ownership.

**Validation status:** Focused sidecar tests passed (`56 passed`), ruff passed
for touched Python files, the dev provider-status sidecar was rebuilt and
recreated, and the restarted sidecar persisted DB row `184984` from a successful
HTTP `200` Grok billing poll with the new safe diagnostics. Publish
`v1.82.3-aawm.83`, then promote through the normal prod `:4000` process and
verify D1-305 hidden-retry symbols before archiving `.analysis/prod-error.log`.

---

### aawm.84 — Passthrough error classification and Gemini adapter hardening

**What changed:** The fork metadata advances to `1.82.3+aawm.84` and includes
the post-`aawm.83` passthrough hardening batch. Native OpenAI/Codex
passthrough now classifies ChatGPT/Codex HTML block pages as
`openai_chatgpt_codex_block_page` and suppresses the oversized traceback
payload for that known provider-side response. The Codex auto-agent native
OpenAI path now leaves transient pre-first-byte retry handling to the shared
pass-through retry loop while still exposing `429` as the retryable upstream
status. Antigravity Gemini Code Assist paths now price `gemini-3.5-flash-low`
against public Gemini 3.5 Flash cost metadata, repair final Google
function-call turn adjacency before egress, and own transient
`500/502/503/504/529` hidden retries when the adapter must bypass the shared
pass-through retry loop. This line also carries the low-alias routing,
Langfuse compaction, diagnostic capture, and Grok side-channel error-context
commits already landed on `develop` after `aawm.83`.

**Why:** Prod error intake showed ChatGPT/Codex HTML block pages and Codex read
timeouts as generic active traceback failures. Dev error intake also showed
Antigravity Gemini pricing gaps, streaming iterator regressions,
function-call adjacency `400`s, and a `503 UNAVAILABLE` that needed
adapter-owned retry metadata because Code Assist paths set
`caller_managed_hidden_retry`.

**Why not upstream:** This is AAWM-specific pass-through classification,
ChatGPT/Codex account-route handling, Antigravity Gemini Code Assist request
repair, local cost attribution, Langfuse/event fitting policy, and runtime
error-intake hygiene.

**Validation status:** Focused dev tests passed for ChatGPT/Codex block-page
classification, Codex native OpenAI shared transient retry wiring, Gemini
function-call adjacency repair, Gemini public-cost fallback, and Google adapter
transient `503` retry metadata. Dev `litellm-dev` was recreated after the
D1-329/D1-331 fixes and reported healthy on `:4001`. The
`v1.82.3-aawm.84` tag was cut before the automatic artifact-version bump moved
`main`, so the guarded fork image publisher failed its current-`main`
reachability gate. Production promotion is superseded by `v1.82.3-aawm.85`.

---

### aawm.85 — Post-artifact-autobump release candidate retag

**What changed:** The fork metadata advances to `1.82.3+aawm.85` on top of the
post-`aawm.84` artifact autobump. The behavior is the passthrough error
classification and Gemini adapter hardening described in `aawm.84`, with
callback overlay metadata advanced to `cb-v0.0.53`, model-config metadata to
`cfg-v0.0.18`, and harness metadata to `h-v0.0.35`.

**Why:** The `v1.82.3-aawm.84` tag was created before the automatic artifact
version bump moved `main` to `4b4c44c971`. The guarded fork image publisher
requires the tagged commit to be reachable from current `main`, so `.84`
failed at the release gate. Per the release runbook, the old tag is preserved
and this replacement tag is cut from current `main`.

**Why not upstream:** This is AAWM release-line bookkeeping for fork image tags
and independently published overlay artifact versions.

**Validation status:** Metadata-only retag on top of the tested `aawm.84`
content and artifact autobump. Publish `v1.82.3-aawm.85`, then promote through
the normal prod `:4000` process and verify prod error-intake cleanup for the
D1-329 ChatGPT/Codex records.

---

### aawm.86 — OpenRouter message-shape failover and Antigravity sidecar auth paths

**What changed:** The fork metadata advances to `1.82.3+aawm.86`. Codex
auto-agent OpenRouter chat-completions egress now sanitizes malformed empty
messages while preserving assistant tool calls and tool-result rows, and the
observed OpenRouter/Cohere `invalid message provided ... must have non-empty
content or tool calls` failure class now maps to `provider_format_rejected` so
alias failover can continue instead of surfacing as an unhandled ASGI `500`.
Antigravity Code Assist OAuth loading now honors
`LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE` before legacy token paths and
`LITELLM_ANTIGRAVITY_SEED_AUTH_FILE`, with LiteLLM remaining a read-only token
consumer. The provider-status sidecar runner can also resolve
`LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE` as its managed Antigravity write path.

**Why:** Prod error intake showed OpenRouter/Cohere rejecting a Codex
Responses-to-chat-completions translation that contained an empty message with
neither content nor tool calls. Prod Antigravity lane-resolution logs also
showed the live runtime still attempting stale silent refresh behavior while the
provider-status sidecar lacked Antigravity write ownership.

**Why not upstream:** This is AAWM-specific Codex auto-agent alias routing,
OpenRouter completion-adapter request shaping, Antigravity Code Assist OAuth
ownership, and provider-status sidecar deployment policy.

**Validation status:** Focused tests passed for OpenRouter sanitizer behavior,
OpenRouter invalid-message failover classification, Antigravity managed/seed
token precedence, and provider-status managed-auth-file resolution. Dev
`litellm-dev` was refreshed and runtime probes confirmed the OpenRouter
sanitizer removed invalid empty messages while preserving tool calls, and that
Antigravity selected the managed token when legacy, managed, and seed files were
all valid. Production still requires publishing `v1.82.3-aawm.86`, rebuilding
`aawm-litellm`, and applying the sibling infrastructure handoff
`.analysis/handoff-litellm-d1-341-antigravity-prod-sidecar-refresh-20260619.md`
so `aawm-provider-status-observations-prod` owns managed Antigravity refresh.

---

### aawm.82 — Post-callback-autobump release candidate retag

**What changed:** The fork metadata advances to `1.82.3+aawm.82` on top of the
post-`aawm.81` callback artifact autobump. The code behavior is the same
passthrough retry classification and Grok billing evidence hardening described
in `aawm.81`, with callback overlay metadata advanced to `cb-v0.0.52`.

**Why:** The `v1.82.3-aawm.81` tag was created before the automatic callback
artifact bump moved `main` to `feb1bb6f6b`. The guarded fork image publisher
requires the tagged commit to be reachable from current `main`, so `aawm.81`
failed at the release gate. Per the release runbook, the old tag is preserved
and a new fork image tag is cut from current `main`.

**Why not upstream:** This is AAWM release-line bookkeeping for fork image tags
and independently published overlay artifact versions.

**Validation status:** Release metadata-only retag on top of the tested
`aawm.81` content and callback overlay autobump. After `main` and `develop`
are converged, publish `v1.82.3-aawm.82`, verify the fork image workflow, then
promote through the normal prod `:4000` process.

---

### cb-v0.0.51 — Callback overlay parity for durable JSONL session-history spool

**Status:** AAWM callback overlay release.

**What changed:** The callback wheel source under
`.wheel-build/aawm_litellm_callbacks/agent_identity.py` is synced with the
in-repo callback at `litellm/integrations/aawm_agent_identity.py` for durable
session-history outage spooling. New spool artifacts are JSONL files with a
metadata line followed by one `type=record` line per protected row, while legacy
`.json` artifacts remain replayable. Spool directory listing failures now report
`spool_pending=unknown` and keep replay retryable instead of looking like an
empty backlog.

**Why:** Production imports `aawm_litellm_callbacks.agent_identity` from the
installed callback wheel, not the in-repo source callback used by `litellm-dev`.
The production container on `aawm-litellm-callbacks=0.0.48` still wrote legacy
`.json` spool files even after the base image and host mount were promoted.

**Why not upstream:** This is specific to AAWM's callback overlay release line,
`aawm_tristore` session-history persistence, and local host-mounted outage
spooling.

**Validation status:** Source and overlay callback files are kept in parity.
Focused tests cover JSONL write/load/replay behavior and legacy `.json`
loading. `cb-v0.0.51` was published manually after GitHub-created tag
suppression left the autobumped tag without a release asset. Production was
rebuilt on base `1.82.3+aawm.80` with callback overlay `0.0.51`, and the
prod-safe spool write/load/replay cleanup proof returned `wrote_jsonl=true`.

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
