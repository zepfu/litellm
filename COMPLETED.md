# Completed

## 2026-04-25

- Promoted the production `:4000` LiteLLM container to the `aawm.35` base image and completed release validation.
  The missing overlay GitHub Releases were published from existing tags for callback `cb-v0.0.9`, model config `cfg-v0.0.6`, and harness `h-v0.0.17`; the prod image was rebuilt with cache busting and verified to install `aawm-litellm-callbacks 0.0.9`, `aawm-litellm-control-plane 0.0.5`, and a model config containing GPT-5.5 pricing. Prod readiness is healthy on `:4000`. The final default prod harness passed at `/tmp/litellm-prod-harness-aawm35-default-final.json` with zero failures and warning-only Gemini/OpenRouter canary notes; explicit prod shards passed for native passthrough at `/tmp/litellm-prod-harness-aawm35-native-openai-rerun.json`, OpenAI/Gemini effort/cache at `/tmp/litellm-prod-harness-aawm35-effort-cache-openai-gemini-rerun.json`, and OpenRouter effort/cache at `/tmp/litellm-prod-harness-aawm35-effort-cache-openrouter.json`.

- Hardened the release process after the `aawm.35` prod cutover.
  `PROD_RELEASE.md` now calls out that overlay git tags are insufficient unless matching GitHub Release assets exist, that `--no-cache` rebuilds are required after publishing missing overlay releases, that built images should be inspected before restarting prod, and that direct `/openai_passthrough/*` prod validation requires `OPENAI_API_KEY` mapped into the container. The infrastructure compose file now maps `OPENAI_API_KEY` from `AAWM_OPENAI_API_KEY`, matching the dev runtime behavior.

- Removed static session ids from opt-in effort/cache harness cases.
  The OpenAI effort, Gemini effort, and OpenRouter effort/cache cases now use the existing generated per-run session id path, avoiding false prod failures caused by older dev `public.session_history` rows sharing the same static session id. The affected prod shard passed cleanly after the change.

- Broadened live harness coverage for the completed `/anthropic` effort/cache translation work.
  Added default-excluded existing-harness cases for Gemini minimal/max effort with cache-control variants, OpenRouter max/none/no-effort with cache-control variants, and an ordered OpenAI two-pass prompt-cache case. The harness now supports repeated HTTP passes, compact repeated-text fixtures, env-expanded HTTP headers, multi-row session-history minimums, and provider cache hit validation without introducing a second harness. Live dev shard artifacts passed with zero failures/warnings at `/tmp/anthropic-effort-cache-gemini-dev.json`, `/tmp/anthropic-effort-cache-openrouter-dev.json`, and `/tmp/anthropic-effort-cache-openai-dev.json`.

- Fixed OpenAI Responses adapter cache-intent persistence for Anthropic `cache_control`.
  The OpenAI Responses adapter now derives bounded `prompt_cache_key` values from Anthropic cache-control material, preserves route-level `litellm_metadata` while adding adapter cache metadata, and `public.session_history.metadata` now persists `openai_prompt_cache_key_present` plus `anthropic_adapter_cache_control_present`. The OpenAI two-pass live shard validates two rows for the generated session and requires a provider cache hit with cached input tokens on the warmed request.

- Validated the broadened effort/cache harness work on dev.
  Focused tests passed for local-ci harness hardening (`14 passed`), AAWM cache/session-history coverage (`9 passed / 55 deselected`), adapter/proxy cache transformation coverage (`28 passed / 350 deselected`), JSON/compile checks, and callback source parity. The full default dev Anthropic adapter harness passed at `/tmp/anthropic-adapter-dev-full-after-effort-cache.json` with no failures; the only warnings were the configured warning-only Gemini canary output/usage checks for `claude_adapter_gemini31_pro` and `claude_adapter_gemini31_flash`.

## 2026-04-24

- Purged the 24 operator-approved ambiguous historical Gemini `unknown` zero-token rows from local `aawm_tristore`.
  The cleanup was deliberately scoped to old dev harness sessions for `native_gemini_passthrough_stream_generate_content` between `2026-04-24 16:21:36+00` and `2026-04-24 16:25:16+00`, with `provider=gemini`, `model=unknown`, zero input/output tokens, and `litellm_environment=dev`. No matching `session_history_tool_activity` rows existed, the delete returned 24 rows, and the post-delete verification count for `gemini/unknown` zero-token rows is zero.

- Extended the existing Anthropic adapter harness tenant assertions across native and adapted lanes.
  The harness now carries a shared `default_tenant_id`, injects `x-aawm-tenant-id` through HTTP, Claude CLI, Codex CLI, and Gemini CLI context setup, and defaults `public.session_history.tenant_id`, `metadata.tenant_id`, and `metadata.tenant_id_source` assertions for every session-history case. Multi-row fanout validations now merge `required_equals.tenant_id` into each expected row. Live dev validation passed for the native Anthropic, OpenAI chat, OpenAI Responses, Codex Responses, Gemini generateContent, and Gemini streamGenerateContent cases with zero failures and zero warnings.

- Broadened opt-in `/anthropic` effort/cache harness coverage.
  Added default-excluded HTTP cases for Gemini `output_config.effort` translation and OpenRouter effort plus cache-control intent. The final opt-in dev bundle passed for `claude_adapter_openai_output_config_effort`, `claude_adapter_gemini_output_config_effort`, `claude_adapter_openrouter_output_config_effort_cache`, and `claude_adapter_nvidia_cache_control_strip` with zero failures and zero warnings. The OpenRouter case validates durable session-history reasoning/cache metadata and row-level provider-cache status; deeper outbound request-payload assertions remain pending until that lane exposes a stable parsed request shape.

- Added a safe historical Gemini control-plane repair mode.
  `scripts/backfill_session_history.py --repair-session-history --repair-gemini-control-plane {delete,mark}` now dry-runs by default, honors the existing request/session/trace/provider/model/time/limit filters, uses id-keyset pagination, deletes tool-activity rows before session rows when applying deletes, and only matches rows with explicit `loadCodeAssist`, `listExperiments`, `retrieveUserQuota`, or `fetchAdminControls` evidence. Local dry-run against `aawm_tristore` matched zero explicit-method rows; the 24 ambiguous `gemini/unknown` zero-token rows without method evidence were intentionally left untouched.

- Validated the updated harness/backfill package on dev.
  Focused tests passed for backfill repair helpers (`6 passed`), adapter/provider/session-history unit coverage (`79 passed` and `24 passed / 138 deselected` in the focused suites), JSON/compile checks passed, `git diff --check` passed, the full default Anthropic adapter harness on `:4001` passed with zero failures, native passthrough tenant validation passed with zero failures/warnings, and the new opt-in effort/cache bundle passed with zero failures/warnings.

- Completed the Claude `/anthropic` reasoning-effort and cache-translation implementation across OpenAI/Codex, Gemini, OpenRouter, and NVIDIA lanes.
  Added a shared Anthropic adapter normalization helper for `output_config.effort`, direct `reasoning_effort`, and `thinking` budgets; OpenAI Responses now preserves `xhigh` when model metadata supports it, Gemini receives normalized `reasoning_effort`, OpenRouter and NVIDIA use config-driven capability checks, and unsupported raw Anthropic fields are stripped before provider egress. Cache intent from Anthropic `cache_control` is preserved as normalized metadata, OpenAI derives bounded `prompt_cache_key` values only when cache intent is present, OpenRouter native cache-control support is metadata-driven, and NVIDIA records an explicit miss state instead of forwarding unsupported cache controls.

- Extended session-history and harness validation for the new adapter effort/cache paths.
  `public.session_history.metadata` now persists normalized reasoning-effort fields, NVIDIA is included in provider-cache family handling, and the existing Anthropic adapter harness now supports `required_equals`, `required_one_of`, `forbidden_paths`, and tenant-id assertions. Focused live dev validation on `:4001` passed for `claude_adapter_openai_output_config_effort` and `claude_adapter_nvidia_cache_control_strip` with zero failures and zero warnings after installing the changed helper files into `litellm-dev`.

- Updated GPT-5.5 pricing to official OpenAI API rates.
  `gpt-5.5` and `chatgpt/gpt-5.5` now use `$5.00 / 1M` input tokens, `$0.50 / 1M` cached input tokens, and `$30.00 / 1M` output tokens in both the primary and bundled fallback model cost maps. Focused session-history coverage confirms GPT-5.5 cost calculation now produces the corrected output-token cost.

- Preserved explicit tenant identity into `public.session_history`.
  `AawmAgentIdentity` now resolves tenant ids from canonical metadata and request headers such as `tenant_id`, `aawm_tenant_id`, `user_api_key_org_id`, org/team aliases, and AAWM/LiteLLM tenant headers before falling back to prompt-text project extraction. Live and Langfuse/backfill records persist both `tenant_id` and `metadata.tenant_id_source`, and spend-log reconstruction now restores stored `proxy_server_request.headers` so historical header-derived tenants can be repaired.

- Enhanced and ran the existing session-history backfill script for tenant and cost repair.
  `scripts/backfill_session_history.py --repair-session-history` can now repair existing rows in place, with `--repair-costs`, `--repair-tenant-ids`, and existing request/session/trace/provider/model/time filters. Local repair updated GPT-5.5 costs for 2,579 rows across scoped apply passes, and filled 1,306 missing `tenant_id` values while synchronizing tenant metadata on 13,323 rows.

- Completed native passthrough logging parity for Anthropic, OpenAI/Codex, and Gemini.
  Added a shared passthrough callback contract that preserves model/provider/cost fields, request headers, existing metadata, passthrough payloads, provider-native usage objects, and downstream `AawmAgentIdentity` derivability while keeping provider-specific parsing in the Anthropic, OpenAI, and Gemini handlers. OpenAI/Codex Responses and chat, Anthropic Messages, and Gemini `generateContent` / `streamGenerateContent` now all feed callbacks/session-history with consistent kwargs and metadata.

- Added native passthrough validation to the existing Anthropic adapter harness.
  `scripts/local-ci/run_anthropic_adapter_acceptance.py` and `anthropic_adapter_config.json` now include selectable native HTTP and real-CLI cases for Anthropic, OpenAI/Codex, and Gemini without introducing a separate harness entrypoint. The CLI cases launch the actual `claude`, `codex`, and `gemini` binaries so local auth/config remains the source of truth, and the harness validates Langfuse plus `public.session_history` for the actual provider session id emitted by each client.

- Fixed Gemini native CLI control-plane calls creating `gemini/unknown` session-history rows.
  Gemini Code Assist startup/admin calls such as `loadCodeAssist`, `listExperiments`, `retrieveUserQuota`, and `fetchAdminControls` now skip model-call logging in both non-streaming and streaming passthrough paths. Only real Gemini model operations (`generateContent`, `streamGenerateContent`, and `predictLongRunning`) produce model `session_history` rows. Live dev validation after the fix produced one `gemini/gemini-2.5-flash` row for the Gemini CLI session and zero new `gemini/unknown` rows.

- Validated the native passthrough logging work and pushed it to `develop`.
  The committed change `1168a12e74 feat: align native passthrough logging` passed the shared passthrough/provider/session-history test set (`125 passed`), the broader passthrough endpoint suite (`212 passed`), compile checks, `git diff --check`, callback source parity, and the combined real-CLI harness on `:4001` for `native_anthropic_passthrough_claude`, `native_openai_passthrough_responses_codex`, and `native_gemini_passthrough_stream_generate_content`.

- Finalized the production release process documentation.
  Added `PROD_RELEASE.md` with the dev/prod runtime split, pre-promotion validation, fork image tagging, overlay artifact handling, infrastructure promotion, prod harness validation, optional provider lanes, finalization, and rollback process. Updated `TEST_HARNESS.md`, `WHEEL.md`, and the bundled local-ci README so the prod process is discoverable and GPT-OSS 20B/120B are documented as explicit opt-in edge checks rather than default hard gates.

- Promoted the local prod `:4000` LiteLLM container to the `aawm.34` image and completed validation.
  Prod readiness reported LiteLLM `1.82.3+aawm.34`; focused `claude_adapter_openrouter_ling_26_flash`, `claude_adapter_peeromega_fanout`, NVIDIA GLM, and NVIDIA DeepSeek passed. `claude_adapter_gpt_oss_120b` hit the exact OpenRouter provider-unavailable signature (`503 provider=OpenInference raw=no healthy upstream`) and was classified as a soft warning during isolated validation.

- Cleaned up the default prod adapter suite after OpenRouter edge-model noise.
  `claude_adapter_gpt_oss_20b` and `claude_adapter_gpt_oss_120b` remain available as explicit `--cases` selections, but are excluded from the default suite because repeated OpenRouter provider-unavailable responses made default promotion runs noisy and slow. Harness `0.0.14` carries the updated defaults, and the clean default prod harness pass at `/tmp/litellm-prod-harness-aawm34-no-gptoss.json` completed with zero failures and zero warnings.

- Captured the next promotion hardening requirement.
  Future prod cutovers should include a production-style preflight against the exact built image / installed wheel path on `:4001`, plus a small explicit promotion-gate set for opt-in provider lanes, before touching the `:4000` container.

- Advanced the harness artifact to `h-v0.0.13` and prepared the follow-up `aawm.34` image tag.
  The `aawm.33` main push correctly triggered the harness artifact autobump because the prod hardening changed harness code. Since the image publisher requires fork image tags to point at current `main`, `aawm.34` carries the same NVIDIA metadata-preservation fix plus the `h-v0.0.13` artifact bump.

- Promoted the local prod `:4000` LiteLLM container to the `aawm.32` image and isolated the remaining NVIDIA blocker.
  Prod readiness reported LiteLLM `1.82.3+aawm.32`; `claude_adapter_peeromega_fanout`, `claude_adapter_openrouter_ling_26_flash`, and `claude_adapter_gpt_oss_120b` passed. Focused NVIDIA validation showed DeepSeek and GLM still landing Langfuse trace `environment=default` despite correct `session_history.litellm_environment=prod`, and MiniMax timed out at 300s.

- Fixed the NVIDIA trace-environment root cause for the next image release.
  The Anthropic-to-completion transformer now preserves completion-adapter metadata through to the inner `litellm.acompletion()` call, so `trace_environment`, `session_id`, and related LiteLLM metadata are available to normal completion callbacks instead of being dropped after `metadata.user_id` conversion.

- Promoted the local prod `:4000` LiteLLM container to the `aawm.31` image and isolated the remaining prod validation blockers.
  Prod readiness reported LiteLLM `1.82.3+aawm.31`; `claude_adapter_peeromega_fanout` and `claude_adapter_openrouter_ling_26_flash` passed. `claude_adapter_gpt_oss_120b` was blocked by OpenRouter `503 provider=OpenInference raw=no healthy upstream`, NVIDIA DeepSeek/GLM succeeded but exposed a Langfuse trace-environment propagation gap, and NVIDIA MiniMax remained too slow for the current 300s opt-in spot-check timeout.

- Hardened the next prod validation release for the remaining adapter issues.
  NVIDIA/OpenRouter completion adapters now mirror trace/session context from `litellm_metadata` into normal completion `metadata` so Langfuse trace environment matches session-history environment. Harness `0.0.12` adds a narrow soft-fail classifier for the exact OpenRouter provider-unavailable timeout signature while keeping other hard-gate timeouts hard.

- Promoted the local prod `:4000` LiteLLM container to the durable `aawm.28` image.
  The running `aawm-litellm` container now reports LiteLLM `1.82.3+aawm.28` with callback wheel `0.0.8` and control-plane wheel `0.0.5`.

- Completed prod Anthropic-adapter harness validation against `:4000`.
  Isolated `claude_adapter_gpt54`, `claude_adapter_gpt55`, and `claude_adapter_gpt54_mini` runs passed, followed by a clean full default prod harness pass at `/tmp/litellm-prod-harness-aawm28.json` with zero failures and zero warnings.

- Hardened Claude harness header injection against local Claude settings precedence.
  Harness version `0.0.8` carries the temporary per-run `--settings` overlay for the harness-controlled `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS`, so `x-litellm-end-user-id` / `langfuse_trace_user_id` validation no longer depends on ambient user or project settings.

- Removed brittle exact-response checks from basic OpenAI smoke cases.
  `gpt-5.4`, `gpt-5.5`, and `gpt-5.4-mini` smoke cases now hard-gate successful command completion, token/cost reporting, adapted-route tags, request payload logging, Langfuse trace/user/session context, runtime logs, and `session_history` rows without failing on harmless natural-language variation from Claude Code context.

## 2026-04-23

- Added LiteLLM runtime and initiating-client identity to `public.session_history`.
  New rows now persist `litellm_environment`, `litellm_version`, `litellm_fork_version`, `litellm_wheel_versions`, `client_name`, `client_version`, and `client_user_agent`; the same values are mirrored into row metadata for Langfuse/backfill context. Live writes derive these from trace environment metadata, installed package versions, associated overlay/config version env vars, request User-Agent, and Claude Code billing headers. Backfill paths only use values already present in spend-log/Langfuse metadata.

- Hardened the Anthropic adapter harness around the new runtime/client identity fields.
  Dev/prod target profile selection now injects `expected_litellm_environment` into session-history checks, and the harness fails rows missing runtime version, fork version, wheel-version JSON, client name, or client version. The callback overlay source remains byte-for-byte synced with the in-repo dev callback and its wheel version is now `0.0.8`.

- Advanced the callback, harness, and model-config artifact versions after branch promotion.
  Remote `cb-v0.0.7`, `h-v0.0.4`, and `cfg-v0.0.4` already existed on a release-only commit, so the durable release artifacts now use fresh non-rewritten versions: callback `0.0.8`, harness `0.0.5`, and model config `0.0.5`.

- Hardened the harness Claude trace-user setup for prod/dev parity.
  Harness version `0.0.6` injects a generated Claude trace-user header value and validates that exact value in Langfuse, without hard-coding an ambient operator identity.

- Advanced the fork image version to `1.82.3+aawm.27` for main-head promotion.
  The initial `v1.82.3-aawm.26` tag was cut before `main` converged, and the guarded image publisher rejects non-main-head tags; `aawm.27` avoids force-moving the published `aawm.26` tag.

- Advanced the fork image version to `1.82.3+aawm.28` after prod-cutover validation found a pass-through trace-user gap.
  Generic and Anthropic pass-through logging now resolve `user_api_key_end_user_id` from standard customer headers such as `x-litellm-end-user-id` when those headers are explicitly supplied, which lets the harness prove custom-header user identity flow end-to-end.

- Repaired `public.session_history` observability gaps in the local `aawm_tristore` database.
  The repair normalized null providers, removed invalid `reasoning_tokens_source=provider_reported` rows with zero reported reasoning tokens, populated target-provider cache statuses, and recalculated cache miss token/cost fields where usage exposed cache-write tokens.

- Hardened the session-history writer against the same regressions.
  New writes now infer missing providers from model/route metadata, sanitize non-positive reasoning token placeholders before persistence, default target-provider cache telemetry from stored cache counters, and preserve valid Gemini `provider_signature_present` reasoning signals.

- Synced the callback overlay source with the in-repo AAWM callback and bumped the callback wheel version to `0.0.6`.
  This prevents future `aawm-litellm` image rebuilds from reinstalling the stale callback wheel that omitted provider-cache and tool-activity fields.

- Broadened git tool-activity extraction.
  Command parsing now handles nested command payloads and git global options such as `git -C /repo commit` / `git --git-dir=/repo/.git push`, with regression coverage for parent `git_commit_count` / `git_push_count` rollups.

- Tightened adapter harness validation for session-history rows.
  Harness validation now selects file/git rollup fields and fails rows with null providers, null reasoning sources, invalid `provider_reported` zero counts, or missing provider-cache status on target provider families.

- Added literal ctx-marker escaping.
  Prompts can now use `\\:#name.ctx#\\:` to preserve visible `:#name.ctx#:` text without triggering `tristore_search_exact`, context appendix injection, or AAWM dynamic-injection metadata.

- Added default-suite harness coverage for escaped ctx markers and provider-cache activity.
  `claude_adapter_ctx_marker_escaped` validates the literal escape path, and `claude_adapter_peeromega_fanout` now requires at least one Anthropic child `session_history` row with `provider_cache_attempted=true` and `provider_cache_status` of `hit` or `write`.

- Added dev/prod target profiles to the Anthropic adapter harness.
  `--target dev` validates `:4001`, `litellm-dev`, and Langfuse trace environment `dev`; `--target prod` validates `:4000`, `aawm-litellm`, and Langfuse trace environment `prod`. The pass-through trace context helper now correctly preserves `session_id` and `trace_environment` metadata on rewritten requests.

- Fixed OpenRouter Anthropic-adapter Responses handling found during prod cutover.
  Native OpenRouter `/v1/responses` payloads now build a Responses-aware logging object instead of falling through OpenAI chat-completions parsing, preventing the prior async `KeyError: choices` logging exception while preserving Codex Responses `local_shell_call` output metadata. Translated Anthropic responses now strip stale upstream `content-length`, `content-encoding`, and `transfer-encoding` headers so synthesized bodies do not trigger `IncompleteRead` / `Too little data for declared Content-Length`. Validated on dev `:4001` with a direct `inclusionai/ling-2.6-flash:free` request returning a complete `200` response.

- Hardened the Anthropic adapter harness against the prod cutover failure modes.
  Runtime-log validation now defaults to failing on async task exceptions, ASGI exceptions, `KeyError: choices`, `h11` protocol errors, stale content-length failures, and upstream 429/5xx passthrough exceptions. Warning-only optional cases still allow configured quality mismatches, but no longer hide command timeouts or runtime-log hard failures. Added local harness unit coverage for timeout hard-fail behavior and default runtime-log detection.

- Fixed the shared OpenAI Responses function-schema failure that was blocking the dev harness.
  Anthropic-to-Responses tool translation now recursively normalizes nested object schemas, including `$defs` nodes that previously declared `type=object` without `properties`. That closes the upstream `400 Invalid schema for function ... object schema missing properties` failure seen on ChatGPT/Codex Responses traffic. Validated with focused passthrough regression tests, isolated live `claude_adapter_gpt54` / `claude_adapter_gpt54_mini` / `claude_adapter_ctx_marker` runs, and a clean full dev harness pass on `:4001`.

- Fixed OpenAI passthrough reconstruction for reasoning-only Responses output items.
  The passthrough logging handler now rebuilds valid `ModelResponse` objects when `response.completed` contains only `ResponseReasoningItem` output, including provider shapes that place reasoning text under `summary` or under `content[type=reasoning_text]`. That removes the prior `Error rebuilding complete responses API stream: Unknown items in responses API response: [ResponseReasoningItem(...)]` warning on `gpt-oss-120b` while preserving usage, hidden `responses_output`, and reasoning text on the reconstructed message. Validated with focused unit coverage plus an isolated live `claude_adapter_gpt_oss_120b` run on `:4001` and a clean overlapping `litellm-dev` log window.

- Added GPT-5.5 support to the Anthropic -> OpenAI Responses adapter.
  The adapter now allowlists `gpt-5.5` / `openai/gpt-5.5`, the local model cost maps include `gpt-5.5` and `chatgpt/gpt-5.5`, and the default Anthropic adapter harness includes `claude_adapter_gpt55`. The cost-map rates match the live Claude CLI `modelUsage` surfaced during validation. Validated with focused unit coverage, JSON/compile checks, and an isolated live `claude_adapter_gpt55` run on dev `:4001` with a clean overlapping `litellm-dev` log window.

## 2026-04-22

- Started direct NVIDIA adapter support for Anthropic-routed agent models.
  Added a dedicated `anthropic_nvidia_completion_adapter` lane for `nvidia/...` models targeting `nvidia:/v1/chat/completions`, normalized provider-prefix handling for `nvidia` / `nvidia_nim`, and compatibility aliasing for `nvidia/minimax/minimax-m2.7` -> `minimaxai/minimax-m2.7`.

- Added NVIDIA-specific egress-family and logging support.
  `integrate.api.nvidia.com` / `ai.api.nvidia.com` now classify as `nvidia` targets for outbound credential-guard checks and as OpenAI-compatible endpoints for passthrough logging / synthetic adapted-route reconstruction.

- Added NVIDIA env passthrough and model metadata needed for local validation.
  `docker-compose.dev.yml` now passes the NVIDIA API/base env vars through to `litellm-dev`, `model_prices_and_context_window.json` now has `nvidia_nim/...` entries for the new agent targets, and the local Anthropic adapter harness config now includes optional NVIDIA spot-check cases excluded from the default suite.

- Added focused unit coverage for the NVIDIA lane.
  Current tests cover NVIDIA egress validation, Anthropic adapter model resolution and route rewriting, and OpenAI-compatible logging host detection for NVIDIA completion endpoints.

- Hardened the NVIDIA completion adapter against provider-specific request and timeout issues.
  The route no longer forwards the empty `standard_callback_dynamic_params` payload into NVIDIA chat-completions requests, and it now applies a short per-attempt timeout plus hidden retry for transient `408` / `429` / `5xx` failures before surfacing an HTTP error.

- Stabilized the high-latency MiniMax path inside the NVIDIA Anthropic adapter.
  The NVIDIA lane now avoids nested OpenAI-provider retries by forcing the inner provider `max_retries` to `0`, uses a higher default adapter timeout, and fake-streams `minimaxai/minimax-m2.7` through Anthropic SSE while keeping the upstream NVIDIA call non-streaming.

- Established stable live NVIDIA spot checks on `:4001`.
  `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`, and `claude_adapter_nvidia_minimax_m27` now pass explicit end-to-end validation against `litellm-dev`, including `session_history` provider/model/cost validation, request-body or adapted-request checks, trace tag checks, and runtime-log traceback guards. They remain excluded from the default suite because the NVIDIA lane is still opt-in and MiniMax is materially slower than the other NVIDIA targets.

- Added normalized provider-cache telemetry to `public.session_history` for Anthropic, OpenAI, Gemini, and OpenRouter.
  Stored fields now include `provider_cache_attempted`, `provider_cache_status`, `provider_cache_miss`, and `provider_cache_miss_reason`.

- Added best-effort provider-cache miss quantification for explicit cache-write misses.
  When the miss token count is knowable from provider usage, `session_history` now also stores `provider_cache_miss_token_count` and `provider_cache_miss_cost_usd` using the write-vs-read delta for the affected tokens.

- Added [scripts/repair_session_history_provider_cache.py](/home/zepfu/projects/litellm/scripts/repair_session_history_provider_cache.py) for historical provider-cache repair when the original proxy spend-log source is not available locally.
  The script repairs `session_history` in place from the stored provider/model/cache counters and persisted metadata, and only prices miss cost when the missed cache token count is explicit enough to do so honestly.

- Added provider-cache metadata enrichment for Langfuse / request metadata.
  Current normalized states are `hit`, `write`, `miss`, `unsupported`, and `not_attempted`.

- Added targeted unit coverage for provider-cache detection from both direct LiteLLM result shaping and Langfuse-trace backfill shaping.

- Tightened the adapter harness and docs around current `session_history` invariants so reasoning, tool activity, and provider-cache signals stay visible during validation work on `:4001`.
