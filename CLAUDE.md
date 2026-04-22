# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
- `make install-dev` - Install core development dependencies
- `make install-proxy-dev` - Install proxy development dependencies with full feature set
- `make install-test-deps` - Install all test dependencies

### Testing
- `make test` - Run all tests
- `make test-unit` - Run unit tests (tests/test_litellm) with 4 parallel workers
- `make test-integration` - Run integration tests (excludes unit tests)
- `pytest tests/` - Direct pytest execution

### Code Quality
- `make lint` - Run all linting (Ruff, MyPy, Black, circular imports, import safety)
- `make format` - Apply Black code formatting
- `make lint-ruff` - Run Ruff linting only
- `make lint-mypy` - Run MyPy type checking only

### Single Test Files
- `./.venv/bin/pytest tests/path/to/test_file.py -v` - Run specific test file
- `./.venv/bin/pytest tests/path/to/test_file.py::test_function -v` - Run specific test

### Running Scripts
- `./.venv/bin/python script.py` - Run Python scripts (use for non-test files)

### GitHub Issue & PR Templates
When contributing to the project, use the appropriate templates:

**Bug Reports** (`.github/ISSUE_TEMPLATE/bug_report.yml`):
- Describe what happened vs. what you expected
- Include relevant log output
- Specify your LiteLLM version

**Feature Requests** (`.github/ISSUE_TEMPLATE/feature_request.yml`):
- Describe the feature clearly
- Explain the motivation and use case

**Pull Requests** (`.github/pull_request_template.md`):
- Add at least 1 test in `tests/litellm/`
- Ensure `make test-unit` passes

## Architecture Overview

LiteLLM is a unified interface for 100+ LLM providers with two main components:

### Core Library (`litellm/`)
- **Main entry point**: `litellm/main.py` - Contains core completion() function
- **Provider implementations**: `litellm/llms/` - Each provider has its own subdirectory
- **Router system**: `litellm/router.py` + `litellm/router_utils/` - Load balancing and fallback logic
- **Type definitions**: `litellm/types/` - Pydantic models and type hints
- **Integrations**: `litellm/integrations/` - Third-party observability, caching, logging
- **Caching**: `litellm/caching/` - Multiple cache backends (Redis, in-memory, S3, etc.)

### Proxy Server (`litellm/proxy/`)
- **Main server**: `proxy_server.py` - FastAPI application
- **Authentication**: `auth/` - API key management, JWT, OAuth2
- **Database**: `db/` - Prisma ORM with PostgreSQL/SQLite support
- **Management endpoints**: `management_endpoints/` - Admin APIs for keys, teams, models
- **Pass-through endpoints**: `pass_through_endpoints/` - Provider-specific API forwarding
- **Guardrails**: `guardrails/` - Safety and content filtering hooks
- **UI Dashboard**: Served from `_experimental/out/` (Next.js build)

## Key Patterns

### Provider Implementation
- Providers inherit from base classes in `litellm/llms/base.py`
- Each provider has transformation functions for input/output formatting
- Support both sync and async operations
- Handle streaming responses and function calling

### Error Handling
- Provider-specific exceptions mapped to OpenAI-compatible errors
- Fallback logic handled by Router system
- Comprehensive logging through `litellm/_logging.py`

### Configuration
- YAML config files for proxy server (see `proxy/example_config_yaml/`)
- Environment variables for API keys and settings
- Database schema managed via Prisma (`proxy/schema.prisma`)

## Development Notes

### Code Style
- Uses Black formatter, Ruff linter, MyPy type checker
- Pydantic v2 for data validation
- Async/await patterns throughout
- Type hints required for all public APIs
- **Avoid imports within methods** — place all imports at the top of the file (module-level). Inline imports inside functions/methods make dependencies harder to trace and hurt readability. The only exception is avoiding circular imports where absolutely necessary.
- **Use dict spread for immutable copies** — prefer `{**original, "key": new_value}` over `dict(obj)` + mutation. The spread produces the final dict in one step and makes intent clear.
- **Guard at resolution time** — when resolving an optional value through a fallback chain (`a or b or ""`), raise immediately if the resolved result being empty is an error. Don't pass empty strings or sentinel values downstream for the callee to deal with.
- **Extract complex comprehensions to named helpers** — a set/dict comprehension that calls into the DB or manager (e.g. "which of these server IDs are OAuth2?") belongs in a named helper function, not inline in the caller.
- **FastAPI parameter declarations** — mark required query/form params with `= Query(...)` / `= Form(...)` explicitly when other params in the same handler are optional. Mixing `str` (required) with `Optional[str] = None` in the same signature causes silent 422s when the required param is missing.

### Anthropic adapter dev validation
- `litellm-dev` on `:4001` is the only supported runtime for Anthropic-route adapter work.
- Native Anthropic egress is enabled again on `:4001`, but top-level Claude runs without an adapted model are still not useful acceptance targets for this adapter work because they do not exercise the adapted lane.
- Real-Claude adapter acceptance lives in:
  - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
  - `scripts/local-ci/anthropic_adapter_config.json`
- Current real-Claude adapted baseline on `:4001`:
  - OpenAI/Codex hard gates: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex-spark`
  - Codex/OpenAI tool-activity hard gate: `claude_adapter_codex_tool_activity`
    - validates the reconstructed `response.output_item.*` /
      `response.function_call_arguments.*` stream path
    - must persist a `Bash` / `pwd` row into
      `public.session_history_tool_activity`
  - Dynamic context hard gate: `claude_adapter_ctx_marker`
    - validates the canonical `:#port-allocation.ctx#:` stored-procedure
      rewrite path on `:4001`
  - Gemini fanout hard gate: `claude_adapter_gemini_fanout`
    - isolates the exact multi-Gemini subagent dispatch path on `:4001`
    - use it before re-running the full adapter suite when Gemini fanout is the suspected regression
    - the full suite runs this before `claude_adapter_peeromega_fanout` so the dedicated Gemini gate is not polluted by the mixed fanout's short-window upstream pressure
    - stable tool-activity invariant: expect session-wide delegated `Agent`
      rows plus at least one Gemini command tool row; do not assume every
      Gemini child model will emit its own command row on every run
  - Google Code Assist canaries: `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`
    - the adapter routes Gemini Anthropic-adapter models directly to Google Code Assist
    - keep this warning-only in the harness because upstream quota windows produce real `429` responses
  - OpenRouter hard gate: `openai/gpt-oss-120b:free`
  - for OpenRouter-adapted cases, rely on trace tags/metadata plus `session_history`; do not hard-gate on Langfuse generation usage fields yet
  - OpenRouter preferred free targets under active validation: `inclusionai/ling-2.6-flash:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-super-120b-a12b:free`
  - OpenRouter warning-only canaries: `openrouter/free`, `inclusionai/ling-2.6-flash:free`, `openai/gpt-oss-20b:free`, `google/gemma-4-31b-it:free`, `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-super-120b-a12b:free`
- OpenRouter manual-only spot checks for now: `meta-llama/llama-3.3-70b-instruct:free`, `minimax/minimax-m2.5:free`
- Current upstream-rate-limited / unstable OpenRouter candidates:
  - `openrouter/free` (moving router)

### Model cost map process
- `model_prices_and_context_window.json` is the only canonical editable model/cost map.
- `litellm/bundled_model_prices_and_context_window_fallback.json` is the packaged fallback mirror used for `LITELLM_LOCAL_MODEL_COST_MAP=True`.
- After changing the canonical file, run `make sync-model-cost-map` and avoid hand-editing the packaged fallback directly.
  - `inclusionai/ling-2.6-flash:free`
  - `google/gemma-4-31b-it:free`
  - `google/gemma-4-26b-a4b-it:free`
  - `nvidia/nemotron-3-super-120b-a12b:free`
  - `minimax/minimax-m2.5:free`
  - `qwen/qwen3-coder:free`
- For adapted models, treat LiteLLM / `session_history` / Langfuse as the cost source of truth, not Claude CLI display cost.
- The current Gemini CLI bundle and the Anthropic adapter use the same Code Assist request envelope: `model`, `project`, `user_prompt_id`, and `request` with `session_id` / `contents` / tools / generation config. If standalone Gemini CLI use is healthy but `claude_adapter_gemini_fanout` fails, treat that first as a local pacing/serialization bug rather than authoritative provider-capacity proof.
- For Google Code Assist adapter work, treat successful real-Claude runs on `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, and `gemini-3.1-flash-lite-preview` as proof of routing correctness. Do not treat `429` / `RESOURCE_EXHAUSTED` / `MODEL_CAPACITY_EXHAUSTED` on their own as authoritative upstream truth; only close those as provider issues after interactive Gemini CLI `/model` corroboration on the same account context.
- `inclusionai/ling-2.6-flash:free` stays on the generic Anthropic -> OpenRouter `Responses` lane with the other `vendor/model:free` targets.
- Dev OpenRouter pacing on `:4001` now uses:
  - short hidden retry budget: `AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS=12`
  - longer per-model post-failure cooldown: `AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS=300`
- Preferred Anthropic-adapter model spellings:
  - direct OpenAI targets: `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.3-codex-spark`
  - direct Google Code Assist targets: `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`
  - direct OpenRouter targets: `openrouter/openai/gpt-oss-120b:free`, `openrouter/inclusionai/ling-2.6-flash:free`, `openrouter/google/gemma-4-31b-it:free`
  - legacy unprefixed or vendor-only spellings still resolve for compatibility, but explicit provider prefixes are preferred because adapter routing is provider-first
  This keeps brief transient recovery local while preventing repeated manual retests from re-burning ~40 seconds on the same throttled backend.
- Anthropic fanout prompts should still use the Claude agent `name:` values from
  `~/.claude/agents` such as `gemini-3-flash-preview` and `gpt-5-4`. The
  provider-prefixed routing lives in the agent file `model:` value.
- `ling-2-6-flash` now validates on the generic OpenRouter `Responses` lane, so its Langfuse / `session_history` shaping should match the other free-model response adapters.

### Runtime performance knobs
- Payload capture is now intentionally gated for debug-only use. `litellm.integrations.aawm_payload_capture` will write captures only when both `AAWM_CAPTURE=1` and `LITELLM_LOG=DEBUG` are set. Normal work should leave this off.
- `session_history` persistence now uses a background batch writer instead of inline per-call writes. Tune with:
  - `AAWM_SESSION_HISTORY_BATCH_SIZE` (default `32`)
  - `AAWM_SESSION_HISTORY_FLUSH_INTERVAL_MS` (default `250`)
- Claude dynamic directive expansion now uses a short TTL cache to avoid repeated DB hits on the same `(session, agent, tenant)` context. Tune with:
  - `AAWM_DYNAMIC_INJECTION_CACHE_TTL_SECONDS` (default `15`)
- Current low-overhead instrumentation surfaces for these optimizations:
  - DEBUG log: `AawmAgentIdentity: flushed N session_history records in Xms`
  - Claude rewrite metadata: `aawm_dynamic_injection_cache_hits`, `aawm_dynamic_injection_cache_misses`, `aawm_dynamic_injection_cache_statuses`
  - existing proxy metadata: `queue_time_seconds`, `completion_start_time`, `response_cost`

### Testing Strategy
- Unit tests in `tests/test_litellm/`
- Integration tests for each provider in `tests/llm_translation/`
- Proxy tests in `tests/proxy_unit_tests/`
- Load tests in `tests/load_tests/`
- **Always add tests when adding new entity types or features** — if the existing test file covers other entity types, add corresponding tests for the new one
- **Keep monkeypatch stubs in sync with real signatures** — when a function gains a new optional parameter, update every `fake_*` / `stub_*` in tests that patch it to also accept that kwarg (even as `**kwargs`). Stale stubs fail with `unexpected keyword argument` and mask real bugs.
- **Test all branches of name→ID resolution** — when adding server/resource lookup that resolves names to UUIDs, test: (1) name resolves and UUID is allowed, (2) name resolves but UUID is not allowed, (3) name does not resolve at all. The silent-fallback path is where access-control bugs hide.

### UI / Backend Consistency
- When wiring a new UI entity type to an existing backend endpoint, verify the backend API contract (single value vs. array, required vs. optional params) and ensure the UI controls match — e.g., use a single-select dropdown when the backend accepts a single value, not a multi-select

### MCP OAuth / OpenAPI Transport Mapping
- `TRANSPORT.OPENAPI` is a UI-only concept. The backend only accepts `"http"`, `"sse"`, or `"stdio"`. Always map it to `"http"` before any API call (including pre-OAuth temp-session calls).
- FastAPI validation errors return `detail` as an array of `{loc, msg, type}` objects. Error extractors must handle: array (map `.msg`), string, nested `{error: string}`, and fallback.
- When an MCP server already has `authorization_url` stored, skip OAuth discovery (`_discovery_metadata`) — the server URL for OpenAPI MCPs is the spec file, not the API base, and fetching it causes timeouts.
- `client_id` should be optional in the `/authorize` endpoint — if the server has a stored `client_id` in credentials, use that. Never require callers to re-supply it.

### MCP Credential Storage
- OAuth credentials and BYOK credentials share the `litellm_mcpusercredentials` table, distinguished by a `"type"` field in the JSON payload (`"oauth2"` vs plain string).
- When deleting OAuth credentials, check type before deleting to avoid accidentally deleting a BYOK credential for the same `(user_id, server_id)` pair.
- Always pass the raw `expires_at` timestamp to the client — never set it to `None` for expired credentials. Let the frontend compute the "Expired" display state from the timestamp.
- Use `RecordNotFoundError` (not bare `except Exception`) when catching "already deleted" in credential delete endpoints.

### Browser Storage Safety (UI)
- Never write LiteLLM access tokens or API keys to `localStorage` — use `sessionStorage` only. `localStorage` survives browser close and is readable by any injected script (XSS).
- Shared utility functions (e.g. `extractErrorMessage`) belong in `src/utils/` — never define them inline in hooks or duplicate them across files.

### Database Migrations
- Prisma handles schema migrations
- Migration files auto-generated with `prisma migrate dev`
- Always test migrations against both PostgreSQL and SQLite

### Proxy database access
- **Do not write raw SQL** for proxy DB operations. Use Prisma model methods instead of `execute_raw` / `query_raw`.
- Use the generated client: `prisma_client.db.<model>` (e.g. `litellm_tooltable`, `litellm_usertable`) with `.upsert()`, `.find_many()`, `.find_unique()`, `.update()`, `.update_many()` as appropriate. This avoids schema/client drift, keeps code testable with simple mocks, and matches patterns used in spend logs and other proxy code.
- **No N+1 queries.** Never query the DB inside a loop. Batch-fetch with `{"in": ids}` and distribute in-memory.
- **Batch writes.** Use `create_many`/`update_many`/`delete_many` instead of individual calls (these return counts only; `update_many`/`delete_many` no-op silently on missing rows). When multiple separate writes target the same table (e.g. in `batch_()`), order by primary key to avoid deadlocks.
- **Push work to the DB.** Filter, sort, group, and aggregate in SQL, not Python. Verify Prisma generates the expected SQL — e.g. prefer `group_by` over `find_many(distinct=...)` which does client-side processing.
- **Bound large result sets.** Prisma materializes full results in memory. For results over ~10 MB, paginate with `take`/`skip` or `cursor`/`take`, always with an explicit `order`. Prefer cursor-based pagination (`skip` is O(n)). Don't paginate naturally small result sets.
- **Limit fetched columns on wide tables.** Use `select` to fetch only needed fields — returns a partial object, so downstream code must not access unselected fields.
- **Check index coverage.** For new or modified queries, check `schema.prisma` for a supporting index. Prefer extending an existing index (e.g. `@@index([a])` → `@@index([a, b])`) over adding a new one, unless it's a `@@unique`. Only add indexes for large/frequent queries.
- **Keep schema files in sync.** Apply schema changes to all `schema.prisma` copies (`schema.prisma`, `litellm/proxy/`, `litellm-proxy-extras/`, `litellm-js/spend-logs/` for SpendLogs) with a migration under `litellm-proxy-extras/litellm_proxy_extras/migrations/`.

### Enterprise Features
- Enterprise-specific code in `enterprise/` directory
- Optional features enabled via environment variables
- Separate licensing and authentication for enterprise features

### HTTP Client Cache Safety
- **Never close HTTP/SDK clients on cache eviction.** `LLMClientCache._remove_key()` must not call `close()`/`aclose()` on evicted clients — they may still be used by in-flight requests. Doing so causes `RuntimeError: Cannot send a request, as the client has been closed.` after the 1-hour TTL expires. Cleanup happens at shutdown via `close_litellm_async_clients()`.

### Troubleshooting: DB schema out of sync after proxy restart
`litellm-proxy-extras` runs `prisma migrate deploy` on startup using **its own** bundled migration files, which may lag behind schema changes in the current worktree. Symptoms: `Unknown column`, `Invalid prisma invocation`, or missing data on new fields.

**Diagnose:** Run `\d "TableName"` in psql and compare against `schema.prisma` — missing columns confirm the issue.

**Fix options:**
1. **Create a Prisma migration** (permanent) — run `prisma migrate dev --name <description>` in the worktree. The generated file will be picked up by `prisma migrate deploy` on next startup.
2. **Apply manually for local dev** — `psql -d litellm -c "ALTER TABLE ... ADD COLUMN IF NOT EXISTS ..."` after each proxy start. Fine for dev, not for production.
3. **Update litellm-proxy-extras** — if the package is installed from PyPI, its migration directory must include the new file. Either update the package or run the migration manually until the next release ships it.
