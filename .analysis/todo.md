## Standing Directives

- Investigation intake and disposition
  - At the start of TODO-driven work, look for `.analysis/investigate-*.md`
    files.
  - For each investigation file, determine whether the failure mode points to:
    message shape/form problems between the orchestrating client, LiteLLM, and
    the destination LLM; a LiteLLM implementation defect; a deterministic
    failure mode that should become a new score; or an orchestrating agent
    session-level issue such as prompting or dispatch instructions.
  - After an investigation has been examined and dispositioned, move the
    original `investigate-*.md` file into `.analysis/investigation/` so the
    active investigation intake stays clean.
  - Create `.analysis/investigations.md` if it does not exist and append one
    entry for every disposition, including non-actionable outcomes. Each entry
    must include the original investigation filename, the outcome, any proposed
    follow-up item IDs, and the post-move file location under
    `.analysis/investigation/`.
  - Any follow-up action that should be taken from an investigation must be
    added under `Proposals (Pending Operator Feedback)` until the operator
    explicitly approves turning it into active TODO work.

- Dashboard reporting handoffs for observability schema changes
  - Any time this repo adds new structure to
    `aawm_tristore.public.session_history`, rate-limit/rate observation tables,
    provider health tables/views, or related reporting/observability tables,
    create or update a sibling dashboard handoff document under
    `/home/zepfu/projects/dashboard-shell/.analysis/handoff-*.md`.
  - The handoff must name changed tables/columns, field semantics,
    interpretation rules, aggregation guidance, API/reporting impact, and the
    dashboard surfaces that need to incorporate the new structure.
  - Do this before treating the LiteLLM-side change as complete, so dashboard
    model/repository reporting does not silently miss new data.

## Current Open Items

Deferred work lives in `.analysis/todo.deferred.md`.

- D1-123 xAI/Grok partial-cache-hit miss-cost reporting gap
  - Promoted from deferred reporting semantics backlog.
  - Initiated: 2026-06-01 13:39:53 EDT.
  - Decide and implement the intended semantics for xAI/Grok Build
    `provider_cache_miss_cost_usd` when a provider reports a cache hit for only
    part of the prompt.
  - Main references:
    - `litellm/integrations/aawm_agent_identity.py`
    - `scripts/repair_session_history_provider_cache.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `model_prices_and_context_window.json`
  - Known hazards:
    - Generic/Gemini cache-hit tests intentionally assert that miss-token/cost
      fields stay null, so a global semantic change would alter the reporting
      contract.
    - `response_cost_usd` already appears to include xAI cache-read pricing; do
      not double-count this analytic miss-cost field as billed spend.

- D1-107 native Grok Build pass-through final live validation
  - Promoted from deferred live-validation backlog.
  - Current state:
    - Local implementation, dev route load, header expansion, protobuf raw-body
      passthrough, prod route cutover, xAI provider health/error capture,
      Grok billing quota capture, and embedding identity enrichment are
      documented in `.analysis/completed.md`.
  - Remaining validation:
    - Prove a fresh Grok embedding row appears in exact database
      `aawm_tristore.public.session_history`.
    - Prove a fresh post-pricing Grok Build online path produces non-empty cost
      without backfill.
    - Run a real Grok Build prod smoke.
  - Main references:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
  - Known hazards:
    - Keep LiteLLM auth separate from Grok OIDC/xAI headers. Do not leak
      `x-litellm-api-key` upstream.
    - Grok CLI auth/header behavior is client-version-sensitive; use live
      request evidence when validating.

- D1-172 xAI OAuth-managed Grok model routing through normal LiteLLM model calls
  - Promoted from deferred auth/access-path expansion backlog.
  - Goal: expose selected xAI Grok models through normal LiteLLM model calls
    using LiteLLM-managed OAuth credentials, so clients call `oa_xai/*` with
    only their LiteLLM key while LiteLLM loads, refreshes, and injects the xAI
    OAuth bearer token.
  - Main references:
    - `.analysis/xai-oauth-managed-models-plan-2026-05-18.md`
    - `litellm/proxy/route_llm_request.py`
    - `litellm/router.py`
    - `litellm/llms/xai/`
    - `model_prices_and_context_window.json`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
  - Acceptance evidence:
    - `oa_xai/grok-4.3`, `oa_xai/grok-4.20-0309-reasoning`,
      `oa_xai/grok-4.20-0309-non-reasoning`, and
      `oa_xai/grok-4.20-multi-agent-0309` map internally to the matching
      `xai/*` upstream models on `https://api.x.ai/v1`.
    - The OAuth credential no longer has to live in `~/.hermes/auth.json`; the
      first implementation uses a LiteLLM-owned credential source such as
      `LITELLM_XAI_OAUTH_AUTH_FILE`, encrypted credential storage, or a
      deployment secret.
    - LiteLLM refreshes expired or near-expired xAI OAuth credentials with
      refresh serialization, and returns a clear reseed/relogin error for
      terminal OAuth failures.
    - `/grok/{endpoint:path}` native Grok Build pass-through still forwards
      client-supplied Grok/OIDC headers exactly as before.
    - `session_history` records `provider=xai` plus clear auth/access metadata
      such as `auth_mode=oauth`, `credential_family=xai_oauth`, and a route
      family distinct from `grok_cli_chat_proxy`.
    - Rate-limit/quota observations roll up to the shared Grok subscription
      bucket rather than inventing separate `oa_xai/*` capacity.
  - Known hazards:
    - Do not collapse this into plain `xai/*`; the public namespace should make
      OAuth-managed routing explicit.
    - Do not make clients read Hermes-local files.
    - Do not treat OAuth-managed `oa_xai/*` and native Grok Build OAuth as
      independent quota pools.
    - Keep API-key `xai/*` behavior separate from OAuth-managed `oa_xai/*`
      behavior.

- D1-173 Grok Build pass-through adapter test harness
  - Promoted from deferred harness automation backlog.
  - Goal: add a reusable test harness for native Grok Build pass-through that
    exercises request shape, auth/header separation, protobuf/raw-body
    passthrough, streaming/final response handling, and observability
    persistence without relying only on ad hoc prod smokes.
  - Main references:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.analysis/xai/`
    - `tests/test_litellm/proxy/pass_through_endpoints/`
  - Acceptance evidence:
    - Harness fixtures can replay representative Grok Build CLI requests with
      Grok/OIDC headers and verify `x-litellm-api-key` is never forwarded
      upstream.
    - Harness covers protobuf/raw-body passthrough, streaming output, final
      response persistence, provider error capture, and quota/rate-limit
      observation metadata.
    - Harness validates `session_history` identity for native Grok Build rows:
      `provider=xai`, `model=grok-build`, `client_name=grok-build`, and
      `passthrough_route_family=grok_cli_chat_proxy`.
    - Harness supports a live-smoke mode gated by explicit credentials and a
      deterministic offline mode suitable for CI.
  - Known hazards:
    - Grok CLI auth/header shape is client-version-sensitive; fixture coverage
      must include client version evidence and avoid hardcoding stale headers.
    - Keep native Grok Build pass-through separate from OAuth-managed
      `oa_xai/*` model routing.
    - Avoid recording or committing bearer/OIDC tokens in fixtures.

- D1-174 OAuth-managed xAI Grok model test harness
  - Promoted from deferred harness automation backlog.
  - Goal: add test-harness coverage for the planned `oa_xai/*` OAuth-managed
    model path, including credential loading, refresh behavior, route mapping,
    quota semantics, and session-history/provider-error observability.
  - Main references:
    - `.analysis/xai-oauth-managed-models-plan-2026-05-18.md`
    - `litellm/llms/xai/`
    - `litellm/router.py`
    - `litellm/proxy/route_llm_request.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/`
  - Acceptance evidence:
    - Offline harness verifies `oa_xai/*` routes map to the intended `xai/*`
      upstream models while preserving the public requested model where the
      chosen reporting contract requires it.
    - Harness exercises OAuth credential load, near-expiry refresh,
      refresh-lock serialization, terminal refresh failure, and missing
      credential errors without using real secrets in fixtures.
    - Harness verifies clients only provide LiteLLM credentials and LiteLLM
      injects upstream xAI OAuth authorization.
    - Harness validates `session_history`, provider-error observations, and
      rate-limit/quota observations for `auth_mode=oauth`,
      `credential_family=xai_oauth`, shared Grok subscription quota, token
      counts, and non-empty cost.
    - Live-smoke mode can call each enabled `oa_xai/*` model behind an explicit
      opt-in flag and records exact DB evidence in `aawm_tristore`.
  - Known hazards:
    - Do not depend on `~/.hermes/auth.json` in tests; use synthetic credential
      fixtures or a LiteLLM-owned test secret path.
    - Do not conflate OAuth-managed model calls with native Grok Build
      `/grok/...` pass-through.
    - Redact all OAuth tokens and refresh tokens from logs, fixtures, and
      assertion diffs.

## Proposals (Pending Operator Feedback)

No current proposals.
