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

- D1-178 support xAI OAuth Grok models on native Codex and Anthropic endpoints
  - Added: 2026-06-02T00:16:14-04:00
  - Goal: make the enabled `oa_xai/*` Grok OAuth models usable through the
    native Codex/OpenAI passthrough surface and the native Anthropic messages
    adapter surface, not only through LiteLLM `/v1/chat/completions`.
    Requests should use normal client keys while LiteLLM supplies the managed
    xAI OAuth credential, preserves public `oa_xai/*` model identity, and
    chooses the correct upstream xAI Chat Completions or Responses endpoint per
    model capability.
  - Main references:
    - D1-176 in `.analysis/completed.md`: `oa_xai/grok-4.3`,
      `oa_xai/grok-4.20-0309-reasoning`, and
      `oa_xai/grok-4.20-0309-non-reasoning` live-smoked successfully through
      chat completions with managed xAI OAuth.
    - D1-177 in `.analysis/completed.md`:
      `oa_xai/grok-4.20-multi-agent-0309` is Responses-backed, live-smoked
      successfully through the chat-to-Responses bridge, and records
      `call_type=responses`.
    - `litellm/llms/xai/oauth.py` owns `oa_xai/*` public-to-upstream model
      mapping, managed OAuth token injection, and `xai_oauth_*` metadata.
    - `litellm/proxy/route_llm_request.py` currently prepares `oa_xai/*`
      requests for normal LiteLLM route types.
    - Native Codex/OpenAI passthrough paths and acceptance harnesses live under
      `litellm/proxy/pass_through_endpoints/`,
      `scripts/local-ci/run_anthropic_adapter_acceptance.py`, and
      `scripts/local-ci/anthropic_adapter_config.json`.
    - Native Anthropic messages adapter behavior is covered in
      `litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py`
      and `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`.
  - Acceptance evidence:
    - Add focused offline tests proving native Codex/OpenAI passthrough requests
      can target the enabled `oa_xai/*` model names and are rewritten to xAI
      with managed OAuth, preserved public model/model_group semantics, quota
      metadata, and the correct upstream endpoint family.
    - Add focused offline tests proving native Anthropic `/anthropic/v1/messages`
      adapter requests can target the same `oa_xai/*` model names, with
      adapter metadata preserving the original Anthropic/Codex-facing model and
      xAI OAuth route family.
    - Prove `oa_xai/grok-4.20-multi-agent-0309` uses xAI `/responses`, not
      `/chat/completions`, through both native endpoint families.
    - Prove the three D1-176 chat-capable `oa_xai/*` models keep their chat
      behavior and do not regress to the multi-agent Responses bridge.
    - Run live dev-proxy smokes with only a LiteLLM client key for:
      native Codex/OpenAI passthrough to at least one chat-capable `oa_xai/*`
      model and the multi-agent Responses-backed model; and native Anthropic
      messages adapter to at least one chat-capable `oa_xai/*` model and the
      multi-agent Responses-backed model.
    - Query exact `aawm_tristore.public.session_history` rows for generated
      live session IDs and verify provider `xai`, public `oa_xai/*`
      model/model_group semantics, correct `call_type`, token/cost fields,
      `xai_oauth_*` metadata, adapter route metadata, and quota-family metadata.
  - Known hazards:
    - Do not create a second xAI OAuth credential surface or duplicate token
      refresh path; reuse `litellm/llms/xai/oauth.py`.
    - Do not expose managed xAI bearer tokens to passthrough clients or logs.
    - Native Codex/OpenAI and Anthropic adapters may have different model-name
      normalization rules; preserve the client-facing public `oa_xai/*` model
      while routing upstream as `xai/*`.
    - Responses-backed multi-agent routing must not break existing
      `oa_xai/*` chat aliases or unrelated OpenAI/Anthropic adapter models.
    - If this changes reporting/observability schema fields, create the
      required sibling dashboard-shell handoff before completion.

- D1-179 validate xAI OAuth quota details in provider rate limits
  - Added: 2026-06-02T00:25:11-04:00
  - Goal: validate and, if needed, implement collection and exposure of quota
    details for the `oa_xai/*` managed OAuth path so native Codex/OpenAI and
    native Anthropic traffic can appear in provider rate-limit telemetry with
    semantics comparable to Grok Build OIDC/native traffic. The target surface
    is provider rate limits backed by `public.rate_limit_observations`, not
    only `session_history` quota-family metadata.
  - Main references:
    - D1-177 in `.analysis/completed.md`: the live multi-agent OAuth smoke
      recorded `shared_quota_family=xai_grok_subscription` in
      `public.session_history`, but the exact
      `public.rate_limit_observations` query for that session returned no
      rows because the table currently records provider quota headers only.
    - D1-103/D1-107 in `.analysis/completed.md`: Grok Build `/grok/v1/billing`
      payloads are normalized into provider rate-limit rows with
      `provider=xai`, `client=grok-build`, `model=grok-build`,
      `quota_key=xai_grok_build_monthly_requests:requests`,
      `quota_type=requests`, `source=grok_billing`, and a billing-period
      reset timestamp.
    - D1-173 and D1-107 in `.analysis/completed.md`: native Grok Build
      pass-through traffic carries `passthrough_route_family=grok_cli_chat_proxy`
      and already has a test/live contract for Grok billing quota metadata.
    - `litellm/llms/xai/oauth.py` owns `oa_xai/*` mapping and managed OAuth
      metadata; `litellm/integrations/aawm_agent_identity.py` and
      `.wheel-build/aawm_litellm_callbacks/agent_identity.py` own normalized
      rate-limit observation capture.
    - `litellm/proxy/pass_through_endpoints/success_handler.py`,
      `scripts/backfill_rate_limit_observations.py`,
      `tests/test_litellm/integrations/test_aawm_agent_identity.py`, and
      `tests/test_scripts/test_backfill_rate_limit_observations.py` cover the
      current provider-rate-limit ingestion and backfill paths.
  - Acceptance evidence:
    - Inventory the exact quota signals available on the xAI OAuth path from
      live native Codex/OpenAI and native Anthropic adapter traffic: response
      headers, response bodies, billing endpoint access, OAuth account
      metadata, or absence of all quota-bearing fields. Do not infer parity
      from `shared_quota_family` alone.
    - Add focused offline tests proving any xAI OAuth quota headers, billing
      payloads, or equivalent managed-OAuth quota metadata are parsed into the
      same normalized provider-rate-limit semantics used by Grok Build OIDC
      where possible, including stable `provider`, `client`, `model`,
      `quota_key`, `quota_type`, `remaining_pct`, `expected_reset_at`,
      `source`, account hashing, and duplicate-suppression behavior.
    - Run live dev-proxy smokes through the native Codex/OpenAI passthrough
      and native Anthropic adapter using only a LiteLLM client key for at
      least one chat-capable `oa_xai/*` model and the Responses-backed
      `oa_xai/grok-4.20-multi-agent-0309` model.
    - Query exact `aawm_tristore.public.session_history` rows for those live
      session IDs and verify public `oa_xai/*` model/model_group semantics,
      `xai_oauth_*` metadata, adapter route metadata, and
      `shared_quota_family=xai_grok_subscription`.
    - Query exact `aawm_tristore.public.rate_limit_observations` rows for the
      same sessions or latest matching xAI OAuth quota snapshots and verify
      they are visible to the provider-rate-limit surface. If xAI OAuth does
      not expose quota headers or billing data, record the live negative
      evidence explicitly and keep the provider-rate-limit surface honest by
      exposing only the closest available quota-family/status metadata.
    - If new columns, quota keys, client names, or interpretation rules are
      added for provider rate limits, create the required sibling
      dashboard-shell handoff before marking this complete.
  - Known hazards:
    - Do not silently mix Grok Build OIDC quota keys with `oa_xai/*` OAuth
      quota keys unless live evidence proves they are the same account and
      reset boundary.
    - Do not expose managed xAI OAuth bearer tokens, account identifiers, or
      raw billing payloads in logs, database rows, dashboard payloads, or test
      fixtures; hash account identifiers consistently.
    - Provider-rate-limit rows should remain normalized snapshots, not raw
      per-request dumps, and should preserve existing duplicate-suppression
      behavior.
    - Native Codex/OpenAI and Anthropic adapter request paths may surface
      different response metadata; validate both before claiming coverage.

## Proposals (Pending Operator Feedback)

No current proposals.
