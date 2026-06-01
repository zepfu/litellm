## 2026-06-01

- Investigation intake `investigate-codex-019e857b-9e88-7f53-90f9-acef3062b597.md`
  - Goal: disposition the new root investigation intake created after the
    D1-176 push.
  - Initiated: 2026-06-01 19:21:29 EDT.
  - Completed: 2026-06-01 19:21:43 EDT.
  - Duration: 14 seconds.
  - Changed paths:
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e857b-9e88-7f53-90f9-acef3062b597.md`
    - `.analysis/completed.md`
    - `suggestion.md`
  - Breakdown:
    - Classified the intake as a context-only session-level subagent read-only
      violation from `/home/zepfu/projects/aawm-tap-dashboard`, not a LiteLLM
      implementation defect.
    - Recorded the failure in `.analysis/investigations.md`, moved the original
      file into `.analysis/investigation/`, and left no follow-up proposal.
  - Verification evidence:
    - `find .analysis -maxdepth 1 -type f \( -name 'handoff*.md' -o -name
      'request*.md' -o -name 'investigate*.md' -o -name 'investigate-*.md' \)
      -printf '%f\n' | sort` returned no files after archival.
    - `git diff --cached --check` passed before the follow-up commit.

- D1-176 xAI OAuth Hermes credential migration and live `oa_xai/*` validation
  - Goal: close the D1-172/D1-174 live-acceptance gap by migrating the real
    Hermes xAI OAuth credential into a LiteLLM-owned path, wiring the dev proxy
    to that path, and proving normal LiteLLM client calls to `oa_xai/*` reach
    xAI with OAuth-backed credentials.
  - Initiated: 2026-06-01 16:26:57 EDT.
  - Completed: 2026-06-01 19:15:58 EDT.
  - Duration: 2 hours 49 minutes 1 second.
  - Changed paths:
    - `docker-compose.dev.yml`
    - `litellm/llms/xai/oauth.py`
    - `litellm/proxy/route_llm_request.py`
    - `scripts/migrate_xai_oauth_credential.py`
    - `tests/llm_translation/test_xai.py`
    - `tests/test_litellm/proxy/test_oa_xai_harness.py`
    - `tests/test_litellm/proxy/test_route_llm_request.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e8570-4259-7a60-851a-00cc1ed2233d.md`
    - `.analysis/investigation/investigate-codex-019e8570-3cb3-75c3-ada6-8b7165e5e335.md`
    - `suggestion.md`
  - Acceptance-gap disposition:
    - D1-172 and D1-174 were incomplete for live acceptance because they stopped
      at offline/synthetic harness coverage and required
      `LITELLM_XAI_OAUTH_AUTH_FILE` to already exist. This item supplied the
      missing real credential migration, dev-runtime wiring, fresh live smokes,
      and exact `aawm_tristore.public.session_history` proof.
    - Future xAI OAuth closure must run a live `/v1/chat/completions` smoke for
      every enabled `oa_xai/*` chat model using only a LiteLLM client key and
      must record the matching `session_history` rows or a concrete upstream
      xAI rejection.
  - Implementation notes:
    - Added `migrate_hermes_xai_oauth_credential` and
      `scripts/migrate_xai_oauth_credential.py` to copy
      `providers.xai-oauth.tokens` or fallback `credential_pool.xai-oauth`
      records from `/home/zepfu/.hermes/auth.json` into
      `/home/zepfu/.litellm/xai/oauth-auth.json` without printing token values.
    - The credential writer now creates parent directories, atomically replaces
      the managed credential file, and applies `0600` permissions to new or
      refreshed token files.
    - Updated `docker-compose.dev.yml` so `litellm-dev` mounts
      `/home/zepfu/.litellm/xai` writable and defaults
      `LITELLM_XAI_OAUTH_AUTH_FILE` to
      `/home/zepfu/.litellm/xai/oauth-auth.json`; Hermes remains only the
      migration source.
    - Updated `route_request` so prepared `oa_xai/*` calls bypass `llm_router`
      after LiteLLM rewrites the public model to `xai/*` and injects managed
      OAuth auth/base/provider data. This fixes the live router rejection where
      `xai/grok-4.3` was treated as an unconfigured healthy deployment.
    - Added tests for both real Hermes shapes, migration target validation,
      `0600` output permissions, refresh serialization, direct provider routing,
      and the existing offline `oa_xai/*` harness behavior.
  - Live credential and runtime evidence:
    - Redacted credential inspection of
      `/home/zepfu/.litellm/xai/oauth-auth.json` showed the scoped key
      `https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828`, expected
      record keys, token fields present, `source=hermes.providers.xai-oauth`,
      and `expires_at=2026-06-02T03:24:28.250768Z` without printing token
      values.
    - `stat -c '%n %a %U %G %s %y' /home/zepfu/.litellm/xai/oauth-auth.json`
      returned mode `600` after `docker exec litellm-dev sh -c 'chmod 600
      /home/zepfu/.litellm/xai/oauth-auth.json'`.
    - `docker exec litellm-dev printenv LITELLM_XAI_OAUTH_AUTH_FILE` returned
      `/home/zepfu/.litellm/xai/oauth-auth.json`.
    - `docker inspect litellm-dev --format ...` showed
      `/home/zepfu/.litellm/xai -> /home/zepfu/.litellm/xai rw= true`, and
      `docker exec litellm-dev sh -c 'test -w /home/zepfu/.litellm/xai'`
      passed.
    - `docker exec litellm-dev sh -c 'test
      /home/zepfu/.litellm/xai/oauth-auth.json -nt /home/zepfu/.hermes/auth.json'`
      passed, proving the runtime credential file is the refreshed
      LiteLLM-owned copy rather than the Hermes source.
  - Live `/v1/chat/completions` evidence using only
    `Authorization: Bearer sk-1234`:
    - `oa_xai/grok-4.3` with session
      `d1-176-verify-grok43-20260601T1910` returned HTTP 200 and exact assistant
      content `oa xai live smoke`.
    - `oa_xai/grok-4.20-0309-non-reasoning` with session
      `d1-176-verify-nonreasoning-20260601T1910` returned HTTP 200 and exact
      assistant content `oa xai live smoke`.
    - `oa_xai/grok-4.20-0309-reasoning` with session
      `d1-176-verify-reasoning-20260601T1910` returned HTTP 200 and exact
      assistant content `oa xai live smoke`.
    - `oa_xai/grok-4.20-multi-agent-0309` with session
      `d1-176-verify-multiagent-20260601T1910` reached xAI and returned HTTP
      400 with `Multi Agent requests are not allowed on chat completions`.
      This is the D1-176E disposition: the chat-completions alias is externally
      rejected by xAI. Keep product follow-up deferred unless operator intent
      requires routing this alias through a different endpoint or disabling it.
  - Exact `aawm_tristore.public.session_history` evidence:
    - Verified target database with `select current_database(),
      inet_server_addr(), inet_server_port();`, which returned
      `aawm_tristore|172.19.0.6|5432`.
    - Querying the four fresh D1-176 session IDs returned exactly the three
      successful completion rows:
      - row `1078117`, trace `9c8960a7-83cd-467e-b41c-923e7a9228dd`,
        session `d1-176-verify-grok43-20260601T1910`, provider `xai`, model
        and model_group `oa_xai/grok-4.3`, call_type `acompletion`, tokens
        `138/5/286`, cost `0.00040810000000000005`, metadata
        `auth_mode=oauth`, `credential_family=xai_oauth`,
        `passthrough_route_family=xai_oauth_api`,
        `xai_oauth_public_model=oa_xai/grok-4.3`,
        `xai_oauth_upstream_model=xai/grok-4.3`,
        `shared_quota_family=xai_grok_subscription`.
      - row `1078133`, trace `5f501dee-a177-4ef0-bd01-322df86252ac`,
        session `d1-176-verify-nonreasoning-20260601T1910`, provider `xai`,
        model and model_group `oa_xai/grok-4.20-0309-non-reasoning`,
        call_type `acompletion`, tokens `130/5/135`, cost
        `0.00010780000000000002`, metadata `auth_mode=oauth`,
        `credential_family=xai_oauth`,
        `passthrough_route_family=xai_oauth_api`,
        `xai_oauth_public_model=oa_xai/grok-4.20-0309-non-reasoning`,
        `xai_oauth_upstream_model=xai/grok-4.20-0309-non-reasoning`,
        `shared_quota_family=xai_grok_subscription`.
      - row `1078135`, trace `ff2ac7a6-ebfa-415b-9886-ea08c70f2918`,
        session `d1-176-verify-reasoning-20260601T1910`, provider `xai`, model
        and model_group `oa_xai/grok-4.20-0309-reasoning`, call_type
        `acompletion`, tokens `132/5/535`, cost `0.0011053000000000002`,
        metadata `auth_mode=oauth`, `credential_family=xai_oauth`,
        `passthrough_route_family=xai_oauth_api`,
        `xai_oauth_public_model=oa_xai/grok-4.20-0309-reasoning`,
        `xai_oauth_upstream_model=xai/grok-4.20-0309-reasoning`,
        `shared_quota_family=xai_grok_subscription`.
    - The same exact-session query returned no `session_history` row for
      `d1-176-verify-multiagent-20260601T1910`, which matches the upstream
      HTTP 400 failure before a successful completion row.
  - Investigation/subagent disposition:
    - Dispositioned and archived
      `.analysis/investigate-codex-019e8570-4259-7a60-851a-00cc1ed2233d.md`
      as a context-only aawm-tap-dashboard session-level subagent failure.
    - Dispositioned and archived
      `.analysis/investigate-codex-019e8570-3cb3-75c3-ada6-8b7165e5e335.md`
      after the D1-176 read-only scout returned unrelated Router wildcard alias
      output and modified `litellm/router.py`; the parent thread removed that
      unrelated diff before closure.
  - Verification evidence:
    - `./.venv/bin/python -m py_compile litellm/llms/xai/oauth.py
      litellm/proxy/route_llm_request.py scripts/migrate_xai_oauth_credential.py
      tests/llm_translation/test_xai.py
      tests/test_litellm/proxy/test_oa_xai_harness.py
      tests/test_litellm/proxy/test_route_llm_request.py` passed.
    - `./.venv/bin/python -m pytest tests/llm_translation/test_xai.py
      tests/test_litellm/proxy/test_oa_xai_harness.py
      tests/test_litellm/proxy/test_route_llm_request.py -q -k 'xai_oauth or
      oa_xai'` passed: `23 passed, 1 skipped, 67 deselected, 1 warning`.
    - `./.venv/bin/ruff check litellm/llms/xai/oauth.py
      litellm/proxy/route_llm_request.py scripts/migrate_xai_oauth_credential.py
      tests/test_litellm/proxy/test_oa_xai_harness.py
      tests/test_litellm/proxy/test_route_llm_request.py` passed.
    - `./.venv/bin/ruff check ... tests/llm_translation/test_xai.py` was not a
      clean gate because that long existing test module reports pre-existing
      unused imports and a print-statement lint outside this D1-176 change.
    - `git diff --check -- docker-compose.dev.yml litellm/llms/xai/oauth.py
      litellm/proxy/route_llm_request.py scripts/migrate_xai_oauth_credential.py
      tests/llm_translation/test_xai.py
      tests/test_litellm/proxy/test_oa_xai_harness.py
      tests/test_litellm/proxy/test_route_llm_request.py suggestion.md
      .analysis/todo.md .analysis/completed.md .analysis/investigations.md
      .analysis/investigation/investigate-codex-019e8570-3cb3-75c3-ada6-8b7165e5e335.md
      .analysis/investigation/investigate-codex-019e8570-4259-7a60-851a-00cc1ed2233d.md`
      passed.

- D1-175 Codex auto-agent null/generic response prevention and investigation
  disposition
  - Goal: investigate the root `investigate-codex-*.md` intake as
    LiteLLM-owned Codex auto-agent path failures, correct the earlier
    wrong-parent disposition, determine whether scoring alone was insufficient,
    and add a scoped prompt/control-plane prevention hook where safe.
  - Initiated: 2026-06-01 16:26:57 EDT.
  - Completed: 2026-06-01 16:58:18 EDT.
  - Duration: 31 minutes 21 seconds.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e84c2-c452-7740-85c3-361dc81c8a2f.md`
    - `.analysis/investigation/investigate-codex-019e84c2-e7d0-7e93-903d-68dd4537b283.md`
    - `.analysis/investigation/investigate-codex-019e84c8-c7ee-7020-8172-2da728f5340d.md`
    - `.analysis/investigation/investigate-codex-019e84d4-04e5-7060-9000-e44d52c33714.md`
    - `.analysis/investigation/investigate-codex-019e84d5-ac54-74c1-b901-ef2033736be0.md`
    - `.analysis/investigation/investigate-codex-019e84d8-4484-7261-a521-2b9375ef1ef7.md`
    - `.analysis/investigation/investigate-codex-019e84da-fc12-72d1-807e-d5565673965f.md`
    - `.analysis/investigation/investigate-codex-019e84de-e586-7432-a25c-22edd28de196.md`
    - `.analysis/investigation/investigate-codex-019e84df-2b2c-7cd1-bfaa-971856b4ef36.md`
    - `.analysis/investigation/investigate-codex-019e84e2-7978-7f12-9cc3-1e21570bd5a0.md`
    - `.analysis/investigation/investigate-codex-019e84ea-8399-7203-86e6-a18daaa66f4a.md`
    - `.analysis/investigation/investigate-codex-019e84ea-4e59-77a2-8b9a-14ab609a47c4.md`
    - `.analysis/investigation/investigate-codex-019e84ec-7aa8-72e1-9f59-a41c581b838c.md`
    - `.analysis/investigation/investigate-codex-019e84ef-7969-7321-b457-02b9058c5481.md`
    - `.analysis/investigation/investigate-codex-019e84f2-19ad-7530-a75b-546c354341a2.md`
    - `.analysis/investigation/investigate-codex-019e8502-bce7-7fd0-803e-833f37775693.md`
    - `suggestion.md`
  - Implementation notes:
    - Added a scoped Codex auto-agent prevention guidance policy that appends
      an idempotent top-level `instructions` block only when the OpenAI
      Responses passthrough request resolves `model=aawm-codex-agent-auto`.
    - The guidance targets the observed failure classes directly: non-empty
      final answer, no internal-plan final output, exact tool/platform error
      before claiming tools are unavailable, no generic file/function
      explanations when changes were requested, and explicit verification
      blockers.
    - Added trace metadata, tags, and a Langfuse span for
      `codex-auto-agent-prevention-guidance` so later traffic can prove whether
      the prevention hook was applied.
    - Kept generic OpenAI/Codex Responses traffic unchanged and removed the
      overly broad Google Code Assist prompt edit produced by a read-only scout.
    - Dispositioned all root Codex investigation intake, including correcting
      the earlier wrong-parent classification for the dashboard subagent
      failures. Parent repo is now treated as failure context, not a dismissal
      reason.
  - Verification evidence:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestResponsesAdapterToolChoice::test_applies_codex_auto_agent_prevention_guidance_to_existing_instructions tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::test_openai_passthrough_codex_auto_agent_alias_uses_alias_router tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::test_openai_passthrough_codex_spark_does_not_get_prevention_guidance -q`
      passed: `3 passed, 71 warnings`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_auto_agent_alias or prevention_guidance'`
      passed: `18 passed, 342 deselected, 71 warnings`.
    - `git diff --check -- litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py .analysis/todo.md .analysis/completed.md .analysis/investigations.md suggestion.md`
      passed.
    - `./.venv/bin/ruff check litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      was not a clean gate because the existing long test module reports
      pre-existing unused imports, print statements, and complexity findings.
    - `./.venv/bin/black` initially reformatted large unrelated portions of
      the two long touched files; that formatting churn was removed before the
      final diff, and the final verification relies on py_compile, focused
      pytest, the Codex alias pytest slice, and `git diff --check`.

- D1-174 OAuth-managed xAI Grok model test harness
  - Goal: add test-harness coverage for the planned `oa_xai/*` OAuth-managed
    model path, including credential loading, refresh behavior, route mapping,
    quota semantics, and session-history/provider-error observability.
  - Initiated: 2026-06-01 15:54:15 EDT.
  - Completed: 2026-06-01 16:01:56 EDT.
  - Duration: 7 minutes 41 seconds.
  - Changed paths:
    - `tests/test_litellm/proxy/test_oa_xai_harness.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e84c2-c452-7740-85c3-361dc81c8a2f.md`
    - `.analysis/investigation/investigate-codex-019e84c2-e7d0-7e93-903d-68dd4537b283.md`
    - `suggestion.md`
  - Implementation notes:
    - Added `OaXaiHarness`, an offline pytest harness covering all current
      `oa_xai/*` public Grok models and their intended `xai/*` upstream route
      mappings.
    - The harness uses synthetic LiteLLM-owned credential fixtures only; it
      verifies scoped credential loading, near-expiry refresh, refresh-lock
      serialization, missing credential errors, terminal refresh errors, and
      credential persistence after refresh.
    - Route assertions prove clients only submit LiteLLM-facing request data
      while LiteLLM injects the upstream xAI OAuth bearer token, API base, and
      `custom_llm_provider=xai`.
    - Observability assertions validate session-history provider/model
      identity, token counts, non-empty cost, provider-error classification,
      `auth_mode=oauth`, `credential_family=xai_oauth`,
      `passthrough_route_family=xai_oauth_api`, and shared Grok subscription
      quota metadata.
    - Added an explicitly gated live smoke test behind
      `AAWM_OA_XAI_LIVE_SMOKE=1` and required live env vars. It is skipped by
      default and does not log or assert real OAuth token values.
    - Dispositioned unrelated root investigation intake
      `investigate-codex-019e84c2-c452-7740-85c3-361dc81c8a2f.md` and
      `investigate-codex-019e84c2-e7d0-7e93-903d-68dd4537b283.md`; both were
      subagent handoff failures for `aawm-tap-dashboard`, not LiteLLM defects.
  - Verification evidence:
    - `./.venv/bin/python -m py_compile tests/test_litellm/proxy/test_oa_xai_harness.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/test_oa_xai_harness.py -q`
      passed: `12 passed, 1 skipped, 1 warning`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/test_oa_xai_harness.py tests/llm_translation/test_xai.py tests/test_litellm/proxy/test_route_llm_request.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'oa_xai or xai_oauth'`
      passed: `21 passed, 1 skipped, 327 deselected, 1 warning`.
    - `./.venv/bin/ruff check tests/test_litellm/proxy/test_oa_xai_harness.py`
      passed.
    - `./.venv/bin/black tests/test_litellm/proxy/test_oa_xai_harness.py`
      reformatted the new harness file; the focused harness pytest was rerun
      afterward and still passed.
    - `git diff --check -- tests/test_litellm/proxy/test_oa_xai_harness.py`
      passed.

- D1-173 Grok Build pass-through adapter test harness
  - Goal: add a reusable deterministic harness for native Grok Build
    pass-through request shape, auth/header separation, protobuf/raw-body
    passthrough, streaming/final response handling, provider error capture,
    quota observation metadata, and session-history identity.
  - Initiated: 2026-06-01 15:42:21 EDT.
  - Completed: 2026-06-01 15:51:36 EDT.
  - Duration: 9 minutes 15 seconds.
  - Changed paths:
    - `tests/test_litellm/proxy/pass_through_endpoints/test_grok_build_passthrough_harness.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e84b5-b1bc-76f3-8912-00dd030ce379.md`
    - `suggestion.md`
  - Implementation notes:
    - Added `GrokBuildPassthroughHarness`, an offline pytest harness that
      builds representative Grok Build request headers/body, invokes
      `grok_proxy_route`, filters forwarded headers through the real pass-through
      helper, normalizes final responses, routes streaming response chunks
      through the OpenAI-compatible logging path, and builds AAWM
      session-history/rate-limit/provider-error records.
    - The harness asserts Grok/OIDC headers are forwarded while
      `x-litellm-api-key` is not forwarded, protobuf trace bodies stay raw, the
      public identity remains `provider=xai`, `model=grok-build`,
      `client_name=grok-build`, and
      `passthrough_route_family=grok_cli_chat_proxy`, and Grok billing quota
      observations use the `xai_grok_build_monthly_requests` request bucket.
    - Added an explicitly gated live smoke test behind
      `AAWM_GROK_BUILD_LIVE_SMOKE=1` and required live credential environment
      variables. It is skipped by default and does not record secrets.
    - Updated the existing xAI embedding passthrough test to match the current
      D1-107 contract: xAI embeddings normalize through the OpenAI-compatible
      handler, not Cohere or generic fallback.
    - Dispositioned subagent investigation intake
      `investigate-codex-019e84b5-b1bc-76f3-8912-00dd030ce379.md` after the
      scout returned unrelated generic output.
  - Verification evidence:
    - `./.venv/bin/python -m py_compile tests/test_litellm/proxy/pass_through_endpoints/test_grok_build_passthrough_harness.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_grok_build_passthrough_harness.py -q`
      passed: `5 passed, 1 skipped, 71 warnings`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_grok_build_passthrough_harness.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'grok or xai'`
      passed: `40 passed, 1 skipped, 656 deselected, 72 warnings`.
    - `git diff --check -- tests/test_litellm/proxy/pass_through_endpoints/test_grok_build_passthrough_harness.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py .analysis/todo.md .analysis/investigate-codex-019e84b5-b1bc-76f3-8912-00dd030ce379.md`
      passed.

- D1-172 xAI OAuth-managed Grok model routing through normal LiteLLM model calls
  - Goal: expose selected xAI Grok models through normal LiteLLM model calls
    using LiteLLM-managed OAuth credentials, so clients request `oa_xai/*`
    with only their LiteLLM key while LiteLLM loads, refreshes, and injects
    the upstream xAI OAuth bearer token.
  - Initiated: 2026-06-01 15:12:40 EDT.
  - Completed: 2026-06-01 15:36:03 EDT.
  - Changed paths:
    - `litellm/llms/xai/oauth.py`
    - `litellm/proxy/route_llm_request.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `tests/llm_translation/test_xai.py`
    - `tests/test_litellm/proxy/test_route_llm_request.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `/home/zepfu/projects/dashboard-shell/.analysis/handoff-xai-oauth-session-history-d1-172.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - Added server-side `oa_xai/*` preparation for
      `oa_xai/grok-4.3`, `oa_xai/grok-4.20-0309-reasoning`,
      `oa_xai/grok-4.20-0309-non-reasoning`, and
      `oa_xai/grok-4.20-multi-agent-0309`, mapping each public model to its
      matching upstream `xai/*` model on `https://api.x.ai/v1`.
    - `LITELLM_XAI_OAUTH_AUTH_FILE` is the LiteLLM-owned credential source;
      the code accepts Grok-style scoped records or flat credential records,
      refreshes expired or near-expired tokens, serializes refreshes per
      credential/scope, and returns reseed/relogin errors for missing or
      terminal credentials.
    - The route preparation injects `api_key`, `api_base`, and
      `custom_llm_provider=xai` before normal route dispatch while preserving
      native `/grok/{endpoint:path}` pass-through behavior.
    - Session-history and provider-error metadata now preserve
      `auth_mode=oauth`, `credential_family=xai_oauth`,
      `passthrough_route_family=xai_oauth_api`,
      `shared_quota_family=xai_grok_subscription`, and related public/upstream
      xAI OAuth model identifiers.
    - Added public and upstream model catalog entries to both model-price JSON
      files using current xAI docs for context and token pricing.
    - Created the required dashboard-shell handoff for the new metadata
      semantics and quota grouping guidance.
  - Verification evidence:
    - `./.venv/bin/python -m py_compile litellm/llms/xai/oauth.py litellm/proxy/route_llm_request.py litellm/integrations/aawm_agent_identity.py litellm/proxy_auth/__init__.py tests/test_litellm/proxy/test_route_llm_request.py tests/llm_translation/test_xai.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m py_compile .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/integrations/aawm_agent_identity.py`
      passed.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/test_route_llm_request.py tests/llm_translation/test_xai.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'oa_xai or xai_oauth'`
      passed: `9 passed, 327 deselected, 1 warning`.
    - `./.venv/bin/python -m json.tool model_prices_and_context_window.json >/tmp/model_prices_check.json`
      passed.
    - `./.venv/bin/python -m json.tool litellm/bundled_model_prices_and_context_window_fallback.json >/tmp/bundled_model_prices_check.json`
      passed.
    - `docker exec litellm-dev printenv LITELLM_XAI_OAUTH_AUTH_FILE` returned
      unset, so no live xAI OAuth smoke was run against `litellm-dev`.

- D1-170 session_history routed-provider normalization for auto-agent,
  OpenRouter, and local embed rows
  - Goal: store routed provider/model values in `public.session_history`
    instead of proxy/transport aliases for Codex auto-agent traffic,
    OpenRouter free/flash routes, and local embedding routes.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - Session-history provider normalization now treats `litellm` as a proxy
      alias, prefers Codex/AAWM auto-agent selected provider metadata, consults
      the LiteLLM model catalog, and detects OpenRouter/local embedding routes
      from model prefix, route family, and API base.
    - Auto-agent model-group aliases are normalized to the selected routed
      model when metadata proves the selected model.
    - Local embedding routes, including Nomic local embeddings, are stored as
      `provider=local_embed`, with `local_embed/` stripped from the persisted
      model and local-route metadata retained.
    - Langfuse ClickHouse reconstruction applies the same routed-provider
      normalization as live writes.
    - Transcript-derived Codex score rows no longer create a fake
      `provider=litellm` / `model=aawm-codex-agent-auto` bucket; synthetic rows
      preserve the raw aliases in metadata and use `model=codex-transcript`
      with no routed provider.
  - Backfill and data-repair evidence:
    - Ran the Langfuse ClickHouse backfill against exact DB `aawm_tristore`
      with `--source-mode langfuse_clickhouse --apply --batch-size 100`; result
      `scanned_rows=34931`, `reconstructable_rows=34569`,
      `inserted_rows=34569`, `skipped_rows=362`.
    - Targeted repair against exact DB `aawm_tristore.public.session_history`
      updated `openrouter_repaired=960`, `local_embed_repaired=493`, and
      `synthetic_transcript_repaired=9`.
    - Full-history verification against exact DB `aawm_tristore` returned:
      `bad_openai_openrouter_or_local=0`, `bad_litellm_auto=0`,
      `good_openrouter_repaired_deepseek=960`, `good_local_nomic=48988`, and
      `synthetic_codex_transcript_not_provider=9`.
  - Local verification evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -p no:rerunfailures`
      passed: `254 passed, 1 warning`.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_score_agent_trace_quality.py -q -p no:rerunfailures`
      passed: `46 passed, 1 warning`.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `git diff --check -- litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py .analysis/todo.md .analysis/completed.md`
      passed.

## 2026-05-28

- D1-165 Codex Responses stream Langfuse output fallback
  - Goal: make Codex `/openai_passthrough/responses` streaming observations log
    a reconstructed standard response instead of the raw
    `cannot parse chunks to standard response object` placeholder when the
    final `response.completed.response` payload omits `output`.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - Missing `output` is normalized to `[]` before strict
      `ResponsesAPIResponse` validation, preserving the existing empty-output
      fallback that reconstructs assistant text, streamed tool state, provider
      usage, and hidden `responses_output` metadata.
    - Added
      `test_openai_streaming_handler_rebuilds_codex_stream_with_missing_output`
      covering reconstructed assistant text, function-call state, provider
      usage, absence of `cannot parse chunks`, and streaming
      `codex.usage_normalize` span preservation.
  - Local verification evidence:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py -q`
      passed: `37 passed, 1 warning`.
    - `./.venv/bin/ruff check litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
      passed.
    - `./.venv/bin/ruff check --select F tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`
      passed.
  - Live verification evidence:
    - Restarted `litellm-dev`; `/health/liveliness` returned `"I'm alive!"`;
      container check showed the patched file was mounted in `/app`.
    - Live dev Codex command:
      `env CODEX_LANGFUSE_TRACE_USER_ID=litellm codex --profile litellm-dev -a never exec --ephemeral --sandbox read-only -C /home/zepfu/projects/litellm "Reply exactly: D1-165 live verification dev"`
      returned exactly `D1-165 live verification dev`.
    - `litellm-dev` logs after restart showed the new
      `POST /openai_passthrough/responses` returned HTTP 200 without the prior
      `output Field required` or `Failed to build complete response from
      OpenAI streaming chunks` errors.
    - Langfuse trace `3f9b2132-58d2-4655-8659-10fac0029819` has
      `environment=dev`, `name=codex`, `userId=litellm`,
      `route:codex_responses`, and generation output
      `D1-165 live verification dev`.
    - Langfuse observation query for that trace returned
      `cannot parse chunks=false`, `D1-165 live verification dev=true`, and
      `codex.usage_normalize` metadata
      `streaming=true`, `call_type=responses`, `total_tokens=24660`,
      `response_cost=0.109468`.
    - Exact DB `aawm_tristore.public.session_history` row
      `session_id=3f9b2132-58d2-4655-8659-10fac0029819` confirmed
      `litellm_environment=dev`, `tenant_id=litellm`,
      `repository=litellm`, `provider=openai`, `model=gpt-5.5`,
      `client_name=codex_exec`, `client_version=0.134.0`,
      `input_tokens=24574`, `output_tokens=86`, `total_tokens=24660`,
      `cache_read_input_tokens=3456`, `provider_cache_status=hit`,
      `provider_response_id=resp_044b2e0e593d764a016a18896396f88194a0edaccb6d4f0cc8`,
      and `metadata.passthrough_route_family=codex_responses`.

## 2026-05-27

- P-INV-003 read-only recommendation wording hardening
  - Goal: harden read-only subagent prompt guidance when asking for
    "minimal recommended patch", "suggested fixes", "implementation slices",
    or similar implementation-shaped output.
  - Changed paths:
    - `/home/zepfu/.codex/AGENTS.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - Added global `## Subagent Read Only Prompt Suggestion` guidance.
    - The suggested prompt now explicitly forbids editing, creating files,
      applying patches, or running worktree-mutating commands.
    - The suggested prompt tells agents to describe needed fixes only and
      requires the final answer to truthfully include
      `No files were modified.`
  - Evidence:
    - Verified `/home/zepfu/.codex/AGENTS.md` contains the new section with
      the agreed wording.
    - Removed `P-INV-003` from `Proposals (Pending Operator Feedback)` in
      `.analysis/todo.md`.

- D1-159 read-only prompt compliance scoring in Langfuse and session_history
  - Goal: extend deterministic agent trace-quality scoring so a session that
    was explicitly scoped read-only is scored for edit-policy compliance and
    the result is visible in `public.session_history`.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
  - Implementation notes:
    - Added Codex transcript scoring mode for local JSONL transcripts via
      `--codex-transcript` and `--codex-parent-transcript`, including parent
      `spawn_agent` prompt extraction.
    - Added transcript-derived `session_history` upsert support behind
      `--upsert-codex-transcript-session-history`, keyed by
      `litellm_call_id='codex-transcript:<session_id>'` when no native
      LiteLLM row exists.
    - Classified Codex `exec_command` as a shell-command tool for read-only
      policy scoring, so heredoc writes, Python write modes, `sed -i`,
      `rm`/`mv`/`cp`, and live DB/container command violations are detected
      when the prompt forbids them.
    - Preserved command timestamps, command snippets, affected paths, matched
      read-only instruction evidence, and violation reasons in score metadata.
  - Incident evidence:
    - Child transcript:
      `/home/zepfu/.codex/sessions/2026/05/26/rollout-2026-05-26T21-04-21-019e66f6-0ba8-7491-9e6a-d446cc1cab59.jsonl`.
    - Parent transcript:
      `/home/zepfu/.codex/sessions/2026/05/21/rollout-2026-05-21T22-46-26-019e4d93-b4d0-7fb0-9b04-0c57da041e0d.jsonl`.
    - Dry-run scoring of the incident transcript produced
      `read_only_policy_compliance_score=0.0`,
      `read_only_policy_violation_count=4`, and violation reason
      `mutating_tool:exec_command` with affected paths including
      `update_json.py`, `config/worker-supervisor.dev.json`,
      `config/worker-supervisor.reconcile-smoke.dev.json`,
      `tests/test_dev_docker_stack.py`, `update_tests.py`, and
      `update_tests_2.py`.
  - Data repair evidence:
    - Applied transcript-derived repair to exact DB
      `aawm_tristore.public.session_history` with
      `--update-session-history-scores` and
      `--upsert-codex-transcript-session-history`; result
      `session_history_update_count=1`.
    - Verification row:
      `id=925993`,
      `litellm_call_id=codex-transcript:019e66f6-0ba8-7491-9e6a-d446cc1cab59`,
      `session_id=trace_id=019e66f6-0ba8-7491-9e6a-d446cc1cab59`,
      `provider=litellm`, `model=aawm-codex-agent-auto`,
      `repository=aawm-tap`, `client_name=codex-tui`,
      `client_version=0.133.0`,
      `read_only_policy_compliance_score=0`,
      `read_only_policy_violation_count=4`,
      `instruction_adherence_score=0`,
      `destructive_action_policy_score=0`, and
      `metadata.session_history_repair_source=
      d1_159_codex_transcript_policy_scoring`.
  - Verification evidence:
    - `./.venv/bin/python -m pytest -p no:rerunfailures tests/test_scripts/test_score_agent_trace_quality.py -q`
      passed: `27 passed, 1 warning`.
    - `./.venv/bin/python -m py_compile scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py`
      passed.

- D1-162 additional LLM traffic content-quality scores in session_history
  - Goal: add follow-up LLM traffic scoring dimensions for answer
    completeness, evidence fidelity, tool-result fidelity, error attribution,
    repetition-loop risk, and context retention, with values written to both
    Langfuse scores and `public.session_history`.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes:
    - Added deterministic/null-aware scorer fields and Langfuse score emission
      for `aawm.agent.answer_completeness`,
      `aawm.agent.evidence_fidelity`,
      `aawm.agent.tool_result_fidelity`,
      `aawm.agent.error_attribution_quality`,
      `aawm.agent.repetition_loop_risk`, and
      `aawm.agent.context_retention`.
    - Added matching `session_history` columns, callback metadata extraction,
      payload insertion/upsert support, and wheel-build callback parity.
    - Kept payload/repository/token-accounting completeness out of this score
      family; those remain data-quality concerns, not LLM traffic scores.
  - Verification evidence:
    - `./.venv/bin/python -m pytest -p no:rerunfailures tests/test_scripts/test_score_agent_trace_quality.py -q`
      passed: `24 passed, 1 warning`.
    - `./.venv/bin/python -m pytest -p no:rerunfailures tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed: `233 passed, 1 warning`.
    - `./.venv/bin/python -m py_compile scripts/score_agent_trace_quality.py litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_scripts/test_score_agent_trace_quality.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `git diff --check -- scripts/score_agent_trace_quality.py litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_scripts/test_score_agent_trace_quality.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - Exact DB `aawm_tristore` now has all `23` score columns on
      `public.session_history`.
    - Representative passing row `924081`, trace
      `64c01202-67db-4e95-acc1-31e9b4d21b68`, persisted
      `answer_completeness_score=1`, `response_meaningfulness_score=1`,
      `tool_use_validity_score=1`, `tool_error_recovery_score=1`,
      `output_contract_compliance_score=1`, `task_progress_score=1`, and
      `scope_control_score=1`; Langfuse trace scores showed matching values.
    - Representative invalid-tool row `924779`, trace
      `eeb31079-59d5-4639-82e6-5f5b9263bbb8`, persisted
      `invalid_tool_call_error=true`, `tool_use_validity_score=0`,
      `error_attribution_quality_score=0`, `repetition_loop_risk_score=1`,
      and matching `agent_score_reasons`; Langfuse trace scores showed matching
      values.

- D1-161 LLM traffic behavior scores in Langfuse and session_history
  - Goal: add deterministic LLM traffic behavior scores for no-op responses,
    prompt-relative instruction adherence, tool validity/recovery, stall risk,
    output-contract compliance, task progress, scope control, and destructive
    action policy.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes:
    - Added `aawm.agent.response_meaningfulness`,
      `aawm.agent.instruction_adherence`, `aawm.agent.tool_use_validity`,
      `aawm.agent.tool_error_recovery`, `aawm.agent.stall_risk`,
      `aawm.agent.output_contract_compliance`,
      `aawm.agent.task_progress`, `aawm.agent.scope_control`, and
      `aawm.agent.destructive_action_policy`.
    - `tool_error_recovery` now emits a passing score for clean
      payload-resolved traces instead of being absent unless an invalid tool
      call occurred.
    - Tightened read-only phrase detection to reduce false positives from
      generic memory/instruction text while still catching broad
      `read-only ...` and `do not edit files` instructions.
  - Verification evidence:
    - Same focused scorer/callback test, py_compile, `cmp`, and
      `git diff --check` evidence as D1-162.
    - Exact DB `aawm_tristore.public.session_history` column verification
      returned all `23` score columns.
    - Row `924081` and Langfuse trace
      `64c01202-67db-4e95-acc1-31e9b4d21b68` showed passing values for the
      D1-161 score family.
    - Row `924779` and Langfuse trace
      `eeb31079-59d5-4639-82e6-5f5b9263bbb8` showed failing/flagged values for
      invalid tool use, error attribution, and repetition-loop risk.

- D1-160 persist deterministic trace-quality scores in session_history
  - Goal: persist deterministic trace-quality score outputs in
    `public.session_history` in addition to Langfuse score objects.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes:
    - Added score-column schema/update support for
      `trace_quality_score`, `empty_completion_failure`,
      `large_tool_result_payload_risk`,
      `destructive_checkout_after_work`, `invalid_tool_call_error`, and
      compact `agent_score_reasons`.
    - Missing-payload handling keeps unresolved boolean/score fields nullable
      instead of silently writing false, while preserving the independently
      stored `invalid_tool_call_count` signal.
    - The callback record builder normalizes score fields from metadata and
      preserves score metadata on insert/upsert.
  - Verification evidence:
    - Same focused scorer/callback test, py_compile, `cmp`, and
      `git diff --check` evidence as D1-162.
    - Exact DB `aawm_tristore` column count for score fields returned `23`.
    - Row `924081` persisted `trace_quality_score=1`,
      `empty_completion_failure=false`,
      `large_tool_result_payload_risk=false`,
      `destructive_checkout_after_work=false`, and
      `invalid_tool_call_error=false`; Langfuse trace
      `64c01202-67db-4e95-acc1-31e9b4d21b68` showed matching score values.
    - Row `924779` persisted `invalid_tool_call_error=true` and matching
      reasons; Langfuse trace `eeb31079-59d5-4639-82e6-5f5b9263bbb8` showed
      matching score values.

## 2026-05-26

- D1-158 session_history prod attribution and zero-token repair
  - Goal: resolve the bounded prod rows found after the 2026-05-26
    connection-storm review: Gemini Codex auto-agent rows whose repository was
    dropped after a malformed rollout/cwd fragment, root Codex
    memory-workspace rows with no project repository, and remaining prod
    zero-token rows that were non-usage or provider-failure observations rather
    than billable completions.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes:
    - Tightened `cwd=` text parsing so `cwd=/home/zepfu/projects/aawm-tap,
      rollout_path=...` extracts only the project path.
    - Added a deliberate root memory-workspace identity:
      `codex-memories (memory)` for `/home/zepfu/.codex/memories` memory
      writer traffic.
    - Preserved `metadata.source_status` in live session-history metadata and
      marks structured-output failure rows as failure before normalization.
    - Changed zero-token classification to ignore
      `reasoning_tokens_estimated`; estimated reasoning alone no longer masks
      a zero prompt/completion/cache usage response.
  - Data repair evidence:
    - In exact DB `aawm_tristore`, updated 96 prod Gemini rows on lane
      `4a044f7f3bb6:vital-involution-bbzck`, session
      `dd219a0a-d70a-57d6-9c81-5215a44d8645`, IDs `878720` through `879268`,
      to `repository=tenant_id=aawm-tap` with
      `metadata.repository_identity_repair_source=
      d1_158_codex_auto_agent_lane_context`.
    - Updated root memory-workspace prod rows `874672`, `874684`, and the
      later-arriving `905874` to
      `repository=tenant_id=codex-memories (memory)`,
      `metadata.source_repository=codex-memories`,
      `metadata.workload_type=agent_memory`, and
      `metadata.repository_identity_repair_source=
      d1_158_root_codex_memory_workspace`.
    - Classified zero-token row `867858` as
      `empty_provider_response_no_usage` with reason
      `gemini_code_assist_adapter_empty_response`.
    - Classified zero-token row `883102` as
      `failed_observation_no_usage` with reason
      `langfuse_observation_failed_without_usage` and
      `metadata.source_status=failure`.
    - Verification over `id >= 850000` and
      `created_at >= '2026-05-26T12:20:00Z'` returned prod
      `missing_repository=0` and prod `unclassified_zero_token=0`.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      passed.
    - Focused shard
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "root_codex_memory_workspace or rollout_fragment or estimated_reasoning_only or structured_output_attempt_for_unrelated_error or normalize_repository_identity_accepts_repo_shapes"`
      passed: `9 passed, 223 deselected`.
    - Full callback unit file
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed: `232 passed`.
    - `git diff --check -- litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py .analysis/todo.md`
      passed.
  - Dev/prod deployment evidence:
    - Restarted `litellm-dev`; `/health/readiness` on `:4001` returned
      healthy with `litellm_version=1.82.3+aawm.60` and `AawmAgentIdentity`
      active.
    - In-container dev callback import returned
      `codex-memories`, normalized `/home/zepfu/.codex/memories` to
      `codex-memories`, and rejected a nested rollout-summary path as `None`.
    - Implementation commit `416e719aa1 fix(aawm): repair session history
      attribution edge cases` was pushed to `origin/main`; artifact bump
      workflow `26482548803` produced release commit `4314b4e07a` and tag
      `cb-v0.0.40`.
    - Published GitHub release `cb-v0.0.40` with
      `aawm_litellm_callbacks-0.0.40-py3-none-any.whl`; asset digest
      `sha256:d0f265a9cc31bf841b2db67a4669a7cef1c5adf383fab1a04bbfb6a1bb879c08`.
    - Moved `cb-latest` to `cb-v0.0.40`.
    - Rebuilt prod image from `/home/zepfu/projects/aawm-infrastructure` with
      `--no-cache`; Docker build installed
      `aawm-litellm-callbacks-0.0.40` and produced local image
      `aawm-litellm:latest` / `6291eb8811a9`.
    - Recreated prod container `aawm-litellm`; `docker ps` showed it
      `Up ... (healthy)` on `127.0.0.1:4000`.
    - `/health/readiness` on `:4000` returned healthy with
      `litellm_version=1.82.3+aawm.62` and `AawmAgentIdentity` active.
    - Running prod callback check returned callback wheel `0.0.40`,
      `codex-memories`, normalized `/home/zepfu/.codex/memories` to
      `codex-memories`, and rejected a nested rollout-summary path as `None`.
    - Filtered prod logs since restart found no `AawmAgentIdentity.*failed`,
      `Traceback`, `maximum recursion`, `ERROR`, or `CRITICAL` matches.
    - Exact DB `aawm_tristore` sanity query over the last two hours returned
      `dev|0|0|4949` and `prod|0|0|1377` for
      `environment|unclassified_zero_token|missing_repo|total`.

- D1-157 Codex auto-review numeric repository attribution
  - Goal: stop Codex auto-review rows from persisting placeholder numeric
    identity values such as `0` into `tenant_id`, `repository`, and
    `metadata.trace_user_id`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes:
    - Added scalar numeric placeholder rejection to repository normalization.
    - Added tenant normalization so placeholder numeric tenant ids do not pass
      through metadata, headers, or repository-derived fallback.
    - Cleans numeric placeholder Codex `trace_user_id` values in hook metadata,
      Langfuse header metadata, and final `session_history` metadata sync.
    - Left an explicit numeric-identity allowlist hook empty so a real numeric
      repo/tenant can be enabled deliberately if one ever exists.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - Focused shard
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "numeric_identity_placeholders or numeric_header_tenant or normalize_repository_identity_rejects_metadata_noise or repository_as_tenant_fallback"`
      passed: `31 passed, 197 deselected`.
    - Full callback unit file
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed: `228 passed`.
    - `git diff --check` passed.
    - Source callback and wheel-build callback mirror matched by `cmp -s`.
  - Data repair evidence:
    - Existing affected prod rows were `832311`, `832327`, `832334`, and
      `832796`: `provider=openai`, `model=codex-auto-review`,
      `client_name=codex-tui`, `client_version=0.133.0`,
      `litellm_environment=prod`, with `tenant_id=0`, `repository=0`,
      `metadata.repository=0`, `metadata.trace_user_id=0`, and
      `metadata.tenant_id_source=repository`.
    - Updated those four exact rows in `aawm_tristore.public.session_history`
      to set `tenant_id` and `repository` to null, remove placeholder
      metadata keys, and preserve previous values under
      `metadata.numeric_identity_previous_*` with
      `metadata.numeric_identity_repair_run_id=
      d1-157-numeric-identity-placeholder-2026-05-26`.
    - Recent id-window verification after repair returned `0` rows where
      `tenant_id=0`, `repository=0`, or `metadata.trace_user_id=0`.
  - Dev validation:
    - Restarted `litellm-dev`; `/health/readiness` on `:4001` returned
      healthy.
    - In-container callback check returned `zero_repo None`,
      `numeric_repo None`, `real_repo aawm-tap`, `zero_tenant None`, and
      `generic_codex_zero True`.
  - Release/prod validation:
    - Committed and pushed `6598f035cc fix(aawm): reject numeric identity
      placeholders`; auto-bump produced `21c8ee8674` and tag `cb-v0.0.39`.
    - Manually published GitHub release `cb-v0.0.39` with
      `aawm_litellm_callbacks-0.0.39-py3-none-any.whl`; asset digest
      `sha256:a86445cade62fce9972bc867fe75223d3b049fcfcc24b7601b61e53b7b16be17`.
    - Moved `cb-latest` to `cb-v0.0.39`.
    - Rebuilt prod image from `/home/zepfu/projects/aawm-infrastructure`;
      Docker build installed `aawm-litellm-callbacks-0.0.39`.
    - Recreated `aawm-litellm`; new image digest
      `sha256:7dd556c68bf6fdc6907a5eed676b815427387cd3733024bcc35851e860ed2c55`,
      start time `2026-05-26T03:36:39.770588598Z`.
    - `/health/readiness` on `:4000` returned healthy with
      `litellm_version=1.82.3+aawm.62` and `AawmAgentIdentity` active.
    - Running prod callback check returned `callbacks 0.0.39`,
      `zero_repo None`, `numeric_repo None`, `real_repo aawm-tap`,
      `zero_tenant None`, and `generic_codex_zero True`.
    - Filtered prod logs since restart found no `AawmAgentIdentity.*failed`,
      `Traceback`, `maximum recursion`, or `ERROR` matches.
    - At verification time there were no prod `session_history` rows yet after
      the `cb-v0.0.39` restart, so no fresh live traffic row existed to inspect.

- D1-156 cb-v0.0.38 prod LiteLLM cutover
  - Goal: move the callback recursion-loss fix and OpenRouter
    `session_history` repair support from the released artifact into the
    production LiteLLM instance on port `4000`.
  - Runtime changed:
    - Rebuilt `/home/zepfu/projects/aawm-infrastructure`
      `docker-compose.litellm.yml` service `litellm`.
    - Recreated production container `aawm-litellm`.
    - New image digest:
      `sha256:69452a3890dd614dca0e1d332b7e60ef30391493e886b8765bfb4dcef801546c`.
  - Release artifact evidence:
    - GitHub release `cb-v0.0.38` contains
      `aawm_litellm_callbacks-0.0.38-py3-none-any.whl`.
    - Release asset digest:
      `sha256:91cf261a0235d29bf9aa9f4795986ac1f8bdb2786f4c9fa3d9a056b325a4ff50`.
    - Docker build logs showed
      `Successfully installed aawm-litellm-callbacks-0.0.38`.
  - Prod verification:
    - Container start time:
      `2026-05-26T02:43:31.395800716Z`.
    - `/health/readiness` returned `status=healthy`,
      `litellm_version=1.82.3+aawm.62`, success callbacks including
      `AawmAgentIdentity`, `use_aiohttp_transport=true`, and
      `log_level=WARNING`.
    - Running-container source check returned `litellm 1.82.3+aawm.62`,
      `callbacks 0.0.38`, `control_plane 0.0.7`,
      `request_payload_contains True`, and `json_safe_max_depth True`.
    - `aawm_tristore.public.session_history` rows since the new container
      start: `274` total, `0` missing model, `0` rows with both token fields
      null, `0` rows with both token totals zero, latest created at
      `2026-05-26 02:47:03.478499+00`.
    - Representative post-restart rows included OpenAI, Anthropic, and
      OpenRouter traffic with repository, model, token, and session-id fields
      populated.
    - Logs since container start showed no `maximum recursion` or
      `AawmAgentIdentity.*failed` callback-recursion failure signatures.
  - Residual observations:
    - Startup logs still contain the pre-existing
      `LITELLM_MASTER_KEY is not set` critical warning.
    - Current traffic still produces Langfuse item-size warnings for very large
      generations; the new size diagnostic warning includes largest metadata
      keys and trace/model context.

- D1-155 OpenRouter backfilled repository attribution tail
  - Goal: resolve the remaining OpenRouter `session_history` rows with
    blank/suspect repository values where deterministic evidence or operator
    policy was available.
  - Changed data:
    - Exact DB: `aawm_tristore.public.session_history`
    - ClickHouse/MinIO evidence was read-only.
  - Repair evidence:
    - Initial D1-154 closeout count was `452` OpenRouter rows since
      `2026-05-20` with blank/suspect repository values; all had model and
      token accounting.
    - Repaired `423` rows to `aawm-tap` from retained ClickHouse trace tags
      with `metadata.openrouter_repository_repair_run_id=
      openrouter_repository_trace_tags_2026_05_26` and
      `metadata.repository_source=clickhouse_trace_tags`.
    - Repaired `21` rows to `aawm-tap` from retained MinIO trace payload tags
      with the same repair run id and
      `metadata.repository_source=minio_trace_payload_tags`.
    - Repaired `6` OpenRouter free-meter smoke/harness rows to `litellm` with
      the same repair run id and
      `metadata.repository_source=openrouter_free_meter_smoke_harness`.
    - User clarified that all Ling models are associated to `aawm-tap`.
      Applied that operator policy to all remaining blank-repo Ling rows,
      updating `282` rows with
      `metadata.openrouter_repository_repair_run_id=
      openrouter_ling_operator_policy_2026_05_26` and
      `metadata.repository_source=operator_ling_model_policy`.
  - Verification:
    - Post-repair Ling blank/suspect repository query returned no rows for
      `model ILIKE '%ling-2.6-flash%'`.
    - Operator-policy repair summary returned `282` rows, id range
      `49843..715808`, time range
      `2026-04-19 01:52:12.703+00` through
      `2026-05-22 14:27:51.27+00`.
    - Post-repair Ling repository distribution returned
      `aawm-tap=43972` and `litellm=37`.
    - Current OpenRouter quality scan since `2026-05-20` returned
      `0|0|0|0|92343`: missing model, both-token-null, both-token-zero,
      suspect repository, total rows.

- D1-154 OpenRouter callback recursion loss repair
  - Goal: prevent recursive/pathological OpenRouter request metadata from
    breaking `AawmAgentIdentity` success logging and dropping otherwise
    successful `session_history` rows.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `scripts/backfill_session_history.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - Added bounded, cycle-safe request payload traversal for cache-control and
      cached-content detection instead of recursive descent over arbitrary
      request bodies.
    - Added cycle/depth guards to JSON-safe metadata sanitization and prompt
      overhead walkers used by OpenAI/OpenRouter request logging.
    - Sanitized session-history metadata values before persistence so recursive
      request payloads cannot raise while building records.
    - Mirrored the callback source into the wheel-build callback artifact.
    - Added provider/model pushdown parameters to the Langfuse ClickHouse
      backfill source so future filtered backfills avoid reconstructing every
      retained generation before filtering.
  - Backfill evidence:
    - Targeted Ling/OpenRouter ClickHouse source rows:
      `3080` retained generations for
      `openrouter/inclusionai/ling-2.6-flash`.
    - Manual targeted backfill run
      `d1-154-openrouter-ling-clickhouse-20260526-v2` built `423` missing
      records, found `2657` existing trace rows, inserted/backdated `423`
      rows, and restored `270866` input tokens plus `223960` output tokens.
    - Backfilled rows were marked with
      `metadata.backfill_run_id=d1-154-openrouter-ling-clickhouse-20260526-v2`
      and `created_at=start_time`.
    - Final Ling anti-join verification returned
      `3080 3080 0 0 0`: retained ClickHouse rows, matching
      `session_history` rows, missing rows, missing input tokens, missing
      output tokens.
  - Current-data verification:
    - OpenRouter `session_history` quality scan since `2026-05-20` returned
      `0` missing models, `0` rows with both token fields null, `0` rows with
      both token totals zero, and `89623` total rows.
    - The same scan found `452` OpenRouter rows with blank/suspect repository
      values; all are now tracked separately as `D1-155` because their model
      and token accounting is present.
    - Retained ClickHouse OpenRouter rows since `2026-05-25` were checked
      against `session_history`. A trace-id anti-join reported one
      `openrouter/owl-alpha` miss, but direct inspection showed it was present
      as row `788500` with client session id
      `d1-064-openrouter-generic-chat-20260525-001`, model
      `openrouter/owl-alpha`, and tokens `98/2`; that row legitimately does
      not use the ClickHouse trace id as `session_id`.
  - Dev validation:
    - Restarted `litellm-dev` after the callback fix; `/health/readiness`
      returned healthy on port `4001`.
    - Container source inspection showed `_request_payload_contains` and
      `_AAWM_JSON_SAFE_MAX_DEPTH` present in
      `/app/litellm/integrations/aawm_agent_identity.py`.
    - Filtered post-restart logs since `2026-05-26T00:40:53Z` found no
      `maximum recursion` or `AawmAgentIdentity.*failed` callback recursion
      messages. Logs did show unrelated OpenRouter upstream `429` responses and
      Langfuse size-limit warnings.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/backfill_session_history.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "recursive_openrouter_request_body or openrouter_provider_cache_miss_from_cache_control_request or recursive_passthrough_payload"`
      passed: `4 passed, 215 deselected`.
    - Earlier focused callback shard passed:
      `130 passed, 89 deselected` for
      `session_history_record or provider_cache or prompt_overhead or recursive`.
    - `git diff --check` passed.

## 2026-05-25

- D1 aawm.62 prod infrastructure pre-restart prep
  - Goal: prepare the production infrastructure image for the `aawm.62`
    LiteLLM release while stopping before any prod container restart.
  - Changed paths:
    - `PATCHES.md`
    - `pyproject.toml`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/integrations/langfuse/langfuse.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `scripts/repair_session_history_repository_identity.py`
    - related focused tests
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - LiteLLM commit `78cb0bdc02` added the DeepSeek chat/completions
      auto-agent adapter path, observability repairs, and fork version
      `1.82.3+aawm.62`.
    - Artifact autobump commit `ec76f5d9df` bumped callback packaging to
      `0.0.37`; `origin/main` and `origin/develop` converged there.
    - Callback release `cb-v0.0.37` was manually published with asset
      `aawm_litellm_callbacks-0.0.37-py3-none-any.whl` after the documented
      recursive workflow-trigger gap left the release missing.
    - Fork tag `v1.82.3-aawm.62` was pushed from `ec76f5d9df`; GitHub Actions
      run `26410163646` succeeded and published
      `ghcr.io/zepfu/litellm:1.82.3-aawm.62`.
    - Infra commit `20edba9 chore(litellm): prep aawm62 image` is pushed to
      `origin/main` and `origin/develop`, pinning both infra image references
      to `ghcr.io/zepfu/litellm:1.82.3-aawm.62`.
    - Pre-restart prod image build completed with local image
      `aawm-litellm:latest` id `0d4524ae130b`.
    - Built-image inspection returned `litellm=1.82.3+aawm.62`,
      `aawm-litellm-callbacks=0.0.37`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Built-image source inspection returned `True` for
      `codex_openrouter_completion_adapter`,
      `anthropic_openrouter_completion_adapter`, and
      `_perform_codex_auto_agent_openrouter_completion_request`.
    - Running prod container `aawm-litellm` still reported
      `StartedAt=2026-05-24T22:48:27.194132158Z` and image
      `sha256:19bec94905dd86492e269efbf328627d29fe9b7465bcbe2c304fd5e06a56c7c9`
      after the build, confirming the release stopped before restart.
  - Verification:
    - `./.venv/bin/python -m py_compile ...` passed for the touched
      passthrough, callback, Langfuse, and repair modules.
    - `git diff --check` passed before the LiteLLM release commit.
    - Focused pytest shard passed: `199 passed, 462 deselected`.
    - `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      completed in `/home/zepfu/projects/aawm-infrastructure`.
    - `docker run --rm --entrypoint python3 aawm-litellm:latest -c "import importlib.metadata as m; print(m.version('litellm')); print(m.version('aawm-litellm-callbacks')); print(m.version('aawm-litellm-control-plane'))"`
      returned `1.82.3+aawm.62`, `0.0.37`, and `0.0.7`.

- D1-153 DeepSeek free auto-agent chat-shape routing
  - Goal: make `deepseek/deepseek-v4-flash:free` usable in the Codex and
    Anthropic auto-agent aliases where possible, instead of only treating
    empty/one-token OpenRouter Responses payloads as rollover signals.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - Changed the Codex auto-agent DeepSeek candidate route family from
      `codex_openrouter_responses` to
      `codex_openrouter_completion_adapter`.
    - Changed the Anthropic auto-agent DeepSeek candidate route family from
      `anthropic_openrouter_responses_adapter` to
      `anthropic_openrouter_completion_adapter`.
    - Removed DeepSeek free from the Anthropic OpenRouter Responses adapter
      allowlist and added it to the OpenRouter completion adapter allowlist, so
      direct `/anthropic/v1/messages` traffic for
      `openrouter/deepseek/deepseek-v4-flash:free` also uses the chat shape.
    - Added `_perform_codex_auto_agent_openrouter_completion_request()`, which
      transforms OpenAI Responses input to chat/completions with
      `LiteLLMCompletionResponsesConfig`, calls OpenRouter through
      `litellm.acompletion`, and transforms the chat response back to
      Responses output. The existing empty-success detector still raises
      `aawm_codex_auto_agent_empty_success` if the chat-shaped response
      transforms into no meaningful output.
    - Streaming Codex DeepSeek responses are wrapped through
      `LiteLLMCompletionStreamingIterator` before being serialized as Responses
      SSE events.
  - Evidence:
    - Read-only agent review confirmed the likely root cause was the
      OpenRouter `/v1/responses` request shape for DeepSeek free. The same
      review recommended the chat-completions bridge now implemented for Codex
      and the completion adapter path now used for Anthropic.
    - Historical `session_history` evidence showed `960` DeepSeek free
      auto-agent rows from `2026-05-23` through `2026-05-24`, all with
      `output_tokens <= 1`, while using the old OpenRouter Responses route.
    - A live OpenRouter chat-completions probe on `2026-05-25` used the new
      target shape but returned upstream `429` from provider `Crucible`; this
      is now handled by the auto-agent cooldown/rollover path rather than by
      accepting a one-token no-op as success.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k "openrouter_completion_adapter_model_supports_chat_shape_models or adapts_deepseek_free_to_openrouter_completion_adapter or routes_deepseek_through_openrouter_completion_adapter or routes_openrouter_candidate or openrouter_empty_success_rolls_to_last_resort or openrouter_one_token_text_is_success or fresh_dispatch_affinity_429_reaches_last_resort"`
      passed: `9 passed, 348 deselected`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k "auto_agent_alias or openrouter_completion_adapter or openrouter_one_token_text_is_success or openrouter_empty_success_rolls_to_last_resort or selected_openrouter_models_to_responses"`
      passed: `43 passed, 314 deselected`.

- D1-152 Gemini auto-agent null-repository raw trace recovery audit
  - Goal: determine whether the `109` `gemini-3-flash-preview`
    auto-agent rows with `repository is null` can be attributed to a real
    repository from deeper raw trace/transcript evidence, then repair only rows
    with deterministic per-row evidence.
  - Changed paths:
    - `.analysis/d1-152-minio-payloads/README.md`
    - `.analysis/d1-152-minio-payloads/manifest.json`
    - `.analysis/d1-152-minio-payloads/session_history_*.json`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Read-only agent audit classified the exact `109` target rows:
      ids `786716` through `792572`, time range
      `2026-05-24 23:56:31.331181+00` to
      `2026-05-25 02:43:42.400156+00`, totals `432965` input tokens and
      `5920` output tokens.
    - Same-session evidence remained mixed in the target window:
      `repository is null=109`, `aawm-tap=132`, and `litellm=37`, so
      session-level inference is not deterministic.
    - ClickHouse evidence found `94` target trace IDs in `traces`, `846`
      `observations` rows across `94` target trace IDs, and `109`
      `blob_storage_file_log` trace blob entries for all `109` target trace
      IDs.
    - MinIO fetch evidence found `109 / 109` raw trace blobs present and
      `0` missing objects.
    - Structured attribution scan found `0 / 109` payloads with deterministic
      fields such as `cwd`, `repository`, `repo`, `source_repository`,
      `project_root`, `workspace_root`, or `workspace`.
    - Content-only mentions were weak and not used as repair evidence:
      `litellm` appeared in rows `786716`, `786733`, `786747`, and `786762`;
      `aawm-tap` appeared in row `791950`; `104` rows had no repo evidence.
    - Representative raw MinIO payloads were exported to
      `.analysis/d1-152-minio-payloads/`, including `manifest.json`, the five
      weak content-only mention candidates where present, and early comparison
      samples.
  - Verification:
    - Exact target-row query:
      `select id, trace_id, litellm_call_id, created_at, input_tokens, output_tokens from public.session_history where session_id='dd219a0a-d70a-57d6-9c81-5215a44d8645' and provider='gemini' and model='gemini-3-flash-preview' and repository is null and metadata->>'codex_auto_agent_alias'='aawm-codex-agent-auto' order by id;`
    - Exact blob lookup query:
      `select entity_id, entity_type, bucket_name, bucket_path from blob_storage_file_log where entity_type='trace' and entity_id in (<109 trace ids>) order by entity_id, created_at;`
    - Export command:
      `./.venv/bin/python /tmp/export_d1_152_minio_payloads.py`
      wrote `8` sample payloads plus `manifest.json` under
      `.analysis/d1-152-minio-payloads/`.
    - Representative payload listing showed `README.md`, `manifest.json`, and
      `8` `session_history_*.json` files.
    - No database rows were changed because the audit found `0` deterministic
      repair candidates.

- D1-151 Session-history null repository attribution classification and repair
  - Goal: classify and repair recent `session_history.repository is null` rows
    using deterministic row, session, transcript, or metadata evidence only,
    while leaving ambiguous/control traffic null.
  - Changed paths:
    - `scripts/repair_session_history_repository_identity.py`
    - `tests/test_scripts/test_repair_session_history_repository_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added `--null-repository-since` to the repair script for focused null
      repository scans. In this mode, session-level repair uses the new
      `_build_unique_session_repository_map()` helper, which only repairs from
      session evidence when the entire session has exactly one valid repository
      from `/home/zepfu/projects`.
    - Added grouped dry-run output for null scans:
      `repair_group <provider>|<model>=<count>` and
      `unresolved_group <provider>|<model>=<count>`, so future repair runs show
      exactly which buckets were repaired or intentionally left unresolved.
    - Tightened rollout-registry repairs so a rollout filename can fill a
      missing repository, but cannot override an already valid repository such
      as `aawm-tap (memory)`.
    - Focused dry run against `aawm_tristore.public.session_history` from
      `2026-05-22T00:00:00+00:00` returned `candidate_rows=15337`,
      `repairable_rows=11345`, and `applied=false`.
    - The applied run repaired `11345` deterministic rows:
      `row_identity_normalization=23`,
      `same_session.session_metadata.repository=11107`, and
      `same_session.session_metadata.tenant_id=215`.
    - Repaired groups were:
      `openrouter|openrouter/qwen/qwen3-embedding-8b=5635`,
      `openrouter|openrouter/inclusionai/ling-2.6-flash=4909`,
      `openai|local_embed/nomic-embed-code.Q8_0.gguf=493`,
      `openai|gpt-5.5=181`, `openai|gpt-5.4=115`,
      `openai|codex-auto-review=7`,
      `openrouter|openrouter/openai/gpt-oss-20b:free=4`, and
      `gemini|gemini-2.5-flash-lite=1`.
    - The 109 `gemini|gemini-3-flash-preview` auto-agent rows were left null
      because their session has valid evidence for both `aawm-tap` and
      `litellm`; session-level inference would be ambiguous. Verification
      showed that Gemini auto-agent session currently contains
      `aawm-tap|aawm-tap|1092`, `aawm-tap-dashboard|aawm-tap-dashboard|294`,
      `litellm|litellm|206`, and `null|null|109`.
    - Remaining unresolved rows after repair are not deterministically
      repairable from session evidence: `3550` `openai|gpt-5.5`, `132`
      `openai|gpt-5.4`, `109` mixed-session
      `gemini|gemini-3-flash-preview`, `108` `openai|gpt-5.4-mini`, `76`
      `openai|codex-auto-review`, `23`
      `openrouter|openrouter/inclusionai/ling-2.6-flash`, and smaller
      no-evidence/control buckets.
  - Verification:
    - `./.venv/bin/python -m py_compile scripts/repair_session_history_repository_identity.py tests/test_scripts/test_repair_session_history_repository_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_repair_session_history_repository_identity.py -q`
      passed: `11 passed, 1 warning`.
    - `./.venv/bin/python scripts/repair_session_history_repository_identity.py --dsn postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore --null-repository-since 2026-05-22T00:00:00+00:00 --batch-size 1000 --preview-limit 10 --apply`
      returned `database=aawm_tristore`, `candidate_rows=15337`,
      `repairable_rows=11345`, and `applied=true`.
    - Follow-up dry run with the same `--null-repository-since` arguments
      returned `candidate_rows=4015`, `repairable_rows=0`, and `applied=false`.
    - `psql -Atqc "select repository, tenant_id, count(*) from public.session_history where id between 688282 and 739403 and provider='openrouter' and model in ('openrouter/qwen/qwen3-embedding-8b','openrouter/inclusionai/ling-2.6-flash') group by repository, tenant_id order by count(*) desc limit 20;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned `aawm-tap|aawm-tap|11458` and `null|null|23`, showing the
      deterministic OpenRouter attribution was repaired while the remaining
      no-evidence ling rows stayed null.
    - Classification query over the remaining null rows returned
      `gemini|gemini-3-flash-preview|109|1|0|109|aawm-tap,litellm`, confirming
      the unresolved Gemini rows are mixed-session ambiguous rather than missed
      deterministic repairs.

- D1-149 Gemini auto-agent empty-success records trigger rollover and stay out
  of usage totals
  - Goal: fix the Gemini Code Assist adapter path used by
    `aawm-codex-agent-auto` so a 200 response with no assistant content and zero
    or one output token is treated as a retryable/malformed candidate response,
    cools down that Gemini model, and rolls to the next auto-agent candidate
    instead of persisting a zero-token successful `session_history` row.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - The Gemini Code Assist non-stream auto-probe path now checks the collected
      `ModelResponse` before Responses transformation and raises
      `aawm_codex_auto_agent_empty_success` for null/no-text/no-tool responses
      with missing, zero, or one output token.
    - The shared empty-success raiser now accepts adapter labels, so Gemini
      diagnostics identify `codex_auto_agent_google_code_assist` instead of
      reporting as OpenRouter-only.
    - The guard raises before
      `LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response`,
      which prevents the malformed Gemini response from returning HTTP 200 and
      entering the normal success logging path.
    - Route-level coverage proves an empty-success first Gemini candidate is
      cooled down for `10800.0` seconds and the selector continues to
      `gemini-3-flash-preview`.
    - Non-regression coverage proves a no-text Gemini response with tool calls
      is not treated as malformed.
    - Dev `litellm-dev` was restarted on `:4001`, and runtime inspection showed
      the new `_is_codex_google_code_assist_empty_success_model_response`
      helper and adapter-labeled raiser signature are loaded.
    - Before live validation, the malformed zero-token Gemini auto-agent
      baseline was `session_history.id=799856` at
      `2026-05-25 12:27:07.850924+00`.
    - A live `codex exec --profile litellm-dev -m aawm-codex-agent-auto`
      smoke returned the exact requested response
      `d1-149-gemini-empty-success-smoke-20260525`.
    - After the baseline, live auto-agent rows showed positive-token traffic
      and no new malformed Gemini successes:
      `gemini|gemini-3.1-flash-lite-preview|codex_google_code_assist_adapter|5|0|2026-05-25 12:58:42.509394+00|2026-05-25 12:58:49.911055+00|57181|176|litellm|litellm`
      and
      `openai|gpt-5.3-codex-spark|codex_responses|25|0|2026-05-25 13:00:20.334811+00|2026-05-25 13:02:19.136528+00|1641992|15004|litellm|litellm`,
      where the fourth column is row count and the fifth is malformed
      zero/one-token candidate count.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k "codex_google_code_assist_auto_probe_empty_success_raises_before_logging or codex_google_code_assist_empty_success_keeps_tool_calls_valid or codex_auto_agent_alias_gemini_empty_success_rolls_to_next_gemini or codex_auto_agent_alias_openrouter_empty_success_rolls_to_last_resort or codex_auto_agent_alias_openrouter_one_token_text_is_success"`
      passed: `5 passed, 350 deselected`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k "codex_auto_agent or codex_google_code_assist or google_code_assist_route or empty_success"`
      passed: `27 passed, 328 deselected`.
    - `docker compose -f docker-compose.dev.yml restart litellm-dev` completed,
      and `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`.
    - `docker exec litellm-dev python -c "import inspect; from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as m; print(inspect.getsource(m._is_codex_google_code_assist_empty_success_model_response).splitlines()[0]); print(inspect.signature(m._raise_codex_auto_agent_empty_success_response))"`
      printed the new helper name and the raiser signature with `adapter` and
      `adapter_label` parameters.
    - `codex exec --profile litellm-dev -m aawm-codex-agent-auto -c model_catalog_json='\"/home/zepfu/.codex/aawm-model-catalog.json\"' "Reply exactly: d1-149-gemini-empty-success-smoke-20260525"`
      passed and returned the expected text.
    - `psql -Atqc "select count(*) from public.session_history where id > 799856 and provider='gemini' and metadata->>'passthrough_route_family'='codex_google_code_assist_adapter' and metadata->>'codex_auto_agent_alias'='aawm-codex-agent-auto' and coalesce(input_tokens,0)=0 and coalesce(output_tokens,0)<=1;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned `0`.
    - `docker logs --since 2026-05-25T12:58:00 litellm-dev 2>&1 | rg -n "aawm_codex_auto_agent_empty_success|empty successful Responses|Traceback|Exception in ASGI|ERROR"`
      returned no matches.

- D1-150 Session-history repository attribution rejects placeholders and repairs
  resolvable memory rows
  - Goal: tighten repository normalization and repair recent bad repository
    attribution so `session_history.repository`, `tenant_id`, metadata
    repository fields, and Langfuse/user identity do not preserve placeholders
    or transcript filenames as repository names.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `scripts/repair_session_history_repository_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_scripts/test_repair_session_history_repository_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Repository normalization now rejects placeholder and transcript-artifact
      values including `...`, `remote`, `memories`, `new`,
      `rollout-2026-`, rollout transcript filenames, and `.json`/`.jsonl`
      labels, including memory-suffixed variants, unless a deterministic memory
      registry lookup maps the value to a real repo.
    - The repair script now loads rollout/thread-to-repository mappings from
      `/home/zepfu/.codex/memories/MEMORY.md` and
      `/home/zepfu/.codex/memories/rollout_summaries/`, supports focused
      `--repository-value` scans, maps resolvable rollout memory rows to
      `<repo> (memory)`, and nulls non-deterministic placeholder fragments.
    - Focused live repair against `aawm_tristore.public.session_history`
      processed `candidate_rows=115` and `repairable_rows=115`, with
      `repair_source rollout_memory_registry=1` and
      `repair_source row_identity_normalization=114`.
    - Exact repair results:
      `id=773374` changed from
      `rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)`
      to `litellm (memory)` for both `repository` and `tenant_id`, with
      metadata `source_repository=litellm` and
      `trace_user_id=litellm (memory)`.
    - Non-deterministic labels were nulled instead of preserved as fake repos:
      `repository='...'` (`80` rows), `remote (memory)` (`16`),
      `memories (memory)` (`13`), `new (memory)` (`4`), and
      `rollout-2026-` (`1`).
  - Verification:
    - `./.venv/bin/python -m py_compile scripts/repair_session_history_repository_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_scripts/test_repair_session_history_repository_identity.py -q -k 'normalize_repository_identity_rejects_metadata_noise or normalize_passthrough_repository_rejects_placeholders_and_transcripts or repair'`
      passed: `37 passed, 540 deselected`.
    - `./.venv/bin/python scripts/repair_session_history_repository_identity.py --dsn postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore --batch-size 500 --preview-limit 40 --repository-value '...' --repository-value 'rollout-2026-' --repository-value 'rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)' --repository-value 'remote (memory)' --repository-value 'memories (memory)' --repository-value 'new (memory)' --apply`
      returned `database=aawm_tristore`, `candidate_rows=115`,
      `repairable_rows=115`, and `applied=true`.
    - `psql -Atqc "select coalesce(repository,'<null>') as repository, coalesce(tenant_id,'<null>') as tenant_id, count(*) from public.session_history where repository in ('...','rollout-2026-','rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)','remote (memory)','memories (memory)','new (memory)') or tenant_id in ('...','rollout-2026-','rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)','remote (memory)','memories (memory)','new (memory)') group by 1,2 order by count(*) desc, 1;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned no rows.
    - `psql -Atqc "select id, repository, tenant_id, metadata->>'repository', metadata->>'source_repository', metadata->>'trace_user_id', metadata->>'repository_repair_source', metadata->>'repository_original_value' from public.session_history where id in (773374,716586,773370) order by id;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned `773374|litellm (memory)|litellm (memory)|litellm (memory)|litellm|litellm (memory)||`,
      while the non-deterministic sample rows `716586` and `773370` had null
      repository/tenant/metadata repository fields.
    - A follow-up focused dry run returned `candidate_rows=0`,
      `repairable_rows=0`, and `applied=false`.

## 2026-05-24

- D1-060 session-history prompt overhead adapted-route final validation
  - Goal: close the remaining adapted-route slice of prompt-overhead
    observability by proving live route-family coverage for Anthropic-adapted
    OpenAI Responses, Google Code Assist, NVIDIA, OpenRouter Chat Completions,
    and OpenRouter Responses traffic, and decide whether exact input-cost
    storage is required beyond proportional estimated allocation.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Fresh dev Anthropic -> OpenAI Responses smoke:
      `POST http://127.0.0.1:4001/anthropic/v1/messages` with
      `model=openai/gpt-5.4-mini`, session
      `d1-060-openai-adapter-drain-20260525-001`, and trace
      `d1-060-openai-adapter-drain-smoke-001` returned HTTP 200 with response
      model `gpt-5.4-mini-2026-03-17` and usage `20/5`.
    - Exact database row `aawm_tristore.public.session_history.id=789302`
      contains `provider=openai`, `model=gpt-5.4-mini`,
      `call_type=pass_through_endpoint`, `input_tokens=20`,
      `output_tokens=5`,
      `passthrough_route_family=anthropic_openai_responses_adapter`,
      `anthropic_adapter_original_model=openai/gpt-5.4-mini`,
      `prompt_overhead_route_family=anthropic_openai_responses_adapter`,
      `prompt_overhead_counted_shape=openai_responses`,
      `prompt_overhead_breakdown_source=request_body_estimate`, and
      `aawm_stream_chunk_count=3`.
    - The OpenAI adapter logging gap was caused by
      `_collect_responses_response_from_stream()` returning immediately on
      `response.completed` and closing the upstream stream before the
      pass-through streaming wrapper reached its post-loop logging callback.
      The collector now captures the completed payload, drains the stream, and
      only then returns the reconstructed response.
    - Final route-family coverage query returned a current row for every
      required adapted route:
      `anthropic_google_completion_adapter|787460|gemini|gemini-3.1-flash-lite-preview|9|3|gemini_generate_content|request_body_estimate`,
      `anthropic_nvidia_completion_adapter|788722|nvidia_nim|minimaxai/minimax-m2.7|43|8|openai_chat_completions|request_body_estimate`,
      `anthropic_openai_responses_adapter|789302|openai|gpt-5.4-mini|20|5|openai_responses|request_body_estimate`,
      `anthropic_openrouter_completion_adapter|788887|openrouter|openrouter/inclusionai/ling-2.6-flash|23|2|openai_chat_completions|request_body_estimate`,
      and
      `anthropic_openrouter_responses_adapter|788442|openrouter|openrouter/owl-alpha|98|2|openai_responses|request_body_estimate`.
    - Exact input-cost storage is not required for this slice. The accepted
      reporting contract remains proportional estimated allocation from stored
      total `response_cost_usd`, because provider tokenization and route-shape
      conversions differ by adapter and exact provider input-cost components
      are not reliably available across all routes. Residual tokens and
      estimator source remain persisted for auditability.
  - Verification:
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"` after restarting `litellm-dev`.
    - `curl -sS -X POST http://127.0.0.1:4001/anthropic/v1/messages -H 'Authorization: Bearer sk-1234' -H 'content-type: application/json' -H 'x-litellm-session-id: d1-060-openai-adapter-drain-20260525-001' -d '{"model":"openai/gpt-5.4-mini","max_tokens":16,"litellm_metadata":{"trace_name":"d1-060-openai-adapter-drain-smoke-001","tenant_id":"litellm","repository":"litellm"},"messages":[{"role":"user","content":"Reply exactly: OK"}]}'`
      returned the expected Anthropic-compatible response with `usage`
      `input_tokens=20` and `output_tokens=5`.
    - `psql -Atqc "SET statement_timeout='15s'; SELECT pg_sleep(2); SELECT id, created_at, session_id, provider, model, model_group, call_type, input_tokens, output_tokens, metadata->>'passthrough_route_family', metadata->>'trace_name', metadata->>'anthropic_adapter_original_model', metadata->>'prompt_overhead_route_family', metadata->>'prompt_overhead_counted_shape', metadata->>'prompt_overhead_breakdown_source', metadata->>'aawm_stream_chunk_count' FROM public.session_history WHERE id >= 789071 AND (session_id = 'd1-060-openai-adapter-drain-20260525-001' OR metadata->>'trace_name' = 'd1-060-openai-adapter-drain-smoke-001' OR metadata->>'passthrough_route_family' = 'anthropic_openai_responses_adapter') ORDER BY id DESC LIMIT 20;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned row `789302` with the expected OpenAI adapter fields.
    - `psql -Atqc "SET statement_timeout='15s'; WITH wanted(route_family) AS (VALUES ('anthropic_openai_responses_adapter'), ('anthropic_google_completion_adapter'), ('anthropic_nvidia_completion_adapter'), ('anthropic_openrouter_completion_adapter'), ('anthropic_openrouter_responses_adapter')), rows AS (SELECT DISTINCT ON (metadata->>'passthrough_route_family') metadata->>'passthrough_route_family' AS route_family, id, created_at, provider, model, call_type, input_tokens, output_tokens, metadata->>'anthropic_adapter_original_model' AS original_model, metadata->>'prompt_overhead_route_family' AS overhead_route, metadata->>'prompt_overhead_counted_shape' AS counted_shape, metadata->>'prompt_overhead_breakdown_source' AS source, metadata->>'aawm_stream_chunk_count' AS stream_chunks FROM public.session_history WHERE id >= 786000 AND metadata->>'passthrough_route_family' IN ('anthropic_openai_responses_adapter','anthropic_google_completion_adapter','anthropic_nvidia_completion_adapter','anthropic_openrouter_completion_adapter','anthropic_openrouter_responses_adapter') ORDER BY metadata->>'passthrough_route_family', id DESC) SELECT wanted.route_family, coalesce(rows.id::text,'MISSING'), rows.created_at, rows.provider, rows.model, rows.call_type, rows.input_tokens, rows.output_tokens, rows.original_model, rows.overhead_route, rows.counted_shape, rows.source, rows.stream_chunks FROM wanted LEFT JOIN rows USING (route_family) ORDER BY wanted.route_family;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned no missing route families.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_anthropic_proxy_route_adapts_allowlisted_openai_model_to_responses tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_collect_responses_stream_drains_after_completed_for_logging tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_collect_responses_stream_reconstructs_arguments_from_done tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_resolve_anthropic_openrouter_completion_adapter_model_supports_ling_replacement -q`
      passed: `5 passed`.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.

- D1-148 native Codex/Gemini runtime cache parity audit gate
  - Goal: keep the native Codex/OpenAI and Gemini runtime/cache parity audit as
    a true numbered item, then close it with current live evidence that native
    rows still satisfy the `session_history` observability contract.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Bounded query over the latest 1000 recent OpenAI/Gemini rows returned
      `gemini|127|0|0|0|7|0|2026-05-25 00:42:05.401781+00` and
      `openai|873|0|0|0|866|0|2026-05-25 00:56:04.776665+00`, where the
      columns were provider, row count, missing runtime fields, missing route
      family, missing token fields, cache-read rows, cache-write rows, and
      latest timestamp.
    - Live cache-write rows were absent in the bounded sample, which remains
      acceptable because cache-write alias mapping is covered by focused
      deterministic tests.
  - Verification:
    - `psql -Atqc "SET statement_timeout='15s'; WITH recent AS (SELECT id, created_at, provider, model, call_type, client_name, client_version, client_user_agent, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions, input_tokens, output_tokens, provider_cache_attempted, provider_cache_status, cache_read_input_tokens, cache_creation_input_tokens, metadata->>'passthrough_route_family' AS route_family FROM public.session_history WHERE id >= (SELECT max(id)-2000 FROM public.session_history) AND provider IN ('openai','gemini') ORDER BY id DESC LIMIT 1000) SELECT provider, count(*), count(*) FILTER (WHERE client_name IS NULL OR client_version IS NULL OR client_user_agent IS NULL OR litellm_environment IS NULL OR litellm_version IS NULL OR litellm_fork_version IS NULL OR litellm_wheel_versions IS NULL) AS missing_runtime, count(*) FILTER (WHERE route_family IS NULL OR route_family = '') AS missing_route_family, count(*) FILTER (WHERE input_tokens IS NULL OR output_tokens IS NULL) AS missing_token_fields, count(*) FILTER (WHERE provider_cache_attempted AND cache_read_input_tokens > 0) AS cache_read_rows, count(*) FILTER (WHERE cache_creation_input_tokens > 0) AS cache_write_rows, max(created_at) FROM recent GROUP BY provider ORDER BY provider;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned zero missing runtime, route-family, and token fields for both
      providers.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_marks_openai_prompt_cache_key_as_cache_attempt_without_usage_details tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_maps_gemini_cache_write_aliases tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_from_langfuse_trace_observation_uses_metadata_usage_object_for_gemini tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_from_langfuse_trace_observation_marks_openai_provider_cache_miss -q`
      passed: `6 passed`.

- D1-147 Native Codex/Gemini runtime and provider-cache parity audit follow-up
  - Goal: verify fresh native Codex/OpenAI and Gemini pass-through traffic still
    carries the same `session_history` observability contract as the Anthropic
    adapter path, including runtime/client attribution, route-family metadata,
    token fields, and provider-cache semantics.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - A bounded sample of the latest 1000 OpenAI/Gemini rows in the last 48
      hours returned `missing_runtime=0`, `missing_route_family=0`, and
      `missing_token_fields=0` for both providers.
    - The same bounded sample returned
      `gemini|211|0|0|0|5|0|2026-05-25 00:31:53.701398+00` and
      `openai|789|0|0|0|784|0|2026-05-25 00:34:26.280111+00`, where the
      columns were provider, row count, missing runtime, missing route family,
      missing token fields, cache-read rows, cache-write rows, and latest row
      timestamp.
    - Latest OpenAI rows show `client_name=codex-tui`,
      `client_version=0.133.0`, user-agent present,
      `litellm_environment=dev`, `litellm_version=1.82.3+aawm.60`,
      `litellm_fork_version=aawm.60`, wheel metadata present,
      `provider_cache_status=hit`, cache-read tokens, and route family
      `codex_responses`.
    - Latest Gemini rows show `client_name=codex-tui`,
      `client_version=0.133.0`, user-agent present,
      `litellm_environment=dev`, `litellm_version=1.82.3+aawm.60`,
      `litellm_fork_version=aawm.60`, wheel metadata present, and route family
      `codex_google_code_assist_adapter`.
    - Provider-cache status counts in the 48-hour window returned:
      `gemini|hit|36`, `gemini|not_attempted|495`,
      `openai|hit|21126`, `openai|miss|1018`, and
      `openai|not_attempted|148`.
    - No fresh live cache-write rows were present for OpenAI/Gemini in the
      bounded sample; cache-write mapping remains covered by deterministic
      focused unit tests because live cache-write traffic is rare.
  - Verification:
    - `psql -Atqc "SET statement_timeout='10s'; WITH recent AS (SELECT id, created_at, provider, model, call_type, client_name, client_version, client_user_agent, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions, input_tokens, output_tokens, provider_cache_attempted, provider_cache_status, cache_read_input_tokens, cache_creation_input_tokens, metadata->>'passthrough_route_family' AS route_family FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider IN ('openai','gemini') ORDER BY id DESC LIMIT 1000) SELECT provider, count(*), count(*) FILTER (WHERE client_name IS NULL OR client_version IS NULL OR client_user_agent IS NULL OR litellm_environment IS NULL OR litellm_version IS NULL OR litellm_fork_version IS NULL OR litellm_wheel_versions IS NULL) AS missing_runtime, count(*) FILTER (WHERE route_family IS NULL OR route_family = '') AS missing_route_family, count(*) FILTER (WHERE input_tokens IS NULL OR output_tokens IS NULL) AS missing_token_fields, count(*) FILTER (WHERE provider_cache_attempted AND cache_read_input_tokens > 0) AS cache_read_rows, count(*) FILTER (WHERE cache_creation_input_tokens > 0) AS cache_write_rows, max(created_at) FROM recent GROUP BY provider ORDER BY provider;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned zero missing fields for both providers.
    - `psql -Atqc "SET statement_timeout='10s'; SELECT id, created_at, provider, model, call_type, client_name, client_version, client_user_agent IS NOT NULL AS has_ua, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions IS NOT NULL AS has_wheels, provider_cache_attempted, provider_cache_status, cache_read_input_tokens, cache_creation_input_tokens, metadata->>'passthrough_route_family' FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider IN ('openai','gemini') ORDER BY id DESC LIMIT 20;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned current OpenAI rows with populated runtime/client/cache fields.
    - `psql -Atqc "SET statement_timeout='10s'; SELECT id, created_at, provider, model, call_type, client_name, client_version, client_user_agent IS NOT NULL AS has_ua, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions IS NOT NULL AS has_wheels, provider_cache_attempted, provider_cache_status, cache_read_input_tokens, cache_creation_input_tokens, metadata->>'passthrough_route_family' FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider='gemini' ORDER BY id DESC LIMIT 10;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned current Gemini rows with populated runtime/client fields and
      route family `codex_google_code_assist_adapter`.
    - `psql -Atqc "SET statement_timeout='10s'; SELECT provider, provider_cache_status, count(*) FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider IN ('openai','gemini') AND provider_cache_status IS NOT NULL GROUP BY provider, provider_cache_status ORDER BY provider, provider_cache_status;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned hit/miss/not-attempted counts for OpenAI and hit/not-attempted
      counts for Gemini.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_marks_openai_prompt_cache_key_as_cache_attempt_without_usage_details tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_maps_gemini_cache_write_aliases tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_from_langfuse_trace_observation_uses_metadata_usage_object_for_gemini tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_from_langfuse_trace_observation_marks_openai_provider_cache_miss -q`
      passed: `6 passed`.

- D1-064 OpenRouter wildcard observability for unmapped chat models
  - Goal: preserve low-touch wildcard routing for explicit `openrouter/*`
    requests from supported OpenRouter paths while ensuring the exact
    caller-provided model string is preserved in Langfuse and
    `public.session_history` even when pricing is not configured.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/integrations/langfuse/langfuse.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_langfuse.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - `session_history` model resolution now prefers explicit
      `openrouter/...` caller strings from adapter metadata and request bodies
      before falling back to provider-stripped adapter model names.
    - Langfuse generation logging now preserves an explicit OpenRouter model
      string from adapter metadata or the standard logging payload instead of
      reconstructing the provider model as `owl-alpha`.
    - OpenAI-compatible pass-through logging now records zero cost for
      unmapped OpenRouter models with usage instead of falling through to the
      generic chat handler, which cannot parse Responses API bodies without
      `choices`.
    - Dev Anthropic adapter smoke:
      `POST http://127.0.0.1:4001/anthropic/v1/messages` with
      `model=openrouter/owl-alpha` returned HTTP 200 and response model
      `openrouter/owl-alpha`.
    - Exact database row `aawm_tristore.public.session_history.id=788442`
      contains `provider=openrouter`, `model=openrouter/owl-alpha`,
      `call_type=pass_through_endpoint`, `input_tokens=98`,
      `output_tokens=2`,
      `passthrough_route_family=anthropic_openrouter_responses_adapter`, and
      `anthropic_adapter_original_model=openrouter/owl-alpha`.
    - Exact Langfuse ClickHouse generation
      `time-00-30-50-656909_gen-1779669051-CyLRGH1rAWexYAtrX8xu` contains
      `provided_model_name=openrouter/owl-alpha`, `usage_details` input/output
      `98/2`, route family `anthropic_openrouter_responses_adapter`, and
      `anthropic_adapter_original_model=openrouter/owl-alpha`.
    - Dev OpenAI-compatible generic chat smoke:
      `POST http://127.0.0.1:4001/v1/chat/completions` with
      `model=openrouter/owl-alpha` returned HTTP 200 and response model
      `openrouter/owl-alpha`.
    - Exact database row `aawm_tristore.public.session_history.id=788500`
      contains `provider=openrouter`, `model=openrouter/owl-alpha`,
      `model_group=openrouter/owl-alpha`, `call_type=acompletion`,
      `input_tokens=98`, and `output_tokens=2`.
    - Exact Langfuse ClickHouse generation
      `time-00-32-23-308654_gen-1779669144-bXBYKz0uGNRZUD8FEc8a` contains
      `provided_model_name=openrouter/owl-alpha` and `usage_details`
      input/output `98/2`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_preserves_explicit_openrouter_model tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_from_langfuse_preserves_explicit_openrouter_model tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py::TestOpenAIPassthroughLoggingHandler::test_openai_passthrough_handler_preserves_unmapped_openrouter_responses_model tests/test_litellm/integrations/test_langfuse.py::TestLangfuseUsageDetails::test_log_langfuse_v2_preserves_explicit_openrouter_model -q`
      passed: `4 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py::TestOpenAIPassthroughLoggingHandler::test_openai_passthrough_handler_preserves_unmapped_openrouter_responses_model tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py::TestOpenAIPassthroughLoggingHandler::test_openai_passthrough_handler_backfills_openrouter_responses_usage_and_model -q`
      passed: `2 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_langfuse.py::TestLangfuseUsageDetails::test_log_langfuse_v2_preserves_explicit_openrouter_model tests/test_litellm/integrations/test_langfuse.py::TestLangfuseUsageDetails::test_log_langfuse_v2_falls_back_to_standard_logging_object_for_zero_usage tests/test_litellm/integrations/langfuse/test_langfuse_payload_size_audit.py -q`
      passed: `5 passed`.
    - `./.venv/bin/python -m py_compile litellm/integrations/langfuse/langfuse.py tests/test_litellm/integrations/test_langfuse.py litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`
      passed.
    - `./.venv/bin/python -m ruff check litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
      passed.
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"` after reloading `litellm-dev`.
    - `psql -Atqc "SET statement_timeout='10s'; SELECT id, created_at, session_id, provider, model, model_group, call_type, input_tokens, output_tokens, response_cost_usd, metadata->>'passthrough_route_family', metadata->>'anthropic_adapter_original_model', metadata->>'anthropic_adapter_model', metadata->>'trace_name' FROM public.session_history WHERE created_at > now() - interval '15 minutes' AND (metadata->>'trace_name' = 'd1-064-openrouter-wildcard-smoke-003' OR metadata->>'anthropic_adapter_original_model' = 'openrouter/owl-alpha' OR model='openrouter/owl-alpha') ORDER BY id DESC LIMIT 10;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned rows `788442` and `788326` with exact model
      `openrouter/owl-alpha`.
    - `docker exec aawm-clickhouse clickhouse-client --database default --query "SELECT id, trace_id, start_time, name, type, provided_model_name, usage_details, provided_usage_details, metadata['passthrough_route_family'], metadata['anthropic_adapter_original_model'], metadata['trace_name'] FROM observations WHERE start_time >= toDateTime64('2026-05-25 00:28:00', 3) AND type = 'GENERATION' AND (provided_model_name LIKE '%owl-alpha%' OR metadata['anthropic_adapter_original_model'] = 'openrouter/owl-alpha' OR metadata['trace_name'] LIKE '%d1-064%') ORDER BY start_time DESC LIMIT 20 FORMAT TabSeparated"`
      returned Langfuse generation
      `time-00-30-50-656909_gen-1779669051-CyLRGH1rAWexYAtrX8xu` with exact
      `provided_model_name=openrouter/owl-alpha`.
    - `psql -Atqc "SET statement_timeout='10s'; SELECT id, created_at, session_id, provider, model, model_group, call_type, input_tokens, output_tokens, response_cost_usd, metadata->>'passthrough_route_family', metadata->>'trace_name' FROM public.session_history WHERE created_at > now() - interval '15 minutes' AND (metadata->>'trace_name' = 'd1-064-openrouter-generic-chat-smoke' OR model='openrouter/owl-alpha') ORDER BY id DESC LIMIT 10;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned row `788500` with exact model `openrouter/owl-alpha`.
    - `docker exec aawm-clickhouse clickhouse-client --database default --query "SELECT id, trace_id, start_time, name, type, provided_model_name, usage_details, metadata['trace_name'], metadata['passthrough_route_family'] FROM observations WHERE start_time >= toDateTime64('2026-05-25 00:30:00', 3) AND type = 'GENERATION' AND (provided_model_name = 'openrouter/owl-alpha' OR metadata['trace_name'] = 'd1-064-openrouter-generic-chat-smoke') ORDER BY start_time DESC LIMIT 20 FORMAT TabSeparated"`
      returned Langfuse generation
      `time-00-32-23-308654_gen-1779669144-bXBYKz0uGNRZUD8FEc8a` with exact
      `provided_model_name=openrouter/owl-alpha`.

- D1-146 Native Codex/Gemini runtime and provider-cache parity audit
  - Goal: verify current native Codex/OpenAI and Gemini passthrough rows still
    populate runtime/client/cache columns with parity to the Anthropic path.
  - Evidence:
    - Recent live `session_history` OpenAI/Codex rows include populated
      `client_name`, `client_version`, `client_user_agent`,
      `litellm_environment`, `litellm_version`, `litellm_fork_version`, and
      `litellm_wheel_versions`.
    - Recent live Gemini passthrough rows include the same runtime/client/wheel
      fields.
    - Recent OpenAI/Codex rows show provider-cache hits from OpenAI Responses
      cached token details, for example row `787685`:
      `provider=openai`, `model=gpt-5.5`, `client_name=codex-tui`,
      `client_version=0.133.0`, `provider_cache_attempted=true`,
      `provider_cache_status=hit`, `cache_read_input_tokens=86912`,
      `passthrough_route_family=codex_responses`.
    - Recent Gemini rows show provider-cache hit mapping, for example row
      `787443`: `provider=gemini`,
      `model=gemini-3.1-flash-lite-preview`,
      `client_name=codex-tui`, `client_version=0.133.0`,
      `provider_cache_attempted=true`, `provider_cache_status=hit`,
      `cache_read_input_tokens=14292`,
      `passthrough_route_family=codex_google_code_assist_adapter`.
    - A bounded 48-hour sample of the latest 1000 OpenAI/Gemini rows returned:
      `gemini|283|0|15|0` and `openai|717|0|713|0`, where the columns were
      provider, row count, missing-runtime count, cache-read rows, and
      cache-write rows.
    - No recent live cache-write rows were present in that sample, so cache-write
      parity was verified by targeted unit coverage rather than rare live
      traffic.
  - Verification:
    - `psql -Atqc "SET statement_timeout='10s'; SELECT id, created_at, provider, model, call_type, repository, client_name, client_version, client_user_agent IS NOT NULL AS has_ua, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions IS NOT NULL AS has_wheels, provider_cache_attempted, provider_cache_status, cache_read_input_tokens, cache_creation_input_tokens, metadata->>'passthrough_route_family', metadata->>'usage_object' FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider='gemini' ORDER BY id DESC LIMIT 20;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned populated runtime fields for recent Gemini rows, including row
      `787443` with a cache hit.
    - `psql -Atqc "SET statement_timeout='10s'; SELECT id, created_at, provider, model, call_type, repository, client_name, client_version, client_user_agent IS NOT NULL AS has_ua, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions IS NOT NULL AS has_wheels, provider_cache_attempted, provider_cache_status, cache_read_input_tokens, cache_creation_input_tokens, metadata->>'passthrough_route_family', metadata->>'prompt_cache_key', metadata->>'usage_object' FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider IN ('openai','gemini') ORDER BY id DESC LIMIT 30;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned populated runtime fields and OpenAI cache hits for recent Codex
      rows.
    - `psql -Atqc "SET statement_timeout='10s'; WITH recent AS (SELECT id, provider, client_name, client_version, client_user_agent, litellm_environment, litellm_version, litellm_fork_version, litellm_wheel_versions, provider_cache_attempted, cache_read_input_tokens, cache_creation_input_tokens FROM public.session_history WHERE created_at >= now() - interval '48 hours' AND provider IN ('openai','gemini') ORDER BY id DESC LIMIT 1000) SELECT provider, count(*), count(*) FILTER (WHERE client_name IS NULL OR client_version IS NULL OR client_user_agent IS NULL OR litellm_environment IS NULL OR litellm_version IS NULL OR litellm_fork_version IS NULL OR litellm_wheel_versions IS NULL) AS missing_runtime, count(*) FILTER (WHERE provider_cache_attempted AND cache_read_input_tokens > 0) AS cache_read_rows, count(*) FILTER (WHERE cache_creation_input_tokens > 0) AS cache_write_rows FROM recent GROUP BY provider ORDER BY provider;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned `missing_runtime=0` for both providers.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_marks_openai_prompt_cache_key_as_cache_attempt_without_usage_details tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_maps_gemini_cache_write_aliases tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_from_langfuse_trace_observation_uses_metadata_usage_object_for_gemini -q`
      passed: `5 passed`.

- D1-138 Google Code Assist streaming adapter response-builder 502
  - Goal: make successful Google Code Assist streaming responses build a
    complete Anthropic-compatible response instead of returning
    `Google Code Assist streaming adapter could not build a complete response`.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Hardened Gemini pass-through stream parsing so a single raw chunk
      containing multiple SSE `data:` frames is split into all JSON payloads
      before response reconstruction.
    - The response builder now preserves the selected model on reconstructed
      stream responses when `stream_chunk_builder` returns a response without
      a model value.
    - Focused unit coverage covers a combined one-chunk Code Assist SSE stream
      with content, usage, and `[DONE]`.
    - Dev `litellm-dev` was recreated from `docker-compose.dev.yml` so the
      bind-mounted handler changes were active.
    - Fresh dev smoke:
      `POST http://127.0.0.1:4001/anthropic/v1/messages` with
      `model=google/gemini-3.1-flash-lite-preview` returned HTTP 200 with
      Anthropic message JSON containing `model=gemini-3.1-flash-lite-preview`,
      `content[0].text=Gemini ok`, and usage `input_tokens=9`,
      `output_tokens=3`.
    - Exact database `aawm_tristore.public.session_history.id=787460` contains
      `provider=gemini`, `model=gemini-3.1-flash-lite-preview`,
      `call_type=pass_through_endpoint`, `input_tokens=9`, `output_tokens=3`,
      `passthrough_route_family=anthropic_google_completion_adapter`, and
      `aawm_stream_chunk_count=2`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py::TestGeminiPassthroughLoggingHandler::test_build_complete_streaming_response_code_assist_combined_sse_chunk tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py::TestGeminiPassthroughLoggingHandler::test_build_complete_streaming_response_code_assist_list_wrapped_sse_chunks tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py::TestGeminiPassthroughLoggingHandler::test_build_complete_streaming_response_code_assist_preserves_tool_call_from_non_final_chunk -q`
      passed: `3 passed`.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py`
      passed.
    - `./.venv/bin/python -m ruff check litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_gemini_passthrough_logging_handler.py`
      passed.
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`.
    - `psql -Atqc "SET statement_timeout='5s'; SELECT id, created_at, session_id, provider, model, model_group, call_type, input_tokens, output_tokens, metadata->>'passthrough_route_family', metadata->>'trace_name', metadata->>'aawm_stream_chunk_count' FROM public.session_history WHERE created_at > now() - interval '10 minutes' AND (session_id = 'd1-138-smoke-20260525-0001' OR metadata->>'trace_name' = 'd1-138-google-stream-smoke' OR model='gemini-3.1-flash-lite-preview') ORDER BY id DESC LIMIT 10;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned row `787460` with the matching Gemini adapter fields.

- D1-129 Codex auto-agent in-flight 429 redispatch hint
  - Goal: make terminal in-flight `aawm-codex-agent-auto` retryable provider
    exhaustion visible to the dispatching Codex orchestrator as an actionable
    child failure message telling it to redispatch a fresh subagent with the
    same alias.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added `_raise_codex_auto_agent_redispatch_required` and wired it into
      `_handle_codex_auto_agent_alias_route` only for stateful continuation
      requests where the selected in-flight candidate hits retryable exhaustion.
    - The shaped `429` uses stable code
      `aawm_codex_auto_agent_redispatch_required`; its message says not to
      continue the child agent and to redispatch a fresh subagent using
      `aawm-codex-agent-auto`.
    - The response includes `redispatch_model`, `redispatch_reason`,
      selected provider/model/route family, cooldown seconds, retry-after
      seconds, error tokens, and candidate detail.
    - Existing already-cooling in-flight behavior remains
      `aawm_codex_auto_agent_in_flight_provider_cooling_down`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::test_codex_auto_agent_alias_in_flight_affinity_429_is_terminal tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::test_codex_auto_agent_alias_in_flight_affinity_cooldown_does_not_switch -q`
      passed: `2 passed`.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m ruff check litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      still reports pre-existing full-file lint debt in the large pass-through
      module/test file, including unrelated unused imports, print statements,
      and complexity findings.

- D1-145 Langfuse oversized event diagnostics
  - Goal: add targeted local diagnostics before the Langfuse SDK emits
    warnings like
    `Item exceeds size limit ..., dropping input / output / metadata of item until it fits`,
    while avoiding raw prompt, output, metadata value, or secret logging.
  - Changed paths:
    - `litellm/integrations/langfuse/langfuse.py`
    - `tests/test_litellm/integrations/langfuse/test_langfuse_payload_size_audit.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added a pre-SDK enqueue size audit immediately before
      `trace.generation(**generation_params)`.
    - The audit logs only when the estimated serialized Langfuse generation
      payload is at least 90 percent of `LANGFUSE_MAX_EVENT_SIZE_BYTES`
      or the SDK default `1_000_000` bytes.
    - The warning includes trace id, generation id/name, model, call type,
      total size, limit/threshold, `input`/`output`/`metadata`/
      `model_parameters` byte sizes, and the largest metadata keys by byte
      size.
    - Metadata values are not logged, and sensitive metadata key names are
      redacted to `<redacted-key>`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/langfuse/test_langfuse_payload_size_audit.py -q`
      passed: `3 passed`.
    - `./.venv/bin/python -m py_compile litellm/integrations/langfuse/langfuse.py tests/test_litellm/integrations/langfuse/test_langfuse_payload_size_audit.py`
      passed.
    - `./.venv/bin/python -m ruff check litellm/integrations/langfuse/langfuse.py tests/test_litellm/integrations/langfuse/test_langfuse_payload_size_audit.py`
      passed.

- D1-095 TinyBERN2 sidecar model-id configuration for live annotation success
  - Goal: confirm the TAP `biomed-tinybern2` sidecar is configured with a real
    model id and that the dev LiteLLM local biomedical pass-through route can
    annotate successfully.
  - Evidence:
    - `curl -sS http://127.0.0.1:4001/aawm/local/tinybern2/health` returned
      `status=ok`, `model_configured=true`,
      `model=dmis-lab/KAZU-NER-module-distil-v1.0`, `device=cuda`, and
      `operational_use_allowed=false`.
    - `docker ps --filter name=tinybern2 --format '{{.Names}} {{.Status}} {{.Ports}}'`
      returned `aawm-tap-biomed-tinybern2-dev Up 5 hours (healthy)` with
      host port `8095`.
    - Dev LiteLLM annotate smoke against
      `/aawm/local/tinybern2/annotate` with text
      `BRCA1 is associated with breast cancer.` returned a non-error payload
      containing proposed entities from `source=tinybern2`.
    - Exact database `aawm_tristore.public.session_history.id=785666` contains
      `provider=local_biomed`, `model=tinybern2`, `model_group=tinybern2`,
      `call_type=pass_through_endpoint`, `tenant_id=litellm`,
      `repository=litellm`, `aawm_local_service=tinybern2`, and
      `aawm_local_endpoint=annotate`.
  - Verification:
    - `psql -Atqc "SELECT id, created_at, start_time, end_time, session_id, provider, model, model_group, call_type, tenant_id, repository, input_tokens, output_tokens, response_cost_usd, metadata->>'passthrough_route_family', metadata->>'aawm_local_route_family', metadata->>'aawm_local_service', metadata->>'aawm_local_endpoint', metadata->>'aawm_local_upstream_api_base' FROM public.session_history WHERE id = 785666;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned route metadata ending in
      `local_biomed|local_biomed_rest|tinybern2|annotate|http://172.20.0.1:8095/annotate`.
  - Notes:
    - The smoke request supplied `x-litellm-session-id`, but this local
      pass-through route stored a generated UUID session id. That does not
      reopen the original D1-095 blocker, which was TinyBERN2 refusing
      annotation due to missing model configuration.

- D1 operational log-noise demotion prod cutover
  - Goal: reduce routine LiteLLM request-flow noise in dev/prod logs while
    preserving warnings for failures, retries, cooldown-active events, and
    telemetry-loss conditions.
  - Changed paths:
    - `litellm/integrations/langfuse/langfuse.py`
    - `litellm/proxy/route_llm_request.py`
    - `litellm/router.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/vertex_passthrough_logging_handler.py`
    - `PATCHES.md`
    - `pyproject.toml`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - LiteLLM commit
      `76a178357d chore(proxy): demote routine operational logs` is pushed to
      `origin/main` and `origin/develop`, and tagged `v1.82.3-aawm.61`.
    - GitHub image workflow run `26374724135` succeeded for
      `v1.82.3-aawm.61` and published
      `ghcr.io/zepfu/litellm:1.82.3-aawm.61`.
    - `litellm-dev` was rebuilt/recreated from the local dev compose file after
      the logging changes; `http://127.0.0.1:4001/health` returned HTTP 200.
    - Infra commit `eaf5d21 chore(litellm): promote aawm.61 image` is pushed
      to `origin/develop` and `origin/main`, and pins both the Dockerfile
      default and compose build arg to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.61`.
    - Built prod image inspection returned `1.82.3+aawm.61`,
      `aawm-litellm-callbacks=0.0.35`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Prod `aawm-litellm` was recreated from the new image. Readiness returned
      HTTP 200 with `status=healthy`, `litellm_version=1.82.3+aawm.61`, and
      `log_level=WARNING`; running package inspection returned
      `1.82.3+aawm.61`, `0.0.35`, and `0.0.7`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/langfuse/langfuse.py litellm/proxy/route_llm_request.py litellm/router.py litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py litellm/proxy/pass_through_endpoints/llm_provider_handlers/vertex_passthrough_logging_handler.py`
      passed.
    - `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      built `aawm-litellm:latest` from
      `ghcr.io/zepfu/litellm:1.82.3-aawm.61`.
    - `docker compose -f docker-compose.litellm.yml up -d litellm` recreated
      prod successfully.

- D1-143 / D1-144 durable `aawm-codex-agent-auto` fallback release and prod cutover
  - Goal: make the D1-142 fresh-dispatch affinity fix durable in a tagged image,
    add OpenRouter DeepSeek empty-success rollover, enforce per-candidate
    approximately three-hour cooldowns, and promote the result to prod.
  - Required behavior closed:
    - Fresh dispatch scans from the top of the ordered list:
      `openai/gpt-5.3-codex-spark`,
      `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`,
      `gemini-3.1-pro-preview`,
      `openrouter/deepseek/deepseek-v4-flash:free`, then
      `openai/gpt-5.4-mini` as last resort.
    - Continuation affinity applies only to the exact active session/thread
      when continuation state is present.
    - Retryable first-attempt `429`, `529`, quota/capacity equivalents, and the
      observed malformed empty-success DeepSeek shape cool down only the
      selected candidate and roll forward.
    - Legitimate one-token text responses are not rejected globally.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `PATCHES.md`
    - `pyproject.toml`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - LiteLLM commit
      `0bd8c5eeeb fix(aawm): harden codex auto-agent fallback` is pushed to
      `origin/main` and `origin/develop`, and tagged `v1.82.3-aawm.60`.
    - GitHub release `v1.82.3-aawm.60` is published with image
      `ghcr.io/zepfu/litellm:1.82.3-aawm.60`; image workflow run
      `26371025214` succeeded.
    - Infra commit `9dad5b8 chore(litellm): pin aawm60 image` is pushed to
      `origin/develop` and `origin/main`, and pins both the Dockerfile default
      and compose build arg to `ghcr.io/zepfu/litellm:1.82.3-aawm.60`.
    - Prod `aawm-litellm` was rebuilt and force-recreated from the pinned image.
      Readiness returned `status=healthy` and
      `litellm_version=1.82.3+aawm.60`; package inspection returned
      `1.82.3+aawm.60`.
    - Live container source inspection returned `True` for
      `_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS = 3 * 60 * 60.0`,
      `status_code in {429, 529}`,
      `_is_codex_auto_agent_empty_success_responses_body`,
      `not has_continuation_state`, and
      `aawm_codex_auto_agent_empty_success`.
    - `docker diff aawm-litellm | rg 'llm_passthrough_endpoints.py|site-packages/litellm'`
      returned no matches after recreation, confirming the passthrough code is
      image-baked rather than a container filesystem hotfix.
    - Prod runtime selector check returned
      `fresh first_available openai gpt-5.3-codex-spark` and
      `continuation session_affinity openrouter deepseek/deepseek-v4-flash:free`.
    - Dev `litellm-dev` was restarted from `docker-compose.dev.yml` after the
      prod cutover. Readiness returned `status=healthy`; the dev image still
      reports package metadata `litellm_version=1.82.3+aawm.25`, but the
      relevant module is bind-mounted from `/app/litellm/...` and live source
      inspection returned `True` for the same D1-143/D1-144 markers. Dev
      runtime selector check returned
      `fresh first_available openai gpt-5.3-codex-spark` and
      `continuation session_affinity openrouter deepseek/deepseek-v4-flash:free`.
    - Read-only `psql` check against exact database
      `aawm_tristore.public.session_history` returned `0` post-cutover rows
      since `2026-05-24 20:00:00+00` for
      `model like '%deepseek-v4-flash:free%'`, `output_tokens <= 1`, and
      `tool_call_count=0`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_auto_agent_alias'`
      passed: `15 passed, 327 deselected`.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `git diff --check` passed.
    - `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      built `aawm-litellm:latest` from
      `ghcr.io/zepfu/litellm:1.82.3-aawm.60`.
    - `docker compose -f docker-compose.litellm.yml up -d --no-build --force-recreate --no-deps litellm`
      recreated the prod container successfully.

- D1 queue reconciliation against completed evidence
  - Goal: remove stale completed status summaries from `.analysis/todo.md` and
    leave only current open work plus deferred adapter-fidelity backlog.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Reconciled `.analysis/todo.md` against `.analysis/completed.md`.
    - Removed completed/historical TODO entries for D1-142, D1-140, D1-139,
      D1-136, D1-132, D1-105, D1-104, D1-100, D1-097, D1-096, D1-094,
      D1-091, D1-090, D1-086, D1-085, D1-081, D1-079, D1-078, D1-077,
      D1-076, and D1-075.
    - At the time of that reconciliation, kept then-open items such as
      D1-143, D1-144, D1-138, D1-129, D1-117, D1-123, D1-107 final live
      validation, D1-106, D1-103 downstream handoff, D1-101, D1-098, D1-095,
      D1-071 reporting, D1-064, D1-060, and the native Codex/Gemini
      runtime/cache parity audit. D1-143 and D1-144 are now closed in the
      entry above.
    - `rg -n "^##|^- D1-|^- Native|^- Anthropic|^- Gemini|^- OpenRouter|^- Raw|^- Prompt|^- Documents|^- Thinking|^- Context|^- Response|^- Full|^- Remote" .analysis/todo.md`
      shows the cleaned active queue and deferred backlog.

- D1-142 `aawm-codex-agent-auto` fresh-dispatch affinity leak runtime hotfix
  - Goal: stop fresh `aawm-codex-agent-auto` medium child dispatches from
    inheriting stale session affinity to OpenRouter
    `deepseek/deepseek-v4-flash:free` after Spark/Gemini cooldowns have
    expired, while preserving provider stickiness for true continuation
    requests.
  - Finding:
    - Recent `aawm-tap` auto-agent rows showed `960` DeepSeek-selected rows in
      the previous 24 hours and all `960` had exactly `output_tokens=1`.
    - A representative trace returned upstream/proxy HTTP `200`, but normalized
      assistant output was empty, `responses_stream_event_types=[]`, and
      `responses_stream_tool_call_count=0`.
    - The first DeepSeek row at `2026-05-23 16:16:55+00` was selected only
      because Spark and Gemini were cooling down; after that, the six-hour
      session-affinity cache pinned fresh child starts to DeepSeek.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
  - Runtime mitigation:
    - Dev `litellm-dev` was restarted with the bind-mounted checkout change.
    - Prod `aawm-litellm` imports LiteLLM from
      `/usr/lib/python3.13/site-packages`, so the same file was copied there
      and prod was restarted.
  - Evidence:
    - Focused tests passed:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_auto_agent_alias'`
      returned `13 passed, 327 deselected`.
    - Dev runtime selector check returned
      `fresh first_available openai gpt-5.3-codex-spark` and
      `continuation session_affinity openrouter deepseek/deepseek-v4-flash:free`.
    - Prod runtime selector check returned
      `fresh first_available openai gpt-5.3-codex-spark` and
      `continuation session_affinity openrouter deepseek/deepseek-v4-flash:free`.
    - Prod readiness after restart returned `status=healthy` and
      `litellm_version=1.82.3+aawm.59`.

## 2026-05-23

- D1-140 session_history Langfuse backfill field-loss and reconciliation repair
  - Goal: repair the May 23 Langfuse-derived `session_history` rows inserted
    with incomplete top-level fields, harden future backfills/live writes, and
    classify remaining zero-token rows without inventing usage.
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/backfill_session_history.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/d1-140-session-history-incident-writeup.md`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Data repair:
    - Repaired deterministic flattened Langfuse `metadata.usage_*` rows and
      backdated repaired `created_at` values to source timing.
    - Repaired Anthropic token-count rows where MinIO trace output still
      contained `{"input_tokens": ...}`.
    - Repaired live row `public.session_history.id=741082` from MinIO object
      `aawm-dev/trace/81e28a2b-b5d1-4a03-bfcf-457782b6df8f/fa367fef-24e7-4d39-8f15-0ceae1508ea4.json`,
      setting `input_tokens=13237` and `total_tokens=13237`.
    - Classified the remaining zero-token rows:
      `source_payload_missing_anthropic_passthrough=345`,
      `empty_provider_response_no_usage=4`,
      `non_usage_rate_limit_observation=4`, and
      `source_payload_missing_gemini_adapter=3`.
  - Code repair:
    - Langfuse backfill reconstruction now merges flattened
      `metadata.usage_*` fields, recovers token-count payloads, parses raw
      OpenAI Responses output usage/model, and recovers model from deterministic
      adapter metadata/input/output/header sources.
    - Live/session-history reconstruction now handles Anthropic token-count
      pass-through wrappers such as
      `StandardPassThroughResponseObject(response='{"input_tokens":13237}')`.
    - Gemini quota/control-plane and empty adapter response rows are explicitly
      marked as non-usage/zero-token classes instead of appearing as
      unexplained tokenless usage.
  - Evidence:
    - Report window `2026-05-20` to `< 2026-06-01` verification returned
      `bad_model_rows=0` and `unclassified_zero_rows=0`.
    - Zero-token classification breakdown returned
      `source_payload_missing_anthropic_passthrough=345`,
      `empty_provider_response_no_usage=4`,
      `non_usage_rate_limit_observation=4`, and
      `source_payload_missing_gemini_adapter=3`.
    - Flattened `metadata.usage_*` mismatch query for affected
      `LangfuseTraces` rows returned `0`.
    - Source-backed reconciliation returned
      `langfuse_backfilled=23820`, `with_source_observation=23820`, and
      `unclassified_source_zero=0`.
    - Fresh dev rows after the final restart at
      `2026-05-23 15:35:52+00` returned `rows=12`, `bad_model=0`, and
      `unclassified_usage_zero=0`.
    - Dev server `GET http://127.0.0.1:4001/health/liveliness` returned
      `200 OK` with `"I'm alive!"`.
    - Root-cause write-up captured in
      `.analysis/d1-140-session-history-incident-writeup.md`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed: `207 passed`.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_backfill_session_history_gemini_control_plane.py -q`
      passed: `13 passed`.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/backfill_session_history.py`
      passed.
    - `git diff --check` passed.

- D1-139 `aawm.59` prod cutover and final session_history backfill
  - Goal: promote the accumulated session-history, OpenRouter `:free`, and
    alias-routing fixes to prod LiteLLM, then run the final missing
    `session_id` / repository repair pass before cleanup.
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
  - Changed paths:
    - `COMPLETED.md`
    - `.analysis/completed.md`
  - Evidence:
    - LiteLLM release code commit
      `d9ca00b2f89c385c50cb563ece9e9e42de5042ab` is tagged
      `v1.82.3-aawm.59`, with image
      `ghcr.io/zepfu/litellm:1.82.3-aawm.59` published. Closeout docs commit
      `ac8b6a5f5313ac5efd70cedb2afa6d3731c23bd5` is pushed to both
      `origin/main` and `origin/develop`.
    - Overlay releases are current at callback `cb-v0.0.35`, model config
      `cfg-v0.0.13`, harness `h-v0.0.31`, and control-plane `cp-v0.0.7`.
    - `/home/zepfu/projects/aawm-infrastructure` commit
      `abc9195cda5d9ec0a392fe7434f1584613c74990` is pushed to both
      `origin/main` and `origin/develop`, pins prod to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.59`, and adds prod routes for
      `deepseek/deepseek-v4-flash:free` plus
      `openrouter/inclusionai/ling-2.6-flash`.
    - Prod `aawm-litellm` was rebuilt and recreated. Running container
      `ce678c8a9c64` is healthy on `:4000`; readiness reports
      `litellm_version=1.82.3+aawm.59`. Runtime package inspection reports
      `litellm=1.82.3+aawm.59`,
      `aawm-litellm-callbacks=0.0.35`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Focused prod harness artifact
      `/tmp/litellm-prod-openrouter-free-meter-aawm59.json` passed
      `native_openrouter_free_daily_meter_chat` with no failures. The only
      warning was the expected duplicate-snapshot fallback for unchanged
      `rate_limit_observations`.
    - Exact database row `public.session_history.id=718637` records the prod
      harness call with
      `session_id=litellm-harness.prod.native_openrouter_free_daily_meter_chat.1779516320-588833.session`,
      `provider=openrouter`,
      `model=openrouter/openai/gpt-oss-20b:free`,
      `client_name=openrouter-free-meter-harness`,
      `litellm_environment=prod`, `litellm_version=1.82.3+aawm.59`,
      `input_tokens=72`, `output_tokens=17`, `total_tokens=89`,
      `response_cost_usd=2.62e-06`, `repository=litellm`, and
      `tenant_id=adapter-harness-tenant`.
    - Final backfill pass found no additional rows requiring updates.
      Post-backfill verification showed `blank_session=0`,
      `blank_session_with_trace=0`, and `repairable_recent_repo=0` for recent
      blank-repository rows that still contain deterministic repository clues
      in metadata.
  - Verification:
    - `curl -sS http://127.0.0.1:4000/health/readiness` returned
      `status=healthy` and `litellm_version=1.82.3+aawm.59`.
    - `docker ps --filter name=aawm-litellm --format ...` returned
      `ce678c8a9c64|aawm-litellm:latest|Up ... (healthy)`.
    - `docker exec aawm-litellm python -c ...` returned
      `1.82.3+aawm.59`, `0.0.35`, and `0.0.7`.
    - Read-only `psql` checks against `aawm_tristore` confirmed the
      session-history row and post-backfill counts above.

- D1-132 OpenRouter free meter live persistence and acceptance gate
  - Goal: prove fresh OpenRouter `:free` traffic through dev LiteLLM writes
    `public.rate_limit_observations`, and add an opt-in harness process that
    hard-gates the same table path.
  - Database target:
    - Exact database `aawm_tristore.public.session_history`,
      `aawm_tristore.public.rate_limit_observations`, and
      `aawm_tristore.public.rate_limit_intervals`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/local-ci/anthropic_adapter_config.json`
    - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
    - `scripts/local-ci/README.md`
    - `TEST_HARNESS.md`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Root cause of the live callback flush failure was the
      `session_history` conflict upsert calling `jsonb_array_length()` on a
      scalar legacy JSONB value. The upsert now guards `tool_names`,
      `file_paths_read`, and `file_paths_modified` with `jsonb_typeof(...)=
      'array'` before comparing array lengths.
    - Dev `litellm-dev` was restarted and runtime inspection showed
      `_AAWM_OPENROUTER_FREE_DAILY_SOURCE=openrouter_free_daily_local_meter`,
      the OpenRouter free observation builder present, and the guarded
      `tool_names` SQL loaded.
    - Fresh direct dev smoke to
      `openrouter/openai/gpt-oss-20b:free` returned `200` response
      `gen-1779499713-LmYbMTMxD9gfkzLswu8E`; exact database
      `aawm_tristore.public.session_history` row `685772` has
      `provider=openrouter`, `model=openrouter/openai/gpt-oss-20b:free`,
      `input_tokens=72`, `output_tokens=17`, and `total_tokens=89`.
    - The same call wrote exact database
      `aawm_tristore.public.rate_limit_observations` row `67881` with
      `provider=openrouter`, `client=openrouter`,
      `quota_key=openrouter_free_daily_requests:requests`,
      `quota_period=daily`, `quota_type=requests`,
      `expected_reset_at=2026-05-24 00:00:00+00`,
      `remaining_pct=99.8`,
      `source=openrouter_free_daily_local_meter`, and the same
      `litellm_call_id=56cac953-d09b-403e-9b40-dddf08cee416`.
    - The stale dev `public.rate_limit_intervals` materialized view definition
      was rebuilt via the callback schema initializer; refreshed
      `rate_limit_intervals` now has the OpenRouter request interval for
      `openrouter_free_daily_requests:requests` with `remaining_pct=99.8` and
      `quota_type=requests`.
    - Added opt-in acceptance case `native_openrouter_free_daily_meter_chat`.
      It sends `openrouter/openai/gpt-oss-20b:free` through dev LiteLLM
      `/chat/completions`, hard-gates `session_history`, and hard-gates
      `rate_limit_observations` for the OpenRouter free daily meter. The
      harness also now retries HTTP-request cases on HTTP `status_code`
      `429`/`503`.
    - Acceptance run passed:
      `./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --target dev --cases native_openrouter_free_daily_meter_chat --write-artifact /tmp/openrouter-free-daily-meter.json`
      (`passed=True`, response `gen-1779500672-ccW2fkZQ1ZYothKAn6rX`,
      `rate_limit_observations.match_source=session`,
      `remaining_pct=99.6`).
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'openrouter_free or scalar_tool_names or scalar_file_paths or rate_limit_intervals_materialized_view'`
      passed (`5 passed, 185 deselected`; pre-existing pytest config warning).
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q -k 'http_request_retry_uses_status_code_for_api_error_status or target_profile_can_skip_trace_environment_validation or openrouter_free_cases_validate_daily_rate_limit_observations or rate_limit_observations_validation'`
      passed (`6 passed, 57 deselected`; pre-existing pytest config warning).
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/local-ci/run_anthropic_adapter_acceptance.py`
      passed.
    - `git diff --check` over the touched code, config, tests, and docs passed.
  - Closure:
    - User clarified that the exact `deepseek/deepseek-v4-flash:free` route is
      not required for this item; the requirement was to prove that OpenRouter
      `:free` traffic populates `public.rate_limit_observations` properly. The
      direct `openrouter/openai/gpt-oss-20b:free` live smoke and
      `native_openrouter_free_daily_meter_chat` hard gate satisfy that
      requirement. Any prod promotion can be tracked separately if needed.

- D1-136 Grok Build client_version backfill for 2026-05-15 through 2026-05-18
  - Goal: populate missing `client_version` for historical xAI/Grok Build
    rows in `public.session_history` using the known Grok Build release windows.
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
  - Changed paths:
    - `.analysis/completed.md`
  - Evidence:
    - Pre-repair dry-run found `2395` targeted rows from UTC window
      `2026-05-15 <= created_at < 2026-05-17`, all with null
      `client_version`, and `18760` targeted rows from UTC window
      `2026-05-17 <= created_at < 2026-05-19`, all with null
      `client_version`.
    - Applied `client_version=0.1.210` to the `2026-05-15` through
      `2026-05-16` Grok Build rows and `client_version=0.1.211` to the
      `2026-05-17` through `2026-05-18` Grok Build rows, restricted to
      `provider=xai` and `model=grok-build`.
    - Updated the row metadata with `client_version`,
      `client_version_backfill_source=grok_build_client_version_backfill_2026_05_23`,
      `client_version_backfilled_at`, and
      `client_version_backfilled_from=created_at_utc_date_window`.
    - Post-repair validation by UTC date: `2026-05-15|0.1.210|3`,
      `2026-05-16|0.1.210|2392`, `2026-05-17|0.1.211|747`, and
      `2026-05-18|0.1.211|18013`; targeted null `client_version` count is
      now `0`.
    - Backfill marker aggregate: `0.1.210` has `2395` marked rows from
      `2026-05-15 21:25:11.336773+00` through
      `2026-05-16 16:01:58.220917+00`; `0.1.211` has `18760` marked rows
      from `2026-05-17 14:27:51.724597+00` through
      `2026-05-18 20:30:01.568973+00`.
  - Not run:
    - No code tests were run because this was a database-only repair.

- D1-135 session_history repository identity repair from project roots and Claude transcripts
  - Goal: normalize malformed `public.session_history.repository` values to
    real `/home/zepfu/projects/*` repository names where evidence is
    deterministic, preserve `zepfu`, map `zepfu/litellm` to `litellm`, and
    null unresolved labels instead of keeping worktree/session/file names as
    repositories.
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
  - Changed paths:
    - `.analysis/completed.md`
  - Evidence:
    - Dry-run source set contained `144,292` non-memory repository mismatches;
      `122` `zepfu` rows were intentionally preserved.
    - Applied `144,170` updates with run marker
      `repository_identity_repair_run_id=repository_identity_repair_2026_05_23`.
    - Repair source counts: `project_path_root=140708`, `row_tenant_id=2691`,
      `same_session_single_real_repo=411`, `unresolved_repository_null=310`,
      `explicit_repository_map=37`, and
      `claude_transcript_session_cwd=13`.
    - Claude transcript evidence scan read `8387` files and `1489484` JSONL
      lines under `/home/zepfu/.claude/projects`, with `6570` relevant cwd
      hits for the unresolved session set and `0` JSON parse errors.
    - Explicit policy results: `zepfu/litellm` now has `0` rows and the `37`
      affected rows are marked as `repository=litellm`; `zepfu` remains as the
      only non-memory mismatch with `122` rows across `11` sessions; `...`,
      `memories`, `rollout-2026-05-07T07-56-05-019e024b-89dc-74a0-9`,
      `rollout-20`, `ze`, and the unresolved `project-a` rows were nulled.
      Transcript-derived `project-a`/`myapp` rows mapped to `aawm` where the
      matching Claude `sessionId` had `cwd=/home/zepfu/projects/aawm`.
    - Post-repair repository classification:
      `blank=321851`, `ignored_memory_suffix=1279`, `real_project_match=298250`,
      and `mismatch=122` where the only remaining mismatch is `zepfu`.
    - Post-repair run-marker aggregate:
      `144170` marked rows, `310` with `repository IS NULL`, `5135` with
      `repository=litellm`, `0` with `repository=zepfu`, and `0` with
      `repository=zepfu/litellm` inside the repair run.
  - Not run:
    - No code tests were run because this was a database-only repair; tenant IDs
      were preserved unless they were blank, repository-sourced, or the same bad
      repository label.

## 2026-05-22

- D1-133 Codex auto-agent Google lane-wide rate/quota cooldown
  - Goal: stop back-to-back `aawm-codex-agent-auto` medium dispatches from
    immediately walking across multiple Gemini candidates when the first Google
    Code Assist failure is account/lane-wide quota exhaustion rather than
    model-specific capacity.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Prod logs for `aawm-litellm` showed the incident sequence:
      `2026-05-22T18:42:26.874Z` Google adapter `429`
      `RATE_LIMIT_EXCEEDED` on
      `google_code_assist/gemini-3.1-flash-lite-preview`, followed by
      `2026-05-22T18:42:56.027Z` `QUOTA_EXHAUSTED` on
      `gemini-3-flash-preview`, and `2026-05-22T18:43:01.322Z`
      `MODEL_CAPACITY_EXHAUSTED` on `gemini-3.1-pro-preview`.
    - Root cause: alias probe calls intentionally set
      `google_adapter_max_retries=0`, so Google adapter retry exhaustion can be
      raised before the adapter sets its own Google lane cooldown. The alias
      catch path cooled only the selected candidate, so an account-wide
      Google quota/rate failure could select the next Gemini candidate
      immediately.
    - Added Google lane exhaustion detection for `RATE_LIMIT_EXCEEDED`,
      `QUOTA_EXHAUSTED`, `RESOURCE_EXHAUSTED`, and generic Google `429`s unless
      the error also carries `MODEL_CAPACITY_EXHAUSTED`.
    - Updated the auto-agent alias catch path to cool all Google auto-agent
      candidates on the same lane for lane-wide Google rate/quota exhaustion
      and to record `cooldown_scope=google_lane` in attempt metadata.
      `MODEL_CAPACITY_EXHAUSTED` remains candidate-scoped so another Gemini
      model can still be tried.
    - Added/updated regression coverage proving account-wide Google exhaustion
      skips the remaining Gemini candidates and proceeds to OpenRouter/last
      resort, while Google model-capacity exhaustion only cools the selected
      Gemini model and tries the next Gemini candidate.
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k codex_auto_agent_alias`
      passed (`13 passed, 324 deselected`; pre-existing pytest/backoff
      warnings).
    - Verification: `git diff --check` passed.
  - Not run:
    - No proxy restart/deploy or live post-fix smoke was run in this turn, so
      the fix is local until promoted to the running proxy.

- D1-128 OpenRouter free-model daily request quota local meter
  - Goal: create first-party `rate_limit_observations` for OpenRouter
    `:free` model daily request usage because OpenRouter does not expose the
    current free-model request counter in `/api/v1/key`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/session-history-rate-limit-usage-materialized.sql`
    - `.analysis/session-history-rate-limit-usage-optimized.sql`
    - `.analysis/completed.md`
  - Evidence:
    - Added a shared OpenRouter free daily request meter sourced from
      `public.session_history` rows where `provider='openrouter'` and the
      model ends with `:free`; the meter writes
      `rate_limit_observations` as `provider=openrouter`,
      `client=openrouter`, `quota_key=openrouter_free_daily_requests:requests`,
      `quota_period=daily`, `quota_type=requests`, and
      `source=openrouter_free_daily_local_meter`.
    - Anchored the default reset to next UTC midnight and added
      `AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT` as an optional override, with
      the default set to the paid-account free-model allowance of `1000`
      requests/day.
    - Added a hybrid exhaustion signal: OpenRouter free-model rate-limit
      failures now emit a `0%` remaining snapshot, preferring provider
      retry/reset hints when present and falling back to the UTC daily window.
    - Updated `public.rate_limit_intervals` SQL and the reusable `.analysis`
      SQL artifacts so OpenRouter request-quota rows flow into the quota
      interval read model.
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'openrouter_free or rate_limit_intervals_materialized_view_includes_xai_requests'`
      passed (`3 passed, 182 deselected`; pre-existing pytest config warning).
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed (`185 passed`; pre-existing pytest config warning).
    - Verification:
      `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - Verification:
      `./.venv/bin/ruff check --ignore PLR0915 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - Verification: `git diff --check` passed.
  - Not run:
    - No proxy restart or live database smoke was run for this local
      implementation.

- D1-127 OpenRouter DeepSeek V4 Flash free route and Codex auto-agent fallback
  - Goal: make `deepseek/deepseek-v4-flash:free` available as an
    OpenRouter-backed LiteLLM route, allow it through the Anthropic
    OpenRouter Responses adapter, and use it in `aawm-codex-agent-auto`
    before the final `gpt-5.4-mini` fallback.
  - Changed paths:
    - `litellm-dev-config.yaml`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/completed.md`
  - Evidence:
    - Added explicit dev config model `deepseek/deepseek-v4-flash:free`
      targeting `openrouter/deepseek/deepseek-v4-flash:free` with
      `AAWM_OPENROUTER_API_KEY`.
    - Added paid-equivalent pricing for the raw and `openrouter/...`
      OpenRouter model keys at `$0.14/M` input, `$0.28/M` output, and
      `$0.07/M` cache read,
      including both `cache_read_input_token_cost` and
      `input_cost_per_token_cache_hit`.
    - Added `deepseek/deepseek-v4-flash:free` to the Anthropic
      OpenRouter Responses adapter allowlist for both raw and
      `openrouter/...` request spellings.
    - Added a dedicated `openrouter` candidate to `aawm-codex-agent-auto`
      after the Google Code Assist candidates and before last-resort
      `gpt-5.4-mini`, routed to OpenRouter `/v1/responses` with
      OpenRouter credentials and route family `codex_openrouter_responses`.
    - Verification: both `model_prices_and_context_window.json` and
      `litellm/bundled_model_prices_and_context_window_fallback.json`
      parsed successfully with `json.load`.
    - Verification: with `LITELLM_LOCAL_MODEL_COST_MAP=True`,
      `litellm.cost_per_token` returned `(0.14, 0.28)` for one million
      input/output tokens for both raw and `openrouter/...` model spellings,
      and `(0.07, 0.0)` when the one million input tokens were cache reads.
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k "openrouter_models_to_responses or codex_auto_agent_alias"`
      passed (`25 passed, 302 deselected`; pre-existing pytest/backoff warnings).
    - Verification: `git diff --check` passed.

## 2026-05-21

- D1-126 Anthropic-adapter session-history provider attribution repair
  - Goal: stop Anthropic-shaped adapter ingress rows from making non-Anthropic
    models look like they were served by Anthropic in `session_history`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/completed.md`
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
    - Exact database `aawm_tristore.public.session_history_tool_activity`.
  - Evidence:
    - Updated provider normalization to let explicit
      `anthropic-adapter-target:*` tags, adapter model tags, route family, and
      non-Claude model prefixes override the Anthropic ingress provider when
      the row came through `/anthropic/v1/messages`.
    - Repaired `1,406` historical `session_history` rows with marker
      `session_history_provider_repair_source=anthropic_adapter_target_2026_05_21`:
      `857` to `provider=gemini`, `416` to `provider=openai`, and `133` to
      `provider=openrouter`.
    - Repaired `499` related `session_history_tool_activity` rows from their
      corrected parent calls: `446` to `provider=gemini` and `53` to
      `provider=openrouter`.
    - Left `16` `session_history` rows and `9` tool-activity rows as
      `provider=anthropic, model=unknown` because their tags were native
      Claude/Anthropic (`route:anthropic_messages`, Claude thinking/billing
      tags) and had no adapter target. Left `104` local CLI synthetic rows
      unchanged.
    - Representative post-repair counts in exact database `aawm_tristore`:
      `gemini-3-flash-preview` now has `3391` rows under `provider=gemini`;
      `gemini-3.1-flash-lite-preview` has `3235` rows under
      `provider=gemini`; `gpt-5.4` has `3481` rows under `provider=openai`;
      `gpt-5.3-codex-spark` has `12704` rows under `provider=openai`;
      `openrouter/free` has `85` rows under `provider=openrouter`; and
      `nvidia/nemotron-3-super-120b-a12b:free` has `84` rows under
      `provider=openrouter`.
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'adapter_target_over_anthropic or langfuse_uses_adapter_target'`
      passed (`4 passed, 179 deselected`; one pre-existing pytest config
      warning).
    - Verification:
      `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed (`183 passed`; one pre-existing pytest config warning).
    - Verification:
      `./.venv/bin/ruff check --ignore PLR0915 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - Verification: `git diff --check` passed.

- D1-125 OpenRouter provider-health total-latency fallback
  - Goal: stop provider-health cells from rendering OpenRouter traffic buckets
    as upstream-latency misses when the read model has passive total request
    latency but lacks the exact `llm_upstream_elapsed_ms` split.
  - Changed paths:
    - `/home/zepfu/projects/dashboard-shell/src/features/dashboard/components/phosphor-dashboard.tsx`
    - `/home/zepfu/projects/dashboard-shell/src/features/dashboard/components/phosphor-dashboard.test.tsx`
    - `.analysis/completed.md`
  - Database target:
    - Exact database `aawm_tristore.public.session_history` and
      `aawm_tristore.public.provider_latency_health_5m`.
  - Evidence:
    - Live OpenRouter aggregate for the last 14 days showed `61429`
      `session_history` rows, `61420` rows with missing
      `llm_upstream_elapsed_ms`, `61429` rows with
      `total_server_elapsed_ms`, and `61420` rows where upstream was missing
      but total latency was present.
    - Sample read-model buckets included
      `2026-05-21 12:20:00+00` for
      `openrouter/qwen/qwen3-embedding-8b` with `requests=18`,
      `upstream_p95_ms=NULL`, `total_p95_ms=6130.985099999987`,
      `missing_upstream_latency=18`, zero provider/error counters, and healthy
      OpenRouter probes (`status_probe_count=4`,
      `status_probe_success_pct=100.00`).
    - Updated the dashboard health-cell classifier to prefer
      `upstream_p95_ms` but fall back to `total_p95_ms` for passive latency
      coloring. Buckets with missing upstream samples and no passive latency
      still render as `miss`.
    - Verification:
      `pnpm vitest run src/features/dashboard/components/phosphor-dashboard.test.tsx`
      passed in `/home/zepfu/projects/dashboard-shell` (`1` file,
      `58` tests).

## 2026-05-20

- D1-124 OpenRouter Ling prompt-cache telemetry check
  - Goal: verify what has to be enabled for
    `openrouter/inclusionai/ling-2.6-flash` prompt caching and make
    session-history cache-attempt classification match OpenRouter's actual
    usage payload shape.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - OpenRouter model metadata for
      `inclusionai/ling-2.6-flash-20260421` shows endpoint provider `Novita`,
      `input_cache_read` pricing, `supports_implicit_caching=false`, and no
      advertised `cache_control` supported parameter.
    - Two live LiteLLM/OpenRouter probes against dev `:4001` returned
      `prompt_tokens_details.cached_tokens=0` and
      `cache_write_tokens=0`: session
      `smoke-ling-cache-dev-2026-05-20` rows `604645`/`604652`, and session
      `smoke-ling-auto-cache-dev-2026-05-20` rows
      `605019`/`605026`/`605037`.
    - Direct OpenRouter probes with an explicit content-block
      `cache_control` marker returned HTTP `200` via Novita but still reported
      `cached_tokens=0` and `cache_write_tokens=0`.
    - Updated provider-cache detection to treat OpenRouter's
      `usage.prompt_tokens_details.cached_tokens` path as OpenAI-style cache
      telemetry and to avoid downgrading an already detected cache miss during
      row normalization.
    - Verification:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "openrouter_provider_cache_miss_from_prompt_tokens_details or openai_provider_cache_miss_from_zero_cached_tokens"`
      passed (`2 passed, 177 deselected`; one pre-existing pytest config
      warning), and `git diff --check` passed.

- D1-122 dev OpenRouter Ling 2.6 Flash paid route
  - Goal: add the non-free OpenRouter
    `inclusionai/ling-2.6-flash` model to the LiteLLM dev server and verify it
    can be used through the dev proxy with normal session-history attribution.
  - Changed paths:
    - `litellm-dev-config.yaml`
    - `model_prices_and_context_window.json`
    - `.analysis/completed.md`
  - Runtime target:
    - `litellm-dev` on `:4001`.
  - Evidence:
    - Added explicit dev route
      `model_name=openrouter/inclusionai/ling-2.6-flash` with
      `api_base=https://openrouter.ai/api/v1` and
      `api_key=os.environ/AAWM_OPENROUTER_API_KEY`.
    - Added non-free model catalog entries for both
      `inclusionai/ling-2.6-flash` and
      `openrouter/inclusionai/ling-2.6-flash` with non-discount OpenRouter
      pricing basis: `$0.10/M` input, `$0.30/M` output, and `$0.02/M`
      cache-read.
    - Validation:
      `./.venv/bin/python -m json.tool model_prices_and_context_window.json`
      passed; YAML parse showed `ling_route True`.
    - Restarted only the dev container:
      `docker compose -f docker-compose.dev.yml restart litellm-dev`.
    - Runtime `/v1/models` on `http://127.0.0.1:4001` listed
      `openrouter/inclusionai/ling-2.6-flash`, and `/model/info` showed
      input `1e-07`, cache-read `2e-08`, output `3e-07`, max input `262144`.
    - Tiny live smoke through `:4001` returned HTTP `200`, model
      `openrouter/inclusionai/ling-2.6-flash`, content `ok`, usage cost
      `3.1e-07`, and OpenRouter upstream provider `Novita`.
    - Exact database `aawm_tristore.public.session_history` row `596136`
      verified the smoke with session
      `smoke-ling-2-6-dev-2026-05-20`, `provider=openrouter`,
      `model=openrouter/inclusionai/ling-2.6-flash`, `call_type=acompletion`,
      `input_tokens=25`, `output_tokens=2`, `response_cost_usd=3.1e-07`,
      `usage_openrouter_cost=0.00000031`, and `litellm_environment=dev`.

- D1-121 session_history agent-style repository cleanup follow-up
  - Goal: repair newly observed `public.session_history.repository` rows where
    transient agent ids or adjacent agent/test labels reappeared as repository
    identity after the earlier historical repository repair.
  - Changed paths:
    - `.analysis/completed.md`
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
  - Evidence:
    - Pre-repair query found `47` rows where `repository` matched
      `^agent-[A-Za-z0-9_-]+`, created from
      `2026-05-15 22:55:15.378144+00` through
      `2026-05-19 15:40:39.890181+00`.
    - Same-session evidence resolved session
      `7843a9af-7f8c-40f8-b0c4-6f2eaf26ae8a` to
      `dashboard-shell` and session
      `f7d968dc-544a-45a3-90e4-e4d3b98a7f2c` to `aawm`.
    - Applied bounded repair for those `47` rows: `17` to
      `dashboard-shell/dashboard-shell` and `30` to `aawm/aawm`, preserving
      previous repository/tenant in metadata and stamping
      `repository_identity_repair_source=manual_same_session_repository_repair_2026_05_20`.
    - Follow-up data scan found adjacent same-session repository noise:
      `token-layer.test.ts` (`4` rows) and `wt-researcher-xxx` (`3` rows).
      These were repaired to the same session repositories.
    - Final field verification returned `0` for current `agent-*`
      repository values, `0` for current `agent-*` tenant values, and `0` for
      the adjacent `token-layer.test.ts` / `wt-researcher-xxx` repository or
      tenant values.
    - Bounded metadata-marker verification returned `54` repaired rows total:
      `aawm/aawm` (`33`, id range `374602..374976`) and
      `dashboard-shell/dashboard-shell` (`21`, id range `393651..429014`).

- D1-120 OpenRouter Elephant Alpha / Ling 2.6 Flash session-history repair
  - Goal: normalize historical `openrouter/elephant-alpha` stealth-release
    session-history rows after OpenRouter revealed the model as
    `inclusionai/ling-2.6-flash`, and reprice them with the requested
    non-discount OpenRouter rates.
  - Changed paths:
    - `.analysis/completed.md`
  - Database target:
    - Exact database `aawm_tristore.public.session_history`.
  - Applied repair:
    - Matched rows with historical model values `openrouter/elephant-alpha`
      and `inclusionai/ling-2.6-flash:free`.
    - Updated `provider='openrouter'`.
    - Updated `model='inclusionai/ling-2.6-flash:stealth'`.
    - Recomputed `response_cost_usd` with input `$0.10/M`, output `$0.30/M`,
      and cache-read `$0.02/M`:
      `(input_tokens - cache_read_input_tokens) * 1e-7
      + cache_read_input_tokens * 2e-8 + output_tokens * 3e-7`.
    - Recomputed `provider_cache_miss_cost_usd` for miss rows with token
      counts as `provider_cache_miss_token_count * 8e-8`, the delta between
      input and cache-read pricing.
    - Added metadata markers including previous model/provider, reveal target,
      `response_cost_source=openrouter_stealth_ling_2_6_flash_reprice_2026_05_20`,
      and
      `usage_response_cost_source=openrouter_non_discount_pricing_2026_05_20`.
  - Evidence:
    - Pre-repair candidates: `300` rows across old groups:
      `inclusionai/ling-2.6-flash:free|openrouter` (`138`),
      `inclusionai/ling-2.6-flash:free|anthropic` (`41`),
      `openrouter/elephant-alpha|openrouter` (`75`), and
      `openrouter/elephant-alpha|anthropic` (`46`).
    - Apply statement returned `300` updated rows, id range `49843` to
      `113865`, created range `2026-04-19 01:52:19.029682+00` to
      `2026-04-29 00:01:51.699099+00`, aggregate
      `response_cost_usd=1.18763964`, and aggregate
      `provider_cache_miss_cost_usd=0.12005928`.
    - Post-repair grouped query returned only
      `inclusionai/ling-2.6-flash:stealth|openrouter|300`, with aggregate
      `input_tokens=11858749`, `output_tokens=5883`,
      `cache_read_input_tokens=2`, `response_cost_usd=1.18763964`, and
      `provider_cache_miss_cost_usd=0.12005928`.
    - Old-model verification returned `0` rows for
      `openrouter/elephant-alpha`, `inclusionai/ling-2.6-flash:free`, or
      misspelled `elphant` model strings.

## 2026-05-19

- D1-119 local CLI session_history backfill importer and apply
  - Goal: build and apply a dry-run-first process to reconstruct aggregate
    `public.session_history` and `public.session_history_tool_activity` rows
    from local Claude, Codex, Gemini, and Grok CLI transcript history, while
    skipping any day that already has `session_history` records.
  - Changed paths:
    - `scripts/backfill_local_cli_session_history.py`
    - `tests/test_scripts/test_backfill_local_cli_session_history.py`
    - `.analysis/completed.md`
  - Evidence:
    - `./.venv/bin/python -m pytest tests/test_scripts/test_backfill_local_cli_session_history.py -q`
      passed (`8 passed`, one pre-existing pytest config warning).
    - `./.venv/bin/python -m py_compile scripts/backfill_local_cli_session_history.py tests/test_scripts/test_backfill_local_cli_session_history.py`
      passed.
    - `./.venv/bin/ruff check --ignore PLR0915,T201 scripts/backfill_local_cli_session_history.py tests/test_scripts/test_backfill_local_cli_session_history.py`
      passed.
    - Dry run against exact database `aawm_tristore`:
      `./.venv/bin/python scripts/backfill_local_cli_session_history.py`.
    - Pre-apply marker check against exact database `aawm_tristore` found
      `session_history_marker_count=0` and
      `tool_activity_marker_count=0`.
    - Apply command:
      `./.venv/bin/python scripts/backfill_local_cli_session_history.py --apply`.
    - Existing `session_history` day buckets were
      `2026-04-10..2026-05-19` in `America/New_York`, so the importer counted
      only candidate transcript days `2026-02-11..2026-04-09`.
    - Follow-up correction: Claude transcripts can split one provider response
      across multiple assistant JSONL rows with the same `message.id` /
      `requestId`, each repeating the same usage block. The importer now
      coalesces those rows before counting tokens while preserving all tool
      blocks.
    - Corrected dry run would add `141843` `session_history` rows and `177176`
      `session_history_tool_activity` rows, with aggregate tokens
      `input=62405976`, `output=11396985`,
      `cache_read=10117624837`, `cache_creation=281618910`,
      and `reasoning=153971`.
    - Cost/cache-miss totals now included in the dry run:
      `rows_with_response_cost=141739`, `response_cost_usd=4994.871018`,
      `cache_miss_rows=1548`, `cache_miss_tokens=52888715`, and
      `cache_miss_cost_usd=226.290344`.
    - Derived activity totals were `file_reads=71116`,
      `file_modifications=23995`, `git_commits=2276`, and
      `git_pushes=1555`.
    - Client split: `claude-code` would add `141104` records / `175598` tool
      rows with `response_cost_usd=4827.103354`; `codex_tui` would add `739`
      records / `1578` tool rows with `response_cost_usd=167.767665`. Gemini
      and Grok rows were discovered but skipped because all their transcript
      days already have `session_history` records.
    - Post-apply verification against exact database `aawm_tristore` found
      `141843` `session_history` rows with
      `metadata->>'source_import' = 'local_cli_history_2026_05_19'`, date range
      `2026-02-11..2026-04-09`, aggregate tokens matching the dry run, rounded
      `response_cost_usd=4994.871018`, `1548` cache-miss rows,
      `cache_miss_tokens=52888715`, and rounded
      `cache_miss_cost_usd=226.290344`.
    - Tool activity verification found `177176` joined
      `session_history_tool_activity` rows and `177176` direct tool rows with
      `metadata->>'source_import' = 'local_cli_history_2026_05_19'`.
    - Identity verification for the inserted session rows found zero gaps for
      provider, model, model_group, `client_user_agent='backfill'`, or blank
      `client_version`.
  - Notes:
    - Database writes have been applied to exact database `aawm_tristore`. The
      script still defaults to dry-run and requires `--apply` to populate rows.
    - Apply mode preserves historical `created_at` from the source timestamp
      rather than using insert-time `NOW()`.
    - Backfilled rows use `client_user_agent='backfill'`, blank
      `client_version`, model-derived provider/model/model_group, and
      cache-write-only rows populate `provider_cache_miss_reason`,
      `provider_cache_miss_token_count`, and `provider_cache_miss_cost_usd`.
    - Tool activity rows also carry the backfill marker and client identity in
      `metadata` so they can be queried directly or through the parent session
      row.
    - Tool arguments are redacted/truncated and raw transcript text is not
      copied into `session_history`.

- D1-118 Langfuse trace-quality score visibility repair
  - Goal: explain and repair why Langfuse showed no available scores after
    deterministic trace-quality scoring and invalid-tool-call tracking were
    added.
  - Changed paths:
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `.analysis/completed.md`
  - Evidence:
    - Live Langfuse trace API returned recent traces with `scores=[]`, and
      `tags=["llm-judge-candidate"]` returned `totalItems=0`, confirming the
      scoped Owl Alpha judge had no selected traces to process.
    - Exact database `aawm_tristore.public.session_history` contained `424`
      rows with `invalid_tool_call_count > 0`, proving the live callback path
      was collecting the signal.
    - Fixed the deterministic scorer so `SessionCandidate` carries
      `invalid_tool_call_count` from `session_history` and scoring uses the max
      of payload-derived invalid-tool errors and the live session-history count.
    - Verification passed:
      `./.venv/bin/python -m py_compile scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py`;
      `./.venv/bin/python -m pytest -p no:rerunfailures tests/test_scripts/test_score_agent_trace_quality.py -q`
      (`13 passed`, one pre-existing pytest config warning).
    - Dry-run on trace `b8824e11-a2fe-40c7-887c-ec9611ba8d1f` now resolves
      MinIO payloads and emits `invalid_tool_call_error_count=2`,
      `reasons=["invalid_tool_call_error_seen"]`.
    - Applied that single trace only with
      `scripts/score_agent_trace_quality.py --trace-id b8824e11-a2fe-40c7-887c-ec9611ba8d1f --include-passing --limit 1 --apply`;
      the run reported `score_write_count=5`.
    - Langfuse trace API for
      `b8824e11-a2fe-40c7-887c-ec9611ba8d1f` now returns five API scores:
      `aawm.agent.trace_quality`, `aawm.agent.empty_completion_failure`,
      `aawm.agent.large_tool_result_payload_risk`,
      `aawm.agent.destructive_checkout_after_work`, and
      `aawm.agent.invalid_tool_call_error`.
  - Notes:
    - Historical/bulk score population has not been run yet.
    - Langfuse tags are still separate from scores; this repair writes score
      objects, not trace tags.

- D1-115/D1-114 aawm58 prod cutover
  - Goal: deploy the prepared `aawm.58` prod container and validate that the
    release line is live on `:4000`.
  - Changed paths:
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `/home/zepfu/projects/aawm-infrastructure/.env`
  - Evidence:
    - `docker compose -f docker-compose.litellm.yml up -d litellm` recreated
      prod from the prepared `aawm-litellm:latest` image.
    - Running prod container `ad7bfc97158c` is healthy on `:4000`.
    - Readiness returned `litellm_version=1.82.3+aawm.58` and listed
      `AawmAgentIdentity` in success callbacks.
    - Runtime package inspection returned `litellm=1.82.3+aawm.58`,
      `aawm-litellm-callbacks=0.0.34`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Callback import inspection returned
      `_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL=claude-auto-review` and
      `_CLAUDE_AUTO_REVIEW_AGENT_NAME=auto-reviewer`.
    - The first default prod harness at
      `/tmp/litellm-prod-harness-aawm58.json` failed OpenRouter-backed cases
      due to upstream `401 User not found`; fingerprint comparison showed
      infrastructure `.env` still had the old OpenRouter key.
    - Refreshed `AAWM_OPENROUTER_API_KEY` and `OPENROUTER_API_KEY` in
      `/home/zepfu/projects/aawm-infrastructure/.env` from the repo `.env`
      without printing the secret, then force-recreated `aawm-litellm`.
    - Post-refresh container env fingerprint matched the repo `.env` key, and
      direct `openrouter/owl-alpha` smoke through `:4000` returned HTTP `200`.
    - Focused recheck
      `/tmp/litellm-prod-focused-aawm58-recheck.json` passed
      `claude_adapter_spark`, `claude_adapter_openrouter_free`, and
      `claude_adapter_nemotron_super`.
    - Remaining `claude_adapter_codex_tool_activity` failure was limited to
      Claude CLI final JSON typoing `/home/zepfu/projects/litellm` as
      `/home/zepvu/projects/litellm`; persisted tool activity still recorded
      the expected `exec_command` tool call with `cmd=pwd`. Per operator
      direction, Claude Code CLI execution of Codex agents is disregarded for
      this deployment gate.
    - Infrastructure commit `62bddf0` is pushed to both `origin/main` and
      `origin/develop`.

- D1-115 release prep for Claude auto-review telemetry attribution
  - Goal: check in/push the D1-115 code, publish release artifacts, and prep
    the prod image pin without restarting `aawm-litellm`.
  - Changed paths:
    - `COMPLETED.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - LiteLLM implementation commit `d7301ba7ff`, autobump commit
      `c12e5e38b7`, and release commit `3ca2c670f0` are pushed.
      `origin/main` and `origin/develop` both point at `3ca2c670f0`.
    - GitHub Actions run `26117919762` succeeded and published
      `ghcr.io/zepfu/litellm:1.82.3-aawm.58`; GitHub Release
      `v1.82.3-aawm.58` is public.
    - Manual overlay recovery published `cb-v0.0.34` with
      `aawm_litellm_callbacks-0.0.34-py3-none-any.whl` and `cfg-v0.0.12`
      with `litellm-model-config-0.0.12.tar.gz`; existing `cp-v0.0.7` and
      `h-v0.0.30` remain current.
    - Infrastructure commit `62bddf0` on `origin/develop` pins the prod base
      image to `ghcr.io/zepfu/litellm:1.82.3-aawm.58`.
    - `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      in `/home/zepfu/projects/aawm-infrastructure` completed and produced
      local `aawm-litellm:latest` image id `ce6e947066e4`.
    - Built-image inspection returned `litellm=1.82.3+aawm.58`,
      `aawm-litellm-callbacks=0.0.34`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Built callback inspection returned
      `_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL=claude-auto-review` and
      `_CLAUDE_AUTO_REVIEW_AGENT_NAME=auto-reviewer`; bundled model config
      includes `claude-auto-review`.
    - No prod restart/cutover was run. Running `aawm-litellm` still reports
      `litellm=1.82.3+aawm.55`, `aawm-litellm-callbacks=0.0.32`, and
      `aawm-litellm-control-plane=0.0.7`.

- D1-116 provider latency-health quota decoupling
  - Goal: make `public.provider_latency_health_5m` a latency/error/probe read
    model only, leaving quota state in `public.rate_limit_intervals` and the
    quota/report datasets.
  - Changed paths:
    - `.analysis/provider-latency-health-5m-materialized-view.sql`
    - `.analysis/provider-latency-health-5m.sql`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Removed the `quota_state` CTE, `public.rate_limit_intervals` read, quota
      union branch, and quota join from
      `.analysis/provider-latency-health-5m-materialized-view.sql`.
    - Kept deprecated compatibility columns in the materialized view as typed
      `NULL` fields:
      `min_remaining_pct`, `max_remaining_pct`, `next_expected_reset_at`, and
      `quota_keys`, because `dashboard-shell` still selects them.
    - Removed those quota fields from
      `.analysis/provider-latency-health-5m.sql`, so the standalone latency
      query now returns latency, provider-error, and active-probe fields only.
    - Applied the rebuild against exact dev database `aawm_tristore`:
      `psql -v ON_ERROR_STOP=1 -f .analysis/provider-latency-health-5m-materialized-view.sql postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      completed with `SELECT 14316`, `CREATE INDEX` x3, and `ANALYZE`.
    - Live dev verification:
      `row_count=14316`, `quota_non_null_rows=0`,
      `view_mentions_rate_limit_intervals=false`, and
      `view_mentions_quota_state=false`.
    - `pg_rewrite` dependency verification for
      `public.provider_latency_health_5m` now returns only
      `public.provider_error_observations`,
      `public.provider_status_observations`, and `public.session_history`.
    - Verified the compatibility schema still exposes the four deprecated
      columns with the previous types, while the backing definition no longer
      reads quota data.
    - Verified concurrent refresh still works:
      `REFRESH MATERIALIZED VIEW CONCURRENTLY public.provider_latency_health_5m; ANALYZE public.provider_latency_health_5m;`
      returned `refreshed|14317`.
    - Recent buckets after rebuild included
      `2026-05-19 18:20:00+00|10|52|28|0`, proving the rebuilt view continues
      to materialize request, probe, and error counts.
    - Read-only dashboard-shell consumer check found follow-up work is still
      needed: `server/report-service.mjs` selects and normalizes the deprecated
      columns, `usage-report.ts` types them, and
      `use-anomaly-detection.ts` uses `next_expected_reset_at` for quota reset
      detection. Added D1-117 to `.analysis/todo.md` for that external
      contract cleanup.

- D1-115 Claude auto-review session-history attribution
  - Goal: build out
    `.analysis/claude-auto-review-session-history-attribution-plan-2026-05-19.md`
    so Claude Code Bash/tool permission classifier traffic is reported as
    `claude-auto-review` in telemetry without changing upstream routing or the
    model used for live cost calculation.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/integrations/langfuse/langfuse.py`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `scripts/backfill_claude_auto_review_session_history.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/logging_callback_tests/test_langfuse_unit_tests.py`
    - `tests/test_scripts/test_backfill_claude_auto_review_session_history.py`
    - `.analysis/claude-auto-review-session-history-attribution-plan-2026-05-19.md`
    - `.analysis/completed.md`
  - Evidence:
    - Success-path callback logic now detects Claude permission-check metadata,
      preserves the already-computed `response_cost_usd`, then rewrites only
      persisted telemetry identity to `model=claude-auto-review`,
      `agent_name=auto-reviewer`, and
      `metadata.trace_name=claude-code.auto-reviewer`.
    - Langfuse header enrichment now applies
      `langfuse_trace_name=claude-code.auto-reviewer` for detected permission
      checks before metadata is synced to standard logging.
    - Same-session repository inheritance runs in the async persistence path,
      using an in-batch identity cache first and then a bounded 30-minute DB
      lookup against non-permission rows in the same `session_id`; ephemeral
      values such as `agent-a3ee0f55d7cda22ec` are left null when no parent
      project identity is found.
    - Failure observations now use `model=claude-auto-review` when permission
      markers are present and preserve `source_model`, trace name, tags, and
      inherited repository/tenant identity in `metadata`.
    - Offline/reporting compatibility now maps `claude-auto-review` to the
      same bundled cost/context/capability fields as `claude-opus-4-7`, while
      live session-history cost remains calculated before aliasing.
    - Added dry-run/apply backfill script
      `scripts/backfill_claude_auto_review_session_history.py`; the backfill
      updates only clear Anthropic permission-check rows, preserves existing
      `response_cost_usd`, sets `source_model`/`logical_model`, and inherits
      repository/tenant from same-session non-permission rows when available.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py scripts/backfill_claude_auto_review_session_history.py`
      passed.
    - `./.venv/bin/python -m json.tool model_prices_and_context_window.json`
      and
      `./.venv/bin/python -m json.tool litellm/bundled_model_prices_and_context_window_fallback.json`
      passed.
    - `git diff --check` passed.
    - `./.venv/bin/ruff check scripts/backfill_claude_auto_review_session_history.py tests/test_scripts/test_backfill_claude_auto_review_session_history.py`
      passed.
    - `./.venv/bin/ruff check litellm/integrations/aawm_agent_identity.py scripts/backfill_claude_auto_review_session_history.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_backfill_claude_auto_review_session_history.py --ignore PLR0915`
      passed; a strict run without the ignore still reports pre-existing
      `PLR0915` function-length findings in
      `litellm/integrations/aawm_agent_identity.py`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_backfill_claude_auto_review_session_history.py -q`
      passed (`179 passed, 1 warning`).
    - Dry-run backfill command
      `./.venv/bin/python scripts/backfill_claude_auto_review_session_history.py --since 2026-05-19T00:00:00Z --limit 200`
      completed without writes and reported
      `{"apply": false, "counts": {}, "repaired_rows": 0, "selected_rows": 0}`.
    - Dev container `litellm-dev` was restarted with the patch loaded only on
      port `4001`; `docker inspect litellm-dev --format '{{.State.StartedAt}}'`
      showed `2026-05-19T16:11:34.819191864Z`.
    - `curl -sS http://127.0.0.1:4001/health/readiness` returned
      `{"status":"healthy",...,"success_callbacks":["AawmAgentIdentity",...]}`
      after the restart.
    - Runtime import check inside `litellm-dev` confirmed
      `_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL == "claude-auto-review"`, confirmed
      `/app/scripts/backfill_claude_auto_review_session_history.py` exists, and
      confirmed both `/app/model_prices_and_context_window.json` and
      `/app/litellm/bundled_model_prices_and_context_window_fallback.json`
      resolve `claude-auto-review` with `litellm_provider=anthropic`.
    - `docker exec litellm-dev python -m py_compile /app/litellm/integrations/aawm_agent_identity.py /app/scripts/backfill_claude_auto_review_session_history.py`
      passed.
    - In-container dry-run
      `docker exec litellm-dev python /app/scripts/backfill_claude_auto_review_session_history.py --since 2026-05-19T00:00:00Z --limit 1`
      completed without writes and reported
      `{"apply": false, "counts": {"with_identity": 1061}, "repaired_rows": 1061, "selected_rows": 4919}`;
      `--limit` limits affected sessions, not repaired rows.
    - `docker logs --tail 80 litellm-dev` after the restart showed normal
      startup, initialized `AawmAgentIdentity` success/failure callbacks, and
      healthy readiness requests with no traceback.
    - Live watch of running Claude session
      `7843a9af-7f8c-40f8-b0c4-6f2eaf26ae8a` on dev showed normal
      `session_history` rows for analyst/engineer/orchestrator calls, and
      auto-review permission checks persisted as `model=claude-auto-review`,
      `agent_name=auto-reviewer`, `tenant_id=dashboard-shell`,
      `repository=dashboard-shell`, nonzero `response_cost_usd`, and
      `metadata.trace_name=claude-code.auto-reviewer`.
    - The initial live watch exposed a Langfuse trace-tag propagation gap:
      `session_history.metadata.request_tags` contained
      `claude-agent:auto-reviewer` and `claude-project:dashboard-shell`, but
      Langfuse `claude-code.auto-reviewer` traces had `tags=[]`.
      `LangFuseLogger._get_langfuse_tags()` now merges tags from the live
      metadata `request_tags`/`tags` fallback as well as
      `standard_logging_object.request_tags`; the AAWM identity sync helper now
      also copies metadata `request_tags` into standard logging request tags.
    - The same watch exposed a Claude side-call identity gap where some traces
      carried `trace_user_id=dashboard-shell` but not explicit
      repository/tenant fields. Claude Code traces now use `trace_user_id` as a
      repository/tenant fallback only when the trace name starts with
      `claude-code`, and derive `agent_name` from `claude-code.<agent>` trace
      names when prompt context is absent.
    - Focused verification passed after the fix:
      `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py litellm/integrations/langfuse/langfuse.py`,
      `./.venv/bin/ruff check litellm/integrations/aawm_agent_identity.py litellm/integrations/langfuse/langfuse.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/logging_callback_tests/test_langfuse_unit_tests.py --ignore PLR0915`,
      `./.venv/bin/python -m pytest tests/logging_callback_tests/test_langfuse_unit_tests.py -q -k get_langfuse_tags`,
      and
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'syncs_metadata_request_tags or permission_check or auto_review or claude_trace_user_identity_fallback'`.
    - Full AAWM identity coverage passed:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      reported `178 passed, 1 warning`.
    - Broader Langfuse e2e coverage still has a repeatable unrelated failure:
      `./.venv/bin/python -m pytest tests/logging_callback_tests/test_langfuse_unit_tests.py::test_langfuse_e2e_sync -q`
      failed because the respx mocked Langfuse ingestion route was not called
      after the Langfuse SDK reported `Unexpected error occurred`; the focused
      tag extraction unit test passed.
    - Dev `litellm-dev` was restarted again to load the tag/identity fixes;
      `docker inspect litellm-dev --format '{{.State.StartedAt}}'` showed
      `2026-05-19T17:57:15.362949866Z`, and
      `curl -sS http://127.0.0.1:4001/health/readiness` returned healthy with
      `AawmAgentIdentity` loaded.
    - Post-restart database proof: exact dev database
      `aawm_tristore.public.session_history` row `430251` for the same session
      has `model=claude-auto-review`, `agent_name=auto-reviewer`,
      `tenant_id=dashboard-shell`, `repository=dashboard-shell`,
      `response_cost_usd=0.20431200000000002`,
      `metadata.trace_name=claude-code.auto-reviewer`, and request tags
      including `claude-agent:auto-reviewer`,
      `claude-project:dashboard-shell`, and `claude-permission-check`.
    - Post-restart Langfuse proof: dev trace
      `9ab14665-b807-4f8f-8fde-146dfe01cde0` for session
      `7843a9af-7f8c-40f8-b0c4-6f2eaf26ae8a` is named
      `claude-code.auto-reviewer`, has `userId=dashboard-shell`, and now has
      Langfuse tags including `claude-agent:auto-reviewer`,
      `claude-project:dashboard-shell`, `claude-permission-check`,
      `claude-permission-check:allow`, `provider-cache-hit`, and
      `route:anthropic_messages`.
    - Post-restart invariant check over rows since `2026-05-19 17:57:20+00`
      reported `7` rows, `1` auto-review row, `1` fully valid auto-review row,
      `0` non-auto-review message rows missing required identity/cost fields,
      and `0` `not_attempted` side rows at that checkpoint; a later checkpoint
      reported `19` rows, `2` auto-review rows, `2` fully valid auto-review
      rows, `0` malformed non-auto-review message rows, and `2` expected
      `not_attempted` side rows. Provider error observations for the
      post-restart active session were `0`.
  - Follow-up status:
    - Dev `litellm-dev` has the patch loaded and healthy on `:4001`.
      Historical backfill has been dry-run only and has not been applied.
      Prod has not been restarted or changed.

- D1-114 Codex auto-agent fresh-dispatch affinity fallback
  - Goal: keep in-flight `aawm-codex-agent-auto` provider `429` responses
    terminal so the orchestrator can redispatch, but allow fresh dispatches
    with stale Spark session affinity to continue through the preferred
    candidates and eventually the `gpt-5.4-mini` last-resort fallback.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Recent exact database rows for
      `session_id=019e3630-4456-7820-9098-30fbeb3a5f7a` showed repeated
      `aawm-codex-agent-auto` calls pinned to
      `gpt-5.3-codex-spark` with
      `codex_auto_agent_selection_reason=session_affinity`.
    - The selector now treats an affinity target with active cooldown as
      terminal only when the request has continuation state. For fresh
      dispatches it records `session_affinity_cooldown` in skipped candidates
      and continues selection.
    - The retry handler now re-raises retryable provider exhaustion only for
      continuation/in-flight requests, with status
      `terminal_in_flight_cooldown_set`; fresh dispatches continue to the next
      candidate after setting the cooldown.
    - Regression coverage:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k codex_auto_agent_alias`
      passed (`10 passed, 313 deselected`).
    - Syntax verification:
      `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
      passed.
    - Restarted `litellm-dev`; `/health/readiness` on `:4001` returned
      healthy and runtime inspection inside the container confirmed the
      patched `signaling redispatch` marker was loaded.
    - Live dev smoke:
      `codex exec --profile litellm-dev -m aawm-codex-agent-auto -c model_catalog_json='"/home/zepfu/.codex/aawm-model-catalog.json"' "Reply exactly: auto-agent-fresh-dispatch-smoke-20260519"`
      returned the exact requested text for session
      `019e40ab-1fc4-7410-9803-afad0bcd2ac6`.
    - Exact database `aawm_tristore.public.session_history` row `427943`
      recorded Spark first with `status=cooldown_set`,
      `error_tokens=["usage_limit_reached"]`, and
      `cooldown_seconds=1261.176`, then completed on
      `gemini-3.1-flash-lite-preview` with
      `requested_model_alias=aawm-codex-agent-auto`,
      `codex_auto_agent_selected_provider=google_code_assist`,
      `codex_auto_agent_selected_model=gemini-3.1-flash-lite-preview`, and
      `codex_auto_agent_selection_reason=first_available`.
  - Follow-up status:
    - Prod/default proxy `:4000` is still `aawm.55` and does not have this
      patch loaded. A prod hot-patch restart was not performed because explicit
      production restart authorization was not available. The copied prod
      module was restored from `v1.82.3-aawm.55`, so the running prod container
      remains at the approved image source until a durable release or explicit
      restart/hot-patch is authorized.

- D1-113 session_history historical repository repair/backfill
  - Goal: repair existing `aawm_tristore.public.session_history` rows where
    repository identity was missing or polluted by transient agent ids,
    role-only values, `waveN-*` agent names, or placeholders such as
    `path`/`project`, and stamp historical xAI/Grok Build rows to
    `repository=aawm-tap`.
  - Changed paths:
    - `scripts/repair_session_history_repository_identity.py`
    - `tests/test_scripts/test_repair_session_history_repository_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Before repair, exact database `aawm_tristore` had
      `bad_repo_all=3657`, `bad_anthropic_30d=3647`,
      `grok_not_aawm_tap_all=21155`, and `grok_null_30d=21140`.
    - Dry-run command:
      `AAWM_DB_HOST=127.0.0.1 AAWM_DB_PORT=5434 AAWM_DB_NAME=aawm_tristore AAWM_DB_USER=aawm AAWM_DB_PASSWORD=aawm_dev ./.venv/bin/python scripts/repair_session_history_repository_identity.py --batch-size 1000 --preview-limit 30 --grok-repository aawm-tap`
      reported `candidate_rows=24834`, `repairable_rows=24833`,
      `grok_repository_override=21155`,
      `same_session.session_metadata.repository=2735`,
      `same_session.session_metadata.tenant_id=936`, and
      `row_identity_normalization=7`.
    - Applied command:
      `AAWM_DB_HOST=127.0.0.1 AAWM_DB_PORT=5434 AAWM_DB_NAME=aawm_tristore AAWM_DB_USER=aawm AAWM_DB_PASSWORD=aawm_dev ./.venv/bin/python scripts/repair_session_history_repository_identity.py --apply --batch-size 1000 --preview-limit 30 --grok-repository aawm-tap`
      reported `candidate_rows=24840`, `repairable_rows=24839`, and
      `applied=true`. A bounded follow-up from `--cursor-id 426000` repaired
      `8` rows created during/after the first pass.
    - Post-repair verification against exact database `aawm_tristore` showed
      `bad_repo_all=0`, `bad_metadata_repo_all=0`,
      `bad_anthropic_30d=0`, `grok_not_aawm_tap_all=0`, and
      `grok_null_30d=0`.
    - Sample rows after repair:
      `public.session_history.id=412004` is `provider=xai`,
      `model=grok-build`, `repository=aawm-tap`, `tenant_id=aawm-tap`,
      `repository_identity_repair_source=grok_repository_override`;
      `id=424973` is `provider=anthropic`, `model=claude-opus-4-7`,
      `repository=dashboard-shell`, `tenant_id=dashboard-shell`,
      `repository_identity_previous_repository=agent-ac357ffbc895e51d4`;
      and new rows `id=426446` / `id=426495` were repaired from
      `agent-a3362227da21a7076` to `dashboard-shell`.
    - Today's repair-source distribution:
      `grok_repository_override=21155`,
      `same_session.session_metadata.repository=3514`,
      `same_session.session_metadata.tenant_id=171`, and
      `row_identity_normalization=7`.
    - Verification commands:
      `./.venv/bin/python -m pytest tests/test_scripts/test_repair_session_history_repository_identity.py -q`
      passed (`5 passed`, one pre-existing pytest config warning);
      `./.venv/bin/python -m py_compile scripts/repair_session_history_repository_identity.py`
      passed;
      `./.venv/bin/ruff check scripts/repair_session_history_repository_identity.py tests/test_scripts/test_repair_session_history_repository_identity.py`
      passed;
      `git diff --check -- scripts/repair_session_history_repository_identity.py tests/test_scripts/test_repair_session_history_repository_identity.py`
      passed.
  - Notes:
    - The repair script now uses same-session repository evidence for
      non-Grok rows and only uses the Grok-wide default when explicitly passed
      `--grok-repository`.

- D1-112 session_history repository identity hardening for Grok Build and Anthropic
  - Goal: prevent Grok Build rows from missing repository identity when the
    request carries workspace context, and stop Anthropic/Claude rows from
    accepting transient agent ids or role names as repository values.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Live database probe against exact database `aawm_tristore` showed recent
      Grok Build rows such as `public.session_history.id=412004` with
      `provider=xai`, `model=grok-build`, `client_name=grok-build`,
      `passthrough_route_family=grok_cli_chat_proxy`, but null
      `repository`, `metadata.repository`, `tenant_id`, and `agent_name`.
    - Live database probe showed Anthropic rows such as
      `public.session_history.id=424973` with `repository` and `tenant_id`
      both set to `agent-ac357ffbc895e51d4`, while current Claude child rows
      carry real project identity as `tenant_id=dashboard-shell` and trace tags
      like `claude-project:dashboard-shell`.
    - Follow-up count over the last 30 days found `3643` Anthropic rows with
      rejected repository values; top examples were `path` (`533` rows),
      `agent-a8133478546f2fe90` (`409`), `agent-a7f4f9d729c7c50f4`
      (`363`), `agent-aeb8bfdba2d8db558` (`285`), and `project` (`189`).
      The same window had `21140` xAI/Grok rows with null repository.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'repository or grok'`
      passed (`45 passed, 448 deselected`).
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      passed.
  - Notes:
    - File-level Ruff remains blocked by pre-existing findings in the touched
      large files, including PLR0915, T201, F841, F811, and unused imports
      outside this patch.
    - Historical bad rows still need a separate repair/backfill decision before
      mutating `aawm_tristore`.

## 2026-05-18

- D1-111 xAI/Grok side-channel data export
  - Goal: write the currently available xAI/Grok side-channel datasets into
    `.analysis/xai/<sidechannel name>.md` so later investigation can inspect
    the exact source queries, rows, and payload-shape evidence without
    rediscovering the data sources.
  - Changed paths:
    - `.analysis/xai/README.md`
    - `.analysis/xai/provider_status_observations.md`
    - `.analysis/xai/grok_billing.md`
    - `.analysis/xai/rate_limit_intervals.md`
    - `.analysis/xai/provider_error_observations.md`
    - `.analysis/xai/provider_latency_health_5m.md`
    - `.analysis/xai/grok_generation_request_shape.md`
    - `.analysis/xai/langfuse_grok_trace_spans.md`
    - `.analysis/xai/dynamic_injection.md`
    - `.analysis/completed.md`
  - Evidence:
    - `ls -la .analysis/xai` showed all nine report files present.
    - `sed -n '1,220p' .analysis/xai/README.md` confirmed the inventory
      covers provider status probes, Grok billing quota snapshots, derived
      rate-limit intervals, provider errors, five-minute latency health,
      resolved Grok generation request shape, Langfuse Grok trace spans, and
      `aawm.dynamic_injection` spans.
    - `wc -l .analysis/xai/*.md` showed `548` total report lines.
    - `rg -n "^(##|###|\\|Source query|Source|Generated|Rows|No rows|Observation|Trace|### Sample|### Latest|### Raw|### Exact)" .analysis/xai`
      confirmed each report includes generated timestamps and source-query or
      source-description sections.
  - Notes:
    - The sampled Grok generation request payloads include system prompt
      content, `temperature`, and `top_p`, but no observed max-token fields.
    - `dynamic_injection.md` is not xAI-specific; it is included as the
      available side-channel span for API-driven memory/context injection.

## 2026-05-17

- D1-110 invalid tool-call counts in scorer and session_history
  - Goal: expose invalid tool-call counts in both deterministic trace scoring
    and the `public.session_history` dataset, especially validation errors such
    as `InputValidationError` after a bad tool parameter.
  - Changed paths:
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/completed.md`
  - Evidence:
    - `./.venv/bin/python -m py_compile scripts/score_agent_trace_quality.py litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `./.venv/bin/python -m pytest -p no:rerunfailures tests/test_scripts/test_score_agent_trace_quality.py -q` (`12 passed`)
    - `./.venv/bin/python -m pytest -p no:rerunfailures tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "invalid_tool_call or session_history_db_payload_sanitizes_zero_reported_reasoning or derives_passthrough_latency_breakdown or persist_session_history_record_executes_insert or persist_session_history_records"` (`8 passed, 155 deselected`)
    - `./.venv/bin/ruff check --ignore PLR0915 scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py` (`All checks passed`)
  - Notes:
    - Plain focused `ruff check` is still blocked by existing
      `PLR0915 too many statements` findings in `aawm_agent_identity.py`.
    - The scorer emits `aawm.agent.invalid_tool_call_error`, records
      `invalid_tool_call_error_count`, and keeps the signal non-fatal by itself.
    - `session_history` now has `invalid_tool_call_count INTEGER NOT NULL
      DEFAULT 0`, mirrors the count into
      `metadata.usage_invalid_tool_call_count`, and extracts the value from
      live passthrough request bodies plus Langfuse observation inputs used by
      historical backfill.

- D1-109 deterministic agent trace-quality scorer
  - Goal: turn known agent failure patterns into deterministic scoring that can
    be run over selected historical Langfuse traces and optionally written back
    as Langfuse scores without invoking an LLM judge.
  - Changed paths:
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `.analysis/completed.md`
  - Evidence:
    - Added dry-run-first CLI `scripts/score_agent_trace_quality.py` that
      reads candidates from exact database `aawm_tristore.public.session_history`,
      checks `public.provider_error_observations`, resolves full Langfuse
      generation payloads from MinIO via ClickHouse `blob_storage_file_log`
      with ClickHouse fallback, and emits compact JSON evidence.
    - Deterministic rules currently score `trace_quality=0` for
      `empty_completion_after_large_final_tool_result` and
      `destructive_checkout_after_mutating_tool_use`. Optional `--apply`
      writes stable Langfuse scores:
      `aawm.agent.trace_quality`,
      `aawm.agent.empty_completion_failure`,
      `aawm.agent.large_tool_result_payload_risk`, and
      `aawm.agent.destructive_checkout_after_work`.
    - Unit verification passed:
      `./.venv/bin/python -m py_compile scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py`;
      `./.venv/bin/ruff check scripts/score_agent_trace_quality.py tests/test_scripts/test_score_agent_trace_quality.py`;
      `./.venv/bin/python -m pytest tests/test_scripts/test_score_agent_trace_quality.py -q`
      (`9 passed`, one pre-existing pytest config warning).
    - Live dry-run against principal incident trace
      `3fac24c0-b911-4b9a-b575-82c777a30913` with
      `--source-mode minio --include-passing --limit 10` produced one score
      candidate from exact database `aawm_tristore`, resolved MinIO object
      `aawm-dev/observation/time-19-50-00-003617_77cb7594-fd62-4b91-b3f9-64c5db032177/74ef8b39-d18f-4ac4-a039-6a0d94c0c880.json`,
      and emitted `trace_quality_score=0.0`,
      `empty_completion_failure=true`,
      `final_tool_result_image_base64_max_bytes=195820`,
      `output_tokens=3`, `tool_call_count=0`, and
      `provider_error_present=false`.
    - Grok sanity dry-run
      `./.venv/bin/python scripts/score_agent_trace_quality.py --model grok-build --limit 20`
      scanned 20 recent `grok-build` candidates from exact database
      `aawm_tristore`, resolved their MinIO payloads, and emitted
      `evidence_count=0`, proving OpenAI/Grok-shaped messages no longer become
      false score candidates from parser misses.

- D1-108 Langfuse LLM-as-judge via LiteLLM OpenRouter Owl Alpha
  - Goal: configure Langfuse LLM-as-judge to use the
    `openrouter/owl-alpha` model through the local/prod LiteLLM gateway, with
    Langfuse trace headers on judge calls and execution scoped to selected
    traces instead of all traffic.
  - Changed paths:
    - `/home/zepfu/projects/aawm/docker-compose.yml`
    - `.analysis/completed.md`
  - Evidence:
    - Recreated only `aawm-langfuse-web` and `aawm-langfuse-worker`; both are
      healthy on Langfuse `3.174.1`.
    - Added `ENCRYPTION_KEY` and
      `LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST=aawm-litellm,litellm-dev` to
      both Langfuse services. Targeted container checks confirmed both
      services loaded the whitelist and a non-empty encryption key.
    - `curl -sS http://127.0.0.1:4000/v1/models` shows prod
      `aawm-litellm` exposes `openrouter/*`; a direct prod smoke to
      `/v1/chat/completions` with model `openrouter/owl-alpha` returned
      `200` and content `ok.`.
    - Created Langfuse LLM connection
      `provider=litellm-openrouter-judge`, `adapter=openai`,
      `baseURL=http://aawm-litellm:4000/v1`,
      `customModels=["openrouter/owl-alpha"]`, `withDefaultModels=false`, and
      `extraHeaderKeys=["langfuse_trace_name","langfuse_trace_user_id","langfuse_trace_metadata"]`.
    - Created project evaluator
      `AAWM Selected Trace Quality (Owl Alpha)` (`cmp9xdgpa0006o607j7124442`)
      with `modelConfig.provider=litellm-openrouter-judge` and
      `modelConfig.model=openrouter/owl-alpha`.
    - Created active evaluation rule
      `AAWM Owl Alpha trace-quality tagged generations`
      (`cmp9xeqx70009o607n2e9izcv`) targeting only `GENERATION`
      observations on traces tagged `llm-judge-candidate`, with judge trace
      names excluded from the filter. `GET
      /api/public/unstable/evaluation-rules` reports `status=active`.
    - Langfuse trace `fb9f4bdf-53be-4158-930f-f11ac3ac2ada` proves the
      evaluator preflight call went through LiteLLM with
      `name=langfuse-llm-judge.openrouter.owl-alpha`,
      `userId=langfuse-evaluator`, and metadata
      `source=langfuse_judge`, `judge_model=openrouter/owl-alpha`,
      `purpose=evaluation`.

- D1-103 provider status observations Docker sidecar
  - Goal: move the active provider-status observation collector out of host
    cron and into a reproducible Docker image/service without losing ICMP,
    DNS, TCP, TLS, DB, or environment behavior.
  - Changed paths:
    - `.dockerignore`
    - `docker-compose.dev.yml`
    - `docker/Dockerfile.provider_status_observations`
    - `scripts/record_provider_status_observations.py`
    - `scripts/run_provider_status_observations_loop.py`
    - `tests/test_scripts/test_record_provider_status_observations.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added Docker service `provider-status-observations` with container name
      `aawm-provider-status-observations`, restart policy `unless-stopped`,
      external network `aawm_default`, DB env parity with `litellm-dev`, and
      `cap_add: NET_RAW` so ICMP remains available in-container.
    - Added a dedicated lightweight image
      `aawm-provider-status-observations:dev` using
      `docker/Dockerfile.provider_status_observations`; final image id is
      `2935f83094d1` and final image size is `150MB`. `.dockerignore` now
      excludes local generated artifacts including `.analysis/`, `.wheel-build/`,
      and `captures/`, which prevented a 4.6GB capture directory from entering
      Docker build context.
    - Added `scripts/run_provider_status_observations_loop.py`, a container
      loop runner with defaults matching the old schedule:
      `AAWM_PROVIDER_STATUS_INTERVAL_SECONDS=300`,
      `AAWM_PROVIDER_STATUS_TIMEOUT=2`,
      `AAWM_PROVIDER_STATUS_PING_COUNT=1`,
      `AAWM_PROVIDER_STATUS_PING_TIMEOUT=2`, `AAWM_PROVIDER_STATUS_APPLY=1`,
      and `AAWM_LITELLM_ENVIRONMENT=dev`.
    - Made `scripts/record_provider_status_observations.py` standalone enough
      for the lightweight image by owning the provider-status DDL locally, with
      a regression test that keeps those constants equal to
      `AawmAgentIdentity` callback constants. Ping subprocess timeouts now
      store an `icmp_timeout` observation row instead of aborting the whole
      cycle.
    - Verification commands passed:
      `./.venv/bin/python -m py_compile scripts/record_provider_status_observations.py scripts/run_provider_status_observations_loop.py`;
      `./.venv/bin/python -m pytest tests/test_scripts/test_record_provider_status_observations.py -q`
      (`9 passed`); `./.venv/bin/ruff check
      scripts/record_provider_status_observations.py
      scripts/run_provider_status_observations_loop.py
      tests/test_scripts/test_record_provider_status_observations.py`; and
      `docker compose -f docker-compose.dev.yml config`.
    - Docker one-shot smoke passed:
      `docker compose -f docker-compose.dev.yml run --rm --no-deps -e
      AAWM_PROVIDER_STATUS_ONCE=1 provider-status-observations` logged
      `row_count=36`, `success_count=36`, `failure_count=0`, `inserted=true`,
      `environment=dev`, `observed_at=2026-05-17T14:28:41Z`.
    - Long-running service was rebuilt and recreated with
      `docker compose -f docker-compose.dev.yml up -d --build --force-recreate
      provider-status-observations`; running container is `0a783922fe21`,
      status `Up`, from image `aawm-provider-status-observations:dev`.
    - The running service logged a successful inserted cycle with
      `row_count=36`, `success_count=36`, `failure_count=0`,
      `observed_at=2026-05-17T14:30:02Z`.
    - Exact database verification against
      `aawm_tristore.public.provider_status_observations` returned
      `current_database=aawm_tristore`, latest observed row
      `2026-05-17 14:30:09.795908+00`, and `156` rows in the last ten
      minutes. Recent rows included `icmp_ping`, `dns`, `tcp_connect`, and
      `tls_handshake` successes for `control:google.com`,
      `cloudcode-pa.googleapis.com:443`,
      `generativelanguage.googleapis.com:443`, and
      `integrate.api.nvidia.com:443`.
    - Removed only the old host crontab provider-status schedule after the
      container and DB proof. `crontab -l` now shows only the unrelated
      two-hour Claude backup job.

## 2026-05-16

- D1-107 Grok Build embedding session-history and identity enrichment
  - Goal: make Grok Build embedding calls create normal
    `public.session_history` rows and enrich Grok traces with repository,
    tenant, trace name, and trace user id without relying on custom AAWM
    headers.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added OpenAI-compatible `/v1/embeddings` detection to passthrough
      success logging for xAI/Grok routes. Embedding responses now build a
      LiteLLM `EmbeddingResponse`, set `call_type=embedding`, preserve native
      usage, and compute cost from the `xai/grok-build` price map even when
      the Grok request relies on `x-grok-model-override` instead of a body
      `model`.
    - Added Grok-native identity promotion in `AawmAgentIdentity`: for
      `grok_cli_chat_proxy`/xAI contexts, generic `grok-build` trace names are
      enriched to `grok-build.<agent>` when prompt context exposes an agent;
      repository and tenant are extracted from the same request body/context
      paths used by Codex; generic Grok trace user ids are promoted to the
      repository, with tenant as fallback.
    - Mirrored the callback implementation into
      `.wheel-build/aawm_litellm_callbacks/agent_identity.py`; byte compare
      between the source callback and wheel-build callback returned `0`.
    - `./.venv/bin/python -m py_compile
      litellm/proxy/pass_through_endpoints/success_handler.py
      litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py
      litellm/integrations/aawm_agent_identity.py
      .wheel-build/aawm_litellm_callbacks/agent_identity.py` passed.
    - `./.venv/bin/python -m pytest
      tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k
      'grok or embedding'` passed (`8 passed, 153 deselected`).
    - `./.venv/bin/python -m pytest
      tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py
      -q -k 'xai_embedding or grok_proxy_route'` passed (`4 passed,
      316 deselected`).
    - `git diff --check` and `./.venv/bin/python -m ruff check --select F
      litellm/proxy/pass_through_endpoints/success_handler.py
      litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py
      litellm/integrations/aawm_agent_identity.py
      .wheel-build/aawm_litellm_callbacks/agent_identity.py
      tests/test_litellm/integrations/test_aawm_agent_identity.py` passed.
      Running the same `--select F` check over the full passthrough endpoint
      test file is still blocked by pre-existing unused import/local findings
      outside this change.
    - Restarted dev `litellm-dev`; `/health/readiness` on `:4001` returned
      healthy. Runtime inspection inside the container confirmed
      `PassThroughEndpointLogging` adapts an xAI/Grok embedding payload to
      `https://api.x.ai/v1/embeddings`, the OpenAI passthrough handler
      recognizes `https://cli-chat-proxy.grok.com/v1/embeddings`, and the
      mounted callback source contains the Grok identity promotion helper.
    - Exact database `aawm_tristore.public.session_history` currently has no
      xAI embedding rows (`max(id)=0` for xAI rows matching embedding
      call/model/metadata filters), so live success proof still requires a
      real Grok Build embedding request through dev.

- D1-100 Anthropic weekly quota interval correction
  - Goal: verify whether the latest Anthropic weekly quota was really `89%`
    remaining or whether the read model was failing to reflect the latest
    provider headers after a reset/recovery.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/session-history-rate-limit-usage-materialized.sql`
    - `.analysis/session-history-rate-limit-usage-optimized.sql`
    - `.analysis/completed.md`
  - Evidence:
    - Latest exact database `aawm_tristore.public.rate_limit_observations`
      rows for `anthropic_unified_7d:7d` showed provider-header source
      `anthropic_response_headers`, reset `2026-05-21 15:00:00+00`, and
      `remaining_pct=93` through row `21080` at
      `2026-05-16 02:18:41.058639+00`.
    - Confirmed the bug was in `public.rate_limit_intervals`: the prior view
      grouped by `(provider, quota_key, quota_type, expected_reset_at,
      remaining_pct)` and used `MIN(observed_at)`, so an older `89%` interval
      from `2026-05-15 17:56:54.475186+00` stayed open after later rows moved
      back through `100`, `99`, `94`, and `93`.
    - Reworked the rate-limit interval view SQL to build ordered transition
      intervals with `LAG(remaining_pct)` and reset-time comparison instead of
      first-seen percentage grouping. Mirrored the change into the callback
      wheel source and reusable session-history/rate-limit SQL artifacts.
    - Rebuilt exact database `aawm_tristore.public.rate_limit_intervals` and
      `public.provider_latency_health_5m`. Current
      `rate_limit_intervals` for `anthropic_unified_7d:7d` is open-ended at
      `remaining_pct=93` from `2026-05-16 02:13:16.321565+00` to
      `9999-12-31`, with reset `2026-05-21 15:00:00+00`.
    - The 5-minute overlay now includes the current weekly value in the
      `2026-05-16 02:10:00+00` Anthropic bucket, but that row also includes
      short quota keys, so `min_remaining_pct` can reflect the 5-hour quota
      while `max_remaining_pct` reflects the 7-day quota.
    - `./.venv/bin/python -m py_compile
      litellm/integrations/aawm_agent_identity.py
      .wheel-build/aawm_litellm_callbacks/agent_identity.py` passed.
    - `./.venv/bin/python -m pytest
      tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k
      'rate_limit_intervals or grok_monthly_billing or invalid_grok_billing or
      rate_limit_observations'` passed (`17 passed, 142 deselected`).
    - `git diff --check` and `./.venv/bin/ruff check --select F
      litellm/integrations/aawm_agent_identity.py
      tests/test_litellm/integrations/test_aawm_agent_identity.py` passed.

- D1-103/D1-107 Grok billing quota capture in rate-limit observations
  - Goal: capture Grok Build `/grok/v1/billing` monthly usage without filling
    `public.rate_limit_observations` with every billable unit, using only
    whole-number remaining percentage changes.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added Grok billing extraction from pass-through callback payloads and
      response bodies shaped as `config.monthlyLimit.val`,
      `config.used.val`, and `config.billingPeriodEnd`.
    - Storage normalizes `remaining_pct` as the rounded whole value of
      `100 - used/monthlyLimit*100`; `expected_reset_at` is the Grok billing
      period end; source is `grok_billing`; provider/client/model are
      `xai` / `grok-build` / `grok-build`.
    - Stored `quota_type`, generated `quota_key`, and emitted limit scope now
      use `requests` after live behavior showed Grok billing increments across
      tool/request traffic rather than only user-facing messages. The in-memory
      observation evidence keeps `quota_unit=grok_billing_used` and
      `quota_unit_interpretation=requests`; the database table stores only the
      normalized snapshot columns.
    - Existing latest-snapshot suppression is reused, so a new row is written
      only when the identity, reset, model, quota type, or rounded
      `remaining_pct` changes.
    - `./.venv/bin/python -m py_compile
      litellm/integrations/aawm_agent_identity.py
      .wheel-build/aawm_litellm_callbacks/agent_identity.py` passed.
    - `./.venv/bin/python -m pytest
      tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k
      'grok_monthly_billing or invalid_grok_billing or
      rate_limit_observations'` passed (`16 passed, 142 deselected`).
    - `git diff --check` and `./.venv/bin/ruff check --select F
      litellm/integrations/aawm_agent_identity.py
      tests/test_litellm/integrations/test_aawm_agent_identity.py` passed.
      Full `ruff check` remains blocked by pre-existing `PLR0915` and `E501`
      findings in this callback/test file.
    - Dev DB smoke against exact database `aawm_tristore` built two identical
      synthetic Grok billing observations and stored one row in
      `public.rate_limit_observations`:
      `provider=xai`, `client=grok-build`, `model=grok-build`,
      `quota_key=xai_grok_build_monthly_requests:requests`,
      `quota_type=requests`, `expected_reset_at=2026-06-01 00:00:00+00`,
      `remaining_pct=99.0`, `source=grok_billing`,
      `account_hash=e95f5b081b15`.
    - Restarted dev `litellm-dev` and verified readiness. Runtime inspection
      inside the container confirmed `/app/litellm/integrations/aawm_agent_identity.py`
      contains `grok_billing` and resolves explicit quota type `requests`.
    - The first post-reload real `/grok/v1/billing` attempt returned upstream
      `400` with `{"code":"The operation was cancelled","error":"Timeout expired"}`,
      so no successful billing payload was available to store. The failure path
      did write exact database `aawm_tristore.public.provider_error_observations`
      row `113` as `provider=xai`, `route_family=grok_cli_chat_proxy`,
      `status_code=400`, `error_class=provider_timeout`, and
      `client_name=grok-pager`.
    - A later successful real dev billing payload wrote exact database
      `aawm_tristore.public.rate_limit_observations` row `20906` with
      `provider=xai`, `client=grok-build`, `model=grok-build`,
      `quota_type=requests`, `remaining_pct=99`,
      `expected_reset_at=2026-06-01 00:00:00+00`, and
      `account_hash=452914a8516e`.
    - Normalized existing exact database
      `aawm_tristore.public.rate_limit_observations` Grok billing rows to
      `quota_key=xai_grok_build_monthly_requests:requests`; after callback
      duplicate cleanup, current `grok_billing` rows are `20901` and `20906`,
      and verification returned `0` remaining `grok_billing` rows whose key
      contains `messages`.
    - Rebuilt exact database `aawm_tristore.public.rate_limit_intervals` so it
      includes `provider=xai` request quota intervals, then rebuilt/refreshed
      `public.provider_latency_health_5m` from the updated SQL. Current
      `rate_limit_intervals` has xAI row
      `xai|grok-build|xai_grok_build_monthly_requests:requests|requests|99`;
      current `provider_latency_health_5m` has xAI traffic buckets and no
      `quota_keys` containing `messages`.
    - Restarted dev `litellm-dev`; `/health/readiness` returned healthy and
      runtime inspection inside the container showed
      `/app/litellm/integrations/aawm_agent_identity.py` contains
      `xai_grok_build_monthly_requests` and no
      `xai_grok_build_monthly_messages`.
    - Live Grok Build traffic in the same window confirmed the request-like
      interpretation: from `2026-05-16 01:01:42.758826+00` through
      `2026-05-16 01:20:53.230149+00`, exact database
      `aawm_tristore.public.session_history` has `22` xAI rows, `14` rows with
      tool calls, and `28` total recorded tool calls.
    - `./.venv/bin/python -m pytest
      tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k
      'grok_monthly_billing or invalid_grok_billing or rate_limit_intervals or
      rate_limit_observations'` passed (`17 passed, 142 deselected`).

## 2026-05-15

- D1-103/D1-107 xAI provider health probes and passive Grok error capture
  - Goal: include Grok/xAI in active provider health probes and preserve
    Grok Build upstream failures in `provider_error_observations` before and
    after prod restart.
  - Changed paths:
    - `scripts/record_provider_status_observations.py`
    - `scripts/backfill_provider_error_observations_from_docker_logs.py`
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_scripts/test_provider_status_observations.py`
    - `tests/test_scripts/test_backfill_provider_error_observations_from_docker_logs.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added active xAI endpoints `api.x.ai:443` and
      `cli-chat-proxy.grok.com:443` to
      `scripts/record_provider_status_observations.py`.
    - Added Grok/xAI passthrough failure enrichment and direct
      `provider_error_observations` capture for `/grok` failures, with
      callback import fallback from source integration to the deployed
      callback wheel package.
    - Made provider-error inserts idempotent on
      `litellm_call_id + provider + route_family + status_code`.
    - Enhanced docker-log backfill classification so `/grok` and xAI auth
      logs are recorded as `provider=xai`,
      `route_family=grok_cli_chat_proxy`, and `model=grok-build` for
      `/grok/v1/responses`.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_provider_status_observations.py tests/test_scripts/test_backfill_provider_error_observations_from_docker_logs.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py -q -k 'default_endpoints_include_xai or grok_auth_failure or provider_error_observation_insert_sql_dedupes_litellm_call_id or provider_status_observations_schema or xai_context_on_failure_logging or callback_wheel_fallback'`
      passed (`6 passed, 225 deselected, 71 warnings`), and targeted
      `py_compile` plus `git diff --check` passed before release.
    - Dev `litellm-dev` was restarted. A controlled `/grok/v1/responses`
      fake-auth smoke returned an upstream-shaped `401` and wrote exact
      database `aawm_tristore.public.provider_error_observations` row `103`
      as `environment=dev`, `provider=xai`, `model=grok-build`,
      `status_code=401`, `error_class=auth_failed`,
      `route_family=grok_cli_chat_proxy`, and
      `litellm_call_id=10fa87a9-6e7a-4fb3-8033-2f3d0e9b792c`; dedupe count
      for that call id was `1`.
    - Before prod restart, docker-log backfill preserved retained xAI/Grok
      errors as exact database rows `104` and `106`; row `106` is
      `environment=prod`, `provider=xai`, `model=grok-build`,
      `status_code=401`, `error_class=auth_failed`,
      `route_family=grok_cli_chat_proxy`, and
      `metadata.source=docker_log_backfill`.
    - Release commits `1cd806a5ee`, `444b0c2477`, and `3de47db732` were
      pushed; tags `v1.82.3-aawm.54` and `v1.82.3-aawm.55` were pushed;
      GitHub Actions image publish runs `25947258471` and `25947731095`
      succeeded.
    - Prod `aawm-litellm` is running `litellm=1.82.3+aawm.55`,
      `aawm-litellm-callbacks=0.0.32`, and
      `aawm-litellm-control-plane=0.0.7`; `/health/readiness` returned
      healthy.
    - A controlled prod `/grok/v1/responses` fake-auth smoke returned an
      upstream-shaped `401` and wrote exact database row `112` as
      `environment=prod`, `provider=xai`, `model=grok-build`,
      `status_code=401`, `error_class=auth_failed`,
      `route_family=grok_cli_chat_proxy`, and
      `litellm_call_id=f1478601-0e00-4090-a0b7-5f86e44c665d`; dedupe count
      for that call id was `1`, and blank `metadata.source` confirms live
      callback/direct capture rather than docker-log backfill.
    - Active probe rows at `2026-05-16 00:35 UTC` show successful DNS, ICMP
      ping, TCP connect, and TLS handshake probes for `api.x.ai:443` and
      `cli-chat-proxy.grok.com:443`, with observed ICMP RTTs of `47.967 ms`
      and `30.105 ms` respectively.
    - Prod logs after `aawm.55` no longer show
      `No module named 'litellm.integrations.aawm_agent_identity'`.

- D1-105 Codex auto-agent continuation scanner hotfix
  - Goal: stop `aawm-codex-agent-auto` from crashing before candidate
    fallback when request payloads contain nested non-string `type` values.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Prod logs showed `/openai_passthrough/responses` failing with
      `TypeError: unhashable type: 'dict'` inside
      `_codex_auto_agent_request_has_continuation_state` before
      `_select_codex_auto_agent_candidate` could run, so the alias had no
      chance to fail over.
    - The scanner now treats continuation `type` markers only when they are
      strings and skips already-seen dict/list objects to avoid recursion
      crashes.
    - Added regression coverage for nested dict-valued `type` fields and
      self-referential payloads.
    - `./.venv/bin/python -m py_compile
      litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py
      tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest
      tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py
      -q -k 'codex_auto_agent'` passed (`11 passed, 308 deselected`).
    - Release commit `3a79ee9858` bumped the fork to
      `1.82.3+aawm.53`; annotated tag `v1.82.3-aawm.53` was pushed; GitHub
      Actions run `25944984955` completed successfully and published
      `ghcr.io/zepfu/litellm:1.82.3-aawm.53`.
    - `/home/zepfu/projects/aawm-infrastructure` commit `033b377` was pushed
      to `develop`; it pins `Dockerfile.litellm` and
      `docker-compose.litellm.yml` to `ghcr.io/zepfu/litellm:1.82.3-aawm.53`
      and commits the already-prod-live OpenRouter Qwen/wildcard config entries
      that were copied into the promoted image.
    - Built prod image `aawm-litellm:latest` as
      `sha256:03d696966c95b36568e1f05c63e7e957b13ad9594bf29e358b392eb12678fe91`.
      Pre-restart image inspection reported `litellm=1.82.3+aawm.53`,
      `aawm-litellm-callbacks=0.0.30`,
      `aawm-litellm-control-plane=0.0.7`, confirmed the
      `isinstance(item_type, str)` source marker, and returned `False` for a
      dict-valued nested `type` payload.
    - Recreated prod `aawm-litellm`; running container `3233a8908770` started
      at `2026-05-15T22:57:11.703183605Z` from image
      `sha256:03d696966c95b36568e1f05c63e7e957b13ad9594bf29e358b392eb12678fe91`.
      `/health/readiness` returned healthy with
      `litellm_version=1.82.3+aawm.53`.
    - Runtime inspection inside `aawm-litellm` reported
      `litellm=1.82.3+aawm.53`, `aawm-litellm-callbacks=0.0.30`,
      `aawm-litellm-control-plane=0.0.7`, confirmed the
      `isinstance(item_type, str)` source marker, and returned `False` for a
      dict-valued nested `type` payload.
    - Synthetic prod `/openai_passthrough/responses` smokes for both
      `gpt-5.4-mini` and `aawm-codex-agent-auto` returned upstream-shaped
      OpenAI `400 invalid_type` responses instead of local HTTP `500`.
      Exact database `aawm_tristore.public.provider_error_observations` row
      `96` records the alias smoke as `provider=openai`,
      `model=aawm-codex-agent-auto`, `status_code=400`,
      `error_class=adapter_error`, and `route_family=openai_responses`.
    - Post-smoke prod log scan:
      `docker logs --tail 500 aawm-litellm 2>&1 | rg -n 'unhashable type|Exception in ASGI application|TypeError'`
      returned no matches.
    - Because the `aawm.53` base also contains D1-107 Grok code, synthetic prod
      `GET /grok/v1/settings` with fake Grok auth and
      `x-grok-client-version: 0.1.210` now reaches
      `https://cli-chat-proxy.grok.com/v1/settings` and returns upstream-shaped
      `401` with `auth_kind=bearer`, not FastAPI route-level `404`.

- D1-107 Grok Build dev smoke, full observed header expansion, and pricing map
  - Goal: record the first real Grok Build success through dev LiteLLM, preserve
    the remaining observed Grok-native identity/session headers, and map
    `grok-build` pricing so future rows can get `response_cost_usd`.
  - Changed paths:
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - User reported a real Grok Build test message worked through the dev
      `/grok` route.
    - Dev logs around the live run showed successful native requests including
      `POST /grok/v1/responses HTTP/1.1" 200 OK` and
      `POST /grok/v1/traces HTTP/1.1" 200 OK`.
    - Exact database `aawm_tristore.public.session_history` contains Grok rows
      `373744`, `373754`, and `373805` as `provider=xai`,
      `model=grok-build`, `route_family=grok_cli_chat_proxy`, and nonzero
      token counts. Their `response_cost_usd` values are blank because they
      were written before the pricing map reload.
    - Added the remaining live-observed Grok headers to the forward allowlist:
      `x-grok-agent-id`, `x-grok-conv-id`, `x-grok-req-id`,
      `x-grok-user-id`, `x-grok-turn-idx`, `x-email`, `x-userid`, and
      `x-teamid`; egress marker detection now treats any `x-grok-*` header as
      xAI.
    - Added `xai/grok-build` to both model price maps with user-supplied Grok
      4.3 pricing: input `1.25e-6`, cached input `2e-7`, and output
      `2.5e-6` dollars per token.
    - `./.venv/bin/python` JSON/load checks confirmed both cost-map entries,
      and `litellm.get_model_info(model='grok-build',
      custom_llm_provider='xai')` resolves `xai/grok-build` with those rates.
    - Focused tests passed:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py -q -k 'grok_proxy_route or raw_body_without_json_parse or xai_grok_headers or adapted_xai or skips_xai_embeddings'`
      (`8 passed, 378 deselected`).
    - Restarted dev with
      `docker compose -f docker-compose.dev.yml restart litellm-dev`; readiness
      returned healthy on `:4001`, runtime allowlist includes the expanded
      Grok header set, and the running container resolves `xai/grok-build`
      pricing as `1.25e-06`, `2e-07`, and `2.5e-06`.
    - Backfilled the three pre-pricing Grok Build rows in exact database
      `aawm_tristore.public.session_history` using uncached input
      `1.25e-6`, cached input `2e-7`, and output `2.5e-6` dollars per token:
      row `373744` now has `response_cost_usd=0.00635275`, row `373754` has
      `0.0223156`, and row `373805` has `0.0062065`.
    - Verification query returned `0|3` for blank-cost Grok Build rows vs.
      total Grok Build rows, and all three rows carry metadata marker
      `response_cost_backfill_source=grok_build_pricing_2026_05_15`.

- D1-107 Grok native header comparison and protobuf passthrough fix
  - Goal: compare native Grok Build request headers against the `/grok`
    forwarding allowlist, fix the `426 Upgrade Required` response that reported
    CLI version `(none)`, and stop protobuf telemetry requests from failing
    local JSON parsing.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - `grok --version` reports local Grok Build `0.1.210 (8b63e9068)`, while
      the upstream response through LiteLLM was `426 Upgrade Required` with
      `Your Grok CLI version (none) is outdated`, indicating a stripped native
      version signal rather than an outdated binary.
    - Static binary string comparison found Grok-native header candidates
      `x-grok-client-version`, `x-grok-client-identifier`, and
      `grok-shell-timestamp`; the first allowlist forwarded
      `Authorization`, `X-XAI-Token-Auth`, `x-grok-model-override`,
      `x-grok-session-id`, `user-agent`, and content headers, but omitted
      `x-grok-client-version`.
    - The `/grok` allowlist now forwards `x-grok-client-version`,
      `x-grok-client-identifier`, and `grok-shell-timestamp`, and includes a
      redacted header-name compare log for any remaining stripped Grok headers.
      Runtime check inside `litellm-dev` shows the active allowlist includes
      `['grok-shell-timestamp', 'x-grok-client-identifier',
      'x-grok-client-version', 'x-grok-model-override', 'x-grok-session-id',
      'x-xai-token-auth']`.
    - Added raw-body passthrough support for native binary/protobuf side-channel
      endpoints. `/grok` marks non-JSON requests as parsed before
      `user_api_key_auth`, avoiding the prior auth-layer JSON parse failure for
      `application/x-protobuf`.
    - Focused tests passed:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py -q -k 'grok_proxy_route or raw_body_without_json_parse'`
      (`4 passed, 381 deselected`), and `./.venv/bin/python -m py_compile
      litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py
      litellm/proxy/pass_through_endpoints/pass_through_endpoints.py` passed.
    - `litellm-dev` was restarted. Live synthetic `POST
      /grok/v1/responses` with fake Grok auth and `x-grok-client-version:
      0.1.210` now reaches upstream and returns upstream-shaped `401` with
      `x-litellm-model-api-base:
      https://cli-chat-proxy.grok.com/v1/responses`. Live synthetic
      `application/x-protobuf` `POST /grok/v1/traces` no longer fails local
      JSON parsing; it also reaches upstream and returns upstream-shaped `401`
      with `x-litellm-model-api-base:
      https://cli-chat-proxy.grok.com/v1/traces`.

- D1-107 dev route reload correction
  - Goal: correct the live dev state after the Grok passthrough route had been
    implemented locally but not loaded by the running ASGI process.
  - Changed paths:
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Before the restart, both `curl -sS -i
      http://127.0.0.1:4000/grok/v1/settings` and `curl -sS -i
      http://127.0.0.1:4001/grok/v1/settings` returned LiteLLM/FastAPI
      route-level `404 {"detail":"Not Found"}`.
    - Runtime file checks showed prod `aawm-litellm` did not contain
      `grok_proxy_route`, while dev `litellm-dev` did contain
      `grok_proxy_route`, `_GROK_CLI_FORWARD_HEADER_ALLOWLIST`, and
      `/grok/{endpoint:path}` in
      `/app/litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`.
    - A standalone import inside `litellm-dev` showed
      `['/grok/{endpoint:path}']`, confirming the route code was present but
      the running server needed restart.
    - `docker compose -f docker-compose.dev.yml restart litellm-dev` completed,
      and `curl -sS -i http://127.0.0.1:4001/grok/v1/settings` then returned
      upstream-shaped `401` with
      `x-litellm-model-api-base:
      https://cli-chat-proxy.grok.com/v1/settings`, proving the dev route is
      now active.
    - Synthetic header smoke:
      `curl -sS -i -H 'Authorization: Bearer fake-oidc-token' -H
      'X-XAI-Token-Auth: fake-xai-token' -H 'x-grok-model-override:
      grok-code-fast-1' http://127.0.0.1:4001/grok/v1/settings` changed the
      upstream error detail to `auth_kind=bearer`, proving the route forwards
      the Grok auth header shape. Real Grok CLI credentials still need a live
      smoke before prod cutover.

- D1-107 native Grok Build pass-through local implementation
  - Goal: add a native passthrough lane for Grok Build that preserves the
    CLI-owned OIDC `Authorization` header plus `X-XAI-Token-Auth` and
    `x-grok-model-override`, while using LiteLLM only as the observability and
    audit layer before forwarding to xAI's cli-chat-proxy.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added `/grok/{endpoint:path}`. The route maps
      `/grok/v1/chat/completions` and `/grok/v1/responses` to
      `https://cli-chat-proxy.grok.com/v1/...` by default and avoids double
      `/v1` when the upstream override already includes `/v1`.
    - LiteLLM auth is separated from Grok auth: `x-litellm-api-key` or a
      `key` query parameter authenticates LiteLLM, while the Grok OIDC
      `Authorization` header is allowed through to xAI. The forwarded header
      allowlist excludes `x-litellm-api-key` and strips the local `key` query
      parameter before egress.
    - The egress guard now recognizes `api.x.ai`, `*.x.ai`,
      `cli-chat-proxy.grok.com`, and `*.grok.com` as `xai`, and treats
      `X-XAI-Token-Auth`, `x-grok-model-override`, and `x-grok-session-id` as
      xAI/Grok credential markers.
    - OpenAI-compatible passthrough logging now adapts xAI/Grok
      chat-completions and Responses payloads through the existing OpenAI
      handler, preserving `custom_llm_provider=xai`.
    - `AawmAgentIdentity` resolves Grok native rows as `provider=xai` and
      prefers `x-grok-model-override` for `session_history.model`; the
      `.wheel-build` callback mirror is byte-for-byte identical to the in-repo
      callback after the change.
    - Focused tests passed:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_openai_passthrough_logging_handler.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'grok_proxy_route or xai or grok_header_model_override or adapted_xai or route_detection or chat_completions_route or responses_route'`
      (`23 passed, 549 deselected`).
    - `./.venv/bin/python -m py_compile` passed for the modified runtime and
      callback files, `git diff --check` passed for the touched files, and
      `cmp -s litellm/integrations/aawm_agent_identity.py
      .wheel-build/aawm_litellm_callbacks/agent_identity.py` passed.
  - Follow-up status: local implementation is complete. Live dev/prod smoke
    and release/cutover are tracked under D1-107 in `.analysis/todo.md`.

- D1-104 / D1-105 prod cutover to `v1.82.3-aawm.52`
  - Goal: deploy OpenRouter Qwen 3.5/3.6 Flash routing, OpenRouter
    provider-reported cost precedence, and Codex auto-agent in-flight provider
    stickiness to prod `aawm-litellm` on `:4000`.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/litellm_core_utils/get_llm_provider_logic.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `litellm-dev-config.yaml`
    - `docker-compose.dev.yml`
    - `PATCHES.md`
    - `pyproject.toml`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/llms/openrouter/test_openrouter_provider_routing.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - Release work was prepared from clean `origin/main` in
      `/tmp/litellm-release-aawm52`. Implementation commit
      `26a196e155` landed on `main`; artifact autobump run `25935928515`
      moved `main` to `6351b80d73` and bumped callback source to `0.0.30`.
    - Focused tests passed against the clean release worktree:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_auto_agent_alias'`
      (`9 passed, 305 deselected`),
      `./.venv/bin/python -m pytest tests/test_litellm/llms/openrouter/test_openrouter_provider_routing.py -q`
      (`15 passed`), and
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'openrouter or provider_error or provider_status_observations_schema_includes_icmp_fields'`
      (`13 passed, 140 deselected`).
    - Manually published callback release `cb-v0.0.30` because the autobump
      tag did not create a GitHub Release. Asset
      `aawm_litellm_callbacks-0.0.30-py3-none-any.whl` was uploaded with
      digest `sha256:522fa671352a0a5c23f07132b0ebb7adcab8f5c802e932b9261178037b78c8e5`.
    - Fork image release workflow run `25936058388` succeeded and published
      GitHub Release `v1.82.3-aawm.52` at `2026-05-15T19:08:32Z`.
    - Prod infrastructure commit `4e6ad61` was pushed to
      `zepfu/aawm-infrastructure` `develop`, pinning
      `Dockerfile.litellm` and `docker-compose.litellm.yml` to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.52`.
    - Rebuilt prod image with
      `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`.
      Built local image `aawm-litellm:latest` has image id
      `sha256:79df84d194ed08fa09fb9388e0b0ceb86faae3f9eebaa545a4157c1c318103b1`
      and pre-restart package inspection reported
      `litellm=1.82.3+aawm.52`, `aawm-litellm-callbacks=0.0.30`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Recreated prod service with
      `docker compose -f docker-compose.litellm.yml up -d --no-build --force-recreate --no-deps litellm`.
      Running container `8997f267a4db` is healthy on `127.0.0.1:4000`,
      from image
      `sha256:79df84d194ed08fa09fb9388e0b0ceb86faae3f9eebaa545a4157c1c318103b1`,
      with start time `2026-05-15T19:11:32.668566838Z`.
    - `/health/readiness` returned `status=healthy` and
      `litellm_version=1.82.3+aawm.52`; runtime package inspection inside the
      container returned `1.82.3+aawm.52`, callback `0.0.30`, and control
      plane `0.0.7`.
    - Runtime code/config proof: installed
      `llm_passthrough_endpoints.py` contains
      `aawm_codex_auto_agent_in_flight_provider_cooling_down`; installed
      callback contains `_first_reported_openrouter_cost`; rendered config
      includes `openrouter/qwen/qwen3.5-flash-02-23`,
      `openrouter/qwen/qwen3.6-flash`, `openrouter/*`, and
      `failure_callback`.
    - Prod OpenRouter smokes returned HTTP `200` for
      `openrouter/qwen/qwen3.6-flash` and
      `openrouter/qwen/qwen3.5-flash-02-23`. Exact database
      `aawm_tristore.public.session_history` wrote rows `368578` and
      `368579` with `litellm_environment=prod`, `provider=openrouter`,
      `litellm_fork_version=aawm.52`, wheel versions
      `{"litellm":"1.82.3+aawm.52","aawm-litellm-callbacks":"0.0.30","aawm-litellm-control-plane":"0.0.7"}`,
      token counts `20/297/317` and `20/570/590`,
      reported reasoning tokens `286` and `559`, and
      `response_cost_usd` matching `metadata.usage_openrouter_cost`
      (`0.000337875` and `0.0001495`).
    - Controlled prod Anthropic invalid `x-api-key` smoke without session
      header reached Anthropic and returned upstream `401 invalid x-api-key`.
      Exact database `aawm_tristore.public.provider_error_observations` wrote
      row `id=88`: `environment=prod`, `provider=anthropic`,
      `model=claude-opus-4-7`, `route_family=anthropic_messages`,
      `status_code=401`, `error_class=auth_failed`, and
      `metadata.observed_signal=normal_traffic_failure`.
    - Active provider status probes remained live after the cutover: in the
      last 15 minutes, exact database
      `public.provider_status_observations` had `21` rows each for `dns`,
      `tcp_connect`, `tls_handshake`, and `icmp_ping`, latest
      `2026-05-15 19:15:06.707436+00`.
    - Manual `REFRESH MATERIALIZED VIEW CONCURRENTLY
      public.provider_latency_health_5m` succeeded. Bucket
      `2026-05-15 19:10:00+00` includes the two OpenRouter Qwen model rows
      with provider/control ping; bucket `2026-05-15 19:15:00+00` includes
      Anthropic provider-error rows plus successful active probes.
    - Follow-up opened as D1-106: Anthropic pass-through requests carrying a
      `session_id` header are currently classified by the egress guard as
      OpenAI-marked and blocked before egress. The no-session invalid
      `x-api-key` smoke proves provider-error capture still works after
      cutover; the correlation-header false positive is tracked separately.
  - Follow-up status: prod cutover complete. Remaining work is D1-106 and the
    existing D1-103 downstream dashboard/reporter handoff.

- D1-105 Codex auto-agent in-flight provider stickiness
  - Goal: prevent `aawm-codex-agent-auto` from switching provider families
    inside an already-running agent attempt when the current affinity target
    hits `429`/quota/capacity exhaustion or has an active cooldown.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added continuation-state detection for Responses-style stateful requests
      (`previous_response_id`, function/tool call ids, tool outputs, MCP calls,
      reasoning items, and tool-role messages).
    - Made Codex auto-agent session affinity sticky. If an affinity target is
      already cooling down, the alias now returns HTTP `429` with code
      `aawm_codex_auto_agent_in_flight_provider_cooling_down` and a
      `Retry-After` header instead of selecting another provider.
    - If the affinity target returns retryable exhaustion during the request,
      the handler still records/sets the cooldown but re-raises the provider
      error as terminal for that agent attempt; first-call selector fallback is
      preserved for sessions that have not yet acquired provider-shaped state.
    - Regression coverage:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_auto_agent_alias'`
      (`9 passed, 305 deselected`).
    - Callback/persistence coverage:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'log_success_event_enqueues_session_history_record or async_log_success_event_enqueues_session_history_record or persist_session_history_record_executes_insert or persist_session_history_records_executes_batch_insert or provider_status_observations_schema_includes_icmp_fields or build_provider_error_observation or log_failure_event_enqueues_provider_error_without_quota or persist_session_history_records_writes_provider_error_observation'`
      (`9 passed, 144 deselected`) and
      `./.venv/bin/python -m pytest tests/test_scripts/test_record_provider_status_observations.py -q`
      (`3 passed`).
    - Dev runtime was restarted so bind-mounted pass-through code was loaded;
      `/health/readiness` returned healthy and startup logs showed
      `Initialized Success Callbacks` with `AawmAgentIdentity` and
      `Initialized Failure Callbacks` with `AawmAgentIdentity`.
    - Controlled dev success smoke on `127.0.0.1:4001/v1/chat/completions`
      with `model=qwen3-heretic-gguf` and session
      `manual-d1-105-observability-1778870002` returned HTTP `200`.
      Exact database `aawm_tristore.public.session_history` wrote row
      `id=367717`: `provider=local_llm`, `model=qwen3-heretic-gguf`,
      `model_group=qwen3-heretic-gguf`, `input_tokens=12`,
      `output_tokens=3`, `total_tokens=15`, `litellm_environment=dev`,
      and `repository=zepfu/litellm`.
    - Controlled dev failure smoke on `127.0.0.1:4001/v1/chat/completions`
      with `model=no-such-model-d1-105` and session
      `manual-d1-105-provider-error-1778870003` returned HTTP `400`.
      Exact database `aawm_tristore.public.provider_error_observations` wrote
      row `id=82`: `environment=dev`, `provider=openai`,
      `model=no-such-model-d1-105`, `status_code=400`,
      `error_class=adapter_error`, and
      `metadata.observed_signal=normal_traffic_failure`.
    - Active status probe writer was run against exact database
      `aawm_tristore`:
      `./.venv/bin/python scripts/record_provider_status_observations.py --apply --dsn postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore --environment dev --ping-count 1 --ping-timeout 1`.
      It inserted DNS/TCP/TLS/ICMP rows for Anthropic, OpenAI, OpenRouter,
      NVIDIA NIM, Gemini/Google Code Assist, and `control:google.com`.
      Fresh true ping averages included OpenAI `31.219 ms`, OpenRouter
      `21.155 ms`, Anthropic `28.18 ms`, and control `40.917 ms`.
    - Manual `REFRESH MATERIALIZED VIEW CONCURRENTLY
      public.provider_latency_health_5m; ANALYZE public.provider_latency_health_5m;`
      materialized the controlled failure and probe rows in bucket
      `2026-05-15 18:15:00+00`. The OpenAI
      `model=no-such-model-d1-105` row showed
      `provider_error_events=1`, `adapter_error_events=1`,
      `status_probe_count=8`, `provider_ping_avg_ms=31.403`,
      `control_ping_avg_ms=35.005`, and `status_probe_success_pct=100.00`.
    - `git diff --check -- litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py litellm/llms/openai/responses/transformation.py tests/test_litellm/llms/openai/responses/test_openai_responses_transformation.py`
      passed. A focused Ruff command on these two large existing files still
      reports pre-existing line-length/unused-import issues outside this patch,
      so it is not a clean gate for this change.
  - Follow-up status: implemented, unit-verified locally, and live-validated on
    dev `:4001` for success-history, failure-history, active status probes, and
    the five-minute rollup. Prod runtime cutover still needs the next LiteLLM
    release/redeploy before prod traffic uses the new policy.

- D1-104 OpenRouter Qwen 3.5/3.6 Flash routing
  - Goal: expose `openrouter/qwen/qwen3.5-flash-02-23` and
    `openrouter/qwen/qwen3.6-flash` through the direct OpenAI-compatible
    proxy route and the Anthropic-to-OpenRouter adapter path.
  - Changed paths:
    - `litellm/litellm_core_utils/get_llm_provider_logic.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/llms/openrouter/test_openrouter_provider_routing.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `litellm-dev-config.yaml`
    - `docker-compose.dev.yml`
    - `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added both Qwen model IDs to the Anthropic OpenRouter Responses adapter
      allowlist and covered both bare `qwen/...` and explicit
      `openrouter/qwen/...` request forms in the adapter route tests.
    - Fixed the OpenRouter provider resolver second-hop case where
      `custom_llm_provider="openrouter"` plus a regular slash-bearing model ID
      such as `qwen/qwen3.6-flash` was incorrectly rewritten to
      `openrouter/qwen/qwen3.6-flash`.
    - Pinned the two direct dev/prod model entries to
      `api_base: "https://openrouter.ai/api/v1"` and corrected the dev compose
      default `OPENROUTER_API_BASE` to `/api/v1`; the Anthropic adapter helper
      already strips `/v1` when it needs the `/api` root.
    - Added `get_llm_provider_logic.py` to the dev compose bind mounts so the
      resolver fix survives future dev container recreates.
    - Focused resolver coverage:
      `./.venv/bin/python -m pytest tests/test_litellm/llms/openrouter/test_openrouter_provider_routing.py -q`
      (`15 passed`).
    - Focused adapter coverage:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'openrouter_models_to_responses or openrouter_responses_adapter_model_supports_openrouter_prefix or openrouter_responses_adapter_model_supports_unknown_openrouter_prefix'`
      (`13 passed, 299 deselected`).
    - Fixed `session_history.response_cost_usd` for OpenRouter success rows to
      prefer OpenRouter's provider-reported `usage.cost` over generic LiteLLM
      computed/zero costs, so Anthropic-adapter OpenRouter calls do not write a
      zero or catalog-estimated cost when OpenRouter reported an actual charge.
    - Focused session-history callback coverage:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'openrouter_reported_cost or openrouter_hidden_response_cost or openrouter_rerank_tokens_from_request'`
      (`4 passed, 149 deselected`).
    - YAML/static checks:
      `./.venv/bin/python -c "import yaml; [yaml.safe_load(open(p)) for p in ['litellm-dev-config.yaml', 'docker-compose.dev.yml', '/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl']]; print('yaml-ok')"`
      returned `yaml-ok`; `git diff --check` passed for the LiteLLM touched
      files and the infra template.
    - Live dev `litellm-dev` was restarted and `/health/readiness` returned
      `status=healthy` with `litellm_version=1.82.3+aawm.25`.
    - Live direct proxy smokes on `127.0.0.1:4001/v1/chat/completions` returned
      HTTP `200` for both new models:
      `openrouter/qwen/qwen3.5-flash-02-23` returned content `OK`, and
      `openrouter/qwen/qwen3.6-flash` returned content `OK`.
    - Live Anthropic-compatible smokes on
      `127.0.0.1:4001/anthropic/v1/messages?beta=true` returned HTTP `200`
      for both new models:
      `openrouter/qwen/qwen3.5-flash-02-23` mapped upstream to
      `qwen/qwen3.5-flash-20260224` and returned `OK`, while
      `openrouter/qwen/qwen3.6-flash` mapped to `qwen/qwen3.6-flash` and
      returned `OK`.
    - Live dev `session_history` verification for controlled prefix
      `manual-openrouter-qwen-history-1778866278` wrote four rows:
      direct Qwen 3.5 row `367187` (`14` input, `124` output, `138` total,
      `response_cost_usd=0.00003315`), direct Qwen 3.6 row `367189` (`14`
      input, `154` output, `168` total,
      `response_cost_usd=0.000175875`), Anthropic-adapter Qwen 3.5 row
      `367190` (`14` input, `199` output, `213` total,
      `response_cost_usd=0.00005265`), and Anthropic-adapter Qwen 3.6 row
      `367192` (`14` input, `179` output, `193` total,
      `response_cost_usd=0.000204`). In all four rows,
      `response_cost_usd` equals metadata `usage_openrouter_cost`.
  - Follow-up status: dev routing and session-history token/cost capture are
    live-proven. The production config template is updated, but prod will need
    the next LiteLLM image/release and config deployment before these routes and
    the OpenRouter cost ordering fix are live on `aawm-litellm`.

- D1-103 provider health observability prod cutover
  - Goal: promote the provider health passive-error capture and rollup
    read-model to the prod `aawm-litellm` runtime without losing retained prod
    provider-error evidence.
  - Changed paths:
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
    - `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Built local prod image `aawm-litellm:latest` from base
      `ghcr.io/zepfu/litellm:1.82.3-aawm.51`; image inspection reported
      `litellm=1.82.3+aawm.51`, `aawm-litellm-callbacks=0.0.29`, and
      `aawm-litellm-control-plane=0.0.7`.
    - `/home/zepfu/projects/aawm-infrastructure` commit `3b04481` was pushed
      to `origin/develop`, pinning the base image and prod failure callback in
      `Dockerfile.litellm`, `docker-compose.litellm.yml`, and
      `config/litellm-config.yaml.tmpl`.
    - Recreated prod service with
      `docker compose -f docker-compose.litellm.yml up -d litellm`. Running
      container `60d5d574e2e6` is healthy on `127.0.0.1:4000`, with image
      `sha256:b62d8f201861550a55345eb46008d8a56e329ed40ff0d89cfd851295a282ca8a`
      and start time `2026-05-15T01:26:44.069267353Z`.
    - `/health/readiness` returned `status=healthy` and
      `litellm_version=1.82.3+aawm.51`; runtime package inspection inside the
      container returned `1.82.3+aawm.51`, callback `0.0.29`, and control
      plane `0.0.7`.
    - Rendered prod config at `/etc/litellm/config.yaml` contains
      `success_callback: ["langfuse", "aawm_litellm_callbacks.agent_identity.aawm_agent_identity_instance"]`
      and
      `failure_callback: ["aawm_litellm_callbacks.agent_identity.aawm_agent_identity_instance"]`.
      Startup logs also showed `Initialized Failure Callbacks` with
      `AawmAgentIdentity`.
    - A controlled prod Anthropic pass-through smoke sent an intentionally
      invalid `x-api-key` to `/anthropic/v1/messages?beta=true` and received
      upstream `401 invalid x-api-key` with LiteLLM call id
      `f8a5d554-fb01-401f-8596-701eb00251da`.
    - Exact database `aawm_tristore.public.provider_error_observations` wrote
      new prod row `id=72` at `2026-05-15 01:27:53.158097+00`:
      `provider=anthropic`, `model=claude-opus-4-7`,
      `route_family=anthropic_messages`, `status_code=401`,
      `error_type=HTTPException`, `error_class=auth_failed`,
      `litellm_call_id=17f21b56-f283-4542-939b-2e1f4e1e9be7`,
      `metadata.observed_signal=normal_traffic_failure`.
    - Active status probes are still writing on the already-installed
      five-minute host cadence: in the last 15 minutes,
      `public.provider_status_observations` had rows for `dns`, `icmp_ping`,
      `tcp_connect`, and `tls_handshake` across Anthropic, OpenAI, OpenRouter,
      NVIDIA NIM, Gemini/Google Code Assist, and
      `control:google.com`, with latest observed time
      `2026-05-15 01:25:11.036151+00`.
    - Manual `REFRESH MATERIALIZED VIEW CONCURRENTLY
      public.provider_latency_health_5m; ANALYZE public.provider_latency_health_5m;`
      completed; the view now has `8667` rows through bucket
      `2026-05-15 01:25:00+00`.
    - The refreshed rollup includes bucket `2026-05-15 01:25:00+00`,
      `environment=all`, `provider=anthropic`, `model=claude-opus-4-7`,
      `auth_failed_events=1`, `provider_error_events=1`,
      `status_probe_count=4`, `provider_ping_avg_ms=28.07`, and
      `control_ping_avg_ms=28.711`.
    - Existing pg_cron jobs are active for
      `aawm_provider_latency_health_5m_refresh` (`*/5 * * * *`) and
      `aawm_provider_latency_health_5m_analyze` (`1-59/5 * * * *`).
  - Follow-up status: prod passive provider-error capture is live. Remaining
    D1-103 work is downstream dashboard/reporter consumption of
    `.analysis/provider-latency-health-5m.sql`.

- D1-103 prod Docker-log provider-error preservation before restart
  - Goal: preserve the current prod `aawm-litellm` retained process-log errors
    into `public.provider_error_observations` before any prod container restart
    can rotate or discard the only raw log source for the currently observed
    Anthropic/Claude classifier failures.
  - Changed paths:
    - `scripts/backfill_provider_error_observations_from_docker_logs.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added a one-shot Docker-log backfill utility that reads
      `docker logs --timestamps`, normalizes LiteLLM `Exception occured - NNN`
      lines and unmatched HTTP `4xx`/`5xx` access lines, correlates nearby
      access logs, tags each inserted row with
      `metadata.source='docker_log_backfill'`, `metadata.container`, and a
      deterministic `metadata.log_signature`, and skips signatures already
      preserved by prior runs.
    - Dry-run command:
      `./.venv/bin/python scripts/backfill_provider_error_observations_from_docker_logs.py --container aawm-litellm --environment prod --since 2026-05-14T15:00:00Z --dsn postgresql://aawm:...@127.0.0.1:5434/aawm_tristore --show-samples 5`.
      It parsed `48,717` retained prod log lines, `43` exception logs,
      `61` HTTP `4xx`/`5xx` access logs, and `61` candidate observations.
    - Applied the preservation backfill with:
      `./.venv/bin/python scripts/backfill_provider_error_observations_from_docker_logs.py --container aawm-litellm --environment prod --since 2026-05-14T15:00:00Z --dsn postgresql://aawm:...@127.0.0.1:5434/aawm_tristore --show-samples 0 --apply`.
      It inserted `61` rows into exact database `aawm_tristore`.
    - Read-only verification against exact database `aawm_tristore` showed
      `public.provider_error_observations` now has `71` rows total, spanning
      `2026-05-14 15:11:55.461232+00` through
      `2026-05-15 00:36:09.022582+00`.
    - Backfilled prod rows by normalized group:
      `400 adapter_error gemini google_code_assist_generate_content = 9`;
      `400 adapter_error openai openai_responses = 3`;
      `401 auth_failed anthropic anthropic_messages = 1`;
      `404 adapter_error anthropic anthropic_messages = 6`;
      `405 adapter_error anthropic anthropic_messages = 6`;
      `500 provider_5xx proxy_internal proxy_internal = 12`;
      `503 provider_5xx openai openai_responses = 13`;
      `529 capacity_exhausted anthropic anthropic_messages = 11`.
    - Anthropic `529` preservation verification returned
      `11` rows from `2026-05-15 00:14:45.880791+00` through
      `2026-05-15 00:18:59.377844+00`. The retained prod log lines did not
      include model values for those `529` errors, so the backfilled rows keep
      `model=NULL` rather than inventing one.
    - A post-apply dry run over the same retained logs returned
      `existing_backfill_signatures=61` and `new_observations=0`, proving the
      preservation script is idempotent for this log set.
  - Follow-up status: current retained prod provider-error logs are preserved
    in `aawm_tristore` and it is safe to proceed to the prod restart/callback
    wheel parity step without losing this error evidence.

- D1-103 provider error pass-through failure capture repair
  - Goal: fix the dev/runtime gap where Anthropic pass-through failures,
    including the observed Claude auto-mode classifier `529` path, invoked
    `AawmAgentIdentity.log_failure_event` but did not insert a
    `public.provider_error_observations` row.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Dev logs for the user repro showed Anthropic pass-through `404` for
      `model: claude-opus-4-7[1m]` and multiple `529 overloaded_error` rows,
      followed each time by
      `AawmAgentIdentity.log_failure_event failed: maximum recursion depth exceeded`.
    - Reproduced the callback failure locally with a pass-through-shaped
      recursive request payload; before the fix,
      `_build_failure_observation_only_record(...)` raised
      `RecursionError: maximum recursion depth exceeded`.
    - Fixed the pass-through failure request payload to copy the parsed body
      before merging logging kwargs, avoiding a self-reference through
      `passthrough_logging_payload.request_body`.
    - Added cycle/depth protection to repository identity scanning so bad
      callback payloads cannot block provider-error observation construction.
    - Fixed failure observation construction to use `kwargs["exception"]` when
      LiteLLM calls custom failure loggers with `response_obj=None`, preserving
      upstream status/error body on the sync failure-callback path.
    - Local repro now returns
      `anthropic claude-opus-4-7[1m] 529 capacity_exhausted`.
    - Restarted `litellm-dev` and verified a controlled Anthropic pass-through
      POST failure writes exact database `aawm_tristore` row `id=6`:
      `environment=dev`, `provider=anthropic`,
      `model=claude-opus-4-7[1m]`, `status_code=401`,
      `error_class=auth_failed`, `route_family=anthropic_messages`, with
      normalized upstream error text in metadata.
    - The user's subsequent real Claude dev repro wrote exact database
      `aawm_tristore` rows `id=7` through `id=10` at
      `2026-05-15 00:35:53+00` through `00:36:09+00`, all with
      `environment=dev`, `provider=anthropic`, `model=claude-opus-4-7`,
      `status_code=529`, `error_class=capacity_exhausted`,
      `route_family=anthropic_messages`, and normalized Anthropic
      `overloaded_error` bodies in metadata.
    - Docker log verification over the repaired window showed the matching
      Anthropic `529 overloaded_error` stack traces and no new
      `AawmAgentIdentity.log_failure_event failed` / maximum-recursion warning.
    - Repository search found no proxy-side source for a `[1m]` model suffix;
      the dev/prod logs show LiteLLM forwarded that literal model string from
      the Claude request.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'recursive_passthrough_payload or provider_error'`
      passed (`5 passed, 146 deselected`, one existing pytest config warning).
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py -q -k 'does_not_mutate_custom_body_on_failure'`
      passed (`1 passed, 63 deselected`, existing deprecation/config warnings).
  - Follow-up status: dev now captures Anthropic pass-through failures in
    `provider_error_observations`; D1-103 remains open for prod cutover/callback
    wheel parity and downstream query handoff.

## 2026-05-14

- D1-103 provider health observability local/dev activation
  - Goal: take the provider health observability implementation beyond a plan
    and prove the active/passive pipeline against the local/dev database and
    dev proxy runtime.
  - Changed paths:
    - `.analysis/provider-latency-rate-limit-observability-plan-2026-05-14.md`
    - `.analysis/provider-latency-health-5m.sql`
    - `.analysis/provider-latency-health-5m-materialized-view.sql`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `litellm-dev-config.yaml`
    - `scripts/record_provider_status_observations.py`
    - `scripts/aawm-provider-status-observations.service`
    - `scripts/aawm-provider-status-observations.timer`
    - `tests/test_scripts/test_record_provider_status_observations.py`
  - Evidence:
    - Applied the callback's own provider-observability DDL constants to exact
      database `aawm_tristore`; verification returned
      `provider_error_observations,provider_status_observations`.
    - Ran the active collector with writes enabled:
      `./.venv/bin/python scripts/record_provider_status_observations.py --apply --environment dev --ping-count 1 --ping-timeout 2 --timeout 2 --dsn postgresql://aawm:...@127.0.0.1:5434/aawm_tristore`.
    - Read-only verification showed `28` new
      `public.provider_status_observations` rows since
      `2026-05-14 22:43:00+00`, covering `dns`, `icmp_ping`,
      `tcp_connect`, and `tls_handshake` for `anthropic`, `openai`,
      `openrouter`, `nvidia_nim`, two `gemini` endpoints, and
      `control:google.com`; all wrote `success=true` in the smoke run.
    - Inserted a controlled passive provider failure through the callback
      persistence path:
      `litellm_call_id=provider-health-smoke-ee168f99ecd74ebfaf85d1bf4cc3e200`,
      `environment=dev`, `provider=openrouter`,
      `model=openrouter/provider-health-smoke`, `error_class=provider_5xx`,
      `status_code=503`, `route_family=provider_health_smoke`,
      `session_id=provider-health-session-ee168f99ecd74ebfaf85d1bf4cc3e200`,
      and matching trace id.
    - Wired `litellm-dev-config.yaml` with
      `failure_callback: ["litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"]`
      and restarted `litellm-dev`; `/health/readiness` returned
      `status=healthy` with `AawmAgentIdentity` in the success callback list.
    - A dev proxy invalid-model smoke through `:4001` wrote a real runtime
      `public.provider_error_observations` row:
      `environment=dev`, `provider=openai`,
      `model=provider-health-invalid-model`, `error_class=adapter_error`,
      `session_id=8210d005-2707-469e-a6c2-5599923dd09b`,
      `trace_id=4cc6cb02-3082-49d0-afce-174c9670ff5e`.
    - Read-only `EXPLAIN (COSTS OFF)` of
      `.analysis/provider-latency-health-5m.sql` against exact database
      `aawm_tristore` succeeded with `89` plan lines.
    - Sample execution of `.analysis/provider-latency-health-5m.sql` returned
      `7` current active/passive rows, including provider-minus-control ping
      deltas for `anthropic`, `gemini`, `nvidia_nim`, `openai`, and
      `openrouter`, plus provider-error rows for the controlled smoke failures.
    - Created live materialized view
      `public.provider_latency_health_5m` in exact database `aawm_tristore`
      from `.analysis/provider-latency-health-5m.sql`; initial creation
      materialized `9151` rows before query-grain cleanup.
    - The first unique-index attempt exposed a real rollup grain issue:
      `model_group` could be both `NULL` and empty string, causing duplicate
      bucket/environment/provider/model rows. Normalized `model_group` with
      `NULLIF(sh.model_group, '')` in both the standalone query and the plan's
      materialized-view SQL.
    - Recreated `public.provider_latency_health_5m` after the normalization;
      duplicate check returned `0`, final row count was `8814`, and the view is
      populated (`pg_matviews.ispopulated = true`).
    - Created materialized-view indexes:
      `provider_latency_health_5m_unique_idx` on
      `(bucket_start, environment, provider, model, model_group) NULLS NOT DISTINCT`,
      `provider_latency_health_5m_provider_time_idx`, and
      `provider_latency_health_5m_model_time_idx`.
    - Verified `REFRESH MATERIALIZED VIEW CONCURRENTLY public.provider_latency_health_5m`
      and `ANALYZE public.provider_latency_health_5m` both run successfully.
    - Installed pg_cron jobs in exact database `aawm_tristore`:
      job `26`, `aawm_provider_latency_health_5m_refresh`, schedule
      `*/5 * * * *`, command
      `REFRESH MATERIALIZED VIEW CONCURRENTLY public.provider_latency_health_5m`;
      job `27`, `aawm_provider_latency_health_5m_analyze`, schedule
      `1-59/5 * * * *`, command
      `ANALYZE public.provider_latency_health_5m`; both are active.
    - Verified scheduled pg_cron execution:
      `aawm_provider_latency_health_5m_analyze` run `1290` succeeded at
      `2026-05-14 23:36:01+00` with return message `ANALYZE`, and
      `aawm_provider_latency_health_5m_refresh` run `1291` succeeded at
      `2026-05-14 23:40:00+00` with return message
      `REFRESH MATERIALIZED VIEW`.
    - Updated `.analysis/provider-latency-health-5m.sql` to use
      `public.provider_latency_health_5m` as the cached read model instead of
      recomputing the raw rollup CTE, for downstream dashboard/reporter reuse.
      Read-only validation against exact database `aawm_tristore` returned
      `explain_rows=5`, `query_row_count=8848` under the query's 14-day filter,
      and `mv_row_count=8849` for the full materialized view.
    - Rebuilt `public.provider_latency_health_5m` from
      `.analysis/provider-latency-health-5m-materialized-view.sql` at a
      cross-environment grain: raw source rows still preserve their original
      dev/prod environment values, but the materialized read model now emits
      `environment='all'` and groups by
      `(bucket_start, provider, model, model_group)`. Rebuild output was
      `DROP MATERIALIZED VIEW`, `SELECT 8596`, `CREATE INDEX` x3, and
      `ANALYZE`.
    - Verified the compressed grain against exact database `aawm_tristore`:
      `environments=all:8596`, duplicate
      `(bucket_start, provider, model, model_group)` groups `0`, handoff query
      validation `explain_rows=5`, `query_row_count=8593`, and post-rebuild
      scheduled pg_cron runs succeeded for refresh at
      `2026-05-15 00:10:00+00` and analyze at `2026-05-15 00:11:00+00`.
    - Added installable systemd templates for the recurring active collector:
      `scripts/aawm-provider-status-observations.service` and
      `scripts/aawm-provider-status-observations.timer`; `systemd-analyze
      verify` exited `0` with only unrelated host unit warnings.
    - Fixed `scripts/record_provider_status_observations.py` DSN precedence so
      explicit `AAWM_DB_*` component configuration wins over an ambient
      `AAWM_DATABASE_URL`; this matches the callback's DSN behavior and avoids
      cron inheriting a container-only `postgres18` URL. Added focused unit
      coverage for that precedence.
    - Installed the local/dev active collector in the user crontab, preserving
      the existing backup job:
      `*/5 * * * * cd /home/zepfu/projects/litellm && AAWM_DB_HOST=127.0.0.1 AAWM_DB_PORT=5434 AAWM_DB_NAME=aawm_tristore AAWM_DB_USER=aawm AAWM_DB_PASSWORD=aawm_dev AAWM_LITELLM_ENVIRONMENT=dev ./.venv/bin/python scripts/record_provider_status_observations.py --apply --environment dev --ping-count 1 --ping-timeout 2 --timeout 2 >> /home/zepfu/projects/litellm/.analysis/provider-status-observations-cron.log 2>&1`.
    - Verified host cron is active (`systemctl is-active cron` returned
      `active`, with `/usr/sbin/cron -f -P` visible in `ps`).
    - The first installed cron attempt exposed the DSN precedence bug and logged
      `failed to resolve host 'postgres18'`; after the fix, a cron-equivalent
      one-shot run wrote another `28` rows, and the real 23:00 UTC cron tick
      wrote a third `28`-row batch. Read-only bucket verification returned:
      `2026-05-14 22:40:00+00:00|28|28`,
      `2026-05-14 22:55:00+00:00|28|28`, and
      `2026-05-14 23:00:00+00:00|28|28`.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py scripts/record_provider_status_observations.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_record_provider_status_observations.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'record_provider_status or provider_error or provider_status_observations or google_capacity or persist_session_history_records_writes_provider_error'`
      passed (`9 passed, 144 deselected`, one existing pytest config warning).
    - `./.venv/bin/ruff check scripts/record_provider_status_observations.py tests/test_scripts/test_record_provider_status_observations.py`
      passed.
  - Follow-up status: local/dev active and passive provider-health monitoring
    is built and scheduled. D1-103 remains open for prod cutover and downstream
    dashboard/reporter query consumption.

- D1-102 provider health observability bootstrap
  - Goal: extend the provider latency/rate-limit observability plan into an
    initial implementation that can monitor provider health passively from real
    failures and actively from non-inference network probes, including ICMP ping
    and a control baseline.
  - Changed paths:
    - `.analysis/provider-latency-rate-limit-observability-plan-2026-05-14.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `litellm/integrations/aawm_agent_identity.py`
    - `scripts/record_provider_status_observations.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_scripts/test_record_provider_status_observations.py`
  - Evidence:
    - Plan now documents ICMP ping as a low-weight active signal, control
      endpoint baselines such as `control:google.com`, provider-minus-control
      dashboard comparison, and correlated-alert rules that do not page on
      ICMP-only degradation.
    - Callback schema bootstrap now creates
      `public.provider_error_observations` and
      `public.provider_status_observations`. The status table includes
      `packet_loss_pct`, `icmp_rtt_min_ms`, `icmp_rtt_avg_ms`,
      `icmp_rtt_max_ms`, `icmp_rtt_mdev_ms`, DNS/TCP/TLS timing fields, and
      endpoint/probe indexes.
    - Failure callbacks now enqueue `provider_error_observations` rows for
      passive normal-traffic failures, including non-quota provider `5xx` rows
      and Google/Gemini `MODEL_CAPACITY_EXHAUSTED` classification with reset
      hints.
    - Added `scripts/record_provider_status_observations.py`, a dry-run-first
      non-inference collector for configured provider/control endpoints. It
      emits `icmp_ping`, `dns`, `tcp_connect`, and `tls_handshake` rows and only
      writes to Postgres when `--apply` is passed.
    - Added `.analysis/provider-latency-health-5m.sql`, a standalone five-minute
      rollup query for the downstream dashboard/reporter repo. It joins
      `session_history`, `provider_error_observations`,
      `provider_status_observations`, and `rate_limit_intervals`, and includes
      provider ping versus control ping fields without adding any dashboard code
      to this repo.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m py_compile scripts/record_provider_status_observations.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed (`150 passed`, one existing pytest config warning).
    - `./.venv/bin/python -m pytest tests/test_scripts/test_record_provider_status_observations.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'record_provider_status or provider_error or provider_status_observations or google_capacity or persist_session_history_records_writes_provider_error'`
      passed (`8 passed, 144 deselected`, one existing pytest config warning).
    - `./.venv/bin/ruff check scripts/record_provider_status_observations.py tests/test_scripts/test_record_provider_status_observations.py`
      passed.
    - Full callback-file Ruff remains blocked by pre-existing `PLR0915` findings
      in large functions in `litellm/integrations/aawm_agent_identity.py`; the
      new script/tests are Ruff-clean.
    - Dry-run active probe outside the sandbox:
      `./.venv/bin/python scripts/record_provider_status_observations.py --environment dev --ping-count 1 --ping-timeout 2 --timeout 2`
      returned successful ICMP/DNS/TCP/TLS rows for `api.anthropic.com`,
      `api.openai.com`, `openrouter.ai`, `integrate.api.nvidia.com`,
      `generativelanguage.googleapis.com`, `cloudcode-pa.googleapis.com`, and
      control `google.com`.
  - Follow-up status: D1-103 tracks deployment, `--apply` DB verification,
    scheduling from an ICMP-capable host/sidecar, and the five-minute rollup
    query handoff. No DB writes were performed by the dry-run collector in this
    slice.

- D1-101 harness follow-up: native Codex/Gemini rate-limit and session-history
  gates
  - Goal: add local-ci harness coverage so native Codex and native Gemini
    canaries verify both `public.session_history` persistence and
    provider-originated `public.rate_limit_observations` rows, matching the
    Anthropic native telemetry gate pattern.
  - Changed paths:
    - `TEST_HARNESS.md`
    - `scripts/local-ci/README.md`
    - `scripts/local-ci/anthropic_adapter_config.json`
    - `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
  - Evidence:
    - `native_openai_passthrough_responses_codex` and
      `native_openai_passthrough_responses_codex_tool_activity` now require
      `codex_response_headers` metadata and validate OpenAI/Codex
      `codex:primary` five-hour plus `codex:secondary` seven-day
      `rate_limit_observations` rows.
    - `native_gemini_passthrough_generate_content` and
      `native_gemini_passthrough_stream_generate_content` now validate
      `session_history.provider=gemini` while checking Google Code Assist quota
      rows as `rate_limit_observations.provider=google`,
      `client=google_code_assist`, and
      `quota_key=google_code_assist_requests_gemini-2.5-flash:model_requests`.
    - Hardening tests now assert that native Codex/Gemini cases keep both
      session-history DB loading requirements and rate-limit DB validation
      requirements in the harness config.
    - `./.venv/bin/python -m json.tool scripts/local-ci/anthropic_adapter_config.json`
      passed.
    - `./.venv/bin/python -m py_compile scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
      passed.
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
      passed (`60 passed`, one pre-existing pytest config warning).
    - `git diff --check` passed.

- D1-100 harness follow-up: Anthropic rate-limit DB gate and fanout default
  policy
  - Goal: make the Anthropic adapter harness assert provider-originated
    `public.rate_limit_observations` for the native Anthropic smoke, and keep
    Claude subagent fanout cases out of the default suite.
  - Changed paths:
    - `TEST_HARNESS.md`
    - `scripts/local-ci/README.md`
    - `scripts/local-ci/anthropic_adapter_config.json`
    - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
    - `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
  - Evidence:
    - `scripts/local-ci/anthropic_adapter_config.json` keeps
      `claude_adapter_gemini_fanout`, `claude_adapter_peeromega_fanout`, and
      `native_anthropic_passthrough_claude` in `default_excluded_cases`.
    - `native_anthropic_passthrough_claude` now requires
      `anthropic_response_headers` metadata and validates Anthropic
      `anthropic_unified_5h:5h` / `anthropic_unified_7d:7d`
      `rate_limit_observations` rows with bounded timestamp/value checks when
      run explicitly.
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
      passed (`59 passed`, one pre-existing pytest config warning).
    - `./.venv/bin/python -m py_compile scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
      passed.
    - `git diff --check` passed.
  - Live validation caveat:
    - A focused dev run of `native_anthropic_passthrough_claude` reached the
      Claude CLI command but timed out before the new DB validator could run.
      A direct HTTP probe without cross-provider markers reached the native
      Anthropic route but returned upstream `401` because the dev runtime lacks
      server-side Anthropic API credentials for that path. Track the live proof
      as a separate follow-up rather than treating this as verified end to end.

- D1-100 follow-up: rate-limit observation timestamp/reset hardening and prod
  verification
  - Goal: stop stale reset rows and wall-clock backfill timestamps from
    contaminating `public.rate_limit_observations`, then prove the prod
    go-forward path with live Anthropic traffic instead of relying on a stale
    report view.
  - Changed paths:
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `.wheel-build/pyproject.toml`
    - `litellm/integrations/aawm_agent_identity.py`
    - `scripts/backfill_rate_limit_observations.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_scripts/test_backfill_rate_limit_observations.py`
  - Evidence:
    - Prod `aawm-litellm` reports `litellm=1.82.3+aawm.50`,
      `aawm-litellm-callbacks=0.0.27`, and
      `aawm-litellm-control-plane=0.0.7`; `/health/readiness` is healthy.
    - Callback synthetic smoke inside prod proves stale Codex reset headers are
      skipped when no relative reset hint exists (`0` observations), while stale
      `reset-at` plus `reset-after-seconds=3600` resolves to
      `observed_at=2026-05-14 14:49:01+00` and
      `provider_resets_at=2026-05-14 15:49:01+00`.
    - Exact database `aawm_tristore`: deleted `219` historical bad-time rows
      where `time-*` backfill `observed_at` was more than 60 seconds away from
      matching `session_history.end_time`.
    - Replayed ClickHouse backfill for
      `2026-05-14T00:00:00Z` through `2026-05-14T16:35:00Z`; script reported
      `candidate_records=8407`, `extracted_observations=32186`,
      `inserted_observations=31733`, `skipped_existing_observations=453`, and
      `source_errors=0`.
    - Deleted `26` duplicate tail rows from that replay, keeping live/non
      `time-*` rows first.
    - User-reported live issue was reproduced: in the two-hour window there
      were `287` Anthropic `session_history` rows but only `9` matching
      Anthropic observation-bearing traces before the focused recent repair.
    - Recent ClickHouse dry run for `2026-05-14T16:40:00Z` through
      `2026-05-14T17:00:00Z` found retained provider headers with
      `candidate_records=171` and `extracted_observations=677`.
    - Applied the same recent window; script reported
      `candidate_records=179`, `extracted_observations=703`, and
      `source_errors=0`. Follow-up cleanup deleted `9` duplicate tail rows
      where live and backfill observations differed only by call id or
      sub-millisecond timestamp precision.
    - Backfill dedupe now keys by source, storage provider, model, quota key,
      millisecond-truncated observed/reset timestamps, remaining pct, and
      trace id, instead of treating synthetic backfill call ids as distinct.
    - Raw retained Anthropic headers for trace
      `d72dadfa-21da-4a1f-8f79-012ad0650f5b` showed
      `anthropic-ratelimit-unified-5h-utilization=0.31`,
      `anthropic-ratelimit-unified-7d-utilization=0.0`, and reset timestamps
      `1778783400` / `1779375600`.
    - After later live Anthropic traffic and materialized view refresh,
      `public.rate_limit_intervals` shows current Anthropic intervals:
      short `68` from `2026-05-14 17:00:16.700282+00`, weekly `99` from
      `2026-05-14 17:00:33.518196+00`, and weekly_special `99` from the same
      timestamp.
    - Final exact database checks after cleanup: duplicate signature groups
      `0|0`, stale reset rows `0`, bad `time-*` rows `0`.
    - `public.rate_limit_intervals` was refreshed concurrently and analyzed in
      `aawm_tristore`; final refresh reported `count=2600`.
    - Prod log scan after deployment showed no `aawm_agent` or
      `session_history` errors. It did show an unrelated upstream Anthropic
      `404` for invalid model string `claude-opus-4-7[1m]`; that is not the
      rate-limit capture path.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "rate_limit_observations or rate_limit_meaningful or repeated_rate_limit or codex_response_headers or anthropic_unified or google_quota"`
      (`16 passed`, `129 deselected`, `1 warning`)
    - `./.venv/bin/python -m pytest tests/test_scripts/test_backfill_rate_limit_observations.py -q`
      (`7 passed`, `1 warning`)
    - `./.venv/bin/python -m py_compile scripts/backfill_rate_limit_observations.py tests/test_scripts/test_backfill_rate_limit_observations.py litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `git diff --check`
  - Follow-up:
    - The report materialized view currently filters `remaining_pct < 100`.
      That means exact-full post-reset weekly rows can bound neither the old
      interval nor the visible report state until a later `<100` row arrives.
      Do not alter the view definition without explicit agreement; the live
      provider data and view behavior need to stay discussed separately.

- D1-100 Anthropic rate-limit observation capture and historical backfill
  - Goal: investigate why recent Anthropic rate-limit details were not flowing
    into `public.rate_limit_observations`, fix the live capture path, and
    backfill retained missing observations.
  - Changed paths:
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `.wheel-build/pyproject.toml`
    - `PATCHES.md`
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `pyproject.toml`
    - `scripts/backfill_rate_limit_observations.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_scripts/test_backfill_rate_limit_observations.py`
  - Evidence:
    - Root cause had two parts: non-stream Anthropic passthrough success
      handling did not preserve upstream response rate-limit headers in
      callback metadata, and retained callback payloads could hide Anthropic
      headers under `_hidden_params.additional_headers` with
      `llm_provider-anthropic-ratelimit-*` names.
    - `success_handler.py` now records sanitized Anthropic non-stream response
      rate-limit headers in `metadata.anthropic_response_headers`.
    - `AawmAgentIdentity` now inspects hidden/header containers, accepts
      mapping-like header objects, parses raw and `llm_provider-`-prefixed
      Anthropic rate-limit headers, and includes those fields in error-payload
      extraction.
    - The callback wheel source remains byte-for-byte identical to the in-repo
      callback source.
    - Backfill marker scanning now includes `anthropic_response_headers`,
      `anthropic-ratelimit`, and `llm_provider-anthropic-ratelimit`.
    - Applied ClickHouse backfill to exact database `aawm_tristore` for
      retained observations from `2026-05-05T00:00:00Z` through
      `2026-05-14T14:35:00Z`; apply reported `candidate_records=7207`,
      `extracted_observations=27955`, `inserted_observations=27955`,
      `source_errors=0`, `applied=true`.
    - Post-backfill exact database verification showed Anthropic observation
      rows present through `2026-05-14 14:49:08.841056+00`; a later check
      showed `2273` Anthropic rows through `2026-05-14 14:52:57.724547+00`.
    - `public.rate_limit_intervals` was refreshed concurrently and analyzed in
      exact database `aawm_tristore`.
    - Fix commit `f16df28f9f` was pushed to `main` and `develop`; fork image
      workflow run `25867223817` published `v1.82.3-aawm.50` and
      `ghcr.io/zepfu/litellm:1.82.3-aawm.50`.
    - Artifact autobump run `25867210855` advanced `main` to `a2eb4c6e58` and
      tagged `cb-v0.0.26`; manual release recovery published
      `cb-v0.0.26` with asset
      `aawm_litellm_callbacks-0.0.26-py3-none-any.whl`.
    - `/home/zepfu/projects/aawm-infrastructure` commit `4dcefb1` pins prod to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.50`.
    - Rebuilt prod image `aawm-litellm:latest` as image id `5d05f2dd5e0e` and
      recreated prod container `dc96d643dff4` on `127.0.0.1:4000`.
    - Post-cutover exact database verification showed `23` Anthropic
      `public.rate_limit_observations` rows at or after
      `2026-05-14 15:10:00+00`, latest
      `2026-05-14 15:16:41.718051+00`, with rows for
      `anthropic_unified_5h:5h` and `anthropic_unified_7d:7d` sourced from
      `anthropic_response_headers`.
    - Post-cutover `public.rate_limit_intervals` refresh/analyze reported
      `current_database=aawm_tristore`, `count=2548`, and Anthropic
      `max(fromdate)=2026-05-14 15:13:01.394973+00`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_backfill_rate_limit_observations.py -q -k 'rate_limit or backfill_rate_limit or anthropic_quota'`
      (`21 passed`, `1 warning`)
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'anthropic_rate_limit_headers'`
      (`2 passed`, warnings only)
    - `./.venv/bin/python -m py_compile scripts/backfill_rate_limit_observations.py litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/proxy/pass_through_endpoints/success_handler.py`
    - `diff -q litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `git diff --check`
    - `docker run --rm --entrypoint python3 aawm-litellm:latest -c "..."`
      reported `1.82.3+aawm.50`, `0.0.26`, `0.0.7`.
    - `curl -sS http://127.0.0.1:4000/health/readiness` returned
      `status=healthy` and `litellm_version=1.82.3+aawm.50`.
    - `docker exec aawm-litellm python3 -c "..."`
      reported `1.82.3+aawm.50`, `0.0.26`, `0.0.7`.
    - `docker logs --since 2026-05-14T15:10:00Z aawm-litellm 2>&1 | rg -n "ERROR|Traceback|Exception|Content-Length|h11|KeyError|Langfuse Layer Error|NoneType|unexpected ASGI"`
      returned no matching lines.

- D1-099 `session_history` malformed repository identity repair
  - Goal: explain and repair the odd `repository` report values where
    free-text rollout descriptors and JSON-schema fragments were being written
    into `public.session_history.repository` and sometimes copied into
    repository-derived `tenant_id`.
  - Changed paths:
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `scripts/repair_session_history_repository_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - Root cause: `_normalize_repository_identity()` accepted any non-empty
      string after URL/path cleanup, so `metadata.repository` values like
      `rollout-2026-05-10...jsonl, updated_at=..., thread_id=...` and
      `{'anyOf': ...}` were treated as repository IDs.
    - Live pre-repair count in exact database `aawm_tristore` showed `3,875`
      malformed `public.session_history.repository` values; the dominant
      malformed value was the JSON-schema text with `3,641` rows.
    - Added strict repository identity validation for repo slugs, `owner/repo`,
      and the existing ` (memory)` suffix; non-string values and strings with
      spaces, commas, schema syntax, rollout descriptors, or CLI option text are
      rejected.
    - Added `scripts/repair_session_history_repository_identity.py` to repair
      historical rows using the same normalizer, recover obvious known local
      repo prefixes, preserve valid non-repo tenant IDs, and remove invalid
      `metadata.repository` keys.
    - Historical repair applied to `3,877` rows in
      `aawm_tristore.public.session_history`; follow-up passes repaired `2` and
      then `147` prod rows that arrived before callback `0.0.24` was serving.
    - Hotfix commit `3878c1a783` was pushed to `main`; artifact autobump run
      `25839627206` advanced `main` to `7bd79f127f` and tagged `cb-v0.0.24`.
    - Manual release recovery published `cb-v0.0.24` with
      `aawm_litellm_callbacks-0.0.24-py3-none-any.whl`; `develop` was
      fast-forwarded to the same commit.
    - Rebuilt the existing prod image on base
      `ghcr.io/zepfu/litellm:1.82.3-aawm.49`; local image id is
      `a07aceecfb05`.
    - Prod was recreated and is running container `d1a51fdc2b4a` on
      `127.0.0.1:4000`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      (`140 passed`, `1 warning`)
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/repair_session_history_repository_identity.py`
    - `diff -q litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `git diff --check`
    - `gh release view cb-v0.0.24 --repo zepfu/litellm` showed asset
      `aawm_litellm_callbacks-0.0.24-py3-none-any.whl`.
    - `docker run --rm --entrypoint python3 aawm-litellm:latest -c "..."`
      reported `1.82.3+aawm.49`, `0.0.24`, `0.0.7`.
    - `curl -sS http://127.0.0.1:4000/health/readiness` returned
      `status=healthy`.
    - `docker exec aawm-litellm python3 -c "..."`
      reported `1.82.3+aawm.49`, `0.0.24`, `0.0.7`.
    - Running-container normalizer smoke returned `zepfu/litellm` for a GitHub
      URL and `None` for rollout descriptor and dict/schema inputs.
    - `scripts/repair_session_history_repository_identity.py --dsn ... --apply`
      reported `database=aawm_tristore`, `candidate_rows=3877`,
      `repairable_rows=3877`, `applied=true`; follow-up passes reported
      `candidate_rows=2` and then `candidate_rows=147`, both `applied=true`.
    - Final exact database checks showed malformed counts of `0` for
      `repository`, `tenant_id`, and `metadata.repository`.
  - Follow-up:
    - Default prod harness still has the pre-existing Spark/Codex quota reset
      follow-up under D1-098; this callback hotfix did not rerun that full
      suite.

## 2026-05-13

- D1-097 aawm.49 prod cutover with callback hotfix `0.0.23`
  - Goal: cut over prod `:4000` to the prepared `aawm.49` release, repair
    validation-discovered native attribution issues, and leave prod running the
    corrected callback overlay.
  - Changed paths:
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `.wheel-build/pyproject.toml`
    - `COMPLETED.md`
    - `TODO.md`
    - `litellm/integrations/aawm_agent_identity.py`
    - `scripts/local-ci/anthropic_adapter_config.json`
    - `scripts/local-ci/harness-version.txt`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - Initial prod native shard found a real callback issue: Codex carried
      `x-aawm-repository="zepfu/litellm"` but a later row stored
      `repository=aegis-dashboard`, and Gemini CLI `0.42.0` identified as
      `GeminiCLI-tui`.
    - Hotfix commit `1801e2e6d1` made repository headers authoritative before
      free-text inference and normalized `GeminiCLI(?:-tui)?/<version>` to
      `gemini-cli`.
    - Artifact autobump run `25835631268` advanced `main` to `94601b242e` and
      tagged `cb-v0.0.23` and `h-v0.0.29`.
    - Manual release recovery published `cb-v0.0.23` with
      `aawm_litellm_callbacks-0.0.23-py3-none-any.whl` and `h-v0.0.29` with
      `litellm-local-ci-harness-0.0.29.tar.gz`.
    - Rebuilt prod image installed `aawm-litellm-callbacks==0.0.23`; local
      image id is `5d0fb153886a`.
    - Prod was recreated and is running container `1563b4463f3d` on
      `127.0.0.1:4000`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      (`129 passed`, `1 warning`)
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
      (`56 passed`, `1 warning`)
    - `git diff --check`
    - `docker run --rm --entrypoint python3 aawm-litellm:latest -c "..."`
      reported `1.82.3+aawm.49`, `0.0.23`, `0.0.7`.
    - `curl -sS http://127.0.0.1:4000/health/readiness` returned
      `status=healthy`.
    - `docker exec aawm-litellm python3 -c "..."`
      reported `1.82.3+aawm.49`, `0.0.23`, `0.0.7`.
    - Native prod shard
      `/tmp/litellm-prod-native-aawm49-cb23.json` passed with zero failures
      and warnings; Codex stored `repository=zepfu/litellm` and Gemini stored
      `client_name=gemini-cli`, `client_version=0.42.0`.
    - Default prod harness
      `/tmp/litellm-prod-harness-aawm49-cb23.json` failed only on
      Spark/Codex `usage_limit_reached` reset `2026-05-18 15:08:41 UTC` plus
      an overlapping transient Codex `503` log attached to
      `claude_adapter_gpt54_mini`; focused rerun
      `/tmp/litellm-prod-gpt54-mini-aawm49-cb23-rerun.json` passed.
    - Final filtered prod log scan returned no release-blocking patterns.

- D1-097 aawm.49 prod release prep
  - Goal: prepare the next LiteLLM fork release for prod cutover without
    restarting prod.
  - Changed paths:
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `.wheel-build/pyproject.toml`
    - `COMPLETED.md`
    - `PATCHES.md`
    - `TODO.md`
    - `WHEEL.md`
    - `litellm-dev-config.yaml`
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `pyproject.toml`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
  - Evidence:
    - Release candidate commit `a38d79fc25` was pushed to `main`.
    - Artifact autobump run `25833914450` succeeded, advanced `main` and
      `develop` to `6dfff5e366`, and tagged `cb-v0.0.22`.
    - Fork image publish run `25833977035` succeeded for
      `v1.82.3-aawm.49`, publishing `ghcr.io/zepfu/litellm:1.82.3-aawm.49`
      and creating the GitHub Release.
    - The callback GitHub Release `cb-v0.0.22` was manually created after the
      autobump tag, and the wheel asset
      `aawm_litellm_callbacks-0.0.22-py3-none-any.whl` was uploaded.
    - Existing overlay releases verified: `cp-v0.0.7`, `cfg-v0.0.10`, and
      `h-v0.0.28`.
    - Prod `:4000` was not restarted.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `diff -q .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/integrations/aawm_agent_identity.py`
    - `./.venv/bin/python -c "import yaml; yaml.safe_load(open('litellm-dev-config.yaml', encoding='utf-8')); print('yaml-ok')"`
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      (`434 passed`, `74 warnings`)
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/integrations/aawm_agent_identity.py`
    - `git diff --check`
    - `curl -sS http://127.0.0.1:4001/health/readiness`
    - `docker logs --tail 500 litellm-dev` filtered for release-blocking error
      patterns returned no matches.
    - `psql ... select count(*), max(fromdate), pg_size_pretty(pg_total_relation_size('public.rate_limit_intervals')) from public.rate_limit_intervals;`
      returned `1687|2026-05-13 23:59:52.907187+00|680 kB`.
    - `psql ... select jobid, jobname, schedule, command from cron.job where jobname like 'aawm_rate_limit_intervals%' order by jobid;`
      returned jobs `24` and `25`.

- D1-093 rate-limit interval materialized-view scheduled refresh
  - Goal: keep `public.rate_limit_intervals` current for the reusable
    session-history rate-limit usage report without requiring a manual refresh
    before every run.
  - Changed paths:
    - `.analysis/session-history-rate-limit-usage-materialized.sql`
  - Evidence:
    - Ran a manual `REFRESH MATERIALIZED VIEW CONCURRENTLY
      public.rate_limit_intervals` against exact database `aawm_tristore`, then
      `ANALYZE public.rate_limit_intervals`.
    - Installed pg_cron job `24`, `aawm_rate_limit_intervals_refresh`, in
      `aawm_tristore` as user `aawm`, active schedule `*/30 * * * *`, command
      `REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals`.
    - Installed pg_cron job `25`, `aawm_rate_limit_intervals_analyze`, in
      `aawm_tristore` as user `aawm`, active schedule `1,31 * * * *`, command
      `ANALYZE public.rate_limit_intervals`.
    - Post-refresh verification showed `1629` rows, latest interval
      `fromdate=2026-05-13 18:09:15.036459+00`, and total relation size
      `656 kB`.
  - Verification:
    - `psql -v ON_ERROR_STOP=1 -Atqc "REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
    - `psql -v ON_ERROR_STOP=1 -Atqc "ANALYZE public.rate_limit_intervals;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
    - `psql -v ON_ERROR_STOP=1 -Atqc "select cron.unschedule(jobname) from cron.job where jobname in ('aawm_rate_limit_intervals_refresh','aawm_rate_limit_intervals_analyze'); select cron.schedule_in_database('aawm_rate_limit_intervals_refresh', '*/30 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals', 'aawm_tristore', 'aawm', true); select cron.schedule_in_database('aawm_rate_limit_intervals_analyze', '1,31 * * * *', 'ANALYZE public.rate_limit_intervals', 'aawm_tristore', 'aawm', true);" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
    - `psql -Atqc "select jobid, jobname, schedule, database, username, active, command from cron.job where jobname in ('aawm_rate_limit_intervals_refresh','aawm_rate_limit_intervals_analyze') order by jobname;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`

- D1-096 Codex auto-agent model alias with quota-aware fallback
  - Goal: expose one Codex-visible model alias,
    `aawm-codex-agent-auto`, that normal Codex sessions and subagent fanout can
    use while LiteLLM selects the first healthy Spark/Gemini target and only
    falls back to `gpt-5.4-mini` after the preferred pool is exhausted.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `/home/zepfu/.codex/aawm-model-catalog.json`
  - Evidence:
    - Added alias candidate order:
      `gpt-5.3-codex-spark`,
      `gemini-3.1-flash-lite-preview`,
      `gemini-3-flash-preview`,
      `gemini-3.1-pro-preview`, and last-resort `gpt-5.4-mini`.
    - Added in-memory cooldowns keyed by actual provider/model/lane and
      session affinity keyed from Codex session plus provider lane context.
    - Retryable pre-stream exhaustion now covers `429`,
      `usage_limit_reached`, `RESOURCE_EXHAUSTED`,
      `MODEL_CAPACITY_EXHAUSTED`, and `RATE_LIMIT_EXCEEDED`.
    - Alias-routed Gemini candidates override the normal Google adapter hidden
      retry budget to one probe per candidate, so the alias router can move to
      the next target instead of spending the same quota failure repeatedly.
    - Added session-history metadata allowlist fields:
      `requested_model_alias`, selected provider/model/route family,
      last-resort flag, selection reason, lane key, attempted candidates, and
      skipped candidates.
    - Restarted `litellm-dev` on `:4001`; `/health/readiness` returned
      `status=healthy` with `AawmAgentIdentity` loaded.
    - Live Codex smoke through dev:
      `codex exec --profile litellm-dev -m aawm-codex-agent-auto -c model_catalog_json='"/home/zepfu/.codex/aawm-model-catalog.json"' "Reply exactly: d1-096-auto-alias-smoke-2"`
      returned `d1-096-auto-alias-smoke-2` for session
      `019e217d-079d-73d1-9287-96a46780ca50`.
    - Dev logs for that smoke showed Spark cooled down with
      `usage_limit_reached`, `gemini-3.1-flash-lite-preview` failed once with
      `429` / `QUOTA_EXHAUSTED`, and the request then completed on
      `gemini-3-flash-preview`.
    - Exact database `aawm_tristore.public.session_history` row `319096`
      recorded `provider=gemini`, `model=gemini-3-flash-preview`,
      `session_id=019e217d-079d-73d1-9287-96a46780ca50`,
      `requested_model_alias=aawm-codex-agent-auto`,
      `codex_auto_agent_selected_provider=google_code_assist`,
      `codex_auto_agent_selected_model=gemini-3-flash-preview`, and
      `codex_auto_agent_selected_route_family=codex_google_code_assist_adapter`.
    - Row `319096` also stored attempted candidates showing
      `gpt-5.3-codex-spark` cooldown `7431.226s`,
      `gemini-3.1-flash-lite-preview` cooldown `300.0s`, and final selected
      `gemini-3-flash-preview`; skipped candidates contained the cooled-down
      Spark and flash-lite targets.
  - Verification:
    - `codex debug models -c model_catalog_json='"/home/zepfu/.codex/aawm-model-catalog.json"'`
      showed `aawm-codex-agent-auto` alongside Spark, `gpt-5.4-mini`, and the
      three Gemini preview models.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_auto_agent_alias or google_adapter_request_honors_inline_retry_overrides or google_adapter_request_strips_internal_wrapper_kwargs'`
      (`9 passed, 298 deselected`)
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q`
      (`307 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'codex_google_code_assist or codex_adapter_quota or google_quota_buckets or session_history_metadata'`
      (`3 passed, 124 deselected`)
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/integrations/aawm_agent_identity.py`
    - `diff -q .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/integrations/aawm_agent_identity.py`
    - `psql -Atqc "select id, created_at, provider, model, tenant_id, session_id, metadata->>'requested_model_alias', metadata->>'codex_auto_agent_selected_provider', metadata->>'codex_auto_agent_selected_model', metadata->>'codex_auto_agent_selected_route_family' from public.session_history where session_id in ('019e2178-54b4-7590-b002-da9a42dd2a9b','019e217d-079d-73d1-9287-96a46780ca50') or metadata->>'requested_model_alias'='aawm-codex-agent-auto' order by created_at desc limit 12;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`

- D1-094 local biomedical pass-through routes on dev LiteLLM
  - Goal: expose the TAP local scispaCy and TinyBERN2 sidecars through dev
    LiteLLM and ensure successful calls are attributed in
    `public.session_history` instead of falling back to `unknown`.
  - Changed paths:
    - `litellm-dev-config.yaml`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - sibling upstream fix: `/home/zepfu/projects/aawm-tap/scripts/dev/tinybern2_sidecar_service.py`
  - Evidence:
    - Added dev pass-through routes:
      `/aawm/local/scispacy/* -> http://172.20.0.1:8094/*` and
      `/aawm/local/tinybern2/* -> http://172.20.0.1:8095/*`.
    - Built and started TAP sidecars
      `aawm-tap-biomed-scispacy-dev` and
      `aawm-tap-biomed-tinybern2-dev`; both health checks are healthy and
      reachable from the `litellm-dev` container via `172.20.0.1`.
    - Fixed the sibling TinyBERN2 sidecar crash-loop syntax error
      `except TypeError, ValueError` -> `except (TypeError, ValueError)`.
    - Restarted `litellm-dev` on `:4001`; `/health/readiness` returned
      `status=healthy` with `AawmAgentIdentity` loaded.
    - Dev pass-through health routes returned `status=ok` for both
      `/aawm/local/scispacy/health` and `/aawm/local/tinybern2/health`.
    - scispaCy extraction through dev LiteLLM returned biomedical mentions and
      wrote exact database `aawm_tristore.public.session_history` row `318432`
      with `session_id=local-biomed-scispacy-smoke-20260513-v2`,
      `provider=local_biomed`, `model=scispacy`,
      `model_group=scispacy`, `call_type=pass_through_endpoint`,
      `repository=litellm`, `passthrough_route_family=local_biomed`,
      `aawm_local_route_family=local_biomed_rest`,
      `aawm_local_service=scispacy`, `aawm_local_endpoint=extract`, and
      `aawm_local_upstream_api_base=http://172.20.0.1:8094/extract`.
    - TinyBERN2 `/annotate` is reachable through LiteLLM but currently returns
      upstream `503` with `TINYBERN2_MODEL_ID is not configured`; no success
      `session_history` row is expected until that sidecar model id is set.
  - Verification:
    - `./.venv/bin/python -c "import yaml; data=yaml.safe_load(open('litellm-dev-config.yaml')); pts=data['general_settings']['pass_through_endpoints']; assert {p['path'] for p in pts} >= {'/aawm/local/scispacy','/aawm/local/tinybern2'}; print('dev passthrough config ok')"`
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` (`127 passed`)
    - `curl -sS http://127.0.0.1:4001/aawm/local/scispacy/health`
    - `curl -sS http://127.0.0.1:4001/aawm/local/tinybern2/health`
    - `curl -sS -X POST http://127.0.0.1:4001/aawm/local/scispacy/extract ...`
    - `psql -Atqc "SELECT id, session_id, provider, model, model_group, call_type, repository, metadata->>'passthrough_route_family', metadata->>'aawm_local_route_family', metadata->>'aawm_local_service', metadata->>'aawm_local_endpoint', metadata->>'aawm_local_upstream_api_base' FROM public.session_history WHERE session_id IN ('local-biomed-scispacy-smoke-20260513-v2','local-biomed-tinybern2-smoke-20260513') ORDER BY id DESC;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`

- D1-091 Codex `aegis-dashboard` session-history repository attribution fix
  - Goal: explain why new Codex traffic from `/home/zepfu/projects/aegis-dashboard`
    was not visible under that repository in `public.session_history`, and
    prevent current Codex workspaces from being misattributed to stale prior
    repository context.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - Exact database `aawm_tristore.public.session_history` did have fresh rows
      through `2026-05-13 01:06:43+00`, but recent rows grouped under
      `tenant_id=aawm-tap`, not `aegis-dashboard`.
    - A controlled native smoke from `/home/zepfu/projects/aegis-dashboard`
      using `codex exec --profile litellm -m gemini-3.1-flash-lite-preview`
      succeeded with session `019e1ee0-d589-7940-a03a-23c26a1d8251`, but
      row `311500` wrote as `tenant_id=aawm-tap`, `repository=aawm-tap`,
      proving misattribution rather than missing writes.
    - The same smoke printed Codex's warning that
      `/home/zepfu/projects/aegis-dashboard/.codex/config.toml` ignored
      project-local `model_providers`, so repo-specific provider headers there
      were not active.
    - Langfuse trace `da31b9d3-202e-4041-82dc-06a187fee980` showed the request
      contained the correct current
      `<cwd>/home/zepfu/projects/aegis-dashboard</cwd>`, while metadata still
      carried `repository=aawm-tap`.
    - Fixed repository inference to prioritize current
      `<environment_context><cwd>...</cwd>` matches and to scan list-shaped
      conversation payloads from newest to oldest before falling back to older
      repository text.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'passthrough_observability' -q`
      passed with `5 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -k 'repository_from_codex_workspace_text or current_codex_cwd' -q`
      passed with `2 passed`.
    - Restarted dev `litellm-dev` on `:4001`; `/health/readiness` returned
      `status=healthy` with `AawmAgentIdentity` loaded.
    - Controlled dev smoke from `/home/zepfu/projects/aegis-dashboard` using
      `codex exec --profile litellm-dev -m gemini-3.1-flash-lite-preview`
      succeeded with session `019e1ee6-f48d-78e0-a760-1e8ac8fc4bf5`.
    - Exact database `aawm_tristore.public.session_history` row `311561`
      recorded that smoke as `tenant_id=aegis-dashboard`,
      `repository=aegis-dashboard`, `client_name=codex_exec`,
      `provider=gemini`, `model=gemini-3.1-flash-lite-preview`,
      `litellm_environment=dev`, and `tenant_id_source=repository`.
  - Follow-up:
    - The fix is code-complete and dev-validated, but not yet deployed to prod.
      Once included in the next normal prod release/restart, adding future repos
      should not require per-repo proxy restarts or per-repo global provider
      config; attribution should fall back from the current Codex cwd.
  - Historical repair:
    - Scanned Langfuse ClickHouse traces since `2026-04-01` by extracting the
      newest `<cwd>...</cwd>` under `/home/zepfu/projects/` and comparing it to
      exact database `aawm_tristore.public.session_history`.
    - Confirmed `6,411` trace-backed project-cwd rows across `27` sessions.
      Before repair, `2,667` normal rows and `59` memory-labeled rows were
      semantically misattributed, primarily to stale `aawm-tap` context.
    - Repaired `2,726` rows using the trace current cwd: `1,358` to `litellm`,
      `962` to `aegis`, `153` to `aawm-tap-dashboard`, `152` to
      `dashboard-shell`, `95` to `aegis-dashboard`, and `6` to `aawm`.
      Memory rows retained the `(memory)` suffix and had `source_repository`
      updated to the cwd repo.
    - Verification after repair showed `remaining_semantic_mismatches=0`.
      Known prod smoke row `311500` now records `tenant_id=aegis-dashboard`,
      `repository=aegis-dashboard`, and
      `metadata.repository_repair_source=langfuse_current_cwd`; dev smoke row
      `311561` was already correct.

- D1-092 `session_history` latest-query performance index
  - Goal: make current/recent `public.session_history` reads stop scanning and
    sorting the whole table for `ORDER BY created_at DESC LIMIT ...` queries.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
  - Evidence:
    - Exact database `aawm_tristore.public.session_history` currently has about
      `250k` rows; before the fix the representative latest-history query used
      a parallel seq scan plus top-N sort and executed in `221.376 ms` on a
      warm cache.
    - Added schema bootstrap index
      `session_history_created_at_idx ON public.session_history (created_at DESC)`
      and applied it live with `CREATE INDEX CONCURRENTLY IF NOT EXISTS`.
    - The live index exists at `5,272 kB`.
  - Verification:
    - Re-running
      `EXPLAIN (ANALYZE, BUFFERS) SELECT id, created_at, session_id, repository, provider, model FROM public.session_history ORDER BY created_at DESC LIMIT 20;`
      used `Index Scan using session_history_created_at_idx` and executed in
      `0.200 ms`.
    - Focused regression tests for D1-091 still pass:
      `5 passed` for pass-through observability and `2 passed` for
      session-history repository inference.

- D1-093 rate-limit usage report query tuning
  - Goal: make the combined `session_history` / `rate_limit_observations`
    usage query return in seconds instead of minutes while preserving the
    original rate interval semantics.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `.analysis/session-history-rate-limit-usage-optimized.sql`
  - Evidence:
    - Original query plan in exact database `aawm_tristore` spent
      `159.880 s` under `EXPLAIN ANALYZE`.
    - The bottleneck was four nested interval joins over the materialized
      `rate_intervals` CTE, with rejected join rows of about `54M`, `239M`,
      `20M`, and `76M`; the rate-limit CTE itself was only about `25 ms`.
    - Added live/schema bootstrap indexes:
      `rate_limit_observations_provider_quota_observed_idx` and
      `rate_limit_observations_provider_type_model_observed_idx`.
    - A direct latest-snapshot lateral rewrite used those indexes and ran in
      `5.139 s`, but produced slightly different grouping semantics from the
      original interval CTE.
    - The exact-semantics optimized SQL saved at
      `.analysis/session-history-rate-limit-usage-optimized.sql` materializes
      the original interval CTE into a temporary indexed table, then uses
      one-row lateral lookups.
    - Follow-up after pgAdmin testing: the reusable no-temp path is a durable
      materialized view, `public.rate_limit_intervals`, with indexes
      `rate_limit_intervals_type_provider_from_idx`,
      `rate_limit_intervals_requests_idx`, and
      `rate_limit_intervals_unique_idx`. The view/index bootstrap is now in
      `litellm/integrations/aawm_agent_identity.py`, and the pgAdmin-ready SQL
      is saved at
      `.analysis/session-history-rate-limit-usage-materialized.sql`.
  - Verification:
    - Temp interval setup built `1,484` interval rows in under `0.5 s`.
    - Full exact-semantics query shape ran in `4.609 s` under
      `EXPLAIN ANALYZE`.
    - Aggregate wrapper over the final query returned `2,290` grouped rows and
      `262,176` contributing traces in `3.171 s` after setup, matching the
      original grouped row count observed in the slow plan.
    - Live materialized-view setup in exact database `aawm_tristore` built
      `1,512` interval rows, total relation size about `616 kB`; the no-temp
      materialized-view query ran in `4.254 s` under `EXPLAIN ANALYZE`.
    - Focused regression tests still pass: `5 passed` for pass-through
      observability and `2 passed` for session-history repository inference.

## 2026-05-12

- D1-090 aawm.48 prod cutover for Codex-native Gemini child-agent hardening
  - Goal: restart prod `:4000` onto the already-prepared `aawm.48` image and
    prove the Codex-native Gemini hardened route writes correct telemetry.
  - Evidence:
    - Ran `docker compose -f docker-compose.litellm.yml up -d --no-build --force-recreate --no-deps litellm`
      in `/home/zepfu/projects/aawm-infrastructure`.
    - Running prod container is `61c1c73e5e81`, healthy on
      `127.0.0.1:4000`, from local image `aawm-litellm:latest` image id
      `c584b3c8eee5`.
    - Running-container package inspection reported
      `litellm=1.82.3+aawm.48`,
      `aawm-litellm-callbacks=0.0.21`, and
      `aawm-litellm-control-plane=0.0.7`.
    - `/health/readiness` returned `status=healthy`,
      `litellm_version=1.82.3+aawm.48`, and success callbacks including
      `LangfusePromptManagement` and `AawmAgentIdentity`; `/health/liveliness`
      returned `"I'm alive!"`.
    - Native prod-profile exact-response smoke passed:
      `codex exec --profile litellm -m gemini-3.1-flash-lite-preview "Reply exactly: prod-gemini-aawm48-smoke"`
      returned exactly `prod-gemini-aawm48-smoke` with session
      `019e1ed2-4107-76c2-923a-bda26eb50887`.
    - Exact database `aawm_tristore.public.session_history` row `311133`
      recorded the exact-response smoke with `provider=gemini`,
      `model=gemini-3.1-flash-lite-preview`, `client_name=codex_exec`,
      `litellm_environment=prod`, `litellm_version=1.82.3+aawm.48`,
      route family `codex_google_code_assist_adapter`, tool-contract policy
      `append`, applied `true`, and version `2026-05-12.v1`.
    - Native prod-profile tool-use smoke passed with session
      `019e1ed3-6cc4-75f3-b80c-a106cb3b50dc`; it ran
      `printf 'D1-089 prod smoke after cutover'` and returned the exact
      command/output.
    - Exact database `aawm_tristore.public.session_history` rows `311143`
      through `311146` recorded the tool-use smoke with the same provider,
      model, environment, route, and policy metadata.
    - `public.session_history_tool_activity` stored structured `exec_command`
      rows for `ls -F .analysis/todo.md`, `cat .analysis/todo.md`, and
      `printf 'D1-089 prod smoke after cutover'`; malformed transcript-command
      check returned `0`.
    - `public.rate_limit_observations` recorded matching
      `google/google_code_assist/google_retrieve_user_quota` rows for the smoke
      sessions.
    - Post-smoke logs showed successful 200 responses on the Codex Google Code
      Assist route. Two transient Google adapter `429` warnings appeared during
      the tool-use smoke, were retried with cooldown, and did not fail the
      request. No release-blocking `model_not_found`, invalid reasoning effort,
      Langfuse layer error, `NoneType`, traceback, or pass-through 400 appeared
      in the inspected logs.

- D1-089 aawm.48 prod-ready release prep for Codex-native Gemini child-agent
  hardening
  - Goal: publish the LiteLLM release artifacts and prepare the production
    image for the D1-086 child-agent tool-contract hardening without restarting
    prod `:4000`.
  - Changed paths:
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `PATCHES.md`
    - `docker-compose.dev.yml`
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `pyproject.toml`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `COMPLETED.md`
    - `TODO.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - Release candidate commit `30c6983849` was pushed to `main`.
    - Artifact autobump run `25770463131` succeeded, advanced `main` to
      `ea34b28ff4`, and tagged `cb-v0.0.21`.
    - `develop`, `main`, `origin/develop`, and `origin/main` were converged at
      `ea34b28ff4`; fork release tag `v1.82.3-aawm.48` was cut from that
      commit.
    - GitHub Actions run `25770532050` succeeded and published
      `ghcr.io/zepfu/litellm:1.82.3-aawm.48`.
    - GitHub Releases/assets verified:
      `v1.82.3-aawm.48`, `cb-v0.0.21` with
      `aawm_litellm_callbacks-0.0.21-py3-none-any.whl`, `cp-v0.0.7`,
      `cfg-v0.0.10`, and `h-v0.0.28`.
    - `/home/zepfu/projects/aawm-infrastructure` commit `2ba93fd` pins the
      standalone LiteLLM build to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.48`.
    - `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      completed without recreating prod and produced `aawm-litellm:latest`
      image id `c584b3c8eee5`.
    - Direct image inspection reported `litellm=1.82.3+aawm.48`,
      `aawm-litellm-callbacks=0.0.21`, and
      `aawm-litellm-control-plane=0.0.7`.
    - Running prod was not restarted: `docker ps --filter name=aawm-litellm`
      still showed container `14ce5872fab4`, image `ae399237b0b2`, healthy on
      `127.0.0.1:4000`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `422 passed`.
    - `./.venv/bin/python -m py_compile .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/integrations/aawm_agent_identity.py litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
      passed.
    - Targeted Ruff on the touched pass-through/callback/test files passed.
    - `git diff --check` passed in both repos.
  - Follow-up:
    - Next explicit step after user approval is prod restart/recreate from the
      already-built `aawm-litellm:latest` image, followed by runtime readiness,
      prod Gemini smoke, DB attribution check, and log scan.

- D1-086 Codex-native Gemini child-agent tool-call schema hardening
  - Goal: reduce low-confidence Gemini child-agent outcomes where Gemini copies
    a prior terminal/tool result into the next tool-call arguments or drifts
    into a generic summary after a tool-result turn.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `docker-compose.dev.yml`
    - `.analysis/codex-native-gemini-child-agent-hardening-2026-05-12.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation:
    - Added a Codex-native Google Code Assist tool-contract prompt overlay,
      controlled by `AAWM_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY`
      (`append` by default, `off` supported).
    - Scoped the overlay to `completion_kwargs_are_openai_chat=True`, so the
      Anthropic-to-Google adapter path is not contaminated with Codex-specific
      wording.
    - Recorded policy name, mode, version, applied flag, prompt char count, and
      policy tags into request metadata and allowed those keys through
      `session_history`.
    - Mounted the current Langfuse handler/prompt-management files in
      `docker-compose.dev.yml` so dev `:4001` uses the already-fixed
      missing-dynamic-params guard during live smokes.
  - Evidence:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestGoogleNativeToolAliases::test_codex_google_tool_contract_policy_appends_to_system_prompt tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestGoogleNativeToolAliases::test_codex_google_tool_contract_policy_off_does_not_rewrite_prompt tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestAnthropicAdapterClaudeCodeAgentProjectMetadata::test_codex_google_code_assist_builder_accepts_openai_chat_tool_choice tests/test_litellm/integrations/test_aawm_agent_identity.py::test_build_session_history_record_uses_codex_google_code_assist_metadata -q`
      (`4 passed`).
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      (`422 passed`, existing warnings only).
    - Recreated dev with `docker compose -f docker-compose.dev.yml up -d litellm-dev`;
      `/health/readiness` on `127.0.0.1:4001` returned `status=healthy`.
    - Native dev-profile Gemini smoke passed:
      `codex exec --profile litellm-dev -m gemini-3.1-flash-lite-preview "Use the shell tool to run: printf 'D1-086 dev smoke after mount'. Then answer with the exact command and exact output. Keep it brief."`
      returned the expected command/output with session
      `019e1e36-199a-7762-960b-4e8824a74afb`.
    - Dev database `aawm_tristore.public.session_history` wrote rows `310540`
      and `310541` for session `019e1e36-199a-7762-960b-4e8824a74afb` as
      `provider=gemini`, `model=gemini-3.1-flash-lite-preview`, with metadata
      `codex_google_code_assist_tool_contract_policy=append`,
      `codex_google_code_assist_tool_contract_policy_applied=true`, and
      version `2026-05-12.v1`.
    - Dev database `public.session_history_tool_activity` stored the tool
      command as `printf 'D1-086 dev smoke after mount'`, not prior terminal
      transcript text.
    - Malformed Gemini tool-activity check for rows at or after
      `2026-05-12T22:02:00Z` returned `0` rows matching `Chunk ID:%`,
      `Wall time:%`, `Process exited with code%`, or `Output:%` as command
      text.
    - `public.rate_limit_observations` recorded matching Google Code Assist
      quota rows for `gemini-3.1-flash-lite-preview`, latest observed at
      `2026-05-12 22:02:08.503268+00` with
      `quota_key=google_code_assist_requests_gemini-3.1-flash-lite-preview:model_requests`.
    - Post-recreate `docker logs --tail 80 litellm-dev` showed the Gemini
      `streamGenerateContent` HTTP 200s and no `Langfuse Layer Error` for the
      smoke; only the expected trace-header overwrite warnings remained.
  - Follow-up:
    - Prod promotion is still pending explicit release go-ahead; this entry is
      dev validation only.

- D1-088 aawm.47 prod cutover for Codex-native Gemini
  - Goal: cut prod `:4000` over to the Codex-native Gemini Code Assist adapter
    and clear the active prod log errors exposed by live smoke testing.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/langfuse/langfuse_prompt_management.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/integrations/langfuse/test_langfuse_prompt_management.py`
    - `pyproject.toml`
    - `COMPLETED.md`
    - `TODO.md`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - Initial `aawm.45` prod cutover exposed Codex `reasoning_effort=xhigh`
      rejection on the Gemini adapter; `aawm.46` commit `0a10837148` normalized
      Codex Gemini `xhigh` to `high` and GitHub Actions run `25756766335`
      published `v1.82.3-aawm.46`.
    - The successful `aawm.46` Gemini smoke exposed non-fatal Langfuse callback
      errors when `standard_callback_dynamic_params` was missing; `aawm.47`
      commit `bdf2a2be6e` defaults missing dynamic params to `{}` for Langfuse
      success/failure logging and GitHub Actions run `25757854966` published
      `v1.82.3-aawm.47`.
    - `/home/zepfu/projects/aawm-infrastructure` pins the standalone LiteLLM
      build to `ghcr.io/zepfu/litellm:1.82.3-aawm.47`.
    - Local `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      completed and produced `aawm-litellm:latest` image id `ae399237b0b2`.
    - Image and running-container inspection reported
      `litellm=1.82.3+aawm.47`,
      `aawm-litellm-callbacks=0.0.20`,
      `aawm-litellm-control-plane=0.0.7`, and `langfuse_none_guard=True`.
    - Prod was recreated as container `14ce5872fab4`, healthy on
      `127.0.0.1:4000`; `/health/readiness` returned `status=healthy`,
      `litellm_version=1.82.3+aawm.47`, and success callbacks including
      `LangfusePromptManagement` and `AawmAgentIdentity`.
    - Native prod-profile smoke
      `codex exec --profile litellm -m gemini-3.1-flash-lite-preview "Reply exactly: prod-gemini-aawm47-smoke"`
      returned exactly `prod-gemini-aawm47-smoke` with session
      `019e1dbe-fea3-73f3-8a71-44ade814aa78` while Codex reported
      `reasoning effort: xhigh`.
    - Exact database `aawm_tristore.public.session_history` row `308938`
      recorded the smoke with `provider=gemini`,
      `model=gemini-3.1-flash-lite-preview`, `client_name=codex_exec`,
      `litellm_environment=prod`, `litellm_version=1.82.3+aawm.47`,
      `metadata.passthrough_route_family=codex_google_code_assist_adapter`, and
      `metadata.codex_adapter_model=gemini-3.1-flash-lite-preview`.
    - Exact database `aawm_tristore.public.rate_limit_observations` recorded
      same-session `google/google_code_assist/google_retrieve_user_quota` rows
      for Gemini Code Assist request quotas, including
      `gemini-3.1-flash-lite-preview` with `remaining_pct=99.125`.
    - Post-smoke prod log scan found no `Langfuse Layer Error`, `NoneType`,
      invalid-reasoning-effort, or model-not-found errors; only known warnings
      remained.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/langfuse/langfuse_prompt_management.py tests/test_litellm/integrations/langfuse/test_langfuse_prompt_management.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/langfuse/test_langfuse_prompt_management.py -q`
      passed with `3 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_google_code_assist'`
      passed with `7 passed, 289 deselected`.
    - `git diff --check` passed before the release commit.

- D1-087 aawm.45 release prep for Codex-native Gemini
  - Goal: publish the LiteLLM `aawm.45` release artifacts for the Codex
    Gemini Code Assist adapter and prepare the prod container image without
    restarting prod.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `pyproject.toml`
    - `COMPLETED.md`
    - `TODO.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - Candidate commit `45eade7328` was pushed to `develop` and
      fast-forwarded to `main`.
    - Artifact autobump run `25754808275` succeeded, advanced `main` to
      `183c3ff397`, and tagged `cb-v0.0.20`.
    - The callback release asset was built locally with
      `./.venv/bin/python -m build --wheel --outdir /tmp/aawm-cb-dist .wheel-build`
      and published at `https://github.com/zepfu/litellm/releases/tag/cb-v0.0.20`
      with asset `aawm_litellm_callbacks-0.0.20-py3-none-any.whl`; `cb-latest`
      now points to `183c3ff397`.
    - Base image release workflow run `25755115747` succeeded and published
      `v1.82.3-aawm.45` / `ghcr.io/zepfu/litellm:1.82.3-aawm.45`.
    - Existing overlay releases/assets remain `cp-v0.0.7`, `cfg-v0.0.10`, and
      `h-v0.0.28`.
    - `/home/zepfu/projects/aawm-infrastructure` commit `7c66290` pins the
      standalone LiteLLM build to `ghcr.io/zepfu/litellm:1.82.3-aawm.45`.
    - `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      completed without recreating prod and produced `aawm-litellm:latest`
      image id `09c88f2ea9f0`.
    - Direct image inspection reported `litellm=1.82.3+aawm.45`,
      `aawm-litellm-callbacks=0.0.20`,
      `aawm-litellm-control-plane=0.0.7`,
      `has_codex_google_adapter=True`, and
      `callback_has_codex_adapter_model=True`.
    - Running prod was not restarted: `docker ps --filter name=aawm-litellm`
      still showed container `1cf65e1ab153`, image `27f4ac92e620`, healthy on
      `127.0.0.1:4000`.
  - Verification:
    - `diff -q .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/integrations/aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_google_code_assist or google_code_assist_builder or google_completion_adapter or unsupported_image_generation or codex_spark_drops_image_generation or codex_non_spark_keeps_image_generation or codex_spawn_agent_tool_description_patch'`
      passed with `28 passed, 267 deselected`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'codex_google_code_assist or codex_adapter_quota or google_quota_buckets'`
      passed with `3 passed, 121 deselected`.
    - `git diff --check` passed.
    - Broad Ruff on the touched large files was attempted but remains blocked
      by pre-existing lint debt in the callback/test modules; no unrelated
      cleanup was mixed into the release.

- D1-086 Codex child-agent Gemini model dispatch through LiteLLM profiles
  - Goal: make normal Codex sessions whose active provider is `litellm` or
    `litellm-dev` able to dispatch child agents on Gemini Code Assist models.
  - Changed paths:
    - `/home/zepfu/.codex/aawm-model-catalog.json`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Generated `/home/zepfu/.codex/aawm-model-catalog.json` as a full Codex
      model-catalog replacement built from current `~/.codex/models_cache.json`
      plus three Gemini preview aliases: `gemini-3-flash-preview`,
      `gemini-3.1-flash-lite-preview`, and `gemini-3.1-pro-preview`.
    - `codex debug models -c model_catalog_json='"/home/zepfu/.codex/aawm-model-catalog.json"'`
      loaded the catalog and listed all normal Codex models plus all three
      Gemini aliases.
    - A normal Codex parent session successfully spawned a child agent on
      `gemini-3.1-flash-lite-preview`; the child used the terminal and
      reported the local time as `14:13 EDT`.
    - Exact database `aawm_tristore.public.session_history` recorded child
      session `019e1d64-8556-79e0-bf18-9559be550950` rows `308322` and
      `308324` at `2026-05-12 18:13:41Z` / `18:13:43Z` with
      `provider=gemini`, `model=gemini-3.1-flash-lite-preview`,
      `client_name=codex-tui`,
      `metadata.passthrough_route_family=codex_google_code_assist_adapter`,
      `metadata.codex_adapter_model=gemini-3.1-flash-lite-preview`, and
      Google Code Assist quota metadata. Row `308322` recorded
      `tool_call_count=1` and `tool_names=["exec_command"]`.
    - Exact database `aawm_tristore.public.rate_limit_observations` recorded
      `google/google_code_assist/google_retrieve_user_quota` rows for the same
      session, including `gemini-3.1-flash-lite-preview`, with
      `remaining_pct=100` and reset `2026-05-13 18:13:38+00`.
  - Verification:
    - `codex debug models -c model_catalog_json='"/home/zepfu/.codex/aawm-model-catalog.json"' | jq -r '.models[].slug'`
      listed `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`,
      `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`,
      `codex-auto-review`, `gemini-3-flash-preview`,
      `gemini-3.1-flash-lite-preview`, and `gemini-3.1-pro-preview`.
    - `psql -Atqc "SELECT id, created_at, session_id, provider, model, client_name, metadata->>'passthrough_route_family', metadata->>'codex_adapter_model', metadata->>'google_retrieve_user_quota', tool_call_count, tool_names FROM public.session_history WHERE created_at >= now() - interval '30 minutes' AND (model LIKE 'gemini-%' OR metadata->>'codex_adapter_model' LIKE 'gemini-%') ORDER BY created_at DESC LIMIT 20;" "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore"`
      returned rows `308324` and `308322` for the Gemini child session with
      `codex_google_code_assist_adapter`.
    - `psql -Atqc "SELECT observed_at, provider, client, model, quota_key, quota_type, remaining_pct, expected_reset_at, source, session_id FROM public.rate_limit_observations WHERE session_id = '019e1d64-8556-79e0-bf18-9559be550950' ORDER BY observed_at DESC, model;" "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore"`
      returned Code Assist request-quota rows for Gemini models on the same
      session.

- D1-085 Codex-native Gemini via Google Code Assist adapter
  - Goal: allow native Codex Responses traffic to call Gemini Code Assist
    models through LiteLLM while keeping Codex-facing input/output as OpenAI
    Responses and Google-facing upstream payloads as native Code Assist
    `streamGenerateContent` requests authenticated with the local Gemini Code
    Assist OAuth/project/quota path.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Added a Codex Responses intercept for Gemini Code Assist aliases on
      `/openai_passthrough/*/responses` and `/openai/*/responses`, including
      `gemini-3.1*`, `gemini-3-flash-preview`, `google/*`,
      `code-assist/*`, `google-code-assist/*`, and `codex-gemini-*` aliases,
      while rejecting `openrouter/*` models.
    - The adapter uses `_load_valid_local_google_oauth_access_token()`,
      `_get_or_load_google_code_assist_project()`,
      `_prime_google_code_assist_session()`, native Code Assist headers, and
      the existing Google adapter cooldown/semaphore path before forwarding to
      `https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent`
      with `alt=sse`.
    - Codex Responses requests are translated at the proxy boundary through the
      Responses-to-chat bridge and then into the native Code Assist envelope:
      top-level `model`, `project`, `user_prompt_id`, and nested
      `request.contents` / `request.tools.functionDeclarations` /
      `generationConfig` / `systemInstruction` / `session_id`. OpenAI
      Responses fields such as `input` and `previous_response_id` are not sent
      to Google.
    - Non-function Codex Responses hosted tools are dropped before Google
      translation so Code Assist only sees function declarations; dropped tool
      types are recorded in logging metadata.
    - Streaming Code Assist output is converted back to OpenAI Responses SSE
      for Codex. The stream path now remembers Gemini tool-call names and
      argument fragments, then repairs follow-up Codex `tool` messages whose
      assistant `tool_calls` were omitted by Responses session reconstruction.
      The repaired turn replays the original function arguments and suppresses
      blank assistant text so Code Assist accepts the native function
      call/response pair.
    - Session-history attribution now preserves `codex_adapter_model` metadata
      for model resolution, whitelists the Codex Google adapter metadata, and
      records adapter quota observations as `provider=google`,
      `client=google_code_assist`.
    - Live dev `:4001` plain smoke
      `codex exec --profile litellm-dev -m gemini-3.1-pro-preview "just a test msg"`
      completed successfully with session
      `019e1c6c-52db-7912-b945-17555a36ca99`.
    - Live dev `:4001` tool-loop smoke
      `codex exec --profile litellm-dev -m gemini-3.1-pro-preview "Use the shell tool to run: printf codex-gemini-tool-smoke. Then reply with exactly the command output."`
      completed successfully with session
      `019e1c88-69fe-74a2-877b-b995f345efcd`, called `exec_command`, and
      returned exactly `codex-gemini-tool-smoke`.
    - Exact database `aawm_tristore` recorded session-history rows `306127`
      and `306128` for session `019e1c88-69fe-74a2-877b-b995f345efcd` with
      `provider=gemini`, `model=gemini-3.1-pro-preview`,
      `metadata.passthrough_route_family=codex_google_code_assist_adapter`,
      `metadata.codex_adapter_model=gemini-3.1-pro-preview`,
      `metadata.google_retrieve_user_quota=true`, and
      `metadata.aawm_stream_logging_endpoint_type=vertex-ai`; row `306127`
      recorded `tool_call_count=1` and `tool_names=["exec_command"]`.
    - Exact database `aawm_tristore.public.rate_limit_observations` recorded
      `google/google_code_assist/google_retrieve_user_quota` rows for that
      session, including `gemini-3.1-pro-preview`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'codex_google_code_assist or google_code_assist_builder or google_completion_adapter or unsupported_image_generation or codex_spark_drops_image_generation or codex_non_spark_keeps_image_generation or codex_spawn_agent_tool_description_patch'`
      passed with `28 passed, 267 deselected`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'codex_google_code_assist or codex_adapter_quota or google_quota_buckets'`
      passed with `3 passed, 121 deselected`.
    - `git diff --check` passed.

- D1-084 prod cutover for aawm.44 Codex Spark and session-history gap telemetry
  - Goal: promote the prepared `aawm.44` release to prod `:4000`, verify the
    D1-081 Spark hosted-tool sanitizer and D1-082 response-gap metric in the
    running runtime, and close the D1-079 prod memory-writer follow-up.
  - Changed paths:
    - `.analysis/completed.md`
    - `.analysis/todo.md`
    - `COMPLETED.md`
    - `TODO.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
  - Evidence:
    - `/home/zepfu/projects/aawm-infrastructure` now pins
      `LITELLM_BASE_IMAGE` to `ghcr.io/zepfu/litellm:1.82.3-aawm.44` in both
      `Dockerfile.litellm` and `docker-compose.litellm.yml`.
    - Running prod container `aawm-litellm` was recreated from local image
      `aawm-litellm:latest` image id `27f4ac92e620`; Docker reported container
      id `1cf65e1ab153`, status `healthy`, and port mapping
      `127.0.0.1:4000->4000/tcp`.
    - `/health/readiness` returned `status=healthy`,
      `litellm_version=1.82.3+aawm.44`, and success callbacks including
      `AawmAgentIdentity`; `/health/liveliness` returned `"I'm alive!"`.
    - Runtime package/source inspection inside `aawm-litellm` reported
      `litellm=1.82.3+aawm.44`, `aawm-litellm-callbacks=0.0.19`,
      `aawm-litellm-control-plane=0.0.7`,
      `passthrough_has_sanitizer=true`, `responses_calls_sanitizer=true`,
      `fallback_has_spark_unsupported_hosted_tools=true`, and
      `callback_has_gap_metric=true`.
    - Native prod-profile Codex smoke
      `codex exec --profile litellm -m gpt-5.3-codex-spark "just a test msg"`
      completed successfully with session
      `019e1c01-5d30-76e3-8fe6-55eb39e474b8`, returning a normal assistant
      response instead of the prior upstream `image_generation` 400. The
      corresponding `public.session_history` row `301249` in exact database
      `aawm_tristore` recorded `provider=openai`,
      `model=gpt-5.3-codex-spark`, and
      `metadata.passthrough_route_family=codex_responses`.
    - The smoke request did not advertise an unsupported hosted
      `image_generation` tool, so no `codex_unsupported_hosted_tool_*`
      removal metadata was expected on row `301249`; the loaded source/model
      metadata above covers requests that do advertise it.
    - Prod logs after the bad manual probe window showed repeated
      `/openai_passthrough/responses` `200 OK` entries. The earlier
      `model_not_found` and invalid-key/max-token errors were from controlled
      bad smoke shapes, not ongoing native Codex traffic.
    - `scripts/backfill_session_history_latency.py --apply --batch-size 5000
      --expected-database aawm_tristore` ran against exact database
      `aawm_tristore` with `applied=true`, `scanned_rows=149826`,
      `derivable_rows=149826`, `changed_rows=0`, `updated_rows=0`,
      `gap_scanned_rows=108054`, `gap_derivable_rows=30`,
      `gap_changed_rows=30`, and `gap_updated_rows=30`.
    - Post-backfill schema/value verification against exact database
      `aawm_tristore` showed
      `previous_response_to_current_request_ms` present and `143040` non-null
      values out of `251287` `public.session_history` rows.
    - Recent multi-row gap samples in session
      `019e179a-1363-7992-b278-7b8a6a96d329` showed non-overlapping gaps such
      as `375.217`, `1455.803`, and `19344.532` ms.
    - D1-079 prod/current verification against exact database `aawm_tristore`
      returned `memory_rows=60`, `unrepaired_workload_rows=0`, and
      `unrepaired_tagged_rows=0`.

- D1-083 aawm.44 release prep for Codex Spark and session-history gap telemetry
  - Goal: prepare the D1-081 Codex Spark hosted-tool sanitizer and D1-082
    session-history response-gap metric for prod cutover without touching
    `/home/zepfu/projects/aawm-infrastructure` or restarting prod `:4000`.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `WHEEL.md`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - `develop` and `main` were fast-forwarded through D1-081, D1-082, and
      release-prep commit `312b976c88`.
    - Artifact autobump workflow `25711367147` succeeded and advanced
      `origin/main` to `289d33f0d7`, creating tags `cb-v0.0.19`,
      `cp-v0.0.7`, and `cfg-v0.0.10`.
    - GitHub-created artifact tags did not create release assets, so the
      missing assets were built locally and published manually:
      `aawm_litellm_callbacks-0.0.19-py3-none-any.whl`
      (`sha256=41c21ca4ebbd181ae41985721ce126068ee9d947a930e3f2fbff1915ed9ac75b`),
      `aawm_litellm_control_plane-0.0.7-py3-none-any.whl`
      (`sha256=a4849c7c7261f103ba2855a7afa64b61f2ecabd2b1a8ece2500ff960e8851a05`),
      and `litellm-model-config-0.0.10.tar.gz`
      (`sha256=04236eb7f5ea060417322417f5bd8b631e0dc32347441e9ca1f52d903e6cfb2f`).
    - `cb-latest`, `cp-latest`, and `cfg-latest` were moved to `289d33f0d7`;
      `h-latest` remains on `h-v0.0.28` because no harness files changed.
    - Annotated tag `v1.82.3-aawm.44` was pushed from `289d33f0d7`; image
      workflow `25711488484` succeeded and published GitHub Release
      `v1.82.3-aawm.44` / `ghcr.io/zepfu/litellm:1.82.3-aawm.44`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k "unsupported_image_generation or codex_spark_drops_image_generation or codex_non_spark_keeps_image_generation or generic_responses_does_not_patch_spawn_agent_tool or codex_spawn_agent_tool_description_patch" -q`
      passed with `6 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py -k "codex_spark_image_generation" -q`
      passed with `1 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "codex_unsupported_hosted_tool_metadata or session_history_record_derives_passthrough_latency_breakdown or persist_session_history_record_executes_insert or persist_session_history_records_executes_batch_insert"`
      passed with `4 passed`.
    - `git diff --check`, targeted Ruff checks, callback/backfill
      `py_compile`, and `scripts/bump_aawm_artifact_versions.py --before
      origin/main --after origin/develop` passed before the main push.
    - `gh release view` verified assets for `cb-v0.0.19`, `cp-v0.0.7`,
      `cfg-v0.0.10`, and `v1.82.3-aawm.44`.
  - Follow-up status: completed by D1-084 above. Prod infrastructure was later
    updated, the prod `:4000` runtime was verified, Codex Spark native traffic
    was smoked, and the D1-082 latency/gap backfill ran against exact database
    `aawm_tristore`.

- D1-082 session_history previous-response gap metric
  - Goal: add a persisted `public.session_history` `_ms` field that measures
    the delay from the previous row's completed response to the current row's
    request start within the same `session_id`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/backfill_session_history_latency.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Added nullable
      `previous_response_to_current_request_ms DOUBLE PRECISION` to the
      dynamic `public.session_history` schema and alter path.
    - Added `session_history_session_start_idx` on
      `(session_id, COALESCE(start_time, created_at), id)` for the prior-row
      lookup used by live writes and backfill.
    - Live callback writes insert/upsert the row first, then updates the new
      field by looking up the immediately prior row in the same session. It
      also recomputes the immediate following row so late-finishing earlier
      requests can repair a later row's gap. The value is
      `current COALESCE(start_time, created_at) - previous end_time` in
      milliseconds; first rows, missing previous `end_time`, and overlapping
      rows remain `NULL` instead of being clamped.
    - `scripts/backfill_session_history_latency.py` now has a second pass for
      the new gap metric and reports `gap_*` counters separately from the
      existing D1-078 latency counters.
    - Synced the callback overlay:
      `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`.
    - Focused tests:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "session_history_record_derives_passthrough_latency_breakdown or persist_session_history_record_executes_insert or persist_session_history_records_executes_batch_insert"`
      (`3 passed`, existing pytest config warning only).
    - Static checks:
      `./.venv/bin/python -m py_compile scripts/backfill_session_history_latency.py`;
      `./.venv/bin/ruff check scripts/backfill_session_history_latency.py`;
      `./.venv/bin/ruff check litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py --ignore PLR0915`;
      `./.venv/bin/python scripts/backfill_session_history_latency.py --help`.
  - Follow-up status: code and backfill tooling are complete locally. Prod/dev
    runtime schema creation and historical `aawm_tristore` backfill were not run
    in this turn; `D1-082-prod-backfill` tracks that operational follow-up.

## 2026-05-11

- D1-081 Codex Spark native hosted-tool sanitizer
  - Goal: stop native Codex Responses traffic for `gpt-5.3-codex-spark` from
    forwarding the unsupported OpenAI hosted tool
    `{"type":"image_generation"}`, which the upstream Codex backend rejects
    with `Tool 'image_generation' is not supported with gpt-5.3-codex-spark`.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/response_api_endpoints/endpoints.py`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `litellm/types/utils.py`
    - `litellm/utils.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `docker-compose.dev.yml`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Added `unsupported_hosted_tools: ["image_generation"]` for both
      `chatgpt/gpt-5.3-codex-spark` and `gpt-5.3-codex-spark` model metadata.
    - Added a Codex-native request sanitizer that removes only matching hosted
      tools, preserves function tools, removes `tool_choice` only when it
      targets a removed hosted tool, and records auditable
      `codex_unsupported_hosted_*` metadata/tags.
    - Covered both native surfaces: `/openai_passthrough/responses` and the
      first-class `/v1/responses` route. The dev compose bind mounts now include
      `litellm/proxy/response_api_endpoints/endpoints.py` so dev `:4001`
      exercises both code paths after recreate.
    - The sanitizer reads live `litellm.model_cost` first and falls back to the
      bundled model-cost map when the remote model-cost map has not yet picked
      up the new field. Container verification showed
      `gpt-5.3-codex-spark -> ['image_generation']`, updated tools
      `[{'type': 'function', 'name': 'read_file'}]`, and removed tools
      `[{'type': 'image_generation', 'index': 0}]`.
    - Focused tests:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'unsupported_image_generation or codex_spark_drops_image_generation or codex_non_spark_keeps_image_generation or generic_responses_does_not_patch_spawn_agent_tool or codex_spawn_agent_tool_description_patch' -q`
      (`6 passed`, existing pytest/backoff warnings only).
    - First-class Responses route test:
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py -k 'codex_spark_image_generation' -q`
      (`1 passed`, existing pytest config warning only).
    - Session-history metadata retention test:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -k 'codex_unsupported_hosted_tool_metadata' -q`
      (`1 passed`, existing pytest config warning only).
    - Static checks: `py_compile` passed for changed source/callback files,
      `git diff --check` passed, and
      `./.venv/bin/python -m ruff check litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/proxy/response_api_endpoints/endpoints.py tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py litellm/types/utils.py litellm/utils.py`
      passed. Full-file ruff on the legacy passthrough test and callback overlay
      still reports pre-existing style/complexity findings.
    - Recreated dev `litellm-dev` on `:4001` with
      `docker compose -f docker-compose.dev.yml up -d --no-build --force-recreate --no-deps litellm-dev`;
      liveliness returned `"I'm alive!"`, readiness returned `status=healthy`,
      and mounted file hashes matched local source for the passthrough file,
      response endpoint file, and bundled model-cost map.
    - Live dev smoke:
      `codex exec --profile litellm-dev -m gpt-5.3-codex-spark "just a test msg"`
      completed with `Test message received.` Recent dev logs show
      `POST /openai_passthrough/responses HTTP/1.1" 200 OK` and no
      `Tool 'image_generation'` 400. The smoke still emitted an unrelated
      Codex model-refresh warning because LiteLLM `/models` is OpenAI-shaped
      (`data`) while this Codex CLI expected a `models` field.
  - Follow-up status: code and dev `:4001` are fixed. Prod `:4000` has not been
    cut over in this turn; `D1-081-prod-promotion` tracks that remaining
    operational step.

## 2026-05-10

- D1-080 Claude Code Agent tool description preservation
  - Goal: stop truncating Claude Code's top-level `Agent` tool description so
    the advertised agent types and model options remain intact in downstream
    traces and provider requests.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/completed.md`
  - Evidence:
    - Added `Agent` to a preserve list for top-level tool descriptions while
      leaving existing schema cleanup in place (`$schema` removal and nested
      schema-description truncation).
    - Added a focused regression test proving an over-360-character
      `Agent.description` is preserved exactly and the schema cleanup still
      logs a Claude tool-advertisement compaction event.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'claude_code_tool_advertisements or agent_tool_description' -q`
      (`4 passed`, existing pytest/backoff warnings only).
    - Restarted only dev container `litellm-dev` on `:4001` with
      `docker compose -f docker-compose.dev.yml restart litellm-dev`; container
      start time became `2026-05-10T16:17:38.708062252Z`.
    - Dev liveliness returned `"I'm alive!"`, readiness returned
      `status=healthy`, and container import verification showed
      `_CLAUDE_TOOL_DESCRIPTION_PRESERVE_NAMES == {'Agent'}`.

## 2026-05-08

- D1-079 Codex memory-writer repository labeling
  - Goal: distinguish Codex memory-generation traffic from user-requested repo
    work in `public.session_history` by labeling the repository as
    `<repo> (memory)` while preserving the original repo name in metadata.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Added Codex memory-workflow detection for native Codex Responses requests
      containing the memory-writer marker plus rollout/raw-memory guard text.
      Matching rows now set `repository` and repository-derived `tenant_id` to
      `<repo> (memory)`, set `metadata.source_repository` to the original repo,
      set `metadata.workload_type=agent_memory`, set
      `metadata.workload_subtype=codex_memory_writer`, and add
      `codex-memory-workflow` / `agent-memory-workload` tags.
    - Synced the callback overlay:
      `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      (`121 passed`, one existing pytest config warning).
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      (`All checks passed`).
    - `git diff --check`.
    - Restarted dev container `litellm-dev` on `:4001`; liveliness returned
      `"I'm alive!"`, readiness returned `status=healthy` with
      `AawmAgentIdentity` in success callbacks, and the container-mounted
      `/app/litellm/integrations/aawm_agent_identity.py` SHA-256 matched the
      local source file:
      `a5b1b78a6dc4f7a88d485b0f161e0617f60bc0b6fbfa39fbab9acb3254053e60`.
    - Backfilled the 15 exact previously identified memory-writer rows in exact
      database `aawm_tristore`: ids `210209`, `210389`, `210395`, `210407`,
      `210466`, `210484`, `210531`, `210533`, `210536`, `210537`, `210541`,
      `210547`, `210549`, `210550`, and `210813`.
    - Final DB verification against exact database `aawm_tristore` returned
      `(15, 15, 0, 0)` for
      `repaired|total|missing_request_tag|missing_tag`; sampled rows showed
      `aawm-lsp-proxy (memory)`, `mcp-pg (memory)`,
      `pytest-testable (memory)`, and `pytest-classifier (memory)` with
      matching `metadata.source_repository` values.

- Prod cutover for D1-075/D1-076/D1-077/D1-078 overlay changes
  - Goal: promote the prepared callback/harness/session-history telemetry
    overlay to the production-style LiteLLM deployment on `127.0.0.1:4000`,
    then repair any in-flight `public.session_history` rows created before the
    callback restart.
  - Changed paths:
    - `.analysis/completed.md`
    - `.analysis/todo.md`
    - `COMPLETED.md`
    - `TODO.md`
  - Evidence:
    - Recreated prod from the already-built local image with
      `docker compose -f docker-compose.litellm.yml up -d --no-build --force-recreate --no-deps litellm`
      in `/home/zepfu/projects/aawm-infrastructure`.
    - Prod readiness on `127.0.0.1:4000` returned `status=healthy`,
      `litellm_version=1.82.3+aawm.43`, and success callbacks including
      `AawmAgentIdentity`; liveliness returned `"I'm alive!"`.
    - Runtime package inspection inside `aawm-litellm` returned
      `litellm=1.82.3+aawm.43`, `aawm-litellm-callbacks=0.0.18`, and
      `aawm-litellm-control-plane=0.0.6`.
    - Targeted provider-cache repair against exact database `aawm_tristore`
      scanned `8` missing-field cache-miss rows and repaired `4`; the remaining
      four rows are NVIDIA `qwen/qwen3-coder-480b-a35b-instruct` rows where the
      token count is present but cost remains intentionally null because there
      is no response cost or bundled NVIDIA pricing to derive from.
    - Latency backfill against exact database `aawm_tristore` dry-ran with
      `changed_rows=967`, `derivable_rows=99619`, then applied
      `updated_rows=967`.
    - Final aggregate verification in `aawm_tristore` returned
      `aawm_tristore|0|0|4|0` for
      `current_database|stale_local|cache_miss_null_tokens|cache_miss_null_cost|derivable_latency_null`.
      That confirms D1-075 stale local-Qwen attribution is clear, D1-076 miss
      token counts are clear, and D1-078 derivable latency nulls are clear; the
      four remaining null costs are the known NVIDIA no-pricing rows.
    - Prod local LLM smoke session `prod-local-llm-release-20260508` returned
      `OK` and wrote `public.session_history` row `206279` with
      `provider=local_llm`, `model=qwen3-heretic-gguf`,
      `model_group=qwen3-heretic-gguf`, `litellm_environment=prod`,
      `litellm_version=1.82.3+aawm.43`, `client_name=curl`,
      `tenant_id=tenant-local-prod-validation`,
      `total_server_elapsed_ms=791.361`,
      `metadata.aawm_local_route_family=local_llm_chat`,
      `metadata.aawm_local_upstream_model=qwen3-4b-heretic-q8`, and
      `metadata.aawm_local_upstream_api_base=http://host.docker.internal:8093/v1`.
      The detailed upstream split fields are null on this non-pass-through
      local route, which is expected.
  - Verification:
    - `curl -sS http://127.0.0.1:4000/health/readiness`
    - `curl -sS http://127.0.0.1:4000/health/liveliness`
    - `docker exec aawm-litellm python3 -c "import importlib.metadata as m; print(m.version('litellm')); print(m.version('aawm-litellm-callbacks')); print(m.version('aawm-litellm-control-plane'))"`
    - `scripts/repair_session_history_provider_cache.py --missing-cache-miss-fields-only --batch-size 1000 --apply`
    - `scripts/backfill_session_history_latency.py --batch-size 5000 --apply`
    - `psql -Atqc` aggregate and smoke-row verification queries against
      `postgresql://aawm:...@127.0.0.1:5434/aawm_tristore`.
  - Follow-up status: the prod cutover and derivable backfills are complete.
    Remaining release follow-up is optional focused/default prod harness
    validation and normal monitoring of new prod Codex Responses prompt-overhead
    rows after the callback `0.0.18` cutover.

## 2026-05-07

- Prod release prep for D1-075/D1-076/D1-077/D1-078 overlay changes
  - Goal: prepare the production-style `:4000` LiteLLM deployment for the
    latest callback/harness/session-history telemetry changes without
    restarting or recreating the running prod container until a general pause
    point.
  - Changed paths:
    - `COMPLETED.md`
    - `TODO.md`
  - Evidence:
    - Read `PROD_RELEASE.md` and `WHEEL.md`; release scope is callback,
      harness, and local backfill scripts, so the release follows the overlay
      path and does not cut a new base fork image.
    - Committed LiteLLM release changes as
      `3cdf6fb237 Prepare AAWM session history telemetry overlays` and pushed
      to `main` and `develop`.
    - Main artifact autobump completed successfully and produced
      `3fbaa81cb0 chore(release): bump artifact versions`, `cb-v0.0.18`, and
      `h-v0.0.28`; `develop` was fast-forwarded to the same commit.
    - Manually published missing GitHub Release assets:
      `aawm_litellm_callbacks-0.0.18-py3-none-any.whl` on `cb-v0.0.18` and
      `litellm-local-ci-harness-0.0.28.tar.gz` on `h-v0.0.28`.
    - Moved `cb-latest` and `h-latest` to the matching versioned tags.
    - Rebuilt `/home/zepfu/projects/aawm-infrastructure` prod image with
      `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`.
      The build installed `litellm=1.82.3+aawm.43`,
      `aawm-litellm-callbacks=0.0.18`, and
      `aawm-litellm-control-plane=0.0.6`.
    - Inspected the built `aawm-litellm:latest` image and confirmed those
      package versions plus the callback latency fields:
      `litellm_processing_ms`, `llm_upstream_elapsed_ms`,
      `total_server_elapsed_ms`, `ttft_ms`, `litellm_pre_send_ms`,
      `litellm_post_response_ms`,
      `llm_upstream_time_to_first_byte_ms`, `llm_upstream_stream_ms`, and
      `latency_unclassified_ms`.
    - Verified the running prod `aawm-litellm` container was not cut over at
      prep time: it remained healthy on `127.0.0.1:4000` and still reported
      callback `0.0.17`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/repair_session_history_provider_cache.py scripts/backfill_session_history_latency.py scripts/local-ci/run_anthropic_adapter_acceptance.py`
      passed.
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/repair_session_history_provider_cache.py scripts/backfill_session_history_latency.py scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
      passed.
    - `git diff --check` passed before and after doc updates.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
  - Follow-up status: superseded by the 2026-05-08 prod cutover entry above;
    the container recreate and prod backfills are now complete.

- D1-078 `session_history` latency breakdown columns and dev backfill
  - Goal: add explicit `public.session_history` fields for non-overlapping
    LiteLLM processing time, upstream provider elapsed time, total server
    elapsed time, time to first token/byte, and residual unclassified latency.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/backfill_session_history_latency.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - Added nullable `DOUBLE PRECISION` columns:
      `litellm_processing_ms`, `llm_upstream_elapsed_ms`,
      `total_server_elapsed_ms`, `ttft_ms`, `litellm_pre_send_ms`,
      `litellm_post_response_ms`,
      `llm_upstream_time_to_first_byte_ms`, `llm_upstream_stream_ms`, and
      `latency_unclassified_ms`.
    - Live write normalization now derives the fields from stored AAWM
      pass-through timing metadata when present, and falls back to
      `start_time`/`end_time` for `total_server_elapsed_ms` where detailed
      timing does not exist.
    - Added `scripts/backfill_session_history_latency.py`, which imports the
      same derivation helper as the callback and refuses to run unless
      `current_database()` matches the expected database (`aawm_tristore` by
      default).
    - Dev backfill ran against exact database `aawm_tristore`. Initial dry-run:
      `161109` derivable changed rows. First apply updated `161128` rows; two
      catch-up applies updated `15` and `30` additional rows created during
      verification.
    - Post-backfill verification in `aawm_tristore`: `161154` total rows;
      populated counts were `total_server_elapsed_ms=161143`,
      `litellm_processing_ms=120357`, `llm_upstream_elapsed_ms=120357`,
      `ttft_ms=111642`, `litellm_pre_send_ms=120357`,
      `litellm_post_response_ms=71207`,
      `llm_upstream_time_to_first_byte_ms=111642`,
      `llm_upstream_stream_ms=111642`, and
      `latency_unclassified_ms=120357`. Rows with timing metadata:
      `120369`; rows with full total/LiteLLM/upstream split: `120357`.
    - Restarted only dev `litellm-dev` on `:4001`; `/health/liveliness`
      returned `"I'm alive!"`. A local `qwen3-heretic-gguf` chat smoke wrote
      `public.session_history` row `205232` in exact database
      `aawm_tristore` with `provider=local_llm`,
      `model=qwen3-heretic-gguf`, and `total_server_elapsed_ms=720.126`.
      Detailed upstream/LiteLLM split fields are expected to be null on this
      non-pass-through local route because it does not emit the pass-through
      `aawm_*` timing metadata.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `119 passed` and one pre-existing pytest config warning for
      `asyncio_default_fixture_loop_scope`.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/backfill_session_history_latency.py`
      passed.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/backfill_session_history_latency.py`
      passed.
    - `git diff --check` passed.
  - Follow-up status: package/release/restart is still required before future
    prod rows are written by the new callback; rerun this latency backfill
    during the next prod migration for rows created before cutover.

- D1-077 Codex Responses prompt-overhead semantic splitter
  - Goal: stop OpenAI/Codex Responses opaque multi-turn state from producing
    large negative `public.session_history.input_breakdown_residual_tokens`
    by being counted as visible conversation.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
    - `scripts/local-ci/anthropic_adapter_config.json`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
  - Evidence:
    - Added an OpenAI Responses semantic splitter that counts visible
      user/assistant message text and `function_call_output.output` as
      conversation, while excluding reasoning/encrypted/item-reference and
      prior provider/tool state from conversation estimates.
    - Added metadata-only `usage_input_opaque_state_tokens_estimated` and
      `prompt_overhead_excluded_component_paths`; the prompt-overhead
      classifier version is now `deterministic-v2`.
    - Updated the prompt-overhead cost-share report to expose opaque-state
      token estimates separately instead of allocating them as prompt-overhead
      cost.
    - Unit coverage now includes a Codex/OpenAI Responses input array with
      visible messages, a visible tool result, `reasoning.encrypted_content`,
      an `item_reference`, and a prior `function_call`, verifying conversation
      tokens exclude opaque state and residual remains positive.
    - Restarted only dev `litellm-dev` on `:4001`. The narrow native Codex
      Responses harness passed with artifact
      `/tmp/native_codex_4001_prompt_overhead_d1_077.json`.
    - Exact database `aawm_tristore` verification found
      `public.session_history` row `200194`, trace
      `22954c7b-9e28-44cc-81ee-1e89a3aaae2c`, provider/model
      `openai/gpt-5.4-mini`, `input_tokens=26934`,
      `input_system_tokens_estimated=3747`,
      `input_tool_advertisement_tokens_estimated=11577`,
      `input_conversation_tokens_estimated=4625`,
      `input_other_tokens_estimated=6985`,
      `input_breakdown_residual_tokens=6985`,
      `prompt_overhead_classifier_version=deterministic-v2`,
      `prompt_overhead_counted_shape=openai_responses`, and
      `prompt_overhead_route_family=codex_responses`.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `118 passed` and one pre-existing pytest config warning for
      `asyncio_default_fixture_loop_scope`.
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q`
      passed with `56 passed` and the same pre-existing pytest config warning.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/repair_session_history_provider_cache.py scripts/local-ci/run_anthropic_adapter_acceptance.py`
      passed.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/local-ci/run_anthropic_adapter_acceptance.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
      passed. Plain ruff remains blocked by pre-existing `PLR0915`/`T201`
      findings in these long local scripts/files.
    - `git diff --check` passed.

- D1-076 provider cache miss token/cost exposure follow-up
  - Goal: close remaining `public.session_history` cache-miss rows where a
    miss reason existed but `provider_cache_miss_token_count` or
    `provider_cache_miss_cost_usd` stayed null for non-Anthropic providers.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/repair_session_history_provider_cache.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - `_compute_provider_cache_miss_cost_state()` now keeps token-count
      enrichment separate from pricing success, maps the cache telemetry
      family `nvidia` to pricing provider `nvidia_nim`, falls back to bundled
      input pricing when cache-read pricing is unavailable, and falls back to a
      response-cost token-share estimate for aliases such as
      `codex-auto-review`.
    - `scripts/repair_session_history_provider_cache.py` now has
      `--cache-misses-only` and `--missing-cache-miss-fields-only`, allowing
      targeted backfills instead of rewriting unrelated historical rows.
    - Targeted backfill against exact database `aawm_tristore` applied in two
      passes: first `195` repaired rows (`openrouter=82`, `nvidia=35`,
      `openai=78`), then `7` zero-usage OpenRouter rows repaired after the
      callback began recording zero-token misses as `0` token / `0` cost.
    - Post-backfill verification in `aawm_tristore`:
      OpenAI miss rows have `431/431` token counts and `431/431` costs;
      Gemini has `315/315` and `315/315`; OpenRouter has `82/82` and `82/82`;
      NVIDIA has `35/35` token counts and `31/35` costs. The four remaining
      NVIDIA cost-null rows are `qwen/qwen3-coder-480b-a35b-instruct` rows
      with no response cost and no bundled NVIDIA pricing, so the cost remains
      intentionally unknown rather than fabricated.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `117 passed` and one pre-existing pytest config warning for
      `asyncio_default_fixture_loop_scope`.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/repair_session_history_provider_cache.py`
      passed.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.

- D1-076 provider cache miss token/cost exposure
  - Goal: make non-Anthropic provider cache misses visible in
    `public.session_history` query fields used for usage reporting, especially
    `provider_cache_miss_token_count` and `provider_cache_miss_cost_usd`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - Live `aawm_tristore.public.session_history` already had miss booleans
      and reasons for OpenAI, Gemini, NVIDIA, and OpenRouter, but most
      non-Anthropic miss rows had null `provider_cache_miss_token_count` and
      null `provider_cache_miss_cost_usd`, so reporting queries summing those
      columns undercounted misses.
    - `_compute_provider_cache_miss_cost_state()` now preserves metadata
      supplied miss fields, sets missed token count from prompt/input tokens
      for ordinary `status=miss` rows, and computes
      `prompt_vs_cache_read_delta` when the model catalog has cache-read
      pricing. Existing Anthropic write-only cache behavior is unchanged.
    - Backfilled exact database `aawm_tristore`: `877` cache-miss rows were
      updated with token counts and `691` received cost deltas. Positive-input
      cache misses with null token count now return `0`.
    - Post-backfill provider aggregates include non-Anthropic token counts:
      Gemini `2,586,008`, NVIDIA `759,987`, OpenAI `38,180,951`, and
      OpenRouter `3,625,554` cache-miss tokens across rows that had
      provider-cache miss flags.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "provider_cache or cache_miss or cache_write or cached_content or prompt_cache_key"`
      passed with `13 passed`, `102 deselected`, and one pre-existing pytest
      config warning for `asyncio_default_fixture_loop_scope`.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `115 passed` and the same pre-existing pytest config warning.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `git diff --check` passed.
    - Restarted only dev container `litellm-dev` on `:4001`; readiness
      returned `status=healthy`.

- D1-071 malformed Codex placeholder header guard
  - Goal: explain and prevent the bad `public.rate_limit_observations` rows
    `825` and `826`, which briefly reset generic Codex quota state to
    `remaining_pct=100` with reset timestamps at the request time.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - Exact database `aawm_tristore` showed rows `821/822` immediately before
      the bad event with generic Codex state `primary=98` resetting at
      `2026-05-07 07:23:43+00` and `secondary=54` resetting at
      `2026-05-12 00:04:35+00`; rows `827/828` immediately after returned to
      those same values.
    - Rows `825/826` came from trace
      `d14e0263-808c-4c8a-9aa4-fca72dee7f01` and call
      `1f690332-c323-45b9-89fc-51e6221421e4`. The raw Codex metadata on the
      Langfuse trace had generic placeholder headers
      `x-codex-primary-window-minutes=0`,
      `x-codex-secondary-window-minutes=0`,
      `x-codex-primary-reset-after-seconds=0`,
      `x-codex-secondary-reset-after-seconds=0`, and used percent `0`; the
      same response had valid `x-codex-bengalfox-*` window headers.
    - Transition rows `401/402` were therefore false
      `capacity_grant_or_random_reset` transitions, and `403/404` were false
      `expected_rollover` transitions caused by accepting those placeholder
      generic rows.
    - The Codex response-header parser now skips a header observation when its
      explicit `*-window-minutes` value is invalid or non-positive, while still
      preserving default windows when the window header is absent.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "codex_response_headers or rate_limit"`
      passed with `13 passed`, `102 deselected`, and one pre-existing pytest
      config warning for `asyncio_default_fixture_loop_scope`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `115 passed` and the same pre-existing pytest config warning.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `git diff --check` passed.
    - After explicit user approval, deleted derived transition rows
      `401,402,403,404` from
      `aawm_tristore.public.rate_limit_transitions`. Verification query
      `select count(*) from public.rate_limit_transitions where id in (401,402,403,404)`
      returned `0`; neighboring rows `399`, `400`, and `405` remained.

- D1-075 local LLM session-history route attribution
  - Goal: make local OpenAI-compatible chat routes identifiable in
    `public.session_history` instead of looking like ordinary OpenAI traffic
    with only `model_group` carrying the LiteLLM-facing alias.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - Existing `aawm_tristore.public.session_history` rows for
      `qwen3-heretic-gguf` showed `provider=openai`,
      `model=qwen3-4b-heretic-q8`, and `model_group=qwen3-heretic-gguf`, but
      no local route/upstream detail in `metadata`.
    - Session-history record construction now detects OpenAI-compatible
      completion calls routed to local/private API bases, persists them as
      `provider=local_llm` with `model=<LiteLLM model_group>`, and adds
      sanitized `aawm_local_*` metadata for route family, model group,
      upstream provider, upstream model, and upstream API base.
    - API base logging strips credentials, query strings, and fragments before
      writing `aawm_local_upstream_api_base`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed with `113 passed` and one pre-existing pytest config warning for
      `asyncio_default_fixture_loop_scope`.
    - Restarted only dev container `litellm-dev` on `:4001` with
      `docker compose -f docker-compose.dev.yml restart litellm-dev`; readiness
      returned `status=healthy`.
    - Backfilled exact database `aawm_tristore`: `411` existing
      `provider=openai`, `model=qwen3-4b-heretic-q8`,
      `model_group=qwen3-heretic-gguf` rows now use `provider=local_llm`,
      `model=qwen3-heretic-gguf`, and local upstream metadata. Final stale
      count for the old OpenAI attribution returned `0`.
    - Dev smoke through `http://127.0.0.1:4001/v1/chat/completions` with
      session/trace `qwen-local-attribution-smoke-20260507` returned `OK`.
      `public.session_history` row `196449` recorded `provider=local_llm`,
      `model=qwen3-heretic-gguf`, `model_group=qwen3-heretic-gguf`,
      `aawm_local_route_family=local_llm_chat`,
      `aawm_local_upstream_model=qwen3-4b-heretic-q8`, and sanitized upstream
      API base `http://172.20.0.1:8093/v1`.

## 2026-05-06

- D1-074 aawm.43 prod promotion for rate-limit observations and local Qwen
  - Goal: promote the D1-071 provider-originated rate-limit observer and the
    local `qwen3-heretic-gguf` chat route from dev `:4001` to prod `:4000`
    through the documented image/overlay/infra path.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `litellm/integrations/aawm_passthrough_shape_capture.py`
    - `litellm/proxy/pass_through_endpoints/google_code_assist_quota.py`
    - `scripts/backfill_rate_limit_observations.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_scripts/test_backfill_rate_limit_observations.py`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
    - `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
  - Evidence:
    - LiteLLM source commit `5be262cf90` was pushed to `develop` and
      fast-forwarded into `main`; autobump commit `1fa5123200` produced
      callback tag `cb-v0.0.17`.
    - Published missing GitHub Release `cb-v0.0.17` with asset
      `aawm_litellm_callbacks-0.0.17-py3-none-any.whl`.
    - Published fork image/release `v1.82.3-aawm.43` with container image
      `ghcr.io/zepfu/litellm:1.82.3-aawm.43`.
    - Infrastructure commit `9c15c2f` pins prod to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.43` and includes the
      `qwen3-heretic-gguf` route pointing at
      `http://host.docker.internal:8093/v1`.
  - Verification:
    - Dev verification before release:
      `py_compile` for callback/pass-through files passed; focused callback
      quota pytest passed with `5 passed`; focused pass-through pytest passed
      with `45 passed`; backfill pytest passed with `4 passed`.
    - Controlled dev Codex request wrote `public.session_history` row
      `019dfad6-0d78-71f3-91ec-b7734c311b1a` in exact database
      `aawm_tristore`; no new `codex_bengalfox:*` full-quota jitter rows were
      inserted after `2026-05-06T01:10:14Z`.
    - GitHub Actions run `25411261051` for `v1.82.3-aawm.43` succeeded:
      runtime import smoke, image build/push, and GitHub release creation all
      passed.
    - Local prod image build used base digest for
      `ghcr.io/zepfu/litellm:1.82.3-aawm.43` and installed
      `aawm-litellm-callbacks==0.0.17`.
    - Built image version inspection returned `1.82.3+aawm.43`, `0.0.17`,
      and `0.0.6`.
    - Running prod `curl -sS http://127.0.0.1:4000/health/readiness` returned
      `status=healthy` and `litellm_version=1.82.3+aawm.43`.
    - Running container version inspection returned `1.82.3+aawm.43`,
      `0.0.17`, and `0.0.6`.
    - Rendered `/etc/litellm/config.yaml` includes `qwen3-heretic-gguf`,
      `qwen3-4b-heretic-q8`, and `host.docker.internal:8093`.
    - Prod local chat smoke through `:4000` for model `qwen3-heretic-gguf`
      returned assistant content `OK`.

## 2026-05-05

- D1-073 aawm.43 release prep for quota observations and local Qwen
  - Goal: prepare the current D1-071 rate-limit observation work and local
    `qwen3-heretic-gguf` route for a production `:4000` promotion without
    touching prod before the release artifacts exist.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `WHEEL.md`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Fork metadata now targets `1.82.3+aawm.43`.
    - `PATCHES.md` has an `aawm.43` entry for provider-originated rate-limit
      observations and the local Qwen route.
    - `WHEEL.md` records that the callback overlay owns
      `public.rate_limit_observations` capture.
    - The prod callback wheel source is synced with the in-repo development
      callback source.
    - The expected artifact autobump from the current changed file set is
      `callback: 0.0.16 -> 0.0.17 (cb-v0.0.17)`.
    - `aawm-infrastructure` has a staged prod config-template route for
      `qwen3-heretic-gguf` using `http://host.docker.internal:8093/v1`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/integrations/aawm_passthrough_shape_capture.py litellm/proxy/pass_through_endpoints/google_code_assist_quota.py scripts/backfill_rate_limit_observations.py`
      passed.
    - `./.venv/bin/python -c "import yaml; yaml.safe_load(open('litellm-dev-config.yaml')); yaml.safe_load(open('/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl')); print('yaml ok')"`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_backfill_rate_limit_observations.py -q`
      passed with `115 passed`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q`
      passed with `284 passed`.
    - Focused rate-limit slice passed with `25 passed, 374 deselected`.
    - Ruff undefined-name/syntax check passed for touched Python paths.
    - `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`
      restarted dev `:4001`; readiness returned `status=healthy`.
    - `curl -sS http://127.0.0.1:4001/model_group/info?model_group=qwen3-heretic-gguf`
      returned `mode=chat`, provider `openai`, and zero input/output token cost.
    - `curl -sS http://127.0.0.1:4001/v1/chat/completions ...` returned
      assistant content `OK`.
    - Exact database `aawm_tristore` has 33
      `public.rate_limit_observations` rows: Anthropic/Claude response headers
      16, OpenAI/Codex response headers 6, Google Code Assist quota 11.
    - Local callback wheel packaging succeeded into
      `/tmp/aawm-cb-prep/aawm_litellm_callbacks-0.0.16-py3-none-any.whl`; the
      publishable version is expected to be `0.0.17` after the main-branch
      autobump.

- D1-072 dev local Qwen3 Heretic chat route
  - Goal: add the new local `qwen3-heretic-gguf` model on port `8093` as a
    pure LLM/chat route, without embedding or rerank exposure.
  - Changed paths:
    - `litellm-dev-config.yaml`
    - `.analysis/completed.md`
  - Evidence:
    - `litellm-dev-config.yaml` now exposes LiteLLM model group
      `qwen3-heretic-gguf` with `model_info.mode=chat`, upstream
      `openai/qwen3-4b-heretic-q8`, and
      `api_base=http://172.20.0.1:8093/v1`.
    - Direct local backend `/v1/models` on `127.0.0.1:8093` advertises
      `qwen3-4b-heretic-q8` with `capabilities=["completion"]`.
  - Verification:
    - `./.venv/bin/python -c "import yaml; data=yaml.safe_load(open('litellm-dev-config.yaml')); route=next(m for m in data['model_list'] if m.get('model_name') == 'qwen3-heretic-gguf'); assert route['litellm_params']['model'] == 'openai/qwen3-4b-heretic-q8'; assert route['model_info']['mode'] == 'chat'; print('yaml ok: qwen3-heretic-gguf -> openai/qwen3-4b-heretic-q8 chat')"`
      passed.
    - Direct backend smoke:
      `curl -sS http://127.0.0.1:8093/v1/chat/completions ...` returned
      assistant content `OK`.
    - Restarted dev proxy with
      `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`.
    - `curl -sS http://127.0.0.1:4001/health/readiness` returned
      `status=healthy`.
    - `curl -sS http://127.0.0.1:4001/v1/models` included
      `qwen3-heretic-gguf`.
    - `curl -sS http://127.0.0.1:4001/model_group/info` showed
      `model_group=qwen3-heretic-gguf`, `providers=["openai"]`, and
      `mode="chat"`.
    - Through-proxy chat smoke:
      `curl -sS http://127.0.0.1:4001/v1/chat/completions ...` returned
      assistant content `OK`.

- D1-072 local Qwen3 Heretic consumer docs
  - Goal: document how local repos should consume the local
    `qwen3-heretic-gguf` chat route through AAWM LiteLLM, matching the
    existing local embedding/rerank consumer docs and making clear this is not
    an embedding or rerank model.
  - Changed paths:
    - `LOCAL_LLM_CONSUMER.md`
    - `LOCAL_EMBED_RERANK_CONSUMER.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added `LOCAL_LLM_CONSUMER.md` with base URL/auth guidance, attribution
      headers, available model table, `/v1/chat/completions` curl example,
      OpenAI SDK example, and operational notes.
    - Cross-linked the local LLM guide from `LOCAL_EMBED_RERANK_CONSUMER.md`
      and linked back to the embed/rerank guide from the LLM guide.
    - The guide documents `qwen3-heretic-gguf` as a pure chat route backed by
      `qwen3-4b-heretic-q8` on local service port `8093`, and explicitly says
      not to use it with `/v1/embeddings`, `/rerank`, or `/v1/rerank`.
  - Verification:
    - `./.venv/bin/python -c "import pathlib, yaml; cfg=yaml.safe_load(open('litellm-dev-config.yaml')); route=next(m for m in cfg['model_list'] if m.get('model_name') == 'qwen3-heretic-gguf'); doc=pathlib.Path('LOCAL_LLM_CONSUMER.md').read_text(); assert 'qwen3-heretic-gguf' in doc; assert route['litellm_params']['model'].split('/', 1)[1] in doc; assert route['model_info']['mode'] == 'chat'; print('doc/config ok')"`
      passed.
    - `git diff --check -- LOCAL_LLM_CONSUMER.md LOCAL_EMBED_RERANK_CONSUMER.md`
      passed.

- D1-071 lean rate-limit observation schema and processed snapshot guard
  - Goal: replace the bulky rate-limit observation storage shape with the
    approved processed table fields and prevent repeated unchanged quota
    snapshots from accumulating.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - `public.rate_limit_observations` now stores the lean processed shape:
      observed/create times, client/version, account hash, provider, model,
      quota key/period/type, expected reset time, remaining percent, source,
      and join IDs.
    - Storage normalizes Google/Gemini quota rows to
      `provider=google`, `client=google_code_assist`; Codex and Claude remain
      `openai/codex` and `anthropic/claude`.
    - Existing rejected `claude_statusline` observation and transition rows are
      removed by schema migration.
    - The insert SQL now serializes writes by provider/client/account/quota
      identity with `pg_advisory_xact_lock` and skips the insert when the
      latest stored row has the same model, quota period/type, reset time, and
      remaining percentage.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'rate_limit_observations or rate_limit_transition or google_quota_buckets or repeated_rate_limit_snapshot or unchanged_latest_snapshot'`
      passed with `9 passed, 102 deselected`.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_backfill_rate_limit_observations.py -q`
      passed with `4 passed`.
    - `./.venv/bin/ruff check litellm/integrations/aawm_agent_identity.py --select E9,F821,F822,F823`
      passed.
    - `./.venv/bin/ruff check tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_backfill_rate_limit_observations.py scripts/backfill_rate_limit_observations.py`
      passed.
    - Applied schema cleanup to exact database `aawm_tristore`; live row count
      is `20`, source counts are `anthropic_response_headers=3`,
      `codex_response_headers=6`, `google_retrieve_user_quota=11`, and the
      duplicate snapshot group query returns zero rows.
    - Rolled-back DB insert verification in `aawm_tristore` returned
      `INSERT 0 0` for an unchanged latest Google snapshot and `INSERT 0 1`
      for a changed `remaining_pct`, then rolled back with row count unchanged.

- D1-071 rejected statusline capture cleanup
  - Goal: remove the rejected Claude/Codex statusline quota path so live quota
    observations can only come from LiteLLM pass-through streams, response
    headers, and provider error payloads.
  - Changed paths:
    - `/home/zepfu/.codex/config.toml`
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_scripts/test_backfill_rate_limit_observations.py`
    - `scripts/capture_claude_statusline_rate_limits.py` deleted
    - `tests/test_scripts/test_capture_claude_statusline_rate_limits.py` deleted
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - `/home/zepfu/.codex/config.toml` now has
      `status_line = ["model-with-reasoning", "context-remaining",
      "current-dir", "context-used"]`; the quota-related `five-hour-limit`
      item is removed from the active Codex status line.
    - `/home/zepfu/.claude/statusline-command.sh` has no capture hook and no
      reference to `capture_claude_statusline_rate_limits.py`; remaining
      `rate_limits` handling there is display-only.
    - `AawmAgentIdentity` no longer recognizes `claude_statusline`,
      `claude_statusline_input`, or `claude_code_statusline` metadata as quota
      observation roots and no longer calls a statusline extractor.
  - Verification:
    - `rg -n "five-hour-limit|capture_claude_statusline|claude_statusline|statusline-command|statusline capture" /home/zepfu/.codex/config.toml /home/zepfu/.claude/statusline-command.sh scripts tests/test_scripts litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py .analysis/todo.md .analysis/completed.md`
      returned only analysis-note references, no executable capture paths.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_backfill_rate_limit_observations.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'rate_limit or backfill_rate_limit or codex_rate_limit_metadata or anthropic_rate_limit_headers or google_adapter_records_retry_error or google_code_assist_prime_returns_sanitized_quota or google_code_assist_quota_logging_sanitizes_identifiers'`
      passed with `18 passed, 371 deselected, 71 warnings`.
    - Restarted dev `litellm-dev` with
      `docker compose -f docker-compose.dev.yml restart litellm-dev`;
      `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`, and logs showed `AawmAgentIdentity` initialized after the
      restart.
    - `git diff --check` passed.

- D1-071 pass-through quota capture paths
  - Goal: move live quota observation capture away from the Claude statusline
    side channel and into LiteLLM provider-originated pass-through
    streams/headers/errors for Codex/OpenAI, Claude/Anthropic, and
    Gemini/Google.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `litellm/proxy/pass_through_endpoints/streaming_handler.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Codex Responses stream logging now preserves direct `token_count` events
      and Codex wrapper events where `payload.type == token_count` and
      `rate_limits` is on the wrapper as `metadata.codex_token_count`.
    - Anthropic streaming passthrough now sanitizes upstream rate-limit headers
      into `metadata.anthropic_response_headers`; authorization and unrelated
      headers are not copied.
    - Google Code Assist now keeps sanitized `retrieveUserQuota` payloads and
      records retry `RESOURCE_EXHAUSTED` / `MODEL_CAPACITY_EXHAUSTED` payloads
      into request metadata so a later successful retry can persist them through
      normal callbacks.
    - `AawmAgentIdentity` extracts Anthropic header observations and Google
      retry metadata through the same `public.rate_limit_observations` processed
      insert path; there is no active statusline capture mechanism.
  - Verification:
    - `./.venv/bin/python -m pytest tests/test_scripts/test_backfill_rate_limit_observations.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'rate_limit or backfill_rate_limit or codex_rate_limit_metadata or anthropic_rate_limit_headers or google_adapter_records_retry_error or google_code_assist_prime_returns_sanitized_quota or google_code_assist_quota_logging_sanitizes_identifiers'`
      passed with `20 passed, 371 deselected, 71 warnings`.
    - `git diff --check` passed.
    - Restarted dev `litellm-dev` on `:4001` with
      `docker compose -f docker-compose.dev.yml restart litellm-dev`.
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`.
    - `docker logs --tail 80 litellm-dev` showed startup complete and
      `AawmAgentIdentity` initialized.
    - Exact database verification:
      `psql -Atqc "select current_database(), count(*) from public.rate_limit_observations;" postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`
      returned `aawm_tristore|12`.
  - Follow-up:
    - Live provider-originated smoke rows still need to be produced by real
      Codex/Claude/Gemini traffic after the patched dev container is running.
      Existing latest rows are historical observations from the rejected
      statusline experiment.

- D1-070 release-doc current-version de-hardcoding
  - Goal: keep durable release process docs from carrying stale "current
    promoted version" prose after each cutover.
  - Changed paths:
    - `PROD_RELEASE.md`
    - `TEST_HARNESS.md`
    - `PATCHES.md`
    - `COMPLETED.md`
    - `.analysis/completed.md`
  - Evidence:
    - `PROD_RELEASE.md` now says exact image, overlay, container, and artifact
      versions belong in release evidence, not the reusable runbook.
    - `TEST_HARNESS.md` now points to `scripts/local-ci/harness-version.txt`
      and the matching `h-v*` GitHub Release instead of naming a current
      harness release in prose.
    - `PATCHES.md` now points current release-state lookup to `pyproject.toml`,
      fork image tags, and published overlay releases.
    - `git diff --check` passed.
    - Targeted `rg` scan found no remaining hard-coded current release line in
      `PROD_RELEASE.md`, `TEST_HARNESS.md`, `PATCHES.md`, or `TODO.md`.

- D1-069 aawm.42 prod cutover
  - Goal: move the prepared `v1.82.3-aawm.42` release to the prod-style
    `aawm-litellm` container on `:4000` after explicit approval, then validate
    release-doc steps, local/OpenRouter embed/rerank session-history logging,
    and the real Claude CLI prod harness.
  - Changed paths:
    - `TODO.md`
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Deployment evidence:
    - `/home/zepfu/projects/aawm-infrastructure` commit `3eaf1e7` already
      pinned the standalone LiteLLM build to
      `ghcr.io/zepfu/litellm:1.82.3-aawm.42`.
    - `docker compose -f docker-compose.litellm.yml up -d litellm` recreated
      prod from the prepared image.
    - Running container:
      `9a393cd2e8592cbd875765fb57c7bc6ff2bc70c01a4c6dfa00ce403cee58600d`,
      image digest
      `sha256:4c983f7498352d8347313a1a84c42b983d379ad84fbe2809ade8684e1764d965`,
      started `2026-05-05T15:24:04.367723754Z`, Docker health `healthy`, port
      `127.0.0.1:4000->4000/tcp`.
  - Runtime evidence:
    - `/health/readiness` returned `status=healthy`,
      `litellm_version=1.82.3+aawm.42`, and success callbacks including
      `LangfusePromptManagement` and `AawmAgentIdentity`.
    - `docker exec aawm-litellm python3 -c ...` reported
      `litellm=1.82.3+aawm.42`, `aawm-litellm-callbacks=0.0.16`, and
      `aawm-litellm-control-plane=0.0.6`.
    - Rendered `/etc/litellm/config.yaml` includes
      `local_embed/nomic-embed-code.Q8_0.gguf`,
      `local_rerank/BAAI/bge-reranker-v2-m3`,
      `openrouter/qwen/qwen3-embedding-8b`,
      `openrouter/cohere/rerank-4-pro`,
      `openrouter/google/gemini-embedding-2-preview`,
      `nvidia_nim/nvidia/nv-embed-v1`, and
      `nvidia_nim/nvidia/rerank-qa-mistral-4b`.
  - Smoke evidence:
    - OpenRouter embed/rerank smoke session
      `prod-or-embed-rerank-1777994780` returned embedding dims `4096` and
      rerank results `2`.
    - Local embed/rerank smoke session
      `prod-local-embed-rerank-1777994797` returned embedding dims `3584` and
      rerank results `2`.
    - `public.session_history` in prod recorded rows for
      `openrouter/qwen/qwen3-embedding-8b`,
      `openrouter/cohere/rerank-4-pro`,
      `local_embed/nomic-embed-code.Q8_0.gguf`, and
      `local_rerank/BAAI/bge-reranker-v2-m3`, with
      `litellm_environment=prod`, `litellm_version=1.82.3+aawm.42`, and the
      expected local/OpenRouter cost values.
  - Harness evidence:
    - Focused prod artifact `/tmp/litellm-prod-focused-aawm42.json` ran the
      changed OpenRouter/free and peeromega lanes. `claude_adapter_gemma_31b`
      was warning-only due OpenRouter upstream 429/no row; peeromega failed
      only for the expected missing Codex Spark session row and forbidden 429
      log marker from quota pressure.
    - Default prod artifact
      `/tmp/litellm-prod-harness-1.82.3-aawm.42.json` passed
      `claude_adapter_gpt54`, `claude_adapter_gpt55`,
      `claude_adapter_gpt54_mini`, `claude_adapter_ctx_marker`,
      `claude_adapter_ctx_marker_escaped`, `claude_adapter_gemini_fanout`,
      `claude_adapter_gpt55_read_pages_sanitizer`,
      `claude_adapter_gemini31_pro`, `claude_adapter_openrouter_free`,
      `claude_adapter_nemotron_super`, and
      `claude_adapter_gemini31_flash`.
    - Default prod artifact failed only
      `claude_adapter_codex_tool_activity`,
      `claude_adapter_peeromega_fanout`, and `claude_adapter_spark`; logs and
      harness output show ChatGPT/Codex `usage_limit_reached` with
      `resets_at=1778018910` (`2026-05-05T22:08:30Z`).
  - Log evidence:
    - Post-harness `docker logs --tail 1200 aawm-litellm` scan found no
      `Content-Length`, `h11`, `KeyError: choices`, ASGI task, or database
      blockers.
    - Observed non-release-blocking noise: the expected Codex quota tracebacks,
      OpenRouter free-tier upstream 429 pacing, startup warning that
      `LITELLM_MASTER_KEY` is not set on this localhost-bound container, and one
      Anthropic `/messages/count_tokens` 404 for `model: openai/gpt-5.5`; the
      associated `claude_adapter_gpt55` message case passed.
  - Remaining follow-up:
    - Rerun the Codex/Spark harness slices after
      `2026-05-05T22:08:30Z` if a fully green prod default artifact is needed.

- D1-068 aawm.42 release-prep refresh after harness 0.0.27 autobump
  - Goal: supersede the stale pre-publication `aawm.41` candidate because the
    GitHub artifact autobump advanced `main` to `h-v0.0.27` after the
    `v1.82.3-aawm.41` tag was cut. The release runbook requires cutting a new
    image tag from current `main` instead of force-moving already-pushed tags.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `TEST_HARNESS.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Version metadata now targets `1.82.3+aawm.42`.
    - Release docs identified `v1.82.3-aawm.42` /
      `ghcr.io/zepfu/litellm:1.82.3-aawm.42` as the next prod candidate once
      published. This records the pre-cutover prep state; D1-069 records the
      later approved prod promotion.
    - Current overlay line is documented as `cb-v0.0.16`, `cp-v0.0.6`,
      `h-v0.0.27`, and `cfg-v0.0.9`.
    - The missing `h-v0.0.27` GitHub Release asset was built with
      `./.venv/bin/python scripts/local-ci/build_harness_bundle.py --outdir /tmp/aawm-h-dist`
      and published as
      `/tmp/aawm-h-dist/litellm-local-ci-harness-0.0.27.tar.gz` with SHA256
      `c866e2115f2f14f32a7e5cb73f2d8f73bac1df9311359acf9484469f9db86e5c`.
  - Deployment note:
    - `v1.82.3-aawm.42` is published at
      `https://github.com/zepfu/litellm/releases/tag/v1.82.3-aawm.42`.
    - `/home/zepfu/projects/aawm-infrastructure` commit `3eaf1e7` pins the
      standalone LiteLLM build to `ghcr.io/zepfu/litellm:1.82.3-aawm.42`.
    - Build-only validation passed with
      `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`.
      Built image `aawm-litellm:latest` has image id `4c983f749835`,
      `litellm=1.82.3+aawm.42`, `aawm-litellm-callbacks=0.0.16`,
      `aawm-litellm-control-plane=0.0.6`, and the expected local/OpenRouter/NVIDIA
      embed/rerank config and model-cost entries.
    - At prep time, prod `aawm-litellm` on `:4000` was not restarted. The
      then-running container was id `d23e6347f2da`, image digest
      `86d47957357f`, started at `2026-05-02T20:23:17.641026988Z`, healthy on
      `127.0.0.1:4000->4000/tcp`.
    - This prep-only deployment note was subsequently superseded by the
      approved D1-069 prod cutover.

- D1-067 aawm.41 release-prep refresh after harness autobump
  - Goal: supersede the stale pre-publication `aawm.40` candidate because the
    GitHub artifact autobump advanced `main` to `h-v0.0.26` after the
    `v1.82.3-aawm.40` tag was cut. This candidate was later superseded before
    image publication when the harness autobump advanced `main` to
    `h-v0.0.27`; use D1-068 / `aawm.42` for current release prep.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `TEST_HARNESS.md`
    - `scripts/local-ci/README.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Version metadata now targets `1.82.3+aawm.41`.
    - Release docs identify `v1.82.3-aawm.41` /
      `ghcr.io/zepfu/litellm:1.82.3-aawm.41` as the next prod candidate once
      published.
    - Current overlay line is documented as `cb-v0.0.16`, `cp-v0.0.6`,
      `h-v0.0.26`, and `cfg-v0.0.9`.
    - The missing `h-v0.0.26` GitHub Release asset was built with
      `./.venv/bin/python scripts/local-ci/build_harness_bundle.py --outdir /tmp/aawm-h-dist`
      and published as
      `/tmp/aawm-h-dist/litellm-local-ci-harness-0.0.26.tar.gz` with SHA256
      `87e59db3488ffc8c4356c39f15cef06dd36c78d4a868b83a0e7dc020a5f9c787`.
  - Deployment note:
    - Prod `aawm-litellm` on `:4000` was not restarted.

- D1-066 aawm.40 release-prep refresh for NVIDIA wildcard
  - Goal: supersede the published `aawm.39` candidate for current `develop`
    because D1-065 added core `nvidia/*` Claude adapter wildcard routing after
    the `aawm.39` image was cut. This candidate was later superseded before
    image publication when the harness autobump advanced `main`; use D1-067 /
    `aawm.41` for current release prep.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `TEST_HARNESS.md`
    - `CLAUDE.md`
    - `scripts/local-ci/README.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - At the time, version metadata targeted `1.82.3+aawm.40`.
    - At the time, release docs identified `v1.82.3-aawm.40` /
      `ghcr.io/zepfu/litellm:1.82.3-aawm.40` as the next prod candidate once
      published, and document that `aawm.39` remains historical because it
      predates explicit `nvidia/*` routing.
    - Focused source checks passed:
      `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`,
      `./.venv/bin/ruff check --ignore F401,T201,F841,F811,PLR0915,F541 litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`,
      `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q`
      (`273 passed`), and `git diff --check`.
    - Live dev artifact `/tmp/nvidia-qwen3-coder-wildcard-cli-4001.json`
      passed with real Claude CLI subagent traffic through
      `nvidia/qwen/qwen3-coder-480b-a35b-instruct`.
  - Deployment note:
    - Prod `aawm-litellm` on `:4000` was not restarted.

- D1-065 explicit NVIDIA wildcard routing for early model testing
  - Goal: let Claude/Anthropic adapter requests with an explicit `nvidia/*`
    model route to NVIDIA NIM without first adding the model to the local
    allowlist or config, while preserving existing bare allowlist behavior and
    current OpenRouter `nvidia/...:free` routing.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation:
    - `_normalize_anthropic_nvidia_responses_adapter_model_name()` now treats
      literal `nvidia/*` as operator intent and returns the stripped upstream
      model after existing alias normalization.
    - Unknown bare names, `nvidia_nim/*`, and `openrouter/*` do not get the
      wildcard behavior.
    - Existing OpenRouter namespace model
      `nvidia/nemotron-3-super-120b-a12b:free` is explicitly preserved for the
      OpenRouter adapter, avoiding a regression in current OpenRouter free-model
      canaries.
  - Evidence:
    - Used subagent fanout:
      - read-only explorer identified minimal resolver/test touchpoints and
        hazards around `nvidia_nim/*` and OpenRouter `nvidia/...:free`.
      - worker added focused route/resolver tests in the pass-through test file.
    - `./.venv/bin/python - <<'PY' ...` local sanity check proved
      `nvidia/acme/new-model -> acme/new-model`,
      `nvidia/minimax/minimax-m2.7 -> minimaxai/minimax-m2.7`,
      bare `acme/new-model -> None`, and
      `openrouter/acme/new-model -> None`.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_resolve_anthropic_nvidia_responses_adapter_model_supports_nvidia_prefix_and_aliases tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_resolve_anthropic_nvidia_responses_adapter_model_supports_unknown_nvidia_prefix tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_resolve_anthropic_nvidia_responses_adapter_model_skips_unknown_non_nvidia_prefixes tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_resolve_anthropic_nvidia_responses_adapter_model_preserves_openrouter_nvidia_namespace tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_anthropic_proxy_route_adapts_selected_nvidia_models_to_completion_adapter tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py::TestClaudePersistedOutputExpansion::test_anthropic_proxy_route_adapts_unknown_nvidia_prefixed_model_to_completion_adapter -q`
      passed: `16 passed`, with only existing pytest/backoff warnings.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'nvidia_responses_adapter_model or nvidia_models_to_completion_adapter or unknown_nvidia_prefixed_model or selected_openrouter_models_to_responses_adapter or unknown_openrouter_prefix' -q`
      passed: `18 passed, 255 deselected`, with only existing warnings.
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/ruff check --ignore F401,T201,F841,F811,PLR0915 litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed. The ignored rules are pre-existing noise in this large test file.
    - `git diff --check` passed.
  - Residual note:
    - Unknown NVIDIA chat models can route for early testing, but cost
      attribution may still be incomplete until a model-specific or fallback
      price entry exists.

- D1-064 focused `owl-alpha` explicit OpenRouter fanout live proof
  - Goal: validate that a Claude Code subagent named `owl-alpha` using the
    otherwise undefined `openrouter/owl-alpha` model routes through the explicit
    OpenRouter wildcard path on dev `:4001` and can issue tool calls, without
    running the broad fanout harness.
  - Changed paths:
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Created one temporary harness config at
      `/tmp/owl-alpha-harness-d876bb52c3ae.json` and artifact at
      `/tmp/owl-alpha-harness-d876bb52c3ae.artifact.json`.
    - Ran
      `LANGFUSE_PUBLIC_KEY=pk-lf-aawm-dev LANGFUSE_SECRET_KEY=sk-lf-aawm-dev AAWM_DB_PASSWORD=aawm_dev ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config /tmp/owl-alpha-harness-d876bb52c3ae.json --target dev --cases claude_adapter_owl_alpha_tool_fanout --write-artifact /tmp/owl-alpha-harness-d876bb52c3ae.artifact.json`.
    - Claude CLI exited successfully with final result
      `OWL ALPHA TOOL FANOUT PASSED`, session
      `e84170ee-bef0-40f1-8ebd-fdc8987ac695`, and model usage for
      `openrouter/owl-alpha`.
    - `public.session_history` in exact dev database `aawm_tristore` recorded
      the subagent row as `provider=openrouter`, `model=owl-alpha`,
      `litellm_environment=dev`, `client_name=claude-cli`, route family
      `anthropic_openrouter_responses_adapter`, counted shape
      `openai_responses`, `input_tokens=2195`, `output_tokens=11`, and
      `total_tokens=2206`.
    - Session history tags include `claude-agent:owl-alpha`,
      `route:anthropic_openrouter_responses_adapter`,
      `anthropic-openrouter-responses-adapter`,
      `anthropic-adapter-model:owl-alpha`, and
      `anthropic-adapter-target:openrouter:/v1/responses`.
    - Tool activity recorded one parent `Agent` tool call and four child
      OpenRouter tool calls: two `Read`, one `Bash` with
      `date -u +%Y-%m-%dT%H:%M:%SZ`, and one `Write` to
      `/tmp/owl-alpha-tool-probe-d876bb52c3ae.txt`.
    - Claude subagent transcript
      `/home/zepfu/.claude/projects/-home-zepfu-projects-litellm/e84170ee-bef0-40f1-8ebd-fdc8987ac695/subagents/agent-aeb63af0c2e39fc9d.jsonl`
      shows all four child tool calls in one assistant message, proving the
      requested parallel-read fanout behavior. Tool results were non-errors,
      including date output `2026-05-05T10:18:47Z`.
    - The write probe exists and contains exactly `OWL_ALPHA_WRITE_OK`.
    - Runtime checks passed: dev health returned `"I'm alive!"`, Docker status
      was `litellm-dev Up 10 hours`, and no forbidden runtime log substrings
      matched.
  - Residual issue:
    - The harness result is red only for generation-level observability quality:
      Langfuse observations were missing calculated cost/token fields and model
      resolved to unknown. Runtime logs explain the cause:
      `model=owl-alpha, custom_llm_provider=openrouter` is not mapped in
      `model_prices_and_context_window.json`, so OpenAI streaming passthrough
      cost tracking cannot calculate cost for this undefined OpenRouter model.
    - Follow-up is tracked in `.analysis/todo.md`; routing and tool execution
      were live-proven.

- D1-063 release-doc and infra-prep refresh for current `develop`
  - Goal: validate release docs and prep source/infra so the remaining
    production movement is publishing/using the current fork image and then
    rebuilding/recreating the prod container, without restarting prod now.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `TEST_HARNESS.md`
    - `WHEEL.md`
    - `/home/zepfu/projects/aawm-infrastructure/Dockerfile.litellm`
    - `/home/zepfu/projects/aawm-infrastructure/docker-compose.litellm.yml`
    - `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
  - Evidence:
    - Release docs now state current `develop` is post-`v1.82.3-aawm.38` and
      uses `v1.82.3-aawm.39` as the current fork image/tag before prod cutover.
    - `pyproject.toml` and Commitizen metadata now target `1.82.3+aawm.39`.
    - Published release/image `v1.82.3-aawm.39` /
      `ghcr.io/zepfu/litellm:1.82.3-aawm.39`.
    - Published missing overlay releases `cb-v0.0.16` and `cfg-v0.0.9`; existing
      `cp-v0.0.6` and `h-v0.0.25` remain current.
    - Infra standalone LiteLLM pin now targets
      `ghcr.io/zepfu/litellm:1.82.3-aawm.39`.
    - Infra config template exposes the seven local aliases
      `tei-medcpt-article`, `tei-medcpt-query`, `specter2-adapter`,
      `tei-indus`, `tei-sapbert`, `nomic-embed-code`, and `tei-reranker`
      through `host.docker.internal` with inline estimated costs matching dev.
    - GitHub Releases were verified for `v1.82.3-aawm.39`, `cb-v0.0.16`,
      `cp-v0.0.6`, `cfg-v0.0.9`, and `h-v0.0.25`.
    - Build-only infrastructure validation passed:
      `docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm`
      built local `aawm-litellm:latest` without restarting prod.
    - Image inspection reported `litellm=1.82.3+aawm.39`,
      `callbacks=0.0.16`, `control_plane=0.0.6`; the installed backup cost
      map contains `local_embed/nomic-embed-code.Q8_0.gguf` and
      `local_rerank/BAAI/bge-reranker-v2-m3`; the image config template
      contains the seven local route aliases and `host.docker.internal` bases.
  - Verification:
    - `./.venv/bin/python` TOML check confirmed both version fields are
      `1.82.3+aawm.39`.
    - `./.venv/bin/python` JSON check confirmed local cost entries in both
      model cost-map files.
    - `./.venv/bin/python` YAML check confirmed `litellm-dev-config.yaml` and
      `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
      parse cleanly, contain the seven local routes, and have no duplicate
      `model_name` entries.
    - `git diff --check` passed in both repositories.
  - Deployment note:
    - Prod `aawm-litellm` on `:4000` was not restarted. The next release
      action is to recreate/restart the container from
      `/home/zepfu/projects/aawm-infrastructure` when approved, then run prod
      validation.

## 2026-05-04

- D1-062 explicit OpenRouter Claude adapter wildcard
  - Goal: let Claude/Anthropic passthrough requests that explicitly name an
    `openrouter/*` model route to OpenRouter even when that model is not in the
    hard-coded OpenRouter adapter allowlist, without changing OpenRouter
    embedding/rerank routing.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
  - Evidence:
    - Added a narrow fallback in
      `_resolve_anthropic_openrouter_responses_adapter_model()` that only
      accepts models whose original provider prefix is explicitly
      `openrouter`, then forwards the stripped OpenRouter model name upstream.
    - Existing allowlisted models still resolve first; `openrouter/elephant-alpha`
      remains on the OpenRouter chat-completions adapter because that resolver
      runs before the Responses adapter resolver.
    - Added resolver coverage proving `openrouter/acme/new-model` resolves to
      `acme/new-model`.
    - Added route coverage proving `openrouter/acme/new-model` targets
      `https://openrouter.ai/api/v1/responses`, sets provider `openrouter`,
      disables header forwarding, and sends `custom_body.model=acme/new-model`.
    - Fixed two existing OpenRouter route-test fixtures so they return valid
      non-empty OpenRouter Responses bodies instead of triggering the
      empty-success diagnostic path.
    - Committed and pushed to `origin/develop`:
      `649cb61b6f Allow explicit OpenRouter Claude adapter models`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'unknown_openrouter_prefix or supports_openrouter_prefix or adapts_openrouter_prefixed_gemma_model_to_responses or adapts_prefixed_openrouter_free_model_to_responses or adapts_selected_openrouter_free_models_to_responses'`
      passed: 12 selected tests, 255 deselected, pre-existing warnings.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q`
      passed: 267 tests, pre-existing warnings.
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
      passed.
    - `./.venv/bin/ruff check --ignore PLR0915,T201,F401,F841,F811,F541 tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed. The extra ignores cover pre-existing findings in the large test
      file.
    - `git diff --check` passed.
    - Recreated only dev `litellm-dev` on `:4001`; prod `aawm-litellm` on
      `:4000` was not restarted. `curl -sS http://127.0.0.1:4001/health/readiness`
      returned `status=healthy`.
    - Full default dev harness against `develop` commit
      `649cb61b6f3ea9840cdc9e4d2bc78482780e014d` wrote artifact
      `/tmp/litellm-dev-harness-2026-05-04-openrouter-wildcard.json`.
      Non-Codex lanes passed, including `claude_adapter_openrouter_free` and
      `claude_adapter_nemotron_super`. The suite overall remained red only in
      Codex-dependent cases: `claude_adapter_codex_tool_activity`,
      `claude_adapter_peeromega_fanout` missing the Codex child row, and
      `claude_adapter_spark`.
    - Dev harness and logs classify the red cases as the known Codex
      `usage_limit_reached` path for `gpt-5.3-codex-spark`, with
      `resets_at=1778018910` (`2026-05-05T22:08:30Z`).
    - No-restart prod readiness checks showed `:4000` readiness/liveliness
      healthy, running container `aawm-litellm` healthy on
      `1.82.3+aawm.37`, and no recent prod log matches for the release-blocker
      scan pattern. Release assets for `v1.82.3-aawm.38`, `cb-v0.0.15`,
      `cp-v0.0.6`, `cfg-v0.0.8`, and `h-v0.0.25` exist.
  - Deployment note:
    - The currently pinned infrastructure base image
      `ghcr.io/zepfu/litellm:1.82.3-aawm.38` predates commit `649cb61b6f`.
      A prod cutover that includes this OpenRouter wildcard needs a new fork
      image/tag from the updated code before rebuilding/restarting
      `/home/zepfu/projects/aawm-infrastructure`.

## 2026-05-02

- D1-061 Nomic local code embedding route
  - Goal: add the local Nomic code embedding endpoint to `litellm-dev` as a
    `local_embed` route, with estimated commercial-equivalent pricing and
    session-history cost visibility.
  - Changed paths:
    - `litellm-dev-config.yaml`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `tests/test_litellm/test_cost_calculator.py`
    - `LOCAL_EMBED_RERANK_CONSUMER.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added LiteLLM alias `nomic-embed-code` backed by
      `local_embed/nomic-embed-code.Q8_0.gguf` at
      `http://172.20.0.1:8082/v1` for the Docker-hosted dev proxy.
    - Added model-cost metadata for
      `local_embed/nomic-embed-code.Q8_0.gguf`: `mode=embedding`,
      provider `local_embed`, `max_input_tokens=32768`,
      `output_vector_size=3584`, and requested estimated input price
      `$0.15/M tokens` (`1.5e-07` per token).
    - Added the same Nomic catalog assertion to
      `test_cost_calculator_local_embedding_catalog_metadata`.
    - Direct local endpoint probe `curl -sS http://127.0.0.1:8082/v1/models`
      reported `nomic-embed-code.Q8_0.gguf`, `n_ctx_train=32768`, and
      `n_embd=3584`.
    - Recreated only `litellm-dev` with
      `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`;
      `curl -sS http://127.0.0.1:4001/health/readiness` returned
      `status=healthy`.
    - `curl -sS "http://127.0.0.1:4001/health?model=nomic-embed-code"`
      returned `healthy_count=1`, `unhealthy_count=0`, and a healthy endpoint
      for `local_embed/nomic-embed-code.Q8_0.gguf`.
    - Live proxy embedding smoke to `POST /v1/embeddings` with model
      `nomic-embed-code` returned `VECTOR_DIM=3584` and usage
      `prompt_tokens=10`, `total_tokens=10`.
    - Dev database verification against exact database `aawm_tristore` found
      the live smoke row for session `local-nomic-2600ef8ca49f`:
      `model_group=nomic-embed-code`, `provider=local_embed`,
      `model=nomic-embed-code.Q8_0.gguf`, `call_type=aembedding`,
      `input_tokens=10`, `total_tokens=10`, and
      `response_cost_usd=1.5e-06`.
    - Added root consumer guide `LOCAL_EMBED_RERANK_CONSUMER.md` covering all
      current local embedding and rerank aliases, attribution headers, example
      proxy calls, local-service ownership notes, and estimated token prices.
  - Verification:
    - `./.venv/bin/python -m py_compile tests/test_litellm/test_cost_calculator.py`
      passed.
    - `./.venv/bin/python -c 'import json, yaml; ...'` parsed both model-cost
      JSON files and `litellm-dev-config.yaml`.
    - `./.venv/bin/python -m pytest tests/test_litellm/test_cost_calculator.py -q -k local_embedding`
      passed: 7 tests selected, 80 deselected, 1 pre-existing pytest config
      warning.
    - `./.venv/bin/ruff check --ignore T201 tests/test_litellm/test_cost_calculator.py`
      passed. `T201` is ignored for pre-existing prints in the file.
    - `git diff --check` passed.

- Local embedding/rerank session-history cost parity
  - Goal: make the new `local_embed` / `local_rerank` dev routes follow the
    same session-history logging and estimated-cost path used by OpenRouter
    embedding/rerank routes.
  - Changed paths:
    - `docker-compose.dev.yml`
    - `litellm-dev-config.yaml`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `litellm/integrations/aawm_agent_identity.py`
    - `litellm/llms/base_llm/rerank/transformation.py`
    - `litellm/llms/huggingface/rerank/transformation.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/test_cost_calculator.py`
  - Evidence:
    - Added local model-cost map entries for the five local embedding routes
      and the local BGE reranker, including `mode`, `litellm_provider`,
      token limits, vector size for embeddings, and the requested estimated
      commercial prices:
      - MedCPT Article, Specter2, SapBERT: `$0.0046/M` input tokens
      - MedCPT Query: `$0.0028/M` input tokens
      - Indus: `$0.0056/M` input tokens
      - BGE reranker: `$0.025/M` input tokens
    - Added the same pricing to `litellm-dev-config.yaml` model info so proxy
      model metadata exposes the local estimates.
    - Updated rerank usage extraction so LiteLLM rerank responses with
      `meta.tokens.input_tokens` populate `session_history.input_tokens`.
    - Updated base rerank cost calculation to prefer
      `billed_units.total_tokens * input_cost_per_token` before falling back to
      query pricing.
    - Updated the HuggingFace/TEI rerank transformer to include estimated
      `total_tokens` in `meta.billed_units`, while still keeping
      `search_units=1`.
    - Recreated only `litellm-dev` with
      `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`.
    - Live smoke session
      `local-logging-cost-smoke-bd8c6cf0-354b-4041-b52d-54a1b6ce44cd`
      wrote 6 rows to exact database `aawm_tristore`:
      - `tei-medcpt-article`: `local_embed`,
        `ncbi/MedCPT-Article-Encoder`, `input_tokens=11`,
        `response_cost_usd=5.06e-08`
      - `tei-medcpt-query`: `local_embed`,
        `ncbi/MedCPT-Query-Encoder`, `input_tokens=11`,
        `response_cost_usd=3.08e-08`
      - `specter2-adapter`: `local_embed`, `allenai/specter2_base`,
        `input_tokens=11`, `response_cost_usd=5.06e-08`
      - `tei-indus`: `local_embed`, `nasa-impact/nasa-ibm-st.38m`,
        `input_tokens=12`, `response_cost_usd=6.72e-08`
      - `tei-sapbert`: `local_embed`,
        `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `input_tokens=11`,
        `response_cost_usd=5.06e-08`
      - `tei-reranker`: `local_rerank`, `BAAI/bge-reranker-v2-m3`,
        `input_tokens=24`, `response_cost_usd=6e-07`
    - `docker exec litellm-dev python -c ... litellm.get_model_info(...)`
      confirmed local model info in the running proxy process includes
      provider/mode/pricing/vector-size fields.
    - `curl -sS http://127.0.0.1:4001/health` reported
      `healthy_count=40`, `unhealthy_count=0`, with all six local endpoints in
      `healthy_endpoints`.
  - Verification:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py litellm/llms/base_llm/rerank/transformation.py litellm/llms/huggingface/rerank/transformation.py tests/test_litellm/test_cost_calculator.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -c 'import json, yaml; ...'` parsed both model-cost
      JSON files, `litellm-dev-config.yaml`, and `docker-compose.dev.yml`.
    - `./.venv/bin/ruff check --ignore PLR0915,T201 ...` passed for changed
      Python/test files. The ignores cover pre-existing complexity/print
      findings in these files.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed: 101 tests, 1 pre-existing pytest config warning.
    - `./.venv/bin/python -m pytest tests/test_litellm/test_cost_calculator.py -q -k 'local_embedding or local_rerank or openrouter_qwen3_embedding_8b or openrouter_cohere_rerank_4_pro'`
      passed: 9 tests selected, 77 deselected, 1 pre-existing pytest config
      warning.
    - `./.venv/bin/python -m pytest tests/test_litellm/llms/huggingface/rerank/test_huggingface_rerank_transformation.py -q`
      passed: 12 tests, with pre-existing warnings.
    - `./.venv/bin/python -m pytest tests/test_litellm/test_cost_calculator.py -q -k 'rerank'`
      passed: 9 tests selected, 77 deselected, 1 pre-existing pytest config
      warning.

- Local TEI embedding and reranker routes added to `litellm-dev`
  - Goal: configure the locally running embedding and reranker services for use
    through the dev LiteLLM proxy on `:4001`.
  - Changed paths:
    - `docker-compose.dev.yml`
    - `litellm-dev-config.yaml`
    - `litellm/types/utils.py`
    - `litellm/utils.py`
    - `litellm/main.py`
    - `litellm/rerank_api/main.py`
    - `litellm/llms/hosted_vllm/embedding/transformation.py`
    - `.analysis/completed.md`
  - Evidence:
    - Added six separate dev model groups:
      `tei-medcpt-article`, `tei-medcpt-query`, `specter2-adapter`,
      `tei-indus`, `tei-sapbert`, and `tei-reranker`.
    - Added LiteLLM-local provider aliases `local_embed` and `local_rerank`;
      the five embedding routes now use `local_embed/...` model params with
      `api_base` values on the Docker host gateway:
      `http://172.20.0.1:8083/v1`, `:8084/v1`, `:8086/v1`, `:8087/v1`, and
      `:8088/v1`.
    - The reranker route now uses `local_rerank/BAAI/bge-reranker-v2-m3` with
      `api_base: http://172.20.0.1:8090`.
    - `local_embed` reuses the TEI-compatible embedding HTTP transformation, and
      `local_rerank` reuses the TEI-compatible rerank transformation internally
      so the upstream service receives `texts` rather than a Hosted VLLM
      `documents` body.
    - `docker-compose.dev.yml` now bind-mounts the provider enum and local
      embedding transformation files so `litellm-dev` loads the alias code from
      the workspace.
    - Recreated only `litellm-dev` with
      `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`.
    - `/v1/models` on `http://127.0.0.1:4001` listed all six new model names.
    - Live proxy calls through `/v1/embeddings` returned vector dimensions:
      `tei-medcpt-article=768`, `tei-medcpt-query=768`,
      `specter2-adapter=768`, `tei-indus=576`, and `tei-sapbert=768`.
    - Live proxy call through `/rerank` for `tei-reranker` returned ranked
      results with the Washington, D.C. document first.
    - `curl -sS http://127.0.0.1:4001/health/readiness` returned
      `status=healthy`.
    - `curl -sS http://127.0.0.1:4001/health` reported `healthy_count=40` and
      `unhealthy_count=0`.
  - Verification:
    - `./.venv/bin/python - <<'PY' ... yaml.safe_load(open("litellm-dev-config.yaml")) ... PY`
      confirmed YAML validity, `40` model entries, and no duplicate model
      names.
    - Host-side LiteLLM probes confirmed `local_embed/...` and
      `local_rerank/...` work against the local services before proxy restart.
    - Live proxy discovery and embedding/rerank requests passed after the
      `litellm-dev` recreate.

- Release-readiness dev harness and OpenAI Responses reasoning-input sanitizer
  - Goal: review prod cutover docs/state without restarting prod, run the full
    dev adapter harness on `:4001`, and resolve any local hard blocker that can
    be fixed before a later prod cutover.
  - Changed paths:
    - `litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Reviewed `PROD_RELEASE.md`, `TEST_HARNESS.md`, `CLAUDE.md`, and the
      `/home/zepfu/projects/aawm-infrastructure` LiteLLM compose/Dockerfile/config
      files. Prod `aawm-litellm` on `:4000` was not restarted or recreated.
    - Confirmed infra source pins `ghcr.io/zepfu/litellm:1.82.3-aawm.38`, while
      running prod still reports `1.82.3+aawm.37`.
    - Verified overlay releases exist with assets: `cb-v0.0.15`, `cp-v0.0.6`,
      `cfg-v0.0.8`, and `h-v0.0.25`.
    - Verified `aawm-infrastructure/.env` has the key names required by the
      current config template (`NVIDIA_NIM_API_KEY` and `AAWM_OPENROUTER_API_KEY`)
      without relying on prod restart.
    - Ran the full dev adapter harness:
      `./.venv/bin/python -m dotenv run -- ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --target dev --write-artifact /tmp/litellm-dev-harness-2026-05-02-release-readiness.json`.
      It completed with `passed=false`: 9 of 14 cases passed, with hard failures
      concentrated in Codex quota-dependent cases plus one Responses sanitizer
      request-shape bug. The Codex upstream reset time in the 429 payload was
      2026-05-05 22:08:30 UTC.
    - Fixed the Responses sanitizer failure by omitting synthetic `status` from
      Anthropic thinking blocks translated into Responses `input` reasoning
      items. Official OpenAI guidance says returned output items can be passed
      back for stateless flows, but `status` is populated when items are returned
      by the API; the ChatGPT/Codex Responses backend rejected our synthetic
      `input[1].status`.
    - Recreated only `litellm-dev` on `:4001` and did not touch prod. Focused
      live verification passed:
      `./.venv/bin/python -m dotenv run -- ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --target dev --cases claude_adapter_gpt55_read_pages_sanitizer --write-artifact /tmp/litellm-dev-gpt55-read-pages-sanitizer-2026-05-02.json`.
    - `curl -sS http://127.0.0.1:4001/health` reported `healthy_count=34` and
      `unhealthy_count=0` after the focused fix verification.
    - Verification:
      - `./.venv/bin/python -m py_compile litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'responses_adapter_reasoning_input_omits_status or responses_adapter_codex_defaults_alias_bash_to_exec_command' -q`
      - `./.venv/bin/ruff check litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py`
    - Targeted Ruff on the large existing pass-through test file still reports
      pre-existing lint debt unrelated to this patch.

- Restored persistent `litellm-dev` runtime on `:4001`
  - Goal: start the stopped dev proxy and configure it to come back after host
    restart.
  - Changed paths:
    - `docker-compose.dev.yml`
    - `litellm-dev-config.yaml`
    - `litellm/main.py`
    - `litellm/utils.py`
    - `litellm/llms/nvidia_nim/embed.py`
    - `litellm/llms/nvidia_nim/rerank/transformation.py`
    - `litellm/litellm_core_utils/health_check_helpers.py`
    - `litellm/rerank_api/main.py`
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `tests/llm_translation/test_nvidia_nim.py`
    - `tests/test_litellm/test_cost_calculator.py`
    - `tests/test_litellm/proxy/test_health_check_max_tokens.py`
    - `.analysis/completed.md`
  - Evidence:
    - Added `restart: always` to the `litellm-dev` compose service.
    - `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`
      recreated and started the service.
    - `docker inspect litellm-dev --format '{{json .HostConfig.RestartPolicy}} {{.State.Status}}'`
      reported `{"Name":"always","MaximumRetryCount":0} running`.
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`.
    - `curl -sS http://127.0.0.1:4001/health/readiness` returned
      `{"status":"healthy", ... "litellm_version":"1.82.3+aawm.25", ...}`.
    - `systemctl is-enabled docker` reported `enabled`, so Docker itself is
      configured to start at host boot.
    - Full `/health` reported `healthy_count=29` and `unhealthy_count=7`; all
      unhealthy checks were direct NVIDIA NIM embedding/rerank routes failing
      with upstream `401` or `404` responses, not proxy startup failures.
    - Follow-up NVIDIA env check found `.env` defines `AAWM_NVIDIA_API_KEY`
      while `litellm-dev-config.yaml` uses `os.environ/NVIDIA_NIM_API_KEY`.
      Updated `docker-compose.dev.yml` so `NVIDIA_NIM_API_KEY` and
      `NVIDIA_API_KEY` fall back to `AAWM_NVIDIA_API_KEY`, recreated
      `litellm-dev`, and verified the running container reports all three
      NVIDIA key env vars as set without printing secret values.
    - After key propagation, the direct NVIDIA `401` health checks changed to
      non-auth failures: `nv-embed-v1` and `nv-embedcode-7b-v1` now fail on
      `encoding_format=None` validation, while the remaining direct NVIDIA
      retriever/rerank routes still return upstream `404`.
    - Fixed NVIDIA embedding health serialization so `encoding_format: null` is
      omitted, health-only NVIDIA embedding checks send `encoding_format=float`
      and `input_type=query`, and embedding health checks no longer leak
      `max_tokens` into the NVIDIA embedding body.
    - Fixed hosted NVIDIA rerank routing so rerank does not inherit
      `NVIDIA_NIM_API_BASE=https://integrate.api.nvidia.com/v1`; it now uses
      `NVIDIA_NIM_RERANK_API_BASE` or the hosted default
      `https://ai.api.nvidia.com`.
    - Fixed NVIDIA hosted rerank model mapping for the shared Mistral reranking
      endpoint and the legacy body model `nv-rerank-qa-mistral-4b:1`.
    - Live probes showed `nvidia/llama-3_2-nemoretriever-300m-embed-v1` returns
      `404`, while `nvidia/llama-3.2-nemoretriever-300m-embed-v1` succeeds.
      Updated the dev alias to forward to the dotted model id and added that
      dotted id to both model-cost maps.
    - Live probes showed the hosted shared rerank endpoint currently exposes
      `nvidia/rerank-qa-mistral-4b` / `nv-rerank-qa-mistral-4b:1` for this key,
      while `nvidia/nv-rerankqa-mistral-4b-v3` returns `Unknown model`. Removed
      the v3 route from dev health coverage.
    - Added bind mounts for the changed NVIDIA and health-check modules so
      `litellm-dev` runs the local fixes instead of stale image-baked files.
    - Recreated `litellm-dev`; host-side
      `./.venv/bin/python -c '... urllib.request.urlopen("http://127.0.0.1:4001/health") ...'`
      reported `34 0`.
    - Verification:
      - `./.venv/bin/python -m py_compile litellm/utils.py litellm/main.py litellm/llms/nvidia_nim/embed.py litellm/litellm_core_utils/health_check_helpers.py litellm/rerank_api/main.py litellm/llms/nvidia_nim/rerank/transformation.py tests/llm_translation/test_nvidia_nim.py tests/test_litellm/proxy/test_health_check_max_tokens.py`
      - `./.venv/bin/python -m pytest tests/llm_translation/test_nvidia_nim.py -k 'embedding_nvidia_nim or nvidia_nim_rerank_uses_hosted_base or nvidia_nim_rerank_mistral_models' tests/test_litellm/proxy/test_health_check_max_tokens.py -q`
      - `./.venv/bin/python -m pytest tests/test_litellm/proxy/test_health_check_max_tokens.py -q`
      - `./.venv/bin/python -m pytest tests/test_litellm/test_cost_calculator.py -k nvidia_free_endpoint_catalog_metadata -q`
      - JSON validation for `model_prices_and_context_window.json` and
        `litellm/bundled_model_prices_and_context_window_fallback.json`.
      - `./.venv/bin/python -c 'import yaml; yaml.safe_load(open("litellm-dev-config.yaml")); yaml.safe_load(open("docker-compose.dev.yml")); print("yaml ok")'`
      - `./.venv/bin/ruff check litellm/utils.py litellm/main.py litellm/llms/nvidia_nim/embed.py litellm/litellm_core_utils/health_check_helpers.py litellm/rerank_api/main.py litellm/llms/nvidia_nim/rerank/transformation.py tests/test_litellm/proxy/test_health_check_max_tokens.py`
      - `./.venv/bin/ruff check --ignore T201,F401,F811,F841,E712 tests/llm_translation/test_nvidia_nim.py`
      - `./.venv/bin/ruff check --ignore T201 tests/test_litellm/test_cost_calculator.py`
    - Targeted Ruff without ignores on `tests/llm_translation/test_nvidia_nim.py`
      still reports pre-existing unused imports, prints, and style warnings in
      that legacy test file.

## 2026-05-01

- Prod release re-prep after model-config additions
  - Goal: make the release state explicit after the OpenRouter/NVIDIA catalog
    work landed after the `aawm.38` base image was published.
  - Changed paths:
    - `PROD_RELEASE.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `WHEEL.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Release docs now describe `ghcr.io/zepfu/litellm:1.82.3-aawm.38` as the
      base image candidate and `cfg-v0.0.8` as the published overlay carrying
      the OpenRouter/NVIDIA rerank+embedding catalog work.
    - Prod model-config verification snippets check representative OpenRouter
      embedding/rerank and NVIDIA NIM embedding/rerank entries.
    - `cfg-v0.0.8` release asset
      `litellm-model-config-0.0.8.tar.gz` is published with asset digest
      `sha256:a6d2ab4d3cfb089612a510c6d3c21429c4222dacc9ccdf08620c7cc8312d9768`;
      its manifest reports source SHA-256
      `19c727f5ab997701bf7db129e4628a1355d4f5e24530ae6b52a8f3a1e75ca4ea` and
      `2651` model entries.
    - `h-v0.0.25` release asset
      `litellm-local-ci-harness-0.0.25.tar.gz` is published with asset digest
      `sha256:51a82b855d4df3b9924ef70f16ce46dd6dd72a0bd8f2cc27286d576df0fbf836`.
    - Remaining prod work is still infrastructure build/restart/validation only
      and remains gated on explicit approval.

- OpenRouter/NVIDIA embedding and rerank config support
  - Goal: expose the OpenRouter rerank/embedding catalog entries and NVIDIA NIM
    free endpoint embedding/rerank models discussed during release prep through
    model metadata, dev config, and the infrastructure config template.
  - Changed paths:
    - `model_prices_and_context_window.json`
    - `litellm/bundled_model_prices_and_context_window_fallback.json`
    - `litellm-dev-config.yaml`
    - `tests/test_litellm/test_cost_calculator.py`
    - `TODO.md`
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `/home/zepfu/projects/aawm-infrastructure/config/litellm-config.yaml.tmpl`
  - Evidence:
    - OpenRouter model map/config coverage includes all current rerank entries
      and the embedding entries captured from the OpenRouter API response.
    - NVIDIA direct config coverage includes
      `nvidia/rerank-qa-mistral-4b`,
      `nvidia/nv-rerankqa-mistral-4b-v3`,
      `nvidia/llama-3_2-nv-rerankqa-1b-v2`,
      `ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2`,
      `nvidia/nv-embedcode-7b-v1`, `nvidia/nv-embed-v1`, and
      `nvidia/llama-3_2-nemoretriever-300m-embed-v1`.
    - Dev NVIDIA routes use `NVIDIA_NIM_API_KEY`, matching the infrastructure
      template.
    - `make check-model-cost-map-sync`
    - `./.venv/bin/python -m pytest tests/test_litellm/test_cost_calculator.py -k 'openrouter_embedding_catalog_metadata or openrouter_rerank_catalog_metadata or nvidia_free_endpoint_catalog_metadata or openrouter_qwen3_embedding_8b_metadata_and_cost or openrouter_cohere_rerank_4_pro_metadata_and_cost' -q` (`37 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/llms/openrouter/rerank/test_openrouter_rerank_transformation.py tests/test_litellm/llms/openrouter/test_openrouter_embedding_transformation.py -q` (`10 passed`)
    - `./.venv/bin/python -m pytest tests/llm_translation/test_nvidia_nim.py -k 'test_embedding_nvidia_nim or test_nvidia_nim_rerank_ranking_endpoint' -q` (`2 passed`)
    - JSON validation for both model map files.
    - YAML validation and duplicate checks for `litellm-dev-config.yaml` (`36`
      entries) and the infrastructure template (`79` entries).
    - `./.venv/bin/ruff check --ignore T201 tests/test_litellm/test_cost_calculator.py`
    - `git diff --check`
    - `git -C /home/zepfu/projects/aawm-infrastructure diff --check`
  - Follow-up:
    - Publish the next model-config artifact and rebuild/deploy infrastructure
      only when prod promotion is approved. No running prod process was touched.

- aawm.38 release candidate prepared for prod cutover
  - Goal: finish release-prep bookkeeping without touching production
    infrastructure.
  - Changed paths:
    - `pyproject.toml`
    - `PATCHES.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `PROD_RELEASE.md`
    - `TEST_HARNESS.md`
    - `scripts/local-ci/README.md`
    - `WHEEL.md`
  - Evidence:
    - The published candidate was cut from `b022a0271c`.
    - `v1.82.3-aawm.38` exists and publishes
      `ghcr.io/zepfu/litellm:1.82.3-aawm.38`.
    - `Build and publish AAWM fork image` run `25209548874` completed
      successfully for head `b022a0271c`.
    - GitHub Releases/assets verified:
      - `cb-v0.0.15` with
        `aawm_litellm_callbacks-0.0.15-py3-none-any.whl`
      - `cp-v0.0.6` with
        `aawm_litellm_control_plane-0.0.6-py3-none-any.whl`
      - `h-v0.0.24` with `litellm-local-ci-harness-0.0.24.tar.gz`
      - `cfg-v0.0.7` with `litellm-model-config-0.0.7.tar.gz`
    - Prod remains untouched and still needs the infrastructure image pin
      update, no-cache build, package-version inspection, restart, and focused
      plus default prod harness validation on `:4000`.

- Documentation audit for harness and prompt-overhead work
  - Goal: reconcile `TEST_HARNESS.md`, bundle-local harness notes, release
    docs, and `.analysis` continuity notes with the current adapter/native
    harness behavior.
  - Changed paths:
    - `TEST_HARNESS.md`
    - `scripts/local-ci/README.md`
    - `WHEEL.md`
    - `CLAUDE.md`
    - `PROD_RELEASE.md`
    - `PATCHES.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/codex-litellm-session-handoff-2026-04-10.md`
    - `.analysis/session-history-persistence-design-2026-04-16.md`
    - `.analysis/local-acceptance-suite-design-2026-04-10.md`
    - `.analysis/remote.md`
  - Evidence:
    - Docs now distinguish standalone `h-v*` baseline harness contents from the
      repo-local Anthropic adapter/native harness.
    - Current hard gates, warning-only cases, native cases, peeromega Agent row
      count, and `claude_adapter_gpt_oss_120b` soft-fail semantics match
      `scripts/local-ci/anthropic_adapter_config.json`.
    - Release/runbook docs now mention prompt-overhead columns and
      `summary.prompt_overhead_cost_share` where relevant.
    - Validation commands are recorded in the final turn.

- Documentation-test cleanup from full-suite collection
  - Goal: close the actionable documentation regressions found while rerunning
    the full repo test suite.
  - Changed paths:
    - `docs/my-website/docs/proxy/config_settings.md`
    - `docs/my-website/docs/exception_mapping.md`
    - `tests/documentation_tests/test_exception_types.py`
    - `COMPLETED.md`
    - `TODO.md`
    - `.analysis/completed.md`
    - `.analysis/todo.md`
  - Evidence:
    - Added missing env/general-setting rows to `config_settings.md`.
    - Added public `RejectedRequestError` and `BadGatewayError` rows to
      `exception_mapping.md`.
    - Made `test_exception_types.py` independent of the shell working
      directory and normalized table parsing to documented error type names.
    - `./.venv/bin/python tests/documentation_tests/test_env_keys.py`
    - `./.venv/bin/python tests/documentation_tests/test_general_setting_keys.py`
    - `./.venv/bin/python tests/documentation_tests/test_exception_types.py`
    - `./.venv/bin/ruff check tests/documentation_tests/test_exception_types.py`
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` (`99 passed`)
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q` (`56 passed`)
    - `./.venv/bin/python -m json.tool scripts/local-ci/anthropic_adapter_config.json`
    - `./.venv/bin/python -m json.tool scripts/local-ci/config.json`
    - `git diff --check`
  - Full-suite note:
    - `make test` still stops during collection with `18172 items / 83 errors`;
      this is improved from the earlier `86 errors` by removing the three doc
      regressions. Remaining errors are broad-suite/environment blockers:
      duplicate module basenames in monolithic collection, missing optional deps
      such as `PIL` and `google.genai`, missing live credentials, old proxy/live
      tests executing at import time, and the pre-existing Vertex vector-store
      transformation import gap.

## 2026-04-30

- D1-060 prompt-overhead cost-share report and native live coverage
  - Goal: make the new prompt-overhead fields useful for comparative CLI
    overhead analysis by adding a repeatable harness-level report and proving it
    on live dev `:4001` traffic.
  - Changed paths:
    - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
    - `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
    - `scripts/local-ci/README.md`
    - `TODO.md`
    - `COMPLETED.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added `summary.prompt_overhead_cost_share` to the Anthropic adapter/native
      passthrough harness artifact. The report groups by case, client, route
      family, counted shape, environment, provider, and model; aggregates
      system/tool/conversation/other/residual/system-classifier token buckets;
      tracks breakdown coverage; and labels cost allocation as estimated because
      `session_history` currently stores total `response_cost_usd`, not exact
      input cost.
    - Added focused unit coverage for report grouping, proportional estimated
      cost allocation, unestimated/zero-input rows, and the shared-session
      OpenAI passthrough case where the report must prefer the selected row
      instead of double-counting prior rows from the same session.
    - `./.venv/bin/python -m py_compile scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q` (`56 passed`)
    - `./.venv/bin/ruff check --ignore PLR0915,T201 scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py` (`All checks passed`; `PLR0915`/`T201` are ignored for pre-existing harness complexity/prints)
    - `./.venv/bin/python -m json.tool scripts/local-ci/anthropic_adapter_config.json >/tmp/anthropic_adapter_config.formatted.json`
    - Confirmed `litellm-dev` was running mounted latest
      `litellm/integrations/aawm_agent_identity.py` by matching SHA-256
      `94d6530a637c4b62fd3cb1a482c7d510d3bd321d3849b3be9bfd09a75198a7ef`
      in the repo and container; container start time remained
      `2026-04-30T18:42:32.124879035Z`.
    - `AAWM_OBSERVE_SERVICE_NAME=pytest-classifier-scan ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config scripts/local-ci/anthropic_adapter_config.json --target dev --cases native_openai_passthrough_responses_codex --write-artifact /tmp/native_codex_4001_prompt_overhead_cost_share.json` passed with zero failures/warnings. Artifact trace `44babd14-58e7-46ed-875e-6aedd6f03e81`, session `019ddfcd-8333-7732-848e-c4d5b4f4aee4`, showed explicit system+tool overhead `14220 / 26101 = 0.544807` and plus-other overhead `20866 / 26101 = 0.799433`.
    - Exact dev DB verification against `aawm_tristore` found
      `public.session_history` row `136103` for the same Codex trace/session:
      provider `openai`, model `gpt-5.4-mini`, `input_tokens=26101`,
      `output_tokens=56`, `input_system_tokens_estimated=2659`,
      `input_tool_advertisement_tokens_estimated=11561`,
      `input_conversation_tokens_estimated=5235`,
      `input_other_tokens_estimated=6646`,
      `input_breakdown_residual_tokens=6646`,
      `prompt_overhead_breakdown_source=request_body_estimate`,
      `prompt_overhead_counted_shape=openai_responses`, and
      `prompt_overhead_route_family=codex_responses`.
    - `AAWM_OBSERVE_SERVICE_NAME=pytest-classifier-scan ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config scripts/local-ci/anthropic_adapter_config.json --target dev --cases native_openai_passthrough_chat,native_openai_passthrough_responses --write-artifact /tmp/native_openai_prompt_overhead_cost_share_4001.json` passed with zero failures/warnings and reported exactly two calls/two groups, proving the shared-session double-count was removed.
    - Broader native live artifact
      `/tmp/native_prompt_overhead_cost_share_4001_after_dedupe.json` produced populated
      prompt-overhead rows for native Claude, OpenAI chat, OpenAI Responses,
      Codex Responses, Gemini generateContent, and Gemini streamGenerateContent:
      total `input_tokens=117426`, explicit system+tool overhead `60646`
      tokens / `0.516461` input share, and plus-other overhead `75997` tokens /
      `0.647191` input share. The suite was not clean because the Gemini
      generateContent case's runtime-log check saw an overlapping `429`
      substring, but the
      Gemini row itself was persisted and included in the report.
  - Follow-up status: D1-060 stays open in `.analysis/todo.md` for adapted-route
    cost-share coverage across Anthropic -> OpenAI/Gemini/NVIDIA/OpenRouter and
    possible exact input-cost storage.

- D1-060 translated-shape coverage and harness overhead assertions
  - Goal: finish the remaining prompt-overhead implementation slice by proving
    translated/billed request shapes are counted correctly and by making the
    live acceptance harness assert the new `session_history` overhead fields.
  - Changed paths:
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
    - `scripts/local-ci/anthropic_adapter_config.json`
    - `tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - Added Anthropic -> OpenAI Responses prompt-overhead coverage using the
      translated billed shape: `instructions`, `tools`, and `input`, with
      `prompt_overhead_counted_shape=openai_responses`.
    - Added Anthropic -> Google Code Assist prompt-overhead coverage using the
      wrapped billed shape: `request.systemInstruction`, `request.tools`, and
      `request.contents`, with
      `prompt_overhead_counted_shape=gemini_generate_content`, while preserving
      existing Google system-prompt policy metadata.
    - Extended the local acceptance harness SQL to select all new overhead
      columns and extended `expected_rows` matching/candidate summaries to
      support row-local prompt-overhead metadata checks.
    - Extended `native_openai_passthrough_responses_codex` acceptance config to
      assert nonzero overhead columns and prompt-overhead metadata in
      `public.session_history`.
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py`
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -k 'prompt_overhead' -q` (`8 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` (`99 passed`)
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -q` (`53 passed`)
    - `./.venv/bin/ruff check --ignore PLR0915,T201 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/local-ci/run_anthropic_adapter_acceptance.py tests/local_ci/test_anthropic_adapter_acceptance_hardening.py` (`All checks passed`; `T201` is ignored for pre-existing harness prints)
    - `./.venv/bin/python -m json.tool scripts/local-ci/anthropic_adapter_config.json >/tmp/anthropic_adapter_config.formatted.json`
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`.
    - `AAWM_OBSERVE_SERVICE_NAME=pytest-classifier-scan ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config scripts/local-ci/anthropic_adapter_config.json --target dev --cases native_openai_passthrough_responses_codex --write-artifact /tmp/native_codex_4001_prompt_overhead_harness_assert.json` passed with zero failures/warnings using the new harness assertions.
    - Dev database verification against exact database `aawm_tristore` found
      `public.session_history` row `135971` for trace
      `e05dde7c-451f-4fa4-b6df-0e716d327104`: provider `openai`, model
      `gpt-5.4-mini`, `input_tokens=26101`,
      `input_system_tokens_estimated=2659`,
      `input_tool_advertisement_tokens_estimated=11561`,
      `input_conversation_tokens_estimated=5235`,
      `input_other_tokens_estimated=6646`,
      `input_breakdown_residual_tokens=6646`,
      `prompt_overhead_breakdown_source=request_body_estimate`,
      `prompt_overhead_counted_shape=openai_responses`,
      `prompt_overhead_classifier_version=deterministic-v1`,
      `prompt_overhead_route_family=codex_responses`.
  - Remaining follow-up kept open in `.analysis/todo.md`: build the broader
    prompt-overhead report/live validation across Claude, Codex, Gemini, and
    adapted route families so cost share can be compared by client and route.

- D1-060 session-history prompt-overhead first live slice
  - Goal: add estimated prompt-overhead token fields to `public.session_history`
    so paid input usage can be split into system/provider-equivalent prompt,
    tool advertisement, conversation, residual/other, and deterministic system
    buckets.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` (`97 passed`)
    - `./.venv/bin/ruff check --ignore PLR0915 litellm/integrations/aawm_agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py` (`All checks passed`)
    - Restarted only dev container `litellm-dev` on `:4001`; verified running
      process started at `2026-04-30T18:42:32Z`, after the mounted
      `aawm_agent_identity.py` mtime `2026-04-30T18:35:13Z`, with matching
      SHA-256 `94d6530a637c4b62fd3cb1a482c7d510d3bd321d3849b3be9bfd09a75198a7ef`.
    - `curl -sS http://127.0.0.1:4001/health/liveliness` returned
      `"I'm alive!"`.
    - `AAWM_OBSERVE_SERVICE_NAME=pytest-classifier-scan ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config scripts/local-ci/anthropic_adapter_config.json --target dev --cases native_openai_passthrough_responses_codex --write-artifact /tmp/native_codex_4001_prompt_overhead.json` passed with zero failures/warnings.
    - Langfuse trace `3127927b-fed6-4f08-91df-a30b82951cbc` recorded
      `environment=dev`, `userId=pytest-classifier`, session
      `019ddfb6-084c-7862-b7a1-4bbb00186742`, and Codex patch tags
      `codex-tool-description-patch` /
      `codex-tool-description-patch:spawn-agent-fanout-policy`.
    - Dev database verification against exact database `aawm_tristore` found
      `public.session_history` row `135861` for trace
      `3127927b-fed6-4f08-91df-a30b82951cbc`: provider `openai`, model
      `gpt-5.4-mini`, `input_tokens=26101`,
      `input_system_tokens_estimated=2659`,
      `input_tool_advertisement_tokens_estimated=11561`,
      `input_conversation_tokens_estimated=5235`,
      `input_other_tokens_estimated=6646`,
      `input_breakdown_residual_tokens=6646`,
      `system_behavior_tokens_estimated=1752`,
      `system_instructional_tokens_estimated=793`,
      `system_unclassified_tokens_estimated=114`,
      `prompt_overhead_breakdown_source=request_body_estimate`,
      `prompt_overhead_counted_shape=openai_responses`,
      `prompt_overhead_classifier_version=deterministic-v1`,
      `prompt_overhead_route_family=codex_responses`.
  - Follow-up status: Anthropic -> OpenAI Responses, Anthropic -> Google
    billed-shape coverage, and harness overhead assertions were completed in
    the later D1-060 entry above. Broader cross-client cost-share reporting
    remains open in `.analysis/todo.md`.

- Native Codex `litellm-dev` profile default attribution fix
  - Goal: fix the plain `codex --profile litellm-dev` path where the profile
    correctly routed to `:4001` but still stamped `langfuse_trace_user_id=codex`,
    causing Langfuse to record `userId=codex` for a pytest-classifier run.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Evidence:
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -k 'codex_repository_over_generic_user_header or explicit_codex_user_header or child_dispatch_trace_metadata or stale_orchestrator_langfuse_trace_header' -q` (`4 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'codex_spawn_agent_tool_description_patch or base_openai_pass_through_handler_sets_trace_environment_and_session or generic_responses_does_not_patch_spawn_agent_tool or openai_passthrough_route_sets_repository_trace_environment_and_session' -q` (`4 passed`)
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -k 'pytest_classifier_harness_user_id or target_profile_codex_cli_uses_pytest_classifier_harness_user_id or native_codex_case_hard_gates_spawn_agent_tool_description_patch' -q` (`2 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` (`91 passed`)
  - Live artifact: restarted only `litellm-dev` on `:4001` and ran
    `codex --profile litellm-dev exec -m gpt-5.4-mini --json ...` from
    `/home/zepfu/projects/pytest-classifier`. Langfuse trace
    `6d1775c7-2a7b-49a2-9aa7-5ee5ff1436dd`, session
    `019ddec6-9680-76d3-af3d-0d61673f94c4`, recorded `environment=dev`,
    `userId=pytest-classifier`, `repository=pytest-classifier`, and patch tags
    `codex-tool-description-patch` /
    `codex-tool-description-patch:spawn-agent-fanout-policy`. Logged request
    text contains `Use subagents to parallelize independent work` and
    `latest frontier model`; both restrictive variants of
    `Only use spawn_agent if and only if...` are absent.

- Native Codex `spawn_agent` policy rewrite live proof and pytest-classifier
  attribution fix
  - Dead end / reopened: the live proof was valid for the explicit harness
    override path, but a later plain `codex --profile litellm-dev` retest showed
    ambient profile header `langfuse_trace_user_id=codex` still won over the
    repository-derived identity. The follow-up item above fixes and verifies the
    alias-shaped path.
  - Goal: prove the Codex-native Responses passthrough rewrites restrictive
    `spawn_agent` tool text in the real Codex CLI request and records the
    pytest-classifier caller as the Langfuse user.
  - Changed paths:
    - `scripts/local-ci/run_acceptance.py`
    - `scripts/local-ci/run_anthropic_adapter_acceptance.py`
    - `litellm/proxy/pass_through_endpoints/success_handler.py`
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/base_passthrough_logging_handler.py`
    - `litellm/integrations/langfuse/langfuse.py`
    - `docker-compose.dev.yml`
    - focused tests and task docs
  - Evidence:
    - `./.venv/bin/python -m pytest tests/local_ci/test_anthropic_adapter_acceptance_hardening.py -k 'pytest_classifier_harness_user_id or native_codex_case_hard_gates_spawn_agent_tool_description_patch' -q` (`2 passed`)
    - `./.venv/bin/python -m pytest tests/pass_through_unit_tests/test_pass_through_unit_tests.py -k 'standard_end_user_header or empty_body' -q` (`2 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_pass_through_endpoints.py -k 'success_handler_applies_logging_hooks or proxy_server_request' -q` (`2 passed`)
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -k 'codex_spawn_agent_tool_description_patch or base_openai_pass_through_handler_sets_trace_environment_and_session or generic_responses_does_not_patch_spawn_agent_tool or openai_passthrough_route_sets_repository_trace_environment_and_session' -q` (`4 passed`)
    - `AAWM_OBSERVE_SERVICE_NAME=pytest-classifier-scan ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py --config scripts/local-ci/anthropic_adapter_config.json --target dev --cases native_openai_passthrough_responses_codex --write-artifact /tmp/native_codex_spawn_agent_tool_description_patch.json` (`passed=True`, zero failures/warnings)
  - Live artifact: trace `092a1ac4-1cd5-4859-949b-9898b7ba3b1c`, session
    `019dde80-69f5-7800-ba0c-039fa72c4011`, Langfuse user
    `pytest-classifier`, patch tags present, generic fanout policy text
    present, restrictive `Only use spawn_agent if and only if...` text absent.


- D1-130 session_history structured-output attempt/failure telemetry (2026-05-22)
  - Goal: make structured-output error-rate reporting first-class in
    `public.session_history` instead of leaving it as a placeholder derived
    from generic provider errors.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes: added idempotent `session_history` DDL/ALTER columns
    for `structured_output_attempted`, `structured_output_failed`,
    `structured_output_mode`, `structured_output_schema_hash`, and
    `structured_output_failure_reason`; stamps success, Langfuse backfill, and
    structured-output failure callback records; keeps non-structured provider
    errors on the existing observation-only path.
  - Evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py` passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` passed (`188 passed`, one pre-existing pytest config warning for `asyncio_default_fixture_loop_scope`).
    - `git diff --check` passed.
  - Live status: code/schema migration is idempotent but was not applied to a
    live database in this session; the columns will be added when the callback
    next runs `_ensure_session_history_schema` in the target environment.


- D1-131 Anthropic endpoint auto-agent alias (2026-05-22)
  - Goal: add an Anthropic-only model alias that uses the existing Anthropic
    Messages endpoint surface while exhausting Codex Spark, Gemini Code Assist,
    OpenRouter DeepSeek free, then native Claude Haiku fallback capacity.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
  - Implementation notes: added `/anthropic/v1/messages` alias
    `aawm-anthropic-agent-auto` with process-local cooldown/session affinity,
    candidate order `gpt-5.3-codex-spark`, Gemini `3.1 flash-lite` / `3 flash`
    / `3.1 pro`, `deepseek/deepseek-v4-flash:free`, then
    `claude-haiku-4-5-20251001`; kept resolution inside `anthropic_proxy_route`
    rather than model-list/global alias routing; native Haiku fallback reuses
    Anthropic passthrough; Gemini alias probes disable hidden retry budget so a
    cooled lane advances to the next candidate; Anthropic `tool_use` /
    `tool_result` blocks now count as stateful continuation markers for
    in-flight provider stickiness.
  - Evidence:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py` passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'anthropic_auto_agent_alias or anthropic_proxy_route_uses_auto_alias or codex_auto_agent_alias'` passed (`21 passed, 315 deselected`; only pre-existing pytest/deprecation warnings).
    - `git diff --check` passed.
  - Live status: source and focused unit coverage are complete locally; no dev
    or prod proxy restart was performed in this session.

- D1-137 session_history memory repository identity repair (2026-05-23)
  - Goal: resolve malformed public.session_history.repository values ending
    in (memory) to real repo names while preserving the memory designation.
  - Changed data:
    - Exact DB: aawm_tristore.public.session_history
    - Rows updated: 79
    - Repair marker:
      metadata.repository_identity_repair_run_id=memory_repository_identity_repair_2026_05_23
  - Implementation notes: repaired only row-scoped current bad labels
    (memories (memory), ho (memory), zepfu (memory), 20 (memory),
    li (memory), lite (memory), and truncated rollout-* (memory) fragments).
    Updated repository, matching tenant_id, metadata.repository, matching
    metadata.tenant_id, metadata.trace_user_id, and base
    metadata.source_repository; preserved previous values in
    metadata.repository_identity_previous_* fields.
  - Evidence:
    - Pre-repair malformed memory rows against real /home/zepfu/projects/*
      repo names: 79.
    - Applied update returned UPDATE 79.
    - Post-repair malformed memory rows against real repo names: 0.
    - Total (memory) repository row count stayed 1279.
    - Repaired-row repository distribution: litellm (memory)=31,
      aawm-tap (memory)=30, dashboard-shell (memory)=10,
      aawm-observe (memory)=6, aawm-lsp-proxy (memory)=1,
      lsp-guard (memory)=1.
    - Repair-source distribution: nearby_codex_session_meta_cwd=45,
      codex_session_meta_cwd=15, neighboring_memory_batch_next_repo=13,
      neighboring_memory_batch_repo_fragment=2, repo_prefix_fragment=2,
      rollout_summary_header_cwd=2.
    - Consistency check for repaired rows where tenant_id,
      metadata.repository, metadata.tenant_id, or metadata.source_repository
      did not match the repaired repository: 0.

- D1-134 auto-agent passthrough session_history freshness repair (2026-05-23)
  - Goal: restore and prove `public.session_history` coverage for current
    passthrough traffic that was present in Langfuse/rate-limit telemetry but
    missing from session history after OpenAI/Codex lost explicit session ids.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `scripts/backfill_session_history.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
  - Implementation notes: passthrough session-history records now fall back to
    deterministic synthetic session ids from trace/call ids when no client
    session id is available, marking `metadata.synthetic_session_id`; Langfuse
    backfill uses the same trace-id fallback for passthrough observations;
    Google Code Assist wrapped requests carry the derived adapter session id in
    `litellm_metadata`; OpenAI Responses stream logging falls back for unknown
    custom-tool response items even when no reasoning summary exists; the
    Langfuse ClickHouse backfill now normalizes UTC-aware timestamps for
    ClickHouse literals and filter comparisons; the misplaced Anthropic route
    decorator is back on `anthropic_proxy_route`.
  - Backfill evidence:
    - Initial freshness baseline: `openai` max `2026-05-21 04:15:30.980003+00`,
      `gemini` max `2026-05-22 12:14:06.535486+00`, `anthropic` max
      `2026-05-22 12:11:18.978877+00`.
    - Successful backfill command:
      `AAWM_DB_HOST=127.0.0.1 AAWM_DB_PORT=5434 AAWM_DB_NAME=aawm_tristore AAWM_DB_USER=aawm AAWM_DB_PASSWORD=aawm_dev ./.venv/bin/python scripts/backfill_session_history.py --source-mode langfuse_clickhouse --from-start-time 2026-05-22T12:11:19Z --apply --batch-size 50`.
    - Backfill result: scanned `20166`, reconstructable `18848`, inserted
      `8896`, updated `9952`, skipped `1318`; session id sources included
      `trace.id.synthetic=8804` and `trace.sessionId=10044`.
    - Final freshness snapshot in exact DB `aawm_tristore`: `openai` max
      `2026-05-23 04:12:42.060927+00` with `8057` rows after the original
      cutoff and `7608` synthetic rows; `gemini` max
      `2026-05-23 04:11:19.332684+00` with `1252` rows after cutoff and
      `1234` synthetic rows; `anthropic` remained at
      `2026-05-22 12:11:18.978877+00`, and `rate_limit_observations` showed no
      Anthropic header/quota observations after that point.
  - Dev validation:
    - Restarted `litellm-dev`; `/health/readiness` returned `status=healthy`.
    - Container source inspection showed patched callback loaded and the OpenAI
      fallback no longer contains the old reasoning-summary guard.
    - Route table verification: `/anthropic/{endpoint:path}` is bound to
      `anthropic_proxy_route`.
    - Direct in-container callback persist wrote row `717876` with
      `litellm_call_id=manual-persist-debug`, `provider=openai`,
      `session_id=manual-persist-debug`, and
      `session_id_source=kwargs.litellm_call_id`.
    - Live OpenAI passthrough smoke wrote row `717882` at
      `2026-05-23 04:07:58.135986+00`, `provider=openai`,
      `model=gpt-5.4-mini`, `trace_name=codex.dev-session-history-smoke`,
      `session_id_source=standard_logging_object.trace_id`, and
      `synthetic_session_id=true`.
    - Live Gemini adapter smoke reached the intended route after the decorator
      fix and wrote row `717908` at `2026-05-23 04:11:19.332684+00`,
      `provider=gemini`, `model=gemini-3.1-flash-lite-preview`,
      `trace_name=orchestrator`; the client response was still the separate
      Google Code Assist streaming response-builder `502` tracked in D1-138.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py scripts/backfill_session_history.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py` passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "synthetic_passthrough_session_id or langfuse_passthrough_uses_synthetic_session_id or codex_google_code_assist_metadata or structured_output_request"` passed (`4 passed, 188 deselected`).
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k "anthropic_pass_through_route_binds_public_proxy_handler or propagates_adapter_session_metadata or unknown_items_falls_back_without_reasoning"` passed (`3 passed, 337 deselected`).

- D1-139 callback runtime DDL removal (2026-05-23)
  - Goal: stop the AAWM LiteLLM callback from applying database structure
    changes at request/write time, especially recreating or dropping
    `public.rate_limit_intervals` from hardcoded Python SQL.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes: removed the hardcoded
    `rate_limit_intervals` materialized-view create SQL, stale-view drop SQL,
    and related materialized-view index statements from the callback module and
    wheel-build mirror. `_ensure_session_history_schema` now preserves its
    historical call surface but only marks the schema gate ready; it no longer
    calls `conn.execute` for `CREATE TABLE`, `ALTER TABLE`, `CREATE INDEX`,
    `CREATE MATERIALIZED VIEW`, or `DROP` operations. Schema ownership is now
    explicitly migration/operations-owned, while the callback remains a data
    writer into pre-existing tables.
  - Evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py` passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k "ensure_session_history_schema or rate_limit_intervals_mview or openrouter_free or scalar_tool_names or scalar_file_paths"` passed (`6 passed, 189 deselected`; only the pre-existing pytest config warning).
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` passed (`195 passed`; only the pre-existing pytest config warning).
    - Static scans found no `_ensure_session_history_schema` schema-mutating
      `conn.execute` calls and no `rate_limit_intervals` materialized-view
      create/drop strings in the source callback or wheel-build mirror.
    - `git diff --check -- litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py` passed.
  - Live status: no database DDL was run for this fix.

- D1-141 rate_limit_intervals Bengalfox/OpenRouter mview apply (2026-05-23)
  - Goal: apply the operator-supplied `public.rate_limit_intervals`
    materialized-view definition to exact database `aawm_tristore`, including
    OpenRouter request rows plus Bengalfox primary/secondary special quota
    mappings.
  - Changed paths:
    - `scripts/apply_rate_limit_intervals_mview_2026_05_23.sql`
  - Changed database objects:
    - `aawm_tristore.public.rate_limit_intervals`
    - `rate_limit_intervals_requests_idx`
    - `rate_limit_intervals_type_provider_from_idx`
    - `rate_limit_intervals_unique_idx`
  - Implementation notes: this was an explicit operator DB operation from a
    dedicated `scripts/` SQL artifact, not callback runtime DDL. The script
    recreated only `public.rate_limit_intervals`, created the three existing
    indexes, and analyzed the mview. Existing pg_cron jobs
    `aawm_rate_limit_intervals_refresh` and
    `aawm_rate_limit_intervals_analyze` remained active.
  - Evidence:
    - Apply command:
      `psql -v ON_ERROR_STOP=1 -f scripts/apply_rate_limit_intervals_mview_2026_05_23.sql postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`.
    - Apply output: `BEGIN`, `DROP MATERIALIZED VIEW`, `SELECT 10419`,
      three `CREATE INDEX`, `ANALYZE`, `COMMIT`.
    - Definition checks returned true for
      `codex_bengalfox:primary`, `codex_bengalfox:secondary`,
      `short_special`, `openrouter`, and `SELECT DISTINCT`.
    - Current row summary:
      `count=10419`, Bengalfox row count `2326`,
      max `fromdate=2026-05-23 05:25:14.517526+00`.
    - Current Codex quota rows:
      `codex:primary|short|1957`, `codex:secondary|weekly|443`,
      `codex_bengalfox:primary|short_special|1721`,
      `codex_bengalfox:secondary|weekly_special|605`.
    - `REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals;
      ANALYZE public.rate_limit_intervals;` succeeded and row count remained
      `10419`.
    - Current owner/size: `aawm|3312 kB`.
    - `git diff --check -- scripts/apply_rate_limit_intervals_mview_2026_05_23.sql` passed.


- D1-140 session_history repository fallback for request-header tenants (2026-05-23)
  - Goal: stop otherwise valid session-history rows from landing with blank
    `repository` when request headers identify a repo-shaped tenant but no
    explicit repository field is present.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
  - Implementation notes: session-history normalization now derives
    `repository` from request-header tenant identity only when the tenant value
    is a clean repository-shaped identifier. Harness/validation tenant labels
    are mapped to `litellm`; `*-dev` and generic tenant labels remain excluded.
    Derived values are marked with
    `metadata.repository_source=tenant_id.request_headers`.
  - Evidence:
    - Focused baseline showed recent missing repositories concentrated in
      `openrouter/python-httpx`, `local_embed/python-httpx`, and Langfuse
      backfilled Codex rows, while live post-repair Codex rows already carried
      repository values.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'repository'`
      passed (`35 passed, 161 deselected`; only the pre-existing pytest config
      warning).
    - `litellm-dev` was restarted and `/health/readiness` returned
      `status=healthy`.
    - Post-restart exact DB `aawm_tristore.public.session_history` rows
      `718351` through `718360` showed non-empty repositories such as
      `aawm-tap` and `dashboard-shell`; post-restart missing-repository count
      was `0` for rows after `2026-05-23 05:14:00+00`.
  - Backfill evidence:
    - Approved targeted update changed `150058` rows in exact DB
      `aawm_tristore.public.session_history`, setting `repository` and
      `metadata.repository_source=tenant_id.request_headers`.
    - Post-backfill target count was `0`. Total blank-repository rows dropped
      from `339060` to `189002`; request-header-sourced blank rows dropped
      from `160469` to `10411`. Remaining request-header blanks were excluded
      cases: missing tenant on LangfuseTraces rows, `aawm-tap-dev`, or harness
      rows with no tenant value.

- D1-166/D1-167/D1-168 agent quality score additions (2026-05-30)
  - Goal: add deterministic scoring for ignored-path force tracking,
    baseline/pre-existing deflection after quality gates, and unsolicited
    sleep/wellness interruptions; persist the fields to session history and move
    the completed callback path into `litellm-dev`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_quality_rules.py`
    - `litellm/integrations/aawm_agent_quality_rules.json`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py`
    - `.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.json`
    - `.wheel-build/pyproject.toml`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `docker-compose.dev.yml`
    - `/home/zepfu/projects/dashboard-shell/.analysis/handoff-session-history-agent-quality-d1-166-168.md`
  - Implementation notes:
    - Added a bounded local rule catalog with TTL hot reload, schema validation,
      last-valid fallback, phrase/count/byte caps, and `AAWM_AGENT_QUALITY_RULES_PATH`.
    - Added D1-166 `ignored_path_tracking_policy_score` and
      `ignored_path_tracking_violation_count`, including confirmed offline Git
      evidence modes and callback/runtime command-only inferred evidence without
      reading `.gitignore`, running Git, or inspecting local repo state in the
      message flow.
    - Added D1-167 `baseline_deflection_attempted_score`,
      `baseline_deflection_incident_score`, quality-gate/probe/fix counters,
      token/time counters, and reason evidence.
    - Added D1-168 `sleep_wellness_interruption_attempted_score`,
      `sleep_wellness_interruption_incident_score`, interruption/pushback/repeat
      counters, token/time counters, suppression handling for user-requested
      wellness/sleep advice, and reason evidence.
    - Extended Langfuse score emission, offline session-history upsert SQL,
      callback metadata normalization, DB payload ordering, and the standalone
      callback wheel mirror.
    - Added dev bind mounts for the new rule module and catalog so
      `litellm-dev` loads the same code path after recreation.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py litellm/integrations/aawm_agent_quality_rules.py .wheel-build/aawm_litellm_callbacks/agent_identity.py .wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py scripts/score_agent_trace_quality.py` passed.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_score_agent_trace_quality.py -q` passed (`41 passed`, one pre-existing pytest config warning).
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q` passed (`237 passed`, one pre-existing pytest config warning).
    - Added regression coverage proving the "Fix-what-blocks-you" persona
      guidance text containing "pre-existing", "baseline", and "not in scope"
      does not trigger D1-167 baseline-deflection scoring when the assistant
      fixes the quality-gate failure.
    - `git diff --check` passed.
    - Targeted `ruff check` on the touched files still reports pre-existing
      `PLR0915 too-many-statements` findings in long callback/scorer/test
      functions; no formatting/compile/test failure was introduced.
  - Database evidence:
    - Applied additive schema to exact DB
      `aawm_tristore.public.session_history` with `ALTER TABLE ... ADD COLUMN IF
      NOT EXISTS` for the 19 new D1-166/D1-167/D1-168 columns.
    - Verification query against `aawm_tristore.public.session_history` returned
      all `19` expected new columns with expected `integer` or
      `double precision` types.
  - Dashboard handoff:
    - Created
      `/home/zepfu/projects/dashboard-shell/.analysis/handoff-session-history-agent-quality-d1-166-168.md`
      with changed columns, score directionality, evidence modes,
      aggregation guidance, and requested dashboard surfaces.
    - Created
      `/home/zepfu/projects/aawm-tap-dashboard/.analysis/2026-05-30-litellm-agent-quality-score-handoff.md`
      with the same TAP-facing contract details before historical score
      backfill, so TAP dashboard work can surface the new score families.
    - Follow-up verification found the TAP-dashboard copy missing while the
      dashboard-shell copy existed; the TAP-dashboard handoff was added at the
      path above and verified by reading the ignored `.analysis` file directly.
  - Backfill evidence:
    - Added scorer operational flags used for the backfill:
      `--summary-only`, `--missing-agent-quality-backfill-only`, and
      `--minio-read-workers`; ClickHouse and MinIO lookups are now chunked or
      bounded so large historical runs do not emit massive per-row JSON or
      exceed ClickHouse query-size limits.
    - Ran exact date-windowed backfill against exact DB
      `aawm_tristore.public.session_history` using
      `scripts/score_agent_trace_quality.py --update-session-history-scores
      --include-passing --missing-agent-quality-backfill-only --summary-only`
      with bounded MinIO workers. The run covered historical rows from
      `2026-02-12` through `2026-05-30`; May 22 and May 26 were resumed with
      lower concurrency after one stale MinIO object key and one transient
      Postgres client-pressure failure.
    - Final verification query against exact DB
      `aawm_tristore.public.session_history` returned:
      `total=860005`, `incident_null=5`, `ignored_null=856003`,
      `ignored_evaluated=4002`, `ignored_violations=0`,
      `baseline_attempts=2130`, `baseline_incidents=1356`,
      `sleep_attempts=2`, `sleep_incidents=2`.
    - The remaining `incident_null=5` rows were all `2026-05-30` live-ingest
      rows that appeared during final verification; two final current-day
      sweeps updated `43` then `18` rows immediately before that check.
    - Follow-up backfill requested on `2026-05-30` first verified the database
      had moved to `total=861784`, `incident_null=1770`,
      `ignored_evaluated=4016`, `baseline_attempts=2130`,
      `baseline_incidents=1356`, `sleep_attempts=2`, and
      `sleep_incidents=2` because new live rows had arrived since the initial
      historical sweep.
    - A bounded missing-only run against exact DB
      `aawm_tristore.public.session_history` updated `1808` session-history
      score rows:
      `./.venv/bin/python scripts/score_agent_trace_quality.py
      --update-session-history-scores --include-passing
      --missing-agent-quality-backfill-only --summary-only --limit 2500
      --minio-read-workers 4`.
    - A final current-hour sweep updated another `244` rows:
      `./.venv/bin/python scripts/score_agent_trace_quality.py
      --update-session-history-scores --include-passing
      --missing-agent-quality-backfill-only --summary-only
      --from-created-at 2026-05-30T20:00:00Z --limit 1000
      --minio-read-workers 4`.
    - Final follow-up verification query against exact DB
      `aawm_tristore.public.session_history` returned:
      `total=862088`, `incident_null=17`, `ignored_null=856792`,
      `ignored_evaluated=5296`, `ignored_violations=0`,
      `baseline_attempts=2500`, `baseline_incidents=1630`,
      `sleep_attempts=2`, `sleep_incidents=2`. The remaining null rows were
      current-hour live-ingest tail, not historical backfill coverage gaps.
  - Dev validation:
    - Recreated `litellm-dev` with
      `docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-dev`.
    - `curl -sS http://127.0.0.1:4001/health/readiness` returned
      `status=healthy`, version `1.82.3+aawm.60`, and success callbacks
      including `AawmAgentIdentity`.
    - In-container import check loaded
      `/app/litellm/integrations/aawm_agent_identity.py` and
      `/app/litellm/integrations/aawm_agent_quality_rules.py`; runtime scorer
      was present and catalog version was `2026-05-30.v1`.
    - In-container synthetic callback smoke returned
      `ignored_path_tracking_policy_score=0.0`,
      `ignored_path_tracking_violation_count=1`, and evidence mode
      `inferred_common_ignored_path` for visible command
      `git add -f .analysis/todo.md`.

- Root handoff archive and failed reconciliation subagent disposition
  (2026-05-30)
  - Goal: process new/root handoff documents in `.analysis` before continuing
    active TODO implementation work.
  - Changed paths:
    - `.analysis/completed-handoff/codex-litellm-session-handoff-2026-04-10.md`
    - `.analysis/completed-handoff/gemini-litellm-session-handoff-2026-04-10.md`
    - `.analysis/investigation/investigate-codex-019e7a70-06b0-7e43-8f60-ef7eb4eb0b40.md`
    - `.analysis/investigations.md`
  - Evidence:
    - Root handoff scan found only the Codex and Gemini 2026-04-10 session
      handoffs. The Codex handoff is already referenced by the completed
      documentation audit, and the Gemini handoff states the transport/auth
      objective is achieved with only optional observability shaping remaining.
    - Both handoffs were moved out of root `.analysis` into
      `.analysis/completed-handoff/`.
    - The `aawm-codex-agent-auto` read-only handoff reconciliation agent
      `019e7a70-06b0-7e43-8f60-ef7eb4eb0b40` returned an incomplete response
      instead of the requested discovery inventory/classification. The failure
      was recorded under `.analysis/investigation/` and dispositioned in
      `.analysis/investigations.md`.

- D1-163/D1-164 discovery inventory and null terminal completion scoring
  (2026-05-30)
  - Goal: add deterministic discovery-inventory coverage scoring for
    communicated broad discovery contracts, and persist/score null or empty
    Codex subagent terminal completions from transcript-derived rows.
  - Changed paths:
    - `litellm/integrations/aawm_agent_quality_rules.py`
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `/home/zepfu/projects/dashboard-shell/.analysis/handoff-session-history-discovery-terminal-d1-163-164.md`
    - `/home/zepfu/projects/aawm-tap-dashboard/.analysis/2026-05-30-litellm-discovery-terminal-score-handoff.md`
    - `.analysis/investigation/investigate-codex-019e7a7b-1148-7e03-a86a-231af4372b68.md`
    - `.analysis/investigation/investigate-codex-019e7a6f-cf1d-7603-9d3f-3ce70146e025.md`
    - `.analysis/investigations.md`
    - `.analysis/todo.md`
  - Implementation notes:
    - Added `discovery_inventory_coverage_score` and
      `discovery_inventory_missing_count`. The score is only evaluated when the
      delegated prompt communicates the discovery-inventory contract; prompts
      that explicitly say no broad discovery inventory is required stay null.
    - Added `terminal_completion_score` for Codex transcript-derived rows, plus
      transcript metadata for terminal state, function-call count,
      unsupported-tool count/names, final-message presence, and non-empty
      assistant progress count.
    - Null/empty/missing terminal completion now fails
      `terminal_completion_score`, `response_meaningfulness_score`, and
      `output_contract_compliance_score`, while `task_progress_score` remains
      able to show useful partial progress.
    - Unsupported transcript tool loops such as `run_command`, `list_dir`, and
      `list_files` now fail `tool_use_validity_score` and
      `tool_error_recovery_score`.
    - Synthetic transcript repairs record `langfuse_score_emit_status` so
      dashboard/reporting consumers can distinguish rows where Langfuse scores
      were not emitted because the source was a local transcript repair.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py litellm/integrations/aawm_agent_quality_rules.py .wheel-build/aawm_litellm_callbacks/agent_identity.py .wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py scripts/score_agent_trace_quality.py` passed.
    - `./.venv/bin/python -m pytest tests/test_scripts/test_score_agent_trace_quality.py tests/test_litellm/integrations/test_aawm_agent_identity.py -q` passed (`284 passed`, one pre-existing pytest config warning).
  - Database evidence:
    - Applied additive schema to exact DB
      `aawm_tristore.public.session_history` after canceling active read-only
      report queries that were holding `AccessShareLock`.
    - Verification query against exact DB
      `aawm_tristore.public.session_history` returned:
      `discovery_inventory_coverage_score:double precision`,
      `discovery_inventory_missing_count:integer`, and
      `terminal_completion_score:double precision`.
    - Transcript-derived upsert command updated `8` affected D1-164 rows with
      `--upsert-codex-transcript-session-history --summary-only --include-passing`.
    - Verification query for sessions
      `019e6918-77e2-7770-8e73-218348a12fdb`,
      `019e691b-840f-74f0-97fb-1dada3edd76a`,
      `019e691b-896a-7963-abe5-fd0020db1189`,
      `019e691c-8690-75c0-a031-761f123c51a9`,
      `019e691c-88d1-7000-9351-497871233700`,
      `019e691c-8aed-7b52-93fc-d0164a46d17a`,
      `019e6928-f793-7c60-893c-38cb16d4458e`, and
      `019e6951-6254-7280-bf62-088861f1d512` showed
      `codex_transcript_terminal_state=null_final_message`,
      `terminal_completion_score=0`, `response_meaningfulness_score=0`,
      `output_contract_compliance_score=0`, and
      `langfuse_score_emit_status=not_emitted_synthetic_transcript` for every
      row. The first three rows retained partial progress
      (`task_progress_score=1` with `123/4/2` tool calls and `16/2/1`
      non-empty assistant progress counts), the three immediate null sessions
      retained `task_progress_score=0`, and
      `019e6928-f793-7c60-893c-38cb16d4458e` retained
      `invalid_tool_call_count=14`, `tool_use_validity_score=0`, and
      `tool_error_recovery_score=0`.
  - Dashboard handoff:
    - Created dashboard-shell and aawm-tap-dashboard handoffs naming the new
      columns, semantics, reason/metadata keys, aggregation guidance, and
      suggested dashboard drilldowns.
  - Subagent disposition:
    - D1-163 worker `019e7a7b-1148-7e03-a86a-231af4372b68` completed with
      `{"completed": null}` and was dispositioned under D1-164.
    - D1-164 explorer `019e7a6f-cf1d-7603-9d3f-3ce70146e025` failed with
      `exceeded retry limit, last status: 429 Too Many Requests`; the main
      thread completed implementation locally.

- D1-169 Claude Code, Codex, and Gemini CLI compact event measure in
  session_history (2026-05-31)
  - Goal: add a reliable `session_history` measure for Claude Code, Codex, and
    Gemini CLI conversation compacts so dashboards/reports can count how many
    compacts happened per session or thread.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `/home/zepfu/projects/dashboard-shell/.analysis/handoff-session-history-compact-summary-d1-169.md`
  - Implementation notes:
    - Added deterministic compact classification for Claude Code compact
      summary prompts, Codex `CONTEXT CHECKPOINT COMPACTION` requests, Gemini
      CLI `compress-*` state snapshot requests, and non-counting context rows
      for Codex resumed handoff and Gemini `compress-*-verify`.
    - Added `is_compact_summary`, `compact_summary_source`,
      `compact_summary_id`, and `compact_summary_role` to the live and
      Langfuse backfill session-history record builders.
    - Extended `public.session_history` payload ordering and upsert merge policy
      so compact fields persist and existing true compact signals are retained.
    - Mirrored the callback logic in `.wheel-build/aawm_litellm_callbacks/`.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -k d1_169 -p no:rerunfailures -q`
      passed (`11 passed`, one pre-existing pytest config warning).
    - Earlier full-file regression pass:
      `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -p no:rerunfailures -q`
      passed (`249 passed`, one pre-existing pytest config warning).
  - Database evidence:
    - Applied additive schema to exact DB
      `aawm_tristore.public.session_history` on
      `postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`:
      `is_compact_summary:boolean:NO:false`,
      `compact_summary_source:text:YES`,
      `compact_summary_id:text:YES`, and
      `compact_summary_role:text:YES`.
    - Built and verified partial index
      `session_history_compact_summary_idx` on
      `(session_id, compact_summary_id, created_at DESC) WHERE
      is_compact_summary`; final `pg_index` check returned
      `indisready=true`, `indisvalid=true`, and `indislive=true`.
  - Backfill evidence:
    - First verified representative compact traces with enhanced
      `scripts/backfill_session_history.py --source-mode langfuse_clickhouse
      --apply` against the LiteLLM dev instance, ClickHouse
      `http://127.0.0.1:8123`, and exact target DB
      `aawm_tristore.public.session_history`.
    - Representative traces applied cleanly before the full sweep:
      Claude compact trace `d7042459-5dbb-4aab-bb87-11ea6c39299f`,
      Codex compact trace `68f25c55-3fd6-4b7b-ae45-f95a3493999f`,
      Codex resumed context trace `9e8be83e-9762-4544-9d33-cd44d7339dde`,
      Gemini compact trace `960c7fa4-473e-4e4a-8aba-6835aa9f5a2f`, and
      Gemini verify trace `cb07bb75-04bd-4da5-89a3-bc40990ea032`.
    - The first unfiltered full-history apply run used `--batch-size 100`
      after a 1,000-row batch timed out; it returned `scanned_rows=5176`,
      `reconstructable_rows=5015`, `inserted_rows=5010`, `updated_rows=5`,
      `existing_rows=5`, and `skipped_rows=161`.
    - Because live Langfuse ingestion continued while the first full sweep ran,
      a second stable-cutoff full sweep was run with
      `--to-start-time 2026-05-31T17:58:28Z --batch-size 100`. It returned
      `scanned_rows=5597`, `reconstructable_rows=5282`,
      `inserted_rows=267`, `updated_rows=5015`, `existing_rows=5015`, and
      `skipped_rows=315`.
    - ClickHouse cutoff verification for
      `default.observations WHERE type='GENERATION' AND is_deleted=0 AND
      start_time <= '2026-05-31 17:58:28.000'` returned
      `generation_rows=5596`, `min_start=2026-05-31 00:02:49.717`, and
      `max_start=2026-05-31 17:58:27.683`.
    - Final compact count query against exact DB returned:
      `claude_compacts=1`, `codex_compacts=431`,
      `codex_resume_context=1592`, `gemini_compacts=1`, and
      `gemini_verify_context=1`.
  - Dashboard handoff:
    - Created
      `/home/zepfu/projects/dashboard-shell/.analysis/handoff-session-history-compact-summary-d1-169.md`
      with changed columns, compact source/role semantics, aggregation and
      dedupe guidance, false-positive caveats, exact dev backfill traces, and
      requested dashboard/report surfaces.

- D1-171 session_history cleanup for codex transcript, unknown, litellm, and
  adapter-harness-tenant entries (2026-06-01)
  - Goal: investigate and resolve user-facing `public.session_history` buckets
    that surfaced as `codex-transcript` / `codex transcript`, `unknown`,
    `litellm`, or `adapter-harness-tenant`.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `scripts/score_agent_trace_quality.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `tests/test_scripts/test_score_agent_trace_quality.py`
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e7faf-db94-7352-a286-72a844334fce.md`
    - `.analysis/investigation/investigate-codex-019e7fb0-0837-7803-9c43-23519085a2cd.md`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
  - Implementation notes:
    - `codex_transcript` synthetic score rows now preserve score evidence while
      marking the rows out of user-facing provider/model reporting with
      `session_history_usage_record=false`,
      `session_history_reporting_excluded=true`, and
      `session_history_reporting_excluded_reason=synthetic_codex_transcript`.
    - `model='unknown'` rows are left unresolved and reporting-excluded unless
      request tags provide recoverable `claude-exp:<claude-model>` evidence.
    - Harness tenants such as `adapter-harness-tenant` normalize to the
      repository-derived tenant while preserving
      `aawm_original_tenant_id=adapter-harness-tenant` and
      `aawm_harness_tenant_alias=true`.
    - The callback mirror was kept byte-for-byte in sync with
      `litellm/integrations/aawm_agent_identity.py`.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py scripts/score_agent_trace_quality.py tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_score_agent_trace_quality.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py tests/test_scripts/test_score_agent_trace_quality.py -q`
      passed (`303 passed`, one pre-existing pytest config warning).
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
  - Database repair evidence:
    - Target was exact DB `aawm_tristore.public.session_history` at
      `postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`.
    - Initial full-history inventory returned `total=956743`,
      `provider_litellm=0`, `model_codex_transcript=9`,
      `model_unknown=166`, `model_group_auto=0`,
      `tenant_adapter_harness=1836`, and `openai_deepseek_or_nomic=0`.
    - Chunked repair updated `265` rows: `updated_codex_transcript=9`,
      `updated_harness_tenant=97`, `updated_unknown_total=159`,
      `updated_unknown_recovered=38`, and
      `updated_unknown_unresolved=121`.
  - Database verification evidence:
    - High-range full verification over IDs `>=350001` returned
      `total_rows=663659`, `provider_litellm=0`, `model_group_auto=0`,
      `openai_deepseek_or_nomic=0`, `codex_transcript_total=9`,
      `codex_transcript_unexcluded=0`, `unknown_total=90`,
      `unknown_flagged_unresolved=90`, `unknown_unflagged=0`,
      `unknown_recovered_from_tags=38`,
      `adapter_harness_tenant_remaining=0`, and
      `adapter_harness_alias_preserved=39`.
    - Low-range full verification over IDs `1..350001` returned
      `total_rows=308879`, `provider_litellm=0`, `model_group_auto=0`,
      `openai_deepseek_or_nomic=0`, `codex_transcript_total=0`,
      `codex_transcript_unexcluded=0`, `unknown_total=31`,
      `unknown_flagged_unresolved=31`, `unknown_unflagged=0`,
      `unknown_recovered_from_tags=7`,
      `adapter_harness_tenant_remaining=0`, and
      `adapter_harness_alias_preserved=1797`.
    - Final root intake check
      `find .analysis -maxdepth 1 -type f \( -name 'handoff*.md' -o -name 'request*.md' -o -name 'investigate-*.md' \) -print | sort`
      returned no files.
  - Investigation disposition:
    - Root investigation intake files
      `investigate-codex-019e7faf-db94-7352-a286-72a844334fce.md` and
      `investigate-codex-019e7fb0-0837-7803-9c43-23519085a2cd.md` were
      dispositioned in `.analysis/investigations.md` and moved under
      `.analysis/investigation/`.

- D1-123 xAI/Grok partial-cache-hit miss-cost reporting gap (2026-06-01)
  - Initiated: 2026-06-01 13:39:53 EDT.
  - Completed: 2026-06-01 14:04:26 EDT.
  - Duration: 24 minutes 33 seconds.
  - Goal: define and implement xAI/Grok Build semantics for
    `provider_cache_miss_cost_usd` when the provider reports a cache hit for
    only part of the prompt.
  - Changed paths:
    - `litellm/integrations/aawm_agent_identity.py`
    - `.wheel-build/aawm_litellm_callbacks/agent_identity.py`
    - `tests/test_litellm/integrations/test_aawm_agent_identity.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `suggestion.md`
  - Implementation notes:
    - Preserved `provider_cache_status=hit` for xAI partial hits so existing
      status consumers still see the provider-reported cache hit.
    - Added xAI-only partial-hit miss accounting when
      `0 < cache_read_input_tokens < input_tokens`, setting
      `provider_cache_miss=true`,
      `provider_cache_miss_reason=partial_cache_hit`,
      `provider_cache_miss_token_count=input_tokens-cache_read_input_tokens`,
      and `provider_cache_miss_cost_usd` from the prompt-vs-cache-read delta.
    - Kept generic/Gemini hit behavior unchanged: hit rows remain
      non-misses with null miss token/cost fields.
    - The shared resolver now repairs historical xAI metadata that was already
      stamped `hit` / `miss=false` when token evidence proves a partial hit.
    - The callback mirror was kept byte-for-byte in sync with
      `litellm/integrations/aawm_agent_identity.py`.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py tests/test_litellm/integrations/test_aawm_agent_identity.py scripts/repair_session_history_provider_cache.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'xai_partial_cache_hit or provider_cache'`
      passed (`10 passed`, one pre-existing pytest config warning).
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q`
      passed (`260 passed`, one pre-existing pytest config warning).
    - `cmp -s litellm/integrations/aawm_agent_identity.py .wheel-build/aawm_litellm_callbacks/agent_identity.py`
      returned `0`.
  - Database evidence:
    - Target was exact DB `aawm_tristore.public.session_history` at
      `postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore`.
    - Before repair, read-only inventory returned
      `aawm_tristore|21155|21144|0`, meaning 21,155 xAI rows, 21,144
      xAI partial-hit rows still unmarked, and 0 partial-hit rows marked.
    - The broad repair script was canceled because its tool-activity join was
      too broad for this targeted fix.
    - Targeted SQL repair updated `20145` xAI rows whose partial-hit metadata
      was not already repaired.
    - Final verification returned `aawm_tristore|21155|0|21144|21144`,
      meaning 21,155 xAI rows, 0 remaining unmarked partial-hit rows, 21,144
      partial-hit rows marked in columns, and 21,144 partial-hit rows marked
      in metadata.
  - Subagent evidence:
    - Read-only scout `019e844a-145c-7532-af25-36d5a78381a6` confirmed the
      early `cache_read_input_tokens > 0` hit path was the gap and recommended
      preserving `provider_cache_status=hit` while layering partial miss
      fields on top. No files were modified by the scout.

- D1-107 native Grok Build pass-through final live validation (2026-06-01)
  - Initiated: 2026-06-01 14:07:21 EDT.
  - Completed: 2026-06-01 15:04:09 EDT.
  - Duration: 56 minutes 48 seconds.
  - Goal: finish live validation for native Grok Build pass-through by proving
    fresh embedding persistence, fresh post-pricing online cost, and a real
    Grok Build prod smoke against `aawm_tristore.public.session_history`.
  - Changed paths:
    - `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    - `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py`
    - `tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
    - `.analysis/todo.md`
    - `.analysis/completed.md`
    - `.analysis/investigations.md`
    - `.analysis/investigation/investigate-codex-019e846d-f573-7c73-af68-0d7b24feaa36.md`
    - `suggestion.md`
  - Implementation notes:
    - Added a logging-only passthrough metadata path so `/grok` GET and
      protobuf/raw-body requests keep byte-identical upstream bodies while
      session-history logging receives Grok Build identity metadata.
    - Native Grok route now supplies `client_name=grok-build`,
      `passthrough_route_family=grok_cli_chat_proxy`, and
      `grok_model_override` metadata for JSON, GET, and raw-body requests.
    - OpenAI-compatible embedding logging now uses the Grok model override for
      xAI Grok Build embedding reporting/cost when the upstream embedding
      service responds with an internal embedding model such as
      `embedding-beta-3-small`.
    - Patched the prod-like `aawm-litellm` container in place with the same
      three Python files and restarted only that LiteLLM service for final
      validation.
  - Test evidence:
    - `./.venv/bin/python -m py_compile litellm/proxy/pass_through_endpoints/pass_through_endpoints.py litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py`
      passed.
    - `./.venv/bin/python -m pytest tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py -q -k 'grok_proxy_route or logging_metadata_without_changing_raw_body or prepare_grok_request_body_for_passthrough or xai_embedding_passthrough_payload'`
      passed (`6 passed`, 352 deselected, pre-existing warnings).
    - `./.venv/bin/python -m pytest tests/test_litellm/integrations/test_aawm_agent_identity.py -q -k 'grok or xai'`
      passed (`9 passed`, 251 deselected, one pre-existing pytest config
      warning).
  - Current Grok client evidence:
    - `grok --version` returned `grok 0.2.16 (f7c09b8d8) [stable]`.
    - `grok update --check --json` with network access returned
      `currentVersion=0.2.16`, `latestVersion=0.2.16`,
      `updateAvailable=false`, `channel=stable`.
    - Public channel pointers
      `https://storage.googleapis.com/grok-build-public-artifacts/cli/stable`
      and `/alpha` both returned `0.2.16`.
  - Dev validation evidence:
    - Restarted `litellm-dev` on port `4001`; `/health/liveliness` returned
      `"I'm alive!"`.
    - Real Grok CLI smoke:
      `GROK_CLI_CHAT_PROXY_BASE_URL=http://127.0.0.1:4001/grok/v1 grok --no-subagents --max-turns 1 -p ...`
      returned `grok smoke ok`.
    - Exact DB `aawm_tristore.public.session_history` row `1072868`
      (`litellm_call_id=3cde1240-5a35-46d3-b93c-cae6670db772`) proved
      dev online path cost: `provider=xai`, `model=grok-build`,
      `model_group=grok-build`, `client_name=grok-build`,
      `client_version=0.2.16`, `response_cost_usd=0.0067715`,
      `input_tokens=376`, `output_tokens=1338`,
      `passthrough_route_family=grok_cli_chat_proxy`.
    - Exact DB row `1072994`
      (`litellm_call_id=b14dad43-c457-4874-8a00-e3fa52e390fa`) proved
      dev embedding persistence after the embedding handler patch:
      `provider=xai`, `model=grok-build`, `model_group=grok-build`,
      `call_type=embedding`, `client_name=grok-build`,
      `client_version=0.2.16`,
      `client_user_agent=grok/0.2.16`,
      `response_cost_usd=1.25e-06`, `input_tokens=1`, `total_tokens=1`,
      `passthrough_route_family=grok_cli_chat_proxy`.
  - Prod-like validation evidence:
    - Patched and restarted `aawm-litellm` on port `4000`; `/health/liveliness`
      returned `"I'm alive!"`.
    - Real Grok CLI prod-like smoke:
      `GROK_CLI_CHAT_PROXY_BASE_URL=http://127.0.0.1:4000/grok/v1 grok --no-subagents --max-turns 1 -p ...`
      returned `grok prod smoke ok`.
    - Exact DB row `1073071`
      (`litellm_call_id=ea17bb0b-101f-480f-bdef-ece2412e4ef4`) proved
      prod-like post-pricing online cost without backfill:
      `provider=xai`, `model=grok-build`, `model_group=grok-build`,
      `client_name=grok-build`, `client_version=0.2.16`,
      `client_user_agent=grok-shell/0.2.16 (linux; x86_64)`,
      `response_cost_usd=0.029462850000000002`, `input_tokens=28187`,
      `output_tokens=154`, `passthrough_route_family=grok_cli_chat_proxy`.
    - Exact DB row `1073085`
      (`litellm_call_id=fa7c836d-1391-4888-9bcf-c432955f37ca`) independently
      proved a second prod-like costed Grok Build online request:
      `response_cost_usd=0.006679000000000001`, `input_tokens=378`,
      `output_tokens=1328`.
    - Exact DB row `1073110`
      (`litellm_call_id=0bbacbe7-d615-40ac-b7bf-62ba87c2bd6f`) proved fresh
      prod-like embedding persistence in
      `aawm_tristore.public.session_history`: `provider=xai`,
      `model=grok-build`, `model_group=grok-build`, `call_type=embedding`,
      `client_name=grok-build`, `client_version=0.2.16`,
      `client_user_agent=grok-shell/0.2.16 (linux; x86_64)`,
      `response_cost_usd=1.25e-06`, `input_tokens=1`, `output_tokens=0`,
      `total_tokens=1`, `passthrough_route_family=grok_cli_chat_proxy`.
  - Subagent evidence:
    - Read-only scout `019e846d-f573-7c73-af68-0d7b24feaa36` failed before
      producing usable output with `stream disconnected before completion:
      Incomplete response returned, reason: max_output_tokens`.
    - The failure was logged and dispositioned in `.analysis/investigations.md`
      and moved under `.analysis/investigation/`.
