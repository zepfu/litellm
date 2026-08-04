"""Focused tests for RR-054 residuals on llm_passthrough_endpoints.py."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane as cp
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


# ---------------------------------------------------------------------------
# #41 native Gemini pass-through missing-key guard (no "Bearer None")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_issue41_gemini_route_missing_key_raises_401_without_auth() -> None:
    from fastapi import HTTPException
    from starlette.requests import Request
    from starlette.responses import Response

    # No `key` query param, no `x-goog-api-key`, no Authorization headers.
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/gemini/v1/models",
        "raw_path": b"/gemini/v1/models",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    request = Request(scope)

    with patch.object(lpe, "user_api_key_auth", new=AsyncMock()) as mock_auth:
        with pytest.raises(HTTPException) as exc_info:
            await lpe.gemini_proxy_route(
                endpoint="v1/models",
                request=request,
                fastapi_response=Response(),
            )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing Google AI Studio pass-through API key."
    # Guard fires before auth, so no "Bearer None" is ever synthesized/passed.
    mock_auth.assert_not_awaited()


# ---------------------------------------------------------------------------
# #7 single pool ownership
# ---------------------------------------------------------------------------


def test_rr054_issue7_pool_helpers_are_control_plane_owned() -> None:
    assert lpe._get_aawm_dynamic_injection_pool is cp._get_aawm_dynamic_injection_pool
    assert lpe._build_aawm_dynamic_injection_dsn is cp._build_aawm_dynamic_injection_dsn
    assert lpe.close_aawm_dynamic_injection_pool is cp.close_aawm_dynamic_injection_pool
    # No second module-global pool on the god-file after consolidation.
    assert (
        not hasattr(lpe, "_aawm_dynamic_injection_pool")
        or lpe._aawm_dynamic_injection_pool is cp._aawm_dynamic_injection_pool
    )


# ---------------------------------------------------------------------------
# #2 durable cooldown max-expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_issue2_durable_write_keeps_longer_existing_expiry() -> None:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    existing_expires = time.time() + 3600.0
    dual.async_get_cache = AsyncMock(return_value={"cooldown_key": "k", "expires_at_epoch": existing_expires})

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as d

    with patch.object(d, "get_aawm_alias_routing_dual_cache", return_value=dual), patch.object(
        d, "build_aawm_alias_routing_durable_cache_key", return_value="cache-key"
    ):
        ok = await lpe._write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="k",
            payload={"cooldown_key": "k"},
            ttl_seconds=30.0,
        )

    assert ok is True
    kwargs = dual.redis_cache.async_set_cache.await_args.kwargs
    written = kwargs["value"]
    assert written["expires_at_epoch"] == pytest.approx(existing_expires, abs=1.0)
    assert kwargs["ttl"] == pytest.approx(3600.0, abs=2.0)


@pytest.mark.asyncio
async def test_rr054_issue2_durable_write_extends_when_new_expiry_later() -> None:
    dual = MagicMock()
    dual.redis_cache = MagicMock()
    dual.redis_cache.async_set_cache = AsyncMock(return_value=None)
    existing_expires = time.time() + 10.0
    dual.async_get_cache = AsyncMock(return_value={"cooldown_key": "k", "expires_at_epoch": existing_expires})

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import durable as d

    with patch.object(d, "get_aawm_alias_routing_dual_cache", return_value=dual), patch.object(
        d, "build_aawm_alias_routing_durable_cache_key", return_value="cache-key"
    ):
        before = time.time()
        ok = await lpe._write_aawm_alias_routing_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key="k",
            payload={"cooldown_key": "k"},
            ttl_seconds=120.0,
        )
        after = time.time()

    assert ok is True
    written = dual.redis_cache.async_set_cache.await_args.kwargs["value"]
    assert written["expires_at_epoch"] >= before + 119.0
    assert written["expires_at_epoch"] <= after + 121.0





# ---------------------------------------------------------------------------
# #20 parallel instruction prepend
# ---------------------------------------------------------------------------


def test_rr054_issue20_parallel_instruction_prepends_not_replaces() -> None:
    original = "You are Claude Code. Return findings directly."
    body = {
        "instructions": original,
        "parallel_tool_calls": True,
        "tools": [
            {"type": "function", "name": "Read", "parameters": {}},
            {"type": "function", "name": "Glob", "parameters": {}},
        ],
    }
    updated, changes = lpe._apply_openai_adapter_parallel_instruction_policy(body)
    assert changes["openai_adapter_parallel_instruction_policy_applied"] is True
    assert changes["openai_adapter_parallel_instruction_mode"] == "prepend"
    assert updated["instructions"].startswith("You are an OpenAI Responses function-calling agent for Claude Code.")
    assert original in updated["instructions"]
    # Second application is a no-op once policy text is present.
    second, second_changes = lpe._apply_openai_adapter_parallel_instruction_policy(updated)
    assert second is updated
    assert second_changes == {}


# ---------------------------------------------------------------------------
# #21 drop paired function_call_output
# ---------------------------------------------------------------------------


def test_rr054_issue21_drops_paired_function_call_output() -> None:
    body = {
        "input": [
            {
                "type": "function_call",
                "name": "Read",
                "call_id": "c1",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "ok",
            },
            {"type": "message", "role": "user", "content": "continue"},
            {
                "type": "function_call_output",
                "call_id": "other",
                "output": "keep",
            },
        ]
    }
    updated, dropped = lpe._drop_anthropic_grok_native_prior_function_call_replay(body)
    types = [item.get("type") for item in updated["input"] if isinstance(item, dict)]
    assert "function_call" not in types
    assert types.count("function_call_output") == 1
    assert any(d.get("type") == "function_call_output" for d in dropped)
    assert len(dropped) == 2


# ---------------------------------------------------------------------------
# #22 output_index 0 key
# ---------------------------------------------------------------------------


def test_rr054_issue22_responses_event_text_key_keeps_zero_index() -> None:
    event = SimpleNamespace(item_id=None, output_index=0)
    assert lpe._responses_event_text_key(event) == "output:0"
    event_dict = {"output_index": 0}
    assert lpe._responses_event_text_key(event_dict) == "output:0"
    event_one = SimpleNamespace(item_id=None, output_index=1)
    assert lpe._responses_event_text_key(event_one) == "output:1"


# ---------------------------------------------------------------------------
# #4 non-blocking host attribution wrapper
# ---------------------------------------------------------------------------


def test_rr054_issue4_sync_host_attribution_disables_blocking_lookup() -> None:
    request = MagicMock()
    with patch.object(
        lpe,
        "resolve_aawm_route_host_attribution",
        return_value={
            "client_ip": "1.2.3.4",
            "client_ip_source": "test",
            "host_name": None,
            "host_name_source": None,
        },
    ) as mock_resolve:
        result = lpe._resolve_auto_agent_alias_route_host_attribution(request)
    assert result["client_ip"] == "1.2.3.4"
    assert mock_resolve.call_args.kwargs.get("allow_blocking_lookup") is False


@pytest.mark.asyncio
async def test_rr054_issue4_async_host_attribution_uses_aresolve() -> None:
    request = MagicMock()
    with patch.object(
        lpe,
        "aresolve_aawm_route_host_attribution",
        new=AsyncMock(
            return_value={
                "client_ip": "1.2.3.4",
                "client_ip_source": "test",
                "host_name": "host",
                "host_name_source": "dns",
            }
        ),
    ) as mock_aresolve:
        result = await lpe._aresolve_auto_agent_alias_route_host_attribution(request)
    assert result["host_name"] == "host"
    assert mock_aresolve.await_args.kwargs.get("allow_blocking_lookup") is True


# ---------------------------------------------------------------------------
# #43 decode body
# ---------------------------------------------------------------------------


def test_rr054_issue43_decode_http_response_body_replaces_invalid_utf8() -> None:
    decoded = lpe._decode_http_response_body(b"ok\xffmore")
    assert "ok" in decoded
    assert "more" in decoded


# ---------------------------------------------------------------------------
# #47 claude root default
# ---------------------------------------------------------------------------


def test_rr054_issue47_claude_persisted_output_root_uses_home(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(lpe.Path, "home", staticmethod(lambda: tmp_path))
    root = lpe._get_claude_persisted_output_root()
    assert root == tmp_path / ".claude" / "projects"
    monkeypatch.setenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", str(tmp_path / "custom"))
    assert lpe._get_claude_persisted_output_root() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# #50 session_key on no-candidate selection
# ---------------------------------------------------------------------------


def test_rr054_issue50_no_candidate_selection_resolves_lane_scoped_session_key() -> None:
    # Runtime coverage lives in test_rr054_stream_merge_and_session_key.py.
    import inspect

    source = inspect.getsource(lpe._emit_auto_agent_alias_no_candidate_event)
    assert "_resolve_codex_auto_agent_session_key" in source
    assert "_resolve_anthropic_auto_agent_session_key" in source
    assert '"session_key": session_key' in source



def test_rr054_issue32_alias_routing_memory_maps_are_bounded() -> None:
    cache = {f"k{i}": float(i) for i in range(10)}
    lpe._bound_aawm_alias_routing_memory_map(cache, max_size=3)
    assert len(cache) == 3
    assert "k0" not in cache


# ---------------------------------------------------------------------------
# Additional packet coverage
# ---------------------------------------------------------------------------


def test_rr054_issue23_non_rate_limit_exception_types() -> None:
    from litellm.proxy._types import ProxyException

    with pytest.raises(ProxyException) as malformed:
        lpe._raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body={"output": []},
            adapter_model="m",
            adapter="a",
            adapter_label="L",
        )
    assert malformed.value.code in {502, "502", 502}
    assert malformed.value.type != "rate_limit_error"

    with pytest.raises(ProxyException) as failed:
        lpe._raise_codex_auto_agent_failed_responses_payload(
            response_body={"status": "failed", "output": []},
            adapter_model="m",
            adapter="a",
            adapter_label="L",
        )
    assert failed.value.type != "rate_limit_error"

    with pytest.raises(ProxyException) as empty:
        lpe._raise_codex_auto_agent_empty_success_response(
            response_body={"status": "completed", "output": []},
            adapter_model="m",
            adapter="a",
            adapter_label="L",
        )
    assert empty.value.type != "rate_limit_error"


def test_rr054_issue24_default_retry_status_codes_include_5xx() -> None:
    codes = list(lpe._AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT)
    assert 429 in codes
    assert 500 in codes
    assert 503 in codes


def test_rr054_issue25_drop_unsupported_params_depth_bound() -> None:
    # Deep nest should not recurse forever / explode.
    node: dict = {"model": "gpt-x"}
    cur = node
    for i in range(200):
        nxt = {"child": {}}
        cur["nested"] = nxt
        cur = nxt["child"]
    updated, removed = lpe._drop_unsupported_codex_request_params_from_request_body(node)
    assert isinstance(updated, dict)


def test_rr054_issue28_repository_extract_is_bounded() -> None:
    huge = {"messages": [{"content": "no repo here"} for _ in range(3000)]}
    assert lpe._extract_passthrough_repository_from_body_text(huge) is None
    body = {"metadata": {"repository": "zepfu/litellm"}}
    assert lpe._extract_passthrough_repository_from_body_text(body) == "zepfu/litellm"


def test_rr054_issue33_no_hardcoded_home_zepfu_auth_defaults() -> None:
    source = Path(lpe.__file__).read_text()
    # Operator-absolute auth defaults must not remain first-priority hardcoded.
    assert '"/home/zepfu/.codex/auth.json"' not in source



def test_rr054_issue48_grok_auth_formats_authorization() -> None:
    request = MagicMock()
    request.headers = {"Authorization": "sk-test-key"}
    request.query_params = {}
    # _format may prefix Bearer
    with patch.object(lpe, "_format_litellm_passthrough_api_key", side_effect=lambda v: f"Bearer {v}"):
        assert lpe._get_grok_litellm_auth_header(request) == "Bearer sk-test-key"


def test_rr054_issue52_affinity_hydrate_keeps_fresher_memory() -> None:
    memory: dict[str, dict] = {
        "s1": {
            "provider": "local",
            "model": "m1",
            "route_family": "r",
            "last_resort": False,
            "expires_at_monotonic": time.monotonic() + 9999,
        }
    }
    out = lpe._hydrate_aawm_alias_routing_affinity_memory(
        memory_map=memory,
        session_key="s1",
        payload={
            "provider": "remote",
            "model": "m2",
            "route_family": "r",
            "last_resort": False,
        },
        expires_at_epoch=time.time() + 10,
    )
    assert out["provider"] == "local"



def test_rr054_issue55_redacts_broader_secrets() -> None:
    redacted = lpe._redact_tool_definition_string("token=ghp_abcdefghijklmnopqrstuv")
    assert "ghp_" not in redacted or redacted != "token=ghp_abcdefghijklmnopqrstuv"


def test_rr054_issue56_headers_filtered_for_metadata() -> None:
    request = MagicMock()
    request.headers = {
        "x-aawm-agent-id": "agent-1",
        "authorization": "Bearer secret",
        "x-evil": "nope",
    }
    with patch.object(lpe, "_safe_get_request_headers", return_value=request.headers):
        sources = lpe._iter_auto_agent_alias_metadata_dicts(request, {})
    assert any(s.get("x-aawm-agent-id") == "agent-1" for s in sources)
    assert all("authorization" not in s for s in sources)


# ---------------------------------------------------------------------------
# #16 vertex live auth-first / docstring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rr054_issue16_vertex_live_requires_auth_before_accept() -> None:
    websocket = MagicMock()
    websocket.headers = {}
    websocket.client_state = lpe.WebSocketState.CONNECTING
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()

    with pytest.raises(ValueError, match="user_api_key_dict is required"):
        await lpe.vertex_ai_live_websocket_passthrough(
            websocket=websocket,
            user_api_key_dict=None,
        )
    websocket.close.assert_awaited()
    websocket.accept.assert_not_awaited()


def test_rr054_issue16_vertex_live_docstring_requires_auth_wrapper() -> None:
    import inspect

    doc = inspect.getdoc(lpe.vertex_ai_live_websocket_passthrough) or ""
    assert "user_api_key_auth_websocket" in doc
    assert "Do not register this function" in doc


# ---------------------------------------------------------------------------
# #38 malformed-tool-call intake logs schedule failures
# ---------------------------------------------------------------------------


def test_rr054_issue38_malformed_tool_call_schedule_logs_on_failure() -> None:
    from litellm.proxy._types import ProxyException

    with patch.object(
        lpe,
        "schedule_persist_malformed_tool_call_detection",
        side_effect=RuntimeError("boom"),
    ), patch.object(lpe.verbose_proxy_logger, "exception") as mock_exc:
        with pytest.raises(ProxyException):
            lpe._raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body={"output": []},
                adapter_model="m",
                adapter="a",
                adapter_label="L",
            )
    mock_exc.assert_called()


# ---------------------------------------------------------------------------
# #1/#11 policy seam module
# ---------------------------------------------------------------------------


def test_rr054_issue1_11_policy_module_exports_cooldowns() -> None:
    from litellm.proxy.pass_through_endpoints import aawm_alias_routing_policy as policy
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        alias_routing_state,
    )

    assert policy.CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS == 30.0
    assert lpe._CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS == (
        policy.CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS
    )
    # State maps/locks are package-owned and re-exported by the god-file.
    assert lpe._codex_auto_agent_lock is alias_routing_state.codex.lock
    assert lpe._anthropic_auto_agent_lock is alias_routing_state.anthropic.lock
    assert (
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key
        is alias_routing_state.codex.cooldown_until_monotonic_by_key
    )
    assert (
        lpe._openrouter_adapter_rate_limit_until_monotonic_by_key
        is alias_routing_state.openrouter_rate_limit.until_monotonic_by_key
    )


def test_rr054_issue11_policy_module_owns_candidate_tables_and_allowlists() -> None:
    from litellm.proxy.pass_through_endpoints import aawm_alias_routing_policy as policy
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        policy as package_policy,
    )

    policy_path = Path(package_policy.__file__).resolve()
    assert policy_path.name == "policy.py"
    assert policy_path.parent.name == "aawm_alias_routing"
    assert "basic" in policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert "work" in policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert lpe._CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS is policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS is package_policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert lpe._ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS is policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert (
        lpe._ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS
        is policy.ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS
    )
    assert "gpt-5.3-codex-spark" in policy.ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS
    # God-file installs same-object policy aliases rather than defining rows.
    god_source = Path(lpe.__file__).read_text()
    assert "_aawm_alias_policy_compat.install_policy_compat_aliases(globals())" in god_source
    assert "_POLICY_CODEX_AUTO_AGENT_CANDIDATES" not in god_source
    assert "_POLICY_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS" not in god_source
    assert (
        package_policy.COMPAT_ALIAS_MAP["_CODEX_AUTO_AGENT_CANDIDATES"]
        == "CODEX_AUTO_AGENT_CANDIDATES"
    )
    assert (
        package_policy.COMPAT_ALIAS_MAP[
            "_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS"
        ]
        == "ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS"
    )
    package_source = policy_path.read_text()
    assert "CODEX_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (" in package_source
    assert '"last_resort": True,' in package_source
    # Compat shim must not own the table literals.
    compat_source = Path(policy.__file__).read_text()
    assert "Compatibility re-export" in compat_source
    assert '"last_resort": True,' not in compat_source


def test_rr054_issue9_responses_adapter_finalize_helper_is_shared() -> None:
    import inspect

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        adapter_config,
        adapter_driver,
    )

    assert hasattr(lpe, "_finalize_anthropic_responses_adapter_upstream_response")
    assert hasattr(lpe, "_finalize_anthropic_responses_adapter_from_config")
    assert hasattr(lpe, "_perform_anthropic_responses_adapter_pass_through")
    assert hasattr(lpe, "_perform_anthropic_completion_adapter_messages_call")
    assert hasattr(lpe, "_apply_anthropic_responses_adapter_common_request_policies")
    assert hasattr(lpe, "_prepare_anthropic_completion_adapter_request_body")
    assert adapter_config.OPENAI_RESPONSES.adapter == "anthropic_openai_responses_adapter"
    for handler_name, prepare_name, provider_module in (
        (
            "_handle_anthropic_openai_responses_adapter_route",
            "_prepare_anthropic_openai_responses_adapter_route",
            lpe._anthropic_openai_provider,
        ),
        (
            "_handle_anthropic_xai_oauth_responses_adapter_route",
            "_prepare_anthropic_xai_oauth_responses_adapter_route",
            lpe._anthropic_xai_provider,
        ),
        (
            "_handle_anthropic_grok_native_oauth_responses_adapter_route",
            "_prepare_anthropic_grok_native_oauth_responses_adapter_route",
            lpe._anthropic_grok_provider,
        ),
        (
            "_handle_anthropic_openrouter_responses_adapter_route",
            "_prepare_anthropic_openrouter_responses_adapter_route",
            lpe._anthropic_openrouter_provider,
        ),
        (
            "_handle_anthropic_opencode_zen_responses_adapter_route",
            "_prepare_anthropic_opencode_zen_responses_adapter_route",
            lpe._anthropic_opencode_zen_provider,
        ),
    ):
        wrapper_source = inspect.getsource(getattr(lpe, handler_name))
        prepare_source = inspect.getsource(getattr(lpe, prepare_name))
        provider_source = inspect.getsource(provider_module.prepare_responses_route)
        assert "_aawm_adapter_driver.run_responses_adapter_route" in wrapper_source
        assert len(wrapper_source.splitlines()) <= 25
        assert "_aawm_adapter_driver.ResponsesAdapterRoutePlan" in prepare_source
        assert "adapter_config." in provider_source
        assert "ResponsesAdapterRoutePlan" in provider_source
    for handler_name, prepare_name, provider_module in (
        (
            "_handle_anthropic_xai_oauth_completion_adapter_route",
            "_prepare_anthropic_xai_oauth_completion_adapter_route",
            lpe._anthropic_xai_provider,
        ),
        (
            "_handle_anthropic_nvidia_completion_adapter_route",
            "_prepare_anthropic_nvidia_completion_adapter_route",
            lpe._anthropic_nvidia_provider,
        ),
        (
            "_handle_anthropic_openrouter_completion_adapter_route",
            "_prepare_anthropic_openrouter_completion_adapter_route",
            lpe._anthropic_openrouter_provider,
        ),
        (
            "_handle_anthropic_opencode_zen_completion_adapter_route",
            "_prepare_anthropic_opencode_zen_completion_adapter_route",
            lpe._anthropic_opencode_zen_provider,
        ),
    ):
        wrapper_source = inspect.getsource(getattr(lpe, handler_name))
        prepare_source = inspect.getsource(getattr(lpe, prepare_name))
        provider_source = inspect.getsource(provider_module.prepare_completion_route)
        assert "_aawm_adapter_driver.run_completion_adapter_route" in wrapper_source
        assert len(wrapper_source.splitlines()) <= 25
        assert "_aawm_adapter_driver.CompletionAdapterRoutePlan" in prepare_source
        assert "adapter_config." in provider_source
        assert "CompletionAdapterRoutePlan" in provider_source
    assert hasattr(adapter_driver, "run_responses_adapter_route")
    assert hasattr(adapter_driver, "run_completion_adapter_route")


def test_rr054_issue12_shared_openrouter_retry_and_auto_agent_cooldown() -> None:
    import inspect

    from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
        retry_transport,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import retry as ar_retry

    assert hasattr(lpe, "_run_openrouter_adapter_retry_loop")
    assert hasattr(lpe, "_apply_auto_agent_alias_cooldown")
    assert hasattr(ar_retry, "wait_for_monotonic_cooldown_map")
    assert hasattr(ar_retry, "set_monotonic_cooldown_map")
    assert hasattr(ar_retry, "projected_hidden_retry_within_budget")
    assert hasattr(ar_retry, "AdapterRetryPolicy")
    assert hasattr(ar_retry, "run_adapter_retry_policy")
    completion = inspect.getsource(lpe._perform_openrouter_completion_adapter_operation)
    passthrough = inspect.getsource(lpe._perform_openrouter_adapter_pass_through_request)
    assert "perform_completion_operation" in completion
    assert "perform_pass_through_request" in passthrough
    assert "run_retry_loop" in inspect.getsource(retry_transport.perform_completion_operation)
    assert "run_retry_loop" in inspect.getsource(retry_transport.perform_pass_through_request)
    codex = inspect.getsource(lpe._apply_codex_auto_agent_alias_cooldown)
    anth = inspect.getsource(lpe._apply_anthropic_auto_agent_alias_cooldown)
    assert "_apply_auto_agent_alias_cooldown" in codex
    assert "_apply_auto_agent_alias_cooldown" in anth
    shared = inspect.getsource(lpe._apply_auto_agent_alias_cooldown)
    assert "set_candidate_cooldown" in shared
    openrouter_wait = inspect.getsource(lpe._wait_for_openrouter_adapter_cooldown_if_needed)
    assert "wait_for_cooldown_if_needed" in openrouter_wait
    assert "wait_for_monotonic_cooldown_map" in inspect.getsource(retry_transport.wait_for_cooldown_if_needed)
    openrouter_loop = inspect.getsource(lpe._run_openrouter_adapter_retry_loop)
    assert "retry_transport.run_retry_loop" in openrouter_loop
    assert "run_adapter_retry_policy" in inspect.getsource(retry_transport.run_retry_loop)


# ---------------------------------------------------------------------------
# #36 local state uses asyncio locks only (no threading.Lock in this file)
# ---------------------------------------------------------------------------


def test_rr054_issue36_alias_state_uses_asyncio_locks() -> None:
    import asyncio

    source = Path(lpe.__file__).read_text()
    assert "threading.Lock" not in source
    assert isinstance(lpe._codex_auto_agent_lock, asyncio.Lock)
    assert isinstance(lpe._anthropic_auto_agent_lock, asyncio.Lock)


def test_rr054_issue10_shared_auto_agent_handler_is_wired() -> None:
    import inspect

    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import candidate_loop
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        anthropic_auto_agent_route,
        codex_auto_agent_route,
    )

    assert hasattr(lpe, "_handle_auto_agent_alias_route")
    assert hasattr(lpe, "_dispatch_auto_agent_alias_candidate_request")
    anth_facade = lpe._handle_anthropic_auto_agent_alias_route
    codex_facade = lpe._handle_codex_auto_agent_alias_route
    assert isinstance(anth_facade, partial)
    assert isinstance(codex_facade, partial)
    assert anth_facade.func is anthropic_auto_agent_route.handle_anthropic_auto_agent_alias_route
    assert codex_facade.func is codex_auto_agent_route.handle_codex_auto_agent_alias_route
    assert isinstance(
        anth_facade.args[0],
        anthropic_auto_agent_route.AnthropicAutoAgentRouteRuntime,
    )
    assert isinstance(
        codex_facade.args[0],
        codex_auto_agent_route.CodexAutoAgentRouteRuntime,
    )
    anth = inspect.getsource(anth_facade.func)
    codex = inspect.getsource(codex_facade.func)
    # Anthropic receives the shared candidate loop through runtime injection;
    # Codex imports the same package owner directly.
    assert "AliasRouteServices" in anth
    assert "runtime.handle_alias_route" in anth
    assert "AliasRouteServices" in codex
    assert codex_auto_agent_route.handle_alias_route is candidate_loop.handle_alias_route
    assert "return await handle_alias_route(" in codex
    assert "runtime.handle_alias_route" not in codex
    shared = inspect.getsource(candidate_loop.handle_alias_route)
    assert "signaling redispatch" in shared
    assert "native_grok_continuation_same_candidate_retry" in shared
    anth_perform = inspect.getsource(lpe._perform_anthropic_auto_agent_alias_candidate_request)
    codex_perform = inspect.getsource(lpe._perform_codex_auto_agent_alias_candidate_request)
    assert "_dispatch_auto_agent_alias_candidate_request" in anth_perform
    assert "_dispatch_auto_agent_alias_candidate_request" in codex_perform
    assert "route_family_handlers" in anth_perform
    assert "route_family_handlers" in codex_perform


# ---------------------------------------------------------------------------
# #27 SSE dict path (no namespace coerce on yield)
# ---------------------------------------------------------------------------


def test_rr054_issue27_sse_iterator_yields_dicts_not_namespaces() -> None:
    import inspect

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import sse as sse_mod

    # Authoritative implementation lives in the extracted sse module.
    sse_source = Path(sse_mod.__file__).read_text()
    # Hot path yields parsed dicts; no coerce-to-namespace on emit.
    assert "yield parsed_chunk" in sse_source
    iterate_src = inspect.getsource(sse_mod._iterate_responses_sse_events)
    assert "_coerce_mapping_to_namespace" not in iterate_src

    # The lpe facade must point to the moved function (same code object).
    assert hasattr(lpe, "_iterate_responses_sse_events")
    assert lpe._iterate_responses_sse_events.__code__ is sse_mod._iterate_responses_sse_events.__code__
    # Facade behavioral assertion: no namespace coercion on yield path.
    facade_src = inspect.getsource(lpe._iterate_responses_sse_events)
    assert "_coerce_mapping_to_namespace" not in facade_src


# ---------------------------------------------------------------------------
# #35 structured/configurable task-state contract
# ---------------------------------------------------------------------------


def test_rr054_issue35_task_state_prefers_structured_flag_and_env_markers() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import task_state

    markers = task_state.resolve_task_state_markers("alpha marker|beta marker")
    assert markers == ("alpha marker", "beta marker")

    messages = [
        {"role": "user", "content": "unrelated filler text"},
        {
            "role": "user",
            "content": "structured wins",
            "metadata": {"aawm_preserve_task_state": True},
        },
        {"role": "user", "content": "alpha marker should not be needed"},
    ]
    selected = task_state.select_task_state_source(
        messages,
        extract_text=lambda m: str(m.get("content") or ""),
        is_skippable=lambda m: m.get("role") == "tool",
        markers=markers,
    )
    assert selected is not None
    assert selected[0] == 1
    assert selected[2] == "structured"



# ---------------------------------------------------------------------------
# #39 access-log annotation does not mutate live query_string
# ---------------------------------------------------------------------------


def test_rr054_issue39_annotate_does_not_mutate_query_string() -> None:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"foo=1",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    request = Request(scope)
    original_qs = scope["query_string"]
    original_path = scope["path"]
    lpe._annotate_request_scope_for_adapted_access_log(request, "https://api.openai.com/v1/responses")
    assert scope["query_string"] == original_qs
    assert scope["path"] == original_path
    assert isinstance(scope.get("_aawm_adapted_access_log_target"), str)
    assert isinstance(scope.get("_aawm_adapted_access_log_display_path"), str)
    assert "foo=1" in scope["_aawm_adapted_access_log_display_path"]
    assert (
        "->" in scope["_aawm_adapted_access_log_display_path"]
        or "adapted_to=" in scope["_aawm_adapted_access_log_display_path"]
    )


# ---------------------------------------------------------------------------
# #27 behavioral: collect dict SSE events (response.completed / items / deltas)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_rr054_issue27_collect_responses_from_dict_sse_stream() -> None:
    """Dict SSE events must collect terminal response (not false-502)."""
    import json
    from starlette.responses import StreamingResponse

    def _sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode("utf-8")

    chunks = [
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [],
                },
            }
        ),
        _sse(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "delta": "Hello",
            }
        ),
        _sse(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "delta": " world",
            }
        ),
        _sse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                },
            }
        ),
        _sse(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Hello world"}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            }
        ),
    ]

    async def _gen():
        for chunk in chunks:
            yield chunk

    response = StreamingResponse(_gen(), media_type="text/event-stream")
    summaries: list[dict] = []
    collected = await lpe._collect_responses_response_from_stream(response, event_summaries=summaries)
    assert collected["id"] == "resp_1"
    assert collected["status"] == "completed"
    assert isinstance(collected.get("output"), list)
    assert len(collected["output"]) >= 1
    assert any(s.get("type") == "response.output_text.delta" for s in summaries)
    assert any(s.get("type") == "response.completed" for s in summaries)


@pytest.mark.asyncio
async def test_rr054_issue27_collect_function_call_args_from_dict_sse() -> None:
    import json
    from starlette.responses import StreamingResponse

    def _sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode("utf-8")

    chunks = [
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "Bash",
                },
            }
        ),
        _sse(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '{"cmd":',
            }
        ),
        _sse(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": '{"cmd":"ls"}',
            }
        ),
        _sse(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_fc",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "Bash",
                            "arguments": '{"cmd":"ls"}',
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }
        ),
    ]

    async def _gen():
        for chunk in chunks:
            yield chunk

    response = StreamingResponse(_gen(), media_type="text/event-stream")
    collected = await lpe._collect_responses_response_from_stream(response)
    assert collected["id"] == "resp_fc"
    output = collected.get("output") or []
    assert any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)



def test_rr054_issue12_retry_env_parsers_are_shared() -> None:
    from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
        retry_transport,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import retry as ar_retry
    import inspect

    assert "parse_non_negative_int_env" in inspect.getsource(retry_transport.get_max_retries)
    assert ar_retry.parse_non_negative_int_env("NOPE", default=2) == 2
    assert ar_retry.parse_non_negative_float_env("NOPE", default=1.5) == 1.5
