"""Stubbed Codex passthrough dispatch for Z.AI Coding Plan.

Proves POST /openai_passthrough/v1/responses with
model=zai_coding_plan/glm-5.3 hits the coding chat URL. HTTP is stubbed;
this is not live Z.AI fanout.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response
from starlette.requests import Request

from litellm.llms.zai_coding_plan.chat.transformation import (
    ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
    ZAICodingPlanAuthenticationError,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
    model_resolution,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _handle_codex_zai_coding_plan_adapter_route,
    _prepare_codex_zai_coding_plan_adapter_route,
    _resolve_codex_zai_coding_plan_adapter_model,
)


_ADAPTER_MODEL = "zai_coding_plan/glm-5.3"
_CODEX_ZAI_ROUTE_FAMILY = "codex_zai_coding_plan_chat_completions_adapter"
_ZAI_COOLDOWN_IDENTITY_TAG = (
    "alias:sota-zai:zai_coding_plan:"
    f"{_ADAPTER_MODEL}:{_CODEX_ZAI_ROUTE_FAMILY}"
)
_ZAI_COOLDOWN_KEY = (
    f"h{_ZAI_COOLDOWN_IDENTITY_TAG}:zai_coding_plan:"
    f"{_ADAPTER_MODEL}:zai_coding_plan"
)


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    request.state = MagicMock()
    return request


def test_should_resolve_prefixed_direct_coding_plan_models_on_responses() -> None:
    assert (
        _resolve_codex_zai_coding_plan_adapter_model(
            {"model": _ADAPTER_MODEL},
            "/v1/responses",
        )
        == _ADAPTER_MODEL
    )
    assert (
        _resolve_codex_zai_coding_plan_adapter_model(
            {"model": _ADAPTER_MODEL},
            "v1/responses",
        )
        == _ADAPTER_MODEL
    )
    assert (
        model_resolution._resolve_codex_zai_coding_plan_adapter_model(
            {"model": _ADAPTER_MODEL},
            "/v1/responses",
        )
        == _ADAPTER_MODEL
    )
    assert (
        _resolve_codex_zai_coding_plan_adapter_model(
            {"model": "glm-5.3"},
            "/v1/responses",
        )
        is None
    )
    assert (
        _resolve_codex_zai_coding_plan_adapter_model(
            {"model": "sota-zai"},
            "/v1/responses",
        )
        is None
    )
    assert (
        _resolve_codex_zai_coding_plan_adapter_model(
            {"model": "zai/glm-5.3"},
            "/v1/responses",
        )
        is None
    )
    assert (
        _resolve_codex_zai_coding_plan_adapter_model(
            {"model": _ADAPTER_MODEL},
            "/v1/chat/completions",
        )
        is None
    )
    assert not hasattr(
        model_resolution, "_resolve_anthropic_zai_coding_plan_adapter_model"
    )


@pytest.mark.asyncio
async def test_should_prepare_codex_coding_plan_route_to_coding_chat_url() -> None:
    plan = await _prepare_codex_zai_coding_plan_adapter_route(
        request=_request(),
        adapter_model=_ADAPTER_MODEL,
        prepared_request_body={
            "model": _ADAPTER_MODEL,
            "input": "hello",
            "stream": False,
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    assert str(plan.target_url) == ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL
    assert "/api/paas/v4" not in str(plan.target_url)
    assert "open.bigmodel.cn" not in str(plan.target_url)
    assert completion_kwargs["model"] == "glm-5.3"
    assert completion_kwargs["custom_llm_provider"] == "zai_coding_plan"
    assert "sota-zai" not in json.dumps(completion_kwargs)


@pytest.mark.asyncio
async def test_should_filter_zai_unsupported_tools_and_keep_function_tools() -> None:
    plan = await _prepare_codex_zai_coding_plan_adapter_route(
        request=_request(),
        adapter_model=_ADAPTER_MODEL,
        prepared_request_body={
            "model": _ADAPTER_MODEL,
            "input": "inspect",
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
                {"type": "custom", "name": "exec_command"},
                {"type": "mcp", "server_label": "external_mcp"},
                {"type": "web_search_preview"},
                {"type": "custom", "name": "apply_patch"},
            ],
            "tool_choice": {"type": "custom", "name": "exec_command"},
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    completion_tools = completion_kwargs["tools"]
    assert [tool["type"] for tool in completion_tools] == ["function", "function"]
    assert [
        tool["function"]["name"] for tool in completion_tools
    ] == ["read_file", "apply_patch"]
    assert "tool_choice" not in completion_kwargs

    metadata = plan.prepared_request_body["litellm_metadata"]
    assert metadata["zai_coding_plan_removed_unsupported_tool_count"] == 3
    assert metadata["zai_coding_plan_removed_unsupported_tool_types"] == [
        "custom",
        "mcp",
        "web_search_preview",
    ]
    assert metadata["zai_coding_plan_removed_unsupported_tool_names"] == [
        "exec_command",
        "external_mcp",
    ]
    assert metadata["zai_coding_plan_removed_unsupported_tools"] == [
        {"index": 1, "type": "custom", "name": "exec_command"},
        {"index": 2, "type": "mcp", "name": "external_mcp"},
        {"index": 3, "type": "web_search_preview"},
    ]
    assert metadata["zai_coding_plan_removed_unsupported_tool_choice"] == {
        "type": "custom",
        "name": "exec_command",
    }
    assert metadata["langfuse_spans"][-1]["name"] == (
        "zai_coding_plan.unsupported_tool_removed"
    )


@pytest.mark.asyncio
async def test_should_remove_zai_tool_choice_when_all_tools_are_filtered() -> None:
    plan = await _prepare_codex_zai_coding_plan_adapter_route(
        request=_request(),
        adapter_model=_ADAPTER_MODEL,
        prepared_request_body={
            "model": _ADAPTER_MODEL,
            "input": "inspect",
            "tools": [{"type": "custom", "name": "exec_command"}],
            "tool_choice": "required",
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    assert not completion_kwargs.get("tools")
    assert "tool_choice" not in completion_kwargs
    assert "tools" not in plan.prepared_request_body
    assert "tool_choice" not in plan.prepared_request_body
    assert plan.prepared_request_body["litellm_metadata"][
        "zai_coding_plan_removed_unsupported_tool_choice"
    ] == "required"


@pytest.mark.asyncio
async def test_should_preserve_zai_function_tools_for_previous_response_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    captured: dict[str, Any] = {}

    async def _session_handler(
        *,
        previous_response_id: str,
        litellm_completion_request: dict[str, Any],
    ) -> dict[str, Any]:
        captured["previous_response_id"] = previous_response_id
        captured["litellm_completion_request"] = litellm_completion_request
        return {
            **litellm_completion_request,
            "litellm_trace_id": "zai-continuation-trace",
        }

    monkeypatch.setattr(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        AsyncMock(side_effect=_session_handler),
    )

    plan = await _prepare_codex_zai_coding_plan_adapter_route(
        request=_request(),
        adapter_model=_ADAPTER_MODEL,
        prepared_request_body={
            "model": _ADAPTER_MODEL,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "read_file:1",
                    "output": "contents",
                }
            ],
            "previous_response_id": "resp_zai_continuation",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": "custom", "name": "exec_command"},
            ],
        },
    )

    continued_request = captured["litellm_completion_request"]
    assert captured["previous_response_id"] == "resp_zai_continuation"
    assert continued_request["tools"][0]["function"]["name"] == "read_file"
    tool_message = continued_request["messages"][0]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "read_file:1"
    assert tool_message["content"] == "contents"
    assert plan.perform_kwargs["completion_kwargs"]["litellm_trace_id"] == (
        "zai-continuation-trace"
    )
    assert plan.perform_kwargs["completion_kwargs"]["tools"][0]["function"][
        "name"
    ] == "read_file"
    assert plan.prepared_request_body["litellm_metadata"][
        "zai_coding_plan_removed_unsupported_tool_names"
    ] == ["exec_command"]


@pytest.mark.asyncio
async def test_should_post_codex_responses_direct_model_to_coding_chat_url(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock,
) -> None:
    import litellm
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints

    watermark_calls: list[dict[str, Any]] = []

    def _watermark_response(body, **kwargs):
        watermark_calls.append({"body": body, **kwargs})
        return body, kwargs["content"]

    def _close_logging_coroutine(async_coroutine, metadata=None):
        _ = metadata
        close = getattr(async_coroutine, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(
        GLOBAL_LOGGING_WORKER,
        "ensure_initialized_and_enqueue",
        _close_logging_coroutine,
    )
    monkeypatch.setattr(
        llm_passthrough_endpoints,
        "maybe_apply_passthrough_watermark_response",
        _watermark_response,
    )
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "enable_preview_features", True)
    respx_mock.post(ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-coding-plan",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        }
    )

    try:
        response = await _handle_codex_zai_coding_plan_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": _ADAPTER_MODEL,
                "input": "hello from openai_passthrough",
                "stream": False,
                "tools": [{"type": "unsupported", "name": "blocked_tool"}],
                "tool_choice": "required",
            },
            adapter_model=_ADAPTER_MODEL,
        )
    finally:
        await GLOBAL_LOGGING_WORKER.stop()
        await litellm.close_litellm_async_clients()

    assert respx_mock.calls
    upstream_request = respx_mock.calls[0].request
    assert str(upstream_request.url) == ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL
    assert "/api/paas/v4" not in str(upstream_request.url)
    assert "open.bigmodel.cn" not in str(upstream_request.url)
    upstream_body = json.loads(upstream_request.content)
    assert upstream_body["model"] == "glm-5.3"
    assert "metadata" not in upstream_body
    assert "litellm_metadata" not in upstream_body
    assert "zai_coding_plan_removed_unsupported_tool_choice" not in upstream_body
    assert "sota-zai" not in upstream_request.content.decode()
    assert "zai_coding_plan/" not in upstream_request.content.decode()
    assert upstream_request.headers["Authorization"] == "Bearer coding-plan-key"
    response_body = json.loads(response.body)
    assert response_body["object"] == "response"
    assert watermark_calls
    assert watermark_calls[0]["endpoint"] == "responses"
    assert any(
        item.get("type") == "message" for item in response_body.get("output", [])
    )


@pytest.mark.asyncio
async def test_alias_candidate_provider_handler_dispatches_coding_plan() -> None:
    sentinel = Response(content=b"coding-plan", status_code=200)
    captured: dict[str, Any] = {}

    async def _handle_coding_plan(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return sentinel

    host: dict[str, Any] = {
        "__builtins__": __builtins__,
        "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "opencode_zen",
        "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "kimi_code",
        "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": "alibaba_token_plan",
        "_CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER": "zai_coding_plan",
        "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "openrouter",
        "_CODEX_AUTO_AGENT_XAI_PROVIDER": "xai",
        "_CODEX_AUTO_AGENT_COHERE_PROVIDER": "cohere",
        "_CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER": "cursor_agent",
        "_handle_codex_alibaba_token_plan_adapter_route": AsyncMock(
            side_effect=AssertionError("alibaba must not win")
        ),
        "_handle_codex_kimi_chat_completions_adapter_route": AsyncMock(
            side_effect=AssertionError("kimi must not win")
        ),
        "_handle_codex_opencode_zen_adapter_route": AsyncMock(
            side_effect=AssertionError("opencode must not win")
        ),
        "_perform_codex_auto_agent_native_openai_request": AsyncMock(
            side_effect=AssertionError("native openai must not win")
        ),
        "_dispatch_auto_agent_alias_candidate_request": None,
    }

    async def _dispatch(*, candidate, provider_handlers, **kwargs: Any) -> Response:
        handler = provider_handlers[candidate["provider"]]
        return await handler()

    host["_dispatch_auto_agent_alias_candidate_request"] = _dispatch
    # install() rebinds owned symbols onto the host; keep the Coding Plan
    # handler mock after that so the dispatcher does not call the live route.
    codex_candidate_calls.install(host)
    host["_handle_codex_zai_coding_plan_adapter_route"] = _handle_coding_plan

    result = await host["_perform_codex_auto_agent_alias_candidate_request"](
        endpoint="/v1/responses",
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=MagicMock(),
        candidate={
            "provider": "zai_coding_plan",
            "model": _ADAPTER_MODEL,
            "route_family": _CODEX_ZAI_ROUTE_FAMILY,
        },
        candidate_body={"model": _ADAPTER_MODEL},
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key=None,
        forward_headers=False,
    )

    assert result is sentinel
    assert captured["adapter_model"] == _ADAPTER_MODEL
    assert captured["use_alias_candidate_probe"] is True


class _StructuredCodingPlanError(Exception):
    def __init__(self, *, code: int, message: str, status_code: int = 429) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = {
            "error": {
                "type": "invalid_request_error",
                "code": code,
                "message": message,
            }
        }


def test_should_map_coding_plan_1113_to_wrong_base_terminal_error() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    candidate = {
        "provider": "zai_coding_plan",
        "model": _ADAPTER_MODEL,
        "route_family": _CODEX_ZAI_ROUTE_FAMILY,
    }
    exc = _StructuredCodingPlanError(code=1113, message="Insufficient balance")
    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            exc,
            candidate=candidate,
        )
        == "provider_terminal_error"
    )
    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            exc,
            candidate={"provider": "openai", "model": "gpt-5"},
        )
        is None
    )


@pytest.mark.parametrize(
    ("error_code", "expected_class"),
    (
        (1000, "provider_terminal_error"),
        (1001, "provider_terminal_error"),
        (1113, "provider_terminal_error"),
        (1302, "rate_limited"),
        (1308, "usage_limit_reached"),
        (1309, "usage_limit_reached"),
        (1310, "usage_limit_reached"),
        (1313, "usage_limit_reached"),
        (1316, "usage_limit_reached"),
        (1317, "usage_limit_reached"),
    ),
)
def test_should_map_coding_plan_business_codes_in_candidate_loop(
    error_code: int,
    expected_class: str,
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    exc = _StructuredCodingPlanError(code=error_code, message=f"coding plan {error_code}")
    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            exc,
            candidate={
                "provider": "zai_coding_plan",
                "model": _ADAPTER_MODEL,
                "route_family": _CODEX_ZAI_ROUTE_FAMILY,
            },
        )
        == expected_class
    )


@pytest.mark.parametrize("error_code", (1211, 1311))
def test_should_map_zai_model_codes_only_for_attributed_provider_returns(
    error_code: int,
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    exc = _StructuredCodingPlanError(
        code=error_code,
        message=f"provider model admission rejected: {error_code}",
        status_code=400,
    )
    setattr(exc, "_aawm_provider_returned", True)

    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            exc,
            candidate={
                "provider": "zai_coding_plan",
                "model": _ADAPTER_MODEL,
                "route_family": _CODEX_ZAI_ROUTE_FAMILY,
            },
            attempted_provider_call=True,
        )
        == "candidate_unavailable"
    )


@pytest.mark.parametrize("error_code", (1211, 1311))
@pytest.mark.parametrize("attempted_provider_call", (False, True))
def test_should_keep_unmarked_local_zai_model_codes_terminal(
    error_code: int,
    attempted_provider_call: bool,
) -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    exc = _StructuredCodingPlanError(
        code=error_code,
        message=f"local model validation rejected: {error_code}",
        status_code=400,
    )

    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            exc,
            candidate={
                "provider": "zai_coding_plan",
                "model": _ADAPTER_MODEL,
                "route_family": _CODEX_ZAI_ROUTE_FAMILY,
            },
            attempted_provider_call=attempted_provider_call,
        )
        == "provider_terminal_error"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", (1211, 1311))
async def test_should_fail_over_in_order_after_zai_model_admission_failure(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
    error_code: int,
) -> None:
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        AliasRoutingStateManager,
    )

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [(b"user-agent", b"codex-cli/1.0")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "scheme": "http",
        }
    )
    zai_candidate = {
        "provider": "zai_coding_plan",
        "model": _ADAPTER_MODEL,
        "route_family": _CODEX_ZAI_ROUTE_FAMILY,
        "cooldown_identity_tag": _ZAI_COOLDOWN_IDENTITY_TAG,
    }
    alibaba_candidate = {
        "provider": "alibaba_token_plan",
        "model": "alibaba_token_plan/glm-5.2",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
    }
    selections = [
        {
            "candidate": zai_candidate,
            "lane_key": "zai_coding_plan",
            "cooldown_key": _ZAI_COOLDOWN_KEY,
            "selection_reason": "first_available",
            "skipped": [],
        },
        {
            "candidate": alibaba_candidate,
            "lane_key": "alibaba_token_plan",
            "cooldown_key": "alibaba_token_plan:alibaba_token_plan/glm-5.2:alibaba_token_plan",
            "selection_reason": "failover",
            "skipped": [],
        },
    ]
    provider_calls: list[str] = []
    publication_plans: list[Any] = []
    failure_attempts: list[dict[str, Any]] = []

    async def _select(
        *,
        request: Request,
        request_body: dict[str, Any],
        excluded_candidate_keys: object = None,
    ) -> dict[str, Any]:
        _ = request, request_body, excluded_candidate_keys
        return selections.pop(0)

    async def _perform(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> object:
        _ = candidate_body
        provider_calls.append(candidate["provider"])
        if candidate["provider"] == "zai_coding_plan":
            exc = _StructuredCodingPlanError(
                code=error_code,
                message=f"model admission rejected: {error_code}",
            )
            setattr(exc, "_aawm_provider_returned", True)
            raise exc
        return {"provider": candidate["provider"], "ok": True}

    def _resolve_publication(**kwargs: Any) -> object:
        resolver_kwargs = dict(kwargs)
        resolver_kwargs["codex_failure_evidence_alias"] = None
        plan = lpe._resolve_auto_agent_cooldown_publication_plan(**resolver_kwargs)
        publication_plans.append(plan)
        return plan

    async def _execute_publication(
        *,
        plan: Any,
        publish_cooldown_memory_fn: Any,
        **_kwargs: Any,
    ) -> None:
        publish_cooldown_memory_fn(
            keys=plan.memory_keys,
            seconds=plan.duration_seconds,
            allow_ttl_shrink=plan.allow_ttl_shrink,
        )

    async def _active_cooldown(_key: str) -> tuple[float, str]:
        return 0.0, "memory"

    async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
        return None

    session_affinity = MagicMock()
    session_affinity.classify_session_owner_replay_safety_body.return_value = None
    session_affinity.is_replay_safe_session_owner_redispatch_body.return_value = True
    session_affinity.resolve_canonical_session_identity.return_value = None
    session_affinity.get_request_codex_auto_review_parent_session_identity.return_value = None
    session_affinity.build_session_owner_attributes.return_value = {}
    session_affinity.ensure_session_owner_guard_for_request = AsyncMock(
        return_value=SimpleNamespace(
            decision=SimpleNamespace(value="no_session"),
            reservation_token=None,
            held_reservation=False,
            provenance=None,
        )
    )
    session_affinity.get_request_session_owner_lease.return_value = None
    session_affinity.finalize_session_owner_lease_on_success = AsyncMock(
        return_value=None
    )
    session_affinity.finalize_session_owner_lease_on_failure = AsyncMock(
        return_value=None
    )
    session_affinity.run_with_session_owner_lease_renewal = None
    session_affinity.SessionOwnerLeaseRenewalError = ()
    session_affinity.SessionOwnerMutationOutcome = SimpleNamespace(
        CONFLICT="conflict",
        ERROR="error",
        NOT_HELD="not_held",
    )

    admission = SimpleNamespace(
        admit_selected_candidate=AsyncMock(
            return_value=SimpleNamespace(allowed=True, lease=None)
        ),
        release_provider_lane_admission=AsyncMock(return_value=None),
    )

    def _record_failure(**kwargs: Any) -> dict[str, Any]:
        failure_attempts.append(dict(kwargs["attempt_record"]))
        return kwargs["prepared_request_body"]

    monkeypatch.setattr(
        candidate_loop,
        "_admission_mod",
        lambda: admission,
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_failure",
        _record_failure,
    )
    monkeypatch.setattr(
        lpe,
        "_record_auto_agent_alias_attempt_success",
        lambda **kwargs: kwargs["prepared_request_body"],
    )

    monkeypatch.setattr(
        candidate_loop,
        "alias_routing_state",
        AliasRoutingStateManager(),
    )
    monkeypatch.setattr(
        candidate_loop,
        "_session_affinity_mod",
        lambda: session_affinity,
    )
    monkeypatch.setattr(
        lpe,
        "_record_codex_failure_evidence",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        lpe,
        "_plan_codex_oauth_account_failover",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        lpe,
        "execute_cooldown_publication_transaction",
        _execute_publication,
    )

    services = SimpleNamespace(
        select_candidate_fn=_select,
        perform_candidate_request_fn=_perform,
        resolve_cooldown_publication_fn=_resolve_publication,
        publish_cooldown_memory_fn=lambda **_kwargs: None,
        persist_cooldown_fn=_noop_async,
        set_session_affinity_fn=_noop_async,
        add_alias_metadata_fn=lambda body, **_kwargs: body,
        raise_redispatch_fn=None,
    )

    response = await candidate_loop.handle_alias_route(
        services,
        alias_family="codex_auto_agent",
        alias_model="sota-zai",
        request=request,
        prepared_request_body={"model": "sota-zai", "input": "hello"},
        max_candidate_attempts=2,
        get_active_cooldown_state_fn=_active_cooldown,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="no candidates",
        log_label="Codex",
    )

    assert response == {"provider": "alibaba_token_plan", "ok": True}
    assert provider_calls == ["zai_coding_plan", "alibaba_token_plan"]
    assert len(publication_plans) == 1
    assert publication_plans[0].memory_keys == (_ZAI_COOLDOWN_KEY,)
    assert publication_plans[0].durable_keys == (_ZAI_COOLDOWN_KEY,)
    assert len(failure_attempts) == 1
    assert failure_attempts[0]["error_class"] == "candidate_unavailable"
    assert failure_attempts[0]["error_code"] == str(error_code)
    assert failure_attempts[0]["error_type"] == "invalid_request_error"
    assert failure_attempts[0]["error_status_code"] == 429
    assert failure_attempts[0]["cooldown_scope"] == "candidate"


def test_should_keep_half_open_zai_cooldown_recovery() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        classification,
    )

    event = classification.classify_failure(
        status_code=429,
        provider="zai_coding_plan",
        message="model admission unavailable",
    )
    gate = classification.CooldownEvidenceGate(
        base_seconds=1.0,
        max_seconds=8.0,
    )
    decision = gate.record(
        cooldown_key=_ZAI_COOLDOWN_KEY,
        event=event,
        now_monotonic=10.0,
    )

    assert decision.should_cool is True
    assert gate.allow_half_open_probe(
        cooldown_key=_ZAI_COOLDOWN_KEY,
        now_monotonic=decision.cooled_until_monotonic + 0.01,
    )
    assert (
        gate.allow_half_open_probe(
            cooldown_key=_ZAI_COOLDOWN_KEY,
            now_monotonic=decision.cooled_until_monotonic + 0.02,
        )
        is False
    )

    gate.record_probe_result(cooldown_key=_ZAI_COOLDOWN_KEY, success=True)
    assert (
        gate.is_cooled(
            cooldown_key=_ZAI_COOLDOWN_KEY,
            now_monotonic=decision.cooled_until_monotonic + 0.03,
        )
        is False
    )


def test_should_map_coding_plan_missing_key_to_terminal_error() -> None:
    from litellm.exceptions import BadRequestError
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        candidate_loop,
    )

    candidate = {
        "provider": "zai_coding_plan",
        "model": _ADAPTER_MODEL,
        "route_family": _CODEX_ZAI_ROUTE_FAMILY,
    }
    missing_key = ZAICodingPlanAuthenticationError()
    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            missing_key,
            candidate=candidate,
        )
        == "provider_terminal_error"
    )
    wrapped = BadRequestError(
        message="GetLLMProvider Exception - Z.AI Coding Plan authentication requires ZAI_KEY or ZAI_CODING_PLAN_API_KEY. Ordinary ZAI_API_KEY is not reused.",
        model=_ADAPTER_MODEL,
        llm_provider="zai_coding_plan",
    )
    wrapped.__cause__ = missing_key
    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            wrapped,
            candidate=candidate,
        )
        == "provider_terminal_error"
    )
    assert (
        candidate_loop._classify_codex_zai_coding_plan_candidate_failure(
            missing_key,
            candidate={"provider": "openai", "model": "gpt-5"},
        )
        is None
    )
