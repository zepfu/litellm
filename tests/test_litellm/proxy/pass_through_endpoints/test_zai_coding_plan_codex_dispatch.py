"""Stubbed Codex passthrough dispatch for Z.AI Coding Plan.

Proves POST /openai_passthrough/v1/responses with
model=zai_coding_plan/glm-5.3 hits the coding chat URL. HTTP is stubbed;
this is not live Z.AI fanout.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

from litellm.llms.zai_coding_plan.chat.transformation import (
    ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
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
async def test_should_post_codex_responses_direct_model_to_coding_chat_url(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock,
) -> None:
    import litellm
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

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
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
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
    assert "sota-zai" not in upstream_request.content.decode()
    assert "zai_coding_plan/" not in upstream_request.content.decode()
    assert upstream_request.headers["Authorization"] == "Bearer coding-plan-key"
    response_body = json.loads(response.body)
    assert response_body["object"] == "response"
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
        (1211, "candidate_unavailable"),
        (1311, "candidate_unavailable"),
        (1308, "usage_limit_reached"),
        (1309, "usage_limit_reached"),
        (1310, "usage_limit_reached"),
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
