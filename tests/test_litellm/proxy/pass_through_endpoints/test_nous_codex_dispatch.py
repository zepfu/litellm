"""Stubbed Codex passthrough dispatch for direct Nous inference.

HTTP is stubbed. No live Nous, OpenRouter, or Hermes file reads.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import Response

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
    model_resolution,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_compiler import (
    compile_yaml,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    selection,
    snapshot_select,
)

_ADAPTER_MODEL = "stealth/ox-alpha"
_CODEX_NOUS_ROUTE_FAMILY = "codex_nous_chat_completions_adapter"
_NOUS_CHAT_COMPLETIONS_URL = (
    "https://inference-api.nousresearch.com/v1/chat/completions"
)
_FIXTURE_JWT = "fixture-nous-invoke-jwt-do-not-leak"


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    request.state = MagicMock()
    return request


def _openai_responses_endpoint(endpoint: str) -> bool:
    normalized = endpoint.strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized in {"/v1/responses", "/responses"} or normalized.startswith(
        "/v1/responses/"
    )


def test_should_resolve_prefixed_direct_nous_models_on_responses() -> None:
    assert not hasattr(
        model_resolution, "_resolve_anthropic_nous_completion_adapter_model"
    )
    assert (
        "_resolve_anthropic_nous_completion_adapter_model"
        not in model_resolution._HOST_FUNCTION_NAMES
    )

    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )

    production_resolver = lpe._resolve_codex_nous_chat_completions_adapter_model
    assert production_resolver is not None

    host: dict[str, Any] = {
        "_is_openai_responses_endpoint": _openai_responses_endpoint,
        "_NOUS_PROVIDER": "nous",
    }
    model_resolution.install(host)
    resolve = host["_resolve_codex_nous_chat_completions_adapter_model"]

    assert (
        resolve({"model": "nous/stealth/ox-alpha"}, endpoint="/v1/responses")
        == _ADAPTER_MODEL
    )
    assert (
        resolve(
            {"model": "nous/upstage/solar-pro4:free"},
            endpoint="v1/responses",
        )
        == "upstage/solar-pro4:free"
    )
    assert resolve({"model": "stealth/ox-alpha"}, endpoint="/v1/responses") is None
    assert (
        resolve(
            {"model": "openrouter/stealth/ox-alpha"},
            endpoint="/v1/responses",
        )
        is None
    )
    assert (
        resolve({"model": "nous/stealth/ox-alpha"}, endpoint="/v1/chat/completions")
        is None
    )

    assert production_resolver.__name__ == (
        "_resolve_codex_nous_chat_completions_adapter_model"
    )


@pytest.mark.asyncio
async def test_should_post_to_nous_inference_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.secret_managers import hermes_nous_auth

    handler = getattr(
        codex_candidate_calls,
        "_handle_codex_nous_chat_completions_adapter_route",
        None,
    )
    assert handler is not None, "missing _handle_codex_nous_chat_completions_adapter_route"

    captured: dict[str, Any] = {}

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
        hermes_nous_auth, "load_nous_invoke_jwt", lambda: _FIXTURE_JWT
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    async def _fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        captured["completion"] = kwargs
        return {
            "id": "chatcmpl-nous",
            "object": "chat.completion",
            "created": 1,
            "model": _ADAPTER_MODEL,
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

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    def _validate_outgoing_egress(
        *, url, headers, credential_family, expected_target_family, **kwargs
    ):
        captured["url"] = str(url)
        captured["headers"] = headers
        captured["credential_family"] = credential_family
        captured["expected_target_family"] = expected_target_family
        captured["egress_kwargs"] = kwargs

    host: dict[str, Any] = {
        "__builtins__": __builtins__,
        "litellm": litellm,
        "httpx": httpx,
        "Response": Response,
        "HttpPassThroughEndpointHelpers": MagicMock(
            validate_outgoing_egress=_validate_outgoing_egress
        ),
        "_annotate_request_scope_for_adapted_access_log": MagicMock(),
        "_build_adapted_route_rollup_kwargs": MagicMock(return_value={}),
        "_emit_adapted_route_access_log": MagicMock(),
        "_add_route_family_logging_metadata": lambda body, family: body,
        "_get_proxy_shared_aiohttp_session": lambda: None,
        "BaseOpenAIPassThroughHandler": MagicMock(
            _assemble_headers=MagicMock(
                return_value={"Authorization": f"Bearer {_FIXTURE_JWT}"}
            )
        ),
    }
    codex_candidate_calls.install(host)
    host["HttpPassThroughEndpointHelpers"] = MagicMock(
        validate_outgoing_egress=_validate_outgoing_egress
    )

    bound_handler = host.get(
        "_handle_codex_nous_chat_completions_adapter_route",
        handler,
    )
    try:
        response = await bound_handler(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": f"nous/{_ADAPTER_MODEL}",
                "input": "hello from openai_passthrough",
                "stream": False,
            },
            adapter_model=_ADAPTER_MODEL,
            use_alias_candidate_probe=True,
        )
    finally:
        await GLOBAL_LOGGING_WORKER.stop()
        await litellm.close_litellm_async_clients()

    target_url = captured.get("url") or ""
    completion_kwargs = captured.get("completion") or {}
    api_base = str(completion_kwargs.get("api_base") or "")
    joined = target_url or api_base
    assert _NOUS_CHAT_COMPLETIONS_URL in joined or joined.rstrip("/").endswith(
        "/v1/chat/completions"
    )
    assert "inference-api.nousresearch.com" in joined
    assert completion_kwargs.get("model") == _ADAPTER_MODEL
    assert captured.get("credential_family") == "nous"
    assert captured.get("expected_target_family") == "nous"

    auth_header = ""
    headers = captured.get("headers") or {}
    if isinstance(headers, dict):
        auth_header = str(
            headers.get("Authorization") or headers.get("authorization") or ""
        )
    if not auth_header:
        auth_header = str(completion_kwargs.get("api_key") or "")
    assert "Authorization" in (headers or {}) or auth_header.startswith("Bearer ")
    assert _FIXTURE_JWT in auth_header

    response_body = json.loads(response.body)
    assert response_body["object"] == "response"


@pytest.mark.asyncio
async def test_alias_candidate_provider_handler_dispatches_nous() -> None:
    sentinel = Response(content=b"nous", status_code=200)
    captured: dict[str, Any] = {}

    async def _handle_nous(**kwargs: Any) -> Response:
        captured.update(kwargs)
        return sentinel

    host: dict[str, Any] = {
        "__builtins__": __builtins__,
        "_CODEX_AUTO_AGENT_NOUS_PROVIDER": "nous",
        "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "openrouter",
        "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "opencode_zen",
        "_CODEX_AUTO_AGENT_OPENCODE_GO_PROVIDER": "opencode_go",
        "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "kimi_code",
        "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": "alibaba_token_plan",
        "_CODEX_AUTO_AGENT_ZAI_CODING_PLAN_PROVIDER": "zai_coding_plan",
        "_CODEX_AUTO_AGENT_XAI_PROVIDER": "xai",
        "_CODEX_AUTO_AGENT_COHERE_PROVIDER": "cohere",
        "_CODEX_AUTO_AGENT_CURSOR_AGENT_PROVIDER": "cursor_agent",
        "_handle_codex_openrouter_completion_adapter_route": AsyncMock(
            side_effect=AssertionError("openrouter must not win")
        ),
        "_handle_codex_alibaba_token_plan_adapter_route": AsyncMock(
            side_effect=AssertionError("alibaba must not win")
        ),
        "_handle_codex_kimi_chat_completions_adapter_route": AsyncMock(
            side_effect=AssertionError("kimi must not win")
        ),
        "_handle_codex_opencode_zen_adapter_route": AsyncMock(
            side_effect=AssertionError("opencode zen must not win")
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
    codex_candidate_calls.install(host)
    host["_handle_codex_nous_chat_completions_adapter_route"] = _handle_nous

    result = await host["_perform_codex_auto_agent_alias_candidate_request"](
        endpoint="/v1/responses",
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=MagicMock(),
        candidate={
            "provider": "nous",
            "model": _ADAPTER_MODEL,
            "route_family": _CODEX_NOUS_ROUTE_FAMILY,
        },
        candidate_body={"model": _ADAPTER_MODEL},
        target_url="https://chatgpt.com/backend-api/codex/responses",
        api_key=None,
        forward_headers=False,
    )

    assert result is sentinel
    assert captured["adapter_model"] == _ADAPTER_MODEL
    assert captured["use_alias_candidate_probe"] is True

    with pytest.raises(ValueError, match="codex_nous_chat_completions_adapter"):
        await host["_perform_codex_auto_agent_alias_candidate_request"](
            endpoint="/v1/responses",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={
                "provider": "nous",
                "model": _ADAPTER_MODEL,
                "route_family": "codex_openrouter_completion_adapter",
            },
            candidate_body={"model": _ADAPTER_MODEL},
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=False,
        )


def test_direct_anthropic_nous_model_fails_closed() -> None:
    from litellm.proxy.pass_through_endpoints import (
        llm_passthrough_endpoints as lpe,
    )

    assert not hasattr(lpe, "_resolve_anthropic_nous_completion_adapter_model")
    assert not hasattr(
        model_resolution, "_resolve_anthropic_nous_completion_adapter_model"
    )
    assert (
        "_resolve_anthropic_nous_completion_adapter_model"
        not in model_resolution._HOST_FUNCTION_NAMES
    )

    resolved = None
    handler_cls = getattr(lpe, "BaseAnthropicMessagesPassThroughHandler", None)
    generic_resolver = getattr(handler_cls, "_resolve_anthropic_adapter_model", None)
    if callable(generic_resolver):
        try:
            resolved = generic_resolver(
                {"model": "nous/stealth/ox-alpha"},
                endpoint="/v1/messages",
            )
        except Exception:
            resolved = None
    openrouter_resolver = getattr(
        lpe, "_resolve_anthropic_openrouter_completion_adapter_model", None
    )
    if callable(openrouter_resolver):
        try:
            openrouter_resolved = openrouter_resolver(
                {"model": "nous/stealth/ox-alpha"},
                "/v1/messages",
            )
        except TypeError:
            openrouter_resolved = openrouter_resolver(
                {"model": "nous/stealth/ox-alpha"},
                endpoint="/v1/messages",
            )
        if resolved is None:
            resolved = openrouter_resolved
        elif openrouter_resolved is not None:
            resolved = openrouter_resolved
    assert resolved is None or resolved == "nous/stealth/ox-alpha"
    if isinstance(resolved, str):
        assert not resolved.startswith("openrouter/")
        assert resolved != "openrouter/stealth/ox-alpha"


def test_anthropic_snapshot_omits_codex_only_nous_candidate() -> None:
    raw = """\
defaults: {}
aliases:
  - name: mixed-ox-alpha
    candidates:
      - provider: nous
        model: stealth/ox-alpha
        route_family: codex_nous_chat_completions_adapter
        priority: 97
      - provider: openrouter
        model: openrouter/stealth/ox-alpha
        route_family: codex_openrouter_completion_adapter
        anthropic_route_family: anthropic_openrouter_completion_adapter
        priority: 95
"""
    snapshot = compile_yaml(raw)
    previous = snapshot_select.get_active_routing_snapshot()
    snapshot_select.set_active_routing_snapshot(snapshot)
    try:
        selected = snapshot_select._select_snapshot_candidates(
            "mixed-ox-alpha",
            ingress="anthropic",
        )
    finally:
        snapshot_select.set_active_routing_snapshot(previous)
    providers = [candidate["provider"] for candidate in selected]
    models = [candidate["model"] for candidate in selected]
    assert "nous" not in providers
    assert "stealth/ox-alpha" not in models or all(
        candidate["provider"] != "nous" for candidate in selected
    )
    assert "openrouter" in providers
    assert "openrouter/stealth/ox-alpha" in models


@pytest.mark.asyncio
async def test_capacity_and_upstream_errors_keep_status_and_redact_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.secret_managers import hermes_nous_auth

    handler = getattr(
        codex_candidate_calls,
        "_handle_codex_nous_chat_completions_adapter_route",
        None,
    )
    assert handler is not None

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
        hermes_nous_auth, "load_nous_invoke_jwt", lambda: _FIXTURE_JWT
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    def _install_host() -> Any:
        def _validate_outgoing_egress(**_kwargs: Any) -> None:
            return None

        host: dict[str, Any] = {
            "__builtins__": __builtins__,
            "litellm": litellm,
            "httpx": httpx,
            "Response": Response,
            "HttpPassThroughEndpointHelpers": MagicMock(
                validate_outgoing_egress=_validate_outgoing_egress
            ),
            "_annotate_request_scope_for_adapted_access_log": MagicMock(),
            "_build_adapted_route_rollup_kwargs": MagicMock(return_value={}),
            "_emit_adapted_route_access_log": MagicMock(),
            "_add_route_family_logging_metadata": lambda body, family: body,
            "_get_proxy_shared_aiohttp_session": lambda: None,
            "BaseOpenAIPassThroughHandler": MagicMock(
                _assemble_headers=MagicMock(
                    return_value={"Authorization": f"Bearer {_FIXTURE_JWT}"}
                )
            ),
        }
        codex_candidate_calls.install(host)
        host["HttpPassThroughEndpointHelpers"] = MagicMock(
            validate_outgoing_egress=_validate_outgoing_egress
        )
        return host.get(
            "_handle_codex_nous_chat_completions_adapter_route",
            handler,
        )

    async def _run_status(status_code: int) -> tuple[int, str]:
        async def _fake_acompletion(**kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            error = httpx.HTTPStatusError(
                f"upstream failed access_token={_FIXTURE_JWT}",
                request=httpx.Request(
                    "POST", _NOUS_CHAT_COMPLETIONS_URL
                ),
                response=httpx.Response(
                    status_code,
                    text=f"denied token={_FIXTURE_JWT}",
                    request=httpx.Request("POST", _NOUS_CHAT_COMPLETIONS_URL),
                ),
            )
            error.status_code = status_code  # type: ignore[attr-defined]
            error.detail = f"denied token={_FIXTURE_JWT}"  # type: ignore[attr-defined]
            raise error

        monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
        bound_handler = _install_host()
        try:
            response = await bound_handler(
                endpoint="/v1/responses",
                request=_request(),
                fastapi_response=MagicMock(spec=Response),
                user_api_key_dict=MagicMock(),
                prepared_request_body={
                    "model": f"nous/{_ADAPTER_MODEL}",
                    "input": "hello",
                    "stream": False,
                },
                adapter_model=_ADAPTER_MODEL,
                use_alias_candidate_probe=True,
            )
            body = (
                response.body.decode("utf-8")
                if isinstance(response.body, (bytes, bytearray))
                else str(response.body)
            )
            return int(response.status_code), body
        except Exception as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            detail = str(getattr(exc, "detail", "") or exc)
            if status == 0:
                raise
            return status, detail
        finally:
            await GLOBAL_LOGGING_WORKER.stop()
            await litellm.close_litellm_async_clients()

    for expected_status in (401, 429, 500):
        status, text = await _run_status(expected_status)
        assert status == expected_status
        assert _FIXTURE_JWT not in text
