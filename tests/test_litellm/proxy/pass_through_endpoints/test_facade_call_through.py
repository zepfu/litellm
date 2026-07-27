"""Facade call-through tests: late host-global lookup, dispatch, and adapter contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request, Response
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    anthropic_adapter_calls,
    anthropic_dispatch,
    codex_candidate_calls,
    codex_dispatch,
    tool_call_restore,
)


def test_tool_restore_facade_uses_live_advertised_tool_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def advertised(
        request_body: dict[str, Any] | None,
        *,
        adapter_model: str,
    ) -> set[str]:
        calls.append(adapter_model)
        return set()

    monkeypatch.setattr(
        lpe,
        "_advertised_custom_tool_function_adapter_names",
        advertised,
    )
    body = {"output": [{"type": "function_call", "name": "unchanged"}]}
    restored, count, error = (
        tool_call_restore._restore_adapted_custom_tool_calls_in_response_body(
            body,
            request_body={"tools": []},
            adapter_model="live-model",
        )
    )

    assert restored is body
    assert count == 0
    assert error is None
    assert calls == ["live-model"]


# ---------------------------------------------------------------------------
# Late host-global interception regression
# ---------------------------------------------------------------------------


def test_wave6b_xai_oauth_god_module_patch_reaches_production_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching lpe.get_grok_native_oauth_access_token must intercept the
    actual runtime call after the Wave 6B extraction.

    The configured production runtime holds a late-binding lambda that looks
    up ``get_grok_native_oauth_access_token`` in the god-module globals at call
    time, so a monkeypatch of the compatibility facade reaches
    ``_prepare_grok_native_oauth_passthrough_request`` through the live
    runtime rather than the frozen module-scope import.
    """
    from litellm.proxy.pass_through_endpoints.providers.xai import (
        request_prep as xai_request_prep,
    )

    runtime = xai_request_prep._require_runtime()

    calls: list[str] = []

    async def fake_token() -> str:
        calls.append("token")
        return "patched-oauth-token"

    monkeypatch.setattr(
        lpe, "get_grok_native_oauth_access_token", fake_token
    )

    result = asyncio.run(runtime.get_grok_native_oauth_access_token())

    assert result == "patched-oauth-token"
    assert calls == ["token"]


# ---------------------------------------------------------------------------
# Adapter-call host-global contracts and live dispatch gates
# ---------------------------------------------------------------------------


def test_wave6f_adapter_call_facades_are_same_object() -> None:
    synthetic_candidate_host: dict[str, Any] = {
        "__builtins__": __builtins__,
    }
    synthetic_dispatch_host: dict[str, Any] = {
        "__builtins__": __builtins__,
    }
    codex_candidate_calls.install(synthetic_candidate_host)
    codex_dispatch.install(synthetic_dispatch_host)

    for symbol in anthropic_adapter_calls._EXTRACTED_FUNCTION_NAMES:
        assert getattr(lpe, symbol) is getattr(
            anthropic_adapter_calls,
            symbol,
        )
    for symbol in codex_candidate_calls._HOST_FUNCTION_NAMES:
        assert getattr(lpe, symbol) is getattr(
            codex_candidate_calls,
            symbol,
        )
    assert lpe.try_dispatch_codex_request is (
        codex_dispatch.try_dispatch_codex_request
    )
    assert lpe.try_dispatch_anthropic_adapter is not (
        anthropic_dispatch.try_dispatch_anthropic_adapter
    )


def test_wave6f_codex_dispatch_uses_live_host_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    expected = Response(content=b"codex-dispatched")

    async def handle_alias(**kwargs: Any) -> Response:
        calls.append(kwargs["prepared_request_body"]["model"])
        return expected

    monkeypatch.setattr(
        lpe,
        "_resolve_codex_auto_agent_alias_model",
        lambda body, *, endpoint: "aawm-codex-agent-auto",
    )
    monkeypatch.setattr(
        lpe,
        "_apply_codex_auto_agent_prevention_guidance_to_request_body",
        lambda body: (body, []),
    )
    monkeypatch.setattr(
        lpe,
        "_apply_aawm_read_agent_guidance_to_request_body",
        lambda body, *, alias_model, target_field: (body, []),
    )
    monkeypatch.setattr(
        lpe,
        "_prepare_request_body_for_passthrough_observability",
        lambda *, request, request_body: request_body,
    )
    monkeypatch.setattr(lpe, "_safe_set_request_parsed_body", lambda *args: None)
    monkeypatch.setattr(lpe, "_handle_codex_auto_agent_alias_route", handle_alias)

    body = {"model": "aawm-codex-agent-auto"}
    result = asyncio.run(
        lpe.try_dispatch_codex_request(
            endpoint="/v1/responses",
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/responses",
                    "headers": [],
                }
            ),
            request_body=body,
            prepared_request_body=body,
            fastapi_response=Response(),
            user_api_key_dict=SimpleNamespace(),
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=False,
        )
    )

    assert result is expected
    assert calls == ["aawm-codex-agent-auto"]


def test_wave6f_anthropic_dispatch_uses_live_host_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    expected = Response(content=b"anthropic-adapted")

    async def handle_openai(**kwargs: Any) -> Response:
        calls.append(kwargs["adapter_model"])
        return expected

    monkeypatch.setattr(
        lpe,
        "_resolve_anthropic_xai_oauth_adapter_model",
        lambda body, *, endpoint: None,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_anthropic_grok_native_oauth_adapter_model",
        lambda body, *, endpoint: None,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_anthropic_openai_responses_adapter_model",
        lambda body, *, endpoint: "gpt-5",
    )
    monkeypatch.setattr(
        lpe,
        "_handle_anthropic_openai_responses_adapter_route",
        handle_openai,
    )

    result = asyncio.run(
        lpe.try_dispatch_anthropic_adapter(
            endpoint="/v1/messages",
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/anthropic/v1/messages",
                    "headers": [],
                }
            ),
            fastapi_response=Response(),
            user_api_key_dict=SimpleNamespace(),
            prepared_request_body={"model": "openai/gpt-5"},
        )
    )

    assert result is expected
    assert calls == ["gpt-5"]


def test_wave6f_codex_candidate_facade_uses_live_host_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpectedLookup(Exception):
        pass

    def live_openrouter_key() -> str:
        raise ExpectedLookup("live OpenRouter key lookup")

    monkeypatch.setattr(lpe, "_get_openrouter_api_key", live_openrouter_key)

    with pytest.raises(ExpectedLookup, match="live OpenRouter key lookup"):
        asyncio.run(
            lpe._perform_codex_auto_agent_openrouter_completion_request(
                request=Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/responses",
                        "headers": [],
                    }
                ),
                adapter_model="openai/gpt-5",
                request_body={"model": "openai/gpt-5"},
            )
        )


def test_wave6f_anthropic_execution_facades_use_live_host_namespace() -> None:
    for symbol in (
        "_perform_anthropic_responses_adapter_pass_through",
        "_perform_normalized_anthropic_completion_adapter_stream",
        "_perform_anthropic_completion_adapter_messages_call",
        "_finalize_anthropic_responses_adapter_upstream_response",
        "_finalize_anthropic_responses_adapter_from_config",
        "_finalize_anthropic_completion_adapter_response",
    ):
        facade = getattr(lpe, symbol)
        assert facade is getattr(anthropic_adapter_calls, symbol)
        assert facade.__globals__ is lpe.__dict__
