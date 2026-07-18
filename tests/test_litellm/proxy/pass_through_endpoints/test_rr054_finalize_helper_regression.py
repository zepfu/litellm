"""RR-054 finalize-helper regression: malformed-intake context naming.

All Anthropic←Responses finalization paths must pass the shared builder
``_build_anthropic_response_from_responses_response`` the keyword
``malformed_intake_context=...`` built by
``_build_malformed_intake_context_for_anthropic_responses_adapter``.

This module is intentionally production-read-only: it documents the current
finalize contract and fails if production diverges via a typo'd helper name
or a typo'd keyword argument (historically ``malformed_inplace_context`` /
``_build_malformed_inplace_context_for_anthropic_responses_adapter``).
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request, Response
from starlette.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    responses_finalize,
)

CORRECT_HELPER_NAME = (
    "_build_malformed_intake_context_for_anthropic_responses_adapter"
)
TYPO_HELPER_NAME = (
    "_build_malformed_inplace_context_for_anthropic_responses_adapter"
)
CORRECT_KWARG = "malformed_intake_context"
TYPO_KWARG = "malformed_inplace_context"

RESPONSES_ADAPTERS: list[tuple[str, adapter_config.AnthropicResponsesAdapterConfig]] = [
    ("openai_responses", adapter_config.OPENAI_RESPONSES),
    ("xai_oauth_responses", adapter_config.XAI_OAUTH_RESPONSES),
    ("grok_native_responses", adapter_config.GROK_NATIVE_RESPONSES),
    ("openrouter_responses", adapter_config.OPENROUTER_RESPONSES),
    ("opencode_zen_responses", adapter_config.OPENCODE_ZEN_RESPONSES),
]


def _minimal_request(path: str = "/v1/messages") -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    return Request(scope)


def _non_empty_responses_body(*, model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "resp_ok",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
                "status": "completed",
            }
        ],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }


def _finalize_source() -> str:
    return inspect.getsource(
        responses_finalize.finalize_anthropic_responses_adapter_upstream_response
    )


def test_rr054_finalize_helper_name_is_defined_and_not_typo() -> None:
    """Canonical helper must exist; the historic typo name must not."""
    assert hasattr(lpe, CORRECT_HELPER_NAME)
    assert not hasattr(lpe, TYPO_HELPER_NAME)
    assert callable(getattr(lpe, CORRECT_HELPER_NAME))


def test_rr054_finalize_paths_call_correct_malformed_intake_helper() -> None:
    """Shared finalize body must invoke the real helper on every builder path."""
    source = _finalize_source()
    assert TYPO_HELPER_NAME not in source
    assert "runtime.build_malformed_context" in source
    assert source.count(f"{CORRECT_KWARG}=") >= 2


def test_rr054_finalize_builder_accepts_only_malformed_intake_kwarg() -> None:
    """Builder contract: only ``malformed_intake_context`` is a real parameter."""
    params = inspect.signature(
        lpe._build_anthropic_response_from_responses_response
    ).parameters
    assert CORRECT_KWARG in params
    assert TYPO_KWARG not in params


def test_rr054_finalize_does_not_pass_typo_kwarg_to_builder() -> None:
    """Non-stream finalize must not pass ``malformed_inplace_context=``."""
    source = _finalize_source()
    assert f"{TYPO_KWARG}=" not in source, (
        "RR-054 defect: non-stream finalize still passes "
        f"`{TYPO_KWARG}=` to "
        "`_build_anthropic_response_from_responses_response`, which only accepts "
        f"`{CORRECT_KWARG}=`. This raises TypeError on every non-stream Responses "
        "adapter finalization path."
    )
    assert f"{CORRECT_KWARG}=" in source


@pytest.mark.parametrize(
    "adapter_name,config",
    RESPONSES_ADAPTERS,
    ids=[name for name, _ in RESPONSES_ADAPTERS],
)
@pytest.mark.asyncio
async def test_rr054_finalize_nonstream_uses_malformed_intake_helper(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    """Config-driven non-stream finalize must call the helper and correct kwarg.

    Fails on production while the non-stream branch still supplies
    ``malformed_inplace_context=`` (unexpected keyword) instead of
    ``malformed_intake_context=``.
    """
    request = _minimal_request()
    body = _non_empty_responses_body(model="m")
    upstream = Response(
        content=json.dumps(body),
        media_type="application/json",
        status_code=200,
        headers={"x-upstream": "yes", "content-length": "999"},
    )
    captured: dict[str, Any] = {}

    def _capture_builder(
        response_body: dict[str, Any],
        *,
        reject_empty_success: bool = False,
        diagnostic_context: Any = None,
        use_codex_native_tools: bool = False,
        retryable_failed_response: bool = False,
        failed_response_adapter_model: Any = None,
        failed_response_adapter: str = "anthropic_responses_adapter",
        failed_response_adapter_label: str = "Responses",
        malformed_intake_context: Any = None,
    ) -> Response:
        # Signature mirrors production so a typo'd keyword raises TypeError.
        captured["kwargs"] = {
            "reject_empty_success": reject_empty_success,
            "diagnostic_context": diagnostic_context,
            "use_codex_native_tools": use_codex_native_tools,
            "retryable_failed_response": retryable_failed_response,
            "failed_response_adapter_model": failed_response_adapter_model,
            "failed_response_adapter": failed_response_adapter,
            "failed_response_adapter_label": failed_response_adapter_label,
            "malformed_intake_context": malformed_intake_context,
        }
        captured["response_body"] = response_body
        return Response(content=b'{"type":"message"}', media_type="application/json")

    real_helper = getattr(lpe, CORRECT_HELPER_NAME)

    with patch.object(
        lpe,
        "_build_anthropic_response_from_responses_response",
        side_effect=_capture_builder,
    ), patch.object(
        lpe,
        CORRECT_HELPER_NAME,
        wraps=real_helper,
    ) as helper_spy:
        try:
            result = await lpe._finalize_anthropic_responses_adapter_from_config(
                config=config,
                upstream_response=upstream,
                request=request,
                translated_request_body={"model": "m", "stream": False},
                adapter_model="m",
                target_url="https://example.test/v1/responses",
                client_requested_stream=False,
                use_alias_candidate_probe=False,
            )
        except TypeError as exc:
            message = str(exc)
            if TYPO_KWARG in message or "unexpected keyword" in message:
                pytest.fail(
                    f"RR-054 defect on adapter={adapter_name}: non-stream finalize "
                    f"raised TypeError from typo'd malformed-intake kwarg: {message}"
                )
            raise

    assert isinstance(result, Response)
    assert helper_spy.called, (
        f"adapter={adapter_name}: expected {CORRECT_HELPER_NAME} to be invoked"
    )
    assert CORRECT_KWARG in captured["kwargs"], (
        f"adapter={adapter_name}: builder missing {CORRECT_KWARG}; "
        f"got keys={sorted(k for k, v in captured['kwargs'].items() if v is not None or k == CORRECT_KWARG)}"
    )
    assert captured["kwargs"][CORRECT_KWARG] is not None
    assert isinstance(captured["kwargs"][CORRECT_KWARG], dict)
    assert result.headers.get("x-upstream") == "yes"
    assert result.headers.get("content-length") != "999"


@pytest.mark.parametrize(
    "adapter_name,config",
    RESPONSES_ADAPTERS,
    ids=[name for name, _ in RESPONSES_ADAPTERS],
)
@pytest.mark.asyncio
async def test_rr054_finalize_stream_collapse_uses_malformed_intake_helper(
    adapter_name: str,
    config: adapter_config.AnthropicResponsesAdapterConfig,
) -> None:
    """Streaming upstream + client_requested_stream=False uses the correct helper."""
    request = _minimal_request()
    body = _non_empty_responses_body(model="m")

    async def _sse_chunks():
        yield b'data: {"type":"response.completed","response":'
        yield json.dumps(body).encode("utf-8")
        yield b"}\n\n"

    upstream = StreamingResponse(
        _sse_chunks(),
        media_type="text/event-stream",
        status_code=201,
        headers={"x-upstream": "stream"},
    )
    captured: dict[str, Any] = {}

    def _capture_builder(response_body: dict[str, Any], **kwargs: Any) -> Response:
        captured["kwargs"] = dict(kwargs)
        return Response(content=b'{"type":"message"}', media_type="application/json")

    real_helper = getattr(lpe, CORRECT_HELPER_NAME)

    with patch.object(
        lpe,
        "_validate_alias_candidate_responses_stream_if_needed",
        new=AsyncMock(side_effect=lambda resp, **kwargs: resp),
    ), patch.object(
        lpe,
        "_collect_responses_response_from_stream",
        new=AsyncMock(return_value=body),
    ), patch.object(
        lpe,
        "_build_anthropic_response_from_responses_response",
        side_effect=_capture_builder,
    ), patch.object(
        lpe,
        CORRECT_HELPER_NAME,
        wraps=real_helper,
    ) as helper_spy:
        result = await lpe._finalize_anthropic_responses_adapter_from_config(
            config=config,
            upstream_response=upstream,
            request=request,
            translated_request_body={"model": "m", "stream": False},
            adapter_model="m",
            target_url="https://example.test/v1/responses",
            client_requested_stream=False,
            use_alias_candidate_probe=False,
        )

    assert isinstance(result, Response)
    assert not isinstance(result, StreamingResponse)
    assert helper_spy.called
    assert CORRECT_KWARG in captured["kwargs"]
    assert TYPO_KWARG not in captured["kwargs"]
    assert result.status_code == 201
