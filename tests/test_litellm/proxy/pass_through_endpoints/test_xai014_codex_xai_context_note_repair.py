"""XAI-014: repair context-note literal calls on Codex xAI response paths."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints import (
    llm_passthrough_endpoints as lpe,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    BaseOpenAIPassThroughHandler,
)


_ROUTES = {
    "native": {
        "request_model": "xai/grok-4.6",
        "prepared_model": "grok-4.6",
        "prepare_method": "_prepare_openai_grok_native_oauth_context",
        "adapter": "codex_auto_agent_grok_native_responses",
        "adapter_label": "Grok native",
        "upstream_url": "http://localhost:4001/grok/v1/responses",
    },
    "managed": {
        "request_model": "oa_xai/grok-4.6",
        "prepared_model": "grok-4.6",
        "prepare_method": "_prepare_openai_oa_xai_context",
        "adapter": "codex_auto_agent_xai_oauth_responses",
        "adapter_label": "xAI OAuth",
        "upstream_url": "https://api.x.ai/v1/responses",
    },
}


def _request_body(*, stream: bool, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": "Run the advertised command.",
        "stream": stream,
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            }
        ],
    }


def _codex_request() -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.state = SimpleNamespace()
    request.headers = {
        "session_id": "xai014-codex-session",
        "user-agent": "codex-cli/1.0",
        "originator": "codex_cli_rs",
    }
    request.query_params = {}
    request.url = httpx.URL("http://127.0.0.1:4001/openai_passthrough/v1/responses")
    request.scope = {
        "path": "/openai_passthrough/v1/responses",
        "query_string": b"",
    }
    return request


def _context_note_block(*, call_id: str, command: str) -> str:
    return (
        "[Context note - prior assistant step; not an executable tool invocation]\n"
        "Tool label: exec_command\n"
        f"Correlation ref: {call_id}\n"
        f'Input payload: {json.dumps({"cmd": command}, ensure_ascii=False)}'
    )


def _response_body(*, text: str, model: str) -> dict[str, Any]:
    return {
        "id": "resp_xai014_context_note",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def _sse_chunks(response_body: dict[str, Any]):
    async def _chunks():
        yield (
            "event: response.completed\n"
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": response_body,
                }
            )
            + "\n\n"
        ).encode("utf-8")

    return _chunks()


async def _invoke_codex_xai_route(
    route: str,
    *,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    stream: bool,
) -> Any:
    route_config = _ROUTES[route]
    request = _codex_request()
    prepared_body = deepcopy(request_body)
    prepared_body["model"] = route_config["prepared_model"]
    prepared_body["stream"] = stream

    if route == "native":
        prepare_result = (
            "http://localhost:4001/grok",
            {"authorization": "Bearer grok-oidc-token"},
            prepared_body,
            route_config["upstream_url"],
        )
    else:
        prepare_result = (
            "https://api.x.ai/v1",
            "xai-oauth-token",
            prepared_body,
            route_config["upstream_url"],
        )

    async def _pass_through_request(**kwargs: Any) -> Response:
        assert kwargs["custom_body"]["model"] == route_config["prepared_model"]
        assert kwargs["custom_body"]["stream"] is stream
        if stream:
            return StreamingResponse(
                _sse_chunks(response_body),
                media_type="text/event-stream",
            )
        return Response(
            content=json.dumps(response_body),
            media_type="application/json",
        )

    candidate = (
        lpe._perform_codex_auto_agent_grok_native_responses_request
        if route == "native"
        else lpe._perform_codex_auto_agent_oa_xai_responses_request
    )
    with patch.object(
        BaseOpenAIPassThroughHandler,
        route_config["prepare_method"],
        new=AsyncMock(return_value=prepare_result),
    ), patch.object(
        lpe,
        "pass_through_request",
        new=AsyncMock(side_effect=_pass_through_request),
    ), patch.object(
        lpe,
        "_maybe_wrap_xai_passthrough_responses_stream",
        side_effect=lambda response, **_: response,
    ):
        return await candidate(
            endpoint="/v1/responses",
            request=request,
            user_api_key_dict=MagicMock(),
            request_body=request_body,
        )


async def _response_body_from_result(response: Any) -> dict[str, Any]:
    if isinstance(response, StreamingResponse):
        chunks = [
            chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            async for chunk in response.body_iterator
        ]
        rendered = "".join(chunks)
        completed_line = next(
            line
            for line in rendered.splitlines()
            if line.startswith("data: ")
            and '"type": "response.completed"' in line
        )
        return json.loads(completed_line.removeprefix("data: "))["response"]
    return json.loads(response.body)


def _function_calls(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response_body.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "managed"])
@pytest.mark.parametrize("stream", [False, True])
async def test_should_repair_repeated_context_note_blocks_on_both_codex_xai_paths(
    route: str,
    stream: bool,
) -> None:
    route_config = _ROUTES[route]
    repeated_ref = "synthetic-or-unrelated-xai014-ref"
    literal_text = "\n".join(
        _context_note_block(
            call_id=repeated_ref,
            command=command,
        )
        for command in ("pwd", "git status --short", "git diff --check")
    )
    request_body = _request_body(
        stream=stream,
        model=route_config["request_model"],
    )
    response_body = _response_body(
        text=literal_text,
        model=route_config["prepared_model"],
    )

    result = await _invoke_codex_xai_route(
        route,
        request_body=request_body,
        response_body=response_body,
        stream=stream,
    )
    repaired = await _response_body_from_result(result)

    rendered = json.dumps(repaired)
    assert "Context note" not in rendered
    assert "Tool label:" not in rendered
    assert "Correlation ref:" not in rendered
    assert "Input payload:" not in rendered
    calls = _function_calls(repaired)
    assert len(calls) == 3
    assert calls[0]["call_id"] == repeated_ref
    assert calls[1]["call_id"].endswith("_repaired_1")
    assert calls[2]["call_id"].endswith("_repaired_2")
    assert [json.loads(call["arguments"])["cmd"] for call in calls] == [
        "pwd",
        "git status --short",
        "git diff --check",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "managed"])
async def test_should_preserve_interleaved_structured_calls_and_surrounding_text(
    route: str,
) -> None:
    route_config = _ROUTES[route]
    literal_text = (
        "Before the repaired call.\n\n"
        + _context_note_block(
            call_id="synthetic-xai014-interleaved-ref",
            command="printf repaired",
        )
        + "\n\nAfter the repaired call."
    )
    before_call = {
        "type": "function_call",
        "id": "fc_xai014_before",
        "call_id": "call_xai014_before",
        "name": "exec_command",
        "arguments": json.dumps({"cmd": "printf before"}),
    }
    after_call = {
        "type": "function_call",
        "id": "fc_xai014_after",
        "call_id": "call_xai014_after",
        "name": "exec_command",
        "arguments": json.dumps({"cmd": "printf after"}),
    }
    response_body = _response_body(
        text=literal_text,
        model=route_config["prepared_model"],
    )
    response_body["output"] = [before_call, *response_body["output"], after_call]

    result = await _invoke_codex_xai_route(
        route,
        request_body=_request_body(
            stream=False,
            model=route_config["request_model"],
        ),
        response_body=response_body,
        stream=False,
    )
    repaired = await _response_body_from_result(result)

    rendered = json.dumps(repaired)
    assert rendered.count("Before the repaired call.") == 1
    assert rendered.count("After the repaired call.") == 1
    assert "Context note" not in rendered
    assert "Tool label:" not in rendered
    calls = _function_calls(repaired)
    assert [call["call_id"] for call in calls] == [
        "call_xai014_before",
        "synthetic-xai014-interleaved-ref",
        "call_xai014_after",
    ]
    assert [json.loads(call["arguments"])["cmd"] for call in calls] == [
        "printf before",
        "printf repaired",
        "printf after",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "managed"])
@pytest.mark.parametrize(
    ("case_name", "literal_text"),
    [
        (
            "unknown",
            "Tool label: not_advertised\n"
            "Correlation ref: synthetic-xai014-unknown\n"
            'Input payload: {"cmd": "pwd"}',
        ),
        (
            "incomplete",
            "Tool label: exec_command\n"
            "Correlation ref: synthetic-xai014-incomplete\n"
            'Input payload: {"cmd": "pwd"',
        ),
        (
            "invalid_json",
            "Tool label: exec_command\n"
            "Correlation ref: synthetic-xai014-invalid-json\n"
            'Input payload: {"cmd": "pwd",}',
        ),
        (
            "schema_invalid",
            "Tool label: exec_command\n"
            "Correlation ref: synthetic-xai014-schema-invalid\n"
            'Input payload: {"cmd": 123}',
        ),
    ],
)
async def test_should_fail_closed_for_invalid_codex_xai_context_note_blocks(
    route: str,
    case_name: str,
    literal_text: str,
) -> None:
    route_config = _ROUTES[route]
    response_body = _response_body(
        text=literal_text,
        model=route_config["prepared_model"],
    )

    with pytest.raises(ProxyException) as exc_info:
        await _invoke_codex_xai_route(
            route,
            request_body=_request_body(
                stream=False,
                model=route_config["request_model"],
            ),
            response_body=response_body,
            stream=False,
        )

    assert case_name
    assert exc_info.value.detail["error"]["code"] == (
        "aawm_auto_agent_malformed_tool_call_text"
    )
