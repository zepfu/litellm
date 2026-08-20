"""XAI-011: repair literal tool-call text on direct /grok/v1/responses."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers, QueryParams

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    grok_proxy_route,
)

# Reuse the D1-419 / D1-439 / D1-472 dump format from the Codex Grok native
# suite: bounded `Tool label:` / `Correlation ref:` / `Input payload:` blocks.
# Do not invent a second parser or a new transcript shape.


def _search_replace_tool_request_body() -> dict[str, Any]:
    return {
        "model": "grok-build",
        "input": "Apply the four advertised search_replace calls.",
        "stream": False,
        "tools": [
            {
                "type": "function",
                "name": "search_replace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                    "additionalProperties": False,
                },
            }
        ],
    }


def _search_replace_literal_blocks() -> tuple[str, list[str], list[dict[str, str]]]:
    payloads = [
        {
            "file_path": "/tmp/xai011_a.py",
            "old_string": "alpha",
            "new_string": "alpha-fixed",
        },
        {
            "file_path": "/tmp/xai011_b.py",
            "old_string": "beta",
            "new_string": "beta-fixed",
        },
        {
            "file_path": "/tmp/xai011_c.py",
            "old_string": "gamma",
            "new_string": "gamma-fixed",
        },
        {
            "file_path": "/tmp/xai011_d.py",
            "old_string": "delta",
            "new_string": "delta-fixed",
        },
    ]
    call_ids = [
        "call-xai011-search_replace-1",
        "call-xai011-search_replace-2",
        "call-xai011-search_replace-3",
        "call-xai011-search_replace-4",
    ]
    preface = "I'll update the four advertised files now."
    blocks = [preface]
    for call_id, payload in zip(call_ids, payloads):
        blocks.append(
            "[Context note - prior assistant step; not an executable tool invocation]\n"
            "Tool label: search_replace\n"
            f"Correlation ref: {call_id}\n"
            f"Input payload: {json.dumps(payload, ensure_ascii=False)}"
        )
    return "\n".join(blocks), call_ids, payloads


def _literal_tool_response_payload(literal_text: str) -> dict[str, Any]:
    return {
        "id": "resp_xai011_direct_grok",
        "object": "response",
        "status": "completed",
        "model": "grok-build",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": literal_text}],
            }
        ],
    }


def _direct_grok_request(*, endpoint: str = "v1/responses") -> MagicMock:
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = f"http://localhost:4000/grok/{endpoint}"
    request.headers = Headers(
        {
            "authorization": "Bearer oidc-token",
            "x-litellm-api-key": "litellm-test-key",
            "x-xai-token-auth": "xai-grok-cli",
            "x-grok-session-id": "session_xai011",
            "user-agent": "grok-shell/0.2.50",
            "content-type": "application/json",
        }
    )
    request.query_params = QueryParams({})
    request.cookies = {}
    return request


async def _invoke_direct_grok_responses(
    *,
    request_body: dict[str, Any],
    upstream_response: Any,
    endpoint: str = "v1/responses",
) -> Any:
    request = _direct_grok_request(endpoint=endpoint)
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.user_api_key_auth",
        AsyncMock(return_value=MagicMock()),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.get_request_body",
        AsyncMock(return_value=request_body),
    ), patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.pass_through_request",
        AsyncMock(return_value=upstream_response),
    ):
        return await grok_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=MagicMock(spec=Response),
        )


def _function_calls(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    output = response_body.get("output")
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


async def _collect_stream_text(response: StreamingResponse) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


async def _wait_for_intake_records(tmp_path, *, timeout_s: float = 1.0) -> list[dict[str, Any]]:
    log_path = tmp_path / "malformed-error.jsonl"
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if log_path.exists():
            lines = [
                line
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                return [json.loads(line) for line in lines]
        await asyncio.sleep(0.01)
    raise AssertionError(f"malformed-error.jsonl was not written under {tmp_path}")


def _assert_malformed_reject(exc: ProxyException) -> None:
    assert str(exc.code) == "502"
    detail = exc.detail
    assert isinstance(detail, dict)
    error = detail.get("error")
    assert isinstance(error, dict)
    assert error["code"] == "aawm_auto_agent_malformed_tool_call_text"
    failure_kind = detail.get("failure_kind") or error.get("failure_kind")
    assert failure_kind == "malformed_tool_call"


def _enable_malformed_intake(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")


@pytest.mark.asyncio
async def test_direct_grok_nonstream_repairs_four_advertised_search_replace_blocks():
    literal_text, call_ids, payloads = _search_replace_literal_blocks()
    upstream = Response(
        content=json.dumps(_literal_tool_response_payload(literal_text)),
        media_type="application/json",
    )

    response = await _invoke_direct_grok_responses(
        request_body=_search_replace_tool_request_body(),
        upstream_response=upstream,
    )

    assert isinstance(response, Response)
    assert not isinstance(response, StreamingResponse)
    repaired = json.loads(response.body)
    rendered = json.dumps(repaired)
    assert repaired["id"] == "resp_xai011_direct_grok"
    assert "I'll update the four advertised files now." in rendered
    assert "Tool label:" not in rendered
    assert "Input payload:" not in rendered
    function_calls = _function_calls(repaired)
    assert len(function_calls) == 4
    assert [item["name"] for item in function_calls] == ["search_replace"] * 4
    assert [item["call_id"] for item in function_calls] == call_ids
    assert [json.loads(item["arguments"]) for item in function_calls] == payloads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("literal_text", "tools"),
    [
        (
            "Tool label: not_advertised_tool\n"
            "Correlation ref: call-xai011-unknown\n"
            'Input payload: {"file_path": "/tmp/x.py", "old_string": "a", "new_string": "b"}',
            _search_replace_tool_request_body()["tools"],
        ),
        (
            "Tool label: search_replace\n"
            "Correlation ref: call-xai011-invalid-json\n"
            'Input payload: {"file_path": "/tmp/x.py"',
            _search_replace_tool_request_body()["tools"],
        ),
        (
            "Tool label: search_replace\n"
            "Correlation ref: call-xai011-schema-invalid\n"
            'Input payload: {"file_path": "/tmp/x.py", "old_string": "a"}',
            _search_replace_tool_request_body()["tools"],
        ),
    ],
)
async def test_direct_grok_nonstream_malformed_literal_blocks_return_502_and_intake(
    monkeypatch,
    tmp_path,
    literal_text,
    tools,
):
    _enable_malformed_intake(monkeypatch, tmp_path)
    request_body = _search_replace_tool_request_body()
    request_body["tools"] = tools
    upstream = Response(
        content=json.dumps(_literal_tool_response_payload(literal_text)),
        media_type="application/json",
    )

    with pytest.raises(ProxyException) as exc_info:
        await _invoke_direct_grok_responses(
            request_body=request_body,
            upstream_response=upstream,
        )

    _assert_malformed_reject(exc_info.value)
    records = await _wait_for_intake_records(tmp_path)
    assert records
    record = records[0]
    assert record["failure_kind"] == "malformed_tool_call"
    assert record["error_code"] == "aawm_auto_agent_malformed_tool_call_text"
    assert record["adapter"] == "direct_grok_responses"
    assert record["terminal_outcome"] == "malformed_tool_call_rejected"
    assert record.get("redispatch_required") is False


@pytest.mark.asyncio
async def test_direct_grok_nonstream_quoted_tool_label_prose_without_payload_is_unchanged():
    prose = (
        'The docs may mention a "Tool label:" heading as an example, '
        "but this turn is ordinary assistant prose with no payload block."
    )
    payload = _literal_tool_response_payload(prose)
    upstream = Response(content=json.dumps(payload), media_type="application/json")

    response = await _invoke_direct_grok_responses(
        request_body=_search_replace_tool_request_body(),
        upstream_response=upstream,
    )

    assert isinstance(response, Response)
    assert json.loads(response.body) == payload
    assert response is upstream


@pytest.mark.asyncio
async def test_direct_grok_stream_without_literal_marker_forwards_original_sse():
    original_chunks = [
        (
            "event: response.output_text.delta\n"
            'data: {"type": "response.output_text.delta", "delta": "hello from grok"}\n\n'
        ).encode("utf-8"),
        (
            "event: response.completed\n"
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_xai011_stream_ok",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "hello from grok",
                                    }
                                ],
                            }
                        ],
                    },
                }
            )
            + "\n\n"
        ).encode("utf-8"),
    ]

    async def _chunks():
        for chunk in original_chunks:
            yield chunk

    upstream = StreamingResponse(_chunks(), media_type="text/event-stream")
    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True

    response = await _invoke_direct_grok_responses(
        request_body=request_body,
        upstream_response=upstream,
    )

    assert isinstance(response, StreamingResponse)
    rendered = await _collect_stream_text(response)
    assert rendered == b"".join(original_chunks).decode("utf-8")
    assert "Tool label:" not in rendered
    assert '"type": "function_call"' not in rendered


@pytest.mark.asyncio
async def test_direct_grok_stream_repairs_literal_marker_into_function_call_events():
    literal_text, call_ids, payloads = _search_replace_literal_blocks()
    response_body = _literal_tool_response_payload(literal_text)

    async def _chunks():
        yield (
            "event: response.completed\n"
            + "data: "
            + json.dumps({"type": "response.completed", "response": response_body})
            + "\n\n"
        ).encode("utf-8")

    upstream = StreamingResponse(_chunks(), media_type="text/event-stream")
    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True

    response = await _invoke_direct_grok_responses(
        request_body=request_body,
        upstream_response=upstream,
    )

    assert isinstance(response, StreamingResponse)
    rendered = await _collect_stream_text(response)
    assert "Tool label:" not in rendered
    assert "Input payload:" not in rendered
    assert "event: response.output_item.added" in rendered
    assert '"type": "function_call"' in rendered
    assert '"name": "search_replace"' in rendered
    assert "event: response.function_call_arguments.done" in rendered
    assert "event: response.completed" in rendered
    completed_line = next(
        line
        for line in rendered.splitlines()
        if line.startswith('data: {"type": "response.completed"')
    )
    completed_event = json.loads(completed_line.removeprefix("data: "))
    function_calls = _function_calls(completed_event["response"])
    assert len(function_calls) == 4
    assert [item["call_id"] for item in function_calls] == call_ids
    assert [json.loads(item["arguments"]) for item in function_calls] == payloads
    assert "I'll update the four advertised files now." in json.dumps(
        completed_event["response"]
    )


@pytest.mark.asyncio
async def test_direct_grok_stream_incomplete_literal_marker_at_ceiling_returns_502(
    monkeypatch,
    tmp_path,
):
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

    _enable_malformed_intake(monkeypatch, tmp_path)
    monkeypatch.setattr(lpe, "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS", 2)

    marker_chunk = (
        "event: response.output_text.delta\n"
        'data: {"type": "response.output_text.delta", '
        '"delta": "Tool label: search_replace\\nCorrelation ref: call-xai011-partial\\n"}\n\n'
    ).encode("utf-8")
    filler_chunk = (
        "event: response.output_text.delta\n"
        'data: {"type": "response.output_text.delta", "delta": "still generating"}\n\n'
    ).encode("utf-8")

    async def _chunks():
        yield marker_chunk
        for _ in range(6):
            yield filler_chunk

    upstream = StreamingResponse(_chunks(), media_type="text/event-stream")
    request_body = _search_replace_tool_request_body()
    request_body["stream"] = True

    with pytest.raises(ProxyException) as exc_info:
        response = await _invoke_direct_grok_responses(
            request_body=request_body,
            upstream_response=upstream,
        )
        if isinstance(response, StreamingResponse):
            await _collect_stream_text(response)

    _assert_malformed_reject(exc_info.value)
    records = await _wait_for_intake_records(tmp_path)
    assert records
    assert records[0]["failure_kind"] == "malformed_tool_call"
    assert records[0]["adapter"] == "direct_grok_responses"
