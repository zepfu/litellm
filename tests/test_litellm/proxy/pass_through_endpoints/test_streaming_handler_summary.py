import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
    _PassThroughStreamLineAccumulator,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


def _responses_sse_event(payload: dict) -> bytes:
    return (
        f"event: {payload['type']}\ndata: "
        + json.dumps(payload, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


def test_stream_line_accumulator_matches_raw_bytes_converter():
    raw_bytes = [
        b'data: {"type":"message_start"}\n\n',
        b'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
    ]
    accumulator = _PassThroughStreamLineAccumulator()
    for chunk in raw_bytes:
        accumulator.feed(chunk)
    incremental = accumulator.finish()
    rebuilt = PassThroughStreamingHandler._convert_raw_bytes_to_str_lines(raw_bytes)
    assert incremental == rebuilt


def test_stream_summary_finalize_eligible_openai_responses_only():
    env_key = PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV
    with patch.dict(os.environ, {env_key: "1"}, clear=False):
        assert PassThroughStreamingHandler._stream_summary_first_finalize_eligible(
            endpoint_type=EndpointType.OPENAI,
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
        )
        assert not PassThroughStreamingHandler._stream_summary_first_finalize_eligible(
            endpoint_type=EndpointType.OPENAI,
            url_route="https://api.openai.com/v1/chat/completions",
            custom_llm_provider="openai",
        )
        assert PassThroughStreamingHandler._stream_summary_first_finalize_eligible(
            endpoint_type=EndpointType.ANTHROPIC,
            url_route="https://api.anthropic.com/v1/messages",
            custom_llm_provider=None,
        )


def test_resolve_stream_logging_lines_defaults_to_raw_bytes_rebuild(monkeypatch):
    monkeypatch.delenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        raising=False,
    )
    raw_bytes = [b'data: {"ok": true}\n\n']
    success_handler_kwargs = {"litellm_params": {"metadata": {}}}
    lines = PassThroughStreamingHandler._resolve_stream_logging_lines(
        raw_bytes=raw_bytes,
        precomputed_lines=None,
        endpoint_type=EndpointType.ANTHROPIC,
        url_route="https://api.anthropic.com/v1/messages",
        custom_llm_provider=None,
        success_handler_kwargs=success_handler_kwargs,
    )
    assert lines == ['data: {"ok": true}']
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_finalize_line_source"
        ]
        == "raw_bytes_rebuild"
    )


def test_resolve_stream_logging_lines_uses_incremental_summary_when_provided():
    raw_bytes = [b"ignored\n"]
    precomputed = ["data: {\"from\": \"summary\"}"]
    success_handler_kwargs = {"litellm_params": {"metadata": {}}}
    lines = PassThroughStreamingHandler._resolve_stream_logging_lines(
        raw_bytes=raw_bytes,
        precomputed_lines=precomputed,
        endpoint_type=EndpointType.ANTHROPIC,
        url_route="https://api.anthropic.com/v1/messages",
        custom_llm_provider=None,
        success_handler_kwargs=success_handler_kwargs,
    )
    assert lines == precomputed
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_finalize_line_source"
        ]
        == "incremental_summary"
    )


@pytest.mark.asyncio
async def test_route_streaming_logging_uses_precomputed_lines(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = {
        "litellm_params": {"metadata": {"aawm_stream_emit_gap_ms": 1.0}},
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }
    captured: dict = {}

    def _capture(**kwargs):
        captured["all_chunks"] = kwargs.get("all_chunks")
        return {"result": {"response": "ok"}, "kwargs": {}}

    completed_event = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [],
        },
    }
    precomputed_lines = [
        'data: {"type":"response.output_text.delta","output_index":0,'
        '"delta":"{\\"outcome\\":\\"allow\\"}"}',
        f"data: {json.dumps(completed_event)}",
    ]

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks",
        side_effect=_capture,
    ), patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup_turn"
    ) as record_turn:
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            response=httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://chatgpt.com/backend-api/codex/responses",
                ),
            ),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4"},
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now() - timedelta(milliseconds=10),
            raw_bytes=[],
            precomputed_lines=precomputed_lines,
            end_time=datetime.now(),
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        )

    assert captured["all_chunks"] == precomputed_lines
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_finalize_line_source"
        ]
        == "incremental_summary"
    )
    reconstructed_response = record_turn.call_args.kwargs["response_body"]
    assert reconstructed_response["status"] == "completed"
    assert reconstructed_response["output"] == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": '{"outcome":"allow"}',
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_route_streaming_logging_syncs_codex_review_decision_event_to_callbacks(
    monkeypatch,
):
    """D1-616: a parsed codex-auto-review decision attached by the route rollup
    must be visible on litellm_logging_obj.model_call_details before success
    callbacks run."""
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "0")
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.async_success_handler = AsyncMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "aawm_route_rollup_context": {
                    "is_codex_auto_review": True,
                    "litellm_call_id": "reviewer-call-1",
                    "canonical_session_identity": "session-1",
                    "review_originating_litellm_call_id": "origin-call-1",
                    "review_parent_actor_id": "parent-actor-1",
                    "review_parent_thread_id": "parent-thread-1",
                }
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }

    def _capture(**kwargs):
        return {"result": {"response": "ok"}, "kwargs": {}}

    decision_text = (
        '{"outcome":"allow","rationale":"safe read-only review",'
        '"risk_level":"low","user_authorization":"high"}'
    )
    message_item = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": decision_text}],
        },
    }
    completed_event = {
        "type": "response.completed",
        "response": {"status": "completed", "output": []},
    }
    precomputed_lines = [
        f"data: {json.dumps(message_item)}",
        f"data: {json.dumps(completed_event)}",
    ]

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks",
        side_effect=_capture,
    ):
        await PassThroughStreamingHandler._route_streaming_logging_to_handler(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            response=httpx.Response(
                200,
                request=httpx.Request(
                    "POST",
                    "https://chatgpt.com/backend-api/codex/responses",
                ),
            ),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "codex-auto-review"},
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now() - timedelta(milliseconds=10),
            raw_bytes=[],
            precomputed_lines=precomputed_lines,
            end_time=datetime.now(),
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        )

    events = logging_obj.model_call_details.get(
        "_aawm_parsed_codex_review_decisions"
    )
    assert isinstance(events, list)
    assert len(events) == 1
    assert events[0]["outcome"] == "allow"
    assert events[0]["reviewer_litellm_call_id"] == "reviewer-call-1"
    assert events[0]["session_id"] == "session-1"
    assert events[0]["parent_litellm_call_id"] == "origin-call-1"
    assert events[0]["parent_agent_id"] == "parent-actor-1"
    assert events[0]["parent_thread_id"] == "parent-thread-1"


@pytest.mark.asyncio
async def test_chunk_processor_increments_summary_lines_for_codex_responses(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )

    async def _aiter_bytes():
        yield b'data: {"delta":1}\n\n'
        yield b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        yield b"data: [DONE]\n\n"

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes

    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    route_kwargs: dict = {}

    async def _capture_route(**kwargs):
        route_kwargs.update(kwargs)

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        side_effect=_capture_route,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs={"litellm_params": {"metadata": {}}},
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    assert chunks == [
        b'data: {"delta":1}\n\n',
        b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert route_kwargs["precomputed_lines"] == [
        'data: {"delta":1}',
        'data: {"type":"response.completed","response":{"status":"completed"}}',
        "data: [DONE]",
    ]



def test_should_buffer_raw_stream_bytes_skips_when_summary_and_capture_off(monkeypatch):
    monkeypatch.delenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", raising=False)
    monkeypatch.delenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE", raising=False)
    monkeypatch.delenv(
        "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_CONTROL_FILE", raising=False
    )
    assert PassThroughStreamingHandler._should_buffer_raw_stream_bytes(
        line_accumulator_enabled=True
    ) is False
    assert PassThroughStreamingHandler._should_buffer_raw_stream_bytes(
        line_accumulator_enabled=False
    ) is True


def test_should_buffer_raw_stream_bytes_keeps_raw_when_full_payload_capture_on(
    monkeypatch,
):
    monkeypatch.setenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", "1")
    assert PassThroughStreamingHandler._should_buffer_raw_stream_bytes(
        line_accumulator_enabled=True
    ) is True


def test_should_buffer_raw_stream_bytes_keeps_raw_when_diagnostic_capture_on(
    monkeypatch,
):
    monkeypatch.setenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE", "1")
    monkeypatch.setenv(
        "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ROUTE_FAMILIES",
        "openai_responses",
    )
    assert PassThroughStreamingHandler._should_buffer_raw_stream_bytes(
        line_accumulator_enabled=True
    ) is True


@pytest.mark.asyncio
async def test_chunk_processor_skips_raw_bytes_when_summary_finalize_and_capture_off(
    monkeypatch,
):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    monkeypatch.delenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", raising=False)
    monkeypatch.delenv("AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE", raising=False)
    monkeypatch.delenv(
        "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_CONTROL_FILE", raising=False
    )

    async def _aiter_bytes():
        yield b'data: {"delta":1}\n\n'
        yield b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        yield b"data: [DONE]\n\n"

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes

    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    route_kwargs: dict = {}

    async def _capture_route(**kwargs):
        route_kwargs.update(kwargs)

    success_handler_kwargs = {"litellm_params": {"metadata": {}}}
    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        side_effect=_capture_route,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    assert chunks == [
        b'data: {"delta":1}\n\n',
        b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert route_kwargs["raw_bytes"] == []
    assert route_kwargs["precomputed_lines"] == [
        'data: {"delta":1}',
        'data: {"type":"response.completed","response":{"status":"completed"}}',
        "data: [DONE]",
    ]
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_raw_bytes_buffered"
        ]
        is False
    )


@pytest.mark.asyncio
async def test_chunk_processor_keeps_raw_bytes_when_summary_finalize_and_capture_on(
    monkeypatch,
):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    monkeypatch.setenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", "1")

    async def _aiter_bytes():
        yield b'data: {"delta":1}\n\n'
        yield b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        yield b"data: [DONE]\n\n"

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes

    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    route_kwargs: dict = {}

    async def _capture_route(**kwargs):
        route_kwargs.update(kwargs)

    success_handler_kwargs = {"litellm_params": {"metadata": {}}}
    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        side_effect=_capture_route,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    assert chunks == [
        b'data: {"delta":1}\n\n',
        b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert route_kwargs["raw_bytes"] == [
        b'data: {"delta":1}\n\n',
        b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert route_kwargs["precomputed_lines"] == [
        'data: {"delta":1}',
        'data: {"type":"response.completed","response":{"status":"completed"}}',
        "data: [DONE]",
    ]
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_raw_bytes_buffered"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_chunk_processor_synthesizes_completed_for_complete_text(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    provider_chunks = [
        _responses_sse_event(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-complete",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                },
            }
        ),
        _responses_sse_event(
            {
                "type": "response.output_text.done",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "complete text",
            }
        ),
    ]

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    logging_obj.async_failure_handler = AsyncMock()
    route_handler = AsyncMock()
    success_handler_kwargs = {"litellm_params": {"metadata": {}}}

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        route_handler,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    terminal_payload = json.loads(
        chunks[-2].split(b"\ndata: ", 1)[1].removesuffix(b"\n\n")
    )
    assert chunks[: len(provider_chunks)] == provider_chunks
    assert terminal_payload["type"] == "response.completed"
    assert terminal_payload["response"]["status"] == "completed"
    assert b"".join(chunks).count(b"data: [DONE]") == 1
    tracker_metadata = success_handler_kwargs["litellm_params"]["metadata"][
        "aawm_stream_tracker_state"
    ]
    assert tracker_metadata["final_assistant_output"] is True
    assert tracker_metadata["provider_terminal_observed"] is False
    assert tracker_metadata["synthetic_terminal_event_type"] == "response.completed"
    route_handler.assert_awaited_once()
    logging_obj.async_failure_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunk_processor_synthesizes_incomplete_for_open_state(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    provider_chunks = [
        _responses_sse_event(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-open",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                },
            }
        ),
        _responses_sse_event(
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "id": "msg_open",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        ),
    ]

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    logging_obj.async_failure_handler = AsyncMock()
    route_handler = AsyncMock()
    success_handler_kwargs = {"litellm_params": {"metadata": {}}}

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        route_handler,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    terminal_payload = json.loads(
        chunks[-2].split(b"\ndata: ", 1)[1].removesuffix(b"\n\n")
    )
    assert chunks[: len(provider_chunks)] == provider_chunks
    assert terminal_payload["type"] == "response.incomplete"
    assert terminal_payload["response"]["incomplete_details"]["reason"] == (
        "upstream_stream_incomplete_item_state"
    )
    tracker_metadata = success_handler_kwargs["litellm_params"]["metadata"][
        "aawm_stream_tracker_state"
    ]
    assert tracker_metadata["open_state_counts"]["output_items"] == 1
    assert tracker_metadata["synthetic_terminal_event_type"] == "response.incomplete"
    route_handler.assert_awaited_once()
    logging_obj.async_failure_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunk_processor_marks_split_partial_frame_incomplete(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    provider_chunks = [
        _responses_sse_event(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-partial-frame",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                },
            }
        ),
        _responses_sse_event(
            {
                "type": "response.output_text.done",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "complete text",
            }
        ),
        b'data: {"type":"response.output_text.delta"',
    ]

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    route_handler = AsyncMock()
    success_handler_kwargs = {"litellm_params": {"metadata": {}}}

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        route_handler,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    terminal_payload = json.loads(
        chunks[-2].split(b"\ndata: ", 1)[1].removesuffix(b"\n\n")
    )
    assert chunks[: len(provider_chunks)] == provider_chunks
    assert terminal_payload["response"]["incomplete_details"]["reason"] == (
        "upstream_stream_partial_frame"
    )
    tracker_metadata = success_handler_kwargs["litellm_params"]["metadata"][
        "aawm_stream_tracker_state"
    ]
    assert tracker_metadata["partial_frame"] is True
    route_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_chunk_processor_deduplicates_done_for_synthetic_terminal(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    provider_chunks = [
        _responses_sse_event(
            {
                "type": "response.created",
                "response": {
                    "id": "resp-done",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                },
            }
        ),
        _responses_sse_event(
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "complete text",
            }
        ),
        b"data: [DONE]\n\n",
    ]

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    route_handler = AsyncMock()

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        route_handler,
    ):
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs={"litellm_params": {"metadata": {}}},
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    combined = b"".join(chunks)
    assert chunks[:2] == provider_chunks[:2]
    assert combined.count(b"data: [DONE]") == 1
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert b'"type":"response.completed"' in chunks[-2]
    route_handler.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_chunks", "expected_terminal_type", "expected_route_status"),
    [
        (
            [
                _responses_sse_event(
                    {
                        "type": "response.created",
                        "response": {
                            "id": "resp-success-complete",
                            "model": "gpt-5.4",
                            "status": "in_progress",
                        },
                    }
                ),
                _responses_sse_event(
                    {
                        "type": "response.output_text.done",
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "text": "complete text",
                    }
                ),
            ],
            "response.completed",
            None,
        ),
        (
            [
                _responses_sse_event(
                    {
                        "type": "response.created",
                        "response": {
                            "id": "resp-success-incomplete",
                            "model": "gpt-5.4",
                            "status": "in_progress",
                        },
                    }
                ),
                _responses_sse_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": "msg_open",
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    }
                ),
            ],
            "response.incomplete",
            "Incomplete",
        ),
    ],
    ids=["completed-omission-normal-success", "incomplete-zero-turn-status"],
)
async def test_chunk_processor_synthetic_terminal_reconciles_route_accounting_once(
    monkeypatch,
    provider_chunks,
    expected_terminal_type,
    expected_route_status,
):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )

    async def _aiter_bytes():
        for provider_chunk in provider_chunks:
            yield provider_chunk

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj._update_completion_start_time = MagicMock()
    success_called = asyncio.Event()

    async def _capture_success(**kwargs):
        success_called.set()

    logging_obj.async_success_handler = AsyncMock(side_effect=_capture_success)
    logging_obj.async_failure_handler = AsyncMock()
    logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False
    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "aawm_route_rollup_context": {
                    "group_header_label": "litellm#Codex[0.141.0]",
                    "incoming_endpoint": "/openai_passthrough/responses",
                    "outgoing_target": "chatgpt.com/backend-api/codex/responses",
                    "model_label": "gpt-5.4",
                    "reasoning_effort": "high",
                }
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
    }

    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks",
        return_value={"result": {"response": "ok"}, "kwargs": {}},
    ) as collected_handler, patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup_turn"
    ) as record_turn, patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.emit_aawm_route_status_event"
    ) as emit_status, patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup"
    ) as record_rollup:
        chunks = []
        async for chunk in PassThroughStreamingHandler.chunk_processor(
            response=response,
            request_body={"model": "gpt-5.4"},
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now(),
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        ):
            chunks.append(chunk)

        await asyncio.wait_for(success_called.wait(), timeout=1)

    assert f'"type":"{expected_terminal_type}"'.encode() in b"".join(chunks)
    collected_handler.assert_called_once()
    logging_obj.async_success_handler.assert_awaited_once()
    logging_obj.async_failure_handler.assert_not_awaited()
    if expected_route_status is None:
        record_turn.assert_called_once()
        response_body = record_turn.call_args.kwargs["response_body"]
        assert response_body["status"] == "completed"
        emit_status.assert_not_called()
        record_rollup.assert_not_called()
    else:
        record_turn.assert_not_called()
        emit_status.assert_called_once()
        assert emit_status.call_args.kwargs["status"] == expected_route_status
        record_rollup.assert_called_once()
        rollup_kwargs = record_rollup.call_args.kwargs
        assert rollup_kwargs["status"] == expected_route_status
        assert rollup_kwargs["turns"] == 0


def test_record_post_first_byte_stream_terminal_rollup_preserves_reasoning_effort():
    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "aawm_route_rollup_context": {
                    "group_header_label": "litellm#Codex[0.141.0]",
                    "incoming_endpoint": "/openai_passthrough/responses",
                    "outgoing_target": "chatgpt.com/backend-api/codex/responses",
                    "model_label": "gpt-5.3-codex-spark(work)",
                    "reasoning_effort": "xhigh",
                }
            }
        }
    }
    failure_context = {
        "failure_kind": "streaming_upstream_read_timeout",
        "stream_failure_stage": "stream_interrupted_after_first_byte",
        "stream_chunks_seen": 2,
        "stream_bytes_seen": 64,
        "model": "gpt-5.3-codex-spark",
        "model_alias": "work",
        "route_family": "codex_responses",
    }
    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.emit_aawm_route_status_event"
    ) as mock_status, patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup"
    ) as mock_rollup:
        PassThroughStreamingHandler._record_post_first_byte_stream_terminal_rollup(
            success_handler_kwargs=success_handler_kwargs,
            failure_context=failure_context,
            exc=httpx.ReadTimeout("upstream timed out", request=None),
        )

    mock_status.assert_called_once()
    mock_rollup.assert_called_once()
    rollup_kwargs = mock_rollup.call_args.kwargs
    assert rollup_kwargs["effort"] == "xhigh"
    assert rollup_kwargs["status"] == "Failed"
    assert rollup_kwargs["turns"] == 0
    assert rollup_kwargs["model_label"] == "work"
    assert rollup_kwargs["group_header_label"] == "litellm#Codex[0.141.0]"
