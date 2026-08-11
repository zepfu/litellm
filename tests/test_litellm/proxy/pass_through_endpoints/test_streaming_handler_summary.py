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
            request_body={"model": "gpt-5.4"},
            endpoint_type=EndpointType.OPENAI,
            start_time=datetime.now() - timedelta(milliseconds=10),
            raw_bytes=[b"data: {\"type\":\"response.completed\"}\n\n"],
            precomputed_lines=['data: {"type":"response.completed"}'],
            end_time=datetime.now(),
            custom_llm_provider="openai",
            success_handler_kwargs=success_handler_kwargs,
        )

    assert captured["all_chunks"] == ['data: {"type":"response.completed"}']
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_finalize_line_source"
        ]
        == "incremental_summary"
    )


@pytest.mark.asyncio
async def test_chunk_processor_increments_summary_lines_for_codex_responses(monkeypatch):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )

    async def _aiter_bytes():
        yield b'data: {"delta":1}\n\n'
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

    assert chunks == [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"]
    assert route_kwargs["precomputed_lines"] == [
        'data: {"delta":1}',
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

    assert chunks == [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"]
    assert route_kwargs["raw_bytes"] == []
    assert route_kwargs["precomputed_lines"] == [
        'data: {"delta":1}',
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

    assert chunks == [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"]
    assert route_kwargs["raw_bytes"] == [
        b'data: {"delta":1}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert route_kwargs["precomputed_lines"] == [
        'data: {"delta":1}',
        "data: [DONE]",
    ]
    assert (
        success_handler_kwargs["litellm_params"]["metadata"][
            "aawm_stream_raw_bytes_buffered"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_chunk_processor_synthesizes_once_for_codex_clean_eof(  # noqa: PLR0915
    monkeypatch,
):
    monkeypatch.setenv(
        PassThroughStreamingHandler._AAWM_STREAM_SUMMARY_FIRST_FINALIZE_ENV,
        "1",
    )
    monkeypatch.setenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", "1")

    first_chunk = (
        b'event: response.created\ndata: {"type":"response.created",'
        b'"response":{"id":"resp-partial","status":"in_progress"}}\n\n'
    )
    second_chunk = (
        b'event: response.output_text.delta\ndata: {"type":'
        b'"response.output_text.delta","delta":"exact bytes"}\n\n'
    )

    async def _aiter_bytes():
        yield first_chunk
        yield second_chunk + b"data: [DONE]\n\n"
        yield b"data: [DO"
        yield b"NE]\n\n"

    response = MagicMock()
    response.headers = httpx.Headers({})
    response.aiter_bytes = _aiter_bytes

    logging_obj = MagicMock()
    logging_obj._update_completion_start_time = MagicMock()
    metadata_at_failure: dict = {}

    async def _capture_failure(**kwargs):
        metadata_at_failure.update(
            success_handler_kwargs["litellm_params"]["metadata"]
        )

    logging_obj.async_failure_handler = AsyncMock(side_effect=_capture_failure)
    logging_obj.async_success_handler = AsyncMock()
    route_handler = AsyncMock()
    status_events: list[dict] = []
    rollup_events: list[dict] = []

    success_handler_kwargs = {
        "litellm_params": {
            "metadata": {
                "existing_marker": "preserved",
                "aawm_route_rollup_context": {
                    "group_header_label": "Codex/litellm",
                    "incoming_endpoint": "/openai_passthrough/responses",
                    "outgoing_target": (
                        "https://chatgpt.com/backend-api/codex/responses"
                    ),
                    "model_label": "work",
                    "reasoning_effort": "high",
                },
            }
        }
    }
    error_log_context = {
        "provider": "openai",
        "model": "gpt-5.4",
        "model_alias": "work",
        "route_family": "codex_responses",
    }
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.streaming_handler.emit_aawm_route_status_event",
        lambda **kwargs: status_events.append(kwargs),
    )
    monkeypatch.setattr(
        "litellm.proxy.pass_through_endpoints.streaming_handler.record_aawm_route_rollup",
        lambda **kwargs: rollup_events.append(kwargs),
    )

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
            error_log_context=error_log_context,
        ):
            chunks.append(chunk)

    await asyncio.sleep(0.05)

    assert chunks[:2] == [first_chunk, second_chunk]
    combined = b"".join(chunks)
    assert combined.startswith(first_chunk + second_chunk)
    assert combined.count(b"event: response.incomplete") == 1
    assert combined.count(b"data: [DONE]") == 1
    assert combined.index(b"event: response.incomplete") < combined.index(
        b"data: [DONE]"
    )

    incomplete_payload = json.loads(
        chunks[-2].split(b"\ndata: ", 1)[1].removesuffix(b"\n\n")
    )
    assert incomplete_payload["type"] == "response.incomplete"
    assert incomplete_payload["response"]["status"] == "incomplete"
    assert incomplete_payload["response"]["incomplete_details"]["reason"] == (
        "upstream_stream_ended_without_terminal_event"
    )
    assert chunks[-1] == b"data: [DONE]\n\n"

    metadata = success_handler_kwargs["litellm_params"]["metadata"]
    assert metadata["existing_marker"] == "preserved"
    assert metadata["aawm_stream_interrupted"] is True
    assert metadata["aawm_stream_incomplete"] is True
    assert metadata["aawm_stream_terminal_emitted"] is True
    assert metadata["aawm_stream_replayable"] is False
    assert metadata["stream_hidden_retry_safe"] is False
    assert metadata["aawm_route_rollup_turn_suppressed"] is True
    assert metadata_at_failure["aawm_stream_terminal_emitted"] is True
    assert metadata_at_failure["aawm_stream_incomplete"] is True
    logging_obj.async_failure_handler.assert_awaited_once()
    logging_obj.async_success_handler.assert_not_awaited()
    route_handler.assert_not_awaited()
    assert len(status_events) == 1
    assert status_events[0]["status"] == "Failed"
    assert status_events[0]["alias_model"] == "work"
    assert len(rollup_events) == 1
    assert rollup_events[0]["status"] == "Failed"
    assert rollup_events[0]["model_label"] == "work"
    assert rollup_events[0]["turns"] == 0

    assert (
        await PassThroughStreamingHandler._terminalize_post_first_byte_responses_clean_eof(
            litellm_logging_obj=logging_obj,
            endpoint_type=EndpointType.OPENAI,
            url_route="https://chatgpt.com/backend-api/codex/responses",
            custom_llm_provider="openai",
            start_time=datetime.now(),
            error_log_context=error_log_context,
            success_handler_kwargs=success_handler_kwargs,
            chunk_count=4,
            total_stream_bytes=sum(
                len(chunk)
                for chunk in (
                    first_chunk,
                    second_chunk + b"data: [DONE]\n\n",
                    b"data: [DO",
                    b"NE]\n\n",
                )
            ),
        )
        == []
    )
    logging_obj.async_failure_handler.assert_awaited_once()
    logging_obj.async_success_handler.assert_not_awaited()
    assert len(status_events) == 1
    assert len(rollup_events) == 1


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
