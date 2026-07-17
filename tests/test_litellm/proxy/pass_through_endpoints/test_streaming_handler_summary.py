import asyncio
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
