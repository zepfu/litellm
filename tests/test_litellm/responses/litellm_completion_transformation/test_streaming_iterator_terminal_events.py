"""Ohmypi openai-responses streams must emit a terminal Responses event.

OpenRouter adapter traffic wraps LiteLLMCompletionStreamingIterator. When the
underlying CustomStreamWrapper ends, the iterator must still emit
response.completed (or failed/incomplete) so Ohmypi does not loop on
"OpenAI responses stream closed before a terminal response event was received".
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.sse import (
    _responses_sse_from_iterator,
)
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from litellm.utils import ModelResponseListIterator

_TERMINAL_EVENT_TYPES = {
    ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ResponsesAPIStreamEvents.RESPONSE_FAILED,
    ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
    "response.completed",
    "response.failed",
    "response.incomplete",
}


def _pong_model_response_stream_chunks(
    *,
    response_id: str,
    model: str,
) -> list[ModelResponseStream]:
    """Real chat.completion.chunk objects that assemble to assistant content PONG."""
    return [
        ModelResponseStream(
            id=response_id,
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content="PO",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields={},
            usage=None,
        ),
        ModelResponseStream(
            id=response_id,
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content="NG",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields={},
            usage=None,
        ),
        ModelResponseStream(
            id=response_id,
            created=1234567890,
            model=model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason="stop",
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content="",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields={},
            usage=None,
        ),
    ]


def _real_custom_stream_wrapper(
    *,
    model: str,
    completion_stream: Optional[Iterable[Any]],
    response_id: Optional[str] = None,
) -> CustomStreamWrapper:
    logging_obj = MagicMock()
    logging_obj.async_success_handler = AsyncMock(return_value=None)
    logging_obj.success_handler = MagicMock()
    logging_obj._llm_caching_handler = None
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.completion_start_time = None
    wrapper = CustomStreamWrapper(
        completion_stream=completion_stream,
        model=model,
        logging_obj=logging_obj,
        custom_llm_provider="openrouter",
    )
    wrapper.response_id = response_id
    return wrapper


def _event_type(event: Any) -> Any:
    value = getattr(event, "type", None)
    if value is None and isinstance(event, dict):
        value = event.get("type")
    if hasattr(value, "value"):
        return value.value
    return value


def _completed_output_text(event: Any) -> str:
    response = getattr(event, "response", None)
    if response is None and isinstance(event, dict):
        response = event.get("response")
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    assembled: list[str] = []
    output_items = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    for item in output_items or []:
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type != "message":
            continue
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if isinstance(text, str) and text:
                assembled.append(text)
    return "".join(assembled)


async def _exhaust_async_iterator(iterator: LiteLLMCompletionStreamingIterator) -> list[Any]:
    events: list[Any] = []
    async for event in iterator:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_completion_streaming_iterator_empty_wrapper_emits_response_completed():
    """An immediately empty CustomStreamWrapper still needs response.completed."""
    model = "openrouter/qwen/qwen3.6-flash:none"
    wrapper = _real_custom_stream_wrapper(
        model=model,
        completion_stream=ModelResponseListIterator(model_responses=[]),
        response_id="chatcmpl-ohmypi-basic-empty",
    )
    iterator = LiteLLMCompletionStreamingIterator(
        model=model,
        litellm_custom_stream_wrapper=wrapper,
        request_input="Reply with exactly the word PONG.",
        responses_api_request={},
        custom_llm_provider="openrouter",
    )

    events = await _exhaust_async_iterator(iterator)
    event_types = [_event_type(event) for event in events]

    assert any(
        event_type in _TERMINAL_EVENT_TYPES for event_type in event_types
    ), (
        "empty completion wrapper must emit a terminal Responses event; "
        f"got {event_types}"
    )
    assert "response.completed" in event_types


@pytest.mark.asyncio
async def test_completion_streaming_iterator_pong_chunks_emit_response_completed():
    """PONG chat chunks must finish with response.completed carrying PONG text."""
    model = "openrouter/qwen/qwen3.6-flash:none"
    response_id = "chatcmpl-ohmypi-basic"
    wrapper = _real_custom_stream_wrapper(
        model=model,
        completion_stream=ModelResponseListIterator(
            model_responses=_pong_model_response_stream_chunks(
                response_id=response_id,
                model=model,
            )
        ),
        response_id=response_id,
    )
    iterator = LiteLLMCompletionStreamingIterator(
        model=model,
        litellm_custom_stream_wrapper=wrapper,
        request_input="Reply with exactly the word PONG.",
        responses_api_request={},
        custom_llm_provider="openrouter",
    )

    events = await _exhaust_async_iterator(iterator)
    event_types = [_event_type(event) for event in events]
    completed_events = [
        event
        for event in events
        if _event_type(event)
        in {ResponsesAPIStreamEvents.RESPONSE_COMPLETED, "response.completed"}
    ]

    assert completed_events, (
        "PONG completion stream must emit response.completed; "
        f"got {event_types}"
    )
    assembled = _completed_output_text(completed_events[-1])
    assert "PONG" in assembled, assembled


@pytest.mark.asyncio
async def test_completion_streaming_iterator_emits_terminal_when_completed_transform_raises():
    """_emit_response_completed_event may raise after response.created was sent.

    `_ensure_terminal_done_event` currently only swallows StopAsyncIteration /
    StopIteration. A transform/usage/pydantic failure on the completed event
    must still return a terminal Responses event instead of aborting the
    iterator, or Ohmypi retries until the budget is exhausted.
    """
    model = "openrouter/qwen/qwen3.6-flash:none"
    response_id = "chatcmpl-ohmypi-transform-fail"
    wrapper = _real_custom_stream_wrapper(
        model=model,
        completion_stream=ModelResponseListIterator(
            model_responses=_pong_model_response_stream_chunks(
                response_id=response_id,
                model=model,
            )
        ),
        response_id=response_id,
    )
    iterator = LiteLLMCompletionStreamingIterator(
        model=model,
        litellm_custom_stream_wrapper=wrapper,
        request_input="Reply with exactly the word PONG.",
        responses_api_request={},
        custom_llm_provider="openrouter",
    )

    def _raise_transform_failed(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("transform failed")

    iterator._emit_response_completed_event = _raise_transform_failed  # type: ignore[method-assign]

    events = await _exhaust_async_iterator(iterator)
    event_types = [_event_type(event) for event in events]

    assert "response.created" in event_types, event_types
    assert any(
        event_type in _TERMINAL_EVENT_TYPES for event_type in event_types
    ), (
        "completed-event transform failure must still emit a terminal "
        f"Responses event; got {event_types}"
    )


def _sse_event_name(chunk: str) -> Optional[str]:
    first_line = chunk.split("\n", 1)[0]
    if first_line.startswith("event: "):
        return first_line[len("event: ") :]
    return None


def _sse_data_payload(chunk: str) -> Any:
    for line in chunk.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[len("data: ") :]
        if raw == "[DONE]":
            return "[DONE]"
        return json.loads(raw)
    return None


async def _collect_sse_chunks(iterator: LiteLLMCompletionStreamingIterator) -> list[str]:
    chunks: list[str] = []
    async for chunk in _responses_sse_from_iterator(iterator):
        chunks.append(chunk if isinstance(chunk, str) else str(chunk))
    return chunks


@pytest.mark.asyncio
async def test_completion_streaming_iterator_wrapper_error_after_created_emits_sse_completed():
    """Live OpenRouter wrap: wrapper exception after response.created must still
    close Ohmypi with response.completed, not only response.failed.

    `_perform_codex_auto_agent_openrouter_completion_request` wraps
    LiteLLMCompletionStreamingIterator in `_responses_sse_from_iterator`
    without on_stream_error. Ohmypi OpenAI treats only
    response.completed / response.incomplete / response.done as a successful
    terminal event. A sparse response.failed inject still throws
    "OpenAI responses stream closed before a terminal response event was received".
    """
    model = "openrouter/qwen/qwen3.6-flash:none"
    wrapper = _real_custom_stream_wrapper(
        model=model,
        completion_stream=ModelResponseListIterator(
            model_responses=_pong_model_response_stream_chunks(
                response_id="chatcmpl-ohmypi-wrapper-fail",
                model=model,
            )
        ),
        response_id="chatcmpl-ohmypi-wrapper-fail",
    )
    iterator = LiteLLMCompletionStreamingIterator(
        model=model,
        litellm_custom_stream_wrapper=wrapper,
        request_input="Reply with exactly the word PONG.",
        responses_api_request={},
        custom_llm_provider="openrouter",
    )

    async def _raise_openrouter_chunk_failed(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("openrouter chunk failed")

    wrapper.__anext__ = _raise_openrouter_chunk_failed  # type: ignore[method-assign]

    chunks = await _collect_sse_chunks(iterator)
    event_names = [_sse_event_name(chunk) for chunk in chunks]

    assert "response.created" in event_names, chunks
    assert "response.completed" in event_names, (
        "Ohmypi OpenAI requires event: response.completed after a post-created "
        "wrapper exception; response.failed alone is not a terminal success "
        f"event; got {event_names}"
    )
    assert chunks[-1] == "data: [DONE]\n\n", chunks
    assert event_names[-1] is None and _sse_data_payload(chunks[-1]) == "[DONE]"

    created_payload = next(
        _sse_data_payload(chunk)
        for chunk in chunks
        if _sse_event_name(chunk) == "response.created"
    )
    completed_payload = next(
        _sse_data_payload(chunk)
        for chunk in chunks
        if _sse_event_name(chunk) == "response.completed"
    )
    created_response = created_payload["response"]
    completed_response = completed_payload["response"]
    assert completed_payload.get("type") == "response.completed"
    assert completed_response["id"] == created_response["id"]
    assert completed_response["object"] == "response"
    assert isinstance(completed_response["created_at"], int)
    assert completed_response.get("status")
