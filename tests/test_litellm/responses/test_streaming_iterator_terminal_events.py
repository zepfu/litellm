"""
Focused tests: BaseResponsesAPIStreamingIterator terminal stream events.

RESPONSE_FAILED and RESPONSE_INCOMPLETE must take the same post-chunk logging
path as RESPONSE_COMPLETED so non-happy terminal streams still set
completed_response and invoke the logging handler.
"""

import json
from unittest.mock import Mock

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _make_iterator(mock_config, mock_logging_obj=None):
    mock_response = Mock()
    mock_response.headers = {}
    if mock_logging_obj is None:
        mock_logging_obj = Mock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {"litellm_params": {}}
    return BaseResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-4",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        litellm_metadata={"model_info": {"id": "model_123"}},
        custom_llm_provider="openai",
    )


def _terminal_event(event_type: ResponsesAPIStreamEvents):
    mock_responses_api_response = Mock(spec=ResponsesAPIResponse)
    mock_responses_api_response.id = "resp_terminal"
    mock_event = Mock()
    mock_event.type = event_type
    mock_event.response = mock_responses_api_response
    return mock_event


@pytest.mark.parametrize(
    "event_type,chunk_type",
    [
        (ResponsesAPIStreamEvents.RESPONSE_COMPLETED, "response.completed"),
        (ResponsesAPIStreamEvents.RESPONSE_FAILED, "response.failed"),
        (ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE, "response.incomplete"),
    ],
)
def test_process_chunk_terminal_events_set_completed_response_and_log(
    event_type, chunk_type
):
    """
    All terminal lifecycle events must store completed_response and call the
    logging handler (same path as RESPONSE_COMPLETED).
    """
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    terminal_event = _terminal_event(event_type)
    mock_config.transform_streaming_response.return_value = terminal_event

    iterator = _make_iterator(mock_config)
    logged = []
    iterator._handle_logging_completed_response = (  # type: ignore[method-assign]
        lambda: logged.append(True)
    )

    chunk = json.dumps(
        {
            "type": chunk_type,
            "response": {
                "id": "resp_terminal",
                "output": [],
            },
        }
    )
    result = iterator._process_chunk(chunk)

    assert result is not None
    assert result.type == event_type
    assert iterator.completed_response is result
    assert logged == [True]


def test_process_chunk_non_terminal_event_skips_logging():
    """Delta / non-terminal events must not set completed_response or log."""
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    delta_event = Mock()
    delta_event.type = ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    # no response attribute
    delattr(delta_event, "response") if hasattr(delta_event, "response") else None
    mock_config.transform_streaming_response.return_value = delta_event

    iterator = _make_iterator(mock_config)
    logged = []
    iterator._handle_logging_completed_response = (  # type: ignore[method-assign]
        lambda: logged.append(True)
    )

    chunk = json.dumps(
        {
            "type": "response.output_text.delta",
            "delta": "hi",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
        }
    )
    result = iterator._process_chunk(chunk)

    assert result is not None
    assert result.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
    assert iterator.completed_response is None
    assert logged == []


def test_process_chunk_terminal_event_string_type_also_logs():
    """
    Providers/mocks may surface type as a plain string; membership in
    RESPONSES_API_TERMINAL_STREAM_EVENTS must still trigger logging.
    """
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    terminal_event = Mock()
    terminal_event.type = "response.failed"  # plain string, not enum
    terminal_event.response = Mock(spec=ResponsesAPIResponse)
    terminal_event.response.id = "resp_failed_str"
    mock_config.transform_streaming_response.return_value = terminal_event

    iterator = _make_iterator(mock_config)
    logged = []
    iterator._handle_logging_completed_response = (  # type: ignore[method-assign]
        lambda: logged.append(True)
    )

    result = iterator._process_chunk(
        json.dumps(
            {
                "type": "response.failed",
                "response": {"id": "resp_failed_str", "output": []},
            }
        )
    )

    assert result is not None
    assert iterator.completed_response is result
    assert logged == [True]
