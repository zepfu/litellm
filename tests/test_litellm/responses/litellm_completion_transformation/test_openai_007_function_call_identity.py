"""OPENAI-007: Responses function_call id vs call_id identity for Chat Completions adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock

from litellm.responses.litellm_completion_transformation.function_call_identity import (
    generate_responses_function_call_item_id,
    is_native_responses_function_call_item_id,
    resolve_responses_function_call_identity,
)
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import (
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    ChatCompletionMessageToolCall,
)


NATIVE_FC_ID = "fc_685c42deefc0819a822b6936faaa30be0c76bc1491ab6619"
NATIVE_FC_UUID_ID = "fc_1fe70e2a-a596-45ef-b72c-9b8567c460e5"


def test_helper_preserves_native_uuid_shaped_fc_ids_unchanged():
    item_id, call_id = resolve_responses_function_call_identity(NATIVE_FC_UUID_ID)
    assert item_id == NATIVE_FC_UUID_ID
    assert call_id == NATIVE_FC_UUID_ID
    assert is_native_responses_function_call_item_id(NATIVE_FC_UUID_ID)
    assert item_id == call_id


def test_helper_preserves_non_empty_provider_id_byte_for_byte_including_whitespace():
    spaced = " call_kimi_ws "
    leading = "\tleading_id"
    trailing = "trailing_id  "
    for provider_id in (spaced, leading, trailing):
        item_id, call_id = resolve_responses_function_call_identity(provider_id)
        assert call_id == provider_id
        assert item_id == generate_responses_function_call_item_id(provider_id)
        assert item_id != call_id
        assert is_native_responses_function_call_item_id(item_id)


def test_helper_whitespace_distinct_provider_ids_do_not_collapse():
    a = "call_ws"
    b = " call_ws"
    c = "call_ws "
    identities = [
        resolve_responses_function_call_identity(value) for value in (a, b, c)
    ]
    call_ids = [call_id for _, call_id in identities]
    item_ids = [item_id for item_id, _ in identities]
    assert call_ids == [a, b, c]
    assert len(set(call_ids)) == 3
    assert len(set(item_ids)) == 3
    assert item_ids[0] == generate_responses_function_call_item_id(a)
    assert item_ids[1] == generate_responses_function_call_item_id(b)
    assert item_ids[2] == generate_responses_function_call_item_id(c)


def test_helper_empty_and_whitespace_only_provider_ids_resolve_empty():
    for provider_id in (None, "", "   ", "\t\n"):
        assert resolve_responses_function_call_identity(provider_id) == ("", "")


def test_helper_preserves_native_fc_ids_unchanged():
    item_id, call_id = resolve_responses_function_call_identity(NATIVE_FC_ID)
    assert item_id == NATIVE_FC_ID
    assert call_id == NATIVE_FC_ID
    assert is_native_responses_function_call_item_id(NATIVE_FC_ID)
    assert item_id == call_id


def test_helper_maps_non_native_provider_id_to_stable_distinct_fc_item_id():
    provider_ids = [
        "call_kimi_1",
        "toolu_alibaba_xyz",
        "openrouter-tool-9",
        "opencode_call_42",
        "fc_1",  # short non-native placeholder
    ]
    seen_item_ids: set[str] = set()
    for provider_id in provider_ids:
        item_id_1, call_id_1 = resolve_responses_function_call_identity(provider_id)
        item_id_2, call_id_2 = resolve_responses_function_call_identity(provider_id)
        assert call_id_1 == provider_id
        assert call_id_2 == provider_id
        assert item_id_1 == item_id_2
        assert item_id_1.startswith("fc_")
        assert is_native_responses_function_call_item_id(item_id_1)
        assert item_id_1 != call_id_1
        assert item_id_1 == generate_responses_function_call_item_id(provider_id)
        assert item_id_1 not in seen_item_ids
        seen_item_ids.add(item_id_1)


def test_non_stream_conversion_uses_provider_id_only_as_call_id():
    provider_id = "call_openrouter_weather"
    response = ModelResponse(
        id="chatcmpl-1",
        created=1,
        model="openrouter/test",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=provider_id,
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments='{"city":"NYC"}',
                            ),
                        )
                    ],
                ),
            )
        ],
    )

    tools = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        chat_completion_response=response
    )
    assert len(tools) == 1
    tool = tools[0]
    expected_item_id, expected_call_id = resolve_responses_function_call_identity(
        provider_id
    )
    assert tool.call_id == expected_call_id == provider_id
    assert tool.id == expected_item_id
    assert tool.id != tool.call_id
    assert tool.id.startswith("fc_")


def test_non_stream_conversion_leaves_native_fc_provider_id_unchanged():
    response = ModelResponse(
        id="chatcmpl-2",
        created=1,
        model="openai/test",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=NATIVE_FC_ID,
                            type="function",
                            function=Function(
                                name="do_thing",
                                arguments="{}",
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    tools = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        chat_completion_response=response
    )
    assert tools[0].id == NATIVE_FC_ID
    assert tools[0].call_id == NATIVE_FC_ID


def test_non_stream_conversion_leaves_native_uuid_shaped_fc_provider_id_unchanged():
    response = ModelResponse(
        id="chatcmpl-uuid",
        created=1,
        model="openai/test",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=NATIVE_FC_UUID_ID,
                            type="function",
                            function=Function(
                                name="do_thing",
                                arguments="{}",
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    tools = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        chat_completion_response=response
    )
    assert tools[0].id == NATIVE_FC_UUID_ID
    assert tools[0].call_id == NATIVE_FC_UUID_ID


def test_non_stream_conversion_preserves_whitespace_bearing_provider_id_byte_for_byte():
    provider_id = " call_openrouter_ws "
    response = ModelResponse(
        id="chatcmpl-ws",
        created=1,
        model="openrouter/test",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=provider_id,
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments='{"city":"NYC"}',
                            ),
                        )
                    ],
                ),
            )
        ],
    )
    tools = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        chat_completion_response=response
    )
    expected_item_id, expected_call_id = resolve_responses_function_call_identity(
        provider_id
    )
    assert tools[0].call_id == expected_call_id == provider_id
    assert tools[0].id == expected_item_id
    assert tools[0].id != tools[0].call_id


def test_stream_events_reuse_same_item_id_across_added_delta_done():
    provider_id = "call_kimi_stream"
    expected_item_id, expected_call_id = resolve_responses_function_call_identity(
        provider_id
    )
    iterator = LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
    )

    chunk = ModelResponseStream(
        id="chunk-1",
        created=123,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": provider_id,
                            "type": "function",
                            "function": {
                                "name": "do_thing",
                                "arguments": '{"x":1}',
                            },
                        }
                    ],
                ),
            )
        ],
    )

    added = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    assert added is not None
    assert added.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    assert added.item.id == expected_item_id
    assert added.item.call_id == expected_call_id
    assert added.item.id != added.item.call_id

    delta = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    assert delta is not None
    assert delta.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA
    assert delta.item_id == expected_item_id

    # Finalize with a completed ModelResponse for done/output_item.done.
    final = ModelResponse(
        id="resp-1",
        created=123,
        model="test-model",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": provider_id,
                            "type": "function",
                            "function": {
                                "name": "do_thing",
                                "arguments": '{"x":1}',
                            },
                            "index": 0,
                        }
                    ],
                },
            }
        ],
    )
    iterator.litellm_model_response = final

    # Drain any remaining argument deltas then require done + output_item.done.
    saw_args_done = False
    saw_item_done = False
    for _ in range(20):
        evt = iterator.common_done_event_logic(sync_mode=True)
        if evt.type == ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE:
            assert evt.item_id == expected_item_id
            saw_args_done = True
        elif evt.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE and getattr(
            evt.item, "type", None
        ) == "function_call":
            assert evt.item.id == expected_item_id
            assert evt.item.call_id == expected_call_id
            assert evt.item.id != evt.item.call_id
            saw_item_done = True
            break
    assert saw_args_done
    assert saw_item_done
