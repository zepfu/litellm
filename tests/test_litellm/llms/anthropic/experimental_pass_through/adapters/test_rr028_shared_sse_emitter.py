"""RR-028: shared Anthropic SSE emitter helpers in adapters/transformation.py.

#1 prompt_cache_key volatility is owned by adapters/observability.derive_prompt_cache_key
(already stable: system+tools only). This file covers #2 — shared event builders so Chat
and Responses lanes cannot invent divergent content_block_* / message_* envelopes.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    derive_prompt_cache_key,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
    build_anthropic_content_block_delta,
    build_anthropic_content_block_start,
    build_anthropic_content_block_stop,
    build_anthropic_input_json_delta,
    build_anthropic_message_delta,
    build_anthropic_message_stop,
    build_anthropic_signature_delta,
    build_anthropic_text_delta,
    build_anthropic_thinking_delta,
    encode_anthropic_sse_event,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def test_rr028_prompt_cache_key_stable_across_user_turns_not_valid_for_this_file() -> None:
    """Issue #1 evidence: stable key lives in observability; this transformation file
    does not reimplement derive_prompt_cache_key. Prove the shared helper stays stable.
    """
    def body(user: str) -> Dict[str, Any]:
        return {
            "system": [
                {
                    "type": "text",
                    "text": "sys",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [
                {
                    "name": "Bash",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

    assert derive_prompt_cache_key(body("turn-a")) == derive_prompt_cache_key(
        body("turn-b")
    )


def test_shared_content_block_builders_match_anthropic_envelope() -> None:
    start = build_anthropic_content_block_start(
        index=1, content_block={"type": "text", "text": ""}
    )
    assert start == {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "text", "text": ""},
    }

    delta = build_anthropic_content_block_delta(
        index=1, delta=build_anthropic_text_delta(text="hello")
    )
    assert delta == {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": "hello"},
    }

    stop = build_anthropic_content_block_stop(index=1)
    assert stop == {"type": "content_block_stop", "index": 1}


def test_shared_delta_builders() -> None:
    assert build_anthropic_input_json_delta(partial_json='{"a":1}') == {
        "type": "input_json_delta",
        "partial_json": '{"a":1}',
    }
    assert build_anthropic_thinking_delta(thinking="reason") == {
        "type": "thinking_delta",
        "thinking": "reason",
    }
    assert build_anthropic_signature_delta(signature="sig") == {
        "type": "signature_delta",
        "signature": "sig",
    }
    assert build_anthropic_message_stop() == {"type": "message_stop"}
    assert build_anthropic_message_delta(stop_reason="end_turn") == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
    }
    assert build_anthropic_message_delta(
        stop_reason="tool_use",
        usage={"input_tokens": 3, "output_tokens": 4},
    ) == {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }


def test_encode_anthropic_sse_event_frame() -> None:
    chunk = build_anthropic_content_block_stop(index=0)
    frame = encode_anthropic_sse_event(chunk)
    assert frame.startswith(b"event: content_block_stop\n")
    assert frame.endswith(b"\n\n")
    # data line is valid JSON equal to the chunk
    data_line = frame.decode().split("\n")[1]
    assert data_line.startswith("data: ")
    assert json.loads(data_line[len("data: ") :]) == chunk


def test_chat_lane_streaming_delta_uses_shared_emitter() -> None:
    """Chat Completions streaming path must produce shared envelope dict shape."""
    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(content="hi"),
        )
    ]
    adapter = LiteLLMAnthropicMessagesAdapter()
    delta_type, delta = adapter._translate_streaming_openai_chunk_to_anthropic(
        choices=choices
    )
    assert delta_type == "text_delta"
    assert dict(delta) == build_anthropic_text_delta(text="hi")

    event = build_anthropic_content_block_delta(
        index=3, delta=dict(delta)
    )
    # Same envelope the translate_streaming_* path returns via shared builder
    stream = ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="hi"),
            )
        ]
    )
    # finish_reason None => content_block_delta branch
    assert stream.choices[0].finish_reason is None
    translated = adapter.translate_streaming_openai_response_to_anthropic(
        response=stream,  # type: ignore[arg-type]
        current_content_block_index=3,
    )
    assert translated["type"] == "content_block_delta"
    assert translated["index"] == 3
    assert translated["delta"]["type"] == "text_delta"
    assert translated["delta"]["text"] == "hi"
    assert dict(translated) == event


def test_chat_lane_partial_json_uses_shared_delta_builder() -> None:
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    choices = [
        StreamingChoices(
            finish_reason=None,
            index=0,
            delta=Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        index=0,
                        id=None,
                        type="function",
                        function=Function(name=None, arguments='{"x":'),
                    )
                ]
            ),
        )
    ]
    adapter = LiteLLMAnthropicMessagesAdapter()
    delta_type, delta = adapter._translate_streaming_openai_chunk_to_anthropic(
        choices=choices
    )
    assert delta_type == "input_json_delta"
    assert dict(delta) == build_anthropic_input_json_delta(partial_json='{"x":')

    stream = ModelResponseStream(choices=choices)
    assert stream.choices[0].finish_reason is None
    event = adapter.translate_streaming_openai_response_to_anthropic(
        response=stream,  # type: ignore[arg-type]
        current_content_block_index=0,
    )
    assert event["type"] == "content_block_delta"
    assert event["delta"]["type"] == "input_json_delta"
    assert event["delta"]["partial_json"] == '{"x":'
    assert dict(event) == build_anthropic_content_block_delta(
        index=0,
        delta=build_anthropic_input_json_delta(partial_json='{"x":'),
    )


def test_shared_builders_are_exported_for_responses_lane() -> None:
    """Responses streaming_iterator already imports sanitize helpers from this module;
    the SSE builders must be importable from the same surface for cross-lane reuse.
    """
    import litellm.llms.anthropic.experimental_pass_through.adapters.transformation as mod

    for name in (
        "build_anthropic_content_block_start",
        "build_anthropic_content_block_delta",
        "build_anthropic_content_block_stop",
        "build_anthropic_text_delta",
        "build_anthropic_input_json_delta",
        "build_anthropic_thinking_delta",
        "build_anthropic_signature_delta",
        "build_anthropic_message_delta",
        "build_anthropic_message_stop",
        "encode_anthropic_sse_event",
    ):
        assert callable(getattr(mod, name)), name
