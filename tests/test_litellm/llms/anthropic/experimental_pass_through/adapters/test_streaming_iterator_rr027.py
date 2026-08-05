"""RR-027: Anthropic stream wrapper SSE emitters and residual contracts."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters import (
    streaming_iterator as si,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


def test_emit_helpers_produce_stable_anthropic_shapes() -> None:
    start = si.emit_message_start(model="claude-3", message_id="msg_fixed")
    assert start["type"] == "message_start"
    assert start["message"]["id"] == "msg_fixed"
    assert start["message"]["model"] == "claude-3"
    assert start["message"]["usage"]["cache_creation_input_tokens"] == 0

    block_start = si.emit_content_block_start(
        index=0, content_block={"type": "text", "text": ""}
    )
    assert block_start == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }

    delta = si.emit_content_block_delta(
        index=0, delta={"type": "text_delta", "text": "hi"}
    )
    assert delta["type"] == "content_block_delta"
    assert delta["delta"]["text"] == "hi"

    assert si.emit_content_block_stop(index=2) == {
        "type": "content_block_stop",
        "index": 2,
    }
    md = si.emit_message_delta(stop_reason="end_turn")
    assert md["type"] == "message_delta"
    assert md["delta"]["stop_reason"] == "end_turn"
    assert si.emit_message_stop() == {"type": "message_stop"}


def test_encode_anthropic_sse_chunk_frames_dict_and_passes_through_bytes() -> None:
    framed = si.encode_anthropic_sse_chunk({"type": "message_stop"})
    assert isinstance(framed, bytes)
    text = framed.decode("utf-8")
    assert text.startswith("event: message_stop\n")
    assert "data: " in text
    assert json.loads(text.split("data: ", 1)[1].strip()) == {"type": "message_stop"}

    raw = b"already-bytes"
    assert si.encode_anthropic_sse_chunk(raw) is raw


class _OneChunkStream:
    def __init__(self, chunk: ModelResponseStream) -> None:
        self._chunk = chunk
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self._chunk

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return self._chunk


def _primed_text_wrapper() -> si.AnthropicStreamWrapper:
    wrapper = si.AnthropicStreamWrapper(
        completion_stream=_OneChunkStream(
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content="hi"),
                        index=0,
                        finish_reason=None,
                    )
                ]
            )
        ),
        model="gemini/x",
    )
    wrapper.sent_first_chunk = True
    wrapper.sent_content_block_start = True
    wrapper.current_content_block_type = "text"
    return wrapper


def _translated_text_delta() -> dict:
    return {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hi"},
    }


def test_sync_stream_preserves_translated_gemini_text_delta() -> None:
    wrapper = _primed_text_wrapper()

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.transformation.LiteLLMAnthropicMessagesAdapter.translate_streaming_openai_response_to_anthropic",
        return_value=_translated_text_delta(),
    ):
        assert next(wrapper) == _translated_text_delta()


@pytest.mark.asyncio
async def test_async_stream_preserves_translated_gemini_text_delta() -> None:
    wrapper = _primed_text_wrapper()

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.transformation.LiteLLMAnthropicMessagesAdapter.translate_streaming_openai_response_to_anthropic",
        return_value=_translated_text_delta(),
    ):
        assert await wrapper.__anext__() == _translated_text_delta()


def test_anthropic_sse_wrapper_uses_shared_encoder() -> None:
    wrapper = si.AnthropicStreamWrapper(completion_stream=iter(()), model="claude-3")
    wrapper.chunk_queue.append(si.emit_message_stop())
    # After stream ends, message_stop already queued; force StopIteration path carefully
    # by empty completion stream and sent_last_message already True with queue
    wrapper.sent_first_chunk = True
    wrapper.sent_last_message = True
    frames = list(wrapper.anthropic_sse_wrapper())
    assert frames
    assert frames[0].startswith(b"event: message_stop\n")


def test_issue1_prompt_cache_key_not_owned_by_streaming_iterator() -> None:
    """RR-027 #1 lives in observability.derive_prompt_cache_key; not this file."""
    import inspect
    import re

    source = inspect.getsource(si)
    # Module may document the ownership boundary in comments; it must not define
    # or call derive_prompt_cache_key itself.
    assert "def derive_prompt_cache_key" not in source
    assert re.search(r"(?<![\w.])derive_prompt_cache_key\s*\(", source) is None
    from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
        derive_prompt_cache_key,
    )

    # Stability: system/tools only — moving user cache_control must not change key.
    base = {
        "system": [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ],
        "tools": [{"name": "t", "input_schema": {"type": "object"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "turn-1",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    next_turn = {
        **base,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "turn-2-different",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    assert derive_prompt_cache_key(base) == derive_prompt_cache_key(next_turn)
    assert derive_prompt_cache_key(base) is not None


def test_shared_emitter_surface_is_importable_for_parallel_trees() -> None:
    """RR-027 #2: Chat/Responses can share these emitters without re-copying framing."""
    names = [
        "emit_message_start",
        "emit_content_block_start",
        "emit_content_block_delta",
        "emit_content_block_stop",
        "emit_message_delta",
        "emit_message_stop",
        "encode_anthropic_sse_chunk",
    ]
    for name in names:
        assert callable(getattr(si, name))


def test_tool_use_multi_tool_path_skips_empty_partial_json_and_uses_shared_emitter() -> (
    None
):
    """Parallel tool chunks must not emit empty input_json_delta after start."""
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
        Usage,
    )

    responses = [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_a",
                                function=Function(name="get_weather", arguments=""),
                                type="function",
                                index=0,
                            ),
                            ChatCompletionDeltaToolCall(
                                id="call_b",
                                function=Function(name="get_time", arguments=""),
                                type="function",
                                index=1,
                            ),
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ]
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id=None,
                                function=Function(name=None, arguments='{"city":"NY"}'),
                                type=None,
                                index=0,
                            ),
                            ChatCompletionDeltaToolCall(
                                id=None,
                                function=Function(name=None, arguments='{"tz":"UTC"}'),
                                type=None,
                                index=1,
                            ),
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ]
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content=""),
                    index=0,
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    ]
    wrapper = si.AnthropicStreamWrapper(
        completion_stream=iter(responses), model="sonnet-4-5"
    )
    chunks = list(wrapper)
    types = [c["type"] for c in chunks]
    deltas = [
        c
        for c in chunks
        if c.get("type") == "content_block_delta"
        and c.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert len(deltas) == 2
    assert all(d["delta"].get("partial_json") for d in deltas)
    assert types.count("content_block_start") >= 2
    assert types.count("content_block_stop") >= 2
    assert types.index("content_block_start") < types.index("content_block_delta")
    assert types.index("content_block_delta") < types.index("content_block_stop")


def test_no_inline_content_block_envelope_literals_in_wrapper_methods() -> None:
    """RR-027 #2: wrapper body should emit via shared helpers, not raw type dicts."""
    import inspect
    import re

    source = inspect.getsource(si.AnthropicStreamWrapper)
    # Allow comments; ban common inline envelope construction patterns.
    banned = [
        r'\{\s*"type":\s*"content_block_start"',
        r'\{\s*"type":\s*"content_block_stop"',
        r'\{\s*"type":\s*"message_start"',
        r'\{\s*"type":\s*"message_stop"',
    ]
    for pat in banned:
        assert re.search(pat, source) is None, pat
