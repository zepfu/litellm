"""RR-031: Responses streaming_iterator uses shared Anthropic SSE emitter.

Residual #1 (volatile prompt_cache_key) is already closed in shared
`adapters/observability.derive_prompt_cache_key` (system+tools only); this file
does not reimplement hashing.

Residual #2: Responses streaming must emit Anthropic event envelopes and SSE
frames through the shared helpers in
`adapters.streaming_iterator` so Chat vs Responses framing cannot drift.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters import (
    streaming_iterator as shared_emitter,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    derive_prompt_cache_key,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


async def _make_stream(*events):
    for event in events:
        yield event


def _completed_response(*, input_tokens=10, output_tokens=5, status="completed"):
    return SimpleNamespace(
        status=status,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
    )


@pytest.mark.asyncio
async def test_should_emit_content_block_delta_via_shared_helpers(monkeypatch) -> None:
    """All content_block_delta payloads must be produced by the shared emitter."""
    seen: list[tuple[int, dict]] = []
    original = shared_emitter.emit_content_block_delta

    def _spy(*, index, delta):
        seen.append((index, delta))
        return original(index=index, delta=delta)

    monkeypatch.setattr(
        "litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator.emit_content_block_delta",
        _spy,
    )

    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.delta",
            item_id="msg_1",
            delta="hello",
        ),
        SimpleNamespace(
            type="response.completed",
            response=_completed_response(),
        ),
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
    )
    chunks = [chunk async for chunk in wrapper]

    assert any(c["type"] == "content_block_delta" for c in chunks)
    assert seen
    assert seen[0][1] == {"type": "text_delta", "text": "hello"}
    # Shared helper shape must match emitted chunk.
    assert chunks[2] == shared_emitter.emit_content_block_delta(
        index=seen[0][0],
        delta={"type": "text_delta", "text": "hello"},
    )


@pytest.mark.asyncio
async def test_should_encode_sse_via_shared_framer(monkeypatch) -> None:
    """async_anthropic_sse_wrapper must use encode_anthropic_sse_chunk."""
    calls: list[object] = []
    original = shared_emitter.encode_anthropic_sse_chunk

    def _spy(chunk):
        calls.append(chunk)
        return original(chunk)

    monkeypatch.setattr(
        "litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator.encode_anthropic_sse_chunk",
        _spy,
    )

    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.delta",
            item_id="msg_1",
            delta="x",
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="message", id="msg_1"),
        ),
        SimpleNamespace(
            type="response.completed",
            response=_completed_response(),
        ),
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
    )
    frames = [frame async for frame in wrapper.async_anthropic_sse_wrapper()]

    assert calls, "shared encode_anthropic_sse_chunk was not used"
    assert all(isinstance(frame, (bytes, bytearray)) for frame in frames)
    # Framing contract matches shared helper for a dict event.
    sample = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "x"},
    }
    expected = shared_emitter.encode_anthropic_sse_chunk(sample)
    assert expected.startswith(b"event: content_block_delta\n")
    assert b"data: " in expected
    # At least one produced frame equals the shared encoding of a queued dict.
    decoded_events = []
    for frame in frames:
        text = frame.decode("utf-8")
        assert text.startswith("event: ")
        assert "\ndata: " in text
        event_line, data_line = text.strip().split("\n", 1)
        event_type = event_line[len("event: ") :]
        payload = json.loads(data_line[len("data: ") :])
        decoded_events.append(event_type)
        assert payload["type"] == event_type
    assert "message_start" in decoded_events
    assert "content_block_delta" in decoded_events
    assert "message_stop" in decoded_events


@pytest.mark.asyncio
async def test_should_use_shared_emitters_for_message_envelope(monkeypatch) -> None:
    start_calls = 0
    stop_calls = 0
    delta_calls = 0

    original_start = shared_emitter.emit_message_start
    original_stop = shared_emitter.emit_message_stop
    original_delta = shared_emitter.emit_message_delta

    def spy_start(**kwargs):
        nonlocal start_calls
        start_calls += 1
        return original_start(**kwargs)

    def spy_stop():
        nonlocal stop_calls
        stop_calls += 1
        return original_stop()

    def spy_delta(**kwargs):
        nonlocal delta_calls
        delta_calls += 1
        return original_delta(**kwargs)

    base = "litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator"
    monkeypatch.setattr(f"{base}.emit_message_start", spy_start)
    monkeypatch.setattr(f"{base}.emit_message_stop", spy_stop)
    monkeypatch.setattr(f"{base}.emit_message_delta", spy_delta)

    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.delta",
            item_id="msg_1",
            delta="ok",
        ),
        SimpleNamespace(
            type="response.completed",
            response=_completed_response(),
        ),
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
    )
    chunks = [chunk async for chunk in wrapper]
    types = [c["type"] for c in chunks]
    assert types[0] == "message_start"
    assert "message_delta" in types
    assert types[-1] == "message_stop"
    assert start_calls == 1
    assert stop_calls >= 1
    assert delta_calls >= 1


def test_prompt_cache_key_residual_closed_via_shared_helper() -> None:
    """Issue #1 evidence: responses_adapters depends on stable shared helper."""

    def body(user_text: str) -> dict:
        return {
            "system": [
                {
                    "type": "text",
                    "text": "stable system",
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
                            "text": user_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

    assert derive_prompt_cache_key(body("turn-a")) == derive_prompt_cache_key(
        body("turn-b-different")
    )


def test_shared_emitter_exports_required_surface() -> None:
    for name in (
        "emit_message_start",
        "emit_content_block_start",
        "emit_content_block_delta",
        "emit_content_block_stop",
        "emit_message_delta",
        "emit_message_stop",
        "encode_anthropic_sse_chunk",
    ):
        assert hasattr(shared_emitter, name)
        assert callable(getattr(shared_emitter, name))


def test_responses_module_imports_shared_emitter_symbols() -> None:
    import litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator as mod

    for name in (
        "emit_content_block_start",
        "emit_content_block_delta",
        "emit_content_block_stop",
        "emit_message_start",
        "emit_message_delta",
        "emit_message_stop",
        "encode_anthropic_sse_chunk",
    ):
        assert hasattr(mod, name)


@pytest.mark.asyncio
async def test_function_call_skips_empty_input_json_delta_and_keeps_order() -> None:
    """Empty arguments deltas must not be framed; real JSON must precede stop."""
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc1",
                call_id="call_1",
                name="Bash",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc1",
            delta="",
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc1",
            delta='{"command":"ls"}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc1",
                call_id="call_1",
                name="Bash",
                arguments='{"command":"ls"}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=_completed_response(),
        ),
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
    )
    chunks = [chunk async for chunk in wrapper]
    types = [c["type"] for c in chunks]
    assert types[:4] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    assert chunks[1]["content_block"]["type"] == "tool_use"
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"command":"ls"}',
    }
    assert types[-1] == "message_stop"


@pytest.mark.asyncio
async def test_incomplete_exec_command_buffer_flushes_on_arguments_done() -> None:
    """Code-execution style incomplete JSON accumulation must flush on .done."""
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_exec",
                call_id="call_exec",
                name="exec_command",
                arguments="",
            ),
        ),
        # Incomplete JSON held in buffer
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_exec",
            delta='{"cmd":"pw',
        ),
        # .done carries full arguments (or buffer flush if empty)
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="fc_exec",
            arguments='{"cmd":"pwd"}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_exec",
                call_id="call_exec",
                name="exec_command",
                arguments='{"cmd":"pwd"}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=_completed_response(),
        ),
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
        use_codex_native_tools=True,
    )
    chunks = [chunk async for chunk in wrapper]
    deltas = [
        c
        for c in chunks
        if c.get("type") == "content_block_delta"
        and c.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert deltas, "expected flushed input_json_delta for exec_command"
    # Mapped to Bash with command key when codex native tools enabled
    joined = "".join(d["delta"]["partial_json"] for d in deltas)
    assert "pwd" in joined
    types = [c["type"] for c in chunks]
    assert types.index("content_block_delta") < types.index("content_block_stop")
    assert types[-2:] == ["message_delta", "message_stop"]
