import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)


# Create a simple test
class MockCompletionStream:
    def __init__(self):
        self.responses = [
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Hello"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" World"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""), index=0, finish_reason="stop"
                    )
                ],
            ),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def test_anthropic_sse_wrapper_format():
    """Test that the SSE wrapper produces proper event and data formatting"""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

    # Get the first chunk from the SSE wrapper
    first_chunk = next(wrapper.anthropic_sse_wrapper())

    # Verify it's bytes
    assert isinstance(first_chunk, bytes)

    # Decode and check format
    chunk_str = first_chunk.decode("utf-8")

    # Should have event line and data line
    lines = chunk_str.split("\n")
    assert len(lines) >= 3  # event line, data line, empty line (+ possibly more)
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    assert lines[2] == ""  # Empty line to end the SSE chunk


def test_anthropic_sse_wrapper_event_types():
    """Test that different chunk types produce correct event types"""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

    chunks = []
    for chunk in wrapper.anthropic_sse_wrapper():
        chunks.append(chunk.decode("utf-8"))
        if len(chunks) >= 3:  # Get first few chunks
            break

    # First chunk should be message_start
    assert "event: message_start" in chunks[0]
    assert '"type": "message_start"' in chunks[0]

    # Second chunk should be content_block_start
    assert "event: content_block_start" in chunks[1]
    assert '"type": "content_block_start"' in chunks[1]

    # Third chunk should be content_block_delta
    assert "event: content_block_delta" in chunks[2]
    assert '"type": "content_block_delta"' in chunks[2]


@pytest.mark.asyncio
async def test_async_anthropic_sse_wrapper():
    """Test the async version of the SSE wrapper"""

    class AsyncMockCompletionStream:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Hello"), index=0, finish_reason=None
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=" World"), index=0, finish_reason=None
                        )
                    ],
                ),
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.responses):
                raise StopAsyncIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockCompletionStream(), model="claude-3"
    )

    # Get the first chunk from the async SSE wrapper
    first_chunk = None
    stream = wrapper.async_anthropic_sse_wrapper()
    try:
        async for chunk in stream:
            first_chunk = chunk
            break
    finally:
        await stream.aclose()

    # Verify it's bytes and properly formatted
    assert first_chunk is not None
    assert isinstance(first_chunk, bytes)

    chunk_str = first_chunk.decode("utf-8")
    assert "event: message_start" in chunk_str
    assert '"type": "message_start"' in chunk_str


def test_sync_wrapper_synthesizes_end_turn_when_stream_ends_without_terminal_chunk():
    class MockCompletionStreamNoTerminal:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Hello"), index=0, finish_reason=None
                        )
                    ],
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamNoTerminal(), model="gemini-3-flash-preview"
    )

    chunks = list(wrapper)
    chunk_types = [chunk["type"] for chunk in chunks]

    assert "content_block_stop" in chunk_types
    assert "message_delta" in chunk_types
    assert chunk_types[-1] == "message_stop"

    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"


def test_sync_wrapper_omits_empty_read_pages_argument_in_complete_json_delta():
    class MockCompletionStreamWithReadTool:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionDeltaToolCall(
                                        id="call_read",
                                        function=Function(
                                            name="read_file",
                                            arguments='{"file_path": "/tmp/example.py", "pages": ""}',
                                        ),
                                        type="function",
                                        index=0,
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason="tool_calls",
                        )
                    ],
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithReadTool(),
        model="gpt-5.5",
        tool_name_mapping={"read_file": "Read"},
    )

    chunks = list(wrapper)
    tool_start = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_start"
        and chunk["content_block"]["type"] == "tool_use"
    )
    tool_delta = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "input_json_delta"
    )

    assert tool_start["content_block"]["name"] == "Read"
    assert tool_delta["delta"]["partial_json"] == '{"file_path":"/tmp/example.py"}'


def test_sync_wrapper_keeps_queued_chunks_instance_scoped():
    class MockCompletionStreamWithReadTool:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionDeltaToolCall(
                                        id="call_read",
                                        function=Function(
                                            name="read_file",
                                            arguments='{"file_path": "/tmp/example.py"}',
                                        ),
                                        type="function",
                                        index=0,
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                )
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    first_wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithReadTool(),
        model="gemini-3.1-pro-preview",
        tool_name_mapping={"read_file": "Read"},
    )

    # Anthropic tool_use contract: content_block_start -> input_json_delta(s)
    # -> content_block_stop. Skipping the delta would drop tool arguments.
    assert next(first_wrapper)["type"] == "message_start"
    start_chunk = next(first_wrapper)
    assert start_chunk["type"] == "content_block_start"
    assert start_chunk["content_block"]["type"] == "tool_use"
    assert start_chunk["content_block"]["name"] == "Read"
    delta_chunk = next(first_wrapper)
    assert delta_chunk["type"] == "content_block_delta"
    assert delta_chunk["delta"]["type"] == "input_json_delta"
    assert '"file_path"' in delta_chunk["delta"]["partial_json"]
    assert next(first_wrapper)["type"] == "content_block_stop"

    # Instance isolation: a second wrapper must not observe first_wrapper's queue.
    second_wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

    assert next(second_wrapper)["type"] == "message_start"


def test_sync_wrapper_preserves_read_tool_after_thinking_block():
    class MockCompletionStreamWithThinkingThenRead:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                reasoning_content="I need to read the plan.",
                                thinking_blocks=[
                                    {
                                        "type": "thinking",
                                        "thinking": "I need to read the plan.",
                                        "signature": None,
                                    }
                                ],
                                content="",
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionDeltaToolCall(
                                        id="call_read__thought__sig_123",
                                        function=Function(
                                            name="read_file",
                                            arguments='{"file_path": "/tmp/plan.md"}',
                                        ),
                                        type="function",
                                        index=0,
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=""),
                            index=0,
                            finish_reason="tool_calls",
                        )
                    ],
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithThinkingThenRead(),
        model="gemini-3.1-pro-preview",
        tool_name_mapping={"read_file": "Read"},
    )

    chunks = list(wrapper)
    tool_start = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_start"
        and chunk["content_block"]["type"] == "tool_use"
    )
    tool_delta = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "input_json_delta"
    )
    thinking_deltas = [
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] in {"thinking_delta", "signature_delta"}
    ]

    assert tool_start["content_block"]["id"] == "call_read"
    assert tool_start["content_block"]["name"] == "Read"
    assert tool_delta["delta"]["partial_json"] == '{"file_path": "/tmp/plan.md"}'
    assert thinking_deltas == []


def test_sync_wrapper_starts_thinking_block_for_reasoning_content():
    class MockCompletionStreamWithReasoningContent:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                reasoning_content="I should answer directly.",
                                content="",
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Done"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=""),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithReasoningContent(),
        model="claude-sonnet-4-6",
    )

    chunks = list(wrapper)
    first_start = next(
        chunk for chunk in chunks if chunk["type"] == "content_block_start"
    )
    first_delta = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "thinking_delta"
    )

    assert first_start["content_block"] == {
        "type": "thinking",
        "thinking": "",
        "signature": "",
    }
    assert first_delta["index"] == first_start["index"]
    assert first_delta["delta"]["thinking"] == "I should answer directly."


def test_sync_wrapper_preserves_first_text_delta_after_reasoning():
    class MockCompletionStreamWithReasoningThenTimestamp:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                reasoning_content="I should return the timestamp.",
                                content="",
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="2026"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="-07-20T23:03:19-04:00"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=""),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    chunks = list(
        AnthropicStreamWrapper(
            completion_stream=MockCompletionStreamWithReasoningThenTimestamp(),
            model="alibaba_token_plan/qwen3.8-max-preview",
        )
    )
    text = "".join(
        chunk["delta"]["text"]
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "text_delta"
    )

    assert text == "2026-07-20T23:03:19-04:00"


@pytest.mark.asyncio
async def test_async_wrapper_preserves_first_text_delta_after_reasoning():
    class AsyncMockCompletionStreamWithReasoningThenTimestamp:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                reasoning_content="I should return the timestamp.",
                                content="",
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="2026"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="-07-20T23:03:19-04:00"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=""),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                ),
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.responses):
                raise StopAsyncIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    chunks = []
    async for chunk in AnthropicStreamWrapper(
        completion_stream=AsyncMockCompletionStreamWithReasoningThenTimestamp(),
        model="alibaba_token_plan/qwen3.8-max-preview",
    ):
        chunks.append(chunk)
    text = "".join(
        chunk["delta"]["text"]
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "text_delta"
    )

    assert text == "2026-07-20T23:03:19-04:00"


def test_sync_wrapper_preserves_terminal_tool_call_after_text_delta():
    class MockCompletionStreamWithTextThenTerminalTool:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                content="I will check the current working directory.\n"
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionDeltaToolCall(
                                        id="call_bash__thought__sig_123",
                                        function=Function(
                                            name="run_shell_command",
                                            arguments='{"command": "pwd"}',
                                        ),
                                        type="function",
                                        index=0,
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason="tool_calls",
                        )
                    ],
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithTextThenTerminalTool(),
        model="gemini-3.1-pro-preview",
        tool_name_mapping={"run_shell_command": "Bash"},
    )

    chunks = list(wrapper)
    tool_start = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_start"
        and chunk["content_block"]["type"] == "tool_use"
    )
    tool_delta = next(
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_delta"
        and chunk["delta"]["type"] == "input_json_delta"
    )
    message_delta = next(
        chunk for chunk in chunks if chunk["type"] == "message_delta"
    )

    assert tool_start["content_block"]["id"] == "call_bash"
    assert tool_start["content_block"]["name"] == "Bash"
    assert tool_delta["delta"]["partial_json"] == '{"command": "pwd"}'
    assert message_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_async_wrapper_synthesizes_end_turn_when_stream_ends_without_terminal_chunk():
    class AsyncMockCompletionStreamNoTerminal:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Hello"), index=0, finish_reason=None
                        )
                    ],
                ),
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.responses):
                raise StopAsyncIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockCompletionStreamNoTerminal(), model="gemini-3-flash-preview"
    )

    chunks = []
    async for chunk in wrapper:
        chunks.append(chunk)

    chunk_types = [chunk["type"] for chunk in chunks]
    assert "content_block_stop" in chunk_types
    assert "message_delta" in chunk_types
    assert chunk_types[-1] == "message_stop"

    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"
