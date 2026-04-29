# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import os
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Literal, Optional

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    from litellm.types.llms.anthropic import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use", "thinking"] = "text"
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None
    holding_stop_reason_chunk: Optional[Any] = None
    queued_usage_chunk: bool = False
    current_content_block_index: int = 0
    current_content_block_start: ContentBlockContentBlockDict = TextBlock(
        type="text",
        text="",
    )
    chunk_queue: deque = deque()  # Queue for buffering multiple chunks
    buffered_tool_calls: Dict[int, Dict[str, Any]]
    tool_call_content_block_indices: Dict[Any, int]
    tool_call_names: Dict[Any, str]

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(completion_stream)
        self.model = model
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}
        self.sent_first_chunk = False
        self.sent_content_block_start = False
        self.sent_content_block_finish = False
        self.current_content_block_type = "text"
        self.sent_last_message = False
        self.holding_chunk = None
        self.holding_stop_reason_chunk = None
        self.queued_usage_chunk = False
        self.current_content_block_index = 0
        self.current_content_block_start = self.TextBlock(type="text", text="")
        self.chunk_queue = deque()
        self.buffered_tool_calls = {}
        self.tool_call_content_block_indices = {}
        self.tool_call_names = {}

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        Create the initial UsageDelta for the message_start event.

        Initializes cache token fields (cache_creation_input_tokens, cache_read_input_tokens)
        to 0 to indicate to clients (like Claude Code) that prompt caching is supported.

        The actual cache token values will be provided in the message_delta event at the
        end of the stream, since Bedrock Converse API only returns usage data in the final
        response chunk.

        Returns:
            UsageDelta with all token counts initialized to 0.
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def _queue_synthetic_end_turn_if_needed(self) -> None:
        """
        Some upstream streaming providers can end a text response without a final
        stop_reason chunk. Anthropic clients still expect the assistant turn to
        close cleanly with `content_block_stop` and `message_delta(stop_reason=end_turn)`
        before `message_stop`.
        """
        if self.sent_content_block_start is False:
            return
        if self.sent_content_block_finish is True:
            return
        if self.holding_stop_reason_chunk is not None:
            return

        self.chunk_queue.append(
            {
                "type": "content_block_stop",
                "index": self.current_content_block_index,
            }
        )
        self.chunk_queue.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            }
        )
        self.sent_content_block_finish = True

    def _sanitize_tool_use_delta_if_needed(self, processed_chunk: Any) -> Any:
        if not isinstance(processed_chunk, dict):
            return processed_chunk
        if processed_chunk.get("type") != "content_block_delta":
            return processed_chunk
        delta = processed_chunk.get("delta")
        if not isinstance(delta, dict) or delta.get("type") != "input_json_delta":
            return processed_chunk
        partial_json = delta.get("partial_json")
        if not isinstance(partial_json, str):
            return processed_chunk
        if not isinstance(self.current_content_block_start, dict):
            return processed_chunk
        if self.current_content_block_start.get("type") != "tool_use":
            return processed_chunk
        tool_name = self.current_content_block_start.get("name")
        if not isinstance(tool_name, str):
            return processed_chunk

        from .transformation import sanitize_anthropic_tool_use_input_json_delta

        sanitized_partial_json = sanitize_anthropic_tool_use_input_json_delta(
            tool_name=tool_name,
            partial_json=partial_json,
        )
        if sanitized_partial_json == partial_json:
            return processed_chunk
        sanitized_delta = dict(delta)
        sanitized_delta["partial_json"] = sanitized_partial_json
        sanitized_chunk = dict(processed_chunk)
        sanitized_chunk["delta"] = sanitized_delta
        return sanitized_chunk

    def _should_suppress_provider_thinking_blocks(self) -> bool:
        return "gemini" in self.model.lower()

    def _should_suppress_provider_thinking_delta(self, processed_chunk: Any) -> bool:
        if not self._should_suppress_provider_thinking_blocks():
            return False
        if not isinstance(processed_chunk, dict):
            return False
        if processed_chunk.get("type") != "content_block_delta":
            return False
        delta = processed_chunk.get("delta")
        if not isinstance(delta, dict):
            return False
        return delta.get("type") in {"thinking_delta", "signature_delta"}

    @staticmethod
    def _chunk_has_tool_call_delta(chunk: "ModelResponseStream") -> bool:
        try:
            tool_calls = chunk.choices[0].delta.tool_calls
        except (AttributeError, IndexError, TypeError):
            return False
        if not tool_calls:
            return False
        return any(
            getattr(tool_call, "function", None) is not None
            for tool_call in tool_calls
        )

    @staticmethod
    def _chunk_has_content_block_signal(chunk: "ModelResponseStream") -> bool:
        try:
            delta = chunk.choices[0].delta
        except (AttributeError, IndexError, TypeError):
            return False

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            return True

        tool_calls = getattr(delta, "tool_calls", None) or []
        if any(
            getattr(tool_call, "function", None) is not None
            for tool_call in tool_calls
        ):
            return True

        thinking_blocks = getattr(delta, "thinking_blocks", None) or []
        for thinking_block in thinking_blocks:
            if not isinstance(thinking_block, dict):
                continue
            if thinking_block.get("thinking") or thinking_block.get("signature"):
                return True

        reasoning_content = getattr(delta, "reasoning_content", None)
        return isinstance(reasoning_content, str) and bool(reasoning_content)

    @staticmethod
    def _get_tool_call_stream_key(tool_call: Any, ordinal: int) -> Any:
        raw_index = getattr(tool_call, "index", None)
        if isinstance(raw_index, int):
            return ("index", raw_index)
        tool_call_id = getattr(tool_call, "id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            return ("id", tool_call_id)
        return ("ordinal", ordinal)

    def _restore_tool_name(self, tool_name: str) -> str:
        return self.tool_name_mapping.get(tool_name, tool_name)

    def _queue_initial_content_block_for_chunk(
        self, chunk: "ModelResponseStream"
    ) -> bool:
        if self.sent_content_block_start:
            return False
        if not self._chunk_has_content_block_signal(chunk):
            return False

        from .transformation import LiteLLMAnthropicMessagesAdapter

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )
        if block_type == "thinking" and self._should_suppress_provider_thinking_blocks():
            return False

        if block_type == "tool_use":
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)
            if tool_block.get("name"):
                tool_block["name"] = self._restore_tool_name(tool_block["name"])

        self.current_content_block_type = block_type
        self.current_content_block_start = content_block_start
        self.sent_content_block_start = True
        self.sent_content_block_finish = False
        self.chunk_queue.append(
            {
                "type": "content_block_start",
                "index": self.current_content_block_index,
                "content_block": self.current_content_block_start,
            }
        )
        return True

    def _queue_streaming_tool_calls_by_index(
        self, chunk: "ModelResponseStream", hold_terminal_message_delta: bool = False
    ) -> bool:
        try:
            choice = chunk.choices[0]
            tool_calls = choice.delta.tool_calls or []
        except (AttributeError, IndexError, TypeError):
            return False

        if not tool_calls and not self.tool_call_content_block_indices:
            return False
        if len(tool_calls) <= 1 and not self.tool_call_content_block_indices:
            return False

        from .transformation import sanitize_anthropic_tool_use_id

        queued_any_tool_delta = False
        for ordinal, tool_call in enumerate(tool_calls):
            function = getattr(tool_call, "function", None)
            if function is None:
                continue

            key = self._get_tool_call_stream_key(tool_call, ordinal)
            block_index = self.tool_call_content_block_indices.get(key)
            function_name = getattr(function, "name", None)
            if block_index is None:
                if (
                    self.sent_content_block_start
                    and self.current_content_block_type != "tool_use"
                    and self.sent_content_block_finish is False
                ):
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True

                if self.sent_content_block_start:
                    self._increment_content_block_index()
                else:
                    self.sent_content_block_start = True

                block_index = self.current_content_block_index
                self.tool_call_content_block_indices[key] = block_index
                original_tool_name = self._restore_tool_name(function_name or "")
                self.tool_call_names[key] = original_tool_name
                self.current_content_block_type = "tool_use"
                self.current_content_block_start = {
                    "type": "tool_use",
                    "id": sanitize_anthropic_tool_use_id(getattr(tool_call, "id", None))
                    or str(uuid.uuid4()),
                    "name": original_tool_name,
                    "input": {},
                }
                self.sent_content_block_finish = False
                self.chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": self.current_content_block_start,
                    }
                )

            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str) and arguments:
                from .transformation import sanitize_anthropic_tool_use_input_json_delta

                tool_name = self.tool_call_names.get(key, function_name or "")
                self.chunk_queue.append(
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": sanitize_anthropic_tool_use_input_json_delta(
                                tool_name=tool_name,
                                partial_json=arguments,
                            ),
                        },
                    }
                )
            queued_any_tool_delta = True

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            for _, block_index in sorted(
                self.tool_call_content_block_indices.items(),
                key=lambda item: item[1],
            ):
                self.chunk_queue.append(
                    {
                        "type": "content_block_stop",
                        "index": block_index,
                    }
            )
            self.tool_call_content_block_indices.clear()
            self.tool_call_names.clear()
            self.sent_content_block_finish = True
            from .transformation import LiteLLMAnthropicMessagesAdapter

            message_delta = (
                LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )
            )
            if hold_terminal_message_delta:
                self.holding_stop_reason_chunk = message_delta
            else:
                self.chunk_queue.append(message_delta)
            return True

        return queued_any_tool_delta

    def _should_buffer_gemini_tool_calls(self, chunk: "ModelResponseStream") -> bool:
        return "gemini" in self.model.lower() and (
            self._chunk_has_tool_call_delta(chunk) or bool(self.buffered_tool_calls)
        )

    def _buffer_tool_call_delta(self, chunk: "ModelResponseStream") -> None:
        try:
            tool_calls = chunk.choices[0].delta.tool_calls or []
        except (AttributeError, IndexError, TypeError):
            return

        for ordinal, tool_call in enumerate(tool_calls):
            function = getattr(tool_call, "function", None)
            if function is None:
                continue
            raw_index = getattr(tool_call, "index", None)
            index = raw_index if isinstance(raw_index, int) else ordinal
            buffered = self.buffered_tool_calls.setdefault(
                index,
                {
                    "id": None,
                    "name": None,
                    "arguments": "",
                },
            )
            tool_call_id = getattr(tool_call, "id", None)
            if tool_call_id:
                buffered["id"] = tool_call_id
            function_name = getattr(function, "name", None)
            if function_name:
                buffered["name"] = function_name
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str) and arguments:
                buffered["arguments"] += arguments

    def _queue_buffered_gemini_tool_calls(
        self, chunk: "ModelResponseStream"
    ) -> bool:
        if not self._should_buffer_gemini_tool_calls(chunk):
            return False

        self._buffer_tool_call_delta(chunk)
        finish_reason = getattr(chunk.choices[0], "finish_reason", None)
        if finish_reason is None:
            return True
        if not self.buffered_tool_calls:
            return False

        from .transformation import (
            LiteLLMAnthropicMessagesAdapter,
            sanitize_anthropic_tool_use_id,
            sanitize_anthropic_tool_use_input_json_delta,
        )

        if self.sent_content_block_start and self.sent_content_block_finish is False:
            self.chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": self.current_content_block_index,
                }
            )

        for _, tool_call in sorted(self.buffered_tool_calls.items()):
            tool_name = tool_call.get("name") or ""
            original_tool_name = self.tool_name_mapping.get(tool_name, tool_name)
            self._increment_content_block_index()
            self.current_content_block_type = "tool_use"
            self.current_content_block_start = {
                "type": "tool_use",
                "id": sanitize_anthropic_tool_use_id(tool_call.get("id"))
                or str(uuid.uuid4()),
                "name": original_tool_name,
                "input": {},
            }
            self.chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": self.current_content_block_index,
                    "content_block": self.current_content_block_start,
                }
            )
            arguments = tool_call.get("arguments")
            if isinstance(arguments, str) and arguments:
                self.chunk_queue.append(
                    {
                        "type": "content_block_delta",
                        "index": self.current_content_block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": sanitize_anthropic_tool_use_input_json_delta(
                                tool_name=original_tool_name,
                                partial_json=arguments,
                            ),
                        },
                    }
                )
            self.chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": self.current_content_block_index,
                }
            )

        self.buffered_tool_calls.clear()
        self.sent_content_block_finish = True
        message_delta = (
            LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                response=chunk,
                current_content_block_index=self.current_content_block_index,
            )
        )
        self.chunk_queue.append(message_delta)
        return True

    def _translate_terminal_tool_call_chunk(
        self,
        chunk: "ModelResponseStream",
        current_content_block_index: int,
    ) -> Any:
        from .transformation import LiteLLMAnthropicMessagesAdapter

        adapter = LiteLLMAnthropicMessagesAdapter()
        terminal_message_delta = adapter.translate_streaming_openai_response_to_anthropic(
            response=chunk,
            current_content_block_index=current_content_block_index,
        )
        if (
            isinstance(terminal_message_delta, dict)
            and terminal_message_delta.get("type") == "message_delta"
        ):
            self.holding_stop_reason_chunk = terminal_message_delta

        (
            _type_of_content,
            content_block_delta,
        ) = adapter._translate_streaming_openai_chunk_to_anthropic(
            choices=chunk.choices  # type: ignore[arg-type]
        )
        return {
            "type": "content_block_delta",
            "index": current_content_block_index,
            "delta": content_block_delta,
        }

    def _queue_held_stop_reason_if_needed(self) -> bool:
        if self.holding_stop_reason_chunk is None:
            return False
        if self.sent_content_block_finish is False:
            self.chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": self.current_content_block_index,
                }
            )
            self.sent_content_block_finish = True
        self.chunk_queue.append(self.holding_stop_reason_chunk)
        self.holding_stop_reason_chunk = None
        return True

    def __next__(self):  # noqa: PLR0915
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                if self._queue_buffered_gemini_tool_calls(chunk):
                    if self.chunk_queue:
                        return self.chunk_queue.popleft()
                    continue

                if self._queue_streaming_tool_calls_by_index(chunk):
                    if self.chunk_queue:
                        return self.chunk_queue.popleft()
                    continue

                if self._queue_initial_content_block_for_chunk(chunk):
                    should_start_new_block = False
                elif self.sent_content_block_start is False:
                    continue
                else:
                    should_start_new_block = self._should_start_new_content_block(chunk)
                    if should_start_new_block:
                        self._increment_content_block_index()


                if (
                    getattr(chunk.choices[0], "finish_reason", None) is not None
                    and self._chunk_has_tool_call_delta(chunk)
                ):
                    processed_chunk = self._translate_terminal_tool_call_chunk(
                        chunk=chunk,
                        current_content_block_index=self.current_content_block_index,
                    )
                else:
                    processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                        response=chunk,
                        current_content_block_index=self.current_content_block_index,
                    )
                processed_chunk = self._sanitize_tool_use_delta_if_needed(
                    processed_chunk
                )
                if self._should_suppress_provider_thinking_delta(processed_chunk):
                    continue

                if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1" and "gemini" in self.model:
                    try:
                        choice = chunk.choices[0]
                        delta = getattr(choice, "delta", None)
                        verbose_logger.warning(
                            "Anthropic wrapper debug(sync): model=%s raw_finish=%s raw_content=%r raw_reasoning=%r raw_tool_calls=%s translated_type=%s translated_delta=%s",
                            self.model,
                            getattr(choice, "finish_reason", None),
                            getattr(delta, "content", None),
                            getattr(delta, "reasoning_content", None),
                            len(getattr(delta, "tool_calls", None) or []),
                            processed_chunk.get("type") if isinstance(processed_chunk, dict) else type(processed_chunk).__name__,
                            processed_chunk.get("delta") if isinstance(processed_chunk, dict) else None,
                        )
                    except Exception:
                        verbose_logger.exception("Anthropic wrapper sync debug logging failed")

                if should_start_new_block and not self.sent_content_block_finish:
                    # Queue the sequence: content_block_stop -> content_block_start.
                    # For tool_use blocks we must also preserve the triggering
                    # input_json_delta chunk; otherwise clients receive an empty
                    # tool input and invoke the tool with missing arguments.
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": max(self.current_content_block_index - 1, 0),
                        }
                    )
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )
                    if (
                        isinstance(processed_chunk, dict)
                        and processed_chunk.get("type") == "content_block_delta"
                        and isinstance(processed_chunk.get("delta"), dict)
                        and processed_chunk["delta"].get("type") == "input_json_delta"
                        and processed_chunk["delta"].get("partial_json")
                    ):
                        self.chunk_queue.append(processed_chunk)
                    self.sent_content_block_finish = False
                    return self.chunk_queue.popleft()

                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    # Queue both the content_block_stop and the message_delta
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()
                elif self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.chunk_queue.append(processed_chunk)
                    self.holding_chunk = None
                    return self.chunk_queue.popleft()
                else:
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if self.holding_chunk is not None:
                self.chunk_queue.append(self.holding_chunk)
                self.holding_chunk = None

            if not self._queue_held_stop_reason_if_needed():
                self._queue_synthetic_end_turn_if_needed()

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration
        except StopIteration:
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            if self.sent_last_message is False:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopIteration
        except Exception as e:
            verbose_logger.error(
                "Anthropic Adapter - {}\n{}".format(e, traceback.format_exc())
            )
            raise StopAsyncIteration

    async def __anext__(self):  # noqa: PLR0915
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                if self._queue_buffered_gemini_tool_calls(chunk):
                    if self.chunk_queue:
                        return self.chunk_queue.popleft()
                    continue

                if self._queue_streaming_tool_calls_by_index(
                    chunk, hold_terminal_message_delta=True
                ):
                    if self.chunk_queue:
                        return self.chunk_queue.popleft()
                    continue

                # Check if we need to start a new content block
                if self._queue_initial_content_block_for_chunk(chunk):
                    should_start_new_block = False
                elif self.sent_content_block_start is False:
                    continue
                else:
                    should_start_new_block = self._should_start_new_content_block(chunk)
                    if should_start_new_block:
                        self._increment_content_block_index()

                if (
                    getattr(chunk.choices[0], "finish_reason", None) is not None
                    and self._chunk_has_tool_call_delta(chunk)
                ):
                    processed_chunk = self._translate_terminal_tool_call_chunk(
                        chunk=chunk,
                        current_content_block_index=self.current_content_block_index,
                    )
                else:
                    processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                        response=chunk,
                        current_content_block_index=self.current_content_block_index,
                    )
                processed_chunk = self._sanitize_tool_use_delta_if_needed(
                    processed_chunk
                )
                if self._should_suppress_provider_thinking_delta(processed_chunk):
                    continue

                if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1" and "gemini" in self.model:
                    try:
                        choice = chunk.choices[0]
                        delta = getattr(choice, "delta", None)
                        verbose_logger.warning(
                            "Anthropic wrapper debug(async): model=%s raw_finish=%s raw_content=%r raw_reasoning=%r raw_tool_calls=%s translated_type=%s translated_delta=%s",
                            self.model,
                            getattr(choice, "finish_reason", None),
                            getattr(delta, "content", None),
                            getattr(delta, "reasoning_content", None),
                            len(getattr(delta, "tool_calls", None) or []),
                            processed_chunk.get("type") if isinstance(processed_chunk, dict) else type(processed_chunk).__name__,
                            processed_chunk.get("delta") if isinstance(processed_chunk, dict) else None,
                        )
                    except Exception:
                        verbose_logger.exception("Anthropic wrapper async debug logging failed")

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if (
                    self.holding_stop_reason_chunk is not None
                    and getattr(chunk, "usage", None) is not None
                    and not self._chunk_has_tool_call_delta(chunk)
                ):
                    # Merge usage into the held stop_reason chunk
                    merged_chunk = self.holding_stop_reason_chunk.copy()
                    if "delta" not in merged_chunk:
                        merged_chunk["delta"] = {}

                    # Add usage to the held chunk
                    uncached_input_tokens = chunk.usage.prompt_tokens or 0
                    if (
                        hasattr(chunk.usage, "prompt_tokens_details")
                        and chunk.usage.prompt_tokens_details
                    ):
                        cached_tokens = (
                            getattr(
                                chunk.usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        )
                        uncached_input_tokens -= cached_tokens

                    usage_dict: UsageDelta = {
                        "input_tokens": uncached_input_tokens,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                    # Add cache tokens if available (for prompt caching support)
                    if (
                        hasattr(chunk.usage, "_cache_creation_input_tokens")
                        and chunk.usage._cache_creation_input_tokens > 0
                    ):
                        usage_dict[
                            "cache_creation_input_tokens"
                        ] = chunk.usage._cache_creation_input_tokens
                    if (
                        hasattr(chunk.usage, "_cache_read_input_tokens")
                        and chunk.usage._cache_read_input_tokens > 0
                    ):
                        usage_dict[
                            "cache_read_input_tokens"
                        ] = chunk.usage._cache_read_input_tokens
                    merged_chunk["usage"] = usage_dict

                    # Queue the merged chunk and reset
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # Check if this processed chunk has a stop_reason - hold it for next chunk

                if not self.queued_usage_chunk:
                    if should_start_new_block and not self.sent_content_block_finish:
                        # Queue the sequence: content_block_stop -> content_block_start.
                        # For tool_use blocks we must also preserve the triggering
                        # input_json_delta chunk; otherwise clients receive an empty
                        # tool input and invoke the tool with missing arguments.

                        # 1. Stop current content block
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": max(self.current_content_block_index - 1, 0),
                            }
                        )

                        # 2. Start new content block
                        self.chunk_queue.append(
                            {
                                "type": "content_block_start",
                                "index": self.current_content_block_index,
                                "content_block": self.current_content_block_start,
                            }
                        )
                        if (
                            isinstance(processed_chunk, dict)
                            and processed_chunk.get("type") == "content_block_delta"
                            and isinstance(processed_chunk.get("delta"), dict)
                            and processed_chunk["delta"].get("type") == "input_json_delta"
                            and processed_chunk["delta"].get("partial_json")
                        ):
                            self.chunk_queue.append(processed_chunk)

                        # Reset state for new block
                        self.sent_content_block_finish = False

                        # Return the first queued item
                        return self.chunk_queue.popleft()

                    if (
                        processed_chunk["type"] == "message_delta"
                        and self.sent_content_block_finish is False
                    ):
                        # Queue both the content_block_stop and the holding chunk
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                        if (
                            processed_chunk.get("delta", {}).get("stop_reason")
                            is not None
                        ):
                            self.holding_stop_reason_chunk = processed_chunk
                        else:
                            self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()
                    elif self.holding_chunk is not None:
                        # Queue both chunks
                        self.chunk_queue.append(self.holding_chunk)
                        self.chunk_queue.append(processed_chunk)
                        self.holding_chunk = None
                        return self.chunk_queue.popleft()
                    else:
                        # Queue the current chunk
                        self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if not self.queued_usage_chunk:
                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None

                if not self._queue_held_stop_reason_if_needed():
                    self._queue_synthetic_end_turn_if_needed()

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration

        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk
            if self._queue_held_stop_reason_if_needed() and self.chunk_queue:
                return self.chunk_queue.popleft()
            if not self.sent_last_message:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopAsyncIteration

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        Similar to the Bedrock bedrock_sse_wrapper implementation.

        This wrapper ensures dict chunks are SSE formatted with both event and data lines.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of anthropic_sse_wrapper.
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    def _increment_content_block_index(self):
        self.current_content_block_index += 1

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        Determine if we should start a new content block based on the processed chunk.
        Override this method with your specific logic for detecting new content blocks.

        Examples of when you might want to start a new content block:
        - Switching from text to tool calls
        - Different content types in the response
        - Specific markers in the content
        """
        from .transformation import LiteLLMAnthropicMessagesAdapter

        # Example logic - customize based on your needs:
        # If chunk indicates a tool call
        if (
            chunk.choices[0].finish_reason is not None
            and not self._chunk_has_tool_call_delta(chunk)
        ):
            return False

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )
        if block_type == "thinking" and self._should_suppress_provider_thinking_blocks():
            return False

        # Restore original tool name if it was truncated for OpenAI's 64-char limit
        if block_type == "tool_use":
            # Type narrowing: content_block_start is ToolUseBlock when block_type is "tool_use"
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)

            if tool_block.get("name"):
                truncated_name = tool_block["name"]
                original_name = self.tool_name_mapping.get(
                    truncated_name, truncated_name
                )
                tool_block["name"] = original_name

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        # For parallel tool calls, we'll necessarily have a new content block
        # if we get a function name since it signals a new tool call
        if block_type == "tool_use":
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)
            if tool_block.get("name"):
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                return True

        return False
