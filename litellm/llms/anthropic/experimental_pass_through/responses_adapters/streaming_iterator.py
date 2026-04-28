# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import Any, AsyncIterator, Dict, Optional

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    sanitize_anthropic_tool_use_input,
    sanitize_anthropic_tool_use_input_json_delta,
)


class AnthropicResponsesStreamWrapper:
    """
    Wraps a Responses API streaming iterator and re-emits events in Anthropic SSE format.

    Responses API event flow (relevant subset):
      response.created                   -> message_start
      response.output_item.added         -> content_block_start (if message/function_call)
      response.output_text.delta         -> content_block_delta (text_delta)
      response.reasoning_summary_text.delta -> content_block_delta (thinking_delta)
      response.function_call_arguments.delta -> content_block_delta (input_json_delta)
      response.output_item.done          -> content_block_stop
      response.completed                 -> message_delta + message_stop
    """

    def __init__(
        self,
        responses_stream: Any,
        model: str,
        request_body: Optional[dict[str, Any]] = None,
    ) -> None:
        self.responses_stream = responses_stream
        self.model = model
        self.request_body = request_body or {}
        self._message_id: str = f"msg_{uuid.uuid4()}"
        self._current_block_index: int = -1
        # Map item_id -> content_block_index so we can stop the right block later
        self._item_id_to_block_index: Dict[str, int] = {}
        # Track open function_call items by item_id so we can emit tool_use start
        self._pending_tool_ids: Dict[
            str, str
        ] = {}  # item_id -> call_id / name accumulator
        self._pending_tool_names: Dict[str, str] = {}
        self._tool_inputs_seeded: set[str] = set()
        self._tool_argument_deltas_seen: set[str] = set()
        self._read_tool_argument_buffers: Dict[str, str] = {}
        self._sent_message_start = False
        self._sent_message_stop = False
        self._chunk_queue: deque = deque()
        self._output_text_parts: list[str] = []
        self._text_delta_keys_seen: set[str] = set()

    @staticmethod
    def _estimate_token_count(value: Any) -> int:
        if value is None:
            return 1
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                text = str(value)
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _event_text_key(event: Any) -> str:
        item_id = getattr(event, "item_id", None) or (
            event.get("item_id") if isinstance(event, dict) else None
        )
        if isinstance(item_id, str) and item_id:
            return item_id
        output_index = getattr(event, "output_index", None) or (
            event.get("output_index") if isinstance(event, dict) else None
        )
        if isinstance(output_index, int):
            return f"output:{output_index}"
        return "output:0"

    def _get_or_create_text_block_index(self, item_id: Any = None) -> int:
        if isinstance(item_id, str) and item_id in self._item_id_to_block_index:
            return self._item_id_to_block_index[item_id]
        if self._current_block_index >= 0 and not isinstance(item_id, str):
            return self._current_block_index

        block_idx = self._next_block_index()
        if isinstance(item_id, str) and item_id:
            self._item_id_to_block_index[item_id] = block_idx
        self._chunk_queue.append(
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            }
        )
        return block_idx

    def _build_fallback_usage_delta(self) -> Dict[str, Any]:
        return {
            "input_tokens": self._estimate_token_count(self.request_body),
            "output_tokens": self._estimate_token_count(
                "".join(self._output_text_parts)
            ),
        }

    def _queue_synthetic_message_stop(self) -> None:
        self._chunk_queue.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": self._build_fallback_usage_delta(),
            }
        )
        self._chunk_queue.append({"type": "message_stop"})
        self._sent_message_stop = True

    @staticmethod
    def _deserialize_tool_input(arguments: Any) -> Dict[str, Any]:
        if arguments in (None, ""):
            return {}
        if isinstance(arguments, dict):
            return arguments
        if hasattr(arguments, "model_dump"):
            dumped = arguments.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _make_message_start(self) -> Dict[str, Any]:
        return {
            "type": "message_start",
            "message": {
                "id": self._message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }

    def _next_block_index(self) -> int:
        self._current_block_index += 1
        return self._current_block_index

    def _process_event(self, event: Any) -> None:  # noqa: PLR0915
        """Convert one Responses API event into zero or more Anthropic chunks queued for emission."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type")

        if event_type is None:
            return

        # ---- message_start ----
        if event_type == "response.created":
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return

        # ---- content_block_start for a new output message item ----
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None) or (
                event.get("item") if isinstance(event, dict) else None
            )
            if item is None:
                return
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            item_id = getattr(item, "id", None) or (
                item.get("id") if isinstance(item, dict) else None
            )

            if item_type == "message":
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            elif item_type == "function_call":
                call_id = (
                    getattr(item, "call_id", None)
                    or (item.get("call_id") if isinstance(item, dict) else None)
                    or ""
                )
                name = (
                    getattr(item, "name", None)
                    or (item.get("name") if isinstance(item, dict) else None)
                    or ""
                )
                input_data = self._deserialize_tool_input(
                    getattr(item, "arguments", None)
                    or (item.get("arguments") if isinstance(item, dict) else None)
                )
                input_data = sanitize_anthropic_tool_use_input(
                    tool_name=name,
                    tool_input=input_data,
                )
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                    self._pending_tool_ids[item_id] = call_id
                    self._pending_tool_names[item_id] = name
                    if input_data:
                        self._tool_inputs_seeded.add(item_id)
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": input_data,
                        },
                    }
                )
            elif item_type == "mcp_call":
                call_id = (
                    getattr(item, "id", None)
                    or (item.get("id") if isinstance(item, dict) else None)
                    or ""
                )
                name = (
                    getattr(item, "name", None)
                    or (item.get("name") if isinstance(item, dict) else None)
                    or ""
                )
                server_label = (
                    getattr(item, "server_label", None)
                    or (item.get("server_label") if isinstance(item, dict) else None)
                    or ""
                )
                input_data = self._deserialize_tool_input(
                    getattr(item, "arguments", None)
                    or (item.get("arguments") if isinstance(item, dict) else None)
                )
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                    if input_data:
                        self._tool_inputs_seeded.add(item_id)
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "mcp_tool_use",
                            "id": call_id,
                            "name": name,
                            "server_name": server_label,
                            "input": input_data,
                        },
                    }
                )
            elif item_type == "reasoning":
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                )
            return

        # ---- text delta ----
        if event_type == "response.output_text.delta":
            item_id = getattr(event, "item_id", None) or (
                event.get("item_id") if isinstance(event, dict) else None
            )
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            if isinstance(delta, str) and delta:
                self._output_text_parts.append(delta)
                self._text_delta_keys_seen.add(self._event_text_key(event))
            block_idx = self._get_or_create_text_block_index(item_id)
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": delta},
                }
            )
            return

        # Some OpenRouter Responses streams omit `response.completed` and only
        # provide the final text in `response.output_text.done`.
        if event_type == "response.output_text.done":
            item_id = getattr(event, "item_id", None) or (
                event.get("item_id") if isinstance(event, dict) else None
            )
            text = getattr(event, "text", None) or (
                event.get("text") if isinstance(event, dict) else None
            )
            text_key = self._event_text_key(event)
            if not isinstance(text, str) or not text or text_key in self._text_delta_keys_seen:
                return
            self._output_text_parts.append(text)
            block_idx = self._get_or_create_text_block_index(item_id)
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": text},
                }
            )
            return

        # ---- reasoning summary text delta ----
        if event_type == "response.reasoning_summary_text.delta":
            item_id = getattr(event, "item_id", None) or (
                event.get("item_id") if isinstance(event, dict) else None
            )
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "thinking_delta", "thinking": delta},
                }
            )
            return

        # ---- function call arguments delta ----
        if event_type == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None) or (
                event.get("item_id") if isinstance(event, dict) else None
            )
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            tool_name = (
                self._pending_tool_names.get(item_id, "") if item_id else ""
            )
            if item_id and tool_name == "Read":
                buffered_delta = self._read_tool_argument_buffers.get(item_id, "") + delta
                try:
                    json.loads(buffered_delta)
                except json.JSONDecodeError:
                    self._read_tool_argument_buffers[item_id] = buffered_delta
                    return
                delta = sanitize_anthropic_tool_use_input_json_delta(
                    tool_name=tool_name,
                    partial_json=buffered_delta,
                )
                self._read_tool_argument_buffers.pop(item_id, None)
                self._tool_argument_deltas_seen.add(item_id)
            elif item_id:
                self._tool_argument_deltas_seen.add(item_id)
            delta = sanitize_anthropic_tool_use_input_json_delta(
                tool_name=tool_name,
                partial_json=delta,
            )
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            return

        # ---- MCP call arguments delta ----
        if event_type == "response.mcp_call_arguments.delta":
            item_id = getattr(event, "item_id", None) or (
                event.get("item_id") if isinstance(event, dict) else None
            )
            if item_id:
                self._tool_argument_deltas_seen.add(item_id)
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            return

        # ---- MCP call completed -> close mcp_tool_use and emit mcp_tool_result ----
        if event_type == "response.mcp_call.completed":
            item = getattr(event, "item", None) or (
                event.get("item") if isinstance(event, dict) else None
            )
            item_id = (
                getattr(item, "id", None)
                or (item.get("id") if isinstance(item, dict) else None)
                if item
                else None
            )
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append({"type": "content_block_stop", "index": block_idx})

            output = getattr(item, "output", None) or (
                item.get("output") if isinstance(item, dict) else None
            )
            error = getattr(item, "error", None) or (
                item.get("error") if isinstance(item, dict) else None
            )
            if isinstance(output, list):
                output_text = "\n".join(
                    part.get("text", "")
                    for part in output
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            elif output is None:
                output_text = ""
            elif isinstance(output, dict):
                output_text = json.dumps(output)
            else:
                output_text = str(output)
            if error is not None and not output_text:
                output_text = str(error)

            result_block_idx = self._next_block_index()
            self._chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": result_block_idx,
                    "content_block": {
                        "type": "mcp_tool_result",
                        "tool_use_id": str(item_id or ""),
                        "is_error": error is not None,
                        "content": [{"type": "text", "text": output_text}],
                    },
                }
            )
            self._chunk_queue.append(
                {"type": "content_block_stop", "index": result_block_idx}
            )
            return

        # ---- output item done -> content_block_stop ----
        if event_type == "response.output_item.done":
            item = getattr(event, "item", None) or (
                event.get("item") if isinstance(event, dict) else None
            )
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            item_id = (
                getattr(item, "id", None)
                or (item.get("id") if isinstance(item, dict) else None)
                if item
                else None
            )
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            if item_type in {"function_call", "mcp_call"} and item_id:
                input_data = self._deserialize_tool_input(
                    getattr(item, "arguments", None)
                    or (item.get("arguments") if isinstance(item, dict) else None)
                )
                if item_type == "function_call":
                    tool_name = (
                        getattr(item, "name", None)
                        or (item.get("name") if isinstance(item, dict) else None)
                        or self._pending_tool_names.get(item_id, "")
                    )
                    input_data = sanitize_anthropic_tool_use_input(
                        tool_name=tool_name,
                        tool_input=input_data,
                    )
                if (
                    input_data
                    and item_id not in self._tool_argument_deltas_seen
                    and item_id not in self._tool_inputs_seeded
                ):
                    self._chunk_queue.append(
                        {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(input_data),
                            },
                        }
                    )
            if item_type == "mcp_call":
                return
            self._chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": block_idx,
                }
            )
            return

        # ---- response completed -> message_delta + message_stop ----
        if event_type in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            response_obj = getattr(event, "response", None) or (
                event.get("response") if isinstance(event, dict) else None
            )
            stop_reason = "end_turn"
            input_tokens = 0
            output_tokens = 0
            cache_creation_tokens = 0
            cache_read_tokens = 0

            if response_obj is not None:
                status = getattr(response_obj, "status", None)
                if status == "incomplete":
                    stop_reason = "max_tokens"
                usage = getattr(response_obj, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    cache_creation_tokens = getattr(usage, "input_tokens_details", None)  # type: ignore[assignment]
                    cache_read_tokens = getattr(usage, "output_tokens_details", None)  # type: ignore[assignment]
                    # Prefer direct cache fields if present
                    cache_creation_tokens = int(
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                    cache_read_tokens = int(
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    )

            # Check if tool_use was in the output to override stop_reason
            if response_obj is not None:
                output = getattr(response_obj, "output", []) or []
                for out_item in output:
                    out_type = getattr(out_item, "type", None) or (
                        out_item.get("type") if isinstance(out_item, dict) else None
                    )
                    if out_type == "function_call":
                        stop_reason = "tool_use"
                        break

            usage_delta: Dict[str, Any] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if cache_creation_tokens:
                usage_delta["cache_creation_input_tokens"] = cache_creation_tokens
            if cache_read_tokens:
                usage_delta["cache_read_input_tokens"] = cache_read_tokens

            self._chunk_queue.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage_delta,
                }
            )
            self._chunk_queue.append({"type": "message_stop"})
            self._sent_message_stop = True
            return

    def __aiter__(self) -> "AnthropicResponsesStreamWrapper":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        # Return any queued chunks first
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        # Consume the upstream stream
        try:
            async for event in self.responses_stream:
                self._process_event(event)
                if self._chunk_queue:
                    first_chunk = self._chunk_queue[0]
                    if (
                        not self._sent_message_start
                        and isinstance(first_chunk, dict)
                        and first_chunk.get("type") != "message_start"
                    ):
                        self._sent_message_start = True
                        self._chunk_queue.appendleft(self._make_message_start())
                    return self._chunk_queue.popleft()
        except StopAsyncIteration:
            pass
        except Exception as e:
            verbose_logger.error(
                f"AnthropicResponsesStreamWrapper error: {e}\n{traceback.format_exc()}"
            )

        # Drain any remaining queued chunks
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        # If the upstream stream never emitted response.created but did yield no data,
        # synthesize a single message_start as the minimal Anthropic envelope.
        if not self._sent_message_start and not self._sent_message_stop:
            self._sent_message_start = True
            return self._make_message_start()

        if self._sent_message_start and not self._sent_message_stop:
            self._queue_synthetic_message_stop()
            return self._chunk_queue.popleft()

        raise StopAsyncIteration

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """Yield SSE-encoded bytes for each Anthropic event chunk."""
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk
