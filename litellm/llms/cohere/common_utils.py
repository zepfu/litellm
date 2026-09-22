import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.cohere.cancellation import (
    aclose_upstream_response_once,
    close_upstream_response_once,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    ProviderSpecificModelInfo,
)


class CohereError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


class CohereModelInfo(BaseLLMModelInfo):
    def get_provider_info(
        self,
        model: str,
    ) -> Optional[ProviderSpecificModelInfo]:
        """
        Default values all models of this provider support.
        """
        return None

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Returns a list of models supported by this provider.
        """
        return []

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> Optional[str]:
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {}

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        """
        Returns the base model name from the given model name.

        Some providers like bedrock - can receive model=`invoke/anthropic.claude-3-opus-20240229-v1:0` or `converse/anthropic.claude-3-opus-20240229-v1:0`
            This function will return `anthropic.claude-3-opus-20240229-v1:0`
        """
        pass

    @staticmethod
    def get_cohere_route(model: str) -> Literal["v1", "v2"]:
        """
        Get the Cohere route for the given model.

        Args:
            model: The model name (e.g., "cohere_chat/v2/command-r-plus", "command-r-plus")

        Returns:
            "v2" for standard Cohere v2 API (default), "v1" for Cohere v1 API
        """
        # Check for explicit v1 route
        if "v1/" in model:
            return "v1"

        # Default to v2 for all other cases
        return "v2"


def validate_environment(
    headers: dict,
    model: str,
    messages: List[AllMessageValues],
    optional_params: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Return headers to use for cohere chat completion request

    Cohere API Ref: https://docs.cohere.com/reference/chat
    Expected headers:
    {
        "Request-Source": "unspecified:litellm",
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer $CO_API_KEY"
    }
    """
    headers.update(
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class ModelResponseIterator:
    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: List = []
        self.tool_index = -1
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            index = int(chunk.get("index", 0))

            if "text" in chunk:
                text = chunk["text"]
            elif "is_finished" in chunk and chunk["is_finished"] is True:
                is_finished = chunk["is_finished"]
                finish_reason = chunk["finish_reason"]

            if "citations" in chunk:
                provider_specific_fields = {"citations": chunk["citations"]}

            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )

            return returned_chunk

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    def convert_str_chunk_to_generic_chunk(self, chunk: str) -> GenericStreamingChunk:
        """
        Convert a string chunk to a GenericStreamingChunk

        Note: This is used for Cohere pass through streaming logging
        """
        str_line = chunk
        if isinstance(chunk, bytes):  # Handle binary data
            str_line = chunk.decode("utf-8")  # Convert bytes to string
            index = str_line.find("data:")
            if index != -1:
                str_line = str_line[index:]

        data_json = json.loads(str_line)
        return self.chunk_parser(chunk=data_json)

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")


class CohereV2ModelResponseIterator:
    """V2-specific response iterator for Cohere streaming"""

    _FINISH_REASON_MAP = {
        "COMPLETE": "stop",
        "STOP_SEQUENCE": "stop",
        "MAX_TOKENS": "length",
        "TOOL_CALL": "tool_calls",
        "ERROR": "error",
        "TIMEOUT": "error",
    }

    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: List = []
        self.tool_index = -1
        self.json_mode = json_mode
        self._tool_calls: Dict[int, Dict[str, Any]] = {}
        self._tool_call_indexes: Dict[str, int] = {}
        self._next_tool_index = 0
        self._pending_sse_payloads: List[str] = []
        self._message_end_received = False
        self._aawm_stream_closed = False

    def _cohere_upstream_target(self) -> Any:
        raw_response = getattr(self, "_aawm_raw_response", None)
        if raw_response is not None:
            return raw_response
        return self.streaming_response

    def close(self) -> None:
        if self._aawm_stream_closed:
            return
        target = self._cohere_upstream_target()
        if not callable(getattr(target, "close", None)):
            return
        self._aawm_stream_closed = True
        close_upstream_response_once(target)

    async def aclose(self) -> None:
        if self._aawm_stream_closed:
            return
        self._aawm_stream_closed = True
        await aclose_upstream_response_once(self._cohere_upstream_target())

    @staticmethod
    def _empty_chunk() -> GenericStreamingChunk:
        return GenericStreamingChunk(
            text="",
            tool_use=None,
            is_finished=False,
            finish_reason="",
            usage=None,
            index=0,
            provider_specific_fields=None,
        )

    @staticmethod
    def _stringify_tool_fragment(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _extract_sse_payloads(self, chunk: Union[str, bytes, dict]) -> List[str]:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        if isinstance(chunk, dict):
            return [json.dumps(chunk, ensure_ascii=False)]
        if not isinstance(chunk, str):
            raise ValueError(f"Unsupported Cohere stream chunk type: {type(chunk)}")

        if not chunk.strip():
            return []

        if not any(
            line.strip().startswith("data:") for line in chunk.splitlines()
        ):
            if all(
                not line.strip()
                or line.strip().startswith((":", "event:"))
                for line in chunk.splitlines()
            ):
                return []
            return [chunk.strip()]

        payloads: List[str] = []
        data_lines: List[str] = []

        def flush_event() -> None:
            if data_lines:
                payloads.append("\n".join(data_lines))
                data_lines.clear()

        for line in chunk.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                flush_event()
                continue
            if stripped_line.startswith(":") or stripped_line.startswith("event:"):
                continue
            if stripped_line.startswith("data:"):
                data_lines.append(stripped_line[5:].lstrip())

        flush_event()
        return payloads

    def _validate_stream_end(self) -> None:
        if self._pending_sse_payloads:
            raise ValueError("Cohere stream ended with unconsumed SSE data")
        if not self._message_end_received:
            raise ValueError("Cohere stream ended without a native message-end event")

    def _parse_sse_json(
        self, chunk: Optional[Union[str, bytes, dict]] = None
    ) -> Optional[dict]:
        if chunk is not None:
            self._pending_sse_payloads.extend(self._extract_sse_payloads(chunk))
        if not self._pending_sse_payloads:
            return None

        payload = self._pending_sse_payloads.pop(0).strip()
        if payload == "[DONE]":
            self._validate_stream_end()
            raise StopIteration

        try:
            parsed_chunk = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed Cohere stream JSON: {e.msg}") from e

        if not isinstance(parsed_chunk, dict):
            raise ValueError(f"Expected Cohere stream event object, got {parsed_chunk!r}")
        return parsed_chunk

    @staticmethod
    def _tool_call_entries(chunk: dict) -> List[dict]:
        delta = chunk.get("delta", {}) or {}
        message = delta.get("message", {}) or {}
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, dict):
            return [tool_calls]
        if isinstance(tool_calls, list):
            return [call for call in tool_calls if isinstance(call, dict)]
        return []

    def _resolve_tool_index(self, chunk: dict, tool_call: dict, position: int = 0) -> int:
        tool_id = tool_call.get("id") or tool_call.get("call_id")
        if isinstance(tool_id, str) and tool_id in self._tool_call_indexes:
            return self._tool_call_indexes[tool_id]

        raw_index = tool_call.get("index")
        if raw_index is None:
            raw_index = chunk.get("index")
        if raw_index is None:
            raw_index = self._next_tool_index + position

        try:
            tool_index = int(raw_index)
        except (TypeError, ValueError):
            tool_index = self._next_tool_index + position

        self._next_tool_index = max(self._next_tool_index, tool_index + 1)
        if isinstance(tool_id, str) and tool_id:
            self._tool_call_indexes[tool_id] = tool_index
        self.tool_index = max(self.tool_index, tool_index)
        return tool_index

    def _tool_call_state(self, tool_index: int) -> Dict[str, Any]:
        return self._tool_calls.setdefault(
            tool_index,
            {
                "id": None,
                "name": None,
                "arguments": "",
            },
        )

    def _parse_content_delta(self, chunk: dict) -> str:
        """Parse content-delta chunks to extract text."""
        delta = chunk.get("delta", {})
        message = delta.get("message", {})
        content = message.get("content", {})
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, str):
            return content
        return ""

    def _parse_tool_call_delta(
        self, chunk: dict
    ) -> Optional[ChatCompletionToolCallChunk]:
        """Parse tool-call-delta chunks to extract tool calls."""
        tool_calls = self._tool_call_entries(chunk)
        if not tool_calls:
            return None

        tool_call = tool_calls[0]
        tool_index = self._resolve_tool_index(chunk, tool_call)
        state = self._tool_call_state(tool_index)
        function = tool_call.get("function", {}) or {}
        tool_id = tool_call.get("id") or tool_call.get("call_id")
        name = function.get("name") or tool_call.get("name")
        arguments = self._stringify_tool_fragment(
            function.get("arguments", tool_call.get("arguments"))
        )

        if tool_id:
            state["id"] = tool_id
            self._tool_call_indexes[str(tool_id)] = tool_index
        if name:
            state["name"] = name
        state["arguments"] += arguments

        return {
            "id": state["id"],
            "type": "function",
            "function": {
                "name": state["name"],
                "arguments": arguments,
            },
            "index": tool_index,
        }

    def _parse_tool_call_start(
        self, chunk: dict
    ) -> Optional[ChatCompletionToolCallChunk]:
        """Parse a tool-call-start event and register its indexed identity."""
        tool_calls = self._tool_call_entries(chunk)
        if not tool_calls:
            return None

        tool_call = tool_calls[0]
        tool_index = self._resolve_tool_index(chunk, tool_call)
        state = self._tool_call_state(tool_index)
        function = tool_call.get("function", {}) or {}
        tool_id = tool_call.get("id") or tool_call.get("call_id")
        name = function.get("name") or tool_call.get("name")
        arguments = self._stringify_tool_fragment(
            function.get("arguments", tool_call.get("arguments"))
        )

        if tool_id:
            state["id"] = tool_id
            self._tool_call_indexes[str(tool_id)] = tool_index
        if name:
            state["name"] = name
        if arguments:
            state["arguments"] = arguments

        return {
            "id": state["id"],
            "type": "function",
            "function": {
                "name": state["name"],
                "arguments": arguments,
            },
            "index": tool_index,
        }

    def _parse_tool_call_end(self, chunk: dict) -> None:
        """Record the end of an indexed tool call without duplicating arguments."""
        tool_calls = self._tool_call_entries(chunk)
        if tool_calls:
            self._resolve_tool_index(chunk, tool_calls[0])
        elif chunk.get("index") is not None:
            self._resolve_tool_index(chunk, {})

    def _parse_tool_plan_delta(self, chunk: dict) -> Optional[dict]:
        """Parse tool-plan-delta events to extract tool plan."""
        delta = chunk.get("delta", {}) or {}
        message = delta.get("message", {}) or {}
        tool_plan = message.get("tool_plan", "")
        if tool_plan:
            return {"tool_plan": tool_plan}
        return None

    def _parse_citation_start(self, chunk: dict) -> Optional[dict]:
        """Parse citation-start events to extract citations."""
        delta = chunk.get("delta", {}) or {}
        message = delta.get("message", {}) or {}
        citations = message.get("citations")
        if citations:
            if isinstance(citations, dict):
                citations = [citations]
            return {"citations": citations}
        return None

    def _parse_message_end(
        self, chunk: dict
    ) -> Tuple[bool, str, Optional[ChatCompletionUsageBlock], Optional[str]]:
        """Parse message-end events to extract finish info and usage."""
        delta = chunk.get("delta", {}) or {}
        is_finished = True
        raw_finish_reason = delta.get("finish_reason")
        normalized_finish_reason = (
            str(raw_finish_reason).strip().upper()
            if raw_finish_reason is not None
            else ""
        )
        native_error = delta.get("error")
        if native_error:
            error_message = str(native_error).strip()
            if error_message:
                raise CohereError(
                    status_code=(
                        408 if normalized_finish_reason == "TIMEOUT" else 500
                    ),
                    message=f"Cohere streaming error: {error_message}",
                )

        if normalized_finish_reason in {"ERROR", "TIMEOUT"}:
            raise CohereError(
                status_code=408 if normalized_finish_reason == "TIMEOUT" else 500,
                message=(
                    f"Cohere streaming terminated with "
                    f"{normalized_finish_reason.lower()}"
                ),
            )

        raw_finish_reason = raw_finish_reason or "COMPLETE"
        finish_reason = self._FINISH_REASON_MAP.get(
            str(raw_finish_reason).upper(), str(raw_finish_reason).lower()
        )

        usage = None
        usage_data = delta.get("usage", {}) or {}
        if usage_data:
            tokens_data = usage_data.get("tokens", {}) or {}
            prompt_tokens = int(tokens_data.get("input_tokens", 0) or 0)
            completion_tokens = int(tokens_data.get("output_tokens", 0) or 0)
            usage = ChatCompletionUsageBlock(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return is_finished, finish_reason, usage, str(raw_finish_reason)

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        """
        Parse Cohere v2 streaming chunks.

        v2 format:
        - Content: chunk.type == "content-delta" -> chunk.delta.message.content.text
        - Tool calls: chunk.type == "tool-call-{start,delta,end}"
        - Tool plan: chunk.type == "tool-plan-delta" -> chunk.delta.message.tool_plan
        - Citations: chunk.type == "citation-start" -> chunk.delta.message.citations
        - Finish: chunk.type == "message-end" -> chunk.delta.finish_reason
        """
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            chunk_type = chunk.get("type", "")

            # Handle different chunk types
            if chunk_type == "content-delta":
                text = self._parse_content_delta(chunk)
            elif chunk_type == "tool-call-start":
                tool_use = self._parse_tool_call_start(chunk)
            elif chunk_type == "tool-call-delta":
                tool_use = self._parse_tool_call_delta(chunk)
            elif chunk_type == "tool-call-end":
                self._parse_tool_call_end(chunk)
            elif chunk_type == "tool-plan-delta":
                provider_specific_fields = self._parse_tool_plan_delta(chunk)
            elif chunk_type == "citation-start":
                provider_specific_fields = self._parse_citation_start(chunk)
            elif chunk_type == "message-end":
                self._message_end_received = True
                (
                    is_finished,
                    finish_reason,
                    usage,
                    raw_finish_reason,
                ) = self._parse_message_end(chunk)
                provider_specific_fields = {
                    "native_finish_reason": raw_finish_reason
                }

            # Handle citations in any chunk type (fallback)
            if "citations" in chunk:
                if provider_specific_fields is None:
                    provider_specific_fields = {}
                provider_specific_fields["citations"] = chunk["citations"]

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=0,
                provider_specific_fields=provider_specific_fields,
            )

        except Exception as e:
            raise ValueError(f"Failed to parse v2 chunk: {e}, chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            chunk = None
            try:
                if self._pending_sse_payloads:
                    parsed_chunk = self._parse_sse_json()
                else:
                    chunk = self.response_iterator.__next__()
                    parsed_chunk = self._parse_sse_json(chunk=chunk)
            except StopIteration:
                try:
                    self._validate_stream_end()
                except ValueError as e:
                    raise RuntimeError(f"Error parsing stream termination: {e}") from e
                raise StopIteration
            except ValueError as e:
                raise RuntimeError(f"Error receiving chunk from stream: {e}")
            except GeneratorExit:
                self.close()
                raise

            try:
                if parsed_chunk is None:
                    continue
                return self.chunk_parser(chunk=parsed_chunk)
            except StopIteration:
                raise StopIteration
            except ValueError as e:
                raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    def convert_str_chunk_to_generic_chunk(
        self, chunk: Union[str, bytes, dict]
    ) -> GenericStreamingChunk:
        """
        Convert a string chunk to a GenericStreamingChunk for v2

        Note: This is used for Cohere v2 pass through streaming logging
        """
        data_json = self._parse_sse_json(chunk=chunk)
        if data_json is None:
            return self._empty_chunk()
        return self.chunk_parser(chunk=data_json)

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        while True:
            chunk = None
            try:
                if self._pending_sse_payloads:
                    parsed_chunk = self._parse_sse_json()
                else:
                    chunk = await self.async_response_iterator.__anext__()
                    parsed_chunk = self._parse_sse_json(chunk=chunk)
            except StopIteration:
                try:
                    self._validate_stream_end()
                except ValueError as e:
                    raise RuntimeError(f"Error parsing stream termination: {e}") from e
                raise StopAsyncIteration
            except StopAsyncIteration:
                try:
                    self._validate_stream_end()
                except ValueError as e:
                    raise RuntimeError(f"Error parsing stream termination: {e}") from e
                raise StopAsyncIteration
            except ValueError as e:
                raise RuntimeError(f"Error receiving chunk from stream: {e}")
            except asyncio.CancelledError:
                await self.aclose()
                raise

            try:
                if parsed_chunk is None:
                    continue
                return self.chunk_parser(chunk=parsed_chunk)
            except StopIteration:
                try:
                    self._validate_stream_end()
                except ValueError as e:
                    raise RuntimeError(f"Error parsing stream termination: {e}") from e
                raise StopAsyncIteration
            except StopAsyncIteration:
                raise StopAsyncIteration
            except ValueError as e:
                raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
