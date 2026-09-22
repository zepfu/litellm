import hashlib
import json
import re
from typing import Any, Dict, List, Literal, NoReturn, Optional, Tuple, Union

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.cohere.chat.citation_translation import (
    cohere_citation_provider_fields,
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


class CohereStreamDiagnostic(ValueError):
    """Bounded Cohere stream-parse diagnostic.

    The message records parse stage, event type, event index, byte count, and
    a payload hash. The payload itself is not stored on the exception.
    """


_COHERE_DIAGNOSTIC_PREFIX = "Cohere stream diagnostic "
_COHERE_PARSE_STAGES = frozenset({"framing", "event-schema", "argument-assembly"})
_COHERE_DIAGNOSTIC_REASONS = frozenset(
    {
        "unsupported_type",
        "invalid_encoding",
        "malformed_json",
        "non_object_event",
        "unconsumed_sse",
        "missing_message_end",
        "event_schema",
        "argument_assembly",
        "receive_error",
        "provider_terminal",
        "unspecified",
    }
)
_COHERE_EVENT_TYPES = frozenset(
    {
        "message-start",
        "content-start",
        "content-delta",
        "content-end",
        "tool-plan-delta",
        "tool-call-start",
        "tool-call-delta",
        "tool-call-end",
        "citation-start",
        "citation-end",
        "message-end",
    }
)
_COHERE_EVENT_INDEX_RE = re.compile(r"\A-?\d{1,12}\Z")
_COHERE_PROVIDER_ERROR_TOKEN_RE = re.compile(r"\A[A-Za-z0-9_.:-]{1,64}\Z")
_COHERE_PROVIDER_ERROR_PREFIXES = (
    "Cohere streaming error: ",
    "Cohere streaming terminated with ",
)


def _cohere_json_default(value: Any) -> str:
    return f"<{type(value).__name__}>"


def _cohere_payload_bytes(payload: Any) -> bytes:
    if payload is None:
        return b""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, str):
        return payload.encode("utf-8", errors="replace")
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_cohere_json_default,
        )
    except (TypeError, ValueError):
        return b""
    return rendered.encode("utf-8", errors="replace")


def _bounded_event_type(value: Any) -> str:
    if isinstance(value, str) and value in _COHERE_EVENT_TYPES:
        return value
    return "unknown"


def _bounded_event_index(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "none"
    if isinstance(value, int):
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value.strip()
    else:
        return "none"
    if _COHERE_EVENT_INDEX_RE.fullmatch(rendered):
        return rendered
    return "none"


def _cohere_event_type(payload: Any) -> str:
    if isinstance(payload, dict):
        return _bounded_event_type(payload.get("type"))
    return "unknown"


def _cohere_event_index(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "none"
    if "index" in payload:
        return _bounded_event_index(payload.get("index"))
    delta = payload.get("delta")
    if not isinstance(delta, dict):
        return "none"
    message = delta.get("message")
    if not isinstance(message, dict):
        return "none"
    tool_calls = message.get("tool_calls")
    candidate: Any = None
    if isinstance(tool_calls, dict):
        candidate = tool_calls.get("index")
    elif (
        isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict)
    ):
        candidate = tool_calls[0].get("index")
    if candidate is None:
        return "none"
    return _bounded_event_index(candidate)


def _cohere_stream_diagnostic(
    *,
    stage: str,
    payload: Any,
    reason: str,
    event_type: Optional[str] = None,
    event_index: Any = None,
) -> str:
    if stage not in _COHERE_PARSE_STAGES:
        stage = "event-schema"
    if reason not in _COHERE_DIAGNOSTIC_REASONS:
        reason = "unspecified"
    bounded_type = (
        _bounded_event_type(event_type)
        if event_type is not None
        else _cohere_event_type(payload)
    )
    bounded_index = (
        _bounded_event_index(event_index)
        if event_index is not None
        else _cohere_event_index(payload)
    )
    raw = _cohere_payload_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    return (
        f"{_COHERE_DIAGNOSTIC_PREFIX}"
        f"stage={stage} "
        f"event_type={bounded_type} "
        f"event_index={bounded_index} "
        f"byte_count={len(raw)} "
        f"payload_sha256={digest} "
        f"reason={reason}"
    )


def _raise_detached(exc: BaseException) -> NoReturn:
    """Raise ``exc`` without a cause or context that could retain payload bytes."""
    try:
        raise exc
    except BaseException as raised:
        raised.__context__ = None
        raised.__cause__ = None
        raised.__suppress_context__ = True
        raise


def _raise_cohere_diagnostic(
    *,
    stage: str,
    payload: Any,
    reason: str,
    event_type: Optional[str] = None,
    event_index: Any = None,
) -> NoReturn:
    _raise_detached(
        CohereStreamDiagnostic(
            _cohere_stream_diagnostic(
                stage=stage,
                payload=payload,
                reason=reason,
                event_type=event_type,
                event_index=event_index,
            )
        )
    )


def _cohere_provider_error_token(native_error: Any) -> Optional[str]:
    if not isinstance(native_error, dict):
        return None
    for key in ("code", "type"):
        token = native_error.get(key)
        if isinstance(token, str) and _COHERE_PROVIDER_ERROR_TOKEN_RE.fullmatch(token):
            return token
    return None


def _safe_provider_failure_message(message: str) -> bool:
    if not message.startswith(_COHERE_PROVIDER_ERROR_PREFIXES):
        return False
    if len(message) > 280 or "\n" in message or "\r" in message:
        return False
    if "{" in message or "}" in message:
        return False
    return True


def _cohere_public_failure_message(
    exc: BaseException,
    *,
    payload: Any,
    stage: str,
    reason: str,
) -> str:
    if isinstance(exc, CohereStreamDiagnostic):
        return str(exc)
    if isinstance(exc, UnicodeDecodeError):
        return _cohere_stream_diagnostic(
            stage="framing",
            payload=payload,
            reason="invalid_encoding",
        )
    message = str(exc)
    if _safe_provider_failure_message(message):
        return message
    return _cohere_stream_diagnostic(
        stage=stage,
        payload=payload,
        reason=reason,
    )


def _raise_cohere_runtime(
    exc: BaseException,
    *,
    payload: Any,
    stage: str,
    reason: str,
) -> NoReturn:
    _raise_detached(
        RuntimeError(
            _cohere_public_failure_message(
                exc,
                payload=payload,
                stage=stage,
                reason=reason,
            )
        )
    )


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
            _raise_cohere_diagnostic(
                stage="framing",
                payload=chunk,
                reason="malformed_json",
            )

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as exc:
            _raise_cohere_runtime(
                exc,
                payload=None,
                stage="framing",
                reason="receive_error",
            )

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as exc:
            _raise_cohere_runtime(
                exc,
                payload=chunk,
                stage="event-schema",
                reason="event_schema",
            )

    def convert_str_chunk_to_generic_chunk(self, chunk: str) -> GenericStreamingChunk:
        """
        Convert a string chunk to a GenericStreamingChunk

        Note: This is used for Cohere pass through streaming logging
        """
        str_line = chunk
        if isinstance(chunk, bytes):  # Handle binary data
            try:
                str_line = chunk.decode("utf-8")  # Convert bytes to string
            except UnicodeDecodeError:
                _raise_cohere_diagnostic(
                    stage="framing",
                    payload=chunk,
                    reason="invalid_encoding",
                )
            index = str_line.find("data:")
            if index != -1:
                str_line = str_line[index:]

        try:
            data_json = json.loads(str_line)
        except json.JSONDecodeError:
            _raise_cohere_diagnostic(
                stage="framing",
                payload=str_line,
                reason="malformed_json",
            )
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
        except ValueError as exc:
            _raise_cohere_runtime(
                exc,
                payload=None,
                stage="framing",
                reason="receive_error",
            )

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as exc:
            _raise_cohere_runtime(
                exc,
                payload=chunk,
                stage="event-schema",
                reason="event_schema",
            )


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
            try:
                chunk = chunk.decode("utf-8")
            except UnicodeDecodeError:
                _raise_cohere_diagnostic(
                    stage="framing",
                    payload=chunk,
                    reason="invalid_encoding",
                )
        if isinstance(chunk, dict):
            try:
                return [json.dumps(chunk, ensure_ascii=False)]
            except (TypeError, ValueError):
                _raise_cohere_diagnostic(
                    stage="event-schema",
                    payload=chunk,
                    reason="event_schema",
                )
        if not isinstance(chunk, str):
            _raise_cohere_diagnostic(
                stage="framing",
                payload=None,
                reason="unsupported_type",
            )

        if not chunk.strip():
            return []

        if not any(line.strip().startswith("data:") for line in chunk.splitlines()):
            if all(
                not line.strip() or line.strip().startswith((":", "event:"))
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
            _raise_cohere_diagnostic(
                stage="framing",
                payload="\n".join(self._pending_sse_payloads),
                reason="unconsumed_sse",
            )
        if not self._message_end_received:
            _raise_cohere_diagnostic(
                stage="framing",
                payload=b"",
                reason="missing_message_end",
            )

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
        except json.JSONDecodeError:
            _raise_cohere_diagnostic(
                stage="framing",
                payload=payload,
                reason="malformed_json",
            )

        if not isinstance(parsed_chunk, dict):
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=parsed_chunk,
                reason="non_object_event",
            )
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

    def _resolve_tool_index(
        self, chunk: dict, tool_call: dict, position: int = 0
    ) -> int:
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
        try:
            arguments = self._stringify_tool_fragment(
                function.get("arguments", tool_call.get("arguments"))
            )
        except (TypeError, ValueError):
            _raise_cohere_diagnostic(
                stage="argument-assembly",
                payload=chunk,
                reason="argument_assembly",
            )

        if tool_id:
            state["id"] = tool_id
            self._tool_call_indexes[str(tool_id)] = tool_index
        if name:
            state["name"] = name
        try:
            state["arguments"] += arguments
        except (TypeError, ValueError):
            _raise_cohere_diagnostic(
                stage="argument-assembly",
                payload=chunk,
                reason="argument_assembly",
            )

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
        try:
            arguments = self._stringify_tool_fragment(
                function.get("arguments", tool_call.get("arguments"))
            )
        except (TypeError, ValueError):
            _raise_cohere_diagnostic(
                stage="argument-assembly",
                payload=chunk,
                reason="argument_assembly",
            )

        if tool_id:
            state["id"] = tool_id
            self._tool_call_indexes[str(tool_id)] = tool_index
        if name:
            state["name"] = name
        if arguments:
            try:
                state["arguments"] = arguments
            except (TypeError, ValueError):
                _raise_cohere_diagnostic(
                    stage="argument-assembly",
                    payload=chunk,
                    reason="argument_assembly",
                )

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
            return cohere_citation_provider_fields(citations)
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
        if isinstance(native_error, str):
            error_message = " ".join(native_error.split())
            if error_message:
                if (
                    len(error_message) <= 240
                    and "{" not in error_message
                    and "}" not in error_message
                ):
                    raise CohereError(
                        status_code=(
                            408 if normalized_finish_reason == "TIMEOUT" else 500
                        ),
                        message=f"Cohere streaming error: {error_message}",
                    )
                _raise_cohere_diagnostic(
                    stage="event-schema",
                    payload=chunk,
                    reason="provider_terminal",
                )
        elif native_error:
            provider_token = _cohere_provider_error_token(native_error)
            if provider_token is not None:
                raise CohereError(
                    status_code=(408 if normalized_finish_reason == "TIMEOUT" else 500),
                    message=f"Cohere streaming error: {provider_token}",
                )
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=chunk,
                reason="provider_terminal",
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
                provider_specific_fields = {"native_finish_reason": raw_finish_reason}

            # Handle citations in any chunk type (fallback). Map fields only;
            # do not copy tool-output bodies into the streamed metadata.
            if "citations" in chunk:
                mapped_citations = cohere_citation_provider_fields(
                    chunk.get("citations")
                )
                if mapped_citations:
                    if provider_specific_fields is None:
                        provider_specific_fields = {}
                    provider_specific_fields["citations"] = mapped_citations[
                        "citations"
                    ]

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=0,
                provider_specific_fields=provider_specific_fields,
            )

        except CohereStreamDiagnostic:
            raise
        except CohereError as exc:
            message = str(exc)
            if message.startswith(_COHERE_DIAGNOSTIC_PREFIX):
                _raise_detached(CohereStreamDiagnostic(message))
            if _safe_provider_failure_message(message):
                _raise_detached(ValueError(message))
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=chunk,
                reason="provider_terminal",
            )
        except Exception:
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=chunk,
                reason="event_schema",
            )

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
                except ValueError as exc:
                    _raise_cohere_runtime(
                        exc,
                        payload=None,
                        stage="framing",
                        reason="missing_message_end",
                    )
                raise StopIteration
            except ValueError as exc:
                _raise_cohere_runtime(
                    exc,
                    payload=chunk,
                    stage="framing",
                    reason="receive_error",
                )

            try:
                if parsed_chunk is None:
                    continue
                return self.chunk_parser(chunk=parsed_chunk)
            except StopIteration:
                raise StopIteration
            except ValueError as exc:
                _raise_cohere_runtime(
                    exc,
                    payload=chunk if chunk is not None else parsed_chunk,
                    stage="event-schema",
                    reason="event_schema",
                )

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
                except ValueError as exc:
                    _raise_cohere_runtime(
                        exc,
                        payload=None,
                        stage="framing",
                        reason="missing_message_end",
                    )
                raise StopAsyncIteration
            except StopAsyncIteration:
                try:
                    self._validate_stream_end()
                except ValueError as exc:
                    _raise_cohere_runtime(
                        exc,
                        payload=None,
                        stage="framing",
                        reason="missing_message_end",
                    )
                raise StopAsyncIteration
            except ValueError as exc:
                _raise_cohere_runtime(
                    exc,
                    payload=chunk,
                    stage="framing",
                    reason="receive_error",
                )

            try:
                if parsed_chunk is None:
                    continue
                return self.chunk_parser(chunk=parsed_chunk)
            except StopIteration:
                try:
                    self._validate_stream_end()
                except ValueError as exc:
                    _raise_cohere_runtime(
                        exc,
                        payload=None,
                        stage="framing",
                        reason="missing_message_end",
                    )
                raise StopAsyncIteration
            except StopAsyncIteration:
                raise StopAsyncIteration
            except ValueError as exc:
                _raise_cohere_runtime(
                    exc,
                    payload=chunk if chunk is not None else parsed_chunk,
                    stage="event-schema",
                    reason="event_schema",
                )
