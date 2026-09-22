import asyncio
import codecs
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Literal, NoReturn, Optional, Set, Tuple, Union

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.cohere.cancellation import (
    aclose_upstream_response_once,
    close_upstream_response_once,
)
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
_COHERE_TERMINAL_IDENTIFIERS = {
    "ERROR": "error",
    "TIMEOUT": "timeout",
}
_COHERE_FIXED_TERMINAL_MESSAGES = frozenset(
    {
        "Cohere streaming terminated with error",
        "Cohere streaming terminated with timeout",
    }
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


def _cohere_v1_chunk_index(raw_index: Any) -> Optional[int]:
    """Parse a V1 chunk index without interpolating a rejected value."""
    if isinstance(raw_index, bool):
        return int(raw_index)
    if isinstance(raw_index, int):
        return raw_index
    if isinstance(raw_index, float):
        if not math.isfinite(raw_index):
            return None
        try:
            return int(raw_index)
        except (OverflowError, TypeError, ValueError):
            return None
    if isinstance(raw_index, str) and _COHERE_EVENT_INDEX_RE.fullmatch(
        raw_index.strip()
    ):
        return int(raw_index.strip())
    return None


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


def _cohere_fixed_terminal_identifier(value: Any) -> Optional[str]:
    """Return a terminal word from the fixed set, never the raw payload text."""
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    return _COHERE_TERMINAL_IDENTIFIERS.get(value.strip().upper())


def _cohere_fixed_terminal_message(identifier: str) -> str:
    if identifier not in _COHERE_TERMINAL_IDENTIFIERS.values():
        identifier = "error"
    return f"Cohere streaming terminated with {identifier}"


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
    if message in _COHERE_FIXED_TERMINAL_MESSAGES:
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


_COHERE_PARSER_FAILURE_REASONS = {
    "Cohere stream ended without a native message-end event": "missing_message_end",
    "Cohere stream ended with an incomplete event": "unconsumed_sse",
    "Malformed Cohere stream JSON": "malformed_json",
    "Cohere stream contained invalid UTF-8": "invalid_encoding",
    "Expected Cohere stream event object": "non_object_event",
    "Unsupported Cohere stream chunk type": "unsupported_type",
    "Cohere V2 tool-call arguments were not a string": "argument_assembly",
}
_COHERE_FRAMING_FAILURE_REASONS = frozenset(
    {
        "missing_message_end",
        "unconsumed_sse",
        "malformed_json",
        "invalid_encoding",
        "unsupported_type",
    }
)


def _cohere_parser_failure_reason(message: str) -> str:
    return _COHERE_PARSER_FAILURE_REASONS.get(message, "event_schema")


def _cohere_parser_failure_stage(reason: str) -> str:
    if reason == "argument_assembly":
        return "argument-assembly"
    if reason in _COHERE_FRAMING_FAILURE_REASONS:
        return "framing"
    return "event-schema"


def _raise_classified_stream_failure(exc: BaseException, *, payload: Any) -> NoReturn:
    """Publish a landed diagnostic for one accepted parser failure."""
    if isinstance(exc, CohereStreamDiagnostic):
        raise exc
    if isinstance(exc, CohereError):
        message = str(exc)
        if message in _COHERE_FIXED_TERMINAL_MESSAGES:
            _raise_detached(ValueError(message))
        _raise_cohere_diagnostic(
            stage="event-schema",
            payload=payload,
            reason="provider_terminal",
        )
    reason = _cohere_parser_failure_reason(str(exc))
    _raise_cohere_diagnostic(
        stage=_cohere_parser_failure_stage(reason),
        payload=payload,
        reason=reason,
    )


def _raise_iterator_terminal(exc: CohereError) -> NoReturn:
    message = str(exc)
    if message in _COHERE_FIXED_TERMINAL_MESSAGES:
        _raise_cohere_runtime(
            ValueError(message),
            payload=None,
            stage="event-schema",
            reason="provider_terminal",
        )
    _raise_cohere_runtime(
        exc,
        payload=None,
        stage="event-schema",
        reason="provider_terminal",
    )


def _raise_iterator_parse_failure(exc: ValueError) -> NoReturn:
    reason = "event_schema"
    stage = "event-schema"
    if not isinstance(exc, CohereStreamDiagnostic) and (
        str(exc) not in _COHERE_FIXED_TERMINAL_MESSAGES
    ):
        reason = _cohere_parser_failure_reason(str(exc))
        stage = _cohere_parser_failure_stage(reason)
    _raise_cohere_runtime(
        exc,
        payload=None,
        stage=stage,
        reason=reason,
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

            index = _cohere_v1_chunk_index(chunk.get("index", 0))
            if index is None:
                _raise_cohere_diagnostic(
                    stage="event-schema",
                    payload=chunk,
                    reason="event_schema",
                )

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
        except CohereStreamDiagnostic:
            raise
        except (TypeError, ValueError):
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=chunk,
                reason="event_schema",
            )

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except Exception as exc:
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
        try:
            return self.chunk_parser(chunk=data_json)
        except CohereStreamDiagnostic:
            raise
        except json.JSONDecodeError:
            _raise_cohere_diagnostic(
                stage="framing",
                payload=str_line,
                reason="malformed_json",
            )
        except (TypeError, ValueError):
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=data_json,
                reason="event_schema",
            )

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except Exception as exc:
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


_COHERE_V2_DOCUMENTED_FINISH_REASONS = {
    "COMPLETE": "stop",
    "STOP_SEQUENCE": "stop",
    "MAX_TOKENS": "length",
    "TOOL_CALL": "tool_calls",
}
_COHERE_V2_CITATION_TYPES = {"TEXT_CONTENT", "THINKING_CONTENT", "PLAN"}
_COHERE_V2_SOURCE_TYPES = {"document", "tool"}
_COHERE_V2_SSE_EVENT_NAMES = (
    "citation-end",
    "citation-start",
    "content-delta",
    "content-end",
    "content-start",
    "debug",
    "message-end",
    "message-start",
    "tool-call-delta",
    "tool-call-end",
    "tool-call-start",
    "tool-plan-delta",
)
_COHERE_V2_SSE_FIELD_INTROS = ("data:", "event:", "id:", "retry:", ":")
_COHERE_V2_JSON_LITERALS = ("true", "false", "null")


class _CohereV2StreamSentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


_COHERE_V2_STREAM_DONE = _CohereV2StreamSentinel("cohere-v2-stream-done")
_COHERE_V2_SOURCE_EXHAUSTED = _CohereV2StreamSentinel("cohere-v2-source-exhausted")


def _empty_generic_chunk() -> GenericStreamingChunk:
    return GenericStreamingChunk(
        text="",
        tool_use=None,
        is_finished=False,
        finish_reason="",
        usage=None,
        index=0,
        provider_specific_fields=None,
    )


def _hex_run(text: str) -> int:
    count = 0
    for char in text:
        if char not in "0123456789abcdefABCDEF":
            break
        count += 1
    return count


def _unicode_escape_is_open(payload: str, pos: int) -> bool:
    """True when a \\u escape has no character after its hex digits yet."""
    start = payload.rfind("\\u", 0, min(len(payload), pos + 2))
    if start < 0 or start > pos:
        return False
    digits = payload[start + 2 :]
    hex_digits = _hex_run(digits)
    return hex_digits == len(digits) and hex_digits <= 4


def _number_form(text: str) -> Optional[str]:
    """Classify text as a complete JSON number, an open prefix, or neither."""
    if text == "":
        return None
    index = 0
    if text[0] == "-":
        index = 1
        if index == len(text) or text[index] == ".":
            return "incomplete" if text in {"-", "-."} else None
    if index >= len(text) or not text[index].isdigit():
        return None
    while index < len(text) and text[index].isdigit():
        index += 1
    if index == len(text):
        return "complete"
    if text[index] == ".":
        index += 1
        fraction_start = index
        while index < len(text) and text[index].isdigit():
            index += 1
        if index == len(text):
            return "incomplete" if fraction_start == index else "complete"
    if index < len(text) and text[index] in "eE":
        index += 1
        if index == len(text):
            return "incomplete"
        if text[index] in "+-":
            index += 1
            if index == len(text):
                return "incomplete"
        if index >= len(text) or not text[index].isdigit():
            return None
        while index < len(text) and text[index].isdigit():
            index += 1
        if index == len(text):
            return "complete"
    return None


def _is_literal_prefix(tail: str) -> bool:
    return any(
        literal.startswith(tail) and literal != tail
        for literal in _COHERE_V2_JSON_LITERALS
    )


def _tail_is_open_json_value(payload: str, pos: int) -> bool:
    tail = payload[pos:]
    if tail == "":
        return False
    if _number_form(tail) == "incomplete":
        return True
    if pos > 0 and payload[pos - 1].isdigit():
        return _number_form("0" + tail) == "incomplete"
    return False


def _json_decode_is_incomplete(exc: json.JSONDecodeError, payload: str) -> bool:
    if exc.msg.startswith("Unterminated"):
        return True
    if exc.msg.startswith("Invalid \\u") and _unicode_escape_is_open(payload, exc.pos):
        return True
    if exc.msg.startswith("Expecting value") and _is_literal_prefix(payload[exc.pos :]):
        return True
    if _tail_is_open_json_value(payload, exc.pos):
        return True
    if exc.msg.startswith("Expecting") and payload[exc.pos :].strip() == "":
        return True
    return False


def _json_payload_is_incomplete(payload: str) -> bool:
    if payload == "":
        return True
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        return _json_decode_is_incomplete(exc, payload)
    return False


def _load_json_object(payload: str, at_end: bool) -> dict:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        if at_end and _json_decode_is_incomplete(exc, payload):
            raise ValueError("Cohere stream ended with an incomplete event") from None
        raise ValueError("Malformed Cohere stream JSON") from None
    if not isinstance(value, dict):
        raise ValueError("Expected Cohere stream event object")
    return value


def _split_first_line(text: str) -> Optional[Tuple[str, str]]:
    newline_at = text.find("\n")
    carriage_at = text.find("\r")
    if newline_at == -1 and carriage_at == -1:
        return None
    if carriage_at != -1 and (newline_at == -1 or carriage_at < newline_at):
        if carriage_at + 1 == len(text):
            return None
        if text[carriage_at + 1] == "\n":
            return text[:carriage_at], text[carriage_at + 2 :]
        return text[:carriage_at], text[carriage_at + 1 :]
    line = text[:newline_at]
    if line.endswith("\r"):
        line = line[:-1]
    return line, text[newline_at + 1 :]


def _is_pending_sse_line(stripped: str) -> bool:
    """True when more bytes can still finish this SSE line."""
    if any(
        intro.startswith(stripped) and intro != stripped
        for intro in _COHERE_V2_SSE_FIELD_INTROS
    ):
        return True
    if not stripped.startswith("event:"):
        return False
    rest = stripped[6:]
    name = rest[1:] if rest.startswith(" ") else rest
    if name == "":
        return True
    return any(
        known.startswith(name) and known != name for known in _COHERE_V2_SSE_EVENT_NAMES
    )


def _is_done_prefix(payload: str) -> bool:
    return "[DONE]".startswith(payload) and payload != "[DONE]"


def _is_complete_logical_line(text: str) -> bool:
    stripped = text.strip()
    if stripped == "":
        return True
    if _is_pending_sse_line(stripped):
        return False
    if stripped.startswith((":", "event:", "id:", "retry:")):
        return True
    if stripped.startswith("data:"):
        payload = stripped[5:].lstrip()
        if payload == "[DONE]":
            return True
        if _is_done_prefix(payload) or _json_payload_is_incomplete(payload):
            return False
        return True
    if stripped.startswith("{"):
        return not _json_payload_is_incomplete(stripped)
    return True


def _require_object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Cohere V2 stream {label} was not an object")
    return value


def _delta_message(event: dict) -> dict:
    delta = _require_object(event.get("delta"), "delta")
    return _require_object(delta.get("message"), "delta.message")


def _require_index(event: dict) -> int:
    index = event.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Cohere V2 stream event omitted an integer index")
    return index


def _optional_index(event: dict) -> Optional[int]:
    if "index" not in event or event.get("index") is None:
        return None
    return _require_index(event)


def _token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Cohere V2 stream usage contained a non-numeric token count")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("Cohere V2 stream usage contained a non-numeric token count")
    count = int(value)
    if count < 0:
        raise ValueError("Cohere V2 stream usage contained a non-numeric token count")
    return count


def _usage_block(usage: Any) -> Optional[ChatCompletionUsageBlock]:
    if usage is None:
        return None
    source_parent = _require_object(usage, "usage")
    source = source_parent.get("tokens")
    if not isinstance(source, dict):
        billed = source_parent.get("billed_units")
        source = billed if isinstance(billed, dict) else None
    if not isinstance(source, dict):
        return None
    if "input_tokens" not in source and "output_tokens" not in source:
        return None
    prompt_tokens = _token_count(source.get("input_tokens", 0))
    completion_tokens = _token_count(source.get("output_tokens", 0))
    block: ChatCompletionUsageBlock = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    cached_tokens = source_parent.get("cached_tokens")
    if cached_tokens is not None:
        block["prompt_tokens_details"] = {"cached_tokens": _token_count(cached_tokens)}
    return block


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Cohere V2 stream {label} was not a non-negative integer")
    return value


def _copy_citation_source(source: Any) -> dict:
    if not isinstance(source, dict):
        raise ValueError("Cohere V2 citation source was not an object")
    source_type = source.get("type")
    if source_type not in _COHERE_V2_SOURCE_TYPES:
        raise ValueError("Cohere V2 citation source type was not a documented value")
    copied: Dict[str, Any] = {"type": source_type}
    if "id" in source:
        source_id = source.get("id")
        if not isinstance(source_id, str):
            raise ValueError("Cohere V2 citation source id was not a string")
        copied["id"] = source_id
    payload_key = "document" if source_type == "document" else "tool_output"
    if payload_key in source:
        payload = source.get(payload_key)
        if not isinstance(payload, dict):
            raise ValueError("Cohere V2 citation source payload was not an object")
        copied[payload_key] = {**payload}
    return copied


def _copy_citation(raw: Any) -> dict:
    citation_raw = _require_object(raw, "citations")
    citation: Dict[str, Any] = {}
    if "start" in citation_raw:
        citation["start"] = _non_negative_int(
            citation_raw.get("start"), "citation start"
        )
    if "end" in citation_raw:
        citation["end"] = _non_negative_int(citation_raw.get("end"), "citation end")
    if "text" in citation_raw:
        text = citation_raw.get("text")
        if not isinstance(text, str):
            raise ValueError("Cohere V2 citation text was not a string")
        citation["text"] = text
    if "sources" in citation_raw:
        sources = citation_raw.get("sources")
        if not isinstance(sources, list):
            raise ValueError("Cohere V2 citation sources were not a list")
        citation["sources"] = [_copy_citation_source(source) for source in sources]
    if "content_index" in citation_raw:
        citation["content_index"] = _non_negative_int(
            citation_raw.get("content_index"), "citation content index"
        )
    if "type" in citation_raw:
        citation_type = citation_raw.get("type")
        if citation_type not in _COHERE_V2_CITATION_TYPES:
            raise ValueError("Cohere V2 citation type was not a documented value")
        citation["type"] = citation_type
    return citation


class _CohereV2ToolCall:
    def __init__(self, index: int, tool_id: str, name: str) -> None:
        self.index = index
        self.tool_id = tool_id
        self.name = name
        self.arguments = ""
        self.ended = False


class _CohereV2EventBuffer:
    """Hold SSE text until a V2 event object or terminal [DONE] is complete."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._raw = ""
        self._data_parts: List[str] = []
        self._received_text = False
        self._done = False
        self._hold_for_newline = False
        self.events: List[Any] = []

    def feed(self, chunk: Union[str, bytes, dict, None]) -> None:
        if chunk is None:
            return
        if isinstance(chunk, dict):
            self._accept_event_object(chunk)
            return
        if isinstance(chunk, bytes):
            # Raw transport fragments end only at a line boundary.
            self._hold_for_newline = True
            self._append_text(self._decode_bytes(chunk))
            self._drain(allow_logical=False)
            if not self._raw:
                self._hold_for_newline = False
            return
        if isinstance(chunk, str):
            if chunk == "":
                self._blank_line()
                return
            # Strings without a held raw fragment are complete logical lines
            # when the SSE field is already finished. Prefixes stay buffered.
            self._append_text(chunk)
            self._drain(allow_logical=not self._hold_for_newline)
            if not self._raw:
                self._hold_for_newline = False
            return
        raise ValueError("Unsupported Cohere stream chunk type")

    def finalize(self) -> None:
        try:
            tail = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raise ValueError("Cohere stream contained invalid UTF-8") from None
        self._append_text(tail)
        if self._raw.endswith("\r"):
            self._raw = self._raw[:-1] + "\n"
        self._drain()
        if self._raw:
            if not _is_complete_logical_line(self._raw):
                raise ValueError("Cohere stream ended with an incomplete event")
            line = self._raw
            self._raw = ""
            self._consume_line(line)
        if self._data_parts:
            self._flush_event(at_end=True)

    def _decode_bytes(self, chunk: bytes) -> str:
        try:
            return self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError:
            raise ValueError("Cohere stream contained invalid UTF-8") from None

    def _append_text(self, text: str) -> None:
        if not self._received_text and text.startswith("\ufeff"):
            text = text[1:]
        if text:
            self._received_text = True
        self._raw += text

    def _drain(self, allow_logical: bool = True) -> None:
        while True:
            split = _split_first_line(self._raw)
            if split is not None:
                line, self._raw = split
                self._consume_line(line)
                continue
            if allow_logical and self._raw and _is_complete_logical_line(self._raw):
                line = self._raw
                self._raw = ""
                self._consume_line(line)
                continue
            return

    def _blank_line(self) -> None:
        if self._raw:
            if not _is_complete_logical_line(self._raw):
                raise ValueError("Malformed Cohere stream JSON")
            line = self._raw
            self._raw = ""
            self._consume_line(line)
        self._hold_for_newline = False
        self._flush_event(at_end=False)

    def _accept_event_object(self, event: dict) -> None:
        self._drain()
        if self._raw.strip() or self._data_parts:
            raise ValueError("Cohere stream ended with an incomplete event")
        if self._done:
            raise ValueError("Cohere stream continued after [DONE]")
        self.events.append(event)

    def _consume_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped == "":
            self._flush_event(at_end=False)
            return
        if stripped.startswith(":"):
            return
        if stripped.startswith(("event:", "id:", "retry:")):
            if self._data_parts:
                self._flush_event(at_end=False)
            return
        if stripped.startswith("data:"):
            self._push_payload(stripped[5:].lstrip())
            return
        if stripped.startswith("{") and not self._data_parts:
            self._push_payload(stripped)
            return
        raise ValueError("Malformed Cohere stream JSON")

    def _push_payload(self, payload: str) -> None:
        if self._done:
            raise ValueError("Cohere stream continued after [DONE]")
        self._data_parts.append(payload)
        joined = "\n".join(self._data_parts).strip()
        if joined == "[DONE]" or not _json_payload_is_incomplete(joined):
            self._flush_event(at_end=False)

    def _flush_event(self, at_end: bool) -> None:
        if not self._data_parts:
            return
        payload = "\n".join(self._data_parts).strip()
        self._data_parts = []
        if self._done:
            raise ValueError("Cohere stream continued after [DONE]")
        if payload == "[DONE]":
            self._done = True
            self.events.append(_COHERE_V2_STREAM_DONE)
            return
        self.events.append(_load_json_object(payload, at_end=at_end))


class CohereV2ModelResponseIterator:
    """Parse Cohere V2 SSE into text, tool-call, citation, and finish chunks.

    Event handling follows the top-level V2 stream schema. Tool-call argument
    fragments stay strings until a later consumer joins them; this iterator
    does not parse partial argument JSON.
    """

    _FINISH_REASON_MAP = _COHERE_V2_DOCUMENTED_FINISH_REASONS

    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.sync_stream = sync_stream
        self.json_mode = json_mode
        self.tool_index = -1
        self._buffer = _CohereV2EventBuffer()
        self._tools: Dict[int, _CohereV2ToolCall] = {}
        self._open_citations: Set[int] = set()
        self._message_started = False
        self._message_end_received = False
        self._source_exhausted = False
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
        return _empty_generic_chunk()

    def _output(
        self,
        text: str = "",
        tool_use: Optional[ChatCompletionToolCallChunk] = None,
        is_finished: bool = False,
        finish_reason: str = "",
        usage: Optional[ChatCompletionUsageBlock] = None,
        provider_specific_fields: Optional[dict] = None,
    ) -> GenericStreamingChunk:
        return GenericStreamingChunk(
            text=text,
            tool_use=tool_use,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=usage,
            index=0,
            provider_specific_fields=provider_specific_fields,
        )

    def _tool_chunk(
        self, tool: _CohereV2ToolCall, arguments_fragment: str
    ) -> GenericStreamingChunk:
        tool_use: ChatCompletionToolCallChunk = {
            "id": tool.tool_id,
            "type": "function",
            "function": {"name": tool.name, "arguments": arguments_fragment},
            "index": tool.index,
        }
        return self._output(tool_use=tool_use)

    def _reduce_message_start(self, event: dict) -> Optional[GenericStreamingChunk]:
        if self._message_started:
            raise ValueError("Cohere V2 stream received a second message-start event")
        self._message_started = True
        delta = event.get("delta")
        if delta is None:
            return None
        message = _require_object(delta, "delta").get("message")
        if message is None:
            return None
        role = _require_object(message, "delta.message").get("role")
        if role is not None and role != "assistant":
            raise ValueError("Cohere V2 message-start role was not assistant")
        return None

    def _reduce_content_start(self, event: dict) -> Optional[GenericStreamingChunk]:
        content = _delta_message(event).get("content")
        if content is None:
            return None
        content_object = _require_object(content, "content")
        content_type = content_object.get("type")
        if content_type is not None and content_type not in {"text", "thinking"}:
            raise ValueError("Cohere V2 content-start type was not a documented value")
        return None

    def _reduce_content_delta(self, event: dict) -> Optional[GenericStreamingChunk]:
        content = _delta_message(event).get("content")
        content_object = _require_object(content, "content")
        text = content_object.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("Cohere V2 content-delta text was not a string")
        thinking = content_object.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            raise ValueError("Cohere V2 content-delta thinking was not a string")
        fields = {"thinking": thinking} if thinking else None
        if text == "" and fields is None:
            return None
        return self._output(text=text, provider_specific_fields=fields)

    def _reduce_content_end(self, event: dict) -> Optional[GenericStreamingChunk]:
        if event.get("index") is not None:
            _require_index(event)
        return None

    def _reduce_tool_plan_delta(self, event: dict) -> Optional[GenericStreamingChunk]:
        tool_plan = _delta_message(event).get("tool_plan", "")
        if not isinstance(tool_plan, str):
            raise ValueError("Cohere V2 tool-plan-delta was not a string")
        if tool_plan == "":
            return None
        return self._output(provider_specific_fields={"tool_plan": tool_plan})

    def _tool_call_object(self, event: dict) -> dict:
        return _require_object(_delta_message(event).get("tool_calls"), "tool_calls")

    def _function_arguments(self, tool_call: dict) -> Tuple[dict, str]:
        function = tool_call.get("function", {})
        if function is None:
            function = {}
        function_object = _require_object(function, "tool call function")
        arguments = function_object.get("arguments", "")
        if not isinstance(arguments, str):
            raise ValueError("Cohere V2 tool-call arguments were not a string")
        return function_object, arguments

    def _reduce_tool_call_start(self, event: dict) -> Optional[GenericStreamingChunk]:
        index = _require_index(event)
        if index in self._tools:
            raise ValueError("Cohere V2 stream repeated a tool-call index")
        tool_call = self._tool_call_object(event)
        tool_id = tool_call.get("id")
        if not isinstance(tool_id, str) or tool_id == "":
            raise ValueError("Cohere V2 tool-call-start omitted a tool call id")
        if tool_call.get("type") != "function":
            raise ValueError("Cohere V2 tool-call-start type was not function")
        function, arguments = self._function_arguments(tool_call)
        name = function.get("name", "")
        if not isinstance(name, str):
            raise ValueError("Cohere V2 tool-call name was not a string")
        tool = _CohereV2ToolCall(index=index, tool_id=tool_id, name=name)
        tool.arguments = arguments
        self._tools[index] = tool
        self.tool_index = index
        return self._tool_chunk(tool, arguments)

    def _reduce_tool_call_delta(self, event: dict) -> Optional[GenericStreamingChunk]:
        index = _require_index(event)
        tool = self._tools.get(index)
        if tool is None or tool.ended:
            raise ValueError(
                "Cohere V2 tool-call-delta did not match an open tool call"
            )
        _function, arguments = self._function_arguments(self._tool_call_object(event))
        tool.arguments += arguments
        if arguments == "":
            return None
        return self._tool_chunk(tool, arguments)

    def _reduce_tool_call_end(self, event: dict) -> Optional[GenericStreamingChunk]:
        index = _require_index(event)
        tool = self._tools.get(index)
        if tool is None or tool.ended:
            raise ValueError("Cohere V2 tool-call-end did not match an open tool call")
        tool.ended = True
        return None

    def _reduce_citation_start(self, event: dict) -> Optional[GenericStreamingChunk]:
        index = _optional_index(event)
        citation = _copy_citation(_delta_message(event).get("citations"))
        if index is not None:
            if index in self._open_citations:
                raise ValueError("Cohere V2 stream repeated a citation index")
            self._open_citations.add(index)
        return self._output(
            provider_specific_fields=cohere_citation_provider_fields([citation])
        )

    def _reduce_citation_end(self, event: dict) -> Optional[GenericStreamingChunk]:
        index = _optional_index(event)
        if index is None:
            return None
        if index not in self._open_citations:
            raise ValueError("Cohere V2 citation-end did not match an open citation")
        self._open_citations.remove(index)
        return None

    def _reduce_message_end(self, event: dict) -> Optional[GenericStreamingChunk]:
        if self._message_end_received:
            raise ValueError("Cohere V2 stream received a second message-end event")
        delta = _require_object(event.get("delta"), "delta")
        self._raise_for_provider_terminal(event, delta)
        normalized = self._documented_finish_reason(delta.get("finish_reason"))
        terminal_identifier = _cohere_fixed_terminal_identifier(normalized)
        if terminal_identifier is not None:
            raise CohereError(
                status_code=408 if terminal_identifier == "timeout" else 500,
                message=_cohere_fixed_terminal_message(terminal_identifier),
            )
        finish_reason = self._FINISH_REASON_MAP[normalized]
        self._message_end_received = True
        return self._output(
            is_finished=True,
            finish_reason=finish_reason,
            usage=_usage_block(delta.get("usage")),
            provider_specific_fields={"native_finish_reason": normalized},
        )

    def _raise_for_provider_terminal(self, event: dict, delta: dict) -> None:
        native_error = delta.get("error")
        if isinstance(native_error, str):
            if native_error.strip():
                _raise_cohere_diagnostic(
                    stage="event-schema",
                    payload=event,
                    reason="provider_terminal",
                )
            return
        if not native_error:
            return
        terminal_identifier = None
        if isinstance(native_error, dict):
            for key in ("code", "type"):
                terminal_identifier = _cohere_fixed_terminal_identifier(
                    native_error.get(key)
                )
                if terminal_identifier is not None:
                    break
        if terminal_identifier is None:
            _raise_cohere_diagnostic(
                stage="event-schema",
                payload=event,
                reason="provider_terminal",
            )
        raise CohereError(
            status_code=408 if terminal_identifier == "timeout" else 500,
            message=_cohere_fixed_terminal_message(terminal_identifier),
        )

    @staticmethod
    def _documented_finish_reason(raw_reason: Any) -> str:
        if not isinstance(raw_reason, str):
            raise ValueError(
                "Cohere V2 stream message-end omitted a documented finish reason"
            )
        normalized = raw_reason.strip().upper()
        if normalized in {"ERROR", "TIMEOUT"}:
            return normalized
        if normalized not in _COHERE_V2_DOCUMENTED_FINISH_REASONS:
            raise ValueError(
                "Cohere V2 stream message-end omitted a documented finish reason"
            )
        return normalized

    def _with_top_level_citation_fields(
        self, event: dict, parsed: Optional[GenericStreamingChunk]
    ) -> Optional[GenericStreamingChunk]:
        if "citations" not in event:
            return parsed
        mapped_citations = cohere_citation_provider_fields(event.get("citations"))
        if not mapped_citations:
            return parsed
        if parsed is None:
            parsed = self._empty_chunk()
        provider_specific_fields = {
            **(parsed.get("provider_specific_fields") or {}),
            "citations": mapped_citations["citations"],
        }
        return {**parsed, "provider_specific_fields": provider_specific_fields}

    def _reduce_event(self, event: dict) -> Optional[GenericStreamingChunk]:
        if not isinstance(event, dict):
            raise ValueError("Expected Cohere stream event object")
        event_type = event.get("type")
        if not isinstance(event_type, str):
            raise ValueError("Cohere V2 stream event omitted a type")
        if self._message_end_received and event_type != "debug":
            raise ValueError("Cohere V2 stream event arrived after message-end")
        if event_type == "debug":
            parsed: Optional[GenericStreamingChunk] = None
        elif event_type == "message-start":
            parsed = self._reduce_message_start(event)
        elif event_type == "content-start":
            parsed = self._reduce_content_start(event)
        elif event_type == "content-delta":
            parsed = self._reduce_content_delta(event)
        elif event_type == "content-end":
            parsed = self._reduce_content_end(event)
        elif event_type == "tool-plan-delta":
            parsed = self._reduce_tool_plan_delta(event)
        elif event_type == "tool-call-start":
            parsed = self._reduce_tool_call_start(event)
        elif event_type == "tool-call-delta":
            parsed = self._reduce_tool_call_delta(event)
        elif event_type == "tool-call-end":
            parsed = self._reduce_tool_call_end(event)
        elif event_type == "citation-start":
            parsed = self._reduce_citation_start(event)
        elif event_type == "citation-end":
            parsed = self._reduce_citation_end(event)
        elif event_type == "message-end":
            parsed = self._reduce_message_end(event)
        else:
            raise ValueError("Unrecognized Cohere V2 stream event")
        return self._with_top_level_citation_fields(event, parsed)

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            parsed = self._reduce_event(chunk)
        except CohereStreamDiagnostic:
            raise
        except Exception as exc:
            _raise_classified_stream_failure(exc, payload=chunk)
        if parsed is None:
            return self._empty_chunk()
        return parsed

    def _pop_ready_chunk(self) -> Optional[GenericStreamingChunk]:
        while self._buffer.events:
            event = self._buffer.events.pop(0)
            if event is _COHERE_V2_STREAM_DONE:
                if not self._message_end_received:
                    _raise_cohere_diagnostic(
                        stage="framing",
                        payload=b"",
                        reason="missing_message_end",
                    )
                raise StopIteration
            try:
                parsed = self._reduce_event(event)
            except CohereStreamDiagnostic:
                raise
            except Exception as exc:
                _raise_classified_stream_failure(exc, payload=event)
            if parsed is not None:
                return parsed
        return None

    def _require_message_end_at_exhaustion(self) -> None:
        """Reject a drained EOF queue that never carried native message-end."""
        if not self._message_end_received:
            raise ValueError("Cohere stream ended without a native message-end event")

    def _push_source_chunk(self, chunk: Any) -> None:
        if chunk is _COHERE_V2_SOURCE_EXHAUSTED:
            self._buffer.finalize()
            self._source_exhausted = True
            if not self._buffer.events and not self._message_end_received:
                _raise_cohere_diagnostic(
                    stage="framing",
                    payload=b"",
                    reason="missing_message_end",
                )
            return
        self._buffer.feed(chunk)

    def _read_sync_chunk(self) -> Any:
        try:
            return self.response_iterator.__next__()
        except StopIteration:
            return _COHERE_V2_SOURCE_EXHAUSTED
        except Exception as exc:
            _raise_cohere_runtime(
                exc,
                payload=None,
                stage="framing",
                reason="receive_error",
            )

    async def _read_async_chunk(self) -> Any:
        try:
            return await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            return _COHERE_V2_SOURCE_EXHAUSTED
        except Exception as exc:
            _raise_cohere_runtime(
                exc,
                payload=None,
                stage="framing",
                reason="receive_error",
            )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            while True:
                parsed = self._pop_ready_chunk()
                if parsed is not None:
                    return parsed
                if self._source_exhausted:
                    self._require_message_end_at_exhaustion()
                    raise StopIteration
                self._push_source_chunk(self._read_sync_chunk())
        except StopIteration:
            raise
        except GeneratorExit:
            self.close()
            raise
        except CohereError as exc:
            _raise_iterator_terminal(exc)
        except ValueError as exc:
            _raise_iterator_parse_failure(exc)
        except Exception as exc:
            _raise_cohere_runtime(
                exc,
                payload=None,
                stage="event-schema",
                reason="event_schema",
            )

    def convert_str_chunk_to_generic_chunk(
        self, chunk: Union[str, bytes, dict]
    ) -> GenericStreamingChunk:
        """Convert one transport chunk, holding incomplete JSON or UTF-8."""
        try:
            self._buffer.feed(chunk)
            parsed = self._pop_ready_chunk()
        except CohereStreamDiagnostic:
            raise
        except ValueError as exc:
            if str(exc) in _COHERE_FIXED_TERMINAL_MESSAGES:
                raise
            _raise_classified_stream_failure(exc, payload=chunk)
        except Exception as exc:
            _raise_classified_stream_failure(exc, payload=chunk)
        if parsed is None:
            return self._empty_chunk()
        return parsed

    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            while True:
                parsed = self._pop_ready_chunk()
                if parsed is not None:
                    return parsed
                if self._source_exhausted:
                    self._require_message_end_at_exhaustion()
                    raise StopAsyncIteration
                self._push_source_chunk(await self._read_async_chunk())
        except StopAsyncIteration:
            raise
        except StopIteration:
            raise StopAsyncIteration from None
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except CohereError as exc:
            _raise_iterator_terminal(exc)
        except ValueError as exc:
            _raise_iterator_parse_failure(exc)
        except Exception as exc:
            _raise_cohere_runtime(
                exc,
                payload=None,
                stage="event-schema",
                reason="event_schema",
            )
