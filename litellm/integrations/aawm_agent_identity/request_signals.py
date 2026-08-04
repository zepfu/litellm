"""Invalid-tool-call detection, structured-output detection/classification, cache-hint scans, compact-summary classification.

Behavior-preserving Wave A4A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

import re
from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

if TYPE_CHECKING:
    import hashlib
    import json

    _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH = 0
    _AAWM_REQUEST_PAYLOAD_SCAN_MAX_ITEMS = 0

    def _clean_non_empty_string(value: Any) -> Optional[str]: ...

    def _content_to_text(content: Any) -> str: ...

    def _extract_first_langfuse_response_message(output_payload: Any) -> Any: ...

    def _extract_first_response_message(result: Any) -> Any: ...

    def _extract_provider_error_dicts(value: Any) -> List[Dict[str, Any]]: ...

    def _extract_provider_error_text(
        result: Any,
        dicts: List[Dict[str, Any]],
    ) -> str: ...

    def _first_non_empty_string(*values: Any) -> Optional[str]: ...

    def _first_non_none(*values: Any) -> Any: ...

    def _json_safe_rate_limit_value(
        value: Any,
        *,
        _seen: Optional[Set[int]] = None,
        _depth: int = 0,
    ) -> Any: ...

    def _maybe_get(obj: Any, key: str, default: Any = None) -> Any: ...

    def _maybe_get_path(
        obj: Any,
        *keys: str,
        default: Any = None,
    ) -> Any: ...

    def _metadata_bool(value: Any) -> bool: ...

    def _safe_json_load(value: Any, default: Any) -> Any: ...


_INVALID_TOOL_CALL_ERROR_RE = re.compile(
    r"("
    r"\bInputValidationError\b"
    r"|<tool_use_error>"
    r"|tool_use_error"
    r"|unexpected (?:parameter|key)"
    r"|unrecognized (?:parameter|key)"
    r"|unknown (?:parameter|key)"
    r"|invalid tool(?: call| use)?"
    r"|tool call validation"
    r"|unable to parse tool parameter json"
    r"|failed due to the following issue"
    r")",
    re.IGNORECASE,
)
_TOOL_RESULT_ERROR_BLOCK_TYPES = {
    "tool_result",
    "tool_use_result",
    "function_call_output",
}


def _invalid_tool_call_error_text_seen(value: Any) -> bool:
    parsed = _safe_json_load(value, value)
    if isinstance(parsed, str):
        return bool(_INVALID_TOOL_CALL_ERROR_RE.search(parsed))
    if isinstance(parsed, dict):
        for key in (
            "content",
            "text",
            "output",
            "error",
            "message",
            "status",
            "name",
            "type",
        ):
            if key in parsed and _invalid_tool_call_error_text_seen(parsed[key]):
                return True
        return False
    if isinstance(parsed, list):
        return any(_invalid_tool_call_error_text_seen(item) for item in parsed)
    return False


def _iter_tool_result_error_candidates(message: Any) -> Iterator[Any]:
    parsed_message = _safe_json_load(message, message)
    if not isinstance(parsed_message, dict):
        return

    message_type = _clean_non_empty_string(parsed_message.get("type"))
    message_role = _clean_non_empty_string(parsed_message.get("role"))
    if message_type in _TOOL_RESULT_ERROR_BLOCK_TYPES or (message_role or "").lower() == "tool":
        yield parsed_message

    content = _safe_json_load(parsed_message.get("content"), parsed_message.get("content"))
    if isinstance(content, dict):
        content_blocks = [content]
    elif isinstance(content, list):
        content_blocks = content
    else:
        content_blocks = []

    for block in content_blocks:
        parsed_block = _safe_json_load(block, block)
        if not isinstance(parsed_block, dict):
            continue
        block_type = _clean_non_empty_string(parsed_block.get("type"))
        if block_type in _TOOL_RESULT_ERROR_BLOCK_TYPES:
            yield parsed_block


def _iter_request_message_payloads(request_body: Dict[str, Any]) -> Iterator[Any]:
    for key in ("messages", "input"):
        value = request_body.get(key)
        parsed = _safe_json_load(value, value)
        if isinstance(parsed, list):
            yield from parsed
        elif isinstance(parsed, dict):
            yield parsed

    nested_request = _safe_json_load(request_body.get("request"), request_body.get("request"))
    if isinstance(nested_request, dict) and nested_request is not request_body:
        yield from _iter_request_message_payloads(nested_request)


def _extract_invalid_tool_call_count_from_request_body(
    request_body: Optional[Dict[str, Any]],
) -> int:
    if not isinstance(request_body, dict):
        return 0

    invalid_count = 0
    for message in _iter_request_message_payloads(request_body):
        for candidate in _iter_tool_result_error_candidates(message):
            if _invalid_tool_call_error_text_seen(candidate):
                invalid_count += 1
    return invalid_count


_STRUCTURED_OUTPUT_JSON_MODE_VALUES = {
    "json",
    "json_object",
    "json_schema",
    "schema",
    "response_schema",
}
_STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS = (
    "body",
    "data",
    "json",
    "payload",
    "request",
    "request_body",
)
_STRUCTURED_OUTPUT_FAILURE_PATTERNS = (
    (
        "schema_validation_error",
        re.compile(
            r"("
            r"structured[-_ ]?output"
            r"|json[-_ ]?schema"
            r"|schema validation"
            r"|validation schema"
            r"|invalid schema"
            r"|schema .*valid"
            r"|does not match (?:the )?schema"
            r"|pydantic"
            r"|jsonschema"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "json_validation_error",
        re.compile(
            r"("
            r"invalid[-_ ]?json"
            r"|malformed json"
            r"|json parse"
            r"|parse json"
            r"|json decode"
            r"|json validation"
            r"|validate json"
            r"|json .*valid"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "response_format_error",
        re.compile(r"(response[-_ ]?format|invalid_response_format)", re.IGNORECASE),
    ),
)


def _empty_structured_output_state() -> Dict[str, Any]:
    return {
        "structured_output_attempted": False,
        "structured_output_failed": False,
        "structured_output_mode": None,
        "structured_output_schema_hash": None,
        "structured_output_failure_reason": None,
    }


def _merge_structured_output_state(
    current: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(candidate, dict) or not candidate.get("structured_output_attempted"):
        return current

    current["structured_output_attempted"] = True
    current["structured_output_failed"] = bool(
        current.get("structured_output_failed") or candidate.get("structured_output_failed")
    )
    for key in (
        "structured_output_mode",
        "structured_output_schema_hash",
        "structured_output_failure_reason",
    ):
        value = _clean_non_empty_string(candidate.get(key))
        if value and not _clean_non_empty_string(current.get(key)):
            current[key] = value
    return current


def _structured_output_schema_hash(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        encoded = json.dumps(
            _json_safe_rate_limit_value(value),
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        encoded = str(value)
    if not encoded or encoded in {"null", "{}", "[]"}:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _structured_output_state_from_format(
    value: Any,
    *,
    default_mode: Optional[str] = None,
) -> Dict[str, Any]:
    parsed = _safe_json_load(value, value)
    state = _empty_structured_output_state()

    if isinstance(parsed, str):
        mode = parsed.strip().lower().replace("-", "_")
        if mode in _STRUCTURED_OUTPUT_JSON_MODE_VALUES or "json" in mode:
            state["structured_output_attempted"] = True
            state["structured_output_mode"] = mode
        return state

    if not isinstance(parsed, dict):
        return state

    raw_mode = _first_non_empty_string(
        parsed.get("type"),
        parsed.get("format"),
        parsed.get("mode"),
        default_mode,
    )
    dict_mode = raw_mode.lower().replace("-", "_") if raw_mode else None
    schema = _first_non_none(
        parsed.get("json_schema"),
        parsed.get("schema"),
        parsed.get("response_schema"),
        parsed.get("responseSchema"),
    )
    mime_type = _first_non_empty_string(
        parsed.get("response_mime_type"),
        parsed.get("responseMimeType"),
        parsed.get("mime_type"),
    )
    has_json_mime = bool(mime_type and "json" in mime_type.lower())
    has_json_mode = bool(
        dict_mode and (dict_mode in _STRUCTURED_OUTPUT_JSON_MODE_VALUES or "json" in dict_mode or "schema" in dict_mode)
    )
    if schema is None and not has_json_mode and not has_json_mime:
        return state

    state["structured_output_attempted"] = True
    state["structured_output_mode"] = dict_mode or ("response_schema" if schema is not None else "json_mime_type")
    state["structured_output_schema_hash"] = _structured_output_schema_hash(schema)
    return state


def _structured_output_state_from_generation_config(value: Any) -> Dict[str, Any]:
    parsed = _safe_json_load(value, value)
    state = _empty_structured_output_state()
    if not isinstance(parsed, dict):
        return state

    schema = _first_non_none(
        parsed.get("responseSchema"),
        parsed.get("response_schema"),
    )
    mime_type = _first_non_empty_string(
        parsed.get("responseMimeType"),
        parsed.get("response_mime_type"),
    )
    if schema is None and not (mime_type and "json" in mime_type.lower()):
        return state

    state["structured_output_attempted"] = True
    state["structured_output_mode"] = "response_schema" if schema is not None else "json_mime_type"
    state["structured_output_schema_hash"] = _structured_output_schema_hash(schema)
    return state


def _detect_structured_output_request(
    request_body: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = _empty_structured_output_state()

    if isinstance(metadata, dict):
        metadata_attempted = any(
            key in metadata and _metadata_bool(metadata.get(key))
            for key in (
                "usage_structured_output_attempted",
                "structured_output_attempted",
            )
        )
        metadata_failed = any(
            key in metadata and _metadata_bool(metadata.get(key))
            for key in (
                "usage_structured_output_failed",
                "structured_output_failed",
            )
        )
        metadata_mode = _first_non_empty_string(
            metadata.get("usage_structured_output_mode"),
            metadata.get("structured_output_mode"),
        )
        metadata_schema_hash = _first_non_empty_string(
            metadata.get("usage_structured_output_schema_hash"),
            metadata.get("structured_output_schema_hash"),
        )
        metadata_reason = _first_non_empty_string(
            metadata.get("usage_structured_output_failure_reason"),
            metadata.get("structured_output_failure_reason"),
        )
        if metadata_attempted or metadata_failed or metadata_mode or metadata_schema_hash:
            state["structured_output_attempted"] = True
            state["structured_output_failed"] = metadata_failed
            state["structured_output_mode"] = metadata_mode
            state["structured_output_schema_hash"] = metadata_schema_hash
            state["structured_output_failure_reason"] = metadata_reason

    parsed_request = _safe_json_load(request_body, request_body)
    if not isinstance(parsed_request, dict):
        return state

    pending: List[Tuple[Any, int]] = [(parsed_request, 0)]
    seen: set[int] = set()
    while pending:
        payload, depth = pending.pop(0)
        if not isinstance(payload, dict):
            continue
        payload_id = id(payload)
        if payload_id in seen:
            continue
        seen.add(payload_id)

        for key in ("response_format", "responseFormat"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_format(payload.get(key)),
                )

        text_config = _safe_json_load(payload.get("text"), payload.get("text"))
        if isinstance(text_config, dict) and "format" in text_config:
            _merge_structured_output_state(
                state,
                _structured_output_state_from_format(text_config.get("format")),
            )

        for key in ("text_format", "textFormat"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_format(payload.get(key)),
                )

        for key in ("output_format", "outputFormat", "output_config", "outputConfig"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_format(payload.get(key)),
                )

        for key in ("generationConfig", "generation_config"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_generation_config(payload.get(key)),
                )

        if "response_schema" in payload or "responseSchema" in payload:
            schema = _first_non_none(
                payload.get("response_schema"),
                payload.get("responseSchema"),
            )
            _merge_structured_output_state(
                state,
                {
                    "structured_output_attempted": True,
                    "structured_output_failed": False,
                    "structured_output_mode": "response_schema",
                    "structured_output_schema_hash": _structured_output_schema_hash(schema),
                    "structured_output_failure_reason": None,
                },
            )

        mime_type = _first_non_empty_string(
            payload.get("response_mime_type"),
            payload.get("responseMimeType"),
        )
        if mime_type and "json" in mime_type.lower():
            _merge_structured_output_state(
                state,
                {
                    "structured_output_attempted": True,
                    "structured_output_failed": False,
                    "structured_output_mode": "json_mime_type",
                    "structured_output_schema_hash": None,
                    "structured_output_failure_reason": None,
                },
            )

        if depth >= 4:
            continue
        for key in _STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS:
            nested = _safe_json_load(payload.get(key), payload.get(key))
            if isinstance(nested, dict):
                pending.append((nested, depth + 1))

    return state


def _collect_structured_output_failure_texts(value: Any) -> List[str]:
    texts: List[str] = []
    pending: List[Tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    while pending and len(texts) < 40:
        current, depth = pending.pop(0)
        current = _safe_json_load(current, current)
        if isinstance(current, str):
            if current.strip():
                texts.append(current.strip()[:1000])
            continue
        if isinstance(current, dict):
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            for key in (
                "message",
                "error",
                "detail",
                "details",
                "code",
                "type",
                "statusMessage",
                "status_message",
            ):
                if key in current:
                    pending.append((current[key], depth + 1))
            if depth < 3:
                for nested_value in list(current.values()):
                    if isinstance(nested_value, (dict, list)):
                        pending.append((nested_value, depth + 1))
            continue
        if isinstance(current, list) and depth < 3:
            for item in current[:40]:
                pending.append((item, depth + 1))
    return texts


def _classify_structured_output_failure(value: Any) -> Optional[str]:
    dicts = _extract_provider_error_dicts(value)
    error_text = _extract_provider_error_text(value, dicts)
    texts = [error_text] if error_text else []
    texts.extend(_collect_structured_output_failure_texts(value))
    combined = "\n".join(text for text in texts if isinstance(text, str))[:5000]
    if not combined.strip():
        return None
    for reason, pattern in _STRUCTURED_OUTPUT_FAILURE_PATTERNS:
        if pattern.search(combined):
            return reason
    return None


def _extract_request_body_from_langfuse_input(value: Any) -> Optional[Dict[str, Any]]:
    parsed = _safe_json_load(value, value)
    if not isinstance(parsed, dict):
        return None

    messages = parsed.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            nested = _safe_json_load(message.get("content"), None)
            if isinstance(nested, dict) and (
                isinstance(nested.get("messages"), list)
                or isinstance(nested.get("input"), (str, list, dict))
                or isinstance(nested.get("instructions"), str)
                or isinstance(nested.get("model"), str)
            ):
                return nested
        return parsed

    body = parsed.get("body")
    if isinstance(body, dict):
        return _extract_request_body_from_langfuse_input(body)
    return None


def _request_payload_contains(
    payload: Any,
    predicate: Any,
) -> bool:
    pending: List[Tuple[Any, int]] = [(payload, 0)]
    seen: Set[int] = set()
    scanned = 0

    while pending and scanned < _AAWM_REQUEST_PAYLOAD_SCAN_MAX_ITEMS:
        value, depth = pending.pop()
        scanned += 1

        if isinstance(value, dict):
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)

            if predicate(value):
                return True
            if depth >= _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
                continue
            pending.extend(
                (nested_value, depth + 1)
                for nested_value in list(value.values())
                if isinstance(nested_value, (dict, list, tuple))
            )
            continue

        if isinstance(value, (list, tuple)):
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)

            if depth >= _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
                continue
            pending.extend((item, depth + 1) for item in list(value) if isinstance(item, (dict, list, tuple)))

    return False
_CODEX_THREAD_ID_RE = re.compile(r"\bCODEX_THREAD_ID=(?P<thread_id>[A-Za-z0-9][A-Za-z0-9._:-]{7,})\b")
_CLAUDE_CODE_COMPACT_REQUEST_MARKERS = (
    "your task is to create a detailed summary of the conversation so far",
    "respond with text only",
    "do not call any tools",
)


def _append_request_content_text(texts: List[str], content: Any) -> None:
    text = _content_to_text(content).strip()
    if text:
        texts.append(text)


def _extract_request_user_texts(request_body: Any) -> List[str]:
    if not isinstance(request_body, dict):
        return []

    texts: List[str] = []
    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() == "user":
                _append_request_content_text(texts, message.get("content"))

    input_items = request_body.get("input")
    if isinstance(input_items, str):
        texts.append(input_items.strip())
    elif isinstance(input_items, list):
        for item in input_items:
            if isinstance(item, str):
                if item.strip():
                    texts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            role = str(item.get("role") or "").lower()
            if item_type == "input_text":
                _append_request_content_text(texts, item.get("text"))
            elif role == "user" and item_type in {"", "message"}:
                _append_request_content_text(texts, item.get("content"))

    return texts


def _join_compact_request_user_texts(request_body: Any) -> str:
    return "\n".join(_extract_request_user_texts(request_body))


def _extract_codex_compact_thread_id(
    metadata: Dict[str, Any],
    request_body: Any,
    request_text: str,
) -> Optional[str]:
    if isinstance(request_body, dict):
        prompt_cache_key = _clean_non_empty_string(request_body.get("prompt_cache_key"))
        if prompt_cache_key is not None:
            return prompt_cache_key

    for candidate in (
        metadata.get("prompt_cache_key"),
        metadata.get("codex_prompt_cache_key"),
        metadata.get("CODEX_THREAD_ID"),
        metadata.get("codex_thread_id"),
    ):
        thread_id = _clean_non_empty_string(candidate)
        if thread_id is not None:
            return thread_id

    match = _CODEX_THREAD_ID_RE.search(request_text)
    if match:
        return match.group("thread_id")
    return None


def _is_claude_code_compact_context(metadata: Dict[str, Any]) -> bool:
    client_name = str(metadata.get("client_name") or "").strip().lower()
    trace_name = str(metadata.get("trace_name") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    return (
        client_name in {"claude-cli", "claude-code"}
        or trace_name.startswith("claude-code")
        or route_family in {"anthropic_messages", "anthropic_completion"}
    )


def _is_codex_compact_context(metadata: Dict[str, Any]) -> bool:
    client_name = str(metadata.get("client_name") or "").strip().lower()
    trace_name = str(metadata.get("trace_name") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    return client_name == "codex-tui" or trace_name.startswith("codex") or route_family == "codex_responses"


def _classify_compact_summary_state(
    *,
    metadata: Dict[str, Any],
    request_body: Any,
    output_payload: Any,
    session_id: Optional[str],
    litellm_call_id: Optional[str],
    trace_id: Optional[str],
) -> Dict[str, Any]:
    request_text = _join_compact_request_user_texts(request_body)
    request_text_lower = request_text.lower()

    if _is_codex_compact_context(metadata):
        compact_id = _extract_codex_compact_thread_id(
            metadata,
            request_body,
            request_text,
        )
        if "context checkpoint compaction" in request_text_lower:
            return {
                "is_compact_summary": True,
                "compact_summary_source": "codex",
                "compact_summary_role": "event",
                "compact_summary_id": compact_id or litellm_call_id or trace_id or session_id,
            }
        if "another language model started to solve this problem" in request_text_lower:
            return {
                "is_compact_summary": False,
                "compact_summary_source": "codex",
                "compact_summary_role": "resume_context",
                "compact_summary_id": compact_id or session_id,
            }

    if _is_claude_code_compact_context(metadata):
        has_compact_tags = "<analysis>" in request_text_lower and "<summary>" in request_text_lower
        strict_prompt_shape = all(marker in request_text_lower for marker in _CLAUDE_CODE_COMPACT_REQUEST_MARKERS)
        compact_summary_phrase = (
            "summarize the current context" in request_text_lower or "context compacted" in request_text_lower
        )
        if has_compact_tags and (strict_prompt_shape or compact_summary_phrase):
            compact_id = litellm_call_id or trace_id or session_id
            return {
                "is_compact_summary": True,
                "compact_summary_source": "claude-code",
                "compact_summary_role": "event",
                "compact_summary_id": compact_id,
            }

    return {
        "is_compact_summary": False,
        "compact_summary_source": None,
        "compact_summary_role": None,
        "compact_summary_id": None,
    }


_HOST_FUNCTION_NAMES = (
    "_INVALID_TOOL_CALL_ERROR_RE",
    "_TOOL_RESULT_ERROR_BLOCK_TYPES",
    "_invalid_tool_call_error_text_seen",
    "_iter_tool_result_error_candidates",
    "_iter_request_message_payloads",
    "_extract_invalid_tool_call_count_from_request_body",
    "_STRUCTURED_OUTPUT_JSON_MODE_VALUES",
    "_STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS",
    "_STRUCTURED_OUTPUT_FAILURE_PATTERNS",
    "_empty_structured_output_state",
    "_merge_structured_output_state",
    "_structured_output_schema_hash",
    "_structured_output_state_from_format",
    "_structured_output_state_from_generation_config",
    "_detect_structured_output_request",
    "_collect_structured_output_failure_texts",
    "_classify_structured_output_failure",
    "_extract_request_body_from_langfuse_input",
    "_request_payload_contains",
    "_CODEX_THREAD_ID_RE",
    "_CLAUDE_CODE_COMPACT_REQUEST_MARKERS",
    "_append_request_content_text",
    "_extract_request_user_texts",
    "_join_compact_request_user_texts",
    "_extract_codex_compact_thread_id",
    "_is_claude_code_compact_context",
    "_is_codex_compact_context",
    "_classify_compact_summary_state",
)


def _rebind_to_host_globals(fn, host_globals):
    rebound = _FunctionType(
        fn.__code__,
        host_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = getattr(fn, "__annotations__", {})
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    rebound.__doc__ = fn.__doc__
    return rebound


def _rebind_installable_callable(value, host_globals):
    if isinstance(value, _FunctionType):
        return _rebind_to_host_globals(value, host_globals)

    wrapped = getattr(value, "__wrapped__", None)
    cache_parameters = getattr(value, "cache_parameters", None)
    if not isinstance(wrapped, _FunctionType) or not callable(cache_parameters):
        return value

    parameters = cache_parameters()
    if not isinstance(parameters, dict) or not {"maxsize", "typed"} <= parameters.keys():
        return value

    rebound_wrapped = _rebind_to_host_globals(wrapped, host_globals)
    rebound = lru_cache(
        maxsize=parameters["maxsize"],
        typed=bool(parameters["typed"]),
    )(rebound_wrapped)
    for attribute, attribute_value in getattr(value, "__dict__", {}).items():
        if attribute != "__wrapped__":
            setattr(rebound, attribute, attribute_value)
    return rebound


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        _installed = _rebind_installable_callable(_original, host_globals)
        mod[_name] = _installed
        host_globals[_name] = _installed
