"""Prompt-overhead buckets/components/breakdown, rerank token estimates.

Behavior-preserving Wave A4A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    import json
    import re

    from litellm._logging import verbose_logger

    _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH = 0
    _PROMPT_OVERHEAD_CLASSIFIER_VERSION = ""
    _PROMPT_OVERHEAD_TOKEN_FIELDS: Tuple[str, ...] = ()

    def _coerce_rerank_text(value: Any) -> str: ...

    def _coerce_usage_object_to_dict(
        usage_obj: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _extract_completion_tokens(usage_obj: Any) -> int: ...

    def _extract_prompt_tokens(usage_obj: Any) -> int: ...

    def _extract_rerank_document_text(
        document: Any,
        rank_fields: Optional[List[str]],
    ) -> str: ...

    def _extract_rerank_request_payload(
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]: ...

    def _extract_total_tokens(
        usage_obj: Any,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> int: ...

    def _get_litellm_module() -> Any: ...

    def _maybe_get_path(
        obj: Any,
        *keys: str,
        default: Any = None,
    ) -> Any: ...

    def _safe_int(value: Any) -> Optional[int]: ...


def _fallback_text_token_estimate(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _empty_prompt_overhead_breakdown() -> Dict[str, Any]:
    return {field: 0 for field in _PROMPT_OVERHEAD_TOKEN_FIELDS}


def _serialize_prompt_overhead_component(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def _estimate_prompt_overhead_tokens(model: str, value: Any) -> int:
    text = _serialize_prompt_overhead_component(value).strip()
    if not text:
        return 0
    try:
        litellm = _get_litellm_module()
        token_count = litellm.token_counter(model=model or "", text=text)
        coerced = _safe_int(token_count)
        if coerced is not None and coerced >= 0:
            return coerced
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate prompt-overhead tokens for model=%s: %s",
            model,
            exc,
        )
    return _fallback_text_token_estimate(text)


def _extract_prompt_text_blocks(
    value: Any,
    *,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> List[str]:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
        return []
    if value is None:
        return []
    if isinstance(value, str):
        return [block.strip() for block in re.split(r"\n{2,}", value) if block.strip()]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        blocks: List[str] = []
        for item in value:
            blocks.extend(
                _extract_prompt_text_blocks(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            )
        return blocks
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        blocks = []
        for key in ("text", "content", "parts", "systemInstruction", "system_instruction"):
            if key in value:
                blocks.extend(
                    _extract_prompt_text_blocks(
                        value.get(key),
                        _seen=_seen,
                        _depth=_depth + 1,
                    )
                )
        if blocks:
            return blocks
        return [_serialize_prompt_overhead_component(value)]
    return [str(value)]


def _classify_system_prompt_block(block: str) -> str:
    lowered = block.lower()
    safety_markers = (
        "safety",
        "unsafe",
        "policy",
        "refuse",
        "disallowed",
        "forbidden",
        "harm",
        "malicious",
        "secret",
        "credential",
        "privacy",
        "security",
        "do not reveal",
        "never reveal",
    )
    if any(marker in lowered for marker in safety_markers):
        return "safety"

    behavior_markers = (
        "you are",
        "persona",
        "personality",
        "tone",
        "style",
        "respond as",
        "communication",
        "be concise",
        "be direct",
    )
    if any(marker in lowered for marker in behavior_markers):
        return "behavior"

    instructional_markers = (
        "always",
        "must",
        "should",
        "use ",
        "follow",
        "workflow",
        "steps",
        "when ",
        "before ",
        "after ",
        "tool",
        "repository",
        "codebase",
        "task",
        "instruction",
    )
    if any(marker in lowered for marker in instructional_markers):
        return "instructional"
    return "unclassified"


def _estimate_system_prompt_bucket_tokens(
    *,
    model: str,
    system_components: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[str]]:
    bucket_tokens = {
        "behavior": 0,
        "safety": 0,
        "instructional": 0,
        "unclassified": 0,
    }
    component_paths: List[str] = []
    for component in system_components:
        path = str(component.get("path") or "system")
        value = component.get("value")
        blocks = _extract_prompt_text_blocks(value)
        if not blocks:
            continue
        component_paths.append(path)
        for block in blocks:
            bucket = _classify_system_prompt_block(block)
            bucket_tokens[bucket] += _estimate_prompt_overhead_tokens(model, block)
    return bucket_tokens, component_paths


def _append_prompt_component(
    components: Dict[str, List[Dict[str, Any]]],
    name: str,
    *,
    path: str,
    value: Any,
) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    if isinstance(value, list) and not value:
        return
    if isinstance(value, dict) and not value:
        return
    components[name].append({"path": path, "value": value})


_RESPONSES_SYSTEM_ROLES = {"system", "developer"}
_RESPONSES_CONVERSATION_ROLES = {"user", "assistant"}
_RESPONSES_TEXT_CONTENT_TYPES = {"input_text", "output_text", "text"}
_RESPONSES_OPAQUE_CONTENT_TYPES = {
    "item_reference",
    "input_audio",
    "audio",
    "input_image",
    "image",
    "image_url",
}
_RESPONSES_OPAQUE_ITEM_TYPES = {
    "reasoning",
    "function_call",
    "mcp_call",
    "file_search_call",
    "web_search_call",
    "computer_call",
    "item_reference",
}


def _append_prompt_text_components(
    components: Dict[str, List[Dict[str, Any]]],
    name: str,
    *,
    path: str,
    values: List[str],
) -> None:
    for value in values:
        _append_prompt_component(components, name, path=path, value=value)


def _extract_responses_visible_text_blocks(
    value: Any,
    *,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> List[str]:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
        return []
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        blocks: List[str] = []
        for item in value:
            blocks.extend(
                _extract_responses_visible_text_blocks(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            )
        return blocks
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        content_type = str(value.get("type") or "").lower()
        if content_type in _RESPONSES_OPAQUE_CONTENT_TYPES:
            return []
        if content_type in _RESPONSES_TEXT_CONTENT_TYPES:
            text = value.get("text")
            return [text.strip()] if isinstance(text, str) and text.strip() else []
        if "text" in value and isinstance(value.get("text"), str):
            text = value["text"].strip()
            return [text] if text else []
        if "content" in value:
            return _extract_responses_visible_text_blocks(
                value.get("content"),
                _seen=_seen,
                _depth=_depth + 1,
            )
    return []


def _responses_message_component_path(role: str) -> str:
    if role in _RESPONSES_SYSTEM_ROLES:
        return "input[type=message][role=system|developer].content"
    if role in _RESPONSES_CONVERSATION_ROLES:
        return f"input[type=message][role={role}].content"
    return "input[type=message].content"


def _record_responses_excluded_fields(
    components: Dict[str, List[Dict[str, Any]]],
    value: Any,
    *,
    path: str,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> None:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
        return
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return
        _seen.add(value_id)
        for item in value:
            _record_responses_excluded_fields(
                components,
                item,
                path=path,
                _seen=_seen,
                _depth=_depth + 1,
            )
        return
    if not isinstance(value, dict):
        return
    value_id = id(value)
    if value_id in _seen:
        return
    _seen.add(value_id)
    content_type = str(value.get("type") or "").lower()
    if content_type == "item_reference":
        _append_prompt_component(
            components,
            "excluded",
            path=f"{path}[type=item_reference]",
            value=value,
        )
        return
    for key, field_value in value.items():
        if key in {"encrypted_content", "reasoning_content"}:
            _append_prompt_component(
                components,
                "excluded",
                path=f"{path}.{key}",
                value=field_value,
            )
        elif isinstance(field_value, (dict, list)):
            _record_responses_excluded_fields(
                components,
                field_value,
                path=f"{path}.{key}",
                _seen=_seen,
                _depth=_depth + 1,
            )


def _append_openai_responses_input_component(
    components: Dict[str, List[Dict[str, Any]]],
    item: Any,
) -> None:
    if isinstance(item, str):
        _append_prompt_component(
            components,
            "conversation",
            path="input",
            value=item,
        )
        return

    if not isinstance(item, dict):
        _append_prompt_component(
            components,
            "conversation",
            path="input",
            value=item,
        )
        return

    item_type = str(item.get("type") or "").lower()
    role = str(item.get("role") or "").lower()
    if item_type in _RESPONSES_OPAQUE_ITEM_TYPES:
        _append_prompt_component(
            components,
            "excluded",
            path=f"input[type={item_type}]",
            value=item,
        )
        return

    if item_type == "function_call_output":
        _append_prompt_component(
            components,
            "conversation",
            path="input[type=function_call_output].output",
            value=item.get("output"),
        )
        return

    _record_responses_excluded_fields(
        components,
        item,
        path=f"input[type={item_type or 'unknown'}]",
    )

    if item_type == "message" or role:
        bucket = "system" if role in _RESPONSES_SYSTEM_ROLES else "conversation"
        path = _responses_message_component_path(role)
        text_blocks = _extract_responses_visible_text_blocks(item.get("content"))
        if not text_blocks and "content" not in item:
            text_blocks = _extract_responses_visible_text_blocks(item)
        _append_prompt_text_components(
            components,
            bucket,
            path=path,
            values=text_blocks,
        )
        return

    text_blocks = _extract_responses_visible_text_blocks(item)
    if text_blocks:
        _append_prompt_text_components(
            components,
            "conversation",
            path="input[type=visible_text]",
            values=text_blocks,
        )
    else:
        _append_prompt_component(
            components,
            "excluded",
            path=f"input[type={item_type or 'unknown'}]",
            value=item,
        )


def _append_openai_responses_input_components(
    components: Dict[str, List[Dict[str, Any]]],
    input_value: Any,
) -> None:
    if isinstance(input_value, list):
        for item in input_value:
            _append_openai_responses_input_component(components, item)
        return
    _append_openai_responses_input_component(components, input_value)


def _split_chat_prompt_messages(messages: Any) -> Tuple[List[Any], List[Any]]:
    if not isinstance(messages, list):
        return [], []
    system_messages: List[Any] = []
    conversation_messages: List[Any] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}:
            system_messages.append(message)
        else:
            conversation_messages.append(message)
    return system_messages, conversation_messages


def _extract_prompt_overhead_components(
    request_body: Dict[str, Any],
    route_family: Optional[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
    components: Dict[str, List[Dict[str, Any]]] = {
        "system": [],
        "tools": [],
        "conversation": [],
        "excluded": [],
    }
    route_family_lower = (route_family or "").lower()
    request_block = request_body.get("request")
    is_nested_gemini = isinstance(request_block, dict) and (
        "gemini" in route_family_lower
        or "google" in route_family_lower
        or "contents" in request_block
        or "systemInstruction" in request_block
    )
    if is_nested_gemini:
        nested_request_block = request_block if isinstance(request_block, dict) else {}
        _append_prompt_component(
            components,
            "system",
            path="request.systemInstruction",
            value=nested_request_block.get("systemInstruction") or nested_request_block.get("system_instruction"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="request.tools",
            value=nested_request_block.get("tools") or request_body.get("tools"),
        )
        _append_prompt_component(
            components,
            "conversation",
            path="request.contents",
            value=nested_request_block.get("contents"),
        )
        return components, "gemini_generate_content"

    if request_body.get("systemInstruction") is not None or request_body.get("contents") is not None:
        _append_prompt_component(
            components,
            "system",
            path="systemInstruction",
            value=request_body.get("systemInstruction") or request_body.get("system_instruction"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="tools",
            value=request_body.get("tools"),
        )
        _append_prompt_component(
            components,
            "conversation",
            path="contents",
            value=request_body.get("contents"),
        )
        return components, "gemini_generate_content"

    if request_body.get("instructions") is not None or request_body.get("input") is not None:
        _append_prompt_component(
            components,
            "system",
            path="instructions",
            value=request_body.get("instructions"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="tools",
            value=request_body.get("tools"),
        )
        _append_openai_responses_input_components(
            components,
            request_body.get("input"),
        )
        return components, "openai_responses"

    if request_body.get("messages") is not None:
        if request_body.get("system") is not None:
            _append_prompt_component(
                components,
                "system",
                path="system",
                value=request_body.get("system"),
            )
            _append_prompt_component(
                components,
                "conversation",
                path="messages",
                value=request_body.get("messages"),
            )
            counted_shape = (
                "anthropic_messages_semantic"
                if "anthropic" in route_family_lower
                else "chat_messages_with_top_level_system"
            )
        else:
            system_messages, conversation_messages = _split_chat_prompt_messages(request_body.get("messages"))
            _append_prompt_component(
                components,
                "system",
                path="messages[role=system|developer]",
                value=system_messages,
            )
            _append_prompt_component(
                components,
                "conversation",
                path="messages[role!=system|developer]",
                value=conversation_messages,
            )
            counted_shape = "openai_chat_completions"
        _append_prompt_component(
            components,
            "tools",
            path="tools",
            value=request_body.get("tools"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="mcp_servers",
            value=request_body.get("mcp_servers"),
        )
        return components, counted_shape

    return components, "unknown"


def _build_prompt_overhead_breakdown(
    *,
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    model: str,
    prompt_tokens: int,
    request_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    breakdown = _empty_prompt_overhead_breakdown()
    if not isinstance(request_body, dict) or prompt_tokens <= 0:
        return breakdown

    route_family = metadata.get("passthrough_route_family")
    if not isinstance(route_family, str) or not route_family.strip():
        route_family = _maybe_get_path(
            kwargs.get("passthrough_logging_payload"),
            "request_body",
            "litellm_metadata",
            "passthrough_route_family",
        )
    route_family = route_family if isinstance(route_family, str) else None

    components, counted_shape = _extract_prompt_overhead_components(
        request_body,
        route_family,
    )
    bucket_tokens, system_paths = _estimate_system_prompt_bucket_tokens(
        model=model,
        system_components=components["system"],
    )
    system_tokens = sum(bucket_tokens.values())
    tool_tokens = sum(_estimate_prompt_overhead_tokens(model, component["value"]) for component in components["tools"])
    conversation_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"]) for component in components["conversation"]
    )
    excluded_components = components.get("excluded", [])
    opaque_state_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"]) for component in excluded_components
    )
    component_total = system_tokens + tool_tokens + conversation_tokens
    residual_tokens = prompt_tokens - component_total

    breakdown.update(
        {
            "input_system_tokens_estimated": system_tokens,
            "input_tool_advertisement_tokens_estimated": tool_tokens,
            "input_conversation_tokens_estimated": conversation_tokens,
            "input_other_tokens_estimated": max(residual_tokens, 0),
            "input_breakdown_residual_tokens": residual_tokens,
            "system_behavior_tokens_estimated": bucket_tokens["behavior"],
            "system_safety_tokens_estimated": bucket_tokens["safety"],
            "system_instructional_tokens_estimated": bucket_tokens["instructional"],
            "system_unclassified_tokens_estimated": bucket_tokens["unclassified"],
        }
    )

    component_paths = {
        "system": system_paths,
        "tools": [str(component.get("path")) for component in components["tools"]],
        "conversation": [str(component.get("path")) for component in components["conversation"]],
    }
    excluded_component_paths = [str(component.get("path")) for component in excluded_components]
    metadata.update(
        {
            "prompt_overhead_breakdown_source": "request_body_estimate",
            "prompt_overhead_counted_shape": counted_shape,
            "prompt_overhead_route_family": route_family,
            "prompt_overhead_tokenizer": "litellm.token_counter_with_char_fallback",
            "prompt_overhead_classifier_version": _PROMPT_OVERHEAD_CLASSIFIER_VERSION,
            "prompt_overhead_component_paths": component_paths,
            "prompt_overhead_excluded_component_paths": excluded_component_paths,
            "usage_input_opaque_state_tokens_estimated": opaque_state_tokens,
        }
    )
    for key, value in breakdown.items():
        metadata[f"usage_{key}"] = value
    return breakdown


def _estimate_rerank_request_tokens(
    *,
    kwargs: Dict[str, Any],
    model: str,
) -> Optional[int]:
    request_payload = _extract_rerank_request_payload(kwargs)
    if not request_payload:
        return None

    query_text = _coerce_rerank_text(request_payload.get("query")).strip()
    documents = request_payload.get("documents")
    if documents is None:
        documents = request_payload.get("texts")
    if not isinstance(documents, list):
        return None

    raw_rank_fields = request_payload.get("rank_fields")
    rank_fields = raw_rank_fields if isinstance(raw_rank_fields, list) else None
    document_texts = [
        text for document in documents if (text := _extract_rerank_document_text(document, rank_fields).strip())
    ]
    combined_text = "\n\n".join([query_text, *document_texts]).strip()
    if not combined_text:
        return None

    try:
        litellm = _get_litellm_module()
        token_count = litellm.token_counter(model=model or "", text=combined_text)
        return _positive_int_or_none(token_count)
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate rerank tokens for model=%s: %s",
            model,
            exc,
        )
        return _fallback_text_token_estimate(combined_text)


def _usage_has_positive_tokens(usage_obj: Any) -> bool:
    prompt_tokens = _extract_prompt_tokens(usage_obj)
    completion_tokens = _extract_completion_tokens(usage_obj)
    total_tokens = _extract_total_tokens(usage_obj, prompt_tokens, completion_tokens)
    return prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0


def _merge_estimated_rerank_tokens_into_usage(
    *,
    kwargs: Dict[str, Any],
    result: Any,
    usage_obj: Any,
    model: str,
) -> Any:
    usage_dict = _coerce_usage_object_to_dict(usage_obj)
    if usage_dict is None:
        return usage_obj
    if _usage_has_positive_tokens(usage_dict):
        return usage_obj

    search_units = _safe_int(usage_dict.get("search_units")) or _safe_int(
        _maybe_get_path(result, "meta", "billed_units", "search_units")
    )
    if not search_units:
        return usage_obj

    estimated_tokens = _estimate_rerank_request_tokens(kwargs=kwargs, model=model)
    if estimated_tokens is None:
        return usage_obj

    merged_usage = dict(usage_dict)
    merged_usage.setdefault("prompt_tokens", estimated_tokens)
    merged_usage.setdefault("completion_tokens", 0)
    merged_usage.setdefault("total_tokens", estimated_tokens)
    return merged_usage


def _positive_int_or_none(value: Any) -> Optional[int]:
    normalized = _safe_int(value)
    if normalized is not None and normalized > 0:
        return normalized
    return None


_HOST_FUNCTION_NAMES = (
    "_fallback_text_token_estimate",
    "_empty_prompt_overhead_breakdown",
    "_serialize_prompt_overhead_component",
    "_estimate_prompt_overhead_tokens",
    "_extract_prompt_text_blocks",
    "_classify_system_prompt_block",
    "_estimate_system_prompt_bucket_tokens",
    "_append_prompt_component",
    "_RESPONSES_SYSTEM_ROLES",
    "_RESPONSES_CONVERSATION_ROLES",
    "_RESPONSES_TEXT_CONTENT_TYPES",
    "_RESPONSES_OPAQUE_CONTENT_TYPES",
    "_RESPONSES_OPAQUE_ITEM_TYPES",
    "_append_prompt_text_components",
    "_extract_responses_visible_text_blocks",
    "_responses_message_component_path",
    "_record_responses_excluded_fields",
    "_append_openai_responses_input_component",
    "_append_openai_responses_input_components",
    "_split_chat_prompt_messages",
    "_extract_prompt_overhead_components",
    "_build_prompt_overhead_breakdown",
    "_estimate_rerank_request_tokens",
    "_usage_has_positive_tokens",
    "_merge_estimated_rerank_tokens_into_usage",
    "_positive_int_or_none",
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
