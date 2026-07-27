"""Google Code Assist request, tool replay, cache, and stream shaping.

This extraction does not own process state. Tool-call cache mappings remain
owned by ``providers.google.process_cache`` and are supplied here through an
explicit runtime configuration.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

import litellm
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    anthropic_replay as _anthropic_replay,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    error_and_schema as _error_and_schema,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache as _process_cache,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    request_assembly as _request_assembly,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    request_building as _request_building,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    response_streaming as _response_streaming,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    response_translation as _response_translation,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    tool_aliasing as _tool_aliasing,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    tool_pairing as _tool_pairing,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload
from litellm.types.llms.openai import (
    ResponsesAPIOptionalRequestParams,
)

_GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH = 64
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_TTL_SECONDS = 6 * 60 * 60
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE = 2048


@dataclass(frozen=True)
class Runtime:
    """Live host callbacks plus canonical process-cache mappings."""

    host_globals: Mapping[str, object]
    tool_call_name_cache: MutableMapping[str, tuple[str, float]]
    tool_call_arguments_cache: MutableMapping[str, tuple[str, float]]
    monotonic: Callable[[], float] = time.monotonic
    cache_ttl_seconds: float = _CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_TTL_SECONDS
    cache_max_size: int = _CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE


_RUNTIME = Runtime(
    host_globals={},
    tool_call_name_cache=_process_cache._codex_google_code_assist_tool_call_name_cache,
    tool_call_arguments_cache=_process_cache._codex_google_code_assist_tool_call_arguments_cache,
)


def configure(
    *,
    host_globals: Mapping[str, object],
    monotonic: Callable[[], float] = time.monotonic,
    cache_ttl_seconds: float = _CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_TTL_SECONDS,
    cache_max_size: int = _CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE,
    tool_call_name_cache: Optional[MutableMapping[str, tuple[str, float]]] = None,
    tool_call_arguments_cache: Optional[MutableMapping[str, tuple[str, float]]] = None,
) -> None:
    """Configure live collaborators without creating another cache owner."""

    global _RUNTIME
    _RUNTIME = Runtime(
        host_globals=host_globals,
        tool_call_name_cache=(
            _process_cache._codex_google_code_assist_tool_call_name_cache
            if tool_call_name_cache is None
            else tool_call_name_cache
        ),
        tool_call_arguments_cache=(
            _process_cache._codex_google_code_assist_tool_call_arguments_cache
            if tool_call_arguments_cache is None
            else tool_call_arguments_cache
        ),
        monotonic=monotonic,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
    )


def _missing_dependency(name: str) -> Callable[..., Any]:
    def missing(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError(f"Google Code Assist runtime dependency is not configured: {name}")

    return missing


class _RuntimeNamespace(Mapping[str, object]):
    """Resolve host monkeypatches first, then module-owned implementations."""

    def __getitem__(self, key: str) -> object:
        if key in _RUNTIME.host_globals:
            return _RUNTIME.host_globals[key]
        if key in globals():
            return globals()[key]
        return _missing_dependency(key)

    def __iter__(self) -> Iterator[str]:
        return iter(set(globals()) | set(_RUNTIME.host_globals))

    def __len__(self) -> int:
        return len(set(globals()) | set(_RUNTIME.host_globals))


def _bind_provider(module: Any) -> None:
    module.bind_runtime(_RuntimeNamespace())


def install(host_globals: dict[str, object]) -> None:
    """Publish extraction facades into the host module for later integration."""

    configure(host_globals=host_globals)
    for name in _OWNED_FUNCTION_NAMES:
        host_globals[name] = globals()[name]


def _merge_google_code_assist_schema_annotations(
    source: dict[str, Any],
    target: dict[str, Any],
) -> None:
    for key in ("description", "title", "default"):
        if key in source and key not in target:
            target[key] = copy.deepcopy(source[key])


def _simplify_google_code_assist_union_schema(  # noqa: PLR0915
    schema_node: dict[str, Any],
) -> int:
    fix_count = 0
    for union_key in ("anyOf", "oneOf", "allOf"):
        variants = schema_node.get(union_key)
        if not isinstance(variants, list):
            continue
        dict_variants = [variant for variant in variants if isinstance(variant, dict)]
        if not dict_variants:
            schema_node.pop(union_key, None)
            fix_count += 1
            continue

        nullable = any(variant.get("type") == "null" for variant in dict_variants)
        non_null_variants = [variant for variant in dict_variants if variant.get("type") != "null"]
        if len(non_null_variants) == 1:
            replacement = copy.deepcopy(non_null_variants[0])
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        string_variant = next(
            (variant for variant in non_null_variants if variant.get("type") == "string"),
            None,
        )
        if string_variant is not None:
            replacement = {
                key: copy.deepcopy(value)
                for key, value in string_variant.items()
                if key in {"type", "description", "title", "enum", "default"}
            }
            replacement.setdefault("type", "string")
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        object_variants = [
            variant
            for variant in non_null_variants
            if variant.get("type") == "object" and isinstance(variant.get("properties"), dict)
        ]
        if object_variants:
            merged_properties: dict[str, Any] = {}
            for variant in object_variants:
                merged_properties.update(copy.deepcopy(variant.get("properties") or {}))
            replacement = {"type": "object", "properties": merged_properties}
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        typed_variant = next(
            (variant for variant in non_null_variants if isinstance(variant.get("type"), str)),
            None,
        )
        if typed_variant is not None:
            replacement = copy.deepcopy(typed_variant)
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        schema_node.pop(union_key, None)
        schema_node.setdefault("type", "object")
        schema_node.setdefault("properties", {})
        fix_count += 1
    return fix_count


def _sanitize_google_code_assist_union_schemas(
    schema_node: Any,
    *,
    _depth: int = 0,
    _seen: Optional[set[int]] = None,
) -> int:
    _bind_provider(_error_and_schema)
    return _error_and_schema._sanitize_google_code_assist_union_schemas(
        schema_node,
        _depth=_depth,
        _seen=_seen,
    )


def _sanitize_google_code_assist_tool_schema(schema_node: Any) -> int:
    _bind_provider(_error_and_schema)
    return _error_and_schema._sanitize_google_code_assist_tool_schema(schema_node)


def _is_anthropic_tool_use_content_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_use"


def _is_anthropic_tool_result_content_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    block_type = block.get("type")
    return block_type == "tool_result" or (
        isinstance(block_type, str) and block_type.endswith("_tool_result")
    )


def _codex_google_code_assist_tool_result_content_to_openai_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _codex_google_code_assist_anthropic_tool_use_to_openai_tool_call(
    *,
    block: dict[str, Any],
    message_index: int,
    content_index: int,
) -> dict[str, Any]:
    tool_use_id = block.get("id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid Anthropic tool_use block at "
                f"messages.{message_index}.content.{content_index}: "
                "missing required non-empty string tool_use.id"
            ),
        )
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    return {
        "id": tool_use_id.strip(),
        "type": "function",
        "function": {
            "name": str(block.get("name") or ""),
            "arguments": json.dumps(tool_input, ensure_ascii=False),
        },
    }


def _codex_google_code_assist_anthropic_tool_result_to_openai_tool_message(
    *,
    block: dict[str, Any],
    message_index: int,
    content_index: int,
) -> dict[str, Any]:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid Anthropic tool_result block at "
                f"messages.{message_index}.content.{content_index}: "
                "missing required non-empty string tool_result.tool_use_id"
            ),
        )
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_use_id.strip(),
        "content": _codex_google_code_assist_tool_result_content_to_openai_content(
            block.get("content")
        ),
    }
    cache_control = block.get("cache_control")
    if cache_control is not None:
        tool_message["cache_control"] = cache_control
    return tool_message


def _normalize_codex_google_code_assist_anthropic_tool_replay(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _bind_provider(_anthropic_replay)
    return _anthropic_replay._normalize_codex_google_code_assist_anthropic_tool_replay(
        completion_kwargs
    )


def _repair_codex_google_code_assist_openai_tool_call_ids(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _bind_provider(_anthropic_replay)
    return _anthropic_replay._repair_codex_google_code_assist_openai_tool_call_ids(
        completion_kwargs
    )


def _codex_google_code_assist_tool_call_cache_key(
    tool_call_id: str,
    *,
    scope_key: Optional[str] = None,
) -> str:
    clean_value = _RUNTIME.host_globals.get("_clean_codex_auth_value")
    cleaned_scope = (
        clean_value(scope_key)
        if callable(clean_value)
        else scope_key.strip() if isinstance(scope_key, str) and scope_key.strip() else None
    )
    return f"{cleaned_scope}:{tool_call_id}" if cleaned_scope else tool_call_id


def _resolve_codex_google_code_assist_tool_call_scope_key(
    *,
    request: Optional[Request] = None,
    request_body: Optional[Payload] = None,
    explicit_scope_key: Optional[str] = None,
) -> Optional[str]:
    clean_value = _RUNTIME.host_globals.get("_clean_codex_auth_value")

    def clean(value: Any) -> Optional[str]:
        if callable(clean_value):
            return clean_value(value)
        return value.strip() if isinstance(value, str) and value.strip() else None

    explicit = clean(explicit_scope_key)
    if explicit is not None:
        return explicit
    body = request_body if isinstance(request_body, dict) else {}
    for source in (body.get("litellm_metadata"), body.get("metadata"), body):
        if not isinstance(source, dict):
            continue
        for key in (
            "session_id",
            "session-id",
            "conversation_id",
            "thread_id",
            "litellm_call_id",
        ):
            value = clean(source.get(key))
            if value is not None:
                return value
    if request is not None:
        get_headers = _RUNTIME.host_globals.get("_safe_get_request_headers")
        headers = get_headers(request) if callable(get_headers) else dict(request.headers)
        if isinstance(headers, dict):
            for key in (
                "session_id",
                "session-id",
                "x-session-id",
                "x-litellm-session-id",
                "x-request-id",
            ):
                value = clean(headers.get(key))
                if value is not None:
                    return value
    return None


def _resolve_cache_max_size() -> int:
    """Resolve cache max size from the live host namespace at call time.

    Consults the host module's globals so that late monkeypatches (for
    example a test patching
    ``lpe._CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE`` after
    ``install()``) are honored without reconfiguration. Falls back to the
    configured/default value when the host namespace does not provide a
    positive int.
    """
    host_value = _RUNTIME.host_globals.get(
        "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE"
    )
    if isinstance(host_value, int) and not isinstance(host_value, bool) and host_value > 0:
        return host_value
    return _RUNTIME.cache_max_size


def _prune_codex_google_code_assist_tool_call_caches(
    now: Optional[float] = None,
) -> None:
    current = _RUNTIME.monotonic() if now is None else now
    expired_keys = [
        key
        for key, entry in _RUNTIME.tool_call_name_cache.items()
        if not isinstance(entry, tuple) or len(entry) < 2 or float(entry[1]) <= current
    ]
    for key in expired_keys:
        _RUNTIME.tool_call_name_cache.pop(key, None)
        _RUNTIME.tool_call_arguments_cache.pop(key, None)
    while len(_RUNTIME.tool_call_name_cache) > _resolve_cache_max_size():
        try:
            oldest_key = next(iter(_RUNTIME.tool_call_name_cache))
        except StopIteration:
            break
        _RUNTIME.tool_call_name_cache.pop(oldest_key, None)
        _RUNTIME.tool_call_arguments_cache.pop(oldest_key, None)


def _normalize_codex_google_code_assist_tool_call_arguments(
    function_arguments: Any,
) -> Optional[str]:
    _bind_provider(_tool_pairing)
    return _tool_pairing._normalize_codex_google_code_assist_tool_call_arguments(
        function_arguments
    )


def _remember_codex_google_code_assist_tool_call_name(
    tool_call_id: Any,
    function_name: Any,
    function_arguments: Any = None,
    *,
    scope_key: Optional[str] = None,
) -> None:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return
    cache_key = _codex_google_code_assist_tool_call_cache_key(
        tool_call_id,
        scope_key=scope_key,
    )
    now = _RUNTIME.monotonic()
    _prune_codex_google_code_assist_tool_call_caches(now)
    expires_at = now + float(_RUNTIME.cache_ttl_seconds)
    if not isinstance(function_name, str) or not function_name:
        cached = _RUNTIME.tool_call_name_cache.get(cache_key)
        if cached is None and "__thought__" in tool_call_id:
            base_key = _codex_google_code_assist_tool_call_cache_key(
                tool_call_id.split("__thought__", 1)[0],
                scope_key=scope_key,
            )
            cached = _RUNTIME.tool_call_name_cache.get(base_key)
        function_name = cached[0] if isinstance(cached, tuple) and cached else None
        if not isinstance(function_name, str) or not function_name:
            return
    _RUNTIME.tool_call_name_cache.pop(cache_key, None)
    _RUNTIME.tool_call_name_cache[cache_key] = (function_name, expires_at)
    _prune_codex_google_code_assist_tool_call_caches(now)

    normalized_arguments = _normalize_codex_google_code_assist_tool_call_arguments(
        function_arguments
    )
    if normalized_arguments is None:
        return
    existing_entry = _RUNTIME.tool_call_arguments_cache.get(cache_key)
    existing_arguments = (
        existing_entry[0] if isinstance(existing_entry, tuple) and existing_entry else ""
    )
    if not existing_arguments:
        merged = normalized_arguments
    elif normalized_arguments.startswith(existing_arguments):
        merged = normalized_arguments
    elif not existing_arguments.endswith(normalized_arguments):
        merged = f"{existing_arguments}{normalized_arguments}"
    else:
        merged = existing_arguments
    _RUNTIME.tool_call_arguments_cache.pop(cache_key, None)
    _RUNTIME.tool_call_arguments_cache[cache_key] = (merged, expires_at)


def _lookup_codex_google_code_assist_tool_call_name(
    tool_call_id: Any,
    *,
    scope_key: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    now = _RUNTIME.monotonic()
    _prune_codex_google_code_assist_tool_call_caches(now)
    cache_key = _codex_google_code_assist_tool_call_cache_key(
        tool_call_id,
        scope_key=scope_key,
    )
    cached = _RUNTIME.tool_call_name_cache.get(cache_key)
    if isinstance(cached, tuple) and cached and float(cached[1]) > now:
        return cached[0]
    if "__thought__" in tool_call_id:
        base_key = _codex_google_code_assist_tool_call_cache_key(
            tool_call_id.split("__thought__", 1)[0],
            scope_key=scope_key,
        )
        cached = _RUNTIME.tool_call_name_cache.get(base_key)
        if isinstance(cached, tuple) and cached and float(cached[1]) > now:
            return cached[0]
    return None


def _lookup_codex_google_code_assist_tool_call_arguments(
    tool_call_id: Any,
    *,
    scope_key: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    now = _RUNTIME.monotonic()
    _prune_codex_google_code_assist_tool_call_caches(now)
    cache_key = _codex_google_code_assist_tool_call_cache_key(
        tool_call_id,
        scope_key=scope_key,
    )
    cached = _RUNTIME.tool_call_arguments_cache.get(cache_key)
    if isinstance(cached, tuple) and cached and float(cached[1]) > now:
        return cached[0]
    if "__thought__" in tool_call_id:
        base_key = _codex_google_code_assist_tool_call_cache_key(
            tool_call_id.split("__thought__", 1)[0],
            scope_key=scope_key,
        )
        cached = _RUNTIME.tool_call_arguments_cache.get(base_key)
        if isinstance(cached, tuple) and cached and float(cached[1]) > now:
            return cached[0]
    return None


def _codex_google_code_assist_tool_call_function_name(
    tool_call: Optional[dict[str, Any]],
) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _codex_google_code_assist_tool_call_function_arguments(
    tool_call: Optional[dict[str, Any]],
) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    return _normalize_codex_google_code_assist_tool_call_arguments(
        function.get("arguments")
    )


def _codex_google_code_assist_tool_result_message_content(
    message: dict[str, Any],
) -> str:
    content = _codex_google_code_assist_tool_result_content_to_openai_content(
        message.get("content")
    )
    return str(content or "").strip()


def _codex_google_code_assist_orphan_tool_result_context_text(
    *,
    tool_call_id: str,
    content: str,
) -> str:
    normalized_content = content.strip()
    if not normalized_content:
        return (
            "Previous tool result context (unmapped tool call "
            f"{tool_call_id}): no output was recorded."
        )
    return (
        "Previous tool result context (unmapped tool call "
        f"{tool_call_id}):\n{normalized_content}"
    )


def _codex_google_code_assist_display_tool_call_id(tool_call_id: str) -> str:
    return tool_call_id.split("__thought__", 1)[0]


def _ensure_codex_google_code_assist_tool_results_have_calls(
    completion_kwargs: dict[str, Any],
    *,
    scope_key: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _bind_provider(_tool_pairing)
    return _tool_pairing._ensure_codex_google_code_assist_tool_results_have_calls(
        completion_kwargs,
        scope_key=scope_key,
    )


async def _build_google_code_assist_request_from_completion_kwargs(
    *,
    completion_kwargs: dict[str, Any],
    adapter_model: str,
    project: str,
    request: Request,
    completion_kwargs_are_openai_chat: bool = False,
    scope_key: Optional[str] = None,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    _bind_provider(_request_assembly)
    return await _request_assembly._build_google_code_assist_request_from_completion_kwargs(
        completion_kwargs=completion_kwargs,
        adapter_model=adapter_model,
        project=project,
        request=request,
        completion_kwargs_are_openai_chat=completion_kwargs_are_openai_chat,
        scope_key=scope_key,
    )


def _drop_codex_google_code_assist_non_function_tools(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    _bind_provider(_request_building)
    return _request_building._drop_codex_google_code_assist_non_function_tools(
        request_body
    )


def _build_codex_google_code_assist_completion_kwargs(
    prepared_request_body: dict[str, Any],
    *,
    adapter_model: str,
) -> tuple[dict[str, Any], Any, ResponsesAPIOptionalRequestParams]:
    _bind_provider(_request_building)
    return _request_building._build_codex_google_code_assist_completion_kwargs(
        prepared_request_body,
        adapter_model=adapter_model,
    )


async def _prepare_codex_google_code_assist_adapter_request(
    *,
    request: Request,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> SimpleNamespace:
    _bind_provider(_request_building)
    return await _request_building._prepare_codex_google_code_assist_adapter_request(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        adapter_provider=adapter_provider,
    )


def _normalize_google_code_assist_httpx_payload(value: Any) -> Any:
    _bind_provider(_tool_aliasing)
    return _tool_aliasing._normalize_google_code_assist_httpx_payload(value)


def _annotate_google_code_assist_duplicate_tool_response_parts(
    contents: list[Any],
    duplicate_tool_results: list[tuple[str, str]],
    *,
    annotate_function_response_id: bool = False,
) -> int:
    return _tool_aliasing._annotate_google_code_assist_duplicate_tool_response_parts(
        contents,
        duplicate_tool_results,
        annotate_function_response_id=annotate_function_response_id,
    )


def _annotate_google_code_assist_duplicate_tool_responses(
    google_request_dict: dict[str, Any],
    completion_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    _bind_provider(_tool_aliasing)
    return _tool_aliasing._annotate_google_code_assist_duplicate_tool_responses(
        google_request_dict,
        completion_messages,
    )


def _annotate_google_code_assist_claude_tool_response_ids(
    google_request_dict: dict[str, Any],
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
) -> dict[str, Any]:
    _bind_provider(_tool_aliasing)
    return _tool_aliasing._annotate_google_code_assist_claude_tool_response_ids(
        google_request_dict,
        completion_messages,
        google_model=google_model,
    )


def _google_code_assist_function_response_id(
    function_response: dict[str, Any],
) -> Optional[str]:
    response_payload = function_response.get("response")
    response_tool_use_id = (
        response_payload.get("tool_use_id")
        if isinstance(response_payload, dict)
        else None
    )
    for candidate in (function_response.get("id"), response_tool_use_id):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _google_code_assist_function_call_args_for_id(
    tool_call_id: str,
    *,
    scope_key: Optional[str] = None,
) -> dict[str, Any]:
    cached_arguments = _lookup_codex_google_code_assist_tool_call_arguments(
        tool_call_id,
        scope_key=scope_key,
    )
    if not isinstance(cached_arguments, str) or not cached_arguments.strip():
        return {}
    try:
        parsed_arguments = json.loads(cached_arguments)
    except Exception:
        return {}
    return parsed_arguments if isinstance(parsed_arguments, dict) else {}


def _insert_google_code_assist_missing_claude_function_call_pairs(
    google_request_dict: dict[str, Any],
    *,
    google_model: str,
    scope_key: Optional[str] = None,
) -> dict[str, Any]:
    _bind_provider(_tool_aliasing)
    return _tool_aliasing._insert_google_code_assist_missing_claude_function_call_pairs(
        google_request_dict,
        google_model=google_model,
        scope_key=scope_key,
    )


def _unwrap_google_code_assist_response_payload(
    payload: Any,
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        return None
    unwrapped = dict(response_payload)
    trace_id = payload.get("traceId")
    if isinstance(trace_id, str) and trace_id and "responseId" not in unwrapped:
        unwrapped["responseId"] = trace_id
    return unwrapped


async def _translate_google_code_assist_response_to_anthropic(
    *,
    response: Response,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    completion_messages: list[dict[str, Any]],
    gemini_optional_params: dict[str, Any],
    litellm_params: dict[str, Any],
    logging_obj: Any,
) -> Response:
    _bind_provider(_response_translation)
    return await _response_translation._translate_google_code_assist_response_to_anthropic(
        response=response,
        adapter_model=adapter_model,
        tool_name_mapping=tool_name_mapping,
        completion_messages=completion_messages,
        gemini_optional_params=gemini_optional_params,
        litellm_params=litellm_params,
        logging_obj=logging_obj,
    )


def _iterate_google_code_assist_unwrapped_stream(
    body_iterator: Any, *, adapter_model: Optional[str] = None, rate_limit_key: Optional[str] = None
) -> Any:
    _bind_provider(_response_streaming)
    return _response_streaming._iterate_google_code_assist_unwrapped_stream(
        body_iterator,
        adapter_model=adapter_model,
        rate_limit_key=rate_limit_key,
    )


def _build_anthropic_streaming_response_from_google_code_assist_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    gemini_optional_params: dict[str, Any],
    rate_limit_key: Optional[str] = None,
) -> StreamingResponse:
    _bind_provider(_response_streaming)
    return _response_streaming._build_anthropic_streaming_response_from_google_code_assist_stream(
        response=response,
        adapter_model=adapter_model,
        tool_name_mapping=tool_name_mapping,
        gemini_optional_params=gemini_optional_params,
        rate_limit_key=rate_limit_key,
    )


def _restore_google_adapter_tool_call_names(
    response_obj: Any,
    tool_name_mapping: dict[str, str],
    *,
    scope_key: Optional[str] = None,
) -> Any:
    _bind_provider(_response_streaming)
    return _response_streaming._restore_google_adapter_tool_call_names(
        response_obj,
        tool_name_mapping,
        scope_key=scope_key,
    )


async def _restore_google_adapter_tool_call_names_stream(
    completion_stream: Any,
    tool_name_mapping: dict[str, str],
    *,
    scope_key: Optional[str] = None,
) -> Any:
    _bind_provider(_response_streaming)
    async for chunk in _response_streaming._restore_google_adapter_tool_call_names_stream(
        completion_stream,
        tool_name_mapping,
        scope_key=scope_key,
    ):
        yield chunk


async def _collect_google_code_assist_model_response_from_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    logging_obj: Any,
) -> Any:
    _bind_provider(_response_streaming)
    return await _response_streaming._collect_google_code_assist_model_response_from_stream(
        response=response,
        adapter_model=adapter_model,
        logging_obj=logging_obj,
    )


async def _collect_google_code_assist_response_from_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    logging_obj: Any,
) -> Response:
    _bind_provider(_response_streaming)
    return await _response_streaming._collect_google_code_assist_response_from_stream(
        response=response,
        adapter_model=adapter_model,
        tool_name_mapping=tool_name_mapping,
        logging_obj=logging_obj,
    )


def _build_codex_streaming_response_from_google_code_assist_stream(
    *,
    response: StreamingResponse,
    adapter_request: SimpleNamespace,
) -> StreamingResponse:
    _bind_provider(_response_streaming)
    return _response_streaming._build_codex_streaming_response_from_google_code_assist_stream(
        response=response,
        adapter_request=adapter_request,
    )


def _sanitize_google_code_assist_request_schemas(
    wrapped_request_body: Any,
) -> int:
    _bind_provider(_response_streaming)
    return _response_streaming._sanitize_google_code_assist_request_schemas(
        wrapped_request_body
    )


_OWNED_FUNCTION_NAMES = (
    "_merge_google_code_assist_schema_annotations",
    "_simplify_google_code_assist_union_schema",
    "_sanitize_google_code_assist_union_schemas",
    "_sanitize_google_code_assist_tool_schema",
    "_is_anthropic_tool_use_content_block",
    "_is_anthropic_tool_result_content_block",
    "_codex_google_code_assist_tool_result_content_to_openai_content",
    "_codex_google_code_assist_anthropic_tool_use_to_openai_tool_call",
    "_codex_google_code_assist_anthropic_tool_result_to_openai_tool_message",
    "_normalize_codex_google_code_assist_anthropic_tool_replay",
    "_repair_codex_google_code_assist_openai_tool_call_ids",
    "_codex_google_code_assist_tool_call_cache_key",
    "_resolve_codex_google_code_assist_tool_call_scope_key",
    "_prune_codex_google_code_assist_tool_call_caches",
    "_normalize_codex_google_code_assist_tool_call_arguments",
    "_remember_codex_google_code_assist_tool_call_name",
    "_lookup_codex_google_code_assist_tool_call_name",
    "_lookup_codex_google_code_assist_tool_call_arguments",
    "_codex_google_code_assist_tool_call_function_name",
    "_codex_google_code_assist_tool_call_function_arguments",
    "_codex_google_code_assist_tool_result_message_content",
    "_codex_google_code_assist_orphan_tool_result_context_text",
    "_codex_google_code_assist_display_tool_call_id",
    "_ensure_codex_google_code_assist_tool_results_have_calls",
    "_build_google_code_assist_request_from_completion_kwargs",
    "_drop_codex_google_code_assist_non_function_tools",
    "_build_codex_google_code_assist_completion_kwargs",
    "_prepare_codex_google_code_assist_adapter_request",
    "_normalize_google_code_assist_httpx_payload",
    "_annotate_google_code_assist_duplicate_tool_response_parts",
    "_annotate_google_code_assist_duplicate_tool_responses",
    "_annotate_google_code_assist_claude_tool_response_ids",
    "_google_code_assist_function_response_id",
    "_google_code_assist_function_call_args_for_id",
    "_insert_google_code_assist_missing_claude_function_call_pairs",
    "_unwrap_google_code_assist_response_payload",
    "_translate_google_code_assist_response_to_anthropic",
    "_iterate_google_code_assist_unwrapped_stream",
    "_build_anthropic_streaming_response_from_google_code_assist_stream",
    "_restore_google_adapter_tool_call_names",
    "_restore_google_adapter_tool_call_names_stream",
    "_collect_google_code_assist_model_response_from_stream",
    "_collect_google_code_assist_response_from_stream",
    "_build_codex_streaming_response_from_google_code_assist_stream",
    "_sanitize_google_code_assist_request_schemas",
)

__all__ = [
    "Runtime",
    "configure",
    "install",
    *_OWNED_FUNCTION_NAMES,
]
