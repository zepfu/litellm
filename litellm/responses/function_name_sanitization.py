"""Request-local function-name rewriting for OpenAI Responses payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

ALGORITHM_VERSION = "responses-function-name-v1"
DEFAULT_MAX_FUNCTION_NAME_LENGTH = 64
_DIGEST_LENGTH = 16
_PRIMARY_ATTEMPTS = 32
_FALLBACK_ATTEMPTS = 32


@dataclass(frozen=True)
class ResponsesFunctionNameDiagnostics:
    """Bounded diagnostics that never contain function names or mappings."""

    algorithm_version: str
    max_length: int
    distinct_rewritten_count: int
    rewritten_occurrence_count: int
    affected_surfaces: tuple[str, ...]
    collision_fallback_used: bool


@dataclass(frozen=True)
class ResponsesFunctionNameRewrite:
    """Immutable request-local rewrite state and sanitized body."""

    body: Any
    original_to_upstream: Mapping[str, str]
    upstream_to_original: Mapping[str, str]
    diagnostics: ResponsesFunctionNameDiagnostics

    @property
    def changed(self) -> bool:
        return bool(self.original_to_upstream)

    def restore_name(self, name: Any) -> Any:
        if not isinstance(name, str):
            return name
        return self.upstream_to_original.get(name, name)


def _empty_rewrite(body: Any, *, max_length: int) -> ResponsesFunctionNameRewrite:
    return ResponsesFunctionNameRewrite(
        body=body,
        original_to_upstream=MappingProxyType({}),
        upstream_to_original=MappingProxyType({}),
        diagnostics=ResponsesFunctionNameDiagnostics(
            algorithm_version=ALGORITHM_VERSION,
            max_length=max_length,
            distinct_rewritten_count=0,
            rewritten_occurrence_count=0,
            affected_surfaces=(),
            collision_fallback_used=False,
        ),
    )


def _validate_max_length(max_length: int) -> None:
    minimum = len("__") + _DIGEST_LENGTH + 1
    if max_length < minimum:
        raise ValueError(
            f"max_length must be at least {minimum} for deterministic rewriting"
        )


def _build_sanitized_name_candidate(
    original: str,
    max_length: int,
    *,
    nonce: int = 0,
) -> str:
    """Build a stable readable candidate no longer than ``max_length``."""
    digest_source = original if nonce == 0 else f"{original}\0{nonce}"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    prefix_length = max_length - len("__") - len(digest)
    return f"{original[:prefix_length]}__{digest}"


def _build_collision_fallback_candidate(
    original: str,
    *,
    ordinal: int,
    max_length: int,
    nonce: int,
) -> str:
    digest = hashlib.sha256(
        f"{ordinal}\0{original}\0{nonce}".encode("utf-8")
    ).hexdigest()[:_DIGEST_LENGTH]
    prefix = f"litellm_fn_{ordinal}_"
    available = max_length - len(prefix)
    if available <= 0:
        return digest[:max_length]
    return f"{prefix}{digest[:available]}"


def _collect_function_names(
    body: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str]]:
    surfaces: dict[str, list[str]] = {
        "input": [],
        "tools": [],
        "tool_choice": [],
    }

    input_items = body.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            name = item.get("name")
            if isinstance(name, str):
                surfaces["input"].append(name)

    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            name = tool.get("name")
            if isinstance(name, str):
                surfaces["tools"].append(name)

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        if isinstance(name, str):
            surfaces["tool_choice"].append(name)

    all_names = sorted({name for names in surfaces.values() for name in names})
    return surfaces, all_names


def _allocate_name_mapping(
    names: list[str],
    *,
    max_length: int,
) -> tuple[dict[str, str], bool]:
    short_names = {name for name in names if len(name) <= max_length}
    long_names = [name for name in names if len(name) > max_length]
    used_names = set(short_names)
    mapping: dict[str, str] = {}
    collision_fallback_used = False

    for ordinal, original in enumerate(long_names):
        selected: str | None = None
        for nonce in range(_PRIMARY_ATTEMPTS):
            candidate = _build_sanitized_name_candidate(
                original,
                max_length,
                nonce=nonce,
            )
            if candidate not in used_names:
                selected = candidate
                collision_fallback_used = collision_fallback_used or nonce > 0
                break
            collision_fallback_used = True

        if selected is None:
            for nonce in range(_FALLBACK_ATTEMPTS):
                candidate = _build_collision_fallback_candidate(
                    original,
                    ordinal=ordinal,
                    max_length=max_length,
                    nonce=nonce,
                )
                if candidate not in used_names:
                    selected = candidate
                    collision_fallback_used = True
                    break

        if selected is None:
            raise ValueError(
                "Unable to allocate a unique Responses function name within "
                f"{max_length} characters"
            )

        mapping[original] = selected
        used_names.add(selected)

    return mapping, collision_fallback_used


def _rewrite_named_items(
    items: Any,
    *,
    item_type: str,
    mapping: Mapping[str, str],
) -> tuple[Any, int]:
    if not isinstance(items, list):
        return items, 0

    updated_items: list[Any] | None = None
    rewritten_count = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != item_type:
            continue
        original = item.get("name")
        if not isinstance(original, str):
            continue
        upstream = mapping.get(original)
        if upstream is None:
            continue
        if updated_items is None:
            updated_items = list(items)
        updated_item = dict(item)
        updated_item["name"] = upstream
        updated_items[index] = updated_item
        rewritten_count += 1

    return (updated_items if updated_items is not None else items), rewritten_count


def sanitize_responses_function_names(
    body: Any,
    *,
    max_length: int = DEFAULT_MAX_FUNCTION_NAME_LENGTH,
) -> ResponsesFunctionNameRewrite:
    """Sanitize the three function-name surfaces using one request-local mapping."""
    _validate_max_length(max_length)
    if not isinstance(body, dict):
        return _empty_rewrite(body, max_length=max_length)

    surfaces, names = _collect_function_names(body)
    mapping, collision_fallback_used = _allocate_name_mapping(
        names,
        max_length=max_length,
    )
    if not mapping:
        return _empty_rewrite(body, max_length=max_length)

    updated_input, input_count = _rewrite_named_items(
        body.get("input"),
        item_type="function_call",
        mapping=mapping,
    )
    updated_tools, tool_count = _rewrite_named_items(
        body.get("tools"),
        item_type="function",
        mapping=mapping,
    )

    updated_tool_choice = body.get("tool_choice")
    tool_choice_count = 0
    if (
        isinstance(updated_tool_choice, dict)
        and updated_tool_choice.get("type") == "function"
        and isinstance(updated_tool_choice.get("name"), str)
    ):
        original_choice = updated_tool_choice["name"]
        upstream_choice = mapping.get(original_choice)
        if upstream_choice is not None:
            updated_tool_choice = dict(updated_tool_choice)
            updated_tool_choice["name"] = upstream_choice
            tool_choice_count = 1

    updated_body = dict(body)
    if input_count:
        updated_body["input"] = updated_input
    if tool_count:
        updated_body["tools"] = updated_tools
    if tool_choice_count:
        updated_body["tool_choice"] = updated_tool_choice

    occurrence_count = input_count + tool_count + tool_choice_count
    affected_surfaces = tuple(
        surface
        for surface in ("input", "tools", "tool_choice")
        if any(name in mapping for name in surfaces[surface])
    )
    upstream_to_original = {
        upstream: original for original, upstream in mapping.items()
    }
    return ResponsesFunctionNameRewrite(
        body=updated_body,
        original_to_upstream=MappingProxyType(dict(mapping)),
        upstream_to_original=MappingProxyType(upstream_to_original),
        diagnostics=ResponsesFunctionNameDiagnostics(
            algorithm_version=ALGORITHM_VERSION,
            max_length=max_length,
            distinct_rewritten_count=len(mapping),
            rewritten_occurrence_count=occurrence_count,
            affected_surfaces=affected_surfaces,
            collision_fallback_used=collision_fallback_used,
        ),
    )


def _restore_function_call_names(
    value: Any,
    upstream_to_original: Mapping[str, str],
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        updated: dict[str, Any] | None = None

        if value.get("type") == "function_call":
            name = value.get("name")
            if isinstance(name, str) and name in upstream_to_original:
                updated = dict(value)
                updated["name"] = upstream_to_original[name]
                changed = True

        source = updated if updated is not None else value
        for key, child in source.items():
            if key == "name" and changed:
                continue
            restored_child, child_changed = _restore_function_call_names(
                child,
                upstream_to_original,
            )
            if not child_changed:
                continue
            if updated is None:
                updated = dict(value)
            updated[key] = restored_child
            changed = True

        return (updated if updated is not None else value), changed

    if isinstance(value, list):
        updated_items: list[Any] | None = None
        for index, item in enumerate(value):
            restored_item, item_changed = _restore_function_call_names(
                item,
                upstream_to_original,
            )
            if not item_changed:
                continue
            if updated_items is None:
                updated_items = list(value)
            updated_items[index] = restored_item
        return (updated_items if updated_items is not None else value), (
            updated_items is not None
        )

    return value, False


def restore_function_names_in_responses_body(
    body: Any,
    upstream_to_original: Mapping[str, str],
) -> Any:
    """Restore exact function-call names in a Responses body or terminal event."""
    if not upstream_to_original:
        return body
    restored, _ = _restore_function_call_names(body, upstream_to_original)
    return restored


def restore_function_names_in_responses_output(
    output: Any,
    upstream_to_original: Mapping[str, str],
) -> Any:
    """Restore exact function-call names in a Responses output list."""
    if not upstream_to_original:
        return output
    restored, _ = _restore_function_call_names(output, upstream_to_original)
    return restored
