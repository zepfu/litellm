"""Wave 6D extraction: Claude persisted-output transformations.

Behavior-preserving extraction from ``llm_passthrough_endpoints.py``.
Do not import ``llm_passthrough_endpoints`` at module scope.

The Google provider owns the compaction algorithms. This module owns the
request-policy delegates and Claude persisted-output file expansion.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    env_policy as _google_env_policy,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    persisted_output as _anthropic_google_shaping,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    lane_keys as _aawm_lane_keys,
)

_CLAUDE_PERSISTED_OUTPUT_PATTERN = (
    _aawm_lane_keys._CLAUDE_PERSISTED_OUTPUT_PATTERN
)
_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN = (
    _aawm_lane_keys._CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN
)
_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN = (
    _aawm_lane_keys._CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN
)
_CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN = (
    _aawm_lane_keys._CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN
)
_get_google_adapter_auxiliary_context_char_cap = (
    _google_env_policy._get_google_adapter_auxiliary_context_char_cap
)
_get_google_adapter_followup_auxiliary_context_char_cap = (
    _google_env_policy._get_google_adapter_followup_auxiliary_context_char_cap
)
_get_google_adapter_followup_persisted_output_char_cap = (
    _google_env_policy._get_google_adapter_followup_persisted_output_char_cap
)
_get_google_adapter_persisted_output_char_cap = (
    _google_env_policy._get_google_adapter_persisted_output_char_cap
)

_persisted_output_logging_callback: Optional[
    Callable[..., dict[str, Any]]
] = None

if TYPE_CHECKING:

    def _format_langfuse_span_timestamp(value: datetime) -> str: ...

_RUNTIME_DEPENDENCY_NAMES = (
    "_anthropic_google_shaping",
    "_CLAUDE_PERSISTED_OUTPUT_PATTERN",
    "_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN",
    "_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN",
    "_CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN",
    "_estimate_google_content_text_chars",
    "_get_google_adapter_auxiliary_context_char_cap",
    "_get_google_adapter_followup_auxiliary_context_char_cap",
    "_get_google_adapter_followup_persisted_output_char_cap",
    "_get_google_adapter_persisted_output_char_cap",
    "_format_langfuse_span_timestamp",
    "_persisted_output_logging_callback",
)

_HOST_FUNCTION_NAMES = (
    "_is_claude_persisted_output_expansion_enabled",
    "_get_claude_persisted_output_root",
    "_resolve_claude_persisted_output_path",
    "_build_claude_persisted_output_source_metadata",
    "_compact_google_adapter_persisted_output_preview_and_expanded_text",
    "_compact_expanded_claude_persisted_output_text_for_google_adapter",
    "_compact_google_adapter_text_part_sequence",
    "_compact_google_adapter_followup_request_contents",
    "_compact_google_adapter_persisted_output_value",
    "_compact_google_adapter_persisted_output_in_anthropic_request_body",
    "_expand_claude_persisted_output_text",
    "_expand_claude_persisted_output_value",
    "_expand_claude_persisted_output_in_anthropic_request_body",
    "_estimate_google_content_text_chars",
)


def bind_runtime(namespace: Mapping[str, object]) -> None:
    """Bind explicit host callbacks/configuration for direct module use."""
    module_globals = globals()
    for name in _RUNTIME_DEPENDENCY_NAMES:
        if name in namespace:
            module_globals[name] = namespace[name]


def install(host_globals: dict[str, Any]) -> None:
    """Publish same-object host facades with live monkeypatch reachability."""
    module_globals = globals()
    for name in _HOST_FUNCTION_NAMES:
        function = module_globals[name]
        rebound = FunctionType(
            function.__code__,
            host_globals,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        rebound.__kwdefaults__ = function.__kwdefaults__
        rebound.__annotations__ = function.__annotations__
        rebound.__doc__ = function.__doc__
        rebound.__module__ = function.__module__
        rebound.__qualname__ = function.__qualname__
        if function.__dict__:
            rebound.__dict__.update(function.__dict__)
        module_globals[name] = rebound
        host_globals[name] = rebound


def configure_persisted_output_logging_callback(
    callback: Optional[Callable[..., dict[str, Any]]],
) -> None:
    """Set the observability-owned logging callback for body-level expansion."""
    global _persisted_output_logging_callback
    _persisted_output_logging_callback = callback


def _estimate_google_content_text_chars(content_block: Any) -> int:
    """Estimate total text characters in a Google content block.

    Single behavior-compatible implementation used by both direct module
    callers and installed host facades, eliminating the prior dual-path
    divergence between content_selection and env_policy delegates.
    """
    if not isinstance(content_block, dict):
        return 0
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return 0
    total = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            total += len(text)
    return total


def _is_claude_persisted_output_expansion_enabled() -> bool:
    value = os.getenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _get_claude_persisted_output_root() -> Path:
    raw = os.getenv("LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT")
    if raw and str(raw).strip():
        return Path(str(raw).strip()).expanduser()
    return Path.home() / ".claude" / "projects"


def _resolve_claude_persisted_output_path(path_str: str) -> Optional[Path]:
    try:
        root = _get_claude_persisted_output_root().resolve(strict=True)
        candidate = Path(path_str).expanduser().resolve(strict=True)
    except Exception:
        return None

    if not candidate.is_file():
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if "tool-results" not in candidate.parts:
        return None
    if not candidate.name.endswith("-additionalContext.txt"):
        return None
    return candidate


def _build_claude_persisted_output_source_metadata(
    *,
    resolved_path: Path,
    file_text: str,
) -> dict[str, Any]:
    file_bytes = file_text.encode("utf-8")
    return {
        "path": str(resolved_path),
        "basename": resolved_path.name,
        "content_hash": hashlib.sha256(file_bytes).hexdigest(),
        "bytes": len(file_bytes),
    }


def _compact_google_adapter_persisted_output_preview_and_expanded_text(
    text: str,
    *,
    cap: int,
) -> tuple[str, int, set[str], list[dict[str, Any]]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_persisted_output_preview_and_expanded_text(
        text,
        cap=cap,
    )


def _compact_expanded_claude_persisted_output_text_for_google_adapter(
    text: str,
    *,
    persisted_output_char_cap: Optional[int] = None,
    auxiliary_context_char_cap: Optional[int] = None,
) -> Tuple[str, int, set[str], list[dict[str, Any]]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_expanded_claude_persisted_output_text_for_google_adapter(
        text,
        persisted_output_char_cap=persisted_output_char_cap,
        auxiliary_context_char_cap=auxiliary_context_char_cap,
    )


def _compact_google_adapter_text_part_sequence(
    parts: list[Any],
) -> Tuple[list[Any], int, set[str], list[dict[str, Any]], bool]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_text_part_sequence(
        parts
    )


def _compact_google_adapter_followup_request_contents(
    request_block: dict[str, Any],
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_followup_request_contents(
        request_block
    )


def _compact_google_adapter_persisted_output_value(
    value: Any,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_persisted_output_value(
        value
    )


def _compact_google_adapter_persisted_output_in_anthropic_request_body(
    request_body: dict[str, Any],
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_persisted_output_in_anthropic_request_body(
        request_body
    )


def _expand_claude_persisted_output_text(
    text: str,
) -> Tuple[str, bool, Optional[str], Optional[dict[str, Any]]]:
    if not _is_claude_persisted_output_expansion_enabled():
        return text, False, None, None

    match = _CLAUDE_PERSISTED_OUTPUT_PATTERN.match(text)
    if match is None:
        return text, False, None, None

    resolved_path = _resolve_claude_persisted_output_path(match.group("path"))
    if resolved_path is None:
        return text, False, None, None

    try:
        file_text = resolved_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).rstrip("\n")
    except Exception:
        return text, False, None, None

    hook = match.group("hook")
    expanded = (
        "<system-reminder>\n"
        f"{hook} hook additional context: <persisted-output>\n"
        f"{file_text}\n"
        "</persisted-output>\n"
        "</system-reminder>\n"
    )
    return (
        expanded,
        True,
        hook.lower(),
        _build_claude_persisted_output_source_metadata(
            resolved_path=resolved_path,
            file_text=file_text,
        ),
    )


def _expand_claude_persisted_output_value(
    value: Any,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            (
                expanded_text,
                was_expanded,
                hook_name,
                source_metadata,
            ) = _expand_claude_persisted_output_text(value["text"])
            if was_expanded:
                updated_value = dict(value)
                updated_value["text"] = expanded_text
                return (
                    updated_value,
                    1,
                    {hook_name} if hook_name else set(),
                    [source_metadata] if source_metadata else [],
                )
            return value, 0, set(), []

        updated_dict: dict[str, Any] = {}
        expanded_count = 0
        hooks: set[str] = set()
        source_metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_expanded_count,
                child_hooks,
                child_source_metadata_items,
            ) = _expand_claude_persisted_output_value(child)
            updated_dict[key] = updated_child
            expanded_count += child_expanded_count
            hooks.update(child_hooks)
            source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                changed = True
        return (
            updated_dict if changed else value,
            expanded_count,
            hooks,
            source_metadata_items,
        )

    if isinstance(value, list):
        updated_list = []
        expanded_count = 0
        list_hooks: set[str] = set()
        list_source_metadata_items: list[dict[str, Any]] = []
        changed = False
        for child in value:
            (
                updated_child,
                child_expanded_count,
                child_hooks,
                child_source_metadata_items,
            ) = _expand_claude_persisted_output_value(child)
            updated_list.append(updated_child)
            expanded_count += child_expanded_count
            list_hooks.update(child_hooks)
            list_source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                changed = True
        return (
            updated_list if changed else value,
            expanded_count,
            list_hooks,
            list_source_metadata_items,
        )

    return value, 0, set(), []


def _expand_claude_persisted_output_in_anthropic_request_body(
    request_body: dict[str, Any],
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    """God-compatible body-level persisted-output expansion orchestration.

    Expands all persisted-output markers in the request body, then delegates
    to the explicitly configured logging callback (owned by
    observability_metadata) for metadata/tag/span attachment and timing.
    """
    span_started_at = datetime.now(timezone.utc)
    (
        updated_body,
        expanded_count,
        hooks,
        source_metadata_items,
    ) = _expand_claude_persisted_output_value(request_body)
    if isinstance(updated_body, dict):
        if expanded_count > 0 and _persisted_output_logging_callback is not None:
            updated_body = _persisted_output_logging_callback(
                updated_body,
                expanded_count,
                hooks,
                source_metadata_items,
            )
            litellm_metadata = updated_body.get("litellm_metadata")
            if isinstance(litellm_metadata, dict):
                langfuse_spans = litellm_metadata.get("langfuse_spans")
                if isinstance(langfuse_spans, list):
                    for span_descriptor in langfuse_spans:
                        if (
                            isinstance(span_descriptor, dict)
                            and span_descriptor.get("name")
                            == "claude.persisted_output_expand"
                        ):
                            span_descriptor["start_time"] = (
                                _format_langfuse_span_timestamp(span_started_at)
                            )
                            span_descriptor["end_time"] = (
                                _format_langfuse_span_timestamp(
                                    datetime.now(timezone.utc)
                                )
                            )
        return updated_body, expanded_count, hooks, source_metadata_items
    return request_body, 0, set(), []
