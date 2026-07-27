"""Wave 6E Anthropic body-preparation extraction.

This module owns OpenAI-adapter Claude-context compaction and its
transaction-specific logging helper, tool-block validation/repair, and
final Anthropic request-body preparation.  It intentionally does not
import ``llm_passthrough_endpoints`` at module scope.

Depends on the Wave 6D ``observability_metadata`` surface through explicit
imports.  Non-owned orchestration steps (persisted-output expansion,
control-plane rewrites, billing-header metadata, etc.) are injected via
``configure_anthropic_body_prep_runtime`` callbacks so that this module
never takes ownership of those concerns.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Tuple

from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth import (
    _clean_codex_auth_value,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.provider_shaping import (
    iter_delimited_spans,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _add_claude_child_agent_observability_metadata,
    _build_langfuse_span_descriptor,
    _merge_litellm_metadata,
)

# ---------------------------------------------------------------------------
# OpenAI-adapter context markers (constant data, owned here)
# ---------------------------------------------------------------------------

_OPENAI_ADAPTER_CONTEXT_MARKERS: tuple[tuple[str, str], ...] = (
    ("SubagentStart hook additional context:", "subagentstart"),
    ("SubAgentStart hook additional context:", "subagentstart"),
    ("# claudeMd", "claude-md"),
    ("CLAUDE.md", "claude-md"),
    ("MEMORY.md", "memory-md"),
    ("# TriStore Inject", "tristore-inject"),
)

# ---------------------------------------------------------------------------
# Runtime callback types for non-owned orchestration steps
# ---------------------------------------------------------------------------

ExpandPersistedOutputFn = Callable[
    [dict[str, Any]],
    Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]],
]
ExtractBillingHeaderFieldsFn = Callable[[dict[str, Any]], dict[str, str]]
ApplyControlPlaneRewritesFn = Callable[..., Any]
ExpandDynamicDirectivesFn = Callable[..., Any]
AddPostRewriteContextFileMetadataFn = Callable[[dict[str, Any]], dict[str, Any]]
SanitizeWebSearchDomainListsFn = Callable[
    [dict[str, Any]], Tuple[dict[str, Any], int]
]
AddBillingHeaderLoggingMetadataFn = Callable[
    [dict[str, Any], dict[str, str]], dict[str, Any]
]
AddRouteFamilyLoggingMetadataFn = Callable[
    [dict[str, Any], str], dict[str, Any]
]
AddRequestBreakoutLoggingMetadataFn = Callable[
    [dict[str, Any]], dict[str, Any]
]
PrepareObservabilityFn = Callable[..., dict[str, Any]]
GetTenantHeaderFn = Callable[[Any], Optional[str]]

_expand_persisted_output: Optional[ExpandPersistedOutputFn] = None
_extract_billing_header_fields: Optional[ExtractBillingHeaderFieldsFn] = None
_apply_control_plane_rewrites: Optional[ApplyControlPlaneRewritesFn] = None
_expand_dynamic_directives: Optional[ExpandDynamicDirectivesFn] = None
_add_post_rewrite_context_file_metadata: Optional[
    AddPostRewriteContextFileMetadataFn
] = None
_sanitize_web_search_domain_lists: Optional[SanitizeWebSearchDomainListsFn] = None
_add_billing_header_logging_metadata: Optional[
    AddBillingHeaderLoggingMetadataFn
] = None
_add_route_family_logging_metadata: Optional[AddRouteFamilyLoggingMetadataFn] = None
_add_request_breakout_logging_metadata: Optional[
    AddRequestBreakoutLoggingMetadataFn
] = None
_prepare_observability: Optional[PrepareObservabilityFn] = None
_get_tenant_header: Optional[GetTenantHeaderFn] = None


def configure_anthropic_body_prep_runtime(
    *,
    expand_persisted_output: Optional[ExpandPersistedOutputFn] = None,
    extract_billing_header_fields: Optional[ExtractBillingHeaderFieldsFn] = None,
    apply_control_plane_rewrites: Optional[ApplyControlPlaneRewritesFn] = None,
    expand_dynamic_directives: Optional[ExpandDynamicDirectivesFn] = None,
    add_post_rewrite_context_file_metadata: Optional[
        AddPostRewriteContextFileMetadataFn
    ] = None,
    sanitize_web_search_domain_lists: Optional[
        SanitizeWebSearchDomainListsFn
    ] = None,
    add_billing_header_logging_metadata: Optional[
        AddBillingHeaderLoggingMetadataFn
    ] = None,
    add_route_family_logging_metadata: Optional[
        AddRouteFamilyLoggingMetadataFn
    ] = None,
    add_request_breakout_logging_metadata: Optional[
        AddRequestBreakoutLoggingMetadataFn
    ] = None,
    prepare_observability: Optional[PrepareObservabilityFn] = None,
    get_tenant_header: Optional[GetTenantHeaderFn] = None,
) -> None:
    """Bind optional host callbacks for non-owned orchestration steps.

    Passing no callbacks restores the behavior-preserving, identity defaults.
    """
    global _expand_persisted_output
    global _extract_billing_header_fields
    global _apply_control_plane_rewrites
    global _expand_dynamic_directives
    global _add_post_rewrite_context_file_metadata
    global _sanitize_web_search_domain_lists
    global _add_billing_header_logging_metadata
    global _add_route_family_logging_metadata
    global _add_request_breakout_logging_metadata
    global _prepare_observability
    global _get_tenant_header

    _expand_persisted_output = expand_persisted_output
    _extract_billing_header_fields = extract_billing_header_fields
    _apply_control_plane_rewrites = apply_control_plane_rewrites
    _expand_dynamic_directives = expand_dynamic_directives
    _add_post_rewrite_context_file_metadata = add_post_rewrite_context_file_metadata
    _sanitize_web_search_domain_lists = sanitize_web_search_domain_lists
    _add_billing_header_logging_metadata = add_billing_header_logging_metadata
    _add_route_family_logging_metadata = add_route_family_logging_metadata
    _add_request_breakout_logging_metadata = add_request_breakout_logging_metadata
    _prepare_observability = prepare_observability
    _get_tenant_header = get_tenant_header


# ---------------------------------------------------------------------------
# OpenAI-adapter Claude-context compaction
# ---------------------------------------------------------------------------


def _get_openai_adapter_claude_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP")
    )
    if raw_value is None:
        return 1200
    try:
        parsed = int(raw_value)
    except Exception:
        return 1200
    return max(256, parsed)


def _detect_openai_adapter_claude_context_markers(text: str) -> set[str]:
    markers: set[str] = set()
    for marker_text, marker_name in _OPENAI_ADAPTER_CONTEXT_MARKERS:
        if marker_text in text:
            markers.add(marker_name)
    return markers


def _select_openai_adapter_context_summary_lines(text: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        include_line = (
            line.startswith("SubagentStart hook additional context:")
            or line.startswith("SubAgentStart hook additional context:")
            or line.startswith("#")
            or line.startswith("Contents of ")
            or line.startswith("You are '")
            or line.startswith("Codebase and user instructions")
            or line.startswith("IMPORTANT:")
        )
        if not include_line:
            continue
        if line in seen:
            continue
        selected.append(line)
        seen.add(line)
        if len(selected) >= 10:
            break
    if selected:
        return selected
    return [line.strip() for line in text.splitlines() if line.strip()][:4]


def _build_openai_adapter_compacted_claude_context_block(
    *,
    original_block: str,
    markers: set[str],
    cap: int,
) -> str:
    marker_text = ", ".join(sorted(markers)) or "unknown"
    heading = (
        "[OpenAI adapter compacted Claude Code context block "
        f"from {len(original_block)} chars. Markers: {marker_text}. "
        "The current child task, tool schemas, and latest user instructions remain authoritative.]"
    )
    summary_budget = max(0, cap - len(heading) - 64)
    summary_text = "\n".join(
        _select_openai_adapter_context_summary_lines(original_block)
    ).strip()
    if len(summary_text) > summary_budget:
        summary_text = summary_text[:summary_budget].rstrip()
    if summary_text:
        body = f"{heading}\n{summary_text}"
    else:
        body = heading
    return f"<system-reminder>\n{body}\n</system-reminder>\n"


def _compact_openai_adapter_claude_context_text(
    text: str,
    *,
    cap: Optional[int] = None,
) -> Tuple[str, int, set[str], list[dict[str, Any]]]:
    effective_cap = cap or _get_openai_adapter_claude_context_char_cap()
    updated_text = text
    compacted_count = 0
    combined_markers: set[str] = set()
    metadata_items: list[dict[str, Any]] = []

    spans = iter_delimited_spans(
        text,
        "<system-reminder>",
        "</system-reminder>",
    )
    for span in reversed(spans):
        reminder_block = text[span.start : span.end]
        markers = _detect_openai_adapter_claude_context_markers(reminder_block)
        if not markers or len(reminder_block) <= effective_cap:
            continue

        compacted_block = _build_openai_adapter_compacted_claude_context_block(
            original_block=reminder_block,
            markers=markers,
            cap=effective_cap,
        )
        updated_text = (
            updated_text[: span.start] + compacted_block + updated_text[span.end :]
        )
        compacted_count += 1
        combined_markers.update(markers)
        metadata_items.append(
            {
                "markers": sorted(markers),
                "original_chars": len(reminder_block),
                "kept_chars": len(compacted_block),
                "mode": "system_reminder_context_cap",
            }
        )

    metadata_items.reverse()
    return updated_text, compacted_count, combined_markers, metadata_items


def _compact_openai_adapter_claude_context_value(
    value: Any,
    *,
    cap: Optional[int] = None,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, str):
        return _compact_openai_adapter_claude_context_text(value, cap=cap)

    if isinstance(value, dict):
        updated_dict: dict[str, Any] = {}
        compacted_count = 0
        markers: set[str] = set()
        metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_count,
                child_markers,
                child_metadata,
            ) = _compact_openai_adapter_claude_context_value(child, cap=cap)
            updated_dict[key] = updated_child
            compacted_count += child_count
            markers.update(child_markers)
            metadata_items.extend(child_metadata)
            changed = changed or updated_child != child
        if changed:
            return updated_dict, compacted_count, markers, metadata_items
        return value, compacted_count, markers, metadata_items

    if isinstance(value, list):
        updated_list: list[Any] = []
        compacted_count = 0
        list_markers: set[str] = set()
        list_metadata_items: list[dict[str, Any]] = []
        changed = False
        for child in value:
            (
                updated_child,
                child_count,
                child_markers,
                child_metadata,
            ) = _compact_openai_adapter_claude_context_value(child, cap=cap)
            updated_list.append(updated_child)
            compacted_count += child_count
            list_markers.update(child_markers)
            list_metadata_items.extend(child_metadata)
            changed = changed or updated_child != child
        if changed:
            return updated_list, compacted_count, list_markers, list_metadata_items
        return value, compacted_count, list_markers, list_metadata_items

    return value, 0, set(), []


def _add_openai_adapter_claude_context_compaction_logging_metadata(
    request_body: dict[str, Any],
    *,
    compacted_count: int,
    markers: set[str],
    metadata_items: list[dict[str, Any]],
    span_started_at: datetime,
    tag_prefix: str = "openai-adapter",
    metadata_prefix: str = "openai_adapter",
    span_name: str = "openai_adapter.claude_context_compaction",
) -> dict[str, Any]:
    original_chars = sum(
        item.get("original_chars", 0)
        for item in metadata_items
        if isinstance(item.get("original_chars"), int)
    )
    compacted_chars = sum(
        item.get("kept_chars", 0)
        for item in metadata_items
        if isinstance(item.get("kept_chars"), int)
    )
    sorted_markers = sorted(markers)
    tags = [
        f"{tag_prefix}-claude-context-compacted",
        *[f"{tag_prefix}-claude-context:{marker}" for marker in sorted_markers],
    ]
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags,
        extra_fields={
            f"{metadata_prefix}_claude_context_compacted": True,
            f"{metadata_prefix}_claude_context_compacted_count": compacted_count,
            f"{metadata_prefix}_claude_context_markers": sorted_markers,
            f"{metadata_prefix}_claude_context_original_chars": original_chars,
            f"{metadata_prefix}_claude_context_compacted_chars": compacted_chars,
            f"{metadata_prefix}_claude_context_saved_chars": max(
                0, original_chars - compacted_chars
            ),
            f"{metadata_prefix}_claude_context_compaction_events": metadata_items,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata={
                        "compacted_count": compacted_count,
                        "markers": sorted_markers,
                        "original_chars": original_chars,
                        "compacted_chars": compacted_chars,
                        "saved_chars": max(0, original_chars - compacted_chars),
                    },
                    start_time=span_started_at,
                    end_time=datetime.now(timezone.utc),
                )
            ],
        },
    )


def _compact_openai_adapter_claude_context_in_anthropic_request_body(
    request_body: dict[str, Any],
    *,
    tag_prefix: str = "openai-adapter",
    metadata_prefix: str = "openai_adapter",
    span_name: str = "openai_adapter.claude_context_compaction",
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    compacted_count = 0
    markers: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    changed = False

    for top_level_key in ("system", "messages"):
        if top_level_key not in request_body:
            continue
        (
            updated_value,
            value_count,
            value_markers,
            value_metadata,
        ) = _compact_openai_adapter_claude_context_value(request_body[top_level_key])
        if value_count > 0:
            updated_body[top_level_key] = updated_value
            compacted_count += value_count
            markers.update(value_markers)
            metadata_items.extend(value_metadata)
            changed = True

    if not changed:
        return request_body, 0, set(), []

    updated_body = _add_openai_adapter_claude_context_compaction_logging_metadata(
        updated_body,
        compacted_count=compacted_count,
        markers=markers,
        metadata_items=metadata_items,
        span_started_at=span_started_at,
        tag_prefix=tag_prefix,
        metadata_prefix=metadata_prefix,
        span_name=span_name,
    )
    return updated_body, compacted_count, markers, metadata_items


# ---------------------------------------------------------------------------
# Tool-block validation and repair
# ---------------------------------------------------------------------------


def _validate_anthropic_tool_blocks_for_passthrough(
    request_body: dict[str, Any],
) -> None:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
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
                continue
            if block_type != "tool_result" and not (
                isinstance(block_type, str) and block_type.endswith("_tool_result")
            ):
                continue
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id.strip():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid Anthropic tool_result block at "
                        f"messages.{message_index}.content.{content_index}: "
                        "missing required non-empty string "
                        f"tool_result.tool_use_id for block type {block_type!r}"
                    ),
                )


def _repair_anthropic_tool_use_ids_for_passthrough(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return request_body, 0

    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    (
        repaired_messages,
        repaired_count,
    ) = LiteLLMAnthropicMessagesAdapter.repair_missing_anthropic_tool_use_ids(messages)
    if repaired_count == 0:
        return request_body, 0

    updated_body = dict(request_body)
    updated_body["messages"] = repaired_messages
    return (
        _merge_litellm_metadata(
            updated_body,
            tags_to_add=["anthropic-tool-use-id-repaired"],
            extra_fields={"anthropic_tool_use_id_repaired_count": repaired_count},
        ),
        repaired_count,
    )


# ---------------------------------------------------------------------------
# Final Anthropic request-body preparation
# ---------------------------------------------------------------------------


async def _prepare_anthropic_request_body_for_passthrough(
    request: Any, request_body: dict[str, Any]
) -> Tuple[dict[str, Any], int, set[str], dict[str, str]]:
    # Step 1: persisted-output expansion (owned by persisted_output.py)
    if _expand_persisted_output is not None:
        (
            updated_body,
            expanded_count,
            hooks,
            _source_metadata_items,
        ) = _expand_persisted_output(request_body)
    else:
        updated_body = request_body
        expanded_count = 0
        hooks = set()

    # Step 2: billing header extraction
    if _extract_billing_header_fields is not None:
        billing_header_fields = _extract_billing_header_fields(updated_body)
    else:
        billing_header_fields = {}

    # Step 3: control-plane rewrites (async)
    if _apply_control_plane_rewrites is not None:
        (
            updated_body,
            _claude_system_prompt_override_events,
            _claude_prompt_patch_events,
        ) = await _apply_control_plane_rewrites(
            updated_body,
            billing_header_fields,
        )

    # Step 4: dynamic directive expansion (async)
    if _expand_dynamic_directives is not None:
        (
            updated_body,
            _aawm_injection_events,
        ) = await _expand_dynamic_directives(updated_body)

    # Step 5: post-rewrite context-file metadata
    if _add_post_rewrite_context_file_metadata is not None:
        updated_body = _add_post_rewrite_context_file_metadata(updated_body)

    # Step 6: web-search domain-list sanitization
    if _sanitize_web_search_domain_lists is not None:
        (
            updated_body,
            _web_search_domain_filter_sanitized_count,
        ) = _sanitize_web_search_domain_lists(updated_body)

    # Step 7: child-agent observability (owned by observability_metadata)
    explicit_tenant_id: Optional[str] = None
    if _get_tenant_header is not None:
        explicit_tenant_id = _get_tenant_header(request)
    updated_body = _add_claude_child_agent_observability_metadata(
        updated_body,
        explicit_tenant_id=explicit_tenant_id,
    )

    # Step 8: billing-header logging metadata
    if billing_header_fields and _add_billing_header_logging_metadata is not None:
        updated_body = _add_billing_header_logging_metadata(
            updated_body,
            billing_header_fields,
        )

    # Step 9: route-family logging metadata
    if _add_route_family_logging_metadata is not None:
        updated_body = _add_route_family_logging_metadata(
            updated_body, "anthropic_messages"
        )

    # Step 10: request-breakout logging metadata
    if _add_request_breakout_logging_metadata is not None:
        updated_body = _add_request_breakout_logging_metadata(updated_body)

    # Step 11: passthrough observability
    if _prepare_observability is not None:
        updated_body = _prepare_observability(
            request=request,
            request_body=updated_body,
        )

    # Step 12: tool-use-id repair (owned here)
    (
        updated_body,
        _repaired_tool_use_id_count,
    ) = _repair_anthropic_tool_use_ids_for_passthrough(updated_body)

    # Step 13: tool-block validation (owned here)
    _validate_anthropic_tool_blocks_for_passthrough(updated_body)

    return updated_body, expanded_count, hooks, billing_header_fields
