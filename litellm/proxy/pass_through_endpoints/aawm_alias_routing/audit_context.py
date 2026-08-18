"""Audit-context extraction: request context, agent dispatch, prior-tool activity.

Wave 5D extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module are injected via :func:`configure_audit_context_runtime`.
``_clean_codex_auth_value`` is consumed from ``codex_oauth.py`` and
``_safe_get_request_headers`` from ``litellm.proxy.common_utils.http_parsing_utils``
through direct sibling imports.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional, Union
from uuid import uuid4

from fastapi import Request
from typing_extensions import NotRequired, TypedDict

from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers

from .codex_oauth import _clean_codex_auth_value
from .request_metadata import (
    _extract_auto_agent_alias_canonical_thread_id,
    _extract_auto_agent_alias_parent_thread_id,
)
from .types import Payload

# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / audit_build dependencies)
# ---------------------------------------------------------------------------

_clean_secret_string: Optional[Callable[[Optional[str]], Optional[str]]] = None
_extract_auto_agent_alias_metadata_value: Optional[Callable[..., Optional[str]]] = None
_extract_auto_agent_alias_client_product_label: Optional[Callable[..., Optional[str]]] = None
_resolve_auto_agent_alias_route_host_attribution: Optional[Callable[..., dict[str, Optional[str]]]] = None
_extract_auto_agent_alias_session_id: Optional[Callable[..., Optional[str]]] = None
_build_auto_agent_alias_rollup_group_header_label: Optional[Callable[..., Optional[str]]] = None
_codex_auto_agent_request_has_continuation_state: Optional[Callable[..., bool]] = None

_host_globals: Optional[dict] = None


def configure_audit_context_runtime(
    *,
    clean_secret_string: Callable[[Optional[str]], Optional[str]],
    extract_metadata_value: Callable[..., Optional[str]],
    extract_client_product_label: Callable[..., Optional[str]],
    resolve_host_attribution: Callable[..., dict[str, Optional[str]]],
    extract_session_id: Callable[..., Optional[str]],
    build_rollup_group_header_label: Callable[..., Optional[str]],
    has_continuation_state: Callable[..., bool],
) -> None:
    """Inject god-module / audit_build dependencies for audit_context functions."""
    global _clean_secret_string
    global _extract_auto_agent_alias_metadata_value
    global _extract_auto_agent_alias_client_product_label
    global _resolve_auto_agent_alias_route_host_attribution
    global _extract_auto_agent_alias_session_id
    global _build_auto_agent_alias_rollup_group_header_label
    global _codex_auto_agent_request_has_continuation_state

    _clean_secret_string = clean_secret_string
    _extract_auto_agent_alias_metadata_value = extract_metadata_value
    _extract_auto_agent_alias_client_product_label = extract_client_product_label
    _resolve_auto_agent_alias_route_host_attribution = resolve_host_attribution
    _extract_auto_agent_alias_session_id = extract_session_id
    _build_auto_agent_alias_rollup_group_header_label = build_rollup_group_header_label
    _codex_auto_agent_request_has_continuation_state = has_continuation_state

    if _host_globals is not None:
        _host_globals.update({
            "_clean_secret_string": _clean_secret_string,
            "_extract_auto_agent_alias_metadata_value": _extract_auto_agent_alias_metadata_value,
            "_extract_auto_agent_alias_client_product_label": _extract_auto_agent_alias_client_product_label,
            "_resolve_auto_agent_alias_route_host_attribution": _resolve_auto_agent_alias_route_host_attribution,
            "_extract_auto_agent_alias_session_id": _extract_auto_agent_alias_session_id,
            "_build_auto_agent_alias_rollup_group_header_label": _build_auto_agent_alias_rollup_group_header_label,
            "_codex_auto_agent_request_has_continuation_state": _codex_auto_agent_request_has_continuation_state,
        })


# ---------------------------------------------------------------------------
# Owned constants
# ---------------------------------------------------------------------------

_AUTO_AGENT_ROLE_DECLARATION_RE = re.compile(
    r"^[ \t]*You are a '(?P<agent>explorer|worker|default)' agent\.[ \t]*$",
    re.MULTILINE,
)
_AUTO_AGENT_KNOWN_ROLE_NAMES = frozenset({"explorer", "worker", "default"})
_AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "tool_use",
        "tool_result",
        "mcp_call",
        "mcp_tool_call",
        "mcp_approval_request",
        "mcp_approval_response",
    }
)
_AUTO_AGENT_FILE_EDIT_TOOL_NAMES = frozenset(
    {
        "apply_patch",
        "create_file",
        "edit_file",
        "multi_replace_file_content",
        "replace_file_content",
        "write_file",
    }
)
_AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY = "aawm_alias_request_litellm_call_id"
_AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY = "aawm_alias_request_context"


# ---------------------------------------------------------------------------
# Owned functions (frozen against baseline 66963d07ce)
# ---------------------------------------------------------------------------


def _extract_auto_agent_alias_text_blobs(request_body: dict[str, Any]) -> list[str]:
    blobs: list[str] = []
    for key in ("instructions", "system"):
        value = request_body.get(key)
        if isinstance(value, str) and value.strip():
            blobs.append(value)
        elif isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            if parts:
                blobs.append("\n".join(parts))
    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages[:5]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            if role not in {"system", "developer"}:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                blobs.append(content)
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    elif isinstance(block, str) and block.strip():
                        parts.append(block)
                if parts:
                    blobs.append("\n".join(parts))
    return blobs


def _extract_auto_agent_alias_role_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    match = _AUTO_AGENT_ROLE_DECLARATION_RE.search(text)
    if not match:
        return None
    role = _clean_codex_auth_value(match.group("agent"))
    if role is None:
        return None
    normalized = role.lower()
    return normalized if normalized in _AUTO_AGENT_KNOWN_ROLE_NAMES else None


def _infer_auto_agent_alias_role_from_request_body(
    request_body: dict[str, Any],
) -> Optional[str]:
    for blob in _extract_auto_agent_alias_text_blobs(request_body):
        role = _extract_auto_agent_alias_role_from_text(blob)
        if role is not None:
            return role
    return None


def _iter_auto_agent_alias_metadata_dicts(
    request: Request,
    request_body: dict[str, Any],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for candidate in (
        request_body.get("litellm_metadata"),
        request_body.get("metadata"),
        request_body.get("source"),
    ):
        if isinstance(candidate, dict):
            sources.append(candidate)
            nested_source = candidate.get("source")
            if isinstance(nested_source, dict):
                sources.append(nested_source)
    headers = _safe_get_request_headers(request)
    if isinstance(headers, dict) and headers:
        # RR-054 #56: only trusted/low-cardinality header keys may influence identity fields.
        allowed = {
            "x-aawm-agent-id",
            "x-aawm-agent-name",
            "x-aawm-agent-role",
            "x-aawm-dispatch-id",
            "x-aawm-session-id",
            "x-session-id",
            "session_id",
            "session-id",
            "x-litellm-session-id",
        }
        filtered = {key: value for key, value in headers.items() if isinstance(key, str) and key.lower() in allowed}
        if filtered:
            sources.append(filtered)
    return sources


def _extract_auto_agent_alias_agent_dispatch_fields(
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Extract explicit agent/dispatch fields with role-declaration fallback.

    Prefer structured metadata when present. Fall back only to the exact
    profile sentence form: ``You are a '<role>' agent.`` for explorer/worker/default.
    """
    fields: dict[str, Any] = {}
    sources = _iter_auto_agent_alias_metadata_dicts(request, request_body)

    def _first_from_sources(*keys: str) -> Optional[str]:
        for source in sources:
            for key in keys:
                value = _clean_codex_auth_value(source.get(key))
                if value is not None:
                    return value
        return None

    agent_name = _first_from_sources(
        "agent_name",
        "aawm_agent_name",
        "aawm_claude_agent_name",
        "codex_agent_name",
    )
    agent_role = _first_from_sources(
        "agent_role",
        "aawm_agent_role",
        "codex_agent_role",
        "agent_nickname",
    )
    agent_id = _first_from_sources(
        "agent_id",
        "aawm_agent_id",
        "codex_agent_id",
        "claude_agent_id",
        "subagent_id",
        "source_agent_id",
    )
    thread_source = _first_from_sources(
        "thread_source",
        "aawm_thread_source",
        "codex_thread_source",
    )
    redispatch_ordinal: Optional[Union[int, str]] = None
    for source in sources:
        for key in (
            "redispatch_ordinal",
            "agent_redispatch_ordinal",
            "dispatch_ordinal",
            "aawm_redispatch_ordinal",
        ):
            raw = source.get(key)
            if raw is None or raw == "":
                continue
            try:
                redispatch_ordinal = int(raw)
            except (TypeError, ValueError):
                redispatch_ordinal = _clean_codex_auth_value(raw)
            if redispatch_ordinal is not None:
                break
        if redispatch_ordinal is not None:
            break
    dispatch_id = _first_from_sources(
        "dispatch_id",
        "agent_dispatch_id",
        "aawm_dispatch_id",
        "codex_dispatch_id",
    )
    agent_profile = _first_from_sources(
        "agent_profile",
        "aawm_agent_profile",
        "codex_agent_profile",
    )

    inferred_role = (
        _infer_auto_agent_alias_role_from_request_body(request_body)
        if agent_name is None or agent_role is None
        else None
    )
    if inferred_role is not None:
        if agent_role is None:
            agent_role = inferred_role
        if agent_name is None:
            agent_name = inferred_role
        if agent_profile is None:
            agent_profile = inferred_role
    if thread_source is None and inferred_role is not None:
        thread_source = "subagent"

    if agent_name is not None:
        fields["agent_name"] = agent_name
    if agent_role is not None:
        fields["agent_role"] = agent_role
    if agent_id is not None:
        fields["agent_id"] = agent_id
    if thread_source is not None:
        fields["thread_source"] = thread_source
    if redispatch_ordinal is not None:
        fields["redispatch_ordinal"] = redispatch_ordinal
    if dispatch_id is not None:
        fields["dispatch_id"] = dispatch_id
    if agent_profile is not None:
        fields["agent_profile"] = agent_profile
    return fields


def _walk_auto_agent_alias_prior_tool_activity(
    value: Any,
    *,
    tool_names: list[str],
    file_edit_tool_names: list[str],
    counters: dict[str, int],
    _seen: Optional[set[int]] = None,
) -> None:
    if isinstance(value, (dict, list)):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return
        _seen.add(value_id)

    def _record_tool_name(name: Any, *, is_call: bool) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        clean_name = name.strip()
        if clean_name not in tool_names:
            tool_names.append(clean_name)
        if not is_call:
            return
        normalized = clean_name.lower().replace("-", "_")
        short_name = re.split(r"[./:]", normalized)[-1]
        if short_name not in _AUTO_AGENT_FILE_EDIT_TOOL_NAMES:
            return
        counters["prior_file_edit_tool_call_count"] += 1
        if clean_name not in file_edit_tool_names:
            file_edit_tool_names.append(clean_name)

    if isinstance(value, dict):
        item_type = value.get("type")
        role = str(value.get("role") or "").lower()
        if isinstance(item_type, str) and item_type in _AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES:
            # RR-054 #58: single membership test for call vs result item types.
            is_tool_call = item_type in {
                "function_call",
                "tool_use",
                "mcp_call",
                "mcp_tool_call",
                "mcp_approval_request",
            }
            if is_tool_call:
                counters["prior_tool_call_count"] += 1
            elif item_type in {
                "function_call_output",
                "tool_result",
                "mcp_approval_response",
            }:
                counters["prior_tool_result_count"] += 1
            name = value.get("name")
            if not isinstance(name, str) or not name:
                function_obj = value.get("function")
                if isinstance(function_obj, dict):
                    name = function_obj.get("name")
            _record_tool_name(name, is_call=is_tool_call)
        if role == "tool" or value.get("tool_call_id") or value.get("tool_calls"):
            if role == "tool" or value.get("tool_call_id"):
                counters["prior_tool_result_count"] += 1
            tool_calls = value.get("tool_calls")
            if isinstance(tool_calls, list):
                counters["prior_tool_call_count"] += len(tool_calls)
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    name = tool_call.get("name")
                    function_obj = tool_call.get("function")
                    if not isinstance(name, str) or not name:
                        if isinstance(function_obj, dict):
                            name = function_obj.get("name")
                    _record_tool_name(name, is_call=True)
        for child in value.values():
            _walk_auto_agent_alias_prior_tool_activity(
                child,
                tool_names=tool_names,
                file_edit_tool_names=file_edit_tool_names,
                counters=counters,
                _seen=_seen,
            )
        return

    if isinstance(value, list):
        for item in value:
            _walk_auto_agent_alias_prior_tool_activity(
                item,
                tool_names=tool_names,
                file_edit_tool_names=file_edit_tool_names,
                counters=counters,
                _seen=_seen,
            )


def _summarize_auto_agent_alias_actual_prior_tool_activity(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Conservative summary of actual prior tool activity in the request.

    Counts only concrete tool-call/result evidence already present in the
    request body. Continuation markers alone do not count as partial activity.
    """
    assert _codex_auto_agent_request_has_continuation_state is not None, (
        "audit_context runtime not configured: _codex_auto_agent_request_has_continuation_state"
    )
    tool_names: list[str] = []
    file_edit_tool_names: list[str] = []
    counters = {
        "prior_tool_call_count": 0,
        "prior_tool_result_count": 0,
        "prior_file_edit_tool_call_count": 0,
    }
    for key in ("input", "messages", "content", "tools_results", "tool_results"):
        if key in request_body:
            _walk_auto_agent_alias_prior_tool_activity(
                request_body.get(key),
                tool_names=tool_names,
                file_edit_tool_names=file_edit_tool_names,
                counters=counters,
            )
    has_actual_prior_tool_activity = bool(counters["prior_tool_call_count"] or counters["prior_tool_result_count"])
    has_previous_response_id = bool(_clean_codex_auth_value(request_body.get("previous_response_id")))
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(request_body)
    return {
        "has_actual_prior_tool_activity": has_actual_prior_tool_activity,
        "prior_tool_call_count": counters["prior_tool_call_count"],
        "prior_tool_result_count": counters["prior_tool_result_count"],
        "prior_tool_names": tool_names[:20],
        "has_prior_file_edit_activity": bool(counters["prior_file_edit_tool_call_count"]),
        "prior_file_edit_tool_call_count": counters["prior_file_edit_tool_call_count"],
        "prior_file_edit_tool_names": file_edit_tool_names[:20],
        "has_previous_response_id": has_previous_response_id,
        "has_continuation_state": bool(has_continuation_state),
    }


def _classify_auto_agent_alias_terminal_activity_status(
    prior_tool_activity_summary: Optional[dict[str, Any]],
) -> str:
    if isinstance(prior_tool_activity_summary, dict) and prior_tool_activity_summary.get(
        "has_actual_prior_tool_activity"
    ):
        return "failed_after_partial_activity"
    return "failed_no_activity"


def _get_or_create_auto_agent_alias_request_call_id(
    request: Request,
    request_body: dict[str, Any],
) -> str:
    """Return one stable call ID for all alias audit events in this request."""
    assert _extract_auto_agent_alias_metadata_value is not None, (
        "audit_context runtime not configured: _extract_auto_agent_alias_metadata_value"
    )
    request_state = getattr(request, "state", None)
    if request_state is not None:
        existing = getattr(
            request_state,
            _AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY,
            None,
        )
        if isinstance(existing, str) and existing.strip():
            return existing.strip()

    litellm_call_id = _extract_auto_agent_alias_metadata_value(
        request_body,
        "litellm_call_id",
        "call_id",
        "aawm_litellm_call_id",
    )
    if litellm_call_id is None:
        scope = request.scope if isinstance(getattr(request, "scope", None), dict) else {}
        for key in ("litellm_call_id", "call_id", "request_id"):
            value = _clean_codex_auth_value(scope.get(key))
            if value is not None:
                litellm_call_id = value
                break
    if litellm_call_id is None and request_state is not None:
        for key in ("litellm_call_id", "call_id", "request_id"):
            value = getattr(request_state, key, None)
            if not isinstance(value, str):
                continue
            cleaned = _clean_codex_auth_value(value)
            if cleaned is not None:
                litellm_call_id = cleaned
                break
    if litellm_call_id is None:
        headers = _safe_get_request_headers(request)
        if isinstance(headers, dict):
            for key in (
                "x-litellm-call-id",
                "litellm-call-id",
                "x-request-id",
            ):
                value = _clean_codex_auth_value(headers.get(key))
                if value is not None:
                    litellm_call_id = value
                    break
    if litellm_call_id is None:
        litellm_call_id = str(uuid4())

    if request_state is not None:
        setattr(
            request_state,
            _AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY,
            litellm_call_id,
        )
    return litellm_call_id


class _AutoAgentAliasRequestContext(TypedDict):
    agent_dispatch: dict[str, Any]
    session_id: Optional[str]
    canonical_thread_id: Optional[str]
    parent_thread_id: Optional[str]
    litellm_call_id: str
    trace_id: Optional[str]
    repository: Optional[str]
    client_product_label: Optional[str]
    host_attribution: dict[str, Optional[str]]
    rollup_group_header_label: Optional[str]
    prior_tool_activity_summary: NotRequired[dict[str, Any]]


def _normalize_auto_agent_alias_request_context(
    value: Mapping[str, object],
) -> _AutoAgentAliasRequestContext:
    assert _clean_secret_string is not None, (
        "audit_context runtime not configured: _clean_secret_string"
    )
    agent_dispatch_value = value.get("agent_dispatch")
    host_attribution_value = value.get("host_attribution")
    context: _AutoAgentAliasRequestContext = {
        "agent_dispatch": (dict(agent_dispatch_value) if isinstance(agent_dispatch_value, dict) else {}),
        "session_id": _clean_optional_string(value.get("session_id")),
        "canonical_thread_id": _clean_optional_string(value.get("canonical_thread_id")),
        "parent_thread_id": _clean_optional_string(value.get("parent_thread_id")),
        "litellm_call_id": _clean_optional_string(value.get("litellm_call_id")) or str(uuid4()),
        "trace_id": _clean_optional_string(value.get("trace_id")),
        "repository": _clean_optional_string(value.get("repository")),
        "client_product_label": _clean_optional_string(value.get("client_product_label")),
        "host_attribution": (
            {str(key): item if isinstance(item, str) else None for key, item in host_attribution_value.items()}
            if isinstance(host_attribution_value, dict)
            else {}
        ),
        "rollup_group_header_label": _clean_optional_string(value.get("rollup_group_header_label")),
    }
    prior_activity = value.get("prior_tool_activity_summary")
    if isinstance(prior_activity, dict):
        context["prior_tool_activity_summary"] = dict(prior_activity)
    return context


def _clean_optional_string(value: object) -> Optional[str]:
    assert _clean_secret_string is not None, (
        "audit_context runtime not configured: _clean_secret_string"
    )
    return _clean_secret_string(value if isinstance(value, str) else None)


def _get_auto_agent_alias_request_context(
    request: Request,
    request_body: Payload,
    *,
    include_activity: bool = False,
) -> _AutoAgentAliasRequestContext:
    assert _extract_auto_agent_alias_metadata_value is not None, (
        "audit_context runtime not configured: _extract_auto_agent_alias_metadata_value"
    )
    assert _extract_auto_agent_alias_client_product_label is not None, (
        "audit_context runtime not configured: _extract_auto_agent_alias_client_product_label"
    )
    assert _resolve_auto_agent_alias_route_host_attribution is not None, (
        "audit_context runtime not configured: _resolve_auto_agent_alias_route_host_attribution"
    )
    assert _extract_auto_agent_alias_session_id is not None, (
        "audit_context runtime not configured: _extract_auto_agent_alias_session_id"
    )
    assert _build_auto_agent_alias_rollup_group_header_label is not None, (
        "audit_context runtime not configured: _build_auto_agent_alias_rollup_group_header_label"
    )
    request_state = getattr(request, "state", None)
    cached_value = (
        getattr(request_state, _AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY, None) if request_state is not None else None
    )
    if isinstance(cached_value, dict):
        cached = _normalize_auto_agent_alias_request_context(cached_value)
    else:
        repository = _extract_auto_agent_alias_metadata_value(
            request_body,
            "repository",
            "repo",
            "repo_name",
            "repository_name",
        )
        client_product_label = _extract_auto_agent_alias_client_product_label(
            request,
            request_body,
        )
        host_attribution = _resolve_auto_agent_alias_route_host_attribution(request)
        canonical_thread_id = _extract_auto_agent_alias_canonical_thread_id(
            request,
            request_body,
        )
        parent_thread_id = _extract_auto_agent_alias_parent_thread_id(
            request,
            request_body,
        )
        legacy_session_id = _extract_auto_agent_alias_session_id(
            request,
            request_body,
        )
        cached = {
            "agent_dispatch": _extract_auto_agent_alias_agent_dispatch_fields(
                request,
                request_body,
            ),
            "session_id": (
                canonical_thread_id or legacy_session_id or parent_thread_id
            ),
            "canonical_thread_id": canonical_thread_id,
            "parent_thread_id": parent_thread_id,
            "litellm_call_id": _get_or_create_auto_agent_alias_request_call_id(
                request,
                request_body,
            ),
            "trace_id": _extract_auto_agent_alias_metadata_value(
                request_body,
                "trace_id",
                "langfuse_trace_id",
                "aawm_trace_id",
            ),
            "repository": repository,
            "client_product_label": client_product_label,
            "host_attribution": host_attribution,
            "rollup_group_header_label": (
                _build_auto_agent_alias_rollup_group_header_label(
                    repository=repository,
                    client_product_label=client_product_label,
                    host_name=host_attribution.get("host_name"),
                )
            ),
        }
        if request_state is not None:
            setattr(request_state, _AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY, cached)
    if include_activity and "prior_tool_activity_summary" not in cached:
        cached["prior_tool_activity_summary"] = _summarize_auto_agent_alias_actual_prior_tool_activity(request_body)
    if request_state is not None:
        setattr(request_state, _AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY, cached)
    return cached


def _attach_auto_agent_alias_terminal_context_fields(
    event: dict[str, Any],
    *,
    request: Request,
    request_body: dict[str, Any],
    selection: Optional[dict[str, Any]] = None,
    candidate: Optional[dict[str, Any]] = None,
    include_activity_status: bool = False,
) -> dict[str, Any]:
    """Attach direct context IDs, agent/dispatch fields, and activity summary."""
    assert _extract_auto_agent_alias_metadata_value is not None, (
        "audit_context runtime not configured: _extract_auto_agent_alias_metadata_value"
    )
    needs_activity = include_activity_status or event.get("event_type") in {
        "no_candidate_available",
        "redispatch_required",
        "agent_session_terminated",
    }
    context = _get_auto_agent_alias_request_context(
        request,
        request_body,
        include_activity=needs_activity,
    )
    agent_dispatch = context["agent_dispatch"]
    for key, value in agent_dispatch.items():
        if value is not None and event.get(key) is None:
            event[key] = value

    # Prefer existing agent_id extraction when structured fields omit it.
    if event.get("agent_id") is None:
        agent_id = _extract_auto_agent_alias_metadata_value(
            request_body,
            "agent_id",
            "aawm_agent_id",
            "codex_agent_id",
            "claude_agent_id",
        )
        if agent_id is not None:
            event["agent_id"] = agent_id

    if event.get("session_id") is None:
        event["session_id"] = context.get("session_id")

    if event.get("canonical_thread_id") is None:
        event["canonical_thread_id"] = context.get("canonical_thread_id")

    if event.get("parent_thread_id") is None:
        event["parent_thread_id"] = context.get("parent_thread_id")

    if event.get("litellm_call_id") is None:
        event["litellm_call_id"] = context.get("litellm_call_id")

    trace_id = context.get("trace_id")
    if trace_id is not None and event.get("trace_id") is None:
        event["trace_id"] = trace_id

    cooldown_state_source = None
    if isinstance(candidate, dict):
        cooldown_state_source = candidate.get("cooldown_state_source")
    if cooldown_state_source is None and isinstance(selection, dict):
        cooldown_state_source = selection.get("cooldown_state_source")
    if cooldown_state_source is not None and event.get("cooldown_state_source") is None:
        event["cooldown_state_source"] = cooldown_state_source

    # RR-054 #29: expensive full-body tool-activity walk only when terminal status
    # classification actually needs it (or explicit include flag).
    if needs_activity:
        prior_summary = context.get("prior_tool_activity_summary")
        event["actual_prior_tool_activity_summary"] = prior_summary
    if include_activity_status:
        event["terminal_activity_status"] = _classify_auto_agent_alias_terminal_activity_status(prior_summary)
    return event


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5D)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_extract_auto_agent_alias_text_blobs",
    "_extract_auto_agent_alias_role_from_text",
    "_infer_auto_agent_alias_role_from_request_body",
    "_iter_auto_agent_alias_metadata_dicts",
    "_extract_auto_agent_alias_agent_dispatch_fields",
    "_walk_auto_agent_alias_prior_tool_activity",
    "_summarize_auto_agent_alias_actual_prior_tool_activity",
    "_classify_auto_agent_alias_terminal_activity_status",
    "_get_or_create_auto_agent_alias_request_call_id",
    "_normalize_auto_agent_alias_request_context",
    "_clean_optional_string",
    "_get_auto_agent_alias_request_context",
    "_attach_auto_agent_alias_terminal_context_fields",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    global _host_globals
    _host_globals = host_globals
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = _FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound
    # Copy seam variables into host_globals so rebound functions resolve them.
    # Only copy seams not already present (preserves god-module defs and prior rebinds).
    for _sk, _sv in (
        ("_clean_secret_string", _clean_secret_string),
        ("_extract_auto_agent_alias_metadata_value", _extract_auto_agent_alias_metadata_value),
        ("_extract_auto_agent_alias_client_product_label", _extract_auto_agent_alias_client_product_label),
        ("_resolve_auto_agent_alias_route_host_attribution", _resolve_auto_agent_alias_route_host_attribution),
        ("_extract_auto_agent_alias_session_id", _extract_auto_agent_alias_session_id),
        ("_build_auto_agent_alias_rollup_group_header_label", _build_auto_agent_alias_rollup_group_header_label),
        ("_codex_auto_agent_request_has_continuation_state", _codex_auto_agent_request_has_continuation_state),
        (
            "_extract_auto_agent_alias_canonical_thread_id",
            _extract_auto_agent_alias_canonical_thread_id,
        ),
        (
            "_extract_auto_agent_alias_parent_thread_id",
            _extract_auto_agent_alias_parent_thread_id,
        ),
    ):
        host_globals.setdefault(_sk, _sv)

# ---------------------------------------------------------------------------
# Module __setattr__ propagation for test-fixture seam restores
# ---------------------------------------------------------------------------
# After install(), rebound functions resolve seams from host_globals.
# Module-local test fixtures restore seams via setattr(module, name, val).
# This hook propagates those restores into host_globals so rebound functions
# see the restored values, preserving test isolation.

import sys as _sys
import types as _types

_SEAM_NAMES = frozenset({
    "_clean_secret_string",
    "_extract_auto_agent_alias_metadata_value",
    "_extract_auto_agent_alias_client_product_label",
    "_resolve_auto_agent_alias_route_host_attribution",
    "_extract_auto_agent_alias_session_id",
    "_build_auto_agent_alias_rollup_group_header_label",
    "_codex_auto_agent_request_has_continuation_state",
})


class _SeamPropagatingModule(_types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        seam_names = self.__dict__.get("_SEAM_NAMES")
        if seam_names is not None and name in seam_names:
            hg = self.__dict__.get("_host_globals")
            if hg is not None:
                hg[name] = value


_sys.modules[__name__].__class__ = _SeamPropagatingModule
