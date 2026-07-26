"""Audit event construction for auto-agent alias routing.

Wave 5D extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module and ``audit_context.py`` are injected via
:func:`configure_audit_build_runtime`.  Direct imports from sibling Wave 4/5
modules (``lane_keys``, ``selection``, ``codex_oauth``) are used where those
modules own the symbols.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from fastapi import HTTPException, Request

from .codex_oauth import _clean_codex_auth_value
from .lane_keys import (
    _CODEX_AUTO_AGENT_REASONING_EFFORT_AUDIT_FIELDS,
    _codex_auto_agent_candidate_key,
)
from .selection import (
    _auto_agent_alias_float,
    _codex_auto_agent_candidate_public_shape,
)

# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / audit_context.py)
# ---------------------------------------------------------------------------

_get_auto_agent_alias_request_context: Optional[
    Callable[..., Mapping[str, Any]]
] = None
_attach_auto_agent_alias_terminal_context_fields: Optional[
    Callable[..., object]
] = None
_format_auto_agent_alias_timestamp: Optional[Callable[[datetime], str]] = None
_extract_auto_agent_alias_metadata_value: Optional[Callable[..., Optional[str]]] = None
_extract_auto_agent_alias_incoming_endpoint: Optional[Callable[..., str]] = None
_resolve_auto_agent_alias_route_rollup_outgoing_target: Optional[
    Callable[..., Optional[str]]
] = None
_auto_agent_alias_int: Optional[Callable[..., Optional[int]]] = None
_auto_agent_alias_cooldown_until: Optional[Callable[..., Optional[str]]] = None

_host_globals: Optional[dict] = None


def configure_audit_build_runtime(
    *,
    get_request_context: Callable[..., Mapping[str, Any]],
    attach_terminal_context_fields: Callable[..., object],
    format_timestamp: Callable[[datetime], str],
    extract_metadata_value: Callable[..., Optional[str]],
    extract_incoming_endpoint: Callable[..., str],
    resolve_outgoing_target: Callable[..., Optional[str]],
    to_int: Callable[..., Optional[int]],
    cooldown_until: Callable[..., Optional[str]],
) -> None:
    """Inject god-module and audit_context.py dependencies."""
    global _get_auto_agent_alias_request_context
    global _attach_auto_agent_alias_terminal_context_fields
    global _format_auto_agent_alias_timestamp
    global _extract_auto_agent_alias_metadata_value
    global _extract_auto_agent_alias_incoming_endpoint
    global _resolve_auto_agent_alias_route_rollup_outgoing_target
    global _auto_agent_alias_int
    global _auto_agent_alias_cooldown_until

    _get_auto_agent_alias_request_context = get_request_context
    _attach_auto_agent_alias_terminal_context_fields = attach_terminal_context_fields
    _format_auto_agent_alias_timestamp = format_timestamp
    _extract_auto_agent_alias_metadata_value = extract_metadata_value
    _extract_auto_agent_alias_incoming_endpoint = extract_incoming_endpoint
    _resolve_auto_agent_alias_route_rollup_outgoing_target = resolve_outgoing_target
    _auto_agent_alias_int = to_int
    _auto_agent_alias_cooldown_until = cooldown_until

    if _host_globals is not None:
        _host_globals.update({
            "_get_auto_agent_alias_request_context": _get_auto_agent_alias_request_context,
            "_attach_auto_agent_alias_terminal_context_fields": _attach_auto_agent_alias_terminal_context_fields,
            "_format_auto_agent_alias_timestamp": _format_auto_agent_alias_timestamp,
            "_extract_auto_agent_alias_metadata_value": _extract_auto_agent_alias_metadata_value,
            "_extract_auto_agent_alias_incoming_endpoint": _extract_auto_agent_alias_incoming_endpoint,
            "_resolve_auto_agent_alias_route_rollup_outgoing_target": _resolve_auto_agent_alias_route_rollup_outgoing_target,
            "_auto_agent_alias_int": _auto_agent_alias_int,
            "_auto_agent_alias_cooldown_until": _auto_agent_alias_cooldown_until,
        })


# ---------------------------------------------------------------------------
# Frozen symbols (baseline 66963d07ce)
# ---------------------------------------------------------------------------


def _is_auto_agent_alias_in_flight_cooldown_http_exception(
    exc: HTTPException,
) -> bool:
    detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
    error = detail.get("error") if isinstance(detail, dict) else None
    error_code = error.get("code") if isinstance(error, dict) else None
    return bool(detail.get("redispatch_required")) or error_code in {
        "aawm_codex_auto_agent_in_flight_provider_cooling_down",
        "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
        "aawm_codex_auto_agent_redispatch_required",
        "aawm_anthropic_auto_agent_redispatch_required",
    }


def _build_auto_agent_alias_audit_event(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    selection: dict[str, Any],
    candidate: dict[str, Any],
    event_type: str,
    candidate_status: str,
    attempt_number: Optional[int] = None,
    selected: bool = False,
    skipped: bool = False,
    selection_reason: Optional[str] = None,
    lane_key: Optional[str] = None,
    cooldown_key: Optional[str] = None,
    cooldown_seconds: Optional[Any] = None,
    cooldown_scope: Optional[str] = None,
    failure_class: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    error_tokens: Optional[Any] = None,
    source_error: Optional[str] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    redispatch_required: bool = False,
) -> dict[str, Any]:
    assert _get_auto_agent_alias_request_context is not None
    assert _attach_auto_agent_alias_terminal_context_fields is not None
    assert _format_auto_agent_alias_timestamp is not None
    assert _extract_auto_agent_alias_metadata_value is not None
    assert _extract_auto_agent_alias_incoming_endpoint is not None
    assert _resolve_auto_agent_alias_route_rollup_outgoing_target is not None
    assert _auto_agent_alias_int is not None
    assert _auto_agent_alias_cooldown_until is not None

    normalized_cooldown_seconds = _auto_agent_alias_float(cooldown_seconds)
    if lane_key is None:
        lane_key = selection.get("lane_key")
    if cooldown_key is None and lane_key is not None:
        cooldown_key = _codex_auto_agent_candidate_key(
            candidate, lane_key, epoch_tag=candidate.get("config_epoch_tag"),
        )
    context = _get_auto_agent_alias_request_context(
        request,
        request_body,
    )
    repository = context.get("repository")
    client_product_label = context.get("client_product_label")
    host_attribution = context["host_attribution"]
    event: dict[str, Any] = {
        "observed_at": _format_auto_agent_alias_timestamp(datetime.now(timezone.utc)),
        "alias_family": alias_family,
        "alias_model": alias_model,
        "session_id": context.get("session_id"),
        "agent_id": _extract_auto_agent_alias_metadata_value(
            request_body,
            "agent_id",
            "aawm_agent_id",
            "codex_agent_id",
            "claude_agent_id",
        ),
        "repository": repository,
        "client_product_label": client_product_label,
        "client_ip": host_attribution.get("client_ip"),
        "client_ip_source": host_attribution.get("client_ip_source"),
        "host_name": host_attribution.get("host_name"),
        "host_name_source": host_attribution.get("host_name_source"),
        "rollup_group_header_label": context.get("rollup_group_header_label"),
        "incoming_endpoint": _extract_auto_agent_alias_incoming_endpoint(request),
        "outgoing_target": _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family=candidate.get("route_family"),
            target_url=candidate.get("target_url"),
        ),
        "target_url": candidate.get("target_url"),
        "session_key": selection.get("session_key"),
        "provider": candidate.get("provider"),
        "model": candidate.get("model"),
        "route_family": candidate.get("route_family"),
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "attempt_number": attempt_number,
        "event_type": event_type,
        "selection_reason": selection_reason,
        "candidate_status": candidate_status,
        "failure_class": failure_class,
        "error_status_code": _auto_agent_alias_int(error_status_code),
        "error_type": error_type,
        "error_code": str(error_code) if error_code is not None else None,
        "source_error": _clean_codex_auth_value(source_error),
        "retry_after_seconds": _auto_agent_alias_float(retry_after_seconds),
        "failure_phase": failure_phase,
        "attempted_provider_call": attempted_provider_call,
        "cooldown_scope": cooldown_scope,
        "cooldown_seconds": (
            round(normalized_cooldown_seconds, 3) if normalized_cooldown_seconds is not None else None
        ),
        "cooldown_until": _auto_agent_alias_cooldown_until(normalized_cooldown_seconds),
        "selected": selected,
        "skipped": skipped,
        "last_resort": bool(candidate.get("last_resort")),
        "in_flight_session": bool(selection.get("in_flight_session")),
        "redispatch_required": redispatch_required,
        "redispatch_threshold_crossed": False,
    }
    if isinstance(error_tokens, list):
        event["error_tokens"] = error_tokens
    elif isinstance(error_tokens, set):
        event["error_tokens"] = sorted(error_tokens)

    cooldown_state_source = candidate.get("cooldown_state_source")
    if cooldown_state_source is None:
        cooldown_state_source = selection.get("cooldown_state_source")
    if cooldown_state_source is not None:
        event["cooldown_state_source"] = cooldown_state_source
    for field in _CODEX_AUTO_AGENT_REASONING_EFFORT_AUDIT_FIELDS:
        value = candidate.get(field)
        if value is not None:
            event[field] = value

    include_activity_status = (
        event_type
        in {
            "no_candidate_available",
            "redispatch_required",
            "candidate_retryable_failure",
        }
        or bool(redispatch_required)
        or (isinstance(error_status_code, int) and error_status_code == 429)
    )
    _attach_auto_agent_alias_terminal_context_fields(
        event,
        request=request,
        request_body=request_body,
        selection=selection,
        candidate=candidate,
        include_activity_status=include_activity_status,
    )
    return {key: value for key, value in event.items() if value is not None}


def _build_auto_agent_alias_audit_events(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    skipped_candidates = selection.get("skipped")
    if isinstance(skipped_candidates, list):
        for skipped_candidate in skipped_candidates:
            if not isinstance(skipped_candidate, dict):
                continue
            reason = str(skipped_candidate.get("reason") or "cooldown")
            event_type = (
                "candidate_skipped_provider_degraded" if reason == "auth_degraded" else "candidate_skipped_cooldown"
            )
            events.append(
                _build_auto_agent_alias_audit_event(
                    alias_family=alias_family,
                    alias_model=alias_model,
                    request=request,
                    request_body=request_body,
                    selection=selection,
                    candidate=skipped_candidate,
                    event_type=event_type,
                    candidate_status=f"skipped_{reason}",
                    selected=False,
                    skipped=True,
                    selection_reason=reason,
                    lane_key=skipped_candidate.get("lane_key"),
                    cooldown_seconds=skipped_candidate.get("cooldown_seconds"),
                    failure_phase=skipped_candidate.get("failure_phase"),
                    attempted_provider_call=skipped_candidate.get("attempted_provider_call"),
                )
            )

    audit_attempts = attempts
    if not audit_attempts and isinstance(selection.get("candidate"), dict):
        audit_attempts = [
            _codex_auto_agent_candidate_public_shape(
                selection["candidate"],
                lane_key=selection.get("lane_key"),
                reason=selection.get("selection_reason"),
            )
        ]

    for index, attempt in enumerate(audit_attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        status = str(attempt.get("status") or "").strip()
        failure_class = attempt.get("error_class")
        redispatch_required = status == "terminal_in_flight_cooldown_set"
        if redispatch_required:
            event_type = "redispatch_required"
        elif failure_class or status == "cooldown_set":
            event_type = "candidate_retryable_failure"
        else:
            event_type = "candidate_selected"
        events.append(
            _build_auto_agent_alias_audit_event(
                alias_family=alias_family,
                alias_model=alias_model,
                request=request,
                request_body=request_body,
                selection=selection,
                candidate=attempt,
                event_type=event_type,
                candidate_status=status or "selected",
                attempt_number=index,
                selected=True,
                skipped=False,
                selection_reason=attempt.get("reason") or selection.get("selection_reason"),
                lane_key=attempt.get("lane_key") or selection.get("lane_key"),
                # RR-054 #51: attach the attempt's own cooldown key (fall back to selection).
                cooldown_key=(
                    attempt.get("cooldown_key") or selection.get("cooldown_key")
                    if index == len(audit_attempts)
                    else attempt.get("cooldown_key")
                ),
                cooldown_seconds=attempt.get("cooldown_seconds"),
                cooldown_scope=attempt.get("cooldown_scope"),
                failure_class=failure_class,
                error_status_code=attempt.get("error_status_code"),
                error_type=attempt.get("error_type"),
                error_code=attempt.get("error_code"),
                error_tokens=attempt.get("error_tokens"),
                source_error=attempt.get("source_error"),
                retry_after_seconds=attempt.get("retry_after_seconds"),
                failure_phase=attempt.get("failure_phase"),
                attempted_provider_call=attempt.get("attempted_provider_call"),
                redispatch_required=redispatch_required,
            )
        )
    return events


def _codex_auto_agent_request_has_continuation_state(
    value: Any,
    _seen: Optional[set[int]] = None,
) -> bool:
    if isinstance(value, (dict, list)):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return False
        _seen.add(value_id)

    if isinstance(value, dict):
        for key in (
            "previous_response_id",
            "call_id",
            "tool_call_id",
            "item_id",
        ):
            if value.get(key):
                return True
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type in {
            "function_call",
            "function_call_output",
            "mcp_call",
            "mcp_approval_request",
            "mcp_approval_response",
            "reasoning",
            "tool_use",
            "tool_result",
        }:
            return True
        if value.get("role") == "tool":
            return True
        if value.get("tool_calls"):
            return True
        return any(_codex_auto_agent_request_has_continuation_state(child, _seen) for child in value.values())
    if isinstance(value, list):
        return any(_codex_auto_agent_request_has_continuation_state(item, _seen) for item in value)
    return False


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5D)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_is_auto_agent_alias_in_flight_cooldown_http_exception",
    "_build_auto_agent_alias_audit_event",
    "_build_auto_agent_alias_audit_events",
    "_codex_auto_agent_request_has_continuation_state",
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
        ("_get_auto_agent_alias_request_context", _get_auto_agent_alias_request_context),
        ("_attach_auto_agent_alias_terminal_context_fields", _attach_auto_agent_alias_terminal_context_fields),
        ("_format_auto_agent_alias_timestamp", _format_auto_agent_alias_timestamp),
        ("_extract_auto_agent_alias_metadata_value", _extract_auto_agent_alias_metadata_value),
        ("_extract_auto_agent_alias_incoming_endpoint", _extract_auto_agent_alias_incoming_endpoint),
        ("_resolve_auto_agent_alias_route_rollup_outgoing_target", _resolve_auto_agent_alias_route_rollup_outgoing_target),
        ("_auto_agent_alias_int", _auto_agent_alias_int),
        ("_auto_agent_alias_cooldown_until", _auto_agent_alias_cooldown_until),
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
    "_get_auto_agent_alias_request_context",
    "_attach_auto_agent_alias_terminal_context_fields",
    "_format_auto_agent_alias_timestamp",
    "_extract_auto_agent_alias_metadata_value",
    "_extract_auto_agent_alias_incoming_endpoint",
    "_resolve_auto_agent_alias_route_rollup_outgoing_target",
    "_auto_agent_alias_int",
    "_auto_agent_alias_cooldown_until",
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
