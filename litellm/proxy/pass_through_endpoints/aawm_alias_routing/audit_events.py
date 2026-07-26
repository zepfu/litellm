"""Audit-events: cross-module terminal-event orchestration.

Wave 5D integrator-owned module.  Owns:
- ``_enrich_auto_agent_alias_terminal_event_from_attempts``
- ``_emit_auto_agent_alias_no_candidate_event``

Dependencies on the god module and sibling Wave 5D modules are injected via
:func:`configure_audit_events_runtime`.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from fastapi import HTTPException, Request

from litellm._logging import verbose_proxy_logger

# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / sibling-module dependencies)
# ---------------------------------------------------------------------------

_get_auto_agent_alias_request_context: Optional[
    Callable[..., Mapping[str, Any]]
] = None
_attach_auto_agent_alias_terminal_context_fields: Optional[Callable[..., Any]] = None
_format_auto_agent_alias_timestamp: Optional[Callable[[datetime], str]] = None
_extract_auto_agent_alias_metadata_value: Optional[Callable[..., Optional[str]]] = None
_extract_auto_agent_alias_incoming_endpoint: Optional[Callable[..., str]] = None
_resolve_codex_auto_agent_session_key: Optional[
    Callable[..., Optional[str]]
] = None
_resolve_anthropic_auto_agent_session_key: Optional[
    Callable[..., Optional[str]]
] = None
_emit_auto_agent_alias_route_event: Optional[Callable[..., None]] = None
_build_auto_agent_alias_audit_events: Optional[Callable[..., list[dict[str, Any]]]] = None
_persist_auto_agent_alias_audit_only_events_best_effort: Optional[Callable[..., str]] = None

_host_globals: Optional[dict] = None


def configure_audit_events_runtime(
    *,
    get_request_context: Callable[..., Mapping[str, Any]],
    attach_terminal_context_fields: Callable[..., Any],
    format_timestamp: Callable[[datetime], str],
    extract_metadata_value: Callable[..., Optional[str]],
    extract_incoming_endpoint: Callable[..., str],
    resolve_codex_session_key: Callable[..., Optional[str]],
    resolve_anthropic_session_key: Callable[..., Optional[str]],
    emit_route_event: Callable[..., None],
    build_audit_events: Callable[..., list[dict[str, Any]]],
    persist_audit_only_events: Callable[..., str],
) -> None:
    """Inject god-module dependencies for audit_events orchestration functions."""
    global _get_auto_agent_alias_request_context
    global _attach_auto_agent_alias_terminal_context_fields
    global _format_auto_agent_alias_timestamp
    global _extract_auto_agent_alias_metadata_value
    global _extract_auto_agent_alias_incoming_endpoint
    global _resolve_codex_auto_agent_session_key
    global _resolve_anthropic_auto_agent_session_key
    global _emit_auto_agent_alias_route_event
    global _build_auto_agent_alias_audit_events
    global _persist_auto_agent_alias_audit_only_events_best_effort

    _get_auto_agent_alias_request_context = get_request_context
    _attach_auto_agent_alias_terminal_context_fields = attach_terminal_context_fields
    _format_auto_agent_alias_timestamp = format_timestamp
    _extract_auto_agent_alias_metadata_value = extract_metadata_value
    _extract_auto_agent_alias_incoming_endpoint = extract_incoming_endpoint
    _resolve_codex_auto_agent_session_key = resolve_codex_session_key
    _resolve_anthropic_auto_agent_session_key = resolve_anthropic_session_key
    _emit_auto_agent_alias_route_event = emit_route_event
    _build_auto_agent_alias_audit_events = build_audit_events
    _persist_auto_agent_alias_audit_only_events_best_effort = persist_audit_only_events

    if _host_globals is not None:
        _host_globals["_get_auto_agent_alias_request_context"] = (
            _get_auto_agent_alias_request_context
        )
        _host_globals["_attach_auto_agent_alias_terminal_context_fields"] = (
            _attach_auto_agent_alias_terminal_context_fields
        )
        _host_globals["_format_auto_agent_alias_timestamp"] = (
            _format_auto_agent_alias_timestamp
        )
        _host_globals["_extract_auto_agent_alias_metadata_value"] = (
            _extract_auto_agent_alias_metadata_value
        )
        _host_globals["_extract_auto_agent_alias_incoming_endpoint"] = (
            _extract_auto_agent_alias_incoming_endpoint
        )
        _host_globals["_resolve_codex_auto_agent_session_key"] = (
            _resolve_codex_auto_agent_session_key
        )
        _host_globals["_resolve_anthropic_auto_agent_session_key"] = (
            _resolve_anthropic_auto_agent_session_key
        )
        _host_globals["_emit_auto_agent_alias_route_event"] = (
            _emit_auto_agent_alias_route_event
        )
        _host_globals["_build_auto_agent_alias_audit_events"] = (
            _build_auto_agent_alias_audit_events
        )
        _host_globals["_persist_auto_agent_alias_audit_only_events_best_effort"] = (
            _persist_auto_agent_alias_audit_only_events_best_effort
        )


# ---------------------------------------------------------------------------
# Integrator-owned functions (frozen against baseline 66963d07ce)
# ---------------------------------------------------------------------------


def _enrich_auto_agent_alias_terminal_event_from_attempts(
    event: dict[str, Any],
    attempts: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    normalized_attempts = [attempt for attempt in attempts or [] if isinstance(attempt, dict)]
    if not normalized_attempts:
        return normalized_attempts

    event["attempt_count"] = len(normalized_attempts)
    event["attempts"] = copy.deepcopy(normalized_attempts)
    last_attempt = normalized_attempts[-1]
    last_failure_class = last_attempt.get("error_class")
    if last_failure_class is not None:
        event["failure_class"] = last_failure_class
    event["attempted_provider_call"] = any(
        attempt.get("attempted_provider_call") is True for attempt in normalized_attempts
    )
    if event["attempted_provider_call"]:
        event["failure_phase"] = "provider_attempt"
    for key in (
        "provider",
        "model",
        "route_family",
        "source_error",
        "error_type",
        "error_code",
    ):
        value = last_attempt.get(key)
        if value is not None:
            event[key] = value
    return normalized_attempts


def _emit_auto_agent_alias_no_candidate_event(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    exc: HTTPException,
    attempts: Optional[list[dict[str, Any]]] = None,
) -> None:
    assert _get_auto_agent_alias_request_context is not None
    assert _attach_auto_agent_alias_terminal_context_fields is not None
    assert _format_auto_agent_alias_timestamp is not None
    assert _extract_auto_agent_alias_metadata_value is not None
    assert _extract_auto_agent_alias_incoming_endpoint is not None
    assert _resolve_codex_auto_agent_session_key is not None
    assert _resolve_anthropic_auto_agent_session_key is not None
    assert _emit_auto_agent_alias_route_event is not None
    assert _build_auto_agent_alias_audit_events is not None
    assert _persist_auto_agent_alias_audit_only_events_best_effort is not None

    detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
    candidates = detail.get("candidates") if isinstance(detail, dict) else None
    context = _get_auto_agent_alias_request_context(
        request,
        request_body,
        include_activity=True,
    )
    repository = context.get("repository")
    client_product_label = context.get("client_product_label")
    host_attribution = context["host_attribution"]
    if alias_family.startswith("anthropic"):
        session_key = _resolve_anthropic_auto_agent_session_key(
            request,
            request_body,
            alias_model=alias_model,
        )
    else:
        session_key = _resolve_codex_auto_agent_session_key(
            request,
            request_body,
            alias_model=alias_model,
        )
    event = {
        "observed_at": _format_auto_agent_alias_timestamp(datetime.now(timezone.utc)),
        "alias_family": alias_family,
        "alias_model": alias_model,
        "session_id": context.get("session_id"),
        "session_key": session_key,
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
        "outgoing_target": "candidate_selection",
        "event_type": "no_candidate_available",
        "candidate_status": "all_candidates_unavailable",
        "failure_phase": "candidate_selection",
        "attempted_provider_call": False,
        "error_status_code": exc.status_code,
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
        "candidates": candidates if isinstance(candidates, list) else None,
    }
    _attach_auto_agent_alias_terminal_context_fields(
        event,
        request=request,
        request_body=request_body,
        include_activity_status=True,
    )
    normalized_attempts = _enrich_auto_agent_alias_terminal_event_from_attempts(
        event,
        attempts,
    )
    event["terminal_outcome"] = "agent_session_terminated"
    event["fallback_result"] = "no_candidate_available"
    event["redispatch_required"] = False
    event["agent_session_killed"] = True
    try:
        from functools import partial

        from litellm.proxy.aawm_runtime_error_logging import (
            persist_agent_terminal_error,
        )

        last_attempt = normalized_attempts[-1] if normalized_attempts else {}
        persist_terminal_error = partial(
            persist_agent_terminal_error,
            error_context={
                **event,
                "endpoint": event.get("incoming_endpoint"),
                "status_code": event.get("error_status_code"),
                "model_alias": alias_model,
                "provider": last_attempt.get("provider"),
                "model": last_attempt.get("model"),
                "route_family": last_attempt.get("route_family"),
                "failure_kind": "agent_alias_no_candidate",
                "error_code": event.get("failure_class") or "all_candidates_unavailable",
            },
            terminal_outcome="agent_session_terminated",
            fallback_result="no_candidate_available",
            redispatch_required=False,
            agent_session_killed=True,
        )
        # RR-054 #42: avoid blocking the event loop on JSONL intake I/O.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.run_in_executor(
                None,
                persist_terminal_error,
            )
        else:
            persist_terminal_error()
    except Exception:
        verbose_proxy_logger.debug(
            "Failed to append terminal alias error intake",
            exc_info=True,
        )
    _emit_auto_agent_alias_route_event(
        event,
        level="warning",
    )
    # Terminal no-candidate outcomes never complete a normal provider write path.
    # Persist audit rows only so partial-activity vs no-op failures remain queryable.
    audit_events: list[dict[str, Any]] = []
    if normalized_attempts:
        last_attempt = normalized_attempts[-1]
        audit_events.extend(
            _build_auto_agent_alias_audit_events(
                alias_family=alias_family,
                alias_model=alias_model,
                request=request,
                request_body=request_body,
                selection={
                    "candidate": last_attempt,
                    "session_key": event.get("session_key"),
                    "lane_key": last_attempt.get("lane_key"),
                    "selection_reason": last_attempt.get("reason"),
                    "skipped": [],
                },
                attempts=normalized_attempts,
            )
        )
    audit_events.append(event)
    _persist_auto_agent_alias_audit_only_events_best_effort(
        audit_events,
        request_body=request_body,
    )


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5D)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_enrich_auto_agent_alias_terminal_event_from_attempts",
    "_emit_auto_agent_alias_no_candidate_event",
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
        ("_resolve_codex_auto_agent_session_key", _resolve_codex_auto_agent_session_key),
        ("_resolve_anthropic_auto_agent_session_key", _resolve_anthropic_auto_agent_session_key),
        ("_emit_auto_agent_alias_route_event", _emit_auto_agent_alias_route_event),
        ("_build_auto_agent_alias_audit_events", _build_auto_agent_alias_audit_events),
        ("_persist_auto_agent_alias_audit_only_events_best_effort", _persist_auto_agent_alias_audit_only_events_best_effort),
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
    "_resolve_codex_auto_agent_session_key",
    "_resolve_anthropic_auto_agent_session_key",
    "_emit_auto_agent_alias_route_event",
    "_build_auto_agent_alias_audit_events",
    "_persist_auto_agent_alias_audit_only_events_best_effort",
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
