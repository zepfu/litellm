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
# Default to the owner-concrete implementation from audit_build (D1-591).
from .audit_build import (
    _format_auto_agent_alias_timestamp as _default_format_timestamp,
)
from .selection import _build_auto_agent_terminal_candidate_inventory

_format_auto_agent_alias_timestamp: Callable[[datetime], str] = _default_format_timestamp
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
_MISSING = object()
_runtime_restore_stacks: dict[str, list[tuple[object, object, object]]] = {}


def _update_host_runtime_callbacks(
    callbacks: Mapping[str, object],
    previous_module_values: Mapping[str, object],
) -> None:
    if _host_globals is None:
        return
    for name, callback in callbacks.items():
        _runtime_restore_stacks.setdefault(name, []).append(
            (
                callback,
                previous_module_values[name],
                _host_globals.get(name, _MISSING),
            )
        )
        _host_globals[name] = callback


def configure_audit_events_runtime(
    *,
    get_request_context: Callable[..., Mapping[str, Any]],
    attach_terminal_context_fields: Callable[..., Any],
    format_timestamp: Optional[Callable[[datetime], str]] = None,
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

    previous_module_values = {
        "_get_auto_agent_alias_request_context": _get_auto_agent_alias_request_context,
        "_attach_auto_agent_alias_terminal_context_fields": _attach_auto_agent_alias_terminal_context_fields,
        "_format_auto_agent_alias_timestamp": _format_auto_agent_alias_timestamp,
        "_extract_auto_agent_alias_metadata_value": _extract_auto_agent_alias_metadata_value,
        "_extract_auto_agent_alias_incoming_endpoint": _extract_auto_agent_alias_incoming_endpoint,
        "_resolve_codex_auto_agent_session_key": _resolve_codex_auto_agent_session_key,
        "_resolve_anthropic_auto_agent_session_key": _resolve_anthropic_auto_agent_session_key,
        "_emit_auto_agent_alias_route_event": _emit_auto_agent_alias_route_event,
        "_build_auto_agent_alias_audit_events": _build_auto_agent_alias_audit_events,
        "_persist_auto_agent_alias_audit_only_events_best_effort": _persist_auto_agent_alias_audit_only_events_best_effort,
    }
    _get_auto_agent_alias_request_context = get_request_context
    _attach_auto_agent_alias_terminal_context_fields = attach_terminal_context_fields
    if format_timestamp is not None:
        _format_auto_agent_alias_timestamp = format_timestamp
    _extract_auto_agent_alias_metadata_value = extract_metadata_value
    _extract_auto_agent_alias_incoming_endpoint = extract_incoming_endpoint
    _resolve_codex_auto_agent_session_key = resolve_codex_session_key
    _resolve_anthropic_auto_agent_session_key = resolve_anthropic_session_key
    _emit_auto_agent_alias_route_event = emit_route_event
    _build_auto_agent_alias_audit_events = build_audit_events
    _persist_auto_agent_alias_audit_only_events_best_effort = persist_audit_only_events

    _update_host_runtime_callbacks(
        {
            "_get_auto_agent_alias_request_context": _get_auto_agent_alias_request_context,
            "_attach_auto_agent_alias_terminal_context_fields": _attach_auto_agent_alias_terminal_context_fields,
            "_format_auto_agent_alias_timestamp": _format_auto_agent_alias_timestamp,
            "_extract_auto_agent_alias_metadata_value": _extract_auto_agent_alias_metadata_value,
            "_extract_auto_agent_alias_incoming_endpoint": _extract_auto_agent_alias_incoming_endpoint,
            "_resolve_codex_auto_agent_session_key": _resolve_codex_auto_agent_session_key,
            "_resolve_anthropic_auto_agent_session_key": _resolve_anthropic_auto_agent_session_key,
            "_emit_auto_agent_alias_route_event": _emit_auto_agent_alias_route_event,
            "_build_auto_agent_alias_audit_events": _build_auto_agent_alias_audit_events,
            "_persist_auto_agent_alias_audit_only_events_best_effort": _persist_auto_agent_alias_audit_only_events_best_effort,
        },
        previous_module_values,
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


def _resolve_auto_agent_alias_terminal_candidates(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    candidates: Any,
    attempts: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ingress = "anthropic" if alias_family.startswith("anthropic") else "codex"
    return _build_auto_agent_terminal_candidate_inventory(
        request=request,
        alias_model=alias_model,
        ingress=ingress,
        attempts=attempts,
        skipped_candidates=candidates if isinstance(candidates, list) else None,
    )


def _emit_auto_agent_alias_pre_attempt_terminal_event(  # noqa: PLR0915
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    event_type: str,
    candidate_status: str,
    failure_phase: str,
    error_status_code: int,
    error_code: Optional[Any] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
    attempts: Optional[list[dict[str, Any]]] = None,
    detail: Any = None,
    failure_class: Optional[str] = None,
    error_type: Optional[str] = None,
    redispatch_required: bool = False,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit and persist a terminal event that stopped before provider I/O."""
    assert _build_auto_agent_alias_audit_events is not None
    assert _emit_auto_agent_alias_route_event is not None
    assert _persist_auto_agent_alias_audit_only_events_best_effort is not None

    try:
        detail_mapping = detail if isinstance(detail, Mapping) else {}
        detail_error = detail_mapping.get("error")
        if not isinstance(detail_error, Mapping):
            detail_error = {}
        if error_code is None:
            error_code = detail_error.get("code")
        if error_type is None:
            error_type = detail_error.get("type")

        terminal_candidate = dict(candidate or {})
        detail_candidate = detail_mapping.get("candidate")
        if not terminal_candidate and isinstance(detail_candidate, Mapping):
            terminal_candidate = dict(detail_candidate)
        for key in (
            "provider",
            "model",
            "route_family",
            "lane_key",
            "cooldown_key",
            "cooldown_seconds",
            "cooldown_scope",
            "retry_after_seconds",
        ):
            if terminal_candidate.get(key) is None:
                value = detail_mapping.get(key)
                if value is None:
                    value = detail_error.get(key)
                if value is not None:
                    terminal_candidate[key] = value

        terminal_selection = dict(selection or {})
        for key in ("session_key", "lane_key", "cooldown_key", "in_flight_session"):
            if terminal_selection.get(key) is None:
                value = detail_mapping.get(key)
                if value is not None:
                    terminal_selection[key] = value
        terminal_selection["candidate"] = terminal_candidate

        normalized_attempts = [
            copy.deepcopy(attempt)
            for attempt in attempts or []
            if isinstance(attempt, dict)
        ]
        terminal_attempt = dict(terminal_candidate)
        terminal_attempt.update(
            {
                "status": candidate_status,
                "attempted_provider_call": False,
                "error_status_code": error_status_code,
                "failure_phase": failure_phase,
            }
        )
        if failure_class is not None:
            terminal_attempt["error_class"] = failure_class
        if error_type is not None:
            terminal_attempt["error_type"] = error_type
        if error_code is not None:
            terminal_attempt["error_code"] = str(error_code)
        if extra_fields:
            terminal_attempt.update(extra_fields)

        matching_keys = tuple(
            key
            for key in ("provider", "model", "route_family", "lane_key")
            if terminal_candidate.get(key) is not None
        )
        if matching_keys and normalized_attempts and all(
            normalized_attempts[-1].get(key) == terminal_candidate.get(key)
            for key in matching_keys
        ):
            normalized_attempts[-1].update(terminal_attempt)
        else:
            normalized_attempts.append(terminal_attempt)

        audit_events = _build_auto_agent_alias_audit_events(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=request_body,
            selection=terminal_selection,
            attempts=normalized_attempts,
        )
        event = copy.deepcopy(audit_events[-1]) if audit_events else {}
        event.update(
            {
                "event_type": event_type,
                "candidate_status": candidate_status,
                "failure_phase": failure_phase,
                "error_status_code": int(error_status_code),
                "attempted_provider_call": False,
                "redispatch_required": bool(redispatch_required),
                "terminal_outcome": (
                    "redispatch_required" if redispatch_required else "failed"
                ),
                "fallback_result": event_type,
                "attempt_count": len(normalized_attempts),
                "attempts": copy.deepcopy(normalized_attempts),
            }
        )
        if failure_class is not None:
            event["failure_class"] = failure_class
        if error_type is not None:
            event["error_type"] = error_type
        if error_code is not None:
            event["error_code"] = str(error_code)
        if extra_fields:
            event.update(copy.deepcopy(dict(extra_fields)))

        _emit_auto_agent_alias_route_event(event, level="warning")
        _persist_auto_agent_alias_audit_only_events_best_effort(
            [*audit_events[:-1], event],
            request_body=request_body,
        )
    except Exception:
        # Observability must never replace the original terminal response.
        verbose_proxy_logger.debug(
            "Failed to emit pre-attempt terminal alias audit event",
            exc_info=True,
        )


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
    assert _extract_auto_agent_alias_metadata_value is not None
    assert _extract_auto_agent_alias_incoming_endpoint is not None
    assert _resolve_codex_auto_agent_session_key is not None
    assert _resolve_anthropic_auto_agent_session_key is not None
    assert _emit_auto_agent_alias_route_event is not None
    assert _build_auto_agent_alias_audit_events is not None
    assert _persist_auto_agent_alias_audit_only_events_best_effort is not None

    detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
    candidates = detail.get("candidates") if isinstance(detail, dict) else None
    terminal_candidates = _resolve_auto_agent_alias_terminal_candidates(
        alias_family=alias_family,
        alias_model=alias_model,
        request=request,
        candidates=candidates,
        attempts=(
            normalized_attempts := [
                {
                    key: value
                    for key, value in attempt.items()
                    if key != "kimi_code_failure" or isinstance(value, Mapping)
                }
                for attempt in attempts or []
                if isinstance(attempt, dict)
            ]
        ),
    )
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
        "candidate_count": len(terminal_candidates),
        "candidates": terminal_candidates,
    }
    _attach_auto_agent_alias_terminal_context_fields(
        event,
        request=request,
        request_body=request_body,
        include_activity_status=True,
    )
    normalized_attempts = _enrich_auto_agent_alias_terminal_event_from_attempts(
        event,
        normalized_attempts,
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
            _ = loop.run_in_executor(
                None,
                persist_terminal_error,
            )
        else:
            _ = persist_terminal_error()
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
        terminal_skipped = [
            candidate
            for candidate in terminal_candidates
            if candidate.get("terminal_disposition") == "skipped"
        ]
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
                    "skipped": terminal_skipped,
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
    "_resolve_auto_agent_alias_terminal_candidates",
    "_emit_auto_agent_alias_pre_attempt_terminal_event",
    "_emit_auto_agent_alias_no_candidate_event",
)


def _host_callback_delegates_to_module(
    name: str,
    callback: object,
    owner_module: object,
) -> bool:
    code = getattr(callback, "__code__", None)
    callback_globals = getattr(callback, "__globals__", None)
    if code is None or not isinstance(callback_globals, dict):
        return False

    owner_callback = getattr(owner_module, name, _MISSING)
    if callback is owner_callback:
        return False
    if (
        callback_globals.get(name) is callback
        and getattr(callback, "__name__", None) != name
    ):
        return True

    referenced_values = [
        callback_globals.get(global_name, _MISSING)
        for global_name in code.co_names
    ]
    closure_values = []
    for cell in getattr(callback, "__closure__", None) or ():
        try:
            closure_values.append(cell.cell_contents)
        except ValueError:
            continue

    if any(value is owner_callback for value in (*referenced_values, *closure_values)):
        return True
    references_seam = name in code.co_names or name in code.co_consts
    return references_seam and any(
        value is owner_module for value in (*referenced_values, *closure_values)
    )


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    global _host_globals
    _mod = globals()
    owner_module = _sys.modules[__name__]
    for _name in _SEAM_NAMES:
        host_callback = host_globals.get(_name, _MISSING)
        if host_callback is _MISSING or _host_callback_delegates_to_module(
            _name,
            host_callback,
            owner_module,
        ):
            continue
        _mod[_name] = host_callback
    _host_globals = host_globals
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
        ("_build_auto_agent_terminal_candidate_inventory", _build_auto_agent_terminal_candidate_inventory),
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
                restore_stacks = self.__dict__.get("_runtime_restore_stacks", {})
                restore_stack = restore_stacks.get(name)
                if restore_stack and value is restore_stack[-1][1]:
                    _, _, prior_host_value = restore_stack.pop()
                    if prior_host_value is self.__dict__["_MISSING"]:
                        hg.pop(name, None)
                    else:
                        hg[name] = prior_host_value
                    return
                hg[name] = value


_sys.modules[__name__].__class__ = _SeamPropagatingModule
