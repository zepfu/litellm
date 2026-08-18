"""Audit-persistence: route-event emission filtering and audit-only spool/enqueue.

Wave 5D extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module are injected via :func:`configure_audit_persist_runtime`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable, Optional

from litellm._logging import verbose_aawm_route_logger, verbose_proxy_logger

# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / host)
# ---------------------------------------------------------------------------

_record_auto_agent_alias_route_status_rollup: Optional[Callable[..., None]] = None

# Owner-concrete implementations (D1-591).  The god-module previously defined
# these inline and injected them via configure_audit_persist_runtime.  A later
# integrator can remove the ~23 god-module lines and either pass these through
# configure or omit the parameters entirely.

_AAWM_ALIAS_ROUTE_VERBOSE_JSON_ENV = "AAWM_ALIAS_ROUTE_VERBOSE_JSON"


def _aawm_alias_route_verbose_json_enabled() -> bool:
    """Whether verbose JSON route logging is enabled via env."""
    return os.getenv(_AAWM_ALIAS_ROUTE_VERBOSE_JSON_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "debug",
        "verbose",
    }


def _aawm_alias_route_healthy_json_enabled() -> bool:
    """Whether healthy-route JSON logging is enabled via env."""
    return os.getenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


# Seam variables: default to the owner-concrete functions above.
# configure_audit_persist_runtime() can still override them for testing.
_aawm_alias_route_verbose_json_enabled: Callable[[], bool] = _aawm_alias_route_verbose_json_enabled
_aawm_alias_route_healthy_json_enabled: Callable[[], bool] = _aawm_alias_route_healthy_json_enabled

_host_globals: Optional[dict] = None
_MISSING = object()
_runtime_restore_stacks: dict[str, list[tuple[object, object, object]]] = {}


def _update_host_runtime_callbacks(
    callbacks: dict[str, object],
    previous_module_values: dict[str, object],
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


def configure_audit_persist_runtime(
    *,
    record_route_status_rollup: Callable[..., None],
    verbose_json_enabled: Optional[Callable[[], bool]] = None,
    healthy_json_enabled: Optional[Callable[[], bool]] = None,
) -> None:
    """Inject god-module dependencies.  Must be called before any frozen function."""
    global _record_auto_agent_alias_route_status_rollup
    global _aawm_alias_route_verbose_json_enabled
    global _aawm_alias_route_healthy_json_enabled

    previous_module_values = {
        "_record_auto_agent_alias_route_status_rollup": _record_auto_agent_alias_route_status_rollup,
        "_aawm_alias_route_verbose_json_enabled": _aawm_alias_route_verbose_json_enabled,
        "_aawm_alias_route_healthy_json_enabled": _aawm_alias_route_healthy_json_enabled,
    }
    _record_auto_agent_alias_route_status_rollup = record_route_status_rollup
    if verbose_json_enabled is not None:
        _aawm_alias_route_verbose_json_enabled = verbose_json_enabled
    if healthy_json_enabled is not None:
        _aawm_alias_route_healthy_json_enabled = healthy_json_enabled

    _update_host_runtime_callbacks(
        {
            "_record_auto_agent_alias_route_status_rollup": _record_auto_agent_alias_route_status_rollup,
            "_aawm_alias_route_verbose_json_enabled": _aawm_alias_route_verbose_json_enabled,
            "_aawm_alias_route_healthy_json_enabled": _aawm_alias_route_healthy_json_enabled,
        },
        previous_module_values,
    )


# ---------------------------------------------------------------------------
# Frozen functions (baseline 66963d07ce)
# ---------------------------------------------------------------------------


def _emit_auto_agent_alias_route_event(
    event: dict[str, Any],
    *,
    level: str = "info",
) -> None:
    assert _record_auto_agent_alias_route_status_rollup is not None

    _record_auto_agent_alias_route_status_rollup(event)
    if not (_aawm_alias_route_verbose_json_enabled() or _aawm_alias_route_healthy_json_enabled()):
        return
    if not _should_emit_auto_agent_alias_route_event(event, level=level):
        return
    log_payload = {"event": "aawm_alias_route", **event}
    message = "AAWM_ALIAS_ROUTE: {}".format(json.dumps(log_payload, sort_keys=True, default=str, separators=(",", ":")))
    verbose_aawm_route_logger.info(message)


def _should_emit_auto_agent_alias_route_event(
    event: dict[str, Any],
    *,
    level: str = "info",
) -> bool:
    if level == "warning":
        return True

    assert _aawm_alias_route_healthy_json_enabled is not None
    if _aawm_alias_route_healthy_json_enabled():
        return True

    if event.get("failure_class") or event.get("error_status_code"):
        return True
    if event.get("redispatch_required") or event.get("redispatch_threshold_crossed"):
        return True

    event_type = str(event.get("event_type") or "")
    candidate_status = str(event.get("candidate_status") or "")
    if event_type in {
        "candidate_attempt_started",
        "candidate_selected",
    }:
        return False
    if candidate_status in {"started", "selected"}:
        return False
    if event.get("selection_reason") == "session_affinity":
        return False
    return True


def _persist_auto_agent_alias_audit_only_events_best_effort(  # noqa: PLR0915
    events: list[dict[str, Any]],
    *,
    request_body: Optional[dict[str, Any]] = None,
) -> str:
    """Best-effort audit-only persistence without session_history inserts.

    Used for terminal/no-candidate Codex+Anthropic alias events that never reach
    a normal success/fallback write path. Failures are swallowed so routing is
    never blocked by observability.
    """
    if not events:
        return "skip_empty"

    max_event_types_for_log = 8

    def _sanitize_identifier(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "<missing>"
        try:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        except Exception:
            digest = "<redacted>"
        return digest

    def _sanitize_log_label(value: Any, *, max_length: int) -> str:
        if not isinstance(value, str) or not value.strip():
            return "<missing>"
        sanitized = re.sub(
            r"[^A-Za-z0-9_.:/-]+",
            "_",
            value.strip(),
        )
        return sanitized[:max_length] or "<missing>"

    def _collect_event_types() -> str:
        event_types: list[str] = []
        omitted_count = 0
        for event in events:
            event_type = event.get("event_type") if isinstance(event, dict) else None
            if not isinstance(event_type, str):
                continue
            sanitized = _sanitize_log_label(event_type, max_length=64)
            if sanitized in event_types:
                continue
            if len(event_types) >= max_event_types_for_log:
                omitted_count += 1
                continue
            event_types.append(sanitized)
        if omitted_count:
            event_types.append(f"+{omitted_count}_more")
        return f"[{','.join(event_types)}]"

    primary = events[-1] if isinstance(events[-1], dict) else {}
    metadata: dict[str, Any] = {
        "aawm_alias_routing_audit_only": True,
        "source": "auto_agent_alias_terminal_or_no_candidate",
    }
    if isinstance(request_body, dict):
        litellm_metadata = request_body.get("litellm_metadata")
        if isinstance(litellm_metadata, dict):
            for key in (
                "requested_model_alias",
                "model_alias_label",
                "repository",
            ):
                value = litellm_metadata.get(key)
                if value is not None:
                    metadata[key] = value

    event_types = _collect_event_types()
    alias = _sanitize_log_label(
        primary.get("alias_model")
        or primary.get("requested_model_alias")
        or metadata.get("requested_model_alias")
        or metadata.get("model_alias_label"),
        max_length=96,
    )
    # Canonical spawned-child thread identity is the durable session key
    # when present; parent-only clients keep primary.session_id.
    durable_session_id = primary.get("canonical_thread_id") or primary.get(
        "session_id"
    )
    for key in (
        "canonical_thread_id",
        "parent_thread_id",
        "has_account_bound_state",
        "account_bound_classification",
        "account_lane",
        "failure_class",
    ):
        value = primary.get(key)
        if value is not None:
            metadata[key] = value
    session_id_hash = _sanitize_identifier(durable_session_id)
    litellm_call_id_hash = _sanitize_identifier(primary.get("litellm_call_id"))
    trace_id_hash = _sanitize_identifier(primary.get("trace_id"))

    try:
        from litellm.integrations.aawm_agent_identity import (
            _build_alias_routing_audit_only_record,
            _enqueue_session_history_record,
            _spool_session_history_records,
        )
    except Exception as import_exc:
        verbose_proxy_logger.warning(
            "AAWM_ALIAS_ROUTE: unable to import alias routing audit-only helpers "
            "(persistence_disposition=fail_import, exception_class=%s, alias=%s, "
            "session_id_hash=%s, litellm_call_id_hash=%s, event_types=%s, "
            "event_count=%d)",
            type(import_exc).__name__,
            alias,
            session_id_hash,
            litellm_call_id_hash,
            event_types,
            len(events),
        )
        return "fail_import"

    try:
        record = _build_alias_routing_audit_only_record(
            events=events,
            session_id=durable_session_id,
            litellm_call_id=primary.get("litellm_call_id"),
            model=primary.get("model") or primary.get("alias_model"),
            provider=primary.get("provider"),
            metadata=metadata,
        )
    except Exception as build_exc:
        verbose_proxy_logger.warning(
            "AAWM_ALIAS_ROUTE: failed to build alias routing audit-only record "
            "(persistence_disposition=fail_build, exception_class=%s, alias=%s, "
            "session_id_hash=%s, litellm_call_id_hash=%s, event_types=%s, "
            "event_count=%d)",
            type(build_exc).__name__,
            alias,
            session_id_hash,
            litellm_call_id_hash,
            event_types,
            len(events),
        )
        return "fail_build"

    try:
        _spool_session_history_records(
            [record],
            reason="alias audit terminal write-ahead",
        )
        return "spool_only"
    except Exception as spool_exc:
        verbose_proxy_logger.warning(
            "AAWM_ALIAS_ROUTE: failed to spool terminal alias audit event; "
            "queue_disposition=spool_failed, alias=%s, session_id_hash=%s, "
            "litellm_call_id_hash=%s, trace_id_hash=%s, event_types=%s, "
            "event_count=%d, exception_class=%s",
            alias,
            session_id_hash,
            litellm_call_id_hash,
            trace_id_hash,
            event_types,
            len(events),
            type(spool_exc).__name__,
        )
        try:
            _enqueue_session_history_record(record)
        except Exception as enqueue_exc:
            verbose_proxy_logger.warning(
                "AAWM_ALIAS_ROUTE: failed terminal alias audit fallback enqueue; "
                "queue_disposition=enqueue_failed, alias=%s, session_id_hash=%s, "
                "litellm_call_id_hash=%s, event_types=%s, event_count=%d, "
                "spool_exception_class=%s, enqueue_exception_class=%s",
                alias,
                session_id_hash,
                litellm_call_id_hash,
                event_types,
                len(events),
                type(spool_exc).__name__,
                type(enqueue_exc).__name__,
            )
            return "spool_enqueue_failed"
        return "spool_fallback_enqueue"


# ---------------------------------------------------------------------------
# Host-globals rebinding (Wave 5D)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType

_HOST_FUNCTION_NAMES = (
    "_emit_auto_agent_alias_route_event",
    "_should_emit_auto_agent_alias_route_event",
    "_persist_auto_agent_alias_audit_only_events_best_effort",
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
        ("_record_auto_agent_alias_route_status_rollup", _record_auto_agent_alias_route_status_rollup),
        ("_aawm_alias_route_verbose_json_enabled", _aawm_alias_route_verbose_json_enabled),
        ("_aawm_alias_route_healthy_json_enabled", _aawm_alias_route_healthy_json_enabled),
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
    "_record_auto_agent_alias_route_status_rollup",
    "_aawm_alias_route_verbose_json_enabled",
    "_aawm_alias_route_healthy_json_enabled",
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
