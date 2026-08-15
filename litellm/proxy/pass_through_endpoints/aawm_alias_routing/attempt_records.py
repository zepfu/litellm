"""Attempt-record mutation, reasoning-effort normalization, and alias metadata composition.

Wave 5C extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Dependencies on the god module are injected via :func:`configure_attempt_records_runtime`.
Signal helpers are consumed from ``error_signals.py`` through explicit seams.
Direct imports from sibling Wave 4/5 modules are used where those modules own
the symbols.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

from fastapi import Request

from .lane_keys import _CODEX_REASONING_EFFORT_TIER_INDEX

_AAWM_ALIAS_REQUEST_CALL_ID_STATE_KEY = "aawm_alias_request_litellm_call_id"
_AAWM_ALIAS_REQUEST_OUTCOME_STATE_KEY = "aawm_alias_request_outcome"

# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / error_signals / classification / state)
# ---------------------------------------------------------------------------

# --- error_signals.py seams ---
_extract_codex_auto_agent_error_tokens: Optional[Callable[..., set[str]]] = None
_extract_codex_auto_agent_error_type_and_code: Optional[
    Callable[..., tuple[Optional[str], Optional[str]]]
] = None
_parse_codex_auto_agent_header_wait_seconds: Optional[Callable[..., Optional[float]]] = None
_get_codex_auto_agent_source_error_summary: Optional[Callable[..., Optional[str]]] = None
_build_safe_kimi_code_selection_telemetry: Optional[Callable[..., dict[str, Any]]] = None
_extract_codex_auto_agent_usage_limit_raw_quota_resets: Optional[
    Callable[..., dict[str, float]]
] = None

# --- god-module / host seams ---
_extract_exception_status_code: Optional[Callable[..., Optional[int]]] = None
_safe_set_request_parsed_body: Optional[Callable[..., None]] = None
_emit_auto_agent_alias_route_event: Optional[Callable[..., None]] = None
_build_auto_agent_alias_audit_event: Optional[Callable[..., dict[str, Any]]] = None
_build_auto_agent_alias_audit_events: Optional[Callable[..., list[dict[str, Any]]]] = None
_persist_auto_agent_alias_audit_only_events_best_effort: Optional[Callable[..., None]] = None
# Default to owner-concrete implementations from audit_persist (D1-591).
from .audit_persist import (
    _aawm_alias_route_healthy_json_enabled as _default_healthy_json_enabled,
    _aawm_alias_route_verbose_json_enabled as _default_verbose_json_enabled,
)

_aawm_alias_route_verbose_json_enabled: Callable[[], bool] = _default_verbose_json_enabled
_aawm_alias_route_healthy_json_enabled: Callable[[], bool] = _default_healthy_json_enabled
_merge_litellm_metadata: Optional[Callable[..., dict[str, Any]]] = None
_normalize_low_cardinality_tag_value: Optional[Callable[..., Optional[str]]] = None
_load_bundled_model_cost_map_for_codex_policy: Optional[Callable[[], dict[str, Any]]] = None

# --- model catalog seams ---
_get_model_info: Optional[Callable[..., Any]] = None
_model_cost: Optional[dict[str, Any]] = None
_openai_provider_value: Optional[str] = None

# --- classification / Codex failure-evidence seams ---
_classify_failure: Optional[Callable[..., Any]] = None
_codex_failure_evidence_gate_record: Optional[Callable[..., Any]] = None


# Reference to host_globals set by install(); configure updates it too.
_host_globals_ref: dict | None = None
_MISSING = object()
_RUNTIME_STATE_NAMES = (
    "_extract_codex_auto_agent_error_tokens",
    "_extract_codex_auto_agent_error_type_and_code",
    "_parse_codex_auto_agent_header_wait_seconds",
    "_get_codex_auto_agent_source_error_summary",
    "_build_safe_kimi_code_selection_telemetry",
    "_extract_codex_auto_agent_usage_limit_raw_quota_resets",
    "_extract_exception_status_code",
    "_safe_set_request_parsed_body",
    "_emit_auto_agent_alias_route_event",
    "_build_auto_agent_alias_audit_event",
    "_build_auto_agent_alias_audit_events",
    "_persist_auto_agent_alias_audit_only_events_best_effort",
    "_aawm_alias_route_verbose_json_enabled",
    "_aawm_alias_route_healthy_json_enabled",
    "_merge_litellm_metadata",
    "_normalize_low_cardinality_tag_value",
    "_load_bundled_model_cost_map_for_codex_policy",
    "_get_model_info",
    "_model_cost",
    "_openai_provider_value",
    "_classify_failure",
    "_codex_failure_evidence_gate_record",
)
_runtime_restore_stacks: dict[str, list[tuple[object, object, object]]] = {}


def _update_host_runtime_callbacks(
    callbacks: Mapping[str, object],
    previous_module_values: Mapping[str, object],
) -> None:
    if _host_globals_ref is None:
        return
    for name, callback in callbacks.items():
        _runtime_restore_stacks.setdefault(name, []).append(
            (
                callback,
                previous_module_values[name],
                _host_globals_ref.get(name, _MISSING),
            )
        )
        _host_globals_ref[name] = callback


def configure_attempt_records_runtime(  # noqa: PLR0915
    *,
    # error_signals.py
    extract_error_tokens: Callable[..., set[str]],
    extract_error_type_and_code: Callable[..., tuple[Optional[str], Optional[str]]],
    parse_header_wait_seconds: Callable[..., Optional[float]],
    get_source_error_summary: Callable[..., Optional[str]],
    build_kimi_telemetry: Callable[..., dict[str, Any]],
    extract_usage_limit_raw_quota_resets: Optional[
        Callable[..., dict[str, float]]
    ] = None,
    # god-module / host
    extract_status_code: Callable[..., Optional[int]],
    safe_set_parsed_body: Callable[..., None],
    emit_route_event: Callable[..., None],
    build_audit_event: Callable[..., dict[str, Any]],
    build_audit_events: Callable[..., list[dict[str, Any]]],
    persist_audit_only_events: Callable[..., Any],
    verbose_json_enabled: Optional[Callable[[], bool]] = None,
    healthy_json_enabled: Optional[Callable[[], bool]] = None,
    merge_metadata: Callable[..., dict[str, Any]],
    normalize_tag_value: Callable[..., Optional[str]],
    load_bundled_model_cost: Callable[[], dict[str, Any]],
    # model catalog
    get_model_info: Callable[..., Any],
    model_cost: dict[str, Any],
    openai_provider_value: str,
    # classification / Codex failure evidence
    classify_failure: Callable[..., Any],
    codex_failure_evidence_gate_record: Callable[..., Any],


) -> None:
    """Bind god-module / error_signals / classification owned dependencies."""
    previous_module_values = {
        name: globals()[name] for name in _RUNTIME_STATE_NAMES
    }
    global _extract_codex_auto_agent_error_tokens
    _extract_codex_auto_agent_error_tokens = extract_error_tokens
    global _extract_codex_auto_agent_error_type_and_code
    _extract_codex_auto_agent_error_type_and_code = extract_error_type_and_code
    global _parse_codex_auto_agent_header_wait_seconds
    _parse_codex_auto_agent_header_wait_seconds = parse_header_wait_seconds
    global _get_codex_auto_agent_source_error_summary
    _get_codex_auto_agent_source_error_summary = get_source_error_summary
    global _build_safe_kimi_code_selection_telemetry
    _build_safe_kimi_code_selection_telemetry = build_kimi_telemetry
    global _extract_codex_auto_agent_usage_limit_raw_quota_resets
    if extract_usage_limit_raw_quota_resets is None:
        from . import error_signals as _error_signals

        _extract_codex_auto_agent_usage_limit_raw_quota_resets = (
            _error_signals._extract_codex_auto_agent_usage_limit_raw_quota_resets
        )
    else:
        _extract_codex_auto_agent_usage_limit_raw_quota_resets = (
            extract_usage_limit_raw_quota_resets
        )
    global _extract_exception_status_code
    _extract_exception_status_code = extract_status_code
    global _safe_set_request_parsed_body
    _safe_set_request_parsed_body = safe_set_parsed_body
    global _emit_auto_agent_alias_route_event
    _emit_auto_agent_alias_route_event = emit_route_event
    global _build_auto_agent_alias_audit_event
    _build_auto_agent_alias_audit_event = build_audit_event
    global _build_auto_agent_alias_audit_events
    _build_auto_agent_alias_audit_events = build_audit_events
    global _persist_auto_agent_alias_audit_only_events_best_effort
    _persist_auto_agent_alias_audit_only_events_best_effort = persist_audit_only_events
    global _aawm_alias_route_verbose_json_enabled
    if verbose_json_enabled is not None:
        _aawm_alias_route_verbose_json_enabled = verbose_json_enabled
    global _aawm_alias_route_healthy_json_enabled
    if healthy_json_enabled is not None:
        _aawm_alias_route_healthy_json_enabled = healthy_json_enabled
    global _merge_litellm_metadata
    _merge_litellm_metadata = merge_metadata
    global _normalize_low_cardinality_tag_value
    _normalize_low_cardinality_tag_value = normalize_tag_value
    global _load_bundled_model_cost_map_for_codex_policy
    _load_bundled_model_cost_map_for_codex_policy = load_bundled_model_cost
    global _get_model_info
    _get_model_info = get_model_info
    global _model_cost
    _model_cost = model_cost
    global _openai_provider_value
    _openai_provider_value = openai_provider_value
    global _classify_failure
    _classify_failure = classify_failure
    global _codex_failure_evidence_gate_record
    _codex_failure_evidence_gate_record = codex_failure_evidence_gate_record
    # If install() has been called, also update host_globals so configured
    # callbacks remain live for facades published there.
    _mod = globals()
    _update_host_runtime_callbacks(
        {name: _mod[name] for name in _RUNTIME_STATE_NAMES},
        previous_module_values,
    )


# ---------------------------------------------------------------------------
# Request-identity outcome reconciliation (OPENAI-012)
# ---------------------------------------------------------------------------


def _resolve_auto_agent_alias_request_identity(
    request: Request,
) -> Optional[str]:
    """Return the request-local call identity when one is already bound."""

    request_state = getattr(request, "state", None)
    if request_state is None:
        return None
    existing = getattr(
        request_state,
        _AAWM_ALIAS_REQUEST_CALL_ID_STATE_KEY,
        None,
    )
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return None


def _bind_auto_agent_alias_request_identity(request: Request) -> Optional[str]:
    """Bind one request-local identity without consulting interval-global state."""

    existing = _resolve_auto_agent_alias_request_identity(request)
    if existing is not None:
        return existing
    request_state = getattr(request, "state", None)
    if request_state is None:
        return None
    identity = None
    for key in ("litellm_call_id", "call_id", "request_id"):
        value = getattr(request_state, key, None)
        if isinstance(value, str) and value.strip():
            identity = value.strip()
            break
    if identity is None:
        identity = str(uuid4())
    setattr(request_state, _AAWM_ALIAS_REQUEST_CALL_ID_STATE_KEY, identity)
    return identity


def _auto_agent_alias_request_outcome_state(
    request: Request,
) -> dict[str, Any]:
    """Return the request-local outcome record; never interval-global."""

    request_state = getattr(request, "state", None)
    if request_state is None:
        return {
            "request_identity": None,
            "pending_failover": False,
            "outcome": None,
            "attempts": [],
        }
    existing = getattr(
        request_state,
        _AAWM_ALIAS_REQUEST_OUTCOME_STATE_KEY,
        None,
    )
    if isinstance(existing, dict):
        if existing.get("request_identity") is None:
            identity = _resolve_auto_agent_alias_request_identity(request)
            if identity is not None:
                existing["request_identity"] = identity
        return existing
    outcome = {
        "request_identity": _resolve_auto_agent_alias_request_identity(request),
        "pending_failover": False,
        "outcome": None,
        "attempts": [],
    }
    setattr(request_state, _AAWM_ALIAS_REQUEST_OUTCOME_STATE_KEY, outcome)
    return outcome


def _stamp_auto_agent_alias_request_identity(
    *,
    request: Request,
    target: dict[str, Any],
) -> Optional[str]:
    identity = _bind_auto_agent_alias_request_identity(request)
    if identity is None:
        identity = _auto_agent_alias_request_outcome_state(request).get(
            "request_identity"
        )
    if not isinstance(identity, str) or not identity:
        return None
    target["request_identity"] = identity
    target.setdefault("litellm_call_id", identity)
    return identity


def _mark_auto_agent_alias_request_failover_pending(
    request: Request,
    attempt_record: dict[str, Any],
) -> dict[str, Any]:
    """Remember a same-request account move without hiding the failed attempt."""

    outcome = _auto_agent_alias_request_outcome_state(request)
    identity = _stamp_auto_agent_alias_request_identity(
        request=request,
        target=attempt_record,
    )
    if identity is not None:
        outcome["request_identity"] = identity
    outcome["pending_failover"] = True
    if outcome.get("outcome") != "recovered":
        outcome["outcome"] = "pending_failover"
    attempt_record["account_failover_planned"] = True
    attempt_record["request_outcome"] = "pending_failover"
    return outcome


def _mark_auto_agent_alias_request_terminal_failure(
    request: Request,
    attempt_record: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    outcome = _auto_agent_alias_request_outcome_state(request)
    if attempt_record is not None:
        identity = _stamp_auto_agent_alias_request_identity(
            request=request,
            target=attempt_record,
        )
        if identity is not None:
            outcome["request_identity"] = identity
        attempt_record["request_outcome"] = "failed"
    outcome["pending_failover"] = False
    outcome["outcome"] = "failed"
    return outcome


def _mark_auto_agent_alias_request_recovered(
    request: Request,
    attempt_record: dict[str, Any],
) -> dict[str, Any]:
    outcome = _auto_agent_alias_request_outcome_state(request)
    recovered = bool(outcome.get("pending_failover") or outcome.get("outcome") == "recovered")
    identity = _stamp_auto_agent_alias_request_identity(
        request=request,
        target=attempt_record,
    )
    if identity is not None:
        outcome["request_identity"] = identity
    if recovered:
        outcome["pending_failover"] = False
        outcome["outcome"] = "recovered"
        attempt_record["status"] = "recovered"
        attempt_record["request_outcome"] = "recovered"
    else:
        attempt_record.setdefault("status", "succeeded")
        attempt_record["request_outcome"] = attempt_record.get("status") or "succeeded"
    return outcome


# ---------------------------------------------------------------------------
# Retryable attempt record mutation
# ---------------------------------------------------------------------------


def _update_codex_auto_agent_retryable_attempt_record(
    *,
    attempt_record: dict[str, Any],
    exc: Any,
    error_class: str,
    cooldown_seconds: float,
    alias_model: str,
    cooldown_scope: Optional[str] = None,
    candidate: Optional[dict[str, Any]] = None,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> set[str]:
    assert _extract_codex_auto_agent_error_tokens is not None
    assert _extract_exception_status_code is not None
    assert _extract_codex_auto_agent_error_type_and_code is not None
    assert _parse_codex_auto_agent_header_wait_seconds is not None
    assert _get_codex_auto_agent_source_error_summary is not None
    assert _extract_codex_auto_agent_usage_limit_raw_quota_resets is not None

    error_tokens = _extract_codex_auto_agent_error_tokens(exc)
    error_status_code = _extract_exception_status_code(exc)
    error_type, error_code = _extract_codex_auto_agent_error_type_and_code(exc)
    retry_after_seconds = _parse_codex_auto_agent_header_wait_seconds(exc)
    source_error = _get_codex_auto_agent_source_error_summary(
        exc,
        status_code=error_status_code,
    )
    update: dict[str, Any] = {
        "status": ("retryable_no_cooldown" if cooldown_scope == "none" else "cooldown_set"),
        "error_class": error_class,
        "error_tokens": sorted(error_tokens),
        "failure_phase": "provider_attempt",
        "attempted_provider_call": True,
        "source_error": source_error,
    }
    if cooldown_scope != "none":
        update["cooldown_seconds"] = round(float(cooldown_seconds), 3)
    if cooldown_scope is not None:
        update["cooldown_scope"] = cooldown_scope
    if error_status_code is not None:
        update["error_status_code"] = error_status_code
    if error_type is not None:
        update["error_type"] = error_type
    if error_code is not None:
        update["error_code"] = str(error_code)
    if retry_after_seconds is not None:
        update["retry_after_seconds"] = round(float(retry_after_seconds), 3)
    if error_class == "usage_limit_reached":
        update.update(
            _extract_codex_auto_agent_usage_limit_raw_quota_resets(exc)
        )
    if candidate is not None and kimi_failure_metadata is not None:
        assert _build_safe_kimi_code_selection_telemetry is not None
        update["kimi_code_failure"] = _build_safe_kimi_code_selection_telemetry(
            alias_model=alias_model,
            candidate=candidate,
            metadata=kimi_failure_metadata,
        )
    attempt_record.update(update)
    return error_tokens


# ---------------------------------------------------------------------------
# Attempt-start / failure records
# ---------------------------------------------------------------------------


def _record_auto_agent_alias_attempt_started(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    assert _safe_set_request_parsed_body is not None
    assert _emit_auto_agent_alias_route_event is not None

    candidate_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    _safe_set_request_parsed_body(request, candidate_body)
    candidate_metadata = candidate_body.get("litellm_metadata")
    audit_events = (
        candidate_metadata.get("aawm_alias_routing_audit_events")
        if isinstance(candidate_metadata, dict)
        else None
    )
    if (
        isinstance(audit_events, list)
        and audit_events
        and (_aawm_alias_route_verbose_json_enabled() or _aawm_alias_route_healthy_json_enabled())
    ):
        latest_event = audit_events[-1]
        if isinstance(latest_event, dict):
            _emit_auto_agent_alias_route_event(latest_event)
    return candidate_body


# ---------------------------------------------------------------------------
# Codex failure evidence (exactly-once per event)
# ---------------------------------------------------------------------------


def _record_codex_failure_evidence(
    *,
    canonical_alias: str,
    cooldown_key: str,
    exc: Any,
    attempt_record: dict[str, Any],
    cooldown_seconds: Optional[float] = None,
) -> None:
    """Classify and record the current Codex alias failure evidence.

    Called from the retry loop BEFORE the cooldown is applied for the same
    attempt, so a structured failure cools immediately (N=1) and a marker
    failure counts toward its N-of-M threshold on this attempt. The evidence
    is attached to the caller-provided configured alias and exact selected
    cooldown key. Classification inputs (status code, source-error text,
    retry-after) are resolved before the post-apply attempt record exists so
    evidence classification can run before the cooldown decision, because those
    record fields are not populated until after the cooldown decision.

    ``origin`` (upstream/client/unknown; only ``upstream`` ever advances a key
    toward cooling) is stamped on the attempt record for downstream audit.
    """
    assert _extract_exception_status_code is not None
    assert _get_codex_auto_agent_source_error_summary is not None
    assert _parse_codex_auto_agent_header_wait_seconds is not None
    assert _classify_failure is not None
    assert _codex_failure_evidence_gate_record is not None
    if not canonical_alias or canonical_alias.strip() != canonical_alias:
        raise ValueError("canonical_alias must be an explicit non-empty alias")
    if not cooldown_key:
        raise ValueError("cooldown_key must be an explicit non-empty key")

    error_status_code = _extract_exception_status_code(exc)
    source_error = _get_codex_auto_agent_source_error_summary(exc, status_code=error_status_code)
    raw_retry_after_seconds = _parse_codex_auto_agent_header_wait_seconds(exc)
    effective_retry_after_seconds = cooldown_seconds
    if effective_retry_after_seconds is None:
        effective_retry_after_seconds = raw_retry_after_seconds
    event = _classify_failure(
        status_code=error_status_code,
        provider=None,
        message=str(source_error or ""),
        retry_after_seconds=effective_retry_after_seconds,
    )
    attempt_record["origin"] = event.origin
    _codex_failure_evidence_gate_record(
        canonical_alias=canonical_alias,
        cooldown_key=cooldown_key,
        event=event,
    )


# ---------------------------------------------------------------------------
# Attempt-failure record
# ---------------------------------------------------------------------------


def _record_auto_agent_alias_attempt_failure(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    error_class: str,
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
    redispatch_required: bool = False,
) -> dict[str, Any]:
    assert _safe_set_request_parsed_body is not None
    assert _build_auto_agent_alias_audit_event is not None
    assert _emit_auto_agent_alias_route_event is not None
    assert _persist_auto_agent_alias_audit_only_events_best_effort is not None

    # Codex failure evidence is recorded in the retry loop BEFORE the cooldown
    # is applied (see ``_record_codex_failure_evidence``), so it
    # is intentionally NOT re-recorded here -- doing so would double-count
    # marker evidence and double-advance the structured attempt counter.
    failure_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    _safe_set_request_parsed_body(request, failure_body)
    failure_metadata = failure_body.get("litellm_metadata")
    full_audit_events = (
        failure_metadata.get("aawm_alias_routing_audit_events")
        if isinstance(failure_metadata, dict)
        else None
    )
    audit_events = [event for event in full_audit_events or [] if isinstance(event, dict)]
    audit_event = audit_events[-1] if audit_events else None
    if audit_event is None:
        audit_event = _build_auto_agent_alias_audit_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=prepared_request_body,
            selection=selection,
            candidate=attempt_record,
            event_type="redispatch_required" if redispatch_required else "candidate_retryable_failure",
            candidate_status=attempt_record.get("status") or "cooldown_set",
            attempt_number=len(attempts),
            selected=True,
            selection_reason=selection.get("selection_reason"),
            lane_key=selection.get("lane_key"),
            cooldown_key=selection.get("cooldown_key"),
            cooldown_seconds=attempt_record.get("cooldown_seconds"),
            cooldown_scope=attempt_record.get("cooldown_scope"),
            failure_class=error_class,
            error_status_code=attempt_record.get("error_status_code"),
            error_type=attempt_record.get("error_type"),
            error_code=attempt_record.get("error_code"),
            error_tokens=attempt_record.get("error_tokens"),
            source_error=attempt_record.get("source_error"),
            retry_after_seconds=attempt_record.get("retry_after_seconds"),
            failure_phase=attempt_record.get("failure_phase"),
            attempted_provider_call=attempt_record.get("attempted_provider_call"),
            redispatch_required=redispatch_required,
        )
        audit_events = [audit_event]
    if attempt_record.get("account_failover_planned"):
        _mark_auto_agent_alias_request_failover_pending(request, attempt_record)
        audit_event["account_failover_planned"] = True
        audit_event["request_outcome"] = "pending_failover"
    elif attempt_record.get("account_failover_limit_reached") or redispatch_required:
        _mark_auto_agent_alias_request_terminal_failure(request, attempt_record)
        if attempt_record.get("account_failover_limit_reached"):
            audit_event["account_failover_limit_reached"] = True
        audit_event["request_outcome"] = "failed"
    _stamp_auto_agent_alias_request_identity(request=request, target=audit_event)
    _emit_auto_agent_alias_route_event(
        audit_event,
        level="warning",
    )
    # Only terminal redispatch outcomes use audit-only persistence. Mid-loop
    # retryable 429s that continue failover still reach a normal success or
    # no-candidate write path and must not double-write audit rows.
    if redispatch_required:
        _persist_auto_agent_alias_audit_only_events_best_effort(
            audit_events,
            request_body=prepared_request_body,
        )
    return failure_body


def _record_auto_agent_alias_attempt_success(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Record same-request alternate-account recovery without hiding prior failures."""

    outcome = _mark_auto_agent_alias_request_recovered(request, attempt_record)
    recovered = outcome.get("outcome") == "recovered"
    success_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    if _safe_set_request_parsed_body is not None:
        _safe_set_request_parsed_body(request, success_body)
    if not recovered or _emit_auto_agent_alias_route_event is None:
        return success_body

    success_metadata = success_body.get("litellm_metadata")
    full_audit_events = (
        success_metadata.get("aawm_alias_routing_audit_events")
        if isinstance(success_metadata, dict)
        else None
    )
    audit_events = [event for event in full_audit_events or [] if isinstance(event, dict)]
    audit_event = audit_events[-1] if audit_events else None
    if audit_event is None and _build_auto_agent_alias_audit_event is not None:
        audit_event = _build_auto_agent_alias_audit_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=prepared_request_body,
            selection=selection,
            candidate=attempt_record,
            event_type="candidate_recovered",
            candidate_status=attempt_record.get("status") or "recovered",
            attempt_number=len(attempts),
            selected=True,
            selection_reason=selection.get("selection_reason"),
            lane_key=selection.get("lane_key"),
            cooldown_key=selection.get("cooldown_key"),
            attempted_provider_call=attempt_record.get("attempted_provider_call"),
        )
    if audit_event is None:
        audit_event = {
            "event_type": "candidate_recovered",
            "candidate_status": attempt_record.get("status") or "recovered",
            "alias_family": alias_family,
            "alias_model": alias_model,
            "selected": True,
            "selection_reason": selection.get("selection_reason"),
        }
    audit_event["event_type"] = "candidate_recovered"
    audit_event["candidate_status"] = attempt_record.get("status") or "recovered"
    audit_event["request_outcome"] = "recovered"
    _stamp_auto_agent_alias_request_identity(request=request, target=audit_event)
    _emit_auto_agent_alias_route_event(audit_event)
    return success_body


# ---------------------------------------------------------------------------
# Reasoning-effort extraction and normalization
# ---------------------------------------------------------------------------


def _extract_codex_reasoning_effort(
    request_body: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    reasoning = request_body.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        value = reasoning.get("effort")
        return (value if isinstance(value, str) else None), "reasoning.effort"
    if "reasoning_effort" in request_body:
        value = request_body.get("reasoning_effort")
        return (value if isinstance(value, str) else None), "reasoning_effort"
    return None, None


def _get_codex_reasoning_effort_ceiling(
    resolved_route: dict[str, Any],
) -> Optional[str]:
    assert _openai_provider_value is not None
    assert _get_model_info is not None
    assert _model_cost is not None
    assert _load_bundled_model_cost_map_for_codex_policy is not None

    if (
        resolved_route.get("provider") != _openai_provider_value
        or resolved_route.get("route_family") != "codex_responses"
    ):
        return None

    model = resolved_route.get("model")
    if not isinstance(model, str) or not model:
        return None
    model_info_sources: list[Mapping[str, Any]] = []
    try:
        resolved_model_info = _get_model_info(
            model=model,
            custom_llm_provider=_openai_provider_value,
        )
        if isinstance(resolved_model_info, dict):
            model_info_sources.append(resolved_model_info)
    except Exception:
        pass
    for model_cost in (
        _model_cost,
        _load_bundled_model_cost_map_for_codex_policy(),
    ):
        catalog_model_info = model_cost.get(model)
        if isinstance(catalog_model_info, dict):
            model_info_sources.append(catalog_model_info)

    if any(model_info.get("supports_max_reasoning_effort") is True for model_info in model_info_sources):
        return "max"
    if any(model_info.get("supports_xhigh_reasoning_effort") is True for model_info in model_info_sources):
        return "xhigh"
    if any(model_info.get("supports_reasoning") is True for model_info in model_info_sources) and any(
        model_info.get("supports_xhigh_reasoning_effort") is False for model_info in model_info_sources
    ):
        return "high"
    return None


def _normalize_codex_reasoning_effort_for_resolved_route(
    request_body: dict[str, Any],
    *,
    resolved_route: dict[str, Any],
    attempt_number: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert _merge_litellm_metadata is not None

    requested_effort, native_field = _extract_codex_reasoning_effort(request_body)
    if requested_effort not in _CODEX_REASONING_EFFORT_TIER_INDEX or native_field is None:
        return request_body, {}

    supported_ceiling = _get_codex_reasoning_effort_ceiling(resolved_route)
    if supported_ceiling is None:
        return request_body, {}

    emitted_effort = requested_effort
    mapping_reason = "within_supported_ceiling"
    if _CODEX_REASONING_EFFORT_TIER_INDEX[requested_effort] > _CODEX_REASONING_EFFORT_TIER_INDEX[supported_ceiling]:
        emitted_effort = supported_ceiling
        mapping_reason = "requested_effort_above_model_supported_ceiling"

    updated_body = dict(request_body)
    if emitted_effort != requested_effort:
        if native_field == "reasoning.effort":
            reasoning = dict(updated_body.get("reasoning") or {})
            reasoning["effort"] = emitted_effort
            updated_body["reasoning"] = reasoning
        else:
            updated_body["reasoning_effort"] = emitted_effort

    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    existing_tags = litellm_metadata.get("tags")
    if isinstance(existing_tags, list):
        litellm_metadata["tags"] = [
            tag
            for tag in existing_tags
            if not (
                isinstance(tag, str)
                and (
                    tag == "reasoning-effort-clamped"
                    or tag.startswith("codex-effort:")
                    or tag.startswith("effort:")
                    or tag.startswith("reasoning-effort-ceiling:")
                    or tag.startswith("reasoning-effort-map:")
                    or tag.startswith("codex-auto-agent-attempt:")
                )
            )
        ]
    updated_body["litellm_metadata"] = litellm_metadata

    provider = str(resolved_route["provider"])
    model = str(resolved_route["model"])
    mapping_metadata: dict[str, Any] = {
        "codex_reasoning_effort": emitted_effort,
        "reasoning_effort_requested": requested_effort,
        "reasoning_effort_source": native_field,
        "reasoning_effort_native_provider": provider,
        "reasoning_effort_native_value": emitted_effort,
        "reasoning_effort_native_field": native_field,
        "reasoning_effort_supported_ceiling": supported_ceiling,
        "reasoning_effort_resolved_model": model,
        "reasoning_effort_resolved_provider": provider,
        "reasoning_effort_mapping_reason": mapping_reason,
        "openai_reasoning_effort": emitted_effort,
    }
    tags_to_add = [
        f"codex-effort:{emitted_effort}",
        f"effort:{emitted_effort}",
        f"reasoning-effort-ceiling:{supported_ceiling}",
    ]
    if attempt_number is not None:
        mapping_metadata["reasoning_effort_candidate_attempt"] = attempt_number
        tags_to_add.append(f"codex-auto-agent-attempt:{attempt_number}")
    if emitted_effort != requested_effort:
        mapping_metadata.update(
            {
                "reasoning_effort_clamped_from": requested_effort,
                "reasoning_effort_clamp_reason": mapping_reason,
            }
        )
        tags_to_add.extend(
            [
                "reasoning-effort-clamped",
                f"reasoning-effort-map:{requested_effort}-to-{emitted_effort}",
            ]
        )

    return (
        _merge_litellm_metadata(
            updated_body,
            tags_to_add=tags_to_add,
            extra_fields=mapping_metadata,
        ),
        mapping_metadata,
    )


# ---------------------------------------------------------------------------
# Codex / Anthropic alias metadata composition
# ---------------------------------------------------------------------------


def _require_selection_alias_model(selection: dict[str, Any]) -> str:
    alias_model = selection.get("alias_model")
    if (
        not isinstance(alias_model, str)
        or not alias_model
        or alias_model.strip() != alias_model
    ):
        raise ValueError("selection must contain an explicit canonical alias_model")
    return alias_model


def _add_codex_auto_agent_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    assert _normalize_low_cardinality_tag_value is not None
    assert _merge_litellm_metadata is not None
    assert _build_auto_agent_alias_audit_events is not None

    candidate = selection["candidate"]
    alias_model = _require_selection_alias_model(selection)
    target_model = candidate["model"]
    updated_body = copy.deepcopy(request_body)
    updated_body["model"] = target_model
    # CFG-006: an optional candidate-level YAML ``reasoning_effort`` is
    # AUTHORITATIVE. It replaces every caller/TUI reasoning representation
    # on the attempt body before the shared route normalizer runs, so the
    # configured value (never the caller's) feeds provider translation and
    # capability clamping. Omission leaves caller intent untouched.
    configured_reasoning_effort = _normalize_low_cardinality_tag_value(
        candidate.get("reasoning_effort")
    )
    if configured_reasoning_effort:
        updated_body.pop("reasoning_effort", None)
        prior_reasoning = updated_body.get("reasoning")
        remaining_reasoning = (
            {k: v for k, v in prior_reasoning.items() if k != "effort"}
            if isinstance(prior_reasoning, dict)
            else {}
        )
        updated_body["reasoning"] = {**remaining_reasoning, "effort": configured_reasoning_effort}
    default_reasoning_effort = _normalize_low_cardinality_tag_value(candidate.get("default_reasoning_effort"))
    default_reasoning_applied = False
    if default_reasoning_effort and "reasoning_effort" not in updated_body:
        reasoning = updated_body.get("reasoning")
        if not isinstance(reasoning, dict):
            updated_body["reasoning"] = {"effort": default_reasoning_effort}
            default_reasoning_applied = True
        elif not reasoning.get("effort"):
            updated_body["reasoning"] = {
                **reasoning,
                "effort": default_reasoning_effort,
            }
            default_reasoning_applied = True
    attempt_number = max(1, len(attempts))
    (
        updated_body,
        reasoning_effort_metadata,
    ) = _normalize_codex_reasoning_effort_for_resolved_route(
        updated_body,
        resolved_route=candidate,
        attempt_number=attempt_number,
    )
    if configured_reasoning_effort:
        reasoning_effort_metadata = {
            **reasoning_effort_metadata,
            "reasoning_effort_config_value": configured_reasoning_effort,
            "reasoning_effort_config_source": "candidate_yaml",
        }
    audit_selection = selection
    if reasoning_effort_metadata:
        if attempts:
            attempts[-1].update(reasoning_effort_metadata)
        else:
            audit_selection = {
                **selection,
                "candidate": {
                    **candidate,
                    **reasoning_effort_metadata,
                },
            }
    selection_trace_values = {
        key: selection[key]
        for key in ("request_mode", "redispatch_ordinal", "affinity_bypassed")
        if key in selection
    }
    if attempts and selection_trace_values:
        attempts[-1].update(selection_trace_values)
    skipped = selection.get("skipped") or []
    audit_events = _build_auto_agent_alias_audit_events(
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=request_body,
        selection=audit_selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "codex-auto-agent-alias",
            f"codex-auto-agent-selected:{target_model}",
            f"codex-auto-agent-route:{candidate['route_family']}",
            f"model-alias:{alias_model}",
            *(["codex-auto-agent-last-resort"] if candidate.get("last_resort") else []),
            *([f"codex-auto-agent-default-effort:{default_reasoning_effort}"] if default_reasoning_applied else []),
            *([f"codex-auto-agent-config-effort:{configured_reasoning_effort}"] if configured_reasoning_effort else []),
            f"codex-auto-agent-alias:{alias_model}",
        ],
        extra_fields={
            "model_alias_label": alias_model,
            "requested_model_alias": alias_model,
            "codex_auto_agent_alias": alias_model,
            "codex_auto_agent_selected_provider": candidate["provider"],
            "codex_auto_agent_selected_model": target_model,
            "codex_auto_agent_selected_route_family": candidate["route_family"],
            "codex_auto_agent_selected_last_resort": bool(candidate.get("last_resort")),
            **(
                {"codex_auto_agent_config_reasoning_effort": configured_reasoning_effort}
                if configured_reasoning_effort
                else {}
            ),
            **(
                {
                    "codex_auto_agent_default_reasoning_effort": (default_reasoning_effort),
                    "codex_reasoning_effort": (
                        reasoning_effort_metadata.get("codex_reasoning_effort") or default_reasoning_effort
                    ),
                }
                if default_reasoning_applied
                else {}
            ),
            "codex_auto_agent_selection_reason": selection.get("selection_reason"),
            "codex_auto_agent_affinity_state_source": selection.get("affinity_state_source"),
            "canonical_session_identity": selection.get("canonical_session_identity"),
            "session_owner_decision": selection.get("session_owner_decision"),
            "session_owner_id": selection.get("session_owner_id"),
            "session_owner_mismatch_reason": selection.get("session_owner_mismatch_reason"),
            "codex_auto_agent_cooldown_state_source": selection.get("cooldown_state_source"),
            "codex_auto_agent_lane_key": selection.get("lane_key"),
            "codex_auto_agent_request_mode": selection.get("request_mode"),
            "codex_auto_agent_redispatch_ordinal": selection.get("redispatch_ordinal"),
            "codex_auto_agent_affinity_bypassed": selection.get("affinity_bypassed"),
            "codex_auto_agent_selected_account_label": candidate.get(
                "codex_oauth_account_label"
            ),
            "codex_auto_agent_selected_account_hash": candidate.get(
                "codex_oauth_account_hash"
            ),
            "codex_auto_agent_selected_account_lane": candidate.get(
                "codex_oauth_lane_key"
            ),
            "codex_auto_agent_quota_snapshot_age_seconds": selection.get(
                "quota_snapshot_age_seconds"
            ),
            "codex_auto_agent_failover_ordinal": selection.get(
                "failover_ordinal"
            ),
            "codex_auto_agent_prior_account_outcome": selection.get(
                "prior_account_outcome"
            ),
            "codex_auto_agent_terminal_reset": selection.get(
                "terminal_reset"
            ),
            "codex_auto_agent_attempts": attempts,
            "codex_auto_agent_skipped_candidates": skipped,
            "codex_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )


def _add_anthropic_auto_agent_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    assert _normalize_low_cardinality_tag_value is not None
    assert _merge_litellm_metadata is not None
    assert _build_auto_agent_alias_audit_events is not None

    candidate = selection["candidate"]
    alias_model = _require_selection_alias_model(selection)
    target_model = candidate["model"]
    updated_body = copy.deepcopy(request_body)
    updated_body["model"] = target_model
    # CFG-006: an optional candidate-level YAML ``reasoning_effort`` is
    # AUTHORITATIVE on the Anthropic Messages ingress as well. Conflicting
    # caller effort/thinking representations (``thinking``, ``output_config``
    # effort, top-level ``reasoning_effort``) are removed ONLY when a config
    # value is set, and the canonical value is placed at the top level so the
    # shared adapter/provider translation seams own native mapping/clamping.
    # Omission preserves caller intent untouched.
    configured_reasoning_effort = _normalize_low_cardinality_tag_value(
        candidate.get("reasoning_effort")
    )
    if configured_reasoning_effort:
        updated_body.pop("thinking", None)
        output_config = updated_body.get("output_config")
        if isinstance(output_config, dict):
            remaining_output_config = {
                key: value for key, value in output_config.items() if key != "effort"
            }
            if remaining_output_config:
                updated_body["output_config"] = remaining_output_config
            else:
                updated_body.pop("output_config", None)
        updated_body["reasoning_effort"] = configured_reasoning_effort
    audit_selection = selection
    if configured_reasoning_effort:
        reasoning_effort_metadata = {
            "reasoning_effort_config_value": configured_reasoning_effort,
            "reasoning_effort_config_source": "candidate_yaml",
        }
        # Mirror the Codex seam: when an attempt record exists it is the
        # audit source of truth, so propagate onto the final attempt; the
        # enriched selection candidate only covers the empty-attempt
        # fallback used by audit construction.
        if attempts:
            attempts[-1].update(reasoning_effort_metadata)
        else:
            audit_selection = {
                **selection,
                "candidate": {
                    **candidate,
                    **reasoning_effort_metadata,
                },
            }
    selection_trace_values = {
        key: selection[key]
        for key in ("request_mode", "redispatch_ordinal", "affinity_bypassed")
        if key in selection
    }
    if attempts and selection_trace_values:
        attempts[-1].update(selection_trace_values)
    skipped = selection.get("skipped") or []
    audit_events = _build_auto_agent_alias_audit_events(
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=request_body,
        selection=audit_selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "anthropic-auto-agent-alias",
            f"anthropic-auto-agent-selected:{target_model}",
            f"anthropic-auto-agent-route:{candidate['route_family']}",
            f"model-alias:{alias_model}",
            *(["anthropic-auto-agent-last-resort"] if candidate.get("last_resort") else []),
            *(
                [f"anthropic-auto-agent-config-effort:{configured_reasoning_effort}"]
                if configured_reasoning_effort
                else []
            ),
            f"anthropic-auto-agent-alias:{alias_model}",
        ],
        extra_fields={
            "model_alias_label": alias_model,
            "requested_model_alias": alias_model,
            "anthropic_auto_agent_alias": alias_model,
            "anthropic_auto_agent_selected_provider": candidate["provider"],
            "anthropic_auto_agent_selected_model": target_model,
            "anthropic_auto_agent_selected_route_family": candidate["route_family"],
            "anthropic_auto_agent_selected_last_resort": bool(candidate.get("last_resort")),
            **(
                {"anthropic_auto_agent_config_reasoning_effort": configured_reasoning_effort}
                if configured_reasoning_effort
                else {}
            ),
            "anthropic_auto_agent_selection_reason": selection.get("selection_reason"),
            "anthropic_auto_agent_affinity_state_source": selection.get("affinity_state_source"),
            "canonical_session_identity": selection.get("canonical_session_identity"),
            "session_owner_decision": selection.get("session_owner_decision"),
            "session_owner_id": selection.get("session_owner_id"),
            "session_owner_mismatch_reason": selection.get("session_owner_mismatch_reason"),
            "anthropic_auto_agent_cooldown_state_source": selection.get("cooldown_state_source"),
            "anthropic_auto_agent_lane_key": selection.get("lane_key"),
            "anthropic_auto_agent_request_mode": selection.get("request_mode"),
            "anthropic_auto_agent_redispatch_ordinal": selection.get("redispatch_ordinal"),
            "anthropic_auto_agent_affinity_bypassed": selection.get("affinity_bypassed"),
            "anthropic_auto_agent_selected_account_label": candidate.get(
                "codex_oauth_account_label"
            ),
            "anthropic_auto_agent_selected_account_hash": candidate.get(
                "codex_oauth_account_hash"
            ),
            "anthropic_auto_agent_selected_account_lane": candidate.get(
                "codex_oauth_lane_key"
            ),
            "anthropic_auto_agent_quota_snapshot_age_seconds": selection.get(
                "quota_snapshot_age_seconds"
            ),
            "anthropic_auto_agent_failover_ordinal": selection.get(
                "failover_ordinal"
            ),
            "anthropic_auto_agent_prior_account_outcome": selection.get(
                "prior_account_outcome"
            ),
            "anthropic_auto_agent_terminal_reset": selection.get(
                "terminal_reset"
            ),
            "anthropic_auto_agent_attempts": attempts,
            "anthropic_auto_agent_skipped_candidates": skipped,
            "anthropic_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )


# ---------------------------------------------------------------------------
# God-module facade installation (Wave 5C)
# ---------------------------------------------------------------------------

_HOST_FUNCTION_NAMES = (
    "_update_codex_auto_agent_retryable_attempt_record",
    "_record_auto_agent_alias_attempt_started",
    "_record_codex_failure_evidence",
    "_record_auto_agent_alias_attempt_failure",
    "_record_auto_agent_alias_attempt_success",
    "_extract_codex_reasoning_effort",
    "_get_codex_reasoning_effort_ceiling",
    "_normalize_codex_reasoning_effort_for_resolved_route",
    "_add_codex_auto_agent_alias_metadata",
    "_add_anthropic_auto_agent_alias_metadata",
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
    """Publish same-object god-module facades for the moved functions.

    Functions retain this module's globals. Host-owned dependencies remain
    late-bound through the callbacks configured by
    :func:`configure_attempt_records_runtime`.
    """
    global _host_globals_ref
    _mod = globals()
    owner_module = _sys.modules[__name__]
    for _name in _RUNTIME_STATE_NAMES:
        host_callback = host_globals.get(_name, _MISSING)
        if host_callback is _MISSING or _host_callback_delegates_to_module(
            _name,
            host_callback,
            owner_module,
        ):
            continue
        _mod[_name] = host_callback
    _host_globals_ref = host_globals
    for _name in _HOST_FUNCTION_NAMES:
        host_globals[_name] = _mod[_name]

    from . import audit_context as _audit_context

    for _name in _audit_context._SEAM_NAMES:
        host_callback = host_globals.get(_name, _MISSING)
        if host_callback is _MISSING or _host_callback_delegates_to_module(
            _name,
            host_callback,
            _audit_context,
        ):
            continue
        setattr(_audit_context, _name, host_callback)


# ---------------------------------------------------------------------------
# Module __setattr__ propagation for callback restores and monkeypatches
# ---------------------------------------------------------------------------

import sys as _sys
import types as _types

_SEAM_NAMES = frozenset(_RUNTIME_STATE_NAMES)


class _SeamPropagatingModule(_types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        seam_names = self.__dict__.get("_SEAM_NAMES")
        if seam_names is None or name not in seam_names:
            return
        host_globals = self.__dict__.get("_host_globals_ref")
        if host_globals is None:
            return
        restore_stacks = self.__dict__.get("_runtime_restore_stacks", {})
        restore_stack = restore_stacks.get(name)
        if restore_stack and value is restore_stack[-1][1]:
            _, _, prior_host_value = restore_stack.pop()
            if prior_host_value is self.__dict__["_MISSING"]:
                host_globals.pop(name, None)
            else:
                host_globals[name] = prior_host_value
            return
        host_globals[name] = value


_sys.modules[__name__].__class__ = _SeamPropagatingModule
