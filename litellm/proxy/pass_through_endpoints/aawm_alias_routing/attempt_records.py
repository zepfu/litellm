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
import math
import time
from contextvars import ContextVar, Token
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

from fastapi import Request

from . import opencode_go_preflight as _opencode_go_preflight
from .lane_keys import _CODEX_REASONING_EFFORT_TIER_INDEX
from .policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_EXHAUSTED_ERROR_CLASSES,
    CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
    CODEX_AUTO_AGENT_NATIVE_PROVIDER,
)
from .request_metadata import (
    _extract_auto_agent_alias_canonical_thread_id,
    _extract_auto_agent_alias_parent_thread_id,
)
from .selection import _attempt_has_provider_call, _provider_attempt_count
from .schema_rejections import (
    SCHEMA_REJECTION_KEY,
    SchemaRejectionDiagnostic,
    extract_schema_rejection,
    normalize_schema_rejection,
    resolve_schema_rejection_failure_identity,
)
from .opencode_go_rejections import (
    OPENCODE_GO_DIRECT_ALIAS_FAMILY,
    OPENCODE_GO_PROVIDER,
    OPENCODE_GO_REJECTION_KEY,
    OPENCODE_GO_ROUTE_FAMILY,
    attach_opencode_go_rejection,
    normalize_opencode_go_rejection,
)
from .skip_identity import _auto_agent_alias_skip_identity

_AAWM_ALIAS_REQUEST_CALL_ID_STATE_KEY = "aawm_alias_request_litellm_call_id"
_AAWM_ALIAS_REQUEST_OUTCOME_STATE_KEY = "aawm_alias_request_outcome"
_AAWM_ALIAS_SKIPPED_EVENT_KEYS_STATE_KEY = (
    "aawm_alias_request_emitted_skipped_event_keys"
)
_AAWM_ALIAS_SKIPPED_EVENT_TYPES = frozenset(
    {
        "candidate_skipped_cooldown",
        "candidate_skipped_provider_degraded",
        "candidate_skipped_semantic_ineligible",
    }
)

# Capability helper resolved lazily (import-safe within the litellm package).
# Unlike the injected ``_get_model_info`` seam, this reads the real model
# catalog and is the config-driven source of truth for xAI xhigh support.
_supports_xhigh_reasoning_effort: Optional[Callable[..., bool]] = None


def _resolve_supports_xhigh_reasoning_effort() -> Optional[Callable[..., bool]]:
    global _supports_xhigh_reasoning_effort
    if _supports_xhigh_reasoning_effort is None:
        try:
            from litellm.utils import supports_xhigh_reasoning_effort

            _supports_xhigh_reasoning_effort = supports_xhigh_reasoning_effort
        except Exception:
            return None
    return _supports_xhigh_reasoning_effort

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
    _emit_auto_agent_alias_route_event as _default_emit_auto_agent_alias_route_event,
    _persist_auto_agent_alias_audit_only_events_best_effort as _default_persist_audit_only_events,
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


def _safe_publication_seconds(value: Any) -> Optional[float]:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return round(seconds, 3)


def _safe_publication_label(
    value: Any,
    *,
    allowed: Optional[set[str]] = None,
) -> Optional[str]:
    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    if allowed is not None:
        return value if value in allowed else None
    if any(not (character.isalnum() or character in "._-") for character in value):
        return None
    return value


def _attach_schema_rejection_to_attempt_record(
    *,
    attempt_record: dict[str, Any],
    exc: Any = None,
    candidate: Optional[Mapping[str, Any]] = None,
    diagnostic: Optional[SchemaRejectionDiagnostic] = None,
) -> Optional[SchemaRejectionDiagnostic]:
    """Attach one bounded schema diagnostic and its stable failure identity."""
    candidate_mapping = candidate if isinstance(candidate, Mapping) else {}
    provider = candidate_mapping.get("provider") or attempt_record.get("provider")
    route_family = candidate_mapping.get("route_family") or attempt_record.get(
        "route_family"
    )
    if diagnostic is None:
        if exc is not None:
            diagnostic = extract_schema_rejection(
                exc,
                provider=provider,
                route_family=route_family,
                attempted_provider_call=attempt_record.get(
                    "attempted_provider_call"
                ),
                failure_phase=attempt_record.get("failure_phase"),
            )
    if diagnostic is None:
        diagnostic = normalize_schema_rejection(
            attempt_record,
            provider=provider,
            route_family=route_family,
        )
    if diagnostic is None:
        return None
    attempt_record[SCHEMA_REJECTION_KEY] = diagnostic.to_dict()
    failure_class, error_code = resolve_schema_rejection_failure_identity(
        failure_class=attempt_record.get("error_class"),
        error_code=attempt_record.get("error_code"),
    )
    attempt_record["error_class"] = failure_class
    attempt_record["error_code"] = error_code
    return diagnostic


def _attach_opencode_go_rejection_to_attempt_record(
    *,
    attempt_record: dict[str, Any],
    exc: Any = None,
    candidate: Optional[Mapping[str, Any]] = None,
    request: Any = None,
    diagnostic: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Attach one bounded OpenCode Go rejection onto the attempt contract."""

    return attach_opencode_go_rejection(
        target=attempt_record,
        request=request,
        exc=exc,
        candidate=candidate,
        diagnostic=diagnostic,
    )


def _persist_opencode_go_direct_rejection_audit(
    *,
    request: Any,
    evidence: Mapping[str, Any],
    adapter_model: Any = None,
    request_body: Optional[dict[str, Any]] = None,
) -> None:
    """Persist one terminal audit event for a direct Go rejection."""

    recorded = normalize_opencode_go_rejection(evidence)
    if recorded is None or recorded.get("call_mode") != "direct":
        return
    model = (
        adapter_model
        if isinstance(adapter_model, str) and adapter_model.strip()
        else recorded.get("provider") or OPENCODE_GO_PROVIDER
    )
    event: dict[str, Any] = {
        "alias_family": OPENCODE_GO_DIRECT_ALIAS_FAMILY,
        "alias_model": model,
        "provider": OPENCODE_GO_PROVIDER,
        "model": model,
        "route_family": OPENCODE_GO_ROUTE_FAMILY,
        "event_type": "provider_rejection",
        "candidate_status": "failed",
        "failure_class": recorded.get("failure_class"),
        "error_status_code": recorded.get("status"),
        "failure_phase": recorded.get("failure_phase"),
        "attempted_provider_call": True,
        OPENCODE_GO_REJECTION_KEY: recorded,
    }
    _stamp_auto_agent_alias_request_identity(request=request, target=event)
    identity = event.get("request_identity")
    if isinstance(identity, str) and identity and recorded.get("request_identity") is None:
        recorded = {
            **recorded,
            "request_identity": identity,
            "litellm_call_id": identity,
        }
        event[OPENCODE_GO_REJECTION_KEY] = recorded
    try:
        _default_emit_auto_agent_alias_route_event(event, level="warning")
    except Exception:
        pass
    try:
        _default_persist_audit_only_events(
            [event],
            request_body=request_body if isinstance(request_body, dict) else None,
        )
    except Exception:
        pass


def _attach_kimi_managed_account_publication_telemetry(
    *,
    attempt_record: dict[str, Any],
    candidate: Optional[dict[str, Any]],
    error_class: Optional[str],
    kimi_failure_metadata: Optional[dict[str, Any]],
    plan: Any,
    requested_ttl_seconds: Any,
    transaction_result: Optional[object] = None,
    publication_error: Optional[BaseException] = None,
) -> None:
    """Attach safe publication evidence for confirmed managed Kimi quota."""

    if (
        not isinstance(candidate, dict)
        or candidate.get("provider") != CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER
        or error_class != "kimi_code_managed_account"
        or not isinstance(kimi_failure_metadata, dict)
        or kimi_failure_metadata.get("scope") != "managed_account"
        or getattr(plan, "applied_scope", None) != "managed_account"
    ):
        return

    durable_keys = getattr(plan, "durable_keys", ())
    if (
        not isinstance(durable_keys, tuple)
        or len(durable_keys) != 1
        or not isinstance(durable_keys[0], str)
        or not durable_keys[0]
    ):
        return
    requested_ttl = _safe_publication_seconds(requested_ttl_seconds)
    effective_plan_ttl = _safe_publication_seconds(
        getattr(plan, "duration_seconds", None)
    )
    if requested_ttl is None or effective_plan_ttl is None:
        return

    telemetry: dict[str, Any] = {
        "classification": error_class,
        "scope": "managed_account",
        "logical_cooldown_key": durable_keys[0],
        "requested_ttl_seconds": requested_ttl,
        "effective_plan_ttl_seconds": effective_plan_ttl,
    }
    if publication_error is not None:
        failure_phase = _safe_publication_label(
            getattr(publication_error, "phase", None),
            allowed={"PREPARED", "DURABLE_COMMITTED", "LOCAL_COMMITTED"},
        )
        telemetry["state_source"] = "durable_publication_failed"
        telemetry["durable_publication_failure"] = {
            "exception_class": type(publication_error).__name__,
            "phase": failure_phase or "publication_transaction",
        }
    elif transaction_result is None:
        telemetry["state_source"] = "local_fallback"
    else:
        transaction_id = _safe_publication_label(
            getattr(transaction_result, "transaction_id", None)
        )
        phase = _safe_publication_label(
            getattr(transaction_result, "phase", None),
            allowed={"PREPARED", "DURABLE_COMMITTED", "LOCAL_COMMITTED"},
        )
        journal = getattr(transaction_result, "journal", None)
        receipt_ttl = _safe_publication_seconds(
            getattr(journal, "requested_ttl", None)
        )
        if transaction_id is None or phase is None or receipt_ttl is None:
            telemetry["state_source"] = "durable_publication_failed"
            telemetry["durable_publication_failure"] = {
                "exception_class": "InvalidPublicationReceipt",
                "phase": "publication_transaction",
            }
        else:
            telemetry["state_source"] = "durable_cache"
            telemetry["durable_transaction_receipt"] = {
                "id": transaction_id,
                "phase": phase,
                "requested_ttl_seconds": receipt_ttl,
            }
    attempt_record["kimi_managed_account_publication"] = telemetry


def _update_codex_auto_agent_retryable_attempt_record(  # noqa: PLR0915
    *,
    attempt_record: dict[str, Any],
    exc: Any,
    error_class: str,
    cooldown_seconds: float,
    alias_model: str,
    cooldown_scope: Optional[str] = None,
    candidate: Optional[dict[str, Any]] = None,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    attempted_provider_call: Optional[bool] = None,
    provider_returned: Optional[bool] = None,
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
    candidate_status = getattr(exc, "candidate_status", None)
    ineligibility_reason = getattr(exc, "ineligibility_reason", None)
    is_deterministically_ineligible = (
        error_class == "candidate_deterministically_ineligible"
    )
    if not isinstance(attempted_provider_call, bool):
        attempted_provider_call = getattr(exc, "attempted_provider_call", None)
    if not isinstance(attempted_provider_call, bool):
        provider_returned = getattr(exc, "_aawm_provider_returned", None)
        if provider_returned is True:
            attempted_provider_call = True
    if not isinstance(attempted_provider_call, bool):
        existing_attempted_provider_call = attempt_record.get(
            "attempted_provider_call"
        )
        if isinstance(existing_attempted_provider_call, bool):
            attempted_provider_call = existing_attempted_provider_call
    if not isinstance(attempted_provider_call, bool):
        attempted_provider_call = False
    if not isinstance(provider_returned, bool):
        existing_provider_returned = attempt_record.get("provider_returned")
        if isinstance(existing_provider_returned, bool):
            provider_returned = existing_provider_returned
    if not isinstance(provider_returned, bool):
        detail = getattr(exc, "detail", None)
        detail_mapping = detail if isinstance(detail, Mapping) else {}
        provider_returned = (
            getattr(exc, "_aawm_provider_returned", False) is True
            or getattr(exc, "provider_returned", False) is True
            or detail_mapping.get("provider_returned") is True
        )
    update: dict[str, Any] = {
        "status": (
            "candidate_ineligible_no_cooldown"
            if is_deterministically_ineligible
            else (
                "retryable_no_cooldown"
                if cooldown_scope == "none"
                else "cooldown_set"
            )
        ),
        "error_class": error_class,
        "error_tokens": sorted(error_tokens),
        "failure_phase": getattr(exc, "failure_phase", "provider_attempt"),
        "attempted_provider_call": attempted_provider_call,
        "provider_returned": provider_returned,
        "source_error": source_error,
    }
    if candidate_status is not None:
        update["candidate_status"] = candidate_status
    if ineligibility_reason is not None:
        update["ineligibility_reason"] = ineligibility_reason
    preflight_reason = getattr(exc, "preflight_reason", None)
    if (
        not isinstance(preflight_reason, str)
        or preflight_reason not in _opencode_go_preflight.OPENCODE_GO_PREFLIGHT_REASONS
    ):
        detail = getattr(exc, "detail", None)
        detail_mapping = detail if isinstance(detail, Mapping) else {}
        preflight_reason = detail_mapping.get("preflight_reason")
    if (
        isinstance(preflight_reason, str)
        and preflight_reason in _opencode_go_preflight.OPENCODE_GO_PREFLIGHT_REASONS
    ):
        update["preflight_reason"] = preflight_reason
    call_mode = getattr(exc, "opencode_go_call_mode", None)
    if (
        not isinstance(call_mode, str)
        or call_mode not in _opencode_go_preflight.OPENCODE_GO_PREFLIGHT_CALL_MODES
    ):
        detail = getattr(exc, "detail", None)
        detail_mapping = detail if isinstance(detail, Mapping) else {}
        call_mode = detail_mapping.get("opencode_go_call_mode")
    if (
        isinstance(call_mode, str)
        and call_mode in _opencode_go_preflight.OPENCODE_GO_PREFLIGHT_CALL_MODES
    ):
        update["opencode_go_call_mode"] = call_mode
    if is_deterministically_ineligible:
        # Deterministic candidate rejection never publishes local or durable
        # cooldown state, even if a caller carried a stale scope or duration.
        attempt_record.pop("cooldown_seconds", None)
        update["cooldown_scope"] = "none"
    else:
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
    elif error_class in CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_EXHAUSTED_ERROR_CLASSES:
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
    _attach_schema_rejection_to_attempt_record(
        attempt_record=attempt_record,
        exc=exc,
        candidate=candidate,
    )
    _attach_opencode_go_rejection_to_attempt_record(
        attempt_record=attempt_record,
        exc=exc,
        candidate=candidate,
    )
    return error_tokens


# ---------------------------------------------------------------------------
# Attempt-start / failure records
# ---------------------------------------------------------------------------


def _is_auto_agent_alias_skipped_audit_event(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "")
    return (
        event_type in _AAWM_ALIAS_SKIPPED_EVENT_TYPES
        or event.get("skipped") is True
    )


def _auto_agent_alias_skipped_event_key(
    event: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return a stable, occurrence-aware request-local identity."""
    return _auto_agent_alias_skip_identity(event)


def _emit_auto_agent_alias_skipped_events_once(
    *,
    request: Request,
    audit_events: list[dict[str, Any]],
    level: Optional[str] = None,
) -> set[tuple[Any, ...]]:
    """Publish each skipped decision once for this request."""
    request_state = getattr(request, "state", None)
    if request_state is None:
        return set()
    emitted_keys = getattr(
        request_state,
        _AAWM_ALIAS_SKIPPED_EVENT_KEYS_STATE_KEY,
        None,
    )
    if not isinstance(emitted_keys, set):
        emitted_keys = set()
        setattr(
            request_state,
            _AAWM_ALIAS_SKIPPED_EVENT_KEYS_STATE_KEY,
            emitted_keys,
        )

    newly_emitted_keys: set[tuple[Any, ...]] = set()
    for event in audit_events:
        if not _is_auto_agent_alias_skipped_audit_event(event):
            continue
        event_key = _auto_agent_alias_skipped_event_key(event)
        if event_key in emitted_keys:
            continue
        emitted_keys.add(event_key)
        newly_emitted_keys.add(event_key)
        _stamp_auto_agent_alias_request_identity(request=request, target=event)
        if level is None:
            _emit_auto_agent_alias_route_event(event)
        else:
            _emit_auto_agent_alias_route_event(event, level=level)
    return newly_emitted_keys


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
        _emit_auto_agent_alias_skipped_events_once(
            request=request,
            audit_events=[
                event for event in audit_events if isinstance(event, dict)
            ],
        )
        latest_event = audit_events[-1]
        if (
            isinstance(latest_event, dict)
            and not _is_auto_agent_alias_skipped_audit_event(latest_event)
        ):
            _stamp_auto_agent_alias_request_identity(
                request=request,
                target=latest_event,
            )
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
    defer_terminal_error: bool = False,
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
            attempt_number=(
                _provider_attempt_count(attempts)
                if _attempt_has_provider_call(attempt_record)
                else None
            ),
            attempt_record_index=(
                len(attempts) - 1
                if attempts and attempts[-1] is attempt_record
                else None
            ),
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
    _attach_opencode_go_rejection_to_attempt_record(
        attempt_record=audit_event,
        candidate=attempt_record,
        request=request,
    )
    if defer_terminal_error:
        audit_event["_aawm_terminal_error_already_emitted"] = True
    _stamp_openrouter_inner_send_fields(audit_event, attempt_record)
    if _is_auto_agent_alias_skipped_audit_event(audit_event):
        _emit_auto_agent_alias_skipped_events_once(
            request=request,
            audit_events=[audit_event],
            level="warning",
        )
    else:
        _emit_auto_agent_alias_route_event(
            audit_event,
            level="warning",
        )
    if (
        audit_events
        and (
            _aawm_alias_route_verbose_json_enabled()
            or _aawm_alias_route_healthy_json_enabled()
        )
    ):
        _emit_auto_agent_alias_skipped_events_once(
            request=request,
            audit_events=audit_events,
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
    if _emit_auto_agent_alias_route_event is None:
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
            event_type="candidate_recovered" if recovered else "candidate_completed",
            candidate_status=attempt_record.get("status") or (
                "recovered" if recovered else "completed"
            ),
            attempt_number=(
                _provider_attempt_count(attempts)
                if _attempt_has_provider_call(attempt_record)
                else None
            ),
            attempt_record_index=(
                len(attempts) - 1
                if attempts and attempts[-1] is attempt_record
                else None
            ),
            selected=True,
            selection_reason=selection.get("selection_reason"),
            lane_key=selection.get("lane_key"),
            cooldown_key=selection.get("cooldown_key"),
            attempted_provider_call=attempt_record.get("attempted_provider_call"),
        )
    if audit_event is None:
        audit_event = {
            "event_type": "candidate_recovered" if recovered else "candidate_completed",
            "candidate_status": attempt_record.get("status") or (
                "recovered" if recovered else "completed"
            ),
            "alias_family": alias_family,
            "alias_model": alias_model,
            "selected": True,
            "selection_reason": selection.get("selection_reason"),
        }
    audit_event["event_type"] = (
        "candidate_recovered" if recovered else "candidate_completed"
    )
    audit_event["candidate_status"] = attempt_record.get("status") or (
        "recovered" if recovered else "completed"
    )
    audit_event["request_outcome"] = "recovered" if recovered else "success"
    audit_event["attempts"] = copy.deepcopy(attempts)
    audit_event["attempt_count"] = _provider_attempt_count(attempts)
    _stamp_openrouter_inner_send_fields(audit_event, attempt_record)
    audit_event["session_owner_continuity_receipt"] = selection.get(
        "session_owner_continuity_receipt"
    )
    _stamp_auto_agent_alias_request_identity(request=request, target=audit_event)
    if (
        audit_events
        and (
            _aawm_alias_route_verbose_json_enabled()
            or _aawm_alias_route_healthy_json_enabled()
        )
    ):
        _emit_auto_agent_alias_skipped_events_once(
            request=request,
            audit_events=audit_events,
        )
    _emit_auto_agent_alias_route_event(audit_event)
    return success_body


def _append_openrouter_inner_subattempt(
    attempt_record: dict[str, Any],
    *,
    inner_attempt: int,
    status: str,
    disposition: str,
    delay_seconds: Optional[float] = None,
    cooldown_seconds: Optional[float] = None,
    error_status_code: Optional[int] = None,
    failure_class: Optional[str] = None,
    attempted_provider_call: bool = True,
) -> dict[str, Any]:
    """Record one inner OpenRouter send on the outer alias attempt.

    Nested subattempts keep the candidate-loop parent as one outer attempt so
    quota/accepted-call accounting is not double-counted, while every wire
    call remains observable with status, delay, cooldown, and disposition.
    """
    subattempts = attempt_record.get("subattempts")
    if not isinstance(subattempts, list):
        subattempts = []
        attempt_record["subattempts"] = subattempts
    payload: dict[str, Any] = {
        "inner_attempt": inner_attempt,
        "status": status,
        "disposition": disposition,
        "attempted_provider_call": attempted_provider_call,
    }
    if delay_seconds is not None:
        payload["delay_seconds"] = round(float(delay_seconds), 3)
        payload["wait_seconds"] = payload["delay_seconds"]
    if cooldown_seconds is not None:
        payload["cooldown_seconds"] = round(float(cooldown_seconds), 3)
    if error_status_code is not None:
        payload["error_status_code"] = error_status_code
    if failure_class is not None:
        payload["failure_class"] = failure_class
    if subattempts and subattempts[-1].get("inner_attempt") == inner_attempt:
        subattempts[-1].update(payload)
        current = subattempts[-1]
    else:
        subattempts.append(payload)
        current = payload
    wire_count = sum(
        1
        for subattempt in subattempts
        if isinstance(subattempt, Mapping)
        and subattempt.get("attempted_provider_call") is not False
    )
    attempt_record["subattempt_count"] = len(subattempts)
    attempt_record["logical_provider_send_count"] = wire_count
    attempt_record["hidden_logical_retry_count"] = max(0, wire_count - 1)
    return current


def _stamp_openrouter_inner_send_fields(
    target: dict[str, Any],
    attempt_record: Mapping[str, Any],
) -> None:
    if not attempt_record.get("subattempts"):
        return
    target["subattempts"] = copy.deepcopy(attempt_record["subattempts"])
    target["subattempt_count"] = attempt_record.get("subattempt_count")
    target["logical_provider_send_count"] = attempt_record.get(
        "logical_provider_send_count"
    )
    target["hidden_logical_retry_count"] = attempt_record.get(
        "hidden_logical_retry_count"
    )
    if attempt_record.get("provider_call_count") is not None:
        target["provider_call_count"] = attempt_record.get("provider_call_count")
    if attempt_record.get("aggregate_usage") is not None:
        target["aggregate_usage"] = copy.deepcopy(attempt_record.get("aggregate_usage"))
    if attempt_record.get("aggregate_usage_status") is not None:
        target["aggregate_usage_status"] = attempt_record.get("aggregate_usage_status")
    if attempt_record.get("aggregate_usage_subtotal") is not None:
        target["aggregate_usage_subtotal"] = copy.deepcopy(
            attempt_record.get("aggregate_usage_subtotal")
        )
    if "ciphertext_repair_retry_eligible" in attempt_record:
        target["ciphertext_repair_retry_eligible"] = attempt_record.get(
            "ciphertext_repair_retry_eligible"
        )
    if attempt_record.get("ciphertext_repair_blocked") is True:
        target["ciphertext_repair_blocked"] = True
    if attempt_record.get("downstream_response_committed") is True:
        target["downstream_response_committed"] = True


def bind_openrouter_inner_send_sink(attempt_record: dict[str, Any]) -> object:
    """Bind the current alias attempt as the OpenRouter inner-send sink."""
    from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
        retry_transport as _openrouter_retry_transport,
    )

    def _sink(**payload: Any) -> dict[str, Any]:
        return _append_openrouter_inner_subattempt(attempt_record, **payload)

    return _openrouter_retry_transport.bind_inner_send_sink(_sink)


def _settle_outstanding_openrouter_inner_subattempt(
    attempt_record: dict[str, Any],
) -> None:
    """Finalize a leftover in_flight inner send before the sink is reset."""
    subattempts = attempt_record.get("subattempts")
    if not isinstance(subattempts, list) or not subattempts:
        return
    current = subattempts[-1]
    if not isinstance(current, Mapping) or current.get("status") != "in_flight":
        return
    inner_attempt = current.get("inner_attempt")
    if not isinstance(inner_attempt, int) or inner_attempt <= 0:
        return
    delay_seconds = current.get("delay_seconds")
    payload: dict[str, Any] = {
        "inner_attempt": inner_attempt,
        "status": "cancelled",
        "disposition": "cancelled",
        "delay_seconds": 0.0 if delay_seconds is None else delay_seconds,
        "attempted_provider_call": True,
        "cooldown_seconds": (
            current["cooldown_seconds"]
            if current.get("cooldown_seconds") is not None
            else 0.0
        ),
    }
    if current.get("error_status_code") is not None:
        payload["error_status_code"] = current["error_status_code"]
    if current.get("failure_class") is not None:
        payload["failure_class"] = current["failure_class"]
    _append_openrouter_inner_subattempt(attempt_record, **payload)


def reset_openrouter_inner_send_sink(
    token: Any,
    attempt_record: Optional[dict[str, Any]] = None,
) -> None:
    from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
        retry_transport as _openrouter_retry_transport,
    )

    if attempt_record is not None:
        _settle_outstanding_openrouter_inner_subattempt(attempt_record)
    _openrouter_retry_transport.reset_inner_send_sink(token)


# ---------------------------------------------------------------------------
# Alibaba ciphertext-repair subattempts
# ---------------------------------------------------------------------------

_ALIBABA_CIPHERTEXT_REPAIR_SINK: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "aawm_alibaba_ciphertext_repair_sink",
    default=None,
)
_ALIBABA_DOWNSTREAM_COMMITTED_STATE_KEY = "aawm_downstream_response_committed"
_ALIBABA_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def bind_alibaba_ciphertext_repair_sink(
    attempt_record: dict[str, Any],
) -> Token[Optional[dict[str, Any]]]:
    """Bind ciphertext-repair generations onto the current outer attempt."""

    return _ALIBABA_CIPHERTEXT_REPAIR_SINK.set(attempt_record)


def _alibaba_attempt_record() -> Optional[dict[str, Any]]:
    attempt_record = _ALIBABA_CIPHERTEXT_REPAIR_SINK.get()
    if isinstance(attempt_record, dict):
        return attempt_record
    return None


def _coerce_usage_mapping(response: Any) -> Optional[dict[str, int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, Mapping):
        raw = dict(usage)
    else:
        raw = {name: getattr(usage, name, None) for name in _ALIBABA_USAGE_FIELDS}
    if not isinstance(raw, Mapping):
        return None
    cleaned: dict[str, int] = {}
    for key in _ALIBABA_USAGE_FIELDS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        cleaned[key] = int(value)
    return cleaned or None


def _refresh_alibaba_ciphertext_repair_aggregates(
    attempt_record: dict[str, Any],
) -> None:
    subattempts = attempt_record.get("subattempts")
    if not isinstance(subattempts, list):
        subattempts = []
    sent_subattempts = [
        subattempt
        for subattempt in subattempts
        if isinstance(subattempt, Mapping)
        and subattempt.get("kind") == "alibaba_ciphertext_generation"
        and subattempt.get("attempted_provider_call") is True
    ]
    provider_call_count = len(sent_subattempts)
    attempt_record["subattempt_count"] = len(subattempts)
    attempt_record["provider_call_count"] = provider_call_count
    attempt_record["logical_provider_send_count"] = provider_call_count
    # Repair generations are visible subattempts, so they are not hidden retries.
    attempt_record["hidden_logical_retry_count"] = 0
    aggregate: dict[str, int] = {}
    known_usage_count = 0
    for subattempt in sent_subattempts:
        usage = subattempt.get("usage")
        if not isinstance(usage, Mapping) or not usage:
            continue
        known_usage_count += 1
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            aggregate[key] = aggregate.get(key, 0) + int(value)
    usage_complete = (
        provider_call_count > 0 and known_usage_count == provider_call_count
    )
    if usage_complete:
        attempt_record["aggregate_usage"] = aggregate
        attempt_record.pop("aggregate_usage_status", None)
        attempt_record.pop("aggregate_usage_subtotal", None)
    else:
        attempt_record.pop("aggregate_usage", None)
        if provider_call_count == 0:
            attempt_record.pop("aggregate_usage_status", None)
            attempt_record.pop("aggregate_usage_subtotal", None)
        elif known_usage_count == 0:
            attempt_record["aggregate_usage_status"] = "unknown"
            attempt_record.pop("aggregate_usage_subtotal", None)
        else:
            attempt_record["aggregate_usage_status"] = "partial"
            attempt_record["aggregate_usage_subtotal"] = aggregate
    latest_eligible = False
    for subattempt in reversed(subattempts):
        if (
            isinstance(subattempt, Mapping)
            and subattempt.get("kind") == "alibaba_ciphertext_generation"
        ):
            latest_eligible = subattempt.get("retry_eligible") is True
            break
    attempt_record["ciphertext_repair_retry_eligible"] = latest_eligible


def _append_alibaba_ciphertext_subattempt(
    attempt_record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    subattempts = attempt_record.get("subattempts")
    if not isinstance(subattempts, list):
        subattempts = []
        attempt_record["subattempts"] = subattempts
    ordinal = payload.get("ordinal")
    if (
        subattempts
        and isinstance(subattempts[-1], dict)
        and subattempts[-1].get("ordinal") == ordinal
        and subattempts[-1].get("kind") == "alibaba_ciphertext_generation"
    ):
        subattempts[-1].update(payload)
        current = subattempts[-1]
    else:
        subattempts.append(payload)
        current = payload
    _refresh_alibaba_ciphertext_repair_aggregates(attempt_record)
    return current


def _failure_phase_is_pre_egress(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace("-", "_")
    return "pre_egress" in normalized


def _confirmed_pre_egress_failure(error: BaseException) -> bool:
    if getattr(error, "attempted_provider_call", None) is False:
        return True
    if _failure_phase_is_pre_egress(getattr(error, "failure_phase", None)):
        return True
    detail = getattr(error, "detail", None)
    if isinstance(detail, Mapping):
        if detail.get("attempted_provider_call") is False:
            return True
        if _failure_phase_is_pre_egress(detail.get("failure_phase")):
            return True
    return False


def _provider_send_provenance(source: Any) -> bool:
    """Return whether *source* records that provider I/O actually started."""

    if source is None:
        return False
    if getattr(source, "_aawm_provider_returned", None) is True:
        return True
    if getattr(source, "provider_returned", None) is True:
        return True
    status_code = getattr(source, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return True
    for logging_obj in (
        getattr(source, "litellm_logging_obj", None),
        getattr(source, "logging_obj", None),
    ):
        details = getattr(logging_obj, "model_call_details", None)
        if isinstance(details, Mapping) and details.get("api_call_start_time") is not None:
            return True
    hidden = getattr(source, "_hidden_params", None)
    if isinstance(hidden, Mapping) and hidden.get("api_call_start_time") is not None:
        return True
    if isinstance(source, Mapping):
        if source.get("_aawm_provider_returned") is True or source.get("provider_returned") is True:
            return True
        mapped_status = source.get("status_code")
        if isinstance(mapped_status, int) and not isinstance(mapped_status, bool):
            return True
        if source.get("api_call_start_time") is not None:
            return True
    return False


def _alibaba_provider_call_was_sent(
    *,
    response: Any = None,
    error: Optional[BaseException] = None,
) -> bool:
    if error is not None and _confirmed_pre_egress_failure(error):
        return False
    if response is not None and _provider_send_provenance(response):
        return True
    if error is not None and _provider_send_provenance(error):
        return True
    # A completion object is itself the send result. Local exceptions are not.
    return response is not None and error is None


def begin_alibaba_ciphertext_subattempt(*, ordinal: int) -> float:
    """Open one Alibaba generation before provider I/O is known."""

    started = time.monotonic()
    attempt_record = _alibaba_attempt_record()
    if attempt_record is None:
        return started
    _append_alibaba_ciphertext_subattempt(
        attempt_record,
        {
            "ordinal": ordinal,
            "kind": "alibaba_ciphertext_generation",
            "role": "ciphertext_repair" if ordinal > 1 else "initial",
            "status": "in_flight",
            "outcome": "in_flight",
            "attempted_provider_call": False,
            "started_at_monotonic": round(started, 6),
            "retry_eligible": False,
        },
    )
    return started


def finish_alibaba_ciphertext_subattempt(
    started_at: float,
    *,
    ordinal: int,
    outcome: str,
    retry_eligible: bool,
    response: Any = None,
    error: Optional[BaseException] = None,
    attempted_provider_call: Optional[bool] = None,
    error_class: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Close one Alibaba generation with a terminal outcome and usage."""

    attempt_record = _alibaba_attempt_record()
    if attempt_record is None:
        return None
    if attempted_provider_call is None:
        attempted_provider_call = _alibaba_provider_call_was_sent(
            response=response,
            error=error,
        )
    ended = time.monotonic()
    payload: dict[str, Any] = {
        "ordinal": ordinal,
        "kind": "alibaba_ciphertext_generation",
        "role": "ciphertext_repair" if ordinal > 1 else "initial",
        "status": outcome,
        "outcome": outcome,
        "attempted_provider_call": attempted_provider_call,
        "started_at_monotonic": round(float(started_at), 6),
        "ended_at_monotonic": round(ended, 6),
        "duration_seconds": round(max(0.0, ended - float(started_at)), 3),
        "retry_eligible": bool(retry_eligible),
    }
    usage = _coerce_usage_mapping(response)
    if usage is not None:
        payload["usage"] = usage
    if error_class:
        payload["error_class"] = error_class
    return _append_alibaba_ciphertext_subattempt(attempt_record, payload)


def _settle_outstanding_alibaba_ciphertext_subattempt(
    attempt_record: dict[str, Any],
) -> None:
    subattempts = attempt_record.get("subattempts")
    if not isinstance(subattempts, list) or not subattempts:
        return
    current = subattempts[-1]
    if (
        not isinstance(current, dict)
        or current.get("kind") != "alibaba_ciphertext_generation"
        or current.get("outcome") != "in_flight"
    ):
        return
    ordinal = current.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        return
    started = current.get("started_at_monotonic")
    if not isinstance(started, (int, float)):
        started = time.monotonic()
    finish_alibaba_ciphertext_subattempt(
        float(started),
        ordinal=ordinal,
        outcome="cancelled",
        retry_eligible=False,
        error_class="cancelled",
    )


def reset_alibaba_ciphertext_repair_sink(
    token: Token[Optional[dict[str, Any]]],
    attempt_record: Optional[dict[str, Any]] = None,
) -> None:
    if attempt_record is not None:
        _settle_outstanding_alibaba_ciphertext_subattempt(attempt_record)
    _ALIBABA_CIPHERTEXT_REPAIR_SINK.reset(token)


def alibaba_ciphertext_repair_prohibited(request: Any) -> bool:
    """Repair cannot start after a downstream response has been committed."""

    state = getattr(request, "state", None)
    if (
        state is not None
        and getattr(state, _ALIBABA_DOWNSTREAM_COMMITTED_STATE_KEY, False) is True
    ):
        return True
    attempt_record = _alibaba_attempt_record()
    return (
        isinstance(attempt_record, dict)
        and attempt_record.get("downstream_response_committed") is True
    )


def mark_alibaba_downstream_response_committed(request: Any) -> None:
    """Record that a downstream response has been handed to the caller."""

    state = getattr(request, "state", None)
    if state is not None:
        try:
            setattr(state, _ALIBABA_DOWNSTREAM_COMMITTED_STATE_KEY, True)
        except Exception:
            pass
    attempt_record = _alibaba_attempt_record()
    if isinstance(attempt_record, dict):
        attempt_record["downstream_response_committed"] = True
        attempt_record["ciphertext_repair_retry_eligible"] = False


def note_alibaba_ciphertext_repair_blocked() -> None:
    attempt_record = _alibaba_attempt_record()
    if not isinstance(attempt_record, dict):
        return
    attempt_record["ciphertext_repair_blocked"] = True
    attempt_record["ciphertext_repair_retry_eligible"] = False


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


def _get_xai_reasoning_effort_ceiling(
    resolved_route: dict[str, Any],
) -> Optional[str]:
    """Return the supported reasoning-effort ceiling for the managed xAI route.

    XAI-008: capability-recognition only. Capability is read from the model
    catalog via the config-driven ``supports_xhigh_reasoning_effort`` flag,
    never from the model name. Prefer the effective runtime map, then fall
    back to the same repository/bundled catalog used by request preparation.
    """
    assert _get_model_info is not None
    assert _model_cost is not None
    assert _load_bundled_model_cost_map_for_codex_policy is not None

    model = resolved_route.get("model")
    if not isinstance(model, str) or not model:
        return None
    supports_xhigh = _resolve_supports_xhigh_reasoning_effort()
    if supports_xhigh is not None:
        try:
            if supports_xhigh(model=model, custom_llm_provider="xai") is True:
                return "xhigh"
        except Exception:
            pass

    model_info_sources: list[Mapping[str, Any]] = []
    try:
        resolved_model_info = _get_model_info(
            model=model,
            custom_llm_provider="xai",
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

    if any(
        model_info.get("supports_xhigh_reasoning_effort") is True
        for model_info in model_info_sources
    ):
        return "xhigh"
    return None


def _build_xai_reasoning_effort_metadata(
    request_body: dict[str, Any],
    *,
    candidate: dict[str, Any],
    attempt_number: Optional[int] = None,
) -> dict[str, Any]:
    """Build audit-only native reasoning-effort metadata for managed xAI routes.

    XAI-008: the xAI Responses adapter never clamps and never rewrites the
    caller's effort on the request body, so the shared clamping normalizer is
    a strict no-op for this route. When the caller effort is within the
    model's catalog-advertised ceiling, the standard reasoning metadata keys
    are still emitted (native provider ``xai``, native field
    ``reasoning.effort``) so audit, session-history, and rollup consumers see
    the requested/native value instead of empty metadata.
    """
    if (
        candidate.get("provider") != "xai"
        or candidate.get("route_family") != "codex_xai_oauth_responses_adapter"
    ):
        return {}
    requested_effort, native_field = _extract_codex_reasoning_effort(request_body)
    if requested_effort not in _CODEX_REASONING_EFFORT_TIER_INDEX or native_field is None:
        return {}
    supported_ceiling = _get_xai_reasoning_effort_ceiling(candidate)
    if supported_ceiling is None:
        return {}
    if _CODEX_REASONING_EFFORT_TIER_INDEX[requested_effort] > _CODEX_REASONING_EFFORT_TIER_INDEX[supported_ceiling]:
        return {}
    metadata: dict[str, Any] = {
        "reasoning_effort_requested": requested_effort,
        "reasoning_effort_source": native_field,
        "reasoning_effort_native_provider": "xai",
        "reasoning_effort_native_value": requested_effort,
        "reasoning_effort_native_field": native_field,
        "reasoning_effort_supported_ceiling": supported_ceiling,
        "reasoning_effort_resolved_model": str(candidate["model"]),
        "reasoning_effort_resolved_provider": "xai",
        "reasoning_effort_mapping_reason": "within_supported_ceiling",
    }
    if attempt_number is not None:
        metadata["reasoning_effort_candidate_attempt"] = attempt_number
    return metadata


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
    inbound_metadata = updated_body.get("litellm_metadata")
    if isinstance(inbound_metadata, dict):
        inbound_metadata.pop("codex_auto_agent_selected_account_display", None)
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
    if not reasoning_effort_metadata:
        # XAI-008: the shared normalizer is a strict no-op for the managed xAI
        # Responses route (no clamping, body preserved). Surface audit-only
        # native reasoning metadata when the caller effort is within the
        # model's catalog-advertised ceiling.
        reasoning_effort_metadata = _build_xai_reasoning_effort_metadata(
            updated_body,
            candidate=candidate,
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
        for key in (
            "request_mode",
            "redispatch_ordinal",
            "affinity_bypassed",
            "has_account_bound_state",
            "account_bound_classification",
        )
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
    canonical_thread_id = _extract_auto_agent_alias_canonical_thread_id(
        request,
        updated_body,
    )
    parent_thread_id = _extract_auto_agent_alias_parent_thread_id(
        request,
        updated_body,
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
            "codex_auto_agent_selected_priority": candidate.get("selection_priority"),
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
            **(
                reasoning_effort_metadata
                if (
                    candidate.get("provider") == "xai"
                    and candidate.get("route_family") == "codex_xai_oauth_responses_adapter"
                    and reasoning_effort_metadata
                )
                else {}
            ),
            "codex_auto_agent_selection_reason": selection.get("selection_reason"),
            "codex_auto_agent_affinity_state_source": selection.get("affinity_state_source"),
            "canonical_session_identity": selection.get("canonical_session_identity"),
            "canonical_thread_id": canonical_thread_id,
            "parent_thread_id": parent_thread_id,
            "session_owner_decision": selection.get("session_owner_decision"),
            "session_owner_id": selection.get("session_owner_id"),
            "session_owner_mismatch_reason": selection.get("session_owner_mismatch_reason"),
            "codex_auto_agent_cooldown_state_source": selection.get("cooldown_state_source"),
            "codex_auto_agent_lane_key": selection.get("lane_key"),
            "codex_auto_agent_request_mode": selection.get("request_mode"),
            "codex_auto_agent_redispatch_ordinal": selection.get("redispatch_ordinal"),
            "codex_auto_agent_affinity_bypassed": selection.get("affinity_bypassed"),
            "has_account_bound_state": selection.get("has_account_bound_state"),
            "account_bound_classification": selection.get(
                "account_bound_classification"
            ),
            "aawm_selected_account_label": (
                candidate.get("xai_oauth_account_label")
                or candidate.get("codex_oauth_account_label")
            ),
            "aawm_selected_account_hash": (
                candidate.get("xai_oauth_account_hash")
                or candidate.get("codex_oauth_account_hash")
            ),
            "aawm_selected_account_lane": (
                candidate.get("xai_oauth_lane_key")
                or candidate.get("codex_oauth_lane_key")
            ),
            "codex_oauth_inventory_generation": candidate.get(
                "codex_oauth_inventory_generation"
            ),
            "aawm_selected_account_scope": candidate.get(
                "xai_oauth_scope_identity"
            ),
            "aawm_selected_account_provider": (
                "xai_oauth"
                if candidate.get("xai_oauth_account_hash")
                else (
                    "codex_oauth"
                    if candidate.get("codex_oauth_account_hash")
                    else None
                )
            ),
            "codex_auto_agent_selected_account_label": candidate.get(
                "codex_oauth_account_label"
            ),
            "codex_auto_agent_selected_account_hash": candidate.get(
                "codex_oauth_account_hash"
            ),
            "codex_auto_agent_selected_account_lane": candidate.get(
                "codex_oauth_lane_key"
            ),
            **(
                {
                    "codex_auto_agent_selected_account_display": candidate.get(
                        "codex_oauth_account_display"
                    )
                }
                if (
                    candidate.get("provider") == CODEX_AUTO_AGENT_NATIVE_PROVIDER
                    and candidate.get("codex_oauth_account_display") is not None
                )
                else {}
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
        for key in (
            "request_mode",
            "redispatch_ordinal",
            "affinity_bypassed",
            "has_account_bound_state",
            "account_bound_classification",
        )
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
    canonical_thread_id = _extract_auto_agent_alias_canonical_thread_id(
        request,
        updated_body,
    )
    parent_thread_id = _extract_auto_agent_alias_parent_thread_id(
        request,
        updated_body,
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
            "anthropic_auto_agent_selected_priority": candidate.get("selection_priority"),
            "anthropic_auto_agent_selected_last_resort": bool(candidate.get("last_resort")),
            **(
                {"anthropic_auto_agent_config_reasoning_effort": configured_reasoning_effort}
                if configured_reasoning_effort
                else {}
            ),
            "anthropic_auto_agent_selection_reason": selection.get("selection_reason"),
            "anthropic_auto_agent_affinity_state_source": selection.get("affinity_state_source"),
            "canonical_session_identity": selection.get("canonical_session_identity"),
            "canonical_thread_id": canonical_thread_id,
            "parent_thread_id": parent_thread_id,
            "session_owner_decision": selection.get("session_owner_decision"),
            "session_owner_id": selection.get("session_owner_id"),
            "session_owner_mismatch_reason": selection.get("session_owner_mismatch_reason"),
            "anthropic_auto_agent_cooldown_state_source": selection.get("cooldown_state_source"),
            "anthropic_auto_agent_lane_key": selection.get("lane_key"),
            "anthropic_auto_agent_request_mode": selection.get("request_mode"),
            "anthropic_auto_agent_redispatch_ordinal": selection.get("redispatch_ordinal"),
            "anthropic_auto_agent_affinity_bypassed": selection.get("affinity_bypassed"),
            "has_account_bound_state": selection.get("has_account_bound_state"),
            "account_bound_classification": selection.get(
                "account_bound_classification"
            ),
            "aawm_selected_account_label": (
                candidate.get("xai_oauth_account_label")
                or candidate.get("codex_oauth_account_label")
            ),
            "aawm_selected_account_hash": (
                candidate.get("xai_oauth_account_hash")
                or candidate.get("codex_oauth_account_hash")
            ),
            "aawm_selected_account_lane": (
                candidate.get("xai_oauth_lane_key")
                or candidate.get("codex_oauth_lane_key")
            ),
            "aawm_selected_account_scope": candidate.get(
                "xai_oauth_scope_identity"
            ),
            "aawm_selected_account_provider": (
                "xai_oauth"
                if candidate.get("xai_oauth_account_hash")
                else (
                    "codex_oauth"
                    if candidate.get("codex_oauth_account_hash")
                    else None
                )
            ),
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
