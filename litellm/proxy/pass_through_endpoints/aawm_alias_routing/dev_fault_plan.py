"""Dev-gated request-local fault plans for managed OpenAI attempt boundaries."""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import HTTPException, Request

from litellm.proxy import aawm_route_logging
from litellm.proxy.common_utils.http_parsing_utils import (
    _safe_get_request_headers,
    _safe_get_request_parsed_body,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _merge_litellm_metadata,
)

from . import attempt_records as _attempt_records
from . import audit_build as _audit_build
from . import audit_persist as _audit_persist
from . import selection as _selection

_OPENAI_FAULT_PLAN_ENABLED_ENV = "AAWM_OPENAI_FAULT_PLAN_ENABLED"
_OPENAI_FAULT_PLAN_ENVIRONMENT_ENV = "AAWM_LITELLM_ENVIRONMENT"
_OPENAI_FAULT_PLAN_ENVIRONMENT_VALUE = "litellm-dev"
_OPENAI_FAULT_PLAN_HEADER = "x-aawm-openai-fault-plan"
_OPENAI_FAULT_PLAN_STATE_KEY = "aawm_openai_fault_plan"
_OPENAI_FAULT_PLAN_SLOT_INDEX_STATE_KEY = "aawm_openai_fault_plan_slot_index"
_OPENAI_FAULT_PLAN_INJECTED_COUNT_STATE_KEY = (
    "aawm_openai_fault_plan_injected_count"
)
_OPENAI_FAULT_PLAN_DIRECT_TRACKING_STATE_KEY = (
    "aawm_openai_fault_plan_direct_tracking"
)
_OPENAI_FAULT_PLAN_DIRECT_ATTEMPT_STATE_KEY = (
    "aawm_openai_fault_plan_direct_attempt"
)
_OPENAI_FAULT_PLAN_DIRECT_SUCCESS_STATE_KEY = (
    "aawm_openai_fault_plan_direct_success_recorded"
)
_OPENAI_FAULT_PLAN_DIRECT_TERMINAL_STATE_KEY = (
    "aawm_openai_fault_plan_direct_terminal_recorded"
)


def _openai_fault_plan_control_enabled() -> bool:
    if os.getenv(_OPENAI_FAULT_PLAN_ENABLED_ENV, "").strip() != "1":
        return False
    return (
        os.getenv(_OPENAI_FAULT_PLAN_ENVIRONMENT_ENV, "").strip()
        == _OPENAI_FAULT_PLAN_ENVIRONMENT_VALUE
    )


class AawmOpenAIFaultPlanError(HTTPException):
    """Typed synthetic usage-limit response consumed by existing classifiers."""

    def __init__(self) -> None:
        super().__init__(
            status_code=429,
            detail={
                "error": {
                    "message": "Managed OpenAI account usage limit reached.",
                    "type": "rate_limit_error",
                    "code": "usage_limit_reached",
                },
                "quota": {"resets_in_seconds": 1.0},
                "failover_disposition": "usage_limit_reached",
            },
            headers={"Retry-After": "1"},
        )


def _get_openai_fault_plan_header(request: Request) -> Optional[str]:
    for key, value in _safe_get_request_headers(request).items():
        if (
            isinstance(key, str)
            and key.lower() == _OPENAI_FAULT_PLAN_HEADER
            and isinstance(value, str)
        ):
            return value
    return None


def _openai_fault_plan_request_present(request: Request) -> bool:
    return bool(_resolve_openai_fault_plan(request))


def _resolve_openai_fault_plan(request: Request) -> tuple[str, ...]:
    """Resolve one exact authorized plan into request-local state."""
    request_state = getattr(request, "state", None)
    if request_state is None:
        return ()
    existing = getattr(request_state, _OPENAI_FAULT_PLAN_STATE_KEY, None)
    if existing in (("fail", "success"), ("fail", "fail")):
        return existing

    if not _openai_fault_plan_control_enabled():
        return ()
    raw = _get_openai_fault_plan_header(request)
    if raw == "fail,success":
        plan = ("fail", "success")
    elif raw == "fail,fail":
        plan = ("fail", "fail")
    else:
        return ()
    setattr(request_state, _OPENAI_FAULT_PLAN_STATE_KEY, plan)
    return plan


def _claim_openai_fault_plan_slot(request: Request) -> Optional[str]:
    plan = _resolve_openai_fault_plan(request)
    if not plan:
        return None
    request_state = getattr(request, "state", None)
    if request_state is None:
        return None
    index = getattr(
        request_state,
        _OPENAI_FAULT_PLAN_SLOT_INDEX_STATE_KEY,
        0,
    )
    if not isinstance(index, int) or index < 0:
        index = 0
    if index >= len(plan):
        return None
    setattr(
        request_state,
        _OPENAI_FAULT_PLAN_SLOT_INDEX_STATE_KEY,
        index + 1,
    )
    return plan[index]


def _managed_openai_oauth_candidate(
    candidate: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not _selection._is_codex_oauth_account_candidate(candidate):
        return None
    assert isinstance(candidate, dict)
    return candidate


def _note_injected_failure(request: Request) -> None:
    request_state = getattr(request, "state", None)
    if request_state is None:
        return
    count = getattr(
        request_state,
        _OPENAI_FAULT_PLAN_INJECTED_COUNT_STATE_KEY,
        0,
    )
    if not isinstance(count, int) or count < 0:
        count = 0
    setattr(
        request_state,
        _OPENAI_FAULT_PLAN_INJECTED_COUNT_STATE_KEY,
        count + 1,
    )


def _authorized_plan_injected_failure(request: Request) -> bool:
    request_state = getattr(request, "state", None)
    if request_state is None:
        return False
    count = getattr(
        request_state,
        _OPENAI_FAULT_PLAN_INJECTED_COUNT_STATE_KEY,
        0,
    )
    return isinstance(count, int) and count > 0


def _raise_if_openai_fault_plan_slot_fails(
    request: Request,
    *,
    candidate: Optional[dict[str, Any]],
) -> None:
    """Inject only at a selected managed OpenAI OAuth account boundary."""
    if (
        _managed_openai_oauth_candidate(candidate) is None
        or not _openai_fault_plan_request_present(request)
    ):
        return
    slot = _claim_openai_fault_plan_slot(request)
    if slot == "fail":
        _note_injected_failure(request)
        raise AawmOpenAIFaultPlanError()


def _direct_alias_model(
    request_body: dict[str, Any],
    selection: dict[str, Any],
) -> str:
    candidate = selection.get("candidate")
    candidate_model = (
        candidate.get("model") if isinstance(candidate, dict) else None
    )
    return str(
        selection.get("alias_model")
        or request_body.get("model")
        or candidate_model
        or "codex_native"
    )


def _direct_attempts(request: Request) -> list[dict[str, Any]]:
    outcome = _attempt_records._auto_agent_alias_request_outcome_state(request)
    attempts = outcome.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
        outcome["attempts"] = attempts
    return attempts


def _add_direct_openai_managed_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    alias_model = _direct_alias_model(request_body, selection)
    current_body = _safe_get_request_parsed_body(request)
    metadata_body = (
        current_body if isinstance(current_body, dict) else request_body
    )
    audit_events = _audit_build._build_auto_agent_alias_audit_events(
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=metadata_body,
        selection=selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        metadata_body,
        extra_fields={
            "codex_auto_agent_attempts": attempts,
            "codex_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )


def _start_direct_tracking(request: Request) -> None:
    request_state = getattr(request, "state", None)
    if request_state is None or getattr(
        request_state,
        _OPENAI_FAULT_PLAN_DIRECT_TRACKING_STATE_KEY,
        False,
    ):
        return
    setattr(
        request_state,
        _OPENAI_FAULT_PLAN_DIRECT_TRACKING_STATE_KEY,
        True,
    )
    aawm_route_logging.register_aawm_route_rollup_access_log_replacement(
        request
    )
    _attempt_records._bind_auto_agent_alias_request_identity(request)
    outcome = _attempt_records._auto_agent_alias_request_outcome_state(request)
    outcome["attempts"] = []


def _new_direct_attempt_record(
    request: Request,
    *,
    selection: dict[str, Any],
) -> dict[str, Any]:
    candidate = selection["candidate"]
    attempt_record = _selection._codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=selection.get("lane_key"),
        reason=selection.get("selection_reason"),
    )
    for field in (
        "quota_snapshot_age_seconds",
        "quota_windows",
        "failover_ordinal",
        "prior_account_outcome",
        "terminal_reset",
    ):
        value = selection.get(field)
        if value is not None:
            attempt_record[field] = value
    _direct_attempts(request).append(attempt_record)
    setattr(
        request.state,
        _OPENAI_FAULT_PLAN_DIRECT_ATTEMPT_STATE_KEY,
        attempt_record,
    )
    return attempt_record


def _current_direct_attempt(
    request: Request,
) -> Optional[dict[str, Any]]:
    request_state = getattr(request, "state", None)
    if request_state is None:
        return None
    attempt_record = getattr(
        request_state,
        _OPENAI_FAULT_PLAN_DIRECT_ATTEMPT_STATE_KEY,
        None,
    )
    return attempt_record if isinstance(attempt_record, dict) else None


def note_direct_openai_managed_attempt(
    request: Request,
    request_body: dict[str, Any],
    *,
    selection: Optional[dict[str, Any]],
) -> None:
    """Start one direct managed attempt only when an authorized slot exists."""
    if not isinstance(selection, dict):
        return
    candidate = _managed_openai_oauth_candidate(selection.get("candidate"))
    if candidate is None or not _openai_fault_plan_request_present(request):
        return
    slot = _claim_openai_fault_plan_slot(request)
    if slot is None:
        return

    _start_direct_tracking(request)
    attempt_record = _new_direct_attempt_record(
        request,
        selection=selection,
    )
    _attempt_records._record_auto_agent_alias_attempt_started(
        alias_family="codex_auto_agent",
        alias_model=_direct_alias_model(request_body, selection),
        request=request,
        prepared_request_body=request_body,
        selection=selection,
        attempts=_direct_attempts(request),
        attempt_record=attempt_record,
        add_alias_metadata_fn=_add_direct_openai_managed_metadata,
    )
    if slot == "fail":
        _note_injected_failure(request)
        raise AawmOpenAIFaultPlanError()


def update_direct_openai_managed_failure_attempt(
    request: Request,
    request_body: dict[str, Any],
    *,
    selection: dict[str, Any],
    exc: Exception,
    cooldown_seconds: float,
) -> Optional[dict[str, Any]]:
    """Apply the existing managed-attempt classifier fields to an injection."""
    if not isinstance(exc, AawmOpenAIFaultPlanError):
        return None
    attempt_record = _current_direct_attempt(request)
    candidate = _managed_openai_oauth_candidate(selection.get("candidate"))
    if attempt_record is None or candidate is None:
        return None
    _attempt_records._update_codex_auto_agent_retryable_attempt_record(
        attempt_record=attempt_record,
        exc=exc,
        error_class="usage_limit_reached",
        cooldown_seconds=cooldown_seconds,
        cooldown_scope="candidate",
        alias_model=_direct_alias_model(request_body, selection),
        candidate=candidate,
    )
    return attempt_record


def note_direct_openai_managed_failure(
    request: Request,
    request_body: dict[str, Any],
    *,
    selection: dict[str, Any],
    attempt_record: Optional[dict[str, Any]],
) -> None:
    if attempt_record is None or not _authorized_plan_injected_failure(request):
        return
    _attempt_records._record_auto_agent_alias_attempt_failure(
        alias_family="codex_auto_agent",
        alias_model=_direct_alias_model(request_body, selection),
        request=request,
        prepared_request_body=request_body,
        selection=selection,
        attempts=_direct_attempts(request),
        attempt_record=attempt_record,
        error_class="usage_limit_reached",
        add_alias_metadata_fn=_add_direct_openai_managed_metadata,
    )


def note_direct_openai_managed_success(
    request: Request,
    request_body: dict[str, Any],
    *,
    selection: Optional[dict[str, Any]],
) -> None:
    request_state = getattr(request, "state", None)
    if (
        request_state is None
        or not isinstance(selection, dict)
        or not _authorized_plan_injected_failure(request)
        or getattr(
            request_state,
            _OPENAI_FAULT_PLAN_DIRECT_SUCCESS_STATE_KEY,
            False,
        )
    ):
        return
    attempt_record = _current_direct_attempt(request)
    if attempt_record is None:
        return
    setattr(
        request_state,
        _OPENAI_FAULT_PLAN_DIRECT_SUCCESS_STATE_KEY,
        True,
    )
    attempt_record["attempted_provider_call"] = True
    _attempt_records._record_auto_agent_alias_attempt_success(
        alias_family="codex_auto_agent",
        alias_model=_direct_alias_model(request_body, selection),
        request=request,
        prepared_request_body=request_body,
        selection=selection,
        attempts=_direct_attempts(request),
        attempt_record=attempt_record,
        add_alias_metadata_fn=_add_direct_openai_managed_metadata,
    )


def note_direct_openai_managed_terminal_exhaustion(
    request: Request,
    request_body: dict[str, Any],
    *,
    selection: Optional[dict[str, Any]],
) -> None:
    request_state = getattr(request, "state", None)
    if (
        request_state is None
        or not isinstance(selection, dict)
        or not _authorized_plan_injected_failure(request)
        or getattr(
            request_state,
            _OPENAI_FAULT_PLAN_DIRECT_TERMINAL_STATE_KEY,
            False,
        )
    ):
        return
    attempt_record = _current_direct_attempt(request)
    candidate = _managed_openai_oauth_candidate(selection.get("candidate"))
    if attempt_record is None or candidate is None:
        return
    setattr(
        request_state,
        _OPENAI_FAULT_PLAN_DIRECT_TERMINAL_STATE_KEY,
        True,
    )
    _attempt_records._mark_auto_agent_alias_request_terminal_failure(
        request,
        attempt_record,
    )
    attempts = _direct_attempts(request)
    current_body = _safe_get_request_parsed_body(request)
    metadata_body = (
        current_body if isinstance(current_body, dict) else request_body
    )
    event = _audit_build._build_auto_agent_alias_audit_event(
        alias_family="codex_auto_agent",
        alias_model=_direct_alias_model(request_body, selection),
        request=request,
        request_body=metadata_body,
        selection=selection,
        candidate=attempt_record,
        event_type="no_candidate_available",
        candidate_status="all_candidates_unavailable",
        attempt_number=len(attempts),
        selected=True,
        selection_reason=selection.get("selection_reason"),
        lane_key=selection.get("lane_key"),
        cooldown_key=selection.get("cooldown_key"),
        cooldown_seconds=attempt_record.get("cooldown_seconds"),
        cooldown_scope=attempt_record.get("cooldown_scope"),
        failure_class=attempt_record.get("error_class"),
        error_status_code=attempt_record.get("error_status_code"),
        error_type=attempt_record.get("error_type"),
        error_code=attempt_record.get("error_code"),
        error_tokens=attempt_record.get("error_tokens"),
        source_error=attempt_record.get("source_error"),
        retry_after_seconds=attempt_record.get("retry_after_seconds"),
        failure_phase=attempt_record.get("failure_phase"),
        attempted_provider_call=attempt_record.get("attempted_provider_call"),
    )
    event["attempt_count"] = len(attempts)
    event["attempts"] = [dict(attempt) for attempt in attempts]
    event["candidates"] = [dict(attempt) for attempt in attempts]
    event["request_outcome"] = "failed"
    _attempt_records._stamp_auto_agent_alias_request_identity(
        request=request,
        target=event,
    )
    _audit_persist._emit_auto_agent_alias_route_event(
        event,
        level="warning",
    )
    _audit_persist._persist_auto_agent_alias_audit_only_events_best_effort(
        [event],
        request_body=metadata_body,
    )


__all__ = [
    "AawmOpenAIFaultPlanError",
    "_claim_openai_fault_plan_slot",
    "_get_openai_fault_plan_header",
    "_openai_fault_plan_control_enabled",
    "_raise_if_openai_fault_plan_slot_fails",
    "_resolve_openai_fault_plan",
    "note_direct_openai_managed_attempt",
    "note_direct_openai_managed_failure",
    "note_direct_openai_managed_success",
    "note_direct_openai_managed_terminal_exhaustion",
    "update_direct_openai_managed_failure_attempt",
]
