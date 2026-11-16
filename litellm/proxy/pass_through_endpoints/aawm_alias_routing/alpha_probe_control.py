"""Fail-closed control and receipts for the alpha OpenAI traversal probe.

This module deliberately contains no route registration, provider calls, or
shared-state mutation.  A route owner may resolve an authenticated,
server-approved plan once and then use the immutable control on that request.
Receipts are request-local and contain only occurrence-level, secret-safe
metadata.  They never represent a provider response or a provider send.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx
from fastapi import HTTPException, Request

from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers

ALPHA_PROBE_ENVIRONMENT_ENV = "AAWM_LITELLM_ENVIRONMENT"
ALPHA_PROBE_ENVIRONMENT = "litellm-alpha"
ALPHA_PROBE_ENABLED_ENV = "AAWM_OPENAI_ALPHA_PROBE_ENABLED"
ALPHA_PROBE_PLAN_HEADER = "x-aawm-openai-alpha-probe-plan"

ALPHA_OVERLOAD_PROBE_ENABLED_ENV = "AAWM_OPENAI_ALPHA_OVERLOAD_PROBE_ENABLED"
ALPHA_OVERLOAD_PROBE_PLAN_HEADER = "x-aawm-openai-alpha-overload-probe-plan"

_CONTROL_STATE_KEY = "aawm_openai_alpha_probe_control"
_RECEIPTS_STATE_KEY = "aawm_openai_alpha_probe_receipts"
_RECEIPT_SEQUENCE_STATE_KEY = "aawm_openai_alpha_probe_receipt_sequence"
_REQUEST_IDENTITY_STATE_KEY = "aawm_alias_request_litellm_call_id"
_RECEIPT_SCHEMA = "aawm_openai_alpha_probe_receipt_v1"
_OVERLOAD_CONTROL_STATE_KEY = "aawm_openai_alpha_overload_probe_control"
_OVERLOAD_SLOT_STATE_KEY = "aawm_openai_alpha_overload_probe_slot_index"
_OVERLOAD_RECEIPTS_STATE_KEY = "aawm_openai_alpha_overload_probe_receipts"
_OVERLOAD_RECEIPT_SEQUENCE_STATE_KEY = (
    "aawm_openai_alpha_overload_probe_receipt_sequence"
)
_OVERLOAD_RECEIPT_SCHEMA = "aawm_openai_alpha_overload_probe_receipt_v1"
_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_.:/@+-]+")
_HEX_RE = re.compile(r"\A[0-9A-Fa-f]{8,128}\Z")

_ALPHA_PLAN_DEFINITIONS: dict[str, tuple[int, str]] = {
    # Basic traversal may inject at each reached ordinary candidate, but the
    # cap remains finite and is enforced by the caller that claims slots.
    "basic": (16, "ordinary_candidates_then_openai_last_resort"),
    "work": (1, "selected_openai_account_then_real_account_failover"),
}
ALPHA_PROBE_PLAN_NAMES = frozenset(_ALPHA_PLAN_DEFINITIONS)
_ALPHA_OVERLOAD_PLAN_DEFINITIONS: dict[str, tuple[str, ...]] = {
    # The first three slots cover the existing logical retry budget.  Each
    # claimed overload remains one synthetic SSE response; after the final
    # slot, a terminal plan fails closed rather than falling through.  The
    # explicit two-slot plan returns control for one real recovery attempt.
    "overload": ("overload",),
    "overload,overload,overload": ("overload", "overload", "overload"),
    "overload,success": ("overload", "success"),
}
ALPHA_OVERLOAD_PROBE_PLAN_NAMES = frozenset(_ALPHA_OVERLOAD_PLAN_DEFINITIONS)
ALPHA_OVERLOAD_PROBE_ERROR_CODES = frozenset(
    {"server_is_overloaded", "service_unavailable"}
)


@dataclass(frozen=True)
class AlphaProbePlan:
    """One server-approved finite alpha probe plan."""

    name: str
    max_injections: int
    disposition: str
    failure_class: str = "candidate_unavailable"
    synthetic: bool = True
    attempted_provider_call: bool = False
    provider_returned: bool = False

    def __post_init__(self) -> None:
        if self.name not in ALPHA_PROBE_PLAN_NAMES:
            raise ValueError("unsupported alpha probe plan")
        if self.max_injections < 1:
            raise ValueError("alpha probe plan must be finite and non-empty")
        if self.failure_class != "candidate_unavailable":
            raise ValueError("alpha probe plan has an unsupported failure class")
        if not self.synthetic:
            raise ValueError("alpha probe observations must be synthetic")
        if self.attempted_provider_call or self.provider_returned:
            raise ValueError("alpha probe observations must be no-I/O")


@dataclass(frozen=True)
class AlphaProbeControl:
    """Request-local authorization and immutable plan state."""

    plan: AlphaProbePlan
    environment: str = ALPHA_PROBE_ENVIRONMENT
    authorization: str = "cfg004_user_api_key_auth"
    request_identity: Optional[str] = None


@dataclass(frozen=True)
class AlphaOverloadProbePlan:
    """One server-approved finite synthetic SSE overload plan."""

    actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.actions not in _ALPHA_OVERLOAD_PLAN_DEFINITIONS.values():
            raise ValueError("unsupported alpha overload probe plan")

    @property
    def is_terminal(self) -> bool:
        """Return whether exhaustion must block any ordinary provider send."""
        return self.actions == ("overload", "overload", "overload")


@dataclass(frozen=True)
class AlphaOverloadProbeControl:
    """Request-local authorization and immutable synthetic plan state."""

    plan: AlphaOverloadProbePlan
    environment: str = ALPHA_PROBE_ENVIRONMENT
    authorization: str = "cfg004_user_api_key_auth"
    request_identity: Optional[str] = None


def alpha_probe_control_enabled() -> bool:
    """Return whether the exact server-side alpha gate is open."""
    return (
        os.getenv(ALPHA_PROBE_ENABLED_ENV, "").strip() == "1"
        and os.getenv(ALPHA_PROBE_ENVIRONMENT_ENV, "").strip()
        == ALPHA_PROBE_ENVIRONMENT
    )


def alpha_overload_probe_control_enabled() -> bool:
    """Return whether the exact server-side synthetic-overload gate is open."""
    return (
        os.getenv(ALPHA_OVERLOAD_PROBE_ENABLED_ENV, "").strip() == "1"
        and os.getenv(ALPHA_PROBE_ENVIRONMENT_ENV, "").strip()
        == ALPHA_PROBE_ENVIRONMENT
    )


def _request_state(request: Any) -> Any:
    return getattr(request, "state", None)


def _request_header(request: Any, name: str) -> Optional[str]:
    try:
        headers = _safe_get_request_headers(request)
    except Exception:
        return None
    for key, value in headers.items():
        if (
            isinstance(key, str)
            and key.casefold() == name.casefold()
            and isinstance(value, str)
        ):
            return value
    return None


def _safe_text(value: Any, *, max_length: int = 160) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return _IDENTIFIER_RE.sub("_", text)[:max_length] or None


def _safe_identifier(value: Any) -> Optional[str]:
    text = _safe_text(value, max_length=160)
    if text is None:
        return None
    return text


def _safe_account_hash(value: Any) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if len(text) > 128 or _HEX_RE.fullmatch(text) is None:
        return None
    return text


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _request_identity(request: Any) -> Optional[str]:
    state = _request_state(request)
    if state is None:
        return None
    return _safe_identifier(getattr(state, _REQUEST_IDENTITY_STATE_KEY, None))


def _resolve_plan(raw_plan: str) -> AlphaProbePlan:
    # Do not normalize whitespace or arbitrary casing: the control value must
    # be one of the exact server-approved finite names.
    definition = _ALPHA_PLAN_DEFINITIONS.get(raw_plan)
    if definition is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_alpha_probe_plan",
                "message": "alpha probe plan is not server-approved",
            },
        )
    max_injections, disposition = definition
    return AlphaProbePlan(
        name=raw_plan,
        max_injections=max_injections,
        disposition=disposition,
    )


def _resolve_overload_plan(raw_plan: str) -> AlphaOverloadProbePlan:
    # Header values are server control values, not user-facing aliases.
    # Do not normalize whitespace, casing, or unknown suffixes.
    actions = _ALPHA_OVERLOAD_PLAN_DEFINITIONS.get(raw_plan)
    if actions is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_alpha_overload_probe_plan",
                "message": "alpha overload probe plan is not server-approved",
            },
        )
    return AlphaOverloadProbePlan(actions=actions)


def _check_cfg004_admin_auth(user_api_key_dict: Any) -> None:
    """Reuse CFG-004's authenticated PROXY_ADMIN/master-key boundary."""
    try:
        from .cooldown_clear import _check_admin_auth

        _check_admin_auth(user_api_key_dict)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "alpha_probe_auth_unavailable",
                "message": "alpha probe authentication unavailable",
            },
        ) from exc


def resolve_alpha_probe_control(
    request: Request,
    *,
    user_api_key_dict: Any = None,
) -> Optional[AlphaProbeControl]:
    """Resolve and cache one authorized request-local alpha probe control.

    A missing gate, missing plan header, or non-alpha environment returns
    ``None`` and leaves the request on its ordinary path.  Once the exact gate
    and header are present, every failure is fail-closed: invalid plans and
    missing/invalid CFG-004 authentication raise without producing a control.
    """
    if not alpha_probe_control_enabled():
        return None

    raw_plan = _request_header(request, ALPHA_PROBE_PLAN_HEADER)
    if raw_plan is None:
        return None
    plan = _resolve_plan(raw_plan)

    state = _request_state(request)
    if state is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "alpha_probe_request_state_unavailable",
                "message": "alpha probe request state unavailable",
            },
        )
    existing = getattr(state, _CONTROL_STATE_KEY, None)
    if existing is not None:
        if not isinstance(existing, AlphaProbeControl) or existing.plan != plan:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "alpha_probe_control_conflict",
                    "message": "alpha probe control is already bound to this request",
                },
            )
        if not isinstance(existing.request_identity, str) or not existing.request_identity.strip():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "alpha_probe_request_identity_unavailable",
                    "message": "alpha probe request identity unavailable",
                },
            )
        return existing

    # The caller must pass the result of the existing user_api_key_auth
    # dependency.  _check_admin_auth performs the exact CFG-004 role and
    # UserAPIKeyAuth master-key hash comparison.
    _check_cfg004_admin_auth(user_api_key_dict)
    from . import attempt_records as _attempt_records

    request_identity = (
        _attempt_records._bind_auto_agent_alias_request_identity(request)
    )
    if not isinstance(request_identity, str) or not request_identity.strip():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "alpha_probe_request_identity_unavailable",
                "message": "alpha probe request identity unavailable",
            },
        )
    control = AlphaProbeControl(
        plan=plan,
        request_identity=request_identity,
    )
    setattr(state, _CONTROL_STATE_KEY, control)
    return control


def validate_alpha_probe_control(
    request: Request,
    *,
    user_api_key_dict: Any = None,
) -> Optional[AlphaProbeControl]:
    """Compatibility name for callers that treat resolution as validation."""
    return resolve_alpha_probe_control(
        request,
        user_api_key_dict=user_api_key_dict,
    )


def get_alpha_probe_control(request: Request) -> Optional[AlphaProbeControl]:
    """Return the already-bound request-local control without authorizing."""
    state = _request_state(request)
    value = getattr(state, _CONTROL_STATE_KEY, None) if state else None
    return value if isinstance(value, AlphaProbeControl) else None


def resolve_alpha_overload_probe(
    request: Request,
    *,
    user_api_key_dict: Any = None,
) -> Optional[AlphaOverloadProbeControl]:
    """Resolve and cache one authenticated synthetic SSE overload control.

    The disabled gate and missing header are ordinary no-control paths.  Once
    the gate and exact header are present, plan validation, CFG-004 primary
    admin authentication, and request-identity binding all fail closed.
    """
    if not alpha_overload_probe_control_enabled():
        return None

    raw_plan = _request_header(request, ALPHA_OVERLOAD_PROBE_PLAN_HEADER)
    if raw_plan is None:
        return None
    plan = _resolve_overload_plan(raw_plan)

    state = _request_state(request)
    if state is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "alpha_overload_probe_request_state_unavailable",
                "message": "alpha overload probe request state unavailable",
            },
        )
    existing = getattr(state, _OVERLOAD_CONTROL_STATE_KEY, None)
    if existing is not None:
        if (
            not isinstance(existing, AlphaOverloadProbeControl)
            or existing.plan != plan
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "alpha_overload_probe_control_conflict",
                    "message": (
                        "alpha overload probe control is already bound "
                        "to this request"
                    ),
                },
            )
        return existing

    _check_cfg004_admin_auth(user_api_key_dict)
    from . import attempt_records as _attempt_records

    request_identity = (
        _attempt_records._bind_auto_agent_alias_request_identity(request)
    )
    if not isinstance(request_identity, str) or not request_identity.strip():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "alpha_overload_probe_request_identity_unavailable",
                "message": "alpha overload probe request identity unavailable",
            },
        )
    control = AlphaOverloadProbeControl(
        plan=plan,
        request_identity=request_identity,
    )
    setattr(state, _OVERLOAD_CONTROL_STATE_KEY, control)
    return control


def get_alpha_overload_probe_control(
    request: Request,
) -> Optional[AlphaOverloadProbeControl]:
    """Return the already-bound synthetic control without authorizing."""
    state = _request_state(request)
    value = getattr(state, _OVERLOAD_CONTROL_STATE_KEY, None) if state else None
    return value if isinstance(value, AlphaOverloadProbeControl) else None


def claim_alpha_overload_slot(
    request: Request,
    *,
    control: Optional[AlphaOverloadProbeControl] = None,
) -> Optional[str]:
    """Claim the next finite plan slot without performing any I/O.

    ``overload`` tells the SSE-owner call site to return the synthetic
    response; ``success`` returns control to its ordinary recovery attempt.
    A missing bound control is not an authorization failure and leaves the
    request untouched.
    """
    bound_control = get_alpha_overload_probe_control(request)
    if control is not None and control is not bound_control:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "alpha_overload_probe_control_mismatch",
                "message": (
                    "alpha overload probe control is not bound to this request"
                ),
            },
        )
    if bound_control is None:
        return None

    state = _request_state(request)
    if state is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "alpha_overload_probe_request_state_unavailable",
                "message": "alpha overload probe request state unavailable",
            },
        )
    slot_index = getattr(state, _OVERLOAD_SLOT_STATE_KEY, 0)
    if isinstance(slot_index, bool) or not isinstance(slot_index, int):
        slot_index = 0
    if slot_index < 0:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "alpha_overload_probe_slot_state_invalid",
                "message": "alpha overload probe slot state is invalid",
            },
        )
    if slot_index >= len(bound_control.plan.actions):
        if not bound_control.plan.is_terminal:
            return None
        raise HTTPException(
            status_code=409,
            detail={
                "error": "alpha_overload_probe_terminal_plan_exhausted",
                "message": "alpha overload probe terminal plan is exhausted",
            },
        )
    action = bound_control.plan.actions[slot_index]
    setattr(state, _OVERLOAD_SLOT_STATE_KEY, slot_index + 1)
    return action


def build_synthetic_overload_response(
    *,
    error_code: str = "server_is_overloaded",
) -> httpx.Response:
    """Build one terminal, no-output synthetic Responses SSE transport.

    The response has the same HTTP/SSE shape that the existing pre-commit
    parser consumes.  It intentionally carries no DONE event and makes no
    network request.
    """
    if error_code not in ALPHA_OVERLOAD_PROBE_ERROR_CODES:
        raise ValueError("unsupported synthetic alpha overload code")
    payload = {
        "type": "response.failed",
        "sequence_number": 1,
        "response": {
            "id": "aawm-alpha-overload-probe",
            "object": "response",
            "created_at": 1700000000,
            "status": "failed",
            "output": [],
            "error": {
                "code": error_code,
                "type": "server_error",
                "message": "Synthetic alpha probe overload.",
            },
        },
    }
    encoded_payload = json.dumps(payload, separators=(",", ":")).encode(
        "utf-8"
    )
    raw_sse = (
        b"event: response.failed\ndata: "
        + encoded_payload
        + b"\n\n"
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=raw_sse,
        request=httpx.Request(
            "POST",
            "https://api.openai.com/v1/responses",
        ),
    )


def build_alpha_overload_probe_receipt(
    request: Request,
    *,
    control: AlphaOverloadProbeControl,
    action: str = "overload",
    error_code: str = "server_is_overloaded",
    candidate: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build and retain one sanitized synthetic no-I/O overload receipt."""
    if control is not get_alpha_overload_probe_control(request):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "alpha_overload_probe_control_mismatch",
                "message": (
                    "alpha overload probe control is not bound to this request"
                ),
            },
        )
    if action != "overload":
        raise ValueError("alpha overload receipts record only synthetic faults")
    if error_code not in ALPHA_OVERLOAD_PROBE_ERROR_CODES:
        raise ValueError("unsupported synthetic alpha overload code")

    selection_mapping = selection if isinstance(selection, Mapping) else {}
    candidate_mapping = _resolve_receipt_candidate(
        candidate,
        selection_mapping,
    )
    receipt: dict[str, Any] = {
        "schema": _OVERLOAD_RECEIPT_SCHEMA,
        "request_identity": control.request_identity,
        "sequence": _next_receipt_sequence(
            request,
            state_key=_OVERLOAD_RECEIPT_SEQUENCE_STATE_KEY,
        ),
        "environment": ALPHA_PROBE_ENVIRONMENT,
        "plan": ",".join(control.plan.actions),
        "action": "inject_synthetic_overload",
        "event_type": "response.failed",
        "error_code": error_code,
        "source": "alpha_overload_probe",
        "synthetic": True,
        "attempted_provider_call": False,
        "provider_returned": False,
        **_receipt_occurrence(
            candidate=candidate_mapping,
            selection=selection_mapping,
        ),
    }
    receipt = {
        key: value
        for key, value in receipt.items()
        if value is not None
    }

    state = _request_state(request)
    if state is not None:
        receipts = getattr(state, _OVERLOAD_RECEIPTS_STATE_KEY, None)
        if not isinstance(receipts, list):
            receipts = []
            setattr(state, _OVERLOAD_RECEIPTS_STATE_KEY, receipts)
        receipts.append(dict(receipt))
    return dict(receipt)


def get_alpha_overload_probe_receipts(
    request: Request,
) -> tuple[dict[str, Any], ...]:
    """Return an immutable snapshot of synthetic request-local receipts."""
    state = _request_state(request)
    receipts = getattr(state, _OVERLOAD_RECEIPTS_STATE_KEY, None) if state else None
    if not isinstance(receipts, list):
        return ()
    return tuple(
        dict(receipt)
        for receipt in receipts
        if isinstance(receipt, dict)
    )


def _mapping_value(
    candidate: Mapping[str, Any],
    *names: str,
) -> Any:
    for name in names:
        if name in candidate and candidate[name] is not None:
            return candidate[name]
    return None


def _mapping_presence_value(
    candidate: Mapping[str, Any],
    name: str,
) -> tuple[bool, Any]:
    return (True, candidate[name]) if name in candidate else (False, None)


def _resolve_receipt_candidate(
    candidate: Optional[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> Mapping[str, Any]:
    explicit_candidate = candidate if isinstance(candidate, Mapping) else None
    selected_candidate = selection.get("candidate")
    if selected_candidate is not None and not isinstance(
        selected_candidate, Mapping
    ):
        raise ValueError("selection candidate must be a mapping")
    if (
        explicit_candidate is not None
        and selected_candidate is not None
        and dict(explicit_candidate) != dict(selected_candidate)
    ):
        raise ValueError(
            "explicit candidate does not match selection candidate"
        )
    if explicit_candidate is not None:
        return explicit_candidate
    if isinstance(selected_candidate, Mapping):
        return selected_candidate
    return {}


def _receipt_occurrence(
    *,
    candidate: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    provider = _safe_identifier(_mapping_value(candidate, "provider"))
    model = _safe_identifier(_mapping_value(candidate, "model"))
    route_family = _safe_identifier(
        _mapping_value(candidate, "route_family")
    )
    resolved_alias = _safe_identifier(
        _mapping_value(candidate, "resolved_alias")
    )
    cooldown_identity_tag = _safe_identifier(
        _mapping_value(candidate, "cooldown_identity_tag")
    )
    priority_present, priority_value = _mapping_presence_value(
        candidate,
        "selection_priority",
    )
    last_resort_present, last_resort_value = _mapping_presence_value(
        candidate,
        "last_resort",
    )
    account_label = _safe_identifier(
        _mapping_value(
            candidate,
            "account_label",
            "codex_oauth_account_label",
            "xai_oauth_account_label",
        )
    )
    account_hash = _safe_account_hash(
        _mapping_value(
            candidate,
            "account_hash",
            "codex_oauth_account_hash",
            "xai_oauth_account_hash",
        )
    )
    account_lane = _safe_identifier(
        _mapping_value(
            candidate,
            "account_lane",
            "codex_oauth_lane_key",
            "xai_oauth_lane_key",
        )
    )
    lane_key = _safe_identifier(
        _mapping_value(candidate, "lane_key") or account_lane
    )

    return {
        "alias_model": _safe_identifier(selection.get("alias_model")),
        "resolved_alias": resolved_alias,
        "cooldown_identity_tag": cooldown_identity_tag,
        "selection_priority_present": priority_present,
        "selection_priority": _safe_int(priority_value),
        "last_resort_present": last_resort_present,
        "last_resort": (
            last_resort_value if isinstance(last_resort_value, bool) else None
        ),
        "provider": provider,
        "model": model,
        "route_family": route_family,
        "account_label": account_label,
        "account_hash": account_hash,
        "account_lane": account_lane,
        "lane_key": lane_key,
    }


def _next_receipt_sequence(
    request: Request,
    *,
    state_key: str = _RECEIPT_SEQUENCE_STATE_KEY,
) -> int:
    state = _request_state(request)
    if state is None:
        return 1
    current = getattr(state, state_key, 0)
    current_int = _safe_int(current)
    next_value = (current_int if current_int is not None and current_int >= 0 else 0) + 1
    setattr(state, state_key, next_value)
    return next_value


def build_alpha_probe_receipt(
    request: Request,
    *,
    control: Optional[AlphaProbeControl] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
    phase: str = "pre_egress",
    injection_ordinal: Optional[int] = None,
    attempt_record_index: Optional[int] = None,
) -> dict[str, Any]:
    """Build and retain one sanitized synthetic no-I/O receipt.

    The builder accepts only the fixed ``candidate_unavailable`` action.  It
    intentionally omits upstream status, quota, cooldown, provider-send, and
    provider-returned fields so downstream consumers cannot mistake the
    receipt for a real provider observation.
    """
    bound_control = get_alpha_probe_control(request)
    if control is not None and control is not bound_control:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "alpha_probe_control_mismatch",
                "message": "alpha probe control is not bound to this request",
            },
        )
    resolved_control = bound_control
    if not isinstance(resolved_control, AlphaProbeControl):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "alpha_probe_control_required",
                "message": "alpha probe control is not bound to this request",
            },
        )
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("alpha probe receipt phase must be non-empty")

    selection_mapping = selection if isinstance(selection, Mapping) else {}
    candidate_mapping = _resolve_receipt_candidate(
        candidate,
        selection_mapping,
    )
    occurrence = _receipt_occurrence(
        candidate=candidate_mapping,
        selection=selection_mapping,
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "request_identity": resolved_control.request_identity
        or _request_identity(request),
        "sequence": _next_receipt_sequence(request),
        "environment": ALPHA_PROBE_ENVIRONMENT,
        "plan": resolved_control.plan.name,
        "phase": _safe_identifier(phase),
        "action": "inject_candidate_unavailable",
        "event_type": "synthetic_candidate_unavailable",
        "source": "alpha_probe",
        "synthetic": True,
        "failure_class": "candidate_unavailable",
        "attempted_provider_call": False,
        "provider_returned": False,
        "injection_ordinal": _safe_int(injection_ordinal),
        "attempt_record_index": _safe_int(attempt_record_index),
        **occurrence,
    }
    receipt = {
        key: value for key, value in receipt.items() if value is not None
    }

    state = _request_state(request)
    if state is not None:
        receipts = getattr(state, _RECEIPTS_STATE_KEY, None)
        if not isinstance(receipts, list):
            receipts = []
            setattr(state, _RECEIPTS_STATE_KEY, receipts)
        receipts.append(dict(receipt))
    return dict(receipt)


def get_alpha_probe_receipts(request: Request) -> tuple[dict[str, Any], ...]:
    """Return an immutable snapshot of request-local sanitized receipts."""
    state = _request_state(request)
    receipts = getattr(state, _RECEIPTS_STATE_KEY, None) if state else None
    if not isinstance(receipts, list):
        return ()
    return tuple(
        dict(receipt)
        for receipt in receipts
        if isinstance(receipt, dict)
    )


__all__ = [
    "ALPHA_PROBE_ENABLED_ENV",
    "ALPHA_PROBE_ENVIRONMENT",
    "ALPHA_PROBE_ENVIRONMENT_ENV",
    "ALPHA_PROBE_PLAN_HEADER",
    "ALPHA_PROBE_PLAN_NAMES",
    "ALPHA_OVERLOAD_PROBE_ENABLED_ENV",
    "ALPHA_OVERLOAD_PROBE_ERROR_CODES",
    "ALPHA_OVERLOAD_PROBE_PLAN_HEADER",
    "ALPHA_OVERLOAD_PROBE_PLAN_NAMES",
    "AlphaOverloadProbeControl",
    "AlphaOverloadProbePlan",
    "AlphaProbeControl",
    "AlphaProbePlan",
    "alpha_probe_control_enabled",
    "alpha_overload_probe_control_enabled",
    "build_alpha_probe_receipt",
    "build_alpha_overload_probe_receipt",
    "build_synthetic_overload_response",
    "claim_alpha_overload_slot",
    "get_alpha_probe_control",
    "get_alpha_probe_receipts",
    "get_alpha_overload_probe_control",
    "get_alpha_overload_probe_receipts",
    "resolve_alpha_probe_control",
    "resolve_alpha_overload_probe",
    "validate_alpha_probe_control",
]
