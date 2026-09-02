"""Shared auto-agent alias candidate retry loop (Wave 2 extraction + R3-1 + CFG-004).

This is the moved body of ``llm_passthrough_endpoints._handle_auto_agent_alias_route``
restructured for R3-1 exact-key single-flight publication and CFG-004
intent-based probing:

- Single-flight is enforced via manager-owned PublicationIntents.  The leader
  creates an intent under the selected probe lock BEFORE provider I/O.
  Followers that acquire the same probe lock while the intent is active await
  its completion and retry selection (no second provider call).
- On failure the leader resolves the immutable publication plan, attaches it
  to the intent, RELEASES the probe lock, then enters cooldown mutation with
  NO pre-held lock.  The mutation acquires the family lock + sorted probe
  locks internally (canonical order: never enter family lock while retaining
  a pre-acquired probe lock).
- AFTER the mutation, the loop applies ``plan.request_local_action``, updates
  the attempt record with ``plan.applied_scope``, signals redispatch, and runs
  the native-grok backoff ``asyncio.sleep`` (never inside any lock).
- D1-586 stamps a shadow-only ``shadow_failure_action`` observability field on
  retryable attempt records; enforcement of class-keyed retry/failover remains
  disabled.

Memory and durable targets are derived once from the same plan so telemetry,
waiter visibility, and Redis state cannot disagree, and no target-key logic is
duplicated between the in-lock and post-release paths.

The god-module is imported lazily inside :func:`handle_alias_route` to avoid a
module-scope import cycle (the god-module imports this package); the loop
otherwise depends only on the typed :class:`AliasRouteServices` seams.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
from typing import TYPE_CHECKING, Any, Mapping, Optional

import httpx
from fastapi import HTTPException

from litellm.llms.zai_coding_plan.chat.transformation import (
    ZAICodingPlanAuthenticationError,
)
from litellm.llms.zai_coding_plan.failure_classification import (
    ZAICodingPlanFailureKind,
    classify_zai_coding_plan_failure,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.aawm_route_logging import (
    register_aawm_route_rollup_access_log_replacement,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.cohere import (
    classify_cohere_failure,
)

from . import codex_oauth as _codex_oauth_mod
from . import error_signals as _error_signals
from . import dev_fault_plan as _dev_fault_plan
from .interfaces import (
    AliasRouteServices,
    ClassifyKimiFailureFn,
    ClassifyRetryableFailureFn,
    CooldownPublicationPlan,
    GetCooldownSecondsFn,
    GetActiveCooldownStateFn,
    GetKimiFailureMetadataFn,
    IsGrokAccountQuotaFailureFn,
    RecordCodexFailureEvidenceFn,
    ResolveCooldownPublicationFn,
)
from .state import alias_routing_state, validate_alias_family


def _session_affinity_mod():
    """Lazy session_affinity import (safe under module rebinding)."""
    import sys

    mod = sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if mod is not None:
        return mod
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as mod,
    )
    return mod


def _request_endpoint_path(request: Any) -> Optional[str]:
    try:
        path = getattr(getattr(request, "url", None), "path", None)
    except Exception:
        path = None
    return path if isinstance(path, str) else None


def _admission_mod():
    """Lazy admission import (safe under module rebinding)."""
    import sys

    mod = sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.admission"
    )
    if mod is not None:
        return mod
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        admission as mod,
    )
    return mod


if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request
    from starlette.responses import Response

    from .types import Payload


# ---------------------------------------------------------------------------
# Injected lane-identity calculator (CFG-004 Wave A facade boundary)
# ---------------------------------------------------------------------------
#
# The candidate loop must NOT import ``cooldown_apply`` directly: that module
# owns the publication transaction and is a sibling owner.  The identity
# calculator is injected through this seam during god-module facade setup
# (``configure_candidate_loop_runtime``); the fallback below reproduces the
# canonical public-identity hash so the loop stays correct standalone and in
# tests that do not wire the integrator.


def _default_lane_identity_hash(*, candidate: dict[str, Any]) -> str:
    """Secret-safe identity hash from public candidate identity (fallback).

    Mirrors ``cooldown_apply.resolve_lane_identity_hash``: snapshot candidates
    use ``cooldown_identity_tag`` while static candidates retain the
    ``provider:model:route_family`` fallback. Identity never includes lane keys
    or credentials.
    """
    identity_input = str(candidate.get("cooldown_identity_tag") or "")
    if not identity_input:
        provider = str(candidate.get("provider") or "")
        model = str(candidate.get("model") or "")
        route_family = str(candidate.get("route_family") or "")
        identity_input = f"{provider}:{model}:{route_family}"
    return hashlib.sha256(identity_input.encode("utf-8")).hexdigest()


_resolve_lane_identity_hash_fn = _default_lane_identity_hash


def configure_candidate_loop_runtime(
    *,
    resolve_lane_identity_hash_fn: Optional[Any] = None,
) -> None:
    """Inject the canonical lane-identity calculator (facade boundary).

    The integrator wires ``cooldown_apply.resolve_lane_identity_hash`` here
    during god-module facade setup so the loop uses the single canonical
    implementation without importing ``cooldown_apply``.  Passing ``None``
    restores the built-in fallback.
    """
    global _resolve_lane_identity_hash_fn
    _resolve_lane_identity_hash_fn = (
        resolve_lane_identity_hash_fn
        if resolve_lane_identity_hash_fn is not None
        else _default_lane_identity_hash
    )


def _active_lane_identity_hash(*, candidate: dict[str, Any]) -> str:
    return _resolve_lane_identity_hash_fn(candidate=candidate)


_CODEX_COHERE_PROVIDER = "cohere"
_CODEX_COHERE_ROUTE_FAMILY = "codex_cohere_chat_completions_adapter"
_CODEX_COHERE_CHAT_V2_URL = httpx.URL("https://api.cohere.com/v2/chat")
_CODEX_ZAI_CODING_PLAN_PROVIDER = "zai_coding_plan"
_CODEX_ZAI_CODING_PLAN_ROUTE_FAMILY = "codex_zai_coding_plan_chat_completions_adapter"


def _accepts_excluded_candidate_keys(select_candidate_fn: Any) -> bool:
    """Support CFG-040 legacy selectors without weakening the typed seam.

    Type-erased callbacks still receive the new keyword. Explicit historical
    keyword-only callbacks reject it, as do callbacks whose signature cannot
    be inspected safely, so omit it before selector traversal.
    """
    try:
        signature = inspect.signature(select_candidate_fn)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get("excluded_candidate_keys")
    if parameter is not None:
        return parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        )
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


_IN_FLIGHT_REDISPATCH_ERROR_CODES = frozenset(
    {
        "aawm_codex_auto_agent_in_flight_provider_cooling_down",
        "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
    }
)
_CURSOR_SESSION_CONTINUATION_FAILURE_MARKER = (
    "_cursor_session_continuation_failure"
)
_CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD = "cursor_sanitized_proto_structure"
_CURSOR_PROTO_STRUCTURE_MAX_DEPTH = 3
_CURSOR_PROTO_STRUCTURE_MAX_ITEMS = 64
_TERMINAL_ERROR_ALREADY_EMITTED_REQUEST_STATE_KEY = (
    "aawm_terminal_error_emitted"
)


def _request_terminal_error_already_emitted(request: Any) -> bool:
    state = getattr(request, "state", None)
    if state is None:
        return False
    try:
        return (
            getattr(
                state,
                _TERMINAL_ERROR_ALREADY_EMITTED_REQUEST_STATE_KEY,
                False,
            )
            is True
        )
    except Exception:
        return False


def _mark_request_terminal_error_emitted(request: Any) -> None:
    state = getattr(request, "state", None)
    if state is None:
        return
    try:
        setattr(state, _TERMINAL_ERROR_ALREADY_EMITTED_REQUEST_STATE_KEY, True)
    except Exception:
        pass


def _is_cursor_session_continuation_failure(
    exc: Any,
    *,
    candidate: Optional[Mapping[str, Any]] = None,
) -> bool:
    return bool(
        isinstance(candidate, Mapping)
        and candidate.get("provider") == "cursor_agent"
        and getattr(exc, _CURSOR_SESSION_CONTINUATION_FAILURE_MARKER, False)
    )


def _extract_cursor_sanitized_proto_structure(
    exc: Any,
    *,
    candidate: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if candidate.get("provider") != "cursor_agent":
        return None
    detail = getattr(exc, "detail", None)
    if not isinstance(detail, Mapping):
        return None
    structure = detail.get(_CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD)
    if not isinstance(structure, Mapping):
        return None
    raw_fields = structure.get("fields")
    if not isinstance(raw_fields, list):
        return None

    item_count = 0

    def _copy_fields(
        fields: Any,
        *,
        depth: int,
    ) -> Optional[list[dict[str, Any]]]:
        nonlocal item_count
        if not isinstance(fields, list) or depth > _CURSOR_PROTO_STRUCTURE_MAX_DEPTH:
            return None

        copied_fields: list[dict[str, Any]] = []
        for raw_field in fields:
            if (
                item_count >= _CURSOR_PROTO_STRUCTURE_MAX_ITEMS
                or not isinstance(raw_field, Mapping)
            ):
                return None
            field_number = raw_field.get("field_number")
            wire_type = raw_field.get("wire_type")
            payload_length = raw_field.get("payload_length")
            if (
                isinstance(field_number, bool)
                or not isinstance(field_number, int)
                or field_number <= 0
                or isinstance(wire_type, bool)
                or not isinstance(wire_type, int)
                or not 0 <= wire_type <= 7
                or isinstance(payload_length, bool)
                or not isinstance(payload_length, int)
                or payload_length < 0
            ):
                return None

            item_count += 1
            copied_field: dict[str, Any] = {
                "field_number": field_number,
                "wire_type": wire_type,
                "payload_length": payload_length,
            }
            nested_fields = raw_field.get("nested_fields")
            if nested_fields is not None:
                if depth >= _CURSOR_PROTO_STRUCTURE_MAX_DEPTH:
                    return None
                copied_nested_fields = _copy_fields(
                    nested_fields,
                    depth=depth + 1,
                )
                if copied_nested_fields is None:
                    return None
                copied_field["nested_fields"] = copied_nested_fields
            copied_fields.append(copied_field)
        return copied_fields

    copied_fields = _copy_fields(raw_fields, depth=0)
    if copied_fields is None:
        return None
    return {"fields": copied_fields}


def _validated_redispatch_terminal_metadata(
    exc: Exception,
    *,
    request: Any = None,
) -> Optional[dict[str, Any]]:
    """Extract only structured redispatch fields from an exception/detail."""
    detail = getattr(exc, "detail", None)
    detail_mapping = detail if isinstance(detail, Mapping) else {}
    detail_error = detail_mapping.get("error")
    detail_error = detail_error if isinstance(detail_error, Mapping) else {}
    redispatch_error_codes = {
        str(value)
        for value in (
            detail_error.get("code"),
            detail_mapping.get("error_code"),
            getattr(exc, "error_code", None),
            getattr(exc, "code", None),
        )
        if value is not None
    }
    if not (
        getattr(exc, "redispatch_required", None) is True
        or detail_mapping.get("redispatch_required") is True
        or bool(redispatch_error_codes & _IN_FLIGHT_REDISPATCH_ERROR_CODES)
    ):
        return None

    def _first_value(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    raw_status_code = _first_value(
        getattr(exc, "status_code", None),
        detail_mapping.get("error_status_code"),
        detail_mapping.get("status_code"),
        detail_error.get("status_code"),
    )
    try:
        status_code = int(raw_status_code)
    except (TypeError, ValueError):
        status_code = 409
    if isinstance(raw_status_code, bool) or not 400 <= status_code <= 599:
        status_code = 409

    error_code = _first_value(
        detail_error.get("code"),
        detail_mapping.get("error_code"),
        getattr(exc, "error_code", None),
        getattr(exc, "code", None),
    )
    error_type = _first_value(
        detail_error.get("type"),
        detail_mapping.get("error_type"),
        getattr(exc, "error_type", None),
        getattr(exc, "type", None),
    )
    failure_phase = _first_value(
        detail_mapping.get("failure_phase"),
        getattr(exc, "failure_phase", None),
    )
    code_text = str(error_code) if error_code is not None else ""
    is_in_flight = code_text in _IN_FLIGHT_REDISPATCH_ERROR_CODES
    event_type = (
        "in_flight_pinned_session_cooldown"
        if is_in_flight
        else "redispatch_required"
    )
    candidate_status = (
        "pinned_session_cooldown" if is_in_flight else "redispatch_required"
    )
    failure_class = _first_value(
        detail_mapping.get("failure_class"),
        detail_mapping.get("error_class"),
        getattr(exc, "failure_class", None),
        "in_flight_pinned_session_cooldown" if is_in_flight else "redispatch_required",
    )
    candidate = _first_value(
        detail_mapping.get("candidate"),
        getattr(exc, "candidate", None),
    )
    if not isinstance(candidate, Mapping):
        candidate = None
    extra_fields: dict[str, Any] = {}
    if _request_terminal_error_already_emitted(request):
        extra_fields["_aawm_terminal_error_already_emitted"] = True
    return {
        "detail": detail,
        "candidate": dict(candidate) if isinstance(candidate, Mapping) else None,
        "event_type": event_type,
        "candidate_status": candidate_status,
        "failure_phase": str(
            failure_phase
            or ("session_affinity_cooldown" if is_in_flight else "candidate_selection")
        ),
        "error_status_code": status_code,
        "error_code": error_code,
        "error_type": error_type,
        "failure_class": str(failure_class),
        "extra_fields": extra_fields,
    }


def _emit_validated_redispatch_terminal_event(
    *,
    exc: Exception,
    request: Any,
    alias_family: str,
    alias_model: str,
    request_body: dict[str, Any],
    selection: Optional[Mapping[str, Any]] = None,
    attempts: list[dict[str, Any]],
    emit_pre_attempt_terminal_event: Any,
) -> bool:
    metadata = _validated_redispatch_terminal_metadata(exc, request=request)
    if metadata is None:
        return False
    terminal_candidate = metadata["candidate"]
    if terminal_candidate is None and isinstance(selection, Mapping):
        selected_candidate = selection.get("candidate")
        if isinstance(selected_candidate, Mapping):
            terminal_candidate = dict(selected_candidate)
    try:
        emit_pre_attempt_terminal_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=request_body,
            event_type=metadata["event_type"],
            candidate_status=metadata["candidate_status"],
            failure_phase=metadata["failure_phase"],
            error_status_code=metadata["error_status_code"],
            error_code=metadata["error_code"],
            candidate=terminal_candidate,
            selection=selection,
            attempts=attempts,
            detail=metadata["detail"],
            failure_class=metadata["failure_class"],
            error_type=metadata["error_type"],
            redispatch_required=True,
            extra_fields=metadata["extra_fields"],
        )
    except Exception:
        # Terminal observability must not replace the validated client error.
        return True
    _mark_request_terminal_error_emitted(request)
    return True


def _classify_codex_cohere_candidate_failure(
    exc: Exception,
    *,
    candidate: Optional[dict[str, Any]],
    is_codex_alias: bool,
) -> Optional[str]:
    """Translate direct Cohere failures into the enforced Codex vocabulary."""

    if (
        not is_codex_alias
        or not isinstance(candidate, dict)
        or candidate.get("provider") != _CODEX_COHERE_PROVIDER
        or candidate.get("route_family") != _CODEX_COHERE_ROUTE_FAMILY
    ):
        return None

    classification = classify_cohere_failure(
        url=_CODEX_COHERE_CHAT_V2_URL,
        custom_llm_provider=_CODEX_COHERE_PROVIDER,
        status_code=_error_signals._extract_adapter_exception_status_code(exc),
        exc=exc,
    )
    if (
        classification is None
        or classification.cooldown_scope != "candidate"
        or not classification.advance_fresh_candidate
    ):
        return None
    if classification.name == "cohere_timeout_connectivity":
        return "upstream_timeout"
    return {
        "auth": "provider_terminal_error",
        "quota_exhausted": "usage_limit_reached",
        "rate_limit": "rate_limited",
        "model_unavailable": "candidate_unavailable",
        "provider_4xx_other": "provider_terminal_error",
        "provider_5xx": "provider_terminal_error",
        "transient": "provider_terminal_error",
    }.get(classification.failure_class)


_ZAI_CODING_PLAN_KIND_TO_ERROR_CLASS = {
    ZAICodingPlanFailureKind.AUTH: "provider_terminal_error",
    ZAICodingPlanFailureKind.QUOTA: "usage_limit_reached",
    ZAICodingPlanFailureKind.RATE: "rate_limited",
    ZAICodingPlanFailureKind.CAPACITY: "capacity_exhausted",
    ZAICodingPlanFailureKind.MODEL_UNAVAILABLE: "candidate_unavailable",
    ZAICodingPlanFailureKind.VALIDATION: "provider_terminal_error",
    ZAICodingPlanFailureKind.ROUTING: "provider_terminal_error",
}


def _classify_codex_zai_coding_plan_candidate_failure(
    exc: Exception,
    *,
    candidate: Optional[dict[str, Any]],
    attempted_provider_call: bool = True,
) -> Optional[str]:
    """Map Coding Plan business codes onto the shared Codex retry vocabulary.

    1113 on the coding base is a wrong-base / wrong-key routing defect, not
    ordinary-balance recharge. Model-unavailable business codes require both
    an attempted call and explicit provider-return attribution. Unknown codes
    return ``None`` so generic classifiers can still inspect HTTP status.
    """

    if (
        not isinstance(candidate, dict)
        or candidate.get("provider") != _CODEX_ZAI_CODING_PLAN_PROVIDER
    ):
        return None
    route_family = candidate.get("route_family")
    if route_family not in (None, _CODEX_ZAI_CODING_PLAN_ROUTE_FAMILY):
        return None

    _error_type, error_code = _error_signals._extract_codex_auto_agent_error_type_and_code(
        exc
    )
    failure = classify_zai_coding_plan_failure(
        status_code=_error_signals._extract_adapter_exception_status_code(exc),
        error_code=error_code,
        upstream_id=candidate.get("model"),
    )
    if failure.kind == ZAICodingPlanFailureKind.MODEL_UNAVAILABLE and (
        not attempted_provider_call
        or getattr(exc, "_aawm_provider_returned", False) is not True
    ):
        return "provider_terminal_error"
    if failure.kind != ZAICodingPlanFailureKind.UNKNOWN:
        return _ZAI_CODING_PLAN_KIND_TO_ERROR_CLASS.get(failure.kind)
    if _exception_chain_contains_type(exc, ZAICodingPlanAuthenticationError):
        return "provider_terminal_error"
    if _error_signals._extract_adapter_exception_status_code(exc) == 401:
        return "provider_terminal_error"
    return None


def _classify_codex_fresh_auth_failure(
    exc: Exception,
    *,
    candidate: Optional[dict[str, Any]],
    selection: dict[str, Any],
    is_codex_alias: bool,
    has_continuation_state: bool,
    has_previous_response_id: bool,
    attempted_provider_call: bool,
) -> Optional[str]:
    """Advance past generic provider auth failures on fresh Codex requests."""
    if (
        not is_codex_alias
        or not isinstance(candidate, dict)
        or has_continuation_state
        or has_previous_response_id
        or selection.get("has_account_bound_state")
        or not attempted_provider_call
        or _error_signals._extract_adapter_exception_status_code(exc) != 401
        or _codex_oauth_mod.is_direct_codex_token_invalidated_error(exc)
    ):
        return None
    return "provider_terminal_error"


def _classify_kimi_invalid_request_failure(
    exc: Exception,
    *,
    candidate: Optional[dict[str, Any]],
    kimi_failure_metadata: Optional[dict[str, Any]],
) -> Optional[str]:
    if (
        not isinstance(candidate, dict)
        or candidate.get("provider") != "kimi_code"
        or (
            kimi_failure_metadata is not None
            and (
                kimi_failure_metadata.get("kind") != "malformed"
                or kimi_failure_metadata.get("scope") != "none"
            )
        )
    ):
        return None
    detail = getattr(exc, "detail", None)
    detail_error = detail.get("error") if isinstance(detail, dict) else None
    status_code = _error_signals._extract_adapter_exception_status_code(exc)
    if (
        status_code in {400, 422}
        and isinstance(detail_error, dict)
        and detail_error.get("code") == "kimi_code_invalid_request"
    ):
        return "kimi_code_no_cooldown"
    return None


def _exception_chain_contains_type(
    exc: BaseException, expected: type[BaseException]
) -> bool:
    current: Optional[BaseException] = exc
    seen: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in seen:
            return False
        if isinstance(current, expected):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _proxy_exception_for_unclassified_probe_failure(exc: Exception) -> Exception:
    """Turn an unclassified alias-probe failure into a FastAPI-safe HTTP error.

    Raw OpenRouter/BadRequest/ZAI exceptions leak as uvicorn
    ``Exception in ASGI application`` plus a full traceback. HTTPException
    and ProxyException are already FastAPI-safe. A recognized provider HTTP
    status (4xx/5xx) is preserved; anything without a valid status is an
    internal error (HTTP 500), never a client error.
    """

    if isinstance(exc, (HTTPException, ProxyException)):
        return exc
    status_code = _error_signals._extract_adapter_exception_status_code(exc)
    if status_code is None or not (400 <= int(status_code) <= 599):
        return ProxyException(
            message=str(getattr(exc, "message", str(exc))),
            type="internal_server_error",
            param="model",
            code=500,
        )
    return ProxyException(
        message=str(getattr(exc, "message", str(exc))),
        type=str(getattr(exc, "type", "invalid_request_error") or "invalid_request_error"),
        param="model",
        code=status_code,
    )


async def handle_alias_route(  # noqa: PLR0915
    services: AliasRouteServices,
    *,
    alias_family: str,
    alias_model: str,
    request: "Request",
    prepared_request_body: "Payload",
    max_candidate_attempts: int,
    get_active_cooldown_state_fn: GetActiveCooldownStateFn,
    attempts_metadata_key: str,
    skipped_candidates_metadata_key: str,
    no_candidate_detail: str,
    log_label: str,
) -> "Response":
    """Shared Anthropic/Codex auto-agent alias candidate loop (RR-054 #10)."""
    # Late import to break the module-scope cycle with the god-module.
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as _lpe
    from litellm.proxy.pass_through_endpoints import (
        pass_through_endpoints as _passthrough_helpers,
    )

    _codex_auto_agent_request_has_continuation_state = _lpe._codex_auto_agent_request_has_continuation_state
    _get_codex_auto_agent_native_grok_continuation_transient_max_attempts = (
        _lpe._get_codex_auto_agent_native_grok_continuation_transient_max_attempts
    )
    _codex_auto_agent_candidate_public_shape = _lpe._codex_auto_agent_candidate_public_shape
    _record_auto_agent_alias_attempt_started = _lpe._record_auto_agent_alias_attempt_started
    _is_auto_agent_alias_in_flight_cooldown_http_exception = _lpe._is_auto_agent_alias_in_flight_cooldown_http_exception
    _emit_auto_agent_alias_pre_attempt_terminal_event = _lpe._emit_auto_agent_alias_pre_attempt_terminal_event
    _emit_auto_agent_alias_no_candidate_event = _lpe._emit_auto_agent_alias_no_candidate_event
    _get_safe_kimi_code_probe_failure_metadata = _lpe._get_safe_kimi_code_probe_failure_metadata
    _classify_kimi_code_auto_agent_probe_failure = _lpe._classify_kimi_code_auto_agent_probe_failure
    _classify_codex_auto_agent_retryable_exhaustion = _lpe._classify_codex_auto_agent_retryable_exhaustion
    _is_codex_auto_agent_grok_account_quota_exhaustion = _lpe._is_codex_auto_agent_grok_account_quota_exhaustion
    _get_codex_auto_agent_cooldown_seconds = _lpe._get_codex_auto_agent_cooldown_seconds
    _record_codex_failure_evidence = _lpe._record_codex_failure_evidence
    _update_codex_auto_agent_retryable_attempt_record = _lpe._update_codex_auto_agent_retryable_attempt_record
    _exclude_codex_auto_agent_request_local_candidate_without_cooldown = (
        _lpe._exclude_codex_auto_agent_request_local_candidate_without_cooldown
    )
    _codex_oauth_candidate_slot = _lpe._codex_oauth_candidate_slot
    _plan_codex_oauth_account_failover = (
        _lpe._plan_codex_oauth_account_failover
    )
    _apply_request_local_cooldown_from_plan = _lpe._apply_request_local_cooldown_from_plan
    _record_auto_agent_alias_attempt_failure = _lpe._record_auto_agent_alias_attempt_failure
    from . import attempt_records as _attempt_records

    _record_auto_agent_alias_attempt_success = getattr(
        _lpe,
        "_record_auto_agent_alias_attempt_success",
        _attempt_records._record_auto_agent_alias_attempt_success,
    )
    _bind_auto_agent_alias_request_identity = (
        _attempt_records._bind_auto_agent_alias_request_identity
    )
    _mark_auto_agent_alias_request_failover_pending = (
        _attempt_records._mark_auto_agent_alias_request_failover_pending
    )
    _mark_auto_agent_alias_request_terminal_failure = (
        _attempt_records._mark_auto_agent_alias_request_terminal_failure
    )
    _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible = (
        _lpe._is_codex_auto_agent_native_grok_continuation_transient_retry_eligible
    )
    _is_codex_auto_agent_native_grok_4_5_candidate = _lpe._is_codex_auto_agent_native_grok_4_5_candidate
    _plan_codex_auto_agent_native_grok_continuation_transient_retry = (
        _lpe._plan_codex_auto_agent_native_grok_continuation_transient_retry
    )
    _get_codex_auto_agent_source_error_summary = _lpe._get_codex_auto_agent_source_error_summary
    _extract_adapter_exception_status_code = _lpe._extract_adapter_exception_status_code
    _merge_passthrough_request_shape_metadata = (
        _passthrough_helpers._merge_passthrough_request_shape_metadata
    )
    _build_passthrough_request_shape_summary = (
        _passthrough_helpers._build_passthrough_request_shape_summary
    )
    verbose_proxy_logger = _lpe.verbose_proxy_logger
    status = _lpe.status
    HTTPException = _lpe.HTTPException

    select_candidate_fn = services.select_candidate_fn
    perform_candidate_request_fn = services.perform_candidate_request_fn
    resolve_cooldown_publication_fn = services.resolve_cooldown_publication_fn
    publish_cooldown_memory_fn = services.publish_cooldown_memory_fn
    persist_cooldown_fn = services.persist_cooldown_fn
    set_session_affinity_fn = services.set_session_affinity_fn
    add_alias_metadata_fn = services.add_alias_metadata_fn
    raise_redispatch_required_fn = services.raise_redispatch_fn
    is_codex_alias = validate_alias_family(alias_family) == "codex"
    replay_safety = (
        _session_affinity_mod().classify_session_owner_replay_safety_body(
            prepared_request_body
        )
        if is_codex_alias
        and alias_model in {"codex-auto-review", "auto-review"}
        else None
    )
    codex_failure_evidence_alias = alias_model if is_codex_alias else None
    failed_provider_candidate_keys: set[str] = set()
    deterministically_ineligible_candidate_keys: set[str] = set()

    register_aawm_route_rollup_access_log_replacement(request)
    _bind_auto_agent_alias_request_identity(request)
    attempts: list[dict[str, Any]] = []
    request_outcome = _attempt_records._auto_agent_alias_request_outcome_state(request)
    request_outcome["attempts"] = attempts
    last_retryable_exc: Optional[Exception] = None
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(prepared_request_body)
    has_previous_response_id = bool(
        prepared_request_body.get("previous_response_id")
    )
    native_grok_continuation_transient_max_attempts = (
        _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
    )
    # Request-scoped total for eligible native Grok continuation transient
    # attempts. Must not reset when the outer candidate-selection loop re-enters.
    native_grok_continuation_transient_provider_attempts = 0
    provider_candidate_attempts = 0
    same_account_transient_attempts_by_slot: dict[Optional[str], int] = {}
    token_invalidated_reload_attempts: set[str] = set()
    account_failover_replay_safe = (
        replay_safety.safe
        if replay_safety is not None
        else _session_affinity_mod().is_replay_safe_session_owner_redispatch_body(
            prepared_request_body
        )
    )

    def _genuinely_fresh_dispatch(selection: Mapping[str, Any]) -> bool:
        return (
            not has_continuation_state
            and not has_previous_response_id
            and not bool(selection.get("has_account_bound_state"))
            and not bool(selection.get("in_flight_session"))
        )

    cursor_replay_rejection_request_shape_summary: Optional[dict[str, Any]] = None
    cursor_replay_fresh_dispatch_reject: Optional[dict[str, Any]] = None

    def _cursor_session_continuation_replay_safe_fresh_dispatch_body(
        continuation_exc: Exception,
    ) -> Optional["Payload"]:
        nonlocal cursor_replay_fresh_dispatch_reject
        nonlocal cursor_replay_rejection_request_shape_summary
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
            codex_candidate_calls,
        )

        cursor_replay_fresh_dispatch_reject = None
        cursor_replay_rejection_request_shape_summary = None
        rejection_diagnostic: dict[str, Any] = {}
        fresh_fallback_body = (
            codex_candidate_calls._build_cursor_replay_safe_fresh_dispatch_body(
                prepared_request_body,
                continuation_exc=continuation_exc,
                rejection_diagnostic_out=rejection_diagnostic,
            )
        )
        raw_rejection = rejection_diagnostic.get(
            codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
        )
        if isinstance(raw_rejection, Mapping):
            cursor_replay_fresh_dispatch_reject = dict(raw_rejection)
            if attempts and isinstance(attempts[-1], dict):
                attempts[-1][
                    codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
                ] = copy.deepcopy(cursor_replay_fresh_dispatch_reject)

        def _capture_request_shape_summary() -> None:
            nonlocal cursor_replay_rejection_request_shape_summary
            try:
                request_shape_metadata: dict[str, Any] = {}
                _merge_passthrough_request_shape_metadata(
                    request_shape_metadata,
                    request=request,
                    parsed_body=prepared_request_body,
                    provider_bound_body=prepared_request_body,
                )
                cursor_replay_rejection_request_shape_summary = (
                    _build_passthrough_request_shape_summary(request_shape_metadata)
                )
            except Exception:
                cursor_replay_rejection_request_shape_summary = None

        if fresh_fallback_body is None:
            _capture_request_shape_summary()
            return None
        if not _session_affinity_mod().is_replay_safe_session_owner_redispatch_body(
            fresh_fallback_body
        ):
            _capture_request_shape_summary()
            replay_safety_classifier = getattr(
                _session_affinity_mod(),
                "classify_session_owner_replay_safety_body",
                None,
            )
            replay_safety = None
            if callable(replay_safety_classifier):
                try:
                    replay_safety = replay_safety_classifier(fresh_fallback_body)
                except Exception:
                    replay_safety = None
            if cursor_replay_fresh_dispatch_reject is None:
                cursor_replay_fresh_dispatch_reject = (
                    codex_candidate_calls._cursor_replay_fresh_dispatch_reject_for_replay_safety(
                        replay_safety
                    )
                    or {
                        "stage": "rebuilt_body_replay_unsafe",
                        "reason": "replay_safety_rejected",
                    }
                )
                if attempts and isinstance(attempts[-1], dict):
                    attempts[-1][
                        codex_candidate_calls._CURSOR_REPLAY_FRESH_DISPATCH_REJECT_FIELD
                    ] = copy.deepcopy(cursor_replay_fresh_dispatch_reject)
            return None
        return fresh_fallback_body

    def _prefer_codex_oauth_account_failover(
        *,
        candidate: dict[str, Any],
        selection: dict[str, Any],
        error_class: Optional[str],
    ) -> bool:
        if error_class in _error_signals._RESPONSES_PRE_COMMIT_TRANSIENT_CLASSES:
            return False
        if error_class not in {
            "capacity_exhausted",
            "rate_limited",
            "token_invalidated",
            "usage_limit_reached",
            "candidate_unavailable",
            "provider_terminal_error",
        }:
            return False
        if not candidate.get("codex_oauth_account_hash"):
            return False
        if (
            selection.get("has_account_bound_state")
            or has_previous_response_id
        ) and not account_failover_replay_safe:
            return False
        return (
            not has_continuation_state
            or candidate.get("codex_oauth_credential_affinity")
            == "interchangeable"
        )

    def _raise_terminal_alias_failure(  # noqa: PLR0915
        exc: Exception,
        *,
        extra_fields: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        last_attempt = attempts[-1] if attempts else {}
        cursor_sanitized_proto_structure = (
            _extract_cursor_sanitized_proto_structure(
                exc,
                candidate=candidate,
            )
        )
        if (
            cursor_sanitized_proto_structure is not None
            and isinstance(last_attempt, dict)
        ):
            last_attempt[_CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD] = copy.deepcopy(
                cursor_sanitized_proto_structure
            )
        attempted_provider_call = last_attempt.get("attempted_provider_call")
        if attempted_provider_call is None:
            attempted_provider_call = getattr(exc, "attempted_provider_call", True)
        attempted_provider_call = bool(attempted_provider_call)
        terminal_exc: Optional[HTTPException] = None
        if _is_cursor_session_continuation_failure(exc, candidate=candidate):
            detail = getattr(exc, "detail", None)
            if not isinstance(detail, dict):
                detail = {
                    "error": {
                        "message": str(getattr(exc, "message", None) or exc),
                        "type": "invalid_request_error",
                        "code": "aawm_codex_auto_agent_candidate_ineligible",
                    }
                }
            terminal_exc = HTTPException(
                status_code=409,
                detail=detail,
            )
            for field in (
                "candidate_status",
                "ineligibility_reason",
                "failure_phase",
                "attempted_provider_call",
            ):
                if hasattr(exc, field):
                    setattr(terminal_exc, field, getattr(exc, field))
        elif _error_signals._is_codex_auto_agent_candidate_deterministically_ineligible(
            exc
        ):
            terminal_exc = HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=getattr(exc, "detail", None),
            )
            for field in (
                "candidate_status",
                "ineligibility_reason",
                "failure_phase",
                "attempted_provider_call",
            ):
                if hasattr(exc, field):
                    setattr(terminal_exc, field, getattr(exc, field))
        kimi_failure_metadata = _get_safe_kimi_code_probe_failure_metadata(
            exc,
            candidate=candidate if isinstance(candidate, dict) else None,
        )
        if terminal_exc is None and _classify_kimi_invalid_request_failure(
            exc,
            candidate=candidate if isinstance(candidate, dict) else None,
            kimi_failure_metadata=kimi_failure_metadata,
        ):
            detail = getattr(exc, "detail", None)
            if isinstance(detail, dict):
                status_code = int(
                    (kimi_failure_metadata or {}).get("status_code")
                    or getattr(exc, "status_code", 400)
                    or 400
                )
                if status_code not in {400, 422}:
                    status_code = 400
                terminal_exc = HTTPException(
                    status_code=status_code,
                    detail=detail,
                )
        if terminal_exc is None:
            error_class = str(
                last_attempt.get("error_class")
                or _classify_codex_auto_agent_retryable_exhaustion(
                    exc,
                    candidate=candidate,
                    attempted_provider_call=attempted_provider_call,
                )
                or "provider_terminal_error"
            )
            if error_class in {
                "capacity_exhausted",
                "rate_limited",
                "usage_limit_reached",
            }:
                terminal_status_code = status.HTTP_429_TOO_MANY_REQUESTS
            elif error_class == "safety_policy_denied":
                terminal_status_code = status.HTTP_403_FORBIDDEN
            elif error_class == "upstream_timeout":
                terminal_status_code = status.HTTP_504_GATEWAY_TIMEOUT
            elif error_class == "candidate_unavailable":
                terminal_status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            else:
                terminal_status_code = status.HTTP_502_BAD_GATEWAY
            source_error = _get_codex_auto_agent_source_error_summary(
                exc,
                status_code=_extract_adapter_exception_status_code(exc),
            )
            terminal_exc = HTTPException(
                status_code=terminal_status_code,
                detail={
                    "error": {
                        "message": source_error,
                        "type": error_class,
                        "code": "all_candidates_unavailable",
                    }
                },
            )
        if cursor_sanitized_proto_structure is not None:
            terminal_detail = getattr(terminal_exc, "detail", None)
            if isinstance(terminal_detail, dict):
                terminal_detail[
                    _CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD
                ] = copy.deepcopy(cursor_sanitized_proto_structure)
            setattr(
                terminal_exc,
                _CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD,
                copy.deepcopy(cursor_sanitized_proto_structure),
            )
        last_attempt = attempts[-1] if attempts else None
        if isinstance(last_attempt, dict):
            _mark_auto_agent_alias_request_terminal_failure(
                request,
                last_attempt,
            )
        _emit_auto_agent_alias_no_candidate_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=prepared_request_body,
            exc=terminal_exc,
            attempts=attempts,
            traversal_budget_exhausted=(
                provider_candidate_attempts >= max_candidate_attempts
            ),
            extra_fields=extra_fields,
        )
        raise terminal_exc from None

    while provider_candidate_attempts < max_candidate_attempts:
        try:
            selection_kwargs: dict[str, Any] = {
                "request": request,
                "request_body": prepared_request_body,
            }
            if replay_safety is not None:
                selection_kwargs["_replay_safety"] = replay_safety
            if _accepts_excluded_candidate_keys(select_candidate_fn):
                selection_kwargs["excluded_candidate_keys"] = frozenset(
                    deterministically_ineligible_candidate_keys
                )
            selection = await select_candidate_fn(**selection_kwargs)
        except HTTPException as exc:
            if attempts:
                _mark_auto_agent_alias_request_terminal_failure(
                    request,
                    attempts[-1],
                )
            selection_detail = exc.detail if isinstance(exc.detail, dict) else {}
            selection_error = selection_detail.get("error")
            selection_error_code = (
                selection_error.get("code")
                if isinstance(selection_error, dict)
                else None
            )
            if (
                exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                and (
                    getattr(exc, "redispatch_required", None) is True
                    or selection_detail.get("redispatch_required") is True
                )
                and selection_error_code not in _IN_FLIGHT_REDISPATCH_ERROR_CODES
            ):
                raise
            if _emit_validated_redispatch_terminal_event(
                exc=exc,
                request=request,
                alias_family=alias_family,
                alias_model=alias_model,
                request_body=prepared_request_body,
                attempts=attempts,
                emit_pre_attempt_terminal_event=(
                    _emit_auto_agent_alias_pre_attempt_terminal_event
                ),
            ):
                raise
            if exc.status_code == 429:
                if selection_error_code in {
                    "aawm_codex_auto_agent_in_flight_provider_cooling_down",
                    "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
                }:
                    _emit_auto_agent_alias_pre_attempt_terminal_event(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        request_body=prepared_request_body,
                        event_type="in_flight_pinned_session_cooldown",
                        candidate_status="pinned_session_cooldown",
                        failure_phase="session_affinity_cooldown",
                        error_status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=exc.detail,
                        attempts=attempts,
                        failure_class="in_flight_pinned_session_cooldown",
                        redispatch_required=True,
                    )
                elif not _is_auto_agent_alias_in_flight_cooldown_http_exception(
                    exc
                ):
                    _emit_auto_agent_alias_no_candidate_event(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        request_body=prepared_request_body,
                        exc=exc,
                        attempts=attempts,
                    )
            raise
        except Exception as exc:
            if attempts:
                _mark_auto_agent_alias_request_terminal_failure(
                    request,
                    attempts[-1],
                )
            _emit_validated_redispatch_terminal_event(
                exc=exc,
                request=request,
                alias_family=alias_family,
                alias_model=alias_model,
                request_body=prepared_request_body,
                attempts=attempts,
                emit_pre_attempt_terminal_event=(
                    _emit_auto_agent_alias_pre_attempt_terminal_event
                ),
            )
            raise
        candidate = selection["candidate"]
        cooldown_key = str(selection["cooldown_key"])
        if cooldown_key in failed_provider_candidate_keys:
            if attempts:
                _mark_auto_agent_alias_request_terminal_failure(
                    request,
                    attempts[-1],
                )
            _raise_terminal_alias_failure(last_retryable_exc or HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": {
                        "message": "No further eligible Codex auto-agent candidates.",
                        "type": "provider_terminal_error",
                        "code": "all_candidates_unavailable",
                    }
                },
            ))
        failover_ordinal = int(selection.get("failover_ordinal") or 0)
        if failover_ordinal == 0:
            provider_candidate_attempts += 1
        attempt_record = _codex_auto_agent_candidate_public_shape(
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
        attempt_record["attempted_provider_call"] = False
        # D1-564: provider/account lane admission after selection and before
        # attempt-start / probe lock / provider I/O. Separate from cooldown and
        # session ownership. Fail-fast only: never queue/sleep/background-retry.
        admission = _admission_mod()
        admission_decision = await admission.admit_selected_candidate(
            candidate=candidate,
            selection=selection,
            attempt_record=attempt_record,
        )
        if not admission_decision.allowed:
            admission_error_class = admission.admission_deny_error_class(
                admission_decision
            )
            account_failover_planned = _plan_codex_oauth_account_failover(
                request,
                candidate=candidate,
                selection=selection,
                attempt_record=attempt_record,
                error_class=admission_error_class,
                has_continuation_state=has_continuation_state,
                has_previous_response_id=has_previous_response_id,
                account_failover_replay_safe=account_failover_replay_safe,
                provider_status_code=attempt_record.get("error_status_code"),
            )
            if account_failover_planned:
                provider_candidate_attempts = max(
                    0,
                    provider_candidate_attempts - 1,
                )
                _mark_auto_agent_alias_request_failover_pending(
                    request,
                    attempt_record,
                )
                _record_auto_agent_alias_attempt_failure(
                    alias_family=alias_family,
                    alias_model=alias_model,
                    request=request,
                    prepared_request_body=prepared_request_body,
                    selection=selection,
                    attempts=attempts,
                    attempt_record=attempt_record,
                    error_class=admission_error_class,
                    add_alias_metadata_fn=add_alias_metadata_fn,
                )
                verbose_proxy_logger.debug(
                    "%s auto-agent alias %s admission denied on lane %s (%s); "
                    "moving once to an independent eligible lane",
                    log_label,
                    alias_model,
                    admission_decision.lane_fingerprint,
                    admission_decision.reason,
                )
                continue
            _emit_auto_agent_alias_pre_attempt_terminal_event(
                alias_family=alias_family,
                alias_model=alias_model,
                request=request,
                request_body=prepared_request_body,
                event_type="provider_lane_admission_rejected",
                candidate_status="admission_denied",
                failure_phase="provider_lane_admission",
                error_status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                error_code=admission_decision.detail_code,
                candidate=attempt_record,
                selection=selection,
                attempts=attempts,
                failure_class=admission_error_class,
                error_type="rate_limit_error",
                extra_fields={
                    "admission_reason": admission_decision.reason,
                    "admission_detail_code": admission_decision.detail_code,
                    "admission_lane_fingerprint": admission_decision.lane_fingerprint,
                    "admission_limit_scope": admission_decision.limit_scope,
                    "admission_exhaustion_kind": admission_decision.exhaustion_kind,
                },
            )
            admission.raise_provider_lane_admission_rejected(
                admission_decision,
                candidate=candidate,
                alias_model=alias_model,
                alias_family=alias_family,
                lane_key=selection.get("lane_key"),
            )
        admission_lease = admission_decision.lease
        try:
            while True:
                # CFG-004: ProbeLease / publication-intent single-flight design.
                #
                # 1. Check active cooldown BEFORE acquiring the probe lock.
                #    The cooldown reader acquires the family lock internally;
                #    calling it outside the probe lock prevents lock inversion
                #    with execute_cooldown_publication_transaction (which
                #    acquires family lock -> sorted probe locks).
                # 2. Acquire the selected probe lock.
                # 3. Check for an active PublicationIntent on this cooldown_key.
                #    If found (follower path): release probe lock, await intent
                #    completion, then break to re-select (no second provider call).
                # 4. Leader path: create intent, perform provider I/O under the
                #    probe lock.
                # 5. On failure: resolve plan, attach to intent, RELEASE probe
                #    lock, then enter cooldown mutation with NO pre-held lock
                #    (execute_cooldown_publication_transaction acquires the
                #    family lock + sorted probe locks internally).
                # 6. Signal intent complete, remove from registry.
                probe_failure_exc: Optional[Exception] = None
                probe_failure_plan: Optional[CooldownPublicationPlan] = None
                skip_after_probe_wait = False
                response: Optional[Response] = None
                intent = None

                # Pre-check active cooldown OUTSIDE the probe lock.  The
                # cooldown reader acquires the family lock; holding the probe
                # lock here would invert the canonical lock order (family ->
                # probe) used by execute_cooldown_publication_transaction.
                try:
                    active_seconds, _active_source = await get_active_cooldown_state_fn(selection["cooldown_key"])
                except Exception as pre_exc:  # noqa: PERF203
                    probe_failure_exc = pre_exc
                    active_seconds = 0.0
                if probe_failure_exc is None and active_seconds > 0:
                    skip_after_probe_wait = True
                    attempt_record["status"] = "skipped_single_flight_cooldown"
                    attempt_record["cooldown_seconds"] = active_seconds

                probe_lock = await alias_routing_state.candidate_probe_lock(
                    alias_family=alias_family,
                    cooldown_key=selection["cooldown_key"],
                )
                await probe_lock.acquire()

                # Follower path: an active intent means a leader is probing or
                # publishing for this key.  Await completion, then re-select.
                # CFG-004 Defect 1 fix: atomic claim_publication_or_wait checks
                # clear reservations AND active intents AND claims a leader intent
                # in ONE registry-lock critical section.  This closes the race
                # where a clear reservation could be created between the intent
                # claim and a separate get_clear_reservation check.
                from .state import ClaimOutcome

                _claim = alias_routing_state.publication_intents.claim_publication_or_wait(
                    alias_family=alias_family,
                    cooldown_keys=frozenset({selection["cooldown_key"]}),
                    identity_hash=_active_lane_identity_hash(candidate=candidate),
                )
                if _claim.outcome is ClaimOutcome.BLOCKED_BY_CLEAR:
                    # Clear reservation covers this key: wait, then reselect
                    # without provider I/O.
                    probe_lock.release()
                    assert _claim.clear_reservation is not None
                    await _claim.clear_reservation.done.wait()
                    skip_after_probe_wait = True
                    break
                if _claim.outcome is ClaimOutcome.FOLLOWER:
                    probe_lock.release()
                    assert _claim.intent is not None
                    await _claim.intent.done.wait()
                    skip_after_probe_wait = True
                    break
                assert _claim.intent is not None
                intent = _claim.intent

                # TOCTOU guard: the pre-check ran BEFORE probe lock acquisition.
                # A concurrent leader may have completed publication (cooldown
                # committed, intent removed) between our pre-check and probe lock
                # acquisition.  The publication transaction holds this same probe
                # lock while mutating cooldown memory, so a lock-free peek here
                # (no family lock, no mutation) sees a consistent snapshot.
                # This closes the final singleflight TOCTOU window without
                # introducing a family-locking read under the probe lock.
                if not skip_after_probe_wait and probe_failure_exc is None:
                    _toctou_remaining = (
                        alias_routing_state.family(alias_family)
                        .peek_cooldown_remaining(selection["cooldown_key"])
                    )
                    if _toctou_remaining > 0:
                        skip_after_probe_wait = True
                        attempt_record["status"] = "skipped_single_flight_cooldown"
                        attempt_record["cooldown_seconds"] = _toctou_remaining

                # If the pre-check found an active cooldown or raised, we still
                # need to create + complete the intent so followers are notified.
                if skip_after_probe_wait or probe_failure_exc is not None:
                    probe_lock.release()
                    intent.complete(error=probe_failure_exc)
                    alias_routing_state.publication_intents.remove(intent)
                    if probe_failure_exc is not None:
                        if _emit_validated_redispatch_terminal_event(
                            exc=probe_failure_exc,
                            request=request,
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            emit_pre_attempt_terminal_event=(
                                _emit_auto_agent_alias_pre_attempt_terminal_event
                            ),
                        ):
                            raise probe_failure_exc
                        raise probe_failure_exc
                    break

                # BaseException-safe: intent is ALWAYS completed and removed,
                # probe lock is ALWAYS released, regardless of exception type
                # (Exception, CancelledError, KeyboardInterrupt, etc.).
                try:
                    session_owner_lease = None
                    attempted_provider_call = False
                    try:
                        sa = _session_affinity_mod()
                        session_owner_identity = selection.get(
                            "session_owner_identity"
                        )
                        canonical_session_identity = selection.get(
                            "canonical_session_identity"
                        )
                        resolved_session_identity = None
                        if (
                            session_owner_identity is None
                            or canonical_session_identity is None
                        ):
                            resolved_session_identity = (
                                sa.resolve_canonical_session_identity(
                                    request,
                                    prepared_request_body,
                                )
                            )
                        if canonical_session_identity is None:
                            canonical_session_identity = (
                                sa.get_request_codex_auto_review_parent_session_identity(
                                    request
                                )
                                or resolved_session_identity
                            )
                        if session_owner_identity is None:
                            session_owner_identity = (
                                resolved_session_identity
                                or canonical_session_identity
                            )
                        owner_attributes = sa.build_session_owner_attributes(
                            candidate=candidate,
                            ingress=alias_family,
                            requested_model=selection.get("alias_model") or alias_model,
                            alias_family=alias_family,
                            endpoint_contract=candidate.get("route_family"),
                            state_format=candidate.get("route_family"),
                        )
                        # Tokenized pre-egress reservation (before upstream send).
                        guard = await sa.ensure_session_owner_guard_for_request(
                            request=request,
                            request_body=prepared_request_body,
                            session_identity=session_owner_identity,
                            requested_attributes=owner_attributes,
                            candidate=candidate,
                            alias_model=selection.get("alias_model") or alias_model,
                            failure_phase="session_owner_pre_egress_reserve",
                        )
                        session_owner_lease = sa.get_request_session_owner_lease(request)
                        # Expose reservation metadata on attempt selection.
                        selection["canonical_session_identity"] = canonical_session_identity
                        selection["session_owner_identity"] = session_owner_identity
                        selection["session_owner_decision"] = guard.decision.value
                        selection["session_owner_reservation_token"] = (
                            guard.reservation_token
                        )
                        selection["session_owner_held_reservation"] = guard.held_reservation
                        if guard.provenance:
                            selection["session_owner_provenance"] = guard.provenance

                        _dev_fault_plan._raise_if_openai_fault_plan_slot_fails(
                            request,
                            candidate=candidate,
                        )
                        candidate_body = _record_auto_agent_alias_attempt_started(
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request=request,
                            prepared_request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            attempt_record=attempt_record,
                            add_alias_metadata_fn=add_alias_metadata_fn,
                        )

                        async def _perform_candidate_request() -> Response:
                            nonlocal attempted_provider_call
                            attempts.append(attempt_record)
                            attempt_record["attempted_provider_call"] = True
                            attempted_provider_call = True
                            try:
                                return await perform_candidate_request_fn(
                                    candidate=candidate,
                                    candidate_body=candidate_body,
                                )
                            except Exception as perform_exc:
                                if (
                                    getattr(
                                        perform_exc,
                                        "attempted_provider_call",
                                        None,
                                    )
                                    is False
                                ):
                                    attempt_record["attempted_provider_call"] = False
                                    attempted_provider_call = False
                                raise

                        run_with_lease_renewal = getattr(
                            sa,
                            "run_with_session_owner_lease_renewal",
                            None,
                        )
                        if callable(run_with_lease_renewal):
                            response = await run_with_lease_renewal(
                                session_owner_lease,
                                _perform_candidate_request,
                            )
                        else:
                            # Keep older extracted-host test seams usable while
                            # the runtime host rolls out the renewal helper.
                            response = await _perform_candidate_request()
                        is_auto_review = (
                            alias_model in {"codex-auto-review", "auto-review"}
                            or sa.get_request_codex_auto_review_parent_session_identity(
                                request
                            )
                            is not None
                        )
                        if is_auto_review:
                            finalize_result = (
                                await sa.finalize_codex_auto_review_lease_on_success(
                                    session_owner_lease
                                )
                            )
                        else:
                            # Authoritative success: promote reserved -> owned.
                            finalize_result = (
                                await sa.finalize_session_owner_lease_on_success(
                                    session_owner_lease,
                                    attributes=owner_attributes,
                                    candidate=candidate,
                                )
                            )
                            if finalize_result is not None and finalize_result.outcome in {
                                sa.SessionOwnerMutationOutcome.CONFLICT,
                                sa.SessionOwnerMutationOutcome.ERROR,
                                sa.SessionOwnerMutationOutcome.NOT_HELD,
                            }:
                                # Success bytes may already be in flight to the
                                # client; fail closed for subsequent requests by
                                # not treating ownership as established. Surface a
                                # structured error for non-streaming callers.
                                sa.raise_session_owner_redispatch_required(
                                    session_identity=session_owner_identity,
                                    mutation=finalize_result,
                                    alias_model=selection.get("alias_model")
                                    or alias_model,
                                    candidate=candidate,
                                    failure_phase="session_owner_promote_after_success",
                                    request=request,
                                )
                    except asyncio.CancelledError:
                        if session_owner_lease is not None:
                            try:
                                await _session_affinity_mod().finalize_session_owner_lease_on_failure(
                                    session_owner_lease
                                )
                            except Exception:  # noqa: BLE001
                                pass
                        raise
                    except Exception as probe_exc:  # noqa: PERF203
                        probe_failure_exc = probe_exc
                        if session_owner_lease is not None:
                            try:
                                await _session_affinity_mod().finalize_session_owner_lease_on_failure(
                                    session_owner_lease
                                )
                            except Exception:  # noqa: BLE001
                                pass
                    finally:
                        # Release probe lock FIRST (unconditional, before any
                        # resolver call that might raise).
                        probe_lock.release()

                    renewal_error_type = getattr(
                        sa,
                        "SessionOwnerLeaseRenewalError",
                        (),
                    )
                    if (
                        probe_failure_exc is not None
                        and isinstance(probe_failure_exc, renewal_error_type)
                    ):
                        attempt_record["attempted_provider_call"] = (
                            attempted_provider_call
                        )
                        sa.raise_session_owner_redispatch_required(
                            session_identity=session_owner_identity,
                            alias_model=selection.get("alias_model") or alias_model,
                            candidate=candidate,
                            failure_phase="session_owner_reservation_renewal",
                            message=(
                                "Session ownership reservation was lost while "
                                "the provider operation was in flight; "
                                "redispatch a fresh response."
                            ),
                            attempted_provider_call=attempted_provider_call,
                            request=request,
                        )

                    if (
                        probe_failure_exc is not None
                        and _error_signals.is_openai_responses_unpersisted_item_not_found_error(
                            probe_failure_exc,
                            candidate=candidate,
                            endpoint=_request_endpoint_path(request),
                            provider_returned=(
                                attempted_provider_call
                                and (
                                    isinstance(probe_failure_exc, ProxyException)
                                    or bool(
                                        getattr(
                                            probe_failure_exc,
                                            "_aawm_provider_returned",
                                            False,
                                        )
                                    )
                                )
                            ),
                        )
                    ):
                        if not has_continuation_state:
                            raise probe_failure_exc
                        attempt_record["status"] = (
                            "terminal_in_flight_unpersisted_item_not_found"
                        )
                        attempt_record["failure_phase"] = (
                            "openai_responses_unpersisted_item_not_found_continuation"
                        )
                        attempt_record["attempted_provider_call"] = True
                        _record_auto_agent_alias_attempt_failure(
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request=request,
                            prepared_request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            attempt_record=attempt_record,
                            error_class="openai_responses_unpersisted_item_not_found",
                            add_alias_metadata_fn=add_alias_metadata_fn,
                            redispatch_required=True,
                        )
                        session_identity = selection.get("session_owner_identity")
                        if not isinstance(session_identity, str) or not session_identity:
                            session_identity = selection.get("canonical_session_identity")
                        if not isinstance(session_identity, str) or not session_identity:
                            session_identity = (
                                _session_affinity_mod().resolve_canonical_session_identity(
                                    request,
                                    prepared_request_body,
                                )
                            )
                        _session_affinity_mod().raise_session_owner_redispatch_required(
                            session_identity=session_identity,
                            alias_model=alias_model,
                            candidate=candidate,
                            failure_phase=(
                                "openai_responses_unpersisted_item_not_found_continuation"
                            ),
                            message=(
                                "OpenAI Responses continuation referenced an "
                                "unpersisted item (store=false). Do not continue "
                                "this session; redispatch a fresh response."
                            ),
                            attempted_provider_call=True,
                            request=request,
                        )

                    # Resolve the plan AFTER lock release.  If the resolver
                    # raises, the outer BaseException handler cleans up the
                    # intent.  No lock is held here (canonical order: no family
                    # lock entry while retaining a pre-acquired probe lock).
                    fresh_codex_auth_error_class: Optional[str] = None
                    if probe_failure_exc is not None:
                        fresh_codex_auth_error_class = _classify_codex_fresh_auth_failure(
                            probe_failure_exc,
                            candidate=candidate,
                            selection=selection,
                            is_codex_alias=codex_failure_evidence_alias is not None,
                            has_continuation_state=has_continuation_state,
                            has_previous_response_id=has_previous_response_id,
                            attempted_provider_call=attempted_provider_call,
                        )
                        probe_failure_plan = _resolve_failure_plan(
                            resolve_cooldown_publication_fn=resolve_cooldown_publication_fn,
                            record_codex_failure_evidence_fn=_record_codex_failure_evidence,
                            request=request,
                            candidate=candidate,
                            selection=selection,
                            attempt_record=attempt_record,
                            exc=probe_failure_exc,
                            codex_failure_evidence_alias=codex_failure_evidence_alias,
                            kimi_failure_metadata_fn=_get_safe_kimi_code_probe_failure_metadata,
                            classify_kimi_fn=_classify_kimi_code_auto_agent_probe_failure,
                            classify_retryable_fn=_classify_codex_auto_agent_retryable_exhaustion,
                            grok_quota_fn=_is_codex_auto_agent_grok_account_quota_exhaustion,
                            cooldown_seconds_fn=_get_codex_auto_agent_cooldown_seconds,
                            fresh_codex_auth_error_class=fresh_codex_auth_error_class,
                        )
                        intent.plan = probe_failure_plan

                    if skip_after_probe_wait:
                        intent.complete()
                        alias_routing_state.publication_intents.remove(intent)
                        break
                    if probe_failure_exc is None:
                        intent.complete()
                        alias_routing_state.publication_intents.remove(intent)
                        await set_session_affinity_fn(
                            selection.get("session_key"),
                            candidate,
                        )
                        assert response is not None
                        attempt_record["attempted_provider_call"] = (
                            attempted_provider_call
                        )
                        _record_auto_agent_alias_attempt_success(
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request=request,
                            prepared_request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            attempt_record=attempt_record,
                            add_alias_metadata_fn=add_alias_metadata_fn,
                        )
                        return response

                    early_pre_commit_error_class = (
                        _classify_kimi_code_auto_agent_probe_failure(
                            _get_safe_kimi_code_probe_failure_metadata(
                                probe_failure_exc,
                                candidate=candidate,
                            )
                        )
                    )
                    if early_pre_commit_error_class is None:
                        early_pre_commit_error_class = (
                            _classify_codex_cohere_candidate_failure(
                                probe_failure_exc,
                                candidate=candidate,
                                is_codex_alias=codex_failure_evidence_alias is not None,
                            )
                        )
                    if early_pre_commit_error_class is None:
                        early_pre_commit_error_class = (
                            _classify_codex_zai_coding_plan_candidate_failure(
                                probe_failure_exc,
                                candidate=candidate,
                                attempted_provider_call=attempted_provider_call,
                            )
                        )
                    if early_pre_commit_error_class is None:
                        early_pre_commit_error_class = (
                            _classify_codex_auto_agent_retryable_exhaustion(
                                probe_failure_exc,
                                candidate=candidate,
                                attempted_provider_call=attempted_provider_call,
                            )
                        )
                    if early_pre_commit_error_class is None:
                        early_pre_commit_error_class = fresh_codex_auth_error_class
                    early_pre_commit_retry_plan = (
                        _error_signals.plan_responses_pre_commit_retry(
                            error_class=early_pre_commit_error_class,
                            same_account_transient_attempts=(
                                same_account_transient_attempts_by_slot.get(
                                    _codex_oauth_candidate_slot(candidate),
                                    0,
                                )
                                + 1
                            ),
                        )
                    )
                    prefer_account_failover = (
                        _prefer_codex_oauth_account_failover(
                            candidate=candidate,
                            selection=selection,
                            error_class=early_pre_commit_error_class,
                        )
                    )
                    skip_cooldown_for_account_failover = (
                        prefer_account_failover
                        and early_pre_commit_error_class
                        == "provider_terminal_error"
                        and _extract_adapter_exception_status_code(
                            probe_failure_exc
                        )
                        == status.HTTP_401_UNAUTHORIZED
                    )
                    if skip_cooldown_for_account_failover:
                        probe_failure_plan = CooldownPublicationPlan(
                            applied_scope="none",
                            grok_account_quota_exhausted=(
                                probe_failure_plan.grok_account_quota_exhausted
                                if probe_failure_plan is not None
                                else False
                            ),
                            kimi_failure_metadata=(
                                probe_failure_plan.kimi_failure_metadata
                                if probe_failure_plan is not None
                                else None
                            ),
                        )
                        intent.plan = probe_failure_plan
                    skip_cooldown_for_same_account_retry = (
                        not prefer_account_failover
                        and
                        early_pre_commit_retry_plan["action"]
                        in {"retry_same_account", "pre_stream_unavailable"}
                    )
                    publication_transaction_result: Optional[object] = None
                    publication_requested_ttl_seconds: Optional[float] = None
                    if (
                        probe_failure_plan is not None
                        and probe_failure_plan.applied_scope == "managed_account"
                    ):
                        publication_requested_ttl_seconds = (
                            _get_codex_auto_agent_cooldown_seconds(
                                probe_failure_exc,
                                candidate=candidate,
                            )
                        )

                    # --- Cooldown mutation: NO pre-held probe lock ------------
                    if (
                        probe_failure_plan is not None
                        and (
                            probe_failure_plan.memory_keys
                            or probe_failure_plan.durable_keys
                        )
                        and not skip_cooldown_for_same_account_retry
                        and not skip_cooldown_for_account_failover
                    ):
                        try:
                            publication_transaction_result = (
                                await _lpe.execute_cooldown_publication_transaction(
                                    alias_family=alias_family,
                                    candidate=candidate,
                                    plan=probe_failure_plan,
                                    publish_cooldown_memory_fn=publish_cooldown_memory_fn,
                                    persist_cooldown_fn=persist_cooldown_fn,
                                )
                            )
                        except BaseException as publication_exc:
                            _attempt_records._attach_kimi_managed_account_publication_telemetry(
                                attempt_record=attempt_record,
                                candidate=candidate,
                                error_class=early_pre_commit_error_class,
                                kimi_failure_metadata=probe_failure_plan.kimi_failure_metadata,
                                plan=probe_failure_plan,
                                requested_ttl_seconds=publication_requested_ttl_seconds,
                                publication_error=publication_exc,
                            )
                            raise
                        _attempt_records._attach_kimi_managed_account_publication_telemetry(
                            attempt_record=attempt_record,
                            candidate=candidate,
                            error_class=early_pre_commit_error_class,
                            kimi_failure_metadata=probe_failure_plan.kimi_failure_metadata,
                            plan=probe_failure_plan,
                            requested_ttl_seconds=publication_requested_ttl_seconds,
                            transaction_result=publication_transaction_result,
                        )
                    # Mutation complete (or no plan): signal intent.
                    intent.complete(error=probe_failure_exc)
                    alias_routing_state.publication_intents.remove(intent)
                except BaseException as cleanup_exc:
                    # BaseException-safe: covers CancelledError, KeyboardInterrupt,
                    # SystemExit, and any exception from cooldown mutation.
                    if not intent.done.is_set():
                        intent.complete(error=cleanup_exc)
                        alias_routing_state.publication_intents.remove(intent)
                    raise

                reload_label = str(
                    candidate.get("codex_oauth_account_label") or ""
                )
                reload_account_hash = str(
                    candidate.get("codex_oauth_account_hash") or ""
                ) or None
                reload_lane_key = str(
                    candidate.get("codex_oauth_lane_key") or ""
                ) or None
                token_invalidated_reload = None
                if (
                    early_pre_commit_error_class == "token_invalidated"
                    and reload_label
                    and reload_account_hash is not None
                    and reload_lane_key is not None
                    and reload_label not in token_invalidated_reload_attempts
                    and _session_affinity_mod().reset_released_request_session_owner_guard(
                        request
                    )
                ):
                    token_invalidated_reload_attempts.add(reload_label)
                    token_invalidated_reload = (
                        await _codex_oauth_mod.reload_codex_oauth_credential_after_token_invalidated(
                            request,
                            account_label=reload_label,
                            model=str(candidate.get("model") or "") or None,
                            expected_account_hash=reload_account_hash,
                            expected_lane_key=reload_lane_key,
                        )
                    )
                    if token_invalidated_reload is not None:
                        if (
                            token_invalidated_reload.account_label != reload_label
                            or token_invalidated_reload.account_hash
                            != reload_account_hash
                            or token_invalidated_reload.lane_key != reload_lane_key
                            or _codex_oauth_mod._bind_codex_oauth_candidate_to_request(
                                request,
                                candidate,
                            )
                            is None
                        ):
                            token_invalidated_reload = None
                    if token_invalidated_reload is not None:
                        attempt_record["token_invalidated_reload"] = {
                            "account_label": reload_label,
                            "credential_material_changed": True,
                            "same_account_retry": True,
                        }
                        _record_auto_agent_alias_attempt_failure(
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request=request,
                            prepared_request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            attempt_record=attempt_record,
                            error_class="token_invalidated",
                            add_alias_metadata_fn=add_alias_metadata_fn,
                        )
                        verbose_proxy_logger.debug(
                            "%s auto-agent alias %s reloaded changed Codex "
                            "OAuth credential for account %s after "
                            "token_invalidated; retrying same account once",
                            log_label,
                            alias_model,
                            reload_label,
                        )
                        attempt_record = _codex_auto_agent_candidate_public_shape(
                            candidate,
                            lane_key=selection.get("lane_key"),
                            reason="token_invalidated_same_account_reload",
                        )
                        attempt_record["attempted_provider_call"] = False
                        continue

                # --- failure handling (post-release) ---------------------------
                failure_exc = probe_failure_exc
                assert failure_exc is not None
                cursor_sanitized_proto_structure = (
                    _extract_cursor_sanitized_proto_structure(
                        failure_exc,
                        candidate=candidate,
                    )
                )
                if cursor_sanitized_proto_structure is not None:
                    attempt_record[
                        _CURSOR_SANITIZED_PROTO_STRUCTURE_FIELD
                    ] = cursor_sanitized_proto_structure
                if _is_cursor_session_continuation_failure(
                    failure_exc,
                    candidate=candidate,
                ):
                    continuation_error_class = (
                        _classify_codex_auto_agent_retryable_exhaustion(
                            failure_exc,
                            candidate=candidate,
                            attempted_provider_call=attempted_provider_call,
                        )
                        or "continuation_state_unavailable"
                    )
                    continuation_plan = probe_failure_plan
                    _update_codex_auto_agent_retryable_attempt_record(
                        attempt_record=attempt_record,
                        exc=failure_exc,
                        error_class=continuation_error_class,
                        cooldown_seconds=(
                            continuation_plan.duration_seconds
                            if continuation_plan is not None
                            else 0.0
                        ),
                        cooldown_scope=(
                            continuation_plan.applied_scope
                            if continuation_plan is not None
                            else "none"
                        ),
                        alias_model=alias_model,
                        candidate=candidate,
                    )
                    _record_auto_agent_alias_attempt_failure(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        prepared_request_body=prepared_request_body,
                        selection=selection,
                        attempts=attempts,
                        attempt_record=attempt_record,
                        error_class=continuation_error_class,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                    )
                    fresh_fallback_body = (
                        _cursor_session_continuation_replay_safe_fresh_dispatch_body(
                            failure_exc
                        )
                    )
                    if fresh_fallback_body is not None:
                        prepared_request_body = fresh_fallback_body
                        has_continuation_state = (
                            _codex_auto_agent_request_has_continuation_state(
                                prepared_request_body
                            )
                        )
                        has_previous_response_id = bool(
                            prepared_request_body.get("previous_response_id")
                        )
                        if replay_safety is not None:
                            replay_safety = (
                                _session_affinity_mod().classify_session_owner_replay_safety_body(
                                    prepared_request_body
                                )
                            )
                        account_failover_replay_safe = (
                            replay_safety.safe
                            if replay_safety is not None
                            else True
                        )
                        provider_candidate_attempts = max(
                            0,
                            provider_candidate_attempts - 1,
                        )
                        deterministically_ineligible_candidate_keys.add(cooldown_key)
                        last_retryable_exc = failure_exc
                        break
                    _raise_terminal_alias_failure(
                        failure_exc,
                        extra_fields=(
                            {
                                field_name: value
                                for field_name, value in (
                                    (
                                        "aawm_passthrough_request_shape_summary",
                                        cursor_replay_rejection_request_shape_summary,
                                    ),
                                    (
                                        "cursor_replay_fresh_dispatch_reject",
                                        cursor_replay_fresh_dispatch_reject,
                                    ),
                                )
                                if value is not None
                            }
                            or None
                        ),
                    )
                if attempted_provider_call:
                    failed_provider_candidate_keys.add(cooldown_key)
                kimi_failure_metadata = _get_safe_kimi_code_probe_failure_metadata(
                    failure_exc,
                    candidate=candidate,
                )
                error_class = _classify_kimi_code_auto_agent_probe_failure(kimi_failure_metadata)
                if error_class is None:
                    error_class = _classify_kimi_invalid_request_failure(
                        failure_exc,
                        candidate=candidate,
                        kimi_failure_metadata=kimi_failure_metadata,
                    )
                if error_class is None:
                    error_class = _classify_codex_cohere_candidate_failure(
                        failure_exc,
                        candidate=candidate,
                        is_codex_alias=codex_failure_evidence_alias is not None,
                    )
                if error_class is None:
                    error_class = _classify_codex_zai_coding_plan_candidate_failure(
                        failure_exc,
                        candidate=candidate,
                        attempted_provider_call=attempted_provider_call,
                    )
                if error_class is None:
                    error_class = _classify_codex_auto_agent_retryable_exhaustion(
                        failure_exc,
                        candidate=candidate,
                        attempted_provider_call=attempted_provider_call,
                    )
                if error_class is None:
                    error_class = fresh_codex_auth_error_class
                if error_class is None:
                    raise _proxy_exception_for_unclassified_probe_failure(failure_exc)
                deterministically_ineligible = (
                    _error_signals._is_codex_auto_agent_candidate_deterministically_ineligible(
                        failure_exc
                    )
                )
                fresh_dispatch = _genuinely_fresh_dispatch(selection)
                marker_reason: str | None = None
                if deterministically_ineligible and fresh_dispatch:
                    marker_reason = (
                        getattr(failure_exc, "ineligibility_reason", None)
                        or "deterministic_candidate_ineligible"
                    )
                elif (
                    fresh_dispatch
                    and account_failover_replay_safe
                    and attempted_provider_call
                    and bool(str(candidate.get("route_family") or "").strip())
                    and not str(candidate.get("route_family") or "").strip().lower().startswith(
                        "anthropic_"
                    )
                    and error_class
                    in {"upstream_timeout", "upstream_transient_internal"}
                ):
                    marker_reason = error_class
                elif (
                    _error_signals._is_codex_auto_agent_cursor_agent_candidate(
                        candidate
                    )
                    and error_class
                    in {"upstream_timeout", "upstream_transient_internal"}
                ):
                    marker_reason = error_class
                if marker_reason is not None:
                    try:
                        semantic_marker = (
                            await alias_routing_state.mark_candidate_semantic_ineligibility(
                                alias_family=alias_family,
                                candidate_key=cooldown_key,
                                reason=marker_reason,
                            )
                        )
                    except Exception:
                        semantic_marker = None
                    if semantic_marker is not None:
                        attempt_record[
                            "candidate_semantic_ineligibility_reason"
                        ] = semantic_marker.get("reason")
                        attempt_record[
                            "candidate_semantic_ineligibility_state_source"
                        ] = semantic_marker.get("state_source") or "memory"
                        attempt_record[
                            "candidate_semantic_ineligibility_remaining_seconds"
                        ] = semantic_marker.get("remaining_seconds")
                if deterministically_ineligible and (
                    fresh_dispatch
                    or (replay_safety is not None and replay_safety.safe)
                ):
                    if fresh_dispatch:
                        provider_candidate_attempts = max(
                            0,
                            provider_candidate_attempts - 1,
                        )
                    deterministically_ineligible_candidate_keys.add(cooldown_key)
                last_retryable_exc = failure_exc
                account_slot = _codex_oauth_candidate_slot(candidate)
                same_account_transient_attempts_by_slot[account_slot] = (
                    same_account_transient_attempts_by_slot.get(account_slot, 0) + 1
                    if error_class
                    in _error_signals._RESPONSES_PRE_COMMIT_TRANSIENT_CLASSES
                    else same_account_transient_attempts_by_slot.get(account_slot, 0)
                )
                pre_commit_retry_plan = _error_signals.plan_responses_pre_commit_retry(
                    error_class=error_class,
                    same_account_transient_attempts=(
                        same_account_transient_attempts_by_slot.get(account_slot, 0)
                    ),
                )
                prefer_account_failover = _prefer_codex_oauth_account_failover(
                    candidate=candidate,
                    selection=selection,
                    error_class=error_class,
                )
                if (
                    not prefer_account_failover
                    and pre_commit_retry_plan["action"] == "retry_same_account"
                ):
                    wait_seconds = float(pre_commit_retry_plan["wait_seconds"] or 0.0)
                    attempt_record["pre_commit_retry"] = {
                        "action": pre_commit_retry_plan["action"],
                        "error_class": error_class,
                        "wait_seconds": wait_seconds,
                        "apply_account_exhaustion_cooldown": False,
                    }
                    _record_auto_agent_alias_attempt_failure(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        prepared_request_body=prepared_request_body,
                        selection=selection,
                        attempts=attempts,
                        attempt_record=attempt_record,
                        error_class=error_class,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                    )
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                    attempt_record = _codex_auto_agent_candidate_public_shape(
                        candidate,
                        lane_key=selection.get("lane_key"),
                        reason="responses_pre_commit_same_account_retry",
                    )
                    attempt_record["attempted_provider_call"] = False
                    continue
                if (
                    not prefer_account_failover
                    and pre_commit_retry_plan["action"] == "pre_stream_unavailable"
                ):
                    attempt_record["pre_commit_retry"] = {
                        "action": pre_commit_retry_plan["action"],
                        "error_class": error_class,
                        "wait_seconds": pre_commit_retry_plan["wait_seconds"],
                        "apply_account_exhaustion_cooldown": False,
                        "retryable": True,
                    }
                    if has_continuation_state:
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={
                                "error": {
                                    "message": (
                                        "Upstream Responses stream failed before "
                                        "the first client byte."
                                    ),
                                    "type": error_class,
                                    "code": "responses_pre_commit_retry_exhausted",
                                    "retryable": True,
                                }
                            },
                            headers={
                                "Retry-After": str(
                                    int(
                                        pre_commit_retry_plan["wait_seconds"]
                                        or 10
                                    )
                                )
                            },
                        )
                    _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
                        request,
                        candidate=candidate,
                        lane_key=selection.get("lane_key"),
                    )
                # The plan was resolved inside the probe lock above.  After probe
                # lock release, execute_cooldown_publication_transaction performed
                # the atomic memory publish + durable commit + local index update
                # under the family lock + sorted probe locks (canonical order).
                # Post-release only the request-local action remains.
                plan = probe_failure_plan
                assert plan is not None
                cooldown_seconds = plan.duration_seconds
                if plan.request_local_action == "request_local_cooldown":
                    _apply_request_local_cooldown_from_plan(
                        request,
                        candidate=candidate,
                        lane_key=selection.get("lane_key"),
                        cooldown_seconds=plan.duration_seconds,
                    )
                cooldown_scope = plan.applied_scope
                error_tokens = _update_codex_auto_agent_retryable_attempt_record(
                    attempt_record=attempt_record,
                    exc=failure_exc,
                    error_class=error_class,
                    cooldown_seconds=cooldown_seconds,
                    cooldown_scope=cooldown_scope,
                    alias_model=alias_model,
                    candidate=candidate,
                    kimi_failure_metadata=kimi_failure_metadata,
                )
                # D1-586: observational shadow action only. Does not change retry,
                # failover, sleep, admission, or cooldown enforcement paths.
                attempt_record["shadow_failure_action"] = (
                    _error_signals.build_shadow_failure_action_decision_from_exc(
                        failure_exc,
                        candidate=candidate,
                        current_error_class=error_class,
                        current_cooldown_scope=cooldown_scope,
                        current_status=attempt_record.get("status"),
                    ).to_observability_dict()
                )
                account_failover_planned = _plan_codex_oauth_account_failover(
                    request,
                    candidate=candidate,
                    selection=selection,
                    attempt_record=attempt_record,
                    error_class=error_class,
                    has_continuation_state=has_continuation_state,
                    has_previous_response_id=has_previous_response_id,
                    account_failover_replay_safe=account_failover_replay_safe,
                    provider_status_code=attempt_record.get("error_status_code"),
                )
                if (
                    cooldown_scope == "none"
                    and not has_continuation_state
                    and not deterministically_ineligible
                ):
                    _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
                        request,
                        candidate=candidate,
                        lane_key=selection.get("lane_key"),
                    )
                if (
                    error_class == "token_invalidated"
                    and has_continuation_state
                    and not account_failover_replay_safe
                    and not account_failover_planned
                ):
                    attempt_record["status"] = (
                        "terminal_in_flight_token_invalidated"
                    )
                    failure_body = _record_auto_agent_alias_attempt_failure(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        prepared_request_body=prepared_request_body,
                        selection=selection,
                        attempts=attempts,
                        attempt_record=attempt_record,
                        error_class=error_class,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                        redispatch_required=True,
                        defer_terminal_error=True,
                    )
                    failure_metadata = failure_body.get("litellm_metadata") or {}
                    try:
                        raise_redispatch_required_fn(
                            candidate=candidate,
                            lane_key=selection.get("lane_key"),
                            cooldown_seconds=0.0,
                            error_tokens=error_tokens,
                            alias_model=alias_model,
                            error_class=error_class,
                            cooldown_scope="none",
                            error_status_code=attempt_record.get("error_status_code"),
                            error_type=attempt_record.get("error_type"),
                            error_code=attempt_record.get("error_code"),
                            retry_after_seconds=0.0,
                            failure_phase="token_invalidated_continuation",
                            attempted_provider_call=attempt_record.get(
                                "attempted_provider_call"
                            ),
                            audit_events=failure_metadata.get(
                                "aawm_alias_routing_audit_events"
                            ),
                            attempts=failure_metadata.get(attempts_metadata_key),
                            skipped_candidates=failure_metadata.get(
                                skipped_candidates_metadata_key
                            ),
                        )
                    except HTTPException as final_exc:
                        _emit_validated_redispatch_terminal_event(
                            exc=final_exc,
                            request=request,
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            emit_pre_attempt_terminal_event=(
                                _emit_auto_agent_alias_pre_attempt_terminal_event
                            ),
                        )
                        raise
                if (
                    has_continuation_state
                    and cooldown_scope != "none"
                    and not account_failover_planned
                ):
                    attempt_record["status"] = "terminal_in_flight_cooldown_set"
                    failure_body = _record_auto_agent_alias_attempt_failure(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        prepared_request_body=prepared_request_body,
                        selection=selection,
                        attempts=attempts,
                        attempt_record=attempt_record,
                        error_class=error_class,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                        redispatch_required=True,
                        defer_terminal_error=True,
                    )
                    failure_metadata = failure_body.get("litellm_metadata") or {}
                    verbose_proxy_logger.debug(
                        "%s auto-agent alias %s target %s/%s hit %s "
                        "for an in-flight session on attempt %s; signaling redispatch",
                        log_label,
                        alias_model,
                        candidate["provider"],
                        candidate["model"],
                        error_class,
                        len(attempts),
                    )
                    try:
                        raise_redispatch_required_fn(
                            candidate=candidate,
                            lane_key=selection.get("lane_key"),
                            cooldown_seconds=cooldown_seconds,
                            error_tokens=error_tokens,
                            alias_model=alias_model,
                            error_class=error_class,
                            cooldown_scope=cooldown_scope,
                            error_status_code=attempt_record.get("error_status_code"),
                            error_type=attempt_record.get("error_type"),
                            error_code=attempt_record.get("error_code"),
                            retry_after_seconds=attempt_record.get("retry_after_seconds"),
                            failure_phase=attempt_record.get("failure_phase"),
                            attempted_provider_call=attempt_record.get(
                                "attempted_provider_call"
                            ),
                            audit_events=failure_metadata.get(
                                "aawm_alias_routing_audit_events"
                            ),
                            attempts=failure_metadata.get(attempts_metadata_key),
                            skipped_candidates=failure_metadata.get(
                                skipped_candidates_metadata_key
                            ),
                        )
                    except HTTPException as final_exc:
                        _emit_validated_redispatch_terminal_event(
                            exc=final_exc,
                            request=request,
                            alias_family=alias_family,
                            alias_model=alias_model,
                            request_body=prepared_request_body,
                            selection=selection,
                            attempts=attempts,
                            emit_pre_attempt_terminal_event=(
                                _emit_auto_agent_alias_pre_attempt_terminal_event
                            ),
                        )
                        raise
                if account_failover_planned:
                    provider_candidate_attempts = max(
                        0,
                        provider_candidate_attempts - 1,
                    )
                    _mark_auto_agent_alias_request_failover_pending(
                        request,
                        attempt_record,
                    )
                    _record_auto_agent_alias_attempt_failure(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        prepared_request_body=prepared_request_body,
                        selection=selection,
                        attempts=attempts,
                        attempt_record=attempt_record,
                        error_class=error_class,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                    )
                    verbose_proxy_logger.debug(
                        "%s auto-agent alias %s moving from Codex OAuth "
                        "account %s after %s",
                        log_label,
                        alias_model,
                        candidate.get("codex_oauth_account_label"),
                        error_class,
                    )
                    break
                if failover_ordinal > 0:
                    provider_candidate_attempts += 1
                native_grok_retry_eligible = _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
                    is_native_grok_4_5_candidate=(_is_codex_auto_agent_native_grok_4_5_candidate(candidate)),
                    has_continuation_state=has_continuation_state,
                    error_class=error_class,
                    cooldown_scope=cooldown_scope,
                )
                if native_grok_retry_eligible:
                    native_grok_continuation_transient_provider_attempts += 1
                    native_grok_provider_attempt = native_grok_continuation_transient_provider_attempts
                else:
                    native_grok_provider_attempt = 0
                (
                    should_retry_same_candidate,
                    same_candidate_backoff_seconds,
                    native_grok_retry_metadata,
                ) = _plan_codex_auto_agent_native_grok_continuation_transient_retry(
                    is_native_grok_4_5_candidate=(_is_codex_auto_agent_native_grok_4_5_candidate(candidate)),
                    has_continuation_state=has_continuation_state,
                    error_class=error_class,
                    cooldown_scope=cooldown_scope,
                    provider_attempt=native_grok_provider_attempt,
                    provider=str(candidate.get("provider") or "") or None,
                    model=str(candidate.get("model") or "") or None,
                    route_family=str(candidate.get("route_family") or "") or None,
                    max_attempts=native_grok_continuation_transient_max_attempts,
                )
                if native_grok_retry_metadata is not None:
                    attempt_record["native_grok_continuation_retry"] = native_grok_retry_metadata
                _record_auto_agent_alias_attempt_failure(
                    alias_family=alias_family,
                    alias_model=alias_model,
                    request=request,
                    prepared_request_body=prepared_request_body,
                    selection=selection,
                    attempts=attempts,
                    attempt_record=attempt_record,
                    error_class=error_class,
                    add_alias_metadata_fn=add_alias_metadata_fn,
                )
                verbose_proxy_logger.debug(
                    "%s auto-agent alias %s target %s/%s hit %s on attempt %s; " "cooldown %.1fs scope=%s tokens=%s",
                    log_label,
                    alias_model,
                    candidate["provider"],
                    candidate["model"],
                    error_class,
                    len(attempts),
                    cooldown_seconds,
                    cooldown_scope,
                    sorted(error_tokens),
                )
                if should_retry_same_candidate:
                    # Native-grok backoff sleep is NEVER inside the probe lock.
                    if same_candidate_backoff_seconds and same_candidate_backoff_seconds > 0:
                        await asyncio.sleep(same_candidate_backoff_seconds)
                    attempt_record = _codex_auto_agent_candidate_public_shape(
                        candidate,
                        lane_key=selection.get("lane_key"),
                        reason="native_grok_continuation_same_candidate_retry",
                    )
                    attempt_record["attempted_provider_call"] = False
                    continue
                if native_grok_retry_eligible:
                    # Same-candidate budget exhausted; do not switch providers.
                    _raise_terminal_alias_failure(last_retryable_exc)
                break

        finally:
            # D1-564: encompass every post-admission path. Lease must not
            # leak on attempt-record, cooldown precheck, probe-lock,
            # follower-wait, provider I/O, cancellation, or exception.
            if admission_lease is not None:
                try:
                    await _admission_mod().release_provider_lane_admission(
                        admission_lease
                    )
                except Exception:  # noqa: BLE001
                    pass
                admission_lease = None
    if last_retryable_exc is not None:
        _raise_terminal_alias_failure(last_retryable_exc)
    raise HTTPException(
        status_code=429,
        detail=no_candidate_detail,
    )


def _resolve_failure_plan(
    *,
    resolve_cooldown_publication_fn: ResolveCooldownPublicationFn,
    record_codex_failure_evidence_fn: RecordCodexFailureEvidenceFn,
    request: "Request",
    candidate: dict[str, Any],
    selection: dict[str, Any],
    attempt_record: dict[str, Any],
    exc: Exception,
    codex_failure_evidence_alias: Optional[str],
    kimi_failure_metadata_fn: GetKimiFailureMetadataFn,
    classify_kimi_fn: ClassifyKimiFailureFn,
    classify_retryable_fn: ClassifyRetryableFailureFn,
    grok_quota_fn: IsGrokAccountQuotaFailureFn,
    cooldown_seconds_fn: GetCooldownSecondsFn,
    fresh_codex_auth_error_class: Optional[str] = None,
) -> CooldownPublicationPlan:
    """Resolve ONE publication plan for ``exc`` (pure, no I/O).

    The resolver records alias-scoped Codex failure evidence when the caller
    identifies a Codex configured alias, then resolves scope/target keys
    without cooldown-map or durable writes.
    """
    if (
        _error_signals._is_codex_auto_agent_candidate_deterministically_ineligible(
            exc
        )
        and not _is_cursor_session_continuation_failure(exc, candidate=candidate)
    ):
        return CooldownPublicationPlan(
            memory_keys=(),
            durable_keys=(),
            duration_seconds=0.0,
            applied_scope="none",
            request_local_action=None,
            grok_account_quota_exhausted=False,
            kimi_failure_metadata=None,
            allow_ttl_shrink=False,
        )
    raw_attempted_provider_call = attempt_record.get("attempted_provider_call")
    if raw_attempted_provider_call is None:
        raw_attempted_provider_call = getattr(exc, "attempted_provider_call", True)
    attempted_provider_call = bool(raw_attempted_provider_call)
    attempt_record["attempted_provider_call"] = attempted_provider_call

    def _accepts_attempted_provider_call(fn: Any) -> bool:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        parameter = signature.parameters.get("attempted_provider_call")
        return parameter is not None or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )

    kimi_failure_metadata = kimi_failure_metadata_fn(exc, candidate=candidate)
    error_class = classify_kimi_fn(kimi_failure_metadata)
    if error_class is None:
        error_class = _classify_kimi_invalid_request_failure(
            exc,
            candidate=candidate,
            kimi_failure_metadata=kimi_failure_metadata,
        )
    if error_class is None:
        error_class = _classify_codex_cohere_candidate_failure(
            exc,
            candidate=candidate,
            is_codex_alias=codex_failure_evidence_alias is not None,
        )
    if error_class is None:
        error_class = _classify_codex_zai_coding_plan_candidate_failure(
            exc,
            candidate=candidate,
            attempted_provider_call=attempted_provider_call,
        )
    if error_class is None:
        if _accepts_attempted_provider_call(classify_retryable_fn):
            error_class = classify_retryable_fn(
                exc,
                candidate=candidate,
                attempted_provider_call=attempted_provider_call,
            )
        else:
            error_class = classify_retryable_fn(exc, candidate=candidate)
    if error_class is None:
        error_class = fresh_codex_auth_error_class
    grok_account_quota_exhausted = grok_quota_fn(exc, candidate=candidate)
    if _accepts_attempted_provider_call(cooldown_seconds_fn):
        cooldown_seconds = cooldown_seconds_fn(
            exc,
            candidate=candidate,
            attempted_provider_call=attempted_provider_call,
        )
    else:
        cooldown_seconds = cooldown_seconds_fn(exc, candidate=candidate)
    if codex_failure_evidence_alias is not None:
        record_codex_failure_evidence_fn(
            canonical_alias=codex_failure_evidence_alias,
            cooldown_key=selection["cooldown_key"],
            exc=exc,
            attempt_record=attempt_record,
            cooldown_seconds=(
                cooldown_seconds if error_class == "usage_limit_reached" else None
            ),
        )
    plan = resolve_cooldown_publication_fn(
        request=request,
        candidate=candidate,
        lane_key=selection.get("lane_key"),
        selected_cooldown_key=selection["cooldown_key"],
        cooldown_seconds=cooldown_seconds,
        error_class=error_class,
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
        codex_failure_evidence_alias=codex_failure_evidence_alias,
    )
    if getattr(plan, "applied_scope", "none") != "none":
        attempt_record["cooldown_seconds"] = round(
            float(getattr(plan, "duration_seconds", 0.0)),
            3,
        )
        attempt_record["cooldown_scope"] = getattr(
            plan,
            "applied_scope",
            "none",
        )
    return plan
