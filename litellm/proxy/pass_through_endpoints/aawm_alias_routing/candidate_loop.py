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
import hashlib
from typing import TYPE_CHECKING, Any, Optional

import httpx

from litellm.proxy.aawm_route_logging import (
    register_aawm_route_rollup_access_log_replacement,
)
from litellm.llms.zai_coding_plan.failure_classification import (
    ZAICodingPlanFailureKind,
    classify_zai_coding_plan_failure,
)
from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.cohere import (
    classify_cohere_failure,
)

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
    ZAICodingPlanFailureKind.VALIDATION: "candidate_unavailable",
    ZAICodingPlanFailureKind.ROUTING: "provider_terminal_error",
}


def _classify_codex_zai_coding_plan_candidate_failure(
    exc: Exception,
    *,
    candidate: Optional[dict[str, Any]],
) -> Optional[str]:
    """Map Coding Plan business codes onto the shared Codex retry vocabulary.

    1113 on the coding base is a wrong-base / wrong-key routing defect, not
    ordinary-balance recharge. Unknown codes return ``None`` so generic
    classifiers can still inspect HTTP status.
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
    if failure.kind == ZAICodingPlanFailureKind.UNKNOWN:
        return None
    return _ZAI_CODING_PLAN_KIND_TO_ERROR_CLASS.get(failure.kind)


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

    _codex_auto_agent_request_has_continuation_state = _lpe._codex_auto_agent_request_has_continuation_state
    _get_codex_auto_agent_native_grok_continuation_transient_max_attempts = (
        _lpe._get_codex_auto_agent_native_grok_continuation_transient_max_attempts
    )
    _codex_auto_agent_candidate_public_shape = _lpe._codex_auto_agent_candidate_public_shape
    _record_auto_agent_alias_attempt_started = _lpe._record_auto_agent_alias_attempt_started
    _is_auto_agent_alias_in_flight_cooldown_http_exception = _lpe._is_auto_agent_alias_in_flight_cooldown_http_exception
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
    codex_failure_evidence_alias = (
        alias_model if validate_alias_family(alias_family) == "codex" else None
    )

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
    account_failover_attempts_by_slot: dict[Optional[str], int] = {}
    same_account_transient_attempts_by_slot: dict[Optional[str], int] = {}

    def _raise_terminal_alias_failure(exc: Exception) -> Any:
        last_attempt = attempts[-1] if attempts else {}
        kimi_failure_metadata = _get_safe_kimi_code_probe_failure_metadata(
            exc,
            candidate=candidate if isinstance(candidate, dict) else None,
        )
        if (
            kimi_failure_metadata is not None
            and kimi_failure_metadata.get("kind") == "malformed"
            and kimi_failure_metadata.get("scope") == "none"
        ):
            detail = getattr(exc, "detail", None)
            if (
                isinstance(detail, dict)
                and isinstance(detail.get("error"), dict)
                and detail["error"].get("code") == "kimi_code_invalid_request"
            ):
                status_code = int(
                    kimi_failure_metadata.get("status_code")
                    or getattr(exc, "status_code", 400)
                    or 400
                )
                if status_code not in {400, 422}:
                    status_code = 400
                last_attempt_record = attempts[-1] if attempts else None
                if isinstance(last_attempt_record, dict):
                    _mark_auto_agent_alias_request_terminal_failure(
                        request,
                        last_attempt_record,
                    )
                raise HTTPException(
                    status_code=status_code,
                    detail=detail,
                ) from None
        error_class = str(
            last_attempt.get("error_class")
            or _classify_codex_auto_agent_retryable_exhaustion(
                exc, candidate=candidate
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
        )
        raise terminal_exc from None

    while provider_candidate_attempts < max_candidate_attempts:
        try:
            selection = await select_candidate_fn(
                request=request,
                request_body=prepared_request_body,
            )
        except HTTPException as exc:
            if exc.status_code == 429 and not _is_auto_agent_alias_in_flight_cooldown_http_exception(exc):
                _emit_auto_agent_alias_no_candidate_event(
                    alias_family=alias_family,
                    alias_model=alias_model,
                    request=request,
                    request_body=prepared_request_body,
                    exc=exc,
                    attempts=attempts,
                )
            raise
        candidate = selection["candidate"]
        failover_ordinal = int(selection.get("failover_ordinal") or 0)
        if failover_ordinal > 0:
            account_failover_slot = _codex_oauth_candidate_slot(candidate)
            account_failover_attempts = (
                account_failover_attempts_by_slot.get(
                    account_failover_slot,
                    0,
                )
                + 1
            )
            account_failover_attempts_by_slot[account_failover_slot] = (
                account_failover_attempts
            )
            if account_failover_attempts > 1:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": {
                            "message": (
                                "Codex OAuth account failover limit was "
                                "reached before dispatch."
                            ),
                            "type": "rate_limit_error",
                            "code": (
                                "aawm_codex_oauth_account_failover_limit"
                            ),
                        }
                    },
                )
        else:
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
        attempts.append(attempt_record)
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
            admission.raise_provider_lane_admission_rejected(
                admission_decision,
                candidate=candidate,
                alias_model=alias_model,
                alias_family=alias_family,
                lane_key=selection.get("lane_key"),
            )
        admission_lease = admission_decision.lease
        try:
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
                        raise probe_failure_exc
                    break

                # BaseException-safe: intent is ALWAYS completed and removed,
                # probe lock is ALWAYS released, regardless of exception type
                # (Exception, CancelledError, KeyboardInterrupt, etc.).
                try:
                    session_owner_lease = None
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
                        response = await perform_candidate_request_fn(
                            candidate=candidate,
                            candidate_body=candidate_body,
                        )
                        # Authoritative success: promote reserved -> owned.
                        promote_result = await sa.finalize_session_owner_lease_on_success(
                            session_owner_lease,
                            attributes=owner_attributes,
                            candidate=candidate,
                        )
                        if promote_result is not None and promote_result.outcome in {
                            sa.SessionOwnerMutationOutcome.CONFLICT,
                            sa.SessionOwnerMutationOutcome.ERROR,
                            sa.SessionOwnerMutationOutcome.NOT_HELD,
                        }:
                            # Success bytes may already be in flight to the client;
                            # fail closed for subsequent requests by not treating
                            # ownership as established. Still surface structured error
                            # for non-streaming callers by raising.
                            sa.raise_session_owner_redispatch_required(
                                session_identity=session_owner_identity,
                                mutation=promote_result,
                                alias_model=selection.get("alias_model") or alias_model,
                                candidate=candidate,
                                failure_phase="session_owner_promote_after_success",
                                request=request,
                            )
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

                    # Resolve the plan AFTER lock release.  If the resolver
                    # raises, the outer BaseException handler cleans up the
                    # intent.  No lock is held here (canonical order: no family
                    # lock entry while retaining a pre-acquired probe lock).
                    if probe_failure_exc is not None:
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
                            )
                        )
                    if early_pre_commit_error_class is None:
                        early_pre_commit_error_class = (
                            _classify_codex_auto_agent_retryable_exhaustion(
                                probe_failure_exc, candidate=candidate
                            )
                        )
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
                    skip_cooldown_for_same_account_retry = (
                        early_pre_commit_retry_plan["action"]
                        in {"retry_same_account", "pre_stream_unavailable"}
                    )

                    # --- Cooldown mutation: NO pre-held probe lock ------------
                    if (
                        probe_failure_plan is not None
                        and not skip_cooldown_for_same_account_retry
                    ):
                        await _lpe.execute_cooldown_publication_transaction(
                            alias_family=alias_family,
                            candidate=candidate,
                            plan=probe_failure_plan,
                            publish_cooldown_memory_fn=publish_cooldown_memory_fn,
                            persist_cooldown_fn=persist_cooldown_fn,
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

                # --- failure handling (post-release) ---------------------------
                failure_exc = probe_failure_exc
                assert failure_exc is not None
                kimi_failure_metadata = _get_safe_kimi_code_probe_failure_metadata(
                    failure_exc,
                    candidate=candidate,
                )
                error_class = _classify_kimi_code_auto_agent_probe_failure(kimi_failure_metadata)
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
                    )
                if error_class is None:
                    error_class = _classify_codex_auto_agent_retryable_exhaustion(
                        failure_exc, candidate=candidate
                    )
                if error_class is None:
                    raise failure_exc
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
                if pre_commit_retry_plan["action"] == "retry_same_account":
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
                    attempts.append(attempt_record)
                    _record_auto_agent_alias_attempt_started(
                        alias_family=alias_family,
                        alias_model=alias_model,
                        request=request,
                        prepared_request_body=prepared_request_body,
                        selection=selection,
                        attempts=attempts,
                        attempt_record=attempt_record,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                    )
                    continue
                if pre_commit_retry_plan["action"] == "pre_stream_unavailable":
                    attempt_record["pre_commit_retry"] = {
                        "action": pre_commit_retry_plan["action"],
                        "error_class": error_class,
                        "wait_seconds": pre_commit_retry_plan["wait_seconds"],
                        "apply_account_exhaustion_cooldown": False,
                        "retryable": True,
                    }
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
                                int(pre_commit_retry_plan["wait_seconds"] or 10)
                            )
                        },
                    )
                cooldown_seconds = _get_codex_auto_agent_cooldown_seconds(
                    failure_exc,
                    candidate=candidate,
                )
                # The plan was resolved inside the probe lock above.  After probe
                # lock release, execute_cooldown_publication_transaction performed
                # the atomic memory publish + durable commit + local index update
                # under the family lock + sorted probe locks (canonical order).
                # Post-release only the request-local action remains.
                plan = probe_failure_plan
                assert plan is not None
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
                )
                if cooldown_scope == "none" and not has_continuation_state:
                    _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
                        request,
                        candidate=candidate,
                        lane_key=selection.get("lane_key"),
                    )
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
                        attempted_provider_call=attempt_record.get("attempted_provider_call"),
                        audit_events=failure_metadata.get("aawm_alias_routing_audit_events"),
                        attempts=failure_metadata.get(attempts_metadata_key),
                        skipped_candidates=failure_metadata.get(skipped_candidates_metadata_key),
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
                        error_class=error_class,
                        add_alias_metadata_fn=add_alias_metadata_fn,
                    )
                    verbose_proxy_logger.debug(
                        "%s auto-agent alias %s moving once from Codex OAuth "
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
                    attempts.append(attempt_record)
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
) -> CooldownPublicationPlan:
    """Resolve ONE publication plan for ``exc`` (pure, no I/O).

    The resolver records alias-scoped Codex failure evidence when the caller
    identifies a Codex configured alias, then resolves scope/target keys
    without cooldown-map or durable writes.
    """
    kimi_failure_metadata = kimi_failure_metadata_fn(exc, candidate=candidate)
    error_class = classify_kimi_fn(kimi_failure_metadata)
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
        )
    if error_class is None:
        error_class = classify_retryable_fn(exc, candidate=candidate)
    grok_account_quota_exhausted = grok_quota_fn(exc, candidate=candidate)
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
    return resolve_cooldown_publication_fn(
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
