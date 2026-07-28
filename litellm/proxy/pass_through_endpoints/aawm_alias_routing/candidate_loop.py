"""Shared auto-agent alias candidate retry loop (Wave 2 extraction + R3-1).

This is the moved body of ``llm_passthrough_endpoints._handle_auto_agent_alias_route``
restructured for R3-1 exact-key single-flight publication:

- The locked region is WIDENED. The exception from the upstream ``perform`` is
  caught INSIDE the ``try`` that still holds the per-candidate ``probe_lock``.
  While holding the lock, the loop runs the pure publication-plan resolver to
  classify the failure, record read-pilot evidence, and produce ONE immutable
  :class:`CooldownPublicationPlan`, then publishes every ``plan.memory_keys``
  (direct ``state.py`` writes for the production seams) BEFORE releasing the
  probe lock. A follower queued on the same probe lock therefore cannot acquire
  it until the cooldown is already visible, so it never re-probes (single-flight).
- AFTER releasing the lock, the loop persists exactly ``plan.durable_keys`` to
  Redis, applies ``plan.request_local_action``, updates the attempt record with
  ``plan.applied_scope``, signals redispatch, and runs the native-grok backoff
  ``asyncio.sleep`` (never inside the lock).

Memory and durable targets are derived once from the same plan so telemetry,
waiter visibility, and Redis state cannot disagree, and no target-key logic is
duplicated between the in-lock and post-release paths.

The production memory publish writes ``state.py`` directly with no awaitable
lock. Durable Redis writes and the legacy async applicator are post-release, so
the probe lock is never held across network I/O.

The god-module is imported lazily inside :func:`handle_alias_route` to avoid a
module-scope import cycle (the god-module imports this package); the loop
otherwise depends only on the typed :class:`AliasRouteServices` seams.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from litellm.proxy.aawm_route_logging import (
    register_aawm_route_rollup_access_log_replacement,
)

from .interfaces import (
    AliasRouteServices,
    ClassifyKimiFailureFn,
    ClassifyRetryableFailureFn,
    CooldownPublicationPlan,
    GetCooldownSecondsFn,
    GetActiveCooldownStateFn,
    GetKimiFailureMetadataFn,
    IsGrokAccountQuotaFailureFn,
    PublishCooldownMemoryFn,
    RecordReadPilotEvidenceFn,
    ResolveCooldownPublicationFn,
)
from .state import alias_routing_state

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request
    from starlette.responses import Response

    from .types import Payload


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
    _READ_PILOT_ALIAS_NAME = _lpe._READ_PILOT_ALIAS_NAME
    _record_read_pilot_cooldown_evidence = _lpe._record_read_pilot_cooldown_evidence
    _update_codex_auto_agent_retryable_attempt_record = _lpe._update_codex_auto_agent_retryable_attempt_record
    _exclude_codex_auto_agent_request_local_candidate_without_cooldown = (
        _lpe._exclude_codex_auto_agent_request_local_candidate_without_cooldown
    )
    _apply_request_local_cooldown_from_plan = _lpe._apply_request_local_cooldown_from_plan
    _record_auto_agent_alias_attempt_failure = _lpe._record_auto_agent_alias_attempt_failure
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

    register_aawm_route_rollup_access_log_replacement(request)
    attempts: list[dict[str, Any]] = []
    last_retryable_exc: Optional[Exception] = None
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(prepared_request_body)
    native_grok_continuation_transient_max_attempts = (
        _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
    )
    # Request-scoped total for eligible native Grok continuation transient
    # attempts. Must not reset when the outer candidate-selection loop re-enters.
    native_grok_continuation_transient_provider_attempts = 0

    def _raise_terminal_alias_failure(exc: Exception) -> Any:
        last_attempt = attempts[-1] if attempts else {}
        error_class = str(
            last_attempt.get("error_class")
            or _classify_codex_auto_agent_retryable_exhaustion(exc)
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
        _emit_auto_agent_alias_no_candidate_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=prepared_request_body,
            exc=terminal_exc,
            attempts=attempts,
        )
        raise terminal_exc from None

    for _attempt_number in range(max_candidate_attempts):
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
        attempt_record = _codex_auto_agent_candidate_public_shape(
            candidate,
            lane_key=selection.get("lane_key"),
            reason=selection.get("selection_reason"),
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
        while True:
            # R3-1: the locked region is widened. ``perform`` runs inside the
            # ``try`` that holds ``probe_lock``; on failure the publication plan
            # is resolved and its memory keys are published while still holding
            # the lock, so a queued follower observes the cooldown before it can
            # acquire the lock and re-probe.
            probe_failure_exc: Optional[Exception] = None
            probe_failure_plan: Optional[CooldownPublicationPlan] = None
            skip_after_probe_wait = False
            response: Optional[Response] = None
            probe_lock = await alias_routing_state.candidate_probe_lock(
                alias_family=alias_family,
                cooldown_key=selection["cooldown_key"],
            )
            await probe_lock.acquire()
            try:
                active_seconds, _active_source = await get_active_cooldown_state_fn(selection["cooldown_key"])
                if active_seconds > 0:
                    skip_after_probe_wait = True
                    attempt_record["status"] = "skipped_single_flight_cooldown"
                    attempt_record["cooldown_seconds"] = active_seconds
                else:
                    response = await perform_candidate_request_fn(
                        candidate=candidate,
                        candidate_body=candidate_body,
                    )
            except Exception as probe_exc:  # noqa: PERF203
                probe_failure_exc = probe_exc
            finally:
                # Resolve the publication plan + publish its memory keys
                # BEFORE releasing the lock so the single-flight invariant
                # holds when the synchronous publisher returns.
                # The nested try/finally guarantees the lock is released
                # even if the resolver or publisher raises or is cancelled,
                # preventing a permanent lock leak that would hang all
                # subsequent same-key requests.
                try:
                    if probe_failure_exc is not None:
                        probe_failure_plan = _resolve_and_publish_failure_memory(
                            resolve_cooldown_publication_fn=resolve_cooldown_publication_fn,
                            publish_cooldown_memory_fn=publish_cooldown_memory_fn,
                            record_read_pilot_evidence_fn=_record_read_pilot_cooldown_evidence,
                            request=request,
                            candidate=candidate,
                            selection=selection,
                            alias_model=alias_model,
                            attempt_record=attempt_record,
                            exc=probe_failure_exc,
                            is_read_pilot_lane=(alias_model == _READ_PILOT_ALIAS_NAME),
                            kimi_failure_metadata_fn=_get_safe_kimi_code_probe_failure_metadata,
                            classify_kimi_fn=_classify_kimi_code_auto_agent_probe_failure,
                            classify_retryable_fn=_classify_codex_auto_agent_retryable_exhaustion,
                            grok_quota_fn=_is_codex_auto_agent_grok_account_quota_exhaustion,
                            cooldown_seconds_fn=_get_codex_auto_agent_cooldown_seconds,
                        )
                finally:
                    probe_lock.release()
            if skip_after_probe_wait:
                break
            if probe_failure_exc is None:
                await set_session_affinity_fn(
                    selection.get("session_key"),
                    candidate,
                )
                assert response is not None
                return response

            # --- failure handling (post-release) ---------------------------
            failure_exc = probe_failure_exc
            assert failure_exc is not None
            kimi_failure_metadata = _get_safe_kimi_code_probe_failure_metadata(
                failure_exc,
                candidate=candidate,
            )
            error_class = _classify_kimi_code_auto_agent_probe_failure(kimi_failure_metadata)
            if error_class is None:
                error_class = _classify_codex_auto_agent_retryable_exhaustion(failure_exc)
            if error_class is None:
                raise failure_exc
            last_retryable_exc = failure_exc
            cooldown_seconds = _get_codex_auto_agent_cooldown_seconds(
                failure_exc,
                candidate=candidate,
            )
            # R3-1: the plan was resolved + memory-published inside the probe
            # lock above. Post-release we persist the durable keys, apply the
            # request-local action, and report the applied scope. No target-key
            # logic is recomputed here.
            plan = probe_failure_plan
            assert plan is not None
            if plan.durable_keys:
                await persist_cooldown_fn(keys=plan.durable_keys, seconds=plan.duration_seconds)
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
            if cooldown_scope == "none" and not has_continuation_state:
                _exclude_codex_auto_agent_request_local_candidate_without_cooldown(
                    request,
                    candidate=candidate,
                    lane_key=selection.get("lane_key"),
                )
            if has_continuation_state and cooldown_scope != "none":
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

    if last_retryable_exc is not None:
        _raise_terminal_alias_failure(last_retryable_exc)
    raise HTTPException(
        status_code=429,
        detail=no_candidate_detail,
    )


def _resolve_and_publish_failure_memory(
    *,
    resolve_cooldown_publication_fn: ResolveCooldownPublicationFn,
    publish_cooldown_memory_fn: PublishCooldownMemoryFn,
    record_read_pilot_evidence_fn: RecordReadPilotEvidenceFn,
    request: "Request",
    candidate: dict[str, Any],
    selection: dict[str, Any],
    alias_model: str,
    attempt_record: dict[str, Any],
    exc: Exception,
    is_read_pilot_lane: bool,
    kimi_failure_metadata_fn: GetKimiFailureMetadataFn,
    classify_kimi_fn: ClassifyKimiFailureFn,
    classify_retryable_fn: ClassifyRetryableFailureFn,
    grok_quota_fn: IsGrokAccountQuotaFailureFn,
    cooldown_seconds_fn: GetCooldownSecondsFn,
) -> CooldownPublicationPlan:
    """Resolve ONE publication plan for ``exc`` and publish its memory keys.

    Called while the probe lock is held. The resolver is pure (it records
    read-pilot evidence and resolves scope/target keys but performs no I/O);
    this helper then publishes every ``plan.memory_keys`` so a queued follower
    observes the cooldown before the lock is released. The publisher is
    strictly synchronous; durable I/O remains post-release. The plan's
    ``applied_scope`` is authoritative.
    """
    kimi_failure_metadata = kimi_failure_metadata_fn(exc, candidate=candidate)
    error_class = classify_kimi_fn(kimi_failure_metadata)
    if error_class is None:
        error_class = classify_retryable_fn(exc)
    grok_account_quota_exhausted = grok_quota_fn(exc, candidate=candidate)
    cooldown_seconds = cooldown_seconds_fn(exc, candidate=candidate)
    # For the read lane only, record this attempt's failure evidence into the
    # N-of-M gate BEFORE resolving the plan, keyed on the live
    # ``provider:model:lane`` key, so the gate's decision drives the applied
    # cooldown for the same attempt.
    if is_read_pilot_lane:
        record_read_pilot_evidence_fn(
            cooldown_key=selection["cooldown_key"],
            exc=exc,
            attempt_record=attempt_record,
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
        is_read_pilot_lane=is_read_pilot_lane,
    )
    if plan.memory_keys:
        publish_cooldown_memory_fn(
            keys=plan.memory_keys,
            seconds=plan.duration_seconds,
        )
    return plan
