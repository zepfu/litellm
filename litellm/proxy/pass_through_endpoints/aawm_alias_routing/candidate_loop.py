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
    RecordCodexFailureEvidenceFn,
    ResolveCooldownPublicationFn,
)
from .state import alias_routing_state, validate_alias_family

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

    Mirrors ``cooldown_apply.resolve_lane_identity_hash``: identity is derived
    from ``provider:model:route_family`` ONLY -- never from lane_key or
    credentials -- so one identity maps to many credential-derived lane keys.
    """
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
    codex_failure_evidence_alias = (
        alias_model if validate_alias_family(alias_family) == "codex" else None
    )

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
                try:
                    response = await perform_candidate_request_fn(
                        candidate=candidate,
                        candidate_body=candidate_body,
                    )
                except Exception as probe_exc:  # noqa: PERF203
                    probe_failure_exc = probe_exc
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
                    return response

                # --- Cooldown mutation: NO pre-held probe lock ------------
                if probe_failure_plan is not None:
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
                error_class = _classify_codex_auto_agent_retryable_exhaustion(failure_exc)
            if error_class is None:
                raise failure_exc
            last_retryable_exc = failure_exc
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
        error_class = classify_retryable_fn(exc)
    grok_account_quota_exhausted = grok_quota_fn(exc, candidate=candidate)
    cooldown_seconds = cooldown_seconds_fn(exc, candidate=candidate)
    if codex_failure_evidence_alias is not None:
        record_codex_failure_evidence_fn(
            canonical_alias=codex_failure_evidence_alias,
            cooldown_key=selection["cooldown_key"],
            exc=exc,
            attempt_record=attempt_record,
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
