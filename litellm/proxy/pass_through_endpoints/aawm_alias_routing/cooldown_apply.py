"""Cooldown publication-plan resolution and application for alias routing.

Wave 5C extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Owns:
- ``_resolve_auto_agent_cooldown_publication_plan`` (pure resolver)
- ``_persist_codex_cooldown_durable`` / ``_persist_anthropic_cooldown_durable``
- ``_apply_auto_agent_alias_cooldown`` (shared apply)
- ``_apply_codex_auto_agent_alias_cooldown`` / ``_apply_anthropic_auto_agent_alias_cooldown``
- ``_apply_basic_pilot_gated_cooldown``
- ``_set_codex_auto_agent_candidate_cooldowns`` (compatibility entry point)

Dependencies on error_signals.py, selection.py, cooldown_state.py, durable.py,
and state.py are injected via :func:`configure_cooldown_apply_runtime`.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Awaitable, Callable, Optional, Sequence

from fastapi import Request

from .interfaces import CooldownPublicationPlan
from .types import Payload

# ---------------------------------------------------------------------------
# Injected runtime seams
# ---------------------------------------------------------------------------

# error_signals.py owned
_get_candidate_cooldown_scope: Optional[
    Callable[..., str]
] = None
_get_kimi_managed_account_cooldown_key: Optional[Callable[[], str]] = None
_get_grok_account_quota_lane_cooldown_key: Optional[
    Callable[[Any, Optional[str]], Optional[str]]
] = None

# selection.py owned (request-local helpers)
_get_request_local_cooldown_key: Optional[
    Callable[..., str]
] = None
_set_request_local_cooldown: Optional[
    Callable[..., None]
] = None
_exclude_request_local_candidate: Optional[
    Callable[..., None]
] = None

# cooldown_state.py owned (durable candidate setters)
_set_codex_cooldown: Optional[Callable[[str, float], Awaitable[object]]] = None
_set_anthropic_cooldown: Optional[Callable[[str, float], Awaitable[object]]] = None

# durable.py owned
_write_durable_payload: Optional[Callable[..., Awaitable[object]]] = None

# state.py owned (basic-pilot gate + memory publication)
_basic_pilot_gate: Optional[Any] = None
_state_manager: Optional[Any] = None


# Reference to host_globals set by install(); configure updates it too.
_host_globals_ref: dict | None = None


def configure_cooldown_apply_runtime(
    *,
    get_candidate_cooldown_scope: Callable[..., str],
    get_kimi_managed_account_cooldown_key: Callable[[], str],
    get_grok_account_quota_lane_cooldown_key: Callable[[Any, Optional[str]], Optional[str]],
    get_request_local_cooldown_key: Callable[..., str],
    set_request_local_cooldown: Callable[..., None],
    exclude_request_local_candidate: Callable[..., None],
    set_codex_cooldown: Callable[[str, float], Awaitable[object]],
    set_anthropic_cooldown: Callable[[str, float], Awaitable[object]],
    write_durable_payload: Callable[..., Awaitable[object]],
    basic_pilot_gate: Any,
    state_manager: Any,


) -> None:
    """Bind error_signals / selection / cooldown_state / durable / state seams."""
    global _get_candidate_cooldown_scope
    _get_candidate_cooldown_scope = get_candidate_cooldown_scope
    global _get_kimi_managed_account_cooldown_key
    _get_kimi_managed_account_cooldown_key = get_kimi_managed_account_cooldown_key
    global _get_grok_account_quota_lane_cooldown_key
    _get_grok_account_quota_lane_cooldown_key = get_grok_account_quota_lane_cooldown_key
    global _get_request_local_cooldown_key
    _get_request_local_cooldown_key = get_request_local_cooldown_key
    global _set_request_local_cooldown
    _set_request_local_cooldown = set_request_local_cooldown
    global _exclude_request_local_candidate
    _exclude_request_local_candidate = exclude_request_local_candidate
    global _set_codex_cooldown
    _set_codex_cooldown = set_codex_cooldown
    global _set_anthropic_cooldown
    _set_anthropic_cooldown = set_anthropic_cooldown
    global _write_durable_payload
    _write_durable_payload = write_durable_payload
    global _basic_pilot_gate
    _basic_pilot_gate = basic_pilot_gate
    global _state_manager
    _state_manager = state_manager
    # If install() has been called, also update host_globals so rebound
    # functions see the new seam values.
    if _host_globals_ref is not None:
        _mod = globals()
        _host_globals_ref["_get_candidate_cooldown_scope"] = _mod["_get_candidate_cooldown_scope"]
        _host_globals_ref["_get_kimi_managed_account_cooldown_key"] = _mod["_get_kimi_managed_account_cooldown_key"]
        _host_globals_ref["_get_grok_account_quota_lane_cooldown_key"] = _mod["_get_grok_account_quota_lane_cooldown_key"]
        _host_globals_ref["_get_request_local_cooldown_key"] = _mod["_get_request_local_cooldown_key"]
        _host_globals_ref["_set_request_local_cooldown"] = _mod["_set_request_local_cooldown"]
        _host_globals_ref["_exclude_request_local_candidate"] = _mod["_exclude_request_local_candidate"]
        _host_globals_ref["_set_codex_cooldown"] = _mod["_set_codex_cooldown"]
        _host_globals_ref["_set_anthropic_cooldown"] = _mod["_set_anthropic_cooldown"]
        _host_globals_ref["_write_durable_payload"] = _mod["_write_durable_payload"]
        _host_globals_ref["_basic_pilot_gate"] = _mod["_basic_pilot_gate"]
        _host_globals_ref["_state_manager"] = _mod["_state_manager"]


# ---------------------------------------------------------------------------
# Publication-plan resolver (R3-1)
# ---------------------------------------------------------------------------


def _resolve_auto_agent_cooldown_publication_plan(
    *,
    request: Optional[Request],
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    is_basic_pilot_lane: bool = False,
) -> CooldownPublicationPlan:
    """Pure resolver: classify one failure into an immutable publication plan (R3-1).

    Resolves the cooldown scope and derives the exact memory/durable target
    keys WITHOUT performing any I/O, so the retry loop can publish the memory
    keys synchronously inside the probe lock and persist the durable keys after
    release -- both derived from this single plan so telemetry, waiter
    visibility, and Redis state cannot disagree.

    Scope targets (preserved exactly from the legacy apply chain):
      - ``none`` / request-local -> no shared keys (request-local action only)
      - ``candidate`` / ``model`` -> the selected candidate key
      - Kimi ``managed_account`` -> the managed-account sentinel ONLY
      - Grok account-quota -> the selected key PLUS the account-lane key

    The basic-pilot lane resolves its scope/duration from the N-of-M evidence
    gate's current decision (fed earlier by the loop via
    ``_record_basic_pilot_cooldown_evidence``); when the gate says do-not-cool,
    the plan carries ``applied_scope="none"`` and empty key sets.
    """
    assert _get_candidate_cooldown_scope is not None
    assert _get_kimi_managed_account_cooldown_key is not None
    assert _get_grok_account_quota_lane_cooldown_key is not None
    assert _basic_pilot_gate is not None

    if is_basic_pilot_lane:
        decision = _basic_pilot_gate.current_decision(cooldown_key=selected_cooldown_key)
        if not decision.should_cool:
            return CooldownPublicationPlan(
                applied_scope="none",
                duration_seconds=0.0,
                grok_account_quota_exhausted=grok_account_quota_exhausted,
                kimi_failure_metadata=kimi_failure_metadata,
            )
        return CooldownPublicationPlan(
            memory_keys=(selected_cooldown_key,),
            durable_keys=(selected_cooldown_key,),
            duration_seconds=float(decision.duration_seconds),
            applied_scope=decision.scope or "candidate",
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
        )

    cooldown_scope = _get_candidate_cooldown_scope(
        error_class,
        candidate=candidate,
        kimi_failure_metadata=kimi_failure_metadata,
    )
    duration = max(0.0, float(cooldown_seconds))
    if cooldown_scope == "none":
        return CooldownPublicationPlan(
            applied_scope="none",
            duration_seconds=duration,
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
        )
    if cooldown_scope == "managed_account":
        managed_key = _get_kimi_managed_account_cooldown_key()
        return CooldownPublicationPlan(
            memory_keys=(managed_key,),
            durable_keys=(managed_key,),
            duration_seconds=duration,
            applied_scope="managed_account",
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
        )
    if cooldown_scope == "candidate":
        memory_keys = [selected_cooldown_key]
        if grok_account_quota_exhausted:
            lane_cooldown_key = _get_grok_account_quota_lane_cooldown_key(
                candidate,
                lane_key,
            )
            if lane_cooldown_key is not None and lane_cooldown_key != selected_cooldown_key:
                memory_keys.append(lane_cooldown_key)
        keys = tuple(memory_keys)
        return CooldownPublicationPlan(
            memory_keys=keys,
            durable_keys=keys,
            duration_seconds=duration,
            applied_scope="candidate",
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
        )
    # request_local: no shared keys; the loop applies the request-local
    # cooldown + exclusion post-release.
    return CooldownPublicationPlan(
        applied_scope=cooldown_scope,
        duration_seconds=duration,
        request_local_action="request_local_cooldown",
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
    )


# ---------------------------------------------------------------------------
# Durable persistence (post-release, R3-1)
# ---------------------------------------------------------------------------


async def _persist_codex_cooldown_durable(*, keys: Sequence[str], seconds: float) -> None:
    """Persist codex cooldown keys to durable Redis (post-release, R3-1)."""
    assert _write_durable_payload is not None
    ttl_seconds = max(0.0, float(seconds))
    if ttl_seconds <= 0:
        return
    for key in keys:
        await _write_durable_payload(
            alias_family="codex",
            state_kind="cooldown",
            state_key=key,
            payload={"cooldown_key": key},
            ttl_seconds=ttl_seconds,
        )


async def _persist_anthropic_cooldown_durable(*, keys: Sequence[str], seconds: float) -> None:
    """Persist anthropic cooldown keys to durable Redis (post-release, R3-1)."""
    assert _write_durable_payload is not None
    ttl_seconds = max(0.0, float(seconds))
    if ttl_seconds <= 0:
        return
    for key in keys:
        await _write_durable_payload(
            alias_family="anthropic",
            state_kind="cooldown",
            state_key=key,
            payload={"cooldown_key": key},
            ttl_seconds=ttl_seconds,
        )


# ---------------------------------------------------------------------------
# Shared cooldown application (RR-054 #12)
# ---------------------------------------------------------------------------


async def _apply_auto_agent_alias_cooldown(
    *,
    request: Request,
    candidate: Payload,
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    set_candidate_cooldown: Callable[[str, float], Awaitable[object]],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Shared auto-agent cooldown apply (RR-054 #12).

    Codex and Anthropic families share scope resolution and request-local
    exclusion; only the durable candidate setter differs.
    """
    assert _get_candidate_cooldown_scope is not None
    assert _get_kimi_managed_account_cooldown_key is not None
    assert _get_grok_account_quota_lane_cooldown_key is not None
    assert _get_request_local_cooldown_key is not None
    assert _set_request_local_cooldown is not None
    assert _exclude_request_local_candidate is not None

    cooldown_scope = _get_candidate_cooldown_scope(
        error_class,
        candidate=candidate,
        kimi_failure_metadata=kimi_failure_metadata,
    )
    if cooldown_scope == "none":
        return cooldown_scope
    if cooldown_scope == "managed_account":
        await set_candidate_cooldown(
            _get_kimi_managed_account_cooldown_key(),
            cooldown_seconds,
        )
        return cooldown_scope
    if cooldown_scope == "candidate":
        await set_candidate_cooldown(
            selected_cooldown_key,
            cooldown_seconds,
        )
        if grok_account_quota_exhausted:
            lane_cooldown_key = _get_grok_account_quota_lane_cooldown_key(
                candidate,
                lane_key,
            )
            if lane_cooldown_key is not None and lane_cooldown_key != selected_cooldown_key:
                await set_candidate_cooldown(
                    lane_cooldown_key,
                    cooldown_seconds,
                )
        return cooldown_scope

    request_local_key = _get_request_local_cooldown_key(
        candidate=candidate,
        lane_key=lane_key,
    )
    _set_request_local_cooldown(
        request,
        cooldown_key=request_local_key,
        cooldown_seconds=cooldown_seconds,
    )
    _exclude_request_local_candidate(
        request,
        cooldown_key=request_local_key,
    )
    return cooldown_scope


# ---------------------------------------------------------------------------
# Codex / Anthropic family wrappers
# ---------------------------------------------------------------------------


async def _apply_codex_auto_agent_alias_cooldown(
    *,
    request: Request,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    is_basic_pilot_lane: bool = False,
) -> str:
    # Route the basic-alias lane to the N-of-M evidence gate by ALIAS identity
    # (``is_basic_pilot_lane``), not by a synthetic ``basic_pilot:`` key prefix.
    # The live selector builds ordinary ``provider:model:lane`` cooldown keys,
    # so the gate now drives the applied cooldown for the basic lane using that
    # exact live key -- the same key the retry loop fed evidence to.
    assert _set_codex_cooldown is not None
    if is_basic_pilot_lane:
        return await _apply_basic_pilot_gated_cooldown(
            selected_cooldown_key=selected_cooldown_key,
            set_candidate_cooldown=_set_codex_cooldown,
        )
    return await _apply_auto_agent_alias_cooldown(
        request=request,
        candidate=candidate,
        lane_key=lane_key,
        selected_cooldown_key=selected_cooldown_key,
        cooldown_seconds=cooldown_seconds,
        error_class=error_class,
        set_candidate_cooldown=_set_codex_cooldown,
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
    )


async def _apply_basic_pilot_gated_cooldown(
    *,
    selected_cooldown_key: str,
    set_candidate_cooldown: Callable[[str, float], Awaitable[object]],
) -> str:
    """Apply the ``CooldownEvidenceGate``'s decision for the basic-pilot lane.

    Delegates to the pure publication-plan resolver
    (:func:`_resolve_auto_agent_cooldown_publication_plan`) so this applicator
    no longer owns a separate memory target or a fire-and-forget durable
    target: the resolver derives the gate-driven scope/duration and the single
    candidate key, this function publishes the memory key synchronously, and
    the durable write is best-effort (must not block the selector-observed
    value). The basic-pilot lane's cooldown-worthiness is decided by
    ``_basic_pilot_cooldown_gate`` (fed via ``_record_basic_pilot_cooldown_evidence``
    on failure); when the gate says "do not cool yet", no cooldown is applied.
    """
    assert _state_manager is not None
    plan = _resolve_auto_agent_cooldown_publication_plan(
        request=None,
        candidate={},
        lane_key=None,
        selected_cooldown_key=selected_cooldown_key,
        cooldown_seconds=0.0,
        error_class=None,
        is_basic_pilot_lane=True,
    )
    if plan.applied_scope == "none" or not plan.memory_keys:
        return "none"
    # Apply to the authoritative in-memory cooldown state synchronously so the
    # selector observes the full gate-resolved duration; the durable write is
    # best-effort and must not block that value.
    for key in plan.memory_keys:
        _state_manager.codex.set_cooldown_memory(key, plan.duration_seconds)
    for key in plan.durable_keys:
        asyncio.ensure_future(set_candidate_cooldown(key, plan.duration_seconds))
    return plan.applied_scope


async def _apply_anthropic_auto_agent_alias_cooldown(
    *,
    request: Request,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    is_basic_pilot_lane: bool = False,
) -> str:
    # The basic pilot lane is Codex-only; the Anthropic applicator accepts the
    # flag for call-site symmetry with the shared retry loop and ignores it.
    _ = is_basic_pilot_lane
    assert _set_anthropic_cooldown is not None
    return await _apply_auto_agent_alias_cooldown(
        request=request,
        candidate=candidate,
        lane_key=lane_key,
        selected_cooldown_key=selected_cooldown_key,
        cooldown_seconds=cooldown_seconds,
        error_class=error_class,
        set_candidate_cooldown=_set_anthropic_cooldown,
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
    )


# ---------------------------------------------------------------------------
# Compatibility entry point
# ---------------------------------------------------------------------------


async def _set_codex_auto_agent_candidate_cooldowns(
    *,
    request: Request,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
    is_basic_pilot_lane: bool = False,
) -> str:
    return await _apply_codex_auto_agent_alias_cooldown(
        request=request,
        candidate=candidate,
        lane_key=lane_key,
        selected_cooldown_key=selected_cooldown_key,
        cooldown_seconds=cooldown_seconds,
        error_class=error_class,
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
        is_basic_pilot_lane=is_basic_pilot_lane,
    )


# ---------------------------------------------------------------------------
# CFG-004: Lane identity resolution
# ---------------------------------------------------------------------------


def resolve_lane_identity_hash(
    *,
    candidate: dict[str, Any],
) -> str:
    """Compute a secret-safe identity hash from public candidate identity.

    Identity is derived from provider:model:route_family ONLY -- never from
    lane_key or credentials.  One identity maps to multiple credential-derived
    lane keys.  Reconstructible after process restart from the active candidate
    enumeration without knowing any lane_key.
    """
    provider = str(candidate.get("provider") or "")
    model = str(candidate.get("model") or "")
    route_family = str(candidate.get("route_family") or "")
    identity_input = f"{provider}:{model}:{route_family}"
    return hashlib.sha256(identity_input.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CFG-004: Cooldown publication transaction orchestrator
# ---------------------------------------------------------------------------


async def execute_cooldown_publication_transaction(  # noqa: PLR0915
    *,
    alias_family: str,
    candidate: dict[str, Any],
    plan: "CooldownPublicationPlan",
    publish_cooldown_memory_fn: Any,
    persist_cooldown_fn: Any,
) -> Optional[object]:
    """Execute the cooldown publication under canonical lock ordering.

    Canonical order:
      1. Normalize family
      2. Acquire family mutation lock
      3. Acquire all affected probe locks in sorted canonical-key order
      4. Mutate: memory publish + durable transaction + local index
      5. Release in reverse order

    Never enters the family lock while retaining a pre-acquired probe lock.

    When Redis is configured: executes the atomic durable transaction
    BEFORE local mutation.  If local commit fails, restores durable
    pre-images and local snapshots.

    When Redis is unconfigured: retains legacy memory-only behavior.
    Does NOT claim or index CFG-004 durable clear support.

    State phases: PREPARED -> DURABLE_COMMITTED -> LOCAL_COMMITTED.
    """
    from .durable import (
        get_aawm_alias_routing_dual_cache,
        publish_cooldown_transaction,
        rollback_cooldown_transaction,
    )
    from .state import alias_routing_state as _default_state, canonicalize_alias_family

    state_mgr = _state_manager if _state_manager is not None else _default_state

    index_family = canonicalize_alias_family(alias_family)

    all_keys = sorted(set(plan.memory_keys) | set(plan.durable_keys))
    if not all_keys:
        return None

    family_state = state_mgr.family(index_family)
    identity_hash = resolve_lane_identity_hash(candidate=candidate)

    # Step 1: Acquire family mutation lock (outer boundary).
    async with family_state.lock:
        # Step 2: Acquire all affected probe locks in sorted canonical order.
        unique_locks: list[asyncio.Lock] = []
        seen_ids: set[int] = set()
        for key in all_keys:
            lock = await state_mgr.candidate_probe_lock(
                alias_family=alias_family,
                cooldown_key=key,
            )
            if id(lock) not in seen_ids:
                seen_ids.add(id(lock))
                unique_locks.append(lock)

        acquired: list[asyncio.Lock] = []
        txn_result = None
        local_snapshots: dict[str, float] = {}
        try:
            for lock in unique_locks:
                await lock.acquire()
                acquired.append(lock)

            # Step 3: Mutation under complete lock set.
            #
            # Canonical order: snapshot -> preflight -> durable commit ->
            # memory publish -> local index.  Local snapshot is taken
            # BEFORE any mutation so rollback can restore exact values.

            # 3a. Snapshot local state BEFORE any mutation.
            for key in plan.memory_keys:
                local_snapshots[key] = (
                    family_state.cooldown_until_monotonic_by_key.get(key, 0.0)
                )

            # 3b. Durable transaction.
            #
            # Distinguish Redis UNCONFIGURED from CONFIGURED-BUT-UNHEALTHY:
            # - Unconfigured: fall back to legacy memory-only + best-effort
            #   persist path (no durable guarantees claimed).
            # - Configured-but-unhealthy: FAIL CLOSED before any local
            #   mutation or legacy persistence seam.  A configured Redis
            #   that is unreachable means the system cannot guarantee the
            #   durability contract; proceeding with local-only state would
            #   create a silent split-brain.
            from .durable import RollbackFailedError as _RollbackFailedError

            dual_cache = get_aawm_alias_routing_dual_cache()
            _has_strict_redis = False
            _redis_configured = False
            if dual_cache is not None:
                _rc = getattr(dual_cache, "redis_cache", None)
                if _rc is not None and callable(getattr(_rc, "init_async_client", None)):
                    _has_strict_redis = True
            else:
                # No dual cache returned.  Check whether Redis IS configured
                # but unhealthy (as opposed to simply unconfigured).
                try:
                    from litellm.proxy.aawm_alias_routing_redis import (
                        get_status as _redis_get_status,
                    )
                    _status = _redis_get_status()
                    if isinstance(_status, dict) and _status.get("configured") is True:
                        _redis_configured = True
                except Exception:
                    pass
                if _redis_configured and plan.durable_keys:
                    # Configured but unhealthy: fail closed BEFORE any local
                    # mutation or legacy persistence.
                    raise RuntimeError(
                        "AAWM alias routing durable publish: Redis is configured "
                        "but unhealthy; failing closed before local mutation"
                    )

            if _has_strict_redis and plan.durable_keys:
                # Preflight local index capacity (reject before mutation).
                if not state_mgr.lane_identity_index.preflight_capacity(
                    identity_hash=identity_hash,
                    lane_keys=list(plan.durable_keys),
                ):
                    from .durable import CapacityRejectedError
                    raise CapacityRejectedError(
                        phase="PREPARED",
                        family=index_family,
                        transaction_id_prefix="preflight",
                        identity_prefix=identity_hash[:12],
                        key_count=len(plan.durable_keys),
                        exception_classes=(),
                    )

                # Execute atomic durable transaction BEFORE local mutation.
                txn_result = await publish_cooldown_transaction(
                    alias_family=index_family,
                    identity_hash=identity_hash,
                    cooldown_keys=list(plan.durable_keys),
                    lane_members=list(plan.durable_keys),
                    ttl_seconds=plan.duration_seconds,
                )

                # 3c+3d. Memory publish + local commit under rollback
                # protection.  Any exception from either step restores
                # durable pre-images and local snapshots.
                from .state import RegisterBatchOutcome as _RBOutcome
                try:
                    # Memory publish (after durable commit succeeds).
                    if plan.memory_keys:
                        publish_cooldown_memory_fn(
                            keys=plan.memory_keys,
                            seconds=plan.duration_seconds,
                        )

                    # Local commit: update index under the same mutation lease.
                    outcome = state_mgr.lane_identity_index.register_batch(
                        identity_hash=identity_hash,
                        lane_keys=list(plan.durable_keys),
                    )
                    if outcome is _RBOutcome.CAPACITY_REJECTED:
                        raise RuntimeError(
                            "local index register_batch rejected (capacity)"
                        )
                    # IDEMPOTENT is safe: repeated publication of the same
                    # lane keys is a no-op, not a rejection.
                except Exception as local_exc:
                    # Memory publish or local commit failed: restore durable
                    # pre-images + local snapshots.
                    # NEVER suppress RollbackFailedError: if rollback itself
                    # fails, propagate the sanitized indeterminate-state error
                    # over the earlier local exception.
                    try:
                        await rollback_cooldown_transaction(
                            alias_family=index_family,
                            journal=txn_result.journal,
                        )
                    except _RollbackFailedError:
                        # Restore local snapshots before propagating.
                        for key in plan.memory_keys:
                            snap = local_snapshots.get(key, 0.0)
                            if snap > 0:
                                family_state.cooldown_until_monotonic_by_key[key] = snap
                            else:
                                family_state.cooldown_until_monotonic_by_key.pop(key, None)
                        raise  # propagate RollbackFailedError, NOT local_exc
                    for key in plan.memory_keys:
                        snap = local_snapshots.get(key, 0.0)
                        if snap > 0:
                            family_state.cooldown_until_monotonic_by_key[key] = snap
                        else:
                            family_state.cooldown_until_monotonic_by_key.pop(key, None)
                    raise local_exc

                # Local commit succeeded: advance phase to LOCAL_COMMITTED.
                # The journal is immutable evidence and is NOT modified.
                from .durable import PHASE_LOCAL_COMMITTED as _PHASE_LOCAL

                txn_result.phase = _PHASE_LOCAL

            else:
                # 3e. Memory publish (Redis unconfigured or no durable keys).
                if plan.memory_keys:
                    publish_cooldown_memory_fn(
                        keys=plan.memory_keys,
                        seconds=plan.duration_seconds,
                    )

                # Legacy persist (only when Redis is unconfigured, NOT when
                # configured-but-unhealthy -- that case already failed closed).
                if not _has_strict_redis and not _redis_configured and plan.durable_keys:
                    await persist_cooldown_fn(
                        keys=plan.durable_keys,
                        seconds=plan.duration_seconds,
                    )

        finally:
            # Step 4: Release in reverse order.
            for lock in reversed(acquired):
                lock.release()

    return txn_result


# ---------------------------------------------------------------------------
# God-module facade installation (Wave 5C)
# ---------------------------------------------------------------------------

_HOST_FUNCTION_NAMES = (
    "_resolve_auto_agent_cooldown_publication_plan",
    "_persist_codex_cooldown_durable",
    "_persist_anthropic_cooldown_durable",
    "_apply_auto_agent_alias_cooldown",
    "_apply_codex_auto_agent_alias_cooldown",
    "_apply_basic_pilot_gated_cooldown",
    "_apply_anthropic_auto_agent_alias_cooldown",
    "_set_codex_auto_agent_candidate_cooldowns",
    "resolve_lane_identity_hash",
    "execute_cooldown_publication_transaction",
)


def install(host_globals: dict) -> None:
    """Publish same-object god-module facades for the moved functions.

    Functions retain this module's globals. Host-owned dependencies remain
    late-bound through the callbacks configured by
    :func:`configure_cooldown_apply_runtime`.
    """
    global _host_globals_ref
    _host_globals_ref = host_globals
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        host_globals[_name] = _mod[_name]
