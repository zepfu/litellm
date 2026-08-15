"""Cooldown publication-plan resolution and application for alias routing.

Wave 5C extraction from ``llm_passthrough_endpoints.py`` with alias-scoped
Codex failure-evidence gating.

Owns:
- ``_resolve_auto_agent_cooldown_publication_plan`` (pure resolver)
- ``_persist_codex_cooldown_durable`` / ``_persist_anthropic_cooldown_durable``
- ``_apply_auto_agent_alias_cooldown`` (shared apply)
- ``_apply_codex_auto_agent_alias_cooldown`` / ``_apply_anthropic_auto_agent_alias_cooldown``
- ``_apply_codex_failure_evidence_cooldown``
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

# state.py owned (Codex failure-evidence gate + memory publication)
_codex_failure_evidence_gate: Optional[Any] = None
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
    codex_failure_evidence_gate: Any,
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
    global _codex_failure_evidence_gate
    _codex_failure_evidence_gate = codex_failure_evidence_gate
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
        _host_globals_ref["_codex_failure_evidence_gate"] = _mod[
            "_codex_failure_evidence_gate"
        ]
        _host_globals_ref["_state_manager"] = _mod["_state_manager"]


# ---------------------------------------------------------------------------
# Publication-plan resolver (R3-1)
# ---------------------------------------------------------------------------


def _is_managed_openai_usage_limit_candidate(
    candidate: dict[str, Any],
    error_class: Optional[str],
) -> bool:
    return bool(
        error_class == "usage_limit_reached"
        and candidate.get("provider") == "openai"
        and candidate.get("codex_oauth_account_hash")
        and candidate.get("codex_oauth_lane_key")
    )


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
    codex_failure_evidence_alias: Optional[str] = None,
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

    Codex configured aliases use the alias-scoped N-of-M failure-evidence
    decision recorded earlier by the loop to authorize shared publication and
    resolve its duration. The existing generic scope resolver still owns
    provider/lane targeting, including Kimi managed-account and Grok
    account-lane keys. Anthropic calls omit ``codex_failure_evidence_alias`` and
    retain the direct generic scope resolver.
    """
    assert _get_candidate_cooldown_scope is not None
    assert _get_kimi_managed_account_cooldown_key is not None
    assert _get_grok_account_quota_lane_cooldown_key is not None
    cooldown_scope = _get_candidate_cooldown_scope(
        error_class,
        candidate=candidate,
        kimi_failure_metadata=kimi_failure_metadata,
    )
    is_last_resort = bool(candidate.get("last_resort"))
    duration = max(0.0, float(cooldown_seconds))
    allow_ttl_shrink = _is_managed_openai_usage_limit_candidate(
        candidate,
        error_class,
    )
    if (
        codex_failure_evidence_alias is not None
        and cooldown_scope not in {"none", "request_local"}
    ):
        assert _codex_failure_evidence_gate is not None
        decision = _codex_failure_evidence_gate.current_decision(
            canonical_alias=codex_failure_evidence_alias,
            cooldown_key=selected_cooldown_key,
        )
        if not decision.should_cool:
            return CooldownPublicationPlan(
                applied_scope="none",
                duration_seconds=0.0,
                grok_account_quota_exhausted=grok_account_quota_exhausted,
                kimi_failure_metadata=kimi_failure_metadata,
            )
        duration = max(0.0, float(decision.duration_seconds))
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
        if is_last_resort:
            if grok_account_quota_exhausted:
                lane_cooldown_key = _get_grok_account_quota_lane_cooldown_key(
                    candidate,
                    lane_key,
                )
                lane_keys = tuple(
                    [lane_cooldown_key]
                    if lane_cooldown_key is not None
                    else []
                )
                return CooldownPublicationPlan(
                    memory_keys=lane_keys,
                    durable_keys=lane_keys,
                    duration_seconds=duration,
                    applied_scope="request_local",
                    request_local_action="request_local_cooldown",
                    grok_account_quota_exhausted=grok_account_quota_exhausted,
                    kimi_failure_metadata=kimi_failure_metadata,
                )

            # Last-resort, non-account-gated failures stay request-local only.
            return CooldownPublicationPlan(
                duration_seconds=duration,
                applied_scope="request_local",
                request_local_action="request_local_cooldown",
                grok_account_quota_exhausted=grok_account_quota_exhausted,
                kimi_failure_metadata=kimi_failure_metadata,
            )

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
            allow_ttl_shrink=allow_ttl_shrink,
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


async def _persist_codex_cooldown_durable(
    *,
    keys: Sequence[str],
    seconds: float,
    allow_ttl_shrink: bool = False,
) -> None:
    """Persist codex cooldown keys to durable Redis (post-release, R3-1)."""
    assert _write_durable_payload is not None
    ttl_seconds = max(0.0, float(seconds))
    if ttl_seconds <= 0:
        return
    for key in keys:
        write_kwargs = {
            "alias_family": "codex",
            "state_kind": "cooldown",
            "state_key": key,
            "payload": {"cooldown_key": key},
            "ttl_seconds": ttl_seconds,
        }
        if allow_ttl_shrink:
            write_kwargs["allow_ttl_shrink"] = True
        await _write_durable_payload(
            **write_kwargs,
        )


async def _persist_anthropic_cooldown_durable(
    *,
    keys: Sequence[str],
    seconds: float,
    allow_ttl_shrink: bool = False,
) -> None:
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
    canonical_alias: str,
    request: Request,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Apply the alias-scoped Codex failure-evidence decision."""
    assert _set_codex_cooldown is not None
    return await _apply_codex_failure_evidence_cooldown(
        canonical_alias=canonical_alias,
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


async def _apply_codex_failure_evidence_cooldown(
    *,
    canonical_alias: str,
    request: Request,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    set_candidate_cooldown: Callable[[str, float], Awaitable[object]],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Apply one configured Codex alias's current failure-evidence decision.

    Delegates to the pure publication-plan resolver
    (:func:`_resolve_auto_agent_cooldown_publication_plan`) so this applicator
    no longer owns a separate memory target or a fire-and-forget durable
    target: the resolver derives the gate-driven duration and provider/lane
    target keys, this function publishes the memory keys synchronously, and the
    durable writes complete before this compatibility entry point returns.
    When the alias-scoped evidence threshold is not met, no shared cooldown is
    applied.
    """
    assert _state_manager is not None
    plan = _resolve_auto_agent_cooldown_publication_plan(
        request=request,
        candidate=candidate,
        lane_key=lane_key,
        selected_cooldown_key=selected_cooldown_key,
        cooldown_seconds=cooldown_seconds,
        error_class=error_class,
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
        codex_failure_evidence_alias=canonical_alias,
    )
    if plan.request_local_action is not None:
        assert _get_request_local_cooldown_key is not None
        assert _set_request_local_cooldown is not None
        assert _exclude_request_local_candidate is not None
        request_local_key = _get_request_local_cooldown_key(
            candidate=candidate,
            lane_key=lane_key,
        )
        _set_request_local_cooldown(
            request,
            cooldown_key=request_local_key,
            cooldown_seconds=plan.duration_seconds,
        )
        _exclude_request_local_candidate(request, cooldown_key=request_local_key)
        for key in plan.memory_keys:
            _state_manager.codex.set_cooldown_memory(key, plan.duration_seconds)
        for key in plan.durable_keys:
            await set_candidate_cooldown(key, plan.duration_seconds)
        return plan.applied_scope
    if plan.applied_scope == "none" or not plan.memory_keys:
        return plan.applied_scope
    # Apply to the authoritative in-memory cooldown state synchronously so the
    # selector observes the full gate-resolved duration before durable I/O.
    for key in plan.memory_keys:
        _state_manager.codex.set_cooldown_memory(key, plan.duration_seconds)
    for key in plan.durable_keys:
        await set_candidate_cooldown(key, plan.duration_seconds)
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
) -> str:
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
    canonical_alias: str,
    request: Request,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    error_class: Optional[str],
    grok_account_quota_exhausted: bool = False,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> str:
    return await _apply_codex_auto_agent_alias_cooldown(
        canonical_alias=canonical_alias,
        request=request,
        candidate=candidate,
        lane_key=lane_key,
        selected_cooldown_key=selected_cooldown_key,
        cooldown_seconds=cooldown_seconds,
        error_class=error_class,
        grok_account_quota_exhausted=grok_account_quota_exhausted,
        kimi_failure_metadata=kimi_failure_metadata,
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
                transaction_kwargs = {
                    "alias_family": index_family,
                    "identity_hash": identity_hash,
                    "cooldown_keys": list(plan.durable_keys),
                    "lane_members": list(plan.durable_keys),
                    "ttl_seconds": plan.duration_seconds,
                }
                if plan.allow_ttl_shrink:
                    transaction_kwargs["allow_ttl_shrink"] = True
                txn_result = await publish_cooldown_transaction(
                    **transaction_kwargs,
                )

                # 3c+3d. Memory publish + local commit under rollback
                # protection.  Any exception from either step restores
                # durable pre-images and local snapshots.
                from .state import RegisterBatchOutcome as _RBOutcome
                try:
                    # Memory publish (after durable commit succeeds).
                    if plan.memory_keys:
                        memory_kwargs = {
                            "keys": plan.memory_keys,
                            "seconds": plan.duration_seconds,
                        }
                        if plan.allow_ttl_shrink:
                            memory_kwargs["allow_ttl_shrink"] = True
                        publish_cooldown_memory_fn(**memory_kwargs)

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
                    memory_kwargs = {
                        "keys": plan.memory_keys,
                        "seconds": plan.duration_seconds,
                    }
                    if plan.allow_ttl_shrink:
                        memory_kwargs["allow_ttl_shrink"] = True
                    publish_cooldown_memory_fn(**memory_kwargs)

                # Legacy persist (only when Redis is unconfigured, NOT when
                # configured-but-unhealthy -- that case already failed closed).
                if not _has_strict_redis and not _redis_configured and plan.durable_keys:
                    persist_kwargs = {
                        "keys": plan.durable_keys,
                        "seconds": plan.duration_seconds,
                    }
                    if plan.allow_ttl_shrink:
                        persist_kwargs["allow_ttl_shrink"] = True
                    await persist_cooldown_fn(**persist_kwargs)

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
    "_apply_codex_failure_evidence_cooldown",
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
