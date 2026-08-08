"""CFG-004 endpoint: POST /aawm/alias-routing/cooldowns/clear.

Implements the full cooldown-clear contract:
- Strict request schema (forbid extra fields, real strings, alias XOR exact)
- Ingress/family validation with anthropic_route_family projection
- Auth: user_api_key_auth + PROXY_ADMIN + explicit master-key check
- Topology gate: AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER=1
- Active-snapshot-only resolution (pure snapshot phase, no local/durable reads)
- Endpoint-owned per-family serialization lock
- Reservation-first sequencing: identity reservation BEFORE any lane/durable inspection
- Post-reservation hydration: local index + durable union
- Bounded publication drain
- Atomic compare-and-clear via durable Lua transaction
- Mandatory DualCache invalidation + strict postcondition verification
- Fail-closed on missing DualCache, Redis errors, or unverifiable state
- Idempotent not_active only after authoritative absence proof
- Structured audit events on EVERY exit (success, not_active, failure, conflict)
- Sanitized response/error/audit fields (no secrets, keys, hashes)
- No provider traffic (state manipulation only)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, Request

from .cooldown_apply import resolve_lane_identity_hash
from .durable import (
    ClearIndeterminateError,
    ClearTransactionResult,
    MembershipDriftError,
    RollbackDriftError,
    RollbackFailedError,
    RollbackReceiptMissingError,
    clear_cooldown_transaction,
    get_aawm_alias_routing_dual_cache,
    get_aawm_alias_routing_state_namespace,
    inspect_identity_set,
    verify_aawm_alias_routing_durable_absence,
)
from .snapshot_select import (
    _resolve_snapshot_alias_candidates,
    get_active_routing_snapshot,
)
from .state import (
    AliasRoutingStateManager,
    ClearReservation,
    alias_routing_state,
    inspect_cooldown_absence,
    validate_alias_family,
)

logger = logging.getLogger("LiteLLMProxy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOPOLOGY_ENV_VAR = "AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER"
_VALID_INGRESS_FAMILIES = frozenset({"codex", "anthropic"})
_ALLOWED_REQUEST_FIELDS = frozenset({"alias", "provider", "model", "ingress"})
_FORBIDDEN_FIELD_NAMES = frozenset({
    "key", "hash", "namespace", "pattern", "global",
    "raw_key", "state_key", "cooldown_key", "identity_hash",
})
_PUBLICATION_DRAIN_TIMEOUT_SECONDS = 10.0

# CFG-004 Defect 5: environment resolution follows the deployed AAWM
# convention (aawm_agent_identity constants).  AAWM_LITELLM_ENVIRONMENT is
# the primary source; LITELLM_ENVIRONMENT is the documented safe fallback
# for environments that predate the AAWM-specific variable.
_ENVIRONMENT_ENV_VARS = (
    "AAWM_LITELLM_ENVIRONMENT",
    "LITELLM_ENVIRONMENT",
)


def _resolve_environment() -> str:
    """Return the deployed environment label.

    Checks AAWM_LITELLM_ENVIRONMENT first (primary), then LITELLM_ENVIRONMENT
    (documented fallback).  Returns ``"unknown"`` only when neither is set.
    """
    for var in _ENVIRONMENT_ENV_VARS:
        val = os.getenv(var, "").strip()
        if val:
            return val
    return "unknown"

# ---------------------------------------------------------------------------
# Endpoint-owned per-family serialization locks (Acceptance #4)
# ---------------------------------------------------------------------------

_endpoint_family_locks: dict[str, asyncio.Lock] = {}
_endpoint_locks_guard = asyncio.Lock()


async def _get_endpoint_family_lock(family: str) -> asyncio.Lock:
    """Return (creating if needed) the endpoint-owned lock for *family*."""
    async with _endpoint_locks_guard:
        lock = _endpoint_family_locks.get(family)
        if lock is None:
            lock = asyncio.Lock()
            _endpoint_family_locks[family] = lock
        return lock


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CooldownClearRequest:
    """Validated clear request: alias XOR (provider, model)."""

    alias: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    ingress: str = ""


@dataclass
class _ResolvedIdentity:
    """One candidate identity resolved from the active snapshot + lane index."""

    identity_hash: str
    provider: str
    model: str
    route_family: str
    lane_keys: list[str] = field(default_factory=list)


@dataclass
class _ResolvedTarget:
    """Internal resolution result: all identities from active snapshot."""

    family: str
    canonical_aliases: tuple[str, ...] = ()
    identities: list[_ResolvedIdentity] = field(default_factory=list)
    target_description: str = ""
    ingress: str = ""

    @property
    def all_cooldown_keys(self) -> list[str]:
        keys: set[str] = set()
        for ident in self.identities:
            keys.update(ident.lane_keys)
        return sorted(keys)

    @property
    def all_identity_hashes(self) -> list[str]:
        return sorted({i.identity_hash for i in self.identities})

    @property
    def candidate_descriptions(self) -> list[dict[str, str]]:
        return [
            {
                "provider": i.provider,
                "model": i.model,
                "route_family": i.route_family,
            }
            for i in self.identities
        ]


@dataclass
class _PriorStateInspection:
    """Result of authoritative prior-state inspection (Acceptance #3/#6)."""

    has_active: bool
    source: str  # "memory" | "durable_cache" | "memory+durable_cache" | "none"
    bounded_remaining_ttl_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Validation (Acceptance #5: strict schema)
# ---------------------------------------------------------------------------


def _parse_and_validate_request(body: dict[str, Any]) -> CooldownClearRequest:
    """Parse request body with strict schema enforcement.

    Finding 5: tracks field presence explicitly.  Explicit null, empty, or
    whitespace-only ``alias``, ``provider``, or ``model`` is invalid whenever
    the field is supplied (present in the JSON object), even if the field
    would be unused under the alias-XOR-exact rule.  Omitted fields are
    allowed according to the alias XOR exact form.

    Forbids extra fields, requires real strings, enforces alias XOR exact
    provider+model, and validates ingress enum.  Raw key/hash/namespace/
    pattern/global fields are rejected by validation, not ignored.
    """
    # Reject any field outside the allowed set.
    extra = set(body.keys()) - _ALLOWED_REQUEST_FIELDS
    if extra:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unexpected_fields",
                "message": f"unexpected fields: {sorted(extra)}",
            },
        )

    # Reject forbidden field names even if they would be "extra".
    for field_name in body:
        if field_name.lower() in _FORBIDDEN_FIELD_NAMES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "forbidden_field",
                    "message": f"field '{field_name}' is not accepted",
                },
            )

    # Require real strings (not int, list, dict, bool).
    for field_name in ("alias", "provider", "model", "ingress"):
        val = body.get(field_name)
        if val is not None and not isinstance(val, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_field_type",
                    "message": f"field '{field_name}' must be a string",
                },
            )

    # Finding 5: explicit null/empty/whitespace supplied fields are invalid.
    for field_name in ("alias", "provider", "model"):
        if field_name in body:
            val = body[field_name]
            if val is None or (isinstance(val, str) and val.strip() == ""):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_field_value",
                        "message": f"field '{field_name}' must be a non-empty string when supplied",
                    },
                )

    alias = body.get("alias")
    provider = body.get("provider")
    model = body.get("model")
    ingress_raw = body.get("ingress")
    ingress = ingress_raw.strip().lower() if isinstance(ingress_raw, str) else ""

    # Ingress validation (explicit enum).
    if ingress not in _VALID_INGRESS_FAMILIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_ingress",
                "message": "ingress must be 'codex' or 'anthropic'",
            },
        )

    has_alias = isinstance(alias, str) and alias.strip() != ""
    has_provider = isinstance(provider, str) and provider.strip() != ""
    has_model = isinstance(model, str) and model.strip() != ""
    has_exact = has_provider or has_model

    if has_alias and has_exact:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ambiguous_target",
                "message": "specify either 'alias' OR ('provider' + 'model'), not both",
            },
        )
    if not has_alias and not has_exact:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_target",
                "message": "specify either 'alias' OR ('provider' + 'model')",
            },
        )
    if has_exact:
        if not has_provider or not has_model:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "incomplete_target",
                    "message": "both 'provider' and 'model' are required for exact targeting",
                },
            )

    return CooldownClearRequest(
        alias=alias.strip() if has_alias else None,
        provider=provider.strip() if has_provider else None,
        model=model.strip() if has_model else None,
        ingress=ingress,
    )


# ---------------------------------------------------------------------------
# Topology gate
# ---------------------------------------------------------------------------


def _check_topology_gate() -> None:
    """Fail closed unless AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER=1."""
    raw = os.getenv(_TOPOLOGY_ENV_VAR, "").strip()
    if raw != "1":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "topology_gate_closed",
                "message": (
                    "cooldown clear requires single-worker topology; "
                    f"set {_TOPOLOGY_ENV_VAR}=1 to enable"
                ),
            },
        )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_admin_auth(user_api_key_dict: Any) -> None:
    """Verify PROXY_ADMIN role and explicit master-key match.

    Fail closed: any missing attribute or mismatch raises 403.
    Never exposes secrets in error messages.
    """
    from litellm.proxy._types import LitellmUserRoles

    user_role = getattr(user_api_key_dict, "user_role", None)
    if user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "cooldown clear requires proxy_admin role",
            },
        )

    from litellm.proxy import proxy_server

    master_key = getattr(proxy_server, "master_key", None)
    if not master_key:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "auth_unavailable",
                "message": "cooldown clear unavailable",
            },
        )

    from litellm.proxy._types import UserAPIKeyAuth

    token_hash = getattr(user_api_key_dict, "token", None)
    if not token_hash:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "cooldown clear requires authenticated token",
            },
        )

    expected_hash = UserAPIKeyAuth._safe_hash_litellm_api_key(master_key)
    if token_hash != expected_hash:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "cooldown clear requires primary authentication",
            },
        )


# ---------------------------------------------------------------------------
# Audit events (Acceptance #6)
# ---------------------------------------------------------------------------


def _emit_audit_event(
    *,
    event_type: str,
    target: _ResolvedTarget,
    result: str,
    error_code: str = "",
    prior_state_source: str = "",
    bounded_remaining_ttl_seconds: float = 0.0,
) -> None:
    """Emit a structured audit/operator event via the standard proxy logger.

    Never includes raw keys, hashes, credentials, auth tokens, receipt
    bodies, or traceback locals.
    """
    try:
        namespace = get_aawm_alias_routing_state_namespace()
    except Exception:
        namespace = "unavailable"
    environment = _resolve_environment()
    audit_payload: dict[str, Any] = {
        "event": f"aawm_cooldown_clear_{event_type}",
        "target_description": target.target_description,
        "family": target.family,
        "ingress": target.ingress,
        "candidates": target.candidate_descriptions,
        "result": result,
        "prior_state_source": prior_state_source,
        "bounded_remaining_ttl_seconds": round(bounded_remaining_ttl_seconds, 2),
        "environment": environment,
        "namespace": namespace,
    }
    if error_code:
        audit_payload["error_code"] = error_code
    logger.info("aawm_cooldown_clear_audit %s", audit_payload)


# ---------------------------------------------------------------------------
# Snapshot resolution (Acceptance #1, #2)
# ---------------------------------------------------------------------------


def _project_route_family(candidate: Any, ingress: str) -> str:
    """Return the ingress-appropriate route_family for a RoutingCandidate.

    For anthropic ingress, uses ``anthropic_route_family`` (fail closed if
    absent).  For codex ingress, uses the primary ``route_family``.
    """
    if ingress == "anthropic":
        rf = getattr(candidate, "anthropic_route_family", None)
        if rf is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "anthropic_projection_unavailable",
                    "message": (
                        "candidate has no anthropic_route_family; "
                        "compile-time validation gap"
                    ),
                },
            )
        return rf
    return getattr(candidate, "route_family", None) or ""


def _resolve_target_from_active_snapshot(
    req: CooldownClearRequest,
) -> _ResolvedTarget:
    """Pure active-snapshot resolution: derive identity hashes ONLY.

    Uses ``resolve_lane_identity_hash`` to derive exact identity_hash values
    from the active routing snapshot.  Does NOT read lane_identity_index,
    durable state, or any local cooldown state.  Lane keys are populated
    later, after the identity-scoped reservation is established (Fix 1).

    For exact provider/model, rejects ambiguity across aliases, route
    families, or multiple distinct identities.
    """
    snapshot = get_active_routing_snapshot()
    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "no_active_snapshot",
                "message": "no active routing snapshot; cannot resolve clear target",
            },
        )

    family = validate_alias_family(req.ingress)

    # Expand aliases through the same concrete projection used by selection.
    # Keep the top-level canonical alias alongside every concrete candidate so
    # alias-scoped failure evidence remains targeted to the caller's alias.
    now_utc = datetime.now(timezone.utc)

    def _expand_alias(alias_name: str) -> list[tuple[str, dict[str, Any]]]:
        alias_obj = snapshot.aliases[alias_name]
        expanded = _resolve_snapshot_alias_candidates(
            alias_obj.name,
            ingress=family,
            client_product_label=None,
            now_utc=now_utc,
            snapshot=snapshot,
        )
        if family == "anthropic" and not expanded:
            # Preserve the existing fail-closed signal for a direct candidate
            # whose required Anthropic projection is absent. Alias references,
            # schedules, dispatch, and cycles remain owned by the expander.
            for entry in alias_obj.candidates:
                if hasattr(entry, "provider"):
                    _project_route_family(entry, req.ingress)
        return [(alias_obj.name, candidate) for candidate in expanded]

    matched_candidates: list[tuple[str, dict[str, Any]]] = []
    canonical_aliases: tuple[str, ...]

    if req.alias is not None:
        alias_obj = snapshot.aliases.get(req.alias)
        if alias_obj is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "alias_not_found",
                    "message": "alias not found in active snapshot",
                },
            )
        matched_candidates = _expand_alias(alias_obj.name)
        canonical_aliases = (alias_obj.name,)
        target_desc = f"alias:{alias_obj.name}"
    else:
        # Exact provider/model: scan every alias's concrete expansion.
        for alias_name in snapshot.aliases:
            for canonical_alias, cand in _expand_alias(alias_name):
                if (
                    cand.get("provider") == req.provider
                    and cand.get("model") == req.model
                ):
                    matched_candidates.append((canonical_alias, cand))
        if not matched_candidates:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "target_not_found",
                    "message": "provider/model not found in active snapshot",
                },
            )
        canonical_aliases = tuple(
            dict.fromkeys(
                canonical_alias for canonical_alias, _candidate in matched_candidates
            )
        )
        target_desc = f"exact:{req.provider}/{req.model}"

    # Candidates are already ingress-projected concrete dictionaries.
    projected = [
        (canonical_alias, candidate, candidate.get("route_family") or "")
        for canonical_alias, candidate in matched_candidates
    ]

    # For exact targeting, reject multiple distinct identities.
    if req.alias is None:
        distinct_identities: set[str] = set()
        for _canonical_alias, cand, rf in projected:
            cand_dict = {
                "provider": cand["provider"],
                "model": cand["model"],
                "route_family": rf,
            }
            distinct_identities.add(
                resolve_lane_identity_hash(candidate=cand_dict)
            )
        if len(distinct_identities) > 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ambiguous_target",
                    "message": (
                        "exact provider/model matches multiple distinct "
                        "identities across aliases or route families"
                    ),
                },
            )

    # Build identities with identity_hash only (no lane keys yet).
    # Lane keys are hydrated AFTER reservation creation (Fix 1).
    identities: list[_ResolvedIdentity] = []
    seen_hashes: set[str] = set()

    for _canonical_alias, cand, rf in projected:
        cand_dict = {
            "provider": cand["provider"],
            "model": cand["model"],
            "route_family": rf,
        }
        id_hash = resolve_lane_identity_hash(candidate=cand_dict)
        if id_hash in seen_hashes:
            continue
        seen_hashes.add(id_hash)

        identities.append(
            _ResolvedIdentity(
                identity_hash=id_hash,
                provider=cand["provider"],
                model=cand["model"],
                route_family=rf,
                lane_keys=[],
            )
        )

    return _ResolvedTarget(
        family=family,
        canonical_aliases=canonical_aliases,
        identities=identities,
        target_description=target_desc,
        ingress=req.ingress,
    )


def _hydrate_from_local_index(
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> None:
    """Populate lane_keys from the local lane_identity_index.

    Called ONLY after the identity-scoped reservation is established,
    ensuring publication claims by identity cannot become leader during
    this first local inspection (Fix 1).

    Mutates ``target.identities`` in place.
    """
    for ident in target.identities:
        local_keys = sorted(
            state_mgr.lane_identity_index.lanes_for(ident.identity_hash)
        )
        if local_keys:
            merged = set(ident.lane_keys) | set(local_keys)
            ident.lane_keys = sorted(merged)


async def _hydrate_identities_from_durable(
    target: _ResolvedTarget,
) -> None:
    """Authoritatively inspect and union durable membership for every identity.

    Finding 1 fix: ALWAYS inspects the durable identity set for each resolved
    identity, even when the local lane_identity_index already has a subset.
    The durable set is the authoritative membership source; the local index
    may be stale or partial after a restart.  The final lane_keys for each
    identity is the sorted union of local and durable members.

    Cross-check: ``inspect_identity_set`` is scoped by ``identity_hash``, so
    unrelated identities can never be cleared.  Fail closed on any inspection
    error.  Mutates ``target.identities`` in place.
    """
    family = target.family
    # Fail closed if DualCache is unavailable before attempting inspection.
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "cache_unavailable",
                "message": (
                    "DualCache unavailable; cannot hydrate lane keys "
                    "from durable identity set; failing closed"
                ),
            },
        )
    for ident in target.identities:
        try:
            inspection = await inspect_identity_set(
                alias_family=family,
                identity_hash=ident.identity_hash,
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "inspection_failed",
                    "message": (
                        "cannot inspect durable identity set for lane hydration; "
                        "failing closed"
                    ),
                },
            )
        if inspection.exists and inspection.cardinality > 0:
            # Union local + durable for the full exact lane set.
            merged = set(ident.lane_keys) | set(inspection.members)
            ident.lane_keys = sorted(merged)


# ---------------------------------------------------------------------------
# Prior-state inspection (Acceptance #3: fail closed)
# ---------------------------------------------------------------------------


async def _inspect_prior_state(
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> _PriorStateInspection:
    """Authoritative prior-state inspection: local + durable.

    ``not_active`` is allowed ONLY after both local and durable inspection
    succeed and prove absence.  Missing DualCache, Redis errors, or
    unverifiable state fail closed with 503.
    """
    family = target.family
    all_keys = target.all_cooldown_keys

    # 1. Local memory inspection.
    local_active = False
    max_remaining = 0.0
    for key in all_keys:
        inspection = inspect_cooldown_absence(
            state_mgr,
            alias_family=family,
            canonical_aliases=target.canonical_aliases,
            cooldown_key=key,
        )
        if inspection.exists:
            local_active = True
            max_remaining = max(max_remaining, inspection.remaining_seconds)

    # 2. Durable inspection -- fail closed on any error.
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "cache_unavailable",
                "message": (
                    "DualCache unavailable; cannot verify durable state; "
                    "failing closed"
                ),
            },
        )

    durable_active = False
    for ident in target.identities:
        try:
            inspection_result = await inspect_identity_set(
                alias_family=family,
                identity_hash=ident.identity_hash,
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "inspection_failed",
                    "message": "cannot inspect durable identity set; failing closed",
                },
            )
        if inspection_result.exists and inspection_result.cardinality > 0:
            durable_active = True
            ttl = inspection_result.ttl_remaining_seconds
            if isinstance(ttl, (int, float)):
                max_remaining = max(max_remaining, float(ttl))

    # Finding 2: include OpenRouter local state in prior-state semantics
    # (inspected under the shared lock for authoritative atomicity).
    openrouter_active = await _inspect_openrouter_prior_state(target, state_mgr)

    if local_active and durable_active:
        source = "memory+durable_cache"
    elif local_active:
        source = "memory"
    elif durable_active:
        source = "durable_cache"
    else:
        source = "none"

    has_active = local_active or durable_active or openrouter_active

    return _PriorStateInspection(
        has_active=has_active,
        source=source,
        bounded_remaining_ttl_seconds=max_remaining,
    )


# ---------------------------------------------------------------------------
# not_active per-key absence proof (Finding 3)
# ---------------------------------------------------------------------------


async def _verify_not_active_absence(
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> None:
    """Prove per-key durable+local absence before returning not_active.

    Finding 3: calls the strict durable/DualCache absence verifier for every
    authoritative lane key.  If no lane keys exist, requires authoritative
    identity-set absence plus local derived-state absence.  Any verifier or
    cache uncertainty fails closed (raises HTTPException 503).
    """
    family = target.family
    all_keys = target.all_cooldown_keys

    if all_keys:
        # Per-key strict durable absence verification.
        for key in all_keys:
            try:
                absent = await verify_aawm_alias_routing_durable_absence(
                    alias_family=family,
                    state_kind="cooldown",
                    state_key=key,
                )
            except Exception:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "absence_verification_failed",
                        "message": (
                            "cannot verify durable absence for not_active; "
                            "failing closed"
                        ),
                    },
                )
            if not absent:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "absence_verification_failed",
                        "message": (
                            "durable key still present despite identity-set "
                            "absence; failing closed"
                        ),
                    },
                )
    else:
        # No lane keys: require identity-set absence (already proven by
        # _inspect_prior_state returning has_active=False) plus local
        # derived-state absence for every identity.
        for ident in target.identities:
            inspection = inspect_cooldown_absence(
                state_mgr,
                alias_family=family,
                canonical_aliases=target.canonical_aliases,
                cooldown_key=ident.identity_hash,
            )
            if inspection.exists:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "absence_verification_failed",
                        "message": (
                            "local derived state present despite no lane keys; "
                            "failing closed"
                        ),
                    },
                )


# ---------------------------------------------------------------------------
# Clear execution (Acceptance #4: reservation sequencing)
# ---------------------------------------------------------------------------


async def _drain_publication_intents(
    registry: Any,
    family: str,
    all_keys: list[str],
    identity_hashes: frozenset[str] = frozenset(),
) -> None:
    """Wait for all active publication intents covering target (bounded).

    Finding 1: drains by BOTH discovered cooldown_keys AND identity hashes.
    An already-leading publication that has not yet registered cooldown_keys
    in the key-based index is discovered via ``scan_active_intents_by_identity``.
    The scan/await loop repeats until no matching active intents remain or a
    bounded timeout is reached.  New claims remain BLOCKED_BY_CLEAR because
    the clear reservation is already installed before this drain runs.

    Called while the clear reservation is active (blocking NEW publications)
    so only already-in-flight intents need draining.
    """
    deadline = asyncio.get_event_loop().time() + _PUBLICATION_DRAIN_TIMEOUT_SECONDS

    while True:
        # Collect intents to await: by key AND by identity.
        intents_to_await: list[Any] = []
        seen_intent_ids: set[int] = set()

        # Key-based lookup (existing path).
        for key in all_keys:
            intent = registry.get(family, key)
            if intent is not None and not intent.done.is_set():
                if id(intent) not in seen_intent_ids:
                    seen_intent_ids.add(id(intent))
                    intents_to_await.append(intent)

        # Identity-based scan (Finding 1: catches unindexed leaders).
        if identity_hashes:
            scanned = registry.scan_active_intents_by_identity(
                family, identity_hashes
            )
            for intent in scanned:
                if id(intent) not in seen_intent_ids:
                    seen_intent_ids.add(id(intent))
                    intents_to_await.append(intent)

        if not intents_to_await:
            return  # All clear.

        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "publication_drain_timeout",
                    "message": (
                        "timed out waiting for active publication; "
                        "retry later"
                    ),
                },
            )

        # Await all discovered intents concurrently (bounded).
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(intent.done.wait() for intent in intents_to_await),
                    return_exceptions=True,
                ),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "publication_drain_timeout",
                    "message": (
                        "timed out waiting for active publication; "
                        "retry later"
                    ),
                },
            )
        # Loop back to rescan: new intents may have appeared.


# ---------------------------------------------------------------------------
# OpenRouter targeted local-state clear (Finding 2)
# ---------------------------------------------------------------------------


def _derive_openrouter_rate_limit_keys(
    target: _ResolvedTarget,
) -> list[str]:
    """Derive exact OpenRouter adapter + upstream model rate-limit keys.

    Finding 2: for active OpenRouter candidates, derive exactly the same
    adapter and upstream model rate-limit keys as retry_transport.  Uses
    the same ``clean_secret_string`` normalization (strip + quote removal)
    and the same ``get_completion_model`` upstream resolution.

    Returns an empty list when no candidate is an OpenRouter provider.
    Never uses identity_hash as a cooldown key.
    """
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.model_resolution import (
        _get_openrouter_completion_adapter_upstream_model,
        _normalize_anthropic_openrouter_adapter_model_name,
    )

    keys: set[str] = set()
    for ident in target.identities:
        if ident.provider != "openrouter":
            continue
        # Adapter model key (same as retry_transport get_rate_limit_key).
        adapter_model = _normalize_anthropic_openrouter_adapter_model_name(
            ident.model
        )
        if adapter_model:
            cleaned = adapter_model.strip()
            if cleaned:
                keys.add(cleaned)
        # Upstream model key (same as retry_transport get_active_cooldown_seconds).
        upstream_model = _get_openrouter_completion_adapter_upstream_model(
            ident.model
        )
        if upstream_model:
            cleaned_up = upstream_model.strip()
            if cleaned_up:
                keys.add(cleaned_up)
    return sorted(keys)


async def _clear_openrouter_local_state(
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> dict[str, Any]:
    """Clear targeted OpenRouter rate-limit and failure-circuit entries.

    Finding 2: clears ONLY matching entries derived from target candidates.
    Never performs a global flush.  Preserves unrelated keys.  Operates
    under the openrouter_rate_limit lock for atomicity.

    Returns a result dict with cleared/preserved counts for audit.
    """
    or_keys = _derive_openrouter_rate_limit_keys(target)
    if not or_keys:
        return {"openrouter_keys_cleared": 0, "openrouter_keys_preserved": 0}

    rate_cleared = 0
    circuit_cleared = 0

    async with state_mgr.openrouter_rate_limit.lock:
        for key in or_keys:
            if state_mgr.openrouter_rate_limit.until_monotonic_by_key.pop(key, None) is not None:
                rate_cleared += 1
            if state_mgr.openrouter_failure_circuit.until_monotonic_by_key.pop(key, None) is not None:
                circuit_cleared += 1

        # Post-removal absence verification under the same lock: no mutation
        # observable within this critical section may survive a cleared result.
        now = time.monotonic()
        for key in or_keys:
            rl_until = state_mgr.openrouter_rate_limit.until_monotonic_by_key.get(key, 0.0)
            if rl_until > now:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "postcondition_failure",
                        "message": "openrouter rate-limit entry survived targeted removal",
                    },
                )
            fc_until = state_mgr.openrouter_failure_circuit.until_monotonic_by_key.get(key, 0.0)
            if fc_until > now:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "postcondition_failure",
                        "message": "openrouter failure-circuit entry survived targeted removal",
                    },
                )

    total_preserved = (
        len(state_mgr.openrouter_rate_limit.until_monotonic_by_key)
        + len(state_mgr.openrouter_failure_circuit.until_monotonic_by_key)
    )

    return {
        "openrouter_keys_cleared": rate_cleared + circuit_cleared,
        "openrouter_keys_preserved": total_preserved,
    }


async def _inspect_openrouter_prior_state(
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> bool:
    """Check whether any OpenRouter local state is active for target keys.

    Finding 2: included in prior-state semantics.  Returns True if any
    matching rate-limit or failure-circuit entry is currently active.

    Authoritative atomicity: inspection of both maps is performed while
    holding the shared openrouter_rate_limit.lock so the result is
    linearizable with concurrent clear/write operations.
    """
    or_keys = _derive_openrouter_rate_limit_keys(target)
    if not or_keys:
        return False
    async with state_mgr.openrouter_rate_limit.lock:
        now = time.monotonic()
        for key in or_keys:
            rl_until = state_mgr.openrouter_rate_limit.until_monotonic_by_key.get(key, 0.0)
            if rl_until > now:
                return True
            fc_until = state_mgr.openrouter_failure_circuit.until_monotonic_by_key.get(key, 0.0)
            if fc_until > now:
                return True
        return False


# ---------------------------------------------------------------------------
# Clear execution (Finding 3: all-or-none multi-identity + rollback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LocalClearPreimage:
    """Immutable snapshot of targeted process-local state before mutation.

    Captured while the clear lock regime (per-key barriers, family mutation
    lock, probe locks) is held and after all durable identities have
    committed, immediately before any local mutation.  Used to restore exact
    prior state if local mutation or postcondition verification fails after
    durable commits.  Holds only the same opaque lane/identity identifiers
    the endpoint already uses; never credentials or secrets.  Session
    affinity and unrelated keys are intentionally never captured.
    """

    family: str
    targeted_keys: tuple[str, ...] = ()
    positive_cooldown: dict[str, float] = field(default_factory=dict)
    negative_cooldown: dict[str, float] = field(default_factory=dict)
    evidence_events: dict[str, tuple[float, ...]] = field(default_factory=dict)
    generation: dict[str, int] = field(default_factory=dict)
    codex_failure_evidence_aliases: tuple[str, ...] = ()
    codex_failure_evidence_key_state: dict[tuple[str, str], Any] = field(
        default_factory=dict
    )
    codex_failure_evidence_events: dict[
        tuple[str, str], tuple[float, ...]
    ] = field(default_factory=dict)
    lane_index_membership: dict[str, frozenset[str]] = field(default_factory=dict)
    openrouter_keys: tuple[str, ...] = ()
    openrouter_rate_limit: dict[str, float] = field(default_factory=dict)
    openrouter_failure_circuit: dict[str, float] = field(default_factory=dict)


async def _capture_local_preimage(
    *,
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> _LocalClearPreimage:
    """Capture exact preimages of every targeted local map before mutation.

    Read-only.  Called under the clear lock regime after all durable
    identities commit and before any local mutation.  Covers exactly the
    state the clear path mutates: family positive/negative cooldown maps,
    evidence events, per-key generation counters, alias-scoped Codex failure
    evidence, lane-identity index membership, and targeted OpenRouter
    rate-limit/failure-circuit entries. Session
    affinity and unrelated keys are intentionally excluded so they are
    preserved exactly.
    """
    family = target.family
    all_keys = target.all_cooldown_keys
    family_state = state_mgr.family(family)

    positive = {
        k: family_state.cooldown_until_monotonic_by_key[k]
        for k in all_keys
        if k in family_state.cooldown_until_monotonic_by_key
    }
    negative = {
        k: family_state.cooldown_negative_until_monotonic_by_key[k]
        for k in all_keys
        if k in family_state.cooldown_negative_until_monotonic_by_key
    }
    evidence = {
        k: tuple(family_state.evidence_events_by_key[k])
        for k in all_keys
        if k in family_state.evidence_events_by_key
    }
    generation = {
        k: family_state.cooldown_generation_by_key[k]
        for k in all_keys
        if k in family_state.cooldown_generation_by_key
    }

    codex_failure_evidence_aliases: tuple[str, ...] = ()
    codex_failure_evidence_key_state: dict[tuple[str, str], Any] = {}
    codex_failure_evidence_events: dict[
        tuple[str, str], tuple[float, ...]
    ] = {}
    if validate_alias_family(family) == "codex":
        codex_failure_evidence_aliases = target.canonical_aliases
        for canonical_alias in codex_failure_evidence_aliases:
            gate = state_mgr.codex_failure_evidence_gate.gate_for_alias(
                canonical_alias=canonical_alias
            )
            if gate is None:
                continue
            for k in all_keys:
                entry = (canonical_alias, k)
                ks = gate._key_state.get(k)
                if ks is not None:
                    # _KeyCooldownState is a dataclass of immutable scalars; a
                    # shallow replace() yields an independent snapshot.
                    codex_failure_evidence_key_state[entry] = replace(ks)
                ev = gate._family_state.evidence_events_by_key.get(k)
                if ev is not None:
                    codex_failure_evidence_events[entry] = tuple(ev)

    # Lane-identity index membership per targeted identity.
    lane_membership: dict[str, frozenset[str]] = {}
    for ident in target.identities:
        if ident.lane_keys:
            lane_membership[ident.identity_hash] = (
                state_mgr.lane_identity_index.lanes_for(ident.identity_hash)
            )

    # Targeted OpenRouter entries captured under their own lock so the
    # snapshot is linearizable with concurrent OpenRouter writers.
    or_keys = _derive_openrouter_rate_limit_keys(target)
    or_rate: dict[str, float] = {}
    or_circuit: dict[str, float] = {}
    if or_keys:
        async with state_mgr.openrouter_rate_limit.lock:
            for k in or_keys:
                if k in state_mgr.openrouter_rate_limit.until_monotonic_by_key:
                    or_rate[k] = state_mgr.openrouter_rate_limit.until_monotonic_by_key[k]
                if k in state_mgr.openrouter_failure_circuit.until_monotonic_by_key:
                    or_circuit[k] = state_mgr.openrouter_failure_circuit.until_monotonic_by_key[k]

    return _LocalClearPreimage(
        family=family,
        targeted_keys=tuple(all_keys),
        positive_cooldown=positive,
        negative_cooldown=negative,
        evidence_events=evidence,
        generation=generation,
        codex_failure_evidence_aliases=codex_failure_evidence_aliases,
        codex_failure_evidence_key_state=codex_failure_evidence_key_state,
        codex_failure_evidence_events=codex_failure_evidence_events,
        lane_index_membership=lane_membership,
        openrouter_keys=tuple(or_keys),
        openrouter_rate_limit=or_rate,
        openrouter_failure_circuit=or_circuit,
    )


async def _restore_local_preimage(  # noqa: PLR0915
    *,
    preimage: _LocalClearPreimage,
    state_mgr: AliasRoutingStateManager,
) -> bool:
    """Restore captured local preimages after a failed clear (rollback).

    Called under the same clear lock regime after durable receipts have been
    rolled back.  Restores family positive/negative cooldown maps, evidence
    events, per-key generation counters, alias-scoped Codex failure evidence,
    lane-identity index membership, and (under its own lock) targeted
    OpenRouter entries to their exact preimage values.  Returns True only if
    every targeted map verifies equal to its preimage afterwards; returns
    False (never raises) if any restoration cannot be proven so the caller
    fails closed.  Never exposes keys, hashes, or traceback locals.
    """
    try:
        family_state = state_mgr.family(preimage.family)
        targeted = preimage.targeted_keys

        # Family cooldown-derived maps: set exact preimage value or remove.
        for k in targeted:
            if k in preimage.positive_cooldown:
                family_state.cooldown_until_monotonic_by_key[k] = preimage.positive_cooldown[k]
            else:
                family_state.cooldown_until_monotonic_by_key.pop(k, None)
            if k in preimage.negative_cooldown:
                family_state.cooldown_negative_until_monotonic_by_key[k] = preimage.negative_cooldown[k]
            else:
                family_state.cooldown_negative_until_monotonic_by_key.pop(k, None)
            if k in preimage.evidence_events:
                family_state.evidence_events_by_key[k] = list(preimage.evidence_events[k])
            else:
                family_state.evidence_events_by_key.pop(k, None)
            if k in preimage.generation:
                family_state.cooldown_generation_by_key[k] = preimage.generation[k]
            else:
                family_state.cooldown_generation_by_key.pop(k, None)

        # Alias-scoped Codex failure evidence.
        for canonical_alias in preimage.codex_failure_evidence_aliases:
            has_preimage = any(
                (canonical_alias, k)
                in preimage.codex_failure_evidence_key_state
                or (canonical_alias, k)
                in preimage.codex_failure_evidence_events
                for k in targeted
            )
            gate = state_mgr.codex_failure_evidence_gate.gate_for_alias(
                canonical_alias=canonical_alias,
                create=has_preimage,
            )
            if gate is None:
                continue
            for k in targeted:
                entry = (canonical_alias, k)
                if entry in preimage.codex_failure_evidence_key_state:
                    gate._key_state[k] = replace(
                        preimage.codex_failure_evidence_key_state[entry]
                    )
                else:
                    gate._key_state.pop(k, None)
                if entry in preimage.codex_failure_evidence_events:
                    gate._family_state.evidence_events_by_key[k] = list(
                        preimage.codex_failure_evidence_events[entry]
                    )
                else:
                    gate._family_state.evidence_events_by_key.pop(k, None)
            state_mgr.codex_failure_evidence_gate.drop_alias_if_empty(
                canonical_alias=canonical_alias
            )

        # Lane-identity index membership per identity.
        for identity_hash, lanes in preimage.lane_index_membership.items():
            state_mgr.lane_identity_index.restore_membership(
                identity_hash=identity_hash,
                lane_keys=lanes,
            )

        # Targeted OpenRouter entries restored + verified under their lock.
        if preimage.openrouter_keys:
            async with state_mgr.openrouter_rate_limit.lock:
                for k in preimage.openrouter_keys:
                    if k in preimage.openrouter_rate_limit:
                        state_mgr.openrouter_rate_limit.until_monotonic_by_key[k] = preimage.openrouter_rate_limit[k]
                    else:
                        state_mgr.openrouter_rate_limit.until_monotonic_by_key.pop(k, None)
                    if k in preimage.openrouter_failure_circuit:
                        state_mgr.openrouter_failure_circuit.until_monotonic_by_key[k] = preimage.openrouter_failure_circuit[k]
                    else:
                        state_mgr.openrouter_failure_circuit.until_monotonic_by_key.pop(k, None)
                rl_slice = {
                    k: state_mgr.openrouter_rate_limit.until_monotonic_by_key[k]
                    for k in preimage.openrouter_keys
                    if k in state_mgr.openrouter_rate_limit.until_monotonic_by_key
                }
                fc_slice = {
                    k: state_mgr.openrouter_failure_circuit.until_monotonic_by_key[k]
                    for k in preimage.openrouter_keys
                    if k in state_mgr.openrouter_failure_circuit.until_monotonic_by_key
                }
                if rl_slice != preimage.openrouter_rate_limit:
                    return False
                if fc_slice != preimage.openrouter_failure_circuit:
                    return False

        # Prove exact restoration of the family maps (targeted slice only).
        pos_slice = {
            k: family_state.cooldown_until_monotonic_by_key[k]
            for k in targeted
            if k in family_state.cooldown_until_monotonic_by_key
        }
        if pos_slice != preimage.positive_cooldown:
            return False
        neg_slice = {
            k: family_state.cooldown_negative_until_monotonic_by_key[k]
            for k in targeted
            if k in family_state.cooldown_negative_until_monotonic_by_key
        }
        if neg_slice != preimage.negative_cooldown:
            return False
        ev_slice = {
            k: tuple(family_state.evidence_events_by_key[k])
            for k in targeted
            if k in family_state.evidence_events_by_key
        }
        if ev_slice != preimage.evidence_events:
            return False
        gen_slice = {
            k: family_state.cooldown_generation_by_key[k]
            for k in targeted
            if k in family_state.cooldown_generation_by_key
        }
        if gen_slice != preimage.generation:
            return False

        # Prove lane-identity index membership restoration.
        for identity_hash, lanes in preimage.lane_index_membership.items():
            if state_mgr.lane_identity_index.lanes_for(identity_hash) != lanes:
                return False

        # Prove alias-scoped Codex failure-evidence restoration.
        present_key_state: set[tuple[str, str]] = set()
        present_events: set[tuple[str, str]] = set()
        for canonical_alias in preimage.codex_failure_evidence_aliases:
            gate = state_mgr.codex_failure_evidence_gate.gate_for_alias(
                canonical_alias=canonical_alias
            )
            if gate is None:
                continue
            present_key_state.update(
                (canonical_alias, k) for k in targeted if k in gate._key_state
            )
            present_events.update(
                (canonical_alias, k)
                for k in targeted
                if k in gate._family_state.evidence_events_by_key
            )
        if present_key_state != set(
            preimage.codex_failure_evidence_key_state.keys()
        ):
            return False
        if present_events != set(preimage.codex_failure_evidence_events.keys()):
            return False

        return True
    except Exception:
        # Restoration outcome cannot be proven; caller fails closed.
        return False


async def _execute_clear(  # noqa: PLR0915
    target: _ResolvedTarget,
    state_mgr: AliasRoutingStateManager,
) -> tuple[int, int]:
    """Execute the clear mutation under the canonical lock order.

    Finding 3: all-or-none multi-identity clear.  Collects every successful
    ClearTransactionResult.  On later durable failure, rolls back prior
    receipts in reverse order via rollback_clear_transaction.  Delays ALL
    local/index/OpenRouter mutation until every durable identity commits.
    If rollback drift/missing/failure occurs, returns sanitized
    rollback_failure/indeterminate and preserves evidence.  If local/
    postcondition failure occurs after durable commits, attempts reverse
    rollback before fail closed.  Never claims success from partial target.

    Canonical lock order (Defect 1 fix -- deadlock-free):
      1. Per-key barrier locks (sorted)  -- outermost
      2. Family mutation lock
      3. Probe locks (sorted)            -- innermost

    Returns (keys_cleared, members_removed).
    """
    family = target.family
    all_keys = target.all_cooldown_keys

    # Defect 2 fix: do NOT early-return solely because lane key set is empty
    # when candidate-derived OpenRouter adapter/upstream blocker keys are
    # active.  Clear and verify the exact OpenRouter rate-limit/failure-circuit
    # entries under their lock, preserve unrelated entries, and return cleared
    # with accurate prior/result semantics.
    if not all_keys:
        or_keys = _derive_openrouter_rate_limit_keys(target)
        if not or_keys:
            # Neither lane nor OpenRouter state: caller handles not_active.
            return 0, 0
        # OpenRouter-only clear (no durable lane state).
        or_result = await _clear_openrouter_local_state(target, state_mgr)
        or_cleared = or_result.get("openrouter_keys_cleared", 0)
        if or_cleared > 0:
            return or_cleared, 0
        # OpenRouter keys derived but none active: no-op.
        return 0, 0

    family_state = state_mgr.family(family)
    total_keys_deleted = 0
    total_members_removed = 0

    # Step 1: Acquire per-key barrier locks in sorted order (outermost).
    barrier_locks: list[asyncio.Lock] = []
    seen_barrier_ids: set[int] = set()
    for key in all_keys:
        block = await state_mgr.key_barrier_lock(key)
        if id(block) not in seen_barrier_ids:
            seen_barrier_ids.add(id(block))
            barrier_locks.append(block)

    acquired_barriers: list[asyncio.Lock] = []
    try:
        for block in barrier_locks:
            await block.acquire()
            acquired_barriers.append(block)

        # Step 2: Acquire family mutation lock.
        async with family_state.lock:
            # Step 3: Acquire probe locks in sorted order (innermost).
            probe_locks: list[asyncio.Lock] = []
            seen_probe_ids: set[int] = set()
            for key in all_keys:
                lock = await state_mgr.candidate_probe_lock(
                    alias_family=family,
                    cooldown_key=key,
                )
                if id(lock) not in seen_probe_ids:
                    seen_probe_ids.add(id(lock))
                    probe_locks.append(lock)

            acquired_probes: list[asyncio.Lock] = []
            try:
                for lock in probe_locks:
                    await lock.acquire()
                    acquired_probes.append(lock)

                # Step 4: Execute durable clear per identity (Finding 3:
                # all-or-none).  Collect committed results; on failure,
                # rollback all prior in reverse order.
                committed_results: list[ClearTransactionResult] = []
                try:
                    for ident in target.identities:
                        if not ident.lane_keys:
                            continue
                        txn_result = await _execute_durable_clear(
                            family=family,
                            identity_hash=ident.identity_hash,
                            cooldown_keys=ident.lane_keys,
                            lane_members=ident.lane_keys,
                        )
                        if txn_result is not None:
                            committed_results.append(txn_result)
                            total_keys_deleted += txn_result.keys_deleted
                            total_members_removed += txn_result.members_removed
                except HTTPException:
                    # Finding 3: durable failure after prior commits.
                    # Rollback all prior receipts in reverse order.
                    await _rollback_committed_results(
                        family=family,
                        committed_results=committed_results,
                    )
                    raise

                # Step 5: ALL durable identities committed.  Now apply
                # local mutations (Finding 3: delay until all commit).
                # Capture exact local preimages BEFORE any mutation so a
                # later local/postcondition failure can restore process-local
                # state (not just durable receipts) under the same lock regime.
                local_preimage = await _capture_local_preimage(
                    target=target,
                    state_mgr=state_mgr,
                )
                try:
                    # 5a. Clear process-local cooldown-derived state.
                    state_mgr.clear_cooldown_state(
                        alias_family=family,
                        canonical_aliases=target.canonical_aliases,
                        cooldown_keys=all_keys,
                    )

                    # 5b. Clear local index entries per identity.
                    for ident in target.identities:
                        if ident.lane_keys:
                            state_mgr.lane_identity_index.unregister_batch(
                                identity_hash=ident.identity_hash,
                                lane_keys=ident.lane_keys,
                            )

                    # 5c. Finding 2: Clear targeted OpenRouter local state.
                    await _clear_openrouter_local_state(target, state_mgr)

                    # Step 6: Strict postcondition verification.
                    await _verify_postconditions(
                        family=family,
                        canonical_aliases=target.canonical_aliases,
                        identities=target.identities,
                        all_keys=all_keys,
                        state_mgr=state_mgr,
                    )
                except HTTPException as local_exc:
                    # Finding 3: local/postcondition failure after durable
                    # commits.  Attempt reverse rollback before fail closed.
                    # Local restoration must ALWAYS be attempted, even when
                    # durable rollback raises a sanitized rollback_failure/
                    # indeterminate error, so the worker is never left locally
                    # uncooled while only durable receipts are rolled back.
                    durable_rollback_exc: Optional[HTTPException] = None
                    try:
                        await _rollback_committed_results(
                            family=family,
                            committed_results=committed_results,
                        )
                    except HTTPException as rollback_exc:
                        durable_rollback_exc = rollback_exc
                    # Restore captured process-local preimages under the same
                    # lock regime regardless of durable rollback outcome.
                    restored = await _restore_local_preimage(
                        preimage=local_preimage,
                        state_mgr=state_mgr,
                    )
                    if not restored:
                        # Local restoration cannot be proven: fail closed with
                        # indeterminate semantics, superseding any durable
                        # rollback error (manual intervention required either
                        # way; do not expose keys/hashes/traceback locals).
                        raise HTTPException(
                            status_code=503,
                            detail={
                                "error": "indeterminate_clear",
                                "message": (
                                    "local state restoration could not be "
                                    "proven after rollback; manual intervention "
                                    "required"
                                ),
                            },
                        )
                    # Local restoration proven.  Preserve the durable rollback
                    # error if rollback failed; otherwise re-raise the original
                    # postcondition/local failure.
                    if durable_rollback_exc is not None:
                        raise durable_rollback_exc
                    raise local_exc

            finally:
                for lock in reversed(acquired_probes):
                    lock.release()

    finally:
        for block in reversed(acquired_barriers):
            block.release()

    return total_keys_deleted, total_members_removed


async def _rollback_committed_results(
    *,
    family: str,
    committed_results: list[ClearTransactionResult],
) -> None:
    """Rollback committed clear transactions in reverse order (Finding 3).

    On rollback drift/missing/failure, raises a sanitized HTTPException
    with rollback_failure or indeterminate error code.  Preserves evidence
    by not suppressing the original exception context.
    """
    from .durable import (
        RollbackDriftError,
        RollbackFailedError,
        RollbackReceiptMissingError,
        rollback_clear_transaction,
    )

    for txn_result in reversed(committed_results):
        try:
            await rollback_clear_transaction(
                alias_family=family,
                journal=txn_result.journal,
            )
        except (RollbackDriftError, RollbackReceiptMissingError):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "rollback_failure",
                    "message": (
                        "clear transaction rollback encountered drift or "
                        "missing receipt; manual intervention required"
                    ),
                },
            )
        except RollbackFailedError:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "rollback_failure",
                    "message": (
                        "clear transaction rollback failed; "
                        "manual intervention required"
                    ),
                },
            )
        except Exception:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "indeterminate_clear",
                    "message": (
                        "clear transaction rollback outcome indeterminate; "
                        "manual intervention required"
                    ),
                },
            )


def _complete_reservation_if_owned(
    registry: Any,
    family: str,
    keys: list[str],
    reservation: ClearReservation,
) -> None:
    """Complete *reservation* only if it is still the active owner.

    Uses the public ``get_clear_reservation`` / ``get_clear_reservation_by_identity``
    APIs and object-identity comparison to avoid completing a coalesced
    reservation owned by another request.
    """
    # Try key-based lookup first.
    if keys:
        active = registry.get_clear_reservation(family, keys[0])
        if active is reservation:
            registry.complete_clear_reservation(reservation)
            return
    # Fallback: identity-based lookup (Finding 2: reservation may be
    # identity-scoped with no cooldown_keys).
    for id_hash in reservation.identity_hashes:
        active = registry.get_clear_reservation_by_identity(family, id_hash)
        if active is reservation:
            registry.complete_clear_reservation(reservation)
            return


async def _execute_durable_clear(
    *,
    family: str,
    identity_hash: str,
    cooldown_keys: list[str],
    lane_members: list[str],
) -> Optional[ClearTransactionResult]:
    """Execute the atomic compare-and-clear Redis Lua transaction.

    Returns None when no durable state exists (idempotent path).
    Raises HTTPException on any failure (fail closed).
    """
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "redis_unavailable",
                "message": "cache unavailable; cooldown clear requires durable state",
            },
        )

    try:
        inspection = await inspect_identity_set(
            alias_family=family,
            identity_hash=identity_hash,
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "redis_unavailable",
                "message": "cannot inspect durable state; failing closed",
            },
        )

    if not inspection.exists or inspection.cardinality == 0:
        return None

    try:
        result = await clear_cooldown_transaction(
            alias_family=family,
            identity_hash=identity_hash,
            cooldown_keys=cooldown_keys,
            expected_members=list(inspection.members),
            lane_members=lane_members,
        )
        return result
    except MembershipDriftError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "membership_drift",
                "message": "identity membership drifted during clear; failing closed",
            },
        )
    except ClearIndeterminateError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "indeterminate_clear",
                "message": "clear transaction outcome indeterminate; failing closed",
            },
        )
    except (RollbackFailedError, RollbackDriftError, RollbackReceiptMissingError):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "rollback_failure",
                "message": "clear transaction rollback failed; manual intervention required",
            },
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "clear_failed",
                "message": "durable clear transaction failed; failing closed",
            },
        )


async def _verify_postconditions(
    *,
    family: str,
    canonical_aliases: tuple[str, ...],
    identities: list[_ResolvedIdentity],
    all_keys: list[str],
    state_mgr: AliasRoutingStateManager,
) -> None:
    """Strict postcondition verification before returning 200.

    Covers every resolved lane and identity.  Raises HTTPException(500)
    on any postcondition failure.
    """
    # 1. Durable cooldown keys absent.
    for key in all_keys:
        try:
            absent = await verify_aawm_alias_routing_durable_absence(
                alias_family=family,
                state_kind="cooldown",
                state_key=key,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "postcondition_failure",
                    "message": "durable absence verification failed",
                },
            )
        if not absent:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "postcondition_failure",
                    "message": "durable cooldown key still present after clear",
                },
            )

    # 2. Identity membership absent (per identity).
    for ident in identities:
        if not ident.lane_keys:
            continue
        try:
            post_inspection = await inspect_identity_set(
                alias_family=family,
                identity_hash=ident.identity_hash,
            )
            if post_inspection.exists:
                remaining = post_inspection.members & frozenset(ident.lane_keys)
                if remaining:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "postcondition_failure",
                            "message": "identity membership still present after clear",
                        },
                    )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "postcondition_failure",
                    "message": "identity membership verification failed",
                },
            )

    # 3. Local cooldown-derived state absent.
    for key in all_keys:
        inspection = inspect_cooldown_absence(
            state_mgr,
            alias_family=family,
            canonical_aliases=canonical_aliases,
            cooldown_key=key,
        )
        if inspection.exists:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "postcondition_failure",
                    "message": "local cooldown state still present after clear",
                },
            )

    # 4. Local index absent (per identity).
    for ident in identities:
        if not ident.lane_keys:
            continue
        remaining_lanes = state_mgr.lane_identity_index.lanes_for(
            ident.identity_hash
        )
        overlap = remaining_lanes & frozenset(ident.lane_keys)
        if overlap:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "postcondition_failure",
                    "message": "local index entries still present after clear",
                },
            )


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


async def handle_cooldown_clear(  # noqa: PLR0915
    request: Request,
    user_api_key_dict: Any,
) -> dict[str, Any]:
    """Main handler for POST /aawm/alias-routing/cooldowns/clear.

    Orchestrates validation, auth, topology gate, pure snapshot resolution,
    identity-scoped reservation, local+durable hydration, drain, inspection,
    and execution.  Returns a sanitized response dict.

    Fix 1: resolution is split into a pure active-snapshot phase (no local/
    durable reads) followed by identity-scoped reservation creation.  Only
    after the reservation exists does code inspect local lane index or
    durable identity membership.

    Fix 2: EVERY HTTPException/fail-closed exit emits exactly one structured
    audit event, including schema, auth, topology, and resolution failures.
    """
    # Partial target for pre-resolution audit events (Fix 2).
    partial_target = _ResolvedTarget(family="", target_description="", ingress="")
    # Track whether the inner (post-resolution) catch already emitted audit.
    _inner_audit_emitted = False
    # Defect 3: track prior inspection for post-inspection failure enrichment.
    # Pre-inspection failures keep these empty.
    _prior_source: str = ""
    _prior_ttl: float = 0.0

    try:
        # 1. Parse and validate request body.
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_body",
                    "message": "request body must be valid JSON",
                },
            )
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_body",
                    "message": "request body must be a JSON object",
                },
            )

        # Finding 5/6: pre-validation audit uses ONLY fixed safe labels.
        # Never echo raw caller alias/provider/model strings.
        _raw_alias = body.get("alias")
        _raw_provider = body.get("provider")
        _raw_model = body.get("model")
        _raw_ingress = body.get("ingress")
        if isinstance(_raw_alias, str) and _raw_alias.strip():
            partial_target.target_description = "alias_present"
        elif isinstance(_raw_provider, str) or isinstance(_raw_model, str):
            partial_target.target_description = "exact_target_present"
        else:
            partial_target.target_description = "target_unavailable"
        # Only trust normalized ingress enum values.
        if isinstance(_raw_ingress, str):
            _norm_ingress = _raw_ingress.strip().lower()
            if _norm_ingress in _VALID_INGRESS_FAMILIES:
                partial_target.ingress = _norm_ingress
            else:
                partial_target.ingress = "unavailable"
        else:
            partial_target.ingress = "unavailable"

        clear_req = _parse_and_validate_request(body)

        # 2. Auth: PROXY_ADMIN + master key.
        _check_admin_auth(user_api_key_dict)

        # 3. Topology gate.
        _check_topology_gate()

        # 4. Pure snapshot resolution (no local/durable reads -- Fix 1).
        state_mgr = alias_routing_state
        target = _resolve_target_from_active_snapshot(clear_req)
        partial_target = target  # Full target available for audit.

        # 5. Acquire endpoint per-family serialization lock (Defect 3).
        endpoint_lock = await _get_endpoint_family_lock(target.family)
        async with endpoint_lock:
            registry = state_mgr.publication_intents
            all_identities = target.all_identity_hashes
            reservation: ClearReservation | None = None

            # 5a. Fix 1: establish identity-scoped reservation IMMEDIATELY
            #     after pure snapshot resolution, BEFORE any local/durable
            #     inspection.  This blocks first-ever publication for these
            #     identities even before cooldown_keys are known.
            if all_identities:
                reservation = registry.create_clear_reservation(
                    alias_family=target.family,
                    identity_hashes=frozenset(all_identities),
                    cooldown_keys=frozenset(),
                )

            try:
                # 5b. Fix 1: hydrate from local index AFTER reservation.
                _hydrate_from_local_index(target, state_mgr)

                # 5c. Hydrate from durable (union with local).
                await _hydrate_identities_from_durable(target)

                # 5d. Extend reservation with discovered keys.
                all_keys = target.all_cooldown_keys
                if reservation is not None and all_keys:
                    registry.extend_clear_reservation(
                        reservation,
                        cooldown_keys=frozenset(all_keys),
                    )

                # 5e. Drain prior publication intents (bounded).
                # Finding 1: drain by identity even when no keys discovered,
                # so already-leading unindexed publications are awaited.
                await _drain_publication_intents(
                    registry,
                    target.family,
                    all_keys,
                    identity_hashes=frozenset(all_identities),
                )

                # 5f. Defect 1 fix: post-drain rehydration.  After the
                # identity-based drain completes, a drained publication may
                # have published NEW lane keys.  Rehydrate both local lane
                # index and bounded durable identity membership for all
                # target identities, union newly published lanes, and extend
                # the active reservation with them.  Because identity
                # reservation blocks replacement leaders, this post-drain
                # hydration is stable.
                _hydrate_from_local_index(target, state_mgr)
                await _hydrate_identities_from_durable(target)

                # 5g. Extend reservation with any newly discovered keys.
                all_keys_post_drain = target.all_cooldown_keys
                if reservation is not None and all_keys_post_drain:
                    registry.extend_clear_reservation(
                        reservation,
                        cooldown_keys=frozenset(all_keys_post_drain),
                    )

                # 5h. Bounded stability check: re-inspect local index once
                # more.  If the key set changes unexpectedly (publication
                # raced despite reservation), fail closed.
                pre_stability_keys = set(target.all_cooldown_keys)
                _hydrate_from_local_index(target, state_mgr)
                post_stability_keys = set(target.all_cooldown_keys)
                if post_stability_keys != pre_stability_keys:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error": "post_drain_instability",
                            "message": (
                                "lane key set changed unexpectedly after "
                                "drain; failing closed"
                            ),
                        },
                    )

                # 6. Authoritative prior-state inspection (fail closed).
                prior = await _inspect_prior_state(target, state_mgr)
                _prior_source = prior.source
                _prior_ttl = prior.bounded_remaining_ttl_seconds

                if not prior.has_active:
                    # Finding 3: per-key durable absence proof before not_active.
                    await _verify_not_active_absence(target, state_mgr)

                    # OpenRouter authoritative atomicity: inspect/verify both
                    # maps under the shared lock immediately before returning
                    # not_active.  A concurrent writer after lock release
                    # represents a new upstream failure and is allowed.
                    or_keys = _derive_openrouter_rate_limit_keys(target)
                    if or_keys:
                        async with state_mgr.openrouter_rate_limit.lock:
                            now_or = time.monotonic()
                            for or_key in or_keys:
                                rl_v = state_mgr.openrouter_rate_limit.until_monotonic_by_key.get(or_key, 0.0)
                                if rl_v > now_or:
                                    raise HTTPException(
                                        status_code=409,
                                        detail={
                                            "error": "openrouter_state_race",
                                            "message": (
                                                "openrouter rate-limit entry appeared "
                                                "during not_active verification; "
                                                "failing closed"
                                            ),
                                        },
                                    )
                                fc_v = state_mgr.openrouter_failure_circuit.until_monotonic_by_key.get(or_key, 0.0)
                                if fc_v > now_or:
                                    raise HTTPException(
                                        status_code=409,
                                        detail={
                                            "error": "openrouter_state_race",
                                            "message": (
                                                "openrouter failure-circuit entry appeared "
                                                "during not_active verification; "
                                                "failing closed"
                                            ),
                                        },
                                    )

                    _emit_audit_event(
                        event_type="not_active",
                        target=target,
                        result="not_active",
                        prior_state_source=prior.source,
                    )
                    return {
                        "result": "not_active",
                        "family": target.family,
                        "target_description": target.target_description,
                        "ingress": target.ingress,
                        "candidates": target.candidate_descriptions,
                        "keys_cleared": 0,
                        "members_removed": 0,
                        "affinity_preserved": True,
                        "prior_state_source": prior.source,
                        "bounded_remaining_ttl_seconds": 0.0,
                        "environment": _resolve_environment(),
                        "namespace": _safe_namespace(),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }

                # 7. Execute the clear under canonical lock order.
                keys_cleared, members_removed = await _execute_clear(
                    target, state_mgr
                )

                _emit_audit_event(
                    event_type="success",
                    target=target,
                    result="cleared",
                    prior_state_source=prior.source,
                    bounded_remaining_ttl_seconds=(
                        prior.bounded_remaining_ttl_seconds
                    ),
                )

                return {
                    "result": "cleared",
                    "family": target.family,
                    "target_description": target.target_description,
                    "ingress": target.ingress,
                    "candidates": target.candidate_descriptions,
                    "keys_cleared": keys_cleared,
                    "members_removed": members_removed,
                    "affinity_preserved": True,
                    "prior_state_source": prior.source,
                    "bounded_remaining_ttl_seconds": round(
                        prior.bounded_remaining_ttl_seconds, 2
                    ),
                    "environment": _resolve_environment(),
                    "namespace": _safe_namespace(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }

            except HTTPException as exc:
                # Fix 2: post-resolution failures emit audit (inner catch).
                _inner_audit_emitted = True
                error_code = ""
                if isinstance(exc.detail, dict):
                    error_code = str(exc.detail.get("error", ""))
                # Defect 3: post-inspection failures include prior state
                # context in audit.  Pre-inspection failures keep empty.
                _emit_audit_event(
                    event_type=(
                        "conflict" if exc.status_code == 409 else "failure"
                    ),
                    target=target,
                    result="error",
                    error_code=error_code,
                    prior_state_source=_prior_source,
                    bounded_remaining_ttl_seconds=_prior_ttl,
                )
                # Post-inspection failure response enrichment: when prior
                # source/TTL and trusted resolved candidate context are
                # known, enrich the re-raised HTTPException detail with
                # safe fields matching the exactly-one audit.  Pre-inspection
                # failures must not invent fields.
                if _prior_source and isinstance(exc.detail, dict):
                    exc.detail["prior_state_source"] = _prior_source
                    exc.detail["bounded_remaining_ttl_seconds"] = round(
                        _prior_ttl, 2
                    )
                    exc.detail["candidates"] = target.candidate_descriptions
                    exc.detail["ingress"] = target.ingress
                    exc.detail["environment"] = _resolve_environment()
                    exc.detail["namespace"] = _safe_namespace()
                raise

            finally:
                # Object-identity safe completion: only complete if this
                # reservation is still the active one (not coalesced).
                if reservation is not None:
                    _complete_reservation_if_owned(
                        registry, target.family,
                        target.all_cooldown_keys,
                        reservation,
                    )

    except HTTPException as exc:
        # Fix 2: pre-resolution failures (schema, auth, topology, resolution)
        # emit exactly one audit event.  Post-resolution failures are already
        # emitted by the inner catch; skip to avoid duplicates.
        if not _inner_audit_emitted:
            error_code = ""
            if isinstance(exc.detail, dict):
                error_code = str(exc.detail.get("error", ""))
            _emit_audit_event(
                event_type=(
                    "conflict" if exc.status_code == 409 else "failure"
                ),
                target=partial_target,
                result="error",
                error_code=error_code,
            )
        raise

    except Exception:
        # Catch unexpected internal failures (e.g. RuntimeError from
        # reservation creation) that are not HTTPException.  Emit exactly
        # one sanitized audit event using only safe partial target fields,
        # then raise a stable 503 without exception text, traceback, vars,
        # raw body, keys, hashes, or auth data.
        # asyncio.CancelledError inherits BaseException (Python 3.9+) so
        # cancellation semantics are preserved.
        if not _inner_audit_emitted:
            _emit_audit_event(
                event_type="failure",
                target=partial_target,
                result="error",
                error_code="internal_error",
            )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "internal_error",
                "message": "cooldown clear failed; failing closed",
            },
        )


def _safe_namespace() -> str:
    try:
        return get_aawm_alias_routing_state_namespace()
    except Exception:
        return "unavailable"
