"""CFG-004 criterion 11: dev-only acceptance harness endpoint.

POST /aawm/alias-routing/cooldowns/acceptance

Operations: prepare, inspect, restore.

Lane derivation reuses the EXACT provider-aware branch from
``selection._build_codex_auto_agent_candidate_state`` and builds cooldown
keys with ``_codex_auto_agent_candidate_key``.  The real ``Request`` is
passed through so OpenAI/native auth lanes derive from the incoming
Authorization header.

Durable seeding publishes ALL candidates first (fail closed), then mutates
local memory/index.  Rollback uses ``rollback_cooldown_transaction`` with
the retained journals.  Restore verifies absence via
``verify_aawm_alias_routing_durable_absence`` and ``inspect_identity_set``.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid as _uuid_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, Request

from .cooldown_apply import resolve_lane_identity_hash
from .durable import (
    CooldownTransactionResult,
    inspect_identity_set,
    build_aawm_alias_routing_durable_cache_key,
    delete_aawm_alias_routing_durable_key,
    get_aawm_alias_routing_dual_cache,
    get_aawm_alias_routing_state_namespace,
    publish_cooldown_transaction,
    rollback_cooldown_transaction,
    RollbackDriftError,
    RollbackReceiptMissingError,
    verify_aawm_alias_routing_durable_absence,
)
from .lane_keys import (
    _codex_auto_agent_candidate_key,
    _resolve_codex_auto_agent_openai_cooldown_lane_key,
    _resolve_codex_auto_agent_xai_lane_key,
)
from .policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY as _ALIBABA_LANE_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER as _ALIBABA_PROVIDER,
    CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY as _KIMI_LANE_KEY,
    CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER as _KIMI_PROVIDER,
    CODEX_AUTO_AGENT_OPENCODE_LANE_KEY as _OPENCODE_LANE_KEY,
    CODEX_AUTO_AGENT_OPENCODE_PROVIDER as _OPENCODE_PROVIDER,
    CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY as _OPENROUTER_LANE_KEY,
    CODEX_AUTO_AGENT_OPENROUTER_PROVIDER as _OPENROUTER_PROVIDER,
    CODEX_AUTO_AGENT_XAI_PROVIDER as _XAI_PROVIDER,
)
from .snapshot_select import get_active_routing_snapshot
from .state import AliasRoutingStateManager, alias_routing_state

logger = logging.getLogger("LiteLLMProxy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENVIRONMENT_VAR = "AAWM_LITELLM_ENVIRONMENT"
_REQUIRED_ENVIRONMENT = "litellm-dev"
_ENABLED_VAR = "AAWM_CFG004_ACCEPTANCE_ENABLED"
_RUN_ID_VAR = "AAWM_CFG004_ACCEPTANCE_RUN_ID"
_TOPOLOGY_VAR = "AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER"
_NAMESPACE_PREFIX = "aawm-routing-dev-cfg004-"
_RUN_ID_PATTERN = re.compile(r"\A[0-9a-f]{32}\Z")

_TARGET_PROVIDER = "alibaba_token_plan"
_TARGET_MODEL = "alibaba_token_plan/qwen3.6-flash"
_TARGET_ROUTE_FAMILY = "codex_alibaba_token_plan_chat_completions_adapter"

_MAX_TTL_SECONDS = 1800.0
_MIN_TTL_SECONDS = 10.0

_OPERATION_PREPARE = "prepare"
_OPERATION_INSPECT = "inspect"
_OPERATION_RESTORE = "restore"
_VALID_OPERATIONS = frozenset({_OPERATION_PREPARE, _OPERATION_INSPECT, _OPERATION_RESTORE})

_PREPARE_FIELDS = frozenset({"operation", "run_id", "alias", "ingress", "provider", "model", "ttl_seconds", "codex_oauth_account_id"})
_INSPECT_FIELDS = frozenset({"operation", "run_id"})
_RESTORE_FIELDS = frozenset({"operation", "run_id"})

# Codex OAuth account descriptor: authoritative Mahaf account-id format is a
# canonical UUID (8-4-4-4-12 lowercase hex with hyphens, exactly 36 chars).
_CODEX_OAUTH_ACCOUNT_ID_UUID_PATTERN = re.compile(
    r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)

# ---------------------------------------------------------------------------
# In-memory prepared-state registry (process-local, single-worker)
# ---------------------------------------------------------------------------


@dataclass
class _PreparedState:
    """Captured state from a prepare operation for restore."""

    run_id: str
    canonical_alias: str
    target_identity_hash: str
    target_lane_key: str
    control_identity_hashes: list[str] = field(default_factory=list)
    control_lane_keys: list[str] = field(default_factory=list)
    all_identity_hashes: list[str] = field(default_factory=list)
    all_lane_keys: list[str] = field(default_factory=list)
    # Full transaction results (journals) for rollback.
    publication_results: list[CooldownTransactionResult] = field(default_factory=list)
    # Target result may be consumed by normal clear; tracked separately.
    target_publication_result: Optional[CooldownTransactionResult] = None
    control_publication_results: list[CooldownTransactionResult] = field(default_factory=list)
    local_preimages: dict[str, float] = field(default_factory=dict)
    prepared_at: float = 0.0
    # Optional Codex OAuth account descriptor used for OpenAI lane seeding.
    codex_oauth_account_id: Optional[str] = None


_prepared_runs: dict[str, _PreparedState] = {}

# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def _check_acceptance_gates(run_id: str) -> None:
    """Fail closed unless ALL acceptance gates pass."""
    env = os.getenv(_ENVIRONMENT_VAR, "").strip()
    if env != _REQUIRED_ENVIRONMENT:
        raise HTTPException(status_code=503, detail={
            "error": "acceptance_gate_closed",
            "message": "acceptance requires exact environment",
        })

    enabled = os.getenv(_ENABLED_VAR, "").strip()
    if enabled != "1":
        raise HTTPException(status_code=503, detail={
            "error": "acceptance_gate_closed",
            "message": "acceptance not enabled",
        })

    env_run_id = os.getenv(_RUN_ID_VAR, "").strip()
    if not _RUN_ID_PATTERN.match(env_run_id):
        raise HTTPException(status_code=503, detail={
            "error": "acceptance_gate_closed",
            "message": "run_id environment invalid",
        })
    if env_run_id != run_id:
        raise HTTPException(status_code=403, detail={
            "error": "run_id_mismatch",
            "message": "body run_id does not match environment",
        })

    expected_ns = f"{_NAMESPACE_PREFIX}{run_id}"
    actual_ns = get_aawm_alias_routing_state_namespace()
    if actual_ns != expected_ns:
        raise HTTPException(status_code=503, detail={
            "error": "namespace_mismatch",
            "message": "runtime namespace does not match acceptance run",
        })

    topology = os.getenv(_TOPOLOGY_VAR, "").strip()
    if topology != "1":
        raise HTTPException(status_code=503, detail={
            "error": "topology_gate_closed",
            "message": "acceptance requires single-worker topology",
        })


def _check_admin_auth(user_api_key_dict: Any) -> None:
    """Verify PROXY_ADMIN role and exact master-key match."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    user_role = getattr(user_api_key_dict, "user_role", None)
    if user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail={
            "error": "forbidden",
            "message": "acceptance requires proxy_admin role",
        })

    from litellm.proxy import proxy_server

    master_key = getattr(proxy_server, "master_key", None)
    if not master_key:
        raise HTTPException(status_code=503, detail={
            "error": "auth_unavailable",
            "message": "acceptance unavailable",
        })

    token_hash = getattr(user_api_key_dict, "token", None)
    if not token_hash:
        raise HTTPException(status_code=403, detail={
            "error": "forbidden",
            "message": "acceptance requires authenticated token",
        })

    expected_hash = UserAPIKeyAuth._safe_hash_litellm_api_key(master_key)
    if token_hash != expected_hash:
        raise HTTPException(status_code=403, detail={
            "error": "forbidden",
            "message": "acceptance requires primary authentication",
        })


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_body(body: dict[str, Any]) -> tuple[str, str]:
    """Validate common fields; return (operation, run_id)."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_body",
            "message": "request body must be a JSON object",
        })

    operation = body.get("operation")
    if not isinstance(operation, str) or operation not in _VALID_OPERATIONS:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_operation",
            "message": f"operation must be one of: {sorted(_VALID_OPERATIONS)}",
        })

    if operation == _OPERATION_PREPARE:
        allowed = _PREPARE_FIELDS
    elif operation == _OPERATION_INSPECT:
        allowed = _INSPECT_FIELDS
    else:
        allowed = _RESTORE_FIELDS

    extra = set(body.keys()) - allowed
    if extra:
        raise HTTPException(status_code=400, detail={
            "error": "unexpected_fields",
            "message": f"unexpected fields: {sorted(extra)}",
        })

    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_run_id",
            "message": "run_id must be exactly 32 lowercase hex characters",
        })

    return operation, run_id


def _validate_codex_oauth_account_id(value: Any) -> Optional[str]:
    """Validate the optional ``codex_oauth_account_id`` descriptor.

    Returns the normalized (lowercase canonical) UUID string, or ``None``
    when the field was not supplied.  Raises ``HTTPException(400)`` for
    invalid input.  The authoritative Mahaf account-id format is a canonical
    UUID: exactly 36 characters, 8-4-4-4-12 hex with hyphens.  JWT/token-
    shaped values and arbitrary non-UUID strings are rejected.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_codex_oauth_account_id",
            "message": "codex_oauth_account_id must be a string",
        })
    stripped = value.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_codex_oauth_account_id",
            "message": "codex_oauth_account_id must be nonempty when supplied",
        })
    normalized = stripped.lower()
    if not _CODEX_OAUTH_ACCOUNT_ID_UUID_PATTERN.match(normalized):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_codex_oauth_account_id",
            "message": "codex_oauth_account_id must be a canonical UUID (8-4-4-4-12 hex with hyphens)",
        })
    # Belt-and-suspenders: ensure stdlib agrees this is a valid UUID.
    try:
        _uuid_module.UUID(normalized)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_codex_oauth_account_id",
            "message": "codex_oauth_account_id is not a parseable UUID",
        })
    return normalized


# ---------------------------------------------------------------------------
# Production lane-key derivation (exact provider-aware branch)
# ---------------------------------------------------------------------------


def resolve_production_lane_key(
    request: Request,
    candidate: dict[str, Any],
    *,
    codex_oauth_account_id: Optional[str] = None,
) -> str:
    """Select the exact lane key using the production provider-aware branch.

    Mirrors ``selection._build_codex_auto_agent_candidate_state`` lines
    699-712 exactly.  XAI is out of scope but the branch is preserved.
    """
    provider = candidate.get("provider", "")
    if provider == _OPENROUTER_PROVIDER:
        return _OPENROUTER_LANE_KEY
    elif provider == _XAI_PROVIDER:
        return _resolve_codex_auto_agent_xai_lane_key(candidate)
    elif provider == _KIMI_PROVIDER:
        return _KIMI_LANE_KEY
    elif provider == _ALIBABA_PROVIDER:
        return _ALIBABA_LANE_KEY
    elif provider == _OPENCODE_PROVIDER:
        return _OPENCODE_LANE_KEY
    else:
        # OpenAI/native: when an explicit Codex OAuth account descriptor is
        # supplied, seed the exact production ``chatgpt-account:<id>`` lane
        # that ``_resolve_codex_auto_agent_openai_lane_key`` derives from the
        # ``chatgpt-account-id`` header.
        if codex_oauth_account_id is not None:
            return f"chatgpt-account:{codex_oauth_account_id}"
        return _resolve_codex_auto_agent_openai_cooldown_lane_key(request)


def resolve_production_cooldown_key(
    request: Request,
    candidate: dict[str, Any],
    *,
    codex_oauth_account_id: Optional[str] = None,
) -> str:
    """Build the exact cooldown_key the production selector reads.

    Uses ``_codex_auto_agent_candidate_key`` after the provider-aware lane
    selection, exactly as ``_build_codex_auto_agent_candidate_state`` does.
    """
    lane_key = resolve_production_lane_key(
        request, candidate, codex_oauth_account_id=codex_oauth_account_id,
    )
    cooldown_identity_tag = candidate.get("cooldown_identity_tag")
    return _codex_auto_agent_candidate_key(
        candidate,
        lane_key,
        cooldown_identity_tag=cooldown_identity_tag,
    )


# ---------------------------------------------------------------------------
# Candidate resolution
# ---------------------------------------------------------------------------


def _resolve_eligible_candidates(
    explicit_alias: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve one explicit configured alias for Codex ingress."""
    from .snapshot_select import _resolve_snapshot_alias_candidates

    snapshot = get_active_routing_snapshot()
    if snapshot is None:
        raise HTTPException(status_code=503, detail={
            "error": "no_active_snapshot",
            "message": "no active routing snapshot available",
        })

    alias = snapshot.aliases.get(explicit_alias)
    if alias is None:
        raise HTTPException(status_code=404, detail={
            "error": "alias_not_found",
            "message": "alias not found in active routing snapshot",
        })
    canonical_alias = alias.name
    eligible = _resolve_snapshot_alias_candidates(
        canonical_alias,
        ingress="codex",
        client_product_label="codex",
        now_utc=datetime.now(timezone.utc),
        snapshot=snapshot,
    )
    if not eligible:
        raise HTTPException(status_code=503, detail={
            "error": "no_eligible_candidates",
            "message": "configured alias has no eligible Codex candidates",
        })

    result = []
    for c in eligible:
        result.append({
            "provider": c["provider"],
            "model": c["model"],
            "route_family": c.get("route_family") or "",
            "config_epoch_tag": c.get("config_epoch_tag"),
            "cooldown_identity_tag": c.get("cooldown_identity_tag"),
        })
    return canonical_alias, result


# ---------------------------------------------------------------------------
# Redis availability gate
# ---------------------------------------------------------------------------


def _require_durable_cache() -> Any:
    """Return DualCache or fail 503. No local-only acceptance."""
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is not None:
        return dual_cache
    # Check if Redis is configured but unhealthy.
    try:
        from litellm.proxy.aawm_alias_routing_redis import get_status as _redis_get_status
        status = _redis_get_status()
        if isinstance(status, dict) and status.get("configured") is True:
            raise HTTPException(status_code=503, detail={
                "error": "redis_unavailable",
                "message": "alias routing Redis configured but unhealthy; failing closed",
            })
    except HTTPException:
        raise
    except Exception:
        pass
    raise HTTPException(status_code=503, detail={
        "error": "redis_unavailable",
        "message": "acceptance requires a usable dedicated DualCache",
    })


# ---------------------------------------------------------------------------
# Prepare operation
# ---------------------------------------------------------------------------


async def _handle_prepare(  # noqa: PLR0915 - bounded acceptance handler
    body: dict[str, Any],
    run_id: str,
    request: Request,
    state_mgr: AliasRoutingStateManager,
) -> dict[str, Any]:
    """Seed controls then target using production lane keys; verify state."""
    alias = body.get("alias")
    if not isinstance(alias, str) or not alias.strip():
        raise HTTPException(status_code=400, detail={
            "error": "invalid_alias",
            "message": "alias must be a non-empty string",
        })

    ingress = body.get("ingress")
    if not isinstance(ingress, str) or ingress.strip().lower() != "codex":
        raise HTTPException(status_code=400, detail={
            "error": "invalid_ingress",
            "message": "ingress must be 'codex'",
        })

    provider = body.get("provider")
    model = body.get("model")
    if not isinstance(provider, str) or provider.strip() != _TARGET_PROVIDER:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_target",
            "message": f"provider must be '{_TARGET_PROVIDER}'",
        })
    if not isinstance(model, str) or model.strip() != _TARGET_MODEL:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_target",
            "message": f"model must be '{_TARGET_MODEL}'",
        })

    ttl_raw = body.get("ttl_seconds")
    if not isinstance(ttl_raw, (int, float)) or isinstance(ttl_raw, bool):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_ttl",
            "message": "ttl_seconds must be a number",
        })
    ttl = float(ttl_raw)
    if ttl < _MIN_TTL_SECONDS or ttl > _MAX_TTL_SECONDS:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_ttl",
            "message": f"ttl_seconds must be between {_MIN_TTL_SECONDS} and {_MAX_TTL_SECONDS}",
        })

    if run_id in _prepared_runs:
        raise HTTPException(status_code=409, detail={
            "error": "already_prepared",
            "message": "run already prepared; restore first",
        })

    # Validate optional Codex OAuth account descriptor.
    codex_oauth_account_id = _validate_codex_oauth_account_id(body.get("codex_oauth_account_id"))

    # Require usable DualCache BEFORE any local mutation (fail closed).
    _require_durable_cache()

    canonical_alias, candidates = _resolve_eligible_candidates(alias.strip())

    target_candidates = [
        c for c in candidates
        if c["provider"] == _TARGET_PROVIDER and c["model"] == _TARGET_MODEL
    ]
    control_candidates = [
        c for c in candidates
        if not (c["provider"] == _TARGET_PROVIDER and c["model"] == _TARGET_MODEL)
    ]

    if not target_candidates:
        raise HTTPException(status_code=503, detail={
            "error": "target_not_found",
            "message": "target candidate not in active eligible inventory",
        })
    if not control_candidates:
        raise HTTPException(status_code=503, detail={
            "error": "no_controls",
            "message": "no control candidates in active eligible inventory",
        })

    # Derive production cooldown keys for all candidates.
    target_identity = resolve_lane_identity_hash(candidate=target_candidates[0])
    target_lane_key = resolve_production_cooldown_key(
        request, target_candidates[0], codex_oauth_account_id=codex_oauth_account_id,
    )

    control_identities: list[str] = []
    control_lane_keys: list[str] = []
    for c in control_candidates:
        ih = resolve_lane_identity_hash(candidate=c)
        lk = resolve_production_cooldown_key(
            request, c, codex_oauth_account_id=codex_oauth_account_id,
        )
        control_identities.append(ih)
        control_lane_keys.append(lk)

    all_identities = [target_identity] + control_identities
    all_lane_keys = [target_lane_key] + control_lane_keys

    # Require targeted prestate absent: local + durable (fail closed).
    family_state = state_mgr.codex
    async with family_state.lock:
        for lk in all_lane_keys:
            existing = family_state.cooldown_until_monotonic_by_key.get(lk, 0.0)
            if existing > time.monotonic():
                raise HTTPException(status_code=409, detail={
                    "error": "prestate_not_absent",
                    "message": "targeted lane already has active cooldown",
                })

    for ih in all_identities:
        inspection = await inspect_identity_set(
            alias_family="codex",
            identity_hash=ih,
        )
        if inspection.exists and inspection.cardinality > 0:
            raise HTTPException(status_code=409, detail={
                "error": "prestate_not_absent",
                "message": "identity set already has members in durable state",
            })

    # Capture local preimages.
    local_preimages: dict[str, float] = {}
    async with family_state.lock:
        for lk in all_lane_keys:
            val = family_state.cooldown_until_monotonic_by_key.get(lk)
            if val is not None:
                local_preimages[lk] = val

    # Publish ALL durably FIRST (controls then target). Fail closed.
    # Register recovery state BEFORE first durable publication so any
    # failure retains the record for rollback.
    prepared_state = _PreparedState(
        run_id=run_id,
        canonical_alias=canonical_alias,
        target_identity_hash=target_identity,
        target_lane_key=target_lane_key,
        control_identity_hashes=control_identities,
        control_lane_keys=control_lane_keys,
        all_identity_hashes=all_identities,
        all_lane_keys=all_lane_keys,
        local_preimages=local_preimages,
        prepared_at=time.monotonic(),
        codex_oauth_account_id=codex_oauth_account_id,
    )
    _prepared_runs[run_id] = prepared_state

    try:
        for i in range(len(control_candidates)):
            txn_result = await publish_cooldown_transaction(
                alias_family="codex",
                identity_hash=control_identities[i],
                cooldown_keys=[control_lane_keys[i]],
                lane_members=[control_lane_keys[i]],
                ttl_seconds=ttl,
            )
            prepared_state.publication_results.append(txn_result)
            prepared_state.control_publication_results.append(txn_result)

        target_txn = await publish_cooldown_transaction(
            alias_family="codex",
            identity_hash=target_identity,
            cooldown_keys=[target_lane_key],
            lane_members=[target_lane_key],
            ttl_seconds=ttl,
        )
        prepared_state.publication_results.append(target_txn)
        prepared_state.target_publication_result = target_txn

    except Exception:
        # Reverse-order rollback of any committed publications.
        for txn_result in reversed(prepared_state.publication_results):
            await rollback_cooldown_transaction(
                alias_family="codex",
                journal=txn_result.journal,
            )
        # Retain recovery record for operator inspection.
        raise HTTPException(status_code=500, detail={
            "error": "durable_publication_failed",
            "message": "durable publication failed; rollback completed",
        })

    # Mutate local memory/index AFTER durable commit.
    try:
        async with family_state.lock:
            until = time.monotonic() + ttl
            for lk in all_lane_keys:
                current = family_state.cooldown_until_monotonic_by_key.get(lk, 0.0)
                if until > current:
                    family_state.cooldown_until_monotonic_by_key[lk] = until

        for i in range(len(control_candidates)):
            state_mgr.lane_identity_index.register(
                identity_hash=control_identities[i],
                lane_key=control_lane_keys[i],
            )
        state_mgr.lane_identity_index.register(
            identity_hash=target_identity,
            lane_key=target_lane_key,
        )
    except Exception:
        # Reverse-order rollback durable publications.
        for txn_result in reversed(prepared_state.publication_results):
            await rollback_cooldown_transaction(
                alias_family="codex",
                journal=txn_result.journal,
            )
        # Restore local preimages.
        async with family_state.lock:
            for lk in all_lane_keys:
                family_state.cooldown_until_monotonic_by_key.pop(lk, None)
            for lk, val in local_preimages.items():
                family_state.cooldown_until_monotonic_by_key[lk] = val
        raise HTTPException(status_code=500, detail={
            "error": "local_mutation_failed",
            "message": "local mutation failed; durable rollback completed",
        })

    # Verify local + durable state.
    async with family_state.lock:
        now = time.monotonic()
        for lk in all_lane_keys:
            until = family_state.cooldown_until_monotonic_by_key.get(lk, 0.0)
            if until <= now:
                raise HTTPException(status_code=500, detail={
                    "error": "seed_verification_failed",
                    "message": "local cooldown not active after seeding",
                })

    for ih in all_identities:
        inspection = await inspect_identity_set(
            alias_family="codex",
            identity_hash=ih,
        )
        if not (inspection.exists and inspection.cardinality > 0):
            raise HTTPException(status_code=500, detail={
                "error": "seed_verification_failed",
                "message": "durable identity membership not active after seeding",
            })

    logger.info("aawm_cfg004_acceptance_prepare run_id=%s target=%s/%s controls=%d ttl=%.1f",
                run_id, _TARGET_PROVIDER, _TARGET_MODEL, len(control_candidates), ttl)

    return {
        "result": "prepared",
        "run_id": run_id,
        "target": {"provider": _TARGET_PROVIDER, "model": _TARGET_MODEL},
        "control_count": len(control_candidates),
        "controls": [
            {"provider": c["provider"], "model": c["model"], "route_family": c["route_family"]}
            for c in control_candidates
        ],
        "ttl_seconds": ttl,
        "environment": _REQUIRED_ENVIRONMENT,
        "namespace": f"{_NAMESPACE_PREFIX}{run_id}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Inspect operation
# ---------------------------------------------------------------------------


async def _handle_inspect(
    run_id: str,
    request: Request,
    state_mgr: AliasRoutingStateManager,
) -> dict[str, Any]:
    """Return public state booleans for prepared candidates."""
    prepared = _prepared_runs.get(run_id)
    if prepared is None:
        raise HTTPException(status_code=404, detail={
            "error": "not_prepared",
            "message": "no prepared state for this run_id",
        })

    _, candidates = _resolve_eligible_candidates(prepared.canonical_alias)
    family_state = state_mgr.codex
    now = time.monotonic()

    candidate_states = []
    for c in candidates:
        ih = resolve_lane_identity_hash(candidate=c)
        lk = resolve_production_cooldown_key(
            request, c,
            codex_oauth_account_id=prepared.codex_oauth_account_id,
        )
        is_target = (c["provider"] == _TARGET_PROVIDER and c["model"] == _TARGET_MODEL)

        async with family_state.lock:
            until = family_state.cooldown_until_monotonic_by_key.get(lk, 0.0)
            local_active = until > now

        inspection = await inspect_identity_set(
            alias_family="codex",
            identity_hash=ih,
        )
        durable_active = inspection.exists and inspection.cardinality > 0

        candidate_states.append({
            "provider": c["provider"],
            "model": c["model"],
            "route_family": c["route_family"],
            "role": "target" if is_target else "control",
            "local_cooldown_active": local_active,
            "durable_cooldown_active": durable_active,
        })

    snapshot = get_active_routing_snapshot()
    config_identity = ""
    if snapshot is not None:
        config_identity = snapshot.config_version

    return {
        "result": "inspected",
        "run_id": run_id,
        "candidates": candidate_states,
        "config_identity": config_identity,
        "environment": _REQUIRED_ENVIRONMENT,
        "namespace": f"{_NAMESPACE_PREFIX}{run_id}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# SET-type durable key deletion (acceptance drift fallback)
# ---------------------------------------------------------------------------


async def _delete_durable_set_key(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> None:
    """Delete a Redis SET-type durable key via type-agnostic DEL.

    ``delete_aawm_alias_routing_durable_key`` validates with GET, which
    raises WRONGTYPE on SET keys (lane_identity).  This helper uses DEL
    directly and verifies absence with EXISTS.  Safe in the acceptance
    context: prepare verified prestate absent and the single-worker
    topology gate prevents concurrent mutation.  Phase 5 postcondition
    verification still enforces full absence via ``inspect_identity_set``.
    """
    context = f"acceptance-set-delete kind={state_kind}"
    dual_cache = get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        raise RuntimeError(f"{context}: no Redis cache available")

    cache_key = build_aawm_alias_routing_durable_cache_key(
        alias_family=alias_family,
        state_kind=state_kind,
        state_key=state_key,
    )

    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        raise RuntimeError(f"{context}: dual cache has no redis_cache")
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RuntimeError(f"{context}: redis_cache missing init_async_client")
    try:
        client = init_fn()
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise RuntimeError(f"{context}: failed to init redis client") from exc
    if client is None:
        raise RuntimeError(f"{context}: redis client unavailable")

    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    namespaced_key = fix_ns(key=cache_key) if callable(fix_ns) else cache_key

    delete_fn = getattr(client, "delete", None)
    if not callable(delete_fn):
        raise RuntimeError(f"{context}: redis client missing delete")
    try:
        await delete_fn(namespaced_key)
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise RuntimeError(f"{context}: redis delete failed") from exc

    # Clear in-memory tier (best-effort; Phase 5 verifies via Redis).
    in_memory_cache = getattr(dual_cache, "in_memory_cache", None)
    if in_memory_cache is not None:
        mem_delete = getattr(in_memory_cache, "delete_cache", None)
        if callable(mem_delete):
            try:
                mem_delete(cache_key)
            except Exception as exc:  # noqa: BLE001 - fail closed
                raise RuntimeError(f"{context}: in-memory delete failed") from exc

    # Verify absence with EXISTS (type-agnostic, unlike GET).
    exists_fn = getattr(client, "exists", None)
    if not callable(exists_fn):
        raise RuntimeError(f"{context}: redis client missing exists")
    try:
        remaining = await exists_fn(namespaced_key)
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise RuntimeError(f"{context}: redis exists check failed") from exc
    if remaining:
        raise RuntimeError(f"{context}: key still present after deletion")


# ---------------------------------------------------------------------------
# Restore operation
# ---------------------------------------------------------------------------


async def _handle_restore(  # noqa: PLR0915 - bounded acceptance handler
    run_id: str,
    state_mgr: AliasRoutingStateManager,
) -> dict[str, Any]:
    """Unconditionally clear prepared state; do NOT pop until verified.

    Ordering contract:
    1. Prove target absent (local + durable) BEFORE any local clearing.
    2. Clear local state and identity index.
    3. Retain target receipt while controls roll back in reverse order.
    4. Restore local preimages.
    5. Verify local + durable postconditions.
    6. Delete target receipt as the FINAL cleanup step.
    7. Pop _prepared_runs only after full success.
    """
    prepared = _prepared_runs.get(run_id)
    if prepared is None:
        return {
            "result": "restored",
            "run_id": run_id,
            "cleared_identities": 0,
            "environment": _REQUIRED_ENVIRONMENT,
            "namespace": f"{_NAMESPACE_PREFIX}{run_id}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    family_state = state_mgr.codex

    def _restore_error(
        *,
        error: str,
        phase: str,
        failure_class: str,
        target_receipt_retained: bool,
        message: str,
        failures: Optional[list[str]] = None,
    ) -> HTTPException:
        detail: dict[str, Any] = {
            "error": error,
            "phase": phase,
            "failure_class": failure_class,
            "run_id": run_id,
            "recovery_state_retained": True,
            "target_receipt_retained": target_receipt_retained,
            "message": message,
        }
        if failures:
            detail["failures"] = failures
        return HTTPException(status_code=500, detail=detail)

    # --- Phase 1: Prove target absent BEFORE clearing any local state. ---
    # This must be non-vacuous: inspect the live local lane and durable
    # identity BEFORE restore mutates anything.
    target_absence_failures: list[str] = []
    now = time.monotonic()
    async with family_state.lock:
        until = family_state.cooldown_until_monotonic_by_key.get(
            prepared.target_lane_key, 0.0
        )
        if until > now:
            target_absence_failures.append(
                "target local cooldown still active before control rollback"
            )

    try:
        target_inspection = await inspect_identity_set(
            alias_family="codex",
            identity_hash=prepared.target_identity_hash,
        )
    except Exception as exc:
        raise _restore_error(
            error="target_inspect_failed",
            phase="target_absence_proof",
            failure_class=type(exc).__name__,
            target_receipt_retained=True,
            message="target identity inspection failed during absence proof",
        )
    if target_inspection.exists and target_inspection.cardinality > 0:
        target_absence_failures.append(
            "target durable identity still has members before control rollback"
        )

    if target_absence_failures:
        raise _restore_error(
            error="target_absence_not_proven",
            phase="target_absence_proof",
            failure_class="precondition",
            target_receipt_retained=True,
            message="target cooldown absence not proven before control rollback",
            failures=target_absence_failures,
        )

    # --- Phase 2: Clear local state and identity index. ---
    # Only reached after target local+durable absence is proven.
    state_mgr.clear_cooldown_state(
        alias_family="codex",
        canonical_aliases=(prepared.canonical_alias,),
        cooldown_keys=prepared.all_lane_keys,
    )
    for ih in prepared.all_identity_hashes:
        state_mgr.lane_identity_index.remove_identity(ih)

    # --- Phase 3: Rollback controls in strict reverse order. ---
    # Target receipt is RETAINED during control rollback.
    # The Lua rollback drift check compares absolute TTL values; natural
    # Redis TTL decay between prepare and restore (e.g. during the proof
    # phase) exceeds its 1-second tolerance and produces a false
    # RollbackDriftError.  When this happens, fall back to direct durable
    # deletion of the acceptance-seeded state.  This is safe because
    # prepare verified prestate was absent and the single-worker topology
    # gate prevents concurrent mutation.  Phase 5 postcondition
    # verification still enforces full absence for genuine drift.
    rollback_drift_fallback = False
    try:
        for txn_result in reversed(prepared.control_publication_results):
            await rollback_cooldown_transaction(
                alias_family="codex",
                journal=txn_result.journal,
            )
    except (RollbackDriftError, RollbackReceiptMissingError):
        rollback_drift_fallback = True
    except Exception as exc:
        raise _restore_error(
            error="control_rollback_failed",
            phase="control_rollback",
            failure_class=type(exc).__name__,
            target_receipt_retained=True,
            message="control rollback failed; prepared state retained for recovery",
        )

    if rollback_drift_fallback:
        try:
            for txn_result in prepared.control_publication_results:
                journal = txn_result.journal
                for ck in journal.cooldown_keys:
                    await delete_aawm_alias_routing_durable_key(
                        alias_family="codex",
                        state_kind="cooldown",
                        state_key=ck,
                    )
                await _delete_durable_set_key(
                    alias_family="codex",
                    state_kind="lane_identity",  # Redis SET; GET-based delete raises WRONGTYPE
                    state_key=journal.identity_hash,
                )
                receipt_state_key = f"txn-receipt:{txn_result.transaction_id}"
                await delete_aawm_alias_routing_durable_key(
                    alias_family="codex",
                    state_kind="txn_receipt",
                    state_key=receipt_state_key,
                )
        except Exception as exc:
            raise _restore_error(
                error="control_rollback_failed",
                phase="control_rollback",
                failure_class=type(exc).__name__,
                target_receipt_retained=True,
                message="control rollback drift fallback failed; prepared state retained for recovery",
            )

    # --- Phase 4: Restore local preimages. ---
    if prepared.local_preimages:
        async with family_state.lock:
            for lk, val in prepared.local_preimages.items():
                family_state.cooldown_until_monotonic_by_key[lk] = val

    # --- Phase 5: Verify local + durable postconditions. ---
    failures: list[str] = []
    now = time.monotonic()
    async with family_state.lock:
        for lk in prepared.all_lane_keys:
            if lk in prepared.local_preimages:
                continue
            until = family_state.cooldown_until_monotonic_by_key.get(lk, 0.0)
            if until > now:
                failures.append("local lane still active")

    for lk in prepared.all_lane_keys:
        try:
            absent = await verify_aawm_alias_routing_durable_absence(
                alias_family="codex",
                state_kind="cooldown",
                state_key=lk,
            )
        except Exception as exc:
            raise _restore_error(
                error="postcondition_verify_failed",
                phase="postcondition_verification",
                failure_class=type(exc).__name__,
                target_receipt_retained=True,
                message="durable cooldown absence check failed",
            )
        if not absent:
            failures.append("durable cooldown key still present")

    for ih in prepared.all_identity_hashes:
        try:
            inspection = await inspect_identity_set(
                alias_family="codex",
                identity_hash=ih,
            )
        except Exception as exc:
            raise _restore_error(
                error="postcondition_inspect_failed",
                phase="postcondition_verification",
                failure_class=type(exc).__name__,
                target_receipt_retained=True,
                message="identity inspection failed during postcondition verification",
            )
        if inspection.exists and inspection.cardinality > 0:
            failures.append("durable identity still has members")

    # Verify control publication receipts are absent (rollback deletes them).
    for txn_result in prepared.control_publication_results:
        receipt_state_key = f"txn-receipt:{txn_result.transaction_id}"
        try:
            receipt_absent = await verify_aawm_alias_routing_durable_absence(
                alias_family="codex",
                state_kind="txn_receipt",
                state_key=receipt_state_key,
            )
        except Exception as exc:
            raise _restore_error(
                error="control_receipt_verify_failed",
                phase="postcondition_verification",
                failure_class=type(exc).__name__,
                target_receipt_retained=True,
                message="control receipt absence verification failed",
            )
        if not receipt_absent:
            failures.append("control publication receipt still present after rollback")

    if failures:
        logger.warning(
            "aawm_cfg004_acceptance_restore verification failed run_id=%s: %s",
            run_id, failures,
        )
        raise _restore_error(
            error="restore_verification_failed",
            phase="postcondition_verification",
            failure_class="verification",
            target_receipt_retained=True,
            message="control rollback completed but absence verification failed",
            failures=failures,
        )

    # --- Phase 6: Delete target receipt as the FINAL cleanup step. ---
    if prepared.target_publication_result is not None:
        target_txn_id = prepared.target_publication_result.transaction_id
        receipt_state_key = f"txn-receipt:{target_txn_id}"
        try:
            await delete_aawm_alias_routing_durable_key(
                alias_family="codex",
                state_kind="txn_receipt",
                state_key=receipt_state_key,
            )
        except Exception as exc:
            raise _restore_error(
                error="target_receipt_deletion_failed",
                phase="target_receipt_cleanup",
                failure_class=type(exc).__name__,
                target_receipt_retained=True,
                message="target receipt deletion failed; prepared state retained",
            )
        # Verify the receipt is absent.
        try:
            receipt_absent = await verify_aawm_alias_routing_durable_absence(
                alias_family="codex",
                state_kind="txn_receipt",
                state_key=receipt_state_key,
            )
        except Exception as exc:
            raise _restore_error(
                error="target_receipt_verify_failed",
                phase="target_receipt_cleanup",
                failure_class=type(exc).__name__,
                target_receipt_retained=False,
                message="target receipt absence verification failed after deletion",
            )
        if not receipt_absent:
            raise _restore_error(
                error="target_receipt_cleanup_failed",
                phase="target_receipt_cleanup",
                failure_class="verification",
                target_receipt_retained=True,
                message="target publication receipt still present after deletion",
            )

    # --- Phase 7: Pop only after full success. ---
    _prepared_runs.pop(run_id, None)

    logger.info("aawm_cfg004_acceptance_restore run_id=%s cleared=%d",
                run_id, len(prepared.all_identity_hashes))

    return {
        "result": "restored",
        "run_id": run_id,
        "cleared_identities": len(prepared.all_identity_hashes),
        "environment": _REQUIRED_ENVIRONMENT,
        "namespace": f"{_NAMESPACE_PREFIX}{run_id}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoint handler
# ---------------------------------------------------------------------------


async def handle_cooldown_acceptance(
    request: Request,
    user_api_key_dict: Any,
) -> dict[str, Any]:
    """Main handler for POST /aawm/alias-routing/cooldowns/acceptance."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={
            "error": "invalid_body",
            "message": "request body must be valid JSON",
        })

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail={
            "error": "invalid_body",
            "message": "request body must be a JSON object",
        })

    operation, run_id = _validate_body(body)
    _check_admin_auth(user_api_key_dict)
    _check_acceptance_gates(run_id)

    state_mgr = alias_routing_state

    if operation == _OPERATION_PREPARE:
        return await _handle_prepare(body, run_id, request, state_mgr)
    elif operation == _OPERATION_INSPECT:
        return await _handle_inspect(run_id, request, state_mgr)
    else:
        return await _handle_restore(run_id, state_mgr)
