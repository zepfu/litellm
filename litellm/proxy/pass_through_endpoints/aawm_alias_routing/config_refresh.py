"""Provider-neutral alias-config refresh handler and YAML loading.

Wave 7 owner preparation from ``llm_passthrough_endpoints.py``.  The
``@router.post`` decorator stays in the god module; this module owns the
handler callable, the YAML source loader, and response shaping.

Cross-module dependencies (snapshot holder, config compiler, default config
path) are injectable via :func:`configure_config_refresh_runtime` so later
serial integration can leave only a thin decorated route delegate in the god
module.  When the runtime is not configured, the module falls back to direct
sibling imports for backward compatibility with existing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request
from pydantic import ValidationError

from .config_compiler import (
    ConfigCompileError as _AawmAliasConfigCompileError,
    compile_yaml as _compile_aawm_alias_routing_yaml,
)
from .config_snapshot import (
    AliasReference as _AliasReference,
    RoutingSnapshot as _RoutingSnapshot,
)
from .snapshot_select import (
    get_active_routing_snapshot,
    set_active_routing_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Path relative to this file: aawm_alias_routing/ -> pass_through_endpoints/
# -> proxy/ -> proxy/aawm_alias_config/basic.yaml
_DEFAULT_AAWM_ALIAS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "aawm_alias_config" / "basic.yaml"
)


def _snapshot_candidate_order(snapshot: Any) -> dict[str, list[dict[str, Any]]]:
    """Return sanitized, authoritative per-alias ordered candidate identities.

    Exposes the COMPLETE sanitized identity of each candidate in the
    snapshot's compiled order: provider, model, route_family,
    anthropic_route_family, priority, and last_resort (when the compiled
    candidate exposes it).  Alias names are emitted in sorted order for
    deterministic output.  Never exposes secrets, raw YAML, weights, or
    schedule internals.  Used by the refresh response so callers (e.g. the
    CFG-003 transactional harness) can prove the active restored full
    candidate order from an authoritative runtime surface rather than a
    locally inferred order.  Callers MUST compare the exact complete ordered
    list (no prefix acceptance, no extra tail).
    """
    order: dict[str, list[dict[str, Any]]] = {}
    try:
        aliases = snapshot.aliases
    except AttributeError:
        return order
    for alias_name in sorted(dict(aliases).keys()):
        alias = aliases[alias_name]
        candidates: list[dict[str, Any]] = []
        for cand in alias.candidates:
            if isinstance(cand, _AliasReference):
                candidates.append(
                    {
                        "alias_reference": cand.alias_name,
                        "priority": cand.priority,
                        "last_resort": cand.priority == 0,
                    }
                )
                continue
            identity: dict[str, Any] = {
                "provider": cand.provider,
                "model": cand.model,
                "route_family": cand.route_family,
                "anthropic_route_family": cand.anthropic_route_family,
                "priority": cand.priority,
            }
            # Derive last_resort from the real compiled rule: priority == 0
            # is reserved for last-resort candidates (config_compiler contract).
            identity["last_resort"] = cand.priority == 0
            candidates.append(identity)
        order[str(alias_name)] = candidates
    return order


# ---------------------------------------------------------------------------
# Injected runtime seams (Wave 7 DI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigRefreshRuntime:
    """Injected dependencies for the config-refresh handler.

    All callables mirror the signatures of the sibling-module functions they
    replace.  ``compile_error_types`` is the tuple of exception classes that
    indicate a compilation failure (caught and turned into a 400 response).
    """

    compile_yaml: Callable[[str], Any]
    get_active_snapshot: Callable[[], Optional[Any]]
    set_active_snapshot: Callable[[Any], Optional[Any]]
    load_default_source_yaml: Callable[[], str]
    compile_error_types: tuple[type[BaseException], ...]


_runtime: Optional[ConfigRefreshRuntime] = None


def configure_config_refresh_runtime(*, runtime: ConfigRefreshRuntime) -> None:
    """Bind the provider-neutral config-refresh runtime.

    Must be called before the handler is invoked via the DI path.  When not
    called, the handler falls back to direct sibling-module imports (backward
    compatible with the Wave 5A wiring).
    """
    global _runtime
    _runtime = runtime


def _get_runtime() -> Optional[ConfigRefreshRuntime]:
    """Return the configured runtime, or ``None`` for fallback mode."""
    return _runtime


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _load_aawm_alias_routing_source_yaml(*, inline_yaml: Optional[str]) -> str:
    """Return the raw YAML to compile: an inline override, or the default file."""
    if inline_yaml is not None:
        return inline_yaml
    runtime = _get_runtime()
    if runtime is not None:
        return runtime.load_default_source_yaml()
    return _DEFAULT_AAWM_ALIAS_CONFIG_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Route handler body (decorator stays in god module)
# ---------------------------------------------------------------------------


async def aawm_alias_config_refresh_route(request: Request) -> dict[str, Any]:
    """Validate + compile the AAWM alias-routing config and atomically activate it.

    Fails closed: a compile/validation error preserves the previously active
    (last-known-good) snapshot untouched and returns a secret-safe error body.
    A no-op re-post (identical content hash) is a successful 200 with
    ``changed: False``.
    """
    try:
        request_body = await request.json()
    except Exception:
        request_body = {}
    inline_yaml = request_body.get("yaml") if isinstance(request_body, dict) else None
    if inline_yaml is not None and not isinstance(inline_yaml, str):
        raise HTTPException(
            status_code=400,
            detail={"error": "AAWM alias-routing config 'yaml' field must be a string"},
        )

    try:
        source_yaml = _load_aawm_alias_routing_source_yaml(inline_yaml=inline_yaml)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "failed to read AAWM alias-routing config source"},
        ) from exc

    runtime = _get_runtime()
    if runtime is not None:
        return await _refresh_via_runtime(runtime, source_yaml)
    return await _refresh_via_direct_imports(source_yaml)


async def _refresh_via_runtime(
    runtime: ConfigRefreshRuntime,
    source_yaml: str,
) -> dict[str, Any]:
    """DI path: compile and activate using injected dependencies."""
    try:
        attempted_snapshot = runtime.compile_yaml(source_yaml)
    except (*runtime.compile_error_types, ValidationError):
        last_known_good = runtime.get_active_snapshot()
        error_detail: dict[str, Any] = {
            "error": "AAWM alias-routing config failed to compile; last-known-good snapshot remains active",
        }
        if last_known_good is not None:
            error_detail["active_config_hash"] = last_known_good.config_hash
            error_detail["config_version"] = last_known_good.config_version
        raise HTTPException(status_code=400, detail=error_detail)

    previous_snapshot = runtime.get_active_snapshot()
    changed = (
        previous_snapshot is None
        or previous_snapshot.config_hash != attempted_snapshot.config_hash
    )
    if changed:
        runtime.set_active_snapshot(attempted_snapshot)
        active_snapshot = attempted_snapshot
    else:
        assert previous_snapshot is not None
        active_snapshot = previous_snapshot

    return {
        "changed": changed,
        "attempted_config_hash": attempted_snapshot.config_hash,
        "active_config_hash": active_snapshot.config_hash,
        "config_version": active_snapshot.config_version,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "active_candidate_order": _snapshot_candidate_order(active_snapshot),
    }


async def _refresh_via_direct_imports(source_yaml: str) -> dict[str, Any]:
    """Fallback path: compile and activate using direct sibling imports."""
    try:
        attempted_snapshot = _compile_aawm_alias_routing_yaml(source_yaml)
    except (_AawmAliasConfigCompileError, ValidationError):
        last_known_good = get_active_routing_snapshot()
        error_detail: dict[str, Any] = {
            "error": "AAWM alias-routing config failed to compile; last-known-good snapshot remains active",
        }
        if last_known_good is not None:
            error_detail["active_config_hash"] = last_known_good.config_hash
            error_detail["config_version"] = last_known_good.config_version
        raise HTTPException(status_code=400, detail=error_detail)

    previous_snapshot = get_active_routing_snapshot()
    changed = (
        previous_snapshot is None
        or previous_snapshot.config_hash != attempted_snapshot.config_hash
    )
    active_snapshot: _RoutingSnapshot
    if changed:
        set_active_routing_snapshot(attempted_snapshot)
        active_snapshot = attempted_snapshot
    else:
        # No-op: identical content already active. Do not replace the
        # snapshot object -- in-flight readers holding a reference to the
        # active snapshot must keep observing the exact same object.
        assert previous_snapshot is not None
        active_snapshot = previous_snapshot

    return {
        "changed": changed,
        "attempted_config_hash": attempted_snapshot.config_hash,
        "active_config_hash": active_snapshot.config_hash,
        "config_version": active_snapshot.config_version,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "active_candidate_order": _snapshot_candidate_order(active_snapshot),
    }
