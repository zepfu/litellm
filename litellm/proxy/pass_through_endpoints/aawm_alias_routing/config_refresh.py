"""Alias-config refresh handler body and YAML loading.

Wave 5A extraction from ``llm_passthrough_endpoints.py``.  The ``@router.post``
decorator stays in the god module; this module owns the handler callable and
the YAML source loader.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Request
from pydantic import ValidationError

from .config_compiler import (
    ConfigCompileError as _AawmAliasConfigCompileError,
    compile_yaml as _compile_aawm_alias_routing_yaml,
)
from .config_snapshot import RoutingSnapshot as _RoutingSnapshot
from .snapshot_select import (
    get_active_routing_snapshot,
    set_active_routing_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Path relative to this file: aawm_alias_routing/ -> pass_through_endpoints/
# -> proxy/ -> proxy/aawm_alias_config/read.yaml
_DEFAULT_AAWM_ALIAS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "aawm_alias_config" / "read.yaml"
)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _load_aawm_alias_routing_source_yaml(*, inline_yaml: Optional[str]) -> str:
    """Return the raw YAML to compile: an inline override, or the default file."""
    if inline_yaml is not None:
        return inline_yaml
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
    changed = previous_snapshot is None or previous_snapshot.config_hash != attempted_snapshot.config_hash
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
    }
