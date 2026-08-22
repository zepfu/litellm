"""session_history check. Live DB queries are out of scope; enabled:false stays skip."""

from __future__ import annotations

from typing import Any, Mapping


def session_history_result(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    spec = checks.get("session_history")
    if not isinstance(spec, dict) or not spec.get("enabled"):
        return {"enabled": False, "skipped": True}
    return {
        "enabled": True,
        "skipped": True,
        "reason": "query not implemented",
    }
