"""TUI drivers. Ohmypi and Codex are implemented. No Claude module."""

from __future__ import annotations

from typing import Any, Mapping

from hv2.drivers.codex import CodexDriver
from hv2.drivers.ohmypi import OhmypiDriver
from hv2.drivers.stub import StubDriver
from hv2.errors import PlanError
from hv2.load_config import as_str_list


def driver_for(tui: str, config: Mapping[str, Any]):
    tuis = config.get("tuis") if isinstance(config.get("tuis"), dict) else {}
    out_of_scope = set(as_str_list(tuis.get("out_of_scope")))
    if tui in out_of_scope or tui == "claude":
        raise PlanError(
            f"TUI {tui!r} is out of scope for harness v2. Claude stays in local-ci."
        )
    implemented = set(as_str_list(tuis.get("implemented")))
    stubs = set(as_str_list(tuis.get("stubs")))
    if tui in stubs and tui not in implemented:
        return StubDriver(tui)
    if tui == "ohmypi":
        return OhmypiDriver(config)
    if tui == "codex":
        return CodexDriver(config)
    raise PlanError(f"unknown TUI driver {tui!r}")
