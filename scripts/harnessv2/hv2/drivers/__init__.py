"""TUI drivers. V1 implements Ohmypi only. No Claude module."""

from __future__ import annotations

from typing import Any, Mapping

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
        from hv2.drivers.stub import StubDriver

        return StubDriver(tui)
    if tui == "ohmypi":
        from hv2.drivers.ohmypi import OhmypiDriver

        return OhmypiDriver(config)
    raise PlanError(f"unknown TUI driver {tui!r}")
