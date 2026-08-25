"""Stub TUI drivers (grok/opencode). Not launch code."""

from __future__ import annotations

from typing import Any

from hv2.errors import PlanError


class StubDriver:
    def __init__(self, name: str) -> None:
        self.name = name

    def _fail(self, *args: Any, **kwargs: Any) -> Any:
        raise PlanError(f"TUI driver {self.name!r} is not implemented")

    select_model = _fail
    send_prompt = _fail
    catalog_picker = _fail
    orchestration = _fail
    launch_argv = _fail

    def describe(self) -> dict[str, Any]:
        return {"tui": self.name, "implemented": False}
