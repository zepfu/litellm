"""RR-001: autobump workflow paths must watch the real bundled cost map file."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_WORKFLOW = _REPO / ".github" / "workflows" / "aawm-artifact-autobump.yml"
_BUNDLED = (
    _REPO / "litellm" / "bundled_model_prices_and_context_window_fallback.json"
)


def test_autobump_workflow_watches_bundled_cost_map() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "litellm/bundled_model_prices_and_context_window_fallback.json" in text
    assert "litellm/model_prices_and_context_window_backup.json" not in text
    assert _BUNDLED.is_file()
