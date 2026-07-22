"""GREEN Wave 6 test: shadow-parity between the snapshot-driven ``read`` pilot
selection and the hard-coded ``CODEX_AAWM_LOW_CANDIDATES`` policy table.

``litellm/proxy/aawm_alias_config/read.yaml`` was authored (Wave 3) to mirror
``CODEX_AAWM_LOW_CANDIDATES`` (policy.py:227-277) 1:1 by descending integer
priority, plus one additional schedule-windowed promo candidate that is
outside its active window as of this test's reference clock. This test
compiles that YAML and asserts the resulting snapshot-driven ``read``
selection reproduces the ``CODEX_AAWM_LOW_CANDIDATES`` ordering/eligibility
(same providers/models/route_families, same order) -- i.e. shadow parity.

Tolerance: the promo candidate (``alibaba_token_plan/qwen3.8-max-preview``,
schedule 2026-07-01..2026-08-01) is INCLUDED in the read.yaml file and is
outside its window relative to the fixed reference clock used here
(2026-06-15, before the window opens), so it is excluded from the eligible
set and the remaining ordering reproduces ``CODEX_AAWM_LOW_CANDIDATES``
exactly. If the reference clock instead falls inside the promo window, the
promo candidate would legitimately outrank everything else -- that is
intentional product behavior, not a parity break, so this test pins its
``now_utc`` outside the window to assert the steady-state parity claim.

Ambient state (cooldown/session-affinity dicts and the process-local active
snapshot holder) is reset before and after via the same
``clear_codex_auto_agent_alias_state``-style approach used by Wave 4's
``test_read_pilot_selection.py`` so this test cannot flap on shared,
process-global state left over from other tests.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_compiler as compiler,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
    CODEX_AAWM_LOW_CANDIDATES,
)

_READ_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
    "litellm",
    "proxy",
    "aawm_alias_config",
    "read.yaml",
)

# Reference clock intentionally BEFORE the promo schedule window
# (2026-07-01T00:00:00Z .. 2026-08-01T00:00:00Z) opens, so the promo
# candidate is excluded from eligibility and the remaining candidates
# reproduce CODEX_AAWM_LOW_CANDIDATES's ordering exactly (documented
# tolerance above).
_REFERENCE_NOW_UTC = dt.datetime(2026, 6, 15, tzinfo=dt.timezone.utc)


@pytest.fixture(autouse=True)
def _reset_alias_routing_ambient_state():
    """Neutralize shared/process-global cooldown, affinity, and snapshot state.

    Mirrors the reset approach in
    ``tests/test_litellm/proxy/pass_through_endpoints/test_llm_pass_through_endpoints.py``'s
    ``clear_codex_auto_agent_alias_state`` autouse fixture, scoped to just the
    state this shadow-parity test could otherwise flap on.
    """
    previous_snapshot = lpe.get_active_routing_snapshot()
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    yield
    lpe._codex_auto_agent_cooldown_until_monotonic_by_key.clear()
    lpe._codex_auto_agent_session_affinity_by_key.clear()
    lpe.set_active_routing_snapshot(previous_snapshot)


def _compile_read_yaml():
    with open(_READ_YAML_PATH, "r", encoding="utf-8") as handle:
        raw_yaml = handle.read()
    return compiler.compile_yaml(raw_yaml)


def test_read_yaml_exists_and_compiles() -> None:
    """The read.yaml pilot config file compiles into a valid snapshot with a read alias."""
    snapshot = _compile_read_yaml()
    assert "read" in snapshot.aliases
    assert len(snapshot.aliases["read"].candidates) > 0


def test_shadow_parity_read_vs_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot-driven ``read`` selection (promo window closed) reproduces
    ``CODEX_AAWM_LOW_CANDIDATES`` ordering/eligibility exactly."""
    snapshot = _compile_read_yaml()
    lpe.set_active_routing_snapshot(snapshot)

    # Force the auto-agent alias resolver's fallback table to resolve "read"
    # from the snapshot regardless of the untouched hard-coded alias-name
    # mapping (Wave 4's helper is keyed on the literal alias name "read").
    selected = lpe._select_read_pilot_snapshot_candidates(
        client_product_label=None,
        now_utc=_REFERENCE_NOW_UTC,
    )

    selected_triples = [(c["provider"], c["model"], c["route_family"]) for c in selected]
    low_triples = [(c["provider"], c["model"], c["route_family"]) for c in CODEX_AAWM_LOW_CANDIDATES]

    assert selected_triples == low_triples, (
        "snapshot-driven read selection diverged from CODEX_AAWM_LOW_CANDIDATES:\n"
        f"selected={selected_triples}\nlow_candidates={low_triples}"
    )

    # last_resort parity: the last candidate in both orderings must be the
    # last-resort (priority: 0) candidate.
    assert selected[-1]["last_resort"] is True
    assert CODEX_AAWM_LOW_CANDIDATES[-1].get("last_resort") is True


def test_shadow_parity_promo_window_open_outranks_everything() -> None:
    """Sanity check on the documented tolerance: inside the promo window, the
    promo candidate legitimately outranks the CODEX_AAWM_LOW_CANDIDATES-parity
    set (this is intentional product behavior, not a parity break)."""
    snapshot = _compile_read_yaml()
    lpe.set_active_routing_snapshot(snapshot)

    now_within_promo_window = dt.datetime(2026, 7, 15, tzinfo=dt.timezone.utc)
    selected = lpe._select_read_pilot_snapshot_candidates(
        client_product_label=None,
        now_utc=now_within_promo_window,
    )
    assert selected[0]["model"] == "alibaba_token_plan/qwen3.8-max-preview"
    assert selected[0]["provider"] == "alibaba_token_plan"
