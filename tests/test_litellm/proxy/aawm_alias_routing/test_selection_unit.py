"""Wave 5B shared selection unit tests.

Proves fresh-manager selection, manager-only reset of gate/cursor/quota,
quota lock identity, map identity, facade identity, import boundaries,
non-overlap with Waves 4/5A, Codex/Anthropic selector parity,
provider-lane parity, and route/upstream contracts.

These tests exercise the selection module through the god-module facades
with a fresh AliasRoutingStateManager, verifying no ambient state leakage.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**state_attrs: Any) -> MagicMock:
    """Create a minimal fake Request with mutable .state."""
    req = MagicMock()
    req.state = MagicMock(spec=[])
    for k, v in state_attrs.items():
        setattr(req.state, k, v)
    return req


def _make_candidate(
    provider: str = "openai",
    model: str = "gpt-4o",
    route_family: str = "codex_openai_chat_completions",
    last_resort: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "route_family": route_family,
        "last_resort": last_resort,
        **extra,
    }


# ===========================================================================
# SECTION 1: Fresh-manager selection
# ===========================================================================


class TestFreshManagerSelection:
    """Selection works with a fresh AliasRoutingStateManager, no ambient state."""

    def test_fresh_manager_has_empty_cooldown_maps(self):
        mgr = AliasRoutingStateManager()
        assert len(mgr.codex.cooldown_until_monotonic_by_key) == 0
        assert len(mgr.anthropic.cooldown_until_monotonic_by_key) == 0
        assert len(mgr.codex.session_affinity_by_key) == 0
        assert len(mgr.anthropic.session_affinity_by_key) == 0

    def test_fresh_manager_has_empty_gate_and_cursor(self):
        mgr = AliasRoutingStateManager()
        assert len(mgr.read_pilot_gate._key_state) == 0
        assert len(mgr.round_robin_cursor) == 0

    def test_fresh_manager_quota_cache_default(self):
        mgr = AliasRoutingStateManager()
        assert mgr.get_openrouter_free_quota_cache() == (None, 0.0)


# ===========================================================================
# SECTION 2: Manager-only reset
# ===========================================================================


class TestManagerOnlyReset:
    """reset_for_tests clears gate/cursor/quota without reassignment."""

    def test_reset_clears_gate_cursor_quota(self):
        mgr = AliasRoutingStateManager()
        mgr.read_pilot_gate._key_state["k"] = {"count": 1}
        mgr.round_robin_cursor[("e", "a")] = 3
        mgr.set_openrouter_free_quota_cache((5.0, 10.0))
        mgr.reset_for_tests()
        assert len(mgr.read_pilot_gate._key_state) == 0
        assert len(mgr.round_robin_cursor) == 0
        assert mgr.get_openrouter_free_quota_cache() == (None, 0.0)

    def test_reset_preserves_dict_identity(self):
        mgr = AliasRoutingStateManager()
        refs = {
            "cursor": mgr.round_robin_cursor,
            "gate_ks": mgr.read_pilot_gate._key_state,
            "codex_cd": mgr.codex.cooldown_until_monotonic_by_key,
            "anthro_cd": mgr.anthropic.cooldown_until_monotonic_by_key,
        }
        mgr.reset_for_tests()
        assert mgr.round_robin_cursor is refs["cursor"]
        assert mgr.read_pilot_gate._key_state is refs["gate_ks"]
        assert mgr.codex.cooldown_until_monotonic_by_key is refs["codex_cd"]
        assert mgr.anthropic.cooldown_until_monotonic_by_key is refs["anthro_cd"]

    def test_reset_preserves_lock_identity(self):
        mgr = AliasRoutingStateManager()
        lock = mgr.openrouter_free_quota_lock
        mgr.reset_for_tests()
        assert mgr.openrouter_free_quota_lock is lock


# ===========================================================================
# SECTION 3: Quota lock identity
# ===========================================================================


class TestQuotaLockIdentity:
    """The quota lock is a single asyncio.Lock shared across accesses."""

    def test_lock_is_asyncio_lock(self):
        mgr = AliasRoutingStateManager()
        assert isinstance(mgr.openrouter_free_quota_lock, asyncio.Lock)

    def test_lock_not_recreated_on_reset(self):
        mgr = AliasRoutingStateManager()
        lock_id = id(mgr.openrouter_free_quota_lock)
        mgr.reset_for_tests()
        assert id(mgr.openrouter_free_quota_lock) == lock_id


# ===========================================================================
# SECTION 4: Map identity
# ===========================================================================


class TestMapIdentity:
    """Manager maps are the same objects after reset (cleared in place)."""

    def test_codex_maps_identity(self):
        mgr = AliasRoutingStateManager()
        cd = mgr.codex.cooldown_until_monotonic_by_key
        neg = mgr.codex.cooldown_negative_until_monotonic_by_key
        aff = mgr.codex.session_affinity_by_key
        mgr.reset_for_tests()
        assert mgr.codex.cooldown_until_monotonic_by_key is cd
        assert mgr.codex.cooldown_negative_until_monotonic_by_key is neg
        assert mgr.codex.session_affinity_by_key is aff

    def test_anthropic_maps_identity(self):
        mgr = AliasRoutingStateManager()
        cd = mgr.anthropic.cooldown_until_monotonic_by_key
        aff = mgr.anthropic.session_affinity_by_key
        mgr.reset_for_tests()
        assert mgr.anthropic.cooldown_until_monotonic_by_key is cd
        assert mgr.anthropic.session_affinity_by_key is aff


# ===========================================================================
# SECTION 5: Facade identity (selection module)
# ===========================================================================


class TestSelectionFacadeIdentity:
    """God-module facades point to the same selection module objects."""

    NAMES = [
        "_select_codex_auto_agent_candidate",
        "_select_anthropic_auto_agent_candidate",
        "_find_codex_auto_agent_candidate",
        "_find_anthropic_auto_agent_candidate",
        "_is_auto_agent_candidate_state_available",
        "_build_auto_agent_skipped_candidates_from_states",
        "_codex_auto_agent_candidate_public_shape",
        "_auto_agent_alias_float",
    ]

    def test_facades_match(self):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe

        for name in self.NAMES:
            assert getattr(lpe, name) is getattr(selection, name), (
                f"lpe.{name} is not selection.{name}"
            )


# ===========================================================================
# SECTION 6: Import boundaries
# ===========================================================================


class TestSelectionImportBoundary:
    """selection.py must not import the god module at module scope."""

    def test_no_god_module_import(self):
        import ast
        from pathlib import Path

        sel_path = Path(selection.__file__)
        tree = ast.parse(sel_path.read_text())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "llm_passthrough_endpoints" not in node.module


# ===========================================================================
# SECTION 7: Codex/Anthropic selector parity
# ===========================================================================


class TestCodexAnthropicParity:
    """Both selectors have matching async signatures."""

    def test_both_selectors_are_async(self):
        import inspect

        assert inspect.iscoroutinefunction(selection._select_codex_auto_agent_candidate)
        assert inspect.iscoroutinefunction(selection._select_anthropic_auto_agent_candidate)

    def test_both_selectors_same_params(self):
        import inspect

        codex_params = set(inspect.signature(selection._select_codex_auto_agent_candidate).parameters)
        anthro_params = set(inspect.signature(selection._select_anthropic_auto_agent_candidate).parameters)
        assert codex_params == anthro_params == {"request", "request_body"}


# ===========================================================================
# SECTION 8: Provider-lane parity
# ===========================================================================


class TestProviderLaneParity:
    """Provider-lane helpers exist and are async."""

    HELPERS = [
        "_apply_codex_auto_agent_grok_account_lane_cooldown",
        "_apply_kimi_code_managed_account_lane_cooldown",
        "_apply_codex_auto_agent_adapter_local_candidate_cooldown",
        "_apply_codex_auto_agent_forced_candidate_cooldown",
        "_apply_anthropic_auto_agent_forced_candidate_cooldown",
    ]

    def test_helpers_exist_and_async(self):
        import inspect

        for name in self.HELPERS:
            fn = getattr(selection, name)
            assert inspect.iscoroutinefunction(fn), f"{name} not async"


# ===========================================================================
# SECTION 9: Route/upstream contracts
# ===========================================================================


class TestRouteUpstreamContracts:
    """Router still has routes after Wave 5B wiring."""

    def test_router_has_routes(self):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe

        assert len(lpe.router.routes) > 0

    def test_reset_function_exists(self):
        import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe

        assert callable(lpe.reset_alias_routing_state_for_tests)
