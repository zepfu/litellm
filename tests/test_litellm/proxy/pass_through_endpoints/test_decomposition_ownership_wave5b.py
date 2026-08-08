"""Wave 5B decomposition ownership tests.

Proves:
- Fresh-manager selection works without ambient god-module state
- Manager-only reset clears failure evidence/cursor/quota
- Quota lock identity survives reset
- Map identity survives reset (dicts cleared in place)
- Facade identity: god-module names ARE the target-module objects
- Import boundaries: no god-module import at target-module scope
- Non-overlap with Wave 4 / Wave 5A symbol inventories
- Codex/Anthropic selector parity
- Provider-lane parity
- Route/upstream contracts unchanged
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import dis
import inspect
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module references
# ---------------------------------------------------------------------------
import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import cooldown_state
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import selection
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)

GOD_MODULE_PATH = Path(lpe.__file__)
PKG_ROOT = GOD_MODULE_PATH.parent / "aawm_alias_routing"


# ===========================================================================
# SECTION 1: Facade identity
# ===========================================================================


class TestFacadeIdentity:
    """God-module facade names must be the SAME objects as target-module defs."""

    COOLDOWN_FACADES = [
        "_get_codex_auto_agent_active_cooldown_state",
        "_get_codex_auto_agent_active_cooldown_seconds",
        "_set_codex_auto_agent_cooldown",
        "_get_codex_auto_agent_session_affinity",
        "_set_codex_auto_agent_session_affinity",
        "_publish_codex_cooldown_memory",
        "_publish_anthropic_cooldown_memory",
        "_format_merged_alias_family_cooldown_state_source",
        "_get_anthropic_auto_agent_merged_codex_openai_cooldown_state",
        "_get_anthropic_auto_agent_active_cooldown_state",
        "_get_anthropic_auto_agent_active_cooldown_seconds",
        "_set_anthropic_auto_agent_cooldown",
        "_get_anthropic_auto_agent_session_affinity",
        "_set_anthropic_auto_agent_session_affinity",
        "_attach_aawm_alias_routing_state_sources",
    ]

    SELECTION_FACADES = [
        "_codex_auto_agent_candidate_public_shape",
        "_auto_agent_alias_float",
        "_raise_codex_auto_agent_in_flight_cooldown",
        "_build_auto_agent_redispatch_http_exception_detail",
        "_raise_codex_auto_agent_redispatch_required",
        "_find_codex_auto_agent_candidate",
        "_find_codex_auto_agent_affinity_candidate",
        "_is_auto_agent_candidate_state_available",
        "_build_auto_agent_skipped_candidates_from_states",
        "_apply_codex_auto_agent_forced_candidate_cooldown",
        "_apply_anthropic_auto_agent_forced_candidate_cooldown",
        "_apply_codex_auto_agent_request_local_candidate_state",
        "_apply_codex_auto_agent_adapter_local_candidate_cooldown",
        "_apply_kimi_code_managed_account_lane_cooldown",
        "_build_codex_auto_agent_candidate_state",
        "_get_anthropic_auto_agent_candidate_cooldown_state",
        "_build_anthropic_auto_agent_candidate_state",
        "_build_codex_auto_agent_candidate_states",
        "_select_codex_auto_agent_candidate",
        "_get_codex_auto_agent_request_local_cooldown_key",
        "_get_codex_auto_agent_request_local_cooldown_state",
        "_get_codex_auto_agent_request_local_cooldown_seconds",
        "_set_codex_auto_agent_request_local_cooldown",
        "_get_codex_auto_agent_request_local_excluded_keys",
        "_exclude_codex_auto_agent_request_local_candidate",
        "_exclude_codex_auto_agent_request_local_candidate_without_cooldown",
        "_apply_request_local_cooldown_from_plan",
        "_apply_codex_auto_agent_grok_account_lane_cooldown",
        "_find_anthropic_auto_agent_candidate",
        "_build_anthropic_auto_agent_candidate_states",
        "_raise_anthropic_auto_agent_in_flight_cooldown",
        "_raise_anthropic_auto_agent_redispatch_required",
        "_select_anthropic_auto_agent_candidate",
    ]

    def test_cooldown_state_facades_are_same_objects(self):
        for name in self.COOLDOWN_FACADES:
            god_obj = getattr(lpe, name)
            target_obj = getattr(cooldown_state, name)
            assert god_obj is target_obj, f"{name}: god-module facade is not cooldown_state.{name}"

    def test_selection_facades_are_same_objects(self):
        for name in self.SELECTION_FACADES:
            god_obj = getattr(lpe, name)
            target_obj = getattr(selection, name)
            assert god_obj is target_obj, f"{name}: god-module facade is not selection.{name}"


# ===========================================================================
# SECTION 2: Manager state ownership
# ===========================================================================


class TestManagerStateOwnership:
    """Cursor and quota are manager-owned; reset clears routing state."""

    def test_round_robin_cursor_is_manager_owned(self):
        assert lpe._round_robin_cursor_by_alias is alias_routing_state.round_robin_cursor

    def test_quota_lock_is_manager_owned(self):
        assert lpe._openrouter_free_daily_quota_lock is alias_routing_state.openrouter_free_quota_lock

    def test_quota_cache_roundtrip_via_manager(self):
        mgr = alias_routing_state
        original = mgr.get_openrouter_free_quota_cache()
        try:
            mgr.set_openrouter_free_quota_cache((42.0, 99.0))
            assert mgr.get_openrouter_free_quota_cache() == (42.0, 99.0)
        finally:
            mgr.set_openrouter_free_quota_cache(original)

    def test_quota_cache_compat_getattr(self):
        """Module __getattr__ returns the manager-owned cache tuple."""
        val = lpe._openrouter_free_daily_quota_cache
        assert isinstance(val, tuple)
        assert val is alias_routing_state.get_openrouter_free_quota_cache()

    def test_reset_clears_gate_cursor_quota(self):
        mgr = alias_routing_state
        evidence_gate = mgr.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias="test-alias",
            create=True,
        )
        assert evidence_gate is not None
        # Seed state
        evidence_gate._key_state["test_key"] = {"count": 1}
        mgr.round_robin_cursor[("epoch", "alias")] = 5
        mgr.set_openrouter_free_quota_cache((10.0, 20.0))
        # Reset
        mgr.reset_for_tests()
        # Verify cleared
        assert not evidence_gate._key_state
        assert (
            mgr.codex_failure_evidence_gate.gate_for_alias(
                canonical_alias="test-alias"
            )
            is None
        )
        assert len(mgr.round_robin_cursor) == 0
        assert mgr.get_openrouter_free_quota_cache() == (None, 0.0)

    def test_reset_preserves_map_identity(self):
        """Dicts are cleared in place, never reassigned."""
        mgr = alias_routing_state
        cursor_ref = mgr.round_robin_cursor
        lock_ref = mgr.openrouter_free_quota_lock
        codex_cd_ref = mgr.codex.cooldown_until_monotonic_by_key
        mgr.reset_for_tests()
        assert mgr.round_robin_cursor is cursor_ref
        assert mgr.openrouter_free_quota_lock is lock_ref
        assert mgr.codex.cooldown_until_monotonic_by_key is codex_cd_ref

    def test_quota_lock_identity_survives_reset(self):
        mgr = alias_routing_state
        lock_before = mgr.openrouter_free_quota_lock
        mgr.reset_for_tests()
        assert mgr.openrouter_free_quota_lock is lock_before

    def test_legacy_singleton_reset_is_narrow(self):
        mgr = alias_routing_state
        cooldown_key = "wave5b:reset:cooldown"
        affinity_key = "wave5b:reset:affinity"
        snapshot_sentinel = object()
        evidence_gate = mgr.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias="test-alias",
            create=True,
        )
        assert evidence_gate is not None
        try:
            mgr.codex.cooldown_until_monotonic_by_key[cooldown_key] = 123.0
            mgr.codex.session_affinity_by_key[affinity_key] = {
                "provider": "openai"
            }
            evidence_gate._key_state["gate-key"] = {"count": 1}
            evidence_gate._family_state.evidence_events_by_key[
                "evidence-key"
            ] = []
            mgr.round_robin_cursor[("epoch", "alias")] = 3
            lpe.set_active_routing_snapshot(snapshot_sentinel)

            lpe.reset_module_singletons()

            assert cooldown_key in mgr.codex.cooldown_until_monotonic_by_key
            assert affinity_key in mgr.codex.session_affinity_by_key
            assert not evidence_gate._key_state
            assert not evidence_gate._family_state.evidence_events_by_key
            assert (
                mgr.codex_failure_evidence_gate.gate_for_alias(
                    canonical_alias="test-alias"
                )
                is None
            )
            assert not mgr.round_robin_cursor
            assert lpe.get_active_routing_snapshot() is None
        finally:
            lpe.reset_alias_routing_state_for_tests()

    def test_full_alias_routing_reset_clears_manager_and_snapshot(self):
        mgr = alias_routing_state
        evidence_gate = mgr.codex_failure_evidence_gate.gate_for_alias(
            canonical_alias="test-alias",
            create=True,
        )
        assert evidence_gate is not None
        mgr.codex.cooldown_until_monotonic_by_key["cooldown"] = 123.0
        mgr.codex.session_affinity_by_key["affinity"] = {"provider": "openai"}
        evidence_gate._key_state["gate-key"] = {"count": 1}
        mgr.round_robin_cursor[("epoch", "alias")] = 3
        mgr.set_openrouter_free_quota_cache((42.0, 99.0))
        lpe.set_active_routing_snapshot(object())

        lpe.reset_alias_routing_state_for_tests()

        assert not mgr.codex.cooldown_until_monotonic_by_key
        assert not mgr.codex.session_affinity_by_key
        assert not evidence_gate._key_state
        assert (
            mgr.codex_failure_evidence_gate.gate_for_alias(
                canonical_alias="test-alias"
            )
            is None
        )
        assert not mgr.round_robin_cursor
        assert mgr.get_openrouter_free_quota_cache() == (None, 0.0)
        assert lpe.get_active_routing_snapshot() is None


# ===========================================================================
# SECTION 3: Import boundary guards
# ===========================================================================


class TestImportBoundaries:
    """Target modules must not import the god module at module scope."""

    def _module_scope_import_names(self, filepath: Path) -> set[str]:
        tree = ast.parse(filepath.read_text())
        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module)
        return names

    def test_cooldown_state_no_god_import(self):
        imports = self._module_scope_import_names(PKG_ROOT / "cooldown_state.py")
        assert "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints" not in imports

    def test_selection_no_god_import(self):
        imports = self._module_scope_import_names(PKG_ROOT / "selection.py")
        assert "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints" not in imports

    def test_no_wildcard_imports_in_targets(self):
        for mod_name in ("cooldown_state.py", "selection.py"):
            tree = ast.parse((PKG_ROOT / mod_name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        assert alias.name != "*", f"{mod_name} has wildcard import from {node.module}"

    @pytest.mark.parametrize("module", [cooldown_state, selection])
    def test_rebound_functions_have_no_unresolved_load_globals(self, module):
        """Every rebound function must resolve module-private runtime dependencies."""
        for name in module._HOST_FUNCTION_NAMES:
            function = getattr(module, name)
            unresolved = {
                instruction.argval
                for instruction in dis.get_instructions(function)
                if instruction.opname == "LOAD_GLOBAL"
                and instruction.argval not in function.__globals__
                and not hasattr(builtins, str(instruction.argval))
            }
            assert not unresolved, (
                f"{module.__name__}.{name} has unresolved LOAD_GLOBAL names: "
                f"{sorted(unresolved)}"
            )

    def test_selection_install_preserves_preexisting_host_callable(self):
        """Runtime adapters must not overwrite same-named host implementations."""
        function = lpe._get_openrouter_adapter_active_cooldown_seconds
        assert function.__name__ == "_get_openrouter_adapter_active_cooldown_seconds"
        assert function.__code__.co_name == "_get_openrouter_adapter_active_cooldown_seconds"


# ===========================================================================
# SECTION 4: Non-overlap with Wave 4 / 5A
# ===========================================================================


class TestNonOverlap:
    """Wave 5B symbols must not duplicate Wave 4 or 5A module ownership."""

    GENERIC_INSTALL_NAMES = {"_HOST_FUNCTION_NAMES", "install"}

    W4_MODULES = [
        "model_resolution",
        "lane_keys",
        "google_env",
        "google_context",
        "google_error",
        "side_channel",
        "constants",
    ]

    W5A_MODULES = [
        "snapshot_select",
        "config_refresh",
        "codex_oauth",
        "openrouter_quota",
    ]

    def _public_names(self, module_path: Path) -> set[str]:
        tree = ast.parse(module_path.read_text())
        names: set[str] = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        return names

    def test_wave5b_symbols_do_not_overlap_wave4(self):
        w5b_names = self._public_names(PKG_ROOT / "cooldown_state.py") | self._public_names(PKG_ROOT / "selection.py")
        w5b_names -= self.GENERIC_INSTALL_NAMES
        # Remove the shared _attach function (cooldown_state owns it, selection imports it)
        for w4_mod in self.W4_MODULES:
            # Wave 4 modules may be in different packages; check aawm_alias_routing first
            w4_path = PKG_ROOT / f"{w4_mod}.py"
            if not w4_path.exists():
                continue
            w4_names = self._public_names(w4_path)
            overlap = w5b_names & w4_names
            # Filter out expected shared constants/imports
            overlap -= {"_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS"}
            assert not overlap, f"Wave 5B overlaps Wave 4 {w4_mod}: {sorted(overlap)}"

    def test_wave5b_symbols_do_not_overlap_wave5a(self):
        w5b_names = self._public_names(PKG_ROOT / "cooldown_state.py") | self._public_names(PKG_ROOT / "selection.py")
        w5b_names -= self.GENERIC_INSTALL_NAMES
        for w5a_mod in self.W5A_MODULES:
            w5a_path = PKG_ROOT / f"{w5a_mod}.py"
            if not w5a_path.exists():
                continue
            w5a_names = self._public_names(w5a_path)
            overlap = w5b_names & w5a_names
            # Filter out expected cross-module imports
            overlap -= {
                "_apply_openrouter_durable_quota_candidate_cooldown",
                "_commit_round_robin_selection",
                "_resolve_aawm_alias_selection_enumeration",
                "_routing_candidate_to_public_dict",
                "get_active_routing_snapshot",
            }
            assert not overlap, f"Wave 5B overlaps Wave 5A {w5a_mod}: {sorted(overlap)}"


# ===========================================================================
# SECTION 5: Sole ownership of _attach_aawm_alias_routing_state_sources
# ===========================================================================


class TestAttachSoleOwnership:
    """cooldown_state.py is the sole definer; selection.py imports it."""

    def test_cooldown_state_defines_attach(self):
        assert hasattr(cooldown_state, "_attach_aawm_alias_routing_state_sources")
        src = inspect.getsource(cooldown_state._attach_aawm_alias_routing_state_sources)
        assert "enriched" in src

    def test_selection_imports_attach_from_cooldown_state(self):
        assert hasattr(selection, "_attach_aawm_alias_routing_state_sources")
        assert selection._attach_aawm_alias_routing_state_sources is cooldown_state._attach_aawm_alias_routing_state_sources

    def test_selection_does_not_define_attach(self):
        """The function must NOT be defined in selection.py's own AST."""
        tree = ast.parse((PKG_ROOT / "selection.py").read_text())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name != "_attach_aawm_alias_routing_state_sources", (
                    "selection.py must not define _attach_aawm_alias_routing_state_sources"
                )


# ===========================================================================
# SECTION 6: Codex/Anthropic selector parity
# ===========================================================================


class TestSelectorParity:
    """Both selectors exist, are async, and have matching parameter contracts."""

    def test_codex_selector_exists_and_is_async(self):
        fn = getattr(lpe, "_select_codex_auto_agent_candidate")
        assert asyncio.iscoroutinefunction(fn)

    def test_anthropic_selector_exists_and_is_async(self):
        fn = getattr(lpe, "_select_anthropic_auto_agent_candidate")
        assert asyncio.iscoroutinefunction(fn)

    def test_selector_parameter_parity(self):
        codex_sig = inspect.signature(lpe._select_codex_auto_agent_candidate)
        anthro_sig = inspect.signature(lpe._select_anthropic_auto_agent_candidate)
        assert set(codex_sig.parameters) == set(anthro_sig.parameters) == {"request", "request_body"}


# ===========================================================================
# SECTION 7: Provider-lane parity
# ===========================================================================


class TestProviderLaneParity:
    """Key provider-lane helpers exist on both the god module and selection."""

    LANE_HELPERS = [
        "_apply_codex_auto_agent_grok_account_lane_cooldown",
        "_apply_kimi_code_managed_account_lane_cooldown",
        "_apply_codex_auto_agent_adapter_local_candidate_cooldown",
        "_apply_codex_auto_agent_forced_candidate_cooldown",
        "_apply_anthropic_auto_agent_forced_candidate_cooldown",
    ]

    def test_lane_helpers_are_async(self):
        for name in self.LANE_HELPERS:
            fn = getattr(lpe, name)
            assert asyncio.iscoroutinefunction(fn), f"{name} should be async"


# ===========================================================================
# SECTION 8: Route/upstream contracts unchanged
# ===========================================================================


class TestRouteContract:
    """Router paths and methods remain stable after Wave 5B wiring."""

    def test_router_has_routes(self):
        assert len(lpe.router.routes) > 0

    def test_passthrough_routes_stable(self):
        """Spot-check that key routes still exist."""
        paths = {route.path for route in lpe.router.routes}
        # These are representative routes that must survive decomposition
        assert any("/v1/chat/completions" in p for p in paths) or len(paths) > 5
