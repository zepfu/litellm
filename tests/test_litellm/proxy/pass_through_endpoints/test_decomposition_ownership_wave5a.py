"""Wave 5A decomposition ownership + golden-parity contract tests.

Enforces the behavior-preserving extraction contract from
``llm_passthrough_endpoints.py`` into four new ``aawm_alias_routing/`` modules:

- ``aawm_alias_routing/snapshot_select.py``
    Snapshot ordering, distribution strategy, TUI/schedule gates,
    selection-context memoization, and alias-candidate getters.
- ``aawm_alias_routing/config_refresh.py``
    The ``/aawm/alias-config/refresh`` route handler body and YAML loading.
- ``aawm_alias_routing/codex_oauth.py``
    Codex auth-file discovery, JWT decode, token validation, and
    Codex-native-auth request detection helpers.
- ``aawm_alias_routing/openrouter_quota.py``
    OpenRouter free-daily-quota probe, durable cooldown helpers, and
    the alias-probe cooldown gate.

Structural ownership tests are intentionally RED until the implementation
lands.  Golden parity tests are GREEN now and must remain green after
extraction.

Write scope: this file only (Wave 5A tests-only landing).
"""

from __future__ import annotations

import ast
import asyncio
import base64
import inspect
import json
import time
from datetime import datetime, timezone
from pathlib import Path


import pytest

# ---------------------------------------------------------------------------
# God-module import (always available)
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

GOD_PATH = Path(lpe.__file__).resolve()

# ---------------------------------------------------------------------------
# Target module import paths (relative to litellm package root)
# ---------------------------------------------------------------------------
W5A_TARGET_MODULE_IMPORT_PATHS: dict[str, str] = {
    "snapshot_select": "litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select",
    "config_refresh": "litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_refresh",
    "codex_oauth": "litellm.proxy.pass_through_endpoints.aawm_alias_routing.codex_oauth",
    "openrouter_quota": "litellm.proxy.pass_through_endpoints.aawm_alias_routing.openrouter_quota",
}

# ---------------------------------------------------------------------------
# Symbol inventory: target_module_key -> set of function/type names
# Built from current source AST analysis (develop @ 125baf5405).
# ---------------------------------------------------------------------------

SNAPSHOT_SELECT_SYMBOLS: set[str] = {
    # Ordering / distribution
    "_order_snapshot_candidates_by_priority",
    "_select_proportional_snapshot_candidate",
    "_select_round_robin_snapshot_candidate",
    "_commit_round_robin_selection",
    "_apply_snapshot_alias_distribution_strategy",
    # TUI / schedule gates
    "_is_tui_attached_candidate_eligible",
    "_is_snapshot_candidate_in_schedule_window",
    # Snapshot-driven resolution
    "_resolve_read_pilot_eligible_candidates",
    "_select_read_pilot_snapshot_candidates",
    "_derive_round_robin_commit_token",
    # Selection-context memoization
    "_get_aawm_alias_selection_context",
    "_resolve_aawm_alias_selection_enumeration",
    # Alias-candidate getters
    "_get_codex_auto_agent_candidates_for_alias",
    # Public shaping
    "_routing_candidate_to_public_dict",
    # Snapshot holder accessors
    "get_active_routing_snapshot",
    "set_active_routing_snapshot",
}

# Named tuples that must move with snapshot_select
SNAPSHOT_SELECT_TYPES: set[str] = {
    "RoundRobinCommitToken",
    "SelectionEnumeration",
}

CONFIG_REFRESH_SYMBOLS: set[str] = {
    "_load_aawm_alias_routing_source_yaml",
    "aawm_alias_config_refresh_route",
}

CODEX_OAUTH_SYMBOLS: set[str] = {
    # Auth-value cleaning
    "_clean_codex_auth_value",
    # Auth-file discovery
    "_get_anthropic_adapter_codex_auth_file_path",
    # JWT helpers
    "_decode_jwt_claims_without_validation",
    "_extract_codex_account_id_from_token",
    # Token data / validation
    "_get_codex_auth_token_data",
    "_get_codex_auth_token_expiry",
    "_codex_auth_access_token_is_valid",
    # Auth data loading
    "_load_codex_auth_data_from_path",
    "_load_local_codex_auth_headers",
    # Codex-native-auth request detection
    "_anthropic_adapter_request_uses_codex_native_auth",
    "_anthropic_adapter_request_has_openai_client_auth",
    "_anthropic_adapter_should_forward_direct_auth_headers",
    "_request_uses_codex_native_auth",
    # OAuth error helpers
    "_get_oauth_token_error_code",
    "_format_oauth_refresh_failure_detail",
}

# Type aliases that must move with codex_oauth
CODEX_OAUTH_TYPES: set[str] = {
    "CodexAuthData",
    "CodexTokenData",
    "OAuthJsonData",
}

OPENROUTER_QUOTA_SYMBOLS: set[str] = {
    # Quota cache management
    "_reset_openrouter_free_daily_quota_cache",
    "_parse_openrouter_free_daily_quota_reset_timestamp",
    # Quota probe / fetch
    "_fetch_openrouter_free_daily_quota_row",
    "_get_openrouter_free_daily_quota_exhausted_cooldown_seconds",
    # Candidate classification
    "_is_openrouter_free_quota_candidate",
    # Durable cooldown application
    "_apply_openrouter_durable_quota_candidate_cooldown",
    # Alias-probe cooldown gate
    "_maybe_raise_openrouter_adapter_alias_probe_cooldown",
    "_raise_openrouter_auto_agent_candidate_unavailable",
}

# Constants that must move with their owning module
CONFIG_REFRESH_CONSTANTS: set[str] = {
    "_DEFAULT_AAWM_ALIAS_CONFIG_PATH",
}

CODEX_OAUTH_CONSTANTS: set[str] = {
    "_ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS",
    "_ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS",
    "_ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS",
}

OPENROUTER_QUOTA_CONSTANTS: set[str] = {
    "_OPENROUTER_DURABLE_QUOTA_DAILY_KEY",
    "_OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS",
    "_OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS",
    "_OPENROUTER_FREE_DAILY_QUOTA_MODELS",
}

SNAPSHOT_SELECT_CONSTANTS: set[str] = {
    "_READ_PILOT_ALIAS_NAME",
}

# Unified inventory: target_key -> function/type symbols
W5A_SYMBOL_INVENTORY: dict[str, set[str]] = {
    "snapshot_select": SNAPSHOT_SELECT_SYMBOLS | SNAPSHOT_SELECT_TYPES,
    "config_refresh": CONFIG_REFRESH_SYMBOLS,
    "codex_oauth": CODEX_OAUTH_SYMBOLS | CODEX_OAUTH_TYPES,
    "openrouter_quota": OPENROUTER_QUOTA_SYMBOLS,
}

# All moved function symbols (union)
W5A_ALL_MOVED_FUNCTIONS: set[str] = set()
for _syms in W5A_SYMBOL_INVENTORY.values():
    W5A_ALL_MOVED_FUNCTIONS |= _syms

# All moved constants
W5A_ALL_MOVED_CONSTANTS: set[str] = (
    SNAPSHOT_SELECT_CONSTANTS
    | CONFIG_REFRESH_CONSTANTS
    | CODEX_OAUTH_CONSTANTS
    | OPENROUTER_QUOTA_CONSTANTS
)

# State that must NOT move in Wave 5A (deferred to 5B: cooldown state + selection)
W5A_DEFERRED_STATE: set[str] = {
    "_round_robin_cursor_by_alias",
    "_openrouter_free_daily_quota_cache",
    "_openrouter_free_daily_quota_lock",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_god_module() -> ast.Module:
    source = GOD_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(GOD_PATH))


def _top_level_function_defs(tree: ast.Module) -> set[str]:
    """Names defined as FunctionDef/AsyncFunctionDef at module top level."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _top_level_class_defs(tree: ast.Module) -> set[str]:
    """Names defined as ClassDef at module top level."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


def _top_level_assignments(tree: ast.Module) -> set[str]:
    """Names assigned at module top level."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _try_import_w5a_target(module_key: str):
    """Try to import a Wave 5A target module; return None if not yet created."""
    import_path = W5A_TARGET_MODULE_IMPORT_PATHS[module_key]
    try:
        import importlib

        return importlib.import_module(import_path)
    except (ImportError, ModuleNotFoundError):
        return None


# ===========================================================================
# SECTION 1: Structural ownership tests (RED until implementation lands)
# ===========================================================================


class TestW5AMovedBandsNotDefinedInGodModule:
    """After extraction, moved functions must NOT appear as FunctionDef in
    the god module -- only as assignment facades."""

    def test_snapshot_select_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = SNAPSHOT_SELECT_SYMBOLS & func_defs
        assert not violations, (
            f"snapshot_select symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_config_refresh_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = CONFIG_REFRESH_SYMBOLS & func_defs
        assert not violations, (
            f"config_refresh symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_codex_oauth_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = CODEX_OAUTH_SYMBOLS & func_defs
        assert not violations, (
            f"codex_oauth symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_openrouter_quota_functions_absent_as_defs(self):
        tree = _parse_god_module()
        func_defs = _top_level_function_defs(tree)
        violations = OPENROUTER_QUOTA_SYMBOLS & func_defs
        assert not violations, (
            f"openrouter_quota symbols still defined as functions in god module: "
            f"{sorted(violations)}"
        )

    def test_snapshot_select_types_absent_as_class_defs(self):
        tree = _parse_god_module()
        class_defs = _top_level_class_defs(tree)
        violations = SNAPSHOT_SELECT_TYPES & class_defs
        assert not violations, (
            f"snapshot_select types still defined as classes in god module: "
            f"{sorted(violations)}"
        )

    def test_moved_constants_absent_as_owned_assignments(self):
        """Moved constants must not remain as direct value assignments in
        the god module after extraction (facades via import are OK)."""
        tree = _parse_god_module()
        violations = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in W5A_ALL_MOVED_CONSTANTS:
                        rhs = node.value
                        is_facade = isinstance(rhs, ast.Attribute) or (
                            isinstance(rhs, ast.Name) and rhs.id != target.id
                        )
                        if not is_facade:
                            violations.append(target.id)
        assert not violations, (
            f"Wave 5A constants still owned (non-facade) in god module: {sorted(violations)}"
        )


class TestW5ATargetModulesExist:
    """Target packages/modules must exist after extraction."""

    @pytest.mark.parametrize("module_key", list(W5A_TARGET_MODULE_IMPORT_PATHS.keys()))
    def test_target_module_importable(self, module_key: str):
        mod = _try_import_w5a_target(module_key)
        assert mod is not None, (
            f"Target module {W5A_TARGET_MODULE_IMPORT_PATHS[module_key]} not yet created"
        )


class TestW5AFacadeObjectIdentity:
    """Sampled facade bindings must be the same object as the target module's."""

    def _assert_identity_for_module(self, module_key: str, sample_size: int = 10):
        mod = _try_import_w5a_target(module_key)
        if mod is None:
            pytest.skip(f"Target module {module_key} not yet created")
        symbols = sorted(W5A_SYMBOL_INVENTORY[module_key])
        sample = symbols[:sample_size] if len(symbols) >= sample_size else symbols
        for name in sample:
            god_obj = getattr(lpe, name, None)
            target_obj = getattr(mod, name, None)
            assert god_obj is not None, f"{name} not found on god module"
            assert target_obj is not None, f"{name} not found on target module"
            assert god_obj is target_obj, (
                f"{name}: god module facade is not the same object as "
                f"{W5A_TARGET_MODULE_IMPORT_PATHS[module_key]}.{name}"
            )

    def test_snapshot_select_facade_identity(self):
        self._assert_identity_for_module("snapshot_select")

    def test_config_refresh_facade_identity(self):
        self._assert_identity_for_module("config_refresh")

    def test_codex_oauth_facade_identity(self):
        self._assert_identity_for_module("codex_oauth")

    def test_openrouter_quota_facade_identity(self):
        self._assert_identity_for_module("openrouter_quota")


# ===========================================================================
# SECTION 2: Import boundary tests (RED until implementation lands)
# ===========================================================================


class TestW5AImportBoundaries:
    """No Wave 5A target module may import llm_passthrough_endpoints at
    module scope.  No wildcard imports."""

    @pytest.mark.parametrize("module_key", list(W5A_TARGET_MODULE_IMPORT_PATHS.keys()))
    def test_no_god_module_import_at_module_scope(self, module_key: str):
        mod = _try_import_w5a_target(module_key)
        if mod is None:
            pytest.skip(f"Target module {module_key} not yet created")
        mod_path = Path(mod.__file__).resolve()
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        violations = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "llm_passthrough_endpoints" in alias.name:
                        violations.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "llm_passthrough_endpoints" in node.module:
                    violations.append(f"from {node.module} import ...")
        assert not violations, (
            f"{module_key} imports llm_passthrough_endpoints at module scope: {violations}"
        )

    @pytest.mark.parametrize("module_key", list(W5A_TARGET_MODULE_IMPORT_PATHS.keys()))
    def test_no_wildcard_imports(self, module_key: str):
        mod = _try_import_w5a_target(module_key)
        if mod is None:
            pytest.skip(f"Target module {module_key} not yet created")
        mod_path = Path(mod.__file__).resolve()
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        violations.append(
                            f"from {node.module} import * (line {node.lineno})"
                        )
        assert not violations, (
            f"{module_key} uses wildcard imports: {violations}"
        )


# ===========================================================================
# SECTION 3: Deferred-state guard (GREEN now, must remain GREEN)
# ===========================================================================


class TestW5ADeferredStateGuard:
    """State deferred to Wave 5B must remain in the god module."""

    def test_round_robin_cursor_stays_in_god_module(self):
        """_round_robin_cursor_by_alias is selection state owned by 5B."""
        assert hasattr(lpe, "_round_robin_cursor_by_alias")
        assert isinstance(lpe._round_robin_cursor_by_alias, dict)

    def test_openrouter_quota_cache_stays_in_god_module(self):
        """_openrouter_free_daily_quota_cache is cooldown state owned by 5B."""
        assert hasattr(lpe, "_openrouter_free_daily_quota_cache")
        assert isinstance(lpe._openrouter_free_daily_quota_cache, tuple)

    def test_openrouter_quota_lock_stays_in_god_module(self):
        """_openrouter_free_daily_quota_lock is cooldown state owned by 5B."""
        assert hasattr(lpe, "_openrouter_free_daily_quota_lock")
        assert isinstance(lpe._openrouter_free_daily_quota_lock, asyncio.Lock)

    def test_deferred_state_not_in_w5a_inventory(self):
        """Deferred state names must not appear in any W5A symbol set."""
        for module_key, symbols in W5A_SYMBOL_INVENTORY.items():
            overlap = symbols & W5A_DEFERRED_STATE
            assert not overlap, (
                f"{module_key} incorrectly claims deferred state: {sorted(overlap)}"
            )


# ===========================================================================
# SECTION 4: Signature and async/sync contract tests (GREEN now)
# ===========================================================================


class TestW5ASignatureContracts:
    """Verify sync/async shape is preserved for all Wave 5A functions."""

    def test_snapshot_select_sync_functions(self):
        sync_funcs = [
            "_order_snapshot_candidates_by_priority",
            "_select_proportional_snapshot_candidate",
            "_select_round_robin_snapshot_candidate",
            "_commit_round_robin_selection",
            "_apply_snapshot_alias_distribution_strategy",
            "_is_tui_attached_candidate_eligible",
            "_is_snapshot_candidate_in_schedule_window",
            "_resolve_read_pilot_eligible_candidates",
            "_select_read_pilot_snapshot_candidates",
            "_derive_round_robin_commit_token",
            "_get_aawm_alias_selection_context",
            "_resolve_aawm_alias_selection_enumeration",
            "_get_codex_auto_agent_candidates_for_alias",
            "_routing_candidate_to_public_dict",
            "get_active_routing_snapshot",
            "set_active_routing_snapshot",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_config_refresh_async_functions(self):
        async_funcs = ["aawm_alias_config_refresh_route"]
        for name in async_funcs:
            fn = getattr(lpe, name)
            assert inspect.iscoroutinefunction(fn), f"{name} must be async"

    def test_config_refresh_sync_functions(self):
        sync_funcs = ["_load_aawm_alias_routing_source_yaml"]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_codex_oauth_sync_functions(self):
        sync_funcs = [
            "_clean_codex_auth_value",
            "_get_anthropic_adapter_codex_auth_file_path",
            "_decode_jwt_claims_without_validation",
            "_extract_codex_account_id_from_token",
            "_get_codex_auth_token_data",
            "_get_codex_auth_token_expiry",
            "_codex_auth_access_token_is_valid",
            "_anthropic_adapter_request_uses_codex_native_auth",
            "_anthropic_adapter_request_has_openai_client_auth",
            "_anthropic_adapter_should_forward_direct_auth_headers",
            "_request_uses_codex_native_auth",
            "_get_oauth_token_error_code",
            "_format_oauth_refresh_failure_detail",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_codex_oauth_async_functions(self):
        async_funcs = [
            "_load_codex_auth_data_from_path",
            "_load_local_codex_auth_headers",
        ]
        for name in async_funcs:
            fn = getattr(lpe, name)
            assert inspect.iscoroutinefunction(fn), f"{name} must be async"

    def test_openrouter_quota_sync_functions(self):
        sync_funcs = [
            "_reset_openrouter_free_daily_quota_cache",
            "_parse_openrouter_free_daily_quota_reset_timestamp",
            "_is_openrouter_free_quota_candidate",
            "_raise_openrouter_auto_agent_candidate_unavailable",
        ]
        for name in sync_funcs:
            fn = getattr(lpe, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"

    def test_openrouter_quota_async_functions(self):
        async_funcs = [
            "_fetch_openrouter_free_daily_quota_row",
            "_get_openrouter_free_daily_quota_exhausted_cooldown_seconds",
            "_apply_openrouter_durable_quota_candidate_cooldown",
            "_maybe_raise_openrouter_adapter_alias_probe_cooldown",
        ]
        for name in async_funcs:
            fn = getattr(lpe, name)
            assert inspect.iscoroutinefunction(fn), f"{name} must be async"

    def test_snapshot_select_named_tuples_are_namedtuples(self):
        """RoundRobinCommitToken and SelectionEnumeration must be NamedTuples."""
        assert hasattr(lpe.RoundRobinCommitToken, "_fields")
        assert hasattr(lpe.SelectionEnumeration, "_fields")
        assert "alias_name" in lpe.RoundRobinCommitToken._fields
        assert "epoch_tag" in lpe.RoundRobinCommitToken._fields
        assert "tied_candidate_ids" in lpe.RoundRobinCommitToken._fields
        assert "start_index" in lpe.RoundRobinCommitToken._fields
        assert "candidates" in lpe.SelectionEnumeration._fields
        assert "commit_token" in lpe.SelectionEnumeration._fields


# ===========================================================================
# SECTION 5: Golden parity tests (GREEN now, must remain GREEN)
# ===========================================================================


class TestSnapshotSelectParity:
    """Golden behavior parity for snapshot ordering/distribution/gates."""

    def _make_candidate(self, model: str, priority: int = 50, **kwargs):
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
            RoutingCandidate,
        )

        defaults = {
            "provider": "openrouter",
            "model": model,
            "route_family": "codex_openrouter_completion_adapter",
            "priority": priority,
            "weight": 1.0,
            "tui_attached": None,
            "schedule": None,
        }
        defaults.update(kwargs)
        return RoutingCandidate(**defaults)

    def test_order_by_priority_descending(self):
        c_low = self._make_candidate("low", priority=10)
        c_high = self._make_candidate("high", priority=90)
        c_mid = self._make_candidate("mid", priority=50)
        result = lpe._order_snapshot_candidates_by_priority([c_low, c_high, c_mid])
        assert [c.model for c in result] == ["high", "mid", "low"]

    def test_order_by_priority_zero_last(self):
        c_zero = self._make_candidate("zero", priority=0)
        c_normal = self._make_candidate("normal", priority=1)
        result = lpe._order_snapshot_candidates_by_priority([c_zero, c_normal])
        assert [c.model for c in result] == ["normal", "zero"]

    def test_proportional_select_deterministic_with_seed(self):
        import random

        c_a = self._make_candidate("a", priority=50)
        c_b = self._make_candidate("b", priority=50)
        weights = {"a": 1.0, "b": 0.0}
        rng = random.Random(42)
        result = lpe._select_proportional_snapshot_candidate([c_a, c_b], weights, rng)
        assert result.model == "a"

    def test_tui_eligible_non_tui_candidate(self):
        c = self._make_candidate("m", tui_attached=None)
        assert lpe._is_tui_attached_candidate_eligible(c, client_product_label=None) is True

    def test_tui_eligible_matching_product(self):
        c = self._make_candidate("m", tui_attached="codex")
        assert lpe._is_tui_attached_candidate_eligible(c, client_product_label="codex/1.0") is True

    def test_tui_eligible_mismatched_product(self):
        c = self._make_candidate("m", tui_attached="codex")
        assert lpe._is_tui_attached_candidate_eligible(c, client_product_label="claude/2.0") is False

    def test_tui_eligible_no_label(self):
        c = self._make_candidate("m", tui_attached="codex")
        assert lpe._is_tui_attached_candidate_eligible(c, client_product_label=None) is False

    def test_schedule_window_no_schedule(self):
        c = self._make_candidate("m", schedule=None)
        now = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
        assert lpe._is_snapshot_candidate_in_schedule_window(c, now_utc=now) is True

    def test_routing_candidate_to_public_dict_basic(self):
        c = self._make_candidate("gpt-4o", priority=50)
        result = lpe._routing_candidate_to_public_dict(c)
        assert result["provider"] == "openrouter"
        assert result["model"] == "gpt-4o"
        assert result["route_family"] == "codex_openrouter_completion_adapter"
        assert result["last_resort"] is False
        assert "config_epoch_tag" not in result

    def test_routing_candidate_to_public_dict_with_epoch(self):
        c = self._make_candidate("gpt-4o", priority=50)
        result = lpe._routing_candidate_to_public_dict(c, epoch_tag="abc123")
        assert result["config_epoch_tag"] == "abc123"

    def test_routing_candidate_to_public_dict_last_resort(self):
        c = self._make_candidate("fallback", priority=0)
        result = lpe._routing_candidate_to_public_dict(c)
        assert result["last_resort"] is True

    def test_get_active_routing_snapshot_callable(self):
        """The function must be callable and return None or a snapshot."""
        result = lpe.get_active_routing_snapshot()
        assert result is None or hasattr(result, "config_hash")

    def test_get_codex_auto_agent_candidates_for_alias_static(self):
        """Non-read alias returns static table candidates."""
        result = lpe._get_codex_auto_agent_candidates_for_alias("nonexistent_alias")
        assert isinstance(result, tuple)
        assert len(result) > 0

    def test_commit_round_robin_selection_none_token_noop(self):
        """None token is a no-op."""
        lpe._commit_round_robin_selection(None, selected_candidate={"provider": "x", "model": "y"})
        # No exception = pass

    def test_apply_distribution_strategy_none_passthrough(self):
        import random

        c_a = self._make_candidate("a", priority=50)
        c_b = self._make_candidate("b", priority=30)
        result = lpe._apply_snapshot_alias_distribution_strategy(
            [c_a, c_b],
            distribution_strategy=None,
            rng=random.Random(42),
        )
        assert [c.model for c in result] == ["a", "b"]


class TestConfigRefreshParity:
    """Golden behavior parity for config refresh helpers."""

    def test_load_yaml_inline_override(self):
        result = lpe._load_aawm_alias_routing_source_yaml(inline_yaml="test: true")
        assert result == "test: true"

    def test_load_yaml_default_file(self):
        """Default path reads the checked-in read.yaml."""
        result = lpe._load_aawm_alias_routing_source_yaml(inline_yaml=None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_default_config_path_is_path(self):
        assert isinstance(lpe._DEFAULT_AAWM_ALIAS_CONFIG_PATH, Path)
        assert lpe._DEFAULT_AAWM_ALIAS_CONFIG_PATH.name == "read.yaml"

    def test_refresh_route_is_fastapi_route(self):
        """The refresh route must be registered on the god-module router."""
        route_paths = [r.path for r in lpe.router.routes]
        assert "/aawm/alias-config/refresh" in route_paths

    def test_refresh_route_endpoint_preserves_god_module_compatibility(self):
        route = next(
            route
            for route in lpe.router.routes
            if route.path == "/aawm/alias-config/refresh"
        )
        assert route.endpoint.__module__ == lpe.__name__


class TestCodexOAuthParity:
    """Golden behavior parity for Codex auth-file/JWT helpers."""

    def test_clean_codex_auth_value_valid(self):
        assert lpe._clean_codex_auth_value("  token123  ") == "token123"

    def test_clean_codex_auth_value_none(self):
        assert lpe._clean_codex_auth_value(None) is None

    def test_clean_codex_auth_value_empty(self):
        assert lpe._clean_codex_auth_value("   ") is None

    def test_clean_codex_auth_value_non_string(self):
        assert lpe._clean_codex_auth_value(123) is None

    def test_decode_jwt_claims_valid(self):
        payload = {"sub": "user1", "exp": 9999999999}
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.signature"
        claims = lpe._decode_jwt_claims_without_validation(token)
        assert claims["sub"] == "user1"
        assert claims["exp"] == 9999999999

    def test_decode_jwt_claims_invalid(self):
        assert lpe._decode_jwt_claims_without_validation("not-a-jwt") == {}

    def test_decode_jwt_claims_empty(self):
        assert lpe._decode_jwt_claims_without_validation("") == {}

    def test_extract_codex_account_id_from_token_valid(self):
        payload = {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-123"
            }
        }
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.sig"
        assert lpe._extract_codex_account_id_from_token(token) == "acct-123"

    def test_extract_codex_account_id_from_token_none(self):
        assert lpe._extract_codex_account_id_from_token(None) is None

    def test_extract_codex_account_id_from_token_no_claim(self):
        payload = {"sub": "user1"}
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"header.{payload_b64}.sig"
        assert lpe._extract_codex_account_id_from_token(token) is None

    def test_get_codex_auth_token_data_nested(self):
        auth_data = {"tokens": {"access_token": "abc", "expires_at": 999}}
        result = lpe._get_codex_auth_token_data(auth_data)
        assert result["access_token"] == "abc"

    def test_get_codex_auth_token_data_flat(self):
        auth_data = {"access_token": "abc", "expires_at": 999}
        result = lpe._get_codex_auth_token_data(auth_data)
        assert result["access_token"] == "abc"

    def test_get_codex_auth_token_expiry_valid(self):
        payload = {"exp": 1234567890}
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"h.{payload_b64}.s"
        assert lpe._get_codex_auth_token_expiry(token) == 1234567890

    def test_get_codex_auth_token_expiry_missing(self):
        payload = {"sub": "user"}
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        token = f"h.{payload_b64}.s"
        assert lpe._get_codex_auth_token_expiry(token) is None

    def test_codex_auth_access_token_is_valid_no_token(self):
        assert lpe._codex_auth_access_token_is_valid({}) is False

    def test_codex_auth_access_token_is_valid_no_expiry(self):
        """Token with no expiry info is treated as valid."""
        token_data = {"access_token": "some-token-without-exp"}
        assert lpe._codex_auth_access_token_is_valid(token_data) is True

    def test_codex_auth_access_token_is_valid_future_expiry(self):
        future_exp = int(time.time()) + 3600
        token_data = {"access_token": "x", "expires_at": future_exp}
        assert lpe._codex_auth_access_token_is_valid(token_data) is True

    def test_codex_auth_access_token_is_valid_past_expiry(self):
        past_exp = int(time.time()) - 3600
        token_data = {"access_token": "x", "expires_at": past_exp}
        assert lpe._codex_auth_access_token_is_valid(token_data) is False

    def test_auth_file_env_vars_are_tuples(self):
        assert isinstance(lpe._ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS, tuple)
        assert len(lpe._ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS) > 0

    def test_token_dir_env_vars_are_tuples(self):
        assert isinstance(lpe._ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS, tuple)
        assert len(lpe._ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS) > 0

    def test_default_auth_paths_are_tuples(self):
        assert isinstance(lpe._ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS, tuple)
        assert len(lpe._ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS) > 0

    def test_type_aliases_are_dicts(self):
        """CodexAuthData/CodexTokenData/OAuthJsonData are dict type aliases."""
        assert lpe.CodexAuthData == dict[str, object]
        assert lpe.CodexTokenData == dict[str, object]
        assert lpe.OAuthJsonData == dict[str, object]


class TestOpenRouterQuotaParity:
    """Golden behavior parity for OpenRouter free-quota helpers."""

    def test_parse_reset_timestamp_int(self):
        assert lpe._parse_openrouter_free_daily_quota_reset_timestamp(1000) == 1000.0

    def test_parse_reset_timestamp_float(self):
        assert lpe._parse_openrouter_free_daily_quota_reset_timestamp(1000.5) == 1000.5

    def test_parse_reset_timestamp_datetime(self):
        dt = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
        result = lpe._parse_openrouter_free_daily_quota_reset_timestamp(dt)
        assert result == dt.timestamp()

    def test_parse_reset_timestamp_datetime_naive(self):
        dt = datetime(2026, 7, 25, 12, 0)
        result = lpe._parse_openrouter_free_daily_quota_reset_timestamp(dt)
        expected = dt.replace(tzinfo=timezone.utc).timestamp()
        assert result == expected

    def test_parse_reset_timestamp_iso_string(self):
        result = lpe._parse_openrouter_free_daily_quota_reset_timestamp("2026-07-25T12:00:00Z")
        expected = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc).timestamp()
        assert result == expected

    def test_parse_reset_timestamp_empty_string(self):
        assert lpe._parse_openrouter_free_daily_quota_reset_timestamp("") is None

    def test_parse_reset_timestamp_invalid_string(self):
        assert lpe._parse_openrouter_free_daily_quota_reset_timestamp("not-a-date") is None

    def test_parse_reset_timestamp_none(self):
        assert lpe._parse_openrouter_free_daily_quota_reset_timestamp(None) is None

    def test_is_openrouter_free_quota_candidate_matching(self):
        candidate = {
            "provider": lpe._CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
            "model": next(iter(lpe._OPENROUTER_FREE_DAILY_QUOTA_MODELS)),
        }
        assert lpe._is_openrouter_free_quota_candidate(candidate) is True

    def test_is_openrouter_free_quota_candidate_wrong_provider(self):
        candidate = {"provider": "other", "model": "some-free-model"}
        assert lpe._is_openrouter_free_quota_candidate(candidate) is False

    def test_is_openrouter_free_quota_candidate_wrong_model(self):
        candidate = {
            "provider": lpe._CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
            "model": "definitely-not-free-model-xyz",
        }
        assert lpe._is_openrouter_free_quota_candidate(candidate) is False

    def test_quota_constants_types(self):
        assert isinstance(lpe._OPENROUTER_DURABLE_QUOTA_DAILY_KEY, str)
        assert isinstance(lpe._OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS, float)
        assert isinstance(lpe._OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS, float)
        assert isinstance(lpe._OPENROUTER_FREE_DAILY_QUOTA_MODELS, (set, frozenset, list, tuple))

    def test_quota_daily_key_value(self):
        assert lpe._OPENROUTER_DURABLE_QUOTA_DAILY_KEY == "openrouter_free_daily_requests:requests"

    def test_quota_cache_ttl_value(self):
        assert lpe._OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS == 30.0

    def test_quota_lookup_timeout_value(self):
        assert lpe._OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS == 0.5

    def test_raise_candidate_unavailable_raises(self):
        with pytest.raises(Exception) as exc_info:
            lpe._raise_openrouter_auto_agent_candidate_unavailable("test message")
        assert getattr(exc_info.value, "message", str(exc_info.value)) == "test message"

    def test_read_pilot_alias_name_value(self):
        assert lpe._READ_PILOT_ALIAS_NAME == "read"


# ===========================================================================
# SECTION 6: Inventory uniqueness (GREEN now)
# ===========================================================================


class TestW5AInventoryUniqueness:
    """Each symbol must appear in exactly one Wave 5A target module."""

    def test_no_cross_target_duplicates(self):
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for module_key, symbols in W5A_SYMBOL_INVENTORY.items():
            for sym in symbols:
                if sym in seen:
                    duplicates.append(f"{sym} in both {seen[sym]} and {module_key}")
                else:
                    seen[sym] = module_key
        assert not duplicates, f"Duplicate symbols across W5A targets: {duplicates}"

    def test_no_overlap_with_deferred_state(self):
        for module_key, symbols in W5A_SYMBOL_INVENTORY.items():
            overlap = symbols & W5A_DEFERRED_STATE
            assert not overlap, (
                f"{module_key} overlaps with deferred state: {sorted(overlap)}"
            )

    def test_no_overlap_with_moved_constants(self):
        for module_key, symbols in W5A_SYMBOL_INVENTORY.items():
            overlap = symbols & W5A_ALL_MOVED_CONSTANTS
            assert not overlap, (
                f"{module_key} overlaps with moved constants: {sorted(overlap)}"
            )

    def test_no_overlap_with_wave4_inventory(self):
        """Wave 5A symbols must not collide with Wave 4 moved symbols."""
        from tests.test_litellm.proxy.pass_through_endpoints.test_decomposition_ownership import (
            ALL_MOVED_FUNCTIONS as W4_ALL_MOVED,
            ALL_RESTORED_CONSTANTS as W4_CONSTANTS,
        )

        w4_all = W4_ALL_MOVED | W4_CONSTANTS
        for module_key, symbols in W5A_SYMBOL_INVENTORY.items():
            overlap = symbols & w4_all
            assert not overlap, (
                f"W5A {module_key} overlaps with Wave 4 symbols: {sorted(overlap)}"
            )



# ===========================================================================
# SECTION 8: God-owned quota cache authority (Wave 5A regression)
# ===========================================================================


class TestW5AGodOwnedQuotaCacheAuthority:
    """The god-module ``_openrouter_free_daily_quota_cache`` tuple is the
    single authoritative state.  Wave 5A modules must observe and mutate it
    through injected getter/setter callbacks, never through a module-local
    copy."""

    def _save_quota_cache(self):
        return lpe._openrouter_free_daily_quota_cache

    def _restore_quota_cache(self, saved):
        # Use the god-module setter so the variable is correctly rebound.
        lpe._set_openrouter_free_daily_quota_cache(saved)

    def test_reset_through_facade_updates_god_module_variable(self):
        """_reset_openrouter_free_daily_quota_cache() called via the god-module
        facade must rebind ``lpe._openrouter_free_daily_quota_cache``."""
        saved = self._save_quota_cache()
        try:
            # Seed a non-default value directly on the god module.
            lpe._set_openrouter_free_daily_quota_cache((999.0, 12345.0))
            assert lpe._openrouter_free_daily_quota_cache == (999.0, 12345.0)

            # Reset through the facade (which delegates to openrouter_quota).
            lpe._reset_openrouter_free_daily_quota_cache()

            # The god-module variable must now be the default.
            assert lpe._openrouter_free_daily_quota_cache == (None, 0.0)
        finally:
            self._restore_quota_cache(saved)

    def test_quota_cache_write_visible_through_god_module(self):
        """A cache update performed by the openrouter_quota module (via the
        injected setter) must be visible as the god-module variable."""
        saved = self._save_quota_cache()
        try:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
                openrouter_quota as oq,
            )

            # Write through the injected setter.
            assert oq._set_quota_cache is not None
            oq._set_quota_cache((42.0, 99.0))

            # Must be visible on the god-module variable.
            assert lpe._openrouter_free_daily_quota_cache == (42.0, 99.0)
        finally:
            self._restore_quota_cache(saved)

    def test_quota_cache_read_observes_god_module(self):
        """The injected getter must return the current god-module value,
        not a stale module-local copy."""
        saved = self._save_quota_cache()
        try:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
                openrouter_quota as oq,
            )

            # Set a value on the god module directly.
            lpe._set_openrouter_free_daily_quota_cache((7.0, 8.0))

            # The injected getter must see it.
            assert oq._get_quota_cache is not None
            assert oq._get_quota_cache() == (7.0, 8.0)
        finally:
            self._restore_quota_cache(saved)

    def test_quota_lock_is_god_module_lock(self):
        """The lock used by openrouter_quota must be the exact same object
        as the god-module lock."""
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            openrouter_quota as oq,
        )

        assert oq._quota_lock is lpe._openrouter_free_daily_quota_lock
