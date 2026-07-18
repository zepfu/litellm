"""RR-054 package ownership tests for extracted ``aawm_alias_routing``.

Verifies:
- package modules are importable and re-export a stable public surface
- policy / state / durable DAL definitions have a single owner
- god-module and compat shim re-exports stay identity-compatible
- no duplicate class/function *definitions* for extracted state/DAL symbols

Does not edit production code. Surfaces any remaining dual-ownership seams.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints import aawm_alias_routing_policy as policy_compat
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_config,
    adapter_driver,
    durable,
    google_oauth,
    memory,
    oauth_token_cache,
    policy,
    provider_shaping,
    responses_finalize,
    retry,
    state,
    streaming,
    task_state,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    AliasRoutingStateManager,
    alias_routing_state,
)

PACKAGE_DIR = Path(package.__file__).resolve().parent
PASS_THROUGH_DIR = PACKAGE_DIR.parent
GOD_PATH = Path(lpe.__file__).resolve()
COMPAT_POLICY_PATH = Path(policy_compat.__file__).resolve()

# Extracted package modules that should remain importable as a unit.
EXPECTED_PACKAGE_MODULES = {
    "adapter_config",
    "adapter_driver",
    "antigravity_oauth",
    "durable",
    "google_oauth",
    "memory",
    "oauth_token_cache",
    "policy",
    "provider_shaping",
    "responses_finalize",
    "retry",
    "state",
    "streaming",
    "task_state",
}

# State/cache class definitions that must not be re-declared outside owners.
STATE_CLASS_OWNERS = {
    "AliasFamilyState": "state.py",
    "MonotonicCooldownMap": "state.py",
    "AliasRoutingStateManager": "state.py",
    "OAuthAccessTokenCache": "oauth_token_cache.py",
}

DAL_FUNCTION_OWNERS = {
    "get_aawm_alias_routing_dual_cache": "durable.py",
    "build_aawm_alias_routing_durable_cache_key": "durable.py",
    "parse_aawm_alias_routing_durable_expiry": "durable.py",
    "read_aawm_alias_routing_durable_payload": "durable.py",
    "write_aawm_alias_routing_durable_payload": "durable.py",
    "get_aawm_alias_routing_state_namespace": "durable.py",
    "bound_memory_map": "memory.py",
    "hydrate_cooldown_memory": "memory.py",
    "hydrate_affinity_memory": "memory.py",
}

POLICY_LITERAL_MARKERS = (
    "CODEX_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (",
    '"last_resort": True,',
    "ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[str, tuple[dict[str, Any], ...]] = {",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module_py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _top_level_defs(path: Path) -> set[str]:
    tree = ast.parse(_read(path), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_rr054_package_modules_are_importable() -> None:
    """Every extracted package module imports cleanly under the package path."""
    for name in sorted(EXPECTED_PACKAGE_MODULES):
        mod = importlib.import_module(
            f"litellm.proxy.pass_through_endpoints.aawm_alias_routing.{name}"
        )
        assert mod.__name__.endswith(f".aawm_alias_routing.{name}")
        assert Path(mod.__file__).resolve().parent == PACKAGE_DIR


def test_rr054_package_public_surface_matches_init() -> None:
    """``__all__`` lists only package-owned submodules/singletons."""
    assert set(package.__all__) == {
        "adapter_config",
        "adapter_driver",
        "alias_routing_state",
        "AliasRoutingStateManager",
        "google_oauth",
        "memory",
        "oauth_token_cache",
        "policy",
        "provider_shaping",
        "responses_finalize",
        "retry",
        "state",
        "streaming",
        "task_state",
    }
    assert package.alias_routing_state is alias_routing_state
    assert package.AliasRoutingStateManager is AliasRoutingStateManager
    assert package.policy is policy
    assert package.memory is memory
    assert package.retry is retry
    assert package.adapter_config is adapter_config
    assert package.adapter_driver is adapter_driver
    assert package.google_oauth is google_oauth
    assert package.oauth_token_cache is oauth_token_cache
    assert package.provider_shaping is provider_shaping
    assert package.responses_finalize is responses_finalize
    assert package.state is state
    assert package.streaming is streaming
    assert package.task_state is task_state


def test_rr054_package_files_live_under_aawm_alias_routing() -> None:
    on_disk = {
        p.stem for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py"
    }
    assert EXPECTED_PACKAGE_MODULES.issubset(on_disk)
    for required in ("policy.py", "state.py", "durable.py", "memory.py", "retry.py"):
        assert (PACKAGE_DIR / required).is_file()


# ---------------------------------------------------------------------------
# Single ownership: state / DAL definitions
# ---------------------------------------------------------------------------


def test_rr054_state_classes_have_single_owner() -> None:
    """State/cache classes are defined once, in the package owners."""
    scan_roots = [
        PACKAGE_DIR,
        PASS_THROUGH_DIR / "llm_passthrough_endpoints.py",
        PASS_THROUGH_DIR / "aawm_alias_routing_policy.py",
    ]
    definitions: dict[str, list[str]] = {name: [] for name in STATE_CLASS_OWNERS}

    for root in scan_roots:
        paths = [root] if root.is_file() else _module_py_files(root)
        for path in paths:
            names = _top_level_defs(path)
            for class_name in STATE_CLASS_OWNERS:
                if class_name in names:
                    definitions[class_name].append(path.name)

    for class_name, owner in STATE_CLASS_OWNERS.items():
        found = definitions[class_name]
        assert found == [owner], (
            f"{class_name} should be defined only in {owner}, found in {found}"
        )


def test_rr054_durable_and_memory_dal_have_single_owner() -> None:
    """Durable cache + memory helpers are defined once in package modules."""
    scan_paths = [
        *list(PACKAGE_DIR.glob("*.py")),
        GOD_PATH,
        COMPAT_POLICY_PATH,
    ]
    definitions: dict[str, list[str]] = {name: [] for name in DAL_FUNCTION_OWNERS}

    for path in scan_paths:
        names = _top_level_defs(path)
        for fn_name in DAL_FUNCTION_OWNERS:
            if fn_name in names:
                definitions[fn_name].append(path.name)

    for fn_name, owner in DAL_FUNCTION_OWNERS.items():
        found = definitions[fn_name]
        assert found == [owner], (
            f"{fn_name} should be defined only in {owner}, found in {found}"
        )


def test_rr054_policy_table_literals_live_only_in_package_policy() -> None:
    """Candidate-table row literals are owned by package policy, not shims."""
    package_policy_source = _read(Path(policy.__file__).resolve())
    compat_source = _read(COMPAT_POLICY_PATH)
    god_source = _read(GOD_PATH)

    for marker in POLICY_LITERAL_MARKERS:
        assert marker in package_policy_source, f"missing policy owner marker: {marker}"
        assert marker not in compat_source, f"compat shim still owns marker: {marker}"
        assert marker not in god_source, f"god-file still owns marker: {marker}"

    assert "Compatibility re-export" in compat_source
    assert "from .aawm_alias_routing import policy as _policy" in compat_source
    assert "import *" not in compat_source


def test_rr054_no_duplicate_state_map_construction_in_god_file() -> None:
    """God-file must bind process maps to package state, not construct new dicts."""
    assert (
        lpe._codex_auto_agent_cooldown_until_monotonic_by_key
        is alias_routing_state.codex.cooldown_until_monotonic_by_key
    )
    assert (
        lpe._anthropic_auto_agent_cooldown_until_monotonic_by_key
        is alias_routing_state.anthropic.cooldown_until_monotonic_by_key
    )
    assert lpe._codex_auto_agent_lock is alias_routing_state.codex.lock
    assert lpe._anthropic_auto_agent_lock is alias_routing_state.anthropic.lock
    assert (
        lpe._codex_auto_agent_session_affinity_by_key
        is alias_routing_state.codex.session_affinity_by_key
    )
    assert (
        lpe._anthropic_auto_agent_session_affinity_by_key
        is alias_routing_state.anthropic.session_affinity_by_key
    )
    assert (
        lpe._openrouter_adapter_rate_limit_until_monotonic_by_key
        is alias_routing_state.openrouter_rate_limit.until_monotonic_by_key
    )
    assert (
        lpe._google_adapter_rate_limit_until_monotonic_by_key
        is alias_routing_state.google_rate_limit.until_monotonic_by_key
    )
    assert lpe._google_oauth_access_token_cache is alias_routing_state.google_oauth.tokens
    assert (
        lpe._antigravity_oauth_access_token_cache
        is alias_routing_state.antigravity_oauth.tokens
    )
    assert lpe._google_oauth_access_token_lock is alias_routing_state.google_oauth.lock
    assert (
        lpe._antigravity_oauth_access_token_lock
        is alias_routing_state.antigravity_oauth.lock
    )

    god_source = _read(GOD_PATH)
    forbidden_constructors = (
        "_codex_auto_agent_cooldown_until_monotonic_by_key: dict",
        "_anthropic_auto_agent_cooldown_until_monotonic_by_key: dict",
        "_codex_auto_agent_session_affinity_by_key: dict",
        "_anthropic_auto_agent_session_affinity_by_key: dict",
        "_google_oauth_access_token_cache: dict",
        "_antigravity_oauth_access_token_cache: dict",
    )
    for snippet in forbidden_constructors:
        assert snippet not in god_source, f"god-file re-owns map via annotation: {snippet}"


# ---------------------------------------------------------------------------
# Re-export compatibility
# ---------------------------------------------------------------------------


def test_rr054_policy_compat_shim_reexports_package_policy() -> None:
    assert policy_compat.CODEX_AUTO_AGENT_CANDIDATES is policy.CODEX_AUTO_AGENT_CANDIDATES
    assert (
        policy_compat.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
        is policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    )
    assert (
        policy_compat.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
        is policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
    )
    assert (
        policy_compat.CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS
        == policy.CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS
    )
    assert "aawm-codex-agent-auto" in policy_compat.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    assert "aawm-code-anthropic" in policy_compat.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS


def test_rr054_god_file_reexports_policy_and_durable_helpers() -> None:
    assert lpe._CODEX_AUTO_AGENT_CANDIDATES is policy.CODEX_AUTO_AGENT_CANDIDATES
    assert (
        lpe._CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
        is policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    )
    assert (
        lpe._ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
        is policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
    )
    assert (
        lpe._CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS
        == policy.CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS
    )
    assert (
        lpe._ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS
        is policy.ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS
    )

    assert (
        lpe._get_aawm_alias_routing_dual_cache
        is durable.get_aawm_alias_routing_dual_cache
    )
    assert (
        lpe._build_aawm_alias_routing_durable_cache_key
        is durable.build_aawm_alias_routing_durable_cache_key
    )
    assert (
        lpe._parse_aawm_alias_routing_durable_expiry
        is durable.parse_aawm_alias_routing_durable_expiry
    )
    assert (
        lpe._read_aawm_alias_routing_durable_payload
        is durable.read_aawm_alias_routing_durable_payload
    )
    assert (
        lpe._write_aawm_alias_routing_durable_payload
        is durable.write_aawm_alias_routing_durable_payload
    )
    assert (
        lpe._AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX
        == durable.AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX
    )
    assert (
        lpe._AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE
        == memory.DEFAULT_MEMORY_STATE_MAX_SIZE
    )


def test_rr054_god_file_memory_wrappers_delegate_to_package() -> None:
    """Thin god wrappers may exist; they must call package memory helpers."""
    god_source = _read(GOD_PATH)
    assert "def _bound_aawm_alias_routing_memory_map(" in god_source
    assert "_aawm_alias_memory.bound_memory_map(" in god_source
    assert "_aawm_alias_memory.hydrate_cooldown_memory(" in god_source
    assert "_aawm_alias_memory.hydrate_affinity_memory(" in god_source

    cache: dict[str, int] = {str(i): i for i in range(5)}
    lpe._bound_aawm_alias_routing_memory_map(cache, max_size=3)
    assert len(cache) == 3


def test_rr054_adapter_config_and_retry_surfaces_are_package_owned() -> None:
    assert adapter_config.OPENAI_RESPONSES.adapter == (
        "anthropic_openai_responses_adapter"
    )
    assert hasattr(adapter_driver, "run_responses_adapter_route")
    assert hasattr(adapter_driver, "run_completion_adapter_route")
    assert hasattr(retry, "wait_for_monotonic_cooldown_map")
    assert hasattr(retry, "set_monotonic_cooldown_map")
    assert hasattr(retry, "AdapterRetryPolicy")
    assert hasattr(retry, "run_adapter_retry_policy")
    assert hasattr(task_state, "select_task_state_source")
    assert hasattr(provider_shaping, "decode_json_prefix")
    assert hasattr(provider_shaping, "iter_delimited_spans")
    assert hasattr(
        responses_finalize,
        "finalize_anthropic_responses_adapter_upstream_response",
    )
    assert hasattr(streaming, "peek_streaming_response")
    assert hasattr(oauth_token_cache, "google_oauth_access_token_cache")
    assert hasattr(oauth_token_cache, "antigravity_oauth_access_token_cache")


def test_rr054_state_singleton_is_shared_process_wide() -> None:
    reimported = importlib.import_module(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.state"
    )
    assert reimported.alias_routing_state is alias_routing_state
    assert isinstance(alias_routing_state, AliasRoutingStateManager)
    assert alias_routing_state.family("codex") is alias_routing_state.codex
    assert alias_routing_state.family("anthropic") is alias_routing_state.anthropic


# ---------------------------------------------------------------------------
# Remaining ownership inventory (documented, not failures of extracted DAL)
# ---------------------------------------------------------------------------


def test_rr054_remaining_duplicate_ownership_inventory() -> None:
    """Document residual dual-ownership seams still intentionally on the god-file.

    These are *not* duplicate state/DAL class definitions (already asserted
    above). They are compatibility wrappers / unextracted adapter process
    caches that future RR-054 increments may still move.
    """
    god_source = _read(GOD_PATH)
    remaining: dict[str, Any] = {
        "thin_memory_wrappers_on_god_file": [
            "_bound_aawm_alias_routing_memory_map",
            "_hydrate_aawm_alias_routing_cooldown_memory",
            "_hydrate_aawm_alias_routing_affinity_memory",
        ],
        "google_adapter_process_caches_still_on_god_file": [
            "_google_code_assist_project_cache",
            "_google_code_assist_prime_until_monotonic_by_key",
            "_google_adapter_semaphores",
            "_google_adapter_user_prompt_turn_counters",
            "_codex_google_code_assist_tool_call_name_cache",
        ],
        "redis_manager_outside_package": "litellm.proxy.aawm_alias_routing_redis",
        "runtime_engines_still_on_god_file": True,
    }

    for name in remaining["thin_memory_wrappers_on_god_file"]:
        assert f"def {name}(" in god_source
    for name in remaining["google_adapter_process_caches_still_on_god_file"]:
        assert name in god_source

    redis_mod = importlib.import_module("litellm.proxy.aawm_alias_routing_redis")
    assert redis_mod.__name__ == remaining["redis_manager_outside_package"]
    assert remaining["runtime_engines_still_on_god_file"] is True


@pytest.mark.parametrize(
    "module_name",
    sorted(EXPECTED_PACKAGE_MODULES),
)
def test_rr054_each_package_module_has_no_local_duplicate_of_state_classes(
    module_name: str,
) -> None:
    """Non-owner package modules must not re-define extracted state classes."""
    path = PACKAGE_DIR / f"{module_name}.py"
    names = _top_level_defs(path)
    for class_name, owner in STATE_CLASS_OWNERS.items():
        if path.name == owner:
            assert class_name in names
        else:
            assert class_name not in names, (
                f"{module_name}.py must not redefine {class_name} (owner={owner})"
            )
