"""RR-054 finding #1 structural extraction tests (AST/source ownership).

Enforces the *accepted* ownership split documented in
``litellm/proxy/pass_through_endpoints/architecture.md`` under
"AAWM alias routing and adapter ownership (RR-054)":

- ``llm_passthrough_endpoints.py`` retains FastAPI wiring + thin wrappers
- candidate tables / policy / state / durable / driver / finalization logic
  live in ``aawm_alias_routing`` package modules with real implementations
- the god-file must *delegate* to those modules and must not re-define the
  declared seam implementations

These tests do not require total elimination of FastAPI route handlers. They do
require provider request preparation and route-plan construction to live under
``llms/anthropic/experimental_pass_through/providers/<provider>/adapter.py``;
the god-file may retain only dependency binding and thin delegates.

Write-only surface: this file. No production/docs/other-test edits.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints import (
    aawm_alias_routing_policy as policy_compat,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(package.__file__).resolve().parent
PASS_THROUGH_DIR = PACKAGE_DIR.parent
GOD_PATH = Path(lpe.__file__).resolve()
COMPAT_POLICY_PATH = Path(policy_compat.__file__).resolve()
ARCHITECTURE_PATH = PASS_THROUGH_DIR / "architecture.md"
PROVIDER_DIR = (
    GOD_PATH.parents[2]
    / "llms"
    / "anthropic"
    / "experimental_pass_through"
    / "providers"
)

# Package modules that architecture.md assigns ownership to.
ARCHITECTURE_OWNED_MODULES = {
    "policy": "Candidate tables, aliases, model allowlists, cooldown defaults",
    "state": "Cooldown, affinity, OAuth, lane-cache, and candidate probe-lock state",
    "durable": "Durable Redis keys, max-expiry writes, negative reads, DualCache",
    "google_oauth": "Google OAuth file/token I/O",
    "antigravity_oauth": "Antigravity OAuth file/token I/O",
    "adapter_config": "Config-driven nine-route preparation/execution plans",
    "adapter_driver": "Config-driven nine-route preparation/execution plans",
    "provider_shaping": "Provider text/JSON shaping primitives",
    "streaming": "Bounded, lossless stream peeking",
    "responses_finalize": "Responses-to-Anthropic finalization",
    "retry": "Shared retry attempt sequencing and cooldown waits",
    "task_state": "Structured task-state source selection",
}

# Supporting modules present in the package but not each named as a row owner.
SUPPORTING_PACKAGE_MODULES = {
    "memory": "Bounded in-memory map helpers used by state/retry",
    "oauth_token_cache": "OAuthAccessTokenCache class used by state",
}

# Declared seam symbols: owner_module -> symbols that must be *defined* there
# and must not be re-defined as FunctionDef/ClassDef on the god-file.
SEAM_DEFINITIONS: dict[str, set[str]] = {
    "policy": {
        "CODEX_AUTO_AGENT_CANDIDATES",
        "CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS",
        "ANTHROPIC_AUTO_AGENT_CANDIDATES",
        "ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS",
        "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS",
        "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS",
        "OPENROUTER_FREE_DAILY_QUOTA_MODELS",
    },
    "state": {
        "AliasFamilyState",
        "MonotonicCooldownMap",
        "AliasRoutingStateManager",
        "alias_routing_state",
    },
    "durable": {
        "get_aawm_alias_routing_dual_cache",
        "build_aawm_alias_routing_durable_cache_key",
        "parse_aawm_alias_routing_durable_expiry",
        "read_aawm_alias_routing_durable_payload",
        "write_aawm_alias_routing_durable_payload",
        "get_aawm_alias_routing_state_namespace",
        "AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX",
    },
    "memory": {
        "bound_memory_map",
        "hydrate_cooldown_memory",
        "hydrate_affinity_memory",
        "extend_monotonic_cooldown",
        "DEFAULT_MEMORY_STATE_MAX_SIZE",
    },
    "retry": {
        "AdapterRetryPolicy",
        "run_adapter_retry_policy",
        "wait_for_monotonic_cooldown_map",
        "set_monotonic_cooldown_map",
        "projected_hidden_retry_within_budget",
    },
    "adapter_driver": {
        "ResponsesAdapterRoutePlan",
        "CompletionAdapterRoutePlan",
        "run_responses_adapter_route",
        "run_completion_adapter_route",
    },
    "adapter_config": {
        "AnthropicResponsesAdapterConfig",
        "AnthropicCompletionAdapterConfig",
        "OPENAI_RESPONSES",
        "responses_finalize_kwargs",
    },
    "responses_finalize": {
        "ResponsesFinalizeRuntime",
        "configure_responses_finalize_runtime",
        "finalize_anthropic_responses_adapter_upstream_response",
    },
    "provider_shaping": {
        "decode_json_prefix",
        "iter_delimited_spans",
        "TextSpan",
    },
    "streaming": {
        "BoundedStreamPeek",
        "peek_streaming_response",
    },
    "task_state": {
        "select_task_state_source",
        "resolve_task_state_markers",
        "message_has_structured_task_state_flag",
    },
    "oauth_token_cache": {
        "OAuthAccessTokenCache",
        "google_oauth_access_token_cache",
        "antigravity_oauth_access_token_cache",
    },
}

# Substance markers that prove a module owns real logic, not re-exports only.
MODULE_SUBSTANCE_MARKERS: dict[str, tuple[str, ...]] = {
    "policy": (
        "CODEX_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (",
        "ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[str, tuple[dict[str, Any], ...]] = {",
        '"last_resort": True,',
        "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS",
    ),
    "state": (
        "class AliasFamilyState",
        "class MonotonicCooldownMap",
        "class AliasRoutingStateManager",
        "alias_routing_state = AliasRoutingStateManager()",
    ),
    "durable": (
        "def write_aawm_alias_routing_durable_payload",
        "def read_aawm_alias_routing_durable_payload",
        "def build_aawm_alias_routing_durable_cache_key",
        "expires_at_epoch",
        # max-expiry ownership (RR-054 #2 behavior lives with durable owner)
        "existing",
    ),
    "memory": (
        "def bound_memory_map",
        "def hydrate_cooldown_memory",
        "def hydrate_affinity_memory",
        "DEFAULT_MEMORY_STATE_MAX_SIZE",
    ),
    "retry": (
        "class AdapterRetryPolicy",
        "def run_adapter_retry_policy",
        "def wait_for_monotonic_cooldown_map",
        "async with cooldown_map.lock",
    ),
    "adapter_driver": (
        "class ResponsesAdapterRoutePlan",
        "class CompletionAdapterRoutePlan",
        "async def run_responses_adapter_route",
        "async def run_completion_adapter_route",
        "await prepare(",
        "await perform(",
    ),
    "adapter_config": (
        "class AnthropicResponsesAdapterConfig",
        "class AnthropicCompletionAdapterConfig",
        "OPENAI_RESPONSES",
        "def responses_finalize_kwargs",
    ),
    "responses_finalize": (
        "class ResponsesFinalizeRuntime",
        "async def finalize_anthropic_responses_adapter_upstream_response",
        "runtime.validate_stream",
        "runtime.build_response",
    ),
    "provider_shaping": (
        "def decode_json_prefix",
        "def iter_delimited_spans",
        "class TextSpan",
    ),
    "streaming": (
        "class BoundedStreamPeek",
        "async def peek_streaming_response",
    ),
    "task_state": (
        "def select_task_state_source",
        "def resolve_task_state_markers",
    ),
    "google_oauth": (
        "def _load_valid_local_google_oauth_access_token",
        "def _refresh_local_google_oauth_credentials",
        "configure_google_oauth_runtime",
    ),
    "antigravity_oauth": (
        "def _load_valid_local_antigravity_access_token",
        "def _refresh_local_antigravity_oauth_token_data",
        "ANTIGRAVITY",
    ),
    "oauth_token_cache": (
        "class OAuthAccessTokenCache",
        "google_oauth_access_token_cache",
        "antigravity_oauth_access_token_cache",
    ),
}

# God-file thin wrappers that are allowed to exist if they *call* the package.
ALLOWED_GOD_THIN_WRAPPERS: dict[str, str] = {
    "_bound_aawm_alias_routing_memory_map": "_aawm_alias_memory.bound_memory_map",
    "_hydrate_aawm_alias_routing_cooldown_memory": (
        "_aawm_alias_memory.hydrate_cooldown_memory"
    ),
    "_hydrate_aawm_alias_routing_affinity_memory": (
        "_aawm_alias_memory.hydrate_affinity_memory"
    ),
    "_finalize_anthropic_responses_adapter_upstream_response": (
        "_aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response"
    ),
}

# Nine Anthropic adapter route handlers that must delegate to package drivers.
# Architecture: "prepare a provider-specific route plan and delegate execution
# to the shared package drivers."
NINE_ROUTE_HANDLERS_TO_DRIVER: dict[str, str] = {
    "_handle_anthropic_openai_responses_adapter_route": (
        "run_responses_adapter_route"
    ),
    "_handle_anthropic_xai_oauth_responses_adapter_route": (
        "run_responses_adapter_route"
    ),
    "_handle_anthropic_grok_native_oauth_responses_adapter_route": (
        "run_responses_adapter_route"
    ),
    "_handle_anthropic_openrouter_responses_adapter_route": (
        "run_responses_adapter_route"
    ),
    "_handle_anthropic_opencode_zen_responses_adapter_route": (
        "run_responses_adapter_route"
    ),
    "_handle_anthropic_xai_oauth_completion_adapter_route": (
        "run_completion_adapter_route"
    ),
    "_handle_anthropic_nvidia_completion_adapter_route": (
        "run_completion_adapter_route"
    ),
    "_handle_anthropic_openrouter_completion_adapter_route": (
        "run_completion_adapter_route"
    ),
    "_handle_anthropic_opencode_zen_completion_adapter_route": (
        "run_completion_adapter_route"
    ),
}

# Durable helpers that god-file must bind to package (assignment re-export OK).
DURABLE_GOD_BINDINGS = {
    "_get_aawm_alias_routing_dual_cache": "get_aawm_alias_routing_dual_cache",
    "_build_aawm_alias_routing_durable_cache_key": (
        "build_aawm_alias_routing_durable_cache_key"
    ),
    "_parse_aawm_alias_routing_durable_expiry": (
        "parse_aawm_alias_routing_durable_expiry"
    ),
    "_read_aawm_alias_routing_durable_payload": (
        "read_aawm_alias_routing_durable_payload"
    ),
    "_write_aawm_alias_routing_durable_payload": (
        "write_aawm_alias_routing_durable_payload"
    ),
    "_get_aawm_alias_routing_state_namespace": (
        "get_aawm_alias_routing_state_namespace"
    ),
}

POLICY_LITERAL_MARKERS = MODULE_SUBSTANCE_MARKERS["policy"][:3]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.AST:
    return ast.parse(_read(path), filename=str(path))


def _top_level_def_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    assert isinstance(tree, ast.Module)
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


def _top_level_function_class_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _function_node(tree: ast.AST, name: str) -> Optional[ast.AST]:
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _call_attr_names(fn_node: ast.AST) -> set[str]:
    """Return dotted attribute call names found under a function body."""
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
            continue
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            cur: ast.AST = func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.add(".".join(reversed(parts)))
    return names


def _is_reexport_only_module(source: str, tree: ast.AST) -> bool:
    """True if module body is only imports / __all__ / docstrings / pass."""
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign):
            targets = [
                t.id for t in node.targets if isinstance(t, ast.Name)
            ]
            if targets and set(targets) <= {"__all__"}:
                continue
            # Any other assignment is real ownership (constants, singletons).
            return False
        if isinstance(node, ast.AnnAssign):
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(node, ast.Pass):
            continue
        return False
    # Imports-only with no owned symbols is re-export-only.
    return True


def _count_nontrivial_functions(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Ignore pure docstring / pass bodies.
        body = [
            s
            for s in node.body
            if not (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str)
            )
        ]
        if not body:
            continue
        if all(isinstance(s, ast.Pass) for s in body):
            continue
        count += 1
    return count


def _assignment_value_is_package_attr(
    tree: ast.AST,
    target_name: str,
    *,
    module_alias: str,
    attr_name: str,
) -> bool:
    """True if ``target_name = module_alias.attr_name`` at module scope."""
    assert isinstance(tree, ast.Module)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_ids = [
            t.id for t in node.targets if isinstance(t, ast.Name)
        ]
        if target_name not in target_ids:
            continue
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == module_alias
            and value.attr == attr_name
        ):
            return True
    return False


@dataclass(frozen=True)
class DiscoveryItem:
    path: str
    classification: str  # enforced | context_only
    note: str


# ---------------------------------------------------------------------------
# Discovery inventory (required by the packet prompt)
# ---------------------------------------------------------------------------


def test_rr054_structural_discovery_inventory() -> None:
    """List every package module + architecture ownership statement inspected.

    Classification:
    - enforced: covered by assertions in this file
    - context_only: read for boundary understanding, not a fail/pass gate
    """
    items: list[DiscoveryItem] = []

    # Architecture ownership statements
    arch = _read(ARCHITECTURE_PATH)
    assert "AAWM alias routing and adapter ownership (RR-054)" in arch
    # Architecture wraps this sentence across lines; normalize whitespace.
    arch_flat = " ".join(arch.split())
    assert (
        "retains FastAPI route registration and thin compatibility wrappers"
        in arch_flat
    )
    for module, concern in ARCHITECTURE_OWNED_MODULES.items():
        owner_path = f"aawm_alias_routing/{module}.py"
        assert owner_path in arch or module in arch, (
            f"architecture ownership row missing for {module}"
        )
        items.append(
            DiscoveryItem(
                path=f"architecture.md::{concern}",
                classification="enforced",
                note=f"owner module {module}.py required to hold real logic",
            )
        )

    items.append(
        DiscoveryItem(
            path="architecture.md::god retains FastAPI wiring + thin wrappers",
            classification="enforced",
            note="tests allow route handlers; require delegation not deletion",
        )
    )
    items.append(
        DiscoveryItem(
            path="architecture.md::provider route preparation ownership",
            classification="enforced",
            note="per-provider algorithms must live below providers/<provider>",
        )
    )
    items.append(
        DiscoveryItem(
            path="architecture.md::runtime invariants (single-flight, stream bounds, ...)",
            classification="context_only",
            note="behavioral; covered by other RR-054 test modules",
        )
    )

    # Every package module on disk
    on_disk = sorted(
        p for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py"
    )
    assert on_disk, "package has no modules"
    for path in on_disk:
        stem = path.stem
        if stem in ARCHITECTURE_OWNED_MODULES or stem in SUPPORTING_PACKAGE_MODULES:
            items.append(
                DiscoveryItem(
                    path=str(path.relative_to(PASS_THROUGH_DIR)),
                    classification="enforced",
                    note="substance + single-owner checks",
                )
            )
        else:
            items.append(
                DiscoveryItem(
                    path=str(path.relative_to(PASS_THROUGH_DIR)),
                    classification="context_only",
                    note="supporting module outside the named ownership table",
                )
            )

    for provider in (
        "grok",
        "nvidia",
        "openai",
        "opencode_zen",
        "openrouter",
        "xai",
    ):
        for filename in ("__init__.py", "adapter.py"):
            path = PROVIDER_DIR / provider / filename
            assert path.is_file(), path
            items.append(
                DiscoveryItem(
                    path=str(path.relative_to(GOD_PATH.parents[3])),
                    classification="enforced",
                    note="provider-owned route preparation",
                )
            )

    items.append(
        DiscoveryItem(
            path="aawm_alias_routing/__init__.py",
            classification="enforced",
            note="public surface must export package owners, not empty shell",
        )
    )
    items.append(
        DiscoveryItem(
            path="aawm_alias_routing_policy.py",
            classification="enforced",
            note="compat re-export only; must not own candidate table literals",
        )
    )
    items.append(
        DiscoveryItem(
            path="llm_passthrough_endpoints.py",
            classification="enforced",
            note="must import package and not redefine declared seam implementations",
        )
    )
    items.append(
        DiscoveryItem(
            path="litellm/proxy/aawm_alias_routing_redis.py",
            classification="context_only",
            note="separate Redis connection manager owner",
        )
    )

    # Require that inventory covers both enforced owners and context boundaries.
    classifications = {i.classification for i in items}
    assert "enforced" in classifications
    assert "context_only" in classifications
    assert len(items) >= 20

    # Fail loudly if a package module appeared that we never classified.
    classified_paths = {
        i.path.split("/")[-1]
        for i in items
        if i.path.startswith("aawm_alias_routing/")
    }
    for path in on_disk:
        assert path.name in classified_paths or path.stem in {
            *(ARCHITECTURE_OWNED_MODULES),
            *(SUPPORTING_PACKAGE_MODULES),
        }


# ---------------------------------------------------------------------------
# Architecture ↔ package module presence
# ---------------------------------------------------------------------------


def test_rr054_architecture_owner_modules_exist_on_disk() -> None:
    for module in ARCHITECTURE_OWNED_MODULES:
        path = PACKAGE_DIR / f"{module}.py"
        assert path.is_file(), f"architecture owner missing: {path}"


def test_rr054_package_init_exports_architecture_owners() -> None:
    """``__all__`` must surface the layered package, not a hollow re-export shell."""
    public = set(package.__all__)
    required = {
        "adapter_config",
        "adapter_driver",
        "policy",
        "state",
        "retry",
        "responses_finalize",
        "alias_routing_state",
        "AliasRoutingStateManager",
    }
    missing = required - public
    assert not missing, f"package __all__ missing owners: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Meaningful extraction (not re-export-only modules)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", sorted(MODULE_SUBSTANCE_MARKERS))
def test_rr054_package_module_is_not_reexport_only(module_name: str) -> None:
    path = PACKAGE_DIR / f"{module_name}.py"
    source = _read(path)
    tree = _parse(path)
    assert not _is_reexport_only_module(source, tree), (
        f"{module_name}.py is re-export-only; RR-054 #1 requires real ownership"
    )
    # Require either nontrivial functions or owned table/constants.
    defs = _top_level_def_names(tree)
    nontrivial = _count_nontrivial_functions(tree)
    markers = MODULE_SUBSTANCE_MARKERS[module_name]
    markers_hit = [m for m in markers if m in source]
    assert markers_hit, (
        f"{module_name}.py missing substance markers {markers}; "
        "looks like a hollow extraction"
    )
    assert nontrivial > 0 or len(defs) >= 3, (
        f"{module_name}.py lacks nontrivial functions/owned symbols "
        f"(nontrivial_fns={nontrivial}, defs={sorted(defs)[:12]})"
    )


@pytest.mark.parametrize(
    "module_name,symbols",
    sorted(SEAM_DEFINITIONS.items()),
)
def test_rr054_seam_symbols_defined_in_owner_module(
    module_name: str, symbols: set[str]
) -> None:
    path = PACKAGE_DIR / f"{module_name}.py"
    names = _top_level_def_names(_parse(path))
    missing = sorted(symbols - names)
    assert not missing, (
        f"{module_name}.py does not define seam symbols {missing}; "
        "ownership is incomplete / re-export shell"
    )


# ---------------------------------------------------------------------------
# Single ownership: no duplicate seam *implementations* on god-file / shims
# ---------------------------------------------------------------------------


def test_rr054_god_file_does_not_redefine_seam_function_or_class_bodies() -> None:
    """God-file must not host FunctionDef/ClassDef for extracted seam symbols.

    Assignment aliases (``_x = package.x``) and thin wrappers that *call* the
    package are allowed and checked separately.
    """
    god_tree = _parse(GOD_PATH)
    god_fn_cls = _top_level_function_class_names(god_tree)

    # Symbols that would be exact name collisions if duplicated.
    forbidden_exact = {
        "AliasFamilyState",
        "MonotonicCooldownMap",
        "AliasRoutingStateManager",
        "AdapterRetryPolicy",
        "ResponsesAdapterRoutePlan",
        "CompletionAdapterRoutePlan",
        "ResponsesFinalizeRuntime",
        "AnthropicResponsesAdapterConfig",
        "AnthropicCompletionAdapterConfig",
        "OAuthAccessTokenCache",
        "TextSpan",
        "BoundedStreamPeek",
        "run_responses_adapter_route",
        "run_completion_adapter_route",
        "run_adapter_retry_policy",
        "wait_for_monotonic_cooldown_map",
        "set_monotonic_cooldown_map",
        "bound_memory_map",
        "hydrate_cooldown_memory",
        "hydrate_affinity_memory",
        "peek_streaming_response",
        "select_task_state_source",
        "get_aawm_alias_routing_dual_cache",
        "build_aawm_alias_routing_durable_cache_key",
        "write_aawm_alias_routing_durable_payload",
        "read_aawm_alias_routing_durable_payload",
        "finalize_anthropic_responses_adapter_upstream_response",
    }
    collisions = sorted(forbidden_exact & god_fn_cls)
    assert not collisions, (
        "god-file redefines extracted seam implementations: "
        f"{collisions}"
    )

    # Underscored god-file *function* definitions that would duplicate package
    # durable DAL instead of aliasing it.
    durable_fn_dupes = {
        f"_{name}"
        for name in (
            "get_aawm_alias_routing_dual_cache",
            "build_aawm_alias_routing_durable_cache_key",
            "parse_aawm_alias_routing_durable_expiry",
            "read_aawm_alias_routing_durable_payload",
            "write_aawm_alias_routing_durable_payload",
            "get_aawm_alias_routing_state_namespace",
        )
        if f"_{name}" in god_fn_cls
    }
    assert not durable_fn_dupes, (
        "god-file defines durable DAL functions instead of assignment aliases: "
        f"{sorted(durable_fn_dupes)}"
    )


def test_rr054_policy_table_literals_not_duplicated_outside_package() -> None:
    package_policy = _read(PACKAGE_DIR / "policy.py")
    god_source = _read(GOD_PATH)
    compat_source = _read(COMPAT_POLICY_PATH)

    for marker in POLICY_LITERAL_MARKERS:
        assert marker in package_policy, f"package policy missing {marker!r}"
        assert marker not in god_source, (
            f"god-file still owns candidate-table literal {marker!r}"
        )
        assert marker not in compat_source, (
            f"compat shim still owns candidate-table literal {marker!r}"
        )

    assert "Compatibility re-export" in compat_source
    assert "from .aawm_alias_routing import policy as _policy" in compat_source
    assert "import *" not in compat_source


def test_rr054_compat_policy_is_reexport_only() -> None:
    tree = _parse(COMPAT_POLICY_PATH)
    # No function/class definitions allowed on the compat shim.
    fn_cls = _top_level_function_class_names(tree)
    assert not fn_cls, (
        f"compat policy shim must not define functions/classes: {sorted(fn_cls)}"
    )
    # Must not look like a second policy owner.
    assert _is_reexport_only_module(_read(COMPAT_POLICY_PATH), tree) or (
        # Allow explicit re-export assign of __all__ etc., still no owned tables.
        "CODEX_AUTO_AGENT_CANDIDATES: tuple" not in _read(COMPAT_POLICY_PATH)
    )


# ---------------------------------------------------------------------------
# God-file delegation for declared seams
# ---------------------------------------------------------------------------


def test_rr054_god_file_imports_package_owners() -> None:
    god_source = _read(GOD_PATH)
    required_imports = [
        "from .aawm_alias_routing import adapter_config as _aawm_adapter_config",
        "from .aawm_alias_routing import adapter_driver as _aawm_adapter_driver",
        "from .aawm_alias_routing import memory as _aawm_alias_memory",
        "from .aawm_alias_routing import responses_finalize as _aawm_responses_finalize",
        "from .aawm_alias_routing import retry as _aawm_alias_retry",
        "from .aawm_alias_routing import durable as _aawm_alias_durable",
        "from .aawm_alias_routing.state import alias_routing_state as _alias_routing_state",
        "from .aawm_alias_routing import google_oauth as _aawm_google_oauth",
        "from .aawm_alias_routing import antigravity_oauth as _aawm_antigravity_oauth",
        "from .aawm_alias_routing import provider_shaping as _aawm_provider_shaping",
        "from .aawm_alias_routing import streaming as _aawm_alias_streaming",
    ]
    missing = [line for line in required_imports if line not in god_source]
    assert not missing, f"god-file missing package imports: {missing}"
    google_provider_source = "\n".join(
        _read(path)
        for path in (PROVIDER_DIR / "google").glob("*.py")
    )
    assert "_aawm_task_state.select_task_state_source" in google_provider_source


def test_rr054_god_file_binds_durable_helpers_to_package() -> None:
    god_tree = _parse(GOD_PATH)
    for god_name, package_attr in DURABLE_GOD_BINDINGS.items():
        assert _assignment_value_is_package_attr(
            god_tree,
            god_name,
            module_alias="_aawm_alias_durable",
            attr_name=package_attr,
        ), (
            f"{god_name} must be assignment-aliased to "
            f"_aawm_alias_durable.{package_attr}"
        )
        # Runtime identity check as well.
        package_durable = getattr(
            __import__(
                "litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable",
                fromlist=["durable"],
            ),
            package_attr,
        )
        assert getattr(lpe, god_name) is package_durable


def test_rr054_god_file_state_maps_are_package_identity_bound() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
        alias_routing_state,
    )

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
    assert lpe._google_oauth_access_token_cache is (
        alias_routing_state.google_oauth.tokens
    )
    assert lpe._antigravity_oauth_access_token_cache is (
        alias_routing_state.antigravity_oauth.tokens
    )

    # God-file must not reconstruct the process maps.
    god_source = _read(GOD_PATH)
    for snippet in (
        "_codex_auto_agent_cooldown_until_monotonic_by_key: dict",
        "_anthropic_auto_agent_cooldown_until_monotonic_by_key: dict",
        "_codex_auto_agent_session_affinity_by_key: dict",
        "_anthropic_auto_agent_session_affinity_by_key: dict",
        "_google_oauth_access_token_cache: dict",
        "_antigravity_oauth_access_token_cache: dict",
    ):
        assert snippet not in god_source, (
            f"god-file re-owns process map via local annotation: {snippet}"
        )


def test_rr054_allowed_god_thin_wrappers_delegate_to_package() -> None:
    god_tree = _parse(GOD_PATH)
    god_source = _read(GOD_PATH)
    for wrapper, required_call in ALLOWED_GOD_THIN_WRAPPERS.items():
        assert f"def {wrapper}(" in god_source or f"async def {wrapper}(" in god_source, (
            f"expected thin wrapper {wrapper} on god-file"
        )
        fn = _function_node(god_tree, wrapper)
        assert fn is not None, f"missing AST node for {wrapper}"
        calls = _call_attr_names(fn)
        assert required_call in calls, (
            f"{wrapper} must call package seam {required_call}; calls={sorted(calls)}"
        )
        # Thinness: wrapper body should not re-implement hydration / finalize
        # control flow (heuristic: few statements).
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        body_stmts = [
            s
            for s in fn.body
            if not (
                isinstance(s, ast.Expr)
                and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str)
            )
        ]
        assert len(body_stmts) <= 6, (
            f"{wrapper} looks too large to be a thin delegate "
            f"({len(body_stmts)} stmts); possible dual implementation"
        )


def test_rr054_nine_adapter_route_handlers_delegate_to_package_driver() -> None:
    """Architecture: nine entrypoints delegate execution to shared drivers."""
    god_tree = _parse(GOD_PATH)
    god_source = _read(GOD_PATH)

    # Driver implementations must live only in the package module.
    assert "async def run_responses_adapter_route" in _read(
        PACKAGE_DIR / "adapter_driver.py"
    )
    assert "async def run_completion_adapter_route" in _read(
        PACKAGE_DIR / "adapter_driver.py"
    )
    assert "async def run_responses_adapter_route" not in god_source
    assert "async def run_completion_adapter_route" not in god_source

    for handler, driver_fn in NINE_ROUTE_HANDLERS_TO_DRIVER.items():
        assert f"def {handler}(" in god_source or f"async def {handler}(" in god_source, (
            f"missing route handler {handler} (architecture expects nine entrypoints)"
        )
        fn = _function_node(god_tree, handler)
        assert fn is not None, handler
        calls = _call_attr_names(fn)
        expected = f"_aawm_adapter_driver.{driver_fn}"
        assert expected in calls, (
            f"{handler} must delegate to {expected}; calls={sorted(calls)}"
        )


PROVIDER_PREPARE_DELEGATES = {
    "_prepare_anthropic_openai_responses_adapter_route": (
        "_anthropic_openai_provider.prepare_responses_route"
    ),
    "_prepare_anthropic_xai_oauth_responses_adapter_route": (
        "_anthropic_xai_provider.prepare_responses_route"
    ),
    "_prepare_anthropic_grok_native_oauth_responses_adapter_route": (
        "_anthropic_grok_provider.prepare_responses_route"
    ),
    "_prepare_anthropic_xai_oauth_completion_adapter_route": (
        "_anthropic_xai_provider.prepare_completion_route"
    ),
    "_prepare_anthropic_nvidia_completion_adapter_route": (
        "_anthropic_nvidia_provider.prepare_completion_route"
    ),
    "_prepare_anthropic_openrouter_completion_adapter_route": (
        "_anthropic_openrouter_provider.prepare_completion_route"
    ),
    "_prepare_anthropic_openrouter_responses_adapter_route": (
        "_anthropic_openrouter_provider.prepare_responses_route"
    ),
    "_prepare_anthropic_opencode_zen_responses_adapter_route": (
        "_anthropic_opencode_zen_provider.prepare_responses_route"
    ),
    "_prepare_anthropic_opencode_zen_completion_adapter_route": (
        "_anthropic_opencode_zen_provider.prepare_completion_route"
    ),
}


def test_rr054_provider_adapter_modules_own_route_preparation() -> None:
    expected_modules = {
        "grok": {"prepare_responses_route"},
        "nvidia": {"prepare_completion_route"},
        "openai": {"prepare_responses_route"},
        "opencode_zen": {"prepare_completion_route", "prepare_responses_route"},
        "openrouter": {"prepare_completion_route", "prepare_responses_route"},
        "xai": {"prepare_completion_route", "prepare_responses_route"},
    }
    for provider, expected_defs in expected_modules.items():
        adapter_path = PROVIDER_DIR / provider / "adapter.py"
        init_path = PROVIDER_DIR / provider / "__init__.py"
        assert adapter_path.is_file(), adapter_path
        assert init_path.is_file(), init_path
        adapter_defs = _top_level_def_names(_parse(adapter_path))
        assert expected_defs <= adapter_defs
        assert "Runtime" in adapter_defs
        source = _read(adapter_path)
        assert "RoutePlan(" in source
        assert "prepared_request_body" in source


def test_rr054_provider_common_owns_shared_shaping_orchestration() -> None:
    common_path = PROVIDER_DIR / "common.py"
    common_defs = _top_level_def_names(_parse(common_path))
    assert {
        "ShapingRuntime",
        "build_responses_request_body",
        "prepare_completion_request_body",
        "apply_responses_policies",
    } <= common_defs

    god_tree = _parse(GOD_PATH)
    expected_delegates = {
        "_build_anthropic_responses_adapter_request_body": (
            "_anthropic_provider_common.build_responses_request_body"
        ),
        "_prepare_anthropic_completion_adapter_request_body": (
            "_anthropic_provider_common.prepare_completion_request_body"
        ),
        "_apply_anthropic_responses_adapter_policies_from_config": (
            "_anthropic_provider_common.apply_responses_policies"
        ),
    }
    for wrapper, expected_call in expected_delegates.items():
        fn = _function_node(god_tree, wrapper)
        assert fn is not None, wrapper
        assert expected_call in _call_attr_names(fn)


def test_rr054_god_file_provider_prepare_wrappers_are_thin_delegates() -> None:
    god_tree = _parse(GOD_PATH)
    for wrapper, provider_call in PROVIDER_PREPARE_DELEGATES.items():
        fn = _function_node(god_tree, wrapper)
        assert isinstance(fn, ast.AsyncFunctionDef), wrapper
        calls = _call_attr_names(fn)
        assert provider_call in calls, (
            f"{wrapper} must delegate to {provider_call}; calls={sorted(calls)}"
        )
        body_stmts = [
            statement
            for statement in fn.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        assert len(body_stmts) == 1, (
            f"{wrapper} retains provider shaping instead of a one-call delegate"
        )


def test_rr054_provider_algorithms_are_not_duplicated_in_god_file() -> None:
    god_source = _read(GOD_PATH)
    prepare_source = god_source[
        god_source.index("async def _prepare_anthropic_openai_responses_adapter_route") :
        god_source.index("async def _handle_anthropic_opencode_zen_adapter_route")
    ]
    for marker in (
        "Compacted Claude Code context for OpenAI Responses adapter",
        "Applied OpenCode Zen adapter explicit Bash tool choice",
        "Applied OpenRouter adapter parallel instruction policy",
    ):
        assert marker not in prepare_source
    assert "RoutePlan(" not in prepare_source


def test_rr054_responses_finalize_control_flow_owned_by_package() -> None:
    package_source = _read(PACKAGE_DIR / "responses_finalize.py")
    god_source = _read(GOD_PATH)

    # Package owns the real finalization control flow.
    for marker in (
        "async def finalize_anthropic_responses_adapter_upstream_response",
        "runtime.validate_stream",
        "runtime.collect_stream",
        "runtime.build_response",
    ):
        assert marker in package_source, f"package finalize missing {marker!r}"

    # God-file must not contain the package control-flow body markers as its
    # own implementation outside the configure/wrapper path. The thin wrapper
    # and runtime callback wiring are required by architecture.
    assert "configure_responses_finalize_runtime" in god_source
    assert (
        "_aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response"
        in god_source
    )

    # Ensure god wrapper does not embed validate/collect/build itself.
    fn = _function_node(
        _parse(GOD_PATH),
        "_finalize_anthropic_responses_adapter_upstream_response",
    )
    assert fn is not None
    calls = _call_attr_names(fn)
    assert (
        "_aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response"
        in calls
    )
    # Local re-implementation would typically call validate/collect helpers
    # directly without going through the package finalize entrypoint only.
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    body_stmts = [
        s
        for s in fn.body
        if not (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str)
        )
    ]
    assert len(body_stmts) <= 4, (
        "god finalize wrapper is not thin; possible duplicated finalization logic"
    )


def test_rr054_retry_and_driver_logic_not_copied_into_god_file() -> None:
    god_source = _read(GOD_PATH)
    package_retry = _read(PACKAGE_DIR / "retry.py")
    package_driver = _read(PACKAGE_DIR / "adapter_driver.py")

    # Distinctive implementation snippets that must remain package-only.
    exclusive_snippets = [
        (package_retry, "class AdapterRetryPolicy"),
        (package_retry, "async def run_adapter_retry_policy"),
        (package_retry, "async def wait_for_monotonic_cooldown_map"),
        (package_driver, "class ResponsesAdapterRoutePlan"),
        (package_driver, "class CompletionAdapterRoutePlan"),
        (package_driver, "async def run_responses_adapter_route"),
        (package_driver, "async def run_completion_adapter_route"),
    ]
    for owner_source, snippet in exclusive_snippets:
        assert snippet in owner_source
        # God may *reference* names via attributes, but must not define them.
        if snippet.startswith("class ") or snippet.startswith("async def "):
            assert snippet not in god_source, (
                f"god-file contains package-owned definition snippet {snippet!r}"
            )


def test_rr054_policy_runtime_identity_from_package() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import policy

    assert lpe._CODEX_AUTO_AGENT_CANDIDATES is policy.CODEX_AUTO_AGENT_CANDIDATES
    assert (
        lpe._CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
        is policy.CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
    )
    assert (
        lpe._ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
        is policy.ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
    )
    assert policy_compat.CODEX_AUTO_AGENT_CANDIDATES is policy.CODEX_AUTO_AGENT_CANDIDATES


# ---------------------------------------------------------------------------
# Explicit non-goals / accepted residuals (document honesty for #1)
# ---------------------------------------------------------------------------


def test_rr054_finding1_route_wiring_remains_without_provider_algorithms() -> None:
    """FastAPI handlers remain, but provider preparation does not."""
    god_source = _read(GOD_PATH)
    remaining_handlers = [
        name
        for name in NINE_ROUTE_HANDLERS_TO_DRIVER
        if f"def {name}(" in god_source or f"async def {name}(" in god_source
    ]
    assert len(remaining_handlers) == len(NINE_ROUTE_HANDLERS_TO_DRIVER)
    for provider_call in PROVIDER_PREPARE_DELEGATES.values():
        assert provider_call in god_source
    assert "_ANTHROPIC_OPENAI_PROVIDER_RUNTIME" in god_source
    assert "_ANTHROPIC_OPENROUTER_PROVIDER_RUNTIME" in god_source


def test_rr054_finding1_structural_acceptance_summary() -> None:
    """Aggregate gate for the original finding #1 responsibility carve."""
    package_modules = {
        p.stem for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py"
    }
    required = set(ARCHITECTURE_OWNED_MODULES) | set(SUPPORTING_PACKAGE_MODULES)
    missing = sorted(required - package_modules)
    assert not missing, f"package missing required modules: {missing}"

    # Every architecture owner has substance markers present.
    for module, markers in MODULE_SUBSTANCE_MARKERS.items():
        source = _read(PACKAGE_DIR / f"{module}.py")
        assert any(m in source for m in markers), module

    god_source = _read(GOD_PATH)
    # No second candidate-table owner.
    assert "CODEX_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (" not in god_source
    # Shared drivers not redefined.
    assert "async def run_responses_adapter_route" not in god_source
    assert "async def run_completion_adapter_route" not in god_source
    assert "class AdapterRetryPolicy" not in god_source
    assert "class AliasRoutingStateManager" not in god_source
    for provider in (
        "grok",
        "nvidia",
        "openai",
        "opencode_zen",
        "openrouter",
        "xai",
    ):
        source = _read(PROVIDER_DIR / provider / "adapter.py")
        assert "RoutePlan(" in source
        assert "class Runtime" in source
