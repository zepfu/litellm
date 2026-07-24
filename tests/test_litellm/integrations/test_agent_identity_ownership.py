"""Wave A2 (identity leaf extractions) ownership/parity tests.

`litellm/integrations/aawm_agent_identity/__init__.py` is CURRENTLY the
package form landed by Wave A1 (verbatim `git mv` of the single-file module).
Wave A2's engineer will move bounded, behavior-preserving leaf bands OUT of
`__init__.py` into new sibling submodules:

  - `constants.py`         <- env tables/regexes/field tuples
  - `coerce.py`             <- generic coercers + DSN build
  - `cost_map.py`           <- bundled model-cost-map helpers
  - `identity_tenant_agent.py` <- tenant/agent-id extraction
  - `identity_repository.py`   <- repo patterns/extractors/memory-workflow
  - `identity_runtime.py`      <- versions/user-agent/client-IP/host
  - `agent_context.py`         <- "You are '<agent>'" paths + codex/grok
                                  predicates + trace promotion

with façade rebinds left behind in `__init__.py` (`name = new_module.name`)
so every existing importer / monkeypatch target keeps working.

These tests are written BEFORE the move (TDD extraction, per
`.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md`
### Wave A2). They are EXPECTED to be RED until the engineer performs the
move: the ownership/facade-identity/rebind-order tests can only pass once
the target submodules exist and `__init__.py` no longer defines the moved
names directly. Do not weaken these assertions to make them pass early --
red-before-green is the intended TDD signal for this wave.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_PKG_DIR = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity"
INIT_PATH = IDENTITY_PKG_DIR / "__init__.py"

# Representative, defensible sampled set per target module (>= 5 names for the
# modules that have >= 5 candidate helpers in the plan's line-range map;
# `cost_map.py` has exactly 4 helpers in the current source, so all 4 are
# used for that module).
_MOVED_NAMES_BY_MODULE: Dict[str, List[str]] = {
    "constants": [
        "_AAWM_LITELLM_ENVIRONMENT_ENV_VARS",
        "_AAWM_LITELLM_VERSION_ENV_VARS",
        "_AAWM_ASSOCIATED_VERSION_ENV_VARS",
        "_AAWM_AGENT_ID_UUID_RE",
        "_AAWM_AGENT_ID_HEX_RE",
        "_AAWM_AGENT_ID_PREFIXED_RE",
        "_AAWM_SESSION_HISTORY_METADATA_KEYS",
        "_AAWM_TENANT_ID_METADATA_KEYS",
        "_AAWM_REPOSITORY_METADATA_KEYS",
    ],
    "coerce": [
        "_clean_secret_string",
        "_get_first_secret_value",
        "_normalize_aawm_sslmode",
        "_build_aawm_dsn",
        "_append_aawm_dsn_query_params",
        "_clean_non_empty_string",
        "_first_non_empty_string",
        "_coerce_string_dict",
    ],
    "cost_map": [
        "_load_bundled_model_cost_map",
        "_bundled_model_cost_casefold_lookup",
        "_lookup_bundled_model_cost_info",
        "_calculate_response_cost_from_bundled_model_cost_map",
    ],
    "identity_tenant_agent": [
        "_extract_claude_trace_agent_name",
        "_extract_claude_trace_user_identity_from_metadata_sources",
        "_extract_tenant_identity_from_kwargs",
        "_extract_tenant_identity_from_langfuse_trace_observation",
        "_is_agent_id_like",
        "_normalize_agent_id_identity",
        "_extract_agent_id_from_metadata_sources",
        "_extract_agent_id_from_kwargs",
        "_extract_agent_id_from_langfuse_trace_observation",
    ],
    "identity_repository": [
        "_normalize_repository_identity",
        "_normalize_repository_identity_from_absolute_path",
        "_extract_repository_identity_from_text",
        "_extract_repository_identity_from_value",
        "_extract_repository_identity_from_metadata_sources",
        "_extract_repository_identity_from_kwargs",
        "_extract_repository_identity_from_langfuse_trace_observation",
        "_is_codex_memory_workflow_request",
        "_apply_codex_memory_workflow_repository",
    ],
    "identity_runtime": [
        "_parse_client_identity_from_user_agent",
        "_extract_claude_code_version_from_metadata",
        "_clean_session_history_client_ip_candidate",
        "_canonical_session_history_client_ip",
    ],
    "agent_context": [
        "_extract_agent_name",
        "_is_native_codex_passthrough_context",
        "_is_codex_client_identity",
        "_is_codex_default_agent_context",
        "_is_codex_subagent_context",
        "_is_native_grok_passthrough_context",
        "_promote_grok_repository_trace_identity",
        "_promote_codex_repository_trace_user_id",
    ],
}

# Flat sample used by the AST-ownership / facade-identity tests.
_ALL_MOVED_NAMES: List[str] = [name for names in _MOVED_NAMES_BY_MODULE.values() for name in names]


def _parse_init_module() -> ast.Module:
    source = INIT_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(INIT_PATH))


def _function_def_names(tree: ast.Module) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _module_level_assign_targets(tree: ast.Module) -> Dict[str, ast.AST]:
    """Map assigned name -> the AST node of its (first) module-level Assign."""
    targets: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in targets:
                    targets[target.id] = node
    return targets


def test_moved_names_not_defined_in_init() -> None:
    """None of the A2-moved helper names may remain as `FunctionDef` in `__init__.py`.

    They may still appear as facade *assignments* (`name = submodule.name`),
    just not as function/async-function definitions. This is RED until the
    engineer performs the extraction.
    """
    tree = _parse_init_module()
    defined_functions = _function_def_names(tree)

    still_defined = sorted(
        name
        for name in _ALL_MOVED_NAMES
        if name in defined_functions
        # constants are never FunctionDefs to begin with; only check names
        # that are actually helper functions in the pre-move source.
        and name not in _MOVED_NAMES_BY_MODULE["constants"]
    )
    assert not still_defined, (
        "expected these A2-moved helper functions to no longer be defined "
        f"directly in __init__.py (they should live in their target "
        f"submodule with only a facade assignment remaining): {still_defined}"
    )


def test_facade_identity() -> None:
    """Sampled facade identity check per target module.

    `getattr(pkg, name) is getattr(submodule, name)` for every A2-moved name,
    once the target submodules exist. Import failures (submodule doesn't
    exist yet) are surfaced as an assertion failure naming the missing
    module, not a bare collection error, so the RED signal is legible.
    """
    import litellm.integrations.aawm_agent_identity as identity_pkg

    missing_modules = []
    mismatched = []
    for module_name, names in _MOVED_NAMES_BY_MODULE.items():
        try:
            submodule = importlib.import_module(f"litellm.integrations.aawm_agent_identity.{module_name}")
        except ModuleNotFoundError:
            missing_modules.append(module_name)
            continue
        for name in names:
            pkg_value = getattr(identity_pkg, name, None)
            sub_value = getattr(submodule, name, None)
            if pkg_value is None or sub_value is None or pkg_value is not sub_value:
                mismatched.append((module_name, name))

    assert not missing_modules, f"expected these A2 target submodules to exist: {missing_modules}"
    assert not mismatched, (
        "expected `getattr(pkg, name) is getattr(submodule, name)` for each "
        f"facade-bound name; mismatches (module, name): {mismatched}"
    )


def test_rebind_order_facades_before_record_install() -> None:
    """Every facade assignment for a moved helper precedes the
    `_bind_session_history_record_apis()` call in `__init__.py`'s AST.

    This preserves the `__globals__`-rebinding contract: record APIs are
    installed via `_bind_session_history_record_apis()`, and any free-name
    helper they reference by module-global lookup must already be bound to
    its facade value (the submodule's live function object) by that point.
    """
    tree = _parse_init_module()

    call_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_bind_session_history_record_apis"
        ):
            call_line = node.lineno
            break

    assert call_line is not None, "expected a call to _bind_session_history_record_apis() in __init__.py"

    # Restrict to moved *function* names (exclude the `constants` module's
    # names): constants are already plain module-level Assign nodes before
    # the move, which would make this assertion vacuously true pre-move.
    # Function facades (`name = submodule.name`) only exist as Assign nodes
    # AFTER the engineer performs the extraction, so this is the real RED
    # gate for this test.
    defined_functions = _function_def_names(tree)
    moved_function_names = [
        name for module_name, names in _MOVED_NAMES_BY_MODULE.items() if module_name != "constants" for name in names
    ]

    assign_targets = _module_level_assign_targets(tree)
    facade_names_present = [
        name for name in moved_function_names if name in assign_targets and name not in defined_functions
    ]

    assert facade_names_present, (
        "expected at least some A2-moved helper functions to appear as "
        "module-level facade assignments (`name = submodule.name`, with no "
        "remaining FunctionDef of the same name) in __init__.py once the "
        "extraction has landed (none found -- this is the expected RED "
        "state before the move)"
    )

    late_facades = [name for name in facade_names_present if assign_targets[name].lineno >= call_line]
    assert not late_facades, (
        "expected every facade assignment for a moved helper to precede the "
        f"_bind_session_history_record_apis() call (line {call_line}); "
        f"these facades assign at or after that call: {late_facades}"
    )


def test_moved_submodules_do_not_import_init_at_module_scope() -> None:
    """New leaf submodules must not import the `__init__` package at module
    scope (would risk a circular import); they should use the lazy
    `_get_litellm_module()`-style pattern if they need `litellm` itself.
    """
    missing_modules = []
    offending = []
    for module_name in _MOVED_NAMES_BY_MODULE:
        module_path = IDENTITY_PKG_DIR / f"{module_name}.py"
        if not module_path.is_file():
            missing_modules.append(module_name)
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "litellm.integrations.aawm_agent_identity":
                offending.append(module_name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "litellm.integrations.aawm_agent_identity":
                        offending.append(module_name)

    assert not missing_modules, f"expected these A2 target submodule files to exist: {missing_modules}"
    assert not offending, f"these submodules import the identity __init__ package at module scope: {offending}"
