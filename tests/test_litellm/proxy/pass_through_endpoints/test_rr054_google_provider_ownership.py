"""RR-054 finding #1 ownership tests for Google and Antigravity.

The cross-cutting review requires provider request/response shaping and
provider process state to live below
``experimental_pass_through/providers/<provider>/``. The pass-through god file
may preserve private compatibility names, but only as thin delegates or direct
bindings to the provider-owned implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
GOD_PATH = (
    REPO_ROOT
    / "litellm"
    / "proxy"
    / "pass_through_endpoints"
    / "llm_passthrough_endpoints.py"
)
PROVIDERS_DIR = (
    REPO_ROOT
    / "litellm"
    / "llms"
    / "anthropic"
    / "experimental_pass_through"
    / "providers"
)
GOOGLE_ADAPTER_PATH = PROVIDERS_DIR / "google" / "adapter.py"
GOOGLE_SHAPING_PATH = PROVIDERS_DIR / "google" / "shaping.py"
GOOGLE_SHAPING_IMPLEMENTATION_PATHS = tuple(
    PROVIDERS_DIR / "google" / f"{module_name}.py"
    for module_name in (
        "content_selection",
        "content_compaction",
        "error_and_schema",
        "schema_and_prompt",
        "anthropic_replay",
        "tool_pairing",
        "request_assembly",
        "request_building",
        "tool_aliasing",
        "response_translation",
        "response_streaming",
        "request_preparation",
        "persisted_output",
    )
)
GOOGLE_PROCESS_CACHE_PATH = PROVIDERS_DIR / "google" / "process_cache.py"
ANTIGRAVITY_ADAPTER_PATH = PROVIDERS_DIR / "antigravity" / "adapter.py"

GOOGLE_REPRESENTATIVE_SHAPING_SEAMS = {
    "_apply_google_adapter_request_shape_policy",
    "_build_google_code_assist_request_from_completion_kwargs",
    "_prepare_anthropic_google_completion_adapter_request",
    "_prepare_codex_google_code_assist_adapter_request",
    "_collect_google_code_assist_response_from_stream",
}
ANTIGRAVITY_REPRESENTATIVE_SHAPING_SEAMS = {
    "_build_antigravity_native_headers",
    "_prepare_antigravity_request_body_for_passthrough",
    "_normalize_antigravity_endpoint_for_target",
}
GOOGLE_PROCESS_CACHE_SEAMS = {
    "_bound_google_adapter_token_cache",
    "_get_or_load_google_code_assist_project",
    "_get_google_adapter_semaphore",
    "_prime_google_code_assist_session",
}
GOOGLE_PROCESS_CACHE_STATE = {
    "_google_code_assist_project_cache",
    "_google_code_assist_project_lock",
    "_google_code_assist_prime_until_monotonic_by_key",
    "_google_code_assist_prime_quota_by_key",
    "_google_code_assist_prime_lock",
    "_google_adapter_semaphores",
    "_google_adapter_user_prompt_turn_counters",
    "_google_adapter_user_prompt_turn_lock",
    "_codex_google_code_assist_tool_call_name_cache",
    "_codex_google_code_assist_tool_call_arguments_cache",
}
SHAPING_VERB_PREFIXES = (
    "_add_",
    "_annotate_",
    "_append_",
    "_apply_",
    "_build_",
    "_collect_",
    "_compact_",
    "_deterministic_",
    "_drop_",
    "_ensure_",
    "_estimate_",
    "_extract_",
    "_has_",
    "_infer_",
    "_inject_",
    "_insert_",
    "_is_",
    "_iterate_",
    "_merge_",
    "_next_",
    "_normalize_",
    "_paired_",
    "_previous_",
    "_prepare_",
    "_repair_",
    "_replace_",
    "_restore_",
    "_sanitize_",
    "_selected_",
    "_split_",
    "_strip_",
    "_summarize_",
    "_translate_",
    "_trim_",
    "_unwrap_",
)


def _source(path: Path) -> str:
    assert path.is_file(), f"RR-054 provider owner is missing: {path}"
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _top_level_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments[node.target.id] = node.value
    return assignments


def _function(
    tree: ast.Module, name: str
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    return _top_level_functions(tree).get(name)


def _calls_module(function: ast.AST, module_name: str, function_name: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == module_name
            and node.func.attr == function_name
        ):
            return True
    return False


def _imported_submodule_alias(
    tree: ast.Module,
    *,
    package_suffix: str,
    submodule: str,
) -> Optional[str]:
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.endswith(package_suffix):
            continue
        for imported_name in node.names:
            if imported_name.name == submodule:
                return imported_name.asname or imported_name.name
    return None


def _provider_shaping_seams(
    god_tree: ast.Module,
    *,
    provider_marker: str,
    representative_seams: set[str],
) -> set[str]:
    discovered = {
        name
        for name in _top_level_functions(god_tree)
        if provider_marker in name
        and name.startswith(SHAPING_VERB_PREFIXES)
        and name not in GOOGLE_PROCESS_CACHE_SEAMS
    }
    assert representative_seams.issubset(discovered), (
        f"shaping discovery lost required {provider_marker} seams: "
        f"{sorted(representative_seams - discovered)}"
    )
    return discovered


def _operational_statements(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    return [
        statement
        for statement in function.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]


def _assert_substantive_owners(
    paths: tuple[Path, ...],
    required_functions: set[str],
) -> None:
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for path in paths:
        source = _source(path)
        assert "llm_passthrough_endpoints" not in source, (
            f"{path} must own provider logic instead of calling back into the god file"
        )
        functions.update(_top_level_functions(_tree(path)))
    missing = required_functions - functions.keys()
    assert not missing, (
        f"{paths} do not own required functions: {sorted(missing)}"
    )

    for name in sorted(required_functions):
        function = functions[name]
        assert len(list(ast.walk(function))) >= 10, (
            f"{name} is too small to prove real implementation ownership"
        )


def _assert_thin_god_delegates(
    *,
    god_tree: ast.Module,
    function_names: set[str],
    module_alias: str,
) -> None:
    for name in sorted(function_names):
        function = _function(god_tree, name)
        assert function is not None, (
            f"compatibility entry point {name} must remain available on the god file"
        )
        statements = _operational_statements(function)
        assert len(statements) <= 3, (
            f"{name} still owns implementation in llm_passthrough_endpoints.py; "
            f"expected at most three delegate statements, found {len(statements)}"
        )
        assert not any(
            isinstance(
                node,
                (
                    ast.For,
                    ast.While,
                    ast.Try,
                    ast.If,
                    ast.Match,
                    ast.With,
                    ast.AsyncWith,
                ),
            )
            for node in ast.walk(function)
        ), f"{name} must be a thin provider delegate without local orchestration"
        assert _calls_module(function, module_alias, name), (
            f"{name} must delegate to {module_alias}.{name}"
        )


def test_rr054_google_and_antigravity_provider_packages_are_real() -> None:
    """Required owners exist as provider packages, not compatibility re-exports."""
    expected_paths = (
        PROVIDERS_DIR / "google" / "__init__.py",
        GOOGLE_ADAPTER_PATH,
        GOOGLE_SHAPING_PATH,
        *GOOGLE_SHAPING_IMPLEMENTATION_PATHS,
        GOOGLE_PROCESS_CACHE_PATH,
        PROVIDERS_DIR / "antigravity" / "__init__.py",
        ANTIGRAVITY_ADAPTER_PATH,
    )
    missing = [str(path.relative_to(REPO_ROOT)) for path in expected_paths if not path.is_file()]
    assert not missing, f"missing RR-054 provider ownership modules: {missing}"

    god_tree = _tree(GOD_PATH)
    google_shaping_seams = _provider_shaping_seams(
        god_tree,
        provider_marker="google",
        representative_seams=GOOGLE_REPRESENTATIVE_SHAPING_SEAMS,
    )
    antigravity_shaping_seams = _provider_shaping_seams(
        god_tree,
        provider_marker="antigravity",
        representative_seams=ANTIGRAVITY_REPRESENTATIVE_SHAPING_SEAMS,
    )
    _assert_substantive_owners(
        GOOGLE_SHAPING_IMPLEMENTATION_PATHS,
        google_shaping_seams,
    )
    _assert_substantive_owners(
        (ANTIGRAVITY_ADAPTER_PATH,),
        antigravity_shaping_seams,
    )
    _assert_substantive_owners(
        (GOOGLE_PROCESS_CACHE_PATH,),
        GOOGLE_PROCESS_CACHE_SEAMS,
    )


def test_rr054_google_and_antigravity_god_functions_are_thin_delegates() -> None:
    """The legacy private surface delegates while behavior remains callable."""
    god_tree = _tree(GOD_PATH)
    google_shaping_alias = _imported_submodule_alias(
        god_tree,
        package_suffix="experimental_pass_through.providers.google",
        submodule="shaping",
    )
    google_process_cache_alias = _imported_submodule_alias(
        god_tree,
        package_suffix="experimental_pass_through.providers.google",
        submodule="process_cache",
    )
    antigravity_adapter_alias = _imported_submodule_alias(
        god_tree,
        package_suffix="experimental_pass_through.providers.antigravity",
        submodule="adapter",
    )
    assert google_shaping_alias is not None
    assert google_process_cache_alias is not None
    assert antigravity_adapter_alias is not None

    google_shaping_seams = _provider_shaping_seams(
        god_tree,
        provider_marker="google",
        representative_seams=GOOGLE_REPRESENTATIVE_SHAPING_SEAMS,
    )
    antigravity_shaping_seams = _provider_shaping_seams(
        god_tree,
        provider_marker="antigravity",
        representative_seams=ANTIGRAVITY_REPRESENTATIVE_SHAPING_SEAMS,
    )

    _assert_thin_god_delegates(
        god_tree=god_tree,
        function_names=google_shaping_seams,
        module_alias=google_shaping_alias,
    )
    _assert_thin_god_delegates(
        god_tree=god_tree,
        function_names=antigravity_shaping_seams,
        module_alias=antigravity_adapter_alias,
    )
    _assert_thin_god_delegates(
        god_tree=god_tree,
        function_names=GOOGLE_PROCESS_CACHE_SEAMS,
        module_alias=google_process_cache_alias,
    )


def test_rr054_google_process_cache_state_has_one_real_owner() -> None:
    """Token-keyed maps and locks are constructed only by the provider cache."""
    owner_tree = _tree(GOOGLE_PROCESS_CACHE_PATH)
    god_tree = _tree(GOD_PATH)
    owner_assignments = _top_level_assignments(owner_tree)
    god_assignments = _top_level_assignments(god_tree)

    missing = GOOGLE_PROCESS_CACHE_STATE - owner_assignments.keys()
    assert not missing, (
        "Google process-cache owner is missing state: "
        f"{sorted(missing)}"
    )

    for name in sorted(GOOGLE_PROCESS_CACHE_STATE):
        god_value = god_assignments.get(name)
        assert isinstance(god_value, ast.Attribute), (
            f"{name} must be a compatibility binding to provider-owned state"
        )
        assert isinstance(god_value.value, ast.Name)
        process_cache_alias = _imported_submodule_alias(
            god_tree,
            package_suffix="experimental_pass_through.providers.google",
            submodule="process_cache",
        )
        assert process_cache_alias is not None
        assert god_value.value.id == process_cache_alias
        assert god_value.attr == name


def test_rr054_god_file_has_no_google_antigravity_owner_only_helpers() -> None:
    """Every discovered shaping helper is covered by a provider owner contract."""
    god_tree = _tree(GOD_PATH)
    google_shaping_seams = _provider_shaping_seams(
        god_tree,
        provider_marker="google",
        representative_seams=GOOGLE_REPRESENTATIVE_SHAPING_SEAMS,
    )
    antigravity_shaping_seams = _provider_shaping_seams(
        god_tree,
        provider_marker="antigravity",
        representative_seams=ANTIGRAVITY_REPRESENTATIVE_SHAPING_SEAMS,
    )
    assert len(google_shaping_seams) > len(GOOGLE_REPRESENTATIVE_SHAPING_SEAMS)
    assert len(antigravity_shaping_seams) >= len(
        ANTIGRAVITY_REPRESENTATIVE_SHAPING_SEAMS
    )
