"""RR-054 OpenRouter retry and transport ownership tests.

OpenRouter-specific error classification belongs to
``providers/openrouter/error_shape.py``; cooldown policy and transport
execution belong to ``providers/openrouter/retry_transport.py``. The
pass-through route module may retain historical entrypoints only as thin
delegates using the exact runtime binding declared here.
"""

from __future__ import annotations

import ast
from pathlib import Path

from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    error_shape,
    retry_transport,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

PROVIDER_PATH = Path(retry_transport.__file__).resolve()
ERROR_SHAPE_PATH = Path(error_shape.__file__).resolve()
GOD_PATH = Path(lpe.__file__).resolve()
GOD_PROVIDER_ALIAS = "_anthropic_openrouter_retry_transport"
GOD_RUNTIME_NAME = "_ANTHROPIC_OPENROUTER_RETRY_TRANSPORT_RUNTIME"

# Historical route-module symbol -> provider-owned implementation.
OPENROUTER_PROVIDER_DELEGATES = {
    "_get_openrouter_adapter_rate_limit_key": "get_rate_limit_key",
    "_is_openrouter_adapter_free_model": "is_free_model",
    "_get_openrouter_adapter_wait_keys": "get_wait_keys",
    "_extract_openrouter_adapter_exception_status_code": (
        "extract_exception_status_code"
    ),
    "_extract_openrouter_adapter_error_payload": "extract_error_payload",
    "_extract_openrouter_adapter_provider_name": "extract_provider_name",
    "_extract_openrouter_adapter_retry_after_seconds": "extract_retry_after_seconds",
    "_extract_openrouter_adapter_raw_message": "extract_raw_message",
    "_is_openrouter_adapter_no_endpoint_candidate_error": (
        "is_no_endpoint_candidate_error"
    ),
    "_maybe_raise_openrouter_adapter_alias_probe_no_endpoint_unavailable": (
        "maybe_raise_alias_probe_no_endpoint_unavailable"
    ),
    "_is_openrouter_adapter_provider_raw_error": "is_provider_raw_error",
    "_extract_openrouter_adapter_error_headers": "extract_error_headers",
    "_get_openrouter_adapter_header_value": "get_header_value",
    "_extract_openrouter_adapter_reset_wait_seconds": "extract_reset_wait_seconds",
    "_is_openrouter_adapter_long_window_rate_limit": "is_long_window_rate_limit",
    "_get_openrouter_adapter_cooldown_keys": "get_cooldown_keys",
    "_get_openrouter_adapter_retry_wait_seconds": "get_retry_wait_seconds",
    "_get_openrouter_adapter_max_retries": "get_max_retries",
    "_get_openrouter_adapter_backoff_seconds": "get_backoff_seconds",
    "_get_openrouter_adapter_hidden_retry_budget_seconds": (
        "get_hidden_retry_budget_seconds"
    ),
    "_get_openrouter_adapter_post_failure_cooldown_seconds": (
        "get_post_failure_cooldown_seconds"
    ),
    "_maybe_raise_openrouter_adapter_failure_circuit_open": (
        "maybe_raise_failure_circuit_open"
    ),
    "_openrouter_adapter_open_failure_circuit": "open_failure_circuit",
    "_clear_openrouter_adapter_failure_circuit": "clear_failure_circuit",
    "_get_openrouter_adapter_active_cooldown_seconds": (
        "get_active_cooldown_seconds"
    ),
    "_wait_for_openrouter_adapter_cooldown_if_needed": "wait_for_cooldown_if_needed",
    "_set_openrouter_adapter_cooldown": "set_cooldown",
    "_run_openrouter_adapter_retry_loop": "run_retry_loop",
    "_perform_openrouter_completion_adapter_operation": (
        "perform_completion_operation"
    ),
    "_perform_openrouter_adapter_pass_through_request": (
        "perform_pass_through_request"
    ),
}

# These functions must contain the provider-specific decisions and execution
# structure, rather than existing as re-export or compatibility shells.
SUBSTANTIVE_CONTROL_FLOW_OWNERS = {
    "extract_error_payload",
    "get_backoff_seconds",
    "get_retry_wait_seconds",
    "is_long_window_rate_limit",
    "maybe_raise_failure_circuit_open",
    "open_failure_circuit",
    "get_active_cooldown_seconds",
    "wait_for_cooldown_if_needed",
    "run_retry_loop",
    "perform_pass_through_request",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    tree = ast.parse(_read(path), filename=str(path))
    assert isinstance(tree, ast.Module)
    return tree


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _provider_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    retry_functions = _functions(_parse(PROVIDER_PATH))
    error_functions = _functions(_parse(ERROR_SHAPE_PATH))
    duplicates = set(retry_functions).intersection(error_functions)
    assert not duplicates, f"OpenRouter provider modules duplicate {sorted(duplicates)}"
    return {**retry_functions, **error_functions}


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _calls(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _dotted_name(child.func)
            if name is not None:
                names.add(name)
    return names


def _call_nodes(node: ast.AST, dotted_name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _dotted_name(child.func) == dotted_name
    ]


def _meaningful_statements(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    return [
        statement
        for statement in node.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]


def test_rr054_openrouter_adapter_owns_retry_and_transport_implementations() -> None:
    provider_functions = _provider_functions()
    missing = sorted(
        set(OPENROUTER_PROVIDER_DELEGATES.values()) - set(provider_functions)
    )
    assert not missing, (
        "OpenRouter provider adapter must own the retry/transport surface; "
        f"missing definitions: {missing}"
    )

    for name in OPENROUTER_PROVIDER_DELEGATES.values():
        assert _meaningful_statements(provider_functions[name]), (
            f"OpenRouter provider owner {name} has no implementation"
        )


def test_rr054_openrouter_owner_contains_substantive_control_flow() -> None:
    provider_functions = _provider_functions()
    control_flow_types = (
        ast.AsyncFunctionDef,
        ast.AsyncWith,
        ast.For,
        ast.If,
        ast.Try,
        ast.While,
    )

    for name in sorted(SUBSTANTIVE_CONTROL_FLOW_OWNERS):
        function = provider_functions[name]
        owned_control_flow = [
            node
            for node in ast.walk(function)
            if node is not function and isinstance(node, control_flow_types)
        ]
        assert owned_control_flow, (
            f"OpenRouter retry/transport owner {name} lacks substantive "
            "provider control flow"
        )

    provider_source = _read(PROVIDER_PATH)
    for marker in (
        "AAWM_OPENROUTER_ADAPTER_MAX_RETRIES",
        "AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS",
        "AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS",
        "AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS",
        "caller_managed_hidden_retry=True",
        "retryable_upstream_status_codes or [429, 500, 502, 503, 504]",
    ):
        assert marker in provider_source, (
            f"OpenRouter provider owner is missing retry/transport marker {marker}"
        )


def test_rr054_openrouter_provider_does_not_back_import_route_implementation() -> None:
    provider_trees = (_parse(PROVIDER_PATH), _parse(ERROR_SHAPE_PATH))
    provider_source = _read(PROVIDER_PATH) + _read(ERROR_SHAPE_PATH)

    assert "llm_passthrough_endpoints" not in provider_source
    forbidden_calls = {
        call
        for provider_tree in provider_trees
        for call in _calls(provider_tree)
        if call.startswith(("lpe.", "_lp.", "llm_passthrough_endpoints."))
    }
    assert not forbidden_calls, (
        "OpenRouter provider ownership must be real, not a back-delegate to the "
        f"route module: {sorted(forbidden_calls)}"
    )


def test_rr054_god_file_imports_exact_openrouter_retry_transport_owner() -> None:
    god_tree = _parse(GOD_PATH)
    matching_imports = [
        alias
        for node in god_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == (
            "litellm.llms.anthropic.experimental_pass_through.providers"
            ".openrouter"
        )
        for alias in node.names
        if alias.name == "retry_transport" and alias.asname == GOD_PROVIDER_ALIAS
    ]
    assert matching_imports, (
        "llm_passthrough_endpoints.py must import OpenRouter retry_transport as "
        f"{GOD_PROVIDER_ALIAS}"
    )

    runtime_bindings = [
        node
        for node in god_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == GOD_RUNTIME_NAME
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    assert len(runtime_bindings) == 1, (
        "llm_passthrough_endpoints.py must define exactly one injected "
        f"{GOD_RUNTIME_NAME}"
    )


def test_rr054_openrouter_route_symbols_are_thin_provider_delegates() -> None:
    god_functions = _functions(_parse(GOD_PATH))

    for wrapper_name, provider_name in OPENROUTER_PROVIDER_DELEGATES.items():
        wrapper = god_functions.get(wrapper_name)
        assert wrapper is not None, (
            f"historical OpenRouter route entrypoint {wrapper_name} must remain "
            "as a compatibility delegate"
        )

        expected_call = f"{GOD_PROVIDER_ALIAS}.{provider_name}"
        calls = _calls(wrapper)
        assert expected_call in calls, (
            f"{wrapper_name} must delegate to {expected_call}; "
            f"calls={sorted(calls)}"
        )
        delegate_calls = _call_nodes(wrapper, expected_call)
        assert len(delegate_calls) == 1, (
            f"{wrapper_name} must call {expected_call} exactly once"
        )
        delegate_call = delegate_calls[0]
        assert delegate_call.args, (
            f"{wrapper_name} must pass {GOD_RUNTIME_NAME} as the first argument"
        )
        first_argument = delegate_call.args[0]
        assert isinstance(first_argument, ast.Name)
        assert first_argument.id == GOD_RUNTIME_NAME, (
            f"{wrapper_name} must use the shared injected runtime "
            f"{GOD_RUNTIME_NAME}, got {ast.dump(first_argument)}"
        )

        statements = _meaningful_statements(wrapper)
        assert len(statements) <= 2, (
            f"{wrapper_name} retains implementation logic instead of being a "
            f"thin provider delegate ({len(statements)} top-level statements)"
        )

        forbidden_control_flow = [
            node
            for node in ast.walk(wrapper)
            if isinstance(
                node,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.If,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                    ast.Match,
                ),
            )
        ]
        assert not forbidden_control_flow, (
            f"{wrapper_name} retains retry/transport control flow outside the "
            "OpenRouter provider adapter"
        )
