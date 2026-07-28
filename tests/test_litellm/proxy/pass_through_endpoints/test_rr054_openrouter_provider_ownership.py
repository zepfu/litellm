"""RR-054 OpenRouter retry and transport ownership tests.

OpenRouter-specific error classification belongs to
``providers/openrouter/error_shape.py``; cooldown policy and transport
execution belong to ``providers/openrouter/retry_transport.py``. The
proxy OpenRouter runtime owns host dependency installation and publishes the
historical route entrypoints as same-object compatibility facades.
"""

from __future__ import annotations

import ast
from pathlib import Path

from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    error_shape,
    retry_transport,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.state import (
    alias_routing_state,
)
from litellm.proxy.pass_through_endpoints.providers.openrouter import (
    runtime as openrouter_runtime,
)

PROVIDER_PATH = Path(retry_transport.__file__).resolve()
ERROR_SHAPE_PATH = Path(error_shape.__file__).resolve()
GOD_PATH = Path(lpe.__file__).resolve()
GOD_RUNTIME_ALIAS = "_wave6b_openrouter_runtime"

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


def test_rr054_god_file_installs_openrouter_runtime_package_owner() -> None:
    god_tree = _parse(GOD_PATH)
    matching_imports = [
        alias
        for node in god_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        == (
            "litellm.proxy.pass_through_endpoints.providers.openrouter"
        )
        for alias in node.names
        if alias.name == "runtime" and alias.asname == GOD_RUNTIME_ALIAS
    ]
    assert matching_imports, (
        "llm_passthrough_endpoints.py must import the proxy OpenRouter runtime "
        f"package as {GOD_RUNTIME_ALIAS}"
    )

    install_calls = _call_nodes(god_tree, f"{GOD_RUNTIME_ALIAS}.install")
    assert len(install_calls) == 1, (
        "llm_passthrough_endpoints.py must install the OpenRouter package "
        "runtime exactly once"
    )
    assert install_calls[0].args
    install_arg = install_calls[0].args[0]
    assert isinstance(install_arg, ast.Call)
    assert _dotted_name(install_arg.func) == "globals"

    configured_runtime = openrouter_runtime._runtime
    assert configured_runtime is not None
    retry_runtime = configured_runtime.retry_transport_runtime
    assert isinstance(retry_runtime, retry_transport.Runtime)
    assert retry_runtime.rate_limit is alias_routing_state.openrouter_rate_limit
    assert (
        retry_runtime.failure_circuit_until_monotonic_by_key
        is alias_routing_state.openrouter_failure_circuit.until_monotonic_by_key
    )

    class HeaderError(Exception):
        upstream_headers = {"Retry-After": "4", "X-Test": "value"}

    headers = retry_runtime.extract_upstream_headers(HeaderError())
    assert dict(headers) == {"Retry-After": "4", "X-Test": "value"}
    assert retry_runtime.parse_retry_after_seconds_from_headers(headers) == 4.0
    assert retry_runtime.get_header_value(headers, "x-test") == "value"


def test_rr054_openrouter_route_symbols_are_installed_package_facades() -> None:
    god_functions = _functions(_parse(GOD_PATH))
    installed_names = set(openrouter_runtime._HOST_FUNCTION_NAMES)

    for wrapper_name in OPENROUTER_PROVIDER_DELEGATES:
        assert wrapper_name not in god_functions, (
            f"{wrapper_name} must be package-owned, not a god-module FunctionDef"
        )
        assert wrapper_name in installed_names, (
            f"OpenRouter install inventory is missing {wrapper_name}"
        )
        assert getattr(lpe, wrapper_name) is getattr(
            openrouter_runtime,
            wrapper_name,
        ), (
            f"{wrapper_name} must be the same object as "
            f"{GOD_RUNTIME_ALIAS}.{wrapper_name}"
        )


# --- Wave 6B extracted runtime forwarding contract -------------------------
#
# The extracted ``providers/openrouter/runtime.py`` keeps the historical
# god-module names as same-object facades.  Each retry/transport facade must
# forward to the canonical ``retry_transport`` owner and must thread the
# configured runtime through ``_retry_runtime()`` rather than constructing or
# importing a runtime of its own.

WAVE6B_RUNTIME_PATH = Path(openrouter_runtime.__file__).resolve()
RETRY_TRANSPORT_MODULE_ALIAS = "_anthropic_openrouter_retry_transport"


def _wave6b_runtime_functions() -> (
    dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
):
    return _functions(_parse(WAVE6B_RUNTIME_PATH))


def test_rr054_openrouter_extracted_runtime_forwards_to_retry_transport() -> None:
    runtime_functions = _wave6b_runtime_functions()

    for wrapper_name, provider_name in OPENROUTER_PROVIDER_DELEGATES.items():
        wrapper = runtime_functions.get(wrapper_name)
        assert wrapper is not None, (
            f"extracted OpenRouter runtime is missing facade {wrapper_name}"
        )

        expected_call = f"{RETRY_TRANSPORT_MODULE_ALIAS}.{provider_name}"
        calls = _calls(wrapper)
        assert expected_call in calls, (
            f"{wrapper_name} must forward to {expected_call}; "
            f"calls={sorted(calls)}"
        )
        assert len(_call_nodes(wrapper, expected_call)) == 1, (
            f"{wrapper_name} must call {expected_call} exactly once"
        )

        # The configured runtime must be threaded through _retry_runtime().
        assert "_retry_runtime" in calls, (
            f"{wrapper_name} must resolve the configured runtime via "
            f"_retry_runtime(); calls={sorted(calls)}"
        )
        retry_runtime_calls = _call_nodes(wrapper, "_retry_runtime")
        assert retry_runtime_calls, (
            f"{wrapper_name} must invoke _retry_runtime()"
        )
        forwarded_runtime = retry_runtime_calls[0]
        target_call = _call_nodes(wrapper, expected_call)[0]
        assert forwarded_runtime in [
            arg for arg in target_call.args
        ] or any(
            isinstance(arg, ast.Call)
            and _dotted_name(arg.func) == "_retry_runtime"
            for arg in target_call.args
        ), (
            f"{wrapper_name} must pass _retry_runtime() as the runtime "
            "argument to the retry_transport owner"
        )

        # Facades must not retain retry/transport control flow.
        forbidden_control_flow = [
            node
            for node in ast.walk(wrapper)
            if node is not wrapper
            and isinstance(
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
            f"{wrapper_name} retains retry/transport control flow instead of "
            "forwarding to the OpenRouter retry_transport owner"
        )


def test_rr054_openrouter_extracted_runtime_pins_configured_runtime() -> None:
    """_retry_runtime() must read the configured module singleton."""
    runtime_functions = _wave6b_runtime_functions()
    retry_runtime = runtime_functions.get("_retry_runtime")
    assert retry_runtime is not None, (
        "extracted OpenRouter runtime must define _retry_runtime()"
    )
    assert "_require_runtime" in _calls(retry_runtime), (
        "_retry_runtime() must resolve the configured runtime through "
        "_require_runtime()"
    )
    # It must surface the injected retry_transport_runtime field.
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "retry_transport_runtime"
        for node in ast.walk(retry_runtime)
    ), (
        "_retry_runtime() must return the configured retry_transport_runtime"
    )
