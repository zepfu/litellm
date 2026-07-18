"""RR-054 OpenCode Zen normalization ownership checks.

Finding #1 assigns provider-specific request and response shaping to
``providers/opencode_zen/normalization.py``. The pass-through god-file may
retain compatibility entrypoints and callback binding, but those entrypoints
must be thin delegates and the provider owner must not import the god-file.
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
NORMALIZATION_PATH = (
    REPO_ROOT
    / "litellm"
    / "llms"
    / "anthropic"
    / "experimental_pass_through"
    / "providers"
    / "opencode_zen"
    / "normalization.py"
)

PROVIDER_OWNED_NORMALIZERS = {
    "get_responses_tool_name",
    "strip_unsupported_responses_tools",
    "chat_message_role",
    "chat_tool_call_id",
    "chat_message_tool_call_ids",
    "chat_message_tool_result_id",
    "collect_following_tool_block",
    "sanitize_completion_messages_for_chat_completion",
    "responses_sse_event",
    "response_payload_for_stream",
    "message_item_for_stream",
    "completed_response_for_stream",
    "normalize_responses_stream_for_codex",
    "build_codex_streaming_response",
    "normalize_codex_request",
}

GOD_COMPAT_DELEGATES = {
    "_get_opencode_zen_responses_tool_name": "get_responses_tool_name",
    "_strip_opencode_zen_unsupported_responses_tools": (
        "strip_unsupported_responses_tools"
    ),
    "_opencode_zen_chat_message_role": "chat_message_role",
    "_opencode_zen_chat_tool_call_id": "chat_tool_call_id",
    "_opencode_zen_chat_message_tool_call_ids": "chat_message_tool_call_ids",
    "_opencode_zen_chat_message_tool_result_id": "chat_message_tool_result_id",
    "_collect_opencode_zen_following_tool_block": "collect_following_tool_block",
    "_sanitize_opencode_zen_completion_messages_for_chat_completion": (
        "sanitize_completion_messages_for_chat_completion"
    ),
    "_opencode_zen_responses_sse_event": "responses_sse_event",
    "_opencode_zen_response_payload_for_stream": "response_payload_for_stream",
    "_opencode_zen_message_item_for_stream": "message_item_for_stream",
    "_opencode_zen_completed_response_for_stream": (
        "completed_response_for_stream"
    ),
    "_normalize_opencode_zen_responses_stream_for_codex": (
        "normalize_responses_stream_for_codex"
    ),
    "_build_codex_opencode_zen_streaming_response": (
        "build_codex_streaming_response"
    ),
}

PROVIDER_SUBSTANCE_MARKERS = {
    "request transformation": (
        "transform_responses_api_request_to_chat_completion_request",
        "async_responses_api_session_handler",
        "class CodexRequestNormalization",
        "async def normalize_codex_request",
    ),
    "tool adjacency normalization": (
        "opencode_zen_chat_tool_adjacency_removed_orphan_tool_count",
        "opencode-zen-chat-tool-adjacency-sanitized",
    ),
    "unsupported tool normalization": (
        "opencode-zen-unsupported-tools-stripped",
        "opencode_zen_removed_unsupported_tool_count",
    ),
    "stream normalization": (
        "response.output_text.delta",
        "response.output_item.added",
        'yield "data: [DONE]\\n\\n"',
    ),
}

GOD_FORBIDDEN_ALGORITHM_MARKERS = {
    "opencode_zen_chat_tool_adjacency_removed_orphan_tool_count",
    "opencode-zen-unsupported-tools-stripped",
    'if event_type == "response.output_text.delta":',
    '"response.output_item.added"',
    'yield "data: [DONE]\\n\\n"',
}

SUBSTANTIVE_FUNCTION_MINIMUM_NODES = {
    "normalize_codex_request": 80,
    "normalize_responses_stream_for_codex": 140,
    "sanitize_completion_messages_for_chat_completion": 80,
    "strip_unsupported_responses_tools": 45,
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    tree = ast.parse(_read(path), filename=str(path))
    assert isinstance(tree, ast.Module)
    return tree


def _function_node(
    tree: ast.Module, name: str
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def _top_level_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _dotted_name(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _called_names(node: ast.AST) -> set[str]:
    return {
        dotted
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        if (dotted := _dotted_name(child.func)) is not None
    }


def _non_docstring_statements(
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


def _assigned_provider_attr(
    tree: ast.Module, target_name: str, provider_attr: str
) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in targets
        ):
            continue
        value = node.value
        return (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "_anthropic_opencode_zen_normalization"
            and value.attr == provider_attr
        )
    return False


def test_rr054_opencode_provider_owns_request_and_stream_normalizers() -> None:
    provider_tree = _parse(NORMALIZATION_PATH)
    provider_functions = _top_level_function_names(provider_tree)

    missing = sorted(PROVIDER_OWNED_NORMALIZERS - provider_functions)
    assert not missing, (
        "OpenCode Zen provider adapter does not own normalization functions: "
        f"{missing}"
    )

    for function_name in sorted(PROVIDER_OWNED_NORMALIZERS):
        node = _function_node(provider_tree, function_name)
        assert node is not None
        assert _non_docstring_statements(node), (
            f"{function_name} is an empty provider ownership shell"
        )


def test_rr054_opencode_normalization_owns_substantive_algorithms() -> None:
    provider_source = _read(NORMALIZATION_PATH)
    provider_tree = _parse(NORMALIZATION_PATH)

    assert "class Runtime" in provider_source
    assert "class CodexRequestNormalization" in provider_source
    for concern, markers in PROVIDER_SUBSTANCE_MARKERS.items():
        for marker in markers:
            assert marker in provider_source, (
                f"OpenCode Zen normalization is missing {concern} marker {marker!r}"
            )

    for function_name, minimum_nodes in SUBSTANTIVE_FUNCTION_MINIMUM_NODES.items():
        node = _function_node(provider_tree, function_name)
        assert node is not None
        node_count = sum(1 for _ in ast.walk(node))
        assert node_count >= minimum_nodes, (
            f"{function_name} has only {node_count} AST nodes; "
            "normalization.py must own the substantive algorithm"
        )


def test_rr054_opencode_normalization_has_no_god_file_back_import() -> None:
    provider_source = _read(NORMALIZATION_PATH)
    provider_tree = _parse(NORMALIZATION_PATH)

    assert "llm_passthrough_endpoints" not in provider_source
    for node in ast.walk(provider_tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.endswith(
                "llm_passthrough_endpoints"
            )
        elif isinstance(node, ast.Import):
            assert all(
                not alias.name.endswith("llm_passthrough_endpoints")
                for alias in node.names
            )


def test_rr054_opencode_algorithms_are_absent_from_god_file() -> None:
    god_source = _read(GOD_PATH)
    god_tree = _parse(GOD_PATH)
    opencode_function_names = {
        *GOD_COMPAT_DELEGATES,
        "_handle_codex_opencode_zen_adapter_route",
    }
    opencode_source = "\n".join(
        ast.get_source_segment(god_source, node) or ""
        for function_name in sorted(opencode_function_names)
        if (node := _function_node(god_tree, function_name)) is not None
    )

    for marker in GOD_FORBIDDEN_ALGORITHM_MARKERS:
        assert marker not in opencode_source, (
            f"god-file still owns OpenCode Zen normalization marker {marker!r}"
        )


def test_rr054_opencode_god_file_builds_normalization_runtime() -> None:
    god_tree = _parse(GOD_PATH)
    runtime_factory = _function_node(
        god_tree,
        "_get_anthropic_opencode_zen_normalization_runtime",
    )

    assert runtime_factory is not None
    assert (
        "_anthropic_opencode_zen_normalization.Runtime"
        in _called_names(runtime_factory)
    )
    statements = _non_docstring_statements(runtime_factory)
    assert len(statements) <= 2, (
        "OpenCode Zen normalization runtime binding must remain a thin factory"
    )


def test_rr054_opencode_god_compatibility_functions_are_thin_delegates() -> None:
    god_tree = _parse(GOD_PATH)

    for wrapper_name, provider_attr in GOD_COMPAT_DELEGATES.items():
        node = _function_node(god_tree, wrapper_name)
        assert node is not None or _assigned_provider_attr(
            god_tree,
            wrapper_name,
            provider_attr,
        ), f"missing OpenCode normalization delegate {wrapper_name}"
        if node is None:
            continue

        expected_call = f"_anthropic_opencode_zen_normalization.{provider_attr}"
        assert expected_call in _called_names(node), (
            f"{wrapper_name} must delegate to {expected_call}"
        )
        statements = _non_docstring_statements(node)
        assert len(statements) <= 2, (
            f"{wrapper_name} has {len(statements)} statements; "
            "compatibility wrappers must not retain normalization logic"
        )


def test_rr054_codex_opencode_route_delegates_request_normalization() -> None:
    god_tree = _parse(GOD_PATH)
    route = _function_node(god_tree, "_handle_codex_opencode_zen_adapter_route")

    assert isinstance(route, ast.AsyncFunctionDef)
    assert (
        "_anthropic_opencode_zen_normalization.normalize_codex_request"
        in _called_names(route)
    )
    forbidden_route_calls = {
        (
            "LiteLLMCompletionResponsesConfig."
            "transform_responses_api_request_to_chat_completion_request"
        ),
        "LiteLLMCompletionResponsesConfig.async_responses_api_session_handler",
        "_sanitize_opencode_zen_completion_messages_for_chat_completion",
        "_strip_opencode_zen_unsupported_responses_tools",
    }
    residual_calls = forbidden_route_calls & _called_names(route)
    assert not residual_calls, (
        "Codex OpenCode Zen route still performs request normalization directly: "
        f"{sorted(residual_calls)}"
    )
