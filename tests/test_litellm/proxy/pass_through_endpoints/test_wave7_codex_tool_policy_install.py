from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Callable

import pytest

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    codex_tool_policy as policy,
)


TARGET_PATH = (
    Path(__file__).parents[4]
    / "litellm"
    / "proxy"
    / "pass_through_endpoints"
    / "aawm_request_policy"
    / "codex_tool_policy.py"
)

EXPECTED_FACADES = (
    "_patch_codex_spawn_agent_tool_description",
    "_get_codex_core_tool_guidance",
    "_append_codex_core_tool_guidance_to_description",
    "_patch_codex_multi_agent_tool_search_description",
    "_patch_codex_core_tool_description",
    "_adapt_codex_custom_tool_definitions",
    "_adapted_custom_tool_call_ids",
    "_adapt_codex_custom_tool_input_items",
    "_adapt_codex_custom_tool_choice",
    "_adapt_codex_namespace_tool_definitions",
    "_adapt_codex_namespace_input_items",
    "_adapt_codex_namespace_tool_choice",
    "_openai_tool_choice_references_tool_type",
    "_get_codex_tool_policy_model_cost_candidates",
    "_get_unsupported_hosted_tool_types_for_model",
    "_get_unsupported_request_param_names_for_model",
    "_get_unsupported_input_item_types_for_model",
    "_get_rewrite_input_item_types_for_model",
    "_get_custom_tool_function_adapter_names_for_model",
    "_get_namespace_tool_function_adapter_names_for_model",
    "_add_codex_custom_tool_function_adapter_logging_metadata",
    "_adapt_codex_custom_tools_to_functions_from_request_body",
    "_add_codex_namespace_tool_function_adapter_logging_metadata",
    "_adapt_codex_namespace_tools_to_functions_from_request_body",
    "_add_codex_unsupported_hosted_tool_logging_metadata",
    "_add_tool_choice_without_tools_logging_metadata",
    "_drop_tool_choice_without_tools_from_request_body",
    "_add_codex_unsupported_request_param_logging_metadata",
    "_drop_unsupported_codex_request_params_from_request_body",
    "_add_codex_unsupported_input_item_logging_metadata",
    "_drop_unsupported_codex_input_items_from_request_body",
    "_drop_unsupported_codex_hosted_tools_from_request_body",
    "_add_codex_tool_description_patch_logging_metadata",
    "_apply_codex_tool_description_patches_to_request_body",
    "_stringify_grok_native_input_item_value",
    "_format_grok_native_function_call_input_message",
    "_format_grok_native_function_call_output_input_message",
    "_rewrite_grok_native_input_item_for_model_input",
    "_is_anthropic_grok_native_responses_adapter_body",
    "_add_grok_native_input_item_rewrite_logging_metadata",
    "_rewrite_grok_native_unsupported_input_items_from_request_body",
    "_rewrite_grok_native_unsupported_input_items_in_place",
)


def _normalize(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _callbacks(
    get_model_cost_map: Callable[[], dict[str, Any]],
    *,
    grok_normalization: Any = None,
    grok_runtime: Any = None,
) -> policy.CodexToolPolicyCallbacks:
    return policy.CodexToolPolicyCallbacks(
        normalize_tag_value=_normalize,
        dedupe_sorted=lambda values: sorted(set(values)),
        merge_metadata=lambda body, **_: body,
        build_span=lambda **kwargs: kwargs,
        get_model_cost_map=get_model_cost_map,
        normalize_grok_native_oauth_model=lambda _: None,
        is_oa_xai_model=lambda _: False,
        resolve_oa_xai_upstream_model=lambda model: model,
        normalize_kimi_model_name=lambda _: None,
        normalize_kimi_custom_tool_outputs=lambda body: body,
        grok_normalization=grok_normalization,
        grok_normalization_runtime=grok_runtime,
    )


class _GrokNormalization:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def rewrite_unsupported_input_items_from_request_body(
        self, runtime: Any, request_body: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.calls.append((runtime, request_body))
        return {"marker": self.marker}, [{"marker": self.marker}]


def test_install_publishes_same_object_facades_with_stable_signatures() -> None:
    host_globals: dict[str, Any] = {
        "_CODEX_TOOL_POLICY_CALLBACKS": _callbacks(lambda: {}),
        "_normalize_low_cardinality_tag_value": _normalize,
    }
    previous_accessors = policy._codex_tool_policy_runtime_accessors
    try:
        policy.install_codex_tool_policy_facades(host_globals)

        assert tuple(policy._HOST_FUNCTION_NAMES) == EXPECTED_FACADES
        for name in EXPECTED_FACADES:
            assert host_globals[name] is getattr(policy, name)
            assert not inspect.iscoroutinefunction(host_globals[name])

        format_signature = inspect.signature(
            host_globals["_format_grok_native_function_call_input_message"]
        )
        rewrite_signature = inspect.signature(
            host_globals["_rewrite_grok_native_input_item_for_model_input"]
        )
        assert format_signature.parameters["include_correlation_ref"].default is False
        assert rewrite_signature.parameters["include_correlation_ref"].default is False
    finally:
        policy._codex_tool_policy_runtime_accessors = previous_accessors


def test_installed_facades_resolve_replaced_host_runtime_at_call_time() -> None:
    first_grok = _GrokNormalization("first")
    second_grok = _GrokNormalization("second")
    live_cost_map = {
        "value": {
            "test-model": {
                "unsupported_hosted_tools": ["first_tool"],
            }
        }
    }
    host_globals: dict[str, Any] = {
        "_CODEX_TOOL_POLICY_CALLBACKS": _callbacks(
            lambda: live_cost_map["value"],
            grok_normalization=first_grok,
            grok_runtime="first-runtime",
        ),
        "_normalize_low_cardinality_tag_value": lambda _: "bash",
    }
    previous_accessors = policy._codex_tool_policy_runtime_accessors
    try:
        policy.install_codex_tool_policy_facades(host_globals)
        lookup = host_globals["_get_unsupported_hosted_tool_types_for_model"]
        rewrite = host_globals[
            "_rewrite_grok_native_unsupported_input_items_from_request_body"
        ]

        assert lookup("test-model") == {"first_tool"}
        live_cost_map["value"] = {
            "test-model": {
                "unsupported_hosted_tools": ["late_tool"],
            }
        }
        assert lookup("test-model") == {"late_tool"}
        assert rewrite({"input": []}) == (
            {"marker": "first"},
            [{"marker": "first"}],
        )
        assert first_grok.calls == [("first-runtime", {"input": []})]

        host_globals["_CODEX_TOOL_POLICY_CALLBACKS"] = _callbacks(
            lambda: {
                "test-model": {
                    "unsupported_hosted_tools": ["replacement_tool"],
                }
            },
            grok_normalization=second_grok,
            grok_runtime="second-runtime",
        )
        assert lookup("test-model") == {"replacement_tool"}
        assert rewrite({"input": ["replacement"]}) == (
            {"marker": "second"},
            [{"marker": "second"}],
        )
        assert second_grok.calls == [
            ("second-runtime", {"input": ["replacement"]})
        ]

        guidance = host_globals["_get_codex_core_tool_guidance"]
        assert guidance("ignored") == policy.CODEX_CORE_TOOL_GUIDANCE_BY_NAME["bash"]
        host_globals["_normalize_low_cardinality_tag_value"] = lambda _: "read"
        assert guidance("ignored") == policy.CODEX_CORE_TOOL_GUIDANCE_BY_NAME["read"]
    finally:
        policy._codex_tool_policy_runtime_accessors = previous_accessors


def test_owner_module_has_no_reverse_god_module_import() -> None:
    tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not [
        module
        for module in imported_modules
        if "llm_passthrough_endpoints" in module
    ]


def test_is_configured_guard_reflects_install_state() -> None:
    """is_codex_tool_policy_runtime_configured must be False before install."""
    previous_accessors = policy._codex_tool_policy_runtime_accessors
    try:
        policy._codex_tool_policy_runtime_accessors = None
        assert policy.is_codex_tool_policy_runtime_configured() is False

        host_globals: dict[str, Any] = {
            "_CODEX_TOOL_POLICY_CALLBACKS": _callbacks(lambda: {}),
            "_normalize_low_cardinality_tag_value": _normalize,
        }
        policy.install_codex_tool_policy_facades(host_globals)
        assert policy.is_codex_tool_policy_runtime_configured() is True
    finally:
        policy._codex_tool_policy_runtime_accessors = previous_accessors


def test_responses_endpoint_guards_underscore_facade_call() -> None:
    """The Responses endpoint must not unconditionally call an underscore facade.

    Regression: D1-591 Wave 7 consumer/install mismatch.  The endpoint must
    check is_codex_tool_policy_runtime_configured() before invoking any
    underscore-prefixed policy facade so that production boot without the
    god-module integrator does not raise RuntimeError.
    """
    endpoints_path = (
        Path(__file__).parents[4]
        / "litellm"
        / "proxy"
        / "response_api_endpoints"
        / "endpoints.py"
    )
    source = endpoints_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Collect all underscore-facade names imported from codex_tool_policy
    underscore_facade_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "codex_tool_policy" in node.module:
            for alias in node.names:
                if alias.name.startswith("_"):
                    underscore_facade_imports.add(alias.name)

    # The guard function must be imported
    assert "is_codex_tool_policy_runtime_configured" in source, (
        "endpoints.py must import is_codex_tool_policy_runtime_configured"
    )

    # Every underscore facade call must be inside an if-block that checks
    # is_codex_tool_policy_runtime_configured.  We verify structurally:
    # find all If nodes testing is_codex_tool_policy_runtime_configured and
    # confirm that underscore facade calls only appear within them.
    guarded_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check if the test is a call to is_codex_tool_policy_runtime_configured
            test = node.test
            is_guard = False
            if isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
                is_guard = test.func.id == "is_codex_tool_policy_runtime_configured"
            if is_guard:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        if child.func.id in underscore_facade_imports:
                            guarded_calls.add(child.func.id)

    # All underscore facade names that appear as calls must be guarded
    all_underscore_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in underscore_facade_imports:
                all_underscore_calls.add(node.func.id)

    assert all_underscore_calls == guarded_calls, (
        f"Unguarded underscore facade calls: {all_underscore_calls - guarded_calls}"
    )


# ---------------------------------------------------------------------------
# configure_and_install_codex_tool_policy one-call surface
# ---------------------------------------------------------------------------


def _make_host_deps(
    *,
    model_cost: dict[str, Any] | None = None,
    grok_normalization: Any = None,
    grok_runtime: Any = None,
) -> policy.CodexToolPolicyHostDeps:
    return policy.CodexToolPolicyHostDeps(
        normalize_tag_value=_normalize,
        dedupe_sorted=lambda values: sorted(set(values)),
        merge_metadata=lambda body, **_: body,
        build_span=lambda **kwargs: kwargs,
        get_model_cost_map=lambda: model_cost or {},
        normalize_grok_native_oauth_model=lambda _: None,
        is_oa_xai_model=lambda _: False,
        resolve_oa_xai_upstream_model=lambda model: model,
        normalize_kimi_model_name=lambda _: None,
        normalize_kimi_custom_tool_outputs=lambda body: body,
        grok_normalization=grok_normalization,
        grok_normalization_runtime=grok_runtime,
    )


class TestConfigureAndInstallOneCall:
    """Tests for configure_and_install_codex_tool_policy."""

    def test_publishes_all_42_facades_as_same_objects(self) -> None:
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps()
            )
            assert len(policy._HOST_FUNCTION_NAMES) == 42
            for name in policy._HOST_FUNCTION_NAMES:
                assert host_globals[name] is getattr(policy, name), (
                    f"{name} not same object"
                )
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_publishes_callbacks_and_normalize_into_host_globals(self) -> None:
        host_globals: dict[str, Any] = {}
        deps = _make_host_deps()
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(host_globals, deps)
            assert "_CODEX_TOOL_POLICY_CALLBACKS" in host_globals
            assert "_normalize_low_cardinality_tag_value" in host_globals
            cb = host_globals["_CODEX_TOOL_POLICY_CALLBACKS"]
            assert isinstance(cb, policy.CodexToolPolicyCallbacks)
            assert host_globals["_normalize_low_cardinality_tag_value"] is _normalize
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_callbacks_built_from_deps_fields(self) -> None:
        cost_map = {"test-model": {"unsupported_hosted_tools": ["web_search"]}}
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps(model_cost=cost_map)
            )
            cb = host_globals["_CODEX_TOOL_POLICY_CALLBACKS"]
            assert cb.get_model_cost_map() == cost_map
            assert cb.normalize_tag_value is _normalize
            assert cb.request_body_walk_max_depth == 64
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_configured_guard_true_after_install(self) -> None:
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy._codex_tool_policy_runtime_accessors = None
            assert policy.is_codex_tool_policy_runtime_configured() is False
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps()
            )
            assert policy.is_codex_tool_policy_runtime_configured() is True
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_fail_closed_unconfigured_raises_runtime_error(self) -> None:
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy._codex_tool_policy_runtime_accessors = None

            with pytest.raises(RuntimeError, match="not configured"):
                policy.get_codex_tool_policy_runtime_callbacks()
            with pytest.raises(RuntimeError, match="not configured"):
                policy.get_codex_tool_policy_runtime_normalize_tag_value()
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    @pytest.mark.parametrize(
        ("host_globals", "facade_name", "args"),
        (
            (
                {"_normalize_low_cardinality_tag_value": _normalize},
                "_get_unsupported_hosted_tool_types_for_model",
                ("test-model",),
            ),
            (
                {"_CODEX_TOOL_POLICY_CALLBACKS": _callbacks(lambda: {})},
                "_get_codex_core_tool_guidance",
                ("bash",),
            ),
        ),
    )
    def test_half_install_fails_closed_with_documented_runtime_error(
        self,
        host_globals: dict[str, Any],
        facade_name: str,
        args: tuple[Any, ...],
    ) -> None:
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.install_codex_tool_policy_facades(host_globals)
            with pytest.raises(
                RuntimeError,
                match="Codex tool-policy runtime accessors are not configured",
            ):
                host_globals[facade_name](*args)
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_configure_and_install_is_idempotent_on_reinstall(self) -> None:
        first_cost = {"m": {"unsupported_hosted_tools": ["first"]}}
        second_cost = {"m": {"unsupported_hosted_tools": ["second"]}}
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps(model_cost=first_cost)
            )
            published_before = {
                name: host_globals[name] for name in policy._HOST_FUNCTION_NAMES
            }
            lookup = host_globals["_get_unsupported_hosted_tool_types_for_model"]
            assert lookup("m") == {"first"}

            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps(model_cost=second_cost)
            )

            assert {
                name: host_globals[name] for name in policy._HOST_FUNCTION_NAMES
            } == published_before
            assert lookup("m") == {"second"}
            assert policy.is_codex_tool_policy_runtime_configured() is True
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_late_host_monkeypatch_replaces_callbacks_at_call_time(self) -> None:
        """Replacing _CODEX_TOOL_POLICY_CALLBACKS in host_globals is live."""
        cost_v1 = {"m": {"unsupported_hosted_tools": ["a"]}}
        cost_v2 = {"m": {"unsupported_hosted_tools": ["b"]}}
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps(model_cost=cost_v1)
            )
            lookup = host_globals["_get_unsupported_hosted_tool_types_for_model"]
            assert lookup("m") == {"a"}

            # Simulate god-module replacing callbacks at runtime
            host_globals["_CODEX_TOOL_POLICY_CALLBACKS"] = (
                policy.CodexToolPolicyCallbacks(
                    normalize_tag_value=_normalize,
                    dedupe_sorted=lambda v: sorted(set(v)),
                    merge_metadata=lambda body, **_: body,
                    build_span=lambda **kw: kw,
                    get_model_cost_map=lambda: cost_v2,
                    normalize_grok_native_oauth_model=lambda _: None,
                    is_oa_xai_model=lambda _: False,
                    resolve_oa_xai_upstream_model=lambda m: m,
                    normalize_kimi_model_name=lambda _: None,
                    normalize_kimi_custom_tool_outputs=lambda b: b,
                )
            )
            assert lookup("m") == {"b"}
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_late_normalize_tag_value_replacement(self) -> None:
        """Replacing _normalize_low_cardinality_tag_value is live."""
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps()
            )
            guidance = host_globals["_get_codex_core_tool_guidance"]
            # _normalize lowercases, so "BASH" -> "bash" -> found
            assert guidance("BASH") == policy.CODEX_CORE_TOOL_GUIDANCE_BY_NAME["bash"]

            # Replace with a function that always returns "read"
            host_globals["_normalize_low_cardinality_tag_value"] = lambda _: "read"
            assert guidance("BASH") == policy.CODEX_CORE_TOOL_GUIDANCE_BY_NAME["read"]
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_all_42_facades_installed_and_callable(self) -> None:
        """Verify all 42 expected facades are installed and callable."""
        host_globals: dict[str, Any] = {}
        previous = policy._codex_tool_policy_runtime_accessors
        try:
            policy.configure_and_install_codex_tool_policy(
                host_globals, _make_host_deps()
            )
            # Verify all 42 facades are installed
            assert len([k for k in host_globals if k in EXPECTED_FACADES]) == 42
            for name in EXPECTED_FACADES:
                assert name in host_globals, f"Missing facade: {name}"
                assert callable(host_globals[name]), f"Not callable: {name}"
        finally:
            policy._codex_tool_policy_runtime_accessors = previous

    def test_no_module_scope_god_import_in_policy_module(self) -> None:
        """Policy module must not import llm_passthrough_endpoints at any scope."""
        tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                assert "llm_passthrough_endpoints" not in (node.module or "")
