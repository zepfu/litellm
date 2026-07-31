"""Wave 5C decomposition ownership and baseline-parity contracts.

Baseline: ``79fc94c3a5``.

The 50 frozen functions are owned by three focused modules. Their normalized
AST must remain identical to the baseline after accounting only for documented
extraction mechanics: configured callback names, fail-fast callback assertions,
and one function-local FastAPI/status import.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints import aawm_alias_routing as package
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    attempt_records,
    cooldown_apply,
    cooldown_state,
    error_signals,
    selection,
)
from litellm.proxy.pass_through_endpoints.providers.openrouter import (
    runtime as openrouter_runtime,
)

GOD_PATH = Path(lpe.__file__).resolve()
PACKAGE_PATH = GOD_PATH.parent / "aawm_alias_routing"
ARCHITECTURE_PATH = GOD_PATH.parent / "architecture.md"

FROZEN_SYMBOLS: dict[str, tuple[str, ...]] = {
    "error_signals": (
        "_codex_auto_agent_error_text",
        "_add_codex_auto_agent_text_error_tokens",
        "_extract_codex_auto_agent_error_tokens",
        "_is_codex_auto_agent_durable_cooldown_error_class",
        "_is_codex_auto_agent_spark_candidate",
        "_is_codex_auto_agent_grok_4_5_candidate",
        "_is_codex_auto_agent_native_grok_4_5_candidate",
        "_is_codex_auto_agent_xai_candidate",
        "_is_kimi_code_auto_agent_candidate",
        "_get_kimi_code_managed_account_cooldown_key",
        "_get_safe_kimi_code_probe_failure_metadata",
        "_classify_kimi_code_auto_agent_probe_failure",
        "_build_safe_kimi_code_selection_telemetry",
        "_is_codex_auto_agent_transient_internal_error_class",
        "_get_codex_auto_agent_native_grok_continuation_transient_max_attempts",
        "_get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds",
        "_is_codex_auto_agent_native_grok_continuation_transient_retry_eligible",
        "_build_codex_auto_agent_native_grok_continuation_retry_metadata",
        "_plan_codex_auto_agent_native_grok_continuation_transient_retry",
        "_get_codex_auto_agent_cooldown_scope",
        "_get_codex_auto_agent_candidate_cooldown_scope",
        "_is_codex_auto_agent_grok_build_usage_balance_exhausted",
        "_is_codex_auto_agent_grok_personal_team_spending_limit",
        "_is_codex_auto_agent_grok_account_quota_candidate",
        "_get_codex_auto_agent_grok_account_quota_lane_cooldown_key",
        "_is_codex_auto_agent_grok_account_quota_exhaustion",
        "_classify_codex_auto_agent_retryable_exhaustion",
        "_is_codex_auto_agent_retryable_exhaustion",
        "_parse_codex_auto_agent_header_wait_seconds",
        "_get_codex_auto_agent_cooldown_seconds",
        "_iter_codex_auto_agent_error_blocks",
        "_extract_codex_auto_agent_error_type_and_code",
        "_get_codex_auto_agent_source_error_summary",
        "_extract_adapter_exception_detail",
        "_extract_adapter_error_payloads",
        "_extract_adapter_exception_status_code",
        "_extract_openrouter_adapter_raw_message",
        "_is_openrouter_adapter_provider_raw_error",
        "_extract_adapter_upstream_headers",
        "_get_adapter_header_value",
        "_parse_retry_after_seconds_from_headers",
        "_parse_rate_limit_reset_wait_seconds_from_headers",
        "_extract_embedded_json_payload_candidates",
        "_parse_json_payloads_from_text_candidates",
    ),
    "cooldown_apply": (
        "_resolve_auto_agent_cooldown_publication_plan",
        "_persist_codex_cooldown_durable",
        "_persist_anthropic_cooldown_durable",
        "_apply_auto_agent_alias_cooldown",
        "_apply_codex_auto_agent_alias_cooldown",
        "_apply_read_pilot_gated_cooldown",
        "_apply_anthropic_auto_agent_alias_cooldown",
        "_set_codex_auto_agent_candidate_cooldowns",
        "resolve_lane_identity_hash",
        "execute_cooldown_publication_transaction",
    ),
    "attempt_records": (
        "_update_codex_auto_agent_retryable_attempt_record",
        "_record_auto_agent_alias_attempt_started",
        "_record_read_pilot_cooldown_evidence",
        "_record_auto_agent_alias_attempt_failure",
        "_extract_codex_reasoning_effort",
        "_get_codex_reasoning_effort_ceiling",
        "_normalize_codex_reasoning_effort_for_resolved_route",
        "_add_codex_auto_agent_alias_metadata",
        "_add_anthropic_auto_agent_alias_metadata",
    ),
}

TARGET_MODULES = {
    "error_signals": error_signals,
    "cooldown_apply": cooldown_apply,
    "attempt_records": attempt_records,
}

CANONICAL_HOST_FACADE_OVERRIDES = {
    "_extract_openrouter_adapter_raw_message": (
        openrouter_runtime._extract_openrouter_adapter_raw_message
    ),
    "_is_openrouter_adapter_provider_raw_error": (
        openrouter_runtime._is_openrouter_adapter_provider_raw_error
    ),
}

# SHA-256 of each baseline function after the narrowly documented AST
# normalization in _BaselineNormalizer. Function signatures remain included.
BASELINE_NORMALIZED_SHA256 = {
    "_codex_auto_agent_error_text": "5fd757caa5588fed578b0952d18f8be66e1a3a058466455586b7589ece142344",
    "_add_codex_auto_agent_text_error_tokens": "d18afa48ee3320fc4b56ede0b6d657d2d44bd64b022694aae68ecad72500ce23",
    "_extract_codex_auto_agent_error_tokens": "c990d4fb23c81389bdb491339e37c55f2a38cb75932bd738a2f2bc35ff77f566",
    "_is_codex_auto_agent_durable_cooldown_error_class": "30784d0a5f07f6175a4a8f4232447f54589a696ab08d81b295a9b876add5de6e",
    "_is_codex_auto_agent_spark_candidate": "babb62af91da25fbbc7d39029399752e76d930449237f90f20de83649c31ccf5",
    "_is_codex_auto_agent_grok_4_5_candidate": "6013f5f7029cf137a4fba30e0ec22bffcc6ec4cb85fd8cb7ea7714cf2cc83f68",
    "_is_codex_auto_agent_native_grok_4_5_candidate": "0a010bafe81aa7fe16e6e5d93cfacc53e5a3a51158ffc12d790d59fa0cee7c8a",
    "_is_codex_auto_agent_xai_candidate": "7805ba2b2d414c920f883c5417b85147bbe49bcc66a202b7d4124082b231c774",
    "_is_kimi_code_auto_agent_candidate": "be78fbcf4c9ca4c5229399c3f5f7b718ceb0e6c749624bbe82604d4086f06de0",
    "_get_kimi_code_managed_account_cooldown_key": "a17f477d73cc29cedde92a0bc536716704f8089e220bd1648e3bfbf8f65bc93e",
    "_get_safe_kimi_code_probe_failure_metadata": "ba6c555adb9d6a007f21d9abc48ba01c8a623b5e1f989dd4e801ba33919ae330",
    "_classify_kimi_code_auto_agent_probe_failure": "8520dee58cfc6b32fd1967ed26a8b4ada1d85e586f263f12b1bfe0b7c6855ec5",
    "_build_safe_kimi_code_selection_telemetry": "9f5d5576b7118c6d77b87abbb5ded1c20bcbe6183be36c68a85f88df53d539c2",
    "_is_codex_auto_agent_transient_internal_error_class": "173b8ee2e15c46a8a290ff8e7dcdd63379d1dbbc1eb1008a4741cf34230f5773",
    "_get_codex_auto_agent_native_grok_continuation_transient_max_attempts": "516ed384cd4ad753d78cd99cbea569fd64a88650e8fd15bfbec0db0ec5809d79",
    "_get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds": "b87a06aa6b53c5112182ab2c8354adcbada2bbbdc92edf7383c94ef71bf30ba0",
    "_is_codex_auto_agent_native_grok_continuation_transient_retry_eligible": "6e7a6847f88c736980bf65e374eed0b8c778719157cefe35d1d3c8fc448809ed",
    "_build_codex_auto_agent_native_grok_continuation_retry_metadata": "e64af0c1a38574bc56bf9c5c19498934558230427404432198bfbfa18fd6060d",
    "_plan_codex_auto_agent_native_grok_continuation_transient_retry": "e415a0d9db6b8f13994b13240508b50b9da2d4b534933dccd1b0791facf46476",
    "_get_codex_auto_agent_cooldown_scope": "3edbe7a558aaf8e097f73f52ce5d38f2f73e76943b3487f7895c2b3713a47b80",
    "_get_codex_auto_agent_candidate_cooldown_scope": "55e471e6191c2f043f11ae92edf54cecd6af783326c831de44c7c8ac3c5c7acc",
    "_is_codex_auto_agent_grok_build_usage_balance_exhausted": "e2d8ef75ed78a931d1ee56061fe56f9278b5e5e7297c4ef82f31cff7a757a569",
    "_is_codex_auto_agent_grok_personal_team_spending_limit": "00d3b86a7c80135ee690988103dbe124653ad81e17d626275859d7e72030744f",
    "_is_codex_auto_agent_grok_account_quota_candidate": "1f655e66bacc29e05bad09e60a320df791aae04a6fb8f0148cced619a5c57a9c",
    "_get_codex_auto_agent_grok_account_quota_lane_cooldown_key": "b7db42909cfec673676444e85f3bd48f23242794476433f0419ab24bc46d694a",
    "_is_codex_auto_agent_grok_account_quota_exhaustion": "59eb9a900b9b4e690e0931e8edc718690bc80ca62015e640a1c1ba62dd8e4065",
    "_classify_codex_auto_agent_retryable_exhaustion": "d652c7332dbe81ca41838739ff4148294b8f11324255c5cc666819898b5607dc",
    "_is_codex_auto_agent_retryable_exhaustion": "b7765d3e4bfed171e76b25cd908f7a290e0ce47c329ca8d1834774e55e915031",
    "_parse_codex_auto_agent_header_wait_seconds": "d207fcec6808d4163d2c849b783fcf9196ece471a3b595995c2638b6a0204839",
    "_get_codex_auto_agent_cooldown_seconds": "f75184fe1edbdb32127a70a4036523f27883060de461d03364ca03f1b048e987",
    "_iter_codex_auto_agent_error_blocks": "b583aeb1d182041b2e3a1877317609965b23cc09e8a3630869f161c6043a5f97",
    "_extract_codex_auto_agent_error_type_and_code": "6560a0cf446bdf9ecf060a4a9e47df3ffd7fae326d71dc73d465e4ae854f791f",
    "_get_codex_auto_agent_source_error_summary": "374a4665762d8fced81f8063349aac15053119abcaea2879d2f8a9d194e6aa2e",
    "_extract_adapter_exception_detail": "45b9f4f5da5f2687bb9a8da2aaa4d60071c856188764ece36bd8c4424092d0f1",
    "_extract_adapter_error_payloads": "78a09ec62ee6671b4b6c573a0b89b245cbfffe4c3e2359b733685581dc2f981a",
    "_extract_adapter_exception_status_code": "a5f29723c1df2761cd14c633a5db24cf1c50986f6d0d106360a606224dc87e9f",
    "_extract_openrouter_adapter_raw_message": "7d205c9e1f4f76cafb5dda0575014fead4af616299007d54181f64da4eea54b3",
    "_is_openrouter_adapter_provider_raw_error": "68e0510ef85ab0d7810871097a110d4fcd15699ff45722282a8432e9f9f8d583",
    "_extract_adapter_upstream_headers": "826721de8790a64cffc073ccf5d42ef84040724608292b1dde00fdfd79f8c34c",
    "_get_adapter_header_value": "36bfd010283a0dafab49732d8af38162e3833d22409e24f4560a7914988b40fb",
    "_parse_retry_after_seconds_from_headers": "9919e2dc963b11d67fe70be0ae41acaa0a40960b4bcb633f7bbe4fececa38f33",
    "_parse_rate_limit_reset_wait_seconds_from_headers": "62b010f08927b8da1adbda76b4f7889f0414cbcdabd668556bade220dc9b85ea",
    "_extract_embedded_json_payload_candidates": "5003f30b814c4737c722f36513a6cf794f035f40d307e41aaa5cf5e04857ece8",
    "_parse_json_payloads_from_text_candidates": "549719c5b92d44b4fa6de2d4a7fe8e1e3e65d3a779d15241732936250fe197f1",
    "_resolve_auto_agent_cooldown_publication_plan": "193ef7f7c521581d70e99785ef0f2f281438b06d1467fbc620dfa228be6c8544",
    "_persist_codex_cooldown_durable": "87fd93555974db1ad42b17df76ea5cafa6e55f97770a6a1b50cc252066f61cd3",
    "_persist_anthropic_cooldown_durable": "bb6c8840fee8982f1994125976c7c6c4daf98f2216320ef21c6780c01bc04b6c",
    "_apply_auto_agent_alias_cooldown": "c2c77e8c9560f0be7912bda50e8c64f9b7714b1e81912e683823bc56483e5390",
    "_apply_codex_auto_agent_alias_cooldown": "d9c241385cd1a373c126b9499e59bbf194eb57c4b8da417b2326c6433dea5547",
    "_apply_read_pilot_gated_cooldown": "b6a79d5edff15084fc582ef525c094caa62e0f2fb0246a6c30c252e02847a09e",
    "_apply_anthropic_auto_agent_alias_cooldown": "2968467c04d3d4cce1f51c6243dccc86bf796fee86f35a1bd26ba97cd12dbe3f",
    "_set_codex_auto_agent_candidate_cooldowns": "437b2906f2ce8e86e69cf6073daff236ad52b5180bf0ce315870a965e0fafca2",
    "resolve_lane_identity_hash": "c13e2b6dbed807a0e8d125eb7cb78b6dbf698e2bbbbfe7cb4460b86e3e23e2ca",
    "execute_cooldown_publication_transaction": "cb82932006c8b5d114fcd850570f16e405e445d03887ebd3e8f40db13fc06afb",
    "_update_codex_auto_agent_retryable_attempt_record": "ecd632224a453568ea22fe24b001aeb437eb2a8e7f87c6eeacd60b99bc867bb4",
    "_record_auto_agent_alias_attempt_started": "7be5154bf3ff55cb6e3a9d515095b7909fac3ee698fbf1f033a744a85ba65953",
    "_record_read_pilot_cooldown_evidence": "5cc58f31d0074d19202949c1e12fae9c4576ab2fc8edfd26174b7f15c877ba8e",
    "_record_auto_agent_alias_attempt_failure": "169d402301c35c1b156236204d14c362c8e842e312fb83ffd48fa9711ee93921",
    "_extract_codex_reasoning_effort": "ab143775e8132aa73f58ff575e97d2a85f1fd319d88816671cc93441a4e03be5",
    "_get_codex_reasoning_effort_ceiling": "f3d94c2bef96dd86af31831869f0195067d9a0f988f540535cf7f63dd4b200a6",
    "_normalize_codex_reasoning_effort_for_resolved_route": "971bec09729674f5d792e982424aa7c77f4bc012ce140d58ef0bef9e2f6fd8d1",
    "_add_codex_auto_agent_alias_metadata": "1022d4a668a86656ff04d9ce03b94fcfc3e017b54931aff2892e33abb84dada5",
    "_add_anthropic_auto_agent_alias_metadata": "6d6f335a3691a84a43e306e8acb8eac129799fe5bdf988d4c77b3a1f13938ec1",
}

# Baseline has no callback assertions. These exact additions are the permitted
# fail-fast extraction difference, and their count is locked per function.
EXPECTED_ASSERT_COUNTS = {
    "_is_codex_auto_agent_durable_cooldown_error_class": 1,
    "_is_codex_auto_agent_grok_build_usage_balance_exhausted": 1,
    "_is_codex_auto_agent_grok_personal_team_spending_limit": 1,
    "_classify_codex_auto_agent_retryable_exhaustion": 2,
    "_get_codex_auto_agent_cooldown_seconds": 1,
    "_get_codex_auto_agent_source_error_summary": 1,
    "_resolve_auto_agent_cooldown_publication_plan": 4,
    "_persist_codex_cooldown_durable": 1,
    "_persist_anthropic_cooldown_durable": 1,
    "_apply_auto_agent_alias_cooldown": 6,
    "_apply_codex_auto_agent_alias_cooldown": 1,
    "_apply_read_pilot_gated_cooldown": 1,
    "_apply_anthropic_auto_agent_alias_cooldown": 1,
    "_update_codex_auto_agent_retryable_attempt_record": 6,
    "_record_auto_agent_alias_attempt_started": 2,
    "_record_read_pilot_cooldown_evidence": 5,
    "_record_auto_agent_alias_attempt_failure": 4,
    "_get_codex_reasoning_effort_ceiling": 4,
    "_normalize_codex_reasoning_effort_for_resolved_route": 1,
    "_add_codex_auto_agent_alias_metadata": 4,
    "_add_anthropic_auto_agent_alias_metadata": 3,
}

SELECTION_OWNED_EXCLUSIONS = {
    "_apply_request_local_cooldown_from_plan",
    "_apply_codex_auto_agent_grok_account_lane_cooldown",
}

COOLDOWN_STATE_OWNED_EXCLUSIONS = {
    "_publish_codex_cooldown_memory",
    "_publish_anthropic_cooldown_memory",
}

_NAME_NORMALIZATION = {
    "_get_candidate_cooldown_scope": "_get_codex_auto_agent_candidate_cooldown_scope",
    "_get_kimi_managed_account_cooldown_key": "_get_kimi_code_managed_account_cooldown_key",
    "_get_grok_account_quota_lane_cooldown_key": "_get_codex_auto_agent_grok_account_quota_lane_cooldown_key",
    "_get_request_local_cooldown_key": "_get_codex_auto_agent_request_local_cooldown_key",
    "_set_request_local_cooldown": "_set_codex_auto_agent_request_local_cooldown",
    "_exclude_request_local_candidate": "_exclude_codex_auto_agent_request_local_candidate",
    "_set_codex_cooldown": "_set_codex_auto_agent_cooldown",
    "_set_anthropic_cooldown": "_set_anthropic_auto_agent_cooldown",
    "_write_durable_payload": "_write_aawm_alias_routing_durable_payload",
    "_read_pilot_gate": "_read_pilot_cooldown_gate",
    "_state_manager": "_alias_routing_state",
    "http_status": "status",
}

_ATTRIBUTE_NORMALIZATION = {
    "_aawm_alias_interfaces.CooldownPublicationPlan": "_CooldownPublicationPlan",
    "_aawm_alias_classification.classify_failure": "_classify_failure",
    "_read_pilot_cooldown_gate.record": "_read_pilot_gate_record",
    "litellm.get_model_info": "_get_model_info",
    "litellm.model_cost": "_model_cost",
    "litellm.LlmProviders.OPENAI.value": "_openai_provider_value",
}


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _BaselineNormalizer(ast.NodeTransformer):
    """Normalize only documented extraction/configuration mechanics."""

    def visit_Assert(self, node: ast.Assert) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.module in {"fastapi", "starlette"}:
            return None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = _NAME_NORMALIZATION.get(node.id, node.id)
        if node.id == "CooldownPublicationPlan":
            node.id = "_CooldownPublicationPlan"
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        replacement = _ATTRIBUTE_NORMALIZATION.get(_dotted_name(node))
        if replacement is not None:
            return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)
        return self.generic_visit(node)


def _module_tree(module: Any) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _top_level_assignments(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _normalized_digest(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = copy.deepcopy(node)
    normalized.decorator_list = []
    normalized.returns = None
    normalized.type_comment = None
    if (
        normalized.body
        and isinstance(normalized.body[0], ast.Expr)
        and isinstance(normalized.body[0].value, ast.Constant)
        and isinstance(normalized.body[0].value.value, str)
    ):
        normalized.body = normalized.body[1:]
    normalized = _BaselineNormalizer().visit(normalized)
    ast.fix_missing_locations(normalized)
    payload = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_frozen_inventory_and_facade_counts() -> None:
    assert {name: len(symbols) for name, symbols in FROZEN_SYMBOLS.items()} == {
        "error_signals": 44,
        "cooldown_apply": 10,
        "attempt_records": 9,
    }
    assert sum(map(len, FROZEN_SYMBOLS.values())) == 63
    assert len(BASELINE_NORMALIZED_SHA256) == 63
    assert set(BASELINE_NORMALIZED_SHA256) == {
        symbol
        for symbols in FROZEN_SYMBOLS.values()
        for symbol in symbols
    }


def test_target_modules_are_sole_function_owners() -> None:
    god_functions = _top_level_functions(ast.parse(GOD_PATH.read_text(encoding="utf-8")))
    for module_name, symbols in FROZEN_SYMBOLS.items():
        module = TARGET_MODULES[module_name]
        target_functions = _top_level_functions(_module_tree(module))
        assert tuple(module._HOST_FUNCTION_NAMES) == symbols
        assert not (set(symbols) & set(god_functions))
        assert set(symbols) <= set(target_functions)
        for symbol in symbols:
            host_facade = getattr(lpe, symbol, None)
            owner_function = getattr(module, symbol)
            canonical_function = CANONICAL_HOST_FACADE_OVERRIDES.get(
                symbol, owner_function
            )
            assert callable(host_facade)
            assert host_facade.__code__ is canonical_function.__code__


def test_all_50_normalized_bodies_and_signatures_match_baseline() -> None:
    for module_name, symbols in FROZEN_SYMBOLS.items():
        functions = _top_level_functions(_module_tree(TARGET_MODULES[module_name]))
        for symbol in symbols:
            assert _normalized_digest(functions[symbol]) == BASELINE_NORMALIZED_SHA256[symbol], (
                f"{module_name}.{symbol} drifted from baseline 79fc94c3a5 "
                "outside the documented extraction normalizations"
            )


def test_fail_fast_assert_deviations_are_exactly_documented() -> None:
    actual: dict[str, int] = {}
    for module_name, symbols in FROZEN_SYMBOLS.items():
        functions = _top_level_functions(_module_tree(TARGET_MODULES[module_name]))
        for symbol in symbols:
            count = sum(
                isinstance(node, ast.Assert)
                for node in ast.walk(functions[symbol])
            )
            if count:
                actual[symbol] = count
    assert actual == EXPECTED_ASSERT_COUNTS
    assert len(actual) == 21


def test_god_module_facades_are_same_objects_with_owner_globals() -> None:
    facade_count = 0
    for module_name, symbols in FROZEN_SYMBOLS.items():
        module = TARGET_MODULES[module_name]
        for symbol in symbols:
            owner_function = getattr(module, symbol)
            host_facade = getattr(lpe, symbol)
            canonical_function = CANONICAL_HOST_FACADE_OVERRIDES.get(
                symbol, owner_function
            )
            assert callable(host_facade)
            assert host_facade.__code__ is canonical_function.__code__
            if symbol in CANONICAL_HOST_FACADE_OVERRIDES:
                assert host_facade is canonical_function
            elif host_facade is not owner_function:
                assert host_facade.__globals__ is vars(lpe)
            facade_count += 1
    assert facade_count == 63


def test_installed_host_contract_retains_candidate_loop_dependencies() -> None:
    host_globals = vars(lpe)
    for module in TARGET_MODULES.values():
        assert module._host_globals_ref is host_globals

    for name in ("status", "HTTPException", "verbose_proxy_logger"):
        assert name in host_globals
        assert getattr(lpe, name) is host_globals[name]


def test_wave5c_modules_have_no_module_scope_god_import() -> None:
    forbidden = "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
    for module in TARGET_MODULES.values():
        for node in _module_tree(module).body:
            if isinstance(node, ast.Import):
                assert forbidden not in {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.module != forbidden


def test_selection_and_cooldown_state_exclusions_remain_owned() -> None:
    cooldown_functions = set(_top_level_functions(_module_tree(cooldown_apply)))
    selection_functions = set(_top_level_functions(_module_tree(selection)))
    state_functions = set(_top_level_functions(_module_tree(cooldown_state)))

    assert not (SELECTION_OWNED_EXCLUSIONS & cooldown_functions)
    assert SELECTION_OWNED_EXCLUSIONS <= selection_functions
    for symbol in SELECTION_OWNED_EXCLUSIONS:
        assert getattr(lpe, symbol) is getattr(selection, symbol)

    assert not (COOLDOWN_STATE_OWNED_EXCLUSIONS & cooldown_functions)
    assert COOLDOWN_STATE_OWNED_EXCLUSIONS <= state_functions
    for symbol in COOLDOWN_STATE_OWNED_EXCLUSIONS:
        assert getattr(lpe, symbol) is getattr(cooldown_state, symbol)


def test_candidate_loop_keeps_monkeypatch_compatible_facade_lookup() -> None:
    tree = ast.parse((PACKAGE_PATH / "candidate_loop.py").read_text(encoding="utf-8"))
    forbidden_relative_imports = {"error_signals", "cooldown_apply", "attempt_records"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden_relative_imports

    handle = _top_level_functions(tree)["handle_alias_route"]
    loaded_attributes = {
        node.attr
        for node in ast.walk(handle)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_lpe"
    }
    required = {
        "_classify_codex_auto_agent_retryable_exhaustion",
        "_record_auto_agent_alias_attempt_failure",
        "_record_read_pilot_cooldown_evidence",
        "_apply_request_local_cooldown_from_plan",
    }
    assert required <= loaded_attributes


def test_package_exports_wave5c_modules() -> None:
    assert {"error_signals", "cooldown_apply", "attempt_records"} <= set(package.__all__)
    assert package.error_signals is error_signals
    assert package.cooldown_apply is cooldown_apply
    assert package.attempt_records is attempt_records


def test_architecture_documents_wave5c_ownership_and_deviations() -> None:
    text = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    for required in (
        "#### Wave 5C ownership and baseline parity",
        "`aawm_alias_routing/error_signals.py`",
        "`aawm_alias_routing/cooldown_apply.py`",
        "`aawm_alias_routing/attempt_records.py`",
        "`79fc94c3a5`",
        "50 functions",
        "Fail-fast `assert` guards",
        "`_apply_request_local_cooldown_from_plan`",
        "`_apply_codex_auto_agent_grok_account_lane_cooldown`",
    ):
        assert required in text
