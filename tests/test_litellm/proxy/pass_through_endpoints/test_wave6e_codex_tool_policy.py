"""Tests for Wave 6E codex_tool_policy extraction.

Covers: spawn-agent description patches, core tool guidance, model-capability
lookups, custom-tool-to-function adaptation, namespace-tool adaptation,
unsupported hosted-tool/parameter/input-item drops, tool-choice cleanup,
Grok-native input-item policy facades, ordering/idempotence, and no-op/error
paths.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from unittest.mock import MagicMock


from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
    CODEX_CORE_TOOL_GUIDANCE_BY_NAME,
    CODEX_SPAWN_AGENT_FANOUT_POLICY,
    CODEX_SPAWN_AGENT_PAYLOAD_FIELD_ORDER,
    CODEX_SPAWN_AGENT_TOOL_NAME,
    CodexToolPolicyCallbacks,
    adapt_codex_custom_tool_choice,
    adapt_codex_custom_tool_definitions,
    adapt_codex_custom_tool_input_items,
    adapt_codex_custom_tools_to_functions_from_request_body,
    adapt_codex_namespace_input_items,
    adapt_codex_namespace_tool_choice,
    adapt_codex_namespace_tool_definitions,
    adapt_codex_namespace_tools_to_functions_from_request_body,
    adapted_custom_tool_function_schema,
    append_codex_core_tool_guidance_to_description,
    apply_codex_tool_description_patches_to_request_body,
    drop_tool_choice_without_tools_from_request_body,
    drop_unsupported_codex_hosted_tools_from_request_body,
    drop_unsupported_codex_input_items_from_request_body,
    drop_unsupported_codex_request_params_from_request_body,
    extract_openai_passthrough_tool_choice,
    get_codex_core_tool_guidance,
    get_codex_tool_policy_model_cost_candidates,
    get_custom_tool_function_adapter_names_for_model,
    get_namespace_tool_function_adapter_names_for_model,
    get_openai_tool_name,
    get_openai_tool_type,
    get_unsupported_hosted_tool_types_for_model,
    get_unsupported_input_item_types_for_model,
    get_unsupported_request_param_names_for_model,
    openai_tool_choice_references_tool_type,
    patch_codex_core_tool_description,
    patch_codex_multi_agent_tool_search_description,
    patch_codex_spawn_agent_description_text,
    patch_codex_spawn_agent_payload_parameters,
    patch_codex_spawn_agent_tool_description,
    request_has_openai_tool_definitions,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _normalize_tag_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned or None
    return None


def _dedupe_sorted(values: list[str]) -> list[str]:
    return sorted({v for v in values if isinstance(v, str) and v})


def _merge_metadata(
    request_body: dict[str, Any],
    *,
    tags_to_add: list[str],
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(request_body)
    meta = dict(updated.get("litellm_metadata") or {})
    tags = list(meta.get("tags") or [])
    for t in tags_to_add:
        if t not in tags:
            tags.append(t)
    meta["tags"] = tags
    meta.update(extra_fields)
    updated["litellm_metadata"] = meta
    return updated


def _build_span(*, name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "metadata": metadata}


def _make_callbacks(
    model_cost: Optional[dict[str, Any]] = None,
) -> CodexToolPolicyCallbacks:
    return CodexToolPolicyCallbacks(
        normalize_tag_value=_normalize_tag_value,
        dedupe_sorted=_dedupe_sorted,
        merge_metadata=_merge_metadata,
        build_span=_build_span,
        get_model_cost_map=lambda: model_cost or {},
        normalize_grok_native_oauth_model=lambda m: None,
        is_oa_xai_model=lambda m: False,
        resolve_oa_xai_upstream_model=lambda m: m,
        normalize_kimi_model_name=lambda m: None,
        normalize_kimi_custom_tool_outputs=lambda b: b,
    )


CB = _make_callbacks()


# ---------------------------------------------------------------------------
# get_openai_tool_name / get_openai_tool_type
# ---------------------------------------------------------------------------


class TestToolAccessors:
    def test_name_from_top_level(self):
        assert get_openai_tool_name({"name": " bash "}) == "bash"

    def test_name_from_function(self):
        assert get_openai_tool_name({"function": {"name": "edit"}}) == "edit"

    def test_name_missing(self):
        assert get_openai_tool_name({"type": "custom"}) is None

    def test_type_present(self):
        assert get_openai_tool_type({"type": " function "}) == "function"

    def test_type_missing(self):
        assert get_openai_tool_type({"name": "x"}) is None


# ---------------------------------------------------------------------------
# extract_openai_passthrough_tool_choice
# ---------------------------------------------------------------------------


class TestExtractToolChoice:
    def test_string(self):
        assert extract_openai_passthrough_tool_choice(" Auto ", normalize_tag_value=_normalize_tag_value) == "auto"

    def test_dict_type(self):
        assert extract_openai_passthrough_tool_choice({"type": "Required"}, normalize_tag_value=_normalize_tag_value) == "required"

    def test_dict_name(self):
        assert extract_openai_passthrough_tool_choice({"name": "Bash"}, normalize_tag_value=_normalize_tag_value) == "bash"

    def test_none(self):
        assert extract_openai_passthrough_tool_choice(42, normalize_tag_value=_normalize_tag_value) is None

    def test_empty_string(self):
        assert extract_openai_passthrough_tool_choice("  ", normalize_tag_value=_normalize_tag_value) is None


# ---------------------------------------------------------------------------
# Spawn-agent description patching
# ---------------------------------------------------------------------------


class TestSpawnAgentDescriptionPatch:
    def test_replaces_restrictive_pattern(self):
        desc = (
            "Only use spawn_agent if and only if the user explicitly asks for "
            "sub-agents, delegation, or parallel agent work."
        )
        updated, count = patch_codex_spawn_agent_description_text(desc)
        assert count == 1
        assert CODEX_SPAWN_AGENT_FANOUT_POLICY in updated

    def test_no_match_returns_original(self):
        desc = "A normal description."
        updated, count = patch_codex_spawn_agent_description_text(desc)
        assert count == 0
        assert updated == desc

    def test_idempotent_after_patch(self):
        desc = (
            "Only use spawn_agent if and only if the user explicitly asks for "
            "sub-agents, delegation, or parallel agent work."
        )
        first, _ = patch_codex_spawn_agent_description_text(desc)
        second, count = patch_codex_spawn_agent_description_text(first)
        assert count == 0
        assert second == first


class TestSpawnAgentPayloadParameters:
    def test_none_parameters_creates_schema(self):
        params, added, removed = patch_codex_spawn_agent_payload_parameters(None)
        assert set(added) == set(CODEX_SPAWN_AGENT_PAYLOAD_FIELD_ORDER)
        assert removed == []
        assert params["type"] == "object"

    def test_removes_fork_context(self):
        params_in = {
            "type": "object",
            "properties": {
                "fork_context": {"type": "boolean"},
                "agent_type": {"type": "string"},
                "model": {"type": "string"},
                "fork_turns": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["fork_context", "message"],
        }
        params, added, removed = patch_codex_spawn_agent_payload_parameters(params_in)
        assert "fork_context" in removed
        assert "fork_context" not in params["properties"]
        assert "fork_context" not in params.get("required", [])
        assert added == []

    def test_non_dict_passthrough(self):
        params, added, removed = patch_codex_spawn_agent_payload_parameters("bad")
        assert params == "bad"
        assert added == []
        assert removed == []

    def test_no_change_returns_original_identity(self):
        params_in = {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string"},
                "model": {"type": "string"},
                "fork_turns": {"type": "string"},
                "message": {"type": "string"},
            },
        }
        params, added, removed = patch_codex_spawn_agent_payload_parameters(params_in)
        assert added == []
        assert removed == []
        assert params is params_in


class TestPatchSpawnAgentToolDescription:
    def test_non_spawn_agent_noop(self):
        tool = {"name": "bash", "description": "hello"}
        result, events = patch_codex_spawn_agent_tool_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert result is tool
        assert events == []

    def test_patches_description_and_params(self):
        restrictive = (
            "Only use spawn_agent if and only if the user explicitly asks for "
            "sub-agents, delegation, or parallel agent work."
        )
        tool = {
            "name": CODEX_SPAWN_AGENT_TOOL_NAME,
            "description": restrictive,
            "parameters": {"type": "object", "properties": {}},
        }
        result, events = patch_codex_spawn_agent_tool_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert result is not tool
        assert CODEX_SPAWN_AGENT_FANOUT_POLICY in result["description"]
        assert len(events) >= 2  # description + payload
        ids = {e["id"] for e in events}
        assert "spawn-agent-fanout-policy" in ids
        assert "spawn-agent-payload-schema" in ids

    def test_function_nested_description(self):
        restrictive = (
            "Only use spawn_agent if and only if the user explicitly asks for "
            "sub-agents, delegation, or parallel agent work."
        )
        tool = {
            "name": CODEX_SPAWN_AGENT_TOOL_NAME,
            "function": {"name": CODEX_SPAWN_AGENT_TOOL_NAME, "description": restrictive},
        }
        result, events = patch_codex_spawn_agent_tool_description(
            tool, tool_index=1, normalize_tag_value=_normalize_tag_value
        )
        assert result is not tool
        assert CODEX_SPAWN_AGENT_FANOUT_POLICY in result["function"]["description"]


# ---------------------------------------------------------------------------
# Core tool guidance
# ---------------------------------------------------------------------------


class TestCoreToolGuidance:
    def test_known_tool(self):
        guidance = get_codex_core_tool_guidance(
            "bash", normalize_tag_value=_normalize_tag_value
        )
        assert guidance == CODEX_CORE_TOOL_GUIDANCE_BY_NAME["bash"]

    def test_unknown_tool(self):
        assert (
            get_codex_core_tool_guidance(
                "unknown_tool", normalize_tag_value=_normalize_tag_value
            )
            is None
        )

    def test_none_tool(self):
        assert (
            get_codex_core_tool_guidance(None, normalize_tag_value=_normalize_tag_value)
            is None
        )

    def test_append_idempotent(self):
        guidance = CODEX_CORE_TOOL_GUIDANCE_BY_NAME["edit"]
        first, changed1 = append_codex_core_tool_guidance_to_description(
            "Existing.", guidance=guidance
        )
        assert changed1
        second, changed2 = append_codex_core_tool_guidance_to_description(
            first, guidance=guidance
        )
        assert not changed2
        assert second == first

    def test_append_empty_description(self):
        guidance = CODEX_CORE_TOOL_GUIDANCE_BY_NAME["read"]
        result, changed = append_codex_core_tool_guidance_to_description(
            "", guidance=guidance
        )
        assert changed
        assert result == guidance


class TestPatchCoreToolDescription:
    def test_no_guidance_noop(self):
        tool = {"name": "random_tool", "description": "hi"}
        result, events = patch_codex_core_tool_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert result is tool
        assert events == []

    def test_appends_guidance(self):
        tool = {"name": "bash", "description": "Run commands."}
        result, events = patch_codex_core_tool_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert result is not tool
        assert CODEX_CORE_TOOL_GUIDANCE_BY_NAME["bash"] in result["description"]
        assert len(events) == 1
        assert events[0]["id"] == "core-tool-guidance-bash"


class TestMultiAgentToolSearchDescription:
    def test_non_tool_search_noop(self):
        tool = {"type": "function", "description": "Multi-agent tools"}
        result, events = patch_codex_multi_agent_tool_search_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert result is tool
        assert events == []

    def test_patches_tool_search(self):
        tool = {"type": "tool_search", "description": "Multi-agent tools search."}
        result, events = patch_codex_multi_agent_tool_search_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert result is not tool
        assert CODEX_SPAWN_AGENT_FANOUT_POLICY in result["description"]
        assert len(events) == 1

    def test_idempotent(self):
        tool = {"type": "tool_search", "description": "Multi-agent tools search."}
        first, _ = patch_codex_multi_agent_tool_search_description(
            tool, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        second, events = patch_codex_multi_agent_tool_search_description(
            first, tool_index=0, normalize_tag_value=_normalize_tag_value
        )
        assert second is first
        assert events == []


# ---------------------------------------------------------------------------
# Model-capability policy lookups
# ---------------------------------------------------------------------------


class TestModelCostCandidates:
    def test_simple_model(self):
        candidates = get_codex_tool_policy_model_cost_candidates(
            "gpt-4o", callbacks=CB
        )
        assert "gpt-4o" in candidates
        assert "openai/gpt-4o" in candidates
        assert "chatgpt/gpt-4o" in candidates

    def test_prefixed_model(self):
        candidates = get_codex_tool_policy_model_cost_candidates(
            "openai/gpt-4o", callbacks=CB
        )
        assert "gpt-4o" in candidates
        assert "openai/gpt-4o" in candidates

    def test_empty_model(self):
        assert get_codex_tool_policy_model_cost_candidates("", callbacks=CB) == []
        assert get_codex_tool_policy_model_cost_candidates(None, callbacks=CB) == []

    def test_grok_native_candidates(self):
        cb = CodexToolPolicyCallbacks(
            normalize_tag_value=_normalize_tag_value,
            dedupe_sorted=_dedupe_sorted,
            merge_metadata=_merge_metadata,
            build_span=_build_span,
            get_model_cost_map=lambda: {},
            normalize_grok_native_oauth_model=lambda m: "grok-3",
            is_oa_xai_model=lambda m: False,
            resolve_oa_xai_upstream_model=lambda m: m,
            normalize_kimi_model_name=lambda m: None,
            normalize_kimi_custom_tool_outputs=lambda b: b,
        )
        candidates = get_codex_tool_policy_model_cost_candidates(
            "some-model", callbacks=cb
        )
        assert "xai/grok-3" in candidates


class TestModelCapabilityLookups:
    def test_unsupported_hosted_tools(self):
        cost = {"gpt-4o": {"unsupported_hosted_tools": ["web_search", "code_interpreter"]}}
        cb = _make_callbacks(cost)
        result = get_unsupported_hosted_tool_types_for_model("gpt-4o", callbacks=cb)
        assert result == {"web_search", "code_interpreter"}

    def test_unsupported_params(self):
        cost = {"gpt-4o": {"unsupported_request_params": ["parallel_tool_calls"]}}
        cb = _make_callbacks(cost)
        result = get_unsupported_request_param_names_for_model("gpt-4o", callbacks=cb)
        assert result == {"parallel_tool_calls"}

    def test_unsupported_input_items(self):
        cost = {"gpt-4o": {"unsupported_input_item_types": ["reasoning"]}}
        cb = _make_callbacks(cost)
        result = get_unsupported_input_item_types_for_model("gpt-4o", callbacks=cb)
        assert result == {"reasoning"}

    def test_custom_tool_adapters(self):
        cost = {"gpt-4o": {"custom_tool_function_adapters": ["apply_patch"]}}
        cb = _make_callbacks(cost)
        result = get_custom_tool_function_adapter_names_for_model("gpt-4o", callbacks=cb)
        assert result == {"apply_patch"}

    def test_namespace_tool_adapters(self):
        cost = {
            "gpt-4o": {
                "namespace_tool_function_adapters": {
                    "multi_agent_v1": ["spawn_agent", "list_agents"]
                }
            }
        }
        cb = _make_callbacks(cost)
        result = get_namespace_tool_function_adapter_names_for_model(
            "gpt-4o", callbacks=cb
        )
        assert result == {"multi_agent_v1": {"spawn_agent", "list_agents"}}

    def test_no_model_info_returns_empty(self):
        cb = _make_callbacks({})
        assert get_unsupported_hosted_tool_types_for_model("unknown", callbacks=cb) == set()
        assert get_unsupported_request_param_names_for_model("unknown", callbacks=cb) == set()
        assert get_unsupported_input_item_types_for_model("unknown", callbacks=cb) == set()
        assert get_custom_tool_function_adapter_names_for_model("unknown", callbacks=cb) == set()
        assert get_namespace_tool_function_adapter_names_for_model("unknown", callbacks=cb) == {}


# ---------------------------------------------------------------------------
# Custom-tool-to-function adaptation
# ---------------------------------------------------------------------------


class TestCustomToolAdaptation:
    def test_adapted_schema(self):
        tool = {"type": "custom", "name": "apply_patch", "description": "Apply patches."}
        result = adapted_custom_tool_function_schema(tool, tool_name="apply_patch")
        assert result["type"] == "function"
        assert result["name"] == "apply_patch"
        assert result["description"] == "Apply patches."
        assert result["parameters"]["properties"]["input"]["type"] == "string"

    def test_adapt_definitions(self):
        tools = [
            {"type": "custom", "name": "apply_patch"},
            {"type": "function", "name": "bash"},
        ]
        updated, adapted = adapt_codex_custom_tool_definitions(
            tools, adapter_names={"apply_patch"}, normalize_tag_value=_normalize_tag_value
        )
        assert updated is not None
        assert len(adapted) == 1
        assert adapted[0]["name"] == "apply_patch"
        assert updated[0]["type"] == "function"
        assert updated[1]["type"] == "function"  # unchanged

    def test_adapt_definitions_not_list(self):
        updated, adapted = adapt_codex_custom_tool_definitions(
            "bad", adapter_names={"x"}, normalize_tag_value=_normalize_tag_value
        )
        assert updated is None
        assert adapted == []

    def test_adapt_input_items(self):
        items = [
            {"type": "custom_tool_call", "name": "apply_patch", "call_id": "c1", "input": "patch text"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
            {"type": "message", "content": "hello"},
        ]
        updated, adapted = adapt_codex_custom_tool_input_items(
            items, adapter_names={"apply_patch"}, normalize_tag_value=_normalize_tag_value
        )
        assert updated is not None
        assert len(adapted) == 2
        assert updated[0]["type"] == "function_call"
        assert json.loads(updated[0]["arguments"]) == {"input": "patch text"}
        assert updated[1]["type"] == "function_call_output"
        assert updated[2]["type"] == "message"  # unchanged

    def test_adapt_input_items_no_match(self):
        items = [{"type": "message", "content": "hi"}]
        updated, adapted = adapt_codex_custom_tool_input_items(
            items, adapter_names={"apply_patch"}, normalize_tag_value=_normalize_tag_value
        )
        assert updated is None
        assert adapted == []

    def test_adapt_tool_choice(self):
        choice = {"type": "custom", "name": "apply_patch"}
        result, changed = adapt_codex_custom_tool_choice(
            choice, adapter_names={"apply_patch"}, normalize_tag_value=_normalize_tag_value
        )
        assert changed
        assert result["type"] == "function"

    def test_adapt_tool_choice_no_match(self):
        choice = {"type": "auto"}
        result, changed = adapt_codex_custom_tool_choice(
            choice, adapter_names={"apply_patch"}, normalize_tag_value=_normalize_tag_value
        )
        assert not changed
        assert result is choice

    def test_full_request_body_no_adapters(self):
        body = {"model": "gpt-4o", "tools": [{"type": "function", "name": "bash"}]}
        result, adapted = adapt_codex_custom_tools_to_functions_from_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert adapted == []

    def test_full_request_body_with_adapters(self):
        cost = {"gpt-4o": {"custom_tool_function_adapters": ["apply_patch"]}}
        cb = _make_callbacks(cost)
        body = {
            "model": "gpt-4o",
            "tools": [{"type": "custom", "name": "apply_patch", "description": "x"}],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }
        result, adapted = adapt_codex_custom_tools_to_functions_from_request_body(
            body, callbacks=cb
        )
        assert result is not body
        assert len(adapted) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tool_choice"]["type"] == "function"
        assert "litellm_metadata" in result


# ---------------------------------------------------------------------------
# Namespace-tool adaptation
# ---------------------------------------------------------------------------


class TestNamespaceToolAdaptation:
    def test_adapt_definitions(self):
        tools = [
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {"type": "object", "properties": {}},
                        "description": "Spawn.",
                    }
                ],
            }
        ]
        adapter_names = {"multi_agent_v1": {"spawn_agent"}}
        updated, adapted, skipped = adapt_codex_namespace_tool_definitions(
            tools, adapter_names=adapter_names, normalize_tag_value=_normalize_tag_value
        )
        assert updated is not None
        assert len(adapted) == 1
        assert adapted[0]["name"] == "spawn_agent"
        assert skipped == []
        assert updated[0]["type"] == "function"
        assert updated[0]["name"] == "spawn_agent"

    def test_name_collision_skipped(self):
        tools = [
            {"type": "function", "name": "spawn_agent", "parameters": {}},
            {
                "type": "namespace",
                "name": "ns",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ]
        adapter_names = {"ns": {"spawn_agent"}}
        updated, adapted, skipped = adapt_codex_namespace_tool_definitions(
            tools, adapter_names=adapter_names, normalize_tag_value=_normalize_tag_value
        )
        assert len(adapted) == 0
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "name_collision"

    def test_child_not_function_skipped(self):
        tools = [
            {
                "type": "namespace",
                "name": "ns",
                "tools": [{"type": "custom", "name": "x", "parameters": {}}],
            }
        ]
        adapter_names = {"ns": {"x"}}
        _, adapted, skipped = adapt_codex_namespace_tool_definitions(
            tools, adapter_names=adapter_names, normalize_tag_value=_normalize_tag_value
        )
        assert adapted == []
        assert skipped[0]["reason"] == "child_not_function"

    def test_not_list_noop(self):
        updated, adapted, skipped = adapt_codex_namespace_tool_definitions(
            "bad", adapter_names={}, normalize_tag_value=_normalize_tag_value
        )
        assert updated is None
        assert adapted == []
        assert skipped == []

    def test_adapt_input_items(self):
        items = [
            {
                "type": "function_call",
                "namespace": "ns",
                "name": "spawn_agent",
                "arguments": "{}",
            },
            {"type": "message", "content": "hi"},
        ]
        adapter_names = {"ns": {"spawn_agent"}}
        updated, adapted = adapt_codex_namespace_input_items(
            items, adapter_names=adapter_names, normalize_tag_value=_normalize_tag_value
        )
        assert updated is not None
        assert len(adapted) == 1
        assert "namespace" not in updated[0]
        assert updated[1]["type"] == "message"

    def test_adapt_input_items_custom_tool_call(self):
        items = [
            {
                "type": "custom_tool_call",
                "namespace": "ns",
                "name": "spawn_agent",
                "input": "raw",
                "call_id": "c1",
            }
        ]
        adapter_names = {"ns": {"spawn_agent"}}
        updated, adapted = adapt_codex_namespace_input_items(
            items, adapter_names=adapter_names, normalize_tag_value=_normalize_tag_value
        )
        assert updated is not None
        assert updated[0]["type"] == "function_call"
        assert updated[0]["arguments"] == "raw"

    def test_adapt_tool_choice(self):
        choice = {"type": "custom", "namespace": "ns", "name": "spawn_agent"}
        adapter_names = {"ns": {"spawn_agent"}}
        result, info = adapt_codex_namespace_tool_choice(
            choice, adapter_names=adapter_names, normalize_tag_value=_normalize_tag_value
        )
        assert info == {"namespace": "ns", "name": "spawn_agent"}
        assert result["type"] == "function"
        assert result["function"]["name"] == "spawn_agent"

    def test_adapt_tool_choice_no_match(self):
        choice = {"type": "auto"}
        result, info = adapt_codex_namespace_tool_choice(
            choice, adapter_names={"ns": {"x"}}, normalize_tag_value=_normalize_tag_value
        )
        assert info is None
        assert result is choice

    def test_full_request_body_no_adapters(self):
        body = {"model": "gpt-4o"}
        result, adapted = adapt_codex_namespace_tools_to_functions_from_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert adapted == []


# ---------------------------------------------------------------------------
# Unsupported hosted-tool drops
# ---------------------------------------------------------------------------


class TestUnsupportedHostedToolDrops:
    def test_drops_unsupported_tools(self):
        cost = {"gpt-4o": {"unsupported_hosted_tools": ["web_search"]}}
        cb = _make_callbacks(cost)
        body = {
            "model": "gpt-4o",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "bash"},
            ],
        }
        result, removed = drop_unsupported_codex_hosted_tools_from_request_body(
            body, callbacks=cb
        )
        assert len(removed) == 1
        assert removed[0]["type"] == "web_search"
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "bash"

    def test_removes_tool_choice_referencing_removed(self):
        cost = {"gpt-4o": {"unsupported_hosted_tools": ["web_search"]}}
        cb = _make_callbacks(cost)
        body = {
            "model": "gpt-4o",
            "tools": [{"type": "web_search"}],
            "tool_choice": {"type": "web_search"},
        }
        result, removed = drop_unsupported_codex_hosted_tools_from_request_body(
            body, callbacks=cb
        )
        assert "tool_choice" not in result

    def test_no_unsupported_noop(self):
        body = {"model": "gpt-4o", "tools": [{"type": "function", "name": "bash"}]}
        result, removed = drop_unsupported_codex_hosted_tools_from_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert removed == []


# ---------------------------------------------------------------------------
# Unsupported request param drops
# ---------------------------------------------------------------------------


class TestUnsupportedParamDrops:
    def test_drops_top_level_param(self):
        cost = {"gpt-4o": {"unsupported_request_params": ["parallel_tool_calls"]}}
        cb = _make_callbacks(cost)
        body = {"model": "gpt-4o", "parallel_tool_calls": True}
        result, removed = drop_unsupported_codex_request_params_from_request_body(
            body, callbacks=cb
        )
        assert "parallel_tool_calls" in removed
        assert "parallel_tool_calls" not in result

    def test_drops_nested_param(self):
        cost = {"gpt-4o": {"unsupported_request_params": ["reasoning.effort"]}}
        cb = _make_callbacks(cost)
        body = {"model": "gpt-4o", "reasoning": {"effort": "high", "summary": "auto"}}
        result, removed = drop_unsupported_codex_request_params_from_request_body(
            body, callbacks=cb
        )
        assert len(removed) == 1
        assert "effort" not in result["reasoning"]
        assert result["reasoning"]["summary"] == "auto"

    def test_no_unsupported_noop(self):
        body = {"model": "gpt-4o", "temperature": 0.7}
        result, removed = drop_unsupported_codex_request_params_from_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert removed == []

    def test_depth_bound(self):
        cost = {"gpt-4o": {"unsupported_request_params": ["deep_param"]}}
        cb = CodexToolPolicyCallbacks(
            normalize_tag_value=_normalize_tag_value,
            dedupe_sorted=_dedupe_sorted,
            merge_metadata=_merge_metadata,
            build_span=_build_span,
            get_model_cost_map=lambda: cost,
            normalize_grok_native_oauth_model=lambda m: None,
            is_oa_xai_model=lambda m: False,
            resolve_oa_xai_upstream_model=lambda m: m,
            normalize_kimi_model_name=lambda m: None,
            normalize_kimi_custom_tool_outputs=lambda b: b,
            request_body_walk_max_depth=2,
        )
        body = {"model": "gpt-4o", "a": {"b": {"c": {"deep_param": 1}}}}
        result, removed = drop_unsupported_codex_request_params_from_request_body(
            body, callbacks=cb
        )
        # deep_param is at depth 4, beyond max_depth=2, so it should NOT be removed
        assert removed == []


# ---------------------------------------------------------------------------
# Unsupported input-item drops
# ---------------------------------------------------------------------------


class TestUnsupportedInputItemDrops:
    def test_drops_reasoning_items(self):
        cost = {"gpt-4o": {"unsupported_input_item_types": ["reasoning"]}}
        cb = _make_callbacks(cost)
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "reasoning", "encrypted_content": "abc"},
                {"type": "message", "content": "hi"},
            ],
        }
        result, removed = drop_unsupported_codex_input_items_from_request_body(
            body, callbacks=cb
        )
        assert len(removed) == 1
        assert removed[0]["type"] == "reasoning"
        assert removed[0]["encrypted_content"] is True
        assert len(result["input"]) == 1

    def test_no_unsupported_noop(self):
        body = {"model": "gpt-4o", "input": [{"type": "message"}]}
        result, removed = drop_unsupported_codex_input_items_from_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert removed == []

    def test_non_list_input_noop(self):
        cost = {"gpt-4o": {"unsupported_input_item_types": ["reasoning"]}}
        cb = _make_callbacks(cost)
        body = {"model": "gpt-4o", "input": "not a list"}
        result, removed = drop_unsupported_codex_input_items_from_request_body(
            body, callbacks=cb
        )
        assert result is body
        assert removed == []


# ---------------------------------------------------------------------------
# Tool-choice cleanup
# ---------------------------------------------------------------------------


class TestToolChoiceCleanup:
    def test_drops_tool_choice_without_tools(self):
        body = {"model": "gpt-4o", "tool_choice": "auto"}
        result, removed = drop_tool_choice_without_tools_from_request_body(
            body, callbacks=CB
        )
        assert removed == "auto"
        assert "tool_choice" not in result

    def test_keeps_tool_choice_with_tools(self):
        body = {
            "model": "gpt-4o",
            "tools": [{"type": "function", "name": "bash"}],
            "tool_choice": "auto",
        }
        result, removed = drop_tool_choice_without_tools_from_request_body(
            body, callbacks=CB
        )
        assert removed is None
        assert result is body

    def test_no_tool_choice_noop(self):
        body = {"model": "gpt-4o"}
        result, removed = drop_tool_choice_without_tools_from_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert removed is None


# ---------------------------------------------------------------------------
# openai_tool_choice_references_tool_type
# ---------------------------------------------------------------------------


class TestToolChoiceReferencesType:
    def test_string_match(self):
        assert openai_tool_choice_references_tool_type(
            "web_search", {"web_search"}, normalize_tag_value=_normalize_tag_value
        )

    def test_dict_type_match(self):
        assert openai_tool_choice_references_tool_type(
            {"type": "web_search"}, {"web_search"}, normalize_tag_value=_normalize_tag_value
        )

    def test_dict_function_name_match(self):
        assert openai_tool_choice_references_tool_type(
            {"function": {"name": "web_search"}},
            {"web_search"},
            normalize_tag_value=_normalize_tag_value,
        )

    def test_no_match(self):
        assert not openai_tool_choice_references_tool_type(
            {"type": "auto"}, {"web_search"}, normalize_tag_value=_normalize_tag_value
        )

    def test_empty_types(self):
        assert not openai_tool_choice_references_tool_type(
            "web_search", set(), normalize_tag_value=_normalize_tag_value
        )


# ---------------------------------------------------------------------------
# request_has_openai_tool_definitions
# ---------------------------------------------------------------------------


class TestRequestHasToolDefs:
    def test_has_tools(self):
        assert request_has_openai_tool_definitions(
            {"tools": [{"type": "function", "name": "bash"}]}
        )

    def test_no_tools(self):
        assert not request_has_openai_tool_definitions({"model": "gpt-4o"})

    def test_empty_tools(self):
        assert not request_has_openai_tool_definitions({"tools": []})

    def test_non_dict_tools(self):
        assert not request_has_openai_tool_definitions({"tools": ["bad"]})


# ---------------------------------------------------------------------------
# Description patch orchestrator
# ---------------------------------------------------------------------------


class TestApplyToolDescriptionPatches:
    def test_no_tools_noop(self):
        body = {"model": "gpt-4o"}
        result, events = apply_codex_tool_description_patches_to_request_body(
            body, callbacks=CB
        )
        assert result is body
        assert events == []

    def test_patches_spawn_agent_and_core(self):
        restrictive = (
            "Only use spawn_agent if and only if the user explicitly asks for "
            "sub-agents, delegation, or parallel agent work."
        )
        body = {
            "model": "gpt-4o",
            "tools": [
                {"name": "spawn_agent", "description": restrictive, "parameters": {"type": "object", "properties": {}}},
                {"name": "bash", "description": "Run commands."},
            ],
        }
        result, events = apply_codex_tool_description_patches_to_request_body(
            body, callbacks=CB
        )
        assert result is not body
        assert len(events) >= 3  # spawn desc + spawn params + bash guidance
        assert "litellm_metadata" in result

    def test_ordering_preserved(self):
        body = {
            "model": "gpt-4o",
            "tools": [
                {"name": "bash", "description": "A"},
                {"name": "edit", "description": "B"},
                {"name": "read", "description": "C"},
            ],
        }
        result, events = apply_codex_tool_description_patches_to_request_body(
            body, callbacks=CB
        )
        names = [get_openai_tool_name(t) for t in result["tools"]]
        assert names == ["bash", "edit", "read"]

    def test_idempotent(self):
        body = {
            "model": "gpt-4o",
            "tools": [{"name": "bash", "description": "Run."}],
        }
        first, _ = apply_codex_tool_description_patches_to_request_body(
            body, callbacks=CB
        )
        second, events2 = apply_codex_tool_description_patches_to_request_body(
            first, callbacks=CB
        )
        assert events2 == []
        assert second is first




# ---------------------------------------------------------------------------
# Regression: wrong-type live-map fallback to bundled/alternate keys
# ---------------------------------------------------------------------------


class TestModelCostWrongTypeFallback:
    """_lookup_model_info_field must skip wrong-typed values and continue."""

    def test_list_field_skips_dict_in_live_map_finds_list_in_bundled(self):
        # Live map has wrong type (dict) for the field; bundled has correct list.
        # Since we cannot easily override the bundled fallback in unit tests,
        # we test with two candidate keys: first key has wrong type, second has right.
        cost = {
            "gpt-4o": {"unsupported_hosted_tools": {"bad": "dict"}},
            "openai/gpt-4o": {"unsupported_hosted_tools": ["web_search"]},
        }
        cb = _make_callbacks(cost)
        result = get_unsupported_hosted_tool_types_for_model("gpt-4o", callbacks=cb)
        assert result == {"web_search"}

    def test_list_field_skips_string_value(self):
        cost = {
            "gpt-4o": {"unsupported_request_params": "not_a_list"},
            "openai/gpt-4o": {"unsupported_request_params": ["parallel_tool_calls"]},
        }
        cb = _make_callbacks(cost)
        result = get_unsupported_request_param_names_for_model("gpt-4o", callbacks=cb)
        assert result == {"parallel_tool_calls"}

    def test_dict_field_skips_list_value(self):
        cost = {
            "gpt-4o": {"namespace_tool_function_adapters": ["wrong"]},
            "openai/gpt-4o": {
                "namespace_tool_function_adapters": {"multi_agent_v1": ["spawn_agent"]}
            },
        }
        cb = _make_callbacks(cost)
        result = get_namespace_tool_function_adapter_names_for_model("gpt-4o", callbacks=cb)
        assert result == {"multi_agent_v1": {"spawn_agent"}}

    def test_all_wrong_type_returns_empty(self):
        cost = {
            "gpt-4o": {"unsupported_hosted_tools": 42},
            "openai/gpt-4o": {"unsupported_hosted_tools": "str"},
        }
        cb = _make_callbacks(cost)
        result = get_unsupported_hosted_tool_types_for_model("gpt-4o", callbacks=cb)
        assert result == set()

    def test_input_item_types_skips_wrong_type(self):
        cost = {
            "gpt-4o": {"unsupported_input_item_types": {"bad": True}},
            "openai/gpt-4o": {"unsupported_input_item_types": ["reasoning"]},
        }
        cb = _make_callbacks(cost)
        result = get_unsupported_input_item_types_for_model("gpt-4o", callbacks=cb)
        assert result == {"reasoning"}

    def test_custom_tool_adapters_skips_wrong_type(self):
        cost = {
            "gpt-4o": {"custom_tool_function_adapters": "apply_patch"},
            "openai/gpt-4o": {"custom_tool_function_adapters": ["apply_patch"]},
        }
        cb = _make_callbacks(cost)
        result = get_custom_tool_function_adapter_names_for_model("gpt-4o", callbacks=cb)
        assert result == {"apply_patch"}


# ---------------------------------------------------------------------------
# Regression: bool and non-string tool_choice normalization
# ---------------------------------------------------------------------------


class TestToolChoiceNormalization:
    """extract_openai_passthrough_tool_choice must use the normalize callback."""

    def test_bool_true_normalized(self):
        result = extract_openai_passthrough_tool_choice(
            True, normalize_tag_value=_normalize_tag_value
        )
        # _normalize_tag_value handles bool -> "true"/"false" but only for
        # isinstance(value, bool) check; the function itself only handles
        # str and dict at the top level. Bool is neither str nor dict, so None.
        # The god path also returns None for non-str/non-dict.
        # But dict values that are bools should be normalized.
        assert result is None

    def test_dict_with_bool_type_value(self):
        # dict {"type": True} - the normalize callback should handle bool
        result = extract_openai_passthrough_tool_choice(
            {"type": True}, normalize_tag_value=_normalize_tag_value
        )
        assert result == "true"

    def test_dict_with_bool_name_value(self):
        result = extract_openai_passthrough_tool_choice(
            {"name": False}, normalize_tag_value=_normalize_tag_value
        )
        assert result == "false"

    def test_dict_with_int_value_returns_none(self):
        # _normalize_tag_value returns None for int
        result = extract_openai_passthrough_tool_choice(
            {"type": 123}, normalize_tag_value=_normalize_tag_value
        )
        assert result is None

    def test_string_uses_callback_not_inline(self):
        # Verify the callback is used: our callback strips and lowercases
        result = extract_openai_passthrough_tool_choice(
            "  REQUIRED  ", normalize_tag_value=_normalize_tag_value
        )
        assert result == "required"

    def test_empty_string_via_callback(self):
        result = extract_openai_passthrough_tool_choice(
            "   ", normalize_tag_value=_normalize_tag_value
        )
        assert result is None

# ---------------------------------------------------------------------------
# Grok-native facades (mock-based)
# ---------------------------------------------------------------------------


class TestGrokNativeFacades:
    def _grok_callbacks(self) -> CodexToolPolicyCallbacks:
        grok_mod = MagicMock()
        grok_mod.stringify_input_item_value.return_value = "stringified"
        grok_mod.format_function_call_input_message.return_value = "call msg"
        grok_mod.format_function_call_output_input_message.return_value = "output msg"
        grok_mod.rewrite_input_item_for_model_input.return_value = {"rewritten": True}
        grok_mod.is_anthropic_responses_adapter_body.return_value = True
        grok_mod.add_input_item_rewrite_logging_metadata.return_value = {"meta": True}
        grok_mod.rewrite_unsupported_input_items_from_request_body.return_value = (
            {"body": True},
            [{"item": 1}],
        )
        grok_mod.rewrite_unsupported_input_items_in_place.return_value = [{"item": 1}]

        runtime = MagicMock()
        return CodexToolPolicyCallbacks(
            normalize_tag_value=_normalize_tag_value,
            dedupe_sorted=_dedupe_sorted,
            merge_metadata=_merge_metadata,
            build_span=_build_span,
            get_model_cost_map=lambda: {},
            normalize_grok_native_oauth_model=lambda m: None,
            is_oa_xai_model=lambda m: False,
            resolve_oa_xai_upstream_model=lambda m: m,
            normalize_kimi_model_name=lambda m: None,
            normalize_kimi_custom_tool_outputs=lambda b: b,
            grok_normalization=grok_mod,
            grok_normalization_runtime=runtime,
        )

    def test_stringify(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            stringify_grok_native_input_item_value,
        )
        cb = self._grok_callbacks()
        assert stringify_grok_native_input_item_value({"x": 1}, callbacks=cb) == "stringified"

    def test_format_call_input(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            format_grok_native_function_call_input_message,
        )
        cb = self._grok_callbacks()
        assert format_grok_native_function_call_input_message({}, callbacks=cb) == "call msg"

    def test_format_call_output(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            format_grok_native_function_call_output_input_message,
        )
        cb = self._grok_callbacks()
        assert (
            format_grok_native_function_call_output_input_message({}, callbacks=cb)
            == "output msg"
        )

    def test_rewrite_item(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            rewrite_grok_native_input_item_for_model_input,
        )
        cb = self._grok_callbacks()
        result = rewrite_grok_native_input_item_for_model_input(
            {}, item_type="reasoning", callbacks=cb
        )
        assert result == {"rewritten": True}

    def test_is_adapter_body(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            is_anthropic_grok_native_responses_adapter_body,
        )
        cb = self._grok_callbacks()
        assert is_anthropic_grok_native_responses_adapter_body({}, callbacks=cb) is True

    def test_rewrite_from_body(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            rewrite_grok_native_unsupported_input_items_from_request_body,
        )
        cb = self._grok_callbacks()
        body, items = rewrite_grok_native_unsupported_input_items_from_request_body(
            {}, callbacks=cb
        )
        assert body == {"body": True}
        assert items == [{"item": 1}]

    def test_rewrite_in_place(self):
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            rewrite_grok_native_unsupported_input_items_in_place,
        )
        cb = self._grok_callbacks()
        items = rewrite_grok_native_unsupported_input_items_in_place({}, callbacks=cb)
        assert items == [{"item": 1}]


class TestKimiCustomToolOutputNormalization:
    """Regression: Kimi custom_tool_call_output must normalize before filter."""

    def test_kimi_custom_tool_output_normalized_before_unsupported_filter(self):
        """Verify normalize_kimi_custom_tool_outputs callback is invoked.

        The extracted codex_tool_policy.drop_unsupported_codex_input_items_from_request_body
        must call callbacks.normalize_kimi_custom_tool_outputs before filtering,
        so custom_tool_call_output becomes function_call_output and is preserved.
        """
        from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
            codex_tool_policy,
        )

        request_body = {
            "model": "kimi-code",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_123",
                    "output": "Sunny, 20C",
                },
            ],
        }

        # Track if normalize_kimi_custom_tool_outputs was called
        normalize_called = []

        def mock_normalize(body):
            normalize_called.append(True)
            # Simulate canonical normalization: custom_tool_call_output -> function_call_output
            for item in body.get("input", []):
                if item.get("type") == "custom_tool_call_output":
                    item["type"] = "function_call_output"
            return body

        # Model cost map that marks custom_tool_call as unsupported
        model_cost = {
            "kimi-code": {
                "unsupported_input_item_types": ["custom_tool_call"],
            },
        }

        callbacks = codex_tool_policy.CodexToolPolicyCallbacks(
            normalize_tag_value=lambda x: x.lower().strip() if isinstance(x, str) else x,
            dedupe_sorted=lambda x: sorted(set(x)),
            merge_metadata=lambda body, **kw: body,
            build_span=lambda **kw: {},
            get_model_cost_map=lambda: model_cost,
            normalize_grok_native_oauth_model=lambda x: x,
            is_oa_xai_model=lambda x: False,
            resolve_oa_xai_upstream_model=lambda x: x,
            normalize_kimi_model_name=lambda x: "kimi-code" if "kimi" in str(x) else None,
            normalize_kimi_custom_tool_outputs=mock_normalize,
            grok_normalization=None,
            grok_normalization_runtime=None,
            request_body_walk_max_depth=10,
        )

        updated_body, removed_items = (
            codex_tool_policy.drop_unsupported_codex_input_items_from_request_body(
                request_body, callbacks=callbacks
            )
        )

        # Verify normalization was called
        assert normalize_called, "normalize_kimi_custom_tool_outputs must be called"

        # Verify custom_tool_call was removed (unsupported)
        assert len(removed_items) == 1
        assert removed_items[0]["type"] == "custom_tool_call"

        # Verify function_call_output was preserved (normalized from custom_tool_call_output)
        remaining_types = [item["type"] for item in updated_body["input"]]
        assert "function_call_output" in remaining_types
        assert "custom_tool_call_output" not in remaining_types
