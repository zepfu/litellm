"""Wave 6E Codex tool-policy extraction.

Owns spawn-agent/core tool description patches, model-capability policy
lookups, custom-tool-to-function adaptation, namespace-tool adaptation,
unsupported hosted-tool/parameter/input-item drops, tool-choice cleanup,
and Grok-native input-item policy facades.

Does NOT import ``llm_passthrough_endpoints`` at module scope.  All external
dependencies are injected through ``CodexToolPolicyCallbacks``.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Optional, Protocol

# ---------------------------------------------------------------------------
# Callback / config seams
# ---------------------------------------------------------------------------


class NormalizeTagValueFn(Protocol):
    def __call__(self, value: Any) -> Optional[str]: ...


class DedupeSortedFn(Protocol):
    def __call__(self, values: list[str]) -> list[str]: ...


class MergeMetadataFn(Protocol):
    def __call__(
        self,
        request_body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]: ...


class BuildSpanFn(Protocol):
    def __call__(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]: ...


class GetModelCostMapFn(Protocol):
    """Return the live ``litellm.model_cost`` dict."""

    def __call__(self) -> dict[str, Any]: ...


class NormalizeGrokModelFn(Protocol):
    def __call__(self, model: str) -> Optional[str]: ...


class IsOaXaiModelFn(Protocol):
    def __call__(self, model: str) -> bool: ...


class ResolveOaXaiModelFn(Protocol):
    def __call__(self, model: str) -> str: ...


class NormalizeKimiModelFn(Protocol):
    def __call__(self, model: Any) -> Optional[str]: ...


class NormalizeKimiToolOutputsFn(Protocol):
    def __call__(self, request_body: dict[str, Any]) -> dict[str, Any]: ...


class GetCodexToolPolicyCallbacksFn(Protocol):
    def __call__(self) -> "CodexToolPolicyCallbacks": ...


class GetCodexToolPolicyNormalizeTagValueFn(Protocol):
    def __call__(self) -> NormalizeTagValueFn: ...


@dataclass(frozen=True, slots=True)
class CodexToolPolicyCallbacks:
    """Explicit seams for every external dependency.

    Construct once at the integration boundary and thread through the
    public functions.  No module-scope god-module import is required.
    """

    normalize_tag_value: NormalizeTagValueFn
    dedupe_sorted: DedupeSortedFn
    merge_metadata: MergeMetadataFn
    build_span: BuildSpanFn
    get_model_cost_map: GetModelCostMapFn
    normalize_grok_native_oauth_model: NormalizeGrokModelFn
    is_oa_xai_model: IsOaXaiModelFn
    resolve_oa_xai_upstream_model: ResolveOaXaiModelFn
    normalize_kimi_model_name: NormalizeKimiModelFn
    normalize_kimi_custom_tool_outputs: NormalizeKimiToolOutputsFn
    # Grok normalization module and runtime are passed as opaque objects
    # to avoid importing the provider module at module scope.
    grok_normalization: Any = None
    grok_normalization_runtime: Any = None
    request_body_walk_max_depth: int = 64


@dataclass(frozen=True, slots=True)
class CodexToolPolicyRuntimeAccessors:
    """Late-bound access to host-owned policy callbacks."""

    get_callbacks: GetCodexToolPolicyCallbacksFn
    get_normalize_tag_value: GetCodexToolPolicyNormalizeTagValueFn


@dataclass(frozen=True, slots=True)
class CodexToolPolicyHostDeps:
    """Raw host dependencies needed to build CodexToolPolicyCallbacks.

    Pass an instance to :func:`configure_and_install_codex_tool_policy` to
    replace the god-module's inline callbacks construction and 42 thin-wrapper
    definitions with a single call.
    """

    normalize_tag_value: NormalizeTagValueFn
    dedupe_sorted: DedupeSortedFn
    merge_metadata: MergeMetadataFn
    build_span: BuildSpanFn
    get_model_cost_map: GetModelCostMapFn
    normalize_grok_native_oauth_model: NormalizeGrokModelFn
    is_oa_xai_model: IsOaXaiModelFn
    resolve_oa_xai_upstream_model: ResolveOaXaiModelFn
    normalize_kimi_model_name: NormalizeKimiModelFn
    normalize_kimi_custom_tool_outputs: NormalizeKimiToolOutputsFn
    grok_normalization: Any = None
    grok_normalization_runtime: Any = None
    request_body_walk_max_depth: int = 64


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODEX_SPAWN_AGENT_TOOL_NAME = "spawn_agent"
CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE = "tool_search"
CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID = "spawn-agent-fanout-policy"
CODEX_SPAWN_AGENT_PAYLOAD_SCHEMA_PATCH_ID = "spawn-agent-payload-schema"
CODEX_CORE_TOOL_GUIDANCE_PATCH_PREFIX = "core-tool-guidance"
CODEX_UNSUPPORTED_HOSTED_TOOLS_MODEL_INFO_FIELD = "unsupported_hosted_tools"
CODEX_UNSUPPORTED_REQUEST_PARAMS_MODEL_INFO_FIELD = "unsupported_request_params"
CODEX_UNSUPPORTED_INPUT_ITEM_TYPES_MODEL_INFO_FIELD = "unsupported_input_item_types"
CODEX_REWRITE_INPUT_ITEM_TYPES_MODEL_INFO_FIELD = "rewrite_input_item_types"
CODEX_CUSTOM_TOOL_FUNCTION_ADAPTERS_MODEL_INFO_FIELD = "custom_tool_function_adapters"
CODEX_NAMESPACE_TOOL_FUNCTION_ADAPTERS_MODEL_INFO_FIELD = (
    "namespace_tool_function_adapters"
)

CODEX_SPAWN_AGENT_FANOUT_POLICY = (
    "Use subagents to parallelize independent work while keeping one local owner "
    "on the critical path. Follow the current operator and project instructions "
    "that authorize fanout; do not treat generic depth or investigation wording "
    "as permission to launch unrelated autonomous fanout. Do not duplicate the "
    "same task across agents.\n\n"
    "Explicitly requested agent_type, model, or fork_turns values take precedence "
    "over all defaults. Copy each explicitly requested value exactly into every "
    "spawn_agent payload; do not substitute or omit those values on the first "
    "attempt or any retry. Apply defaults only when the current task did not "
    "provide an explicit value for that field.\n\n"
    "For read-only or exploration workers, call multi_agent_v1.spawn_agent with "
    "lower-case payload fields. If the current task did not explicitly provide "
    'a model, require an explicit supported alias or model name. If the current task '
    "did provide a model, keep that provided value unchanged and do not "
    "substitute any default, including on retries. Use "
    'fork_turns="none" unless context sharing is explicitly needed, and message '
    "containing the read-only boundary plus the audit task. If a fix is needed, "
    "the worker should describe the patch only.\n\n"
    "For coding workers, this read-only payload does not apply. Include the "
    "selected coding model from the configured coding-model priority order, "
    "assign a clear disjoint write set, and tell workers they are not alone in "
    "the codebase. They must not revert unrelated edits.\n\n"
    "Use the latest frontier model for cross-document architecture, migration-risk "
    "review, and high-stakes database safety reasoning. Use the latest Codex model "
    "for bounded implementation tasks with clear, disjoint write ownership. Use "
    "mini-class agents for narrow grep/read-only scans, documentation consistency "
    "checks, test inventory, and quick QA passes. For database or migration "
    "work, prefer read-only explorer subagents; the main owner should run live "
    "database commands so target verification and credential handling stay in "
    "one place."
)

CODEX_SPAWN_AGENT_PAYLOAD_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "agent_type": {
        "type": "string",
        "description": (
            "Optional configured agent role. Use a role whose config selects the "
            "required model and execution policy."
        ),
    },
    "model": {
        "type": "string",
        "description": (
            "Optional lower-case model override accepted by the orchestrator. "
            "If the current task provided a model, keep that value unchanged, "
            "including on retries. Otherwise use an explicit supported alias "
            "or model name for read-only/exploration workers, or the selected "
            "coding model for coding workers."
        ),
    },
    "fork_turns": {
        "type": "string",
        "enum": ["none", "all"],
        "description": (
            "Which parent turns to fork into the worker. Use none for isolated "
            "workers unless the complete parent context is explicitly required."
        ),
    },
    "message": {
        "type": "string",
        "description": (
            "Plain-text task prompt for the worker, including read-only or "
            "coding scope, file boundaries, and final-answer requirements."
        ),
    },
}

CODEX_SPAWN_AGENT_PAYLOAD_FIELD_ORDER = (
    "agent_type",
    "model",
    "fork_turns",
    "message",
)

CODEX_CORE_TOOL_GUIDANCE_BY_NAME: dict[str, str] = {
    "bash": (
        "Claude Code core tool reliability guidance: Use Bash for inspection, "
        "test, and simple commands. Prefer structured Edit or Write tools for "
        "source changes instead of complex sed, perl, awk, or shell-quoted "
        "rewrites. After a shell quoting or syntax error, do not retry a more "
        "complex one-liner; switch to a smaller structured edit or report the "
        "exact blocker."
    ),
    "edit": (
        "Claude Code core tool reliability guidance: Edit old_string must be "
        "copied from the current file contents. If an Edit fails with "
        "`String to replace not found in file`, do not retry the same "
        "old_string. Re-read the exact target span, narrow the hunk to the "
        "smallest stable current context, and then retry once with current "
        "text."
    ),
    "read": (
        "Claude Code core tool reliability guidance: Use bounded reads for "
        "large transcript, task-output, or log files. For .output transcript "
        "files, use offset/limit or available transcript search/meta tools "
        "instead of unbounded full-file reads."
    ),
    "write": (
        "Claude Code core tool reliability guidance: Use Write for new files or "
        "known full-file replacements. Before overwriting an existing file, read "
        "the current file first and preserve unrelated content."
    ),
}

CODEX_SPAWN_AGENT_RESTRICTIVE_DESCRIPTION_PATTERNS = (
    re.compile(
        r"Only use `?spawn_agent`? if and only if the user explicitly asks for "
        r"sub-?agents, delegation, or parallel agent work\.\s*"
        r"Requests for depth, thoroughness, research, investigation, or detailed "
        r"codebase analysis do not count as permission to spawn\.\s*"
        r"Agent-role guidance below only helps choose which agent to use after "
        r"spawning is already authorized; it never authorizes spawning by itself\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"Only use `?spawn_agent`? if and only if the user explicitly asks for "
        r"sub-?agents, delegation, or parallel agent work\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"I may only use `?spawn_agent`? when the user explicitly asks for "
        r"sub-?agents, delegation, or parallel agent work\.",
        re.IGNORECASE,
    ),
)


# ---------------------------------------------------------------------------
# Pure helpers (no external deps)
# ---------------------------------------------------------------------------


def get_openai_tool_name(tool: dict[str, Any]) -> Optional[str]:
    """Extract the tool name from an OpenAI-shaped tool definition."""
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = tool.get("function")
    if isinstance(function, dict):
        function_name = function.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name.strip()
    return None


def get_openai_tool_type(tool: dict[str, Any]) -> Optional[str]:
    """Extract the tool type from an OpenAI-shaped tool definition."""
    tool_type = tool.get("type")
    if isinstance(tool_type, str) and tool_type.strip():
        return tool_type.strip()
    return None


def extract_openai_passthrough_tool_choice(
    value: Any,
    *,
    normalize_tag_value: NormalizeTagValueFn,
) -> Optional[str]:
    """Normalize a tool_choice value to a low-cardinality tag string.

    Uses the configured *normalize_tag_value* callback exactly like the
    god-module path so that bool and non-string values are handled
    consistently.
    """
    if isinstance(value, str):
        return normalize_tag_value(value)
    if isinstance(value, dict):
        for key in ("type", "name"):
            normalized = normalize_tag_value(value.get(key))
            if normalized:
                return normalized
    return None


# ---------------------------------------------------------------------------
# Spawn-agent description patching
# ---------------------------------------------------------------------------


def patch_codex_spawn_agent_description_text(description: str) -> tuple[str, int]:
    """Ensure a spawn_agent description includes the fanout policy."""
    if CODEX_SPAWN_AGENT_FANOUT_POLICY in description:
        return description, 0

    updated_description = description
    replacement_count = 0
    for pattern in CODEX_SPAWN_AGENT_RESTRICTIVE_DESCRIPTION_PATTERNS:
        updated_description, count = pattern.subn(
            CODEX_SPAWN_AGENT_FANOUT_POLICY,
            updated_description,
            count=1,
        )
        replacement_count += count
    if replacement_count:
        return updated_description, replacement_count
    if not description.strip():
        return CODEX_SPAWN_AGENT_FANOUT_POLICY, 0
    return f"{description.rstrip()}\n\n{CODEX_SPAWN_AGENT_FANOUT_POLICY}", 0


def patch_codex_spawn_agent_payload_parameters(
    parameters: Any,
) -> tuple[Any, list[str], list[str]]:
    """Ensure spawn_agent payload has canonical fields; remove fork_context."""
    if parameters is None:
        updated_parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
        }
    elif isinstance(parameters, dict):
        updated_parameters = copy.deepcopy(parameters)
        if "type" not in updated_parameters:
            updated_parameters["type"] = "object"
    else:
        return parameters, [], []

    properties = updated_parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    else:
        properties = dict(properties)

    removed_fields: list[str] = []
    if "fork_context" in properties:
        properties.pop("fork_context")
        removed_fields.append("fork_context")

    added_fields: list[str] = []
    for field_name in CODEX_SPAWN_AGENT_PAYLOAD_FIELD_ORDER:
        if field_name in properties:
            continue
        properties[field_name] = copy.deepcopy(
            CODEX_SPAWN_AGENT_PAYLOAD_FIELD_SCHEMAS[field_name]
        )
        added_fields.append(field_name)

    required = updated_parameters.get("required")
    if isinstance(required, list) and "fork_context" in required:
        updated_parameters["required"] = [
            field_name for field_name in required if field_name != "fork_context"
        ]

    if not added_fields and not removed_fields:
        return parameters, [], []

    updated_parameters["properties"] = properties
    return updated_parameters, added_fields, removed_fields


def _apply_spawn_agent_parameter_patches(
    updated_tool: dict[str, Any],
    original_tool: dict[str, Any],
    *,
    tool_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Patch spawn_agent payload parameters, returning (tool, events)."""
    patch_events: list[dict[str, Any]] = []
    parameter_targets: list[tuple[str, str]] = []
    function = updated_tool.get("function")
    if isinstance(function, dict):
        parameter_targets.append(("function", f"tools.{tool_index}.function.parameters"))
    if "parameters" in updated_tool or not parameter_targets:
        parameter_targets.append(("tool", f"tools.{tool_index}.parameters"))

    for target_kind, path in parameter_targets:
        if target_kind == "function":
            current_function = updated_tool.get("function")
            if not isinstance(current_function, dict):
                continue
            parameters = current_function.get("parameters")
        else:
            parameters = updated_tool.get("parameters")

        (
            updated_parameters,
            added_fields,
            removed_fields,
        ) = patch_codex_spawn_agent_payload_parameters(parameters)
        if (not added_fields and not removed_fields) or updated_parameters is parameters:
            continue

        if updated_tool is original_tool:
            updated_tool = dict(original_tool)

        if target_kind == "function":
            current_function = updated_tool.get("function")
            if not isinstance(current_function, dict):
                continue
            updated_function = dict(current_function)
            updated_function["parameters"] = updated_parameters
            updated_tool["function"] = updated_function
        else:
            updated_tool["parameters"] = updated_parameters

        patch_events.append(
            {
                "id": CODEX_SPAWN_AGENT_PAYLOAD_SCHEMA_PATCH_ID,
                "status": "applied",
                "tool_name": CODEX_SPAWN_AGENT_TOOL_NAME,
                "path": path,
                "fields_added": added_fields,
                "fields_removed": removed_fields,
                "occurrences": 0,
            }
        )

    return updated_tool, patch_events


def patch_codex_spawn_agent_tool_description(
    tool: dict[str, Any],
    *,
    tool_index: int,
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Patch spawn_agent tool description and payload parameters."""
    if get_openai_tool_name(tool) != CODEX_SPAWN_AGENT_TOOL_NAME:
        return tool, []

    updated_tool = tool
    patch_events: list[dict[str, Any]] = []
    description_targets: list[tuple[dict[str, Any], str, str]] = [
        (tool, "description", f"tools.{tool_index}.description")
    ]
    function = tool.get("function")
    if isinstance(function, dict):
        description_targets.append(
            (
                function,
                "description",
                f"tools.{tool_index}.function.description",
            )
        )

    for container, key, path in description_targets:
        description = container.get(key)
        if not isinstance(description, str):
            continue

        (
            updated_description,
            replacement_count,
        ) = patch_codex_spawn_agent_description_text(description)
        if updated_description == description:
            continue

        if updated_tool is tool:
            updated_tool = dict(tool)

        if container is tool:
            updated_tool[key] = updated_description
        else:
            updated_function = dict(container)
            updated_function[key] = updated_description
            updated_tool["function"] = updated_function

        patch_events.append(
            {
                "id": CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID,
                "status": "applied",
                "tool_name": CODEX_SPAWN_AGENT_TOOL_NAME,
                "path": path,
                "occurrences": replacement_count,
            }
        )

    updated_tool, param_events = _apply_spawn_agent_parameter_patches(
        updated_tool, tool, tool_index=tool_index
    )
    patch_events.extend(param_events)

    return updated_tool, patch_events


# ---------------------------------------------------------------------------
# Core tool guidance
# ---------------------------------------------------------------------------


def get_codex_core_tool_guidance(
    tool_name: Optional[str],
    *,
    normalize_tag_value: NormalizeTagValueFn,
) -> Optional[str]:
    normalized_tool_name = normalize_tag_value(tool_name)
    if not normalized_tool_name:
        return None
    return CODEX_CORE_TOOL_GUIDANCE_BY_NAME.get(normalized_tool_name)


def append_codex_core_tool_guidance_to_description(
    description: Any,
    *,
    guidance: str,
) -> tuple[str, bool]:
    existing_description = description if isinstance(description, str) else ""
    if guidance in existing_description:
        return existing_description, False
    if not existing_description.strip():
        return guidance, True
    return f"{existing_description.rstrip()}\n\n{guidance}", True


def patch_codex_multi_agent_tool_search_description(
    tool: dict[str, Any],
    *,
    tool_index: int,
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if normalize_tag_value(tool.get("type")) != CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE:
        return tool, []

    description = tool.get("description")
    if not isinstance(description, str):
        return tool, []
    if CODEX_SPAWN_AGENT_FANOUT_POLICY in description:
        return tool, []
    if (
        "Multi-agent tools" not in description
        and "Spawn and manage sub-agents" not in description
    ):
        return tool, []

    updated_tool = dict(tool)
    updated_tool["description"] = (
        f"{description.rstrip()}\n\n{CODEX_SPAWN_AGENT_FANOUT_POLICY}"
    )
    return updated_tool, [
        {
            "id": CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID,
            "status": "applied",
            "tool_name": CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE,
            "path": f"tools.{tool_index}.description",
            "occurrences": 0,
            "guidance_chars": len(CODEX_SPAWN_AGENT_FANOUT_POLICY),
        }
    ]


def patch_codex_core_tool_description(
    tool: dict[str, Any],
    *,
    tool_index: int,
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool_name = get_openai_tool_name(tool)
    guidance = get_codex_core_tool_guidance(
        tool_name, normalize_tag_value=normalize_tag_value
    )
    if guidance is None:
        return tool, []

    updated_tool = tool
    patch_events: list[dict[str, Any]] = []
    description_targets: list[tuple[dict[str, Any], str, str]] = []
    function = tool.get("function")
    if isinstance(function, dict):
        description_targets.append(
            (
                function,
                "description",
                f"tools.{tool_index}.function.description",
            )
        )
    if "description" in tool or not description_targets:
        description_targets.append(
            (tool, "description", f"tools.{tool_index}.description")
        )

    for container, key, path in description_targets:
        updated_description, changed = append_codex_core_tool_guidance_to_description(
            container.get(key),
            guidance=guidance,
        )
        if not changed:
            continue

        if updated_tool is tool:
            updated_tool = dict(tool)

        if container is tool:
            updated_tool[key] = updated_description
        else:
            updated_function = dict(container)
            updated_function[key] = updated_description
            updated_tool["function"] = updated_function

        normalized_tool_name = normalize_tag_value(tool_name) or "unknown"
        patch_events.append(
            {
                "id": f"{CODEX_CORE_TOOL_GUIDANCE_PATCH_PREFIX}-{normalized_tool_name}",
                "status": "applied",
                "tool_name": tool_name,
                "path": path,
                "occurrences": 0,
                "guidance_chars": len(guidance),
            }
        )

    return updated_tool, patch_events


# ---------------------------------------------------------------------------
# Model-capability policy lookups
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_bundled_model_cost_map_for_codex_policy() -> dict[str, Any]:
    try:
        content = (
            files("litellm")
            .joinpath("bundled_model_prices_and_context_window_fallback.json")
            .read_text(encoding="utf-8")
        )
        loaded = json.loads(content)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return {}
    return {}


def get_codex_tool_policy_model_cost_candidates(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> list[str]:
    if not isinstance(model, str) or not model.strip():
        return []

    model_name = model.strip()
    split_model_name = (
        model_name.split("/", 1)[1] if "/" in model_name else model_name
    )
    candidates = [
        model_name,
        model_name.lower(),
        split_model_name,
        split_model_name.lower(),
        f"chatgpt/{split_model_name}",
        f"chatgpt/{split_model_name.lower()}",
        f"openai/{split_model_name}",
        f"openai/{split_model_name.lower()}",
    ]
    grok_native_model = callbacks.normalize_grok_native_oauth_model(model_name)
    if grok_native_model is not None:
        candidates.extend(
            [
                f"xai/{grok_native_model}",
                f"xai/{grok_native_model.lower()}",
            ]
        )
    if callbacks.is_oa_xai_model(model_name):
        try:
            xai_oauth_upstream_model = callbacks.resolve_oa_xai_upstream_model(
                model_name
            )
            candidates.extend(
                [
                    xai_oauth_upstream_model,
                    xai_oauth_upstream_model.lower(),
                ]
            )
        except Exception:
            pass

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _lookup_model_info_field(
    model: Any,
    field_name: str,
    *,
    expected_type: type,
    callbacks: CodexToolPolicyCallbacks,
) -> Optional[Any]:
    """Shared lookup across live model_cost and bundled fallback.

    Continues across all model-cost sources and candidate keys until a
    value of *expected_type* is found, rather than stopping on the first
    non-None wrong-typed value.
    """
    candidate_keys = get_codex_tool_policy_model_cost_candidates(
        model, callbacks=callbacks
    )
    if not candidate_keys:
        return None

    model_cost_sources = [
        callbacks.get_model_cost_map(),
        load_bundled_model_cost_map_for_codex_policy(),
    ]
    for model_cost in model_cost_sources:
        for key in candidate_keys:
            model_info = model_cost.get(key)
            if not isinstance(model_info, dict):
                continue
            value = model_info.get(field_name)
            if isinstance(value, expected_type):
                return value
    return None


def get_unsupported_hosted_tool_types_for_model(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> set[str]:
    raw = _lookup_model_info_field(
        model,
        CODEX_UNSUPPORTED_HOSTED_TOOLS_MODEL_INFO_FIELD,
        expected_type=list,
        callbacks=callbacks,
    )
    if not isinstance(raw, list):
        return set()
    return {
        normalized
        for value in raw
        if (normalized := callbacks.normalize_tag_value(value))
    }


def get_unsupported_request_param_names_for_model(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> set[str]:
    raw = _lookup_model_info_field(
        model,
        CODEX_UNSUPPORTED_REQUEST_PARAMS_MODEL_INFO_FIELD,
        expected_type=list,
        callbacks=callbacks,
    )
    if not isinstance(raw, list):
        return set()
    return {
        normalized
        for value in raw
        if (normalized := callbacks.normalize_tag_value(value))
    }


def get_unsupported_input_item_types_for_model(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> set[str]:
    raw = _lookup_model_info_field(
        model,
        CODEX_UNSUPPORTED_INPUT_ITEM_TYPES_MODEL_INFO_FIELD,
        expected_type=list,
        callbacks=callbacks,
    )
    if not isinstance(raw, list):
        return set()
    return {
        normalized
        for value in raw
        if (normalized := callbacks.normalize_tag_value(value))
    }


def get_rewrite_input_item_types_for_model(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> set[str]:
    raw = _lookup_model_info_field(
        model,
        CODEX_REWRITE_INPUT_ITEM_TYPES_MODEL_INFO_FIELD,
        expected_type=list,
        callbacks=callbacks,
    )
    if not isinstance(raw, list):
        return set()
    return {
        normalized
        for value in raw
        if (normalized := callbacks.normalize_tag_value(value))
    }


def get_custom_tool_function_adapter_names_for_model(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> set[str]:
    raw = _lookup_model_info_field(
        model,
        CODEX_CUSTOM_TOOL_FUNCTION_ADAPTERS_MODEL_INFO_FIELD,
        expected_type=list,
        callbacks=callbacks,
    )
    if not isinstance(raw, list):
        return set()
    return {
        normalized
        for value in raw
        if (normalized := callbacks.normalize_tag_value(value))
    }


def get_namespace_tool_function_adapter_names_for_model(
    model: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, set[str]]:
    raw = _lookup_model_info_field(
        model,
        CODEX_NAMESPACE_TOOL_FUNCTION_ADAPTERS_MODEL_INFO_FIELD,
        expected_type=dict,
        callbacks=callbacks,
    )
    if not isinstance(raw, dict):
        return {}

    normalized_adapters: dict[str, set[str]] = {}
    for namespace, tool_names in raw.items():
        normalized_namespace = callbacks.normalize_tag_value(namespace)
        if normalized_namespace is None or not isinstance(tool_names, list):
            continue
        normalized_names = {
            normalized
            for value in tool_names
            if (normalized := callbacks.normalize_tag_value(value))
        }
        if normalized_names:
            normalized_adapters[normalized_namespace] = normalized_names
    return normalized_adapters


# ---------------------------------------------------------------------------
# Custom-tool-to-function adaptation
# ---------------------------------------------------------------------------


def adapted_custom_tool_function_schema(
    tool: dict[str, Any],
    *,
    tool_name: str,
) -> dict[str, Any]:
    adapted_tool: dict[str, Any] = {
        "type": "function",
        "name": tool_name,
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": (
                        "Raw input for the client-hosted custom tool. For "
                        "apply_patch this must be the complete patch text."
                    ),
                }
            },
            "required": ["input"],
            "additionalProperties": False,
        },
    }
    description = tool.get("description")
    if isinstance(description, str) and description.strip():
        adapted_tool["description"] = description
    return adapted_tool


def adapt_codex_custom_tool_definitions(
    tools: Any,
    *,
    adapter_names: set[str],
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[Optional[list[Any]], list[dict[str, Any]]]:
    if not isinstance(tools, list):
        return None, []

    updated_tools: list[Any] = []
    adapted_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue
        tool_type = normalize_tag_value(get_openai_tool_type(tool))
        tool_name = normalize_tag_value(get_openai_tool_name(tool))
        if tool_type != "custom" or tool_name not in adapter_names:
            updated_tools.append(tool)
            continue
        updated_tools.append(
            adapted_custom_tool_function_schema(
                tool,
                tool_name=tool_name,
            )
        )
        adapted_tools.append(
            {
                "name": tool_name,
                "tool_index": index,
            }
        )
    return updated_tools, adapted_tools


def adapted_custom_tool_call_ids(
    input_items: Any,
    *,
    adapter_names: set[str],
    normalize_tag_value: NormalizeTagValueFn,
) -> set[str]:
    if not isinstance(input_items, list):
        return set()

    adapted_call_ids: set[str] = set()
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "custom_tool_call":
            continue
        item_name = normalize_tag_value(item.get("name"))
        call_id = item.get("call_id")
        if (
            item_name in adapter_names
            and isinstance(call_id, str)
            and call_id.strip()
        ):
            adapted_call_ids.add(call_id.strip())
    return adapted_call_ids


def adapt_codex_custom_tool_input_items(
    input_items: Any,
    *,
    adapter_names: set[str],
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[Optional[list[Any]], list[dict[str, Any]]]:
    adapted_call_ids = adapted_custom_tool_call_ids(
        input_items,
        adapter_names=adapter_names,
        normalize_tag_value=normalize_tag_value,
    )
    if not isinstance(input_items, list) or not adapted_call_ids:
        return None, []

    updated_input_items: list[Any] = []
    adapted_input_items: list[dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue

        item_type = item.get("type")
        call_id = item.get("call_id")
        normalized_call_id = (
            call_id.strip() if isinstance(call_id, str) and call_id.strip() else None
        )
        if item_type == "custom_tool_call":
            item_name = normalize_tag_value(item.get("name"))
            raw_input = item.get("input")
            if (
                item_name in adapter_names
                and normalized_call_id in adapted_call_ids
                and isinstance(raw_input, str)
            ):
                adapted_item = dict(item)
                adapted_item["type"] = "function_call"
                adapted_item["arguments"] = json.dumps(
                    {"input": raw_input},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                adapted_item.pop("input", None)
                updated_input_items.append(adapted_item)
                adapted_input_items.append(
                    {
                        "type": item_type,
                        "name": item_name,
                        "input_index": index,
                    }
                )
                continue
        elif (
            item_type == "custom_tool_call_output"
            and normalized_call_id in adapted_call_ids
        ):
            adapted_item = dict(item)
            adapted_item["type"] = "function_call_output"
            updated_input_items.append(adapted_item)
            adapted_input_items.append(
                {
                    "type": item_type,
                    "input_index": index,
                }
            )
            continue

        updated_input_items.append(item)
    return updated_input_items, adapted_input_items


def adapt_codex_custom_tool_choice(
    tool_choice: Any,
    *,
    adapter_names: set[str],
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[Any, bool]:
    if not isinstance(tool_choice, dict):
        return tool_choice, False
    tool_choice_type = normalize_tag_value(tool_choice.get("type"))
    tool_choice_name = normalize_tag_value(tool_choice.get("name"))
    if tool_choice_type != "custom" or tool_choice_name not in adapter_names:
        return tool_choice, False
    return (
        {
            **tool_choice,
            "type": "function",
            "name": tool_choice_name,
        },
        True,
    )


def add_codex_custom_tool_function_adapter_logging_metadata(
    request_body: dict[str, Any],
    *,
    adapted_tools: list[dict[str, Any]],
    adapted_input_items: list[dict[str, Any]],
    adapted_tool_choice: bool,
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    adapted_names = callbacks.dedupe_sorted(
        [
            str(item["name"])
            for item in adapted_tools
            if isinstance(item.get("name"), str)
        ]
    )
    updated_body = callbacks.merge_metadata(
        request_body,
        tags_to_add=[
            "codex-custom-tool-function-adapted",
            *(f"codex-custom-tool-function:{name}" for name in adapted_names),
        ],
        extra_fields={
            "codex_custom_tool_function_adapter_count": len(adapted_tools),
            "codex_custom_tool_function_adapter_names": adapted_names,
            "codex_custom_tool_function_adapter_tools": adapted_tools,
            "codex_custom_tool_function_adapter_input_item_count": len(
                adapted_input_items
            ),
            "codex_custom_tool_function_adapter_input_items": adapted_input_items,
            "codex_custom_tool_function_adapter_tool_choice": adapted_tool_choice,
            "langfuse_spans": [
                callbacks.build_span(
                    name="codex.custom_tool_function_adapted",
                    metadata={
                        "tool_count": len(adapted_tools),
                        "tool_names": adapted_names,
                        "input_item_count": len(adapted_input_items),
                        "tool_choice_adapted": adapted_tool_choice,
                    },
                )
            ],
        },
    )
    return updated_body


def adapt_codex_custom_tools_to_functions_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter_names = get_custom_tool_function_adapter_names_for_model(
        request_body.get("model"), callbacks=callbacks
    )
    if not adapter_names:
        return request_body, []

    updated_tools, adapted_tools = adapt_codex_custom_tool_definitions(
        request_body.get("tools"),
        adapter_names=adapter_names,
        normalize_tag_value=callbacks.normalize_tag_value,
    )
    updated_input_items, adapted_input_items = adapt_codex_custom_tool_input_items(
        request_body.get("input"),
        adapter_names=adapter_names,
        normalize_tag_value=callbacks.normalize_tag_value,
    )
    updated_tool_choice, adapted_tool_choice = adapt_codex_custom_tool_choice(
        request_body.get("tool_choice"),
        adapter_names=adapter_names,
        normalize_tag_value=callbacks.normalize_tag_value,
    )
    if not adapted_tools and not adapted_input_items and not adapted_tool_choice:
        return request_body, []

    updated_body = dict(request_body)
    if updated_tools is not None and adapted_tools:
        updated_body["tools"] = updated_tools
    if updated_input_items is not None and adapted_input_items:
        updated_body["input"] = updated_input_items
    if adapted_tool_choice:
        updated_body["tool_choice"] = updated_tool_choice
    updated_body = add_codex_custom_tool_function_adapter_logging_metadata(
        updated_body,
        adapted_tools=adapted_tools,
        adapted_input_items=adapted_input_items,
        adapted_tool_choice=adapted_tool_choice,
        callbacks=callbacks,
    )
    return updated_body, adapted_tools


# ---------------------------------------------------------------------------
# Namespace-tool adaptation
# ---------------------------------------------------------------------------


def adapt_codex_namespace_tool_definitions(
    tools: Any,
    *,
    adapter_names: dict[str, set[str]],
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[
    Optional[list[Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not isinstance(tools, list):
        return None, [], []

    occupied_names = {
        normalized_name
        for tool in tools
        if isinstance(tool, dict)
        and normalize_tag_value(get_openai_tool_type(tool)) != "namespace"
        and (normalized_name := normalize_tag_value(get_openai_tool_name(tool)))
    }
    updated_tools: list[Any] = []
    adapted_tools: list[dict[str, Any]] = []
    skipped_tools: list[dict[str, Any]] = []
    changed = False

    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue

        tool_type = normalize_tag_value(get_openai_tool_type(tool))
        namespace = normalize_tag_value(get_openai_tool_name(tool))
        allowed_names = adapter_names.get(namespace or "")
        if tool_type != "namespace" or allowed_names is None:
            updated_tools.append(tool)
            continue

        changed = True
        namespace_tools = tool.get("tools")
        if not isinstance(namespace_tools, list):
            skipped_tools.append(
                {
                    "namespace": namespace,
                    "tool_index": tool_index,
                    "reason": "tools_not_list",
                }
            )
            continue

        for child_index, child in enumerate(namespace_tools):
            skip_detail: dict[str, Any] = {
                "namespace": namespace,
                "tool_index": tool_index,
                "child_index": child_index,
            }
            if not isinstance(child, dict):
                skipped_tools.append({**skip_detail, "reason": "child_not_object"})
                continue

            child_type = normalize_tag_value(get_openai_tool_type(child))
            child_name = normalize_tag_value(get_openai_tool_name(child))
            if child_type != "function":
                skipped_tools.append(
                    {
                        **skip_detail,
                        "name": child_name,
                        "reason": "child_not_function",
                    }
                )
                continue
            if child_name not in allowed_names:
                skipped_tools.append(
                    {
                        **skip_detail,
                        "name": child_name,
                        "reason": "child_not_configured",
                    }
                )
                continue
            if not isinstance(child.get("parameters"), dict):
                skipped_tools.append(
                    {
                        **skip_detail,
                        "name": child_name,
                        "reason": "parameters_not_object",
                    }
                )
                continue
            if child_name in occupied_names:
                skipped_tools.append(
                    {
                        **skip_detail,
                        "name": child_name,
                        "reason": "name_collision",
                    }
                )
                continue

            adapted_tool: dict[str, Any] = {
                "type": "function",
                "name": child_name,
                "parameters": copy.deepcopy(child["parameters"]),
            }
            description = child.get("description")
            if isinstance(description, str) and description.strip():
                adapted_tool["description"] = description
            strict = child.get("strict")
            if isinstance(strict, bool):
                adapted_tool["strict"] = strict

            updated_tools.append(adapted_tool)
            occupied_names.add(child_name)
            adapted_tools.append(
                {
                    "namespace": namespace,
                    "name": child_name,
                    "tool_index": tool_index,
                    "child_index": child_index,
                }
            )

    return (updated_tools if changed else None), adapted_tools, skipped_tools


def adapt_codex_namespace_input_items(
    input_items: Any,
    *,
    adapter_names: dict[str, set[str]],
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[Optional[list[Any]], list[dict[str, Any]]]:
    if not isinstance(input_items, list):
        return None, []

    updated_items: list[Any] = []
    adapted_items: list[dict[str, Any]] = []
    changed = False
    for input_index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_items.append(item)
            continue

        namespace = normalize_tag_value(item.get("namespace"))
        item_name = normalize_tag_value(item.get("name"))
        if (
            namespace is None
            or item_name not in adapter_names.get(namespace, set())
            or item.get("type") not in {"function_call", "custom_tool_call"}
        ):
            updated_items.append(item)
            continue

        adapted_item = dict(item)
        original_type = adapted_item.get("type")
        adapted_item.pop("namespace", None)
        if original_type == "custom_tool_call":
            raw_input = adapted_item.get("input")
            if not isinstance(raw_input, str):
                updated_items.append(item)
                continue
            adapted_item["type"] = "function_call"
            adapted_item["arguments"] = raw_input
            adapted_item.pop("input", None)

        updated_items.append(adapted_item)
        adapted_items.append(
            {
                "namespace": namespace,
                "name": item_name,
                "input_index": input_index,
                "source_type": original_type,
            }
        )
        changed = True

    return (updated_items if changed else None), adapted_items


def adapt_codex_namespace_tool_choice(
    tool_choice: Any,
    *,
    adapter_names: dict[str, set[str]],
    normalize_tag_value: NormalizeTagValueFn,
) -> tuple[Any, Optional[dict[str, str]]]:
    if not isinstance(tool_choice, dict):
        return tool_choice, None

    namespace = normalize_tag_value(tool_choice.get("namespace"))
    tool_name = normalize_tag_value(tool_choice.get("name"))
    if namespace is None or tool_name not in adapter_names.get(namespace, set()):
        return tool_choice, None

    tool_choice_type = normalize_tag_value(tool_choice.get("type"))
    if tool_choice_type not in {"custom", "function"}:
        return tool_choice, None

    updated_choice = {
        "type": "function",
        "function": {"name": tool_name},
    }
    return updated_choice, {"namespace": namespace, "name": tool_name}


def add_codex_namespace_tool_function_adapter_logging_metadata(
    request_body: dict[str, Any],
    *,
    adapted_tools: list[dict[str, Any]],
    adapted_input_items: list[dict[str, Any]],
    adapted_tool_choice: Optional[dict[str, str]],
    skipped_tools: list[dict[str, Any]],
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    adapted_namespaces = callbacks.dedupe_sorted(
        [
            str(item["namespace"])
            for item in adapted_tools
            if isinstance(item.get("namespace"), str)
        ]
    )
    adapted_names = callbacks.dedupe_sorted(
        [
            str(item["name"])
            for item in adapted_tools
            if isinstance(item.get("name"), str)
        ]
    )
    tags_to_add = ["codex-namespace-tool-function-adapted"]
    tags_to_add.extend(
        f"codex-namespace-tool-function:{namespace}"
        for namespace in adapted_namespaces
    )
    if skipped_tools:
        tags_to_add.append("codex-namespace-tool-function-skipped")

    span_metadata: dict[str, Any] = {
        "tool_count": len(adapted_tools),
        "namespaces": adapted_namespaces,
        "tool_names": adapted_names,
        "input_item_count": len(adapted_input_items),
        "tool_choice_adapted": adapted_tool_choice is not None,
        "skipped_count": len(skipped_tools),
    }
    return callbacks.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_namespace_tool_function_adapter_count": len(adapted_tools),
            "codex_namespace_tool_function_adapter_namespaces": adapted_namespaces,
            "codex_namespace_tool_function_adapter_names": adapted_names,
            "codex_namespace_tool_function_adapter_tools": adapted_tools,
            "codex_namespace_tool_function_adapter_input_item_count": len(
                adapted_input_items
            ),
            "codex_namespace_tool_function_adapter_input_items": adapted_input_items,
            "codex_namespace_tool_function_adapter_tool_choice": adapted_tool_choice,
            "codex_namespace_tool_function_adapter_skipped_count": len(skipped_tools),
            "codex_namespace_tool_function_adapter_skipped_tools": skipped_tools,
            "langfuse_spans": [
                callbacks.build_span(
                    name="codex.namespace_tool_function_adapted",
                    metadata=span_metadata,
                )
            ],
        },
    )


def adapt_codex_namespace_tools_to_functions_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter_names = get_namespace_tool_function_adapter_names_for_model(
        request_body.get("model"), callbacks=callbacks
    )
    if not adapter_names:
        return request_body, []

    updated_tools, adapted_tools, skipped_tools = (
        adapt_codex_namespace_tool_definitions(
            request_body.get("tools"),
            adapter_names=adapter_names,
            normalize_tag_value=callbacks.normalize_tag_value,
        )
    )
    active_adapter_names = adapter_names
    if isinstance(request_body.get("tools"), list):
        active_adapter_names = {}
        for adapted_tool in adapted_tools:
            namespace = adapted_tool.get("namespace")
            tool_name = adapted_tool.get("name")
            if isinstance(namespace, str) and isinstance(tool_name, str):
                active_adapter_names.setdefault(namespace, set()).add(tool_name)
    updated_input, adapted_input_items = adapt_codex_namespace_input_items(
        request_body.get("input"),
        adapter_names=active_adapter_names,
        normalize_tag_value=callbacks.normalize_tag_value,
    )
    updated_tool_choice, adapted_tool_choice = adapt_codex_namespace_tool_choice(
        request_body.get("tool_choice"),
        adapter_names=active_adapter_names,
        normalize_tag_value=callbacks.normalize_tag_value,
    )
    if (
        updated_tools is None
        and updated_input is None
        and adapted_tool_choice is None
        and not skipped_tools
    ):
        return request_body, []

    updated_body = dict(request_body)
    if updated_tools is not None:
        updated_body["tools"] = updated_tools
    if updated_input is not None:
        updated_body["input"] = updated_input
    if adapted_tool_choice is not None:
        updated_body["tool_choice"] = updated_tool_choice
    updated_body = add_codex_namespace_tool_function_adapter_logging_metadata(
        updated_body,
        adapted_tools=adapted_tools,
        adapted_input_items=adapted_input_items,
        adapted_tool_choice=adapted_tool_choice,
        skipped_tools=skipped_tools,
        callbacks=callbacks,
    )
    return updated_body, adapted_tools


# ---------------------------------------------------------------------------
# Unsupported hosted-tool / parameter / input-item drops
# ---------------------------------------------------------------------------


def openai_tool_choice_references_tool_type(
    tool_choice: Any,
    tool_types: set[str],
    *,
    normalize_tag_value: NormalizeTagValueFn,
) -> bool:
    if not tool_types:
        return False

    candidates: list[Any] = []
    if isinstance(tool_choice, str):
        candidates.append(tool_choice)
    elif isinstance(tool_choice, dict):
        candidates.extend([tool_choice.get("type"), tool_choice.get("name")])
        function = tool_choice.get("function")
        if isinstance(function, dict):
            candidates.append(function.get("name"))

    for candidate in candidates:
        normalized = normalize_tag_value(candidate)
        if normalized in tool_types:
            return True
    return False


def add_codex_unsupported_hosted_tool_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_tools: list[dict[str, Any]],
    removed_tool_choice: Optional[Any],
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    removed_tool_types = callbacks.dedupe_sorted(
        [
            tool["type"]
            for tool in removed_tools
            if isinstance(tool.get("type"), str) and tool["type"]
        ]
    )
    span_metadata: dict[str, Any] = {
        "removed_count": len(removed_tools),
        "removed_tool_types": removed_tool_types,
    }
    if removed_tool_choice is not None:
        span_metadata["removed_tool_choice"] = removed_tool_choice

    tags_to_add = ["codex-unsupported-hosted-tool-removed"]
    tags_to_add.extend(
        f"codex-unsupported-hosted-tool:{tool_type}"
        for tool_type in removed_tool_types
    )
    if removed_tool_choice is not None:
        tags_to_add.append("codex-unsupported-hosted-tool-choice-removed")

    extra_fields: dict[str, Any] = {
        "codex_unsupported_hosted_tool_removed_count": len(removed_tools),
        "codex_unsupported_hosted_tool_types_removed": removed_tool_types,
        "codex_unsupported_hosted_tools_removed": removed_tools,
        "langfuse_spans": [
            callbacks.build_span(
                name="codex.unsupported_hosted_tool_removed",
                metadata=span_metadata,
            )
        ],
    }
    if removed_tool_choice is not None:
        extra_fields["codex_unsupported_hosted_tool_choice_removed"] = (
            removed_tool_choice
        )

    return callbacks.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def request_has_openai_tool_definitions(request_body: dict[str, Any]) -> bool:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return False

    for tool in tools:
        if isinstance(tool, dict) and get_openai_tool_type(tool):
            return True
    return False


def add_tool_choice_without_tools_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_tool_choice: Any,
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    span_metadata: dict[str, Any] = {
        "removed_tool_choice": removed_tool_choice,
        "reason": "missing_tools",
    }
    extracted_tool_choice = extract_openai_passthrough_tool_choice(
        removed_tool_choice,
        normalize_tag_value=callbacks.normalize_tag_value,
    )
    if extracted_tool_choice:
        span_metadata["tool_choice"] = extracted_tool_choice

    tags_to_add = ["xai-tool-choice-without-tools-removed"]
    if extracted_tool_choice:
        tags_to_add.append(f"xai-tool-choice-without-tools:{extracted_tool_choice}")

    return callbacks.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "xai_tool_choice_without_tools_removed": removed_tool_choice,
            "xai_tool_choice_without_tools_removed_reason": "missing_tools",
            "langfuse_spans": [
                callbacks.build_span(
                    name="xai.tool_choice_without_tools_removed",
                    metadata=span_metadata,
                )
            ],
        },
    )


def drop_tool_choice_without_tools_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], Optional[Any]]:
    if "tool_choice" not in request_body:
        return request_body, None

    if request_has_openai_tool_definitions(request_body):
        return request_body, None

    updated_body = dict(request_body)
    removed_tool_choice = updated_body.pop("tool_choice", None)
    updated_body = add_tool_choice_without_tools_logging_metadata(
        updated_body,
        removed_tool_choice=removed_tool_choice,
        callbacks=callbacks,
    )
    return updated_body, removed_tool_choice


def add_codex_unsupported_request_param_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_params: list[str],
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    normalized_params = callbacks.dedupe_sorted(
        [
            normalized
            for param in removed_params
            if (normalized := callbacks.normalize_tag_value(param))
        ]
    )
    span_metadata: dict[str, Any] = {
        "removed_count": len(removed_params),
        "removed_params": normalized_params,
    }
    tags_to_add = ["codex-unsupported-request-param-removed"]
    tags_to_add.extend(
        f"codex-unsupported-request-param:{param}" for param in normalized_params
    )
    return callbacks.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_unsupported_request_param_removed_count": len(removed_params),
            "codex_unsupported_request_params_removed": normalized_params,
            "langfuse_spans": [
                callbacks.build_span(
                    name="codex.unsupported_request_param_removed",
                    metadata=span_metadata,
                )
            ],
        },
    )


def drop_unsupported_codex_request_params_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[str]]:
    unsupported_params = get_unsupported_request_param_names_for_model(
        request_body.get("model"), callbacks=callbacks
    )
    if not unsupported_params:
        return request_body, []

    max_depth = callbacks.request_body_walk_max_depth
    normalize = callbacks.normalize_tag_value

    def _drop_from_value(
        value: Any,
        *,
        path: tuple[str, ...] = (),
        _depth: int = 0,
    ) -> tuple[Any, list[str], bool]:
        if _depth > max_depth:
            return value, [], False
        if isinstance(value, dict):
            updated_dict: dict[str, Any] = {}
            removed: list[str] = []
            changed = False
            for key, child_value in value.items():
                normalized_key = normalize(key)
                normalized_path = (
                    ".".join([*path, normalized_key])
                    if normalized_key is not None
                    else None
                )
                if normalized_key in unsupported_params or (
                    normalized_path in unsupported_params
                ):
                    removed.append(
                        normalized_path
                        if normalized_key not in unsupported_params
                        and normalized_path in unsupported_params
                        else key
                    )
                    changed = True
                    continue
                updated_child, child_removed, child_changed = _drop_from_value(
                    child_value,
                    path=(
                        (*path, normalized_key)
                        if normalized_key is not None
                        else path
                    ),
                    _depth=_depth + 1,
                )
                updated_dict[key] = updated_child
                removed.extend(child_removed)
                changed = changed or child_changed
            return (updated_dict if changed else value), removed, changed

        if isinstance(value, list):
            updated_list: list[Any] = []
            list_removed: list[str] = []
            changed = False
            for item in value:
                updated_item, item_removed, item_changed = _drop_from_value(
                    item,
                    path=path,
                    _depth=_depth + 1,
                )
                updated_list.append(updated_item)
                list_removed.extend(item_removed)
                changed = changed or item_changed
            return (updated_list if changed else value), list_removed, changed

        return value, [], False

    updated_value, removed_params, changed = _drop_from_value(request_body)
    if not removed_params:
        return request_body, []

    updated_body = (
        updated_value
        if changed and isinstance(updated_value, dict)
        else dict(request_body)
    )

    updated_body = add_codex_unsupported_request_param_logging_metadata(
        updated_body,
        removed_params=removed_params,
        callbacks=callbacks,
    )
    return updated_body, removed_params


def add_codex_unsupported_input_item_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_items: list[dict[str, Any]],
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    removed_item_types = callbacks.dedupe_sorted(
        [
            item["type"]
            for item in removed_items
            if isinstance(item.get("type"), str) and item["type"]
        ]
    )
    span_metadata: dict[str, Any] = {
        "removed_count": len(removed_items),
        "removed_item_types": removed_item_types,
    }

    tags_to_add = ["codex-unsupported-input-item-removed"]
    tags_to_add.extend(
        f"codex-unsupported-input-item:{item_type}"
        for item_type in removed_item_types
    )

    return callbacks.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_unsupported_input_item_removed_count": len(removed_items),
            "codex_unsupported_input_item_types_removed": removed_item_types,
            "codex_unsupported_input_items_removed": removed_items,
            "langfuse_spans": [
                callbacks.build_span(
                    name="codex.unsupported_input_item_removed",
                    metadata=span_metadata,
                )
            ],
        },
    )


def drop_unsupported_codex_input_items_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        callbacks.normalize_kimi_model_name(request_body.get("model"))
        is not None
    ):
        request_body = callbacks.normalize_kimi_custom_tool_outputs(request_body)

    unsupported_input_item_types = get_unsupported_input_item_types_for_model(
        request_body.get("model"), callbacks=callbacks
    )
    if not unsupported_input_item_types:
        return request_body, []

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, []

    updated_input_items: list[Any] = []
    removed_items: list[dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue

        item_type = callbacks.normalize_tag_value(item.get("type"))
        if item_type in unsupported_input_item_types:
            removed_item: dict[str, Any] = {
                "type": item_type,
                "index": index,
            }
            if item_type == "reasoning" and isinstance(
                item.get("encrypted_content"), str
            ):
                removed_item["encrypted_content"] = True
            removed_items.append(removed_item)
            continue

        updated_input_items.append(item)

    if not removed_items:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    updated_body = add_codex_unsupported_input_item_logging_metadata(
        updated_body,
        removed_items=removed_items,
        callbacks=callbacks,
    )
    return updated_body, removed_items


def drop_unsupported_codex_hosted_tools_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unsupported_tool_types = get_unsupported_hosted_tool_types_for_model(
        request_body.get("model"), callbacks=callbacks
    )
    if not unsupported_tool_types:
        return request_body, []

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body, []

    updated_tools: list[Any] = []
    removed_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue

        tool_type = callbacks.normalize_tag_value(get_openai_tool_type(tool))
        if tool_type in unsupported_tool_types:
            removed_tool: dict[str, Any] = {
                "type": tool_type,
                "index": index,
            }
            tool_name = get_openai_tool_name(tool)
            if tool_name:
                removed_tool["name"] = tool_name
            removed_tools.append(removed_tool)
            continue

        updated_tools.append(tool)

    if not removed_tools:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["tools"] = updated_tools

    removed_tool_types = {
        tool["type"]
        for tool in removed_tools
        if isinstance(tool.get("type"), str) and tool["type"]
    }
    removed_tool_choice = None
    if openai_tool_choice_references_tool_type(
        updated_body.get("tool_choice"),
        removed_tool_types,
        normalize_tag_value=callbacks.normalize_tag_value,
    ):
        removed_tool_choice = updated_body.pop("tool_choice", None)

    updated_body = add_codex_unsupported_hosted_tool_logging_metadata(
        updated_body,
        removed_tools=removed_tools,
        removed_tool_choice=removed_tool_choice,
        callbacks=callbacks,
    )
    return updated_body, removed_tools


# ---------------------------------------------------------------------------
# Tool description patch orchestrator
# ---------------------------------------------------------------------------


def add_codex_tool_description_patch_logging_metadata(
    request_body: dict[str, Any],
    patch_events: list[dict[str, Any]],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    patch_ids = callbacks.dedupe_sorted(
        [
            event["id"]
            for event in patch_events
            if isinstance(event.get("id"), str) and event["id"]
        ]
    )
    replacement_count = sum(
        event["occurrences"]
        for event in patch_events
        if isinstance(event.get("occurrences"), int)
    )
    span_metadata: dict[str, Any] = {
        "patch_count": len(patch_events),
        "replacement_count": replacement_count,
    }
    if patch_ids:
        span_metadata["patch_ids"] = patch_ids

    tags_to_add = ["codex-tool-description-patch"]
    tags_to_add.extend(
        f"codex-tool-description-patch:{patch_id}" for patch_id in patch_ids
    )

    return callbacks.merge_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_tool_description_patch_count": len(patch_events),
            "codex_tool_description_patch_replacement_count": replacement_count,
            "codex_tool_description_patch_ids": patch_ids,
            "codex_tool_description_patch_events": patch_events,
            "langfuse_spans": [
                callbacks.build_span(
                    name="codex.tool_description_patch",
                    metadata=span_metadata,
                )
            ],
        },
    )


def apply_codex_tool_description_patches_to_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body, []

    updated_tools: list[Any] = []
    patch_events: list[dict[str, Any]] = []
    changed = False
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue
        updated_tool, tool_patch_events = patch_codex_spawn_agent_tool_description(
            tool,
            tool_index=index,
            normalize_tag_value=callbacks.normalize_tag_value,
        )
        (
            updated_tool,
            tool_search_patch_events,
        ) = patch_codex_multi_agent_tool_search_description(
            updated_tool,
            tool_index=index,
            normalize_tag_value=callbacks.normalize_tag_value,
        )
        updated_tool, core_tool_patch_events = patch_codex_core_tool_description(
            updated_tool,
            tool_index=index,
            normalize_tag_value=callbacks.normalize_tag_value,
        )
        updated_tools.append(updated_tool)
        patch_events.extend(tool_patch_events)
        patch_events.extend(tool_search_patch_events)
        patch_events.extend(core_tool_patch_events)
        if updated_tool is not tool:
            changed = True

    if not changed or not patch_events:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["tools"] = updated_tools
    updated_body = add_codex_tool_description_patch_logging_metadata(
        updated_body,
        patch_events,
        callbacks=callbacks,
    )
    return updated_body, patch_events


# ---------------------------------------------------------------------------
# Grok-native input-item policy facades
# ---------------------------------------------------------------------------


def stringify_grok_native_input_item_value(
    value: Any,
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> str:
    return callbacks.grok_normalization.stringify_input_item_value(value)


def format_grok_native_function_call_input_message(
    item: dict[str, Any],
    *,
    include_correlation_ref: bool = True,
    callbacks: CodexToolPolicyCallbacks,
) -> str:
    return callbacks.grok_normalization.format_function_call_input_message(
        item,
        include_correlation_ref=include_correlation_ref,
    )


def format_grok_native_function_call_output_input_message(
    item: dict[str, Any],
    *,
    include_correlation_ref: bool = True,
    callbacks: CodexToolPolicyCallbacks,
) -> str:
    return callbacks.grok_normalization.format_function_call_output_input_message(
        item,
        include_correlation_ref=include_correlation_ref,
    )


def rewrite_grok_native_input_item_for_model_input(
    item: dict[str, Any],
    *,
    item_type: str,
    include_correlation_ref: bool = True,
    callbacks: CodexToolPolicyCallbacks,
) -> Optional[dict[str, Any]]:
    return callbacks.grok_normalization.rewrite_input_item_for_model_input(
        item,
        item_type=item_type,
        include_correlation_ref=include_correlation_ref,
    )


def is_anthropic_grok_native_responses_adapter_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> bool:
    return callbacks.grok_normalization.is_anthropic_responses_adapter_body(
        request_body
    )


def add_grok_native_input_item_rewrite_logging_metadata(
    request_body: dict[str, Any],
    *,
    rewritten_items: list[dict[str, Any]],
    callbacks: CodexToolPolicyCallbacks,
) -> dict[str, Any]:
    return callbacks.grok_normalization.add_input_item_rewrite_logging_metadata(
        callbacks.grok_normalization_runtime,
        request_body,
        rewritten_items=rewritten_items,
    )


def rewrite_grok_native_unsupported_input_items_from_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return (
        callbacks.grok_normalization.rewrite_unsupported_input_items_from_request_body(
            callbacks.grok_normalization_runtime, request_body
        )
    )


def rewrite_grok_native_unsupported_input_items_in_place(
    request_body: dict[str, Any],
    *,
    callbacks: CodexToolPolicyCallbacks,
) -> list[dict[str, Any]]:
    return callbacks.grok_normalization.rewrite_unsupported_input_items_in_place(
        callbacks.grok_normalization_runtime,
        request_body,
    )


# ---------------------------------------------------------------------------
# Host facade publication
# ---------------------------------------------------------------------------

_codex_tool_policy_runtime_accessors: Optional[
    CodexToolPolicyRuntimeAccessors
] = None

_HOST_FUNCTION_NAMES = (
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


def configure_codex_tool_policy_runtime_accessors(
    accessors: CodexToolPolicyRuntimeAccessors,
) -> None:
    """Configure late-bound runtime access for published policy facades."""
    global _codex_tool_policy_runtime_accessors
    _codex_tool_policy_runtime_accessors = accessors


def get_codex_tool_policy_runtime_callbacks() -> CodexToolPolicyCallbacks:
    """Resolve the current host callback bundle at call time."""
    if _codex_tool_policy_runtime_accessors is None:
        raise RuntimeError("Codex tool-policy runtime accessors are not configured")
    return _codex_tool_policy_runtime_accessors.get_callbacks()


def get_codex_tool_policy_runtime_normalize_tag_value() -> NormalizeTagValueFn:
    """Resolve the current host normalization callback at call time."""
    if _codex_tool_policy_runtime_accessors is None:
        raise RuntimeError("Codex tool-policy runtime accessors are not configured")
    return _codex_tool_policy_runtime_accessors.get_normalize_tag_value()


def is_codex_tool_policy_runtime_configured() -> bool:
    """Return True when runtime accessors have been installed.

    Consumers outside the god-module integration boundary (e.g. the
    Responses endpoint) should check this before calling underscore facades.
    """
    return _codex_tool_policy_runtime_accessors is not None


def _get_required_codex_tool_policy_host_global(
    host_globals: dict[str, Any], name: str
) -> Any:
    try:
        return host_globals[name]
    except KeyError:
        raise RuntimeError(
            "Codex tool-policy runtime accessors are not configured"
        ) from None


def install_codex_tool_policy_facades(host_globals: dict[str, Any]) -> None:
    """Publish same-object facades backed by live host-global lookups."""
    configure_codex_tool_policy_runtime_accessors(
        CodexToolPolicyRuntimeAccessors(
            get_callbacks=lambda: _get_required_codex_tool_policy_host_global(
                host_globals, "_CODEX_TOOL_POLICY_CALLBACKS"
            ),
            get_normalize_tag_value=lambda: _get_required_codex_tool_policy_host_global(
                host_globals, "_normalize_low_cardinality_tag_value"
            ),
        )
    )
    module_globals = globals()
    for name in _HOST_FUNCTION_NAMES:
        host_globals[name] = module_globals[name]


def configure_and_install_codex_tool_policy(
    host_globals: dict[str, Any],
    deps: CodexToolPolicyHostDeps,
) -> None:
    """One-call replacement for the god-module's callbacks + 42 thin wrappers.

    Builds :class:`CodexToolPolicyCallbacks` from *deps*, publishes the
    callbacks and normalize function into *host_globals*, then delegates to
    :func:`install_codex_tool_policy_facades` for late-bound facade
    publication.  The resulting host-global surface is identical to the
    hand-written wrappers: same names, same signatures, same late-bound
    monkeypatch behavior.

    Usage in the god module::

        _aawm_codex_tool_policy.configure_and_install_codex_tool_policy(
            globals(),
            _aawm_codex_tool_policy.CodexToolPolicyHostDeps(
                normalize_tag_value=_normalize_low_cardinality_tag_value,
                dedupe_sorted=_dedupe_sorted_str_list,
                merge_metadata=_merge_litellm_metadata,
                build_span=_build_langfuse_span_descriptor,
                get_model_cost_map=lambda: litellm.model_cost,
                normalize_grok_native_oauth_model=normalize_grok_native_oauth_model,
                is_oa_xai_model=is_oa_xai_model,
                resolve_oa_xai_upstream_model=resolve_oa_xai_upstream_model,
                normalize_kimi_model_name=_normalize_kimi_code_chat_completions_adapter_model_name,
                normalize_kimi_custom_tool_outputs=lambda b: _kimi_code_adapters.normalize_kimi_code_custom_tool_outputs(b),
                grok_normalization=_anthropic_grok_normalization,
                grok_normalization_runtime=_get_anthropic_grok_normalization_runtime(),
                request_body_walk_max_depth=_AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
            ),
        )
    """
    callbacks = CodexToolPolicyCallbacks(
        normalize_tag_value=deps.normalize_tag_value,
        dedupe_sorted=deps.dedupe_sorted,
        merge_metadata=deps.merge_metadata,
        build_span=deps.build_span,
        get_model_cost_map=deps.get_model_cost_map,
        normalize_grok_native_oauth_model=deps.normalize_grok_native_oauth_model,
        is_oa_xai_model=deps.is_oa_xai_model,
        resolve_oa_xai_upstream_model=deps.resolve_oa_xai_upstream_model,
        normalize_kimi_model_name=deps.normalize_kimi_model_name,
        normalize_kimi_custom_tool_outputs=deps.normalize_kimi_custom_tool_outputs,
        grok_normalization=deps.grok_normalization,
        grok_normalization_runtime=deps.grok_normalization_runtime,
        request_body_walk_max_depth=deps.request_body_walk_max_depth,
    )
    host_globals["_CODEX_TOOL_POLICY_CALLBACKS"] = callbacks
    host_globals["_normalize_low_cardinality_tag_value"] = deps.normalize_tag_value
    install_codex_tool_policy_facades(host_globals)


def _patch_codex_spawn_agent_tool_description(
    tool: dict[str, Any], *, tool_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return patch_codex_spawn_agent_tool_description(
        tool,
        tool_index=tool_index,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _get_codex_core_tool_guidance(tool_name: Optional[str]) -> Optional[str]:
    return get_codex_core_tool_guidance(
        tool_name,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _append_codex_core_tool_guidance_to_description(
    description: Any, *, guidance: str
) -> tuple[str, bool]:
    return append_codex_core_tool_guidance_to_description(
        description,
        guidance=guidance,
    )


def _patch_codex_multi_agent_tool_search_description(
    tool: dict[str, Any], *, tool_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return patch_codex_multi_agent_tool_search_description(
        tool,
        tool_index=tool_index,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _patch_codex_core_tool_description(
    tool: dict[str, Any], *, tool_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return patch_codex_core_tool_description(
        tool,
        tool_index=tool_index,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapt_codex_custom_tool_definitions(
    tools: list[Any], *, adapter_names: set[str]
) -> tuple[list[Any], list[dict[str, Any]]]:
    return adapt_codex_custom_tool_definitions(
        tools,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapted_custom_tool_call_ids(
    input_items: Any, *, adapter_names: set[str]
) -> set[str]:
    return adapted_custom_tool_call_ids(
        input_items,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapt_codex_custom_tool_input_items(
    input_items: Any, *, adapter_names: set[str]
) -> tuple[Any, list[dict[str, Any]]]:
    return adapt_codex_custom_tool_input_items(
        input_items,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapt_codex_custom_tool_choice(
    tool_choice: Any, *, adapter_names: set[str]
) -> tuple[Any, Optional[dict[str, Any]]]:
    return adapt_codex_custom_tool_choice(
        tool_choice,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapt_codex_namespace_tool_definitions(
    tools: list[Any], *, adapter_names: dict[str, set[str]]
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return adapt_codex_namespace_tool_definitions(
        tools,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapt_codex_namespace_input_items(
    input_items: Any, *, adapter_names: dict[str, set[str]]
) -> tuple[Any, list[dict[str, Any]]]:
    return adapt_codex_namespace_input_items(
        input_items,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _adapt_codex_namespace_tool_choice(
    tool_choice: Any, *, adapter_names: dict[str, set[str]]
) -> tuple[Any, Optional[dict[str, Any]]]:
    return adapt_codex_namespace_tool_choice(
        tool_choice,
        adapter_names=adapter_names,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _openai_tool_choice_references_tool_type(
    tool_choice: Any, tool_types: set[str]
) -> bool:
    return openai_tool_choice_references_tool_type(
        tool_choice,
        tool_types,
        normalize_tag_value=get_codex_tool_policy_runtime_normalize_tag_value(),
    )


def _get_codex_tool_policy_model_cost_candidates(model: Any) -> list[str]:
    return get_codex_tool_policy_model_cost_candidates(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _get_unsupported_hosted_tool_types_for_model(model: Any) -> set[str]:
    return get_unsupported_hosted_tool_types_for_model(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _get_unsupported_request_param_names_for_model(model: Any) -> set[str]:
    return get_unsupported_request_param_names_for_model(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _get_unsupported_input_item_types_for_model(model: Any) -> set[str]:
    return get_unsupported_input_item_types_for_model(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _get_rewrite_input_item_types_for_model(model: Any) -> set[str]:
    return get_rewrite_input_item_types_for_model(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _get_custom_tool_function_adapter_names_for_model(model: Any) -> set[str]:
    return get_custom_tool_function_adapter_names_for_model(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _get_namespace_tool_function_adapter_names_for_model(
    model: Any,
) -> dict[str, set[str]]:
    return get_namespace_tool_function_adapter_names_for_model(
        model,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_codex_custom_tool_function_adapter_logging_metadata(
    request_body: dict[str, Any],
    *,
    adapted_tools: list[dict[str, Any]],
    adapted_input_items: list[dict[str, Any]],
    adapted_tool_choice: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return add_codex_custom_tool_function_adapter_logging_metadata(
        request_body,
        adapted_tools=adapted_tools,
        adapted_input_items=adapted_input_items,
        adapted_tool_choice=adapted_tool_choice,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _adapt_codex_custom_tools_to_functions_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return adapt_codex_custom_tools_to_functions_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_codex_namespace_tool_function_adapter_logging_metadata(
    request_body: dict[str, Any],
    *,
    adapted_tools: list[dict[str, Any]],
    adapted_input_items: list[dict[str, Any]],
    adapted_tool_choice: Optional[dict[str, Any]],
    skipped_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return add_codex_namespace_tool_function_adapter_logging_metadata(
        request_body,
        adapted_tools=adapted_tools,
        adapted_input_items=adapted_input_items,
        adapted_tool_choice=adapted_tool_choice,
        skipped_tools=skipped_tools,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _adapt_codex_namespace_tools_to_functions_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return adapt_codex_namespace_tools_to_functions_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_codex_unsupported_hosted_tool_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_tools: list[dict[str, Any]],
    removed_tool_choice: Optional[Any],
) -> dict[str, Any]:
    return add_codex_unsupported_hosted_tool_logging_metadata(
        request_body,
        removed_tools=removed_tools,
        removed_tool_choice=removed_tool_choice,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_tool_choice_without_tools_logging_metadata(
    request_body: dict[str, Any], *, removed_tool_choice: Any
) -> dict[str, Any]:
    return add_tool_choice_without_tools_logging_metadata(
        request_body,
        removed_tool_choice=removed_tool_choice,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _drop_tool_choice_without_tools_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], Optional[Any]]:
    return drop_tool_choice_without_tools_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_codex_unsupported_request_param_logging_metadata(
    request_body: dict[str, Any], *, removed_params: list[str]
) -> dict[str, Any]:
    return add_codex_unsupported_request_param_logging_metadata(
        request_body,
        removed_params=removed_params,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _drop_unsupported_codex_request_params_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    return drop_unsupported_codex_request_params_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_codex_unsupported_input_item_logging_metadata(
    request_body: dict[str, Any], *, removed_items: list[dict[str, Any]]
) -> dict[str, Any]:
    return add_codex_unsupported_input_item_logging_metadata(
        request_body,
        removed_items=removed_items,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _drop_unsupported_codex_input_items_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return drop_unsupported_codex_input_items_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _drop_unsupported_codex_hosted_tools_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return drop_unsupported_codex_hosted_tools_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_codex_tool_description_patch_logging_metadata(
    request_body: dict[str, Any], patch_events: list[dict[str, Any]]
) -> dict[str, Any]:
    return add_codex_tool_description_patch_logging_metadata(
        request_body,
        patch_events,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _apply_codex_tool_description_patches_to_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return apply_codex_tool_description_patches_to_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _stringify_grok_native_input_item_value(value: Any) -> str:
    return stringify_grok_native_input_item_value(
        value,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _format_grok_native_function_call_input_message(
    item: dict[str, Any], *, include_correlation_ref: bool = False
) -> str:
    return format_grok_native_function_call_input_message(
        item,
        include_correlation_ref=include_correlation_ref,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _format_grok_native_function_call_output_input_message(
    item: dict[str, Any], *, include_correlation_ref: bool = False
) -> str:
    return format_grok_native_function_call_output_input_message(
        item,
        include_correlation_ref=include_correlation_ref,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _rewrite_grok_native_input_item_for_model_input(
    item: dict[str, Any],
    *,
    item_type: str,
    include_correlation_ref: bool = False,
) -> Optional[dict[str, Any]]:
    return rewrite_grok_native_input_item_for_model_input(
        item,
        item_type=item_type,
        include_correlation_ref=include_correlation_ref,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _is_anthropic_grok_native_responses_adapter_body(
    request_body: dict[str, Any],
) -> bool:
    return is_anthropic_grok_native_responses_adapter_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _add_grok_native_input_item_rewrite_logging_metadata(
    request_body: dict[str, Any], *, rewritten_items: list[dict[str, Any]]
) -> dict[str, Any]:
    return add_grok_native_input_item_rewrite_logging_metadata(
        request_body,
        rewritten_items=rewritten_items,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _rewrite_grok_native_unsupported_input_items_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return rewrite_grok_native_unsupported_input_items_from_request_body(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )


def _rewrite_grok_native_unsupported_input_items_in_place(
    request_body: dict[str, Any],
) -> list[dict[str, Any]]:
    return rewrite_grok_native_unsupported_input_items_in_place(
        request_body,
        callbacks=get_codex_tool_policy_runtime_callbacks(),
    )
