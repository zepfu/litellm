"""Alias-specific system instruction shaping for AAWM passthrough requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.lane_keys import (
    _AAWM_READ_AGENT_GUIDANCE_POLICY_NAME,
    _AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION,
    _AAWM_READ_AGENT_GUIDANCE_PROMPT,
    _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME,
    _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION,
    _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _build_langfuse_span_descriptor as _default_build_langfuse_span_descriptor,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _merge_litellm_metadata as _default_merge_litellm_metadata,
)

# Canonical prompt/name/version constants are imported from
# ``aawm_alias_routing.lane_keys`` above, guaranteeing same-object identity
# with the god module (which also sources them from lane_keys).


class MergeLiteLLMMetadataFn(Protocol):
    def __call__(
        self,
        request_body: dict[str, Any],
        *,
        tags_to_add: list[str],
        extra_fields: dict[str, Any],
    ) -> dict[str, Any]: ...


class BuildLangfuseSpanDescriptorFn(Protocol):
    def __call__(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AliasGuidanceConfig:
    codex_auto_agent_prevention_guidance_policy_name: str = (
        _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME
    )
    codex_auto_agent_prevention_guidance_policy_version: str = (
        _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION
    )
    codex_auto_agent_prevention_guidance_prompt: str = (
        _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT
    )
    aawm_read_agent_guidance_policy_name: str = (
        _AAWM_READ_AGENT_GUIDANCE_POLICY_NAME
    )
    aawm_read_agent_guidance_policy_version: str = (
        _AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION
    )
    aawm_read_agent_guidance_prompt: str = _AAWM_READ_AGENT_GUIDANCE_PROMPT
    codex_aawm_read_alias: str = "basic"
    anthropic_aawm_read_alias: str = "basic"


@dataclass(frozen=True, slots=True)
class AliasGuidanceCallbacks:
    merge_litellm_metadata: MergeLiteLLMMetadataFn
    build_langfuse_span_descriptor: BuildLangfuseSpanDescriptorFn


DEFAULT_ALIAS_GUIDANCE_CONFIG = AliasGuidanceConfig()


# ── Runtime configuration seam ─────────────────────────────────────

_runtime_callbacks: Optional[AliasGuidanceCallbacks] = None


def configure_alias_guidance_runtime(
    *,
    callbacks: Optional[AliasGuidanceCallbacks] = None,
) -> None:
    """Bind module-level callbacks for live god-module call sites.

    Call once at startup to configure the callbacks used when live call sites
    invoke ``_apply_*`` without an explicit ``callbacks`` argument.  Passing
    ``None`` (or calling with no arguments) restores the default
    ``observability_metadata`` implementations.
    """
    global _runtime_callbacks
    _runtime_callbacks = callbacks


def _resolve_callbacks(
    explicit: Optional[AliasGuidanceCallbacks],
) -> AliasGuidanceCallbacks:
    if explicit is not None:
        return explicit
    if _runtime_callbacks is not None:
        return _runtime_callbacks
    return AliasGuidanceCallbacks(
        merge_litellm_metadata=_default_merge_litellm_metadata,
        build_langfuse_span_descriptor=_default_build_langfuse_span_descriptor,
    )


def _append_codex_auto_agent_prevention_guidance_to_instructions(
    instructions: str | None,
    *,
    config: AliasGuidanceConfig = DEFAULT_ALIAS_GUIDANCE_CONFIG,
) -> str:
    existing_instructions = instructions.strip() if isinstance(instructions, str) else ""
    if config.codex_auto_agent_prevention_guidance_prompt in existing_instructions:
        return existing_instructions
    if not existing_instructions:
        return config.codex_auto_agent_prevention_guidance_prompt
    return (
        f"{existing_instructions}\n\n"
        f"{config.codex_auto_agent_prevention_guidance_prompt}"
    )


def _is_aawm_read_agent_alias_model(
    alias_model: Any,
    *,
    config: AliasGuidanceConfig = DEFAULT_ALIAS_GUIDANCE_CONFIG,
) -> bool:
    if not isinstance(alias_model, str):
        return False
    return alias_model in {
        config.codex_aawm_read_alias,
        config.anthropic_aawm_read_alias,
    }


def _append_aawm_read_agent_guidance_to_text(
    value: str | None,
    *,
    config: AliasGuidanceConfig = DEFAULT_ALIAS_GUIDANCE_CONFIG,
) -> str:
    existing_value = value.strip() if isinstance(value, str) else ""
    if config.aawm_read_agent_guidance_prompt in existing_value:
        return existing_value
    if not existing_value:
        return config.aawm_read_agent_guidance_prompt
    return f"{existing_value}\n\n{config.aawm_read_agent_guidance_prompt}"


def _append_aawm_read_agent_guidance_to_anthropic_system(
    system_value: Any,
    *,
    config: AliasGuidanceConfig = DEFAULT_ALIAS_GUIDANCE_CONFIG,
) -> tuple[Any, bool, int]:
    if system_value is None or isinstance(system_value, str):
        original_chars = len(system_value) if isinstance(system_value, str) else 0
        updated_system = _append_aawm_read_agent_guidance_to_text(
            system_value,
            config=config,
        )
        return updated_system, updated_system != system_value, original_chars

    if not isinstance(system_value, list):
        return system_value, False, 0

    original_chars = 0
    for item in system_value:
        text_value: str | None = None
        if isinstance(item, str):
            text_value = item
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text_value = item["text"]
        if text_value is None:
            continue
        original_chars += len(text_value)
        if config.aawm_read_agent_guidance_prompt in text_value:
            return system_value, False, original_chars

    return (
        [
            *system_value,
            {
                "type": "text",
                "text": config.aawm_read_agent_guidance_prompt,
            },
        ],
        True,
        original_chars,
    )


def _apply_aawm_read_agent_guidance_to_request_body(
    request_body: dict[str, Any],
    *,
    alias_model: Any,
    target_field: str,
    callbacks: Optional[AliasGuidanceCallbacks] = None,
    config: AliasGuidanceConfig = DEFAULT_ALIAS_GUIDANCE_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_aawm_read_agent_alias_model(alias_model, config=config):
        return request_body, {}

    updated_body = dict(request_body)
    original_chars = 0
    if target_field == "instructions":
        existing_instructions = request_body.get("instructions")
        if existing_instructions is not None and not isinstance(
            existing_instructions, str
        ):
            return request_body, {}
        updated_value = _append_aawm_read_agent_guidance_to_text(
            existing_instructions,
            config=config,
        )
        if updated_value == existing_instructions:
            return request_body, {}
        updated_body["instructions"] = updated_value
        original_chars = (
            len(existing_instructions) if isinstance(existing_instructions, str) else 0
        )
    elif target_field == "system":
        (
            updated_system,
            changed,
            original_chars,
        ) = _append_aawm_read_agent_guidance_to_anthropic_system(
            request_body.get("system"),
            config=config,
        )
        if not changed:
            return request_body, {}
        updated_body["system"] = updated_system
    else:
        return request_body, {}

    guidance_metadata = {
        "aawm_read_agent_guidance_policy_name": (
            config.aawm_read_agent_guidance_policy_name
        ),
        "aawm_read_agent_guidance_policy_version": (
            config.aawm_read_agent_guidance_policy_version
        ),
        "aawm_read_agent_guidance_applied": True,
        "aawm_read_agent_guidance_alias": alias_model,
        "aawm_read_agent_guidance_target_field": target_field,
        "aawm_read_agent_guidance_original_chars": original_chars,
        "aawm_read_agent_guidance_prompt_chars": len(
            config.aawm_read_agent_guidance_prompt
        ),
    }
    effective_callbacks = _resolve_callbacks(callbacks)
    updated_body = effective_callbacks.merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "aawm-read-agent-guidance",
            (
                "aawm-read-agent-guidance:"
                f"{config.aawm_read_agent_guidance_policy_version}"
            ),
            f"aawm-read-agent-guidance-alias:{alias_model}",
        ],
        extra_fields={
            **guidance_metadata,
            "langfuse_spans": [
                effective_callbacks.build_langfuse_span_descriptor(
                    name="aawm.read_agent_guidance",
                    metadata=guidance_metadata,
                )
            ],
        },
    )
    return updated_body, guidance_metadata


def _apply_codex_auto_agent_prevention_guidance_to_request_body(
    request_body: dict[str, Any],
    *,
    callbacks: Optional[AliasGuidanceCallbacks] = None,
    config: AliasGuidanceConfig = DEFAULT_ALIAS_GUIDANCE_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_instructions = request_body.get("instructions")
    if existing_instructions is not None and not isinstance(existing_instructions, str):
        return request_body, {}

    updated_instructions = (
        _append_codex_auto_agent_prevention_guidance_to_instructions(
            existing_instructions,
            config=config,
        )
    )
    if updated_instructions == existing_instructions:
        return request_body, {}

    updated_body = dict(request_body)
    updated_body["instructions"] = updated_instructions
    original_chars = (
        len(existing_instructions) if isinstance(existing_instructions, str) else 0
    )
    guidance_metadata = {
        "codex_auto_agent_prevention_guidance_policy_name": (
            config.codex_auto_agent_prevention_guidance_policy_name
        ),
        "codex_auto_agent_prevention_guidance_policy_version": (
            config.codex_auto_agent_prevention_guidance_policy_version
        ),
        "codex_auto_agent_prevention_guidance_applied": True,
        "codex_auto_agent_prevention_guidance_original_instruction_chars": (
            original_chars
        ),
        "codex_auto_agent_prevention_guidance_prompt_chars": len(
            config.codex_auto_agent_prevention_guidance_prompt
        ),
    }
    effective_callbacks = _resolve_callbacks(callbacks)
    updated_body = effective_callbacks.merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "codex-auto-agent-prevention-guidance",
            (
                "codex-auto-agent-prevention-guidance:"
                f"{config.codex_auto_agent_prevention_guidance_policy_version}"
            ),
        ],
        extra_fields={
            **guidance_metadata,
            "langfuse_spans": [
                effective_callbacks.build_langfuse_span_descriptor(
                    name="codex.auto_agent_prevention_guidance",
                    metadata=guidance_metadata,
                )
            ],
        },
    )
    return updated_body, guidance_metadata
