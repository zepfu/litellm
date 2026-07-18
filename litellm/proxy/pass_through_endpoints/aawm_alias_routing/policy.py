"""AAWM alias-routing policy constants (RR-054 #1/#11).

Owned by ``aawm_alias_routing`` package. Runtime engines remain in
``llm_passthrough_endpoints.py``; this module owns static cooldown durations,
free-quota model set, auto-agent candidate tables, alias→candidate maps, and
adapter allowed-model surfaces.
"""

from __future__ import annotations

from typing import Any

# Default cooldowns for auto-agent alias candidates.
CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS = 3 * 60 * 60.0
CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS = (
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = (
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS = (
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS = 30.0

OPENROUTER_FREE_DAILY_QUOTA_MODELS = frozenset(
    {
        "openrouter/cohere/north-mini-code:free",
        "openrouter/owl-alpha",
    }
)

# Provider / lane identity strings used by candidate tables.
CODEX_AUTO_AGENT_NATIVE_PROVIDER = "openai"
CODEX_AUTO_AGENT_GOOGLE_PROVIDER = "google_code_assist"
CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER = "antigravity"
CODEX_AUTO_AGENT_OPENROUTER_PROVIDER = "openrouter"
CODEX_AUTO_AGENT_XAI_PROVIDER = "xai"
OPENCODE_ZEN_PROVIDER = "opencode_zen"
CODEX_AUTO_AGENT_OPENCODE_PROVIDER = OPENCODE_ZEN_PROVIDER
CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY = "openrouter"
CODEX_AUTO_AGENT_XAI_LANE_KEY = "xai_grok_native"
CODEX_AUTO_AGENT_OPENCODE_LANE_KEY = OPENCODE_ZEN_PROVIDER
ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = "anthropic"
ANTHROPIC_AUTO_AGENT_HAIKU_MODEL = "claude-haiku-4-5-20251001"

CODEX_AUTO_AGENT_MODEL_ALIAS = "aawm-codex-agent-auto"
CODEX_AAWM_READ_ALIAS = "aawm-read"
CODEX_AAWM_SOTA_ALIAS = "aawm-sota"
CODEX_AAWM_CODE_ALIAS = "aawm-code"
CODEX_AAWM_LOW_ALIAS = "aawm-low"
CODEX_AAWM_ORCHESTRATION_ALIAS = "aawm-orchestration"
CODEX_AAWM_SOTA_OPENAI_ALIAS = "aawm-sota-openai"
CODEX_AAWM_SOTA_XAI_ALIAS = "aawm-sota-xai"
ANTHROPIC_AUTO_AGENT_MODEL_ALIAS = "aawm-anthropic-agent-auto"
ANTHROPIC_AAWM_READ_ALIAS = "aawm-read-anthropic"
ANTHROPIC_AAWM_SOTA_ALIAS = "aawm-sota-anthropic"
ANTHROPIC_AAWM_CODE_ALIAS = "aawm-code-anthropic"
ANTHROPIC_AAWM_LOW_ALIAS = "aawm-low-anthropic"
ANTHROPIC_AAWM_ORCHESTRATION_ALIAS = "aawm-orchestration-anthropic"

CODEX_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "deepseek/deepseek-v4-flash",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.4-mini",
        "route_family": "codex_responses",
        "last_resort": True,
    },
)
CODEX_AAWM_READ_ALIAS = "aawm-read"
CODEX_AAWM_SOTA_ALIAS = "aawm-sota"
CODEX_AAWM_CODE_ALIAS = "aawm-code"
CODEX_AAWM_LOW_ALIAS = "aawm-low"
CODEX_AAWM_ORCHESTRATION_ALIAS = "aawm-orchestration"
CODEX_AAWM_SOTA_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.6-sol",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.5",
        "route_family": "codex_responses",
        "last_resort": True,
    },
)
CODEX_AAWM_SOTA_OPENAI_ALIAS = "aawm-sota-openai"
CODEX_AAWM_SOTA_XAI_ALIAS = "aawm-sota-xai"
CODEX_AAWM_SOTA_OPENAI_CANDIDATES: tuple[
    dict[str, Any], ...
] = CODEX_AAWM_SOTA_CANDIDATES
CODEX_AAWM_SOTA_XAI_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "oa_xai/grok-4.5",
        "route_family": "codex_xai_oauth_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-4.5",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-build",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": True,
    },
)
CODEX_AAWM_CODE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "xai/grok-4.5",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-composer-2.5-fast",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "oa_xai/grok-build",
        "route_family": "codex_xai_oauth_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.6-terra",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.5",
        "route_family": "codex_responses",
        "last_resort": True,
        "default_reasoning_effort": "medium",
    },
)
CODEX_AAWM_LOW_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/cohere/north-mini-code:free",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/owl-alpha",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "deepseek-v4-flash",
        "route_family": "codex_opencode_zen_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "big-pickle",
        "route_family": "codex_opencode_zen_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.6-luna",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.4-mini",
        "route_family": "codex_responses",
        "last_resort": True,
    },
)
CODEX_AAWM_ORCHESTRATION_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.6-terra",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.5",
        "route_family": "codex_responses",
        "last_resort": True,
    },
)
CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[str, tuple[dict[str, Any], ...]] = {
    CODEX_AUTO_AGENT_MODEL_ALIAS: CODEX_AUTO_AGENT_CANDIDATES,
    CODEX_AAWM_READ_ALIAS: CODEX_AUTO_AGENT_CANDIDATES,
    CODEX_AAWM_SOTA_ALIAS: CODEX_AAWM_SOTA_CANDIDATES,
    CODEX_AAWM_SOTA_OPENAI_ALIAS: CODEX_AAWM_SOTA_OPENAI_CANDIDATES,
    CODEX_AAWM_SOTA_XAI_ALIAS: CODEX_AAWM_SOTA_XAI_CANDIDATES,
    CODEX_AAWM_CODE_ALIAS: CODEX_AAWM_CODE_CANDIDATES,
    CODEX_AAWM_LOW_ALIAS: CODEX_AAWM_LOW_CANDIDATES,
    CODEX_AAWM_ORCHESTRATION_ALIAS: CODEX_AAWM_ORCHESTRATION_CANDIDATES,
}
ANTHROPIC_AUTO_AGENT_MODEL_ALIAS = "aawm-anthropic-agent-auto"
ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = "anthropic"
ANTHROPIC_AUTO_AGENT_HAIKU_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "anthropic_openai_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "deepseek/deepseek-v4-flash",
        "route_family": "anthropic_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": ANTHROPIC_AUTO_AGENT_HAIKU_MODEL,
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
ANTHROPIC_AAWM_READ_ALIAS = "aawm-read-anthropic"
ANTHROPIC_AAWM_SOTA_ALIAS = "aawm-sota-anthropic"
ANTHROPIC_AAWM_CODE_ALIAS = "aawm-code-anthropic"
ANTHROPIC_AAWM_LOW_ALIAS = "aawm-low-anthropic"
ANTHROPIC_AAWM_ORCHESTRATION_ALIAS = "aawm-orchestration-anthropic"
ANTHROPIC_AAWM_SOTA_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-fable-5",
        "route_family": "anthropic_messages",
        "last_resort": False,
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-opus-4-8[1m]",
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
ANTHROPIC_AAWM_CODE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "anthropic_openai_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "xai/grok-4.5",
        "route_family": "anthropic_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-composer-2.5-fast",
        "route_family": "anthropic_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "oa_xai/grok-build",
        "route_family": "anthropic_xai_oauth_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-sonnet-5[1m]",
        "route_family": "anthropic_messages",
        "last_resort": False,
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-sonnet-5",
        "route_family": "anthropic_messages",
        "last_resort": False,
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-sonnet-4-6",
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-opus-4-8[1m]",
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
ANTHROPIC_AAWM_LOW_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/cohere/north-mini-code:free",
        "route_family": "anthropic_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/owl-alpha",
        "route_family": "anthropic_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "deepseek-v4-flash",
        "route_family": "anthropic_opencode_zen_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "big-pickle",
        "route_family": "anthropic_opencode_zen_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": ANTHROPIC_AUTO_AGENT_HAIKU_MODEL,
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[str, tuple[dict[str, Any], ...]] = {
    ANTHROPIC_AUTO_AGENT_MODEL_ALIAS: ANTHROPIC_AUTO_AGENT_CANDIDATES,
    ANTHROPIC_AAWM_READ_ALIAS: ANTHROPIC_AUTO_AGENT_CANDIDATES,
    ANTHROPIC_AAWM_SOTA_ALIAS: ANTHROPIC_AAWM_SOTA_CANDIDATES,
    ANTHROPIC_AAWM_CODE_ALIAS: ANTHROPIC_AAWM_CODE_CANDIDATES,
    ANTHROPIC_AAWM_LOW_ALIAS: ANTHROPIC_AAWM_LOW_CANDIDATES,
    ANTHROPIC_AAWM_ORCHESTRATION_ALIAS: ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES,
}

ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.3-codex-spark",
    }
)
ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "deepseek-ai/deepseek-v3.1-terminus",
        "deepseek-ai/deepseek-v3.2",
        "minimaxai/minimax-m2.7",
        "mistralai/devstral-2-123b-instruct-2512",
        "z-ai/glm4.7",
    }
)
ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "openrouter/free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "minimax/minimax-m2.5:free",
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
        "gpt-oss-20b:free",
        "gpt-oss-120b:free",
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen3.6-flash",
        "qwen/qwen3-coder:free",
    }
)
ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "cohere/north-mini-code:free",
        "deepseek/deepseek-v4-flash:free",
        "openrouter/elephant-alpha",
        "inclusionai/ling-2.6-flash",
        "owl-alpha",
    }
)
ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    "gemini-3.1",
    "gemini-3-flash-preview",
)
CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    "gemini-3.1",
    "gemini-3-flash-preview",
)
ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER = "antigravity"
ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "chat_20706",
        "chat_23310",
        "claude-opus-4-6-thinking",
        "claude-sonnet-4-6",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-thinking",
        "gemini-2.5-pro",
        "gemini-3-flash",
        "gemini-3-flash-agent",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-high",
        "gemini-3.1-pro-low",
        "gemini-3.5-flash-extra-low",
        "gemini-3.5-flash-low",
        "gemini-pro-agent",
        "gpt-oss-120b-medium",
        "tab_flash_lite_preview",
        "tab_jump_flash_lite_preview",
    }
)
