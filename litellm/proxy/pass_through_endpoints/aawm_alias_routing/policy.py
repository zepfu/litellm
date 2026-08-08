"""Generic AAWM alias-routing policy constants.

Configured alias identity, candidate order, and fallback behavior belong only
to the compiled routing snapshot. This module retains provider, lane, cooldown,
quota, allowlist, and adapter-capability policy shared by runtime components.
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

# Provider and lane identity strings.
CODEX_AUTO_AGENT_NATIVE_PROVIDER = "openai"
CODEX_AUTO_AGENT_OPENROUTER_PROVIDER = "openrouter"
CODEX_AUTO_AGENT_XAI_PROVIDER = "xai"
CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER = "kimi_code"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER = "alibaba_token_plan"
OPENCODE_ZEN_PROVIDER = "opencode_zen"
CODEX_AUTO_AGENT_OPENCODE_PROVIDER = OPENCODE_ZEN_PROVIDER
ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = "anthropic"

CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY = "openrouter"
CODEX_AUTO_AGENT_XAI_LANE_KEY = "xai_grok_native"
CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY = "xai_oauth_managed"
CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY = "kimi_code_managed_account"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY = "alibaba_token_plan"
CODEX_AUTO_AGENT_OPENCODE_LANE_KEY = OPENCODE_ZEN_PROVIDER

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
KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "kimi_code/k3",
        "kimi_code/k3-low",
        "kimi_code/k3-high",
        "kimi_code/k3-max",
        "kimi_code/kimi-for-coding",
        "kimi_code/kimi-for-coding-highspeed",
    }
)
ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "alibaba_token_plan/qwen3.8-max",
        "alibaba_token_plan/qwen3.7-plus",
        "alibaba_token_plan/qwen3.7-max",
        "alibaba_token_plan/qwen3.6-flash",
        "alibaba_token_plan/deepseek-v4-pro",
        "alibaba_token_plan/glm-5.2",
    }
)

# Generic compatibility publication for the pass-through integration module.
COMPAT_ALIAS_MAP: dict[str, str] = {
    "_CODEX_AUTO_AGENT_NATIVE_PROVIDER": "CODEX_AUTO_AGENT_NATIVE_PROVIDER",
    "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER",
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": (
        "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER"
    ),
    "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "CODEX_AUTO_AGENT_OPENROUTER_PROVIDER",
    "_CODEX_AUTO_AGENT_XAI_PROVIDER": "CODEX_AUTO_AGENT_XAI_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "CODEX_AUTO_AGENT_OPENCODE_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY": (
        "CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_XAI_LANE_KEY": "CODEX_AUTO_AGENT_XAI_LANE_KEY",
    "_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY": (
        "CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY": (
        "CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY": (
        "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY"
    ),
    "_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY": "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY",
    "_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS"
    ),
    "_CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS": (
        "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS"
    ),
    "_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER": "ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER",
    "_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS"
    ),
    "_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS"
    ),
    "_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS"
    ),
    "_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS": (
        "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS"
    ),
    "_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS": (
        "KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS"
    ),
    "_ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS": (
        "ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS"
    ),
}
COMPAT_ALIAS_COUNT = len(COMPAT_ALIAS_MAP)


def install_policy_compat_aliases(host_globals: dict[str, Any]) -> None:
    """Install retained generic policy aliases into ``host_globals``."""
    if not isinstance(host_globals, dict):
        raise TypeError(
            f"host_globals must be a dict, got {type(host_globals).__name__}"
        )
    policy_namespace = globals()
    for local_name, policy_name in COMPAT_ALIAS_MAP.items():
        host_globals[local_name] = policy_namespace[policy_name]


__all__ = [
    "ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER",
    "ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS",
    "ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY",
    "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER",
    "CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS",
    "CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY",
    "CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER",
    "CODEX_AUTO_AGENT_NATIVE_PROVIDER",
    "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY",
    "CODEX_AUTO_AGENT_OPENCODE_PROVIDER",
    "CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY",
    "CODEX_AUTO_AGENT_OPENROUTER_PROVIDER",
    "CODEX_AUTO_AGENT_XAI_LANE_KEY",
    "CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY",
    "CODEX_AUTO_AGENT_XAI_PROVIDER",
    "COMPAT_ALIAS_COUNT",
    "COMPAT_ALIAS_MAP",
    "KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS",
    "OPENCODE_ZEN_PROVIDER",
    "OPENROUTER_FREE_DAILY_QUOTA_MODELS",
    "install_policy_compat_aliases",
]
