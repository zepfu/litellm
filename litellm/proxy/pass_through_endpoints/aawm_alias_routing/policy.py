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
CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS = CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS = CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS = 30.0

OPENROUTER_FREE_DAILY_QUOTA_MODELS = frozenset(
    {
        "openrouter/cohere/north-mini-code:free",
        "openrouter/owl-alpha",
    }
)

# Provider / lane identity strings used by candidate tables.
CODEX_AUTO_AGENT_NATIVE_PROVIDER = "openai"
CODEX_AUTO_AGENT_OPENROUTER_PROVIDER = "openrouter"
CODEX_AUTO_AGENT_XAI_PROVIDER = "xai"
CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER = "kimi_code"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER = "alibaba_token_plan"
OPENCODE_ZEN_PROVIDER = "opencode_zen"
CODEX_AUTO_AGENT_OPENCODE_PROVIDER = OPENCODE_ZEN_PROVIDER
CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY = "openrouter"
CODEX_AUTO_AGENT_XAI_LANE_KEY = "xai_grok_native"
CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY = "xai_oauth_managed"
CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY = "kimi_code_managed_account"
CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY = "alibaba_token_plan"
CODEX_AUTO_AGENT_OPENCODE_LANE_KEY = OPENCODE_ZEN_PROVIDER
ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = "anthropic"
ANTHROPIC_AUTO_AGENT_HAIKU_MODEL = "claude-haiku-4-5-20251001"

AAWM_BASIC_ALIAS = "basic"
AAWM_RETIRED_ALIASES = frozenset({"read"})
AAWM_RETIRED_ALIAS_HASHES = frozenset(
    {
        "f50865badf3ac5ba1d3e91eb4681a4f4e004ff2e38d2f99c2cd0006b1be191bc",
        "7df3de3d02c2afaaa9033078e0052ffec2d2b64f3d3a1d8f4a10169a9dda4368",
        "5de379251e74acdad023aac30ea2163d54a5f5776aa868bbeeab3451126e5dda",
        "766d4ab5594413b3b546fa11ac4d820389e5deb6cb8c353b16f18e9dddd46b62",
        "2a9c5ad8ab80ede5ac0deb0b7f5226ff7c763df1064e7d9be2cfd7096c552ac0",
        "aa6535a2427bb5e7d46b713541a9cafb65702546a54de35c03dc50d6441d5f37",
        "b6a36789c8ac5815559b1bb7cd5e61c0efb9938e98121892835c291c2ae8d59b",
        "7942fb23a14e87df30f6000c779dbc55723a18aeb2eda4e0f09610681e1685a4",
        "91085282475e9e8d973c46d84df1c0f78c7ee3a61668432b0989b80a0cb86d67",
        "136f3c13382b157518efaefa64e2b2d03e349416da5aac648f79bb3646d90c34",
        "81df25742cace1a6de5fad57e0400e14b2c554d2a09e3cec3688f29b1ebbc539",
        "2727ec8557c43176f7366e0227ae9c2f6ec62cdfc50f551f84474cffedbe179e",
        "d5ace7473125cf5b14c582ca2fe9c318d5dcd3da507708016daf7e819f8ba005",
        "cebe1de4946d4826cdb900f138c7ffb9051e835d8428681eeeed1c9c0abe5cbe",
        "adc064a3db595face740b696622b748eabb5421ca77b3581a36b972343fdc317",
        "b0bf4c425e6df7790a5753eba7326a2ecb86d5751fe5a4b5f94e68cba9e68eec",
        "48d6b59556978de1af06e555d2c64da2f4be66bf7af09bb6c536973c40c80c5d",
        "9028dcbe8bc7925955cb70318e5bba8860364b895f5f373cd0901894634cc52b",
    }
)

CODEX_AUTO_AGENT_MODEL_ALIAS = AAWM_BASIC_ALIAS
CODEX_AAWM_READ_ALIAS = AAWM_BASIC_ALIAS
CODEX_AAWM_SOTA_ALIAS = "sota"
CODEX_AAWM_CODE_ALIAS = "work"
CODEX_AAWM_LOW_ALIAS = AAWM_BASIC_ALIAS
CODEX_AAWM_ORCHESTRATION_ALIAS = "work"
CODEX_AAWM_SOTA_OPENAI_ALIAS = "sota-openai"
CODEX_AAWM_SOTA_XAI_ALIAS = "sota-xai"
CODEX_AAWM_SOTA_MOONSHOT_ALIAS = "sota-moonshot"
CODEX_AAWM_SOTA_ALIBABA_ALIAS = "sota-alibaba"
CODEX_AAWM_SOTA_DEEPSEEK_ALIAS = "sota-deepseek"
CODEX_AAWM_SOTA_GLM_ALIAS = "alibaba_token_plan/glm-5.2"
ANTHROPIC_AUTO_AGENT_MODEL_ALIAS = AAWM_BASIC_ALIAS
ANTHROPIC_AAWM_READ_ALIAS = AAWM_BASIC_ALIAS
ANTHROPIC_AAWM_SOTA_ALIAS = "sota-anthropic"
ANTHROPIC_AAWM_CODE_ALIAS = "work"
ANTHROPIC_AAWM_LOW_ALIAS = AAWM_BASIC_ALIAS
ANTHROPIC_AAWM_ORCHESTRATION_ALIAS = "work"

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
CODEX_AAWM_READ_ALIAS = AAWM_BASIC_ALIAS
CODEX_AAWM_SOTA_ALIAS = "sota"
CODEX_AAWM_CODE_ALIAS = "work"
CODEX_AAWM_LOW_ALIAS = AAWM_BASIC_ALIAS
CODEX_AAWM_ORCHESTRATION_ALIAS = "work"
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
CODEX_AAWM_SOTA_OPENAI_ALIAS = "sota-openai"
CODEX_AAWM_SOTA_XAI_ALIAS = "sota-xai"
CODEX_AAWM_SOTA_OPENAI_CANDIDATES: tuple[dict[str, Any], ...] = CODEX_AAWM_SOTA_CANDIDATES
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
CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-max",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
        "metadata_gate": "think_effort",
    },
    {
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-high",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": True,
        "metadata_gate": "think_effort",
    },
)
CODEX_AAWM_SOTA_ALIBABA_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.8-max",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.7-max",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
        "last_resort": True,
    },
)
CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/deepseek-v4-pro",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
        "last_resort": True,
    },
)
CODEX_AAWM_SOTA_GLM_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/glm-5.2",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
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
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-high",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
        "metadata_gate": "think_effort",
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
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.6-flash",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/kimi-for-coding",
        "route_family": "codex_kimi_chat_completions_adapter",
        "last_resort": False,
        "metadata_gate": "model_id",
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
    AAWM_BASIC_ALIAS: CODEX_AUTO_AGENT_CANDIDATES,
}
ANTHROPIC_AUTO_AGENT_MODEL_ALIAS = AAWM_BASIC_ALIAS
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
ANTHROPIC_AAWM_READ_ALIAS = AAWM_BASIC_ALIAS
ANTHROPIC_AAWM_SOTA_ALIAS = "sota-anthropic"
ANTHROPIC_AAWM_CODE_ALIAS = "work"
ANTHROPIC_AAWM_LOW_ALIAS = AAWM_BASIC_ALIAS
ANTHROPIC_AAWM_ORCHESTRATION_ALIAS = "work"
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
ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-max",
        "route_family": "anthropic_kimi_chat_completions_adapter",
        "last_resort": False,
        "metadata_gate": "think_effort",
    },
    {
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-high",
        "route_family": "anthropic_kimi_chat_completions_adapter",
        "last_resort": True,
        "metadata_gate": "think_effort",
    },
)
ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.8-max",
        "route_family": "anthropic_alibaba_token_plan_chat_completions_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.7-max",
        "route_family": "anthropic_alibaba_token_plan_chat_completions_adapter",
        "last_resort": True,
    },
)
ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/deepseek-v4-pro",
        "route_family": "anthropic_alibaba_token_plan_chat_completions_adapter",
        "last_resort": True,
    },
)
ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/glm-5.2",
        "route_family": "anthropic_alibaba_token_plan_chat_completions_adapter",
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
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/k3-high",
        "route_family": "anthropic_kimi_chat_completions_adapter",
        "last_resort": False,
        "metadata_gate": "think_effort",
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
        "provider": CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        "model": "alibaba_token_plan/qwen3.6-flash",
        "route_family": "anthropic_alibaba_token_plan_chat_completions_adapter",
        "last_resort": False,
    },
    {
        "provider": CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        "model": "kimi_code/kimi-for-coding",
        "route_family": "anthropic_kimi_chat_completions_adapter",
        "last_resort": False,
        "metadata_gate": "model_id",
    },
    {
        "provider": ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": ANTHROPIC_AUTO_AGENT_HAIKU_MODEL,
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[str, tuple[dict[str, Any], ...]] = {
    AAWM_BASIC_ALIAS: ANTHROPIC_AUTO_AGENT_CANDIDATES,
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
        "alibaba_token_plan/qwen3.8-max",
        "alibaba_token_plan/qwen3.7-plus",
        "alibaba_token_plan/qwen3.7-max",
        "alibaba_token_plan/qwen3.6-flash",
        "alibaba_token_plan/deepseek-v4-pro",
        "alibaba_token_plan/glm-5.2",
    }
)

# ---------------------------------------------------------------------------
# Compatibility publication API (D1-591).
#
# ``llm_passthrough_endpoints.py`` currently contains ~65 lines of
# ``_LOCAL_NAME = _POLICY_PUBLIC_NAME`` boilerplate (lines ~405-470) that
# re-bind policy-owned constants under underscore-prefixed module-local names.
# ``install_policy_compat_aliases`` replaces that block with a single call,
# installing same-object references into a caller-supplied ``host_globals``
# mapping (typically the god module's ``globals()``).
#
# The mapping is deterministic, idempotent, and never imports the god module.
# ---------------------------------------------------------------------------

# Ordered mapping: god-module local name -> policy.py public constant name.
# The iteration order matches the historical assignment order in
# llm_passthrough_endpoints.py so diffs and audits stay line-aligned.
COMPAT_ALIAS_MAP: dict[str, str] = {
    "_CODEX_AUTO_AGENT_MODEL_ALIAS": "CODEX_AUTO_AGENT_MODEL_ALIAS",
    "_CODEX_AUTO_AGENT_NATIVE_PROVIDER": "CODEX_AUTO_AGENT_NATIVE_PROVIDER",
    "_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER": "CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER",
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER": "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER": "CODEX_AUTO_AGENT_OPENROUTER_PROVIDER",
    "_CODEX_AUTO_AGENT_XAI_PROVIDER": "CODEX_AUTO_AGENT_XAI_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENCODE_PROVIDER": "CODEX_AUTO_AGENT_OPENCODE_PROVIDER",
    "_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY": "CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY",
    "_CODEX_AUTO_AGENT_XAI_LANE_KEY": "CODEX_AUTO_AGENT_XAI_LANE_KEY",
    "_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY": "CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY",
    "_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY": "CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY",
    "_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY": "CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY",
    "_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY": "CODEX_AUTO_AGENT_OPENCODE_LANE_KEY",
    "_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS": "CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS",
    "_CODEX_AUTO_AGENT_CANDIDATES": "CODEX_AUTO_AGENT_CANDIDATES",
    "_CODEX_AAWM_READ_ALIAS": "CODEX_AAWM_READ_ALIAS",
    "_CODEX_AAWM_SOTA_ALIAS": "CODEX_AAWM_SOTA_ALIAS",
    "_CODEX_AAWM_CODE_ALIAS": "CODEX_AAWM_CODE_ALIAS",
    "_CODEX_AAWM_LOW_ALIAS": "CODEX_AAWM_LOW_ALIAS",
    "_CODEX_AAWM_ORCHESTRATION_ALIAS": "CODEX_AAWM_ORCHESTRATION_ALIAS",
    "_CODEX_AAWM_SOTA_CANDIDATES": "CODEX_AAWM_SOTA_CANDIDATES",
    "_CODEX_AAWM_SOTA_OPENAI_ALIAS": "CODEX_AAWM_SOTA_OPENAI_ALIAS",
    "_CODEX_AAWM_SOTA_XAI_ALIAS": "CODEX_AAWM_SOTA_XAI_ALIAS",
    "_CODEX_AAWM_SOTA_MOONSHOT_ALIAS": "CODEX_AAWM_SOTA_MOONSHOT_ALIAS",
    "_CODEX_AAWM_SOTA_ALIBABA_ALIAS": "CODEX_AAWM_SOTA_ALIBABA_ALIAS",
    "_CODEX_AAWM_SOTA_DEEPSEEK_ALIAS": "CODEX_AAWM_SOTA_DEEPSEEK_ALIAS",
    "_CODEX_AAWM_SOTA_GLM_ALIAS": "CODEX_AAWM_SOTA_GLM_ALIAS",
    "_CODEX_AAWM_SOTA_OPENAI_CANDIDATES": "CODEX_AAWM_SOTA_OPENAI_CANDIDATES",
    "_CODEX_AAWM_SOTA_XAI_CANDIDATES": "CODEX_AAWM_SOTA_XAI_CANDIDATES",
    "_CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES": "CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES",
    "_CODEX_AAWM_SOTA_ALIBABA_CANDIDATES": "CODEX_AAWM_SOTA_ALIBABA_CANDIDATES",
    "_CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES": "CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES",
    "_CODEX_AAWM_SOTA_GLM_CANDIDATES": "CODEX_AAWM_SOTA_GLM_CANDIDATES",
    "_CODEX_AAWM_CODE_CANDIDATES": "CODEX_AAWM_CODE_CANDIDATES",
    "_CODEX_AAWM_LOW_CANDIDATES": "CODEX_AAWM_LOW_CANDIDATES",
    "_CODEX_AAWM_ORCHESTRATION_CANDIDATES": "CODEX_AAWM_ORCHESTRATION_CANDIDATES",
    "_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS": "CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS",
    "_ANTHROPIC_AUTO_AGENT_MODEL_ALIAS": "ANTHROPIC_AUTO_AGENT_MODEL_ALIAS",
    "_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER": "ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER",
    "_ANTHROPIC_AUTO_AGENT_HAIKU_MODEL": "ANTHROPIC_AUTO_AGENT_HAIKU_MODEL",
    "_ANTHROPIC_AUTO_AGENT_CANDIDATES": "ANTHROPIC_AUTO_AGENT_CANDIDATES",
    "_ANTHROPIC_AAWM_READ_ALIAS": "ANTHROPIC_AAWM_READ_ALIAS",
    "_ANTHROPIC_AAWM_SOTA_ALIAS": "ANTHROPIC_AAWM_SOTA_ALIAS",
    "_ANTHROPIC_AAWM_CODE_ALIAS": "ANTHROPIC_AAWM_CODE_ALIAS",
    "_ANTHROPIC_AAWM_LOW_ALIAS": "ANTHROPIC_AAWM_LOW_ALIAS",
    "_ANTHROPIC_AAWM_ORCHESTRATION_ALIAS": "ANTHROPIC_AAWM_ORCHESTRATION_ALIAS",
    "_ANTHROPIC_AAWM_SOTA_CANDIDATES": "ANTHROPIC_AAWM_SOTA_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES": "ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES": "ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES": "ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES",
    "_ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES": "ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES",
    "_ANTHROPIC_AAWM_CODE_CANDIDATES": "ANTHROPIC_AAWM_CODE_CANDIDATES",
    "_ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES": "ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES",
    "_ANTHROPIC_AAWM_LOW_CANDIDATES": "ANTHROPIC_AAWM_LOW_CANDIDATES",
    "_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS": "ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS",
    "_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS",
    "_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS": "ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS",
    "_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS": "KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS",
    "_ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS": "ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS",
}

# Number of god-module assignment lines this replaces (lines ~405-470 minus
# the leading comment line).
COMPAT_ALIAS_COUNT: int = len(COMPAT_ALIAS_MAP)  # 65


def install_policy_compat_aliases(host_globals: dict[str, Any]) -> None:
    """Install historical god-module policy alias names into *host_globals*.

    Each key in :data:`COMPAT_ALIAS_MAP` is set to the **same object** as the
    corresponding policy-owned constant (identity, not copy).  The function is
    idempotent: calling it twice produces the same result.

    Parameters
    ----------
    host_globals:
        A mutable mapping, typically ``globals()`` from the consuming module.

    Raises
    ------
    TypeError
        If *host_globals* is not a ``dict`` (or ``MutableMapping``).
    """
    if not isinstance(host_globals, dict):
        raise TypeError(
            f"host_globals must be a dict, got {type(host_globals).__name__}"
        )
    _policy_ns = globals()
    for local_name, policy_name in COMPAT_ALIAS_MAP.items():
        host_globals[local_name] = _policy_ns[policy_name]
