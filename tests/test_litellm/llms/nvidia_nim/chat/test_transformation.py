import json
from pathlib import Path

import litellm

from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig


REPO_ROOT = Path(__file__).resolve().parents[5]
NVIDIA_ADAPTER_COST_MAP_MODELS = {
    "nvidia_nim/deepseek-ai/deepseek-v3.1-terminus",
    "nvidia_nim/deepseek-ai/deepseek-v3.2",
    "nvidia_nim/minimaxai/minimax-m2.7",
    "nvidia_nim/mistralai/devstral-2-123b-instruct-2512",
    "nvidia_nim/z-ai/glm4.7",
}


def test_nvidia_nim_reasoning_effort_supported_from_model_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "nvidia_nim/vendor/reasoning-model",
        {
            "input_cost_per_token": 0.0,
            "litellm_provider": "nvidia_nim",
            "mode": "chat",
            "output_cost_per_token": 0.0,
            "supports_reasoning": True,
        },
    )

    config = NvidiaNimConfig()

    supported_params = config.get_supported_openai_params("vendor/reasoning-model")
    optional_params = config.map_openai_params(
        non_default_params={
            "reasoning_effort": "high",
            "cache_control": {"type": "ephemeral"},
        },
        optional_params={},
        model="vendor/reasoning-model",
        drop_params=False,
    )

    assert "reasoning_effort" in supported_params
    assert optional_params["reasoning_effort"] == "high"
    assert "cache_control" not in optional_params


def test_nvidia_nim_reasoning_effort_stripped_without_model_metadata() -> None:
    config = NvidiaNimConfig()

    supported_params = config.get_supported_openai_params("vendor/plain-model")
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="vendor/plain-model",
        drop_params=False,
    )

    assert "reasoning_effort" not in supported_params
    assert "reasoning_effort" not in optional_params


def test_nvidia_anthropic_adapter_models_have_nonzero_cost_map_coverage() -> None:
    for path in (
        REPO_ROOT / "model_prices_and_context_window.json",
        REPO_ROOT / "litellm" / "bundled_model_prices_and_context_window_fallback.json",
    ):
        model_cost = json.loads(path.read_text())
        for model in NVIDIA_ADAPTER_COST_MAP_MODELS:
            entry = model_cost[model]
            assert entry["litellm_provider"] == "nvidia_nim"
            assert entry["mode"] == "chat"
            assert entry["input_cost_per_token"] > 0
            assert entry["output_cost_per_token"] > 0
            assert entry["supports_function_calling"] is True
            assert entry["supports_tool_choice"] is True


def test_nvidia_minimax_cost_map_uses_openrouter_fallback_pricing_basis() -> None:
    for path in (
        REPO_ROOT / "model_prices_and_context_window.json",
        REPO_ROOT / "litellm" / "bundled_model_prices_and_context_window_fallback.json",
    ):
        entry = json.loads(path.read_text())["nvidia_nim/minimaxai/minimax-m2.7"]
        assert entry["pricing_source_model"] == "openrouter/minimax/minimax-m2.5"
        assert entry["pricing_source"] == "https://openrouter.ai/minimax/minimax-m2.5"
        assert entry["input_cost_per_token"] == 1.5e-07
        assert entry["output_cost_per_token"] == 1.15e-06
