import litellm

from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig


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
