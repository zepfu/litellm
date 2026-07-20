import json

import litellm
import pytest
from litellm.types.utils import Usage


KIMI_CODE_REFERENCE_PRICING = {
    "kimi_code/k3": {
        "cache_read_input_token_cost": 3e-07,
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "source": "https://www.kimi.com/resources/kimi-k3-pricing",
    },
    "kimi_code/kimi-for-coding": {
        "cache_read_input_token_cost": 1.9e-07,
        "input_cost_per_token": 9.5e-07,
        "output_cost_per_token": 4e-06,
        "source": "https://www.kimi.com/resources/kimi-k2-7-code-pricing",
    },
    "kimi_code/kimi-for-coding-highspeed": {
        "cache_read_input_token_cost": 3.8e-07,
        "input_cost_per_token": 1.9e-06,
        "output_cost_per_token": 8e-06,
        "source": "https://www.kimi.com/resources/kimi-k2-7-code-pricing",
    },
}
KIMI_CODE_REFERENCE_COST_COMMENT = (
    "Public reference costs for the kimi_code subscription route, not known invoice billing."
)


@pytest.fixture(autouse=True)
def use_canonical_kimi_code_cost_map():
    previous_model_cost_map = litellm.model_cost
    with open("model_prices_and_context_window.json") as f:
        litellm.model_cost = json.load(f)
    try:
        yield
    finally:
        litellm.model_cost = previous_model_cost_map


def _get_kimi_code_model_info(model: str):
    _, model_id = model.split("/", maxsplit=1)
    return litellm.get_model_info(
        model=model_id,
        custom_llm_provider="kimi_code",
    )


def test_should_expose_kimi_code_public_reference_pricing_in_both_model_maps():
    with open("model_prices_and_context_window.json") as f:
        canonical_model_costs = json.load(f)
    with open("litellm/bundled_model_prices_and_context_window_fallback.json") as f:
        bundled_model_costs = json.load(f)

    for model, expected_pricing in KIMI_CODE_REFERENCE_PRICING.items():
        model_info = _get_kimi_code_model_info(model)
        canonical_model_info = canonical_model_costs[model]
        bundled_model_info = bundled_model_costs[model]

        assert model_info["litellm_provider"] == "kimi_code"
        assert canonical_model_info["litellm_provider"] == "kimi_code"
        assert bundled_model_info["litellm_provider"] == "kimi_code"
        assert canonical_model_info["comment"] == KIMI_CODE_REFERENCE_COST_COMMENT
        assert bundled_model_info["comment"] == KIMI_CODE_REFERENCE_COST_COMMENT
        assert "max_output_tokens" not in canonical_model_info
        assert "max_output_tokens" not in bundled_model_info
        for key, expected_value in expected_pricing.items():
            assert canonical_model_info[key] == expected_value
            assert bundled_model_info[key] == expected_value
        for key in (
            "cache_read_input_token_cost",
            "input_cost_per_token",
            "output_cost_per_token",
        ):
            assert model_info[key] == expected_pricing[key]


def test_should_expose_only_documented_k3_cost_capabilities():
    k3_info = _get_kimi_code_model_info("kimi_code/k3")

    assert k3_info["supports_function_calling"] is True
    assert k3_info["supports_prompt_caching"] is True
    assert k3_info["supports_response_schema"] is True
    assert k3_info["supports_tool_choice"] is True


@pytest.mark.parametrize(
    "model, cache_read_rate, input_rate, output_rate",
    [
        (
            "kimi_code/k3",
            3e-07,
            3e-06,
            1.5e-05,
        ),
        (
            "kimi_code/kimi-for-coding",
            1.9e-07,
            9.5e-07,
            4e-06,
        ),
        (
            "kimi_code/kimi-for-coding-highspeed",
            3.8e-07,
            1.9e-06,
            8e-06,
        ),
    ],
)
@pytest.mark.parametrize(
    "prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens",
    [
        (1_000_000, 1_000_000, 0, 0),
        (1_000_000, 0, 0, 0),
        (0, 0, 1_000_000, 0),
        (1_000_000, 400_000, 1_000_000, 400_000),
    ],
)
def test_should_calculate_kimi_code_reference_costs_without_a_second_reasoning_charge(
    model: str,
    cache_read_rate: float,
    input_rate: float,
    output_rate: float,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
):
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=cached_tokens,
        prompt_tokens_details={"cached_tokens": cached_tokens},
        completion_tokens_details={"reasoning_tokens": reasoning_tokens},
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model.split("/", maxsplit=1)[1],
        custom_llm_provider="kimi_code",
        usage_object=usage,
    )

    assert prompt_cost == pytest.approx(
        ((prompt_tokens - cached_tokens) * input_rate) + (cached_tokens * cache_read_rate)
    )
    assert completion_cost == pytest.approx(completion_tokens * output_rate)
