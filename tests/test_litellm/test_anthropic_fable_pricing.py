import json

import litellm


def test_anthropic_fable_5_pricing_metadata():
    expected_rates = {
        "cache_creation_input_token_cost": 1.25e-05,
        "cache_creation_input_token_cost_above_1hr": 2e-05,
        "cache_read_input_token_cost": 1e-06,
        "input_cost_per_token": 1e-05,
        "output_cost_per_token": 5e-05,
    }
    expected_models = {
        "claude-fable-5": "anthropic",
        "anthropic.claude-fable-5": "bedrock_converse",
        "vertex_ai/claude-fable-5": "vertex_ai-anthropic_models",
        "vertex_ai/claude-fable-5@default": "vertex_ai-anthropic_models",
    }

    for model, provider in expected_models.items():
        model_info = litellm.get_model_info(model)
        assert model_info["litellm_provider"] == provider
        assert model_info["max_input_tokens"] == 1000000
        assert model_info["max_output_tokens"] == 128000
        assert model_info["max_tokens"] == 128000
        for key, expected_value in expected_rates.items():
            assert model_info[key] == expected_value

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="claude-fable-5",
        prompt_tokens=1000,
        completion_tokens=10,
    )
    assert prompt_cost == 0.01
    assert completion_cost == 0.0005

    with open("litellm/bundled_model_prices_and_context_window_fallback.json") as f:
        bundled_model_costs = json.load(f)

    for model, provider in expected_models.items():
        model_info = bundled_model_costs[model]
        assert model_info["litellm_provider"] == provider
        for key, expected_value in expected_rates.items():
            assert model_info[key] == expected_value

