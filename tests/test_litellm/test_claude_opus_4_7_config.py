"""
Validate Claude Opus 4.7 model configuration entries.
"""

import json
import os


def test_opus_4_7_model_pricing_and_capabilities():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        model_data = json.load(f)

    assert "claude-opus-4-7" in model_data, "Missing model entry: claude-opus-4-7"
    info = model_data["claude-opus-4-7"]

    assert info["litellm_provider"] == "anthropic"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1000000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000

    assert info["input_cost_per_token"] == 5e-06
    assert info["output_cost_per_token"] == 2.5e-05
    assert info["cache_creation_input_token_cost"] == 6.25e-06
    assert info["cache_read_input_token_cost"] == 5e-07
    assert info["input_cost_per_token_above_200k_tokens"] == 1e-05
    assert info["output_cost_per_token_above_200k_tokens"] == 3.75e-05
    assert info["cache_creation_input_token_cost_above_200k_tokens"] == 1.25e-05
    assert info["cache_read_input_token_cost_above_200k_tokens"] == 1e-06

    assert info["supports_assistant_prefill"] is False
    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["tool_use_system_prompt_tokens"] == 346
