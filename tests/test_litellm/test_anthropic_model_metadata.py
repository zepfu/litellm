import json

import litellm


def test_anthropic_sonnet_5_and_fable_5_metadata_offline(monkeypatch):
    """
    Offline metadata, cost, and provider detection test for Claude Sonnet 5 and Fable 5.
    No live Anthropic calls. Verifies model_prices_and_context_window entries,
    get_model_info, cost_per_token, and bundled fallback parity.
    Sonnet 5 uses standard Sonnet 4.6-equivalent pricing per D1-457.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    # Sonnet 5 base (standard Sonnet 4.6-equivalent pricing, not promotional)
    sonnet5_base_expected = {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
    }
    sonnet5_base_models = [
        "claude-sonnet-5",
        "anthropic.claude-sonnet-5",
        "azure_ai/claude-sonnet-5",
        "vertex_ai/claude-sonnet-5",
        "vertex_ai/claude-sonnet-5@default",
        "global.anthropic.claude-sonnet-5",
    ]

    for model in sonnet5_base_models:
        info = litellm.get_model_info(model)
        for k, v in sonnet5_base_expected.items():
            assert info.get(k) == v, f"{model} {k} mismatch: {info.get(k)} != {v}"
        # cache/read rates exist and are Sonnet-4.6-style (non-promotional)
        assert info.get("cache_creation_input_token_cost") == 3.75e-06
        assert info.get("cache_read_input_token_cost") == 3e-07
        assert info.get("supports_reasoning") is True
        assert info.get("supports_prompt_caching") is True
        # above-200k tiers preserved from Sonnet 4.6 style (for models that carry them)
        if info.get("input_cost_per_token_above_200k_tokens") is not None:
            assert info.get("input_cost_per_token_above_200k_tokens") == 6e-06
            assert info.get("output_cost_per_token_above_200k_tokens") == 2.25e-05

    # Regional Bedrock variants use uplift (mirrors sonnet-4-6)
    regional_expected = {
        "input_cost_per_token": 3.3e-06,
        "output_cost_per_token": 1.65e-05,
    }
    regional_models = [
        "eu.anthropic.claude-sonnet-5",
        "au.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
    ]
    for model in regional_models:
        info = litellm.get_model_info(model)
        assert info["litellm_provider"] == "bedrock_converse"
        for k, v in regional_expected.items():
            assert info.get(k) == v, f"{model} {k} mismatch"

    # Cost sanity for base Sonnet 5 using sub-200k tokens (avoids tier math)
    p, c = litellm.cost_per_token(model="claude-sonnet-5", prompt_tokens=100_000, completion_tokens=10_000)
    assert abs(p - 0.3) < 1e-9
    assert abs(c - 0.15) < 1e-9

    # Fable 5 (audited + extended coverage; base pricing for added variants)
    fable_expected_rates = {
        "cache_creation_input_token_cost": 1.25e-05,
        "cache_creation_input_token_cost_above_1hr": 2e-05,
        "cache_read_input_token_cost": 1e-06,
        "input_cost_per_token": 1e-05,
        "output_cost_per_token": 5e-05,
    }
    fable_models = {
        "claude-fable-5": "anthropic",
        "anthropic.claude-fable-5": "bedrock_converse",
        "vertex_ai/claude-fable-5": "vertex_ai-anthropic_models",
        "vertex_ai/claude-fable-5@default": "vertex_ai-anthropic_models",
        "azure_ai/claude-fable-5": "azure_ai",
        "eu.anthropic.claude-fable-5": "bedrock_converse",
        "au.anthropic.claude-fable-5": "bedrock_converse",
        "us.anthropic.claude-fable-5": "bedrock_converse",
        "global.anthropic.claude-fable-5": "bedrock_converse",
    }

    for model, provider in fable_models.items():
        info = litellm.get_model_info(model)
        assert info["litellm_provider"] == provider
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        for key, expected_value in fable_expected_rates.items():
            assert info.get(key) == expected_value, f"{model} {key}"

    # Cost for Fable 5
    p, c = litellm.cost_per_token(model="claude-fable-5", prompt_tokens=1000, completion_tokens=10)
    assert p == 0.01
    assert c == 0.0005

    # Bundled fallback parity (used for fully offline runs)
    with open("litellm/bundled_model_prices_and_context_window_fallback.json") as f:
        bundled = json.load(f)

    for model in sonnet5_base_models + list(fable_models.keys()):
        assert model in bundled, f"{model} missing from bundled fallback"
        b = bundled[model]
        if "sonnet-5" in model and model not in regional_models:
            assert b["input_cost_per_token"] == 3e-06
            assert b["output_cost_per_token"] == 1.5e-05
        if "fable-5" in model:
            assert b["input_cost_per_token"] == 1e-05
            assert b["output_cost_per_token"] == 5e-05

    # Provider detection / routing surface smoke (no live)
    provider_cases = {
        "claude-sonnet-5": "anthropic",
        "anthropic.claude-sonnet-5": "bedrock_converse",
        "vertex_ai/claude-sonnet-5": "vertex_ai",
        "azure_ai/claude-sonnet-5": "azure_ai",
        "claude-fable-5": "anthropic",
        "anthropic.claude-fable-5": "bedrock_converse",
        "vertex_ai/claude-fable-5": "vertex_ai",
        "azure_ai/claude-fable-5": "azure_ai",
    }
    for model, expected_provider in provider_cases.items():
        _, provider, _, _ = litellm.get_llm_provider(model)
        assert provider == expected_provider
