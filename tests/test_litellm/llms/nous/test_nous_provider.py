"""NOUS-002: direct Nous provider identity, conservative params, cost-map rows."""

from __future__ import annotations

import json
import os

import pytest

from litellm import get_llm_provider

_NOUS_API_BASE = "https://inference-api.nousresearch.com/v1"
_SEVEN_NOUS_IDS = (
    "stealth/ox-alpha",
    "upstage/solar-pro4:free",
    "meituan/longcat-2.0:free",
    "tencent/hy3:free",
    "poolside/laguna-s-2.1:free",
    "stepfun/step-3.7-flash:free",
    "poolside/laguna-xs-2.1:free",
)
_UNVERIFIED_CAPABILITY_KEYS = (
    "supports_function_calling",
    "supports_tool_choice",
    "supports_streaming",
    "max_input_tokens",
    "max_output_tokens",
    "max_tokens",
)


def _repo_root() -> str:
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )


def test_get_llm_provider_nous():
    model, provider, _, _ = get_llm_provider(model="nous/stealth/ox-alpha")
    assert provider == "nous"
    assert model == "stealth/ox-alpha"

    nested_free, nested_provider, _, _ = get_llm_provider(
        model="nous/upstage/solar-pro4:free"
    )
    assert nested_provider == "nous"
    assert nested_free == "upstage/solar-pro4:free"

    for model_id in _SEVEN_NOUS_IDS:
        stripped, resolved_provider, _, _ = get_llm_provider(model=f"nous/{model_id}")
        assert resolved_provider == "nous"
        assert stripped == model_id


def test_nous_config_initialization():
    from litellm.llms.nous.chat.transformation import NousChatConfig

    config = NousChatConfig()
    assert config.custom_llm_provider == "nous"


def test_nous_get_openai_compatible_provider_info_default_base():
    from litellm.llms.nous.chat.transformation import NousChatConfig

    config = NousChatConfig()
    api_base, _api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == _NOUS_API_BASE

    custom_base = "https://custom.nous.example/v1"
    api_base, api_key = config._get_openai_compatible_provider_info(
        custom_base, "test-key"
    )
    assert api_base == custom_base
    assert api_key == "test-key"


def test_nous_in_provider_lists():
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
    )

    assert "nous" in openai_compatible_providers
    assert _NOUS_API_BASE in openai_compatible_endpoints


def test_nous_supported_params_conservative_until_d3():
    from litellm.exceptions import UnsupportedParamsError
    from litellm.llms.nous.chat.transformation import NousChatConfig

    config = NousChatConfig()
    supported_params = config.get_supported_openai_params("nous/stealth/ox-alpha")

    assert "stream" not in supported_params
    assert "tools" not in supported_params
    assert "tool_choice" not in supported_params

    for param in ("stream", "tools", "tool_choice"):
        with pytest.raises(UnsupportedParamsError):
            config.map_openai_params(
                non_default_params={param: True if param == "stream" else []},
                optional_params={},
                model="nous/stealth/ox-alpha",
                drop_params=False,
            )


def test_nous_cost_map_rows_exist_for_seven_ids():
    canonical_path = os.path.join(_repo_root(), "model_prices_and_context_window.json")
    fallback_path = os.path.join(
        _repo_root(),
        "litellm",
        "bundled_model_prices_and_context_window_fallback.json",
    )
    with open(canonical_path, "r", encoding="utf-8") as handle:
        canonical = json.load(handle)
    with open(fallback_path, "r", encoding="utf-8") as handle:
        bundled = json.load(handle)

    for model_id in _SEVEN_NOUS_IDS:
        cost_key = f"nous/{model_id}"
        assert cost_key in canonical
        row = canonical[cost_key]
        assert row["litellm_provider"] == "nous"
        assert row.get("input_cost_per_token", 1) == 0
        assert row.get("output_cost_per_token", 1) == 0
        for capability_key in _UNVERIFIED_CAPABILITY_KEYS:
            assert capability_key not in row
        assert bundled[cost_key] == row
