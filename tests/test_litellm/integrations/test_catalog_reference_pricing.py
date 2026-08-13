import json
from pathlib import Path
from typing import Any, Dict

import pytest

from litellm.integrations import aawm_agent_identity


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CATALOG = ROOT / "model_prices_and_context_window.json"
BUNDLED_CATALOG = ROOT / "litellm/bundled_model_prices_and_context_window_fallback.json"


def _load_catalog() -> Dict[str, Any]:
    return json.loads(CANONICAL_CATALOG.read_text(encoding="utf-8"))


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    value = _load_catalog()
    monkeypatch.setattr(
        aawm_agent_identity,
        "_load_bundled_model_cost_map",
        lambda: value,
    )
    return value


def test_should_keep_reference_pricing_catalogs_equal_and_gate_unverified_routes():
    canonical = json.loads(CANONICAL_CATALOG.read_text(encoding="utf-8"))
    bundled = json.loads(BUNDLED_CATALOG.read_text(encoding="utf-8"))

    assert canonical == bundled
    assert (
        canonical["alibaba_token_plan/qwen3.8-max"]["provider_specific_entry"][
            "alibaba_token_plan"
        ]["aawm_reference_pricing"]["status"]
        == "unpriced"
    )
    assert (
        canonical["openrouter/owl-alpha"]["provider_specific_entry"]["openrouter"][
            "aawm_reference_pricing"
        ]["unpriced_reason"]
        == "no_verified_longcat_alias_or_redirect"
    )
    assert "input_cost_per_token" not in canonical["openrouter/owl-alpha"]
    assert "output_cost_per_token" not in canonical["openrouter/owl-alpha"]
    assert (
        canonical["opencode/big-pickle"]["provider_specific_entry"]["opencode_zen"][
            "aawm_reference_pricing"
        ]["unpriced_reason"]
        == "no_verified_zai_mapping"
    )
    assert (
        canonical["opencode/deepseek-v4-flash-free"]["provider_specific_entry"][
            "opencode_zen"
        ]["aawm_reference_pricing"]["unpriced_reason"]
        == "no_exact_paid_flash_equivalence"
    )


def test_should_fail_closed_on_exact_provider_key_mismatch(catalog):
    assert (
        aawm_agent_identity.resolve_aawm_reference_pricing(
            provider="deepseek",
            model="openrouter/deepseek/deepseek-v4-flash:free",
            prompt_tokens=10,
            completion_tokens=2,
            usage_obj={},
        )
        is None
    )


def test_should_price_alibaba_qwen36_at_the_256k_boundary(catalog):
    contract = catalog["alibaba_token_plan/qwen3.6-flash"][
        "provider_specific_entry"
    ]["alibaba_token_plan"]["aawm_reference_pricing"]
    base_tier = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="alibaba_token_plan",
        model="qwen3.6-flash",
        prompt_tokens=256_000,
        completion_tokens=100_000,
        usage_obj={},
    )
    high_tier = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="alibaba_token_plan",
        model="qwen3.6-flash",
        prompt_tokens=256_001,
        completion_tokens=1,
        usage_obj={},
    )

    assert base_tier is not None
    assert high_tier is not None
    assert contract["billing_mode"] == "alibaba_token_plan_subscription"
    assert contract["actual_invoice_cost_known"] is False
    assert contract["source"] == {
        "kind": "official_provider_direct_list_rate_catalog",
        "label": "Alibaba Cloud Model Studio international direct list rates",
        "urls": [
            "https://www.alibabacloud.com/help/en/model-studio/model-pricing"
        ],
        "version": "2026-08-12",
        "verified_on": "2026-08-12",
    }
    assert base_tier["reference_cost_model"] == "alibaba_token_plan/qwen3.6-flash"
    assert base_tier["reference_cost_rate_schedule"]["meter"] == "input_tokens"
    assert base_tier["reference_cost_total_usd"] == pytest.approx(
        (256_000 * 0.25 + 100_000 * 1.50) / 1_000_000
    )
    assert high_tier["reference_cost_total_usd"] == pytest.approx(
        (256_001 * 1.0 + 1 * 4.0) / 1_000_000
    )


def test_should_price_direct_deepseek_with_cache_and_mark_unknown_cache_unpriced(
    catalog,
):
    priced = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_tokens=1_000,
        completion_tokens=200,
        usage_obj={"cache_read_input_tokens": 100},
    )
    unknown_cache = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_tokens=1_000,
        completion_tokens=200,
        usage_obj={},
    )

    assert priced is not None
    assert priced["reference_cost_model"] == "deepseek/deepseek-v4-flash"
    assert priced["reference_cost_total_usd"] == pytest.approx(
        (900 * 0.14 + 100 * 0.0028 + 200 * 0.28) / 1_000_000
    )
    assert unknown_cache is not None
    assert unknown_cache["reference_cost_status"] == "unpriced"
    assert unknown_cache["reference_cost_unpriced_reason"] == "cache_token_count_unknown"


@pytest.mark.parametrize(
    (
        "prompt_tokens",
        "cached_input_tokens",
        "completion_tokens",
        "expected_uncached_input_usd",
        "expected_cached_input_usd",
        "expected_output_usd",
    ),
    (
        (1_000, 1_000, 0, 0.0, 1_000 * 0.003625 / 1_000_000, 0.0),
        (1_000, 0, 0, 1_000 * 0.435 / 1_000_000, 0.0, 0.0),
        (
            1_000,
            400,
            0,
            600 * 0.435 / 1_000_000,
            400 * 0.003625 / 1_000_000,
            0.0,
        ),
        (0, 0, 200, 0.0, 0.0, 200 * 0.87 / 1_000_000),
    ),
)
def test_should_price_direct_deepseek_v4_pro_components(
    catalog,
    prompt_tokens,
    cached_input_tokens,
    completion_tokens,
    expected_uncached_input_usd,
    expected_cached_input_usd,
    expected_output_usd,
):
    contract = catalog["deepseek/deepseek-v4-pro"]["provider_specific_entry"][
        "deepseek"
    ]["aawm_reference_pricing"]
    result = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="deepseek",
        model="deepseek-v4-pro",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usage_obj={"cache_read_input_tokens": cached_input_tokens},
    )

    assert contract["rates"] == {
        "input_usd_per_million_tokens": 0.435,
        "cache_read_usd_per_million_tokens": 0.003625,
        "output_usd_per_million_tokens": 0.87,
    }
    assert result is not None
    assert result["reference_cost_status"] == "priced"
    assert result["actual_invoice_cost_known"] is False
    assert result["billing_mode"] == "deepseek_direct_provider"
    assert result["reference_cost_model"] == "deepseek/deepseek-v4-pro"
    assert result["reference_cost_basis_provider"] == "deepseek"
    assert result["reference_cost_basis_model"] == "deepseek/deepseek-v4-pro"
    assert result["reference_cost_source_kind"] == "official_provider_catalog"
    assert result["reference_cost_source_label"] == "DeepSeek API pricing"
    assert result["reference_cost_uncached_input_usd"] == pytest.approx(
        expected_uncached_input_usd
    )
    assert result["reference_cost_cached_input_usd"] == pytest.approx(
        expected_cached_input_usd
    )
    assert result["reference_cost_output_usd"] == pytest.approx(expected_output_usd)
    assert result["reference_cost_total_usd"] == pytest.approx(
        expected_uncached_input_usd
        + expected_cached_input_usd
        + expected_output_usd
    )


def test_should_require_exact_equivalence_for_rates_from_model(monkeypatch):
    direct_rates = {
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
    }
    synthetic_catalog = {
        "deepseek/deepseek-v4-flash": direct_rates,
        "openrouter/example": {
            "provider_specific_entry": {
                "openrouter": {
                    "aawm_reference_pricing": {
                        "schema_version": 1,
                        "status": "priced",
                        "kind": "hosted_reference",
                        "billing_mode": "hosted",
                        "actual_invoice_cost_known": False,
                        "currency": "USD",
                        "rates_from_model": {
                            "provider": "deepseek",
                            "model": "deepseek-v4-flash",
                        },
                        "cache_mode": "none",
                        "equivalence_status": "not_exact",
                        "source": {},
                    }
                }
            }
        },
    }
    monkeypatch.setattr(
        aawm_agent_identity,
        "_load_bundled_model_cost_map",
        lambda: synthetic_catalog,
    )

    result = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="openrouter",
        model="example",
        prompt_tokens=10,
        completion_tokens=2,
        usage_obj={},
    )

    assert result is not None
    assert result["reference_cost_status"] == "unpriced"
    assert (
        result["reference_cost_unpriced_reason"]
        == "rates_from_model_requires_exact_equivalence"
    )


def test_should_preserve_reference_total_as_metadata_only(catalog):
    contract = catalog["openrouter/cohere/north-mini-code:free"][
        "provider_specific_entry"
    ]["openrouter"]["aawm_reference_pricing"]
    result = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="openrouter",
        model="cohere/north-mini-code:free",
        prompt_tokens=1_000,
        completion_tokens=200,
        usage_obj={},
    )

    assert result is not None
    assert contract["cache_mode"] == "none"
    assert "cache_read_usd_per_million_tokens" not in contract["rates"]
    assert result["reference_cost_total_usd"] == pytest.approx(
        (1_000 * 0.2 + 200 * 0.8) / 1_000_000
    )
    assert result["actual_invoice_cost_known"] is False
    assert result["reference_cost_model"] == "openrouter/cohere/north-mini-code:free"
    assert result["reference_cost_basis_model"] == "cohere/north-mini-code-1-0"
    assert (
        result["reference_cost_source_kind"]
        == "third_party_hosted_catalog_consensus"
    )
    assert (
        result["reference_cost_source_label"]
        == "NanoGPT and Routeway hosted-model catalogs"
    )


@pytest.mark.parametrize("reported_cost", [0.0, 0.123])
def test_should_keep_provider_reported_cost_authoritative(catalog, reported_cost):
    kwargs = {
        "litellm_call_id": "call-reference-cost-provider-wins",
        "model": "cohere/north-mini-code:free",
        "custom_llm_provider": "openrouter",
        "call_type": "completion",
        "response_cost": 0.456,
        "litellm_params": {
            "metadata": {
                "session_id": "session-reference-cost-provider-wins",
            }
        },
        "standard_logging_object": {
            "metadata": {},
            "request_tags": [],
        },
        "passthrough_logging_payload": {
            "request_body": {"messages": []},
        },
    }
    record = aawm_agent_identity._build_session_history_record(
        kwargs=kwargs,
        result={
            "model": "cohere/north-mini-code:free",
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 200,
                "total_tokens": 1_200,
                "cost": reported_cost,
            },
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
        start_time="2026-08-12T12:00:00Z",
        end_time="2026-08-12T12:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    assert record["response_cost_usd"] == pytest.approx(reported_cost)
    assert record["metadata"]["reference_cost_total_usd"] == pytest.approx(
        (1_000 * 0.2 + 200 * 0.8) / 1_000_000
    )
