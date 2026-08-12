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
    low = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="alibaba_token_plan",
        model="qwen3.6-flash",
        prompt_tokens=255_999,
        completion_tokens=1,
        usage_obj={},
    )
    high = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="alibaba_token_plan",
        model="qwen3.6-flash",
        prompt_tokens=256_000,
        completion_tokens=1,
        usage_obj={},
    )

    assert low is not None
    assert high is not None
    assert low["reference_cost_model"] == "alibaba_token_plan/qwen3.6-flash"
    assert low["reference_cost_total_usd"] == pytest.approx(
        (255_999 * 0.25 + 1 * 1.50) / 1_000_000
    )
    assert high["reference_cost_total_usd"] == pytest.approx(
        (256_000 * 1.0 + 1 * 4.0) / 1_000_000
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
    result = aawm_agent_identity.resolve_aawm_reference_pricing(
        provider="openrouter",
        model="cohere/north-mini-code:free",
        prompt_tokens=1_000,
        completion_tokens=200,
        usage_obj={},
    )

    assert result is not None
    assert result["reference_cost_total_usd"] == pytest.approx(
        (1_000 * 0.2 + 200 * 0.8) / 1_000_000
    )
    assert result["actual_invoice_cost_known"] is False
    assert result["reference_cost_model"] == "openrouter/cohere/north-mini-code:free"


def test_should_keep_provider_reported_cost_authoritative(catalog):
    kwargs = {
        "litellm_call_id": "call-reference-cost-provider-wins",
        "model": "cohere/north-mini-code:free",
        "custom_llm_provider": "openrouter",
        "call_type": "completion",
        "response_cost": 0.123,
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
            },
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
        start_time="2026-08-12T12:00:00Z",
        end_time="2026-08-12T12:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    assert record["response_cost_usd"] == pytest.approx(0.123)
    assert record["metadata"]["reference_cost_total_usd"] == pytest.approx(
        (1_000 * 0.2 + 200 * 0.8) / 1_000_000
    )
