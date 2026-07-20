import json

import litellm
import pytest

from litellm.integrations.aawm_agent_identity import (
    _build_session_history_db_payload,
    _build_session_history_record,
)


KIMI_CODE_REFERENCE_PRICING = {
    "k3": {
        "cache_read_input_token_cost": 3e-07,
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "source": "https://www.kimi.com/resources/kimi-k3-pricing",
    },
    "kimi-for-coding": {
        "cache_read_input_token_cost": 1.9e-07,
        "input_cost_per_token": 9.5e-07,
        "output_cost_per_token": 4e-06,
        "source": "https://www.kimi.com/resources/kimi-k2-7-code-pricing",
    },
    "kimi-for-coding-highspeed": {
        "cache_read_input_token_cost": 3.8e-07,
        "input_cost_per_token": 1.9e-06,
        "output_cost_per_token": 8e-06,
        "source": "https://www.kimi.com/resources/kimi-k2-7-code-pricing",
    },
}


@pytest.fixture(autouse=True)
def use_canonical_kimi_code_cost_map():
    previous_model_cost_map = litellm.model_cost
    with open("model_prices_and_context_window.json") as f:
        litellm.model_cost = json.load(f)
    try:
        yield
    finally:
        litellm.model_cost = previous_model_cost_map


def _kimi_code_kwargs(model: str) -> dict:
    return {
        "litellm_call_id": f"call-{model}",
        "model": model,
        "custom_llm_provider": "kimi_code",
        "call_type": "acompletion",
        "litellm_params": {
            "metadata": {
                "session_id": f"session-{model}",
                "trace_name": "moonshot.cost-accounting",
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
        "passthrough_logging_payload": {"request_body": {"messages": []}},
    }


@pytest.mark.parametrize("model", KIMI_CODE_REFERENCE_PRICING)
@pytest.mark.parametrize(
    "prompt_tokens,cached_tokens,completion_tokens,reasoning_tokens",
    [
        (1_000_000, 1_000_000, 0, 0),
        (1_000_000, 0, 0, 0),
        (0, 0, 1_000_000, 0),
        (1_000_000, 400_000, 1_000_000, 400_000),
    ],
    ids=["cached-only", "uncached-only", "output-only", "mixed-reasoning"],
)
def test_should_persist_kimi_code_managed_reference_cost_provenance(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
) -> None:
    record = _build_session_history_record(
        kwargs=_kimi_code_kwargs(model),
        result={
            "id": f"resp-{model}",
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
                "completion_tokens_details": {
                    "reasoning_tokens": reasoning_tokens,
                },
            },
            "choices": [{"message": {"role": "assistant", "content": "ack"}}],
        },
        start_time="2026-07-19T18:00:00Z",
        end_time="2026-07-19T18:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    assert record["provider"] == "kimi_code"
    assert record["response_cost_usd"] is None

    expected = KIMI_CODE_REFERENCE_PRICING[model]
    uncached_tokens = prompt_tokens - cached_tokens
    metadata = record["metadata"]
    assert metadata["billing_mode"] == "kimi_code_subscription"
    assert metadata["actual_invoice_cost_known"] is False
    assert metadata["reference_cost_kind"] == "official_public_subscription_route"
    assert metadata["reference_cost_currency"] == "USD"
    assert metadata["reference_cost_model"] == f"kimi_code/{model}"
    assert metadata["reference_cost_source"] == expected["source"]
    assert metadata["reference_cost_cached_input_usd"] == pytest.approx(
        cached_tokens * expected["cache_read_input_token_cost"]
    )
    assert metadata["reference_cost_uncached_input_usd"] == pytest.approx(
        uncached_tokens * expected["input_cost_per_token"]
    )
    assert metadata["reference_cost_output_usd"] == pytest.approx(completion_tokens * expected["output_cost_per_token"])
    assert metadata["reference_cost_total_usd"] == pytest.approx(
        metadata["reference_cost_cached_input_usd"]
        + metadata["reference_cost_uncached_input_usd"]
        + metadata["reference_cost_output_usd"]
    )
    assert metadata["reference_cost_total_usd"] == pytest.approx(
        cached_tokens * expected["cache_read_input_token_cost"]
        + uncached_tokens * expected["input_cost_per_token"]
        + completion_tokens * expected["output_cost_per_token"]
    )

    persisted_metadata = json.loads(_build_session_history_db_payload(record)[52])
    for key in (
        "billing_mode",
        "actual_invoice_cost_known",
        "reference_cost_kind",
        "reference_cost_currency",
        "reference_cost_model",
        "reference_cost_source",
        "reference_cost_cached_input_usd",
        "reference_cost_uncached_input_usd",
        "reference_cost_output_usd",
        "reference_cost_total_usd",
    ):
        assert persisted_metadata[key] == metadata[key]


@pytest.mark.parametrize("model", ["k3-low", "k3-high", "k3-max"])
def test_should_price_k3_effort_variants_with_the_k3_upstream_rates(
    model: str,
) -> None:
    record = _build_session_history_record(
        kwargs=_kimi_code_kwargs(model),
        result={
            "id": f"resp-{model}",
            "model": model,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "choices": [{"message": {"role": "assistant", "content": "ack"}}],
        },
        start_time="2026-07-20T00:00:00Z",
        end_time="2026-07-20T00:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    metadata = record["metadata"]
    assert metadata["reference_cost_model"] == f"kimi_code/{model}"
    assert metadata["reference_cost_source"] == KIMI_CODE_REFERENCE_PRICING["k3"]["source"]
    assert metadata["reference_cost_cached_input_usd"] == pytest.approx(4 * 3e-07)
    assert metadata["reference_cost_uncached_input_usd"] == pytest.approx(6 * 3e-06)
    assert metadata["reference_cost_output_usd"] == pytest.approx(5 * 1.5e-05)
