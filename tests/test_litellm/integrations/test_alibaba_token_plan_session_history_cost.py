import json

from litellm.integrations.aawm_agent_identity import (
    _build_session_history_db_payload,
    _build_session_history_record,
)


def _alibaba_kwargs(model: str) -> dict:
    return {
        "litellm_call_id": f"call-{model}",
        "model": model,
        "custom_llm_provider": "alibaba_token_plan",
        "call_type": "acompletion",
        "litellm_params": {
            "metadata": {
                "session_id": f"session-{model}",
                "trace_name": "alibaba.cost-accounting",
            }
        },
        "standard_logging_object": {"metadata": {}, "request_tags": []},
        "passthrough_logging_payload": {"request_body": {"messages": []}},
    }


def test_should_persist_honest_token_plan_cost_provenance() -> None:
    kwargs = _alibaba_kwargs("qwen3.8-max-preview")
    kwargs["response_cost"] = 0.125
    record = _build_session_history_record(
        kwargs=kwargs,
        result={
            "id": "resp-alibaba",
            "model": "qwen3.8-max-preview",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
            },
            "choices": [{"message": {"role": "assistant", "content": "ack"}}],
        },
        start_time="2026-07-20T23:00:00Z",
        end_time="2026-07-20T23:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    assert record["provider"] == "alibaba_token_plan"
    assert record["response_cost_usd"] is None
    metadata = record["metadata"]
    assert metadata["billing_mode"] == "alibaba_token_plan_subscription"
    assert metadata["actual_invoice_cost_known"] is False
    assert metadata["reference_cost_kind"] == "provider_token_plan_no_public_per_token_rate"
    assert metadata["reference_cost_currency"] == "USD"
    assert metadata["reference_cost_model"] == ("alibaba_token_plan/qwen3.8-max-preview")
    assert metadata["reference_cost_source"] == ("https://www.alibabacloud.com/help/en/model-studio/coding-plan")
    assert "reference_cost_total_usd" not in metadata

    persisted_metadata = json.loads(_build_session_history_db_payload(record)[52])
    for key in (
        "billing_mode",
        "actual_invoice_cost_known",
        "reference_cost_kind",
        "reference_cost_currency",
        "reference_cost_model",
        "reference_cost_source",
    ):
        assert persisted_metadata[key] == metadata[key]


def test_should_not_apply_token_plan_provenance_to_other_providers() -> None:
    kwargs = _alibaba_kwargs("qwen3.8-max-preview")
    kwargs["custom_llm_provider"] = "dashscope"
    record = _build_session_history_record(
        kwargs=kwargs,
        result={
            "id": "resp-dashscope",
            "model": "qwen3.8-max-preview",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "choices": [{"message": {"role": "assistant", "content": "ack"}}],
        },
        start_time="2026-07-20T23:00:00Z",
        end_time="2026-07-20T23:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    assert "billing_mode" not in record["metadata"]
    assert "reference_cost_kind" not in record["metadata"]
