import json
from pathlib import Path
from typing import Iterator

import litellm
import pytest

from litellm.utils import supports_xhigh_reasoning_effort
from litellm.llms.xai.reference_cost import (
    build_xai_grok_46_reference_cost_metadata,
    calculate_xai_reference_cost,
)
from litellm.llms.xai.route_descriptors import (
    GROK_NATIVE_ROUTE_DESCRIPTORS,
    OA_XAI_ROUTE_DESCRIPTORS,
)
from litellm.integrations.aawm_agent_identity import (
    _build_session_history_db_payload,
    _build_session_history_record,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CATALOG_PATHS = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm/bundled_model_prices_and_context_window_fallback.json",
)


@pytest.fixture(autouse=True)
def use_canonical_model_cost_map() -> Iterator[None]:
    previous_model_cost_map = litellm.model_cost
    litellm.model_cost = json.loads(
        (REPO_ROOT / "model_prices_and_context_window.json").read_text()
    )
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = previous_model_cost_map
        litellm.get_model_info.cache_clear()


def test_should_keep_grok_46_catalog_entries_in_parity() -> None:
    for catalog_path in CATALOG_PATHS:
        catalog = json.loads(catalog_path.read_text())
        native = catalog["xai/grok-4.6"]
        managed = catalog["oa_xai/grok-4.6"]
        retained = catalog["xai/grok-4.5"]

        assert "xai/grok-4.5" in catalog
        assert "oa_xai/grok-4.5" in catalog
        assert native == managed
        assert native["created"] == 1785974400
        assert native["owned_by"] == "xai"
        assert native["aliases"] == []
        assert native["max_input_tokens"] == 500000
        assert native["input_cost_per_token"] == pytest.approx(2e-6)
        assert native["cache_read_input_token_cost"] == pytest.approx(0.5e-6)
        assert native["output_cost_per_token"] == pytest.approx(6e-6)
        assert native["input_cost_per_token_above_200k_tokens"] == pytest.approx(4e-6)
        assert native["cache_read_input_token_cost_above_200k_tokens"] == pytest.approx(
            1e-6
        )
        assert native["output_cost_per_token_above_200k_tokens"] == pytest.approx(
            12e-6
        )
        assert native["input_cost_per_image_token"] == pytest.approx(2e-6)
        assert native["verified"] == "2026-08-12"
        for capability in (
            "supports_function_calling",
            "supports_reasoning",
            "supports_tool_choice",
            "supports_vision",
            "supports_web_search",
            "custom_tool_function_adapters",
            "unsupported_hosted_tools",
            "rewrite_input_item_types",
        ):
            assert native[capability] == retained[capability]


def test_should_advertise_grok_46_xhigh_support_and_reasoning_fields() -> None:
    for catalog_path in CATALOG_PATHS:
        catalog = json.loads(catalog_path.read_text())
        for model in ("xai/grok-4.6", "oa_xai/grok-4.6"):
            entry = catalog[model]
            assert entry["supports_xhigh_reasoning_effort"] is True
            assert entry["unsupported_request_params"] == [
                "external_web_access",
                "reasoning_effort",
                "reasoningEffort",
            ]
            # Nested reasoning.effort remains allowed; only flat aliases are rejected.
            assert "reasoning" not in entry["unsupported_request_params"]
            # Reasoning history input items are rejected.
            assert entry["unsupported_input_item_types"] == ["reasoning"]


def test_should_advertise_grok_46_collaboration_namespace_tool_adapters() -> None:
    expected_collaboration = [
        "followup_task",
        "interrupt_agent",
        "list_agents",
        "send_message",
        "spawn_agent",
        "wait_agent",
    ]
    for catalog_path in CATALOG_PATHS:
        catalog = json.loads(catalog_path.read_text())
        for model in ("xai/grok-4.6", "oa_xai/grok-4.6"):
            entry = catalog[model]
            assert entry["namespace_tool_function_adapters"] == {
                "collaboration": expected_collaboration
            }


def test_should_report_grok_46_xhigh_support_through_capability_lookup() -> None:
    assert (
        supports_xhigh_reasoning_effort(
            model="xai/grok-4.6", custom_llm_provider="xai"
        )
        is True
    )
    assert (
        supports_xhigh_reasoning_effort(
            model="oa_xai/grok-4.6", custom_llm_provider="xai"
        )
        is True
    )


def test_should_register_native_and_managed_grok_46_descriptors() -> None:
    managed = OA_XAI_ROUTE_DESCRIPTORS["oa_xai/grok-4.6"]
    native = GROK_NATIVE_ROUTE_DESCRIPTORS["grok-4.6"]

    assert managed.upstream_model == "xai/grok-4.6"
    assert managed.route_family == "xai_oauth_api"
    assert managed.credential_family == "xai_oauth"
    assert native.upstream_model == "grok-4.6"
    assert native.route_family == "grok_cli_chat_proxy"
    assert native.credential_family == "xai_grok_oidc"


def test_should_activate_grok_46_for_sota_xai() -> None:
    yaml_path = REPO_ROOT / "litellm/proxy/aawm_alias_config/sota-xai.yaml"
    yaml_text = yaml_path.read_text()

    assert "model: oa_xai/grok-4.6" in yaml_text
    assert "model: oa_xai/grok-4.5" not in yaml_text


def test_should_not_override_caller_reasoning_effort_for_sota_xai() -> None:
    yaml_path = REPO_ROOT / "litellm/proxy/aawm_alias_config/sota-xai.yaml"
    yaml_text = yaml_path.read_text()

    assert "reasoning_effort: max" not in yaml_text
    assert "reasoning_effort" not in yaml_text


@pytest.mark.parametrize(
    ("prompt_tokens", "cached_tokens", "completion_tokens", "tier", "expected"),
    (
        (100_000, 20_000, 10_000, "base", 0.23),
        (200_001, 50_001, 1_000, "above_200k_tokens", 0.662001),
    ),
)
def test_should_price_grok_46_using_whole_request_tier(
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    tier: str,
    expected: float,
) -> None:
    result = calculate_xai_reference_cost(
        model="oa_xai/grok-4.6",
        prompt_tokens=prompt_tokens,
        cache_read_input_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )

    assert result["reference_cost_tier"] == tier
    assert result["reference_cost_total_usd"] == pytest.approx(expected)


def test_should_price_grok_46_from_the_exact_catalog_entry() -> None:
    baseline = calculate_xai_reference_cost(
        model="oa_xai/grok-4.6",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )
    litellm.model_cost["oa_xai/grok-4.6"]["input_cost_per_token"] = 3e-6
    litellm.get_model_info.cache_clear()
    changed = calculate_xai_reference_cost(
        model="oa_xai/grok-4.6",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )

    assert baseline["reference_cost_total_usd"] == pytest.approx(0.23)
    assert changed["reference_cost_total_usd"] == pytest.approx(0.31)


def test_should_record_grok_46_reference_provenance_without_invoice_cost() -> None:
    metadata = build_xai_grok_46_reference_cost_metadata(
        provider="xai",
        model="oa_xai/grok-4.6",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )

    assert metadata is not None
    assert metadata["actual_invoice_cost_known"] is False
    assert metadata["reference_cost_source"].endswith("/grok-4.6")
    assert metadata["reference_cost_created"] == 1785974400
    assert metadata["reference_cost_verified"] == "2026-08-12"
    assert metadata["reference_cost_input_usd_per_million"] == 2.0
    assert metadata["reference_cost_cache_read_input_usd_per_million"] == 0.5
    assert metadata["reference_cost_output_usd_per_million"] == 6.0
    assert metadata["reference_cost_image_input_usd_per_million"] == 2.0
    assert "response_cost" not in metadata
    assert "response_cost_usd" not in metadata


def test_should_recognize_catalog_configured_future_managed_xai_model() -> None:
    configured_model = "oa_xai/future-managed-reference-model"
    configured_entry = dict(litellm.model_cost["oa_xai/grok-4.6"])
    configured_entry.update(
        {
            "source": "https://docs.x.ai/developers/models/future-managed-reference-model",
            "created": 1785888000,
            "verified": "2026-08-11",
        }
    )
    litellm.model_cost[configured_model] = configured_entry
    litellm.get_model_info.cache_clear()

    metadata = build_xai_grok_46_reference_cost_metadata(
        provider="xai",
        model=configured_model,
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )

    assert metadata["reference_cost_model"] == configured_model
    assert metadata["reference_cost_source"].endswith(
        "/future-managed-reference-model"
    )
    assert metadata["reference_cost_total_usd"] == pytest.approx(0.23)
    assert metadata["actual_invoice_cost_known"] is False


def test_should_fail_closed_for_ineligible_or_unconfigured_xai_model() -> None:
    ineligible_model = "xai/catalog-only-reference-model"
    litellm.model_cost[ineligible_model] = dict(
        litellm.model_cost["oa_xai/grok-4.6"]
    )
    litellm.get_model_info.cache_clear()

    ineligible_metadata = build_xai_grok_46_reference_cost_metadata(
        provider="xai",
        model=ineligible_model,
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )
    unconfigured_metadata = build_xai_grok_46_reference_cost_metadata(
        provider="xai",
        model="oa_xai/missing-catalog-reference-model",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )

    assert ineligible_metadata == {}
    assert unconfigured_metadata == {}


def test_should_leave_managed_grok_46_session_history_invoice_cost_unknown() -> None:
    record = _build_session_history_record(
        kwargs={
            "litellm_call_id": "call-grok-46-reference",
            "model": "oa_xai/grok-4.6",
            "custom_llm_provider": "xai",
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "session_id": "session-grok-46-reference",
                    "requested_model_alias": "sota-xai",
                    "model_alias_label": "sota-xai",
                    "xai_oauth_public_model": "oa_xai/grok-4.6",
                    "xai_oauth_upstream_model": "xai/grok-4.6",
                }
            },
            "standard_logging_object": {"metadata": {}, "request_tags": []},
            "passthrough_logging_payload": {"request_body": {"messages": []}},
        },
        result={
            "id": "resp-grok-46-reference",
            "model": "grok-4.6",
            "usage": {
                "prompt_tokens": 100_000,
                "completion_tokens": 10_000,
                "total_tokens": 110_000,
            },
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
        start_time="2026-08-12T18:00:00Z",
        end_time="2026-08-12T18:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    assert record["provider"] == "xai"
    assert record["model"] == "oa_xai/grok-4.6"
    assert record["response_cost_usd"] is None
    persisted = record["metadata"]
    assert persisted["requested_model_alias"] == "sota-xai"
    assert persisted["model_alias_label"] == "sota-xai"
    assert persisted["xai_oauth_public_model"] == "oa_xai/grok-4.6"
    assert persisted["xai_oauth_upstream_model"] == "xai/grok-4.6"
    assert persisted["actual_invoice_cost_known"] is False
    assert persisted["reference_cost_model"] == "oa_xai/grok-4.6"
    assert persisted["reference_cost_created"] == 1785974400
    assert persisted["reference_cost_verified"] == "2026-08-12"
    assert persisted["reference_cost_owner"] == "xai"
    assert persisted["reference_cost_aliases"] == []
    assert persisted["reference_cost_threshold_tokens"] == 200_000
    assert persisted["reference_cost_context_window_tokens"] == 500_000
    assert persisted["reference_cost_input_usd_per_million"] == 2.0
    assert persisted["reference_cost_cache_read_input_usd_per_million"] == 0.5
    assert persisted["reference_cost_output_usd_per_million"] == 6.0
    assert persisted["reference_cost_image_input_usd_per_million"] == 2.0
    assert (
        persisted["reference_cost_input_usd_per_million_above_200k_tokens"]
        == 4.0
    )
    assert (
        persisted[
            "reference_cost_cache_read_input_usd_per_million_above_200k_tokens"
        ]
        == 1.0
    )
    assert (
        persisted["reference_cost_output_usd_per_million_above_200k_tokens"]
        == 12.0
    )
    assert persisted["reference_cost_tier"] == "base"
    assert persisted["reference_cost_whole_request_tokens"] == 100_000
    assert persisted["reference_cost_cached_input_tokens"] == 0
    assert persisted["reference_cost_uncached_input_tokens"] == 100_000
    assert persisted["reference_cost_output_tokens"] == 10_000
    assert persisted["reference_cost_total_usd"] == pytest.approx(0.26)

    db_metadata = json.loads(_build_session_history_db_payload(record)[52])
    for key, value in persisted.items():
        assert db_metadata[key] == value


def test_should_persist_requested_and_native_xhigh_reasoning_effort_metadata() -> None:
    effort_metadata = {
        "reasoning_effort_requested": "xhigh",
        "reasoning_effort_source": "reasoning.effort",
        "reasoning_effort_native_provider": "xai",
        "reasoning_effort_native_value": "xhigh",
        "reasoning_effort_native_field": "reasoning.effort",
        "reasoning_effort_supported_ceiling": "xhigh",
        "reasoning_effort_resolved_model": "oa_xai/grok-4.6",
        "reasoning_effort_resolved_provider": "xai",
        "reasoning_effort_candidate_attempt": 1,
        "reasoning_effort_mapping_reason": "within_supported_ceiling",
    }
    record = _build_session_history_record(
        kwargs={
            "litellm_call_id": "call-grok-46-xhigh",
            "model": "oa_xai/grok-4.6",
            "custom_llm_provider": "xai",
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "session_id": "session-grok-46-xhigh",
                    "requested_model_alias": "sota-xai",
                    "model_alias_label": "sota-xai",
                    "xai_oauth_public_model": "oa_xai/grok-4.6",
                    "xai_oauth_upstream_model": "xai/grok-4.6",
                    **effort_metadata,
                }
            },
            "standard_logging_object": {"metadata": {}, "request_tags": []},
            "passthrough_logging_payload": {
                "request_body": {"reasoning": {"effort": "xhigh"}}
            },
        },
        result={
            "id": "resp-grok-46-xhigh",
            "model": "grok-4.6",
            "usage": {
                "prompt_tokens": 100_000,
                "completion_tokens": 10_000,
                "total_tokens": 110_000,
            },
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
        start_time="2026-08-16T12:00:00Z",
        end_time="2026-08-16T12:00:01Z",
        allow_runtime_identity=False,
    )

    assert record is not None
    persisted = record["metadata"]
    for key, value in effort_metadata.items():
        assert persisted[key] == value
    assert "reasoning_effort_clamped_from" not in persisted
    assert "reasoning_effort_clamp_reason" not in persisted

    db_metadata = json.loads(_build_session_history_db_payload(record)[52])
    for key, value in effort_metadata.items():
        assert db_metadata[key] == value
