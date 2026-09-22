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
    XAI_NATIVE_RESPONSES_TOOL_HISTORY_CAPABILITY,
    _get_xai_model_capabilities,
    get_grok_native_route_descriptor,
    get_oa_xai_route_descriptor,
    has_grok_native_route_capability,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CATALOG_PATHS = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm/bundled_model_prices_and_context_window_fallback.json",
)
CURSOR_GROK_46_HIGH = "cursor_agent/cursor-grok-4.6-high"


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


def test_should_keep_grok_47_catalog_entries_in_parity() -> None:
    for catalog_path in CATALOG_PATHS:
        catalog = json.loads(catalog_path.read_text())
        native = catalog["xai/grok-4.7"]
        managed = catalog["oa_xai/grok-4.7"]
        grok_46 = catalog["xai/grok-4.6"]
        cursor_grok = catalog[CURSOR_GROK_46_HIGH]

        assert {
            key: value
            for key, value in native.items()
            if key != "provider_specific_entry"
        } == managed
        assert native["created"] == 1790035200
        assert native["owned_by"] == "xai"
        assert native["aliases"] == []
        assert native["max_input_tokens"] == 500000
        assert native["max_output_tokens"] == 500000
        assert native["max_tokens"] == 500000
        assert native["mode"] == "responses"
        assert native["source"] == "https://docs.x.ai/developers/models/grok-4.7"
        assert native["verified"] == "2026-09-22"
        assert native["input_cost_per_token"] == pytest.approx(2e-6)
        assert native["cache_read_input_token_cost"] == pytest.approx(5e-7)
        assert native["output_cost_per_token"] == pytest.approx(6e-6)
        assert native["input_cost_per_token_above_200k_tokens"] == pytest.approx(4e-6)
        assert native["cache_read_input_token_cost_above_200k_tokens"] == pytest.approx(
            1e-6
        )
        assert native["output_cost_per_token_above_200k_tokens"] == pytest.approx(
            1.2e-5
        )
        assert native["input_cost_per_image_token"] == pytest.approx(2e-6)
        assert native["supports_function_calling"] is True
        assert native["supports_reasoning"] is True
        assert native["supports_tool_choice"] is True
        assert native["supports_vision"] is True
        assert native["supports_xhigh_reasoning_effort"] is True
        assert native["unsupported_input_item_types"] == []
        assert grok_46["source"] == "https://docs.x.ai/developers/models/grok-4.6"
        assert grok_46["created"] == 1785974400
        assert grok_46["input_cost_per_token"] == pytest.approx(2e-6)
        assert cursor_grok["litellm_provider"] == "cursor_agent"
        assert cursor_grok != native
        assert cursor_grok != managed


def test_should_advertise_grok_47_xhigh_support_and_encrypted_reasoning_history() -> None:
    for catalog_path in CATALOG_PATHS:
        catalog = json.loads(catalog_path.read_text())
        for model in ("xai/grok-4.7", "oa_xai/grok-4.7"):
            entry = catalog[model]
            assert entry["supports_xhigh_reasoning_effort"] is True
            assert entry["unsupported_request_params"] == [
                "external_web_access",
                "reasoning_effort",
                "reasoningEffort",
            ]
            assert "reasoning" not in entry["unsupported_request_params"]
            assert entry["unsupported_input_item_types"] == []


def test_should_report_grok_47_xhigh_support_through_capability_lookup() -> None:
    assert (
        supports_xhigh_reasoning_effort(
            model="xai/grok-4.7", custom_llm_provider="xai"
        )
        is True
    )
    assert (
        supports_xhigh_reasoning_effort(
            model="oa_xai/grok-4.7", custom_llm_provider="xai"
        )
        is True
    )


def test_should_read_grok_47_tool_history_capability_when_live_cost_map_is_stripped() -> None:
    previous = litellm.model_cost
    live_entry = previous.get("xai/grok-4.7")
    if isinstance(live_entry, dict):
        stripped_entry = {
            key: value
            for key, value in live_entry.items()
            if key != "provider_specific_entry"
        }
    else:
        stripped_entry = {"litellm_provider": "xai", "mode": "responses"}
    litellm.model_cost = {"xai/grok-4.7": stripped_entry}
    litellm.get_model_info.cache_clear()
    try:
        capabilities = _get_xai_model_capabilities("grok-4.7")
        assert XAI_NATIVE_RESPONSES_TOOL_HISTORY_CAPABILITY in capabilities
        assert has_grok_native_route_capability(
            "grok-4.7", XAI_NATIVE_RESPONSES_TOOL_HISTORY_CAPABILITY
        )
        assert has_grok_native_route_capability(
            "xai/grok-4.7", XAI_NATIVE_RESPONSES_TOOL_HISTORY_CAPABILITY
        )
    finally:
        litellm.model_cost = previous
        litellm.get_model_info.cache_clear()


def test_should_activate_grok_47_for_sota_xai_oidc_then_oauth() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        DEFAULT_CONFIG_DIR,
        compile_directory,
    )

    yaml_path = REPO_ROOT / "litellm/proxy/aawm_alias_config/sota-xai.yaml"
    yaml_text = yaml_path.read_text()

    assert "model: xai/grok-4.7" in yaml_text
    assert "model: oa_xai/grok-4.7" in yaml_text
    assert "model: xai/grok-4.6" not in yaml_text
    assert "model: oa_xai/grok-4.6" not in yaml_text
    assert "cursor_agent/cursor-grok-4.6-high" not in yaml_text
    assert "reasoning_effort: max" not in yaml_text
    assert "reasoning_effort" not in yaml_text

    snapshot = compile_directory(DEFAULT_CONFIG_DIR)
    sota_xai = [entry.model for entry in snapshot.aliases["sota-xai"].candidates]
    provider_xai = [entry.model for entry in snapshot.aliases["provider-xai"].candidates]
    xai_basic = [entry.model for entry in snapshot.aliases["xai_basic"].candidates]
    expert_other = [entry.model for entry in snapshot.aliases["expert-other"].candidates]
    sota_cursor = [
        entry.model for entry in snapshot.aliases["sota-cursor"].candidates
    ]
    assert sota_xai == ["xai/grok-4.7", "oa_xai/grok-4.7"]
    assert provider_xai == ["xai/grok-4.7", "oa_xai/grok-4.7"]
    assert "xai/grok-4.7" in xai_basic
    assert "xai/grok-4.6" not in xai_basic
    assert "oa_xai/grok-4.6" not in xai_basic
    assert "xai/grok-4.7" in expert_other
    assert "xai/grok-4.6" not in expert_other
    assert sota_cursor == ["cursor_agent/cursor-grok-4.6-high"]


def test_should_register_native_and_managed_grok_47_descriptors() -> None:
    managed = OA_XAI_ROUTE_DESCRIPTORS["oa_xai/grok-4.7"]
    native = GROK_NATIVE_ROUTE_DESCRIPTORS["grok-4.7"]
    resolved_managed = get_oa_xai_route_descriptor("oa_xai/grok-4.7")
    resolved_native = get_grok_native_route_descriptor("xai/grok-4.7")

    assert managed.upstream_model == "xai/grok-4.7"
    assert managed.route_family == "xai_oauth_api"
    assert managed.credential_family == "xai_oauth"
    assert managed.auth_mode == "oauth"
    assert native.upstream_model == "grok-4.7"
    assert native.route_family == "grok_cli_chat_proxy"
    assert native.credential_family == "xai_grok_oidc"
    assert native.auth_mode == "grok_oidc"
    assert resolved_managed == managed
    assert resolved_native is not None
    assert resolved_native.upstream_model == "grok-4.7"
    assert resolved_native.credential_family == "xai_grok_oidc"


@pytest.mark.parametrize(
    ("prompt_tokens", "cached_tokens", "completion_tokens", "tier", "expected"),
    (
        (100_000, 20_000, 10_000, "base", 0.23),
        (200_001, 50_001, 1_000, "above_200k_tokens", 0.662001),
    ),
)
def test_should_price_grok_47_using_whole_request_tier(
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    tier: str,
    expected: float,
) -> None:
    result = calculate_xai_reference_cost(
        model="oa_xai/grok-4.7",
        prompt_tokens=prompt_tokens,
        cache_read_input_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )

    assert result["reference_cost_tier"] == tier
    assert result["reference_cost_total_usd"] == pytest.approx(expected)


def test_should_price_grok_47_from_the_exact_catalog_entry() -> None:
    baseline = calculate_xai_reference_cost(
        model="oa_xai/grok-4.7",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )
    litellm.model_cost["oa_xai/grok-4.7"]["input_cost_per_token"] = 3e-6
    litellm.get_model_info.cache_clear()
    changed = calculate_xai_reference_cost(
        model="oa_xai/grok-4.7",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )

    assert baseline["reference_cost_total_usd"] == pytest.approx(0.23)
    assert changed["reference_cost_total_usd"] == pytest.approx(0.31)


def test_should_record_grok_47_reference_provenance_without_invoice_cost() -> None:
    metadata = build_xai_grok_46_reference_cost_metadata(
        provider="xai",
        model="oa_xai/grok-4.7",
        prompt_tokens=100_000,
        cache_read_input_tokens=20_000,
        completion_tokens=10_000,
    )

    assert metadata is not None
    assert metadata["actual_invoice_cost_known"] is False
    assert metadata["reference_cost_source"].endswith("/grok-4.7")
    assert metadata["reference_cost_created"] == 1790035200
    assert metadata["reference_cost_verified"] == "2026-09-22"
    assert metadata["reference_cost_input_usd_per_million"] == 2.0
    assert metadata["reference_cost_cache_read_input_usd_per_million"] == 0.5
    assert metadata["reference_cost_output_usd_per_million"] == 6.0
    assert metadata["reference_cost_image_input_usd_per_million"] == 2.0
    assert "response_cost" not in metadata
    assert "response_cost_usd" not in metadata
