"""
Tests for get_litellm_params and related helpers.

Ensures backward compatibility after sparse kwargs extraction optimization.
"""

from copy import deepcopy

import pytest

from litellm.litellm_core_utils.get_litellm_params import (
    _OPTIONAL_KWARGS_KEYS,
    _get_base_model_from_litellm_call_metadata,
    get_litellm_params,
)


class TestGetBaseModelFromLitellmCallMetadata:
    def test_none_metadata_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata(None) is None

    def test_empty_metadata_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({}) is None

    def test_missing_model_info_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"foo": "bar"}) is None

    def test_model_info_none_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"model_info": None}) is None

    def test_model_info_empty_dict_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"model_info": {}}) is None

    def test_returns_base_model(self):
        result = _get_base_model_from_litellm_call_metadata(
            {"model_info": {"base_model": "gpt-4"}}
        )
        assert result == "gpt-4"


class TestGetLitellmParamsKwargsExtraction:
    """Verify that optional kwargs are correctly extracted via sparse extraction."""

    def test_no_kwargs_omits_optional_keys(self):
        """When no kwargs passed, optional keys should not be in result."""
        result = get_litellm_params(api_key="test-key")
        for key in _OPTIONAL_KWARGS_KEYS:
            assert key not in result

    def test_present_kwargs_are_extracted(self):
        result = get_litellm_params(
            aws_region_name="us-east-1",
            timeout=30,
            rpm=100,
        )
        assert result["aws_region_name"] == "us-east-1"
        assert result["timeout"] == 30
        assert result["rpm"] == 100

    def test_subset_of_kwargs_only_includes_provided(self):
        """Only provided kwargs appear, others remain absent."""
        result = get_litellm_params(azure_ad_token="token123")
        assert result["azure_ad_token"] == "token123"
        assert "aws_region_name" not in result
        assert "timeout" not in result

    def test_unknown_kwargs_are_ignored(self):
        result = get_litellm_params(some_random_kwarg="value")
        assert "some_random_kwarg" not in result

    def test_all_optional_kwargs_extractable(self):
        """Every key in _OPTIONAL_KWARGS_KEYS can be extracted."""
        kwargs = {key: f"val_{key}" for key in _OPTIONAL_KWARGS_KEYS}
        result = get_litellm_params(**kwargs)
        for key in _OPTIONAL_KWARGS_KEYS:
            assert result[key] == f"val_{key}"


class TestGetLitellmParamsBaseModel:
    """Verify base_model resolution precedence."""

    def test_explicit_base_model_takes_precedence(self):
        result = get_litellm_params(
            base_model="explicit",
            metadata={"model_info": {"base_model": "from-metadata"}},
        )
        assert result["base_model"] == "explicit"

    def test_falls_back_to_metadata(self):
        result = get_litellm_params(
            metadata={"model_info": {"base_model": "from-metadata"}}
        )
        assert result["base_model"] == "from-metadata"

    def test_none_when_no_source(self):
        result = get_litellm_params()
        assert result["base_model"] is None


class TestGetLitellmParamsExplicitFields:
    """Verify explicit parameters are always present in the result."""

    def test_explicit_params_always_present(self):
        result = get_litellm_params()
        # Spot-check a few explicit keys that should always be in the dict
        expected_keys = [
            "acompletion",
            "api_key",
            "force_timeout",
            "verbose",
            "custom_llm_provider",
            "api_base",
            "metadata",
            "model_info",
            "max_retries",
            "ssl_verify",
            "api_version",
        ]
        for key in expected_keys:
            assert key in result

    def test_no_log_from_kwargs(self):
        """no-log can come via **kwargs as well as the explicit param."""
        result = get_litellm_params(**{"no-log": True})
        assert result["no-log"] is True

    def test_no_log_from_explicit_param(self):
        result = get_litellm_params(no_log=True)
        assert result["no-log"] is True


class TestGetLitellmParamsTraceAndSessionSeparation:
    def test_metadata_session_id_does_not_backfill_trace_id(self):
        result = get_litellm_params(metadata={"session_id": "session-123"})
        assert result["litellm_session_id"] == "session-123"
        assert result["litellm_trace_id"] is None

    def test_metadata_trace_id_does_not_backfill_session_id(self):
        result = get_litellm_params(metadata={"trace_id": "trace-123"})
        assert result["litellm_trace_id"] == "trace-123"
        assert result["litellm_session_id"] is None

    def test_explicit_trace_and_session_ids_are_preserved(self):
        result = get_litellm_params(
            litellm_trace_id="trace-abc",
            litellm_session_id="session-xyz",
            metadata={
                "trace_id": "metadata-trace",
                "session_id": "metadata-session",
            },
        )
        assert result["litellm_trace_id"] == "trace-abc"
        assert result["litellm_session_id"] == "session-xyz"


class TestGetLitellmParamsMetadataMerge:
    """RR-017: soft-merge litellm_metadata into metadata for callbacks."""

    def test_litellm_metadata_only_becomes_metadata(self):
        lm = {"user_api_key_hash": "h1", "model_info": {"id": "d1"}}
        result = get_litellm_params(litellm_metadata=lm)
        assert result["metadata"] == lm
        assert result["litellm_metadata"] == lm

    def test_soft_merge_does_not_overwrite_existing_metadata_keys(self):
        result = get_litellm_params(
            metadata={"user_api_key": "sk-real", "keep": 1},
            litellm_metadata={
                "user_api_key": "sk-other",
                "user_api_key_hash": "hash-lm",
            },
        )
        assert result["metadata"]["user_api_key"] == "sk-real"
        assert result["metadata"]["keep"] == 1
        assert result["metadata"]["user_api_key_hash"] == "hash-lm"

    def test_does_not_mutate_caller_metadata_dict(self):
        original = {"user_api_key": "sk-real"}
        get_litellm_params(
            metadata=original,
            litellm_metadata={"user_api_key_hash": "h"},
        )
        assert original == {"user_api_key": "sk-real"}

    def test_generic_soft_merge_preserves_caller_wins_and_inputs(self):
        metadata = {
            "route_family": "caller-route",
            "custom_field": "caller-value",
            "tags": ["caller-tag", "shared-tag"],
        }
        litellm_metadata = {
            "route_family": "internal-route",
            "custom_field": "internal-value",
            "tags": ["internal-tag", "shared-tag"],
        }
        metadata_before = deepcopy(metadata)
        litellm_metadata_before = deepcopy(litellm_metadata)

        result = get_litellm_params(
            metadata=metadata,
            litellm_metadata=litellm_metadata,
        )

        assert result["metadata"]["route_family"] == "caller-route"
        assert result["metadata"]["custom_field"] == "caller-value"
        assert result["metadata"]["tags"] == ["caller-tag", "shared-tag"]
        assert metadata == metadata_before
        assert litellm_metadata == litellm_metadata_before

    def test_xai_internal_fields_win_and_tags_are_stable_unioned(self):
        metadata = {
            "auth_mode": "caller-auth",
            "credential_family": "caller-credential",
            "passthrough_route_family": "caller-passthrough",
            "route_family": "caller-route",
            "xai_oauth_managed": False,
            "xai_oauth_public_model": "caller-public",
            "xai_oauth_upstream_model": "caller-upstream",
            "xai_quota_family": "caller-quota",
            "shared_quota_family": "caller-shared-quota",
            "grok_subscription_quota_shared": False,
            "model_group": "caller-model-group",
            "xai_responses_previous_response_id_decoded": False,
            "codex_unsupported_input_item_removed_count": 99,
            "codex_unsupported_input_item_types_removed": ["caller-reasoning"],
            "codex_unsupported_input_items_removed": [{"caller": True}],
            "custom_field": "caller-value",
            "tags": ["caller-tag", "shared-tag", "caller-tag"],
        }
        litellm_metadata = {
            "auth_mode": "oauth",
            "credential_family": "xai_oauth",
            "passthrough_route_family": "xai_oauth_api",
            "route_family": "xai_oauth_api",
            "xai_oauth_managed": True,
            "xai_oauth_public_model": "oa_xai/grok-4.3",
            "xai_oauth_upstream_model": "xai/grok-4.3",
            "xai_quota_family": "xai_grok_subscription",
            "shared_quota_family": "xai_grok_subscription",
            "grok_subscription_quota_shared": True,
            "model_group": "oa_xai/grok-4.3",
            "xai_responses_previous_response_id_decoded": True,
            "codex_unsupported_input_item_removed_count": 1,
            "codex_unsupported_input_item_types_removed": ["reasoning"],
            "codex_unsupported_input_items_removed": [
                {"type": "reasoning", "index": 0}
            ],
            "custom_field": "internal-value",
            "tags": ["internal-tag", "shared-tag", "internal-tag"],
        }
        metadata_before = deepcopy(metadata)
        litellm_metadata_before = deepcopy(litellm_metadata)

        result = get_litellm_params(
            metadata=metadata,
            litellm_metadata=litellm_metadata,
        )

        canonical = result["metadata"]
        for key in (
            "auth_mode",
            "credential_family",
            "passthrough_route_family",
            "route_family",
            "xai_oauth_managed",
            "xai_oauth_public_model",
            "xai_oauth_upstream_model",
            "xai_quota_family",
            "shared_quota_family",
            "grok_subscription_quota_shared",
            "model_group",
            "xai_responses_previous_response_id_decoded",
            "codex_unsupported_input_item_removed_count",
            "codex_unsupported_input_item_types_removed",
            "codex_unsupported_input_items_removed",
        ):
            assert canonical[key] == litellm_metadata[key]
        assert canonical["custom_field"] == "caller-value"
        assert canonical["tags"] == [
            "caller-tag",
            "shared-tag",
            "internal-tag",
        ]
        assert metadata == metadata_before
        assert litellm_metadata == litellm_metadata_before

    def test_xai_marker_alone_does_not_enable_internal_authority(self):
        metadata = {
            "auth_mode": "caller-auth",
            "route_family": "caller-route",
            "tags": ["caller-tag", "caller-tag"],
        }
        litellm_metadata = {
            "xai_oauth_managed": True,
            "auth_mode": "oauth",
            "route_family": "xai_oauth_api",
            "tags": ["internal-tag"],
        }
        metadata_before = deepcopy(metadata)
        litellm_metadata_before = deepcopy(litellm_metadata)

        result = get_litellm_params(
            metadata=metadata,
            litellm_metadata=litellm_metadata,
        )

        assert result["metadata"]["auth_mode"] == "caller-auth"
        assert result["metadata"]["route_family"] == "caller-route"
        assert result["metadata"]["tags"] == ["caller-tag", "caller-tag"]
        assert metadata == metadata_before
        assert litellm_metadata == litellm_metadata_before
