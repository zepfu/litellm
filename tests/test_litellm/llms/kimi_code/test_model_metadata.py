from litellm.llms.kimi_code.model_metadata import (
    get_k3_default_think_effort,
    get_kimi_code_model_metadata,
    is_always_thinking_eligible,
    is_k2_7_model_id,
    is_k3_model_id,
    is_managed_kimi_code_model_id,
    parse_kimi_code_models_payload,
    supports_explicit_capabilities,
    supports_k3_think_effort,
)


def _live_models_payload():
    return {
        "data": [
            {
                "id": "k3",
                "context_length": 1048576,
                "supports_reasoning": True,
                "supports_image_in": True,
                "supports_video_in": True,
                "supports_thinking_type": "only",
                "think_efforts": {
                    "support": True,
                    "valid_efforts": ["low", "high", "max"],
                    "default_effort": "high",
                },
            },
            {
                "id": "kimi-for-coding",
                "context_length": 262144,
                "supports_reasoning": True,
                "supports_image_in": True,
                "supports_video_in": True,
                "supports_thinking_type": "only",
            },
            {
                "id": "kimi-for-coding-highspeed",
                "context_length": 262144,
                "supports_reasoning": True,
                "supports_image_in": True,
                "supports_video_in": True,
                "supports_thinking_type": "only",
            },
        ]
    }


def test_should_parse_only_managed_exact_model_ids():
    payload = _live_models_payload()
    payload["data"].append({"id": "k3-preview", "context_length": 1})

    metadata_by_id = parse_kimi_code_models_payload(payload)

    assert set(metadata_by_id) == {
        "k3",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
    }
    assert is_managed_kimi_code_model_id("k3")
    assert is_managed_kimi_code_model_id("kimi-for-coding")
    assert not is_managed_kimi_code_model_id("k3-preview")
    assert not is_managed_kimi_code_model_id("kimi-for-coding-v2")
    assert is_k3_model_id("k3")
    assert not is_k3_model_id("k3-preview")
    assert is_k2_7_model_id("kimi-for-coding-highspeed")
    assert not is_k2_7_model_id("kimi-for-coding-highspeed-preview")
    assert get_kimi_code_model_metadata(payload, "k3-preview") is None


def test_should_preserve_exact_authenticated_live_metadata():
    metadata_by_id = parse_kimi_code_models_payload(_live_models_payload())

    assert metadata_by_id["k3"].context_length == 1048576
    assert metadata_by_id["kimi-for-coding"].context_length == 262144
    assert metadata_by_id["kimi-for-coding-highspeed"].context_length == 262144
    for metadata in metadata_by_id.values():
        assert metadata.supports_reasoning is True
        assert metadata.supports_image_in is True
        assert metadata.supports_video_in is True
        assert metadata.supports_thinking_type == "only"
        assert metadata.max_output_tokens is None


def test_should_leave_missing_or_malformed_context_and_output_unknown():
    payload = {
        "data": [
            {
                "id": "k3",
                "context_length": True,
                "max_output_tokens": "32768",
            },
            {
                "id": "kimi-for-coding",
                "context_length": 131072,
            },
        ]
    }

    metadata_by_id = parse_kimi_code_models_payload(payload)

    assert metadata_by_id["k3"].context_length is None
    assert metadata_by_id["k3"].max_output_tokens is None
    assert metadata_by_id["kimi-for-coding"].max_output_tokens is None


def test_should_admit_only_explicit_k3_think_efforts_and_default():
    k3_metadata = get_kimi_code_model_metadata(_live_models_payload(), "k3")

    assert supports_k3_think_effort(k3_metadata, "low")
    assert supports_k3_think_effort(k3_metadata, "high")
    assert supports_k3_think_effort(k3_metadata, "max")
    assert not supports_k3_think_effort(k3_metadata, "medium")
    assert get_k3_default_think_effort(k3_metadata) == "high"


def test_should_not_synthesize_k3_efforts_for_k2_7_models():
    payload = _live_models_payload()
    for model_id in ("kimi-for-coding", "kimi-for-coding-highspeed"):
        k2_7_metadata = get_kimi_code_model_metadata(payload, model_id)

        assert k2_7_metadata is not None
        assert k2_7_metadata.think_efforts is None
        assert not supports_k3_think_effort(k2_7_metadata, "high")
        assert get_k3_default_think_effort(k2_7_metadata) is None


def test_should_map_only_marker_to_always_thinking_eligibility():
    payload = _live_models_payload()

    assert is_always_thinking_eligible(get_kimi_code_model_metadata(payload, "kimi-for-coding"))
    assert is_always_thinking_eligible(get_kimi_code_model_metadata(payload, "kimi-for-coding-highspeed"))
    assert is_always_thinking_eligible(get_kimi_code_model_metadata(payload, "k3"))
    assert not is_always_thinking_eligible(
        get_kimi_code_model_metadata(
            {
                "data": [
                    {
                        "id": "k3",
                        "supports_thinking_type": "always",
                    }
                ]
            },
            "k3",
        )
    )


def test_should_fail_closed_for_missing_false_or_unknown_capabilities():
    payload = _live_models_payload()
    k3_metadata = get_kimi_code_model_metadata(payload, "k3")
    highspeed_metadata = get_kimi_code_model_metadata(payload, "kimi-for-coding-highspeed")
    missing_capability_metadata = get_kimi_code_model_metadata(
        {"data": [{"id": "k3", "supports_reasoning": True}]}, "k3"
    )

    assert supports_explicit_capabilities(k3_metadata, ["supports_reasoning"])
    assert supports_explicit_capabilities(k3_metadata, ["supports_reasoning", "supports_image_in"])
    assert supports_explicit_capabilities(
        highspeed_metadata,
        ["supports_reasoning", "supports_image_in", "supports_video_in"],
    )
    assert not supports_explicit_capabilities(missing_capability_metadata, ["supports_image_in"])
    assert not supports_explicit_capabilities(k3_metadata, ["supports_chat"])
    assert not supports_explicit_capabilities(k3_metadata, ["supports_tools"])
    assert not supports_explicit_capabilities(k3_metadata, ["supports_streaming"])


def test_should_ignore_unknown_fields_and_keep_each_model_independent():
    payload = _live_models_payload()
    payload["data"][0]["unknown_field"] = {"nested": "ignored"}
    payload["data"][1]["think_efforts"] = {"support": "not-a-boolean"}

    metadata_by_id = parse_kimi_code_models_payload(payload)
    k3_metadata = metadata_by_id["k3"]
    k2_7_metadata = metadata_by_id["kimi-for-coding"]

    assert not hasattr(k3_metadata, "unknown_field")
    assert k3_metadata.max_output_tokens is None
    assert k2_7_metadata.max_output_tokens is None
    assert k2_7_metadata.think_efforts is None
    assert is_always_thinking_eligible(k3_metadata)
    assert is_always_thinking_eligible(k2_7_metadata)
