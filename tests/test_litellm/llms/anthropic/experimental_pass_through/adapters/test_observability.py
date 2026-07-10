from unittest.mock import patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
    normalize_reasoning_effort_for_provider,
)


@pytest.mark.parametrize(
    (
        "supports_max",
        "supports_xhigh",
        "expected_effort",
        "expected_clamped_from",
        "expected_reason",
    ),
    [
        (True, True, "max", None, None),
        (
            False,
            True,
            "xhigh",
            "max",
            "openai_max_effort_clamped_to_xhigh",
        ),
        (
            False,
            False,
            "high",
            "max",
            "openai_max_effort_clamped_to_high",
        ),
    ],
)
def test_openai_max_reasoning_respects_model_capability_ceiling(
    supports_max,
    supports_xhigh,
    expected_effort,
    expected_clamped_from,
    expected_reason,
):
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.observability.supports_max_reasoning_effort",
        return_value=supports_max,
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.observability.supports_xhigh_reasoning_effort",
        return_value=supports_xhigh,
    ):
        normalized = normalize_reasoning_effort_for_provider(
            reasoning_effort="max",
            model="resolved-model",
            custom_llm_provider="openai",
            native_provider="openai",
            native_field="reasoning.effort",
        )

    assert normalized is not None
    assert normalized.native_value == expected_effort
    assert normalized.clamped_from == expected_clamped_from
    assert normalized.clamp_reason == expected_reason
    assert normalized.metadata()["reasoning_effort_native_value"] == expected_effort
    assert (
        "reasoning-effort-clamped" in normalized.tags()
    ) is (expected_clamped_from is not None)


def test_openai_xhigh_reasoning_is_not_marked_clamped_when_supported():
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.adapters.observability.supports_xhigh_reasoning_effort",
        return_value=True,
    ):
        normalized = normalize_reasoning_effort_for_provider(
            reasoning_effort="xhigh",
            model="resolved-model",
            custom_llm_provider="openai",
            native_provider="openai",
            native_field="reasoning.effort",
        )

    assert normalized is not None
    assert normalized.native_value == "xhigh"
    assert normalized.clamped_from is None
    assert normalized.clamp_reason is None
