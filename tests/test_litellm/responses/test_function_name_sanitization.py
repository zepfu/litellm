from types import MappingProxyType
from unittest.mock import patch

import pytest

from litellm.responses.function_name_sanitization import (
    ALGORITHM_VERSION,
    DEFAULT_MAX_FUNCTION_NAME_LENGTH,
    ResponsesFunctionNameRewrite,
    _build_sanitized_name_candidate,
    restore_function_names_in_responses_body,
    restore_function_names_in_responses_output,
    sanitize_responses_function_names,
)


def _request_body(name: str) -> dict:
    return {
        "input": [
            {
                "type": "function_call",
                "name": name,
                "call_id": "call_original",
                "arguments": "{}",
            }
        ],
        "tools": [{"type": "function", "name": name, "parameters": {}}],
        "tool_choice": {"type": "function", "name": name},
    }


def test_sanitizes_one_name_consistently_across_all_surfaces() -> None:
    original = "tool_" + ("x" * 138)
    result = sanitize_responses_function_names(_request_body(original))

    upstream = result.original_to_upstream[original]
    assert len(upstream) <= DEFAULT_MAX_FUNCTION_NAME_LENGTH
    assert result.body["input"][0]["name"] == upstream
    assert result.body["tools"][0]["name"] == upstream
    assert result.body["tool_choice"]["name"] == upstream
    assert result.body["input"][0]["call_id"] == "call_original"
    assert result.diagnostics.rewritten_occurrence_count == 3
    assert result.diagnostics.affected_surfaces == (
        "input",
        "tools",
        "tool_choice",
    )


def test_short_names_are_unchanged_and_return_identity_body() -> None:
    body = _request_body("get_weather")
    result = sanitize_responses_function_names(body)

    assert result.body is body
    assert result.original_to_upstream == {}
    assert result.upstream_to_original == {}
    assert result.changed is False


def test_same_prefix_long_names_remain_distinct_and_deterministic() -> None:
    prefix = "shared_prefix_" + ("p" * 60)
    first = prefix + "_first"
    second = prefix + "_second"
    body = {
        "tools": [
            {"type": "function", "name": second},
            {"type": "function", "name": first},
        ]
    }

    first_result = sanitize_responses_function_names(body)
    second_result = sanitize_responses_function_names(body)

    assert first_result.original_to_upstream == second_result.original_to_upstream
    assert (
        first_result.original_to_upstream[first]
        != first_result.original_to_upstream[second]
    )


def test_same_tool_set_in_different_order_yields_same_mapping() -> None:
    """Mapping must not depend on tools[] dict/list order — only the name set."""
    names = [
        "alpha_" + ("a" * 100),
        "beta_" + ("b" * 100),
        "gamma_" + ("c" * 100),
        "short_ok",
    ]
    body_forward = {
        "tools": [{"type": "function", "name": name} for name in names],
    }
    body_reversed = {
        "tools": [{"type": "function", "name": name} for name in reversed(names)],
    }
    body_scrambled = {
        "tools": [
            {"type": "function", "name": names[2]},
            {"type": "function", "name": names[0]},
            {"type": "function", "name": names[3]},
            {"type": "function", "name": names[1]},
        ],
    }

    forward = sanitize_responses_function_names(body_forward)
    reversed_result = sanitize_responses_function_names(body_reversed)
    scrambled = sanitize_responses_function_names(body_scrambled)

    assert forward.original_to_upstream == reversed_result.original_to_upstream
    assert forward.original_to_upstream == scrambled.original_to_upstream
    # Prefer nonce=0 derived from the original alone when free.
    for long_name in names[:3]:
        preferred = _build_sanitized_name_candidate(
            long_name,
            DEFAULT_MAX_FUNCTION_NAME_LENGTH,
            nonce=0,
        )
        assert forward.original_to_upstream[long_name] == preferred


def test_colliding_long_names_get_stable_nonces() -> None:
    """When preferred candidates collide, sorted-original order assigns nonces."""
    # Lexicographic order: a_... before z_... so a wins nonce=0, z takes nonce=1.
    earlier = "a_colliding_long_name_" + ("e" * 80)
    later = "z_colliding_long_name_" + ("l" * 80)
    shared_candidate = "shared_forced_candidate_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    assert len(shared_candidate) <= DEFAULT_MAX_FUNCTION_NAME_LENGTH

    def forced_candidate(
        original: str,
        max_length: int,
        *,
        nonce: int = 0,
    ) -> str:
        if original in {earlier, later} and nonce == 0:
            return shared_candidate
        return _build_sanitized_name_candidate(
            original,
            max_length,
            nonce=nonce,
        )

    body_later_first = {
        "tools": [
            {"type": "function", "name": later},
            {"type": "function", "name": earlier},
        ]
    }
    body_earlier_first = {
        "tools": [
            {"type": "function", "name": earlier},
            {"type": "function", "name": later},
        ]
    }

    with patch(
        "litellm.responses.function_name_sanitization."
        "_build_sanitized_name_candidate",
        side_effect=forced_candidate,
    ):
        later_first = sanitize_responses_function_names(body_later_first)
        earlier_first = sanitize_responses_function_names(body_earlier_first)
        alone_earlier = sanitize_responses_function_names(
            {"tools": [{"type": "function", "name": earlier}]}
        )
        alone_later = sanitize_responses_function_names(
            {"tools": [{"type": "function", "name": later}]}
        )

    assert later_first.original_to_upstream == earlier_first.original_to_upstream
    assert later_first.original_to_upstream[earlier] == shared_candidate
    expected_later = _build_sanitized_name_candidate(
        later,
        DEFAULT_MAX_FUNCTION_NAME_LENGTH,
        nonce=1,
    )
    assert later_first.original_to_upstream[later] == expected_later
    assert later_first.diagnostics.collision_fallback_used is True

    # Alone, each name still prefers nonce=0 when free (cross-request stability
    # for the uncontested preferred candidate).
    assert alone_earlier.original_to_upstream[earlier] == shared_candidate
    assert alone_later.original_to_upstream[later] == shared_candidate


def test_valid_short_name_is_reserved_when_generated_candidate_collides() -> None:
    short_name = "reserved_short_name"
    long_name = "reserved_short_name_" + ("x" * 80)
    body = {
        "tools": [
            {"type": "function", "name": short_name},
            {"type": "function", "name": long_name},
        ]
    }

    def forced_candidate(
        original: str,
        max_length: int,
        *,
        nonce: int = 0,
    ) -> str:
        if original == long_name and nonce == 0:
            return short_name
        return _build_sanitized_name_candidate(
            original,
            max_length,
            nonce=nonce,
        )

    with patch(
        "litellm.responses.function_name_sanitization."
        "_build_sanitized_name_candidate",
        side_effect=forced_candidate,
    ):
        result = sanitize_responses_function_names(body)

    assert short_name not in result.original_to_upstream
    assert result.body["tools"][0]["name"] == short_name
    assert result.body["tools"][1]["name"] != short_name
    assert result.diagnostics.collision_fallback_used is True


def test_ordinary_long_name_does_not_report_collision_fallback() -> None:
    result = sanitize_responses_function_names(_request_body("x" * 143))
    assert result.diagnostics.collision_fallback_used is False


def test_request_rewrite_is_limited_to_top_level_responses_surfaces() -> None:
    long_name = "nested_" + ("n" * 80)
    body = {
        "metadata": {
            "nested": {"type": "function", "name": long_name},
            "name": long_name,
        },
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "function", "name": long_name}],
            },
            {
                "type": "function_call_output",
                "name": long_name,
                "call_id": "call_1",
                "output": "done",
            },
        ],
        "tools": [
            {"type": "mcp", "name": long_name},
            {"type": "custom", "name": long_name},
            {"type": "function", "name": long_name},
        ],
    }

    result = sanitize_responses_function_names(body)

    assert result.body["metadata"] is body["metadata"]
    assert result.body["input"] is body["input"]
    assert result.body["tools"][0]["name"] == long_name
    assert result.body["tools"][1]["name"] == long_name
    assert result.body["tools"][2]["name"] != long_name


def test_copy_on_write_preserves_unmodified_substructures() -> None:
    metadata = {"trace_id": "trace-1"}
    body = {
        "metadata": metadata,
        "tools": [{"type": "function", "name": "x" * 143}],
    }

    result = sanitize_responses_function_names(body)

    assert result.body is not body
    assert result.body["metadata"] is metadata
    assert result.body["tools"] is not body["tools"]
    assert result.body["tools"][0] is not body["tools"][0]


def test_result_mappings_are_immutable_and_request_local() -> None:
    first_name = "first_" + ("a" * 80)
    second_name = "second_" + ("b" * 80)
    first = sanitize_responses_function_names(_request_body(first_name))
    second = sanitize_responses_function_names(_request_body(second_name))

    assert isinstance(first, ResponsesFunctionNameRewrite)
    assert isinstance(first.original_to_upstream, MappingProxyType)
    assert set(first.original_to_upstream) == {first_name}
    assert set(second.original_to_upstream) == {second_name}
    with pytest.raises(TypeError):
        first.original_to_upstream[first_name] = "changed"  # type: ignore[index]


def test_diagnostics_are_bounded_and_contain_no_names() -> None:
    original = "sensitive_generated_name_" + ("z" * 80)
    result = sanitize_responses_function_names(_request_body(original))
    diagnostics = result.diagnostics

    assert diagnostics.algorithm_version == ALGORITHM_VERSION
    assert diagnostics.max_length == DEFAULT_MAX_FUNCTION_NAME_LENGTH
    assert diagnostics.distinct_rewritten_count == 1
    assert original not in repr(diagnostics)
    assert result.original_to_upstream[original] not in repr(diagnostics)


def test_invalid_max_length_fails_before_rewriting() -> None:
    with pytest.raises(ValueError, match="max_length must be at least"):
        sanitize_responses_function_names(_request_body("x" * 143), max_length=8)


def test_restores_names_in_output_and_terminal_response_payloads() -> None:
    original = "restore_" + ("r" * 80)
    rewrite = sanitize_responses_function_names(_request_body(original))
    upstream = rewrite.original_to_upstream[original]
    response = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": upstream,
                    "call_id": "call_1",
                    "arguments": "{}",
                }
            ],
        },
    }

    restored = restore_function_names_in_responses_body(
        response,
        rewrite.upstream_to_original,
    )

    assert restored["response"]["output"][0]["name"] == original
    assert restored["response"]["output"][0]["call_id"] == "call_1"


def test_restores_only_exact_function_call_names() -> None:
    mapping = MappingProxyType({"upstream_name": "original_name"})
    value = {
        "output": [
            {"type": "function_call", "name": "upstream_name"},
            {"type": "function_call", "name": "unknown_name"},
            {"type": "function", "name": "upstream_name"},
            {"type": "message", "content": "upstream_name"},
        ]
    }

    restored = restore_function_names_in_responses_body(value, mapping)

    assert restored["output"][0]["name"] == "original_name"
    assert restored["output"][1]["name"] == "unknown_name"
    assert restored["output"][2]["name"] == "upstream_name"
    assert restored["output"][3]["content"] == "upstream_name"


def test_output_restoration_with_empty_mapping_preserves_identity() -> None:
    output = [{"type": "function_call", "name": "unchanged"}]
    assert restore_function_names_in_responses_output(output, {}) is output
