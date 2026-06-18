import json
import logging
from copy import deepcopy

from litellm.integrations.langfuse.langfuse import (
    _LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV,
    _LANGFUSE_INPUT_SUMMARY_TYPE,
    _build_langfuse_input_shape_hash_summary,
    _build_langfuse_payload_size_summary,
    _fit_langfuse_generation_params_to_event_size,
    _json_size_bytes,
    _langfuse_event_fit_target_bytes,
    _log_langfuse_payload_size_if_needed,
    _strip_langfuse_generation_metadata,
)


def _build_candidate_heavy_langfuse_metadata() -> dict:
    return {
        "prompt_overhead_component_paths": {
            "system": [f"input.instructions.{index}" for index in range(30)],
            "tools": [f"tools.{index}.function.parameters" for index in range(60)],
            "conversation": [f"input.messages.{index}.content" for index in range(40)],
        },
        "prompt_overhead_excluded_component_paths": [
            f"metadata.hidden_state.{index}.payload" for index in range(80)
        ],
        "codex_response_headers": {
            "source": "codex_response_headers",
            "x-codex-active-limit": "codex",
            "x-codex-bengalfox-limit-name": "gpt-5.5",
            "x-request-id": "req-sensitive-id",
            **{f"x-extra-header-{index}": "value" * 20 for index in range(40)},
        },
        "responses_stream_tool_state": [
            {
                "type": "function_call",
                "name": "Bash",
                "call_id": f"call-{index}",
                "arguments": "secret argument text " * 100,
            }
            for index in range(20)
        ],
        "claude_tool_advertisement_compaction_events": [
            {
                "tool_name": "Bash",
                "status": "compacted",
                "cc_version": "1.2.3",
                "original_chars": 3000,
                "compacted_chars": 1000,
                "saved_chars": 2000,
                "details": "audit detail " * 200,
            }
            for _ in range(10)
        ],
        "tags": ["aawm", "codex", "langfuse-size-guardrail"],
        "responses_stream_event_counts": {
            "response.created": 1,
            "response.output_item.added": 20,
            "response.completed": 1,
        },
        "responses_stream_event_types": [
            "response.created",
            "response.output_item.added",
            "response.completed",
        ],
    }


def _build_post_d1_314_compaction_evidence(generation_params: dict) -> dict:
    original_metadata = generation_params["metadata"]
    compacted_metadata = _strip_langfuse_generation_metadata(original_metadata)
    compacted_params = {
        **generation_params,
        "metadata": compacted_metadata,
    }
    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        compacted_params,
        max_event_size_bytes=24_000,
    )
    handled_fields = (
        "prompt_overhead_component_paths",
        "prompt_overhead_excluded_component_paths",
        "codex_response_headers",
        "responses_stream_tool_state",
        "claude_tool_advertisement_compaction_events",
    )
    unchanged_fields = (
        "output",
        "tags",
        "responses_stream_event_counts",
        "responses_stream_event_types",
    )
    return {
        "already_handled": {
            "metadata": {
                "original_size_bytes": _json_size_bytes(original_metadata),
                "final_size_bytes": _json_size_bytes(compacted_metadata),
            },
            **{
                field: {
                    "original_size_bytes": _json_size_bytes(
                        original_metadata.get(field)
                    ),
                    "final_size_bytes": _json_size_bytes(
                        compacted_metadata.get(field)
                    ),
                    "summary_type": (
                        compacted_metadata.get(field, {}).get("type")
                        if isinstance(compacted_metadata.get(field), dict)
                        else None
                    ),
                }
                for field in handled_fields
            },
        },
        "remaining_candidate": {
            "input": {
                "original_size_bytes": _json_size_bytes(generation_params["input"]),
                "final_size_bytes": _json_size_bytes(fitted_params["input"]),
                "input_truncated": bool(
                    fit_summary and fit_summary.get("input_truncated")
                ),
            }
        },
        "unchanged": {
            field: {
                "original_size_bytes": _json_size_bytes(
                    (
                        generation_params.get(field)
                        if field == "output"
                        else original_metadata.get(field)
                    )
                ),
                "final_size_bytes": _json_size_bytes(
                    (
                        fitted_params.get(field)
                        if field == "output"
                        else compacted_metadata.get(field)
                    )
                ),
            }
            for field in unchanged_fields
        },
        "fit_summary": fit_summary,
    }


def test_langfuse_payload_size_summary_ignores_normal_payload() -> None:
    generation_params = {
        "id": "generation-small",
        "name": "litellm-completion",
        "model": "gpt-test",
        "input": "hello",
        "output": "world",
        "metadata": {"repository": "litellm"},
    }

    summary = _build_langfuse_payload_size_summary(
        generation_params,
        trace_id="trace-small",
        call_type="completion",
        max_event_size_bytes=10_000,
    )

    assert summary is None


def test_langfuse_payload_size_summary_reports_identifiers_and_sizes() -> None:
    generation_params = {
        "id": "generation-large",
        "name": "aawm.large",
        "model": "openrouter/example",
        "input": "input text is intentionally not logged",
        "output": "x" * 800,
        "model_parameters": {"temperature": 0},
        "metadata": {
            "repository": "litellm",
            "large_blob": "y" * 700,
            "api_key_secret": "sk-should-not-appear",
        },
    }

    summary = _build_langfuse_payload_size_summary(
        generation_params,
        trace_id="trace-large",
        call_type="completion",
        max_event_size_bytes=1_000,
    )

    assert summary is not None
    assert summary["trace_id"] == "trace-large"
    assert summary["generation_id"] == "generation-large"
    assert summary["generation_name"] == "aawm.large"
    assert summary["model"] == "openrouter/example"
    assert summary["call_type"] == "completion"
    assert summary["total_size_bytes"] >= 900
    assert summary["input_size_bytes"] > 0
    assert summary["output_size_bytes"] > 0
    assert summary["metadata_size_bytes"] > 0
    assert summary["largest_metadata_keys"][0]["key"] == "large_blob"
    assert {item["key"] for item in summary["largest_metadata_keys"]} >= {
        "large_blob",
        "<redacted-key>",
    }


def test_langfuse_metadata_candidate_compaction_reduces_candidate_size() -> None:
    metadata = _build_candidate_heavy_langfuse_metadata()

    compacted_metadata = _strip_langfuse_generation_metadata(metadata)

    assert _json_size_bytes(compacted_metadata) < _json_size_bytes(metadata) / 2
    component_summary = compacted_metadata["prompt_overhead_component_paths"]
    assert component_summary["type"] == "litellm_langfuse_metadata_compacted"
    assert component_summary["count"] == 130
    assert component_summary["bucket_counts"] == {
        "system": 30,
        "tools": 60,
        "conversation": 40,
    }
    assert len(component_summary["sample_paths"]["tools"]) == 5
    excluded_summary = compacted_metadata["prompt_overhead_excluded_component_paths"]
    assert excluded_summary["count"] == 80
    assert "hash" in excluded_summary
    header_summary = compacted_metadata["codex_response_headers"]
    assert header_summary["source"] == "codex_response_headers"
    assert header_summary["header_count"] == 43
    assert "x-codex-active-limit" in header_summary["rate_limit_header_names"]
    assert header_summary["request_id_present"] is True
    assert "req-sensitive-id" not in str(header_summary)
    tool_state_summary = compacted_metadata["responses_stream_tool_state"]
    assert tool_state_summary["tool_call_count"] == 20
    assert tool_state_summary["tool_names"] == ["Bash"]
    assert "arguments_hash" in tool_state_summary["sample_tool_calls"][0]
    assert "secret argument text" not in str(tool_state_summary)
    claude_summary = compacted_metadata["claude_tool_advertisement_compaction_events"]
    assert claude_summary["count"] == 10
    assert claude_summary["total_saved_chars"] == 20_000
    assert "audit detail" not in str(claude_summary)


def test_langfuse_post_d1_314_compaction_evidence_classifies_pressure() -> None:
    metadata = _build_candidate_heavy_langfuse_metadata()
    generation_params = {
        "id": "generation-d1-321",
        "name": "aawm.post-d1-314",
        "model": "gpt-test",
        "input": [
            {"role": "system", "content": "system shape marker"},
            *[
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"conversation item {index} " + ("x" * 320),
                }
                for index in range(80)
            ],
            {"role": "assistant", "content": "tail shape marker"},
        ],
        "output": "small output",
        "metadata": metadata,
    }

    evidence = _build_post_d1_314_compaction_evidence(generation_params)

    assert set(evidence) == {
        "already_handled",
        "remaining_candidate",
        "unchanged",
        "fit_summary",
    }
    assert set(evidence["already_handled"]) == {
        "metadata",
        "prompt_overhead_component_paths",
        "prompt_overhead_excluded_component_paths",
        "codex_response_headers",
        "responses_stream_tool_state",
        "claude_tool_advertisement_compaction_events",
    }
    assert set(evidence["remaining_candidate"]) == {"input"}
    assert set(evidence["unchanged"]) == {
        "output",
        "tags",
        "responses_stream_event_counts",
        "responses_stream_event_types",
    }

    for field, field_evidence in evidence["already_handled"].items():
        assert field_evidence["final_size_bytes"] < field_evidence[
            "original_size_bytes"
        ], field
        if field != "metadata":
            assert (
                field_evidence["summary_type"]
                == "litellm_langfuse_metadata_compacted"
            )

    # Fixture guardrail: after D1-314 metadata compaction, the only oversized
    # pressure this fixture should need to fit is the input payload.
    fit_summary = evidence["fit_summary"]
    assert fit_summary is not None
    assert fit_summary["event_fit_failed"] is False
    assert fit_summary["truncated_fields"] == ["input"]
    assert evidence["remaining_candidate"]["input"]["input_truncated"] is True
    assert evidence["remaining_candidate"]["input"]["final_size_bytes"] < evidence[
        "remaining_candidate"
    ]["input"]["original_size_bytes"]

    for field, field_evidence in evidence["unchanged"].items():
        assert field_evidence["final_size_bytes"] == field_evidence[
            "original_size_bytes"
        ], field

    compacted_text = str(_strip_langfuse_generation_metadata(metadata))
    assert "secret argument text" not in compacted_text
    assert "audit detail" not in compacted_text
    assert "req-sensitive-id" not in compacted_text


def test_langfuse_payload_size_warning_is_sanitized(caplog, monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_MAX_EVENT_SIZE_BYTES", "900000")
    generation_params = {
        "id": "generation-log",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "raw input should not appear",
        "output": "raw output should not appear" + ("x" * 900_000),
        "metadata": {
            "large_blob": "raw metadata value should not appear" + ("y" * 50_000)
        },
    }

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            generation_params,
            trace_id="trace-log",
            call_type="completion",
        )

    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Langfuse event near/exceeds size limit before SDK enqueue" in logged_text
    assert "trace-log" in logged_text
    assert "generation-log" in logged_text
    assert "gpt-test" in logged_text
    assert "large_blob" in logged_text
    assert "raw input should not appear" not in logged_text
    assert "raw output should not appear" not in logged_text
    assert "raw metadata value should not appear" not in logged_text


def test_langfuse_payload_size_near_limit_under_max_is_debug(caplog) -> None:
    generation_params = {
        "id": "generation-near-log",
        "name": "aawm.near",
        "model": "gpt-test",
        "input": "raw input should not appear",
        "output": "raw output should not appear" + ("x" * 900_000),
        "metadata": {"repository": "litellm"},
    }

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            generation_params,
            trace_id="trace-near-log",
            call_type="completion",
        )

    warning_records = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert warning_records == []
    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Langfuse event size audit below SDK limit before enqueue" in logged_text
    assert "trace-near-log" in logged_text
    assert "generation-near-log" in logged_text
    assert "raw input should not appear" not in logged_text
    assert "raw output should not appear" not in logged_text


def test_langfuse_string_input_is_truncated_without_mutating_non_input_fields() -> None:
    original_input = "start-" + ("secret prompt text " * 500) + "-end"
    generation_params = {
        "id": "generation-string-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "start_time": "2026-06-12T12:00:00Z",
        "end_time": "2026-06-12T12:00:01Z",
        "input": original_input,
        "output": {"content": "raw output survives unchanged"},
        "metadata": {"route": "aawm-code", "attempt": 1},
        "usage": {"input": 100, "output": 10, "total": 110},
        "usage_details": {"input": 100, "output": 10, "total": 110},
        "cost_details": {"total": 0.01},
        "level": "DEFAULT",
        "version": "test-version",
        "completion_start_time": "2026-06-12T12:00:00.500Z",
    }

    fitted_params, truncation_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_200,
    )

    assert truncation_summary is not None
    assert fitted_params is not generation_params
    assert generation_params["input"] == original_input
    assert fitted_params["input"] != original_input
    assert str(fitted_params["input"]).startswith("start-")
    assert str(fitted_params["input"]).endswith("-end")
    assert "litellm_langfuse_input_truncated" in str(fitted_params["input"])
    assert _json_size_bytes(fitted_params) <= _langfuse_event_fit_target_bytes(1_200)

    for key, value in generation_params.items():
        if key != "input":
            assert fitted_params[key] == value

    assert truncation_summary["input_truncated"] is True
    assert truncation_summary["original_input_size_bytes"] > truncation_summary[
        "final_input_size_bytes"
    ]
    assert truncation_summary["truncated_input_bytes"] > 0
    assert truncation_summary["omitted_input_count"] > 0


def test_langfuse_structured_input_keeps_head_tail_marker_without_mutation() -> None:
    original_input = [
        {"role": "system", "content": "head context"},
        *[
            {
                "role": "user",
                "content": f"middle message {index} " + ("x" * 240),
            }
            for index in range(12)
        ],
        {"role": "assistant", "content": "tail context"},
    ]
    original_input_snapshot = deepcopy(original_input)
    generation_params = {
        "id": "generation-structured-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": original_input,
        "output": {"content": "output survives unchanged"},
        "metadata": {"route": "aawm-code", "nested": {"keep": True}},
        "usage": {"input": 100, "output": 10, "total": 110},
        "usage_details": {"input": 100, "output": 10, "total": 110},
        "cost_details": {"total": 0.01},
        "level": "DEFAULT",
        "version": "test-version",
    }

    fitted_params, truncation_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_900,
    )

    assert truncation_summary is not None
    assert generation_params["input"] == original_input_snapshot
    assert fitted_params["input"] != original_input
    assert isinstance(fitted_params["input"], list)
    assert fitted_params["input"][0] == original_input[0]
    assert fitted_params["input"][-1] == original_input[-1]
    markers = [
        item
        for item in fitted_params["input"]
        if isinstance(item, dict)
        and item.get("type") == "litellm_langfuse_input_truncated"
    ]
    assert len(markers) == 1
    assert markers[0]["omitted_items"] > 0
    assert markers[0]["omitted_bytes_estimate"] > 0
    assert _json_size_bytes(fitted_params) <= _langfuse_event_fit_target_bytes(1_900)

    for key, value in generation_params.items():
        if key != "input":
            assert fitted_params[key] == value


def test_langfuse_payload_size_successful_fit_below_threshold_is_not_warning(
    caplog, monkeypatch
) -> None:
    monkeypatch.setenv("LANGFUSE_MAX_EVENT_SIZE_BYTES", "10_000")
    generation_params = {
        "id": "generation-log-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "raw input should not appear" + ("x" * 5_000),
        "output": "raw output should not appear",
        "metadata": {
            "large_blob": "raw metadata value should not appear",
        },
    }
    fitted_params, truncation_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_000,
    )

    assert truncation_summary is not None
    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            fitted_params,
            trace_id="trace-log-fit",
            call_type="completion",
            input_truncation_summary=truncation_summary,
        )

    warning_records = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert warning_records == []
    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Langfuse event size audit below SDK limit before enqueue" in logged_text
    assert "trace-log-fit" in logged_text
    assert "generation-log-fit" in logged_text
    assert "original_input_size_bytes" in logged_text
    assert "final_input_size_bytes" in logged_text
    assert "truncated_input_bytes" in logged_text
    assert "omitted_input_count" in logged_text
    assert "raw input should not appear" not in logged_text
    assert "raw output should not appear" not in logged_text
    assert "raw metadata value should not appear" not in logged_text


def test_langfuse_payload_size_successful_fit_still_near_limit_is_debug(
    caplog, monkeypatch
) -> None:
    monkeypatch.setenv("LANGFUSE_MAX_EVENT_SIZE_BYTES", "1_000")
    generation_params = {
        "id": "generation-near-limit-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "small input",
        "output": "raw output should not appear" + ("x" * 850),
    }
    fit_summary = {
        "event_fit_failed": False,
        "event_fit_target_bytes": 950,
        "field_truncations": [
            {
                "field": "input",
                "original_size_bytes": 5_000,
                "final_size_bytes": 20,
                "truncated_bytes": 4_980,
                "omitted_count": 1,
                "strategy": "truncated",
            }
        ],
        "truncated_fields": ["input"],
        "omitted_fields": [],
    }

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            generation_params,
            trace_id="trace-near-limit-fit",
            call_type="completion",
            input_truncation_summary=fit_summary,
        )

    warning_records = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert warning_records == []
    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Langfuse event size audit below SDK limit before enqueue" in logged_text
    assert "trace-near-limit-fit" in logged_text
    assert "generation-near-limit-fit" in logged_text
    assert "raw output should not appear" not in logged_text


def test_langfuse_oversized_output_is_fit_after_small_input() -> None:
    original_output = "raw output should not appear in warning " + ("x" * 5_000)
    generation_params = {
        "id": "generation-output-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "small input",
        "output": original_output,
        "metadata": {"route": "aawm-code"},
    }

    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_000,
    )

    assert fit_summary is not None
    assert fitted_params["input"] == "small input"
    assert fitted_params["output"] != original_output
    assert _json_size_bytes(fitted_params) <= _langfuse_event_fit_target_bytes(1_000)
    assert fit_summary["truncated_fields"] == ["output"]
    assert fit_summary["field_truncations"][0]["field"] == "output"
    assert fit_summary["field_truncations"][0]["strategy"] == "truncated"
    assert fit_summary["event_fit_failed"] is False


def test_langfuse_oversized_metadata_is_fit_without_log_value_leak(caplog) -> None:
    generation_params = {
        "id": "generation-metadata-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "small input",
        "output": "small output",
        "metadata": {
            "repository": "litellm",
            "large_blob": "raw metadata value should not appear" + ("y" * 5_000),
            "api_key_secret": "secret metadata value should not appear"
            + ("z" * 2_000),
        },
    }

    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_200,
    )

    assert fit_summary is not None
    assert _json_size_bytes(fitted_params) <= _langfuse_event_fit_target_bytes(1_200)
    assert "metadata" in fit_summary["truncated_fields"]
    assert fit_summary["omitted_fields"] == ["metadata"]
    assert fitted_params["metadata"]["type"] == "litellm_langfuse_field_omitted"
    assert fit_summary["event_fit_failed"] is False

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            fitted_params,
            trace_id="trace-metadata-fit",
            call_type="pass_through_endpoint",
            input_truncation_summary=fit_summary,
        )

    warning_records = [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]
    assert warning_records == []
    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Langfuse event size audit below SDK limit before enqueue" in logged_text
    assert "metadata" in logged_text
    assert "field_truncations" in logged_text
    assert "raw metadata value should not appear" not in logged_text
    assert "secret metadata value should not appear" not in logged_text


def test_langfuse_oversized_model_parameters_are_omitted_safely() -> None:
    generation_params = {
        "id": "generation-params-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "small input",
        "output": "small output",
        "model_parameters": {
            "temperature": 0,
            "huge": "raw model parameter value should not survive" + ("p" * 6_000),
        },
        "metadata": {"repository": "litellm"},
    }

    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=900,
    )

    assert fit_summary is not None
    assert _json_size_bytes(fitted_params) <= _langfuse_event_fit_target_bytes(900)
    assert "model_parameters" in fit_summary["truncated_fields"]
    assert fitted_params["model_parameters"] != generation_params["model_parameters"]
    assert "raw model parameter value should not survive" not in str(
        fitted_params["model_parameters"]
    )


def test_langfuse_combined_oversized_event_continues_past_input() -> None:
    generation_params = {
        "id": "generation-combined-fit",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "raw input should not appear" + ("i" * 5_000),
        "output": "raw output should not appear" + ("o" * 5_000),
        "metadata": {"large_blob": "raw metadata should not appear" + ("m" * 5_000)},
    }

    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_100,
    )

    assert fit_summary is not None
    assert _json_size_bytes(fitted_params) <= _langfuse_event_fit_target_bytes(1_100)
    assert set(fit_summary["truncated_fields"]) >= {"input", "output"}
    assert fit_summary["input_truncated"] is True
    assert fit_summary["event_fit_failed"] is False


def test_langfuse_still_too_large_after_fitting_reports_fail_closed(caplog) -> None:
    generation_params = {
        "id": "generation-fail-closed-" + ("core" * 2_000),
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "small input",
        "metadata": {"route": "aawm-code"},
    }

    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=500,
    )

    assert fit_summary is not None
    assert fit_summary["event_fit_failed"] is True
    assert fit_summary["final_total_size_bytes"] > fit_summary["event_fit_target_bytes"]

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            fitted_params,
            trace_id="trace-fail-closed",
            call_type="completion",
            input_truncation_summary=fit_summary,
        )

    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "event_fit_failed" in logged_text
    assert "generation-fail-closed-core" not in logged_text
    assert "small input" not in logged_text

def _build_sensitive_langfuse_input_samples():
    return {
        "string": "start-secret prompt text " + ("x" * 1200) + " end",
        "list": [
            {
                "role": "system",
                "content": "system prompt with /home/zepfu/projects/litellm/secret.py",
            },
            *[
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": [
                        {"type": "text", "text": "user prompt secret text " + ("x" * 320)},
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "arguments": {"command": "cat /etc/passwd"},
                        },
                    ],
                }
                for index in range(24)
            ],
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "tool output secret text",
                    }
                ],
            },
        ],
        "dict": {
            "instructions": "dict prompt secret text " + ("y" * 2000),
            "messages": [
                {
                    "role": "user",
                    "content": "message secret text " + ("z" * 1500),
                }
                for _ in range(12)
            ],
            "headers": {
                "authorization": "Bearer sk-secret-token",
                "cookie": "session=secret-cookie",
                "api_key": "sk-secret-api-key",
            },
            "source": "def secret_function():\n    return 'source secret'",
            "file_content": "local file secret content",
        },
    }


def test_langfuse_input_shape_hash_mode_disabled_keeps_default_truncation(monkeypatch) -> None:
    monkeypatch.delenv(_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV, raising=False)
    samples = {
        "string": "start-" + ("secret prompt text " * 500) + "-end",
        "list": [
            {"role": "system", "content": "system marker"},
            *[
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"list item {index} " + ("x" * 320),
                }
                for index in range(24)
            ],
            {"role": "assistant", "content": "tail marker"},
        ],
        "dict": {
            "instructions": "dict prompt secret text " + ("y" * 2000),
            "messages": [
                {
                    "role": "user",
                    "content": "message secret text " + ("z" * 1500),
                }
                for _ in range(12)
            ],
        },
    }

    for container_type, original_input in samples.items():
        generation_params = {
            "id": f"generation-default-fit-{container_type}",
            "name": "aawm.large",
            "model": "gpt-test",
            "input": original_input,
            "output": {"content": "output survives unchanged"},
            "metadata": {"route": "aawm-code"},
        }
        max_event_size_bytes = {
            "string": 1_200,
            "list": 2_000,
            "dict": 1_200,
        }[container_type]

        fitted_params, truncation_summary = _fit_langfuse_generation_params_to_event_size(
            generation_params,
            max_event_size_bytes=max_event_size_bytes,
        )

        assert truncation_summary is not None
        assert truncation_summary["input_truncated"] is True
        assert truncation_summary.get("input_shape_hash_summary") is not True
        fitted_input = fitted_params["input"]
        if container_type == "string":
            assert isinstance(fitted_input, str)
            assert "litellm_langfuse_input_truncated" in fitted_input
        elif container_type == "list":
            assert isinstance(fitted_input, list)
            assert any(
                isinstance(item, dict)
                and item.get("type") == "litellm_langfuse_input_truncated"
                for item in fitted_input
            )
        else:
            assert isinstance(fitted_input, dict)
            assert fitted_input.get("type") == "litellm_langfuse_input_truncated"


def test_langfuse_input_shape_hash_mode_replaces_string_list_and_dict_inputs(monkeypatch) -> None:
    monkeypatch.setenv(_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV, "1")
    samples = _build_sensitive_langfuse_input_samples()

    for container_type, original_input in samples.items():
        generation_params = {
            "id": f"generation-shape-{container_type}",
            "name": "aawm.large",
            "model": "gpt-test",
            "input": original_input,
            "output": "small output",
            "metadata": {"route": "aawm-code"},
        }

        max_event_size_bytes = {
            "string": 700,
            "list": 2_000,
            "dict": 1_200,
        }[container_type]
        fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
            generation_params,
            max_event_size_bytes=max_event_size_bytes,
        )

        assert fit_summary is not None
        summary_input = fitted_params["input"]
        assert isinstance(summary_input, dict)
        assert summary_input["type"] == _LANGFUSE_INPUT_SUMMARY_TYPE
        assert summary_input["container_type"] == container_type
        assert summary_input["hash"]
        assert summary_input["original_size_bytes"] > summary_input["final_size_bytes"]
        assert summary_input["item_count"] > 0
        assert summary_input["omitted_items"] >= 0
        assert summary_input["raw_reconstruction"]["source"] == "not_available_by_default"
        assert fit_summary["input_shape_hash_summary"] is True
        assert fit_summary["input_truncated"] is False

        summary_text = str(summary_input)
        assert "secret prompt text" not in summary_text
        assert "user prompt secret text" not in summary_text
        assert "tool output secret text" not in summary_text
        assert "dict prompt secret text" not in summary_text
        assert "message secret text" not in summary_text
        assert "local file secret content" not in summary_text
        assert "source secret" not in summary_text
        assert "sk-secret-token" not in summary_text
        assert "secret-cookie" not in summary_text
        assert "sk-secret-api-key" not in summary_text
        assert "/home/zepfu/projects/litellm/secret.py" not in summary_text
        assert "instructions" not in summary_text
        assert "messages" not in summary_text
        assert "file_content" not in summary_text
        assert "authorization" not in summary_text
        assert "Bash" not in summary_text
        assert "call-0" not in summary_text
        assert "call_id" not in summary_text
        if container_type in {"list", "dict"}:
            assert "key_descriptor" in summary_text
            assert "key_hash" in summary_text


def test_langfuse_input_shape_hash_mode_logs_do_not_leak_raw_input(caplog, monkeypatch) -> None:
    monkeypatch.setenv(_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV, "1")
    generation_params = {
        "id": "generation-shape-log",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": "raw input should not appear" + ("x" * 5_000),
        "output": "small output",
        "metadata": {"repository": "litellm"},
    }
    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_000,
    )

    assert fit_summary is not None
    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            fitted_params,
            trace_id="trace-shape-log",
            call_type="completion",
            input_truncation_summary=fit_summary,
        )

    logged_text = '\n'.join(record.getMessage() for record in caplog.records)
    assert "Langfuse event size audit below SDK limit before enqueue" in logged_text
    assert "input_shape_hash_summary" in logged_text
    assert "raw input should not appear" not in logged_text


class _CustomReprObject:
    def __repr__(self) -> str:
        return "CUSTOM_REPR_SECRET_VALUE"


def test_langfuse_input_shape_hash_summary_omits_raw_keys_identifiers_and_repr(monkeypatch) -> None:
    monkeypatch.setenv(_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV, "1")
    original_input = {
        "instructions": "dict prompt secret text " + ("y" * 2000),
        "messages": [
            {
                "role": "user",
                "id": "message-id-secret-123",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "call_id": "call-secret-456",
                        "arguments": {"command": "cat /etc/passwd"},
                    }
                ],
            }
        ],
        "custom_object": _CustomReprObject(),
    }
    summary = _build_langfuse_input_shape_hash_summary(
        original_input,
        original_input_size_bytes=_json_size_bytes(original_input),
        metadata=None,
    )
    summary_text = str(summary)

    assert summary["type"] == _LANGFUSE_INPUT_SUMMARY_TYPE
    assert "instructions" not in summary_text
    assert "messages" not in summary_text
    assert "Bash" not in summary_text
    assert "message-id-secret-123" not in summary_text
    assert "call-secret-456" not in summary_text
    assert "CUSTOM_REPR_SECRET_VALUE" not in summary_text
    assert "key_descriptor" in summary_text
    assert "key_hash" in summary_text
    assert '"preview"' not in json.dumps(summary, default=str, sort_keys=True)

    generation_params = {
        "id": "generation-shape-privacy",
        "name": "aawm.large",
        "model": "gpt-test",
        "input": original_input,
        "output": "small output",
        "metadata": {"repository": "litellm"},
    }
    fitted_params, fit_summary = _fit_langfuse_generation_params_to_event_size(
        generation_params,
        max_event_size_bytes=1_200,
    )
    assert fit_summary is not None
    assert fit_summary["input_shape_hash_summary"] is True
    fitted_text = str(fitted_params["input"])
    assert "Bash" not in fitted_text
    assert "call-secret-456" not in fitted_text
    assert "CUSTOM_REPR_SECRET_VALUE" not in fitted_text


def test_langfuse_input_shape_hash_summary_handles_single_key_dict(monkeypatch) -> None:
    monkeypatch.setenv(_LANGFUSE_INPUT_SHAPE_HASH_ONLY_ENV, "1")
    original_input = {"one_secret_key": "one secret value"}

    summary = _build_langfuse_input_shape_hash_summary(
        original_input,
        original_input_size_bytes=_json_size_bytes(original_input),
        metadata=None,
    )

    assert summary["type"] == _LANGFUSE_INPUT_SUMMARY_TYPE
    assert summary["head"]
    assert summary["tail"]
    summary_text = str(summary)
    assert "one_secret_key" not in summary_text
    assert "one secret value" not in summary_text


def test_langfuse_input_shape_hash_reconstruction_status_variants(monkeypatch) -> None:
    samples = _build_sensitive_langfuse_input_samples()
    original_input = samples["list"]

    default_summary = _build_langfuse_input_shape_hash_summary(
        original_input,
        original_input_size_bytes=_json_size_bytes(original_input),
        metadata=None,
    )
    assert default_summary["raw_reconstruction"]["source"] == "not_available_by_default"
    assert default_summary["raw_reconstruction"]["cold_storage_object_key_present"] is False
    assert default_summary["raw_reconstruction"]["full_payload_capture_required"] is False

    cold_storage_summary = _build_langfuse_input_shape_hash_summary(
        original_input,
        original_input_size_bytes=_json_size_bytes(original_input),
        metadata={"cold_storage_object_key": "s3://bucket/object.json"},
    )
    assert cold_storage_summary["raw_reconstruction"]["source"] == "cold_storage_object_key"
    assert cold_storage_summary["raw_reconstruction"]["cold_storage_object_key_present"] is True

    monkeypatch.setenv("AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS", "1")
    full_payload_summary = _build_langfuse_input_shape_hash_summary(
        original_input,
        original_input_size_bytes=_json_size_bytes(original_input),
        metadata=None,
    )
    assert full_payload_summary["raw_reconstruction"]["source"] == "full_payload_capture_required"
    assert full_payload_summary["raw_reconstruction"]["full_payload_capture_required"] is True
