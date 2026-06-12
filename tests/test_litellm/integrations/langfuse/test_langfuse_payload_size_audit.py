import logging
from copy import deepcopy

from litellm.integrations.langfuse.langfuse import (
    _build_langfuse_payload_size_summary,
    _fit_langfuse_generation_params_to_event_size,
    _json_size_bytes,
    _langfuse_event_fit_target_bytes,
    _log_langfuse_payload_size_if_needed,
)


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


def test_langfuse_payload_size_warning_is_sanitized(caplog) -> None:
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


def test_langfuse_payload_size_warning_includes_truncation_metadata(caplog) -> None:
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
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        _log_langfuse_payload_size_if_needed(
            fitted_params,
            trace_id="trace-log-fit",
            call_type="completion",
            input_truncation_summary=truncation_summary,
        )

    logged_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "Langfuse event near/exceeds size limit before SDK enqueue" in logged_text
    assert "trace-log-fit" in logged_text
    assert "generation-log-fit" in logged_text
    assert "original_input_size_bytes" in logged_text
    assert "final_input_size_bytes" in logged_text
    assert "truncated_input_bytes" in logged_text
    assert "omitted_input_count" in logged_text
    assert "raw input should not appear" not in logged_text
    assert "raw output should not appear" not in logged_text
    assert "raw metadata value should not appear" not in logged_text
