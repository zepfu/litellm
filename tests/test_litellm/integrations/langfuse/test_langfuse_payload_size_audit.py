import logging

from litellm.integrations.langfuse.langfuse import (
    _build_langfuse_payload_size_summary,
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
