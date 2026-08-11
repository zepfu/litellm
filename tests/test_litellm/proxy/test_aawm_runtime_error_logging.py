"""Focused tests for AAWM runtime error JSONL intake bounds (RR-044)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from litellm.proxy import aawm_runtime_error_logging as rel


def _composer_call_message() -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "composer_call"}],
    }


def _many_malformed_response(*, count: int) -> dict:
    return {
        "id": "resp_many",
        "status": "completed",
        "model": "test-model",
        "output": [_composer_call_message() for _ in range(count)],
    }


@pytest.fixture(autouse=True)
def _reset_runtime_error_intake_health_state() -> None:
    rel._reset_runtime_error_intake_health_for_tests()
    yield
    rel._reset_runtime_error_intake_health_for_tests()


def _configure_disabled_intake_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable both intake sinks without touching logging paths."""
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_ENABLED", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_DIR", raising=False)


def _configure_enabled_sink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sink: str,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
        monkeypatch.delenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", raising=False)
    else:
        monkeypatch.setenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", "1")
        monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", raising=False)


def _malformed_records() -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "failure_kind": rel.MALFORMED_TOOL_CALL_FAILURE_KIND,
            "error_code": rel.MALFORMED_TOOL_CALL_ERROR_CODE,
            "malformed_tool_call_index": 0,
        }
    ]


def _terminal_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "failure_kind": "agent_terminal_error",
        "error_code": "agent_terminal_error",
        "message": "terminal test",
    }


def _append_to_sink(
    sink: str,
    payload: dict[str, object] | list[dict[str, object]],
) -> rel.RuntimeErrorIntakeDisposition:
    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        records = payload if isinstance(payload, list) else [payload]
        return rel.append_malformed_tool_call_detections(records)
    return rel.append_agent_terminal_error(
        payload if isinstance(payload, dict) else payload[0]
    )


def _sink_log_path(tmp_path: Path, sink: str) -> Path:
    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        return tmp_path / "malformed-error.jsonl"
    return tmp_path / "test-error.jsonl"


def test_persist_malformed_tool_call_detection_caps_evidence_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_ITEMS", raising=False)

    assert rel.persist_malformed_tool_call_detection(
        response_body=_many_malformed_response(count=20),
        adapter_model="test-model",
        adapter="codex_auto_agent_openrouter_responses",
        adapter_label="OpenRouter",
        intake_context={"repository": "litellm", "session_id": "sess-cap"},
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "malformed-error.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == rel._DEFAULT_MAX_MALFORMED_EVIDENCE_ITEMS
    assert [record["malformed_tool_call_index"] for record in records] == list(
        range(rel._DEFAULT_MAX_MALFORMED_EVIDENCE_ITEMS)
    )
    assert {record["malformed_tool_call_count"] for record in records} == {
        rel._DEFAULT_MAX_MALFORMED_EVIDENCE_ITEMS
    }


def test_persist_malformed_tool_call_detection_respects_max_items_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_ITEMS", "3")

    assert rel.persist_malformed_tool_call_detection(
        response_body=_many_malformed_response(count=10),
        adapter_model="test-model",
        adapter="codex_auto_agent_openrouter_responses",
        adapter_label="OpenRouter",
    )

    lines = [
        line
        for line in (tmp_path / "malformed-error.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(lines) == 3


def test_max_error_log_file_bytes_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", raising=False)

    assert (
        rel._max_malformed_error_log_file_bytes()
        == rel._DEFAULT_MAX_ERROR_LOG_FILE_BYTES
    )
    assert (
        rel._max_agent_terminal_error_log_file_bytes()
        == rel._DEFAULT_MAX_ERROR_LOG_FILE_BYTES
    )


def test_append_malformed_tool_call_detection_enforces_default_size_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES", raising=False)
    monkeypatch.setattr(rel, "_DEFAULT_MAX_ERROR_LOG_FILE_BYTES", 32)

    log_path = tmp_path / "malformed-error.jsonl"
    log_path.write_text("x" * 40, encoding="utf-8")

    disposition = rel.append_malformed_tool_call_detection(
        {
            "schema_version": 1,
            "failure_kind": "malformed_tool_call",
            "error_code": "aawm_auto_agent_malformed_tool_call_text",
        }
    )
    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SATURATED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 1
    assert disposition.records_written == 0
    assert log_path.read_text(encoding="utf-8") == "x" * 40


def test_append_agent_terminal_error_enforces_default_size_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", raising=False)
    monkeypatch.setattr(rel, "_DEFAULT_MAX_ERROR_LOG_FILE_BYTES", 32)

    log_path = tmp_path / "test-error.jsonl"
    log_path.write_text("y" * 40, encoding="utf-8")

    disposition = rel.append_agent_terminal_error(
        {
            "schema_version": 1,
            "failure_kind": "agent_terminal_error",
            "error_code": "agent_terminal_error",
        }
    )
    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SATURATED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 1
    assert disposition.records_written == 0
    assert log_path.read_text(encoding="utf-8") == "y" * 40


@pytest.mark.parametrize(
    "sink",
    [
        rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL,
        rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR,
    ],
)
def test_append_disposition_disabled_for_both_sinks(
    sink: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_disabled_intake_sinks(monkeypatch)

    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        disposition = _append_to_sink(sink, _malformed_records())
        expected_attempted = 1
    else:
        disposition = _append_to_sink(sink, _terminal_record())
        expected_attempted = 1

    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_DISABLED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == expected_attempted
    assert disposition.records_written == 0

    health = rel.get_runtime_error_intake_health()
    assert not _sink_log_path(tmp_path, sink).exists()
    assert health["sinks"][sink]["status_counts"][disposition.status] == 1
    assert health["sinks"][sink]["attempted_records"] == expected_attempted
    assert health["sinks"][sink]["written_records"] == 0
    assert health["sinks"][sink]["last_disposition"] == {
        "sink": sink,
        "status": rel._RUNTIME_ERROR_INTAKE_STATUS_DISABLED,
        "reason_code": rel._RUNTIME_ERROR_INTAKE_REASON[
            rel._RUNTIME_ERROR_INTAKE_STATUS_DISABLED
        ],
        "records_attempted": expected_attempted,
        "records_written": 0,
        "retryable": rel._RUNTIME_ERROR_INTAKE_RETRYABLE[
            rel._RUNTIME_ERROR_INTAKE_STATUS_DISABLED
        ],
    }


@pytest.mark.parametrize(
    "sink",
    [
        rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL,
        rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR,
    ],
)
def test_append_disposition_empty_input_for_both_sinks(
    sink: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        disposition = _append_to_sink(sink, [])
    else:
        disposition = _append_to_sink(sink, {})

    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_EMPTY
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 0
    assert disposition.records_written == 0

    health = rel.get_runtime_error_intake_health()
    assert health["sinks"][sink]["status_counts"][disposition.status] == 1
    assert not _sink_log_path(tmp_path, sink).exists()


@pytest.mark.parametrize(
    "sink",
    [
        rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL,
        rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR,
    ],
)
def test_append_disposition_saturated_ceiling_for_both_sinks(
    sink: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_enabled_sink(monkeypatch, tmp_path, sink)
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", raising=False)
    monkeypatch.setattr(rel, "_DEFAULT_MAX_ERROR_LOG_FILE_BYTES", 32)

    log_path = _sink_log_path(tmp_path, sink)
    log_path.write_text("x" * 40, encoding="utf-8")

    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        disposition = _append_to_sink(sink, _malformed_records())
    else:
        disposition = _append_to_sink(sink, _terminal_record())

    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SATURATED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 1
    assert disposition.records_written == 0
    assert log_path.read_text(encoding="utf-8") == "x" * 40


@pytest.mark.parametrize(
    "sink",
    [
        rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL,
        rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR,
    ],
)
def test_append_disposition_serialization_failure_for_both_sinks(
    sink: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_enabled_sink(monkeypatch, tmp_path, sink)

    def _raise_json_dumps(*_args, **_kwargs):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(rel, "safe_dumps", _raise_json_dumps)

    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        disposition = _append_to_sink(sink, _malformed_records())
    else:
        disposition = _append_to_sink(sink, _terminal_record())

    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SERIALIZATION_FAILED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 1
    assert disposition.records_written == 0


@pytest.mark.parametrize(
    "sink",
    [
        rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL,
        rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR,
    ],
)
def test_append_disposition_write_failure_for_both_sinks(
    sink: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_enabled_sink(monkeypatch, tmp_path, sink)

    def _fail_metadata_normalization(_path: str) -> None:
        raise RuntimeError("metadata blocked")

    monkeypatch.setattr(
        rel,
        "_normalize_aawm_error_log_file_metadata",
        _fail_metadata_normalization,
    )

    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        expected_record = _malformed_records()[0]
        disposition = _append_to_sink(sink, [expected_record])
    else:
        expected_record = _terminal_record()
        disposition = _append_to_sink(sink, expected_record)

    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_WRITE_FAILED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 1
    assert disposition.records_written == 1
    written_records = [
        json.loads(line)
        for line in _sink_log_path(tmp_path, sink)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert written_records == [expected_record]


@pytest.mark.parametrize(
    "sink,should_write",
    [
        (rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL, True),
        (rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL, False),
        (rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR, True),
        (rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR, False),
    ],
)
def test_append_disposition_truthiness_for_both_sinks(
    sink: str,
    should_write: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if should_write:
        _configure_enabled_sink(monkeypatch, tmp_path, sink)
        if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
            disposition = _append_to_sink(sink, _malformed_records())
        else:
            disposition = _append_to_sink(sink, _terminal_record())
    else:
        _configure_disabled_intake_sinks(monkeypatch)
        if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
            disposition = _append_to_sink(sink, _malformed_records())
        else:
            disposition = _append_to_sink(sink, _terminal_record())

    assert bool(disposition) == should_write
    assert disposition.success is should_write


def test_runtime_error_intake_health_reports_sanitized_totals_and_last_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL
    terminal = rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR

    _configure_enabled_sink(monkeypatch, tmp_path, malformed)
    malformed_disposition = _append_to_sink(malformed, _malformed_records())

    _configure_enabled_sink(monkeypatch, tmp_path, terminal)
    _configure_disabled_intake_sinks(monkeypatch)
    terminal_disposition = _append_to_sink(terminal, {})

    health = rel.get_runtime_error_intake_health()
    assert health["totals"]["records_attempted"] == (
        malformed_disposition.records_attempted + terminal_disposition.records_attempted
    )
    assert health["totals"]["records_written"] == (
        malformed_disposition.records_written + terminal_disposition.records_written
    )

    malformed_health = health["sinks"][malformed]
    assert malformed_health["attempted_records"] == malformed_disposition.records_attempted
    assert malformed_health["written_records"] == malformed_disposition.records_written
    assert malformed_health["status_counts"][malformed_disposition.status] == 1
    assert set(malformed_health["status_counts"]) == set(
        rel._RUNTIME_ERROR_INTAKE_STATUS_LIST
    )
    assert malformed_health["last_disposition"] == {
        "sink": malformed,
        "status": malformed_disposition.status,
        "reason_code": malformed_disposition.reason_code,
        "records_attempted": malformed_disposition.records_attempted,
        "records_written": malformed_disposition.records_written,
        "retryable": malformed_disposition.retryable,
    }

    terminal_health = health["sinks"][terminal]
    assert terminal_health["attempted_records"] == terminal_disposition.records_attempted
    assert terminal_health["written_records"] == terminal_disposition.records_written
    assert terminal_health["status_counts"][terminal_disposition.status] == 1
    assert set(terminal_health["status_counts"]) == set(
        rel._RUNTIME_ERROR_INTAKE_STATUS_LIST
    )
    assert terminal_health["last_disposition"] == {
        "sink": terminal,
        "status": terminal_disposition.status,
        "reason_code": terminal_disposition.reason_code,
        "records_attempted": terminal_disposition.records_attempted,
        "records_written": terminal_disposition.records_written,
        "retryable": terminal_disposition.retryable,
    }

    snapshot = rel.get_runtime_error_intake_health()
    snapshot["totals"]["records_attempted"] = -1
    assert rel.get_runtime_error_intake_health()["totals"]["records_attempted"] != -1


@pytest.mark.parametrize(
    "sink",
    [
        rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL,
        rel._RUNTIME_ERROR_INTAKE_SINK_AGENT_TERMINAL_ERROR,
    ],
)
def test_runtime_error_intake_warning_is_first_and_rate_limited_by_sink_status(
    sink: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure_enabled_sink(monkeypatch, tmp_path, sink)

    def _raise_json_dumps(*_args, **_kwargs):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(rel, "safe_dumps", _raise_json_dumps)
    monotonic_values = iter([10.0, 20.0, 80.0])
    monkeypatch.setattr(rel, "monotonic", lambda: next(monotonic_values))

    if sink == rel._RUNTIME_ERROR_INTAKE_SINK_MALFORMED_TOOL_CALL:
        append_payload = _malformed_records()
    else:
        append_payload = _terminal_record()

    with caplog.at_level(logging.WARNING, logger=rel._RUNTIME_ERROR_INTAKE_LOGGER.name):
        first = _append_to_sink(sink, append_payload)
        second = _append_to_sink(sink, append_payload)
        third = _append_to_sink(sink, append_payload)

    assert first.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SERIALIZATION_FAILED
    assert second.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SERIALIZATION_FAILED
    assert third.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SERIALIZATION_FAILED
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 2
    for warning in warnings:
        lowered = warning.lower()
        assert "path" not in lowered
        assert "payload" not in lowered
        assert "exception" not in lowered


def test_append_malformed_tool_call_detections_batches_under_single_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")

    open_calls: list[str] = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        open_calls.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)

    records = [
        {
            "schema_version": 1,
            "failure_kind": "malformed_tool_call",
            "error_code": "aawm_auto_agent_malformed_tool_call_text",
            "malformed_tool_call_index": index,
        }
        for index in range(5)
    ]
    disposition = rel.append_malformed_tool_call_detections(records)
    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_WRITTEN
    assert disposition.success is True
    assert disposition
    assert disposition.records_attempted == 5
    assert disposition.records_written == 5

    log_path = str(tmp_path / "malformed-error.jsonl")
    assert open_calls.count(log_path) == 1
    written = [
        json.loads(line)
        for line in Path(log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["malformed_tool_call_index"] for row in written] == [0, 1, 2, 3, 4]


def test_persist_malformed_tool_call_detection_still_writes_without_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")

    assert rel.persist_malformed_tool_call_detection(
        response_body={
            "id": "resp_empty",
            "status": "completed",
            "model": "test-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "plain assistant text"}],
                }
            ],
        },
        adapter_model="test-model",
        adapter="codex_auto_agent_openrouter_responses",
        adapter_label="OpenRouter",
        intake_context={"session_id": "sess-empty"},
    )

    record = json.loads((tmp_path / "malformed-error.jsonl").read_text(encoding="utf-8"))
    assert record["failure_kind"] == "malformed_tool_call"
    assert record["session_id"] == "sess-empty"
    assert record.get("malformed_tool_call_evidence") is None


def test_append_malformed_tool_call_detections_rejects_batch_that_would_exceed_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES", raising=False)

    seed = {"schema_version": 1, "payload": "seed"}
    seed_line = rel._encode_jsonl_record_line(seed)
    seed_bytes = len(seed_line.encode("utf-8"))
    # Cap is exactly large enough for the seed alone; any additional record must fail.
    monkeypatch.setattr(rel, "_DEFAULT_MAX_ERROR_LOG_FILE_BYTES", seed_bytes)

    log_path = tmp_path / "malformed-error.jsonl"
    log_path.write_text(seed_line, encoding="utf-8")

    pending = {
        "schema_version": 1,
        "failure_kind": "malformed_tool_call",
        "error_code": "aawm_auto_agent_malformed_tool_call_text",
        "payload": "x" * 32,
    }
    disposition = rel.append_malformed_tool_call_detections([pending])
    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_SATURATED
    assert disposition.success is False
    assert not disposition
    assert disposition.records_attempted == 1
    assert disposition.records_written == 0
    assert log_path.read_text(encoding="utf-8") == seed_line


def test_append_malformed_tool_call_detections_allows_batch_within_projected_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.delenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES", raising=False)

    records = [
        {
            "schema_version": 1,
            "failure_kind": "malformed_tool_call",
            "error_code": "aawm_auto_agent_malformed_tool_call_text",
            "malformed_tool_call_index": index,
        }
        for index in range(2)
    ]
    pending_bytes = rel._projected_jsonl_batch_bytes(records)
    monkeypatch.setattr(rel, "_DEFAULT_MAX_ERROR_LOG_FILE_BYTES", pending_bytes)

    disposition = rel.append_malformed_tool_call_detections(records)
    assert disposition.status == rel._RUNTIME_ERROR_INTAKE_STATUS_WRITTEN
    assert disposition.success is True
    assert disposition
    assert disposition.records_attempted == 2
    assert disposition.records_written == 2
    lines = [
        line
        for line in (tmp_path / "malformed-error.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_schedule_persist_malformed_tool_call_detection_offloads_to_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")

    calls: list[dict] = []

    def _fake_persist(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(rel, "persist_malformed_tool_call_detection", _fake_persist)

    to_thread_kwargs: list[dict] = []
    real_to_thread = asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        to_thread_kwargs.append({"func": func, "args": args, "kwargs": kwargs})
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(rel.asyncio, "to_thread", tracking_to_thread)

    rel.schedule_persist_malformed_tool_call_detection(
        response_body=_many_malformed_response(count=1),
        adapter_model="test-model",
        adapter="codex_auto_agent_openrouter_responses",
        adapter_label="OpenRouter",
        intake_context={"session_id": "sess-async"},
    )

    # Yield so the created offload task can finish.
    for _ in range(50):
        if calls:
            break
        await asyncio.sleep(0.01)

    assert len(to_thread_kwargs) == 1
    assert to_thread_kwargs[0]["func"] is _fake_persist
    assert len(calls) == 1
    assert calls[0]["intake_context"]["session_id"] == "sess-async"


def test_schedule_persist_malformed_tool_call_detection_sync_path_is_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")

    calls: list[dict] = []

    def _fake_persist(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(rel, "persist_malformed_tool_call_detection", _fake_persist)

    rel.schedule_persist_malformed_tool_call_detection(
        response_body=_many_malformed_response(count=1),
        adapter_model="test-model",
        adapter="codex_auto_agent_openrouter_responses",
        adapter_label="OpenRouter",
        intake_context={"session_id": "sess-sync"},
    )

    assert len(calls) == 1
    assert calls[0]["intake_context"]["session_id"] == "sess-sync"

def test_max_malformed_evidence_items_hard_clamps_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_ITEMS", "9999")
    assert (
        rel._max_malformed_evidence_items()
        == rel._HARD_MAX_MALFORMED_EVIDENCE_ITEMS
    )


def test_extract_malformed_tool_call_evidence_hard_clamps_explicit_max_items() -> None:
    evidence = rel.extract_malformed_tool_call_evidence(
        _many_malformed_response(count=rel._HARD_MAX_MALFORMED_EVIDENCE_ITEMS + 20),
        max_items=10_000,
    )
    assert len(evidence) == rel._HARD_MAX_MALFORMED_EVIDENCE_ITEMS
