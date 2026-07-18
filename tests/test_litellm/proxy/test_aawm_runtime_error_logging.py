"""Focused tests for AAWM runtime error JSONL intake bounds (RR-044)."""

from __future__ import annotations

import asyncio
import json
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

    assert (
        rel.append_malformed_tool_call_detection(
            {
                "schema_version": 1,
                "failure_kind": "malformed_tool_call",
                "error_code": "aawm_auto_agent_malformed_tool_call_text",
            }
        )
        is False
    )
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

    assert (
        rel.append_agent_terminal_error(
            {
                "schema_version": 1,
                "failure_kind": "agent_terminal_error",
                "error_code": "agent_terminal_error",
            }
        )
        is False
    )
    assert log_path.read_text(encoding="utf-8") == "y" * 40


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
    assert rel.append_malformed_tool_call_detections(records) is True

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
    assert rel.append_malformed_tool_call_detections([pending]) is False
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

    assert rel.append_malformed_tool_call_detections(records) is True
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
