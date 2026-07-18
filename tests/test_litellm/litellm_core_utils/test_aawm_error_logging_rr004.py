"""Focused RR-004 residual coverage for litellm/_logging.py.

Covers:
1. Offloaded AAWM error-log disk I/O (bounded queue; caller thread does not hold write lock)
2. Langfuse support-string diagnostics attach independent of callbacks / JSON_LOGS
3. Content-bearing context fields excluded by default; explicit opt-in required
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from pathlib import Path
import pytest

import litellm
from litellm._logging import (
    AawmErrorLogFileHandler,
    JsonFormatter,
    LangfuseSupportStringDiagnosticFilter,
    _AAWM_ERROR_LOG_CONTENT_BEARING_CONTEXT_FIELDS,
    _AAWM_ERROR_LOG_CONTEXT_FIELDS,
    _AAWM_ERROR_LOG_DEFAULT_CONTEXT_FIELDS,
    _LANGFUSE_SUPPORT_STRING,
    _aawm_error_log_include_content_fields,
    _build_aawm_error_log_context,
    _build_aawm_error_log_record,
    _ensure_langfuse_support_string_diagnostics_attached,
    _get_aawm_error_log_active_context_fields,
    _get_loggers_to_initialize,
    _initialize_loggers_with_handler,
    clear_langfuse_enqueue_size_audits,
    clear_langfuse_support_string_coalesce_state,
)


@pytest.fixture(autouse=True)
def _reset_langfuse_state():
    clear_langfuse_enqueue_size_audits()
    clear_langfuse_support_string_coalesce_state()
    yield
    clear_langfuse_enqueue_size_audits()
    clear_langfuse_support_string_coalesce_state()


def _wait_for_file(path: Path, *, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def test_content_bearing_fields_excluded_from_default_allowlist(monkeypatch):
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", raising=False)

    assert "aawm_passthrough_request_shape_error_body_preview" in (
        _AAWM_ERROR_LOG_CONTENT_BEARING_CONTEXT_FIELDS
    )
    assert "grok_side_channel_request_body_digest_source" in (
        _AAWM_ERROR_LOG_CONTENT_BEARING_CONTEXT_FIELDS
    )
    assert "aawm_passthrough_request_shape_error_body_preview" not in (
        _AAWM_ERROR_LOG_DEFAULT_CONTEXT_FIELDS
    )
    assert "grok_side_channel_request_body_digest_source" not in (
        _AAWM_ERROR_LOG_DEFAULT_CONTEXT_FIELDS
    )
    # Full enum remains available for callers that need the complete field catalog.
    assert set(_AAWM_ERROR_LOG_CONTENT_BEARING_CONTEXT_FIELDS).issubset(
        set(_AAWM_ERROR_LOG_CONTEXT_FIELDS)
    )
    assert not _aawm_error_log_include_content_fields()
    active = _get_aawm_error_log_active_context_fields()
    assert "aawm_passthrough_request_shape_error_body_preview" not in active
    assert "grok_side_channel_request_body_digest_source" not in active


def test_content_bearing_fields_require_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", raising=False)

    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="shape failure",
        args=(),
        exc_info=None,
    )
    setattr(
        record,
        "aawm_passthrough_request_shape_error_body_preview",
        "PROMPT_SECRET_SHOULD_NOT_PERSIST",
    )
    setattr(
        record,
        "grok_side_channel_request_body_digest_source",
        "raw-body-source",
    )
    setattr(record, "failure_kind", "request_shape_deserialization_failed")
    setattr(
        record,
        "aawm_passthrough_request_shape_error_message_class",
        "model_input_deserialization_failed",
    )

    context = _build_aawm_error_log_context(record)
    assert context.get("failure_kind") == "request_shape_deserialization_failed"
    assert (
        context.get("aawm_passthrough_request_shape_error_message_class")
        == "model_input_deserialization_failed"
    )
    assert "aawm_passthrough_request_shape_error_body_preview" not in context
    assert "grok_side_channel_request_body_digest_source" not in context
    assert "PROMPT_SECRET_SHOULD_NOT_PERSIST" not in json.dumps(context)

    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", "1")
    assert _aawm_error_log_include_content_fields() is True
    opt_in_context = _build_aawm_error_log_context(record)
    assert (
        opt_in_context["aawm_passthrough_request_shape_error_body_preview"]
        == "PROMPT_SECRET_SHOULD_NOT_PERSIST"
    )
    assert (
        opt_in_context["grok_side_channel_request_body_digest_source"]
        == "raw-body-source"
    )


def test_langfuse_diagnostics_attach_without_callbacks_or_json_logs(monkeypatch):
    """Diagnostics must attach even when success_callback is empty and JSON_LOGS is off."""
    original_success = litellm.success_callback.copy()
    original_failure = litellm.failure_callback.copy()
    langfuse_logger = logging.getLogger("langfuse")
    original_filters = langfuse_logger.filters[:]

    try:
        litellm.success_callback = []
        litellm.failure_callback = []
        monkeypatch.delenv("JSON_LOGS", raising=False)

        # Drop the filter if present, then re-ensure via the public helper.
        langfuse_logger.filters = [
            f
            for f in langfuse_logger.filters
            if not isinstance(f, LangfuseSupportStringDiagnosticFilter)
        ]
        assert not any(
            isinstance(f, LangfuseSupportStringDiagnosticFilter)
            for f in langfuse_logger.filters
        )

        _ensure_langfuse_support_string_diagnostics_attached()
        assert any(
            isinstance(f, LangfuseSupportStringDiagnosticFilter)
            for f in langfuse_logger.filters
        )

        loggers = _get_loggers_to_initialize()
        assert langfuse_logger in loggers

        # Plain-formatter path: filter must enrich the LogRecord without JSON_LOGS.
        stream = io.StringIO()
        plain = logging.StreamHandler(stream)
        plain.setFormatter(logging.Formatter("%(message)s"))
        test_logger = logging.getLogger("langfuse-rr004-plain")
        test_logger.handlers = [plain]
        test_logger.filters = []
        test_logger.setLevel(logging.ERROR)
        test_logger.propagate = False
        test_logger.addFilter(LangfuseSupportStringDiagnosticFilter())

        test_logger.error(_LANGFUSE_SUPPORT_STRING)
        # Message text preserved for plain formatters.
        assert stream.getvalue().strip() == _LANGFUSE_SUPPORT_STRING

        # Record-level diagnostics applied for any downstream handler (incl. AAWM JSONL).
        record = logging.LogRecord(
            name="langfuse",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=_LANGFUSE_SUPPORT_STRING,
            args=(),
            exc_info=None,
        )
        assert LangfuseSupportStringDiagnosticFilter().filter(record) is True
        assert getattr(record, "langfuse_support_string", False) is True
        assert getattr(record, "source", None) == "langfuse_sdk"
        assert getattr(record, "callback_phase", None) == (
            "sdk_background_ingestion_upload"
        )
    finally:
        litellm.success_callback = original_success
        litellm.failure_callback = original_failure
        langfuse_logger.filters = original_filters


def test_langfuse_diagnostics_survive_initialize_loggers_without_callbacks(
    monkeypatch, tmp_path
):
    original_success = litellm.success_callback.copy()
    original_failure = litellm.failure_callback.copy()
    langfuse_logger = logging.getLogger("langfuse")
    original_handlers = langfuse_logger.handlers[:]
    original_filters = langfuse_logger.filters[:]
    original_propagate = langfuse_logger.propagate
    original_level = langfuse_logger.level
    stream = io.StringIO()
    test_handler = logging.StreamHandler(stream)
    test_handler.setFormatter(JsonFormatter())

    try:
        litellm.success_callback = []  # empty: historical attach condition fails
        litellm.failure_callback = []
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
        monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

        _initialize_loggers_with_handler(test_handler)
        assert any(
            isinstance(f, LangfuseSupportStringDiagnosticFilter)
            for f in langfuse_logger.filters
        )
        assert langfuse_logger in _get_loggers_to_initialize()

        langfuse_logger.error(_LANGFUSE_SUPPORT_STRING)
        output = stream.getvalue().strip()
        obj = json.loads(output)
        assert obj["langfuse_support_string"] is True
        assert obj["source"] == "langfuse_sdk"
    finally:
        litellm.success_callback = original_success
        litellm.failure_callback = original_failure
        langfuse_logger.handlers = original_handlers
        langfuse_logger.filters = original_filters
        langfuse_logger.propagate = original_propagate
        langfuse_logger.setLevel(original_level)


def test_aawm_error_log_emit_offloads_disk_io_from_caller_thread(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    # Disable pytest auto-flush so we can observe async worker behavior explicitly.
    handler = AawmErrorLogFileHandler()
    writer_threads: set[int] = set()
    real_write = handler._write_line

    def tracking_write(line: str) -> None:
        writer_threads.add(threading.get_ident())
        return real_write(line)

    handler._write_line = tracking_write  # type: ignore[method-assign]
    caller_thread = threading.get_ident()
    log_file = tmp_path / "rr004-error.jsonl"

    try:
        record = logging.LogRecord(
            name="LiteLLM",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="offload-probe",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        # emit must return without requiring the write lock to finish on this thread
        handler.flush()
        _wait_for_file(log_file)

        assert writer_threads, "expected background writer to perform disk I/O"
        assert caller_thread not in writer_threads

        rows = [
            json.loads(line)
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(row.get("message") == "offload-probe" for row in rows)
        assert handler._worker is not None
        assert handler._worker.is_alive() or handler._closed is False
    finally:
        handler.close()

def test_aawm_error_log_queue_drops_when_full_without_blocking(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_QUEUE_MAXSIZE", "1")
    handler = AawmErrorLogFileHandler()
    # Pause the worker before it consumes so the bounded queue can fill without
    # holding the write lock (avoids close/flush deadlocks under a stalled write).
    pause = threading.Event()
    pause.clear()
    release = threading.Event()
    original_main = handler._worker_main

    def paused_main() -> None:
        pause.set()
        release.wait(timeout=3.0)
        return original_main()

    # Restart worker with paused main.
    handler._closed = True
    try:
        handler._queue.put_nowait(object())  # wake old worker if blocked on get
    except Exception:
        pass
    if handler._worker is not None:
        handler._worker.join(timeout=0.5)
    handler._closed = False
    handler._queue = __import__("queue").Queue(maxsize=1)
    handler._worker = threading.Thread(
        target=paused_main,
        name="aawm-error-log-writer-paused",
        daemon=True,
    )
    handler._worker.start()
    assert pause.wait(timeout=1.0)

    try:
        # Fill the single queue slot while worker is paused.
        assert handler._enqueue_line("line-1\n") is True
        start = time.monotonic()
        # Next enqueue should time out / drop without hanging.
        assert handler._enqueue_line("line-2\n") is False
        elapsed = time.monotonic() - start
        assert elapsed < 1.0
        assert handler.dropped_record_count() >= 1
        assert handler._dropped_records >= 1
    finally:
        release.set()
        handler.close()



def test_aawm_error_log_drop_count_accessor_and_throttled_warning(
    monkeypatch, tmp_path, caplog
):
    """Overload must be visible via counter + at most one warning per window."""
    import logging as _logging

    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_QUEUE_MAXSIZE", "1")

    # Shrink warning interval so second window can fire if needed.
    monkeypatch.setattr(
        "litellm._logging._AAWM_ERROR_LOG_DROP_WARNING_INTERVAL_SECONDS",
        60.0,
        raising=False,
    )

    handler = AawmErrorLogFileHandler()
    pause = threading.Event()
    pause.clear()
    release = threading.Event()
    original_main = handler._worker_main

    def paused_main() -> None:
        pause.set()
        release.wait(timeout=3.0)
        return original_main()

    handler._closed = True
    try:
        handler._queue.put_nowait(object())
    except Exception:
        pass
    if handler._worker is not None:
        handler._worker.join(timeout=0.5)
    handler._closed = False
    handler._queue = __import__("queue").Queue(maxsize=1)
    handler._worker = threading.Thread(
        target=paused_main,
        name="aawm-error-log-writer-paused-obs",
        daemon=True,
    )
    handler._worker.start()
    assert pause.wait(timeout=1.0)

    try:
        with caplog.at_level(_logging.WARNING, logger="LiteLLM"):
            assert handler._enqueue_line("keep\n") is True
            assert handler._enqueue_line("drop-1\n") is False
            assert handler._enqueue_line("drop-2\n") is False
        assert handler.dropped_record_count() >= 2
        drop_warnings = [
            r
            for r in caplog.records
            if "AAWM error-log write queue full" in r.getMessage()
        ]
        # Throttled: first drop may warn; subsequent drops in same window must not spam.
        assert 1 <= len(drop_warnings) <= 1
        assert "total_dropped=" in drop_warnings[0].getMessage()
    finally:
        release.set()
        handler.close()


def test_aawm_error_log_close_is_shutdown_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    handler = AawmErrorLogFileHandler()
    log_file = tmp_path / "rr004-error.jsonl"
    try:
        handler.emit(
            logging.LogRecord(
                name="LiteLLM",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="pre-close",
                args=(),
                exc_info=None,
            )
        )
        handler.close()
        # Second close must not raise.
        handler.close()
        # Post-close emit is a no-op / non-raising.
        handler.emit(
            logging.LogRecord(
                name="LiteLLM",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="post-close",
                args=(),
                exc_info=None,
            )
        )
        if log_file.exists():
            text = log_file.read_text(encoding="utf-8")
            assert "pre-close" in text
            assert "post-close" not in text
    finally:
        try:
            handler.close()
        except Exception:
            pass


def test_aawm_error_log_rotation_still_works_with_offload(monkeypatch, tmp_path):
    """Preserve already-landed size rotation under the offloaded writer."""
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", "200")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT", "3")

    handler = AawmErrorLogFileHandler()
    log_file = tmp_path / "rr004-error.jsonl"
    backup_1 = Path(f"{log_file}.1")
    try:
        for i in range(8):
            handler.emit(
                logging.LogRecord(
                    name="LiteLLM",
                    level=logging.ERROR,
                    pathname=__file__,
                    lineno=1,
                    msg=f"rotation-probe-{i}-padding-{'x' * 80}",
                    args=(),
                    exc_info=None,
                )
            )
        handler.flush()
        # Allow worker to finish any straggler writes under pytest auto-flush off edge.
        deadline = time.time() + 2.0
        while time.time() < deadline and not backup_1.exists():
            time.sleep(0.02)
            handler.flush()

        assert log_file.exists()
        assert backup_1.exists()
        assert handler._rotating_handler is not None
        from logging.handlers import RotatingFileHandler

        assert isinstance(handler._rotating_handler, RotatingFileHandler)
    finally:
        handler.close()


def test_build_record_omits_content_fields_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_INCLUDE_CONTENT_FIELDS", raising=False)
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    setattr(record, "aawm_passthrough_request_shape_error_body_preview", "BODY")
    built = _build_aawm_error_log_record(record, formatter=logging.Formatter())
    assert "aawm_passthrough_request_shape_error_body_preview" not in built["context"]
    assert "BODY" not in json.dumps(built)

def test_worker_marks_drained_leftovers_task_done(monkeypatch, tmp_path):
    """Shutdown drain must call task_done for every leftover queue item."""
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_QUEUE_MAXSIZE", "8")

    handler = AawmErrorLogFileHandler()
    pause = threading.Event()
    pause.clear()
    release = threading.Event()
    original_main = handler._worker_main

    def paused_main() -> None:
        pause.set()
        release.wait(timeout=3.0)
        return original_main()

    handler._closed = True
    try:
        handler._queue.put_nowait(object())
    except Exception:
        pass
    if handler._worker is not None:
        handler._worker.join(timeout=0.5)
    handler._closed = False
    handler._queue = __import__("queue").Queue(maxsize=8)
    handler._worker = threading.Thread(
        target=paused_main,
        name="aawm-error-log-writer-drain",
        daemon=True,
    )
    handler._worker.start()
    assert pause.wait(timeout=1.0)

    try:
        assert handler._enqueue_line("drain-a\n") is True
        assert handler._enqueue_line("drain-b\n") is True
        assert handler._queue.unfinished_tasks >= 2
        release.set()
        handler.close()
        # unfinished_tasks should settle; a missed task_done would leave a sticky count.
        deadline = time.time() + 2.0
        while time.time() < deadline and handler._queue.unfinished_tasks != 0:
            time.sleep(0.01)
        assert handler._queue.unfinished_tasks == 0
    finally:
        release.set()
        try:
            handler.close()
        except Exception:
            pass


def test_note_dropped_record_updates_counters_under_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "rr004")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    handler = AawmErrorLogFileHandler()
    try:
        # Concurrent note_dropped must not lose increments.
        barrier = threading.Barrier(8)
        def worker():
            barrier.wait(timeout=2.0)
            for _ in range(50):
                handler._note_dropped_record()
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
        assert handler.dropped_record_count() == 8 * 50
    finally:
        handler.close()
