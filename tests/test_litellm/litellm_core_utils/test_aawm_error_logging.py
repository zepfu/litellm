import json
import logging
from pathlib import Path

import pytest

from litellm._logging import (
    AawmErrorLogFileHandler,
    _AAWM_ERROR_LOG_DEFAULT_BACKUP_COUNT,
    _AAWM_ERROR_LOG_DEFAULT_MAX_BYTES,
    _get_aawm_error_log_backup_count,
    _get_aawm_error_log_max_bytes,
    _get_aawm_error_log_path,
)


@pytest.fixture
def aawm_error_log_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    # We need to ensure the handler is using the latest env vars
    handler = AawmErrorLogFileHandler()

    # Use a dedicated logger for testing to avoid side effects
    logger = logging.getLogger("aawm-test-logger")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    yield logger, tmp_path / "test-error.jsonl", handler

    try:
        handler.close()
    except Exception:
        pass
    logger.handlers = []


def test_aawm_error_log_filtering(aawm_error_log_setup):
    logger, log_file, _handler = aawm_error_log_setup

    # 1. ERROR should be captured
    logger.error("test error")

    # 2. WARNING with exc_info should be captured
    try:
        raise ValueError("test traceback")
    except Exception:
        logger.warning("warning with traceback", exc_info=True)

    # 3. CRITICAL should be captured
    logger.critical("test critical")

    # 4. INFO should NOT be captured
    logger.info("test info")

    # 5. WARNING WITHOUT exc_info should NOT be captured
    logger.warning("test warning without traceback")

    # Offloaded writer: wait for disk visibility explicitly (no pytest env branch).
    _handler.flush()

    if not log_file.exists():
        pytest.fail("Log file was not created")

    records = [json.loads(line) for line in log_file.read_text().strip().split("\n")]

    messages = [r["message"] for r in records]

    assert "test error" in messages
    assert "warning with traceback" in messages
    assert "test critical" in messages
    assert "test info" not in messages
    assert "test warning without traceback" not in messages


def test_aawm_error_log_disabled_by_default(monkeypatch, tmp_path):
    """Opt-in stays off: without enable flags, emit must not create a log file."""
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_ENABLED", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_DIR", raising=False)
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    # Point cwd-relative default away from the real tree if someone flips enable.
    monkeypatch.chdir(tmp_path)

    assert _get_aawm_error_log_path() is None

    handler = AawmErrorLogFileHandler()
    logger = logging.getLogger("aawm-test-logger-disabled")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    try:
        logger.error("should not be written")
        assert list(tmp_path.glob("*-error.jsonl")) == []
        assert list(tmp_path.glob("*-error.jsonl.*")) == []
    finally:
        handler.close()
        logger.handlers = []


def test_aawm_error_log_rotation_defaults(monkeypatch):
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", raising=False)
    monkeypatch.delenv("LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT", raising=False)

    assert _get_aawm_error_log_max_bytes() == _AAWM_ERROR_LOG_DEFAULT_MAX_BYTES
    assert _get_aawm_error_log_max_bytes() == 10 * 1024 * 1024
    assert _get_aawm_error_log_backup_count() == _AAWM_ERROR_LOG_DEFAULT_BACKUP_COUNT
    assert _get_aawm_error_log_backup_count() == 5


def test_aawm_error_log_rotation_env_overrides(monkeypatch):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT", "2")

    assert _get_aawm_error_log_max_bytes() == 4096
    assert _get_aawm_error_log_backup_count() == 2


def test_aawm_error_log_rotates_when_max_bytes_exceeded(monkeypatch, tmp_path):
    """Prove size-based rotation: a tiny maxBytes creates a .1 backup file."""
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "test")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))
    # Smaller than a single JSONL payload so the second write forces rollover.
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", "200")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT", "3")

    handler = AawmErrorLogFileHandler()
    logger = logging.getLogger("aawm-test-logger-rotate")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_file = tmp_path / "test-error.jsonl"
    backup_1 = Path(f"{log_file}.1")

    try:
        # Each ERROR produces a multi-hundred-byte JSONL row; a few emits
        # exceed the 200-byte ceiling and rotate into backup files.
        for i in range(8):
            logger.error(
                "rotation-probe-%s-padding-%s",
                i,
                "x" * 80,
            )

        handler.flush()

        assert log_file.exists(), "active JSONL sink should exist after emits"
        assert backup_1.exists(), (
            "RotatingFileHandler should create a .1 backup once maxBytes is exceeded"
        )

        # Active + rotated files together must still be valid JSONL lines.
        all_lines: list[str] = []
        for path in sorted(tmp_path.glob("test-error.jsonl*"), reverse=True):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                all_lines.extend(text.split("\n"))

        assert all_lines, "expected at least one JSONL record across rotated files"
        for line in all_lines:
            row = json.loads(line)
            assert "message" in row
            assert row["message"].startswith("rotation-probe-")
            assert row.get("schema_version") == 1

        # backupCount=3 must cap retained backups at .1 .. .3
        assert not Path(f"{log_file}.4").exists()
    finally:
        handler.close()
        logger.handlers = []


def test_aawm_error_log_uses_rotating_file_handler(aawm_error_log_setup):
    logger, log_file, handler = aawm_error_log_setup

    logger.error("probe rotating handler")
    handler.flush()

    assert log_file.exists()
    assert handler._rotating_handler is not None
    from logging.handlers import RotatingFileHandler

    assert isinstance(handler._rotating_handler, RotatingFileHandler)
    assert handler._rotating_handler_path == str(log_file)
    assert handler._rotating_handler_max_bytes == _get_aawm_error_log_max_bytes()
    assert handler._rotating_handler_backup_count == _get_aawm_error_log_backup_count()
