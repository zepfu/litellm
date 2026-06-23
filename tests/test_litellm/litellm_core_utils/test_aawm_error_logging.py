import json
import logging

import pytest

from litellm._logging import AawmErrorLogFileHandler


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

    return logger, tmp_path / "test-error.jsonl"


def test_aawm_error_log_filtering(aawm_error_log_setup):
    logger, log_file = aawm_error_log_setup

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

    if not log_file.exists():
        pytest.fail("Log file was not created")

    records = [json.loads(line) for line in log_file.read_text().strip().split("\n")]

    messages = [r["message"] for r in records]

    assert "test error" in messages
    assert "warning with traceback" in messages
    assert "test critical" in messages
    assert "test info" not in messages
    assert "test warning without traceback" not in messages
