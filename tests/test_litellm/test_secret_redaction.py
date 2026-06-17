import json
import logging
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from litellm._logging import (
    AawmErrorLogFileHandler,
    JsonFormatter,
    _get_aawm_error_log_path,
    _get_aawm_legacy_error_log_path,
    _redact_string,
    _secret_filter,
    _setup_json_exception_handlers,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)

SECRET = "sk-proj-abc123def456ghi789jklmnopqrst"


@pytest.fixture(autouse=True)
def _enable_redaction():
    """Ensure secret redaction is on (the default) for all tests in this module."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", True):
        yield


def _capture_logger_output(fn):
    """Run fn with all litellm loggers wired to a StringIO buffer, return output."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.addFilter(_secret_filter)
    loggers = [verbose_logger, verbose_proxy_logger, verbose_router_logger]
    saved = [(lg, lg.handlers[:], lg.level) for lg in loggers]
    for lg in loggers:
        lg.handlers.clear()
        lg.addHandler(h)
        lg.setLevel(logging.DEBUG)
    try:
        fn()
        return buf.getvalue()
    finally:
        for lg, handlers, level in saved:
            lg.handlers.clear()
            for old_h in handlers:
                lg.addHandler(old_h)
            lg.setLevel(level)


def test_redact_string_catches_secret_patterns():
    """Core regex patterns redact known secret formats."""
    cases = [
        "Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
        "api_key=a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "password=supersecretpassword123",
        "postgresql://admin:s3cretpass@db.example.com:5432/mydb",
        SECRET,
    ]
    for secret in cases:
        result = _redact_string("msg: " + secret)
        assert secret not in result, f"{secret!r} was not redacted"
        assert "REDACTED" in result

    normal = "Loaded model gpt-4 with 3 replicas on us-east-1"
    assert _redact_string(normal) == normal


def test_filter_redacts_secrets_in_logger_output():
    def log_messages():
        verbose_logger.debug("Key: " + SECRET)
        verbose_logger.debug("Normal message with no secrets")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output
    assert "Normal message with no secrets" in output


def test_filter_redacts_percent_style_args():
    """Secrets passed as %-style args should be redacted."""

    def log_messages():
        verbose_logger.debug("key=%s region=%s", SECRET, "us-east-1")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "us-east-1" in output


def test_filter_redacts_non_string_args():
    """Secrets inside dicts/lists passed as %-style args should be redacted."""

    def log_messages():
        verbose_logger.debug("Config: %s", {"nested": {"key": SECRET}})
        verbose_logger.debug("Keys: %s", [SECRET])

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output


def test_filter_redacts_exception_tracebacks():
    """Secrets embedded in exception messages must be redacted in tracebacks."""

    def log_messages():
        try:
            raise ValueError(f"Auth failed with key {SECRET}")
        except ValueError:
            verbose_logger.exception("Something went wrong")

    output = _capture_logger_output(log_messages)
    assert SECRET not in output
    assert "REDACTED" in output
    assert "Something went wrong" in output


def test_filter_redacts_extra_fields():
    """Secrets passed via extra={...} must be redacted on the record."""
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    record.api_key = SECRET
    record.region = "us-east-1"

    _secret_filter.filter(record)

    assert SECRET not in record.api_key
    assert "REDACTED" in record.api_key
    assert record.region == "us-east-1"


def test_disable_redaction_passes_secrets_through():
    """When LITELLM_DISABLE_REDACT_SECRETS=true, secrets pass through."""
    with patch("litellm._logging._ENABLE_SECRET_REDACTION", False):
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="key=" + SECRET,
            args=(),
            exc_info=None,
        )
        _secret_filter.filter(record)
        assert "sk-proj-" in record.msg


def test_x_api_key_regex_does_not_consume_json_delimiters():
    """x-api-key pattern must stop before closing quotes/braces so JSON stays valid."""
    # Simulates a JSON log line containing an x-api-key header value
    json_line = '{"headers": {"x-api-key": "secret123"}, "status": 200}'
    result = _redact_string(json_line)
    # The secret value should be redacted
    assert "secret123" not in result
    assert "REDACTED" in result
    # Closing delimiter must survive so the line is still valid-ish JSON
    assert '"status": 200' in result
    assert "}" in result


def test_json_excepthook_redacts_secrets():
    """Unhandled exceptions in JSON mode must have secrets redacted."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    h.addFilter(_secret_filter)

    # Capture what the excepthook would emit
    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg=f"Connection failed with key {SECRET}",
        args=(),
        exc_info=None,
    )
    # Simulate the filter + formatter pipeline
    _secret_filter.filter(record)
    output = h.formatter.format(record)
    assert SECRET not in output
    assert "REDACTED" in output


def test_json_excepthook_redacts_traceback_secrets():
    """Unhandled exception tracebacks in JSON mode must have secrets redacted."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    h.addFilter(_secret_filter)

    try:
        raise RuntimeError(f"Failed to auth with {SECRET}")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="LiteLLM",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg=str(exc_info[1]),
        args=(),
        exc_info=exc_info,
    )
    _secret_filter.filter(record)
    output = h.formatter.format(record)
    assert SECRET not in output
    assert "REDACTED" in output


def _read_aawm_error_jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_aawm_error_log_handler_writes_redacted_traceback(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "dev")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    handler = AawmErrorLogFileHandler()
    logger = logging.getLogger("test-aawm-error-log-handler")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    try:
        try:
            raise RuntimeError(f"failed with key {SECRET}")
        except RuntimeError:
            logger.exception("managed error api_key=%s", SECRET)
    finally:
        logger.handlers.clear()
        for saved_handler in saved_handlers:
            logger.addHandler(saved_handler)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate

    payload = _read_aawm_error_jsonl(tmp_path / "dev-error.jsonl")
    assert payload["schema_version"] == 1
    assert payload["environment"] == "dev"
    assert payload["logger"] == "test-aawm-error-log-handler"
    assert payload["level"] == "ERROR"
    assert "managed error" in payload["message"]
    assert "REDACTED" in payload["message"]
    assert "Traceback (most recent call last)" in payload["traceback_text"]
    assert "RuntimeError" in payload["raw_text"]
    assert payload["traceback_lines"]
    assert payload["fingerprint"]
    assert payload["context"]["provider"] is None
    assert SECRET not in json.dumps(payload)
    assert "REDACTED" in payload["raw_text"]


def test_aawm_error_log_handler_writes_context_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "dev")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    handler = AawmErrorLogFileHandler()
    logger = logging.getLogger("test-aawm-error-log-context")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    try:
        try:
            raise RuntimeError("provider failure")
        except RuntimeError:
            logger.exception(
                "upstream error",
                extra={
                    "source": "pass_through_endpoint",
                    "container": "litellm-dev",
                    "endpoint": "/anthropic/v1/messages",
                    "upstream_url": "https://api.example.test/v1/messages",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "model_alias": "aawm-code-anthropic",
                    "route_family": "anthropic_messages",
                    "status_code": 529,
                    "trace_id": "trace-123",
                    "litellm_call_id": f"call-{SECRET}",
                },
            )
    finally:
        logger.handlers.clear()
        for saved_handler in saved_handlers:
            logger.addHandler(saved_handler)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate

    payload = _read_aawm_error_jsonl(tmp_path / "dev-error.jsonl")
    assert payload["observed_at"].endswith("+00:00")
    assert payload["context"] == {
        "source": "pass_through_endpoint",
        "container": "litellm-dev",
        "endpoint": "/anthropic/v1/messages",
        "upstream_url": "https://api.example.test/v1/messages",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_alias": "aawm-code-anthropic",
        "route_family": "anthropic_messages",
        "status_code": 529,
        "trace_id": "trace-123",
        "litellm_call_id": "call-REDACTED",
        "callback_name": None,
        "callback_phase": None,
        "handler_branch": None,
        "langfuse_failure_class": None,
        "event_type": None,
        "worker_timeout_seconds": None,
        "queue_depth": None,
        "queue_maxsize": None,
        "coroutine_name": None,
        "worker_delivery_state": None,
    }
    assert SECRET not in json.dumps(payload)


def test_aawm_error_log_path_uses_sanitized_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "Dev Env/With Spaces")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    assert _get_aawm_error_log_path() == str(
        tmp_path / "dev-env-with-spaces-error.jsonl"
    )
    assert _get_aawm_legacy_error_log_path() == str(
        tmp_path / "dev-env-with-spaces-error.log"
    )


def test_aawm_error_log_handler_groups_identical_events_by_fingerprint(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "dev")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    handler = AawmErrorLogFileHandler()
    logger = logging.getLogger("test-aawm-error-log-fingerprint")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    try:
        for _ in range(2):
            try:
                raise RuntimeError("same failure")
            except RuntimeError:
                logger.exception("managed error")
    finally:
        logger.handlers.clear()
        for saved_handler in saved_handlers:
            logger.addHandler(saved_handler)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate

    payloads = [
        json.loads(line)
        for line in (tmp_path / "dev-error.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(payloads) == 2
    assert payloads[0]["fingerprint"] == payloads[1]["fingerprint"]


def test_aawm_error_log_handler_ignores_non_exception_critical_notice(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "dev")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    handler = AawmErrorLogFileHandler()
    record = logging.LogRecord(
        name="LiteLLM Proxy",
        level=logging.CRITICAL,
        pathname="",
        lineno=0,
        msg="LITELLM_MASTER_KEY is not set",
        args=(),
        exc_info=None,
    )

    handler.handle(record)

    assert not (tmp_path / "dev-error.jsonl").exists()
    assert not (tmp_path / "dev-error.log").exists()


def test_json_exception_hook_writes_aawm_error_log(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "1")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_ENV", "dev")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_DIR", str(tmp_path))

    original_excepthook = sys.excepthook
    try:
        _setup_json_exception_handlers(JsonFormatter())
        try:
            raise RuntimeError(f"uncaught path key {SECRET}")
        except RuntimeError:
            exc_info = sys.exc_info()
        sys.excepthook(*exc_info)
    finally:
        sys.excepthook = original_excepthook

    payload = _read_aawm_error_jsonl(tmp_path / "dev-error.jsonl")
    assert "uncaught path key" in payload["message"]
    assert "Traceback (most recent call last)" in payload["traceback_text"]
    assert SECRET not in json.dumps(payload)
    assert "REDACTED" in payload["raw_text"]
