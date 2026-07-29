"""Focused coverage for the D1-574 real-Codex expected-429 harness path."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"
CONFIG_PATH = ROOT / "scripts" / "local-ci" / "anthropic_adapter_config.json"
CASE_NAME = "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location(
        "run_anthropic_adapter_acceptance_target_db_test_module",
        HARNESS_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def harness():
    module = _load_harness_module()
    module._CONTAINER_ENV_CACHE.clear()
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _dev_profile() -> dict[str, str]:
    return {
        key: str(value)
        for key, value in _config()["target_profiles"]["dev"].items()
    }


def _neutralize_case_dependencies(
    harness,
    monkeypatch,
    *,
    provider_error_failures: list[str] | None = None,
) -> None:
    monkeypatch.setattr(
        harness.RA, "_poll_langfuse_session_traces", lambda **kw: ([], None)
    )
    monkeypatch.setattr(
        harness.RA,
        "_validate_generation_observations",
        lambda **kw: ([], [], []),
    )
    monkeypatch.setattr(
        harness.RA, "_validate_trace_enrichment", lambda **kw: ({}, [], [])
    )
    monkeypatch.setattr(
        harness.RA, "_validate_trace_context", lambda **kw: ({}, [])
    )
    monkeypatch.setattr(
        harness.RA, "_validate_generation_metadata", lambda **kw: ({}, [])
    )
    monkeypatch.setattr(
        harness.RA, "_validate_span_observations", lambda **kw: ({}, [], [])
    )
    monkeypatch.setattr(
        harness,
        "_validate_logged_request_payload_checks",
        lambda **kw: ({}, [], []),
    )
    monkeypatch.setattr(
        harness.RA,
        "_validate_logged_request_text_checks",
        lambda **kw: ({}, [], []),
    )
    monkeypatch.setattr(
        harness, "_validate_session_history", lambda **kw: ({"record": None}, [])
    )
    monkeypatch.setattr(
        harness, "_validate_rate_limit_observations", lambda **kw: ({}, [], [])
    )
    monkeypatch.setattr(
        harness,
        "_validate_provider_error_observations",
        lambda **kw: (
            {"matched_records": [{"status_code": 429}]},
            list(provider_error_failures or []),
        ),
    )
    monkeypatch.setattr(
        harness, "_validate_runtime_postcondition", lambda **kw: ({}, [])
    )
    monkeypatch.setattr(
        harness, "_validate_runtime_logs", lambda **kw: ({}, [], [])
    )


def _run_expected_error_case(
    harness,
    monkeypatch,
    *,
    stdout: str,
    stderr: str = "",
    attempt_status: int | None = 429,
    extra_config: dict | None = None,
) -> dict:
    _neutralize_case_dependencies(harness, monkeypatch)
    run = {"exit_code": 1, "stdout": stdout, "stderr": stderr}
    attempts = [
        {
            "attempt": 1,
            "api_error_status": attempt_status,
            "is_error": True,
            "exit_code": 1,
        }
    ]
    monkeypatch.setattr(
        harness,
        "_run_command_with_retry",
        lambda config: (harness.RA._utcnow(), run, attempts),
    )
    monkeypatch.setattr(
        harness, "_extract_command_session_id", lambda value: "test-session"
    )
    config = {
        "expected_api_error_status": 429,
        "match_trace_session_id_from_stdout": False,
        "expected_trace_session_id": "test-session",
        "provider_error_observations_validation": {
            "expected_rows": [{"required_equals": {"status_code": 429}}]
        },
    }
    config.update(extra_config or {})
    return harness._validate_case(
        "case",
        config,
        query_url="http://127.0.0.1:3000",
        public_key="pk",
        secret_key="sk",
        litellm_base_url="http://127.0.0.1:4001",
    )


def _result_json(status: int = 429, result: str = "short") -> str:
    return json.dumps(
        {
            "type": "result",
            "is_error": True,
            "status_code": status,
            "result": result,
        }
    )


def _provider_checks() -> dict:
    return _config()["cases"][CASE_NAME]["provider_error_observations_validation"]


def _clean_provider_record() -> dict:
    return {
        "provider": "opencode_zen",
        "model": "big-pickle",
        "route_family": "codex_opencode_zen_adapter",
        "status_code": 429,
        "retry_after_seconds": 60.0,
        "session_id": "test-session",
        "litellm_call_id": "call-123",
        "metadata": {"diagnostic": "free usage capacity exhausted"},
    }


def test_container_env_resolver_uses_targeted_printenv_and_cache(harness, monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = "secret\n"
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Completed(),
    )
    assert harness._resolve_container_env_value("litellm-dev", "KEY") == "secret"
    assert harness._resolve_container_env_value("litellm-dev", "KEY") == "secret"
    assert len(calls) == 1
    command = calls[0][0][0]
    assert command == ["docker", "exec", "litellm-dev", "printenv", "KEY"]
    assert calls[0][1]["capture_output"] is True


@pytest.mark.parametrize("failure", ["returncode", "timeout"])
def test_container_env_resolver_failure_is_silent(harness, monkeypatch, failure):
    if failure == "timeout":
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=30)
    else:
        class Completed:
            returncode = 1
            stdout = ""
            stderr = "secret must not be logged"

        def fake_run(*args, **kwargs):
            return Completed()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    assert harness._resolve_container_env_value("litellm-dev", "KEY") is None


def test_target_profile_applies_db_settings_to_all_db_validators(harness):
    case = {
        key: {"db_port": 5434}
        for key in (
            "session_history_validation",
            "rate_limit_observations_validation",
            "provider_error_observations_validation",
            "tool_activity_validation",
        )
    }
    harness._apply_profile_validation_db_overrides(case, _dev_profile())
    for block in case.values():
        assert block["db_port"] == 6433
        assert block["db_name"] == "litellm_dev"
        assert block["db_user"] == "litellm_dev"
        assert block["db_password_container"] == "litellm-dev"
        assert block["db_password_container_env"] == "AAWM_DB_PASSWORD"


def test_db_container_credential_wins_over_stale_host(harness, monkeypatch):
    monkeypatch.setenv("AAWM_DB_PASSWORD", "stale-host")
    monkeypatch.setattr(
        harness, "_resolve_container_env_value", lambda *args: "container-secret"
    )
    settings, failures = harness._validation_db_settings(
        family="case",
        checks={
            "db_password_env": "AAWM_DB_PASSWORD",
            "db_password_container": "litellm-dev",
            "db_password_container_env": "AAWM_DB_PASSWORD",
        },
        validation_name="provider_error_observations",
    )
    assert failures == []
    assert settings["password"] == "container-secret"


def test_db_container_credential_failure_is_fail_closed_and_redacted(
    harness, monkeypatch
):
    monkeypatch.setenv("AAWM_DB_PASSWORD", "stale-host-secret")
    monkeypatch.setattr(harness, "_resolve_container_env_value", lambda *args: None)
    settings, failures = harness._validation_db_settings(
        family="case",
        checks={
            "db_password_env": "AAWM_DB_PASSWORD",
            "db_password_container": "litellm-dev",
            "db_password_container_env": "AAWM_DB_PASSWORD",
        },
        validation_name="provider_error_observations",
    )
    assert settings is None
    assert failures == [
        "case could not retrieve target-owned DB credential for "
        "provider_error_observations validation"
    ]
    assert "stale-host-secret" not in failures[0]


def test_db_host_env_remains_supported_without_container_ownership(
    harness, monkeypatch
):
    monkeypatch.setenv("AAWM_DB_PASSWORD", "host-secret")
    settings, failures = harness._validation_db_settings(
        family="case",
        checks={"db_password_env": "AAWM_DB_PASSWORD"},
        validation_name="session_history",
    )
    assert failures == []
    assert settings["password"] == "host-secret"


def test_langfuse_container_credentials_win_over_stale_host(harness, monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "stale-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "stale-sk")
    values = {
        "LANGFUSE_PUBLIC_KEY": "container-pk",
        "LANGFUSE_SECRET_KEY": "container-sk",
    }
    monkeypatch.setattr(
        harness,
        "_resolve_container_env_value",
        lambda container, name: values[name],
    )

    class Args:
        langfuse_query_url = None

    result = harness._resolve_main_credentials(
        config={}, args=Args(), profile=_dev_profile()
    )
    assert result[:3] == (
        "container-pk",
        "container-sk",
        "http://127.0.0.1:3000",
    )


def test_langfuse_container_credential_failure_is_fail_closed_and_redacted(
    harness, monkeypatch
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "stale-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "stale-sk")
    monkeypatch.setattr(harness, "_resolve_container_env_value", lambda *args: None)
    errors = []
    monkeypatch.setattr(harness, "_emit_stderr", errors.append)

    class Args:
        langfuse_query_url = None

    assert (
        harness._resolve_main_credentials(
            config={}, args=Args(), profile=_dev_profile()
        )
        == 2
    )
    assert errors == ["Could not retrieve target-owned Langfuse credentials"]
    assert "stale-" not in errors[0]


def test_langfuse_host_env_and_cli_url_work_without_container_ownership(
    harness, monkeypatch
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "host-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "host-sk")

    class Args:
        langfuse_query_url = "http://explicit.example:3000"

    result = harness._resolve_main_credentials(
        config={},
        args=Args(),
        profile=dict(harness.BUILT_IN_TARGET_PROFILES["prod"]),
    )
    assert result[:3] == ("host-pk", "host-sk", "http://explicit.example:3000")


@pytest.mark.parametrize(
    ("attempt_status", "stdout_status", "passed"),
    [
        (429, 429, True),
        (None, 429, True),
        (500, 500, False),
        (None, None, False),
    ],
)
def test_expected_api_error_status_paths(
    harness, monkeypatch, attempt_status, stdout_status, passed
):
    stdout = (
        _result_json(stdout_status)
        if stdout_status is not None
        else json.dumps({"type": "result", "is_error": True})
    )
    result = _run_expected_error_case(
        harness,
        monkeypatch,
        stdout=stdout,
        attempt_status=attempt_status,
    )
    assert result["passed"] is passed
    assert any("command failed" in failure for failure in result["failures"]) is (
        not passed
    )
    if passed:
        assert result["session_history"]["skipped"] == "expected_api_error"
        assert (
            result["langfuse"]["trace_enrichment"]["skipped"]
            == "expected_api_error"
        )


def test_clean_provider_error_record_passes(harness):
    matched, failures = harness._validate_provider_error_records(
        family="case",
        records=[_clean_provider_record()],
        checks=_provider_checks(),
    )
    assert failures == []
    assert matched[0]["litellm_call_id"] == "call-123"


@pytest.mark.parametrize(
    "metadata",
    [
        {"raw_body": {"error": "raw provider body"}},
        {"account_identifier": "acct-secret"},
        {"contact": "owner@example.com"},
        {"diagnostic": "x" * 5000},
    ],
)
def test_provider_error_record_rejects_leakage_or_unbounded_metadata(
    harness, metadata
):
    record = _clean_provider_record()
    record["metadata"] = metadata
    _, failures = harness._validate_provider_error_records(
        family="case",
        records=[record],
        checks=_provider_checks(),
    )
    assert failures


@pytest.mark.parametrize("retry_after", ["sixty", -1, 86401])
def test_provider_error_record_rejects_invalid_retry_after(harness, retry_after):
    record = _clean_provider_record()
    record["retry_after_seconds"] = retry_after
    _, failures = harness._validate_provider_error_records(
        family="case",
        records=[record],
        checks=_provider_checks(),
    )
    assert failures


@pytest.mark.parametrize(
    ("status", "allowed"),
    [(429, True), (500, False), (502, False), (503, False), (504, False)],
)
def test_expected_429_runtime_log_suppression_is_status_specific(
    harness, monkeypatch, status, allowed
):
    signature = f"pass_through_endpoint(): Exception occured - {status}: handled"
    monkeypatch.setattr(
        harness,
        "_read_runtime_logs_since",
        lambda **kwargs: (
            {
                "docker_logs_exit_code": 0,
                "docker_logs_since": "start",
                "docker_logs_until": "end",
                "log_excerpt": signature,
            },
            signature,
        ),
    )
    _, failures, _ = harness._validate_runtime_logs(
        family="case",
        started=harness.RA._utcnow(),
        checks={"disable_default_429_traceback_check": True},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
    )
    assert (failures == []) is allowed


@pytest.mark.parametrize(
    "payload",
    [
        "sk-secret",
        '"raw_body":{"error":"upstream"}',
        '"account_identifier":"acct-secret"',
        "owner@example.com",
        "Traceback (most recent call last)",
        "500 Internal Server Error",
    ],
)
def test_raw_stdout_security_checks_cannot_be_hidden_by_short_result(
    harness, monkeypatch, payload
):
    stdout = (
        json.dumps({"type": "event", "diagnostic": payload})
        + "\n"
        + _result_json(result="short")
    )
    assert harness._extract_command_output_text(stdout) == "short"
    case = _config()["cases"][CASE_NAME]
    result = _run_expected_error_case(
        harness,
        monkeypatch,
        stdout=stdout,
        extra_config={
            "command_stdout_text_checks": case["command_stdout_text_checks"],
            "command_stderr_text_checks": case["command_stderr_text_checks"],
        },
    )
    assert result["passed"] is False
    assert any("command stdout" in failure for failure in result["failures"])


def test_raw_stdout_13k_jsonl_fails_even_with_short_extracted_result(
    harness, monkeypatch
):
    stdout = (
        json.dumps({"type": "event", "diagnostic": "x" * 13000})
        + "\n"
        + _result_json(result="short")
    )
    assert harness._extract_command_output_text(stdout) == "short"
    checks = _config()["cases"][CASE_NAME]["command_stdout_text_checks"]
    result = _run_expected_error_case(
        harness,
        monkeypatch,
        stdout=stdout,
        extra_config={"command_stdout_text_checks": checks},
    )
    assert result["passed"] is False
    assert any("above maximum length" in failure for failure in result["failures"])


def test_raw_stderr_is_checked(harness, monkeypatch):
    checks = _config()["cases"][CASE_NAME]["command_stderr_text_checks"]
    result = _run_expected_error_case(
        harness,
        monkeypatch,
        stdout=_result_json(),
        stderr="500 Internal Server Error",
        extra_config={"command_stderr_text_checks": checks},
    )
    assert result["passed"] is False
    assert any("command stderr" in failure for failure in result["failures"])


def test_config_case_uses_failure_observability_and_read_only_codex():
    cfg = _config()
    case = cfg["cases"][CASE_NAME]
    assert CASE_NAME in cfg["default_excluded_cases"]
    assert case["expected_api_error_status"] == 429
    assert "opencode_zen/big-pickle" in case["command"]
    assert "-s" in case["command"] and "read-only" in case["command"]
    assert "-C" in case["command"] and "{repository_root}" in case["command"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in case["command"]
    assert "required_trace_tags" not in case
    assert "allowed_generation_routes" not in case
    shv = case["session_history_validation"]
    assert shv["expected_rows"][0]["required_equals"]["provider"] == "opencode_zen"
    assert shv["expected_rows"][0]["required_equals"]["model"] == "big-pickle"
    assert "required_generation_metadata_truthy" not in case
    expected = case["provider_error_observations_validation"]["expected_rows"][0]
    assert expected["required_equals"] == {
        "provider": "opencode_zen",
        "model": "big-pickle",
        "route_family": "codex_opencode_zen_adapter",
        "status_code": 429,
    }
    assert expected["required_truthy"] == ["litellm_call_id"]
    assert case["command_stdout_text_checks"]["maximum_chars"] == 12000
    assert case["command_stderr_text_checks"]["maximum_chars"] == 12000
    assert case["runtime_log_checks"]["disable_default_429_traceback_check"] is True


def test_config_dev_profile_declares_container_owned_credentials():
    dev = _config()["target_profiles"]["dev"]
    assert dev["validation_db_port"] == "6433"
    assert dev["validation_db_name"] == "litellm_dev"
    assert dev["validation_db_user"] == "litellm_dev"
    assert dev["validation_db_password_container_env"] == "AAWM_DB_PASSWORD"
    assert dev["langfuse_public_key_container_env"] == "LANGFUSE_PUBLIC_KEY"
    assert dev["langfuse_secret_key_container_env"] == "LANGFUSE_SECRET_KEY"
    assert "validation_db_host" not in _config()["target_profiles"]["prod"]


def test_failure_observability_with_session_history_calls_validator(
    harness, monkeypatch
):
    """session_history_validation must execute for failure-observability cases."""
    _neutralize_case_dependencies(harness, monkeypatch)
    call_log: list[dict] = []

    def tracking_validate_session_history(**kwargs):
        call_log.append(kwargs)
        return {"record": None, "records": []}, ["session row missing"]

    monkeypatch.setattr(
        harness, "_validate_session_history", tracking_validate_session_history
    )
    # Build run result manually to avoid _run_expected_error_case re-neutralizing
    run = {"exit_code": 1, "stdout": _result_json(429), "stderr": ""}
    attempts = [
        {"attempt": 1, "api_error_status": 429, "is_error": True, "exit_code": 1}
    ]
    monkeypatch.setattr(
        harness,
        "_run_command_with_retry",
        lambda config: (harness.RA._utcnow(), run, attempts),
    )
    monkeypatch.setattr(
        harness, "_extract_command_session_id", lambda value: "test-session"
    )
    config = {
        "expected_api_error_status": 429,
        "match_trace_session_id_from_stdout": False,
        "expected_trace_session_id": "test-session",
        "provider_error_observations_validation": {
            "expected_rows": [{"required_equals": {"status_code": 429}}]
        },
        "session_history_validation": {
            "expected_rows": [
                {"required_equals": {"provider": "opencode_zen"}}
            ]
        },
    }
    result = harness._validate_case(
        "case",
        config,
        query_url="http://127.0.0.1:3000",
        public_key="pk",
        secret_key="sk",
        litellm_base_url="http://127.0.0.1:4001",
    )
    assert len(call_log) == 1
    assert call_log[0]["family"] == "case"
    assert call_log[0]["session_id"] == "test-session"
    assert result["passed"] is False
    assert any("session row missing" in f for f in result["failures"])


def test_failure_observability_without_session_history_still_skips(
    harness, monkeypatch
):
    """No session_history_validation config preserves skip behavior."""
    result = _run_expected_error_case(
        harness,
        monkeypatch,
        stdout=_result_json(429),
    )
    assert result["passed"] is True
    assert result["session_history"]["skipped"] == "expected_api_error"


def test_validate_case_records_command_thread_id(harness, monkeypatch):
    """_validate_case records command_thread_id separately from session_id."""
    _neutralize_case_dependencies(harness, monkeypatch)
    stdout_with_thread = (
        json.dumps({"type": "thread.started", "thread_id": "thr-abc"})
        + "\n"
        + _result_json(429)
    )
    result = _run_expected_error_case(
        harness,
        monkeypatch,
        stdout=stdout_with_thread,
    )
    assert result["langfuse"]["command_thread_id"] == "thr-abc"
    assert result["langfuse"]["command_session_id"] == "test-session"
    assert result["langfuse"]["command_thread_id"] != result["langfuse"]["command_session_id"]
