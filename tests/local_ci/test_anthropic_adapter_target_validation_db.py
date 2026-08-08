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
        harness.RA,
        "_recent_langfuse_all_traces",
        lambda **kw: [
            {
                "id": "trace-expected-error",
                "sessionId": "test-session",
            }
        ],
    )
    monkeypatch.setattr(
        harness.RA,
        "_recent_langfuse_generation_observations_for_trace_ids",
        lambda **kw: [],
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
        "expected_api_error_langfuse_poll_timeout_seconds": 0.001,
        "expected_api_error_langfuse_poll_interval_seconds": 0.001,
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
        assert block["db_port"] == 6435
        assert block["db_name"] == "aawm_tristore"
        assert block["db_user"] == "aawm"
        assert block["db_password_container"] == "thoth-aawm-dev-pgbouncer"
        assert block["db_password_container_env"] == "PGBOUNCER_AUTH_PASSWORD"


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
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "container-pk",
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": "container-sk",
    }
    seen_containers = []
    monkeypatch.setattr(
        harness,
        "_resolve_container_env_value",
        lambda container, name: seen_containers.append(container) or values[name],
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
    # CFG-003 item 6: credentials resolved from the dedicated Langfuse
    # container, never the target litellm-dev container.
    assert seen_containers == ["aawm-langfuse-web", "aawm-langfuse-web"]


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
    # CFG-003 Initiation 2 item 4: dev evidence DB is aawm_tristore on 6435.
    assert dev["validation_db_host"] == "127.0.0.1"
    assert dev["validation_db_port"] == "6435"
    assert dev["validation_db_name"] == "aawm_tristore"
    assert dev["validation_db_user"] == "aawm"
    assert dev["validation_db_password_container"] == "thoth-aawm-dev-pgbouncer"
    assert dev["validation_db_password_container_env"] == "PGBOUNCER_AUTH_PASSWORD"
    # CFG-003 Initiation 2 item 6: Langfuse credentials are owned by a
    # dedicated container distinct from the target litellm-dev container.
    assert dev["langfuse_credential_container"] == "aawm-langfuse-web"
    assert dev["langfuse_public_key_container_env"] == "LANGFUSE_INIT_PROJECT_PUBLIC_KEY"
    assert dev["langfuse_secret_key_container_env"] == "LANGFUSE_INIT_PROJECT_SECRET_KEY"
    assert "validation_db_host" not in _config()["target_profiles"]["prod"]


def test_validation_db_password_container_separation(harness):
    """Explicit validation_db_password_container takes precedence over
    docker_container_name for credential resolution."""
    profile = {
        "validation_db_host": "127.0.0.1",
        "validation_db_port": "6433",
        "validation_db_name": "litellm_dev",
        "validation_db_user": "litellm_dev",
        "docker_container_name": "litellm-dev",
        "validation_db_password_container": "thoth-litellm-dev-pgbouncer",
        "validation_db_password_container_env": "PGBOUNCER_AUTH_PASSWORD",
    }
    case = {"session_history_validation": {"db_port": 5434}}
    harness._apply_profile_validation_db_overrides(case, profile)
    block = case["session_history_validation"]
    assert block["db_password_container"] == "thoth-litellm-dev-pgbouncer"
    assert block["db_password_container_env"] == "PGBOUNCER_AUTH_PASSWORD"


def test_validation_db_password_container_fallback_to_docker_container(harness):
    """When validation_db_password_container is absent, falls back to
    docker_container_name for backward compatibility."""
    profile = {
        "validation_db_host": "127.0.0.1",
        "validation_db_port": "6433",
        "validation_db_name": "litellm_dev",
        "validation_db_user": "litellm_dev",
        "docker_container_name": "litellm-dev",
        "validation_db_password_container_env": "AAWM_DB_PASSWORD",
    }
    case = {"session_history_validation": {"db_port": 5434}}
    harness._apply_profile_validation_db_overrides(case, profile)
    block = case["session_history_validation"]
    assert block["db_password_container"] == "litellm-dev"
    assert block["db_password_container_env"] == "AAWM_DB_PASSWORD"


def test_cfg003_db_settings_uses_explicit_validation_container(harness, monkeypatch):
    """_cfg003_db_settings resolves credentials from the explicit
    validation_db_password_container, not docker_container_name."""
    calls = []
    monkeypatch.setattr(
        harness,
        "_resolve_container_env_value",
        lambda container, env: calls.append((container, env)) or "pgb-secret",
    )
    profile = {
        "validation_db_host": "127.0.0.1",
        "validation_db_port": "6433",
        "validation_db_name": "litellm_dev",
        "validation_db_user": "litellm_dev",
        "docker_container_name": "litellm-dev",
        "validation_db_password_container": "thoth-litellm-dev-pgbouncer",
        "validation_db_password_container_env": "PGBOUNCER_AUTH_PASSWORD",
    }
    settings = harness._cfg003_db_settings({}, profile=profile)
    assert settings is not None
    assert settings["password"] == "pgb-secret"
    assert calls == [("thoth-litellm-dev-pgbouncer", "PGBOUNCER_AUTH_PASSWORD")]


def test_cfg003_db_settings_fallback_without_explicit_container(harness, monkeypatch):
    """_cfg003_db_settings falls back to docker_container_name when
    validation_db_password_container is absent."""
    calls = []
    monkeypatch.setattr(
        harness,
        "_resolve_container_env_value",
        lambda container, env: calls.append((container, env)) or "legacy-secret",
    )
    profile = {
        "validation_db_host": "127.0.0.1",
        "validation_db_port": "6433",
        "validation_db_name": "litellm_dev",
        "validation_db_user": "litellm_dev",
        "docker_container_name": "litellm-dev",
        "validation_db_password_container_env": "AAWM_DB_PASSWORD",
    }
    settings = harness._cfg003_db_settings({}, profile=profile)
    assert settings is not None
    assert settings["password"] == "legacy-secret"
    assert calls == [("litellm-dev", "AAWM_DB_PASSWORD")]


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


# ---------------------------------------------------------------------------
# CFG-003 Initiation 2: Fix 1 - CLI template placeholder resolution
# ---------------------------------------------------------------------------


class TestTemplatePlaceholderResolution:
    """Unresolved {placeholder} values must not become effective IDs."""

    def test_is_template_placeholder_detects_placeholders(self, harness):
        assert harness._is_template_placeholder("{harness_user_id}") is True
        assert harness._is_template_placeholder("{session_id}") is True
        assert harness._is_template_placeholder("{case_name}") is True
        assert harness._is_template_placeholder("adapter-harness-tenant") is False
        assert harness._is_template_placeholder("litellm-harness.dev.case.123") is False
        assert harness._is_template_placeholder("") is False

    def test_contains_unresolved_placeholder(self, harness):
        assert harness._contains_unresolved_placeholder("{harness_user_id}") is True
        assert harness._contains_unresolved_placeholder(
            ["codex", "-c", "x={session_id}"]
        ) is True
        assert harness._contains_unresolved_placeholder(
            {"key": "{case_name}"}
        ) is True
        assert harness._contains_unresolved_placeholder("concrete-value") is False
        assert harness._contains_unresolved_placeholder(
            ["codex", "-m", "basic"]
        ) is False

    def test_cli_harness_context_resolves_placeholder_user_ids(self, harness):
        """Config with expected_user_ids=['{harness_user_id}'] must derive a
        concrete ID, not pass the placeholder through."""
        config = {
            "cli_passthrough": "codex",
            "command": ["codex", "exec", "-m", "basic"],
            "expected_user_ids": ["{harness_user_id}"],
            "tenant_id": "explicit-operator-tenant",
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_cli_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        # The effective user ID must be concrete, not a placeholder.
        for uid in result["expected_user_ids"]:
            assert not harness._is_template_placeholder(uid)
            assert "{" not in uid

    def test_cli_harness_context_preserves_concrete_user_ids(self, harness):
        """A non-harness explicit tenant ID is preserved as the expected
        trace user identity (repository mapping applies only to
        harness/validation tenant aliases)."""
        config = {
            "cli_passthrough": "codex",
            "command": ["codex", "exec", "-m", "basic"],
            "expected_user_ids": ["adapter-harness-tenant"],
            "tenant_id": "operator-explicit-tenant",
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_cli_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        assert result["expected_user_ids"] == ["operator-explicit-tenant"]

    def test_codex_cli_expected_user_id_resolves_to_repository_tenant(
        self, harness, monkeypatch
    ):
        """Codex harness/validation tenant aliases map to the repository
        identity for trace correlation, while emitted headers/session keep
        the transient harness user ID."""
        monkeypatch.setattr(
            harness, "_resolve_harness_repository", lambda: "aawm/litellm"
        )
        config = {
            "cli_passthrough": "codex",
            "command": ["codex", "exec", "-m", "basic"],
            "tenant_id": "adapter-harness-tenant",
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_cli_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        harness_user_id = harness.RA._build_claude_harness_user_id(
            target="dev", case_name="test_case"
        )
        assert result["expected_user_ids"] == ["aawm/litellm"]
        assert harness_user_id in " ".join(str(v) for v in result["command"])
        assert result["expected_trace_session_id"] == f"{harness_user_id}.session"

    def test_basic_alias_codex_case_no_unresolved_placeholders_after_resolution(
        self, harness
    ):
        """The real basic-alias Codex collaboration case must have no unresolved
        placeholders after _ensure_cli_harness_context."""
        cfg = _config()
        case = dict(
            cfg["cases"]["native_openai_passthrough_responses_codex_basic_alias_collaboration"]
        )
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_cli_harness_context(
            case,
            profile=profile,
            target="dev",
            case_name="native_openai_passthrough_responses_codex_basic_alias_collaboration",
        )
        assert not harness._contains_unresolved_placeholder(result["command"])
        assert not harness._contains_unresolved_placeholder(result["expected_user_ids"])


# ---------------------------------------------------------------------------
# CFG-003 Initiation 2: Fix 2 - Codex basic-alias collaboration command shape
# ---------------------------------------------------------------------------


class TestCodexBasicAliasCollaborationCommandShape:
    """The basic-alias Codex collaboration case must follow the proven alibaba
    collaboration command shape with multi_agent, opencode child, catalog,
    read-only sandbox, and repository context."""

    def _case(self):
        return _config()["cases"][
            "native_openai_passthrough_responses_codex_basic_alias_collaboration"
        ]

    def test_multi_agent_enabled(self):
        cmd = self._case()["command"]
        assert "--enable" in cmd
        enable_indices = [i for i, v in enumerate(cmd) if v == "--enable"]
        enabled_features = {cmd[i + 1] for i in enable_indices}
        assert "multi_agent" in enabled_features
        assert "multi_agent_v2" in enabled_features

    def test_opencode_child_agent_configured(self):
        cmd = self._case()["command"]
        cmd_str = " ".join(str(v) for v in cmd)
        assert "agents.opencode.config_file=" in cmd_str
        assert "codex_opencode_agent.toml" in cmd_str

    def test_model_catalog_configured(self):
        cmd = self._case()["command"]
        cmd_str = " ".join(str(v) for v in cmd)
        assert "model_catalog_json=" in cmd_str
        assert "codex_basic_alias_model_catalog.json" in cmd_str

    def test_read_only_sandbox_and_repository_context(self):
        cmd = self._case()["command"]
        assert "-s" in cmd
        s_idx = cmd.index("-s")
        assert cmd[s_idx + 1] == "read-only"
        assert "-C" in cmd
        c_idx = cmd.index("-C")
        assert cmd[c_idx + 1] == "{repository_root}"

    def test_model_is_basic_alias(self):
        cmd = self._case()["command"]
        assert "-m" in cmd
        m_idx = cmd.index("-m")
        assert cmd[m_idx + 1] == "basic"

    def test_uses_litellm_dev_egress(self):
        cmd = self._case()["command"]
        cmd_str = " ".join(str(v) for v in cmd)
        assert "model_provider=" in cmd_str
        assert "{codex_profile}" in cmd_str
        assert "model_providers.{codex_profile}.base_url=" in cmd_str

    def test_no_dangerously_bypass_flag(self):
        cmd = self._case()["command"]
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd

    def test_spawn_agent_prompt_present(self):
        cmd = self._case()["command"]
        prompt = cmd[-1]
        assert "spawn_agent" in prompt
        assert 'agent_type="opencode"' in prompt
        assert 'model="basic"' in prompt

    def test_codex_basic_alias_model_catalog_parses(self):
        catalog_path = (
            ROOT / "scripts" / "local-ci" / "codex_basic_alias_model_catalog.json"
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        assert "models" in catalog
        assert len(catalog["models"]) == 1
        model = catalog["models"][0]
        assert model["slug"] == "basic"
        assert model["multi_agent_version"] == "v2"
        assert model["supports_parallel_tool_calls"] is True

    def test_opencode_agent_toml_exists(self):
        agent_path = ROOT / "scripts" / "local-ci" / "codex_opencode_agent.toml"
        assert agent_path.exists()
        content = agent_path.read_text(encoding="utf-8")
        assert 'name = "opencode"' in content


# ---------------------------------------------------------------------------
# CFG-003 Initiation 2: Fix 3 - Claude parent contract prompt
# ---------------------------------------------------------------------------


class TestClaudeBasicAliasParentContract:
    """The Claude basic-alias parent prompt must treat the child marker as
    opaque and emit exactly BASIC_ALIAS_PARALLEL_TOOLS_PASSED."""

    def _case(self):
        return _config()["cases"][
            "claude_adapter_basic_alias_child_parallel_read_tools"
        ]

    def test_parent_prompt_requires_opaque_marker(self):
        cmd = self._case()["command"]
        prompt = cmd[2]  # -p argument
        assert "Treat that marker as opaque" in prompt
        assert "BASIC_ALIAS_PARALLEL_TOOLS_PASSED" in prompt

    def test_parent_prompt_forbids_surrounding_text(self):
        cmd = self._case()["command"]
        prompt = cmd[2]
        assert "do not summarize it" in prompt
        assert "Markdown" in prompt
        assert "before or after" in prompt

    def test_parent_prompt_has_bounded_blocker_fallback(self):
        cmd = self._case()["command"]
        prompt = cmd[2]
        assert "exact bounded blocker" in prompt

    def test_command_json_checks_require_exact_marker(self):
        checks = self._case()["command_json_checks"]
        assert checks["required_equals"]["result"] == "BASIC_ALIAS_PARALLEL_TOOLS_PASSED"

    def test_child_parallel_read_tools_preserved(self):
        """The child agent contract must still require parallel Read/Glob/Grep."""
        agents = self._case()["claude_agents"]
        child = agents["harness-basic-alias-parallel-read-tools"]
        assert set(child["tools"]) == {"Read", "Glob", "Grep"}
        assert child["model"] == "basic"


# ---------------------------------------------------------------------------
# CFG-003 Initiation 2: Fix 6 - Langfuse credential container separation
# ---------------------------------------------------------------------------


class TestLangfuseCredentialContainerSeparation:
    """Dev profile must use a dedicated Langfuse credential container."""

    def test_langfuse_credential_container_in_dev_profile(self):
        dev = _config()["target_profiles"]["dev"]
        assert dev["langfuse_credential_container"] == "aawm-langfuse-web"
        assert dev["langfuse_public_key_container_env"] == "LANGFUSE_INIT_PROJECT_PUBLIC_KEY"
        assert dev["langfuse_secret_key_container_env"] == "LANGFUSE_INIT_PROJECT_SECRET_KEY"

    def test_backward_compat_fallback_to_docker_container(self, harness, monkeypatch):
        """Profile without langfuse_credential_container falls back to
        docker_container_name."""
        calls = []
        monkeypatch.setattr(
            harness,
            "_resolve_container_env_value",
            lambda container, name: calls.append((container, name)) or "val",
        )
        profile = {
            "docker_container_name": "litellm-dev",
            "langfuse_public_key_container_env": "LANGFUSE_PUBLIC_KEY",
            "langfuse_secret_key_container_env": "LANGFUSE_SECRET_KEY",
        }

        class Args:
            langfuse_query_url = None

        result = harness._resolve_main_credentials(
            config={}, args=Args(), profile=profile
        )
        assert result != 2
        assert all(c[0] == "litellm-dev" for c in calls)

    def test_credential_values_never_logged(self, harness, monkeypatch):
        """Credential values must never appear in stderr output."""
        errors = []
        monkeypatch.setattr(harness, "_emit_stderr", errors.append)
        monkeypatch.setattr(harness, "_resolve_container_env_value", lambda *a: None)
        profile = {
            "langfuse_credential_container": "aawm-langfuse-web",
            "langfuse_public_key_container_env": "LANGFUSE_INIT_PROJECT_PUBLIC_KEY",
            "langfuse_secret_key_container_env": "LANGFUSE_INIT_PROJECT_SECRET_KEY",
        }

        class Args:
            langfuse_query_url = None

        result = harness._resolve_main_credentials(
            config={}, args=Args(), profile=profile
        )
        assert result == 2
        for err in errors:
            assert "LANGFUSE_INIT_PROJECT" not in err or "credential" in err.lower()


# ---------------------------------------------------------------------------
# Validator 019fb548 item 2: session ID + controlled header placeholder
# ---------------------------------------------------------------------------


class TestSessionIdAndHeaderPlaceholderResolution:
    """Unresolved {session_id} and controlled header placeholders must be
    replaced with concrete values; concrete explicit values stay authoritative."""

    def test_http_session_id_placeholder_derived(self, harness):
        """An http_request session_id of '{session_id}' must be replaced with
        a concrete derived session ID."""
        config = {
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "session_id": "{session_id}",
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        sid = result["http_request"]["session_id"]
        assert not harness._is_template_placeholder(sid)
        assert "{" not in sid
        assert result["expected_trace_session_id"] == sid

    def test_http_concrete_session_id_preserved(self, harness):
        """A concrete explicit session_id must remain authoritative."""
        config = {
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "session_id": "my-explicit-session",
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        assert result["http_request"]["session_id"] == "my-explicit-session"

    def test_http_controlled_header_placeholder_replaced(self, harness):
        """A controlled header carrying an unresolved placeholder must be
        replaced, not preserved by setdefault."""
        config = {
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "headers": {
                    "x-litellm-end-user-id": "{harness_user_id}",
                    "session_id": "{session_id}",
                },
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        headers = result["http_request"]["headers"]
        assert not harness._contains_unresolved_placeholder(headers)
        assert "{" not in headers["x-litellm-end-user-id"]
        assert "{" not in headers["session_id"]

    def test_http_concrete_header_preserved(self, harness):
        """A concrete explicit controlled header value remains authoritative."""
        config = {
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "headers": {
                    "x-litellm-end-user-id": "explicit-user-42",
                },
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        # Concrete explicit value preserved (authoritative).
        assert result["http_request"]["headers"]["x-litellm-end-user-id"] == "explicit-user-42"

    def test_cli_session_id_placeholder_derived(self, harness):
        """A CLI expected_trace_session_id of '{session_id}' must be replaced
        with a concrete derived session ID."""
        config = {
            "cli_passthrough": "codex",
            "command": ["codex", "exec", "-m", "basic"],
            "expected_trace_session_id": "{session_id}",
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_cli_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        sid = result["expected_trace_session_id"]
        assert not harness._is_template_placeholder(sid)
        assert "{" not in sid

    def test_cli_concrete_session_id_preserved(self, harness):
        """A concrete explicit CLI session ID remains authoritative."""
        config = {
            "cli_passthrough": "codex",
            "command": ["codex", "exec", "-m", "basic"],
            "expected_trace_session_id": "explicit-cli-session",
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_cli_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        assert result["expected_trace_session_id"] == "explicit-cli-session"

    def test_basic_alias_cases_fully_resolved_after_context(self, harness):
        """Both basic-alias cases must have no unresolved placeholders in
        command, expected_user_ids, session, and headers after resolution."""
        cfg = _config()
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        codex_case = dict(cfg["cases"][
            "native_openai_passthrough_responses_codex_basic_alias_collaboration"
        ])
        codex_result = harness._ensure_cli_harness_context(
            codex_case, profile=profile, target="dev",
            case_name="native_openai_passthrough_responses_codex_basic_alias_collaboration",
        )
        assert not harness._contains_unresolved_placeholder(codex_result["command"])
        assert not harness._contains_unresolved_placeholder(codex_result["expected_user_ids"])
        assert not harness._contains_unresolved_placeholder(
            codex_result["expected_trace_session_id"]
        )


def test_codex_tool_activity_rate_limit_rows_reflect_provider_truth():
    """The tool-activity Codex case expects one seven_day codex:primary row
    (x-codex-primary-window-minutes=10080) and no secondary row."""
    case = _config()["cases"][
        "native_openai_passthrough_responses_codex_tool_activity"
    ]
    rows = case["rate_limit_observations_validation"]["expected_rows"]
    assert [(r["quota_key"], r["required_equals"]["quota_period"]) for r in rows] == [
        ("codex:primary", "seven_day")
    ]


def test_grok_build_cli_uses_live_model_with_grok_build_client():
    """The Grok CLI case invokes/validates grok-4.5 while the client
    identity remains grok-build."""
    case = _config()["cases"]["native_grok_cli_passthrough_grok_build"]
    cmd = case["command"]
    shv = case["session_history_validation"]
    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert shv["expected_model"] == "grok-4.5"
    assert shv["expected_client_name"] == "grok-build"


# ---------------------------------------------------------------------------
# Validator cross-field session resolution: concrete expected_trace_session_id
# wins over an unresolved request session placeholder.
# ---------------------------------------------------------------------------


class TestCrossFieldSessionResolution:
    """When http_request.session_id is unresolved but expected_trace_session_id
    is concrete, the concrete expected ID must be used everywhere."""

    def test_concrete_expected_session_wins_over_request_placeholder(self, harness):
        """Validator finding: expected_trace_session_id='explicit-session-id',
        request session placeholder, placeholder headers -> all resolve to
        'explicit-session-id', not a derived <user>.session."""
        config = {
            "expected_trace_session_id": "explicit-session-id",
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "session_id": "{session_id}",
                "headers": {
                    "session_id": "{session_id}",
                    "x-litellm-end-user-id": "{harness_user_id}",
                },
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        # Request session, controlled header, and expected all use the
        # concrete explicit ID, not a derived user session.
        assert result["http_request"]["session_id"] == "explicit-session-id"
        assert result["expected_trace_session_id"] == "explicit-session-id"
        assert result["http_request"]["headers"]["session_id"] == "explicit-session-id"
        assert not harness._contains_unresolved_placeholder(
            result["http_request"]["headers"]
        )
        # Must NOT be a derived <user>.session value.
        assert not result["http_request"]["session_id"].endswith(".session")

    def test_concrete_request_session_precedence_preserved(self, harness):
        """When both request session and expected are concrete, the request
        session wins (existing contract)."""
        config = {
            "expected_trace_session_id": "expected-concrete",
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "session_id": "request-concrete",
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        assert result["http_request"]["session_id"] == "request-concrete"

    def test_both_placeholder_derives_user_session(self, harness):
        """When neither field is concrete, derive <user>.session."""
        config = {
            "expected_trace_session_id": "{session_id}",
            "http_request": {
                "method": "POST",
                "path": "/chat/completions",
                "session_id": "{session_id}",
            },
        }
        profile = {
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
        }
        result = harness._ensure_http_harness_context(
            config, profile=profile, target="dev", case_name="test_case"
        )
        sid = result["http_request"]["session_id"]
        assert sid.endswith(".session")
        assert not harness._is_template_placeholder(sid)
        assert result["expected_trace_session_id"] == sid
