import importlib.util
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS_PATH = ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location(
        "run_anthropic_adapter_acceptance_test_module",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warning_only_timeout_still_fails_hard():
    harness = _load_harness_module()

    result = harness._warning_only_error_result(
        "claude_adapter_openrouter_ling_26_flash",
        subprocess.TimeoutExpired(["claude"], 180),
        {"warning_only": True},
    )

    assert result["passed"] is False
    assert result["failures"]
    assert result.get("soft_failures") in (None, [])


def test_provider_unavailable_timeout_can_soft_fail_with_exact_log_signature(monkeypatch):
    harness = _load_harness_module()

    def fake_read_runtime_logs_since(**kwargs):
        return (
            {"docker_logs_exit_code": 0, "log_excerpt": "provider unavailable"},
            (
                "OpenRouter adapter upstream attempt 1/4\n"
                "failed with 503 (ProxyException, provider=OpenInference, "
                "raw=no healthy upstream)"
            ),
        )

    monkeypatch.setattr(
        harness,
        "_read_runtime_logs_since",
        fake_read_runtime_logs_since,
    )

    result = harness._provider_unavailable_timeout_error_result(
        "claude_adapter_gpt_oss_120b",
        subprocess.TimeoutExpired(["claude"], 180),
        {
            "soft_fail_timeout_runtime_log_check": {
                "required_substrings": [
                    "OpenRouter adapter upstream attempt",
                    "failed with 503",
                    "provider=OpenInference",
                    "raw=no healthy upstream",
                ]
            },
            "runtime_postconditions": {"docker_container_name": "aawm-litellm"},
        },
        started="2026-04-24T11:12:45+00:00",
    )

    assert result is not None
    assert result["passed"] is True
    assert result["failures"] == []
    assert result["soft_failures"]
    assert result["warnings"]
    assert result["runtime_logs"]["matched_required_substrings"] == [
        "OpenRouter adapter upstream attempt",
        "failed with 503",
        "provider=OpenInference",
        "raw=no healthy upstream",
    ]


def test_provider_unavailable_timeout_stays_hard_without_exact_log_signature(monkeypatch):
    harness = _load_harness_module()

    monkeypatch.setattr(
        harness,
        "_read_runtime_logs_since",
        lambda **kwargs: (
            {"docker_logs_exit_code": 0, "log_excerpt": "adapter traceback"},
            "OpenRouter adapter upstream attempt but local adapter failed",
        ),
    )

    result = harness._provider_unavailable_timeout_error_result(
        "claude_adapter_gpt_oss_120b",
        subprocess.TimeoutExpired(["claude"], 180),
        {
            "soft_fail_timeout_runtime_log_check": {
                "required_substrings": [
                    "OpenRouter adapter upstream attempt",
                    "failed with 503",
                    "provider=OpenInference",
                    "raw=no healthy upstream",
                ]
            },
            "runtime_postconditions": {"docker_container_name": "aawm-litellm"},
        },
        started="2026-04-24T11:12:45+00:00",
    )

    assert result is None


def test_provider_unavailable_command_failure_can_soft_fail_with_exact_log_signature():
    harness = _load_harness_module()

    failures, soft_failures, warnings, runtime_logs = (
        harness._provider_unavailable_failure_soft_fail_result(
            failures=[
                "claude_adapter_gpt_oss_120b command failed",
                "missing claude_adapter_gpt_oss_120b trace name: claude-code.orchestrator",
            ],
            warnings=[],
            config={
                "soft_fail_timeout_runtime_log_check": {
                    "required_substrings": [
                        "OpenRouter adapter upstream attempt",
                        "failed with 503",
                        "provider=OpenInference",
                        "raw=no healthy upstream",
                    ]
                }
            },
            runtime_logs={
                "log_excerpt": (
                    "OpenRouter adapter upstream attempt 1/4\n"
                    "failed with 503 (ProxyException, provider=OpenInference, "
                    "raw=no healthy upstream)"
                )
            },
        )
    )

    assert failures == []
    assert soft_failures == [
        "claude_adapter_gpt_oss_120b command failed",
        "missing claude_adapter_gpt_oss_120b trace name: claude-code.orchestrator",
    ]
    assert warnings
    assert runtime_logs["matched_soft_fail_substrings"] == [
        "OpenRouter adapter upstream attempt",
        "failed with 503",
        "provider=OpenInference",
        "raw=no healthy upstream",
    ]


def test_provider_unavailable_command_failure_does_not_mask_runtime_log_failures():
    harness = _load_harness_module()

    failures, soft_failures, warnings, _ = (
        harness._provider_unavailable_failure_soft_fail_result(
            failures=[
                "claude_adapter_gpt_oss_120b runtime logs contained forbidden substring `KeyError: 'choices'`",
            ],
            warnings=[],
            config={
                "soft_fail_timeout_runtime_log_check": {
                    "required_substrings": [
                        "OpenRouter adapter upstream attempt",
                        "failed with 503",
                        "provider=OpenInference",
                        "raw=no healthy upstream",
                    ]
                }
            },
            runtime_logs={
                "log_excerpt": (
                    "OpenRouter adapter upstream attempt 1/4\n"
                    "failed with 503 (ProxyException, provider=OpenInference, "
                    "raw=no healthy upstream)"
                )
            },
        )
    )

    assert failures
    assert soft_failures == []
    assert warnings == []


def test_warning_only_noncritical_exception_remains_soft():
    harness = _load_harness_module()

    result = harness._warning_only_error_result(
        "optional_provider_case",
        RuntimeError("temporary upstream quality mismatch"),
        {"warning_only": True},
    )

    assert result["passed"] is True
    assert result["failures"] == []
    assert result["soft_failures"]
    assert result["warnings"]


def test_runtime_log_defaults_catch_async_and_content_length_errors(monkeypatch):
    harness = _load_harness_module()

    class Completed:
        returncode = 0
        stdout = (
            "Task exception was never retrieved\n"
            "KeyError: 'choices'\n"
            "h11._util.LocalProtocolError: Too little data for declared Content-Length"
        )
        stderr = ""

    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *args, **kwargs: Completed(),
    )

    summary, failures, warnings = harness._validate_runtime_logs(
        family="claude_adapter_openrouter_ling_26_flash",
        started="2026-04-23T14:06:00+00:00",
        checks={},
        runtime_postconditions={"docker_container_name": "litellm-dev"},
    )

    assert warnings == []
    assert failures
    assert "Task exception was never retrieved" in summary["matched_forbidden_substrings"]
    assert "KeyError: 'choices'" in summary["matched_forbidden_substrings"]
    assert (
        "Too little data for declared Content-Length"
        in summary["matched_forbidden_substrings"]
    )


def test_target_profile_sets_session_history_runtime_identity_expectations(monkeypatch):
    monkeypatch.setenv("AAWM_CLAUDE_HARNESS_USER_ID", "litellm-harness-test")
    harness = _load_harness_module()

    config = {
        "cases": {
            "claude_adapter_gpt55": {
                "env": {"ANTHROPIC_BASE_URL": "placeholder"},
                "session_history_validation": {"expected_provider": "openai"},
            }
        }
    }
    updated = harness._apply_target_profile_to_config(
        config,
        target="dev",
        profile={
            "litellm_base_url": "http://127.0.0.1:4001",
            "anthropic_base_url": "http://127.0.0.1:4001/anthropic",
            "docker_container_name": "litellm-dev",
            "expected_trace_environment": "dev",
        },
    )

    validation = updated["cases"]["claude_adapter_gpt55"][
        "session_history_validation"
    ]
    case_env = updated["cases"]["claude_adapter_gpt55"]["env"]
    assert validation["expected_provider"] == "openai"
    assert validation["expected_litellm_environment"] == "dev"
    assert validation["require_runtime_identity"] is True
    assert updated["cases"]["claude_adapter_gpt55"]["require_trace_user_id"] is True
    assert updated["cases"]["claude_adapter_gpt55"]["expected_user_ids"] == [
        "litellm-harness-test"
    ]
    assert case_env["ANTHROPIC_CUSTOM_HEADERS"] == (
        "x-litellm-end-user-id: litellm-harness-test\n"
        "langfuse_trace_user_id: litellm-harness-test\n"
        "langfuse_trace_name: claude-code"
    )


def test_claude_command_uses_settings_overlay_for_harness_headers(monkeypatch):
    harness = _load_harness_module()
    captured = {}

    class Completed:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(command, **kwargs):
        settings_path = pathlib.Path(command[command.index("--settings") + 1])
        captured["command"] = command
        captured["settings_path"] = settings_path
        captured["settings"] = json.loads(settings_path.read_text())
        captured["exists_during_run"] = settings_path.exists()
        captured["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(harness.RA.subprocess, "run", fake_run)

    result = harness.RA._run_command(
        ["claude", "-p", "hi"],
        extra_env={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4001/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": (
                "x-litellm-end-user-id: litellm-harness-test\n"
                "langfuse_trace_user_id: litellm-harness-test\n"
                "langfuse_trace_name: claude-code"
            ),
            "AAWM_DB_PASSWORD": "not-written-to-settings",
        },
    )

    assert "--settings" in captured["command"]
    assert captured["exists_during_run"] is True
    assert captured["settings"] == {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4001/anthropic",
            "ANTHROPIC_CUSTOM_HEADERS": (
                "x-litellm-end-user-id: litellm-harness-test\n"
                "langfuse_trace_user_id: litellm-harness-test\n"
                "langfuse_trace_name: claude-code"
            ),
        }
    }
    assert "AAWM_DB_PASSWORD" not in captured["settings"]["env"]
    assert captured["env"]["AAWM_DB_PASSWORD"] == "not-written-to-settings"
    assert not captured["settings_path"].exists()
    assert result["command"] == captured["command"]
