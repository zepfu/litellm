"""RR-082 residuals for scripts/local-ci/run_anthropic_adapter_acceptance.py.

Findings covered:
  1. Bound all docker subprocess invocations with timeout=
  2. Tighten unrelated runtime-log ignore so missing attribution alone is insufficient
  3. Scope provider-unavailable soft-fail to connectivity/timeout-class failures
  4. Portable Claude projects root via Path.home() fallback
  5. Surface skipped-case summary and opt-in fail_on_skip
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"


def _load_module():
    name = "run_anthropic_adapter_acceptance_rr082"
    # Fresh load so constants/helpers reflect current file contents.
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness():
    return _load_module()


# --- Finding 1: docker subprocess timeouts ---------------------------------


def test_should_pass_timeout_to_docker_status_subprocess(harness, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Completed:
        returncode = 0
        stdout = "Up 3 minutes"
        stderr = ""

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    assert harness._docker_status_for_container("litellm-dev") == "Up 3 minutes"
    assert (
        captured["kwargs"]["timeout"]
        == harness.DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS
    )


def test_should_return_empty_status_when_docker_status_times_out(
    harness, monkeypatch
) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else "docker", timeout=kwargs.get("timeout")
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    assert harness._docker_status_for_container("litellm-dev") == ""


def test_should_pass_timeout_to_runtime_postcondition_docker_ps(
    harness, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    class Completed:
        returncode = 0
        stdout = "Up 1 minute"
        stderr = ""

    def fake_run(*args, **kwargs):
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    # Avoid real healthcheck network call.
    monkeypatch.setattr(
        harness.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip health")),
    )

    summary, failures = harness._validate_runtime_postcondition(
        family="case",
        litellm_base_url="http://127.0.0.1:9",
        checks={"docker_container_name": "litellm-dev"},
    )
    assert (
        captured["kwargs"]["timeout"]
        == harness.DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS
    )
    assert summary["docker_status"] == "Up 1 minute"
    assert any("healthcheck failed" in f for f in failures)


def test_should_pass_timeout_to_docker_logs_and_surface_timeout_exit(
    harness, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(*args, **kwargs):
        captured["kwargs"] = kwargs
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else "docker", timeout=kwargs.get("timeout")
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    summary, log_text = harness._read_runtime_logs_since(
        started="2026-07-17T00:00:00+00:00",
        checks={"docker_container_name": "litellm-dev", "tail_lines": 10},
        runtime_postconditions={},
    )
    assert (
        captured["kwargs"]["timeout"]
        == harness.DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_SECONDS
    )
    assert summary["docker_logs_exit_code"] == 124
    assert "timed out" in summary.get("docker_logs_error", "")
    assert log_text == ""


# --- Finding 2: unrelated runtime-log ignore tightening ---------------------


def test_should_keep_upstream_error_hard_without_foreign_model_evidence(
    harness,
) -> None:
    """Missing attribution + unrelated signature alone must not soft-ignore 503."""
    context = (
        "pass_through_endpoint(): Exception occured - 503: b'upstream "
        "connect error or disconnect/reset before headers. reset reason: "
        "connection timeout'\n"
        "https://chatgpt.com/backend-api/codex/responses\n"
    )
    assert (
        harness._is_unrelated_runtime_log_match(
            substring="pass_through_endpoint(): Exception occured - 503:",
            context=context,
            attribution_substrings=[
                "claude_adapter_spark",
                "gpt-5.3-codex-spark",
                "active-spark-session",
            ],
        )
        is False
    )


def test_should_ignore_upstream_error_only_with_foreign_model_near_match(
    harness,
) -> None:
    context = (
        'Langfuse warning: {"model": "gpt-5.5"}\n'
        "pass_through_endpoint(): Exception occured - 503: b'upstream "
        "connect error or disconnect/reset before headers. reset reason: "
        "connection timeout'\n"
        "https://chatgpt.com/backend-api/codex/responses\n"
    )
    assert (
        harness._is_unrelated_runtime_log_match(
            substring="pass_through_endpoint(): Exception occured - 503:",
            context=context,
            attribution_substrings=[
                "claude_adapter_spark",
                "gpt-5.3-codex-spark",
                "active-spark-session",
            ],
        )
        is True
    )


def test_should_keep_match_hard_when_attribution_present_even_with_foreign_noise(
    harness,
) -> None:
    context = (
        'Langfuse warning: {"model": "gpt-5.5"}\n'
        "session=active-spark-session model=gpt-5.3-codex-spark\n"
        "pass_through_endpoint(): Exception occured - 503: b'upstream "
        "connect error or disconnect/reset before headers. reset reason: "
        "connection timeout'\n"
        "https://chatgpt.com/backend-api/codex/responses\n"
    )
    assert (
        harness._is_unrelated_runtime_log_match(
            substring="pass_through_endpoint(): Exception occured - 503:",
            context=context,
            attribution_substrings=[
                "claude_adapter_spark",
                "gpt-5.3-codex-spark",
                "active-spark-session",
            ],
        )
        is False
    )


# --- Finding 3: scoped provider-unavailable soft-fail -----------------------


def test_should_soft_fail_only_connectivity_class_provider_unavailable_failures(
    harness,
) -> None:
    (
        failures,
        soft_failures,
        warnings,
        runtime_logs,
    ) = harness._provider_unavailable_failure_soft_fail_result(
        failures=[
            "claude_adapter_gpt_oss_120b command failed",
            "missing claude_adapter_gpt_oss_120b trace name: claude-code.orchestrator",
            "claude_adapter_gpt_oss_120b session_history model mismatch: expected `x`, got `y`",
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

    assert soft_failures == ["claude_adapter_gpt_oss_120b command failed"]
    assert (
        "missing claude_adapter_gpt_oss_120b trace name: claude-code.orchestrator"
        in failures
    )
    assert any("session_history model mismatch" in f for f in failures)
    assert any("provider-unavailable soft-fail" in w for w in warnings)
    assert runtime_logs["matched_soft_fail_substrings"]


def test_should_still_block_provider_unavailable_soft_fail_on_forbidden_runtime_logs(
    harness,
) -> None:
    (
        failures,
        soft_failures,
        warnings,
        _,
    ) = harness._provider_unavailable_failure_soft_fail_result(
        failures=[
            "claude_adapter_gpt_oss_120b command failed",
            "claude_adapter_gpt_oss_120b runtime logs contained forbidden substring `KeyError: 'choices'`",
        ],
        warnings=[],
        config={
            "soft_fail_timeout_runtime_log_check": {
                "required_substrings": [
                    "OpenRouter adapter upstream attempt",
                    "failed with 503",
                ]
            }
        },
        runtime_logs={
            "log_excerpt": ("OpenRouter adapter upstream attempt 1/4\nfailed with 503")
        },
    )
    assert soft_failures == []
    assert len(failures) == 2
    assert warnings == []


# --- Finding 4: portable Claude projects root -------------------------------


def test_should_use_path_home_for_claude_projects_root_default(
    harness, monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("CLAUDE_PROJECTS_ROOT", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECTS_DIR", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(
        harness.pathlib.Path, "home", classmethod(lambda cls: fake_home)
    )

    root = harness._claude_projects_root({})
    assert root == fake_home / ".claude" / "projects"
    assert "/home/zepfu/.claude/projects" not in str(root)


def test_should_prefer_env_and_config_over_home_fallback(
    harness, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CLAUDE_PROJECTS_ROOT", str(tmp_path / "from-env"))
    assert harness._claude_projects_root({}) == tmp_path / "from-env"

    configured = tmp_path / "from-config"
    assert (
        harness._claude_projects_root({"claude_projects_root": str(configured)})
        == configured
    )


# --- Finding 5: skip summary + fail_on_skip ---------------------------------


def test_should_include_skipped_count_in_summary(harness) -> None:
    summary = harness._build_summary(
        {
            "case_a": {
                "passed": True,
                "skipped": True,
                "failures": [],
                "warnings": ["missing required env: FOO"],
            },
            "case_b": {
                "passed": True,
                "skipped": False,
                "failures": [],
                "warnings": [],
            },
            "case_c": {
                "passed": False,
                "skipped": False,
                "failures": ["boom"],
                "warnings": [],
            },
        }
    )
    assert summary["skipped_count"] == 1
    assert summary["skipped_cases"] == ["case_a"]
    assert summary["passed"] is False
    assert summary["failures"] == ["case_c: boom"]
    assert any("missing required env" in w for w in summary["warnings"])


def test_should_fail_missing_required_env_when_fail_on_skip_case_flag(
    harness, monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "cfg.json"
    artifact_path = tmp_path / "out.json"
    config_path.write_text(
        """
{
  "default_target_profile": "dev",
  "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
  "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
  "cases": {
    "needs_secret": {
      "required_env": ["RR082_REQUIRED_SECRET"],
      "fail_on_skip": true
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delenv("RR082_REQUIRED_SECRET", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_anthropic_adapter_acceptance.py",
            "--config",
            str(config_path),
            "--write-artifact",
            str(artifact_path),
            "--cases",
            "needs_secret",
        ],
    )
    monkeypatch.setattr(harness, "_docker_status_for_container", lambda name: "Up")
    monkeypatch.setattr(harness, "_load_dotenv_into_environment", lambda path: None)

    exit_code = harness.main()
    assert exit_code == 1
    artifact = __import__("json").loads(artifact_path.read_text(encoding="utf-8"))
    result = artifact["results"]["needs_secret"]
    assert result["skipped"] is True
    assert result["passed"] is False
    assert any("missing required env" in f for f in result["failures"])
    assert artifact["summary"]["skipped_count"] == 1
    assert artifact["summary"]["skipped_cases"] == ["needs_secret"]
    assert artifact["summary"]["passed"] is False


def test_should_soft_skip_missing_required_env_without_fail_on_skip(
    harness, monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "cfg.json"
    artifact_path = tmp_path / "out.json"
    config_path.write_text(
        """
{
  "default_target_profile": "dev",
  "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
  "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
  "cases": {
    "optional_secret": {
      "required_env": ["RR082_OPTIONAL_SECRET"]
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delenv("RR082_OPTIONAL_SECRET", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_anthropic_adapter_acceptance.py",
            "--config",
            str(config_path),
            "--write-artifact",
            str(artifact_path),
            "--cases",
            "optional_secret",
        ],
    )
    monkeypatch.setattr(harness, "_docker_status_for_container", lambda name: "Up")
    monkeypatch.setattr(harness, "_load_dotenv_into_environment", lambda path: None)

    exit_code = harness.main()
    assert exit_code == 0
    artifact = __import__("json").loads(artifact_path.read_text(encoding="utf-8"))
    result = artifact["results"]["optional_secret"]
    assert result["skipped"] is True
    assert result["passed"] is True
    assert result["failures"] == []
    assert any("missing required env" in w for w in result["warnings"])
    assert artifact["summary"]["skipped_count"] == 1
    assert artifact["summary"]["passed"] is True


def test_should_mark_fail_on_skip_verification_status_failed(harness) -> None:
    assert (
        harness._verification_status_for_case(
            {
                "passed": False,
                "skipped": True,
                "failures": ["missing required env: FOO"],
            }
        )
        == "failed"
    )
    assert (
        harness._verification_status_for_case(
            {
                "passed": True,
                "skipped": True,
                "failures": [],
                "warnings": ["missing required env: FOO"],
            }
        )
        == "skipped"
    )
