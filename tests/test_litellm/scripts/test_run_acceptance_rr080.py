"""RR-080 residuals for scripts/local-ci/run_acceptance.py.

Covers:
1. Scrubbed child CLI environment (no Langfuse/DB secrets).
2. Wiring of skip_generation_quality_checks / allow_zero_cost config flags.
3. stdout/stderr size cap on captured CLI output.
4. Enforcement of minimum_trace_count when present in family config.
Also coordinates with RR-077/RR-079 portable @{config_dir} path expansion at load time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "local-ci" / "run_acceptance.py"


def _load_module():
    name = "run_acceptance_rr080"
    # Reload-friendly unique name per process is fine; overwrite for isolation.
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ra():
    return _load_module()


# ---------------------------------------------------------------------------
# Finding #1: scrubbed child env
# ---------------------------------------------------------------------------


def test_should_scrub_langfuse_and_db_secrets_from_child_env(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-should-not-leak")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-should-not-leak")
    monkeypatch.setenv("LANGFUSE_QUERY_URL", "http://127.0.0.1:3000")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost/db")
    monkeypatch.setenv("AAWM_DB_PASSWORD", "db-pass")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pg-pass")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:4000/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ok-for-cli")
    monkeypatch.setenv("UNRELATED_RANDOM", "nope")

    env = ra._scrubbed_child_env(
        {"ANTHROPIC_CUSTOM_HEADERS": "x-litellm-end-user-id: harness"}
    )
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/test"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000/anthropic"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-ok-for-cli"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "x-litellm-end-user-id: harness"
    for denied in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_QUERY_URL",
        "DATABASE_URL",
        "AAWM_DB_PASSWORD",
        "POSTGRES_PASSWORD",
        "UNRELATED_RANDOM",
    ):
        assert denied not in env, f"{denied} must not reach child CLI env"


def test_should_not_allow_extra_env_to_reintroduce_denied_secrets(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    env = ra._scrubbed_child_env(
        {
            "LANGFUSE_SECRET_KEY": "sk-injected",
            "DATABASE_URL": "postgres://injected",
            "ANTHROPIC_BASE_URL": "http://example/anthropic",
        }
    )
    assert "LANGFUSE_SECRET_KEY" not in env
    assert "DATABASE_URL" not in env
    assert env["ANTHROPIC_BASE_URL"] == "http://example/anthropic"


def test_should_deny_litellm_admin_secrets_but_allow_routing_vars(
    ra, monkeypatch
) -> None:
    """LiteLLM proxy admin secrets must not inherit; non-secret routing may.

    Provider prefixes (ANTHROPIC_/OPENAI_/…) intentionally allow env-based CLI
    auth tokens. LITELLM_* is default-deny with a narrow non-secret allowlist so
    LITELLM_MASTER_KEY cannot bypass the SECRET substring path via a trusted
    prefix exemption.
    """
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/test")
    # Non-secret LiteLLM routing / logging knobs — allowed.
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4001")
    monkeypatch.setenv("LITELLM_API_BASE", "http://127.0.0.1:4001/v1")
    monkeypatch.setenv("LITELLM_LOG", "INFO")
    monkeypatch.setenv("LITELLM_MODE", "PRODUCTION")
    # Proxy admin / secret-bearing LiteLLM material — denied.
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-litellm-master")
    monkeypatch.setenv("LITELLM_SALT_KEY", "salt-should-not-leak")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm-api")  # not on allowlist
    monkeypatch.setenv("LITELLM_FOO_TOKEN", "tok")
    monkeypatch.setenv("LITELLM_DB_PASSWORD", "pw")
    monkeypatch.setenv("LITELLM_SOME_SECRET", "sec")
    # Langfuse + DB still denied.
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    # Provider auth intentionally allowed (env-based CLI credentials).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex")

    env = ra._scrubbed_child_env()
    assert env.get("LITELLM_BASE_URL") == "http://127.0.0.1:4001"
    assert env.get("LITELLM_API_BASE") == "http://127.0.0.1:4001/v1"
    assert env.get("LITELLM_LOG") == "INFO"
    assert env.get("LITELLM_MODE") == "PRODUCTION"
    for denied in (
        "LITELLM_MASTER_KEY",
        "LITELLM_SALT_KEY",
        "LITELLM_API_KEY",
        "LITELLM_FOO_TOKEN",
        "LITELLM_DB_PASSWORD",
        "LITELLM_SOME_SECRET",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "DATABASE_URL",
    ):
        assert denied not in env, f"{denied} must not reach child CLI env"
    # Provider prefixes remain intentionally allowed for CLI auth.
    assert env.get("ANTHROPIC_API_KEY") == "sk-ant"
    assert env.get("OPENAI_API_KEY") == "sk-openai"
    assert env.get("CODEX_API_KEY") == "sk-codex"


def test_should_deny_litellm_secrets_even_when_passed_via_extra_env(
    ra, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    env = ra._scrubbed_child_env(
        {
            "LITELLM_MASTER_KEY": "sk-master-injected",
            "LITELLM_SALT_KEY": "salt-injected",
            "LITELLM_FOO_TOKEN": "tok-injected",
            "LITELLM_DB_PASSWORD": "pw-injected",
            "LANGFUSE_SECRET_KEY": "sk-lf-injected",
            "DATABASE_URL": "postgres://injected",
            "LITELLM_BASE_URL": "http://127.0.0.1:4001",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000/anthropic",
        }
    )
    for denied in (
        "LITELLM_MASTER_KEY",
        "LITELLM_SALT_KEY",
        "LITELLM_FOO_TOKEN",
        "LITELLM_DB_PASSWORD",
        "LANGFUSE_SECRET_KEY",
        "DATABASE_URL",
    ):
        assert denied not in env, f"extra_env must not reintroduce {denied}"
    assert env["LITELLM_BASE_URL"] == "http://127.0.0.1:4001"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000/anthropic"


def test_provider_prefix_auth_allowance_is_intentional_not_litellm_admin(
    ra, monkeypatch
) -> None:
    """Document: provider API keys allowed; LiteLLM master key is not a provider key."""
    monkeypatch.setenv("PATH", "/usr/bin")
    assert "LITELLM_" not in ra._CHILD_ENV_ALLOW_PREFIXES
    assert "LITELLM_MASTER_KEY" in ra._CHILD_ENV_DENY_KEYS or ra._is_denied_child_env_key(
        "LITELLM_MASTER_KEY"
    )
    assert ra._is_denied_child_env_key("LITELLM_MASTER_KEY")
    assert ra._is_denied_child_env_key("LITELLM_RANDOM_SECRET")
    assert ra._is_denied_child_env_key("LITELLM_ANYTHING_NOT_ALLOWLISTED")
    assert not ra._is_denied_child_env_key("LITELLM_BASE_URL")
    assert ra._is_allowed_child_env_key("LITELLM_BASE_URL")
    # Provider auth keys: allowed even though they contain key material.
    assert not ra._is_denied_child_env_key("ANTHROPIC_API_KEY")
    assert ra._is_allowed_child_env_key("ANTHROPIC_API_KEY")
    assert not ra._is_denied_child_env_key("OPENAI_API_KEY")
    assert ra._is_allowed_child_env_key("OPENAI_API_KEY")


def test_run_command_should_pass_scrubbed_env_to_subprocess(ra, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-should-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("HOME", str(tmp_path))

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["env"] = kwargs.get("env")
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    with patch.object(ra.subprocess, "run", side_effect=fake_run):
        result = ra._run_command(
            ["codex", "exec", "-p", "hi"],
            extra_env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:9/anthropic"},
            timeout_seconds=5,
        )

    assert result["exit_code"] == 0
    env = captured["env"]
    assert env is not None
    assert "LANGFUSE_SECRET_KEY" not in env
    assert "DATABASE_URL" not in env
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9/anthropic"
    assert env["PATH"] == "/usr/bin"
    # Must not be a full os.environ.copy()
    assert set(env.keys()) <= (
        set(ra._CHILD_ENV_BASE_KEYS)
        | set(ra._CHILD_ENV_ALLOW_KEYS)
        | {k for k in env if any(k.startswith(p) for p in ra._CHILD_ENV_ALLOW_PREFIXES)}
    )


# ---------------------------------------------------------------------------
# Finding #2: generation quality flags wired from config
# ---------------------------------------------------------------------------


def test_generation_quality_flags_read_config_keys(ra) -> None:
    assert ra._generation_quality_flags({}) == (False, False)
    assert ra._generation_quality_flags({"allow_zero_cost": True}) == (False, True)
    assert ra._generation_quality_flags(
        {"skip_generation_quality_checks": True}
    ) == (True, False)
    assert ra._generation_quality_flags(
        {"skip_quality_checks": True, "allow_zero_cost": 1}
    ) == (True, True)


def test_validate_generation_observations_honors_allow_zero_cost(ra) -> None:
    observation = {
        "id": "g1",
        "traceId": "t1",
        "name": "gen",
        "model": "gpt-test",
        "promptTokens": 1,
        "completionTokens": 1,
        "totalTokens": 2,
        "costDetails": {"total": 0},
        "calculatedTotalCost": 0,
    }

    def fake_recent(**kwargs):  # noqa: ANN003
        return [observation]

    with patch.object(
        ra, "_recent_langfuse_generation_observations_for_trace_ids", fake_recent
    ):
        _obs, _summaries, failures = ra._validate_generation_observations(
            family="codex",
            query_url="http://lf",
            public_key="pk",
            secret_key="sk",
            trace_ids=["t1"],
            start_time=ra._utcnow(),
            allow_zero_cost=False,
        )
        assert any("costDetails.total" in f for f in failures)

        _obs, _summaries, failures_ok = ra._validate_generation_observations(
            family="codex",
            query_url="http://lf",
            public_key="pk",
            secret_key="sk",
            trace_ids=["t1"],
            start_time=ra._utcnow(),
            allow_zero_cost=True,
        )
        assert not any("costDetails.total" in f for f in failures_ok)

        _obs, _summaries, failures_skip = ra._validate_generation_observations(
            family="codex",
            query_url="http://lf",
            public_key="pk",
            secret_key="sk",
            trace_ids=["t1"],
            start_time=ra._utcnow(),
            skip_quality_checks=True,
        )
        assert failures_skip == []


# ---------------------------------------------------------------------------
# Finding #3: stdout/stderr size cap
# ---------------------------------------------------------------------------


def test_truncate_captured_text_marks_and_limits(ra) -> None:
    text, truncated = ra._truncate_captured_text("x" * 100, max_chars=40)
    assert truncated is True
    assert len(text) <= 40
    assert "truncated" in text

    text2, truncated2 = ra._truncate_captured_text("short", max_chars=40)
    assert truncated2 is False
    assert text2 == "short"


def test_run_command_truncates_large_stdout_and_stderr(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp")
    monkeypatch.delenv("ACCEPTANCE_CLI_OUTPUT_MAX_CHARS", raising=False)

    huge = "A" * 5000
    huge_err = "B" * 5000

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=0, stdout=huge, stderr=huge_err)

    with patch.object(ra.subprocess, "run", side_effect=fake_run):
        result = ra._run_command(
            ["echo", "x"],
            timeout_seconds=5,
            output_max_chars=200,
        )

    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert result["stdout_original_chars"] == 5000
    assert result["stderr_original_chars"] == 5000
    assert len(result["stdout"]) <= 200
    assert len(result["stderr"]) <= 200
    assert "truncated" in result["stdout"]
    assert result["output_max_chars"] == 200


# ---------------------------------------------------------------------------
# Finding #4: minimum_trace_count enforcement
# ---------------------------------------------------------------------------


def test_enforce_minimum_trace_count(ra) -> None:
    traces = [{"id": "1"}, {"id": "2"}]
    assert ra._enforce_minimum_trace_count(
        family="claude", traces=traces, config={}
    ) == []
    assert ra._enforce_minimum_trace_count(
        family="claude", traces=traces, config={"minimum_trace_count": 2}
    ) == []
    failures = ra._enforce_minimum_trace_count(
        family="claude", traces=traces, config={"minimum_trace_count": 5}
    )
    assert len(failures) == 1
    assert "minimum_trace_count 5" in failures[0]
    assert "trace count 2" in failures[0]


def test_validate_codex_applies_minimum_trace_count(ra) -> None:
    config = {
        "command": ["codex", "exec", "-p", "hi"],
        "timeout_seconds": 5,
        "expected_trace_names": [],
        "expected_user_ids": [],
        "minimum_trace_count": 3,
        "allowed_generation_routes": [],
    }

    def fake_run(command, **kwargs):  # noqa: ANN001
        return {
            "command": command,
            "command_string": "codex",
            "exit_code": 0,
            "duration_seconds": 0.1,
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_original_chars": 0,
            "stderr_original_chars": 0,
            "output_max_chars": 200_000,
            "response_excerpt": "",
        }

    with (
        patch.object(ra, "_run_command", side_effect=fake_run),
        patch.object(ra, "_poll_langfuse_named_traces", return_value=[]),
        patch.object(
            ra,
            "_validate_generation_observations",
            return_value=([], [], []),
        ),
        patch.object(
            ra,
            "_validate_trace_enrichment",
            return_value=({}, [], []),
        ),
        patch.object(ra, "_validate_trace_context", return_value=({}, [])),
        patch.object(ra, "_validate_generation_metadata", return_value=({}, [])),
        patch.object(
            ra, "_validate_span_observations", return_value=([], [], [])
        ),
    ):
        result = ra._validate_codex(
            config,
            query_url="http://lf",
            public_key="pk",
            secret_key="sk",
        )

    assert result["passed"] is False
    assert any("minimum_trace_count" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# Portable @{config_dir} expansion (coordinate with RR-077/RR-079)
# ---------------------------------------------------------------------------


def test_should_expand_config_dir_placeholder_when_loading_suite(ra, tmp_path: Path) -> None:
    prompt = tmp_path / "claude_acceptance_prompt.txt"
    prompt.write_text("hello", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "claude": {
                    "command": [
                        "claude",
                        "-p",
                        "@{config_dir}/claude_acceptance_prompt.txt",
                        "--output-format",
                        "json",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = ra._load_suite_config(config_path)
    expanded = loaded["claude"]["command"][2]
    assert expanded == f"@{prompt.resolve()}"
    assert "{config_dir}" not in expanded
    # Relative @path also expands
    loaded2 = ra._rewrite_config_path_tokens(
        {"command": ["claude", "-p", "@claude_acceptance_prompt.txt"]},
        tmp_path,
    )
    assert loaded2["command"][2] == f"@{prompt.resolve()}"
    # Absolute @ remains unchanged
    abs_token = "@/tmp/elsewhere/prompt.txt"
    assert ra._expand_at_path_token(abs_token, tmp_path) == abs_token
