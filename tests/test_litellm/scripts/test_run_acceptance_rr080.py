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
import subprocess
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


# ---------------------------------------------------------------------------
# D1-574/MS-033: Target profile rewriting (dev/prod)
# ---------------------------------------------------------------------------


def _base_config() -> dict:
    """Minimal config with all three families for target rewriting tests."""
    return {
        "suite_version": 1,
        "litellm_base_url": "http://127.0.0.1:4000",
        "codex": {
            "command": ["codex", "exec", "-p", "litellm", "--json", "hi"],
            "expected_trace_environment": "prod",
        },
        "claude": {
            "command": ["claude", "-p", "prompt.txt", "--output-format", "json"],
            "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000/anthropic"},
            "expected_trace_environment": "prod",
            "fanout_modes": {
                "minimal": {
                    "command": ["claude", "-p", "prompt.txt"],
                    "expected_trace_environment": "prod",
                },
            },
        },
        "gemini": {
            "command": ["gemini", "-p", "hi", "--output-format", "json"],
            "expected_trace_environment": "prod",
        },
    }


def test_dev_target_rewrites_all_families(ra) -> None:
    config = _base_config()
    profile = ra._resolve_target_profile(config, target="dev")
    assert profile is not None
    assert profile["codex_profile"] == "litellm-dev"
    assert profile["expected_trace_environment"] == "dev"

    ra._apply_target_profile(config, profile)

    # Codex: -p litellm -> -p litellm-dev
    codex_cmd = config["codex"]["command"]
    p_idx = codex_cmd.index("-p")
    assert codex_cmd[p_idx + 1] == "litellm-dev"
    assert config["codex"]["expected_trace_environment"] == "dev"

    # Claude: ANTHROPIC_BASE_URL -> :4001
    assert (
        config["claude"]["env"]["ANTHROPIC_BASE_URL"]
        == "http://127.0.0.1:4001/anthropic"
    )
    assert config["claude"]["expected_trace_environment"] == "dev"
    assert (
        config["claude"]["fanout_modes"]["minimal"]["expected_trace_environment"]
        == "dev"
    )

    # Gemini: CODE_ASSIST_ENDPOINT injected
    assert (
        config["gemini"]["env"]["CODE_ASSIST_ENDPOINT"]
        == "http://127.0.0.1:4001/gemini"
    )
    assert config["gemini"]["expected_trace_environment"] == "dev"

    # Top-level base URL rewritten
    assert config["litellm_base_url"] == "http://127.0.0.1:4001"


def test_prod_target_preserves_prod_routes(ra) -> None:
    config = _base_config()
    profile = ra._resolve_target_profile(config, target="prod")
    assert profile is not None
    assert profile["codex_profile"] == "litellm"

    ra._apply_target_profile(config, profile)

    codex_cmd = config["codex"]["command"]
    p_idx = codex_cmd.index("-p")
    assert codex_cmd[p_idx + 1] == "litellm"
    assert config["codex"]["expected_trace_environment"] == "prod"
    assert (
        config["claude"]["env"]["ANTHROPIC_BASE_URL"]
        == "http://127.0.0.1:4000/anthropic"
    )
    assert (
        config["gemini"]["env"]["CODE_ASSIST_ENDPOINT"]
        == "http://127.0.0.1:4000/gemini"
    )
    assert config["litellm_base_url"] == "http://127.0.0.1:4000"


def test_unknown_effective_url_fails_closed(ra, monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    config["litellm_base_url"] = "http://10.9.9.9:9999"
    with pytest.raises(SystemExit, match="does not match any configured"):
        ra._resolve_target_profile(config, target=None)


def test_unknown_target_raises(ra) -> None:
    config = _base_config()
    with pytest.raises(SystemExit, match="Unknown acceptance target"):
        ra._resolve_target_profile(config, target="staging")


def test_config_target_profiles_override_built_in(ra) -> None:
    config = _base_config()
    config["target_profiles"] = {
        "dev": {
            "litellm_base_url": "http://10.0.0.1:4001",
            "anthropic_base_url": "http://10.0.0.1:4001/anthropic",
            "gemini_base_url": "http://10.0.0.1:4001/gemini",
            "codex_profile": "custom-dev",
            "docker_container_name": "custom-dev",
            "expected_trace_environment": "dev",
        },
    }
    profile = ra._resolve_target_profile(config, target="dev")
    assert profile is not None
    assert profile["codex_profile"] == "custom-dev"
    assert profile["litellm_base_url"] == "http://10.0.0.1:4001"


# ---------------------------------------------------------------------------
# D1-574/MS-033: Langfuse credential fail-closed from target container
# ---------------------------------------------------------------------------


def test_langfuse_credentials_from_container_fail_closed(ra, monkeypatch) -> None:
    """When credential_source=target_container, missing container creds must
    raise SystemExit, not fall back to populated host env vars."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-stale-host")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-stale-host")

    config = {
        "langfuse_credential_source": "target_container",
        "langfuse_public_key_env": "LANGFUSE_PUBLIC_KEY",
        "langfuse_secret_key_env": "LANGFUSE_SECRET_KEY",
    }
    profile = {"docker_container_name": "litellm-dev"}

    # Container returns nothing
    monkeypatch.setattr(
        ra, "_resolve_container_env_value", lambda c, e, **kwargs: None
    )

    with pytest.raises(SystemExit, match="Refusing to fall back"):
        ra._resolve_langfuse_credentials(config, profile)


def test_langfuse_credentials_from_container_success(ra, monkeypatch) -> None:
    """When container has credentials, they are used instead of host env."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-stale-host")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-stale-host")

    config = {
        "langfuse_credential_source": "target_container",
    }
    profile = {"docker_container_name": "litellm-dev"}

    container_values = {
        ("litellm-dev", "LANGFUSE_PUBLIC_KEY"): "pk-container",
        ("litellm-dev", "LANGFUSE_SECRET_KEY"): "sk-container",
    }
    monkeypatch.setattr(
        ra,
        "_resolve_container_env_value",
        lambda c, e, **kwargs: container_values.get((c, e)),
    )

    pk, sk = ra._resolve_langfuse_credentials(config, profile)
    assert pk == "pk-container"
    assert sk == "sk-container"


def test_langfuse_credentials_host_env_default(ra, monkeypatch) -> None:
    """Without target_container source, host env is used (existing behavior)."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-host")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-host")

    config = {}
    pk, sk = ra._resolve_langfuse_credentials(config, None)
    assert pk == "pk-host"
    assert sk == "sk-host"


def test_langfuse_credentials_host_env_missing_raises(ra, monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    config = {}
    with pytest.raises(SystemExit, match="Missing Langfuse credentials"):
        ra._resolve_langfuse_credentials(config, None)


# ---------------------------------------------------------------------------
# D1-574/MS-033: TimeoutExpired partial-output evidence retention
# ---------------------------------------------------------------------------


def test_run_command_timeout_preserves_partial_output(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp")

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=5,
            output="partial stdout here",
            stderr="partial stderr here",
        )

    with patch.object(ra.subprocess, "run", side_effect=fake_run):
        result = ra._run_command(["codex", "exec", "-p", "hi"], timeout_seconds=5)

    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 5
    assert result["exit_code"] == -1
    assert "partial stdout here" in result["stdout"]
    assert "partial stderr here" in result["stderr"]
    assert result["command"] == ["codex", "exec", "-p", "hi"]
    assert result["duration_seconds"] >= 0


def test_run_command_timeout_empty_partial(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp")

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=10)

    with patch.object(ra.subprocess, "run", side_effect=fake_run):
        result = ra._run_command(["gemini", "-p", "hi"], timeout_seconds=10)

    assert result["timed_out"] is True
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["exit_code"] == -1


# ---------------------------------------------------------------------------
# D1-574/MS-033: Post-subprocess observability failure retains run evidence
# ---------------------------------------------------------------------------


def test_validate_codex_timeout_produces_precise_failure(ra) -> None:
    """When codex times out, the result has timed_out evidence and a precise
    failure message rather than a generic family error."""
    config = {
        "command": ["codex", "exec", "-p", "litellm-dev", "--json", "hi"],
        "timeout_seconds": 5,
        "expected_trace_names": [],
        "expected_user_ids": [],
    }

    timeout_run = {
        "command": config["command"],
        "command_string": "codex exec",
        "exit_code": -1,
        "duration_seconds": 5.0,
        "stdout": "partial",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_original_chars": 7,
        "stderr_original_chars": 0,
        "output_max_chars": 200_000,
        "response_excerpt": "partial",
        "timed_out": True,
        "timeout_seconds": 5,
    }

    with (
        patch.object(ra, "_run_command", return_value=timeout_run),
        patch.object(ra, "_poll_langfuse_named_traces", return_value=[]),
        patch.object(ra, "_validate_generation_observations", return_value=([], [], [])),
        patch.object(ra, "_validate_trace_enrichment", return_value=({}, [], [])),
        patch.object(ra, "_validate_trace_context", return_value=({}, [])),
        patch.object(ra, "_validate_generation_metadata", return_value=({}, [])),
        patch.object(ra, "_validate_span_observations", return_value=([], [], [])),
    ):
        result = ra._validate_codex(
            config, query_url="http://lf", public_key="pk", secret_key="sk"
        )

    assert result["passed"] is False
    assert any("timed out after 5s" in f for f in result["failures"])
    assert result["timed_out"] is True
    assert result["stdout"] == "partial"
    assert result["command"] == config["command"]


# ---------------------------------------------------------------------------
# D1-574/MS-033 (review): URL-inferred target selection
# ---------------------------------------------------------------------------


def test_url_override_4001_infers_dev_target(ra, monkeypatch) -> None:
    """Exact command: LITELLM_BASE_URL=http://127.0.0.1:4001 with no explicit
    target must still select the dev profile and rewrite all families."""
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4001")
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    profile = ra._resolve_target_profile(config, target=None)
    assert profile is not None
    assert profile["expected_trace_environment"] == "dev"
    assert profile["codex_profile"] == "litellm-dev"

    ra._apply_target_profile(config, profile)
    codex_cmd = config["codex"]["command"]
    assert codex_cmd[codex_cmd.index("-p") + 1] == "litellm-dev"
    assert (
        config["claude"]["env"]["ANTHROPIC_BASE_URL"]
        == "http://127.0.0.1:4001/anthropic"
    )


def test_url_override_4000_infers_prod_target(ra, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000")
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    profile = ra._resolve_target_profile(config, target=None)
    assert profile is not None
    assert profile["expected_trace_environment"] == "prod"
    assert profile["codex_profile"] == "litellm"


def test_default_config_url_infers_prod(ra, monkeypatch) -> None:
    """No env override; config default :4000 selects prod."""
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()  # litellm_base_url = :4000
    profile = ra._resolve_target_profile(config, target=None)
    assert profile is not None
    assert profile["expected_trace_environment"] == "prod"


def test_explicit_target_precedes_config_url(ra, monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    config = _base_config()
    profile = ra._resolve_target_profile(config, target="dev")
    assert profile is not None
    assert profile["expected_trace_environment"] == "dev"
    assert profile["codex_profile"] == "litellm-dev"


def test_env_acceptance_target_precedes_config_url(ra, monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.setenv("ACCEPTANCE_TARGET", "dev")
    config = _base_config()
    profile = ra._resolve_target_profile(config, target=None)
    assert profile is not None
    assert profile["expected_trace_environment"] == "dev"


def test_unknown_url_no_explicit_target_fails_closed(ra, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://10.9.9.9:9999")
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    with pytest.raises(SystemExit, match="does not match any configured"):
        ra._resolve_target_profile(config, target=None)


# ---------------------------------------------------------------------------
# D1-574/MS-033 (review): fail-closed route rewriting
# ---------------------------------------------------------------------------


def test_apply_target_profile_fails_closed_when_codex_lacks_profile_flag(ra) -> None:
    config = _base_config()
    config["codex"]["command"] = ["codex", "exec", "--json", "hi"]  # no -p
    profile = ra._resolve_target_profile(config, target="dev")
    assert profile is not None
    with pytest.raises(SystemExit, match="exactly one"):
        ra._apply_target_profile(config, profile)


# ---------------------------------------------------------------------------
# D1-574/MS-033 (review): TimeoutExpired bytes partial output + phase
# ---------------------------------------------------------------------------


def test_run_command_timeout_decodes_bytes_partial_output(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp")

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=5,
            output=b"partial bytes stdout \xff\xfe",
            stderr=b"partial bytes stderr",
        )

    with patch.object(ra.subprocess, "run", side_effect=fake_run):
        result = ra._run_command(["codex", "exec", "-p", "hi"], timeout_seconds=5)

    assert result["timed_out"] is True
    assert result["phase"] == "client_subprocess"
    assert "partial bytes stdout" in result["stdout"]
    assert "partial bytes stderr" in result["stderr"]
    assert result["exit_code"] == -1


def test_decode_partial_output_helper(ra) -> None:
    assert ra._decode_partial_output(None) == ""
    assert ra._decode_partial_output("  str  ") == "str"
    assert ra._decode_partial_output(b"  bytes  ") == "bytes"
    # Invalid utf-8 must not raise
    decoded = ra._decode_partial_output(b"\xff\xfe")
    assert isinstance(decoded, str)


# ---------------------------------------------------------------------------
# D1-574/MS-033 (review): post-subprocess observability failure evidence
# ---------------------------------------------------------------------------


def test_family_error_result_retains_run_evidence(ra) -> None:
    run = {
        "command": ["codex", "exec", "-p", "litellm-dev"],
        "command_string": "codex exec",
        "exit_code": 0,
        "duration_seconds": 1.5,
        "stdout": "codex routed",
        "stderr": "",
        "response_excerpt": "codex routed",
    }
    result = ra._family_error_result(
        "codex",
        Exception("HTTP Error 401: Unauthorized"),
        run_evidence=run,
        phase="observability_validation",
    )
    assert result["passed"] is False
    assert result["command"] == ["codex", "exec", "-p", "litellm-dev"]
    assert result["exit_code"] == 0
    assert result["stdout"] == "codex routed"
    assert result["phase"] == "observability_validation"
    assert any("observability_validation error" in f for f in result["failures"])
    assert any("401" in f for f in result["failures"])


def test_run_family_with_evidence_retains_run_on_observability_failure(ra) -> None:
    """End-to-end: a validator that records run evidence then raises during
    observability must yield a result retaining command/exit/stdout + phase."""
    config = {
        "command": ["codex", "exec", "-p", "litellm-dev", "--json", "hi"],
        "timeout_seconds": 5,
        "expected_trace_names": ["codex"],
        "expected_user_ids": ["litellm"],
    }
    run_result = {
        "command": config["command"],
        "command_string": "codex exec",
        "exit_code": 0,
        "duration_seconds": 1.5,
        "stdout": "codex routed",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_original_chars": 12,
        "stderr_original_chars": 0,
        "output_max_chars": 200_000,
        "response_excerpt": "codex routed",
    }

    with (
        patch.object(ra, "_run_command", return_value=run_result),
        patch.object(
            ra,
            "_poll_langfuse_named_traces",
            side_effect=Exception("HTTP Error 401: Unauthorized"),
        ),
    ):
        result = ra._run_family_with_evidence(
            "codex",
            ra._validate_codex,
            config,
            query_url="http://lf",
            public_key="pk",
            secret_key="sk",
        )

    assert result["passed"] is False
    assert result["command"] == config["command"]
    assert result["exit_code"] == 0
    assert result["stdout"] == "codex routed"
    assert result["phase"] == "observability_validation"
    assert any("401" in f for f in result["failures"])


def test_custom_profile_url_is_used_for_inference(ra, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://10.0.0.1:4100")
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    config["target_profiles"] = {
        "custom": {
            "litellm_base_url": "http://10.0.0.1:4100",
            "anthropic_base_url": "http://10.0.0.1:4100/anthropic",
            "gemini_base_url": "http://10.0.0.1:4100/gemini",
            "codex_profile": "litellm-custom",
            "docker_container_name": "litellm-custom",
            "expected_trace_environment": "custom",
        }
    }

    profile = ra._resolve_target_profile(config)

    assert profile["target_name"] == "custom"
    assert profile["codex_profile"] == "litellm-custom"


def test_ambiguous_configured_profile_url_fails_closed(ra, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4001")
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    config["target_profiles"] = {
        "duplicate-dev": {
            **ra.BUILT_IN_TARGET_PROFILES["dev"],
            "docker_container_name": "duplicate-dev",
        }
    }

    with pytest.raises(SystemExit, match="ambiguous"):
        ra._resolve_target_profile(config)


def test_explicit_target_url_conflict_fails_closed(ra, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000")

    with pytest.raises(SystemExit, match="conflicts"):
        ra._resolve_target_profile(_base_config(), target="dev")


def test_config_target_url_conflict_fails_closed(ra, monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    config["target"] = "dev"

    with pytest.raises(SystemExit, match="config target"):
        ra._resolve_target_profile(config)


def test_target_container_credentials_require_resolved_profile(ra) -> None:
    with pytest.raises(SystemExit, match="requires a resolved"):
        ra._resolve_langfuse_credentials(
            {"langfuse_credential_source": "target_container"},
            None,
        )


def test_container_credential_failures_are_not_cached(ra, monkeypatch) -> None:
    calls = 0

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return SimpleNamespace(returncode=0, stdout="pk-container\n", stderr="")

    cache: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(ra.subprocess, "run", fake_run)

    assert (
        ra._resolve_container_env_value(
            "litellm-dev", "LANGFUSE_PUBLIC_KEY", cache=cache
        )
        is None
    )
    assert (
        ra._resolve_container_env_value(
            "litellm-dev", "LANGFUSE_PUBLIC_KEY", cache=cache
        )
        == "pk-container"
    )
    assert (
        ra._resolve_container_env_value(
            "litellm-dev", "LANGFUSE_PUBLIC_KEY", cache=cache
        )
        == "pk-container"
    )
    assert calls == 2


def test_timeout_partial_output_records_truncation_metadata(ra, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp")

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=5,
            output=b"A" * 1000,
            stderr=b"B" * 900,
        )

    with patch.object(ra.subprocess, "run", side_effect=fake_run):
        result = ra._run_command(
            ["codex", "exec", "-p", "hi"],
            timeout_seconds=5,
            output_max_chars=100,
        )

    assert result["phase"] == "client_subprocess"
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert result["stdout_original_chars"] == 1000
    assert result["stderr_original_chars"] == 900
    assert result["output_max_chars"] == 100


def test_success_then_pre_subprocess_failure_does_not_reuse_evidence(ra) -> None:
    run = {
        "command": ["codex", "exec"],
        "command_string": "codex exec",
        "exit_code": 0,
        "duration_seconds": 0.1,
        "stdout": "ok",
        "stderr": "",
        "response_excerpt": "ok",
    }

    def successful_validator():
        ra._record_family_run_evidence(run)
        return {"passed": True, "failures": [], "warnings": []}

    def pre_subprocess_failure():
        raise ValueError("failed before subprocess")

    assert ra._run_family_with_evidence("codex", successful_validator)["passed"]
    result = ra._run_family_with_evidence("codex", pre_subprocess_failure)
    assert result["phase"] == "initialization"
    assert result["command"] == []
    assert result["stdout"] == ""


def test_claude_fanout_effective_env_uses_selected_target(ra) -> None:
    config = {
        "command": ["claude", "-p", "prompt", "--output-format", "json"],
        "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000/anthropic"},
        "fanout_modes": {
            "minimal": {
                "command": ["claude", "-p", "prompt", "--output-format", "json"],
                "env": {
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000/anthropic"
                },
            }
        },
    }
    profile = ra._resolve_target_profile(_base_config(), target="dev")
    suite_config = _base_config()
    suite_config["claude"] = config
    ra._apply_target_profile(suite_config, profile)
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured["extra_env"] = kwargs["extra_env"]
        return {
            "command": command,
            "command_string": "claude",
            "exit_code": 0,
            "duration_seconds": 0.1,
            "stdout": "",
            "stderr": "",
            "response_excerpt": "",
        }

    with (
        patch.object(ra, "_run_command", side_effect=fake_run),
        patch.object(
            ra,
            "_poll_langfuse_required_name_traces",
            side_effect=RuntimeError("stop after subprocess"),
        ),
    ):
        with pytest.raises(RuntimeError, match="stop after subprocess"):
            ra._validate_claude(
                config,
                query_url="http://lf",
                public_key="pk",
                secret_key="sk",
                fanout_mode="minimal",
            )

    assert captured["extra_env"]["ANTHROPIC_BASE_URL"] == (
        "http://127.0.0.1:4001/anthropic"
    )


def test_public_target_profile_does_not_serialize_credentials(ra) -> None:
    profile = {
        **ra.BUILT_IN_TARGET_PROFILES["dev"],
        "target_name": "dev",
        "LANGFUSE_PUBLIC_KEY": "pk-secret",
        "LANGFUSE_SECRET_KEY": "sk-secret",
        "credential_value": "never-serialize",
    }

    serialized = json.dumps(ra._public_target_profile(profile))

    assert "pk-secret" not in serialized
    assert "sk-secret" not in serialized
    assert "never-serialize" not in serialized


@pytest.mark.parametrize("invalid_profiles", [None, []])
def test_non_object_target_profiles_are_rejected(
    ra, monkeypatch, invalid_profiles
) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    config = _base_config()
    config["target_profiles"] = invalid_profiles

    with pytest.raises(SystemExit, match="target_profiles must be an object"):
        ra._resolve_target_profile(config, target="dev")


def test_non_object_top_level_config_is_rejected(ra, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="must be a JSON object"):
        ra._load_suite_config(config_path)


@pytest.mark.parametrize("invalid_value", [None, 123, [], {}])
def test_non_string_required_target_values_are_rejected(
    ra, monkeypatch, invalid_value
) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    config = _base_config()
    config["target_profiles"] = {"dev": {"codex_profile": invalid_value}}

    with pytest.raises(SystemExit, match="non-string required keys"):
        ra._resolve_target_profile(config, target="dev")


@pytest.mark.parametrize(
    ("family_name", "family_value"),
    [
        ("codex", None),
        ("claude", []),
        ("gemini", "invalid"),
    ],
)
def test_missing_or_non_object_families_are_rejected(
    ra, family_name, family_value
) -> None:
    config = _base_config()
    config[family_name] = family_value
    profile = {**ra.BUILT_IN_TARGET_PROFILES["dev"], "target_name": "dev"}

    with pytest.raises(SystemExit, match=f"`{family_name}` as an object"):
        ra._apply_target_profile(config, profile)


@pytest.mark.parametrize(
    "invalid_command",
    [None, "codex exec", [], ["codex", 7]],
)
def test_family_commands_must_be_non_empty_lists_of_strings(
    ra, invalid_command
) -> None:
    config = _base_config()
    config["gemini"]["command"] = invalid_command
    profile = {**ra.BUILT_IN_TARGET_PROFILES["dev"], "target_name": "dev"}

    with pytest.raises(SystemExit, match=r"gemini\.command.*list\[str\]"):
        ra._apply_target_profile(config, profile)


def test_multiple_codex_profile_flags_are_rejected(ra) -> None:
    config = _base_config()
    config["codex"]["command"] = [
        "codex",
        "exec",
        "-p",
        "litellm",
        "-p",
        "litellm-dev",
        "hi",
    ]
    profile = {**ra.BUILT_IN_TARGET_PROFILES["dev"], "target_name": "dev"}

    with pytest.raises(SystemExit, match="exactly one"):
        ra._apply_target_profile(config, profile)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["claude"].update({"env": "invalid"}),
        lambda config: config["claude"].update({"fanout_modes": []}),
        lambda config: config["claude"]["fanout_modes"]["minimal"].update(
            {"env": "invalid"}
        ),
        lambda config: config["claude"]["fanout_modes"]["minimal"].update(
            {"command": ["claude", 3]}
        ),
        lambda config: config["gemini"].update({"env": []}),
    ],
)
def test_env_and_fanout_shapes_are_rejected(ra, mutation) -> None:
    config = _base_config()
    mutation(config)
    profile = {**ra.BUILT_IN_TARGET_PROFILES["dev"], "target_name": "dev"}

    with pytest.raises(SystemExit):
        ra._apply_target_profile(config, profile)


def test_resolve_target_json_validates_full_executable_config(
    ra, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    config = _base_config()
    del config["gemini"]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = SimpleNamespace(
        config=str(config_path),
        write_artifact=None,
        langfuse_query_url=None,
        claude_fanout_mode="minimal",
        target="dev",
        resolve_target_json=True,
    )

    with patch.object(ra.argparse.ArgumentParser, "parse_args", return_value=args):
        with pytest.raises(SystemExit, match="requires `gemini`"):
            ra.main()


@pytest.mark.parametrize(
    (
        "target",
        "expected_url",
        "expected_codex_profile",
        "expected_anthropic_url",
        "expected_gemini_url",
        "expected_environment",
    ),
    [
        (
            "dev",
            "http://127.0.0.1:4001",
            "litellm-dev",
            "http://127.0.0.1:4001/anthropic",
            "http://127.0.0.1:4001/gemini",
            "dev",
        ),
        (
            "prod",
            "http://127.0.0.1:4000",
            "litellm",
            "http://127.0.0.1:4000/anthropic",
            "http://127.0.0.1:4000/gemini",
            "prod",
        ),
    ],
)
def test_main_artifact_uses_resolved_profile_and_omits_credentials(
    ra,
    monkeypatch,
    tmp_path: Path,
    target,
    expected_url,
    expected_codex_profile,
    expected_anthropic_url,
    expected_gemini_url,
    expected_environment,
) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("ACCEPTANCE_TARGET", raising=False)
    config = _base_config()
    config["langfuse_credential_source"] = "target_container"
    config_path = tmp_path / f"{target}-config.json"
    artifact_path = tmp_path / f"{target}-artifact.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = SimpleNamespace(
        config=str(config_path),
        write_artifact=str(artifact_path),
        langfuse_query_url=None,
        claude_fanout_mode="minimal",
        target=target,
        resolve_target_json=False,
    )
    captured: dict[str, dict[str, Any]] = {}

    def codex_failure(family_config, **kwargs):  # noqa: ANN001
        captured["codex"] = json.loads(json.dumps(family_config))
        ra._record_family_run_evidence(
            {
                "command": list(family_config["command"]),
                "command_string": "codex exec",
                "exit_code": 0,
                "duration_seconds": 0.2,
                "stdout": "codex routed",
                "stderr": "",
                "response_excerpt": "codex routed",
            }
        )
        raise RuntimeError("observability unavailable")

    def successful_family(name):
        def validator(family_config, **kwargs):  # noqa: ANN001
            captured[name] = json.loads(json.dumps(family_config))
            return {"passed": True, "failures": [], "warnings": []}

        return validator

    with (
        patch.object(ra.argparse.ArgumentParser, "parse_args", return_value=args),
        patch.object(
            ra,
            "_resolve_langfuse_credentials",
            return_value=("pk-container-secret", "sk-container-secret"),
        ),
        patch.object(ra, "_validate_codex", side_effect=codex_failure),
        patch.object(ra, "_validate_gemini", side_effect=successful_family("gemini")),
        patch.object(ra, "_validate_claude", side_effect=successful_family("claude")),
        patch.object(ra, "_docker_status", return_value="mocked"),
        patch.object(ra, "_git_value", return_value="mocked"),
    ):
        assert ra.main() == 1

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    serialized = json.dumps(artifact)
    assert artifact["environment"]["litellm_base_url"] == expected_url
    assert artifact["environment"]["target_profile"]["target_name"] == target
    assert (
        artifact["environment"]["target_profile"]["expected_trace_environment"]
        == expected_environment
    )
    codex_command = captured["codex"]["command"]
    assert codex_command[codex_command.index("-p") + 1] == expected_codex_profile
    assert (
        captured["claude"]["env"]["ANTHROPIC_BASE_URL"]
        == expected_anthropic_url
    )
    assert (
        captured["claude"]["fanout_modes"]["minimal"]["env"][
            "ANTHROPIC_BASE_URL"
        ]
        == expected_anthropic_url
    )
    assert captured["gemini"]["env"]["CODE_ASSIST_ENDPOINT"] == expected_gemini_url
    assert artifact["results"]["codex"]["phase"] == "observability_validation"
    assert artifact["results"]["codex"]["exit_code"] == 0
    assert artifact["results"]["codex"]["stdout"] == "codex routed"
    assert "pk-container-secret" not in serialized
    assert "sk-container-secret" not in serialized
