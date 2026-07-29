"""RR-081 residuals for scripts/local-ci/run_acceptance.sh.

Already closed in prior commits and re-asserted here:
  High #1  — fingerprint persisted only after successful compose
  Medium #2 — fingerprint excludes .venv/node_modules/__pycache__/.env

This file focuses on Medium #3 (shell-side .env inheritance):
  run_acceptance.sh must not `set -a; source .env` (which re-exports every
  secret into the process environment that child CLIs inherit). Instead it
  must selectively export only harness-needed keys (LANGFUSE_*, LITELLM_BASE_URL,
  harness overrides) and leave DB/provider secrets out of the shell env.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "local-ci" / "run_acceptance.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_shell_does_not_source_dotenv_with_set_a_allexport() -> None:
    text = _script_text()
    # No active allexport source of .env (comment mentions are fine).
    # Strip full-line comments then assert.
    active_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Keep inline-code but drop trailing comments carefully enough for this check
        active_lines.append(stripped)
    active = "\n".join(active_lines)

    assert "set -a" not in active
    # Must not source .env (even without set -a)
    assert not re.search(r"(^|[\s;])source\s+(\./)?\.env\b", active)
    assert not re.search(r"(^|[\s;])\.\s+(\./)?\.env\b", active)

    # Selective loader present
    assert "load_harness_dotenv" in text
    assert "LANGFUSE_" in text
    assert "LITELLM_BASE_URL" in text
    assert 'line="${line//$\'\\r\'/}"' in text


def test_load_harness_dotenv_exports_only_allowlisted_keys(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        textwrap.dedent(
            """\
            # comment
            LANGFUSE_PUBLIC_KEY=pk-test
            LANGFUSE_SECRET_KEY="sk-test"
            export LANGFUSE_HOST=http://127.0.0.1:3000
            LANGFUSE_QUERY_URL=http://127.0.0.1:3000
            LITELLM_BASE_URL=http://127.0.0.1:4001
            LITELLM_PORT=4001
            AAWM_HARNESS_RUN_ID=run-xyz
            AAWM_DB_PASSWORD=super-secret-db
            AAWM_DB_USER=aawm
            AAWM_OPENAI_API_KEY=sk-openai
            AAWM_NVIDIA_API_KEY=nv-secret
            DATABASE_URL=postgres://user:pass@localhost/db
            POSTGRES_PASSWORD=pg-pass
            UNRELATED=nope
            """
        ),
        encoding="utf-8",
    )

    # Extract the function body from the real script and run it under env -i.
    text = _script_text()
    start = text.index("declare -A HARNESS_INCOMING_ENV_PRESENT")
    end = text.index('\nload_harness_dotenv "$ROOT/.env"', start)
    fn_src = text[start:end]

    probe = tmp_path / "probe.sh"
    probe.write_text(
        fn_src
        + textwrap.dedent(
            f"""
            load_harness_dotenv "{env_file}"
            # Print exported allowlist
            for k in LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_HOST LANGFUSE_QUERY_URL LITELLM_BASE_URL LITELLM_PORT AAWM_HARNESS_RUN_ID; do
              eval "printf '%s=%s\\n' \\"$k\\" \\"\\${{$k-}}\\""
            done
            # Fail if any denied key is present in the environment
            for k in AAWM_DB_PASSWORD AAWM_DB_USER AAWM_OPENAI_API_KEY AAWM_NVIDIA_API_KEY DATABASE_URL POSTGRES_PASSWORD UNRELATED; do
              if printenv "$k" >/dev/null 2>&1; then
                echo "LEAKED:$k" >&2
                exit 3
              fi
            done
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["env", "-i", f"PATH={os.environ.get('PATH', '/usr/bin')}", "bash", "--noprofile", "--norc", str(probe)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out = result.stdout
    assert "LANGFUSE_PUBLIC_KEY=pk-test" in out
    assert "LANGFUSE_SECRET_KEY=sk-test" in out
    assert "LANGFUSE_HOST=http://127.0.0.1:3000" in out
    assert "LITELLM_BASE_URL=http://127.0.0.1:4001" in out
    assert "AAWM_HARNESS_RUN_ID=run-xyz" in out
    assert "LEAKED:" not in out
    assert "super-secret-db" not in out
    assert "sk-openai" not in out


def test_load_harness_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    text = _script_text()
    start = text.index("declare -A HARNESS_INCOMING_ENV_PRESENT")
    end = text.index('\nload_harness_dotenv "$ROOT/.env"', start)
    fn_src = text[start:end]
    probe = tmp_path / "probe.sh"
    missing = tmp_path / "no-such.env"
    probe.write_text(
        fn_src
        + f'\nload_harness_dotenv "{missing}"\necho ok\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["env", "-i", f"PATH={os.environ.get('PATH', '/usr/bin')}", "bash", "--noprofile", "--norc", str(probe)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_fingerprint_still_persisted_only_after_compose() -> None:
    """Regression guard for High #1 (already fixed; keep under RR-081 suite)."""
    text = _script_text()
    start = text.index("should_rebuild_litellm_dev()")
    end = text.index("persist_litellm_dev_build_state()", start)
    body = text[start:end]
    assert re.search(r">\s*[\"']?\$BUILD_STATE_PATH", body) is None
    rebuild_start = text.index(
        'if [[ "$REBUILD_LITELLM_DEV" == "1" ]] && should_rebuild_litellm_dev; then'
    )
    rebuild_block = text[rebuild_start : rebuild_start + 900]
    assert rebuild_block.index("docker compose") < rebuild_block.index(
        "persist_litellm_dev_build_state"
    )


def test_fingerprint_excludes_still_cover_heavy_trees() -> None:
    """Regression guard for Medium #2."""
    text = _script_text()
    start = text.index("compute_build_fingerprint()")
    heredoc_start = text.index("<<'PY'", start) + len("<<'PY'")
    heredoc_end = text.index("\nPY\n", heredoc_start)
    block = text[heredoc_start:heredoc_end]
    for needle in (".venv/", "node_modules/", "__pycache__/", "dist/", ".env"):
        assert needle in block


def _write_wrapper_fakes(tmp_path: Path) -> tuple[Path, Path, Path]:
    python_log = tmp_path / "python.log"
    docker_log = tmp_path / "docker.log"
    python_wrapper = tmp_path / "python-wrapper"
    python_wrapper.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-c" || "$*" == *"--resolve-target-json"* ]]; then
              exec "$REAL_PYTHON" "$@"
            fi
            printf '%s\n' "$*" >> "$PYTHON_LOG"
            exit 0
            """
        ),
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    docker = tmp_path / "docker"
    docker.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\n' "$*" >> "$DOCKER_LOG"
            if [[ "${1:-}" == "ps" ]]; then
              printf 'Up (healthy)\n'
            fi
            """
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return python_wrapper, python_log, docker_log


def test_exact_4001_wrapper_command_resolves_dev_before_lifecycle(tmp_path: Path) -> None:
    result, python_calls, docker_calls = _run_isolated_wrapper(
        tmp_path,
        dotenv="",
        incoming={
            "LITELLM_BASE_URL": "http://127.0.0.1:4001",
            "REBUILD_LITELLM_DEV": "0",
        },
        real_preflight=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    preflight_call = next(
        line for line in python_calls.splitlines() if "--resolve-target-json" in line
    )
    assert "--target" not in preflight_call
    assert "--target dev" in python_calls
    assert "--write-artifact" in python_calls
    assert "images" not in docker_calls
    assert "compose" not in docker_calls


def test_wrapper_target_url_conflict_fails_before_docker(tmp_path: Path) -> None:
    python_wrapper, _python_log, docker_log = _write_wrapper_fakes(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ.get('PATH', '/usr/bin')}",
        "PYTHON_BIN": str(python_wrapper),
        "REAL_PYTHON": sys.executable,
        "PYTHON_LOG": str(tmp_path / "python.log"),
        "DOCKER_LOG": str(docker_log),
        "LITELLM_BASE_URL": "http://127.0.0.1:4000",
        "ACCEPTANCE_TARGET": "dev",
        "REBUILD_LITELLM_DEV": "0",
    }
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=str(_REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "conflicts" in result.stderr
    assert not docker_log.exists()


def _run_isolated_wrapper(
    tmp_path: Path,
    *,
    dotenv: str = "",
    incoming: dict[str, str] | None = None,
    target_name: str = "prod",
    target_url: str = "http://127.0.0.1:4000",
    real_preflight: bool = False,
    config_text: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    root = tmp_path / "repo"
    script_dir = root / "scripts" / "local-ci"
    script_dir.mkdir(parents=True)
    script = script_dir / "run_acceptance.sh"
    script.write_text(_script_text(), encoding="utf-8")
    script.chmod(0o755)
    if real_preflight:
        for filename in (
            "run_acceptance.py",
            "config.json",
            "claude_acceptance_prompt.txt",
            "claude_acceptance_prompt_full_fanout.txt",
        ):
            source = _REPO / "scripts" / "local-ci" / filename
            (script_dir / filename).write_bytes(source.read_bytes())
    if config_text is not None:
        (script_dir / "config.json").write_text(config_text, encoding="utf-8")
    (root / ".env").write_text(dotenv, encoding="utf-8")

    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    python_log = root / "python.log"
    docker_log = root / "docker.log"
    python_wrapper = fake_bin / "python-wrapper"
    python_wrapper.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${1:-}" == "-c" ]]; then
              exec "$REAL_PYTHON" "$@"
            fi
            printf '%s\\n' "$*" >> "$PYTHON_LOG"
            if [[ "${REAL_PREFLIGHT:-0}" == "1" && "$*" == *"--resolve-target-json"* ]]; then
              exec "$REAL_PYTHON" "$@"
            fi
            if [[ "$*" == *"--resolve-target-json"* ]]; then
              printf '%s\\n' "$TARGET_RESOLUTION_JSON"
            fi
            exit 0
            """
        ),
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    docker = fake_bin / "docker"
    docker.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$*" >> "$DOCKER_LOG"
            if [[ "${1:-}" == "ps" ]]; then
              printf 'Up (healthy)\\n'
            elif [[ "${1:-}" == "images" ]]; then
              printf 'image-id\\n'
            fi
            """
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)

    excluded = {
        "ACCEPTANCE_CONFIG_PATH",
        "ACCEPTANCE_TARGET",
        "BUILD_STATE_PATH",
        "LITELLM_BASE_URL",
        "PYTHON_BIN",
        "REBUILD_LITELLM_DEV",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in excluded and not key.startswith("LANGFUSE_")
    }
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '/usr/bin')}",
            "PYTHON_BIN": str(python_wrapper),
            "REAL_PYTHON": sys.executable,
            "PYTHON_LOG": str(python_log),
            "DOCKER_LOG": str(docker_log),
            "REAL_PREFLIGHT": "1" if real_preflight else "0",
            "TARGET_RESOLUTION_JSON": (
                f'{{"target_name":"{target_name}",'
                f'"litellm_base_url":"{target_url}"}}'
            ),
        }
    )
    if incoming:
        env.update(incoming)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=str(root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    python_calls = (
        python_log.read_text(encoding="utf-8") if python_log.exists() else ""
    )
    docker_calls = (
        docker_log.read_text(encoding="utf-8") if docker_log.exists() else ""
    )
    return result, python_calls, docker_calls


def test_dotenv_does_not_override_caller_set_values(tmp_path: Path) -> None:
    result, python_calls, _docker_calls = _run_isolated_wrapper(
        tmp_path,
        dotenv=(
            "ACCEPTANCE_CONFIG_PATH=dotenv.json\n"
            "ACCEPTANCE_TARGET=dev\n"
        ),
        incoming={
            "ACCEPTANCE_CONFIG_PATH": "caller.json",
            "ACCEPTANCE_TARGET": "prod",
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "--config caller.json" in python_calls
    assert "--target prod" in python_calls
    assert "dotenv.json" not in python_calls


def test_dotenv_preserves_explicitly_empty_caller_value(tmp_path: Path) -> None:
    result, python_calls, docker_calls = _run_isolated_wrapper(
        tmp_path,
        dotenv=(
            "ACCEPTANCE_TARGET=prod\n"
            "REBUILD_LITELLM_DEV=1\n"
        ),
        incoming={
            "ACCEPTANCE_TARGET": "",
            "REBUILD_LITELLM_DEV": "",
        },
        target_name="dev",
        target_url="http://127.0.0.1:4001",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    preflight_call = next(
        line for line in python_calls.splitlines() if "--resolve-target-json" in line
    )
    assert "--target" not in preflight_call
    assert "--target dev" in python_calls
    assert "images" not in docker_calls
    assert "compose" not in docker_calls


def test_dotenv_only_values_apply_before_defaults_and_lifecycle(tmp_path: Path) -> None:
    result, python_calls, docker_calls = _run_isolated_wrapper(
        tmp_path,
        dotenv=(
            "ACCEPTANCE_CONFIG_PATH=dotenv-only.json\n"
            "LITELLM_BASE_URL=http://127.0.0.1:4001\n"
            "REBUILD_LITELLM_DEV=0\n"
        ),
        target_name="dev",
        target_url="http://127.0.0.1:4001",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "--config dotenv-only.json" in python_calls
    assert "--target dev" in python_calls
    assert "images" not in docker_calls
    assert "compose" not in docker_calls


def test_wrapper_defaults_apply_after_empty_dotenv(tmp_path: Path) -> None:
    result, python_calls, docker_calls = _run_isolated_wrapper(tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "--config scripts/local-ci/config.json" in python_calls
    assert "--target prod" in python_calls
    assert docker_calls == ""


def test_production_wrapper_has_no_dev_lifecycle_operation(tmp_path: Path) -> None:
    result, python_calls, docker_calls = _run_isolated_wrapper(
        tmp_path,
        incoming={"ACCEPTANCE_TARGET": "prod"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "--target prod" in python_calls
    assert docker_calls == ""


def test_null_target_profiles_preflight_fails_before_docker(tmp_path: Path) -> None:
    result, python_calls, docker_calls = _run_isolated_wrapper(
        tmp_path,
        dotenv="",
        incoming={
            "LITELLM_BASE_URL": "http://127.0.0.1:4001",
            "REBUILD_LITELLM_DEV": "0",
        },
        real_preflight=True,
        config_text='{"target_profiles": null}\n',
    )

    assert result.returncode != 0
    assert "target_profiles must be an object" in result.stderr
    assert "--resolve-target-json" in python_calls
    assert docker_calls == ""
