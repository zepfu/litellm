"""RR-083 residuals for scripts/local-ci/run_anthropic_adapter_acceptance.sh.

Finding (Medium/operational): after optionally starting litellm-dev, the wrapper
used a fixed `sleep 5` plus a single `curl -sf` liveliness probe. On cold/slow
hosts that can hard-fail the whole acceptance run while the service is still
booting.

Required shape:
  - No fixed sleep as the readiness strategy after compose up.
  - Bounded retry/backoff readiness gate against /health/liveliness.
  - Readiness gate does not start/restart containers itself.
  - Existing invocation semantics (args, env overrides, optional compose start
    when container is absent) are preserved.
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _active_lines(text: str) -> str:
    active = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        active.append(stripped)
    return "\n".join(active)


def _readiness_function_body(text: str) -> str:
    start = text.index("wait_for_litellm_dev_ready()")
    end = text.index("\ndocker_status=", start)
    return text[start:end]


def test_script_has_no_fixed_sleep_readiness_strategy() -> None:
    """Must not use a bare sleep-then-single-curl readiness pattern."""
    text = _script_text()

    # Fixed sleep 5 (or similar bare sleeps) between compose up and health check
    # is the flaky pattern under review. sleep inside the readiness helper for
    # backoff is expected and lives only inside wait_for_litellm_dev_ready.
    assert "sleep 5" not in text
    # No top-level bare sleep outside the readiness function.
    body = _readiness_function_body(text)
    outside = text[: text.index("wait_for_litellm_dev_ready()")] + text[
        text.index("\ndocker_status=") :
    ]
    assert not re.search(r"(?m)^\s*sleep\s+\d+\s*$", outside)
    assert "wait_for_litellm_dev_ready" in text
    assert "while" in body
    assert "attempt" in body


def test_readiness_gate_uses_retry_and_backoff() -> None:
    text = _script_text()
    body = _readiness_function_body(text)

    assert "curl" in body
    assert "/health/liveliness" in text
    assert "current_delay" in body
    assert "max_attempts" in body
    assert "attempt" in body
    # Exponential backoff, capped.
    assert "current_delay * 2" in body or "current_delay*2" in body.replace(" ", "")
    assert "max_delay" in body
    # Bounded — must not loop forever.
    assert "return 1" in body
    assert "return 0" in body


def test_readiness_gate_does_not_start_or_restart_containers() -> None:
    """The readiness helper itself must not invoke docker lifecycle commands."""
    text = _script_text()
    body = _readiness_function_body(text).lower()
    for banned in (
        "docker compose",
        "docker-compose",
        "docker start",
        "docker restart",
        "docker run",
        "up -d",
        "force-recreate",
    ):
        assert banned not in body, f"readiness gate must not run {banned!r}"


def test_optional_compose_start_still_present_before_readiness() -> None:
    """Preserve invocation semantics: start only if container absent, then gate."""
    text = _script_text()
    docker_idx = text.index("docker ps --filter name=^litellm-dev$")
    up_idx = text.index("docker compose -f docker-compose.dev.yml up -d litellm-dev")
    call_match = re.search(r"if\s+!\s+wait_for_litellm_dev_ready\b", text)
    assert call_match is not None
    call_idx = call_match.start()
    assert docker_idx < up_idx < call_idx
    between = text[up_idx:call_idx]
    assert "sleep" not in between


def test_wait_for_litellm_dev_ready_succeeds_on_first_healthy_probe(
    tmp_path: Path,
) -> None:
    fn_src = _readiness_function_body(_script_text())

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    sleep_log = tmp_path / "sleep.log"
    sleep = fake_bin / "sleep"
    sleep.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SLEEP_LOG"\n',
        encoding="utf-8",
    )
    sleep.chmod(0o755)

    probe = tmp_path / "probe.sh"
    probe.write_text(
        fn_src
        + textwrap.dedent(
            """
            LIVELINESS_URL="http://127.0.0.1:4001/health/liveliness"
            READY_MAX_ATTEMPTS=5
            READY_INITIAL_DELAY_SECONDS=1
            READY_MAX_DELAY_SECONDS=5
            READY_CURL_MAX_TIME_SECONDS=1
            wait_for_litellm_dev_ready
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin')}",
            "SLEEP_LOG": str(sleep_log),
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "is healthy" in result.stdout
    assert not sleep_log.exists(), "must not sleep when first probe succeeds"


def test_wait_for_litellm_dev_ready_retries_with_backoff_then_fails(
    tmp_path: Path,
) -> None:
    fn_src = _readiness_function_body(_script_text())

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 22\n", encoding="utf-8")
    curl.chmod(0o755)
    sleep_log = tmp_path / "sleep.log"
    sleep = fake_bin / "sleep"
    sleep.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SLEEP_LOG"\n',
        encoding="utf-8",
    )
    sleep.chmod(0o755)

    probe = tmp_path / "probe.sh"
    probe.write_text(
        fn_src
        + textwrap.dedent(
            """
            LIVELINESS_URL="http://127.0.0.1:9/health/liveliness"
            READY_MAX_ATTEMPTS=3
            READY_INITIAL_DELAY_SECONDS=1
            READY_MAX_DELAY_SECONDS=5
            READY_CURL_MAX_TIME_SECONDS=1
            wait_for_litellm_dev_ready
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin')}",
            "SLEEP_LOG": str(sleep_log),
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "not healthy" in result.stdout
    # 3 attempts => 2 sleeps with backoff 1 then 2 (capped by max 5).
    delays = sleep_log.read_text(encoding="utf-8").splitlines()
    assert delays == ["1", "2"], delays


def test_wait_for_litellm_dev_ready_succeeds_after_transient_failures(
    tmp_path: Path,
) -> None:
    fn_src = _readiness_function_body(_script_text())

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state = tmp_path / "curl_state"
    state.write_text("0", encoding="utf-8")
    curl = fake_bin / "curl"
    curl.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            n="$(cat "{state}")"
            n=$((n + 1))
            printf '%s' "$n" > "{state}"
            # Fail first two probes, succeed on third.
            if [[ "$n" -lt 3 ]]; then
              exit 22
            fi
            exit 0
            """
        ),
        encoding="utf-8",
    )
    curl.chmod(0o755)
    sleep_log = tmp_path / "sleep.log"
    sleep = fake_bin / "sleep"
    sleep.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SLEEP_LOG"\n',
        encoding="utf-8",
    )
    sleep.chmod(0o755)

    probe = tmp_path / "probe.sh"
    probe.write_text(
        fn_src
        + textwrap.dedent(
            """
            LIVELINESS_URL="http://127.0.0.1:4001/health/liveliness"
            READY_MAX_ATTEMPTS=5
            READY_INITIAL_DELAY_SECONDS=1
            READY_MAX_DELAY_SECONDS=5
            READY_CURL_MAX_TIME_SECONDS=1
            wait_for_litellm_dev_ready
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin')}",
            "SLEEP_LOG": str(sleep_log),
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "attempt 3/5" in result.stdout
    delays = sleep_log.read_text(encoding="utf-8").splitlines()
    assert delays == ["1", "2"], delays


def test_invocation_args_and_exec_semantics_preserved() -> None:
    text = _script_text()
    assert 'ARTIFACT_PATH="${1:-' in text
    assert 'CASES_ARG="${2:-${ANTHROPIC_ADAPTER_CASES:-}}"' in text
    assert "run_anthropic_adapter_acceptance.py" in text
    assert "--write-artifact" in text
    assert 'ARGS+=(--cases "$CASES_ARG")' in text
    assert 'exec "$PYTHON_BIN" "${ARGS[@]}"' in text
