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
    start = text.index("load_harness_dotenv()")
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
    start = text.index("load_harness_dotenv()")
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
