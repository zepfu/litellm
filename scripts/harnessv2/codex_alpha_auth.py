#!/usr/bin/env python3
"""Emit the existing litellm-alpha bearer credential for Codex command auth.

The credential is owned by the running alpha container. This helper accepts
only the non-secret alpha audience guard, reads the container environment
through Docker, and writes the raw bearer token to stdout. It never accepts a
credential through argv or the helper environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import NoReturn


_ALPHA_CONTAINER = "litellm-alpha"
_ALPHA_AUDIENCE = "alpha"
_CREDENTIAL_ENV = "LITELLM_MASTER_KEY"
_DOCKER_TIMEOUT_SECONDS = 15


def _fail(message: str) -> NoReturn:
    print(f"codex alpha auth helper: {message}", file=sys.stderr)
    raise SystemExit(1)


def _docker_env() -> dict[str, str]:
    """Keep credentials from the caller environment out of Docker CLI."""

    env = {"PATH": os.environ.get("PATH", os.defpath)}
    home = os.environ.get("HOME")
    if home:
        env["HOME"] = home
    return env


def _read_alpha_credential(docker: str) -> str:
    try:
        result = subprocess.run(
            [
                docker,
                "--context",
                "default",
                "exec",
                _ALPHA_CONTAINER,
                "printenv",
                _CREDENTIAL_ENV,
            ],
            capture_output=True,
            check=False,
            env=_docker_env(),
            text=False,
            timeout=_DOCKER_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        _fail("alpha credential source is unavailable")

    if result.returncode != 0:
        _fail("alpha credential source is unavailable")

    raw = result.stdout.strip()
    if not raw:
        _fail("alpha credential source is empty")

    try:
        token = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("alpha credential source is invalid")

    if not token or any(
        character.isspace() or not character.isprintable() for character in token
    ):
        _fail("alpha credential source is invalid")
    return token


def main(argv: list[str]) -> int:
    if argv != ["--audience", _ALPHA_AUDIENCE]:
        _fail("expected the non-secret invocation `--audience alpha`")

    docker = shutil.which("docker")
    if docker is None:
        _fail("docker is unavailable")

    token = _read_alpha_credential(docker)
    sys.stdout.write(token)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
