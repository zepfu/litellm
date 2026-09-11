"""Prod runtime must load git-archive /app ahead of a lagging wheel overlay."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prod_dockerfile_sets_pythonpath_to_app() -> None:
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "WORKDIR /app" in dockerfile
    assert "ENV PYTHONPATH=/app" in dockerfile


def test_prod_entrypoint_exports_pythonpath_app_default() -> None:
    entrypoint = (_REPO_ROOT / "docker" / "prod_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert 'export PYTHONPATH="${PYTHONPATH:-/app}"' in entrypoint
    assert "exec litellm" in entrypoint
