"""RR-081: build fingerprint must not be persisted before successful rebuild.

Also covers Medium #2: fingerprint exclude list must skip heavy/volatile trees
(.venv, node_modules, __pycache__, etc.) so compute_build_fingerprint stays
cheap and stable.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "local-ci" / "run_acceptance.sh"


def _fingerprint_python_block() -> str:
    """Extract the embedded Python body of compute_build_fingerprint()."""
    text = _SCRIPT.read_text(encoding="utf-8")
    start = text.index("compute_build_fingerprint()")
    # The heredoc body sits between <<'PY' and the closing PY on its own line.
    heredoc_start = text.index("<<'PY'", start) + len("<<'PY'")
    heredoc_end = text.index("\nPY\n", heredoc_start)
    return text[heredoc_start:heredoc_end]


def test_should_rebuild_does_not_write_build_state_path() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")
    # Extract should_rebuild function body roughly
    start = text.index("should_rebuild_litellm_dev()")
    end = text.index("persist_litellm_dev_build_state()", start)
    body = text[start:end]
    assert 'echo "$state_json" > "$BUILD_STATE_PATH"' not in body
    # must not write BUILD_STATE_PATH in any form (High #1 / RR-081)
    assert re.search(r">\s*[\"']?\$BUILD_STATE_PATH", body) is None
    assert "BUILD_STATE_PATH" in body  # may still *read* previous fingerprint
    assert "return 0" in body
    # Decision-only comment retained
    assert "never persist BUILD_STATE_PATH" in body or "Decision only" in body


def test_fingerprint_persisted_only_after_successful_compose() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")
    # After docker compose up --force-recreate, persist must run
    rebuild_block_start = text.index(
        'if [[ "$REBUILD_LITELLM_DEV" == "1" ]] && should_rebuild_litellm_dev; then'
    )
    rebuild_block = text[rebuild_block_start : rebuild_block_start + 800]
    assert "docker compose -f docker-compose.dev.yml build litellm-dev" in rebuild_block
    assert "up -d --force-recreate litellm-dev" in rebuild_block
    assert "persist_litellm_dev_build_state" in rebuild_block
    # Order: compose commands before persist
    assert rebuild_block.index("docker compose") < rebuild_block.index(
        "persist_litellm_dev_build_state"
    )
    # Persist must not appear before the rebuild decision returns
    should_body_start = text.index("should_rebuild_litellm_dev()")
    should_body_end = text.index("persist_litellm_dev_build_state()", should_body_start)
    should_body = text[should_body_start:should_body_end]
    assert "persist_litellm_dev_build_state" not in should_body
    assert "> \"$BUILD_STATE_PATH\"" not in should_body
    assert '> "$BUILD_STATE_PATH"' not in should_body


def test_fingerprint_excludes_heavy_volatile_trees() -> None:
    """Medium #2: fingerprint must skip .venv, node_modules, caches, etc."""
    block = _fingerprint_python_block()
    required_prefixes = [
        ".venv/",
        "venv/",
        "node_modules/",
        "ui/litellm-dashboard/node_modules/",
        "__pycache__/",
        ".pytest_cache/",
        "dist/",
        ".git/",
        ".analysis/",
    ]
    for prefix in required_prefixes:
        assert prefix in block, f"expected exclude prefix {prefix!r} in fingerprint block"

    required_exact = [".env"]
    for name in required_exact:
        assert name in block, f"expected exclude exact {name!r} in fingerprint block"

    # Suffix / compiled artifact filtering
    assert ".pyc" in block
    # Nested path filtering so package-local node_modules/__pycache__ are skipped
    assert "/node_modules/" in block
    assert "/__pycache__/" in block


def test_fingerprint_exclude_logic_skips_sample_paths() -> None:
    """Execute the include() helper against representative paths (unit-level)."""
    # Replicate the exclude rules from the script so we can unit-test them
    # without walking the full repo.
    exclude_prefixes = [
        ".git/",
        ".analysis/",
        "captures/",
        "scripts/local-ci/",
        ".gemini/",
        ".codex/",
        ".venv/",
        "venv/",
        "node_modules/",
        "ui/litellm-dashboard/node_modules/",
        "ui/litellm-dashboard/.next/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".tox/",
        "dist/",
        "build/",
        ".wheel-build/",
        "htmlcov/",
        ".coverage/",
        "docs/my-website/node_modules/",
        "docs/my-website/build/",
        "docs/my-website/.docusaurus/",
    ]
    exclude_exact = {
        "langfuse-traces.png",
        "litellm/integrations/aawm_agent_identity.py",
        "litellm/integrations/aawm_payload_capture.py",
        "litellm-dev-config.yaml",
        ".env",
        ".env.local",
    }
    exclude_suffixes = (".pyc", ".pyo", ".pyd", ".so", ".egg-info")
    exclude_name_parts = (
        "/__pycache__/",
        "/node_modules/",
        "/.venv/",
        "/venv/",
        "/.pytest_cache/",
        "/.mypy_cache/",
        "/.ruff_cache/",
        "/.git/",
        "/dist/",
        "/build/",
    )

    def include(rel: str) -> bool:
        if rel in exclude_exact:
            return False
        if any(rel.startswith(prefix) for prefix in exclude_prefixes):
            return False
        if any(part in f"/{rel}/" or part in f"/{rel}" for part in exclude_name_parts):
            return False
        if any(rel.endswith(suffix) for suffix in exclude_suffixes):
            return False
        name = rel.rsplit("/", 1)[-1]
        if name == ".env" or name.startswith(".env."):
            return False
        if name.endswith(".pyc") or name == "__pycache__":
            return False
        return True

    # Must exclude
    must_exclude = [
        ".venv/lib/python3.12/site-packages/foo.py",
        "venv/bin/activate",
        "node_modules/react/index.js",
        "ui/litellm-dashboard/node_modules/lodash/index.js",
        "litellm/__pycache__/main.cpython-312.pyc",
        "tests/test_litellm/__pycache__/test_x.pyc",
        ".pytest_cache/v/cache/nodeids",
        "dist/litellm-1.0.0-py3-none-any.whl",
        ".env",
        ".env.local",
        "some/package/foo.pyc",
        "ui/litellm-dashboard/.next/cache/webpack.js",
        ".git/objects/aa/bb",
        ".analysis/artifacts/litellm-dev-build-state.json",
        "docs/my-website/node_modules/foo/bar.js",
        "litellm/foo/node_modules/bar/index.js",  # nested
    ]
    for rel in must_exclude:
        assert include(rel) is False, f"expected exclude: {rel}"

    # Must include (source / config that should trigger rebuilds)
    must_include = [
        "litellm/main.py",
        "litellm/proxy/proxy_server.py",
        "docker-compose.dev.yml",
        "Dockerfile",
        "pyproject.toml",
        "requirements.txt",
        "ui/litellm-dashboard/src/app/page.tsx",
    ]
    for rel in must_include:
        assert include(rel) is True, f"expected include: {rel}"

    # Cross-check script still lists the same heavy prefixes (drift guard)
    block = _fingerprint_python_block()
    for prefix in (".venv/", "node_modules/", "__pycache__/", "dist/", ".pytest_cache/"):
        assert f'"{prefix}"' in block or f"'{prefix}'" in block


def test_agent_identity_wheel_parity_script_exists() -> None:
    sync = _REPO / "scripts" / "sync_aawm_agent_identity_to_wheel.py"
    assert sync.is_file()
    wheel = (
        _REPO
        / ".wheel-build"
        / "aawm_litellm_callbacks"
        / "agent_identity.py"
    )
    canonical = _REPO / "litellm" / "integrations" / "aawm_agent_identity.py"
    assert wheel.is_file() and canonical.is_file()
    assert wheel.read_bytes() == canonical.read_bytes()
