"""RR-081: build fingerprint must not be persisted before successful rebuild."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "local-ci" / "run_acceptance.sh"


def test_should_rebuild_does_not_write_build_state_path() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")
    # Extract should_rebuild function body roughly
    start = text.index("should_rebuild_litellm_dev()")
    end = text.index("persist_litellm_dev_build_state()", start)
    body = text[start:end]
    assert 'echo "$state_json" > "$BUILD_STATE_PATH"' not in body
    assert "BUILD_STATE_PATH" in body  # may still *read* previous fingerprint
    assert "return 0" in body


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
