"""Named LiteLLM containers start Hypercorn for inbound Cursor Agent CLI HTTP/2."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from litellm.proxy.pass_through_endpoints.cursor_agent_cli_hypercorn import (
    configure_hypercorn_for_cursor_agent_cli,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_configure_hypercorn_keeps_http1_and_lengthens_keepalive() -> None:
    config = MagicMock()
    config.alpn_protocols = ["h2", "http/1.1"]
    config.keep_alive_timeout = 5
    config.h2_max_inbound_frame_size = 16384
    configure_hypercorn_for_cursor_agent_cli(config)
    assert config.keep_alive_timeout == 600
    assert "h2" in config.alpn_protocols
    assert "http/1.1" in config.alpn_protocols
    assert config.h2_max_inbound_frame_size == 16 * 1024 * 1024


def test_hypercorn_helper_is_used_by_proxy_cli() -> None:
    source = _read("litellm/proxy/proxy_cli.py")
    assert "configure_hypercorn_for_cursor_agent_cli" in source
    assert "cursor_agent_cli_hypercorn" in source
    assert "--run_hypercorn" in source


def test_named_container_start_commands_enable_hypercorn() -> None:
    """Drive the shipped compose/Dockerfiles; do not pin content hashes."""
    compose_dev = _read("docker-compose.dev.yml")
    compose_alpha = _read("docker-compose.alpha.yml")
    dockerfile_dev = _read("Dockerfile.dev")
    dockerfile_alpha = _read("Dockerfile.alpha")
    requirements = _read("requirements.txt")

    assert "container_name: litellm-dev" in compose_dev
    assert "container_name: litellm-alpha" in compose_alpha
    assert "--run_hypercorn" in compose_dev
    assert "--run_hypercorn" in compose_alpha
    assert "--run_hypercorn" in dockerfile_dev
    assert "--run_hypercorn" in dockerfile_alpha
    assert "hypercorn==0.15.0" in requirements
