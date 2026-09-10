"""Hypercorn HTTP/2 settings for inbound Cursor Agent CLI Connect."""

from __future__ import annotations

from typing import Any

CURSOR_AGENT_CLI_HTTP2_KEEPALIVE_SECONDS = 600
CURSOR_AGENT_CLI_H2_MAX_INBOUND_FRAME_SIZE = 16 * 1024 * 1024


def configure_hypercorn_for_cursor_agent_cli(config: Any) -> Any:
    """Keep HTTP/1.1 available while enabling HTTP/2 bidi for Agent CLI turns.

    TLS uses ALPN ``h2`` / ``http/1.1``. Cleartext ``http://`` listeners start
    as HTTP/1.1 and Hypercorn upgrades prior-knowledge ``PRI *`` to HTTP/2,
    which is the default Cursor Agent CLI ``Run`` path on ``--agent-endpoint``.
    """
    config.keep_alive_timeout = float(CURSOR_AGENT_CLI_HTTP2_KEEPALIVE_SECONDS)
    config.h2_max_inbound_frame_size = int(CURSOR_AGENT_CLI_H2_MAX_INBOUND_FRAME_SIZE)
    alpn = list(getattr(config, "alpn_protocols", []) or [])
    if "h2" not in alpn:
        alpn.insert(0, "h2")
    if "http/1.1" not in alpn:
        alpn.append("http/1.1")
    config.alpn_protocols = alpn
    return config
