"""Wave 4 extraction: restored constants.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
"""

from __future__ import annotations

_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "authorization",
        "content-type",
        "user-agent",
        "x-goog-api-client",
        "x-goog-fieldmask",
        "x-goog-request-params",
        "x-goog-request-reason",
    }
)
