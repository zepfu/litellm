"""Wave 4 extraction: restored constants.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
"""

from __future__ import annotations

_OPENCODE_ZEN_DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"

_OPENCODE_ZEN_PROVIDER = "opencode_zen"

# OC-010: Zen-distinct egress families. Zen credential selection and egress
# bind the Zen credential family to the Zen target family so an OpenCode Go
# credential (or its target identity) can never be selected or sent on a Zen
# route. Go callers keep the historical host family until a later item.
_OPENCODE_ZEN_CREDENTIAL_FAMILY = "opencode_zen"
_OPENCODE_ZEN_TARGET_FAMILY = "opencode_zen"
_OPENCODE_GO_CREDENTIAL_FAMILY = "opencode_go"
_OPENCODE_GO_TARGET_FAMILY = "opencode_go"

_OPENCODE_ZEN_AUTH_FILE_ENV_VARS = (
    "LITELLM_OPENCODE_AUTH_FILE",
    "OPENCODE_AUTH_FILE",
)

_OPENCODE_ZEN_API_KEY_ENV_VARS = (
    "LITELLM_OPENCODE_API_KEY",
    "OPENCODE_API_KEY",
)

_OPENCODE_ZEN_DEFAULT_AUTH_PATHS = (
    "~/.local/share/opencode/auth.json",
    "~/.local/share/opencode/auth.json",
)

_OPENCODE_ZEN_FREE_MODELS = frozenset(
    {
        "big-pickle",
        "mini-v2.5",
        "north-mini-code",
        "nemotron-3-ultra",
        "deepseek-v4-flash-free",
    }
)

_OPENCODE_ZEN_ANTHROPIC_COMPLETION_MODELS = frozenset({"big-pickle"})

_OPENCODE_GO_DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

_OPENCODE_GO_PROVIDER = "opencode_go"

_OPENCODE_GO_FREE_MODELS = frozenset({"ox-alpha-free"})
