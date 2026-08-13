"""Direct Cohere credential and target-base runtime.

Resolves the canonical ``COHERE_API_KEY`` credential and the native Cohere
API base for AAWM direct-provider routes. This module performs no provider
I/O and never creates an HTTP client; it exists so credential failures are
raised as one typed, sanitized error before any transport work.

Compatibility note: ``COHERE_KEY`` is accepted only at the Compose level,
which maps it into ``COHERE_API_KEY`` before the process starts. The runtime
itself reads only the canonical name. ``CO_API_KEY`` and inbound TUI
authorization values are never accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlsplit

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException
from litellm.secret_managers.main import get_secret_str


COHERE_CANONICAL_API_KEY_ENV_VAR = "COHERE_API_KEY"
COHERE_API_BASE_ENV_VAR = "COHERE_API_BASE"
COHERE_DEFAULT_API_BASE = "https://api.cohere.com/v2/chat"
COHERE_CHAT_V2_PATH = "/v2/chat"
COHERE_ALLOWED_API_HOSTS = frozenset({"api.cohere.com", "api.cohere.ai"})

# Compatibility-only source name consumed by docker-compose.dev.yml. It is
# intentionally absent from runtime resolution so provider code can never
# depend on it.
COHERE_COMPOSE_COMPATIBILITY_KEY_ENV_VAR = "COHERE_KEY"


class CohereMissingCredentialError(ProxyException):
    """Typed missing-credential failure raised before any provider I/O."""

    def __init__(self) -> None:
        super().__init__(
            message=(
                "Direct Cohere route is unavailable: canonical credential "
                "COHERE_API_KEY is missing. Configure COHERE_API_KEY for the "
                "AAWM runtime; COHERE_KEY is accepted only as a Compose-level "
                "compatibility source mapped into COHERE_API_KEY."
            ),
            type="authentication_error",
            param=None,
            code=401,
        )


class CohereInvalidTargetBaseError(ProxyException):
    """Typed invalid-target failure raised before any provider I/O."""

    def __init__(self) -> None:
        super().__init__(
            message=(
                "Invalid COHERE_API_BASE. Configure an HTTPS URL for the "
                "native Cohere Chat V2 endpoint with host api.cohere.com or "
                "api.cohere.ai, path /v2/chat, no userinfo/query/fragment, "
                "and no port or port 443."
            ),
            type="invalid_request_error",
            param=COHERE_API_BASE_ENV_VAR,
            code=400,
        )


def _default_get_secret(name: str) -> Optional[str]:
    return get_secret_str(name)


def _default_clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {'"', "'"}
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


@dataclass(frozen=True)
class CohereRuntimeDependencies:
    """Callbacks supplied by the passthrough host during integration."""

    get_secret: Callable[[str], Optional[str]]
    clean_secret_string: Callable[[Optional[str]], Optional[str]]
    log_debug: Callable[..., None]


DEFAULT_COHERE_RUNTIME_DEPENDENCIES = CohereRuntimeDependencies(
    get_secret=_default_get_secret,
    clean_secret_string=_default_clean_secret_string,
    log_debug=verbose_proxy_logger.debug,
)

_runtime_dependencies = DEFAULT_COHERE_RUNTIME_DEPENDENCIES


def configure_cohere_runtime(
    dependencies: CohereRuntimeDependencies,
) -> None:
    """Install the callbacks used by the extracted Cohere runtime."""

    global _runtime_dependencies
    _runtime_dependencies = dependencies


def _get_cohere_api_key() -> Optional[str]:
    """Resolve only the canonical ``COHERE_API_KEY`` credential.

    Values are whitespace-trimmed and optionally unquoted; ``None`` is
    returned when no usable value exists. No key material is logged.
    """
    return _runtime_dependencies.clean_secret_string(
        _runtime_dependencies.get_secret(COHERE_CANONICAL_API_KEY_ENV_VAR)
    )


def _require_cohere_api_key() -> str:
    """Return the canonical Cohere credential or fail before provider I/O."""
    api_key = _get_cohere_api_key()
    if api_key is None:
        _runtime_dependencies.log_debug(
            "Direct Cohere credential resolution failed: canonical env var "
            "%s is missing or blank after cleanup",
            COHERE_CANONICAL_API_KEY_ENV_VAR,
        )
        raise CohereMissingCredentialError()
    return api_key


def _get_cohere_target_base() -> str:
    """Resolve the native Cohere Chat V2 URL with validation.

    Defaults to ``https://api.cohere.com/v2/chat`` after quote/whitespace
    cleanup. Configured values must identify the native Chat V2 endpoint.
    """
    cleaned = _runtime_dependencies.clean_secret_string(
        _runtime_dependencies.get_secret(COHERE_API_BASE_ENV_VAR)
    )
    if not cleaned:
        return COHERE_DEFAULT_API_BASE

    try:
        parsed = urlsplit(cleaned)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        raise CohereInvalidTargetBaseError() from None

    if (
        "?" in cleaned
        or "#" in cleaned
        or parsed.scheme != "https"
        or hostname not in COHERE_ALLOWED_API_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.path != COHERE_CHAT_V2_PATH
        or parsed.query
        or parsed.fragment
    ):
        raise CohereInvalidTargetBaseError()

    return f"https://{hostname}{COHERE_CHAT_V2_PATH}"
