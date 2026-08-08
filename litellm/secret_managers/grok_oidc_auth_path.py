"""Canonical native Grok OIDC credential-path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

DEFAULT_GROK_OIDC_AUTH_FILE = "~/.grok/auth.json"
AAWM_GROK_OIDC_AUTH_FILE_ENV = "AAWM_GROK_OIDC_AUTH_FILE"
GROK_OIDC_AUTH_FILE_ENV_VARS = (
    "LITELLM_XAI_GROK_AUTH_FILE",
    "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
    "GROK_AUTH_FILE",
)
GROK_HOME_ENV = "GROK_HOME"

AuthPathValue = Union[str, os.PathLike[str]]
AuthPathValueGetter = Callable[[str], Any]


@dataclass(frozen=True)
class GrokOidcAuthPathResolution:
    """Expanded path and allow-listed winning source for safe telemetry."""

    path: Path = field(repr=False)
    source: str


def _clean_path_value(value: Any) -> Optional[str]:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _expanded_path(value: str) -> Path:
    return Path(value).expanduser()


def resolve_grok_oidc_auth_path(
    explicit_auth_file: Optional[AuthPathValue] = None,
    *,
    value_getter: Optional[AuthPathValueGetter] = None,
    default_auth_file: AuthPathValue = DEFAULT_GROK_OIDC_AUTH_FILE,
) -> GrokOidcAuthPathResolution:
    """Resolve native Grok OIDC auth path with deterministic precedence.

    Precedence is the AAWM override, an explicit non-default caller path,
    LiteLLM/native legacy variables, ``GROK_HOME/auth.json``, then the portable
    user-home default. The source is always a fixed label, never a configured
    path or credential value.
    """

    get_value = value_getter or os.getenv
    cleaned_default = _clean_path_value(default_auth_file) or DEFAULT_GROK_OIDC_AUTH_FILE
    default_path = _expanded_path(cleaned_default)

    aawm_auth_file = _clean_path_value(get_value(AAWM_GROK_OIDC_AUTH_FILE_ENV))
    if aawm_auth_file is not None:
        return GrokOidcAuthPathResolution(
            path=_expanded_path(aawm_auth_file),
            source=AAWM_GROK_OIDC_AUTH_FILE_ENV,
        )

    explicit_value = _clean_path_value(explicit_auth_file)
    if explicit_value is not None:
        explicit_path = _expanded_path(explicit_value)
        if explicit_path != default_path:
            return GrokOidcAuthPathResolution(
                path=explicit_path,
                source="explicit",
            )

    for env_name in GROK_OIDC_AUTH_FILE_ENV_VARS:
        configured_path = _clean_path_value(get_value(env_name))
        if configured_path is not None:
            return GrokOidcAuthPathResolution(
                path=_expanded_path(configured_path),
                source=env_name,
            )

    grok_home = _clean_path_value(get_value(GROK_HOME_ENV))
    if grok_home is not None:
        return GrokOidcAuthPathResolution(
            path=_expanded_path(grok_home) / "auth.json",
            source=GROK_HOME_ENV,
        )

    return GrokOidcAuthPathResolution(path=default_path, source="default")
