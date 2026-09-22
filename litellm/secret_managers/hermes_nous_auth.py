"""Read-only Hermes Nous Portal invoke-JWT loader.

LiteLLM request handling is a consumer of ``~/.hermes/auth.json``. The
provider-status sidecar is the sole automatic writer. This module never
writes Hermes state and never treats xAI or Copilot slots as Nous.

Request loading, refresh defaults, sidecar configuration, and passive health
share one auth-file precedence:

explicit path, ``AAWM_NOUS_OAUTH_AUTH_FILE``, ``LITELLM_NOUS_OAUTH_AUTH_FILE``,
``LITELLM_HERMES_AUTH_FILE``, ``AAWM_HERMES_AUTH_FILE``, then
``~/.hermes/auth.json``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from litellm.secret_managers.credential_error_sanitizer import (
    sanitize_credential_error_message,
)

logger = logging.getLogger("LiteLLM")

_DEFAULT_HERMES_AUTH_PATH = os.path.join("~", ".hermes", "auth.json")
_EXPLICIT_SOURCE = "explicit"
_DEFAULT_SOURCE = "default"
NOUS_SPECIFIC_AUTH_FILE_ENV_VARS = (
    "AAWM_NOUS_OAUTH_AUTH_FILE",
    "LITELLM_NOUS_OAUTH_AUTH_FILE",
)
HERMES_WIDE_AUTH_FILE_ENV_VARS = (
    "LITELLM_HERMES_AUTH_FILE",
    "AAWM_HERMES_AUTH_FILE",
)
SHARED_NOUS_AUTH_FILE_PRECEDENCE = (
    _EXPLICIT_SOURCE,
    *NOUS_SPECIFIC_AUTH_FILE_ENV_VARS,
    *HERMES_WIDE_AUTH_FILE_ENV_VARS,
)
LEGACY_REQUEST_NOUS_AUTH_FILE_PRECEDENCE = (
    "LITELLM_NOUS_OAUTH_AUTH_FILE",
    "LITELLM_HERMES_AUTH_FILE",
    "AAWM_HERMES_AUTH_FILE",
)
LEGACY_SIDECAR_NOUS_AUTH_FILE_PRECEDENCE = (
    "AAWM_NOUS_OAUTH_AUTH_FILE",
    _EXPLICIT_SOURCE,
    "LITELLM_NOUS_OAUTH_AUTH_FILE",
    "AAWM_HERMES_AUTH_FILE",
)
_SUPPORTED_AUTH_FILE_ENV_VARS = (
    *NOUS_SPECIFIC_AUTH_FILE_ENV_VARS,
    *HERMES_WIDE_AUTH_FILE_ENV_VARS,
)
_warned_migration_keys: set[tuple[str, ...]] = set()
write_and_publish_private_text = None

ValueGetter = Callable[[str], Any]


@dataclass(frozen=True)
class HermesNousAuthPathResolution:
    """Expanded auth path and the fixed source label that selected it."""

    path: str = field(repr=False)
    source: str


def _clean_path_value(value: Any) -> Optional[str]:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _expand_path(value: str) -> str:
    return str(Path(value).expanduser())


def _same_auth_file(left: str, right: str) -> bool:
    return str(Path(left).expanduser().resolve(strict=False)) == str(
        Path(right).expanduser().resolve(strict=False)
    )


def _explicit_path(explicit_auth_file: Any) -> Optional[str]:
    """Treat the portable default sentinel as an unspecified caller path."""

    cleaned = _clean_path_value(explicit_auth_file)
    if cleaned is None or cleaned == _DEFAULT_HERMES_AUTH_PATH:
        return None
    return cleaned


def _configured_auth_files(
    explicit_auth_file: Any,
    get_value: ValueGetter,
) -> Mapping[str, str]:
    """Collect every supported input that is present for this resolution."""

    configured: dict[str, str] = {}
    explicit_path = _explicit_path(explicit_auth_file)
    if explicit_path is not None:
        configured[_EXPLICIT_SOURCE] = explicit_path
    for name in _SUPPORTED_AUTH_FILE_ENV_VARS:
        cleaned = _clean_path_value(get_value(name))
        if cleaned is not None:
            configured[name] = cleaned
    return configured


def _select_auth_file(
    precedence: Sequence[str],
    configured: Mapping[str, str],
) -> tuple[str, str]:
    for source in precedence:
        raw_path = configured.get(source)
        if raw_path:
            return _expand_path(raw_path), source
    return _expand_path(_DEFAULT_HERMES_AUTH_PATH), _DEFAULT_SOURCE


def _warn_legacy_precedence_mismatch(
    shared_path: str,
    shared_source: str,
    configured: Mapping[str, str],
) -> None:
    """Warn when an old consumer order would open a different file.

    The message names sources only. It must not include the path or any
    credential material.
    """

    mismatches: list[str] = []
    for consumer, precedence in (
        ("request", LEGACY_REQUEST_NOUS_AUTH_FILE_PRECEDENCE),
        ("sidecar", LEGACY_SIDECAR_NOUS_AUTH_FILE_PRECEDENCE),
    ):
        legacy_path, legacy_source = _select_auth_file(precedence, configured)
        if not _same_auth_file(shared_path, legacy_path):
            mismatches.append(f"legacy {consumer} source {legacy_source}")
    if not mismatches:
        return
    warning_key = (shared_source, *mismatches)
    if warning_key in _warned_migration_keys:
        return
    _warned_migration_keys.add(warning_key)
    logger.warning(
        "Nous auth-file precedence migration selected source %s; %s would "
        "select a different file.",
        shared_source,
        " and ".join(mismatches),
    )


def resolve_hermes_nous_auth_resolution(
    explicit_auth_file: Any = None,
    *,
    value_getter: Optional[ValueGetter] = None,
) -> HermesNousAuthPathResolution:
    """Resolve one expanded Nous auth file for every credential consumer."""

    get_value = value_getter or os.getenv
    configured = _configured_auth_files(explicit_auth_file, get_value)
    path, source = _select_auth_file(SHARED_NOUS_AUTH_FILE_PRECEDENCE, configured)
    _warn_legacy_precedence_mismatch(path, source, configured)
    return HermesNousAuthPathResolution(path=path, source=source)


def resolve_hermes_nous_auth_path(
    explicit_auth_file: Any = None,
    *,
    value_getter: Optional[ValueGetter] = None,
) -> str:
    """Return the expanded path from the shared Nous auth-file resolver."""

    return resolve_hermes_nous_auth_resolution(
        explicit_auth_file,
        value_getter=value_getter,
    ).path


def _usable_access_token(slot: Any) -> Optional[str]:
    if not isinstance(slot, dict):
        return None
    token = slot.get("access_token")
    if isinstance(token, str) and token.strip():
        return token
    return None


def _slot_from_document(payload: Any, collection: str) -> Any:
    if not isinstance(payload, dict):
        return None
    group = payload.get(collection)
    if not isinstance(group, dict):
        return None
    return group.get("nous")


def load_nous_invoke_jwt() -> str:
    path = resolve_hermes_nous_auth_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        token = _usable_access_token(_slot_from_document(payload, "providers"))
        if token is not None:
            return token
        token = _usable_access_token(_slot_from_document(payload, "credential_pool"))
        if token is not None:
            return token
        raise RuntimeError("Nous Portal invoke JWT is missing from Hermes auth")
    except Exception as exc:
        message = sanitize_credential_error_message(str(exc))
        raise type(exc)(message) from None
