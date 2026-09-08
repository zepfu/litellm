"""Shared managed xAI OAuth file and scope resolution.

This module is intentionally side-effect free. Request, refresh, health, and
preflight callers all use the same exact-scope policy, while refresh callers
retain ownership of file locking and publication.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

DEFAULT_XAI_OAUTH_AUTH_FILE = "~/.litellm/xai/oauth-auth.json"
DEFAULT_XAI_OAUTH_LOCK_FILE = "~/.litellm/xai/oauth-auth.json.lock"
DEFAULT_XAI_OAUTH_SCOPE = (
    "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
)

XAI_OAUTH_AUTH_FILE_ENV_VARS = (
    "LITELLM_XAI_OAUTH_AUTH_FILE",
    "LITELLM_XAI_OAUTH_MIGRATED_AUTH_FILE",
)
XAI_OAUTH_SCOPE_ENV_VARS = (
    "AAWM_XAI_OAUTH_SCOPE",
    "LITELLM_XAI_OAUTH_SCOPE",
)

AuthPathValue = str | os.PathLike[str]
ValueGetter = Callable[[str], Any]


@dataclass(frozen=True)
class XaiOAuthAuthPathResolution:
    """The selected configured path and its frozen canonical read target."""

    path: Path
    canonical_path: Path
    source: str


@dataclass(frozen=True)
class XaiOAuthScopeResolution:
    """The selected exact scope and its provenance label."""

    scope: str
    source: str


@dataclass(frozen=True)
class XaiOAuthCredentialResolution:
    """The single managed file/scope binding shared by all consumers."""

    auth_file: Path
    canonical_auth_file: Path
    auth_file_source: str
    scope: str
    scope_source: str
    credential_identity: str


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _expand_path(value: str) -> Path:
    return Path(value).expanduser()


def resolve_xai_oauth_auth_path(
    explicit_auth_file: Optional[AuthPathValue] = None,
    *,
    value_getter: Optional[ValueGetter] = None,
    default_auth_file: AuthPathValue = DEFAULT_XAI_OAUTH_AUTH_FILE,
) -> XaiOAuthAuthPathResolution:
    """Resolve one managed auth path and reject conflicting configuration."""

    get_value = value_getter or os.getenv
    cleaned_default = (
        _clean_string(default_auth_file) or DEFAULT_XAI_OAUTH_AUTH_FILE
    )
    default_path = _expand_path(cleaned_default)

    configured_paths: list[tuple[str, Path, Path]] = []

    def add_candidate(source: str, value: Any) -> None:
        cleaned = _clean_string(value)
        if cleaned is None:
            return
        path = _expand_path(cleaned)
        configured_paths.append((source, path, path.resolve(strict=False)))

    # Precedence chooses the source label only after every supplied value
    # agrees. In particular, an explicit value equal to the default remains
    # explicit input and participates in conflict detection.
    add_candidate("AAWM_XAI_OAUTH_AUTH_FILE", get_value("AAWM_XAI_OAUTH_AUTH_FILE"))
    add_candidate("explicit", explicit_auth_file)
    for env_name in XAI_OAUTH_AUTH_FILE_ENV_VARS:
        add_candidate(env_name, get_value(env_name))

    if configured_paths:
        selected_source, selected_path, selected_canonical = configured_paths[0]
        if any(
            canonical_path != selected_canonical
            for _source, _path, canonical_path in configured_paths[1:]
        ):
            sources = ", ".join(source for source, _path, _canonical in configured_paths)
            raise ValueError(
                "Conflicting xAI OAuth auth-file configuration sources: "
                f"{sources}."
            )
        return XaiOAuthAuthPathResolution(
            path=selected_path,
            canonical_path=selected_canonical,
            source=selected_source,
        )

    return XaiOAuthAuthPathResolution(
        path=default_path,
        canonical_path=default_path.resolve(strict=False),
        source="default",
    )


def resolve_xai_oauth_scope(
    explicit_scope: Optional[str] = None,
    *,
    value_getter: Optional[ValueGetter] = None,
    env_names: Sequence[str] = XAI_OAUTH_SCOPE_ENV_VARS,
    default_scope: str = DEFAULT_XAI_OAUTH_SCOPE,
) -> XaiOAuthScopeResolution:
    """Resolve one exact managed scope and reject conflicting configuration."""

    get_value = value_getter or os.getenv
    configured_scopes: list[tuple[str, str]] = []

    def add_candidate(source: str, value: Any) -> None:
        cleaned = _clean_string(value)
        if cleaned is not None:
            configured_scopes.append((source, cleaned))

    add_candidate("explicit", explicit_scope)
    for env_name in env_names:
        add_candidate(env_name, get_value(env_name))

    if configured_scopes:
        selected_source, selected_scope = configured_scopes[0]
        if any(
            scope_value != selected_scope
            for _source, scope_value in configured_scopes[1:]
        ):
            sources = ", ".join(source for source, _scope in configured_scopes)
            raise ValueError(
                "Conflicting xAI OAuth scope configuration sources: "
                f"{sources}."
            )
        return XaiOAuthScopeResolution(
            scope=selected_scope,
            source=selected_source,
        )

    resolved_default = _clean_string(default_scope) or DEFAULT_XAI_OAUTH_SCOPE
    return XaiOAuthScopeResolution(scope=resolved_default, source="default")


def resolve_xai_oauth_credentials(
    explicit_auth_file: Optional[AuthPathValue] = None,
    explicit_scope: Optional[str] = None,
    *,
    value_getter: Optional[ValueGetter] = None,
    scope_env_names: Sequence[str] = XAI_OAUTH_SCOPE_ENV_VARS,
    default_auth_file: AuthPathValue = DEFAULT_XAI_OAUTH_AUTH_FILE,
    default_scope: str = DEFAULT_XAI_OAUTH_SCOPE,
) -> XaiOAuthCredentialResolution:
    """Resolve one immutable managed file/scope binding before file I/O."""

    path_resolution = resolve_xai_oauth_auth_path(
        explicit_auth_file,
        value_getter=value_getter,
        default_auth_file=default_auth_file,
    )
    scope_resolution = resolve_xai_oauth_scope(
        explicit_scope,
        value_getter=value_getter,
        env_names=scope_env_names,
        default_scope=default_scope,
    )
    credential_identity = _credential_identity(
        path_resolution.canonical_path,
        scope_resolution.scope,
    )
    return XaiOAuthCredentialResolution(
        auth_file=path_resolution.path,
        canonical_auth_file=path_resolution.canonical_path,
        auth_file_source=path_resolution.source,
        scope=scope_resolution.scope,
        scope_source=scope_resolution.source,
        credential_identity=credential_identity,
    )


def default_xai_oauth_lock_path(auth_file: AuthPathValue) -> Path:
    """Return the canonical sibling lock path for one managed auth file."""

    path = _expand_path(_clean_string(auth_file) or DEFAULT_XAI_OAUTH_AUTH_FILE)
    canonical_path = path.resolve(strict=False)
    return canonical_path.with_name(f"{canonical_path.name}.lock")


def resolve_xai_oauth_lock_path(
    auth_file: AuthPathValue,
    explicit_lock_file: Optional[AuthPathValue] = None,
) -> Path:
    """Resolve one lock identity without weakening auth-file safety checks.

    The default lock is derived from the canonical auth-file target, so
    relative paths and existing symlink aliases coordinate on one lock. The
    portable default lock is also remapped to that sibling for custom auth
    files, preventing a legacy fixed default from creating a second lock
    identity. Explicit lock values are accepted only when they resolve to that
    same canonical sibling; lock symlinks and auth-file collisions are rejected.
    """

    if explicit_lock_file is None:
        return default_xai_oauth_lock_path(auth_file)
    raw_lock = _clean_string(explicit_lock_file)
    if raw_lock is None:
        return default_xai_oauth_lock_path(auth_file)

    lock_path = _expand_path(raw_lock)
    if lock_path.is_symlink():
        raise ValueError("xAI OAuth lock path must not be a symlink.")
    canonical_lock = lock_path.resolve(strict=False)
    canonical_auth = _expand_path(
        _clean_string(auth_file) or DEFAULT_XAI_OAUTH_AUTH_FILE
    ).resolve(strict=False)
    if canonical_lock == canonical_auth:
        raise ValueError("xAI OAuth lock path must differ from the auth file.")
    canonical_default_lock = _expand_path(
        DEFAULT_XAI_OAUTH_LOCK_FILE
    ).resolve(strict=False)
    if (
        canonical_lock == canonical_default_lock
        and canonical_auth != canonical_default_lock
    ):
        canonical_lock = default_xai_oauth_lock_path(auth_file)
    canonical_sibling = default_xai_oauth_lock_path(auth_file)
    if canonical_lock != canonical_sibling:
        raise ValueError(
            "xAI OAuth lock path must resolve to the canonical auth-file "
            "sibling lock."
        )
    return canonical_sibling


def _credential_identity(canonical_path: Path, scope: str) -> str:
    """Hash only the canonical managed file and exact scope namespace."""

    identity_input = "\x00".join(
        ("xai-oauth-file-scope-v1", str(canonical_path), scope)
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(identity_input).hexdigest()}"


def looks_like_xai_oauth_credential(value: Mapping[str, Any]) -> bool:
    """Return whether a mapping contains at least one credential field."""

    return any(
        value.get(field_name)
        for field_name in ("key", "access_token", "refresh_token")
    )


def _is_unambiguous_flat_record(payload: Mapping[str, Any]) -> bool:
    if not looks_like_xai_oauth_credential(payload):
        return False

    # A top-level token alongside another credential-like mapping is a mixed
    # document, not a legacy flat record. Require its configured scope.
    return not any(
        isinstance(value, Mapping) and looks_like_xai_oauth_credential(value)
        for value in payload.values()
    )


def select_xai_oauth_credential_record(
    payload: Mapping[str, Any],
    scope: str,
    *,
    provider_label: str = "xAI OAuth",
) -> MutableMapping[str, Any]:
    """Select a mutable credential record without an order-based fallback.

    A legacy flat object remains valid only when it is unambiguous. Every
    multi-record object must contain the exact configured scope key; missing
    scopes fail before callers can make provider/token calls or write files.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{provider_label} credential selection failed: auth file must "
            "contain a JSON object."
        )
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(
            f"{provider_label} credential selection failed: scope must not be "
            "empty."
        )

    if _is_unambiguous_flat_record(payload):
        if isinstance(payload, MutableMapping):
            return payload
        raise ValueError(
            f"{provider_label} credential selection failed: record is not "
            "mutable."
        )

    scoped_record = payload.get(scope)
    if not isinstance(scoped_record, MutableMapping):
        raise ValueError(
            f"{provider_label} credential selection failed: auth file does "
            "not contain the configured scope. Exact scope matching is "
            "required."
        )
    if not looks_like_xai_oauth_credential(scoped_record):
        raise ValueError(
            f"{provider_label} credential selection failed: scope does not "
            "contain a usable credential record."
        )
    return scoped_record
