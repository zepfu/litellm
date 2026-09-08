"""Shared managed xAI OAuth credential resolution and identity helpers.

This module contains only non-I/O credential policy. Request paths remain
read-only and sidecar paths remain responsible for locking, refresh, and
atomic publication.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

DEFAULT_XAI_OAUTH_AUTH_FILE = "~/.litellm/xai/oauth-auth.json"
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
    """Canonical expanded auth path and a safe source label."""

    path: Path
    source: str


@dataclass(frozen=True)
class XaiOAuthScopeResolution:
    """Canonical scope and a safe source label."""

    scope: str
    source: str


@dataclass(frozen=True)
class XaiOAuthCredentialResolution:
    """Canonical managed file/scope pair used by request and sidecar paths."""

    auth_file: Path
    auth_file_source: str
    scope: str
    scope_source: str


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
    """Resolve the managed xAI auth file with one deterministic precedence.

    The AAWM override is intentionally first because the sidecar and
    containerized request paths commonly receive different environment
    surfaces. An explicit non-default CLI/config value follows, then the
    LiteLLM managed and migrated-path variables, and finally the portable
    default.
    """

    get_value = value_getter or os.getenv
    cleaned_default = (
        _clean_string(default_auth_file) or DEFAULT_XAI_OAUTH_AUTH_FILE
    )
    default_path = _expand_path(cleaned_default)

    configured_paths: list[tuple[str, Path]] = []
    aawm_auth_file = _clean_string(get_value("AAWM_XAI_OAUTH_AUTH_FILE"))
    if aawm_auth_file is not None:
        configured_paths.append(
            ("AAWM_XAI_OAUTH_AUTH_FILE", _expand_path(aawm_auth_file))
        )
    explicit_value = _clean_string(explicit_auth_file)
    if explicit_value is not None:
        explicit_path = _expand_path(explicit_value)
        if explicit_value != cleaned_default:
            configured_paths.append(("explicit", explicit_path))

    for env_name in XAI_OAUTH_AUTH_FILE_ENV_VARS:
        configured_path = _clean_string(get_value(env_name))
        if configured_path is not None:
            configured_paths.append(
                (env_name, _expand_path(configured_path))
            )

    if configured_paths:
        selected_path = configured_paths[0][1]
        if any(
            path.resolve(strict=False) != selected_path.resolve(strict=False)
            for _source, path in configured_paths[1:]
        ):
            sources = ", ".join(source for source, _path in configured_paths)
            raise ValueError(
                "Conflicting xAI OAuth auth-file configuration sources: "
                f"{sources}."
            )
        return XaiOAuthAuthPathResolution(
            path=selected_path,
            source=configured_paths[0][0],
        )

    return XaiOAuthAuthPathResolution(path=default_path, source="default")


def resolve_xai_oauth_scope(
    explicit_scope: Optional[str] = None,
    *,
    value_getter: Optional[ValueGetter] = None,
    env_names: Sequence[str] = XAI_OAUTH_SCOPE_ENV_VARS,
    default_scope: str = DEFAULT_XAI_OAUTH_SCOPE,
) -> XaiOAuthScopeResolution:
    """Resolve one exact credential scope with deterministic precedence."""

    get_value = value_getter or os.getenv
    configured_scopes: list[tuple[str, str]] = []
    explicit_value = _clean_string(explicit_scope)
    if explicit_value is not None:
        configured_scopes.append(("explicit", explicit_value))
    for env_name in env_names:
        configured_scope = _clean_string(get_value(env_name))
        if configured_scope is not None:
            configured_scopes.append((env_name, configured_scope))

    if configured_scopes:
        selected_scope = configured_scopes[0][1]
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
            source=configured_scopes[0][0],
        )

    return XaiOAuthScopeResolution(
        scope=_clean_string(default_scope) or DEFAULT_XAI_OAUTH_SCOPE,
        source="default",
    )


def resolve_xai_oauth_credentials(
    explicit_auth_file: Optional[AuthPathValue] = None,
    explicit_scope: Optional[str] = None,
    *,
    value_getter: Optional[ValueGetter] = None,
    scope_env_names: Sequence[str] = XAI_OAUTH_SCOPE_ENV_VARS,
    default_auth_file: AuthPathValue = DEFAULT_XAI_OAUTH_AUTH_FILE,
    default_scope: str = DEFAULT_XAI_OAUTH_SCOPE,
) -> XaiOAuthCredentialResolution:
    """Resolve the canonical managed auth file and scope together."""

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
    return XaiOAuthCredentialResolution(
        auth_file=path_resolution.path,
        auth_file_source=path_resolution.source,
        scope=scope_resolution.scope,
        scope_source=scope_resolution.source,
    )


def looks_like_xai_oauth_credential(value: Mapping[str, Any]) -> bool:
    """Return whether a mapping contains at least one usable token field."""

    return any(
        _clean_string(value.get(field_name)) is not None
        for field_name in ("key", "access_token", "refresh_token")
    )


def _is_unambiguous_flat_record(payload: Mapping[str, Any]) -> bool:
    if not looks_like_xai_oauth_credential(payload):
        return False

    # A top-level token plus another credential-like nested record is not a
    # legacy flat record. Treating that mixed shape as flat would bypass the
    # configured scope and reintroduce order-dependent selection.
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
    """Select exactly one record without a first-nested-record fallback."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{provider_label} auth file must contain a JSON object.")

    resolved_scope = _clean_string(scope)
    if resolved_scope is None:
        raise ValueError(f"{provider_label} credential scope must not be empty.")

    if _is_unambiguous_flat_record(payload):
        # Callers mutate the selected record during sidecar refresh. The JSON
        # reader returns a dict in all supported paths; reject exotic mapping
        # implementations rather than copy and mutate the wrong object.
        if isinstance(payload, MutableMapping):
            return payload
        raise ValueError(f"{provider_label} credential record is not mutable.")

    scoped_record = payload.get(resolved_scope)
    if not isinstance(scoped_record, MutableMapping):
        raise ValueError(
            f"{provider_label} auth file does not contain the configured "
            "credential scope. Exact scope matching is required."
        )
    if not looks_like_xai_oauth_credential(scoped_record):
        raise ValueError(
            f"{provider_label} credential scope does not contain a usable "
            "credential record."
        )
    return scoped_record


def credential_identity(
    auth_path: Optional[AuthPathValue] = None,
    record: Optional[Mapping[str, Any]] = None,
    *,
    scope: Optional[str] = None,
) -> Optional[str]:
    """Return a stable nonsecret generation identity.

    The identity deliberately excludes access/refresh/id tokens and filesystem
    paths. File metadata plus nonsecret lifecycle fields distinguishes atomic
    generations while remaining safe for scheduler state and telemetry.
    """

    stat_parts: dict[str, Any] = {}
    if auth_path is not None:
        try:
            stat_result = Path(auth_path).expanduser().stat()
        except OSError:
            stat_result = None
        if stat_result is not None:
            stat_parts = {
                "mtime_ns": stat_result.st_mtime_ns,
                "size": stat_result.st_size,
            }
    safe_record: dict[str, Any] = {}
    if isinstance(record, Mapping):
        for field_name in (
            "expires_at",
            "expires_in",
            "issued_at",
            "obtained_at",
            "refreshed_at",
            "oidc_client_id",
            "client_id",
            "token_type",
            "source",
        ):
            value = record.get(field_name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_record[field_name] = value
    if not stat_parts and not safe_record and _clean_string(scope) is None:
        return None
    payload = {
        "scope": _clean_string(scope),
        "stat": stat_parts,
        "record": safe_record,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"
