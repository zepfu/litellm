"""Shared managed xAI OAuth file and scope resolution.

This module is intentionally side-effect free. Request, refresh, health, and
preflight callers all use the same exact-scope policy, while refresh callers
retain ownership of file locking and publication.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

DEFAULT_XAI_OAUTH_AUTH_FILE = "~/.litellm/xai/oauth-auth.json"
DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS = 300
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


def credential_identity(
    auth_path: Optional[AuthPathValue] = None,
    record: Optional[Mapping[str, Any]] = None,
    *,
    scope: Optional[str] = None,
    stat_result: Optional[os.stat_result] = None,
) -> Optional[str]:
    """Return a stable nonsecret identity for one published credential generation.

    Access and refresh token values are intentionally excluded. Callers that
    already opened and validated a file should pass its descriptor metadata so
    request code does not perform a second blocking filesystem operation.
    """

    if stat_result is None and auth_path is not None:
        try:
            stat_result = Path(auth_path).stat()
        except OSError:
            stat_result = None

    stat_parts: dict[str, Any] = {}
    if stat_result is not None:
        stat_parts = {
            "st_dev": stat_result.st_dev,
            "st_ino": stat_result.st_ino,
            "st_mtime_ns": stat_result.st_mtime_ns,
            "st_ctime_ns": stat_result.st_ctime_ns,
            "st_size": stat_result.st_size,
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


def _account_identity_value(value: Any) -> Optional[str | int]:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def credential_account_identity(
    record: Optional[Mapping[str, Any]] = None,
    *,
    scope: Optional[str] = None,
) -> Optional[str]:
    """Return a stable nonsecret identity only when account evidence is present."""

    account_evidence: dict[str, Any] = {}
    if isinstance(record, Mapping):
        for field_name in (
            "account_id",
            "source_account_id",
            "subject",
        ):
            value = _account_identity_value(record.get(field_name))
            if value is not None:
                account_evidence[field_name] = value
    if not account_evidence:
        return None

    context: dict[str, Any] = {}
    if isinstance(record, Mapping):
        for field_name in ("client_id", "oidc_client_id"):
            value = _account_identity_value(record.get(field_name))
            if value is not None:
                context[field_name] = value

    payload = {
        "account": account_evidence,
        "context": context,
        "scope": _clean_string(scope),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


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


def credential_access_token(record: Mapping[str, Any]) -> Optional[str]:
    """Return the access credential without exposing its value in metadata."""
    token = _clean_string(record.get("access_token")) or _clean_string(
        record.get("key")
    )
    return token


def credential_refresh_token(record: Mapping[str, Any]) -> Optional[str]:
    return _clean_string(record.get("refresh_token"))


def parse_xai_timestamp(value: Any) -> Optional[float]:
    """Parse epoch seconds/milliseconds or an ISO-8601 timestamp."""
    numeric = _as_finite_number(value)
    if numeric is not None:
        if numeric >= 1_000_000_000_000:
            numeric /= 1000.0
        return numeric
    if isinstance(value, datetime):
        timestamp = value.timestamp()
        return timestamp if math.isfinite(timestamp) else None
    normalized = _clean_string(value)
    if normalized is None:
        return None
    try:
        numeric = float(normalized)
    except ValueError:
        numeric = None
    if numeric is not None:
        if not math.isfinite(numeric):
            return None
        if numeric >= 1_000_000_000_000:
            numeric /= 1000.0
        return numeric
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    timestamp = parsed.astimezone(timezone.utc).timestamp()
    return timestamp if math.isfinite(timestamp) else None


def parse_xai_expires_at(value: Any) -> Optional[datetime]:
    timestamp = parse_xai_timestamp(value)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def _jwt_time_claims(access_token: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    try:
        parts = access_token.split(".")
        if len(parts) != 3 or any(not part for part in parts):
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(claims, Mapping):
        return None
    issued_at = _as_finite_number(claims.get("iat"))
    expires_at = _as_finite_number(claims.get("exp"))
    if issued_at is None or expires_at is None or expires_at <= issued_at:
        return None
    return issued_at, expires_at


def credential_issued_at(record: Mapping[str, Any]) -> Optional[float]:
    for field_name in ("issued_at", "obtained_at", "refreshed_at"):
        timestamp = parse_xai_timestamp(record.get(field_name))
        if timestamp is not None:
            return timestamp
    return None


def credential_expires_at(record: Mapping[str, Any]) -> Optional[datetime]:
    """Resolve expiry from explicit, JWT, or persisted lifetime metadata."""
    explicit_expiry = parse_xai_expires_at(record.get("expires_at"))
    if explicit_expiry is not None:
        return explicit_expiry

    jwt_claims = _jwt_time_claims(credential_access_token(record))
    if jwt_claims is not None:
        try:
            return datetime.fromtimestamp(jwt_claims[1], tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    issued_at = credential_issued_at(record)
    lifetime = _as_finite_number(record.get("expires_in"))
    if issued_at is not None and lifetime is not None and lifetime > 0:
        try:
            return datetime.fromtimestamp(
                issued_at + lifetime,
                tz=timezone.utc,
            )
        except (OSError, OverflowError, ValueError):
            return None
    return None


def issued_lifetime_metadata(
    *,
    expires_in: Any = None,
    access_token: Optional[str] = None,
    expires_at: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
) -> Tuple[Optional[float], str]:
    """Return the authoritative issued lifetime and its source."""
    provider_lifetime = _as_finite_number(expires_in)
    if provider_lifetime is not None and provider_lifetime > 0:
        return provider_lifetime, "expires_in"

    jwt_claims = _jwt_time_claims(access_token)
    if jwt_claims is not None:
        return jwt_claims[1] - jwt_claims[0], "jwt"

    persisted_issued_at = next(
        (
            timestamp
            for timestamp in (
                parse_xai_timestamp(issued_at),
                parse_xai_timestamp(obtained_at),
                parse_xai_timestamp(refreshed_at),
            )
            if timestamp is not None
        ),
        None,
    )
    persisted_expires_at = parse_xai_timestamp(expires_at)
    if (
        persisted_issued_at is not None
        and persisted_expires_at is not None
        and persisted_expires_at > persisted_issued_at
    ):
        return persisted_expires_at - persisted_issued_at, "persisted_timestamp"
    return None, "fallback"


def refresh_threshold_metadata(
    record: Mapping[str, Any],
    *,
    min_seconds: float = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS,
) -> Tuple[float, str, bool]:
    """Return the proactive refresh threshold without changing route safety."""
    lifetime, source = issued_lifetime_metadata(
        expires_in=record.get("expires_in"),
        access_token=credential_access_token(record),
        expires_at=record.get("expires_at"),
        issued_at=record.get("issued_at"),
        obtained_at=record.get("obtained_at"),
        refreshed_at=record.get("refreshed_at"),
    )
    if lifetime is None:
        return float(min_seconds), "fallback", True
    return max(float(min_seconds), lifetime * 0.5), source, False


def _resolve_now(now: Optional[Callable[[], datetime]]) -> datetime:
    value = now() if now is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def evaluate_xai_oauth_credential_lifecycle(
    record: Mapping[str, Any],
    *,
    now: Optional[Callable[[], datetime]] = None,
    route_safety_buffer_seconds: float = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS,
    refresh_min_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Classify one credential for request readiness and sidecar refresh.

    ``route_safety_buffer_seconds`` controls whether an access token may be
    sent. ``refresh_min_seconds`` controls the minimum proactive refresh
    threshold. Keeping them separate lets a credential be refresh-due while
    remaining route-usable.
    """
    observed_at = _resolve_now(now)
    route_buffer = max(0.0, float(route_safety_buffer_seconds))
    refresh_min = (
        route_buffer
        if refresh_min_seconds is None
        else max(0.0, float(refresh_min_seconds))
    )
    structural_valid = looks_like_xai_oauth_credential(record)
    access_available = credential_access_token(record) is not None
    refresh_possible = credential_refresh_token(record) is not None
    expires_at = credential_expires_at(record)
    expiry_available = expires_at is not None
    threshold_seconds, threshold_source, threshold_degraded = (
        refresh_threshold_metadata(record, min_seconds=refresh_min)
    )
    expired = bool(expiry_available and expires_at <= observed_at)
    refresh_due = not access_available or not expiry_available or (
        observed_at >= expires_at - timedelta(seconds=threshold_seconds)
    )
    route_unusable = (
        not structural_valid
        or not access_available
        or not expiry_available
        or expired
        or observed_at >= expires_at - timedelta(seconds=route_buffer)
    )
    route_usable = not route_unusable

    if not structural_valid:
        health_status = "malformed"
    elif expired:
        health_status = "expired"
    elif route_usable:
        health_status = "fresh"
    else:
        health_status = "degraded"

    if not structural_valid:
        state = "malformed"
    elif not access_available:
        state = "access_unavailable"
    elif not expiry_available:
        state = "expiry_unavailable"
    elif expired:
        state = "expired"
    elif refresh_due:
        state = "refresh_due"
    else:
        state = "fresh"

    route_unusable_reason: Optional[str] = None
    if not structural_valid:
        route_unusable_reason = "structurally_invalid"
    elif not access_available:
        route_unusable_reason = "access_credential_unavailable"
    elif not expiry_available:
        route_unusable_reason = "expiry_unavailable"
    elif expired:
        route_unusable_reason = "expired"
    elif observed_at >= expires_at - timedelta(seconds=route_buffer):
        route_unusable_reason = "inside_route_safety_buffer"

    return {
        "observed_at": observed_at,
        "expires_at": expires_at,
        "refresh_due_at": (
            expires_at - timedelta(seconds=threshold_seconds)
            if expiry_available
            else None
        ),
        "route_unusable_at": (
            expires_at - timedelta(seconds=route_buffer)
            if expiry_available
            else None
        ),
        "structurally_valid": structural_valid,
        "access_available": access_available,
        "refresh_possible": refresh_possible,
        "route_usable": route_usable,
        "route_unusable": route_unusable,
        "refresh_due": refresh_due,
        "expired": expired,
        "terminal_unrefreshable": refresh_due and not refresh_possible,
        "credential_health": health_status,
        "state": state,
        "route_unusable_reason": route_unusable_reason,
        "refresh_threshold_seconds": threshold_seconds,
        "refresh_threshold_source": threshold_source,
        "refresh_threshold_degraded": threshold_degraded,
        "expiry_available": expiry_available,
    }


def _as_finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
