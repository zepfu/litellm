#!/usr/bin/env python3
"""Refresh a managed xAI OAuth auth JSON file for the provider-status sidecar."""

from __future__ import annotations

import base64
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, MutableMapping, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from litellm.secret_managers.credential_file_lock import credential_file_lock
from litellm.secret_managers.credential_file_metadata import (
    CredentialFileMetadata,
    apply_credential_file_metadata,
    resolve_credential_file_metadata,
    snapshot_credential_file_metadata,
)
from litellm.secret_managers.credential_file_write import (
    write_and_publish_private_text,
    write_private_file_text,
)

from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)
from litellm.secret_managers.xai_oauth_credentials import (
    DEFAULT_XAI_OAUTH_AUTH_FILE as _FOUNDATION_DEFAULT_XAI_OAUTH_AUTH_FILE,
    DEFAULT_XAI_OAUTH_LOCK_FILE as _FOUNDATION_DEFAULT_XAI_OAUTH_LOCK_FILE,
    DEFAULT_XAI_OAUTH_SCOPE as _FOUNDATION_DEFAULT_XAI_OAUTH_SCOPE,
    evaluate_xai_oauth_credential_lifecycle,
    refresh_threshold_metadata as _foundation_refresh_threshold_metadata,
    resolve_xai_oauth_credentials,
    resolve_xai_oauth_lock_path,
    select_xai_oauth_credential_record,
)

# Portable ~ defaults (expanded via Path.expanduser at use sites).
DEFAULT_XAI_OAUTH_AUTH_FILE = _FOUNDATION_DEFAULT_XAI_OAUTH_AUTH_FILE
DEFAULT_XAI_OAUTH_LOCK_FILE = _FOUNDATION_DEFAULT_XAI_OAUTH_LOCK_FILE
DEFAULT_XAI_OAUTH_SCOPE = _FOUNDATION_DEFAULT_XAI_OAUTH_SCOPE
DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS = 300
DEFAULT_XAI_OAUTH_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_XAI_OAUTH_AUTH_FILE_MODE = 0o600
DEFAULT_XAI_OAUTH_ERROR_MESSAGE_LIMIT = 500

_XAI_OAUTH_TERMINAL_REFRESH_ERROR_CLASSES = frozenset(
    {"invalid_grant", "refresh_token_reused"}
)
_XAI_OAUTH_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429})

# Keep historical module alias; redaction lives in secret_managers.
_SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES


class XaiOAuthRefreshError(ValueError):
    """A sanitized xAI OAuth refresh failure with a stable classification."""

    def __init__(self, error_class: str, message: str) -> None:
        self.error_class = error_class
        super().__init__(message)


def _issued_lifetime_seconds(
    *,
    expires_in: Any = None,
    access_token: Optional[str] = None,
    expires_at: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
) -> Optional[float]:
    lifetime, _source = _issued_lifetime_metadata(
        expires_in=expires_in,
        access_token=access_token,
        expires_at=expires_at,
        issued_at=issued_at,
        obtained_at=obtained_at,
        refreshed_at=refreshed_at,
    )
    return lifetime


def _issued_lifetime_metadata(
    *,
    expires_in: Any = None,
    access_token: Optional[str] = None,
    expires_at: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
) -> Tuple[Optional[float], str]:
    """Derive an issued lifetime and identify the authoritative source.

    Authority order is provider ``expires_in``, validated JWT ``iat``/``exp``,
    then a persisted obtained/refreshed timestamp paired with ``expires_at``.
    """
    provider_lifetime = _as_finite_number(expires_in)
    if provider_lifetime is not None and provider_lifetime > 0:
        return provider_lifetime, "expires_in"

    jwt_claims = _jwt_time_claims(access_token)
    if jwt_claims is not None:
        issued_timestamp, expiry_timestamp = jwt_claims
        return expiry_timestamp - issued_timestamp, "jwt"

    persisted_issued_at = _first_timestamp_seconds(
        issued_at,
        obtained_at,
        refreshed_at,
    )
    persisted_expires_at = _timestamp_seconds(expires_at)
    if (
        persisted_issued_at is not None
        and persisted_expires_at is not None
        and persisted_expires_at > persisted_issued_at
    ):
        return persisted_expires_at - persisted_issued_at, "persisted_timestamp"

    return None, "fallback"


def _refresh_threshold_seconds(
    *,
    expires_in: Any = None,
    access_token: Optional[str] = None,
    expires_at: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
    min_seconds: float = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS,
) -> float:
    """Return proportional refresh threshold (max of min or half-life)."""
    threshold, _source, _degraded = _refresh_threshold_metadata(
        expires_in=expires_in,
        access_token=access_token,
        expires_at=expires_at,
        issued_at=issued_at,
        obtained_at=obtained_at,
        refreshed_at=refreshed_at,
        min_seconds=min_seconds,
    )
    return threshold


def _refresh_threshold_metadata(
    *,
    expires_in: Any = None,
    access_token: Optional[str] = None,
    expires_at: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
    min_seconds: float = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS,
) -> Tuple[float, str, bool]:
    lifetime, source = _issued_lifetime_metadata(
        expires_in=expires_in,
        access_token=access_token,
        expires_at=expires_at,
        issued_at=issued_at,
        obtained_at=obtained_at,
        refreshed_at=refreshed_at,
    )
    if lifetime is None:
        return float(min_seconds), "fallback", True
    return max(float(min_seconds), lifetime * 0.5), source, False


@dataclass(frozen=True)
class XaiOAuthRefreshSummary:
    attempted: bool
    refreshed: bool
    skipped: bool
    auth_file: str
    scope: str
    expires_at: Optional[str] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    auth_degraded: bool = False
    refresh_threshold_seconds: Optional[float] = None
    refresh_threshold_source: Optional[str] = None
    refresh_threshold_degraded: bool = False
    credential_identity: Optional[str] = None
    auth_file_source: Optional[str] = None
    scope_source: Optional[str] = None
    structurally_valid: Optional[bool] = None
    access_available: Optional[bool] = None
    refresh_possible: Optional[bool] = None
    route_usable: Optional[bool] = None
    route_unusable: Optional[bool] = None
    refresh_due: Optional[bool] = None
    expiry_available: Optional[bool] = None
    terminal_unrefreshable: Optional[bool] = None
    lifecycle_state: Optional[str] = None
    route_unusable_reason: Optional[str] = None
    refresh_due_at: Optional[str] = None
    route_unusable_at: Optional[str] = None
    route_safety_buffer_seconds: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "refreshed": self.refreshed,
            "skipped": self.skipped,
            "auth_file": self.auth_file,
            "scope": self.scope,
            "expires_at": self.expires_at,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "auth_degraded": self.auth_degraded,
            "refresh_threshold_seconds": self.refresh_threshold_seconds,
            "refresh_threshold_source": self.refresh_threshold_source,
            "refresh_threshold_degraded": self.refresh_threshold_degraded,
            "credential_identity": self.credential_identity,
            "auth_file_source": self.auth_file_source,
            "scope_source": self.scope_source,
            "structurally_valid": self.structurally_valid,
            "access_available": self.access_available,
            "refresh_possible": self.refresh_possible,
            "route_usable": self.route_usable,
            "route_unusable": self.route_unusable,
            "refresh_due": self.refresh_due,
            "expiry_available": self.expiry_available,
            "terminal_unrefreshable": self.terminal_unrefreshable,
            "lifecycle_state": self.lifecycle_state,
            "route_unusable_reason": self.route_unusable_reason,
            "refresh_due_at": self.refresh_due_at,
            "route_unusable_at": self.route_unusable_at,
            "route_safety_buffer_seconds": self.route_safety_buffer_seconds,
        }


def inspect_xai_oauth_credential_health(
    auth_file: str | Path,
    *,
    scope: Optional[str] = None,
    buffer_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Read and classify xAI OAuth state without locks, writes, or HTTP."""
    resolved_auth_file = Path(auth_file).expanduser()
    resolved_scope = _resolve_scope(scope)
    refresh_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)
    route_safety_buffer_seconds = _resolve_route_safety_buffer_seconds()
    try:
        resolution = resolve_xai_oauth_credentials(
            auth_file,
            scope,
            value_getter=os.getenv,
        )
        resolved_auth_file = resolution.auth_file
        resolved_scope = resolution.scope
        credential = _select_credential_record(
            _read_credential_payload(resolution.canonical_auth_file),
            resolved_scope,
        )
        lifecycle = evaluate_xai_oauth_credential_lifecycle(
            credential,
            route_safety_buffer_seconds=route_safety_buffer_seconds,
            refresh_min_seconds=refresh_buffer_seconds,
        )
        error_class: Optional[str] = None
        error_message: Optional[str] = None
        if not lifecycle["structurally_valid"]:
            error_class = "CredentialStructureInvalid"
            error_message = "xAI OAuth credential record is malformed."
        elif not lifecycle["access_available"]:
            error_class = "CredentialAccessUnavailable"
            error_message = (
                "xAI OAuth credential does not contain an access credential."
            )
        elif not lifecycle["expiry_available"]:
            error_class = "CredentialExpiryUnavailable"
            error_message = (
                "xAI OAuth credential expires_at is missing or invalid."
            )
        elif lifecycle["expired"]:
            error_class = "CredentialExpiredError"
            error_message = "xAI OAuth credential is expired."
        return _xai_health_summary(
            resolved_auth_file,
            resolved_scope,
            str(lifecycle["credential_health"]),
            lifecycle.get("expires_at"),
            error_class=error_class,
            error_message=error_message,
            credential_identity=resolution.credential_identity,
            auth_file_source=resolution.auth_file_source,
            scope_source=resolution.scope_source,
            lifecycle=lifecycle,
            route_safety_buffer_seconds=route_safety_buffer_seconds,
        )
    except Exception as exc:
        return _xai_health_summary(
            resolved_auth_file,
            resolved_scope,
            "malformed",
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
            route_safety_buffer_seconds=route_safety_buffer_seconds,
        )


def _xai_health_summary(
    auth_file: Path,
    scope: str,
    health_status: str,
    expires_at: Optional[datetime] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
    credential_identity: Optional[str] = None,
    auth_file_source: Optional[str] = None,
    scope_source: Optional[str] = None,
    lifecycle: Optional[Mapping[str, Any]] = None,
    route_safety_buffer_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    lifecycle = lifecycle or {}
    return {
        "attempted": True,
        "refreshed": False,
        "skipped": False,
        "auth_file": str(auth_file),
        "scope": scope,
        "health_status": health_status,
        "expires_at": _format_expires_at(expires_at),
        "error_class": error_class,
        "error_message": error_message,
        "credential_identity": credential_identity,
        "auth_file_source": auth_file_source,
        "scope_source": scope_source,
        "structurally_valid": lifecycle.get("structurally_valid"),
        "access_available": lifecycle.get("access_available"),
        "refresh_possible": lifecycle.get("refresh_possible"),
        "route_usable": lifecycle.get("route_usable"),
        "route_unusable": lifecycle.get("route_unusable"),
        "refresh_due": lifecycle.get("refresh_due"),
        "expiry_available": lifecycle.get("expiry_available"),
        "terminal_unrefreshable": lifecycle.get("terminal_unrefreshable"),
        "lifecycle_state": lifecycle.get("state"),
        "route_unusable_reason": lifecycle.get("route_unusable_reason"),
        "refresh_due_at": _format_expires_at(lifecycle.get("refresh_due_at")),
        "route_unusable_at": _format_expires_at(
            lifecycle.get("route_unusable_at")
        ),
        "refresh_threshold_seconds": lifecycle.get("refresh_threshold_seconds"),
        "refresh_threshold_source": lifecycle.get("refresh_threshold_source"),
        "refresh_threshold_degraded": lifecycle.get("refresh_threshold_degraded"),
        "route_safety_buffer_seconds": route_safety_buffer_seconds,
        "usable": lifecycle.get("route_usable"),
    }


def _lifecycle_summary_fields(
    lifecycle: Mapping[str, Any],
    *,
    route_safety_buffer_seconds: Optional[float],
) -> Dict[str, Any]:
    return {
        "structurally_valid": lifecycle.get("structurally_valid"),
        "access_available": lifecycle.get("access_available"),
        "refresh_possible": lifecycle.get("refresh_possible"),
        "route_usable": lifecycle.get("route_usable"),
        "route_unusable": lifecycle.get("route_unusable"),
        "refresh_due": lifecycle.get("refresh_due"),
        "expiry_available": lifecycle.get("expiry_available"),
        "terminal_unrefreshable": lifecycle.get("terminal_unrefreshable"),
        "lifecycle_state": lifecycle.get("state"),
        "route_unusable_reason": lifecycle.get("route_unusable_reason"),
        "refresh_due_at": _format_expires_at(lifecycle.get("refresh_due_at")),
        "route_unusable_at": _format_expires_at(
            lifecycle.get("route_unusable_at")
        ),
        "route_safety_buffer_seconds": route_safety_buffer_seconds,
    }


def _write_private_file_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Thin wrapper over shared private write (no umask window, symlink-safe)."""
    write_private_file_text(
        path,
        content,
        mode=mode,
        default_mode=DEFAULT_XAI_OAUTH_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def refresh_xai_oauth_auth_file(
    auth_file: str | Path,
    *,
    scope: Optional[str] = None,
    buffer_seconds: Optional[int] = None,
    force: bool = False,
    lock_file: str | Path | None = None,
    token_endpoint: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    http_timeout_seconds: float = DEFAULT_XAI_OAUTH_HTTP_TIMEOUT_SECONDS,
    on_token_endpoint_attempt: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Refresh a managed xAI OAuth auth file when near expiry or forced."""

    resolved_auth_file = Path(auth_file).expanduser()
    resolved_scope = _resolve_scope(scope)
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)
    route_safety_buffer_seconds = _resolve_route_safety_buffer_seconds()
    resolution = None
    credential: Optional[MutableMapping[str, Any]] = None
    lifecycle: Optional[Mapping[str, Any]] = None

    try:
        resolution = resolve_xai_oauth_credentials(
            auth_file,
            scope,
            value_getter=os.getenv,
        )
        resolved_auth_file = resolution.auth_file
        resolved_scope = resolution.scope
        resolved_read_auth_file = resolution.canonical_auth_file
        resolved_lock_file = resolve_xai_oauth_lock_path(
            resolution.canonical_auth_file,
            lock_file,
        )
        with _credential_file_lock(resolved_lock_file):
            raw_payload = _read_credential_payload(resolved_read_auth_file)
            credential = _select_credential_record(raw_payload, resolved_scope)
            lifecycle = evaluate_xai_oauth_credential_lifecycle(
                credential,
                route_safety_buffer_seconds=route_safety_buffer_seconds,
                refresh_min_seconds=resolved_buffer_seconds,
            )
            threshold, threshold_source, threshold_degraded = _credential_refresh_threshold_metadata(
                credential,
                min_seconds=resolved_buffer_seconds,
            )
            current_expires_at = _format_expires_at(
                lifecycle.get("expires_at")
            )

            if not force and not lifecycle["refresh_due"]:
                return XaiOAuthRefreshSummary(
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    auth_file=str(resolved_auth_file),
                    scope=resolved_scope,
                    expires_at=current_expires_at,
                    auth_degraded=threshold_degraded,
                    refresh_threshold_seconds=threshold,
                    refresh_threshold_source=threshold_source,
                    refresh_threshold_degraded=threshold_degraded,
                    credential_identity=resolution.credential_identity,
                    auth_file_source=resolution.auth_file_source,
                    scope_source=resolution.scope_source,
                    **_lifecycle_summary_fields(
                        lifecycle,
                        route_safety_buffer_seconds=route_safety_buffer_seconds,
                    ),
                ).as_dict()

            refreshed = _refresh_credential_record(
                credential,
                token_endpoint=token_endpoint,
                client_id=client_id,
                client_secret=client_secret,
                http_timeout_seconds=http_timeout_seconds,
                on_token_endpoint_attempt=on_token_endpoint_attempt,
            )
            _update_credential_record(credential, refreshed)
            _write_credential_payload(resolved_auth_file, raw_payload)
            lifecycle = evaluate_xai_oauth_credential_lifecycle(
                credential,
                route_safety_buffer_seconds=route_safety_buffer_seconds,
                refresh_min_seconds=resolved_buffer_seconds,
            )
            threshold, threshold_source, threshold_degraded = (
                _credential_refresh_threshold_metadata(
                    credential,
                    min_seconds=resolved_buffer_seconds,
                )
            )
            return XaiOAuthRefreshSummary(
                attempted=True,
                refreshed=True,
                skipped=False,
                auth_file=str(resolved_auth_file),
                scope=resolved_scope,
                expires_at=_format_expires_at(lifecycle.get("expires_at")),
                auth_degraded=threshold_degraded,
                refresh_threshold_seconds=threshold,
                refresh_threshold_source=threshold_source,
                refresh_threshold_degraded=threshold_degraded,
                credential_identity=resolution.credential_identity,
                auth_file_source=resolution.auth_file_source,
                scope_source=resolution.scope_source,
                **_lifecycle_summary_fields(
                    lifecycle,
                    route_safety_buffer_seconds=route_safety_buffer_seconds,
                ),
            ).as_dict()
    except Exception as exc:
        threshold: Optional[float] = None
        threshold_source: Optional[str] = None
        threshold_degraded = False
        if credential is not None:
            try:
                threshold, threshold_source, threshold_degraded = _credential_refresh_threshold_metadata(
                    credential,
                    min_seconds=resolved_buffer_seconds,
                )
            except Exception:
                pass
        error_class, error_message = _refresh_error_summary(exc)
        return XaiOAuthRefreshSummary(
            attempted=True,
            refreshed=False,
            skipped=False,
            auth_file=str(resolved_auth_file),
            scope=resolved_scope,
            error_class=error_class,
            error_message=error_message,
            auth_degraded=threshold_degraded,
            refresh_threshold_seconds=threshold,
            refresh_threshold_source=threshold_source,
            refresh_threshold_degraded=threshold_degraded,
            credential_identity=(
                resolution.credential_identity if resolution is not None else None
            ),
            auth_file_source=(
                resolution.auth_file_source if resolution is not None else None
            ),
            scope_source=resolution.scope_source if resolution is not None else None,
            **(
                _lifecycle_summary_fields(
                    lifecycle,
                    route_safety_buffer_seconds=route_safety_buffer_seconds,
                )
                if lifecycle is not None
                else {}
            ),
        ).as_dict()


def inspect_xai_oauth_refresh_eligibility(
    auth_file: str | Path,
    *,
    buffer_seconds: int = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS,
    now: Optional[Callable[[], datetime]] = None,
    poll_interval_seconds: float = 300.0,
    scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect managed xAI OAuth refresh eligibility without side effects."""
    observed_at = _resolve_wall_now(now)
    resolved_scope = _resolve_scope(scope)
    resolved_buffer_seconds = max(0, int(buffer_seconds))
    route_safety_buffer_seconds = _resolve_route_safety_buffer_seconds()
    resolution = None
    credential: Optional[MutableMapping[str, Any]] = None
    try:
        resolution = resolve_xai_oauth_credentials(
            auth_file,
            scope,
            value_getter=os.getenv,
        )
        resolved_scope = resolution.scope
        payload = _read_credential_payload(resolution.canonical_auth_file)
        credential = _select_credential_record(payload, resolved_scope)
        lifecycle = evaluate_xai_oauth_credential_lifecycle(
            credential,
            now=lambda: observed_at,
            route_safety_buffer_seconds=route_safety_buffer_seconds,
            refresh_min_seconds=resolved_buffer_seconds,
        )
        expires_at = lifecycle.get("expires_at")
        refresh_due_at = lifecycle.get("refresh_due_at")
        next_refresh_check_at = (
            refresh_due_at
            if isinstance(refresh_due_at, datetime)
            and observed_at < refresh_due_at
            else observed_at
            + timedelta(seconds=max(1.0, poll_interval_seconds))
        )
        error_class: Optional[str] = None
        error_message: Optional[str] = None
        if not lifecycle["structurally_valid"]:
            error_class = "CredentialStructureInvalid"
            error_message = "xAI OAuth credential record is malformed."
        elif not lifecycle["access_available"]:
            error_class = "CredentialAccessUnavailable"
            error_message = (
                "xAI OAuth credential does not contain an access credential."
            )
        elif not lifecycle["expiry_available"]:
            error_class = "CredentialExpiryUnavailable"
            error_message = (
                "xAI OAuth credential expires_at is missing or invalid."
            )
        return _eligibility_summary(
            observed_at=observed_at,
            expires_at=expires_at,
            refresh_due_at=refresh_due_at,
            next_refresh_check_at=next_refresh_check_at,
            eligible=bool(lifecycle["refresh_due"]),
            credential_health=str(lifecycle["credential_health"]),
            usable=bool(lifecycle["route_usable"]),
            error_class=error_class,
            error_message=error_message,
            credential_identity=resolution.credential_identity,
            auth_file_source=resolution.auth_file_source,
            scope_source=resolution.scope_source,
            lifecycle=lifecycle,
            route_safety_buffer_seconds=route_safety_buffer_seconds,
        )
    except Exception as exc:
        return _eligibility_summary(
            observed_at=observed_at,
            expires_at=None,
            refresh_due_at=None,
            next_refresh_check_at=observed_at
            + timedelta(seconds=max(1.0, poll_interval_seconds)),
            eligible=True,
            credential_health="malformed",
            usable=False,
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
            credential_identity=(
                resolution.credential_identity if resolution is not None else None
            ),
            auth_file_source=(
                resolution.auth_file_source if resolution is not None else None
            ),
            scope_source=resolution.scope_source if resolution is not None else None,
            refresh_threshold_seconds=float(resolved_buffer_seconds),
            refresh_threshold_source="fallback",
            refresh_threshold_degraded=True,
            route_safety_buffer_seconds=route_safety_buffer_seconds,
        )


def _eligibility_summary(
    *,
    observed_at: datetime,
    expires_at: Optional[datetime],
    refresh_due_at: Optional[datetime],
    next_refresh_check_at: datetime,
    eligible: bool,
    credential_health: str,
    usable: bool,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
    credential_identity: Optional[str] = None,
    auth_file_source: Optional[str] = None,
    scope_source: Optional[str] = None,
    lifecycle: Optional[Mapping[str, Any]] = None,
    route_safety_buffer_seconds: Optional[float] = None,
    refresh_threshold_seconds: Optional[float] = None,
    refresh_threshold_source: Optional[str] = None,
    refresh_threshold_degraded: bool = False,
) -> Dict[str, Any]:
    lifecycle = lifecycle or {}
    return {
        "eligibility_checked_at": _format_expires_at(observed_at),
        "expires_at": _format_expires_at(expires_at),
        "refresh_due_at": _format_expires_at(refresh_due_at),
        "next_refresh_check_at": _format_expires_at(next_refresh_check_at),
        "eligible": eligible,
        "credential_health": credential_health,
        "usable": usable,
        "error_class": error_class,
        "error_message": error_message,
        "credential_identity": credential_identity,
        "auth_file_source": auth_file_source,
        "scope_source": scope_source,
        "structurally_valid": lifecycle.get("structurally_valid"),
        "access_available": lifecycle.get("access_available"),
        "refresh_possible": lifecycle.get("refresh_possible"),
        "route_usable": lifecycle.get("route_usable"),
        "route_unusable": lifecycle.get("route_unusable"),
        "refresh_due": lifecycle.get("refresh_due"),
        "expiry_available": lifecycle.get("expiry_available"),
        "terminal_unrefreshable": lifecycle.get("terminal_unrefreshable"),
        "lifecycle_state": lifecycle.get("state"),
        "route_unusable_reason": lifecycle.get("route_unusable_reason"),
        "route_unusable_at": _format_expires_at(
            lifecycle.get("route_unusable_at")
        ),
        "refresh_threshold_seconds": (
            lifecycle.get("refresh_threshold_seconds")
            if lifecycle
            else refresh_threshold_seconds
        ),
        "refresh_threshold_source": (
            lifecycle.get("refresh_threshold_source")
            if lifecycle
            else refresh_threshold_source
        ),
        "refresh_threshold_degraded": (
            lifecycle.get("refresh_threshold_degraded")
            if lifecycle
            else refresh_threshold_degraded
        ),
        "route_safety_buffer_seconds": route_safety_buffer_seconds,
    }


def _resolve_wall_now(now: Optional[Callable[[], datetime]]) -> datetime:
    value = now() if now is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_scope(scope: Optional[str]) -> str:
    if isinstance(scope, str) and scope.strip():
        return scope.strip()
    env_scope = os.getenv("AAWM_XAI_OAUTH_SCOPE") or os.getenv(
        "LITELLM_XAI_OAUTH_SCOPE"
    )
    if isinstance(env_scope, str) and env_scope.strip():
        return env_scope.strip()
    return DEFAULT_XAI_OAUTH_SCOPE


def _resolve_buffer_seconds(buffer_seconds: Optional[int]) -> int:
    if buffer_seconds is not None:
        return max(0, int(buffer_seconds))
    raw_value = os.getenv("AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS") or os.getenv(
        "LITELLM_XAI_OAUTH_REFRESH_BUFFER_SECONDS"
    )
    if raw_value is None or not raw_value.strip():
        return DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS


def _resolve_route_safety_buffer_seconds() -> int:
    raw_value = os.getenv("LITELLM_XAI_OAUTH_REFRESH_BUFFER_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS


@contextmanager
def _credential_file_lock(lock_path: Path) -> Iterator[None]:
    """Delegate to shared nonblocking, fail-closed credential_file_lock."""
    with credential_file_lock(lock_path):
        yield


def _snapshot_credential_file_metadata(
    auth_path: Path,
) -> CredentialFileMetadata:
    return snapshot_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_XAI_OAUTH_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _resolve_credential_file_metadata(auth_path: Path) -> CredentialFileMetadata:
    """Resolve ownership/mode for ``auth_path`` via shared helpers.

    Snapshot goes through ``_snapshot_credential_file_metadata`` so tests and
    monkeypatches of the thin local wrapper remain effective. Symlink targets
    are refused during snapshot/resolve.
    """
    return resolve_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_XAI_OAUTH_AUTH_FILE_MODE,
        mode_env="AAWM_XAI_OAUTH_AUTH_FILE_MODE",
        uid_env="AAWM_XAI_OAUTH_AUTH_FILE_UID",
        gid_env="AAWM_XAI_OAUTH_AUTH_FILE_GID",
        base_metadata=_snapshot_credential_file_metadata(auth_path),
        refuse_symlink=True,
    )


def _apply_credential_file_metadata(
    target_path: Path,
    metadata: CredentialFileMetadata,
) -> None:
    apply_credential_file_metadata(
        target_path,
        metadata,
        default_mode=DEFAULT_XAI_OAUTH_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _read_credential_payload(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"xAI OAuth auth file not found at {auth_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"xAI OAuth auth file at {auth_path} is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("xAI OAuth auth file must contain a JSON object.")
    return payload


def _select_credential_record(
    payload: MutableMapping[str, Any],
    scope: str,
) -> MutableMapping[str, Any]:
    return select_xai_oauth_credential_record(
        payload,
        scope,
        provider_label="xAI OAuth",
    )


def _looks_like_credential_record(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("key") or value.get("access_token") or value.get("refresh_token")
    )


def _credential_needs_refresh(
    credential: Mapping[str, Any],
    *,
    buffer_seconds: int,
) -> bool:
    """Return True when the credential should be refreshed.

    Uses the proportional half-life threshold derived from the credential's
    own ``expires_in`` and ``access_token``, falling back to the passed
    ``buffer_seconds`` when no lifetime metadata is available.
    """
    lifecycle = evaluate_xai_oauth_credential_lifecycle(
        credential,
        route_safety_buffer_seconds=_resolve_route_safety_buffer_seconds(),
        refresh_min_seconds=max(0, int(buffer_seconds)),
    )
    return bool(lifecycle["refresh_due"])


def _parse_expires_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _datetime_from_epoch_numeric(float(value))
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        try:
            return _datetime_from_epoch_numeric(float(normalized))
        except ValueError:
            pass
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _datetime_from_epoch_numeric(raw_value: float) -> datetime:
    if raw_value >= 1_000_000_000_000:
        raw_value = raw_value / 1000.0
    return datetime.fromtimestamp(raw_value, tz=timezone.utc)


def _format_expires_at(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _credential_refresh_threshold_metadata(
    credential: Mapping[str, Any],
    *,
    min_seconds: float = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS,
) -> Tuple[float, str, bool]:
    return _foundation_refresh_threshold_metadata(
        credential,
        min_seconds=min_seconds,
    )


def _credential_expires_at(credential: Mapping[str, Any]) -> Optional[datetime]:
    explicit_expiry = _parse_expires_at(credential.get("expires_at"))
    if explicit_expiry is not None:
        return explicit_expiry

    jwt_claims = _jwt_time_claims(
        credential.get("access_token") or credential.get("key")
    )
    if jwt_claims is not None:
        try:
            return datetime.fromtimestamp(jwt_claims[1], tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    issued_at = _credential_issued_at(credential)
    lifetime = _as_finite_number(credential.get("expires_in"))
    if issued_at is not None and lifetime is not None and lifetime > 0:
        try:
            return datetime.fromtimestamp(
                issued_at + lifetime,
                tz=timezone.utc,
            )
        except (OSError, OverflowError, ValueError):
            return None
    return None


def _credential_issued_at(credential: Mapping[str, Any]) -> Optional[float]:
    return _first_timestamp_seconds(
        credential.get("issued_at"),
        credential.get("obtained_at"),
        credential.get("refreshed_at"),
    )


def _jwt_time_claims(access_token: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("ascii"))
        )
        if not isinstance(claims, dict):
            return None
        issued_at = _as_finite_number(claims.get("iat"))
        expires_at = _as_finite_number(claims.get("exp"))
        if (
            issued_at is None
            or expires_at is None
            or expires_at <= issued_at
        ):
            return None
        return issued_at, expires_at
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _first_timestamp_seconds(*values: Any) -> Optional[float]:
    for value in values:
        timestamp = _timestamp_seconds(value)
        if timestamp is not None:
            return timestamp
    return None


def _timestamp_seconds(value: Any) -> Optional[float]:
    numeric = _as_finite_number(value)
    if numeric is not None:
        if numeric >= 1_000_000_000_000:
            numeric /= 1000.0
        return numeric
    if isinstance(value, datetime):
        timestamp = value.timestamp()
        return timestamp if math.isfinite(timestamp) else None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    try:
        numeric = float(normalized)
    except ValueError:
        pass
    else:
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


def _refresh_credential_record(
    credential: Mapping[str, Any],
    *,
    token_endpoint: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    http_timeout_seconds: float,
    on_token_endpoint_attempt: Optional[Callable[[], None]] = None,
) -> Mapping[str, Any]:
    refresh_token = _clean_oauth_string(credential.get("refresh_token"))
    if refresh_token is None:
        raise XaiOAuthRefreshError(
            "credential_unavailable",
            "xAI OAuth credential cannot be refreshed without a refresh token.",
        )

    resolved_client_id = (
        _clean_oauth_string(client_id)
        or _clean_oauth_string(credential.get("oidc_client_id"))
        or _clean_oauth_string(credential.get("client_id"))
    )
    if resolved_client_id is None:
        raise XaiOAuthRefreshError(
            "credential_configuration_error",
            "xAI OAuth refresh requires a configured client identifier.",
        )

    resolved_token_endpoint = (
        _clean_oauth_string(token_endpoint)
        or _clean_oauth_string(credential.get("token_endpoint"))
        or DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT
    )
    form_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": resolved_client_id,
    }
    resolved_client_secret = _clean_oauth_string(client_secret) or _clean_oauth_string(
        credential.get("client_secret")
    )
    if resolved_client_secret is not None:
        form_data["client_secret"] = resolved_client_secret

    body = urllib_parse.urlencode(form_data).encode("utf-8")
    request = urllib_request.Request(
        resolved_token_endpoint,
        data=body,
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        },
        method="POST",
    )
    try:
        if on_token_endpoint_attempt is not None:
            on_token_endpoint_attempt()
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            response_status = int(getattr(response, "status", 200))
            response_body = response.read()
    except urllib_error.HTTPError as exc:
        try:
            error_body = exc.read()
        except OSError:
            error_body = b""
        raise _xai_oauth_http_error(exc.code, error_body) from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise XaiOAuthRefreshError(
            "transport_error",
            "xAI OAuth token endpoint transport failed.",
        ) from exc

    if response_status < 200 or response_status >= 300:
        raise _xai_oauth_http_error(response_status, response_body)

    try:
        payload = json.loads(response_body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise XaiOAuthRefreshError(
            "malformed_response",
            "xAI OAuth token endpoint returned an invalid response.",
        ) from exc
    if not isinstance(payload, Mapping):
        raise XaiOAuthRefreshError(
            "malformed_response",
            "xAI OAuth token endpoint returned an invalid response.",
        )
    if _clean_oauth_string(payload.get("access_token")) is None:
        raise XaiOAuthRefreshError(
            "malformed_response",
            "xAI OAuth token endpoint returned an incomplete response.",
        )
    return payload


def _xai_oauth_http_error(
    status_code: int,
    response_body: bytes,
) -> XaiOAuthRefreshError:
    """Classify an HTTP failure without retaining provider response content."""
    if status_code == 400:
        terminal_error = _xai_oauth_terminal_error_code(response_body)
        if terminal_error is not None:
            return XaiOAuthRefreshError(
                terminal_error,
                "xAI OAuth token endpoint rejected the refresh grant.",
            )
    error_class = (
        "retryable_http_error"
        if status_code in _XAI_OAUTH_RETRYABLE_HTTP_STATUS_CODES
        or status_code >= 500
        else "http_error"
    )
    return XaiOAuthRefreshError(
        error_class,
        f"xAI OAuth token endpoint returned HTTP {status_code}.",
    )


def _xai_oauth_terminal_error_code(response_body: bytes) -> Optional[str]:
    """Return only exact allowlisted terminal codes from a 400 JSON response."""
    try:
        payload = json.loads(response_body)
    except (TypeError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    error_code = payload.get("error")
    if (
        isinstance(error_code, str)
        and error_code in _XAI_OAUTH_TERMINAL_REFRESH_ERROR_CLASSES
    ):
        return error_code
    return None


def _refresh_error_summary(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, XaiOAuthRefreshError):
        return exc.error_class, _sanitize_error_message(str(exc))
    return (
        "local_refresh_error",
        "xAI OAuth refresh could not complete.",
    )


def _update_credential_record(
    credential: MutableMapping[str, Any],
    refreshed: Mapping[str, Any],
    *,
    now: Optional[Callable[[], datetime]] = None,
) -> None:
    access_token = _clean_oauth_string(refreshed.get("access_token"))
    if access_token is not None:
        credential["key"] = access_token
        credential["access_token"] = access_token

    refresh_token = _clean_oauth_string(refreshed.get("refresh_token"))
    if refresh_token is not None:
        credential["refresh_token"] = refresh_token

    id_token = _clean_oauth_string(refreshed.get("id_token"))
    if id_token is not None:
        credential["id_token"] = id_token

    observed_at = _resolve_wall_now(now)
    credential["obtained_at"] = _format_expires_at(observed_at)
    credential.pop("issued_at", None)
    credential.pop("refreshed_at", None)

    expires_in = _as_finite_number(refreshed.get("expires_in"))
    effective_access_token = access_token
    if effective_access_token is None:
        effective_access_token = credential.get("access_token") or credential.get("key")
    jwt_claims = _jwt_time_claims(effective_access_token)
    if expires_in is not None and expires_in > 0:
        credential["expires_in"] = _json_number(expires_in)
        expires_at = observed_at + timedelta(seconds=expires_in)
        credential["expires_at"] = _format_expires_at(expires_at)
    elif jwt_claims is not None:
        issued_at, expires_at_timestamp = jwt_claims
        credential["issued_at"] = _json_number(issued_at)
        credential["expires_in"] = None
        credential["expires_at"] = _format_expires_at(
            datetime.fromtimestamp(expires_at_timestamp, tz=timezone.utc)
        )
    else:
        credential["expires_in"] = None
        credential["expires_at"] = None

    token_type = _clean_oauth_string(refreshed.get("token_type"))
    if token_type is not None:
        credential["token_type"] = token_type


def _write_credential_payload(auth_path: Path, payload: Mapping[str, Any]) -> None:
    """Publish credential JSON via shared exclusive temp + atomic replace.

    Uses ``write_and_publish_private_text`` so temp names are not pid-only,
    symlink targets are refused, and failed temps are cleaned up consistently.
    """
    try:
        # Shared one-shot path: exclusive private temp, symlink refusal, metadata
        # apply on temp, atomic publish, and failed-temp cleanup. Symlink targets
        # are refused both when resolving metadata and when publishing.
        metadata = _resolve_credential_file_metadata(auth_path)
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        write_and_publish_private_text(
            auth_path,
            content,
            metadata=metadata,
            default_mode=DEFAULT_XAI_OAUTH_AUTH_FILE_MODE,
            mkdir_parents=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to persist refreshed xAI OAuth auth data: {exc}"
        ) from exc


def _clean_oauth_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _sanitize_error_message(
    message: str, *, limit: int = DEFAULT_XAI_OAUTH_ERROR_MESSAGE_LIMIT
) -> str:
    """Redact secret *values* keyed by known field names (not just the labels)."""
    return sanitize_credential_error_message(message, limit=limit)

# Backward-compatible alias for tests and callers that still reference
# the old buffer-seconds constant.
DEFAULT_XAI_OAUTH_REFRESH_BUFFER_SECONDS = DEFAULT_XAI_OAUTH_REFRESH_MIN_SECONDS
