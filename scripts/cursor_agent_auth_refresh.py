#!/usr/bin/env python3
"""Refresh the Cursor Agent auth file owned by the provider-status sidecar."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)
from urllib import error as urllib_error
from urllib import request as urllib_request

from litellm.llms.cursor_agent.constants import (
    CURSOR_AGENT_AUTH_EXCHANGE_PATH,
    CURSOR_AGENT_DASHBOARD_HOST,
)
from litellm.llms.cursor_agent.dashboard import cursor_agent_user_agent
from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)
from litellm.secret_managers.credential_file_lock import credential_file_lock
from litellm.secret_managers.credential_file_metadata import (
    CredentialFileMetadata,
    apply_credential_file_metadata,
    ensure_not_symlink_path,
    is_symlink_path,
    resolve_credential_file_metadata,
    snapshot_credential_file_metadata,
)
from litellm.secret_managers.credential_file_write import (
    write_and_publish_private_text,
)

DEFAULT_CURSOR_AGENT_AUTH_FILE = "/home/zepfu/.config/cursor/auth.json"
DEFAULT_CURSOR_AGENT_AUTH_LOCK_FILE = (
    "/home/zepfu/.config/cursor/auth.json.lock"
)
DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS = 300.0
DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS = 300
DEFAULT_CURSOR_AGENT_AUTH_FORCE_REFRESH = False
DEFAULT_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_CURSOR_AGENT_AUTH_FILE_MODE = 0o600
DEFAULT_CURSOR_AGENT_AUTH_ERROR_MESSAGE_LIMIT = 500
DEFAULT_CURSOR_AGENT_AUTH_EXCHANGE_URL = (
    f"{CURSOR_AGENT_DASHBOARD_HOST}{CURSOR_AGENT_AUTH_EXCHANGE_PATH}"
)

_SECRET_FIELD_NAMES = frozenset(
    set(DEFAULT_SECRET_FIELD_NAMES)
    | {
        "accessToken",
        "refreshToken",
        "apiKey",
        "access_token",
        "refresh_token",
        "api_key",
    }
)
_ACCESS_TOKEN_KEYS = ("accessToken", "access_token")
_REFRESH_TOKEN_KEYS = ("refreshToken", "refresh_token")
_API_KEY_KEYS = ("apiKey", "api_key")
_EXPIRY_KEYS = (
    "expiresAt",
    "expires_at",
    "expiresAtMs",
    "expires_at_ms",
)
_RELATIVE_EXPIRY_KEYS = ("expiresIn", "expires_in")
_ALL_EXPIRY_KEYS = (*_EXPIRY_KEYS, *_RELATIVE_EXPIRY_KEYS)


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
    """Derive an issued lifetime and identify its authoritative source.

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
    persisted_expires_at = _parse_timestamp(expires_at)
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
    min_seconds: float = DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS,
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
    min_seconds: float = DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS,
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


NowValue = float | int | datetime
NowProvider = Callable[[], NowValue]


class CursorAgentAuthError(RuntimeError):
    """Base error for a Cursor Agent auth-file operation."""


class CursorAgentCredentialError(CursorAgentAuthError):
    """The auth file does not contain a supported credential shape."""


class CursorAgentRefreshTokenOnlyError(CursorAgentCredentialError):
    """A due credential has no verified refresh-token-only flow."""


class CursorAgentAuthExchangeError(CursorAgentAuthError):
    """The verified Cursor API-key exchange failed."""


@dataclass(frozen=True)
class _CredentialState:
    access_token: Optional[str]
    refresh_token: Optional[str]
    api_key: Optional[str]
    expires_at: Optional[float]
    shape: str
    fingerprint: Optional[str]
    expires_in: Any = None
    issued_at: Any = None
    obtained_at: Any = None
    refreshed_at: Any = None


@dataclass
class _SingleflightCall:
    event: threading.Event
    result: Optional[Dict[str, Any]] = None


_SINGLEFLIGHT_GUARD = threading.Lock()
_SINGLEFLIGHT_CALLS: Dict[tuple[Any, ...], _SingleflightCall] = {}


def _credential_refresh_threshold_metadata(
    state: _CredentialState,
) -> Tuple[float, str, bool]:
    return _refresh_threshold_metadata(
        expires_in=state.expires_in,
        access_token=state.access_token,
        expires_at=state.expires_at,
        issued_at=state.issued_at,
        obtained_at=state.obtained_at,
        refreshed_at=state.refreshed_at,
    )


def inspect_cursor_agent_auth_credential_health(
    auth_file: str | Path | None = None,
    *,
    now: Optional[NowProvider | NowValue] = None,
) -> Dict[str, Any]:
    """Inspect Cursor auth state without acquiring locks, writing, or HTTP."""
    resolved_auth_file = _resolve_auth_file(auth_file)
    observed_at = _resolve_wall_now(now)
    try:
        payload = _read_auth_data(resolved_auth_file)
        state = _credential_state(payload)
        return _health_summary(
            resolved_auth_file,
            state,
            observed_at=observed_at,
        )
    except Exception as exc:
        return _health_summary(
            resolved_auth_file,
            None,
            observed_at=observed_at,
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
        )


def inspect_cursor_agent_auth_refresh_eligibility(
    auth_file: str | Path | None = None,
    *,
    buffer_seconds: Optional[int] = None,
    now: Optional[NowProvider | NowValue] = None,
    poll_interval_seconds: Optional[float] = None,
    force: Optional[bool] = None,
) -> Dict[str, Any]:
    """Inspect Cursor auth refresh eligibility without side effects."""
    resolved_auth_file = _resolve_auth_file(auth_file)
    observed_at = _resolve_wall_now(now)
    try:
        resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)
        resolved_force = _resolve_force_refresh(force)
        resolved_poll_interval = _resolve_refresh_interval_seconds(
            poll_interval_seconds
        )
        payload = _read_auth_data(resolved_auth_file)
        state = _credential_state(payload)
        access_state = _access_token_state(
            state,
            observed_at,
            buffer_seconds=resolved_buffer_seconds,
        )
        refresh_due_at = _refresh_due_at(
            state.expires_at,
            buffer_seconds=resolved_buffer_seconds,
            access_token=state.access_token,
            expires_in=state.expires_in,
            issued_at=state.issued_at,
            obtained_at=state.obtained_at,
            refreshed_at=state.refreshed_at,
        )
        if refresh_due_at is None:
            next_refresh_check_at = observed_at + timedelta(
                seconds=resolved_poll_interval
            )
        elif observed_at < refresh_due_at and not resolved_force:
            next_refresh_check_at = refresh_due_at
        else:
            next_refresh_check_at = observed_at + timedelta(
                seconds=resolved_poll_interval
            )
        eligible = resolved_force or _refresh_is_due(
            state,
            observed_at=observed_at,
            buffer_seconds=resolved_buffer_seconds,
        )
        error_class: Optional[str] = None
        error_message: Optional[str] = None
        if eligible and state.api_key is None:
            if state.access_token is not None and state.refresh_token is not None:
                error_class = CursorAgentRefreshTokenOnlyError.__name__
                error_message = _sanitize_error_message(
                    "Cursor Agent refresh requires apiKey; the verified "
                    "Cursor contract has no refreshToken-only grant."
                )
            elif state.access_token is not None:
                error_class = CursorAgentCredentialError.__name__
                error_message = _sanitize_error_message(
                    "Cursor Agent refresh requires apiKey when the access "
                    "token is due."
                )
            elif state.refresh_token is not None:
                error_class = CursorAgentRefreshTokenOnlyError.__name__
                error_message = _sanitize_error_message(
                    "Cursor Agent auth file contains refreshToken without "
                    "the verified apiKey exchange credential."
                )
            else:
                error_class = CursorAgentCredentialError.__name__
                error_message = _sanitize_error_message(
                    "Cursor Agent auth file has no accessToken or apiKey."
                )
        return _eligibility_summary(
            resolved_auth_file,
            state,
            observed_at=observed_at,
            refresh_due_at=refresh_due_at,
            next_refresh_check_at=next_refresh_check_at,
            eligible=eligible,
            access_state=access_state,
            error_class=error_class,
            error_message=error_message,
        )
    except Exception as exc:
        return _eligibility_summary(
            resolved_auth_file,
            None,
            observed_at=observed_at,
            refresh_due_at=None,
            next_refresh_check_at=observed_at
            + timedelta(
                seconds=_resolve_positive_float(
                    _resolve_refresh_interval_seconds(poll_interval_seconds),
                    default=DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS,
                )
            ),
            eligible=True,
            access_state="malformed",
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
        )


def refresh_cursor_agent_auth_file(
    auth_file: str | Path | None = None,
    *,
    buffer_seconds: Optional[int] = None,
    force: Optional[bool] = None,
    lock_file: str | Path | None = None,
    dashboard_base: Optional[str] = None,
    http_timeout_seconds: Optional[float] = None,
    now: Optional[NowProvider | NowValue] = None,
    on_exchange_attempt: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Refresh a Cursor Agent auth file through the verified API-key exchange."""
    resolved_auth_file = _resolve_auth_file(auth_file)
    resolved_lock_file = _resolve_lock_file(
        auth_file=auth_file,
        auth_path=resolved_auth_file,
        lock_file=lock_file,
    )
    try:
        resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)
        resolved_force = _resolve_force_refresh(force)
        resolved_timeout = _resolve_timeout_seconds(http_timeout_seconds)
        resolved_dashboard_base = _resolve_dashboard_base(dashboard_base)
    except Exception as exc:
        return _refresh_error_summary(
            resolved_auth_file,
            None,
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
        )

    key = (
        str(resolved_auth_file),
        str(resolved_lock_file),
        resolved_buffer_seconds,
        resolved_force,
        resolved_dashboard_base,
        resolved_timeout,
    )

    def operation() -> Dict[str, Any]:
        return _refresh_locked(
            resolved_auth_file,
            resolved_lock_file,
            buffer_seconds=resolved_buffer_seconds,
            force=resolved_force,
            dashboard_base=resolved_dashboard_base,
            http_timeout_seconds=resolved_timeout,
            now=now,
            on_exchange_attempt=on_exchange_attempt,
        )

    def failure(exc: Exception) -> Dict[str, Any]:
        return _refresh_error_summary(
            resolved_auth_file,
            None,
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
        )

    return _run_singleflight(key, operation, failure)


def _refresh_locked(
    auth_path: Path,
    lock_path: Path,
    *,
    buffer_seconds: int,
    force: bool,
    dashboard_base: str,
    http_timeout_seconds: float,
    now: Optional[NowProvider | NowValue],
    on_exchange_attempt: Optional[Callable[[], None]],
) -> Dict[str, Any]:
    state: Optional[_CredentialState] = None
    observed_at = _resolve_wall_now(now)
    try:
        ensure_not_symlink_path(lock_path, role="Cursor Agent auth lock path")
        with _credential_file_lock(lock_path):
            auth_data = _read_auth_data(auth_path)
            state = _credential_state(auth_data)
            _repair_credential_file_metadata(auth_path)

            if not force and not _refresh_is_due(
                state,
                observed_at=observed_at,
                buffer_seconds=buffer_seconds,
            ):
                return _refresh_summary(
                    auth_path,
                    state,
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    observed_at=observed_at,
                    refresh_due_at=_refresh_due_at(
                        state.expires_at,
                        buffer_seconds=buffer_seconds,
                        access_token=state.access_token,
                        expires_in=state.expires_in,
                        issued_at=state.issued_at,
                        obtained_at=state.obtained_at,
                        refreshed_at=state.refreshed_at,
                    ),
                    refresh_method="none",
                )

            if state.api_key is None:
                _raise_missing_exchange_credential(
                    state,
                    observed_at=observed_at,
                    buffer_seconds=buffer_seconds,
                    force=force,
                )

            exchanged = _exchange_api_key(
                state.api_key,
                dashboard_base=dashboard_base,
                http_timeout_seconds=http_timeout_seconds,
                on_exchange_attempt=on_exchange_attempt,
            )
            updated_auth_data = _merge_exchange_result(
                auth_data,
                exchanged,
                api_key=state.api_key,
                observed_at=observed_at,
            )
            _write_auth_data(auth_path, updated_auth_data)
            refreshed_state = _credential_state(updated_auth_data)
            return _refresh_summary(
                auth_path,
                refreshed_state,
                attempted=True,
                refreshed=True,
                skipped=False,
                observed_at=observed_at,
                refresh_due_at=_refresh_due_at(
                    refreshed_state.expires_at,
                    buffer_seconds=buffer_seconds,
                    access_token=refreshed_state.access_token,
                    expires_in=refreshed_state.expires_in,
                    issued_at=refreshed_state.issued_at,
                    obtained_at=refreshed_state.obtained_at,
                    refreshed_at=refreshed_state.refreshed_at,
                ),
                refresh_method="apiKey_exchange",
                previous_credential_fingerprint=state.fingerprint,
            )
    except Exception as exc:
        return _refresh_error_summary(
            auth_path,
            state,
            observed_at=observed_at,
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
            refresh_due_at=(
                _refresh_due_at(
                    state.expires_at,
                    buffer_seconds=buffer_seconds,
                    access_token=state.access_token,
                    expires_in=state.expires_in,
                    issued_at=state.issued_at,
                    obtained_at=state.obtained_at,
                    refreshed_at=state.refreshed_at,
                )
                if state is not None
                else None
            ),
        )


def _raise_missing_exchange_credential(
    state: _CredentialState,
    *,
    observed_at: datetime,
    buffer_seconds: int,
    force: bool,
) -> None:
    if (
        state.access_token is not None
        and state.refresh_token is not None
        and (
            force
            or _refresh_is_due(
                state,
                observed_at=observed_at,
                buffer_seconds=buffer_seconds,
            )
        )
    ):
        raise CursorAgentRefreshTokenOnlyError(
            "Cursor Agent refresh requires apiKey; the verified Cursor "
            "contract has no refreshToken-only grant."
        )
    if state.access_token is None and state.refresh_token is not None:
        raise CursorAgentRefreshTokenOnlyError(
            "Cursor Agent auth file contains refreshToken without the "
            "verified apiKey exchange credential."
        )
    raise CursorAgentCredentialError(
        "Cursor Agent refresh requires apiKey when the access token is due."
    )


def _exchange_api_key(
    api_key: str,
    *,
    dashboard_base: str,
    http_timeout_seconds: float,
    on_exchange_attempt: Optional[Callable[[], None]],
) -> Dict[str, str]:
    url = f"{dashboard_base.rstrip('/')}{CURSOR_AGENT_AUTH_EXCHANGE_PATH}"
    request = urllib_request.Request(
        url,
        data=b"{}",
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "user-agent": cursor_agent_user_agent(),
            "accept": "application/json",
        },
        method="POST",
    )
    try:
        if on_exchange_attempt is not None:
            on_exchange_attempt()
        with urllib_request.urlopen(
            request,
            timeout=http_timeout_seconds,
        ) as response:
            status_code = _response_status_code(response)
            body = response.read()
    except urllib_error.HTTPError as exc:
        raise CursorAgentAuthExchangeError(
            f"Cursor Agent API-key exchange failed with HTTP {int(exc.code)}."
        ) from exc
    except urllib_error.URLError as exc:
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange failed while contacting Cursor."
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange failed while contacting Cursor."
        ) from exc
    except Exception as exc:
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange failed."
        ) from exc

    if status_code is not None and status_code >= 400:
        raise CursorAgentAuthExchangeError(
            f"Cursor Agent API-key exchange failed with HTTP {status_code}."
        )
    try:
        if isinstance(body, bytes):
            decoded_body = body.decode("utf-8")
        elif isinstance(body, str):
            decoded_body = body
        else:
            raise TypeError
        payload = json.loads(decoded_body)
    except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange returned invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange returned a non-object payload."
        )

    access_token = _clean_string(payload.get("accessToken"))
    refresh_token = _clean_string(payload.get("refreshToken"))
    if access_token is None:
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange did not return accessToken."
        )
    if refresh_token is None:
        raise CursorAgentAuthExchangeError(
            "Cursor Agent API-key exchange did not return refreshToken."
        )
    return {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        **_exchange_expiry_fields(payload),
    }


def _exchange_expiry_fields(payload: Mapping[str, Any]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for key in _ALL_EXPIRY_KEYS:
        value = payload.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            fields[key] = str(value)
    return fields


def _merge_exchange_result(
    auth_data: Mapping[str, Any],
    exchanged: Mapping[str, str],
    *,
    api_key: str,
    observed_at: datetime,
) -> Dict[str, Any]:
    updated: Dict[str, Any] = dict(auth_data)
    for key in _ALL_EXPIRY_KEYS:
        updated.pop(key, None)

    _set_existing_or_canonical(
        updated,
        _ACCESS_TOKEN_KEYS,
        exchanged["accessToken"],
    )
    _set_existing_or_canonical(
        updated,
        _REFRESH_TOKEN_KEYS,
        exchanged["refreshToken"],
    )
    _set_existing_or_canonical(updated, _API_KEY_KEYS, api_key)
    updated["obtained_at"] = _json_number(observed_at.timestamp())
    updated.pop("issued_at", None)
    updated.pop("refreshed_at", None)

    for key in _ALL_EXPIRY_KEYS:
        if key in exchanged:
            updated[key] = exchanged[key]
    if "expiresIn" in exchanged and not any(
        key in exchanged for key in _EXPIRY_KEYS
    ):
        try:
            seconds = float(exchanged["expiresIn"])
            if math.isfinite(seconds) and seconds >= 0:
                updated["expiresAt"] = int(
                    (observed_at + timedelta(seconds=seconds)).timestamp()
                )
        except (OverflowError, ValueError):
            pass
    if "expires_in" in exchanged and not any(
        key in exchanged for key in _EXPIRY_KEYS
    ):
        try:
            seconds = float(exchanged["expires_in"])
            if math.isfinite(seconds) and seconds >= 0:
                updated["expires_at"] = int(
                    (observed_at + timedelta(seconds=seconds)).timestamp()
                )
        except (OverflowError, ValueError):
            pass
    return updated


def _set_existing_or_canonical(
    payload: MutableMapping[str, Any],
    keys: tuple[str, ...],
    value: str,
) -> None:
    existing = [key for key in keys if key in payload]
    for key in existing or [keys[0]]:
        payload[key] = value


def _health_summary(
    auth_path: Path,
    state: Optional[_CredentialState],
    *,
    observed_at: datetime,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    if state is None:
        return _summary(
            auth_path,
            None,
            observed_at=observed_at,
            status_class="malformed",
            health_status="malformed",
            access_state="malformed",
            usable=False,
            attempted=True,
            refreshed=False,
            skipped=False,
            error_class=error_class,
            error_message=error_message,
        )
    access_state = _access_token_state(state, observed_at)
    health_status = (
        "expired"
        if access_state == "expired"
        else "degraded"
        if access_state in {"missing", "unknown"}
        else "fresh"
    )
    return _summary(
        auth_path,
        state,
        observed_at=observed_at,
        status_class=health_status,
        health_status=health_status,
        access_state=access_state,
        usable=_access_token_usable(state, observed_at),
        attempted=True,
        refreshed=False,
        skipped=False,
        error_class=error_class,
        error_message=error_message,
    )


def _eligibility_summary(
    auth_path: Path,
    state: Optional[_CredentialState],
    *,
    observed_at: datetime,
    refresh_due_at: Optional[datetime],
    next_refresh_check_at: datetime,
    eligible: bool,
    access_state: str,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    if state is None:
        return _summary(
            auth_path,
            None,
            observed_at=observed_at,
            status_class="malformed",
            health_status="malformed",
            access_state=access_state,
            usable=False,
            attempted=True,
            refreshed=False,
            skipped=False,
            eligible=eligible,
            refresh_due_at=refresh_due_at,
            next_refresh_check_at=next_refresh_check_at,
            error_class=error_class,
            error_message=error_message,
        )
    health_status = (
        "expired"
        if access_state == "expired"
        else "degraded"
        if access_state in {"missing", "unknown"}
        else "fresh"
    )
    return _summary(
        auth_path,
        state,
        observed_at=observed_at,
        status_class="eligible" if eligible else health_status,
        health_status=health_status,
        access_state=access_state,
        usable=_access_token_usable(state, observed_at),
        attempted=True,
        refreshed=False,
        skipped=False,
        eligible=eligible,
        refresh_due_at=refresh_due_at,
        next_refresh_check_at=next_refresh_check_at,
        error_class=error_class,
        error_message=error_message,
    )


def _refresh_summary(
    auth_path: Path,
    state: _CredentialState,
    *,
    attempted: bool,
    refreshed: bool,
    skipped: bool,
    observed_at: datetime,
    refresh_due_at: Optional[datetime],
    refresh_method: str,
    previous_credential_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    access_state = _access_token_state(state, observed_at)
    health_status = (
        "expired"
        if access_state == "expired"
        else "degraded"
        if access_state in {"missing", "unknown"}
        else "fresh"
    )
    return _summary(
        auth_path,
        state,
        observed_at=observed_at,
        status_class="refreshed"
        if refreshed
        else "skipped"
        if skipped
        else health_status,
        health_status=health_status,
        access_state=access_state,
        usable=_access_token_usable(state, observed_at),
        attempted=attempted,
        refreshed=refreshed,
        skipped=skipped,
        refresh_due_at=refresh_due_at,
        refresh_method=refresh_method,
        previous_credential_fingerprint=previous_credential_fingerprint,
        error_class=None,
        error_message=None,
    )


def _refresh_error_summary(
    auth_path: Path,
    state: Optional[_CredentialState],
    *,
    observed_at: Optional[datetime] = None,
    refresh_due_at: Optional[datetime] = None,
    error_class: str,
    error_message: str,
) -> Dict[str, Any]:
    observed = observed_at or datetime.now(timezone.utc)
    if state is None:
        return _summary(
            auth_path,
            None,
            observed_at=observed,
            status_class="error",
            health_status="malformed",
            access_state="unknown",
            usable=False,
            attempted=True,
            refreshed=False,
            skipped=False,
            refresh_due_at=refresh_due_at,
            refresh_method="none",
            error_class=error_class,
            error_message=error_message,
        )
    access_state = _access_token_state(state, observed)
    health_status = (
        "expired"
        if access_state == "expired"
        else "degraded"
        if access_state in {"missing", "unknown"}
        else "fresh"
    )
    return _summary(
        auth_path,
        state,
        observed_at=observed,
        status_class="error",
        health_status=health_status,
        access_state=access_state,
        usable=_access_token_usable(state, observed),
        attempted=True,
        refreshed=False,
        skipped=False,
        refresh_due_at=refresh_due_at,
        refresh_method="apiKey_exchange"
        if state.api_key is not None
        else "none",
        error_class=error_class,
        error_message=error_message,
    )


def _summary(
    auth_path: Path,
    state: Optional[_CredentialState],
    *,
    observed_at: datetime,
    status_class: str,
    health_status: str,
    access_state: str,
    usable: bool,
    attempted: bool,
    refreshed: bool,
    skipped: bool,
    eligible: Optional[bool] = None,
    refresh_due_at: Optional[datetime] = None,
    next_refresh_check_at: Optional[datetime] = None,
    refresh_method: str = "none",
    previous_credential_fingerprint: Optional[str] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    if state is None:
        issued_lifetime_seconds: Optional[float] = None
        threshold = float(DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS)
        threshold_source = "fallback"
        threshold_degraded = True
    else:
        (
            issued_lifetime_seconds,
            threshold,
            threshold_source,
            threshold_degraded,
        ) = _credential_refresh_threshold_metadata(state)
    result: Dict[str, Any] = {
        "provider": "cursor_agent",
        "status_class": status_class,
        "attempted": attempted,
        "refreshed": refreshed,
        "skipped": skipped,
        "auth_file": str(auth_path),
        "credential_shape": state.shape if state is not None else None,
        "credential_fingerprint": (
            state.fingerprint if state is not None else None
        ),
        "previous_credential_fingerprint": previous_credential_fingerprint,
        "has_access_token": bool(state and state.access_token),
        "has_refresh_token": bool(state and state.refresh_token),
        "has_api_key": bool(state and state.api_key),
        "refresh_capability": (
            "apiKey_exchange"
            if state and state.api_key
            else "unsupported_refreshToken_only"
            if state and state.refresh_token
            else "none"
        ),
        "health_status": health_status,
        "credential_health": health_status,
        "access_token_state": access_state,
        "usable": usable,
        "expires_at": _format_timestamp(
            state.expires_at if state is not None else None
        ),
        "refresh_due_at": _format_datetime(refresh_due_at),
        "next_refresh_check_at": _format_datetime(next_refresh_check_at),
        "refresh_method": refresh_method,
        "eligibility_checked_at": _format_datetime(observed_at),
        "issued_lifetime_seconds": issued_lifetime_seconds,
        "refresh_threshold_seconds": threshold,
        "auth_degraded": threshold_degraded,
        "refresh_threshold_source": threshold_source,
        "refresh_threshold_degraded": threshold_degraded,
        "error_class": error_class,
        "error_message": error_message,
    }
    if eligible is not None:
        result["eligible"] = eligible
    return result


def _run_singleflight(
    key: tuple[Any, ...],
    operation: Callable[[], Dict[str, Any]],
    failure: Callable[[Exception], Dict[str, Any]],
) -> Dict[str, Any]:
    with _SINGLEFLIGHT_GUARD:
        call = _SINGLEFLIGHT_CALLS.get(key)
        if call is None:
            call = _SingleflightCall(event=threading.Event())
            _SINGLEFLIGHT_CALLS[key] = call
            owner = True
        else:
            owner = False

    if not owner:
        call.event.wait()
        return dict(call.result or failure(RuntimeError("singleflight failed")))

    try:
        result = operation()
    except Exception as exc:
        result = failure(exc)
    with _SINGLEFLIGHT_GUARD:
        call.result = dict(result)
        _SINGLEFLIGHT_CALLS.pop(key, None)
        call.event.set()
    return dict(result)


@contextmanager
def _credential_file_lock(lock_path: Path) -> Iterator[None]:
    """Delegate locking to the shared fail-closed credential lock."""
    with credential_file_lock(lock_path):
        yield


def _snapshot_credential_file_metadata(
    auth_path: Path,
) -> CredentialFileMetadata:
    return snapshot_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_CURSOR_AGENT_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _resolve_credential_file_metadata(
    auth_path: Path,
) -> CredentialFileMetadata:
    return resolve_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_CURSOR_AGENT_AUTH_FILE_MODE,
        mode_env="AAWM_CURSOR_AGENT_AUTH_FILE_MODE",
        uid_env="AAWM_CURSOR_AGENT_AUTH_FILE_UID",
        gid_env="AAWM_CURSOR_AGENT_AUTH_FILE_GID",
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
        default_mode=DEFAULT_CURSOR_AGENT_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _repair_credential_file_metadata(auth_path: Path) -> None:
    if not auth_path.exists() and not is_symlink_path(auth_path):
        return
    metadata = _resolve_credential_file_metadata(auth_path)
    _apply_credential_file_metadata(auth_path, metadata)


def _write_auth_data(
    auth_path: Path,
    auth_data: Mapping[str, Any],
) -> None:
    metadata = _resolve_credential_file_metadata(auth_path)
    payload = json.dumps(auth_data, indent=2) + "\n"
    write_and_publish_private_text(
        auth_path,
        payload,
        metadata=metadata,
        default_mode=DEFAULT_CURSOR_AGENT_AUTH_FILE_MODE,
        mkdir_parents=True,
    )


def _read_auth_data(auth_path: Path) -> Dict[str, Any]:
    ensure_not_symlink_path(auth_path, role="Cursor Agent auth path")
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise CursorAgentCredentialError(
            "Cursor Agent auth file was not found."
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise CursorAgentCredentialError(
            "Cursor Agent auth file could not be read."
        ) from exc
    except json.JSONDecodeError as exc:
        raise CursorAgentCredentialError(
            "Cursor Agent auth file is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise CursorAgentCredentialError(
            "Cursor Agent auth file must contain a JSON object."
        )
    return payload


def _credential_state(payload: Mapping[str, Any]) -> _CredentialState:
    _validate_known_field_types(payload)
    access_token = _first_string(payload, _ACCESS_TOKEN_KEYS)
    refresh_token = _first_string(payload, _REFRESH_TOKEN_KEYS)
    api_key = _first_string(payload, _API_KEY_KEYS)
    expires_at = _credential_expiry(payload, access_token)
    shape = _credential_shape(
        access_token=access_token,
        refresh_token=refresh_token,
        api_key=api_key,
    )
    fingerprint = _credential_fingerprint(
        access_token=access_token,
        refresh_token=refresh_token,
        api_key=api_key,
    )
    return _CredentialState(
        access_token=access_token,
        refresh_token=refresh_token,
        api_key=api_key,
        expires_at=expires_at,
        shape=shape,
        fingerprint=fingerprint,
        expires_in=_credential_relative_expiry(payload),
        issued_at=payload.get("issued_at"),
        obtained_at=payload.get("obtained_at"),
        refreshed_at=payload.get("refreshed_at"),
    )


def _validate_known_field_types(payload: Mapping[str, Any]) -> None:
    for key in (*_ACCESS_TOKEN_KEYS, *_REFRESH_TOKEN_KEYS, *_API_KEY_KEYS):
        if key in payload and payload[key] is not None and not isinstance(
            payload[key], str
        ):
            raise CursorAgentCredentialError(
                f"Cursor Agent auth field {key} must be a string."
            )


def _first_string(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Optional[str]:
    for key in keys:
        value = _clean_string(payload.get(key))
        if value is not None:
            return value
    return None


def _credential_relative_expiry(payload: Mapping[str, Any]) -> Any:
    first_value: Any = None
    for key in _RELATIVE_EXPIRY_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if first_value is None:
            first_value = value
        lifetime = _as_finite_number(value)
        if lifetime is not None and lifetime > 0:
            return value
    return first_value


def _credential_shape(
    *,
    access_token: Optional[str],
    refresh_token: Optional[str],
    api_key: Optional[str],
) -> str:
    fields = [
        name
        for name, value in (
            ("accessToken", access_token),
            ("refreshToken", refresh_token),
            ("apiKey", api_key),
        )
        if value is not None
    ]
    return "+".join(fields) if fields else "empty"


def _credential_fingerprint(
    *,
    access_token: Optional[str],
    refresh_token: Optional[str],
    api_key: Optional[str],
) -> Optional[str]:
    values = (
        ("accessToken", access_token),
        ("refreshToken", refresh_token),
        ("apiKey", api_key),
    )
    material = "\0".join(
        f"{name}={value}" for name, value in values if value is not None
    )
    if not material:
        return None
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _credential_expiry(
    payload: Mapping[str, Any],
    access_token: Optional[str],
) -> Optional[float]:
    for key in _EXPIRY_KEYS:
        if key in payload:
            parsed = _parse_timestamp(payload.get(key))
            if parsed is not None:
                return parsed
    if access_token is not None:
        jwt_expiry = _decode_jwt_exp(access_token)
        if jwt_expiry is not None:
            return jwt_expiry
    issued_at = _first_timestamp_seconds(
        payload.get("issued_at"),
        payload.get("obtained_at"),
        payload.get("refreshed_at"),
    )
    lifetime = _as_finite_number(_credential_relative_expiry(payload))
    if issued_at is not None and lifetime is not None and lifetime > 0:
        expires_at = issued_at + lifetime
        if math.isfinite(expires_at):
            return expires_at
    return None


def _jwt_time_claims(access_token: Any) -> Optional[Tuple[float, float]]:
    payload = _decode_jwt_payload(access_token)
    if payload is None:
        return None
    issued_at = _as_finite_number(payload.get("iat"))
    expires_at = _as_finite_number(payload.get("exp"))
    if (
        issued_at is None
        or expires_at is None
        or expires_at <= issued_at
    ):
        return None
    return issued_at, expires_at


def _decode_jwt_payload(token: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(token, str) or not token.strip():
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        encoded_payload = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(encoded_payload.encode("ascii"))
        )
    except (
        ValueError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        base64.binascii.Error,
    ):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _decode_jwt_exp(token: str) -> Optional[float]:
    payload = _decode_jwt_payload(token)
    if payload is None:
        return None
    value = payload.get("exp")
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed > 1_000_000_000_000:
            parsed /= 1000
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.astimezone(timezone.utc).timestamp()
        return timestamp if math.isfinite(timestamp) else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = float(text)
    except ValueError:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed_datetime = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=timezone.utc)
        return parsed_datetime.astimezone(timezone.utc).timestamp()
    if parsed > 1_000_000_000_000:
        parsed /= 1000
    return parsed if math.isfinite(parsed) else None


def _first_timestamp_seconds(*values: Any) -> Optional[float]:
    for value in values:
        timestamp = _parse_timestamp(value)
        if timestamp is not None:
            return timestamp
    return None


def _access_token_state(
    state: _CredentialState,
    observed_at: datetime,
    *,
    buffer_seconds: int = 0,
) -> str:
    if state.access_token is None:
        return "missing"
    if state.expires_at is None:
        return "unknown"
    if state.expires_at <= observed_at.timestamp():
        return "expired"
    threshold, _source, _degraded = _credential_refresh_threshold_metadata(state)
    if state.expires_at <= observed_at.timestamp() + max(0, threshold):
        return "due"
    return "fresh"


def _access_token_usable(
    state: _CredentialState,
    observed_at: datetime,
) -> bool:
    return state.access_token is not None and (
        state.expires_at is None
        or state.expires_at > observed_at.timestamp()
    )


def _refresh_is_due(
    state: _CredentialState,
    *,
    observed_at: datetime,
    buffer_seconds: int,
) -> bool:
    if state.access_token is None:
        return True
    if state.expires_at is None:
        # Cursor's consumer treats opaque access tokens as usable because no
        # expiry claim is available. A known expired token never reaches this
        # branch.
        return False
    threshold, _source, _degraded = _credential_refresh_threshold_metadata(state)
    return observed_at.timestamp() >= state.expires_at - max(0, threshold)


def _refresh_due_at(
    expires_at: Optional[float],
    *,
    buffer_seconds: int,
    access_token: Optional[str] = None,
    expires_in: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
) -> Optional[datetime]:
    if expires_at is None:
        return None
    threshold = _refresh_threshold_seconds(
        expires_in=expires_in,
        access_token=access_token,
        expires_at=expires_at,
        issued_at=issued_at,
        obtained_at=obtained_at,
        refreshed_at=refreshed_at,
    )
    try:
        return datetime.fromtimestamp(
            expires_at - max(0, threshold),
            timezone.utc,
        )
    except (OSError, OverflowError, ValueError):
        return None


def _resolve_auth_file(auth_file: str | Path | None) -> Path:
    if auth_file is not None:
        return Path(auth_file).expanduser()
    return Path(
        os.getenv(
            "AAWM_CURSOR_AGENT_AUTH_FILE",
            DEFAULT_CURSOR_AGENT_AUTH_FILE,
        )
    ).expanduser()


def _resolve_lock_file(
    *,
    auth_file: str | Path | None,
    auth_path: Path,
    lock_file: str | Path | None,
) -> Path:
    if lock_file is not None:
        return Path(lock_file).expanduser()
    env_lock = os.getenv("AAWM_CURSOR_AGENT_AUTH_LOCK_FILE")
    if isinstance(env_lock, str) and env_lock.strip():
        return Path(env_lock).expanduser()
    return auth_path.with_name(f"{auth_path.name}.lock")


def _resolve_buffer_seconds(buffer_seconds: Optional[int]) -> int:
    if buffer_seconds is not None:
        return max(0, int(buffer_seconds))
    raw_value = os.getenv("AAWM_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS
    return max(0, int(raw_value))


def _resolve_refresh_interval_seconds(
    interval_seconds: Optional[float],
) -> float:
    if interval_seconds is not None:
        return _resolve_positive_float(
            interval_seconds,
            default=DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS,
        )
    raw_value = os.getenv("AAWM_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS
    return _resolve_positive_float(
        raw_value,
        default=DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS,
    )


def _resolve_force_refresh(force: Optional[bool]) -> bool:
    if force is not None:
        return bool(force)
    raw_value = os.getenv("AAWM_CURSOR_AGENT_AUTH_FORCE_REFRESH")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CURSOR_AGENT_AUTH_FORCE_REFRESH
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "AAWM_CURSOR_AGENT_AUTH_FORCE_REFRESH must be a boolean value."
    )


def _resolve_timeout_seconds(timeout: Optional[float]) -> float:
    if timeout is not None:
        return _resolve_positive_float(
            timeout,
            default=DEFAULT_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS,
        )
    raw_value = os.getenv("AAWM_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS
    return _resolve_positive_float(
        raw_value,
        default=DEFAULT_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS,
    )


def _resolve_dashboard_base(dashboard_base: Optional[str]) -> str:
    if dashboard_base is None or not dashboard_base.strip():
        return CURSOR_AGENT_DASHBOARD_HOST
    return dashboard_base.strip()


def _resolve_positive_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        return default
    return parsed


def _resolve_wall_now(now: Optional[NowProvider | NowValue]) -> datetime:
    value: Any
    if now is None:
        value = time.time()
    elif callable(now):
        value = now()
    else:
        value = now
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(value), timezone.utc)


def _response_status_code(response: Any) -> Optional[int]:
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        value = getcode()
        if value is not None:
            return int(value)
    value = getattr(response, "status", None)
    if value is None:
        return None
    return int(value)


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _format_timestamp(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    try:
        return (
            datetime.fromtimestamp(value, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, ValueError):
        return None


def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
    message: str,
    *,
    limit: int = DEFAULT_CURSOR_AGENT_AUTH_ERROR_MESSAGE_LIMIT,
) -> str:
    return sanitize_credential_error_message(
        message,
        limit=limit,
        field_names=_SECRET_FIELD_NAMES,
    )


__all__ = (
    "DEFAULT_CURSOR_AGENT_AUTH_ERROR_MESSAGE_LIMIT",
    "DEFAULT_CURSOR_AGENT_AUTH_EXCHANGE_URL",
    "DEFAULT_CURSOR_AGENT_AUTH_FILE",
    "DEFAULT_CURSOR_AGENT_AUTH_FILE_MODE",
    "DEFAULT_CURSOR_AGENT_AUTH_FORCE_REFRESH",
    "DEFAULT_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS",
    "DEFAULT_CURSOR_AGENT_AUTH_LOCK_FILE",
    "DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS",
    "DEFAULT_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS",
    "DEFAULT_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS",
    "CursorAgentAuthError",
    "CursorAgentAuthExchangeError",
    "CursorAgentCredentialError",
    "CursorAgentRefreshTokenOnlyError",
    "inspect_cursor_agent_auth_credential_health",
    "inspect_cursor_agent_auth_refresh_eligibility",
    "refresh_cursor_agent_auth_file",
)

# Backward-compatible alias for tests and callers that still reference
# the old buffer-seconds constant.
DEFAULT_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS = DEFAULT_CURSOR_AGENT_AUTH_REFRESH_MIN_SECONDS
