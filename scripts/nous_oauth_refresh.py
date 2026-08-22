#!/usr/bin/env python3
"""Refresh the Hermes Nous Portal OAuth credential for the provider-status sidecar.

Sidecar-only writer. LiteLLM request handling must never import this to mutate
``~/.hermes/auth.json``. Lock path follows Hermes-native
``Path.with_suffix(".lock")`` (``auth.json`` -> ``auth.lock``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)
from litellm.secret_managers.credential_file_lock import credential_file_lock
from litellm.secret_managers.credential_file_metadata import (
    CredentialFileMetadata,
    apply_credential_file_metadata,
    resolve_credential_file_metadata,
    snapshot_credential_file_metadata,
)
from litellm.secret_managers.credential_file_write import (
    write_and_publish_private_text,
    write_private_temp_file_text,
)

# Portable ~ defaults (expanded via Path.expanduser at use sites).
DEFAULT_NOUS_OAUTH_AUTH_FILE = "~/.hermes/auth.json"
DEFAULT_NOUS_OAUTH_LOCK_FILE = "~/.hermes/auth.lock"
DEFAULT_NOUS_OAUTH_PORTAL_BASE_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_OAUTH_TOKEN_ENDPOINT = "https://portal.nousresearch.com/api/oauth/token"
DEFAULT_NOUS_OAUTH_CLIENT_ID = "hermes-cli"
DEFAULT_NOUS_OAUTH_SCOPE = "inference:invoke"
DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS = 900
DEFAULT_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS = 300
DEFAULT_NOUS_OAUTH_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_NOUS_OAUTH_AUTH_FILE_MODE = 0o600
DEFAULT_NOUS_OAUTH_FORCE_REFRESH = False
DEFAULT_NOUS_OAUTH_ERROR_MESSAGE_LIMIT = 500
NOUS_REFRESH_TOKEN_HEADER = "x-nous-refresh-token"
_SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES
_TOKEN_PATH = "/api/oauth/token"


@dataclass(frozen=True)
class NousOAuthRefreshSummary:
    attempted: bool
    refreshed: bool
    skipped: bool
    auth_file: str
    scope: str = DEFAULT_NOUS_OAUTH_SCOPE
    expires_at: Optional[str] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None

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
        }


def inspect_nous_oauth_credential_health(
    auth_file: str | Path, *, scope: Optional[str] = None
) -> Dict[str, Any]:
    """Read and classify Nous OAuth state without locks, writes, or HTTP."""
    resolved_auth_file = Path(auth_file).expanduser()
    resolved_scope = _resolve_scope(scope)
    try:
        payload = _read_credential_payload(resolved_auth_file)
        record = _select_nous_record(payload)
        expires_at = _earliest_expiry(record)
        usable = _record_usable(record)
        if expires_at is None:
            return _health_summary(
                resolved_auth_file,
                resolved_scope,
                "degraded",
                error_class="CredentialExpiryUnavailable",
                error_message="Nous OAuth credential expires_at is missing or invalid.",
            )
        if expires_at <= datetime.now(timezone.utc):
            return _health_summary(
                resolved_auth_file,
                resolved_scope,
                "expired",
                expires_at,
                error_class="CredentialExpiredError",
                error_message="Nous OAuth credential is expired.",
            )
        return _health_summary(
            resolved_auth_file, resolved_scope, "fresh", expires_at, usable=usable
        )
    except Exception as exc:
        return _health_summary(
            resolved_auth_file,
            resolved_scope,
            "malformed",
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
            usable=False,
        )


def inspect_nous_oauth_refresh_eligibility(
    auth_file: str | Path,
    *,
    buffer_seconds: int = DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS,
    now: Optional[Callable[[], datetime]] = None,
    poll_interval_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Inspect Hermes Nous refresh eligibility without lock, write, or HTTP."""
    resolved_auth_file = Path(auth_file).expanduser()
    observed_at = _resolve_wall_now(now)
    try:
        payload = _read_credential_payload(resolved_auth_file)
        record = _select_nous_record(payload)
        usable = _record_usable(record)
        expires_at, expiry_unavailable = _earliest_expiry_with_status(record)
        if expiry_unavailable:
            return _eligibility_summary(
                observed_at=observed_at,
                expires_at=None,
                refresh_due_at=None,
                next_refresh_check_at=observed_at
                + timedelta(seconds=max(1.0, poll_interval_seconds)),
                eligible=True,
                credential_health="degraded",
                usable=usable,
                error_class="CredentialExpiryUnavailable",
                error_message="Nous OAuth credential expires_at is missing or invalid.",
            )
        assert expires_at is not None
        refresh_due_at = expires_at - timedelta(seconds=max(0, int(buffer_seconds)))
        return _eligibility_summary(
            observed_at=observed_at,
            expires_at=expires_at,
            refresh_due_at=refresh_due_at,
            next_refresh_check_at=(
                refresh_due_at
                if observed_at < refresh_due_at
                else observed_at + timedelta(seconds=max(1.0, poll_interval_seconds))
            ),
            eligible=observed_at >= refresh_due_at,
            credential_health="expired" if expires_at <= observed_at else "fresh",
            usable=usable and expires_at > observed_at,
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
        )


def refresh_nous_oauth_auth_file(
    auth_file: str | Path,
    *,
    buffer_seconds: Optional[int] = None,
    force: bool = False,
    lock_file: str | Path | None = None,
    http_timeout_seconds: float = DEFAULT_NOUS_OAUTH_HTTP_TIMEOUT_SECONDS,
    on_token_endpoint_attempt: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Refresh Hermes ``providers.nous`` / ``credential_pool.nous`` when due."""
    resolved_auth_file = Path(auth_file).expanduser()
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)
    resolved_lock_file = (
        Path(lock_file).expanduser()
        if lock_file is not None
        else _default_lock_path(resolved_auth_file)
    )
    current_scope = DEFAULT_NOUS_OAUTH_SCOPE
    current_expires_at: Optional[str] = None

    if not force:
        eligibility = inspect_nous_oauth_refresh_eligibility(
            resolved_auth_file,
            buffer_seconds=resolved_buffer_seconds,
        )
        if not eligibility.get("eligible"):
            return NousOAuthRefreshSummary(
                attempted=False,
                refreshed=False,
                skipped=True,
                auth_file=str(resolved_auth_file),
                scope=current_scope,
                expires_at=eligibility.get("expires_at"),
            ).as_dict()

    try:
        with credential_file_lock(resolved_lock_file):
            payload = _read_credential_payload(resolved_auth_file)
            record = _select_nous_record(payload)
            current_scope = (
                _clean_oauth_string(record.get("scope")) or DEFAULT_NOUS_OAUTH_SCOPE
            )
            current_expires_at = _format_expires_at(_earliest_expiry(record))
            if not force and not _credential_needs_refresh(
                record, buffer_seconds=resolved_buffer_seconds
            ):
                return NousOAuthRefreshSummary(
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    auth_file=str(resolved_auth_file),
                    scope=current_scope,
                    expires_at=current_expires_at,
                ).as_dict()

            refreshed = _refresh_credential_record(
                record,
                http_timeout_seconds=http_timeout_seconds,
                on_token_endpoint_attempt=on_token_endpoint_attempt,
            )
            _apply_refreshed_tokens(payload, record, refreshed)
            _write_credential_payload(resolved_auth_file, payload)
            updated = _select_nous_record(payload)
            return NousOAuthRefreshSummary(
                attempted=True,
                refreshed=True,
                skipped=False,
                auth_file=str(resolved_auth_file),
                scope=_clean_oauth_string(updated.get("scope")) or current_scope,
                expires_at=_format_expires_at(_earliest_expiry(updated)),
            ).as_dict()
    except Exception as exc:
        error_class, error_message = _classify_oauth_error(exc)
        return NousOAuthRefreshSummary(
            attempted=True,
            refreshed=False,
            skipped=False,
            auth_file=str(resolved_auth_file),
            scope=current_scope,
            expires_at=current_expires_at,
            error_class=error_class,
            error_message=error_message,
        ).as_dict()


def _health_summary(
    auth_file: Path,
    scope: str,
    health_status: str,
    expires_at: Optional[datetime] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
    usable: bool = True,
) -> Dict[str, Any]:
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
        "usable": usable and health_status == "fresh",
    }


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
) -> Dict[str, Any]:
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
    }


def _default_lock_path(auth_path: Path) -> Path:
    return auth_path.with_suffix(".lock")


def _resolve_wall_now(now: Optional[Callable[[], datetime]]) -> datetime:
    value = now() if now is not None else datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_scope(scope: Optional[str]) -> str:
    if isinstance(scope, str) and scope.strip():
        return scope.strip()
    return DEFAULT_NOUS_OAUTH_SCOPE


def _resolve_buffer_seconds(buffer_seconds: Optional[int]) -> int:
    if buffer_seconds is not None:
        return max(0, int(buffer_seconds))
    raw_value = os.getenv("AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_NOUS_OAUTH_REFRESH_BUFFER_SECONDS


def _snapshot_credential_file_metadata(auth_path: Path) -> CredentialFileMetadata:
    return snapshot_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_NOUS_OAUTH_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _resolve_credential_file_metadata(auth_path: Path) -> CredentialFileMetadata:
    return resolve_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_NOUS_OAUTH_AUTH_FILE_MODE,
        mode_env="AAWM_NOUS_OAUTH_AUTH_FILE_MODE",
        uid_env="AAWM_NOUS_OAUTH_AUTH_FILE_UID",
        gid_env="AAWM_NOUS_OAUTH_AUTH_FILE_GID",
        base_metadata=_snapshot_credential_file_metadata(auth_path),
        refuse_symlink=True,
    )


def _apply_credential_file_metadata(
    target_path: Path, metadata: CredentialFileMetadata
) -> None:
    apply_credential_file_metadata(
        target_path,
        metadata,
        default_mode=DEFAULT_NOUS_OAUTH_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _read_credential_payload(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Nous OAuth auth file not found at {auth_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Nous OAuth auth file at {auth_path} is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Nous OAuth auth file must contain a JSON object.")
    return payload


def _select_nous_record(payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
    providers = payload.get("providers")
    if isinstance(providers, Mapping):
        singleton = providers.get("nous")
        if isinstance(singleton, dict) and _looks_like_nous_record(singleton):
            return singleton
    pool = payload.get("credential_pool")
    if isinstance(pool, Mapping):
        entries = pool.get("nous")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and _looks_like_nous_record(entry):
                    return entry
    raise ValueError("Hermes auth file does not contain a usable providers.nous record.")


def _looks_like_nous_record(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("refresh_token")
        or value.get("access_token")
        or value.get("agent_key")
    )


def _record_usable(record: Mapping[str, Any]) -> bool:
    return bool(
        _clean_oauth_string(record.get("access_token"))
        or _clean_oauth_string(record.get("agent_key"))
        or _clean_oauth_string(record.get("refresh_token"))
    )


def _earliest_expiry(record: Mapping[str, Any]) -> Optional[datetime]:
    expires_at, unavailable = _earliest_expiry_with_status(record)
    if unavailable:
        return None
    return expires_at


def _earliest_expiry_with_status(
    record: Mapping[str, Any],
) -> tuple[Optional[datetime], bool]:
    access_raw = record.get("expires_at")
    agent_raw = record.get("agent_key_expires_at")
    access_missing = access_raw in (None, "")
    agent_missing = agent_raw in (None, "")
    if access_missing and agent_missing:
        return None, True
    parsed: list[datetime] = []
    unparseable = False
    for raw in (access_raw, agent_raw):
        if raw in (None, ""):
            continue
        value = _parse_expires_at(raw)
        if value is None:
            unparseable = True
        else:
            parsed.append(value)
    if unparseable or not parsed:
        return None, True
    return min(parsed), False


def _credential_needs_refresh(
    record: Mapping[str, Any], *, buffer_seconds: int
) -> bool:
    expires_at, unavailable = _earliest_expiry_with_status(record)
    if unavailable or expires_at is None:
        return True
    return datetime.now(timezone.utc) >= expires_at - timedelta(seconds=buffer_seconds)


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


def _token_endpoint_for(record: Mapping[str, Any]) -> str:
    portal = _clean_oauth_string(record.get("portal_base_url")) or (
        DEFAULT_NOUS_OAUTH_PORTAL_BASE_URL
    )
    return portal.rstrip("/") + _TOKEN_PATH


def _refresh_credential_record(
    record: Mapping[str, Any],
    *,
    http_timeout_seconds: float,
    on_token_endpoint_attempt: Optional[Callable[[], None]] = None,
) -> Mapping[str, Any]:
    refresh_token = _clean_oauth_string(record.get("refresh_token"))
    if refresh_token is None:
        raise ValueError(
            "Nous OAuth credential is expired or near expiry and has no refresh_token."
        )
    client_id = (
        _clean_oauth_string(record.get("client_id")) or DEFAULT_NOUS_OAUTH_CLIENT_ID
    )
    body = urllib_parse.urlencode(
        {"grant_type": "refresh_token", "client_id": client_id}
    ).encode("utf-8")
    request = urllib_request.Request(
        _token_endpoint_for(record),
        data=body,
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
            NOUS_REFRESH_TOKEN_HEADER: refresh_token,
        },
        method="POST",
    )
    try:
        if on_token_endpoint_attempt is not None:
            on_token_endpoint_attempt()
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            response_body = response.read()
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise _NousOAuthHttpError(exc.code, error_body) from exc
    except urllib_error.URLError as exc:
        raise ValueError(
            f"Nous OAuth refresh failed: {_sanitize_error_message(str(exc.reason))}"
        ) from exc

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Nous OAuth refresh response was not valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Nous OAuth refresh response must contain a JSON object.")
    if _clean_oauth_string(payload.get("access_token")) is None:
        raise ValueError("Nous OAuth refresh response did not contain an access_token.")
    return payload


def _apply_refreshed_tokens(
    payload: MutableMapping[str, Any],
    selected: MutableMapping[str, Any],
    refreshed: Mapping[str, Any],
) -> None:
    access_token = _clean_oauth_string(refreshed.get("access_token"))
    if access_token is None:
        raise ValueError("Nous OAuth refresh response did not contain an access_token.")
    rotated_refresh = _clean_oauth_string(refreshed.get("refresh_token"))
    expires_in = refreshed.get("expires_in")
    now = datetime.now(timezone.utc)
    if isinstance(expires_in, (int, float)):
        expires_at = now + timedelta(seconds=float(expires_in))
        expires_at_text = _format_expires_at(expires_at)
        expires_in_value: Any = int(expires_in)
    else:
        expires_at_text = _format_expires_at(now + timedelta(seconds=3600))
        expires_in_value = 3600

    updates: Dict[str, Any] = {
        "access_token": access_token,
        "agent_key": access_token,
        "agent_key_reused": False,
        "expires_at": expires_at_text,
        "expires_in": expires_in_value,
        "agent_key_expires_at": expires_at_text,
        "agent_key_expires_in": expires_in_value,
        "obtained_at": _format_expires_at(now),
        "agent_key_obtained_at": _format_expires_at(now),
    }
    if rotated_refresh is not None:
        updates["refresh_token"] = rotated_refresh
    token_type = _clean_oauth_string(refreshed.get("token_type"))
    if token_type is not None:
        updates["token_type"] = token_type
    scope = _clean_oauth_string(refreshed.get("scope"))
    if scope is not None:
        updates["scope"] = scope

    selected.update(updates)
    providers = payload.get("providers")
    if isinstance(providers, dict):
        existing = providers.get("nous")
        if isinstance(existing, dict):
            existing.update(updates)
        else:
            providers["nous"] = dict(selected)
    pool = payload.get("credential_pool")
    if isinstance(pool, dict):
        entries = pool.get("nous")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and _looks_like_nous_record(entry):
                    entry.update(updates)


def _write_credential_payload(auth_path: Path, payload: Mapping[str, Any]) -> None:
    """Publish Hermes JSON via shared exclusive temp + atomic replace.

    Uses ``write_and_publish_private_text`` so temp names are not pid-only,
    symlink targets are refused, and failed temps are cleaned up consistently.
    Do not ``sort_keys``; Hermes is a multi-provider document.
    """
    try:
        metadata = _resolve_credential_file_metadata(auth_path)
        content = json.dumps(payload, indent=2) + "\n"
        write_and_publish_private_text(
            auth_path,
            content,
            metadata=metadata,
            default_mode=DEFAULT_NOUS_OAUTH_AUTH_FILE_MODE,
            mkdir_parents=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to persist refreshed Nous OAuth auth data: {exc}"
        ) from exc


def _clean_oauth_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _sanitize_error_message(
    message: str, *, limit: int = DEFAULT_NOUS_OAUTH_ERROR_MESSAGE_LIMIT
) -> str:
    return sanitize_credential_error_message(message, limit=limit)


class _NousOAuthHttpError(ValueError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        oauth_error = _parse_oauth_error_code(body)
        sanitized = _sanitize_error_message(body)
        super().__init__(
            f"Nous OAuth refresh failed with HTTP {status_code}: {sanitized}"
        )
        self.oauth_error = oauth_error


def _parse_oauth_error_code(body: str) -> Optional[str]:
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping):
        code = payload.get("error")
        if isinstance(code, str) and code.strip():
            return code.strip()
    return None


def _classify_oauth_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, _NousOAuthHttpError) and exc.oauth_error:
        return exc.oauth_error, _sanitize_error_message(str(exc))
    return exc.__class__.__name__, _sanitize_error_message(str(exc))
