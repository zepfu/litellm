#!/usr/bin/env python3
"""Refresh a managed xAI OAuth auth JSON file for the provider-status sidecar."""

from __future__ import annotations

from litellm.secret_managers.credential_file_lock import credential_file_lock

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

DEFAULT_XAI_OAUTH_AUTH_FILE = "/home/zepfu/.litellm/xai/oauth-auth.json"
DEFAULT_XAI_OAUTH_LOCK_FILE = "/home/zepfu/.litellm/xai/oauth-auth.json.lock"
DEFAULT_XAI_OAUTH_SCOPE = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
DEFAULT_XAI_OAUTH_REFRESH_BUFFER_SECONDS = 300
DEFAULT_XAI_OAUTH_HTTP_TIMEOUT_SECONDS = 30.0

_SECRET_FIELD_NAMES = {
    "access_token",
    "client_secret",
    "id_token",
    "key",
    "refresh_token",
}


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


def _write_private_file_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Create/write path with restrictive mode at creation time (no umask window)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(path, mode)
    except OSError:
        pass



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
) -> Dict[str, Any]:
    """Refresh a managed xAI OAuth auth file when near expiry or forced."""

    resolved_auth_file = Path(auth_file).expanduser()
    resolved_scope = _resolve_scope(scope)
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)
    resolved_lock_file = (
        Path(lock_file).expanduser()
        if lock_file is not None
        else resolved_auth_file.with_name(f"{resolved_auth_file.name}.lock")
    )

    with _credential_file_lock(resolved_lock_file):
        try:
            raw_payload = _read_credential_payload(resolved_auth_file)
            credential = _select_credential_record(raw_payload, resolved_scope)
            current_expires_at = _format_expires_at(_parse_expires_at(credential.get("expires_at")))

            if not force and not _credential_needs_refresh(
                credential,
                buffer_seconds=resolved_buffer_seconds,
            ):
                return XaiOAuthRefreshSummary(
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    auth_file=str(resolved_auth_file),
                    scope=resolved_scope,
                    expires_at=current_expires_at,
                ).as_dict()

            refreshed = _refresh_credential_record(
                credential,
                token_endpoint=token_endpoint,
                client_id=client_id,
                client_secret=client_secret,
                http_timeout_seconds=http_timeout_seconds,
            )
            _update_credential_record(credential, refreshed)
            _write_credential_payload(resolved_auth_file, raw_payload)
            return XaiOAuthRefreshSummary(
                attempted=True,
                refreshed=True,
                skipped=False,
                auth_file=str(resolved_auth_file),
                scope=resolved_scope,
                expires_at=_format_expires_at(
                    _parse_expires_at(credential.get("expires_at"))
                ),
            ).as_dict()
        except Exception as exc:
            return XaiOAuthRefreshSummary(
                attempted=True,
                refreshed=False,
                skipped=False,
                auth_file=str(resolved_auth_file),
                scope=resolved_scope,
                error_class=exc.__class__.__name__,
                error_message=_sanitize_error_message(str(exc)),
            ).as_dict()


def _resolve_scope(scope: Optional[str]) -> str:
    if isinstance(scope, str) and scope.strip():
        return scope.strip()
    env_scope = os.getenv("AAWM_XAI_OAUTH_SCOPE") or os.getenv("LITELLM_XAI_OAUTH_SCOPE")
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
        return DEFAULT_XAI_OAUTH_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_XAI_OAUTH_REFRESH_BUFFER_SECONDS


@contextmanager
def _credential_file_lock(lock_path: Path) -> Iterator[None]:
    """Delegate to shared credential_file_lock (fcntl + warning on failure)."""
    with credential_file_lock(lock_path):
        yield

    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def _read_credential_payload(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"xAI OAuth auth file not found at {auth_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"xAI OAuth auth file at {auth_path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("xAI OAuth auth file must contain a JSON object.")
    return payload


def _select_credential_record(
    payload: MutableMapping[str, Any],
    scope: str,
) -> MutableMapping[str, Any]:
    if _looks_like_credential_record(payload):
        return payload

    scoped_record = payload.get(scope)
    if isinstance(scoped_record, dict):
        return scoped_record

    for value in payload.values():
        if isinstance(value, dict) and _looks_like_credential_record(value):
            return value

    raise ValueError(
        "xAI OAuth auth file does not contain a usable credential record. "
        "Expected a scoped record or a flat object with key/access_token."
    )


def _looks_like_credential_record(value: Mapping[str, Any]) -> bool:
    return bool(value.get("key") or value.get("access_token") or value.get("refresh_token"))


def _credential_needs_refresh(
    credential: Mapping[str, Any],
    *,
    buffer_seconds: int,
) -> bool:
    expires_at = _parse_expires_at(credential.get("expires_at"))
    if expires_at is None:
        return False
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


def _refresh_credential_record(
    credential: Mapping[str, Any],
    *,
    token_endpoint: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    http_timeout_seconds: float,
) -> Mapping[str, Any]:
    refresh_token = _clean_oauth_string(credential.get("refresh_token"))
    if refresh_token is None:
        raise ValueError(
            "xAI OAuth credential is expired or near expiry and has no refresh_token."
        )

    resolved_client_id = (
        _clean_oauth_string(client_id)
        or _clean_oauth_string(credential.get("oidc_client_id"))
        or _clean_oauth_string(credential.get("client_id"))
    )
    if resolved_client_id is None:
        raise ValueError("xAI OAuth refresh requires oidc_client_id or client_id.")

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
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            response_body = response.read()
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"xAI OAuth refresh failed with HTTP {exc.code}: {_sanitize_error_message(error_body)}"
        ) from exc
    except urllib_error.URLError as exc:
        raise ValueError(
            f"xAI OAuth refresh failed: {_sanitize_error_message(str(exc.reason))}"
        ) from exc

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("xAI OAuth refresh response was not valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("xAI OAuth refresh response must contain a JSON object.")
    if _clean_oauth_string(payload.get("access_token")) is None:
        raise ValueError("xAI OAuth refresh response did not contain an access_token.")
    return payload


def _update_credential_record(
    credential: MutableMapping[str, Any],
    refreshed: Mapping[str, Any],
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

    expires_in = refreshed.get("expires_in")
    if isinstance(expires_in, (int, float)):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
        credential["expires_at"] = expires_at.isoformat().replace("+00:00", "Z")

    token_type = _clean_oauth_string(refreshed.get("token_type"))
    if token_type is not None:
        credential["token_type"] = token_type


def _write_credential_payload(auth_path: Path, payload: Mapping[str, Any]) -> None:
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = auth_path.with_name(f".{auth_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        try:
            current_mode = auth_path.stat().st_mode & 0o777
            if current_mode & 0o077:
                current_mode = 0o600
        except OSError:
            current_mode = 0o600
        _write_private_file_text(tmp_path, content, mode=current_mode)
        os.replace(tmp_path, auth_path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"Failed to persist refreshed xAI OAuth auth data: {exc}") from exc


def _clean_oauth_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _sanitize_error_message(message: str, *, limit: int = 500) -> str:
    sanitized = str(message)
    for field_name in _SECRET_FIELD_NAMES:
        sanitized = sanitized.replace(field_name, f"{field_name[:3]}***")
    if len(sanitized) > limit:
        sanitized = sanitized[: limit - 3] + "..."
    return sanitized

