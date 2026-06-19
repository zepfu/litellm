#!/usr/bin/env python3
"""Refresh a Codex/ChatGPT OAuth auth JSON file for the provider-status sidecar."""

from __future__ import annotations

import base64
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_CODEX_AUTH_FILE = "/home/zepfu/.codex/auth.json"
DEFAULT_CODEX_LOCK_FILE = "/home/zepfu/.codex/auth.json.lock"
DEFAULT_CODEX_OAUTH_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
DEFAULT_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_CODEX_REFRESH_BUFFER_SECONDS = 300
DEFAULT_CODEX_HTTP_TIMEOUT_SECONDS = 30.0

_SECRET_FIELD_NAMES = {
    "access_token",
    "client_secret",
    "id_token",
    "key",
    "refresh_token",
}


@dataclass(frozen=True)
class CodexOAuthRefreshSummary:
    attempted: bool
    refreshed: bool
    skipped: bool
    auth_file: str
    account_id: Optional[str] = None
    expires_at: Optional[str] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "refreshed": self.refreshed,
            "skipped": self.skipped,
            "auth_file": self.auth_file,
            "account_id": self.account_id,
            "expires_at": self.expires_at,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


def refresh_codex_oauth_auth_file(
    auth_file: str | Path,
    *,
    buffer_seconds: Optional[int] = None,
    force: bool = False,
    lock_file: str | Path | None = None,
    token_endpoint: Optional[str] = None,
    client_id: Optional[str] = None,
    http_timeout_seconds: float = DEFAULT_CODEX_HTTP_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Refresh a Codex OAuth auth file when it is near expiry or forced."""

    resolved_auth_file = Path(auth_file).expanduser()
    resolved_lock_file = (
        Path(lock_file).expanduser()
        if lock_file is not None
        else resolved_auth_file.with_name(f"{resolved_auth_file.name}.lock")
    )
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)

    with _credential_file_lock(resolved_lock_file):
        try:
            auth_data = _read_auth_data(resolved_auth_file)
            token_data = _get_token_data(auth_data)
            current_expires_at = _format_expires_at(_get_token_expiry(token_data))
            current_account_id = _extract_account_id(token_data)

            if not force and not _token_needs_refresh(
                token_data,
                buffer_seconds=resolved_buffer_seconds,
            ):
                return CodexOAuthRefreshSummary(
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    auth_file=str(resolved_auth_file),
                    account_id=current_account_id,
                    expires_at=current_expires_at,
                ).as_dict()

            refreshed = _refresh_token_data(
                token_data,
                token_endpoint=token_endpoint,
                client_id=client_id,
                http_timeout_seconds=http_timeout_seconds,
            )
            _update_token_data(token_data, refreshed)
            auth_data["last_refresh"] = datetime.now(timezone.utc).isoformat()
            _write_auth_data(resolved_auth_file, auth_data)
            return CodexOAuthRefreshSummary(
                attempted=True,
                refreshed=True,
                skipped=False,
                auth_file=str(resolved_auth_file),
                account_id=_extract_account_id(token_data),
                expires_at=_format_expires_at(_get_token_expiry(token_data)),
            ).as_dict()
        except Exception as exc:
            return CodexOAuthRefreshSummary(
                attempted=True,
                refreshed=False,
                skipped=False,
                auth_file=str(resolved_auth_file),
                error_class=exc.__class__.__name__,
                error_message=_sanitize_error_message(str(exc)),
            ).as_dict()


def _resolve_buffer_seconds(buffer_seconds: Optional[int]) -> int:
    if buffer_seconds is not None:
        return max(0, int(buffer_seconds))
    raw_value = os.getenv("AAWM_CODEX_OAUTH_REFRESH_BUFFER_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CODEX_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_CODEX_REFRESH_BUFFER_SECONDS


@contextmanager
def _credential_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_auth_data(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            auth_data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Codex OAuth auth file not found at {auth_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Codex OAuth auth file at {auth_path} is not valid JSON.") from exc

    if not isinstance(auth_data, dict):
        raise ValueError("Codex OAuth auth file must contain a JSON object.")
    return auth_data


def _get_token_data(auth_data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    token_data = auth_data.get("tokens")
    if isinstance(token_data, dict):
        return token_data
    return auth_data


def _decode_jwt_claims_without_validation(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}


def _get_token_expiry(token_data: Mapping[str, Any]) -> Optional[float]:
    expires_at = token_data.get("expires_at")
    if isinstance(expires_at, (int, float)):
        return float(expires_at)
    if isinstance(expires_at, str) and expires_at.strip():
        try:
            return float(expires_at.strip())
        except ValueError:
            pass

    access_token = _clean_string(token_data.get("access_token"))
    if access_token is None:
        return None
    exp = _decode_jwt_claims_without_validation(access_token).get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


def _format_expires_at(expires_at: Optional[float]) -> Optional[str]:
    if expires_at is None:
        return None
    try:
        return datetime.fromtimestamp(float(expires_at), timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        )
    except (OSError, OverflowError, ValueError):
        return None


def _token_needs_refresh(
    token_data: Mapping[str, Any],
    *,
    buffer_seconds: int,
) -> bool:
    access_token = _clean_string(token_data.get("access_token"))
    if access_token is None:
        return True
    expires_at = _get_token_expiry(token_data)
    if expires_at is None:
        return False
    return time.time() >= expires_at - max(0, buffer_seconds)


def _refresh_token_data(
    token_data: Mapping[str, Any],
    *,
    token_endpoint: Optional[str],
    client_id: Optional[str],
    http_timeout_seconds: float,
) -> Mapping[str, Any]:
    refresh_token = _clean_string(token_data.get("refresh_token"))
    if refresh_token is None:
        raise ValueError(
            "Codex OAuth access token is expired and the auth file does not "
            "contain a refresh_token."
        )

    payload = {
        "client_id": client_id or DEFAULT_CODEX_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": "openid profile email",
    }
    request = urllib_request.Request(
        token_endpoint or DEFAULT_CODEX_OAUTH_TOKEN_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            body = response.read()
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Codex OAuth refresh failed with HTTP {exc.code}: {_sanitize_error_message(body)}"
        ) from exc
    except urllib_error.URLError as exc:
        raise ValueError(
            f"Codex OAuth refresh failed: {_sanitize_error_message(str(exc.reason))}"
        ) from exc

    try:
        refreshed = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Codex OAuth refresh response was not valid JSON.") from exc
    if not isinstance(refreshed, Mapping):
        raise ValueError("Codex OAuth refresh response must contain a JSON object.")
    if _clean_string(refreshed.get("access_token")) is None:
        raise ValueError("Codex OAuth refresh response did not contain an access_token.")
    return refreshed


def _update_token_data(
    token_data: MutableMapping[str, Any],
    refreshed: Mapping[str, Any],
) -> None:
    access_token = _clean_string(refreshed.get("access_token"))
    if access_token is None:
        raise ValueError("Codex OAuth refresh response did not contain an access_token.")

    refresh_token = _clean_string(refreshed.get("refresh_token")) or _clean_string(
        token_data.get("refresh_token")
    )
    token_data["access_token"] = access_token
    if refresh_token is not None:
        token_data["refresh_token"] = refresh_token

    id_token = _clean_string(refreshed.get("id_token"))
    if id_token is not None:
        token_data["id_token"] = id_token

    expires_at = _get_token_expiry({"access_token": access_token})
    expires_in = refreshed.get("expires_in")
    if expires_at is None and isinstance(expires_in, (int, float)):
        expires_at = time.time() + float(expires_in)
    if expires_at is not None:
        token_data["expires_at"] = int(expires_at)

    account_id = _extract_account_id_from_tokens(
        id_token=_clean_string(token_data.get("id_token")),
        access_token=access_token,
    ) or _clean_string(token_data.get("account_id"))
    if account_id is not None:
        token_data["account_id"] = account_id


def _write_auth_data(auth_path: Path, auth_data: Mapping[str, Any]) -> None:
    tmp_path = auth_path.with_name(f".{auth_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        payload = json.dumps(auth_data, indent=2) + "\n"
        tmp_path.write_text(payload, encoding="utf-8")
        try:
            current_mode = auth_path.stat().st_mode & 0o777
            os.chmod(tmp_path, current_mode)
        except OSError:
            pass
        os.replace(tmp_path, auth_path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"Failed to persist refreshed Codex OAuth auth data: {exc}") from exc


def _extract_account_id(token_data: Mapping[str, Any]) -> Optional[str]:
    account_id = _clean_string(token_data.get("account_id"))
    if account_id is not None:
        return account_id
    return _extract_account_id_from_tokens(
        id_token=_clean_string(token_data.get("id_token")),
        access_token=_clean_string(token_data.get("access_token")),
    )


def _extract_account_id_from_tokens(
    *,
    id_token: Optional[str],
    access_token: Optional[str],
) -> Optional[str]:
    for key in ("id_token", "access_token"):
        token = id_token if key == "id_token" else access_token
        if token is None:
            continue
        claims = _decode_jwt_claims_without_validation(token)
        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            account_id = _clean_string(auth_claims.get("chatgpt_account_id"))
            if account_id is not None:
                return account_id
    return None


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _sanitize_error_message(message: str, *, limit: int = 500) -> str:
    sanitized = str(message)
    for field_name in _SECRET_FIELD_NAMES:
        sanitized = sanitized.replace(field_name, f"{field_name[:3]}***")
    if len(sanitized) > limit:
        sanitized = sanitized[: limit - 3] + "..."
    return sanitized
