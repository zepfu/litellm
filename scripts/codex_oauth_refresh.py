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

from litellm.secret_managers.credential_file_lock import credential_file_lock
from litellm.secret_managers.credential_file_metadata import (
    CredentialFileMetadata,
    apply_credential_file_metadata,
    resolve_credential_file_metadata,
    snapshot_credential_file_metadata,
)
from litellm.secret_managers.credential_file_write import (
    write_and_publish_private_text,
)

from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)

# Portable ~ defaults (expanded via Path.expanduser at use sites).
DEFAULT_CODEX_AUTH_FILE = "~/.codex/auth.json"
DEFAULT_CODEX_LOCK_FILE = "~/.codex/auth.json.lock"
DEFAULT_CODEX_OAUTH_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
DEFAULT_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_CODEX_REFRESH_BUFFER_SECONDS = 300
DEFAULT_CODEX_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_CODEX_AUTH_FILE_MODE = 0o600

# Keep historical module alias; redaction lives in secret_managers.
_SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES


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
    error_hint: Optional[str] = None

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
            "error_hint": self.error_hint,
        }


def inspect_codex_oauth_credential_health(auth_file: str | Path) -> Dict[str, Any]:
    """Read and classify Codex OAuth state without locks, writes, or HTTP."""
    resolved_auth_file = Path(auth_file).expanduser()
    try:
        token_data = _get_token_data(_read_auth_data(resolved_auth_file))
        if _clean_string(token_data.get("access_token")) is None:
            raise ValueError("Codex OAuth credential is missing access_token.")
        expires_at = _get_token_expiry(token_data)
        account_id = _extract_account_id(token_data)
        if expires_at is None:
            return _codex_health_summary(
                resolved_auth_file,
                account_id,
                "degraded",
                error_class="CredentialExpiryUnavailable",
                error_message="Codex OAuth credential expiry is unavailable.",
            )
        expires_at_text = _format_expires_at(expires_at)
        if expires_at <= time.time():
            return _codex_health_summary(
                resolved_auth_file,
                account_id,
                "expired",
                expires_at_text,
                error_class="CredentialExpiredError",
                error_message="Codex OAuth credential is expired.",
            )
        return _codex_health_summary(
            resolved_auth_file, account_id, "fresh", expires_at_text
        )
    except Exception as exc:
        return _codex_health_summary(
            resolved_auth_file,
            None,
            "malformed",
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
        )


def _codex_health_summary(
    auth_file: Path,
    account_id: Optional[str],
    health_status: str,
    expires_at: Optional[str] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "attempted": True,
        "refreshed": False,
        "skipped": False,
        "auth_file": str(auth_file),
        "account_id": account_id,
        "health_status": health_status,
        "expires_at": expires_at,
        "error_class": error_class,
        "error_message": error_message,
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
    if lock_file is not None:
        resolved_lock_file = Path(lock_file).expanduser()
    else:
        resolved_lock_file = resolved_auth_file.with_name(
            f"{resolved_auth_file.name}.lock"
        )
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)

    with _credential_file_lock(resolved_lock_file):
        try:
            auth_data = _read_auth_data(resolved_auth_file)
            _repair_credential_file_metadata(
                resolved_auth_file,
                resolved_lock_file,
            )
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
                error_hint=_extract_oauth_error_hint(exc),
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
    """Delegate to shared credential_file_lock (module-scoped fcntl + warnings)."""
    with credential_file_lock(lock_path):
        yield


def _snapshot_credential_file_metadata(
    auth_path: Path,
) -> CredentialFileMetadata:
    return snapshot_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_CODEX_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _resolve_credential_file_metadata(auth_path: Path) -> CredentialFileMetadata:
    """Resolve ownership/mode for ``auth_path`` via shared helpers.

    Snapshot goes through ``_snapshot_credential_file_metadata`` so tests and
    monkeypatches of the thin local wrapper remain effective.
    """
    return resolve_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_CODEX_AUTH_FILE_MODE,
        mode_env="AAWM_CODEX_AUTH_FILE_MODE",
        uid_env="AAWM_CODEX_AUTH_FILE_UID",
        gid_env="AAWM_CODEX_AUTH_FILE_GID",
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
        default_mode=DEFAULT_CODEX_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _repair_credential_file_metadata(
    auth_path: Path,
    lock_path: Path,
) -> None:
    metadata = _resolve_credential_file_metadata(auth_path)
    _apply_credential_file_metadata(auth_path, metadata)
    if lock_path.exists():
        _apply_credential_file_metadata(lock_path, metadata)


def _read_auth_data(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            auth_data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Codex OAuth auth file not found at {auth_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Codex OAuth auth file at {auth_path} is not valid JSON."
        ) from exc

    if not isinstance(auth_data, dict):
        raise ValueError("Codex OAuth auth file must contain a JSON object.")
    return auth_data


def _get_token_data(auth_data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Return the mutable token container from an auth payload.

    When a ``tokens`` key is present it must be a dict; a non-dict value is a
    schema error (partial write / corruption), not a cue to fall back to the
    top-level object.
    """
    if "tokens" not in auth_data:
        return auth_data
    token_data = auth_data.get("tokens")
    if isinstance(token_data, dict):
        return token_data
    raise ValueError(
        "Codex OAuth auth file field 'tokens' must be a JSON object when present."
    )


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
        return (
            datetime.fromtimestamp(float(expires_at), timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
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
        with urllib_request.urlopen(
            request,
            timeout=http_timeout_seconds,
        ) as response:
            body = response.read()
    except urllib_error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        hint = _extract_oauth_error_hint(error_body)
        message = f"Codex OAuth refresh failed with HTTP {exc.code}"
        if hint:
            message += f": {hint}"
        elif exc.reason:
            message += f": {exc.reason}"
        raise ValueError(message) from exc
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
        raise ValueError(
            "Codex OAuth refresh response did not contain an access_token."
        )
    return refreshed


def _update_token_data(
    token_data: MutableMapping[str, Any],
    refreshed: Mapping[str, Any],
) -> None:
    access_token = _clean_string(refreshed.get("access_token"))
    if access_token is None:
        raise ValueError(
            "Codex OAuth refresh response did not contain an access_token."
        )

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
    try:
        # Shared one-shot path: exclusive private temp, symlink refusal, metadata
        # apply on temp, atomic publish, and failed-temp cleanup. Symlink targets
        # are refused both when resolving metadata and when publishing.
        metadata = _resolve_credential_file_metadata(auth_path)
        payload = json.dumps(auth_data, indent=2) + "\n"
        write_and_publish_private_text(
            auth_path,
            payload,
            metadata=metadata,
            default_mode=DEFAULT_CODEX_AUTH_FILE_MODE,
            mkdir_parents=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to persist refreshed Codex OAuth auth data: {exc}"
        ) from exc


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


def _extract_oauth_error_hint(response_body: Any) -> Optional[str]:
    if isinstance(response_body, BaseException):
        text = str(response_body)
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
    else:
        try:
            payload = json.loads(response_body)
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(payload, dict):
        return None
    for key in ("error", "error_description", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_error_message(value.strip())
    return None


def _sanitize_error_message(message: str, *, limit: int = 500) -> str:
    """Redact secret *values* keyed by known field names (not just the labels)."""
    return sanitize_credential_error_message(message, limit=limit)
