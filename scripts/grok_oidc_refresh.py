#!/usr/bin/env python3
"""Refresh a Grok-style OIDC auth JSON file for the provider-status sidecar."""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

DEFAULT_GROK_OIDC_SCOPE = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_GROK_OIDC_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS = 300
DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS = 30.0

_SECRET_FIELD_NAMES = {
    "access_token",
    "client_secret",
    "id_token",
    "key",
    "refresh_token",
}


@dataclass(frozen=True)
class GrokOidcRefreshSummary:
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


@dataclass(frozen=True)
class GrokOidcMetadataRepairSummary:
    attempted: bool
    repaired: bool
    auth_file: str
    error_class: Optional[str] = None
    error_message: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "repaired": self.repaired,
            "auth_file": self.auth_file,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


def refresh_grok_oidc_auth_file(
    auth_file: str | Path,
    *,
    scope: Optional[str] = None,
    buffer_seconds: Optional[int] = None,
    force: bool = False,
    lock_file: str | Path | None = None,
    token_endpoint: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    http_timeout_seconds: float = DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Refresh a Grok-style OIDC auth file when it is near expiry or forced."""

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
            current_expires_at = _format_expires_at(credential.get("expires_at"))

            if not force and not _credential_needs_refresh(
                credential,
                buffer_seconds=resolved_buffer_seconds,
            ):
                return GrokOidcRefreshSummary(
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    auth_file=str(resolved_auth_file),
                    scope=resolved_scope,
                    expires_at=current_expires_at,
                ).as_dict()

            refreshed_payload = _refresh_credential_record(
                credential,
                token_endpoint=token_endpoint,
                client_id=client_id,
                client_secret=client_secret,
                http_timeout_seconds=http_timeout_seconds,
            )
            _update_credential_record(credential, refreshed_payload)
            _write_credential_payload(resolved_auth_file, raw_payload)
            return GrokOidcRefreshSummary(
                attempted=True,
                refreshed=True,
                skipped=False,
                auth_file=str(resolved_auth_file),
                scope=resolved_scope,
                expires_at=_format_expires_at(credential.get("expires_at")),
            ).as_dict()
        except Exception as exc:
            return GrokOidcRefreshSummary(
                attempted=True,
                refreshed=False,
                skipped=False,
                auth_file=str(resolved_auth_file),
                scope=resolved_scope,
                error_class=exc.__class__.__name__,
                error_message=_sanitize_error_message(str(exc)),
            ).as_dict()


def repair_grok_oidc_auth_file_metadata(
    auth_file: str | Path,
    *,
    lock_file: str | Path | None = None,
) -> Dict[str, Any]:
    """Repair ownership/mode on the Grok auth file without touching tokens."""

    resolved_auth_file = Path(auth_file).expanduser()
    resolved_lock_file = (
        Path(lock_file).expanduser()
        if lock_file is not None
        else resolved_auth_file.with_name(f"{resolved_auth_file.name}.lock")
    )

    with _credential_file_lock(resolved_lock_file):
        try:
            before = _snapshot_credential_file_metadata(resolved_auth_file)
            target_metadata = _resolve_credential_file_metadata(resolved_auth_file)
            repaired = before != target_metadata
            if repaired:
                _apply_credential_file_metadata(resolved_auth_file, target_metadata)
            return GrokOidcMetadataRepairSummary(
                attempted=True,
                repaired=repaired,
                auth_file=str(resolved_auth_file),
            ).as_dict()
        except Exception as exc:
            return GrokOidcMetadataRepairSummary(
                attempted=True,
                repaired=False,
                auth_file=str(resolved_auth_file),
                error_class=exc.__class__.__name__,
                error_message=_sanitize_error_message(str(exc)),
            ).as_dict()


def _resolve_scope(scope: Optional[str]) -> str:
    if isinstance(scope, str) and scope.strip():
        return scope.strip()
    env_scope = os.getenv("LITELLM_XAI_GROK_OAUTH_SCOPE") or os.getenv(
        "LITELLM_XAI_OAUTH_SCOPE"
    )
    if isinstance(env_scope, str) and env_scope.strip():
        return env_scope.strip()
    return DEFAULT_GROK_OIDC_SCOPE


def _resolve_buffer_seconds(buffer_seconds: Optional[int]) -> int:
    if buffer_seconds is not None:
        return max(0, int(buffer_seconds))
    env_value = os.getenv("LITELLM_XAI_OAUTH_REFRESH_BUFFER_SECONDS")
    if env_value is None or not env_value.strip():
        return DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(env_value))
    except ValueError:
        return DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS


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


def _read_credential_payload(credential_path: Path) -> Dict[str, Any]:
    try:
        with credential_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Grok OIDC auth file not found at {credential_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Grok OIDC auth file at {credential_path} is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Grok OIDC auth file must contain a JSON object.")
    return payload


def _select_credential_record(payload: Mapping[str, Any], scope: str) -> Dict[str, Any]:
    if isinstance(payload, dict) and _looks_like_credential_record(payload):
        return payload

    scoped_record = payload.get(scope)
    if isinstance(scoped_record, dict):
        return scoped_record

    for value in payload.values():
        if isinstance(value, dict) and _looks_like_credential_record(value):
            return value

    raise ValueError(
        "Grok OIDC auth file does not contain a usable credential record."
    )


def _looks_like_credential_record(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("key")
        or value.get("access_token")
        or value.get("refresh_token")
    )


def _credential_needs_refresh(
    credential: Mapping[str, Any],
    *,
    buffer_seconds: int,
) -> bool:
    expires_at = _parse_expires_at(credential.get("expires_at"))
    if expires_at is None:
        return False
    return datetime.now(timezone.utc) >= expires_at - timedelta(
        seconds=buffer_seconds
    )


def _parse_expires_at(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
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


def _format_expires_at(value: Any) -> Optional[str]:
    parsed = _parse_expires_at(value)
    if parsed is None:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def _refresh_credential_record(
    credential: MutableMapping[str, Any],
    *,
    token_endpoint: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    http_timeout_seconds: float,
) -> Dict[str, Any]:
    refresh_token = credential.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise ValueError(
            "Grok OIDC credential is expired or near expiry and has no refresh_token."
        )

    resolved_client_id = (
        client_id
        or os.getenv("LITELLM_XAI_OAUTH_CLIENT_ID")
        or credential.get("oidc_client_id")
        or credential.get("client_id")
    )
    if not isinstance(resolved_client_id, str) or not resolved_client_id.strip():
        raise ValueError(
            "Grok OIDC credential refresh requires oidc_client_id or client_id."
        )

    resolved_token_endpoint = (
        token_endpoint
        or os.getenv("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT")
        or credential.get("token_endpoint")
        or DEFAULT_GROK_OIDC_TOKEN_ENDPOINT
    )
    if not isinstance(resolved_token_endpoint, str) or not resolved_token_endpoint.strip():
        raise ValueError("Grok OIDC token endpoint is missing.")

    form_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token.strip(),
        "client_id": resolved_client_id.strip(),
    }
    resolved_client_secret = (
        client_secret
        or os.getenv("LITELLM_XAI_OAUTH_CLIENT_SECRET")
        or credential.get("client_secret")
    )
    if isinstance(resolved_client_secret, str) and resolved_client_secret.strip():
        form_data["client_secret"] = resolved_client_secret.strip()

    encoded_body = urllib_parse.urlencode(form_data).encode("utf-8")
    request = urllib_request.Request(
        resolved_token_endpoint.strip(),
        data=encoded_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_hint = _extract_oauth_error_hint(exc.read().decode("utf-8", errors="replace"))
        raise ValueError(
            "Grok OIDC credential refresh failed"
            + (f" ({error_hint})" if error_hint else "")
            + "."
        ) from exc
    except urllib_error.URLError as exc:
        raise ValueError(
            "Grok OIDC credential refresh failed while contacting the token endpoint."
        ) from exc

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Grok OIDC token endpoint returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Grok OIDC token endpoint returned a non-object payload.")
    return payload


def _extract_oauth_error_hint(response_body: str) -> Optional[str]:
    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("error", "error_description", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _sanitize_error_message(value.strip())
    return None


def _update_credential_record(
    credential: MutableMapping[str, Any],
    refreshed: Mapping[str, Any],
) -> None:
    access_token = refreshed.get("access_token")
    if isinstance(access_token, str) and access_token.strip():
        credential["key"] = access_token.strip()
        credential["access_token"] = access_token.strip()

    refresh_token = refreshed.get("refresh_token")
    if isinstance(refresh_token, str) and refresh_token.strip():
        credential["refresh_token"] = refresh_token.strip()

    id_token = refreshed.get("id_token")
    if isinstance(id_token, str) and id_token.strip():
        credential["id_token"] = id_token.strip()

    expires_in = refreshed.get("expires_in")
    if isinstance(expires_in, (int, float)):
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
        credential["expires_at"] = expires_at.isoformat().replace("+00:00", "Z")

    token_type = refreshed.get("token_type")
    if isinstance(token_type, str) and token_type.strip():
        credential["token_type"] = token_type.strip()


DEFAULT_GROK_OIDC_AUTH_FILE_MODE = 0o600


@dataclass(frozen=True)
class _CredentialFileMetadata:
    uid: Optional[int]
    gid: Optional[int]
    mode: int


def _parse_optional_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(str(value).strip(), 0)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _resolve_credential_file_mode_override() -> Optional[int]:
    mode = _parse_optional_positive_int(
        os.getenv("AAWM_GROK_OIDC_AUTH_FILE_MODE")
    )
    if mode is None:
        return None
    # Keep credential files user-private; never widen beyond the default.
    mode = mode & 0o777
    if mode & 0o077:
        return DEFAULT_GROK_OIDC_AUTH_FILE_MODE
    return mode


def _resolve_credential_file_metadata(credential_path: Path) -> _CredentialFileMetadata:
    metadata = _snapshot_credential_file_metadata(credential_path)
    uid_override = _parse_optional_positive_int(
        os.getenv("AAWM_GROK_OIDC_AUTH_FILE_UID")
    )
    gid_override = _parse_optional_positive_int(
        os.getenv("AAWM_GROK_OIDC_AUTH_FILE_GID")
    )
    mode_override = _resolve_credential_file_mode_override()
    mode = mode_override if mode_override is not None else metadata.mode
    if mode & 0o077:
        mode = DEFAULT_GROK_OIDC_AUTH_FILE_MODE
    return _CredentialFileMetadata(
        uid=uid_override if uid_override is not None else metadata.uid,
        gid=gid_override if gid_override is not None else metadata.gid,
        mode=mode,
    )


def _snapshot_credential_file_metadata(credential_path: Path) -> _CredentialFileMetadata:
    if not credential_path.exists():
        return _CredentialFileMetadata(
            uid=None,
            gid=None,
            mode=DEFAULT_GROK_OIDC_AUTH_FILE_MODE,
        )
    file_stat = credential_path.stat()
    return _CredentialFileMetadata(
        uid=file_stat.st_uid,
        gid=file_stat.st_gid,
        mode=stat.S_IMODE(file_stat.st_mode),
    )


def _apply_credential_file_metadata(
    target_path: Path,
    metadata: _CredentialFileMetadata,
) -> None:
    if metadata.uid is not None or metadata.gid is not None:
        os.chown(
            target_path,
            metadata.uid if metadata.uid is not None else -1,
            metadata.gid if metadata.gid is not None else -1,
        )
    os.chmod(target_path, metadata.mode)


def _write_credential_payload(credential_path: Path, payload: Mapping[str, Any]) -> None:
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _resolve_credential_file_metadata(credential_path)
    tmp_path = credential_path.with_name(
        f".{credential_path.name}.{os.getpid()}.tmp"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _apply_credential_file_metadata(tmp_path, metadata)
        os.replace(tmp_path, credential_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _sanitize_error_message(message: str) -> str:
    sanitized = message
    for field_name in _SECRET_FIELD_NAMES:
        sanitized = re.sub(
            rf"(?i)\b{re.escape(field_name)}\b\s*[:=]\s*\S+",
            f"{field_name}=[REDACTED]",
            sanitized,
        )
    return sanitized
