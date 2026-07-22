#!/usr/bin/env python3
"""Refresh a Grok-style OIDC auth JSON file for the provider-status sidecar."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional
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
    CredentialPathIsSymlinkError,
    write_and_publish_private_text,
    write_private_file_text,
)

from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)

DEFAULT_GROK_OIDC_SCOPE = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
DEFAULT_GROK_OIDC_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS = 300
DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_GROK_OIDC_AUTH_FILE_MODE = 0o600
DEFAULT_GROK_OIDC_ERROR_MESSAGE_LIMIT = 500

# Alias keeps existing tests/callers that construct the private name.
_CredentialFileMetadata = CredentialFileMetadata

# Keep historical module alias; redaction lives in secret_managers.
_SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES

# Re-export for tests/callers that need the shared symlink refusal type.
__all__ = (
    "CredentialPathIsSymlinkError",
    "GrokOidcMetadataRepairSummary",
    "GrokOidcRefreshSummary",
    "inspect_grok_oidc_credential_health",
    "refresh_grok_oidc_auth_file",
    "repair_grok_oidc_auth_file_metadata",
)


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


def inspect_grok_oidc_credential_health(
    auth_file: str | Path, *, scope: Optional[str] = None
) -> Dict[str, Any]:
    """Read and classify Grok OIDC state without locks, writes, or HTTP."""
    resolved_auth_file = Path(auth_file).expanduser()
    resolved_scope = _resolve_scope(scope)
    try:
        credential = _select_credential_record(
            _read_credential_payload(resolved_auth_file), resolved_scope
        )
        if not _looks_like_credential_record(credential):
            raise ValueError("Grok OIDC credential has no usable access credential.")
        expires_at = _parse_expires_at(credential.get("expires_at"))
        if expires_at is None:
            return _grok_health_summary(
                resolved_auth_file,
                resolved_scope,
                "degraded",
                error_class="CredentialExpiryUnavailable",
                error_message="Grok OIDC credential expires_at is missing or invalid.",
            )
        if expires_at <= datetime.now(timezone.utc):
            return _grok_health_summary(
                resolved_auth_file,
                resolved_scope,
                "expired",
                expires_at,
                error_class="CredentialExpiredError",
                error_message="Grok OIDC credential is expired.",
            )
        return _grok_health_summary(
            resolved_auth_file, resolved_scope, "fresh", expires_at
        )
    except Exception as exc:
        return _grok_health_summary(
            resolved_auth_file,
            resolved_scope,
            "malformed",
            error_class=exc.__class__.__name__,
            error_message=_sanitize_error_message(str(exc)),
        )


def _grok_health_summary(
    auth_file: Path,
    scope: str,
    health_status: str,
    expires_at: Optional[datetime] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
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
    }


def _write_private_file_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Thin wrapper over shared private write (no umask window, symlink-safe)."""
    write_private_file_text(
        path,
        content,
        mode=mode,
        default_mode=DEFAULT_GROK_OIDC_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


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
    """Delegate to shared credential_file_lock (module-scoped fcntl + warnings)."""
    with credential_file_lock(lock_path):
        yield


def _snapshot_credential_file_metadata(
    credential_path: Path,
) -> CredentialFileMetadata:
    """Delegate to shared snapshot helper (RR-075 residual #3)."""
    return snapshot_credential_file_metadata(
        credential_path,
        default_mode=DEFAULT_GROK_OIDC_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _resolve_credential_file_metadata(credential_path: Path) -> CredentialFileMetadata:
    """Delegate to shared resolve helper with Grok env names.

    Snapshot goes through ``_snapshot_credential_file_metadata`` so tests and
    operators can monkeypatch ownership without forking the shared module.
    Symlink targets are refused during snapshot/resolve.
    """
    return resolve_credential_file_metadata(
        credential_path,
        default_mode=DEFAULT_GROK_OIDC_AUTH_FILE_MODE,
        mode_env="AAWM_GROK_OIDC_AUTH_FILE_MODE",
        uid_env="AAWM_GROK_OIDC_AUTH_FILE_UID",
        gid_env="AAWM_GROK_OIDC_AUTH_FILE_GID",
        base_metadata=_snapshot_credential_file_metadata(credential_path),
        refuse_symlink=True,
    )


def _apply_credential_file_metadata(
    target_path: Path,
    metadata: CredentialFileMetadata,
) -> None:
    """Delegate to shared apply helper (mode clamp + ownership safety)."""
    apply_credential_file_metadata(
        target_path,
        metadata,
        default_mode=DEFAULT_GROK_OIDC_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


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

    raise ValueError("Grok OIDC auth file does not contain a usable credential record.")


def _looks_like_credential_record(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("key") or value.get("access_token") or value.get("refresh_token")
    )


def _credential_needs_refresh(
    credential: Mapping[str, Any],
    *,
    buffer_seconds: int,
) -> bool:
    """Return True when the credential should not be used as-is.

    Missing or unparseable ``expires_at`` fails safe toward refresh (not
    permanently fresh).
    """
    expires_at = _parse_expires_at(credential.get("expires_at"))
    if expires_at is None:
        return True
    return datetime.now(timezone.utc) >= expires_at - timedelta(seconds=buffer_seconds)


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
    if (
        not isinstance(resolved_token_endpoint, str)
        or not resolved_token_endpoint.strip()
    ):
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
        error_hint = _extract_oauth_error_hint(
            exc.read().decode("utf-8", errors="replace")
        )
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
        raise ValueError("Grok OIDC token endpoint returned invalid JSON.") from exc

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


def _write_credential_payload(
    credential_path: Path, payload: Mapping[str, Any]
) -> None:
    """Publish credential JSON via shared exclusive temp + atomic replace.

    Uses ``write_and_publish_private_text`` so temp names are not pid-only,
    symlink targets are refused, and failed temps are cleaned up consistently.
    """
    metadata = _resolve_credential_file_metadata(credential_path)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_and_publish_private_text(
        credential_path,
        content,
        metadata=metadata,
        default_mode=DEFAULT_GROK_OIDC_AUTH_FILE_MODE,
        mkdir_parents=True,
    )


def _sanitize_error_message(
    message: str, *, limit: int = DEFAULT_GROK_OIDC_ERROR_MESSAGE_LIMIT
) -> str:
    """Redact secret *values* keyed by known field names (not just the labels)."""
    return sanitize_credential_error_message(message, limit=limit)
