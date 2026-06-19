#!/usr/bin/env python3
"""Refresh an Antigravity OAuth token file for the provider-status sidecar."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Optional
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

DEFAULT_ANTIGRAVITY_AUTH_FILE = (
    "/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token"
)
DEFAULT_ANTIGRAVITY_LOCK_FILE = (
    "/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token.lock"
)
DEFAULT_ANTIGRAVITY_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
DEFAULT_ANTIGRAVITY_REFRESH_BUFFER_SECONDS = 300
DEFAULT_ANTIGRAVITY_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_ANTIGRAVITY_CLI_REFRESH_TIMEOUT_SECONDS = 30.0
DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE = 0o600

_DEFAULT_CLI_BINARY_PATHS = (
    "/home/zepfu/.local/bin/agy",
    "~/.local/bin/agy",
)
_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN = re.compile(
    rb"(?P<value>\d{6,}-[A-Za-z0-9_.-]+\.apps\.googleusercontent\.com)"
)
_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN = re.compile(
    rb"(?P<value>GOCSPX-[A-Za-z0-9_-]+)"
)
_SECRET_FIELD_NAMES = {
    "access_token",
    "client_secret",
    "id_token",
    "key",
    "refresh_token",
}


@dataclass(frozen=True)
class AntigravityRefreshSummary:
    attempted: bool
    refreshed: bool
    skipped: bool
    auth_file: str
    expires_at: Optional[str] = None
    refresh_method: Optional[str] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "refreshed": self.refreshed,
            "skipped": self.skipped,
            "auth_file": self.auth_file,
            "expires_at": self.expires_at,
            "refresh_method": self.refresh_method,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


def refresh_antigravity_oauth_token_file(
    auth_file: str | Path,
    *,
    buffer_seconds: Optional[int] = None,
    force: bool = False,
    lock_file: str | Path | None = None,
    token_endpoint: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    cli_path: Optional[str | Path] = None,
    http_timeout_seconds: float = DEFAULT_ANTIGRAVITY_HTTP_TIMEOUT_SECONDS,
    cli_timeout_seconds: float = DEFAULT_ANTIGRAVITY_CLI_REFRESH_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Refresh an Antigravity OAuth token file when near expiry or forced."""

    resolved_auth_file = Path(auth_file).expanduser()
    resolved_lock_file = (
        Path(lock_file).expanduser()
        if lock_file is not None
        else resolved_auth_file.with_name(f"{resolved_auth_file.name}.lock")
    )
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)

    with _credential_file_lock(resolved_lock_file):
        try:
            token_data = _read_token_data(resolved_auth_file)
            current_expires_at = _format_expires_at(_get_token_expiry(token_data))
            if not force and _token_is_valid(
                token_data,
                buffer_seconds=resolved_buffer_seconds,
            ):
                return AntigravityRefreshSummary(
                    attempted=False,
                    refreshed=False,
                    skipped=True,
                    auth_file=str(resolved_auth_file),
                    expires_at=current_expires_at,
                ).as_dict()

            try:
                refreshed_token_data = _refresh_token_data_direct(
                    token_data,
                    token_endpoint=token_endpoint,
                    client_id=client_id,
                    client_secret=client_secret,
                    cli_path=cli_path,
                    http_timeout_seconds=http_timeout_seconds,
                )
                _write_token_data(resolved_auth_file, refreshed_token_data)
                return AntigravityRefreshSummary(
                    attempted=True,
                    refreshed=True,
                    skipped=False,
                    auth_file=str(resolved_auth_file),
                    expires_at=_format_expires_at(
                        _get_token_expiry(refreshed_token_data)
                    ),
                    refresh_method="direct",
                ).as_dict()
            except _DirectRefreshNeedsCliFallback:
                refreshed_token_data = _refresh_token_data_via_cli(
                    resolved_auth_file,
                    original_token_data=token_data,
                    cli_path=cli_path,
                    timeout_seconds=cli_timeout_seconds,
                )
                return AntigravityRefreshSummary(
                    attempted=True,
                    refreshed=True,
                    skipped=False,
                    auth_file=str(resolved_auth_file),
                    expires_at=_format_expires_at(
                        _get_token_expiry(refreshed_token_data)
                    ),
                    refresh_method="agy_cli",
                ).as_dict()
        except Exception as exc:
            return AntigravityRefreshSummary(
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
    raw_value = os.getenv("AAWM_ANTIGRAVITY_REFRESH_BUFFER_SECONDS")
    if raw_value is None or not raw_value.strip():
        return DEFAULT_ANTIGRAVITY_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return DEFAULT_ANTIGRAVITY_REFRESH_BUFFER_SECONDS


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


def _read_token_data(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            token_data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Antigravity OAuth token file not found at {auth_path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Antigravity OAuth token file at {auth_path} is not valid JSON."
        ) from exc

    if not isinstance(token_data, dict):
        raise ValueError("Antigravity OAuth token file must contain a JSON object.")
    return token_data


def _get_token_block(token_data: Mapping[str, Any]) -> Mapping[str, Any]:
    token_block = token_data.get("token")
    if not isinstance(token_block, Mapping):
        raise ValueError("Antigravity OAuth token data does not contain a token object.")
    return token_block


def _get_token_expiry(token_data: Mapping[str, Any]) -> Optional[datetime]:
    token_block = token_data.get("token")
    if not isinstance(token_block, Mapping):
        return None
    return _parse_expiry(token_block.get("expiry"))


def _parse_expiry(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
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


def _format_expires_at(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _token_is_valid(
    token_data: Mapping[str, Any],
    *,
    buffer_seconds: int,
) -> bool:
    token_block = token_data.get("token")
    if not isinstance(token_block, Mapping):
        return False
    access_token = _clean_value(token_block.get("access_token"))
    if access_token is None:
        return False
    expiry = _parse_expiry(token_block.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc) + timedelta(seconds=buffer_seconds)


def _token_is_unexpired(token_data: Mapping[str, Any]) -> bool:
    token_block = token_data.get("token")
    if not isinstance(token_block, Mapping):
        return False
    access_token = _clean_value(token_block.get("access_token"))
    if access_token is None:
        return False
    expiry = _parse_expiry(token_block.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc)


def _refresh_token_data_direct(
    token_data: Mapping[str, Any],
    *,
    token_endpoint: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    cli_path: Optional[str | Path],
    http_timeout_seconds: float,
) -> Dict[str, Any]:
    token_block = _get_token_block(token_data)
    refresh_token = _clean_value(token_block.get("refresh_token"))
    if refresh_token is None:
        raise ValueError(
            "Antigravity OAuth token data does not contain a refresh_token."
        )

    client_candidates = _get_client_value_candidates(
        token_data,
        client_id=client_id,
        client_secret=client_secret,
        cli_path=cli_path,
    )
    if not client_candidates:
        raise ValueError(
            "Antigravity OAuth refresh requires client_id/client_secret values."
        )

    last_error: Optional[Exception] = None
    last_oauth_error: Optional[str] = None
    for candidate_client_id, candidate_client_secret in client_candidates:
        try:
            refreshed = _post_refresh_request(
                token_endpoint=token_endpoint,
                client_id=candidate_client_id,
                client_secret=candidate_client_secret,
                refresh_token=refresh_token,
                http_timeout_seconds=http_timeout_seconds,
            )
        except _RefreshHttpError as exc:
            last_error = exc
            last_oauth_error = exc.oauth_error
            if exc.oauth_error in {
                "invalid_client",
                "invalid_grant",
                "unauthorized_client",
            }:
                continue
            raise ValueError(
                f"Failed to refresh Antigravity OAuth access token ({exc})."
            ) from exc
        return _apply_refresh_payload(token_data, refreshed, refresh_token)

    if last_oauth_error in {"invalid_client", "invalid_grant", "unauthorized_client"}:
        raise _DirectRefreshNeedsCliFallback() from last_error
    if last_error is not None:
        raise ValueError(
            f"Failed to refresh Antigravity OAuth access token ({last_error})."
        ) from last_error
    raise ValueError("Failed to refresh Antigravity OAuth access token.")


class _DirectRefreshNeedsCliFallback(ValueError):
    pass


class _RefreshHttpError(ValueError):
    def __init__(self, message: str, *, oauth_error: Optional[str] = None) -> None:
        super().__init__(message)
        self.oauth_error = oauth_error


def _post_refresh_request(
    *,
    token_endpoint: Optional[str],
    client_id: str,
    client_secret: str,
    refresh_token: str,
    http_timeout_seconds: float,
) -> Dict[str, Any]:
    form_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    encoded_body = urllib_parse.urlencode(form_data).encode("utf-8")
    request = urllib_request.Request(
        token_endpoint or DEFAULT_ANTIGRAVITY_TOKEN_ENDPOINT,
        data=encoded_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=http_timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        oauth_error = _extract_oauth_error_hint(body)
        raise _RefreshHttpError(
            f"status={exc.code}" + (f", error={oauth_error}" if oauth_error else ""),
            oauth_error=oauth_error,
        ) from exc
    except urllib_error.URLError as exc:
        raise ValueError(
            "Antigravity OAuth refresh failed while contacting the token endpoint."
        ) from exc

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError("Antigravity OAuth token endpoint returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Antigravity OAuth token endpoint returned a non-object payload.")
    return payload


def _apply_refresh_payload(
    token_data: Mapping[str, Any],
    refreshed: Mapping[str, Any],
    refresh_token: str,
) -> Dict[str, Any]:
    access_token = _clean_value(refreshed.get("access_token"))
    if access_token is None:
        raise ValueError(
            "Antigravity OAuth refresh response did not contain an access_token."
        )
    expires_in = refreshed.get("expires_in")
    if not isinstance(expires_in, (int, float)):
        raise ValueError(
            "Antigravity OAuth refresh response did not contain expires_in."
        )

    updated_token_block: MutableMapping[str, Any] = dict(_get_token_block(token_data))
    updated_token_block.update(refreshed)
    updated_token_block["access_token"] = access_token
    updated_token_block["refresh_token"] = refresh_token
    updated_token_block["expiry"] = (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    ).isoformat()

    updated_token_data = dict(token_data)
    updated_token_data["token"] = dict(updated_token_block)
    return updated_token_data


def _refresh_token_data_via_cli(
    auth_path: Path,
    *,
    original_token_data: Mapping[str, Any],
    cli_path: Optional[str | Path],
    timeout_seconds: float,
) -> Dict[str, Any]:
    refresh_home = _get_cli_refresh_home(auth_path)
    cli_binary = _first_cli_binary(cli_path)
    if refresh_home is None or cli_binary is None:
        raise ValueError(
            "Antigravity OAuth direct refresh failed and AGY CLI silent refresh "
            "is unavailable for this auth-file path."
        )

    log_path = Path(os.getenv("TMPDIR") or "/tmp") / (
        f"litellm-antigravity-refresh-{os.getpid()}-{time.monotonic_ns()}.log"
    )
    env = dict(os.environ)
    env["HOME"] = str(refresh_home)
    try:
        process = subprocess.run(
            [str(cli_binary), "--log-file", str(log_path), "models"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=max(timeout_seconds, 1.0),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("AGY CLI silent auth refresh timed out.") from exc
    except OSError as exc:
        raise ValueError(
            f"AGY CLI silent auth refresh failed ({type(exc).__name__})."
        ) from exc
    finally:
        try:
            log_path.unlink()
        except OSError:
            pass

    if process.returncode != 0:
        raise ValueError(
            "AGY CLI silent auth refresh failed. Re-authenticate Antigravity CLI "
            "before using Antigravity passthrough."
        )

    refreshed_token_data = _read_token_data(auth_path)
    if not _token_is_valid(refreshed_token_data, buffer_seconds=60):
        if (
            refreshed_token_data == original_token_data
            and _token_is_unexpired(refreshed_token_data)
        ):
            return refreshed_token_data
        raise ValueError("AGY CLI silent auth refresh did not produce a valid token.")
    return refreshed_token_data


def _get_cli_refresh_home(auth_path: Path) -> Optional[Path]:
    parts = auth_path.expanduser().parts
    if len(parts) < 4:
        return None
    if parts[-3:] != (
        ".gemini",
        "antigravity-cli",
        "antigravity-oauth-token",
    ):
        return None
    return auth_path.expanduser().parents[2]


def _get_client_value_candidates(
    token_data: Mapping[str, Any],
    *,
    client_id: Optional[str],
    client_secret: Optional[str],
    cli_path: Optional[str | Path],
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    _add_client_candidate(
        candidates,
        seen,
        client_id or _first_env("AAWM_ANTIGRAVITY_OAUTH_CLIENT_ID", "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_ID", "ANTIGRAVITY_OAUTH_CLIENT_ID"),
        client_secret or _first_env("AAWM_ANTIGRAVITY_OAUTH_CLIENT_SECRET", "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_SECRET", "ANTIGRAVITY_OAUTH_CLIENT_SECRET"),
    )
    _add_client_candidate(
        candidates,
        seen,
        _first_mapping_value(token_data, ("client_id", "clientId")),
        _first_mapping_value(token_data, ("client_secret", "clientSecret")),
    )
    for cli_client_id, cli_client_secret in _load_cli_client_value_candidates(cli_path):
        _add_client_candidate(
            candidates,
            seen,
            cli_client_id,
            cli_client_secret,
        )
    return candidates


def _add_client_candidate(
    candidates: list[tuple[str, str]],
    seen: set[tuple[str, str]],
    client_id: Optional[str],
    client_secret: Optional[str],
) -> None:
    if not client_id or not client_secret:
        return
    candidate = (client_id, client_secret)
    if candidate in seen:
        return
    seen.add(candidate)
    candidates.append(candidate)


def _load_cli_client_value_candidates(
    cli_path: Optional[str | Path],
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in _iter_cli_binary_candidates(cli_path):
        try:
            cli_bytes = candidate.read_bytes()
        except OSError:
            continue
        for client_id, client_secret in _extract_cli_client_value_candidates(cli_bytes):
            _add_client_candidate(candidates, seen, client_id, client_secret)
    return candidates


def _iter_cli_binary_candidates(cli_path: Optional[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    seen_paths: set[str] = set()
    explicit_paths = []
    if cli_path is not None:
        explicit_paths.append(str(cli_path))
    env_path = _first_env("AAWM_ANTIGRAVITY_CLI_PATH", "LITELLM_ANTIGRAVITY_CLI_PATH", "ANTIGRAVITY_CLI_PATH")
    if env_path:
        explicit_paths.append(env_path)
    for raw_path in (*explicit_paths, *_DEFAULT_CLI_BINARY_PATHS):
        candidate = Path(raw_path).expanduser()
        if not candidate.is_file():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidates.append(candidate)
    return candidates


def _first_cli_binary(cli_path: Optional[str | Path]) -> Optional[Path]:
    candidates = _iter_cli_binary_candidates(cli_path)
    return candidates[0] if candidates else None


def _extract_cli_client_value_candidates(cli_bytes: bytes) -> list[tuple[str, str]]:
    client_secret_matches = list(_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN.finditer(cli_bytes))
    client_id_matches = list(_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN.finditer(cli_bytes))
    if not client_secret_matches or not client_id_matches:
        return []
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for client_secret_match in client_secret_matches:
        client_secret = client_secret_match.group("value").decode("ascii", errors="ignore")
        for client_id_match in sorted(
            client_id_matches,
            key=lambda match: abs(match.start() - client_secret_match.start()),
        ):
            _add_client_candidate(
                candidates,
                seen,
                client_id_match.group("value").decode("ascii", errors="ignore"),
                client_secret,
            )
    return candidates


def _first_mapping_value(
    values: Mapping[str, Any],
    candidate_keys: tuple[str, ...],
) -> Optional[str]:
    for key in candidate_keys:
        value = _clean_value(values.get(key))
        if value is not None:
            return value
    return None


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = _clean_value(os.getenv(name))
        if value is not None:
            return value
    return None


def _clean_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class _CredentialFileMetadata:
    uid: Optional[int]
    gid: Optional[int]
    mode: int


def _snapshot_credential_file_metadata(credential_path: Path) -> _CredentialFileMetadata:
    if not credential_path.exists():
        return _CredentialFileMetadata(
            uid=None,
            gid=None,
            mode=DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE,
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


def _write_token_data(auth_path: Path, token_data: Mapping[str, Any]) -> None:
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _snapshot_credential_file_metadata(auth_path)
    if metadata.mode & 0o077:
        metadata = _CredentialFileMetadata(
            uid=metadata.uid,
            gid=metadata.gid,
            mode=DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE,
        )
    tmp_path = auth_path.with_name(f".{auth_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(token_data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _apply_credential_file_metadata(tmp_path, metadata)
        os.replace(tmp_path, auth_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


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


def _sanitize_error_message(message: str) -> str:
    sanitized = message
    for field_name in _SECRET_FIELD_NAMES:
        sanitized = re.sub(
            rf"(?i)\b{re.escape(field_name)}\b\s*[:=]\s*\S+",
            f"{field_name}=[REDACTED]",
            sanitized,
        )
    return sanitized
