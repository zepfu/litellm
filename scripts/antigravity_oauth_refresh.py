#!/usr/bin/env python3
"""Refresh an Antigravity OAuth token file for the provider-status sidecar."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
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
    write_and_publish_private_text,
    write_private_file_text,
)
from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)

logger = logging.getLogger(__name__)

_STAGED_HOME_DIR_MODE = 0o700
_STAGED_AUTH_FILE_MODE = 0o600

# Portable ~ defaults (expanded via Path.expanduser at use sites).
DEFAULT_ANTIGRAVITY_AUTH_FILE = "~/.gemini/antigravity-cli/antigravity-oauth-token"
DEFAULT_ANTIGRAVITY_LOCK_FILE = "~/.gemini/antigravity-cli/antigravity-oauth-token.lock"
DEFAULT_ANTIGRAVITY_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
DEFAULT_ANTIGRAVITY_REFRESH_BUFFER_SECONDS = 300
DEFAULT_ANTIGRAVITY_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_ANTIGRAVITY_CLI_REFRESH_TIMEOUT_SECONDS = 30.0
DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE = 0o600

_DEFAULT_CLI_BINARY_PATHS = ("~/.local/bin/agy",)
_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN = re.compile(
    # Length-bound the random segment so binary padding between two client
    # IDs cannot glue them into one match.
    rb"(?P<value>\d{6,}-[A-Za-z0-9]{8,64}\.apps\.googleusercontent\.com)"
)
_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN = re.compile(
    # Non-greedy body that stops before another GOCSPX blob or a URL scheme.
    # Real `agy` embeds two secrets back-to-back then `https://...`.
    rb"(?P<value>GOCSPX-[A-Za-z0-9_-]{8,48}?)(?=GOCSPX-|https?://|[^A-Za-z0-9_-]|$)"
)
# Prefer nearest id/secret adjacency; discard pairs beyond this distance when a
# nearer pair exists for the same secret. Keeps binary scan a last-resort guess
# with a tighter correctness surface than full id×secret cartesian products.
_CLI_BINARY_PAIR_MAX_DISTANCE_BYTES = 2_000_000
_CLI_BINARY_MAX_CANDIDATE_PAIRS = 4
_CLI_BINARY_MAX_IDS_PER_SECRET = 2
# Keep historical module alias; redaction lives in secret_managers.
_SECRET_FIELD_NAMES = DEFAULT_SECRET_FIELD_NAMES


@dataclass(frozen=True)
class _ClientCredentialCandidate:
    """OAuth client pair with non-secret provenance for diagnostics."""

    client_id: str
    client_secret: str
    source: str
    proximity_bytes: Optional[int] = None


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
    seed_auth_file: str | Path | None = None,
    token_endpoint: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    cli_path: Optional[str | Path] = None,
    http_timeout_seconds: float = DEFAULT_ANTIGRAVITY_HTTP_TIMEOUT_SECONDS,
    cli_timeout_seconds: float = DEFAULT_ANTIGRAVITY_CLI_REFRESH_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Refresh an Antigravity OAuth token file when near expiry or forced."""

    resolved_auth_file = Path(auth_file).expanduser()
    resolved_seed_auth_file = _resolve_seed_auth_file(
        auth_file=resolved_auth_file,
        seed_auth_file=seed_auth_file,
    )
    resolved_lock_file = (
        Path(lock_file).expanduser()
        if lock_file is not None
        else resolved_auth_file.with_name(f"{resolved_auth_file.name}.lock")
    )
    resolved_buffer_seconds = _resolve_buffer_seconds(buffer_seconds)

    with _credential_file_lock(resolved_lock_file):
        try:
            loaded_from_seed = False
            try:
                token_data = _read_token_data(resolved_auth_file)
            except _TokenFileNotFound:
                if resolved_seed_auth_file is None:
                    raise
                token_data = _read_token_data(resolved_seed_auth_file)
                loaded_from_seed = True
            current_expires_at = _format_expires_at(_get_token_expiry(token_data))
            if not force and _token_is_valid(
                token_data,
                buffer_seconds=resolved_buffer_seconds,
            ):
                if loaded_from_seed:
                    _write_token_data(resolved_auth_file, token_data)
                    return AntigravityRefreshSummary(
                        attempted=True,
                        refreshed=True,
                        skipped=False,
                        auth_file=str(resolved_auth_file),
                        expires_at=current_expires_at,
                        refresh_method="seed_copy",
                    ).as_dict()
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
                    seed_auth_file=resolved_seed_auth_file,
                    original_token_data=token_data,
                    cli_path=cli_path,
                    timeout_seconds=cli_timeout_seconds,
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


def _resolve_seed_auth_file(
    *,
    auth_file: Path,
    seed_auth_file: str | Path | None,
) -> Optional[Path]:
    if seed_auth_file is not None:
        resolved_seed_auth_file = Path(seed_auth_file).expanduser()
        if resolved_seed_auth_file == auth_file:
            return None
        return resolved_seed_auth_file

    for env_name in (
        "AAWM_ANTIGRAVITY_SEED_AUTH_FILE",
        "LITELLM_ANTIGRAVITY_SEED_AUTH_FILE",
        "ANTIGRAVITY_SEED_AUTH_FILE",
    ):
        env_value = _clean_value(os.getenv(env_name))
        if env_value is None:
            continue
        resolved_seed_auth_file = Path(env_value).expanduser()
        if resolved_seed_auth_file == auth_file:
            return None
        return resolved_seed_auth_file
    return None


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
    """Delegate to shared credential_file_lock (module-scoped fcntl + warnings)."""
    with credential_file_lock(lock_path):
        yield


def _read_token_data(auth_path: Path) -> Dict[str, Any]:
    try:
        with auth_path.open("r", encoding="utf-8") as handle:
            token_data = json.load(handle)
    except FileNotFoundError as exc:
        raise _TokenFileNotFound(
            f"Antigravity OAuth token file not found at {auth_path}."
        ) from exc
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
        raise ValueError(
            "Antigravity OAuth token data does not contain a token object."
        )
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
    logger.info(
        "antigravity oauth direct refresh: %s client_id/secret candidate pair(s)",
        len(client_candidates),
    )
    for pair_index, candidate in enumerate(client_candidates, start=1):
        pair_id = _client_pair_diagnostic_id(
            candidate.client_id, candidate.client_secret
        )
        proximity = (
            f" proximity_bytes={candidate.proximity_bytes}"
            if candidate.proximity_bytes is not None
            else ""
        )
        logger.info(
            "antigravity oauth direct refresh: trying pair %s/%s id=%s "
            "source=%s%s",
            pair_index,
            len(client_candidates),
            pair_id,
            candidate.source,
            proximity,
        )
        try:
            refreshed = _post_refresh_request(
                token_endpoint=token_endpoint,
                client_id=candidate.client_id,
                client_secret=candidate.client_secret,
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
                logger.info(
                    "antigravity oauth direct refresh: pair id=%s source=%s "
                    "rejected (%s); trying next candidate if any",
                    pair_id,
                    candidate.source,
                    exc.oauth_error,
                )
                continue
            raise ValueError(
                f"Failed to refresh Antigravity OAuth access token ({exc})."
            ) from exc
        logger.info(
            "antigravity oauth direct refresh: selected pair id=%s source=%s",
            pair_id,
            candidate.source,
        )
        return _apply_refresh_payload(
            token_data,
            refreshed,
            refresh_token,
            client_id=candidate.client_id,
            client_secret=candidate.client_secret,
        )

    if last_oauth_error in {"invalid_client", "invalid_grant", "unauthorized_client"}:
        raise _DirectRefreshNeedsCliFallback() from last_error
    if last_error is not None:
        raise ValueError(
            f"Failed to refresh Antigravity OAuth access token ({last_error})."
        ) from last_error
    raise ValueError("Failed to refresh Antigravity OAuth access token.")


class _DirectRefreshNeedsCliFallback(ValueError):
    pass


class _TokenFileNotFound(ValueError):
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
        raise ValueError(
            "Antigravity OAuth token endpoint returned invalid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "Antigravity OAuth token endpoint returned a non-object payload."
        )
    return payload


def _apply_refresh_payload(
    token_data: Mapping[str, Any],
    refreshed: Mapping[str, Any],
    refresh_token: str,
    *,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
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
    # Persist the validated client pair so future refreshes prefer the
    # token-file source over fragile CLI binary discovery (RR-065 #7).
    if client_id and client_secret:
        updated_token_block["client_id"] = client_id
        updated_token_block["client_secret"] = client_secret
    updated_token_block["expiry"] = (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    ).isoformat()

    updated_token_data = dict(token_data)
    updated_token_data["token"] = dict(updated_token_block)
    if client_id and client_secret:
        updated_token_data["client_id"] = client_id
        updated_token_data["client_secret"] = client_secret
    return updated_token_data


def _mkdir_private(path: Path, *, mode: int = _STAGED_HOME_DIR_MODE) -> None:
    """Create a directory with restrictive permissions from the start."""
    path.mkdir(mode=mode, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _write_private_text(
    path: Path, content: str, *, mode: int = _STAGED_AUTH_FILE_MODE
) -> None:
    """Write ``content`` to ``path`` with private mode at creation time.

    Delegates to shared ``write_private_file_text`` so staged CLI auth files get
    the same no-umask-window create, mode clamp, and symlink refusal as final
    credential publishes (RR-065 consumer migration).
    """
    write_private_file_text(
        path,
        content,
        mode=mode,
        default_mode=_STAGED_AUTH_FILE_MODE,
        exclusive=False,
        refuse_symlink=True,
    )


def _stage_cli_auth_home(staged_home: Path, seed_auth_file: Path) -> Path:
    """Stage a private HOME tree containing the seed OAuth token file.

    Returns the staged auth-file path. Caller must remove ``staged_home``.
    """
    _mkdir_private(staged_home)
    gemini_dir = staged_home / ".gemini"
    _mkdir_private(gemini_dir)
    cli_dir = gemini_dir / "antigravity-cli"
    _mkdir_private(cli_dir)
    staged_auth_path = cli_dir / "antigravity-oauth-token"
    _write_private_text(
        staged_auth_path,
        seed_auth_file.read_text(encoding="utf-8"),
        mode=_STAGED_AUTH_FILE_MODE,
    )
    return staged_auth_path


def _cleanup_staged_home(staged_home: Optional[Path]) -> None:
    if staged_home is None:
        return
    try:
        shutil.rmtree(staged_home)
    except OSError:
        pass


def _make_private_temp_dir(prefix: str) -> Path:
    """Create an unpredictable private (0700) temp directory under TMPDIR.

    ``tempfile.mkdtemp`` defaults to mode 0o700; re-chmod to the staged-home
    constant so permission intent stays explicit across platforms.
    """
    path = Path(
        tempfile.mkdtemp(
            prefix=prefix,
            dir=os.getenv("TMPDIR") or None,
        )
    )
    try:
        os.chmod(path, _STAGED_HOME_DIR_MODE)
    except OSError:
        pass
    return path


def _refresh_token_data_via_cli(
    auth_path: Path,
    *,
    seed_auth_file: Optional[Path],
    original_token_data: Mapping[str, Any],
    cli_path: Optional[str | Path],
    timeout_seconds: float,
) -> Dict[str, Any]:
    cli_auth_path = seed_auth_file or auth_path
    refresh_home = _get_cli_refresh_home(cli_auth_path)
    cli_binary = _first_cli_binary(cli_path)
    if refresh_home is None or cli_binary is None:
        raise ValueError(
            "Antigravity OAuth direct refresh failed and AGY CLI silent refresh "
            "is unavailable for this auth-file path."
        )

    staged_home: Optional[Path] = None
    staged_auth_path = cli_auth_path
    log_path: Optional[Path] = None
    try:
        if seed_auth_file is not None and seed_auth_file != auth_path:
            staged_home = _make_private_temp_dir("litellm-antigravity-cli-home-")
            staged_auth_path = _stage_cli_auth_home(staged_home, seed_auth_file)
            refresh_home = staged_home

        # Prefer the private 0700 staged home for CLI logs; otherwise create a
        # private temp dir solely for the log so /tmp is never world-writable
        # host for token-adjacent diagnostics.
        log_parent = staged_home
        if log_parent is None:
            log_parent = _make_private_temp_dir("litellm-antigravity-cli-log-")
            staged_home = log_parent  # ensure unconditional cleanup
        log_path = log_parent / "litellm-antigravity-refresh.log"
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

        if process.returncode != 0:
            raise ValueError(
                "AGY CLI silent auth refresh failed. Re-authenticate Antigravity CLI "
                "before using Antigravity passthrough."
            )

        refreshed_token_data = _read_token_data(staged_auth_path)
        if not _token_is_valid(refreshed_token_data, buffer_seconds=60):
            if refreshed_token_data == original_token_data and _token_is_unexpired(
                refreshed_token_data
            ):
                return refreshed_token_data
            raise ValueError(
                "AGY CLI silent auth refresh did not produce a valid token."
            )
        return refreshed_token_data
    finally:
        # Always remove staged credential/log dirs, including timeout / OSError /
        # non-zero CLI exit paths that previously left orphan token trees.
        if log_path is not None:
            try:
                log_path.unlink()
            except OSError:
                pass
        _cleanup_staged_home(staged_home)


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
) -> list[_ClientCredentialCandidate]:
    """Resolve client credentials with an explicit safer-first discovery chain.

    Order (RR-065 #7):
      1. explicit function args / env vars (operator-configured)
      2. values already stored on the token file (validated prior success)
      3. CLI binary scan only as last-resort fallback when safer sources empty

    Binary scan is never preferred over configured/token sources. When a safer
    source already produced candidates, the binary is not opened.
    """
    candidates: list[_ClientCredentialCandidate] = []
    seen: set[tuple[str, str]] = set()

    # Explicit kwargs win over env so callers can inject one-shot credentials.
    _add_client_candidate(
        candidates,
        seen,
        client_id,
        client_secret,
        source="explicit_args",
    )
    _add_client_candidate(
        candidates,
        seen,
        _first_env(
            "AAWM_ANTIGRAVITY_OAUTH_CLIENT_ID",
            "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_ID",
            "ANTIGRAVITY_OAUTH_CLIENT_ID",
        ),
        _first_env(
            "AAWM_ANTIGRAVITY_OAUTH_CLIENT_SECRET",
            "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_SECRET",
            "ANTIGRAVITY_OAUTH_CLIENT_SECRET",
        ),
        source="env",
    )

    token_block = token_data.get("token")
    token_mapping = token_block if isinstance(token_block, Mapping) else {}
    _add_client_candidate(
        candidates,
        seen,
        _first_mapping_value(token_data, ("client_id", "clientId"))
        or _first_mapping_value(token_mapping, ("client_id", "clientId")),
        _first_mapping_value(token_data, ("client_secret", "clientSecret"))
        or _first_mapping_value(token_mapping, ("client_secret", "clientSecret")),
        source="token_file",
    )

    if candidates:
        logger.info(
            "antigravity oauth client discovery: sources=%s pairs=%s "
            "(skipping cli binary scan; safer source available)",
            ",".join(sorted({c.source for c in candidates})),
            len(candidates),
        )
        return candidates

    logger.info(
        "antigravity oauth client discovery: no configured/token client pair; "
        "falling back to cli binary scan"
    )
    for binary_candidate in _load_cli_client_value_candidates(cli_path):
        _add_client_candidate(
            candidates,
            seen,
            binary_candidate.client_id,
            binary_candidate.client_secret,
            source=binary_candidate.source,
            proximity_bytes=binary_candidate.proximity_bytes,
        )
    return candidates


def _add_client_candidate(
    candidates: list[_ClientCredentialCandidate],
    seen: set[tuple[str, str]],
    client_id: Optional[str],
    client_secret: Optional[str],
    *,
    source: str,
    proximity_bytes: Optional[int] = None,
) -> None:
    if not client_id or not client_secret:
        return
    key = (client_id, client_secret)
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        _ClientCredentialCandidate(
            client_id=client_id,
            client_secret=client_secret,
            source=source,
            proximity_bytes=proximity_bytes,
        )
    )


def _load_cli_client_value_candidates(
    cli_path: Optional[str | Path],
) -> list[_ClientCredentialCandidate]:
    candidates: list[_ClientCredentialCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate_path in _iter_cli_binary_candidates(cli_path):
        try:
            cli_bytes = candidate_path.read_bytes()
        except OSError as exc:
            logger.info(
                "antigravity cli binary scan: path_unreadable reason=%s",
                type(exc).__name__,
            )
            continue
        for extracted in _extract_cli_client_value_candidates(cli_bytes):
            _add_client_candidate(
                candidates,
                seen,
                extracted.client_id,
                extracted.client_secret,
                source=extracted.source,
                proximity_bytes=extracted.proximity_bytes,
            )
    return candidates


def _iter_cli_binary_candidates(cli_path: Optional[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    seen_paths: set[str] = set()
    explicit_paths = []
    if cli_path is not None:
        explicit_paths.append(str(cli_path))
    env_path = _first_env(
        "AAWM_ANTIGRAVITY_CLI_PATH",
        "LITELLM_ANTIGRAVITY_CLI_PATH",
        "ANTIGRAVITY_CLI_PATH",
    )
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


def _client_pair_diagnostic_id(client_id: str, client_secret: str) -> str:
    """Stable non-secret identifier for a client_id/secret pair.

    Never logs full client IDs or secrets; only a short hash prefix of the pair.
    """
    digest = hashlib.sha256(f"{client_id}\0{client_secret}".encode("utf-8")).hexdigest()
    return digest[:12]


def _extract_cli_client_value_candidates(
    cli_bytes: bytes,
) -> list[_ClientCredentialCandidate]:
    """Recover OAuth client pairs by scanning CLI binary bytes.

    Last-resort discovery only. Prefer env/token-file values via
    `_get_client_value_candidates` so this path is skipped when safer sources
    exist. No official AGY subcommand exposes client credentials (verified
    against `agy --help`); config files under the CLI home do not either.

    Hardening vs naive byte-proximity scrape (RR-065 #7):
      - bounded secret regex (no adjacent-secret concatenation)
      - nearest-id ranking per secret with a max distance budget
      - cap ids-per-secret and total pairs to avoid combinatorial wrong pairs
      - log counts + proximity diagnostics without secret material
    """
    client_secret_matches = list(
        _CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN.finditer(cli_bytes)
    )
    client_id_matches = list(_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN.finditer(cli_bytes))
    if not client_secret_matches or not client_id_matches:
        logger.info(
            "antigravity cli binary scan: secrets=%s client_ids=%s pairs=0",
            len(client_secret_matches),
            len(client_id_matches),
        )
        return []

    ranked: list[tuple[int, str, str]] = []
    for client_secret_match in client_secret_matches:
        client_secret = client_secret_match.group("value").decode(
            "ascii", errors="ignore"
        )
        if not _looks_like_oauth_client_secret(client_secret):
            continue
        nearest = sorted(
            client_id_matches,
            key=lambda match: abs(match.start() - client_secret_match.start()),
        )
        accepted_for_secret = 0
        for client_id_match in nearest:
            distance = abs(client_id_match.start() - client_secret_match.start())
            if distance > _CLI_BINARY_PAIR_MAX_DISTANCE_BYTES:
                break
            client_id = client_id_match.group("value").decode(
                "ascii", errors="ignore"
            )
            if not _looks_like_oauth_client_id(client_id):
                continue
            ranked.append((distance, client_id, client_secret))
            accepted_for_secret += 1
            if accepted_for_secret >= _CLI_BINARY_MAX_IDS_PER_SECRET:
                break

    ranked.sort(key=lambda item: item[0])
    candidates: list[_ClientCredentialCandidate] = []
    seen: set[tuple[str, str]] = set()
    for distance, client_id, client_secret in ranked:
        if len(candidates) >= _CLI_BINARY_MAX_CANDIDATE_PAIRS:
            break
        _add_client_candidate(
            candidates,
            seen,
            client_id,
            client_secret,
            source="cli_binary_scan",
            proximity_bytes=distance,
        )

    nearest_dist = ranked[0][0] if ranked else None
    logger.info(
        "antigravity cli binary scan: secrets=%s client_ids=%s pairs=%s "
        "nearest_proximity_bytes=%s max_distance=%s",
        len(client_secret_matches),
        len(client_id_matches),
        len(candidates),
        nearest_dist,
        _CLI_BINARY_PAIR_MAX_DISTANCE_BYTES,
    )
    return candidates


def _looks_like_oauth_client_id(client_id: str) -> bool:
    if not client_id or not client_id.endswith(".apps.googleusercontent.com"):
        return False
    if client_id.count(".apps.googleusercontent.com") != 1:
        return False
    prefix = client_id[: -len(".apps.googleusercontent.com")]
    if "-" not in prefix:
        return False
    project, _, random_part = prefix.partition("-")
    return (
        project.isdigit()
        and 6 <= len(project) <= 20
        and random_part.isalnum()
        and 8 <= len(random_part) <= 64
    )


def _looks_like_oauth_client_secret(client_secret: str) -> bool:
    # Google OAuth client secrets are GOCSPX- plus a moderate body; reject
    # concatenated adjacent secrets and URL fragments from binary scrapes.
    if not client_secret or not client_secret.startswith("GOCSPX-"):
        return False
    if client_secret.count("GOCSPX-") != 1:
        return False
    body = client_secret[len("GOCSPX-") :]
    if not (8 <= len(body) <= 48):
        return False
    if body.lower().endswith(("http", "https", "www")):
        return False
    return body.replace("-", "").replace("_", "").isalnum()


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


def _snapshot_credential_file_metadata(
    credential_path: Path,
) -> CredentialFileMetadata:
    """Delegate to shared metadata owner (RR-065 residual #5)."""
    return snapshot_credential_file_metadata(
        credential_path,
        default_mode=DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _resolve_credential_file_metadata(auth_path: Path) -> CredentialFileMetadata:
    """Snapshot path metadata and apply optional Antigravity env overrides."""
    return resolve_credential_file_metadata(
        auth_path,
        default_mode=DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE,
        mode_env="AAWM_ANTIGRAVITY_AUTH_FILE_MODE",
        uid_env="AAWM_ANTIGRAVITY_AUTH_FILE_UID",
        gid_env="AAWM_ANTIGRAVITY_AUTH_FILE_GID",
        base_metadata=_snapshot_credential_file_metadata(auth_path),
        refuse_symlink=True,
    )


def _apply_credential_file_metadata(
    target_path: Path,
    metadata: CredentialFileMetadata,
) -> None:
    """Delegate to shared metadata owner (RR-065 residual #5)."""
    apply_credential_file_metadata(
        target_path,
        metadata,
        default_mode=DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE,
        refuse_symlink=True,
    )


def _write_token_data(auth_path: Path, token_data: Mapping[str, Any]) -> None:
    """Atomically publish token JSON via shared private write helpers.

    Uses unpredictable exclusive same-dir temps, refuses symlink targets, applies
    ownership/mode metadata without following links, and cleans failed temps.
    """
    metadata = _resolve_credential_file_metadata(auth_path)
    payload = json.dumps(token_data, indent=2, sort_keys=True) + "\n"
    write_and_publish_private_text(
        auth_path,
        payload,
        metadata=metadata,
        default_mode=DEFAULT_ANTIGRAVITY_AUTH_FILE_MODE,
        mkdir_parents=True,
    )


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


def _sanitize_error_message(message: str, *, limit: int = 500) -> str:
    """Redact secret *values* keyed by known field names (not just the labels).

    Output is bounded so long upstream bodies cannot flood observations/logs.
    """
    return sanitize_credential_error_message(message, limit=limit)
