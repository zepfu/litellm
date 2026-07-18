"""Antigravity OAuth credential helpers for AAWM pass-through (RR-054 #1).

Owns auth-file discovery, token load/validate, client-id extraction, refresh
(direct + CLI fallback), and process-local access-token cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
from fastapi import HTTPException

from .state import alias_routing_state
from .types import Payload

_ANTIGRAVITY_AUTH_FILE_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_AUTH_FILE",
    "ANTIGRAVITY_OAUTH_TOKEN_FILE",
)
_ANTIGRAVITY_MANAGED_AUTH_FILE_ENV_VARS = ("LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE",)
_ANTIGRAVITY_SEED_AUTH_FILE_ENV_VARS = ("LITELLM_ANTIGRAVITY_SEED_AUTH_FILE",)
_ANTIGRAVITY_DEFAULT_AUTH_PATHS = (
    "~/.gemini/antigravity-cli/antigravity-oauth-token",
    "~/.gemini/antigravity-cli/antigravity-oauth-token",
)
_ANTIGRAVITY_OAUTH_CLIENT_ID_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_ID",
    "ANTIGRAVITY_OAUTH_CLIENT_ID",
)
_ANTIGRAVITY_OAUTH_CLIENT_SECRET_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_SECRET",
    "ANTIGRAVITY_OAUTH_CLIENT_SECRET",
)
_ANTIGRAVITY_CLI_BINARY_PATH_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_CLI_PATH",
    "ANTIGRAVITY_CLI_PATH",
)
_ANTIGRAVITY_DEFAULT_CLI_BINARY_PATHS = (
    "~/.local/bin/agy",
    "~/.local/bin/agy",
)
_ANTIGRAVITY_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN = re.compile(
    r"(?P<value>[0-9]+-[A-Za-z0-9_-]+\.apps\.googleusercontent\.com)"
)

_ANTIGRAVITY_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN = re.compile(
    r"(?P<value>GOCSPX-[A-Za-z0-9_-]+)"
)


GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

_antigravity_oauth_access_token_cache = alias_routing_state.antigravity_oauth.tokens
_antigravity_oauth_access_token_lock = alias_routing_state.antigravity_oauth.lock

_clean_value: Callable[[object], Optional[str]]
_get_first_secret_value_fn: Optional[Callable[[tuple[str, ...]], Optional[str]]] = None
_invalidate_lane_cache_fn: Optional[Callable[[], None]] = None
_write_json_atomic_fn: Optional[Callable[..., None]] = None
_iter_cli_binaries_fn: Optional[Callable[[], list[Path]]] = None
_oauth_error_code_fn: Optional[Callable[[httpx.Response], Optional[str]]] = None
_format_refresh_failure_fn: Optional[Callable[..., str]] = None


def _default_clean(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


_clean_value = _default_clean


def configure_antigravity_oauth_runtime(
    *,
    clean_value: Callable[[object], Optional[str]],
    get_first_secret_value: Callable[[tuple[str, ...]], Optional[str]],
    invalidate_lane_cache: Callable[[], None],
    write_json_atomic: Callable[..., None],
    iter_cli_binaries: Callable[[], list[Path]],
    oauth_error_code: Callable[[httpx.Response], Optional[str]],
    format_refresh_failure: Callable[..., str],
) -> None:
    global _clean_value, _get_first_secret_value_fn, _invalidate_lane_cache_fn
    global _write_json_atomic_fn, _iter_cli_binaries_fn, _oauth_error_code_fn
    global _format_refresh_failure_fn
    _clean_value = clean_value
    _get_first_secret_value_fn = get_first_secret_value
    _invalidate_lane_cache_fn = invalidate_lane_cache
    _write_json_atomic_fn = write_json_atomic
    _iter_cli_binaries_fn = iter_cli_binaries
    _oauth_error_code_fn = oauth_error_code
    _format_refresh_failure_fn = format_refresh_failure


def _clean(value: object) -> Optional[str]:
    return _clean_value(value)


def _secret(env_var_names: tuple[str, ...]) -> Optional[str]:
    if _get_first_secret_value_fn is None:
        for name in env_var_names:
            cleaned = _clean(os.getenv(name))
            if cleaned:
                return cleaned
        return None
    return _get_first_secret_value_fn(env_var_names)


def _invalidate_lane() -> None:
    if _invalidate_lane_cache_fn is not None:
        _invalidate_lane_cache_fn()


def _write_json_atomic(path: Path, data: object, failure_label: str = "token") -> None:
    if _write_json_atomic_fn is None:
        raise RuntimeError("antigravity_oauth write_json_atomic not configured")
    return _write_json_atomic_fn(path, data, failure_label=failure_label)


def _iter_cli_binaries() -> list[Path]:
    if _iter_cli_binaries_fn is None:
        return []
    return _iter_cli_binaries_fn()


def _oauth_error_code(response: httpx.Response) -> Optional[str]:
    if _oauth_error_code_fn is None:
        return None
    return _oauth_error_code_fn(response)


def _format_refresh_failure(*, provider_label: str, response: object) -> str:
    if _format_refresh_failure_fn is None:
        return f"Failed to refresh {provider_label} OAuth access token."
    return _format_refresh_failure_fn(provider_label=provider_label, response=response)

def _iter_antigravity_auth_file_path_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen_paths: set[str] = set()
    for env_name in (
        *_ANTIGRAVITY_MANAGED_AUTH_FILE_ENV_VARS,
        *_ANTIGRAVITY_AUTH_FILE_ENV_VARS,
        *_ANTIGRAVITY_SEED_AUTH_FILE_ENV_VARS,
    ):
        raw_value = _clean(os.getenv(env_name))
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidates.append(path)

    for candidate_str in _ANTIGRAVITY_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidates.append(candidate)
    return candidates


def _get_antigravity_auth_file_path() -> Optional[Path]:
    candidates = _iter_antigravity_auth_file_path_candidates()
    if not candidates:
        return None
    return candidates[0]


async def _load_antigravity_oauth_token_data_from_path(
    auth_path: Path,
) -> Payload:
    try:
        token_data = json.loads(auth_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Antigravity OAuth token data from {auth_path}: {exc}",
        ) from exc

    if not isinstance(token_data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Antigravity OAuth token data at {auth_path} is not a JSON object.",
        )

    return token_data


async def _load_local_antigravity_oauth_token_data() -> tuple[
    Payload, Path
]:
    candidates = _iter_antigravity_auth_file_path_candidates()
    if not candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity passthrough requires local OAuth token data at "
                "'~/.gemini/antigravity-cli/antigravity-oauth-token', "
                "'LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE', "
                "'LITELLM_ANTIGRAVITY_SEED_AUTH_FILE', or "
                "'LITELLM_ANTIGRAVITY_AUTH_FILE'."
            ),
        )

    first_loaded: Optional[tuple[Payload, Path]] = None
    first_error: Optional[HTTPException] = None
    for auth_path in candidates:
        try:
            token_data = await _load_antigravity_oauth_token_data_from_path(auth_path)
        except HTTPException as exc:
            if first_error is None:
                first_error = exc
            continue
        if first_loaded is None:
            first_loaded = (token_data, auth_path)
        if _antigravity_access_token_is_valid(token_data):
            return token_data, auth_path

    if first_loaded is not None:
        return first_loaded
    if first_error is not None:
        raise first_error
    raise HTTPException(
        status_code=500,
        detail="Antigravity OAuth token candidate paths could not be loaded.",
    )


def _parse_antigravity_token_expiry(expiry: object) -> Optional[datetime]:
    if not isinstance(expiry, str) or not expiry.strip():
        return None
    cleaned = expiry.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _antigravity_access_token_is_valid(token_data: Payload) -> bool:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        return False
    access_token = _clean(token_block.get("access_token"))
    if access_token is None:
        return False
    expiry = _parse_antigravity_token_expiry(token_block.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc) + timedelta(seconds=60)


def _antigravity_access_token_is_unexpired(token_data: Payload) -> bool:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        return False
    access_token = _clean(token_block.get("access_token"))
    if access_token is None:
        return False
    expiry = _parse_antigravity_token_expiry(token_block.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc)


def _antigravity_oauth_cached_token_is_valid(cached_token: tuple[str, int]) -> bool:
    _access_token, expiry_date = cached_token
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return expiry_date > now_ms + 60_000


def _get_antigravity_oauth_expiry_date(token_data: Payload) -> Optional[int]:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        return None
    expiry = _parse_antigravity_token_expiry(token_block.get("expiry"))
    if expiry is None:
        return None
    return int(expiry.timestamp() * 1000)


def _add_antigravity_oauth_client_candidate(
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


def _extract_antigravity_oauth_client_values_from_cli_text(
    cli_text: str,
) -> tuple[Optional[str], Optional[str]]:
    candidates = _extract_antigravity_oauth_client_value_candidates_from_cli_text(
        cli_text
    )
    if not candidates:
        return None, None
    return candidates[0]


def _extract_antigravity_oauth_client_value_candidates_from_cli_text(
    cli_text: str,
) -> list[tuple[str, str]]:
    client_secret_matches = list(
        _ANTIGRAVITY_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN.finditer(cli_text)
    )
    client_id_matches = list(
        _ANTIGRAVITY_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN.finditer(cli_text)
    )
    if not client_secret_matches or not client_id_matches:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for client_secret_match in client_secret_matches:
        client_secret = _clean(client_secret_match.group("value"))
        for client_id_match in sorted(
            client_id_matches,
            key=lambda match: abs(match.start() - client_secret_match.start()),
        ):
            _add_antigravity_oauth_client_candidate(
                candidates,
                seen,
                _clean(client_id_match.group("value")),
                client_secret,
            )
    return candidates


def _load_antigravity_oauth_client_values_from_local_cli_binary() -> (
    tuple[Optional[str], Optional[str]]
):
    candidates = _load_antigravity_oauth_client_value_candidates_from_local_cli_binary()
    if not candidates:
        return None, None
    return candidates[0]


def _load_antigravity_oauth_client_value_candidates_from_local_cli_binary() -> (
    list[tuple[str, str]]
):
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in _iter_cli_binaries():
        try:
            cli_text = candidate.read_bytes().decode("latin1", errors="ignore")
        except OSError:
            continue
        client_value_candidates = (
            _extract_antigravity_oauth_client_value_candidates_from_cli_text(cli_text)
        )
        for client_id, client_secret in client_value_candidates:
            _add_antigravity_oauth_client_candidate(
                candidates,
                seen,
                client_id,
                client_secret,
            )
    return candidates


def _get_antigravity_oauth_client_value_from_token_data(
    token_data: Payload,
    candidate_keys: tuple[str, ...],
) -> Optional[str]:
    for key in candidate_keys:
        value = _clean(token_data.get(key))
        if value is not None:
            return value
    return None


def _get_antigravity_oauth_client_value_candidates(
    token_data: Payload,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    env_client_id = _secret(_ANTIGRAVITY_OAUTH_CLIENT_ID_ENV_VARS)
    env_client_secret = _secret(
        _ANTIGRAVITY_OAUTH_CLIENT_SECRET_ENV_VARS
    )
    token_client_id = _get_antigravity_oauth_client_value_from_token_data(
        token_data,
        ("client_id", "clientId"),
    )
    token_client_secret = _get_antigravity_oauth_client_value_from_token_data(
        token_data,
        ("client_secret", "clientSecret"),
    )
    _add_antigravity_oauth_client_candidate(
        candidates,
        seen,
        env_client_id,
        env_client_secret,
    )
    _add_antigravity_oauth_client_candidate(
        candidates,
        seen,
        token_client_id,
        token_client_secret,
    )
    for (
        client_id,
        client_secret,
    ) in _load_antigravity_oauth_client_value_candidates_from_local_cli_binary():
        _add_antigravity_oauth_client_candidate(
            candidates,
            seen,
            client_id,
            client_secret,
        )
    return candidates


def _write_antigravity_oauth_token_data_atomic(
    auth_path: Path,
    token_data: Payload,
) -> None:
    _write_json_atomic(
        auth_path,
        token_data,
        failure_label="Antigravity OAuth token",
    )


def _get_antigravity_cli_refresh_home(auth_path: Path) -> Optional[Path]:
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


def _get_antigravity_cli_refresh_timeout_seconds() -> float:
    raw_value = _clean(
        os.getenv("AAWM_ANTIGRAVITY_CLI_REFRESH_TIMEOUT_SECONDS")
    )
    if raw_value is None:
        return 30.0
    try:
        parsed = float(raw_value)
    except ValueError:
        return 30.0
    return max(parsed, 1.0)


async def _refresh_local_antigravity_oauth_token_data_via_cli(
    auth_path: Path,
    original_token_data: Optional[Payload] = None,
) -> Payload:
    refresh_home = _get_antigravity_cli_refresh_home(auth_path)
    cli_candidates = _iter_cli_binaries()
    if refresh_home is None or not cli_candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth direct refresh failed and AGY CLI silent "
                "refresh is unavailable for this auth-file path."
            ),
        )

    log_path = Path(os.getenv("TMPDIR") or "/tmp") / (
        f"litellm-antigravity-refresh-{os.getpid()}-{time.monotonic_ns()}.log"
    )
    env = dict(os.environ)
    env["HOME"] = str(refresh_home)
    try:
        process = await asyncio.create_subprocess_exec(
            str(cli_candidates[0]),
            "--log-file",
            str(log_path),
            "models",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        await asyncio.wait_for(
            process.communicate(),
            timeout=_get_antigravity_cli_refresh_timeout_seconds(),
        )
    except asyncio.TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(OSError, RuntimeError):
            await process.wait()
        raise HTTPException(
            status_code=500,
            detail="AGY CLI silent auth refresh timed out.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AGY CLI silent auth refresh failed ({type(exc).__name__}).",
        ) from exc
    finally:
        with contextlib.suppress(OSError):
            log_path.unlink()

    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(
                "AGY CLI silent auth refresh failed. Re-authenticate "
                "Antigravity CLI before using Antigravity passthrough."
            ),
        )

    refreshed_token_data = await _load_antigravity_oauth_token_data_from_path(auth_path)
    if not _antigravity_access_token_is_valid(refreshed_token_data):
        if (
            original_token_data is not None
            and refreshed_token_data == original_token_data
            and _antigravity_access_token_is_unexpired(refreshed_token_data)
        ):
            return refreshed_token_data
        raise HTTPException(
            status_code=500,
            detail="AGY CLI silent auth refresh did not produce a valid token.",
        )
    return refreshed_token_data


async def _refresh_local_antigravity_oauth_token_data(
    token_data: Payload,
    auth_path: Optional[Path] = None,
) -> Payload:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        raise HTTPException(
            status_code=500,
            detail=("Antigravity OAuth token data does not contain a token object."),
        )
    refresh_token = _clean(token_block.get("refresh_token"))
    if refresh_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth token data does not contain a refresh_token. "
                "Re-authenticate Antigravity CLI before using Antigravity passthrough."
            ),
        )

    client_candidates = _get_antigravity_oauth_client_value_candidates(token_data)
    if not client_candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth token data does not contain client_id/client_secret "
                "and no fallback env vars or Antigravity CLI binary values were found."
            ),
        )

    response: Optional[httpx.Response] = None
    async with httpx.AsyncClient(timeout=30.0) as client:
        for client_id, client_secret in client_candidates:
            response = await client.post(
                GOOGLE_OAUTH_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status_code == 200:
                break
            error_code = _oauth_error_code(response)
            if error_code not in {
                "invalid_client",
                "invalid_grant",
                "unauthorized_client",
            }:
                break

    if response is None or response.status_code != 200:
        if (
            response is not None
            and auth_path is not None
            and _oauth_error_code(response)
            in {"invalid_client", "invalid_grant", "unauthorized_client"}
        ):
            return await _refresh_local_antigravity_oauth_token_data_via_cli(
                auth_path,
                token_data,
            )
        raise HTTPException(
            status_code=500,
            detail=_format_refresh_failure(
                provider_label="Antigravity",
                response=response,
            )
            if response is not None
            else "Failed to refresh Antigravity OAuth access token.",
        )

    refreshed = response.json()
    refreshed_access_token = _clean(refreshed.get("access_token"))
    if refreshed_access_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth refresh response did not contain an access_token."
            ),
        )
    expires_in = refreshed.get("expires_in")
    if not isinstance(expires_in, (int, float)):
        raise HTTPException(
            status_code=500,
            detail="Antigravity OAuth refresh response did not contain expires_in.",
        )
    expiry = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

    updated_token_block = dict(token_block)
    updated_token_block.update(refreshed)
    updated_token_block["access_token"] = refreshed_access_token
    updated_token_block["refresh_token"] = refresh_token
    updated_token_block["expiry"] = expiry.isoformat()

    updated_token_data = dict(token_data)
    updated_token_data["token"] = updated_token_block
    _invalidate_lane()
    return updated_token_data


async def _load_valid_local_antigravity_access_token() -> str:
    token_data, auth_path = await _load_local_antigravity_oauth_token_data()
    cache_key = str(auth_path.expanduser())
    cached_token = _antigravity_oauth_access_token_cache.get(cache_key)
    if cached_token is not None and _antigravity_oauth_cached_token_is_valid(
        cached_token
    ):
        return cached_token[0]

    async with _antigravity_oauth_access_token_lock:
        cached_token = _antigravity_oauth_access_token_cache.get(cache_key)
        if cached_token is not None and _antigravity_oauth_cached_token_is_valid(
            cached_token
        ):
            return cached_token[0]

        token_data, auth_path = await _load_local_antigravity_oauth_token_data()
        if not _antigravity_access_token_is_valid(token_data):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Antigravity OAuth token is expired or invalid. The "
                    "provider-status sidecar owns Antigravity auth refresh; "
                    "confirm the sidecar can write the configured token file "
                    f"and refresh {auth_path}."
                ),
            )

    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Antigravity OAuth token data at {auth_path} does not contain "
                "a token object."
            ),
        )
    access_token = _clean(token_block.get("access_token"))
    if access_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Antigravity OAuth token data at {auth_path} does not contain "
                "an access_token."
            ),
        )
    expiry_date = _get_antigravity_oauth_expiry_date(token_data)
    if expiry_date is not None:
        _antigravity_oauth_access_token_cache[cache_key] = (access_token, expiry_date)
    _invalidate_lane()
    return access_token
