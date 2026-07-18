"""Google OAuth credential helpers for AAWM pass-through adapters (RR-054 #1).

Owns Gemini OAuth file discovery, client-id bundle extraction, refresh, and the
process-local access-token cache used by Anthropic/Codex Google adapter routes.
"""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
from fastapi import HTTPException

from .state import alias_routing_state
from .types import Payload

# Auth path / client defaults (moved from llm_passthrough_endpoints).
_ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS = (
    "LITELLM_GEMINI_AUTH_FILE",
    "GEMINI_OAUTH_CREDS_FILE",
)
_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS = (
    "~/.gemini/oauth_creds.json",
    "~/.gemini/oauth_creds.json",
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS = (
    "LITELLM_GEMINI_OAUTH_CLIENT_ID",
    "GEMINI_OAUTH_CLIENT_ID",
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS = (
    "LITELLM_GEMINI_OAUTH_CLIENT_SECRET",
    "GEMINI_OAUTH_CLIENT_SECRET",
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS = (
    "LITELLM_GEMINI_CLI_BUNDLE_PATH",
    "GEMINI_CLI_BUNDLE_PATH",
)
_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS = (
    "~/.nvm/versions/node/*/lib/node_modules/@google/gemini-cli/bundle",
    "~/.nvm/versions/node/*/lib/node_modules/@google/gemini-cli/bundle",
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN = re.compile(
    r'OAUTH_CLIENT_ID\s*=\s*"(?P<value>[^"]+)"'
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN = re.compile(
    r'OAUTH_CLIENT_SECRET\s*=\s*"(?P<value>[^"]+)"'
)

# Cache aliases from package state manager.
_google_oauth_access_token_cache = alias_routing_state.google_oauth.tokens
_google_oauth_access_token_lock = alias_routing_state.google_oauth.lock


def _default_clean(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


# Optional dependency hooks set by llm_passthrough_endpoints at import time so we
# reuse existing secret/env helpers without circular imports of the god-module.
_clean_value: Callable[[object], Optional[str]] = _default_clean
_get_first_secret_value: Optional[Callable[[tuple[str, ...]], Optional[str]]] = None
_invalidate_google_lane_cache: Optional[Callable[[], None]] = None


def configure_google_oauth_runtime(
    *,
    clean_value: Callable[[object], Optional[str]],
    get_first_secret_value: Callable[[tuple[str, ...]], Optional[str]],
    invalidate_google_lane_cache: Callable[[], None],
) -> None:
    global _clean_value, _get_first_secret_value, _invalidate_google_lane_cache
    _clean_value = clean_value
    _get_first_secret_value = get_first_secret_value
    _invalidate_google_lane_cache = invalidate_google_lane_cache


def _clean(value: object) -> Optional[str]:
    return _clean_value(value)


def _secret(env_var_names: tuple[str, ...]) -> Optional[str]:
    if _get_first_secret_value is None:
        for name in env_var_names:
            raw = os.getenv(name)
            cleaned = _clean(raw)
            if cleaned:
                return cleaned
        return None
    return _get_first_secret_value(env_var_names)


def _invalidate_lane() -> None:
    if _invalidate_google_lane_cache is not None:
        _invalidate_google_lane_cache()

def _get_anthropic_adapter_google_auth_file_path() -> Optional[Path]:
    for env_name in _ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS:
        raw_value = _clean(os.getenv(env_name))
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if path.exists():
            return path

    for candidate_str in _ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.exists():
            return candidate

    return None


def _extract_google_oauth_client_values_from_bundle_text(
    bundle_text: str,
) -> tuple[Optional[str], Optional[str]]:
    client_id_match = _ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN.search(
        bundle_text
    )
    client_secret_match = (
        _ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN.search(bundle_text)
    )
    client_id = (
        _clean(client_id_match.group("value"))
        if client_id_match is not None
        else None
    )
    client_secret = (
        _clean(client_secret_match.group("value"))
        if client_secret_match is not None
        else None
    )
    return client_id, client_secret


def _add_google_cli_bundle_candidate_files(
    raw_path: Path, candidate_files: list[Path], seen_paths: set[str]
) -> None:
    path = raw_path.expanduser()
    if not path.exists():
        return

    if path.is_file():
        resolved = str(path.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            candidate_files.append(path)
        return

    bundle_dir = path
    if (bundle_dir / "bundle").is_dir():
        bundle_dir = bundle_dir / "bundle"
    elif (
        bundle_dir / "lib" / "node_modules" / "@google" / "gemini-cli" / "bundle"
    ).is_dir():
        bundle_dir = (
            bundle_dir / "lib" / "node_modules" / "@google" / "gemini-cli" / "bundle"
        )

    chunk_files = sorted(bundle_dir.glob("chunk-*.js"))
    gemini_bundle = bundle_dir / "gemini.js"
    ordered_files = chunk_files + ([gemini_bundle] if gemini_bundle.is_file() else [])
    for candidate in ordered_files:
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_files.append(candidate)


def _iter_google_oauth_client_bundle_candidates() -> list[Path]:
    candidate_files: list[Path] = []
    seen_paths: set[str] = set()

    for env_name in _ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS:
        raw_value = _clean(os.getenv(env_name))
        if raw_value:
            _add_google_cli_bundle_candidate_files(
                Path(raw_value), candidate_files, seen_paths
            )

    for bundle_glob in _ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS:
        for matched_path in sorted(
            glob.glob(os.path.expanduser(bundle_glob)), reverse=True
        ):
            _add_google_cli_bundle_candidate_files(
                Path(matched_path), candidate_files, seen_paths
            )

    return candidate_files


def _load_google_oauth_client_values_from_local_gemini_cli_bundle() -> (
    tuple[Optional[str], Optional[str]]
):
    for candidate in _iter_google_oauth_client_bundle_candidates():
        try:
            bundle_text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        client_id, client_secret = _extract_google_oauth_client_values_from_bundle_text(
            bundle_text
        )
        if client_id and client_secret:
            return client_id, client_secret

    return None, None


async def _load_local_google_oauth_credentials() -> tuple[Payload, Path]:
    auth_path = _get_anthropic_adapter_google_auth_file_path()
    if auth_path is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Anthropic adapter requests for Gemini models require local Google OAuth creds at "
                "'~/.gemini/oauth_creds.json' or 'LITELLM_GEMINI_AUTH_FILE'."
            ),
        )

    try:
        auth_data = json.loads(auth_path.read_text())
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Gemini OAuth credentials from {auth_path}: {exc}",
        ) from exc

    if not isinstance(auth_data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Gemini OAuth credentials at {auth_path} are not a JSON object.",
        )

    return auth_data, auth_path


def _google_oauth_token_is_valid(auth_data: Payload) -> bool:
    access_token = _clean(auth_data.get("access_token"))
    expiry_date = auth_data.get("expiry_date")
    if access_token is None:
        return False
    if not isinstance(expiry_date, (int, float)):
        return False
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return int(expiry_date) > now_ms + 60_000


def _google_oauth_cached_token_is_valid(cached_token: tuple[str, int]) -> bool:
    _access_token, expiry_date = cached_token
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return expiry_date > now_ms + 60_000


def _get_google_oauth_expiry_date(auth_data: Payload) -> Optional[int]:
    expiry_date = auth_data.get("expiry_date")
    if isinstance(expiry_date, (int, float)):
        return int(expiry_date)
    return None


def _get_google_oauth_client_value(
    auth_data: Payload,
    candidate_keys: tuple[str, ...],
    env_var_names: tuple[str, ...],
) -> Optional[str]:
    for key in candidate_keys:
        value = _clean(auth_data.get(key))
        if value is not None:
            return value
    return _secret(env_var_names)


async def _refresh_local_google_oauth_credentials(
    auth_data: Payload,
) -> Payload:
    refresh_token = _clean(auth_data.get("refresh_token"))
    if refresh_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini OAuth credentials do not contain a refresh_token. "
                "Re-authenticate Gemini CLI before using Gemini Anthropic adapter models."
            ),
        )

    client_id = _get_google_oauth_client_value(
        auth_data,
        ("client_id", "clientId"),
        _ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS,
    )
    client_secret = _get_google_oauth_client_value(
        auth_data,
        ("client_secret", "clientSecret"),
        _ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS,
    )
    if client_id is None or client_secret is None:
        (
            bundle_client_id,
            bundle_client_secret,
        ) = _load_google_oauth_client_values_from_local_gemini_cli_bundle()
        client_id = client_id or bundle_client_id
        client_secret = client_secret or bundle_client_secret
    if client_id is None or client_secret is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini OAuth credentials do not contain client_id/client_secret and no fallback env vars or Gemini CLI bundle were found. "
                "Re-authenticate Gemini CLI or configure Gemini OAuth client env vars before using Gemini Anthropic adapter models."
            ),
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to refresh Gemini OAuth access token for Anthropic adapter models: "
                f"{response.text}"
            ),
        )

    refreshed = response.json()
    expires_in = refreshed.get("expires_in")
    expiry_date: Optional[int] = None
    if isinstance(expires_in, (int, float)):
        expiry_date = int(datetime.now(timezone.utc).timestamp() * 1000) + int(
            expires_in * 1000
        )

    updated_auth_data = dict(auth_data)
    updated_auth_data.update(refreshed)
    updated_auth_data["refresh_token"] = refresh_token
    if expiry_date is not None:
        updated_auth_data["expiry_date"] = expiry_date

    _invalidate_lane()
    return updated_auth_data


async def _load_valid_local_google_oauth_access_token() -> str:
    auth_data, _auth_path = await _load_local_google_oauth_credentials()
    cache_key = str(_auth_path.expanduser())
    cached_token = _google_oauth_access_token_cache.get(cache_key)
    if cached_token is not None and _google_oauth_cached_token_is_valid(cached_token):
        return cached_token[0]

    async with _google_oauth_access_token_lock:
        cached_token = _google_oauth_access_token_cache.get(cache_key)
        if cached_token is not None and _google_oauth_cached_token_is_valid(
            cached_token
        ):
            return cached_token[0]

        auth_data, _auth_path = await _load_local_google_oauth_credentials()
        if not _google_oauth_token_is_valid(auth_data):
            auth_data = await _refresh_local_google_oauth_credentials(auth_data)

        access_token = _clean(auth_data.get("access_token"))
        if access_token is None:
            raise HTTPException(
                status_code=500,
                detail="Gemini OAuth credentials did not yield a valid access_token.",
            )
        expiry_date = _get_google_oauth_expiry_date(auth_data)
        if expiry_date is not None:
            _google_oauth_access_token_cache[cache_key] = (access_token, expiry_date)
        _invalidate_lane()
        return access_token


def _load_local_google_oauth_access_token() -> Optional[str]:
    auth_path = _get_anthropic_adapter_google_auth_file_path()
    if auth_path is None:
        return None

    try:
        auth_data = json.loads(auth_path.read_text())
    except Exception:
        return None

    access_token = _clean(auth_data.get("access_token"))
    if access_token is None:
        return None
    return access_token
