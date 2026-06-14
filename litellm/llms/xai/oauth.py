import asyncio
from contextlib import contextmanager
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import httpx

from litellm.constants import XAI_API_BASE
from litellm.secret_managers.main import get_secret_str


OA_XAI_PROVIDER_PREFIX = "oa_xai/"
XAI_OAUTH_ROUTE_FAMILY = "xai_oauth_api"
XAI_OAUTH_CREDENTIAL_FAMILY = "xai_oauth"
XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY = "xai_grok_subscription"
GROK_NATIVE_OAUTH_ROUTE_FAMILY = "grok_cli_chat_proxy"
GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY = "xai_grok_oidc"
GROK_NATIVE_OAUTH_CLIENT_NAME = "grok-build"

_DEFAULT_XAI_OAUTH_SCOPE = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
_DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
_DEFAULT_REFRESH_BUFFER_SECONDS = 300
_DEFAULT_HERMES_XAI_OAUTH_PROVIDER_ID = "xai-oauth"
_DEFAULT_HERMES_AUTH_PATH = "~/.hermes/auth.json"
_DEFAULT_LITELLM_XAI_OAUTH_AUTH_PATH = "~/.litellm/xai/oauth-auth.json"
_DEFAULT_GROK_XAI_OAUTH_AUTH_PATH = "~/.grok/auth.json"

_GROK_NATIVE_OAUTH_MODELS = frozenset(
    {
        "grok-build",
        "grok-build-0.1",
        "grok-composer-2.5-fast",
    }
)

_XAI_OAUTH_MODEL_MAP = {
    "oa_xai/grok-4.3": "xai/grok-4.3",
    "oa_xai/grok-4.20-0309-reasoning": "xai/grok-4.20-0309-reasoning",
    "oa_xai/grok-4.20-0309-non-reasoning": "xai/grok-4.20-0309-non-reasoning",
    "oa_xai/grok-4.20-multi-agent-0309": "xai/grok-4.20-multi-agent-0309",
}

_refresh_locks: Dict[str, asyncio.Lock] = {}


def is_oa_xai_model(model: Any) -> bool:
    return isinstance(model, str) and model.startswith(OA_XAI_PROVIDER_PREFIX)


def normalize_grok_native_oauth_model(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if candidate.startswith("xai/"):
        candidate = candidate[len("xai/") :]
    if candidate in _GROK_NATIVE_OAUTH_MODELS:
        return candidate
    return None


def is_grok_native_oauth_model(model: Any) -> bool:
    return normalize_grok_native_oauth_model(model) is not None


def resolve_oa_xai_upstream_model(model: str) -> str:
    if model in _XAI_OAUTH_MODEL_MAP:
        return _XAI_OAUTH_MODEL_MAP[model]
    if is_oa_xai_model(model):
        return "xai/" + model[len(OA_XAI_PROVIDER_PREFIX) :]
    raise ValueError(f"Unsupported xAI OAuth-managed model: {model}")


def build_oa_xai_metadata(public_model: str, upstream_model: str) -> Dict[str, Any]:
    return {
        "auth_mode": "oauth",
        "credential_family": XAI_OAUTH_CREDENTIAL_FAMILY,
        "passthrough_route_family": XAI_OAUTH_ROUTE_FAMILY,
        "route_family": XAI_OAUTH_ROUTE_FAMILY,
        "xai_oauth_managed": True,
        "xai_oauth_public_model": public_model,
        "xai_oauth_upstream_model": upstream_model,
        "xai_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "shared_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "grok_subscription_quota_shared": True,
        "model_group": public_model,
        "tags": [
            "route:xai_oauth_api",
            "auth:xai_oauth",
            "provider:xai",
            "quota:xai_grok_subscription",
        ],
    }


def build_grok_native_oauth_metadata(public_model: str) -> Dict[str, Any]:
    return {
        "auth_mode": "grok_oidc",
        "credential_family": GROK_NATIVE_OAUTH_CREDENTIAL_FAMILY,
        "client_name": GROK_NATIVE_OAUTH_CLIENT_NAME,
        "grok_cli_chat_proxy": True,
        "grok_model_override": public_model,
        "grok_native_oauth_managed": True,
        "model_group": public_model,
        "passthrough_route_family": GROK_NATIVE_OAUTH_ROUTE_FAMILY,
        "route_family": GROK_NATIVE_OAUTH_ROUTE_FAMILY,
        "shared_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "xai_cli_chat_proxy": True,
        "xai_quota_family": XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY,
        "tags": [
            "grok-build",
            "route:grok_cli_chat_proxy",
            "auth:grok_oidc",
            "provider:xai",
            "quota:xai_grok_subscription",
            f"grok-model:{public_model}",
        ],
    }


async def prepare_oa_xai_request(data: Dict[str, Any]) -> bool:
    public_model = data.get("model")
    if not is_oa_xai_model(public_model):
        return False

    upstream_model = resolve_oa_xai_upstream_model(public_model)
    data["model"] = upstream_model
    data["api_base"] = get_secret_str("LITELLM_XAI_OAUTH_API_BASE") or XAI_API_BASE
    data["api_key"] = await get_xai_oauth_access_token()
    data["custom_llm_provider"] = "xai"

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata
    _merge_metadata(metadata, build_oa_xai_metadata(public_model, upstream_model))

    litellm_metadata = data.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        _merge_metadata(litellm_metadata, build_oa_xai_metadata(public_model, upstream_model))

    return True


def _merge_metadata(target: Dict[str, Any], incoming: Dict[str, Any]) -> None:
    incoming_tags = incoming.get("tags")
    existing_tags = target.get("tags")
    merged_tags = list(existing_tags) if isinstance(existing_tags, list) else []
    if isinstance(incoming_tags, list):
        for tag in incoming_tags:
            if isinstance(tag, str) and tag not in merged_tags:
                merged_tags.append(tag)

    for key, value in incoming.items():
        if key == "tags":
            continue
        target.setdefault(key, value)
    if merged_tags:
        target["tags"] = merged_tags


async def get_xai_oauth_access_token() -> str:
    credential_path = get_secret_str("LITELLM_XAI_OAUTH_AUTH_FILE")
    if not credential_path:
        raise ValueError(
            "xAI OAuth-managed models require LITELLM_XAI_OAUTH_AUTH_FILE to "
            "point at a LiteLLM-owned xAI OAuth credential file. Reseed or "
            "relogin the managed credential before calling oa_xai/*."
        )

    scope = get_secret_str("LITELLM_XAI_OAUTH_SCOPE") or _DEFAULT_XAI_OAUTH_SCOPE
    lock_key = f"{credential_path}:{scope}"
    lock = _refresh_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        return await _get_xai_oauth_access_token_locked(
            credential_path=Path(credential_path),
            scope=scope,
            is_grok_native_oauth=False,
        )


async def get_grok_native_oauth_access_token() -> str:
    credential_path = default_grok_xai_oauth_auth_path()
    scope = (
        get_secret_str("LITELLM_XAI_GROK_OAUTH_SCOPE")
        or get_secret_str("LITELLM_XAI_OAUTH_SCOPE")
        or _DEFAULT_XAI_OAUTH_SCOPE
    )
    lock_key = f"grok-native:{credential_path}:{scope}"
    lock = _refresh_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        return await _get_xai_oauth_access_token_locked(
            credential_path=credential_path,
            scope=scope,
            lock_path=default_grok_xai_oauth_auth_lock_path(credential_path),
            is_grok_native_oauth=True,
        )


def default_litellm_xai_oauth_auth_path() -> Path:
    configured = get_secret_str("LITELLM_XAI_OAUTH_MIGRATED_AUTH_FILE")
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()
    return Path(_DEFAULT_LITELLM_XAI_OAUTH_AUTH_PATH).expanduser()


def default_grok_xai_oauth_auth_path() -> Path:
    configured = (
        get_secret_str("LITELLM_XAI_GROK_AUTH_FILE")
        or get_secret_str("LITELLM_XAI_OAUTH_GROK_AUTH_FILE")
        or get_secret_str("GROK_AUTH_FILE")
    )
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()

    grok_home = get_secret_str("GROK_HOME")
    if isinstance(grok_home, str) and grok_home.strip():
        return Path(grok_home.strip()).expanduser() / "auth.json"

    return Path(_DEFAULT_GROK_XAI_OAUTH_AUTH_PATH).expanduser()


def default_grok_xai_oauth_seed_auth_path() -> Optional[Path]:
    configured = get_secret_str("LITELLM_XAI_GROK_SEED_AUTH_FILE")
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()
    return None


def default_grok_xai_oauth_auth_lock_path(credential_path: Path) -> Path:
    configured = get_secret_str("LITELLM_XAI_GROK_AUTH_LOCK_FILE")
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()
    return credential_path.with_name(f"{credential_path.name}.lock")


def migrate_hermes_xai_oauth_credential(
    *,
    hermes_auth_file: Optional[Path] = None,
    target_auth_file: Optional[Path] = None,
    scope: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    source_path = (
        hermes_auth_file
        or Path(
            get_secret_str("LITELLM_XAI_OAUTH_HERMES_AUTH_FILE")
            or _DEFAULT_HERMES_AUTH_PATH
        )
    ).expanduser()
    target_path = (
        target_auth_file or default_litellm_xai_oauth_auth_path()
    ).expanduser()
    _validate_xai_oauth_migration_target(source_path, target_path)

    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"xAI OAuth target credential already exists at {target_path}. "
            "Pass overwrite=True or choose a different LiteLLM-owned path."
        )

    hermes_payload = _read_json_object(
        source_path,
        description="Hermes auth file",
    )
    credential_scope = (
        scope
        or get_secret_str("LITELLM_XAI_OAUTH_SCOPE")
        or _DEFAULT_XAI_OAUTH_SCOPE
    )
    credential = _build_litellm_xai_oauth_record_from_hermes(
        hermes_payload,
        scope=credential_scope,
    )
    _write_credential_payload(target_path, {credential_scope: credential})
    return target_path


def _validate_xai_oauth_migration_target(source_path: Path, target_path: Path) -> None:
    resolved_source_parent = source_path.expanduser().resolve().parent
    resolved_target = target_path.expanduser().resolve()
    if any(part == ".hermes" for part in resolved_target.parts):
        raise ValueError(
            "xAI OAuth migration target must be outside the user's .hermes directory."
        )
    if resolved_target == source_path.expanduser().resolve():
        raise ValueError("xAI OAuth migration target cannot be the Hermes source file.")
    try:
        resolved_target.relative_to(resolved_source_parent)
    except ValueError:
        return
    raise ValueError(
        "xAI OAuth migration target must be outside the Hermes auth directory."
    )


def _read_json_object(path: Path, *, description: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"{description} not found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} at {path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object.")
    return payload


def _build_litellm_xai_oauth_record_from_hermes(
    payload: Dict[str, Any],
    *,
    scope: str,
) -> Dict[str, Any]:
    provider = _extract_hermes_xai_oauth_provider(payload)
    provider_tokens = provider.get("tokens") if isinstance(provider, dict) else None
    credential = (
        _record_from_hermes_provider_tokens(provider, provider_tokens, scope=scope)
        if isinstance(provider_tokens, dict)
        else None
    )
    if credential is None:
        credential = _record_from_hermes_credential_pool(payload, scope=scope)
    if credential is None:
        raise ValueError(
            "Hermes auth file does not contain a usable xai-oauth credential."
        )
    return credential


def _extract_hermes_xai_oauth_provider(payload: Dict[str, Any]) -> Dict[str, Any]:
    providers = payload.get("providers")
    if isinstance(providers, dict):
        provider = providers.get(_DEFAULT_HERMES_XAI_OAUTH_PROVIDER_ID)
        if isinstance(provider, dict):
            return provider
    return {}


def _record_from_hermes_provider_tokens(
    provider: Dict[str, Any],
    tokens: Dict[str, Any],
    *,
    scope: str,
) -> Optional[Dict[str, Any]]:
    access_token = _clean_oauth_string(tokens.get("access_token"))
    refresh_token = _clean_oauth_string(tokens.get("refresh_token"))
    if not access_token and not refresh_token:
        return None

    record = _base_xai_oauth_record(scope)
    _copy_oauth_token_fields(record, tokens)

    discovery = provider.get("discovery")
    if isinstance(discovery, dict):
        token_endpoint = _clean_oauth_string(discovery.get("token_endpoint"))
        if token_endpoint:
            record["token_endpoint"] = token_endpoint

    last_refresh = _parse_expires_at(provider.get("last_refresh"))
    expires_in = tokens.get("expires_in")
    if last_refresh is not None and isinstance(expires_in, (int, float)):
        expires_at = last_refresh + timedelta(seconds=float(expires_in))
        record["expires_at"] = expires_at.isoformat().replace("+00:00", "Z")

    auth_mode = _clean_oauth_string(provider.get("auth_mode"))
    if auth_mode:
        record["source_auth_mode"] = auth_mode
    redirect_uri = _clean_oauth_string(provider.get("redirect_uri"))
    if redirect_uri:
        record["redirect_uri"] = redirect_uri
    record["source"] = "hermes.providers.xai-oauth"
    return record


def _record_from_hermes_credential_pool(
    payload: Dict[str, Any],
    *,
    scope: str,
) -> Optional[Dict[str, Any]]:
    credential_pool = payload.get("credential_pool")
    if not isinstance(credential_pool, dict):
        return None
    pool = credential_pool.get(_DEFAULT_HERMES_XAI_OAUTH_PROVIDER_ID)
    if not isinstance(pool, list):
        return None

    for item in pool:
        if not isinstance(item, dict):
            continue
        access_token = _clean_oauth_string(item.get("access_token"))
        refresh_token = _clean_oauth_string(item.get("refresh_token"))
        if not access_token and not refresh_token:
            continue
        record = _base_xai_oauth_record(scope)
        _copy_oauth_token_fields(record, item)
        base_url = _clean_oauth_string(item.get("base_url"))
        if base_url:
            record["source_base_url"] = base_url
        last_refresh = _parse_expires_at(item.get("last_refresh"))
        if last_refresh is not None:
            record["source_last_refresh"] = last_refresh.isoformat().replace(
                "+00:00",
                "Z",
            )
        source = _clean_oauth_string(item.get("source"))
        record["source"] = source or "hermes.credential_pool.xai-oauth"
        return record
    return None


def _base_xai_oauth_record(scope: str) -> Dict[str, Any]:
    client_id = _clean_oauth_string(
        scope.rsplit("::", 1)[-1] if "::" in scope else None
    )
    record: Dict[str, Any] = {
        "token_endpoint": _DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT,
    }
    if client_id:
        record["oidc_client_id"] = client_id
    return record


def _copy_oauth_token_fields(record: Dict[str, Any], source: Dict[str, Any]) -> None:
    access_token = _clean_oauth_string(source.get("access_token"))
    if access_token:
        record["key"] = access_token
        record["access_token"] = access_token
    for key in ("refresh_token", "id_token", "token_type"):
        value = _clean_oauth_string(source.get(key))
        if value:
            record[key] = value


def _clean_oauth_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _oauth_credential_subject(is_grok_native_oauth: bool) -> str:
    return (
        "the managed xAI OAuth credential"
        if not is_grok_native_oauth
        else "the Grok OIDC credential"
    )


def _oauth_refresh_action(is_grok_native_oauth: bool) -> str:
    return (
        "xAI OAuth credential refresh"
        if not is_grok_native_oauth
        else "Grok OIDC credential refresh"
    )


async def _get_xai_oauth_access_token_locked(
    *,
    credential_path: Path,
    scope: str,
    lock_path: Optional[Path] = None,
    is_grok_native_oauth: bool,
) -> str:
    with _credential_file_lock(lock_path):
        if is_grok_native_oauth:
            _sync_grok_native_oauth_seed_credential(credential_path)
        raw_payload = _read_credential_payload(credential_path)
        credential = _select_credential_record(raw_payload, scope)
        token = _credential_access_token(credential)
        if token and not _credential_needs_refresh(credential):
            return token

        refreshed = await _refresh_xai_oauth_credential(
            credential,
            is_grok_native_oauth=is_grok_native_oauth,
        )
        _update_credential_record(credential, refreshed)
        _write_credential_payload(credential_path, raw_payload)
        refreshed_token = _credential_access_token(credential)
        if refreshed_token:
            return refreshed_token
    if is_grok_native_oauth:
        raise ValueError(
            "Grok OIDC credential refresh did not return an access token. "
            "Reseed or relogin the Grok OIDC credential."
        )
    raise ValueError(
        "xAI OAuth refresh did not return an access token. Reseed or relogin "
        "the managed xAI OAuth credential before calling oa_xai/*."
    )


@contextmanager
def _credential_file_lock(lock_path: Optional[Path]) -> Iterator[None]:
    if lock_path is None:
        yield
        return

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
            f"xAI OAuth credential file not found at {credential_path}. Reseed "
            "or relogin the LiteLLM-managed xAI OAuth credential."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"xAI OAuth credential file at {credential_path} is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("xAI OAuth credential file must contain a JSON object.")
    return payload


def _sync_grok_native_oauth_seed_credential(credential_path: Path) -> None:
    seed_path = default_grok_xai_oauth_seed_auth_path()
    if seed_path is None or seed_path == credential_path:
        return

    try:
        seed_stat = seed_path.stat()
    except FileNotFoundError:
        return

    try:
        credential_stat = credential_path.stat()
    except FileNotFoundError:
        should_sync = True
    else:
        should_sync = seed_stat.st_mtime_ns > credential_stat.st_mtime_ns

    if not should_sync:
        return

    seed_payload = _read_json_object(
        seed_path,
        description="Grok OIDC seed auth file",
    )
    _write_credential_payload(credential_path, seed_payload)


def _select_credential_record(
    payload: Dict[str, Any],
    scope: str,
) -> Dict[str, Any]:
    if _looks_like_credential_record(payload):
        return payload

    scoped_record = payload.get(scope)
    if isinstance(scoped_record, dict):
        return scoped_record

    for value in payload.values():
        if isinstance(value, dict) and _looks_like_credential_record(value):
            return value

    raise ValueError(
        "xAI OAuth credential file does not contain a usable credential record. "
        "Expected a Grok-style scoped record or a flat object with key/access_token."
    )


def _looks_like_credential_record(value: Dict[str, Any]) -> bool:
    return bool(value.get("key") or value.get("access_token") or value.get("refresh_token"))


def _credential_access_token(credential: Dict[str, Any]) -> Optional[str]:
    token = credential.get("access_token") or credential.get("key")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _credential_needs_refresh(credential: Dict[str, Any]) -> bool:
    expires_at = _parse_expires_at(credential.get("expires_at"))
    if expires_at is None:
        return False
    buffer_seconds = _refresh_buffer_seconds()
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


def _refresh_buffer_seconds() -> int:
    raw_value = get_secret_str("LITELLM_XAI_OAUTH_REFRESH_BUFFER_SECONDS")
    if raw_value is None:
        return _DEFAULT_REFRESH_BUFFER_SECONDS
    try:
        return max(0, int(raw_value))
    except ValueError:
        return _DEFAULT_REFRESH_BUFFER_SECONDS


async def _refresh_xai_oauth_credential(
    credential: Dict[str, Any],
    *,
    is_grok_native_oauth: bool,
) -> Dict[str, Any]:
    refresh_token = credential.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise ValueError(
            "Grok OIDC credential is expired or near expiry and has no refresh_token. "
            "Reseed or relogin the Grok OIDC credential."
            if is_grok_native_oauth
            else "xAI OAuth credential is expired or near expiry and has no refresh_token. "
            "Reseed or relogin the managed xAI OAuth credential."
        )

    client_id = (
        get_secret_str("LITELLM_XAI_OAUTH_CLIENT_ID")
        or credential.get("oidc_client_id")
        or credential.get("client_id")
    )
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError(
            _oauth_refresh_action(is_grok_native_oauth)
            + " requires oidc_client_id or LITELLM_XAI_OAUTH_CLIENT_ID. "
            + (
                f"Reseed {_oauth_credential_subject(is_grok_native_oauth)}."
                if is_grok_native_oauth
                else "Reseed the managed credential."
            )
        )

    token_endpoint = (
        get_secret_str("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT")
        or credential.get("token_endpoint")
        or _DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT
    )
    if not isinstance(token_endpoint, str) or not token_endpoint.strip():
        raise ValueError(
            "xAI OAuth token endpoint is missing."
            if not is_grok_native_oauth
            else "Grok OIDC token endpoint is missing."
        )

    form_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token.strip(),
        "client_id": client_id.strip(),
    }
    client_secret = get_secret_str("LITELLM_XAI_OAUTH_CLIENT_SECRET") or credential.get(
        "client_secret"
    )
    if isinstance(client_secret, str) and client_secret.strip():
        form_data["client_secret"] = client_secret.strip()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                token_endpoint.strip(),
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.HTTPError as exc:
        raise ValueError(
            f"{_oauth_refresh_action(is_grok_native_oauth)} failed while contacting "
            "the token endpoint. "
            f"Reseed or relogin {_oauth_credential_subject(is_grok_native_oauth)}."
        ) from exc

    if response.status_code >= 400:
        error_hint = _extract_oauth_error_hint(response)
        raise ValueError(
            f"{_oauth_refresh_action(is_grok_native_oauth)} failed"
            + (f" ({error_hint})" if error_hint else "")
            + ". Reseed or relogin "
            + _oauth_credential_subject(is_grok_native_oauth).rstrip(".")
            + "."
        )

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("xAI OAuth token endpoint returned a non-object payload.")
    return payload


def _extract_oauth_error_hint(response: httpx.Response) -> Optional[str]:
    try:
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("error", "error_description", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _update_credential_record(
    credential: Dict[str, Any],
    refreshed: Dict[str, Any],
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


def _write_credential_payload(credential_path: Path, payload: Dict[str, Any]) -> None:
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = credential_path.with_name(
        f".{credential_path.name}.{os.getpid()}.tmp"
    )
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.chmod(0o600)
    os.replace(tmp_path, credential_path)
