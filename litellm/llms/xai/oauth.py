import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from litellm.constants import XAI_API_BASE
from litellm.secret_managers.main import get_secret_str


OA_XAI_PROVIDER_PREFIX = "oa_xai/"
XAI_OAUTH_ROUTE_FAMILY = "xai_oauth_api"
XAI_OAUTH_CREDENTIAL_FAMILY = "xai_oauth"
XAI_GROK_SUBSCRIPTION_QUOTA_FAMILY = "xai_grok_subscription"

_DEFAULT_XAI_OAUTH_SCOPE = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
_DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
_DEFAULT_REFRESH_BUFFER_SECONDS = 300

_XAI_OAUTH_MODEL_MAP = {
    "oa_xai/grok-4.3": "xai/grok-4.3",
    "oa_xai/grok-4.20-0309-reasoning": "xai/grok-4.20-0309-reasoning",
    "oa_xai/grok-4.20-0309-non-reasoning": "xai/grok-4.20-0309-non-reasoning",
    "oa_xai/grok-4.20-multi-agent-0309": "xai/grok-4.20-multi-agent-0309",
}

_refresh_locks: Dict[str, asyncio.Lock] = {}


def is_oa_xai_model(model: Any) -> bool:
    return isinstance(model, str) and model.startswith(OA_XAI_PROVIDER_PREFIX)


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
        )


async def _get_xai_oauth_access_token_locked(
    *,
    credential_path: Path,
    scope: str,
) -> str:
    raw_payload = _read_credential_payload(credential_path)
    credential = _select_credential_record(raw_payload, scope)
    token = _credential_access_token(credential)
    if token and not _credential_needs_refresh(credential):
        return token

    refreshed = await _refresh_xai_oauth_credential(credential)
    _update_credential_record(credential, refreshed)
    _write_credential_payload(credential_path, raw_payload)
    refreshed_token = _credential_access_token(credential)
    if refreshed_token:
        return refreshed_token
    raise ValueError(
        "xAI OAuth refresh did not return an access token. Reseed or relogin "
        "the managed xAI OAuth credential before calling oa_xai/*."
    )


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
) -> Dict[str, Any]:
    refresh_token = credential.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise ValueError(
            "xAI OAuth credential is expired or near expiry and has no refresh_token. "
            "Reseed or relogin the managed xAI OAuth credential."
        )

    client_id = (
        get_secret_str("LITELLM_XAI_OAUTH_CLIENT_ID")
        or credential.get("oidc_client_id")
        or credential.get("client_id")
    )
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError(
            "xAI OAuth credential refresh requires oidc_client_id or "
            "LITELLM_XAI_OAUTH_CLIENT_ID. Reseed the managed credential."
        )

    token_endpoint = (
        get_secret_str("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT")
        or credential.get("token_endpoint")
        or _DEFAULT_XAI_OAUTH_TOKEN_ENDPOINT
    )
    if not isinstance(token_endpoint, str) or not token_endpoint.strip():
        raise ValueError("xAI OAuth token endpoint is missing.")

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
            "xAI OAuth credential refresh failed while contacting the token "
            "endpoint. Reseed or relogin the managed xAI OAuth credential."
        ) from exc

    if response.status_code >= 400:
        error_hint = _extract_oauth_error_hint(response)
        raise ValueError(
            "xAI OAuth credential refresh failed"
            + (f" ({error_hint})" if error_hint else "")
            + ". Reseed or relogin the managed xAI OAuth credential."
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
    os.replace(tmp_path, credential_path)
