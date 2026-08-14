"""Codex OAuth inventory loading, token validation, and request detection.

Runtime dependency ``_get_request_header_or_passthrough_alias`` is injected via
:func:`configure_codex_oauth_runtime` (the function lives in the god module and
depends on the pass-through header prefix constant).
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import httpx
from fastapi import HTTPException, Request

from litellm.llms.chatgpt.common_utils import (
    CHATGPT_API_BASE,
    get_chatgpt_default_headers,
)
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
from litellm.secret_managers.codex_oauth_inventory import (
    CodexOAuthCredentialRecord,
    CodexOAuthCredentialSnapshot,
    CodexOAuthInventoryError,
    load_codex_oauth_credential,
    load_codex_oauth_inventory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Legacy facade symbols retained for decomposition compatibility. The active
# loader below does not consult these paths or enroll credentials from them.
_ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS = (
    "LITELLM_CODEX_AUTH_FILE",
    "CHATGPT_AUTH_FILE",
)
_ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS = (
    "LITELLM_CODEX_TOKEN_DIR",
    "CHATGPT_TOKEN_DIR",
)
_ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS = (
    "~/.codex/auth.json",
    "~/.codex/auth.json",
    "~/.config/litellm/chatgpt/auth.json",
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CodexAuthData = dict[str, object]
CodexTokenData = dict[str, object]
OAuthJsonData = dict[str, object]

# ---------------------------------------------------------------------------
# Injected runtime dependencies
# ---------------------------------------------------------------------------
_get_request_header_or_passthrough_alias: Optional[
    Callable[[Request, str], Optional[str]]
] = None


def configure_codex_oauth_runtime(
    *,
    get_request_header_or_passthrough_alias: Callable[[Request, str], Optional[str]],
) -> None:
    """Bind god-module request-header helpers after module load."""
    global _get_request_header_or_passthrough_alias
    _get_request_header_or_passthrough_alias = get_request_header_or_passthrough_alias


# ---------------------------------------------------------------------------
# Auth-value cleaning
# ---------------------------------------------------------------------------


def _clean_codex_auth_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


# ---------------------------------------------------------------------------
# Explicit inventory compatibility accessor
# ---------------------------------------------------------------------------


def _get_anthropic_adapter_codex_auth_file_path() -> Optional[Path]:
    """Compatibility accessor for the first explicit enabled inventory record."""
    inventory = load_codex_oauth_inventory()
    return inventory.select_record().auth_path


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _decode_jwt_claims_without_validation(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
    except Exception:
        return {}


def _extract_codex_account_id_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    claims = _decode_jwt_claims_without_validation(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


# ---------------------------------------------------------------------------
# Token data / validation
# ---------------------------------------------------------------------------


def _get_codex_auth_token_data(auth_data: CodexAuthData) -> CodexTokenData:
    token_data = auth_data.get("tokens")
    if isinstance(token_data, dict):
        return dict(token_data)
    return auth_data


def _get_codex_auth_token_expiry(access_token: str) -> Optional[int]:
    claims = _decode_jwt_claims_without_validation(access_token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return int(exp)
    return None


def _codex_auth_access_token_is_valid(token_data: CodexTokenData) -> bool:
    access_token = _clean_codex_auth_value(token_data.get("access_token"))
    if access_token is None:
        return False
    expires_at = token_data.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = _get_codex_auth_token_expiry(access_token)
    if not isinstance(expires_at, (int, float)):
        return True
    return time.time() < float(expires_at) - 60


# ---------------------------------------------------------------------------
# Auth data loading
# ---------------------------------------------------------------------------


async def _load_codex_auth_data_from_path(auth_path: Path) -> Optional[CodexAuthData]:
    try:
        auth_data = json.loads(auth_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(auth_data, dict):
        return None
    return auth_data


@dataclass(frozen=True)
class CodexOAuthRequestAuth:
    """Selected account metadata plus headers, with secrets hidden from repr."""

    account_label: str
    account_hash: str
    lane_key: str
    headers: dict[str, str] = field(repr=False)


def _codex_oauth_account_lane_key(
    *,
    account_label: str,
    account_hash: str,
) -> str:
    """Return a server-owned, secret-safe lane for one configured account."""
    return f"codex-oauth:{account_label}:{account_hash}"


def _codex_oauth_candidate_identity(
    candidate: dict[str, Any],
) -> Optional[dict[str, str]]:
    """Return the safe selected-account identity carried by a candidate."""
    account_label = _clean_codex_auth_value(
        candidate.get("codex_oauth_account_label")
    )
    account_hash = _clean_codex_auth_value(
        candidate.get("codex_oauth_account_hash")
    )
    lane_key = _clean_codex_auth_value(candidate.get("codex_oauth_lane_key"))
    present_count = sum(
        value is not None for value in (account_label, account_hash, lane_key)
    )
    if present_count == 0:
        return None
    if present_count != 3:
        raise HTTPException(
            status_code=500,
            detail="Selected Codex OAuth account context is incomplete.",
        )
    assert account_label is not None
    assert account_hash is not None
    assert lane_key is not None
    expected_lane = _codex_oauth_account_lane_key(
        account_label=account_label,
        account_hash=account_hash,
    )
    if lane_key != expected_lane:
        raise HTTPException(
            status_code=500,
            detail="Selected Codex OAuth account lane is invalid.",
        )
    return {
        "account_label": account_label,
        "account_hash": account_hash,
        "lane_key": lane_key,
    }


def _bind_codex_oauth_candidate_to_request(
    request: Request,
    candidate: dict[str, Any],
) -> Optional[dict[str, str]]:
    """Bind only the safe selected-account identity to this request."""
    identity = _codex_oauth_candidate_identity(candidate)
    if identity is None:
        setattr(request.state, "aawm_codex_oauth_selected_account", None)
        return None
    bound = {
        **identity,
        "model": str(candidate.get("model") or ""),
    }
    setattr(request.state, "aawm_codex_oauth_selected_account", bound)
    return dict(bound)


def _get_bound_codex_oauth_candidate_identity(
    request: Request,
) -> Optional[dict[str, str]]:
    bound = getattr(request.state, "aawm_codex_oauth_selected_account", None)
    if not isinstance(bound, dict):
        return None
    candidate = {
        "codex_oauth_account_label": bound.get("account_label"),
        "codex_oauth_account_hash": bound.get("account_hash"),
        "codex_oauth_lane_key": bound.get("lane_key"),
    }
    identity = _codex_oauth_candidate_identity(candidate)
    if identity is None:
        return None
    identity["model"] = str(bound.get("model") or "")
    return identity


def _codex_oauth_responses_target_url() -> str:
    """Return the OAuth-only ChatGPT Codex Responses target."""
    return f"{(os.getenv('CHATGPT_API_BASE') or CHATGPT_API_BASE).rstrip('/')}/responses"


def _codex_oauth_credential_snapshot_is_valid(
    credential: CodexOAuthCredentialSnapshot,
) -> bool:
    if credential.expires_at is None:
        return True
    return time.time() < credential.expires_at - 60


async def _load_codex_oauth_headers_for_record(
    request: Request,
    record: CodexOAuthCredentialRecord,
) -> CodexOAuthRequestAuth:
    """Build headers only from one already-selected immutable record."""
    try:
        credential = load_codex_oauth_credential(record)
    except CodexOAuthInventoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None

    if not _codex_oauth_credential_snapshot_is_valid(credential):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Codex OAuth credential '{record.label}' "
                f"(account_hash={credential.account_hash}) is expired or "
                "invalid. The "
                "provider-status sidecar owns Codex auth refresh; confirm the "
                "configured account can be refreshed."
            ),
        )

    headers = _safe_get_request_headers(request)
    assert _get_request_header_or_passthrough_alias is not None
    session_id = (
        _get_request_header_or_passthrough_alias(request, "session_id")
        or headers.get("x-claude-code-session-id")
        or headers.get("X-Claude-Code-Session-Id")
    )

    return CodexOAuthRequestAuth(
        account_label=record.label,
        account_hash=credential.account_hash,
        lane_key=_codex_oauth_account_lane_key(
            account_label=record.label,
            account_hash=credential.account_hash,
        ),
        headers=get_chatgpt_default_headers(
            access_token=credential.access_token,
            account_id=credential.account_id,
            session_id=session_id,
        ),
    )


async def _load_local_codex_auth_selection(
    request: Request,
    *,
    account_label: Optional[str] = None,
    model: Optional[str] = None,
) -> CodexOAuthRequestAuth:
    """Select from the explicit inventory and load exactly that record."""
    try:
        inventory = load_codex_oauth_inventory()
        record = inventory.select_record(label=account_label, model=model)
    except CodexOAuthInventoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None
    return await _load_codex_oauth_headers_for_record(request, record)


async def _load_bound_codex_oauth_auth(
    request: Request,
) -> CodexOAuthRequestAuth:
    """Load exactly the server-selected account or fail closed without secrets."""
    identity = _get_bound_codex_oauth_candidate_identity(request)
    if identity is None:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": (
                        "Codex OAuth dispatch requires a server-selected "
                        "configured account."
                    ),
                    "type": "rate_limit_error",
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                },
                "failure_phase": "pre_dispatch_auth",
                "attempted_provider_call": False,
            },
        )
    try:
        selection = await _load_local_codex_auth_selection(
            request,
            account_label=identity["account_label"],
            model=identity["model"] or None,
        )
    except HTTPException:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": (
                        "Selected Codex OAuth account is not currently "
                        "authentication-ready."
                    ),
                    "type": "rate_limit_error",
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                },
                "account": identity,
                "failure_phase": "pre_dispatch_auth",
                "attempted_provider_call": False,
            },
        ) from None
    if (
        selection.account_hash != identity["account_hash"]
        or selection.lane_key != identity["lane_key"]
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": (
                        "Selected Codex OAuth account identity changed before "
                        "dispatch."
                    ),
                    "type": "rate_limit_error",
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                },
                "account": identity,
                "failure_phase": "pre_dispatch_auth",
                "attempted_provider_call": False,
            },
        )
    return selection


async def _load_local_codex_auth_headers(request: Request) -> dict[str, str]:
    """Compatibility wrapper for the current first-eligible account consumer."""
    selection = await _load_local_codex_auth_selection(request)
    return dict(selection.headers)



# ---------------------------------------------------------------------------
# Direct concrete Responses inventory binding (OPENAI-006)
# ---------------------------------------------------------------------------


def _endpoint_is_openai_responses_path(endpoint: str) -> bool:
    endpoint_path = httpx.URL(endpoint).path.rstrip("/")
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    return (
        endpoint_path == "/responses"
        or endpoint_path == "/v1/responses"
        or endpoint_path.startswith("/responses/")
        or endpoint_path.startswith("/v1/responses/")
    )


def _normalize_supported_endpoint_set(endpoints: Any) -> set[str]:
    if not isinstance(endpoints, (list, tuple, set)):
        return set()
    normalized: set[str] = set()
    for endpoint in endpoints:
        cleaned = _clean_codex_auth_value(endpoint)
        if cleaned is None:
            continue
        path = cleaned if cleaned.startswith("/") else f"/{cleaned}"
        normalized.add(path.rstrip("/") or "/")
    return normalized


def _load_direct_codex_oauth_model_cost_maps() -> tuple[dict[str, Any], ...]:
    """Return configured model catalogs without hardcoding target names."""
    maps: list[dict[str, Any]] = []
    try:
        from litellm.proxy.pass_through_endpoints.aawm_request_policy.codex_tool_policy import (
            load_bundled_model_cost_map_for_codex_policy,
        )

        bundled = load_bundled_model_cost_map_for_codex_policy()
        if isinstance(bundled, dict) and bundled:
            maps.append(bundled)
    except Exception:
        pass
    try:
        import litellm as _litellm

        live = getattr(_litellm, "model_cost", None)
        if isinstance(live, dict) and live:
            maps.append(live)
    except Exception:
        pass
    return tuple(maps)


def _iter_direct_codex_oauth_model_info_entries(
    model: str,
) -> list[dict[str, Any]]:
    cleaned = _clean_codex_auth_value(model)
    if cleaned is None:
        return []
    bare = cleaned.split("/", 1)[-1]
    lookup_keys = (cleaned, bare, f"chatgpt/{bare}")
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cost_map in _load_direct_codex_oauth_model_cost_maps():
        for key in lookup_keys:
            info = cost_map.get(key)
            if not isinstance(info, dict):
                continue
            marker = id(info)
            if marker in seen:
                continue
            seen.add(marker)
            entries.append(info)
    return entries


def _model_info_is_chatgpt_codex_inventory_target(info: Mapping[str, Any]) -> bool:
    """Classify ChatGPT/Codex inventory targets from structured model metadata."""
    provider = str(info.get("litellm_provider") or "").strip().lower()
    if provider == "chatgpt":
        return True
    if provider != "openai":
        return False
    endpoints = _normalize_supported_endpoint_set(info.get("supported_endpoints"))
    if "/v1/responses" not in endpoints and "/responses" not in endpoints:
        return False
    # Dual public chat+responses OpenAI API models stay on API-key routes.
    # Codex/ChatGPT inventory targets are responses-catalogued without chat
    # completions (gpt-5.6-*, exclusive codex rows, chatgpt-provider twins).
    has_chat_completions = any(
        endpoint == "/v1/chat/completions" or endpoint.endswith("/chat/completions")
        for endpoint in endpoints
    )
    return not has_chat_completions


def _snapshot_lists_openai_codex_responses_model(model: str) -> bool:
    """True when the active alias snapshot lists model on openai/codex_responses."""
    cleaned = _clean_codex_auth_value(model)
    if cleaned is None:
        return False
    bare = cleaned.split("/", 1)[-1]
    try:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
            get_active_routing_snapshot,
        )
    except Exception:
        return False
    try:
        snapshot = get_active_routing_snapshot()
    except Exception:
        return False
    if snapshot is None:
        return False
    aliases = getattr(snapshot, "aliases", None)
    if not isinstance(aliases, Mapping):
        return False
    for alias in aliases.values():
        candidates = getattr(alias, "candidates", ()) or ()
        for candidate in candidates:
            provider = str(getattr(candidate, "provider", "") or "").strip().lower()
            route_family = str(
                getattr(candidate, "route_family", "") or ""
            ).strip().lower()
            candidate_model = _clean_codex_auth_value(getattr(candidate, "model", None))
            if provider != "openai" or route_family != "codex_responses":
                continue
            if candidate_model is None:
                continue
            if candidate_model == cleaned or candidate_model.split("/", 1)[-1] == bare:
                return True
    return False


def _is_direct_codex_oauth_inventory_model(model: Any) -> bool:
    """True when model is a configured ChatGPT/Codex inventory target."""
    cleaned = _clean_codex_auth_value(model)
    if cleaned is None:
        return False
    if _snapshot_lists_openai_codex_responses_model(cleaned):
        return True
    bare = cleaned.split("/", 1)[-1]
    # A chatgpt/* catalog twin is authoritative ChatGPT/Codex inventory metadata
    # even when the bare OpenAI row also advertises dual public chat endpoints.
    for cost_map in _load_direct_codex_oauth_model_cost_maps():
        twin = cost_map.get(f"chatgpt/{bare}")
        if isinstance(twin, dict) and _model_info_is_chatgpt_codex_inventory_target(
            twin
        ):
            return True
    for info in _iter_direct_codex_oauth_model_info_entries(cleaned):
        if _model_info_is_chatgpt_codex_inventory_target(info):
            return True
    return False


def _direct_codex_oauth_concrete_model_name(model: Any) -> Optional[str]:
    """Return cleaned model when it is a ChatGPT/Codex inventory target."""
    cleaned = _clean_codex_auth_value(model)
    if cleaned is None:
        return None
    if _is_direct_codex_oauth_inventory_model(cleaned):
        return cleaned
    return None


def _should_bind_direct_codex_oauth_inventory(
    request: Request,
    *,
    endpoint: str,
    request_body: Optional[dict[str, Any]] = None,
) -> bool:
    """True when a direct Responses request must use server Codex OAuth inventory.

    - Explicit inventory-classified models always bind.
    - When inventory is configured, every explicit Responses model binds.
    - Model-less requests bind only under the Codex-native auth contract.
    """
    if not _endpoint_is_openai_responses_path(endpoint):
        return False
    body = request_body if isinstance(request_body, dict) else {}
    cleaned_model = _clean_codex_auth_value(body.get("model"))
    if cleaned_model is not None:
        return _is_direct_codex_oauth_inventory_model(cleaned_model) or bool(
            os.getenv("LITELLM_CODEX_OAUTH_INVENTORY", "").strip()
        )
    # Model-less: only the established Codex-native auth contract is non-model-scoped.
    return _request_uses_codex_native_auth(request)


def _direct_codex_oauth_affinity_from_session_owner(
    owner_affinity: Optional[dict[str, Any]],
    *,
    model: str,
) -> Optional[dict[str, Any]]:
    """Keep only openai/codex account pins from durable session ownership."""
    if not isinstance(owner_affinity, dict):
        return None
    provider = str(owner_affinity.get("provider") or "").strip().lower()
    route_family = str(owner_affinity.get("route_family") or "").strip().lower()
    if provider and provider not in {"openai"}:
        return None
    if route_family and route_family not in {
        "codex_oauth",
        "codex_responses",
        "openai_responses",
    }:
        return None
    label = _clean_codex_auth_value(owner_affinity.get("codex_oauth_account_label"))
    account_hash = _clean_codex_auth_value(
        owner_affinity.get("codex_oauth_account_hash")
    )
    lane_key = _clean_codex_auth_value(owner_affinity.get("codex_oauth_lane_key"))
    if not all((label, account_hash, lane_key)):
        return None
    return {
        "provider": "openai",
        "model": str(owner_affinity.get("model") or model),
        "route_family": str(owner_affinity.get("route_family") or "codex_responses"),
        "last_resort": False,
        "codex_oauth_account_label": label,
        "codex_oauth_account_hash": account_hash,
        "codex_oauth_lane_key": lane_key,
        "affinity_state_source": owner_affinity.get("affinity_state_source")
        or "session_owner",
    }



async def _resolve_model_less_direct_codex_oauth_contexts(
    request: Request,
    *,
    candidate_template: dict[str, Any],
) -> list[dict[str, Any]]:
    """Auth-check enabled inventory accounts without a model scope filter."""
    # Local helpers (this module).
    from litellm.secret_managers.codex_oauth_inventory import (
        CodexOAuthInventoryError,
        load_codex_oauth_inventory,
    )

    try:
        inventory = load_codex_oauth_inventory()
        records = inventory.ordered_records(enabled_only=True, model=None)
    except CodexOAuthInventoryError:
        records = ()

    if not records:
        return [
            {
                "candidate": dict(candidate_template),
                "lane_key": "codex-oauth:unavailable",
                "auth_status": "degraded",
                "skip_reason": "auth_degraded",
                "failure_phase": "account_inventory_unavailable",
                "attempted_provider_call": False,
            }
        ]

    contexts: list[dict[str, Any]] = []
    for record in records:
        lane_key = _codex_oauth_account_lane_key(
            account_label=record.label,
            account_hash=record.expected_account_hash,
        )
        account_candidate = {
            **candidate_template,
            "codex_oauth_account_label": record.label,
            "codex_oauth_account_hash": record.expected_account_hash,
            "codex_oauth_lane_key": lane_key,
            "codex_oauth_account_priority": record.priority,
            "codex_oauth_account_weight": record.weight,
        }
        context: dict[str, Any] = {
            "candidate": account_candidate,
            "lane_key": lane_key,
            "auth_status": "healthy",
        }
        try:
            loaded = await _load_codex_oauth_headers_for_record(request, record)
        except HTTPException:
            context.update(
                {
                    "auth_status": "degraded",
                    "skip_reason": "auth_degraded",
                    "failure_phase": "pre_dispatch_auth",
                    "attempted_provider_call": False,
                }
            )
        else:
            if (
                loaded.account_hash != record.expected_account_hash
                or loaded.lane_key != lane_key
            ):
                context.update(
                    {
                        "auth_status": "degraded",
                        "skip_reason": "auth_degraded",
                        "failure_phase": "account_identity_mismatch",
                        "attempted_provider_call": False,
                    }
                )
        contexts.append(context)
    return contexts


async def select_and_bind_direct_codex_oauth_inventory(  # noqa: PLR0915
    request: Request,
    *,
    request_body: Optional[dict[str, Any]] = None,
) -> tuple["CodexOAuthRequestAuth", dict[str, Any], dict[str, Any]]:
    """Bind direct concrete Responses traffic to enabled account inventory.

    Reuses the alias-path account context, quota, cooldown, and failover
    machinery. Returns ``(selected_auth, selection_state, metadata_body)``.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        CODEX_AUTO_AGENT_NATIVE_PROVIDER,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        selection as _selection,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        session_affinity as _sa,
    )
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _merge_litellm_metadata,
        _safe_set_request_parsed_body,
    )

    body = dict(request_body) if isinstance(request_body, dict) else {}
    explicit_model = _clean_codex_auth_value(body.get("model"))
    native_auth = _request_uses_codex_native_auth(request)
    if explicit_model is not None:
        model = explicit_model
        inventory_model: Optional[str] = explicit_model
    else:
        if not native_auth:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Direct Codex OAuth inventory binding requires an explicit "
                    "model unless Codex-native auth markers are present."
                ),
            )
        # Model-less native contract: account selection is not model-scoped.
        model = ""
        inventory_model = None

    candidate_template: dict[str, Any] = {
        "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": model or "codex_native",
        "route_family": "codex_responses",
        "last_resort": False,
        "selection_priority": 100,
    }

    affinity: Optional[dict[str, Any]] = None
    session_identity = _sa.resolve_canonical_session_identity(request, body)
    if session_identity is not None:
        owner_record, _cache_key, owner_error = await _sa.get_session_owner_record(
            session_identity=session_identity,
        )
        if owner_error is None and isinstance(owner_record, dict):
            affinity = _direct_codex_oauth_affinity_from_session_owner(
                _sa.owner_record_as_affinity_hint(owner_record),
                model=model,
            )

    # Body/request metadata pin (continuation metadata) when no owner pin.
    if affinity is None:
        metadata = body.get("litellm_metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        pin_label = _clean_codex_auth_value(
            meta.get("codex_oauth_account_label")
            or meta.get("codex_auto_agent_selected_account_label")
            or body.get("codex_oauth_account_label")
        )
        pin_hash = _clean_codex_auth_value(
            meta.get("codex_oauth_account_hash")
            or meta.get("codex_auto_agent_selected_account_hash")
            or body.get("codex_oauth_account_hash")
        )
        pin_lane = _clean_codex_auth_value(
            meta.get("codex_oauth_lane_key")
            or meta.get("codex_auto_agent_selected_account_lane")
            or body.get("codex_oauth_lane_key")
        )
        if all((pin_label, pin_hash, pin_lane)):
            affinity = {
                "provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
                "model": model,
                "route_family": "codex_responses",
                "last_resort": False,
                "codex_oauth_account_label": pin_label,
                "codex_oauth_account_hash": pin_hash,
                "codex_oauth_lane_key": pin_lane,
                "affinity_state_source": "request_metadata",
            }

    if inventory_model is None and affinity is None:
        # Model-less native path: reuse inventory model=None eligibility
        # (selection resolve stringifies model and would filter every record).
        contexts = await _resolve_model_less_direct_codex_oauth_contexts(
            request,
            candidate_template=candidate_template,
        )
    else:
        resolve_template = dict(candidate_template)
        if inventory_model is not None:
            resolve_template["model"] = inventory_model
        elif affinity is not None:
            pinned_model = _clean_codex_auth_value(affinity.get("model"))
            if pinned_model is not None:
                resolve_template["model"] = pinned_model
        contexts = await _selection._resolve_codex_oauth_account_candidate_contexts(
            request,
            candidate_template=resolve_template,
            affinity=affinity,
        )
    await _selection._hydrate_codex_oauth_quota_observations(contexts)

    states: list[dict[str, Any]] = []
    for context in contexts:
        state = await _selection._build_codex_auto_agent_candidate_state(
            request,
            candidate_template=context["candidate"],
            openai_lane_key=context.get("lane_key"),
        )
        states.append(
            _selection._apply_codex_oauth_account_context_to_state(
                request,
                state,
                context=context,
            )
        )

    skipped = _selection._build_auto_agent_skipped_candidates_from_states(states)
    selected_state = _selection._select_first_available_codex_oauth_account_state(
        states
    )

    if selected_state is None:
        detail: dict[str, Any] = {
            "error": {
                "message": (
                    "All enabled Codex OAuth inventory accounts are currently "
                    "cooled down or unavailable for direct Responses traffic."
                ),
                "type": "rate_limit_error",
                "code": "aawm_codex_oauth_direct_inventory_unavailable",
            },
            "skipped_candidates": skipped,
            "attempted_provider_call": False,
            "failure_phase": "direct_inventory_selection",
        }
        if affinity is not None:
            detail["account"] = {
                "account_label": affinity.get("codex_oauth_account_label"),
                "account_hash": affinity.get("codex_oauth_account_hash"),
                "account_lane": affinity.get("codex_oauth_lane_key"),
            }
        raise HTTPException(status_code=429, detail=detail)

    candidate = dict(selected_state["candidate"])
    bound = _bind_codex_oauth_candidate_to_request(request, candidate)
    if bound is None:
        raise HTTPException(
            status_code=500,
            detail="Selected Codex OAuth account context is incomplete.",
        )
    selected_auth = await _load_bound_codex_oauth_auth(request)

    if int(selected_state.get("failover_ordinal") or 0) > 0:
        selection_reason = "codex_oauth_account_failover"
    elif affinity is not None and affinity.get("affinity_state_source") == "session_owner":
        selection_reason = "session_owner_pin"
    elif (
        affinity is not None
        and affinity.get("affinity_state_source") == "request_metadata"
    ):
        selection_reason = "request_metadata_pin"
    else:
        selection_reason = "direct_inventory_first_available"
    selection_state = {
        **selected_state,
        "selection_reason": selection_reason,
        "skipped": skipped,
        "alias_model": model,
        "request_mode": (
            "ordinary_continuation" if affinity is not None else "fresh"
        ),
    }

    metadata_body = _merge_litellm_metadata(
        body,
        tags_to_add=[
            "codex-oauth-direct-inventory",
            "route:codex_responses",
            f"codex-oauth-account:{selected_auth.account_label}",
        ],
        extra_fields={
            "openai_passthrough_route_family": "codex_responses",
            "codex_oauth_direct_inventory": True,
            "codex_oauth_account_label": selected_auth.account_label,
            "codex_oauth_account_hash": selected_auth.account_hash,
            "codex_oauth_lane_key": selected_auth.lane_key,
            "codex_auto_agent_selected_provider": CODEX_AUTO_AGENT_NATIVE_PROVIDER,
            "codex_auto_agent_selected_model": model,
            "codex_auto_agent_selected_route_family": "codex_responses",
            "codex_auto_agent_selected_account_label": selected_auth.account_label,
            "codex_auto_agent_selected_account_hash": selected_auth.account_hash,
            "codex_auto_agent_selected_account_lane": selected_auth.lane_key,
            "codex_auto_agent_lane_key": selected_state.get("lane_key")
            or selected_auth.lane_key,
            "codex_auto_agent_selection_reason": selection_reason,
            "codex_auto_agent_cooldown_state_source": selected_state.get(
                "cooldown_state_source"
            ),
            "codex_auto_agent_quota_snapshot_age_seconds": selected_state.get(
                "quota_snapshot_age_seconds"
            ),
            "codex_auto_agent_failover_ordinal": selected_state.get(
                "failover_ordinal"
            ),
            "codex_auto_agent_prior_account_outcome": selected_state.get(
                "prior_account_outcome"
            ),
            "codex_auto_agent_terminal_reset": selected_state.get(
                "terminal_reset"
            ),
            "codex_auto_agent_skipped_candidates": skipped,
            "codex_auto_agent_attempts": [
                _selection._codex_auto_agent_candidate_public_shape(
                    candidate,
                    lane_key=selected_state.get("lane_key"),
                    reason=selection_reason,
                )
            ],
        },
    )
    _safe_set_request_parsed_body(request, metadata_body)
    setattr(request.state, "aawm_direct_codex_oauth_inventory", True)
    return selected_auth, selection_state, metadata_body


def is_direct_codex_usage_limit_error(exc: HTTPException) -> bool:
    """Return whether a direct Codex request hit account quota exhaustion."""
    if exc.status_code != 429 or not isinstance(exc.detail, dict):
        return False
    error = exc.detail.get("error")
    return bool(
        isinstance(error, dict)
        and error.get("code") == "usage_limit_reached"
        and exc.detail.get("failover_disposition") == "usage_limit_reached"
    )


def direct_codex_usage_limit_retry_after_seconds(
    exc: HTTPException,
    *,
    now_epoch: Optional[float] = None,
) -> float:
    """Resolve the provider reset interval, with a bounded fallback."""
    headers = exc.headers or {}
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    try:
        parsed_retry_after = float(retry_after)
    except (TypeError, ValueError):
        parsed_retry_after = 0.0
    if parsed_retry_after > 0:
        return parsed_retry_after

    detail = exc.detail if isinstance(exc.detail, dict) else {}
    quota = detail.get("quota")
    if isinstance(quota, dict):
        try:
            resets_in_seconds = float(quota.get("resets_in_seconds"))
        except (TypeError, ValueError):
            resets_in_seconds = 0.0
        if resets_in_seconds > 0:
            return resets_in_seconds
        try:
            resets_at = float(quota.get("resets_at"))
        except (TypeError, ValueError):
            resets_at = 0.0
        if resets_at > 0:
            return max(1.0, resets_at - (time.time() if now_epoch is None else now_epoch))
    return 300.0


# ---------------------------------------------------------------------------
# Codex-native-auth request detection
# ---------------------------------------------------------------------------


def _anthropic_adapter_request_uses_codex_native_auth(request: Request) -> bool:
    assert _get_request_header_or_passthrough_alias is not None
    chatgpt_account_id = _get_request_header_or_passthrough_alias(request, "ChatGPT-Account-Id")
    originator = _get_request_header_or_passthrough_alias(request, "originator")
    user_agent = _get_request_header_or_passthrough_alias(request, "user-agent")
    session_id = _get_request_header_or_passthrough_alias(request, "session_id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


def _anthropic_adapter_request_has_openai_client_auth(request: Request) -> bool:
    # On the Anthropic route, direct Authorization headers are typically Anthropic auth
    # from Claude clients, not OpenAI/Codex credentials. Treat direct auth as OpenAI
    # client auth only when the request also carries Codex-native request markers.
    assert _get_request_header_or_passthrough_alias is not None
    if _get_request_header_or_passthrough_alias(
        request, "x-pass-authorization"
    ) or _get_request_header_or_passthrough_alias(request, "x-pass-api-key"):
        return True

    if _anthropic_adapter_request_uses_codex_native_auth(request):
        return bool(
            _get_request_header_or_passthrough_alias(request, "authorization")
            or _get_request_header_or_passthrough_alias(request, "api-key")
        )

    return False


def _anthropic_adapter_should_forward_direct_auth_headers(request: Request) -> bool:
    return _anthropic_adapter_request_has_openai_client_auth(request)


def _request_uses_codex_native_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    chatgpt_account_id = headers.get("chatgpt-account-id") or headers.get("ChatGPT-Account-Id")
    originator = headers.get("originator") or headers.get("Originator")
    user_agent = headers.get("user-agent") or headers.get("User-Agent")
    session_id = headers.get("session_id") or headers.get("Session_Id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


# ---------------------------------------------------------------------------
# OAuth error helpers
# ---------------------------------------------------------------------------


def _get_oauth_token_error_code(response: httpx.Response) -> Optional[str]:
    try:
        response_body = response.json()
    except ValueError:
        return None
    if not isinstance(response_body, dict):
        return None
    return _clean_codex_auth_value(response_body.get("error"))


def _format_oauth_refresh_failure_detail(
    *,
    provider_label: str,
    response: httpx.Response,
) -> str:
    error_code = _get_oauth_token_error_code(response)
    suffix = f"status={response.status_code}, error={error_code}" if error_code else f"status={response.status_code}"
    return (
        f"Failed to refresh {provider_label} OAuth access token ({suffix}). "
        f"Re-authenticate {provider_label} CLI or configure valid OAuth client "
        "environment overrides."
    )
