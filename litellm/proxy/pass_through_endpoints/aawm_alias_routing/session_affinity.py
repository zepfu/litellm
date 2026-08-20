"""D1-612: Redis-authoritative tokenized session ownership.

Canonical session identity alone keys one durable ownership record. Alias,
provider, model, route family, credential/account lane, and ingress are record
attributes, never key components.

Lifecycle (every alias / direct / nested path must use this guard):

1. ``guard_session_owner_before_egress`` — before any upstream send or client
   first-byte path: fail closed on Redis uncertainty, owner mismatch, removed
   owner, or a competing reservation; NX-reserve unowned sessions with a
   reservation token; renew our own live reservation.
2. ``promote_session_owner_reservation`` — on authoritative success / first
   byte only, CAS-promote reserved → immutable owned. Requires complete
   provider/model/route/endpoint/state-format and account identity for
   account-scoped routes. Owned records are persistent (no fixed nonrenewing
   6h expiry); reserved records use a short renewable TTL.
3. ``release_session_owner_reservation`` — on failure/terminal error, delete
   only our still-reserved tokenized hold. Never erases an owned record.

Process-local state never authorizes ownership. Conflicts and errors raise
structured ``redispatch_required`` (never ignored).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.credential_error_sanitizer import (
    sanitize_credential_error_message,
)

from . import durable
from .types import Payload


class SessionOwnerRecordState(str, Enum):
    RESERVED = "reserved"
    OWNED = "owned"


class SessionOwnerGuardDecision(str, Enum):
    NO_SESSION = "no_session"
    UNOWNED_RESERVED = "unowned_reserved"
    RESERVATION_RENEWED = "reservation_renewed"
    COMPATIBLE_OWNER = "compatible_owner"
    REDISPATCH_REQUIRED = "redispatch_required"


class SessionOwnerMutationOutcome(str, Enum):
    PROMOTED = "promoted"
    ALREADY_OWNED = "already_owned"
    RELEASED = "released"
    NOT_HELD = "not_held"
    CONFLICT = "conflict"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SessionOwnerGuardResult:
    decision: SessionOwnerGuardDecision
    session_identity: Optional[str]
    cache_key: Optional[str] = None
    reservation_token: Optional[str] = None
    owner_id: Optional[str] = None
    owner_record: Optional[Payload] = None
    mismatch_reason: Optional[str] = None
    provenance: Optional[Payload] = None
    held_reservation: bool = False


@dataclass(frozen=True)
class SessionOwnerMutationResult:
    outcome: SessionOwnerMutationOutcome
    session_identity: Optional[str]
    cache_key: Optional[str] = None
    reservation_token: Optional[str] = None
    owner_id: Optional[str] = None
    owner_record: Optional[Payload] = None
    error: Optional[str] = None


@dataclass
class SessionOwnerLease:
    """Request-scoped lease returned by the pre-egress guard."""

    session_identity: Optional[str]
    cache_key: Optional[str] = None
    reservation_token: Optional[str] = None
    owner_id: Optional[str] = None
    held_reservation: bool = False
    decision: Optional[str] = None
    attributes: Payload = field(default_factory=dict)
    promoted: bool = False
    released: bool = False


_SESSION_OWNER_STATE_KIND = "session_owner"
_SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_PREFIX = (
    "aawm-session-owner-redispatch-v1:"
)
_SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_DOMAIN_SEPARATOR = (
    "aawm-session-owner-redispatch-v1\x00"
)
_REQUEST_STATE_EFFECTIVE_SESSION_IDENTITY_ATTR = (
    "_aawm_session_owner_effective_identity"
)
_CODEX_AUTO_REVIEW_SESSION_IDENTITY_SUFFIX = ":codex-auto-review"
_REQUEST_STATE_CODEX_AUTO_REVIEW_SESSION_IDENTITY_ATTR = (
    "_aawm_codex_auto_review_session_identity"
)
_REQUEST_STATE_CODEX_AUTO_REVIEW_PARENT_SESSION_IDENTITY_ATTR = (
    "_aawm_codex_auto_review_parent_session_identity"
)
_RECORD_STATE_FIELD = "state"
_RECORD_OWNER_FIELD = "owner"
_RECORD_ATTRIBUTES_FIELD = "attributes"
_RECORD_TOKEN_FIELD = "reservation_token"
_RECORD_RESERVED_AT_FIELD = "reserved_at_epoch"
_RECORD_OWNED_AT_FIELD = "owned_at_epoch"
_RECORD_LAST_RENEWED_AT_FIELD = "last_renewed_at_epoch"

_OWNER_ATTRIBUTE_FIELDS = (
    "provider",
    "hosted_provider",
    "model",
    "route_family",
    "account_label",
    "account_hash",
    "account_lane",
    "endpoint_contract",
    "state_format",
    "credential_affinity",
    "ingress",
    "requested_model",
    "alias_family",
)

_CORE_OWNER_ATTRIBUTE_KEYS = (
    "provider",
    "hosted_provider",
    "model",
    "route_family",
    "account_label",
    "account_hash",
    "account_lane",
    "endpoint_contract",
    "state_format",
    "credential_affinity",
)

# Same-hosted-provider fields stay mutable last-used attributes. They are
# stored on the owner record but never part of hard owner identity.
_MUTABLE_SAME_HOSTED_PROVIDER_ATTRIBUTE_KEYS = (
    "model",
    "requested_model",
    "account_hash",
    "account_label",
    "account_lane",
    "credential_affinity",
)

# Canonical endpoint/state for the two equivalent managed direct-OpenAI shapes.
_MANAGED_DIRECT_OPENAI_OWNER_ID_ENDPOINT = "codex_responses"
_MANAGED_DIRECT_OPENAI_OWNER_ID_STATE = "codex_responses"

_REQUIRED_OWNER_ATTRIBUTE_KEYS = (
    "provider",
    "model",
    "route_family",
    "endpoint_contract",
    "state_format",
)

# Short renewable hold while upstream I/O is in flight. Not a fixed ownership
# expiry — owned records are persistent until explicit retirement.
_DEFAULT_RESERVATION_TTL_SECONDS = 120.0
_MIN_RESERVATION_TTL_SECONDS = 30.0
_MAX_RESERVATION_TTL_SECONDS = 900.0
_DEFAULT_RESERVATION_WAIT_TIMEOUT_SECONDS = 0.25
_DEFAULT_RESERVATION_WAIT_POLL_SECONDS = 0.025
_MAX_RESERVATION_WAIT_TIMEOUT_SECONDS = 1.0
_MAX_RESERVATION_WAIT_POLL_SECONDS = 0.1

# Account-scoped route families require credential/account identity on promote.
_ACCOUNT_SCOPED_ROUTE_MARKERS = (
    "codex_oauth",
    "chatgpt_codex",
    "openai_codex",
    "account",
)

# One-way safe hash for credential/account header identity (never the raw
# header value). Matches the established lane-key hash shape in lane_keys.py.
def _hash_account_identity_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_session_identity(session_identity: str) -> str:
    cleaned = _clean_optional_str(session_identity)
    if cleaned is None:
        raise ValueError("session_identity must be a non-empty string")
    return cleaned


def get_request_effective_session_identity(request: Any) -> Optional[str]:
    """Return the server-only redispatch identity already set on *request*."""

    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        state_values = object.__getattribute__(state, "_state")
    except AttributeError:
        state_values = None
    if isinstance(state_values, Mapping):
        return _clean_optional_str(
            state_values.get(_REQUEST_STATE_EFFECTIVE_SESSION_IDENTITY_ATTR)
        )
    try:
        value = object.__getattribute__(
            state, _REQUEST_STATE_EFFECTIVE_SESSION_IDENTITY_ATTR
        )
    except AttributeError:
        return None
    return _clean_optional_str(value)


def get_request_codex_auto_review_session_identity(request: Any) -> Optional[str]:
    """Return the server-only Codex auto-review identity set on *request*."""

    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        state_values = object.__getattribute__(state, "_state")
    except AttributeError:
        state_values = None
    if isinstance(state_values, Mapping):
        return _clean_optional_str(
            state_values.get(_REQUEST_STATE_CODEX_AUTO_REVIEW_SESSION_IDENTITY_ATTR)
        )
    try:
        value = object.__getattribute__(
            state, _REQUEST_STATE_CODEX_AUTO_REVIEW_SESSION_IDENTITY_ATTR
        )
    except AttributeError:
        return None
    return _clean_optional_str(value)


def get_request_codex_auto_review_parent_session_identity(
    request: Any,
) -> Optional[str]:
    """Return the logical parent identity for a Codex auto-review request."""

    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        state_values = object.__getattribute__(state, "_state")
    except AttributeError:
        state_values = None
    if isinstance(state_values, Mapping):
        return _clean_optional_str(
            state_values.get(
                _REQUEST_STATE_CODEX_AUTO_REVIEW_PARENT_SESSION_IDENTITY_ATTR
            )
        )
    try:
        value = object.__getattribute__(
            state, _REQUEST_STATE_CODEX_AUTO_REVIEW_PARENT_SESSION_IDENTITY_ATTR
        )
    except AttributeError:
        return None
    return _clean_optional_str(value)


def request_has_effective_session_identity(request: Any) -> bool:
    return get_request_effective_session_identity(request) is not None


def activate_codex_auto_review_session_identity(
    *,
    request: Any,
    parent_session_identity: Optional[str],
) -> Optional[str]:
    """Set one deterministic server-side identity for a Codex auto-review request."""

    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    existing_identity = get_request_codex_auto_review_session_identity(request)
    if existing_identity is not None:
        if get_request_codex_auto_review_parent_session_identity(request) is None:
            parent_identity = (
                existing_identity[
                    : -len(_CODEX_AUTO_REVIEW_SESSION_IDENTITY_SUFFIX)
                ]
                if existing_identity.endswith(
                    _CODEX_AUTO_REVIEW_SESSION_IDENTITY_SUFFIX
                )
                else existing_identity
            )
            setattr(
                state,
                _REQUEST_STATE_CODEX_AUTO_REVIEW_PARENT_SESSION_IDENTITY_ATTR,
                parent_identity or existing_identity,
            )
        return existing_identity
    parent_identity = _clean_optional_str(parent_session_identity)
    if parent_identity is None:
        return None
    if parent_identity.endswith(_CODEX_AUTO_REVIEW_SESSION_IDENTITY_SUFFIX):
        review_identity = parent_identity
        logical_parent_identity = (
            parent_identity[: -len(_CODEX_AUTO_REVIEW_SESSION_IDENTITY_SUFFIX)]
            or parent_identity
        )
    else:
        logical_parent_identity = parent_identity
        review_identity = (
            f"{parent_identity}{_CODEX_AUTO_REVIEW_SESSION_IDENTITY_SUFFIX}"
        )
    setattr(
        state,
        _REQUEST_STATE_CODEX_AUTO_REVIEW_PARENT_SESSION_IDENTITY_ATTR,
        logical_parent_identity,
    )
    setattr(
        state,
        _REQUEST_STATE_CODEX_AUTO_REVIEW_SESSION_IDENTITY_ATTR,
        review_identity,
    )
    return review_identity


def activate_session_owner_redispatch_effective_identity(
    *,
    request: Any,
    base_session_identity: Optional[str],
) -> Optional[str]:
    """Set one deterministic server-side redispatch identity for this request.

    The identity is derived only from the already resolved canonical base
    identity. It is never exposed through request headers or request bodies,
    and an existing effective identity is never derived again.
    """

    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None or get_request_effective_session_identity(request) is not None:
        return None
    base = _clean_optional_str(base_session_identity)
    if base is None:
        return None
    digest = hashlib.sha256(
        (
            _SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_DOMAIN_SEPARATOR + base
        ).encode("utf-8")
    ).hexdigest()
    effective_identity = (
        f"{_SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_PREFIX}{digest}"
    )
    setattr(state, _REQUEST_STATE_EFFECTIVE_SESSION_IDENTITY_ATTR, effective_identity)
    return effective_identity


def is_replay_safe_session_owner_redispatch_body(
    request_body: Optional[Mapping[str, Any]],
) -> bool:
    """Whether a full request body can be replayed under one new owner key."""

    return isinstance(request_body, Mapping) and "previous_response_id" not in request_body


def _strip_legacy_affinity_prefixes(raw: str) -> str:
    """Return the bare session id from legacy alias:session:lane keys."""

    cleaned = raw.strip()
    if not cleaned:
        return cleaned
    # Legacy affinity keys: "{alias_model}:{session_id}:{lane}"
    parts = cleaned.split(":")
    if len(parts) >= 3:
        # UUID-like and opaque ids commonly sit in the middle segment(s).
        # Prefer the longest middle segment that looks like a session id when
        # the key matches the alias-prefixed pattern.
        middle = ":".join(parts[1:-1]).strip()
        if middle:
            return middle
    return cleaned


def resolve_canonical_session_identity(
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    *,
    session_identity: Optional[str] = None,
) -> Optional[str]:
    """Resolve the provider-neutral canonical session identity.

    The current execution thread is authoritative when present, with the
    session id used only as a fallback. Alias/provider/model/lane prefixes on
    legacy affinity keys are stripped.
    """

    if session_identity is not None:
        cleaned = _clean_optional_str(session_identity)
        if cleaned is None:
            return None
        return _strip_legacy_affinity_prefixes(cleaned)

    effective_identity = get_request_effective_session_identity(request)
    if effective_identity is not None:
        return effective_identity

    review_identity = get_request_codex_auto_review_session_identity(request)
    if review_identity is not None:
        return review_identity

    body = request_body if isinstance(request_body, Mapping) else {}
    metadata = body.get("litellm_metadata") if isinstance(body, Mapping) else None
    if isinstance(metadata, dict):
        for key in (
            "thread_id",
            "aawm_thread_id",
            "codex_thread_id",
            "claude_thread_id",
        ):
            value = _clean_optional_str(metadata.get(key))
            if value is not None:
                return _strip_legacy_affinity_prefixes(value)

    headers = None
    if request is not None:
        headers = getattr(request, "headers", None)
    if headers is not None:
        try:
            items = headers.items()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            items = []
        header_map = {
            str(name).lower(): _clean_optional_str(value) for name, value in items
        }
        for key in (
            "thread-id",
            "x-thread-id",
            "x-aawm-thread-id",
            "x-codex-thread-id",
            "x-claude-thread-id",
        ):
            value = header_map.get(key)
            if value is not None:
                return _strip_legacy_affinity_prefixes(value)
        # Common dash/underscore variants
        for name, value in header_map.items():
            if value and name.replace("-", "_") in {
                "thread_id",
                "x_thread_id",
                "x_aawm_thread_id",
                "x_codex_thread_id",
                "x_claude_thread_id",
            }:
                return _strip_legacy_affinity_prefixes(value)

    if isinstance(body, Mapping):
        for key in (
            "thread_id",
            "aawm_thread_id",
            "codex_thread_id",
            "claude_thread_id",
        ):
            value = _clean_optional_str(body.get(key))
            if value is not None:
                return _strip_legacy_affinity_prefixes(value)

    if isinstance(metadata, dict):
        for key in (
            "session_id",
            "aawm_session_id",
            "codex_session_id",
            "claude_session_id",
            "anthropic_session_id",
        ):
            value = _clean_optional_str(metadata.get(key))
            if value is not None:
                return _strip_legacy_affinity_prefixes(value)

    if headers is not None:
        for key in (
            "session_id",
            "x-session-id",
            "x-aawm-session-id",
            "x-codex-session-id",
            "x-claude-session-id",
            "anthropic-beta-session-id",
        ):
            value = header_map.get(key)
            if value is not None:
                return _strip_legacy_affinity_prefixes(value)
        # Common dash/underscore variants
        for name, value in header_map.items():
            if value and name.replace("-", "_") in {
                "session_id",
                "aawm_session_id",
                "codex_session_id",
                "claude_session_id",
            }:
                return _strip_legacy_affinity_prefixes(value)

    if isinstance(body, Mapping):
        for key in ("session_id", "aawm_session_id"):
            value = _clean_optional_str(body.get(key))
            if value is not None:
                return _strip_legacy_affinity_prefixes(value)
    return None


def build_aawm_alias_routing_session_owner_cache_key(
    *,
    session_identity: str,
) -> str:
    """Build the durable cache key from canonical session identity only."""

    canonical = _clean_session_identity(session_identity)
    namespace = durable.get_aawm_alias_routing_state_namespace()
    opaque = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return (
        f"{durable.AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX}:{namespace}:"
        f"{_SESSION_OWNER_STATE_KIND}:{opaque}"
    )


def _hosted_provider_from_attributes(attrs: Mapping[str, Any]) -> str:
    """Hard owner identity: openai, xai, cursor, moonshot, or normalized provider."""

    provider = (_clean_optional_str(attrs.get("provider")) or "").strip().lower()
    if not provider:
        provider = (
            _clean_optional_str(attrs.get("hosted_provider")) or ""
        ).strip().lower()
    route_family = (_clean_optional_str(attrs.get("route_family")) or "").strip().lower()
    if provider == "xai":
        return "xai"
    if provider in {"cursor_agent", "cursor"}:
        return "cursor"
    if provider in {"kimi_code", "moonshot"}:
        return "moonshot"
    if (
        not provider
        or provider in {"openai", "codex", "codex_oauth", "chatgpt"}
        or "codex_oauth" in route_family
        or "codex_responses" in route_family
    ):
        return "openai"
    return provider


def _hosted_providers_match(
    left: Mapping[str, Any],
    right: Optional[Mapping[str, Any]] = None,
) -> bool:
    right = right or {}
    left_host = _hosted_provider_from_attributes(left)
    right_host = _hosted_provider_from_attributes(right)
    return bool(left_host) and left_host == right_host


def _normalized_owner_id_endpoint_state(attrs: Mapping[str, Any]) -> tuple[str, str]:
    """Collapse equivalent managed direct-OpenAI shapes onto one owner id."""

    if _managed_direct_openai_owner_shape(attrs) is not None:
        return (
            _MANAGED_DIRECT_OPENAI_OWNER_ID_ENDPOINT,
            _MANAGED_DIRECT_OPENAI_OWNER_ID_STATE,
        )
    return (
        str(attrs.get("endpoint_contract") or "default"),
        str(attrs.get("state_format") or "default"),
    )


def build_session_owner_attributes(
    *,
    provider: Any = None,
    model: Any = None,
    route_family: Any = None,
    account_label: Any = None,
    account_hash: Any = None,
    account_lane: Any = None,
    endpoint_contract: Any = None,
    state_format: Any = None,
    ingress: Any = None,
    requested_model: Any = None,
    alias_family: Any = None,
    credential_affinity: Any = None,
    candidate: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Payload:
    """Build owner attributes from a selection/candidate/direct context."""

    source: dict[str, Any] = {}
    if isinstance(candidate, Mapping):
        source.update(dict(candidate))
    if account_label is None:
        account_label = source.get("account_label") or source.get(
            "codex_oauth_account_label"
        )
    if account_hash is None:
        account_hash = source.get("account_hash") or source.get(
            "codex_oauth_account_hash"
        )
    if account_lane is None:
        account_lane = (
            source.get("account_lane")
            or source.get("codex_oauth_lane_key")
            or source.get("lane_key")
        )
    if provider is None:
        provider = source.get("provider")
    if model is None:
        model = source.get("model")
    if route_family is None:
        route_family = source.get("route_family")
    if endpoint_contract is None:
        endpoint_contract = source.get("endpoint_contract") or source.get(
            "route_family"
        )
    if state_format is None:
        state_format = source.get("state_format") or source.get("route_family")
    if ingress is None:
        ingress = source.get("ingress")
    if requested_model is None:
        requested_model = source.get("requested_model") or source.get("alias_model")
    if alias_family is None:
        alias_family = source.get("alias_family")
    if credential_affinity is None:
        credential_affinity = source.get("codex_oauth_credential_affinity")

    attributes: Payload = {}
    values = {
        "provider": provider,
        "model": model,
        "route_family": route_family,
        "account_label": account_label,
        "account_hash": account_hash,
        "account_lane": account_lane,
        "endpoint_contract": endpoint_contract,
        "state_format": state_format,
        "credential_affinity": credential_affinity,
        "ingress": ingress,
        "requested_model": requested_model,
        "alias_family": alias_family,
    }
    for key in _OWNER_ATTRIBUTE_FIELDS:
        cleaned = _clean_optional_str(values.get(key))
        if cleaned is not None:
            attributes[key] = cleaned
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if key in attributes:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned_extra = (
                    _clean_optional_str(value) if isinstance(value, str) else value
                )
                if cleaned_extra is not None:
                    attributes[str(key)] = cleaned_extra
    hosted = _hosted_provider_from_attributes(attributes)
    if hosted:
        attributes["hosted_provider"] = hosted
    return attributes


def _core_owner_attributes(attributes: Mapping[str, Any]) -> Payload:
    core: Payload = {
        key: attributes[key]
        for key in _CORE_OWNER_ATTRIBUTE_KEYS
        if key in attributes and attributes[key] is not None
    }
    hosted = _hosted_provider_from_attributes(attributes)
    if hosted:
        core["hosted_provider"] = hosted
    return core


def _accounts_are_interchangeable(
    left: Mapping[str, Any],
    right: Optional[Mapping[str, Any]] = None,
) -> bool:
    right = right or {}
    providers = {
        str(left.get("provider") or "").strip().lower(),
        str(right.get("provider") or "").strip().lower(),
    }
    providers.discard("")
    if providers != {"openai"}:
        return False
    return any(
        str(attrs.get("credential_affinity") or "").strip().lower()
        == "interchangeable"
        for attrs in (left, right)
    )


def build_session_owner_id(
    attributes: Optional[Mapping[str, Any]] = None,
    *,
    provider: Any = None,
    model: Any = None,
    route_family: Any = None,
    account_lane: Any = None,
    candidate: Optional[Mapping[str, Any]] = None,
) -> str:
    attrs = build_session_owner_attributes(
        provider=provider,
        model=model,
        route_family=route_family,
        account_lane=account_lane,
        candidate=candidate,
        extra=attributes,
    )
    if attributes:
        merged = dict(attrs)
        for key, value in attributes.items():
            if value is not None and str(value).strip() != "":
                merged[key] = value
        attrs = cast(Payload, merged)
        hosted = _hosted_provider_from_attributes(attrs)
        if hosted:
            attrs["hosted_provider"] = hosted
    hosted = _hosted_provider_from_attributes(attrs) or "unknown"
    endpoint, state = _normalized_owner_id_endpoint_state(attrs)
    return "|".join((hosted, endpoint, state))


def route_requires_account_identity(attributes: Mapping[str, Any]) -> bool:
    """True when ownership promotion must carry credential/account identity."""

    if _accounts_are_interchangeable(attributes):
        return False
    if attributes.get("account_lane") or attributes.get("account_hash"):
        return True
    if attributes.get("account_label"):
        return True
    route_family = str(attributes.get("route_family") or "").lower()
    endpoint = str(attributes.get("endpoint_contract") or "").lower()
    provider = str(attributes.get("provider") or "").lower()
    blob = f"{route_family} {endpoint} {provider}"
    return any(marker in blob for marker in _ACCOUNT_SCOPED_ROUTE_MARKERS)


# Exact OpenAI/Codex model-list paths. Do not treat /models/{id} as discovery.
_OPENAI_MODELS_DISCOVERY_PATHS = frozenset(
    {
        "/models",
        "/v1/models",
        "/openai/models",
        "/openai/v1/models",
        "/openai_passthrough/models",
        "/openai_passthrough/v1/models",
        "/backend-api/codex/models",
    }
)


def _normalize_openai_models_discovery_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:  # noqa: BLE001
            return None
    path = getattr(value, "path", None)
    raw = path if isinstance(path, str) and path else (value if isinstance(value, str) else None)
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.split("?", 1)[0].split("#", 1)[0]
    if "://" in raw:
        _, rest = raw.split("://", 1)
        slash = rest.find("/")
        raw = rest[slash:] if slash >= 0 else "/"
    if not raw.startswith("/"):
        raw = "/" + raw
    if len(raw) > 1:
        raw = raw.rstrip("/")
    return raw or None


def _iter_openai_models_discovery_paths(
    request: Any,
    *,
    endpoint: Any = None,
    url: Any = None,
) -> list[str]:
    candidates = [endpoint, url]
    if request is not None:
        candidates.append(getattr(request, "url", None))
        scope = getattr(request, "scope", None)
        if isinstance(scope, Mapping):
            candidates.append(scope.get("path"))
            candidates.append(scope.get("raw_path"))
    paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_openai_models_discovery_path(candidate)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def _request_has_inbound_openai_auth(request: Any) -> bool:
    headers = getattr(request, "headers", None) if request is not None else None
    if headers is None:
        return False
    try:
        items = headers.items()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False
    for name, value in items:
        if str(name).lower() not in {"authorization", "api-key"}:
            continue
        if isinstance(value, str) and value.strip():
            return True
    return False


def _request_uses_codex_native_auth_markers(request: Any) -> bool:
    if request is None:
        return False
    try:
        from .codex_oauth import _request_uses_codex_native_auth
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(_request_uses_codex_native_auth(request))
    except Exception:  # noqa: BLE001
        return False


def should_skip_session_owner_for_openai_models_discovery(
    request: Any = None,
    *,
    endpoint: Any = None,
    url: Any = None,
) -> bool:
    """True for authenticated Codex-native GET OpenAI/Codex model listing.

    OPENAI-013: ``GET /models`` and ``GET /v1/models`` create no generation or
    continuation state, so both enforcement layers skip reservation/validation.
    Responses, chat, and other state-bearing calls remain guarded.
    """

    method = getattr(request, "method", None) if request is not None else None
    if not isinstance(method, str) or method.upper() != "GET":
        return False
    if not _request_has_inbound_openai_auth(request):
        return False
    if not _request_uses_codex_native_auth_markers(request):
        return False
    return any(
        path in _OPENAI_MODELS_DISCOVERY_PATHS
        for path in _iter_openai_models_discovery_paths(
            request,
            endpoint=endpoint,
            url=url,
        )
    )


def incomplete_owner_attribute_reason(
    attributes: Optional[Mapping[str, Any]],
    *,
    for_promotion: bool = False,
) -> Optional[str]:
    if not isinstance(attributes, Mapping) or not attributes:
        return "session_owner: owner attributes missing"
    missing = [
        key
        for key in _REQUIRED_OWNER_ATTRIBUTE_KEYS
        if not _clean_optional_str(attributes.get(key))
    ]
    if missing:
        return f"session_owner: incomplete owner attributes missing={missing}"
    if for_promotion and route_requires_account_identity(attributes):
        has_account = any(
            _clean_optional_str(attributes.get(key))
            for key in ("account_hash", "account_label", "account_lane")
        )
        if not has_account:
            return (
                "session_owner: account-scoped route requires credential/"
                "account identity on promotion"
            )
    return None


def _normalize_reservation_ttl(ttl_seconds: float) -> float:
    try:
        ttl = float(ttl_seconds)
    except (TypeError, ValueError):
        ttl = _DEFAULT_RESERVATION_TTL_SECONDS
    if not math.isfinite(ttl):
        ttl = _DEFAULT_RESERVATION_TTL_SECONDS
    return max(_MIN_RESERVATION_TTL_SECONDS, min(_MAX_RESERVATION_TTL_SECONDS, ttl))


def _owner_attributes(record: Optional[Mapping[str, Any]]) -> Payload:
    if not isinstance(record, Mapping):
        return {}
    attrs = record.get(_RECORD_ATTRIBUTES_FIELD)
    if isinstance(attrs, Mapping):
        return {
            str(k): v
            for k, v in attrs.items()
            if v is not None and str(v).strip() != ""
        }
    return {}


def _record_state(record: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(record, Mapping):
        return None
    state = _clean_optional_str(record.get(_RECORD_STATE_FIELD))
    if state in {
        SessionOwnerRecordState.RESERVED.value,
        SessionOwnerRecordState.OWNED.value,
    }:
        return state
    # Legacy draft records without state were treated as owned claims.
    if record.get(_RECORD_OWNER_FIELD):
        return SessionOwnerRecordState.OWNED.value
    return None


_MANAGED_DIRECT_OPENAI_OWNER_SHAPES = frozenset(
    {
        ("codex_responses", "codex_responses", "codex_responses"),
        ("codex_oauth", "openai_responses", "openai_responses"),
    }
)


def _managed_direct_openai_owner_shape(
    attributes: Mapping[str, Any],
) -> Optional[tuple[str, str, str]]:
    """Return the complete managed direct-OpenAI tuple when it is exact."""

    if _clean_optional_str(attributes.get("provider")) != "openai":
        return None
    shape = (
        _clean_optional_str(attributes.get("route_family")),
        _clean_optional_str(attributes.get("endpoint_contract")),
        _clean_optional_str(attributes.get("state_format")),
    )
    if None in shape:
        return None
    typed = (str(shape[0]), str(shape[1]), str(shape[2]))
    if typed not in _MANAGED_DIRECT_OPENAI_OWNER_SHAPES:
        return None
    return typed


def _managed_direct_openai_owner_shapes_are_equivalent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Treat the two complete managed direct-OpenAI owner tuples as one contract.

    Comparison-only for stored route_family/endpoint/state. Public
    openai_responses/openai_responses/openai_responses remains distinct.
    Owner ids collapse the two managed direct-OpenAI shapes onto one triple.
    """

    left_shape = _managed_direct_openai_owner_shape(left)
    right_shape = _managed_direct_openai_owner_shape(right)
    return left_shape is not None and right_shape is not None


def _attributes_exactly_equal(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_core = _core_owner_attributes(left)
    right_core = _core_owner_attributes(right)
    if _hosted_providers_match(left_core, right_core):
        for key in _MUTABLE_SAME_HOSTED_PROVIDER_ATTRIBUTE_KEYS:
            left_core.pop(key, None)
            right_core.pop(key, None)
    elif _accounts_are_interchangeable(left_core, right_core):
        for key in (
            "account_hash",
            "account_label",
            "account_lane",
            "credential_affinity",
        ):
            left_core.pop(key, None)
            right_core.pop(key, None)
    if _managed_direct_openai_owner_shapes_are_equivalent(left_core, right_core):
        for key in ("route_family", "endpoint_contract", "state_format"):
            left_core.pop(key, None)
            right_core.pop(key, None)
    if set(left_core.keys()) != set(right_core.keys()):
        return False
    for key, value in left_core.items():
        if str(value) != str(right_core.get(key)):
            return False
    return True


def _compatibility_mismatch_reason(
    *,
    owner_record: Mapping[str, Any],
    requested_attributes: Optional[Mapping[str, Any]],
    require_exact_attributes: bool,
) -> Optional[str]:
    state = _record_state(owner_record)
    if state == SessionOwnerRecordState.RESERVED.value:
        return "session_owner: session has an active competing reservation"
    if state != SessionOwnerRecordState.OWNED.value:
        return "session_owner: malformed ownership state"
    owner_attrs = _owner_attributes(owner_record)
    if not owner_attrs:
        return "session_owner: owned record missing attributes"
    incomplete = incomplete_owner_attribute_reason(owner_attrs, for_promotion=False)
    if incomplete is not None:
        return incomplete
    if not requested_attributes:
        return None
    requested_core = _core_owner_attributes(requested_attributes)
    hosted_match = _hosted_providers_match(owner_attrs, requested_core)
    if require_exact_attributes:
        if not _attributes_exactly_equal(left=owner_attrs, right=requested_core):
            return "session_owner: requested route does not exactly match owner"
        return None
    equivalent_managed_direct_openai = (
        _managed_direct_openai_owner_shapes_are_equivalent(
            owner_attrs,
            requested_core,
        )
    )
    owner_hosted = _hosted_provider_from_attributes(owner_attrs)
    requested_hosted = _hosted_provider_from_attributes(requested_core)
    if owner_hosted and requested_hosted and owner_hosted != requested_hosted:
        return (
            "session_owner: hosted_provider mismatch "
            f"owner={owner_hosted} requested={requested_hosted}"
        )
    for key in ("route_family",):
        if equivalent_managed_direct_openai:
            continue
        req = _clean_optional_str(requested_core.get(key))
        own = _clean_optional_str(owner_attrs.get(key))
        if req is not None and own is not None and req != own:
            return f"session_owner: {key} mismatch owner={own} requested={req}"
    if not hosted_match:
        for key in ("account_hash", "account_lane", "account_label"):
            req = _clean_optional_str(requested_core.get(key))
            own = _clean_optional_str(owner_attrs.get(key))
            if req is not None and own is not None and req != own:
                return f"session_owner: {key} mismatch"
    for key in ("endpoint_contract", "state_format"):
        if equivalent_managed_direct_openai:
            continue
        req = _clean_optional_str(requested_core.get(key))
        own = _clean_optional_str(owner_attrs.get(key))
        if req is not None and own is not None and req != own:
            return f"session_owner: {key} mismatch owner={own} requested={req}"
    return None


def build_session_owner_provenance(
    *,
    session_identity: Optional[str],
    decision: str,
    owner_record: Optional[Mapping[str, Any]] = None,
    owner_id: Optional[str] = None,
    mismatch_reason: Optional[str] = None,
    cache_key: Optional[str] = None,
    reservation_token: Optional[str] = None,
    claim_outcome: Optional[str] = None,
) -> Payload:
    attrs = _owner_attributes(owner_record)
    provenance: Payload = {
        "canonical_session_identity": session_identity,
        "session_owner_decision": decision,
        "session_owner_id": owner_id
        or (
            owner_record.get(_RECORD_OWNER_FIELD)
            if isinstance(owner_record, Mapping)
            else None
        ),
        "session_owner_state": _record_state(owner_record),
        "session_owner_mismatch_reason": mismatch_reason,
        "session_owner_cache_key_fingerprint": (
            hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
            if cache_key
            else None
        ),
        "session_owner_provider": attrs.get("provider"),
        "session_owner_model": attrs.get("model"),
        "session_owner_route_family": attrs.get("route_family"),
        "session_owner_endpoint_contract": attrs.get("endpoint_contract"),
        "session_owner_state_format": attrs.get("state_format"),
        "session_owner_account_lane": attrs.get("account_lane"),
        "session_owner_mutation_outcome": claim_outcome,
        # Never include reservation tokens or secrets in provenance.
        "session_owner_has_reservation_token": bool(reservation_token),
    }
    return {k: v for k, v in provenance.items() if v is not None}


def attach_session_owner_metadata(
    target: Optional[dict[str, Any]],
    *,
    provenance: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    destination = target if isinstance(target, dict) else {}
    if not provenance:
        return destination
    for key, value in provenance.items():
        if value is not None:
            destination[key] = value
    return destination


def owner_record_as_affinity_hint(
    owner_record: Optional[Mapping[str, Any]],
    *,
    preserve_account_identity: bool = False,
) -> Optional[dict[str, Any]]:
    if not isinstance(owner_record, Mapping):
        return None
    if _record_state(owner_record) != SessionOwnerRecordState.OWNED.value:
        return None
    attrs = _owner_attributes(owner_record)
    if not attrs:
        return None
    affinity: dict[str, Any] = {
        "provider": attrs.get("provider"),
        "model": attrs.get("model"),
        "route_family": attrs.get("route_family"),
        "last_resort": False,
        "affinity_state_source": "session_owner",
    }
    interchangeable = _accounts_are_interchangeable(attrs)
    if interchangeable:
        affinity["codex_oauth_credential_affinity"] = "interchangeable"
    include_account_identity = preserve_account_identity or not interchangeable
    if attrs.get("account_label") and include_account_identity:
        affinity["codex_oauth_account_label"] = attrs.get("account_label")
    if attrs.get("account_hash") and include_account_identity:
        affinity["codex_oauth_account_hash"] = attrs.get("account_hash")
    if attrs.get("account_lane") and include_account_identity:
        affinity["codex_oauth_lane_key"] = attrs.get("account_lane")
    return {k: v for k, v in affinity.items() if v is not None}


def _build_reserved_record(
    *,
    owner_id: str,
    attributes: Mapping[str, Any],
    reservation_token: str,
    now: Optional[float] = None,
) -> Payload:
    ts = time.time() if now is None else float(now)
    return {
        _RECORD_STATE_FIELD: SessionOwnerRecordState.RESERVED.value,
        _RECORD_OWNER_FIELD: owner_id,
        _RECORD_ATTRIBUTES_FIELD: dict(_core_owner_attributes(attributes)),
        _RECORD_TOKEN_FIELD: reservation_token,
        _RECORD_RESERVED_AT_FIELD: ts,
        _RECORD_LAST_RENEWED_AT_FIELD: ts,
    }


def _build_owned_record(
    *,
    owner_id: str,
    attributes: Mapping[str, Any],
    reservation_token: Optional[str],
    reserved_at_epoch: Optional[float] = None,
    now: Optional[float] = None,
) -> Payload:
    ts = time.time() if now is None else float(now)
    record: Payload = {
        _RECORD_STATE_FIELD: SessionOwnerRecordState.OWNED.value,
        _RECORD_OWNER_FIELD: owner_id,
        _RECORD_ATTRIBUTES_FIELD: dict(_core_owner_attributes(attributes)),
        _RECORD_OWNED_AT_FIELD: ts,
        _RECORD_LAST_RENEWED_AT_FIELD: ts,
        durable.PERSISTENT_MARKER: True,
    }
    if reserved_at_epoch is not None:
        record[_RECORD_RESERVED_AT_FIELD] = float(reserved_at_epoch)
    if reservation_token:
        # Retained only for audit of which reservation promoted; not required
        # for subsequent owned reads.
        record["promoted_from_reservation_token_fingerprint"] = hashlib.sha256(
            reservation_token.encode("utf-8")
        ).hexdigest()[:16]
    return record


async def _get_redis_cache() -> tuple[Optional[Any], Optional[str]]:
    dual_cache = durable.get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None, "session_owner: durable cache unavailable"
    redis_cache = getattr(dual_cache, "redis_cache", None)
    if redis_cache is None:
        return None, "session_owner: durable cache missing redis_cache"
    return redis_cache, None


def _namespaced_key(redis_cache: Any, cache_key: str) -> str:
    fix_ns = getattr(redis_cache, "check_and_fix_namespace", None)
    if callable(fix_ns):
        return cast(str, fix_ns(key=cache_key))
    return cache_key


async def _raw_redis_client(redis_cache: Any) -> Any:
    init_fn = getattr(redis_cache, "init_async_client", None)
    if not callable(init_fn):
        raise RuntimeError("session_owner: redis client unavailable")
    client = init_fn()
    if client is None:
        raise RuntimeError("session_owner: redis client unavailable")
    return client


def _decode_redis_value(raw: Any, *, redis_cache: Any) -> Optional[Payload]:
    if raw is None:
        return None
    logic = getattr(redis_cache, "_get_cache_logic", None)
    if callable(logic):
        parsed = logic(cached_response=raw)
    else:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise RuntimeError("session_owner: malformed ownership payload")
    return cast(Payload, parsed)


async def _read_session_owner_record(
    *,
    redis_cache: Any,
    cache_key: str,
) -> Optional[Payload]:
    client = await _raw_redis_client(redis_cache)
    namespaced = _namespaced_key(redis_cache, cache_key)
    try:
        raw = await client.get(namespaced)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"session_owner: redis get failed: {exc}") from exc
    if raw is None:
        return None
    try:
        return _decode_redis_value(raw, redis_cache=redis_cache)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"session_owner: redis decode failed: {exc}") from exc


_REQUEST_STATE_RESERVATION_WAIT_DEADLINE_ATTR = (
    "_aawm_session_owner_reservation_wait_deadline"
)


def _get_reservation_wait_deadline(request: Any) -> Optional[float]:
    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        value = object.__getattribute__(
            state, _REQUEST_STATE_RESERVATION_WAIT_DEADLINE_ATTR
        )
    except AttributeError:
        value = getattr(state, _REQUEST_STATE_RESERVATION_WAIT_DEADLINE_ATTR, None)
    try:
        deadline = float(value)
    except (TypeError, ValueError):
        return None
    return deadline if math.isfinite(deadline) else None


def _set_reservation_wait_deadline(
    request: Any,
    deadline: float,
) -> None:
    if request is None:
        return
    state = getattr(request, "state", None)
    if state is None:
        return
    setattr(state, _REQUEST_STATE_RESERVATION_WAIT_DEADLINE_ATTR, deadline)


def _normalize_reservation_wait_timeout(timeout_seconds: Optional[float]) -> float:
    if timeout_seconds is None:
        timeout = _DEFAULT_RESERVATION_WAIT_TIMEOUT_SECONDS
    else:
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError):
            timeout = _DEFAULT_RESERVATION_WAIT_TIMEOUT_SECONDS
    if not math.isfinite(timeout):
        timeout = _DEFAULT_RESERVATION_WAIT_TIMEOUT_SECONDS
    return max(0.0, min(_MAX_RESERVATION_WAIT_TIMEOUT_SECONDS, timeout))


def _normalize_reservation_wait_poll(poll_seconds: Optional[float]) -> float:
    if poll_seconds is None:
        poll = _DEFAULT_RESERVATION_WAIT_POLL_SECONDS
    else:
        try:
            poll = float(poll_seconds)
        except (TypeError, ValueError):
            poll = _DEFAULT_RESERVATION_WAIT_POLL_SECONDS
    if not math.isfinite(poll):
        poll = _DEFAULT_RESERVATION_WAIT_POLL_SECONDS
    return max(0.001, min(_MAX_RESERVATION_WAIT_POLL_SECONDS, poll))


async def _wait_for_foreign_reserved_session_owner(
    *,
    redis_cache: Any,
    cache_key: str,
    record: Payload,
    request: Any = None,
    reservation_token: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    poll_seconds: Optional[float] = None,
) -> tuple[Optional[Payload], Optional[str]]:
    """Re-read a foreign reservation within one request-scoped wait budget."""

    if _record_state(record) != SessionOwnerRecordState.RESERVED.value:
        return record, None
    record_token = _clean_optional_str(record.get(_RECORD_TOKEN_FIELD))
    if reservation_token is not None and record_token == reservation_token:
        return record, None

    deadline = _get_reservation_wait_deadline(request)
    if deadline is None:
        deadline = time.monotonic() + _normalize_reservation_wait_timeout(
            timeout_seconds
        )
        _set_reservation_wait_deadline(request, deadline)
    poll = _normalize_reservation_wait_poll(poll_seconds)
    current = record

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return current, None
        await asyncio.sleep(min(poll, remaining))
        try:
            current = await _read_session_owner_record(
                redis_cache=redis_cache,
                cache_key=cache_key,
            )
        except RuntimeError as exc:
            return current, str(exc)
        if current is None:
            return None, None
        current_token = _clean_optional_str(current.get(_RECORD_TOKEN_FIELD))
        if (
            _record_state(current) != SessionOwnerRecordState.RESERVED.value
            or (
                reservation_token is not None
                and current_token == reservation_token
            )
        ):
            return current, None


async def get_session_owner_record(
    *,
    session_identity: Optional[str],
    request: Any = None,
    wait_for_foreign_reservation: bool = False,
    reservation_token: Optional[str] = None,
    reservation_wait_timeout_seconds: Optional[float] = None,
    reservation_wait_poll_seconds: Optional[float] = None,
) -> tuple[Optional[Payload], Optional[str], Optional[str]]:
    """Return (record, cache_key, error). error set => fail closed."""

    cleaned = _clean_optional_str(session_identity)
    if cleaned is None:
        return None, None, None
    cleaned = _strip_legacy_affinity_prefixes(cleaned)
    cache_key = build_aawm_alias_routing_session_owner_cache_key(
        session_identity=cleaned
    )
    redis_cache, error = await _get_redis_cache()
    if error is not None or redis_cache is None:
        return None, cache_key, error or "session_owner: durable cache unavailable"
    try:
        record = await _read_session_owner_record(
            redis_cache=redis_cache,
            cache_key=cache_key,
        )
        if (
            wait_for_foreign_reservation
            and record is not None
            and _record_state(record) == SessionOwnerRecordState.RESERVED.value
        ):
            record, wait_error = await _wait_for_foreign_reserved_session_owner(
                redis_cache=redis_cache,
                cache_key=cache_key,
                record=record,
                request=request,
                reservation_token=reservation_token,
                timeout_seconds=reservation_wait_timeout_seconds,
                poll_seconds=reservation_wait_poll_seconds,
            )
            if wait_error is not None:
                return record, cache_key, wait_error
    except RuntimeError as exc:
        return None, cache_key, str(exc)
    return record, cache_key, None


async def guard_session_owner_before_egress(  # noqa: PLR0915
    *,
    session_identity: Optional[str] = None,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    requested_attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    owner_id: Optional[str] = None,
    reservation_token: Optional[str] = None,
    reservation_ttl_seconds: float = _DEFAULT_RESERVATION_TTL_SECONDS,
    require_exact_attributes: bool = False,
    reserve_if_unowned: bool = True,
    reservation_wait_timeout_seconds: Optional[float] = None,
    reservation_wait_poll_seconds: Optional[float] = None,
) -> SessionOwnerGuardResult:
    """Single pre-egress lifecycle guard for every route family.

    Concurrent unowned competitors cannot both reserve: NX claim is atomic and
    the loser receives ``redispatch_required`` before upstream send.
    """

    cleaned = resolve_canonical_session_identity(
        request,
        request_body,
        session_identity=session_identity,
    )
    if cleaned is None:
        provenance = build_session_owner_provenance(
            session_identity=None,
            decision=SessionOwnerGuardDecision.NO_SESSION.value,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.NO_SESSION,
            session_identity=None,
            provenance=provenance,
        )

    attrs = build_session_owner_attributes(
        candidate=candidate,
        extra=requested_attributes,
    )
    if requested_attributes:
        merged = dict(attrs)
        merged.update(dict(requested_attributes))
        attrs = build_session_owner_attributes(extra=merged, candidate=candidate)
    attrs = _core_owner_attributes(attrs)
    cache_key = build_aawm_alias_routing_session_owner_cache_key(
        session_identity=cleaned
    )
    # When the caller supplied known owner identity, pin exactly on hard
    # owner identity (hosted provider + endpoint/state). Model and OpenAI
    # account remain mutable last-used attributes.
    if attrs and not require_exact_attributes:
        hosted = _hosted_provider_from_attributes(attrs)
        if hosted and all(
            _clean_optional_str(attrs.get(key))
            for key in ("endpoint_contract", "state_format")
        ):
            require_exact_attributes = True
    if attrs:
        core_incomplete = incomplete_owner_attribute_reason(
            attrs, for_promotion=False
        )
        if core_incomplete is not None:
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                mismatch_reason=core_incomplete,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                mismatch_reason=core_incomplete,
                provenance=provenance,
            )
        if route_requires_account_identity(attrs):
            account_incomplete = incomplete_owner_attribute_reason(
                attrs, for_promotion=True
            )
            if account_incomplete is not None:
                provenance = build_session_owner_provenance(
                    session_identity=cleaned,
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                    mismatch_reason=account_incomplete,
                    cache_key=cache_key,
                )
                return SessionOwnerGuardResult(
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=cleaned,
                    cache_key=cache_key,
                    mismatch_reason=account_incomplete,
                    provenance=provenance,
                )
    resolved_owner_id = owner_id or (
        build_session_owner_id(attributes=attrs) if attrs else "pending"
    )
    token = _clean_optional_str(reservation_token) or str(uuid.uuid4())
    ttl = _normalize_reservation_ttl(reservation_ttl_seconds)

    redis_cache, error = await _get_redis_cache()
    if error is not None or redis_cache is None:
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
            mismatch_reason=error,
            cache_key=cache_key,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=cleaned,
            cache_key=cache_key,
            mismatch_reason=error,
            provenance=provenance,
        )

    try:
        existing = await _read_session_owner_record(
            redis_cache=redis_cache,
            cache_key=cache_key,
        )
    except RuntimeError as exc:
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
            mismatch_reason=str(exc),
            cache_key=cache_key,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=cleaned,
            cache_key=cache_key,
            mismatch_reason=str(exc),
            provenance=provenance,
        )

    if existing is not None:
        state = _record_state(existing)
        existing_token = _clean_optional_str(existing.get(_RECORD_TOKEN_FIELD))
        existing_owner = _clean_optional_str(existing.get(_RECORD_OWNER_FIELD))

        if state == SessionOwnerRecordState.RESERVED.value:
            if existing_token and existing_token == token:
                renewed = await _renew_reservation(
                    redis_cache=redis_cache,
                    cache_key=cache_key,
                    record=existing,
                    ttl_seconds=ttl,
                )
                if renewed is None:
                    reason = "session_owner: reservation renewal failed"
                    provenance = build_session_owner_provenance(
                        session_identity=cleaned,
                        decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                        mismatch_reason=reason,
                        cache_key=cache_key,
                    )
                    return SessionOwnerGuardResult(
                        decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                        session_identity=cleaned,
                        cache_key=cache_key,
                        mismatch_reason=reason,
                        provenance=provenance,
                    )
                provenance = build_session_owner_provenance(
                    session_identity=cleaned,
                    decision=SessionOwnerGuardDecision.RESERVATION_RENEWED.value,
                    owner_record=renewed,
                    owner_id=existing_owner,
                    cache_key=cache_key,
                    reservation_token=token,
                )
                return SessionOwnerGuardResult(
                    decision=SessionOwnerGuardDecision.RESERVATION_RENEWED,
                    session_identity=cleaned,
                    cache_key=cache_key,
                    reservation_token=token,
                    owner_id=existing_owner,
                    owner_record=renewed,
                    provenance=provenance,
                    held_reservation=True,
                )
            existing, wait_error = await _wait_for_foreign_reserved_session_owner(
                redis_cache=redis_cache,
                cache_key=cache_key,
                record=existing,
                request=request,
                reservation_token=token,
                timeout_seconds=reservation_wait_timeout_seconds,
                poll_seconds=reservation_wait_poll_seconds,
            )
            if wait_error is not None:
                provenance = build_session_owner_provenance(
                    session_identity=cleaned,
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                    owner_record=existing,
                    owner_id=(
                        _clean_optional_str(existing.get(_RECORD_OWNER_FIELD))
                        if isinstance(existing, Mapping)
                        else None
                    ),
                    mismatch_reason=wait_error,
                    cache_key=cache_key,
                )
                return SessionOwnerGuardResult(
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=cleaned,
                    cache_key=cache_key,
                    owner_id=(
                        _clean_optional_str(existing.get(_RECORD_OWNER_FIELD))
                        if isinstance(existing, Mapping)
                        else None
                    ),
                    owner_record=existing,
                    mismatch_reason=wait_error,
                    provenance=provenance,
                )
            if existing is not None:
                state = _record_state(existing)
                existing_owner = _clean_optional_str(
                    existing.get(_RECORD_OWNER_FIELD)
                )

        if existing is not None:
            if state == SessionOwnerRecordState.OWNED.value:
                mismatch = _compatibility_mismatch_reason(
                    owner_record=existing,
                    requested_attributes=attrs or None,
                    require_exact_attributes=require_exact_attributes,
                )
                if mismatch is not None:
                    provenance = build_session_owner_provenance(
                        session_identity=cleaned,
                        decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                        owner_record=existing,
                        owner_id=existing_owner,
                        mismatch_reason=mismatch,
                        cache_key=cache_key,
                    )
                    return SessionOwnerGuardResult(
                        decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                        session_identity=cleaned,
                        cache_key=cache_key,
                        owner_id=existing_owner,
                        owner_record=existing,
                        mismatch_reason=mismatch,
                        provenance=provenance,
                    )
                provenance = build_session_owner_provenance(
                    session_identity=cleaned,
                    decision=SessionOwnerGuardDecision.COMPATIBLE_OWNER.value,
                    owner_record=existing,
                    owner_id=existing_owner,
                    cache_key=cache_key,
                )
                return SessionOwnerGuardResult(
                    decision=SessionOwnerGuardDecision.COMPATIBLE_OWNER,
                    session_identity=cleaned,
                    cache_key=cache_key,
                    owner_id=existing_owner,
                    owner_record=existing,
                    provenance=provenance,
                )
            if state == SessionOwnerRecordState.RESERVED.value:
                reason = "session_owner: concurrent reservation held by another request"
                provenance = build_session_owner_provenance(
                    session_identity=cleaned,
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                    owner_record=existing,
                    owner_id=existing_owner,
                    mismatch_reason=reason,
                    cache_key=cache_key,
                )
                return SessionOwnerGuardResult(
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                    session_identity=cleaned,
                    cache_key=cache_key,
                    owner_id=existing_owner,
                    owner_record=existing,
                    mismatch_reason=reason,
                    provenance=provenance,
                )

            reason = "session_owner: malformed ownership record"
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                owner_record=existing,
                mismatch_reason=reason,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                owner_record=existing,
                mismatch_reason=reason,
                provenance=provenance,
            )

    if not reserve_if_unowned:
        # Consult-only egress is forbidden. Unowned sessions must reserve.
        reason = "session_owner: unowned session requires reservation before egress"
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
            mismatch_reason=reason,
            cache_key=cache_key,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=cleaned,
            cache_key=cache_key,
            mismatch_reason=reason,
            provenance=provenance,
        )

    reserved_record = _build_reserved_record(
        owner_id=resolved_owner_id,
        attributes=attrs,
        reservation_token=token,
    )
    try:
        claimed = await redis_cache.async_set_cache(
            key=cache_key,
            value=reserved_record,
            ttl=ttl,
            nx=True,
            raise_on_error=True,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"session_owner: redis reserve failed: {exc}"
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
            mismatch_reason=reason,
            cache_key=cache_key,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=cleaned,
            cache_key=cache_key,
            mismatch_reason=reason,
            provenance=provenance,
        )

    if claimed:
        try:
            durable_record = await _read_session_owner_record(
                redis_cache=redis_cache,
                cache_key=cache_key,
            )
        except RuntimeError as exc:
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                mismatch_reason=str(exc),
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                reservation_token=token,
                mismatch_reason=str(exc),
                provenance=provenance,
                held_reservation=True,
            )
        if durable_record is None:
            reason = "session_owner: reserve write not visible on read-back"
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                mismatch_reason=reason,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                reservation_token=token,
                mismatch_reason=reason,
                provenance=provenance,
                held_reservation=True,
            )
        durable_token = _clean_optional_str(durable_record.get(_RECORD_TOKEN_FIELD))
        if durable_token != token or _record_state(durable_record) != (
            SessionOwnerRecordState.RESERVED.value
        ):
            reason = "session_owner: reserve lost race on read-back"
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                owner_record=durable_record,
                mismatch_reason=reason,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                owner_record=durable_record,
                mismatch_reason=reason,
                provenance=provenance,
            )
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.UNOWNED_RESERVED.value,
            owner_record=durable_record,
            owner_id=resolved_owner_id,
            cache_key=cache_key,
            reservation_token=token,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.UNOWNED_RESERVED,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            owner_id=resolved_owner_id,
            owner_record=durable_record,
            provenance=provenance,
            held_reservation=True,
        )

    # NX lost — another worker reserved or owned first.
    try:
        winner = await _read_session_owner_record(
            redis_cache=redis_cache,
            cache_key=cache_key,
        )
    except RuntimeError as exc:
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
            mismatch_reason=str(exc),
            cache_key=cache_key,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=cleaned,
            cache_key=cache_key,
            mismatch_reason=str(exc),
            provenance=provenance,
        )
    if winner is None:
        reason = "session_owner: reserve race left no durable record"
        provenance = build_session_owner_provenance(
            session_identity=cleaned,
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
            mismatch_reason=reason,
            cache_key=cache_key,
        )
        return SessionOwnerGuardResult(
            decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
            session_identity=cleaned,
            cache_key=cache_key,
            mismatch_reason=reason,
            provenance=provenance,
        )
    if _record_state(winner) == SessionOwnerRecordState.RESERVED.value:
        winner, wait_error = await _wait_for_foreign_reserved_session_owner(
            redis_cache=redis_cache,
            cache_key=cache_key,
            record=winner,
            request=request,
            reservation_token=token,
            timeout_seconds=reservation_wait_timeout_seconds,
            poll_seconds=reservation_wait_poll_seconds,
        )
        if wait_error is not None:
            reason = wait_error
            winner_owner = (
                _clean_optional_str(winner.get(_RECORD_OWNER_FIELD))
                if isinstance(winner, Mapping)
                else None
            )
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                owner_record=winner,
                owner_id=winner_owner,
                mismatch_reason=reason,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                owner_id=winner_owner,
                owner_record=winner,
                mismatch_reason=reason,
                provenance=provenance,
            )
        elif winner is None:
            # The foreign hold was released or expired after the NX race.
            # Retry the ordinary atomic reservation once with this request's
            # token; no token is shared with the other HTTP request.
            try:
                claimed = await redis_cache.async_set_cache(
                    key=cache_key,
                    value=_build_reserved_record(
                        owner_id=resolved_owner_id,
                        attributes=attrs,
                        reservation_token=token,
                    ),
                    ttl=ttl,
                    nx=True,
                    raise_on_error=True,
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"session_owner: redis reserve failed: {exc}"
            else:
                if claimed:
                    try:
                        retry_record = await _read_session_owner_record(
                            redis_cache=redis_cache,
                            cache_key=cache_key,
                        )
                    except RuntimeError as exc:
                        reason = str(exc)
                    else:
                        if (
                            retry_record is not None
                            and _record_state(retry_record)
                            == SessionOwnerRecordState.RESERVED.value
                            and _clean_optional_str(
                                retry_record.get(_RECORD_TOKEN_FIELD)
                            )
                            == token
                        ):
                            provenance = build_session_owner_provenance(
                                session_identity=cleaned,
                                decision=SessionOwnerGuardDecision.UNOWNED_RESERVED.value,
                                owner_record=retry_record,
                                owner_id=resolved_owner_id,
                                cache_key=cache_key,
                                reservation_token=token,
                            )
                            return SessionOwnerGuardResult(
                                decision=SessionOwnerGuardDecision.UNOWNED_RESERVED,
                                session_identity=cleaned,
                                cache_key=cache_key,
                                reservation_token=token,
                                owner_id=resolved_owner_id,
                                owner_record=retry_record,
                                provenance=provenance,
                                held_reservation=True,
                            )
                        reason = "session_owner: reserve lost race on read-back"
                else:
                    reason = "session_owner: concurrent reservation won the race"
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                mismatch_reason=reason,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=cleaned,
                cache_key=cache_key,
                mismatch_reason=reason,
                provenance=provenance,
            )

    winner_state = _record_state(winner)
    winner_owner = _clean_optional_str(winner.get(_RECORD_OWNER_FIELD))
    if winner_state == SessionOwnerRecordState.OWNED.value:
        mismatch = _compatibility_mismatch_reason(
            owner_record=winner,
            requested_attributes=attrs or None,
            require_exact_attributes=require_exact_attributes,
        )
        if mismatch is None:
            provenance = build_session_owner_provenance(
                session_identity=cleaned,
                decision=SessionOwnerGuardDecision.COMPATIBLE_OWNER.value,
                owner_record=winner,
                owner_id=winner_owner,
                cache_key=cache_key,
            )
            return SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.COMPATIBLE_OWNER,
                session_identity=cleaned,
                cache_key=cache_key,
                owner_id=winner_owner,
                owner_record=winner,
                provenance=provenance,
            )
        reason = mismatch
    elif winner_state == SessionOwnerRecordState.RESERVED.value:
        reason = "session_owner: concurrent reservation won the race"
    else:
        reason = "session_owner: malformed ownership record"
    provenance = build_session_owner_provenance(
        session_identity=cleaned,
        decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
        owner_record=winner,
        owner_id=winner_owner,
        mismatch_reason=reason,
        cache_key=cache_key,
    )
    return SessionOwnerGuardResult(
        decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
        session_identity=cleaned,
        cache_key=cache_key,
        owner_id=winner_owner,
        owner_record=winner,
        mismatch_reason=reason,
        provenance=provenance,
    )


async def _renew_reservation(
    *,
    redis_cache: Any,
    cache_key: str,
    record: Mapping[str, Any],
    ttl_seconds: float,
) -> Optional[Payload]:
    """Safe renewal of a held reservation (extends TTL, preserves token)."""

    token = _clean_optional_str(record.get(_RECORD_TOKEN_FIELD))
    if token is None:
        return None
    renewed = dict(record)
    renewed[_RECORD_LAST_RENEWED_AT_FIELD] = time.time()
    client = await _raw_redis_client(redis_cache)
    namespaced = _namespaced_key(redis_cache, cache_key)
    # CAS-style: only overwrite if still our reserved token.
    lua = """
    local raw = redis.call('GET', KEYS[1])
    if not raw then
      return 0
    end
    local ok, current = pcall(cjson.decode, raw)
    if not ok or type(current) ~= 'table' then
      return -1
    end
    if current['state'] ~= 'reserved' or current['reservation_token'] ~= ARGV[1] then
      return 0
    end
    local payload = cjson.decode(ARGV[2])
    redis.call('SET', KEYS[1], cjson.encode(payload), 'EX', tonumber(ARGV[3]))
    return 1
    """
    try:
        result = await client.eval(
            lua,
            1,
            namespaced,
            token,
            json.dumps(renewed),
            str(int(math.ceil(ttl_seconds))),
        )
    except Exception:  # noqa: BLE001
        return None
    if int(result or 0) != 1:
        return None
    return cast(Payload, renewed)


async def promote_session_owner_reservation(  # noqa: PLR0911
    *,
    session_identity: Optional[str],
    reservation_token: Optional[str],
    attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    owner_id: Optional[str] = None,
    provider: Any = None,
    model: Any = None,
    route_family: Any = None,
    endpoint_contract: Any = None,
    state_format: Any = None,
    ingress: Any = None,
    requested_model: Any = None,
    alias_family: Any = None,
) -> SessionOwnerMutationResult:
    """CAS-promote a held reservation to immutable owned after success/first-byte."""

    cleaned = _clean_optional_str(session_identity)
    token = _clean_optional_str(reservation_token)
    if cleaned is None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.SKIPPED,
            session_identity=None,
        )
    cleaned = _strip_legacy_affinity_prefixes(cleaned)
    cache_key = build_aawm_alias_routing_session_owner_cache_key(
        session_identity=cleaned
    )
    if token is None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            error="session_owner: promotion requires reservation_token",
        )

    owner_attributes = _core_owner_attributes(
        build_session_owner_attributes(
            provider=provider,
            model=model,
            route_family=route_family,
            endpoint_contract=endpoint_contract,
            state_format=state_format,
            ingress=ingress,
            requested_model=requested_model,
            alias_family=alias_family,
            candidate=candidate,
            extra=attributes,
        )
    )
    incomplete = incomplete_owner_attribute_reason(
        owner_attributes, for_promotion=True
    )
    if incomplete is not None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            error=incomplete,
        )
    resolved_owner_id = owner_id or build_session_owner_id(attributes=owner_attributes)
    owned_record = _build_owned_record(
        owner_id=resolved_owner_id,
        attributes=owner_attributes,
        reservation_token=token,
    )

    redis_cache, error = await _get_redis_cache()
    if error is not None or redis_cache is None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            error=error,
        )

    client = await _raw_redis_client(redis_cache)
    namespaced = _namespaced_key(redis_cache, cache_key)
    lua = """
    local raw = redis.call('GET', KEYS[1])
    if not raw then
      return {0, 'missing'}
    end
    local ok, current = pcall(cjson.decode, raw)
    if not ok or type(current) ~= 'table' then
      return {-1, 'malformed'}
    end
    if current['state'] == 'owned' then
      return {2, cjson.encode(current)}
    end
    if current['state'] ~= 'reserved' or current['reservation_token'] ~= ARGV[1] then
      return {0, cjson.encode(current)}
    end
    local payload = cjson.decode(ARGV[2])
    if current['reserved_at_epoch'] ~= nil then
      payload['reserved_at_epoch'] = current['reserved_at_epoch']
    end
    redis.call('SET', KEYS[1], cjson.encode(payload))
    redis.call('PERSIST', KEYS[1])
    return {1, cjson.encode(payload)}
    """
    try:
        result = await client.eval(
            lua,
            1,
            namespaced,
            token,
            json.dumps(owned_record),
        )
    except Exception as exc:  # noqa: BLE001
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            error=f"session_owner: promote failed: {exc}",
        )

    code = int(result[0]) if isinstance(result, (list, tuple)) and result else -99
    raw_payload = (
        result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else None
    )
    payload: Optional[Payload] = None
    if isinstance(raw_payload, (bytes, bytearray)):
        raw_payload = raw_payload.decode("utf-8")
    if isinstance(raw_payload, str) and raw_payload not in {"missing", "malformed"}:
        try:
            decoded = json.loads(raw_payload)
            if isinstance(decoded, dict):
                payload = cast(Payload, decoded)
        except Exception:  # noqa: BLE001
            payload = None

    if code == 1:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.PROMOTED,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            owner_id=resolved_owner_id,
            owner_record=payload or owned_record,
        )
    if code == 2:
        existing_owner = (
            _clean_optional_str(payload.get(_RECORD_OWNER_FIELD))
            if isinstance(payload, Mapping)
            else None
        )
        if (
            isinstance(payload, Mapping)
            and existing_owner == resolved_owner_id
            and _attributes_exactly_equal(
                left=_owner_attributes(payload),
                right=owner_attributes,
            )
        ):
            return SessionOwnerMutationResult(
                outcome=SessionOwnerMutationOutcome.ALREADY_OWNED,
                session_identity=cleaned,
                cache_key=cache_key,
                reservation_token=token,
                owner_id=existing_owner,
                owner_record=payload,
            )
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.CONFLICT,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            owner_id=existing_owner,
            owner_record=payload,
            error="session_owner: promote found different owned record",
        )
    if code == 0 and raw_payload == "missing":
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.NOT_HELD,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            error="session_owner: reservation missing at promote",
        )
    return SessionOwnerMutationResult(
        outcome=SessionOwnerMutationOutcome.CONFLICT,
        session_identity=cleaned,
        cache_key=cache_key,
        reservation_token=token,
        owner_record=payload,
        error="session_owner: promote lost reservation token",
    )


async def release_session_owner_reservation(
    *,
    session_identity: Optional[str],
    reservation_token: Optional[str],
) -> SessionOwnerMutationResult:
    """Release only our still-reserved tokenized hold. Never deletes owned."""

    cleaned = _clean_optional_str(session_identity)
    token = _clean_optional_str(reservation_token)
    if cleaned is None or token is None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.SKIPPED,
            session_identity=cleaned,
            reservation_token=token,
        )
    cleaned = _strip_legacy_affinity_prefixes(cleaned)
    cache_key = build_aawm_alias_routing_session_owner_cache_key(
        session_identity=cleaned
    )
    redis_cache, error = await _get_redis_cache()
    if error is not None or redis_cache is None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            error=error,
        )
    client = await _raw_redis_client(redis_cache)
    namespaced = _namespaced_key(redis_cache, cache_key)
    lua = """
    local raw = redis.call('GET', KEYS[1])
    if not raw then
      return 0
    end
    local ok, current = pcall(cjson.decode, raw)
    if not ok or type(current) ~= 'table' then
      return -1
    end
    if current['state'] == 'owned' then
      return 2
    end
    if current['state'] == 'reserved' and current['reservation_token'] == ARGV[1] then
      redis.call('DEL', KEYS[1])
      return 1
    end
    return 0
    """
    try:
        result = await client.eval(lua, 1, namespaced, token)
    except Exception as exc:  # noqa: BLE001
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
            error=f"session_owner: release failed: {exc}",
        )
    code = int(result or 0)
    if code == 1:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.RELEASED,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
        )
    if code == 2:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ALREADY_OWNED,
            session_identity=cleaned,
            cache_key=cache_key,
            reservation_token=token,
        )
    return SessionOwnerMutationResult(
        outcome=SessionOwnerMutationOutcome.NOT_HELD,
        session_identity=cleaned,
        cache_key=cache_key,
        reservation_token=token,
    )


def lease_from_guard_result(
    guard: SessionOwnerGuardResult,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
) -> SessionOwnerLease:
    return SessionOwnerLease(
        session_identity=guard.session_identity,
        cache_key=guard.cache_key,
        reservation_token=guard.reservation_token,
        owner_id=guard.owner_id,
        held_reservation=guard.held_reservation,
        decision=guard.decision.value,
        attributes=dict(attributes or _owner_attributes(guard.owner_record)),
    )


async def finalize_session_owner_lease_on_success(
    lease: Optional[SessionOwnerLease],
    *,
    attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
) -> Optional[SessionOwnerMutationResult]:
    if lease is None or not lease.held_reservation or lease.promoted or lease.released:
        return None
    result = await promote_session_owner_reservation(
        session_identity=lease.session_identity,
        reservation_token=lease.reservation_token,
        attributes=attributes or lease.attributes,
        candidate=candidate,
        owner_id=lease.owner_id,
    )
    if result.outcome in {
        SessionOwnerMutationOutcome.PROMOTED,
        SessionOwnerMutationOutcome.ALREADY_OWNED,
    }:
        lease.promoted = True
    return result


async def finalize_session_owner_lease_on_failure(
    lease: Optional[SessionOwnerLease],
) -> Optional[SessionOwnerMutationResult]:
    if lease is None or not lease.held_reservation or lease.promoted or lease.released:
        return None
    result = await release_session_owner_reservation(
        session_identity=lease.session_identity,
        reservation_token=lease.reservation_token,
    )
    if result.outcome in {
        SessionOwnerMutationOutcome.RELEASED,
        SessionOwnerMutationOutcome.NOT_HELD,
        SessionOwnerMutationOutcome.ALREADY_OWNED,
    }:
        lease.released = True
    return result


async def finalize_request_session_owner_lease(
    request: Any = None,
    response: Any = None,
    *,
    lease: Optional[SessionOwnerLease] = None,
    exc: Optional[BaseException] = None,
    attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    failure_phase: str = "session_owner_nested_promote",
    raise_on_promote_failure: bool = True,
) -> Optional[SessionOwnerMutationResult]:
    """Promote on authoritative success/first-byte; release on failure.

    Shared lifecycle finalizer for alias, pass-through, and nested
    ``acompletion`` paths. Treats a returned stream/response object without a
    failing status as success (first-byte/control-plane established).
    """

    active = lease if lease is not None else get_request_session_owner_lease(request)
    if active is None or not active.held_reservation or active.promoted or active.released:
        return None
    if exc is not None:
        return await finalize_session_owner_lease_on_failure(active)

    status = getattr(response, "status_code", None)
    ok = response is not None and (
        status is None or (isinstance(status, int) and status < 300)
    )
    if not ok:
        return await finalize_session_owner_lease_on_failure(active)

    result = await finalize_session_owner_lease_on_success(
        active,
        attributes=attributes or active.attributes,
        candidate=candidate,
    )
    if (
        raise_on_promote_failure
        and result is not None
        and result.outcome
        in {
            SessionOwnerMutationOutcome.CONFLICT,
            SessionOwnerMutationOutcome.ERROR,
            SessionOwnerMutationOutcome.NOT_HELD,
        }
    ):
        raise_session_owner_redispatch_required(
            session_identity=active.session_identity,
            mutation=result,
            failure_phase=failure_phase,
            request=request,
        )
    return result


def extract_account_identity_from_context(
    *,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    kwargs: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, Any]] = None,
) -> Payload:
    """Derive non-secret account/credential lane labels from known metadata.

    Never reads raw Authorization/API keys. Never stores, returns, logs, or
    otherwise exposes raw ``chatgpt-account-id`` / ``ChatGPT-Account-Id``
    values. When that inbound header is present it is converted with the same
    one-way sha256[:12] digest used by codex_oauth/lane-key attempt metadata
    (``_hash_account_identity_value``), producing only safe
    ``account_hash`` / ``account_lane`` labels.
    """

    body = request_body if isinstance(request_body, Mapping) else {}
    metadata = body.get("litellm_metadata") if isinstance(body, Mapping) else None
    if not isinstance(metadata, Mapping):
        metadata = {}
    kw = kwargs if isinstance(kwargs, Mapping) else {}
    hdrs: dict[str, Any] = {}
    if headers is None and request is not None:
        raw_headers = getattr(request, "headers", None)
        if raw_headers is not None:
            try:
                hdrs = {str(k).lower(): v for k, v in raw_headers.items()}
            except Exception:  # noqa: BLE001
                hdrs = {}
    elif isinstance(headers, Mapping):
        hdrs = {str(k).lower(): v for k, v in headers.items()}

    def _pick(*keys: str) -> Optional[str]:
        for source in (metadata, body, kw):
            if not isinstance(source, Mapping):
                continue
            for key in keys:
                value = _clean_optional_str(source.get(key))
                if value is not None:
                    return value
        for key in keys:
            value = _clean_optional_str(hdrs.get(key.lower()))
            if value is not None:
                return value
            value = _clean_optional_str(hdrs.get(key.lower().replace("_", "-")))
            if value is not None:
                return value
        return None

    out: Payload = {}
    label = _pick(
        "account_label",
        "codex_oauth_account_label",
        "codex_auto_agent_selected_account_label",
        "anthropic_auto_agent_selected_account_label",
        "x-aawm-account-label",
    )
    # Prefer already-safe hashes/lanes from oauth/attempt metadata. Never pick
    # chatgpt-account-id here — that header is handled only via one-way digest.
    acct_hash = _pick(
        "account_hash",
        "codex_oauth_account_hash",
        "codex_auto_agent_selected_account_hash",
        "anthropic_auto_agent_selected_account_hash",
        "x-aawm-account-hash",
    )
    lane = _pick(
        "account_lane",
        "codex_oauth_lane_key",
        "lane_key",
        "codex_auto_agent_selected_account_lane",
        "anthropic_auto_agent_selected_account_lane",
        "x-aawm-account-lane",
    )
    # Codex/ChatGPT OAuth account lane: recognize inbound chatgpt-account-id
    # (header or explicit kwargs/body alias) only as a one-way safe hash/lane.
    # Reuses the established sha256[:12] digest shape from lane_keys/codex
    # attempt metadata. The raw header value is never retained.
    chatgpt_account_id = _pick(
        "chatgpt-account-id",
        "chatgpt_account_id",
        "ChatGPT-Account-Id",
    )
    # Local only — never assigned into ``out``.
    if chatgpt_account_id is not None:
        digest = _hash_account_identity_value(chatgpt_account_id)
        # Match codex_oauth / attempt-metadata safe representations.
        safe_hash = f"chatgpt-account-hash:{digest}"
        safe_lane = f"chatgpt-account:{digest}"
        if acct_hash is None or acct_hash == chatgpt_account_id:
            acct_hash = safe_hash
        if (
            lane is None
            or lane == chatgpt_account_id
            or lane == f"chatgpt-account:{chatgpt_account_id}"
        ):
            lane = safe_lane
        if label == chatgpt_account_id:
            label = None
    if label:
        out["account_label"] = label
    if acct_hash:
        out["account_hash"] = acct_hash
    if lane:
        out["account_lane"] = lane
    return out


# ---------------------------------------------------------------------------
# Reopened D1-614: one best-effort sanitized proxy WARNING and route-rollup failure for
# every handled session-owner redispatch. Observability failures must never
# alter the 409 detail or permit egress.
# ---------------------------------------------------------------------------

_SESSION_OWNER_REDISPATCH_ERROR_CODE = "aawm_session_owner_redispatch_required"
_SESSION_OWNER_LOG_STATUS_CODE = 409
_SESSION_OWNER_LOG_HASH_CHARS = 16
_SESSION_OWNER_LOG_MAX_LABEL_CHARS = 96
_SESSION_OWNER_LOG_MAX_REASON_CHARS = 240
_SESSION_OWNER_LOG_MAX_SUMMARY_CHARS = 480
_SESSION_OWNER_ROLLUP_CONTEXT_METADATA_KEY = "aawm_route_rollup_context"

# D1-614 acceptance 3: credential-shaped substrings must not survive the log
# label sanitizer. Reuse the shared RR-065/074/075/092 field-value redactor
# for keyed secrets (extended with api_key/token-shaped names), then strip
# the bare Bearer / sk- token shapes that keyed redaction does not cover.
_SESSION_OWNER_LOG_SECRET_FIELD_NAMES = (
    "api_key",
    "apikey",
    "key",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "secret",
    "token",
    "password",
)
_SESSION_OWNER_LOG_BEARER_TOKEN_RE = re.compile(
    r"(?i)\bbearer\s+['\"]?[A-Za-z0-9._~+/=-]{8,}"
)
_SESSION_OWNER_LOG_SK_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")


def _sanitize_session_owner_log_label(
    value: Any, *, max_length: int
) -> Optional[str]:
    """Bound a free-form log label to printable ASCII (route-log convention).

    Credential-shaped substrings (``api_key=...``, ``sk-...``, Bearer
    tokens) are redacted before bounding so they cannot survive in emitted
    mismatch reasons, owner account fields, or attribution labels.
    """
    cleaned = _clean_optional_str(value)
    if cleaned is None:
        return None
    sanitized = sanitize_credential_error_message(
        cleaned,
        field_names=_SESSION_OWNER_LOG_SECRET_FIELD_NAMES,
    )
    sanitized = _SESSION_OWNER_LOG_BEARER_TOKEN_RE.sub(
        "Bearer [REDACTED]", sanitized
    )
    sanitized = _SESSION_OWNER_LOG_SK_TOKEN_RE.sub("sk-[REDACTED]", sanitized)
    sanitized = re.sub(r"[^\x20-\x7E]", "_", sanitized)
    return sanitized[:max_length] or None


def _hash_session_owner_log_identifier(value: Any) -> Optional[str]:
    """Bounded one-way fingerprint for log identifiers (never raw values)."""
    cleaned = _clean_optional_str(value)
    if cleaned is None:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[
        :_SESSION_OWNER_LOG_HASH_CHARS
    ]


# request.state keys written by aawm_alias_routing.audit_context during normal
# request correlation. Read-only here; identifiers are hashed before reuse.
_SESSION_OWNER_REQUEST_CONTEXT_STATE_KEY = "aawm_alias_request_context"
_SESSION_OWNER_REQUEST_CALL_ID_STATE_KEY = "aawm_alias_request_litellm_call_id"


def _build_session_owner_rollup_kwargs(
    *,
    request: Any,
    session_identity: Optional[str],
    alias_model: Optional[str],
    failure_phase: str,
    shaped_candidate: Mapping[str, Any],
    candidate_endpoint: Optional[str],
    owner_attrs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the minimal standard kwargs metadata consumed by the rollup."""
    cached_context: Mapping[str, Any] = {}
    request_call_id: Optional[str] = None
    incoming_endpoint: Optional[str] = None
    try:
        state = getattr(request, "state", None)
        context = (
            getattr(state, _SESSION_OWNER_REQUEST_CONTEXT_STATE_KEY, None)
            if state is not None
            else None
        )
        if isinstance(context, Mapping):
            cached_context = context
            raw_context_call_id = context.get("litellm_call_id")
            if isinstance(raw_context_call_id, str):
                request_call_id = raw_context_call_id
        if state is not None and request_call_id is None:
            raw_state_call_id = getattr(
                state, _SESSION_OWNER_REQUEST_CALL_ID_STATE_KEY, None
            )
            if isinstance(raw_state_call_id, str):
                request_call_id = raw_state_call_id
        request_url = getattr(request, "url", None)
        raw_path = getattr(request_url, "path", None)
        if isinstance(raw_path, str):
            incoming_endpoint = raw_path
    except Exception:  # noqa: BLE001
        pass

    existing_rollup_context = cached_context.get(
        _SESSION_OWNER_ROLLUP_CONTEXT_METADATA_KEY
    )
    existing = (
        existing_rollup_context
        if isinstance(existing_rollup_context, Mapping)
        else {}
    )
    max_label = _SESSION_OWNER_LOG_MAX_LABEL_CHARS

    def _rollup_label(*values: Any, default: str) -> str:
        for value in values:
            sanitized = _sanitize_session_owner_log_label(
                value, max_length=max_label
            )
            if sanitized is not None:
                return sanitized
        return default

    rollup_context = {
        "group_header_label": _rollup_label(
            existing.get("group_header_label"),
            cached_context.get("rollup_group_header_label"),
            cached_context.get("client_product_label"),
            default="LiteLLM Proxy",
        ),
        "incoming_endpoint": _rollup_label(
            existing.get("incoming_endpoint"),
            incoming_endpoint,
            failure_phase,
            default="session-owner",
        ),
        "outgoing_target": _rollup_label(
            existing.get("outgoing_target"),
            candidate_endpoint,
            owner_attrs.get("endpoint_contract"),
            default="no-provider-egress",
        ),
        "model_label": _rollup_label(
            existing.get("model_label"),
            alias_model,
            shaped_candidate.get("model"),
            owner_attrs.get("model"),
            default="session-owner",
        ),
        "reasoning_effort": _rollup_label(
            existing.get("reasoning_effort"),
            default="none",
        ),
    }
    metadata: dict[str, Any] = {
        _SESSION_OWNER_ROLLUP_CONTEXT_METADATA_KEY: rollup_context,
    }
    call_id_hash = _hash_session_owner_log_identifier(request_call_id)
    if call_id_hash is not None:
        metadata["litellm_call_id"] = call_id_hash
    session_hash = _hash_session_owner_log_identifier(session_identity)
    if session_hash is not None:
        metadata["canonical_session_identity"] = session_hash
    return {"litellm_params": {"metadata": metadata}}


def _build_session_owner_redispatch_summary(
    *,
    mismatch_reason: Optional[str],
    failure_phase: str,
) -> str:
    reason = _sanitize_session_owner_log_label(
        mismatch_reason or failure_phase,
        max_length=_SESSION_OWNER_LOG_MAX_REASON_CHARS,
    ) or "session-owner mismatch"
    return (
        "LiteLLM Proxy: HTTP 409 session-owner mismatch requires redispatch; "
        "redispatch_required=true; attempted_provider_call=false; "
        f"reason={reason}; action=redispatch with a fresh session"
    )[:_SESSION_OWNER_LOG_MAX_SUMMARY_CHARS]


def _emit_session_owner_redispatch_observability(
    *,
    session_identity: Optional[str],
    failure_phase: str,
    mismatch_reason: Optional[str],
    alias_model: Optional[str],
    shaped_candidate: Mapping[str, Any],
    candidate_endpoint: Optional[str],
    owner_attrs: Mapping[str, Any],
    request: Any,
) -> None:
    """Emit one proxy WARNING and one rollup failure. Never raises."""
    summary = _build_session_owner_redispatch_summary(
        mismatch_reason=mismatch_reason,
        failure_phase=failure_phase,
    )
    try:
        verbose_proxy_logger.warning(
            "%s",
            summary,
            extra={
                "source": "session_owner_affinity",
                "status_code": _SESSION_OWNER_LOG_STATUS_CODE,
                "failure_kind": "session_owner_mismatch",
                "redispatch_required": True,
                "attempted_provider_call": False,
            },
            exc_info=False,
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from litellm.proxy.aawm_route_logging import (
            record_aawm_route_rollup_failure,
        )

        rollup_kwargs = _build_session_owner_rollup_kwargs(
            request=request,
            session_identity=session_identity,
            alias_model=alias_model,
            failure_phase=failure_phase,
            shaped_candidate=shaped_candidate,
            candidate_endpoint=candidate_endpoint,
            owner_attrs=owner_attrs,
        )
        record_aawm_route_rollup_failure(
            rollup_kwargs,
            message=summary,
            status="Failed",
        )
    except Exception:  # noqa: BLE001
        pass


def raise_session_owner_redispatch_required(
    *,
    session_identity: Optional[str],
    guard: Optional[SessionOwnerGuardResult] = None,
    mutation: Optional[SessionOwnerMutationResult] = None,
    alias_model: Optional[str] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    failure_phase: str = "session_owner_mismatch",
    message: Optional[str] = None,
    attribution: Optional[Mapping[str, Any]] = None,
    request: Any = None,
) -> None:
    """Fail before egress with structured redispatch_required. Never returns.

    Reopened D1-614 observability is best-effort and cannot alter this response.
    """

    owner_record: Optional[Mapping[str, Any]] = None
    owner_id: Optional[str] = None
    mismatch_reason: Optional[str] = None
    decision = "redispatch_required"
    cache_key: Optional[str] = None
    claim_outcome: Optional[str] = None

    if guard is not None:
        owner_record = guard.owner_record
        owner_id = guard.owner_id
        mismatch_reason = guard.mismatch_reason
        decision = guard.decision.value
        cache_key = guard.cache_key
        session_identity = guard.session_identity or session_identity
    if mutation is not None:
        owner_record = mutation.owner_record or owner_record
        owner_id = mutation.owner_id or owner_id
        mismatch_reason = mutation.error or mismatch_reason or mutation.outcome.value
        claim_outcome = mutation.outcome.value
        cache_key = mutation.cache_key or cache_key
        session_identity = mutation.session_identity or session_identity
        decision = "redispatch_required"

    owner_attrs = _owner_attributes(owner_record)
    shaped_candidate: dict[str, Any] = {}
    candidate_endpoint: Optional[str] = None
    if isinstance(candidate, Mapping):
        for key in (
            "provider",
            "model",
            "route_family",
            "last_resort",
            "codex_oauth_account_label",
            "codex_oauth_account_hash",
            "codex_oauth_lane_key",
        ):
            if candidate.get(key) is not None:
                shaped_candidate[key] = candidate.get(key)
        candidate_endpoint = _clean_optional_str(
            candidate.get("endpoint_contract")
        )
    if not shaped_candidate and owner_attrs:
        shaped_candidate = {
            "provider": owner_attrs.get("provider"),
            "model": owner_attrs.get("model"),
            "route_family": owner_attrs.get("route_family"),
            "account_lane": owner_attrs.get("account_lane"),
        }

    provenance = build_session_owner_provenance(
        session_identity=session_identity,
        decision=decision,
        owner_record=owner_record,
        owner_id=owner_id if isinstance(owner_id, str) else None,
        mismatch_reason=mismatch_reason,
        cache_key=cache_key,
        claim_outcome=claim_outcome,
    )

    detail: dict[str, Any] = {
        "error": {
            "message": message
            or (
                "Session ownership requires a fresh dispatch. The current "
                "session is pinned to a different or unavailable owner; do not "
                "continue this session against another provider/model/route/"
                "account. Redispatch with a new session identity."
            ),
            "type": "invalid_request_error",
            "code": _SESSION_OWNER_REDISPATCH_ERROR_CODE,
        },
        "redispatch_required": True,
        "redispatch_reason": mismatch_reason or failure_phase,
        "failure_phase": failure_phase,
        "attempted_provider_call": False,
        "canonical_session_identity": session_identity,
        "session_owner": provenance,
        "candidate": shaped_candidate,
    }
    if alias_model is not None:
        detail["alias_model"] = alias_model
        detail["redispatch_model"] = alias_model
    if owner_attrs:
        detail["selected_provider"] = owner_attrs.get("provider")
        detail["selected_model"] = owner_attrs.get("model")
        detail["selected_route_family"] = owner_attrs.get("route_family")

    # Keep the legacy attribution argument for callers; reopened D1-614 intentionally
    # does not emit its values.
    _ = attribution
    _emit_session_owner_redispatch_observability(
        session_identity=session_identity,
        failure_phase=failure_phase,
        mismatch_reason=mismatch_reason,
        alias_model=alias_model,
        shaped_candidate=shaped_candidate,
        candidate_endpoint=candidate_endpoint,
        owner_attrs=owner_attrs,
        request=request,
    )

    raise HTTPException(status_code=409, detail=detail)




_REQUEST_STATE_LEASE_ATTR = "_aawm_session_owner_lease"
_REQUEST_STATE_GUARDED_ATTR = "_aawm_session_owner_guarded"


def get_request_session_owner_lease(request: Any) -> Optional[SessionOwnerLease]:
    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        lease = object.__getattribute__(state, _REQUEST_STATE_LEASE_ATTR)
    except AttributeError:
        lease = getattr(state, _REQUEST_STATE_LEASE_ATTR, None)
    return lease if isinstance(lease, SessionOwnerLease) else None


def set_request_session_owner_lease(
    request: Any, lease: Optional[SessionOwnerLease]
) -> None:
    if request is None:
        return
    state = getattr(request, "state", None)
    if state is None:
        return
    setattr(state, _REQUEST_STATE_LEASE_ATTR, lease)
    setattr(state, _REQUEST_STATE_GUARDED_ATTR, True)


def request_session_owner_already_guarded(request: Any) -> bool:
    if request is None:
        return False
    state = getattr(request, "state", None)
    if state is None:
        return False
    try:
        value = object.__getattribute__(state, _REQUEST_STATE_GUARDED_ATTR)
    except AttributeError:
        value = getattr(state, _REQUEST_STATE_GUARDED_ATTR, False)
    # Strict True only — MagicMock/default objects must not count as guarded.
    return value is True


def reset_released_request_session_owner_guard(request: Any) -> bool:
    """Allow a fresh-request retry after its reservation was released."""
    if request is None:
        return False
    state = getattr(request, "state", None)
    if state is None:
        return False
    lease = get_request_session_owner_lease(request)
    if lease is not None and not lease.released:
        return False
    setattr(state, _REQUEST_STATE_LEASE_ATTR, None)
    setattr(state, _REQUEST_STATE_GUARDED_ATTR, False)
    return True


def clear_non_held_request_session_owner_lease(request: Any) -> bool:
    """Clear only a failed guard result that did not acquire a reservation."""

    if request is None:
        return False
    state = getattr(request, "state", None)
    if state is None:
        return False
    lease = get_request_session_owner_lease(request)
    if lease is not None and lease.held_reservation:
        return False
    setattr(state, _REQUEST_STATE_LEASE_ATTR, None)
    setattr(state, _REQUEST_STATE_GUARDED_ATTR, False)
    return True


def is_exact_owned_session_owner_route_mismatch(
    *,
    guard: SessionOwnerGuardResult,
    requested_attributes: Mapping[str, Any],
) -> bool:
    """Return whether a guard failure is an exact, complete owned-route mismatch."""

    if guard.decision is not SessionOwnerGuardDecision.REDISPATCH_REQUIRED:
        return False
    owner_record = guard.owner_record
    if not isinstance(owner_record, Mapping) or _record_state(owner_record) != "owned":
        return False
    owner_attributes = _owner_attributes(owner_record)
    requested = _core_owner_attributes(requested_attributes)
    if not owner_attributes or not requested:
        return False
    if incomplete_owner_attribute_reason(owner_attributes, for_promotion=True):
        return False
    if incomplete_owner_attribute_reason(requested, for_promotion=True):
        return False
    if _attributes_exactly_equal(left=owner_attributes, right=requested):
        return False
    return dict(owner_attributes) != dict(requested)


async def ensure_session_owner_guard_for_request(
    *,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    session_identity: Optional[str] = None,
    requested_attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    owner_id: Optional[str] = None,
    require_exact_attributes: bool = False,
    alias_model: Optional[str] = None,
    failure_phase: str = "session_owner_pre_egress",
    raise_on_redispatch: bool = True,
) -> SessionOwnerGuardResult:
    """Idempotent request-scoped guard used by every route family.

    If this request already holds a lease, renew/validate rather than creating
    a second competing reservation.
    """

    existing = get_request_session_owner_lease(request)
    active_lease = (
        existing
        if existing is not None and not existing.released and not existing.promoted
        else None
    )
    token = active_lease.reservation_token if active_lease is not None else None
    guard = await guard_session_owner_before_egress(
        session_identity=session_identity,
        request=request,
        request_body=request_body,
        requested_attributes=requested_attributes
        or (active_lease.attributes if active_lease is not None else None),
        candidate=candidate,
        owner_id=owner_id
        or (active_lease.owner_id if active_lease is not None else None),
        reservation_token=token,
        require_exact_attributes=require_exact_attributes,
    )
    if (
        raise_on_redispatch
        and guard.decision is SessionOwnerGuardDecision.REDISPATCH_REQUIRED
    ):
        raise_session_owner_redispatch_required(
            session_identity=guard.session_identity or session_identity,
            guard=guard,
            alias_model=alias_model,
            candidate=candidate or requested_attributes,
            failure_phase=failure_phase,
            request=request,
        )
    if active_lease is not None and guard.held_reservation:
        active_lease.reservation_token = guard.reservation_token
        active_lease.held_reservation = True
        active_lease.decision = guard.decision.value
        active_lease.owner_id = guard.owner_id or active_lease.owner_id
        active_lease.promoted = False
        active_lease.released = False
        set_request_session_owner_lease(request, active_lease)
    else:
        lease = lease_from_guard_result(
            guard,
            attributes=requested_attributes
            or (active_lease.attributes if active_lease is not None else None),
        )
        set_request_session_owner_lease(request, lease)
    return guard


def refresh_request_session_owner_lease_attributes(
    request: Any,
    attributes: Optional[Mapping[str, Any]],
) -> None:
    """Refresh the held lease's promotion attributes after concrete resolution.

    Reservation-time attributes on a request-scoped lease may be generic
    (e.g. a nested dispatch placeholder). Once the concrete provider, model,
    route family, endpoint contract, state format, and safe account lane are
    resolved -- but before provider send -- call this so the lease finalized
    on success promotes to the exact concrete owner identity. No-op when the
    request holds no lease or no attributes are supplied.
    """

    if not attributes:
        return
    lease = get_request_session_owner_lease(request)
    if lease is None:
        return
    lease.attributes = dict(
        _core_owner_attributes(build_session_owner_attributes(extra=attributes))
    )
# Back-compat aliases used by draft call sites / tests naming.
SessionOwnerConsultDecision = SessionOwnerGuardDecision
SessionOwnerClaimOutcome = SessionOwnerMutationOutcome
SessionOwnerConsultResult = SessionOwnerGuardResult
SessionOwnerClaimResult = SessionOwnerMutationResult


async def consult_session_owner_before_egress(**kwargs: Any) -> SessionOwnerGuardResult:
    """Alias of :func:`guard_session_owner_before_egress` (reserve path)."""

    return await guard_session_owner_before_egress(**kwargs)


async def claim_session_owner_on_success(**kwargs: Any) -> SessionOwnerMutationResult:
    """Promote path alias — requires reservation_token in kwargs."""

    return await promote_session_owner_reservation(**kwargs)
