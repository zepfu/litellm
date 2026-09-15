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
from typing import Any, Awaitable, Callable, Mapping, Optional, TypeVar, cast

from fastapi import HTTPException

from litellm._logging import verbose_aawm_route_logger, verbose_proxy_logger
from litellm.llms.xai.route_descriptors import (
    GROK_NATIVE_OAUTH_ROUTE_FAMILY,
    XAI_OAUTH_ROUTE_FAMILY,
)
from litellm.secret_managers.credential_error_sanitizer import (
    sanitize_credential_error_message,
)

from . import durable
from .audit_persist import _emit_aawm_terminal_error
from .types import Payload


_SessionOwnerLeaseOperationT = TypeVar("_SessionOwnerLeaseOperationT")


class SessionOwnerRecordState(str, Enum):
    RESERVED = "reserved"
    OWNED = "owned"


class SessionOwnerGuardDecision(str, Enum):
    NO_SESSION = "no_session"
    UNOWNED_RESERVED = "unowned_reserved"
    RESERVATION_RENEWED = "reservation_renewed"
    COMPATIBLE_OWNER = "compatible_owner"
    REDISPATCH_REQUIRED = "redispatch_required"


class SessionOwnerLeasePolicy(str, Enum):
    PERSIST_ON_COMPLETED = "persist_on_completed"
    RELEASE_ON_TERMINAL = "release_on_terminal"


class SessionOwnerMutationOutcome(str, Enum):
    PROMOTED = "promoted"
    ALREADY_OWNED = "already_owned"
    RELEASED = "released"
    NOT_HELD = "not_held"
    CONFLICT = "conflict"
    ERROR = "error"
    SKIPPED = "skipped"


class SessionOwnerLeaseRenewalError(RuntimeError):
    """The in-flight reservation could not be renewed safely."""

    failure_phase = "session_owner_reservation_renewal"

    def __init__(
        self,
        *,
        session_identity: Optional[str],
        reason: str = "reservation renewal failed",
    ) -> None:
        self.session_identity = session_identity
        self.reason = reason
        super().__init__(f"session_owner: {reason}")


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
    renewal_task: Optional[Any] = field(
        default=None,
        repr=False,
        compare=False,
    )
    finalizing: bool = False
    # Native OpenAI Responses ownership cannot be promoted from HTTP 2xx. The
    # lease remains renewable until the final wire coordinator reports a
    # terminal disposition.
    wire_terminal_pending: bool = False
    wire_disposition: Optional[str] = None
    last_finalization_outcome: Optional[str] = None
    policy: SessionOwnerLeasePolicy = (
        SessionOwnerLeasePolicy.PERSIST_ON_COMPLETED
    )


@dataclass(frozen=True)
class SessionOwnerLeaseRebindResult:
    """Exact outcome of a validated portable failover lease rebinding."""

    rebound: bool
    rejection_reason: Optional[str] = None
    source_attributes: Optional[Mapping[str, Any]] = None


def _normalize_session_owner_lease_policy(
    value: Any,
) -> SessionOwnerLeasePolicy:
    if isinstance(value, SessionOwnerLeasePolicy):
        return value
    try:
        return SessionOwnerLeasePolicy(str(value))
    except (TypeError, ValueError):
        return SessionOwnerLeasePolicy.PERSIST_ON_COMPLETED


def session_owner_lease_is_release_only(
    lease: Optional[SessionOwnerLease],
) -> bool:
    return (
        lease is not None
        and _normalize_session_owner_lease_policy(lease.policy)
        is SessionOwnerLeasePolicy.RELEASE_ON_TERMINAL
    )


def session_owner_lease_success_outcomes(
    lease: Optional[SessionOwnerLease],
) -> set[SessionOwnerMutationOutcome]:
    if session_owner_lease_is_release_only(lease):
        return {
            SessionOwnerMutationOutcome.RELEASED,
            SessionOwnerMutationOutcome.NOT_HELD,
        }
    return {
        SessionOwnerMutationOutcome.PROMOTED,
        SessionOwnerMutationOutcome.ALREADY_OWNED,
    }


def _session_owner_lease_invariant_result(
    lease: SessionOwnerLease,
    *,
    reason: str,
) -> SessionOwnerMutationResult:
    return SessionOwnerMutationResult(
        outcome=SessionOwnerMutationOutcome.ERROR,
        session_identity=lease.session_identity,
        cache_key=lease.cache_key,
        reservation_token=lease.reservation_token,
        owner_id=lease.owner_id,
        error=f"session_owner: lease policy invariant violated: {reason}",
    )


def _session_owner_lease_release_invariant(
    lease: Optional[SessionOwnerLease],
    *,
    force_release_only: bool = False,
) -> Optional[SessionOwnerMutationResult]:
    if lease is None or not (
        force_release_only or session_owner_lease_is_release_only(lease)
    ):
        return None
    if lease.promoted:
        return _session_owner_lease_invariant_result(
            lease,
            reason="release-only lease was promoted",
        )
    return None


def _request_is_codex_auto_review(
    request: Any,
    *,
    alias_model: Optional[str] = None,
) -> bool:
    normalized_alias = (
        alias_model.strip().casefold()
        if isinstance(alias_model, str)
        else ""
    )
    return (
        normalized_alias
        in {
            "codex-auto-review",
            "auto-review",
            "chatgpt/codex-auto-review",
        }
        or get_request_codex_auto_review_parent_session_identity(request)
        is not None
        or get_request_codex_auto_review_session_identity(request) is not None
    )


def resolve_session_owner_lease_policy(
    request: Any,
    *,
    alias_model: Optional[str] = None,
    existing_lease: Optional[SessionOwnerLease] = None,
    policy: Optional[SessionOwnerLeasePolicy] = None,
) -> SessionOwnerLeasePolicy:
    if policy is not None:
        return _normalize_session_owner_lease_policy(policy)
    if _request_is_codex_auto_review(request, alias_model=alias_model):
        return SessionOwnerLeasePolicy.RELEASE_ON_TERMINAL
    if session_owner_lease_is_release_only(existing_lease):
        return SessionOwnerLeasePolicy.RELEASE_ON_TERMINAL
    return SessionOwnerLeasePolicy.PERSIST_ON_COMPLETED


@dataclass(frozen=True)
class SessionOwnerReplaySafetyResult:
    """Pure structural replay-safety classification for one request body."""

    safe: bool
    field_path: Optional[str] = None
    classification: Optional[str] = None


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
_CODEX_AUTO_REVIEW_SESSION_OWNER_IDENTITY_PREFIX = (
    "aawm-codex-auto-review-owner-v1:"
)
_REQUEST_STATE_CODEX_AUTO_REVIEW_SESSION_IDENTITY_ATTR = (
    "_aawm_codex_auto_review_session_identity"
)
_REQUEST_STATE_CODEX_AUTO_REVIEW_PARENT_SESSION_IDENTITY_ATTR = (
    "_aawm_codex_auto_review_parent_session_identity"
)
_REQUEST_STATE_CONTINUITY_RECEIPT_ATTR = "_aawm_session_owner_continuity_receipt"
_CONTINUITY_IDENTITY_KEYS = (
    "thread_id",
    "aawm_thread_id",
    "codex_thread_id",
    "claude_thread_id",
    "session_id",
    "aawm_session_id",
    "codex_session_id",
    "claude_session_id",
    "anthropic_session_id",
)
_CONTINUITY_PHASES = frozenset(
    {"identity_resolve", "owner_lookup", "owner_guard", "owner_finalize",
     "owner_rejection", "legacy_affinity"}
)
_CONTINUITY_SOURCES = frozenset(
    {"explicit", "request_effective", "codex_auto_review", "session_identity",
     "durable_cache", "redis", "request_lease", "session_owner_guard", "success",
     "failure", "non_success_response", "redispatch", "memory", "durable",
     "wire_terminal"}
    | {
        f"{location}.{key}"
        for location in ("client_metadata", "litellm_metadata", "body")
        for key in _CONTINUITY_IDENTITY_KEYS
    }
    | {
        f"header.{key}"
        for key in (
            "thread_id", "x_thread_id", "x_aawm_thread_id", "x_codex_thread_id",
            "x_claude_thread_id", "session_id", "x_session_id",
            "x_aawm_session_id", "x_codex_session_id", "x_claude_session_id",
            "anthropic_beta_session_id", "aawm_session_id", "codex_session_id",
            "claude_session_id",
        )
    }
)
_CONTINUITY_OUTCOMES = frozenset(
    {"resolved", "owned", "reserved", "missing", "error", "unknown",
     "skipped_missing_identity", "pending_wire_terminal", "written", "unverified",
     "not_written", "already_finalized"}
    | {decision.value for decision in SessionOwnerGuardDecision}
    | {outcome.value for outcome in SessionOwnerMutationOutcome}
)
_CONTINUITY_REASONS = frozenset(
    {"durable_cache_unavailable", "owner_read_failed", "reservation_wait_failed",
     "identity_conflict", "guard_rejected", "mutation_failed"}
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
    "account_scope",
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
    "account_scope",
    "endpoint_contract",
    "state_format",
    "credential_affinity",
)

# Same-hosted-provider last-used fields stored on the owner record.
# Model stays mutable for every hosted-provider match. Account identity is
# mutable only for the OPENAI-020 OpenAI hosted-provider contract.
_MUTABLE_SAME_HOSTED_PROVIDER_ATTRIBUTE_KEYS = (
    "model",
    "requested_model",
)
_MUTABLE_OPENAI_ACCOUNT_ATTRIBUTE_KEYS = (
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

_REPLAY_SAFETY_EXPLICIT_REFERENCE_KEYS = frozenset(
    {
        "item_id",
        "item_reference",
        "provider_item_id",
        "response_item_id",
    }
)
_REPLAY_SAFETY_TERMINAL_FIELD_NAMES = (
    _REPLAY_SAFETY_EXPLICIT_REFERENCE_KEYS
    | frozenset({"previous_response_id", "id"})
)
_REPLAY_SAFETY_FULL_REASONING_ITEM_KEYS = frozenset(
    {
        "summary",
        "encrypted_content",
        "content",
        "internal_chat_message_metadata_passthrough",
    }
)
_REPLAY_SAFETY_CLASSIFICATIONS = frozenset(
    {
        "previous_response_id",
        "id_only_reasoning_reference",
        "explicit_item_reference",
        "invalid_body_shape",
    }
)
_REPLAY_SAFETY_MAX_FIELD_PATH_CHARS = 256
_REPLAY_SAFETY_MAPPING_PATH_SEGMENT = object()

# Short renewable hold while upstream I/O is in flight. Not a fixed ownership
# expiry — owned records are persistent until explicit retirement.
_DEFAULT_RESERVATION_TTL_SECONDS = 120.0
_MIN_RESERVATION_TTL_SECONDS = 30.0
_MAX_RESERVATION_TTL_SECONDS = 900.0
_DEFAULT_RESERVATION_RENEWAL_INTERVAL_SECONDS = 30.0
_MIN_RESERVATION_RENEWAL_INTERVAL_SECONDS = 1.0
_DEFAULT_RESERVATION_WAIT_TIMEOUT_SECONDS = 0.25
_DEFAULT_RESERVATION_WAIT_POLL_SECONDS = 0.025
_MAX_RESERVATION_WAIT_TIMEOUT_SECONDS = 1.0
_MAX_RESERVATION_WAIT_POLL_SECONDS = 0.1
# A replay-safe competing request may legitimately overlap a long upstream
# turn. Keep retrying until the normal reservation TTL has elapsed so clients
# do not spin on repeated 409 responses while the original request settles.
DEFAULT_COMPETING_RESERVATION_RETRY_ATTEMPTS = int(
    _DEFAULT_RESERVATION_TTL_SECONDS
)

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


def _clean_identity_str(value: Any) -> Optional[str]:
    """Accept only non-empty strings for externally supplied identities."""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
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


def record_session_owner_continuity_receipt(
    request: Any,
    *,
    phase: str,
    source: Optional[str] = None,
    session_identity: Optional[str] = None,
    outcome: Optional[str] = None,
    cache_key: Optional[str] = None,
    reason_code: Optional[str] = None,
    held_reservation: Optional[bool] = None,
) -> None:
    """Keep bounded, secret-free evidence for identity and owner transitions."""
    try:
        state = getattr(request, "state", None) if request is not None else None
        if state is None:
            return
        existing = get_session_owner_continuity_receipt(request) or {}
        sequence = int(existing.get("sequence") or 0) + 1
        phase_value = phase if phase in _CONTINUITY_PHASES else "unknown"
        normalized_source = source.replace("-", "_") if isinstance(source, str) else None
        source_value = (
            normalized_source if normalized_source in _CONTINUITY_SOURCES else "unknown"
        )
        event: Payload = {
            "sequence": sequence,
            "phase": phase_value,
            "source": source_value,
            "outcome": outcome if outcome in _CONTINUITY_OUTCOMES else "unknown",
            "reason_code": reason_code if reason_code in _CONTINUITY_REASONS else None,
            "canonical_session_identity_hash": _hash_session_owner_log_identifier(
                session_identity
            ),
            (
                "legacy_key_hash" if phase_value == "legacy_affinity" else "owner_key_hash"
            ): _hash_session_owner_log_identifier(cache_key),
            "held_reservation": held_reservation if isinstance(held_reservation, bool) else None,
            "recorded_at_epoch": round(time.time(), 3),
        }
        event = {key: value for key, value in event.items() if value is not None}
        events = list(existing.get("events") or ())
        events.append(event)
        identity = existing.get("identity")
        if identity is None and phase_value == "identity_resolve" and session_identity:
            identity = {
                **event,
                "identity_kind": (
                    "thread" if "thread_id" in source_value else
                    "session" if "session_id" in source_value else "server_derived"
                ),
            }
        dropped_events = int(existing.get("dropped_events") or 0)
        dropped_events += max(0, len(events) - 16)
        receipt = {
            "latest": event,
            "events": events[-16:],
            "sequence": sequence,
            "dropped_events": dropped_events,
            "identity": identity,
            "intake": existing.get("intake"),
        }
        setattr(state, _REQUEST_STATE_CONTINUITY_RECEIPT_ATTR, receipt)
    except Exception:  # noqa: BLE001
        return


def get_session_owner_continuity_receipt(request: Any) -> Optional[Payload]:
    try:
        state = getattr(request, "state", None) if request is not None else None
        receipt = (
            getattr(state, _REQUEST_STATE_CONTINUITY_RECEIPT_ATTR, None)
            if state is not None
            else None
        )
        if not isinstance(receipt, Mapping):
            return None
        def snapshot_event(value: Any) -> Payload:
            if not isinstance(value, Mapping):
                return {}
            result: Payload = {}
            for key, labels in (
                ("phase", _CONTINUITY_PHASES),
                ("source", _CONTINUITY_SOURCES),
                ("outcome", _CONTINUITY_OUTCOMES),
                ("reason_code", _CONTINUITY_REASONS),
                ("identity_kind", {"thread", "session", "server_derived"}),
            ):
                if isinstance(value.get(key), str) and value[key] in labels:
                    result[key] = value[key]
            for key in ("canonical_session_identity_hash", "owner_key_hash", "legacy_key_hash"):
                item = value.get(key)
                if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{16}", item):
                    result[key] = item
            for key in ("sequence", "recorded_at_epoch"):
                item = value.get(key)
                if type(item) in (int, float) and math.isfinite(item) and item >= 0:
                    result[key] = item
            if isinstance(value.get("held_reservation"), bool):
                result["held_reservation"] = value["held_reservation"]
            return result

        events = receipt.get("events")
        identity = snapshot_event(receipt.get("identity"))
        intake = receipt.get("intake")
        intake_specs = {
            "body": _CONTINUITY_IDENTITY_KEYS,
            "client_metadata": _CONTINUITY_IDENTITY_KEYS,
            "litellm_metadata": _CONTINUITY_IDENTITY_KEYS,
            "headers": (
                "thread_id", "x_thread_id", "x_aawm_thread_id",
                "x_codex_thread_id", "x_claude_thread_id", "session_id",
                "x_session_id", "x_aawm_session_id", "x_codex_session_id",
                "x_claude_session_id", "anthropic_beta_session_id",
            ),
        }
        intake_snapshot: Optional[Payload] = None
        if isinstance(intake, Mapping):
            intake_snapshot = {}
            for key, expected_fields in intake_specs.items():
                raw_values = intake.get(key)
                if (
                    not isinstance(raw_values, list)
                    or len(raw_values) != len(expected_fields)
                ):
                    continue
                accepted: list[str] = []
                for expected_field, raw_value in zip(expected_fields, raw_values):
                    if not isinstance(raw_value, str):
                        break
                    field, separator, state_value = raw_value.partition(":")
                    if (
                        not separator
                        or field != expected_field
                        or state_value not in {"string", "missing", "invalid"}
                    ):
                        break
                    accepted.append(raw_value)
                if len(accepted) == len(expected_fields):
                    intake_snapshot[key] = list(accepted)
        return {
            "latest": snapshot_event(receipt.get("latest")),
            "events": [snapshot_event(event) for event in events[-16:]]
            if isinstance(events, list) else [],
            "sequence": max(0, int(receipt.get("sequence") or 0)),
            "dropped_events": max(0, int(receipt.get("dropped_events") or 0)),
            "identity": identity or None,
            "intake": intake_snapshot,
        }
    except Exception:  # noqa: BLE001
        return None


def _record_session_identity_intake(request: Any, body: Mapping[str, Any]) -> None:
    """Record only supported field names and types, never request content."""
    try:
        state = getattr(request, "state", None)
        if state is None:
            return
        receipt = get_session_owner_continuity_receipt(request) or {}
        if receipt.get("intake") is not None:
            return
        headers = {
            str(key).lower().replace("-", "_"): value
            for key, value in request.headers.items()
        }
        intake = {}
        for location, mapping, keys in (
            ("body", body, _CONTINUITY_IDENTITY_KEYS),
            ("client_metadata", body.get("client_metadata"), _CONTINUITY_IDENTITY_KEYS),
            ("litellm_metadata", body.get("litellm_metadata"), _CONTINUITY_IDENTITY_KEYS),
            ("headers", headers, (
                "thread_id", "x_thread_id", "x_aawm_thread_id", "x_codex_thread_id",
                "x_claude_thread_id", "session_id", "x_session_id",
                "x_aawm_session_id", "x_codex_session_id", "x_claude_session_id",
                "anthropic_beta_session_id",
            )),
        ):
            values = mapping if isinstance(mapping, Mapping) else {}
            intake[location] = [
                f"{key}:"
                + ("missing" if key not in values else
                   "string" if _clean_identity_str(values[key]) is not None else "invalid")
                for key in keys
            ]
        receipt["intake"] = intake
        setattr(state, _REQUEST_STATE_CONTINUITY_RECEIPT_ATTR, receipt)
    except Exception:  # noqa: BLE001
        return


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


def activate_codex_auto_review_session_owner_identity(
    *,
    request: Any,
    parent_session_identity: Optional[str],
    request_call_identity: Optional[str] = None,
) -> Optional[str]:
    """Bind one request-local owner identity for a Codex auto-review request.

    The logical guardian identity is retained separately for correlation.  The
    owner identity is opaque, unique to the request-call identity, and must
    never become durable guardian affinity.
    """

    if request is None:
        return None
    parent_identity = _clean_optional_str(parent_session_identity)
    request_call = _clean_optional_str(request_call_identity)
    if parent_identity is None or request_call is None:
        return None
    digest = hashlib.sha256(
        (
            _CODEX_AUTO_REVIEW_SESSION_OWNER_IDENTITY_PREFIX
            + "\x00"
            + parent_identity
            + "\x00"
            + request_call
        ).encode("utf-8")
    ).hexdigest()
    owner_identity = (
        f"{_CODEX_AUTO_REVIEW_SESSION_OWNER_IDENTITY_PREFIX}{digest}"
    )
    state = getattr(request, "state", None)
    if state is not None:
        existing_identity = get_request_effective_session_identity(request)
        if existing_identity is None:
            setattr(
                state,
                _REQUEST_STATE_EFFECTIVE_SESSION_IDENTITY_ATTR,
                owner_identity,
            )
    return owner_identity


def activate_session_owner_redispatch_effective_identity(
    *,
    request: Any,
    base_session_identity: Optional[str],
    replace_existing_auto_review_owner: bool = False,
) -> Optional[str]:
    """Set one deterministic server-side redispatch identity for this request.

    The identity is derived only from the already resolved canonical base
    identity and is never exposed through request headers or request bodies.
    An auto-review request-local owner may be replaced once; other existing
    effective identities are never derived again.
    """

    if request is None:
        return None
    state = getattr(request, "state", None)
    if state is None:
        return None
    base = _clean_optional_str(base_session_identity)
    if base is None:
        return None
    existing_identity = get_request_effective_session_identity(request)
    if existing_identity is not None and not (
        replace_existing_auto_review_owner
        and existing_identity == base
        and existing_identity.startswith(
            _CODEX_AUTO_REVIEW_SESSION_OWNER_IDENTITY_PREFIX
        )
    ):
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


def _replay_safety_json_path(path: tuple[object, ...]) -> str:
    if not path:
        return "$"
    terminal_field = path[-1] if isinstance(path[-1], str) else ""
    if terminal_field not in _REPLAY_SAFETY_TERMINAL_FIELD_NAMES:
        return "$"
    result = "$"
    for segment in path[:-1]:
        if isinstance(segment, int):
            result += f"[{segment}]"
        else:
            result += ".*"
    result += f".{terminal_field}"
    if len(result) > _REPLAY_SAFETY_MAX_FIELD_PATH_CHARS:
        return f"$.**.{terminal_field}"
    return result


def _replay_safety_failure(
    path: tuple[object, ...],
    classification: str,
) -> SessionOwnerReplaySafetyResult:
    return SessionOwnerReplaySafetyResult(
        safe=False,
        field_path=_replay_safety_json_path(path),
        classification=classification,
    )


def _has_nonempty_replay_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return bool(value)
    return bool(value)


def _is_full_client_carried_reasoning_item(value: Mapping[str, Any]) -> bool:
    item_type = value.get("type")
    item_id = value.get("id")
    if (
        not isinstance(item_type, str)
        or item_type.strip().casefold() != "reasoning"
        or not isinstance(item_id, str)
        or not item_id.strip().casefold().startswith("rs_")
        or not _REPLAY_SAFETY_FULL_REASONING_ITEM_KEYS.issubset(value)
    ):
        return False
    metadata = value.get("internal_chat_message_metadata_passthrough")
    return isinstance(metadata, Mapping) and _has_nonempty_replay_value(
        metadata.get("turn_id")
    )


def classify_session_owner_replay_safety_body(
    request_body: Optional[Mapping[str, Any]],
) -> SessionOwnerReplaySafetyResult:
    """Classify whether a request body can move under a new owner key.

    The result contains only a bounded structural path and a fixed
    classification. It never returns request values, prompts, or encrypted
    state.
    """

    if not isinstance(request_body, Mapping):
        return _replay_safety_failure((), "invalid_body_shape")

    seen: set[int] = set()

    def _find_failure(
        value: Any,
        path: tuple[object, ...],
    ) -> Optional[SessionOwnerReplaySafetyResult]:
        if isinstance(value, Mapping):
            value_id = id(value)
            if value_id in seen:
                return None
            seen.add(value_id)

            for key, child in value.items():
                normalized_key = (
                    key.strip().casefold() if isinstance(key, str) else ""
                )
                child_path = (*path, _REPLAY_SAFETY_MAPPING_PATH_SEGMENT)
                if normalized_key == "previous_response_id" and _has_nonempty_replay_value(
                    child
                ):
                    return _replay_safety_failure(
                        (*path, normalized_key),
                        "previous_response_id",
                    )
                if (
                    normalized_key in _REPLAY_SAFETY_EXPLICIT_REFERENCE_KEYS
                    and _has_nonempty_replay_value(child)
                ):
                    return _replay_safety_failure(
                        (*path, normalized_key),
                        "explicit_item_reference",
                    )
                if (
                    normalized_key == "id"
                    and isinstance(child, str)
                    and child.strip().casefold().startswith("rs_")
                ):
                    encrypted_content = value.get("encrypted_content")
                    self_contained_encrypted_item = (
                        isinstance(encrypted_content, str)
                        and bool(encrypted_content.strip())
                    )
                    if not self_contained_encrypted_item and not (
                        _is_full_client_carried_reasoning_item(value)
                    ):
                        classification = (
                            "id_only_reasoning_reference"
                            if str(value.get("type", "")).strip().casefold()
                            == "reasoning"
                            else "explicit_item_reference"
                        )
                        return _replay_safety_failure(
                            (*path, normalized_key),
                            classification,
                        )

                failure = _find_failure(child, child_path)
                if failure is not None:
                    return failure
            return None

        if isinstance(value, list):
            value_id = id(value)
            if value_id in seen:
                return None
            seen.add(value_id)
            for index, item in enumerate(value):
                failure = _find_failure(item, (*path, index))
                if failure is not None:
                    return failure
        return None

    failure = _find_failure(request_body, ())
    return failure or SessionOwnerReplaySafetyResult(safe=True)


def is_replay_safe_session_owner_redispatch_body(
    request_body: Optional[Mapping[str, Any]],
) -> bool:
    """Whether a full request body can be replayed under one new owner key."""

    return classify_session_owner_replay_safety_body(request_body).safe


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

    def _resolved(value: Any, source: str) -> Optional[str]:
        cleaned = _clean_identity_str(value)
        if cleaned is None:
            return None
        canonical = _strip_legacy_affinity_prefixes(cleaned)
        record_session_owner_continuity_receipt(
            request,
            phase="identity_resolve",
            source=source,
            session_identity=canonical,
            outcome="resolved",
        )
        return canonical

    if session_identity is not None:
        cleaned = _clean_identity_str(session_identity)
        if cleaned is None:
            return None
        return _resolved(cleaned, "explicit")

    effective_identity = get_request_effective_session_identity(request)
    if effective_identity is not None:
        return _resolved(effective_identity, "request_effective")

    review_identity = get_request_codex_auto_review_session_identity(request)
    if review_identity is not None:
        return _resolved(review_identity, "codex_auto_review")

    body = request_body if isinstance(request_body, Mapping) else {}
    _record_session_identity_intake(request, body)
    metadata = body.get("litellm_metadata")
    client_metadata = body.get("client_metadata")
    thread_keys = (
        "thread_id",
        "aawm_thread_id",
        "codex_thread_id",
        "claude_thread_id",
    )
    session_keys = (
        "session_id",
        "aawm_session_id",
        "codex_session_id",
        "claude_session_id",
        "anthropic_session_id",
    )

    def _mapping_value(mapping: Any, key: str) -> Optional[str]:
        if not isinstance(mapping, Mapping):
            return None
        return _clean_identity_str(mapping.get(key))

    # Execution-thread identity is authoritative regardless of where the
    # provider client placed it. Session identifiers are considered only after
    # all supported thread sources have been checked.
    for mapping_name, mapping in (
        ("client_metadata", client_metadata),
        ("litellm_metadata", metadata),
    ):
        for key in thread_keys:
            value = _mapping_value(mapping, key)
            if value is not None:
                return _resolved(value, f"{mapping_name}.{key}")

    headers = None
    if request is not None:
        headers = getattr(request, "headers", None)
    if headers is not None:
        try:
            items = headers.items()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            items = []
        header_map = {
            str(name).lower(): _clean_identity_str(value) for name, value in items
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
                return _resolved(value, f"header.{key}")
        # Common dash/underscore variants
        for name, value in header_map.items():
            if value and name.replace("-", "_") in {
                "thread_id",
                "x_thread_id",
                "x_aawm_thread_id",
                "x_codex_thread_id",
                "x_claude_thread_id",
            }:
                return _resolved(value, f"header.{name}")

    if isinstance(body, Mapping):
        for key in (
            "thread_id",
            "aawm_thread_id",
            "codex_thread_id",
            "claude_thread_id",
        ):
            value = _clean_identity_str(body.get(key))
            if value is not None:
                return _resolved(value, f"body.{key}")

    for mapping_name, mapping in (
        ("client_metadata", client_metadata),
        ("litellm_metadata", metadata),
    ):
        for key in session_keys:
            value = _mapping_value(mapping, key)
            if value is not None:
                return _resolved(value, f"{mapping_name}.{key}")

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
                return _resolved(value, f"header.{key}")
        # Common dash/underscore variants
        for name, value in header_map.items():
            if value and name.replace("-", "_") in {
                "session_id",
                "aawm_session_id",
                "codex_session_id",
                "claude_session_id",
            }:
                return _resolved(value, f"header.{name}")

    if isinstance(body, Mapping):
        for key in ("session_id", "aawm_session_id"):
            value = _clean_identity_str(body.get(key))
            if value is not None:
                return _resolved(value, f"body.{key}")
    record_session_owner_continuity_receipt(
        request,
        phase="identity_resolve",
        source="session_identity",
        outcome="missing",
    )
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


def derive_session_owner_effective_identity(
    base_session_identity: Optional[str],
) -> Optional[str]:
    """Derive, without activating, the first-generation redispatch identity."""

    base = _clean_optional_str(base_session_identity)
    if base is None:
        return None
    digest = hashlib.sha256(
        (
            _SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_DOMAIN_SEPARATOR + base
        ).encode("utf-8")
    ).hexdigest()
    return (
        f"{_SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_PREFIX}{digest}"
    )


def is_session_owner_redispatch_effective_identity(
    session_identity: Optional[str],
) -> bool:
    """Return whether an identity is already a first-generation derived id."""

    identity = _clean_optional_str(session_identity)
    return bool(
        identity
        and identity.startswith(_SESSION_OWNER_REDISPATCH_EFFECTIVE_IDENTITY_PREFIX)
    )


def _hosted_providers_match(
    left: Mapping[str, Any],
    right: Optional[Mapping[str, Any]] = None,
) -> bool:
    right = right or {}
    left_host = _hosted_provider_from_attributes(left)
    right_host = _hosted_provider_from_attributes(right)
    return bool(left_host) and left_host == right_host


def _same_hosted_provider_account_identity_is_mutable(
    left: Mapping[str, Any],
    right: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Account label/hash/lane are mutable only on the OpenAI OPENAI-020 contract."""

    if _accounts_are_interchangeable(left, right):
        return True
    return (
        _hosted_providers_match(left, right)
        and _hosted_provider_from_attributes(left) == "openai"
    )


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
    account_scope: Any = None,
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
        account_label = (
            source.get("xai_oauth_account_label")
            or source.get("account_label")
            or source.get("codex_oauth_account_label")
        )
    if account_hash is None:
        account_hash = (
            source.get("xai_oauth_account_hash")
            or source.get("account_hash")
            or source.get("codex_oauth_account_hash")
        )
    if account_lane is None:
        account_lane = (
            source.get("xai_oauth_lane_key")
            or source.get("account_lane")
            or source.get("codex_oauth_lane_key")
            or source.get("lane_key")
        )
    if account_scope is None:
        account_scope = source.get("xai_oauth_scope_identity") or source.get(
            "account_scope"
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
        credential_affinity = (
            source.get("xai_oauth_credential_affinity")
            or source.get("codex_oauth_credential_affinity")
        )

    attributes: Payload = {}
    values = {
        "provider": provider,
        "model": model,
        "route_family": route_family,
        "account_label": account_label,
        "account_hash": account_hash,
        "account_lane": account_lane,
        "account_scope": account_scope,
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
    route_family = str(attributes.get("route_family") or "").lower()
    if for_promotion and "xai_oauth" in route_family:
        if not _clean_optional_str(attributes.get("account_scope")):
            return (
                "session_owner: managed xAI OAuth route requires an exact "
                "record scope identity on promotion"
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
    if _same_hosted_provider_account_identity_is_mutable(left_core, right_core):
        for key in (*_MUTABLE_OPENAI_ACCOUNT_ATTRIBUTE_KEYS, "account_scope"):
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


def _strict_managed_openai_owner_comparison(
    *,
    owner_attributes: Mapping[str, Any],
    requested_attributes: Mapping[str, Any],
) -> tuple[bool, Optional[str]]:
    """Compare a managed OpenAI owner without ordinary mobility exceptions.

    The normal owner comparator intentionally permits model and account changes
    for same-hosted-provider OpenAI routes. This separate mode is used only by
    callers that have established an ordinary managed-OpenAI continuation.
    Explicit portable transitions must leave this mode disabled and continue
    through their existing transition validation.
    """

    owner = _core_owner_attributes(owner_attributes)
    requested = _core_owner_attributes(requested_attributes)
    owner_shape = _managed_direct_openai_owner_shape(owner)
    requested_shape = _managed_direct_openai_owner_shape(requested)
    if owner_shape is None and requested_shape is None:
        return False, None
    if owner_shape is None or requested_shape is None:
        return (
            True,
            "session_owner: strict managed OpenAI route contract mismatch",
        )

    for key in ("provider", "model"):
        owner_value = _clean_optional_str(owner.get(key))
        requested_value = _clean_optional_str(requested.get(key))
        if owner_value != requested_value:
            return True, f"session_owner: strict {key} mismatch"

    for key in ("account_label", "account_hash", "account_lane"):
        owner_value = _clean_optional_str(owner.get(key))
        requested_value = _clean_optional_str(requested.get(key))
        if owner_value is None or requested_value is None:
            return (
                True,
                "session_owner: strict managed OpenAI owner account identity "
                "is incomplete",
            )
        if owner_value != requested_value:
            return True, f"session_owner: strict {key} mismatch"

    if not _managed_direct_openai_owner_shapes_are_equivalent(owner, requested):
        return (
            True,
            "session_owner: strict managed OpenAI route contract mismatch",
        )
    return True, None


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
    if not _same_hosted_provider_account_identity_is_mutable(
        owner_attrs,
        requested_core,
    ):
        for key in _MUTABLE_OPENAI_ACCOUNT_ATTRIBUTE_KEYS:
            if key == "credential_affinity":
                continue
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
        "session_owner_account_scope": attrs.get("account_scope"),
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
    managed_xai_oauth = (
        str(attrs.get("provider") or "").strip().lower() == "xai"
        and "xai_oauth" in str(attrs.get("route_family") or "").lower()
    )
    interchangeable = _accounts_are_interchangeable(attrs)
    if interchangeable:
        affinity["codex_oauth_credential_affinity"] = "interchangeable"
    include_account_identity = preserve_account_identity or not interchangeable
    if managed_xai_oauth:
        for source_field, affinity_field in (
            ("account_label", "xai_oauth_account_label"),
            ("account_hash", "xai_oauth_account_hash"),
            ("account_scope", "xai_oauth_scope_identity"),
            ("account_lane", "xai_oauth_lane_key"),
        ):
            if attrs.get(source_field):
                affinity[affinity_field] = attrs.get(source_field)
        affinity["xai_oauth_credential_affinity"] = "pinned"
    elif attrs.get("account_label") and include_account_identity:
        affinity["codex_oauth_account_label"] = attrs.get("account_label")
    if not managed_xai_oauth and attrs.get("account_hash") and include_account_identity:
        affinity["codex_oauth_account_hash"] = attrs.get("account_hash")
    if not managed_xai_oauth and attrs.get("account_lane") and include_account_identity:
        affinity["codex_oauth_lane_key"] = attrs.get("account_lane")
    return {k: v for k, v in affinity.items() if v is not None}


def owner_record_as_strict_affinity_hint(
    owner_record: Optional[Mapping[str, Any]],
    *,
    preserve_account_identity: bool = False,
) -> Optional[dict[str, Any]]:
    if owner_record is None or not isinstance(owner_record, Mapping):
        return None
    if _record_state(owner_record) not in {
        SessionOwnerRecordState.OWNED.value,
        SessionOwnerRecordState.RESERVED.value,
    }:
        return None
    return owner_record_as_affinity_hint(
        owner_record,
        preserve_account_identity=preserve_account_identity,
    )


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
_REQUEST_STATE_DEFER_COMPETING_RESERVATION_LOG_ATTR = (
    "_aawm_session_owner_defer_competing_reservation_log"
)


def set_competing_reservation_log_deferred(
    request: Any,
    deferred: bool,
) -> None:
    if request is None:
        return
    state = getattr(request, "state", None)
    if state is not None:
        setattr(state, _REQUEST_STATE_DEFER_COMPETING_RESERVATION_LOG_ATTR, deferred)


def _competing_reservation_log_is_deferred(request: Any) -> bool:
    if request is None:
        return False
    state = getattr(request, "state", None)
    return bool(
        getattr(state, _REQUEST_STATE_DEFER_COMPETING_RESERVATION_LOG_ATTR, False)
        if state is not None
        else False
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
    now = time.monotonic()
    if deadline is None or deadline <= now:
        deadline = now + _normalize_reservation_wait_timeout(timeout_seconds)
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
        record_session_owner_continuity_receipt(
            request,
            phase="owner_lookup",
            source="session_identity",
            outcome="skipped_missing_identity",
        )
        return None, None, None
    cleaned = _strip_legacy_affinity_prefixes(cleaned)
    cache_key = build_aawm_alias_routing_session_owner_cache_key(
        session_identity=cleaned
    )
    redis_cache, error = await _get_redis_cache()
    if error is not None or redis_cache is None:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_lookup",
            source="durable_cache",
            session_identity=cleaned,
            cache_key=cache_key,
            outcome="error",
            reason_code="durable_cache_unavailable",
        )
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
                record_session_owner_continuity_receipt(
                    request,
                    phase="owner_lookup",
                    source="redis",
                    session_identity=cleaned,
                    cache_key=cache_key,
                    outcome="error",
                    reason_code="reservation_wait_failed",
                )
                return record, cache_key, wait_error
    except RuntimeError as exc:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_lookup",
            source="redis",
            session_identity=cleaned,
            cache_key=cache_key,
            outcome="error",
            reason_code="owner_read_failed",
        )
        return None, cache_key, str(exc)
    record_session_owner_continuity_receipt(
        request,
        phase="owner_lookup",
        source="redis",
        session_identity=cleaned,
        cache_key=cache_key,
        outcome=(
            "owned"
            if _record_state(record) == SessionOwnerRecordState.OWNED.value
            else (
                "reserved"
                if _record_state(record) == SessionOwnerRecordState.RESERVED.value
                else "missing"
            )
        ),
    )
    return record, cache_key, None


async def _cleanup_unpublished_session_owner_reservation(
    *,
    session_identity: str,
    reservation_token: str,
) -> Optional[SessionOwnerMutationResult]:
    """Release an aborted reservation without replacing its primary failure."""

    cleanup_task = asyncio.ensure_future(
        release_session_owner_reservation(
            session_identity=session_identity,
            reservation_token=reservation_token,
        )
    )
    try:
        return await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        try:
            return await asyncio.shield(cleanup_task)
        except BaseException:  # noqa: BLE001
            return None
    except BaseException:  # noqa: BLE001
        return None


async def _acquire_session_owner_reservation(
    *,
    operation: Awaitable[Any],
    session_identity: str,
    reservation_token: str,
) -> Any:
    """Settle a cancellable NX write before cleaning up a claimed token."""

    acquisition_task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(acquisition_task)
    except asyncio.CancelledError:
        claimed = False
        try:
            claimed = bool(await asyncio.shield(acquisition_task))
        except BaseException:  # noqa: BLE001
            pass
        if claimed:
            await _cleanup_unpublished_session_owner_reservation(
                session_identity=session_identity,
                reservation_token=reservation_token,
            )
        raise


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
    strict_managed_openai_owner: bool = False,
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
    # owner identity (hosted provider + endpoint/state). Model stays
    # mutable for every hosted-provider match; account identity stays
    # mutable only for the OpenAI OPENAI-020 contract.
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
                strict_applied = False
                mismatch: Optional[str] = None
                if strict_managed_openai_owner:
                    strict_applied, mismatch = (
                        _strict_managed_openai_owner_comparison(
                            owner_attributes=_owner_attributes(existing),
                            requested_attributes=attrs,
                        )
                    )
                if not strict_applied:
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
        claimed = await _acquire_session_owner_reservation(
            operation=redis_cache.async_set_cache(
                key=cache_key,
                value=reserved_record,
                ttl=ttl,
                nx=True,
                raise_on_error=True,
            ),
            session_identity=cleaned,
            reservation_token=token,
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
            await _cleanup_unpublished_session_owner_reservation(
                session_identity=cleaned,
                reservation_token=token,
            )
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
        except BaseException:  # noqa: BLE001
            await _cleanup_unpublished_session_owner_reservation(
                session_identity=cleaned,
                reservation_token=token,
            )
            raise
        if durable_record is None:
            reason = "session_owner: reserve write not visible on read-back"
            await _cleanup_unpublished_session_owner_reservation(
                session_identity=cleaned,
                reservation_token=token,
            )
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
        durable_token = _clean_optional_str(durable_record.get(_RECORD_TOKEN_FIELD))
        if durable_token != token or _record_state(durable_record) != (
            SessionOwnerRecordState.RESERVED.value
        ):
            reason = "session_owner: reserve lost race on read-back"
            await _cleanup_unpublished_session_owner_reservation(
                session_identity=cleaned,
                reservation_token=token,
            )
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
                claimed = await _acquire_session_owner_reservation(
                    operation=redis_cache.async_set_cache(
                        key=cache_key,
                        value=_build_reserved_record(
                            owner_id=resolved_owner_id,
                            attributes=attrs,
                            reservation_token=token,
                        ),
                        ttl=ttl,
                        nx=True,
                        raise_on_error=True,
                    ),
                    session_identity=cleaned,
                    reservation_token=token,
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
                        await _cleanup_unpublished_session_owner_reservation(
                            session_identity=cleaned,
                            reservation_token=token,
                        )
                        reason = str(exc)
                    except BaseException:  # noqa: BLE001
                        await _cleanup_unpublished_session_owner_reservation(
                            session_identity=cleaned,
                            reservation_token=token,
                        )
                        raise
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
                        await _cleanup_unpublished_session_owner_reservation(
                            session_identity=cleaned,
                            reservation_token=token,
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
        strict_applied = False
        mismatch: Optional[str] = None
        if strict_managed_openai_owner:
            strict_applied, mismatch = _strict_managed_openai_owner_comparison(
                owner_attributes=_owner_attributes(winner),
                requested_attributes=attrs,
            )
        if not strict_applied:
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


def _session_owner_lease_is_renewable(
    lease: Optional[SessionOwnerLease],
) -> bool:
    return (
        lease is not None
        and lease.held_reservation
        and not lease.finalizing
        and not lease.promoted
        and not lease.released
    )


async def _barrier_session_owner_lease_renewal(
    lease: SessionOwnerLease,
) -> None:
    lease.finalizing = True
    task = lease.renewal_task
    if task is None:
        return
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        current_task = None
    if task is not current_task:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if lease.renewal_task is task:
        lease.renewal_task = None


def _stop_session_owner_lease_renewal(
    lease: Optional[SessionOwnerLease],
) -> None:
    if lease is None:
        return
    task = lease.renewal_task
    if task is None or task.done():
        return
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        current_task = None
    if task is not current_task:
        task.cancel()


def _normalize_reservation_renewal_interval(
    ttl_seconds: float,
    interval_seconds: Optional[float],
) -> float:
    ttl = _normalize_reservation_ttl(ttl_seconds)
    if interval_seconds is None:
        return min(
            _DEFAULT_RESERVATION_RENEWAL_INTERVAL_SECONDS,
            max(_MIN_RESERVATION_RENEWAL_INTERVAL_SECONDS, ttl / 3.0),
        )
    try:
        interval = float(interval_seconds)
    except (TypeError, ValueError):
        interval = _DEFAULT_RESERVATION_RENEWAL_INTERVAL_SECONDS
    if not math.isfinite(interval) or interval <= 0:
        interval = _DEFAULT_RESERVATION_RENEWAL_INTERVAL_SECONDS
    return max(0.001, min(ttl / 2.0, interval))


async def _renew_session_owner_lease_once(
    lease: SessionOwnerLease,
    *,
    ttl_seconds: float,
) -> Optional[Payload]:
    """Renew one held lease without ever reacquiring a missing reservation."""

    if not _session_owner_lease_is_renewable(lease):
        return None
    session_identity = _clean_optional_str(lease.session_identity)
    token = _clean_optional_str(lease.reservation_token)
    if session_identity is None or token is None:
        return None
    cache_key = _clean_optional_str(lease.cache_key)
    if cache_key is None:
        cache_key = build_aawm_alias_routing_session_owner_cache_key(
            session_identity=session_identity
        )

    redis_cache, error = await _get_redis_cache()
    if error is not None or redis_cache is None:
        return None
    try:
        record = await _read_session_owner_record(
            redis_cache=redis_cache,
            cache_key=cache_key,
        )
        if (
            record is None
            or _record_state(record) != SessionOwnerRecordState.RESERVED.value
            or _clean_optional_str(record.get(_RECORD_TOKEN_FIELD)) != token
        ):
            return None
        return await _renew_reservation(
            redis_cache=redis_cache,
            cache_key=cache_key,
            record=record,
            ttl_seconds=ttl_seconds,
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        return None


async def _session_owner_lease_renewal_loop(
    lease: SessionOwnerLease,
    *,
    ttl_seconds: float,
    interval_seconds: float,
) -> None:
    while _session_owner_lease_is_renewable(lease):
        try:
            await asyncio.sleep(interval_seconds)
            if not _session_owner_lease_is_renewable(lease):
                return
            renewed = await _renew_session_owner_lease_once(
                lease,
                ttl_seconds=ttl_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SessionOwnerLeaseRenewalError(
                session_identity=lease.session_identity,
            ) from exc
        if renewed is None:
            if not _session_owner_lease_is_renewable(lease):
                return
            raise SessionOwnerLeaseRenewalError(
                session_identity=lease.session_identity,
            )


async def _cancel_and_await_tasks(*tasks: Any) -> None:
    active_tasks = [
        task
        for task in tasks
        if task is not None and not task.done()
    ]
    for task in active_tasks:
        task.cancel()
    if not tasks:
        return
    cleanup = asyncio.gather(*tasks, return_exceptions=True)
    cancelled = False
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        cancelled = True
        await cleanup
    if cancelled:
        raise asyncio.CancelledError


def _session_owner_renewal_task_error(
    task: Any,
    lease: SessionOwnerLease,
) -> Optional[BaseException]:
    if not task.done():
        return None
    if not _session_owner_lease_is_renewable(lease):
        if task.cancelled():
            return None
        # Consume an unexpected task exception after a nested finalizer has
        # already made the lease terminal.
        try:
            task.exception()
        except BaseException:
            pass
        return None
    if task.cancelled():
        return SessionOwnerLeaseRenewalError(
            session_identity=lease.session_identity,
        )
    return task.exception()


def start_session_owner_lease_renewal(
    lease: Optional[SessionOwnerLease],
    *,
    reservation_ttl_seconds: float = _DEFAULT_RESERVATION_TTL_SECONDS,
    renewal_interval_seconds: Optional[float] = None,
) -> Optional[Any]:
    """Start a response-owned renewer for a still-held reservation.

    ``run_with_session_owner_lease_renewal`` owns its task only for the
    provider operation. Deferred responses need the same renewal contract
    while their body iterator is being consumed, so the response lifecycle
    takes ownership of a separate task and joins it during finalization.
    """

    if not _session_owner_lease_is_renewable(lease):
        return None
    assert lease is not None

    existing_task = lease.renewal_task
    if existing_task is not None:
        if not existing_task.done():
            return existing_task
        existing_error = _session_owner_renewal_task_error(existing_task, lease)
        if existing_error is not None:
            return existing_task
        lease.renewal_task = None

    ttl = _normalize_reservation_ttl(reservation_ttl_seconds)
    interval = _normalize_reservation_renewal_interval(
        ttl,
        renewal_interval_seconds,
    )
    renewal_task = asyncio.create_task(
        _session_owner_lease_renewal_loop(
            lease,
            ttl_seconds=ttl,
            interval_seconds=interval,
        )
    )
    lease.renewal_task = renewal_task
    return renewal_task


async def run_with_session_owner_lease_renewal(
    lease: Optional[SessionOwnerLease],
    operation: Callable[[], Awaitable[_SessionOwnerLeaseOperationT]],
    *,
    reservation_ttl_seconds: float = _DEFAULT_RESERVATION_TTL_SECONDS,
    renewal_interval_seconds: Optional[float] = None,
) -> _SessionOwnerLeaseOperationT:
    """Run one provider operation while renewing its held reservation.

    A lost or replaced reservation cancels the operation and raises a
    fail-closed error. The provider and renewal tasks are always joined before
    this function returns or raises.
    """

    if not _session_owner_lease_is_renewable(lease):
        return await operation()
    assert lease is not None

    existing_task = lease.renewal_task
    if existing_task is not None and not existing_task.done():
        # A nested adapter operation is covered by the outer operation's
        # renewal task; never create competing renewers for one lease.
        return await operation()
    lease.renewal_task = None

    ttl = _normalize_reservation_ttl(reservation_ttl_seconds)
    interval = _normalize_reservation_renewal_interval(
        ttl,
        renewal_interval_seconds,
    )
    operation_task = asyncio.ensure_future(operation())
    try:
        renewal_task = asyncio.create_task(
            _session_owner_lease_renewal_loop(
                lease,
                ttl_seconds=ttl,
                interval_seconds=interval,
            )
        )
    except BaseException:
        await _cancel_and_await_tasks(operation_task)
        raise
    lease.renewal_task = renewal_task

    operation_succeeded = False
    try:
        done, _ = await asyncio.wait(
            {operation_task, renewal_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            renewal_error = (
                _session_owner_renewal_task_error(renewal_task, lease)
                if renewal_task in done
                else None
            )
            try:
                result = operation_task.result()
            except BaseException:
                # The provider's own failure/cancellation remains primary, but
                # the renewal task exception has already been consumed above.
                raise
            if renewal_error is not None:
                raise renewal_error
            operation_succeeded = True
            return result

        renewal_error = _session_owner_renewal_task_error(renewal_task, lease)
        if renewal_error is not None:
            await _cancel_and_await_tasks(operation_task)
            raise renewal_error
        result = await operation_task
        operation_succeeded = True
        return result
    finally:
        preserve_deferred_wire_renewal = (
            operation_succeeded
            and lease.wire_terminal_pending
            and lease.wire_disposition is None
            and not lease.finalizing
            and not lease.promoted
            and not lease.released
            and lease.renewal_task is renewal_task
        )
        if preserve_deferred_wire_renewal:
            # A nested pass-through may defer this lease after returning a
            # streaming response. Keep the existing renewer alive until the
            # final wire disposition releases or promotes the lease.
            if not operation_task.done():
                await _cancel_and_await_tasks(operation_task)
        else:
            try:
                await _cancel_and_await_tasks(operation_task, renewal_task)
            finally:
                if lease.renewal_task is renewal_task:
                    lease.renewal_task = None


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


async def rebind_session_owner_for_portable_failover(  # noqa: PLR0911
    *,
    session_identity: Optional[str],
    source_owner_id: Optional[str],
    source_attributes: Optional[Mapping[str, Any]],
    destination_attributes: Optional[Mapping[str, Any]],
    authorization: Optional[str],
    failover_ordinal: Any,
) -> SessionOwnerMutationResult:
    """Atomically move one durable owner across an authorized portable failover."""

    cleaned = resolve_canonical_session_identity(
        session_identity=session_identity,
    )
    if cleaned is None:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.SKIPPED,
            session_identity=None,
        )
    try:
        cache_key = build_aawm_alias_routing_session_owner_cache_key(
            session_identity=cleaned
        )
    except Exception as exc:  # noqa: BLE001
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            error=f"session_owner: portable failover cache key failed: {exc}",
        )

    def _error(reason: str, **kwargs: Any) -> SessionOwnerMutationResult:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=cleaned,
            cache_key=cache_key,
            error=reason,
            **kwargs,
        )

    if authorization != "codex_oauth_portable_account_failover":
        return _error("session_owner: portable failover authorization rejected")
    if type(failover_ordinal) is not int or failover_ordinal != 1:
        return _error("session_owner: portable failover ordinal rejected")

    source_owner = _clean_optional_str(source_owner_id)
    if source_owner is None:
        return _error("session_owner: portable failover source owner missing")
    if not isinstance(source_attributes, Mapping):
        return _error("session_owner: portable failover source attributes missing")
    if not isinstance(destination_attributes, Mapping):
        return _error(
            "session_owner: portable failover destination attributes missing"
        )

    source = _core_owner_attributes(
        build_session_owner_attributes(extra=source_attributes)
    )
    destination = _core_owner_attributes(
        build_session_owner_attributes(extra=destination_attributes)
    )
    for label, attributes in (("source", source), ("destination", destination)):
        incomplete = incomplete_owner_attribute_reason(
            attributes,
            for_promotion=True,
        )
        if incomplete is not None:
            return _error(
                f"session_owner: portable failover {label} attributes incomplete"
            )
    if not _accounts_are_interchangeable(source, destination):
        return _error(
            "session_owner: portable failover accounts are not interchangeable"
        )
    if _clean_optional_str(source.get("model")) != _clean_optional_str(
        destination.get("model")
    ):
        return _error("session_owner: portable failover model mismatch")
    route_mismatch = _compatibility_mismatch_reason(
        owner_record={
            _RECORD_STATE_FIELD: SessionOwnerRecordState.OWNED.value,
            _RECORD_OWNER_FIELD: source_owner,
            _RECORD_ATTRIBUTES_FIELD: source,
        },
        requested_attributes=destination,
        require_exact_attributes=True,
    )
    if route_mismatch is not None:
        return _error(
            "session_owner: portable failover route mismatch",
        )

    destination_owner = build_session_owner_id(attributes=destination)
    owned_record = _build_owned_record(
        owner_id=destination_owner,
        attributes=destination,
        reservation_token=None,
    )

    def _destination_record_matches(record: Optional[Mapping[str, Any]]) -> bool:
        if not isinstance(record, Mapping):
            return False
        if record.get(_RECORD_STATE_FIELD) != SessionOwnerRecordState.OWNED.value:
            return False
        if (
            _clean_optional_str(record.get(_RECORD_OWNER_FIELD))
            != destination_owner
        ):
            return False
        actual_attributes = _owner_attributes(record)
        if set(actual_attributes) != set(destination):
            return False
        return all(
            str(actual_attributes[key]) == str(value)
            for key, value in destination.items()
        )

    try:
        redis_cache, error = await _get_redis_cache()
    except Exception as exc:  # noqa: BLE001
        return _error(f"session_owner: portable failover cache failed: {exc}")
    if error is not None or redis_cache is None:
        return _error(error or "session_owner: durable cache unavailable")
    try:
        client = await _raw_redis_client(redis_cache)
        namespaced = _namespaced_key(redis_cache, cache_key)
    except Exception as exc:  # noqa: BLE001
        return _error(f"session_owner: portable failover redis unavailable: {exc}")

    lua = """
    local function attributes_equal(left, right)
      if type(left) ~= 'table' or type(right) ~= 'table' then
        return false
      end
      local left_count = 0
      for key, value in pairs(left) do
        left_count = left_count + 1
        if right[key] ~= value then
          return false
        end
      end
      local right_count = 0
      for key, _ in pairs(right) do
        right_count = right_count + 1
      end
      return left_count == right_count
    end

    local raw = redis.call('GET', KEYS[1])
    if not raw then
      return {0, 'missing'}
    end
    local ok, current = pcall(cjson.decode, raw)
    if not ok or type(current) ~= 'table' then
      return {-1, 'malformed'}
    end
    local source_ok, expected_source = pcall(cjson.decode, ARGV[2])
    local destination_ok, expected_destination = pcall(cjson.decode, ARGV[4])
    if (
      not source_ok
      or type(expected_source) ~= 'table'
      or not destination_ok
      or type(expected_destination) ~= 'table'
    ) then
      return {-1, 'malformed'}
    end
    if type(current['attributes']) ~= 'table' then
      return {-1, 'malformed'}
    end
    if (
      current['state'] == 'owned'
      and current['owner'] == ARGV[3]
      and attributes_equal(
        current['attributes'],
        expected_destination['attributes']
      )
    ) then
      return {2, cjson.encode(current)}
    end
    if current['state'] ~= 'owned' then
      return {3, cjson.encode(current)}
    end
    if (
      current['owner'] ~= ARGV[1]
      or not attributes_equal(current['attributes'], expected_source)
    ) then
      return {0, cjson.encode(current)}
    end
    local payload = expected_destination
    if current['owned_at_epoch'] ~= nil then
      payload['owned_at_epoch'] = current['owned_at_epoch']
    end
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
            source_owner,
            json.dumps(source),
            destination_owner,
            json.dumps(owned_record),
        )
    except Exception as exc:  # noqa: BLE001
        return _error(f"session_owner: portable failover rebind failed: {exc}")

    if not isinstance(result, (list, tuple)) or not result:
        return _error("session_owner: portable failover returned malformed result")
    try:
        code = int(result[0])
    except (TypeError, ValueError):
        return _error("session_owner: portable failover returned invalid result")
    raw_payload = result[1] if len(result) > 1 else None
    if isinstance(raw_payload, (bytes, bytearray)):
        raw_payload = raw_payload.decode("utf-8", errors="replace")
    payload: Optional[Payload] = None
    if isinstance(raw_payload, Mapping):
        payload = cast(Payload, dict(raw_payload))
    elif isinstance(raw_payload, str) and raw_payload not in {
        "missing",
        "malformed",
    }:
        try:
            decoded = json.loads(raw_payload)
            if isinstance(decoded, dict):
                payload = cast(Payload, decoded)
        except Exception:  # noqa: BLE001
            payload = None

    actual_owner = (
        _clean_optional_str(payload.get(_RECORD_OWNER_FIELD))
        if isinstance(payload, Mapping)
        else None
    )
    if code == 1:
        if not _destination_record_matches(payload):
            return _error(
                "session_owner: portable failover returned invalid owned record",
                owner_id=actual_owner,
                owner_record=payload,
            )
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.PROMOTED,
            session_identity=cleaned,
            cache_key=cache_key,
            owner_id=destination_owner,
            owner_record=payload,
        )
    if code == 2:
        if _destination_record_matches(payload):
            return SessionOwnerMutationResult(
                outcome=SessionOwnerMutationOutcome.ALREADY_OWNED,
                session_identity=cleaned,
                cache_key=cache_key,
                owner_id=destination_owner,
                owner_record=payload,
            )
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.CONFLICT,
            session_identity=cleaned,
            cache_key=cache_key,
            owner_id=actual_owner,
            owner_record=payload,
            error="session_owner: portable failover found different owned record",
        )
    if code == 0 and raw_payload == "missing":
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.NOT_HELD,
            session_identity=cleaned,
            cache_key=cache_key,
            error="session_owner: portable failover record missing",
        )
    if code in {0, 3}:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.CONFLICT,
            session_identity=cleaned,
            cache_key=cache_key,
            owner_id=actual_owner,
            owner_record=payload,
            error=(
                "session_owner: portable failover source owner or attributes "
                "did not match"
                if code == 0
                else "session_owner: portable failover record is not owned"
            ),
        )
    if code == -1:
        return _error(
            "session_owner: portable failover record is malformed",
            owner_id=actual_owner,
            owner_record=payload,
        )
    return _error(
        "session_owner: portable failover returned unknown result",
        owner_id=actual_owner,
        owner_record=payload,
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
    policy: SessionOwnerLeasePolicy = SessionOwnerLeasePolicy.PERSIST_ON_COMPLETED,
) -> SessionOwnerLease:
    return SessionOwnerLease(
        session_identity=guard.session_identity,
        cache_key=guard.cache_key,
        reservation_token=guard.reservation_token,
        owner_id=guard.owner_id,
        held_reservation=guard.held_reservation,
        decision=guard.decision.value,
        attributes=dict(attributes or _owner_attributes(guard.owner_record)),
        policy=_normalize_session_owner_lease_policy(policy),
    )


async def _release_session_owner_lease_on_terminal(
    lease: Optional[SessionOwnerLease],
    *,
    request: Any = None,
    force_release_only: bool = False,
    respect_wire_pending: bool = True,
) -> Optional[SessionOwnerMutationResult]:
    if lease is None:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            outcome="no_session",
        )
        return None

    release_only = force_release_only or session_owner_lease_is_release_only(lease)
    invariant = _session_owner_lease_release_invariant(
        lease,
        force_release_only=force_release_only,
    )
    if invariant is not None:
        lease.last_finalization_outcome = invariant.outcome.value
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=invariant.outcome.value,
            reason_code="mutation_failed",
        )
        return invariant
    if not lease.held_reservation:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome="not_held",
        )
        return None
    if lease.released:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=lease.last_finalization_outcome or "already_finalized",
        )
        return None
    if respect_wire_pending and lease.wire_terminal_pending:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="wire_terminal",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome="pending_wire_terminal",
            held_reservation=True,
        )
        return None

    await _barrier_session_owner_lease_renewal(lease)
    result = await release_session_owner_reservation(
        session_identity=lease.session_identity,
        reservation_token=lease.reservation_token,
    )
    if release_only and result.outcome is SessionOwnerMutationOutcome.ALREADY_OWNED:
        invariant = _session_owner_lease_invariant_result(
            lease,
            reason="release returned already_owned",
        )
        lease.last_finalization_outcome = invariant.outcome.value
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=invariant.outcome.value,
            reason_code="mutation_failed",
        )
        return invariant
    if result.outcome in {
        SessionOwnerMutationOutcome.RELEASED,
        SessionOwnerMutationOutcome.NOT_HELD,
        SessionOwnerMutationOutcome.ALREADY_OWNED,
    }:
        lease.released = True
        _stop_session_owner_lease_renewal(lease)
    lease.last_finalization_outcome = result.outcome.value
    record_session_owner_continuity_receipt(
        request,
        phase="owner_finalize",
        source="failure",
        session_identity=lease.session_identity,
        cache_key=lease.cache_key,
        outcome=result.outcome.value,
        reason_code="mutation_failed" if result.error else None,
    )
    return result


async def finalize_session_owner_lease_on_success(
    lease: Optional[SessionOwnerLease],
    *,
    request: Any = None,
    attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
) -> Optional[SessionOwnerMutationResult]:
    if lease is None:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="success", outcome="no_session"
        )
        return None
    invariant = _session_owner_lease_release_invariant(lease)
    if invariant is not None:
        lease.last_finalization_outcome = invariant.outcome.value
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="success",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=invariant.outcome.value,
            reason_code="mutation_failed",
        )
        return invariant
    if not lease.held_reservation:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="success",
            session_identity=lease.session_identity, cache_key=lease.cache_key,
            outcome="not_held",
        )
        return None
    if lease.promoted or lease.released:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="success",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=(
                "already_owned"
                if lease.promoted
                else (lease.last_finalization_outcome or "already_finalized")
            ),
        )
        return None
    if lease.wire_terminal_pending:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="success",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome="pending_wire_terminal",
        )
        return None
    if session_owner_lease_is_release_only(lease):
        return await _release_session_owner_lease_on_terminal(
            lease,
            request=request,
            respect_wire_pending=False,
        )
    await _barrier_session_owner_lease_renewal(lease)
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
        _stop_session_owner_lease_renewal(lease)
    lease.last_finalization_outcome = result.outcome.value
    record_session_owner_continuity_receipt(
        request,
        phase="owner_finalize",
        source="success",
        session_identity=lease.session_identity,
        cache_key=lease.cache_key,
        outcome=result.outcome.value,
        reason_code="mutation_failed" if result.error else None,
    )
    return result


async def finalize_session_owner_lease_on_failure(
    lease: Optional[SessionOwnerLease],
    *,
    request: Any = None,
) -> Optional[SessionOwnerMutationResult]:
    if lease is None:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="failure", outcome="no_session"
        )
        return None
    invariant = _session_owner_lease_release_invariant(lease)
    if invariant is not None:
        lease.last_finalization_outcome = invariant.outcome.value
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=invariant.outcome.value,
            reason_code="mutation_failed",
        )
        return invariant
    if not lease.held_reservation:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="failure",
            session_identity=lease.session_identity, cache_key=lease.cache_key,
            outcome="not_held",
        )
        return None
    if lease.promoted or lease.released:
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="failure",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=(
                "already_owned"
                if lease.promoted
                else (lease.last_finalization_outcome or "already_finalized")
            ),
        )
        return None
    if lease.wire_terminal_pending and lease.wire_disposition is None:
        lease.wire_disposition = "failed"
        lease.wire_terminal_pending = False
    return await _release_session_owner_lease_on_terminal(
        lease,
        request=request,
        respect_wire_pending=False,
    )


def defer_session_owner_lease_until_wire_terminal(request: Any) -> bool:
    """Keep a native OpenAI lease reserved until final wire disposition."""

    lease = get_request_session_owner_lease(request)
    if lease is None or not lease.held_reservation or lease.promoted or lease.released:
        return False
    lease.wire_terminal_pending = True
    lease.wire_disposition = None
    try:
        start_session_owner_lease_renewal(lease)
    except RuntimeError:
        # The lease can be marked pending before a response lifecycle owns an
        # active loop; the first async lifecycle boundary will renew or release.
        return True
    return True


async def finalize_session_owner_lease_on_wire_disposition(
    request: Any,
    *,
    disposition: str,
    attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
) -> Optional[SessionOwnerMutationResult]:
    """Finalize one deferred lease from the immutable final-wire decision."""

    lease = get_request_session_owner_lease(request)
    if lease is None:
        return None
    invariant = _session_owner_lease_release_invariant(lease)
    if invariant is not None:
        lease.last_finalization_outcome = invariant.outcome.value
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="wire_terminal",
            session_identity=lease.session_identity,
            cache_key=lease.cache_key,
            outcome=invariant.outcome.value,
            reason_code="mutation_failed",
        )
        return invariant
    if not lease.held_reservation or lease.released or lease.promoted:
        return None
    if lease.wire_disposition is not None:
        return None
    normalized = str(disposition or "").strip().lower()
    lease.wire_disposition = normalized or "failed"
    lease.wire_terminal_pending = False
    if normalized == "completed":
        return await finalize_session_owner_lease_on_success(
            lease,
            request=request,
            attributes=attributes,
            candidate=candidate,
        )
    return await finalize_session_owner_lease_on_failure(lease, request=request)


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
    if active is None:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="request_lease", outcome="no_session"
        )
        return None
    invariant = _session_owner_lease_release_invariant(active)
    if invariant is not None:
        active.last_finalization_outcome = invariant.outcome.value
        record_session_owner_continuity_receipt(
            request,
            phase="owner_finalize",
            source="request_lease",
            session_identity=active.session_identity,
            cache_key=active.cache_key,
            outcome=invariant.outcome.value,
            reason_code="mutation_failed",
        )
        if raise_on_promote_failure:
            raise_session_owner_redispatch_required(
                session_identity=active.session_identity,
                mutation=invariant,
                failure_phase=failure_phase,
                request=request,
            )
        return invariant
    if not active.held_reservation:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="request_lease",
            session_identity=active.session_identity, cache_key=active.cache_key,
            outcome="not_held",
        )
        return None
    if active.promoted or active.released:
        record_session_owner_continuity_receipt(
            request, phase="owner_finalize", source="request_lease",
            session_identity=active.session_identity, cache_key=active.cache_key,
            outcome=(
                "already_owned"
                if active.promoted
                else (active.last_finalization_outcome or "already_finalized")
            ),
        )
        return None
    if exc is not None:
        result = await finalize_session_owner_lease_on_failure(active, request=request)
        return result

    status = getattr(response, "status_code", None)
    ok = response is not None and (
        status is None or (isinstance(status, int) and status < 300)
    )
    if not ok:
        result = await finalize_session_owner_lease_on_failure(active, request=request)
        return result

    result = await finalize_session_owner_lease_on_success(
        active,
        request=request,
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


def bind_deferred_session_owner_lease_to_streaming_response(  # noqa: PLR0915
    response: Any,
    *,
    request: Any,
    lease: Optional[SessionOwnerLease],
    attributes: Optional[Mapping[str, Any]] = None,
    candidate: Optional[Mapping[str, Any]] = None,
    failure_phase: str = "session_owner_stream_promote",
    success_finalizer: Optional[
        Callable[[], Awaitable[Optional[SessionOwnerMutationResult]]]
    ] = None,
    success_outcomes: Optional[set[SessionOwnerMutationOutcome]] = None,
    on_success: Optional[Callable[[], Awaitable[None]]] = None,
    on_failure: Optional[Callable[[Optional[BaseException]], Awaitable[None]]] = None,
) -> bool:
    """Bind ownership to the complete, validated response stream.

    The provider operation ends when a ``StreamingResponse`` is constructed,
    but ownership remains pending until the response validator has observed a
    terminal valid event and the client-side iterator reaches EOF. The
    response therefore owns renewal, finalization, and iterator cleanup.
    """

    from fastapi.responses import StreamingResponse
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_responses_wire import (
        OpenAIResponsesStreamingResponse,
        OpenAIResponsesWireDisposition,
    )

    observe = _make_xai_deferred_stream_observer(
        request,
        lease,
        response,
        success_finalizer,
    )
    if not isinstance(response, StreamingResponse):
        observe("binding", binding_outcome="not_streaming_response")
        return False
    if getattr(
        response,
        "_aawm_session_owner_deferred_finalizer_bound",
        False,
    ):
        observe("binding", binding_outcome="already_bound")
        return True
    original_iterator = getattr(response, "body_iterator", None)
    if original_iterator is None:
        observe("binding", binding_outcome="missing_iterator")
        return False

    def _is_terminal_wire_owner_context(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        provider = str(
            value.get("provider")
            or value.get("custom_llm_provider")
            or value.get("hosted_provider")
            or ""
        ).strip().casefold()
        route_family = str(
            value.get("route_family") or value.get("endpoint_contract") or ""
        ).strip().casefold()
        return (
            provider == "xai"
            and route_family in _XAI_DEFERRED_STREAM_ROUTE_FAMILIES
        ) or (provider == "muse_code" and route_family == "muse_code")

    request_state = getattr(request, "state", None)
    candidate_context = getattr(
        request_state,
        "aawm_openai_candidate_context",
        None,
    )
    wire_trace = getattr(response, "wire_trace", None)
    terminal_wire_path = (
        isinstance(response, OpenAIResponsesStreamingResponse)
        and wire_trace is not None
        and (
            _is_terminal_wire_owner_context(getattr(lease, "attributes", None))
            or _is_terminal_wire_owner_context(candidate_context)
        )
    )

    renewal_task = start_session_owner_lease_renewal(lease)

    async def _promote() -> Optional[SessionOwnerMutationResult]:
        return await finalize_session_owner_lease_on_success(
            lease,
            request=request,
            attributes=attributes,
            candidate=candidate,
        )

    if success_finalizer is None:
        success_finalizer = _promote
    if success_outcomes is None:
        success_outcomes = session_owner_lease_success_outcomes(lease)

    def _validation_status() -> tuple[bool, str]:
        state = getattr(response, "_aawm_responses_validation_state", None)
        if isinstance(state, Mapping):
            complete = state.get("complete") is True
            valid = state.get("valid") is True
            terminal_seen = state.get("terminal_seen") is True
            terminal_status = state.get("terminal_status")
            if (
                complete
                and valid
                and terminal_seen
                and terminal_status == "completed"
            ):
                return True, ""
            reason = state.get("reason")
            if complete and valid and terminal_seen:
                reason = "response validation terminal status was not completed"
            return (
                False,
                str(reason)
                if reason is not None
                else "response validation did not reach a valid terminal event",
            )
        return False, "response validation state was not available"

    def _mutation_error(reason: str) -> SessionOwnerMutationResult:
        return SessionOwnerMutationResult(
            outcome=SessionOwnerMutationOutcome.ERROR,
            session_identity=(
                lease.session_identity if lease is not None else None
            ),
            cache_key=lease.cache_key if lease is not None else None,
            reservation_token=(
                lease.reservation_token if lease is not None else None
            ),
            error=reason,
        )

    def _raise_structured_failure(
        *,
        mutation: SessionOwnerMutationResult,
        phase: str,
    ) -> None:
        raise_session_owner_redispatch_required(
            session_identity=(
                lease.session_identity if lease is not None else None
            ),
            mutation=mutation,
            candidate=candidate,
            failure_phase=phase,
            attempted_provider_call=True,
            request=request,
        )

    def _renewal_error() -> Optional[BaseException]:
        if renewal_task is None or lease is None:
            return None
        return _session_owner_renewal_task_error(renewal_task, lease)

    finalization_task: Optional[Any] = None
    terminal_delivered = bool(
        terminal_wire_path
        and getattr(wire_trace, "terminal_wire_committed", False)
    )

    def _wire_terminal_success_delivered() -> bool:
        if not terminal_wire_path:
            return True
        return (
            bool(getattr(wire_trace, "terminal_wire_committed", False))
            and getattr(wire_trace, "disposition", None)
            is OpenAIResponsesWireDisposition.COMPLETED
            and _validation_status()[0]
        )

    async def _notify_failure(cause: Optional[BaseException]) -> None:
        if on_failure is not None:
            await on_failure(cause)

    async def _attempt_failure_cleanup(
        cause: Optional[BaseException],
    ) -> tuple[
        Optional[SessionOwnerMutationResult],
        Optional[BaseException],
    ]:
        """Release a failed deferred lease without losing the failure callback."""
        cleanup_task = asyncio.ensure_future(
            finalize_session_owner_lease_on_failure(lease, request=request)
        )
        release_result: Optional[SessionOwnerMutationResult] = None
        cleanup_error: Optional[BaseException] = None
        try:
            release_result = await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            try:
                release_result = await asyncio.shield(cleanup_task)
            except BaseException as exc:  # noqa: BLE001
                cleanup_error = exc
        except BaseException as exc:  # noqa: BLE001
            cleanup_error = exc
        observe(
            "release_result",
            result=release_result,
            requested_success=True,
            iterator=wrapped_iterator,
            finalization_task=finalization_task,
        )
        try:
            await _notify_failure(cause)
        except BaseException as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc
        observe(
            "cleanup_outcome",
            result=release_result,
            error=cleanup_error,
            requested_success=True,
            iterator=wrapped_iterator,
            finalization_task=finalization_task,
        )
        return release_result, cleanup_error

    async def _run_finalization(
        success: bool,
        cause: Optional[BaseException],
    ) -> None:
        if not success:
            release_result = await finalize_session_owner_lease_on_failure(
                lease, request=request
            )
            observe(
                "release_result",
                result=release_result,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            await _notify_failure(cause)
            if (
                cause is None
                and release_result is not None
                and release_result.outcome is SessionOwnerMutationOutcome.ERROR
            ):
                _raise_structured_failure(
                    mutation=release_result,
                    phase=f"{failure_phase}_release",
                )
            return

        renewal_error = _renewal_error()
        if renewal_error is not None:
            observe(
                "renewal_failed",
                renewal_error=renewal_error,
                site="finalization",
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            release_result = await finalize_session_owner_lease_on_failure(
                lease, request=request
            )
            observe(
                "release_result",
                result=release_result,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            await _notify_failure(renewal_error)
            _raise_structured_failure(
                mutation=(
                    release_result
                    if release_result is not None
                    and release_result.outcome
                    is SessionOwnerMutationOutcome.ERROR
                    else _mutation_error(str(renewal_error))
                ),
                phase="session_owner_reservation_renewal",
            )

        validation_ok, validation_reason = _validation_status()
        observe(
            "validator_decision",
            validation_ok=validation_ok,
            requested_success=success,
            iterator=wrapped_iterator,
            finalization_task=finalization_task,
        )
        if not validation_ok:
            release_result = await finalize_session_owner_lease_on_failure(
                lease, request=request
            )
            observe(
                "release_result",
                result=release_result,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            validation_error = RuntimeError(
                f"session_owner: deferred response validation failed: "
                f"{validation_reason}"
            )
            await _notify_failure(validation_error)
            _raise_structured_failure(
                mutation=_mutation_error(str(validation_error)),
                phase=f"{failure_phase}_validation",
            )

        try:
            observe(
                "finalizer_enter",
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            result = await success_finalizer()
            observe(
                "finalizer_result",
                result=result,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
        except BaseException as finalization_error:  # noqa: BLE001
            observe(
                "finalizer_result",
                error=finalization_error,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            release_result, cleanup_error = await _attempt_failure_cleanup(
                finalization_error
            )
            if (
                release_result is not None
                and release_result.outcome is SessionOwnerMutationOutcome.ERROR
            ):
                _raise_structured_failure(
                    mutation=release_result,
                    phase=f"{failure_phase}_release",
                )
            if cleanup_error is not None:
                _raise_structured_failure(
                    mutation=_mutation_error(str(cleanup_error)),
                    phase=f"{failure_phase}_release",
                )
            raise
        if (
            result is not None
            and result.outcome not in success_outcomes
        ):
            release_result = await finalize_session_owner_lease_on_failure(
                lease, request=request
            )
            observe(
                "release_result",
                result=release_result,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
            finalization_error = RuntimeError(
                "session_owner: deferred lease finalization did not commit "
                f"outcome={result.outcome.value}"
            )
            await _notify_failure(finalization_error)
            _raise_structured_failure(
                mutation=result,
                phase=failure_phase,
            )
        if on_success is not None:
            await on_success()

    def _select_finalization_task(
        success: bool,
        cause: Optional[BaseException],
        *,
        basis: str,
    ) -> Any:
        nonlocal finalization_task
        if finalization_task is None:
            finalization_task = asyncio.create_task(
                _run_finalization(success, cause)
            )
            observe(
                "finalization_task_created",
                requested_success=success,
                finalization_basis=basis,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
        else:
            observe(
                "finalization_task_reused",
                requested_success=success,
                finalization_basis=basis,
                iterator=wrapped_iterator,
                finalization_task=finalization_task,
            )
        return finalization_task

    async def _finalize(
        success: bool,
        cause: Optional[BaseException] = None,
    ) -> None:
        observe(
            "finalize_enter",
            error=cause,
            requested_success=success,
            iterator=wrapped_iterator,
            finalization_task=finalization_task,
        )
        task = _select_finalization_task(
            success,
            cause,
            basis="iterator_eof" if success else "failure",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as wait_error:
            observe(
                "finalization_wait_cancelled",
                error=wait_error,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=task,
            )
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as second_wait_error:
                observe(
                    "finalization_wait_cancelled",
                    error=second_wait_error,
                    requested_success=success,
                    iterator=wrapped_iterator,
                    finalization_task=task,
                )
                raise
            except BaseException as finalization_error:  # noqa: BLE001
                observe(
                    "finalization_task_raised",
                    error=finalization_error,
                    requested_success=success,
                    iterator=wrapped_iterator,
                    finalization_task=task,
                )
                raise
            observe(
                "finalization_task_returned",
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=task,
            )
            raise
        except BaseException as finalization_error:  # noqa: BLE001
            observe(
                "finalization_task_raised",
                error=finalization_error,
                requested_success=success,
                iterator=wrapped_iterator,
                finalization_task=task,
            )
            raise
        observe(
            "finalization_task_returned",
            requested_success=success,
            iterator=wrapped_iterator,
            finalization_task=task,
        )

    original_closed = False

    async def _close_original_iterator() -> None:
        nonlocal original_closed
        if original_closed:
            return
        original_closed = True
        close = getattr(original_iterator, "aclose", None)
        if callable(close):
            try:
                await close()
            except BaseException:
                verbose_proxy_logger.debug(
                    "Failed to close deferred session-owner stream iterator",
                    exc_info=True,
                )
        cleanup = getattr(
            response,
            "_aawm_responses_validation_cleanup",
            None,
        )
        if callable(cleanup):
            try:
                await cleanup()
            except BaseException:
                verbose_proxy_logger.debug(
                    "Failed to close deferred Responses validation resources",
                    exc_info=True,
                )

    class _DeferredLeaseIterator:
        def __init__(self) -> None:
            self._iterator = original_iterator.__aiter__()
            self._closed = False
            self._completed = False
            self._first_pull_observed = False

        def __aiter__(self) -> "_DeferredLeaseIterator":
            return self

        async def __anext__(self) -> Any:
            if self._closed:
                raise StopAsyncIteration
            if not self._first_pull_observed:
                self._first_pull_observed = True
                observe(
                    "first_pull",
                    iterator=self,
                    finalization_task=finalization_task,
                )
            renewal_error = _renewal_error()
            if renewal_error is not None:
                observe(
                    "renewal_failed",
                    renewal_error=renewal_error,
                    site="iterator_pre_pull",
                    iterator=self,
                    finalization_task=finalization_task,
                )
                await _finalize(False, renewal_error)
                await _close_original_iterator()
                _raise_structured_failure(
                    mutation=_mutation_error(str(renewal_error)),
                    phase="session_owner_reservation_renewal",
                )
            next_task = asyncio.ensure_future(self._iterator.__anext__())
            try:
                if renewal_task is None:
                    chunk = await next_task
                else:
                    done, _ = await asyncio.wait(
                        {next_task, renewal_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if renewal_task in done:
                        renewal_error = _renewal_error()
                        if renewal_error is not None:
                            observe(
                                "renewal_failed",
                                renewal_error=renewal_error,
                                site="iterator_wait",
                                iterator=self,
                                finalization_task=finalization_task,
                            )
                            await _cancel_and_await_tasks(next_task)
                            await _finalize(False, renewal_error)
                            await _close_original_iterator()
                            self._closed = True
                            _raise_structured_failure(
                                mutation=_mutation_error(str(renewal_error)),
                                phase="session_owner_reservation_renewal",
                            )
                    chunk = await next_task
            except StopAsyncIteration:
                self._completed = True
                observe(
                    "iterator_eof",
                    iterator=self,
                    finalization_task=finalization_task,
                )
                eof_success = (
                    _wire_terminal_success_delivered()
                    if terminal_wire_path
                    else True
                )
                try:
                    await _finalize(eof_success)
                finally:
                    self._closed = True
                    await _close_original_iterator()
                raise
            except BaseException as exc:
                observe(
                    (
                        "iterator_cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else (
                            "iterator_closed"
                            if isinstance(exc, GeneratorExit)
                            else "iterator_exception"
                        )
                    ),
                    error=exc,
                    iterator=self,
                    finalization_task=finalization_task,
                )
                try:
                    await _cancel_and_await_tasks(next_task)
                    await _finalize(False, exc)
                finally:
                    self._closed = True
                    await _close_original_iterator()
                raise
            renewal_error = _renewal_error()
            if renewal_error is not None:
                observe(
                    "renewal_failed",
                    renewal_error=renewal_error,
                    site="iterator_post_pull",
                    iterator=self,
                    finalization_task=finalization_task,
                )
                try:
                    await _finalize(False, renewal_error)
                finally:
                    self._closed = True
                    await _close_original_iterator()
                _raise_structured_failure(
                    mutation=_mutation_error(str(renewal_error)),
                    phase="session_owner_reservation_renewal",
                )
            return chunk

        async def aclose(self) -> None:
            if self._closed:
                return
            if not self._completed:
                observe(
                    "close_before_eof",
                    iterator=self,
                    finalization_task=finalization_task,
                )
            self._closed = True
            try:
                if not self._completed:
                    await _finalize(False)
            finally:
                await _close_original_iterator()

    wrapped_iterator = _DeferredLeaseIterator()
    response.body_iterator = wrapped_iterator
    setattr(response, "_aawm_session_owner_deferred_finalizer_bound", True)

    original_stream_response = getattr(response, "stream_response", None)
    stream_response_wrapped = False
    if callable(original_stream_response):

        async def _stream_response_with_finalizer(send: Any) -> None:
            async def _send_with_owner_terminal(message: Any) -> None:
                nonlocal terminal_delivered
                terminal_was_committed = bool(
                    getattr(wire_trace, "terminal_wire_committed", False)
                )
                await send(message)
                terminal_is_committed = bool(
                    getattr(wire_trace, "terminal_wire_committed", False)
                )
                if (
                    terminal_wire_path
                    and not terminal_delivered
                    and not terminal_was_committed
                    and terminal_is_committed
                ):
                    terminal_delivered = True
                    validation_ok, _ = _validation_status()
                    wire_disposition = getattr(wire_trace, "disposition", None)
                    observe(
                        "terminal_delivered",
                        terminal_delivered=True,
                        wire_disposition=wire_disposition,
                        finalization_basis="terminal_delivery",
                        validation_ok=validation_ok,
                        iterator=wrapped_iterator,
                        finalization_task=finalization_task,
                    )
                    if (
                        wire_disposition
                        is OpenAIResponsesWireDisposition.COMPLETED
                        and validation_ok
                    ):
                        _select_finalization_task(
                            True,
                            None,
                            basis="terminal_delivery",
                        )

            try:
                renewal_error = _renewal_error()
                if renewal_error is not None:
                    observe(
                        "renewal_failed",
                        renewal_error=renewal_error,
                        site="stream_response",
                        iterator=wrapped_iterator,
                        finalization_task=finalization_task,
                    )
                    await _finalize(False, renewal_error)
                    _raise_structured_failure(
                        mutation=_mutation_error(str(renewal_error)),
                        phase="session_owner_reservation_renewal",
                    )
                await original_stream_response(
                    _send_with_owner_terminal if terminal_wire_path else send
                )
            except BaseException as exc:
                observe(
                    (
                        "stream_response_cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "stream_response_exception"
                    ),
                    error=exc,
                    iterator=wrapped_iterator,
                    finalization_task=finalization_task,
                )
                await _finalize(False, exc)
                raise
            finally:
                await wrapped_iterator.aclose()

        response.stream_response = _stream_response_with_finalizer
        stream_response_wrapped = True
    observe(
        "binding",
        binding_outcome="bound",
        iterator_wrapped=True,
        stream_response_wrapped=stream_response_wrapped,
        iterator=wrapped_iterator,
        finalization_task=finalization_task,
    )
    return True


def _bound_xai_oauth_account_identity(request: Any) -> Payload:
    """Return a request-bound managed xAI account identity when available."""

    if request is None:
        return {}
    try:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.xai_oauth import (
            get_bound_xai_oauth_selected_account,
            xai_oauth_selected_account_metadata,
        )

        selected_account = get_bound_xai_oauth_selected_account(request)
    except Exception:  # noqa: BLE001
        return {}
    if selected_account is None:
        return {}
    selected_metadata = xai_oauth_selected_account_metadata(selected_account)
    return {
        "account_label": selected_metadata["xai_oauth_account_label"],
        "account_hash": selected_metadata["xai_oauth_account_hash"],
        "account_lane": selected_metadata["xai_oauth_lane_key"],
        "account_scope": selected_metadata["xai_oauth_scope_identity"],
    }


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

    bound_xai_identity = _bound_xai_oauth_account_identity(request)
    if bound_xai_identity:
        return bound_xai_identity

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


_XAI_DEFERRED_STREAM_EVENT = "session_owner_deferred_stream"
_XAI_DEFERRED_STREAM_PHASES = frozenset(
    {
        "binding",
        "first_pull",
        "iterator_eof",
        "iterator_cancelled",
        "iterator_closed",
        "iterator_exception",
        "close_before_eof",
        "renewal_failed",
        "stream_response_cancelled",
        "stream_response_exception",
        "terminal_delivered",
        "finalization_wait_cancelled",
        "finalization_task_returned",
        "finalization_task_raised",
        "release_result",
        "cleanup_outcome",
        "validator_decision",
        "finalizer_enter",
        "finalizer_result",
        "finalize_enter",
        "finalization_task_created",
        "finalization_task_reused",
    }
)
_XAI_DEFERRED_STREAM_BINDING_OUTCOMES = frozenset(
    {"not_streaming_response", "already_bound", "missing_iterator", "bound"}
)
_XAI_DEFERRED_STREAM_SITES = frozenset(
    {
        "finalization",
        "iterator_pre_pull",
        "iterator_wait",
        "iterator_post_pull",
        "stream_response",
    }
)
_XAI_DEFERRED_STREAM_ROUTE_FAMILIES = frozenset(
    {
        XAI_OAUTH_ROUTE_FAMILY,
        GROK_NATIVE_OAUTH_ROUTE_FAMILY,
        "codex_xai_oauth_responses_adapter",
        "codex_grok_native_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
        "codex_auto_agent_xai_oauth_responses",
        "codex_auto_agent_grok_native_responses",
    }
)
_XAI_DEFERRED_STREAM_REQUESTED_SUCCESS_PHASES = frozenset(
    {
        "finalizer_enter",
        "finalizer_result",
        "finalize_enter",
        "finalization_task_created",
        "finalization_task_reused",
        "finalization_wait_cancelled",
        "finalization_task_returned",
        "finalization_task_raised",
    }
)
_XAI_DEFERRED_STREAM_MUTATION_OUTCOMES = frozenset(
    outcome.value for outcome in SessionOwnerMutationOutcome
) | {"none"}
_XAI_DEFERRED_STREAM_DECISIONS = frozenset(
    decision.value for decision in SessionOwnerGuardDecision
)
_XAI_DEFERRED_STREAM_OWNER_STATES = frozenset({"reserved", "owned"})
_XAI_DEFERRED_STREAM_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "failed",
        "cancelled",
        "incomplete",
        "in_progress",
        "requires_action",
    }
)
_XAI_DEFERRED_STREAM_WIRE_DISPOSITIONS = frozenset(
    {"completed", "failed", "incomplete", "cancelled", "disconnected", "error"}
)
_XAI_DEFERRED_STREAM_FINALIZATION_BASES = frozenset(
    {"terminal_delivery", "iterator_eof", "failure"}
)


def _make_xai_deferred_stream_observer(
    request: Any,
    lease: Optional[SessionOwnerLease],
    response: Any,
    success_finalizer: Optional[Callable[..., Any]],
) -> Callable[..., None]:
    """Build a bounded, synchronous observer for the selected xAI stream.

    This is intentionally observational. It captures request/lease
    correlation at binding time and reads mutable stream state only inside the
    guarded emitter so diagnostic failures cannot affect ownership lifecycle.
    """

    def _noop_observe(_phase: Any, **_fields: Any) -> None:
        return None

    try:
        missing = object()

        def _field(value: Any, key: str, default: Any = missing) -> Any:
            if isinstance(value, Mapping):
                return value.get(key, default)
            try:
                return getattr(value, key, default)
            except BaseException:  # noqa: BLE001
                return default

        def _clean_correlation(value: Any) -> Optional[str]:
            return value.strip() if isinstance(value, str) and value.strip() else None

        def _is_xai_context(value: Any) -> bool:
            if not isinstance(value, Mapping):
                return False
            provider = _clean_correlation(
                value.get("provider") or value.get("custom_llm_provider")
            )
            hosted_provider = _clean_correlation(value.get("hosted_provider"))
            route_family = _clean_correlation(
                value.get("route_family") or value.get("endpoint_contract")
            )
            provider_value = provider.casefold() if provider is not None else ""
            hosted_value = (
                hosted_provider.casefold() if hosted_provider is not None else ""
            )
            route_value = route_family.casefold() if route_family is not None else ""
            return (
                provider_value == "xai"
                or hosted_value == "xai"
                or route_value in _XAI_DEFERRED_STREAM_ROUTE_FAMILIES
            )

        state = _field(request, "state", None)
        lease_attributes = _field(lease, "attributes", None)
        candidate_context = _field(state, "aawm_openai_candidate_context", None)
        if not (
            _is_xai_context(lease_attributes)
            or _is_xai_context(candidate_context)
        ):
            return _noop_observe

        request_context = _field(
            state, _SESSION_OWNER_REQUEST_CONTEXT_STATE_KEY, None
        )
        request_call_id = _clean_correlation(
            _field(state, _SESSION_OWNER_REQUEST_CALL_ID_STATE_KEY, None)
        )
        if request_call_id is None:
            request_call_id = _clean_correlation(
                _field(request_context, "litellm_call_id", None)
            )
        trace_id = _clean_correlation(_field(request_context, "trace_id", None))

        attempt_id = None
        for key in (
            "aawm_alias_request_attempt_id",
            "aawm_alias_attempt_id",
            "attempt_id",
            "attempt_identity",
        ):
            attempt_id = _clean_correlation(_field(state, key, None))
            if attempt_id is not None:
                break
        if attempt_id is None:
            for key in ("attempt_id", "attempt_identity", "provider_attempt_id"):
                attempt_id = _clean_correlation(_field(request_context, key, None))
                if attempt_id is not None:
                    break

        dispatch_id = None
        agent_dispatch = _field(request_context, "agent_dispatch", None)
        if isinstance(agent_dispatch, Mapping):
            dispatch_id = _clean_correlation(agent_dispatch.get("dispatch_id"))

        correlation: dict[str, Any] = {}
        for key, value in (
            ("litellm_call_id", request_call_id),
            ("trace_id", trace_id),
            ("attempt_id", attempt_id),
            ("dispatch_id", dispatch_id),
        ):
            if value is not None:
                hashed = _hash_session_owner_log_identifier(value)
                if hashed is not None:
                    correlation[key] = hashed

        session_identity = _field(lease, "session_identity", None)
        cache_key = _field(lease, "cache_key", None)
        session_hash = _hash_session_owner_log_identifier(session_identity)
        owner_key_hash = _hash_session_owner_log_identifier(cache_key)
        if session_hash is not None:
            correlation["canonical_session_identity_hash"] = session_hash
        if owner_key_hash is not None:
            correlation["owner_key_hash"] = owner_key_hash

        success_finalizer_source = (
            "default" if success_finalizer is None else "supplied"
        )

        def _normalize_status(
            value: Any,
            allowed: frozenset[str],
            *,
            missing_value: str = "unknown",
        ) -> str:
            raw = value.value if isinstance(value, Enum) else value
            if not isinstance(raw, str) or not raw.strip():
                return missing_value
            normalized = raw.strip().casefold()
            return normalized if normalized in allowed else "other"

        def _normalize_phase(value: Any) -> str:
            raw = value.value if isinstance(value, Enum) else value
            if not isinstance(raw, str) or not raw.strip():
                return "unknown"
            normalized = raw.strip().casefold()
            return (
                normalized
                if normalized in _XAI_DEFERRED_STREAM_PHASES
                else "unknown"
            )

        def _normalize_finalization_basis(value: Any) -> str:
            raw = value.value if isinstance(value, Enum) else value
            if not isinstance(raw, str) or not raw.strip():
                return "unknown"
            normalized = raw.strip().casefold()
            return (
                normalized
                if normalized in _XAI_DEFERRED_STREAM_FINALIZATION_BASES
                else "unknown"
            )

        def _optional_bool(value: Any, *, absent: Any = "unknown") -> Any:
            return value if isinstance(value, bool) else absent

        def _iterator_snapshot(iterator: Any) -> dict[str, Any]:
            if iterator is missing:
                return {
                    "iterator_completed": "unknown",
                    "iterator_closed": "unknown",
                }
            if iterator is None:
                return {
                    "iterator_completed": "unknown",
                    "iterator_closed": "unknown",
                }
            completed = _field(iterator, "_completed", missing)
            if completed is missing:
                completed = _field(iterator, "completed", missing)
            closed = _field(iterator, "_closed", missing)
            if closed is missing:
                closed = _field(iterator, "closed", missing)
            return {
                "iterator_completed": (
                    _optional_bool(completed)
                    if completed is not missing
                    else "unknown"
                ),
                "iterator_closed": (
                    _optional_bool(closed) if closed is not missing else "unknown"
                ),
            }

        def _finalization_task_snapshot(task: Any) -> dict[str, Any]:
            if task is missing:
                return {"finalization_task_present": "unknown"}
            return {
                "finalization_task_present": task is not None,
            }

        def _lease_snapshot() -> dict[str, Any]:
            present = lease is not None
            if not present:
                return {
                    "lease_present": False,
                    "lease_decision": "unknown",
                    "held_reservation": False,
                    "released": False,
                    "promoted": False,
                    "wire_terminal_pending": False,
                    "wire_disposition": "unknown",
                    "renewal_task_present": False,
                }
            held = _field(lease, "held_reservation", missing)
            released = _field(lease, "released", missing)
            promoted = _field(lease, "promoted", missing)
            pending = _field(lease, "wire_terminal_pending", missing)
            renewal_task = _field(lease, "renewal_task", missing)
            return {
                "lease_present": True,
                "lease_decision": _normalize_status(
                    _field(lease, "decision", missing),
                    _XAI_DEFERRED_STREAM_DECISIONS,
                ),
                "held_reservation": (
                    _optional_bool(held) if held is not missing else "unknown"
                ),
                "released": (
                    _optional_bool(released) if released is not missing else "unknown"
                ),
                "promoted": (
                    _optional_bool(promoted) if promoted is not missing else "unknown"
                ),
                "wire_terminal_pending": (
                    _optional_bool(pending) if pending is not missing else "unknown"
                ),
                "wire_disposition": _normalize_status(
                    _field(lease, "wire_disposition", missing),
                    _XAI_DEFERRED_STREAM_WIRE_DISPOSITIONS,
                ),
                "renewal_task_present": (
                    renewal_task is not missing and renewal_task is not None
                ),
            }

        def _validation_snapshot() -> dict[str, Any]:
            state_value = _field(
                response, "_aawm_responses_validation_state", None
            )
            if not isinstance(state_value, Mapping):
                return {
                    "validation_state_present": False,
                    "complete": "unknown",
                    "valid": "unknown",
                    "terminal_seen": "unknown",
                    "terminal_status": "unknown",
                }
            complete = state_value.get("complete")
            valid = state_value.get("valid")
            terminal_seen = state_value.get("terminal_seen")
            terminal_status = _normalize_status(
                state_value.get("terminal_status"),
                _XAI_DEFERRED_STREAM_TERMINAL_STATUSES,
            )
            return {
                "validation_state_present": True,
                "complete": (
                    complete if isinstance(complete, bool) else "unknown"
                ),
                "valid": valid if isinstance(valid, bool) else "unknown",
                "terminal_seen": (
                    terminal_seen
                    if isinstance(terminal_seen, bool)
                    else "unknown"
                ),
                "terminal_status": terminal_status,
            }

        def _exception_category(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, asyncio.CancelledError):
                return "cancelled"
            if isinstance(value, (TimeoutError, asyncio.TimeoutError)):
                return "timeout"
            if isinstance(value, SessionOwnerLeaseRenewalError):
                return "renewal"
            if isinstance(value, ConnectionError):
                return "connection"
            if isinstance(value, HTTPException):
                return "http"
            if isinstance(value, (ValueError, TypeError, KeyError, AttributeError)):
                return "programming"
            if isinstance(value, RuntimeError):
                return "runtime"
            return "other"

        def _mutation_snapshot(result: Any, phase: str) -> dict[str, Any]:
            snapshot: dict[str, Any] = {
                "result_present": result is not None,
                "mutation_outcome": (
                    "none"
                    if result is None
                    else _normalize_status(
                        _field(result, "outcome", missing),
                        _XAI_DEFERRED_STREAM_MUTATION_OUTCOMES,
                    )
                ),
            }
            operation = {
                "finalizer_enter": "success_finalizer",
                "finalizer_result": "success_finalizer",
                "release_result": "release",
                "cleanup_outcome": "release",
            }.get(phase)
            if operation is not None:
                snapshot["operation"] = operation
            if result is None:
                snapshot.update(
                    {
                        "result_error_present": False,
                        "returned_owner_record_present": False,
                        "returned_owner_state": "unknown",
                    }
                )
                return snapshot
            result_error = _field(result, "error", missing)
            owner_record = _field(result, "owner_record", missing)
            snapshot["result_error_present"] = (
                result_error is not missing and result_error is not None
            )
            snapshot["returned_owner_record_present"] = (
                owner_record is not missing and owner_record is not None
            )
            snapshot["returned_owner_state"] = _normalize_status(
                _field(owner_record, "state", missing),
                _XAI_DEFERRED_STREAM_OWNER_STATES,
            )
            result_key = _field(result, "cache_key", missing)
            result_key_hash = (
                _hash_session_owner_log_identifier(result_key)
                if result_key is not missing
                else None
            )
            if result_key_hash is not None:
                snapshot["result_owner_key_hash"] = result_key_hash
            return snapshot

        def _observe(phase: Any, **fields: Any) -> None:
            try:
                normalized_phase = _normalize_phase(phase)
                payload: dict[str, Any] = {
                    "event": _XAI_DEFERRED_STREAM_EVENT,
                    "phase": normalized_phase,
                    **correlation,
                    "success_finalizer_source": success_finalizer_source,
                    **_lease_snapshot(),
                    **_validation_snapshot(),
                    **_iterator_snapshot(fields.get("iterator", missing)),
                    **_finalization_task_snapshot(
                        fields.get("finalization_task", missing)
                    ),
                }

                if normalized_phase == "binding":
                    payload["binding_outcome"] = _normalize_status(
                        fields.get("binding_outcome", missing),
                        _XAI_DEFERRED_STREAM_BINDING_OUTCOMES,
                    )
                elif normalized_phase == "renewal_failed":
                    payload["site"] = _normalize_status(
                        fields.get("site", missing),
                        _XAI_DEFERRED_STREAM_SITES,
                    )
                elif normalized_phase == "validator_decision":
                    payload["validation_ok"] = _optional_bool(
                        fields.get("validation_ok", missing)
                    )
                elif normalized_phase == "terminal_delivered":
                    payload["terminal_delivered"] = _optional_bool(
                        fields.get("terminal_delivered", missing)
                    )
                    payload["wire_disposition"] = _normalize_status(
                        fields.get("wire_disposition", missing),
                        _XAI_DEFERRED_STREAM_WIRE_DISPOSITIONS,
                    )
                    payload["finalization_basis"] = (
                        _normalize_finalization_basis(
                            fields.get("finalization_basis", missing)
                        )
                    )
                    payload["validation_ok"] = _optional_bool(
                        fields.get("validation_ok", missing)
                    )
                elif normalized_phase in {
                    "finalization_task_created",
                    "finalization_task_reused",
                }:
                    payload["finalization_basis"] = (
                        _normalize_finalization_basis(
                            fields.get("finalization_basis", missing)
                        )
                    )

                if normalized_phase in _XAI_DEFERRED_STREAM_REQUESTED_SUCCESS_PHASES:
                    payload["requested_success"] = _optional_bool(
                        fields.get("requested_success", missing)
                    )

                for key in ("iterator_wrapped", "stream_response_wrapped"):
                    if key in fields:
                        payload[key] = _optional_bool(fields.get(key))

                error_category = _exception_category(fields.get("error"))
                if error_category is None:
                    error_category = _exception_category(fields.get("renewal_error"))
                if error_category is not None:
                    payload["exception_category"] = error_category

                if "result" in fields:
                    payload.update(
                        _mutation_snapshot(fields.get("result"), normalized_phase)
                    )

                verbose_aawm_route_logger.info(
                    "AAWM_XAI_DEFERRED_STREAM: "
                    + json.dumps(payload, sort_keys=True, separators=(",", ":"))
                )
            except BaseException:  # noqa: BLE001
                return None

        return _observe
    except BaseException:  # noqa: BLE001
        return _noop_observe


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
    attempted_provider_call: bool = False,
) -> str:
    reason = _sanitize_session_owner_log_label(
        mismatch_reason or failure_phase,
        max_length=_SESSION_OWNER_LOG_MAX_REASON_CHARS,
    ) or "session-owner mismatch"
    return (
        "LiteLLM Proxy: HTTP 409 session-owner mismatch requires redispatch; "
        "redispatch_required=true; "
        f"attempted_provider_call={str(bool(attempted_provider_call)).lower()}; "
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
    attempted_provider_call: bool = False,
    terminal_marker: Any = None,
    replay_safety: Optional[SessionOwnerReplaySafetyResult] = None,
) -> None:
    """Emit one proxy ERROR and one rollup failure. Never raises."""
    summary = _build_session_owner_redispatch_summary(
        mismatch_reason=mismatch_reason,
        failure_phase=failure_phase,
        attempted_provider_call=attempted_provider_call,
    )
    continuity_receipt = get_session_owner_continuity_receipt(request)
    try:
        endpoint = None
        if request is not None:
            request_url = getattr(request, "url", None)
            endpoint = getattr(request_url, "path", None)
        request_context: Mapping[str, Any] = {}
        request_call_id = None
        if request is not None:
            state = getattr(request, "state", None)
            context = (
                getattr(state, _SESSION_OWNER_REQUEST_CONTEXT_STATE_KEY, None)
                if state is not None
                else None
            )
            if isinstance(context, Mapping):
                request_context = context
                request_call_id = context.get("litellm_call_id")
            if request_call_id is None and state is not None:
                request_call_id = getattr(
                    state,
                    _SESSION_OWNER_REQUEST_CALL_ID_STATE_KEY,
                    None,
                )
        _emit_aawm_terminal_error(
            {
                "event_type": "redispatch_required",
                "endpoint": endpoint,
                "alias_family": (
                    owner_attrs.get("state_format")
                    or owner_attrs.get("route_family")
                    or "session_owner"
                ),
                "alias_model": alias_model,
                "selected_provider": (
                    owner_attrs.get("provider")
                    or shaped_candidate.get("provider")
                ),
                "selected_model": (
                    owner_attrs.get("model")
                    or shaped_candidate.get("model")
                ),
                "selected_route": (
                    owner_attrs.get("route_family")
                    or shaped_candidate.get("route_family")
                    or candidate_endpoint
                ),
                "account_hash": (
                    owner_attrs.get("account_hash")
                    or shaped_candidate.get("account_hash")
                    or shaped_candidate.get("codex_oauth_account_hash")
                    or shaped_candidate.get("xai_oauth_account_hash")
                ),
                "account_lane": (
                    owner_attrs.get("account_lane")
                    or shaped_candidate.get("account_lane")
                    or shaped_candidate.get("codex_oauth_lane_key")
                    or shaped_candidate.get("xai_oauth_lane_key")
                ),
                "status_code": _SESSION_OWNER_LOG_STATUS_CODE,
                "error_code": _SESSION_OWNER_REDISPATCH_ERROR_CODE,
                "failure_class": "session_owner_redispatch",
                "failure_phase": failure_phase,
                "attempted_provider_call": bool(attempted_provider_call),
                "redispatch_required": True,
                "terminal_outcome": "redispatch_required",
                "fallback_result": "none",
                "litellm_call_id": request_call_id,
                "trace_id": request_context.get("trace_id"),
                "session_id": session_identity,
            },
            marker=terminal_marker,
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        from .audit_events import (
            _emit_auto_agent_alias_pre_attempt_terminal_event,
        )

        _emit_auto_agent_alias_pre_attempt_terminal_event(
            alias_family=(
                owner_attrs.get("state_format")
                or owner_attrs.get("route_family")
                or "session_owner"
            ),
            alias_model=alias_model or "unknown",
            request=request,
            request_body={},
            event_type="redispatch_required",
            candidate_status="redispatch_required",
            failure_phase=failure_phase,
            error_status_code=_SESSION_OWNER_LOG_STATUS_CODE,
            error_code=_SESSION_OWNER_REDISPATCH_ERROR_CODE,
            candidate={
                **dict(shaped_candidate),
                "attempted_provider_call": bool(attempted_provider_call),
            },
            detail={
                "session_id": session_identity,
                "trace_id": request_context.get("trace_id"),
                "litellm_call_id": request_call_id,
            },
            extra_fields={
                "session_id": session_identity,
                "trace_id": request_context.get("trace_id"),
                "litellm_call_id": request_call_id,
                "replay_safety": _bounded_replay_safety_detail(replay_safety),
                "attempted_provider_call": bool(attempted_provider_call),
                "session_owner_mismatch_reason": _sanitize_session_owner_log_label(
                    mismatch_reason,
                    max_length=_SESSION_OWNER_LOG_MAX_REASON_CHARS,
                ),
                "session_owner_continuity_receipt": continuity_receipt,
            },
            failure_class="session_owner_redispatch",
            attempts=None,
            redispatch_required=True,
        )
    except Exception:
        pass
    try:
        # Keep the bounded summary for the durable route rollup. It is
        # intentionally separate from the operator ERROR field allowlist.
        verbose_proxy_logger.debug(
            "Session-owner redispatch ERROR boundary recorded",
            extra={
                "source": "session_owner_affinity",
                "status_code": _SESSION_OWNER_LOG_STATUS_CODE,
            },
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


def _register_session_owner_inbound_access_log_replacement(request: Any) -> None:
    """Consume leftover uvicorn ACCESS for exact POST responses 409."""
    try:
        from litellm.proxy.aawm_route_logging import (
            _register_aawm_route_access_log_replacement,
        )

        if request is None:
            return
        scope = getattr(request, "scope", None)
        if not isinstance(scope, dict):
            return
        if scope.get("method") != "POST":
            return
        if scope.get("path") != "/openai_passthrough/responses":
            return
        _register_aawm_route_access_log_replacement(
            request,
            suppress_all_statuses=True,
        )
    except Exception:  # noqa: BLE001
        pass


def _bounded_replay_safety_detail(
    replay_safety: Optional[SessionOwnerReplaySafetyResult],
) -> Optional[dict[str, str]]:
    if (
        replay_safety is None
        or replay_safety.safe
        or replay_safety.field_path is None
        or replay_safety.classification not in _REPLAY_SAFETY_CLASSIFICATIONS
        or len(replay_safety.field_path) > _REPLAY_SAFETY_MAX_FIELD_PATH_CHARS
    ):
        return None
    field_path = replay_safety.field_path
    classification = replay_safety.classification
    if classification == "invalid_body_shape":
        if field_path != "$":
            return None
    else:
        allowed_terminal_fields = (
            {"previous_response_id"}
            if classification == "previous_response_id"
            else (
                {"id"}
                if classification == "id_only_reasoning_reference"
                else _REPLAY_SAFETY_EXPLICIT_REFERENCE_KEYS
            )
        )
        terminal_field = next(
            (
                field
                for field in allowed_terminal_fields
                if field_path.endswith(f".{field}")
            ),
            None,
        )
        if terminal_field is None:
            return None
        structural_prefix = field_path[: -len(terminal_field) - 1]
        if re.fullmatch(
            r"\$(?:(?:\.\*|\.\*\*|\[\d+\]))*",
            structural_prefix,
        ) is None:
            return None
    return {
        "field_path": field_path,
        "classification": classification,
    }


def _build_minimized_replay_unsafe_detail(
    *,
    error_detail: Mapping[str, str],
    failure_phase: str,
    alias_model: Optional[str],
    replay_safety_detail: Optional[dict[str, str]],
) -> Optional[dict[str, Any]]:
    if (
        failure_phase != "session_owner_replay_unsafe_auto_review"
        or replay_safety_detail is None
    ):
        return None
    detail: dict[str, Any] = {
        "error": dict(error_detail),
        "redispatch_required": True,
        "redispatch_reason": replay_safety_detail["classification"],
        "failure_phase": failure_phase,
        "attempted_provider_call": False,
        "replay_safety": replay_safety_detail,
    }
    if alias_model in {"codex-auto-review", "auto-review"}:
        detail["alias_model"] = alias_model
        detail["redispatch_model"] = alias_model
    return detail


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
    attempted_provider_call: bool = False,
    terminal_marker: Any = None,
    replay_safety: Optional[SessionOwnerReplaySafetyResult] = None,
) -> None:
    """Raise structured redispatch_required with truthful egress state.

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
            "xai_oauth_account_label",
            "xai_oauth_account_hash",
            "xai_oauth_scope_identity",
            "xai_oauth_lane_key",
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

    error_detail = {
        "message": message
        or (
            "Session ownership requires a fresh dispatch. The current "
            "session is pinned to a different or unavailable owner; do not "
            "continue this session against another provider/model/route/"
            "account. Redispatch with a new session identity."
        ),
        "type": "invalid_request_error",
        "code": _SESSION_OWNER_REDISPATCH_ERROR_CODE,
        "retryable": True,
    }
    replay_safety_detail = _bounded_replay_safety_detail(replay_safety)
    detail = _build_minimized_replay_unsafe_detail(
        error_detail=error_detail,
        failure_phase=failure_phase,
        alias_model=alias_model,
        replay_safety_detail=replay_safety_detail,
    )
    if detail is None:
        detail = {
            "error": error_detail,
            "redispatch_required": True,
            "redispatch_reason": mismatch_reason or failure_phase,
            "failure_phase": failure_phase,
            "attempted_provider_call": bool(attempted_provider_call),
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
        if replay_safety_detail is not None:
            detail["replay_safety"] = replay_safety_detail

    # Keep the legacy attribution argument for callers; reopened D1-614 intentionally
    # does not emit its values.
    _ = attribution
    # A competing reservation is a bounded, pre-egress coordination race.
    # The alias selector may retry it internally; emit terminal observability
    # only if the bounded retry ultimately surfaces the conflict.
    if not (
        failure_phase == "session_owner_competing_reservation"
        and _competing_reservation_log_is_deferred(request)
    ):
        record_session_owner_continuity_receipt(
            request,
            phase="owner_rejection",
            source="redispatch",
            session_identity=session_identity,
            cache_key=cache_key,
            outcome="redispatch_required",
            reason_code="guard_rejected",
        )
        _emit_session_owner_redispatch_observability(
            session_identity=session_identity,
            failure_phase=failure_phase,
            mismatch_reason=mismatch_reason,
            alias_model=alias_model,
            shaped_candidate=shaped_candidate,
            candidate_endpoint=candidate_endpoint,
            owner_attrs=owner_attrs,
            request=request,
            attempted_provider_call=attempted_provider_call,
            terminal_marker=(
                terminal_marker
                if terminal_marker is not None
                else getattr(request, "state", None)
            ),
            replay_safety=replay_safety,
        )
    # Direct OpenAI / nested Codex guards raise 409 before
    # pass_through_request registers ACCESS replacement. Register once so
    # the leftover uvicorn ACCESS for this exact POST
    # /openai_passthrough/responses 409 is consumed.
    _register_session_owner_inbound_access_log_replacement(request)

    raise HTTPException(
        status_code=409,
        detail=detail,
        headers={"Retry-After": "1"},
    )




_REQUEST_STATE_LEASE_ATTR = "_aawm_session_owner_lease"
_REQUEST_STATE_GUARDED_ATTR = "_aawm_session_owner_guarded"


def _cursor_replay_body_fingerprint(body: Any) -> Optional[str]:
    """Return a stable digest for a JSON request body, without retaining values."""

    if not isinstance(body, dict):
        return None
    try:
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


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


def validate_cursor_replay_matches_body(
    request: Any,
    *,
    body: Any,
) -> bool:
    """Return true only for validation bound to this exact body object."""

    if request is None:
        return False
    state = getattr(request, "state", None)
    if state is None:
        return False
    validation = getattr(
        state,
        "_aawm_validated_cursor_replay",
        None,
    )
    if (
        not isinstance(validation, Mapping)
        or validation.get("body_ref") is not body
        or validation.get("body_id") != id(body)
    ):
        return False
    fingerprint = _cursor_replay_body_fingerprint(body)
    return (
        fingerprint is not None
        and validation.get("body_fingerprint") == fingerprint
    )


def set_validated_cursor_replay(
    request: Any,
    *,
    body: Any,
    stage: str,
    reason: str,
) -> None:
    """Bind server-owned replay validation to the exact rebuilt body."""

    if request is None or not isinstance(body, dict):
        return
    state = getattr(request, "state", None)
    if state is None:
        return
    setattr(
        state,
        "_aawm_validated_cursor_replay",
        {
            "body_ref": body,
            "body_id": id(body),
            "body_fingerprint": _cursor_replay_body_fingerprint(body),
            "stage": stage,
            "reason": reason,
        },
    )


def reset_released_request_session_owner_guard(request: Any) -> bool:
    """Clear a released fresh-request reservation before account failover.

    This only resets request-local state from a fresh dispatch whose reservation
    was released. It does not rebind a durable compatible owner.
    """
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
    if lease is not None and (lease.held_reservation or lease.promoted):
        return False
    setattr(state, _REQUEST_STATE_LEASE_ATTR, None)
    setattr(state, _REQUEST_STATE_GUARDED_ATTR, False)
    return True


def clear_expected_non_held_request_session_owner_lease(
    request: Any,
    *,
    expected_session_identity: str,
) -> bool:
    """Clear only the expected compatible, non-held base lease."""

    if request is None:
        return False
    state = getattr(request, "state", None)
    if state is None:
        return False
    lease = get_request_session_owner_lease(request)
    if lease is None:
        return True
    expected = _clean_optional_str(expected_session_identity)
    lease_identity = _clean_optional_str(lease.session_identity)
    if (
        expected is None
        or lease_identity is None
        or is_session_owner_redispatch_effective_identity(expected)
        or lease.decision != SessionOwnerGuardDecision.COMPATIBLE_OWNER.value
        or lease.held_reservation
        or lease.promoted
        or lease.released
        or _clean_optional_str(lease.owner_id) is None
    ):
        return False
    if (
        _strip_legacy_affinity_prefixes(lease_identity)
        != _strip_legacy_affinity_prefixes(expected)
    ):
        return False
    return clear_non_held_request_session_owner_lease(request)


async def clear_compatible_non_held_request_session_owner_guard_for_failover(
    *,
    request: Any,
    request_body: Optional[Mapping[str, Any]],
    alternate_attributes: Optional[Mapping[str, Any]],
    current_attributes: Optional[Mapping[str, Any]] = None,
    account_failover_planned: bool,
    account_failover_replay_safe: bool,
    has_account_bound_state: bool,
    post_commit_retry: bool,
    failover_ordinal: int = 1,
    validate_durable_owner: bool = True,
) -> SessionOwnerLeaseRebindResult:
    """Validate one portable account move, then clear only request state.

    A released fresh reservation may retain an ``UNOWNED_RESERVED`` or
    ``RESERVATION_RENEWED`` decision, which is distinct from a live
    ``COMPATIBLE_OWNER`` durable owner. Released request-local state is cleared
    only after the same portability and ownership checks.
    """

    if not request_session_owner_already_guarded(request):
        return SessionOwnerLeaseRebindResult(False, "guard_not_acquired")
    lease = get_request_session_owner_lease(request)
    if lease is None:
        return SessionOwnerLeaseRebindResult(False, "lease_missing")
    released_decisions = {
        SessionOwnerGuardDecision.UNOWNED_RESERVED.value,
        SessionOwnerGuardDecision.RESERVATION_RENEWED.value,
        SessionOwnerGuardDecision.COMPATIBLE_OWNER.value,
    }
    if (
        lease.released
        and lease.decision not in released_decisions
    ) or (
        not lease.released
        and lease.decision != SessionOwnerGuardDecision.COMPATIBLE_OWNER.value
    ):
        return SessionOwnerLeaseRebindResult(
            False,
            "lease_decision_not_compatible_owner",
        )
    if lease.promoted:
        return SessionOwnerLeaseRebindResult(False, "lease_promoted")
    if not lease.released and lease.held_reservation:
        return SessionOwnerLeaseRebindResult(False, "lease_held_reservation")
    if _clean_optional_str(lease.owner_id) is None:
        return SessionOwnerLeaseRebindResult(False, "lease_owner_missing")
    if not account_failover_planned:
        return SessionOwnerLeaseRebindResult(
            False,
            "account_failover_not_planned",
        )
    if not account_failover_replay_safe:
        return SessionOwnerLeaseRebindResult(False, "replay_unsafe")
    if has_account_bound_state:
        return SessionOwnerLeaseRebindResult(
            False,
            "account_bound_state_nonportable",
        )
    if post_commit_retry:
        return SessionOwnerLeaseRebindResult(False, "post_commit_retry")
    if failover_ordinal != 1:
        return SessionOwnerLeaseRebindResult(False, "unexpected_failover_ordinal")
    if not is_replay_safe_session_owner_redispatch_body(request_body):
        return SessionOwnerLeaseRebindResult(False, "body_not_replay_safe")

    current = _core_owner_attributes(
        build_session_owner_attributes(extra=current_attributes or lease.attributes)
    )
    alternate = _core_owner_attributes(
        build_session_owner_attributes(extra=alternate_attributes or {})
    )
    for attrs in (current, alternate):
        if (
            incomplete_owner_attribute_reason(attrs, for_promotion=True)
            is not None
            or not any(
                _clean_optional_str(attrs.get(key))
                for key in ("account_hash", "account_lane", "account_label")
            )
        ):
            return SessionOwnerLeaseRebindResult(
                False,
                "incomplete_owner_attributes",
            )
    if not _accounts_are_interchangeable(current, alternate):
        return SessionOwnerLeaseRebindResult(
            False,
            "accounts_not_interchangeable",
        )
    if _clean_optional_str(current.get("model")) != _clean_optional_str(
        alternate.get("model")
    ):
        return SessionOwnerLeaseRebindResult(False, "model_mismatch")
    if (
        _compatibility_mismatch_reason(
            owner_record={
                _RECORD_STATE_FIELD: SessionOwnerRecordState.OWNED.value,
                _RECORD_OWNER_FIELD: lease.owner_id,
                _RECORD_ATTRIBUTES_FIELD: current,
            },
            requested_attributes=alternate,
            require_exact_attributes=True,
        )
        is not None
    ):
        return SessionOwnerLeaseRebindResult(False, "request_lease_owner_mismatch")

    source_attributes: Optional[Mapping[str, Any]] = None
    if validate_durable_owner:
        owner_record, _, error = await get_session_owner_record(
            session_identity=lease.session_identity,
            request=request,
            wait_for_foreign_reservation=False,
        )
        durable_record_absent = owner_record is None and error is None
        if lease.released and durable_record_absent:
            owner_record = None
        elif (
            error is not None
            or owner_record is None
            or _record_state(owner_record) != SessionOwnerRecordState.OWNED.value
            or _clean_optional_str(owner_record.get(_RECORD_OWNER_FIELD))
            != _clean_optional_str(lease.owner_id)
        ):
            return SessionOwnerLeaseRebindResult(
                False,
                "durable_owner_changed_or_missing",
            )
        if owner_record is not None:
            owner_attributes = _core_owner_attributes(
                _owner_attributes(owner_record)
            )
            if (
                incomplete_owner_attribute_reason(
                    owner_attributes, for_promotion=True
                )
                is not None
                or not _accounts_are_interchangeable(
                    owner_attributes, alternate
                )
                or _compatibility_mismatch_reason(
                    owner_record=owner_record,
                    requested_attributes=current,
                    require_exact_attributes=True,
                )
                is not None
            ):
                return SessionOwnerLeaseRebindResult(
                    False,
                    "durable_owner_mismatch",
                )
            source_attributes = dict(owner_attributes)

    clear_request_lease = (
        reset_released_request_session_owner_guard
        if lease.released
        else clear_non_held_request_session_owner_lease
    )
    if not clear_request_lease(request):
        return SessionOwnerLeaseRebindResult(False, "request_lease_clear_failed")
    return SessionOwnerLeaseRebindResult(
        True,
        source_attributes=source_attributes,
    )


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
    strict_managed_openai_owner: bool = False,
    alias_model: Optional[str] = None,
    failure_phase: str = "session_owner_pre_egress",
    raise_on_redispatch: bool = True,
    policy: Optional[SessionOwnerLeasePolicy] = None,
) -> SessionOwnerGuardResult:
    """Idempotent request-scoped guard used by every route family.

    If this request already holds a lease, renew/validate rather than creating
    a second competing reservation.
    """

    resolved_session_identity = resolve_canonical_session_identity(
        request,
        request_body,
        session_identity=session_identity,
    )
    existing = get_request_session_owner_lease(request)
    lease_policy = resolve_session_owner_lease_policy(
        request,
        alias_model=alias_model,
        existing_lease=existing,
        policy=policy,
    )
    active_lease = (
        existing
        if existing is not None and not existing.released and not existing.promoted
        else None
    )
    if active_lease is not None:
        active_lease.policy = lease_policy
        lease_identity = _clean_optional_str(active_lease.session_identity)
        identities_match = (
            resolved_session_identity is not None
            and lease_identity is not None
            and _strip_legacy_affinity_prefixes(lease_identity)
            == _strip_legacy_affinity_prefixes(resolved_session_identity)
        )
        if not identities_match:
            mismatch_reason = (
                "session_owner: request lease identity does not match "
                "the requested session identity"
            )
            mismatch_cache_key = (
                build_aawm_alias_routing_session_owner_cache_key(
                    session_identity=resolved_session_identity
                )
                if resolved_session_identity is not None
                else None
            )
            guard = SessionOwnerGuardResult(
                decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED,
                session_identity=resolved_session_identity,
                cache_key=mismatch_cache_key,
                owner_id=active_lease.owner_id,
                reservation_token=active_lease.reservation_token,
                mismatch_reason=mismatch_reason,
                provenance=build_session_owner_provenance(
                    session_identity=resolved_session_identity,
                    decision=SessionOwnerGuardDecision.REDISPATCH_REQUIRED.value,
                    owner_id=active_lease.owner_id,
                    mismatch_reason=mismatch_reason,
                    cache_key=mismatch_cache_key,
                    reservation_token=active_lease.reservation_token,
                ),
            )
            record_session_owner_continuity_receipt(
                request,
                phase="owner_guard",
                source="request_lease",
                session_identity=resolved_session_identity,
                cache_key=mismatch_cache_key,
                outcome=guard.decision.value,
                reason_code="identity_conflict",
            )
            if raise_on_redispatch:
                raise_session_owner_redispatch_required(
                    session_identity=resolved_session_identity,
                    guard=guard,
                    alias_model=alias_model,
                    candidate=requested_attributes or candidate,
                    failure_phase="session_owner_request_lease_identity_conflict",
                    request=request,
                )
            return guard
    token = active_lease.reservation_token if active_lease is not None else None
    guard = await guard_session_owner_before_egress(
        session_identity=resolved_session_identity,
        request=request,
        request_body=request_body,
        requested_attributes=requested_attributes
        or (active_lease.attributes if active_lease is not None else None),
        candidate=candidate,
        owner_id=owner_id
        or (active_lease.owner_id if active_lease is not None else None),
        reservation_token=token,
        require_exact_attributes=require_exact_attributes,
        strict_managed_openai_owner=strict_managed_openai_owner,
    )
    record_session_owner_continuity_receipt(
        request,
        phase="owner_guard",
        source="session_owner_guard",
        session_identity=guard.session_identity,
        cache_key=guard.cache_key,
        outcome=guard.decision.value,
        reason_code="guard_rejected"
        if guard.mismatch_reason
        else None,
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
        active_lease.session_identity = guard.session_identity
        active_lease.cache_key = guard.cache_key
        active_lease.reservation_token = guard.reservation_token
        active_lease.held_reservation = True
        active_lease.decision = guard.decision.value
        active_lease.owner_id = guard.owner_id or active_lease.owner_id
        active_lease.promoted = False
        active_lease.released = False
        active_lease.policy = lease_policy
        set_request_session_owner_lease(request, active_lease)
    else:
        lease = lease_from_guard_result(
            guard,
            attributes=requested_attributes
            or (active_lease.attributes if active_lease is not None else None),
            policy=lease_policy,
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


async def finalize_codex_auto_review_lease_on_success(
    lease: Optional[SessionOwnerLease],
) -> Optional[SessionOwnerMutationResult]:
    """Release the request-local reservation after an authoritative response.

    Auto-review ownership is scoped to one approval invocation.  Reusing the
    normal promotion helper would create durable guardian affinity, defeating
    free candidate selection on the next review.
    """

    return await _release_session_owner_lease_on_terminal(
        lease,
        force_release_only=True,
    )
