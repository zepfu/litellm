"""Stdlib ChatGPT conversation-init observer for the provider-status sidecar.

Maps a credential-safe browser-boundary snapshot of
``POST /backend-api/conversation/init`` into sanitized rate-limit
observations. This module is sidecar-packaged by itself; it must not import
the ChatGPT authenticator, ``common_utils``, ``httpx``, or any other
provider implementation.

The current frontend contract is POST with no body. GET is invalid. The
sidecar never reads Oracle cookies, never launches a browser, never
HTTP-calls chatgpt.com, and never sends a model message. It consumes an
already-collected JSON snapshot and sanitizes it again before persistence.

The browser-boundary collector lives in this same stdlib module. It accepts
an injected transport, issues the no-body POST contract, writes a
credential-safe snapshot, and can be fixture-tested without live auth. It
does not copy Oracle cookies or persist headers, tokens, or personal data.
Bound Oracle capture requires an exact existing CDP target and an expected
canonical account hash. The bound path only verifies an authoritative
``account_id``/``chatgpt_account_id`` from the same response; missing identity
fails closed because this module has no established same-context metadata
contract to substitute. Live authenticated Oracle-browser proof remains a
separate acceptance gate.

The native history observer is a separate attach-only, read-only path. It
performs one ordinary ChatGPT home navigation, observes at most one native
``GET /backend-api/conversations`` request, and returns structural metadata
only. It never calls the endpoint directly, supplies guessed headers, reads
cookies or storage, follows pagination, requests conversation details, or
sends model/mutation traffic.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from math import isfinite
import multiprocessing
import os
import re
import select
import signal
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)
from urllib import request as urllib_request
from urllib.parse import urlsplit

CHATGPT_CONVERSATION_INIT_PROVIDER = "chatgpt"
CHATGPT_CONVERSATION_INIT_SOURCE = "chatgpt_conversation_init"
CHATGPT_CONVERSATION_INIT_PARSER_VERSION = "chatgpt_conversation_init_v1"
CHATGPT_CONVERSATION_INIT_CLIENT = "chatgpt-web"
CHATGPT_CONVERSATION_INIT_METHOD = "POST"
CHATGPT_CONVERSATION_INIT_PATH = "/backend-api/conversation/init"
CHATGPT_CONVERSATION_INIT_DEFAULT_URL = (
    "https://chatgpt.com/backend-api/conversation/init"
)
CHATGPT_CONVERSATION_INIT_SNAPSHOT_QUOTA_KEY = (
    "chatgpt_conversation_init:snapshot"
)
CHATGPT_CONVERSATION_INIT_ACCOUNT_IDENTITY_SOURCE = (
    "chatgpt.conversation_init.account"
)
CHATGPT_CONVERSATION_INIT_BROWSER_ACCOUNT_IDENTITY_SOURCE = (
    "chatgpt.conversation_init.oracle_browser.account_id"
)
CHATGPT_CONVERSATION_INIT_VERIFIED_PAYLOAD_IDENTITY_SOURCE = "provider_payload"
CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE = (
    "native_request_header"
)
CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_SELECTOR_EVIDENCE = (
    "request_and_extra_info"
)
CHATGPT_CONVERSATION_INIT_SOURCE_IDENTITY_SOURCE = (
    "chatgpt.conversation_init.collector_source_path"
)
CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH = 12
CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM = "sha256"
CHATGPT_CONVERSATION_INIT_BROWSER_CDP_ENDPOINT_ENV = (
    "AAWM_CHATGPT_CONVERSATION_INIT_BROWSER_CDP_ENDPOINT"
)
ORACLE_BROWSER_CDP_ENDPOINT_ENV = "ORACLE_BROWSER_CDP_ENDPOINT"
DEFAULT_ORACLE_BROWSER_CDP_ENDPOINT = "http://127.0.0.1:9222"
ORACLE_BROWSER_BOUNDARY_NAME = "oracle_browser_cdp_attach"
CHATGPT_NATIVE_HISTORY_HOME_URL = "https://chatgpt.com/"
CHATGPT_NATIVE_HISTORY_INDEX_PATH = "/backend-api/conversations"
CHATGPT_NATIVE_HISTORY_EXPECTED_ACCOUNT_HASH = "8e92854835c4"
CHATGPT_NATIVE_HISTORY_OBSERVER = "chatgpt_native_history"
CHATGPT_NATIVE_HISTORY_ROLE_ENV = "AAWM_CHATGPT_NATIVE_HISTORY_ROLE"
CHATGPT_NATIVE_HISTORY_DEFAULT_TIMEOUT_SECONDS = 150.0
CHATGPT_NATIVE_HISTORY_MAX_RESPONSE_BYTES = 1_048_576
# Shared across the history worker and its parent.  A target stays owned until
# the close acknowledgement is published; the parent must not reap the worker
# while the state is still pending.
_NATIVE_HISTORY_TARGET_NONE = 0
_NATIVE_HISTORY_TARGET_CREATING = 1
_NATIVE_HISTORY_TARGET_ATTACHED = 2
_NATIVE_HISTORY_TARGET_CLEANUP_PENDING = 3
_NATIVE_HISTORY_TARGET_CLOSED = 4
_NATIVE_HISTORY_TARGET_CLEANUP_HANDOFF = 5
_NATIVE_HISTORY_TARGET_NO_CREATE = 6
_NATIVE_HISTORY_TARGET_BROWSER_TERMINATED = 7
_NATIVE_HISTORY_CLEANUP_RESERVE_SECONDS = 10.0
_NATIVE_HISTORY_TERM_GRACE_SECONDS = 1.0
_NATIVE_HISTORY_KILL_GRACE_SECONDS = 1.0
_NATIVE_HISTORY_REAP_GRACE_SECONDS = 1.0
_NATIVE_HISTORY_MAX_PROCESS_INVENTORY_ENTRIES = 4096


# Truncation does not silently claim completeness.
_TRUNCATION_WARNING = "truncated"

INVALID_CONVERSATION_INIT_METHODS = ("GET", "HEAD")
MAX_CONVERSATION_INIT_SOURCE_BYTES = 1_000_000
_ORACLE_BROWSER_IPC_FRAME_BYTES = 4096
_ORACLE_BROWSER_IPC_MAX_BYTES = MAX_CONVERSATION_INIT_SOURCE_BYTES
MAX_PROJECTION_DEPTH = 5
MAX_PROJECTION_LIST_ITEMS = 200
MAX_PROJECTION_OBJECT_KEYS = 200
MAX_SAFE_STRING_LENGTH = 256
_NATIVE_HISTORY_MAX_INSPECTED_NODES = 512
_NATIVE_HISTORY_MAX_COUNT = 200
_NATIVE_HISTORY_MAX_PENDING_NETWORKS = 256
_NATIVE_HISTORY_FIELD_NAMES = (
    "conversations",
    "items",
    "total",
    "offset",
    "limit",
    "has_missing_conversations",
    "updated_at",
    "updatedAt",
    "update_time",
    "updateTime",
)
_NATIVE_HISTORY_MODEL_FIELD_NAMES = (
    "model_slug",
    "requested_model",
    "default_model_slug",
    "model",
    "recorded_model",
    "recorded_final_model",
)
_NATIVE_HISTORY_FIELD_COUNTER_SPECS = {
    "model": (
        ("model_slug", ("model_slug", "modelSlug")),
        (
            "requested_model",
            (
                "requested_model",
                "requestedModel",
                "requested_model_slug",
                "requestedModelSlug",
            ),
        ),
        ("default_model_slug", ("default_model_slug", "defaultModelSlug")),
        ("model", ("model",)),
        ("recorded_model", ("recorded_model", "recordedModel")),
        (
            "recorded_final_model",
            ("recorded_final_model", "recordedFinalModel"),
        ),
    ),
    "updated_time": (
        ("updated_at", ("updated_at", "updatedAt")),
        ("update_time", ("update_time", "updateTime")),
        (
            "last_updated_at",
            ("last_updated_at", "lastUpdatedAt"),
        ),
    ),
    "pagination": (
        ("total", ("total",)),
        ("offset", ("offset",)),
        ("limit", ("limit",)),
        ("page", ("page",)),
        ("page_size", ("page_size", "pageSize")),
        ("has_more", ("has_more", "hasMore")),
        (
            "has_missing_conversations",
            ("has_missing_conversations", "hasMissingConversations"),
        ),
        ("next_cursor", ("next_cursor", "nextCursor")),
        ("cursor", ("cursor",)),
    ),
}
_NATIVE_HISTORY_VALUE_TYPES = (
    "null",
    "boolean",
    "number",
    "string",
    "object",
    "array",
    "unsupported",
)
_NATIVE_HISTORY_MODEL_FIELD_ALIASES = {
    "model_slug": ("model_slug", "modelSlug"),
    "requested_model": (
        "requested_model",
        "requestedModel",
        "requested_model_slug",
        "requestedModelSlug",
    ),
    "default_model_slug": ("default_model_slug", "defaultModelSlug"),
    "model": ("model",),
    "recorded_model": ("recorded_model", "recordedModel"),
    "recorded_final_model": (
        "recorded_final_model",
        "recordedFinalModel",
    ),
}
_NATIVE_HISTORY_STREAM_CHUNK_BYTES = 64 * 1024
_NATIVE_HISTORY_BOOTSTRAP_READ_PATHS = frozenset(
    {
        "/api/auth/session",
        "/backend-api/me",
        "/backend-api/models",
        CHATGPT_CONVERSATION_INIT_PATH,
        CHATGPT_NATIVE_HISTORY_INDEX_PATH,
    }
)

_FORBIDDEN_REQUEST_CONTENT_FIELDS = (
    "messages",
    "message",
    "input",
    "prompt",
    "conversation",
    "conversation_id",
    "conversationId",
    "content",
)
_ENVELOPE_KEYS = {
    "headers",
    "header",
    "cookies",
    "cookie",
    "authorization",
    "status_code",
    "status",
    "statuscode",
    "request",
    "url",
    "method",
    "body",
    "payload",
    "response",
    "json",
    "data",
    "account_hash",
    "account_id",
    "chatgpt_account_id",
    "account_identity_hash_algorithm",
    "account_identity_hash_length",
    "account_identity_fields",
    "account_identity_source",
    "account_identity_verification_source",
    "account_identity_verified",
    "collector_source",
    "browser_boundary",
    "payload_schema",
    "payload_state",
    "redacted_field_count",
    "source_identity_hash",
    "native_capture",
    "native_capture_error",
    "browser_challenge",
    "retry_after_seconds",
    "request_body_omitted",
    "projection_truncated",
    "malformed_collection_projection",
}
_PAYLOAD_ENVELOPE_KEYS = ("body", "payload", "response", "json", "data")
_SECRET_KEY_MARKERS = (
    "cookie",
    "set-cookie",
    "set_cookie",
    "authorization",
    "auth_header",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "apikey",
    "api-key",
    "client_secret",
    "password",
    "passwd",
    "secret",
    "bearer",
    "token",
    "session",
    "storage",
    "indexeddb",
    "private_key",
    "privatekey",
    "email",
    "phone",
    "ssn",
    "credential",
)
_ACCOUNT_IDENTITY_KEYS = (
    "userid",
    "user_id",
    "accountid",
    "account_id",
    "chatgpt_account_id",
    "chatgptaccountid",
    "account_user_id",
    "profile_id",
    "profileid",
    "membershipid",
    "membership_id",
)
_CANONICAL_ACCOUNT_ID_KEYS = frozenset(
    {
        "account_id",
        "chatgpt_account_id",
    }
)
_CANONICAL_ACTIVE_ACCOUNT_ID_PATHS = (
    ("account_id",),
    ("chatgpt_account_id",),
    ("account", "account_id"),
    ("account", "chatgpt_account_id"),
)
_CANONICAL_ACCOUNT_ID_PROVENANCE_FIELDS = frozenset(
    ".".join(path) for path in _CANONICAL_ACTIVE_ACCOUNT_ID_PATHS
)
_NATIVE_CAPTURE_ERRORS = frozenset(
    {
        "native_capture_incomplete",
        "native_capture_invalid_account_hash",
        "native_capture_invalid_identity_source",
        "native_capture_incomplete_selector_evidence",
        "native_capture_uncorrelated",
        "native_capture_invalid_request_method",
        "native_capture_invalid_body_observation",
        "native_capture_invalid_browser_challenge",
        "conflicting_native_payload_account_id",
        "native_payload_identity_mismatch",
        "invalid_retry_after_seconds",
    }
)
_CANONICAL_ACCOUNT_HASH_RE = re.compile(
    rf"^[0-9a-f]{{{CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH}}}$"
)
_ACCOUNT_IDENTITY_CONTAINERS = ("user", "account", "profile", "membership")
_IDENTITY_FIELD_NAMES = (
    "feature",
    "feature_name",
    "feature_id",
    "featureId",
    "feature_slug",
    "model",
    "model_slug",
    "modelSlug",
    "slug",
    "id",
    "name",
    "key",
    "code",
    "type",
)
_PII_FIELD_NAMES = (
    "title",
    "titles",
    "display_name",
    "displayname",
    "full_name",
    "fullname",
    "first_name",
    "firstname",
    "last_name",
    "lastname",
    "username",
    "usernames",
    "user_name",
    "workspace",
    "workspace_name",
    "workspacename",
    "feature_note",
    "featurenote",
    "note",
    "notes",
    "description",
    "email",
)
_SAFE_STRING_FIELD_NAMES = {
    _name.lower()
    for _name in (
        *_IDENTITY_FIELD_NAMES,
        "default_model_slug",
        "intended_default_model_slug",
        "reset_after",
        "reset_at",
        "resets_at",
        "unit",
        "status",
        "state",
        "mode",
        "version",
        "category",
        "period",
        "window",
        "malformed_fields",
    )
}
_SAFE_NUMERIC_TOKEN_KEY_MARKERS = (
    "input",
    "output",
    "total",
    "count",
    "limit",
    "remaining",
    "used",
    "usage",
    "budget",
    "max",
    "cache",
    "cached",
    "prompt",
    "completion",
    "reasoning",
)

_REMAINING_KEYS = ("remaining", "remaining_count", "remainingCount", "left")
_LIMIT_KEYS = ("limit", "quota", "max", "total")
_USED_KEYS = ("used", "usage", "consumed")
_RESET_KEYS = (
    "reset_after",
    "resetAfter",
    "reset_at",
    "resetAt",
    "resets_at",
    "resetsAt",
)
_MALFORMED_FIELDS_KEY = "_malformed_fields"
_USAGE_NUMBER_FIELD_ALIASES = (
    *_REMAINING_KEYS,
    *_LIMIT_KEYS,
    *_USED_KEYS,
)
_NAMED_USAGE_COLLECTIONS = frozenset({"model_limits", "limits_progress"})
_NAMED_COLLECTION_IDENTITY_FIELDS = {
    "limits_progress": "feature",
    "model_limits": "model",
}
_COLLECTION_STATE_VALUES = frozenset(
    {"absent_unknown", "empty_unknown", "present", "partial", "malformed"}
)
_MAX_COLLECTION_DIAGNOSTIC_COUNT = MAX_PROJECTION_LIST_ITEMS * 4
_CONVERSATION_INIT_MARKERS = {
    "model_limits",
    "limits_progress",
    "blocked_features",
    "default_model_slug",
    "intended_default_model_slug",
}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_JWT_RE = re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{32,}$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ChatGPTConversationInitError(ValueError):
    """Sanitized conversation-init collector/parser failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        telemetry_class: str = "malformed_telemetry",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.telemetry_class = telemetry_class


def conversation_init_request_contract(
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the current frontend request contract: POST, no body."""

    resolved = (url or CHATGPT_CONVERSATION_INIT_DEFAULT_URL).strip()
    return {
        "method": CHATGPT_CONVERSATION_INIT_METHOD,
        "path": CHATGPT_CONVERSATION_INIT_PATH,
        "url": resolved or CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        "body": None,
        "body_omitted": True,
        "content_length": 0,
        "has_model_message": False,
        "has_conversation_content": False,
        "forbidden_fields": list(_FORBIDDEN_REQUEST_CONTENT_FIELDS),
        "invalid_methods": list(INVALID_CONVERSATION_INIT_METHODS),
        "notes": (
            "Current ChatGPT frontend issues POST /backend-api/conversation/init "
            "with no body. GET is invalid (browser GET -> 400, headless GET -> 403). "
            "The collector must not submit a model message or conversation content."
        ),
    }


def build_conversation_init_request(
    url: Optional[str] = None,
) -> urllib_request.Request:
    """Build the no-body POST Request the browser-boundary collector must send."""

    contract = conversation_init_request_contract(url)
    request = urllib_request.Request(
        contract["url"],
        data=None,
        method=CHATGPT_CONVERSATION_INIT_METHOD,
    )
    return request


def request_has_conversation_content(
    request: urllib_request.Request,
) -> bool:
    """Return True if the collector request would send conversation/message content."""

    if str(request.get_method() or "").upper() != CHATGPT_CONVERSATION_INIT_METHOD:
        return True
    body = request.data
    if body in (None, b"", ""):
        return False
    return True


def looks_like_conversation_init_payload(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return bool(_CONVERSATION_INIT_MARKERS.intersection(str(key) for key in payload))


def hash_chatgpt_conversation_init_account_identity(
    payload: Mapping[str, Any],
) -> Tuple[Optional[str], List[str]]:
    """Hash account identity fields. Never return raw tokens or account ids."""

    identity_parts: List[str] = []
    identity_fields: List[str] = []
    for path, mapping in _mapping_nodes(payload):
        for field_name, value in mapping.items():
            normalized = _normalize_key(field_name)
            if normalized not in _ACCOUNT_IDENTITY_KEYS:
                continue
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                prefix = ".".join(path + (str(field_name),)) if path else str(field_name)
                identity_parts.append(f"{prefix}={value}")
                identity_fields.append(prefix)
        for container_name in _ACCOUNT_IDENTITY_CONTAINERS:
            nested = mapping.get(container_name)
            if not isinstance(nested, Mapping):
                continue
            value = nested.get("id")
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                prefix = (
                    ".".join(path + (container_name, "id"))
                    if path
                    else f"{container_name}.id"
                )
                identity_parts.append(f"{prefix}={value}")
                identity_fields.append(prefix)
    if not identity_parts:
        return None, []
    unique_parts, unique_fields = _unique_pairs(identity_parts, identity_fields)
    material = (
        "chatgpt-conversation-init-account|" + "|".join(unique_parts)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest(), unique_fields


def hash_chatgpt_conversation_init_canonical_account_id(
    account_id: Any,
) -> Optional[str]:
    """Return the inventory-compatible canonical account hash.

    This deliberately accepts only the canonical account identity value. It
    does not hash user ids, source paths, composite field material, or an
    expected hash supplied by the caller.
    """

    cleaned = _clean_canonical_account_id(account_id)
    if cleaned is None:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[
        :CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
    ]


def _clean_canonical_account_id(value: Any) -> Optional[str]:
    """Apply the same string normalization used by the OAuth inventory."""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or any(ord(character) < 32 for character in cleaned):
        return None
    return cleaned


def _extract_canonical_account_identity(
    raw: Any,
    payload: Any,
) -> Tuple[Optional[str], List[str], Optional[str]]:
    """Extract account ids only from fixed active-account structures."""

    candidates: List[Tuple[str, str]] = []
    sources: List[Any] = [payload]
    if raw is not payload:
        sources.append(raw)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for path in _CANONICAL_ACTIVE_ACCOUNT_ID_PATHS:
            field_path = ".".join(path)
            for value in _values_at_structural_path(source, path):
                account_id = _clean_canonical_account_id(value)
                if account_id is not None:
                    candidates.append((account_id, field_path))

    unique_ids = {account_id for account_id, _field_path in candidates}
    fields = _unique_pairs(
        [account_id for account_id, _field_path in candidates],
        [field_path for _account_id, field_path in candidates],
    )[1]
    if not unique_ids:
        return None, fields, "missing_authoritative_account_id"
    if len(unique_ids) != 1:
        return None, fields, "conflicting_authoritative_account_id"
    account_id = next(iter(unique_ids))
    return account_id, fields, None


def hash_chatgpt_conversation_init_source_identity(source_path: str) -> str:
    """Hash a collector source path. Never return the raw path."""

    normalized = str(Path(str(source_path)).expanduser())
    material = f"chatgpt-conversation-init-source|{normalized}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def load_conversation_init_source(path: str) -> Any:
    """Load browser-boundary JSON from a regular file without following secrets."""

    source = Path(path).expanduser()
    try:
        if not source.is_file() or source.is_symlink():
            raise ChatGPTConversationInitError(
                "ChatGPT conversation-init source is missing or not a regular file.",
                telemetry_class="auth",
            )
        size = source.stat().st_size
        if size > MAX_CONVERSATION_INIT_SOURCE_BYTES:
            raise ChatGPTConversationInitError(
                "ChatGPT conversation-init source exceeds the sanitized size limit.",
                telemetry_class="malformed_telemetry",
            )
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except ChatGPTConversationInitError:
        raise
    except json.JSONDecodeError as exc:
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init source returned invalid JSON.",
            telemetry_class="malformed_telemetry",
        ) from exc
    except OSError as exc:
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init source is missing or unreadable.",
            telemetry_class="auth",
        ) from exc


def _parse_retry_after_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = float(value)
    except (OverflowError, ValueError):
        return None
    if not isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _sanitize_native_capture(
    value: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(value, Mapping):
        return None, "native_capture_incomplete"
    if not _is_canonical_account_hash(value.get("account_hash")):
        return None, "native_capture_invalid_account_hash"
    if (
        value.get("identity_source")
        != CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
    ):
        return None, "native_capture_invalid_identity_source"
    if (
        value.get("selector_evidence")
        != CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_SELECTOR_EVIDENCE
    ):
        return None, "native_capture_incomplete_selector_evidence"
    if value.get("request_response_correlated") is not True:
        return None, "native_capture_uncorrelated"
    if value.get("request_method") != CHATGPT_CONVERSATION_INIT_METHOD:
        return None, "native_capture_invalid_request_method"
    if not isinstance(value.get("request_body_omitted"), bool):
        return None, "native_capture_invalid_body_observation"
    if "browser_challenge" in value and not isinstance(
        value.get("browser_challenge"), bool
    ):
        return None, "native_capture_invalid_browser_challenge"

    retry_after_seconds = None
    if "retry_after_seconds" in value and value.get("retry_after_seconds") is not None:
        retry_after_seconds = _parse_retry_after_seconds(
            value.get("retry_after_seconds")
        )
        if retry_after_seconds is None:
            return None, "invalid_retry_after_seconds"

    result: Dict[str, Any] = {
        "account_hash": value["account_hash"],
        "identity_source": CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE,
        "selector_evidence": (
            CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_SELECTOR_EVIDENCE
        ),
        "request_response_correlated": True,
        "request_method": CHATGPT_CONVERSATION_INIT_METHOD,
        "request_body_omitted": value["request_body_omitted"],
    }
    if "browser_challenge" in value:
        result["browser_challenge"] = value["browser_challenge"]
    if retry_after_seconds is not None:
        result["retry_after_seconds"] = retry_after_seconds
    return result, None


def _native_payload_identity_error(
    raw: Any,
    payload: Any,
    native_account_hash: str,
) -> Optional[str]:
    account_id, _identity_fields, identity_error = (
        _extract_canonical_account_identity(raw, payload)
    )
    if identity_error not in (None, "missing_authoritative_account_id"):
        return "conflicting_native_payload_account_id"
    if account_id is None:
        return None
    payload_account_hash = hash_chatgpt_conversation_init_canonical_account_id(
        account_id
    )
    if payload_account_hash != native_account_hash:
        return "native_payload_identity_mismatch"
    return None


def sanitize_conversation_init_boundary(  # noqa: PLR0915 - boundary projection
    raw: Any,
    *,
    source_path: Optional[str] = None,
    request_url: Optional[str] = None,
    _allow_verified_envelope_identity: bool = False,
) -> Dict[str, Any]:
    """Strip credentials/raw headers and project a persistence-safe snapshot."""

    contract = conversation_init_request_contract(request_url)
    source_identity_hash = (
        hash_chatgpt_conversation_init_source_identity(source_path)
        if isinstance(source_path, str) and source_path.strip()
        else None
    )
    if not isinstance(raw, Mapping):
        return {
            "request": contract,
            "status_code": None,
            "payload": None,
            "payload_schema": {},
            "payload_state": "malformed",
            "account_hash": None,
            "account_identity_fields": [],
            "account_identity_source": None,
            "account_identity_verified": False,
            "account_identity_hash_algorithm": None,
            "account_identity_hash_length": None,
            "account_identity_verification_source": None,
            "source_identity_hash": source_identity_hash,
            "redacted_field_count": 0,
            "request_body_omitted": contract["body_omitted"],
            "retry_after_seconds": None,
            "browser_challenge": False,
            "malformed_collection_projection": False,
        }

    status_code, payload_raw, envelope_redacted = _split_boundary_envelope(raw)
    retained_native_capture_error = raw.get("native_capture_error")
    native_capture_field_present = "native_capture" in raw
    native_capture_present = native_capture_field_present or (
        retained_native_capture_error in _NATIVE_CAPTURE_ERRORS
    )
    raw_native_capture = raw.get("native_capture")
    raw_browser_challenge = raw.get("browser_challenge")
    browser_challenge = (
        raw_browser_challenge if isinstance(raw_browser_challenge, bool) else False
    )
    native_capture, native_capture_error = (
        _sanitize_native_capture(raw.get("native_capture"))
        if native_capture_field_present
        else (None, None)
    )
    if native_capture is not None and native_capture_error is None:
        native_capture_error = _native_payload_identity_error(
            raw,
            payload_raw,
            native_capture["account_hash"],
        )
    if (
        native_capture_error is None
        and retained_native_capture_error in _NATIVE_CAPTURE_ERRORS
    ):
        native_capture_error = retained_native_capture_error
    retry_after_seconds = (
        native_capture.get("retry_after_seconds")
        if isinstance(native_capture, Mapping)
        else None
    )
    request_body_omitted = (
        native_capture.get("request_body_omitted")
        if isinstance(native_capture, Mapping)
        else None if native_capture_present else contract["body_omitted"]
    )
    if isinstance(raw_native_capture, Mapping):
        raw_body_omitted = raw_native_capture.get("request_body_omitted")
        if isinstance(raw_body_omitted, bool):
            request_body_omitted = raw_body_omitted
        raw_browser_challenge = raw_native_capture.get("browser_challenge")
        if isinstance(raw_browser_challenge, bool):
            browser_challenge = browser_challenge or raw_browser_challenge
        raw_retry_after = raw_native_capture.get("retry_after_seconds")
        if raw_retry_after is not None:
            parsed_retry_after = _parse_retry_after_seconds(raw_retry_after)
            if parsed_retry_after is None:
                if native_capture_error is None:
                    native_capture_error = "invalid_retry_after_seconds"
            else:
                retry_after_seconds = parsed_retry_after
    if isinstance(native_capture, Mapping) and isinstance(
        native_capture.get("browser_challenge"), bool
    ):
        browser_challenge = browser_challenge or native_capture["browser_challenge"]
    if native_capture is None and isinstance(raw_native_capture, Mapping):
        safe_native_capture: Dict[str, Any] = {}
        if isinstance(raw_native_capture.get("request_body_omitted"), bool):
            safe_native_capture["request_body_omitted"] = raw_native_capture[
                "request_body_omitted"
            ]
        if isinstance(raw_native_capture.get("browser_challenge"), bool):
            safe_native_capture["browser_challenge"] = raw_native_capture[
                "browser_challenge"
            ]
        if retry_after_seconds is not None:
            safe_native_capture["retry_after_seconds"] = retry_after_seconds
        native_capture = safe_native_capture or None
    account_identity_verified = False
    account_identity_hash_algorithm = None
    account_identity_hash_length = None
    retained_bound_identity = (
        _retained_bound_envelope_identity(raw)
        if _allow_verified_envelope_identity
        else (None, [])
    )
    if retained_bound_identity[0]:
        account_hash, account_identity_fields = retained_bound_identity
        account_identity_verified = True
        account_identity_hash_algorithm = CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM
        account_identity_hash_length = CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
    elif native_capture_present:
        account_hash, account_identity_fields = (
            (native_capture["account_hash"], ["native_capture.account_hash"])
            if native_capture is not None and native_capture_error is None
            else (None, [])
        )
    else:
        account_hash, account_identity_fields = (
            hash_chatgpt_conversation_init_account_identity(payload_raw)
            if isinstance(payload_raw, Mapping)
            else (None, [])
        )
    if not account_hash:
        if not native_capture_present:
            account_hash, account_identity_fields = _retained_envelope_identity(raw)
    if native_capture_error is not None:
        account_hash = None
        account_identity_fields = []
        account_identity_verified = False
        account_identity_hash_algorithm = None
        account_identity_hash_length = None
    if isinstance(payload_raw, Mapping):
        payload, payload_schema, value_redacted = _redact_mapping(
            payload_raw,
            depth=0,
        )
        payload_state = "present"
    elif payload_raw is None:
        payload, payload_schema, value_redacted = None, {}, 0
        payload_state = "absent"
    else:
        payload, payload_schema, value_redacted = None, {}, 1
        payload_state = "malformed"

    malformed_collection_projection = (
        _schema_collection_projection_has_dropped_entries(payload_schema)
        if isinstance(payload_raw, Mapping)
        else False
    )
    if _allow_verified_envelope_identity and raw.get(
        "malformed_collection_projection"
    ) is True:
        malformed_collection_projection = True

    account_identity_source = _resolve_account_identity_source(
        raw,
        account_hash=account_hash,
        source_identity_hash=source_identity_hash,
    )

    result = {
        "request": contract,
        "status_code": status_code,
        "payload": payload,
        "payload_schema": payload_schema,
        "payload_state": payload_state,
        "projection_truncated": (
            (
                isinstance(payload_raw, Mapping)
                and (
                    len(payload_raw) > MAX_PROJECTION_OBJECT_KEYS
                    or _schema_projection_truncated(payload_schema)
                )
            )
            or (
                _allow_verified_envelope_identity
                and raw.get("projection_truncated") is True
            )
        ),
        "malformed_collection_projection": malformed_collection_projection,
        "account_hash": account_hash,
        "account_identity_fields": account_identity_fields,
        "account_identity_source": account_identity_source,
        "account_identity_verified": account_identity_verified,
        "account_identity_hash_algorithm": account_identity_hash_algorithm,
        "account_identity_hash_length": account_identity_hash_length,
        "account_identity_verification_source": (
            (
                CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
                if account_identity_source
                == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
                else CHATGPT_CONVERSATION_INIT_BROWSER_ACCOUNT_IDENTITY_SOURCE
            )
            if account_identity_verified
            else None
        ),
        "source_identity_hash": source_identity_hash,
        "redacted_field_count": envelope_redacted + value_redacted,
        "request_body_omitted": request_body_omitted,
        "retry_after_seconds": retry_after_seconds,
        "browser_challenge": browser_challenge,
    }
    if native_capture is not None:
        result["native_capture"] = native_capture
    if native_capture_error is not None:
        result["native_capture_error"] = native_capture_error
    return result


def _is_verified_bound_identity(
    sanitized: Mapping[str, Any],
    account_hash: Any,
) -> bool:
    if (
        sanitized.get("account_identity_verified") is not True
        or sanitized.get("native_capture_error") is not None
        or sanitized.get("browser_challenge") is True
        or not _is_canonical_account_hash(account_hash)
        or sanitized.get("account_identity_hash_algorithm")
        != CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM
        or sanitized.get("account_identity_hash_length")
        != CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
    ):
        return False
    source = sanitized.get("account_identity_source")
    if source == CHATGPT_CONVERSATION_INIT_VERIFIED_PAYLOAD_IDENTITY_SOURCE:
        return (
            sanitized.get("account_identity_verification_source")
            == CHATGPT_CONVERSATION_INIT_BROWSER_ACCOUNT_IDENTITY_SOURCE
        )
    if source != CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE:
        return False
    native_capture, native_capture_error = _sanitize_native_capture(
        sanitized.get("native_capture")
    )
    return bool(
        native_capture_error is None
        and native_capture is not None
        and native_capture.get("account_hash") == account_hash
        and sanitized.get("account_identity_verification_source")
        == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
    )


def _resolve_parse_guard(  # noqa: PLR0915 - parser guard ordering
    sanitized: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve identity, payload, and early-return guard for parsing."""

    request = dict(sanitized.get("request") or conversation_init_request_contract())
    status_code = sanitized.get("status_code")
    payload = sanitized.get("payload")
    payload_schema = (
        dict(sanitized.get("payload_schema") or {})
        if isinstance(sanitized.get("payload_schema"), Mapping)
        else {}
    )
    account_hash = sanitized.get("account_hash")
    source_identity_hash = sanitized.get("source_identity_hash")
    account_identity_fields = list(sanitized.get("account_identity_fields") or [])
    account_identity_source = sanitized.get("account_identity_source")
    native_capture_error = sanitized.get("native_capture_error")
    account_identity_verified = _is_verified_bound_identity(
        sanitized,
        account_hash,
    )
    if (
        not account_hash
        and native_capture_error is None
        and isinstance(source_identity_hash, str)
        and source_identity_hash
    ):
        account_hash = source_identity_hash
        if not account_identity_fields:
            account_identity_fields = [
                CHATGPT_CONVERSATION_INIT_SOURCE_IDENTITY_SOURCE
            ]
        account_identity_source = account_identity_source or "source_path"
    request_body_omitted = sanitized.get("request_body_omitted")
    if not isinstance(request_body_omitted, bool):
        request_body_omitted = (
            None
            if native_capture_error is not None
            or account_identity_source
            == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
            else bool(request.get("body_omitted"))
        )
    retry_after_seconds = sanitized.get("retry_after_seconds")
    browser_challenge = sanitized.get("browser_challenge") is True

    summary: Dict[str, Any] = {
        "source_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
        "request_method": request.get("method"),
        "request_path": request.get("path"),
        "request_body_omitted": request_body_omitted,
        "has_model_message": False,
        "has_conversation_content": False,
        "status_code": status_code,
        "account_identity_hashed": account_hash is not None,
        "account_hash": account_hash,
        "account_identity_verified": account_identity_verified,
        "account_identity_hash_algorithm": (
            CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM
            if account_identity_verified
            else None
        ),
        "account_identity_hash_length": (
            CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
            if account_identity_verified
            else None
        ),
        "account_identity_verification_source": (
            sanitized.get("account_identity_verification_source")
            if account_identity_verified
            else None
        ),
        "account_identity_fields": account_identity_fields,
        "account_identity_source": account_identity_source,
        "native_capture": (
            dict(sanitized["native_capture"])
            if isinstance(sanitized.get("native_capture"), Mapping)
            else None
        ),
        "native_capture_error": native_capture_error,
        "browser_challenge": browser_challenge,
        "retry_after_seconds": retry_after_seconds,
        "model_limits_state": "absent_unknown",
        "limits_progress_state": "absent_unknown",
        "blocked_features_state": "absent_unknown",
        "discovered_feature_identities": [],
        "discovered_model_identities": [],
        "discovered_blocked_identities": [],
        "malformed_entry_count": 0,
        "valid_observation_count": 0,
        "projection_truncated": sanitized.get("projection_truncated") is True,
        "malformed_collection_projection": (
            sanitized.get("malformed_collection_projection") is True
        ),
        "telemetry_status": "valid",
        "last_good_state_retained": False,
    }

    if browser_challenge:
        summary["telemetry_status"] = "auth"
        summary["telemetry_class"] = "browser_challenge"
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}
    if native_capture_error:
        summary["telemetry_status"] = "malformed"
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}
    http_error = _http_status_failure(status_code)
    if http_error is not None:
        summary["telemetry_status"] = http_error
        summary["telemetry_class"] = (
            "auth" if http_error == "auth" else "http_error"
        )
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}
    if (
        account_identity_source
        == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
        and not account_identity_verified
    ):
        summary["telemetry_status"] = "auth"
        summary["telemetry_class"] = "auth"
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}
    if sanitized.get("payload_state") != "present" or not isinstance(payload, Mapping):
        summary["telemetry_status"] = "malformed"
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}
    if not looks_like_conversation_init_payload(payload):
        summary["telemetry_status"] = "malformed"
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}
    if not account_hash:
        summary["telemetry_status"] = "missing_account_identity"
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return {"error": True, "summary": summary}

    return {
        "error": False,
        "summary": summary,
        "payload": payload,
        "payload_schema": payload_schema,
        "request": request,
        "account_hash": account_hash,
        "account_identity_verified": account_identity_verified,
        "account_identity_fields": account_identity_fields,
        "account_identity_source": account_identity_source,
    }


def parse_conversation_init_observations(
    sanitized: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a sanitized conversation-init snapshot into observation dicts."""

    guard = _resolve_parse_guard(sanitized)
    if guard["error"]:
        return [], guard["summary"]

    summary = guard["summary"]
    payload = guard["payload"]
    payload_schema = guard["payload_schema"]
    request = guard["request"]
    account_identity_fields = guard["account_identity_fields"]
    account_identity_source = guard["account_identity_source"]

    parsed = _parse_collections(payload, summary)
    summary["model_limits_state"] = parsed["model_limits_state"]
    summary["limits_progress_state"] = parsed["limits_progress_state"]
    summary["blocked_features_state"] = parsed["blocked_features_state"]
    summary["discovered_feature_identities"] = parsed["feature_ids"]
    summary["discovered_model_identities"] = parsed["model_ids"] + [
        item for item in parsed["default_model_ids"] if item not in parsed["model_ids"]
    ]
    summary["discovered_blocked_identities"] = parsed["blocked_ids"]
    summary["malformed_entry_count"] = parsed["malformed"]
    summary["projection_truncated"] = parsed["truncated"]
    if _collections_are_wholly_malformed(parsed):
        summary["telemetry_status"] = "malformed"
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return [], summary
    if parsed["truncated"]:
        summary["telemetry_status"] = "partial"
    elif parsed["malformed"] and not parsed["model_ids"] and not parsed["feature_ids"]:
        summary["telemetry_status"] = "partial"
    elif parsed["malformed"]:
        summary["telemetry_status"] = "partial"

    snapshot = _snapshot_observation(
        payload=payload,
        payload_schema=payload_schema,
        summary=summary,
        request=request,
        observed_at=observed_at,
        account_identity_fields=account_identity_fields,
        account_identity_source=account_identity_source,
    )
    all_observations: List[Dict[str, Any]] = [snapshot]
    all_observations.extend(parsed["feature_rows"])
    all_observations.extend(parsed["model_rows"])
    all_observations.extend(parsed["default_model_rows"])
    all_observations.extend(parsed["blocked_rows"])
    summary["valid_observation_count"] = len(all_observations)
    summary["quota_keys"] = [row["quota_key"] for row in all_observations]
    return all_observations, summary



def _parse_collections(
    payload: Mapping[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract all collection observations from the payload.

    Returns a dict with rows, identities, state labels, and truncation
    flags so the caller can reassemble the final observation list and
    populate the summary without exceeding ruff's statement limit.
    """

    feature_rows, feature_ids, feature_state, feature_malformed = (
        _parse_limits_progress(payload)
    )
    model_rows, model_ids, model_state, model_malformed = _parse_model_limits(payload)
    blocked_rows, blocked_ids, blocked_state, blocked_malformed = (
        _parse_blocked_features(payload)
    )
    default_model_rows, default_model_ids = _parse_default_model_identities(
        payload, existing_ids=set(model_ids),
    )
    truncated = summary.get("projection_truncated") is True
    projection_malformed = (
        summary.get("malformed_collection_projection") is True
    )
    return {
        "feature_rows": feature_rows,
        "feature_ids": feature_ids,
        "model_rows": model_rows,
        "model_ids": model_ids,
        "blocked_rows": blocked_rows,
        "blocked_ids": blocked_ids,
        "default_model_rows": default_model_rows,
        "default_model_ids": default_model_ids,
        "model_limits_state": model_state,
        "limits_progress_state": feature_state,
        "blocked_features_state": blocked_state,
        "malformed": (
            feature_malformed
            + model_malformed
            + blocked_malformed
            + (1 if projection_malformed else 0)
        ),
        "projection_malformed": projection_malformed,
        "truncated": truncated,
    }


def _collections_are_wholly_malformed(parsed: Mapping[str, Any]) -> bool:
    return bool(parsed["malformed"]) and not any(
        parsed[key] for key in ("feature_rows", "model_rows", "blocked_rows")
    )


def _collection_diagnostics(
    payload: Any,
    *,
    malformed_collection_projection: bool,
    projection_truncated: bool,
) -> Dict[str, Any]:
    diagnostics = {
        "model_limits_state": "absent_unknown",
        "limits_progress_state": "absent_unknown",
        "blocked_features_state": "absent_unknown",
        "malformed_entry_count": 0,
        "valid_observation_count": 0,
        "projection_truncated": bool(projection_truncated),
        "malformed_collection_projection": bool(
            malformed_collection_projection
        ),
    }
    if not isinstance(payload, Mapping):
        return diagnostics

    parsed = _parse_collections(
        payload,
        {
            "malformed_collection_projection": (
                malformed_collection_projection
            ),
            "projection_truncated": projection_truncated,
        },
    )
    for field_name in (
        "model_limits_state",
        "limits_progress_state",
        "blocked_features_state",
    ):
        state = parsed[field_name]
        diagnostics[field_name] = (
            state if state in _COLLECTION_STATE_VALUES else "malformed"
        )
    diagnostics["malformed_entry_count"] = min(
        max(int(parsed["malformed"]), 0),
        _MAX_COLLECTION_DIAGNOSTIC_COUNT,
    )
    diagnostics["projection_truncated"] = bool(
        projection_truncated or parsed["truncated"] is True
    )
    diagnostics["malformed_collection_projection"] = (
        parsed["projection_malformed"] is True
    )
    if _collections_are_wholly_malformed(parsed):
        return diagnostics
    valid_rows = sum(
        len(parsed[field_name])
        for field_name in (
            "feature_rows",
            "model_rows",
            "default_model_rows",
            "blocked_rows",
        )
    )
    diagnostics["valid_observation_count"] = min(
        valid_rows + 1,
        _MAX_COLLECTION_DIAGNOSTIC_COUNT,
    )
    return diagnostics


def build_conversation_init_rate_limit_tuples(
    observations: Sequence[Mapping[str, Any]],
    *,
    observed_at: datetime,
    account_hash: str,
) -> List[Tuple[Any, ...]]:
    """Convert parsed observation dicts into rate_limit_observations tuples."""

    stamp = observed_at.strftime("%Y%m%d%H%M%S")
    payloads: List[Tuple[Any, ...]] = []
    for index, row in enumerate(observations):
        payloads.append(
            (
                observed_at,
                CHATGPT_CONVERSATION_INIT_CLIENT,
                None,
                account_hash,
                CHATGPT_CONVERSATION_INIT_PROVIDER,
                row.get("model") or CHATGPT_CONVERSATION_INIT_CLIENT,
                row["quota_key"],
                row.get("quota_period"),
                row.get("quota_type"),
                row.get("expected_reset_at"),
                row.get("remaining_pct"),
                row.get("quota_limit"),
                row.get("quota_used"),
                row.get("quota_remaining"),
                row.get("billing_period_start_at"),
                row.get("billing_period_end_at"),
                json.dumps(row.get("raw_provider_fields") or {}, sort_keys=True),
                json.dumps(row.get("evidence") or {}, sort_keys=True),
                CHATGPT_CONVERSATION_INIT_SOURCE,
                None,
                None,
                f"chatgpt-conversation-init-{stamp}-{index:02d}",
            )
        )
    return payloads


def collect_conversation_init_observations(
    source_path: str,
    *,
    observed_at: Optional[datetime] = None,
    request_url: Optional[str] = None,
) -> Tuple[List[Tuple[Any, ...]], Dict[str, Any]]:
    """Load a fixture-backed snapshot, sanitize it, and emit persist tuples.

    The sidecar never HTTP-calls chatgpt.com. Browser cookies stay on the
    Oracle boundary; this helper only rereads already-collected JSON.
    """

    observed = observed_at or datetime.now(timezone.utc)
    contract = conversation_init_request_contract(request_url)
    collector_request = build_conversation_init_request(contract["url"])
    if request_has_conversation_content(collector_request):
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init collector request must be POST with no body.",
            telemetry_class="malformed_telemetry",
        )
    raw = load_conversation_init_source(source_path)
    sanitized = sanitize_conversation_init_boundary(
        raw,
        source_path=source_path,
        request_url=request_url,
        _allow_verified_envelope_identity=True,
    )
    observations, summary = parse_conversation_init_observations(
        sanitized,
        observed_at=observed,
    )
    summary["request_method"] = contract["method"]
    summary["request_path"] = contract["path"]
    summary["request_url"] = contract["url"]
    summary["request_body_omitted"] = sanitized.get("request_body_omitted")
    summary["has_model_message"] = False
    summary["has_conversation_content"] = False
    summary["collector_source"] = "file"
    summary["native_capture"] = (
        dict(sanitized["native_capture"])
        if isinstance(sanitized.get("native_capture"), Mapping)
        else None
    )
    summary["native_capture_error"] = sanitized.get("native_capture_error")
    summary["browser_challenge"] = bool(sanitized.get("browser_challenge"))
    summary["retry_after_seconds"] = sanitized.get("retry_after_seconds")
    account_hash = (
        None
        if sanitized.get("native_capture_error")
        else sanitized.get("account_hash")
        or sanitized.get("source_identity_hash")
    )
    summary["account_hash"] = account_hash
    summary["account_identity_verified"] = bool(
        summary.get("account_identity_verified")
    )
    summary["account_identity_source"] = sanitized.get("account_identity_source")
    summary["account_identity_verification_source"] = sanitized.get(
        "account_identity_verification_source"
    )
    summary["account_identity_verification_error"] = sanitized.get(
        "account_identity_verification_error"
    )
    native_identity_verified = (
        summary.get("account_identity_source")
        != CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
        or summary.get("account_identity_verified") is True
    )
    if (
        observations
        and native_identity_verified
        and not summary["browser_challenge"]
        and isinstance(account_hash, str)
        and account_hash
    ):
        payloads = build_conversation_init_rate_limit_tuples(
            observations,
            observed_at=observed,
            account_hash=account_hash,
        )
    else:
        payloads = []
    summary["observation_count"] = len(payloads)
    if not payloads:
        summary["last_good_state_retained"] = True
    return payloads, summary


def _http_status_failure(status_code: Any) -> Optional[str]:
    if status_code is None:
        return None
    try:
        code = int(status_code)
    except (OverflowError, TypeError, ValueError):
        return None
    if 200 <= code < 300:
        return None
    if code in {401, 403}:
        return "auth"
    return "http_error"


def _split_boundary_envelope(
    raw: Mapping[str, Any],
) -> Tuple[Optional[int], Any, int]:
    redacted = 0
    lower_keys = {_normalize_key(key) for key in raw}
    has_envelope = bool(lower_keys & _ENVELOPE_KEYS) and not looks_like_conversation_init_payload(
        raw
    )
    status_code = _parse_status_code(raw)
    for key, value in raw.items():
        if _is_secret_key(str(key), value):
            redacted += 1
    if not has_envelope and looks_like_conversation_init_payload(raw):
        return status_code, raw, redacted
    payload = None
    for key in _PAYLOAD_ENVELOPE_KEYS:
        if key in raw:
            payload = raw.get(key)
            break
        for actual in raw:
            if _normalize_key(actual) == key:
                payload = raw.get(actual)
                break
        if payload is not None:
            break
    if payload is None and looks_like_conversation_init_payload(raw):
        payload = {
            key: value
            for key, value in raw.items()
            if _normalize_key(key) not in _ENVELOPE_KEYS
            and not _is_secret_key(str(key), value)
        }
    if has_envelope:
        for key in raw:
            if _normalize_key(key) in {"headers", "header", "cookies", "cookie"}:
                redacted += 1
    return status_code, payload, redacted


def _parse_status_code(raw: Mapping[str, Any]) -> Optional[int]:
    for key in ("status_code", "status", "statusCode"):
        if key in raw:
            value = raw[key]
            break
        value = None
    else:
        for actual, value in raw.items():
            if _normalize_key(actual) in {"status_code", "status", "statuscode"}:
                break
        else:
            return None
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _redact_mapping(
    mapping: Mapping[str, Any],
    *,
    depth: int,
    parent_key: Optional[str] = None,
    collection_entry: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    redacted_count = 0
    payload: Dict[str, Any] = {}
    schema: Dict[str, Any] = {}
    malformed_fields: List[str] = []
    items = list(mapping.items())[:MAX_PROJECTION_OBJECT_KEYS]
    parent_normalized = _normalize_key(parent_key) if parent_key else ""
    identity_container = parent_normalized in _ACCOUNT_IDENTITY_CONTAINERS
    collection_map = (
        parent_normalized in _NAMED_USAGE_COLLECTIONS
        and not collection_entry
    )
    for key, value in items:
        name = str(key)
        normalized = _normalize_key(name)
        if (
            _is_secret_key(name, value)
            or _is_pii_field_name(name)
            or normalized in _ACCOUNT_IDENTITY_KEYS
            or (identity_container and normalized == "id")
            or (identity_container and normalized == "name")
            or _is_unsafe_mapping_key(name)
        ):
            schema[_schema_key_for_redaction(name)] = {"kind": "redacted"}
            redacted_count += 1
            continue
        scalar_member = collection_map and not isinstance(value, (Mapping, list))
        child_parent_key = "remaining" if scalar_member else name
        malformed_usage_field = (
            _malformed_usage_field(child_parent_key, value)
            if not collection_map or scalar_member
            else None
        )
        if malformed_usage_field is not None:
            if scalar_member:
                marker = {_MALFORMED_FIELDS_KEY: [child_parent_key]}
                marker_schema = {
                    _MALFORMED_FIELDS_KEY: {
                        "kind": "array",
                        "length": 1,
                        "state": "present",
                        "item_kinds": ["slug"],
                        "projection": [child_parent_key],
                        "dropped_items": 0,
                        "truncated": False,
                    }
                }
                payload[name] = marker
                schema[name] = {
                    "kind": "object",
                    "keys": [_MALFORMED_FIELDS_KEY],
                    "schema_fingerprint": _schema_fingerprint(marker_schema),
                    "state": "present",
                    "projection": marker,
                    "projection_dropped": True,
                    "truncated": False,
                }
            else:
                payload[name] = None
                schema[name] = {"kind": "null", "malformed": True}
                malformed_fields.append(malformed_usage_field)
            redacted_count += 1
            continue
        redaction_value = value
        if collection_map and isinstance(value, Mapping):
            identity_field = _NAMED_COLLECTION_IDENTITY_FIELDS.get(
                parent_normalized
            )
            identity = _safe_identity(name)
            if (
                identity_field is not None
                and identity is not None
                and not any(
                    _normalize_key(existing_key)
                    == _normalize_key(identity_field)
                    for existing_key in value
                )
            ):
                redaction_value = dict(value)
                redaction_value[identity_field] = identity
        redacted_value, node, nested_redacted = _redact_value(
            redaction_value,
            depth=depth + 1,
            parent_key=child_parent_key,
            collection_entry=(
                parent_normalized in _NAMED_USAGE_COLLECTIONS
                and isinstance(value, Mapping)
            ),
        )
        redacted_count += nested_redacted
        schema[name] = node
        if node.get("kind") != "redacted":
            payload[name] = redacted_value
    if malformed_fields:
        marker = sorted(set(malformed_fields))
        payload[_MALFORMED_FIELDS_KEY] = marker
        schema[_MALFORMED_FIELDS_KEY] = {
            "kind": "array",
            "length": len(marker),
            "state": "present",
            "item_kinds": ["slug"],
            "projection": marker,
            "dropped_items": 0,
            "truncated": False,
        }
    return payload, schema, redacted_count


def _redact_value(
    value: Any,
    *,
    depth: int,
    parent_key: Optional[str] = None,
    collection_entry: bool = False,
) -> Tuple[Any, Dict[str, Any], int]:
    if depth > MAX_PROJECTION_DEPTH:
        return None, {"kind": "truncated"}, 0
    if value is None:
        return None, {"kind": "null"}, 0
    if isinstance(value, bool):
        return value, {"kind": "bool", "value": value}, 0
    if isinstance(value, int):
        finite_value = _finite_float(value)
        if finite_value is None:
            return None, {"kind": "redacted"}, 1
        return value, {"kind": "int", "value": value}, 0
    if isinstance(value, float):
        if _finite_float(value) is None:
            return None, {"kind": "redacted"}, 1
        return value, {"kind": "float", "value": value}, 0
    if isinstance(value, str):
        if _is_safe_usage_number_string(parent_key, value):
            stripped = value.strip()
            try:
                numeric_value = float(stripped)
            except (OverflowError, ValueError):
                numeric_value = None
            if numeric_value is not None and isfinite(numeric_value):
                kind = "float" if not numeric_value.is_integer() else "int"
                projected: Any = int(numeric_value) if kind == "int" else numeric_value
                return projected, {"kind": kind, "value": projected}, 0
        if not _is_safe_string_value(
            value,
            field_name=parent_key,
        ):
            return None, {"kind": "redacted"}, 1
        clipped = value if len(value) <= MAX_SAFE_STRING_LENGTH else value[:MAX_SAFE_STRING_LENGTH]
        kind = "slug" if _SLUG_RE.match(clipped) else "string"
        return clipped, {"kind": kind, "value": clipped}, 0
    if isinstance(value, Mapping):
        nested_payload, nested_schema, redacted = _redact_mapping(
            value,
            depth=depth,
            parent_key=parent_key,
            collection_entry=collection_entry,
        )
        fingerprint = _schema_fingerprint(nested_schema)
        object_node: Dict[str, Any] = {
            "kind": "object",
            "keys": sorted(nested_schema),
            "schema_fingerprint": fingerprint,
            "state": "present" if nested_schema else "empty_unknown",
            "projection": nested_payload,
            "projection_dropped": (
                len(value) <= MAX_PROJECTION_OBJECT_KEYS
                and len(nested_payload) < len(value)
            ),
            "truncated": (
                len(value) > MAX_PROJECTION_OBJECT_KEYS
                or _schema_projection_truncated(nested_schema)
            ),
        }
        return nested_payload, object_node, redacted
    if isinstance(value, list):
        items: List[Any] = []
        item_kinds: List[str] = []
        redacted = 0
        dropped_items = 0
        nested_truncated = False
        bounded = value[:MAX_PROJECTION_LIST_ITEMS]
        for item in bounded:
            nested_value, item_node, nested_redacted = _redact_value(
                item,
                depth=depth + 1,
                parent_key=parent_key,
                collection_entry=(
                    _normalize_key(parent_key) in _NAMED_USAGE_COLLECTIONS
                    and isinstance(item, Mapping)
                ),
            )
            redacted += nested_redacted
            item_kinds.append(str(item_node.get("kind") or "unknown"))
            nested_truncated = nested_truncated or (
                item_node.get("kind") == "truncated"
                or item_node.get("truncated") is True
            )
            if item_node.get("kind") != "redacted":
                items.append(nested_value)
            else:
                dropped_items += 1
        state = "empty_unknown" if not value else "present"
        truncated = len(value) > MAX_PROJECTION_LIST_ITEMS
        if truncated:
            state = _TRUNCATION_WARNING
        list_node: Dict[str, Any] = {
            "kind": "array",
            "length": len(value),
            "state": state,
            "item_kinds": sorted(set(item_kinds)),
            "projection": items,
            "dropped_items": dropped_items,
            "truncated": truncated or nested_truncated,
        }
        return items, list_node, redacted
    return None, {"kind": "redacted"}, 1


def _schema_projection_truncated(schema: Mapping[str, Any]) -> bool:
    return any(
        isinstance(node, Mapping)
        and (node.get("kind") == "truncated" or node.get("truncated") is True)
        for node in schema.values()
    )


def _schema_collection_projection_has_dropped_entries(
    schema: Mapping[str, Any],
) -> bool:
    for collection_name in (
        "model_limits",
        "limits_progress",
        "blocked_features",
    ):
        node = schema.get(collection_name)
        if not isinstance(node, Mapping):
            continue
        if node.get("kind") == "array" and node.get("dropped_items", 0):
            return True
        if node.get("kind") == "object" and node.get("projection_dropped") is True:
            return True
    return False


def _schema_fingerprint(schema: Mapping[str, Any]) -> str:
    parts = []
    for key in sorted(schema):
        kind = schema[key].get("kind") if isinstance(schema[key], Mapping) else "unknown"
        parts.append(f"{key}:{kind}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _snapshot_observation(
    *,
    payload: Mapping[str, Any],
    payload_schema: Mapping[str, Any],
    summary: Mapping[str, Any],
    request: Mapping[str, Any],
    observed_at: datetime,
    account_identity_fields: Sequence[str],
    account_identity_source: Optional[str],
) -> Dict[str, Any]:
    top_level_keys = sorted(str(key) for key in payload)
    unknown_keys = [
        key
        for key in top_level_keys
        if key
        not in {
            "model_limits",
            "limits_progress",
            "blocked_features",
            "default_model_slug",
            "intended_default_model_slug",
        }
    ]
    raw_provider_fields = {
        "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
        "collection_state": {
            "model_limits": summary["model_limits_state"],
            "limits_progress": summary["limits_progress_state"],
            "blocked_features": summary["blocked_features_state"],
        },
        "top_level_keys": top_level_keys,
        "unknown_top_level_keys": unknown_keys,
        "schema_fingerprint": _schema_fingerprint(payload_schema),
        "top_level_projections": dict(payload_schema),
        "empty_collections_are_unknown": True,
        "request_method": request.get("method"),
        "request_body_omitted": summary["request_body_omitted"],
        "account_identity_source": account_identity_source,
        "native_capture": (
            dict(summary["native_capture"])
            if isinstance(summary.get("native_capture"), Mapping)
            else None
        ),
        "browser_challenge": bool(summary.get("browser_challenge")),
        "retry_after_seconds": summary.get("retry_after_seconds"),
    }
    evidence = {
        "signals": ["chatgpt_conversation_init", "chatgpt_conversation_init_snapshot"],
        "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
        "telemetry_status": summary["telemetry_status"],
        "account_identity_fields": list(account_identity_fields),
        "account_identity_source": account_identity_source,
        "account_identity_verified": bool(
            summary.get("account_identity_verified")
        ),
        "account_identity_verification_source": summary.get(
            "account_identity_verification_source"
        ),
        "native_capture": (
            dict(summary["native_capture"])
            if isinstance(summary.get("native_capture"), Mapping)
            else None
        ),
        "browser_challenge": bool(summary.get("browser_challenge")),
        "retry_after_seconds": summary.get("retry_after_seconds"),
        "account_hash": (
            summary.get("account_hash")
            if summary.get("account_identity_verified")
            else None
        ),
        "request_method": request.get("method"),
        "request_body_omitted": summary["request_body_omitted"],
        "has_model_message": False,
        "empty_collections_are_unknown": True,
    }
    default_model = _safe_identity(payload.get("default_model_slug"))
    result: Dict[str, Any] = {
        "quota_key": CHATGPT_CONVERSATION_INIT_SNAPSHOT_QUOTA_KEY,
        "quota_period": None,
        "quota_type": "schema",
        "model": default_model or CHATGPT_CONVERSATION_INIT_CLIENT,
        "expected_reset_at": None,
        "remaining_pct": None,
        "quota_limit": None,
        "quota_used": None,
        "quota_remaining": None,
        "billing_period_start_at": None,
        "billing_period_end_at": None,
        "raw_provider_fields": raw_provider_fields,
        "evidence": evidence,
    }
    if summary.get("projection_truncated"):
        result["raw_provider_fields"]["projection_truncated"] = True
    if summary.get("malformed_collection_projection"):
        result["raw_provider_fields"]["malformed_collection_projection"] = True
    return result


def _parse_limits_progress(
    payload: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], str, int]:
    if "limits_progress" not in payload:
        return [], [], "absent_unknown", 0
    raw = payload.get("limits_progress")
    entries, malformed = _normalize_named_collection(raw)
    if isinstance(raw, list) and not raw:
        return [], [], "empty_unknown", malformed
    if isinstance(raw, Mapping) and not raw:
        return [], [], "empty_unknown", malformed
    if not isinstance(raw, (list, Mapping)):
        return [], [], "malformed", malformed + 1
    rows: List[Dict[str, Any]] = []
    identities: List[str] = []
    for entry in entries:
        if _entry_has_malformed_usage_fields(entry):
            malformed += 1
            continue
        identity = _entry_identity(entry)
        if identity is None:
            malformed += 1
            continue
        parsed = _parse_usage_entry(
            entry,
            identity=identity,
            quota_key_kind="feature",
            signals=["chatgpt_conversation_init", "chatgpt_conversation_init_feature"],
        )
        if parsed is None:
            malformed += 1
            continue
        rows.append(parsed)
        identities.append(identity)
    state = "present" if identities else ("malformed" if malformed else "empty_unknown")
    if identities and malformed:
        state = "partial"
    return rows, identities, state, malformed


def _parse_model_limits(
    payload: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], str, int]:
    if "model_limits" not in payload:
        return [], [], "absent_unknown", 0
    raw = payload.get("model_limits")
    entries, malformed = _normalize_named_collection(raw)
    if isinstance(raw, list) and not raw:
        return [], [], "empty_unknown", malformed
    if isinstance(raw, Mapping) and not raw:
        return [], [], "empty_unknown", malformed
    if not isinstance(raw, (list, Mapping)):
        return [], [], "malformed", malformed + 1
    rows: List[Dict[str, Any]] = []
    identities: List[str] = []
    for entry in entries:
        if _entry_has_malformed_usage_fields(entry):
            malformed += 1
            continue
        identity = _entry_identity(entry) or _safe_identity(entry.get("slug"))
        if identity is None:
            malformed += 1
            continue
        parsed = _parse_usage_entry(
            entry,
            identity=identity,
            quota_key_kind="model",
            signals=["chatgpt_conversation_init", "chatgpt_conversation_init_model"],
        )
        if parsed is None:
            malformed += 1
            continue
        rows.append(parsed)
        identities.append(identity)
    state = "present" if identities else ("malformed" if malformed else "empty_unknown")
    if identities and malformed:
        state = "partial"
    return rows, identities, state, malformed


def _parse_blocked_features(
    payload: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str], str, int]:
    if "blocked_features" not in payload:
        return [], [], "absent_unknown", 0
    raw = payload.get("blocked_features")
    if raw is None:
        return [], [], "absent_unknown", 0
    if isinstance(raw, list) and not raw:
        return [], [], "empty_unknown", 0
    if not isinstance(raw, (list, Mapping)):
        return [], [], "malformed", 1
    rows: List[Dict[str, Any]] = []
    identities: List[str] = []
    malformed = 0
    items: Iterable[Any]
    if isinstance(raw, Mapping):
        items = [
            {"_identity": str(key), **(value if isinstance(value, Mapping) else {"blocked": value})}
            for key, value in raw.items()
        ]
    else:
        items = raw
    for item in items:
        if isinstance(item, str):
            identity = _safe_identity(item)
            entry: Mapping[str, Any] = {"blocked": True}
            blocked_state: Optional[bool] = True
        elif isinstance(item, Mapping):
            identity = _entry_identity(item) or _safe_identity(item.get("_identity"))
            entry = item
            blocked_value = entry.get("blocked")
            blocked_state = blocked_value if isinstance(blocked_value, bool) else None
        else:
            malformed += 1
            continue
        if identity is None:
            malformed += 1
            continue
        if blocked_state is None:
            continue
        rows.append(
            {
                "quota_key": _quota_key("blocked", identity),
                "quota_period": None,
                "quota_type": "blocked",
                "model": identity,
                "expected_reset_at": None,
                "remaining_pct": None,
                "quota_limit": None,
                "quota_used": None,
                "quota_remaining": None,
                "billing_period_start_at": None,
                "billing_period_end_at": None,
                "raw_provider_fields": {
                    "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
                    "identity": identity,
                    "blocked": blocked_state,
                    "collection": "blocked_features",
                    "entry_projections": _entry_projections(entry),
                },
                "evidence": {
                    "signals": [
                        "chatgpt_conversation_init",
                        "chatgpt_conversation_init_blocked_feature",
                    ],
                    "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
                    "identity": identity,
                    "blocked": blocked_state,
                },
            }
        )
        identities.append(identity)
    state = "present" if identities else ("malformed" if malformed else "empty_unknown")
    if identities and malformed:
        state = "partial"
    return rows, identities, state, malformed


def _parse_default_model_identities(
    payload: Mapping[str, Any],
    *,
    existing_ids: Iterable[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    existing = set(existing_ids)
    rows: List[Dict[str, Any]] = []
    identities: List[str] = []
    for field_name in ("default_model_slug", "intended_default_model_slug"):
        identity = _safe_identity(payload.get(field_name))
        if identity is None or identity in existing or identity in identities:
            continue
        rows.append(
            {
                "quota_key": _quota_key("model", identity),
                "quota_period": None,
                "quota_type": "unknown",
                "model": identity,
                "expected_reset_at": None,
                "remaining_pct": None,
                "quota_limit": None,
                "quota_used": None,
                "quota_remaining": None,
                "billing_period_start_at": None,
                "billing_period_end_at": None,
                "raw_provider_fields": {
                    "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
                    "identity": identity,
                    "identity_source": field_name,
                    "collection_state": "unknown",
                    "empty_collections_are_unknown": True,
                },
                "evidence": {
                    "signals": [
                        "chatgpt_conversation_init",
                        "chatgpt_conversation_init_model_identity",
                    ],
                    "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
                    "identity": identity,
                    "identity_source": field_name,
                    "quota_unknown": True,
                },
            }
        )
        identities.append(identity)
    return rows, identities


def _parse_usage_entry(
    entry: Mapping[str, Any],
    *,
    identity: str,
    quota_key_kind: str,
    signals: Sequence[str],
) -> Optional[Dict[str, Any]]:
    remaining = _first_number(entry, _REMAINING_KEYS)
    limit = _first_number(entry, _LIMIT_KEYS)
    used = _first_number(entry, _USED_KEYS)
    reset_at = _first_timestamp(entry, _RESET_KEYS)
    remaining_malformed = _field_malformed(entry, _REMAINING_KEYS, remaining)
    limit_malformed = _field_malformed(entry, _LIMIT_KEYS, limit)
    used_malformed = _field_malformed(entry, _USED_KEYS, used)
    if remaining_malformed or limit_malformed or used_malformed:
        return None
    remaining_pct = None
    if limit is not None and limit > 0 and remaining is not None:
        remaining_pct = max(0.0, min(100.0, remaining / limit * 100.0))
    elif limit is not None and limit > 0 and used is not None:
        remaining_pct = max(0.0, min(100.0, 100.0 - (used / limit * 100.0)))
    quota_type = "count" if remaining is not None or limit is not None or used is not None else "unknown"
    return {
        "quota_key": _quota_key(quota_key_kind, identity),
        "quota_period": None,
        "quota_type": quota_type,
        "model": identity,
        "expected_reset_at": reset_at,
        "remaining_pct": remaining_pct,
        "quota_limit": limit,
        "quota_used": used,
        "quota_remaining": remaining,
        "billing_period_start_at": None,
        "billing_period_end_at": reset_at,
        "raw_provider_fields": {
            "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
            "identity": identity,
            "collection": (
                "limits_progress" if quota_key_kind == "feature" else "model_limits"
            ),
            "empty_collections_are_unknown": True,
            "entry_projections": _entry_projections(entry),
        },
        "evidence": {
            "signals": list(signals),
            "parser_version": CHATGPT_CONVERSATION_INIT_PARSER_VERSION,
            "identity": identity,
            "quota_unknown": remaining is None and limit is None and used is None,
        },
    }


def _entry_has_malformed_usage_fields(entry: Mapping[str, Any]) -> bool:
    marker = entry.get(_MALFORMED_FIELDS_KEY)
    if not isinstance(marker, list):
        return False
    return any(
        isinstance(field_name, str)
        and _is_usage_number_field_name(field_name)
        for field_name in marker
    )


def _normalize_named_collection(
    raw: Any,
) -> Tuple[List[Mapping[str, Any]], int]:
    malformed = 0
    entries: List[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                merged = dict(value)
                merged.setdefault("_identity", str(key))
                entries.append(merged)
            elif isinstance(value, (int, float, bool, str)) or value is None:
                entries.append({"_identity": str(key), "remaining": value})
            else:
                malformed += 1
        return entries, malformed
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                if len(item) == 1 and _entry_identity(item) is None:
                    key, value = next(iter(item.items()))
                    normalized_key = _normalize_key(key)
                    if normalized_key in {
                        _normalize_key("_identity"),
                        *(_normalize_key(field) for field in _IDENTITY_FIELD_NAMES),
                    }:
                        entries.append(item)
                        continue
                    if _is_usage_number_field_name(normalized_key) or any(
                        normalized_key == _normalize_key(reset_key)
                        for reset_key in _RESET_KEYS
                    ):
                        entries.append(item)
                        continue
                    identity = _safe_identity(key)
                    if identity is not None and isinstance(value, Mapping):
                        merged = dict(value)
                        merged.setdefault("_identity", identity)
                        entries.append(merged)
                        continue
                    if identity is not None and (
                        isinstance(value, (int, float, str)) or value is None
                    ):
                        entries.append(
                            {"_identity": identity, "remaining": value}
                        )
                        continue
                entries.append(item)
            elif isinstance(item, str):
                entries.append({"_identity": item})
            else:
                malformed += 1
        return entries, malformed
    return [], 1


def _entry_identity(entry: Mapping[str, Any]) -> Optional[str]:
    explicit = entry.get("_identity")
    identity = _safe_identity(explicit)
    if identity is not None:
        return identity
    for field_name in _IDENTITY_FIELD_NAMES:
        identity = _safe_identity(entry.get(field_name))
        if identity is not None:
            return identity
    return None



def _is_pii_field_value(value: str) -> bool:
    """Return True when a string value is likely personal/account content.

    Heuristic: the value contains whitespace-delimited tokens (a phrase,
    sentence, or multi-word name) rather than a single slug/identifier/key.
    """

    if _is_unsafe_string(value):
        return True
    return len(value.split()) > 1


def _is_timestamp_value(value: str) -> bool:
    """Return True when a string value is likely a timestamp that varies per poll."""
    text = value.strip()
    if not text:
        return False
    if text.endswith('Z') or ('+' in text and ':' in text.rsplit('+', 1)[-1]):
        try:
            from datetime import datetime as _dt
            normalized = text
            if normalized.endswith('Z'):
                normalized = normalized[:-1] + '+00:00'
            _dt.fromisoformat(normalized)
            return True
        except (ValueError, TypeError):
            pass
    if text.replace('-', '').replace(':', '').replace('T', '').replace(' ', '').isdigit() and len(text) >= 10:
        return True
    return False


def _is_pii_field_name(name: str) -> bool:
    """Return True when a field name is likely to carry personal/account content."""
    normalized = _normalize_key(name)
    if not normalized:
        return False
    if normalized in _ACCOUNT_IDENTITY_KEYS:
        return True
    if normalized in _PII_FIELD_NAMES:
        return True
    return False


def _is_unsafe_mapping_key(name: str) -> bool:
    """Reject dynamic keys that can carry personal data before persistence."""
    text = str(name)
    return (
        len(text) > MAX_SAFE_STRING_LENGTH
        or any(character.isspace() for character in text)
        or _is_unsafe_string(text)
    )


def _schema_key_for_redaction(_name: str) -> str:
    """Avoid persisting arbitrary sensitive key names in the schema."""
    return "redacted_field"


def _is_safe_telemetry_string(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > MAX_SAFE_STRING_LENGTH:
        return False
    if any(ord(character) < 32 for character in text):
        return False
    if _is_unsafe_string(text):
        return False
    return bool(_SLUG_RE.match(text) or _is_timestamp_value(text))


def _is_safe_string_value(
    value: str,
    *,
    field_name: Optional[str],
) -> bool:
    """Allow strings only for explicitly safe telemetry field semantics."""
    normalized = _normalize_key(field_name) if field_name else ""
    identity_collection = normalized in {
        "model_limits", "limits_progress", "blocked_features",
    }
    reset_field = normalized in {_normalize_key(key) for key in _RESET_KEYS}
    return (
        normalized in _SAFE_STRING_FIELD_NAMES or identity_collection or reset_field
    ) and _is_safe_telemetry_string(value)


def _entry_projections(entry: Mapping[str, Any]) -> Dict[str, Any]:
    projections: Dict[str, Any] = {}
    for key, value in list(entry.items())[:MAX_PROJECTION_OBJECT_KEYS]:
        name = str(key)
        if (
            name == "_identity"
            or _is_secret_key(name, value)
            or _is_pii_field_name(name)
            or _normalize_key(name) == "name"
        ):
            continue
        projected, node, _redacted = _redact_value(
            value,
            depth=1,
            parent_key=name,
        )
        if node.get("kind") != "redacted":
            projections[name] = projected
    return projections


def _first_number(
    mapping: Mapping[str, Any], names: Sequence[str]
) -> Optional[float]:
    present, value = _first_present(mapping, names)
    if not present:
        return None
    return _parse_usage_number(value)


def _first_timestamp(
    mapping: Mapping[str, Any], names: Sequence[str]
) -> Optional[datetime]:
    present, value = _first_present(mapping, names)
    if not present:
        return None
    return _parse_usage_timestamp(value)


def _field_malformed(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    parsed: Optional[float],
) -> bool:
    present, value = _first_present(mapping, names)
    if not present:
        return False
    if value is None:
        return False
    return parsed is None


def _first_present(
    mapping: Mapping[str, Any], names: Sequence[str]
) -> Tuple[bool, Any]:
    for name in names:
        if name in mapping:
            return True, mapping[name]
        for actual in mapping:
            if str(actual) == name:
                return True, mapping[actual]
    return False, None


def _parse_usage_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not isfinite(number) or number < 0:
        return None
    return number


def _malformed_usage_field(
    field_name: Optional[str],
    value: Any,
) -> Optional[str]:
    normalized = _normalize_key(field_name) if field_name else ""
    if not _is_usage_number_field_name(normalized):
        return None
    if value is None or _parse_usage_number(value) is not None:
        return None
    return normalized


def _is_usage_number_field_name(field_name: str) -> bool:
    normalized = _normalize_key(field_name)
    return any(
        normalized == _normalize_key(alias)
        for alias in _USAGE_NUMBER_FIELD_ALIASES
    )


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _parse_usage_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = _finite_float(value)
        if numeric is None:
            return None
        if numeric > 10_000_000_000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _quota_key(kind: str, identity: str) -> str:
    return f"chatgpt_conversation_init:{kind}:{identity}"


def _safe_identity(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        finite_value = _finite_float(value)
        if finite_value is None:
            return None
        text = str(value) if isinstance(value, int) else (
            str(int(value)) if finite_value.is_integer() else str(value)
        )
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if not text or _is_unsafe_string(text):
        return None
    if _SLUG_RE.match(text):
        return text
    cleaned = re.sub(r"[^A-Za-z0-9._:-]+", "_", text).strip("_")
    if not cleaned:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return cleaned[:128]


def _is_secret_key(name: str, value: Any = None) -> bool:
    normalized = _normalize_key(name)
    if not normalized:
        return False
    if _is_safe_numeric_token_value(name, value):
        return False
    if normalized in {"authorization", "cookie", "cookies", "set-cookie", "set_cookie"}:
        return True
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def _is_safe_numeric_token_value(name: str, value: Any) -> bool:
    """Allow finite numeric token counters while rejecting token material."""
    if _finite_float(value) is None:
        return False
    normalized = _normalize_key(name)
    return "token" in normalized and any(
        marker in normalized.split("_")
        for marker in _SAFE_NUMERIC_TOKEN_KEY_MARKERS
    )


def _is_safe_usage_number_string(field_name: Optional[str], value: str) -> bool:
    """Permit provider-supplied numeric usage values, including JSON numbers."""

    normalized = _normalize_key(field_name)
    return not _is_unsafe_string(value) and any(
        marker in normalized
        for marker in (
            "limit",
            "quota",
            "max",
            "total",
            "remaining",
            "left",
            "used",
            "usage",
            "consumed",
        )
    )


def _is_unsafe_string(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("bearer "):
        return True
    if _EMAIL_RE.match(text) or _JWT_RE.match(text) or _HEX_RE.match(text):
        return True
    if "eyJ" in text and "." in text:
        return True
    return False


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return "unsupported"


def _mapping_nodes(
    payload: Mapping[str, Any],
) -> List[Tuple[Tuple[str, ...], Mapping[str, Any]]]:
    nodes: List[Tuple[Tuple[str, ...], Mapping[str, Any]]] = []
    pending: List[Tuple[Tuple[str, ...], Any]] = [((), payload)]
    while pending:
        path, value = pending.pop(0)
        if len(path) > 5:
            continue
        if isinstance(value, Mapping):
            nodes.append((path, value))
            for key, nested in value.items():
                pending.append((path + (str(key),), nested))
        elif isinstance(value, list):
            for index, nested in enumerate(value[:MAX_PROJECTION_LIST_ITEMS]):
                pending.append((path + (str(index),), nested))
    return nodes


def _values_at_structural_path(
    source: Mapping[str, Any],
    path: Sequence[str],
) -> List[Any]:
    """Read only a fixed key path; never traverse arbitrary child names."""

    values: List[Any] = [source]
    for segment in path:
        normalized_segment = _normalize_key(segment)
        next_values: List[Any] = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            for key, nested in value.items():
                if _normalize_key(key) == normalized_segment:
                    next_values.append(nested)
        values = next_values
        if not values:
            break
    return values


def _unique_pairs(
    parts: Sequence[str], fields: Sequence[str]
) -> Tuple[List[str], List[str]]:
    seen = set()
    unique_parts: List[str] = []
    unique_fields: List[str] = []
    for part, field_name in zip(parts, fields):
        if part in seen:
            continue
        seen.add(part)
        unique_parts.append(part)
        unique_fields.append(field_name)
    return unique_parts, unique_fields


@runtime_checkable
class ConversationInitTransport(Protocol):
    """Injected browser-boundary transport for conversation-init collection.

    Implementations own any authenticated session. The collector never reads
    Oracle cookies, never copies Authorization headers, and never adds a
    request body. Return a mapping with ``status_code`` and ``payload``;
    leaked cookies or headers are redacted and never written.
    """

    def fetch(self, request: urllib_request.Request) -> Mapping[str, Any]:
        """Execute the no-body POST contract and return a raw envelope."""


class OracleBrowserBoundaryUnavailable(ChatGPTConversationInitError):
    """The existing Oracle browser boundary cannot safely serve this poll."""

    def __init__(self, message: str) -> None:
        super().__init__(message, telemetry_class="auth")


class OracleBrowserCleanupError(OracleBrowserBoundaryUnavailable):
    """Native-history cleanup could not be proven within its deadline."""


class _OracleTargetCloserStage(IntEnum):
    UNKNOWN = 0
    PROCESS_GROUP_SETUP = 1
    ROLE_MARKER_SETUP = 2
    DEADLINE_CHECK = 3
    PLAYWRIGHT_START = 4
    CDP_CONNECT = 5
    CDP_SESSION = 6
    INITIAL_LISTING = 7
    INITIAL_LISTING_INVALID = 8
    INITIAL_TARGETS_INVALID = 9
    TARGET_MATCH = 10
    TARGET_MATCH_NOT_UNIQUE = 11
    TARGET_LOOKUP = 12
    TARGET_MISSING = 13
    CLOSE_REQUEST = 14
    CLOSE_NOT_ACKNOWLEDGED = 15
    ABSENCE_LISTING = 16
    ABSENCE_LISTING_INVALID = 17
    ABSENCE_TARGETS_INVALID = 18
    TARGET_ABSENCE_CHECK = 19
    TARGET_STILL_PRESENT = 20


_ORACLE_TARGET_CLOSER_FAILURE_MESSAGES = {
    stage: f"Oracle browser owned-target cleanup failed at {stage.name.lower()}."
    for stage in _OracleTargetCloserStage
}
CHATGPT_NATIVE_HISTORY_CLOSER_FAILURE_SUBREASONS = {
    message: f"target_closer_{stage.name.lower()}"
    for stage, message in _ORACLE_TARGET_CLOSER_FAILURE_MESSAGES.items()
}


@dataclass(frozen=True)
class NativeHistoryTargetProof:
    """Authoritative proof that one owned target is absent after close."""

    target_id: Optional[str]
    anchor_target_id: str
    creation_url: str


@dataclass(frozen=True)
class NativeHistoryReleaseProof:
    """Owner-derived authorization to release one history interception worker."""

    registration_id: str
    kind: str
    target_id: Optional[str]
    anchor_target_id: str
    creation_url: str


@dataclass
class NativeHistoryCloseRegistration:
    """Process registration for the independent exact-target closer."""

    process: Any
    private_process_group: Any
    private_process_start_time: Any
    target_proof: Any
    driver_done: Any
    reap_ack: Any
    target_id: Optional[str]
    creation_url: str
    start_state: str = "not_started"
    started: bool = False
    reaped: bool = False
    scope_reaped_proven: bool = False
    failure_stage: Any = None


@dataclass
class NativeHistoryLifecycleRegistration:
    """Nonserialized native-history ownership state held by its browser owner."""

    process: Any
    private_process_group: Any
    private_process_start_time: Any
    creation_state: Any
    owned_target: Any
    creation_url: str
    abort_event: Any
    release_event: Any
    release_control_state: Any
    creation_gate: Any
    creation_settled: Any
    creation_issued: Any
    cdp_endpoint: str
    anchor_target_id: str
    deadline: float
    operation_start: float
    target_close_budget: float
    finalization_gate: Any = None
    finalizing: bool = False
    cleanup_callback: Optional[Callable[..., Any]] = None
    close_registration: Optional[NativeHistoryCloseRegistration] = None
    start_state: str = "not_started"
    released: bool = False
    release_authorized: bool = False
    release_control_failed: bool = False
    retired: bool = False
    worker_retirement_proven: bool = False
    release_proof: Optional[NativeHistoryReleaseProof] = None
    target_resolution: Optional[NativeHistoryTargetProof] = None
    cleanup_plan: Optional[Dict[str, float]] = None
    cleanup_failure: Optional[str] = None
    shutdown_deadline: Optional[float] = None
    worker_scope_reaped_proven: bool = False
    registration_id: str = ""

    def __post_init__(self) -> None:
        if not self.registration_id:
            self.registration_id = "native-history-" + os.urandom(16).hex()


class NativeHistoryLifecycleCapability(Protocol):
    """Owner-side operations supplied only by a private browser owner."""

    def register_native_history(
        self, registration: NativeHistoryLifecycleRegistration
    ) -> None:
        ...

    def register_native_history_closer(
        self,
        registration: NativeHistoryLifecycleRegistration,
        closer: NativeHistoryCloseRegistration,
    ) -> None:
        ...

    def retain_native_history(
        self,
        registration: NativeHistoryLifecycleRegistration,
        reason: str,
    ) -> None:
        ...

    def release_native_history(
        self,
        registration: NativeHistoryLifecycleRegistration,
        *,
        proof: NativeHistoryReleaseProof,
    ) -> None:
        ...

    def retire_native_history(
        self,
        registration: NativeHistoryLifecycleRegistration,
    ) -> None:
        ...

    def prepare_native_history_cleanup_plan(
        self,
        registration: NativeHistoryLifecycleRegistration,
        *,
        cleanup_deadline: float,
        operation_deadline: float,
    ) -> Mapping[str, float]:
        ...

    def terminate_owned_browser(
        self,
        *,
        term_deadline: float,
        kill_deadline: float,
        reap_deadline: float,
        final_deadline: float,
        poll_only: bool = False,
    ) -> bool:
        ...

    def bind_native_history_endpoint(
        self,
        *,
        cdp_endpoint: str,
        anchor_target_id: str,
    ) -> None:
        ...


def _native_init_header(headers: Mapping[str, Any], name: str) -> Any:
    for key in headers:
        if key.lower() == name:
            return headers[key]
    return None


def _native_init_account_hash(headers: Mapping[str, Any]) -> Optional[str]:
    value = _native_init_header(headers, "chatgpt-account-id")
    if not isinstance(value, str) or value.strip() == "default":
        return None
    return hash_chatgpt_conversation_init_canonical_account_id(value)


def _native_init_retry_after(headers: Mapping[str, Any]) -> Optional[float]:
    from email.utils import parsedate_to_datetime
    from math import isfinite

    value = _native_init_header(headers, "retry-after")
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                return None
            seconds = max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if isfinite(seconds) and seconds >= 0 else None


def _observe_native_oracle_init(  # noqa: PLR0915 - callbacks share one capture lifetime
    page: Any,
    *,
    session: Any,
    request_url: str,
    expected_account_hash: str,
    deadline: float,
) -> Mapping[str, Any]:
    """Capture one native request without retaining its body or credentials."""
    import base64

    target = urlsplit(request_url)
    origin = f"https://{target.netloc}"
    capture: Dict[str, Any] = {}
    page_response: Dict[str, Any] = {}
    extra_hashes: Dict[str, Optional[str]] = {}
    failure: Optional[Dict[str, Any]] = None
    browser_challenge = False
    boundary_error = False
    init_fetch_id: Optional[str] = None
    # Leave the existing worker deadline some room to close its owned target.
    capture_deadline = deadline - min(2.0, _remaining_browser_timeout(deadline) / 5)

    def is_init(url: str) -> bool:
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.netloc == target.netloc
            and parsed.path == CHATGPT_CONVERSATION_INIT_PATH
        )

    def guard_request(event: Mapping[str, Any]) -> None:
        nonlocal boundary_error, init_fetch_id
        request = event.get("request", {})
        parsed = urlsplit(request.get("url", ""))
        redirected = event.get("redirectedRequestId")
        forbidden = (
            event.get("resourceType") == "Document"
            and (
                parsed.scheme != "https"
                or parsed.netloc != target.netloc
                or parsed.path not in {"", "/"}
                or redirected is not None
            )
        ) or (
            redirected is not None
            and (
                is_init(request.get("url", ""))
                or redirected == init_fetch_id
            )
        )
        # No model-message route is needed to render an empty page.
        forbidden = forbidden or (
            parsed.path in {"/backend-api/conversation", "/backend-api/f/conversation"}
            and request.get("method") == "POST"
        )
        if is_init(request.get("url", "")):
            forbidden = forbidden or init_fetch_id is not None
            init_fetch_id = event["requestId"]
        if forbidden:
            boundary_error = True
        if forbidden or boundary_error or failure is not None:
            session.send(
                "Fetch.failRequest",
                {"requestId": event["requestId"], "errorReason": "BlockedByClient"},
            )
        else:
            session.send("Fetch.continueRequest", {"requestId": event["requestId"]})

    def request_seen(event: Mapping[str, Any]) -> None:
        nonlocal boundary_error
        request = event.get("request", {})
        request_id = event.get("requestId")
        if event.get("redirectResponse") and (
            request_id == capture.get("request_id")
            or is_init(request.get("url", ""))
        ):
            boundary_error = True
        if capture or not is_init(request.get("url", "")):
            return
        capture.update(
            request_id=request_id,
            account_hash=_native_init_account_hash(request.get("headers", {})),
            method=request.get("method"),
            body_omitted=(
                not request.get("hasPostData", False) and "postData" not in request
            ),
        )
        if (
            capture["method"] != "POST"
            or capture["account_hash"] != expected_account_hash
        ):
            boundary_error = True

    def extra_seen(event: Mapping[str, Any]) -> None:
        request_id = event.get("requestId")
        if not isinstance(request_id, str):
            return
        account_hash = _native_init_account_hash(event.get("headers", {}))
        captured_id = capture.get("request_id")
        if account_hash is None and request_id != captured_id:
            return
        if request_id not in extra_hashes and len(extra_hashes) >= 256:
            oldest = next(
                (key for key in extra_hashes if key != captured_id), None
            )
            if oldest is not None:
                del extra_hashes[oldest]
        extra_hashes[request_id] = account_hash

    def response_seen(event: Mapping[str, Any]) -> None:
        nonlocal failure, boundary_error, browser_challenge
        response = event.get("response", {})
        status = response.get("status")
        headers = response.get("headers", {})
        request_id = event.get("requestId")
        if (
            event.get("type") == "Document"
            and response.get("url") == origin + "/"
        ):
            page_response.update(
                status_code=status,
                retry_after_seconds=_native_init_retry_after(headers),
            )
        if request_id == capture.get("request_id"):
            if not is_init(response.get("url", "")) or 300 <= status < 400:
                boundary_error = True
                return
            capture.update(
                status_code=status,
                retry_after_seconds=_native_init_retry_after(headers),
            )
        challenged = _native_init_header(headers, "cf-mitigated") == "challenge"
        browser_challenge = browser_challenge or challenged
        if status in {401, 403, 429} or challenged:
            candidate = {
                "status_code": status,
                "retry_after_seconds": _native_init_retry_after(headers),
                "correlated": request_id == capture.get("request_id"),
            }
            # A later authentication response must not erase throttle backoff.
            if failure is None or (
                status == 429 and failure.get("status_code") != 429
            ):
                failure = candidate
            elif status == 429 and failure.get("status_code") == 429:
                delays = [
                    delay
                    for delay in (
                        failure.get("retry_after_seconds"),
                        candidate["retry_after_seconds"],
                    )
                    if delay is not None
                ]
                failure["retry_after_seconds"] = max(delays) if delays else None

    def loading_finished(event: Mapping[str, Any]) -> None:
        if event.get("requestId") == capture.get("request_id"):
            capture["finished"] = True

    def loading_failed(event: Mapping[str, Any]) -> None:
        nonlocal boundary_error
        if event.get("requestId") == capture.get("request_id"):
            boundary_error = True

    def envelope(payload: Any, failed: bool = False) -> Mapping[str, Any]:
        state = failure if failed and failure is not None else capture
        request_id = capture.get("request_id")
        verified = (
            isinstance(request_id, str)
            and capture.get("account_hash") == expected_account_hash
            and extra_hashes.get(request_id) == expected_account_hash
        )
        native: Dict[str, Any] = {
            "account_hash": expected_account_hash if verified else None,
            "identity_source": "native_request_header",
            "selector_evidence": (
                "request_and_extra_info" if verified else "unverified"
            ),
            "request_response_correlated": (
                bool(state.get("correlated")) if failed else True
            ),
            "request_method": capture.get("method"),
            "request_body_omitted": capture.get("body_omitted"),
            "browser_challenge": browser_challenge,
        }
        retry_after = state.get("retry_after_seconds")
        if retry_after is not None:
            native["retry_after_seconds"] = retry_after
        return {
            "status_code": state.get("status_code"),
            "payload": payload,
            "native_capture": native,
        }

    try:
        session.on("Fetch.requestPaused", guard_request)
        session.on("Network.requestWillBeSent", request_seen)
        session.on("Network.requestWillBeSentExtraInfo", extra_seen)
        session.on("Network.responseReceived", response_seen)
        session.on("Network.loadingFinished", loading_finished)
        session.on("Network.loadingFailed", loading_failed)
        session.send(
            "Network.enable",
            {
                "maxTotalBufferSize": MAX_CONVERSATION_INIT_SOURCE_BYTES * 2,
                "maxResourceBufferSize": MAX_CONVERSATION_INIT_SOURCE_BYTES,
                "maxPostDataSize": 0,
            },
        )
        session.send("Network.setBypassServiceWorker", {"bypass": True})
        session.send(
            "Fetch.enable",
            {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]},
        )
        try:
            page.goto(
                origin + "/",
                wait_until="commit",
                timeout=_browser_timeout_milliseconds(
                    _remaining_browser_timeout(capture_deadline)
                ),
            )
        except Exception:
            if failure is None:
                raise
        while True:
            if failure is not None:
                return envelope(None, failed=True)
            if boundary_error:
                raise OracleBrowserBoundaryUnavailable(
                    "Native Oracle init identity, method, or redirect was rejected."
                )
            _raise_if_browser_deadline_expired(capture_deadline)
            request_id = capture.get("request_id")
            if capture.get("finished") and request_id in extra_hashes:
                if extra_hashes[request_id] != expected_account_hash:
                    raise OracleBrowserBoundaryUnavailable(
                        "Native Oracle init ExtraInfo account did not match inventory."
                    )
                try:
                    body = session.send(
                        "Network.getResponseBody", {"requestId": request_id}
                    )
                except Exception:
                    if failure is not None:
                        return envelope(None, failed=True)
                    raise
                # Synchronous CDP calls can dispatch queued network callbacks.
                if failure is not None:
                    return envelope(None, failed=True)
                if boundary_error:
                    raise OracleBrowserBoundaryUnavailable(
                        "Native Oracle init boundary changed during response capture."
                    )
                _raise_if_browser_deadline_expired(capture_deadline)
                content = body.get("body", "")
                if len(content) > MAX_CONVERSATION_INIT_SOURCE_BYTES:
                    raise OracleBrowserBoundaryUnavailable(
                        "Native Oracle init response exceeded the size limit."
                    )
                if body.get("base64Encoded"):
                    content = base64.b64decode(content, validate=True).decode("utf-8")
                try:
                    payload = json.loads(content)
                except ValueError:
                    payload = None
                return envelope(payload)
            if page.evaluate(
                "() => [...document.querySelectorAll("
                "'form#challenge-form[action*=\"__cf_chl\"],"
                "form#challenge-form[action^=\"/cdn-cgi/challenge-platform/\"],"
                "#challenge-running')].some(node => {"
                "const rect = node.getBoundingClientRect();"
                "return rect.width > 0 && rect.height > 0 &&"
                "getComputedStyle(node).visibility !== 'hidden';})"
            ):
                browser_challenge = True
                if failure is None:
                    failure = {**page_response, "correlated": False}
                return envelope(None, failed=True)
            page.wait_for_timeout(50)
    finally:
        # Keep Fetch interception installed until the caller closes the target.
        # Closing that target tears down the session; do not detach it here.
        boundary_error = True


def _native_history_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return "unsupported"


def _native_history_metadata(value: Any, *, present: bool = True) -> Dict[str, Any]:
    if not present:
        return {
            "present": False,
            "container": "missing",
            "type": "missing",
            "count": None,
            "count_truncated": False,
            "count_is_lower_bound": False,
        }
    if isinstance(value, list):
        truncated = len(value) > _NATIVE_HISTORY_MAX_COUNT
        return {
            "present": True,
            "container": "array",
            "type": "array",
            "count": min(len(value), _NATIVE_HISTORY_MAX_COUNT),
            "count_truncated": truncated,
            "count_is_lower_bound": truncated,
        }
    if isinstance(value, Mapping):
        truncated = len(value) > _NATIVE_HISTORY_MAX_COUNT
        return {
            "present": True,
            "container": "object",
            "type": "object",
            "count": min(len(value), _NATIVE_HISTORY_MAX_COUNT),
            "count_truncated": truncated,
            "count_is_lower_bound": truncated,
        }
    return {
        "present": True,
        "container": "scalar",
        "type": _native_history_type_name(value),
        "count": None,
        "count_truncated": False,
        "count_is_lower_bound": False,
    }


def _native_history_fixed_field_value(
    payload: Any,
    field_name: str,
) -> Tuple[bool, Any]:
    if not isinstance(payload, Mapping):
        return False, None
    aliases = {
        "has_missing_conversations": ("has_missing_conversations", "hasMissingConversations"),
    }.get(field_name, (field_name,))
    candidates: List[Mapping[str, Any]] = [payload]
    for wrapper_name in ("data", "result"):
        wrapped = payload.get(wrapper_name)
        if isinstance(wrapped, Mapping):
            candidates.append(wrapped)
    for candidate in candidates:
        for alias in aliases:
            if alias in candidate:
                return True, candidate[alias]
    return False, None


def _native_history_model_presence(payload: Any) -> Dict[str, Optional[bool]]:
    counters, traversal = _native_history_field_presence_type_counts(payload)
    fields: Dict[str, Optional[bool]] = {}
    for field_name in _NATIVE_HISTORY_MODEL_FIELD_NAMES:
        field = counters["model"][field_name]
        if field["presence_count"]:
            fields[field_name] = True
        elif traversal["absence_is_known"]:
            fields[field_name] = False
        else:
            fields[field_name] = None
    return fields


def _native_history_empty_field_counters() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        group_name: {
            field_name: {
                "presence_count": 0,
                "count_is_lower_bound": False,
                "type_counts": {
                    type_name: 0 for type_name in _NATIVE_HISTORY_VALUE_TYPES
                },
            }
            for field_name, _aliases in field_specs
        }
        for group_name, field_specs in _NATIVE_HISTORY_FIELD_COUNTER_SPECS.items()
    }


def _native_history_field_presence_type_counts(
    payload: Any,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    counters = _native_history_empty_field_counters()
    if payload is None or not isinstance(payload, (Mapping, list)):
        return counters, {
            "visited_nodes": 0,
            "max_nodes": _NATIVE_HISTORY_MAX_INSPECTED_NODES,
            "truncated": False,
            "absence_is_known": False,
        }

    pending: List[Any] = [payload]
    visited = 0
    truncated = False
    while pending:
        if visited >= _NATIVE_HISTORY_MAX_INSPECTED_NODES:
            truncated = True
            break
        value = pending.pop()
        if isinstance(value, Mapping):
            visited += 1
            for group_name, field_specs in _NATIVE_HISTORY_FIELD_COUNTER_SPECS.items():
                for field_name, aliases in field_specs:
                    present_alias = next(
                        (alias for alias in aliases if alias in value),
                        None,
                    )
                    if present_alias is None:
                        continue
                    field_counter = counters[group_name][field_name]
                    field_counter["presence_count"] += 1
                    type_name = _native_history_type_name(
                        value[present_alias]
                    )
                    field_counter["type_counts"][type_name] += 1
            children = list(value.values())
            if len(children) > MAX_PROJECTION_OBJECT_KEYS:
                truncated = True
                children = children[:MAX_PROJECTION_OBJECT_KEYS]
            pending.extend(children)
        elif isinstance(value, list):
            visited += 1
            children = value
            if len(children) > MAX_PROJECTION_LIST_ITEMS:
                truncated = True
                children = children[:MAX_PROJECTION_LIST_ITEMS]
            pending.extend(children)

    if truncated:
        for group_counters in counters.values():
            for field_counter in group_counters.values():
                field_counter["count_is_lower_bound"] = True
    return counters, {
        "visited_nodes": visited,
        "max_nodes": _NATIVE_HISTORY_MAX_INSPECTED_NODES,
        "truncated": truncated,
        "absence_is_known": not truncated,
    }


def _native_history_structural_projection(payload: Any) -> Dict[str, Any]:
    root = _native_history_metadata(payload)
    field_metadata: Dict[str, Dict[str, Any]] = {}
    present_count = 0
    for field_name in _NATIVE_HISTORY_FIELD_NAMES:
        present, value = _native_history_fixed_field_value(payload, field_name)
        field_metadata[field_name] = _native_history_metadata(
            value,
            present=present,
        )
        present_count += int(present)
    if not isinstance(payload, (Mapping, list)):
        schema_state = "malformed"
    elif present_count:
        schema_state = "recognized"
    else:
        schema_state = "unrecognized"
    field_counters, traversal = _native_history_field_presence_type_counts(
        payload
    )
    return {
        "root": root,
        "fields": field_metadata,
        "schema_state": schema_state,
        "field_presence_type_counts": field_counters,
        "traversal": traversal,
    }


def _native_history_base_observation(
    *,
    page_target_id_matched: bool,
    request_count: int = 0,
    history_request_count: int = 0,
) -> Dict[str, Any]:
    bounded_request_count = min(
        max(int(request_count), 0),
        _NATIVE_HISTORY_MAX_COUNT,
    )
    bounded_history_request_count = min(
        max(int(history_request_count), 0),
        _NATIVE_HISTORY_MAX_COUNT,
    )
    request_count_truncated = request_count > _NATIVE_HISTORY_MAX_COUNT
    history_request_count_truncated = (
        history_request_count > _NATIVE_HISTORY_MAX_COUNT
    )
    return {
        "observer": CHATGPT_NATIVE_HISTORY_OBSERVER,
        "observation_state": "no_history_observed",
        "route_class": "none",
        "http_status": None,
        "account_identity_verified": False,
        "account_identity_source": None,
        "identity_match": False,
        "request_response_correlated": False,
        "request_method": None,
        "request_body_omitted": None,
        "page_target_id_matched": bool(page_target_id_matched),
        "request_count": bounded_request_count,
        "request_count_truncated": request_count_truncated,
        "request_count_is_lower_bound": request_count_truncated,
        "history_request_count": bounded_history_request_count,
        "history_request_count_truncated": history_request_count_truncated,
        "history_request_count_is_lower_bound": history_request_count_truncated,
        "response_bytes": None,
        "response_content_type": "missing",
        "retry_after_seconds": None,
        "browser_challenge": False,
        "structural_metadata": {
            "root": _native_history_metadata(None, present=False),
            "fields": {
                field_name: _native_history_metadata(None, present=False)
                for field_name in _NATIVE_HISTORY_FIELD_NAMES
            },
            "schema_state": "absent",
        },
        "model_field_presence": {
            field_name: None for field_name in _NATIVE_HISTORY_MODEL_FIELD_NAMES
        },
        "failure_reason": "no_history_observed",
        "warnings": [],
    }


def _native_history_content_type(headers: Any) -> str:
    if not isinstance(headers, Mapping):
        return "missing"
    value = _native_init_header(headers, "content-type")
    if not isinstance(value, str):
        return "missing"
    normalized = value.split(";", 1)[0].strip().lower()
    if normalized in {"application/json", "application/problem+json"}:
        return "json"
    if normalized in {"text/html", "text/plain"}:
        return "text"
    return "other"


def _native_history_content_length(headers: Any) -> Optional[int]:
    if not isinstance(headers, Mapping):
        return None
    value = _native_init_header(headers, "content-length")
    if not isinstance(value, str):
        return None
    try:
        length = int(value.strip())
    except (TypeError, ValueError):
        return None
    return length if length >= 0 else None


def _native_history_status(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        status = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return status if 100 <= status <= 999 else None


def _native_history_request_method(value: Any) -> Optional[str]:
    if value == "GET":
        return "GET"
    if isinstance(value, str):
        return "other"
    return None


def _native_history_failure_for_status(status: Optional[int]) -> Optional[str]:
    if status is None:
        return "invalid_status"
    if 200 <= status < 300 and status not in {204, 206}:
        return None
    if status in {401, 403}:
        return "http_auth"
    if status == 429:
        return "http_rate_limited"
    return "http_error"


def _native_history_finalize_observation(
    capture: Mapping[str, Any],
    *,
    expected_account_hash: str,
    page_target_id_matched: bool,
    request_count: int,
    history_request_count: int,
    max_response_bytes: int,
    structural: Optional[Mapping[str, Any]] = None,
    failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    request_id = capture.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return _native_history_no_route_observation(
            page_target_id_matched=page_target_id_matched,
            request_count=request_count,
            history_request_count=history_request_count,
            failure_reason=(
                failure_reason
                or capture.get("boundary_reason")
                or "no_history_observed"
            ),
            browser_challenge=bool(capture.get("browser_challenge")),
            retry_after_seconds=capture.get("retry_after_seconds"),
        )
    result = _native_history_base_observation(
        page_target_id_matched=page_target_id_matched,
        request_count=request_count,
        history_request_count=history_request_count,
    )
    request_hash = capture.get("request_account_hash")
    extra_hash = capture.get("extra_account_hash")
    identity_match = (
        not capture.get("identity_conflict")
        and request_hash == expected_account_hash
        and extra_hash == expected_account_hash
    )
    status = _native_history_status(capture.get("status_code"))
    response_bytes = capture.get("response_bytes")
    if isinstance(response_bytes, bool) or not isinstance(response_bytes, int):
        response_bytes = None
    if response_bytes is not None:
        response_bytes = min(max(response_bytes, 0), max_response_bytes)
    result.update(
        {
            "observation_state": "history_observed",
            "route_class": "modern_history_index",
            "http_status": status,
            "account_identity_verified": identity_match,
            "account_identity_source": (
                CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
                if (
                    request_hash is not None
                    or extra_hash is not None
                    or capture.get("identity_conflict")
                )
                else None
            ),
            "identity_match": identity_match,
            "request_response_correlated": bool(
                capture.get("response_received")
                and capture.get("response_request_id")
                == capture.get("request_id")
            ),
            "request_method": _native_history_request_method(
                capture.get("method")
            ),
            "request_body_omitted": capture.get("body_omitted"),
            "response_bytes": response_bytes,
            "response_content_type": capture.get(
                "content_type",
                "missing",
            ),
            "retry_after_seconds": capture.get("retry_after_seconds"),
            "browser_challenge": bool(capture.get("browser_challenge")),
        }
    )
    if structural is not None:
        result["structural_metadata"] = dict(structural)
        result["model_field_presence"] = _native_history_model_presence(
            capture.get("payload")
        )
        result["field_presence_type_counts"] = dict(
            structural.get("field_presence_type_counts", {})
        )
        result["traversal"] = dict(structural.get("traversal", {}))
    else:
        result["structural_metadata"] = {
            "root": _native_history_metadata(None, present=False),
            "fields": {
                field_name: _native_history_metadata(None, present=False)
                for field_name in _NATIVE_HISTORY_FIELD_NAMES
            },
            "schema_state": "absent",
        }
        result["field_presence_type_counts"] = (
            _native_history_empty_field_counters()
        )
        result["traversal"] = {
            "visited_nodes": 0,
            "max_nodes": _NATIVE_HISTORY_MAX_INSPECTED_NODES,
            "truncated": False,
            "absence_is_known": False,
        }
    intentional_disposal = (
        capture.get("intentional_disposal_request_id")
        == capture.get("fetch_request_id")
        and capture.get("response_stream_complete") is True
    )
    terminal_reason = capture.get("terminal_reason")
    if isinstance(terminal_reason, str) and terminal_reason:
        failure_reason = terminal_reason
    elif capture.get("boundary_reason"):
        failure_reason = str(capture["boundary_reason"])
    elif failure_reason is None and capture.get("body_failure_reason"):
        failure_reason = str(capture["body_failure_reason"])
    if failure_reason is None and capture.get("loading_failed") and not intentional_disposal:
        failure_reason = "history_response_failed"
    if failure_reason is None and not capture.get("response_received"):
        failure_reason = "history_response_missing"
    if failure_reason is None and not identity_match:
        failure_reason = (
            "account_identity_mismatch"
            if (
                request_hash is not None
                or extra_hash is not None
                or capture.get("identity_conflict")
            )
            else "identity_evidence_missing"
        )
    if failure_reason is None and capture.get("browser_challenge"):
        failure_reason = "browser_challenge"
    if failure_reason is None:
        status_failure = _native_history_failure_for_status(status)
        if status_failure is not None:
            failure_reason = status_failure
    if failure_reason is None and capture.get("content_type") != "json":
        failure_reason = "non_json_response"
    if failure_reason is None and capture.get("response_stream_complete") is not True:
        failure_reason = "history_response_missing"
    if failure_reason is not None:
        result["observation_state"] = "history_observation_failed"
    result["failure_reason"] = failure_reason
    return result


def _native_history_no_route_observation(
    *,
    page_target_id_matched: bool,
    request_count: int,
    history_request_count: int,
    failure_reason: Optional[str],
    browser_challenge: bool,
    retry_after_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    result = _native_history_base_observation(
        page_target_id_matched=page_target_id_matched,
        request_count=request_count,
        history_request_count=history_request_count,
    )
    result["browser_challenge"] = bool(browser_challenge)
    result["retry_after_seconds"] = retry_after_seconds
    result["failure_reason"] = failure_reason or "no_history_observed"
    if result["failure_reason"] != "no_history_observed":
        result["observation_state"] = "history_observation_failed"
    if failure_reason == "no_history_observed" or failure_reason is None:
        result["warnings"] = ["no_native_history_index_request"]
    return result


def _native_history_response_headers(headers: Any) -> Mapping[str, Any]:
    if isinstance(headers, Mapping):
        return headers
    if not isinstance(headers, list):
        return {}
    allowed = {
        "content-type",
        "content-length",
        "retry-after",
        "cf-mitigated",
    }
    normalized: Dict[str, str] = {}
    for header in headers:
        if not isinstance(header, Mapping):
            continue
        name = header.get("name")
        value = header.get("value")
        if (
            isinstance(name, str)
            and name.strip().lower() in allowed
            and isinstance(value, str)
        ):
            normalized[name.strip().lower()] = value
    return normalized


def _capture_native_history_body_stream(  # noqa: PLR0915 - bounded response stream
    session: Any,
    capture: Dict[str, Any],
    *,
    fetch_request_id: str,
    response_headers: Any,
    deadline: float,
    max_response_bytes: int,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
    import binascii
    import base64

    headers = _native_history_response_headers(response_headers)
    content_length = _native_history_content_length(headers)
    if content_length is not None and content_length > max_response_bytes:
        return None, "response_too_large", content_length
    if should_stop is not None and should_stop():
        return None, "capture_interrupted", 0
    try:
        stream_result = session.send(
            "Fetch.takeResponseBodyAsStream",
            {"requestId": fetch_request_id},
        )
    except Exception:
        return None, "response_body_unavailable", None
    if not isinstance(stream_result, Mapping):
        return None, "response_body_unavailable", None
    stream_handle = stream_result.get("stream")
    if not isinstance(stream_handle, str) or not stream_handle:
        return None, "response_body_unavailable", None
    if should_stop is not None and should_stop():
        try:
            session.send("IO.close", {"handle": stream_handle})
        except Exception:
            pass
        return None, "capture_interrupted", 0

    raw_bytes = bytearray()
    try:
        while True:
            _raise_if_browser_deadline_expired(deadline)
            if should_stop is not None and should_stop():
                return None, "capture_interrupted", len(raw_bytes)
            remaining = max_response_bytes - len(raw_bytes)
            read_size = min(
                _NATIVE_HISTORY_STREAM_CHUNK_BYTES,
                remaining + 1,
            )
            if read_size <= 0:
                return None, "response_too_large", len(raw_bytes) + 1
            chunk_result = session.send(
                "IO.read",
                {"handle": stream_handle, "size": read_size},
            )
            if should_stop is not None and should_stop():
                return None, "capture_interrupted", len(raw_bytes)
            if not isinstance(chunk_result, Mapping):
                return None, "invalid_response_body", len(raw_bytes)
            chunk = chunk_result.get("data")
            if not isinstance(chunk, str):
                return None, "invalid_response_body", len(raw_bytes)
            try:
                chunk_bytes = (
                    base64.b64decode(chunk, validate=True)
                    if chunk_result.get("base64Encoded")
                    else chunk.encode("utf-8")
                )
            except (UnicodeDecodeError, ValueError, binascii.Error):
                return None, "invalid_response_body", len(raw_bytes)
            if len(raw_bytes) + len(chunk_bytes) > max_response_bytes:
                return (
                    None,
                    "response_too_large",
                    len(raw_bytes) + len(chunk_bytes),
                )
            raw_bytes.extend(chunk_bytes)
            if should_stop is not None and should_stop():
                return None, "capture_interrupted", len(raw_bytes)
            if chunk_result.get("eof") is True:
                break
            if not chunk_bytes:
                return None, "invalid_response_body", len(raw_bytes)
    finally:
        try:
            session.send("IO.close", {"handle": stream_handle})
        except Exception:
            pass

    response_bytes = len(raw_bytes)
    if content_length is not None:
        response_bytes = max(response_bytes, content_length)
    if should_stop is not None and should_stop():
        return None, "capture_interrupted", response_bytes
    try:
        payload = json.loads(bytes(raw_bytes).decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError):
        return None, "invalid_json", response_bytes
    if should_stop is not None and should_stop():
        return None, "capture_interrupted", response_bytes
    # Keep the parsed value only for this bounded in-process projection.
    capture["payload"] = payload
    capture["response_stream_complete"] = True
    return _native_history_structural_projection(payload), None, response_bytes


def _observe_native_history_oracle_page(  # noqa: PLR0915 - bounded CDP lifetime
    page: Any,
    *,
    session: Any,
    expected_account_hash: str,
    deadline: float,
    max_response_bytes: int,
    abort_event: Optional[Any] = None,
) -> Mapping[str, Any]:
    capture: Dict[str, Any] = {}
    boundary_reason: Optional[str] = None
    terminal_failure: Optional[Dict[str, Any]] = None
    pending_response: Optional[Dict[str, Any]] = None
    pending_network: Dict[str, Dict[str, Any]] = {}
    pending_network_order: List[str] = []
    history_network_ids: set[str] = set()
    seen_fetch_request_ids: set[str] = set()
    fetch_actions: Dict[Tuple[str, str], str] = {}
    init_fetch_id: Optional[str] = None
    observer_closed = False
    response_read_in_progress = False
    browser_challenge = False
    request_count = 0
    history_request_count = 0
    history_admitted = False
    retry_after_max: Optional[float] = None
    capture_deadline = deadline - min(2.0, _remaining_browser_timeout(deadline) / 5)

    def set_boundary(reason: str) -> None:
        nonlocal boundary_reason
        if boundary_reason is None:
            boundary_reason = reason
        capture.setdefault("boundary_reason", boundary_reason)

    def is_stopped() -> bool:
        return (
            observer_closed
            or boundary_reason is not None
            or terminal_failure is not None
            or (abort_event is not None and abort_event.is_set())
        )

    def is_history_url(url: Any) -> bool:
        if not isinstance(url, str):
            return False
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.netloc == "chatgpt.com"
            and parsed.path == CHATGPT_NATIVE_HISTORY_INDEX_PATH
        )

    def is_init_url(url: Any) -> bool:
        if not isinstance(url, str):
            return False
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.netloc == "chatgpt.com"
            and parsed.path == CHATGPT_CONVERSATION_INIT_PATH
        )

    def is_conversation_detail_path(path: str) -> bool:
        return (
            path.startswith("/backend-api/conversation/")
            or path.startswith("/backend-api/f/conversation/")
            or path.startswith("/backend-api/conversations/")
            or path.startswith("/backend-api/f/conversations/")
        )

    def is_model_or_mutation_path(path: str) -> bool:
        return (
            path in {
                "/backend-api/conversation",
                "/backend-api/f/conversation",
            }
            or is_conversation_detail_path(path)
        )

    def terminal_priority(reason: str) -> int:
        if "rate_limited" in reason:
            return 4
        if "auth" in reason:
            return 3
        if "challenge" in reason:
            return 2
        return 1

    def latch_terminal(
        reason: str,
        *,
        retry_after_seconds: Optional[float] = None,
        challenged: bool = False,
    ) -> None:
        nonlocal browser_challenge, retry_after_max, terminal_failure
        browser_challenge = browser_challenge or challenged
        if (
            isinstance(retry_after_seconds, (int, float))
            and not isinstance(retry_after_seconds, bool)
            and isfinite(float(retry_after_seconds))
            and retry_after_seconds >= 0
        ):
            retry_after_max = max(
                retry_after_max or 0.0,
                float(retry_after_seconds),
            )
        candidate = {
            "reason": reason,
            "retry_after_seconds": retry_after_max,
            "browser_challenge": browser_challenge,
        }
        if terminal_failure is None or terminal_priority(reason) > terminal_priority(
            str(terminal_failure.get("reason") or "")
        ):
            terminal_failure = candidate
        elif terminal_failure is not None:
            terminal_failure["retry_after_seconds"] = retry_after_max
            terminal_failure["browser_challenge"] = bool(
                terminal_failure.get("browser_challenge") or browser_challenge
            )
        capture["browser_challenge"] = browser_challenge
        capture["terminal_reason"] = terminal_failure["reason"]
        capture["retry_after_seconds"] = retry_after_max

    def terminal_reason() -> Optional[str]:
        if terminal_failure is None:
            return None
        reason = terminal_failure.get("reason")
        return reason if isinstance(reason, str) else "terminal_failure"

    def terminal_retry_after() -> Optional[float]:
        if isinstance(retry_after_max, (int, float)):
            return retry_after_max
        return None

    def fail_fetch(
        request_id: Any,
        *,
        pause_stage: str,
        intentional_disposal: bool = False,
    ) -> None:
        if observer_closed:
            return
        if not isinstance(request_id, str) or not request_id:
            return
        action_key = (request_id, pause_stage)
        if action_key in fetch_actions:
            return
        fetch_actions[action_key] = "fail"
        if request_id == capture.get("fetch_request_id"):
            # Network.loadingFailed can be caused by this observer's own
            # rejection; keep it separate from an upstream transport failure.
            capture["observer_rejected_fetch_request_id"] = request_id
        if intentional_disposal:
            capture["intentional_disposal_request_id"] = request_id
        try:
            session.send(
                "Fetch.failRequest",
                {
                    "requestId": request_id,
                    "errorReason": "BlockedByClient",
                },
            )
        except Exception:
            if not observer_closed:
                set_boundary("fetch_control_failed")

    def continue_fetch(request_id: Any) -> None:
        if observer_closed or is_stopped():
            return
        if not isinstance(request_id, str) or not request_id:
            set_boundary("fetch_control_failed")
            return
        action_key = (request_id, "request")
        if action_key in fetch_actions:
            return
        fetch_actions[action_key] = "continue"
        try:
            session.send("Fetch.continueRequest", {"requestId": request_id})
        except Exception:
            set_boundary("fetch_control_failed")

    def remember_network(network_id: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(network_id, str) or not network_id:
            return None
        if network_id not in pending_network:
            if len(pending_network_order) >= _NATIVE_HISTORY_MAX_PENDING_NETWORKS:
                evict_index = next(
                    (
                        index
                        for index, candidate in enumerate(pending_network_order)
                        if candidate != capture.get("network_request_id")
                    ),
                    None,
                )
                if evict_index is None:
                    capture["pending_network_truncated"] = True
                    return None
                oldest = pending_network_order.pop(evict_index)
                pending_network.pop(oldest, None)
            pending_network[network_id] = {
                "network_id": network_id,
                "request_hash": None,
                "extra_hash": None,
                "identity_conflict": False,
            }
            pending_network_order.append(network_id)
        return pending_network[network_id]

    def note_history_request(network_id: Any) -> None:
        nonlocal history_request_count
        if not isinstance(network_id, str) or not network_id:
            return
        if network_id in history_network_ids:
            return
        if len(history_network_ids) < _NATIVE_HISTORY_MAX_COUNT + 1:
            history_network_ids.add(network_id)
        history_request_count = min(
            history_request_count + 1,
            _NATIVE_HISTORY_MAX_COUNT + 1,
        )

    def record_identity(
        record: Dict[str, Any],
        field_name: str,
        account_hash: Optional[str],
    ) -> None:
        if account_hash is None:
            return
        prior = record.get(field_name)
        if prior is not None and prior != account_hash:
            record["identity_conflict"] = True
        if account_hash != expected_account_hash:
            record["identity_conflict"] = True
            latch_terminal("account_identity_mismatch")
        if prior is None:
            record[field_name] = account_hash
        if record.get("identity_conflict"):
            capture["identity_conflict"] = True

    def apply_network_record(record: Optional[Mapping[str, Any]]) -> None:
        if not isinstance(record, Mapping):
            return
        if record.get("request_seen"):
            capture["network_request_seen"] = True
        if record.get("extra_seen"):
            capture["extra_seen"] = True
        for field_name in (
            "request_hash",
            "extra_hash",
            "response_received",
            "response_request_id",
            "status_code",
            "content_type",
            "content_length",
            "response_url_valid",
            "response_retry_after_seconds",
            "network_loading_finished",
            "loading_failed",
        ):
            if field_name in record:
                if field_name == "response_retry_after_seconds":
                    response_retry = record[field_name]
                    if isinstance(response_retry, (int, float)) and (
                        capture.get("retry_after_seconds") is None
                        or response_retry > capture["retry_after_seconds"]
                    ):
                        capture["retry_after_seconds"] = response_retry
                    continue
                target_name = {
                    "request_hash": "request_account_hash",
                    "extra_hash": "extra_account_hash",
                }.get(field_name, field_name)
                capture[target_name] = record[field_name]
        if record.get("identity_conflict"):
            capture["identity_conflict"] = True
        if record.get("response_error_reason"):
            capture["response_error_observed"] = True

    def record_network_request(
        network_id: Any,
        *,
        request: Mapping[str, Any],
        redirected: Any = None,
    ) -> Optional[Dict[str, Any]]:
        record = remember_network(network_id)
        if record is None:
            return None
        url = request.get("url")
        method = str(request.get("method") or "").upper()
        record["request_seen"] = True
        record["method"] = method
        record["body_omitted"] = (
            not request.get("hasPostData", False)
            and "postData" not in request
        )
        record["redirected"] = redirected is not None or bool(
            request.get("redirectResponse")
        )
        record["history"] = is_history_url(url)
        headers = request.get("headers")
        account_hash = (
            _native_init_account_hash(headers)
            if isinstance(headers, Mapping)
            else None
        )
        record_identity(record, "request_hash", account_hash)
        if record.get("history"):
            note_history_request(network_id)
        if network_id == capture.get("network_request_id"):
            apply_network_record(record)
        return record

    def record_network_extra(
        network_id: Any,
        *,
        headers: Any,
    ) -> Optional[Dict[str, Any]]:
        record = remember_network(network_id)
        if record is None:
            return None
        record["extra_seen"] = True
        account_hash = (
            _native_init_account_hash(headers)
            if isinstance(headers, Mapping)
            else None
        )
        record_identity(record, "extra_hash", account_hash)
        if network_id == capture.get("network_request_id"):
            apply_network_record(record)
        return record

    def record_relevant_response(
        *,
        response_url: Any,
        status: Optional[int],
        headers: Any,
    ) -> None:
        nonlocal browser_challenge
        normalized_headers = _native_history_response_headers(headers)
        challenged = (
            _native_init_header(normalized_headers, "cf-mitigated") == "challenge"
        )
        browser_challenge = browser_challenge or challenged
        retry_after = _native_init_retry_after(normalized_headers)
        if challenged:
            latch_terminal(
                "browser_challenge",
                retry_after_seconds=retry_after,
                challenged=True,
            )
        parsed_url = urlsplit(response_url) if isinstance(response_url, str) else None
        if parsed_url is None or parsed_url.netloc != "chatgpt.com":
            return
        if parsed_url.path == "/":
            route_name = "home"
        elif parsed_url.path == CHATGPT_NATIVE_HISTORY_INDEX_PATH:
            route_name = "history"
        elif parsed_url.path == CHATGPT_CONVERSATION_INIT_PATH:
            route_name = "init"
        elif parsed_url.path == "/api/auth/session":
            route_name = "auth_session"
        elif parsed_url.path in _NATIVE_HISTORY_BOOTSTRAP_READ_PATHS:
            route_name = "bootstrap"
        else:
            return
        if status in {401, 403}:
            latch_terminal(
                f"{route_name}_auth",
                retry_after_seconds=retry_after,
            )
        elif status == 429:
            latch_terminal(
                f"{route_name}_rate_limited",
                retry_after_seconds=retry_after,
            )

    def store_response(
        *,
        request_id: Any,
        response_url: Any,
        status: Optional[int],
        headers: Any,
        classify_boundary: bool = True,
    ) -> None:
        record = remember_network(request_id)
        if record is None:
            return
        normalized_headers = _native_history_response_headers(headers)
        record.update(
            response_received=True,
            response_request_id=request_id,
            status_code=status,
            content_type=_native_history_content_type(normalized_headers),
            content_length=_native_history_content_length(normalized_headers),
            response_retry_after_seconds=_native_init_retry_after(
                normalized_headers
            ),
            response_url_valid=is_history_url(response_url),
        )
        record_relevant_response(
            response_url=response_url,
            status=status,
            headers=normalized_headers,
        )
        if classify_boundary and is_history_url(response_url):
            if not record["response_url_valid"]:
                set_boundary("history_response_mismatch")
            elif status is None:
                set_boundary("invalid_status")
            elif 300 <= status < 400:
                set_boundary("history_response_redirected")
        if request_id == capture.get("network_request_id"):
            apply_network_record(record)

    def store_loading_finished(request_id: Any, event: Mapping[str, Any]) -> None:
        record = remember_network(request_id)
        if record is None:
            return
        record["network_loading_finished"] = True
        encoded = event.get("encodedDataLength")
        if (
            isinstance(encoded, (int, float))
            and not isinstance(encoded, bool)
            and encoded >= 0
        ):
            record["encoded_data_length"] = int(encoded)
        if request_id == capture.get("network_request_id"):
            apply_network_record(record)

    def store_loading_failed(request_id: Any) -> None:
        record = remember_network(request_id)
        if record is None:
            return
        record["loading_failed"] = True
        if request_id != capture.get("network_request_id"):
            return
        fetch_request_id = capture.get("fetch_request_id")
        observer_rejected = (
            isinstance(fetch_request_id, str)
            and capture.get("observer_rejected_fetch_request_id")
            == fetch_request_id
        )
        if observer_rejected:
            intentional_disposal = (
                capture.get("intentional_disposal_request_id")
                == fetch_request_id
                and capture.get("response_stream_complete") is True
            )
            if (
                intentional_disposal
                or capture.get("body_failure_reason") is not None
            ):
                capture["loading_failed_ignored"] = True
                return
        capture["loading_failed"] = True
        if capture.get("body_failure_reason") is None:
            capture["body_failure_reason"] = "history_response_failed"
        capture["network_loading_finished"] = True

    def response_read_should_stop() -> bool:
        return (
            observer_closed
            or boundary_reason is not None
            or terminal_failure is not None
            or (abort_event is not None and abort_event.is_set())
            or capture.get("identity_conflict") is True
            or (
                capture.get("loading_failed") is True
                and capture.get("intentional_disposal_request_id")
                != capture.get("fetch_request_id")
            )
            or _remaining_browser_timeout(capture_deadline) <= 0
        )

    def process_pending_response() -> None:
        nonlocal pending_response, response_read_in_progress
        if pending_response is None or response_read_in_progress:
            return
        owned_response = pending_response
        pending_response = None
        if is_stopped():
            fail_fetch(
                owned_response.get("fetch_request_id"),
                pause_stage="response",
            )
            capture["finished"] = True
            return
        if (
            capture.get("request_account_hash") != expected_account_hash
            or capture.get("extra_account_hash") != expected_account_hash
            or capture.get("identity_conflict")
        ):
            pending_response = owned_response
            return
        response_read_in_progress = True
        try:
            structural, body_error, response_bytes = (
                _capture_native_history_body_stream(
                    session,
                    capture,
                    fetch_request_id=str(owned_response["fetch_request_id"]),
                    response_headers=owned_response.get("response_headers"),
                    deadline=capture_deadline,
                    max_response_bytes=max_response_bytes,
                    should_stop=response_read_should_stop,
                )
            )
        except Exception:
            structural, body_error, response_bytes = (
                None,
                "response_body_unavailable",
                None,
            )
        finally:
            response_read_in_progress = False
        if response_bytes is not None:
            capture["response_bytes"] = response_bytes
        if structural is not None:
            capture["structural"] = structural
        if body_error is not None and capture.get("body_failure_reason") is None:
            capture["body_failure_reason"] = body_error
        capture["finished"] = True
        if (
            body_error is None
            and not is_stopped()
            and capture.get("response_stream_complete") is True
            and not capture.get("loading_failed")
        ):
            fail_fetch(
                owned_response.get("fetch_request_id"),
                pause_stage="response",
                intentional_disposal=True,
            )
        else:
            fail_fetch(
                owned_response.get("fetch_request_id"),
                pause_stage="response",
            )

    def guard_response(event: Mapping[str, Any]) -> None:
        nonlocal pending_response
        if is_stopped():
            fail_fetch(event.get("requestId"), pause_stage="response")
            return
        request = event.get("request")
        response_url = (
            request.get("url") if isinstance(request, Mapping) else None
        )
        fetch_request_id = event.get("requestId")
        network_request_id = event.get("networkId")
        if not is_history_url(response_url):
            set_boundary("history_response_unowned")
            fail_fetch(fetch_request_id, pause_stage="response")
            return
        if (
            not history_admitted
            or fetch_request_id != capture.get("fetch_request_id")
            or network_request_id != capture.get("network_request_id")
        ):
            set_boundary("history_response_unowned")
            fail_fetch(fetch_request_id, pause_stage="response")
            return
        status = _native_history_status(event.get("responseStatusCode"))
        headers = event.get("responseHeaders")
        if "responseErrorReason" in event:
            store_response(
                request_id=network_request_id,
                response_url=response_url,
                status=status,
                headers=headers,
                classify_boundary=False,
            )
            capture["response_error_observed"] = True
            capture["body_failure_reason"] = "history_response_transport_failed"
            capture["finished"] = True
            set_boundary("history_response_transport_failed")
            fail_fetch(fetch_request_id, pause_stage="response")
            return
        store_response(
            request_id=network_request_id,
            response_url=response_url,
            status=status,
            headers=headers,
        )
        status_failure = _native_history_failure_for_status(status)
        if terminal_failure is not None or boundary_reason is not None:
            fail_fetch(fetch_request_id, pause_stage="response")
            capture["finished"] = True
            return
        if status_failure is not None:
            capture["body_failure_reason"] = status_failure
            fail_fetch(fetch_request_id, pause_stage="response")
            capture["finished"] = True
            return
        if capture.get("content_type") != "json":
            capture["body_failure_reason"] = "non_json_response"
            fail_fetch(fetch_request_id, pause_stage="response")
            capture["finished"] = True
            return
        pending_response = {
            "fetch_request_id": fetch_request_id,
            "network_request_id": network_request_id,
            "response_headers": headers,
        }
        process_pending_response()

    def guard_request(  # noqa: PLR0915 - admission branches share one boundary
        event: Mapping[str, Any],
    ) -> None:
        nonlocal history_admitted, init_fetch_id
        if (
            "responseStatusCode" in event
            or "responseHeaders" in event
            or "responseErrorReason" in event
        ):
            guard_response(event)
            return
        if observer_closed:
            fail_fetch(event.get("requestId"), pause_stage="request")
            return
        if is_stopped():
            fail_fetch(event.get("requestId"), pause_stage="request")
            return
        request = event.get("request")
        request_id = event.get("requestId")
        if not isinstance(request, Mapping):
            set_boundary("malformed_request_event")
            fail_fetch(request_id, pause_stage="request")
            return
        url = request.get("url", "")
        parsed = urlsplit(url if isinstance(url, str) else "")
        method = str(request.get("method") or "").upper()
        redirected = event.get("redirectedRequestId")
        history = is_history_url(url)
        blocked = False
        mutation_block_branch: Optional[str] = None
        if isinstance(init_fetch_id, str) and redirected == init_fetch_id:
            set_boundary("init_request_redirected")
            blocked = True
        if isinstance(request_id, str):
            if request_id in seen_fetch_request_ids:
                set_boundary("fetch_request_replayed")
                blocked = True
            else:
                seen_fetch_request_ids.add(request_id)
        if history:
            network_request_id = event.get("networkId")
            record = record_network_request(
                network_request_id,
                request=request,
                redirected=redirected,
            )
            if (
                history_admitted
                and network_request_id != capture.get("network_request_id")
            ):
                set_boundary("history_request_limit_exceeded")
            elif (
                history_admitted
                and network_request_id == capture.get("network_request_id")
            ):
                set_boundary("history_request_replayed")
            elif len(history_network_ids) > 1:
                set_boundary("history_request_limit_exceeded")
            elif not isinstance(request_id, str):
                set_boundary("history_request_identity_missing")
            elif not isinstance(network_request_id, str):
                set_boundary("history_request_identity_missing")
            elif method != "GET":
                set_boundary("history_request_invalid_method")
            elif redirected is not None or (
                isinstance(record, Mapping) and record.get("redirected")
            ):
                set_boundary("history_request_redirected")
            elif isinstance(record, Mapping) and record.get("identity_conflict"):
                set_boundary("account_identity_mismatch")
            elif terminal_failure is not None:
                set_boundary("terminal_failure")
            elif boundary_reason is not None:
                pass
            else:
                history_admitted = True
                capture.update(
                    request_id=network_request_id,
                    network_request_id=network_request_id,
                    fetch_request_id=request_id,
                    method=method,
                    body_omitted=(
                        not request.get("hasPostData", False)
                        and "postData" not in request
                    ),
                )
                apply_network_record(record)
                if (
                    capture.get("loading_failed")
                    and capture.get("body_failure_reason") is None
                ):
                    capture["body_failure_reason"] = "history_response_failed"
        elif is_init_url(url):
            if method != "POST":
                set_boundary("init_request_invalid_method")
            elif redirected is not None or request.get("redirectResponse"):
                set_boundary("init_request_redirected")
            elif init_fetch_id is not None:
                set_boundary("init_request_limit_exceeded")
            elif not isinstance(request_id, str):
                set_boundary("init_request_identity_missing")
            else:
                init_fetch_id = request_id
        elif is_conversation_detail_path(parsed.path):
            blocked = True
            set_boundary("conversation_detail_blocked")
        elif event.get("resourceType") == "Document":
            home_navigation = (
                parsed.scheme == "https"
                and parsed.netloc == "chatgpt.com"
                and parsed.path == "/"
                and method == "GET"
                and redirected is None
            )
            blocked = not home_navigation
            if not home_navigation:
                set_boundary("unexpected_document_navigation")
        elif method in {"POST", "PUT", "PATCH", "DELETE"}:
            # Reject this request without stopping unrelated permitted reads.
            blocked = True
            mutation_block_branch = "mutating_method"
        elif (
            parsed.netloc == "chatgpt.com"
            and (
                parsed.path.startswith("/backend-api/")
                or parsed.path.startswith("/api/")
            )
            and parsed.path not in _NATIVE_HISTORY_BOOTSTRAP_READ_PATHS
        ):
            blocked = True
            set_boundary("bootstrap_route_blocked")
        elif parsed.path == CHATGPT_NATIVE_HISTORY_INDEX_PATH and method != "GET":
            blocked = True
            set_boundary("history_request_invalid_method")
        if is_model_or_mutation_path(parsed.path) and not history and not is_init_url(
            url
        ):
            blocked = True
            mutation_block_branch = mutation_block_branch or "model_or_mutation_path"
            set_boundary("model_or_mutation_blocked")
        if mutation_block_branch is not None and "blocked_request" not in capture:
            if parsed.scheme == "https" and parsed.netloc == "chatgpt.com":
                origin_category = "chatgpt"
            elif parsed.scheme == "https":
                origin_category = "other_https"
            else:
                origin_category = "other"
            if is_model_or_mutation_path(parsed.path):
                route_family = "model_or_mutation"
            elif parsed.path in _NATIVE_HISTORY_BOOTSTRAP_READ_PATHS:
                route_family = "bootstrap_read"
            elif parsed.path.startswith("/backend-api/"):
                route_family = "other_backend_api"
            elif parsed.path.startswith("/api/"):
                route_family = "other_api"
            else:
                route_family = "other"
            resource_type = event.get("resourceType")
            capture["blocked_request"] = {
                "branch": mutation_block_branch,
                "method": (
                    method
                    if method in {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}
                    else "other"
                ),
                "resource_type": (
                    resource_type
                    if resource_type in ("Document", "XHR", "Fetch", "Ping", "Script")
                    else "other"
                ),
                "origin_category": origin_category,
                "route_family": route_family,
            }
        if is_stopped():
            blocked = True
        if blocked:
            fail_fetch(request_id, pause_stage="request")
        elif not is_stopped():
            continue_fetch(request_id)

    def request_seen(event: Mapping[str, Any]) -> None:
        nonlocal request_count
        request_count = min(
            request_count + 1,
            _NATIVE_HISTORY_MAX_COUNT + 1,
        )
        request = event.get("request")
        if not isinstance(request, Mapping):
            set_boundary("malformed_request_event")
            return
        network_id = event.get("requestId")
        if not isinstance(network_id, str):
            return
        url = request.get("url", "")
        record = record_network_request(
            network_id,
            request=request,
            redirected=event.get("redirectedRequestId"),
        )
        if is_history_url(url):
            if history_admitted and network_id != capture.get("network_request_id"):
                set_boundary("history_request_limit_exceeded")
            elif not history_admitted and len(history_network_ids) > 1:
                set_boundary("history_request_limit_exceeded")
        elif is_init_url(url) and (
            event.get("redirectResponse")
            or event.get("redirectedRequestId") is not None
        ):
            set_boundary("init_request_redirected")
        if (
            isinstance(init_fetch_id, str)
            and event.get("redirectedRequestId") == init_fetch_id
        ):
            set_boundary("init_request_redirected")
        if network_id == capture.get("network_request_id"):
            apply_network_record(record)

    def extra_seen(event: Mapping[str, Any]) -> None:
        request_id = event.get("requestId")
        record = record_network_extra(
            request_id,
            headers=event.get("headers"),
        )
        if record is None:
            return
        if request_id == capture.get("network_request_id"):
            apply_network_record(record)
        if not response_read_in_progress and not is_stopped():
            process_pending_response()

    def response_seen(event: Mapping[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, Mapping):
            set_boundary("malformed_response_event")
            return
        request_id = event.get("requestId")
        response_url = response.get("url")
        headers = response.get("headers")
        status = _native_history_status(response.get("status"))
        record_relevant_response(
            response_url=response_url,
            status=status,
            headers=headers,
        )
        store_response(
            request_id=request_id,
            response_url=response_url,
            status=status,
            headers=headers,
        )

    def loading_finished(event: Mapping[str, Any]) -> None:
        request_id = event.get("requestId")
        store_loading_finished(request_id, event)

    def loading_failed(event: Mapping[str, Any]) -> None:
        request_id = event.get("requestId")
        store_loading_failed(request_id)

    def finalize_capture(failure_reason: Optional[str] = None) -> Mapping[str, Any]:
        if pending_response is not None:
            fail_fetch(
                pending_response.get("fetch_request_id"),
                pause_stage="response",
            )
        if boundary_reason is not None:
            capture["boundary_reason"] = boundary_reason
        if terminal_failure is not None:
            capture["terminal_reason"] = terminal_reason()
        capture["retry_after_seconds"] = retry_after_max
        if isinstance(capture.get("request_id"), str):
            result = _native_history_finalize_observation(
                capture,
                expected_account_hash=expected_account_hash,
                page_target_id_matched=True,
                request_count=request_count,
                history_request_count=history_request_count,
                max_response_bytes=max_response_bytes,
                structural=capture.get("structural"),
                failure_reason=failure_reason,
            )
        else:
            result = _native_history_no_route_observation(
                page_target_id_matched=True,
                request_count=request_count,
                history_request_count=history_request_count,
                failure_reason=(
                    terminal_reason()
                    or boundary_reason
                    or failure_reason
                    or "no_history_observed"
                ),
                browser_challenge=browser_challenge,
                retry_after_seconds=terminal_retry_after(),
            )
        if "blocked_request" in capture:
            result["blocked_request"] = capture["blocked_request"]
        return result
    try:
        session.on("Fetch.requestPaused", guard_request)
        session.on("Network.requestWillBeSent", request_seen)
        session.on("Network.requestWillBeSentExtraInfo", extra_seen)
        session.on("Network.responseReceived", response_seen)
        session.on("Network.loadingFinished", loading_finished)
        session.on("Network.loadingFailed", loading_failed)
        session.send(
            "Network.enable",
            {
                "maxTotalBufferSize": max_response_bytes * 2,
                "maxResourceBufferSize": max_response_bytes,
                "maxPostDataSize": 0,
            },
        )
        session.send("Network.setBypassServiceWorker", {"bypass": True})
        session.send(
            "Fetch.enable",
            {
                "patterns": [
                    {"urlPattern": "*", "requestStage": "Request"},
                    {
                        "urlPattern": "*://chatgpt.com/backend-api/conversations*",
                        "requestStage": "Response",
                    },
                ]
            },
        )
        if is_stopped():
            set_boundary("history_capture_aborted")
        else:
            try:
                page.goto(
                    CHATGPT_NATIVE_HISTORY_HOME_URL,
                    wait_until="commit",
                    timeout=_browser_timeout_milliseconds(
                        _remaining_browser_timeout(capture_deadline)
                    ),
                )
            except Exception:
                set_boundary("home_navigation_failed")
        while True:
            process_pending_response()
            if terminal_failure is not None or boundary_reason is not None:
                return finalize_capture(terminal_reason() or boundary_reason)
            if capture.get("body_failure_reason"):
                return finalize_capture(capture.get("body_failure_reason"))
            if (
                capture.get("response_stream_complete") is True
                and capture.get("extra_seen")
            ):
                return finalize_capture()
            if _remaining_browser_timeout(capture_deadline) <= 0:
                if capture:
                    set_boundary("history_response_timeout")
                    return finalize_capture()
                return finalize_capture("no_history_observed")
            try:
                if is_stopped():
                    continue
                challenge_visible = bool(
                    page.evaluate(
                        "() => [...document.querySelectorAll("
                        "'form#challenge-form[action*=\"__cf_chl\"],"
                        "form#challenge-form[action^=\"/cdn-cgi/challenge-platform/\"],"
                        "#challenge-running')].some(node => {"
                        "const rect = node.getBoundingClientRect();"
                        "return rect.width > 0 && rect.height > 0 &&"
                        "getComputedStyle(node).visibility !== 'hidden';})"
                    )
                )
            except Exception:
                challenge_visible = False
            if challenge_visible:
                latch_terminal("browser_challenge", challenged=True)
                continue
            if is_stopped():
                continue
            page.wait_for_timeout(
                min(
                    50,
                    _browser_timeout_milliseconds(
                        _remaining_browser_timeout(capture_deadline)
                    ),
                )
            )
    finally:
        # Fetch interception remains installed until the owned target closes.
        observer_closed = True
        capture["observer_closed"] = True


def observe_native_chatgpt_history_from_oracle_browser(
    *,
    page_target_id: str,
    cdp_endpoint: Optional[str] = None,
    lifecycle_capability: Optional[NativeHistoryLifecycleCapability] = None,
    expected_account_hash: str = CHATGPT_NATIVE_HISTORY_EXPECTED_ACCOUNT_HASH,
    timeout_seconds: float = CHATGPT_NATIVE_HISTORY_DEFAULT_TIMEOUT_SECONDS,
    max_response_bytes: int = CHATGPT_NATIVE_HISTORY_MAX_RESPONSE_BYTES,
) -> Dict[str, Any]:
    """Observe one native ChatGPT history index request through Oracle CDP.

    The caller supplies an exact existing CDP page target as a context anchor
    and the canonical account hash pin. The observer creates one owned page in
    that context, performs one ordinary ChatGPT home navigation, and returns
    bounded structural metadata. It never issues a direct history request,
    reads browser storage, exports headers, follows pagination, or requests
    conversation details. Native history requires a lifecycle capability from
    the owner that actually launched the private browser; a bare CDP endpoint
    is intentionally insufficient.
    """

    if lifecycle_capability is None or not all(
        callable(getattr(lifecycle_capability, method_name, None))
        for method_name in (
            "register_native_history",
            "register_native_history_closer",
            "retain_native_history",
            "release_native_history",
            "retire_native_history",
            "prepare_native_history_cleanup_plan",
            "terminate_native_history_process_scope",
            "terminate_owned_browser",
            "bind_native_history_endpoint",
        )
    ):
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history requires private browser lifecycle ownership."
        )
    if expected_account_hash != CHATGPT_NATIVE_HISTORY_EXPECTED_ACCOUNT_HASH:
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history requires the pinned inventory account hash."
        )
    if not isinstance(timeout_seconds, (int, float)) or isinstance(
        timeout_seconds,
        bool,
    ):
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history timeout is invalid."
        )
    if (
        not isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
        or timeout_seconds > CHATGPT_NATIVE_HISTORY_DEFAULT_TIMEOUT_SECONDS
    ):
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history timeout exceeds the 150-second limit."
        )
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes <= 0
        or max_response_bytes > CHATGPT_NATIVE_HISTORY_MAX_RESPONSE_BYTES
    ):
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history response budget exceeds the 1 MiB limit."
        )
    target_id = _validate_page_target_id(page_target_id)
    endpoint = cdp_endpoint
    if endpoint is None:
        endpoint = os.getenv(CHATGPT_CONVERSATION_INIT_BROWSER_CDP_ENDPOINT_ENV)
    if endpoint is None:
        endpoint = os.getenv(ORACLE_BROWSER_CDP_ENDPOINT_ENV)
    endpoint = str(endpoint or DEFAULT_ORACLE_BROWSER_CDP_ENDPOINT).strip()
    if not endpoint:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser CDP endpoint is not configured."
        )
    deadline = time.monotonic() + float(timeout_seconds)
    operation_start = time.monotonic()
    target_close_budget = min(
        10.0,
        max(0.0, deadline - operation_start) / 4,
    )
    try:
        lifecycle_capability.bind_native_history_endpoint(
            cdp_endpoint=endpoint,
            anchor_target_id=target_id,
        )
    except Exception as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history lifecycle endpoint binding failed."
        ) from exc
    try:
        return dict(
            _run_oracle_browser_history_observation_in_worker(
                cdp_endpoint=endpoint,
                page_target_id=target_id,
                expected_account_hash=expected_account_hash,
                deadline=deadline,
                operation_start=operation_start,
                target_close_budget=target_close_budget,
                max_response_bytes=max_response_bytes,
                lifecycle_capability=lifecycle_capability,
            )
        )
    except OracleBrowserBoundaryUnavailable:
        raise
    except Exception as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser boundary is unavailable for native ChatGPT history."
        ) from exc


class OracleBrowserConversationInitTransport:
    """Observe native init traffic in a dedicated page of Oracle's context.

    The pinned existing target identifies the context only. This transport
    never navigates that target, launches a browser, reads cookies or storage,
    fabricates an init request, or returns request bodies or credentials.
    """

    boundary_name = ORACLE_BROWSER_BOUNDARY_NAME

    def __init__(
        self,
        *,
        cdp_endpoint: Optional[str] = None,
        page_target_id: str,
        expected_account_hash: str,
        timeout_seconds: float = 30.0,
        playwright_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        endpoint = cdp_endpoint
        if endpoint is None:
            endpoint = os.getenv(CHATGPT_CONVERSATION_INIT_BROWSER_CDP_ENDPOINT_ENV)
        if endpoint is None:
            endpoint = os.getenv(ORACLE_BROWSER_CDP_ENDPOINT_ENV)
        self.cdp_endpoint = (
            str(endpoint or DEFAULT_ORACLE_BROWSER_CDP_ENDPOINT).strip()
        )
        if not self.cdp_endpoint:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser CDP endpoint is not configured."
            )
        if timeout_seconds <= 0:
            raise ValueError("Oracle browser CDP timeout must be greater than 0.")
        self.page_target_id = _validate_page_target_id(page_target_id)
        if not isinstance(expected_account_hash, str) or not re.fullmatch(
            r"[0-9a-f]{12}", expected_account_hash
        ):
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser requires a canonical inventory account hash."
            )
        self.expected_account_hash = expected_account_hash
        self.timeout_seconds = timeout_seconds
        self._playwright_factory = playwright_factory

    def fetch(self, request: urllib_request.Request) -> Mapping[str, Any]:
        _validate_oracle_browser_request(request)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            return _run_oracle_browser_capture_in_worker(
                cdp_endpoint=self.cdp_endpoint,
                page_target_id=self.page_target_id,
                expected_account_hash=self.expected_account_hash,
                request_url=request.full_url,
                deadline=deadline,
                playwright_factory=self._playwright_factory,
            )
        except OracleBrowserBoundaryUnavailable:
            raise
        except ChatGPTConversationInitError:
            raise
        except Exception as exc:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary is unavailable for conversation-init."
            ) from exc

    def _start_playwright(self) -> Any:
        return _start_playwright_from_factory(self._playwright_factory)


def _run_oracle_browser_capture_in_worker(
    *,
    cdp_endpoint: str,
    page_target_id: str,
    expected_account_hash: str,
    request_url: str,
    deadline: float,
    playwright_factory: Optional[Callable[[], Any]],
) -> Mapping[str, Any]:
    context = _oracle_browser_process_context(playwright_factory)
    receiver, sender = context.Pipe(duplex=False)
    private_process_group = context.RawValue("q", 0)
    private_process_start_time = context.RawValue("q", 0)
    owned_target = context.RawArray("c", 256)
    # Publish the state last so a kill cannot expose a partially copied ID.
    creation_state = context.RawValue("b", 0)
    creation_url = "about:blank#oracle-native-init-" + os.urandom(16).hex()
    # Keep cleanup inside the caller's deadline, even if capture is SIGKILLed.
    cleanup_budget = min(3.0, max(0.0, _remaining_browser_timeout(deadline)) / 4)
    capture_deadline = deadline - cleanup_budget
    process = context.Process(
        target=_oracle_browser_capture_worker,
        args=(
            sender,
            cdp_endpoint,
            page_target_id,
            request_url,
            capture_deadline,
            playwright_factory,
            private_process_group,
            private_process_start_time,
            expected_account_hash,
            owned_target,
            creation_state,
            creation_url,
        ),
    )
    try:
        process.start()
    except Exception as exc:
        sender.close()
        receiver.close()
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser boundary worker could not start."
        ) from exc
    sender.close()
    try:
        message = _receive_oracle_browser_worker_message(receiver, capture_deadline)
        remaining_seconds = _remaining_browser_timeout(capture_deadline)
        if remaining_seconds <= 0:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser conversation-init capture timed out."
            )
        process.join(remaining_seconds)
        if process.is_alive():
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser conversation-init cleanup timed out."
            )
        if not isinstance(message, Mapping) or message.get("ok") is not True:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary is unavailable for conversation-init."
            )
        return _coerce_browser_response(message.get("result"))
    finally:
        receiver.close()
        _terminate_oracle_browser_worker(
            process,
            private_process_group.value,
            private_process_start_time,
        )
        process.join(timeout=min(0.1, max(0.0, _remaining_browser_timeout(deadline))))
        target_id = owned_target.value if creation_state.value == 2 else b""
        if creation_state.value:
            _close_owned_oracle_target(
                cdp_endpoint=cdp_endpoint,
                target_id=target_id.decode("ascii") if target_id else None,
                anchor_target_id=page_target_id,
                creation_url=creation_url,
                deadline=deadline,
                playwright_factory=playwright_factory,
            )


def _oracle_browser_capture_worker(
    sender: Any,
    cdp_endpoint: str,
    page_target_id: str,
    request_url: str,
    deadline: float,
    playwright_factory: Optional[Callable[[], Any]],
    private_process_group: Any,
    private_process_start_time: Any,
    expected_account_hash: str,
    owned_target: Any,
    creation_state: Any,
    creation_url: str,
) -> None:
    _enter_oracle_browser_worker_process_group(
        private_process_group,
        private_process_start_time,
    )
    playwright = None
    browser = None
    owned_page = None
    target_session = None
    result = None
    successful = False
    try:
        _raise_if_browser_deadline_expired(deadline)
        playwright = _start_playwright_from_factory(playwright_factory)
        _raise_if_browser_deadline_expired(deadline)
        browser = playwright.chromium.connect_over_cdp(
            cdp_endpoint,
            timeout=_browser_timeout_milliseconds(
                _remaining_browser_timeout(deadline)
            ),
        )
        _raise_if_browser_deadline_expired(deadline)
        source_page = _find_existing_chatgpt_page(
            browser,
            page_target_id,
            request_url,
            deadline=deadline,
        )
        if source_page is None:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser has no exact bound ChatGPT or about:blank "
                "context anchor for the conversation-init request."
            )
        _raise_if_browser_deadline_expired(deadline)
        target_session = browser.new_browser_cdp_session()
        owned_page = _create_owned_oracle_page(
            target_session,
            source_page,
            page_target_id,
            owned_target,
            creation_state,
            creation_url,
            deadline,
        )
        result = _observe_native_oracle_init(
            owned_page,
            session=owned_page.context.new_cdp_session(owned_page),
            request_url=request_url,
            expected_account_hash=expected_account_hash,
            deadline=deadline,
        )
        _raise_if_browser_deadline_expired(deadline)
        successful = True
    except Exception:
        successful = False
    finally:
        if owned_target.value and target_session is not None:
            try:
                closed = target_session.send(
                    "Target.closeTarget",
                    {"targetId": owned_target.value.decode("ascii")},
                )
                if closed.get("success") is not True:
                    raise OracleBrowserBoundaryUnavailable(
                        "Oracle browser did not close its owned target."
                    )
                creation_state.value = 0
                owned_target.value = b""
            except Exception:
                successful = False
        try:
            _disconnect_attached_browser(playwright, browser)
        except Exception:
            successful = False
        try:
            _send_oracle_browser_worker_message(
                sender,
                {
                    "ok": successful,
                    "result": result if successful else None,
                },
            )
        except Exception:
            try:
                _send_oracle_browser_worker_message(
                    sender,
                    {"ok": False, "result": None},
                )
            except Exception:
                pass
        finally:
            sender.close()


def _run_oracle_browser_history_observation_in_worker(  # noqa: PLR0915 - bounded lifecycle supervisor
    *,
    cdp_endpoint: str,
    page_target_id: str,
    expected_account_hash: str,
    deadline: float,
    operation_start: float,
    target_close_budget: float,
    max_response_bytes: int,
    lifecycle_capability: NativeHistoryLifecycleCapability,
) -> Mapping[str, Any]:
    context = _oracle_browser_process_context(None)
    receiver, sender = context.Pipe(duplex=False)
    private_process_group = context.RawValue("q", 0)
    private_process_start_time = context.RawValue("q", 0)
    owned_target = context.RawArray("c", 256)
    creation_state = context.RawValue("b", 0)
    creation_issued = context.RawValue("b", False)
    creation_gate = context.Lock()
    creation_settled = context.Event()
    abort_event = context.Event()
    release_event = context.Event()
    release_control_failed = context.RawValue("b", False)
    creation_url = "about:blank#oracle-native-history-" + os.urandom(16).hex()
    registration_id = "native-history-" + os.urandom(16).hex()
    role_marker = "history-observer:" + registration_id
    target_close_budget = min(
        _NATIVE_HISTORY_CLEANUP_RESERVE_SECONDS,
        max(0.0, target_close_budget),
    )
    target_close_reserve = min(
        target_close_budget,
        max(0.0, deadline - operation_start),
    )
    capture_deadline = max(
        operation_start,
        deadline - target_close_reserve,
    )
    process = context.Process(
        target=_oracle_browser_history_observation_worker,
        args=(
            sender,
            cdp_endpoint,
            page_target_id,
            capture_deadline,
            deadline,
            expected_account_hash,
            max_response_bytes,
            private_process_group,
            private_process_start_time,
            role_marker,
            owned_target,
            creation_state,
            creation_url,
            creation_issued,
            creation_gate,
            creation_settled,
            abort_event,
            release_event,
            release_control_failed,
        ),
    )
    registration = NativeHistoryLifecycleRegistration(
        process=process,
        private_process_group=private_process_group,
        private_process_start_time=private_process_start_time,
        creation_state=creation_state,
        owned_target=owned_target,
        creation_url=creation_url,
        abort_event=abort_event,
        release_event=release_event,
        release_control_state=release_control_failed,
        creation_gate=creation_gate,
        creation_settled=creation_settled,
        creation_issued=creation_issued,
        cdp_endpoint=cdp_endpoint,
        anchor_target_id=page_target_id,
        deadline=deadline,
        operation_start=operation_start,
        target_close_budget=target_close_budget,
        finalization_gate=context.Lock(),
        registration_id=registration_id,
    )
    registration.cleanup_callback = lambda cleanup_deadline, poll_only=False: (
        _finalize_native_history_registration(
            registration,
            lifecycle_capability,
            cleanup_deadline=cleanup_deadline,
            operation_deadline=deadline,
            poll_only=poll_only,
        )
    )
    try:
        # The owner must know every shared handle before a child can publish
        # browser or target state.
        lifecycle_capability.register_native_history(registration)
    except Exception as exc:
        sender.close()
        receiver.close()
        raise OracleBrowserCleanupError(
            "Native ChatGPT history lifecycle registration failed."
        ) from exc
    operation_error: Optional[Exception] = None
    result: Optional[Mapping[str, Any]] = None
    try:
        registration.start_state = "starting"
        process.start()
        registration.start_state = "started"
    except Exception as exc:
        registration.start_state = "failed"
        operation_error = OracleBrowserBoundaryUnavailable(
            "Oracle browser history observer worker could not start."
        )
        operation_error.__cause__ = exc
    sender.close()
    try:
        if operation_error is None:
            try:
                message = _receive_oracle_browser_worker_message(
                    receiver,
                    capture_deadline,
                    should_stop=registration.abort_event.is_set,
                )
                if not isinstance(message, Mapping) or message.get("ok") is not True:
                    raise OracleBrowserBoundaryUnavailable(
                        "Oracle browser history observer is unavailable."
                    )
                candidate = message.get("result")
                if not isinstance(candidate, Mapping):
                    raise OracleBrowserBoundaryUnavailable(
                        "Oracle browser history observer returned an invalid result."
                    )
                result = dict(candidate)
            except Exception as exc:
                operation_error = exc
    finally:
        receiver.close()

    cleanup_error: Optional[Exception] = None
    try:
        if not _finalize_native_history_registration(
            registration,
            lifecycle_capability,
            cleanup_deadline=deadline,
            operation_deadline=deadline,
        ):
            cleanup_error = OracleBrowserCleanupError(
                registration.cleanup_failure
                or "Native ChatGPT history cleanup could not be proven."
            )
        elif registration.cleanup_failure is not None:
            cleanup_error = OracleBrowserCleanupError(
                registration.cleanup_failure
            )
    except Exception as exc:
        cleanup_error = (
            exc
            if isinstance(exc, OracleBrowserCleanupError)
            else OracleBrowserCleanupError(
                "Native ChatGPT history cleanup failed."
            )
        )
        if cleanup_error is not exc:
            cleanup_error.__cause__ = exc
    if cleanup_error is not None:
        raise cleanup_error
    if operation_error is not None:
        raise operation_error
    if result is None:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser history observer returned no result."
        )
    return result


def _native_history_mark_no_create(
    *,
    creation_state: Any,
    creation_issued: Any,
    creation_settled: Any,
) -> bool:
    """Publish an explicit no-create acknowledgement, never infer it from NONE."""

    if bool(creation_issued.value):
        return False
    creation_state.value = _NATIVE_HISTORY_TARGET_NO_CREATE
    creation_settled.set()
    return True


def _native_history_wait_for_release(
    release_event: Any,
) -> bool:
    while True:
        try:
            if release_event.wait(timeout=0.05):
                return True
        except (OSError, ValueError):
            return False


def _native_history_cleanup_phase_plan(
    registration: NativeHistoryLifecycleRegistration,
    cleanup_deadline: float,
    operation_deadline: float,
) -> Dict[str, float]:
    """Allocate one immutable cleanup plan and only clamp it earlier."""

    ceiling = min(
        float(registration.deadline),
        float(operation_deadline),
        float(cleanup_deadline),
    )
    if registration.shutdown_deadline is not None:
        ceiling = min(ceiling, float(registration.shutdown_deadline))
    now = time.monotonic()
    if registration.cleanup_plan is None:
        start = min(now, ceiling)
        reserve = min(
            _NATIVE_HISTORY_CLEANUP_RESERVE_SECONDS,
            max(0.0, registration.target_close_budget),
            max(0.0, ceiling - start),
        )
        end = start + reserve
        registration.cleanup_plan = {
            "target_close_deadline": start + reserve / 3,
            "term_deadline": start + reserve / 2,
            "kill_deadline": start + reserve * 2 / 3,
            "reap_deadline": start + reserve * 5 / 6,
            "final_deadline": end,
        }
    else:
        for phase_name, phase_deadline in list(registration.cleanup_plan.items()):
            registration.cleanup_plan[phase_name] = min(
                phase_deadline,
                ceiling,
            )
    return registration.cleanup_plan


def _native_history_process_start_time(pid: Any) -> Optional[int]:
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return None
    if os.name != "posix" or process_id <= 0:
        return None
    try:
        fields = Path(f"/proc/{process_id}/stat").read_text().rsplit(
            ")",
            1,
        )[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _native_history_process_identity_matches(
    pid: Any,
    start_time: Any,
) -> bool:
    try:
        expected = int(getattr(start_time, "value", start_time) or 0)
    except (TypeError, ValueError):
        return False
    return expected > 0 and _native_history_process_start_time(pid) == expected


def _native_history_process_group_member_count(
    group_id: int,
    *,
    deadline: Optional[float] = None,
) -> Optional[int]:
    if os.name != "posix" or group_id <= 0:
        return 0
    count = 0
    inspected = 0
    inspection_failed = False
    if deadline is not None and time.monotonic() >= deadline:
        return None
    try:
        entries = Path("/proc").iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return None
            inspected += 1
            if inspected > _NATIVE_HISTORY_MAX_PROCESS_INVENTORY_ENTRIES:
                return None
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group_id:
                    count += 1
            except (OSError, ValueError, IndexError):
                inspection_failed = True
    except OSError:
        return None
    if inspection_failed:
        return None
    return count


def _native_history_process_group_alive(
    group: Any,
    process: Any,
    private_process_start_time: Any,
    *,
    deadline: Optional[float] = None,
) -> Optional[bool]:
    try:
        group_id = int(getattr(group, "value", group) or 0)
    except (TypeError, ValueError):
        return False
    process_id = getattr(process, "pid", None)
    if os.name != "posix" or group_id <= 0:
        return False
    identity_matches = (
        group_id == process_id
        and _native_history_process_identity_matches(
            group_id,
            private_process_start_time,
        )
    )
    if not identity_matches:
        member_count = _native_history_process_group_member_count(
            group_id,
            deadline=deadline,
        )
        if member_count == 0:
            return False
        return None
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _signal_native_history_worker(
    process: Any,
    private_process_group: Any,
    private_process_start_time: Any,
    signal_number: int,
) -> None:
    try:
        group_id = int(getattr(private_process_group, "value", private_process_group) or 0)
    except (TypeError, ValueError):
        group_id = 0
    if (
        os.name == "posix"
        and group_id > 0
        and group_id != os.getpgrp()
        and _native_history_process_identity_matches(
            getattr(process, "pid", None),
            private_process_start_time,
        )
    ):
        try:
            os.killpg(group_id, signal_number)
        except OSError:
            pass
    try:
        if _native_history_process_alive(process):
            process.send_signal(signal_number)
    except AttributeError:
        try:
            if signal_number == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()
        except (AttributeError, OSError):
            pass
    except OSError:
        pass


def _native_history_process_alive(process: Any) -> bool:
    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return False


def _native_history_process_reaped(
    process: Any,
    private_process_group: Any,
    private_process_start_time: Any,
    *,
    term_deadline: float,
    kill_deadline: float,
    reap_deadline: float,
    poll_only: bool = False,
    scope_reaped_state: Optional[Dict[str, bool]] = None,
) -> bool:
    """Retire a worker tree without borrowing a later phase."""

    def direct_child_reaped() -> bool:
        if getattr(process, "pid", None) is None:
            return True
        try:
            process.join(timeout=0)
        except (AssertionError, OSError):
            return False
        return (
            not _native_history_process_alive(process)
            and getattr(process, "exitcode", None) is not None
        )

    def scope_reaped() -> bool:
        # A capability-level scope proof covers descendants, but never
        # substitutes for reaping this directly owned multiprocessing child.
        if not direct_child_reaped():
            return False
        if (
            scope_reaped_state is not None
            and scope_reaped_state.get("proven") is True
        ):
            return True
        try:
            group_id = int(
                getattr(private_process_group, "value", private_process_group)
                or 0
            )
        except (TypeError, ValueError):
            group_id = 0
        if getattr(process, "pid", None) is None:
            if group_id <= 0:
                if scope_reaped_state is not None:
                    scope_reaped_state["proven"] = True
                return True
            if poll_only:
                return False
            scope_alive = _native_history_process_group_alive(
                private_process_group,
                process,
                private_process_start_time,
                deadline=(
                    time.monotonic()
                    if poll_only
                    else reap_deadline
                ),
            )
            if scope_alive is False and scope_reaped_state is not None:
                scope_reaped_state["proven"] = True
            return scope_alive is False
        if poll_only and group_id > 0:
            # A post-leader group scan is a blocking inventory operation. In
            # poll-only servicing, rely on prior sealed scope evidence or the
            # sidecar's retained role pidfds instead of inventing proof.
            return False
        scope_alive = _native_history_process_group_alive(
            private_process_group,
            process,
            private_process_start_time,
            deadline=(
                time.monotonic()
                if poll_only
                else reap_deadline
            ),
        )
        if scope_alive is False and scope_reaped_state is not None:
            scope_reaped_state["proven"] = True
        return scope_alive is False

    if scope_reaped():
        return True
    now = time.monotonic()
    if now < term_deadline:
        _signal_native_history_worker(
            process,
            private_process_group,
            private_process_start_time,
            signal.SIGTERM,
        )
        if poll_only:
            return scope_reaped()
        try:
            process.join(timeout=max(0.0, term_deadline - time.monotonic()))
        except (AssertionError, OSError):
            return False
        if scope_reaped():
            return True
    if poll_only:
        # Poll-only servicing may not restart the TERM grace window. Once its
        # cutoff passes, escalate immediately and only observe retirement.
        _signal_native_history_worker(
            process,
            private_process_group,
            private_process_start_time,
            signal.SIGKILL,
        )
        return scope_reaped()
    _signal_native_history_worker(
        process,
        private_process_group,
        private_process_start_time,
        signal.SIGKILL,
    )
    if not poll_only and time.monotonic() < kill_deadline:
        try:
            process.join(timeout=max(0.0, kill_deadline - time.monotonic()))
        except (AssertionError, OSError):
            return False
    if scope_reaped():
        return True
    if not poll_only and time.monotonic() < reap_deadline:
        try:
            process.join(timeout=max(0.0, reap_deadline - time.monotonic()))
        except (AssertionError, OSError):
            return False
    return scope_reaped()


def _finalize_native_history_registration(  # noqa: PLR0915 - bounded lifecycle finalizer
    registration: NativeHistoryLifecycleRegistration,
    lifecycle_capability: NativeHistoryLifecycleCapability,
    *,
    cleanup_deadline: float,
    operation_deadline: float,
    poll_only: bool = False,
) -> bool:
    """Close, authorize release, and retire one history operation."""

    try:
        registration.abort_event.set()
    except (OSError, ValueError):
        registration.cleanup_failure = (
            "Native ChatGPT history abort control failed."
        )
    gate = registration.finalization_gate
    gate_acquired = False
    if gate is not None:
        try:
            gate_acquired = bool(gate.acquire(False))
            if not gate_acquired:
                return False
        except (TypeError, OSError, ValueError):
            registration.cleanup_failure = (
                "Native ChatGPT history finalization control failed."
            )
            try:
                lifecycle_capability.retain_native_history(
                    registration,
                    registration.cleanup_failure,
                )
            except Exception:
                pass
            return False
    try:
        if registration.start_state == "starting":
            registration.cleanup_failure = (
                "Native ChatGPT history worker start is still in progress."
            )
            lifecycle_capability.retain_native_history(
                registration,
                registration.cleanup_failure,
            )
            return False

        lifecycle_capability.prepare_native_history_cleanup_plan(
            registration,
            cleanup_deadline=cleanup_deadline,
            operation_deadline=operation_deadline,
        )
        plan = _native_history_cleanup_phase_plan(
            registration,
            cleanup_deadline,
            operation_deadline,
        )
        state = int(registration.creation_state.value)
        close_registration = registration.close_registration

        def promote_closer_proof() -> None:
            nonlocal state
            closer = registration.close_registration
            if closer is None or not bool(closer.target_proof.value):
                return
            registration.target_resolution = NativeHistoryTargetProof(
                target_id=closer.target_id,
                anchor_target_id=registration.anchor_target_id,
                creation_url=registration.creation_url,
            )
            registration.creation_state.value = _NATIVE_HISTORY_TARGET_CLOSED
            state = _NATIVE_HISTORY_TARGET_CLOSED

        # A closer may have published target safety before its own driver was
        # reaped. Promote that proof once and never create a second closer for
        # the same target.
        promote_closer_proof()

        if registration.target_resolution is None:
            plan = _native_history_cleanup_phase_plan(
                registration,
                cleanup_deadline,
                operation_deadline,
            )
            if (
                state == _NATIVE_HISTORY_TARGET_NONE
                and registration.start_state in {"not_started", "failed"}
                and not bool(registration.creation_issued.value)
            ):
                _native_history_mark_no_create(
                    creation_state=registration.creation_state,
                    creation_issued=registration.creation_issued,
                    creation_settled=registration.creation_settled,
                )
                state = _NATIVE_HISTORY_TARGET_NO_CREATE
                registration.target_resolution = NativeHistoryTargetProof(
                    target_id=None,
                    anchor_target_id=registration.anchor_target_id,
                    creation_url=registration.creation_url,
                )
            elif state == _NATIVE_HISTORY_TARGET_NO_CREATE:
                registration.target_resolution = NativeHistoryTargetProof(
                    target_id=None,
                    anchor_target_id=registration.anchor_target_id,
                    creation_url=registration.creation_url,
                )
            else:
                if not poll_only and time.monotonic() < plan["target_close_deadline"]:
                    target_bytes = registration.owned_target.value
                    target_id = None
                    if target_bytes:
                        try:
                            target_id = target_bytes.decode("ascii")
                        except (UnicodeDecodeError, AttributeError):
                            target_id = None
                    try:
                        proof = _close_owned_oracle_target(
                            cdp_endpoint=registration.cdp_endpoint,
                            target_id=target_id,
                            anchor_target_id=registration.anchor_target_id,
                            creation_url=registration.creation_url,
                            deadline=plan["target_close_deadline"],
                            playwright_factory=None,
                            lifecycle_capability=lifecycle_capability,
                            lifecycle_registration=registration,
                        )
                        registration.target_resolution = proof
                        registration.creation_state.value = (
                            _NATIVE_HISTORY_TARGET_CLOSED
                        )
                        state = _NATIVE_HISTORY_TARGET_CLOSED
                    except Exception as exc:
                        registration.cleanup_failure = str(exc)
                    promote_closer_proof()
                    close_registration = registration.close_registration

        close_registration = registration.close_registration

        if registration.target_resolution is None:
            plan = _native_history_cleanup_phase_plan(
                registration,
                cleanup_deadline,
                operation_deadline,
            )
            try:
                if lifecycle_capability.terminate_owned_browser(
                    term_deadline=plan["term_deadline"],
                    kill_deadline=plan["kill_deadline"],
                    reap_deadline=plan["reap_deadline"],
                    final_deadline=plan["final_deadline"],
                    poll_only=poll_only,
                ):
                    registration.creation_state.value = (
                        _NATIVE_HISTORY_TARGET_BROWSER_TERMINATED
                    )
                    state = _NATIVE_HISTORY_TARGET_BROWSER_TERMINATED
            except Exception as exc:
                registration.cleanup_failure = str(exc)

        if registration.target_resolution is None and state != (
            _NATIVE_HISTORY_TARGET_BROWSER_TERMINATED
        ):
            registration.cleanup_failure = (
                registration.cleanup_failure
                or "Native ChatGPT history target ownership could not be closed."
            )
            lifecycle_capability.retain_native_history(
                registration,
                registration.cleanup_failure,
            )
            return False

        if not registration.release_authorized:
            if state == _NATIVE_HISTORY_TARGET_BROWSER_TERMINATED:
                proof_kind = "owned_browser_termination_proven"
                target_id = None
            elif state == _NATIVE_HISTORY_TARGET_NO_CREATE:
                proof_kind = "no_create_acknowledged"
                target_id = None
            else:
                proof_kind = "target_close_and_absence_proven"
                target_id = (
                    registration.target_resolution.target_id
                    if registration.target_resolution is not None
                    else None
                )
            release_proof = NativeHistoryReleaseProof(
                registration_id=registration.registration_id,
                kind=proof_kind,
                target_id=target_id,
                anchor_target_id=registration.anchor_target_id,
                creation_url=registration.creation_url,
            )
            lifecycle_capability.release_native_history(
                registration,
                proof=release_proof,
            )
            registration.release_authorized = True
            registration.released = True
            registration.release_proof = release_proof

        def worker_tree_reaped(
            process: Any,
            private_process_group: Any,
            private_process_start_time: Any,
            *,
            phase_poll_only: bool,
        ) -> bool:
            close_scope = registration.close_registration
            is_closer = (
                close_scope is not None
                and process is close_scope.process
            )
            scope_reaped_state = {
                "proven": (
                    close_scope.scope_reaped_proven
                    if is_closer and close_scope is not None
                    else registration.worker_scope_reaped_proven
                ),
            }
            try:
                scope_proof = bool(
                    lifecycle_capability.terminate_native_history_process_scope(
                        registration_id=registration.registration_id,
                        term_deadline=plan["term_deadline"],
                        kill_deadline=plan["kill_deadline"],
                        reap_deadline=plan["reap_deadline"],
                        poll_only=phase_poll_only,
                    )
                )
            except Exception as exc:
                registration.cleanup_failure = str(exc)
                return False
            if scope_proof:
                # Retain the actual owner result even when the directly owned
                # multiprocessing child still needs a later join.
                scope_reaped_state["proven"] = True
                if is_closer and close_scope is not None:
                    close_scope.scope_reaped_proven = True
                else:
                    registration.worker_scope_reaped_proven = True
            if _native_history_process_reaped(
                process,
                private_process_group,
                private_process_start_time,
                term_deadline=plan["term_deadline"],
                kill_deadline=plan["kill_deadline"],
                reap_deadline=plan["reap_deadline"],
                poll_only=phase_poll_only,
                scope_reaped_state=scope_reaped_state,
            ):
                if is_closer and close_scope is not None:
                    close_scope.scope_reaped_proven = bool(
                        scope_reaped_state["proven"]
                    )
                else:
                    registration.worker_scope_reaped_proven = bool(
                        scope_reaped_state["proven"]
                    )
                return True
            # The browser driver can outlive its multiprocessing leader. Give
            # the owning sidecar a chance to signal its retained role-marked
            # pidfds before declaring the worker tree unreaped.
            try:
                lifecycle_capability.terminate_owned_browser(
                    term_deadline=plan["term_deadline"],
                    kill_deadline=plan["kill_deadline"],
                    reap_deadline=plan["reap_deadline"],
                    final_deadline=plan["final_deadline"],
                    poll_only=phase_poll_only,
                )
            except Exception as exc:
                registration.cleanup_failure = str(exc)
                return False
            reaped = _native_history_process_reaped(
                process,
                private_process_group,
                private_process_start_time,
                term_deadline=plan["term_deadline"],
                kill_deadline=plan["kill_deadline"],
                reap_deadline=plan["reap_deadline"],
                poll_only=True,
                scope_reaped_state=scope_reaped_state,
            )
            if is_closer and close_scope is not None:
                close_scope.scope_reaped_proven = bool(
                    scope_reaped_state["proven"]
                )
            else:
                registration.worker_scope_reaped_proven = bool(
                    scope_reaped_state["proven"]
                )
            return reaped

        close_registration = registration.close_registration
        if close_registration is not None and not close_registration.reaped:
            plan = _native_history_cleanup_phase_plan(
                registration,
                cleanup_deadline,
                operation_deadline,
            )
            if not worker_tree_reaped(
                close_registration.process,
                close_registration.private_process_group,
                close_registration.private_process_start_time,
                phase_poll_only=poll_only,
            ):
                registration.cleanup_failure = (
                    "Native ChatGPT history closer was not reaped."
                )
                lifecycle_capability.retain_native_history(
                    registration,
                    registration.cleanup_failure,
                )
                return False
            close_registration.reaped = True
            close_registration.reap_ack.set()

        plan = _native_history_cleanup_phase_plan(
            registration,
            cleanup_deadline,
            operation_deadline,
        )
        if not worker_tree_reaped(
            registration.process,
            registration.private_process_group,
            registration.private_process_start_time,
            phase_poll_only=poll_only,
        ):
            registration.cleanup_failure = (
                "Native ChatGPT history interception worker was not reaped."
            )
            lifecycle_capability.retain_native_history(
                registration,
                registration.cleanup_failure,
            )
            return False
        registration.worker_retirement_proven = True
        release_control_state = registration.release_control_state
        registration.release_control_failed = bool(
            registration.release_control_failed
            or (
                release_control_state is not None
                and bool(release_control_state.value)
            )
        )
        if registration.release_control_failed:
            registration.cleanup_failure = (
                "Native ChatGPT history release control failed."
            )
            lifecycle_capability.retain_native_history(
                registration,
                registration.cleanup_failure,
            )
        registration.owned_target.value = b""
        lifecycle_capability.retire_native_history(registration)
        registration.retired = True
        return True
    except Exception as exc:
        registration.cleanup_failure = str(exc)
        lifecycle_capability.retain_native_history(
            registration,
            registration.cleanup_failure,
        )
        return False
    finally:
        if gate is not None and gate_acquired:
            try:
                gate.release()
            except (RuntimeError, OSError, ValueError):
                pass


def _oracle_browser_history_observation_worker(  # noqa: PLR0915 - bounded cleanup handshake
    sender: Any,
    cdp_endpoint: str,
    page_target_id: str,
    capture_deadline: float,
    cleanup_deadline: float,
    expected_account_hash: str,
    max_response_bytes: int,
    private_process_group: Any,
    private_process_start_time: Any,
    role_marker: str,
    owned_target: Any,
    creation_state: Any,
    creation_url: str,
    creation_issued: Any,
    creation_gate: Any,
    creation_settled: Any,
    abort_event: Any,
    release_event: Any,
    release_control_failed: Any,
) -> None:
    _enter_oracle_browser_worker_process_group(
        private_process_group,
        private_process_start_time,
    )
    # Install the role marker before Playwright launches any driver process.
    # The marker is inherited by the actual browser driver and lets the owner
    # retain its pidfd after this worker exits.
    os.environ[CHATGPT_NATIVE_HISTORY_ROLE_ENV] = role_marker
    playwright = None
    browser = None
    target_session = None
    result = None
    successful = False
    try:
        _raise_if_browser_deadline_expired(capture_deadline)
        if abort_event.is_set():
            raise OracleBrowserBoundaryUnavailable(
                "Native ChatGPT history was aborted before browser attachment."
            )
        playwright = _start_playwright_from_factory(None)
        _raise_if_browser_deadline_expired(capture_deadline)
        browser = playwright.chromium.connect_over_cdp(
            cdp_endpoint,
            timeout=_browser_timeout_milliseconds(
                _remaining_browser_timeout(capture_deadline)
            ),
        )
        _raise_if_browser_deadline_expired(capture_deadline)
        source_page = _find_existing_chatgpt_page(
            browser,
            page_target_id,
            CHATGPT_NATIVE_HISTORY_HOME_URL,
            deadline=capture_deadline,
        )
        if source_page is None:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser has no exact bound ChatGPT context anchor "
                "for native history."
            )
        target_session = browser.new_browser_cdp_session()
        owned_page = _create_native_history_owned_page(
            target_session,
            source_page,
            page_target_id,
            owned_target,
            creation_state,
            creation_url,
            capture_deadline,
            creation_issued,
            creation_gate,
            creation_settled,
            abort_event,
        )
        if abort_event.is_set():
            raise OracleBrowserBoundaryUnavailable(
                "Native ChatGPT history was aborted before navigation."
            )
        result = _observe_native_history_oracle_page(
            owned_page,
            session=owned_page.context.new_cdp_session(owned_page),
            expected_account_hash=expected_account_hash,
            deadline=capture_deadline,
            max_response_bytes=max_response_bytes,
            abort_event=abort_event,
        )
        _raise_if_browser_deadline_expired(capture_deadline)
        successful = not abort_event.is_set()
    except Exception:
        successful = False
    finally:
        if (
            not bool(creation_issued.value)
            and creation_state.value == _NATIVE_HISTORY_TARGET_NONE
        ):
            _native_history_mark_no_create(
                creation_state=creation_state,
                creation_issued=creation_issued,
                creation_settled=creation_settled,
            )
        elif (
            creation_state.value == _NATIVE_HISTORY_TARGET_CREATING
            and creation_settled.is_set()
        ):
            creation_state.value = _NATIVE_HISTORY_TARGET_CLEANUP_PENDING
        if abort_event.is_set():
            successful = False
        if (
            not bool(creation_issued.value)
            and creation_state.value == _NATIVE_HISTORY_TARGET_NONE
        ):
            _native_history_mark_no_create(
                creation_state=creation_state,
                creation_issued=creation_issued,
                creation_settled=creation_settled,
            )
        try:
            _send_oracle_browser_worker_message(
                sender,
                {
                    "ok": successful,
                    "result": result if successful else None,
                },
            )
        except Exception:
            try:
                _send_oracle_browser_worker_message(
                    sender,
                    {"ok": False, "result": None},
                )
            except Exception:
                pass
        finally:
            sender.close()
        # Keep Fetch interception and the attached browser alive until the
        # owner proves target closure or terminates its private browser.
        if _native_history_wait_for_release(release_event):
            try:
                _disconnect_attached_browser(playwright, browser)
            except Exception:
                pass
        else:
            # A broken release channel cannot authorize detachment. Keep the
            # interception session and browser references parked until the
            # owner proves target safety and retires this worker directly.
            release_control_failed.value = True
            while True:
                time.sleep(0.05)


def _create_native_history_owned_page(
    target_session: Any,
    source_page: Any,
    anchor_target_id: str,
    owned_target: Any,
    creation_state: Any,
    creation_url: str,
    deadline: float,
    creation_issued: Any,
    creation_gate: Any,
    creation_settled: Any,
    abort_event: Any,
) -> Any:
    anchor = target_session.send(
        "Target.getTargetInfo", {"targetId": anchor_target_id}
    )["targetInfo"]
    create_options = {"url": creation_url}
    if anchor.get("browserContextId"):
        create_options["browserContextId"] = anchor["browserContextId"]
    gate_timeout = max(0.0, _remaining_browser_timeout(deadline))
    acquired = creation_gate.acquire(timeout=gate_timeout)
    if not acquired:
        raise OracleBrowserBoundaryUnavailable(
            "Native ChatGPT history target-creation gate timed out."
        )
    try:
        if abort_event.is_set():
            _native_history_mark_no_create(
                creation_state=creation_state,
                creation_issued=creation_issued,
                creation_settled=creation_settled,
            )
            raise OracleBrowserBoundaryUnavailable(
                "Native ChatGPT history was aborted before target creation."
            )
        creation_state.value = _NATIVE_HISTORY_TARGET_CREATING
        creation_issued.value = True
        try:
            with source_page.context.expect_page(
                predicate=lambda page: page.url == creation_url,
                timeout=_browser_timeout_milliseconds(
                    _remaining_browser_timeout(deadline)
                ),
            ) as page_event:
                created = target_session.send(
                    "Target.createTarget",
                    create_options,
                )
                target_id = _validate_page_target_id(created["targetId"])
                if target_id == anchor_target_id:
                    raise OracleBrowserBoundaryUnavailable(
                        "Oracle browser returned the context anchor as an owned target."
                    )
                owned_target.value = target_id.encode("ascii")
                creation_state.value = _NATIVE_HISTORY_TARGET_ATTACHED
        finally:
            creation_settled.set()
    finally:
        creation_gate.release()
    candidate = page_event.value
    if _page_target_id(source_page.context, candidate) != target_id:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser owned page did not match its created target."
        )
    return candidate


def _create_owned_oracle_page(
    target_session: Any,
    source_page: Any,
    anchor_target_id: str,
    owned_target: Any,
    creation_state: Any,
    creation_url: str,
    deadline: float,
) -> Any:
    anchor = target_session.send(
        "Target.getTargetInfo", {"targetId": anchor_target_id}
    )["targetInfo"]
    create_options = {"url": creation_url}
    if anchor.get("browserContextId"):
        create_options["browserContextId"] = anchor["browserContextId"]
    # Publish ownership before waiting for Playwright's Page or starting capture.
    with source_page.context.expect_page(
        predicate=lambda page: page.url == creation_url,
        timeout=_browser_timeout_milliseconds(_remaining_browser_timeout(deadline))
    ) as page_event:
        creation_state.value = _NATIVE_HISTORY_TARGET_CREATING
        created = target_session.send("Target.createTarget", create_options)
        target_id = _validate_page_target_id(created["targetId"])
        if target_id == anchor_target_id:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser returned the context anchor as an owned target."
            )
        owned_target.value = target_id.encode("ascii")
        creation_state.value = _NATIVE_HISTORY_TARGET_ATTACHED
    candidate = page_event.value
    if _page_target_id(source_page.context, candidate) != target_id:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser owned page did not match its created target."
        )
    return candidate


def _close_owned_oracle_target(  # noqa: PLR0915 - bounded target cleanup
    *,
    cdp_endpoint: str,
    target_id: Optional[str],
    anchor_target_id: str,
    creation_url: str,
    deadline: float,
    playwright_factory: Optional[Callable[[], Any]],
    lifecycle_capability: Optional[NativeHistoryLifecycleCapability] = None,
    lifecycle_registration: Optional[NativeHistoryLifecycleRegistration] = None,
) -> NativeHistoryTargetProof:
    """Give exact-target cleanup its own bounded, killable driver."""
    if lifecycle_registration is not None:
        existing = lifecycle_registration.close_registration
        if existing is not None:
            if bool(existing.target_proof.value):
                return NativeHistoryTargetProof(
                    target_id=existing.target_id,
                    anchor_target_id=anchor_target_id,
                    creation_url=creation_url,
                )
            raise OracleBrowserCleanupError(
                "Oracle browser owned-target closer is already registered."
            )
    _raise_if_browser_deadline_expired(deadline)
    if target_id == anchor_target_id:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser cleanup cannot close the context anchor."
        )
    if target_id is not None:
        _validate_page_target_id(target_id)
    _raise_if_browser_deadline_expired(deadline)
    context = _oracle_browser_process_context(playwright_factory)
    private_process_group = context.RawValue("q", 0)
    private_process_start_time = context.RawValue("q", 0)
    target_proof = context.RawValue("b", False)
    failure_stage = context.RawValue("i", _OracleTargetCloserStage.UNKNOWN)
    driver_done = context.Event()
    reap_ack = context.Event()
    role_marker = (
        "history-closer:"
        + (
            lifecycle_registration.registration_id
            if lifecycle_registration is not None
            else os.urandom(16).hex()
        )
    )
    process = context.Process(
        target=_oracle_browser_close_target_worker,
        args=(
            cdp_endpoint,
            target_id,
            anchor_target_id,
            creation_url,
            deadline,
            playwright_factory,
            private_process_group,
            private_process_start_time,
            role_marker,
            target_proof,
            driver_done,
            lifecycle_registration is not None,
            lifecycle_registration is not None,
            failure_stage,
        ),
    )
    close_registration: Optional[NativeHistoryCloseRegistration] = None
    if lifecycle_capability is not None and lifecycle_registration is not None:
        close_registration = NativeHistoryCloseRegistration(
            process=process,
            private_process_group=private_process_group,
            private_process_start_time=private_process_start_time,
            target_proof=target_proof,
            driver_done=driver_done,
            reap_ack=reap_ack,
            target_id=target_id,
            creation_url=creation_url,
            failure_stage=failure_stage,
        )
        lifecycle_capability.register_native_history_closer(
            lifecycle_registration,
            close_registration,
        )
    try:
        if close_registration is not None:
            close_registration.start_state = "starting"
        process.start()
        if close_registration is not None:
            close_registration.start_state = "started"
            close_registration.started = True
    except Exception as exc:
        if close_registration is not None:
            close_registration.start_state = "failed"
        raise OracleBrowserCleanupError(
            "Oracle browser owned-target closer could not start."
        ) from exc
    scope_reaped_proven = False

    def closer_scope_reaped() -> bool:
        nonlocal scope_reaped_proven
        if scope_reaped_proven:
            return True
        if (
            _native_history_process_alive(process)
            or getattr(process, "exitcode", None) is None
        ):
            return False
        scope_alive = _native_history_process_group_alive(
            private_process_group,
            process,
            private_process_start_time,
            deadline=deadline,
        )
        if scope_alive is False:
            scope_reaped_proven = True
            if close_registration is not None:
                close_registration.scope_reaped_proven = True
            return True
        return False

    try:
        process.join(timeout=max(0.0, _remaining_browser_timeout(deadline)))
        if (
            not closer_scope_reaped()
        ):
            _terminate_oracle_browser_worker(
                process,
                private_process_group.value,
                private_process_start_time,
            )
            process.join(timeout=max(0.0, _remaining_browser_timeout(deadline)))
        if (
            not closer_scope_reaped()
        ):
            if (
                close_registration is not None
                and bool(close_registration.target_proof.value)
            ):
                raise OracleBrowserCleanupError(
                    "Oracle browser target was closed but its closer remains active."
                )
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser owned-target closer was not reaped."
            )
        if close_registration is not None:
            close_registration.reaped = True
            reap_ack.set()
        if not target_proof.value:
            raise OracleBrowserBoundaryUnavailable(
                _ORACLE_TARGET_CLOSER_FAILURE_MESSAGES.get(
                    failure_stage.value,
                    _ORACLE_TARGET_CLOSER_FAILURE_MESSAGES[
                        _OracleTargetCloserStage.UNKNOWN
                    ],
                )
            )
        return NativeHistoryTargetProof(
            target_id=target_id,
            anchor_target_id=anchor_target_id,
            creation_url=creation_url,
        )
    finally:
        if not closer_scope_reaped():
            _terminate_oracle_browser_worker(
                process,
                private_process_group.value,
                private_process_start_time,
            )
            try:
                process.join(
                    timeout=max(0.0, _remaining_browser_timeout(deadline))
                )
            except (AssertionError, OSError):
                pass
        if (
            closer_scope_reaped()
            and close_registration is not None
        ):
            close_registration.reaped = True
            reap_ack.set()


def _oracle_browser_close_target_worker(  # noqa: PLR0915 - fixed closer phase diagnostics
    cdp_endpoint: str,
    target_id: Optional[str],
    anchor_target_id: str,
    creation_url: str,
    deadline: float,
    playwright_factory: Optional[Callable[[], Any]],
    private_process_group: Any,
    private_process_start_time: Any,
    role_marker: str,
    target_proof: Any,
    driver_done: Any,
    require_target_present: bool,
    require_target_absence: bool,
    failure_stage: Any = None,
) -> None:
    def record_stage(stage: _OracleTargetCloserStage) -> None:
        if failure_stage is not None:
            failure_stage.value = stage

    record_stage(_OracleTargetCloserStage.PROCESS_GROUP_SETUP)
    _enter_oracle_browser_worker_process_group(
        private_process_group,
        private_process_start_time,
    )
    # Install the role marker before Playwright launches its driver. The
    # sidecar uses this marker plus pidfds, never a stale parent PID.
    record_stage(_OracleTargetCloserStage.ROLE_MARKER_SETUP)
    os.environ[CHATGPT_NATIVE_HISTORY_ROLE_ENV] = role_marker
    playwright = None
    browser = None
    try:
        record_stage(_OracleTargetCloserStage.DEADLINE_CHECK)
        _raise_if_browser_deadline_expired(deadline)
        record_stage(_OracleTargetCloserStage.PLAYWRIGHT_START)
        playwright = _start_playwright_from_factory(playwright_factory)
        record_stage(_OracleTargetCloserStage.CDP_CONNECT)
        browser = playwright.chromium.connect_over_cdp(
            cdp_endpoint,
            timeout=_browser_timeout_milliseconds(
                _remaining_browser_timeout(deadline)
            ),
        )
        record_stage(_OracleTargetCloserStage.CDP_SESSION)
        session = browser.new_browser_cdp_session()
        # Only this published target may be closed; never close the browser.
        record_stage(_OracleTargetCloserStage.INITIAL_LISTING)
        target_listing = session.send("Target.getTargets")
        record_stage(_OracleTargetCloserStage.INITIAL_LISTING_INVALID)
        if not isinstance(target_listing, Mapping):
            return
        targets = target_listing.get("targetInfos")
        record_stage(_OracleTargetCloserStage.INITIAL_TARGETS_INVALID)
        if not isinstance(targets, (list, tuple)):
            return
        if any(
            not isinstance(info, Mapping)
            or not isinstance(info.get("targetId"), str)
            or not info.get("targetId")
            for info in targets
        ):
            return
        if target_id is None:
            # A lost createTarget reply is recoverable only by the unique URL
            # assigned before creation, never by host, page title, or account.
            record_stage(_OracleTargetCloserStage.TARGET_MATCH)
            matches = [
                info.get("targetId")
                for info in targets
                if isinstance(info, Mapping)
                and info.get("url") == creation_url
                and info.get("type") == "page"
                and info.get("targetId") != anchor_target_id
            ]
            if len(matches) != 1 or not isinstance(matches[0], str):
                record_stage(_OracleTargetCloserStage.TARGET_MATCH_NOT_UNIQUE)
                return
            target_id = matches[0]
        record_stage(_OracleTargetCloserStage.TARGET_LOOKUP)
        if not any(
            isinstance(info, Mapping) and info.get("targetId") == target_id
            for info in targets
        ):
            record_stage(_OracleTargetCloserStage.TARGET_MISSING)
            target_proof.value = not require_target_present
            return
        record_stage(_OracleTargetCloserStage.CLOSE_REQUEST)
        result = session.send("Target.closeTarget", {"targetId": target_id})
        if not isinstance(result, Mapping) or result.get("success") is not True:
            record_stage(_OracleTargetCloserStage.CLOSE_NOT_ACKNOWLEDGED)
            return
        if not require_target_absence:
            target_proof.value = True
            return
        record_stage(_OracleTargetCloserStage.DEADLINE_CHECK)
        # Close acknowledgment can precede removal from the target listing.
        while True:
            _raise_if_browser_deadline_expired(deadline)
            record_stage(_OracleTargetCloserStage.ABSENCE_LISTING)
            post_close_listing = session.send("Target.getTargets")
            record_stage(_OracleTargetCloserStage.ABSENCE_LISTING_INVALID)
            if not isinstance(post_close_listing, Mapping):
                return
            if "targetInfos" not in post_close_listing:
                return
            post_close_targets = post_close_listing["targetInfos"]
            record_stage(_OracleTargetCloserStage.ABSENCE_TARGETS_INVALID)
            if not isinstance(post_close_targets, (list, tuple)):
                return
            if any(
                not isinstance(info, Mapping)
                or not isinstance(info.get("targetId"), str)
                or not info.get("targetId")
                for info in post_close_targets
            ):
                return
            record_stage(_OracleTargetCloserStage.TARGET_ABSENCE_CHECK)
            target_proof.value = not any(
                isinstance(info, Mapping) and info.get("targetId") == target_id
                for info in post_close_targets
            )
            if target_proof.value:
                return
            record_stage(_OracleTargetCloserStage.TARGET_STILL_PRESENT)
            remaining_seconds = _remaining_browser_timeout(deadline)
            if remaining_seconds <= 0:
                return
            time.sleep(min(remaining_seconds, 0.05))
    except Exception:
        target_proof.value = False
    finally:
        try:
            _disconnect_attached_browser(playwright, browser)
        finally:
            driver_done.set()


def _receive_oracle_browser_worker_message(
    receiver: Any,
    deadline: float,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Mapping[str, Any]:
    pipe_buffer = bytearray()
    payload = bytearray()
    while True:
        frame = _read_oracle_browser_worker_frame(
            receiver,
            pipe_buffer,
            deadline,
            should_stop=should_stop,
        )
        if not frame:
            break
        if len(payload) + len(frame) > _ORACLE_BROWSER_IPC_MAX_BYTES:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary worker response exceeded the size limit."
            )
        payload.extend(frame)
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser boundary worker returned an invalid response."
        ) from exc
    if not isinstance(message, Mapping):
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser boundary worker returned an invalid response."
        )
    return message


def _read_oracle_browser_worker_frame(
    receiver: Any,
    pipe_buffer: bytearray,
    deadline: float,
    should_stop: Optional[Callable[[], bool]] = None,
) -> bytes:
    try:
        pipe_fd = receiver.fileno()
        os.set_blocking(pipe_fd, False)
    except (AttributeError, OSError) as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser boundary worker pipe is unavailable."
        ) from exc
    _fill_oracle_browser_pipe_buffer(
        pipe_fd,
        pipe_buffer,
        4,
        deadline,
        should_stop=should_stop,
    )
    frame_length = int.from_bytes(pipe_buffer[:4], byteorder="big", signed=True)
    del pipe_buffer[:4]
    if frame_length < 0 or frame_length > _ORACLE_BROWSER_IPC_FRAME_BYTES:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser boundary worker returned an invalid response."
        )
    _fill_oracle_browser_pipe_buffer(
        pipe_fd,
        pipe_buffer,
        frame_length,
        deadline,
        should_stop=should_stop,
    )
    frame = bytes(pipe_buffer[:frame_length])
    del pipe_buffer[:frame_length]
    return frame


def _fill_oracle_browser_pipe_buffer(
    pipe_fd: int,
    pipe_buffer: bytearray,
    required_bytes: int,
    deadline: float,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    while len(pipe_buffer) < required_bytes:
        if should_stop is not None and should_stop():
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser conversation-init capture was aborted."
            )
        remaining_seconds = _remaining_browser_timeout(deadline)
        if remaining_seconds <= 0:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser conversation-init capture timed out."
            )
        try:
            readable, _, _ = select.select(
                [pipe_fd],
                [],
                [],
                min(remaining_seconds, 0.1),
            )
        except (OSError, ValueError) as exc:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary worker pipe is unavailable."
            ) from exc
        if not readable:
            continue
        try:
            chunk = os.read(pipe_fd, 65536)
        except BlockingIOError:
            continue
        except OSError as exc:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary worker ended without a response."
            ) from exc
        if not chunk:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary worker ended without a response."
            )
        pipe_buffer.extend(chunk)


def _send_oracle_browser_worker_message(sender: Any, message: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        dict(message),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(serialized) > _ORACLE_BROWSER_IPC_MAX_BYTES:
        serialized = b'{"ok":false,"result":null}'
    for offset in range(0, len(serialized), _ORACLE_BROWSER_IPC_FRAME_BYTES):
        sender.send_bytes(
            serialized[offset : offset + _ORACLE_BROWSER_IPC_FRAME_BYTES]
        )
    sender.send_bytes(b"")


def _oracle_browser_process_context(
    playwright_factory: Optional[Callable[[], Any]],
) -> Any:
    if playwright_factory is not None and os.name == "posix":
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _enter_oracle_browser_worker_process_group(
    private_process_group: Any,
    private_process_start_time: Optional[Any] = None,
) -> None:
    if os.name == "posix":
        # Fail before starting the driver unless cleanup owns an isolated group.
        os.setsid()
        private_process_group.value = os.getpgrp()
        if private_process_start_time is not None:
            start_time = _native_history_process_start_time(os.getpid())
            if start_time is None:
                raise RuntimeError(
                    "Oracle browser worker identity could not be established."
                )
            private_process_start_time.value = start_time


def _terminate_oracle_browser_worker(
    process: Any,
    private_process_group: int,
    private_process_start_time: Optional[Any] = None,
) -> None:
    process_id = getattr(process, "pid", None)
    if (
        os.name == "posix"
        and isinstance(process_id, int)
        and private_process_group == process_id
        and private_process_group > 0
        and private_process_group != os.getpgrp()
        and private_process_start_time is not None
        and _native_history_process_identity_matches(
            process_id,
            private_process_start_time,
        )
    ):
        try:
            # The driver can survive its leader; group cleanup is unconditional.
            os.killpg(private_process_group, signal.SIGKILL)
        except OSError:
            pass
    try:
        process_alive = process.is_alive()
    except (AssertionError, OSError, ValueError):
        process_alive = False
    if not process_alive:
        return
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
            return
        except (AssertionError, OSError, ValueError):
            pass
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        try:
            terminate()
        except (AssertionError, OSError, ValueError):
            pass


def _start_playwright_from_factory(
    playwright_factory: Optional[Callable[[], Any]],
) -> Any:
    if playwright_factory is not None:
        return playwright_factory()
    try:
        playwright_api = importlib.import_module("playwright.sync_api")
        sync_playwright = playwright_api.sync_playwright
    except (ImportError, AttributeError) as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Live Oracle browser collection requires Playwright in the "
            "browser-boundary environment."
        ) from exc
    try:
        return sync_playwright().start()
    except Exception as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Live Oracle browser collection could not start Playwright."
        ) from exc


def _raise_if_browser_deadline_expired(deadline: float) -> None:
    if _remaining_browser_timeout(deadline) <= 0:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser conversation-init capture timed out."
        )


def build_oracle_browser_conversation_init_transport(
    *,
    cdp_endpoint: Optional[str] = None,
    page_target_id: str,
    expected_account_hash: str,
    timeout_seconds: float = 30.0,
) -> OracleBrowserConversationInitTransport:
    """Build native capture in an owned page of the pinned target's context."""

    return OracleBrowserConversationInitTransport(
        cdp_endpoint=cdp_endpoint,
        page_target_id=page_target_id,
        expected_account_hash=expected_account_hash,
        timeout_seconds=timeout_seconds,
    )


def _validate_oracle_browser_request(request: urllib_request.Request) -> None:
    if request_has_conversation_content(request):
        raise OracleBrowserBoundaryUnavailable(
            "Oracle conversation-init transport requires POST with no body."
        )
    parsed = urlsplit(request.full_url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "chatgpt.com",
        "www.chatgpt.com",
        "chat.openai.com",
    }:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle conversation-init transport only permits ChatGPT HTTPS."
        )
    if parsed.path != CHATGPT_CONVERSATION_INIT_PATH:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle conversation-init transport only permits the "
            "conversation-init path."
        )


def _validate_page_target_id(page_target_id: Any) -> str:
    if not isinstance(page_target_id, str):
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser conversation-init requires an exact CDP page target id."
        )
    cleaned = page_target_id.strip()
    if not cleaned or any(ord(character) < 32 for character in cleaned):
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser conversation-init requires an exact CDP page target id."
        )
    return cleaned


def _remaining_browser_timeout(deadline: float) -> float:
    return deadline - time.monotonic()


def _browser_timeout_milliseconds(timeout_seconds: float) -> int:
    return max(1, int(timeout_seconds * 1000))


def _find_existing_chatgpt_page(
    browser: Any,
    page_target_id: str,
    request_url: str,
    *,
    deadline: Optional[float] = None,
) -> Any:
    target_host = urlsplit(request_url).hostname
    if not target_host:
        return None
    for context in getattr(browser, "contexts", ()):
        for page in getattr(context, "pages", ()):
            if deadline is not None and _remaining_browser_timeout(deadline) <= 0:
                raise OracleBrowserBoundaryUnavailable(
                    "Oracle browser conversation-init capture timed out."
                )
            page_url = str(getattr(page, "url", "") or "")
            page_host = urlsplit(page_url).hostname
            # A blank anchor selects context only; native headers prove identity.
            if (
                (page_url == "about:blank" or page_host == target_host)
                and _page_target_id(context, page) == page_target_id
            ):
                return page
    return None


def _page_target_id(context: Any, page: Any) -> Optional[str]:
    """Read a page's CDP target id without creating or navigating a page."""

    for attribute_name in ("target_id", "targetId"):
        value = getattr(page, attribute_name, None)
        if isinstance(value, str) and value:
            return value

    new_cdp_session = getattr(context, "new_cdp_session", None)
    if not callable(new_cdp_session):
        return None
    session = None
    try:
        session = new_cdp_session(page)
        result = session.send("Target.getTargetInfo")
        target_info = (
            result.get("targetInfo")
            if isinstance(result, Mapping)
            and isinstance(result.get("targetInfo"), Mapping)
            else result
        )
        if isinstance(target_info, Mapping):
            value = target_info.get("targetId")
            if isinstance(value, str) and value:
                return value
    except Exception:
        return None
    finally:
        detach = getattr(session, "detach", None)
        if callable(detach):
            try:
                detach()
            except Exception:
                pass
    return None


def _coerce_browser_response(result: Any) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser returned no conversation-init response."
        )
    status_code = result.get("status_code")
    if isinstance(status_code, bool) or not isinstance(
        status_code, (int, float, str)
    ):
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser returned an invalid conversation-init status."
        )
    try:
        status = int(status_code)
    except (TypeError, ValueError) as exc:
        raise OracleBrowserBoundaryUnavailable(
            "Oracle browser returned an invalid conversation-init status."
        ) from exc
    response: Dict[str, Any] = {
        "status_code": status,
        "payload": result.get("payload"),
    }
    native = result.get("native_capture")
    if isinstance(native, Mapping):
        response["native_capture"] = dict(native)
    return response


def _disconnect_attached_browser(playwright: Any, browser: Any) -> None:
    """Disconnect the driver without closing Oracle's browser or profile."""
    if browser is not None:
        disconnect = getattr(browser, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass
    if playwright is not None:
        stop = getattr(playwright, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass


def write_conversation_init_snapshot(path: str, snapshot: Mapping[str, Any]) -> None:
    """Atomically write a credential-safe snapshot; refuse symlink destinations."""

    dest = Path(path).expanduser()
    if dest.is_symlink():
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init snapshot path must not be a symlink.",
            telemetry_class="malformed_telemetry",
        )
    if dest.exists() and not dest.is_file():
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init snapshot path must be a regular file.",
            telemetry_class="malformed_telemetry",
        )
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init snapshot directory is not writable.",
            telemetry_class="malformed_telemetry",
        ) from exc
    serialized = json.dumps(dict(snapshot), sort_keys=True, ensure_ascii=True)
    if len(serialized.encode("utf-8")) > MAX_CONVERSATION_INIT_SOURCE_BYTES:
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init snapshot exceeds the sanitized size limit.",
            telemetry_class="malformed_telemetry",
        )
    fd, tmp_name = tempfile.mkstemp(
        prefix=".conversation-init.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if dest.is_symlink():
            raise ChatGPTConversationInitError(
                "ChatGPT conversation-init snapshot path must not be a symlink.",
                telemetry_class="malformed_telemetry",
            )
        os.replace(tmp_name, dest)
    except ChatGPTConversationInitError:
        _unlink_quietly(tmp_name)
        raise
    except OSError as exc:
        _unlink_quietly(tmp_name)
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init snapshot could not be written.",
            telemetry_class="malformed_telemetry",
        ) from exc


class ChatGPTConversationInitCollector:
    """Fixture-testable browser-boundary collector for conversation-init JSON."""

    def __init__(
        self,
        transport: ConversationInitTransport,
        *,
        request_url: Optional[str] = None,
    ) -> None:
        self.transport = transport
        self.request_url = request_url

    def collect(self, source_path: str) -> Dict[str, Any]:
        return collect_conversation_init_snapshot(
            source_path,
            transport=self.transport,
            request_url=self.request_url,
        )


def collect_conversation_init_snapshot(  # noqa: PLR0915 - collector state
    source_path: str,
    *,
    transport: ConversationInitTransport,
    request_url: Optional[str] = None,
) -> Dict[str, Any]:
    """POST with no body through an injected transport and write a safe snapshot.

    This helper is the browser-boundary collector. The sidecar loop remains a
    file consumer and must not call chatgpt.com. Live authenticated Oracle
    browser proof is a separate acceptance gate and is never implied here.
    """

    contract = conversation_init_request_contract(request_url)
    request = build_conversation_init_request(contract["url"])
    if request_has_conversation_content(request):
        raise ChatGPTConversationInitError(
            "ChatGPT conversation-init collector request must be POST with no body.",
            telemetry_class="malformed_telemetry",
        )
    summary = _collector_summary(contract, source_path)
    try:
        raw = transport.fetch(request)
    except ChatGPTConversationInitError as exc:
        summary["telemetry_class"] = exc.telemetry_class
        summary["status_code"] = exc.status_code
        summary["last_good_state_retained"] = True
        return summary
    except Exception:
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return summary
    if not isinstance(raw, Mapping):
        summary["telemetry_class"] = "malformed_telemetry"
        summary["last_good_state_retained"] = True
        return summary

    sanitized = sanitize_conversation_init_boundary(
        raw,
        source_path=source_path,
        request_url=request_url,
    )
    summary["status_code"] = sanitized.get("status_code")
    summary["redacted_field_count"] = sanitized.get("redacted_field_count")
    summary["account_identity_hashed"] = bool(sanitized.get("account_hash"))
    summary["account_hash"] = sanitized.get("account_hash")
    summary["account_identity_source"] = sanitized.get(
        "account_identity_source"
    )
    summary["account_identity_verified"] = bool(
        sanitized.get("account_identity_verified")
    )
    summary["payload_state"] = sanitized.get("payload_state")
    summary["request_body_omitted"] = sanitized.get("request_body_omitted")
    summary["retry_after_seconds"] = sanitized.get("retry_after_seconds")
    summary["browser_challenge"] = bool(sanitized.get("browser_challenge"))
    summary["native_capture"] = (
        dict(sanitized["native_capture"])
        if isinstance(sanitized.get("native_capture"), Mapping)
        else None
    )
    summary["native_capture_error"] = sanitized.get("native_capture_error")
    summary.update(
        _collection_diagnostics(
            sanitized.get("payload"),
            malformed_collection_projection=(
                sanitized.get("malformed_collection_projection") is True
            ),
            projection_truncated=sanitized.get("projection_truncated") is True,
        )
    )
    persistability_failure_reason = _snapshot_persistability_failure_reason(
        sanitized
    )
    writable = persistability_failure_reason is None
    reusable = _destination_has_reusable_snapshot(source_path)
    if not writable:
        summary["failure_reason"] = persistability_failure_reason
        summary["telemetry_status"] = _failure_telemetry_status(sanitized)
        summary["telemetry_class"] = _failure_telemetry_class(sanitized)
        summary["last_good_state_retained"] = reusable
        if reusable:
            return summary
    try:
        write_conversation_init_snapshot(source_path, sanitized)
    except ChatGPTConversationInitError as exc:
        summary["failure_reason"] = "snapshot_write_failed"
        summary["telemetry_class"] = exc.telemetry_class
        summary["last_good_state_retained"] = True
        return summary
    summary["written"] = True
    summary["last_good_state_retained"] = False
    if writable:
        summary["telemetry_status"] = "valid"
        summary["telemetry_class"] = None
    else:
        summary["telemetry_status"] = _failure_telemetry_status(sanitized)
        summary["telemetry_class"] = _failure_telemetry_class(sanitized)
    return summary


def collect_conversation_init_snapshot_from_oracle_browser(
    source_path: str,
    *,
    cdp_endpoint: Optional[str] = None,
    page_target_id: str,
    expected_account_hash: str,
    timeout_seconds: float = 30.0,
    request_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect one verified, account-bound snapshot through Oracle's browser.

    The caller must provide an already-running Oracle browser with an existing
    authenticated ChatGPT page, its exact CDP target id, a reachable CDP
    endpoint, and the canonical12 account hash pin. This function never
    launches Chrome, opens a page, reads browser storage, or exports
    credentials. It writes only a current response whose authoritative
    ``account_id``/``chatgpt_account_id`` hashes to the configured pin.
    The returned summary includes ``account_identity_verified``,
    canonical12 ``account_hash``, ``account_identity_source``,
    ``account_identity_verification_error``, ``snapshot_fresh``, and
    ``last_good_state_retained``.
    """

    contract = conversation_init_request_contract(request_url)
    summary = _collector_summary(contract, source_path)
    expected = _normalize_expected_account_hash(expected_account_hash)
    if expected is None:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=None,
            error="invalid_expected_account_hash",
            telemetry_status="auth",
            telemetry_class="auth",
        )
    try:
        transport = build_oracle_browser_conversation_init_transport(
            cdp_endpoint=cdp_endpoint,
            page_target_id=page_target_id,
            expected_account_hash=expected,
            timeout_seconds=timeout_seconds,
        )
    except (ChatGPTConversationInitError, ValueError) as exc:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected,
            error="browser_boundary_unavailable",
            telemetry_status="auth",
            telemetry_class=(
                exc.telemetry_class
                if isinstance(exc, ChatGPTConversationInitError)
                else "auth"
            ),
        )

    return _collect_bound_conversation_init_snapshot(
        source_path,
        transport=transport,
        request_url=request_url,
        expected_account_hash=expected,
        summary=summary,
    )


def _collect_bound_conversation_init_snapshot(  # noqa: PLR0915 - bound state
    source_path: str,
    *,
    transport: ConversationInitTransport,
    request_url: Optional[str],
    expected_account_hash: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    request = build_conversation_init_request(
        conversation_init_request_contract(request_url)["url"]
    )
    reusable = _destination_has_reusable_bound_snapshot(
        source_path,
        expected_account_hash=expected_account_hash,
    )
    try:
        raw = transport.fetch(request)
    except ChatGPTConversationInitError as exc:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error="browser_boundary_unavailable",
            telemetry_status="auth",
            telemetry_class=exc.telemetry_class,
            reusable=reusable,
        )
    except Exception:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error="browser_boundary_unavailable",
            telemetry_status="auth",
            telemetry_class="auth",
            reusable=reusable,
        )
    if not isinstance(raw, Mapping):
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error="malformed_browser_response",
            telemetry_status="malformed",
            telemetry_class="malformed_telemetry",
            reusable=reusable,
        )

    sanitized = sanitize_conversation_init_boundary(
        raw,
        source_path=source_path,
        request_url=request_url,
        _allow_verified_envelope_identity=False,
    )
    summary["status_code"] = sanitized.get("status_code")
    summary["redacted_field_count"] = sanitized.get("redacted_field_count")
    summary["payload_state"] = sanitized.get("payload_state")
    summary["request_body_omitted"] = sanitized.get("request_body_omitted")
    summary["retry_after_seconds"] = sanitized.get("retry_after_seconds")
    summary["browser_challenge"] = bool(sanitized.get("browser_challenge"))
    summary["native_capture"] = (
        dict(sanitized["native_capture"])
        if isinstance(sanitized.get("native_capture"), Mapping)
        else None
    )
    summary["native_capture_error"] = sanitized.get("native_capture_error")
    summary.update(
        _collection_diagnostics(
            sanitized.get("payload"),
            malformed_collection_projection=(
                sanitized.get("malformed_collection_projection") is True
            ),
            projection_truncated=sanitized.get("projection_truncated") is True,
        )
    )
    summary["account_hash"] = sanitized.get("account_hash")
    summary["account_identity_hashed"] = bool(sanitized.get("account_hash"))
    summary["account_identity_source"] = sanitized.get("account_identity_source")
    summary["collector_source"] = ORACLE_BROWSER_BOUNDARY_NAME
    summary["browser_boundary"] = ORACLE_BROWSER_BOUNDARY_NAME

    if summary["browser_challenge"]:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error="browser_challenge",
            telemetry_status="auth",
            telemetry_class="browser_challenge",
            reusable=reusable,
        )
    status_failure = _http_status_failure(sanitized.get("status_code"))
    if status_failure is not None:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error=f"http_{status_failure}",
            telemetry_status=status_failure,
            telemetry_class="auth" if status_failure == "auth" else "http_error",
            reusable=reusable,
        )
    if summary["native_capture_error"]:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error=summary["native_capture_error"],
            telemetry_status="malformed",
            telemetry_class="malformed_telemetry",
            reusable=reusable,
        )

    _status_code, payload_raw, _envelope_redacted = _split_boundary_envelope(raw)
    native_capture = sanitized.get("native_capture")
    if isinstance(native_capture, Mapping):
        native_account_hash = native_capture.get("account_hash")
        if native_account_hash != expected_account_hash:
            return _bound_capture_failure(
                summary,
                source_path=source_path,
                expected_account_hash=expected_account_hash,
                error="account_identity_mismatch",
                telemetry_status="auth",
                telemetry_class="auth",
                reusable=reusable,
            )
        sanitized = _apply_verified_bound_identity(
            sanitized,
            account_hash=expected_account_hash,
            account_identity_fields=sanitized.get("account_identity_fields")
            or ["native_capture.account_hash"],
            account_identity_source=(
                CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
            ),
            account_identity_verification_source=(
                CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
            ),
        )
    else:
        account_id, identity_fields, identity_error = (
            _extract_canonical_account_identity(raw, payload_raw)
        )
        if identity_error is not None or account_id is None:
            return _bound_capture_failure(
                summary,
                source_path=source_path,
                expected_account_hash=expected_account_hash,
                error=identity_error or "missing_authoritative_account_id",
                telemetry_status="missing_account_identity",
                telemetry_class="malformed_telemetry",
                reusable=reusable,
            )
        actual_account_hash = (
            hash_chatgpt_conversation_init_canonical_account_id(account_id)
        )
        if actual_account_hash != expected_account_hash:
            return _bound_capture_failure(
                summary,
                source_path=source_path,
                expected_account_hash=expected_account_hash,
                error="account_identity_mismatch",
                telemetry_status="auth",
                telemetry_class="auth",
                reusable=reusable,
            )
        sanitized = _apply_verified_bound_identity(
            sanitized,
            account_hash=actual_account_hash,
            account_identity_fields=identity_fields,
        )

    summary["account_identity_hashed"] = True
    summary["account_hash"] = sanitized["account_hash"]
    summary["account_identity_verified"] = True
    summary["account_identity_fields"] = sanitized["account_identity_fields"]
    summary["account_identity_source"] = sanitized["account_identity_source"]
    summary["account_identity_hash_algorithm"] = (
        CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM
    )
    summary["account_identity_hash_length"] = (
        CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
    )
    summary["account_identity_verification_source"] = sanitized[
        "account_identity_verification_source"
    ]
    summary["live_authenticated_oracle_browser"] = True
    persistability_failure_reason = _snapshot_persistability_failure_reason(
        sanitized,
        expected_account_hash=expected_account_hash,
        require_verified_identity=True,
    )
    if persistability_failure_reason is not None:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error=persistability_failure_reason,
            telemetry_status=_failure_telemetry_status(sanitized),
            telemetry_class=_failure_telemetry_class(sanitized),
            reusable=reusable,
        )
    try:
        write_conversation_init_snapshot(source_path, sanitized)
    except ChatGPTConversationInitError as exc:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error="snapshot_write_failed",
            telemetry_status="malformed",
            telemetry_class=exc.telemetry_class,
            reusable=reusable,
        )
    summary["written"] = True
    summary["snapshot_fresh"] = True
    summary["last_good_state_retained"] = False
    summary["telemetry_status"] = "valid"
    summary["telemetry_class"] = None
    return summary


def _normalize_expected_account_hash(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if _CANONICAL_ACCOUNT_HASH_RE.fullmatch(cleaned) is None:
        return None
    return cleaned


def _bound_capture_failure(
    summary: Dict[str, Any],
    *,
    source_path: str,
    expected_account_hash: Optional[str],
    error: str,
    telemetry_status: str,
    telemetry_class: str,
    reusable: Optional[bool] = None,
) -> Dict[str, Any]:
    if reusable is None:
        reusable = bool(
            expected_account_hash
            and _destination_has_reusable_bound_snapshot(
                source_path,
                expected_account_hash=expected_account_hash,
            )
        )
    summary["collector_source"] = ORACLE_BROWSER_BOUNDARY_NAME
    summary["browser_boundary"] = ORACLE_BROWSER_BOUNDARY_NAME
    summary["written"] = False
    summary["snapshot_fresh"] = False
    summary["last_good_state_retained"] = reusable
    summary["failure_reason"] = error
    summary["account_identity_verification_error"] = error
    summary["telemetry_status"] = telemetry_status
    summary["telemetry_class"] = telemetry_class
    return summary


def _collector_summary(contract: Mapping[str, Any], source_path: str) -> Dict[str, Any]:
    return {
        "written": False,
        "collector_source": "browser_boundary",
        "browser_boundary": None,
        "live_authenticated_oracle_browser": False,
        "request_method": contract.get("method"),
        "request_path": contract.get("path"),
        "request_url": contract.get("url"),
        "request_body_omitted": bool(contract.get("body_omitted")),
        "has_model_message": False,
        "has_conversation_content": False,
        "status_code": None,
        "payload_state": None,
        "account_identity_hashed": False,
        "account_hash": None,
        "account_identity_verified": False,
        "account_identity_fields": [],
        "account_identity_source": None,
        "account_identity_hash_algorithm": None,
        "account_identity_hash_length": None,
        "account_identity_verification_source": None,
        "account_identity_verification_error": None,
        "failure_reason": None,
        "native_capture": None,
        "native_capture_error": None,
        "browser_challenge": False,
        "retry_after_seconds": None,
        "model_limits_state": "absent_unknown",
        "limits_progress_state": "absent_unknown",
        "blocked_features_state": "absent_unknown",
        "malformed_entry_count": 0,
        "valid_observation_count": 0,
        "projection_truncated": False,
        "malformed_collection_projection": False,
        "redacted_field_count": 0,
        "source_identity_hash": hash_chatgpt_conversation_init_source_identity(
            source_path
        ),
        "telemetry_status": None,
        "telemetry_class": None,
        "snapshot_fresh": False,
        "last_good_state_retained": False,
    }


def _apply_verified_bound_identity(
    sanitized: Mapping[str, Any],
    *,
    account_hash: str,
    account_identity_fields: Sequence[str],
    account_identity_source: str = (
        CHATGPT_CONVERSATION_INIT_VERIFIED_PAYLOAD_IDENTITY_SOURCE
    ),
    account_identity_verification_source: str = (
        CHATGPT_CONVERSATION_INIT_BROWSER_ACCOUNT_IDENTITY_SOURCE
    ),
) -> Dict[str, Any]:
    result = dict(sanitized)
    result["account_hash"] = account_hash
    result["account_identity_fields"] = _safe_identity_fields(
        account_identity_fields
    )
    result["account_identity_source"] = account_identity_source
    result["account_identity_verification_source"] = (
        account_identity_verification_source
    )
    result["account_identity_verified"] = True
    result["account_identity_hash_algorithm"] = (
        CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM
    )
    result["account_identity_hash_length"] = (
        CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
    )
    result["account_identity_verification_error"] = None
    result["collector_source"] = ORACLE_BROWSER_BOUNDARY_NAME
    result["browser_boundary"] = ORACLE_BROWSER_BOUNDARY_NAME
    return result


def _safe_identity_fields(fields: Sequence[str]) -> List[str]:
    safe: List[str] = []
    for field in fields[:MAX_PROJECTION_LIST_ITEMS]:
        if not isinstance(field, str):
            continue
        cleaned = field.strip()
        if (
            cleaned
            and len(cleaned) <= MAX_SAFE_STRING_LENGTH
            and not _is_unsafe_string(cleaned)
        ):
            safe.append(cleaned)
    return safe


def _snapshot_is_persistable(
    sanitized: Mapping[str, Any],
    *,
    expected_account_hash: Optional[str] = None,
    require_verified_identity: bool = False,
) -> bool:
    return (
        _snapshot_persistability_failure_reason(
            sanitized,
            expected_account_hash=expected_account_hash,
            require_verified_identity=require_verified_identity,
        )
        is None
    )


def _snapshot_persistability_failure_reason(
    sanitized: Mapping[str, Any],
    *,
    expected_account_hash: Optional[str] = None,
    require_verified_identity: bool = False,
) -> Optional[str]:
    if sanitized.get("browser_challenge") is True:
        return "browser_challenge"
    if sanitized.get("native_capture_error"):
        return "native_capture_error"
    if (
        sanitized.get("account_identity_source")
        == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
        and not _is_verified_bound_identity(
            sanitized,
            sanitized.get("account_hash"),
        )
    ):
        return "native_identity_unverified"
    if _http_status_failure(sanitized.get("status_code")) is not None:
        return "http_error"
    if sanitized.get("payload_state") != "present":
        return "payload_not_present"
    payload = sanitized.get("payload")
    if not isinstance(payload, Mapping):
        return "payload_not_mapping"
    if not looks_like_conversation_init_payload(payload):
        return "payload_marker_missing"
    if _collections_are_wholly_malformed(
        _parse_collections(
            payload,
            {
                "malformed_collection_projection": (
                    sanitized.get("malformed_collection_projection") is True
                )
            },
        )
    ):
        return "collections_wholly_malformed"
    if require_verified_identity:
        if not _is_verified_bound_identity(
            sanitized,
            sanitized.get("account_hash"),
        ):
            return "account_identity_unverified"
        if (
            expected_account_hash is not None
            and sanitized.get("account_hash") != expected_account_hash
        ):
            return "account_identity_mismatch"
    elif not sanitized.get("account_hash") and not sanitized.get("source_identity_hash"):
        return "account_identity_missing"
    return None


def _snapshot_fits_write_budget(sanitized: Mapping[str, Any]) -> bool:
    return (
        len(
            json.dumps(
                dict(sanitized),
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        )
        <= MAX_CONVERSATION_INIT_SOURCE_BYTES
    )


def _destination_has_reusable_snapshot(path: str) -> bool:
    try:
        raw = load_conversation_init_source(path)
    except ChatGPTConversationInitError:
        return False
    sanitized = sanitize_conversation_init_boundary(
        raw,
        source_path=path,
        _allow_verified_envelope_identity=True,
    )
    return (
        _snapshot_is_persistable(sanitized)
        and _snapshot_fits_write_budget(sanitized)
    )


def _destination_has_reusable_bound_snapshot(
    path: str,
    *,
    expected_account_hash: str,
) -> bool:
    try:
        raw = load_conversation_init_source(path)
    except ChatGPTConversationInitError:
        return False
    sanitized = sanitize_conversation_init_boundary(
        raw,
        source_path=path,
        _allow_verified_envelope_identity=True,
    )
    return _snapshot_is_persistable(
        sanitized,
        expected_account_hash=expected_account_hash,
        require_verified_identity=True,
    ) and _snapshot_fits_write_budget(sanitized)


def _failure_telemetry_status(sanitized: Mapping[str, Any]) -> str:
    if sanitized.get("browser_challenge") is True:
        return "auth"
    if sanitized.get("native_capture_error"):
        return "malformed"
    http_error = _http_status_failure(sanitized.get("status_code"))
    if http_error is not None:
        return http_error
    if (
        sanitized.get("account_identity_source")
        == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
        and not _is_verified_bound_identity(
            sanitized,
            sanitized.get("account_hash"),
        )
    ):
        return "auth"
    if not sanitized.get("account_hash") and not sanitized.get("source_identity_hash"):
        return "missing_account_identity"
    return "malformed"


def _failure_telemetry_class(sanitized: Mapping[str, Any]) -> str:
    if sanitized.get("browser_challenge") is True:
        return "browser_challenge"
    status = _failure_telemetry_status(sanitized)
    if status == "auth":
        return "auth"
    if status in {"malformed", "missing_account_identity"}:
        return "malformed_telemetry"
    return "http_error"


def _retained_envelope_identity(raw: Any) -> Tuple[Optional[str], List[str]]:
    if not isinstance(raw, Mapping):
        return None, []
    account_hash = raw.get("account_hash")
    if not _is_retained_account_hash(account_hash):
        return None, []
    retained_fields = raw.get("account_identity_fields")
    fields: List[str] = []
    if isinstance(retained_fields, list):
        for field in retained_fields[:MAX_PROJECTION_LIST_ITEMS]:
            if (
                isinstance(field, str)
                and field.strip()
                and not _is_unsafe_string(field)
            ):
                fields.append(field.strip()[:128])
    return str(account_hash), fields


def _retained_bound_envelope_identity(
    raw: Any,
) -> Tuple[Optional[str], List[str]]:
    if not isinstance(raw, Mapping):
        return None, []
    if raw.get("account_identity_verified") is not True:
        return None, []
    if raw.get("browser_challenge") is True:
        return None, []
    identity_source = raw.get("account_identity_source")
    if identity_source == CHATGPT_CONVERSATION_INIT_VERIFIED_PAYLOAD_IDENTITY_SOURCE:
        if raw.get("account_identity_verification_source") != (
            CHATGPT_CONVERSATION_INIT_BROWSER_ACCOUNT_IDENTITY_SOURCE
        ):
            return None, []
    elif (
        identity_source
        == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
    ):
        if raw.get("account_identity_verification_source") != (
            CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
        ):
            return None, []
        native_capture, native_capture_error = _sanitize_native_capture(
            raw.get("native_capture")
        )
        if native_capture_error is not None or native_capture is None:
            return None, []
    else:
        return None, []
    if raw.get("account_identity_hash_algorithm") != (
        CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_ALGORITHM
    ):
        return None, []
    if raw.get("account_identity_hash_length") != (
        CHATGPT_CONVERSATION_INIT_ACCOUNT_HASH_LENGTH
    ):
        return None, []
    if raw.get("browser_boundary") != ORACLE_BROWSER_BOUNDARY_NAME:
        return None, []
    if raw.get("collector_source") != ORACLE_BROWSER_BOUNDARY_NAME:
        return None, []
    account_hash = raw.get("account_hash")
    if not _is_canonical_account_hash(account_hash):
        return None, []
    retained_fields = raw.get("account_identity_fields")
    fields = (
        _safe_identity_fields(retained_fields)
        if isinstance(retained_fields, list)
        else []
    )
    if not fields:
        return None, []
    if identity_source == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE:
        if fields != ["native_capture.account_hash"]:
            return None, []
        if native_capture is None or native_capture["account_hash"] != account_hash:
            return None, []
    elif any(
        field not in _CANONICAL_ACCOUNT_ID_PROVENANCE_FIELDS for field in fields
    ):
        return None, []
    return account_hash, fields


def _resolve_account_identity_source(
    raw: Any,
    *,
    account_hash: Optional[str],
    source_identity_hash: Optional[str],
) -> Optional[str]:
    if isinstance(raw, Mapping):
        retained_native_capture_error = raw.get("native_capture_error")
        if "native_capture" in raw or (
            retained_native_capture_error in _NATIVE_CAPTURE_ERRORS
        ):
            if account_hash:
                return CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
            if retained_native_capture_error in _NATIVE_CAPTURE_ERRORS:
                return CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
            return None
        retained = raw.get("account_identity_source")
        if (
            retained
            in {
                "provider_payload",
                "source_path",
                CHATGPT_CONVERSATION_INIT_BROWSER_ACCOUNT_IDENTITY_SOURCE,
            }
            and account_hash
        ):
            return str(retained)
    if account_hash:
        return "provider_payload"
    if source_identity_hash:
        return "source_path"
    return None


def _is_retained_account_hash(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _is_canonical_account_hash(value: Any) -> bool:
    return isinstance(value, str) and _CANONICAL_ACCOUNT_HASH_RE.fullmatch(value) is not None


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
