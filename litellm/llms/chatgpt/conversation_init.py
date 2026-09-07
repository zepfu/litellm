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
from datetime import datetime, timezone
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
        }

    status_code, payload_raw, envelope_redacted = _split_boundary_envelope(raw)
    native_capture_present = "native_capture" in raw
    raw_native_capture = raw.get("native_capture")
    raw_browser_challenge = raw.get("browser_challenge")
    browser_challenge = (
        raw_browser_challenge if isinstance(raw_browser_challenge, bool) else False
    )
    native_capture, native_capture_error = (
        _sanitize_native_capture(raw.get("native_capture"))
        if native_capture_present
        else (None, None)
    )
    if native_capture is not None and native_capture_error is None:
        native_capture_error = _native_payload_identity_error(
            raw,
            payload_raw,
            native_capture["account_hash"],
        )
    retained_native_capture_error = raw.get("native_capture_error")
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
        "projection_truncated": False,
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
    truncated = (
        len(feature_rows) >= MAX_PROJECTION_LIST_ITEMS
        or len(model_rows) >= MAX_PROJECTION_LIST_ITEMS
        or len(blocked_rows) >= MAX_PROJECTION_LIST_ITEMS
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
        "malformed": feature_malformed + model_malformed + blocked_malformed,
        "truncated": truncated,
    }
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
    except (TypeError, ValueError):
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
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    redacted_count = 0
    payload: Dict[str, Any] = {}
    schema: Dict[str, Any] = {}
    items = list(mapping.items())[:MAX_PROJECTION_OBJECT_KEYS]
    parent_normalized = _normalize_key(parent_key) if parent_key else ""
    identity_container = parent_normalized in _ACCOUNT_IDENTITY_CONTAINERS
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
        redacted_value, node, nested_redacted = _redact_value(
            value,
            depth=depth + 1,
            parent_key=name,
        )
        redacted_count += nested_redacted
        schema[name] = node
        if node.get("kind") != "redacted":
            payload[name] = redacted_value
    return payload, schema, redacted_count


def _redact_value(
    value: Any,
    *,
    depth: int,
    parent_key: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], int]:
    if depth > MAX_PROJECTION_DEPTH:
        return None, {"kind": "truncated"}, 0
    if value is None:
        return None, {"kind": "null"}, 0
    if isinstance(value, bool):
        return value, {"kind": "bool", "value": value}, 0
    if isinstance(value, int):
        return value, {"kind": "int", "value": value}, 0
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None, {"kind": "redacted"}, 1
        return value, {"kind": "float", "value": value}, 0
    if isinstance(value, str):
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
        )
        fingerprint = _schema_fingerprint(nested_schema)
        object_node: Dict[str, Any] = {
            "kind": "object",
            "keys": sorted(nested_schema),
            "schema_fingerprint": fingerprint,
            "state": "present" if nested_schema else "empty_unknown",
            "projection": nested_payload,
        }
        return nested_payload, object_node, redacted
    if isinstance(value, list):
        items: List[Any] = []
        item_kinds: List[str] = []
        redacted = 0
        bounded = value[:MAX_PROJECTION_LIST_ITEMS]
        for item in bounded:
            nested_value, item_node, nested_redacted = _redact_value(
                item,
                depth=depth + 1,
                parent_key=parent_key,
            )
            redacted += nested_redacted
            item_kinds.append(str(item_node.get("kind") or "unknown"))
            if item_node.get("kind") != "redacted":
                items.append(nested_value)
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
        }
        return items, list_node, redacted
    return None, {"kind": "redacted"}, 1


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
    return normalized in _SAFE_STRING_FIELD_NAMES and _is_safe_telemetry_string(value)


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
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:
        return None
    return number


def _parse_usage_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric != numeric:
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
        if value != value:
            return None
        text = str(int(value)) if float(value).is_integer() else str(value)
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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        return False
    normalized = _normalize_key(name)
    return "token" in normalized and any(
        marker in normalized.split("_")
        for marker in _SAFE_NUMERIC_TOKEN_KEY_MARKERS
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


_ORACLE_BROWSER_FETCH_SCRIPT = """
async (request) => {
  const expectedHost = String(request.expected_host || "").toLowerCase();
  const executingHost = String(window.location.hostname || "").toLowerCase();
  if (!expectedHost || executingHost !== expectedHost) {
    throw new Error("conversation-init browser target host changed");
  }
  const controller = new AbortController();
  const timeoutMs = Math.max(1, Number(request.timeout_ms));
  let timeoutId = null;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      controller.abort();
      reject(new Error("conversation-init browser fetch timed out"));
    }, timeoutMs);
  });
  const fetchResponse = (async () => {
    const response = await fetch(request.url, {
      method: "POST",
      credentials: "include",
      redirect: "error",
      signal: controller.signal
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    return {
      status_code: response.status,
      payload
    };
  })();
  try {
    return await Promise.race([fetchResponse, timeout]);
  } finally {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
  }
}
"""


class OracleBrowserConversationInitTransport:
    """Attach to an existing Oracle browser and issue the frontend POST.

    This transport deliberately uses CDP attachment and an existing ChatGPT
    page. It never launches a browser, creates a persistent context, reads
    cookies or storage, or returns response headers.
    """

    boundary_name = ORACLE_BROWSER_BOUNDARY_NAME

    def __init__(
        self,
        *,
        cdp_endpoint: Optional[str] = None,
        page_target_id: str,
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
        self.timeout_seconds = timeout_seconds
        self._playwright_factory = playwright_factory

    def fetch(self, request: urllib_request.Request) -> Mapping[str, Any]:
        _validate_oracle_browser_request(request)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            return _run_oracle_browser_capture_in_worker(
                cdp_endpoint=self.cdp_endpoint,
                page_target_id=self.page_target_id,
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
    request_url: str,
    deadline: float,
    playwright_factory: Optional[Callable[[], Any]],
) -> Mapping[str, Any]:
    context = _oracle_browser_process_context(playwright_factory)
    receiver, sender = context.Pipe(duplex=False)
    private_process_group = context.RawValue("q", 0)
    process = context.Process(
        target=_oracle_browser_capture_worker,
        args=(
            sender,
            cdp_endpoint,
            page_target_id,
            request_url,
            deadline,
            playwright_factory,
            private_process_group,
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
        message = _receive_oracle_browser_worker_message(receiver, deadline)
        remaining_seconds = _remaining_browser_timeout(deadline)
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
        _terminate_oracle_browser_worker(process, private_process_group.value)
        process.join(timeout=0)


def _oracle_browser_capture_worker(
    sender: Any,
    cdp_endpoint: str,
    page_target_id: str,
    request_url: str,
    deadline: float,
    playwright_factory: Optional[Callable[[], Any]],
    private_process_group: Any,
) -> None:
    _enter_oracle_browser_worker_process_group(private_process_group)
    playwright = None
    browser = None
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
        page = _find_existing_chatgpt_page(
            browser,
            page_target_id,
            request_url,
            deadline=deadline,
        )
        if page is None:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser has no existing ChatGPT CDP target for the "
                "conversation-init request."
            )
        target_host = urlsplit(request_url).hostname
        if not target_host:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser conversation-init request has no host."
            )
        _raise_if_browser_deadline_expired(deadline)
        result = page.evaluate(
            _ORACLE_BROWSER_FETCH_SCRIPT,
            {
                "url": request_url,
                "expected_host": target_host,
                "timeout_ms": _browser_timeout_milliseconds(
                    _remaining_browser_timeout(deadline)
                ),
            },
        )
        _raise_if_browser_deadline_expired(deadline)
        successful = True
    except Exception:
        successful = False
    finally:
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


def _receive_oracle_browser_worker_message(
    receiver: Any,
    deadline: float,
) -> Mapping[str, Any]:
    pipe_buffer = bytearray()
    payload = bytearray()
    while True:
        frame = _read_oracle_browser_worker_frame(
            receiver,
            pipe_buffer,
            deadline,
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
    )
    frame = bytes(pipe_buffer[:frame_length])
    del pipe_buffer[:frame_length]
    return frame


def _fill_oracle_browser_pipe_buffer(
    pipe_fd: int,
    pipe_buffer: bytearray,
    required_bytes: int,
    deadline: float,
) -> None:
    while len(pipe_buffer) < required_bytes:
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
                remaining_seconds,
            )
        except (OSError, ValueError) as exc:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser boundary worker pipe is unavailable."
            ) from exc
        if not readable:
            raise OracleBrowserBoundaryUnavailable(
                "Oracle browser conversation-init capture timed out."
            )
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


def _enter_oracle_browser_worker_process_group(private_process_group: Any) -> None:
    if os.name == "posix":
        # Fail before starting the driver unless cleanup owns an isolated group.
        os.setsid()
        private_process_group.value = os.getpgrp()


def _terminate_oracle_browser_worker(process: Any, private_process_group: int) -> None:
    process_id = getattr(process, "pid", None)
    if (
        os.name == "posix"
        and isinstance(process_id, int)
        and private_process_group == process_id
        and private_process_group > 0
        and private_process_group != os.getpgrp()
    ):
        try:
            # The driver can survive its leader; group cleanup is unconditional.
            os.killpg(private_process_group, signal.SIGKILL)
        except OSError:
            pass
    if not process.is_alive():
        return
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
            return
        except OSError:
            pass
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        try:
            terminate()
        except OSError:
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
    timeout_seconds: float = 30.0,
) -> OracleBrowserConversationInitTransport:
    """Build the attach-only transport for the established Oracle boundary."""

    return OracleBrowserConversationInitTransport(
        cdp_endpoint=cdp_endpoint,
        page_target_id=page_target_id,
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
            if page_host == target_host and _page_target_id(context, page) == page_target_id:
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
    for key in _CANONICAL_ACCOUNT_ID_KEYS:
        if key in result:
            response[key] = result[key]
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
    writable = _snapshot_is_persistable(sanitized)
    reusable = _destination_has_reusable_snapshot(source_path)
    if not writable:
        summary["telemetry_status"] = _failure_telemetry_status(sanitized)
        summary["telemetry_class"] = _failure_telemetry_class(sanitized)
        summary["last_good_state_retained"] = reusable
        if reusable:
            return summary
    try:
        write_conversation_init_snapshot(source_path, sanitized)
    except ChatGPTConversationInitError as exc:
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
    writable = _snapshot_is_persistable(
        sanitized,
        expected_account_hash=expected_account_hash,
        require_verified_identity=True,
    )
    if not writable:
        return _bound_capture_failure(
            summary,
            source_path=source_path,
            expected_account_hash=expected_account_hash,
            error="current_snapshot_not_persistable",
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
        "native_capture": None,
        "native_capture_error": None,
        "browser_challenge": False,
        "retry_after_seconds": None,
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
    if sanitized.get("browser_challenge") is True:
        return False
    if sanitized.get("native_capture_error"):
        return False
    if (
        sanitized.get("account_identity_source")
        == CHATGPT_CONVERSATION_INIT_NATIVE_REQUEST_IDENTITY_SOURCE
        and not _is_verified_bound_identity(
            sanitized,
            sanitized.get("account_hash"),
        )
    ):
        return False
    if _http_status_failure(sanitized.get("status_code")) is not None:
        return False
    if sanitized.get("payload_state") != "present":
        return False
    payload = sanitized.get("payload")
    if not looks_like_conversation_init_payload(payload):
        return False
    if require_verified_identity:
        return bool(
            _is_verified_bound_identity(
                sanitized,
                sanitized.get("account_hash"),
            )
            and (
                expected_account_hash is None
                or sanitized.get("account_hash") == expected_account_hash
            )
        )
    return bool(sanitized.get("account_hash") or sanitized.get("source_identity_hash"))


def _destination_has_reusable_snapshot(path: str) -> bool:
    try:
        raw = load_conversation_init_source(path)
    except ChatGPTConversationInitError:
        return False
    sanitized = sanitize_conversation_init_boundary(raw, source_path=path)
    return _snapshot_is_persistable(sanitized)


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
    )


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
        if native_capture["account_hash"] != account_hash:
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
        if "native_capture" in raw:
            if account_hash:
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
