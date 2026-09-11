"""Read-only ChatGPT history adapter for fixture and browser transports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, cast

from .models import AdaptedPage, CapabilityRecord, ConversationSummary, MessageRecord
from .privacy import (
    ADAPTER_VERSION,
    classify_surface,
    sanitize_identity,
    sanitize_metadata,
    sanitize_token,
)
from .timeutil import parse_datetime

MODERN_INDEX = "/backend-api/conversations"
MODERN_DETAIL = "/backend-api/conversations/{conversation_id}"
MODERN_MESSAGES = "/backend-api/conversations/{conversation_id}/messages"
LEGACY_DETAIL = "/backend-api/conversation/{conversation_id}"
SESSION_ROUTE = "/api/auth/session"
INIT_ROUTE = "/backend-api/conversation/init"
ALLOWED_METHODS = {"GET"}
REQUIRED_IDENTITY_FIELDS = (
    "provider_user_id",
    "workspace_id",
    "quota_owner_id",
)
ALLOWED_PATH_PREFIXES = (
    "/backend-api/conversations",
    "/backend-api/conversation/",
    "/api/auth/session",
)


class AdapterError(RuntimeError):
    """Raised when a history page cannot be adapted safely."""


class AuthenticationRequiredError(AdapterError):
    """An authentication challenge requires the account to pause."""

    def __init__(self, message: str, *, status: int, path: str) -> None:
        super().__init__(message)
        self.status = status
        self.path = path


class RateLimitedError(AdapterError):
    """HTTP 429 from a history read; never a Chat Pro quota-exhaustion signal."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 429,
        retry_after: str | None = None,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.path = path


class HistoryTransport(Protocol):
    def request(self, method: str, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ...


@dataclass
class FixtureTransport:
    """Serve reviewed synthetic pages from a directory of JSON fixtures."""

    root: Path
    requests: list[dict[str, Any]]

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.requests = []

    def request(self, method: str, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        method = method.upper()
        if method not in ALLOWED_METHODS:
            raise AdapterError(f"method not allowlisted: {method} {path}")
        if not _is_allowed_path(path):
            raise AdapterError(f"path not allowlisted: {path}")
        self.requests.append({"method": method, "path": path, "params": dict(params or {})})
        payload = _load_fixture(self.root, method, path, params or {})
        _raise_if_authentication_required(payload, path)
        if str(payload.get("content_type") or "").lower().startswith("text/html"):
            raise AuthenticationRequiredError(
                f"HTML login page for {path}",
                status=int(payload.get("http_status") or 200),
                path=path,
            )
        _raise_if_rate_limited(payload, path)
        return payload


class ChatGPTHistoryAdapter:
    schema_version = ADAPTER_VERSION

    def __init__(
        self,
        transport: HistoryTransport,
        *,
        expected_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.expected_identity = dict(expected_identity or {})
        self.capabilities = CapabilityRecord(
            adapter_version=ADAPTER_VERSION,
            index_scopes=("active", "archived"),
            archive_behavior="query_param",
            project_coverage="unknown",
            modern_detail="available",
            pagination="offset_and_cursor",
            legacy_support="fallback_on_404_405",
            branch_visibility="include_has_versions",
            model_metadata="message_metadata",
            quota_metadata="optional_init",
        )

    def inspect_session(self) -> dict[str, Any]:
        try:
            payload = self.transport.request("GET", SESSION_ROUTE)
            _raise_if_authentication_required(payload, SESSION_ROUTE)
            _raise_if_rate_limited(payload, SESSION_ROUTE)
        except AuthenticationRequiredError:
            return {"surface": "unknown", "auth_state": "auth_required"}
        user = payload.get("user")
        user_mapping = user if isinstance(user, Mapping) else {}
        account = payload.get("account")
        account_mapping = account if isinstance(account, Mapping) else {}
        observed = {
            "provider_user_id": user_mapping.get("id")
            or payload.get("user_id")
            or payload.get("id"),
            "workspace_id": payload.get("workspace_id")
            or account_mapping.get("workspace_id"),
            "quota_owner_id": payload.get("quota_owner_id")
            or account_mapping.get("quota_owner_id"),
            "surface": classify_surface(payload, default=None),
        }
        identity = sanitize_identity(observed)
        authenticated = bool(
            user_mapping
            or payload.get("user_id")
            or payload.get("id")
        )
        if not authenticated:
            identity["auth_state"] = "auth_required"
            return identity

        errors: list[str] = []
        for field in REQUIRED_IDENTITY_FIELDS:
            expected = _optional_str(self.expected_identity.get(field))
            actual = identity.get(field)
            if expected is None:
                errors.append(f"missing_expected_{field}")
            elif actual is None:
                errors.append(f"missing_observed_{field}")
            elif actual != expected:
                errors.append(f"{field}_mismatch")
        identity["identity_errors"] = errors
        if any(error.startswith("missing_expected_") for error in errors):
            identity["auth_state"] = "unconfigured"
        elif errors:
            identity["auth_state"] = "identity_mismatch"
        else:
            identity["auth_state"] = "ready"
        return identity

    def list_conversations(
        self,
        *,
        archived: bool,
        offset: int = 0,
        limit: int = 100,
        order: str = "updated",
    ) -> AdaptedPage:
        payload = self.transport.request(
            "GET",
            MODERN_INDEX,
            {
                "offset": offset,
                "limit": limit,
                "order": order,
                "is_archived": str(archived).lower(),
            },
        )
        path = MODERN_INDEX
        _raise_if_authentication_required(payload, path)
        _raise_if_rate_limited(payload, path)
        return adapt_conversation_index(payload, archived=archived, offset=offset, limit=limit)

    def fetch_conversation(self, conversation_id: str) -> dict[str, Any]:
        modern_path = MODERN_DETAIL.format(conversation_id=conversation_id)
        payload = self.transport.request(
            "GET",
            modern_path,
            {"include_has_versions": "true", "num_turns": 100},
        )
        _raise_if_authentication_required(payload, modern_path)
        _raise_if_rate_limited(payload, modern_path)
        status = int(payload.get("http_status") or 200)
        if status in {404, 405}:
            legacy_path = LEGACY_DETAIL.format(conversation_id=conversation_id)
            legacy_payload = self.transport.request("GET", legacy_path)
            _raise_if_authentication_required(legacy_payload, legacy_path)
            _raise_if_rate_limited(legacy_payload, legacy_path)
            return legacy_payload
        return payload

    def fetch_messages(
        self,
        conversation_id: str,
        *,
        before: str | None = None,
        num_turns: int = 100,
        conversation_surface: str = "unknown",
    ) -> AdaptedPage:
        params: dict[str, Any] = {"include_has_versions": "true", "num_turns": num_turns}
        if before:
            params["before"] = before
        payload = self.transport.request(
            "GET",
            MODERN_MESSAGES.format(conversation_id=conversation_id),
            params,
        )
        path = MODERN_MESSAGES.format(conversation_id=conversation_id)
        _raise_if_authentication_required(payload, path)
        _raise_if_rate_limited(payload, path)
        return adapt_message_page(
            payload,
            conversation_id=conversation_id,
            conversation_surface=conversation_surface,
        )

    def close(self) -> None:
        closer = getattr(self.transport, "close", None)
        if callable(closer):
            closer()


def _load_fixture(root: Path, method: str, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("routes", []):
            if entry.get("method", "GET").upper() != method:
                continue
            if entry.get("path") != path:
                continue
            if not _params_match(entry.get("params") or {}, params):
                continue
            file_name = entry["fixture"]
            payload = json.loads((root / file_name).read_text(encoding="utf-8"))
            payload.setdefault("http_status", 200)
            return payload
    # Filename convention: conversations-active-offset-0.json
    archived = str(params.get("is_archived", "")).lower()
    offset = params.get("offset")
    before = params.get("before")
    candidates = []
    if "conversations" in path and path.count("/") == 2:
        label = "archived" if archived == "true" else "active"
        candidates.append(root / f"conversations-{label}-offset-{offset or 0}.json")
    if path.endswith("/messages"):
        conversation_id = path.split("/")[-2]
        suffix = before or "latest"
        candidates.append(root / f"messages-{conversation_id}-{suffix}.json")
    if path.startswith("/backend-api/conversations/") and not path.endswith("/messages"):
        conversation_id = path.rsplit("/", 1)[-1]
        candidates.append(root / f"conversation-{conversation_id}.json")
    if path.startswith("/backend-api/conversation/") and not path.endswith("/messages"):
        conversation_id = path.rsplit("/", 1)[-1]
        candidates.append(root / f"legacy-conversation-{conversation_id}.json")
    if path == SESSION_ROUTE:
        candidates.append(root / "session.json")
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload.setdefault("http_status", 200)
            return payload
    raise AdapterError(f"no fixture for {method} {path} {dict(params)}")


def _params_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if str(actual.get(key)) != str(value):
            return False
    return True


def adapt_conversation_index(
    payload: Mapping[str, Any],
    *,
    archived: bool,
    offset: int,
    limit: int,
) -> AdaptedPage:
    if payload.get("content_type") == "text/html" or isinstance(payload.get("items"), str):
        raise AdapterError("unrecognized conversation index: HTML or non-list items")
    items_raw = payload.get("items")
    if items_raw is None:
        items_raw = payload.get("conversations")
    if not isinstance(items_raw, list):
        return AdaptedPage(
            items=[],
            continuation=None,
            exhausted=False,
            schema_version=ADAPTER_VERSION,
            coverage="unrecognized",
            warnings=["missing items array"],
        )
    summaries: list[ConversationSummary] = []
    warnings: list[str] = []
    for item in items_raw:
        if not isinstance(item, Mapping):
            warnings.append("non-object conversation item")
            continue
        conversation_id = str(item.get("id") or item.get("conversation_id") or "").strip()
        if not conversation_id:
            warnings.append("conversation missing id")
            continue
        updated = parse_datetime(item.get("update_time") or item.get("updated_at") or item.get("create_time"))
        created = parse_datetime(item.get("create_time") or item.get("created_at"))
        surface = classify_surface(item, default=None)
        summaries.append(
            ConversationSummary(
                conversation_id=conversation_id,
                created_at=created,
                updated_at=updated,
                is_archived=bool(item.get("is_archived", archived)),
                workspace_id=_optional_str(item.get("workspace_id")),
                project_id=_optional_str(item.get("gizmo_id") or item.get("project_id")),
                surface=surface,
                origin=_optional_str(item.get("origin")),
                has_versions=item.get("has_versions") if isinstance(item.get("has_versions"), bool) else None,
                current_node=_optional_str(item.get("current_node")),
                coverage="validated_page",
            )
        )
    total = payload.get("total")
    continuation: str | int | None = None
    exhausted = False
    if isinstance(total, int) and offset + len(items_raw) >= total:
        exhausted = True
    elif payload.get("has_missing_conversations"):
        exhausted = False
        warnings.append("index reported missing conversations")
    elif len(items_raw) < limit:
        exhausted = True
    else:
        continuation = offset + len(items_raw)
    coverage = "validated_page"
    if warnings:
        coverage = "partial"
    if not isinstance(total, int) and continuation is None and not exhausted:
        coverage = "unrecognized"
    return AdaptedPage(
        items=summaries,
        continuation=continuation,
        exhausted=exhausted,
        schema_version=ADAPTER_VERSION,
        coverage=coverage,
        warnings=warnings,
    )


def adapt_message_page(
    payload: Mapping[str, Any],
    *,
    conversation_id: str,
    conversation_surface: str = "unknown",
) -> AdaptedPage:
    warnings: list[str] = []
    records: list[MessageRecord] = []
    has_mapping = isinstance(payload.get("mapping"), Mapping)
    has_messages = "messages" in payload
    messages_raw = payload.get("messages")
    if has_mapping:
        records.extend(
            iter_mapping_messages(
                payload["mapping"],
                conversation_id=conversation_id,
                warnings=warnings,
                conversation_surface=classify_surface(payload, default=conversation_surface),
            )
        )
    elif has_messages and isinstance(messages_raw, list):
        for item in messages_raw:
            if isinstance(item, Mapping):
                record = message_from_node(item, conversation_id=conversation_id, warnings=warnings, conversation_surface=conversation_surface)
                if record is not None:
                    records.append(record)
                else:
                    warnings.append("message item missing id")
            else:
                warnings.append("non-object message item")
    else:
        warnings.append("unrecognized_detail_shape")
        if has_messages:
            warnings.append("messages_not_array")
        if "mapping" in payload:
            warnings.append("mapping_not_object")
        return AdaptedPage(
            items=records,
            continuation=None,
            exhausted=False,
            schema_version=ADAPTER_VERSION,
            coverage="unrecognized",
            warnings=warnings,
        )

    # The legacy endpoint returns a complete mapping and has no modern cursor.
    # A modern messages response must prove whether older pages remain.
    if has_mapping and not has_messages and "page_info" not in payload:
        return AdaptedPage(
            items=records,
            continuation=None,
            exhausted=True,
            schema_version=ADAPTER_VERSION,
            coverage="partial" if warnings else "validated_page",
            warnings=warnings,
        )

    page_info_raw = payload.get("page_info")
    if not isinstance(page_info_raw, Mapping):
        warnings.append("missing_pagination_controls")
        return AdaptedPage(
            items=records,
            continuation=None,
            exhausted=False,
            schema_version=ADAPTER_VERSION,
            coverage="unrecognized",
            warnings=warnings,
        )

    cursor = page_info_raw.get("start_cursor")
    has_previous = page_info_raw.get("has_previous_page")
    exhausted = False
    continuation: str | int | None = None
    invalid_pagination = False
    if not isinstance(has_previous, bool):
        warnings.append("non_boolean_has_previous_page")
        invalid_pagination = True
    elif has_previous:
        if not isinstance(cursor, str) or not cursor.strip():
            warnings.append("has_previous_page_without_start_cursor")
            invalid_pagination = True
        else:
            continuation = cursor.strip()
    elif cursor not in (None, ""):
        warnings.append("terminal_page_has_cursor")
        invalid_pagination = True
    else:
        exhausted = True
    if payload.get("repeated_cursor"):
        warnings.append("repeated_cursor")
        exhausted = False
        continuation = None
    coverage = (
        "unrecognized"
        if invalid_pagination
        else "partial"
        if warnings
        else "validated_page"
    )
    return AdaptedPage(
        items=records,
        continuation=continuation,
        exhausted=exhausted,
        schema_version=ADAPTER_VERSION,
        coverage=coverage,
        warnings=warnings,
    )


def iter_mapping_messages(
    mapping: Mapping[str, Any],
    *,
    conversation_id: str,
    warnings: list[str],
    conversation_surface: str = "unknown",
) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    for node_id, node in mapping.items():
        if not isinstance(node, Mapping):
            warnings.append(f"non-object mapping node {node_id}")
            continue
        record = message_from_node(node, conversation_id=conversation_id, node_id=str(node_id), warnings=warnings, conversation_surface=conversation_surface)
        if record is not None:
            records.append(record)
    return records


def message_from_node(
    node: Mapping[str, Any],
    *,
    conversation_id: str,
    node_id: str | None = None,
    warnings: list[str],
    conversation_surface: str = "unknown",
) -> MessageRecord | None:
    message_raw: Any = node.get("message")
    message: Mapping[str, Any] = message_raw if isinstance(message_raw, Mapping) else node
    message_id = (
        sanitize_token(message.get("id"))
        or sanitize_token(node.get("id"))
        or sanitize_token(node_id)
    )
    if not message_id:
        warnings.append("message missing id")
        return None
    author_raw = message.get("author")
    author: Mapping[str, Any] = author_raw if isinstance(author_raw, Mapping) else {}
    author_role = sanitize_token(author.get("role"))
    metadata_raw = message.get("metadata")
    metadata = sanitize_metadata(
        metadata_raw if isinstance(metadata_raw, Mapping) else {}
    )
    children_raw = node.get("children") if isinstance(node.get("children"), list) else message.get("children")
    children = tuple(
        safe_child
        for item in (children_raw or [])
        if (safe_child := sanitize_token(item)) is not None
    )
    requested = (
        metadata.get("requested_model")
        or metadata.get("requested_model_slug")
        or metadata.get("model_slug")
        if author_role == "user"
        else metadata.get("requested_model")
    )
    recorded = None
    if author_role == "assistant":
        recorded = metadata.get("model_slug") or sanitize_metadata(
            {"model_slug": message.get("model_slug")}
        ).get("model_slug")
    return MessageRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        node_id=(
            sanitize_token(node.get("id"))
            or sanitize_token(node_id)
            or message_id
        ),
        parent_id=(
            sanitize_token(node.get("parent"))
            or sanitize_token(message.get("parent"))
            or sanitize_token(metadata.get("parent_id"))
        ),
        children=children,
        role=author_role,
        channel=(
            sanitize_token(message.get("channel"))
            or sanitize_token(metadata.get("channel"))
        ),
        created_at=parse_datetime(message.get("create_time") or node.get("create_time")),
        status=(
            sanitize_token(message.get("status"))
            or sanitize_token(metadata.get("status"))
        ),
        end_turn=message.get("end_turn") if isinstance(message.get("end_turn"), bool) else None,
        requested_model_raw=sanitize_token(requested) if author_role == "user" else sanitize_token(metadata.get("requested_model")),
        requested_mode_raw=sanitize_token(metadata.get("requested_mode")),
        requested_reasoning_effort_raw=sanitize_token(metadata.get("reasoning_effort")),
        recorded_final_model_raw=sanitize_token(recorded),
        generation_id=(
            sanitize_token(metadata.get("generation_id"))
            or sanitize_token(metadata.get("message_request_id"))
        ),
        request_id=sanitize_token(metadata.get("request_id")),
        surface=classify_surface(message, default=classify_surface(metadata, default=conversation_surface)),
        origin=(
            sanitize_token(metadata.get("origin"))
            or ("shared" if metadata.get("from_shared") is True else None)
        ),
        metadata=metadata,
    )


def _raise_if_rate_limited(payload: Mapping[str, Any], path: str) -> None:
    status = int(payload.get("http_status") or 200)
    if status != 429:
        return
    headers = cast(
        Mapping[str, Any],
        payload.get("headers") if isinstance(payload.get("headers"), Mapping) else {},
    )
    retry_after = payload.get("retry_after") or headers.get("Retry-After") or headers.get("retry-after")
    raise RateLimitedError(
        f"rate limited (429) for {path}; no legacy fallback and not quota exhaustion",
        status=429,
        retry_after=None if retry_after is None else str(retry_after),
        path=path,
    )


def _raise_if_authentication_required(payload: Mapping[str, Any], path: str) -> None:
    status = int(payload.get("http_status") or 200)
    content_type = str(payload.get("content_type") or "").lower()
    if status not in {401, 403} and not content_type.startswith("text/html"):
        return
    raise AuthenticationRequiredError(
        f"authentication required ({status}) for {path}; legacy fallback is disabled",
        status=status if status in {401, 403} else 401,
        path=path,
    )


def _is_allowed_path(path: str) -> bool:
    if "?" in path or "#" in path:
        return False
    if path.endswith("/delete") or "/delete/" in path:
        return False
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in ALLOWED_PATH_PREFIXES)


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "" or value is False:
        return None
    if value is True:
        return "true"
    return str(value)
