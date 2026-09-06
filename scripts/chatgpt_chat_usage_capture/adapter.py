"""Read-only ChatGPT history adapter for fixture and browser transports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from .models import AdaptedPage, CapabilityRecord, ConversationSummary, MessageRecord
from .privacy import (
    ADAPTER_VERSION,
    classify_surface,
    sanitize_identity,
    sanitize_mapping,
)
from .timeutil import parse_datetime

MODERN_INDEX = "/backend-api/conversations"
MODERN_DETAIL = "/backend-api/conversations/{conversation_id}"
MODERN_MESSAGES = "/backend-api/conversations/{conversation_id}/messages"
LEGACY_DETAIL = "/backend-api/conversation/{conversation_id}"
SESSION_ROUTE = "/api/auth/session"
INIT_ROUTE = "/backend-api/conversation/init"
ALLOWED_METHODS = {"GET"}
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
        identity = sanitize_identity(
            {
                "provider_user_id": payload.get("user", {}).get("id")
                if isinstance(payload.get("user"), Mapping)
                else payload.get("user_id") or payload.get("id"),
                "workspace_id": payload.get("workspace_id") or payload.get("account", {}).get("id")
                if isinstance(payload.get("account"), Mapping)
                else payload.get("workspace_id"),
                "quota_owner_id": payload.get("quota_owner_id")
                or payload.get("account", {}).get("id")
                if isinstance(payload.get("account"), Mapping)
                else payload.get("quota_owner_id"),
                "surface": classify_surface(payload, default=None),
                "auth_state": "ready" if payload.get("accessToken") or payload.get("user") else "auth_required",
            }
        )
        expected_user = self.expected_identity.get("provider_user_id")
        expected_workspace = self.expected_identity.get("workspace_id")
        if expected_user and identity.get("provider_user_id") not in {expected_user, None}:
            identity["auth_state"] = "identity_mismatch"
        if expected_workspace and identity.get("workspace_id") not in {expected_workspace, None}:
            identity["auth_state"] = "identity_mismatch"
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


def adapt_message_page(payload: Mapping[str, Any], *, conversation_id: str, conversation_surface: str = "unknown") -> AdaptedPage:
    warnings: list[str] = []
    records: list[MessageRecord] = []
    if isinstance(payload.get("mapping"), Mapping):
        records.extend(
            iter_mapping_messages(
                payload["mapping"],
                conversation_id=conversation_id,
                warnings=warnings,
                conversation_surface=classify_surface(payload, default=conversation_surface),
            )
        )
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, Mapping):
                record = message_from_node(item, conversation_id=conversation_id, warnings=warnings, conversation_surface=conversation_surface)
                if record is not None:
                    records.append(record)
    page_info_raw = payload.get("page_info")
    page_info: Mapping[str, Any] = page_info_raw if isinstance(page_info_raw, Mapping) else {}
    cursor = page_info.get("start_cursor")
    has_previous = page_info.get("has_previous_page")
    exhausted = has_previous is False
    continuation = cursor if has_previous else None
    if has_previous is True and not cursor:
        warnings.append("has_previous_page without start_cursor")
        coverage = "unrecognized"
        exhausted = False
        continuation = None
    elif has_previous is None and not records and not payload.get("mapping"):
        coverage = "unrecognized"
        exhausted = False
    else:
        coverage = "partial" if warnings else "validated_page"
    if payload.get("repeated_cursor"):
        warnings.append("repeated_cursor")
        coverage = "partial"
        exhausted = False
        continuation = None
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
    message_id = str(message.get("id") or node.get("id") or node_id or "").strip()
    if not message_id:
        warnings.append("message missing id")
        return None
    author_raw = message.get("author")
    author: Mapping[str, Any] = author_raw if isinstance(author_raw, Mapping) else {}
    metadata_raw = message.get("metadata")
    metadata: Mapping[str, Any] = metadata_raw if isinstance(metadata_raw, Mapping) else {}
    children_raw = node.get("children") if isinstance(node.get("children"), list) else message.get("children")
    children = tuple(str(item) for item in (children_raw or []) if item)
    requested = (
        metadata.get("requested_model")
        or metadata.get("requested_model_slug")
        or metadata.get("model_slug")
        if author.get("role") == "user"
        else metadata.get("requested_model")
    )
    recorded = None
    if author.get("role") == "assistant":
        recorded = metadata.get("model_slug") or message.get("model_slug")
    return MessageRecord(
        conversation_id=conversation_id,
        message_id=message_id,
        node_id=str(node.get("id") or node_id or message_id),
        parent_id=_optional_str(node.get("parent") or message.get("parent") or metadata.get("parent_id")),
        children=children,
        role=_optional_str(author.get("role")),
        channel=_optional_str(message.get("channel") or metadata.get("channel")),
        created_at=parse_datetime(message.get("create_time") or node.get("create_time")),
        status=_optional_str(message.get("status") or metadata.get("status")),
        end_turn=message.get("end_turn") if isinstance(message.get("end_turn"), bool) else None,
        requested_model_raw=_optional_str(requested) if author.get("role") == "user" else _optional_str(metadata.get("requested_model")),
        requested_mode_raw=_optional_str(metadata.get("requested_mode")),
        requested_reasoning_effort_raw=_optional_str(metadata.get("reasoning_effort")),
        recorded_final_model_raw=_optional_str(recorded),
        generation_id=_optional_str(metadata.get("generation_id") or metadata.get("message_request_id")),
        request_id=_optional_str(metadata.get("request_id")),
        surface=classify_surface(message, default=classify_surface(metadata, default=conversation_surface)),
        origin=_optional_str(metadata.get("origin") or metadata.get("from_shared") and "shared"),
        metadata=sanitize_mapping(metadata),
    )


def _raise_if_rate_limited(payload: Mapping[str, Any], path: str) -> None:
    status = int(payload.get("http_status") or 200)
    if status != 429:
        return
    headers = payload.get("headers") if isinstance(payload.get("headers"), Mapping) else {}
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
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in ALLOWED_PATH_PREFIXES)


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "" or value is False:
        return None
    if value is True:
        return "true"
    return str(value)
