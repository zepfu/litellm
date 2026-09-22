"""Bounded OpenCode Go rejection evidence for attempt and terminal audit.

OC-030: persist secret-safe Go target, tool, phase, and failure evidence on
the durable attempt record and alias/direct terminal audit contracts.

The payload never includes full URLs, exception objects, prompts, tool
parameters, query, userinfo, keys, or arbitrary provider response content.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit
from uuid import uuid4

OPENCODE_GO_REJECTION_KEY = "opencode_go_rejection"
OPENCODE_GO_REJECTION_STATE_KEY = "opencode_go_provider_rejection_evidence"
OPENCODE_GO_LOGGER_EXTRA_KEY = "opencode_go_provider_rejection"
OPENCODE_GO_PROVIDER = "opencode_go"
OPENCODE_GO_ROUTE_FAMILY = "codex_opencode_go_adapter"
OPENCODE_GO_EXPECTED_TARGET_FAMILY = "opencode"
OPENCODE_GO_CHAT_COMPLETIONS_PATH = "/zen/go/v1/chat/completions"
OPENCODE_GO_RESPONSES_PATH = "/zen/go/v1/responses"
OPENCODE_GO_DIRECT_ALIAS_FAMILY = "codex_opencode_go"

_MAX_INDEX = 4096
_MAX_TOOL_TYPES = 64
_MAX_IDENTIFIER_CHARS = 64
_TOOLS_INDEX_RE = re.compile(r"tools\[(\d+)\]")
_TOOL_TYPE_RE = re.compile(
    rf"[A-Za-z][A-Za-z0-9_]{{0,{_MAX_IDENTIFIER_CHARS - 1}}}\Z"
)
_FAMILY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_PHASES = frozenset(
    {
        "candidate_preflight",
        "request_preparation",
        "provider_attempt",
        "provider_dispatch",
        "upstream_4xx",
    }
)
_CALL_MODES = frozenset({"alias", "direct"})
_FAILURE_CLASSES = frozenset(
    {
        "auth",
        "rate_limit",
        "transient",
        "provider_5xx",
        "provider_4xx_other",
        "unknown",
    }
)
_KNOWN_TARGET_PATHS = {
    OPENCODE_GO_CHAT_COMPLETIONS_PATH: OPENCODE_GO_EXPECTED_TARGET_FAMILY,
    OPENCODE_GO_RESPONSES_PATH: OPENCODE_GO_EXPECTED_TARGET_FAMILY,
    "/zen/v1/chat/completions": "opencode_zen",
    "/zen/v1/responses": "opencode_zen",
}
_ALLOWED_EVIDENCE_KEYS = frozenset(
    {
        "provider",
        "route_family",
        "route",
        "expected_target_family",
        "actual_target_family",
        "actual_target_path",
        "target_url_family",
        "status",
        "failure_class",
        "failure_phase",
        "call_mode",
        "tool_index",
        "tool_type",
        "tool_count",
        "tool_types",
        "offending_index",
        "offending_type",
        "request_identity",
        "litellm_call_id",
        "originating_attempt_id",
        "error",
    }
)
_FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "arguments",
        "api_key",
        "authorization",
        "body",
        "cookie",
        "credential",
        "detail",
        "exception",
        "headers",
        "message",
        "parameters",
        "password",
        "prompt",
        "query",
        "raw",
        "secret",
        "target_url",
        "token",
        "userinfo",
    }
)


def _bounded_index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > _MAX_INDEX:
        return None
    return value


def _bounded_status(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 100 or value > 599:
        return None
    return value


def _safe_tool_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or _TOOL_TYPE_RE.fullmatch(token) is None:
        return None
    return token


def _safe_family(value: Any, *, default: str) -> str:
    if isinstance(value, str):
        token = value.strip()
        if token and _FAMILY_RE.fullmatch(token) is not None:
            return token
    return default


def _safe_phase(value: Any) -> str:
    if isinstance(value, str) and value.strip() in _PHASES:
        return value.strip()
    return "provider_attempt"


def _safe_call_mode(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip() in _CALL_MODES:
        return value.strip()
    return None


def _safe_failure_class(value: Any) -> str:
    if isinstance(value, str) and value.strip() in _FAILURE_CLASSES:
        return value.strip()
    return "unknown"


def _safe_identity(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > 128:
        return None
    if any(not (character.isalnum() or character in "._-") for character in token):
        return None
    return token


def opencode_go_tool_type(tool: Any) -> Optional[str]:
    if isinstance(tool, str):
        return _safe_tool_type(tool)
    if not isinstance(tool, dict):
        return None
    tool_type = _safe_tool_type(tool.get("type"))
    if tool_type is not None:
        return tool_type
    function = tool.get("function")
    if isinstance(function, dict):
        return "function"
    return None


def opencode_go_tool_types(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    summarized: list[str] = []
    for tool in tools[:_MAX_TOOL_TYPES]:
        summarized.append(opencode_go_tool_type(tool) or "unknown")
    return summarized


def extract_opencode_go_offending_tool_index(message: Any) -> Optional[int]:
    text = str(message or "")
    match = _TOOLS_INDEX_RE.search(text)
    if match is None:
        return None
    try:
        return _bounded_index(int(match.group(1)))
    except (TypeError, ValueError):
        return None


def _exception_status(exc: Any) -> Optional[int]:
    status = _bounded_status(getattr(exc, "status_code", None))
    if status is not None:
        return status
    response = getattr(exc, "response", None)
    return _bounded_status(getattr(response, "status_code", None))


def _exception_index_source(exc: Any) -> str:
    parts: list[str] = []
    for value in (
        getattr(exc, "message", None),
        getattr(exc, "detail", None),
        str(exc) if exc is not None else "",
    ):
        if value is None:
            continue
        if isinstance(value, Mapping):
            continue
        text = str(value)
        if text:
            parts.append(text)
    return " ".join(parts)


def _failure_class_from_status(*, status: Optional[int], exc: Any) -> str:
    if isinstance(exc, asyncio.TimeoutError) or (
        isinstance(exc, BaseException) and type(exc).__name__ == "TimeoutError"
    ):
        return "transient"
    if status == 408:
        return "transient"
    if status in {401, 403}:
        return "auth"
    if status == 429:
        return "rate_limit"
    if status is not None and 500 <= status <= 599:
        return "provider_5xx"
    if status is not None and 400 <= status <= 499:
        return "provider_4xx_other"
    return "unknown"


def _bounded_target_path(target_url: Any) -> str:
    raw = str(target_url or "").strip()
    path = ""
    if raw:
        try:
            parsed = urlsplit(raw)
        except Exception:
            parsed = None
        if parsed is not None and (parsed.scheme or parsed.netloc or parsed.path):
            path = parsed.path or ""
        else:
            path = raw.split("?", 1)[0].split("#", 1)[0]
    for known_path in _KNOWN_TARGET_PATHS:
        if known_path in path or path.endswith(known_path):
            return known_path
    return "unknown"


def _actual_target_family(path: str) -> str:
    return _KNOWN_TARGET_PATHS.get(path, "unknown")


def _provider_bound_tool_index_and_type(
    *,
    index: Optional[int],
    provider_bound_types: list[str],
) -> tuple[Optional[int], Optional[str]]:
    if index is None:
        return None, None
    if 0 <= index < len(provider_bound_types):
        return index, provider_bound_types[index]
    return None, None


def _new_originating_attempt_id() -> str:
    return str(uuid4())


def _originating_attempt_id(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        nested = getattr(payload, OPENCODE_GO_REJECTION_KEY, None)
        if isinstance(nested, Mapping):
            payload = nested
        else:
            return None
    nested = payload.get(OPENCODE_GO_REJECTION_KEY)
    if isinstance(nested, Mapping):
        identity = _safe_identity(nested.get("originating_attempt_id"))
        if identity is not None:
            return identity
    return _safe_identity(payload.get("originating_attempt_id"))


def _ensure_originating_attempt_id(payload: Mapping[str, Any]) -> dict[str, Any]:
    identity = _originating_attempt_id(payload) or _new_originating_attempt_id()
    return {**payload, "originating_attempt_id": identity}


def _with_correlation_ids(
    normalized: Mapping[str, Any],
    payload: Mapping[str, Any],
    identity: Optional[str],
) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if identity is not None:
        extra["request_identity"] = identity
        extra["litellm_call_id"] = identity
    originating_attempt_id = _originating_attempt_id(payload)
    if originating_attempt_id is not None:
        extra["originating_attempt_id"] = originating_attempt_id
    return {**normalized, **extra}


def _evidence_matches_originating_attempt(
    diagnostic: Mapping[str, Any],
    originating_attempt_id: Optional[str],
) -> bool:
    if originating_attempt_id is None:
        return True
    source_id = _originating_attempt_id(diagnostic)
    return source_id is not None and source_id == originating_attempt_id


def _bind_request_identity(request: Any) -> Optional[str]:
    state = getattr(request, "state", None)
    if state is None:
        return None
    existing = getattr(state, "aawm_alias_request_litellm_call_id", None)
    identity = _safe_identity(existing)
    if identity is not None:
        return identity
    for key in ("litellm_call_id", "call_id", "request_id"):
        identity = _safe_identity(getattr(state, key, None))
        if identity is not None:
            break
    if identity is None:
        identity = str(uuid4())
    try:
        setattr(state, "aawm_alias_request_litellm_call_id", identity)
    except Exception:
        return identity
    return identity


def build_opencode_go_rejection_evidence(
    *,
    target_url: Any,
    exc: BaseException,
    advertised_tools: Any = None,
    completion_tools: Any = None,
    api_key: Any = None,
    request: Any = None,
    expected_target_family: Any = OPENCODE_GO_EXPECTED_TARGET_FAMILY,
    failure_phase: Any = "provider_attempt",
    call_mode: Any = None,
) -> dict[str, Any]:
    """Build one secret-safe Go rejection record.

    ``advertised_tools`` and ``api_key`` are accepted for call-site
    compatibility only. Tool identity is resolved against the final
    provider-bound ``completion_tools`` list. Keys, URLs, and exception
    payloads are never copied into the result.
    """

    _ = advertised_tools, api_key
    provider_bound_types = opencode_go_tool_types(completion_tools)
    status = _exception_status(exc)
    index = extract_opencode_go_offending_tool_index(_exception_index_source(exc))
    tool_index, tool_type = _provider_bound_tool_index_and_type(
        index=index,
        provider_bound_types=provider_bound_types,
    )
    actual_target_path = _bounded_target_path(target_url)
    identity = _bind_request_identity(request)
    originating_attempt_id = _new_originating_attempt_id()
    failure_class = _failure_class_from_status(status=status, exc=exc)
    evidence: dict[str, Any] = {
        "provider": OPENCODE_GO_PROVIDER,
        "route_family": OPENCODE_GO_ROUTE_FAMILY,
        "route": OPENCODE_GO_ROUTE_FAMILY,
        "expected_target_family": _safe_family(
            expected_target_family,
            default=OPENCODE_GO_EXPECTED_TARGET_FAMILY,
        ),
        "actual_target_family": _actual_target_family(actual_target_path),
        "actual_target_path": actual_target_path,
        "target_url_family": actual_target_path,
        "status": status,
        "failure_class": failure_class,
        "failure_phase": _safe_phase(failure_phase),
        "tool_count": len(provider_bound_types),
        "error": {"status": status},
    }
    mode = _safe_call_mode(call_mode)
    if mode is not None:
        evidence["call_mode"] = mode
    if tool_index is not None:
        evidence["tool_index"] = tool_index
        evidence["offending_index"] = tool_index
    if tool_type is not None:
        evidence["tool_type"] = tool_type
        evidence["offending_type"] = tool_type
    if provider_bound_types:
        evidence["tool_types"] = list(provider_bound_types)
    if identity is not None:
        evidence["request_identity"] = identity
        evidence["litellm_call_id"] = identity
    evidence["originating_attempt_id"] = originating_attempt_id
    return normalize_opencode_go_rejection(evidence) or evidence


def _is_opencode_go_rejection_payload(payload: Mapping[str, Any]) -> bool:
    if payload.get("route") == OPENCODE_GO_ROUTE_FAMILY:
        return True
    if payload.get("call_mode") in _CALL_MODES:
        return True
    if payload.get("target_url_family") in _KNOWN_TARGET_PATHS:
        return True
    if payload.get("actual_target_path") in _KNOWN_TARGET_PATHS:
        return True
    return False


def normalize_opencode_go_rejection(
    payload: Any,
    *,
    provider: Any = None,
    route_family: Any = None,
) -> Optional[dict[str, Any]]:
    """Return a copy that contains only the bounded allowlisted fields."""

    if not isinstance(payload, Mapping):
        return None
    nested = payload.get(OPENCODE_GO_REJECTION_KEY)
    if isinstance(nested, Mapping):
        payload = nested
    if not _is_opencode_go_rejection_payload(payload):
        return None
    payload = {
        key: value
        for key, value in payload.items()
        if key not in _FORBIDDEN_EVIDENCE_KEYS
    }
    status = _bounded_status(payload.get("status"))
    error_payload = payload.get("error")
    if status is None and isinstance(error_payload, Mapping):
        status = _bounded_status(error_payload.get("status"))
    actual_target_path = payload.get("actual_target_path") or payload.get(
        "target_url_family"
    )
    if actual_target_path not in _KNOWN_TARGET_PATHS and actual_target_path != "unknown":
        actual_target_path = _bounded_target_path(actual_target_path)
    if not actual_target_path:
        actual_target_path = "unknown"
    actual_target_family = payload.get("actual_target_family")
    if actual_target_family not in set(_KNOWN_TARGET_PATHS.values()) | {"unknown"}:
        actual_target_family = _actual_target_family(str(actual_target_path))
    tool_types = [
        token
        for token in opencode_go_tool_types(payload.get("tool_types"))
        if token
    ]
    if not tool_types and isinstance(payload.get("tool_types"), list):
        tool_types = [
            _safe_tool_type(item) or "unknown"
            for item in payload.get("tool_types")[:_MAX_TOOL_TYPES]
            if isinstance(item, str)
        ]
    tool_index = _bounded_index(
        payload.get("tool_index", payload.get("offending_index"))
    )
    tool_type = _safe_tool_type(
        payload.get("tool_type", payload.get("offending_type"))
    )
    if tool_index is not None and tool_type is None and tool_types:
        if 0 <= tool_index < len(tool_types):
            tool_type = tool_types[tool_index]
        else:
            tool_index = None
    identity = _safe_identity(
        payload.get("request_identity") or payload.get("litellm_call_id")
    )
    tool_count = payload.get("tool_count")
    if not isinstance(tool_count, int) or isinstance(tool_count, bool) or tool_count < 0:
        tool_count = len(tool_types)
    elif tool_count > _MAX_TOOL_TYPES:
        tool_count = _MAX_TOOL_TYPES
    normalized: dict[str, Any] = {
        "provider": _safe_family(
            payload.get("provider") or provider,
            default=OPENCODE_GO_PROVIDER,
        ),
        "route_family": _safe_family(
            payload.get("route_family") or payload.get("route") or route_family,
            default=OPENCODE_GO_ROUTE_FAMILY,
        ),
        "route": OPENCODE_GO_ROUTE_FAMILY,
        "expected_target_family": _safe_family(
            payload.get("expected_target_family"),
            default=OPENCODE_GO_EXPECTED_TARGET_FAMILY,
        ),
        "actual_target_family": actual_target_family,
        "actual_target_path": actual_target_path,
        "target_url_family": actual_target_path,
        "status": status,
        "failure_class": _safe_failure_class(payload.get("failure_class")),
        "failure_phase": _safe_phase(payload.get("failure_phase")),
        "tool_count": tool_count,
        "error": {"status": status},
    }
    mode = _safe_call_mode(payload.get("call_mode"))
    if mode is not None:
        normalized["call_mode"] = mode
    if tool_index is not None:
        normalized["tool_index"] = tool_index
        normalized["offending_index"] = tool_index
    if tool_type is not None:
        normalized["tool_type"] = tool_type
        normalized["offending_type"] = tool_type
    if tool_types:
        normalized["tool_types"] = tool_types
    normalized = _with_correlation_ids(normalized, payload, identity)
    return {key: value for key, value in normalized.items() if key in _ALLOWED_EVIDENCE_KEYS}


def _iter_explicit_rejection_sources(*sources: Any) -> list[Any]:
    return [source for source in sources if source is not None]


def _iter_exception_rejection_sources(exc: Any = None) -> list[Any]:
    collected: list[Any] = []
    current = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        collected.append(getattr(current, OPENCODE_GO_REJECTION_KEY, None))
        collected.append(getattr(current, OPENCODE_GO_REJECTION_STATE_KEY, None))
        detail = getattr(current, "detail", None)
        if isinstance(detail, Mapping):
            collected.append(detail.get(OPENCODE_GO_REJECTION_KEY))
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return collected


def _iter_request_rejection_sources(request: Any = None) -> list[Any]:
    collected: list[Any] = []
    if request is None:
        return collected
    state = getattr(request, "state", None)
    if state is None:
        return collected
    collected.append(getattr(state, OPENCODE_GO_REJECTION_STATE_KEY, None))
    collected.append(getattr(state, OPENCODE_GO_REJECTION_KEY, None))
    extra = getattr(state, "opencode_go_logger_extra", None)
    if isinstance(extra, Mapping):
        collected.append(extra.get(OPENCODE_GO_LOGGER_EXTRA_KEY))
        collected.append(extra.get(OPENCODE_GO_REJECTION_KEY))
    return collected


def _first_recognizable_rejection(
    sources: Any,
    *,
    provider: Any = None,
    route_family: Any = None,
    originating_attempt_id: Optional[str] = None,
    require_originating_attempt: bool = False,
) -> Optional[dict[str, Any]]:
    bound_id = _safe_identity(originating_attempt_id)
    if require_originating_attempt and bound_id is None:
        return None
    for source in sources:
        diagnostic = normalize_opencode_go_rejection(
            source,
            provider=provider,
            route_family=route_family,
        )
        if diagnostic is None:
            continue
        if not _evidence_matches_originating_attempt(diagnostic, bound_id):
            continue
        return diagnostic
    return None


def extract_opencode_go_rejection(
    *sources: Any,
    request: Any = None,
    exc: Any = None,
    provider: Any = None,
    route_family: Any = None,
    originating_attempt_id: Any = None,
) -> Optional[dict[str, Any]]:
    bound_id = _safe_identity(originating_attempt_id)
    explicit = _first_recognizable_rejection(
        _iter_explicit_rejection_sources(*sources),
        provider=provider,
        route_family=route_family,
        originating_attempt_id=bound_id,
    )
    if explicit is not None:
        return explicit
    from_exception = _first_recognizable_rejection(
        _iter_exception_rejection_sources(exc),
        provider=provider,
        route_family=route_family,
        originating_attempt_id=bound_id,
    )
    if from_exception is not None:
        return from_exception
    return _first_recognizable_rejection(
        _iter_request_rejection_sources(request),
        provider=provider,
        route_family=route_family,
        originating_attempt_id=bound_id,
        require_originating_attempt=True,
    )


def _bound_originating_attempt_id(*payloads: Any) -> Optional[str]:
    for payload in payloads:
        diagnostic = normalize_opencode_go_rejection(payload)
        if diagnostic is None:
            continue
        identity = _originating_attempt_id(diagnostic)
        if identity is not None:
            return identity
    return None


def attach_opencode_go_rejection(
    *,
    target: dict[str, Any],
    request: Any = None,
    exc: Any = None,
    candidate: Optional[Mapping[str, Any]] = None,
    diagnostic: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Copy one normalized Go rejection onto ``target`` when present."""

    candidate_mapping = candidate if isinstance(candidate, Mapping) else {}
    provider = candidate_mapping.get("provider") or target.get("provider")
    route_family = candidate_mapping.get("route_family") or target.get(
        "route_family"
    )
    bound_id = _bound_originating_attempt_id(target, candidate_mapping)
    recorded = normalize_opencode_go_rejection(
        diagnostic,
        provider=provider,
        route_family=route_family,
    )
    if recorded is not None and not _evidence_matches_originating_attempt(
        recorded,
        bound_id,
    ):
        recorded = None
    if recorded is None:
        recorded = extract_opencode_go_rejection(
            target,
            candidate_mapping,
            request=request,
            exc=exc,
            provider=provider,
            route_family=route_family,
            originating_attempt_id=bound_id,
        )
    if recorded is None:
        return None
    existing = normalize_opencode_go_rejection(
        target,
        provider=provider,
        route_family=route_family,
    )
    if existing is not None:
        existing_id = _originating_attempt_id(existing)
        recorded_id = _originating_attempt_id(recorded)
        if existing_id is not None and existing_id != recorded_id:
            recorded = existing
    if recorded.get("request_identity") is None:
        identity = _bind_request_identity(request) or _safe_identity(
            target.get("request_identity") or target.get("litellm_call_id")
        )
        if identity is not None:
            recorded = {
                **recorded,
                "request_identity": identity,
                "litellm_call_id": identity,
            }
    target[OPENCODE_GO_REJECTION_KEY] = recorded
    if recorded.get("status") is not None and target.get("error_status_code") is None:
        target["error_status_code"] = recorded["status"]
    if recorded.get("failure_class") and not target.get("error_class"):
        target["error_class"] = recorded["failure_class"]
    if recorded.get("failure_phase") and not target.get("failure_phase"):
        target["failure_phase"] = recorded["failure_phase"]
    return recorded


def record_opencode_go_rejection_evidence(
    request: Any,
    evidence: Mapping[str, Any],
    *,
    exc: Any = None,
) -> dict[str, Any]:
    """Store the bounded record on request state and the raising exception."""

    recorded = (
        normalize_opencode_go_rejection(evidence)
        or dict(evidence)
    )
    recorded = _ensure_originating_attempt_id(recorded)
    identity = recorded.get("request_identity") or _bind_request_identity(request)
    if identity is not None and recorded.get("request_identity") is None:
        recorded = {
            **recorded,
            "request_identity": identity,
            "litellm_call_id": identity,
        }
    state = getattr(request, "state", None)
    if state is not None:
        try:
            setattr(state, OPENCODE_GO_REJECTION_STATE_KEY, recorded)
            setattr(state, OPENCODE_GO_REJECTION_KEY, recorded)
        except Exception:
            pass
        extra = getattr(state, "opencode_go_logger_extra", None)
        if not isinstance(extra, dict):
            extra = {}
            try:
                setattr(state, "opencode_go_logger_extra", extra)
            except Exception:
                extra = {}
        extra[OPENCODE_GO_LOGGER_EXTRA_KEY] = recorded
        extra[OPENCODE_GO_REJECTION_KEY] = recorded
    if exc is not None:
        try:
            setattr(exc, OPENCODE_GO_REJECTION_KEY, recorded)
        except Exception:
            pass
    return recorded
