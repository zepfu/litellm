"""Claude permission-check detection, auto-review identity, and parent-identity inheritance.

Behavior-preserving Wave A4B extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports of identity helpers are intentionally
absent here."""

import re
from datetime import datetime, timezone
from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:

    def _append_langfuse_span(
        metadata: Dict[str, Any],
        *,
        name: str,
        span_metadata: Optional[Dict[str, Any]] = None,
        input_data: Any = None,
        output_data: Any = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None: ...

    def _clean_non_empty_string(value: Any) -> Optional[str]: ...

    def _content_to_text(content: Any) -> str: ...

    def _extract_first_response_message(result: Any) -> Any: ...

    def _first_non_empty_string(*values: Any) -> Optional[str]: ...

    def _maybe_get(obj: Any, key: str, default: Any = None) -> Any: ...

    def _maybe_get_path(obj: Any, *keys: str, default: Any = None) -> Any: ...

    def _maybe_parse_json_text(value: str) -> Any: ...

    def _merge_tags(metadata: Dict[str, Any], tags_to_add: List[str]) -> None: ...

    def _normalize_repository_identity(value: Any) -> Optional[str]: ...

    def _parse_datetime_value(value: Any) -> Optional[datetime]: ...

    def _safe_float(value: Any) -> Optional[float]: ...

    def _safe_int(value: Any) -> Optional[int]: ...

_CLAUDE_PERMISSION_CHECK_OUTPUT_RE = re.compile(
    r"^<block>\s*(?P<decision>yes|no)\s*$",
    re.IGNORECASE,
)

_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL = "claude-auto-review"

_CLAUDE_AUTO_REVIEW_TRACE_NAME = "claude-code.auto-reviewer"

_CLAUDE_AUTO_REVIEW_AGENT_NAME = "auto-reviewer"


def _permission_check_probeable_value(value: Any) -> bool:
    """True when *value* is a concrete response-shaped container we should walk.

    Restricts attribute probing to dicts and objects that already expose the
    known fields, so free-form getattr on test doubles / arbitrary objects is
    not required in production code.
    """
    if isinstance(value, (str, list, dict)):
        return True
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return False
    for key in ("content", "choices", "response", "message"):
        try:
            if isinstance(value, dict) and key in value:
                return True
            obj_dict = getattr(value, "__dict__", None)
            if isinstance(obj_dict, dict) and key in obj_dict:
                return True
        except Exception:
            continue
    return False


def _extract_claude_permission_check_decision_from_value(
    value: Any,
    *,
    _depth: int = 0,
) -> Optional[str]:
    if value is None or _depth > 8:
        return None

    if isinstance(value, str):
        stripped_value = value.strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(stripped_value)
        if match is not None:
            return match.group("decision").lower()
        parsed_value = _maybe_parse_json_text(stripped_value)
        if parsed_value is not None:
            return _extract_claude_permission_check_decision_from_value(parsed_value, _depth=_depth + 1)
        return None

    if isinstance(value, list):
        text_value = _content_to_text(value).strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(text_value)
        if match is not None:
            return match.group("decision").lower()
        for item in value:
            decision = _extract_claude_permission_check_decision_from_value(item, _depth=_depth + 1)
            if decision is not None:
                return decision
        return None

    if not _permission_check_probeable_value(value):
        return None

    content = _maybe_get(value, "content")
    if content is not None and content is not value:
        decision = _extract_claude_permission_check_decision_from_value(content, _depth=_depth + 1)
        if decision is not None:
            return decision

    message = _extract_first_response_message(value)
    if message is not None and message is not value:
        decision = _extract_claude_permission_check_decision_from_value(message, _depth=_depth + 1)
        if decision is not None:
            return decision

    response = _maybe_get(value, "response")
    if response is not None and response is not value:
        decision = _extract_claude_permission_check_decision_from_value(response, _depth=_depth + 1)
        if decision is not None:
            return decision

    return None


def _extract_claude_permission_check_decision(
    result: Any,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    decision = _extract_claude_permission_check_decision_from_value(result)
    if decision is not None:
        return decision

    if isinstance(standard_logging_object, dict):
        for candidate in (
            standard_logging_object.get("response"),
            standard_logging_object.get("output"),
        ):
            decision = _extract_claude_permission_check_decision_from_value(candidate)
            if decision is not None:
                return decision

    return None


def _extract_claude_permission_check_models(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
) -> Tuple[Optional[str], Optional[str]]:
    request_model = _first_non_empty_string(
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(
            kwargs.get("litellm_params"),
            "proxy_server_request",
            "body",
            "model",
        ),
        _maybe_get_path(standard_logging_object, "request_body", "model"),
    )
    response_model = _first_non_empty_string(
        _maybe_get(result, "model"),
        _maybe_get_path(standard_logging_object, "response", "model"),
        standard_logging_object.get("model"),
        kwargs.get("model"),
        metadata.get("model"),
    )
    return request_model, response_model


def _enrich_claude_permission_check_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
    *,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> None:
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    decision = _extract_claude_permission_check_decision(
        result,
        standard_logging_object=standard_logging_object,
    )
    if decision is None:
        return

    blocked = decision == "yes"
    request_model, response_model = _extract_claude_permission_check_models(
        kwargs,
        standard_logging_object,
        metadata,
        result,
    )

    metadata["claude_internal_check"] = True
    metadata["claude_internal_check_type"] = "permission_check"
    metadata["claude_permission_check"] = True
    metadata["claude_permission_check_decision"] = decision
    metadata["claude_permission_check_blocked"] = blocked
    if request_model:
        metadata["claude_permission_check_request_model"] = request_model
    if response_model:
        metadata["claude_permission_check_response_model"] = response_model

    _merge_tags(
        metadata,
        [
            "claude-internal-check",
            "claude-permission-check",
            f"claude-permission-check:{decision}",
            "claude-permission-check:block" if blocked else "claude-permission-check:allow",
        ],
    )

    existing_spans = metadata.get("langfuse_spans") or []
    if not isinstance(existing_spans, list):
        existing_spans = []
    if any(isinstance(span, dict) and span.get("name") == "claude.permission_check" for span in existing_spans):
        return

    span_metadata: Dict[str, Any] = {
        "decision": decision,
        "blocked": blocked,
        "source": "claude_code_block_output",
    }
    for key in (
        "cc_version",
        "cc_entrypoint",
        "client_name",
        "client_version",
        "litellm_environment",
    ):
        value = metadata.get(key)
        if value is not None:
            span_metadata[key] = value
    if request_model:
        span_metadata["request_model"] = request_model
    if response_model:
        span_metadata["response_model"] = response_model

    now = datetime.now(timezone.utc)
    _append_langfuse_span(
        metadata,
        name="claude.permission_check",
        span_metadata=span_metadata,
        input_data={"check_type": "permission_check"},
        output_data={"decision": decision, "blocked": blocked},
        start_time=now,
        end_time=now,
    )


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _metadata_request_tags(metadata: Dict[str, Any]) -> List[str]:
    request_tags = metadata.get("request_tags")
    tags = metadata.get("tags")
    merged: List[str] = []
    for source in (request_tags, tags):
        if not isinstance(source, list):
            continue
        for tag in source:
            if isinstance(tag, str) and tag.strip() and tag not in merged:
                merged.append(tag)
    return merged


def _is_claude_permission_check_metadata(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    if _metadata_bool(metadata.get("claude_permission_check")):
        return True
    for tag in _metadata_request_tags(metadata):
        if tag == "claude-permission-check" or tag.startswith("claude-permission-check:"):
            return True
    return False


def _extract_claude_project_from_metadata_tags(
    metadata: Dict[str, Any],
) -> Optional[str]:
    for tag in _metadata_request_tags(metadata):
        if not tag.startswith("claude-project:"):
            continue
        repository = _normalize_repository_identity(tag.split(":", 1)[1])
        if repository:
            return repository
    return None


def _extract_claude_auto_review_source_model(
    metadata: Dict[str, Any],
    fallback_model: Optional[str] = None,
) -> Optional[str]:
    return _first_non_empty_string(
        metadata.get("source_model"),
        metadata.get("claude_permission_check_response_model"),
        metadata.get("claude_permission_check_request_model"),
        fallback_model,
    )


def _apply_claude_auto_review_metadata(
    metadata: Dict[str, Any],
    *,
    repository: Optional[str] = None,
    tenant_id: Optional[str] = None,
    source_model: Optional[str] = None,
) -> None:
    metadata["trace_name"] = _CLAUDE_AUTO_REVIEW_TRACE_NAME
    metadata["agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["aawm_claude_agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["logical_model"] = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL

    resolved_source_model = _extract_claude_auto_review_source_model(
        metadata,
        source_model,
    )
    if resolved_source_model and resolved_source_model != _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL:
        metadata["source_model"] = resolved_source_model

    normalized_repository = _normalize_repository_identity(repository)
    normalized_tenant = _normalize_repository_identity(tenant_id)
    inherited_identity = normalized_repository or normalized_tenant
    if inherited_identity:
        metadata["repository"] = inherited_identity
        metadata["tenant_id"] = inherited_identity
        metadata["aawm_tenant_id"] = inherited_identity
        metadata["aawm_claude_project"] = inherited_identity
        metadata["trace_user_id"] = inherited_identity

    tags_to_add = [
        "claude-internal-check",
        "claude-permission-check",
        f"claude-agent:{_CLAUDE_AUTO_REVIEW_AGENT_NAME}",
    ]
    if inherited_identity:
        tags_to_add.append(f"claude-project:{inherited_identity}")
    _merge_tags(metadata, tags_to_add)
    existing_request_tags = metadata.get("request_tags") or []
    if not isinstance(existing_request_tags, list):
        existing_request_tags = []
    merged_request_tags = list(existing_request_tags)
    for tag in tags_to_add:
        if tag and tag not in merged_request_tags:
            merged_request_tags.append(tag)
    metadata["request_tags"] = merged_request_tags


def _apply_claude_auto_review_identity_to_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    if not _is_claude_permission_check_metadata(metadata):
        return

    source_model = _extract_claude_auto_review_source_model(
        metadata,
        _clean_non_empty_string(record.get("model")),
    )
    repository = _normalize_repository_identity(record.get("repository"))
    tenant_id = _normalize_repository_identity(record.get("tenant_id"))
    if repository is None:
        repository = _extract_claude_project_from_metadata_tags(metadata)
    if tenant_id is None:
        tenant_id = repository

    _apply_claude_auto_review_metadata(
        metadata,
        repository=repository,
        tenant_id=tenant_id,
        source_model=source_model,
    )
    record["metadata"] = metadata
    record["model"] = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    record["agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    if repository is not None:
        record["repository"] = repository
    resolved_tenant = tenant_id or repository
    if resolved_tenant is not None:
        record["tenant_id"] = resolved_tenant


def _extract_claude_auto_review_identity_from_row(
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    repository = (
        _normalize_repository_identity(row.get("repository"))
        or _extract_claude_project_from_metadata_tags(metadata)
        or _normalize_repository_identity(metadata.get("aawm_claude_project"))
        or _normalize_repository_identity(metadata.get("repository"))
        or _normalize_repository_identity(row.get("tenant_id"))
        or _normalize_repository_identity(metadata.get("tenant_id"))
    )
    if not repository:
        return None

    return {
        "repository": repository,
        "tenant_id": repository,
        "source_row_id": row.get("id"),
        "source": "same_session.session_history",
    }


def _apply_claude_auto_review_parent_identity(
    payload: Dict[str, Any],
    identity: Dict[str, Any],
) -> None:
    repository = _normalize_repository_identity(identity.get("repository"))
    if not repository:
        return

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    payload["repository"] = repository
    payload["tenant_id"] = repository
    _apply_claude_auto_review_metadata(
        metadata,
        repository=repository,
        tenant_id=repository,
        source_model=_extract_claude_auto_review_source_model(
            metadata,
            _clean_non_empty_string(payload.get("model")),
        ),
    )
    metadata["claude_auto_review_parent_identity_source"] = identity.get("source")
    if identity.get("source_row_id") is not None:
        metadata["claude_auto_review_parent_identity_source_row_id"] = identity["source_row_id"]
    payload["metadata"] = metadata


def _build_session_identity_cache(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    identity_by_session: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.get("_skip_session_history"):
            continue
        session_id = _clean_non_empty_string(record.get("session_id"))
        if not session_id:
            continue
        metadata = record.get("metadata")
        if _is_claude_permission_check_metadata(metadata):
            continue
        identity = _extract_claude_auto_review_identity_from_row(record)
        if identity:
            identity_by_session[session_id] = identity
    return identity_by_session


def _build_permission_usage_fields(
    *,
    metadata: Dict[str, Any],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    response_cost_usd: Optional[float],
) -> Dict[str, Any]:
    if not _metadata_bool(metadata.get("claude_permission_check")):
        return {
            "token_permission_input": 0,
            "token_permission_output": 0,
            "permission_usd_cost": 0.0,
        }

    return {
        "token_permission_input": _safe_int(prompt_tokens) or 0,
        "token_permission_output": _safe_int(completion_tokens) or 0,
        "permission_usd_cost": _safe_float(response_cost_usd) or 0.0,
    }


async def _lookup_claude_auto_review_parent_identity(
    conn: Any,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    session_id = _clean_non_empty_string(payload.get("session_id"))
    if not session_id:
        return None
    reference_time = (
        _parse_datetime_value(payload.get("start_time"))
        or _parse_datetime_value(payload.get("observed_at"))
        or _parse_datetime_value(payload.get("end_time"))
    )
    rows = await conn.fetch(
        _AAWM_CLAUDE_AUTO_REVIEW_PARENT_IDENTITY_SQL,  # type: ignore[name-defined]  # noqa: F821
        session_id,
        reference_time,
    )
    for row in rows:
        try:
            candidate = dict(row)
        except Exception:
            candidate = {
                "id": _maybe_get(row, "id"),
                "repository": _maybe_get(row, "repository"),
                "tenant_id": _maybe_get(row, "tenant_id"),
                "agent_name": _maybe_get(row, "agent_name"),
                "metadata": _maybe_get(row, "metadata"),
            }
        identity = _extract_claude_auto_review_identity_from_row(candidate)
        if identity:
            return identity
    return None


async def _apply_claude_auto_review_parent_identity_from_store(
    conn: Any,
    payload: Dict[str, Any],
    identity_by_session: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    metadata = payload.get("metadata")
    if not _is_claude_permission_check_metadata(metadata):
        return

    session_id = _clean_non_empty_string(payload.get("session_id"))
    identity = (identity_by_session or {}).get(session_id or "")
    if identity is None:
        identity = await _lookup_claude_auto_review_parent_identity(conn, payload)
    if identity is not None:
        _apply_claude_auto_review_parent_identity(payload, identity)
        return

    _apply_claude_auto_review_identity_to_record(payload)


_HOST_FUNCTION_NAMES = (
    "_permission_check_probeable_value",
    "_extract_claude_permission_check_decision_from_value",
    "_extract_claude_permission_check_decision",
    "_extract_claude_permission_check_models",
    "_enrich_claude_permission_check_metadata",
    "_metadata_bool",
    "_metadata_request_tags",
    "_is_claude_permission_check_metadata",
    "_extract_claude_project_from_metadata_tags",
    "_extract_claude_auto_review_source_model",
    "_apply_claude_auto_review_metadata",
    "_apply_claude_auto_review_identity_to_record",
    "_extract_claude_auto_review_identity_from_row",
    "_apply_claude_auto_review_parent_identity",
    "_build_session_identity_cache",
    "_build_permission_usage_fields",
    "_lookup_claude_auto_review_parent_identity",
    "_apply_claude_auto_review_parent_identity_from_store",
)


def _rebind_to_host_globals(fn, host_globals):
    rebound = _FunctionType(
        fn.__code__,
        host_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = getattr(fn, "__annotations__", {})
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    rebound.__doc__ = fn.__doc__
    return rebound


def _rebind_installable_callable(value, host_globals):
    if isinstance(value, _FunctionType):
        return _rebind_to_host_globals(value, host_globals)

    wrapped = getattr(value, "__wrapped__", None)
    cache_parameters = getattr(value, "cache_parameters", None)
    if not isinstance(wrapped, _FunctionType) or not callable(cache_parameters):
        return value

    parameters = cache_parameters()
    if not isinstance(parameters, dict) or not {"maxsize", "typed"} <= parameters.keys():
        return value

    rebound_wrapped = _rebind_to_host_globals(wrapped, host_globals)
    rebound = lru_cache(
        maxsize=parameters["maxsize"],
        typed=bool(parameters["typed"]),
    )(rebound_wrapped)
    for attribute, attribute_value in getattr(value, "__dict__", {}).items():
        if attribute != "__wrapped__":
            setattr(rebound, attribute, attribute_value)
    return rebound


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        _installed = _rebind_installable_callable(_original, host_globals)
        mod[_name] = _installed
        host_globals[_name] = _installed
