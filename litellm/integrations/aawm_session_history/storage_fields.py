"""Rate-limit storage mappers and DB payload builders.

Behavior-preserving Wave A4D extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports of identity helpers are intentionally
absent here."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    # Host-global function dependencies (resolved via __globals__ at runtime)
    def _build_rate_limit_transition(*args: Any, **kwargs: Any) -> Any: ...

    def _classify_rate_limit_transition(*args: Any, **kwargs: Any) -> Any: ...

    def _clean_non_empty_string(*args: Any, **kwargs: Any) -> Any: ...

    def _first_non_empty_string(*args: Any, **kwargs: Any) -> Any: ...

    def _first_non_none(*args: Any, **kwargs: Any) -> Any: ...

    def _json_safe_rate_limit_value(*args: Any, **kwargs: Any) -> Any: ...

    def _maybe_get(*args: Any, **kwargs: Any) -> Any: ...

    def _maybe_get_path(*args: Any, **kwargs: Any) -> Any: ...

    def _merged_rate_limit_metadata(*args: Any, **kwargs: Any) -> Any: ...

    def _metadata_bool(*args: Any, **kwargs: Any) -> Any: ...

    def _nonnegative_float_or_none(*args: Any, **kwargs: Any) -> Any: ...

    def _parse_datetime_value(*args: Any, **kwargs: Any) -> Any: ...

    def _parse_provider_timestamp(*args: Any, **kwargs: Any) -> Any: ...

    def _rate_limit_observation_has_meaningful_change(*args: Any, **kwargs: Any) -> Any: ...

    def _safe_float(*args: Any, **kwargs: Any) -> Any: ...

    def _safe_int(*args: Any, **kwargs: Any) -> Any: ...

    # Host-global constant dependencies (resolved via __globals__ at runtime)
    _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_SQL: str = ""
    _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATIONS_BATCH_SQL: str = ""


_AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_FIELDS: Tuple[str, ...] = (
    "observed_at",
    "source",
    "provider",
    "client_family",
    "account_hash",
    "environment",
    "tenant_id",
    "repository",
    "limit_key",
    "limit_id",
    "limit_name",
    "limit_scope",
    "window_minutes",
    "quota_period",
    "provider_resets_at",
    "inferred_window_start_at",
    "used_percentage",
    "remaining_requests",
    "used_requests",
    "total_requests",
    "status",
    "exhausted",
    "exhaustion_kind",
    "reset_hint_seconds",
    "model",
    "quota_limit",
    "quota_used",
    "quota_remaining",
    "billing_period_start_at",
    "billing_period_end_at",
    "model_family",
    "model_tier",
    "parent_limit_key",
    "session_id",
    "trace_id",
    "litellm_call_id",
    "route_family",
    "request_model",
    "response_model",
    "client_name",
    "client_version",
    "client_user_agent",
    "raw_provider_fields",
    "evidence",
    "metadata",
)


def _rate_limit_storage_provider(record: Dict[str, Any]) -> str:
    provider = _clean_non_empty_string(record.get("provider")) or "unknown"
    source = str(record.get("source") or "").lower()
    client_family = str(record.get("client_family") or "").lower()
    if provider == "antigravity" or client_family == "antigravity_code_assist" or source.startswith("antigravity_"):
        return "antigravity"
    if provider in {"opencode", "opencode_zen"} or client_family == "opencode_zen" or source.startswith("opencode_"):
        return "opencode_zen"
    if (
        provider in {"gemini", "google_code_assist"}
        or client_family in {"gemini", "google_code_assist"}
        or source.startswith("google_")
        or source.startswith("gemini_")
    ):
        return "google"
    return provider


def _rate_limit_storage_client(record: Dict[str, Any]) -> Optional[str]:
    return _first_non_empty_string(
        record.get("client_family"),
        record.get("client_name"),
        _maybe_get_path(record.get("metadata"), "client_name"),
    )


def _rate_limit_storage_quota_key(record: Dict[str, Any]) -> str:
    limit_id = _clean_non_empty_string(record.get("limit_id"))
    limit_scope = _clean_non_empty_string(record.get("limit_scope"))
    if limit_id and limit_scope:
        return f"{limit_id}:{limit_scope}"
    return (
        _clean_non_empty_string(record.get("limit_key"))
        or _clean_non_empty_string(record.get("limit_name"))
        or ":".join(
            part
            for part in (
                _clean_non_empty_string(record.get("source")),
                _clean_non_empty_string(record.get("model")),
            )
            if part
        )
        or "unknown_quota"
    )


def _rate_limit_storage_quota_type(record: Dict[str, Any]) -> str:
    explicit_quota_type = _clean_non_empty_string(record.get("quota_type"))
    if explicit_quota_type:
        return explicit_quota_type

    limit_scope = str(record.get("limit_scope") or "").lower()
    raw_provider_fields = record.get("raw_provider_fields")
    token_type = (
        str(raw_provider_fields.get("tokenType") or "").lower() if isinstance(raw_provider_fields, dict) else ""
    )
    source = str(record.get("source") or "").lower()
    provider = _rate_limit_storage_provider(record)

    if "request" in limit_scope or limit_scope == "requests" or token_type == "requests":
        return "requests"
    if "message" in limit_scope or token_type == "messages":
        return "messages"
    if "token" in limit_scope or limit_scope == "tokens" or token_type == "tokens":
        return "tokens"
    if limit_scope == "model_capacity" or "capacity" in source:
        return "capacity"
    if provider == "google":
        return "requests"
    if provider in {"openai", "anthropic"}:
        return "tokens"
    return "unknown"


def _rate_limit_storage_remaining_pct(record: Dict[str, Any]) -> Optional[float]:
    remaining_pct = _safe_float(record.get("remaining_pct"))
    if remaining_pct is not None:
        return max(0.0, min(100.0, remaining_pct))

    remaining_fraction = _safe_float(_maybe_get_path(record.get("raw_provider_fields"), "remainingFraction"))
    if remaining_fraction is not None:
        return max(0.0, min(100.0, remaining_fraction * 100.0))

    used_percentage = _safe_float(record.get("used_percentage"))
    if used_percentage is not None:
        return max(0.0, min(100.0, 100.0 - used_percentage))

    if bool(record.get("exhausted")):
        return 0.0
    return None


def _rate_limit_storage_numeric_detail(
    record: Dict[str, Any],
    key: str,
    *raw_paths: str,
) -> Optional[float]:
    direct_value = _nonnegative_float_or_none(record.get(key))
    if direct_value is not None:
        return direct_value
    raw_provider_fields = record.get("raw_provider_fields")
    if not isinstance(raw_provider_fields, dict):
        return None
    for raw_path in raw_paths:
        value: Any = raw_provider_fields
        for part in raw_path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        normalized = _nonnegative_float_or_none(value.get("val") if isinstance(value, dict) else value)
        if normalized is not None:
            return normalized
    return None


def _rate_limit_storage_quota_limit(record: Dict[str, Any]) -> Optional[float]:
    return _first_non_none(
        _rate_limit_storage_numeric_detail(
            record,
            "quota_limit",
            "monthlyLimit",
            "total",
            "limit",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
        ),
        _nonnegative_float_or_none(record.get("total_requests")),
    )


def _rate_limit_storage_quota_used(record: Dict[str, Any]) -> Optional[float]:
    return _first_non_none(
        _rate_limit_storage_numeric_detail(record, "quota_used", "used"),
        _nonnegative_float_or_none(record.get("used_requests")),
    )


def _rate_limit_storage_quota_remaining(record: Dict[str, Any]) -> Optional[float]:
    return _first_non_none(
        _rate_limit_storage_numeric_detail(
            record,
            "quota_remaining",
            "remaining",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
        ),
        _nonnegative_float_or_none(record.get("remaining_requests")),
    )


def _rate_limit_storage_timestamp_detail(
    record: Dict[str, Any],
    key: str,
    *raw_paths: str,
) -> Optional[datetime]:
    direct_value = _parse_provider_timestamp(record.get(key))
    if direct_value is not None:
        return direct_value
    raw_provider_fields = record.get("raw_provider_fields")
    if not isinstance(raw_provider_fields, dict):
        return None
    for raw_path in raw_paths:
        value: Any = raw_provider_fields
        for part in raw_path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        parsed = _parse_provider_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _rate_limit_storage_billing_period_start_at(
    record: Dict[str, Any],
) -> Optional[datetime]:
    return _rate_limit_storage_timestamp_detail(
        record,
        "billing_period_start_at",
        "billingPeriodStart",
    )


def _rate_limit_storage_billing_period_end_at(
    record: Dict[str, Any],
) -> Optional[datetime]:
    return _first_non_none(
        _rate_limit_storage_timestamp_detail(
            record,
            "billing_period_end_at",
            "billingPeriodEnd",
        ),
        _parse_provider_timestamp(record.get("provider_resets_at"))
        if record.get("quota_period") == "monthly"
        else None,
    )


def _build_rate_limit_observation_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record["observed_at"],
        _rate_limit_storage_client(record),
        record.get("client_version"),
        record.get("account_hash"),
        _rate_limit_storage_provider(record),
        record.get("model"),
        _rate_limit_storage_quota_key(record),
        record.get("quota_period"),
        _rate_limit_storage_quota_type(record),
        record.get("provider_resets_at"),
        _rate_limit_storage_remaining_pct(record),
        _rate_limit_storage_quota_limit(record),
        _rate_limit_storage_quota_used(record),
        _rate_limit_storage_quota_remaining(record),
        _rate_limit_storage_billing_period_start_at(record),
        _rate_limit_storage_billing_period_end_at(record),
        json.dumps(_json_safe_rate_limit_value(record.get("raw_provider_fields") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("evidence") or {})),
        record.get("source"),
        record.get("session_id"),
        record.get("trace_id"),
        record.get("litellm_call_id"),
    )


def _build_rate_limit_transition_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record["transition_key"],
        record["limit_key"],
        record.get("provider"),
        record.get("client_family"),
        record.get("account_hash"),
        record["transition_type"],
        record.get("confidence") or 0.0,
        json.dumps(_json_safe_rate_limit_value(record.get("signals") or [])),
        record.get("source"),
        record.get("old_observed_at"),
        record["new_observed_at"],
        record.get("old_provider_resets_at"),
        record.get("new_provider_resets_at"),
        record.get("old_used_percentage"),
        record.get("new_used_percentage"),
        record.get("old_remaining_requests"),
        record.get("new_remaining_requests"),
        record.get("old_used_requests"),
        record.get("new_used_requests"),
        record.get("old_total_requests"),
        record.get("new_total_requests"),
        record.get("inferred_window_start_at"),
        record.get("detection_window_start_at"),
        record.get("detection_window_end_at"),
        json.dumps(_json_safe_rate_limit_value(record.get("session_usage_summary") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("old_observation") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("new_observation") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("metadata") or {})),
    )


def _build_provider_error_observation_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record["observed_at"],
        record.get("environment"),
        record["provider"],
        record.get("model"),
        record.get("model_group"),
        record.get("route_family"),
        record.get("status_code"),
        record.get("error_type"),
        record.get("error_code"),
        record["error_class"],
        record.get("retry_after_seconds"),
        record.get("expected_reset_at"),
        record.get("session_id"),
        record.get("trace_id"),
        record.get("litellm_call_id"),
        json.dumps(_json_safe_rate_limit_value(record.get("metadata") or {})),
    )


def _extract_alias_routing_audit_events(
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metadata = record.get("metadata")
    event_sources: List[Any] = [record.get("aawm_alias_routing_audit_events")]
    if isinstance(metadata, dict):
        event_sources.extend(
            [
                metadata.get("aawm_alias_routing_audit_events"),
                metadata.get("codex_auto_agent_audit_events"),
                metadata.get("anthropic_auto_agent_audit_events"),
            ]
        )
    events: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for source in event_sources:
        if not isinstance(source, list):
            continue
        for event in source:
            if not isinstance(event, dict):
                continue
            try:
                fingerprint = json.dumps(
                    _json_safe_rate_limit_value(event),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except Exception:
                fingerprint = str(id(event))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            events.append(event)
    return events


def _alias_routing_audit_observed_at(
    record: Dict[str, Any],
    event: Dict[str, Any],
) -> datetime:
    return (
        _parse_datetime_value(event.get("observed_at"))
        or _parse_datetime_value(record.get("start_time"))
        or _parse_datetime_value(record.get("end_time"))
        or datetime.now(timezone.utc)
    )


def _alias_routing_audit_event_key(
    *,
    record: Dict[str, Any],
    event: Dict[str, Any],
    event_index: int,
) -> Optional[str]:
    litellm_call_id = _clean_non_empty_string(event.get("litellm_call_id") or record.get("litellm_call_id"))
    if litellm_call_id is None:
        return None
    key_material = [
        litellm_call_id,
        event.get("alias_family"),
        event.get("alias_model"),
        event.get("event_type"),
        event.get("provider"),
        event.get("model"),
        event.get("attempt_number"),
        event.get("candidate_status"),
        event_index,
    ]
    digest = hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
    return f"{litellm_call_id}:alias-routing:{digest}"


def _infer_alias_routing_family(
    event: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    return (
        _clean_non_empty_string(event.get("alias_family"))
        or ("codex_auto_agent" if _clean_non_empty_string(metadata.get("codex_auto_agent_alias")) else None)
        or ("anthropic_auto_agent" if _clean_non_empty_string(metadata.get("anthropic_auto_agent_alias")) else None)
        or "unknown"
    )


def _build_alias_routing_audit_db_payload(
    record: Dict[str, Any],
    event: Dict[str, Any],
    event_index: int,
) -> Tuple[Any, ...]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    event_metadata = dict(event)
    event_metadata["event_index"] = event_index
    event_metadata.setdefault("session_history_provider", record.get("provider"))
    event_metadata.setdefault("session_history_model", record.get("model"))
    event_metadata.setdefault("session_history_model_group", record.get("model_group"))
    event_metadata.setdefault("session_history_repository", record.get("repository"))
    return (
        _alias_routing_audit_event_key(
            record=record,
            event=event,
            event_index=event_index,
        ),
        _alias_routing_audit_observed_at(record, event),
        _clean_non_empty_string(event.get("session_id")) or _clean_non_empty_string(record.get("session_id")),
        _clean_non_empty_string(event.get("session_key")),
        _clean_non_empty_string(event.get("trace_id")) or _clean_non_empty_string(record.get("trace_id")),
        _clean_non_empty_string(event.get("litellm_call_id")) or _clean_non_empty_string(record.get("litellm_call_id")),
        _clean_non_empty_string(event.get("alias_model"))
        or _clean_non_empty_string(metadata.get("requested_model_alias"))
        or "unknown",
        _infer_alias_routing_family(event, metadata),
        _clean_non_empty_string(event.get("route_family")),
        _clean_non_empty_string(event.get("provider")),
        _clean_non_empty_string(event.get("model")),
        _clean_non_empty_string(event.get("lane_key")),
        _clean_non_empty_string(event.get("cooldown_key")),
        _safe_int(event.get("attempt_number")),
        _clean_non_empty_string(event.get("event_type")) or "unknown",
        _clean_non_empty_string(event.get("selection_reason")),
        _clean_non_empty_string(event.get("candidate_status")),
        _clean_non_empty_string(event.get("failure_class")),
        _safe_int(event.get("error_status_code")),
        _clean_non_empty_string(event.get("cooldown_scope")),
        _safe_float(event.get("cooldown_seconds")),
        _parse_datetime_value(event.get("cooldown_until")),
        _metadata_bool(event.get("selected")),
        _metadata_bool(event.get("skipped")),
        _metadata_bool(event.get("last_resort")),
        _metadata_bool(event.get("in_flight_session")),
        _metadata_bool(event.get("redispatch_required")),
        _metadata_bool(event.get("redispatch_threshold_crossed")),
        json.dumps(_json_safe_rate_limit_value(event_metadata)),
    )


def _rate_limit_previous_observation_row_to_dict(row: Any) -> Dict[str, Any]:
    try:
        row_dict = dict(row)
    except Exception:
        return {key: _maybe_get(row, key) for key in _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_FIELDS}
    row_dict.pop("input_limit_key", None)
    return row_dict


async def _fetch_previous_rate_limit_observation(
    conn: Any,
    observation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    quota_key = _rate_limit_storage_quota_key(observation)
    if not quota_key or not observation.get("observed_at"):
        return None
    row = await conn.fetchrow(
        _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_SQL,
        quota_key,
        _rate_limit_storage_provider(observation),
        _rate_limit_storage_client(observation),
        observation.get("account_hash"),
        observation.get("source"),
        observation["observed_at"],
    )
    if row is None:
        return None
    return _rate_limit_previous_observation_row_to_dict(row)


async def _fetch_previous_rate_limit_observations(
    conn: Any,
    observations: List[Dict[str, Any]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    first_observation_by_limit_key: Dict[str, Dict[str, Any]] = {}
    for observation in observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        if (
            not isinstance(limit_key, str)
            or not limit_key
            or not observation.get("observed_at")
            or limit_key in first_observation_by_limit_key
        ):
            continue
        first_observation_by_limit_key[limit_key] = observation

    if not first_observation_by_limit_key:
        return {}

    limit_keys: List[str] = []
    providers: List[str] = []
    clients: List[Optional[str]] = []
    account_hashes: List[Optional[str]] = []
    sources: List[Optional[str]] = []
    observed_ats: List[Any] = []
    for limit_key, observation in first_observation_by_limit_key.items():
        limit_keys.append(limit_key)
        providers.append(_rate_limit_storage_provider(observation))
        clients.append(_rate_limit_storage_client(observation))
        account_hashes.append(observation.get("account_hash"))
        sources.append(observation.get("source"))
        observed_ats.append(observation["observed_at"])

    previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {limit_key: None for limit_key in limit_keys}
    rows = await conn.fetch(
        _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATIONS_BATCH_SQL,
        limit_keys,
        providers,
        clients,
        account_hashes,
        sources,
        observed_ats,
    )
    for row in rows:
        limit_key = _maybe_get(row, "input_limit_key")
        if isinstance(limit_key, str) and limit_key in previous_by_limit_key:
            previous_by_limit_key[limit_key] = _rate_limit_previous_observation_row_to_dict(row)
    return previous_by_limit_key


async def _derive_rate_limit_transitions(
    conn: Any,
    observations: List[Dict[str, Any]],
    initial_previous_by_limit_key: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = dict(initial_previous_by_limit_key or {})
    ordered_observations = sorted(
        observations,
        key=lambda item: (
            _rate_limit_storage_quota_key(item),
            item.get("observed_at") or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )
    missing_previous_observations: List[Dict[str, Any]] = []
    for observation in ordered_observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        if isinstance(limit_key, str) and limit_key and limit_key not in previous_by_limit_key:
            previous_by_limit_key[limit_key] = None
            missing_previous_observations.append(observation)
    if missing_previous_observations:
        previous_by_limit_key.update(
            await _fetch_previous_rate_limit_observations(
                conn,
                missing_previous_observations,
            )
        )
    for observation in ordered_observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        if not isinstance(limit_key, str) or not limit_key:
            continue
        previous = previous_by_limit_key.get(limit_key)
        if previous is not None:
            classification = _classify_rate_limit_transition(previous, observation)
            if classification is not None:
                transitions.append(_build_rate_limit_transition(previous, observation, classification))
        previous_by_limit_key[limit_key] = observation
    return transitions


async def _filter_meaningful_rate_limit_observations(
    conn: Any,
    observations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[Dict[str, Any]]]]:
    kept_by_index: List[Tuple[int, Dict[str, Any]]] = []
    rolling_previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {}
    initial_previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {}
    indexed_observations = [
        (index, observation)
        for index, observation in enumerate(observations)
        if isinstance(observation.get("limit_key"), str) and observation.get("limit_key")
    ]
    indexed_observations.sort(
        key=lambda item: (
            _rate_limit_storage_quota_key(item[1]),
            item[1].get("observed_at") or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        )
    )

    initial_previous_by_limit_key.update(
        await _fetch_previous_rate_limit_observations(
            conn,
            [observation for _index, observation in indexed_observations],
        )
    )
    rolling_previous_by_limit_key.update(initial_previous_by_limit_key)

    for index, observation in indexed_observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        previous = rolling_previous_by_limit_key.get(limit_key)
        if not _rate_limit_observation_has_meaningful_change(previous, observation):
            continue

        kept_by_index.append((index, observation))
        rolling_previous_by_limit_key[limit_key] = observation

    kept_by_index.sort(key=lambda item: item[0])
    return [observation for _index, observation in kept_by_index], initial_previous_by_limit_key


def _rate_limit_observation_only_requested(kwargs: Dict[str, Any]) -> bool:
    metadata = _merged_rate_limit_metadata(kwargs)
    return bool(metadata.get("aawm_rate_limit_observation_only"))



_HOST_FUNCTION_NAMES: Tuple[str, ...] = (
    "_rate_limit_storage_provider",
    "_rate_limit_storage_client",
    "_rate_limit_storage_quota_key",
    "_rate_limit_storage_quota_type",
    "_rate_limit_storage_remaining_pct",
    "_rate_limit_storage_numeric_detail",
    "_rate_limit_storage_quota_limit",
    "_rate_limit_storage_quota_used",
    "_rate_limit_storage_quota_remaining",
    "_rate_limit_storage_timestamp_detail",
    "_rate_limit_storage_billing_period_start_at",
    "_rate_limit_storage_billing_period_end_at",
    "_build_rate_limit_observation_db_payload",
    "_build_rate_limit_transition_db_payload",
    "_build_provider_error_observation_db_payload",
    "_extract_alias_routing_audit_events",
    "_alias_routing_audit_observed_at",
    "_alias_routing_audit_event_key",
    "_infer_alias_routing_family",
    "_build_alias_routing_audit_db_payload",
    "_rate_limit_previous_observation_row_to_dict",
    "_fetch_previous_rate_limit_observation",
    "_fetch_previous_rate_limit_observations",
    "_derive_rate_limit_transitions",
    "_filter_meaningful_rate_limit_observations",
    "_rate_limit_observation_only_requested",
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
    return value


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
