"""Rate-limit observation foundations (shared idioms, context, finalization).

Behavior-preserving Wave A3A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _parse_provider_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if numeric_value <= 0:
            return None
        if numeric_value > 1_000_000_000_000:
            numeric_value = numeric_value / 1000.0
        try:
            return datetime.fromtimestamp(numeric_value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    parsed = _parse_datetime_value(value)
    if parsed is not None:
        return parsed
    if isinstance(value, str):
        numeric_string_value = _safe_float(value.strip())
        if numeric_string_value is not None:
            return _parse_provider_timestamp(numeric_string_value)
    return None


def _infer_window_start_at(
    provider_resets_at: Optional[datetime],
    window_minutes: Optional[int],
) -> Optional[datetime]:
    if provider_resets_at is None or window_minutes is None or window_minutes <= 0:
        return None
    return provider_resets_at - timedelta(minutes=window_minutes)


def _quota_period_from_window_minutes(window_minutes: Optional[int]) -> Optional[str]:
    if window_minutes is None:
        return None
    if window_minutes == 60:
        return "hourly"
    if window_minutes == 300:
        return "five_hour"
    if window_minutes == 10080:
        return "seven_day"
    if window_minutes == 1440:
        return "daily"
    return f"{window_minutes}_minutes"


def _parse_reset_hint_seconds(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None and parsed >= 0:
            return parsed
        if not isinstance(value, str):
            continue
        match = _RESET_AFTER_SECONDS_RE.search(value)
        if match is None:
            continue
        parsed = _safe_int(match.group("seconds"))
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _resolve_rate_limit_reset_at(
    reset_value: Any,
    observed_at: Any,
    reset_hint_seconds: Optional[int] = None,
) -> Tuple[Optional[datetime], bool]:
    provider_resets_at = _parse_provider_timestamp(reset_value)
    observed_dt = _normalize_datetime(observed_at)
    if (
        provider_resets_at is not None
        and observed_dt is not None
        and provider_resets_at < observed_dt - _AAWM_RATE_LIMIT_STALE_RESET_TOLERANCE
    ):
        if reset_hint_seconds is not None:
            return observed_dt + timedelta(seconds=reset_hint_seconds), False
        return None, True
    if provider_resets_at is None and reset_hint_seconds is not None and observed_dt is not None:
        return observed_dt + timedelta(seconds=reset_hint_seconds), False
    return provider_resets_at, False


def _json_safe_rate_limit_value(
    value: Any,
    *,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> Any:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_JSON_SAFE_MAX_DEPTH:
        return "<max_depth>"
    if isinstance(value, datetime):
        return _format_langfuse_span_timestamp(value)
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _seen:
            return "<recursive>"
        _seen.add(value_id)
        try:
            return {
                str(key): _json_safe_rate_limit_value(
                    nested_value,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for key, nested_value in list(value.items())
                if isinstance(key, (str, int, float, bool))
            }
        finally:
            _seen.discard(value_id)
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return ["<recursive>"]
        _seen.add(value_id)
        try:
            return [
                _json_safe_rate_limit_value(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for item in value[:100]
            ]
        finally:
            _seen.discard(value_id)
    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in _seen:
            return ["<recursive>"]
        _seen.add(value_id)
        try:
            return [
                _json_safe_rate_limit_value(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for item in value[:100]
            ]
        finally:
            _seen.discard(value_id)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")[:500]
        except Exception:
            return "<bytes>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return value[:1000]
        return value
    return str(value)[:500]


def _coerce_rate_limit_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if hasattr(value, "items"):
        try:
            return {str(key): nested_value for key, nested_value in list(value.items())}
        except Exception:
            return None
    if isinstance(value, bytes):
        try:
            return _coerce_rate_limit_payload(value.decode("utf-8", errors="replace"))
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    # Fail closed on unbounded attacker/provider-influenced text before
    # JSON/literal evaluation (ast.literal_eval is not DoS-safe on deep nests).
    if len(stripped) > 8192:
        return None
    parsed = _safe_json_load(stripped, None)
    if parsed is not None:
        return parsed
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            literal_value = ast.literal_eval(stripped)
    except Exception:
        return None
    if isinstance(literal_value, bytes):
        return _coerce_rate_limit_payload(literal_value)
    if isinstance(literal_value, (dict, list)):
        return literal_value
    return None


def _iter_rate_limit_dicts(*roots: Any) -> List[Dict[str, Any]]:
    pending: List[Tuple[Any, int]] = [(root, 0) for root in roots if root is not None]
    seen: set = set()
    dicts: List[Dict[str, Any]] = []
    while pending and len(seen) < 512:
        value, depth = pending.pop(0)
        coerced = _coerce_rate_limit_payload(value)
        if coerced is not None:
            value = coerced
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, dict):
            dicts.append(value)
            if depth >= 6:
                continue
            for nested_value in list(value.values()):
                if isinstance(nested_value, (dict, list, str, bytes)):
                    pending.append((nested_value, depth + 1))
        elif isinstance(value, list):
            if depth >= 6:
                continue
            for item in value[:200]:
                if isinstance(item, (dict, list, str, bytes)):
                    pending.append((item, depth + 1))
    return dicts


def _merged_rate_limit_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        standard_metadata = standard_logging_object.get("metadata")
        if isinstance(standard_metadata, dict):
            metadata.update(dict(standard_metadata))
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        litellm_metadata = litellm_params.get("metadata")
        if isinstance(litellm_metadata, dict):
            metadata.update(dict(litellm_metadata))
    return metadata


def _extract_headers_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for candidate in (
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "headers"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers"),
        _maybe_get_path(kwargs.get("standard_pass_through_logging_payload"), "request_headers"),
        _maybe_get_path(kwargs.get("standard_logging_object"), "request_headers"),
        kwargs.get("headers"),
    ):
        if not isinstance(candidate, dict):
            continue
        for key, value in list(candidate.items()):
            if isinstance(key, str) and value is not None:
                headers[key.lower()] = str(value)
    return headers


def _extract_rate_limit_account_hash(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    headers = _extract_headers_from_kwargs(kwargs)
    user_api_key_dict = kwargs.get("user_api_key_dict") or kwargs.get("user_api_key")
    candidates = [
        metadata.get("user_api_key_hash"),
        metadata.get("api_key_hash"),
        metadata.get("provider_account_hash"),
        metadata.get("provider_account_id"),
        metadata.get("organization_id"),
        metadata.get("org_id"),
        kwargs.get("user_api_key_hash"),
        _maybe_get(user_api_key_dict, "api_key_hash"),
        _maybe_get(user_api_key_dict, "token"),
        _maybe_get(user_api_key_dict, "api_key"),
        headers.get("x-litellm-user-api-key-hash"),
        headers.get("x-api-key-hash"),
        headers.get("x-goog-user-project"),
        headers.get("anthropic-organization-id"),
        headers.get("openai-organization"),
        headers.get("x-grok-user-id"),
        headers.get("x-userid"),
        headers.get("x-teamid"),
        headers.get("x-email"),
        headers.get("x-xai-token-auth"),
        headers.get("authorization"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue
        return _short_hash(candidate_text.encode("utf-8"))
    return None


def _resolve_rate_limit_model(
    kwargs: Dict[str, Any],
    result: Any,
    metadata: Dict[str, Any],
) -> str:
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    try:
        model = _resolve_session_history_model(
            kwargs=kwargs,
            standard_logging_object=standard_logging_object,
            metadata=metadata,
            result=result,
        )
        if model and model != "unknown":
            return model
    except Exception:
        pass
    for candidate in (
        kwargs.get("model"),
        metadata.get("model"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        _maybe_get(result, "model"),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return "unknown"


def _infer_model_family_and_tier(*values: Any) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join(str(value) for value in values if value is not None).lower()

    def _has_token(token: str) -> bool:
        return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text) is not None

    model_tier = None
    if _has_token("sonnet"):
        model_tier = "sonnet"
    elif _has_token("opus"):
        model_tier = "opus"
    elif _has_token("haiku"):
        model_tier = "haiku"
    elif "flash-lite" in text or "flash_lite" in text or _has_token("flash-lite"):
        model_tier = "flash_lite"
    elif _has_token("flash"):
        model_tier = "flash"
    elif _has_token("pro"):
        model_tier = "pro"

    if "claude" in text or model_tier in {"sonnet", "opus", "haiku"}:
        return "claude", model_tier
    # Prefer explicit OpenAI/Codex markers over tier-based gemini inference so
    # names like gpt-5-pro / o1-pro and metadata containing "project"/"prod"
    # do not get misclassified as gemini solely because of a "pro" token.
    if "gpt" in text or "openai" in text or _has_token("o1") or _has_token("o3") or _has_token("o4"):
        return "openai", model_tier
    if "codex" in text:
        return "codex", model_tier
    if "gemini" in text or "gemma" in text or model_tier in {"pro", "flash", "flash_lite"}:
        return "gemini", model_tier
    return None, model_tier


def _infer_rate_limit_client_family(
    provider: Optional[str],
    model: str,
    metadata: Dict[str, Any],
    source: Optional[str],
) -> Optional[str]:
    source_lower = str(source or "").lower()
    route_family = str(metadata.get("passthrough_route_family") or "").lower()
    model_lower = str(model or "").lower()
    client_text = " ".join(
        str(value)
        for value in (
            metadata.get("client_name"),
            metadata.get("client_version"),
            metadata.get("trace_name"),
            metadata.get("cc_version"),
            metadata.get("cc_entrypoint"),
        )
        if value is not None
    ).lower()
    credential_family = str(metadata.get("credential_family") or "").lower()
    retired_context = " ".join(
        (
            str(provider or ""),
            source_lower,
            route_family,
            str(metadata.get("aawm_stream_logging_custom_llm_provider") or ""),
            str(metadata.get("custom_llm_provider") or ""),
            model_lower,
        )
    ).lower()
    if any(
        marker in retired_context
        for marker in (
            "antigravity",
            "google_code_assist",
            "google-code-assist",
            "google_retrieve_user_quota",
        )
    ):
        return None
    if (
        "opencode" in source_lower
        or credential_family == "opencode"
        or metadata.get("opencode_zen") is True
        or "opencode" in route_family
        or model_lower.startswith(("opencode/", "opencode-zen/", "zen/"))
    ):
        return "opencode_zen"
    if (
        "xai_oauth" in source_lower
        or credential_family == "xai_oauth"
        or metadata.get("xai_oauth_managed") is True
        or metadata.get("xai_oauth_public_model") is not None
        or "xai_oauth" in route_family
    ):
        return "xai_oauth"
    if "codex" in source_lower or "codex" in route_family or "codex" in model_lower:
        return "codex"
    if "gemini" in source_lower or "gemini" in route_family or "gemini" in model_lower:
        return "gemini"
    if (
        "grok" in source_lower
        or "grok" in route_family
        or "xai" in route_family
        or "grok" in model_lower
        or "grok-build" in client_text
    ):
        return "grok-build"
    if "claude" in source_lower or "claude" in route_family or "claude" in client_text or "cc_version" in metadata:
        return "claude"
    return provider


def _build_rate_limit_key(
    *,
    provider: Optional[str],
    client_family: Optional[str],
    account_hash: Optional[str],
    limit_id: Optional[str],
    limit_name: Optional[str],
    limit_scope: Optional[str],
    quota_period: Optional[str],
    window_minutes: Optional[int],
    model: Optional[str],
    model_family: Optional[str],
) -> str:
    identity = (
        _clean_non_empty_string(limit_id)
        or _clean_non_empty_string(limit_name)
        or (_clean_non_empty_string(model) if str(limit_scope or "").startswith("model") else None)
        or _clean_non_empty_string(model_family)
        or "default"
    )
    parts = (
        provider or "unknown_provider",
        client_family or "unknown_client",
        account_hash or "unknown_account",
        identity,
        limit_scope or quota_period or "unknown_scope",
        str(window_minutes or "unknown_window"),
    )
    normalized_parts = [
        re.sub(r"[^a-z0-9_.-]+", "_", str(part).strip().lower()).strip("_") or "unknown" for part in parts
    ]
    return ":".join(normalized_parts)


def _build_rate_limit_context(
    kwargs: Dict[str, Any],
    result: Any,
    end_time: Any,
    source: Optional[str],
) -> Dict[str, Any]:
    """Build (and request-cache) shared rate-limit observation context.

    Repository/tenant extraction can deep-walk large request payloads. Cache the
    expensive identity fields once per kwargs object so the nine extractors that
    call this helper do not re-scan the miss path.
    """
    cache: Optional[Dict[str, Any]] = None
    if isinstance(kwargs, dict):
        raw_cache = kwargs.get(_AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY)
        if isinstance(raw_cache, dict):
            cache = raw_cache
        else:
            cache = {}
            kwargs[_AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY] = cache

    cache_key = (
        id(result),
        source,
        id(end_time) if not isinstance(end_time, (str, int, float)) else end_time,
    )
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, dict):
            # Return a shallow copy so extractors can mutate client_family safely.
            return dict(cached)

    metadata = _merged_rate_limit_metadata(kwargs)
    model = _resolve_rate_limit_model(kwargs, result, metadata)
    provider = _normalize_session_history_provider(
        kwargs.get("custom_llm_provider"),
        model,
        metadata,
    )
    client_family = _infer_rate_limit_client_family(provider, model, metadata, source)
    model_family, model_tier = _infer_model_family_and_tier(
        model,
        metadata.get("model"),
        metadata.get("anthropic_adapter_model"),
        metadata.get("codex_adapter_model"),
    )
    runtime_identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=True,
    )

    identity_cache_key = "_identity"
    identity: Optional[Dict[str, Any]] = None
    if cache is not None and isinstance(cache.get(identity_cache_key), dict):
        identity = cache[identity_cache_key]
    if identity is None:
        tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        repository = _extract_repository_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        identity = {
            "tenant_id": tenant_id,
            "repository": repository,
            "session_id": _extract_session_id(kwargs),
            "trace_id": _extract_trace_id(kwargs),
            "account_hash": _extract_rate_limit_account_hash(kwargs, metadata),
            "environment": runtime_identity.get("litellm_environment"),
            "client_name": runtime_identity.get("client_name"),
            "client_version": runtime_identity.get("client_version"),
            "client_user_agent": runtime_identity.get("client_user_agent"),
        }
        if cache is not None:
            cache[identity_cache_key] = identity

    context = {
        "observed_at": _normalize_datetime(end_time) or datetime.now(timezone.utc),
        "provider": provider,
        "client_family": client_family,
        "account_hash": identity["account_hash"],
        "environment": identity["environment"],
        "tenant_id": identity["tenant_id"],
        "repository": identity["repository"],
        "session_id": identity["session_id"],
        "trace_id": identity["trace_id"],
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "route_family": metadata.get("passthrough_route_family"),
        "request_model": _first_non_empty_string(
            _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
            _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        ),
        "response_model": _first_non_empty_string(
            _maybe_get(result, "model"),
            _maybe_get_path(kwargs.get("standard_logging_object"), "response", "model"),
        ),
        "model": model,
        "model_family": model_family,
        "model_tier": model_tier,
        "client_name": identity["client_name"],
        "client_version": identity["client_version"],
        "client_user_agent": identity["client_user_agent"],
        "metadata": metadata,
    }
    if cache is not None:
        cache[cache_key] = context
    return dict(context)


def _finalize_rate_limit_observation(
    observation: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    finalized = dict(context)
    finalized.update(observation)
    finalized["observed_at"] = (
        _normalize_datetime(finalized.get("observed_at")) or context.get("observed_at") or datetime.now(timezone.utc)
    )
    finalized["provider_resets_at"] = _parse_provider_timestamp(finalized.get("provider_resets_at"))
    window_minutes = _safe_int(finalized.get("window_minutes"))
    finalized["window_minutes"] = window_minutes
    if finalized.get("quota_period") is None:
        finalized["quota_period"] = _quota_period_from_window_minutes(window_minutes)
    finalized["inferred_window_start_at"] = _infer_window_start_at(
        finalized.get("provider_resets_at"),
        window_minutes,
    )
    finalized["used_percentage"] = _safe_float(finalized.get("used_percentage"))
    finalized["remaining_requests"] = _safe_int(finalized.get("remaining_requests"))
    finalized["used_requests"] = _safe_int(finalized.get("used_requests"))
    finalized["total_requests"] = _safe_int(finalized.get("total_requests"))
    finalized["reset_hint_seconds"] = _safe_int(finalized.get("reset_hint_seconds"))
    model_family, model_tier = _infer_model_family_and_tier(
        finalized.get("model"),
        finalized.get("limit_name"),
        finalized.get("raw_provider_fields"),
    )
    finalized["model_family"] = finalized.get("model_family") or model_family
    finalized["model_tier"] = finalized.get("model_tier") or model_tier
    finalized_metadata = finalized.get("metadata")
    if not isinstance(finalized_metadata, dict):
        finalized_metadata = {}
    finalized["client_family"] = finalized.get("client_family") or _infer_rate_limit_client_family(
        finalized.get("provider"),
        str(finalized.get("model") or ""),
        finalized_metadata,
        finalized.get("source"),
    )
    finalized["limit_key"] = _build_rate_limit_key(
        provider=finalized.get("provider"),
        client_family=finalized.get("client_family"),
        account_hash=finalized.get("account_hash"),
        limit_id=_clean_non_empty_string(finalized.get("limit_id")),
        limit_name=_clean_non_empty_string(finalized.get("limit_name")),
        limit_scope=_clean_non_empty_string(finalized.get("limit_scope")),
        quota_period=_clean_non_empty_string(finalized.get("quota_period")),
        window_minutes=window_minutes,
        model=_clean_non_empty_string(finalized.get("model")),
        model_family=_clean_non_empty_string(finalized.get("model_family")),
    )
    raw_provider_fields = finalized.get("raw_provider_fields")
    if not isinstance(raw_provider_fields, dict):
        raw_provider_fields = {}
    finalized["raw_provider_fields"] = raw_provider_fields
    evidence = finalized.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    finalized["evidence"] = evidence
    metadata = finalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    finalized["metadata"] = {
        key: _json_safe_rate_limit_value(metadata.get(key))
        for key in _AAWM_RATE_LIMIT_METADATA_KEYS
        if metadata.get(key) is not None
    }
    finalized["exhausted"] = bool(finalized.get("exhausted"))
    if finalized.get("status") is None:
        finalized["status"] = "exhausted" if finalized["exhausted"] else "observed"
    return finalized


def _dedupe_rate_limit_observations(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for observation in observations:
        key = (
            observation.get("source"),
            observation.get("limit_key"),
            observation.get("provider_resets_at"),
            observation.get("used_percentage"),
            observation.get("remaining_requests"),
            observation.get("used_requests"),
            observation.get("total_requests"),
            observation.get("status"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(observation)
    return deduped


def _rate_limit_snapshot_signature(
    observation: Dict[str, Any],
    *,
    include_reset: bool = True,
) -> Tuple[Any, ...]:
    provider_resets_at = _parse_provider_timestamp(observation.get("provider_resets_at"))
    body = (
        _safe_float(observation.get("used_percentage")),
        _safe_int(observation.get("remaining_requests")),
        _safe_int(observation.get("used_requests")),
        _safe_int(observation.get("total_requests")),
        _rate_limit_storage_quota_limit(observation),
        _rate_limit_storage_quota_used(observation),
        _rate_limit_storage_quota_remaining(observation),
        _rate_limit_storage_billing_period_start_at(observation),
        _rate_limit_storage_billing_period_end_at(observation),
        _clean_non_empty_string(observation.get("status")),
        bool(observation.get("exhausted")),
        _clean_non_empty_string(observation.get("exhaustion_kind")),
        None if provider_resets_at is not None else _safe_int(observation.get("reset_hint_seconds")),
    )
    if include_reset:
        return (provider_resets_at, *body)
    return body


def _rate_limit_observation_has_meaningful_change(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> bool:
    if previous is None:
        return True

    previous_reset = _parse_provider_timestamp(previous.get("provider_resets_at"))
    current_reset = _parse_provider_timestamp(current.get("provider_resets_at"))
    previous_without_reset = _rate_limit_snapshot_signature(previous, include_reset=False)
    current_without_reset = _rate_limit_snapshot_signature(current, include_reset=False)
    if previous_without_reset != current_without_reset:
        return True
    if previous_reset is None or current_reset is None:
        return previous_reset != current_reset
    return (
        abs((current_reset - previous_reset).total_seconds()) >= _AAWM_RATE_LIMIT_MEANINGFUL_RESET_SHIFT.total_seconds()
    )


def _rate_limit_candidate_roots(kwargs: Dict[str, Any], result: Any) -> List[Any]:
    metadata = _merged_rate_limit_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    litellm_params = kwargs.get("litellm_params") or {}
    roots: List[Any] = [
        result,
        metadata,
        standard_logging_object.get("metadata") if isinstance(standard_logging_object, dict) else None,
        standard_logging_object.get("response") if isinstance(standard_logging_object, dict) else None,
        standard_logging_object.get("output") if isinstance(standard_logging_object, dict) else None,
        kwargs.get("passthrough_logging_payload"),
        kwargs.get("standard_pass_through_logging_payload"),
        litellm_params.get("metadata") if isinstance(litellm_params, dict) else None,
    ]
    for candidate in (
        result,
        standard_logging_object.get("response") if isinstance(standard_logging_object, dict) else None,
        standard_logging_object.get("output") if isinstance(standard_logging_object, dict) else None,
    ):
        for attr_name in (
            "_hidden_params",
            "hidden_params",
            "additional_headers",
            "_response_headers",
            "response_headers",
            "headers",
            "upstream_headers",
        ):
            attr_value = _maybe_get(candidate, attr_name)
            if attr_value is not None:
                roots.append(attr_value)
                additional_headers = _maybe_get(attr_value, "additional_headers")
                if additional_headers is not None:
                    roots.append(additional_headers)
    for key in (
        "rate_limits",
        "codex_rate_limits",
        "codex_token_count",
        "codex_response_headers",
        "anthropic_response_headers",
        "anthropic_rate_limit_headers",
        "xai_oauth_response_headers",
        "gemini_model_status",
        "google_model_status",
    ):
        if key in metadata:
            roots.append(metadata.get(key))
    return [root for root in roots if root is not None]


_HOST_FUNCTION_NAMES = (
    "_parse_provider_timestamp",
    "_infer_window_start_at",
    "_quota_period_from_window_minutes",
    "_parse_reset_hint_seconds",
    "_resolve_rate_limit_reset_at",
    "_json_safe_rate_limit_value",
    "_coerce_rate_limit_payload",
    "_iter_rate_limit_dicts",
    "_merged_rate_limit_metadata",
    "_extract_headers_from_kwargs",
    "_extract_rate_limit_account_hash",
    "_resolve_rate_limit_model",
    "_infer_model_family_and_tier",
    "_infer_rate_limit_client_family",
    "_build_rate_limit_key",
    "_build_rate_limit_context",
    "_finalize_rate_limit_observation",
    "_dedupe_rate_limit_observations",
    "_rate_limit_snapshot_signature",
    "_rate_limit_observation_has_meaningful_change",
    "_rate_limit_candidate_roots",
)


from types import FunctionType as _FunctionType


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


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        if isinstance(_original, _FunctionType):
            _installed = _rebind_to_host_globals(_original, host_globals)
            mod[_name] = _installed
            host_globals[_name] = _installed
        else:
            host_globals[_name] = _original
