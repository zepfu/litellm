"""Provider-error observation build / classify / fingerprint.

Behavior-preserving Wave A3A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _extract_provider_error_dicts(value: Any) -> List[Dict[str, Any]]:
    dicts: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        dicts.append(value)
    dicts.extend(_extract_error_payload_dicts(value))
    for source in (
        value,
        str(value) if value is not None else None,
        getattr(value, "detail", None),
        getattr(value, "message", None),
        getattr(value, "body", None),
    ):
        dicts.extend(_extract_embedded_json_payload_dicts(source))

    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in dicts:
        try:
            key = json.dumps(
                _json_safe_rate_limit_value(candidate),
                sort_keys=True,
                default=str,
            )
        except Exception:
            key = str(id(candidate))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _extract_embedded_json_payload_dicts(value: Any) -> List[Dict[str, Any]]:
    text = _clean_non_empty_string(value)
    if text is None:
        return []

    decoder = json.JSONDecoder()
    dicts: List[Dict[str, Any]] = []
    attempts = 0
    for match in re.finditer(r"\{", text[:_AAWM_EMBEDDED_JSON_SCAN_CHARS]):
        if len(dicts) >= _AAWM_EMBEDDED_JSON_MAX_SUCCESS:
            break
        if attempts >= _AAWM_EMBEDDED_JSON_MAX_ATTEMPTS:
            break
        attempts += 1
        try:
            parsed, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            dicts.extend(_iter_rate_limit_dicts(parsed))
    return dicts


def _extract_provider_error_headers(value: Any) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    candidates = [
        value,
        getattr(value, "response", None),
        getattr(value, "headers", None),
        getattr(value, "response_headers", None),
        getattr(value, "upstream_headers", None),
    ]
    response = getattr(value, "response", None)
    if response is not None:
        candidates.extend(
            [
                getattr(response, "headers", None),
                getattr(response, "response_headers", None),
            ]
        )

    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, dict) and hasattr(candidate, "items"):
            try:
                candidate = dict(candidate.items())
            except Exception:
                continue
        if not isinstance(candidate, dict):
            continue
        for key, nested_value in list(candidate.items()):
            key_text = _clean_non_empty_string(key)
            value_text = _clean_non_empty_string(nested_value)
            if key_text and value_text:
                headers[key_text.lower()] = value_text
    return headers


def _extract_provider_error_status_code(result: Any, dicts: List[Dict[str, Any]]) -> Optional[int]:
    for candidate in (
        getattr(result, "status_code", None),
        getattr(getattr(result, "response", None), "status_code", None),
        getattr(result, "code", None),
    ):
        status_code = _safe_int(candidate)
        if status_code is not None:
            return status_code

    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        for key in ("status_code", "statusCode", "http_status", "code"):
            status_code = _safe_int(error.get(key)) if isinstance(error, dict) else None
            if status_code is not None:
                return status_code
    return None


def _extract_provider_error_text(result: Any, dicts: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for candidate in (
        getattr(result, "message", None),
        getattr(result, "detail", None),
        str(result) if result is not None else None,
    ):
        cleaned = _clean_non_empty_string(candidate)
        if cleaned:
            parts.append(cleaned)

    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        for key in ("message", "detail", "status", "type", "code", "reason"):
            cleaned = _clean_non_empty_string(error.get(key))
            if cleaned:
                parts.append(cleaned)
        raw_details = error.get("details")
        details: List[Any] = raw_details if isinstance(raw_details, list) else []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            for key in ("reason", "domain"):
                cleaned = _clean_non_empty_string(detail.get(key))
                if cleaned:
                    parts.append(cleaned)
            metadata = detail.get("metadata")
            if isinstance(metadata, dict):
                for key in ("reason", "model"):
                    cleaned = _clean_non_empty_string(metadata.get(key))
                    if cleaned:
                        parts.append(cleaned)
    return " ".join(parts)


def _extract_provider_error_code_and_type(
    result: Any,
    dicts: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    error_code = _first_non_empty_string(
        getattr(result, "code", None),
        getattr(result, "error_code", None),
    )
    error_type = _first_non_empty_string(
        getattr(result, "type", None),
        getattr(result, "error_type", None),
        type(result).__name__ if result is not None else None,
    )
    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        error_code = error_code or _first_non_empty_string(
            error.get("status"),
            error.get("reason"),
            error.get("code"),
        )
        error_type = error_type or _first_non_empty_string(
            error.get("type"),
            error.get("error_type"),
            error.get("status"),
        )
    return error_code, error_type


def _extract_provider_error_retry_after_seconds(
    *,
    kwargs: Dict[str, Any],
    result: Any,
    dicts: List[Dict[str, Any]],
    error_text: str,
) -> Optional[float]:
    headers = _extract_headers_from_kwargs(kwargs)
    headers.update(_extract_provider_error_headers(result))
    retry_after = _first_non_empty_string(
        headers.get("retry-after"),
        headers.get("x-ratelimit-reset-after"),
        headers.get("x-codex-primary-reset-after-seconds"),
        headers.get("x-codex-secondary-reset-after-seconds"),
    )
    retry_after_seconds = _safe_float(retry_after)
    if retry_after_seconds is not None and retry_after_seconds >= 0:
        return retry_after_seconds

    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        for key in ("retry_after_seconds", "retryAfterSeconds", "resetAfterSeconds"):
            parsed = _safe_float(error.get(key))
            if parsed is not None and parsed >= 0:
                return parsed
    reset_hint = _parse_reset_hint_seconds(error_text)
    return float(reset_hint) if reset_hint is not None else None


def _extract_litellm_provider_error_model_group(error_text: str) -> Optional[str]:
    match = _LITELLM_PROVIDER_ERROR_MODEL_GROUP_RE.search(error_text)
    if not match:
        return None
    return _clean_non_empty_string(match.group("model_group"))


def _clean_litellm_provider_error_fallbacks(value: Any) -> Optional[str]:
    fallbacks = _clean_non_empty_string(value)
    if fallbacks is None:
        return None
    for marker in (
        " LiteLLM Retried:",
        " litellm.",
        " RateLimitError:",
        " Traceback ",
        " During handling ",
    ):
        marker_index = fallbacks.find(marker)
        if marker_index > 0:
            fallbacks = fallbacks[:marker_index].strip()
    if fallbacks.lower().startswith("none"):
        return "None"
    return fallbacks[:500]


def _extract_litellm_provider_error_retry_context(error_text: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    retry_match = _LITELLM_PROVIDER_ERROR_RETRIES_RE.search(error_text)
    if retry_match:
        retry_count = _safe_int(retry_match.group("retry_count"))
        max_retries = _safe_int(retry_match.group("max_retries"))
        if retry_count is not None:
            metadata["litellm_retry_count"] = retry_count
        if max_retries is not None:
            metadata["litellm_max_retries"] = max_retries
        if retry_count is not None and max_retries is not None:
            metadata["litellm_retries_exhausted"] = retry_count >= max_retries

    fallbacks_match = _LITELLM_PROVIDER_ERROR_FALLBACKS_RE.search(error_text)
    if fallbacks_match:
        fallbacks = _clean_litellm_provider_error_fallbacks(fallbacks_match.group("fallbacks"))
        if fallbacks is not None:
            metadata["available_model_group_fallbacks"] = fallbacks
            metadata["no_model_group_fallbacks"] = fallbacks.lower() in {
                "none",
                "null",
                "[]",
            }
    return metadata


def _extract_provider_error_payload_metadata_value(
    dicts: List[Dict[str, Any]],
    *keys: str,
) -> Any:
    for candidate in dicts:
        pools: List[Dict[str, Any]] = []
        if isinstance(candidate, dict):
            pools.append(candidate)
            error = candidate.get("error")
            if isinstance(error, dict):
                pools.append(error)
                error_metadata = error.get("metadata")
                if isinstance(error_metadata, dict):
                    pools.append(error_metadata)
            metadata = candidate.get("metadata")
            if isinstance(metadata, dict):
                pools.append(metadata)
        for pool in pools:
            for key in keys:
                if key in pool and pool[key] not in (None, ""):
                    return pool[key]
    return None


def _resolve_provider_error_model_group(
    *,
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    error_text: str,
    model: str,
) -> Optional[str]:
    return _first_non_empty_string(
        _get_session_history_model_group(metadata, standard_logging_object),
        _maybe_get_path(
            kwargs.get("litellm_params"),
            "proxy_server_request",
            "body",
            "model",
        ),
        _maybe_get_path(
            kwargs.get("passthrough_logging_payload"),
            "request_body",
            "model",
        ),
        _maybe_get_path(
            kwargs.get("standard_pass_through_logging_payload"),
            "request_body",
            "model",
        ),
        _maybe_get_path(standard_logging_object, "request_body", "model"),
        _extract_litellm_provider_error_model_group(error_text),
        kwargs.get("model"),
        metadata.get("model"),
        model if model != "unknown" else None,
    )


def _redact_upstream_error_raw(value: Any) -> Optional[str]:
    """Redact auth-header-shaped substrings from upstream error raw text."""
    text = _clean_non_empty_string(value)
    if text is None:
        return None

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('label')}{match.group('sep')}[REDACTED]"

    return _UPSTREAM_ERROR_SECRET_RE.sub(_replace, text)


def _build_provider_error_fingerprint(
    *,
    provider: str,
    model: Optional[str],
    model_group: Optional[str],
    status_code: Optional[int],
    error_code: Optional[str],
    error_type: Optional[str],
    error_class: str,
    observation_metadata: Dict[str, Any],
) -> str:
    # Exclude volatile upstream_error_raw so fingerprints can dedupe the same
    # error class across request-specific raw bodies.
    fingerprint_source = {
        "provider": provider,
        "model": model,
        "model_group": model_group,
        "status_code": status_code,
        "error_code": error_code,
        "error_type": error_type,
        "error_class": error_class,
        "upstream_provider_name": observation_metadata.get("upstream_provider_name"),
    }
    return hashlib.sha256(
        json.dumps(
            _json_safe_rate_limit_value(fingerprint_source),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _enrich_provider_error_observation_metadata(
    *,
    observation_metadata: Dict[str, Any],
    dicts: List[Dict[str, Any]],
    error_text: str,
) -> None:
    upstream_provider_name = _clean_non_empty_string(
        _extract_provider_error_payload_metadata_value(dicts, "provider_name")
    )
    if upstream_provider_name is not None:
        observation_metadata["upstream_provider_name"] = upstream_provider_name

    upstream_is_byok = _extract_provider_error_payload_metadata_value(dicts, "is_byok")
    if upstream_is_byok is not None:
        observation_metadata["upstream_is_byok"] = _metadata_bool(upstream_is_byok)

    upstream_error_raw = _redact_upstream_error_raw(_extract_provider_error_payload_metadata_value(dicts, "raw"))
    if upstream_error_raw is not None:
        observation_metadata["upstream_error_raw"] = upstream_error_raw[:1000]

    observation_metadata.update(_extract_litellm_provider_error_retry_context(error_text))


def _classify_provider_error(
    *,
    status_code: Optional[int],
    error_code: Optional[str],
    error_type: Optional[str],
    error_text: str,
) -> str:
    normalized = " ".join(
        part
        for part in (
            str(status_code or ""),
            error_code or "",
            error_type or "",
            error_text,
        )
        if part
    ).lower()
    if "usage_limit_reached" in normalized:
        return "usage_limit_reached"
    if (
        "model_capacity_exhausted" in normalized
        or "capacity_exhausted" in normalized
        or "model is overloaded" in normalized
        or "overloaded" in normalized
    ):
        return "capacity_exhausted"
    if (
        status_code == 429
        or "resource_exhausted" in normalized
        or "rate_limit" in normalized
        or "rate limit" in normalized
        or "too many requests" in normalized
    ):
        return "rate_limited"
    if status_code in {401, 403} or any(
        marker in normalized
        for marker in (
            "x-api-key",
            "api key",
            "authentication",
            "unauthorized",
            "permission denied",
            "forbidden",
        )
    ):
        return "auth_failed"
    if status_code is not None and status_code >= 500:
        return "provider_5xx"
    if any(marker in normalized for marker in ("timeout", "timed out", "deadline")):
        return "provider_timeout"
    if any(
        marker in normalized
        for marker in (
            "connection error",
            "connection refused",
            "connection reset",
            "dns",
            "tls",
            "ssl",
            "network",
        )
    ):
        return "network_error"
    return "adapter_error"


def _extract_provider_error_fields(
    kwargs: Dict[str, Any],
    result: Any,
) -> Dict[str, Any]:
    """Pure extraction of raw provider-error signals from the upstream result.

    Wave A3A U16 split: isolates the deterministic field extraction (payload
    dicts, status code, error text/code/type, retry-after) from the
    observation assembly performed by :func:`_build_provider_error_observation`.
    Behavior-preserving: these are the exact extraction calls the assembly
    previously inlined, in the same order.
    """
    dicts = _extract_provider_error_dicts(result)
    status_code = _extract_provider_error_status_code(result, dicts)
    error_text = _extract_provider_error_text(result, dicts)
    error_code, error_type = _extract_provider_error_code_and_type(result, dicts)
    retry_after_seconds = _extract_provider_error_retry_after_seconds(
        kwargs=kwargs,
        result=result,
        dicts=dicts,
        error_text=error_text,
    )
    return {
        "dicts": dicts,
        "status_code": status_code,
        "error_text": error_text,
        "error_code": error_code,
        "error_type": error_type,
        "retry_after_seconds": retry_after_seconds,
    }


def _build_provider_error_observation(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> Optional[Dict[str, Any]]:
    observed_at = _parse_datetime_value(end_time) or _parse_datetime_value(start_time) or datetime.now(timezone.utc)
    metadata = _merged_rate_limit_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    if not isinstance(standard_logging_object, dict):
        standard_logging_object = {}
    model = _resolve_rate_limit_model(kwargs, result, metadata)
    source_model = model if model != "unknown" else None
    if _is_claude_permission_check_metadata(metadata):
        repository = _extract_repository_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        _apply_claude_auto_review_metadata(
            metadata,
            repository=repository,
            tenant_id=tenant_id,
            source_model=source_model,
        )
        model = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    provider = (
        _normalize_session_history_provider(
            kwargs.get("custom_llm_provider"),
            model,
            metadata,
        )
        or "unknown"
    )
    fields = _extract_provider_error_fields(kwargs, result)
    dicts = fields["dicts"]
    status_code = fields["status_code"]
    error_text = fields["error_text"]
    error_code = fields["error_code"]
    error_type = fields["error_type"]
    retry_after_seconds = fields["retry_after_seconds"]
    error_class = _classify_provider_error(
        status_code=status_code,
        error_code=error_code,
        error_type=error_type,
        error_text=error_text,
    )
    expected_reset_at = (
        observed_at + timedelta(seconds=retry_after_seconds) if retry_after_seconds is not None else None
    )
    runtime_identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=True,
    )
    observation_metadata = {
        "client_name": runtime_identity.get("client_name"),
        "client_version": runtime_identity.get("client_version"),
        "client_user_agent": runtime_identity.get("client_user_agent"),
        "normalized_error_text": error_text[:500] if error_text else None,
        "observed_signal": "normal_traffic_failure",
    }
    structured_output_state = _detect_structured_output_request(
        _extract_provider_cache_request_body(kwargs),
        metadata,
    )
    if structured_output_state.get("structured_output_attempted"):
        structured_failure_reason = _first_non_empty_string(
            structured_output_state.get("structured_output_failure_reason"),
            _classify_structured_output_failure(result),
        )
        observation_metadata["structured_output_attempted"] = True
        observation_metadata["structured_output_failed"] = bool(
            structured_output_state.get("structured_output_failed") or structured_failure_reason
        )
        for key in (
            "structured_output_mode",
            "structured_output_schema_hash",
        ):
            value = _clean_non_empty_string(structured_output_state.get(key))
            if value is not None:
                observation_metadata[key] = value
        if structured_failure_reason is not None:
            observation_metadata["structured_output_failure_reason"] = structured_failure_reason
    _enrich_provider_error_observation_metadata(
        observation_metadata=observation_metadata,
        dicts=dicts,
        error_text=error_text,
    )
    if _is_claude_permission_check_metadata(metadata):
        for key in (
            "source_model",
            "logical_model",
            "trace_name",
            "trace_user_id",
            "repository",
            "tenant_id",
            "request_tags",
            "tags",
            "claude_permission_check",
            "claude_permission_check_decision",
            "claude_permission_check_blocked",
            "claude_permission_check_request_model",
            "claude_permission_check_response_model",
        ):
            value = metadata.get(key)
            if value is not None:
                observation_metadata[key] = value
    for key in (
        "auth_mode",
        "credential_family",
        "xai_oauth_managed",
        "xai_oauth_public_model",
        "xai_oauth_upstream_model",
        "xai_quota_family",
        "shared_quota_family",
        "grok_subscription_quota_shared",
        "passthrough_route_family",
        "grok_side_channel",
        "grok_side_channel_endpoint_type",
        "grok_side_channel_endpoint_path_template",
        "grok_side_channel_request_content_type",
        "grok_side_channel_request_body_byte_length",
        "grok_side_channel_request_body_digest_source",
        "grok_side_channel_request_json_container_type",
        "grok_side_channel_request_array_length",
    ):
        value = metadata.get(key)
        if value is not None:
            observation_metadata[key] = value
    model_group = _resolve_provider_error_model_group(
        kwargs=kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
        error_text=error_text,
        model=model,
    )
    observation_metadata["provider_error_fingerprint"] = _build_provider_error_fingerprint(
        provider=provider,
        model=model if model != "unknown" else None,
        model_group=model_group,
        status_code=status_code,
        error_code=error_code,
        error_type=error_type,
        error_class=error_class,
        observation_metadata=observation_metadata,
    )
    return {
        "observed_at": observed_at,
        "environment": runtime_identity.get("litellm_environment"),
        "provider": provider,
        "model": model if model != "unknown" else None,
        "model_group": model_group,
        "route_family": _clean_non_empty_string(
            metadata.get("passthrough_route_family")
            or metadata.get("codex_auto_agent_selected_route_family")
            or metadata.get("aawm_local_route_family")
        ),
        "status_code": status_code,
        "error_type": error_type,
        "error_code": error_code,
        "error_class": error_class,
        "retry_after_seconds": retry_after_seconds,
        "expected_reset_at": expected_reset_at,
        "session_id": _extract_session_id(kwargs),
        "trace_id": _first_non_empty_string(
            metadata.get("trace_id"),
            metadata.get("langfuse_trace_id"),
            standard_logging_object.get("trace_id"),
            kwargs.get("trace_id"),
        ),
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "metadata": observation_metadata,
    }


_HOST_FUNCTION_NAMES = (
    "_extract_provider_error_dicts",
    "_extract_embedded_json_payload_dicts",
    "_extract_provider_error_headers",
    "_extract_provider_error_status_code",
    "_extract_provider_error_text",
    "_extract_provider_error_code_and_type",
    "_extract_provider_error_retry_after_seconds",
    "_extract_litellm_provider_error_model_group",
    "_clean_litellm_provider_error_fallbacks",
    "_extract_litellm_provider_error_retry_context",
    "_extract_provider_error_payload_metadata_value",
    "_resolve_provider_error_model_group",
    "_redact_upstream_error_raw",
    "_build_provider_error_fingerprint",
    "_enrich_provider_error_observation_metadata",
    "_classify_provider_error",
    "_extract_provider_error_fields",
    "_build_provider_error_observation",
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
