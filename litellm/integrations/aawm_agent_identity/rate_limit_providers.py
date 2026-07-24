"""Per-provider rate-limit observation extractors (normalized to ObservationExtractor).

Behavior-preserving Wave A3A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations


def _openrouter_free_daily_request_limit() -> int:
    configured_limit = _safe_int(get_secret_str("AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT"))
    if configured_limit is not None and configured_limit > 0:
        return configured_limit
    return _AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT


def _openrouter_free_shared_account_hash() -> str:
    return _short_hash(b"openrouter_free_daily_shared_pool")


def _is_openrouter_free_model(model: Any) -> bool:
    return str(model or "").strip().lower().endswith(":free")


def _openrouter_free_daily_window(observed_at: Any) -> Tuple[datetime, datetime]:
    observed_dt = _normalize_datetime(observed_at) or datetime.now(timezone.utc)
    day_start = observed_dt.astimezone(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return day_start, day_start + timedelta(days=1)


def _openrouter_free_daily_observation_context_from_record(
    record: Dict[str, Any],
    observed_at: datetime,
) -> Dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "observed_at": observed_at,
        "provider": "openrouter",
        "client_family": "openrouter",
        "account_hash": _openrouter_free_shared_account_hash(),
        "environment": record.get("litellm_environment"),
        "tenant_id": record.get("tenant_id"),
        "repository": record.get("repository"),
        "session_id": record.get("session_id"),
        "trace_id": record.get("trace_id"),
        "litellm_call_id": record.get("litellm_call_id"),
        "route_family": metadata.get("passthrough_route_family"),
        "request_model": record.get("model"),
        "response_model": None,
        "model": None,
        "model_family": "openrouter",
        "model_tier": "free",
        "client_name": record.get("client_name"),
        "client_version": record.get("client_version"),
        "client_user_agent": record.get("client_user_agent"),
        "metadata": metadata,
    }


def _build_openrouter_free_daily_observation(
    *,
    context: Dict[str, Any],
    day_start: datetime,
    day_end: datetime,
    used_requests: int,
    total_requests: int,
    signal: str,
    status: str = "observed",
    exhausted: bool = False,
    reset_hint_seconds: Optional[int] = None,
    provider_resets_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    bounded_total = max(total_requests, 1)
    bounded_used = max(0, used_requests)
    remaining_requests = max(0, bounded_total - bounded_used)
    used_percentage = round(
        min(100.0, (bounded_used / bounded_total) * 100.0),
        3,
    )
    remaining_pct = round(max(0.0, 100.0 - used_percentage), 3)
    return _finalize_rate_limit_observation(
        {
            "observed_at": context["observed_at"],
            "source": _AAWM_OPENROUTER_FREE_DAILY_SOURCE,
            "provider": "openrouter",
            "client_family": "openrouter",
            "account_hash": _openrouter_free_shared_account_hash(),
            "limit_id": "openrouter_free_daily_requests",
            "limit_name": "OpenRouter free daily requests",
            "limit_scope": "requests",
            "window_minutes": 1440,
            "quota_period": "daily",
            "quota_type": "requests",
            "provider_resets_at": provider_resets_at or day_end,
            "remaining_pct": remaining_pct,
            "used_percentage": used_percentage,
            "remaining_requests": remaining_requests,
            "used_requests": bounded_used,
            "total_requests": bounded_total,
            "status": status,
            "exhausted": exhausted or remaining_requests <= 0,
            "exhaustion_kind": "request_quota" if exhausted else None,
            "reset_hint_seconds": reset_hint_seconds,
            "model": None,
            "model_family": "openrouter",
            "model_tier": "free",
            "raw_provider_fields": {
                "dailyLimit": bounded_total,
                "usedRequests": bounded_used,
                "remainingRequests": remaining_requests,
                "windowStart": _json_safe_rate_limit_value(day_start),
                "windowEnd": _json_safe_rate_limit_value(day_end),
                "reset_anchor": "utc_midnight",
                "model_scope": "openrouter_:free_shared_pool",
                "meter_source": "local_session_history",
            },
            "evidence": {
                "signals": [signal],
                "provider_fields": [],
                "scope_note": (
                    "OpenRouter documents free-model quota as account-level; "
                    "provider does not expose current free request usage."
                ),
            },
        },
        context,
    )


def _openrouter_free_record_observed_at(record: Dict[str, Any]) -> datetime:
    return (
        _normalize_datetime(record.get("end_time"))
        or _normalize_datetime(record.get("start_time"))
        or datetime.now(timezone.utc)
    )


def _is_openrouter_free_session_history_record(record: Dict[str, Any]) -> bool:
    model = record.get("model")
    if not _is_openrouter_free_model(model):
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = _normalize_session_history_provider(
        record.get("provider"),
        str(model or ""),
        metadata,
    )
    return provider == "openrouter"


def _extract_codex_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(kwargs, result, observed_at, "codex_token_count")
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        rate_limits = candidate.get("rate_limits")
        if not isinstance(rate_limits, dict):
            continue
        if not (isinstance(rate_limits.get("primary"), dict) or isinstance(rate_limits.get("secondary"), dict)):
            continue
        limit_id = _clean_non_empty_string(rate_limits.get("limit_id"))
        limit_name = _clean_non_empty_string(rate_limits.get("limit_name"))
        for limit_scope in ("primary", "secondary"):
            window = rate_limits.get(limit_scope)
            if not isinstance(window, dict):
                continue
            window_minutes = _safe_int(window.get("window_minutes"))
            used_percentage = _safe_float(window.get("used_percent"))
            provider_resets_at = _parse_provider_timestamp(window.get("resets_at"))
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "codex_token_count",
                        "provider": "openai",
                        "client_family": "codex",
                        "limit_id": limit_id,
                        "limit_name": limit_name,
                        "limit_scope": limit_scope,
                        "window_minutes": window_minutes,
                        "provider_resets_at": provider_resets_at,
                        "used_percentage": used_percentage,
                        "exhausted": bool(used_percentage is not None and used_percentage >= 100),
                        "exhaustion_kind": (
                            rate_limits.get("rate_limit_reached_type")
                            if rate_limits.get("rate_limit_reached_type")
                            else None
                        ),
                        "raw_provider_fields": {
                            "limit_id": limit_id,
                            "limit_name": limit_name,
                            "limit_scope": limit_scope,
                            "window_minutes": window.get("window_minutes"),
                            "used_percent": window.get("used_percent"),
                            "resets_at": window.get("resets_at"),
                            "plan_type": rate_limits.get("plan_type"),
                            "rate_limit_reached_type": rate_limits.get("rate_limit_reached_type"),
                        },
                        "evidence": {
                            "signals": ["provider_rate_limits"],
                            "provider_fields": [
                                f"rate_limits.{limit_scope}.used_percent",
                                f"rate_limits.{limit_scope}.window_minutes",
                                f"rate_limits.{limit_scope}.resets_at",
                            ],
                        },
                    },
                    context,
                )
            )
    return _dedupe_rate_limit_observations(observations)


def _extract_codex_header_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "codex_response_headers",
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        lower_headers = _rate_limit_header_map(candidate)
        source = str(candidate.get("source") or "").lower()
        has_codex_header = any(
            isinstance(key, str) and key.lower().startswith("x-codex-") for key in list(candidate.keys())
        )
        if not has_codex_header and source != "codex_response_headers":
            continue

        active_limit = _get_rate_limit_header_value(
            candidate,
            "x-codex-active-limit",
            lower_headers=lower_headers,
        )
        header_groups = [
            {
                "header_prefix": "x-codex",
                "limit_id": "codex",
                "limit_name": (f"Codex {active_limit}" if active_limit else "Codex"),
            }
        ]
        bengalfox_limit_name = _clean_non_empty_string(
            _get_rate_limit_header_value(
                candidate,
                "x-codex-bengalfox-limit-name",
                lower_headers=lower_headers,
            )
        )
        if bengalfox_limit_name:
            header_groups.append(
                {
                    "header_prefix": "x-codex-bengalfox",
                    "limit_id": "codex_bengalfox",
                    "limit_name": bengalfox_limit_name,
                }
            )

        for header_group in header_groups:
            header_prefix = header_group["header_prefix"]
            for limit_scope, window_minutes in (
                ("primary", 300),
                ("secondary", 10080),
            ):
                reset_key = f"{header_prefix}-{limit_scope}-reset-at"
                reset_after_key = f"{header_prefix}-{limit_scope}-reset-after-seconds"
                used_percent_key = f"{header_prefix}-{limit_scope}-used-percent"
                window_minutes_key = f"{header_prefix}-{limit_scope}-window-minutes"
                over_limit_key = f"{header_prefix}-{limit_scope}-over-secondary-limit-percent"
                reset_value = _get_rate_limit_header_value(candidate, reset_key, lower_headers=lower_headers)
                reset_hint_seconds = _safe_int(
                    _get_rate_limit_header_value(candidate, reset_after_key, lower_headers=lower_headers)
                )
                used_percentage = _safe_float(
                    _get_rate_limit_header_value(candidate, used_percent_key, lower_headers=lower_headers)
                )
                raw_window_minutes = _get_rate_limit_header_value(
                    candidate, window_minutes_key, lower_headers=lower_headers
                )
                parsed_window_minutes = _safe_int(raw_window_minutes)
                if raw_window_minutes is not None and (parsed_window_minutes is None or parsed_window_minutes <= 0):
                    continue
                observed_window_minutes = parsed_window_minutes or window_minutes
                over_limit_percent = _safe_float(
                    _get_rate_limit_header_value(candidate, over_limit_key, lower_headers=lower_headers)
                )
                if (
                    reset_value is None
                    and reset_hint_seconds is None
                    and used_percentage is None
                    and over_limit_percent is None
                ):
                    continue
                provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                    reset_value,
                    context["observed_at"],
                    reset_hint_seconds,
                )
                if stale_reset:
                    continue
                observations.append(
                    _finalize_rate_limit_observation(
                        {
                            "observed_at": context["observed_at"],
                            "source": "codex_response_headers",
                            "provider": "openai",
                            "client_family": "codex",
                            "limit_id": header_group["limit_id"],
                            "limit_name": header_group["limit_name"],
                            "limit_scope": limit_scope,
                            "window_minutes": observed_window_minutes,
                            "provider_resets_at": provider_resets_at,
                            "used_percentage": used_percentage,
                            "reset_hint_seconds": reset_hint_seconds,
                            "exhausted": (
                                (used_percentage is not None and used_percentage >= 100)
                                or (over_limit_percent is not None and over_limit_percent > 0)
                            ),
                            "raw_provider_fields": {
                                reset_key: reset_value,
                                reset_after_key: _get_rate_limit_header_value(
                                    candidate,
                                    reset_after_key,
                                    lower_headers=lower_headers,
                                ),
                                over_limit_key: _get_rate_limit_header_value(
                                    candidate,
                                    over_limit_key,
                                    lower_headers=lower_headers,
                                ),
                                used_percent_key: _get_rate_limit_header_value(
                                    candidate,
                                    used_percent_key,
                                    lower_headers=lower_headers,
                                ),
                                window_minutes_key: _get_rate_limit_header_value(
                                    candidate,
                                    window_minutes_key,
                                    lower_headers=lower_headers,
                                ),
                                "x-codex-active-limit": _get_rate_limit_header_value(
                                    candidate,
                                    "x-codex-active-limit",
                                    lower_headers=lower_headers,
                                ),
                                "x-codex-credits-unlimited": _get_rate_limit_header_value(
                                    candidate,
                                    "x-codex-credits-unlimited",
                                    lower_headers=lower_headers,
                                ),
                            },
                            "evidence": {
                                "signals": ["codex_response_rate_limit_headers"],
                                "provider_fields": [
                                    reset_key,
                                    reset_after_key,
                                    used_percent_key,
                                    window_minutes_key,
                                    over_limit_key,
                                ],
                            },
                        },
                        context,
                    )
                )
    return _dedupe_rate_limit_observations(observations)


def _extract_error_payload_dicts(value: Any) -> List[Dict[str, Any]]:
    roots: List[Any] = [value, str(value)]
    for attr in (
        "detail",
        "body",
        "response",
        "message",
        "_hidden_params",
        "hidden_params",
        "additional_headers",
        "headers",
        "response_headers",
        "upstream_headers",
    ):
        try:
            attr_value = getattr(value, attr)
        except Exception:
            attr_value = None
        if attr_value is not None:
            roots.append(attr_value)
            if attr == "response":
                roots.extend(
                    [
                        getattr(attr_value, "text", None),
                        getattr(attr_value, "content", None),
                        getattr(attr_value, "headers", None),
                    ]
                )
    return _iter_rate_limit_dicts(*roots)


def _extract_codex_usage_limit_error_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "codex_usage_limit_error",
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _extract_error_payload_dicts(result) + _iter_rate_limit_dicts(
        *_rate_limit_candidate_roots(kwargs, result)
    ):
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        error_type = _clean_non_empty_string(error.get("type")) or _clean_non_empty_string(error.get("code"))
        message = _clean_non_empty_string(error.get("message"))
        if error_type != "usage_limit_reached" and not (isinstance(message, str) and "usage limit" in message.lower()):
            continue
        reset_hint_seconds = _parse_reset_hint_seconds(
            error.get("resets_in_seconds"),
            message,
        )
        provider_resets_at = _parse_provider_timestamp(error.get("resets_at"))
        if provider_resets_at is None and reset_hint_seconds is not None:
            provider_resets_at = context["observed_at"] + timedelta(seconds=reset_hint_seconds)
        limit_name = (
            _clean_non_empty_string(error.get("limit_name")) or _clean_non_empty_string(context.get("model")) or "codex"
        )
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": "codex_usage_limit_error",
                    "provider": "openai",
                    "client_family": "codex",
                    "limit_id": _clean_non_empty_string(error.get("limit_id")),
                    "limit_name": limit_name,
                    "limit_scope": _clean_non_empty_string(error.get("rate_limit_reached_type")) or "usage_limit",
                    "provider_resets_at": provider_resets_at,
                    "used_percentage": 100.0,
                    "status": "exhausted",
                    "exhausted": True,
                    "exhaustion_kind": "usage_limit_reached",
                    "reset_hint_seconds": reset_hint_seconds,
                    "raw_provider_fields": {
                        "type": error_type,
                        "message": message,
                        "plan_type": error.get("plan_type"),
                        "resets_at": error.get("resets_at"),
                        "resets_in_seconds": error.get("resets_in_seconds"),
                        "rate_limit_reached_type": error.get("rate_limit_reached_type"),
                    },
                    "evidence": {
                        "signals": ["usage_limit_error"],
                        "provider_fields": [
                            "error.type",
                            "error.resets_at",
                            "error.resets_in_seconds",
                        ],
                    },
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _rate_limit_header_map(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase header keys once per candidate for repeated lookups."""
    return {str(key).lower(): value for key, value in list(candidate.items()) if isinstance(key, str)}


def _get_rate_limit_header_value(
    candidate: Dict[str, Any],
    *header_names: str,
    lower_headers: Optional[Dict[str, Any]] = None,
) -> Any:
    if lower_headers is None:
        lower_headers = _rate_limit_header_map(candidate)
    for header_name in header_names:
        normalized_header_name = header_name.lower()
        for candidate_name in (
            normalized_header_name,
            f"llm_provider-{normalized_header_name}",
        ):
            value = lower_headers.get(candidate_name)
            if value is not None:
                return value
    return None


def _looks_like_claude_rate_limit_context(context: Dict[str, Any]) -> bool:
    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    context_text = " ".join(
        str(value)
        for value in (
            context.get("client_name"),
            context.get("client_user_agent"),
            context.get("route_family"),
            metadata.get("trace_name"),
            metadata.get("client_name"),
            metadata.get("cc_version"),
            metadata.get("cc_entrypoint"),
        )
        if value is not None
    ).lower()
    return "claude" in context_text or "cc_version" in metadata


def _extract_anthropic_header_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "anthropic_response_headers",
    )
    observations: List[Dict[str, Any]] = []
    client_family = "claude" if _looks_like_claude_rate_limit_context(context) else "anthropic"
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        lower_headers = _rate_limit_header_map(candidate)
        source = str(candidate.get("source") or "").lower()
        has_anthropic_header = any(
            isinstance(key, str)
            and (
                key.lower().startswith("anthropic-ratelimit-")
                or key.lower().startswith("llm_provider-anthropic-ratelimit-")
            )
            for key in list(candidate.keys())
        )
        if not has_anthropic_header and source != "anthropic_response_headers":
            continue
        for limit_scope, display_name, window_minutes in (
            ("5h", "Anthropic unified 5h", 300),
            ("7d", "Anthropic unified 7d", 10080),
            ("7d_oi", "Anthropic unified 7d overage included", 10080),
            ("7d_sonnet", "Anthropic unified 7d Sonnet", 10080),
        ):
            reset_key = f"anthropic-ratelimit-unified-{limit_scope}-reset"
            status_key = f"anthropic-ratelimit-unified-{limit_scope}-status"
            utilization_key = f"anthropic-ratelimit-unified-{limit_scope}-utilization"
            threshold_key = f"anthropic-ratelimit-unified-{limit_scope}-surpassed-threshold"
            reset_value = _get_rate_limit_header_value(candidate, reset_key, lower_headers=lower_headers)
            status_value = _clean_non_empty_string(
                _get_rate_limit_header_value(candidate, status_key, lower_headers=lower_headers)
            )
            utilization = _safe_float(
                _get_rate_limit_header_value(candidate, utilization_key, lower_headers=lower_headers)
            )
            threshold = _safe_float(_get_rate_limit_header_value(candidate, threshold_key, lower_headers=lower_headers))
            if reset_value is None and status_value is None and utilization is None:
                continue
            provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                reset_value,
                context["observed_at"],
            )
            if stale_reset:
                continue
            used_percentage = utilization * 100 if utilization is not None and utilization <= 1 else utilization
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "anthropic_response_headers",
                        "provider": "anthropic",
                        "client_family": client_family,
                        "limit_id": f"anthropic_unified_{limit_scope}",
                        "limit_name": display_name,
                        "limit_scope": limit_scope,
                        "window_minutes": window_minutes,
                        "provider_resets_at": provider_resets_at,
                        "used_percentage": used_percentage,
                        "status": status_value,
                        "exhausted": status_value in {"rejected", "exhausted"},
                        "raw_provider_fields": {
                            reset_key: reset_value,
                            status_key: status_value,
                            utilization_key: _get_rate_limit_header_value(
                                candidate, utilization_key, lower_headers=lower_headers
                            ),
                            threshold_key: _get_rate_limit_header_value(
                                candidate, threshold_key, lower_headers=lower_headers
                            ),
                            "surpassed_threshold": threshold,
                            "anthropic-ratelimit-unified-representative-claim": _get_rate_limit_header_value(
                                candidate,
                                "anthropic-ratelimit-unified-representative-claim",
                                lower_headers=lower_headers,
                            ),
                            "anthropic-ratelimit-unified-overage-status": _get_rate_limit_header_value(
                                candidate,
                                "anthropic-ratelimit-unified-overage-status",
                                lower_headers=lower_headers,
                            ),
                        },
                        "evidence": {
                            "signals": ["anthropic_unified_rate_limit_headers"],
                            "provider_fields": [
                                reset_key,
                                status_key,
                                utilization_key,
                                threshold_key,
                            ],
                        },
                    },
                    context,
                )
            )
        for limit_scope, total_key, remaining_key, reset_key in (
            (
                "requests",
                "anthropic-ratelimit-requests-limit",
                "anthropic-ratelimit-requests-remaining",
                "anthropic-ratelimit-requests-reset",
            ),
            (
                "tokens",
                "anthropic-ratelimit-tokens-limit",
                "anthropic-ratelimit-tokens-remaining",
                "anthropic-ratelimit-tokens-reset",
            ),
        ):
            total = _safe_int(_get_rate_limit_header_value(candidate, total_key, lower_headers=lower_headers))
            remaining = _safe_int(_get_rate_limit_header_value(candidate, remaining_key, lower_headers=lower_headers))
            reset_value = _get_rate_limit_header_value(candidate, reset_key, lower_headers=lower_headers)
            if total is None and remaining is None and reset_value is None:
                continue
            provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                reset_value,
                context["observed_at"],
            )
            if stale_reset:
                continue
            used = max(0, total - remaining) if total is not None and remaining is not None else None
            used_percentage = (used / total) * 100 if used is not None and total is not None and total > 0 else None
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "anthropic_response_headers",
                        "provider": "anthropic",
                        "client_family": client_family,
                        "limit_id": f"anthropic_{limit_scope}",
                        "limit_name": f"Anthropic {limit_scope} rate limit",
                        "limit_scope": limit_scope,
                        "provider_resets_at": provider_resets_at,
                        "used_percentage": used_percentage,
                        "remaining_requests": remaining,
                        "used_requests": used,
                        "total_requests": total,
                        "raw_provider_fields": {
                            total_key: _get_rate_limit_header_value(candidate, total_key, lower_headers=lower_headers),
                            remaining_key: _get_rate_limit_header_value(
                                candidate, remaining_key, lower_headers=lower_headers
                            ),
                            reset_key: reset_value,
                        },
                        "evidence": {
                            "signals": ["anthropic_response_rate_limit_headers"],
                            "provider_fields": [
                                total_key,
                                remaining_key,
                                reset_key,
                            ],
                        },
                    },
                    context,
                )
            )
    return _dedupe_rate_limit_observations(observations)


def _first_quota_number(candidate: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = _safe_int(candidate.get(key))
        if value is not None:
            return value
    return None


def _first_quota_float(candidate: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _safe_float(candidate.get(key))
        if value is not None:
            return value
    return None


def _looks_like_xai_oauth_rate_limit_context(context: Dict[str, Any]) -> bool:
    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    credential_family = str(metadata.get("credential_family") or "").lower()
    route_family = str(
        metadata.get("passthrough_route_family") or metadata.get("route_family") or context.get("route_family") or ""
    ).lower()
    model = str(context.get("model") or "").lower()
    request_model = str(context.get("request_model") or "").lower()
    return (
        credential_family == "xai_oauth"
        or metadata.get("xai_oauth_managed") is True
        or metadata.get("xai_oauth_public_model") is not None
        or "xai_oauth" in route_family
        or model.startswith("oa_xai/")
        or request_model.startswith("oa_xai/")
    )


def _extract_xai_oauth_account_hash(metadata: Dict[str, Any]) -> Optional[str]:
    for key in ("xai_oauth_account_hash", "provider_account_hash"):
        value = _clean_non_empty_string(metadata.get(key))
        if value:
            return value
    for key in (
        "xai_oauth_account_id",
        "provider_account_id",
        "organization_id",
        "org_id",
    ):
        value = _clean_non_empty_string(metadata.get(key))
        if value:
            return _short_hash(value.encode("utf-8"))
    return None


def _xai_oauth_header_remaining_pct(
    total: Optional[int],
    remaining: Optional[int],
) -> Optional[float]:
    if total is None or remaining is None or total <= 0:
        return None
    return round(max(0.0, min(100.0, (remaining / total) * 100.0)), 3)


def _next_utc_month_start(value: Any) -> Optional[datetime]:
    observed_dt = _normalize_datetime(value)
    if observed_dt is None:
        return None
    observed_dt = observed_dt.astimezone(timezone.utc)
    if observed_dt.month == 12:
        return datetime(observed_dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(observed_dt.year, observed_dt.month + 1, 1, tzinfo=timezone.utc)


def _is_xai_oauth_subscription_quota_context(metadata: Dict[str, Any]) -> bool:
    quota_family = str(metadata.get("xai_quota_family") or metadata.get("shared_quota_family") or "").strip().lower()
    return quota_family == "xai_grok_subscription" or metadata.get("grok_subscription_quota_shared") is True


def _extract_xai_oauth_billing_period_end(
    *,
    candidate: Dict[str, Any],
    metadata: Dict[str, Any],
    observed_at: Any,
) -> Tuple[Optional[datetime], Optional[str]]:
    for source, value in (
        ("payload_billing_period_end", candidate.get("billingPeriodEnd")),
        (
            "payload_config_billing_period_end",
            _maybe_get_path(candidate, "config", "billingPeriodEnd"),
        ),
        ("metadata_billing_period_end", metadata.get("billingPeriodEnd")),
        (
            "metadata_xai_oauth_billing_period_end",
            metadata.get("xai_oauth_billing_period_end"),
        ),
    ):
        parsed = _parse_provider_timestamp(value)
        if parsed is not None:
            return parsed, source

    if _is_xai_oauth_subscription_quota_context(metadata):
        fallback = _next_utc_month_start(observed_at)
        if fallback is not None:
            return fallback, "xai_grok_subscription_month_boundary"

    return None, None


def _extract_xai_oauth_header_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "xai_oauth_response_headers",
    )
    if context.get("provider") != "xai" or not _looks_like_xai_oauth_rate_limit_context(context):
        return []
    raw_metadata = context.get("metadata")
    metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    account_hash = _extract_xai_oauth_account_hash(metadata)
    model = _clean_non_empty_string(metadata.get("xai_oauth_public_model")) or (
        _clean_non_empty_string(context.get("model")) if context.get("model") != "unknown" else None
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        lower_headers = _rate_limit_header_map(candidate)
        source = str(candidate.get("source") or "").lower()
        has_xai_header = any(
            isinstance(key, str) and key.lower().startswith("x-ratelimit-") for key in list(candidate.keys())
        )
        if not has_xai_header and source != "xai_oauth_response_headers":
            continue

        for limit_scope, total_key, remaining_key, reset_keys in (
            (
                "requests",
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                (
                    "x-ratelimit-reset-requests",
                    "x-ratelimit-reset-request",
                    "x-ratelimit-reset",
                ),
            ),
            (
                "tokens",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-tokens",
                (
                    "x-ratelimit-reset-tokens",
                    "x-ratelimit-reset-token",
                    "x-ratelimit-reset",
                ),
            ),
        ):
            total = _safe_int(_get_rate_limit_header_value(candidate, total_key, lower_headers=lower_headers))
            remaining = _safe_int(_get_rate_limit_header_value(candidate, remaining_key, lower_headers=lower_headers))
            reset_value = _get_rate_limit_header_value(candidate, *reset_keys, lower_headers=lower_headers)
            reset_hint_seconds = _parse_reset_hint_seconds(
                _get_rate_limit_header_value(candidate, "retry-after", lower_headers=lower_headers)
            )
            if total is None and remaining is None and reset_value is None and reset_hint_seconds is None:
                continue
            if total is not None and total <= 0:
                continue
            provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                reset_value,
                context["observed_at"],
                reset_hint_seconds,
            )
            if stale_reset:
                continue
            reset_source = "response_header" if provider_resets_at is not None else None
            if provider_resets_at is None and reset_hint_seconds is None:
                (
                    provider_resets_at,
                    reset_source,
                ) = _extract_xai_oauth_billing_period_end(
                    candidate=candidate,
                    metadata=metadata,
                    observed_at=context["observed_at"],
                )
            elif reset_hint_seconds is not None and provider_resets_at is not None:
                reset_source = "retry_after"
            used = max(0, total - remaining) if total is not None and remaining is not None else None
            remaining_pct = _xai_oauth_header_remaining_pct(total, remaining)
            used_percentage = (
                round(max(0.0, min(100.0, 100.0 - remaining_pct)), 3) if remaining_pct is not None else None
            )
            exhausted = remaining is not None and remaining <= 0
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "xai_oauth_response_headers",
                        "provider": "xai",
                        "client_family": "xai_oauth",
                        "account_hash": account_hash,
                        "limit_id": f"xai_oauth_{limit_scope}",
                        "limit_name": f"xAI OAuth {limit_scope} rate limit",
                        "limit_scope": limit_scope,
                        "quota_period": (
                            "monthly"
                            if reset_source
                            in {
                                "payload_billing_period_end",
                                "payload_config_billing_period_end",
                                "metadata_billing_period_end",
                                "metadata_xai_oauth_billing_period_end",
                                "xai_grok_subscription_month_boundary",
                            }
                            else None
                        ),
                        "quota_type": limit_scope,
                        "provider_resets_at": provider_resets_at,
                        "remaining_pct": remaining_pct,
                        "quota_limit": float(total) if total is not None else None,
                        "quota_used": float(used) if used is not None else None,
                        "quota_remaining": (float(remaining) if remaining is not None else None),
                        "billing_period_end_at": provider_resets_at
                        if reset_source
                        in {
                            "payload_billing_period_end",
                            "payload_config_billing_period_end",
                            "metadata_billing_period_end",
                            "metadata_xai_oauth_billing_period_end",
                            "xai_grok_subscription_month_boundary",
                        }
                        else None,
                        "used_percentage": used_percentage,
                        "remaining_requests": remaining,
                        "used_requests": used,
                        "total_requests": total,
                        "status": "quota_exhausted" if exhausted else "observed",
                        "exhausted": exhausted,
                        "exhaustion_kind": "rate_limit" if exhausted else None,
                        "reset_hint_seconds": reset_hint_seconds,
                        "model": model,
                        "model_family": "grok",
                        "raw_provider_fields": {
                            total_key: _get_rate_limit_header_value(candidate, total_key, lower_headers=lower_headers),
                            remaining_key: _get_rate_limit_header_value(
                                candidate, remaining_key, lower_headers=lower_headers
                            ),
                            "reset": reset_value,
                            "retry-after": _get_rate_limit_header_value(
                                candidate, "retry-after", lower_headers=lower_headers
                            ),
                            "billingPeriodEnd": _json_safe_rate_limit_value(
                                _maybe_get_path(candidate, "config", "billingPeriodEnd")
                                or candidate.get("billingPeriodEnd")
                                or metadata.get("xai_oauth_billing_period_end")
                                or metadata.get("billingPeriodEnd")
                            ),
                            "quota_unit": f"xai_oauth_{limit_scope}",
                            "quota_unit_interpretation": limit_scope,
                        },
                        "evidence": {
                            "signals": ["xai_oauth_response_rate_limit_headers"],
                            "provider_fields": [
                                total_key,
                                remaining_key,
                                *reset_keys,
                                "retry-after",
                            ],
                            "reset_absent": provider_resets_at is None,
                            "reset_header_absent": (reset_value is None and reset_hint_seconds is None),
                            "reset_source": reset_source,
                        },
                    },
                    context,
                )
            )
    return _dedupe_rate_limit_observations(observations)


def _grok_billing_quota_value(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("val")
    return _safe_float(value)


def _grok_billing_current_period(config: Dict[str, Any]) -> Dict[str, Any]:
    current_period = config.get("currentPeriod")
    return current_period if isinstance(current_period, dict) else {}


def _grok_billing_is_weekly_period(config: Dict[str, Any]) -> bool:
    current_period = _grok_billing_current_period(config)
    period_type = str(current_period.get("type") or "").strip()
    return period_type == USAGE_PERIOD_TYPE_WEEKLY


def _grok_billing_period_bounds(
    config: Dict[str, Any],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    current_period = _grok_billing_current_period(config)
    billing_period_start_at = _parse_provider_timestamp(config.get("billingPeriodStart") or current_period.get("start"))
    billing_period_end_at = _parse_provider_timestamp(config.get("billingPeriodEnd") or current_period.get("end"))
    return billing_period_start_at, billing_period_end_at


def _is_grok_billing_context(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> bool:
    route_text = " ".join(
        str(value)
        for value in (
            metadata.get("passthrough_route_family"),
            metadata.get("user_api_key_request_route"),
            metadata.get("api_base"),
            _maybe_get_path(kwargs.get("standard_pass_through_logging_payload"), "url"),
            _maybe_get_path(kwargs.get("passthrough_logging_payload"), "url"),
        )
        if value is not None
    ).lower()
    if "/billing" in route_text and ("grok" in route_text or "xai" in route_text or "x.ai" in route_text):
        return True
    if metadata.get("grok_cli_chat_proxy") is True or metadata.get("xai_cli_chat_proxy") is True:
        return True
    headers = _extract_headers_from_kwargs(kwargs)
    return any(header_name.startswith("x-grok-") or header_name == "x-xai-token-auth" for header_name in headers)


def _extract_grok_billing_config(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    config = candidate.get("config") if isinstance(candidate.get("config"), dict) else candidate
    if not isinstance(config, dict):
        return None
    has_absolute_quota = isinstance(config.get("monthlyLimit"), dict) and isinstance(
        config.get("used"),
        dict,
    )
    has_percentage_quota = _safe_float(config.get("creditUsagePercent")) is not None
    has_weekly_period = _grok_billing_is_weekly_period(config)
    billing_period_start_at, billing_period_end_at = _grok_billing_period_bounds(config)
    has_period_bounds = billing_period_start_at is not None or billing_period_end_at is not None
    if not has_absolute_quota and not has_percentage_quota and not (has_weekly_period and has_period_bounds):
        return None
    return config


def _grok_billing_model(
    context: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    return (
        (_clean_non_empty_string(context.get("model")) if context.get("model") != "unknown" else None)
        or _clean_non_empty_string(metadata.get("grok_model_override"))
        or "grok-build"
    )


def _grok_billing_request_contract_evidence(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    fingerprint = _clean_non_empty_string(metadata.get("grok_billing_passthrough_request_contract_fingerprint"))
    if not fingerprint:
        return {}

    evidence: Dict[str, Any] = {
        "request_contract_fingerprint": fingerprint,
    }
    for metadata_key, evidence_key in (
        ("grok_billing_passthrough_http_client", "request_contract_http_client"),
        ("grok_billing_passthrough_request_method", "request_contract_method"),
        ("grok_billing_passthrough_target_host", "request_contract_target_host"),
        ("grok_billing_passthrough_target_path", "request_contract_target_path"),
        ("grok_billing_passthrough_user_agent", "request_contract_user_agent"),
    ):
        value = _clean_non_empty_string(metadata.get(metadata_key))
        if value:
            evidence[evidence_key] = value

    for metadata_key, evidence_key in (
        ("grok_billing_passthrough_query_keys", "request_contract_query_keys"),
        ("grok_billing_passthrough_header_names", "request_contract_header_names"),
    ):
        value = metadata.get(metadata_key)
        if isinstance(value, list):
            evidence[evidence_key] = [str(item) for item in value if isinstance(item, (str, int, float)) and str(item)]

    configured = metadata.get("grok_billing_passthrough_x_xai_token_auth_configured")
    if configured is not None:
        evidence["request_contract_x_xai_token_auth_configured"] = bool(configured)

    return evidence


def _grok_billing_snapshot_parts(
    config: Dict[str, Any],
    *,
    base_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(config, dict):
        return None

    billing_period_start_at, billing_period_end_at = _grok_billing_period_bounds(config)
    is_weekly = _grok_billing_is_weekly_period(config)
    monthly_limit = _grok_billing_quota_value(config.get("monthlyLimit"))
    used = _grok_billing_quota_value(config.get("used"))
    credit_usage_percent = _safe_float(config.get("creditUsagePercent"))

    evidence: Dict[str, Any] = dict(base_evidence or {})
    signals = list(evidence.get("signals") or [])
    if "grok_billing_payload" not in signals:
        signals.append("grok_billing_payload")
    evidence["signals"] = signals

    if monthly_limit is not None and monthly_limit > 0 and used is not None and used >= 0:
        used_percentage = max(0.0, min(100.0, (used / monthly_limit) * 100.0))
        remaining_pct = float(int(math.floor(max(0.0, min(100.0, 100.0 - used_percentage)) + 0.5)))
        quota_remaining = max(0.0, monthly_limit - used)
        if "grok_billing_monthly_counter" not in signals:
            signals.append("grok_billing_monthly_counter")
        return {
            "quota_key": GROK_BILLING_MONTHLY_REQUESTS_QUOTA_KEY,
            "quota_period": "monthly",
            "quota_type": "requests",
            "limit_id": "xai_grok_build_monthly_requests",
            "limit_name": "Grok Build monthly requests",
            "limit_scope": "requests",
            "remaining_pct": remaining_pct,
            "used_percentage": float(100.0 - remaining_pct),
            "quota_limit": monthly_limit,
            "quota_used": used,
            "quota_remaining": quota_remaining,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "monthlyLimit": _json_safe_rate_limit_value(config.get("monthlyLimit")),
                "used": _json_safe_rate_limit_value(config.get("used")),
                "onDemandCap": _json_safe_rate_limit_value(config.get("onDemandCap")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_used",
                "quota_unit_interpretation": "requests",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.monthlyLimit.val",
                    "config.used.val",
                    "config.billingPeriodEnd",
                ],
                "rounding": "whole_remaining_percentage",
                "unit_note": ("Grok billing does not label used.val; observed tool " "traffic behaves request-like."),
            },
        }

    if is_weekly and credit_usage_percent is not None:
        used_percentage = max(0.0, min(100.0, credit_usage_percent))
        remaining_pct = max(0.0, min(100.0, 100.0 - used_percentage))
        for signal in (
            "grok_billing_weekly_credit",
            "grok_billing_percentage_only",
        ):
            if signal not in signals:
                signals.append(signal)
        return {
            "quota_key": GROK_BILLING_WEEKLY_CREDITS_QUOTA_KEY,
            "quota_period": "weekly",
            "quota_type": "credits",
            "limit_id": "xai_grok_build_weekly_credits",
            "limit_name": "Grok Build weekly credits",
            "limit_scope": "credits",
            "remaining_pct": remaining_pct,
            "used_percentage": used_percentage,
            "quota_limit": None,
            "quota_used": None,
            "quota_remaining": None,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "creditUsagePercent": _json_safe_rate_limit_value(config.get("creditUsagePercent")),
                "productUsage": _json_safe_rate_limit_value(config.get("productUsage")),
                "currentPeriod": _json_safe_rate_limit_value(config.get("currentPeriod")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_credit_usage_percent",
                "quota_unit_interpretation": "percent_of_credit_quota",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.creditUsagePercent",
                    "config.productUsage",
                    "config.currentPeriod.type",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Grok billing provided percentage-only weekly credit usage; "
                    "absolute quota counts are intentionally left null."
                ),
            },
        }

    if is_weekly and (billing_period_start_at is not None or billing_period_end_at is not None):
        if "grok_billing_weekly_fresh_period" not in signals:
            signals.append("grok_billing_weekly_fresh_period")
        return {
            "quota_key": GROK_BILLING_WEEKLY_CREDITS_QUOTA_KEY,
            "quota_period": "weekly",
            "quota_type": "credits",
            "limit_id": "xai_grok_build_weekly_credits",
            "limit_name": "Grok Build weekly credits",
            "limit_scope": "credits",
            "remaining_pct": 100.0,
            "used_percentage": 0.0,
            "quota_limit": None,
            "quota_used": None,
            "quota_remaining": None,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "currentPeriod": _json_safe_rate_limit_value(config.get("currentPeriod")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_weekly_credit_fresh_period",
                "quota_unit_interpretation": "percent_of_credit_quota",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.currentPeriod.type",
                    "config.currentPeriod.start",
                    "config.currentPeriod.end",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Fresh weekly Grok Build credit periods omit creditUsagePercent; "
                    "remaining percent is inferred as 100% used / 0% consumed."
                ),
            },
        }

    if credit_usage_percent is not None:
        used_percentage = max(0.0, min(100.0, credit_usage_percent))
        remaining_pct = max(0.0, min(100.0, 100.0 - used_percentage))
        if "grok_billing_percentage_only" not in signals:
            signals.append("grok_billing_percentage_only")
        if "grok_billing_legacy_monthly_credit" not in signals:
            signals.append("grok_billing_legacy_monthly_credit")
        return {
            "quota_key": GROK_BILLING_MONTHLY_CREDITS_QUOTA_KEY,
            "quota_period": "monthly",
            "quota_type": "credits",
            "limit_id": "xai_grok_build_monthly_credits",
            "limit_name": "Grok Build monthly credits",
            "limit_scope": "credits",
            "remaining_pct": remaining_pct,
            "used_percentage": used_percentage,
            "quota_limit": None,
            "quota_used": None,
            "quota_remaining": None,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "creditUsagePercent": _json_safe_rate_limit_value(config.get("creditUsagePercent")),
                "productUsage": _json_safe_rate_limit_value(config.get("productUsage")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_credit_usage_percent",
                "quota_unit_interpretation": "percent_of_credit_quota",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.creditUsagePercent",
                    "config.productUsage",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Grok billing provided percentage-only credit usage; "
                    "absolute quota counts are intentionally left null."
                ),
            },
        }

    return None


def _extract_grok_billing_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    metadata = _merged_rate_limit_metadata(kwargs)
    if not _is_grok_billing_context(kwargs, metadata):
        return []

    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "grok_billing",
    )
    request_contract_evidence = _grok_billing_request_contract_evidence(metadata)
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        config = _extract_grok_billing_config(candidate)
        if config is None:
            continue

        snapshot = _grok_billing_snapshot_parts(
            config,
            base_evidence=request_contract_evidence,
        )
        if snapshot is None:
            continue

        model = _grok_billing_model(context, metadata)
        provider_resets_at = snapshot.get("billing_period_end_at")
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": "grok_billing",
                    "provider": "xai",
                    "client_family": "grok-build",
                    "limit_id": snapshot["limit_id"],
                    "limit_name": snapshot["limit_name"],
                    "limit_scope": snapshot["limit_scope"],
                    "quota_period": snapshot["quota_period"],
                    "quota_type": snapshot["quota_type"],
                    "provider_resets_at": provider_resets_at,
                    "remaining_pct": snapshot["remaining_pct"],
                    "quota_limit": snapshot["quota_limit"],
                    "quota_used": snapshot["quota_used"],
                    "quota_remaining": snapshot["quota_remaining"],
                    "billing_period_start_at": snapshot["billing_period_start_at"],
                    "billing_period_end_at": snapshot["billing_period_end_at"],
                    "used_percentage": snapshot["used_percentage"],
                    "model": model,
                    "model_family": "grok",
                    "raw_provider_fields": snapshot["raw_provider_fields"],
                    "evidence": snapshot["evidence"],
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _extract_openrouter_free_error_reset_at(
    kwargs: Dict[str, Any],
    result: Any,
    dicts: List[Dict[str, Any]],
    error_text: str,
    observed_at: datetime,
) -> Tuple[datetime, Optional[int]]:
    headers = _extract_headers_from_kwargs(kwargs)
    headers.update(_extract_provider_error_headers(result))
    retry_after_seconds = _extract_provider_error_retry_after_seconds(
        kwargs=kwargs,
        result=result,
        dicts=dicts,
        error_text=error_text,
    )
    reset_hint_seconds = (
        int(retry_after_seconds) if retry_after_seconds is not None and retry_after_seconds >= 0 else None
    )
    reset_value = _first_non_empty_string(
        headers.get("x-ratelimit-reset"),
        headers.get("x-rate-limit-reset"),
        headers.get("x-ratelimit-reset-at"),
        headers.get("x-rate-limit-reset-at"),
    )
    provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
        reset_value,
        observed_at,
        reset_hint_seconds,
    )
    if provider_resets_at is not None and not stale_reset:
        return provider_resets_at, reset_hint_seconds
    _day_start, day_end = _openrouter_free_daily_window(observed_at)
    return day_end, reset_hint_seconds


def _extract_openrouter_free_error_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        _AAWM_OPENROUTER_FREE_DAILY_SOURCE,
    )
    if context.get("provider") != "openrouter":
        return []
    model_candidates = (
        context.get("model"),
        context.get("request_model"),
        _maybe_get_path(
            kwargs.get("passthrough_logging_payload"),
            "request_body",
            "model",
        ),
        _maybe_get_path(
            kwargs.get("litellm_params"),
            "proxy_server_request",
            "body",
            "model",
        ),
    )
    if not any(_is_openrouter_free_model(model) for model in model_candidates):
        return []

    dicts = _extract_provider_error_dicts(result)
    status_code = _extract_provider_error_status_code(result, dicts)
    error_text = _extract_provider_error_text(result, dicts)
    error_code, error_type = _extract_provider_error_code_and_type(result, dicts)
    error_class = _classify_provider_error(
        status_code=status_code,
        error_code=error_code,
        error_type=error_type,
        error_text=error_text,
    )
    if error_class not in {"rate_limited", "usage_limit_reached"}:
        return []

    observed_dt = context["observed_at"]
    day_start, day_end = _openrouter_free_daily_window(observed_dt)
    provider_resets_at, reset_hint_seconds = _extract_openrouter_free_error_reset_at(
        kwargs,
        result,
        dicts,
        error_text,
        observed_dt,
    )
    context = dict(context)
    context["account_hash"] = _openrouter_free_shared_account_hash()
    context["client_family"] = "openrouter"
    context["model"] = None
    total_requests = _openrouter_free_daily_request_limit()
    return [
        _build_openrouter_free_daily_observation(
            context=context,
            day_start=day_start,
            day_end=day_end,
            used_requests=total_requests,
            total_requests=total_requests,
            signal="openrouter_free_model_rate_limit_error",
            status="quota_exhausted",
            exhausted=True,
            reset_hint_seconds=reset_hint_seconds,
            provider_resets_at=provider_resets_at,
        )
    ]


def _looks_like_google_quota_candidate(candidate: Dict[str, Any]) -> bool:
    request_quota_keys = {
        "buckets",
        "modelId",
        "tokenType",
        "remainingFraction",
        "remainingRequests",
        "remaining_requests",
        "requestsRemaining",
        "usedRequests",
        "used_requests",
        "requestsUsed",
        "totalRequests",
        "total_requests",
        "requestLimit",
        "dailyLimit",
        "quotaId",
        "quotaName",
    }
    weak_quota_keys = {
        "usagePercentage",
        "usedPercentage",
        "used_percentage",
    }
    candidate_keys = set(candidate.keys())
    if request_quota_keys.intersection(candidate_keys):
        return True
    source = str(candidate.get("source") or "").lower()
    return bool(weak_quota_keys.intersection(candidate_keys)) and ("google" in source or "gemini" in source)


def _antigravity_quota_pool_for_model(model: Optional[str]) -> Tuple[str, str, str]:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("claude-") or normalized.startswith("gpt-oss"):
        return (
            "vertex_pool",
            "Antigravity Code Assist Vertex pool",
            "vertex",
        )
    return (
        "gemini_pool",
        "Antigravity Code Assist Gemini pool",
        "gemini",
    )


def _extract_google_quota_observations(  # noqa: PLR0915
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "google_retrieve_user_quota",
    )
    observations: List[Dict[str, Any]] = []
    raw_metadata = context.get("metadata")
    metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    default_quota_source = _clean_non_empty_string(_maybe_get(metadata.get("google_retrieve_user_quota"), "source"))
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        if not _looks_like_google_quota_candidate(candidate):
            continue
        quota_source = (
            _clean_non_empty_string(candidate.get("source")) or default_quota_source or "google_retrieve_user_quota"
        )
        remaining_requests = _first_quota_number(
            candidate,
            "remainingRequests",
            "remaining_requests",
            "requestsRemaining",
            "remaining",
        )
        used_requests = _first_quota_number(
            candidate,
            "usedRequests",
            "used_requests",
            "requestsUsed",
            "currentUsage",
            "used",
        )
        total_requests = _first_quota_number(
            candidate,
            "totalRequests",
            "total_requests",
            "requestLimit",
            "dailyLimit",
            "limit",
            "quota",
        )
        used_percentage = _first_quota_float(
            candidate,
            "usedPercentage",
            "used_percentage",
            "usagePercentage",
        )
        remaining_fraction = _first_quota_float(
            candidate,
            "remainingFraction",
            "remaining_fraction",
        )
        if used_percentage is None and used_requests is not None and total_requests:
            used_percentage = (used_requests / total_requests) * 100
        if used_percentage is None and remaining_fraction is not None:
            used_percentage = max(0.0, min(100.0, (1 - remaining_fraction) * 100))
        reset_source_value = _first_non_none(
            candidate.get("resetsAt"),
            candidate.get("resets_at"),
            candidate.get("resetAt"),
            candidate.get("resetTime"),
        )
        provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
            reset_source_value,
            context["observed_at"],
        )
        if stale_reset:
            continue
        if (
            remaining_requests is None
            and used_requests is None
            and total_requests is None
            and used_percentage is None
            and reset_source_value is None
            and _clean_non_empty_string(candidate.get("quotaId")) is None
            and _clean_non_empty_string(candidate.get("quotaName")) is None
        ):
            continue
        model = (
            _clean_non_empty_string(candidate.get("modelId"))
            or _clean_non_empty_string(candidate.get("model"))
            or context.get("model")
        )
        model_family, model_tier = _infer_model_family_and_tier(model)
        token_type = _clean_non_empty_string(candidate.get("tokenType"))
        provider = context.get("provider") or "gemini"
        client_family = context.get("client_family") or _infer_rate_limit_client_family(
            provider,
            str(model or ""),
            metadata,
            quota_source,
        )
        is_antigravity_quota = (
            provider == "antigravity"
            or client_family == "antigravity_code_assist"
            or str(quota_source or "").lower().startswith("antigravity_")
        )
        if is_antigravity_quota and provider_resets_at is None:
            continue
        explicit_quota_period = _normalize_quota_period(candidate.get("quotaPeriod")) or _normalize_quota_period(
            candidate.get("period")
        )
        quota_period = explicit_quota_period or ("five_hour" if is_antigravity_quota else "daily")
        window_minutes = None
        for window_candidate in (
            candidate.get("windowMinutes"),
            candidate.get("window_minutes"),
            candidate.get("windowMinutesEstimate"),
        ):
            parsed_window_minutes = _safe_int(window_candidate)
            if parsed_window_minutes is not None and parsed_window_minutes > 0:
                window_minutes = parsed_window_minutes
                break
        window_minutes = window_minutes or _window_minutes_from_quota_period(quota_period)
        if is_antigravity_quota:
            limit_scope, limit_name, model_family = _antigravity_quota_pool_for_model(model)
            limit_id = "antigravity_code_assist"
            stored_model = None
            quota_type = "wtus"
            model_tier = None
            provider = "antigravity"
            client_family = "antigravity_code_assist"
        else:
            limit_scope = (
                "model_requests"
                if _clean_non_empty_string(candidate.get("modelId")) or _clean_non_empty_string(candidate.get("model"))
                else "daily_request_pool"
            )
            limit_id = (
                f"google_code_assist_requests_{model}"
                if limit_scope == "model_requests" and model
                else "google_code_assist_requests"
            )
            limit_name = f"Google Code Assist {model} requests" if model else "Google Code Assist requests"
            stored_model = model
            quota_type = None
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": quota_source,
                    "provider": provider,
                    "client_family": client_family,
                    "limit_id": limit_id
                    if is_antigravity_quota
                    else _clean_non_empty_string(candidate.get("quotaId")) or limit_id,
                    "limit_name": limit_name
                    if is_antigravity_quota
                    else _clean_non_empty_string(candidate.get("quotaName")) or limit_name,
                    "limit_scope": limit_scope,
                    "window_minutes": window_minutes,
                    "quota_period": quota_period,
                    "quota_type": quota_type,
                    "provider_resets_at": provider_resets_at,
                    "used_percentage": used_percentage,
                    "remaining_requests": remaining_requests,
                    "used_requests": used_requests,
                    "total_requests": total_requests,
                    "model": stored_model,
                    "model_family": model_family,
                    "model_tier": model_tier,
                    "raw_provider_fields": {
                        key: candidate.get(key)
                        for key in (
                            "modelId",
                            "tokenType",
                            "remainingFraction",
                            "remainingRequests",
                            "remaining_requests",
                            "usedRequests",
                            "used_requests",
                            "totalRequests",
                            "total_requests",
                            "usagePercentage",
                            "usedPercentage",
                            "quotaPeriod",
                            "period",
                            "model",
                            "quotaId",
                            "resetsAt",
                            "resets_at",
                            "resetAt",
                            "resetTime",
                            "windowMinutes",
                            "window_minutes",
                            "windowMinutesEstimate",
                        )
                        if key in candidate
                    },
                    "evidence": {
                        "signals": ["google_quota_payload"],
                        "provider_fields": sorted(
                            key
                            for key in list(candidate.keys())
                            if "quota" in key.lower()
                            or "request" in key.lower()
                            or "usage" in key.lower()
                            or "fraction" in key.lower()
                            or "reset" in key.lower()
                            or key in {"modelId", "tokenType"}
                        )[:20],
                        "token_type": token_type,
                    },
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _extract_google_error_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "google_generate_content_error",
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _extract_error_payload_dicts(result) + _iter_rate_limit_dicts(
        *_rate_limit_candidate_roots(kwargs, result)
    ):
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        status_text = _clean_non_empty_string(error.get("status"))
        code = _safe_int(error.get("code"))
        message = _clean_non_empty_string(error.get("message")) or ""
        raw_details = error.get("details")
        details: List[Any] = raw_details if isinstance(raw_details, list) else []
        reasons = [
            _clean_non_empty_string(_maybe_get(detail, "reason")) for detail in details if isinstance(detail, dict)
        ]
        reasons = [reason for reason in reasons if reason]
        metadata_models = [
            _clean_non_empty_string(_maybe_get_path(detail, "metadata", "model"))
            for detail in details
            if isinstance(detail, dict)
        ]
        metadata_models = [model for model in metadata_models if model]
        is_resource_exhausted = (
            code == 429
            or status_text in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}
            or any(reason in {"MODEL_CAPACITY_EXHAUSTED", "RATE_LIMIT_EXCEEDED"} for reason in reasons)
        )
        if not is_resource_exhausted:
            continue
        is_capacity = any(reason == "MODEL_CAPACITY_EXHAUSTED" for reason in reasons)
        reset_hint_seconds = _parse_reset_hint_seconds(message)
        provider_resets_at = (
            context["observed_at"] + timedelta(seconds=reset_hint_seconds) if reset_hint_seconds is not None else None
        )
        model = metadata_models[0] if metadata_models else context.get("model")
        model_family, model_tier = _infer_model_family_and_tier(model)
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": ("google_model_capacity_error" if is_capacity else "google_generate_content_error"),
                    "provider": "gemini",
                    "client_family": "google_code_assist",
                    "limit_id": ("google_model_capacity" if is_capacity else "google_code_assist_requests"),
                    "limit_name": ("Google model capacity" if is_capacity else "Google Code Assist requests"),
                    "limit_scope": "model_capacity" if is_capacity else "daily_request_pool",
                    "quota_period": None if is_capacity else "daily",
                    "window_minutes": None if is_capacity else 1440,
                    "provider_resets_at": provider_resets_at,
                    "status": ("model_capacity_exhausted" if is_capacity else "quota_exhausted"),
                    "exhausted": not is_capacity,
                    "exhaustion_kind": ("model_capacity" if is_capacity else "request_quota"),
                    "reset_hint_seconds": reset_hint_seconds,
                    "model": model,
                    "model_family": model_family,
                    "model_tier": model_tier,
                    "raw_provider_fields": {
                        "code": code,
                        "status": status_text,
                        "message": message,
                        "reasons": reasons,
                        "metadata_models": metadata_models,
                    },
                    "evidence": {
                        "signals": [
                            "google_resource_exhausted",
                            "model_capacity" if is_capacity else "quota_exhaustion",
                        ],
                        "corroboration_required": is_capacity,
                    },
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _build_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> List[Dict[str, Any]]:
    observed_at = _parse_datetime_value(end_time) or _parse_datetime_value(start_time) or datetime.now(timezone.utc)
    observations: List[Dict[str, Any]] = []
    observations.extend(_extract_codex_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_codex_header_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_codex_usage_limit_error_observations(kwargs, result, observed_at))
    observations.extend(_extract_anthropic_header_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_xai_oauth_header_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_grok_billing_observations(kwargs, result, observed_at))
    observations.extend(_extract_openrouter_free_error_observations(kwargs, result, observed_at))
    observations.extend(_extract_google_quota_observations(kwargs, result, observed_at))
    observations.extend(_extract_google_error_observations(kwargs, result, observed_at))
    return _dedupe_rate_limit_observations(observations)


_HOST_FUNCTION_NAMES = (
    "_openrouter_free_daily_request_limit",
    "_openrouter_free_shared_account_hash",
    "_is_openrouter_free_model",
    "_openrouter_free_daily_window",
    "_openrouter_free_daily_observation_context_from_record",
    "_build_openrouter_free_daily_observation",
    "_openrouter_free_record_observed_at",
    "_is_openrouter_free_session_history_record",
    "_extract_codex_rate_limit_observations",
    "_extract_codex_header_rate_limit_observations",
    "_extract_error_payload_dicts",
    "_extract_codex_usage_limit_error_observations",
    "_rate_limit_header_map",
    "_get_rate_limit_header_value",
    "_looks_like_claude_rate_limit_context",
    "_extract_anthropic_header_rate_limit_observations",
    "_first_quota_number",
    "_first_quota_float",
    "_looks_like_xai_oauth_rate_limit_context",
    "_extract_xai_oauth_account_hash",
    "_xai_oauth_header_remaining_pct",
    "_next_utc_month_start",
    "_is_xai_oauth_subscription_quota_context",
    "_extract_xai_oauth_billing_period_end",
    "_extract_xai_oauth_header_rate_limit_observations",
    "_grok_billing_quota_value",
    "_grok_billing_current_period",
    "_grok_billing_is_weekly_period",
    "_grok_billing_period_bounds",
    "_is_grok_billing_context",
    "_extract_grok_billing_config",
    "_grok_billing_model",
    "_grok_billing_request_contract_evidence",
    "_grok_billing_snapshot_parts",
    "_extract_grok_billing_observations",
    "_extract_openrouter_free_error_reset_at",
    "_extract_openrouter_free_error_observations",
    "_looks_like_google_quota_candidate",
    "_antigravity_quota_pool_for_model",
    "_extract_google_quota_observations",
    "_extract_google_error_observations",
    "_build_rate_limit_observations",
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
