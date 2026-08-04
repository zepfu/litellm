"""Provider-cache state machine (Wave A3B extraction).

Behavior-preserving extraction from the identity package ``__init__``.
Function bodies resolve free names through the identity host namespace
after :func:`install` rebinds ``__globals__`` (record.py contract), so
module-level imports are intentionally absent here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

from .interfaces import ProviderCacheState

if TYPE_CHECKING:
    def _append_langfuse_span(
        metadata: Dict[str, Any],
        *,
        name: str,
        span_metadata: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
    ) -> None: ...

    def _ensure_mutable_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]: ...

    def _extract_cache_creation_input_tokens(usage_obj: Any) -> int: ...

    def _extract_cache_read_input_tokens(usage_obj: Any) -> int: ...

    def _extract_completion_tokens(usage_obj: Any) -> int: ...

    def _extract_prompt_tokens(usage_obj: Any) -> int: ...

    def _extract_prompt_tokens_details(usage_obj: Any) -> Any: ...

    def _extract_total_tokens(
        usage_obj: Any,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> int: ...

    def _extract_usage_object(kwargs: Dict[str, Any], result: Any) -> Any: ...

    def _first_non_none(*values: Any) -> Any: ...

    def _get_litellm_module() -> Any: ...

    def _has_nested_path(obj: Any, *keys: str) -> bool: ...

    def _lookup_bundled_model_cost_info(
        *,
        model: str,
        custom_llm_provider: Optional[str],
    ) -> Optional[Dict[str, Any]]: ...

    def _maybe_get(obj: Any, key: str) -> Any: ...

    def _maybe_get_path(obj: Any, *keys: str) -> Any: ...

    def _merge_tags(metadata: Dict[str, Any], tags: list[str]) -> None: ...

    def _request_payload_contains(
        payload: Any,
        predicate: Callable[[Dict[str, Any]], bool],
    ) -> bool: ...

    def _resolve_session_history_model(
        *,
        kwargs: Dict[str, Any],
        standard_logging_object: Any,
        metadata: Dict[str, Any],
        result: Any,
    ) -> str: ...

    def _safe_float(value: Any) -> Optional[float]: ...

    def _safe_int(value: Any) -> Optional[int]: ...

_PROVIDER_CACHE_TARGET_FAMILIES = {
    "anthropic",
    "openai",
    "openrouter",
    "opencode_zen",
    "gemini",
    "nvidia",
    "xai",
}

# Inert historical retirement data. aawm_session_history/record.py still
# publishes this constant onto the identity host namespace, so keep it as
# plain data; no cache code consumes it anymore.
_RETIRED_PROVIDER_CACHE_NAMES = frozenset(
    {
        "antigravity",
        "agy",
        "google-antigravity",
        "google_code_assist",
        "google-code-assist",
    }
)


def _normalize_provider_cache_family(
    provider: Any,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    route_family = (metadata or {}).get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if "grok" in route_family_lower or "xai" in route_family_lower:
            return "xai"
        if "nvidia" in route_family_lower:
            return "nvidia"
        if "openrouter" in route_family_lower:
            return "openrouter"
        if "opencode" in route_family_lower:
            return "opencode_zen"
        if "gemini" in route_family_lower or "google" in route_family_lower:
            return "gemini"
        if "anthropic" in route_family_lower:
            return "anthropic"
        if "openai" in route_family_lower or "codex" in route_family_lower:
            return "openai"

    if isinstance(provider, str) and provider.strip():
        provider_lower = provider.strip().lower()
        if provider_lower == "google":
            return "gemini"
        if provider_lower in {"nvidia_nim", "nvidia-nim"}:
            return "nvidia"
        if provider_lower in {"opencode", "opencode-zen", "opencode_zen", "zen"}:
            return "opencode_zen"
        if provider_lower in _PROVIDER_CACHE_TARGET_FAMILIES:
            return provider_lower

    model_lower = str(model or "").strip().lower()
    if model_lower.startswith("nvidia_nim/") or model_lower.startswith("nvidia/"):
        return "nvidia"
    if model_lower.startswith("xai/") or model_lower.startswith("grok"):
        return "xai"
    if model_lower.startswith("openrouter/"):
        return "openrouter"
    if model_lower.startswith(("opencode/", "opencode-zen/", "zen/")):
        return "opencode_zen"
    if "gemini" in model_lower or "gemma" in model_lower or model_lower.startswith("google/"):
        return "gemini"
    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return "anthropic"
    if (
        model_lower.startswith("gpt")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
        or model_lower.startswith("o4")
        or model_lower.startswith("openai/")
        or "codex" in model_lower
    ):
        return "openai"
    return None


def _supports_prompt_caching_safe(
    *,
    model: str,
    provider: Optional[str],
) -> Optional[bool]:
    normalized_model = str(model or "").strip()
    if not normalized_model or normalized_model.lower() == "unknown":
        return None
    try:
        litellm = _get_litellm_module()
        return bool(
            litellm.supports_prompt_caching(
                model=normalized_model,
                custom_llm_provider=provider,
            )
        )
    except Exception:
        return None


def _extract_provider_cache_request_body(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = (
        _maybe_get(kwargs.get("passthrough_logging_payload"), "request_body"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body"),
        kwargs.get("request_body"),
        _maybe_get(kwargs.get("standard_logging_object"), "request_body"),
        _maybe_get_path(kwargs.get("standard_logging_object"), "request", "body"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return None


def _request_contains_cache_control(payload: Any) -> bool:
    return _request_payload_contains(
        payload,
        lambda item: item.get("cache_control") is not None or item.get("cacheControl") is not None,
    )


def _request_contains_cached_content(payload: Any) -> bool:
    def _has_cached_content(item: Dict[str, Any]) -> bool:
        cached_content = item.get("cachedContent")
        if isinstance(cached_content, str) and cached_content.strip():
            return True
        cached_content_alias = item.get("cached_content")
        return isinstance(cached_content_alias, str) and bool(cached_content_alias.strip())

    return _request_payload_contains(payload, _has_cached_content)


def _request_contains_prompt_cache_key(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    prompt_cache_key = payload.get("prompt_cache_key")
    return isinstance(prompt_cache_key, str) and bool(prompt_cache_key.strip())


def _openai_style_cached_tokens_source(usage_obj: Any) -> Optional[str]:
    for path, source in (
        (
            ("prompt_tokens_details", "cached_tokens"),
            "usage.prompt_tokens_details.cached_tokens",
        ),
        (
            ("prompt_tokens_details", "cachedTokens"),
            "usage.prompt_tokens_details.cachedTokens",
        ),
        (
            ("input_tokens_details", "cached_tokens"),
            "usage.input_tokens_details.cached_tokens",
        ),
        (
            ("input_tokens_details", "cachedTokens"),
            "usage.input_tokens_details.cachedTokens",
        ),
        (
            ("promptTokensDetails", "cached_tokens"),
            "usage.promptTokensDetails.cached_tokens",
        ),
        (
            ("promptTokensDetails", "cachedTokens"),
            "usage.promptTokensDetails.cachedTokens",
        ),
        (
            ("inputTokensDetails", "cached_tokens"),
            "usage.inputTokensDetails.cached_tokens",
        ),
        (
            ("inputTokensDetails", "cachedTokens"),
            "usage.inputTokensDetails.cachedTokens",
        ),
    ):
        if _has_nested_path(usage_obj, *path):
            return source
    return None


def _usage_has_openai_style_cached_tokens_field(usage_obj: Any) -> bool:
    return _openai_style_cached_tokens_source(usage_obj) is not None


def _usage_has_gemini_style_cached_content_field(usage_obj: Any) -> bool:
    return _has_nested_path(usage_obj, "cachedContentTokenCount")


def _openai_cache_attempt_source(usage_obj: Any, request_body: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    if _request_contains_prompt_cache_key(request_body):
        return "prompt_cache_key_requested_without_hit", "request.prompt_cache_key"
    cached_tokens_source = _openai_style_cached_tokens_source(usage_obj)
    if cached_tokens_source is not None:
        return "cached_tokens_reported_zero", cached_tokens_source
    return None


def _extract_service_tier_hint(
    usage_obj: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = metadata or {}
    for candidate in (
        _maybe_get(usage_obj, "service_tier"),
        _maybe_get(usage_obj, "serviceTier"),
        metadata.get("service_tier"),
        metadata.get("serviceTier"),
        metadata.get("openai_service_tier"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _price_cache_miss(
    state: Mapping[str, Any],
    cost_map: Mapping[str, Any],
) -> float:
    """Pure pricing: compute cache-miss cost from token count and per-token pricing.

    ``state`` must carry ``miss_token_count``.  ``cost_map`` carries
    ``input_cost_per_token`` and optionally ``cache_read_input_token_cost``.
    When ``cache_read_input_token_cost`` is absent the full
    ``input_cost_per_token`` is used as the per-token price.
    """
    miss_token_count_value = state.get("miss_token_count")
    if miss_token_count_value is None:
        return 0.0
    try:
        miss_token_count = float(miss_token_count_value)
    except (TypeError, ValueError):
        return 0.0
    if miss_token_count <= 0:
        return 0.0

    input_cost_total = cost_map.get("input_cost_total")
    if input_cost_total is not None:
        cache_read_input_total = cost_map.get("cache_read_input_total", 0.0)
        if cache_read_input_total is None:
            cache_read_input_total = 0.0
        try:
            return max(
                float(input_cost_total) - float(cache_read_input_total),
                0.0,
            )
        except (TypeError, ValueError):
            return 0.0

    input_cost_per_token = cost_map.get("input_cost_per_token")
    if input_cost_per_token is None:
        return 0.0
    cache_read_cost_per_token = cost_map.get(
        "cache_read_input_token_cost",
        0.0,
    )
    if cache_read_cost_per_token is None:
        cache_read_cost_per_token = 0.0
    try:
        delta = float(input_cost_per_token) - float(cache_read_cost_per_token)
    except (TypeError, ValueError):
        return 0.0
    return max(delta * miss_token_count, 0.0)


def _determine_cache_attempt(
    *,
    provider_family: str,
    usage_obj: Any,
    request_body: Optional[Dict[str, Any]],
    state_from_metadata: Optional[ProviderCacheState],
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Determine whether provider caching was attempted.

    Returns ``(attempted, miss_reason, source)``.
    """
    attempted = False
    miss_reason: Optional[str] = None
    source: Optional[str] = None

    if state_from_metadata is not None and state_from_metadata.attempted:
        attempted = True
        miss_reason = (
            state_from_metadata.miss_reason or "cache_attempted_without_hit"
        )
        source = (
            state_from_metadata.source or "metadata.provider_cache_attempted"
        )

    request_has_cache_control = _request_contains_cache_control(request_body)
    request_has_cached_content = _request_contains_cached_content(request_body)
    usage_has_openai_cached_tokens = _openai_style_cached_tokens_source(usage_obj) is not None
    usage_has_gemini_cached_content = _usage_has_gemini_style_cached_content_field(usage_obj)

    if provider_family == "anthropic":
        attempted = attempted or request_has_cache_control
        if attempted:
            miss_reason = miss_reason or "cache_control_requested_without_hit"
            source = source or "request.cache_control"
    elif provider_family == "openrouter":
        if request_has_cache_control:
            attempted = True
            miss_reason = miss_reason or "cache_control_requested_without_hit"
            source = source or "request.cache_control"
        elif usage_has_openai_cached_tokens:
            attempted = True
            miss_reason = miss_reason or "cached_tokens_reported_zero"
            source = source or _openai_style_cached_tokens_source(usage_obj)
    elif provider_family == "gemini":
        if request_has_cached_content:
            attempted = True
            miss_reason = miss_reason or "cached_content_requested_without_hit"
            source = source or "request.cached_content"
        elif usage_has_gemini_cached_content:
            attempted = True
            miss_reason = miss_reason or "cached_tokens_reported_zero"
            source = source or "usage.cached_content_token_count"
    elif provider_family == "openai":
        openai_cache_attempt_source = _openai_cache_attempt_source(usage_obj, request_body)
        if openai_cache_attempt_source:
            attempted = True
            source_miss_reason, source_name = openai_cache_attempt_source
            miss_reason = miss_reason or source_miss_reason
            source = source or source_name

    return attempted, miss_reason, source


def _determine_cache_outcome(
    *,
    provider_family: str,
    usage_obj: Any,
    request_body: Optional[Dict[str, Any]],
    state_from_metadata: Optional[ProviderCacheState],
    supports_prompt_caching: Optional[bool],
) -> ProviderCacheState:
    """Determine the complete provider-cache outcome."""
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    prompt_tokens = _extract_prompt_tokens(usage_obj)

    if (
        provider_family == "xai"
        and state_from_metadata is not None
        and state_from_metadata.status == "hit"
        and cache_read_input_tokens > 0
        and prompt_tokens > cache_read_input_tokens
    ):
        return ProviderCacheState(
            attempted=True,
            status="hit",
            miss=True,
            miss_reason="partial_cache_hit",
            miss_token_count=prompt_tokens - cache_read_input_tokens,
            source=(
                state_from_metadata.source
                or "usage.cache_read_input_tokens"
            ),
            supports_prompt_caching=state_from_metadata.supports_prompt_caching,
            record_fields=frozenset(
                {
                    "attempted",
                    "status",
                    "miss",
                    "miss_reason",
                    "miss_token_count",
                    "source",
                    "supports_prompt_caching",
                }
            ),
        )

    if state_from_metadata is not None and state_from_metadata.status is not None:
        return state_from_metadata

    common_record_fields = frozenset(
        {
            "attempted",
            "status",
            "miss",
            "miss_reason",
            "source",
            "supports_prompt_caching",
        }
    )
    if cache_read_input_tokens > 0:
        partial_hit = (
            provider_family == "xai"
            and prompt_tokens > cache_read_input_tokens
        )
        return ProviderCacheState(
            attempted=True,
            status="hit",
            miss=partial_hit,
            miss_reason="partial_cache_hit" if partial_hit else None,
            miss_token_count=(
                prompt_tokens - cache_read_input_tokens
                if partial_hit
                else None
            ),
            source="usage.cache_read_input_tokens",
            supports_prompt_caching=supports_prompt_caching,
            record_fields=(
                common_record_fields.union({"miss_token_count"})
                if partial_hit
                else common_record_fields
            ),
        )

    if cache_creation_input_tokens > 0:
        return ProviderCacheState(
            attempted=True,
            status="write",
            miss=True,
            miss_reason="cache_write_only",
            source="usage.cache_creation_input_tokens",
            supports_prompt_caching=supports_prompt_caching,
            record_fields=common_record_fields,
        )

    attempted, miss_reason, source = _determine_cache_attempt(
        provider_family=provider_family,
        usage_obj=usage_obj,
        request_body=request_body,
        state_from_metadata=state_from_metadata,
    )

    if not attempted:
        return ProviderCacheState(
            attempted=False,
            status="not_attempted",
            miss=False,
            miss_reason=None,
            source=None,
            supports_prompt_caching=supports_prompt_caching,
            record_fields=common_record_fields,
        )

    if supports_prompt_caching is False and source and source.startswith("request."):
        return ProviderCacheState(
            attempted=True,
            status="unsupported",
            miss=False,
            miss_reason=None,
            source=source,
            supports_prompt_caching=supports_prompt_caching,
            record_fields=common_record_fields,
        )

    return ProviderCacheState(
        attempted=True,
        status="miss",
        miss=True,
        miss_reason=miss_reason,
        source=source,
        supports_prompt_caching=supports_prompt_caching,
        record_fields=common_record_fields,
    )


def _compute_provider_cache_miss_cost_state(  # noqa: PLR0915
    *,
    provider_family: Optional[str],
    model: str,
    usage_obj: Any,
    cache_state: Optional[
        Union[ProviderCacheState, Mapping[str, Any]]
    ],
    metadata: Optional[Dict[str, Any]] = None,
    response_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    cache_state_mapping = (
        cache_state.to_record_dict()
        if isinstance(cache_state, ProviderCacheState)
        else cache_state
    )
    existing_miss_token_count = (
        _safe_int(cache_state_mapping.get("miss_token_count"))
        if cache_state_mapping is not None
        else None
    )
    existing_miss_cost_usd = (
        _safe_float(cache_state_mapping.get("miss_cost_usd"))
        if cache_state_mapping is not None
        else None
    )
    existing_miss_cost_basis = (
        str(cache_state_mapping.get("miss_cost_basis")).strip()
        if cache_state_mapping is not None
        and cache_state_mapping.get("miss_cost_basis") is not None
        and str(cache_state_mapping.get("miss_cost_basis")).strip()
        else None
    )
    result: Dict[str, Any] = {
        "miss_token_count": existing_miss_token_count,
        "miss_cost_usd": existing_miss_cost_usd,
        "miss_cost_basis": existing_miss_cost_basis,
    }
    if provider_family is None or cache_state_mapping is None:
        return result

    cost_provider_family = "nvidia_nim" if provider_family == "nvidia" else provider_family
    cache_status = cache_state_mapping.get("status")
    cache_missed = bool(cache_state_mapping.get("miss"))
    cache_miss_reason = cache_state_mapping.get("miss_reason")
    service_tier = _extract_service_tier_hint(usage_obj, metadata)

    def _fallback_miss_cost(
        miss_token_count: int,
    ) -> Tuple[Optional[float], Optional[str]]:
        model_info = _lookup_bundled_model_cost_info(
            model=model,
            custom_llm_provider=cost_provider_family,
        )
        input_cost_per_token = (
            _safe_float(model_info.get("input_cost_per_token")) if isinstance(model_info, dict) else None
        )
        if input_cost_per_token is not None:
            return (
                _price_cache_miss(
                    {"miss_token_count": miss_token_count},
                    {"input_cost_per_token": input_cost_per_token},
                ),
                "prompt_input_cost_no_cache_read_pricing",
            )

        response_cost = _safe_float(
            _first_non_none(
                response_cost_usd,
                _maybe_get(usage_obj, "cost"),
                _maybe_get(usage_obj, "response_cost"),
                _maybe_get(usage_obj, "responseCost"),
                (metadata or {}).get("litellm_response_cost"),
                (metadata or {}).get("response_cost"),
                (metadata or {}).get("usage_openrouter_cost"),
            )
        )
        if response_cost is None or response_cost < 0:
            return None, None
        if response_cost == 0:
            return (
                _price_cache_miss(
                    {"miss_token_count": miss_token_count},
                    {"input_cost_total": 0.0},
                ),
                "response_cost_zero",
            )

        prompt_tokens = _extract_prompt_tokens(usage_obj)
        completion_tokens = _extract_completion_tokens(usage_obj)
        total_tokens = _extract_total_tokens(
            usage_obj,
            prompt_tokens,
            completion_tokens,
        )
        if total_tokens > 0:
            estimated_cost = float(response_cost) * min(
                float(miss_token_count) / float(total_tokens),
                1.0,
            )
            return (
                _price_cache_miss(
                    {"miss_token_count": miss_token_count},
                    {"input_cost_total": estimated_cost},
                ),
                "response_cost_token_share_estimate",
            )
        return (
            _price_cache_miss(
                {"miss_token_count": miss_token_count},
                {"input_cost_total": response_cost},
            ),
            "response_cost_estimate",
        )

    def _populate_prompt_vs_cache_read_delta_cost(miss_token_count: int) -> Dict[str, Any]:
        if result["miss_cost_usd"] is not None:
            return result
        try:
            from litellm.litellm_core_utils.llm_cost_calc.utils import (
                _get_token_base_cost,
            )
            from litellm.types.utils import ModelInfo, Usage
            from litellm.utils import get_model_info

            usage_for_cost = Usage(
                prompt_tokens=miss_token_count,
                completion_tokens=0,
                total_tokens=miss_token_count,
            )
            try:
                model_info: Any = get_model_info(
                    model=model,
                    custom_llm_provider=cost_provider_family,
                )
            except Exception:
                model_info = _lookup_bundled_model_cost_info(
                    model=model,
                    custom_llm_provider=cost_provider_family,
                )
            if not isinstance(model_info, dict):
                fallback_cost, fallback_basis = _fallback_miss_cost(miss_token_count)
                if fallback_cost is not None:
                    result["miss_cost_usd"] = fallback_cost
                    result["miss_cost_basis"] = fallback_basis
                return result
            if "cache_read_input_token_cost" not in model_info:
                fallback_cost, fallback_basis = _fallback_miss_cost(miss_token_count)
                if fallback_cost is not None:
                    result["miss_cost_usd"] = fallback_cost
                    result["miss_cost_basis"] = fallback_basis
                return result
            typed_model_info = cast(ModelInfo, model_info)
            (
                prompt_base_cost,
                _completion_base_cost,
                _cache_creation_cost,
                _cache_creation_cost_above_1hr,
                cache_read_cost,
            ) = _get_token_base_cost(
                model_info=typed_model_info,
                usage=usage_for_cost,
                service_tier=service_tier,
            )

            if prompt_base_cost is None or cache_read_cost is None:
                return result

            miss_cost = _price_cache_miss(
                {"miss_token_count": miss_token_count},
                {
                    "input_cost_per_token": prompt_base_cost,
                    "cache_read_input_token_cost": cache_read_cost,
                },
            )
            result["miss_cost_usd"] = miss_cost
            result["miss_cost_basis"] = "prompt_vs_cache_read_delta"
            return result
        except Exception:
            fallback_cost, fallback_basis = _fallback_miss_cost(miss_token_count)
            if fallback_cost is not None:
                result["miss_cost_usd"] = fallback_cost
                result["miss_cost_basis"] = fallback_basis
            return result

    if cache_status == "hit" and cache_missed and cache_miss_reason == "partial_cache_hit":
        miss_token_count = (
            existing_miss_token_count
            if existing_miss_token_count is not None and existing_miss_token_count > 0
            else None
        )
        if miss_token_count is None:
            prompt_tokens = _extract_prompt_tokens(usage_obj)
            cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
            if prompt_tokens > cache_read_input_tokens > 0:
                miss_token_count = prompt_tokens - cache_read_input_tokens
        if miss_token_count is None or miss_token_count <= 0:
            return result
        result["miss_token_count"] = miss_token_count
        return _populate_prompt_vs_cache_read_delta_cost(miss_token_count)

    if cache_status == "miss" and cache_missed:
        miss_token_count = _extract_prompt_tokens(usage_obj)
        if miss_token_count <= 0:
            if existing_miss_token_count is not None and existing_miss_token_count > 0:
                result["miss_token_count"] = existing_miss_token_count
                if result["miss_cost_usd"] is not None:
                    return result
                fallback_cost, fallback_basis = _fallback_miss_cost(existing_miss_token_count)
                if fallback_cost is not None:
                    result["miss_cost_usd"] = fallback_cost
                    result["miss_cost_basis"] = fallback_basis
                return result
            result["miss_token_count"] = 0
            fallback_cost, fallback_basis = _fallback_miss_cost(0)
            if fallback_cost is not None:
                result["miss_cost_usd"] = fallback_cost
                result["miss_cost_basis"] = fallback_basis
            return result

        result["miss_token_count"] = miss_token_count
        return _populate_prompt_vs_cache_read_delta_cost(miss_token_count)

    if cache_status != "write":
        return result

    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    if cache_creation_input_tokens <= 0:
        return result

    result["miss_token_count"] = cache_creation_input_tokens
    prompt_tokens = max(_extract_prompt_tokens(usage_obj), cache_creation_input_tokens)

    try:
        from litellm.litellm_core_utils.llm_cost_calc.utils import (
            _get_token_base_cost,
            calculate_cache_writing_cost,
        )
        from litellm.types.utils import CacheCreationTokenDetails, Usage
        from litellm.utils import get_model_info

        prompt_tokens_details = _extract_prompt_tokens_details(usage_obj)
        cache_creation_token_details = None
        if isinstance(prompt_tokens_details, dict):
            detail_5m = _safe_int(_maybe_get(prompt_tokens_details, "ephemeral_5m_input_tokens"))
            detail_1h = _safe_int(_maybe_get(prompt_tokens_details, "ephemeral_1h_input_tokens"))
            if detail_5m is not None or detail_1h is not None:
                cache_creation_token_details = CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=detail_5m,
                    ephemeral_1h_input_tokens=detail_1h,
                )

        usage_for_cost = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        )
        model_info = get_model_info(
            model=model,
            custom_llm_provider=cost_provider_family,
        )
        (
            _prompt_base_cost,
            _completion_base_cost,
            cache_creation_cost,
            cache_creation_cost_above_1hr,
            cache_read_cost,
        ) = _get_token_base_cost(
            model_info=model_info,
            usage=usage_for_cost,
            service_tier=service_tier,
        )

        write_cost = calculate_cache_writing_cost(
            cache_creation_tokens=cache_creation_input_tokens,
            cache_creation_token_details=cache_creation_token_details,
            cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
            cache_creation_cost=cache_creation_cost,
        )
        read_cost = float(cache_creation_input_tokens) * float(cache_read_cost or 0.0)
        miss_cost = _price_cache_miss(
            {"miss_token_count": cache_creation_input_tokens},
            {
                "input_cost_total": write_cost,
                "cache_read_input_total": read_cost,
            },
        )
        result["miss_cost_usd"] = miss_cost
        result["miss_cost_basis"] = "write_vs_read_delta"
        return result
    except Exception:
        return result


def _provider_cache_state_from_metadata(
    metadata: Dict[str, Any],
    provider_family: Optional[str],
) -> Optional[ProviderCacheState]:
    status = metadata.get("usage_provider_cache_status")
    if status is None and provider_family:
        status = metadata.get(f"{provider_family}_provider_cache_status")
    attempted = metadata.get("usage_provider_cache_attempted")
    if attempted is None and provider_family:
        attempted = metadata.get(f"{provider_family}_provider_cache_attempted")
    miss = metadata.get("usage_provider_cache_miss")
    if miss is None and provider_family:
        miss = metadata.get(f"{provider_family}_provider_cache_miss")
    miss_reason = metadata.get("usage_provider_cache_miss_reason")
    if miss_reason is None and provider_family:
        miss_reason = metadata.get(f"{provider_family}_provider_cache_miss_reason")
    miss_token_count = metadata.get("usage_provider_cache_miss_token_count")
    if miss_token_count is None and provider_family:
        miss_token_count = metadata.get(f"{provider_family}_provider_cache_miss_token_count")
    miss_cost_usd = metadata.get("usage_provider_cache_miss_cost_usd")
    if miss_cost_usd is None and provider_family:
        miss_cost_usd = metadata.get(f"{provider_family}_provider_cache_miss_cost_usd")
    miss_cost_basis = metadata.get("usage_provider_cache_miss_cost_basis")
    if miss_cost_basis is None and provider_family:
        miss_cost_basis = metadata.get(f"{provider_family}_provider_cache_miss_cost_basis")
    source = metadata.get("usage_provider_cache_source")
    if source is None and provider_family:
        source = metadata.get(f"{provider_family}_provider_cache_source")
    if (
        status is None
        and attempted is None
        and miss is None
        and miss_reason is None
        and miss_token_count is None
        and miss_cost_usd is None
        and miss_cost_basis is None
        and source is None
    ):
        return None
    normalized_status = str(status).strip() if isinstance(status, str) and status.strip() else None
    return ProviderCacheState(
        attempted=(
            bool(attempted)
            if attempted is not None
            else bool(normalized_status and normalized_status != "not_attempted")
        ),
        status=normalized_status,
        miss=(
            bool(miss)
            if miss is not None
            else normalized_status in {"miss", "write"}
        ),
        miss_reason=(
            str(miss_reason).strip() if isinstance(miss_reason, str) and str(miss_reason).strip() else None
        ),
        miss_token_count=_safe_int(miss_token_count),
        miss_cost_usd=_safe_float(miss_cost_usd),
        miss_cost_basis=(
            str(miss_cost_basis).strip() if isinstance(miss_cost_basis, str) and str(miss_cost_basis).strip() else None
        ),
        source=(
            str(source).strip()
            if isinstance(source, str) and str(source).strip()
            else None
        ),
        record_fields=frozenset(
            {
                "attempted",
                "status",
                "miss",
                "miss_reason",
                "miss_token_count",
                "miss_cost_usd",
                "miss_cost_basis",
                "source",
            }
        ),
    )


def _resolve_provider_cache_state(  # noqa: PLR0915
    *,
    provider: Any,
    model: str,
    usage_obj: Any,
    metadata: Optional[Dict[str, Any]] = None,
    request_body: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    metadata = metadata or {}
    provider_family = _normalize_provider_cache_family(provider, model, metadata)
    if provider_family is None:
        return None

    state_from_metadata = _provider_cache_state_from_metadata(metadata, provider_family)
    supports_prompt_caching = (
        None
        if state_from_metadata is not None
        and state_from_metadata.status is not None
        else _supports_prompt_caching_safe(
            model=model,
            provider=provider_family,
        )
    )
    return _determine_cache_outcome(
        provider_family=provider_family,
        usage_obj=usage_obj,
        request_body=request_body,
        state_from_metadata=state_from_metadata,
        supports_prompt_caching=supports_prompt_caching,
    ).to_record_dict()


def _enrich_provider_cache_metadata(kwargs: Dict[str, Any], result: Any) -> None:  # noqa: PLR0915
    metadata = _ensure_mutable_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    resolved_model = _resolve_session_history_model(
        kwargs=kwargs,
        standard_logging_object=standard_logging_object,
        metadata=metadata,
        result=result,
    )
    usage_obj = _extract_usage_object(kwargs, result)
    request_body = _extract_provider_cache_request_body(kwargs)
    response_cost_usd = _safe_float(
        _first_non_none(
            kwargs.get("response_cost"),
            (kwargs.get("standard_logging_object") or {}).get("response_cost"),
            metadata.get("litellm_response_cost"),
            metadata.get("response_cost"),
            metadata.get("usage_openrouter_cost"),
            _maybe_get(usage_obj, "cost"),
        )
    )
    provider_family = _normalize_provider_cache_family(
        kwargs.get("custom_llm_provider"),
        resolved_model,
        metadata,
    )
    resolved_cache_state = _resolve_provider_cache_state(
        provider=kwargs.get("custom_llm_provider"),
        model=resolved_model,
        usage_obj=usage_obj,
        metadata=metadata,
        request_body=request_body,
    )
    if provider_family is None or resolved_cache_state is None:
        return
    cache_state = ProviderCacheState.from_record_dict(resolved_cache_state)
    cache_miss_cost_state = _compute_provider_cache_miss_cost_state(
        provider_family=provider_family,
        model=resolved_model,
        usage_obj=usage_obj,
        cache_state=cache_state,
        metadata=metadata,
        response_cost_usd=response_cost_usd,
    )
    cache_state = cache_state.with_cost_state(cache_miss_cost_state)

    metadata["usage_provider_cache_attempted"] = cache_state.attempted
    metadata["usage_provider_cache_status"] = cache_state.status
    metadata["usage_provider_cache_miss"] = cache_state.miss
    if cache_state.miss_reason:
        metadata["usage_provider_cache_miss_reason"] = cache_state.miss_reason
    if cache_state.miss_token_count is not None:
        metadata["usage_provider_cache_miss_token_count"] = cache_state.miss_token_count
    if cache_state.miss_cost_usd is not None:
        metadata["usage_provider_cache_miss_cost_usd"] = cache_state.miss_cost_usd
    if cache_state.miss_cost_basis:
        metadata["usage_provider_cache_miss_cost_basis"] = cache_state.miss_cost_basis
    if cache_state.source:
        metadata["usage_provider_cache_source"] = cache_state.source

    metadata[f"{provider_family}_provider_cache_attempted"] = cache_state.attempted
    metadata[f"{provider_family}_provider_cache_status"] = cache_state.status
    metadata[f"{provider_family}_provider_cache_miss"] = cache_state.miss
    if cache_state.miss_reason:
        metadata[f"{provider_family}_provider_cache_miss_reason"] = cache_state.miss_reason
    if cache_state.miss_token_count is not None:
        metadata[f"{provider_family}_provider_cache_miss_token_count"] = cache_state.miss_token_count
    if cache_state.miss_cost_usd is not None:
        metadata[f"{provider_family}_provider_cache_miss_cost_usd"] = cache_state.miss_cost_usd
    if cache_state.miss_cost_basis:
        metadata[f"{provider_family}_provider_cache_miss_cost_basis"] = cache_state.miss_cost_basis
    if cache_state.source:
        metadata[f"{provider_family}_provider_cache_source"] = cache_state.source

    tags_to_add = []
    status = cache_state.status
    if isinstance(status, str) and status in {"hit", "write", "miss", "unsupported"}:
        tags_to_add.extend(
            [
                f"provider-cache-status:{status}",
                f"{provider_family}-provider-cache-status:{status}",
            ]
        )
    if cache_state.miss:
        tags_to_add.extend(
            [
                "provider-cache-miss",
                f"{provider_family}-provider-cache-miss",
            ]
        )
    if status == "hit":
        tags_to_add.extend(["provider-cache-hit", f"{provider_family}-provider-cache-hit"])
    elif status == "write":
        tags_to_add.extend(["provider-cache-write", f"{provider_family}-provider-cache-write"])
    elif status == "unsupported":
        tags_to_add.extend(["provider-cache-unsupported", f"{provider_family}-provider-cache-unsupported"])
    if cache_state.miss_reason == "partial_cache_hit":
        tags_to_add.extend(["provider-cache-partial-hit", f"{provider_family}-provider-cache-partial-hit"])
    if tags_to_add:
        _merge_tags(metadata, tags_to_add)
        _append_langfuse_span(
            metadata,
            name=f"{provider_family}.provider_cache",
            span_metadata={
                "attempted": cache_state.attempted,
                "status": cache_state.status,
                "miss": cache_state.miss,
                "miss_reason": cache_state.miss_reason,
                "miss_token_count": cache_state.miss_token_count,
                "miss_cost_usd": cache_state.miss_cost_usd,
                "miss_cost_basis": cache_state.miss_cost_basis,
                "source": cache_state.source,
            },
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )



_HOST_FUNCTION_NAMES = (
    "_normalize_provider_cache_family",
    "_supports_prompt_caching_safe",
    "_extract_provider_cache_request_body",
    "_request_contains_cache_control",
    "_request_contains_cached_content",
    "_request_contains_prompt_cache_key",
    "_openai_style_cached_tokens_source",
    "_usage_has_openai_style_cached_tokens_field",
    "_usage_has_gemini_style_cached_content_field",
    "_openai_cache_attempt_source",
    "_extract_service_tier_hint",
    "_price_cache_miss",
    "_determine_cache_attempt",
    "_determine_cache_outcome",
    "_compute_provider_cache_miss_cost_state",
    "_provider_cache_state_from_metadata",
    "_resolve_provider_cache_state",
    "_enrich_provider_cache_metadata",
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
