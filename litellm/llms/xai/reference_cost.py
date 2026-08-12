"""Catalog-backed reference pricing for managed xAI routes."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from litellm._logging import verbose_logger

_MANAGED_XAI_REFERENCE_MODELS = frozenset({"oa_xai/grok-4.6"})
_ABOVE_TOKEN_RATE_RE = re.compile(
    r"^input_cost_per_token_(above_(?P<threshold_k>\d+)k_tokens)$"
)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if normalized != normalized or normalized < 0:
        return None
    return normalized


def _load_xai_reference_pricing(
    model: str,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], str, int]]:
    import litellm

    catalog_entry = litellm.model_cost.get(model)
    if not isinstance(catalog_entry, dict):
        return None
    try:
        model_info = dict(
            litellm.get_model_info(
                model=model,
                custom_llm_provider="xai",
            )
        )
    except Exception as exc:
        verbose_logger.debug(
            "XAIPricing: failed to load reference pricing for model=%s: %s",
            model,
            exc,
        )
        return None
    if model_info.get("key") != model:
        return None

    configured_tiers = []
    for field_name in catalog_entry:
        match = _ABOVE_TOKEN_RATE_RE.fullmatch(field_name)
        if match is None:
            continue
        tier_name = match.group(1)
        threshold_tokens = int(match.group("threshold_k")) * 1000
        if all(
            _nonnegative_float(model_info.get(rate_field)) is not None
            for rate_field in (
                f"input_cost_per_token_{tier_name}",
                f"cache_read_input_token_cost_{tier_name}",
                f"output_cost_per_token_{tier_name}",
            )
        ):
            configured_tiers.append((threshold_tokens, tier_name))
    if len(configured_tiers) != 1:
        return None

    threshold_tokens, tier_name = configured_tiers[0]
    required_rates = (
        "input_cost_per_token",
        "cache_read_input_token_cost",
        "output_cost_per_token",
        "input_cost_per_image_token",
    )
    if any(
        _nonnegative_float(model_info.get(rate_field)) is None
        for rate_field in required_rates
    ):
        return None
    return catalog_entry, model_info, tier_name, threshold_tokens


def calculate_xai_reference_cost(
    *,
    model: str,
    prompt_tokens: Any,
    cache_read_input_tokens: Any,
    completion_tokens: Any,
) -> Dict[str, Any]:
    """Calculate reference cost from the exact catalog entry for ``model``."""

    pricing = _load_xai_reference_pricing(model)
    if pricing is None:
        return {}
    _, model_info, configured_tier, threshold_tokens = pricing

    whole_request_tokens = _nonnegative_int(prompt_tokens)
    cached_input_tokens = min(
        _nonnegative_int(cache_read_input_tokens),
        whole_request_tokens,
    )
    uncached_input_tokens = whole_request_tokens - cached_input_tokens
    output_tokens = _nonnegative_int(completion_tokens)
    active_tier = (
        configured_tier if whole_request_tokens > threshold_tokens else "base"
    )
    rate_suffix = "" if active_tier == "base" else f"_{active_tier}"
    cached_rate = _nonnegative_float(
        model_info.get(f"cache_read_input_token_cost{rate_suffix}")
    )
    input_rate = _nonnegative_float(
        model_info.get(f"input_cost_per_token{rate_suffix}")
    )
    output_rate = _nonnegative_float(
        model_info.get(f"output_cost_per_token{rate_suffix}")
    )
    if cached_rate is None or input_rate is None or output_rate is None:
        return {}

    cached_cost = cached_input_tokens * cached_rate
    uncached_cost = uncached_input_tokens * input_rate
    output_cost = output_tokens * output_rate
    return {
        "reference_cost_tier": active_tier,
        "reference_cost_whole_request_tokens": whole_request_tokens,
        "reference_cost_cached_input_tokens": cached_input_tokens,
        "reference_cost_uncached_input_tokens": uncached_input_tokens,
        "reference_cost_output_tokens": output_tokens,
        "reference_cost_cached_input_usd": cached_cost,
        "reference_cost_uncached_input_usd": uncached_cost,
        "reference_cost_output_usd": output_cost,
        "reference_cost_total_usd": cached_cost + uncached_cost + output_cost,
    }


def build_xai_grok_46_reference_cost_metadata(
    *,
    provider: Any,
    model: Any,
    prompt_tokens: Any,
    cache_read_input_tokens: Any,
    completion_tokens: Any,
) -> Dict[str, Any]:
    """Build non-invoice provenance for the managed Grok 4.6 route."""

    normalized_model = str(model or "").strip()
    if (
        str(provider or "").strip().lower() != "xai"
        or normalized_model not in _MANAGED_XAI_REFERENCE_MODELS
    ):
        return {}

    pricing = _load_xai_reference_pricing(normalized_model)
    if pricing is None:
        return {}
    catalog_entry, model_info, configured_tier, threshold_tokens = pricing
    cost_metadata = calculate_xai_reference_cost(
        model=normalized_model,
        prompt_tokens=prompt_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        completion_tokens=completion_tokens,
    )
    if not cost_metadata:
        return {}

    source = catalog_entry.get("source")
    owner = catalog_entry.get("owned_by")
    aliases = catalog_entry.get("aliases")
    created = catalog_entry.get("created")
    verified = catalog_entry.get("verified")
    context_window = model_info.get("max_input_tokens")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(owner, str)
        or not isinstance(aliases, list)
        or not isinstance(created, int)
        or not isinstance(verified, str)
        or not isinstance(context_window, int)
    ):
        return {}

    rate_fields = {
        "reference_cost_input_usd_per_million": "input_cost_per_token",
        "reference_cost_cache_read_input_usd_per_million": (
            "cache_read_input_token_cost"
        ),
        "reference_cost_output_usd_per_million": "output_cost_per_token",
        "reference_cost_image_input_usd_per_million": "input_cost_per_image_token",
        f"reference_cost_input_usd_per_million_{configured_tier}": (
            f"input_cost_per_token_{configured_tier}"
        ),
        f"reference_cost_cache_read_input_usd_per_million_{configured_tier}": (
            f"cache_read_input_token_cost_{configured_tier}"
        ),
        f"reference_cost_output_usd_per_million_{configured_tier}": (
            f"output_cost_per_token_{configured_tier}"
        ),
    }
    rate_metadata = {
        metadata_key: float(model_info[catalog_key]) * 1_000_000
        for metadata_key, catalog_key in rate_fields.items()
    }
    metadata: Dict[str, Any] = {
        "billing_mode": "xai_managed_reference_rate",
        "actual_invoice_cost_known": False,
        "reference_cost_kind": "official_public_api_rate_reference",
        "reference_cost_currency": "USD",
        "reference_cost_model": model_info["key"],
        "reference_cost_source": source,
        "reference_cost_created": created,
        "reference_cost_owner": owner,
        "reference_cost_aliases": list(aliases),
        "reference_cost_verified": verified,
        "reference_cost_context_window_tokens": context_window,
        "reference_cost_threshold_tokens": threshold_tokens,
        **rate_metadata,
        **cost_metadata,
    }
    return metadata


__all__ = [
    "build_xai_grok_46_reference_cost_metadata",
    "calculate_xai_reference_cost",
]
