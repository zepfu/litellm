"""Reference-cost helpers for managed Kimi Code usage."""

from __future__ import annotations

from typing import Any, Dict, Optional

from litellm._logging import verbose_logger


def _normalize_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_non_negative_int(value: object) -> int:
    value_int = _normalize_int(value)
    if value_int is None:
        return 0
    return max(value_int, 0)


def _normalize_non_negative_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if float_value != float_value:
        return None
    return max(float_value, 0.0)


def build_kimi_code_reference_cost_metadata(
    *,
    provider: Optional[str],
    model: str,
    prompt_tokens: int,
    cache_read_input_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    """
    Build non-invoice reference-cost metadata for managed Kimi Code usage.

    For model aliases `k3-low`, `k3-high`, and `k3-max`, upstream pricing uses
    the `k3` route while preserving the selected variant in metadata.
    """
    if str(provider or "").strip().lower() != "kimi_code":
        return {}

    normalized_model = str(model or "").strip()
    if normalized_model.lower().startswith("kimi_code/"):
        normalized_model = normalized_model.split("/", 1)[1]
    if not normalized_model or normalized_model == "unknown":
        return {}

    pricing_model = (
        "k3"
        if normalized_model in {"k3-low", "k3-high", "k3-max"}
        else normalized_model
    )
    try:
        import litellm

        model_info = litellm.get_model_info(
            model=pricing_model,
            custom_llm_provider="kimi_code",
        )
        model_cost_entry = litellm.model_cost.get(
            f"kimi_code/{normalized_model}"
        ) or litellm.model_cost.get(f"kimi_code/{pricing_model}")
        source = None
        if isinstance(model_cost_entry, dict):
            source = model_cost_entry.get("source")
        if not isinstance(source, str) or not source.strip():
            return {}

        cache_read_rate = _normalize_non_negative_float(
            model_info.get("cache_read_input_token_cost")
        )
        input_rate = _normalize_non_negative_float(
            model_info.get("input_cost_per_token")
        )
        output_rate = _normalize_non_negative_float(
            model_info.get("output_cost_per_token")
        )
        if cache_read_rate is None or input_rate is None or output_rate is None:
            return {}

        normalized_prompt_tokens = _normalize_non_negative_int(prompt_tokens)
        normalized_cached_tokens = _normalize_non_negative_int(cache_read_input_tokens)
        normalized_cached_tokens = min(
            normalized_cached_tokens, normalized_prompt_tokens
        )
        normalized_completion_tokens = _normalize_non_negative_int(completion_tokens)
        uncached_input_tokens = normalized_prompt_tokens - normalized_cached_tokens

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=pricing_model,
            prompt_tokens=normalized_prompt_tokens,
            completion_tokens=normalized_completion_tokens,
            cache_read_input_tokens=normalized_cached_tokens,
            custom_llm_provider="kimi_code",
        )
    except Exception as exc:
        verbose_logger.debug(
            "KimiCodePricing: failed to calculate reference cost for model=%s: %s",
            normalized_model,
            exc,
        )
        return {}

    return {
        "billing_mode": "kimi_code_subscription",
        "actual_invoice_cost_known": False,
        "reference_cost_kind": "official_public_subscription_route",
        "reference_cost_currency": "USD",
        "reference_cost_model": f"kimi_code/{normalized_model}",
        "reference_cost_source": source,
        "reference_cost_cached_input_usd": normalized_cached_tokens
        * cache_read_rate,
        "reference_cost_uncached_input_usd": uncached_input_tokens * input_rate,
        "reference_cost_output_usd": normalized_completion_tokens * output_rate,
        "reference_cost_total_usd": prompt_cost + completion_cost,
    }


__all__ = ["build_kimi_code_reference_cost_metadata"]
