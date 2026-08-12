"""Bundled model-cost-map lookup helpers for AAWM identity.

Behavior-preserving Wave A2 extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so most module-level imports are intentionally absent here.
"""

# ruff: noqa: F821 - free names resolve via host globals after rebind
from __future__ import annotations

from functools import lru_cache  # noqa: F401 - decorator for module-local caches
from litellm._logging import verbose_logger  # noqa: F401 - referenced by module-local caches


@lru_cache(maxsize=1)
def _load_bundled_model_cost_map() -> Dict[str, Any]:
    try:
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

        return GetModelCostMap.load_local_model_cost_map()
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to load bundled model cost map: %s",
            exc,
        )
        return {}


@lru_cache(maxsize=1)
def _bundled_model_cost_casefold_lookup() -> Dict[str, str]:
    return {key.lower(): key for key in _load_bundled_model_cost_map() if isinstance(key, str)}


def _lookup_bundled_model_cost_info(
    *,
    model: str,
    custom_llm_provider: Optional[str],
) -> Optional[Dict[str, Any]]:
    model_cost = _load_bundled_model_cost_map()
    if not model_cost:
        return None

    # Prefer provider-qualified keys when an explicit provider is supplied so
    # ambiguous bare model names cannot win over the intended provider entry.
    candidates: List[str] = []
    if custom_llm_provider:
        provider_prefix = f"{custom_llm_provider}/"
        if model.startswith(provider_prefix):
            candidates.append(model)
            stripped_model = model[len(provider_prefix) :]
            if stripped_model:
                candidates.append(stripped_model)
        else:
            candidates.append(f"{provider_prefix}{model}")
            candidates.append(model)
    else:
        candidates.append(model)

    lookup = _bundled_model_cost_casefold_lookup()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        if candidate in model_cost and isinstance(model_cost[candidate], dict):
            return model_cost[candidate]
        matched_key = lookup.get(candidate.lower())
        if matched_key is not None and isinstance(model_cost.get(matched_key), dict):
            return model_cost[matched_key]

    return None


def _calculate_response_cost_from_bundled_model_cost_map(
    *,
    model: str,
    custom_llm_provider: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    usage_obj: Any,
) -> Optional[float]:
    model_info = _lookup_bundled_model_cost_info(
        model=model,
        custom_llm_provider=custom_llm_provider,
    )
    if not model_info:
        return None

    search_units = _safe_int(_maybe_get(usage_obj, "search_units"))
    input_cost_per_query = _safe_float(model_info.get("input_cost_per_query"))
    if search_units and input_cost_per_query is not None and input_cost_per_query > 0:
        return search_units * input_cost_per_query

    has_token_pricing = "input_cost_per_token" in model_info or "output_cost_per_token" in model_info
    if not has_token_pricing:
        return None

    input_cost_per_token = _safe_float(model_info.get("input_cost_per_token")) or 0.0
    output_cost_per_token = _safe_float(model_info.get("output_cost_per_token")) or 0.0
    return (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)


def _extract_aawm_cached_input_tokens(usage_obj: Any) -> Optional[int]:
    for key in (
        "cache_read_input_tokens",
        "cached_input_tokens",
        "cached_tokens",
    ):
        value = _safe_int(_maybe_get(usage_obj, key))
        if value is not None and value >= 0:
            return value

    prompt_tokens_details = _maybe_get(usage_obj, "prompt_tokens_details")
    value = _safe_int(_maybe_get(prompt_tokens_details, "cached_tokens"))
    if value is not None and value >= 0:
        return value
    return None


def _aawm_reference_rate(
    rates: Dict[str, Any],
    *,
    component: str,
) -> Optional[Tuple[float, str]]:
    rate_aliases = {
        "input": (
            (
                "input_usd_per_million_tokens",
                "input_cost_per_million_tokens",
            ),
            ("input_usd_per_token", "input_cost_per_token"),
        ),
        "output": (
            (
                "output_usd_per_million_tokens",
                "output_cost_per_million_tokens",
            ),
            ("output_usd_per_token", "output_cost_per_token"),
        ),
        "cache_read": (
            (
                "cache_read_usd_per_million_tokens",
                "cache_read_cost_per_million_tokens",
            ),
            (
                "cache_read_usd_per_token",
                "cache_read_cost_per_token",
                "cache_read_input_token_cost",
            ),
        ),
    }
    per_million_keys, per_token_keys = rate_aliases[component]
    for key in per_million_keys:
        value = _safe_float(rates.get(key))
        if value is not None and value >= 0:
            return value, "per_million"
    for key in per_token_keys:
        value = _safe_float(rates.get(key))
        if value is not None and value >= 0:
            return value, "per_token"
    return None


def _aawm_reference_component_cost(
    rates: Dict[str, Any],
    *,
    component: str,
    tokens: int,
) -> Optional[float]:
    rate = _aawm_reference_rate(rates, component=component)
    if rate is None:
        return None
    value, unit = rate
    if unit == "per_million":
        return tokens * value / 1_000_000
    return tokens * value


def resolve_aawm_reference_pricing(  # noqa: PLR0915
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    usage_obj: Any,
) -> Optional[Dict[str, Any]]:
    """Resolve one exact provider-specific reference-pricing contract.

    The returned amount is provenance metadata only. Callers must not use
    ``reference_cost_total_usd`` as LiteLLM's actual response cost.
    """
    if not isinstance(provider, str) or not provider:
        return None
    if not isinstance(model, str) or not model:
        return None

    model_cost = _load_bundled_model_cost_map()
    if provider == "opencode_zen":
        if model.startswith("opencode/"):
            route_key = model
        elif model.startswith("opencode_zen/"):
            route_key = f"opencode/{model.split('/', 1)[1]}"
        else:
            route_key = f"opencode/{model}"
    else:
        route_key = (
            model if model.startswith(f"{provider}/") else f"{provider}/{model}"
        )
    if not isinstance(model_cost.get(route_key), dict):
        return None
    model_info = model_cost[route_key]

    provider_specific_entry = model_info.get("provider_specific_entry")
    if not isinstance(provider_specific_entry, dict):
        return None
    provider_entry = provider_specific_entry.get(provider)
    if not isinstance(provider_entry, dict):
        return None
    contract = provider_entry.get("aawm_reference_pricing")
    if not isinstance(contract, dict):
        return None

    basis = contract.get("basis")
    if not isinstance(basis, dict):
        basis = {}
    source = contract.get("source")
    if not isinstance(source, dict):
        source = {}
    equivalence = contract.get("equivalence")
    if not isinstance(equivalence, dict):
        equivalence = {}
    equivalence_status = contract.get(
        "equivalence_status", equivalence.get("status")
    )

    metadata: Dict[str, Any] = {
        "reference_cost_status": contract.get("status"),
        "reference_cost_kind": contract.get("kind"),
        "billing_mode": contract.get("billing_mode"),
        "actual_invoice_cost_known": False,
        "reference_cost_currency": contract.get("currency"),
        "reference_cost_model": route_key,
        "reference_cost_basis_provider": basis.get("provider"),
        "reference_cost_basis_model": basis.get("model"),
        "reference_cost_source_kind": source.get("kind"),
        "reference_cost_source_label": source.get("label"),
        "reference_cost_source_urls": source.get("urls"),
        "reference_cost_source_version": source.get("version"),
        "reference_cost_verified_on": source.get("verified_on"),
        "reference_cost_source": source.get("label"),
        "reference_cost_schema_version": contract.get("schema_version"),
        "reference_cost_cache_mode": contract.get("cache_mode"),
        "reference_cost_rate_schedule": contract.get("rate_schedule"),
        "reference_cost_equivalence_status": equivalence_status,
        "reference_cost_equivalence_evidence": contract.get(
            "equivalence_evidence", equivalence.get("evidence")
        ),
        "reference_cost_unpriced_reason": contract.get("unpriced_reason"),
    }

    if contract.get("schema_version") != 1:
        metadata["reference_cost_status"] = "unpriced"
        metadata["reference_cost_unpriced_reason"] = "unsupported_schema_version"
        return metadata
    if contract.get("actual_invoice_cost_known") is not False:
        metadata["reference_cost_status"] = "unpriced"
        metadata["reference_cost_unpriced_reason"] = (
            "reference_contract_invoice_flag_must_be_false"
        )
        return metadata
    if contract.get("currency") != "USD":
        metadata["reference_cost_status"] = "unpriced"
        metadata["reference_cost_unpriced_reason"] = "unsupported_currency"
        return metadata
    if contract.get("status") != "priced":
        return metadata

    rates = contract.get("rates")
    rates_from_model = contract.get("rates_from_model")
    if rates_from_model is not None:
        if equivalence_status != "exact":
            metadata["reference_cost_status"] = "unpriced"
            metadata["reference_cost_unpriced_reason"] = (
                "rates_from_model_requires_exact_equivalence"
            )
            return metadata
        if isinstance(rates_from_model, str):
            basis_key = rates_from_model
        elif isinstance(rates_from_model, dict):
            basis_provider = rates_from_model.get("provider")
            basis_model = rates_from_model.get("model")
            if not isinstance(basis_provider, str) or not isinstance(
                basis_model, str
            ):
                metadata["reference_cost_status"] = "unpriced"
                metadata["reference_cost_unpriced_reason"] = (
                    "invalid_rates_from_model_pointer"
                )
                return metadata
            basis_key = (
                basis_model
                if basis_model.startswith(f"{basis_provider}/")
                else f"{basis_provider}/{basis_model}"
            )
        else:
            metadata["reference_cost_status"] = "unpriced"
            metadata["reference_cost_unpriced_reason"] = (
                "invalid_rates_from_model_pointer"
            )
            return metadata
        rates = model_cost.get(basis_key)

    rate_schedule = contract.get("rate_schedule")
    if not isinstance(rates, dict) and not isinstance(rate_schedule, list):
        metadata["reference_cost_status"] = "unpriced"
        metadata["reference_cost_unpriced_reason"] = (
            metadata.get("reference_cost_unpriced_reason")
            or "reference_rates_missing"
        )
        return metadata

    prompt_tokens = max(_safe_int(prompt_tokens) or 0, 0)
    completion_tokens = max(_safe_int(completion_tokens) or 0, 0)
    cached_input_tokens = _extract_aawm_cached_input_tokens(usage_obj)
    cache_mode = contract.get("cache_mode")
    if cache_mode == "none":
        if cached_input_tokens not in (None, 0):
            metadata["reference_cost_status"] = "unpriced"
            metadata["reference_cost_unpriced_reason"] = (
                "cache_reported_for_uncached_route"
            )
            return metadata
        cached_input_tokens = 0
    elif cache_mode == "provider_cache_read":
        if cached_input_tokens is None:
            metadata["reference_cost_status"] = "unpriced"
            metadata["reference_cost_unpriced_reason"] = "cache_token_count_unknown"
            return metadata
        cached_input_tokens = min(cached_input_tokens, prompt_tokens)
    else:
        metadata["reference_cost_status"] = "unpriced"
        metadata["reference_cost_unpriced_reason"] = "unsupported_cache_mode"
        return metadata

    total_tokens = prompt_tokens + completion_tokens
    selected_rates = rates
    if isinstance(rate_schedule, list):
        selected_rates = None
        for schedule_entry in rate_schedule:
            if not isinstance(schedule_entry, dict):
                continue
            minimum = _safe_int(schedule_entry.get("min_total_tokens")) or 0
            maximum = schedule_entry.get("max_total_tokens")
            maximum_int = _safe_int(maximum) if maximum is not None else None
            if total_tokens >= minimum and (
                maximum_int is None or total_tokens <= maximum_int
            ):
                selected_rates = schedule_entry
                break
        if selected_rates is None:
            metadata["reference_cost_status"] = "unpriced"
            metadata["reference_cost_unpriced_reason"] = "rate_schedule_no_match"
            return metadata
    if isinstance(selected_rates.get("rates"), dict):
        selected_rates = selected_rates["rates"]

    uncached_input_tokens = prompt_tokens - (cached_input_tokens or 0)
    input_cost = _aawm_reference_component_cost(
        selected_rates,
        component="input",
        tokens=uncached_input_tokens,
    )
    output_cost = _aawm_reference_component_cost(
        selected_rates,
        component="output",
        tokens=completion_tokens,
    )
    if cache_mode == "provider_cache_read":
        cache_cost = _aawm_reference_component_cost(
            selected_rates,
            component="cache_read",
            tokens=cached_input_tokens or 0,
        )
    else:
        cache_cost = 0.0
    if input_cost is None or output_cost is None or cache_cost is None:
        metadata["reference_cost_status"] = "unpriced"
        metadata["reference_cost_unpriced_reason"] = "reference_rates_incomplete"
        return metadata

    total_cost = input_cost + cache_cost + output_cost
    metadata.update(
        {
            "reference_cost_uncached_input_usd": input_cost,
            "reference_cost_cached_input_usd": cache_cost,
            "reference_cost_output_usd": output_cost,
            "reference_cost_total_usd": total_cost,
            "reference_cost_components": {
                "uncached_input_usd": input_cost,
                "cached_input_usd": cache_cost,
                "output_usd": output_cost,
                "total_usd": total_cost,
            },
        }
    )
    return metadata


_HOST_FUNCTION_NAMES = (
    "_load_bundled_model_cost_map",
    "_bundled_model_cost_casefold_lookup",
    "_lookup_bundled_model_cost_info",
    "_calculate_response_cost_from_bundled_model_cost_map",
    "_extract_aawm_cached_input_tokens",
    "_aawm_reference_rate",
    "_aawm_reference_component_cost",
    "resolve_aawm_reference_pricing",
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
    ``functools.lru_cache`` wrappers keep this module's globals (their bodies
    only reference module-local names) and are published by reference so the
    facade-identity invariant holds.
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
