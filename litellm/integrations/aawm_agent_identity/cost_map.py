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


_HOST_FUNCTION_NAMES = (
    "_load_bundled_model_cost_map",
    "_bundled_model_cost_casefold_lookup",
    "_lookup_bundled_model_cost_info",
    "_calculate_response_cost_from_bundled_model_cost_map",
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
