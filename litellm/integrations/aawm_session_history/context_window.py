"""Anthropic 1M context-window beta classification and enrichment.

Behavior-preserving Wave A4C extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports of identity helpers are intentionally
absent here."""

from __future__ import annotations

from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:

    def _clean_non_empty_string(value: Any) -> Optional[str]: ...

    def _extract_request_headers_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]: ...

    def _get_header_value(headers: Any, *names: str) -> Optional[str]: ...

_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX = "[1m]"


_ANTHROPIC_CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"


_ANTHROPIC_CONTEXT_1M_BETA_PREFIX = "context-1m"


_ANTHROPIC_CONTEXT_WINDOW_DEFAULT_TOKEN_COUNT = 200_000


_ANTHROPIC_CONTEXT_WINDOW_1M_TOKEN_COUNT = 1_000_000


_ANTHROPIC_CONTEXT_WINDOW_METADATA_KEYS = (
    "anthropic_context_window_mode",
    "anthropic_context_window_requested_tokens",
    "anthropic_context_window_source",
    "anthropic_context_window_beta",
    "anthropic_context_window_classification",
)


def _is_anthropic_session_history_context(
    *,
    provider: Optional[str],
    resolved_model: str,
    metadata: Dict[str, Any],
) -> bool:
    provider_lower = str(provider or "").strip().lower()
    if provider_lower in {"anthropic", "azure_ai", "bedrock"}:
        return True
    route_family = (
        str(
            metadata.get("passthrough_route_family")
            or metadata.get("route_family")
            or metadata.get("openai_passthrough_route_family")
            or ""
        )
        .strip()
        .lower()
    )
    if "anthropic" in route_family:
        return True
    model_lower = str(resolved_model or "").strip().lower()
    if model_lower.startswith("claude") or "claude" in model_lower:
        return True
    for key in (
        "anthropic_adapter_model",
        "anthropic_adapter_original_model",
        "anthropic_auto_agent_selected_model",
    ):
        candidate = str(metadata.get(key) or "").strip().lower()
        if candidate.startswith("claude") or "anthropic" in candidate:
            return True
    return False


def _iter_anthropic_beta_header_candidates(
    headers: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[str]:
    candidates: List[str] = []
    header_names = (
        "anthropic-beta",
        "x-pass-anthropic-beta",
        "llm_provider-anthropic-beta",
    )
    if headers:
        for header_name in header_names:
            value = _get_header_value(headers, header_name)
            if value:
                candidates.append(value)
        for key, value in headers.items():
            key_lower = str(key).lower()
            if key_lower in {"anthropic-beta", "x-pass-anthropic-beta"}:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)
            elif key_lower.startswith("llm_provider-") and "anthropic-beta" in key_lower:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)

    for meta_key in (
        "anthropic-beta",
        "anthropic_beta",
        "llm_provider-anthropic-beta",
        "x-pass-anthropic-beta",
    ):
        cleaned = _clean_non_empty_string(metadata.get(meta_key))
        if cleaned:
            candidates.append(cleaned)

    for nested_key in ("hidden_params", "_hidden_params", "additional_headers"):
        nested = metadata.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            key_lower = str(key).lower()
            if key_lower in {"anthropic-beta", "x-pass-anthropic-beta"}:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)
            elif key_lower.startswith("llm_provider-") and "anthropic-beta" in key_lower:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)
    return candidates


def _split_anthropic_beta_values(raw_value: str) -> List[str]:
    return [token.strip() for token in str(raw_value).replace(";", ",").split(",") if token.strip()]


def _extract_context_1m_beta_values(
    headers: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[str]:
    matched: List[str] = []
    seen: Set[str] = set()
    for raw in _iter_anthropic_beta_header_candidates(headers, metadata):
        for beta_value in _split_anthropic_beta_values(raw):
            beta_lower = beta_value.lower()
            if beta_lower == _ANTHROPIC_CONTEXT_1M_BETA_HEADER.lower() or beta_lower.startswith(
                _ANTHROPIC_CONTEXT_1M_BETA_PREFIX
            ):
                if beta_value not in seen:
                    seen.add(beta_value)
                    matched.append(beta_value)
    return matched


def _model_strings_indicate_context_1m_suffix(*model_values: Any) -> bool:
    suffix_lower = _ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX.lower()
    for value in model_values:
        cleaned = _clean_non_empty_string(value)
        if cleaned and cleaned.lower().endswith(suffix_lower):
            return True
    return False


def _select_safe_anthropic_context_window_beta(beta_values: List[str]) -> Optional[str]:
    if not beta_values:
        return None
    for beta_value in beta_values:
        if beta_value.lower() == _ANTHROPIC_CONTEXT_1M_BETA_HEADER.lower():
            return beta_value
    for beta_value in beta_values:
        if beta_value.lower().startswith(_ANTHROPIC_CONTEXT_1M_BETA_PREFIX):
            return beta_value
    return beta_values[0]


def _apply_anthropic_context_window_metadata_fields(
    metadata: Dict[str, Any],
    *,
    mode: str,
    requested_tokens: Optional[int],
    source: str,
    beta: Optional[str] = None,
    classification: Optional[str] = None,
) -> None:
    metadata["anthropic_context_window_mode"] = mode
    metadata["anthropic_context_window_requested_tokens"] = requested_tokens
    metadata["anthropic_context_window_source"] = source
    if beta is not None:
        metadata["anthropic_context_window_beta"] = beta
    else:
        metadata.pop("anthropic_context_window_beta", None)
    if classification is not None:
        metadata["anthropic_context_window_classification"] = classification
    else:
        metadata.pop("anthropic_context_window_classification", None)


def _classify_anthropic_context_window_from_retained_evidence(
    metadata: Dict[str, Any],
    *,
    resolved_model: str,
    inbound_model_alias: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
    allow_implicit_default: bool = False,
) -> Optional[Dict[str, Any]]:
    beta_values = _extract_context_1m_beta_values(headers, metadata)
    if beta_values:
        return {
            "mode": "extended_1m",
            "requested_tokens": _ANTHROPIC_CONTEXT_WINDOW_1M_TOKEN_COUNT,
            "source": "anthropic_beta_header",
            "beta": _select_safe_anthropic_context_window_beta(beta_values),
            "classification": "classified",
        }

    if _model_strings_indicate_context_1m_suffix(
        inbound_model_alias,
        metadata.get("inbound_model_alias"),
        metadata.get("requested_model_alias"),
        metadata.get("model_alias_label"),
        metadata.get("anthropic_native_passthrough_model_alias"),
        metadata.get("source_model"),
        metadata.get("model"),
        resolved_model,
    ):
        return {
            "mode": "extended_1m",
            "requested_tokens": _ANTHROPIC_CONTEXT_WINDOW_1M_TOKEN_COUNT,
            "source": "model_suffix_1m",
            "beta": None,
            "classification": "classified",
        }

    if not _is_anthropic_session_history_context(
        provider=str(metadata.get("custom_llm_provider") or metadata.get("provider") or ""),
        resolved_model=resolved_model,
        metadata=metadata,
    ):
        return None

    if allow_implicit_default:
        return {
            "mode": "default_200k",
            "requested_tokens": _ANTHROPIC_CONTEXT_WINDOW_DEFAULT_TOKEN_COUNT,
            "source": "no_extended_context_evidence",
            "beta": None,
            "classification": "classified",
        }

    return {
        "mode": "unknown",
        "requested_tokens": None,
        "source": "unavailable",
        "beta": None,
        "classification": "unavailable",
    }


def _enrich_anthropic_context_window_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    resolved_model: Optional[str] = None,
    inbound_model_alias: Optional[str] = None,
    provider: Optional[str] = None,
    allow_implicit_default: bool = True,
) -> None:
    model_value = _clean_non_empty_string(resolved_model or metadata.get("model") or kwargs.get("model")) or "unknown"
    provider_value = _clean_non_empty_string(
        provider or kwargs.get("custom_llm_provider") or metadata.get("custom_llm_provider")
    )
    if provider_value:
        metadata.setdefault("custom_llm_provider", provider_value)

    headers = _extract_request_headers_from_kwargs(kwargs)
    classification = _classify_anthropic_context_window_from_retained_evidence(
        metadata,
        resolved_model=model_value,
        inbound_model_alias=inbound_model_alias,
        headers=headers,
        allow_implicit_default=allow_implicit_default,
    )
    if classification is None:
        for key in _ANTHROPIC_CONTEXT_WINDOW_METADATA_KEYS:
            metadata.pop(key, None)
        return

    _apply_anthropic_context_window_metadata_fields(
        metadata,
        mode=classification["mode"],
        requested_tokens=classification["requested_tokens"],
        source=classification["source"],
        beta=classification.get("beta"),
        classification=classification.get("classification"),
    )


def _enrich_backfill_anthropic_context_window_metadata(
    record: Dict[str, Any],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return
    provider = _clean_non_empty_string(record.get("provider"))
    if provider:
        metadata.setdefault("custom_llm_provider", provider)
    classification = _classify_anthropic_context_window_from_retained_evidence(
        metadata,
        resolved_model=str(record.get("model") or metadata.get("model") or "unknown"),
        inbound_model_alias=record.get("inbound_model_alias"),
        headers=None,
        allow_implicit_default=False,
    )
    if classification is None:
        return
    _apply_anthropic_context_window_metadata_fields(
        metadata,
        mode=classification["mode"],
        requested_tokens=classification["requested_tokens"],
        source=classification["source"],
        beta=classification.get("beta"),
        classification=classification.get("classification"),
    )
    record["metadata"] = metadata



_HOST_FUNCTION_NAMES = (
    "_is_anthropic_session_history_context",
    "_iter_anthropic_beta_header_candidates",
    "_split_anthropic_beta_values",
    "_extract_context_1m_beta_values",
    "_model_strings_indicate_context_1m_suffix",
    "_select_safe_anthropic_context_window_beta",
    "_apply_anthropic_context_window_metadata_fields",
    "_classify_anthropic_context_window_from_retained_evidence",
    "_enrich_anthropic_context_window_metadata",
    "_enrich_backfill_anthropic_context_window_metadata",
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
