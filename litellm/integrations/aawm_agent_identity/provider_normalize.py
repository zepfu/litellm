"""Provider/model/route-family normalization, api-base/local-route/model-group resolution.

Behavior-preserving Wave A4A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import ipaddress
    from urllib.parse import urlsplit, urlunsplit

    def _clean_non_empty_string(value: Any) -> Optional[str]: ...

    def _extract_request_headers_from_kwargs(
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def _extract_responses_completed_payload_from_passthrough_fallback_text(
        response_text: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _first_explicit_openrouter_model_string(
        *candidates: Any,
    ) -> Optional[str]: ...

    def _first_non_empty_string(*values: Any) -> Optional[str]: ...

    def _get_header_value(headers: Any, *names: str) -> Optional[str]: ...

    def _maybe_get(obj: Any, key: str, default: Any = None) -> Any: ...

    def _maybe_get_path(
        obj: Any,
        *keys: str,
        default: Any = None,
    ) -> Any: ...

    def _metadata_request_tags(metadata: Dict[str, Any]) -> List[str]: ...

    def _session_history_model_from_request_tags(
        metadata: Dict[str, Any],
    ) -> Optional[str]: ...


def _normalize_session_history_provider_name(candidate: Any) -> Optional[str]:
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate_lower = candidate.strip().lower()
    if candidate_lower in {"unknown", "none", "null", "litellm"}:
        return None
    if candidate_lower == "google":
        return "gemini"
    if candidate_lower in {"nvidia", "nvidia_nim", "nvidia-nim"}:
        return "nvidia_nim"
    if candidate_lower in {"opencode", "opencode-zen", "opencode_zen", "zen"}:
        return "opencode_zen"
    if candidate_lower == "grok":
        return "xai"
    if candidate_lower in {
        "local_embed",
        "local-embed",
        "local_rerank",
        "local-rerank",
        "local_llm",
        "local-llm",
        "local_biomed",
        "local-biomed",
        "openrouter",
        "opencode_zen",
        "openai",
        "anthropic",
        "gemini",
        "xai",
    }:
        return candidate_lower.replace("-", "_")
    return candidate_lower


@lru_cache(maxsize=512)
def _session_history_provider_from_model_catalog(model: str) -> Optional[str]:
    normalized_model = str(model or "").strip()
    if not normalized_model or normalized_model.lower() == "unknown":
        return None
    try:
        from litellm.utils import get_model_info

        model_info = get_model_info(model=normalized_model)
    except Exception:
        return None
    if not isinstance(model_info, dict):
        return None
    return _normalize_session_history_provider_name(model_info.get("litellm_provider"))


def _session_history_provider_from_model(model: Any) -> Optional[str]:
    model_lower = str(model or "").strip().lower()
    if not model_lower or model_lower == "unknown":
        return None
    if model_lower.startswith("local_embed/"):
        return "local_embed"
    if model_lower.startswith("local_rerank/"):
        return "local_rerank"
    if model_lower.startswith("local_llm/"):
        return "local_llm"
    if model_lower.startswith("local_biomed/"):
        return "local_biomed"
    if model_lower.startswith("nvidia/"):
        return "nvidia_nim"
    if model_lower.startswith("xai/") or model_lower.startswith("grok"):
        return "xai"
    if model_lower.startswith("openrouter/"):
        return "openrouter"
    if model_lower.startswith(("opencode/", "opencode-zen/", "zen/")):
        return "opencode_zen"
    if "gemini" in model_lower or model_lower.startswith("google/"):
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
    return _session_history_provider_from_model_catalog(str(model or ""))


def _session_history_provider_from_route_family(route_family: Any) -> Optional[str]:
    if not isinstance(route_family, str) or not route_family.strip():
        return None
    route_lower = route_family.lower()
    if "grok" in route_lower or "xai" in route_lower:
        return "xai"
    if "nvidia" in route_lower:
        return "nvidia_nim"
    if "openrouter" in route_lower:
        return "openrouter"
    if "opencode" in route_lower:
        return "opencode_zen"
    if "local_embed" in route_lower or "local-embed" in route_lower:
        return "local_embed"
    if "local_rerank" in route_lower or "local-rerank" in route_lower:
        return "local_rerank"
    if "local_llm" in route_lower or "local-llm" in route_lower:
        return "local_llm"
    if "local_biomed" in route_lower or "local-biomed" in route_lower:
        return "local_biomed"
    if "gemini" in route_lower or "google" in route_lower:
        return "gemini"
    if "codex" in route_lower or "openai" in route_lower:
        return "openai"
    if "anthropic" in route_lower:
        return "anthropic"
    return None


def _session_history_adapter_target_provider(
    metadata: Dict[str, Any],
) -> Optional[str]:
    for tag in _metadata_request_tags(metadata):
        tag_lower = tag.strip().lower()
        if not tag_lower.startswith("anthropic-adapter-target:"):
            continue
        target = tag_lower.split(":", 1)[1].strip()
        if target.startswith("openrouter"):
            return "openrouter"
        if target.startswith(("opencode", "opencode_zen", "zen")):
            return "opencode_zen"
        if target.startswith("nvidia"):
            return "nvidia_nim"
        if target.startswith(("xai", "grok")):
            return "xai"
        if target.startswith(("responses", "openai", "codex", "/v1/responses")):
            return "openai"
    return None


def _session_history_auto_agent_selected_provider(
    metadata: Dict[str, Any],
) -> Optional[str]:
    selected_provider = _normalize_session_history_provider_name(metadata.get("codex_auto_agent_selected_provider"))
    if selected_provider is not None:
        return selected_provider
    selected_provider = _normalize_session_history_provider_name(metadata.get("anthropic_auto_agent_selected_provider"))
    if selected_provider is not None:
        return selected_provider
    return _normalize_session_history_provider_name(metadata.get("aawm_auto_agent_selected_provider"))


def _session_history_adapter_model(metadata: Dict[str, Any]) -> Optional[str]:
    prefix = "anthropic-adapter-model:"
    for tag in _metadata_request_tags(metadata):
        stripped_tag = tag.strip()
        if stripped_tag.lower().startswith(prefix):
            return stripped_tag[len(prefix) :].strip() or None
    return None
def _normalize_session_history_provider(
    provider: Any,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = metadata or {}
    adapter_target_provider = _session_history_adapter_target_provider(metadata)
    if adapter_target_provider is not None:
        return adapter_target_provider

    auto_agent_provider = _session_history_auto_agent_selected_provider(metadata)
    if auto_agent_provider is not None:
        return auto_agent_provider

    credential_family = str(metadata.get("credential_family") or "").strip().lower()
    if (
        credential_family == "xai_oauth"
        or metadata.get("xai_oauth_managed") is True
        or metadata.get("xai_oauth_public_model") is not None
    ):
        return "xai"

    route_provider = _session_history_provider_from_route_family(metadata.get("passthrough_route_family"))
    if route_provider is not None and route_provider != "anthropic":
        return route_provider

    model_provider = _session_history_provider_from_model(model)

    normalized_provider = _normalize_session_history_provider_name(provider)
    if (
        normalized_provider in {"anthropic", "openai"}
        and model_provider is not None
        and model_provider != normalized_provider
    ):
        return model_provider
    if normalized_provider is not None:
        return normalized_provider

    for key in (
        "custom_llm_provider",
        "provider",
        "litellm_provider",
        "aawm_stream_logging_custom_llm_provider",
    ):
        normalized_provider = _normalize_session_history_provider_name(metadata.get(key))
        if (
            normalized_provider in {"anthropic", "openai"}
            and model_provider is not None
            and model_provider != normalized_provider
        ):
            return model_provider
        if normalized_provider is not None:
            return normalized_provider

    if route_provider is not None:
        return route_provider

    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str) and request_route.strip():
        route_lower = request_route.lower()
        if "gemini" in route_lower or "google" in route_lower:
            return "gemini"
        if route_lower.startswith("/v1/"):
            return "openai"
        if route_lower.startswith("/anthropic/"):
            return "anthropic"

    api_base = metadata.get("api_base") or _maybe_get(metadata.get("hidden_params"), "api_base")
    if isinstance(api_base, str) and api_base.strip():
        api_base_lower = api_base.lower()
        if "api.x.ai" in api_base_lower or "cli-chat-proxy.grok.com" in api_base_lower:
            return "xai"
        if "integrate.api.nvidia.com" in api_base_lower:
            return "nvidia_nim"
        if "openrouter.ai" in api_base_lower:
            return "openrouter"
        if "opencode.ai/zen" in api_base_lower:
            return "opencode_zen"
        if "anthropic.com" in api_base_lower:
            return "anthropic"
        if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
            return "gemini"
        if "openai.com" in api_base_lower:
            return "openai"

    return model_provider


def _sanitize_session_history_api_base(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None

    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return None

    if not parsed.scheme or not parsed.netloc:
        return cleaned.split("?", 1)[0].split("#", 1)[0].rstrip("/") or None

    hostname = parsed.hostname
    if not hostname:
        return None

    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "")) or None


def _is_local_session_history_api_base(value: Any) -> bool:
    sanitized = _sanitize_session_history_api_base(value)
    if not sanitized:
        return False

    try:
        hostname = urlsplit(sanitized).hostname
    except ValueError:
        return False
    if not hostname:
        return False

    hostname_lower = hostname.lower()
    if hostname_lower in {"localhost", "host.docker.internal"}:
        return True

    try:
        parsed_ip = ipaddress.ip_address(hostname_lower)
    except ValueError:
        return False

    return parsed_ip.is_loopback or parsed_ip.is_private or parsed_ip.is_link_local or parsed_ip.is_unspecified


def _extract_session_history_api_base(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        litellm_params = {}

    for candidate in (
        standard_logging_object.get("api_base"),
        _maybe_get_path(standard_logging_object, "hidden_params", "api_base"),
        litellm_params.get("api_base"),
        metadata.get("api_base"),
        _maybe_get(metadata.get("hidden_params"), "api_base"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "url"),
        _maybe_get_path(kwargs.get("standard_pass_through_logging_payload"), "url"),
    ):
        sanitized = _sanitize_session_history_api_base(candidate)
        if sanitized:
            return sanitized
    return None


def _get_session_history_model_group(
    metadata: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty_string(
        metadata.get("model_group"),
        standard_logging_object.get("model_group"),
    )


def _resolve_inbound_model_alias(
    *,
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    resolved_model: str,
) -> str:
    return (
        _first_non_empty_string(
            metadata.get("model_alias_label"),
            metadata.get("requested_model_alias"),
            metadata.get("codex_auto_agent_alias"),
            metadata.get("anthropic_auto_agent_alias"),
            metadata.get("aawm_auto_agent_alias"),
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
            _maybe_get_path(standard_logging_object, "request_body", "model"),
            kwargs.get("model"),
            standard_logging_object.get("model"),
            metadata.get("model"),
            resolved_model,
        )
        or "unknown"
    )


def _resolve_inbound_model_alias_from_langfuse(
    *,
    observation: Dict[str, Any],
    metadata: Dict[str, Any],
    input_model: Optional[str],
    output_model: Optional[str],
    resolved_model: str,
) -> str:
    return (
        _first_non_empty_string(
            metadata.get("model_alias_label"),
            metadata.get("requested_model_alias"),
            metadata.get("codex_auto_agent_alias"),
            metadata.get("anthropic_auto_agent_alias"),
            metadata.get("aawm_auto_agent_alias"),
            input_model,
            metadata.get("model"),
            observation.get("model"),
            output_model,
            resolved_model,
        )
        or "unknown"
    )


def _normalize_session_history_model_group(
    model_group: Optional[str],
    metadata: Dict[str, Any],
    resolved_model: str,
) -> Optional[str]:
    normalized_group = _clean_non_empty_string(model_group)
    if normalized_group is None:
        return None
    group_lower = normalized_group.lower()

    auto_agent_aliases: Tuple[Tuple[Optional[str], Tuple[Any, ...]], ...] = (
        (
            _clean_non_empty_string(metadata.get("codex_auto_agent_alias")),
            (
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("aawm_auto_agent_selected_model"),
            ),
        ),
        (
            _clean_non_empty_string(metadata.get("anthropic_auto_agent_alias")),
            (
                metadata.get("anthropic_auto_agent_selected_model"),
                metadata.get("aawm_auto_agent_selected_model"),
            ),
        ),
        (
            _clean_non_empty_string(metadata.get("aawm_auto_agent_alias")),
            (
                metadata.get("aawm_auto_agent_selected_model"),
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("anthropic_auto_agent_selected_model"),
            ),
        ),
        (
            _clean_non_empty_string(metadata.get("requested_model_alias")),
            (
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("anthropic_auto_agent_selected_model"),
                metadata.get("aawm_auto_agent_selected_model"),
            ),
        ),
    )
    for auto_alias, selected_model_candidates in auto_agent_aliases:
        if auto_alias and group_lower == auto_alias.lower():
            return _first_non_empty_string(*selected_model_candidates, resolved_model)

    return normalized_group


def _is_completion_call_type(call_type: Any) -> bool:
    if not isinstance(call_type, str) or not call_type.strip():
        return False
    return "completion" in call_type.strip().lower()


def _is_embedding_call_type(call_type: Any, api_base: Optional[str]) -> bool:
    call_lower = str(call_type or "").strip().lower()
    if "embedding" in call_lower or "aembedding" in call_lower:
        return True
    sanitized = _sanitize_session_history_api_base(api_base)
    if not sanitized:
        return False
    try:
        path = urlsplit(sanitized).path.lower()
    except ValueError:
        return False
    return "embedding" in path


def _strip_local_provider_model_prefix(model: str) -> str:
    normalized = str(model or "").strip()
    lowered = normalized.lower()
    for prefix in ("local_embed/", "local_rerank/", "local_llm/", "local_biomed/"):
        if lowered.startswith(prefix):
            return normalized[len(prefix) :].strip() or normalized
    return normalized


def _session_history_provider_from_api_base(
    api_base: Any,
    *,
    call_type: Any = None,
) -> Optional[str]:
    sanitized = _sanitize_session_history_api_base(api_base)
    if not sanitized:
        return None
    api_base_lower = sanitized.lower()
    if "api.x.ai" in api_base_lower or "cli-chat-proxy.grok.com" in api_base_lower:
        return "xai"
    if "integrate.api.nvidia.com" in api_base_lower:
        return "nvidia_nim"
    if "openrouter.ai" in api_base_lower:
        return "openrouter"
    if "opencode.ai/zen" in api_base_lower:
        return "opencode_zen"
    if "anthropic.com" in api_base_lower:
        return "anthropic"
    if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
        return "gemini"
    if "openai.com" in api_base_lower:
        return "openai"
    if _is_local_session_history_api_base(sanitized) and _is_embedding_call_type(
        call_type,
        sanitized,
    ):
        return "local_embed"
    return None


def _apply_local_embedding_route_metadata(
    *,
    metadata: Dict[str, Any],
    resolved_provider: Optional[str],
    resolved_model: str,
    model_group: Optional[str],
    call_type: Any,
    api_base: Optional[str],
) -> Tuple[Optional[str], str]:
    if not _is_embedding_call_type(call_type, api_base):
        return resolved_provider, resolved_model
    if not _is_local_session_history_api_base(api_base):
        return resolved_provider, resolved_model
    if resolved_provider not in {None, "openai", "local_embed"}:
        return resolved_provider, resolved_model

    upstream_model = _strip_local_provider_model_prefix(resolved_model)
    route_model = _clean_non_empty_string(upstream_model) or _clean_non_empty_string(model_group)
    if not route_model:
        return "local_embed", resolved_model

    metadata["aawm_local_route"] = True
    metadata["aawm_local_route_family"] = "local_embedding"
    if model_group:
        metadata["aawm_local_model_group"] = model_group
    metadata["aawm_local_upstream_provider"] = "local_embed"
    metadata["aawm_local_upstream_model"] = route_model
    sanitized_api_base = _sanitize_session_history_api_base(api_base)
    if sanitized_api_base:
        metadata["aawm_local_upstream_api_base"] = sanitized_api_base

    return "local_embed", route_model


def _apply_local_llm_route_metadata(
    *,
    metadata: Dict[str, Any],
    resolved_provider: Optional[str],
    resolved_model: str,
    model_group: Optional[str],
    call_type: Any,
    api_base: Optional[str],
) -> Tuple[Optional[str], str]:
    if (
        resolved_provider != "openai"
        or not model_group
        or not api_base
        or not _is_completion_call_type(call_type)
        or not _is_local_session_history_api_base(api_base)
    ):
        return resolved_provider, resolved_model

    upstream_model = _clean_non_empty_string(_strip_local_provider_model_prefix(resolved_model)) or model_group

    metadata["aawm_local_route"] = True
    metadata["aawm_local_route_family"] = "local_llm_chat"
    metadata["aawm_local_model_group"] = model_group
    metadata["aawm_local_upstream_provider"] = "openai"
    metadata["aawm_local_upstream_model"] = upstream_model
    sanitized_api_base = _sanitize_session_history_api_base(api_base)
    if sanitized_api_base:
        metadata["aawm_local_upstream_api_base"] = sanitized_api_base

    return "local_llm", model_group


_LOCAL_BIOMED_SESSION_HISTORY_ROUTES = {
    (8094, "/extract"): {
        "model": "scispacy",
        "service": "scispacy",
        "endpoint": "extract",
    },
    (8095, "/annotate"): {
        "model": "tinybern2",
        "service": "tinybern2",
        "endpoint": "annotate",
    },
}


def _resolve_local_biomed_session_history_route(
    api_base: Optional[str],
) -> Optional[Dict[str, str]]:
    sanitized = _sanitize_session_history_api_base(api_base)
    if not sanitized:
        return None

    try:
        parsed = urlsplit(sanitized)
    except ValueError:
        return None

    route_info = _LOCAL_BIOMED_SESSION_HISTORY_ROUTES.get((parsed.port or 0, parsed.path.rstrip("/")))
    if route_info is None:
        return None
    return dict(route_info)


def _apply_local_biomed_route_metadata(
    *,
    metadata: Dict[str, Any],
    resolved_provider: Optional[str],
    resolved_model: str,
    model_group: Optional[str],
    call_type: Any,
    api_base: Optional[str],
) -> Tuple[Optional[str], str, Optional[str]]:
    if str(call_type or "").strip().lower() != "pass_through_endpoint":
        return resolved_provider, resolved_model, model_group

    route_info = _resolve_local_biomed_session_history_route(api_base)
    if route_info is None:
        return resolved_provider, resolved_model, model_group

    route_model = route_info["model"]
    sanitized_api_base = _sanitize_session_history_api_base(api_base)
    metadata["aawm_local_route"] = True
    metadata["aawm_local_route_family"] = "local_biomed_rest"
    metadata["aawm_local_model_group"] = route_model
    metadata["aawm_local_service"] = route_info["service"]
    metadata["aawm_local_endpoint"] = route_info["endpoint"]
    metadata["aawm_local_upstream_provider"] = "local_rest"
    metadata["aawm_local_upstream_model"] = route_model
    if sanitized_api_base:
        metadata["aawm_local_upstream_api_base"] = sanitized_api_base
        metadata["aawm_local_upstream_url"] = sanitized_api_base
    metadata.setdefault("passthrough_route_family", "local_biomed")

    return "local_biomed", route_model, model_group or route_model


def _resolve_session_history_model(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
) -> str:
    grok_model_override = _resolve_xai_grok_model_override(kwargs, metadata)
    if grok_model_override:
        return grok_model_override

    explicit_openrouter_model = _first_explicit_openrouter_model_string(
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("aawm_auto_agent_selected_model"),
        metadata.get("anthropic_adapter_original_model"),
        metadata.get("codex_adapter_original_model"),
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
        _maybe_get_path(standard_logging_object, "request_body", "model"),
        metadata.get("model"),
        kwargs.get("model"),
        standard_logging_object.get("model"),
    )
    if explicit_openrouter_model is not None:
        return explicit_openrouter_model

    if str(kwargs.get("custom_llm_provider") or "").lower() == "openrouter":
        for candidate in (
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
            _maybe_get_path(standard_logging_object, "request_body", "model"),
        ):
            if candidate is None:
                continue
            normalized = str(candidate).strip()
            if normalized.startswith("openrouter/"):
                return normalized

    result_completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        _maybe_get(result, "response")
    )
    standard_completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        _maybe_get(standard_logging_object.get("response"), "response")
    )
    candidates = (
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("aawm_auto_agent_selected_model"),
        kwargs.get("model"),
        standard_logging_object.get("model"),
        _session_history_model_from_request_tags(metadata),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        _session_history_adapter_model(metadata),
        metadata.get("anthropic_adapter_model"),
        metadata.get("codex_adapter_model"),
        metadata.get("model"),
        _maybe_get(result, "model"),
        _maybe_get(_maybe_get(result_completed_payload, "response"), "model"),
        _maybe_get(_maybe_get(standard_completed_payload, "response"), "model"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if normalized and normalized.lower() != "unknown":
            return normalized
    return "unknown"


def _resolve_xai_grok_model_override(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    provider = str(kwargs.get("custom_llm_provider") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    if provider not in {"xai", "grok"} and "grok" not in route_family:
        return None

    headers = _extract_request_headers_from_kwargs(kwargs)
    for candidate in (
        _get_header_value(headers, "x-grok-model-override"),
        metadata.get("grok_model_override"),
        metadata.get("model_group"),
        metadata.get("model"),
    ):
        normalized = _clean_non_empty_string(candidate)
        if normalized and normalized.lower() != "unknown":
            return normalized
    return None


_HOST_FUNCTION_NAMES = (
    "_normalize_session_history_provider_name",
    "_session_history_provider_from_model_catalog",
    "_session_history_provider_from_model",
    "_session_history_provider_from_route_family",
    "_session_history_adapter_target_provider",
    "_session_history_auto_agent_selected_provider",
    "_session_history_adapter_model",
    "_normalize_session_history_provider",
    "_sanitize_session_history_api_base",
    "_is_local_session_history_api_base",
    "_extract_session_history_api_base",
    "_get_session_history_model_group",
    "_resolve_inbound_model_alias",
    "_resolve_inbound_model_alias_from_langfuse",
    "_normalize_session_history_model_group",
    "_is_completion_call_type",
    "_is_embedding_call_type",
    "_strip_local_provider_model_prefix",
    "_session_history_provider_from_api_base",
    "_apply_local_embedding_route_metadata",
    "_apply_local_llm_route_metadata",
    "_LOCAL_BIOMED_SESSION_HISTORY_ROUTES",
    "_resolve_local_biomed_session_history_route",
    "_apply_local_biomed_route_metadata",
    "_resolve_session_history_model",
    "_resolve_xai_grok_model_override",
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

    wrapped = getattr(value, "__wrapped__", None)
    cache_parameters = getattr(value, "cache_parameters", None)
    if not isinstance(wrapped, _FunctionType) or not callable(cache_parameters):
        return value

    parameters = cache_parameters()
    if not isinstance(parameters, dict) or not {"maxsize", "typed"} <= parameters.keys():
        return value

    rebound_wrapped = _rebind_to_host_globals(wrapped, host_globals)
    rebound = lru_cache(
        maxsize=parameters["maxsize"],
        typed=bool(parameters["typed"]),
    )(rebound_wrapped)
    for attribute, attribute_value in getattr(value, "__dict__", {}).items():
        if attribute != "__wrapped__":
            setattr(rebound, attribute, attribute_value)
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
        _installed = _rebind_installable_callable(_original, host_globals)
        mod[_name] = _installed
        host_globals[_name] = _installed
