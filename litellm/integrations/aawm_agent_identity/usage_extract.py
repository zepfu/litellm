"""Usage extraction: usage objects from all sources, token field extractors, reasoning tokens, rerank payloads.

Behavior-preserving Wave A4A extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports are intentionally absent here.
"""

import re
from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import json

    from litellm._logging import verbose_logger

    def _clean_non_empty_string(value: Any) -> Optional[str]: ...

    def _ensure_mutable_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]: ...

    def _extract_provider_cache_request_body(
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]: ...

    def _extract_provider_specific_fields(message: Any) -> Dict[str, Any]: ...

    def _extract_request_body_from_langfuse_input(
        value: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _extract_responses_completed_payload_from_passthrough_fallback_text(
        response_text: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _first_non_none(*values: Any) -> Any: ...

    def _get_litellm_module() -> Any: ...

    def _get_rate_limit_header_value(
        candidate: Dict[str, Any],
        *header_names: str,
        lower_headers: Optional[Dict[str, Any]] = None,
    ) -> Any: ...

    def _maybe_get(obj: Any, key: str, default: Any = None) -> Any: ...

    def _maybe_get_path(
        obj: Any,
        *keys: str,
        default: Any = None,
    ) -> Any: ...

    def _maybe_parse_json_text(value: str) -> Any: ...

    def _merge_tags(
        metadata: Dict[str, Any],
        tags_to_add: List[str],
    ) -> None: ...

    def _metadata_request_tags(metadata: Dict[str, Any]) -> List[str]: ...

    def _safe_int(value: Any) -> Optional[int]: ...


def _build_usage_object_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None

    usage_object = metadata.get("usage_object")
    reconstructed: Dict[str, Any] = dict(usage_object) if isinstance(usage_object, dict) and usage_object else {}

    input_tokens = _safe_int(metadata.get("usage_input_tokens"))
    output_tokens = _safe_int(metadata.get("usage_output_tokens"))
    total_tokens = _safe_int(metadata.get("usage_total_tokens"))
    cache_read_input_tokens = _safe_int(metadata.get("usage_cache_read_input_tokens"))
    cache_creation_input_tokens = _safe_int(metadata.get("usage_cache_creation_input_tokens"))
    reasoning_tokens_reported = _safe_int(metadata.get("usage_reasoning_tokens_reported"))

    if not any(
        value is not None
        for value in (
            input_tokens,
            output_tokens,
            total_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            reasoning_tokens_reported,
        )
    ):
        return reconstructed or None

    if input_tokens is not None:
        reconstructed["input_tokens"] = input_tokens
        reconstructed["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        reconstructed["output_tokens"] = output_tokens
        reconstructed["completion_tokens"] = output_tokens
    if total_tokens is not None:
        reconstructed["total_tokens"] = total_tokens
    if cache_read_input_tokens is not None:
        reconstructed["cache_read_input_tokens"] = cache_read_input_tokens
        input_tokens_details = dict(reconstructed.get("input_tokens_details") or {})
        input_tokens_details["cached_tokens"] = cache_read_input_tokens
        reconstructed["input_tokens_details"] = input_tokens_details
    if cache_creation_input_tokens is not None:
        reconstructed["cache_creation_input_tokens"] = cache_creation_input_tokens
    if reasoning_tokens_reported is not None:
        reconstructed["reasoning_tokens"] = reasoning_tokens_reported
        output_tokens_details = dict(reconstructed.get("output_tokens_details") or {})
        output_tokens_details["reasoning_tokens"] = reasoning_tokens_reported
        reconstructed["output_tokens_details"] = output_tokens_details

    return reconstructed or None


def _build_usage_object_from_token_count_payload(
    output_payload: Any,
) -> Optional[Dict[str, Any]]:
    if isinstance(output_payload, str):
        parsed_payload = _maybe_parse_json_text(output_payload)
        if parsed_payload is None:
            return None
        return _build_usage_object_from_token_count_payload(parsed_payload)

    if not isinstance(output_payload, dict):
        return None

    input_tokens = _safe_int(
        _first_non_none(
            output_payload.get("prompt_tokens"),
            output_payload.get("input_tokens"),
            output_payload.get("inputTokens"),
        )
    )
    output_tokens = _safe_int(
        _first_non_none(
            output_payload.get("completion_tokens"),
            output_payload.get("output_tokens"),
            output_payload.get("outputTokens"),
        )
    )
    total_tokens = _safe_int(
        _first_non_none(
            output_payload.get("total_tokens"),
            output_payload.get("totalTokens"),
        )
    )
    # Only accept generic "total" when sibling token keys already establish this
    # as a token-count payload, not a pagination/billing envelope.
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = _safe_int(output_payload.get("total"))

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    usage_object: Dict[str, Any] = {}
    usage_object["token_count_response"] = True
    if input_tokens is not None:
        usage_object["prompt_tokens"] = input_tokens
        usage_object["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage_object["completion_tokens"] = output_tokens
        usage_object["output_tokens"] = output_tokens
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    if total_tokens is not None:
        usage_object["total_tokens"] = total_tokens

    return usage_object or None


def _extract_responses_completed_response_from_langfuse_output(
    output_payload: Any,
) -> Optional[Dict[str, Any]]:
    raw_text = output_payload
    if isinstance(output_payload, dict):
        if isinstance(output_payload.get("response"), dict):
            return output_payload["response"]
        if isinstance(output_payload.get("raw_output"), str):
            raw_text = output_payload["raw_output"]

    completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(raw_text)
    if not isinstance(completed_payload, dict):
        return None
    response_payload = completed_payload.get("response")
    return response_payload if isinstance(response_payload, dict) else None


def _build_usage_object_from_langfuse_output(output_payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(output_payload, dict):
        usage = output_payload.get("usage")
        if isinstance(usage, dict) and usage:
            return dict(usage)

    token_count_usage = _build_usage_object_from_token_count_payload(output_payload)
    if token_count_usage is not None:
        return token_count_usage

    response_payload = _extract_responses_completed_response_from_langfuse_output(output_payload)
    if not isinstance(response_payload, dict):
        return None
    usage = response_payload.get("usage")
    return dict(usage) if isinstance(usage, dict) and usage else None


def _extract_codex_model_from_response_headers(metadata: Dict[str, Any]) -> Optional[str]:
    headers = metadata.get("codex_response_headers")
    if not isinstance(headers, dict):
        return None

    limit_name = _clean_non_empty_string(_get_rate_limit_header_value(headers, "x-codex-bengalfox-limit-name"))
    if not limit_name:
        return None

    normalized = re.sub(r"[^a-z0-9._-]+", "-", limit_name.lower()).strip("-")
    if normalized.startswith("gpt-") and "codex" in normalized:
        return normalized
    return None


def _session_history_metadata_model(metadata: Dict[str, Any]) -> Optional[str]:
    hidden_params = metadata.get("hidden_params")
    return _first_known_model_string(
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("codex_adapter_model"),
        metadata.get("litellm_model"),
        _session_history_model_from_request_tags(metadata),
        metadata.get("model"),
        _maybe_get(hidden_params, "model"),
    )


_SESSION_HISTORY_CLAUDE_MODEL_TAG_RE = re.compile(
    r"^claude-(?:opus|sonnet|haiku)-[a-z0-9_.-]+$",
    re.IGNORECASE,
)


def _session_history_model_from_request_tags(
    metadata: Dict[str, Any],
) -> Optional[str]:
    for tag in _metadata_request_tags(metadata):
        if not isinstance(tag, str):
            continue
        stripped_tag = tag.strip()
        tag_lower = stripped_tag.lower()
        if not tag_lower.startswith("claude-exp:"):
            continue
        candidate = stripped_tag.split(":", 1)[1].strip()
        if _SESSION_HISTORY_CLAUDE_MODEL_TAG_RE.fullmatch(candidate):
            return candidate
    return None


def _extract_model_from_langfuse_input(input_payload: Any) -> Optional[str]:
    request_body = _extract_request_body_from_langfuse_input(input_payload)
    if not isinstance(request_body, dict):
        return None
    body = request_body.get("body")
    return _first_known_model_string(
        request_body.get("model"),
        _maybe_get(body, "model"),
    )


def _extract_model_from_langfuse_output(output_payload: Any) -> Optional[str]:
    if isinstance(output_payload, dict):
        model = output_payload.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()

    response_payload = _extract_responses_completed_response_from_langfuse_output(output_payload)
    model = _maybe_get(response_payload, "model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _first_known_model_string(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip()
        if not cleaned or cleaned.lower() in {"unknown", "none", "null"}:
            continue
        return cleaned
    return None


def _first_explicit_openrouter_model_string(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip()
        if cleaned.lower().startswith("openrouter/") and len(cleaned) > len("openrouter/"):
            return cleaned
    return None


def _coerce_usage_object_to_dict(usage_obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(usage_obj, dict):
        return dict(usage_obj)

    model_dump = getattr(usage_obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    dict_method = getattr(usage_obj, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method(exclude_none=True)
        except TypeError:
            dumped = dict_method()
        if isinstance(dumped, dict):
            return dumped

    return None


def _extract_metadata_usage_object(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        metadata = standard_logging_object.get("metadata")
        if isinstance(metadata, dict):
            usage_object = metadata.get("usage_object")
            if isinstance(usage_object, dict) and usage_object:
                return dict(usage_object)
            reconstructed_usage = _build_usage_object_from_metadata(metadata)
            if reconstructed_usage is not None:
                return reconstructed_usage

    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, dict):
            usage_object = metadata.get("usage_object")
            if isinstance(usage_object, dict) and usage_object:
                return dict(usage_object)
            reconstructed_usage = _build_usage_object_from_metadata(metadata)
            if reconstructed_usage is not None:
                return reconstructed_usage

    return None


def _merge_usage_object_with_metadata(
    usage_obj: Any,
    metadata_usage_object: Optional[Dict[str, Any]],
) -> Any:
    if metadata_usage_object is None:
        return usage_obj

    usage_dict = _coerce_usage_object_to_dict(usage_obj)
    if usage_dict is None:
        return metadata_usage_object

    merged_usage = dict(usage_dict)
    for key, value in list(metadata_usage_object.items()):
        if key not in merged_usage or merged_usage.get(key) in (None, {}, []):
            merged_usage[key] = value

    return merged_usage


def _extract_usage_object(kwargs: Dict[str, Any], result: Any) -> Any:
    usage_obj = _maybe_get(result, "usage")
    metadata_usage_object = _extract_metadata_usage_object(kwargs)
    if usage_obj is not None:
        return _merge_usage_object_with_metadata(usage_obj, metadata_usage_object)

    token_count_usage = _build_usage_object_from_token_count_payload(result)
    if token_count_usage is not None:
        return _merge_usage_object_with_metadata(
            token_count_usage,
            metadata_usage_object,
        )
    token_count_usage = _build_usage_object_from_token_count_payload(_maybe_get(result, "response"))
    if token_count_usage is not None:
        return _merge_usage_object_with_metadata(
            token_count_usage,
            metadata_usage_object,
        )

    meta_obj = _maybe_get(result, "meta")
    billed_units = _maybe_get(meta_obj, "billed_units")
    token_units = _maybe_get(meta_obj, "tokens")
    if billed_units is not None:
        search_units = _safe_int(_maybe_get(billed_units, "search_units"))
        total_tokens = _safe_int(_maybe_get(billed_units, "total_tokens"))
        input_tokens = _safe_int(_maybe_get(token_units, "input_tokens"))
        prompt_tokens = total_tokens or input_tokens
        rerank_usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens or prompt_tokens,
        }
        if search_units:
            rerank_usage["search_units"] = search_units
        return _merge_usage_object_with_metadata(
            rerank_usage,
            metadata_usage_object,
        )

    completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        _maybe_get(result, "response")
    )
    if isinstance(completed_payload, dict):
        usage_obj = _maybe_get(completed_payload.get("response"), "usage")
        if usage_obj is not None:
            return _merge_usage_object_with_metadata(
                usage_obj,
                metadata_usage_object,
            )

    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        response = standard_logging_object.get("response")
        if isinstance(response, dict) and response.get("usage") is not None:
            return _merge_usage_object_with_metadata(
                response["usage"],
                metadata_usage_object,
            )
        token_count_usage = _build_usage_object_from_token_count_payload(response)
        if token_count_usage is not None:
            return _merge_usage_object_with_metadata(
                token_count_usage,
                metadata_usage_object,
            )
        token_count_usage = _build_usage_object_from_token_count_payload(standard_logging_object.get("output"))
        if token_count_usage is not None:
            return _merge_usage_object_with_metadata(
                token_count_usage,
                metadata_usage_object,
            )

    if metadata_usage_object is not None:
        return metadata_usage_object

    if isinstance(standard_logging_object, dict):
        completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
            _maybe_get(standard_logging_object.get("response"), "response")
        )
        if isinstance(completed_payload, dict):
            usage_obj = _maybe_get(completed_payload.get("response"), "usage")
            if usage_obj is not None:
                return _merge_usage_object_with_metadata(
                    usage_obj,
                    metadata_usage_object,
                )

    return None


def _enrich_token_count_usage_metadata(kwargs: Dict[str, Any], result: Any) -> None:
    metadata = _ensure_mutable_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object")
    if not isinstance(standard_logging_object, dict):
        standard_logging_object = {}

    passthrough_logging_payload = kwargs.get("passthrough_logging_payload")
    standard_passthrough_logging_payload = kwargs.get("standard_pass_through_logging_payload")
    candidates = (
        result,
        _maybe_get(result, "response"),
        standard_logging_object.get("response"),
        standard_logging_object.get("output"),
        _maybe_get_path(passthrough_logging_payload, "response_body"),
        _maybe_get_path(passthrough_logging_payload, "response"),
        _maybe_get_path(standard_passthrough_logging_payload, "response_body"),
        _maybe_get_path(standard_passthrough_logging_payload, "response"),
    )

    token_count_usage: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        token_count_usage = _build_usage_object_from_token_count_payload(candidate)
        if token_count_usage is not None:
            break
    if token_count_usage is None:
        return

    prompt_tokens = _extract_prompt_tokens(token_count_usage)
    completion_tokens = _extract_completion_tokens(token_count_usage)
    total_tokens = _extract_total_tokens(
        token_count_usage,
        prompt_tokens,
        completion_tokens,
    )
    metadata["usage_token_count_response"] = True
    metadata["usage_input_tokens"] = prompt_tokens
    metadata["usage_output_tokens"] = completion_tokens
    metadata["usage_total_tokens"] = total_tokens
    _merge_tags(metadata, ["token-count-response"])


def _extract_prompt_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "prompt_tokens"))
        or _safe_int(_maybe_get(usage_obj, "input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "input"))
        or 0
    )


def _extract_completion_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "completion_tokens"))
        or _safe_int(_maybe_get(usage_obj, "output_tokens"))
        or _safe_int(_maybe_get(usage_obj, "candidatesTokenCount"))
        or 0
    )


def _extract_total_tokens(usage_obj: Any, prompt_tokens: int, completion_tokens: int) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "total_tokens"))
        or _safe_int(_maybe_get(usage_obj, "totalTokenCount"))
        or (prompt_tokens + completion_tokens)
    )


def _extract_prompt_tokens_details(usage_obj: Any) -> Any:
    return _first_non_none(
        _maybe_get(usage_obj, "prompt_tokens_details"),
        _maybe_get(usage_obj, "input_tokens_details"),
        _maybe_get(usage_obj, "promptTokensDetails"),
        _maybe_get(usage_obj, "inputTokensDetails"),
    )


def _extract_completion_tokens_details(usage_obj: Any) -> Any:
    return _first_non_none(
        _maybe_get(usage_obj, "completion_tokens_details"),
        _maybe_get(usage_obj, "output_tokens_details"),
        _maybe_get(usage_obj, "completionTokensDetails"),
        _maybe_get(usage_obj, "outputTokensDetails"),
        _maybe_get(usage_obj, "responseTokensDetails"),
        _maybe_get(usage_obj, "candidatesTokensDetails"),
    )


def _extract_cache_read_input_tokens(usage_obj: Any) -> int:
    prompt_tokens_details = _extract_prompt_tokens_details(usage_obj)
    return (
        _safe_int(_maybe_get(usage_obj, "cache_read_input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheReadInputTokens"))
        or _safe_int(_maybe_get(usage_obj, "cachedContentTokenCount"))
        or _safe_int(_maybe_get(prompt_tokens_details, "cached_tokens"))
        or _safe_int(_maybe_get(prompt_tokens_details, "cachedTokens"))
        or 0
    )


def _extract_cache_creation_input_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "cache_creation_input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheWriteInputTokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheWriteInputTokenCount"))
        or _safe_int(_maybe_get(usage_obj, "cacheCreationInputTokens"))
        or 0
    )


def _has_nested_path(obj: Any, *keys: str) -> bool:
    sentinel = object()
    return _maybe_get_path(obj, *keys, default=sentinel) is not sentinel


def _extract_reported_reasoning_tokens(usage_obj: Any) -> Optional[int]:
    completion_tokens_details = _extract_completion_tokens_details(usage_obj)
    explicit_reasoning_tokens = _first_non_none(
        _safe_int(_maybe_get(usage_obj, "reasoning_tokens")),
        _safe_int(_maybe_get(usage_obj, "reasoningTokens")),
        _safe_int(_maybe_get(usage_obj, "reasoning_token_count")),
        _safe_int(_maybe_get(usage_obj, "thoughtsTokenCount")),
        _safe_int(_maybe_get(completion_tokens_details, "reasoning_tokens")),
        _safe_int(_maybe_get(completion_tokens_details, "reasoningTokens")),
    )
    if explicit_reasoning_tokens is not None and explicit_reasoning_tokens > 0:
        return explicit_reasoning_tokens

    modality_reasoning_counts: list[int] = []
    for details in (
        completion_tokens_details,
        _maybe_get(usage_obj, "responseTokensDetails"),
        _maybe_get(usage_obj, "candidatesTokensDetails"),
    ):
        if not isinstance(details, list):
            continue
        detail_reasoning_tokens = 0
        has_reasoning_detail = False
        for detail in details:
            modality = _maybe_get(detail, "modality")
            if not isinstance(modality, str):
                continue
            if modality.upper() not in {"THOUGHT", "REASONING"}:
                continue
            token_count = _safe_int(_maybe_get(detail, "tokenCount"))
            if token_count is None or token_count <= 0:
                continue
            detail_reasoning_tokens += token_count
            has_reasoning_detail = True
        if has_reasoning_detail:
            modality_reasoning_counts.append(detail_reasoning_tokens)

    if modality_reasoning_counts:
        return max(modality_reasoning_counts)

    return None


def _fallback_gemini_reasoning_tokens_from_signatures(metadata: Dict[str, Any], message: Any = None) -> Optional[int]:
    signature_count = _safe_int(metadata.get("gemini_thought_signature_count"))
    if signature_count is not None and signature_count > 0:
        return signature_count

    provider_specific_fields = _extract_provider_specific_fields(message) if message is not None else {}
    thought_signatures = provider_specific_fields.get("thought_signatures")
    if isinstance(thought_signatures, list):
        non_empty_signatures = [
            signature for signature in thought_signatures if isinstance(signature, str) and signature.strip()
        ]
        if non_empty_signatures:
            return len(non_empty_signatures)

    if metadata.get("gemini_thought_signature_present") is True:
        return 1
    if metadata.get("thinking_signature_present") is True:
        return 1

    return None


def _determine_reasoning_tokens_source(
    *,
    provider_reported_reasoning_tokens: Optional[int],
    reported_reasoning_tokens: Optional[int],
    estimated_reasoning_tokens: Optional[int],
    reasoning_present: bool,
) -> str:
    if provider_reported_reasoning_tokens is not None and reported_reasoning_tokens is not None:
        return "provider_reported"
    if reported_reasoning_tokens is not None:
        return "provider_signature_present"
    if estimated_reasoning_tokens is not None:
        return "estimated_from_reasoning_text"
    if reasoning_present:
        return "not_available"
    return "not_applicable"


def _estimate_reasoning_tokens(model: str, reasoning_text: str) -> Optional[int]:
    stripped_reasoning = reasoning_text.strip()
    if not stripped_reasoning:
        return None

    try:
        litellm = _get_litellm_module()
        return litellm.token_counter(
            model=model or "",
            text=stripped_reasoning,
            count_response_tokens=True,
        )
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate reasoning tokens for model=%s: %s",
            model,
            exc,
        )
        return None


def _extract_rerank_request_payload(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = (
        _extract_provider_cache_request_body(kwargs),
        kwargs,
        _maybe_get(kwargs.get("standard_logging_object"), "optional_params"),
        kwargs.get("optional_params"),
    )
    for candidate in candidates:
        if (
            isinstance(candidate, dict)
            and candidate.get("query") is not None
            and (candidate.get("documents") is not None or candidate.get("texts") is not None)
        ):
            return candidate
    return None


def _coerce_rerank_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _coerce_rerank_text(item).strip()))
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            return str(value)
    return str(value)


def _extract_rerank_document_text(
    document: Any,
    rank_fields: Optional[List[str]],
) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        if rank_fields:
            return "\n".join(
                text for field in rank_fields if (text := _coerce_rerank_text(document.get(field)).strip())
            )
        if "text" in document:
            return _coerce_rerank_text(document.get("text"))
    return _coerce_rerank_text(document)


_HOST_FUNCTION_NAMES = (
    "_build_usage_object_from_metadata",
    "_build_usage_object_from_token_count_payload",
    "_extract_responses_completed_response_from_langfuse_output",
    "_build_usage_object_from_langfuse_output",
    "_extract_codex_model_from_response_headers",
    "_session_history_metadata_model",
    "_SESSION_HISTORY_CLAUDE_MODEL_TAG_RE",
    "_session_history_model_from_request_tags",
    "_extract_model_from_langfuse_input",
    "_extract_model_from_langfuse_output",
    "_first_known_model_string",
    "_first_explicit_openrouter_model_string",
    "_coerce_usage_object_to_dict",
    "_extract_metadata_usage_object",
    "_merge_usage_object_with_metadata",
    "_extract_usage_object",
    "_enrich_token_count_usage_metadata",
    "_extract_prompt_tokens",
    "_extract_completion_tokens",
    "_extract_total_tokens",
    "_extract_prompt_tokens_details",
    "_extract_completion_tokens_details",
    "_extract_cache_read_input_tokens",
    "_extract_cache_creation_input_tokens",
    "_has_nested_path",
    "_extract_reported_reasoning_tokens",
    "_fallback_gemini_reasoning_tokens_from_signatures",
    "_determine_reasoning_tokens_source",
    "_estimate_reasoning_tokens",
    "_extract_rerank_request_payload",
    "_coerce_rerank_text",
    "_extract_rerank_document_text",
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
