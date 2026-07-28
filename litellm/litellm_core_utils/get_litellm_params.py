from typing import Any, Optional

# Pre-define optional kwargs keys as frozenset for O(1) lookups
# These are extracted from kwargs only if present, avoiding unnecessary .get() calls
_OPTIONAL_KWARGS_KEYS = frozenset(
    {
        "azure_ad_token",
        "tenant_id",
        "client_id",
        "client_secret",
        "azure_username",
        "azure_password",
        "azure_scope",
        "timeout",
        "bucket_name",
        "vertex_credentials",
        "vertex_project",
        "vertex_location",
        "vertex_ai_project",
        "vertex_ai_location",
        "vertex_ai_credentials",
        "aws_region_name",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_session_name",
        "aws_profile_name",
        "aws_role_name",
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_external_id",
        "aws_bedrock_runtime_endpoint",
        "tpm",
        "rpm",
    }
)

_XAI_OAUTH_AUTHORITATIVE_METADATA_KEYS = frozenset(
    {
        "auth_mode",
        "credential_family",
        "passthrough_route_family",
        "route_family",
        "xai_oauth_managed",
        "xai_oauth_public_model",
        "xai_oauth_upstream_model",
        "xai_quota_family",
        "shared_quota_family",
        "grok_subscription_quota_shared",
        "model_group",
        "xai_responses_previous_response_id_decoded",
        "codex_unsupported_input_item_removed_count",
        "codex_unsupported_input_item_types_removed",
        "codex_unsupported_input_items_removed",
    }
)


def _stable_union_metadata_tags(*tag_lists: Any) -> list[str]:
    tags: list[str] = []
    for tag_list in tag_lists:
        if not isinstance(tag_list, list):
            continue
        for tag in tag_list:
            if isinstance(tag, str) and tag and tag not in tags:
                tags.append(tag)
    return tags


def _is_authoritative_xai_oauth_metadata(metadata: dict) -> bool:
    public_model = metadata.get("xai_oauth_public_model")
    upstream_model = metadata.get("xai_oauth_upstream_model")
    if not isinstance(public_model, str) or not public_model.startswith("oa_xai/"):
        return False
    if upstream_model != f"xai/{public_model[len('oa_xai/') :]}":
        return False

    return (
        metadata.get("xai_oauth_managed") is True
        and metadata.get("auth_mode") == "oauth"
        and metadata.get("credential_family") == "xai_oauth"
        and metadata.get("passthrough_route_family") == "xai_oauth_api"
        and metadata.get("route_family") == "xai_oauth_api"
        and metadata.get("xai_quota_family") == "xai_grok_subscription"
        and metadata.get("shared_quota_family") == "xai_grok_subscription"
        and metadata.get("grok_subscription_quota_shared") is True
        and metadata.get("model_group") == public_model
    )


def merge_metadata_for_logging(
    metadata: Optional[dict],
    litellm_metadata: Optional[dict],
) -> Optional[dict]:
    """Merge metadata without mutation, with gated internal-field authority."""
    caller_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    internal_metadata = (
        dict(litellm_metadata) if isinstance(litellm_metadata, dict) else {}
    )
    if metadata is None and not internal_metadata:
        return None

    merged_metadata = dict(internal_metadata)
    merged_metadata.update(caller_metadata)

    if _is_authoritative_xai_oauth_metadata(internal_metadata):
        for key in _XAI_OAUTH_AUTHORITATIVE_METADATA_KEYS:
            if key in internal_metadata:
                merged_metadata[key] = internal_metadata[key]
        if isinstance(caller_metadata.get("tags"), list) or isinstance(
            internal_metadata.get("tags"), list
        ):
            merged_metadata["tags"] = _stable_union_metadata_tags(
                caller_metadata.get("tags"),
                internal_metadata.get("tags"),
            )

    return merged_metadata


def _get_base_model_from_litellm_call_metadata(
    metadata: Optional[dict],
) -> Optional[str]:
    if metadata is None:
        return None
    model_info = metadata.get("model_info")
    if model_info:
        return model_info.get("base_model")
    return None


def get_litellm_params(
    api_key: Optional[str] = None,
    force_timeout=600,
    azure=False,
    logger_fn=None,
    verbose=False,
    hugging_face=False,
    replicate=False,
    together_ai=False,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    litellm_call_id=None,
    model_alias_map=None,
    completion_call_id=None,
    metadata: Optional[dict] = None,
    model_info=None,
    proxy_server_request=None,
    acompletion=None,
    aembedding=None,
    preset_cache_key=None,
    no_log=None,
    input_cost_per_second=None,
    input_cost_per_token=None,
    output_cost_per_token=None,
    output_cost_per_second=None,
    cost_per_query=None,
    cooldown_time=None,
    text_completion=None,
    azure_ad_token_provider=None,
    user_continue_message=None,
    base_model: Optional[str] = None,
    litellm_trace_id: Optional[str] = None,
    litellm_session_id: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    custom_prompt_dict: Optional[dict] = None,
    litellm_metadata: Optional[dict] = None,
    disable_add_transform_inline_image_block: Optional[bool] = None,
    drop_params: Optional[bool] = None,
    prompt_id: Optional[str] = None,
    prompt_variables: Optional[dict] = None,
    async_call: Optional[bool] = None,
    ssl_verify: Optional[bool] = None,
    merge_reasoning_content_in_choices: Optional[bool] = None,
    use_litellm_proxy: Optional[bool] = None,
    api_version: Optional[str] = None,
    max_retries: Optional[int] = None,
    litellm_request_debug: Optional[bool] = None,
    **kwargs,
) -> dict:
    # Derive explicit session/trace ids from matching metadata keys only.
    _meta = metadata or {}
    if litellm_session_id is None:
        litellm_session_id = _meta.get("session_id")
    if litellm_trace_id is None:
        litellm_trace_id = _meta.get("trace_id")

    # Merge litellm_metadata into metadata so callbacks (e.g. Langfuse)
    # that read from litellm_params["metadata"] see API key fields even when
    # the request uses "litellm_metadata" (e.g. /v1/messages from Claude Code).
    _merged_metadata = merge_metadata_for_logging(metadata, litellm_metadata)

    # Build base dict with explicit parameters (always included)
    litellm_params = {
        "acompletion": acompletion,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "api_base": api_base,
        "litellm_call_id": litellm_call_id,
        "model_alias_map": model_alias_map,
        "completion_call_id": completion_call_id,
        "aembedding": aembedding,
        "metadata": _merged_metadata,
        "model_info": model_info,
        "proxy_server_request": proxy_server_request,
        "preset_cache_key": preset_cache_key,
        "no-log": no_log or kwargs.get("no-log"),
        "stream_response": {},  # litellm_call_id: ModelResponse Dict
        "input_cost_per_token": input_cost_per_token,
        "input_cost_per_second": input_cost_per_second,
        "output_cost_per_token": output_cost_per_token,
        "output_cost_per_second": output_cost_per_second,
        "cost_per_query": cost_per_query,
        "cooldown_time": cooldown_time,
        "text_completion": text_completion,
        "azure_ad_token_provider": azure_ad_token_provider,
        "user_continue_message": user_continue_message,
        "base_model": base_model
        or (
            _get_base_model_from_litellm_call_metadata(metadata=metadata)
            if metadata
            else None
        ),
        "litellm_trace_id": litellm_trace_id,
        "litellm_session_id": litellm_session_id,
        "hf_model_name": hf_model_name,
        "custom_prompt_dict": custom_prompt_dict,
        "litellm_metadata": litellm_metadata,
        "disable_add_transform_inline_image_block": disable_add_transform_inline_image_block,
        "drop_params": drop_params,
        "prompt_id": prompt_id,
        "prompt_variables": prompt_variables,
        "async_call": async_call,
        "ssl_verify": ssl_verify,
        "merge_reasoning_content_in_choices": merge_reasoning_content_in_choices,
        "api_version": api_version,
        "max_retries": max_retries,
        "use_litellm_proxy": use_litellm_proxy,
        "litellm_request_debug": litellm_request_debug,
    }

    # Sparse extraction: only add kwargs keys that are actually present
    if kwargs:
        for key in _OPTIONAL_KWARGS_KEYS:
            if key in kwargs:
                litellm_params[key] = kwargs[key]

    return litellm_params
