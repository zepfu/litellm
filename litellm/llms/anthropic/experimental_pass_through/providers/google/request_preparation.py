"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Optional

import httpx
from fastapi import Request

import litellm


@dataclass(frozen=True)
class Runtime:
    _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER: Any
    _add_route_family_logging_metadata: Any
    _apply_google_adapter_request_shape_policy: Any
    _build_code_assist_adapter_native_headers: Any
    _build_google_code_assist_request_from_completion_kwargs: Any
    _build_langfuse_span_descriptor: Any
    _compact_google_adapter_persisted_output_in_anthropic_request_body: Any
    _get_code_assist_adapter_target_base: Any
    _get_google_adapter_rate_limit_key: Any
    _get_or_load_google_code_assist_project: Any
    _load_valid_local_antigravity_access_token: Any
    _load_valid_local_google_oauth_access_token: Any
    _log_google_completion_adapter_debug: Any
    _merge_litellm_metadata: Any
    _normalize_google_completion_adapter_model_name: Any
    _prime_google_code_assist_session: Any
    _sanitize_google_code_assist_request_schemas: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER=namespace["_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER"],
        _add_route_family_logging_metadata=namespace["_add_route_family_logging_metadata"],
        _apply_google_adapter_request_shape_policy=namespace["_apply_google_adapter_request_shape_policy"],
        _build_code_assist_adapter_native_headers=namespace["_build_code_assist_adapter_native_headers"],
        _build_google_code_assist_request_from_completion_kwargs=namespace[
            "_build_google_code_assist_request_from_completion_kwargs"
        ],
        _build_langfuse_span_descriptor=namespace["_build_langfuse_span_descriptor"],
        _compact_google_adapter_persisted_output_in_anthropic_request_body=namespace[
            "_compact_google_adapter_persisted_output_in_anthropic_request_body"
        ],
        _get_code_assist_adapter_target_base=namespace["_get_code_assist_adapter_target_base"],
        _get_google_adapter_rate_limit_key=namespace["_get_google_adapter_rate_limit_key"],
        _get_or_load_google_code_assist_project=namespace["_get_or_load_google_code_assist_project"],
        _load_valid_local_antigravity_access_token=namespace["_load_valid_local_antigravity_access_token"],
        _load_valid_local_google_oauth_access_token=namespace["_load_valid_local_google_oauth_access_token"],
        _log_google_completion_adapter_debug=namespace["_log_google_completion_adapter_debug"],
        _merge_litellm_metadata=namespace["_merge_litellm_metadata"],
        _normalize_google_completion_adapter_model_name=namespace["_normalize_google_completion_adapter_model_name"],
        _prime_google_code_assist_session=namespace["_prime_google_code_assist_session"],
        _sanitize_google_code_assist_request_schemas=namespace["_sanitize_google_code_assist_request_schemas"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


async def _prepare_anthropic_google_completion_adapter_request(
    *,
    request: Request,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> SimpleNamespace:
    google_access_token = (
        await _runtime()._load_valid_local_antigravity_access_token()
        if adapter_provider == _runtime()._ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
        else await _runtime()._load_valid_local_google_oauth_access_token()
    )
    google_project = await _runtime()._get_or_load_google_code_assist_project(
        google_access_token,
        adapter_provider=adapter_provider,
    )
    google_quota_observation = await _runtime()._prime_google_code_assist_session(
        google_access_token,
        google_project,
        adapter_provider=adapter_provider,
    )

    is_antigravity_adapter = adapter_provider == _runtime()._ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
    route_family = (
        "anthropic_antigravity_completion_adapter" if is_antigravity_adapter else "anthropic_google_completion_adapter"
    )
    adapter_tag = (
        "anthropic-antigravity-completion-adapter" if is_antigravity_adapter else "anthropic-google-completion-adapter"
    )
    target_provider_label = "antigravity" if is_antigravity_adapter else "google"
    requested_model = prepared_request_body.get("model")
    google_target_base = _runtime()._get_code_assist_adapter_target_base(adapter_provider)
    google_model = _runtime()._normalize_google_completion_adapter_model_name(adapter_model)
    google_adapter_rate_limit_key = _runtime()._get_google_adapter_rate_limit_key(
        google_model,
        access_token=google_access_token,
        companion_project=google_project,
    )
    if is_antigravity_adapter:
        google_adapter_rate_limit_key = f"antigravity:{google_adapter_rate_limit_key}"
    client_requested_stream = bool(prepared_request_body.get("stream"))
    is_stream = True
    target_endpoint_label = "/v1internal:streamGenerateContent"
    target_query_params = {"alt": "sse"}
    target_url = f"{google_target_base.rstrip('/')}{target_endpoint_label}"
    annotated_target_url = (
        httpx.URL(target_url).copy_with(params=target_query_params) if target_query_params else httpx.URL(target_url)
    )

    (
        prepared_request_body,
        google_persisted_output_compacted_count,
        google_persisted_output_hooks,
        google_persisted_output_metadata,
    ) = _runtime()._compact_google_adapter_persisted_output_in_anthropic_request_body(prepared_request_body)

    prepared_request_body = _runtime()._merge_litellm_metadata(
        _runtime()._add_route_family_logging_metadata(prepared_request_body, route_family),
        tags_to_add=[
            adapter_tag,
            f"anthropic-adapter-model:{google_model}",
            f"anthropic-adapter-target:{target_provider_label}:{target_endpoint_label}",
            *(
                [
                    "google-adapter-persisted-output-compacted",
                    *[
                        f"google-adapter-persisted-output-hook:{hook}"
                        for hook in sorted(google_persisted_output_hooks)
                        if hook
                    ],
                ]
                if google_persisted_output_compacted_count
                else []
            ),
        ],
        extra_fields={
            "anthropic_adapter_model": google_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_provider": adapter_provider,
            "anthropic_adapter_target_endpoint": (f"{target_provider_label}:{target_endpoint_label}"),
            **({"antigravity_code_assist": True} if is_antigravity_adapter else {}),
            "google_adapter_persisted_output_compacted": bool(google_persisted_output_compacted_count),
            "google_adapter_persisted_output_compacted_count": google_persisted_output_compacted_count,
            "google_adapter_persisted_output_hooks": sorted(google_persisted_output_hooks),
            "google_adapter_persisted_output_metadata": google_persisted_output_metadata,
            **({"google_retrieve_user_quota": google_quota_observation} if google_quota_observation else {}),
            "langfuse_spans": [
                _runtime()._build_langfuse_span_descriptor(
                    name=(
                        "anthropic.antigravity_completion_adapter"
                        if is_antigravity_adapter
                        else "anthropic.google_completion_adapter"
                    ),
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": google_model,
                        "adapter_provider": adapter_provider,
                        "stream": client_requested_stream,
                        "upstream_stream": True,
                        "persisted_output_compacted_count": google_persisted_output_compacted_count,
                    },
                )
            ],
        },
    )

    (
        wrapped_request_body,
        tool_name_mapping,
        completion_messages,
        gemini_optional_params,
        litellm_params,
        completion_message_window_changes,
    ) = await _runtime()._build_google_code_assist_request_from_completion_kwargs(
        completion_kwargs=prepared_request_body,
        adapter_model=google_model,
        project=google_project,
        request=request,
    )
    if isinstance(prepared_request_body.get("litellm_metadata"), dict):
        wrapped_request_body["litellm_metadata"] = {
            **dict(wrapped_request_body.get("litellm_metadata") or {}),
            **dict(prepared_request_body["litellm_metadata"]),
        }

    generation_policy_changes = _runtime()._apply_google_adapter_request_shape_policy(wrapped_request_body)

    adapter_headers = _runtime()._build_code_assist_adapter_native_headers(
        adapter_provider=adapter_provider,
        access_token=google_access_token,
        model=google_model,
        accept="*/*",
    )
    if isinstance(wrapped_request_body.get("litellm_metadata"), dict):
        if completion_message_window_changes:
            wrapped_request_body["litellm_metadata"][
                "google_adapter_completion_message_window"
            ] = completion_message_window_changes
        if generation_policy_changes:
            wrapped_request_body["litellm_metadata"]["google_adapter_request_shape_policy"] = generation_policy_changes

    sanitized_schema_fix_count = _runtime()._sanitize_google_code_assist_request_schemas(wrapped_request_body)
    _runtime()._log_google_completion_adapter_debug(
        prepared_request_body=prepared_request_body,
        wrapped_request_body=wrapped_request_body,
        google_model=google_model,
        adapter_headers=adapter_headers,
        sanitized_schema_fix_count=sanitized_schema_fix_count,
        generation_policy_changes=generation_policy_changes,
    )

    return SimpleNamespace(
        adapter_headers=adapter_headers,
        annotated_target_url=annotated_target_url,
        client_requested_stream=client_requested_stream,
        completion_messages=completion_messages,
        gemini_optional_params=gemini_optional_params,
        google_adapter_rate_limit_key=google_adapter_rate_limit_key,
        google_model=google_model,
        is_stream=is_stream,
        litellm_params=litellm_params,
        custom_llm_provider=adapter_provider,
        target_query_params=target_query_params,
        target_url=target_url,
        tool_name_mapping=tool_name_mapping,
        wrapped_request_body=wrapped_request_body,
    )
