"""Google Code Assist request-shaping orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, TypeGuard

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload

GoogleRequestBuildResult = tuple[
    Payload,
    dict[str, str],
    list[Payload],
    Payload,
    Payload,
    Payload,
]


class GenerationPolicy(Protocol):
    def __call__(self, request_block: Payload, *, model: Optional[str]) -> Payload:
        ...


class BuildRequest(Protocol):
    async def __call__(
        self,
        *,
        completion_kwargs: Payload,
        adapter_model: str,
        project: str,
        request: object,
        completion_kwargs_are_openai_chat: bool,
        scope_key: Optional[str],
    ) -> GoogleRequestBuildResult:
        ...


class LoadAccessToken(Protocol):
    async def __call__(self, adapter_provider: str) -> str:
        ...


class GetProject(Protocol):
    async def __call__(self, access_token: str, *, adapter_provider: str) -> str:
        ...


class PrimeSession(Protocol):
    async def __call__(
        self,
        access_token: str,
        companion_project: str,
        *,
        adapter_provider: str,
    ) -> Optional[Payload]:
        ...


class PrepareAdapterRequest(Protocol):
    async def __call__(
        self,
        *,
        request: object,
        prepared_request_body: Payload,
        adapter_model: str,
        adapter_provider: str,
        access_token: str,
        companion_project: str,
        quota_observation: Optional[Payload],
        input_shape: str,
    ) -> object:
        ...


class CollectModelResponse(Protocol):
    async def __call__(
        self,
        *,
        response: object,
        adapter_model: str,
        logging_obj: object,
    ) -> object:
        ...


class TranslateAnthropicResponse(Protocol):
    def __call__(
        self,
        model_response: object,
        *,
        tool_name_mapping: dict[str, str],
    ) -> object:
        ...


@dataclass(frozen=True)
class Runtime:
    """Injected low-level transforms and route services."""

    split_inline_context_and_prompt: Callable[[Payload], Payload]
    compact_followup_contents: Callable[[Payload], Payload]
    trim_followup_tools: Callable[[Payload], Payload]
    compact_oversized_text_parts: Callable[[Payload], Payload]
    apply_contents_window_policy: Callable[[Payload], Payload]
    repair_function_call_adjacency: Callable[[Payload], Payload]
    apply_generation_config_policy: GenerationPolicy
    build_request: BuildRequest
    load_access_token: LoadAccessToken
    get_project: GetProject
    prime_session: PrimeSession
    prepare_adapter_request: PrepareAdapterRequest
    collect_model_response: CollectModelResponse
    translate_anthropic_response: TranslateAnthropicResponse
    build_anthropic_response: Callable[[object], object]
    default_provider: str = "gemini"


def _is_payload(value: object) -> TypeGuard[Payload]:
    return isinstance(value, dict) and all(
        isinstance(key, str) for key in value
    )


def _apply_google_adapter_request_shape_policy(
    payload: Payload,
    *,
    runtime: Runtime,
) -> Payload:
    """Apply the ordered Google request-shaping pipeline."""
    request_value = payload.get("request")
    if not _is_payload(request_value):
        return {}
    request_block = request_value
    model_value = payload.get("model")
    model = model_value if isinstance(model_value, str) else None
    changes: Payload = {}
    transforms = (
        runtime.split_inline_context_and_prompt,
        runtime.compact_followup_contents,
        runtime.trim_followup_tools,
        runtime.compact_oversized_text_parts,
        runtime.apply_contents_window_policy,
        runtime.repair_function_call_adjacency,
    )
    for transform in transforms:
        transform_changes = transform(request_block)
        if transform_changes:
            changes.update(transform_changes)
    generation_changes = runtime.apply_generation_config_policy(
        request_block,
        model=model,
    )
    if generation_changes:
        changes.update(generation_changes)
    return changes


async def _build_google_code_assist_request_from_completion_kwargs(
    *,
    runtime: Runtime,
    completion_kwargs: Payload,
    adapter_model: str,
    project: str,
    request: object,
    completion_kwargs_are_openai_chat: bool = False,
    scope_key: Optional[str] = None,
) -> GoogleRequestBuildResult:
    """Run the provider-owned entry point for Google request construction."""
    result = await runtime.build_request(
        completion_kwargs=completion_kwargs,
        adapter_model=adapter_model,
        project=project,
        request=request,
        completion_kwargs_are_openai_chat=completion_kwargs_are_openai_chat,
        scope_key=scope_key,
    )
    if len(result) != 6:
        raise ValueError("Google request construction must return the six established " "request-shaping artifacts.")
    return result


async def _prepare_provider_adapter_request(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    adapter_provider: Optional[str],
    input_shape: str,
) -> object:
    provider = adapter_provider or runtime.default_provider
    access_token = await runtime.load_access_token(provider)
    companion_project = await runtime.get_project(
        access_token,
        adapter_provider=provider,
    )
    quota_observation = await runtime.prime_session(
        access_token,
        companion_project,
        adapter_provider=provider,
    )
    return await runtime.prepare_adapter_request(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        adapter_provider=provider,
        access_token=access_token,
        companion_project=companion_project,
        quota_observation=quota_observation,
        input_shape=input_shape,
    )


async def _prepare_anthropic_google_completion_adapter_request(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    adapter_provider: Optional[str] = None,
) -> object:
    """Prepare the Anthropic-shaped Google completion route."""
    return await _prepare_provider_adapter_request(
        runtime=runtime,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        adapter_provider=adapter_provider,
        input_shape="anthropic_messages",
    )


async def _prepare_codex_google_code_assist_adapter_request(
    *,
    runtime: Runtime,
    request: object,
    prepared_request_body: Payload,
    adapter_model: str,
    adapter_provider: Optional[str] = None,
) -> object:
    """Prepare the OpenAI Responses-shaped Google completion route."""
    return await _prepare_provider_adapter_request(
        runtime=runtime,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        adapter_provider=adapter_provider,
        input_shape="openai_responses",
    )


async def _collect_google_code_assist_response_from_stream(
    *,
    runtime: Runtime,
    response: object,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    logging_obj: object,
) -> object:
    """Collect a Google stream and translate it to an Anthropic response."""
    model_response = await runtime.collect_model_response(
        response=response,
        adapter_model=adapter_model,
        logging_obj=logging_obj,
    )
    anthropic_response = runtime.translate_anthropic_response(
        model_response,
        tool_name_mapping=tool_name_mapping,
    )
    return runtime.build_anthropic_response(anthropic_response)


__all__ = [
    "GoogleRequestBuildResult",
    "Runtime",
    "_apply_google_adapter_request_shape_policy",
    "_build_google_code_assist_request_from_completion_kwargs",
    "_collect_google_code_assist_response_from_stream",
    "_prepare_anthropic_google_completion_adapter_request",
    "_prepare_codex_google_code_assist_adapter_request",
]
