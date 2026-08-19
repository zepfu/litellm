"""
Cursor Agent CLI transformation.

Maps OpenAI chat completions onto HTTP/2 Connect
`POST /agent.v1.AgentService/Run`. This is not Cloud Agents `/cursor`
and is not openai_like.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

from ..common_utils import (
    CURSOR_AGENT_PROVIDER,
    CURSOR_AGENT_TURN_HOST,
    CursorAgentError,
    build_run_request,
    build_turn_headers,
    extract_text_from_agent_payload,
    resolve_access_token,
    resolve_provider_info,
    run_url,
    strip_provider_prefix,
)
from .streaming_iterator import CursorAgentModelResponseIterator

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CursorAgentConfig(BaseConfig):
    """Configuration for the Cursor Agent CLI Connect `Run` route."""

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return CURSOR_AGENT_PROVIDER

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        HTTP/2 `Run` is a Connect bidi stream, not OpenAI SSE.

        Collect the JSON frames through `transform_response` and fake-stream
        them. Native Connect proto framing is deferred.
        """
        return bool(stream)

    def _get_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[str, Optional[str]]:
        return resolve_provider_info(api_base, api_key)

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "tools",
            "tool_choice",
            "max_tokens",
            "temperature",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param in {"stream", "tools", "tool_choice", "max_tokens", "temperature"}:
                optional_params[param] = value
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        access_token = resolve_access_token(api_key, allow_exchange=False)
        request_id = None
        if isinstance(headers, dict):
            request_id = headers.get("x-request-id") or headers.get("X-Request-Id")
        return build_turn_headers(
            access_token,
            extra_headers=headers,
            request_id=request_id,
            http2=True,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return run_url(api_base or CURSOR_AGENT_TURN_HOST)

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if not messages:
            raise CursorAgentError(
                status_code=400,
                message="cursor_agent requires at least one message",
            )
        return build_run_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            payload = raw_response.json()
        except Exception as exc:
            raise CursorAgentError(
                status_code=raw_response.status_code,
                message=f"Failed to parse cursor_agent response: {exc}",
                headers=dict(raw_response.headers),
            )

        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            message = (
                error.get("message", "Unknown cursor_agent error")
                if isinstance(error, dict)
                else str(error)
            )
            raise CursorAgentError(
                status_code=raw_response.status_code,
                message=message,
                headers=dict(raw_response.headers),
            )

        text_parts: List[str] = []
        turn_ended = False
        if isinstance(payload, list):
            frames = payload
        elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
            frames = payload["events"]
        else:
            frames = [payload]
        for frame in frames:
            chunk_text, ended = extract_text_from_agent_payload(frame)
            if chunk_text:
                text_parts.append(chunk_text)
            if ended:
                turn_ended = True

        model_response.choices = [
            Choices(
                finish_reason="stop" if turn_ended or text_parts else "stop",
                index=0,
                message=Message(
                    content="".join(text_parts),
                    role="assistant",
                ),
            )
        ]
        model_response.model = f"{CURSOR_AGENT_PROVIDER}/{strip_provider_prefix(model)}"
        model_response.id = str(uuid.uuid4())
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator, AsyncIterator, Any],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> CursorAgentModelResponseIterator:
        return CursorAgentModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        headers_dict = dict(headers) if isinstance(headers, httpx.Headers) else headers
        return CursorAgentError(
            status_code=status_code,
            message=error_message,
            headers=headers_dict,
        )
