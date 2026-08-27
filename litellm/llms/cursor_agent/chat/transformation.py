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

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse

from ..common_utils import (
    CURSOR_AGENT_PROVIDER,
    CURSOR_AGENT_TURN_HOST,
    CursorAgentError,
    build_run_request,
    build_turn_headers,
    resolve_access_token,
    resolve_provider_info,
    run_url,
    strip_provider_prefix,
)
from ..connect import (
    CursorConnectError,
    decode_cursor_agent_response_payloads,
    ensure_cursor_http2_available,
    encode_cursor_run_request,
    parse_cursor_agent_payloads,
    require_http2_response,
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

        Collect the framed Connect messages through `transform_response` and
        fake-stream them as OpenAI-compatible chunks.
        """
        return bool(stream)

    def _get_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[str, Optional[str]]:
        return resolve_provider_info(api_base, api_key)

    def get_supported_openai_params(self, model: str) -> List[str]:
        # AgentRunRequest has mcp_tools / conversation_history /
        # conversation_state, not OpenAI temperature / max_tokens / tool_choice.
        return [
            "stream",
            "tools",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # temperature / max_tokens / tool_choice have no AgentRunRequest field.
        # Reject them unless drop_params is set; do not copy or invent proto keys.
        unsupported: List[str] = []
        for param, value in non_default_params.items():
            if param in {"stream", "tools"}:
                optional_params[param] = value
            elif param in {"temperature", "max_tokens", "tool_choice"}:
                if not drop_params:
                    unsupported.append(param)
        if unsupported:
            raise UnsupportedParamsError(
                message=(
                    f"{CURSOR_AGENT_PROVIDER} does not support parameters: "
                    f"{unsupported}, for model={model}. AgentRunRequest has "
                    "mcp_tools / conversation_history / conversation_state, "
                    "not OpenAI temperature / max_tokens / tool_choice."
                ),
                llm_provider=CURSOR_AGENT_PROVIDER,
                model=model,
            )
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
        try:
            ensure_cursor_http2_available()
        except CursorConnectError as exc:
            raise CursorAgentError(
                status_code=exc.status_code,
                message=exc.message,
                headers=exc.headers,
            ) from exc
        access_token = resolve_access_token(api_key, allow_exchange=True)
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

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> tuple[dict, Optional[bytes]]:
        _ = optional_params, api_base, api_key, model, stream, fake_stream
        try:
            ensure_cursor_http2_available()
            return headers, encode_cursor_run_request(request_data)
        except CursorConnectError as exc:
            raise CursorAgentError(
                status_code=exc.status_code,
                message=exc.message,
                headers=exc.headers,
            ) from exc

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
        response_headers = dict(raw_response.headers)
        content_type = response_headers.get("content-type", "").lower()
        try:
            require_http2_response(raw_response)
            if "application/connect+proto" in content_type:
                payload = decode_cursor_agent_response_payloads(
                    raw_response.content
                )
            else:
                payload = raw_response.json()
        except CursorAgentError:
            raise
        except CursorConnectError as exc:
            raise CursorAgentError(
                status_code=exc.status_code,
                message=exc.message,
                headers=exc.headers,
            ) from exc
        except Exception as exc:
            raise CursorAgentError(
                status_code=raw_response.status_code,
                message=f"Failed to parse cursor_agent response: {exc}",
                headers=response_headers,
            ) from exc

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
                headers=response_headers,
            )

        if isinstance(payload, list):
            frames = payload
        elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
            frames = payload["events"]
        else:
            frames = [payload]
        result = parse_cursor_agent_payloads(
            frame for frame in frames if isinstance(frame, dict)
        )
        try:
            result.validate_terminal()
        except CursorConnectError as exc:
            raise CursorAgentError(
                status_code=exc.status_code,
                message=exc.message,
                headers=response_headers,
            ) from exc

        model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=result.text,
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
