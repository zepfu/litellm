"""
OpenAI Passthrough Logging Handler

Handles cost tracking and logging for OpenAI passthrough endpoints, specifically /chat/completions.
"""

from datetime import datetime
from typing import List, Optional, Union
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.llms.openai.openai import OpenAIConfig as OpenAIConfigType
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import (
    Choices,
    ImageResponse,
    LlmProviders,
    Message,
    PassthroughCallTypes,
)
from litellm.utils import ModelResponse, TextCompletionResponse


class OpenAIPassthroughLoggingHandler(BasePassthroughLoggingHandler):
    """
    OpenAI-specific passthrough logging handler that provides cost tracking for /chat/completions endpoints.
    """

    @property
    def llm_provider_name(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def get_provider_config(self, model: str) -> OpenAIConfigType:
        """Get OpenAI provider configuration for the given model."""
        return OpenAIConfig()

    @staticmethod
    def is_openai_chat_completions_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI chat completions endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
                or "chatgpt.com" in parsed_url.hostname
            )
            and "/v1/chat/completions" in parsed_url.path
        )

    @staticmethod
    def is_openai_image_generation_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image generation endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
                or "chatgpt.com" in parsed_url.hostname
            )
            and "/v1/images/generations" in parsed_url.path
        )

    @staticmethod
    def is_openai_image_editing_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image editing endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
                or "chatgpt.com" in parsed_url.hostname
            )
            and "/v1/images/edits" in parsed_url.path
        )

    @staticmethod
    def is_openai_responses_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI responses API endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
                or "chatgpt.com" in parsed_url.hostname
            )
            and ("/v1/responses" in parsed_url.path or "/responses" in parsed_url.path)
        )

    def _get_user_from_metadata(
        self,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        """Extract user information from passthrough logging payload."""
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return request_body.get("user")
        return None

    @staticmethod
    def _append_langfuse_span_to_kwargs(
        kwargs: dict,
        *,
        name: str,
        span_metadata: Optional[dict] = None,
    ) -> None:
        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        langfuse_spans = metadata.get("langfuse_spans") or []
        if not isinstance(langfuse_spans, list):
            langfuse_spans = []
        descriptor = {"name": name}
        if span_metadata:
            descriptor["metadata"] = span_metadata
        langfuse_spans.append(descriptor)
        metadata["langfuse_spans"] = langfuse_spans
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params

    @staticmethod
    def _calculate_image_generation_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image generation."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")
            quality = request_body.get("quality", None)

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=quality,
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Error calculating image generation cost: {str(e)}"
            )
            return 0.0

    @staticmethod
    def _calculate_image_editing_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image editing."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            # Image edit typically uses multipart/form-data (because of files), so all fields arrive as strings (e.g., n = "1").
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=None,  # Image editing doesn't have quality parameter
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Error calculating image editing cost: {str(e)}"
            )
            return 0.0

    @staticmethod
    def openai_passthrough_handler(  # noqa: PLR0915
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle OpenAI passthrough logging with cost tracking for chat completions, image generation, image editing, and responses API.
        """
        # Check if this is a supported endpoint for cost tracking
        is_chat_completions = (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
        )
        is_image_generation = (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route)
        )
        is_image_editing = (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
        )
        is_responses = OpenAIPassthroughLoggingHandler.is_openai_responses_route(
            url_route
        )

        if not (
            is_chat_completions
            or is_image_generation
            or is_image_editing
            or is_responses
        ):
            # For unsupported endpoints, return None to let the system fall back to generic behavior
            return {
                "result": None,
                "kwargs": kwargs,
            }

        # Extract model from request or response
        model = request_body.get("model", response_body.get("model", ""))
        if not model:
            verbose_proxy_logger.warning(
                "No model found in request or response for OpenAI passthrough cost tracking"
            )
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

        try:
            response_cost = 0.0
            litellm_model_response: Optional[
                Union[ModelResponse, TextCompletionResponse, ImageResponse]
            ] = None
            handler_instance = OpenAIPassthroughLoggingHandler()

            custom_llm_provider = kwargs.get("custom_llm_provider", "openai")

            if is_chat_completions:
                # Handle chat completions with existing logic
                provider_config = handler_instance.get_provider_config(model=model)
                # Preserve existing litellm_params to maintain metadata tags
                existing_litellm_params = kwargs.get("litellm_params", {}) or {}
                litellm_model_response = provider_config.transform_response(
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    model=model,
                    messages=request_body.get("messages", []),
                    logging_obj=logging_obj,
                    optional_params=request_body.get("optional_params", {}),
                    api_key="",
                    request_data=request_body,
                    encoding=litellm.encoding,
                    json_mode=request_body.get("response_format", {}).get("type")
                    == "json_object",
                    litellm_params=existing_litellm_params,
                )

                # Calculate cost using LiteLLM's cost calculator
                response_cost = litellm.completion_cost(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
            elif is_image_generation:
                # Handle image generation cost calculation
                response_cost = (
                    OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
                        model=model,
                        response_body=response_body,
                        request_body=request_body,
                    )
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = (
                        PassthroughCallTypes.passthrough_image_generation.value
                    )
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_image_editing:
                # Handle image editing cost calculation
                response_cost = (
                    OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
                        model=model,
                        response_body=response_body,
                        request_body=request_body,
                    )
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = (
                        PassthroughCallTypes.passthrough_image_generation.value
                    )
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_responses:
                # Handle responses API cost calculation
                provider_config = handler_instance.get_provider_config(model=model)
                existing_litellm_params = kwargs.get("litellm_params", {}) or {}
                litellm_model_response = provider_config.transform_response(
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    model=model,
                    messages=request_body.get("messages", []),
                    logging_obj=logging_obj,
                    optional_params=request_body.get("optional_params", {}),
                    api_key="",
                    request_data=request_body,
                    encoding=litellm.encoding,
                    json_mode=False,
                    litellm_params=existing_litellm_params,
                )

                # Calculate cost using LiteLLM's cost calculator with responses call type
                response_cost = litellm.completion_cost(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    call_type="responses",
                )

            # Update kwargs with cost information
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider

            # Extract user information for tracking
            passthrough_logging_payload: Optional[
                PassthroughStandardLoggingPayload
            ] = kwargs.get("passthrough_logging_payload")
            if passthrough_logging_payload:
                user = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs["litellm_params"].setdefault(
                        "proxy_server_request", {}
                    ).setdefault("body", {})["user"] = user
                request_headers = passthrough_logging_payload.get("request_headers")
                if request_headers:
                    kwargs["litellm_params"].setdefault(
                        "proxy_server_request", {}
                    )["headers"] = request_headers

            # Create standard logging object
            if litellm_model_response is not None:
                if (
                    is_responses
                    and (
                        (kwargs.get("litellm_params", {}) or {}).get("metadata", {})
                        or {}
                    ).get("passthrough_route_family")
                    == "codex_responses"
                ):
                    usage = getattr(litellm_model_response, "usage", None)
                    handler_instance._append_langfuse_span_to_kwargs(
                        kwargs,
                        name="codex.usage_normalize",
                        span_metadata={
                            "streaming": False,
                            "call_type": "responses",
                            "total_tokens": getattr(usage, "total_tokens", None),
                            "response_cost": response_cost,
                        },
                    )
                kwargs["standard_logging_object"] = get_standard_logging_object_payload(
                    kwargs=kwargs,
                    init_response_obj=litellm_model_response,
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=logging_obj,
                    status="success",
                )

            # Update logging object with cost information
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["response_cost"] = response_cost

            endpoint_type = (
                "chat_completions"
                if is_chat_completions
                else "image_generation"
                if is_image_generation
                else "image_editing"
            )
            verbose_proxy_logger.debug(
                f"OpenAI passthrough cost tracking - Endpoint: {endpoint_type}, Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in OpenAI passthrough cost tracking: {str(e)}"
            )
            # Fall back to base handler without cost tracking
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

    def _build_complete_streaming_response(
        self,
        all_chunks: list,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
        request_body: Optional[dict] = None,
        litellm_params: Optional[dict] = None,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw chunks for OpenAI streaming responses.

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        try:
            if self.is_openai_responses_route(url_route):
                return self._build_complete_streaming_responses_api_response(
                    all_chunks=all_chunks,
                    litellm_logging_obj=litellm_logging_obj,
                    model=model,
                    request_body=request_body or {},
                    litellm_params=litellm_params or {},
                )
            # OpenAI's response iterator to parse chunks
            from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator

            openai_iterator = OpenAIChatCompletionResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )

            all_openai_chunks = []
            for chunk_str in all_chunks:
                try:
                    # Parse the string chunk using the base iterator's string parser
                    from litellm.llms.base_llm.base_model_iterator import (
                        BaseModelResponseIterator,
                    )

                    # Convert string chunk to dict
                    stripped_json_chunk = (
                        BaseModelResponseIterator._string_to_dict_parser(
                            str_line=chunk_str
                        )
                    )

                    if stripped_json_chunk:
                        # Parse the chunk using OpenAI's chunk parser
                        transformed_chunk = openai_iterator.chunk_parser(
                            chunk=stripped_json_chunk
                        )
                        if transformed_chunk is not None:
                            all_openai_chunks.append(transformed_chunk)

                except (StopIteration, StopAsyncIteration, Exception) as e:
                    verbose_proxy_logger.debug(f"Error parsing streaming chunk: {e}")
                    continue

            if not all_openai_chunks:
                verbose_proxy_logger.warning(
                    "No valid chunks found in streaming response"
                )
                return None

            # Build complete response from chunks
            complete_streaming_response = litellm.stream_chunk_builder(
                chunks=all_openai_chunks
            )

            return complete_streaming_response

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error building complete streaming response: {str(e)}"
            )
            return None

    def _build_complete_streaming_responses_api_response(
        self,
        all_chunks: list,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        request_body: dict,
        litellm_params: dict,
    ) -> Optional[ModelResponse]:
        """
        Rebuild a complete response from Responses API streaming events.

        Native Codex passthrough streams the ChatGPT backend's `/responses` SSE shape.
        Replaying those chunks through the chat-completions stream builder drops the
        `response.completed.response.usage` payload, which then zeros out cost logging.
        """
        try:
            from litellm.llms.base_llm.base_model_iterator import (
                BaseModelResponseIterator,
            )

            responses_config = OpenAIResponsesAPIConfig()
            responses_transformer = LiteLLMResponsesTransformationHandler()
            completed_response = None
            output_text_parts: List[str] = []

            for chunk_str in all_chunks:
                parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(
                    str_line=chunk_str
                )
                if not parsed_chunk:
                    continue
                if parsed_chunk.get("type") == "response.output_text.delta":
                    delta = parsed_chunk.get("delta")
                    if isinstance(delta, str):
                        output_text_parts.append(delta)
                transformed_chunk = responses_config.transform_streaming_response(
                    model=model,
                    parsed_chunk=parsed_chunk,
                    logging_obj=litellm_logging_obj,
                )
                if getattr(transformed_chunk, "type", None) == "response.completed":
                    completed_response = getattr(transformed_chunk, "response", None)

            if completed_response is None:
                verbose_proxy_logger.warning(
                    "No response.completed event found in OpenAI responses stream"
                )
                return None

            if len(getattr(completed_response, "output", []) or []) == 0:
                # Codex streams can emit text only through response.output_text.delta
                # while leaving response.completed.output empty.
                model_response = litellm.ModelResponse()
                model_response.model = model
                model_response.choices = [
                    Choices(
                        message=Message(
                            role="assistant", content="".join(output_text_parts)
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ]
                model_response.usage = (
                    ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                        completed_response.usage
                    )
                )
                raw_response_hidden_params = getattr(
                    completed_response, "_hidden_params", {}
                )
                if raw_response_hidden_params:
                    model_response._hidden_params.update(raw_response_hidden_params)
                return model_response

            return responses_transformer.transform_response(
                model=model,
                raw_response=completed_response,
                model_response=litellm.ModelResponse(),
                logging_obj=litellm_logging_obj,
                request_data=request_body,
                messages=request_body.get("messages", []),
                optional_params=request_body.get("optional_params", {}),
                litellm_params=litellm_params,
                encoding=litellm.encoding,
                json_mode=False,
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error rebuilding complete responses API stream: {str(e)}"
            )
            return None

    @staticmethod
    def _handle_logging_openai_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle logging for collected OpenAI streaming chunks with cost tracking.
        """
        try:
            # Extract model from request body
            model = request_body.get("model", "gpt-4o")

            # Build complete response from chunks using our streaming handler
            handler = OpenAIPassthroughLoggingHandler()
            handler_instance = handler
            is_responses = handler.is_openai_responses_route(url_route)
            existing_litellm_params = (
                litellm_logging_obj.model_call_details.get("litellm_params", {}) or {}
            )
            complete_response = handler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
                url_route=url_route,
                request_body=request_body,
                litellm_params=existing_litellm_params,
            )

            if complete_response is None:
                verbose_proxy_logger.warning(
                    "Failed to build complete response from OpenAI streaming chunks"
                )
                return {
                    "result": None,
                    "kwargs": {},
                }

            custom_llm_provider = litellm_logging_obj.model_call_details.get(
                "custom_llm_provider", "openai"
            )
            # Calculate cost using LiteLLM's cost calculator
            response_cost = litellm.completion_cost(
                completion_response=complete_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
                call_type="responses" if is_responses else None,
            )

            # Preserve existing litellm_params to maintain metadata tags
            # Prepare kwargs for logging
            kwargs = {
                "response_cost": response_cost,
                "model": model,
                "custom_llm_provider": custom_llm_provider,
                "litellm_params": existing_litellm_params.copy(),
            }

            # Extract user information for tracking
            passthrough_logging_payload: Optional[
                PassthroughStandardLoggingPayload
            ] = litellm_logging_obj.model_call_details.get(
                "passthrough_logging_payload"
            )
            if passthrough_logging_payload:
                user = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs["litellm_params"].setdefault(
                        "proxy_server_request", {}
                    ).setdefault("body", {})["user"] = user
                request_headers = passthrough_logging_payload.get("request_headers")
                if request_headers:
                    kwargs["litellm_params"].setdefault(
                        "proxy_server_request", {}
                    )["headers"] = request_headers

            # Create standard logging object
            if (
                is_responses
                and (
                    (kwargs.get("litellm_params", {}) or {}).get("metadata", {}) or {}
                ).get("passthrough_route_family")
                == "codex_responses"
            ):
                usage = getattr(complete_response, "usage", None)
                handler._append_langfuse_span_to_kwargs(
                    kwargs,
                    name="codex.usage_normalize",
                    span_metadata={
                        "streaming": True,
                        "call_type": "responses",
                        "total_tokens": getattr(usage, "total_tokens", None),
                        "response_cost": response_cost,
                    },
                )
            kwargs["standard_logging_object"] = get_standard_logging_object_payload(
                kwargs=kwargs,
                init_response_obj=complete_response,
                start_time=start_time,
                end_time=end_time,
                logging_obj=litellm_logging_obj,
                status="success",
            )

            # Update logging object with cost information
            litellm_logging_obj.model_call_details["model"] = model
            litellm_logging_obj.model_call_details[
                "custom_llm_provider"
            ] = custom_llm_provider
            litellm_logging_obj.model_call_details["response_cost"] = response_cost

            verbose_proxy_logger.debug(
                f"OpenAI streaming passthrough cost tracking - Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": complete_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in OpenAI streaming passthrough cost tracking: {str(e)}"
            )
            return {
                "result": None,
                "kwargs": {},
            }
