from datetime import datetime
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

import litellm

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardPassThroughResponseObject
from litellm.utils import executor as thread_pool_executor

from .google_code_assist_quota import (
    sanitize_google_code_assist_quota_for_logging,
)
from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
)
from .llm_provider_handlers.cursor_passthrough_logging_handler import (
    CursorPassthroughLoggingHandler,
)
from .llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)


class PassThroughEndpointLogging:
    def __init__(self):
        self.TRACKED_VERTEX_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "predict",
            "rawPredict",
            "streamRawPredict",
            "search",
            "batchPredictionJobs",
            "predictLongRunning",
        ]

        # Anthropic
        self.TRACKED_ANTHROPIC_ROUTES = ["/messages", "/v1/messages/batches"]

        # Cohere
        self.TRACKED_COHERE_ROUTES = ["/v2/chat", "/v1/embed"]
        self.assemblyai_passthrough_logging_handler = (
            AssemblyAIPassthroughLoggingHandler()
        )

        # Langfuse
        self.TRACKED_LANGFUSE_ROUTES = ["/langfuse/"]

        # Gemini
        self.TRACKED_GEMINI_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "predictLongRunning",
        ]

        # Cursor Cloud Agents
        self.TRACKED_CURSOR_ROUTES = [
            "/v0/agents",
            "/v0/me",
            "/v0/models",
            "/v0/repositories",
        ]

        # Vertex AI Live API WebSocket
        self.TRACKED_VERTEX_AI_LIVE_ROUTES = ["/vertex_ai/live"]

    async def _handle_logging(
        self,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: Union[
            StandardPassThroughResponseObject,
            PassThroughEndpointLoggingResultValues,
            dict,
        ],
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """Run pass-through logging hooks using the same callback contracts as normal LiteLLM success handling."""
        call_type = getattr(logging_obj, "call_type", "pass_through_endpoint")
        current_kwargs = dict(kwargs)
        current_kwargs.setdefault("standard_callback_dynamic_params", {})
        current_result: Union[
            StandardPassThroughResponseObject,
            PassThroughEndpointLoggingResultValues,
            dict,
        ] = standard_logging_response_object

        sync_callbacks = logging_obj.get_combined_callback_list(
            dynamic_success_callbacks=logging_obj.dynamic_success_callbacks,
            global_callbacks=litellm.success_callback,
        )
        for callback in sync_callbacks:
            logging_hook = getattr(callback, "logging_hook", None)
            if callable(logging_hook):
                try:
                    hook_result = logging_hook(
                        current_kwargs,
                        current_result,
                        call_type,
                    )
                    if (
                        isinstance(hook_result, tuple)
                        and len(hook_result) == 2
                        and isinstance(hook_result[0], dict)
                    ):
                        current_kwargs, current_result = hook_result
                except Exception as exc:
                    verbose_proxy_logger.warning(
                        "Pass-through logging_hook failed for callback=%s: %s",
                        callback,
                        exc,
                    )

        for callback in sync_callbacks:
            log_success_event = getattr(callback, "log_success_event", None)
            if callable(log_success_event):
                thread_pool_executor.submit(
                    log_success_event,
                    current_kwargs,
                    current_result,
                    start_time,
                    end_time,
                )

        async_callbacks = logging_obj.get_combined_callback_list(
            dynamic_success_callbacks=logging_obj.dynamic_async_success_callbacks,
            global_callbacks=litellm._async_success_callback,
        )
        for callback in async_callbacks:
            async_logging_hook = getattr(callback, "async_logging_hook", None)
            if callable(async_logging_hook):
                current_kwargs, current_result = await async_logging_hook(
                    kwargs=current_kwargs,
                    result=current_result,
                    call_type=call_type,
                )

            async_log_success_event = getattr(callback, "async_log_success_event", None)
            if callable(async_log_success_event):
                await async_log_success_event(
                    current_kwargs,
                    current_result,
                    start_time,
                    end_time,
                )

    def normalize_llm_passthrough_logging_payload(
        self,
        httpx_response: httpx.Response,
        response_body: Optional[dict],
        request_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        return_dict = {
            "standard_logging_response_object": None,
            "kwargs": kwargs,
        }
        standard_logging_response_object: Optional[Any] = None

        adapted_openai_url_route = self._get_adapted_openai_logging_route(
            response_body=response_body,
            custom_llm_provider=custom_llm_provider,
        )

        if self.is_gemini_route(url_route, custom_llm_provider):
            gemini_passthrough_logging_handler_result = (
                GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                gemini_passthrough_logging_handler_result["result"]
            )
            kwargs = gemini_passthrough_logging_handler_result["kwargs"]
        elif self.is_vertex_route(url_route):
            vertex_passthrough_logging_handler_result = (
                VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                    httpx_response=httpx_response,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                vertex_passthrough_logging_handler_result["result"]
            )
            kwargs = vertex_passthrough_logging_handler_result["kwargs"]
        elif adapted_openai_url_route is not None or (
            self.is_openai_route(url_route) and self._is_supported_openai_endpoint(url_route)
        ):
            from .llm_provider_handlers.openai_passthrough_logging_handler import (
                OpenAIPassthroughLoggingHandler,
            )

            openai_passthrough_logging_handler_result = (
                OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=adapted_openai_url_route or url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            )

            standard_logging_response_object = (
                openai_passthrough_logging_handler_result["result"]
            )
            kwargs = openai_passthrough_logging_handler_result["kwargs"]
        elif self.is_anthropic_route(url_route):
            anthropic_passthrough_logging_handler_result = (
                AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )

            standard_logging_response_object = (
                anthropic_passthrough_logging_handler_result["result"]
            )
            kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
        elif self.is_cohere_route(url_route):
            cohere_passthrough_logging_handler_result = (
                CoherePassthroughLoggingHandler.cohere_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                cohere_passthrough_logging_handler_result["result"]
            )
            kwargs = cohere_passthrough_logging_handler_result["kwargs"]
        elif self.is_cursor_route(url_route, custom_llm_provider):
            cursor_passthrough_logging_handler_result = (
                CursorPassthroughLoggingHandler.cursor_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                cursor_passthrough_logging_handler_result["result"]
            )
            kwargs = cursor_passthrough_logging_handler_result["kwargs"]

        return_dict["standard_logging_response_object"] = standard_logging_response_object
        return_dict["kwargs"] = kwargs
        return return_dict

    async def pass_through_async_success_handler(
        self,
        httpx_response: httpx.Response,
        response_body: Optional[dict],
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        standard_logging_response_object: Optional[
            PassThroughEndpointLoggingResultValues
        ] = None
        logging_obj.model_call_details[
            "passthrough_logging_payload"
        ] = passthrough_logging_payload
        if self.is_assemblyai_route(url_route):
            if (
                AssemblyAIPassthroughLoggingHandler._should_log_request(
                    httpx_response.request.method
                )
                is not True
            ):
                return
            self.assemblyai_passthrough_logging_handler.assemblyai_passthrough_logging_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
            return
        elif self.is_langfuse_route(url_route):
            # Don't log langfuse pass-through requests
            return
        elif custom_llm_provider == "gemini" and not self.is_gemini_route(
            url_route, custom_llm_provider
        ):
            # Gemini CLI performs Code Assist control-plane calls before model
            # generation. Most do not contain model/usage data and should not
            # create fallback session_history rows. retrieveUserQuota is the
            # exception: it contains account quota state, so log it as a
            # rate-limit observation only.
            if "retrieveUserQuota" not in url_route:
                return
            sanitized_quota = sanitize_google_code_assist_quota_for_logging(
                response_body
            )
            if not sanitized_quota:
                return
            litellm_params = kwargs.get("litellm_params")
            if not isinstance(litellm_params, dict):
                litellm_params = {}
                kwargs["litellm_params"] = litellm_params
            metadata = litellm_params.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                litellm_params["metadata"] = metadata
            metadata["google_retrieve_user_quota"] = sanitized_quota
            metadata["aawm_rate_limit_observation_only"] = True
        else:
            normalized_llm_passthrough_logging_payload = (
                self.normalize_llm_passthrough_logging_payload(
                    httpx_response=httpx_response,
                    response_body=response_body,
                    request_body=request_body,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                normalized_llm_passthrough_logging_payload[
                    "standard_logging_response_object"
                ]
            )
            kwargs = normalized_llm_passthrough_logging_payload["kwargs"]
        if standard_logging_response_object is None:
            standard_logging_response_object = StandardPassThroughResponseObject(
                response=httpx_response.text
            )

        kwargs = self._set_cost_per_request(
            logging_obj=logging_obj,
            passthrough_logging_payload=passthrough_logging_payload,
            kwargs=kwargs,
        )

        await self._handle_logging(
            logging_obj=logging_obj,
            standard_logging_response_object=standard_logging_response_object,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            standard_pass_through_logging_payload=passthrough_logging_payload,
            **kwargs,
        )

    def is_vertex_route(self, url_route: str):
        for route in self.TRACKED_VERTEX_ROUTES:
            if route in url_route:
                return True
        return False

    def is_anthropic_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        path = parsed_url.path if parsed_url.scheme else url_route
        if not path or "/count_tokens" in path:
            return False
        for route in self.TRACKED_ANTHROPIC_ROUTES:
            if route in path:
                return True
        return False

    def is_cohere_route(self, url_route: str):
        for route in self.TRACKED_COHERE_ROUTES:
            if route in url_route:
                return True

    def is_assemblyai_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        if parsed_url.hostname == "api.assemblyai.com":
            return True
        elif "/transcript" in parsed_url.path:
            return True
        return False

    def is_langfuse_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        for route in self.TRACKED_LANGFUSE_ROUTES:
            if route in parsed_url.path:
                return True
        return False

    def is_vertex_ai_live_route(self, url_route: str):
        """Check if the URL route is a Vertex AI Live API WebSocket route."""
        if not url_route:
            return False
        for route in self.TRACKED_VERTEX_AI_LIVE_ROUTES:
            if route in url_route:
                return True
        return False

    def is_cursor_route(
        self, url_route: str, custom_llm_provider: Optional[str] = None
    ):
        """Check if the URL route is a Cursor Cloud Agents API route."""
        if custom_llm_provider == "cursor":
            return True
        parsed_url = urlparse(url_route)
        if parsed_url.hostname and "api.cursor.com" in parsed_url.hostname:
            return True
        for route in self.TRACKED_CURSOR_ROUTES:
            if route in url_route:
                path = parsed_url.path if parsed_url.scheme else url_route
                if path.startswith("/v0/"):
                    return custom_llm_provider == "cursor"
        return False

    def is_openai_route(self, url_route: str):
        """Check if the URL route is an OpenAI-compatible API route."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
                or parsed_url.hostname == "integrate.api.nvidia.com"
                or parsed_url.hostname == "ai.api.nvidia.com"
                or parsed_url.hostname == "openrouter.ai"
                or parsed_url.hostname.endswith(".openrouter.ai")
            )
        )

    def is_gemini_route(
        self, url_route: str, custom_llm_provider: Optional[str] = None
    ):
        """Check if the URL route is a Gemini API route."""
        normalized_url_route = url_route.lower()
        for route in self.TRACKED_GEMINI_ROUTES:
            if route.lower() in normalized_url_route and custom_llm_provider == "gemini":
                return True
        return False

    def _get_adapted_openai_logging_route(
        self,
        response_body: Optional[dict],
        custom_llm_provider: Optional[str],
    ) -> Optional[str]:
        if custom_llm_provider not in {"openai", "openrouter", "nvidia_nim"}:
            return None
        if not isinstance(response_body, dict):
            return None

        is_chat_completions_payload = isinstance(response_body.get("choices"), list)
        is_responses_payload = response_body.get("object") == "response" or isinstance(
            response_body.get("output"), list
        )
        if not (is_chat_completions_payload or is_responses_payload):
            return None

        if custom_llm_provider == "openrouter":
            base_url = "https://openrouter.ai/api"
        elif custom_llm_provider == "nvidia_nim":
            base_url = "https://integrate.api.nvidia.com"
        else:
            base_url = "https://api.openai.com"

        if is_responses_payload:
            return f"{base_url}/v1/responses"
        return f"{base_url}/v1/chat/completions"

    def _is_supported_openai_endpoint(self, url_route: str) -> bool:
        """Check if the OpenAI endpoint is supported by the passthrough logging handler."""
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        return (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                url_route
            )
            or OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
        )

    def _set_cost_per_request(
        self,
        logging_obj: LiteLLMLoggingObj,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        kwargs: dict,
    ):
        """
        Helper function to set the cost per request in the logging object

        Only set the cost per request if it's set in the passthrough logging payload.
        If it's not set, don't set it in the logging object.
        """
        #########################################################
        # Check if cost per request is set
        #########################################################
        if passthrough_logging_payload.get("cost_per_request") is not None:
            kwargs["response_cost"] = passthrough_logging_payload.get(
                "cost_per_request"
            )
            logging_obj.model_call_details[
                "response_cost"
            ] = passthrough_logging_payload.get("cost_per_request")

        return kwargs
