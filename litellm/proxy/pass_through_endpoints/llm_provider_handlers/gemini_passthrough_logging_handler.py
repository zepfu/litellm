import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as GeminiModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    apply_passthrough_logging_contract,
)
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class GeminiPassthroughLoggingHandler:
    @staticmethod
    def _parse_stream_chunk_json(chunk: str) -> Optional[Dict[str, Any]]:
        normalized_chunk = chunk.strip()
        if normalized_chunk.startswith("data:"):
            normalized_chunk = normalized_chunk[len("data:") :].strip()
        if normalized_chunk in {"", "[DONE]"}:
            return None
        try:
            parsed_chunk = json.loads(normalized_chunk)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed_chunk, list):
            for item in reversed(parsed_chunk):
                if isinstance(item, dict):
                    parsed_chunk = item
                    break
            else:
                return None
        if not isinstance(parsed_chunk, dict):
            return None
        return parsed_chunk

    @staticmethod
    def _unwrap_code_assist_response_body(response_body: Any) -> Any:
        if isinstance(response_body, list):
            for item in reversed(response_body):
                unwrapped_item = GeminiPassthroughLoggingHandler._unwrap_code_assist_response_body(
                    item
                )
                if isinstance(unwrapped_item, dict):
                    return unwrapped_item
            return response_body
        if isinstance(response_body, dict) and isinstance(response_body.get("response"), dict):
            return response_body["response"]
        return response_body

    @staticmethod
    def _response_for_transform(
        httpx_response: httpx.Response,
        response_body: Any,
    ) -> httpx.Response:
        unwrapped_response = GeminiPassthroughLoggingHandler._unwrap_code_assist_response_body(
            response_body
        )
        if unwrapped_response is response_body:
            return httpx_response

        sanitized_headers = dict(httpx_response.headers)
        for header_name in (
            "content-encoding",
            "transfer-encoding",
            "content-length",
        ):
            sanitized_headers.pop(header_name, None)

        return httpx.Response(
            status_code=httpx_response.status_code,
            headers=sanitized_headers,
            content=json.dumps(unwrapped_response).encode("utf-8"),
            request=getattr(httpx_response, "request", None),
        )

    @staticmethod
    def _extract_usage_object_from_response_body(
        response_body: Any,
    ) -> Optional[Dict[str, Any]]:
        unwrapped_response = GeminiPassthroughLoggingHandler._unwrap_code_assist_response_body(
            response_body
        )
        if not isinstance(unwrapped_response, dict):
            return None
        usage_metadata = unwrapped_response.get("usageMetadata")
        if not isinstance(usage_metadata, dict) or not usage_metadata:
            return None
        return dict(usage_metadata)

    @staticmethod
    def _extract_usage_object_from_stream_chunks(
        all_chunks: List[str],
    ) -> Optional[Dict[str, Any]]:
        usage_object: Optional[Dict[str, Any]] = None
        for chunk in all_chunks:
            parsed_chunk = GeminiPassthroughLoggingHandler._parse_stream_chunk_json(
                chunk
            )
            if parsed_chunk is None:
                continue
            extracted_usage = GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
                parsed_chunk
            )
            if extracted_usage:
                usage_object = extracted_usage
        return usage_object

    @staticmethod
    def _store_usage_object_in_kwargs(
        kwargs: dict,
        usage_object: Optional[Dict[str, Any]],
    ) -> dict:
        if not isinstance(usage_object, dict) or not usage_object:
            return kwargs

        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
            kwargs["litellm_params"] = litellm_params
        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            litellm_params["metadata"] = metadata
        metadata["usage_object"] = dict(usage_object)

        standard_logging_object = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            standard_logging_metadata = standard_logging_object.get("metadata")
            if not isinstance(standard_logging_metadata, dict):
                standard_logging_metadata = {}
                standard_logging_object["metadata"] = standard_logging_metadata
            standard_logging_metadata["usage_object"] = dict(usage_object)

        return kwargs

    @staticmethod
    def gemini_passthrough_handler(
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
        if "predictLongRunning" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            gemini_video_config = GeminiVideoConfig()
            litellm_video_response = (
                gemini_video_config.transform_video_create_response(
                    model=model,
                    raw_response=httpx_response,
                    logging_obj=logging_obj,
                    custom_llm_provider="gemini",
                    request_data=request_body,
                )
            )
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "gemini"
            logging_obj.custom_llm_provider = "gemini"

            response_cost = litellm.completion_cost(
                completion_response=litellm_video_response,
                model=model,
                custom_llm_provider="gemini",
                call_type="create_video",
            )

            # Set response_cost in _hidden_params to prevent recalculation
            if not hasattr(litellm_video_response, "_hidden_params"):
                litellm_video_response._hidden_params = {}
            litellm_video_response._hidden_params["response_cost"] = response_cost

            apply_passthrough_logging_contract(
                litellm_response=litellm_video_response,
                model=model,
                kwargs=kwargs,
                logging_obj=logging_obj,
                response_cost=response_cost,
                custom_llm_provider="gemini",
                set_response_id=True,
            )
            return {
                "result": litellm_video_response,
                "kwargs": kwargs,
            }

        if "generateContent" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(
                url=url_route,
                request_body=request_body,
            )
            transformed_httpx_response = GeminiPassthroughLoggingHandler._response_for_transform(
                httpx_response=httpx_response,
                response_body=response_body,
            )

            # Use Gemini config for transformation
            instance_of_gemini_llm = litellm.GoogleAIStudioGeminiConfig()
            litellm_model_response: ModelResponse = (
                instance_of_gemini_llm.transform_response(
                    model=model,
                    messages=[
                        {"role": "user", "content": "no-message-pass-through-endpoint"}
                    ],
                    raw_response=transformed_httpx_response,
                    model_response=litellm.ModelResponse(),
                    logging_obj=logging_obj,
                    optional_params={},
                    litellm_params={},
                    api_key="",
                    request_data={},
                    encoding=litellm.encoding,
                )
            )
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
                usage_object=GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
                    response_body
                ),
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def _handle_logging_gemini_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        model: Optional[str],
        end_time: datetime,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Gemini passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs = kwargs or {}
        model = model or GeminiPassthroughLoggingHandler.extract_model_from_url(
            url=url_route,
            request_body=request_body,
        )
        complete_streaming_response = (
            GeminiPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
                url_route=url_route,
            )
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Gemini passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="gemini",
            usage_object=GeminiPassthroughLoggingHandler._extract_usage_object_from_stream_chunks(
                all_chunks
            ),
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        parsed_chunks = []
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            parsed_json_chunks = []
            for chunk in all_chunks:
                parsed_chunk = GeminiPassthroughLoggingHandler._parse_stream_chunk_json(
                    chunk
                )
                if parsed_chunk is not None:
                    parsed_json_chunks.append(parsed_chunk)

            code_assist_chunks = [
                chunk.get("response")
                for chunk in parsed_json_chunks
                if isinstance(chunk, dict) and isinstance(chunk.get("response"), dict)
            ]
            if len(code_assist_chunks) > 0:
                if len(code_assist_chunks) > 1:
                    complete_streaming_response = (
                        GeminiPassthroughLoggingHandler._build_complete_response_from_gemini_stream_chunks(
                            all_chunks=[
                                json.dumps(chunk) for chunk in code_assist_chunks
                            ],
                            litellm_logging_obj=litellm_logging_obj,
                        )
                    )
                    if complete_streaming_response is not None:
                        return complete_streaming_response

                transformed_httpx_response = httpx.Response(
                    status_code=200,
                    content=json.dumps(code_assist_chunks[-1]).encode("utf-8"),
                )
                instance_of_gemini_llm = litellm.GoogleAIStudioGeminiConfig()
                return instance_of_gemini_llm.transform_response(
                    model=model,
                    messages=[
                        {"role": "user", "content": "no-message-pass-through-endpoint"}
                    ],
                    raw_response=transformed_httpx_response,
                    model_response=litellm.ModelResponse(),
                    logging_obj=litellm_logging_obj,
                    optional_params={},
                    litellm_params={},
                    api_key="",
                    request_data={},
                    encoding=litellm.encoding,
                )

            complete_streaming_response = (
                GeminiPassthroughLoggingHandler._build_complete_response_from_gemini_stream_chunks(
                    all_chunks=all_chunks,
                    litellm_logging_obj=litellm_logging_obj,
                )
            )
            if complete_streaming_response is not None:
                return complete_streaming_response
        else:
            return None

        return None

    @staticmethod
    def _build_complete_response_from_gemini_stream_chunks(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> Optional[ModelResponse]:
        gemini_iterator: Any = GeminiModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
            logging_obj=litellm_logging_obj,
        )
        chunk_parsing_logic: Any = gemini_iterator._common_chunk_parsing_logic
        all_openai_chunks = []
        for chunk in all_chunks:
            parsed_chunk = chunk_parsing_logic(chunk)
            if parsed_chunk is not None:
                all_openai_chunks.append(parsed_chunk)
            while gemini_iterator.pending_model_response_chunks:
                all_openai_chunks.append(
                    gemini_iterator.pending_model_response_chunks.pop(0)
                )

        if (
            gemini_iterator.chunk_type == "accumulated_json"
            and gemini_iterator.accumulated_json
        ):
            parsed_chunk = gemini_iterator.handle_accumulated_json_chunk(chunk="")
            if parsed_chunk is not None:
                all_openai_chunks.append(parsed_chunk)
            while gemini_iterator.pending_model_response_chunks:
                all_openai_chunks.append(
                    gemini_iterator.pending_model_response_chunks.pop(0)
                )

        if len(all_openai_chunks) == 0:
            return None

        return litellm.stream_chunk_builder(chunks=all_openai_chunks)

    @staticmethod
    def extract_model_from_url(
        url: str, request_body: Optional[Dict[str, Any]] = None
    ) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        if request_body is not None and isinstance(request_body.get("model"), str):
            return request_body["model"]
        return "unknown"

    @staticmethod
    def _create_gemini_response_logging_payload_for_generate_content(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
        usage_object: Optional[Dict[str, Any]] = None,
    ):
        """
        Create the standard logging object for Gemini passthrough generateContent (streaming and non-streaming)
        """
        kwargs = GeminiPassthroughLoggingHandler._store_usage_object_in_kwargs(
            kwargs,
            usage_object,
        )

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="gemini",
        )

        apply_passthrough_logging_contract(
            litellm_response=litellm_model_response,
            model=model,
            kwargs=kwargs,
            logging_obj=logging_obj,
            response_cost=response_cost,
            custom_llm_provider=custom_llm_provider,
            native_usage_object=usage_object,
            set_response_id=True,
        )

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", kwargs)

        return kwargs
