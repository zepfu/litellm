import re
import time
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.cohere import CohereV2ChatResponse
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAnnotation,
    ChatCompletionToolCallChunk,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import ModelResponse, Usage, add_provider_specific_fields

from ..common_utils import CohereError
from ..common_utils import CohereV2ModelResponseIterator
from ..common_utils import validate_environment as cohere_validate_environment
from .citation_translation import translate_cohere_v2_citations

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

# Provider finish reasons are uppercase tokens. Reject anything else so a
# payload cannot carry raw response text into native_finish_reason or errors.
_SAFE_COHERE_FINISH_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

# Chat finish_reason values for ERROR/TIMEOUT are not OpenAI "stop". Choices()
# construction runs map_finish_reason, which currently collapses ERROR to stop.
_COHERE_NONSTREAM_FINISH_REASON_OUTCOMES: Dict[str, Dict[str, Any]] = {
    "COMPLETE": {"finish_reason": "stop"},
    "STOP_SEQUENCE": {"finish_reason": "stop"},
    "MAX_TOKENS": {
        "finish_reason": "length",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    "TOOL_CALL": {"finish_reason": "tool_calls"},
    "ERROR": {
        "finish_reason": "error",
        "error": {
            "code": "server_error",
            "message": "Provider generation ended with ERROR",
        },
    },
    "TIMEOUT": {
        "finish_reason": "timeout",
        "error": {
            "code": "server_error",
            "message": "Provider generation ended with TIMEOUT",
        },
    },
}


def resolve_cohere_nonstream_finish_reason(
    raw_finish_reason: Any,
) -> Optional[Dict[str, Any]]:
    """Map one Cohere nonstream finish_reason onto chat completion fields.

    Returns None when the provider value is missing or not a safe token.
    Known failures stay on error/timeout instead of stop.
    """
    if not isinstance(raw_finish_reason, str):
        return None
    token = raw_finish_reason.strip().upper()
    if not _SAFE_COHERE_FINISH_REASON.fullmatch(token):
        return None
    outcome = _COHERE_NONSTREAM_FINISH_REASON_OUTCOMES.get(token)
    if outcome is None:
        return {
            "finish_reason": map_finish_reason(token),
            "native_finish_reason": token,
        }
    resolved = deepcopy(outcome)
    resolved["native_finish_reason"] = token
    return resolved


def apply_cohere_nonstream_finish_reason(
    model_response: ModelResponse,
    raw_finish_reason: Any,
) -> None:
    """Write the nonstream finish reason onto an existing chat completion."""
    resolved = resolve_cohere_nonstream_finish_reason(raw_finish_reason)
    if resolved is None or not model_response.choices:
        return
    choice = model_response.choices[0]
    # Assign after construction. Choices.__init__ would remap ERROR to stop.
    choice.finish_reason = resolved["finish_reason"]
    existing_fields = getattr(choice, "provider_specific_fields", None)
    provider_fields = dict(existing_fields) if isinstance(existing_fields, dict) else {}
    choice.provider_specific_fields = {
        **provider_fields,
        "native_finish_reason": resolved["native_finish_reason"],
    }
    incomplete_details = resolved.get("incomplete_details")
    if incomplete_details is not None:
        model_response.incomplete_details = incomplete_details
    error = resolved.get("error")
    if error is not None:
        model_response.error = error


class CohereV2ChatConfig(OpenAIGPTConfig):
    """
    Configuration class for Cohere's API interface.

    Args:
        preamble (str, optional): When specified, the default Cohere preamble will be replaced with the provided one.
        chat_history (List[Dict[str, str]], optional): A list of previous messages between the user and the model.
        generation_id (str, optional): Unique identifier for the generated reply.
        response_id (str, optional): Unique identifier for the response.
        conversation_id (str, optional): An alternative to chat_history, creates or resumes a persisted conversation.
        prompt_truncation (str, optional): Dictates how the prompt will be constructed. Options: 'AUTO', 'AUTO_PRESERVE_ORDER', 'OFF'.
        connectors (List[Dict[str, str]], optional): List of connectors (e.g., web-search) to enrich the model's reply.
        search_queries_only (bool, optional): When true, the response will only contain a list of generated search queries.
        documents (List[Dict[str, str]], optional): A list of relevant documents that the model can cite.
        temperature (float, optional): A non-negative float that tunes the degree of randomness in generation.
        max_tokens (int, optional): The maximum number of tokens the model will generate as part of the response.
        k (int, optional): Ensures only the top k most likely tokens are considered for generation at each step.
        p (float, optional): Ensures that only the most likely tokens, with total probability mass of p, are considered for generation.
        frequency_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        presence_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        tools (List[Dict[str, str]], optional): A list of available tools (functions) that the model may suggest invoking.
        tool_results (List[Dict[str, Any]], optional): A list of results from invoking tools.
        seed (int, optional): A seed to assist reproducibility of the model's response.
    """

    preamble: Optional[str] = None
    chat_history: Optional[list] = None
    generation_id: Optional[str] = None
    response_id: Optional[str] = None
    conversation_id: Optional[str] = None
    prompt_truncation: Optional[str] = None
    connectors: Optional[list] = None
    search_queries_only: Optional[bool] = None
    documents: Optional[list] = None
    temperature: Optional[int] = None
    max_tokens: Optional[int] = None
    k: Optional[int] = None
    p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    tools: Optional[list] = None
    tool_results: Optional[list] = None
    seed: Optional[int] = None

    def __init__(
        self,
        preamble: Optional[str] = None,
        chat_history: Optional[list] = None,
        generation_id: Optional[str] = None,
        response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        prompt_truncation: Optional[str] = None,
        connectors: Optional[list] = None,
        search_queries_only: Optional[bool] = None,
        documents: Optional[list] = None,
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        k: Optional[int] = None,
        p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        tools: Optional[list] = None,
        tool_results: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

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
        return cohere_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
        )

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "tools",
            "tool_choice",
            "seed",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["num_generations"] = value
            if param == "top_p":
                optional_params["p"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "seed":
                optional_params["seed"] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Cohere v2 chat api is in openai format, so we can use the openai transform request function to transform the request.
        """
        data = super().transform_request(
            model,
            self._project_messages_for_cohere_v2(messages),
            optional_params,
            litellm_params,
            headers,
        )

        return data

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return await super().async_transform_request(
            model=model,
            messages=self._project_messages_for_cohere_v2(messages),
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @staticmethod
    def _project_messages_for_cohere_v2(
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """Remove OpenAI streaming-only tool-call indexes from request history."""
        projected_messages = deepcopy(messages)
        for message in projected_messages:
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_call.pop("index", None)
        return projected_messages

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
            raw_response_json = raw_response.json()
        except Exception:
            raise CohereError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        try:
            cohere_v2_chat_response = CohereV2ChatResponse(**raw_response_json)  # type: ignore
        except Exception:
            raise CohereError(message=raw_response.text, status_code=422)

        cohere_content = cohere_v2_chat_response["message"].get("content", None)
        if cohere_content is not None:
            model_response.choices[0].message.content = "".join(  # type: ignore
                [
                    content.get("text", "")
                    for content in cohere_content
                    if content is not None
                ]
            )

        ## ADD CITATIONS AS ANNOTATIONS OR PROVIDER METADATA
        annotations: Optional[List[ChatCompletionAnnotation]] = None
        citation_fields: Optional[dict] = None
        citations = None

        if (
            "message" in cohere_v2_chat_response
            and "citations" in cohere_v2_chat_response["message"]
        ):
            citations = cohere_v2_chat_response["message"]["citations"]

        if citations:
            annotations, citation_fields = translate_cohere_v2_citations(citations)
            if not annotations:
                annotations = None
            if not citation_fields:
                citation_fields = None

        ## Tool calling response
        cohere_tools_response = cohere_v2_chat_response["message"].get("tool_calls", [])
        current_message = model_response.choices[0].message  # type: ignore
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls: List[ChatCompletionToolCallChunk] = []
            for index, tool in enumerate(cohere_tools_response):
                tool_call: ChatCompletionToolCallChunk = {
                    **tool,  # type: ignore
                    "index": index,
                }
                tool_calls.append(tool_call)
            # Attach calls to the message that already holds assistant text so
            # Chat-to-Responses can emit that text once and one function_call
            # item per call.
            current_message.tool_calls = litellm.Message(
                tool_calls=tool_calls,
            ).tool_calls
        if annotations:
            current_message.annotations = annotations
        if citation_fields:
            existing_fields = getattr(current_message, "provider_specific_fields", None)
            merged_fields = (
                {**existing_fields, **citation_fields}
                if isinstance(existing_fields, dict)
                else citation_fields
            )
            add_provider_specific_fields(current_message, merged_fields)

        if citation_fields:
            self._merge_citation_provider_fields(model_response, citation_fields)

        ## CALCULATING USAGE - use cohere `billed_units` for returning usage
        token_usage = cohere_v2_chat_response["usage"].get("tokens", {})
        prompt_tokens = token_usage.get("input_tokens", 0)
        completion_tokens = token_usage.get("output_tokens", 0)

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        apply_cohere_nonstream_finish_reason(
            model_response=model_response,
            raw_finish_reason=cohere_v2_chat_response.get("finish_reason"),
        )
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return CohereV2ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
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
        """
        Get the complete URL for Cohere v2 chat completion.
        The api_base should already include the full path.
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CohereError(status_code=status_code, message=error_message)

    @staticmethod
    def _merge_citation_provider_fields(
        model_response: ModelResponse,
        citation_fields: dict,
    ) -> None:
        """Publish filtered citation metadata where Responses conversion reads it."""
        hidden_params = getattr(model_response, "_hidden_params", None)
        if isinstance(hidden_params, dict):
            hidden_params = {**hidden_params}
        else:
            hidden_params = {}
        existing_fields = hidden_params.get("provider_specific_fields")
        if isinstance(existing_fields, dict):
            provider_fields = {**existing_fields, **citation_fields}
        else:
            provider_fields = {**citation_fields}
        hidden_params["provider_specific_fields"] = provider_fields
        model_response._hidden_params = hidden_params

    def _translate_citations_to_openai_annotations(
        self, citations: List[dict]
    ) -> List[ChatCompletionAnnotation]:
        """
        Transform Cohere citations to OpenAI url_citation annotations.

        Document sources become annotations only when they already have a real
        HTTP(S) URL, a title, and offsets. Tool sources are not forced into
        that shape; ``translate_cohere_v2_citations`` keeps their ids and
        offsets in provider metadata without tool-output bodies or fabricated URLs.
        """
        annotations, _citation_fields = translate_cohere_v2_citations(citations)
        return annotations
