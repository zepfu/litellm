import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.cohere import CohereV2ChatResponse
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolCallChunk,
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import CohereError
from ..common_utils import CohereV2ModelResponseIterator
from ..common_utils import validate_environment as cohere_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CohereV2StrictToolsError(UnsupportedParamsError):
    """Unrepresentable Cohere function strictness. Raised before provider egress."""

    attempted_provider_call = False

    def __init__(self, message: str) -> None:
        super().__init__(message=message, llm_provider="cohere")
        self.attempted_provider_call = False


def _tool_strict_flags(tool: Any) -> List[bool]:
    """Read OpenAI strict flags without treating a missing flag as strict."""
    if not isinstance(tool, dict):
        return []
    flags: List[bool] = []
    if "strict" in tool:
        flags.append(_coerce_strict_flag(tool.get("strict")))
    function = tool.get("function")
    if isinstance(function, dict) and "strict" in function:
        flags.append(_coerce_strict_flag(function.get("strict")))
    if any(flags) and not all(flags):
        raise CohereV2StrictToolsError(
            "Cohere strict_tools cannot represent conflicting strict flags on one function."
        )
    return flags


def _coerce_strict_flag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise CohereV2StrictToolsError(
        "Cohere strict_tools cannot represent a non-boolean function strict flag."
    )


def _tool_declares_strict_key(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    if "strict" in tool:
        return True
    function = tool.get("function")
    return isinstance(function, dict) and "strict" in function


def resolve_cohere_v2_strict_tools(tools: List[Any]) -> bool:
    """Return whether every tool requested strict function semantics.

    Absent and false flags are not strict. A mix of strict and non-strict
    tools cannot be represented by Cohere's request-level ``strict_tools``
    boolean, so that combination fails before egress.
    """
    if not tools:
        return False
    requested = [
        all(flags) if flags else False
        for flags in (_tool_strict_flags(tool) for tool in tools)
    ]
    if any(requested) and not all(requested):
        raise CohereV2StrictToolsError(
            "Cohere strict_tools cannot represent mixed function strictness."
        )
    return all(requested)


def _strip_openai_function_strict(tools: List[Any]) -> List[Any]:
    """Drop OpenAI strict flags. Leave the function schema otherwise unchanged."""
    stripped_tools = deepcopy(tools)
    for tool in stripped_tools:
        if not isinstance(tool, dict):
            continue
        tool.pop("strict", None)
        function = tool.get("function")
        if isinstance(function, dict):
            function.pop("strict", None)
    return stripped_tools


def apply_cohere_v2_strict_tools(request_data: dict) -> dict:
    """Record uniform function strictness without stripping flags.

    OpenAI ``strict`` flags stay on the tools until post-override finalization.
    Mixed or invalid strictness is left in place so a later tool override can
    still be judged before egress.
    """
    tools = request_data.get("tools")
    if not isinstance(tools, list) or not tools:
        return request_data
    if not any(_tool_declares_strict_key(tool) for tool in tools):
        return request_data
    try:
        tools_request_strict = resolve_cohere_v2_strict_tools(tools)
    except CohereV2StrictToolsError:
        return request_data
    if tools_request_strict:
        return {**request_data, "strict_tools": True}
    if "strict_tools" not in request_data:
        return request_data
    updated = dict(request_data)
    updated.pop("strict_tools", None)
    return updated


def finalize_cohere_v2_strict_tools(request_data: dict) -> dict:
    """Apply strictness from the effective tools and request flag.

    Call this after overrides such as ``extra_body`` are merged. Conflicting
    combinations raise before the caller can send the body.
    """
    if not isinstance(request_data, dict):
        return request_data
    tools = request_data.get("tools")
    flag_present = "strict_tools" in request_data
    flag = request_data.get("strict_tools") if flag_present else None
    if flag_present and not isinstance(flag, bool):
        raise CohereV2StrictToolsError(
            "Cohere strict_tools cannot represent a non-boolean request flag."
        )
    if not isinstance(tools, list) or not tools:
        if flag is True:
            raise CohereV2StrictToolsError(
                "Cohere strict_tools cannot enable strictness without strict function tools."
            )
        return request_data

    tools_request_strict = resolve_cohere_v2_strict_tools(tools)
    if flag is True and not tools_request_strict:
        raise CohereV2StrictToolsError(
            "Cohere strict_tools cannot enable strictness for tools whose caller did not request it."
        )
    if flag is False and tools_request_strict:
        raise CohereV2StrictToolsError(
            "Cohere strict_tools cannot disable strictness for tools that requested it."
        )

    finalized = {
        **request_data,
        "tools": _strip_openai_function_strict(tools),
    }
    if tools_request_strict:
        finalized["strict_tools"] = True
    elif flag is False:
        finalized["strict_tools"] = False
    else:
        finalized.pop("strict_tools", None)
    return finalized


def shield_cohere_responses_function_parameters(
    responses_api_request: dict,
) -> tuple:
    """Copy non-dict function parameters out of a Responses request.

    The shared Responses converter replaces falsy parameters and raises on
    values that ``dict(...)`` cannot accept. Cohere keeps those original
    values. The returned request is a shallow copy with only the shielded
    function tools replaced.
    """
    if not isinstance(responses_api_request, dict):
        return responses_api_request, []
    tools = responses_api_request.get("tools")
    if not isinstance(tools, list):
        return responses_api_request, []

    preserved: List[tuple] = []
    shielded_tools: List[Any] = []
    function_index = 0
    changed = False
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            if "parameters" in tool and not isinstance(tool.get("parameters"), dict):
                preserved.append((function_index, tool.get("parameters")))
                shielded_tools.append({**tool, "parameters": {"type": "object"}})
                changed = True
            else:
                shielded_tools.append(tool)
            function_index += 1
        else:
            shielded_tools.append(tool)
    if not changed:
        return responses_api_request, []
    return {**responses_api_request, "tools": shielded_tools}, preserved


def restore_cohere_responses_function_parameters(
    completion_kwargs: dict,
    preserved: List[tuple],
) -> dict:
    """Put shielded Responses parameter values back onto converted tools."""
    if not preserved or not isinstance(completion_kwargs, dict):
        return completion_kwargs
    tools = completion_kwargs.get("tools")
    if not isinstance(tools, list):
        return completion_kwargs
    preserved_by_index = {index: parameters for index, parameters in preserved}
    restored_tools: List[Any] = []
    function_index = 0
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            if function_index in preserved_by_index:
                function = tool.get("function")
                original_parameters = preserved_by_index[function_index]
                if isinstance(function, dict):
                    tool = {
                        **tool,
                        "function": {
                            **function,
                            "parameters": original_parameters,
                        },
                    }
                else:
                    tool = {**tool, "parameters": original_parameters}
            function_index += 1
        restored_tools.append(tool)
    return {**completion_kwargs, "tools": restored_tools}


def prepare_cohere_v2_strict_completion_kwargs(
    completion_kwargs: dict,
) -> dict:
    """Fail mixed strictness before completion and keep strict=true visible.

    All-strict flags stay on the tools so the V2 transformer can set
    ``strict_tools``. All-nonstrict flags are removed here so Cohere does not
    receive the OpenAI field. Caller-owned tool objects are not mutated.
    """
    extra_body = completion_kwargs.get("extra_body")
    if isinstance(extra_body, dict) and (
        "tools" in extra_body or "strict_tools" in extra_body
    ):
        return completion_kwargs
    tools = completion_kwargs.get("tools")
    if not isinstance(tools, list) or not tools:
        return completion_kwargs
    if resolve_cohere_v2_strict_tools(tools):
        return completion_kwargs
    return {
        **completion_kwargs,
        "tools": _strip_openai_function_strict(tools),
    }


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

        return apply_cohere_v2_strict_tools(data)

    def finalize_strict_tools_request(self, request_data: dict) -> dict:
        """Decide strictness from the body after ``extra_body`` overrides."""
        return finalize_cohere_v2_strict_tools(request_data)

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        data = await super().async_transform_request(
            model=model,
            messages=self._project_messages_for_cohere_v2(messages),
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if not isinstance(data, dict):
            return data
        return apply_cohere_v2_strict_tools(data)

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

        ## ADD CITATIONS AS ANNOTATIONS
        annotations: Optional[List[ChatCompletionAnnotation]] = None
        citations = None

        if (
            "message" in cohere_v2_chat_response
            and "citations" in cohere_v2_chat_response["message"]
        ):
            citations = cohere_v2_chat_response["message"]["citations"]

        if citations:
            annotations = self._translate_citations_to_openai_annotations(citations)

        ## Tool calling response
        cohere_tools_response = cohere_v2_chat_response["message"].get("tool_calls", [])
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls: List[ChatCompletionToolCallChunk] = []
            for index, tool in enumerate(cohere_tools_response):
                tool_call: ChatCompletionToolCallChunk = {
                    **tool,  # type: ignore
                    "index": index,
                }
                tool_calls.append(tool_call)
            _message = litellm.Message(
                tool_calls=tool_calls,
                content=None,
                annotations=annotations,
            )
            model_response.choices[0].message = _message  # type: ignore
        else:
            if annotations:
                current_message = model_response.choices[0].message  # type: ignore
                current_message.annotations = annotations

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

    def _translate_citations_to_openai_annotations(
        self, citations: List[dict]
    ) -> List[ChatCompletionAnnotation]:
        """
        Transform Cohere citations to OpenAI annotations format.

        Creates separate annotations for each source in a citation, allowing multiple
        annotations with the same start/end index if they reference different sources.

        Args:
            citations: List of Cohere citation objects with format:
                {
                    "start": int,
                    "end": int,
                    "text": str,
                    "sources": [
                        {
                            "type": "document",
                            "document": {
                                "title": str,
                                "snippet": str,
                                ...
                            },
                            "id": str
                        }
                    ]
                }

        Returns:
            List of OpenAI ChatCompletionAnnotation objects (one per source)
        """
        annotations: List[ChatCompletionAnnotation] = []

        for citation in citations:
            start_index = citation.get("start", 0)
            end_index = citation.get("end", 0)

            # Extract source information - loop through all sources
            sources = citation.get("sources", [])
            if not sources:
                continue

            # Create an annotation for each source
            for source in sources:
                if source.get("type") == "document" and "document" in source:
                    document = source["document"]
                    title = document.get("title", "")
                    url = source.get("url") or f"source:{source.get('id', 'unknown')}"

                    url_citation: ChatCompletionAnnotationURLCitation = {
                        "start_index": start_index,
                        "end_index": end_index,
                        "title": title,
                        "url": url,
                    }

                    annotation: ChatCompletionAnnotation = {
                        "type": "url_citation",
                        "url_citation": url_citation,
                    }

                    annotations.append(annotation)

        return annotations
