import time
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    List,
    NoReturn,
    Optional,
    Union,
)

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

COHERE_V2_NAMESPACE_FUNCTION_MAP_KEY = "cohere_v2_namespace_function_map"
_COHERE_V2_PROVIDER = "cohere"

from ..common_utils import CohereError
from ..common_utils import CohereV2ModelResponseIterator
from ..common_utils import validate_environment as cohere_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _cohere_v2_tool_type(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    tool_type = value.get("type")
    if isinstance(tool_type, str):
        stripped = tool_type.strip()
        if stripped:
            return stripped.lower()
    return None


def _reject_cohere_v2_tool_schema(reason: str) -> NoReturn:
    raise UnsupportedParamsError(
        message=f"Cohere V2 cannot represent tool schema before egress: {reason}",
        llm_provider=_COHERE_V2_PROVIDER,
    )


def _cohere_v2_nonempty_name(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _cohere_v2_function_parts(
    tool: dict[str, Any],
) -> tuple[Optional[str], Any, Any]:
    """Return the function name, parameters, and description without rewriting them."""
    function = tool.get("function")
    if isinstance(function, dict):
        return (
            _cohere_v2_nonempty_name(function.get("name")),
            function.get("parameters"),
            function.get("description"),
        )
    return (
        _cohere_v2_nonempty_name(tool.get("name")),
        tool.get("parameters"),
        tool.get("description"),
    )


def _reject_deferred_or_nested_function(
    tool: dict[str, Any],
    *,
    location: str,
) -> None:
    if tool.get("defer_loading") is True:
        _reject_cohere_v2_tool_schema(f"{location} uses defer_loading, which Cohere V2 cannot represent")
    if _cohere_v2_tool_type(tool) == "namespace" or isinstance(tool.get("tools"), list):
        _reject_cohere_v2_tool_schema(f"{location} nests another namespace, which Cohere V2 cannot represent")


def _responses_function_tool(
    tool: dict[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    _reject_deferred_or_nested_function(tool, location=location)
    if _cohere_v2_tool_type(tool) != "function":
        _reject_cohere_v2_tool_schema(f"{location} has unsupported type {tool.get('type')!r}")
    name, parameters, description = _cohere_v2_function_parts(tool)
    if name is None:
        _reject_cohere_v2_tool_schema(f"{location} is missing a function name")
    if not isinstance(parameters, dict):
        _reject_cohere_v2_tool_schema(f"{location} is missing a parameters object for {name}")
    translated: dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": deepcopy(parameters),
    }
    if isinstance(description, str) and description.strip():
        translated["description"] = description
    return translated


def _chat_function_tool(
    tool: dict[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    flat_tool = _responses_function_tool(tool, location=location)
    function: dict[str, Any] = {
        "name": flat_tool["name"],
        "parameters": flat_tool["parameters"],
    }
    description = flat_tool.get("description")
    if isinstance(description, str):
        function["description"] = description
    return {"type": "function", "function": function}


def _claim_cohere_v2_function_name(
    name: str,
    occupied: dict[str, str],
    *,
    location: str,
    namespace: Optional[str] = None,
) -> None:
    previous = occupied.get(name)
    if previous is not None:
        _reject_cohere_v2_tool_schema(f"{location} reuses function name {name!r} already claimed by {previous}")
    occupied[name] = namespace or location


def _tool_namespace(tool: dict[str, Any]) -> Optional[str]:
    namespace = tool.get("namespace")
    if isinstance(namespace, str) and namespace.strip():
        return namespace
    function = tool.get("function")
    if isinstance(function, dict):
        nested_namespace = function.get("namespace")
        if isinstance(nested_namespace, str) and nested_namespace.strip():
            return nested_namespace
    return None


def _responses_tools_need_namespace_translation(tools: list[Any]) -> bool:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if _cohere_v2_tool_type(tool) == "namespace" or _tool_namespace(tool) is not None:
            return True
    return False


def translate_cohere_v2_responses_tools(
    tools: Any,
) -> tuple[Any, dict[str, str]]:
    """Flatten namespace/function tools into Responses function tools.

    Names and parameter objects are copied. Unrepresentable namespace
    semantics raise before a provider call instead of being dropped.
    ``strict`` is left off the flattened tool; Cohere ``strict_tools``
    remains COHERE-012.
    """
    if not isinstance(tools, list) or not _responses_tools_need_namespace_translation(tools):
        return tools, {}

    occupied: dict[str, str] = {}
    namespace_by_name: dict[str, str] = {}
    translated: list[Any] = []
    changed = False
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            translated.append(tool)
            continue
        tool_type = _cohere_v2_tool_type(tool)
        if tool_type == "function":
            name, _, _ = _cohere_v2_function_parts(tool)
            if name is not None:
                _claim_cohere_v2_function_name(
                    name,
                    occupied,
                    location=f"tools[{index}]",
                )
            namespace = _tool_namespace(tool)
            if namespace is not None:
                if name is None:
                    _reject_cohere_v2_tool_schema(f"tools[{index}] has a namespace without a function name")
                namespace_by_name[name] = namespace
                translated.append(_responses_function_tool(tool, location=f"tools[{index}]"))
                changed = True
            else:
                translated.append(tool)
            continue
        if tool_type != "namespace":
            translated.append(tool)
            continue

        changed = True
        namespace_name = _cohere_v2_nonempty_name(tool.get("name"))
        if namespace_name is None:
            _reject_cohere_v2_tool_schema(f"tools[{index}] namespace is missing a name")
        children = tool.get("tools")
        if not isinstance(children, list):
            _reject_cohere_v2_tool_schema(f"tools[{index}] namespace {namespace_name!r} is missing a function list")
        if tool.get("defer_loading") is True:
            _reject_cohere_v2_tool_schema(f"tools[{index}] namespace {namespace_name!r} uses defer_loading")
        for child_index, child in enumerate(children):
            location = f"tools[{index}].tools[{child_index}]"
            if not isinstance(child, dict):
                _reject_cohere_v2_tool_schema(f"{location} is not an object")
            flat_tool = _responses_function_tool(child, location=location)
            child_name = flat_tool["name"]
            _claim_cohere_v2_function_name(
                child_name,
                occupied,
                location=location,
                namespace=namespace_name,
            )
            namespace_by_name[child_name] = namespace_name
            translated.append(flat_tool)
    if not changed:
        return tools, {}
    return translated, namespace_by_name


def _chat_tools_need_translation(tools: list[Any]) -> bool:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = _cohere_v2_tool_type(tool)
        if tool_type == "namespace" or _tool_namespace(tool) is not None:
            return True
        if tool_type == "function" and not isinstance(tool.get("function"), dict):
            return True
    return False


def _extend_chat_namespace_tool(
    tool: dict[str, Any],
    *,
    location: str,
    occupied: dict[str, str],
    namespace_by_name: dict[str, str],
    translated: list[Any],
) -> None:
    namespace_name = _cohere_v2_nonempty_name(tool.get("name"))
    if namespace_name is None:
        _reject_cohere_v2_tool_schema(f"{location} namespace is missing a name")
    children = tool.get("tools")
    if not isinstance(children, list):
        _reject_cohere_v2_tool_schema(f"{location} namespace {namespace_name!r} is missing a function list")
    if tool.get("defer_loading") is True:
        _reject_cohere_v2_tool_schema(f"{location} namespace {namespace_name!r} uses defer_loading")
    for child_index, child in enumerate(children):
        child_location = f"{location}.tools[{child_index}]"
        if not isinstance(child, dict):
            _reject_cohere_v2_tool_schema(f"{child_location} is not an object")
        chat_tool = _chat_function_tool(child, location=child_location)
        child_name = chat_tool["function"]["name"]
        _claim_cohere_v2_function_name(
            child_name,
            occupied,
            location=child_location,
            namespace=namespace_name,
        )
        namespace_by_name[child_name] = namespace_name
        translated.append(chat_tool)


def _translate_chat_function_tool(
    tool: dict[str, Any],
    *,
    location: str,
    occupied: dict[str, str],
    namespace_by_name: dict[str, str],
) -> dict[str, Any]:
    name, _, _ = _cohere_v2_function_parts(tool)
    if name is not None:
        _claim_cohere_v2_function_name(name, occupied, location=location)
    namespace = _tool_namespace(tool)
    if namespace is not None:
        if name is None:
            _reject_cohere_v2_tool_schema(f"{location} has a namespace without a function name")
        namespace_by_name[name] = namespace
    if isinstance(tool.get("function"), dict) and namespace is None:
        return tool
    return _chat_function_tool(tool, location=location)


def translate_cohere_v2_chat_tools(
    tools: Any,
) -> tuple[Any, dict[str, str]]:
    """Flatten namespace tools into Cohere V2 nested function schemas."""
    if not isinstance(tools, list) or not _chat_tools_need_translation(tools):
        return tools, {}

    occupied: dict[str, str] = {}
    namespace_by_name: dict[str, str] = {}
    translated: list[Any] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            translated.append(tool)
            continue
        location = f"tools[{index}]"
        tool_type = _cohere_v2_tool_type(tool)
        if tool_type == "namespace":
            _extend_chat_namespace_tool(
                tool,
                location=location,
                occupied=occupied,
                namespace_by_name=namespace_by_name,
                translated=translated,
            )
            continue
        if tool_type == "function":
            translated.append(
                _translate_chat_function_tool(
                    tool,
                    location=location,
                    occupied=occupied,
                    namespace_by_name=namespace_by_name,
                )
            )
            continue
        translated.append(tool)
    return translated, namespace_by_name


def _tool_choice_function_name(tool_choice: dict[str, Any]) -> Optional[str]:
    function = tool_choice.get("function")
    if isinstance(function, dict):
        nested_name = _cohere_v2_nonempty_name(function.get("name"))
        if nested_name is not None:
            return nested_name
    return _cohere_v2_nonempty_name(tool_choice.get("name"))


def translate_cohere_v2_tool_choice(
    tool_choice: Any,
    namespace_by_name: dict[str, str],
) -> Any:
    """Keep a namespace function choice pointed at the same function name.

    Cohere V2 ``tool_choice`` itself only accepts ``REQUIRED`` and ``NONE``.
    Mapping those enums belongs to COHERE-010. A choice that names one
    translated function keeps that name in the OpenAI function-choice shape
    so later mapping cannot collapse it to an unnamed required choice.
    A namespace container, or a name that was not translated, fails closed.
    """
    if not isinstance(tool_choice, dict):
        return tool_choice
    tool_type = _cohere_v2_tool_type(tool_choice)
    namespace = tool_choice.get("namespace")
    has_namespace = isinstance(namespace, str) and bool(namespace.strip())
    if tool_type != "namespace" and not has_namespace:
        return tool_choice
    if tool_type == "namespace":
        _reject_cohere_v2_tool_schema("tool_choice selects a namespace container rather than a function")
    function_name = _tool_choice_function_name(tool_choice)
    if function_name is None:
        _reject_cohere_v2_tool_schema("tool_choice has a namespace without a function name")
    advertised_namespace = namespace_by_name.get(function_name)
    if advertised_namespace is None or advertised_namespace != namespace:
        _reject_cohere_v2_tool_schema(
            f"tool_choice names {function_name!r} in namespace {namespace!r}, "
            "which is not an advertised translated function"
        )
    return {"type": "function", "function": {"name": function_name}}


def reject_unrepresentable_cohere_v2_tool_choice(tool_choice: Any) -> None:
    """Fail closed when a chat tool_choice still carries namespace semantics."""
    if not isinstance(tool_choice, dict):
        return
    if _cohere_v2_tool_type(tool_choice) == "namespace" or (
        isinstance(tool_choice.get("namespace"), str) and tool_choice.get("namespace").strip()
    ):
        _reject_cohere_v2_tool_schema("tool_choice still carries namespace semantics Cohere V2 cannot represent")


def translate_cohere_v2_responses_input(
    input_items: Any,
    namespace_by_name: dict[str, str],
) -> Any:
    """Drop namespace labels from correlated calls without changing ids or arguments."""
    if not isinstance(input_items, list) or not namespace_by_name:
        return input_items
    translated: list[Any] = []
    changed = False
    for index, item in enumerate(input_items):
        if not isinstance(item, dict) or "namespace" not in item:
            translated.append(item)
            continue
        item_type = item.get("type")
        namespace = item.get("namespace")
        if item_type == "function_call_output":
            translated.append({key: value for key, value in item.items() if key != "namespace"})
            changed = True
            continue
        if item_type != "function_call":
            _reject_cohere_v2_tool_schema(f"input[{index}] type {item_type!r} carries a namespace")
        function_name = _cohere_v2_nonempty_name(item.get("name"))
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
            or function_name is None
            or namespace_by_name.get(function_name) != namespace
        ):
            _reject_cohere_v2_tool_schema(
                f"input[{index}] function_call does not match a translated namespace function"
            )
        translated.append({key: value for key, value in item.items() if key != "namespace"})
        changed = True
    if not changed:
        return input_items
    return translated


def translate_cohere_v2_responses_request(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Translate a Responses request before the shared chat converter."""
    translated_tools, namespace_by_name = translate_cohere_v2_responses_tools(request_body.get("tools"))
    translated_choice = translate_cohere_v2_tool_choice(
        request_body.get("tool_choice"),
        namespace_by_name,
    )
    translated_input = translate_cohere_v2_responses_input(
        request_body.get("input"),
        namespace_by_name,
    )
    if (
        translated_tools is request_body.get("tools")
        and translated_choice is request_body.get("tool_choice")
        and translated_input is request_body.get("input")
    ):
        return request_body, namespace_by_name
    updated_body = dict(request_body)
    if translated_tools is not request_body.get("tools"):
        updated_body["tools"] = translated_tools
    if translated_choice is not request_body.get("tool_choice"):
        updated_body["tool_choice"] = translated_choice
    if translated_input is not request_body.get("input"):
        updated_body["input"] = translated_input
    if namespace_by_name:
        metadata = dict(updated_body.get("litellm_metadata") or {})
        metadata[COHERE_V2_NAMESPACE_FUNCTION_MAP_KEY] = dict(namespace_by_name)
        updated_body["litellm_metadata"] = metadata
    return updated_body, namespace_by_name


def translate_cohere_v2_chat_optional_params(
    optional_params: dict[str, Any],
) -> dict[str, Any]:
    translated_tools, namespace_by_name = translate_cohere_v2_chat_tools(optional_params.get("tools"))
    if translated_tools is optional_params.get("tools") and not namespace_by_name:
        reject_unrepresentable_cohere_v2_tool_choice(optional_params.get("tool_choice"))
        return optional_params
    translated_choice = translate_cohere_v2_tool_choice(
        optional_params.get("tool_choice"),
        namespace_by_name,
    )
    updated = dict(optional_params)
    if translated_tools is not optional_params.get("tools"):
        updated["tools"] = translated_tools
    if translated_choice is not optional_params.get("tool_choice"):
        updated["tool_choice"] = translated_choice
    return updated


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
            if param == "tool_choice":
                # REQUIRED/NONE mapping is COHERE-010. Reject only a choice
                # that still carries namespace semantics; do not drop it.
                reject_unrepresentable_cohere_v2_tool_choice(value)
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
        projected_messages = self._project_messages_for_cohere_v2(messages)
        data = super().transform_request(
            model,
            projected_messages,
            translate_cohere_v2_chat_optional_params(optional_params),
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
                if not isinstance(tool_call, dict):
                    continue
                if _cohere_v2_tool_type(tool_call) == "namespace":
                    _reject_cohere_v2_tool_schema(
                        "message tool_call uses type namespace, which Cohere V2 cannot represent"
                    )
                tool_call.pop("index", None)
                tool_call.pop("namespace", None)
                function = tool_call.get("function")
                if isinstance(function, dict):
                    function.pop("namespace", None)
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
            raise CohereError(message=raw_response.text, status_code=raw_response.status_code)

        try:
            cohere_v2_chat_response = CohereV2ChatResponse(**raw_response_json)  # type: ignore
        except Exception:
            raise CohereError(message=raw_response.text, status_code=422)

        cohere_content = cohere_v2_chat_response["message"].get("content", None)
        if cohere_content is not None:
            model_response.choices[0].message.content = "".join(  # type: ignore
                [content.get("text", "") for content in cohere_content if content is not None]
            )

        ## ADD CITATIONS AS ANNOTATIONS
        annotations: Optional[List[ChatCompletionAnnotation]] = None
        citations = None

        if "message" in cohere_v2_chat_response and "citations" in cohere_v2_chat_response["message"]:
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

    def _translate_citations_to_openai_annotations(self, citations: List[dict]) -> List[ChatCompletionAnnotation]:
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
