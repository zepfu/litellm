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
from litellm.exceptions import UnsupportedParamsError
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


_COHERE_TOOL_CHOICE_AUTO = "auto"
_COHERE_TOOL_CHOICE_REQUIRED = "required"
_COHERE_TOOL_CHOICE_NONE = "none"
_COHERE_SUPPORTED_TOOL_CHOICE_FORMS = frozenset(
    {
        _COHERE_TOOL_CHOICE_AUTO,
        _COHERE_TOOL_CHOICE_REQUIRED,
        _COHERE_TOOL_CHOICE_NONE,
    }
)
_COHERE_WIRE_TOOL_CHOICE = {
    _COHERE_TOOL_CHOICE_REQUIRED: "REQUIRED",
    _COHERE_TOOL_CHOICE_NONE: "NONE",
}
_COHERE_CHAT_PROVIDERS = frozenset({"cohere", "cohere_chat"})


class CohereUnsupportedToolChoiceError(UnsupportedParamsError):
    """Tool choice rejected locally, before any Cohere provider request."""

    def __init__(self, message: str, model: str) -> None:
        super().__init__(
            message=message,
            llm_provider="cohere",
            model=model,
            status_code=400,
        )
        self.attempted_provider_call = False
        self.failure_phase = "cohere_tool_choice_preflight"
        self.detail = {
            "attempted_provider_call": False,
            "param": "tool_choice",
        }
        self.args = (self.message,)

    def __str__(self) -> str:
        return str(self.message)


def _has_cohere_tools(params: dict) -> bool:
    tools = params.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _normalize_cohere_request_model(model: str) -> str:
    """Return the model name Cohere is sent, for catalog lookup.

    Provider resolution drops one leading ``cohere`` or ``cohere_chat``
    prefix. The v2 completion path then removes every ``v2/`` route
    prefix before transport. Forced-tool support is stored on that
    catalog name, not on the route-prefixed alias.
    """
    normalized = model
    provider, separator, remainder = normalized.partition("/")
    if separator and provider in _COHERE_CHAT_PROVIDERS:
        normalized = remainder
    if "v2/" in normalized:
        normalized = normalized.replace("v2/", "")
    return normalized


def _cohere_model_info(model: str) -> Optional[dict]:
    """Return Cohere catalog metadata for ``model``, or None when unmapped."""
    from litellm.utils import get_model_info

    normalized = _normalize_cohere_request_model(model)
    lookups = (
        (normalized, "cohere_chat"),
        (normalized, "cohere"),
        (f"cohere/{normalized}", "cohere"),
        (f"cohere/{normalized}", "cohere_chat"),
        (normalized, None),
    )
    seen = set()
    for candidate_model, provider in lookups:
        key = (candidate_model, provider)
        if key in seen:
            continue
        seen.add(key)
        try:
            if provider is None:
                info = get_model_info(model=candidate_model)
            else:
                info = get_model_info(
                    model=candidate_model,
                    custom_llm_provider=provider,
                )
        except Exception:
            continue
        if info.get("litellm_provider") in _COHERE_CHAT_PROVIDERS:
            return info
    return None


def _model_supports_forced_tool_choice(model: str) -> bool:
    """Cohere REQUIRED/NONE exists only on models flagged for forced tools."""
    info = _cohere_model_info(model)
    if info is None:
        return False
    provider_specific_entry = info.get("provider_specific_entry")
    if not isinstance(provider_specific_entry, dict):
        return False
    cohere_entry = provider_specific_entry.get("cohere")
    if not isinstance(cohere_entry, dict):
        return False
    return cohere_entry.get("supports_forced_tool_choice") is True


def _tool_choice_names_a_function(tool_choice: dict) -> bool:
    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return True
    if tool_choice.get("type") == "function":
        return True
    name = tool_choice.get("name")
    return isinstance(name, str) and bool(name.strip())


def _classify_cohere_tool_choice(tool_choice: Any, model: str) -> str:
    """Return auto, required, or none. Raise for forms Cohere cannot preserve."""
    if isinstance(tool_choice, str):
        form = tool_choice.strip().casefold()
        if form in _COHERE_SUPPORTED_TOOL_CHOICE_FORMS:
            return form
        raise CohereUnsupportedToolChoiceError(
            message=(
                "Cohere V2 does not support tool_choice "
                f"{tool_choice.strip()!r}. Supported forms are auto, required, "
                "and none. The request was rejected before provider transport."
            ),
            model=model,
        )
    if isinstance(tool_choice, dict):
        if _tool_choice_names_a_function(tool_choice):
            raise CohereUnsupportedToolChoiceError(
                message=(
                    "Cohere V2 cannot constrain a named function through "
                    "tool_choice. Supported forms are auto, required, and none. "
                    "The request was rejected before provider transport."
                ),
                model=model,
            )
        choice_type = tool_choice.get("type")
        if isinstance(choice_type, str):
            form = choice_type.strip().casefold()
            if form in _COHERE_SUPPORTED_TOOL_CHOICE_FORMS:
                return form
        raise CohereUnsupportedToolChoiceError(
            message=(
                "Cohere V2 does not support this tool_choice object. Supported "
                "forms are auto, required, and none. The request was rejected "
                "before provider transport."
            ),
            model=model,
        )
    raise CohereUnsupportedToolChoiceError(
        message=(
            "Cohere V2 does not support this tool_choice value. Supported forms "
            "are auto, required, and none. The request was rejected before "
            "provider transport."
        ),
        model=model,
    )


def _map_cohere_v2_tool_choice(
    tool_choice: Any,
    *,
    model: str,
    tools_present: bool,
) -> Optional[str]:
    """Map one OpenAI tool_choice onto a Cohere V2 wire value.

    ``auto`` is Cohere's documented default and is preserved by omitting
    ``tool_choice``. ``required`` and ``none`` stay REQUIRED and NONE.
    Unsupported forms, including null and a named function, are rejected
    instead of being weakened to automatic selection.
    """
    if tool_choice is None:
        raise CohereUnsupportedToolChoiceError(
            message=(
                "Cohere V2 does not support tool_choice null. Supported forms "
                "are auto, required, and none. The request was rejected before "
                "provider transport."
            ),
            model=model,
        )
    form = _classify_cohere_tool_choice(tool_choice, model)
    if form == _COHERE_TOOL_CHOICE_AUTO:
        return None
    if not _model_supports_forced_tool_choice(model):
        raise CohereUnsupportedToolChoiceError(
            message=(
                f"Cohere model {model} does not support forced tool_choice "
                f"{form!r}. The request was rejected before provider transport."
            ),
            model=model,
        )
    if form == _COHERE_TOOL_CHOICE_REQUIRED and not tools_present:
        raise CohereUnsupportedToolChoiceError(
            message=(
                "Cohere V2 tool_choice required needs at least one tool. The "
                "request was rejected before provider transport."
            ),
            model=model,
        )
    return _COHERE_WIRE_TOOL_CHOICE[form]


def _project_cohere_v2_tool_choice(model: str, optional_params: dict) -> dict:
    if "tool_choice" not in optional_params:
        return optional_params
    mapped_tool_choice = _map_cohere_v2_tool_choice(
        optional_params.get("tool_choice"),
        model=model,
        tools_present=_has_cohere_tools(optional_params),
    )
    if mapped_tool_choice is None:
        return {
            key: value for key, value in optional_params.items() if key != "tool_choice"
        }
    return {**optional_params, "tool_choice": mapped_tool_choice}


def _apply_cohere_v2_tool_choice_to_request_body(
    model: str, request_data: dict
) -> None:
    """Map ``tool_choice`` on the body that will be sent to Cohere."""
    if "tool_choice" not in request_data:
        return
    mapped_tool_choice = _map_cohere_v2_tool_choice(
        request_data.get("tool_choice"),
        model=model,
        tools_present=_has_cohere_tools(request_data),
    )
    if mapped_tool_choice is None:
        request_data.pop("tool_choice", None)
        return
    request_data["tool_choice"] = mapped_tool_choice


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
        # drop_params must not rewrite an unsupported tool_choice into auto.
        _ = drop_params
        if "tool_choice" in non_default_params:
            tool_choice_params = {
                "tool_choice": non_default_params["tool_choice"],
            }
            if _has_cohere_tools(non_default_params):
                tool_choice_params["tools"] = non_default_params["tools"]
            elif _has_cohere_tools(optional_params):
                tool_choice_params["tools"] = optional_params["tools"]
            projected_tool_choice = _project_cohere_v2_tool_choice(
                model,
                tool_choice_params,
            )
            if "tool_choice" in projected_tool_choice:
                optional_params["tool_choice"] = projected_tool_choice["tool_choice"]
            else:
                optional_params.pop("tool_choice", None)
        for param, value in non_default_params.items():
            if param == "tool_choice":
                continue
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
    ):
        """Validate tool_choice after extra_body has been merged.

        The shared handler merges ``extra_body`` onto the transformed body
        and then calls this method, before any provider request. Mapping
        here covers the JSON that is actually sent.
        """
        body_model = request_data.get("model")
        if not isinstance(body_model, str) or not body_model.strip():
            body_model = model or ""
        _apply_cohere_v2_tool_choice_to_request_body(body_model, request_data)
        return super().sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

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
            _project_cohere_v2_tool_choice(model, optional_params),
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
            optional_params=_project_cohere_v2_tool_choice(model, optional_params),
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
