"""Z.AI Coding Plan OpenAI-compatible chat transformation."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ...openai.common_utils import OpenAIError

ZAI_CODING_PLAN_API_BASE = "https://api.z.ai/api/coding/paas/v4"
ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL = (
    f"{ZAI_CODING_PLAN_API_BASE}/chat/completions"
)
ZAI_CODING_PLAN_PROVIDER_NAME = "zai_coding_plan"
ZAI_CODING_PLAN_USER_AGENT_PREFIX = "litellm-zai-coding-plan"
ZAI_CODING_PLAN_MODEL_IDS = frozenset({"glm-5.3", "glm-5-turbo", "glm-4.7"})
ZAI_CODING_PLAN_REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}
ZAI_CODING_PLAN_REASONING_EFFORTS = ("low", "high", "max")
_DEFAULT_THINKING = {"type": "enabled", "clear_thinking": False}
_CANONICAL_API_BASE = ZAI_CODING_PLAN_API_BASE.rstrip("/")
_CANONICAL_COMPLETIONS_URL = ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL.rstrip("/")


class ZAICodingPlanAuthenticationError(OpenAIError):
    """A bounded local error for a missing Coding Plan credential."""

    def __init__(self) -> None:
        super().__init__(
            status_code=401,
            message=(
                "Z.AI Coding Plan authentication requires ZAI_KEY or "
                "ZAI_CODING_PLAN_API_KEY. Ordinary ZAI_API_KEY is not reused."
            ),
            headers={},
        )


class ZAICodingPlanApiBaseError(OpenAIError):
    """A caller-supplied api_base is not the canonical Coding Plan endpoint."""

    def __init__(self) -> None:
        super().__init__(
            status_code=400,
            message=(
                "Z.AI Coding Plan is a managed provider; caller-supplied "
                "api_base must be the canonical coding endpoint "
                f"{ZAI_CODING_PLAN_API_BASE}. Ordinary /api/paas/v4 and "
                "open.bigmodel.cn bases are not permitted."
            ),
            headers={},
        )


class ZAICodingPlanChatConfig(OpenAIGPTConfig):
    """OpenAI-compatible transport bound to the Z.AI Coding Plan chat base."""

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return ZAI_CODING_PLAN_PROVIDER_NAME

    @staticmethod
    def _model_id(model: str) -> str:
        """Admit a documented Coding Plan model as a bare ID or prefixed route."""

        if not isinstance(model, str):
            raise ValueError(
                f"Unsupported Z.AI Coding Plan model {model!r}. "
                "Coding Plan routes require zai_coding_plan/<model-id>."
            )
        provider, separator, model_id = model.partition("/")
        if separator:
            if provider != ZAI_CODING_PLAN_PROVIDER_NAME:
                raise ValueError(
                    f"Unsupported Z.AI Coding Plan model {model!r}. "
                    "Coding Plan routes require a bare model ID or a "
                    "zai_coding_plan/<model-id> route."
                )
            normalized = model_id
        else:
            normalized = provider
        normalized = normalized.strip()
        if not normalized or "/" in normalized:
            raise ValueError(
                f"Unsupported Z.AI Coding Plan model {model!r}. "
                "Coding Plan routes require a nonempty model ID."
            )
        if normalized not in ZAI_CODING_PLAN_MODEL_IDS:
            raise ValueError(
                f"Unsupported Z.AI Coding Plan model {model!r}. "
                "Admitted models are glm-5.3, glm-5-turbo, and glm-4.7."
            )
        return normalized

    @staticmethod
    def _validate_api_base(api_base: Optional[str]) -> None:
        if api_base is None:
            return
        if not isinstance(api_base, str) or not api_base.strip():
            raise ZAICodingPlanApiBaseError()
        supplied = api_base.strip().rstrip("/")
        if supplied in {_CANONICAL_API_BASE, _CANONICAL_COMPLETIONS_URL}:
            return
        parsed = urlparse(supplied)
        if parsed.scheme != "https" or parsed.netloc != "api.z.ai":
            raise ZAICodingPlanApiBaseError()
        path = parsed.path.rstrip("/")
        if path in {"/api/coding/paas/v4", "/api/coding/paas/v4/chat/completions"}:
            return
        raise ZAICodingPlanApiBaseError()

    @classmethod
    def _get_canonical_api_key(cls) -> str:
        for env_name in ("ZAI_KEY", "ZAI_CODING_PLAN_API_KEY", "ZHIPU_API_KEY"):
            api_key = get_secret_str(env_name)
            if isinstance(api_key, str) and api_key.strip():
                return api_key.strip()
        raise ZAICodingPlanAuthenticationError()

    @staticmethod
    def _user_agent() -> str:
        env_fork = get_secret_str("AAWM_LITELLM_FORK_VERSION") or get_secret_str(
            "LITELLM_FORK_VERSION"
        )
        if isinstance(env_fork, str) and env_fork.strip():
            version = env_fork.strip()
        else:
            try:
                from litellm._version import version as litellm_version
            except Exception:
                litellm_version = "unknown"
            if (
                isinstance(litellm_version, str)
                and "+" in litellm_version
                and litellm_version.split("+", 1)[1].strip()
            ):
                version = litellm_version.split("+", 1)[1].strip()
            elif isinstance(litellm_version, str) and litellm_version.strip() and litellm_version != "unknown":
                version = litellm_version.strip()
            else:
                version = "dev"
        return f"{ZAI_CODING_PLAN_USER_AGENT_PREFIX}/{version}"

    @staticmethod
    def _map_reasoning_effort(value: object) -> str:
        if not isinstance(value, str) or value not in ZAI_CODING_PLAN_REASONING_EFFORT_MAP:
            supported = ", ".join(ZAI_CODING_PLAN_REASONING_EFFORTS)
            raise ValueError(
                "Z.AI Coding Plan does not support reasoning_effort="
                f"{value!r}. Supported efforts: {supported}."
            )
        return ZAI_CODING_PLAN_REASONING_EFFORT_MAP[value]

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        _ = api_key
        self._validate_api_base(api_base)
        return ZAI_CODING_PLAN_API_BASE, self._get_canonical_api_key()

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        _ = api_key, optional_params, litellm_params, stream
        self._model_id(model)
        self._validate_api_base(api_base)
        return ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        """Preserve cache_control; GLM Coding Plan supports it."""

        _ = model
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        self._model_id(model)
        return [
            "frequency_penalty",
            "extra_headers",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "prompt_cache_key",
            "reasoning_effort",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "thinking",
            "tool_choice",
            "tools",
            "top_p",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        _ = drop_params
        supported_params = self.get_supported_openai_params(model)
        mapped = dict(optional_params)
        reasoning_effort = non_default_params.get("reasoning_effort")
        reasoning = non_default_params.get("reasoning")
        if reasoning_effort is None and isinstance(reasoning, dict):
            reasoning_effort = reasoning.get("effort")
        if reasoning_effort is not None:
            mapped["reasoning_effort"] = self._map_reasoning_effort(reasoning_effort)
        for param, value in non_default_params.items():
            if param in {"reasoning_effort", "reasoning", "thinking"}:
                continue
            if param in supported_params:
                mapped[param] = value
        existing_extra_body = mapped.get("extra_body")
        if existing_extra_body is None:
            extra_body: dict[str, Any] = {}
        elif isinstance(existing_extra_body, dict):
            extra_body = dict(existing_extra_body)
        else:
            raise ValueError("Z.AI Coding Plan extra_body must be an object")
        thinking = non_default_params.get("thinking", mapped.get("thinking"))
        extra_body["thinking"] = (
            dict(thinking) if isinstance(thinking, dict) else dict(_DEFAULT_THINKING)
        )
        mapped.pop("thinking", None)
        mapped["extra_body"] = extra_body
        return mapped

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        upstream_model = self._model_id(model)
        optional_params = dict(optional_params)
        existing_extra_body = optional_params.get("extra_body")
        if existing_extra_body is None:
            extra_body: dict[str, Any] = {}
        elif isinstance(existing_extra_body, dict):
            extra_body = dict(existing_extra_body)
        else:
            raise ValueError("Z.AI Coding Plan extra_body must be an object")
        thinking = optional_params.pop("thinking", extra_body.get("thinking"))
        extra_body["thinking"] = (
            dict(thinking) if isinstance(thinking, dict) else dict(_DEFAULT_THINKING)
        )
        optional_params["extra_body"] = extra_body
        optional_params["extra_headers"] = self.validate_environment(
            headers=headers,
            model=upstream_model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        if optional_params.get("stream"):
            caller_stream_options = optional_params.get("stream_options", {})
            if not isinstance(caller_stream_options, dict):
                raise ValueError("Z.AI Coding Plan stream_options must be an object")
            optional_params["stream_options"] = {
                **caller_stream_options,
                "include_usage": True,
            }
        request = super().transform_request(
            model=upstream_model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        request["model"] = upstream_model
        request_extra_body = request.get("extra_body")
        if not isinstance(request_extra_body, dict):
            request_extra_body = {}
        else:
            request_extra_body = dict(request_extra_body)
        if "thinking" not in request_extra_body:
            request_extra_body["thinking"] = dict(_DEFAULT_THINKING)
        request["extra_body"] = request_extra_body
        request.pop("thinking", None)
        return request

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
        _ = headers, messages, optional_params, litellm_params, api_key
        self._model_id(model)
        self._validate_api_base(api_base)
        return {
            "Authorization": f"Bearer {self._get_canonical_api_key()}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
            "User-Agent": self._user_agent(),
        }
