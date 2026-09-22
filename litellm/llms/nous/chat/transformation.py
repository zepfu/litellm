"""Translate OpenAI chat completions to Nous Portal `/v1/chat/completions`."""

from typing import Optional, Tuple

from litellm.exceptions import UnsupportedParamsError
from litellm.utils import (
    _get_model_info_helper,
    supports_function_calling,
    supports_native_streaming,
    supports_tool_choice,
)

from ...openai_like.chat.transformation import OpenAILikeChatConfig

_NOUS_API_BASE = "https://inference-api.nousresearch.com/v1"
_CAPABILITY_GATED_PARAMS = ("stream", "tools", "tool_choice")


class NousChatConfig(OpenAILikeChatConfig):
    """OpenAI-compatible Nous Portal chat config."""

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "nous"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or _NOUS_API_BASE
        if not api_key:
            try:
                from litellm.secret_managers.hermes_nous_auth import (
                    load_nous_invoke_jwt,
                )

                api_key = load_nous_invoke_jwt()
            except Exception:
                api_key = api_key or None
        return api_base, api_key

    def get_supported_openai_params(self, model: str) -> list:
        supported_params = [
            "messages",
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "response_format",
            "seed",
            "user",
        ]

        if self._supports_explicit_native_streaming(model):
            supported_params.append("stream")
        if supports_function_calling(model=model, custom_llm_provider="nous"):
            supported_params.append("tools")
        if supports_tool_choice(model=model, custom_llm_provider="nous"):
            supported_params.append("tool_choice")

        return supported_params

    @staticmethod
    def _supports_explicit_native_streaming(model: str) -> bool:
        if not supports_native_streaming(model=model, custom_llm_provider="nous"):
            return False
        try:
            model_info = _get_model_info_helper(model=model, custom_llm_provider="nous")
        except Exception:
            return False
        return model_info.get("supports_native_streaming") is True

    @staticmethod
    def _capability_value_requests_feature(param: str, value: object) -> bool:
        """True when the value asks for streaming, tools, or a real tool choice."""
        if param == "stream":
            return value is True
        if param == "tools":
            if isinstance(value, (str, list, tuple, set, dict)):
                return len(value) > 0
            return bool(value)
        if param == "tool_choice":
            return value is not None and not (isinstance(value, str) and value == "auto")
        return False

    def reject_unsupported_capability_params(
        self,
        *,
        non_default_params: dict,
        model: str,
        reject_nonsemantic: bool,
    ) -> None:
        """Reject unsupported stream, tools, and tool_choice before egress.

        Semantic requests are rejected even when ``drop_params`` is set.
        ``stream=False`` stays eligible. Semantically empty tool payloads are
        rejected only when ``reject_nonsemantic`` is set, which preserves the
        direct mapper contract for an explicit empty declaration.
        """
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param in _CAPABILITY_GATED_PARAMS:
            if param not in non_default_params or param in supported_openai_params:
                continue
            semantic = self._capability_value_requests_feature(param, non_default_params[param])
            if param == "stream":
                if not semantic:
                    continue
            elif not semantic and not reject_nonsemantic:
                continue
            raise UnsupportedParamsError(
                message=f"{param} is not supported for this Nous Portal model.",
                llm_provider="nous",
                model=model,
            )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        **kwargs,
    ) -> dict:
        self.reject_unsupported_capability_params(
            non_default_params=non_default_params,
            model=model,
            reject_nonsemantic=not drop_params,
        )
        return super().map_openai_params(non_default_params, optional_params, model, drop_params)
