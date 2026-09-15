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

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        **kwargs,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param in _CAPABILITY_GATED_PARAMS:
            if param in non_default_params and param not in supported_openai_params and not drop_params:
                raise UnsupportedParamsError(
                    message=(f"{param} is not supported for this Nous Portal model."),
                    llm_provider="nous",
                    model=model,
                )
        return super().map_openai_params(non_default_params, optional_params, model, drop_params)
