"""Translate OpenAI chat completions to Nous Portal `/v1/chat/completions`."""

from typing import Optional, Tuple

from litellm.exceptions import UnsupportedParamsError
from ...openai_like.chat.transformation import OpenAILikeChatConfig

_NOUS_API_BASE = "https://inference-api.nousresearch.com/v1"
_UNSUPPORTED_UNTIL_D3 = frozenset({"stream", "tools", "tool_choice"})


class NousChatConfig(OpenAILikeChatConfig):
    """OpenAI-compatible Nous Portal chat config. Streaming/tools stay off until D3."""

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
        return [
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

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        **kwargs,
    ) -> dict:
        for param in _UNSUPPORTED_UNTIL_D3:
            if param in non_default_params and not drop_params:
                raise UnsupportedParamsError(
                    message=(
                        f"{param} is not supported for Nous Portal models until "
                        "direct evidence is captured."
                    ),
                    llm_provider="nous",
                    model=model,
                )
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
