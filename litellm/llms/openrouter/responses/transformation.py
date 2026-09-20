"""
OpenRouter Responses API Configuration.

OpenRouter supports the Responses API at https://openrouter.ai/api/v1/responses
with OpenAI-compatible request/response format, including reasoning with
encrypted_content for multi-turn stateless workflows.

Docs: https://openrouter.ai/docs/api/reference/responses/overview
"""

from typing import Optional

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openrouter.common_utils import (
    apply_openrouter_auth_headers,
    resolve_openrouter_complete_url,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenRouterResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OpenRouter's Responses API.

    Inherits from OpenAIResponsesAPIConfig since OpenRouter's Responses API
    is compatible with OpenAI's Responses API specification.

    Key difference from direct OpenAI:
    - Uses the shared OpenRouter credential/base resolver
    - Canonical API base is https://openrouter.ai/api with one /v1 segment
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENROUTER

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        return apply_openrouter_auth_headers(
            headers,
            api_key=litellm_params.api_key,
            api_base=litellm_params.api_base,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_key = None
        if isinstance(litellm_params, dict):
            api_key = litellm_params.get("api_key")
        elif litellm_params is not None:
            api_key = getattr(litellm_params, "api_key", None)
        return resolve_openrouter_complete_url(
            "responses",
            api_base=api_base,
            api_key=api_key,
        )

    def supports_native_websocket(self) -> bool:
        """OpenRouter does not support native WebSocket for Responses API"""
        return False
