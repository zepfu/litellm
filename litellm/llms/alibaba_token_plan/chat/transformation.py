"""Alibaba Token Plan OpenAI-compatible chat transformation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig
from litellm.secret_managers.main import get_secret_str

from ...openai.common_utils import OpenAIError

ALIBABA_TOKEN_PLAN_API_BASE = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL = f"{ALIBABA_TOKEN_PLAN_API_BASE}/chat/completions"
ALIBABA_TOKEN_PLAN_API_KEY_ENV = "ALIBABA_KEY"
ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV = (
    "LITELLM_ALIBABA_TOKEN_PLAN_SETTINGS_FILE"
)
ALIBABA_TOKEN_PLAN_PROVIDER_NAME = "alibaba_token_plan"
# Catalog metadata only: credential discovery and model admission are
# validated structurally, never against this static enumeration.
ALIBABA_TOKEN_PLAN_MODEL_IDS = frozenset(
    {
        "qwen3.8-max-preview",
        "qwen3.8-max",
        "qwen3.7-plus",
        "qwen3.7-max",
        "qwen3.6-flash",
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "glm-5.2",
    }
)


class AlibabaTokenPlanAuthenticationError(OpenAIError):
    """A bounded local error for a missing canonical Token Plan credential."""

    def __init__(self) -> None:
        super().__init__(
            status_code=401,
            message=(
                "Alibaba Token Plan authentication requires the existing "
                "ALIBABA_KEY credential or the canonical Qwen settings file."
            ),
            headers={},
        )


class AlibabaTokenPlanChatConfig(DashScopeChatConfig):
    """DashScope-compatible transport with Token Plan identity and credentials."""

    @staticmethod
    def _model_id(model: str) -> str:
        """Accept a bare internal model suffix or exactly
        ``alibaba_token_plan/<single nonempty suffix>``.

        Foreign provider prefixes, empty suffixes, and nested namespaces are
        rejected. Admission is structural and never gated on a static model
        enumeration.
        """
        provider, separator, model_id = model.partition("/")
        if separator:
            if provider != ALIBABA_TOKEN_PLAN_PROVIDER_NAME:
                raise ValueError(
                    f"Unsupported Alibaba Token Plan model {model!r}. "
                    "Token Plan routes require a bare model ID or an "
                    "alibaba_token_plan/<model-id> route."
                )
            normalized = model_id
        else:
            normalized = provider
        normalized = normalized.strip()
        if not normalized or "/" in normalized:
            raise ValueError(
                f"Unsupported Alibaba Token Plan model {model!r}. "
                "Token Plan routes require a nonempty model ID."
            )
        return normalized

    @staticmethod
    def _get_qwen_settings_api_key() -> Optional[str]:
        settings_file = os.getenv(ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV)
        if not isinstance(settings_file, str) or not settings_file.strip():
            return None
        try:
            settings = json.loads(
                Path(settings_file.strip()).read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            return None
        if not isinstance(settings, dict):
            return None

        provider_groups = settings.get("modelProviders")
        if not isinstance(provider_groups, dict):
            return None
        providers = provider_groups.get("openai")
        if not isinstance(providers, list):
            return None

        # Provider/credential-structure validation only: the provider entry
        # must point at the canonical Token Plan base URL with a structurally
        # valid id and envKey. The static model catalog is never an admission
        # gate here.
        env_keys = {
            provider.get("envKey")
            for provider in providers
            if isinstance(provider, dict)
            and str(provider.get("baseUrl") or "").rstrip("/")
            == ALIBABA_TOKEN_PLAN_API_BASE
            and isinstance(provider.get("id"), str)
            and provider["id"].strip()
            and isinstance(provider.get("envKey"), str)
            and provider["envKey"].strip()
        }
        if len(env_keys) != 1:
            return None

        configured_env = settings.get("env")
        if not isinstance(configured_env, dict):
            return None
        api_key = configured_env.get(next(iter(env_keys)))
        if not isinstance(api_key, str) or not api_key.strip():
            return None
        return api_key.strip()

    @classmethod
    def _get_canonical_api_key(cls) -> str:
        api_key = get_secret_str(ALIBABA_TOKEN_PLAN_API_KEY_ENV)
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
        settings_api_key = cls._get_qwen_settings_api_key()
        if settings_api_key is not None:
            return settings_api_key
        raise AlibabaTokenPlanAuthenticationError()

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        _ = api_base, api_key
        return ALIBABA_TOKEN_PLAN_API_BASE, self._get_canonical_api_key()

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        _ = api_base, api_key, optional_params, litellm_params, stream
        self._model_id(model)
        return ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL
