"""Alibaba Token Plan OpenAI-compatible chat transformation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.common_utils import OpenAIError

ALIBABA_TOKEN_PLAN_API_BASE = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL = f"{ALIBABA_TOKEN_PLAN_API_BASE}/chat/completions"
ALIBABA_TOKEN_PLAN_API_KEY_ENV = "ALIBABA_KEY"
ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV = (
    "LITELLM_ALIBABA_TOKEN_PLAN_SETTINGS_FILE"
)
ALIBABA_TOKEN_PLAN_PROVIDER_NAME = "alibaba_token_plan"
ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID_ENV = "ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID"
ALIBABA_TOKEN_PLAN_SUBSCRIPTION_IDENTITY_SOURCE = "instance_code_sha256"
ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY_PREFIX = (
    "alibaba_token_plan:__account_quota__:alibaba_token_plan"
)
ALIBABA_TOKEN_PLAN_RAW_CHOICES_HIDDEN_PARAM = "_alibaba_token_plan_raw_choices"
# Catalog metadata only: credential discovery and model admission are
# validated structurally, never against this static enumeration.
ALIBABA_TOKEN_PLAN_MODEL_IDS = frozenset(
    {
        "qwen3.8-max-preview",
        "qwen3.8-max",
        "qwen3.7-plus",
        "qwen3.7-max",
        "qwen3.6-flash",
        "deepseek-v4.1-flash",
        "deepseek-v4-pro",
        "glm-5.2",
    }
)


def _is_alibaba_subscription_identity(value: str) -> bool:
    """True for the sha256 hex identity, never for a raw key or instance code."""

    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def alibaba_token_plan_subscription_identity(instance_code: str) -> Optional[str]:
    """Hash a non-secret Token Plan instance code.

    The raw instance code is not returned. API keys, RAM secrets, and bearer
    tokens are not accepted as identity material.
    """

    if not isinstance(instance_code, str):
        return None
    normalized = instance_code.strip()
    if not normalized or any(ord(character) < 32 for character in normalized):
        return None
    material = f"alibaba-token-plan|instanceCode={normalized}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def alibaba_token_plan_account_quota_cooldown_key(
    subscription_identity: Optional[str],
) -> Optional[str]:
    """Scope account cooldown to one subscription identity."""

    identity = str(subscription_identity or "").strip().lower()
    if not _is_alibaba_subscription_identity(identity):
        return None
    return f"{ALIBABA_TOKEN_PLAN_ACCOUNT_QUOTA_COOLDOWN_KEY_PREFIX}:{identity}"


def _subscription_identity_from_settings() -> Optional[str]:
    """Read one non-secret instanceCode from the Qwen settings file."""

    settings_file = os.getenv(ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV)
    if not isinstance(settings_file, str) or not settings_file.strip():
        return None
    try:
        settings = json.loads(Path(settings_file.strip()).read_text(encoding="utf-8"))
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
    identities: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        if str(provider.get("baseUrl") or "").rstrip("/") != ALIBABA_TOKEN_PLAN_API_BASE:
            continue
        identity = alibaba_token_plan_subscription_identity(
            provider.get("instanceCode")
            if isinstance(provider.get("instanceCode"), str)
            else ""
        )
        if identity is not None:
            identities.add(identity)
    if len(identities) != 1:
        return None
    return next(iter(identities))


def resolve_alibaba_token_plan_subscription_identity() -> Optional[str]:
    """Resolve the inference lane's subscription identity.

    Precedence is the non-secret ``ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID``
    instance code, then a single ``instanceCode`` on the Token Plan settings
    entry. Credential env values are never hashed.
    """

    configured = os.getenv(ALIBABA_TOKEN_PLAN_SUBSCRIPTION_ID_ENV)
    if isinstance(configured, str) and configured.strip():
        return alibaba_token_plan_subscription_identity(configured)
    return _subscription_identity_from_settings()


def _candidate_subscription_identity(value: Any) -> Optional[str]:
    """Accept only the contract hash. Empty and non-string values are unknown."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not _is_alibaba_subscription_identity(normalized):
        return None
    return normalized


def _candidate_instance_identity(value: Any) -> Optional[str]:
    """Hash a candidate instance code. Empty and non-string values are unknown."""

    if not isinstance(value, str):
        return None
    return alibaba_token_plan_subscription_identity(value)


def subscription_identity_for_candidate(
    candidate: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Return the lane identity, or None when it is missing or mismatched.

    A present identity field that is empty, non-string, or not the contract
    hash stays unknown. When both the hash and instance code are supplied,
    they must name the same subscription. The process-wide identity is used
    only for an Alibaba Token Plan candidate that did not supply either field.
    """

    if not isinstance(candidate, Mapping):
        return None
    has_identity = "subscription_identity" in candidate
    has_instance = "instance_code" in candidate
    if has_identity or has_instance:
        identity = (
            _candidate_subscription_identity(candidate.get("subscription_identity"))
            if has_identity
            else None
        )
        if has_identity and identity is None:
            return None
        instance_identity = (
            _candidate_instance_identity(candidate.get("instance_code"))
            if has_instance
            else None
        )
        if has_instance and instance_identity is None:
            return None
        if (
            identity is not None
            and instance_identity is not None
            and identity != instance_identity
        ):
            return None
        return identity or instance_identity
    if candidate.get("provider") != ALIBABA_TOKEN_PLAN_PROVIDER_NAME:
        return None
    return resolve_alibaba_token_plan_subscription_identity()


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

    def get_response_hidden_params(self, response: Any) -> dict[str, Any]:
        """Retain provider-native choices before OpenAI SDK conversion."""

        if not isinstance(response, dict):
            return {}
        raw_choices = response.get("choices")
        if not isinstance(raw_choices, list):
            return {}
        return {ALIBABA_TOKEN_PLAN_RAW_CHOICES_HIDDEN_PARAM: raw_choices}

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """Retain provider-native choices for the auto-review gate."""

        response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        try:
            raw_body = raw_response.json()
        except Exception:
            return response
        if not isinstance(raw_body, dict):
            return response
        raw_choices = raw_body.get("choices")
        if not isinstance(raw_choices, list):
            return response

        hidden_params = getattr(response, "_hidden_params", None)
        if not isinstance(hidden_params, dict):
            hidden_params = {}
            response._hidden_params = hidden_params
        hidden_params.update(self.get_response_hidden_params(raw_body))
        return response

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
