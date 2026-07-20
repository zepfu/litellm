"""Managed Kimi Code chat-completions transformation."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union, cast

import httpx

from litellm._version import version as litellm_version
from litellm.llms.kimi_code.model_metadata import (
    MANAGED_KIMI_CODE_MODEL_IDS,
    is_managed_kimi_code_model_id,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream
from litellm.utils import _get_model_info_helper, convert_to_model_response_object

from ...openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)
from ...openai.common_utils import OpenAIError


KIMI_CODE_API_BASE = "https://api.kimi.com/coding/v1"
KIMI_CODE_CHAT_COMPLETIONS_URL = f"{KIMI_CODE_API_BASE}/chat/completions"
KIMI_CODE_CREDENTIAL_PATH_ENV = "LITELLM_KIMI_OAUTH_AUTH_FILE"
KIMI_CODE_DEFAULT_CREDENTIAL_PATH = "~/.kimi-code/credentials/kimi-code.json"
KIMI_CODE_REASONING_EFFORT_LEVELS = ("low", "high", "max")


class KimiCodeAuthenticationError(OpenAIError):
    """A redacted local authentication failure for managed Kimi Code credentials."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            status_code=401,
            message=(
                "Kimi Code OAuth credentials are unavailable or invalid: "
                f"{reason}. Refresh the existing Kimi Code CLI credential in place."
            ),
            headers={},
        )


class KimiCodeChatConfig(OpenAIGPTConfig):
    """
    Kimi Code's managed OAuth-backed OpenAI-compatible chat endpoint.

    This config deliberately reads, but never refreshes or writes, the local Kimi
    Code credential on every request. It is separate from the API-key Moonshot
    provider and its https://api.moonshot.ai/v1 endpoint.
    """

    @staticmethod
    def _model_id(model: str) -> str:
        model_parts = model.split("/", maxsplit=1)
        if len(model_parts) == 2 and model_parts[0] != "kimi_code":
            model_id = model
        else:
            model_id = model_parts[-1]
        if not is_managed_kimi_code_model_id(model_id):
            supported_models = ", ".join(sorted(MANAGED_KIMI_CODE_MODEL_IDS))
            raise ValueError(
                f"Unsupported managed Kimi Code model {model!r}. " f"Supported model IDs: {supported_models}."
            )
        return model_id

    @staticmethod
    def _credential_path() -> Path:
        return Path(
            os.getenv(
                KIMI_CODE_CREDENTIAL_PATH_ENV,
                KIMI_CODE_DEFAULT_CREDENTIAL_PATH,
            )
        ).expanduser()

    @classmethod
    def _get_access_token(cls) -> str:
        credential_path = cls._credential_path()
        try:
            payload = json.loads(credential_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KimiCodeAuthenticationError("credential file is missing") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise KimiCodeAuthenticationError("credential file is unreadable or malformed") from exc

        if not isinstance(payload, dict):
            raise KimiCodeAuthenticationError("credential payload is malformed")

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise KimiCodeAuthenticationError("access token is missing")

        expires_at = cls._parse_expiry(payload.get("expires_at"))
        if expires_at is None:
            raise KimiCodeAuthenticationError("credential expiry is missing or malformed")
        if expires_at <= time.time():
            raise KimiCodeAuthenticationError("access token is expired")

        return access_token

    @staticmethod
    def _parse_expiry(value: object) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value) / 1000 if value > 10_000_000_000 else float(value)
        if isinstance(value, str):
            try:
                return KimiCodeChatConfig._parse_expiry(float(value))
            except ValueError:
                try:
                    normalized_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
                    parsed_value = datetime.fromisoformat(normalized_value)
                    if parsed_value.tzinfo is None:
                        parsed_value = parsed_value.replace(tzinfo=timezone.utc)
                    return parsed_value.timestamp()
                except ValueError:
                    return None
        return None

    @staticmethod
    def _user_agent() -> str:
        version = litellm_version if litellm_version else "unknown"
        return f"litellm/{version}"

    @classmethod
    def _managed_headers(cls, headers: object, include_authorization: bool) -> dict:
        if headers is None:
            managed_headers = {}
        elif isinstance(headers, dict):
            managed_headers = {
                name: value
                for name, value in headers.items()
                if not name.lower().startswith("x-msh-") and name.lower() not in {"authorization", "user-agent"}
            }
        else:
            raise ValueError("Kimi Code extra_headers must be an object")

        managed_headers["User-Agent"] = cls._user_agent()
        if not any(name.lower() == "content-type" for name in managed_headers):
            managed_headers["Content-Type"] = "application/json"
        if include_authorization:
            managed_headers["Authorization"] = f"Bearer {cls._get_access_token()}"
        return managed_headers

    @classmethod
    def _supported_reasoning_efforts(cls, model: str) -> Tuple[str, ...]:
        model_id = cls._model_id(model)
        try:
            model_info = _get_model_info_helper(
                model=model_id,
                custom_llm_provider="kimi_code",
            )
        except Exception:
            return ()
        return tuple(
            effort
            for effort in KIMI_CODE_REASONING_EFFORT_LEVELS
            if model_info.get(f"supports_{effort}_reasoning_effort") is True
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Ignore external API-key and base overrides: this managed provider is
        # authenticated only with the current Kimi Code OAuth credential.
        return KIMI_CODE_API_BASE, self._get_access_token()

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        self._model_id(model)
        return KIMI_CODE_CHAT_COMPLETIONS_URL

    def get_supported_openai_params(self, model: str) -> list:
        supported_reasoning_efforts = self._supported_reasoning_efforts(model)
        supported_params = [
            "frequency_penalty",
            "extra_headers",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "prompt_cache_key",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
        ]
        if supported_reasoning_efforts:
            supported_params.append("reasoning_effort")
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        supported_reasoning_efforts = self._supported_reasoning_efforts(model)

        if "max_completion_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params["max_completion_tokens"]
        elif "max_completion_tokens" not in optional_params and "max_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params["max_tokens"]

        for param, value in non_default_params.items():
            if param in {"max_tokens", "max_completion_tokens"}:
                continue
            if param == "reasoning_effort":
                if value not in supported_reasoning_efforts:
                    supported = ", ".join(supported_reasoning_efforts) or "none"
                    raise ValueError(
                        "Kimi Code does not support reasoning_effort={!r} for model "
                        "{}. Supported efforts: {}.".format(value, model, supported)
                    )
                existing_extra_body = optional_params.get("extra_body")
                if existing_extra_body is None:
                    extra_body = {}
                elif isinstance(existing_extra_body, dict):
                    extra_body = dict(existing_extra_body)
                else:
                    raise ValueError("Kimi Code extra_body must be an object")
                extra_body["thinking"] = {
                    "type": "enabled",
                    "effort": value,
                    "keep": "all",
                }
                optional_params["extra_body"] = extra_body
            elif param in supported_params:
                optional_params[param] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        self._model_id(model)
        optional_params = dict(optional_params)
        combined_headers = dict(headers)
        caller_extra_headers = optional_params.get("extra_headers")
        if caller_extra_headers is not None:
            if not isinstance(caller_extra_headers, dict):
                raise ValueError("Kimi Code extra_headers must be an object")
            combined_headers.update(caller_extra_headers)
        optional_params["extra_headers"] = self._managed_headers(
            combined_headers,
            include_authorization=True,
        )
        if optional_params.get("stream"):
            caller_stream_options = optional_params.get("stream_options", {})
            if not isinstance(caller_stream_options, dict):
                raise ValueError("Kimi Code stream_options must be an object")
            optional_params = dict(optional_params)
            optional_params["stream_options"] = {
                **caller_stream_options,
                "include_usage": True,
            }
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

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
        self._model_id(model)
        return self._managed_headers(headers, include_authorization=True)

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return KimiCodeChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
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
        try:
            completion_response = raw_response.json()
        except Exception:
            return super().transform_response(
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

        self._promote_choice_usage(completion_response)
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=completion_response,
            additional_args={"complete_input_dict": request_data},
        )
        response_headers = dict(raw_response.headers)
        return cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=completion_response,
                model_response_object=model_response,
                hidden_params={"headers": response_headers},
                _response_headers=response_headers,
            ),
        )

    @staticmethod
    def _promote_choice_usage(response: object) -> None:
        if not isinstance(response, dict) or response.get("usage") is not None:
            return
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return
        usage = first_choice.get("usage")
        if isinstance(usage, dict):
            response["usage"] = usage


class KimiCodeChatCompletionStreamingHandler(OpenAIChatCompletionStreamingHandler):
    """Normalize Kimi Code's nested usage chunks before shared OpenAI parsing."""

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        normalized_chunk = dict(chunk)
        original_choices = chunk.get("choices", [])
        normalized_choices = []
        promoted_usage = normalized_chunk.get("usage")

        if isinstance(original_choices, list):
            for index, choice in enumerate(original_choices):
                if not isinstance(choice, dict):
                    normalized_choices.append(choice)
                    continue
                normalized_choice = dict(choice)
                if index == 0 and promoted_usage is None:
                    nested_usage = normalized_choice.pop("usage", None)
                    if isinstance(nested_usage, dict):
                        promoted_usage = nested_usage
                normalized_choices.append(normalized_choice)
            normalized_chunk["choices"] = normalized_choices

        if isinstance(promoted_usage, dict):
            normalized_chunk["usage"] = promoted_usage
            if not normalized_choices:
                normalized_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": None}]

        return super().chunk_parser(normalized_chunk)
