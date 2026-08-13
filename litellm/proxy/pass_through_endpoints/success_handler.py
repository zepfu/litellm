from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union
from urllib.parse import urlparse

import httpx

import litellm

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.redact_messages import (
    redact_message_input_output_from_custom_logger,
    redact_message_input_output_from_logging,
)
from litellm.proxy.aawm_route_logging import record_aawm_route_rollup_turn
from litellm.litellm_core_utils.litellm_logging import (
    Logging as LiteLLMLoggingObj,
    emit_standard_logging_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardPassThroughResponseObject
from litellm.utils import executor as thread_pool_executor

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
)
from .llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
)
from .llm_provider_handlers.cursor_passthrough_logging_handler import (
    CursorPassthroughLoggingHandler,
)
from .llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from .provider_failure_classifiers.cohere import is_cohere_api_url

_ANTHROPIC_RATE_LIMIT_HEADER_PREFIXES = (
    "anthropic-ratelimit-",
    "x-ratelimit-",
)
_ANTHROPIC_RATE_LIMIT_HEADER_NAMES = {
    "retry-after",
}
_XAI_OAUTH_RATE_LIMIT_HEADER_PREFIXES = (
    "x-ratelimit-",
)
_XAI_OAUTH_RATE_LIMIT_HEADER_NAMES = {
    "retry-after",
}
_COHERE_DIRECT_ROUTE_FAMILY = "codex_cohere_chat_completions_adapter"
_COHERE_LOCAL_OBSERVATION_SOURCE = "locally_counted"


class PassThroughEndpointLogging:
    def __init__(self):
        self.TRACKED_VERTEX_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "predict",
            "rawPredict",
            "streamRawPredict",
            "search",
            "batchPredictionJobs",
            "predictLongRunning",
        ]

        # Anthropic
        self.TRACKED_ANTHROPIC_ROUTES = ["/messages", "/v1/messages/batches"]

        # Cohere
        self.TRACKED_COHERE_ROUTES = ["/v2/chat", "/v1/embed"]
        self.assemblyai_passthrough_logging_handler = (
            AssemblyAIPassthroughLoggingHandler()
        )

        # Langfuse
        self.TRACKED_LANGFUSE_ROUTES = ["/langfuse/"]

        # Gemini
        self.TRACKED_GEMINI_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "predictLongRunning",
        ]

        # Cursor Cloud Agents
        self.TRACKED_CURSOR_ROUTES = [
            "/v0/agents",
            "/v0/me",
            "/v0/models",
            "/v0/repositories",
        ]

        # Vertex AI Live API WebSocket
        self.TRACKED_VERTEX_AI_LIVE_ROUTES = ["/vertex_ai/live"]

    @staticmethod
    def _ensure_metadata(kwargs: dict) -> dict:
        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
            kwargs["litellm_params"] = litellm_params

        metadata = litellm_params.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            litellm_params["metadata"] = metadata
        return metadata

    @staticmethod
    def _sanitize_anthropic_rate_limit_headers(
        response_headers: httpx.Headers,
    ) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for header_name, header_value in response_headers.items():
            normalized_name = str(header_name).lower()
            if not (
                normalized_name.startswith(_ANTHROPIC_RATE_LIMIT_HEADER_PREFIXES)
                or normalized_name in _ANTHROPIC_RATE_LIMIT_HEADER_NAMES
            ):
                continue
            sanitized[normalized_name] = str(header_value)
        if sanitized:
            sanitized["source"] = "anthropic_response_headers"
        return sanitized

    @staticmethod
    def _sanitize_xai_oauth_rate_limit_headers(
        response_headers: httpx.Headers,
    ) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for header_name, header_value in response_headers.items():
            normalized_name = str(header_name).lower()
            if not (
                normalized_name.startswith(_XAI_OAUTH_RATE_LIMIT_HEADER_PREFIXES)
                or normalized_name in _XAI_OAUTH_RATE_LIMIT_HEADER_NAMES
            ):
                continue
            sanitized[normalized_name] = str(header_value)
        if sanitized:
            sanitized["source"] = "xai_oauth_response_headers"
        return sanitized

    @staticmethod
    def _is_xai_oauth_metadata(metadata: dict) -> bool:
        if metadata.get("xai_oauth_managed") is True:
            return True
        if metadata.get("grok_native_oauth_managed") is True:
            return True
        credential_family = str(metadata.get("credential_family") or "").lower()
        route_family = str(
            metadata.get("passthrough_route_family")
            or metadata.get("route_family")
            or ""
        ).lower()
        return (
            credential_family == "xai_oauth"
            or "xai_oauth" in route_family
            or metadata.get("xai_oauth_public_model") is not None
        )

    def _record_upstream_rate_limit_headers_metadata(
        self,
        kwargs: dict,
        *,
        httpx_response: httpx.Response,
        url_route: str,
        custom_llm_provider: Optional[str],
    ) -> None:
        metadata = self._ensure_metadata(kwargs)
        if custom_llm_provider == "xai" and self._is_xai_oauth_metadata(metadata):
            sanitized_headers = self._sanitize_xai_oauth_rate_limit_headers(
                httpx_response.headers
            )
            if sanitized_headers:
                metadata["xai_oauth_response_headers"] = sanitized_headers
            return
        if not (
            custom_llm_provider == "anthropic"
            or self.is_anthropic_route(url_route)
        ):
            return
        sanitized_headers = self._sanitize_anthropic_rate_limit_headers(
            httpx_response.headers
        )
        if sanitized_headers:
            metadata["anthropic_response_headers"] = sanitized_headers

    @staticmethod
    def _cohere_context_value(
        *,
        name: str,
        kwargs: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> Any:
        model_call_details = getattr(logging_obj, "model_call_details", None)
        metadata = {}
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict) and isinstance(
            litellm_params.get("metadata"), dict
        ):
            metadata = litellm_params["metadata"]

        for values in (kwargs, metadata, model_call_details):
            if isinstance(values, dict) and name in values:
                value = values[name]
                if value is not None:
                    return value
        value = getattr(logging_obj, name, None)
        if value is not None:
            return value
        return None

    def _is_direct_cohere_success(
        self,
        *,
        httpx_response: httpx.Response,
        url_route: str,
        custom_llm_provider: Optional[str],
        kwargs: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> bool:
        status_code = getattr(httpx_response, "status_code", None)
        if status_code != 200:
            return False

        parsed_url = urlparse(url_route)
        path = parsed_url.path if parsed_url.scheme else url_route
        if not parsed_url.scheme or not is_cohere_api_url(url_route):
            return False
        normalized_path = path[:-1] if path.endswith("/") else path
        if normalized_path != "/v2/chat":
            return False
        if str(custom_llm_provider or "").strip().lower() != "cohere":
            return False
        if not self.is_cohere_route(
            url_route,
            custom_llm_provider=custom_llm_provider,
        ):
            return False

        route_family = self._cohere_context_value(
            name="passthrough_route_family",
            kwargs=kwargs,
            logging_obj=logging_obj,
        )
        if route_family is None:
            route_family = self._cohere_context_value(
                name="route_family",
                kwargs=kwargs,
                logging_obj=logging_obj,
            )
        return route_family == _COHERE_DIRECT_ROUTE_FAMILY

    @staticmethod
    def _cohere_text_context_value(
        *,
        name: str,
        kwargs: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> Optional[str]:
        value = PassThroughEndpointLogging._cohere_context_value(
            name=name,
            kwargs=kwargs,
            logging_obj=logging_obj,
        )
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _cohere_rate_limit_observations(
        *,
        state: Any,
        model: Optional[str],
        observed_at: datetime,
    ) -> list[dict[str, Any]]:
        observed_at_utc = observed_at
        if observed_at_utc.tzinfo is None:
            observed_at_utc = observed_at_utc.replace(tzinfo=timezone.utc)
        else:
            observed_at_utc = observed_at_utc.astimezone(timezone.utc)

        monthly_limit = max(0, int(getattr(state, "monthly_limit", 0) or 0))
        monthly_used = max(0, int(getattr(state, "monthly_used", 0) or 0))
        monthly_remaining_value = getattr(state, "monthly_remaining", None)
        if monthly_remaining_value is None:
            monthly_remaining_value = monthly_limit - monthly_used
        monthly_remaining = max(0, int(monthly_remaining_value or 0))
        if monthly_limit > 0:
            monthly_remaining = min(monthly_limit, monthly_remaining)

        rpm_limit_value = getattr(state, "rpm_limit", None)
        rpm_remaining_value = getattr(state, "rpm_remaining", None)
        rpm_limit = (
            None
            if rpm_limit_value is None
            else max(0, int(rpm_limit_value or 0))
        )
        rpm_remaining = (
            None
            if rpm_remaining_value is None
            else max(0, int(rpm_remaining_value or 0))
        )
        rpm_used_value = getattr(state, "rpm_used", None)
        rpm_used = (
            None
            if rpm_used_value is None
            else max(0, int(rpm_used_value or 0))
        )
        if rpm_limit is None or rpm_remaining is None or rpm_limit <= 0:
            rpm_limit = None
            rpm_remaining = None
        else:
            rpm_remaining = min(rpm_limit, rpm_remaining)

        month_end = getattr(state, "month_end", None)
        if not isinstance(month_end, datetime):
            month_end = observed_at_utc
        if month_end.tzinfo is None:
            month_end = month_end.replace(tzinfo=timezone.utc)
        else:
            month_end = month_end.astimezone(timezone.utc)

        def _observation(
            *,
            quota_type: str,
            quota_period: str,
            remaining: Optional[int],
            limit: Optional[int],
            quota_used: Optional[int],
            expected_reset_at: datetime,
            window_minutes: Optional[int],
        ) -> dict[str, Any]:
            quota_is_known = remaining is not None and limit is not None and limit > 0
            remaining_pct = (
                round((remaining / limit) * 100, 4) if quota_is_known else None
            )
            exhausted = remaining <= 0 if quota_is_known else None
            return {
                "provider": "cohere",
                "model": model,
                "quota_key": "cohere_trial_default",
                "quota_type": quota_type,
                "limit_scope": "credential",
                "quota_period": quota_period,
                "window_minutes": window_minutes,
                "quota_used": quota_used,
                "remaining_pct": remaining_pct,
                "observed_at": observed_at_utc.isoformat(),
                "expected_reset_at": expected_reset_at.isoformat(),
                "status": (
                    "exhausted"
                    if exhausted is True
                    else "available"
                    if exhausted is False
                    else "unknown"
                ),
                "exhausted": exhausted,
                "source": _COHERE_LOCAL_OBSERVATION_SOURCE,
            }

        return [
            _observation(
                quota_type="monthly",
                quota_period="calendar_month",
                remaining=monthly_remaining,
                limit=monthly_limit,
                quota_used=monthly_used,
                expected_reset_at=month_end,
                window_minutes=None,
            ),
            _observation(
                quota_type="rpm",
                quota_period="rolling",
                remaining=rpm_remaining,
                limit=rpm_limit,
                quota_used=(
                    rpm_used
                    if rpm_limit is not None and rpm_remaining is not None
                    else None
                ),
                expected_reset_at=observed_at_utc + timedelta(minutes=1),
                window_minutes=1,
            ),
        ]

    async def _record_direct_cohere_success(
        self,
        *,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        custom_llm_provider: Optional[str],
        end_time: datetime,
        result: str,
        request_body: dict,
        kwargs: dict,
    ) -> None:
        if result != "complete":
            return

        if not self._is_direct_cohere_success(
            httpx_response=httpx_response,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
            kwargs=kwargs,
            logging_obj=logging_obj,
        ):
            return

        litellm_call_id = self._cohere_text_context_value(
            name="litellm_call_id",
            kwargs=kwargs,
            logging_obj=logging_obj,
        )
        if litellm_call_id is None:
            verbose_proxy_logger.warning(
                "Cohere accepted-call accounting skipped: missing stable call id"
            )
            return

        model = request_body.get("model") if isinstance(request_body, dict) else None
        if not isinstance(model, str) or not model.strip():
            model = self._cohere_text_context_value(
                name="model",
                kwargs=kwargs,
                logging_obj=logging_obj,
            )
        else:
            model = model.strip()

        session_id = self._cohere_text_context_value(
            name="session_id",
            kwargs=kwargs,
            logging_obj=logging_obj,
        )
        trace_id = self._cohere_text_context_value(
            name="trace_id",
            kwargs=kwargs,
            logging_obj=logging_obj,
        )

        try:
            from litellm.integrations.aawm_session_history.cohere_accepted_calls import (  # noqa: PLC0415
                CohereAcceptedCallState,
                record_cohere_accepted_call,
            )

            state = await record_cohere_accepted_call(
                litellm_call_id=litellm_call_id,
                accepted_at=end_time,
                model=model,
                session_id=session_id,
                trace_id=trace_id,
                source=_COHERE_DIRECT_ROUTE_FAMILY,
            )
            if not isinstance(state, CohereAcceptedCallState):
                raise TypeError("unexpected Cohere accepted-call state")
        except Exception as exc:
            verbose_proxy_logger.warning(
                "Cohere accepted-call accounting failed: %s",
                type(exc).__name__,
            )
            return

        observations = self._cohere_rate_limit_observations(
            state=state,
            model=model,
            observed_at=end_time,
        )
        metadata = self._ensure_metadata(kwargs)
        existing_observations = metadata.get("rate_limit_observations")
        if not isinstance(existing_observations, list):
            existing_observations = []
        existing_observations.extend(observations)
        metadata["rate_limit_observations"] = existing_observations
        kwargs["rate_limit_observations"] = existing_observations

    @staticmethod
    def _ensure_standard_logging_object(
        *,
        logging_obj: LiteLLMLoggingObj,
        model_call_details: dict,
        current_kwargs: dict,
        current_result: Any,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
    ) -> None:
        """
        Attach a standard_logging_object for non-streaming pass-through callbacks.

        Prefer an object already built by a provider handler. Otherwise build one
        with the same helper the native LiteLLM success path uses, without calling
        async_success_handler (which would re-run callbacks / spend persistence).
        """
        existing = current_kwargs.get("standard_logging_object")
        if existing is None and isinstance(model_call_details, dict):
            existing = model_call_details.get("standard_logging_object")

        # Native builder reads logging_obj.model_call_details; only mutate that dict
        # when it is a real mapping so callback kwargs stay free of bookkeeping.
        logging_details = getattr(logging_obj, "model_call_details", None)
        if not isinstance(logging_details, dict):
            logging_details = None

        if existing is None:
            if logging_details is not None:
                logging_details.setdefault("log_event_type", "successful_api_call")
                logging_details["end_time"] = end_time
                logging_details["cache_hit"] = cache_hit
                for key in (
                    "response_cost",
                    "model",
                    "custom_llm_provider",
                    "litellm_params",
                    "passthrough_logging_payload",
                    "standard_pass_through_logging_payload",
                    "messages",
                    "prompt",
                    "input",
                ):
                    if key in current_kwargs and (
                        key not in logging_details
                        or logging_details.get(key) is None
                    ):
                        logging_details[key] = current_kwargs[key]
                if current_kwargs.get("response_cost") is not None:
                    logging_details["response_cost"] = current_kwargs["response_cost"]

            build_fn = getattr(logging_obj, "_build_standard_logging_payload", None)
            if callable(build_fn):
                try:
                    payload = build_fn(current_result, start_time, end_time)
                except Exception as exc:
                    verbose_proxy_logger.warning(
                        "Pass-through standard_logging_object build failed: %s",
                        exc,
                    )
                    payload = None
                # StandardLoggingPayload is a TypedDict/dict; ignore non-dict mocks.
                if isinstance(payload, dict):
                    existing = payload
                    emit_standard_logging_payload(payload)

        if existing is not None:
            current_kwargs["standard_logging_object"] = existing
            if isinstance(model_call_details, dict):
                model_call_details["standard_logging_object"] = existing
            if logging_details is not None:
                logging_details["standard_logging_object"] = existing
            # Keep response_cost visible on kwargs for spend-track fallbacks.
            if (
                current_kwargs.get("response_cost") is None
                and isinstance(existing, dict)
                and existing.get("response_cost") is not None
            ):
                current_kwargs["response_cost"] = existing.get("response_cost")

    async def _handle_logging(
        self,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: Union[
            StandardPassThroughResponseObject,
            PassThroughEndpointLoggingResultValues,
            dict,
        ],
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """Run pass-through logging hooks using the same callback contracts as normal LiteLLM success handling."""
        call_type = getattr(logging_obj, "call_type", "pass_through_endpoint")
        current_kwargs = dict(kwargs)
        current_kwargs.setdefault("standard_callback_dynamic_params", {})
        current_result: Union[
            StandardPassThroughResponseObject,
            PassThroughEndpointLoggingResultValues,
            dict,
        ] = standard_logging_response_object

        # Match native async_success_handler ordering: global redaction first so
        # turn_off_message_logging is honored for non-streaming pass-through.
        model_call_details = getattr(logging_obj, "model_call_details", None)
        if not isinstance(model_call_details, dict):
            model_call_details = current_kwargs
        else:
            model_call_details.setdefault(
                "standard_callback_dynamic_params",
                current_kwargs.get("standard_callback_dynamic_params", {}),
            )

        # Build / attach standard_logging_object before redaction + callbacks so
        # non-streaming pass-through matches native LiteLLM payload parity.
        self._ensure_standard_logging_object(
            logging_obj=logging_obj,
            model_call_details=model_call_details,
            current_kwargs=current_kwargs,
            current_result=current_result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
        )

        current_result = redact_message_input_output_from_logging(
            model_call_details=model_call_details,
            result=current_result,
        )
        if model_call_details is not current_kwargs:
            for _key in ("messages", "prompt", "input"):
                if _key in model_call_details:
                    current_kwargs[_key] = model_call_details[_key]

        sync_callbacks = logging_obj.get_combined_callback_list(
            dynamic_success_callbacks=logging_obj.dynamic_success_callbacks,
            global_callbacks=litellm.success_callback,
        )
        for callback in sync_callbacks:
            if isinstance(callback, CustomLogger):
                current_result = redact_message_input_output_from_custom_logger(
                    result=current_result,
                    litellm_logging_obj=logging_obj,
                    custom_logger=callback,
                )
            logging_hook = getattr(callback, "logging_hook", None)
            if callable(logging_hook):
                try:
                    hook_result = logging_hook(
                        current_kwargs,
                        current_result,
                        call_type,
                    )
                    if (
                        isinstance(hook_result, tuple)
                        and len(hook_result) == 2
                        and isinstance(hook_result[0], dict)
                    ):
                        current_kwargs, current_result = hook_result
                except Exception as exc:
                    verbose_proxy_logger.warning(
                        "Pass-through logging_hook failed for callback=%s: %s",
                        callback,
                        exc,
                    )

        for callback in sync_callbacks:
            log_success_event = getattr(callback, "log_success_event", None)
            if callable(log_success_event):
                thread_pool_executor.submit(
                    log_success_event,
                    current_kwargs,
                    current_result,
                    start_time,
                    end_time,
                )

        async_callbacks = logging_obj.get_combined_callback_list(
            dynamic_success_callbacks=logging_obj.dynamic_async_success_callbacks,
            global_callbacks=litellm._async_success_callback,
        )
        for callback in async_callbacks:
            # Isolate each async callback so one failure cannot abort later
            # callbacks or pass_through_async_success_handler route-rollup work.
            try:
                if isinstance(callback, CustomLogger):
                    current_result = redact_message_input_output_from_custom_logger(
                        result=current_result,
                        litellm_logging_obj=logging_obj,
                        custom_logger=callback,
                    )
                async_logging_hook = getattr(callback, "async_logging_hook", None)
                if callable(async_logging_hook):
                    try:
                        hook_result = await async_logging_hook(
                            kwargs=current_kwargs,
                            result=current_result,
                            call_type=call_type,
                        )
                        if (
                            isinstance(hook_result, tuple)
                            and len(hook_result) == 2
                            and isinstance(hook_result[0], dict)
                        ):
                            current_kwargs, current_result = hook_result
                    except Exception as exc:
                        verbose_proxy_logger.warning(
                            "Pass-through async_logging_hook failed for callback=%s: %s",
                            callback,
                            exc,
                        )

                async_log_success_event = getattr(
                    callback, "async_log_success_event", None
                )
                if callable(async_log_success_event):
                    try:
                        await async_log_success_event(
                            current_kwargs,
                            current_result,
                            start_time,
                            end_time,
                        )
                    except Exception as exc:
                        verbose_proxy_logger.warning(
                            "Pass-through async_log_success_event failed for callback=%s: %s",
                            callback,
                            exc,
                        )
            except Exception as exc:
                verbose_proxy_logger.warning(
                    "Pass-through async success callback failed for callback=%s: %s",
                    callback,
                    exc,
                )

    def normalize_llm_passthrough_logging_payload(
        self,
        httpx_response: httpx.Response,
        response_body: Optional[dict],
        request_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        return_dict = {
            "standard_logging_response_object": None,
            "kwargs": kwargs,
        }
        standard_logging_response_object: Optional[Any] = None

        adapted_openai_url_route = self._get_adapted_openai_logging_route(
            response_body=response_body,
            custom_llm_provider=custom_llm_provider,
            url_route=url_route,
        )

        if self.is_gemini_route(url_route, custom_llm_provider):
            gemini_passthrough_logging_handler_result = (
                GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    custom_llm_provider=custom_llm_provider or "gemini",
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                gemini_passthrough_logging_handler_result["result"]
            )
            kwargs = gemini_passthrough_logging_handler_result["kwargs"]
        elif self.is_vertex_route(url_route):
            vertex_passthrough_logging_handler_result = (
                VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                    httpx_response=httpx_response,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                vertex_passthrough_logging_handler_result["result"]
            )
            kwargs = vertex_passthrough_logging_handler_result["kwargs"]
        elif adapted_openai_url_route is not None or (
            self.is_openai_route(url_route) and self._is_supported_openai_endpoint(url_route)
        ):
            from .llm_provider_handlers.openai_passthrough_logging_handler import (
                OpenAIPassthroughLoggingHandler,
            )

            openai_passthrough_logging_handler_result = (
                OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=adapted_openai_url_route or url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            )

            standard_logging_response_object = (
                openai_passthrough_logging_handler_result["result"]
            )
            kwargs = openai_passthrough_logging_handler_result["kwargs"]
        elif self.is_anthropic_route(url_route):
            anthropic_passthrough_logging_handler_result = (
                AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )

            standard_logging_response_object = (
                anthropic_passthrough_logging_handler_result["result"]
            )
            kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
        elif custom_llm_provider == "xai" and self.is_openai_route(url_route):
            return return_dict
        elif self.is_cohere_route(
            url_route,
            custom_llm_provider=custom_llm_provider,
        ):
            cohere_passthrough_logging_handler_result = (
                CoherePassthroughLoggingHandler.cohere_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                cohere_passthrough_logging_handler_result["result"]
            )
            kwargs = cohere_passthrough_logging_handler_result["kwargs"]
        elif self.is_cursor_route(url_route, custom_llm_provider):
            cursor_passthrough_logging_handler_result = (
                CursorPassthroughLoggingHandler.cursor_passthrough_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                cursor_passthrough_logging_handler_result["result"]
            )
            kwargs = cursor_passthrough_logging_handler_result["kwargs"]

        return_dict["standard_logging_response_object"] = standard_logging_response_object
        return_dict["kwargs"] = kwargs
        return return_dict

    async def pass_through_async_success_handler(
        self,
        httpx_response: httpx.Response,
        response_body: Optional[dict],
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        standard_logging_response_object: Optional[
            PassThroughEndpointLoggingResultValues
        ] = None
        logging_obj.model_call_details[
            "passthrough_logging_payload"
        ] = passthrough_logging_payload
        if self.is_assemblyai_route(url_route):
            if (
                AssemblyAIPassthroughLoggingHandler._should_log_request(
                    httpx_response.request.method
                )
                is not True
            ):
                return
            self.assemblyai_passthrough_logging_handler.assemblyai_passthrough_logging_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
            return
        elif self.is_langfuse_route(url_route):
            # Don't log langfuse pass-through requests
            return
        else:
            self._record_upstream_rate_limit_headers_metadata(
                kwargs,
                httpx_response=httpx_response,
                url_route=url_route,
                custom_llm_provider=custom_llm_provider,
            )
            normalized_llm_passthrough_logging_payload = (
                self.normalize_llm_passthrough_logging_payload(
                    httpx_response=httpx_response,
                    response_body=response_body,
                    request_body=request_body,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    custom_llm_provider=custom_llm_provider,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                normalized_llm_passthrough_logging_payload[
                    "standard_logging_response_object"
                ]
            )
            kwargs = normalized_llm_passthrough_logging_payload["kwargs"]
        if standard_logging_response_object is None:
            standard_logging_response_object = StandardPassThroughResponseObject(
                response=httpx_response.text
            )

        kwargs = self._set_cost_per_request(
            logging_obj=logging_obj,
            passthrough_logging_payload=passthrough_logging_payload,
            kwargs=kwargs,
        )

        # Attach any validated Codex auto-review decision to private callback
        # kwargs (and record the rollup turn) BEFORE sync/async success
        # callbacks run, so the AawmAgentIdentity session-history success
        # handler sees the parser-produced event.
        record_aawm_route_rollup_turn(
            kwargs,
            response_body=response_body,
        )
        await self._record_direct_cohere_success(
            httpx_response=httpx_response,
            logging_obj=logging_obj,
            url_route=url_route,
            custom_llm_provider=custom_llm_provider,
            end_time=end_time,
            result=result,
            request_body=request_body,
            kwargs=kwargs,
        )
        await self._handle_logging(
            logging_obj=logging_obj,
            standard_logging_response_object=standard_logging_response_object,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            standard_pass_through_logging_payload=passthrough_logging_payload,
            **kwargs,
        )

    def is_vertex_route(self, url_route: str):
        for route in self.TRACKED_VERTEX_ROUTES:
            if route in url_route:
                return True
        return False

    def is_anthropic_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        path = parsed_url.path if parsed_url.scheme else url_route
        if not path or "/count_tokens" in path:
            return False
        for route in self.TRACKED_ANTHROPIC_ROUTES:
            if route in path:
                return True
        return False

    def is_cohere_route(
        self,
        url_route: str,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        parsed_url = urlparse(url_route)
        path = parsed_url.path if parsed_url.scheme else url_route
        is_supported_path = any(
            path == route or path.startswith(f"{route}/")
            for route in self.TRACKED_COHERE_ROUTES
        )
        if not is_supported_path:
            return False
        if parsed_url.scheme:
            return is_cohere_api_url(url_route)
        return str(custom_llm_provider or "").strip().lower() == "cohere"

    def is_assemblyai_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        if parsed_url.hostname == "api.assemblyai.com":
            return True
        elif "/transcript" in parsed_url.path:
            return True
        return False

    def is_langfuse_route(self, url_route: str):
        parsed_url = urlparse(url_route)
        for route in self.TRACKED_LANGFUSE_ROUTES:
            if route in parsed_url.path:
                return True
        return False

    def is_vertex_ai_live_route(self, url_route: str):
        """Check if the URL route is a Vertex AI Live API WebSocket route."""
        if not url_route:
            return False
        for route in self.TRACKED_VERTEX_AI_LIVE_ROUTES:
            if route in url_route:
                return True
        return False

    def is_cursor_route(
        self, url_route: str, custom_llm_provider: Optional[str] = None
    ):
        """Check if the URL route is a Cursor Cloud Agents API route."""
        if custom_llm_provider == "cursor":
            return True
        parsed_url = urlparse(url_route)
        if parsed_url.hostname and "api.cursor.com" in parsed_url.hostname:
            return True
        for route in self.TRACKED_CURSOR_ROUTES:
            if route in url_route:
                path = parsed_url.path if parsed_url.scheme else url_route
                if path.startswith("/v0/"):
                    return custom_llm_provider == "cursor"
        return False

    def is_openai_route(self, url_route: str):
        """Check if the URL route is an OpenAI-compatible API route."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
                or parsed_url.hostname == "integrate.api.nvidia.com"
                or parsed_url.hostname == "ai.api.nvidia.com"
                or parsed_url.hostname == "openrouter.ai"
                or parsed_url.hostname.endswith(".openrouter.ai")
                or parsed_url.hostname == "api.x.ai"
                or parsed_url.hostname == "cli-chat-proxy.grok.com"
            )
        )

    def is_gemini_route(
        self, url_route: str, custom_llm_provider: Optional[str] = None
    ):
        """Check if the URL route is a Gemini API route."""
        normalized_url_route = url_route.lower()
        for route in self.TRACKED_GEMINI_ROUTES:
            if route.lower() in normalized_url_route and custom_llm_provider == "gemini":
                return True
        return False

    def _get_adapted_openai_logging_route(
        self,
        response_body: Optional[dict],
        custom_llm_provider: Optional[str],
        url_route: Optional[str] = None,
    ) -> Optional[str]:
        if custom_llm_provider not in {"openai", "openrouter", "nvidia_nim", "xai"}:
            return None
        if not isinstance(response_body, dict):
            return None

        is_chat_completions_payload = isinstance(response_body.get("choices"), list)
        is_responses_payload = response_body.get("object") == "response" or isinstance(
            response_body.get("output"), list
        )
        is_embeddings_payload = self._is_openai_compatible_embedding_payload(
            response_body
        )
        if not (
            is_chat_completions_payload
            or is_responses_payload
            or is_embeddings_payload
        ):
            return None

        if custom_llm_provider == "openrouter":
            base_url = "https://openrouter.ai/api"
        elif custom_llm_provider == "nvidia_nim":
            base_url = "https://integrate.api.nvidia.com"
        elif custom_llm_provider == "xai":
            base_url = "https://api.x.ai"
        else:
            base_url = "https://api.openai.com"

        if is_responses_payload:
            return f"{base_url}/v1/responses"
        if is_embeddings_payload:
            parsed_url = urlparse(url_route or "")
            if parsed_url.path and "/embeddings" not in parsed_url.path:
                return None
            return f"{base_url}/v1/embeddings"
        return f"{base_url}/v1/chat/completions"

    @staticmethod
    def _is_openai_compatible_embedding_payload(response_body: dict) -> bool:
        data = response_body.get("data")
        if response_body.get("object") != "list" or not isinstance(data, list):
            return False
        if not data:
            return "/embeddings" in str(response_body.get("url", "")).lower()
        first_item = data[0]
        return isinstance(first_item, dict) and (
            first_item.get("object") == "embedding" or "embedding" in first_item
        )

    def _is_supported_openai_endpoint(self, url_route: str) -> bool:
        """Check if the OpenAI endpoint is supported by the passthrough logging handler."""
        from .llm_provider_handlers.openai_passthrough_logging_handler import (
            OpenAIPassthroughLoggingHandler,
        )

        return (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(
                url_route
            )
            or OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)
            or OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(url_route)
        )

    def _set_cost_per_request(
        self,
        logging_obj: LiteLLMLoggingObj,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        kwargs: dict,
    ):
        """
        Helper function to set the cost per request in the logging object

        Only set the cost per request if it's set in the passthrough logging payload.
        If it's not set, don't set it in the logging object.
        """
        #########################################################
        # Check if cost per request is set
        #########################################################
        if passthrough_logging_payload.get("cost_per_request") is not None:
            kwargs["response_cost"] = passthrough_logging_payload.get(
                "cost_per_request"
            )
            logging_obj.model_call_details[
                "response_cost"
            ] = passthrough_logging_payload.get("cost_per_request")

        return kwargs
