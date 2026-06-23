import ast
import asyncio
import hashlib
import importlib
import json
import os
import traceback
from base64 import b64encode
from datetime import datetime, timezone
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidStatus,
)

import litellm
from litellm._logging import (
    trigger_egress_guard_alert,
    verbose_proxy_logger,
)
from litellm._uuid import uuid
from litellm.constants import MAXIMUM_TRACEBACK_LINES_TO_LOG, PASS_THROUGH_HEADER_PREFIX
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_shape,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.passthrough import BasePassthroughUtils
from litellm.proxy._types import (
    ConfigFieldInfo,
    ConfigFieldUpdate,
    LiteLLMRoutes,
    PassThroughEndpointResponse,
    PassThroughGenericEndpoint,
    ProxyException,
    UserAPIKeyAuth,
    hash_token,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.aawm_route_logging import (
    build_aawm_route_access_log_line,  # noqa: F401 - re-exported for existing tests/callers
    emit_aawm_route_access_log,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
)
from litellm.proxy.litellm_pre_call_utils import clean_headers
from litellm.proxy.utils import get_server_root_path, normalize_route_for_root_path
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardLoggingUserAPIKeyMetadata

from .streaming_handler import PassThroughStreamingHandler
from .success_handler import PassThroughEndpointLogging

router = APIRouter()

pass_through_endpoint_logging = PassThroughEndpointLogging()

PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS: Tuple[int, ...] = (
    5,
    15,
    30,
    60,
    120,
)
PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES = frozenset(
    {500, 502, 503, 504, 529}
)
_ANTHROPIC_INVALID_AUTHENTICATION_MARKER = "invalid authentication credentials"
_ANTHROPIC_MODEL_NOT_FOUND_PREFIX = "model:"

# Global registry to track registered pass-through routes and prevent memory leaks
_registered_pass_through_routes: Dict[
    str, Dict[str, Union[str, List[str], Dict[str, Any]]]
] = {}

_AAWM_PASSTHROUGH_ERROR_LOG_MAX_FIELD_CHARS = 240
_AAWM_PASSTHROUGH_ERROR_LOG_SAFE_QUERY_KEYS = frozenset(
    {
        "alt",
        "api-version",
        "beta",
        "stream",
    }
)
_AAWM_PASSTHROUGH_ERROR_LOG_AUTH_HEADER_NAMES = (
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
)
_AAWM_PASSTHROUGH_ERROR_LOG_MODEL_METADATA_KEYS = (
    "codex_auto_agent_selected_model",
    "anthropic_auto_agent_selected_model",
    "anthropic_adapter_model",
    "xai_oauth_upstream_model",
    "xai_oauth_public_model",
    "grok_model_override",
    "model_group",
)
_AAWM_PASSTHROUGH_ERROR_LOG_MODEL_ALIAS_METADATA_KEYS = (
    "inbound_model_alias",
    "requested_model_alias",
    "model_alias_label",
    "anthropic_auto_agent_alias",
    "codex_auto_agent_alias",
)
_AAWM_PASSTHROUGH_ERROR_LOG_PROVIDER_METADATA_KEYS = (
    "provider",
    "custom_llm_provider",
    "litellm_provider",
)
_AAWM_PASSTHROUGH_ERROR_LOG_ROUTE_FAMILY_METADATA_KEYS = (
    "route_family",
    "passthrough_route_family",
    "openai_passthrough_route_family",
)
_AAWM_PASSTHROUGH_ERROR_LOG_TRACE_METADATA_KEYS = (
    "trace_id",
    "langfuse_trace_id",
    "existing_trace_id",
    "langfuse_existing_trace_id",
)
_AAWM_PASSTHROUGH_ERROR_LOG_REQUEST_SHAPE_KEYS = (
    "aawm_passthrough_inbound_content_type",
    "aawm_passthrough_json_egress_content_type_removed",
    "aawm_passthrough_json_egress_content_type_removed_value",
    "aawm_passthrough_body_container_type",
    "aawm_passthrough_body_top_level_keys",
    "aawm_passthrough_input_container_type",
    "aawm_passthrough_input_item_count",
    "aawm_passthrough_input_item_type_counts",
    "aawm_passthrough_tool_count",
    "aawm_passthrough_tool_type_counts",
)
_AAWM_PASSTHROUGH_ERROR_LOG_GROK_SIDE_CHANNEL_FIELDS = (
    "grok_side_channel",
    "grok_side_channel_endpoint_type",
    "grok_side_channel_endpoint_path_template",
    "grok_side_channel_request_content_type",
    "grok_side_channel_request_body_byte_length",
    "grok_side_channel_request_body_digest_source",
    "grok_side_channel_request_json_container_type",
    "grok_side_channel_request_array_length",
)
_GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_CODE = "the operation was cancelled"
_GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_ERROR = "timeout expired"
_GROK_SIGNALS_AUTH_CONTEXT_ERROR_MARKERS = (
    "invalid or expired credentials",
    "x_xai_token_auth=xai-grok-cli",
    "no auth context",
)
_CHATGPT_CODEX_BLOCK_PAGE_MARKERS = (
    "unable to load site",
    "cdn-cgi/challenge-platform",
)
_GOOGLE_CODE_ASSIST_HOST_SUFFIX = "cloudcode-pa.googleapis.com"
_GOOGLE_CODE_ASSIST_TOS_REASON = "TOS_VIOLATION"
_GOOGLE_CODE_ASSIST_PERMISSION_DENIED_STATUS = "PERMISSION_DENIED"


def get_response_body(response: httpx.Response) -> Optional[dict]:
    try:
        return response.json()
    except Exception:
        return None


def _normalize_openai_function_tool_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    normalized_parameters = dict(parameters)
    if normalized_parameters.get("type") is None:
        normalized_parameters["type"] = "object"
    _sanitize_openai_object_schema_properties(normalized_parameters)

    return normalized_parameters


def _sanitize_openai_object_schema_properties(schema_node: Any) -> int:
    fix_count = 0
    if isinstance(schema_node, dict):
        if schema_node.get("type") == "object" and not isinstance(
            schema_node.get("properties"), dict
        ):
            schema_node["properties"] = {}
            fix_count += 1
        for value in schema_node.values():
            fix_count += _sanitize_openai_object_schema_properties(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            fix_count += _sanitize_openai_object_schema_properties(item)
    return fix_count


def _normalize_openai_function_tool_schemas_in_body(body: Optional[dict]) -> int:
    if not isinstance(body, dict):
        return 0

    tools = body.get("tools")
    if not isinstance(tools, list):
        return 0

    normalized_count = 0
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        if "parameters" in tool:
            normalized_parameters = _normalize_openai_function_tool_parameters(
                tool.get("parameters")
            )
            if normalized_parameters != tool.get("parameters"):
                normalized_count += 1
            tool["parameters"] = normalized_parameters

        function_block = tool.get("function")
        if isinstance(function_block, dict):
            normalized_parameters = _normalize_openai_function_tool_parameters(
                function_block.get("parameters")
            )
            if normalized_parameters != function_block.get("parameters"):
                normalized_count += 1
            function_block["parameters"] = normalized_parameters

    return normalized_count


def _collect_invalid_openai_function_tool_schemas(
    body: Optional[dict],
) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []

    tools = body.get("tools")
    if not isinstance(tools, list):
        return []

    invalid_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        parameters = tool.get("parameters")
        if isinstance(parameters, dict):
            invalid_nodes = _collect_invalid_openai_object_schema_nodes(
                parameters,
                path=f"tools[{index}].parameters",
            )
            if invalid_nodes:
                invalid_tools.append(
                    {
                        "index": index,
                        "name": tool.get("name"),
                        "shape": "flat",
                        "invalid_nodes": invalid_nodes[:10],
                    }
                )

        function_block = tool.get("function")
        if isinstance(function_block, dict):
            parameters = function_block.get("parameters")
            if isinstance(parameters, dict):
                invalid_nodes = _collect_invalid_openai_object_schema_nodes(
                    parameters,
                    path=f"tools[{index}].function.parameters",
                )
                if invalid_nodes:
                    invalid_tools.append(
                        {
                            "index": index,
                            "name": function_block.get("name"),
                            "shape": "nested",
                            "invalid_nodes": invalid_nodes[:10],
                        }
                    )

    return invalid_tools


def _collect_invalid_openai_object_schema_nodes(
    schema_node: Any,
    *,
    path: str,
) -> list[dict[str, Any]]:
    invalid_nodes: list[dict[str, Any]] = []
    if isinstance(schema_node, dict):
        if schema_node.get("type") == "object" and not isinstance(
            schema_node.get("properties"), dict
        ):
            invalid_nodes.append({"path": path, "schema": schema_node})
        for key, value in schema_node.items():
            invalid_nodes.extend(
                _collect_invalid_openai_object_schema_nodes(
                    value,
                    path=f"{path}.{key}",
                )
            )
    elif isinstance(schema_node, list):
        for index, item in enumerate(schema_node):
            invalid_nodes.extend(
                _collect_invalid_openai_object_schema_nodes(
                    item,
                    path=f"{path}[{index}]",
                )
            )
    return invalid_nodes


def _coerce_upstream_error_payload(detail: Any) -> Optional[dict[str, Any]]:
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="replace")
    elif isinstance(detail, str):
        detail_text = detail
        stripped_detail = detail_text.strip()
        if stripped_detail.startswith(("b'", 'b"')):
            try:
                literal_detail = ast.literal_eval(stripped_detail)
            except Exception:
                literal_detail = None
            if isinstance(literal_detail, bytes):
                detail_text = literal_detail.decode("utf-8", errors="replace")
            elif isinstance(literal_detail, str):
                detail_text = literal_detail
    elif isinstance(detail, dict):
        return detail
    else:
        return None

    try:
        parsed = json.loads(detail_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(float(value))
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _chatgpt_codex_usage_limit_retry_after_seconds(
    error_block: dict[str, Any],
) -> Optional[int]:
    resets_in_seconds = _coerce_positive_int(error_block.get("resets_in_seconds"))
    if resets_in_seconds is not None:
        return resets_in_seconds

    resets_at = _coerce_positive_int(error_block.get("resets_at"))
    if resets_at is None:
        return None
    return max(1, int(resets_at - time.time()))


def _build_chatgpt_codex_usage_limit_detail(
    *,
    error: httpx.HTTPStatusError,
    upstream_payload: dict[str, Any],
) -> Optional[tuple[dict[str, Any], Optional[int]]]:
    if error.response.status_code != 429:
        return None
    request_url = error.request.url
    if (
        str(request_url.host or "").lower() != "chatgpt.com"
        or "/backend-api/codex/" not in str(request_url.path or "").lower()
    ):
        return None
    upstream_error = upstream_payload.get("error")
    if not isinstance(upstream_error, dict):
        return None
    if upstream_error.get("type") != "usage_limit_reached":
        return None

    retry_after_seconds = _chatgpt_codex_usage_limit_retry_after_seconds(
        upstream_error
    )
    structured_detail = {
        "error": {
            "message": (
                "ChatGPT Codex usage limit has been reached for the upstream "
                "account. Treat this as quota exhaustion, not transient high "
                "demand."
            ),
            "type": "rate_limit_error",
            "code": "usage_limit_reached",
            "upstream_type": upstream_error.get("type"),
            "upstream_message": upstream_error.get("message"),
        },
        "upstream_status_code": error.response.status_code,
        "upstream_url": str(error.request.url),
        "quota": {
            "plan_type": upstream_error.get("plan_type"),
            "resets_at": upstream_error.get("resets_at"),
            "resets_in_seconds": upstream_error.get("resets_in_seconds"),
            "eligible_promo": upstream_error.get("eligible_promo"),
        },
        "retry_after_seconds": retry_after_seconds,
        "failover_disposition": "usage_limit_reached",
    }
    return structured_detail, retry_after_seconds


def _build_http_exception_from_upstream_status_error(
    error: httpx.HTTPStatusError, detail: Any
) -> HTTPException:
    upstream_headers = {
        str(header_name): str(header_value)
        for header_name, header_value in error.response.headers.items()
    }
    upstream_payload = _coerce_upstream_error_payload(detail)
    if upstream_payload is not None:
        usage_limit_detail = _build_chatgpt_codex_usage_limit_detail(
            error=error,
            upstream_payload=upstream_payload,
        )
        if usage_limit_detail is not None:
            structured_detail, retry_after_seconds = usage_limit_detail
            if retry_after_seconds is not None:
                upstream_headers["Retry-After"] = str(retry_after_seconds)
            return HTTPException(
                status_code=error.response.status_code,
                detail=structured_detail,
                headers=upstream_headers or None,
            )

    return HTTPException(
        status_code=error.response.status_code,
        detail=detail,
        headers=upstream_headers or None,
    )


def _extract_exception_status_code(exc: Exception) -> Optional[int]:
    if isinstance(exc, (httpx.TimeoutException, httpx.ReadTimeout)):
        return status.HTTP_504_GATEWAY_TIMEOUT
    if isinstance(exc, httpx.ConnectError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    for attr_name in ("status_code", "code"):
        value = getattr(exc, attr_name, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return None




@dataclass(frozen=True)
class PassthroughRequestObservabilityEnvelope:
    """Reusable passthrough request-body observability envelope."""

    parsed_body: Optional[dict]
    serialized_body: str
    logging_input: List[Dict[str, str]]

    @property
    def complete_input_dict(self) -> Optional[dict]:
        return self.parsed_body


def _build_passthrough_request_observability_envelope(
    parsed_body: Optional[dict],
) -> PassthroughRequestObservabilityEnvelope:
    serialized_body = safe_dumps(parsed_body)
    return PassthroughRequestObservabilityEnvelope(
        parsed_body=parsed_body,
        serialized_body=serialized_body,
        logging_input=[{"role": "user", "content": serialized_body}],
    )


def _build_passthrough_logging_input(
    parsed_body: Optional[dict],
) -> List[Dict[str, str]]:
    """Build the serialized passthrough logging envelope once for reuse."""
    return _build_passthrough_request_observability_envelope(parsed_body).logging_input


def _shallow_copy_request_dict(body: dict) -> dict:
    return dict(body)


def _copy_custom_body_for_passthrough(custom_body: dict) -> dict:
    """Shallow-copy route-owned custom bodies instead of deep-copying large payloads."""
    return _shallow_copy_request_dict(custom_body)


def _detach_passthrough_body_for_kwargs(
    parsed_body: Optional[dict],
) -> tuple[dict, Optional[dict]]:
    """
    Copy-on-write detach for litellm param stripping.

    When litellm-owned keys are present, return a shallow mutable copy for kwargs
    construction while preserving the original body reference for observability
    envelopes and logging.
    """
    if not isinstance(parsed_body, dict):
        return {}, None

    from litellm.types.utils import all_litellm_params

    litellm_param_keys = [key for key in all_litellm_params if key in parsed_body]
    if not litellm_param_keys:
        return parsed_body, parsed_body

    working_body = _shallow_copy_request_dict(parsed_body)
    return working_body, parsed_body


def _provider_bound_body_from_kwargs(kwargs: Optional[dict]) -> Optional[dict]:
    if not isinstance(kwargs, dict):
        return None

    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return None

    proxy_server_request = litellm_params.get("proxy_server_request")
    if not isinstance(proxy_server_request, dict):
        return None

    provider_bound_body = proxy_server_request.get("body")
    if isinstance(provider_bound_body, dict):
        return provider_bound_body

    return None


def _extract_passthrough_exception_detail(exc: Exception) -> Optional[str]:
    for attr_name in ("detail", "message"):
        value = getattr(exc, attr_name, None)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            try:
                return safe_dumps(value)
            except Exception:
                continue
        value_text = str(value).strip()
        if value_text:
            return value_text
    return None


def _extract_passthrough_grok_billing_timeout_cancel_hint(
    detail: Optional[Any],
) -> Optional[str]:
    if detail is None:
        return None

    if isinstance(detail, dict):
        code = detail.get("code")
        error = detail.get("error")
        if isinstance(code, str) and code.strip():
            return code.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
        return None

    detail_text = str(detail).strip()
    if not detail_text:
        return None

    try:
        parsed_detail = json.loads(detail_text)
    except json.JSONDecodeError:
        return detail_text

    if isinstance(parsed_detail, dict):
        return _extract_passthrough_grok_billing_timeout_cancel_hint(parsed_detail)
    return detail_text


def _is_known_grok_billing_passthrough_timeout_cancel_response(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_400_BAD_REQUEST:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    request_path = HttpPassThroughEndpointHelpers._get_passthrough_request_url_path(
        request
    )
    upstream_path = urlparse(str(url or "")).path
    if not (
        request_path.rstrip("/").endswith("/grok/v1/billing")
        or upstream_path.rstrip("/").endswith("/v1/billing")
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    error_hint = _extract_passthrough_grok_billing_timeout_cancel_hint(detail)
    if not error_hint:
        return False

    normalized_hint = error_hint.strip().lower()
    return (
        _GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_CODE in normalized_hint
        or _GROK_BILLING_PASSTHROUGH_TIMEOUT_CANCEL_ERROR in normalized_hint
    )


def _is_grok_signals_path(path: str) -> bool:
    normalized_path = path.rstrip("/")
    return (
        normalized_path.startswith("/grok/v1/sessions/")
        and normalized_path.endswith("/signals")
    ) or (
        normalized_path.startswith("/v1/sessions/")
        and normalized_path.endswith("/signals")
    )


def _is_known_grok_signals_auth_context_response(
    *,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_401_UNAUTHORIZED:
        return False

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    request_path = HttpPassThroughEndpointHelpers._get_passthrough_request_url_path(
        request
    )
    upstream_path = urlparse(str(url or "")).path
    if not (_is_grok_signals_path(request_path) or _is_grok_signals_path(upstream_path)):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).strip().lower()
    return all(
        marker in normalized_detail
        for marker in _GROK_SIGNALS_AUTH_CONTEXT_ERROR_MARKERS
    )


def _is_known_chatgpt_codex_block_page_response(
    *,
    url: Optional[httpx.URL],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_403_FORBIDDEN:
        return False

    parsed_url = urlparse(str(url or ""))
    if (
        str(parsed_url.hostname or "").lower() != "chatgpt.com"
        or "/backend-api/codex/" not in str(parsed_url.path or "").lower()
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).lower()
    return any(
        marker in normalized_detail for marker in _CHATGPT_CODEX_BLOCK_PAGE_MARKERS
    )


def _is_google_code_assist_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return (
        provider in {"google_code_assist", "antigravity"}
        or hostname.endswith(_GOOGLE_CODE_ASSIST_HOST_SUFFIX)
    )


def _is_google_code_assist_tos_violation_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    code = error.get("code")
    status_text = str(error.get("status") or "").strip().upper()
    if code != status.HTTP_403_FORBIDDEN and status_text != (
        _GOOGLE_CODE_ASSIST_PERMISSION_DENIED_STATUS
    ):
        return False

    details = error.get("details")
    if not isinstance(details, list):
        return False
    for detail in details:
        if not isinstance(detail, dict):
            continue
        reason = str(detail.get("reason") or "").strip().upper()
        domain = str(detail.get("domain") or "").strip().lower()
        if reason == _GOOGLE_CODE_ASSIST_TOS_REASON and domain.endswith(
            _GOOGLE_CODE_ASSIST_HOST_SUFFIX
        ):
            return True
    return False


def _is_known_google_code_assist_tos_violation_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_403_FORBIDDEN:
        return False

    if not _is_google_code_assist_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if detail is None:
        return False

    payload = _coerce_upstream_error_payload(detail)
    return _is_google_code_assist_tos_violation_payload(payload)


def _is_anthropic_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return provider == "anthropic" or hostname == "api.anthropic.com"


def _get_known_anthropic_passthrough_failure_kind(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[str]:
    if status_code not in {status.HTTP_401_UNAUTHORIZED, status.HTTP_404_NOT_FOUND}:
        return None

    if not _is_anthropic_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return None

    detail = _extract_passthrough_exception_detail(exc)
    if detail is None:
        return None

    payload = _coerce_upstream_error_payload(detail)
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None

    error_type = str(error.get("type") or "").strip().lower()
    message = str(error.get("message") or "").strip().lower()
    if (
        status_code == status.HTTP_401_UNAUTHORIZED
        and error_type == "authentication_error"
        and _ANTHROPIC_INVALID_AUTHENTICATION_MARKER in message
    ):
        return "anthropic_client_authentication_error"
    if (
        status_code == status.HTTP_404_NOT_FOUND
        and error_type == "not_found_error"
        and message.startswith(_ANTHROPIC_MODEL_NOT_FOUND_PREFIX)
    ):
        return "anthropic_model_not_found"
    return None


def _get_case_insensitive_mapping_value(
    values: Optional[dict],
    key_name: str,
) -> Optional[str]:
    if not isinstance(values, dict):
        return None
    wanted = key_name.lower()
    for key, value in values.items():
        if str(key).lower() != wanted or value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            return value_text
    return None


def _is_xai_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return (
        provider == "xai"
        or hostname in {
            "api.x.ai",
            "cli-chat-proxy.grok.com",
        }
        or hostname.endswith(".x.ai")
        or hostname.endswith(".grok.com")
    )


def _enrich_passthrough_failure_request_payload(
    *,
    request_payload: dict,
    request: Request,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> None:
    if custom_llm_provider:
        request_payload["custom_llm_provider"] = custom_llm_provider
        litellm_params = request_payload.setdefault("litellm_params", {})
        if isinstance(litellm_params, dict):
            metadata = litellm_params.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.setdefault("custom_llm_provider", custom_llm_provider)
                if url is not None:
                    metadata.setdefault("api_base", str(url))

    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return

    litellm_params = request_payload.setdefault("litellm_params", {})
    if not isinstance(litellm_params, dict):
        return
    metadata = litellm_params.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return

    metadata.setdefault("custom_llm_provider", "xai")
    metadata.setdefault("passthrough_route_family", "grok_cli_chat_proxy")
    metadata.setdefault("xai_cli_chat_proxy", True)
    if url is not None:
        metadata.setdefault("api_base", str(url))

    headers = _safe_get_request_headers(request)
    model_override = _get_case_insensitive_mapping_value(
        headers,
        "x-grok-model-override",
    )
    if model_override:
        metadata.setdefault("grok_model_override", model_override)
        metadata.setdefault("model_group", model_override)
        if not str(request_payload.get("model") or "").strip():
            request_payload["model"] = model_override


async def _direct_capture_xai_passthrough_failure(
    *,
    request_payload: dict,
    original_exception: Exception,
    user_api_key_dict: Any,
    traceback_str: Optional[str],
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> None:
    if not _is_xai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return

    try:
        aawm_agent_identity_instance = None
        for module_name in (
            "litellm.integrations.aawm_agent_identity",
            "aawm_litellm_callbacks.agent_identity",
        ):
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            aawm_agent_identity_instance = getattr(
                module,
                "aawm_agent_identity_instance",
                None,
            )
            if aawm_agent_identity_instance is not None:
                break
        if aawm_agent_identity_instance is None:
            raise ModuleNotFoundError(
                "No AAWM agent identity callback module is importable"
            )

        await aawm_agent_identity_instance.async_post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=original_exception,
            request_data=request_payload,
            traceback_str=traceback_str,
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "Failed to directly capture xAI passthrough failure for provider health: %s",
            exc,
        )


def _ensure_passthrough_metadata(kwargs: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(kwargs, dict):
        return {}

    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        litellm_params = {}
        kwargs["litellm_params"] = litellm_params

    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        litellm_params["metadata"] = metadata

    return metadata


def _passthrough_header_value(headers: dict, header_name: str) -> Optional[str]:
    normalized_name = header_name.lower()
    for key, value in headers.items():
        if str(key).lower() == normalized_name and value is not None:
            return str(value)
    return None


def _safe_passthrough_query_keys(
    *,
    url: httpx.URL,
    requested_query_params: Optional[dict],
) -> list[str]:
    query_keys = {
        key
        for key, _value in parse_qsl(
            str(urlparse(str(url)).query),
            keep_blank_values=True,
        )
        if key
    }
    if isinstance(requested_query_params, dict):
        query_keys.update(str(key) for key in requested_query_params if str(key))
    return sorted(query_keys)


def _is_grok_billing_passthrough_request(
    *,
    request: Request,
    url: httpx.URL,
    metadata: dict[str, Any],
    custom_llm_provider: Optional[str],
) -> bool:
    route_text = " ".join(
        str(value)
        for value in (
            str(url),
            getattr(request, "url", ""),
            metadata.get("user_api_key_request_route"),
            metadata.get("passthrough_route_family"),
            custom_llm_provider,
        )
        if value is not None
    ).lower()
    return "/billing" in route_text and (
        "grok" in route_text or "xai" in route_text or "x.ai" in route_text
    )


def _record_grok_billing_passthrough_request_contract(
    *,
    request: Request,
    url: httpx.URL,
    headers: dict,
    requested_query_params: Optional[dict],
    metadata: dict[str, Any],
    custom_llm_provider: Optional[str],
) -> None:
    if not _is_grok_billing_passthrough_request(
        request=request,
        url=url,
        metadata=metadata,
        custom_llm_provider=custom_llm_provider,
    ):
        return

    header_names = sorted({str(key).lower() for key in headers if str(key)})
    query_keys = _safe_passthrough_query_keys(
        url=url,
        requested_query_params=requested_query_params,
    )
    user_agent = _passthrough_header_value(headers, "user-agent")
    fingerprint_payload = {
        "header_names": header_names,
        "http_client": "httpx",
        "method": getattr(request, "method", None),
        "query_keys": query_keys,
        "target_host": url.host,
        "target_path": url.path or "/",
        "user_agent": user_agent,
        "x_xai_token_auth_configured": "x-xai-token-auth" in header_names,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    metadata.update(
        {
            "grok_billing_passthrough_http_client": "httpx",
            "grok_billing_passthrough_request_method": getattr(
                request,
                "method",
                None,
            ),
            "grok_billing_passthrough_target_host": url.host,
            "grok_billing_passthrough_target_path": url.path or "/",
            "grok_billing_passthrough_query_keys": query_keys,
            "grok_billing_passthrough_query_present": bool(query_keys),
            "grok_billing_passthrough_header_names": header_names,
            "grok_billing_passthrough_user_agent": user_agent,
            "grok_billing_passthrough_x_xai_token_auth_configured": (
                "x-xai-token-auth" in header_names
            ),
            "grok_billing_passthrough_request_contract_fingerprint": fingerprint,
        }
    )


async def _passthrough_hidden_retry_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _get_passthrough_hidden_retry_wait_seconds(attempt_index: int) -> float:
    if attempt_index < 0:
        return 0.0
    if attempt_index < len(PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS):
        return float(PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS[attempt_index])
    return float(PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS[-1])


def _classify_passthrough_hidden_retry_failure(exc: Exception) -> Tuple[
    Optional[int],
    str,
    Optional[str],
]:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code, f"http_status_{status_code}", None

    if isinstance(exc, httpx.TimeoutException):
        return None, "upstream_connectivity_failure", "upstream_connectivity_failure"

    if isinstance(exc, httpx.ConnectError):
        message = str(exc).lower()
        if (
            "name resolution" in message
            or "temporary failure in name resolution" in message
        ):
            return None, "transport_dns_failure", "transport_dns_failure"
        return None, "upstream_connectivity_failure", "upstream_connectivity_failure"

    if isinstance(exc, httpx.ReadError):
        return None, "upstream_connectivity_failure", "upstream_connectivity_failure"

    status_code = _extract_exception_status_code(exc)
    if status_code is not None:
        return status_code, f"http_status_{status_code}", None

    return None, exc.__class__.__name__, None


def _is_passthrough_pre_first_byte_hidden_retryable(
    exc: Exception,
    *,
    status_code: Optional[int],
    failure_class: str,
) -> bool:
    if status_code == 429:
        return False
    if status_code in PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if failure_class in {"transport_dns_failure", "upstream_connectivity_failure"}:
        return True
    return False


def _should_log_passthrough_terminal_failure_without_traceback(
    *,
    exc: Exception,
    kwargs: Optional[dict],
    status_code: Optional[int],
) -> bool:
    try:
        metadata = _ensure_passthrough_metadata(kwargs)
        final_outcome = metadata.get("aawm_passthrough_hidden_retry_final_outcome")
        if final_outcome not in {"failed_after_retry", "failed_without_retry"}:
            return False

        if (
            status_code in PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES
            or status_code == status.HTTP_429_TOO_MANY_REQUESTS
            or status_code == status.HTTP_507_INSUFFICIENT_STORAGE
        ):
            return True

        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
            return True

        failure_classification = metadata.get(
            "aawm_passthrough_hidden_retry_failure_classification"
        )
        return failure_classification in {
            "upstream_connectivity_failure",
            "transport_dns_failure",
        }
    except Exception as classification_exc:
        verbose_proxy_logger.warning(
            "Failed to classify pass-through terminal failure logging: %s",
            classification_exc,
        )
        return False


def _is_passthrough_rate_limit_retryable(
    *,
    status_code: Optional[int],
    metadata: dict,
) -> bool:
    if status_code != 429:
        return False
    return metadata.get("aawm_passthrough_rate_limit_retry_enabled") is True


def _is_passthrough_retryable(
    *,
    exc: Exception,
    status_code: Optional[int],
    metadata: dict,
) -> bool:
    if _is_passthrough_rate_limit_retryable(status_code=status_code, metadata=metadata):
        return True

    if status_code in PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES:
        # If the exception is a standard HTTPException (e.g. raised by LiteLLM or FastAPI),
        # we preserve the original behavior and log with traceback.
        if isinstance(exc, HTTPException):
            return False
        return True

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True

    failure_classification = metadata.get(
        "aawm_passthrough_hidden_retry_failure_classification"
    )
    return failure_classification in {
        "upstream_connectivity_failure",
        "transport_dns_failure",
    }


def _is_passthrough_expected_provider_rate_limit(
    *,
    status_code: Optional[int],
) -> bool:
    return status_code == status.HTTP_429_TOO_MANY_REQUESTS


def _get_passthrough_terminal_failure_kind(
    *,
    hidden_retry_failure_classification: Optional[Any],
) -> str:
    if hidden_retry_failure_classification in {
        "transport_dns_failure",
        "upstream_connectivity_failure",
    }:
        return "transient_provider_connectivity"
    return "expected_upstream_capacity_or_internal"


def _get_passthrough_grok_billing_timeout_failure_kind() -> str:
    return "degraded_grok_billing_timeout"


def _get_passthrough_grok_signals_auth_context_failure_kind() -> str:
    return "degraded_grok_signals_auth_context"


def _record_passthrough_hidden_retry_metadata(
    kwargs: Optional[dict],
    *,
    attempt_number: int,
    max_attempts: int,
    status_code: Optional[int],
    failure_class: str,
    wait_seconds: float,
    final_outcome: Optional[str] = None,
    failure_classification: Optional[str] = None,
) -> None:
    if not isinstance(kwargs, dict):
        return
    metadata = _ensure_passthrough_metadata(kwargs)

    attempts = metadata.get("aawm_passthrough_hidden_retry_attempts")
    if not isinstance(attempts, list):
        attempts = []
        metadata["aawm_passthrough_hidden_retry_attempts"] = attempts

    attempt_record: Dict[str, Any] = {
        "attempt": attempt_number,
        "max_attempts": max_attempts,
        "failure_class": failure_class,
        "wait_seconds": round(wait_seconds, 3),
    }
    if status_code is not None:
        attempt_record["status_code"] = status_code
    if failure_classification is not None:
        attempt_record["failure_classification"] = failure_classification
    attempts.append(attempt_record)

    metadata["aawm_passthrough_hidden_retry_count"] = len(attempts)
    if final_outcome is not None:
        metadata["aawm_passthrough_hidden_retry_final_outcome"] = final_outcome
    if failure_classification is not None:
        metadata["aawm_passthrough_hidden_retry_failure_classification"] = (
            failure_classification
        )


async def _execute_passthrough_pre_first_byte_with_hidden_retries(
    *,
    kwargs: Optional[dict],
    operation_name: str,
    operation: Any,
    caller_managed_hidden_retry: bool,
) -> Any:
    if caller_managed_hidden_retry:
        return await operation()

    max_attempts = len(PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS) + 1
    attempt_number = 0
    while True:
        attempt_number += 1
        try:
            result = await operation()
            if attempt_number > 1:
                _record_passthrough_hidden_retry_metadata(
                    kwargs,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    status_code=None,
                    failure_class="success",
                    wait_seconds=0.0,
                    final_outcome="success_after_retry",
                )
            return result
        except Exception as exc:
            status_code, failure_class, failure_classification = (
                _classify_passthrough_hidden_retry_failure(exc)
            )
            should_retry = _is_passthrough_pre_first_byte_hidden_retryable(
                exc,
                status_code=status_code,
                failure_class=failure_class,
            )
            if not should_retry or attempt_number >= max_attempts:
                _record_passthrough_hidden_retry_metadata(
                    kwargs,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    status_code=status_code,
                    failure_class=failure_class,
                    wait_seconds=0.0,
                    final_outcome=(
                        "failed_after_retry"
                        if attempt_number > 1
                        else "failed_without_retry"
                    ),
                    failure_classification=failure_classification,
                )
                raise

            wait_seconds = _get_passthrough_hidden_retry_wait_seconds(
                attempt_number - 1
            )
            _record_passthrough_hidden_retry_metadata(
                kwargs,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                status_code=status_code,
                failure_class=failure_class,
                wait_seconds=wait_seconds,
                failure_classification=failure_classification,
            )
            verbose_proxy_logger.info(
                "Pass-through %s hidden retry attempt %s/%s after %s; sleeping %.1fs",
                operation_name,
                attempt_number,
                max_attempts,
                failure_class,
                wait_seconds,
            )
            await _passthrough_hidden_retry_sleep(wait_seconds)


def _clean_passthrough_error_context_value(value: Any) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if not isinstance(value, (str, int, float)):
        return None

    cleaned = "".join(
        char if char.isprintable() and char not in "\r\n\t" else " "
        for char in str(value).strip()
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None

    lower_cleaned = cleaned.lower()
    if lower_cleaned.startswith(("bearer ", "sk-", "pk-", "xai-", "ya29.")):
        return None

    if len(cleaned) > _AAWM_PASSTHROUGH_ERROR_LOG_MAX_FIELD_CHARS:
        cleaned = (
            cleaned[: _AAWM_PASSTHROUGH_ERROR_LOG_MAX_FIELD_CHARS - 3] + "..."
        )
    return cleaned


def _first_passthrough_error_context_value(
    metadata: Dict[str, Any],
    keys: Tuple[str, ...],
) -> Optional[str]:
    for key in keys:
        value = _clean_passthrough_error_context_value(metadata.get(key))
        if value:
            return value
    return None


def _passthrough_error_context_metadata_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        cleaned_list = [
            cleaned
            for item in value[:20]
            if (cleaned := _clean_passthrough_error_context_value(item)) is not None
        ]
        return cleaned_list
    if isinstance(value, dict):
        cleaned_dict: Dict[str, Any] = {}
        for key, item in list(value.items())[:20]:
            cleaned_key = _clean_passthrough_error_context_value(key)
            cleaned_value = _passthrough_error_context_metadata_value(item)
            if cleaned_key is not None and cleaned_value is not None:
                cleaned_dict[cleaned_key] = cleaned_value
        return cleaned_dict
    return _clean_passthrough_error_context_value(value)


def _build_passthrough_error_log_request_shape_context(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for field_name in _AAWM_PASSTHROUGH_ERROR_LOG_REQUEST_SHAPE_KEYS:
        value = _passthrough_error_context_metadata_value(metadata.get(field_name))
        if value is not None:
            context[field_name] = value
    return context


def _build_passthrough_error_log_grok_side_channel_context(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for field_name in _AAWM_PASSTHROUGH_ERROR_LOG_GROK_SIDE_CHANNEL_FIELDS:
        value = _passthrough_error_context_metadata_value(metadata.get(field_name))
        if value is not None:
            context[field_name] = value
    return context


def _normalize_passthrough_header_mapping(headers: Optional[dict]) -> Dict[str, Any]:
    return {str(key).lower(): value for key, value in dict(headers or {}).items()}


def _build_passthrough_error_log_auth_context(
    *,
    request: Request,
    final_headers: Optional[dict],
    custom_headers: Optional[dict],
) -> Dict[str, Any]:
    request_headers = _normalize_passthrough_header_mapping(
        _safe_get_request_headers(request)
    )
    custom_header_map = _normalize_passthrough_header_mapping(custom_headers)
    final_header_map = _normalize_passthrough_header_mapping(final_headers)
    auth_header_names: list[str] = []
    auth_header_sources: list[str] = []

    for header_name in _AAWM_PASSTHROUGH_ERROR_LOG_AUTH_HEADER_NAMES:
        if header_name not in final_header_map:
            continue

        auth_header_names.append(header_name)
        pass_through_header_name = f"{PASS_THROUGH_HEADER_PREFIX}{header_name}"
        if pass_through_header_name in request_headers:
            source = f"incoming_pass_through_header:{header_name}"
        elif header_name in custom_header_map:
            source = f"route_custom_header:{header_name}"
        elif header_name in request_headers:
            source = f"incoming_request:{header_name}"
        else:
            source = f"derived_final_header:{header_name}"
        auth_header_sources.append(source)

    credential_sources = {
        source.split(":", 1)[0] for source in auth_header_sources if source
    }
    if not credential_sources:
        auth_credential_source = "none"
    elif len(credential_sources) == 1:
        auth_credential_source = next(iter(credential_sources))
    else:
        auth_credential_source = "mixed"

    return {
        "auth_header_names": auth_header_names,
        "auth_header_sources": auth_header_sources,
        "auth_credential_source": auth_credential_source,
    }


def _passthrough_container_type(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _count_passthrough_item_types(items: Any) -> Dict[str, int]:
    if not isinstance(items, list):
        return {}

    type_counts: Dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            item_type = _clean_passthrough_error_context_value(item.get("type"))
            if not item_type:
                item_type = "object_without_type"
        else:
            item_type = _passthrough_container_type(item)
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    return dict(sorted(type_counts.items()))


def _merge_passthrough_request_shape_metadata(
    metadata: Dict[str, Any],
    *,
    request: Request,
    parsed_body: Optional[dict],
    provider_bound_body: Optional[dict],
    json_egress_content_type_removed: Optional[str] = None,
) -> None:
    inbound_content_type = _clean_passthrough_error_context_value(
        request.headers.get("content-type")
    )
    if inbound_content_type:
        metadata.setdefault(
            "aawm_passthrough_inbound_content_type",
            inbound_content_type,
        )

    if json_egress_content_type_removed:
        metadata["aawm_passthrough_json_egress_content_type_removed"] = True
        metadata["aawm_passthrough_json_egress_content_type_removed_value"] = (
            _clean_passthrough_error_context_value(json_egress_content_type_removed)
        )
    else:
        metadata.pop("aawm_passthrough_json_egress_content_type_removed", None)
        metadata.pop("aawm_passthrough_json_egress_content_type_removed_value", None)

    body = provider_bound_body if isinstance(provider_bound_body, dict) else parsed_body
    metadata["aawm_passthrough_body_container_type"] = _passthrough_container_type(body)
    if not isinstance(body, dict):
        return

    top_level_keys = [
        key
        for key in (
            _clean_passthrough_error_context_value(key)
            for key in sorted(body.keys(), key=str)
        )
        if key is not None and key != "litellm_logging_obj"
    ]
    metadata["aawm_passthrough_body_top_level_keys"] = top_level_keys[:40]

    input_value = body.get("input")
    metadata["aawm_passthrough_input_container_type"] = _passthrough_container_type(
        input_value
    )
    if isinstance(input_value, list):
        metadata["aawm_passthrough_input_item_count"] = len(input_value)
        metadata["aawm_passthrough_input_item_type_counts"] = (
            _count_passthrough_item_types(input_value)
        )

    tools = body.get("tools")
    if isinstance(tools, list):
        metadata["aawm_passthrough_tool_count"] = len(tools)
        metadata["aawm_passthrough_tool_type_counts"] = (
            _count_passthrough_item_types(tools)
        )


def _headers_for_json_passthrough_egress(
    headers: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[str]]:
    json_headers = dict(headers)
    removed_content_type: Optional[str] = None
    has_json_content_type = False
    for header_name in list(json_headers.keys()):
        if str(header_name).lower() == "content-type":
            content_type = _clean_passthrough_error_context_value(
                json_headers[header_name]
            )
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type == "application/json" or media_type.endswith("+json"):
                has_json_content_type = True
                continue
            removed_content_type = content_type
            json_headers.pop(header_name)
    if not has_json_content_type:
        json_headers["content-type"] = "application/json"
    return json_headers, removed_content_type


def _is_json_passthrough_egress(
    *,
    request: Request,
    raw_body: Optional[bytes],
    provider_bound_body: Optional[dict],
) -> bool:
    if request.method == "GET" or raw_body is not None:
        return False
    return not (
        HttpPassThroughEndpointHelpers.is_multipart(request) is True
        and not provider_bound_body
    )


def _build_passthrough_error_log_endpoint(
    request: Request,
    *,
    grok_side_channel_endpoint_path_template: Optional[str] = None,
) -> Optional[str]:
    request_url = getattr(request, "url", None)
    parsed_url = urlparse(str(request_url or ""))
    path = parsed_url.path

    if not path:
        direct_path = _clean_passthrough_error_context_value(
            getattr(request_url, "path", None)
        )
        path = direct_path or "/"

    if grok_side_channel_endpoint_path_template:
        if path.startswith("/grok/v1/"):
            path = f"/grok/v1{grok_side_channel_endpoint_path_template}"
        elif path.startswith("/v1/"):
            path = f"/v1{grok_side_channel_endpoint_path_template}"
        else:
            path = grok_side_channel_endpoint_path_template

    safe_query_pairs: list[tuple[str, str]] = []
    for key, value in dict(getattr(request, "query_params", {}) or {}).items():
        normalized_key = str(key).lower()
        if normalized_key not in _AAWM_PASSTHROUGH_ERROR_LOG_SAFE_QUERY_KEYS:
            continue
        safe_key = _clean_passthrough_error_context_value(key)
        safe_value = _clean_passthrough_error_context_value(value)
        if safe_key and safe_value is not None:
            safe_query_pairs.append((safe_key, safe_value))

    if not safe_query_pairs:
        return path
    return f"{path}?{urlencode(safe_query_pairs)}"


def _build_passthrough_error_log_upstream_url(
    url: Optional[httpx.URL],
    *,
    grok_side_channel_endpoint_path_template: Optional[str] = None,
) -> Optional[str]:
    if url is None:
        return None

    parsed_url = urlparse(str(url))
    if parsed_url.scheme and parsed_url.hostname:
        host = parsed_url.hostname
        if parsed_url.port is not None:
            host = f"{host}:{parsed_url.port}"
        path = parsed_url.path or "/"
        if grok_side_channel_endpoint_path_template:
            if path.startswith("/v1/"):
                path = f"/v1{grok_side_channel_endpoint_path_template}"
            else:
                path = grok_side_channel_endpoint_path_template
        return f"{parsed_url.scheme}://{host}{path}"
    return _clean_passthrough_error_context_value(str(url))


def _build_passthrough_error_log_context(
    *,
    request: Request,
    url: Optional[httpx.URL],
    parsed_body: Optional[dict],
    kwargs: Optional[dict],
    passthrough_logging_metadata: Optional[dict],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    litellm_call_id: Optional[str],
    final_headers: Optional[dict] = None,
    custom_headers: Optional[dict] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(parsed_body, dict):
        for metadata_key in ("litellm_metadata", "metadata"):
            metadata_value = parsed_body.get(metadata_key)
            if isinstance(metadata_value, dict):
                metadata.update(metadata_value)

    if isinstance(kwargs, dict):
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            kwargs_metadata = litellm_params.get("metadata")
            if isinstance(kwargs_metadata, dict):
                metadata.update(kwargs_metadata)

    model = None
    if isinstance(parsed_body, dict):
        model = _clean_passthrough_error_context_value(parsed_body.get("model"))
    model = model or _first_passthrough_error_context_value(
        metadata,
        _AAWM_PASSTHROUGH_ERROR_LOG_MODEL_METADATA_KEYS,
    )

    provider = _clean_passthrough_error_context_value(
        custom_llm_provider
    ) or _first_passthrough_error_context_value(
        metadata,
        _AAWM_PASSTHROUGH_ERROR_LOG_PROVIDER_METADATA_KEYS,
    )

    grok_side_channel_context: Dict[str, Any] = {}
    if isinstance(passthrough_logging_metadata, dict):
        grok_side_channel_context = _build_passthrough_error_log_grok_side_channel_context(
            passthrough_logging_metadata
        )
    grok_side_channel_endpoint_path_template = _clean_passthrough_error_context_value(
        grok_side_channel_context.get("grok_side_channel_endpoint_path_template")
    )

    context = {
        "source": "pass_through_endpoint",
        "container": _clean_passthrough_error_context_value(os.getenv("HOSTNAME")),
        "endpoint": _build_passthrough_error_log_endpoint(
            request,
            grok_side_channel_endpoint_path_template=grok_side_channel_endpoint_path_template,
        ),
        "upstream_url": _build_passthrough_error_log_upstream_url(
            url,
            grok_side_channel_endpoint_path_template=grok_side_channel_endpoint_path_template,
        ),
        "provider": provider,
        "model": model,
        "model_alias": _first_passthrough_error_context_value(
            metadata,
            _AAWM_PASSTHROUGH_ERROR_LOG_MODEL_ALIAS_METADATA_KEYS,
        ),
        "route_family": _first_passthrough_error_context_value(
            metadata,
            _AAWM_PASSTHROUGH_ERROR_LOG_ROUTE_FAMILY_METADATA_KEYS,
        ),
        "status_code": status_code,
        "trace_id": _first_passthrough_error_context_value(
            metadata,
            _AAWM_PASSTHROUGH_ERROR_LOG_TRACE_METADATA_KEYS,
        ),
        "litellm_call_id": _clean_passthrough_error_context_value(
            litellm_call_id
        ),
    }
    context.update(_build_passthrough_error_log_request_shape_context(metadata))
    context.update(grok_side_channel_context)
    context.update(
        _build_passthrough_error_log_auth_context(
            request=request,
            final_headers=final_headers,
            custom_headers=custom_headers,
        )
    )
    return context


def _format_passthrough_span_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _append_passthrough_lifecycle_span(
    kwargs: Optional[dict],
    *,
    name: str,
    start_time: datetime,
    end_time: datetime,
    span_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    metadata = _ensure_passthrough_metadata(kwargs)
    if not metadata:
        return

    langfuse_spans = metadata.get("langfuse_spans")
    if not isinstance(langfuse_spans, list):
        langfuse_spans = []
        metadata["langfuse_spans"] = langfuse_spans

    descriptor: Dict[str, Any] = {
        "name": name,
        "start_time": _format_passthrough_span_timestamp(start_time),
        "end_time": _format_passthrough_span_timestamp(end_time),
    }
    if span_metadata:
        descriptor["metadata"] = span_metadata
    langfuse_spans.append(descriptor)


def _record_passthrough_duration(
    kwargs: Optional[dict],
    *,
    metric_key: str,
    span_name: str,
    start_time: datetime,
    end_time: datetime,
    span_metadata: Optional[Dict[str, Any]] = None,
) -> float:
    duration_ms = max(0.0, (end_time - start_time).total_seconds() * 1000.0)
    metadata = _ensure_passthrough_metadata(kwargs)
    if metadata:
        metadata[metric_key] = round(duration_ms, 3)
    _append_passthrough_lifecycle_span(
        kwargs,
        name=span_name,
        start_time=start_time,
        end_time=end_time,
        span_metadata={
            "duration_ms": round(duration_ms, 3),
            **(span_metadata or {}),
        },
    )
    return duration_ms


async def set_env_variables_in_header(custom_headers: Optional[dict]) -> Optional[dict]:
    """
    checks if any headers on config.yaml are defined as os.environ/COHERE_API_KEY etc

    only runs for headers defined on config.yaml

    example header can be

    {"Authorization": "Bearer os.environ/COHERE_API_KEY"}
    """
    if custom_headers is None:
        return None
    headers = {}
    for key, value in custom_headers.items():
        # langfuse Api requires base64 encoded headers - it's simpleer to just ask litellm users to set their langfuse public and secret keys
        # we can then get the b64 encoded keys here
        if key == "LANGFUSE_PUBLIC_KEY" or key == "LANGFUSE_SECRET_KEY":
            # langfuse requires b64 encoded headers - we construct that here
            _langfuse_public_key = custom_headers["LANGFUSE_PUBLIC_KEY"]
            _langfuse_secret_key = custom_headers["LANGFUSE_SECRET_KEY"]
            if isinstance(
                _langfuse_public_key, str
            ) and _langfuse_public_key.startswith("os.environ/"):
                _langfuse_public_key = get_secret_str(_langfuse_public_key)
            if isinstance(
                _langfuse_secret_key, str
            ) and _langfuse_secret_key.startswith("os.environ/"):
                _langfuse_secret_key = get_secret_str(_langfuse_secret_key)
            headers["Authorization"] = "Basic " + b64encode(
                f"{_langfuse_public_key}:{_langfuse_secret_key}".encode("utf-8")
            ).decode("ascii")
        else:
            # for all other headers
            headers[key] = value
            if isinstance(value, str) and "os.environ/" in value:
                verbose_proxy_logger.debug(
                    "pass through endpoint - looking up 'os.environ/' variable"
                )
                # get string section that is os.environ/
                start_index = value.find("os.environ/")
                _variable_name = value[start_index:]

                verbose_proxy_logger.debug(
                    "pass through endpoint - getting secret for variable name: %s",
                    _variable_name,
                )
                _secret_value = get_secret_str(_variable_name)
                if _secret_value is not None:
                    new_value = value.replace(_variable_name, _secret_value)
                    headers[key] = new_value
    return headers


async def chat_completion_pass_through_endpoint(  # noqa: PLR0915
    fastapi_response: Response,
    request: Request,
    adapter_id: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = {}
    try:
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except Exception:
            data = json.loads(body_str)

        data["adapter_id"] = adapter_id

        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
        )
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        data = await add_litellm_data_to_request(
            data=data,  # type: ignore
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        # Check key-specific aliases
        if (
            isinstance(data["model"], str)
            and user_api_key_dict.aliases
            and isinstance(user_api_key_dict.aliases, dict)
            and data["model"] in user_api_key_dict.aliases
        ):
            data["model"] = user_api_key_dict.aliases[data["model"]]

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="text_completion"
        )

        ### ROUTE THE REQUESTs ###
        router_model_names = llm_router.model_names if llm_router is not None else []
        # skip router if user passed their key
        if "api_key" in data:
            llm_response = asyncio.create_task(litellm.aadapter_completion(**data))
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif llm_router is not None and llm_router.has_model_id(
            data["model"]
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and (
                llm_router.default_deployment is not None
                or len(llm_router.pattern_router.patterns) > 0
            )
        ):  # check for wildcard routes or default deployment before checking deployment_names
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router (lowest priority)
            llm_response = asyncio.create_task(
                llm_router.aadapter_completion(**data, specific_deployment=True)
            )
        elif user_model is not None:  # `litellm --model <your-model-name>`
            llm_response = asyncio.create_task(litellm.aadapter_completion(**data))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "completion: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        # Await the llm_response task
        response = await llm_response

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        verbose_proxy_logger.debug("final response: %s", response)

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
            )
        )

        verbose_proxy_logger.debug("\nResponse from Litellm:\n{}".format(response))
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.completion(): Exception occured - {}".format(
                str(e)
            )
        )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


class HttpPassThroughEndpointHelpers(BasePassthroughUtils):
    @staticmethod
    def _raise_egress_guard_block(
        *,
        detail: str,
        url: Union[str, httpx.URL],
        credential_family: Optional[str],
        target_family: Optional[str],
        marker_families: Optional[set[str]] = None,
    ) -> None:
        alert_state = trigger_egress_guard_alert(
            reason=detail,
            target=str(url),
            credential_family=credential_family,
            target_family=target_family,
        )
        verbose_proxy_logger.critical(
            "Egress guard blocked passthrough request: detail=%s target=%s credential_family=%s target_family=%s marker_families=%s trigger_count=%s",
            detail,
            str(url),
            credential_family,
            target_family,
            sorted(marker_families) if marker_families else [],
            alert_state.get("trigger_count"),
        )
        raise HTTPException(status_code=500, detail=detail)

    @staticmethod
    def get_target_provider_family(url: Union[str, httpx.URL]) -> str:
        parsed_url = urlparse(str(url))
        hostname = (parsed_url.hostname or "").lower()

        if hostname == "api.anthropic.com" or hostname.endswith(".anthropic.com"):
            return "anthropic"
        if (
            hostname == "api.openai.com"
            or hostname.endswith(".openai.com")
            or hostname == "chatgpt.com"
            or hostname.endswith(".chatgpt.com")
        ):
            return "openai"
        if (
            hostname == "openrouter.ai"
            or hostname.endswith(".openrouter.ai")
        ):
            return "openrouter"
        if hostname == "opencode.ai" or hostname.endswith(".opencode.ai"):
            return "opencode"
        if (
            hostname == "api.x.ai"
            or hostname.endswith(".x.ai")
            or hostname == "cli-chat-proxy.grok.com"
            or hostname.endswith(".grok.com")
        ):
            return "xai"
        if hostname in {"integrate.api.nvidia.com", "ai.api.nvidia.com"}:
            return "nvidia"
        if (
            hostname == "generativelanguage.googleapis.com"
            or hostname.endswith(".googleapis.com")
            or hostname.endswith(".google.com")
            or hostname.endswith(".googleusercontent.com")
        ):
            return "google"
        return "generic"

    @staticmethod
    def get_credential_marker_families(
        headers: Optional[dict], url: Optional[Union[str, httpx.URL]] = None
    ) -> set[str]:
        normalized_headers = {
            str(key).lower(): value for key, value in dict(headers or {}).items()
        }
        marker_families: set[str] = set()

        if (
            "x-api-key" in normalized_headers
            or "anthropic-version" in normalized_headers
            or "anthropic-beta" in normalized_headers
            or "anthropic-dangerous-direct-browser-access" in normalized_headers
        ):
            marker_families.add("anthropic")

        originator = normalized_headers.get("originator")
        user_agent = normalized_headers.get("user-agent")
        if (
            "chatgpt-account-id" in normalized_headers
            or "session_id" in normalized_headers
            or "session-id" in normalized_headers
            or (
                isinstance(originator, str)
                and ("codex" in originator.lower() or "openai" in originator.lower())
            )
            or (
                isinstance(user_agent, str)
                and ("codex" in user_agent.lower() or "openai" in user_agent.lower())
            )
        ):
            marker_families.add("openai")

        if (
            "x-goog-api-key" in normalized_headers
            or "x-goog-api-client" in normalized_headers
            or "goog-api-client" in normalized_headers
        ):
            marker_families.add("google")

        if "x-xai-token-auth" in normalized_headers or any(
            header_name.startswith("x-grok-") for header_name in normalized_headers
        ):
            marker_families.add("xai")

        if url is not None:
            parsed_url = urlparse(str(url))
            query_params = httpx.QueryParams(parsed_url.query)
            if query_params.get("key"):
                marker_families.add("google")

        return marker_families

    @staticmethod
    def validate_outgoing_egress(
        *,
        url: Union[str, httpx.URL],
        headers: Optional[dict],
        credential_family: Optional[str] = None,
        expected_target_family: Optional[str] = None,
    ) -> None:
        target_family = HttpPassThroughEndpointHelpers.get_target_provider_family(url)
        if (
            expected_target_family is not None
            and target_family != "generic"
            and target_family != expected_target_family
        ):
            HttpPassThroughEndpointHelpers._raise_egress_guard_block(
                detail=(
                    f"Blocked passthrough egress: expected target family "
                    f"{expected_target_family}, got {target_family}."
                ),
                url=url,
                credential_family=credential_family,
                target_family=target_family,
            )

        if (
            credential_family is not None
            and target_family != "generic"
            and target_family != credential_family
        ):
            HttpPassThroughEndpointHelpers._raise_egress_guard_block(
                detail=(
                    f"Blocked passthrough egress: credential family "
                    f"{credential_family} cannot be sent to {target_family}."
                ),
                url=url,
                credential_family=credential_family,
                target_family=target_family,
            )

        marker_families = HttpPassThroughEndpointHelpers.get_credential_marker_families(
            headers=headers,
            url=url,
        )
        cross_provider_markers = {
            marker
            for marker in marker_families
            if target_family != "generic" and marker != target_family
        }
        if cross_provider_markers:
            HttpPassThroughEndpointHelpers._raise_egress_guard_block(
                detail=(
                    "Blocked passthrough egress due to cross-provider credential/header "
                    f"markers: target={target_family}, markers={sorted(cross_provider_markers)}."
                ),
                url=url,
                credential_family=credential_family,
                target_family=target_family,
                marker_families=cross_provider_markers,
            )

    @staticmethod
    def get_masked_passthrough_headers(headers: Optional[dict]) -> dict:
        masked_headers = dict(headers or {})
        sensitive_header_names = {
            "authorization",
            "api-key",
            "x-api-key",
            "proxy-authorization",
        }
        for header_name, header_value in list(masked_headers.items()):
            if header_name.lower() not in sensitive_header_names:
                continue
            if isinstance(header_value, str) and header_value.lower().startswith(
                "bearer "
            ):
                masked_headers[header_name] = "Bearer ***"
            else:
                masked_headers[header_name] = "***"
        return masked_headers

    @staticmethod
    def _get_passthrough_request_url_path(request: Request) -> str:
        request_url = getattr(request, "url", "")
        return urlparse(str(request_url)).path

    @staticmethod
    def _is_openai_responses_client_auth_passthrough_request(
        request: Request,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> bool:
        request_path = HttpPassThroughEndpointHelpers._get_passthrough_request_url_path(
            request
        )
        if "/openai_passthrough/" not in request_path and "/openai/" not in request_path:
            return False

        target_url = passthrough_logging_payload.get("url") or ""
        parsed_target = urlparse(str(target_url))
        target_path = parsed_target.path.rstrip("/")
        is_openai_target = bool(
            parsed_target.hostname
            and (
                "api.openai.com" in parsed_target.hostname
                or "openai.azure.com" in parsed_target.hostname
                or "chatgpt.com" in parsed_target.hostname
            )
        )
        if not is_openai_target:
            return False

        is_responses_route = (
            target_path == "/v1/responses"
            or target_path.startswith("/v1/responses/")
            or target_path == "/backend-api/codex/responses"
            or target_path.startswith("/backend-api/codex/responses/")
            or target_path == "/responses"
            or target_path.startswith("/responses/")
        )
        if not is_responses_route:
            return False

        headers = _safe_get_request_headers(request)
        return bool(
            headers.get("authorization")
            or headers.get("Authorization")
            or headers.get("api-key")
            or headers.get("Api-Key")
        )

    @staticmethod
    def _get_safe_passthrough_user_api_key_hash(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        if (
            HttpPassThroughEndpointHelpers._is_openai_responses_client_auth_passthrough_request(
                request=request,
                passthrough_logging_payload=passthrough_logging_payload,
            )
        ):
            headers = _safe_get_request_headers(request)
            auth_header = headers.get("authorization") or headers.get("Authorization")
            if isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
                return hash_token(auth_header[len("Bearer ") :])

            api_key_header = headers.get("api-key") or headers.get("Api-Key")
            if isinstance(api_key_header, str) and api_key_header:
                return hash_token(api_key_header)

        return user_api_key_dict.api_key

    @staticmethod
    def get_response_headers(
        headers: httpx.Headers,
        litellm_call_id: Optional[str] = None,
        custom_headers: Optional[dict] = None,
    ) -> dict:
        excluded_headers = {"transfer-encoding", "content-encoding"}

        return_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in excluded_headers
        }
        if litellm_call_id:
            return_headers["x-litellm-call-id"] = litellm_call_id
        if custom_headers:
            return_headers.update(custom_headers)

        return return_headers

    @staticmethod
    def get_endpoint_type(url: str) -> EndpointType:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""
        if (
            (hostname == "opencode.ai" or hostname.endswith(".opencode.ai"))
            and (path.endswith("/messages") or "/messages/" in path)
        ):
            return EndpointType.ANTHROPIC
        if (
            ("generateContent") in url
            or ("streamGenerateContent") in url
            or ("rawPredict") in url
            or ("streamRawPredict") in url
        ):
            return EndpointType.VERTEX_AI
        elif parsed_url.hostname == "api.anthropic.com":
            return EndpointType.ANTHROPIC
        elif (
            parsed_url.hostname == "api.openai.com"
            or parsed_url.hostname == "openai.azure.com"
            or parsed_url.hostname == "chatgpt.com"
            or parsed_url.hostname == "openrouter.ai"
            or (parsed_url.hostname and parsed_url.hostname.endswith(".openrouter.ai"))
            or hostname == "opencode.ai"
            or hostname.endswith(".opencode.ai")
            or parsed_url.hostname == "api.x.ai"
            or parsed_url.hostname == "cli-chat-proxy.grok.com"
            or (parsed_url.hostname and "openai.com" in parsed_url.hostname)
        ):
            return EndpointType.OPENAI
        return EndpointType.GENERIC

    @staticmethod
    async def _make_non_streaming_http_request(
        request: Request,
        async_client: httpx.AsyncClient,
        url: str,
        headers: dict,
        requested_query_params: Optional[dict] = None,
        custom_body: Optional[dict] = None,
    ) -> httpx.Response:
        """
        Make a non-streaming HTTP request

        If request is GET, don't include a JSON body
        """
        if request.method == "GET":
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
            )
        else:
            json_headers, _removed_content_type = _headers_for_json_passthrough_egress(
                headers
            )
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=json_headers,
                params=requested_query_params,
                json=custom_body,
            )
        return response

    @staticmethod
    async def non_streaming_http_request_handler(
        request: Request,
        async_client: httpx.AsyncClient,
        url: httpx.URL,
        headers: dict,
        requested_query_params: Optional[dict] = None,
        _parsed_body: Optional[dict] = None,
        raw_body: Optional[bytes] = None,
    ) -> httpx.Response:
        """
        Handle non-streaming HTTP requests

        Handles special cases when GET requests, multipart/form-data requests, and generic httpx requests
        """
        if request.method == "GET":
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
            )
        elif raw_body is not None:
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
                content=raw_body,
            )
        elif (
            HttpPassThroughEndpointHelpers.is_multipart(request) is True
            and not _parsed_body
        ):
            # Only use multipart handler if we don't have a parsed body
            # (parsed body means it was JSON despite multipart content-type header)
            return await HttpPassThroughEndpointHelpers.make_multipart_http_request(
                request=request,
                async_client=async_client,
                url=url,
                headers=headers,
                requested_query_params=requested_query_params,
            )
        else:
            # Generic httpx method
            json_headers, _removed_content_type = _headers_for_json_passthrough_egress(
                headers
            )
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=json_headers,
                params=requested_query_params,
                json=_parsed_body,
            )
        return response

    @staticmethod
    def is_multipart(request: Request) -> bool:
        """Check if the request is a multipart/form-data request"""
        return "multipart/form-data" in request.headers.get("content-type", "")

    @staticmethod
    async def _build_request_files_from_upload_file(
        upload_file: Union[UploadFile, StarletteUploadFile],
    ) -> Tuple[Optional[str], bytes, Optional[str]]:
        """Build a request files dict from an UploadFile object"""
        file_content = await upload_file.read()
        return (upload_file.filename, file_content, upload_file.content_type)

    @staticmethod
    async def make_multipart_http_request(
        request: Request,
        async_client: httpx.AsyncClient,
        url: httpx.URL,
        headers: dict,
        requested_query_params: Optional[dict] = None,
    ) -> httpx.Response:
        """Process multipart/form-data requests, handling both files and form fields"""
        form_data = await request.form()
        files = {}
        form_data_dict = {}

        for field_name, field_value in form_data.items():
            if isinstance(field_value, (StarletteUploadFile, UploadFile)):
                files[
                    field_name
                ] = await HttpPassThroughEndpointHelpers._build_request_files_from_upload_file(
                    upload_file=field_value
                )
            else:
                form_data_dict[field_name] = field_value

        # Remove content-type header - httpx will set it correctly with the new boundary
        # when it creates the multipart body from files/data parameters
        headers_copy = headers.copy()
        headers_copy.pop("content-type", None)

        response = await async_client.request(
            method=request.method,
            url=url,
            headers=headers_copy,
            params=requested_query_params,
            files=files,
            data=form_data_dict,
        )
        return response

    @staticmethod
    def _init_kwargs_for_pass_through_endpoint(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        logging_obj: LiteLLMLoggingObj,
        _parsed_body: Optional[dict] = None,
        litellm_call_id: Optional[str] = None,
    ) -> dict:
        """
        Filter out litellm params from the request body
        """
        from litellm.types.utils import all_litellm_params

        _parsed_body = _parsed_body or {}
        from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body

        working_body, _provider_body = _detach_passthrough_body_for_kwargs(_parsed_body)

        request_headers = dict(request.headers) if request.headers is not None else None
        resolved_end_user_id = (
            get_end_user_id_from_request_body(
                request_body=dict(_parsed_body),
                request_headers=request_headers,
            )
            or user_api_key_dict.end_user_id
        )
        safe_user_api_key_hash = (
            HttpPassThroughEndpointHelpers._get_safe_passthrough_user_api_key_hash(
                request=request,
                user_api_key_dict=user_api_key_dict,
                passthrough_logging_payload=passthrough_logging_payload,
            )
        )

        litellm_params_in_body = {}
        for k in all_litellm_params:
            if k in working_body:
                litellm_params_in_body[k] = working_body.pop(k, None)

        _metadata = dict(
            StandardLoggingUserAPIKeyMetadata(
                user_api_key_hash=safe_user_api_key_hash,
                user_api_key_alias=user_api_key_dict.key_alias,
                user_api_key_user_email=user_api_key_dict.user_email,
                user_api_key_user_id=user_api_key_dict.user_id,
                user_api_key_team_id=user_api_key_dict.team_id,
                user_api_key_org_id=user_api_key_dict.org_id,
                user_api_key_project_id=user_api_key_dict.project_id,
                user_api_key_team_alias=user_api_key_dict.team_alias,
                user_api_key_end_user_id=resolved_end_user_id,
                user_api_key_request_route=user_api_key_dict.request_route,
                user_api_key_spend=user_api_key_dict.spend,
                user_api_key_max_budget=user_api_key_dict.max_budget,
                user_api_key_budget_reset_at=(
                    user_api_key_dict.budget_reset_at.isoformat()
                    if user_api_key_dict.budget_reset_at
                    else None
                ),
                user_api_key_auth_metadata=user_api_key_dict.metadata,
            )
        )

        _metadata["user_api_key"] = safe_user_api_key_hash

        litellm_metadata = litellm_params_in_body.pop("litellm_metadata", None)
        metadata = litellm_params_in_body.pop("metadata", None)
        if litellm_metadata:
            _metadata.update(litellm_metadata)
        if metadata:
            _metadata.update(metadata)
        if resolved_end_user_id:
            _metadata["user_api_key_end_user_id"] = resolved_end_user_id

        _metadata = _update_metadata_with_tags_in_header(
            request=request,
            metadata=_metadata,
        )

        kwargs = {
            "litellm_params": {
                **litellm_params_in_body,  # type: ignore
                "metadata": _metadata,
                "proxy_server_request": {
                    "url": str(request.url),
                    "method": request.method,
                    "body": _shallow_copy_request_dict(working_body),
                    "headers": request_headers or {},
                },
            },
            "call_type": "pass_through_endpoint",
            "litellm_call_id": litellm_call_id,
            "passthrough_logging_payload": passthrough_logging_payload,
        }

        logging_obj.model_call_details[
            "passthrough_logging_payload"
        ] = passthrough_logging_payload

        return kwargs

    @staticmethod
    def _merge_passthrough_logging_metadata(
        parsed_body: dict,
        passthrough_logging_metadata: Optional[dict[str, Any]],
    ) -> dict:
        if not passthrough_logging_metadata:
            return parsed_body

        updated_body = dict(parsed_body)
        existing_metadata = updated_body.get("litellm_metadata")
        metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}

        existing_tags = metadata.get("tags")
        merged_tags = list(existing_tags) if isinstance(existing_tags, list) else []
        incoming_tags = passthrough_logging_metadata.get("tags")
        if isinstance(incoming_tags, list):
            for tag in incoming_tags:
                if isinstance(tag, str) and tag not in merged_tags:
                    merged_tags.append(tag)

        for key, value in passthrough_logging_metadata.items():
            if key == "tags":
                continue
            metadata[key] = value

        if merged_tags:
            metadata["tags"] = merged_tags
        updated_body["litellm_metadata"] = metadata
        return updated_body

    @staticmethod
    def construct_target_url_with_subpath(
        base_target: str, subpath: str, include_subpath: Optional[bool]
    ) -> str:
        """
        Helper function to construct the full target URL with subpath handling.

        Args:
            base_target: The base target URL
            subpath: The captured subpath from the request
            include_subpath: Whether to include the subpath in the target URL

        Returns:
            The constructed full target URL
        """
        if not include_subpath:
            return base_target

        if not subpath:
            return base_target

        # Ensure base_target ends with / and subpath doesn't start with /
        if not base_target.endswith("/"):
            base_target = base_target + "/"
        if subpath.startswith("/"):
            subpath = subpath[1:]

        return base_target + subpath

    @staticmethod
    def _update_stream_param_based_on_request_body(
        parsed_body: dict,
        stream: Optional[bool] = None,
    ) -> Optional[bool]:
        """
        If stream is provided in the request body, use it.
        Otherwise, use the stream parameter passed to the `pass_through_request` function
        """
        if "stream" in parsed_body:
            return parsed_body.get("stream", stream)
        return stream


async def pass_through_request(  # noqa: PLR0915
    request: Request,
    target: str,
    custom_headers: dict,
    user_api_key_dict: UserAPIKeyAuth,
    custom_body: Optional[dict] = None,
    forward_headers: Optional[bool] = False,
    merge_query_params: Optional[bool] = False,
    query_params: Optional[dict] = None,
    default_query_params: Optional[dict] = None,
    stream: Optional[bool] = None,
    cost_per_request: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    guardrails_config: Optional[dict] = None,
    egress_credential_family: Optional[str] = None,
    expected_target_family: Optional[str] = None,
    allowed_forward_headers: Optional[list[str]] = None,
    allowed_pass_through_prefixed_headers: Optional[list[str]] = None,
    blocked_pass_through_prefixed_headers: Optional[list[str]] = None,
    retryable_upstream_status_codes: Optional[list[int]] = None,
    caller_managed_hidden_retry: bool = False,
    raw_body_passthrough: bool = False,
    passthrough_logging_metadata: Optional[dict[str, Any]] = None,
):
    """
    Pass through endpoint handler, makes the httpx request for pass-through endpoints and ensures logging hooks are called

    Args:
        request: The incoming request
        target: The target URL
        custom_headers: The custom headers
        user_api_key_dict: The user API key dictionary
        custom_body: The custom body
        forward_headers: Whether to forward headers
        merge_query_params: Whether to merge query params
        query_params: The query params
        default_query_params: The default query params to be applied if not overridden by client
        stream: Whether to stream the response
        cost_per_request: Optional field - cost per request to the target endpoint
        custom_llm_provider: Optional field - custom LLM provider for the endpoint
        guardrails_config: Optional field - guardrails configuration for passthrough endpoint
        egress_credential_family: Optional provider family for sensitive local/client credentials
        expected_target_family: Optional provider family expected for the final egress target
        retryable_upstream_status_codes: Optional upstream status codes that will be retried by the
            caller, so generic passthrough failure logging should be deferred to the adapter layer
        caller_managed_hidden_retry: When true, disables shared pre-first-byte hidden retries so
            adapter/candidate-rotation callers do not double-retry upstream failures
        raw_body_passthrough: Forward the original request body as bytes while
            using a small synthetic body for logging. This is intended for
            native binary/protobuf side-channel endpoints.
        passthrough_logging_metadata: Optional metadata to merge only into the
            logging body. This lets routes identify GET/raw-body requests
            without changing the upstream request body.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.pass_through_endpoints.passthrough_guardrails import (
        PassthroughGuardrailHandler,
    )
    from litellm.proxy.proxy_server import proxy_logging_obj

    #########################################################
    # Initialize variables
    #########################################################
    litellm_call_id = str(uuid.uuid4())
    url: Optional[httpx.URL] = None

    # parsed request body
    _parsed_body: Optional[dict] = None
    # kwargs for pass through endpoint, contains metadata, litellm_params, call_type, litellm_call_id, passthrough_logging_payload
    kwargs: Optional[dict] = None
    error_log_context: Optional[Dict[str, Any]] = None
    raw_body: Optional[bytes] = None
    route_custom_headers = dict(custom_headers or {})
    headers: Dict[str, Any] = dict(route_custom_headers)
    retryable_status_codes = {
        status_code
        for status_code in (retryable_upstream_status_codes or [])
        if isinstance(status_code, int)
    }

    #########################################################
    try:
        start_time = datetime.now()
        url = httpx.URL(target)
        headers = HttpPassThroughEndpointHelpers.forward_headers_from_request(
            request_headers=_safe_get_request_headers(request).copy(),
            headers=headers,
            forward_headers=forward_headers,
            allowed_forward_headers=allowed_forward_headers,
            allowed_pass_through_prefixed_headers=allowed_pass_through_prefixed_headers,
            blocked_pass_through_prefixed_headers=blocked_pass_through_prefixed_headers,
        )

        # Apply default query parameters if provided, regardless of merge_query_params setting
        if default_query_params or merge_query_params:
            # Determine what to merge based on settings
            request_params = dict(request.query_params) if merge_query_params else {}

            # Create a new URL with the merged query params
            url = url.copy_with(
                query=urlencode(
                    HttpPassThroughEndpointHelpers.get_merged_query_parameters(
                        existing_url=url,
                        request_query_params=request_params,
                        default_query_params=default_query_params,
                    )
                ).encode("ascii")
            )

        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url=url,
            headers=headers,
            credential_family=egress_credential_family,
            expected_target_family=expected_target_family,
        )

        endpoint_type: EndpointType = HttpPassThroughEndpointHelpers.get_endpoint_type(
            str(url)
        )

        # Skip body parsing for multipart requests - make_multipart_http_request will handle it
        # But if custom_body is provided (e.g., JSON parsed despite multipart content-type), use it
        is_multipart = (
            HttpPassThroughEndpointHelpers.is_multipart(request) and not custom_body
        )

        if custom_body:
            _parsed_body = _copy_custom_body_for_passthrough(custom_body)
        elif is_multipart:
            # Don't parse multipart body here - it will be handled by make_multipart_http_request
            _parsed_body = {}
        elif raw_body_passthrough:
            raw_body = await request.body()
            _parsed_body = {
                "raw_body_passthrough": True,
                "raw_body_content_type": request.headers.get("content-type"),
                "raw_body_bytes": len(raw_body),
            }
        else:
            _parsed_body = await _read_request_body(request)
        _parsed_body = HttpPassThroughEndpointHelpers._merge_passthrough_logging_metadata(
            parsed_body=_parsed_body,
            passthrough_logging_metadata=passthrough_logging_metadata,
        )
        normalized_tool_schema_count = _normalize_openai_function_tool_schemas_in_body(
            _parsed_body
        )
        if normalized_tool_schema_count > 0 and isinstance(_parsed_body, dict):
            metadata = _parsed_body.get("litellm_metadata")
            if isinstance(metadata, dict):
                metadata["openai_function_tool_schema_fix_count"] = (
                    metadata.get("openai_function_tool_schema_fix_count", 0)
                    + normalized_tool_schema_count
                )
        masked_headers = HttpPassThroughEndpointHelpers.get_masked_passthrough_headers(
            headers=headers
        )
        verbose_proxy_logger.debug(
            "Pass through endpoint sending request to \nURL {}\nheaders: {}\nbody: {}\n".format(
                url, masked_headers, _parsed_body
            )
        )

        ### COLLECT GUARDRAILS FOR PASSTHROUGH ENDPOINT ###
        # Passthrough endpoints are opt-in only for guardrails
        # When enabled, collect guardrails from org/team/key levels + passthrough-specific
        guardrails_to_run = PassthroughGuardrailHandler.collect_guardrails(
            user_api_key_dict=user_api_key_dict,
            passthrough_guardrails_config=guardrails_config,
        )

        # Add guardrails to metadata if any should run
        if guardrails_to_run and len(guardrails_to_run) > 0:
            if _parsed_body is None:
                _parsed_body = {}
            if "metadata" not in _parsed_body:
                _parsed_body["metadata"] = {}
            _parsed_body["metadata"]["guardrails"] = guardrails_to_run
            verbose_proxy_logger.debug(
                f"Added guardrails to passthrough request metadata: {guardrails_to_run}"
            )

        ## LOGGING OBJECT ## - initialize before pre_call_hook so guardrails can access it
        logging_obj = Logging(
            model="unknown",
            messages=None,
            stream=False,
            call_type="pass_through_endpoint",
            start_time=start_time,
            litellm_call_id=litellm_call_id,
            function_id="1245",
        )

        # Store passthrough guardrails config on logging_obj for field targeting
        logging_obj.passthrough_guardrails_config = guardrails_config

        # Store logging_obj in data so guardrails can access it
        if _parsed_body is None:
            _parsed_body = {}
        _parsed_body["litellm_logging_obj"] = logging_obj

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        _parsed_body = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=_parsed_body,
            call_type="pass_through_endpoint",
        )
        normalized_tool_schema_count = _normalize_openai_function_tool_schemas_in_body(
            _parsed_body
        )
        if normalized_tool_schema_count > 0 and isinstance(_parsed_body, dict):
            metadata = _parsed_body.get("litellm_metadata")
            if isinstance(metadata, dict):
                metadata["openai_function_tool_schema_fix_count"] = (
                    metadata.get("openai_function_tool_schema_fix_count", 0)
                    + normalized_tool_schema_count
                )
        invalid_openai_tool_schemas = _collect_invalid_openai_function_tool_schemas(
            _parsed_body
        )
        if invalid_openai_tool_schemas:
            verbose_proxy_logger.warning(
                "Pass-through request still contains invalid OpenAI function tool schemas for target=%s invalid_tools=%s",
                str(url),
                invalid_openai_tool_schemas[:10],
            )

        async_client_obj = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PassThroughEndpoint,
            params={"timeout": 600},
        )
        async_client = async_client_obj.client
        _cleaned_headers = clean_headers(request.headers)
        passthrough_logging_payload = PassthroughStandardLoggingPayload(
            url=str(url),
            request_body=_parsed_body,
            request_method=getattr(request, "method", None),
            cost_per_request=cost_per_request,
            request_headers=_cleaned_headers,
        )
        kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
            user_api_key_dict=user_api_key_dict,
            _parsed_body=_parsed_body,
            passthrough_logging_payload=passthrough_logging_payload,
            litellm_call_id=litellm_call_id,
            request=request,
            logging_obj=logging_obj,
        )
        provider_bound_body = _provider_bound_body_from_kwargs(kwargs)
        if provider_bound_body is None:
            provider_bound_body = _parsed_body if isinstance(_parsed_body, dict) else {}
        local_prepare_completed_at = datetime.now()
        local_prepare_ms = _record_passthrough_duration(
            kwargs,
            metric_key="aawm_local_prepare_ms",
            span_name="proxy.pre_send_prepare",
            start_time=start_time,
            end_time=local_prepare_completed_at,
            span_metadata={"stage": "pre_send_prepare"},
        )

        metadata = _ensure_passthrough_metadata(kwargs)
        metadata["aawm_passthrough_endpoint_type"] = endpoint_type.value
        _merge_passthrough_request_shape_metadata(
            metadata,
            request=request,
            parsed_body=_parsed_body,
            provider_bound_body=provider_bound_body,
        )
        error_log_context = _build_passthrough_error_log_context(
            request=request,
            url=url,
            parsed_body=_parsed_body,
            kwargs=kwargs,
            passthrough_logging_metadata=passthrough_logging_metadata,
            final_headers=headers,
            custom_headers=route_custom_headers,
            custom_llm_provider=custom_llm_provider,
            status_code=None,
            litellm_call_id=litellm_call_id,
        )
        try:
            emit_aawm_route_access_log(
                request=request,
                target=url,
                request_body=_parsed_body,
                kwargs=kwargs,
            )
        except Exception:
            verbose_proxy_logger.debug(
                "Failed to emit AAWM route access log",
                exc_info=True,
            )

        # Store custom_llm_provider in kwargs and logging object if provided
        if custom_llm_provider:
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["litellm_params"] = kwargs.get(
                "litellm_params", {}
            )

        # done for supporting 'parallel_request_limiter.py' with pass-through endpoints
        logging_obj.update_environment_variables(
            model="unknown",
            user="unknown",
            optional_params={},
            litellm_params=kwargs["litellm_params"],
            call_type="pass_through_endpoint",
        )
        logging_obj.model_call_details["litellm_call_id"] = litellm_call_id

        # combine url with query params for logging
        requested_query_params: Optional[dict]
        if query_params is not None:
            requested_query_params = query_params
        else:
            requested_query_params = dict(request.query_params)

        requested_query_params_str = None
        if requested_query_params:
            requested_query_params_str = "&".join(
                f"{k}={v}" for k, v in requested_query_params.items()
            )

        logging_url = str(url)
        if requested_query_params_str:
            if "?" in str(url):
                logging_url = str(url) + "&" + requested_query_params_str
            else:
                logging_url = str(url) + "?" + requested_query_params_str

        _record_grok_billing_passthrough_request_contract(
            request=request,
            url=url,
            headers=headers,
            requested_query_params=requested_query_params,
            metadata=metadata,
            custom_llm_provider=custom_llm_provider,
        )

        request_observability_envelope = (
            _build_passthrough_request_observability_envelope(_parsed_body)
        )
        passthrough_logging_input = request_observability_envelope.logging_input
        logging_obj.update_messages(passthrough_logging_input)

        logging_obj.pre_call(
            input=passthrough_logging_input,
            api_key="",
            additional_args={
                "complete_input_dict": request_observability_envelope.complete_input_dict,
                "api_base": str(logging_url),
                "headers": masked_headers,
            },
        )
        stream = (
            HttpPassThroughEndpointHelpers._update_stream_param_based_on_request_body(
                parsed_body=_parsed_body,
                stream=stream,
            )
        )
        use_json_egress = _is_json_passthrough_egress(
            request=request,
            raw_body=raw_body,
            provider_bound_body=provider_bound_body,
        )

        if stream:
            upstream_wait_started_at = datetime.now()

            async def _send_stream_pre_first_byte() -> Tuple[httpx.Response, httpx.Request]:
                stream_headers = headers
                if use_json_egress:
                    stream_headers, removed_content_type = (
                        _headers_for_json_passthrough_egress(headers)
                    )
                    if removed_content_type:
                        _merge_passthrough_request_shape_metadata(
                            metadata,
                            request=request,
                            parsed_body=_parsed_body,
                            provider_bound_body=provider_bound_body,
                            json_egress_content_type_removed=removed_content_type,
                        )
                        if error_log_context is not None:
                            error_log_context.update(
                                _build_passthrough_error_log_request_shape_context(
                                    metadata
                                )
                            )
                req = async_client.build_request(
                    "POST",
                    url,
                    json=provider_bound_body,
                    params=requested_query_params,
                    headers=stream_headers,
                )
                response = await async_client.send(req, stream=stream)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    error_content = await e.response.aread()
                    capture_passthrough_shape(
                        mode="stream_error",
                        provider=custom_llm_provider or endpoint_type.value,
                        endpoint_type=endpoint_type,
                        url_route=str(url),
                        request_body=_parsed_body,
                        response=e.response,
                        upstream_request=getattr(e.response, "request", None) or req,
                        response_content=error_content,
                        litellm_call_id=litellm_call_id,
                        extra_metadata={"stream": True},
                    )
                    raise _build_http_exception_from_upstream_status_error(
                        e,
                        error_content,
                    ) from e
                return response, req

            response, req = await _execute_passthrough_pre_first_byte_with_hidden_retries(
                kwargs=kwargs,
                operation_name="stream_pre_first_byte",
                operation=_send_stream_pre_first_byte,
                caller_managed_hidden_retry=caller_managed_hidden_retry,
            )
            upstream_wait_completed_at = datetime.now()
            _record_passthrough_duration(
                kwargs,
                metric_key="aawm_upstream_wait_ms",
                span_name="proxy.upstream_wait",
                start_time=upstream_wait_started_at,
                end_time=upstream_wait_completed_at,
                span_metadata={"stage": "upstream_wait", "stream": True},
            )

            return StreamingResponse(
                PassThroughStreamingHandler.chunk_processor(
                    response=response,
                    request_body=_parsed_body,
                    litellm_logging_obj=logging_obj,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    passthrough_success_handler_obj=pass_through_endpoint_logging,
                    url_route=str(url),
                    passthrough_logging_payload=passthrough_logging_payload,
                    custom_llm_provider=custom_llm_provider,
                    success_handler_kwargs=kwargs,
                    upstream_wait_started_at=upstream_wait_started_at,
                    upstream_wait_completed_at=upstream_wait_completed_at,
                    local_prepare_ms=local_prepare_ms,
                    error_log_context=error_log_context,
                ),
                headers=HttpPassThroughEndpointHelpers.get_response_headers(
                    headers=response.headers,
                    litellm_call_id=litellm_call_id,
                ),
                status_code=response.status_code,
            )

        upstream_wait_started_at = datetime.now()

        async def _send_non_stream_pre_first_byte() -> httpx.Response:
            non_stream_headers = headers
            if use_json_egress:
                non_stream_headers, removed_content_type = (
                    _headers_for_json_passthrough_egress(headers)
                )
                if removed_content_type:
                    _merge_passthrough_request_shape_metadata(
                        metadata,
                        request=request,
                        parsed_body=_parsed_body,
                        provider_bound_body=provider_bound_body,
                        json_egress_content_type_removed=removed_content_type,
                    )
                    if error_log_context is not None:
                        error_log_context.update(
                            _build_passthrough_error_log_request_shape_context(
                                metadata
                            )
                        )
            response = (
                await HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
                    request=request,
                    async_client=async_client,
                    url=url,
                    headers=non_stream_headers,
                    requested_query_params=requested_query_params,
                    _parsed_body=provider_bound_body,
                    raw_body=raw_body,
                )
            )
            if _is_streaming_response(response) is True:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    error_content = await e.response.aread()
                    capture_passthrough_shape(
                        mode="stream_error",
                        provider=custom_llm_provider or endpoint_type.value,
                        endpoint_type=endpoint_type,
                        url_route=str(url),
                        request_body=_parsed_body,
                        response=e.response,
                        upstream_request=getattr(e.response, "request", None),
                        response_content=error_content,
                        litellm_call_id=litellm_call_id,
                        extra_metadata={"stream": True},
                    )
                    raise _build_http_exception_from_upstream_status_error(
                        e,
                        error_content,
                    ) from e
                return response

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                capture_passthrough_shape(
                    mode="nonstream_error",
                    provider=custom_llm_provider or endpoint_type.value,
                    endpoint_type=endpoint_type,
                    url_route=str(url),
                    request_body=_parsed_body,
                    response=e.response,
                    upstream_request=getattr(e.response, "request", None),
                    response_content=e.response.content,
                    litellm_call_id=litellm_call_id,
                    extra_metadata={"stream": False},
                )
                raise _build_http_exception_from_upstream_status_error(
                    e,
                    e.response.text,
                ) from e
            return response

        response = await _execute_passthrough_pre_first_byte_with_hidden_retries(
            kwargs=kwargs,
            operation_name="non_stream_pre_first_byte",
            operation=_send_non_stream_pre_first_byte,
            caller_managed_hidden_retry=caller_managed_hidden_retry,
        )
        upstream_wait_completed_at = datetime.now()
        _record_passthrough_duration(
            kwargs,
            metric_key="aawm_upstream_wait_ms",
            span_name="proxy.upstream_wait",
            start_time=upstream_wait_started_at,
            end_time=upstream_wait_completed_at,
            span_metadata={"stage": "upstream_wait", "stream": False},
        )
        verbose_proxy_logger.debug("response.headers= %s", response.headers)

        if _is_streaming_response(response) is True:
            return StreamingResponse(
                PassThroughStreamingHandler.chunk_processor(
                    response=response,
                    request_body=_parsed_body,
                    litellm_logging_obj=logging_obj,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    passthrough_success_handler_obj=pass_through_endpoint_logging,
                    url_route=str(url),
                    passthrough_logging_payload=passthrough_logging_payload,
                    custom_llm_provider=custom_llm_provider,
                    success_handler_kwargs=kwargs,
                    upstream_wait_started_at=upstream_wait_started_at,
                    upstream_wait_completed_at=upstream_wait_completed_at,
                    local_prepare_ms=local_prepare_ms,
                    error_log_context=error_log_context,
                ),
                headers=HttpPassThroughEndpointHelpers.get_response_headers(
                    headers=response.headers,
                    litellm_call_id=litellm_call_id,
                ),
                status_code=response.status_code,
            )

        if response.status_code >= 300:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
                headers={
                    str(header_name): str(header_value)
                    for header_name, header_value in response.headers.items()
                }
                or None,
            )

        finalize_started_at = datetime.now()
        content = await response.aread()

        ## LOG SUCCESS
        response_body: Optional[dict] = get_response_body(response)
        passthrough_logging_payload["response_body"] = response_body
        capture_passthrough_shape(
            mode="nonstream",
            provider=custom_llm_provider or endpoint_type.value,
            endpoint_type=endpoint_type,
            url_route=str(url),
            request_body=_parsed_body,
            response=response,
            upstream_request=getattr(response, "request", None),
            response_body=response_body,
            response_content=content,
            litellm_call_id=litellm_call_id,
            extra_metadata={"stream": False},
        )
        end_time = datetime.now()
        asyncio.create_task(
            pass_through_endpoint_logging.pass_through_async_success_handler(
                httpx_response=response,
                response_body=response_body,
                url_route=str(url),
                result="",
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                cache_hit=False,
                request_body=_parsed_body,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )
        )
        local_finalize_ms = _record_passthrough_duration(
            kwargs,
            metric_key="aawm_local_finalize_ms",
            span_name="proxy.post_response_finalize",
            start_time=finalize_started_at,
            end_time=end_time,
            span_metadata={"stage": "post_response_finalize", "stream": False},
        )
        metadata = _ensure_passthrough_metadata(kwargs)
        if metadata:
            metadata["aawm_total_proxy_overhead_ms"] = round(
                local_prepare_ms + local_finalize_ms, 3
            )
            metadata["aawm_total_proxy_duration_ms"] = round(
                max(0.0, (end_time - start_time).total_seconds() * 1000.0), 3
            )

        ## CUSTOM HEADERS - `x-litellm-*`
        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=litellm_call_id,
            model_id=None,
            cache_key=None,
            api_base=str(url._uri_reference),
        )

        return Response(
            content=content,
            status_code=response.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                headers=response.headers,
                custom_headers=custom_headers,
            ),
        )
    except Exception as e:
        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=litellm_call_id,
            model_id=None,
            cache_key=None,
            api_base=str(url._uri_reference) if url else None,
        )
        status_code = _extract_exception_status_code(e)
        suppress_retryable_failure_logging = (
            status_code in retryable_status_codes if status_code is not None else False
        )
        if error_log_context is None:
            error_log_context = _build_passthrough_error_log_context(
                request=request,
                url=url,
                parsed_body=_parsed_body,
                kwargs=kwargs,
                passthrough_logging_metadata=passthrough_logging_metadata,
                final_headers=headers,
                custom_headers=route_custom_headers,
                custom_llm_provider=custom_llm_provider,
                status_code=status_code,
                litellm_call_id=litellm_call_id,
            )
        else:
            error_log_context = {
                **error_log_context,
                "status_code": status_code,
            }
        suppress_terminal_failure_traceback = (
            _should_log_passthrough_terminal_failure_without_traceback(
                exc=e,
                kwargs=kwargs,
                status_code=status_code,
            )
        )
        suppress_provider_rate_limit_traceback = (
            _is_passthrough_expected_provider_rate_limit(status_code=status_code)
        )
        suppress_grok_billing_timeout_traceback = (
            _is_known_grok_billing_passthrough_timeout_cancel_response(
                request=request,
                url=url,
                custom_llm_provider=custom_llm_provider,
                status_code=status_code,
                exc=e,
            )
        )
        suppress_grok_signals_auth_context_traceback = (
            _is_known_grok_signals_auth_context_response(
                request=request,
                url=url,
                custom_llm_provider=custom_llm_provider,
                status_code=status_code,
                exc=e,
            )
        )
        suppress_chatgpt_codex_block_page_traceback = (
            _is_known_chatgpt_codex_block_page_response(
                url=url,
                status_code=status_code,
                exc=e,
            )
        )
        suppress_google_code_assist_tos_traceback = (
            _is_known_google_code_assist_tos_violation_response(
                url=url,
                custom_llm_provider=custom_llm_provider,
                status_code=status_code,
                exc=e,
            )
        )
        known_anthropic_failure_kind = _get_known_anthropic_passthrough_failure_kind(
            url=url,
            custom_llm_provider=custom_llm_provider,
            status_code=status_code,
            exc=e,
        )
        if suppress_retryable_failure_logging:
            verbose_proxy_logger.debug(
                "Pass through endpoint received retryable upstream status=%s; deferring failure logging to adapter handling",
                status_code,
            )
        elif suppress_provider_rate_limit_traceback:
            verbose_proxy_logger.warning(
                "Pass through endpoint surfaced upstream rate limit status=%s error=%s",
                status_code,
                str(e),
                extra={
                    **error_log_context,
                    "failure_kind": "expected_provider_rate_limit",
                },
            )
        elif suppress_grok_billing_timeout_traceback:
            verbose_proxy_logger.warning(
                "Pass through endpoint surfaced known Grok billing timeout/cancel status=%s error=%s",
                status_code,
                str(e),
                extra={
                    **error_log_context,
                    "failure_kind": _get_passthrough_grok_billing_timeout_failure_kind(),
                },
            )
        elif suppress_grok_signals_auth_context_traceback:
            verbose_proxy_logger.warning(
                "Pass through endpoint surfaced known Grok signals auth-context status=%s error=%s",
                status_code,
                str(e),
                extra={
                    **error_log_context,
                    "failure_kind": _get_passthrough_grok_signals_auth_context_failure_kind(),
                },
            )
        elif suppress_chatgpt_codex_block_page_traceback:
            verbose_proxy_logger.warning(
                "Pass through endpoint surfaced ChatGPT Codex block page status=%s error=%s",
                status_code,
                str(e),
                extra={
                    **error_log_context,
                    "failure_kind": "openai_chatgpt_codex_block_page",
                },
            )
        elif suppress_google_code_assist_tos_traceback:
            verbose_proxy_logger.warning(
                "Pass through endpoint surfaced Google Code Assist account TOS violation status=%s error=%s",
                status_code,
                str(e),
                extra={
                    **error_log_context,
                    "failure_kind": "google_code_assist_tos_violation",
                },
            )
        elif known_anthropic_failure_kind is not None:
            verbose_proxy_logger.warning(
                "Pass through endpoint surfaced Anthropic provider/client failure status=%s error=%s",
                status_code,
                str(e),
                extra={
                    **error_log_context,
                    "failure_kind": known_anthropic_failure_kind,
                },
            )
        elif suppress_terminal_failure_traceback:
            hidden_retry_metadata = _ensure_passthrough_metadata(kwargs)
            hidden_retry_failure_classification = hidden_retry_metadata.get(
                "aawm_passthrough_hidden_retry_failure_classification"
            )
            terminal_failure_context = {
                **error_log_context,
                "failure_kind": _get_passthrough_terminal_failure_kind(
                    hidden_retry_failure_classification=hidden_retry_failure_classification,
                ),
                "hidden_retry_final_outcome": hidden_retry_metadata.get(
                    "aawm_passthrough_hidden_retry_final_outcome"
                ),
                "hidden_retry_failure_classification": hidden_retry_failure_classification,
                "hidden_retry_count": hidden_retry_metadata.get(
                    "aawm_passthrough_hidden_retry_count"
                ),
            }
            verbose_proxy_logger.error(
                (
                    "Pass through endpoint exhausted hidden retries for upstream failure "
                    "status=%s error=%s final_outcome=%s retry_count=%s"
                ),
                status_code,
                str(e),
                hidden_retry_metadata.get("aawm_passthrough_hidden_retry_final_outcome"),
                hidden_retry_metadata.get("aawm_passthrough_hidden_retry_count"),
                extra=terminal_failure_context,
                exc_info=False,
            )
        else:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.pass_through_endpoint(): Exception occured - {}".format(
                    str(e)
                ),
                extra=error_log_context,
            )

        #########################################################
        # Monitoring: Trigger post_call_failure_hook
        # for pass through endpoint failure
        #########################################################
        request_payload: dict = dict(_parsed_body or {})
        # add user_api_key_dict, litellm_call_id, passthrough_logging_payloa for logging
        if kwargs:
            for key, value in kwargs.items():
                request_payload[key] = value

        if (
            "model" not in request_payload
            and _parsed_body
            and isinstance(_parsed_body, dict)
        ):
            request_payload["model"] = _parsed_body.get("model", "")
        if "custom_llm_provider" not in request_payload and custom_llm_provider:
            request_payload["custom_llm_provider"] = custom_llm_provider
        _enrich_passthrough_failure_request_payload(
            request_payload=request_payload,
            request=request,
            url=url,
            custom_llm_provider=custom_llm_provider,
        )

        traceback_str = None
        if (
            not suppress_terminal_failure_traceback
            and not suppress_provider_rate_limit_traceback
            and not suppress_grok_billing_timeout_traceback
            and not suppress_grok_signals_auth_context_traceback
            and not suppress_chatgpt_codex_block_page_traceback
            and not suppress_google_code_assist_tos_traceback
            and known_anthropic_failure_kind is None
        ):
            traceback_str = traceback.format_exc(
                limit=MAXIMUM_TRACEBACK_LINES_TO_LOG,
            )
        if (
            not suppress_retryable_failure_logging
            and not suppress_grok_billing_timeout_traceback
            and not suppress_grok_signals_auth_context_traceback
        ):
            try:
                await proxy_logging_obj.post_call_failure_hook(
                    user_api_key_dict=user_api_key_dict,
                    original_exception=e,
                    request_data=request_payload,
                    traceback_str=traceback_str,
                )
            finally:
                await _direct_capture_xai_passthrough_failure(
                    user_api_key_dict=user_api_key_dict,
                    original_exception=e,
                    request_payload=request_payload,
                    traceback_str=traceback_str,
                    url=url,
                    custom_llm_provider=custom_llm_provider,
                )

        #########################################################

        if isinstance(e, HTTPException):
            proxy_exc = ProxyException(
                message=getattr(e, "message", str(getattr(e, "detail", str(e)))),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=status_code
                if status_code is not None
                else getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
                headers=custom_headers,
            )
            setattr(proxy_exc, "detail", getattr(e, "detail", None))
            upstream_headers = getattr(e, "headers", None)
            if isinstance(upstream_headers, dict):
                setattr(
                    proxy_exc,
                    "upstream_headers",
                    {
                        str(header_name): str(header_value)
                        for header_name, header_value in upstream_headers.items()
                    },
                )
            raise proxy_exc
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=status_code
                if status_code is not None
                else getattr(e, "status_code", 500),
                headers=custom_headers,
            )


def _update_metadata_with_tags_in_header(request: Request, metadata: dict) -> dict:
    """
    If tags are in the request headers, add them to the metadata

    Used for google and vertex JS SDKs, and Azure passthrough
    Checks both 'tags' and 'x-litellm-tags' headers
    """
    tags_to_add = []

    # Check for 'tags' header first
    _tags = request.headers.get("tags")
    if _tags:
        tags_to_add.extend([tag.strip() for tag in _tags.split(",")])

    _tags = request.headers.get("x-litellm-tags")
    if _tags:
        tags_to_add.extend([tag.strip() for tag in _tags.split(",")])

    # Only add tags key if there are tags to add
    if tags_to_add:
        if "tags" not in metadata:
            metadata["tags"] = []
        metadata["tags"].extend(tags_to_add)

    return metadata


async def _parse_request_data_by_content_type(
    request: Request,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Parse request data based on content type.

    Handles JSON, multipart/form-data, and URL-encoded form data.

    Returns:
        Tuple of (query_params_data, custom_body_data, file_data, stream)
    """
    content_type = request.headers.get("content-type", "")

    query_params_data = None
    custom_body_data = None
    file_data = None
    stream = None

    if "application/json" in content_type:
        # ✅ Handle JSON
        try:
            body = await request.json()
            query_params_data = body.get("query_params")
            custom_body_data = body.get("custom_body")
            stream = body.get("stream")
        except json.JSONDecodeError:
            # Handle requests with no body (e.g., DELETE requests)
            pass
    elif "multipart/form-data" in content_type:
        # ✅ Try to parse as JSON first (handles misconfigured clients sending JSON with multipart content-type)
        # If that fails, skip parsing - pass_through_request will handle actual multipart
        try:
            body = await request.json()
            # Successfully parsed as JSON - treat as JSON body
            query_params_data = body.get("query_params")
            custom_body_data = body.get("custom_body")
            stream = body.get("stream")
            # If custom_body is not set, use the entire body
            if custom_body_data is None and body:
                custom_body_data = body
        except (json.JSONDecodeError, Exception):
            # Not JSON - this is actual multipart data
            # Skip parsing here to avoid consuming the request body stream
            # make_multipart_http_request will handle it
            pass

    elif "application/x-www-form-urlencoded" in content_type:
        # ✅ Handle URL-encoded form data
        form = await request.form()
        query_params_data = form.get("query_params")
        custom_body_data = form.get("custom_body")

    else:
        # ✅ Fallback: maybe no body, just query params
        query_params_data = dict(request.query_params) or None

    return query_params_data, custom_body_data, file_data, stream


def create_pass_through_route(
    endpoint,
    target: str,
    custom_headers: Optional[dict] = None,
    _forward_headers: Optional[bool] = False,
    _merge_query_params: Optional[bool] = False,
    dependencies: Optional[List] = None,
    include_subpath: Optional[bool] = False,
    cost_per_request: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    is_streaming_request: Optional[bool] = False,
    query_params: Optional[dict] = None,
    default_query_params: Optional[dict] = None,
    guardrails: Optional[Dict[str, Any]] = None,
    egress_credential_family: Optional[str] = None,
    expected_target_family: Optional[str] = None,
    allowed_forward_headers: Optional[list[str]] = None,
    allowed_pass_through_prefixed_headers: Optional[list[str]] = None,
    blocked_pass_through_prefixed_headers: Optional[list[str]] = None,
    caller_managed_hidden_retry: bool = False,
):
    # check if target is an adapter.py or a url
    from litellm._uuid import uuid
    from litellm.proxy.types_utils.utils import get_instance_fn

    try:
        if isinstance(target, CustomLogger):
            adapter = target
        else:
            adapter = get_instance_fn(value=target)
        adapter_id = str(uuid.uuid4())
        litellm.adapters = [{"id": adapter_id, "adapter": adapter}]

        async def endpoint_func(  # type: ignore
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
            subpath: str = "",  # captures sub-paths when include_subpath=True
            custom_body: Optional[
                dict
            ] = None,  # accepted for signature compatibility with URL-based path; not forwarded because chat_completion_pass_through_endpoint does not support it
        ):
            return await chat_completion_pass_through_endpoint(
                fastapi_response=fastapi_response,
                request=request,
                adapter_id=adapter_id,
                user_api_key_dict=user_api_key_dict,
            )

    except Exception:
        verbose_proxy_logger.debug("Defaulting to target being a url.")

        async def endpoint_func(  # type: ignore
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
            subpath: str = "",  # captures sub-paths when include_subpath=True
            custom_body: Optional[
                dict
            ] = None,  # caller-supplied body takes precedence over request-parsed body
        ):
            from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
                InitPassThroughEndpointHelpers,
            )

            path = request.url.path

            # Parse request data based on content type
            (
                query_params_data,
                custom_body_data,
                file_data,
                stream,
            ) = await _parse_request_data_by_content_type(request)

            if not InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                route=path
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Pass-through endpoint {endpoint} not found. This could have been deleted or not yet added to the proxy.",
                )

            passthrough_params = (
                InitPassThroughEndpointHelpers.get_registered_pass_through_route(
                    route=path, method=request.method
                )
            )
            target_params = {
                "target": target,
                "custom_headers": custom_headers,
                "forward_headers": _forward_headers,
                "merge_query_params": _merge_query_params,
                "cost_per_request": cost_per_request,
                "guardrails": None,
                "egress_credential_family": egress_credential_family,
                "expected_target_family": expected_target_family,
                "allowed_forward_headers": allowed_forward_headers,
                "allowed_pass_through_prefixed_headers": allowed_pass_through_prefixed_headers,
                "blocked_pass_through_prefixed_headers": blocked_pass_through_prefixed_headers,
                "caller_managed_hidden_retry": caller_managed_hidden_retry,
            }

            if passthrough_params is not None:
                target_params.update(passthrough_params.get("passthrough_params", {}))

            # Extract and cast parameters with proper types
            param_target = target_params.get("target") or target
            param_custom_headers = target_params.get("custom_headers", custom_headers)
            param_forward_headers = target_params.get(
                "forward_headers", _forward_headers
            )
            param_merge_query_params = target_params.get(
                "merge_query_params", _merge_query_params
            )
            param_cost_per_request = target_params.get(
                "cost_per_request", cost_per_request
            )
            param_guardrails = target_params.get("guardrails", None)
            param_default_query_params = target_params.get("default_query_params", None)
            param_egress_credential_family = target_params.get(
                "egress_credential_family", egress_credential_family
            )
            param_expected_target_family = target_params.get(
                "expected_target_family", expected_target_family
            )
            param_allowed_forward_headers = target_params.get(
                "allowed_forward_headers", allowed_forward_headers
            )
            param_allowed_pass_through_prefixed_headers = target_params.get(
                "allowed_pass_through_prefixed_headers",
                allowed_pass_through_prefixed_headers,
            )
            param_blocked_pass_through_prefixed_headers = target_params.get(
                "blocked_pass_through_prefixed_headers",
                blocked_pass_through_prefixed_headers,
            )
            param_caller_managed_hidden_retry = target_params.get(
                "caller_managed_hidden_retry",
                caller_managed_hidden_retry,
            )

            # Construct the full target URL with subpath if needed
            full_target = (
                HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
                    base_target=cast(str, param_target),
                    subpath=subpath,
                    include_subpath=include_subpath,
                )
            )

            # Ensure custom_headers is a dict
            headers_dict = (
                param_custom_headers if isinstance(param_custom_headers, dict) else {}
            )

            # Ensure query_params and custom_body are dicts or None
            final_query_params = (
                query_params_data if isinstance(query_params_data, dict) else {}
            )
            if query_params:
                final_query_params.update(query_params)
            # Caller-supplied custom_body takes precedence over the request-parsed body
            final_custom_body: Optional[dict] = None
            if custom_body is not None:
                final_custom_body = custom_body
            elif isinstance(custom_body_data, dict):
                final_custom_body = custom_body_data

            return await pass_through_request(  # type: ignore
                request=request,
                target=full_target,
                custom_headers=headers_dict,
                user_api_key_dict=user_api_key_dict,
                forward_headers=cast(Optional[bool], param_forward_headers),
                merge_query_params=cast(Optional[bool], param_merge_query_params),
                query_params=final_query_params,
                default_query_params=cast(Optional[dict], param_default_query_params),
                stream=is_streaming_request or stream,
                custom_body=final_custom_body,
                cost_per_request=cast(Optional[float], param_cost_per_request),
                custom_llm_provider=custom_llm_provider,
                guardrails_config=cast(Optional[dict], param_guardrails),
                egress_credential_family=cast(
                    Optional[str], param_egress_credential_family
                ),
                expected_target_family=cast(Optional[str], param_expected_target_family),
                allowed_forward_headers=cast(
                    Optional[list[str]], param_allowed_forward_headers
                ),
                allowed_pass_through_prefixed_headers=cast(
                    Optional[list[str]], param_allowed_pass_through_prefixed_headers
                ),
                blocked_pass_through_prefixed_headers=cast(
                    Optional[list[str]], param_blocked_pass_through_prefixed_headers
                ),
                caller_managed_hidden_retry=bool(param_caller_managed_hidden_retry),
            )

    return endpoint_func


def create_websocket_passthrough_route(
    endpoint: str,
    target: str,
    custom_headers: Optional[dict] = None,
    _forward_headers: Optional[bool] = False,
    dependencies: Optional[List] = None,
    cost_per_request: Optional[float] = None,
):
    """
    Create a WebSocket passthrough route function.

    Args:
        endpoint: The endpoint path (for logging purposes)
        target: The target WebSocket URL (e.g., "wss://api.example.com/ws")
        custom_headers: Custom headers to include in the WebSocket connection
        _forward_headers: Whether to forward incoming headers
        dependencies: FastAPI dependencies to inject

    Returns:
        A WebSocket passthrough function that can be registered with app.websocket()
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket

    async def websocket_endpoint_func(
        websocket: WebSocket,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth_websocket),
        **kwargs,  # For additional query parameters
    ):
        """
        WebSocket passthrough endpoint function.

        This function handles the WebSocket connection by:
        1. Accepting the incoming WebSocket connection
        2. Establishing a connection to the target WebSocket
        3. Forwarding messages bidirectionally
        4. Handling connection cleanup
        """
        return await websocket_passthrough_request(
            websocket=websocket,
            target=target,
            custom_headers=custom_headers or {},
            user_api_key_dict=user_api_key_dict,
            forward_headers=_forward_headers,
            endpoint=endpoint,
            cost_per_request=cost_per_request,
            accept_websocket=True,  # Generic usage should accept the WebSocket
        )

    return websocket_endpoint_func


async def websocket_passthrough_request(  # noqa: PLR0915
    websocket: WebSocket,
    target: str,
    custom_headers: dict,
    user_api_key_dict: UserAPIKeyAuth,
    forward_headers: Optional[bool] = False,
    endpoint: Optional[str] = None,
    cost_per_request: Optional[float] = None,
    accept_websocket: bool = True,
):
    """
    WebSocket passthrough request handler.

    Args:
        websocket: The incoming WebSocket connection
        target: The target WebSocket URL
        custom_headers: Custom headers to include in the connection
        user_api_key_dict: The user API key dictionary
        forward_headers: Whether to forward incoming headers
        endpoint: The endpoint path (for logging purposes)
        cost_per_request: Optional field - cost per request to the target endpoint
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.proxy_server import proxy_logging_obj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    # Initialize tracking variables
    start_time = datetime.now()
    websocket_messages: list[dict[str, Any]] = []
    litellm_call_id = str(uuid.uuid4())

    verbose_proxy_logger.info(
        f"WebSocket passthrough ({endpoint}): Starting WebSocket connection to {target}"
    )

    # Only accept the WebSocket if requested (for generic usage)
    if accept_websocket:
        await websocket.accept()
        verbose_proxy_logger.debug(
            f"WebSocket passthrough ({endpoint}): WebSocket connection accepted"
        )

    # Prepare headers for the upstream connection
    upstream_headers = custom_headers.copy()

    if forward_headers:
        # Forward relevant headers from the incoming request
        incoming_headers = dict(websocket.headers)
        for header_name, header_value in incoming_headers.items():
            # Only forward certain headers to avoid conflicts
            if header_name.lower() in [
                "authorization",
                "x-api-key",
                "x-goog-user-project",
            ]:
                upstream_headers[header_name] = header_value

    # Initialize logging object similar to HTTP passthrough
    logging_obj = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,  # WebSockets are inherently streaming
        call_type="pass_through_endpoint",
        start_time=start_time,
        litellm_call_id=litellm_call_id,
        function_id="websocket_passthrough",
    )

    # Create passthrough logging payload
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=target,
        request_body={},  # WebSocket doesn't have a traditional request body
        request_method="WEBSOCKET",
        cost_per_request=cost_per_request,
    )

    # Create a dummy request object for WebSocket connections to maintain compatibility
    # with the existing _init_kwargs_for_pass_through_endpoint function
    class DummyRequest:
        def __init__(
            self, url: str, method: str = "WEBSOCKET", headers: Optional[dict] = None
        ):
            self.url = url
            self.method = method
            self.headers = headers or {}

        def __str__(self):
            return f"DummyRequest(url={self.url}, method={self.method})"

    dummy_request = DummyRequest(
        url=target,
        method="WEBSOCKET",
        headers=dict(websocket.headers) if hasattr(websocket, "headers") else {},
    )

    # Initialize kwargs for logging using the same pattern as HTTP passthrough
    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        user_api_key_dict=user_api_key_dict,
        _parsed_body={},  # WebSocket doesn't have a traditional request body
        passthrough_logging_payload=passthrough_logging_payload,
        litellm_call_id=litellm_call_id,
        request=dummy_request,  # type: ignore
        logging_obj=logging_obj,
    )

    # Update logging environment variables
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params=dict(kwargs.get("litellm_params", {})),
        call_type="pass_through_endpoint",
    )
    logging_obj.model_call_details["litellm_call_id"] = litellm_call_id

    # Pre-call logging
    logging_obj.pre_call(
        input=[{"role": "user", "content": "WebSocket connection"}],
        api_key="",
        additional_args={
            "complete_input_dict": {},
            "api_base": target,
            "headers": upstream_headers,
        },
    )

    ### CALL HOOKS ### - modify incoming data / reject request before calling the model
    websocket_data: dict[str, Any] = {}
    websocket_data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=websocket_data,
        call_type="pass_through_endpoint",
    )

    try:
        verbose_proxy_logger.debug(
            f"WebSocket passthrough ({endpoint}): Establishing upstream connection to {target}"
        )
        async with connect(
            target,
            additional_headers=upstream_headers,
        ) as upstream_ws:
            verbose_proxy_logger.info(
                f"WebSocket passthrough ({endpoint}): Upstream connection established successfully"
            )

            async def forward_client_to_upstream() -> None:
                """Forward messages from client to upstream WebSocket"""
                try:
                    while True:
                        message = await websocket.receive()
                        message_type = message.get("type")
                        if message_type == "websocket.disconnect":
                            await upstream_ws.close()
                            break

                        text_data = message.get("text")
                        bytes_data = message.get("bytes")

                        if text_data is not None:
                            # Try to extract model from client setup message for Vertex AI Live
                            if endpoint and "/vertex_ai/live" in endpoint:
                                verbose_proxy_logger.debug(
                                    f"WebSocket passthrough ({endpoint}): Processing client message for model extraction"
                                )
                                try:
                                    client_message = json.loads(text_data)
                                    if (
                                        isinstance(client_message, dict)
                                        and "setup" in client_message
                                    ):
                                        setup_data = client_message["setup"]
                                        verbose_proxy_logger.debug(
                                            f"WebSocket passthrough ({endpoint}): Found setup data in client message: {setup_data}"
                                        )
                                        if (
                                            isinstance(setup_data, dict)
                                            and "model" in setup_data
                                        ):
                                            extracted_model = (
                                                _extract_model_from_vertex_ai_setup(
                                                    setup_data
                                                )
                                            )
                                            if extracted_model:
                                                kwargs["model"] = extracted_model
                                                kwargs[
                                                    "custom_llm_provider"
                                                ] = "vertex_ai-language-models"
                                                # Update logging object with correct model
                                                logging_obj.model = extracted_model
                                                logging_obj.model_call_details[
                                                    "model"
                                                ] = extracted_model
                                                logging_obj.model_call_details[
                                                    "custom_llm_provider"
                                                ] = "vertex_ai"
                                                verbose_proxy_logger.info(
                                                    f"WebSocket passthrough ({endpoint}): Successfully extracted model '{extracted_model}' and set provider to 'vertex_ai' from client setup message"
                                                )
                                            else:
                                                verbose_proxy_logger.warning(
                                                    f"WebSocket passthrough ({endpoint}): Failed to extract model from client setup data: {setup_data}"
                                                )
                                        else:
                                            verbose_proxy_logger.debug(
                                                f"WebSocket passthrough ({endpoint}): Setup data does not contain model field: {setup_data}"
                                            )
                                    else:
                                        verbose_proxy_logger.debug(
                                            f"WebSocket passthrough ({endpoint}): Client message does not contain setup data"
                                        )
                                except (json.JSONDecodeError, KeyError, TypeError) as e:
                                    verbose_proxy_logger.debug(
                                        f"WebSocket passthrough ({endpoint}): Client message is not a valid setup message: {e}"
                                    )
                                    pass  # Not a JSON message or doesn't contain setup data

                            await upstream_ws.send(text_data)
                        elif bytes_data is not None:
                            await upstream_ws.send(bytes_data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    verbose_proxy_logger.exception(
                        f"WebSocket passthrough ({endpoint}): error forwarding client message"
                    )
                    await upstream_ws.close()

            async def forward_upstream_to_client() -> None:
                """Forward messages from upstream to client WebSocket"""
                try:
                    # Wait for the first response from upstream
                    raw_response = await upstream_ws.recv(decode=False)
                    # Ensure raw_response is bytes before decoding
                    if isinstance(raw_response, str):
                        raw_response = raw_response.encode("ascii")
                    setup_response = json.loads(raw_response.decode("ascii"))
                    verbose_proxy_logger.debug(f"Setup response: {setup_response}")

                    # Extract model and provider from setup response for Vertex AI Live
                    if endpoint and "/vertex_ai/live" in endpoint:
                        verbose_proxy_logger.debug(
                            f"WebSocket passthrough ({endpoint}): Processing server setup response for model extraction"
                        )
                        extracted_model = _extract_model_from_vertex_ai_setup(
                            setup_response
                        )
                        if extracted_model:
                            kwargs["model"] = extracted_model
                            kwargs["custom_llm_provider"] = "vertex_ai_language_models"
                            # Update logging object with correct model
                            logging_obj.model = extracted_model
                            logging_obj.model_call_details["model"] = extracted_model
                            logging_obj.model_call_details[
                                "custom_llm_provider"
                            ] = "vertex_ai_language_models"
                            verbose_proxy_logger.debug(
                                f"WebSocket passthrough ({endpoint}): Successfully extracted model '{extracted_model}' and set provider to 'vertex_ai' from server setup response"
                            )
                        else:
                            verbose_proxy_logger.warning(
                                f"WebSocket passthrough ({endpoint}): Failed to extract model from server setup response: {setup_response}"
                            )
                    else:
                        verbose_proxy_logger.debug(
                            f"WebSocket passthrough ({endpoint}): Not a Vertex AI Live endpoint, skipping model extraction"
                        )

                    # Send the setup response to the client
                    await websocket.send_text(json.dumps(setup_response))

                    # Now continuously forward messages from upstream to client
                    async for upstream_message in upstream_ws:
                        if isinstance(upstream_message, bytes):
                            await websocket.send_bytes(upstream_message)
                            # Parse and collect for cost tracking
                            try:
                                message_data = json.loads(upstream_message.decode())
                                websocket_messages.append(message_data)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass
                        else:
                            await websocket.send_text(upstream_message)
                            # Parse and collect for cost tracking
                            try:
                                message_data = json.loads(upstream_message)
                                websocket_messages.append(message_data)
                            except json.JSONDecodeError:
                                pass

                except (ConnectionClosedOK, ConnectionClosedError) as e:
                    verbose_proxy_logger.debug(
                        f"Upstream WebSocket connection closed: {e}"
                    )
                    pass
                except asyncio.CancelledError:
                    verbose_proxy_logger.debug(
                        "asyncio.CancelledError in forward_upstream_to_client"
                    )
                    raise
                except Exception as e:
                    verbose_proxy_logger.debug(
                        f"Exception in forward_upstream_to_client: {e}"
                    )
                    verbose_proxy_logger.exception(
                        f"WebSocket passthrough ({endpoint}): error forwarding upstream message"
                    )
                    raise

            # Create tasks for bidirectional message forwarding
            tasks = [
                asyncio.create_task(forward_client_to_upstream()),
                asyncio.create_task(forward_upstream_to_client()),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check for exceptions in completed tasks
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception

            end_time = datetime.now()

            # Update passthrough logging payload with response data
            passthrough_logging_payload["response_body"] = websocket_messages  # type: ignore
            passthrough_logging_payload["end_time"] = end_time  # type: ignore

            # Remove logging_obj from kwargs to avoid duplicate keyword argument
            success_kwargs = kwargs.copy()
            success_kwargs.pop("logging_obj", None)

            # # Add user authentication context for database logging
            # if user_api_key_dict:
            #     success_kwargs.setdefault('litellm_params', {})
            #     success_kwargs['litellm_params'].update({
            #         'proxy_server_request': {
            #             'body': {
            #                 'user': user_api_key_dict.user_id,
            #                 'team_id': user_api_key_dict.team_id,
            #                 'end_user_id': user_api_key_dict.end_user_id,
            #             }
            #         }
            #     })
            #     # Also add the user_api_key for direct access
            #     success_kwargs['user_api_key'] = user_api_key_dict.api_key

            # Create a dummy httpx.Response for WebSocket connections
            class MockWebSocketResponse:
                def __init__(self, target_url: str):
                    self.status_code = 200
                    self.text = "WebSocket connection successful"
                    self.headers: dict[str, str] = {}
                    self.request = MockWebSocketRequest(target_url)

            class MockWebSocketRequest:
                def __init__(self, target_url: str):
                    self.method = "WEBSOCKET"
                    self.url = target_url

            mock_response = MockWebSocketResponse(target)

            # Use the same success handler as HTTP passthrough endpoints
            asyncio.create_task(
                pass_through_endpoint_logging.pass_through_async_success_handler(
                    httpx_response=mock_response,  # type: ignore
                    response_body=websocket_messages,  # type: ignore
                    url_route=endpoint or "",
                    result="websocket_connection_successful",
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=logging_obj,
                    cache_hit=False,
                    request_body={},
                    **success_kwargs,
                )
            )

            # Call the proxy logging success hook
            if proxy_logging_obj:
                await proxy_logging_obj.post_call_success_hook(
                    data={},
                    user_api_key_dict=user_api_key_dict,
                    response={"status": "websocket_connection_successful"},  # type: ignore
                )

    except InvalidStatus as exc:
        verbose_proxy_logger.exception(
            f"WebSocket passthrough ({endpoint}): upstream rejected WebSocket connection"
        )

        # Prepare request payload for logging
        request_payload = {}
        if kwargs:
            for key, value in kwargs.items():
                request_payload[key] = value

        # Log the connection failure using the same pattern as HTTP
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=exc,
            request_data=request_payload,
            traceback_str=traceback.format_exc(
                limit=MAXIMUM_TRACEBACK_LINES_TO_LOG,
            ),
        )

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(
                code=getattr(exc, "status_code", 1011),
                reason="Upstream connection rejected",
            )
    except Exception as e:
        verbose_proxy_logger.exception(
            f"WebSocket passthrough ({endpoint}): unexpected error while proxying WebSocket"
        )

        # Prepare request payload for logging
        request_payload = {}
        if kwargs:
            for key, value in kwargs.items():
                request_payload[key] = value

        # Log the unexpected error using the same pattern as HTTP
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_payload,
            traceback_str=traceback.format_exc(
                limit=MAXIMUM_TRACEBACK_LINES_TO_LOG,
            ),
        )

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason="WebSocket passthrough error")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


def _is_streaming_response(response: httpx.Response) -> bool:
    _content_type = response.headers.get("content-type")
    if _content_type is not None and "text/event-stream" in _content_type:
        return True
    return False


def _extract_model_from_vertex_ai_setup(setup_response: dict) -> Optional[str]:
    """
    Extract the model name from Vertex AI Live setup response.

    The setup response can contain a model field in two formats:
    1. Direct: {"model": "projects/.../models/gemini-2.0-flash-live-preview-04-09"}
    2. Nested: {"setup": {"model": "projects/.../models/gemini-2.0-flash-live-preview-04-09"}}

    We extract just the model name: "gemini-2.0-flash-live-preview-04-09"
    """
    try:
        # Handle both direct model field and nested setup.model field
        model_path = None
        if isinstance(setup_response, dict):
            if "model" in setup_response:
                model_path = setup_response["model"]
            elif (
                "setup" in setup_response
                and isinstance(setup_response["setup"], dict)
                and "model" in setup_response["setup"]
            ):
                model_path = setup_response["setup"]["model"]

        if isinstance(model_path, str) and "/models/" in model_path:
            # Extract the model name after the last "/models/"
            model_name = model_path.split("/models/")[-1]
            return model_name
    except Exception as e:
        verbose_proxy_logger.debug(f"Error extracting model from setup response: {e}")
    return None


class SafeRouteAdder:
    """
    Wrapper class for adding routes to FastAPI app.
    Only adds routes if they don't already exist on the app.
    """

    @staticmethod
    def _is_path_registered(app: FastAPI, path: str, methods: List[str]) -> bool:
        """
        Check if a path with any of the specified methods is already registered on the app.

        Args:
            app: The FastAPI application instance
            path: The path to check (e.g., "/v1/chat/completions")
            methods: List of HTTP methods to check (e.g., ["GET", "POST"])

        Returns:
            True if the path is already registered with any of the methods, False otherwise
        """
        for route in app.routes:
            # Use getattr to safely access route attributes
            route_path = getattr(route, "path", None)
            route_methods = getattr(route, "methods", None)

            if route_path == path and route_methods is not None:
                # Check if any of the methods overlap
                if any(method in route_methods for method in methods):
                    return True
        return False

    @staticmethod
    def add_api_route_if_not_exists(
        app: FastAPI,
        path: str,
        endpoint: Any,
        methods: List[str],
        dependencies: Optional[List] = None,
    ) -> bool:
        """
        Add an API route to the app only if it doesn't already exist.

        Args:
            app: The FastAPI application instance
            path: The path for the route
            endpoint: The endpoint function/callable
            methods: List of HTTP methods
            dependencies: Optional list of dependencies

        Returns:
            True if route was added, False if it already existed
        """
        if SafeRouteAdder._is_path_registered(app=app, path=path, methods=methods):
            verbose_proxy_logger.debug(
                "Skipping route registration - path %s with methods %s already registered on app",
                path,
                methods,
            )
            return False

        app.add_api_route(
            path=path,
            endpoint=endpoint,
            methods=methods,
            dependencies=dependencies,
        )
        verbose_proxy_logger.debug(
            "Successfully added route: %s with methods %s",
            path,
            methods,
        )
        return True


class InitPassThroughEndpointHelpers:
    @staticmethod
    def add_exact_path_route(
        app: FastAPI,
        path: str,
        target: str,
        custom_headers: Optional[dict],
        forward_headers: Optional[bool],
        merge_query_params: Optional[bool],
        dependencies: Optional[List],
        cost_per_request: Optional[float],
        endpoint_id: str,
        guardrails: Optional[dict] = None,
        methods: Optional[List[str]] = None,
        default_query_params: Optional[dict] = None,
    ):
        """Add exact path route for pass-through endpoint"""
        # Default to all methods if none specified (backward compatibility)
        if methods is None or len(methods) == 0:
            methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]

        # Create route key that includes methods for uniqueness
        methods_str = ",".join(sorted(methods))
        route_key = f"{endpoint_id}:exact:{path}:{methods_str}"

        # Check if this exact route is already registered
        if route_key in _registered_pass_through_routes:
            verbose_proxy_logger.debug(
                "Updating duplicate exact pass through endpoint: %s with methods %s (already registered)",
                path,
                methods,
            )

        verbose_proxy_logger.debug(
            "adding exact pass through endpoint: %s, methods: %s, dependencies: %s",
            path,
            methods,
            dependencies,
        )

        # Use SafeRouteAdder to only add route if it doesn't exist on the app
        SafeRouteAdder.add_api_route_if_not_exists(
            app=app,
            path=path,
            endpoint=create_pass_through_route(  # type: ignore
                path,
                target,
                custom_headers,
                forward_headers,
                merge_query_params,
                dependencies,
                cost_per_request=cost_per_request,
                default_query_params=default_query_params,
                guardrails=guardrails,
            ),
            methods=methods,
            dependencies=dependencies,
        )

        # Always register/update the route metadata (headers, target) even if FastAPI route exists
        _registered_pass_through_routes[route_key] = {
            "endpoint_id": endpoint_id,
            "path": path,
            "type": "exact",
            "methods": methods,
            "passthrough_params": {
                "target": target,
                "custom_headers": custom_headers,
                "forward_headers": forward_headers,
                "merge_query_params": merge_query_params,
                "default_query_params": default_query_params,
                "dependencies": dependencies,
                "cost_per_request": cost_per_request,
                "guardrails": guardrails,
            },
        }

    @staticmethod
    def add_subpath_route(
        app: FastAPI,
        path: str,
        target: str,
        custom_headers: Optional[dict],
        forward_headers: Optional[bool],
        merge_query_params: Optional[bool],
        dependencies: Optional[List],
        cost_per_request: Optional[float],
        endpoint_id: str,
        guardrails: Optional[dict] = None,
        methods: Optional[List[str]] = None,
        default_query_params: Optional[dict] = None,
    ):
        """Add wildcard route for sub-paths"""
        # Default to all methods if none specified (backward compatibility)
        if methods is None or len(methods) == 0:
            methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]

        wildcard_path = f"{path}/{{subpath:path}}"
        methods_str = ",".join(sorted(methods))
        route_key = f"{endpoint_id}:subpath:{path}:{methods_str}"

        # Check if this subpath route is already registered
        if route_key in _registered_pass_through_routes:
            verbose_proxy_logger.debug(
                "Updating duplicate wildcard pass through endpoint: %s with methods %s (already registered)",
                wildcard_path,
                methods,
            )

        verbose_proxy_logger.debug(
            "adding wildcard pass through endpoint: %s, methods: %s, dependencies: %s",
            wildcard_path,
            methods,
            dependencies,
        )

        # Use SafeRouteAdder to only add route if it doesn't exist on the app
        SafeRouteAdder.add_api_route_if_not_exists(
            app=app,
            path=wildcard_path,
            endpoint=create_pass_through_route(  # type: ignore
                path,
                target,
                custom_headers,
                forward_headers,
                merge_query_params,
                dependencies,
                include_subpath=True,
                cost_per_request=cost_per_request,
                default_query_params=default_query_params,
                guardrails=guardrails,
            ),
            methods=methods,
            dependencies=dependencies,
        )

        # Register the route to prevent duplicates only if it was added
        _registered_pass_through_routes[route_key] = {
            "endpoint_id": endpoint_id,
            "path": path,
            "type": "subpath",
            "methods": methods,
            "passthrough_params": {
                "target": target,
                "custom_headers": custom_headers,
                "forward_headers": forward_headers,
                "merge_query_params": merge_query_params,
                "default_query_params": default_query_params,
                "dependencies": dependencies,
                "cost_per_request": cost_per_request,
                "guardrails": guardrails,
            },
        }

    @staticmethod
    def remove_endpoint_routes(endpoint_id: str):
        """Remove all routes for a specific endpoint ID from the registry"""
        keys_to_remove = [
            key
            for key, value in _registered_pass_through_routes.items()
            if value["endpoint_id"] == endpoint_id
        ]
        for key in keys_to_remove:
            del _registered_pass_through_routes[key]
            verbose_proxy_logger.debug(
                "Removed pass-through route from registry: %s", key
            )

    @staticmethod
    def clear_all_pass_through_routes():
        """Clear all pass-through routes from the registry"""
        _registered_pass_through_routes.clear()

    @staticmethod
    def get_all_registered_pass_through_routes() -> List[str]:
        """Get all registered pass-through endpoints from the registry"""
        return list(_registered_pass_through_routes.keys())

    @staticmethod
    def _build_full_path_with_root(path: str) -> str:
        """
        Build full path by prepending server root path if needed.

        Args:
            path: The relative path to build

        Returns:
            Full path with server root prepended (if root is not "/")
        """
        root_path = get_server_root_path()
        if root_path == "/":
            return path
        return f"{root_path}{path}"

    @staticmethod
    def is_registered_pass_through_route(route: str) -> bool:
        """
        Check if route is a registered pass-through endpoint from DB

        Uses the in-memory registry to avoid additional DB queries
        Optimized for minimal latency

        Args:
            route: The route to check

        Returns:
            bool: True if route is a registered pass-through endpoint, False otherwise
        """
        ## CHECK IF MAPPED PASS THROUGH ENDPOINT
        normalized_route = normalize_route_for_root_path(route)
        if normalized_route is not None:
            for mapped_route in LiteLLMRoutes.mapped_pass_through_routes.value:
                if normalized_route.startswith(mapped_route):
                    return True

        # Fast path: check if any registered route key contains this path
        # Keys are in format: "{endpoint_id}:exact:{path}:{methods}" or "{endpoint_id}:subpath:{path}:{methods}"
        # For backward compatibility, also support old format: "{endpoint_id}:exact:{path}" or "{endpoint_id}:subpath:{path}"
        # Extract unique paths from keys for quick checking
        for key in _registered_pass_through_routes.keys():
            parts = key.split(":", 3)  # Split into [endpoint_id, type, path, methods?]
            if len(parts) >= 3:
                route_type = parts[1]
                registered_path = (
                    InitPassThroughEndpointHelpers._build_full_path_with_root(parts[2])
                )
                if route_type == "exact" and route == registered_path:
                    return True
                elif route_type == "subpath":
                    if route == registered_path or route.startswith(
                        registered_path + "/"
                    ):
                        return True

        return False

    @staticmethod
    def get_registered_pass_through_route(
        route: str, method: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get passthrough params for a given route and optionally filter by HTTP method"""
        for key in _registered_pass_through_routes.keys():
            parts = key.split(":", 3)  # Split into [endpoint_id, type, path, methods?]
            if len(parts) >= 3:
                route_type = parts[1]
                registered_path = (
                    InitPassThroughEndpointHelpers._build_full_path_with_root(parts[2])
                )

                # Get the methods for this route
                route_methods = _registered_pass_through_routes[key].get("methods", [])

                # Check if path matches
                path_matches = False
                if route_type == "exact" and route == registered_path:
                    path_matches = True
                elif route_type == "subpath":
                    if route == registered_path or route.startswith(
                        registered_path + "/"
                    ):
                        path_matches = True

                # If path matches and method filter is provided, check if method is allowed
                if path_matches:
                    if method is None or not route_methods or method in route_methods:
                        return _registered_pass_through_routes[key]

        return None


def _get_combined_pass_through_endpoints(
    pass_through_endpoints: Union[List[Dict], List[PassThroughGenericEndpoint]],
    config_pass_through_endpoints: List[Dict],
):
    """Get combined pass-through endpoints from db + config"""
    return pass_through_endpoints + config_pass_through_endpoints


async def initialize_pass_through_endpoints(
    pass_through_endpoints: Union[List[Dict], List[PassThroughGenericEndpoint]],
):
    """
    1. Create a global list of pass-through endpoints (db + config)
    2. Clear all existing pass-through endpoints from the FastAPI app routes
    3. Add new endpoints to the in-memory registry

    Initialize a list of pass-through endpoints by adding them to the FastAPI app routes

    Args:
        pass_through_endpoints: List of pass-through endpoints to initialize

    Returns:
        None
    """
    from litellm._uuid import uuid

    verbose_proxy_logger.debug("initializing pass through endpoints")
    from litellm.proxy._types import CommonProxyErrors, LiteLLMRoutes
    from litellm.proxy.proxy_server import (
        app,
        config_passthrough_endpoints,
        premium_user,
    )

    ## get combined pass-through endpoints from db + config
    combined_pass_through_endpoints: List[Union[Dict, PassThroughGenericEndpoint]]

    if config_passthrough_endpoints is not None:
        combined_pass_through_endpoints = _get_combined_pass_through_endpoints(  # type: ignore
            pass_through_endpoints, config_passthrough_endpoints
        )
    else:
        combined_pass_through_endpoints = pass_through_endpoints  # type: ignore

    ## clear all existing pass-through endpoints from the FastAPI app routes
    # InitPassThroughEndpointHelpers.clear_all_pass_through_routes()

    # get a list of all registered pass-through endpoints
    # mark the ones that are visited in the list
    # remove the ones that are not visited from the list
    registered_pass_through_endpoints = (
        InitPassThroughEndpointHelpers.get_all_registered_pass_through_routes()
    )

    visited_endpoints = set()

    for endpoint in combined_pass_through_endpoints:
        if isinstance(endpoint, PassThroughGenericEndpoint):
            endpoint = endpoint.model_dump()

        # Auto-generate ID for backwards compatibility if not present
        if endpoint.get("id") is None:
            endpoint["id"] = str(uuid.uuid4())

        # Get the endpoint_id as a string (guaranteed to be set at this point)
        endpoint_id: str = endpoint["id"]

        _target = endpoint.get("target", None)
        _path: Optional[str] = endpoint.get("path", None)
        if _path is None:
            raise ValueError("Path is required for pass-through endpoint")
        _custom_headers = endpoint.get("headers", None)
        _custom_headers = await set_env_variables_in_header(
            custom_headers=_custom_headers
        )
        _forward_headers = endpoint.get("forward_headers", None)
        _merge_query_params = endpoint.get("merge_query_params", None)
        _default_query_params = endpoint.get("default_query_params", None)
        _auth = endpoint.get("auth", None)
        _dependencies = None
        if _auth is not None and str(_auth).lower() == "true":
            if premium_user is not True:
                raise ValueError(
                    "Error Setting Authentication on Pass Through Endpoint: {}".format(
                        CommonProxyErrors.not_premium_user.value
                    )
                )
            _dependencies = [Depends(user_api_key_auth)]
            LiteLLMRoutes.openai_routes.value.append(_path)

        if _target is None:
            continue

        # Get guardrails config if present
        _guardrails = endpoint.get("guardrails", None)

        # Get methods list if present (None means all methods for backward compatibility)
        _methods = endpoint.get("methods", None)

        # Add exact path route
        verbose_proxy_logger.debug(
            "Initializing pass through endpoint: %s (ID: %s)", _path, endpoint_id
        )
        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path=_path,
            target=_target,
            custom_headers=_custom_headers,
            forward_headers=_forward_headers,
            merge_query_params=_merge_query_params,
            dependencies=_dependencies,
            cost_per_request=endpoint.get("cost_per_request", None),
            endpoint_id=endpoint_id,
            guardrails=_guardrails,
            methods=_methods,
            default_query_params=_default_query_params,
        )

        # Generate route key with methods for tracking
        methods_for_key = (
            _methods if _methods else ["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        methods_str = ",".join(sorted(methods_for_key))
        visited_endpoints.add(f"{endpoint_id}:exact:{_path}:{methods_str}")

        # Add wildcard route for sub-paths
        if endpoint.get("include_subpath", False) is True:
            InitPassThroughEndpointHelpers.add_subpath_route(
                app=app,
                path=_path,
                target=_target,
                custom_headers=_custom_headers,
                forward_headers=_forward_headers,
                merge_query_params=_merge_query_params,
                dependencies=_dependencies,
                cost_per_request=endpoint.get("cost_per_request", None),
                endpoint_id=endpoint_id,
                guardrails=_guardrails,
                methods=_methods,
                default_query_params=_default_query_params,
            )

            visited_endpoints.add(f"{endpoint_id}:subpath:{_path}:{methods_str}")

        verbose_proxy_logger.debug(
            "Added new pass through endpoint: %s (ID: %s)", _path, endpoint_id
        )

    # remove the ones that are not visited from the list
    for endpoint_key in registered_pass_through_endpoints:
        if endpoint_key not in visited_endpoints:
            InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_key)


def _get_pass_through_endpoints_from_config() -> List[PassThroughGenericEndpoint]:
    """
    Get pass-through endpoints defined in the config file.
    These are read-only and cannot be edited via the UI.
    Malformed endpoints are logged and skipped; they do not crash the function.
    """
    from pydantic import ValidationError

    from litellm.proxy.proxy_server import config_passthrough_endpoints

    if config_passthrough_endpoints is None or len(config_passthrough_endpoints) == 0:
        return []

    returned_endpoints: List[PassThroughGenericEndpoint] = []
    for endpoint in config_passthrough_endpoints:
        try:
            if isinstance(endpoint, dict):
                endpoint_dict = dict(endpoint)
                endpoint_dict["is_from_config"] = True
                returned_endpoints.append(PassThroughGenericEndpoint(**endpoint_dict))
            elif isinstance(endpoint, PassThroughGenericEndpoint):
                # Create a copy with is_from_config=True
                endpoint_dict = endpoint.model_dump()
                endpoint_dict["is_from_config"] = True
                returned_endpoints.append(PassThroughGenericEndpoint(**endpoint_dict))
        except ValidationError as e:
            verbose_proxy_logger.warning(
                "Skipping malformed pass-through endpoint from config: %s",
                e,
                exc_info=False,
            )

    return returned_endpoints


async def _get_pass_through_endpoints_from_db(
    endpoint_id: Optional[str] = None,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
) -> List[PassThroughGenericEndpoint]:
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.proxy_server import get_config_general_settings

    try:
        if user_api_key_dict is None:
            user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        return []

    pass_through_endpoint_data: Optional[List] = response.field_value
    if pass_through_endpoint_data is None:
        return []

    returned_endpoints: List[PassThroughGenericEndpoint] = []
    if endpoint_id is None:
        # Return all endpoints from DB, mark as not from config
        for endpoint in pass_through_endpoint_data:
            if isinstance(endpoint, dict):
                endpoint_dict = dict(endpoint)
                endpoint_dict["is_from_config"] = False
                returned_endpoints.append(PassThroughGenericEndpoint(**endpoint_dict))
            elif isinstance(endpoint, PassThroughGenericEndpoint):
                endpoint_dict = endpoint.model_dump()
                endpoint_dict["is_from_config"] = False
                returned_endpoints.append(PassThroughGenericEndpoint(**endpoint_dict))
    else:
        # Find specific endpoint by ID
        found_endpoint = _find_endpoint_by_id(pass_through_endpoint_data, endpoint_id)
        if found_endpoint is not None:
            endpoint_dict = (
                found_endpoint.model_dump()
                if isinstance(found_endpoint, PassThroughGenericEndpoint)
                else dict(found_endpoint)
            )
            endpoint_dict["is_from_config"] = False
            returned_endpoints.append(PassThroughGenericEndpoint(**endpoint_dict))

    return returned_endpoints


async def _filter_endpoints_by_team_allowed_routes(
    team_id: str,
    pass_through_endpoints: List[PassThroughGenericEndpoint],
    prisma_client,
) -> List[PassThroughGenericEndpoint]:
    """
    Filter pass-through endpoints based on team's allowed_passthrough_routes metadata.

    Args:
        team_id: The team ID to check permissions for
        pass_through_endpoints: List of endpoints to filter
        prisma_client: Database client

    Returns:
        Filtered list of endpoints based on team permissions

    Raises:
        HTTPException: If team is not found
    """
    # retrieve team from db
    team = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id},
    )
    if team is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Team not found"},
        )

    # retrieve team metadata
    team_metadata = team.metadata
    if (
        team_metadata is not None
        and team_metadata.get("allowed_passthrough_routes") is not None
    ):
        ## FILTER pass_through_endpoints by allowed_passthrough_routes
        pass_through_endpoints = [
            endpoint
            for endpoint in pass_through_endpoints
            if endpoint.path in team_metadata.get("allowed_passthrough_routes")
        ]

    return pass_through_endpoints


@router.get(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
@router.get(
    "/config/pass_through_endpoint/team/{team_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
async def get_pass_through_endpoints(
    endpoint_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    team_id: Optional[str] = None,
):
    """
    GET configured pass through endpoint.

    If no endpoint_id given, return all configured endpoints.
    """  ## Get existing pass-through endpoint field value
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Get endpoints from DB (editable via UI)
    db_endpoints = await _get_pass_through_endpoints_from_db(
        endpoint_id=endpoint_id, user_api_key_dict=user_api_key_dict
    )

    # Get endpoints from config file (read-only, not editable via UI)
    config_endpoints = _get_pass_through_endpoints_from_config()

    # Merge: config endpoints not in DB + all DB endpoints (DB overrides config for same path)
    db_paths = {ep.path for ep in db_endpoints}
    config_only_endpoints = [ep for ep in config_endpoints if ep.path not in db_paths]
    if endpoint_id is not None:
        # When filtering by endpoint_id, only return if found in DB (config endpoints use generated IDs)
        pass_through_endpoints = db_endpoints
    else:
        pass_through_endpoints = config_only_endpoints + db_endpoints

    if team_id is not None:
        pass_through_endpoints = await _filter_endpoints_by_team_allowed_routes(
            team_id=team_id,
            pass_through_endpoints=pass_through_endpoints,
            prisma_client=prisma_client,
        )

    return PassThroughEndpointResponse(endpoints=pass_through_endpoints)


@router.post(
    "/config/pass_through_endpoint/{endpoint_id}",
    dependencies=[Depends(user_api_key_auth)],
)
async def update_pass_through_endpoints(
    endpoint_id: str,
    data: PassThroughGenericEndpoint,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update a pass-through endpoint by ID.
    """
    from litellm.proxy.proxy_server import (
        get_config_general_settings,
        update_config_general_settings,
    )

    ## Get existing pass-through endpoint field value
    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail={"error": "No pass-through endpoints found"},
        )

    pass_through_endpoint_data: Optional[List] = response.field_value
    if pass_through_endpoint_data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "No pass-through endpoints found"},
        )

    # Find the endpoint to update
    found_endpoint = _find_endpoint_by_id(pass_through_endpoint_data, endpoint_id)

    if found_endpoint is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Endpoint with ID '{endpoint_id}' not found"},
        )

    # Find the index for updating the list
    endpoint_index = None
    for idx, endpoint in enumerate(pass_through_endpoint_data):
        _endpoint = (
            PassThroughGenericEndpoint(**endpoint)
            if isinstance(endpoint, dict)
            else endpoint
        )
        if _endpoint.id == endpoint_id:
            endpoint_index = idx
            break

    if endpoint_index is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Could not find index for endpoint with ID '{endpoint_id}'"
            },
        )

    # Get the update data as dict, excluding None values for partial updates
    # Exclude is_from_config as it's a response-only field (computed at read time)
    update_data = data.model_dump(exclude_none=True, exclude={"is_from_config"})

    # Start with existing endpoint data
    endpoint_dict = found_endpoint.model_dump()

    # Update with new data (only non-None values)
    endpoint_dict.update(update_data)

    # Preserve existing ID if not provided in update and endpoint has ID
    if "id" not in update_data and found_endpoint.id is not None:
        endpoint_dict["id"] = found_endpoint.id

    # Remove is_from_config before saving - it's a response-only field (computed at read time)
    endpoint_dict.pop("is_from_config", None)

    # Create updated endpoint object
    updated_endpoint = PassThroughGenericEndpoint(**endpoint_dict)

    # Update the list
    pass_through_endpoint_data[endpoint_index] = endpoint_dict

    # Remove old routes from registry before they get re-registered
    InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_id)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=pass_through_endpoint_data,
        config_type="general_settings",
    )

    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    # Re-register the route with updated headers
    _custom_headers: Optional[dict] = updated_endpoint.headers or {}
    _custom_headers = await set_env_variables_in_header(custom_headers=_custom_headers)

    if updated_endpoint.include_subpath:
        InitPassThroughEndpointHelpers.add_subpath_route(
            app=request.app,
            path=updated_endpoint.path,
            target=updated_endpoint.target,
            custom_headers=_custom_headers,
            forward_headers=None,  # Defaults not available in model? assuming None logic handles it
            merge_query_params=None,
            dependencies=None,
            cost_per_request=updated_endpoint.cost_per_request,
            endpoint_id=updated_endpoint.id or endpoint_id or "",
            guardrails=getattr(updated_endpoint, "guardrails", None),
            methods=updated_endpoint.methods,
            default_query_params=updated_endpoint.default_query_params,
        )
    else:
        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=request.app,
            path=updated_endpoint.path,
            target=updated_endpoint.target,
            custom_headers=_custom_headers,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=updated_endpoint.cost_per_request,
            endpoint_id=updated_endpoint.id or endpoint_id or "",
            guardrails=getattr(updated_endpoint, "guardrails", None),
            methods=updated_endpoint.methods,
            default_query_params=updated_endpoint.default_query_params,
        )

    return PassThroughEndpointResponse(
        endpoints=[updated_endpoint] if updated_endpoint else []
    )


@router.post(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
)
async def create_pass_through_endpoints(
    data: PassThroughGenericEndpoint,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create new pass-through endpoint
    """
    from litellm._uuid import uuid
    from litellm.proxy.proxy_server import (
        get_config_general_settings,
        update_config_general_settings,
    )

    ## Get existing pass-through endpoint field value

    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        response = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=None
        )

    ## Auto-generate ID if not provided
    # Exclude is_from_config as it's a response-only field (computed at read time)
    data_dict = data.model_dump(exclude={"is_from_config"})
    if data_dict.get("id") is None:
        data_dict["id"] = str(uuid.uuid4())

    if response.field_value is None:
        response.field_value = [data_dict]
    elif isinstance(response.field_value, List):
        response.field_value.append(data_dict)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=response.field_value,
        config_type="general_settings",
    )
    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    # Return the created endpoint with the generated ID
    created_endpoint = PassThroughGenericEndpoint(**data_dict)

    # Register the new route
    _custom_headers: Optional[dict] = created_endpoint.headers or {}
    _custom_headers = await set_env_variables_in_header(custom_headers=_custom_headers)

    if created_endpoint.include_subpath:
        InitPassThroughEndpointHelpers.add_subpath_route(
            app=request.app,
            path=created_endpoint.path,
            target=created_endpoint.target,
            custom_headers=_custom_headers,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=created_endpoint.cost_per_request,
            endpoint_id=created_endpoint.id or "",
            guardrails=getattr(created_endpoint, "guardrails", None),
            methods=created_endpoint.methods,
            default_query_params=created_endpoint.default_query_params,
        )
    else:
        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=request.app,
            path=created_endpoint.path,
            target=created_endpoint.target,
            custom_headers=_custom_headers,
            forward_headers=None,
            merge_query_params=None,
            dependencies=None,
            cost_per_request=created_endpoint.cost_per_request,
            endpoint_id=created_endpoint.id or "",
            guardrails=getattr(created_endpoint, "guardrails", None),
            methods=created_endpoint.methods,
            default_query_params=created_endpoint.default_query_params,
        )

    return PassThroughEndpointResponse(endpoints=[created_endpoint])


@router.delete(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
async def delete_pass_through_endpoints(
    endpoint_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a pass-through endpoint by ID.

    Returns - the deleted endpoint
    """
    from litellm.proxy.proxy_server import (
        get_config_general_settings,
        update_config_general_settings,
    )

    ## Get existing pass-through endpoint field value

    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        response = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=None
        )

    ## Update field by removing endpoint
    pass_through_endpoint_data: Optional[List] = response.field_value
    if response.field_value is None or pass_through_endpoint_data is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "There are no pass-through endpoints setup."},
        )

    # Find the endpoint to delete
    found_endpoint = _find_endpoint_by_id(pass_through_endpoint_data, endpoint_id)

    if found_endpoint is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Endpoint with ID '{}' was not found in pass-through endpoint list.".format(
                    endpoint_id
                )
            },
        )

    # Find the index for deleting from the list
    endpoint_index = None
    for idx, endpoint in enumerate(pass_through_endpoint_data):
        _endpoint = (
            PassThroughGenericEndpoint(**endpoint)
            if isinstance(endpoint, dict)
            else endpoint
        )
        if _endpoint.id == endpoint_id:
            endpoint_index = idx
            break

    if endpoint_index is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Could not find index for endpoint with ID '{endpoint_id}'"
            },
        )

    # Remove the endpoint
    pass_through_endpoint_data.pop(endpoint_index)
    response_obj = found_endpoint

    # Remove routes from registry
    InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_id)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=pass_through_endpoint_data,
        config_type="general_settings",
    )
    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    return PassThroughEndpointResponse(endpoints=[response_obj])


def _find_endpoint_by_id(
    endpoints_data: List,
    endpoint_id: str,
) -> Optional[PassThroughGenericEndpoint]:
    """
    Find an endpoint by ID.

    Args:
        endpoints_data: List of endpoint data (dicts or PassThroughGenericEndpoint objects)
        endpoint_id: ID to search for

    Returns:
        Found endpoint or None if not found
    """
    for endpoint in endpoints_data:
        _endpoint: Optional[PassThroughGenericEndpoint] = None
        if isinstance(endpoint, dict):
            _endpoint = PassThroughGenericEndpoint(**endpoint)
        elif isinstance(endpoint, PassThroughGenericEndpoint):
            _endpoint = endpoint

        # Only compare IDs to IDs
        if _endpoint is not None and _endpoint.id == endpoint_id:
            return _endpoint

    return None


async def initialize_pass_through_endpoints_in_db():
    """
    Gets all pass-through endpoints from db and initializes them in the proxy server.
    """
    pass_through_endpoints = await _get_pass_through_endpoints_from_db()
    await initialize_pass_through_endpoints(
        pass_through_endpoints=pass_through_endpoints
    )
