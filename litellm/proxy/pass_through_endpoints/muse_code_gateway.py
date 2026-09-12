"""Alpha-gated Muse Code catalog facade and Responses helpers.

Serves ``GET /muse-code/models`` in the schema Muse probes at
``{base-url}/muse-code/models``. The router is inert unless
``AAWM_MUSE_CODE_FACADE_ENABLED`` is truthy (``1`` / ``true`` / ``yes`` / ``on``).
When disabled, the path 404s like an unregistered route.

``POST /responses`` stays on the existing Responses handler. When the facade is
enabled and the request is Muse-shaped, helpers here flatten namespace tools,
stamp session/observability metadata, and wrap SSE so Muse sees
``event:`` lines, ``sequence_number``, ``response.in_progress``, and ``[DONE]``.

Inbound auth is LiteLLM virtual-key ``user_api_key_auth``. Prefer
``Authorization`` bearer; also accept ``x-litellm-api-key`` (dual-key facades).
Do not hmac-compare to a Meta token.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import AsyncIterator, Callable
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.aawm_route_logging import (
    emit_aawm_route_access_log,
    record_aawm_route_rollup_failure,
    record_aawm_route_rollup_turn,
    register_aawm_route_rollup_access_log_replacement,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import RESPONSES_API_TERMINAL_STREAM_EVENTS

AAWM_MUSE_CODE_FACADE_ENABLED_ENV = "AAWM_MUSE_CODE_FACADE_ENABLED"
AAWM_MUSE_CODE_MODEL_IDS_ENV = "AAWM_MUSE_CODE_MODEL_IDS"
MUSE_CODE_GATEWAY_PREFIX = "/muse-code"
MUSE_CODE_ROUTE_FAMILY = "muse_code"
MUSE_CODE_CATALOG_TARGET = "/muse-code/models"
MUSE_CODE_RESPONSES_TARGET = "/responses"
MUSE_CODE_CLIENT_ID_PREFIX = "tbh:"
MUSE_CODE_USER_AGENT_PREFIX = "muse-build"
MUSE_CODE_MIN_OUTPUT_TOKENS = 16
MUSE_CODE_TOOL_CHOICE_AUTO = "auto"
_MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS = 500
_MUSE_CODE_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_MUSE_CODE_SESSION_HEADER_NAMES: tuple[str, ...] = (
    "x-meta-ai-gateway-session-id",
    "x-tbh-session-id",
)
_MUSE_CODE_SUPPORTED_TOOL_ITEM_TYPES = frozenset({"function"})
_MUSE_CODE_SUPPORTED_TOOL_CHOICE_VALUES = frozenset({MUSE_CODE_TOOL_CHOICE_AUTO})
_MUSE_CODE_TERMINAL_EVENT_TYPES = frozenset(RESPONSES_API_TERMINAL_STREAM_EVENTS) | {
    "response.failed",
    "response.incomplete",
    "response.completed",
}

DEFAULT_MUSE_CODE_MODEL_IDS: tuple[str, ...] = (
    "muse-spark-1.3-contributor",
    "muse-spark-1.3",
    "muse-spark-1.2-contributor",
    "muse-spark-1.2",
)

# Identity mapping for later call routing. Unknown Muse ids stay themselves;
# they are never coerced onto basic/work/OpenAI aliases.
MUSE_CODE_LITELLM_ALIASES: dict[str, str] = {
    model_id: model_id for model_id in DEFAULT_MUSE_CODE_MODEL_IDS
}

# Sanitized MUSE-001 capture of GET /muse-code/models capability metadata.
_MUSE_CODE_CATALOG_BY_ID: dict[str, dict[str, Any]] = {
    "muse-spark-1.3-contributor": {
        "id": "muse-spark-1.3-contributor",
        "object": "model",
        "created": 1788320726,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.3-contributor",
                "family": "avocado",
                "release_date": "2026-09-02",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "description": (
                    "Your content, including inter-session messages, may be used for product improvement."
                ),
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {
                        "reasoningEffort": "xhigh",
                        "selector": {
                            "description": "Use this for deepest analysis and complex fixes."
                        },
                    },
                    "max": {"reasoningEffort": "max"},
                },
            }
        },
    },
    "muse-spark-1.3": {
        "id": "muse-spark-1.3",
        "object": "model",
        "created": 1787927249,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.3",
                "family": "avocado",
                "release_date": "2026-09-02",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {"reasoningEffort": "xhigh"},
                    "max": {"reasoningEffort": "max"},
                },
                "description": None,
            }
        },
    },
    "muse-spark-1.2-contributor": {
        "id": "muse-spark-1.2-contributor",
        "object": "model",
        "created": 1785432904,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.2-contributor",
                "family": "avocado",
                "release_date": "2026-08-05",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "description": (
                    "Your content, including inter-session messages, may be used for product improvement."
                ),
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {
                        "reasoningEffort": "xhigh",
                        "selector": {
                            "description": "Use this for deepest analysis and complex fixes."
                        },
                    },
                },
            }
        },
    },
    "muse-spark-1.2": {
        "id": "muse-spark-1.2",
        "object": "model",
        "created": 1785432700,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.2",
                "family": "avocado",
                "release_date": "2026-08-05",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {
                        "reasoningEffort": "xhigh",
                        "selector": {
                            "description": "Use this for deepest analysis and complex fixes."
                        },
                    },
                },
                "description": None,
            }
        },
    },
}

router = APIRouter(prefix=MUSE_CODE_GATEWAY_PREFIX, tags=["muse code gateway"])


def is_muse_code_facade_enabled() -> bool:
    """Return True only when the alpha Muse Code facade env flag is truthy."""

    raw = os.getenv(AAWM_MUSE_CODE_FACADE_ENABLED_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _MUSE_CODE_TRUTHY_VALUES


def _muse_code_virtual_key(request: Request) -> str:
    """LiteLLM virtual key from Authorization or x-litellm-api-key.

    Dual-key facades send the virtual key in ``x-litellm-api-key``; Muse's
    native probe uses ``Authorization``. Prefer the explicit LiteLLM header
    when both are present. Do not treat either value as a Meta token.
    """

    header_key = request.headers.get("x-litellm-api-key")
    authorization = request.headers.get("Authorization") or request.headers.get(
        "authorization"
    )
    raw = header_key or authorization or ""
    cleaned = raw.strip() if isinstance(raw, str) else ""
    if not cleaned:
        return ""
    if cleaned.lower().startswith("bearer "):
        return cleaned
    return f"Bearer {cleaned}"


def resolve_muse_code_litellm_alias(muse_model_id: str) -> str:
    """Return the LiteLLM alias for a Muse catalog id.

    Captured spark ids map to themselves. Unknown ids are not rewritten onto
    basic/work/OpenAI aliases.
    """

    return MUSE_CODE_LITELLM_ALIASES.get(muse_model_id, muse_model_id)


def _header_value(request: Request, name: str) -> str:
    value = request.headers.get(name) or request.headers.get(name.lower())
    if not isinstance(value, str):
        return ""
    return value.strip()


def _muse_code_user_agent(request: Request) -> str:
    return _header_value(request, "user-agent")


def _muse_code_client_id(request: Request) -> str:
    return _header_value(request, "x-client-id")


def _muse_code_session_id(request: Request) -> str:
    for name in _MUSE_CODE_SESSION_HEADER_NAMES:
        value = _header_value(request, name)
        if value:
            return value
    return ""


def configured_muse_code_model_ids() -> tuple[str, ...]:
    """Return the catalog ids the alpha facade currently advertises."""

    return _configured_muse_code_model_ids()


def is_known_muse_code_model_id(model_id: Any) -> bool:
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    return model_id.strip() in configured_muse_code_model_ids()


def muse_code_session_id_from_request(request: Request) -> str:
    """Session UUID from Muse gateway headers, never from body ``session_id``."""

    return _muse_code_session_id(request)


class MuseCodeResponsesError(Exception):
    """Handled Muse ``POST /responses`` validation error with a native body."""

    def __init__(self, *, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.payload = payload
        message = ""
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        super().__init__(message)


def is_muse_code_responses_request(
    request: Request,
    data: Optional[dict[str, Any]] = None,
) -> bool:
    """True when the facade is on and this ``POST /responses`` is Muse-shaped.

    Detection is facade-gated. Non-Muse / facade-off traffic keeps the stock
    Responses handler. Shape is ``x-client-id`` ``tbh:…``, User-Agent
    ``muse-build…``, or a configured Muse catalog model id.
    """

    if not is_muse_code_facade_enabled():
        return False

    client_id = _muse_code_client_id(request)
    if client_id.startswith(MUSE_CODE_CLIENT_ID_PREFIX):
        return True

    user_agent = _muse_code_user_agent(request)
    if user_agent.lower().startswith(MUSE_CODE_USER_AGENT_PREFIX):
        return True

    model_id = None
    if isinstance(data, dict):
        model_id = data.get("model")
    return is_known_muse_code_model_id(model_id)


def _raise_muse_code_model_api_error(
    *,
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> None:
    raise MuseCodeResponsesError(
        status_code=status_code,
        payload={
            "error": {
                "code": code,
                "message": message,
                "param": param,
                "type": error_type,
            }
        },
    )


def _raise_muse_code_model_not_found() -> None:
    _raise_muse_code_model_api_error(
        status_code=404,
        message="The requested model was not found.",
        code="model_not_found",
    )


def _raise_muse_code_unknown_parameter(param: str) -> None:
    _raise_muse_code_model_api_error(
        status_code=400,
        message=f"unknown parameter `{param}`",
        param=param,
    )


def _raise_muse_code_max_output_tokens() -> None:
    _raise_muse_code_model_api_error(
        status_code=400,
        message="`max_output_tokens` The number must be `>= 16`.",
        param="max_output_tokens",
    )


def _raise_muse_code_tool_choice() -> None:
    _raise_muse_code_model_api_error(
        status_code=400,
        message=(
            'only `"auto"` is supported for `tool_choice`. `"none"`, '
            '`"required"`, and named function choices are not currently supported'
        ),
        param="tool_choice",
    )


def _raise_muse_code_invalid_tools(message: str) -> None:
    _raise_muse_code_model_api_error(
        status_code=400,
        message=message,
        param="tools",
    )


def _raise_muse_code_unknown_previous_response() -> None:
    _raise_muse_code_model_api_error(
        status_code=400,
        message="The previous_response_id was not found.",
        param="previous_response_id",
    )


def _flatten_muse_code_function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(tool)
    flattened["type"] = "function"
    name = flattened.get("name")
    if not isinstance(name, str) or not name.strip():
        _raise_muse_code_invalid_tools("Each function tool must include a name.")
    flattened["name"] = name.strip()
    flattened.pop("namespace", None)
    return flattened


def flatten_muse_code_tools(tools: Any) -> list[Any]:
    """Flatten Muse ``type=namespace`` wrappers into function tools.

    Names such as ``bash`` / ``read_file`` are preserved. Unsupported item
    types are rejected. Tools are advertised only; LiteLLM does not execute
    client-local shell, filesystem, or MCP tools.
    """

    if tools is None:
        return []
    if not isinstance(tools, list):
        _raise_muse_code_invalid_tools("`tools` must be an array.")

    flattened: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            _raise_muse_code_invalid_tools("Each tool item must be an object.")
        tool_type = tool.get("type")
        if tool_type == "namespace":
            children = tool.get("tools")
            if not isinstance(children, list):
                _raise_muse_code_invalid_tools(
                    "Namespace tools must include a `tools` array."
                )
            for child in children:
                if not isinstance(child, dict):
                    _raise_muse_code_invalid_tools(
                        "Each namespaced tool item must be an object."
                    )
                child_type = child.get("type") or "function"
                if child_type not in _MUSE_CODE_SUPPORTED_TOOL_ITEM_TYPES:
                    _raise_muse_code_invalid_tools(
                        f"Unsupported tool type `{child_type}`."
                    )
                flattened.append(_flatten_muse_code_function_tool(child))
            continue
        if tool_type in _MUSE_CODE_SUPPORTED_TOOL_ITEM_TYPES or (
            tool_type is None and isinstance(tool.get("name"), str)
        ):
            flattened.append(_flatten_muse_code_function_tool(tool))
            continue
        displayed_type = tool_type if isinstance(tool_type, str) else "unknown"
        _raise_muse_code_invalid_tools(
            f"Unsupported tool type `{displayed_type}`."
        )
    return flattened


def _muse_code_tool_choice_is_auto(tool_choice: Any) -> bool:
    if tool_choice is None:
        return True
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in _MUSE_CODE_SUPPORTED_TOOL_CHOICE_VALUES
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if isinstance(choice_type, str):
            return choice_type.strip().lower() in _MUSE_CODE_SUPPORTED_TOOL_CHOICE_VALUES
    return False


def _is_litellm_encoded_response_id(response_id: str) -> bool:
    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)
    original = decoded.get("response_id")
    return bool(original) and original != response_id


def validate_muse_code_previous_response_id(previous_response_id: Any) -> None:
    """Fail closed when continuation cannot bind the same LiteLLM lane.

    Muse retries with ``store: false`` replay the full ``input[]`` and omit
    ``previous_response_id``. When the id is present it must be a LiteLLM
    encoded ``resp_`` value so the existing Responses decoder can recover
    provider/model. Unknown ids do not silently start a new session.
    """

    if previous_response_id is None:
        return
    if not isinstance(previous_response_id, str) or not previous_response_id.strip():
        _raise_muse_code_unknown_previous_response()
    cleaned = previous_response_id.strip()
    if not cleaned.startswith("resp_"):
        _raise_muse_code_unknown_previous_response()
    if not _is_litellm_encoded_response_id(cleaned):
        _raise_muse_code_unknown_previous_response()


def _muse_code_observability_tags(
    *,
    model_id: str,
    session_id: str,
) -> list[str]:
    tags = [
        f"route:{MUSE_CODE_ROUTE_FAMILY}",
        "muse-code",
    ]
    if model_id:
        tags.append(f"muse-code-model:{model_id}")
    if session_id:
        tags.append(f"muse-code-session:{session_id}")
    return tags


def attach_muse_code_responses_observability(
    data: dict[str, Any],
    *,
    request: Request,
    requested_model: str,
    session_id: str,
    prompt_cache_key: Optional[str],
) -> dict[str, Any]:
    """Stamp Muse route family, tags, and session ids onto Responses metadata.

    Session identity is headers plus ``prompt_cache_key``. Body ``session_id``
    is rejected before this runs. Existing Responses session_affinity reads
    ``litellm_metadata.session_id`` when that pre-call check is enabled.
    """

    updated = dict(data)
    litellm_metadata = dict(updated.get("litellm_metadata") or {})
    existing_tags = litellm_metadata.get("tags") or []
    if not isinstance(existing_tags, list):
        existing_tags = []
    merged_tags = list(existing_tags)
    for tag in _muse_code_observability_tags(
        model_id=requested_model, session_id=session_id
    ):
        if tag not in merged_tags:
            merged_tags.append(tag)

    extra_fields: dict[str, Any] = {
        "passthrough_route_family": MUSE_CODE_ROUTE_FAMILY,
        "route_family": MUSE_CODE_ROUTE_FAMILY,
        "client_name": "muse-code",
        "muse_code_facade": True,
        "requested_model": requested_model,
    }
    if session_id:
        extra_fields["session_id"] = session_id
        extra_fields["muse_code_session_id"] = session_id
        updated["litellm_session_id"] = session_id
    client_id = _muse_code_client_id(request)
    if client_id:
        extra_fields["muse_code_client_id"] = client_id
    if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
        extra_fields["prompt_cache_key"] = prompt_cache_key.strip()
        extra_fields["muse_code_prompt_cache_key"] = prompt_cache_key.strip()

    litellm_metadata.update(extra_fields)
    litellm_metadata["tags"] = merged_tags
    updated["litellm_metadata"] = litellm_metadata
    return updated


def prepare_muse_code_responses_request(
    request: Request,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Validate and rewrite a Muse-shaped Responses body for the existing handler."""

    if "session_id" in data:
        _raise_muse_code_unknown_parameter("session_id")

    requested_model = data.get("model")
    if not is_known_muse_code_model_id(requested_model):
        _raise_muse_code_model_not_found()
    assert isinstance(requested_model, str)

    max_output_tokens = data.get("max_output_tokens")
    if max_output_tokens is not None:
        try:
            token_count = int(max_output_tokens)
        except (TypeError, ValueError):
            _raise_muse_code_max_output_tokens()
        if token_count < MUSE_CODE_MIN_OUTPUT_TOKENS:
            _raise_muse_code_max_output_tokens()

    if not _muse_code_tool_choice_is_auto(data.get("tool_choice")):
        _raise_muse_code_tool_choice()

    validate_muse_code_previous_response_id(data.get("previous_response_id"))

    updated = dict(data)
    if "tools" in updated:
        updated["tools"] = flatten_muse_code_tools(updated.get("tools"))
    if "tool_choice" in updated:
        updated["tool_choice"] = MUSE_CODE_TOOL_CHOICE_AUTO
    updated["model"] = resolve_muse_code_litellm_alias(requested_model)

    session_id = _muse_code_session_id(request)
    prompt_cache_key = updated.get("prompt_cache_key")
    updated = attach_muse_code_responses_observability(
        updated,
        request=request,
        requested_model=requested_model,
        session_id=session_id,
        prompt_cache_key=prompt_cache_key if isinstance(prompt_cache_key, str) else None,
    )
    return updated


def _build_muse_code_responses_route_state(
    *,
    model_id: str,
    session_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    request_payload: dict[str, object] = {"model": model_id}
    metadata: dict[str, object] = {
        "custom_llm_provider": "muse_code",
        "route_family": MUSE_CODE_ROUTE_FAMILY,
        "passthrough_route_family": MUSE_CODE_ROUTE_FAMILY,
    }
    if session_id:
        metadata["session_id"] = session_id
    kwargs: dict[str, object] = {"litellm_params": {"metadata": metadata}}
    return request_payload, kwargs


def emit_muse_code_responses_route_context(
    *,
    request: Request,
    model_id: str,
    session_id: str,
    completed: bool = False,
) -> dict[str, object]:
    request_payload, route_kwargs = _build_muse_code_responses_route_state(
        model_id=model_id,
        session_id=session_id,
    )
    emit_aawm_route_access_log(
        request=request,
        target=MUSE_CODE_RESPONSES_TARGET,
        request_body=request_payload,
        kwargs=route_kwargs,
        route_type="MUSE",
        completed=completed,
    )
    return route_kwargs


def apply_muse_code_responses_ingress(
    request: Request,
    data: dict[str, Any],
) -> tuple[bool, dict[str, Any], Optional[dict[str, object]], Optional[JSONResponse]]:
    """Prepare a Muse-shaped Responses request or leave the stock body unchanged.

    Returns ``(is_muse, data, route_kwargs, early_response)``. When
    ``early_response`` is set, the handler should return it immediately.
    """

    if not is_muse_code_responses_request(request, data):
        return False, data, None, None

    requested_model = data.get("model") if isinstance(data.get("model"), str) else ""
    session_id = muse_code_session_id_from_request(request)
    try:
        prepared = prepare_muse_code_responses_request(request, data)
    except MuseCodeResponsesError as exc:
        log_muse_code_responses_failure(
            request=request,
            model_id=requested_model or "muse-code/unknown",
            session_id=session_id,
            status_code=exc.status_code,
            detail=str(exc),
            failure_kind="muse_code_responses_rejected",
        )
        return (
            True,
            data,
            None,
            JSONResponse(status_code=exc.status_code, content=exc.payload),
        )

    route_kwargs = emit_muse_code_responses_route_context(
        request=request,
        model_id=requested_model or str(prepared.get("model") or ""),
        session_id=session_id,
    )
    return True, prepared, route_kwargs, None


def log_muse_code_responses_failure(
    *,
    request: Request,
    model_id: str,
    session_id: str,
    status_code: int,
    detail: object,
    failure_kind: str,
) -> None:
    route_kwargs = emit_muse_code_responses_route_context(
        request=request,
        model_id=model_id or "muse-code/unknown",
        session_id=session_id,
    )
    request_payload = {"model": model_id or "muse-code/unknown"}
    summary = _sanitize_error_summary(detail, status_code=status_code)
    record_aawm_route_rollup_failure(route_kwargs, message=summary)
    log_fn = (
        verbose_proxy_logger.warning
        if status_code < 500
        else verbose_proxy_logger.error
    )
    log_fn(
        "Muse Code responses surfaced handled client/provider error status=%s error=%s",
        status_code,
        summary,
        extra={
            "source": "muse_code_gateway",
            "container": os.getenv("HOSTNAME"),
            "endpoint": request.url.path,
            "provider": "muse_code",
            "model": request_payload["model"],
            "model_alias": None,
            "route_family": MUSE_CODE_ROUTE_FAMILY,
            "status_code": status_code,
            "session_id": session_id or None,
            "litellm_call_id": request.headers.get("x-litellm-call-id"),
            "failure_kind": failure_kind,
        },
        exc_info=False,
    )


def _event_type_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _chunk_to_event_dict(chunk: Any) -> Optional[dict[str, Any]]:
    if chunk is None:
        return None
    if isinstance(chunk, (bytes, bytearray)):
        try:
            chunk = chunk.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(chunk, str):
        text = chunk.strip()
        if not text or text == STREAM_SSE_DONE_STRING:
            return None
        payload = text
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                payload = stripped[len("data:") :].strip()
                break
        if not payload or payload == STREAM_SSE_DONE_STRING:
            return None
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(chunk, dict):
        return dict(chunk)
    if isinstance(chunk, BaseModel):
        dumped = chunk.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        return dumped if isinstance(dumped, dict) else None
    if hasattr(chunk, "model_dump"):
        try:
            dumped = chunk.model_dump(
                mode="json", exclude_none=True, exclude_unset=True
            )
        except Exception:
            return None
        return dumped if isinstance(dumped, dict) else None
    return None


def _serialize_muse_code_sse_event(event: dict[str, Any]) -> str:
    event_type = _event_type_value(event.get("type"))
    payload = json.dumps(event, ensure_ascii=False, default=str)
    if event_type:
        return f"event: {event_type}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


def _stamp_sequence_number(event: dict[str, Any], next_sequence: list[int]) -> dict[str, Any]:
    stamped = dict(event)
    existing = stamped.get("sequence_number")
    if isinstance(existing, int) and existing >= next_sequence[0]:
        next_sequence[0] = existing + 1
        return stamped
    stamped["sequence_number"] = next_sequence[0]
    next_sequence[0] += 1
    return stamped


def _response_snapshot_from_event(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    response = event.get("response")
    if isinstance(response, dict):
        return dict(response)
    return None


def _inject_in_progress_event(
    created_event: dict[str, Any], next_sequence: list[int]
) -> dict[str, Any]:
    snapshot = _response_snapshot_from_event(created_event) or {}
    snapshot["status"] = snapshot.get("status") or "in_progress"
    return _stamp_sequence_number(
        {
            "type": "response.in_progress",
            "response": snapshot,
        },
        next_sequence,
    )


def _inject_terminal_event(
    *,
    last_event: Optional[dict[str, Any]],
    next_sequence: list[int],
    status: str = "incomplete",
) -> dict[str, Any]:
    snapshot = _response_snapshot_from_event(last_event or {}) or {}
    snapshot.setdefault("object", "response")
    snapshot["status"] = status
    snapshot.setdefault("output", [])
    event_type = (
        "response.incomplete" if status == "incomplete" else "response.completed"
    )
    return _stamp_sequence_number(
        {
            "type": event_type,
            "response": snapshot,
        },
        next_sequence,
    )


def _done_sse_frame() -> str:
    return f"data: {STREAM_SSE_DONE_STRING}\n\n"


class _MuseCodeSseState:
    """Track Muse-required SSE frames while wrapping the stock generator."""

    def __init__(self) -> None:
        self.next_sequence = [0]
        self.saw_in_progress = False
        self.saw_terminal = False
        self.pending_in_progress_from: Optional[dict[str, Any]] = None
        self.last_event: Optional[dict[str, Any]] = None
        self.emitted_done = False

    def flush_pending_in_progress(self) -> Optional[dict[str, Any]]:
        if self.pending_in_progress_from is None or self.saw_in_progress:
            self.pending_in_progress_from = None
            return None
        in_progress = _inject_in_progress_event(
            self.pending_in_progress_from, self.next_sequence
        )
        self.saw_in_progress = True
        self.pending_in_progress_from = None
        self.last_event = in_progress
        return in_progress

    def stamp(self, event: dict[str, Any]) -> dict[str, Any]:
        stamped = _stamp_sequence_number(event, self.next_sequence)
        self.last_event = stamped
        return stamped

    def note_event_type(self, event_type: Optional[str]) -> Optional[dict[str, Any]]:
        if event_type == "response.in_progress":
            self.saw_in_progress = True
            self.pending_in_progress_from = None
            return None
        if event_type in _MUSE_CODE_TERMINAL_EVENT_TYPES:
            self.saw_terminal = True
            return self.flush_pending_in_progress()
        if self.pending_in_progress_from is not None:
            return self.flush_pending_in_progress()
        return None

    def terminal_if_needed(self, *, status: str = "incomplete") -> Optional[dict[str, Any]]:
        if self.saw_terminal:
            return None
        terminal = _inject_terminal_event(
            last_event=self.last_event,
            next_sequence=self.next_sequence,
            status=status,
        )
        self.saw_terminal = True
        self.last_event = terminal
        return terminal

    def done_if_needed(self) -> Optional[str]:
        if self.emitted_done:
            return None
        self.emitted_done = True
        return _done_sse_frame()


def _iter_required_closeout_frames(
    state: _MuseCodeSseState, *, status: str = "incomplete"
) -> list[str]:
    frames: list[str] = []
    flushed = state.flush_pending_in_progress()
    if flushed is not None:
        frames.append(_serialize_muse_code_sse_event(flushed))
    terminal = state.terminal_if_needed(status=status)
    if terminal is not None:
        frames.append(_serialize_muse_code_sse_event(terminal))
    done = state.done_if_needed()
    if done is not None:
        frames.append(done)
    return frames


async def muse_code_responses_data_generator(
    response: Any,
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict,
    *,
    inner_generator: Callable[..., AsyncIterator[str]],
) -> AsyncIterator[str]:
    """Wrap the existing Responses SSE generator with Muse-required frames.

    Adds ``event:`` lines, ``sequence_number``, a missing
    ``response.in_progress`` after ``response.created``, a terminal
    ``response.completed`` / ``response.incomplete`` if the inner stream
    ends without one, and a final ``data: [DONE]`` with no event line.
    """

    state = _MuseCodeSseState()
    try:
        async for raw_chunk in inner_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        ):
            event = _chunk_to_event_dict(raw_chunk)
            if event is None:
                if isinstance(raw_chunk, str) and STREAM_SSE_DONE_STRING in raw_chunk:
                    for frame in _iter_required_closeout_frames(state):
                        yield frame
                    continue
                if isinstance(raw_chunk, str):
                    yield raw_chunk if raw_chunk.endswith("\n\n") else f"{raw_chunk}\n\n"
                continue

            event_type = _event_type_value(event.get("type"))
            pending = state.note_event_type(event_type)
            if pending is not None:
                yield _serialize_muse_code_sse_event(pending)
            stamped = state.stamp(event)
            if event_type == "response.created" and not state.saw_in_progress:
                state.pending_in_progress_from = stamped
            yield _serialize_muse_code_sse_event(stamped)

        for frame in _iter_required_closeout_frames(state):
            yield frame
    except Exception:
        for frame in _iter_required_closeout_frames(state, status="incomplete"):
            yield frame
        raise


def _configured_muse_code_model_ids() -> tuple[str, ...]:
    raw = os.getenv(AAWM_MUSE_CODE_MODEL_IDS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MUSE_CODE_MODEL_IDS

    seen: set[str] = set()
    ordered: list[str] = []
    for part in raw.split(","):
        model_id = part.strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)
    if not ordered:
        return DEFAULT_MUSE_CODE_MODEL_IDS
    return tuple(ordered)


def _catalog_entry_for_id(model_id: str) -> dict[str, Any]:
    captured = _MUSE_CODE_CATALOG_BY_ID.get(model_id)
    if captured is not None:
        return copy.deepcopy(captured)
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": model_id,
            }
        },
    }


def _build_muse_code_catalog() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            _catalog_entry_for_id(model_id)
            for model_id in _configured_muse_code_model_ids()
        ],
    }


def _build_route_state() -> tuple[dict[str, object], dict[str, object]]:
    request_payload: dict[str, object] = {"model": "muse-code/catalog"}
    kwargs: dict[str, object] = {
        "litellm_params": {
            "metadata": {
                "custom_llm_provider": "muse_code",
                "route_family": MUSE_CODE_ROUTE_FAMILY,
            }
        }
    }
    return request_payload, kwargs


def _emit_route_context(
    *,
    request: Request,
    request_payload: dict[str, object],
    kwargs: dict[str, object],
) -> None:
    emit_aawm_route_access_log(
        request=request,
        target=MUSE_CODE_CATALOG_TARGET,
        request_body=request_payload,
        kwargs=kwargs,
        route_type="MUSE",
    )


def _sanitize_error_summary(detail: object, *, status_code: int) -> str:
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="replace")
    elif isinstance(detail, str):
        detail_text = detail
    else:
        detail_text = ""
    normalized_summary = " ".join(detail_text.split())
    redacted_summary = _redact_string(normalized_summary)
    if not redacted_summary:
        return f"HTTP {status_code} request rejected"
    if len(redacted_summary) > _MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS:
        return redacted_summary[: _MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS - 3] + "..."
    return redacted_summary


def _proxy_exception_status_code(exc: ProxyException) -> int:
    try:
        return int(exc.code) if exc.code is not None else 401
    except (TypeError, ValueError):
        return 401


def _log_catalog_failure(
    *,
    request: Request,
    request_payload: dict[str, object],
    kwargs: dict[str, object],
    status_code: int,
    detail: object,
    failure_kind: str,
) -> None:
    _emit_route_context(
        request=request,
        request_payload=request_payload,
        kwargs=kwargs,
    )
    summary = _sanitize_error_summary(detail, status_code=status_code)
    record_aawm_route_rollup_failure(kwargs, message=summary)
    log_fn = (
        verbose_proxy_logger.warning
        if status_code < 500
        else verbose_proxy_logger.error
    )
    log_fn(
        "Muse Code catalog surfaced handled client/provider error status=%s error=%s",
        status_code,
        summary,
        extra={
            "source": "muse_code_gateway",
            "container": os.getenv("HOSTNAME"),
            "endpoint": request.url.path,
            "provider": "muse_code",
            "model": request_payload["model"],
            "model_alias": None,
            "route_family": MUSE_CODE_ROUTE_FAMILY,
            "status_code": status_code,
            "litellm_call_id": request.headers.get("x-litellm-call-id"),
            "failure_kind": failure_kind,
        },
        exc_info=False,
    )


def _raise_disabled_not_found() -> None:
    raise HTTPException(status_code=404, detail="Not Found")


async def _authenticate_muse_code_catalog(request: Request) -> UserAPIKeyAuth:
    if not is_muse_code_facade_enabled():
        _raise_disabled_not_found()

    register_aawm_route_rollup_access_log_replacement(request)
    request_payload, route_kwargs = _build_route_state()
    try:
        return await user_api_key_auth(
            request=request,
            api_key=_muse_code_virtual_key(request),
        )
    except HTTPException as exc:
        _log_catalog_failure(
            request=request,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=exc.status_code,
            detail=exc.detail,
            failure_kind="muse_code_authentication_rejected",
        )
        raise
    except ProxyException as exc:
        _log_catalog_failure(
            request=request,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=_proxy_exception_status_code(exc),
            detail=exc.message,
            failure_kind="muse_code_authentication_rejected",
        )
        raise


@router.get("/models")
async def get_muse_code_models(
    request: Request,
    _user_api_key_dict: UserAPIKeyAuth = Depends(_authenticate_muse_code_catalog),
) -> JSONResponse:
    request_payload, route_kwargs = _build_route_state()
    _emit_route_context(
        request=request,
        request_payload=request_payload,
        kwargs=route_kwargs,
    )
    record_aawm_route_rollup_turn(route_kwargs)
    return JSONResponse(content=_build_muse_code_catalog())
