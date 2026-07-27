"""
What is this?

Provider-specific Pass-Through Endpoints

Use litellm with Anthropic SDK, Vertex AI SDK, Cohere SDK, etc.
"""

import ast
import asyncio
import codecs  # noqa: F401 - compatibility binding for extracted Wave 6A facades
import copy
import hashlib
import json
import os
import random  # noqa: F401 - compatibility binding for extracted Wave 5C facades
import re
import time
from datetime import datetime, timedelta, timezone
from inspect import isawaitable  # noqa: F401 - Wave 6A facade host binding
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Never,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    WebSocket,
    status as fastapi_status,
)
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState
from typing_extensions import TypeGuard

globals()["status"] = fastapi_status

import litellm
from litellm import get_llm_provider
from litellm._logging import verbose_proxy_logger
from uuid import NAMESPACE_URL, uuid5
from litellm._uuid import uuid4
from litellm.constants import (
    ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS,
    BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES,
    XAI_API_BASE,
)
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,  # noqa: F401 - Wave 6A host binding
    is_malformed_grok_literal_tool_label_transcript_text,  # noqa: F401 - Wave 6A host binding
)
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_shape,
)
from litellm.proxy.aawm_runtime_error_logging import (
    schedule_persist_malformed_tool_call_detection,  # noqa: F401 - Wave 6A host binding
)
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_API_BASE,
)
from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key
from litellm.types.llms.anthropic import ANTHROPIC_OAUTH_BETA_HEADER
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.llms.xai.oauth import (
    is_oa_xai_model,
    get_grok_native_oauth_access_token,  # noqa: F401  # compatibility host-global
    normalize_grok_native_oauth_model,
    resolve_oa_xai_upstream_model,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._types import *
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    _safe_set_request_parsed_body,
    get_form_data,
    get_request_body,
)
from litellm.proxy.pass_through_endpoints.common_utils import get_litellm_virtual_key
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS,  # noqa: F401  # consumed by rebound env_policy functions
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES,
    create_pass_through_route,
    create_websocket_passthrough_route,
    pass_through_request,
    websocket_passthrough_request,
    _classify_passthrough_hidden_retry_failure,
    _get_passthrough_handled_http_error_summary,
    _get_passthrough_hidden_retry_wait_seconds,  # noqa: F401  # consumed by rebound env_policy functions
    _is_known_grok_build_usage_balance_exhausted_response,
    _is_known_grok_personal_team_spending_limit_response,
    _record_passthrough_hidden_retry_metadata,
)
from litellm.proxy.pass_through_endpoints.google_code_assist_quota import (
    sanitize_google_code_assist_quota_for_logging as _sanitize_google_code_assist_quota_for_logging,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    adapter as _anthropic_grok_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    composer_repair as _anthropic_grok_composer_repair,  # noqa: F401 - Wave 6A host binding
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    normalization as _anthropic_grok_normalization,
)
from litellm.llms.anthropic.experimental_pass_through.providers import (
    common as _anthropic_provider_common,
)
from litellm.llms.anthropic.experimental_pass_through.providers.antigravity import (
    adapter as _anthropic_antigravity_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache as _anthropic_google_process_cache,
)
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    shaping as _anthropic_google_shaping,
)
from litellm.llms.anthropic.experimental_pass_through.providers.nvidia import (
    adapter as _anthropic_nvidia_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.openai import (
    adapter as _anthropic_openai_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    adapter as _anthropic_opencode_zen_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers import (
    common as _anthropic_providers_common,
)
from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import (
    normalization as _anthropic_opencode_zen_normalization,
)
from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    adapter as _anthropic_openrouter_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    retry_transport as _anthropic_openrouter_retry_transport,
)
from litellm.llms.anthropic.experimental_pass_through.providers.xai import (
    adapter as _anthropic_xai_provider,
)
from litellm.proxy.aawm_route_logging import (
    aresolve_aawm_route_host_attribution,
    build_aawm_route_rollup_group_header_label,
    emit_aawm_route_access_log,
    emit_aawm_route_status_event,
    record_aawm_route_rollup,
    record_aawm_route_rollup_turn,
    resolve_aawm_route_host_attribution,
)

try:
    from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
        add_claude_post_rewrite_context_file_logging_metadata as _aawm_add_claude_post_rewrite_context_file_logging_metadata,
    )
    from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
        apply_claude_control_plane_rewrites_to_anthropic_request_body as _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body,
    )
    from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (
        expand_aawm_dynamic_directives_in_anthropic_request_body as _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body,
    )
except ImportError:

    def _aawm_add_claude_post_rewrite_context_file_logging_metadata(
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        return request_body

    async def _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
        request_body: dict[str, Any],
        billing_header_fields: dict[str, str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        return request_body, [], []

    async def _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body(
        request_body: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return request_body, []


# RR-054 #7: single process-wide AAWM dynamic-injection pool ownership lives in
# aawm_claude_control_plane. Re-export stable helpers so OpenRouter quota and
# tests keep the historical import surface without opening a second pool.
from litellm.proxy.pass_through_endpoints.aawm_claude_control_plane import (  # noqa: F401
    _build_aawm_dynamic_injection_dsn,
    _call_aawm_get_agent_memories,
    _get_aawm_dynamic_injection_application_name,
    _get_aawm_dynamic_injection_pool,
    _get_aawm_dynamic_injection_server_settings,
    _initialize_aawm_dynamic_injection_connection,
    close_aawm_dynamic_injection_pool,
)
from litellm.proxy.utils import is_known_model
from litellm.proxy.vector_store_endpoints.utils import (
    is_allowed_to_call_vector_store_endpoint,
)
from litellm.secret_managers.main import get_secret_str
from litellm.llms.alibaba_token_plan.adapters import (
    adapter as _alibaba_token_plan_adapters,
)
from litellm.llms.kimi_code.adapters import adapter as _kimi_code_adapters
from litellm.types.llms.openai import (
    RESPONSES_API_TERMINAL_STREAM_EVENTS,  # noqa: F401 - Wave 6A host binding
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from .passthrough_endpoint_router import PassthroughEndpointRouter
from .aawm_alias_routing_policy import (
    ANTHROPIC_AAWM_CODE_ALIAS as _POLICY_ANTHROPIC_AAWM_CODE_ALIAS,
    ANTHROPIC_AAWM_CODE_CANDIDATES as _POLICY_ANTHROPIC_AAWM_CODE_CANDIDATES,
    ANTHROPIC_AAWM_LOW_ALIAS as _POLICY_ANTHROPIC_AAWM_LOW_ALIAS,
    ANTHROPIC_AAWM_LOW_CANDIDATES as _POLICY_ANTHROPIC_AAWM_LOW_CANDIDATES,
    ANTHROPIC_AAWM_ORCHESTRATION_ALIAS as _POLICY_ANTHROPIC_AAWM_ORCHESTRATION_ALIAS,
    ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES as _POLICY_ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES,
    ANTHROPIC_AAWM_READ_ALIAS as _POLICY_ANTHROPIC_AAWM_READ_ALIAS,
    ANTHROPIC_AAWM_SOTA_ALIAS as _POLICY_ANTHROPIC_AAWM_SOTA_ALIAS,
    ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES as _POLICY_ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES,
    ANTHROPIC_AAWM_SOTA_CANDIDATES as _POLICY_ANTHROPIC_AAWM_SOTA_CANDIDATES,
    ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES as _POLICY_ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES,
    ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES as _POLICY_ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES,
    ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES as _POLICY_ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES,
    ANTHROPIC_AUTO_AGENT_CANDIDATES as _POLICY_ANTHROPIC_AUTO_AGENT_CANDIDATES,
    ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS as _POLICY_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS,
    ANTHROPIC_AUTO_AGENT_HAIKU_MODEL as _POLICY_ANTHROPIC_AUTO_AGENT_HAIKU_MODEL,
    ANTHROPIC_AUTO_AGENT_MODEL_ALIAS as _POLICY_ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER as _POLICY_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
    ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES as _POLICY_ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES,
    ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS as _POLICY_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS,
    ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS as _POLICY_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS,
    ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS as _POLICY_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS,
    ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS as _POLICY_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS,
    ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS as _POLICY_ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS,
    ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER as _POLICY_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
    ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS as _POLICY_ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS,
    CODEX_AAWM_CODE_ALIAS as _POLICY_CODEX_AAWM_CODE_ALIAS,
    CODEX_AAWM_CODE_CANDIDATES as _POLICY_CODEX_AAWM_CODE_CANDIDATES,
    CODEX_AAWM_LOW_ALIAS as _POLICY_CODEX_AAWM_LOW_ALIAS,
    CODEX_AAWM_LOW_CANDIDATES as _POLICY_CODEX_AAWM_LOW_CANDIDATES,
    CODEX_AAWM_ORCHESTRATION_ALIAS as _POLICY_CODEX_AAWM_ORCHESTRATION_ALIAS,
    CODEX_AAWM_ORCHESTRATION_CANDIDATES as _POLICY_CODEX_AAWM_ORCHESTRATION_CANDIDATES,
    CODEX_AAWM_READ_ALIAS as _POLICY_CODEX_AAWM_READ_ALIAS,
    CODEX_AAWM_SOTA_ALIAS as _POLICY_CODEX_AAWM_SOTA_ALIAS,
    CODEX_AAWM_SOTA_ALIBABA_ALIAS as _POLICY_CODEX_AAWM_SOTA_ALIBABA_ALIAS,
    CODEX_AAWM_SOTA_ALIBABA_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_ALIBABA_CANDIDATES,
    CODEX_AAWM_SOTA_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_CANDIDATES,
    CODEX_AAWM_SOTA_DEEPSEEK_ALIAS as _POLICY_CODEX_AAWM_SOTA_DEEPSEEK_ALIAS,
    CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES,
    CODEX_AAWM_SOTA_GLM_ALIAS as _POLICY_CODEX_AAWM_SOTA_GLM_ALIAS,
    CODEX_AAWM_SOTA_GLM_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_GLM_CANDIDATES,
    CODEX_AAWM_SOTA_OPENAI_ALIAS as _POLICY_CODEX_AAWM_SOTA_OPENAI_ALIAS,
    CODEX_AAWM_SOTA_OPENAI_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_OPENAI_CANDIDATES,
    CODEX_AAWM_SOTA_MOONSHOT_ALIAS as _POLICY_CODEX_AAWM_SOTA_MOONSHOT_ALIAS,
    CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES,
    CODEX_AAWM_SOTA_XAI_ALIAS as _POLICY_CODEX_AAWM_SOTA_XAI_ALIAS,
    CODEX_AAWM_SOTA_XAI_CANDIDATES as _POLICY_CODEX_AAWM_SOTA_XAI_CANDIDATES,
    CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER as _POLICY_CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY as _POLICY_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY,
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER as _POLICY_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    CODEX_AUTO_AGENT_CANDIDATES as _POLICY_CODEX_AUTO_AGENT_CANDIDATES,
    CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS as _POLICY_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS,
    CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS as _POLICY_CAPACITY_COOLDOWN,
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS as _POLICY_DEFAULT_COOLDOWN,
    CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS as _POLICY_RATE_LIMIT_COOLDOWN,
    CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS as _POLICY_TRANSIENT_COOLDOWN,
    CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS as _POLICY_USAGE_LIMIT_COOLDOWN,
    CODEX_AUTO_AGENT_GOOGLE_PROVIDER as _POLICY_CODEX_AUTO_AGENT_GOOGLE_PROVIDER,
    CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY as _POLICY_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
    CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER as _POLICY_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
    CODEX_AUTO_AGENT_MODEL_ALIAS as _POLICY_CODEX_AUTO_AGENT_MODEL_ALIAS,
    CODEX_AUTO_AGENT_NATIVE_PROVIDER as _POLICY_CODEX_AUTO_AGENT_NATIVE_PROVIDER,
    CODEX_AUTO_AGENT_OPENCODE_LANE_KEY as _POLICY_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY,
    CODEX_AUTO_AGENT_OPENCODE_PROVIDER as _POLICY_CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
    CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY as _POLICY_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY,
    CODEX_AUTO_AGENT_OPENROUTER_PROVIDER as _POLICY_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
    CODEX_AUTO_AGENT_XAI_LANE_KEY as _POLICY_CODEX_AUTO_AGENT_XAI_LANE_KEY,
    CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY as _POLICY_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY,
    CODEX_AUTO_AGENT_XAI_PROVIDER as _POLICY_CODEX_AUTO_AGENT_XAI_PROVIDER,
    CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES as _POLICY_CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES,
    KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS as _POLICY_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS,
)

from .aawm_alias_routing import adapter_config as _aawm_adapter_config
from .aawm_alias_routing import adapter_driver as _aawm_adapter_driver
from .aawm_alias_routing import classification as _aawm_alias_classification
from .aawm_alias_routing import memory as _aawm_alias_memory
from .aawm_alias_routing import provider_shaping as _aawm_provider_shaping  # noqa: F401 - runtime globals() binding
from .aawm_alias_routing import responses_finalize as _aawm_responses_finalize
# Compatibility host global for transplanted Google env-policy functions/tests.
from .aawm_alias_routing import retry as _aawm_alias_retry  # noqa: F401
from .aawm_alias_routing import streaming as _aawm_alias_streaming
from .aawm_alias_routing import google_oauth as _aawm_google_oauth
from .aawm_alias_routing import antigravity_oauth as _aawm_antigravity_oauth
from .aawm_alias_routing import candidate_loop as _aawm_alias_candidate_loop
from .aawm_alias_routing import durable as _aawm_alias_durable

# Wave 4 pure-leaf extraction imports
from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import constants as _opencode_zen_constants
from litellm.llms.anthropic.experimental_pass_through.providers.antigravity import constants as _antigravity_constants
from .aawm_alias_routing import lane_keys as _aawm_lane_keys
from .aawm_adapter_runtime import model_resolution as _aawm_adapter_model_resolution
from . import aawm_adapter_runtime as _aawm_adapter_runtime
from .aawm_request_policy import alias_guidance as _aawm_alias_guidance
from .aawm_request_policy import observability_metadata as _aawm_observability_metadata
from .aawm_request_policy import persisted_output as _aawm_persisted_output
from .aawm_request_policy import codex_tool_policy as _aawm_codex_tool_policy
from .aawm_request_policy import claude_prompt_replacement as _aawm_claude_prompt_replacement
from .aawm_request_policy import anthropic_body_prep as _aawm_anthropic_body_prep
from litellm.llms.anthropic.experimental_pass_through.providers.google import env_policy as _google_env_policy
from litellm.llms.anthropic.experimental_pass_through.providers.google import context_window as _google_context_window
from litellm.llms.anthropic.experimental_pass_through.providers.google import error_signals as _google_error_signals
from litellm.llms.anthropic.experimental_pass_through.providers.grok import side_channel as _grok_side_channel

# Wave 6B extracted provider modules
from litellm.proxy.pass_through_endpoints.providers import common as _wave6b_common
from litellm.proxy.pass_through_endpoints.providers.antigravity import runtime as _wave6b_antigravity_runtime
from litellm.proxy.pass_through_endpoints.providers.openrouter import runtime as _wave6b_openrouter_runtime
from litellm.proxy.pass_through_endpoints.providers.nvidia import runtime as _wave6b_nvidia_runtime
from litellm.proxy.pass_through_endpoints.providers.opencode_zen import runtime as _wave6b_opencode_zen_runtime
from litellm.proxy.pass_through_endpoints.providers.xai import request_prep as _wave6b_xai_request_prep
from litellm.proxy.pass_through_endpoints.providers.google import retry_runtime as _google_retry_runtime
from litellm.proxy.pass_through_endpoints.providers.google import codex_code_assist as _google_codex_code_assist

from .aawm_alias_routing import interfaces as _aawm_alias_interfaces
from .aawm_alias_routing.state import alias_routing_state as _alias_routing_state


# ---------------------------------------------------------------------------
# RR-054 early package re-exports / state ownership (must exist before use)
# ---------------------------------------------------------------------------
_AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE = _aawm_alias_memory.DEFAULT_MEMORY_STATE_MAX_SIZE

# RR-054 runtime budgets/constants introduced in residual work.
_AAWM_REQUEST_BODY_WALK_MAX_DEPTH = 64
_AAWM_REQUEST_BODY_WALK_MAX_NODES = 4000
_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS = 5000
_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES = 8 * 1024 * 1024
_AAWM_COOLDOWN_NEGATIVE_CACHE_TTL_SECONDS = 5.0
_CODEX_AUTO_AGENT_GOOGLE_LANE_NEGATIVE_TTL_SECONDS = 15.0
_CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY = "google:auth_degraded"
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_TTL_SECONDS = _google_codex_code_assist._CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_TTL_SECONDS
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE = _google_codex_code_assist._CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE
_codex_google_code_assist_tool_call_name_cache = (
    _anthropic_google_process_cache._codex_google_code_assist_tool_call_name_cache
)
_codex_google_code_assist_tool_call_arguments_cache = (
    _anthropic_google_process_cache._codex_google_code_assist_tool_call_arguments_cache
)
# Default probe-compatible retry set includes 429 + common 5xx.
_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT = [
    429,
    500,
    502,
    503,
    504,
]
_NativeGrokContinuationRetryMetadata = dict[str, Any]
_RetryResultT = TypeVar("_RetryResultT")
_WalkResultT = TypeVar("_WalkResultT")


def _should_log_aawm_alias_routing_event(log_key: str) -> bool:
    now = time.monotonic()
    until = _aawm_alias_routing_log_until_monotonic_by_key.get(log_key, 0.0)
    if now < until:
        return False
    _aawm_alias_routing_log_until_monotonic_by_key[log_key] = now + 30.0
    _bound_aawm_alias_routing_memory_map(_aawm_alias_routing_log_until_monotonic_by_key)
    return True


def _replace_request_body_in_place(
    request_body: Payload,
    updated_body: Payload,
) -> None:
    if updated_body is request_body:
        return
    request_body.clear()
    request_body.update(updated_body)


def _bound_aawm_alias_routing_memory_map(
    cache: dict,
    *,
    max_size: int = _AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE,
) -> None:
    _aawm_alias_memory.bound_memory_map(cache, max_size=max_size)


def _hydrate_aawm_alias_routing_cooldown_memory(
    *,
    memory_map: dict[str, float],
    cooldown_key: str,
    expires_at_epoch: float,
) -> None:
    _aawm_alias_memory.hydrate_cooldown_memory(
        memory_map=memory_map,
        cooldown_key=cooldown_key,
        expires_at_epoch=expires_at_epoch,
        max_size=_AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE,
    )


def _hydrate_aawm_alias_routing_affinity_memory(
    *,
    memory_map: dict[str, dict[str, Any]],
    session_key: str,
    payload: dict[str, Any],
    expires_at_epoch: float,
) -> dict[str, Any]:
    return _aawm_alias_memory.hydrate_affinity_memory(
        memory_map=memory_map,
        session_key=session_key,
        payload=payload,
        expires_at_epoch=expires_at_epoch,
        max_size=_AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE,
    )


class _ChangeAccumulator:
    """RR-054 #8: accumulate transform change dicts without silent merge-order loss."""

    def __init__(self) -> None:
        self._changes: Payload = {}
        self._names: list[str] = []

    def record(self, name: str, changes: Optional[Payload] = None) -> None:
        if not changes:
            return
        self._names.append(name)
        for key, value in changes.items():
            if key in self._changes and self._changes[key] != value:
                alt_key = f"{name}:{key}"
                self._changes[alt_key] = value
            else:
                self._changes[key] = value

    def as_dict(self) -> Payload:
        if self._names:
            self._changes.setdefault("google_adapter_change_steps", list(self._names))
        return dict(self._changes)


# Process-local maps/locks owned by aawm_alias_routing.state.
_codex_auto_agent_cooldown_until_monotonic_by_key = _alias_routing_state.codex.cooldown_until_monotonic_by_key
_codex_auto_agent_session_affinity_by_key = _alias_routing_state.codex.session_affinity_by_key
_codex_auto_agent_lock = _alias_routing_state.codex.lock
_anthropic_auto_agent_cooldown_until_monotonic_by_key = _alias_routing_state.anthropic.cooldown_until_monotonic_by_key
_anthropic_auto_agent_session_affinity_by_key = _alias_routing_state.anthropic.session_affinity_by_key
_anthropic_auto_agent_lock = _alias_routing_state.anthropic.lock
_codex_auto_agent_google_lane_key_until_monotonic_by_key = _alias_routing_state.google_lane_key_until_monotonic_by_key
_codex_auto_agent_google_lane_key_by_key = _alias_routing_state.google_lane_key_by_key
_codex_auto_agent_antigravity_lane_key_until_monotonic_by_key = (
    _alias_routing_state.antigravity_lane_key_until_monotonic_by_key
)
_codex_auto_agent_antigravity_lane_key_by_key = _alias_routing_state.antigravity_lane_key_by_key
_codex_auto_agent_lane_state_cache_lock = _alias_routing_state.lane_state_cache_lock
_codex_auto_agent_cooldown_negative_until_monotonic_by_key = (
    _alias_routing_state.codex.cooldown_negative_until_monotonic_by_key
)
_anthropic_auto_agent_cooldown_negative_until_monotonic_by_key = (
    _alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key
)
_aawm_alias_routing_log_until_monotonic_by_key = _alias_routing_state.log_until_monotonic_by_key

# Policy aliases (RR-054 #11): static tables live in aawm_alias_routing.policy.
_CODEX_AUTO_AGENT_MODEL_ALIAS = _POLICY_CODEX_AUTO_AGENT_MODEL_ALIAS
_CODEX_AUTO_AGENT_NATIVE_PROVIDER = _POLICY_CODEX_AUTO_AGENT_NATIVE_PROVIDER
_CODEX_AUTO_AGENT_GOOGLE_PROVIDER = _POLICY_CODEX_AUTO_AGENT_GOOGLE_PROVIDER
_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER = _POLICY_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER
_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER = _POLICY_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER
_CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER = _POLICY_CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER
_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER = _POLICY_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER
_CODEX_AUTO_AGENT_XAI_PROVIDER = _POLICY_CODEX_AUTO_AGENT_XAI_PROVIDER
_CODEX_AUTO_AGENT_OPENCODE_PROVIDER = _POLICY_CODEX_AUTO_AGENT_OPENCODE_PROVIDER
_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY = _POLICY_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
_CODEX_AUTO_AGENT_XAI_LANE_KEY = _POLICY_CODEX_AUTO_AGENT_XAI_LANE_KEY
_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY = _POLICY_CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY
_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY = _POLICY_CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY
_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY = _POLICY_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_LANE_KEY
_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY = _POLICY_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS = _POLICY_DEFAULT_COOLDOWN
_CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS = _POLICY_CAPACITY_COOLDOWN
_CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = _POLICY_RATE_LIMIT_COOLDOWN
_CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS = _POLICY_USAGE_LIMIT_COOLDOWN
_CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS = _POLICY_TRANSIENT_COOLDOWN
_CODEX_AUTO_AGENT_CANDIDATES = _POLICY_CODEX_AUTO_AGENT_CANDIDATES
_CODEX_AAWM_READ_ALIAS = _POLICY_CODEX_AAWM_READ_ALIAS
_CODEX_AAWM_SOTA_ALIAS = _POLICY_CODEX_AAWM_SOTA_ALIAS
_CODEX_AAWM_CODE_ALIAS = _POLICY_CODEX_AAWM_CODE_ALIAS
_CODEX_AAWM_LOW_ALIAS = _POLICY_CODEX_AAWM_LOW_ALIAS
_CODEX_AAWM_ORCHESTRATION_ALIAS = _POLICY_CODEX_AAWM_ORCHESTRATION_ALIAS
_CODEX_AAWM_SOTA_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_CANDIDATES
_CODEX_AAWM_SOTA_OPENAI_ALIAS = _POLICY_CODEX_AAWM_SOTA_OPENAI_ALIAS
_CODEX_AAWM_SOTA_XAI_ALIAS = _POLICY_CODEX_AAWM_SOTA_XAI_ALIAS
_CODEX_AAWM_SOTA_MOONSHOT_ALIAS = _POLICY_CODEX_AAWM_SOTA_MOONSHOT_ALIAS
_CODEX_AAWM_SOTA_ALIBABA_ALIAS = _POLICY_CODEX_AAWM_SOTA_ALIBABA_ALIAS
_CODEX_AAWM_SOTA_DEEPSEEK_ALIAS = _POLICY_CODEX_AAWM_SOTA_DEEPSEEK_ALIAS
_CODEX_AAWM_SOTA_GLM_ALIAS = _POLICY_CODEX_AAWM_SOTA_GLM_ALIAS
_CODEX_AAWM_SOTA_OPENAI_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_OPENAI_CANDIDATES
_CODEX_AAWM_SOTA_XAI_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_XAI_CANDIDATES
_CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_MOONSHOT_CANDIDATES
_CODEX_AAWM_SOTA_ALIBABA_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_ALIBABA_CANDIDATES
_CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_DEEPSEEK_CANDIDATES
_CODEX_AAWM_SOTA_GLM_CANDIDATES = _POLICY_CODEX_AAWM_SOTA_GLM_CANDIDATES
_CODEX_AAWM_CODE_CANDIDATES = _POLICY_CODEX_AAWM_CODE_CANDIDATES
_CODEX_AAWM_LOW_CANDIDATES = _POLICY_CODEX_AAWM_LOW_CANDIDATES
_CODEX_AAWM_ORCHESTRATION_CANDIDATES = _POLICY_CODEX_AAWM_ORCHESTRATION_CANDIDATES
_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS = _POLICY_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS
_ANTHROPIC_AUTO_AGENT_MODEL_ALIAS = _POLICY_ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = _POLICY_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER
_ANTHROPIC_AUTO_AGENT_HAIKU_MODEL = _POLICY_ANTHROPIC_AUTO_AGENT_HAIKU_MODEL
_ANTHROPIC_AUTO_AGENT_CANDIDATES = _POLICY_ANTHROPIC_AUTO_AGENT_CANDIDATES
_ANTHROPIC_AAWM_READ_ALIAS = _POLICY_ANTHROPIC_AAWM_READ_ALIAS
_ANTHROPIC_AAWM_SOTA_ALIAS = _POLICY_ANTHROPIC_AAWM_SOTA_ALIAS
_ANTHROPIC_AAWM_CODE_ALIAS = _POLICY_ANTHROPIC_AAWM_CODE_ALIAS
_ANTHROPIC_AAWM_LOW_ALIAS = _POLICY_ANTHROPIC_AAWM_LOW_ALIAS
_ANTHROPIC_AAWM_ORCHESTRATION_ALIAS = _POLICY_ANTHROPIC_AAWM_ORCHESTRATION_ALIAS
_ANTHROPIC_AAWM_SOTA_CANDIDATES = _POLICY_ANTHROPIC_AAWM_SOTA_CANDIDATES
_ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES = _POLICY_ANTHROPIC_AAWM_SOTA_MOONSHOT_CANDIDATES
_ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES = _POLICY_ANTHROPIC_AAWM_SOTA_ALIBABA_CANDIDATES
_ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES = _POLICY_ANTHROPIC_AAWM_SOTA_DEEPSEEK_CANDIDATES
_ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES = _POLICY_ANTHROPIC_AAWM_SOTA_GLM_CANDIDATES
_ANTHROPIC_AAWM_CODE_CANDIDATES = _POLICY_ANTHROPIC_AAWM_CODE_CANDIDATES
_ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES = _POLICY_ANTHROPIC_AAWM_ORCHESTRATION_CANDIDATES
_ANTHROPIC_AAWM_LOW_CANDIDATES = _POLICY_ANTHROPIC_AAWM_LOW_CANDIDATES
_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS = _POLICY_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS
_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS = _POLICY_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS
_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS = _POLICY_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS
_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS = _POLICY_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS
_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS = _POLICY_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS
_ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    _POLICY_ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES
)
_CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    _POLICY_CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES
)
_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER = _POLICY_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
_ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS = _POLICY_ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS
_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS = _POLICY_KIMI_CODE_CHAT_COMPLETIONS_ADAPTER_ALLOWED_MODELS
_ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS = _POLICY_ALIBABA_TOKEN_PLAN_ADAPTER_ALLOWED_MODELS


# ---------------------------------------------------------------------------
# Wave 4 (D1-583): config-snapshot-driven candidate resolution for the
# ``read`` pilot alias only. Every other alias continues to resolve from the
# hard-coded ``policy.py`` tables above via ``_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS``
# / ``_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS``. No new session_history /
# routing-decision persistence is introduced here -- selection stays
# in-memory/process-local, matching the existing auto-agent alias lanes.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wave 5A: facade imports from aawm_alias_routing extraction modules.
# Each name below is the SAME object as the target module's definition.
# ---------------------------------------------------------------------------
from .aawm_alias_routing import snapshot_select as _aawm_snapshot_select
from .aawm_alias_routing import config_refresh as _aawm_config_refresh
from .aawm_alias_routing import codex_oauth as _aawm_codex_oauth
from .aawm_alias_routing import openrouter_quota as _aawm_openrouter_quota

# Wave 5B: cooldown_state + selection extraction modules
from .aawm_alias_routing import cooldown_state as _aawm_cooldown_state
from .aawm_alias_routing import selection as _aawm_selection

# Wave 5C: error_signals + cooldown_apply + attempt_records extraction modules
from .aawm_alias_routing import error_signals as _aawm_error_signals
from .aawm_alias_routing import cooldown_apply as _aawm_cooldown_apply
from .aawm_alias_routing import attempt_records as _aawm_attempt_records
# Wave 5D: audit_context + audit_build + audit_persist + audit_events extraction modules
from .aawm_alias_routing import audit_context as _aawm_audit_context
from .aawm_alias_routing import audit_build as _aawm_audit_build
from .aawm_alias_routing import audit_persist as _aawm_audit_persist
from .aawm_alias_routing import audit_events as _aawm_audit_events

# -- snapshot_select facades --
_READ_PILOT_ALIAS_NAME = _aawm_snapshot_select._READ_PILOT_ALIAS_NAME
get_active_routing_snapshot = _aawm_snapshot_select.get_active_routing_snapshot
set_active_routing_snapshot = _aawm_snapshot_select.set_active_routing_snapshot
_routing_candidate_to_public_dict = _aawm_snapshot_select._routing_candidate_to_public_dict
_order_snapshot_candidates_by_priority = _aawm_snapshot_select._order_snapshot_candidates_by_priority
_select_proportional_snapshot_candidate = _aawm_snapshot_select._select_proportional_snapshot_candidate
RoundRobinCommitToken = _aawm_snapshot_select.RoundRobinCommitToken
SelectionEnumeration = _aawm_snapshot_select.SelectionEnumeration
_select_round_robin_snapshot_candidate = _aawm_snapshot_select._select_round_robin_snapshot_candidate
_commit_round_robin_selection = _aawm_snapshot_select._commit_round_robin_selection
_apply_snapshot_alias_distribution_strategy = _aawm_snapshot_select._apply_snapshot_alias_distribution_strategy
_is_tui_attached_candidate_eligible = _aawm_snapshot_select._is_tui_attached_candidate_eligible
_is_snapshot_candidate_in_schedule_window = _aawm_snapshot_select._is_snapshot_candidate_in_schedule_window
_resolve_read_pilot_eligible_candidates = _aawm_snapshot_select._resolve_read_pilot_eligible_candidates
_select_read_pilot_snapshot_candidates = _aawm_snapshot_select._select_read_pilot_snapshot_candidates
_derive_round_robin_commit_token = _aawm_snapshot_select._derive_round_robin_commit_token
_get_aawm_alias_selection_context = _aawm_snapshot_select._get_aawm_alias_selection_context
_resolve_aawm_alias_selection_enumeration = _aawm_snapshot_select._resolve_aawm_alias_selection_enumeration
_get_codex_auto_agent_candidates_for_alias = _aawm_snapshot_select._get_codex_auto_agent_candidates_for_alias

# -- config_refresh facades --
_DEFAULT_AAWM_ALIAS_CONFIG_PATH = _aawm_config_refresh._DEFAULT_AAWM_ALIAS_CONFIG_PATH
_load_aawm_alias_routing_source_yaml = _aawm_config_refresh._load_aawm_alias_routing_source_yaml
aawm_alias_config_refresh_route = _aawm_config_refresh.aawm_alias_config_refresh_route

# -- codex_oauth facades --
_ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS = _aawm_codex_oauth._ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS
_ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS = _aawm_codex_oauth._ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS
_ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS = _aawm_codex_oauth._ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS
CodexAuthData = _aawm_codex_oauth.CodexAuthData
CodexTokenData = _aawm_codex_oauth.CodexTokenData
OAuthJsonData = _aawm_codex_oauth.OAuthJsonData
_clean_codex_auth_value = _aawm_codex_oauth._clean_codex_auth_value
_get_anthropic_adapter_codex_auth_file_path = _aawm_codex_oauth._get_anthropic_adapter_codex_auth_file_path
_decode_jwt_claims_without_validation = _aawm_codex_oauth._decode_jwt_claims_without_validation
_extract_codex_account_id_from_token = _aawm_codex_oauth._extract_codex_account_id_from_token
_get_codex_auth_token_data = _aawm_codex_oauth._get_codex_auth_token_data
_get_codex_auth_token_expiry = _aawm_codex_oauth._get_codex_auth_token_expiry
_codex_auth_access_token_is_valid = _aawm_codex_oauth._codex_auth_access_token_is_valid
_load_codex_auth_data_from_path = _aawm_codex_oauth._load_codex_auth_data_from_path
_load_local_codex_auth_headers = _aawm_codex_oauth._load_local_codex_auth_headers
_anthropic_adapter_request_uses_codex_native_auth = _aawm_codex_oauth._anthropic_adapter_request_uses_codex_native_auth
_anthropic_adapter_request_has_openai_client_auth = _aawm_codex_oauth._anthropic_adapter_request_has_openai_client_auth
_anthropic_adapter_should_forward_direct_auth_headers = _aawm_codex_oauth._anthropic_adapter_should_forward_direct_auth_headers
_request_uses_codex_native_auth = _aawm_codex_oauth._request_uses_codex_native_auth
_get_oauth_token_error_code = _aawm_codex_oauth._get_oauth_token_error_code
_format_oauth_refresh_failure_detail = _aawm_codex_oauth._format_oauth_refresh_failure_detail

# -- openrouter_quota facades --
_OPENROUTER_DURABLE_QUOTA_DAILY_KEY = _aawm_openrouter_quota._OPENROUTER_DURABLE_QUOTA_DAILY_KEY
_OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS = _aawm_openrouter_quota._OPENROUTER_DURABLE_QUOTA_CACHE_TTL_SECONDS
_OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS = _aawm_openrouter_quota._OPENROUTER_DURABLE_QUOTA_LOOKUP_TIMEOUT_SECONDS
_OPENROUTER_FREE_DAILY_QUOTA_MODELS = _aawm_openrouter_quota._OPENROUTER_FREE_DAILY_QUOTA_MODELS
_raise_openrouter_auto_agent_candidate_unavailable = _aawm_openrouter_quota._raise_openrouter_auto_agent_candidate_unavailable
_maybe_raise_openrouter_adapter_alias_probe_cooldown = _aawm_openrouter_quota._maybe_raise_openrouter_adapter_alias_probe_cooldown
_reset_openrouter_free_daily_quota_cache = _aawm_openrouter_quota._reset_openrouter_free_daily_quota_cache
_parse_openrouter_free_daily_quota_reset_timestamp = _aawm_openrouter_quota._parse_openrouter_free_daily_quota_reset_timestamp
_fetch_openrouter_free_daily_quota_row = _aawm_openrouter_quota._fetch_openrouter_free_daily_quota_row
_get_openrouter_free_daily_quota_exhausted_cooldown_seconds = _aawm_openrouter_quota._get_openrouter_free_daily_quota_exhausted_cooldown_seconds
_is_openrouter_free_quota_candidate = _aawm_openrouter_quota._is_openrouter_free_quota_candidate
_apply_openrouter_durable_quota_candidate_cooldown = _aawm_openrouter_quota._apply_openrouter_durable_quota_candidate_cooldown

# Wave 5B: gate, cursor, and quota cache are now manager-owned.
_read_pilot_cooldown_gate = _alias_routing_state.read_pilot_gate
_round_robin_cursor_by_alias = _alias_routing_state.round_robin_cursor

# Wave 5A/5B: bind the round-robin cursor into snapshot_select.
_aawm_snapshot_select.configure_snapshot_runtime(
    round_robin_cursor=_round_robin_cursor_by_alias,
    get_candidates_for_alias=lambda *a, **kw: globals()["_get_codex_auto_agent_candidates_for_alias"](*a, **kw),
)


def reset_module_singletons() -> None:
    """Clear legacy god-module singleton state (test-support).

    Wave 5B moved the read-pilot gate and round-robin cursor onto
    ``AliasRoutingStateManager``. Preserve this helper's historical narrow
    behavior by clearing only those manager-owned surfaces plus the active
    routing snapshot.
    """
    _read_pilot_cooldown_gate._key_state.clear()
    _read_pilot_cooldown_gate._family_state.evidence_events_by_key.clear()
    _round_robin_cursor_by_alias.clear()
    set_active_routing_snapshot(None)


def reset_alias_routing_state_for_tests() -> None:
    """Reset ALL process-local alias-routing state (test-support only).

    Clears manager-owned state (including gate/cursor/quota since Wave 5B)
    via ``alias_routing_state.reset_for_tests()`` and the snapshot via
    ``reset_module_singletons()``.
    """
    _alias_routing_state.reset_for_tests()
    set_active_routing_snapshot(None)


def _get_anthropic_auto_agent_candidates_for_alias(
    alias_model: str,
) -> tuple[dict[str, Any], ...]:
    candidates = _ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(
        alias_model,
        _ANTHROPIC_AUTO_AGENT_CANDIDATES,
    )
    return candidates


# RR-054 durable alias-routing helpers (package-owned).
_get_aawm_alias_routing_state_namespace = _aawm_alias_durable.get_aawm_alias_routing_state_namespace
_build_aawm_alias_routing_durable_cache_key = _aawm_alias_durable.build_aawm_alias_routing_durable_cache_key
_get_aawm_alias_routing_dual_cache = _aawm_alias_durable.get_aawm_alias_routing_dual_cache
_parse_aawm_alias_routing_durable_expiry = _aawm_alias_durable.parse_aawm_alias_routing_durable_expiry
_read_aawm_alias_routing_durable_payload = _aawm_alias_durable.read_aawm_alias_routing_durable_payload
_write_aawm_alias_routing_durable_payload = _aawm_alias_durable.write_aawm_alias_routing_durable_payload
_AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX = _aawm_alias_durable.AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX
_AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT = _aawm_alias_durable.AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT

_ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL = _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL
_ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS = _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS
_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS = _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS = (
    _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS = (
    _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS = (
    _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS
)
_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS = (
    _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN = (
    _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN = (
    _aawm_google_oauth._ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN
)


# --- restored missing constants from HEAD (ordered) ---

_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST = _antigravity_constants._ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST

_OPENCODE_ZEN_DEFAULT_BASE_URL = _opencode_zen_constants._OPENCODE_ZEN_DEFAULT_BASE_URL
_OPENCODE_ZEN_PROVIDER = _opencode_zen_constants._OPENCODE_ZEN_PROVIDER
_OPENCODE_ZEN_AUTH_FILE_ENV_VARS = _opencode_zen_constants._OPENCODE_ZEN_AUTH_FILE_ENV_VARS
_OPENCODE_ZEN_API_KEY_ENV_VARS = _opencode_zen_constants._OPENCODE_ZEN_API_KEY_ENV_VARS
_OPENCODE_ZEN_DEFAULT_AUTH_PATHS = _opencode_zen_constants._OPENCODE_ZEN_DEFAULT_AUTH_PATHS
_OPENCODE_ZEN_FREE_MODELS = _opencode_zen_constants._OPENCODE_ZEN_FREE_MODELS
_OPENCODE_ZEN_ANTHROPIC_COMPLETION_MODELS = _opencode_zen_constants._OPENCODE_ZEN_ANTHROPIC_COMPLETION_MODELS

_GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL = _grok_side_channel._GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL

_GROK_CLI_FORWARD_HEADER_ALLOWLIST = _grok_side_channel._GROK_CLI_FORWARD_HEADER_ALLOWLIST

_GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE = _grok_side_channel._GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE

_CLAUDE_PERSISTED_OUTPUT_PATTERN = _aawm_lane_keys._CLAUDE_PERSISTED_OUTPUT_PATTERN

_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN = _aawm_lane_keys._CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN
_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN = _aawm_lane_keys._CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN
_CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN = _aawm_lane_keys._CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN
_ANTHROPIC_BILLING_HEADER_PREFIX = _anthropic_providers_common._ANTHROPIC_BILLING_HEADER_PREFIX
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME = _google_env_policy._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION = _google_env_policy._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV = _google_env_policy._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT = _google_env_policy._GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT
_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT = _google_env_policy._GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME = _google_env_policy._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION = _google_env_policy._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV = _google_env_policy._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT = _google_env_policy._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT = _google_env_policy._CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT

_CODEX_REASONING_EFFORT_TIERS = _aawm_lane_keys._CODEX_REASONING_EFFORT_TIERS
_CODEX_REASONING_EFFORT_TIER_INDEX = _aawm_lane_keys._CODEX_REASONING_EFFORT_TIER_INDEX
_CODEX_AUTO_AGENT_REASONING_EFFORT_AUDIT_FIELDS = _aawm_lane_keys._CODEX_AUTO_AGENT_REASONING_EFFORT_AUDIT_FIELDS
_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME = _aawm_lane_keys._CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME
_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION = _aawm_lane_keys._CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION
_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT = _aawm_lane_keys._CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT
_AAWM_READ_AGENT_GUIDANCE_POLICY_NAME = _aawm_lane_keys._AAWM_READ_AGENT_GUIDANCE_POLICY_NAME
_AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION = _aawm_lane_keys._AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION
_AAWM_READ_AGENT_GUIDANCE_PROMPT = _aawm_lane_keys._AAWM_READ_AGENT_GUIDANCE_PROMPT
_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS = _aawm_lane_keys._CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS
_CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS = _aawm_lane_keys._CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS

_CODEX_AUTO_AGENT_MALFORMED_TOOL_CALL_COOLDOWN_SECONDS = _aawm_lane_keys._CODEX_AUTO_AGENT_MALFORMED_TOOL_CALL_COOLDOWN_SECONDS
_CODEX_AUTO_AGENT_SPARK_MODEL = _aawm_lane_keys._CODEX_AUTO_AGENT_SPARK_MODEL
_CODEX_AUTO_AGENT_SPARK_DURABLE_COOLDOWN_SECONDS = _aawm_lane_keys._CODEX_AUTO_AGENT_SPARK_DURABLE_COOLDOWN_SECONDS
_CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS = _grok_side_channel._CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS
_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN = _grok_side_channel._CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN
_CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN = _grok_side_channel._CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN
_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL = _grok_side_channel._CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL

_CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES = _aawm_lane_keys._CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS = _aawm_lane_keys._CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS_ENV = _aawm_lane_keys._CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS_ENV
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS = 0.05
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS = 1.0
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS = 0.05

_CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES = frozenset(
    {
        "capacity_exhausted",
        "candidate_unavailable",
        "malformed_tool_call_text",
        "provider_terminal_error",
        "rate_limited",
        "usage_limit_reached",
        "upstream_overloaded",
        "upstream_timeout",
    }
)
_AAWM_ALIAS_ROUTE_VERBOSE_JSON_ENV = "AAWM_ALIAS_ROUTE_VERBOSE_JSON"
_CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS = 5 * 60.0
_CODEX_AUTO_AGENT_AUTH_DEGRADED_LOG_INTERVAL_SECONDS = 60.0
_CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY = "antigravity:auth_degraded"

_CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS = frozenset(
    {
        "HIGH_DEMAND",
        "MODEL_AT_CAPACITY",
        "MODEL_CAPACITY_EXHAUSTED",
        "MODEL_OVERLOADED",
        "UPSTREAM_BUSY",
    }
)
_CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS = frozenset(
    {
        "429",
        "RESOURCE_EXHAUSTED",
        "RATE_LIMIT_EXCEEDED",
        "rate_limit_exceeded",
    }
)

_GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING = "# Preserved Project And Safety Instructions"
_GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING = "# Original Claude System Instructions"
_GOOGLE_ADAPTER_CLAUDE_OVERHEAD_MARKERS = (
    "you are claude code",
    "anthropic's official cli for claude",
    "anthropic's official claude cli",
    "claude code's slash commands",
    "claude code slash commands",
)
_GOOGLE_ADAPTER_SYNTHETIC_TOOL_CONTEXT_PATTERN = re.compile(
    r"\ACalling (?:tool [A-Za-z0-9_.:-]+|tools: [A-Za-z0-9_.:,\-\s]+)\.\Z"
)
_OPENAI_ADAPTER_CONTEXT_MARKERS: tuple[tuple[str, str], ...] = (
    ("SubagentStart hook additional context:", "subagentstart"),
    ("SubAgentStart hook additional context:", "subagentstart"),
    ("# claudeMd", "claude-md"),
    ("CLAUDE.md", "claude-md"),
    ("MEMORY.md", "memory-md"),
    ("# TriStore Inject", "tristore-inject"),
)
_OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS = """You are an OpenAI Responses function-calling agent for Claude Code.

Parallel tool calls are enabled. When the current user task asks for multiple independent tool calls, emit all independent function calls together in one response output array before receiving any tool result. Do not serialize independent Read, Glob, Grep, Bash, WebSearch, or WebFetch calls when their arguments are already specified or can be determined from the current task.

Follow the latest user task exactly. Use the provided tool schemas as the source of truth for arguments. Emit no assistant text before tool calls when the task asks for tool calls only. After tool results return, provide the requested final answer.

Do NOT Write report/summary/findings/analysis .md files unless EXPLICITLY asked to do. Regardless of a file write-- you need to return findings directly as your final assistant message."""
_PASSTHROUGH_SESSION_ID_HEADER_NAMES = (
    "session_id",
    "Session_Id",
    "x-session-id",
    "X-Session-Id",
)
_PASS_THROUGH_HEADER_PREFIX = "x-pass-"
_AAWM_TENANT_ID_HEADER_NAMES = (
    "x-aawm-tenant-id",
    "x-litellm-tenant-id",
    "x-litellm-organization-id",
    "x-litellm-org-id",
    "x-organization-id",
    "x-org-id",
    "x-litellm-team-id",
    "x-team-id",
)
_PASSTHROUGH_REPOSITORY_HEADER_NAMES = (
    "x-aawm-repository",
    "x-litellm-repository",
    "x-repository",
    "x-git-repository",
)
_PASSTHROUGH_REPOSITORY_BODY_KEYS = frozenset(
    {
        "repository",
        "repo",
        "workspace_root",
        "workspaceRoot",
        "project_root",
        "projectRoot",
        "root_path",
        "rootPath",
        "working_directory",
        "workingDirectory",
        "cwd_path",
        "cwdPath",
        "cwd_uri",
        "cwdUri",
    }
)
_PASSTHROUGH_REPOSITORY_TEXT_PATTERNS = (
    re.compile(
        r"<environment_context>[\s\S]{0,2000}<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>",
        re.IGNORECASE,
    ),
    re.compile(r"<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>"),
    re.compile(r"AGENTS\.md instructions for\s+[`'\"]?(?P<path>/[^\n<`'\"]+)"),
    re.compile(r"\bcwd\b\s*[:=]\s*[`'\"]?(?P<path>/[^`'\"\n<]+)"),
    re.compile(
        r"\*{0,2}Workspace Directories:\*{0,2}\s*\n\s*[-*]\s*[`'\"]?(?P<path>/[^\n`'\"]+)",
        re.IGNORECASE,
    ),
)
_PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES = {
    "...",
    "memories",
    "new",
    "path",
    "project",
    "remote",
    "repo",
    "repository",
    "unknown",
}
_PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES = {
    "agent",
    "analyst",
    "architect",
    "engineer",
    "infra",
    "ops",
    "orchestrator",
    "principal",
    "qa",
    "researcher",
    "reviewer",
    "salvage",
    "tester",
}
_PASSTHROUGH_REPOSITORY_AGENT_ID_RE = re.compile(
    r"^agent-[a-f0-9]{3,}$",
    re.IGNORECASE,
)
_PASSTHROUGH_REPOSITORY_WAVE_AGENT_RE = re.compile(
    r"^wave\d+-(?:analyst|engineer|infra|ops|principal|qa|researcher|reviewer|salvage|tester)$",
    re.IGNORECASE,
)
_PASSTHROUGH_REPOSITORY_TRANSCRIPT_ARTIFACT_RE = re.compile(
    r"^(?:rollout-\d{4}(?:-[A-Za-z0-9_.-]*)?|.*\.jsonl?)$",
    re.IGNORECASE,
)
_ANTHROPIC_RESPONSES_ADAPTER_ENDPOINTS = frozenset({"/messages", "/v1/messages"})

_CODEX_GOOGLE_CODE_ASSIST_DEFAULT_MAX_TOKENS = 8192

_ANTHROPIC_ADAPTER_OPENAI_FORWARD_HEADER_ALLOWLIST = (
    "authorization",
    "api-key",
    "chatgpt-account-id",
    "originator",
    "user-agent",
    "session_id",
    "session-id",
)
_ANTHROPIC_ADAPTER_OPENAI_XPASS_HEADER_ALLOWLIST = (
    "authorization",
    "api-key",
    "chatgpt-account-id",
    "originator",
    "user-agent",
    "session_id",
    "session-id",
)
_ANTHROPIC_ADAPTER_OPENROUTER_API_KEY_ENV_VARS = (
    "AAWM_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
)
_ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS = (
    "AAWM_NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_API_KEY",
)
_ANTHROPIC_ADAPTER_NVIDIA_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES = sorted(
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES - {429}
)


vertex_llm_base = VertexBase()

router = APIRouter()
default_vertex_config = None

passthrough_endpoint_router = PassthroughEndpointRouter()


# ---------------------------------------------------------------------------
# Wave 5 (D1-583): operator alias-config refresh endpoint. Reads/validates/
# compiles the AAWM alias-routing YAML and atomically activates it via the
# Wave 3 snapshot holder (``get_active_routing_snapshot``/
# ``set_active_routing_snapshot``, defined below). Intentionally no auth
# dependency -- matches this instance's no-auth posture. Never echoes raw
# YAML/secrets back; only compiled snapshot identity fields
# (hash/version/changed) are returned.
# ---------------------------------------------------------------------------


async def _aawm_alias_config_refresh_route_endpoint(
    request: Request,
) -> dict[str, Any]:
    return await aawm_alias_config_refresh_route(request)


_aawm_alias_config_refresh_route_endpoint.__name__ = (
    aawm_alias_config_refresh_route.__name__
)
_aawm_alias_config_refresh_route_endpoint.__qualname__ = (
    aawm_alias_config_refresh_route.__qualname__
)
_aawm_alias_config_refresh_route_endpoint.__doc__ = (
    aawm_alias_config_refresh_route.__doc__
)


# Wave 5A: route registration delegates to the extracted config-refresh handler.
router.post(
    "/aawm/alias-config/refresh",
    tags=["AAWM Alias Routing"],
)(_aawm_alias_config_refresh_route_endpoint)


def _decode_http_response_body(body: Any) -> str:
    # RR-054 #43: never raise UnicodeDecodeError into JSON parse call sites.
    return bytes(body).decode("utf-8", errors="replace")


_GEMINI_OAUTH_FORWARD_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "authorization",
        "content-type",
        "user-agent",
        "x-goog-api-client",
    }
)

_ANTIGRAVITY_CODE_ASSIST_DEFAULT_BASE_URL = "https://daily-cloudcode-pa.googleapis.com"
_ANTIGRAVITY_CLIENT_HEADER_DEFAULT = "antigravity-cli/1.0.4"
# RR-054 #1: Antigravity OAuth path/client constants owned by package.
_ANTIGRAVITY_AUTH_FILE_ENV_VARS = _aawm_antigravity_oauth._ANTIGRAVITY_AUTH_FILE_ENV_VARS
_ANTIGRAVITY_MANAGED_AUTH_FILE_ENV_VARS = _aawm_antigravity_oauth._ANTIGRAVITY_MANAGED_AUTH_FILE_ENV_VARS
_ANTIGRAVITY_SEED_AUTH_FILE_ENV_VARS = _aawm_antigravity_oauth._ANTIGRAVITY_SEED_AUTH_FILE_ENV_VARS
_ANTIGRAVITY_DEFAULT_AUTH_PATHS = _aawm_antigravity_oauth._ANTIGRAVITY_DEFAULT_AUTH_PATHS
_ANTIGRAVITY_OAUTH_CLIENT_ID_ENV_VARS = _aawm_antigravity_oauth._ANTIGRAVITY_OAUTH_CLIENT_ID_ENV_VARS
_ANTIGRAVITY_OAUTH_CLIENT_SECRET_ENV_VARS = _aawm_antigravity_oauth._ANTIGRAVITY_OAUTH_CLIENT_SECRET_ENV_VARS
_ANTIGRAVITY_CLI_BINARY_PATH_ENV_VARS = _aawm_antigravity_oauth._ANTIGRAVITY_CLI_BINARY_PATH_ENV_VARS
_ANTIGRAVITY_DEFAULT_CLI_BINARY_PATHS = _aawm_antigravity_oauth._ANTIGRAVITY_DEFAULT_CLI_BINARY_PATHS
_ANTIGRAVITY_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN = _aawm_antigravity_oauth._ANTIGRAVITY_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN
_ANTIGRAVITY_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN = (
    _aawm_antigravity_oauth._ANTIGRAVITY_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN
)
_CLAUDE_AGENT_SPEC_DIR_ENV_VARS = (
    "LITELLM_CLAUDE_AGENTS_DIR",
    "CLAUDE_AGENTS_DIR",
)
_CLAUDE_AGENT_SPEC_DEFAULT_DIRS = (
    "~/.claude/agents",
    "~/.claude/agents",
)
_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = Path(__file__).resolve().parents[3] / "context-replacement" / "claude-code"
_CLAUDE_AUTO_MEMORY_TEMPLATE_PATH = _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR / "2.1.110" / "auto-memory-replacement.md"
_CLAUDE_PROMPT_PATCH_MANIFEST_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR / "prompt-patches" / "roman01la-2026-04-02.json"
)
_CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION = (2, 1, 110)
_CLAUDE_AUTO_MEMORY_SECTION_PATTERN = re.compile(r"(?ms)^# auto memory\n.*?(?=^# Environment\b|\Z)")
_CLAUDE_TYPES_XML_BLOCK_PATTERN = re.compile(r"<types>\n.*?\n</types>", re.DOTALL)
_CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
_CLAUDE_CC_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
_CLAUDE_AGENT_TENANT_PATTERN = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project\b"
)
_CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS: tuple[tuple[str, str], ...] = (
    ("MEMORY.md", "memory-md"),
    ("CLAUDE.md", "claude-md"),
)
# Wave 5B: OpenRouter quota cache/lock are now manager-owned.
_openrouter_free_daily_quota_lock = _alias_routing_state.openrouter_free_quota_lock
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}
_claude_agent_model_cache: dict[Path, tuple[Optional[int], Optional[str]]] = {}
# RR-054 #1: OAuth access-token caches owned by aawm_alias_routing package.
_google_oauth_access_token_cache = _alias_routing_state.google_oauth.tokens
_google_oauth_access_token_lock = _alias_routing_state.google_oauth.lock
_antigravity_oauth_access_token_cache = _alias_routing_state.antigravity_oauth.tokens
_antigravity_oauth_access_token_lock = _alias_routing_state.antigravity_oauth.lock
# RR-054 #1/#3: Google process state is constructed only by the provider owner.
_GOOGLE_ADAPTER_TOKEN_CACHE_MAX_SIZE = _anthropic_google_process_cache._GOOGLE_ADAPTER_TOKEN_CACHE_MAX_SIZE
_google_code_assist_project_cache = _anthropic_google_process_cache._google_code_assist_project_cache
_google_code_assist_project_lock = _anthropic_google_process_cache._google_code_assist_project_lock
_google_code_assist_prime_until_monotonic_by_key = (
    _anthropic_google_process_cache._google_code_assist_prime_until_monotonic_by_key
)
_google_code_assist_prime_quota_by_key = _anthropic_google_process_cache._google_code_assist_prime_quota_by_key
_google_code_assist_prime_lock = _anthropic_google_process_cache._google_code_assist_prime_lock
_google_adapter_semaphores = _anthropic_google_process_cache._google_adapter_semaphores
_google_adapter_rate_limit_lock = _alias_routing_state.google_rate_limit.lock
_google_adapter_rate_limit_until_monotonic_by_key = _alias_routing_state.google_rate_limit.until_monotonic_by_key


async def _post_code_assist_json(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, object],
    timeout: float,
) -> _anthropic_google_process_cache.HttpResponse:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(url, headers=headers, json=body)


def _raise_code_assist_process_cache_http_error(
    *,
    status_code: int,
    detail: str,
) -> Never:
    raise HTTPException(status_code=status_code, detail=detail)


@lru_cache(maxsize=1)
def _get_anthropic_google_process_cache_runtime() -> _anthropic_google_process_cache.Runtime:
    return _anthropic_google_process_cache.Runtime(
        get_target_base=lambda provider: _get_code_assist_adapter_target_base(provider),
        build_headers=lambda **kwargs: _build_code_assist_adapter_native_headers(**kwargs),
        validate_egress=lambda **kwargs: (HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)),
        post_json=_post_code_assist_json,
        capture_shape=lambda **kwargs: capture_passthrough_shape(**kwargs),
        clean_value=lambda value: _clean_codex_auth_value(value),
        raise_http_error=_raise_code_assist_process_cache_http_error,
        get_prime_ttl_seconds=lambda: _get_google_code_assist_prime_ttl_seconds(),
        get_prime_cache_key=lambda token, project: (_get_google_code_assist_prime_cache_key(token, project)),
        sanitize_quota=lambda value, source: (_sanitize_google_code_assist_quota_for_logging(value, source=source)),
        get_max_concurrent=lambda: _get_google_adapter_max_concurrent(),
        get_rate_limit_key=lambda model, **kwargs: (_get_google_adapter_rate_limit_key(model, **kwargs)),
        monotonic=time.monotonic,
        debug_enabled=lambda: os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1",
        log_info=lambda message, value: verbose_proxy_logger.info(message, value),
        default_provider=litellm.LlmProviders.GEMINI.value,
    )


def _bound_google_adapter_token_cache(cache: dict, *, max_size: int = _GOOGLE_ADAPTER_TOKEN_CACHE_MAX_SIZE) -> None:
    return _anthropic_google_process_cache._bound_google_adapter_token_cache(
        cache,
        max_size=max_size,
    )


_google_adapter_user_prompt_turn_lock = _anthropic_google_process_cache._google_adapter_user_prompt_turn_lock
_google_adapter_user_prompt_turn_counters = _anthropic_google_process_cache._google_adapter_user_prompt_turn_counters
_openrouter_adapter_rate_limit_lock = _alias_routing_state.openrouter_rate_limit.lock
_openrouter_adapter_rate_limit_until_monotonic_by_key = (
    _alias_routing_state.openrouter_rate_limit.until_monotonic_by_key
)
_openrouter_adapter_failure_circuit_until_monotonic_by_key = (
    _alias_routing_state.openrouter_failure_circuit.until_monotonic_by_key
)
_CODEX_SPAWN_AGENT_TOOL_NAME = "spawn_agent"
_CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE = "tool_search"
_CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID = "spawn-agent-fanout-policy"
_CODEX_SPAWN_AGENT_PAYLOAD_SCHEMA_PATCH_ID = "spawn-agent-payload-schema"
_CODEX_CORE_TOOL_GUIDANCE_PATCH_PREFIX = "core-tool-guidance"
_CODEX_UNSUPPORTED_HOSTED_TOOLS_MODEL_INFO_FIELD = "unsupported_hosted_tools"
_CODEX_UNSUPPORTED_REQUEST_PARAMS_MODEL_INFO_FIELD = "unsupported_request_params"
_CODEX_UNSUPPORTED_INPUT_ITEM_TYPES_MODEL_INFO_FIELD = "unsupported_input_item_types"
_CODEX_REWRITE_INPUT_ITEM_TYPES_MODEL_INFO_FIELD = "rewrite_input_item_types"
_CODEX_CUSTOM_TOOL_FUNCTION_ADAPTERS_MODEL_INFO_FIELD = "custom_tool_function_adapters"
_CODEX_NAMESPACE_TOOL_FUNCTION_ADAPTERS_MODEL_INFO_FIELD = "namespace_tool_function_adapters"
_CODEX_SPAWN_AGENT_FANOUT_POLICY = (
    "Use subagents to parallelize independent work while keeping one local owner "
    "on the critical path. Follow the current operator and project instructions "
    "that authorize fanout; do not treat generic depth or investigation wording "
    "as permission to launch unrelated autonomous fanout. Do not duplicate the "
    "same task across agents.\n\n"
    "For read-only or exploration workers, call multi_agent_v1.spawn_agent with "
    'lower-case payload fields: model="aawm-codex-agent-auto", '
    'fork_turns="none" unless context sharing is explicitly needed, and message '
    "containing the read-only boundary plus the audit task. If a fix is needed, "
    "the worker should describe the patch only.\n\n"
    "For coding workers, this read-only payload does not apply. Include the "
    "selected coding model from the configured coding-model priority order, "
    "assign a clear disjoint write set, and tell workers they are not alone in "
    "the codebase. They must not revert unrelated edits.\n\n"
    "Use the latest frontier model for cross-document architecture, migration-risk "
    "review, and high-stakes database safety reasoning. Use the latest Codex model "
    "for bounded implementation tasks with clear, disjoint write ownership. Use "
    "mini-class agents for narrow grep/read-only scans, documentation consistency "
    "checks, test inventory, and quick QA passes. For database or migration "
    "work, prefer read-only explorer subagents; the main owner should run live "
    "database commands so target verification and credential handling stay in "
    "one place."
)
_CODEX_SPAWN_AGENT_PAYLOAD_FIELD_SCHEMAS: dict[str, dict[str, Any]] = {
    "agent_type": {
        "type": "string",
        "description": (
            "Optional configured agent role. Use a role whose config selects the "
            "required model and execution policy."
        ),
    },
    "model": {
        "type": "string",
        "description": (
            "Optional lower-case model override accepted by the orchestrator. "
            "Use aawm-codex-agent-auto for read-only/exploration workers; use "
            "the selected coding model for coding workers."
        ),
    },
    "fork_turns": {
        "type": "string",
        "enum": ["none", "all"],
        "description": (
            "Which parent turns to fork into the worker. Use none for isolated "
            "workers unless the complete parent context is explicitly required."
        ),
    },
    "message": {
        "type": "string",
        "description": (
            "Plain-text task prompt for the worker, including read-only or "
            "coding scope, file boundaries, and final-answer requirements."
        ),
    },
}
_CODEX_SPAWN_AGENT_PAYLOAD_FIELD_ORDER = (
    "agent_type",
    "model",
    "fork_turns",
    "message",
)
_CODEX_CORE_TOOL_GUIDANCE_BY_NAME: dict[str, str] = {
    "bash": (
        "Claude Code core tool reliability guidance: Use Bash for inspection, "
        "test, and simple commands. Prefer structured Edit or Write tools for "
        "source changes instead of complex sed, perl, awk, or shell-quoted "
        "rewrites. After a shell quoting or syntax error, do not retry a more "
        "complex one-liner; switch to a smaller structured edit or report the "
        "exact blocker."
    ),
    "edit": (
        "Claude Code core tool reliability guidance: Edit old_string must be "
        "copied from the current file contents. If an Edit fails with "
        "`String to replace not found in file`, do not retry the same "
        "old_string. Re-read the exact target span, narrow the hunk to the "
        "smallest stable current context, and then retry once with current "
        "text."
    ),
    "read": (
        "Claude Code core tool reliability guidance: Use bounded reads for "
        "large transcript, task-output, or log files. For .output transcript "
        "files, use offset/limit or available transcript search/meta tools "
        "instead of unbounded full-file reads."
    ),
    "write": (
        "Claude Code core tool reliability guidance: Use Write for new files or "
        "known full-file replacements. Before overwriting an existing file, read "
        "the current file first and preserve unrelated content."
    ),
}
_CODEX_SPAWN_AGENT_RESTRICTIVE_DESCRIPTION_PATTERNS = (
    re.compile(
        r"Only use `?spawn_agent`? if and only if the user explicitly asks for "
        r"sub-?agents, delegation, or parallel agent work\.\s*"
        r"Requests for depth, thoroughness, research, investigation, or detailed "
        r"codebase analysis do not count as permission to spawn\.\s*"
        r"Agent-role guidance below only helps choose which agent to use after "
        r"spawning is already authorized; it never authorizes spawning by itself\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"Only use `?spawn_agent`? if and only if the user explicitly asks for "
        r"sub-?agents, delegation, or parallel agent work\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"I may only use `?spawn_agent`? when the user explicitly asks for "
        r"sub-?agents, delegation, or parallel agent work\.",
        re.IGNORECASE,
    ),
)


def _is_openai_responses_endpoint(endpoint: str) -> bool:
    normalized_path = httpx.URL(endpoint).path.rstrip("/")
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    return (
        normalized_path == "/responses"
        or normalized_path == "/v1/responses"
        or normalized_path.startswith("/responses/")
        or normalized_path.startswith("/v1/responses/")
    )


def _is_openai_models_endpoint(endpoint: str) -> bool:
    normalized_path = httpx.URL(endpoint).path.rstrip("/")
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    return normalized_path == "/models" or normalized_path == "/v1/models"


def _get_openai_passthrough_route_family(endpoint: str) -> str:
    normalized_path = httpx.URL(endpoint).path.rstrip("/")
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    if _is_openai_responses_endpoint(endpoint):
        return "openai_responses"
    if normalized_path in {"/chat/completions", "/v1/chat/completions"}:
        return "openai_chat_completions"
    return "openai_passthrough"


_is_oa_xai_request_body = _wave6b_xai_request_prep._is_oa_xai_request_body


_is_grok_native_oauth_request_body = _wave6b_xai_request_prep._is_grok_native_oauth_request_body


@lru_cache(maxsize=1)
def _load_local_model_metadata() -> dict[str, Any]:
    model_metadata_path = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
    try:
        with model_metadata_path.open("r", encoding="utf-8") as model_metadata_file:
            metadata = json.load(model_metadata_file)
    except Exception:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _get_model_metadata_entry(model: Any) -> Optional[dict[str, Any]]:
    if not isinstance(model, str):
        return None
    candidate_models = [model]
    if is_oa_xai_model(model):
        try:
            candidate_models.append(resolve_oa_xai_upstream_model(model))
        except Exception:
            pass
    local_model_metadata = _load_local_model_metadata()
    for candidate_model in candidate_models:
        model_info = litellm.model_cost.get(candidate_model)
        if isinstance(model_info, dict):
            return model_info
        local_model_info = local_model_metadata.get(candidate_model)
        if isinstance(local_model_info, dict):
            return local_model_info
    return None


_is_oa_xai_responses_model = _wave6b_xai_request_prep._is_oa_xai_responses_model


_to_xai_native_passthrough_model = _wave6b_xai_request_prep._to_xai_native_passthrough_model


_xai_responses_sanitized_tool_changes = _wave6b_xai_request_prep._xai_responses_sanitized_tool_changes


_sanitize_xai_responses_request_body = _wave6b_xai_request_prep._sanitize_xai_responses_request_body


_coerce_grok_native_function_call_arguments_value = _wave6b_xai_request_prep._coerce_grok_native_function_call_arguments_value


_get_anthropic_grok_normalization_runtime = _wave6b_xai_request_prep._get_anthropic_grok_normalization_runtime


_sanitize_grok_native_function_call_arguments_request_body = _wave6b_xai_request_prep._sanitize_grok_native_function_call_arguments_request_body


_sanitize_grok_native_function_call_arguments_in_place = _wave6b_xai_request_prep._sanitize_grok_native_function_call_arguments_in_place


_sanitize_xai_responses_request_body_in_place = _wave6b_xai_request_prep._sanitize_xai_responses_request_body_in_place


_prepare_oa_xai_passthrough_request = _wave6b_xai_request_prep._prepare_oa_xai_passthrough_request


_get_grok_native_oauth_client_version = _wave6b_xai_request_prep._get_grok_native_oauth_client_version


_get_grok_native_oauth_session_id = _wave6b_xai_request_prep._get_grok_native_oauth_session_id


_get_grok_native_oauth_request_id = _wave6b_xai_request_prep._get_grok_native_oauth_request_id


_build_grok_native_oauth_headers = _wave6b_xai_request_prep._build_grok_native_oauth_headers


_add_grok_native_oauth_metadata = _wave6b_xai_request_prep._add_grok_native_oauth_metadata


_prepare_grok_native_oauth_passthrough_request = _wave6b_xai_request_prep._prepare_grok_native_oauth_passthrough_request


def _get_gemini_passthrough_route_family(endpoint: str) -> Optional[str]:
    normalized_endpoint = endpoint.lower()
    if "streamgeneratecontent" in normalized_endpoint:
        return "gemini_stream_generate_content"
    if "generatecontent" in normalized_endpoint:
        return "gemini_generate_content"
    if "predictlongrunning" in normalized_endpoint:
        return "gemini_predict_long_running"
    return None


def _request_has_openai_client_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    return bool(
        headers.get("authorization") or headers.get("Authorization") or headers.get("api-key") or headers.get("Api-Key")
    )


def _get_request_header_or_passthrough_alias(request: Request, header_name: str) -> Optional[str]:
    headers = _safe_get_request_headers(request)
    candidates = (
        header_name,
        header_name.lower(),
        f"{_PASS_THROUGH_HEADER_PREFIX}{header_name}",
        f"{_PASS_THROUGH_HEADER_PREFIX}{header_name.lower()}",
    )
    for candidate in candidates:
        value = headers.get(candidate)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

# Wave 5A: bind request-header helper into codex_oauth.
_aawm_codex_oauth.configure_codex_oauth_runtime(
    get_request_header_or_passthrough_alias=_get_request_header_or_passthrough_alias,
)


def _get_aawm_tenant_header(request: Request) -> Optional[str]:
    for header_name in _AAWM_TENANT_ID_HEADER_NAMES:
        value = _get_request_header_or_passthrough_alias(request, header_name)
        if value:
            return value
    return None


def _has_direct_request_header(request: Request, header_name: str) -> bool:
    headers = _safe_get_request_headers(request)
    value = headers.get(header_name) or headers.get(header_name.lower())
    return isinstance(value, str) and len(value.strip()) > 0


_normalize_anthropic_adapter_model_name = _aawm_adapter_model_resolution._normalize_anthropic_adapter_model_name


_split_anthropic_adapter_provider_prefix = _aawm_adapter_model_resolution._split_anthropic_adapter_provider_prefix


_get_anthropic_adapter_model_candidates = _aawm_adapter_model_resolution._get_anthropic_adapter_model_candidates


_has_anthropic_responses_adapter_endpoint = _aawm_adapter_model_resolution._has_anthropic_responses_adapter_endpoint


_normalize_anthropic_openai_responses_adapter_model_name = _aawm_adapter_model_resolution._normalize_anthropic_openai_responses_adapter_model_name


_normalize_anthropic_nvidia_responses_adapter_model_name = _aawm_adapter_model_resolution._normalize_anthropic_nvidia_responses_adapter_model_name


_normalize_anthropic_openrouter_adapter_model_name = _aawm_adapter_model_resolution._normalize_anthropic_openrouter_adapter_model_name


_get_openrouter_completion_adapter_upstream_model = _aawm_adapter_model_resolution._get_openrouter_completion_adapter_upstream_model


_normalize_opencode_zen_adapter_model_name = _aawm_adapter_model_resolution._normalize_opencode_zen_adapter_model_name


_normalize_kimi_code_chat_completions_adapter_model_name = _aawm_adapter_model_resolution._normalize_kimi_code_chat_completions_adapter_model_name


_normalize_alibaba_token_plan_adapter_model_name = _aawm_adapter_model_resolution._normalize_alibaba_token_plan_adapter_model_name


_normalize_anthropic_google_completion_adapter_model_name = _aawm_adapter_model_resolution._normalize_anthropic_google_completion_adapter_model_name


_normalize_antigravity_code_assist_adapter_model_name = _aawm_adapter_model_resolution._normalize_antigravity_code_assist_adapter_model_name


_normalize_codex_google_code_assist_adapter_model_name = _aawm_adapter_model_resolution._normalize_codex_google_code_assist_adapter_model_name


_resolve_codex_opencode_zen_adapter_model = _aawm_adapter_model_resolution._resolve_codex_opencode_zen_adapter_model


_resolve_codex_kimi_chat_completions_adapter_model = _aawm_adapter_model_resolution._resolve_codex_kimi_chat_completions_adapter_model


_resolve_codex_alibaba_token_plan_adapter_model = _aawm_adapter_model_resolution._resolve_codex_alibaba_token_plan_adapter_model


_resolve_anthropic_opencode_zen_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_opencode_zen_adapter_model


_resolve_anthropic_kimi_chat_completions_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_kimi_chat_completions_adapter_model


_resolve_anthropic_alibaba_token_plan_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_alibaba_token_plan_adapter_model


_resolve_anthropic_antigravity_code_assist_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_antigravity_code_assist_adapter_model


_resolve_codex_google_code_assist_adapter_model = _aawm_adapter_model_resolution._resolve_codex_google_code_assist_adapter_model


_resolve_codex_antigravity_code_assist_adapter_model = _aawm_adapter_model_resolution._resolve_codex_antigravity_code_assist_adapter_model


_normalize_codex_auto_agent_alias_model = _aawm_adapter_model_resolution._normalize_codex_auto_agent_alias_model


_is_codex_auto_agent_alias_model = _aawm_adapter_model_resolution._is_codex_auto_agent_alias_model


_resolve_codex_auto_agent_alias_model = _aawm_adapter_model_resolution._resolve_codex_auto_agent_alias_model


_get_codex_auto_agent_header = _aawm_lane_keys._get_codex_auto_agent_header


_hash_codex_auto_agent_lane_value = _aawm_lane_keys._hash_codex_auto_agent_lane_value


_resolve_codex_auto_agent_openai_lane_key = _aawm_lane_keys._resolve_codex_auto_agent_openai_lane_key


_resolve_codex_auto_agent_openai_cooldown_lane_key = _aawm_lane_keys._resolve_codex_auto_agent_openai_cooldown_lane_key


_get_codex_auto_agent_lane_state_cache_ttl_seconds = _aawm_lane_keys._get_codex_auto_agent_lane_state_cache_ttl_seconds


_get_codex_auto_agent_google_lane_cache_key = _aawm_lane_keys._get_codex_auto_agent_google_lane_cache_key


_get_codex_auto_agent_antigravity_lane_cache_key = _aawm_lane_keys._get_codex_auto_agent_antigravity_lane_cache_key


def _invalidate_codex_auto_agent_google_lane_cache() -> None:
    cache_key = _get_codex_auto_agent_google_lane_cache_key()
    _codex_auto_agent_google_lane_key_until_monotonic_by_key.pop(cache_key, None)
    _codex_auto_agent_google_lane_key_by_key.pop(cache_key, None)


def _invalidate_codex_auto_agent_antigravity_lane_cache() -> None:
    cache_key = _get_codex_auto_agent_antigravity_lane_cache_key()
    _codex_auto_agent_antigravity_lane_key_until_monotonic_by_key.pop(
        cache_key,
        None,
    )
    _codex_auto_agent_antigravity_lane_key_by_key.pop(cache_key, None)


def _invalidate_codex_auto_agent_lane_state_caches() -> None:
    _invalidate_codex_auto_agent_google_lane_cache()
    _invalidate_codex_auto_agent_antigravity_lane_cache()


async def _resolve_codex_auto_agent_google_lane_key() -> str:
    if time.monotonic() < _alias_routing_state.google_lane_negative_until_monotonic:
        return _CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY
    cache_key = _get_codex_auto_agent_google_lane_cache_key()
    ttl_seconds = _get_codex_auto_agent_lane_state_cache_ttl_seconds()
    if ttl_seconds > 0:
        async with _codex_auto_agent_lane_state_cache_lock:
            cached_until = _codex_auto_agent_google_lane_key_until_monotonic_by_key.get(cache_key, 0.0)
            if cached_until > time.monotonic():
                cached_lane_key = _codex_auto_agent_google_lane_key_by_key.get(cache_key)
                if isinstance(cached_lane_key, str) and cached_lane_key:
                    return cached_lane_key

    try:
        google_access_token = await _load_valid_local_google_oauth_access_token()
        google_project = await _get_or_load_google_code_assist_project(google_access_token)
        lane_key = _get_google_adapter_rate_limit_key(
            None,
            access_token=google_access_token,
            companion_project=google_project,
        )
    except Exception:
        _invalidate_codex_auto_agent_google_lane_cache()
        _alias_routing_state.google_lane_negative_until_monotonic = (
            time.monotonic() + _CODEX_AUTO_AGENT_GOOGLE_LANE_NEGATIVE_TTL_SECONDS
        )
        if _should_log_aawm_alias_routing_event("google-lane-resolve-failed"):
            verbose_proxy_logger.warning(
                "Codex auto-agent alias could not resolve Google Code Assist lane; "
                "marking auth-degraded (negative-cached %.1fs)",
                _CODEX_AUTO_AGENT_GOOGLE_LANE_NEGATIVE_TTL_SECONDS,
                exc_info=True,
            )
        return _CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY

    if ttl_seconds > 0:
        async with _codex_auto_agent_lane_state_cache_lock:
            _codex_auto_agent_google_lane_key_by_key[cache_key] = lane_key
            _codex_auto_agent_google_lane_key_until_monotonic_by_key[cache_key] = time.monotonic() + ttl_seconds
    return lane_key


async def _resolve_codex_auto_agent_google_lane_state() -> Payload:
    lane_key = await _resolve_codex_auto_agent_google_lane_key()
    if lane_key != _CODEX_AUTO_AGENT_GOOGLE_AUTH_DEGRADED_LANE_KEY:
        return {"lane_key": lane_key}
    return {
        "lane_key": lane_key,
        "forced_cooldown_seconds": _CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS,
        "skip_reason": "auth_degraded",
        "cooldown_state_source": "auth_degraded",
        "failure_phase": "auth",
        "attempted_provider_call": False,
    }


def _is_codex_auto_agent_antigravity_auth_degraded_exception(exc: Any) -> bool:
    return _anthropic_antigravity_provider._is_codex_auto_agent_antigravity_auth_degraded_exception(
        exc,
        runtime=_get_anthropic_antigravity_runtime(),
    )


def _log_codex_auto_agent_antigravity_auth_degraded(exc: HTTPException) -> None:
    now = time.monotonic()
    if now < _alias_routing_state.antigravity_auth_degraded_log_until_monotonic:
        return
    _alias_routing_state.antigravity_auth_degraded_log_until_monotonic = (
        now + _CODEX_AUTO_AGENT_AUTH_DEGRADED_LOG_INTERVAL_SECONDS
    )
    verbose_proxy_logger.warning(
        "Codex auto-agent alias marked Antigravity Code Assist lane degraded; "
        "using auth-degraded lane sentinel until sidecar refresh is available "
        "(provider=antigravity, failure_kind=auth_degraded, status_code=%s, "
        "cooldown_seconds=%.1f, detail=%s)",
        exc.status_code,
        _CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS,
        str(exc.detail),
    )


async def _resolve_codex_auto_agent_antigravity_lane_key() -> str:
    cache_key = _get_codex_auto_agent_antigravity_lane_cache_key()
    ttl_seconds = _get_codex_auto_agent_lane_state_cache_ttl_seconds()
    if ttl_seconds > 0:
        async with _codex_auto_agent_lane_state_cache_lock:
            cached_until = _codex_auto_agent_antigravity_lane_key_until_monotonic_by_key.get(cache_key, 0.0)
            if cached_until > time.monotonic():
                cached_lane_key = _codex_auto_agent_antigravity_lane_key_by_key.get(cache_key)
                if isinstance(cached_lane_key, str) and cached_lane_key:
                    return cached_lane_key

    try:
        antigravity_access_token = await _load_valid_local_antigravity_access_token()
        antigravity_project = await _get_or_load_google_code_assist_project(
            antigravity_access_token,
            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
        )
        lane_key = "antigravity:{}".format(
            _get_google_adapter_rate_limit_key(
                None,
                access_token=antigravity_access_token,
                companion_project=antigravity_project,
            )
        )
    except Exception as exc:
        if _is_codex_auto_agent_antigravity_auth_degraded_exception(exc):
            _invalidate_codex_auto_agent_antigravity_lane_cache()
            _log_codex_auto_agent_antigravity_auth_degraded(cast(HTTPException, exc))
            return _CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY
        _invalidate_codex_auto_agent_antigravity_lane_cache()
        verbose_proxy_logger.error(
            "Codex auto-agent alias could not resolve Antigravity Code Assist lane; using default lane",
            exc_info=True,
        )
        return "__default__"

    if ttl_seconds > 0 and lane_key != "__default__":
        async with _codex_auto_agent_lane_state_cache_lock:
            _codex_auto_agent_antigravity_lane_key_by_key[cache_key] = lane_key
            _codex_auto_agent_antigravity_lane_key_until_monotonic_by_key[cache_key] = time.monotonic() + ttl_seconds
    return lane_key


async def _resolve_codex_auto_agent_antigravity_lane_state() -> dict[str, Any]:
    lane_key = await _resolve_codex_auto_agent_antigravity_lane_key()
    if lane_key != _CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY:
        return {"lane_key": lane_key}
    return {
        "lane_key": lane_key,
        "forced_cooldown_seconds": _CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS,
        "skip_reason": "auth_degraded",
        "cooldown_state_source": "auth_degraded",
        "failure_phase": "auth",
        "attempted_provider_call": False,
    }


def _resolve_codex_auto_agent_session_key(
    request: Request,
    request_body: dict[str, Any],
    *,
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    metadata_session_id = metadata.get("session_id") if isinstance(metadata, dict) else None
    session_id = _clean_codex_auth_value(metadata_session_id)
    headers = _safe_get_request_headers(request)
    if session_id is None:
        session_id = _get_codex_auto_agent_header(headers, "session_id") or _get_codex_auto_agent_header(
            headers, "session-id"
        )
    if session_id is None:
        return None
    if alias_model == _CODEX_AUTO_AGENT_MODEL_ALIAS:
        return f"{session_id}:{_resolve_codex_auto_agent_openai_lane_key(request)}"
    return f"{alias_model}:{session_id}:" f"{_resolve_codex_auto_agent_openai_lane_key(request)}"


_codex_auto_agent_candidate_key = _aawm_lane_keys._codex_auto_agent_candidate_key


_codex_auto_agent_candidate_public_shape = _aawm_selection._codex_auto_agent_candidate_public_shape


_auto_agent_alias_float = _aawm_selection._auto_agent_alias_float


def _auto_agent_alias_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_auto_agent_alias_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _auto_agent_alias_cooldown_until(
    cooldown_seconds: Optional[float],
) -> Optional[str]:
    if cooldown_seconds is None:
        return None
    return _format_auto_agent_alias_timestamp(
        datetime.now(timezone.utc) + timedelta(seconds=max(0.0, cooldown_seconds))
    )


def _extract_auto_agent_alias_session_id(
    request: Request,
    request_body: dict[str, Any],
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        session_id = _clean_codex_auth_value(metadata.get("session_id"))
        if session_id is not None:
            return session_id
    passthrough_session_id = _extract_passthrough_session_id(request, request_body)
    if passthrough_session_id is not None:
        return passthrough_session_id
    headers = _safe_get_request_headers(request)
    for header_name in ("session_id", "session-id", "x-session-id"):
        header_value = _get_codex_auto_agent_header(headers, header_name)
        if header_value is not None:
            return header_value
    return None


def _extract_auto_agent_alias_metadata_value(
    request_body: dict[str, Any],
    *keys: str,
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = _clean_codex_auth_value(metadata.get(key))
        if value is not None:
            return value
    return None


def _normalize_auto_agent_alias_client_product(value: Any) -> Optional[str]:
    cleaned = _clean_codex_auth_value(value)
    if cleaned is None:
        return None
    product = cleaned.split()[0].strip("()")
    if not product:
        return None
    if "/" not in product:
        return product
    name, version = product.split("/", 1)
    normalized_name = name.lower().replace("_", "-")
    if normalized_name in {"codex", "codex-cli", "codex-tui", "codex-cli-rs"}:
        name = "Codex"
    elif normalized_name in {"claude", "claude-cli", "claude-code"}:
        name = "Claude"
    elif normalized_name in {"grok", "grok-build", "grok-pager"}:
        name = "Grok"
    return f"{name}/{version}"


def _extract_auto_agent_alias_client_product_label(
    request: Request,
    request_body: dict[str, Any],
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        for key in (
            "client_name_version",
            "client_label",
            "client_user_agent",
            "user_agent",
        ):
            value = _normalize_auto_agent_alias_client_product(metadata.get(key))
            if value:
                return value
        name = _normalize_auto_agent_alias_client_product(metadata.get("client_name"))
        version = _clean_codex_auth_value(metadata.get("client_version"))
        if name and version and "/" not in name:
            return f"{name}/{version}"
        if name:
            return name
    headers = _safe_get_request_headers(request)
    for header_name in (
        "x-aawm-client",
        "x-litellm-client",
        "x-client-name-version",
        "user-agent",
    ):
        value = _normalize_auto_agent_alias_client_product(_get_codex_auto_agent_header(headers, header_name))
        if value:
            return value
    return None


def _extract_auto_agent_alias_incoming_endpoint(request: Request) -> str:
    parsed_url = urlparse(str(getattr(request, "url", "") or ""))
    path = parsed_url.path or getattr(request, "path", None) or "/"
    safe_pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed_url.query, keep_blank_values=True):
        if key.lower() not in {"alt", "api-version", "beta", "stream"}:
            continue
        safe_key = _clean_codex_auth_value(key)
        safe_value = _clean_codex_auth_value(value)
        if safe_key and safe_value is not None:
            safe_pairs.append((safe_key, safe_value))
    if not safe_pairs:
        return path
    return f"{path}?{urlencode(safe_pairs)}"


def _resolve_auto_agent_alias_route_rollup_outgoing_target(
    *,
    route_family: Optional[str],
    target_url: Optional[Union[str, httpx.URL]] = None,
) -> Optional[str]:
    cleaned_route_family = _clean_codex_auth_value(route_family)
    if target_url is not None:
        return _get_anthropic_adapter_access_log_target_label(target_url)
    route_family_target_labels = {
        "codex_opencode_zen_adapter": "opencode.ai/zen/v1/chat/completions",
        "codex_openrouter_completion_adapter": "openrouter.ai/api/v1/chat/completions",
        "anthropic_opencode_zen_responses_adapter": "opencode.ai/zen/v1/responses",
        "anthropic_opencode_zen_completion_adapter": "opencode.ai/zen/v1/chat/completions",
    }
    return route_family_target_labels.get(cleaned_route_family or "", cleaned_route_family)


def _build_adapted_route_rollup_kwargs(
    litellm_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "litellm_params": {
            "metadata": dict(litellm_metadata),
        }
    }


def _emit_adapted_route_access_log(
    *,
    request: Request,
    target_url: str,
    request_body: dict[str, Any],
    rollup_kwargs: dict[str, Any],
    adapter_label: str,
) -> None:
    try:
        emit_aawm_route_access_log(
            request=request,
            target=target_url,
            request_body=request_body,
            kwargs=rollup_kwargs,
        )
    except Exception:
        verbose_proxy_logger.debug(
            "Failed to emit AAWM route access log for %s adapter",
            adapter_label,
            exc_info=True,
        )


def _record_adapted_completed_route_rollup_turn(
    rollup_kwargs: dict[str, Any],
    *,
    adapter_label: str,
) -> None:
    try:
        record_aawm_route_rollup_turn(rollup_kwargs)
    except Exception:
        verbose_proxy_logger.debug(
            "Failed to record AAWM route rollup turn for %s adapter",
            adapter_label,
            exc_info=True,
        )


def _record_adapted_completed_route_rollup_after_stream(
    response: StreamingResponse,
    rollup_kwargs: dict[str, Any],
    *,
    adapter_label: str,
) -> StreamingResponse:
    original_iterator = response.body_iterator
    recorded = False

    async def _wrapped_iterator() -> Any:
        nonlocal recorded
        async for chunk in original_iterator:
            yield chunk
        if not recorded:
            recorded = True
            _record_adapted_completed_route_rollup_turn(
                rollup_kwargs,
                adapter_label=adapter_label,
            )

    response.body_iterator = _wrapped_iterator()
    return response


def _auto_agent_alias_model_rollup_label(event: dict[str, Any]) -> Optional[str]:
    model = _clean_codex_auth_value(event.get("model"))
    alias_model = _clean_codex_auth_value(event.get("alias_model"))
    if model and alias_model and model != alias_model:
        return f"{model}({alias_model})"
    return model or alias_model


def _auto_agent_alias_route_rollup_status(event: dict[str, Any]) -> Optional[str]:
    event_type = str(event.get("event_type") or "")
    candidate_status = str(event.get("candidate_status") or "")
    selection_reason = str(event.get("selection_reason") or "")
    failure_class = str(event.get("failure_class") or "")
    cooldown_scope = str(event.get("cooldown_scope") or "")
    if event_type == "no_candidate_available":
        return "Exhausted"
    if "auth_degraded" in candidate_status or "auth_degraded" in selection_reason:
        return "Degraded"
    # request-local / no-cooldown failures must not look like durable cool-downs.
    # Note: do not substring-match "cooldown" — retryable_no_cooldown contains it.
    if candidate_status == "retryable_no_cooldown" or cooldown_scope == "none":
        if event.get("error_status_code") or failure_class:
            return "Failed"
        return None
    if cooldown_scope == "request_local":
        if event.get("error_status_code") or failure_class or event.get("redispatch_required"):
            return "Failed"
        return None
    if candidate_status in {
        "cooldown_set",
        "terminal_in_flight_cooldown_set",
        "skipped_cooldown",
    } or (
        candidate_status.startswith("skipped_")
        and "cooldown" in candidate_status
        and "auth_degraded" not in candidate_status
    ):
        return "Cooling Down"
    if cooldown_scope == "candidate" or (event.get("redispatch_required") and cooldown_scope != "request_local"):
        return "Cooling Down"
    if failure_class in {"rate_limited", "capacity_exhausted", "transient_error"}:
        return "Cooling Down"
    if event.get("error_status_code") or failure_class:
        return "Failed"
    return None


def _auto_agent_alias_route_status_message(event: dict[str, Any]) -> str:
    parts: list[str] = []
    source_error = _clean_codex_auth_value(event.get("source_error"))
    if source_error is not None:
        parts.append(f"source_error={source_error}")
    for key in (
        "failure_class",
        "error_type",
        "error_code",
        "error_status_code",
        "candidate_status",
        "selection_reason",
    ):
        value = event.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    error_tokens = event.get("error_tokens")
    if isinstance(error_tokens, list) and error_tokens:
        parts.append("error_tokens={}".format(",".join(str(v) for v in error_tokens[:5])))
    return "; ".join(parts) or "route status changed"


def _resolve_auto_agent_alias_route_host_attribution(
    request: Request,
) -> dict[str, Optional[str]]:
    """Non-blocking host attribution for sync audit/event builders (RR-054 #4).

    Reverse-DNS is never performed inline on the event loop: the shared resolver
    defaults to ``allow_blocking_lookup=False`` and schedules background enrichment.
    Async request paths that need a full lookup should await
    ``_aresolve_auto_agent_alias_route_host_attribution``.
    """
    try:
        return resolve_aawm_route_host_attribution(
            request,
            allow_blocking_lookup=False,
        )
    except Exception:
        return {
            "client_ip": None,
            "client_ip_source": None,
            "host_name": None,
            "host_name_source": None,
        }


async def _aresolve_auto_agent_alias_route_host_attribution(
    request: Request,
) -> dict[str, Optional[str]]:
    """Async host attribution that offloads DNS via aresolve (RR-054 #4)."""
    try:
        return await aresolve_aawm_route_host_attribution(
            request,
            allow_blocking_lookup=True,
        )
    except Exception:
        return {
            "client_ip": None,
            "client_ip_source": None,
            "host_name": None,
            "host_name_source": None,
        }


def _build_auto_agent_alias_rollup_group_header_label(
    *,
    repository: Optional[str],
    client_product_label: Optional[str],
    host_name: Optional[str],
) -> Optional[str]:
    return build_aawm_route_rollup_group_header_label(
        repository=repository,
        client_product_label=client_product_label,
        host_name=host_name,
    )


def _resolve_auto_agent_alias_route_rollup_group_header_label(
    event: dict[str, Any],
) -> Optional[str]:
    group_header_label = _clean_codex_auth_value(event.get("rollup_group_header_label"))
    if not group_header_label:
        return None
    host_name = _clean_codex_auth_value(event.get("host_name"))
    if "@" in group_header_label or not host_name:
        return group_header_label
    return f"{group_header_label}@{host_name}"


def _record_auto_agent_alias_route_status_rollup(event: dict[str, Any]) -> None:
    status = _auto_agent_alias_route_rollup_status(event)
    if status is None:
        return
    alias_model = _clean_codex_auth_value(event.get("alias_model"))
    model_labels: list[str] = []
    model_label = _auto_agent_alias_model_rollup_label(event)
    if model_label:
        model_labels.append(model_label)
    candidates = event.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_model = _clean_codex_auth_value(candidate.get("model"))
            if candidate_model and alias_model and candidate_model != alias_model:
                candidate_model = f"{candidate_model}({alias_model})"
            if candidate_model and candidate_model not in model_labels:
                model_labels.append(candidate_model)
    if not model_labels:
        return

    message = _auto_agent_alias_route_status_message(event)
    for label in model_labels:
        emit_aawm_route_status_event(
            alias_model=alias_model,
            model_label=label.split("(", 1)[0],
            status=status,
            message=message,
        )

    group_header_label = _resolve_auto_agent_alias_route_rollup_group_header_label(event)
    incoming_endpoint = _clean_codex_auth_value(event.get("incoming_endpoint"))
    outgoing_target = (
        _clean_codex_auth_value(event.get("outgoing_target"))
        or _resolve_auto_agent_alias_route_rollup_outgoing_target(
            route_family=_clean_codex_auth_value(event.get("route_family")),
            target_url=event.get("target_url"),
        )
        or "candidate_selection"
    )
    if not group_header_label or not incoming_endpoint:
        return
    for label in model_labels:
        record_aawm_route_rollup(
            group_header_label=group_header_label,
            incoming_endpoint=incoming_endpoint,
            outgoing_target=outgoing_target,
            model_label=label,
            turns=0,
            status=status,
            message=_clean_codex_auth_value(event.get("source_error")),
        )


# ---------------------------------------------------------------------------
# Wave 5D: facade imports from audit_context / audit_build / audit_persist /
# audit_events extraction modules.  Each name below is the SAME object as the
# target module's definition.
# ---------------------------------------------------------------------------

# -- audit_persist logger compatibility binding --
# Same logger object used by audit_persist; bound here so existing tests that
# monkeypatch ``lpe.verbose_aawm_route_logger.info`` intercept live emission.
verbose_aawm_route_logger = _aawm_audit_persist.verbose_aawm_route_logger

# -- audit_persist facades --
_emit_auto_agent_alias_route_event = _aawm_audit_persist._emit_auto_agent_alias_route_event
_should_emit_auto_agent_alias_route_event = _aawm_audit_persist._should_emit_auto_agent_alias_route_event
_persist_auto_agent_alias_audit_only_events_best_effort = _aawm_audit_persist._persist_auto_agent_alias_audit_only_events_best_effort

# -- audit_context constants --
_AUTO_AGENT_ROLE_DECLARATION_RE = _aawm_audit_context._AUTO_AGENT_ROLE_DECLARATION_RE
_AUTO_AGENT_KNOWN_ROLE_NAMES = _aawm_audit_context._AUTO_AGENT_KNOWN_ROLE_NAMES
_AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES = _aawm_audit_context._AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES
_AUTO_AGENT_FILE_EDIT_TOOL_NAMES = _aawm_audit_context._AUTO_AGENT_FILE_EDIT_TOOL_NAMES
_AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY = _aawm_audit_context._AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY
_AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY = _aawm_audit_context._AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY

# -- audit_context facades --
_extract_auto_agent_alias_text_blobs = _aawm_audit_context._extract_auto_agent_alias_text_blobs
_extract_auto_agent_alias_role_from_text = _aawm_audit_context._extract_auto_agent_alias_role_from_text
_infer_auto_agent_alias_role_from_request_body = _aawm_audit_context._infer_auto_agent_alias_role_from_request_body
_iter_auto_agent_alias_metadata_dicts = _aawm_audit_context._iter_auto_agent_alias_metadata_dicts
_extract_auto_agent_alias_agent_dispatch_fields = _aawm_audit_context._extract_auto_agent_alias_agent_dispatch_fields
_walk_auto_agent_alias_prior_tool_activity = _aawm_audit_context._walk_auto_agent_alias_prior_tool_activity
_summarize_auto_agent_alias_actual_prior_tool_activity = _aawm_audit_context._summarize_auto_agent_alias_actual_prior_tool_activity
_classify_auto_agent_alias_terminal_activity_status = _aawm_audit_context._classify_auto_agent_alias_terminal_activity_status
_get_or_create_auto_agent_alias_request_call_id = _aawm_audit_context._get_or_create_auto_agent_alias_request_call_id
_AutoAgentAliasRequestContext = _aawm_audit_context._AutoAgentAliasRequestContext
_normalize_auto_agent_alias_request_context = _aawm_audit_context._normalize_auto_agent_alias_request_context
_clean_optional_string = _aawm_audit_context._clean_optional_string
_get_auto_agent_alias_request_context = _aawm_audit_context._get_auto_agent_alias_request_context
_attach_auto_agent_alias_terminal_context_fields = _aawm_audit_context._attach_auto_agent_alias_terminal_context_fields

# -- audit_build facades --
_is_auto_agent_alias_in_flight_cooldown_http_exception = _aawm_audit_build._is_auto_agent_alias_in_flight_cooldown_http_exception
_build_auto_agent_alias_audit_event = _aawm_audit_build._build_auto_agent_alias_audit_event
_build_auto_agent_alias_audit_events = _aawm_audit_build._build_auto_agent_alias_audit_events
_codex_auto_agent_request_has_continuation_state = _aawm_audit_build._codex_auto_agent_request_has_continuation_state

# -- audit_events facades --
_enrich_auto_agent_alias_terminal_event_from_attempts = _aawm_audit_events._enrich_auto_agent_alias_terminal_event_from_attempts
_emit_auto_agent_alias_no_candidate_event = _aawm_audit_events._emit_auto_agent_alias_no_candidate_event


_raise_codex_auto_agent_in_flight_cooldown = _aawm_selection._raise_codex_auto_agent_in_flight_cooldown


_build_auto_agent_redispatch_http_exception_detail = _aawm_selection._build_auto_agent_redispatch_http_exception_detail


_raise_codex_auto_agent_redispatch_required = _aawm_selection._raise_codex_auto_agent_redispatch_required


_get_codex_auto_agent_active_cooldown_state = _aawm_cooldown_state._get_codex_auto_agent_active_cooldown_state


_get_codex_auto_agent_active_cooldown_seconds = _aawm_cooldown_state._get_codex_auto_agent_active_cooldown_seconds


_set_codex_auto_agent_cooldown = _aawm_cooldown_state._set_codex_auto_agent_cooldown


_get_codex_auto_agent_session_affinity = _aawm_cooldown_state._get_codex_auto_agent_session_affinity


_set_codex_auto_agent_session_affinity = _aawm_cooldown_state._set_codex_auto_agent_session_affinity


_find_codex_auto_agent_candidate = _aawm_selection._find_codex_auto_agent_candidate


_find_codex_auto_agent_affinity_candidate = _aawm_selection._find_codex_auto_agent_affinity_candidate


_is_auto_agent_candidate_state_available = _aawm_selection._is_auto_agent_candidate_state_available


_build_auto_agent_skipped_candidates_from_states = _aawm_selection._build_auto_agent_skipped_candidates_from_states


_apply_codex_auto_agent_forced_candidate_cooldown = _aawm_selection._apply_codex_auto_agent_forced_candidate_cooldown


_apply_anthropic_auto_agent_forced_candidate_cooldown = _aawm_selection._apply_anthropic_auto_agent_forced_candidate_cooldown


_apply_codex_auto_agent_request_local_candidate_state = _aawm_selection._apply_codex_auto_agent_request_local_candidate_state


_apply_codex_auto_agent_adapter_local_candidate_cooldown = _aawm_selection._apply_codex_auto_agent_adapter_local_candidate_cooldown


_apply_kimi_code_managed_account_lane_cooldown = _aawm_selection._apply_kimi_code_managed_account_lane_cooldown


_build_codex_auto_agent_candidate_state = _aawm_selection._build_codex_auto_agent_candidate_state


_get_anthropic_auto_agent_candidate_cooldown_state = _aawm_selection._get_anthropic_auto_agent_candidate_cooldown_state


_build_anthropic_auto_agent_candidate_state = _aawm_selection._build_anthropic_auto_agent_candidate_state


_build_codex_auto_agent_candidate_states = _aawm_selection._build_codex_auto_agent_candidate_states


_attach_aawm_alias_routing_state_sources = _aawm_cooldown_state._attach_aawm_alias_routing_state_sources


_select_codex_auto_agent_candidate = _aawm_selection._select_codex_auto_agent_candidate

# Wave 5C: error_signals facades
_add_codex_auto_agent_text_error_tokens = _aawm_error_signals._add_codex_auto_agent_text_error_tokens
_build_codex_auto_agent_native_grok_continuation_retry_metadata = _aawm_error_signals._build_codex_auto_agent_native_grok_continuation_retry_metadata
_build_safe_kimi_code_selection_telemetry = _aawm_error_signals._build_safe_kimi_code_selection_telemetry
_classify_codex_auto_agent_retryable_exhaustion = _aawm_error_signals._classify_codex_auto_agent_retryable_exhaustion
_classify_kimi_code_auto_agent_probe_failure = _aawm_error_signals._classify_kimi_code_auto_agent_probe_failure
_codex_auto_agent_error_text = _aawm_error_signals._codex_auto_agent_error_text
_extract_codex_auto_agent_error_tokens = _aawm_error_signals._extract_codex_auto_agent_error_tokens
_extract_codex_auto_agent_error_type_and_code = _aawm_error_signals._extract_codex_auto_agent_error_type_and_code
_get_codex_auto_agent_candidate_cooldown_scope = _aawm_error_signals._get_codex_auto_agent_candidate_cooldown_scope
_get_codex_auto_agent_cooldown_scope = _aawm_error_signals._get_codex_auto_agent_cooldown_scope
_get_codex_auto_agent_cooldown_seconds = _aawm_error_signals._get_codex_auto_agent_cooldown_seconds
_get_codex_auto_agent_grok_account_quota_lane_cooldown_key = _aawm_error_signals._get_codex_auto_agent_grok_account_quota_lane_cooldown_key
_get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds = _aawm_error_signals._get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds
_get_codex_auto_agent_native_grok_continuation_transient_max_attempts = _aawm_error_signals._get_codex_auto_agent_native_grok_continuation_transient_max_attempts
_get_codex_auto_agent_source_error_summary = _aawm_error_signals._get_codex_auto_agent_source_error_summary
_get_kimi_code_managed_account_cooldown_key = _aawm_error_signals._get_kimi_code_managed_account_cooldown_key
_get_safe_kimi_code_probe_failure_metadata = _aawm_error_signals._get_safe_kimi_code_probe_failure_metadata
_is_codex_auto_agent_durable_cooldown_error_class = _aawm_error_signals._is_codex_auto_agent_durable_cooldown_error_class
_is_codex_auto_agent_grok_4_5_candidate = _aawm_error_signals._is_codex_auto_agent_grok_4_5_candidate
_is_codex_auto_agent_grok_account_quota_candidate = _aawm_error_signals._is_codex_auto_agent_grok_account_quota_candidate
_is_codex_auto_agent_grok_account_quota_exhaustion = _aawm_error_signals._is_codex_auto_agent_grok_account_quota_exhaustion
_is_codex_auto_agent_grok_build_usage_balance_exhausted = _aawm_error_signals._is_codex_auto_agent_grok_build_usage_balance_exhausted
_is_codex_auto_agent_grok_personal_team_spending_limit = _aawm_error_signals._is_codex_auto_agent_grok_personal_team_spending_limit
_is_codex_auto_agent_native_grok_4_5_candidate = _aawm_error_signals._is_codex_auto_agent_native_grok_4_5_candidate
_is_codex_auto_agent_native_grok_continuation_transient_retry_eligible = _aawm_error_signals._is_codex_auto_agent_native_grok_continuation_transient_retry_eligible
_is_codex_auto_agent_retryable_exhaustion = _aawm_error_signals._is_codex_auto_agent_retryable_exhaustion
_is_codex_auto_agent_spark_candidate = _aawm_error_signals._is_codex_auto_agent_spark_candidate
_is_codex_auto_agent_transient_internal_error_class = _aawm_error_signals._is_codex_auto_agent_transient_internal_error_class
_is_codex_auto_agent_xai_candidate = _aawm_error_signals._is_codex_auto_agent_xai_candidate
_is_kimi_code_auto_agent_candidate = _aawm_error_signals._is_kimi_code_auto_agent_candidate
_iter_codex_auto_agent_error_blocks = _aawm_error_signals._iter_codex_auto_agent_error_blocks
_parse_codex_auto_agent_header_wait_seconds = _aawm_error_signals._parse_codex_auto_agent_header_wait_seconds
_plan_codex_auto_agent_native_grok_continuation_transient_retry = _aawm_error_signals._plan_codex_auto_agent_native_grok_continuation_transient_retry
_KIMI_CODE_MANAGED_ACCOUNT_COOLDOWN_MODEL = _aawm_error_signals._KIMI_CODE_MANAGED_ACCOUNT_COOLDOWN_MODEL
_KIMI_CODE_SAFE_FAILURE_KINDS = _aawm_error_signals._KIMI_CODE_SAFE_FAILURE_KINDS
_KIMI_CODE_SAFE_FAILURE_SCOPES = _aawm_error_signals._KIMI_CODE_SAFE_FAILURE_SCOPES
_KIMI_CODE_SAFE_METADATA_GATES = _aawm_error_signals._KIMI_CODE_SAFE_METADATA_GATES
_KIMI_CODE_SAFE_RESET_REASONS = _aawm_error_signals._KIMI_CODE_SAFE_RESET_REASONS
_KIMI_CODE_SAFE_UPSTREAM_IDS = _aawm_error_signals._KIMI_CODE_SAFE_UPSTREAM_IDS

# Wave 5C: cooldown_apply facades
_apply_anthropic_auto_agent_alias_cooldown = _aawm_cooldown_apply._apply_anthropic_auto_agent_alias_cooldown
_apply_auto_agent_alias_cooldown = _aawm_cooldown_apply._apply_auto_agent_alias_cooldown
_apply_codex_auto_agent_alias_cooldown = _aawm_cooldown_apply._apply_codex_auto_agent_alias_cooldown
_apply_read_pilot_gated_cooldown = _aawm_cooldown_apply._apply_read_pilot_gated_cooldown
_persist_anthropic_cooldown_durable = _aawm_cooldown_apply._persist_anthropic_cooldown_durable
_persist_codex_cooldown_durable = _aawm_cooldown_apply._persist_codex_cooldown_durable
_resolve_auto_agent_cooldown_publication_plan = _aawm_cooldown_apply._resolve_auto_agent_cooldown_publication_plan
_set_codex_auto_agent_candidate_cooldowns = _aawm_cooldown_apply._set_codex_auto_agent_candidate_cooldowns

# Wave 5C: attempt_records facades
_add_anthropic_auto_agent_alias_metadata = _aawm_attempt_records._add_anthropic_auto_agent_alias_metadata
_add_codex_auto_agent_alias_metadata = _aawm_attempt_records._add_codex_auto_agent_alias_metadata
_extract_codex_reasoning_effort = _aawm_attempt_records._extract_codex_reasoning_effort
_get_codex_reasoning_effort_ceiling = _aawm_attempt_records._get_codex_reasoning_effort_ceiling
_normalize_codex_reasoning_effort_for_resolved_route = _aawm_attempt_records._normalize_codex_reasoning_effort_for_resolved_route
_record_auto_agent_alias_attempt_failure = _aawm_attempt_records._record_auto_agent_alias_attempt_failure
_record_auto_agent_alias_attempt_started = _aawm_attempt_records._record_auto_agent_alias_attempt_started
_record_read_pilot_cooldown_evidence = _aawm_attempt_records._record_read_pilot_cooldown_evidence
_update_codex_auto_agent_retryable_attempt_record = _aawm_attempt_records._update_codex_auto_agent_retryable_attempt_record



def _aawm_alias_route_verbose_json_enabled() -> bool:
    return os.getenv(_AAWM_ALIAS_ROUTE_VERBOSE_JSON_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "debug",
        "verbose",
    }


def _aawm_alias_route_healthy_json_enabled() -> bool:
    return os.getenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


_get_codex_auto_agent_request_local_cooldown_key = _aawm_selection._get_codex_auto_agent_request_local_cooldown_key


_get_codex_auto_agent_request_local_cooldown_state = _aawm_selection._get_codex_auto_agent_request_local_cooldown_state


_get_codex_auto_agent_request_local_cooldown_seconds = _aawm_selection._get_codex_auto_agent_request_local_cooldown_seconds


_set_codex_auto_agent_request_local_cooldown = _aawm_selection._set_codex_auto_agent_request_local_cooldown


_get_codex_auto_agent_request_local_excluded_keys = _aawm_selection._get_codex_auto_agent_request_local_excluded_keys


_exclude_codex_auto_agent_request_local_candidate = _aawm_selection._exclude_codex_auto_agent_request_local_candidate


_exclude_codex_auto_agent_request_local_candidate_without_cooldown = _aawm_selection._exclude_codex_auto_agent_request_local_candidate_without_cooldown


_apply_request_local_cooldown_from_plan = _aawm_selection._apply_request_local_cooldown_from_plan


_publish_codex_cooldown_memory = _aawm_cooldown_state._publish_codex_cooldown_memory


_publish_anthropic_cooldown_memory = _aawm_cooldown_state._publish_anthropic_cooldown_memory


_resolve_codex_auto_agent_xai_lane_key = _aawm_lane_keys._resolve_codex_auto_agent_xai_lane_key


_apply_codex_auto_agent_grok_account_lane_cooldown = _aawm_selection._apply_codex_auto_agent_grok_account_lane_cooldown


_normalize_anthropic_auto_agent_alias_model = _aawm_selection._normalize_anthropic_auto_agent_alias_model


def _is_anthropic_auto_agent_alias_model(model: Any) -> bool:
    return _normalize_anthropic_auto_agent_alias_model(model) is not None


def _resolve_anthropic_auto_agent_alias_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    return _normalize_anthropic_auto_agent_alias_model(request_body.get("model"))


_resolve_anthropic_auto_agent_native_lane_key = _aawm_lane_keys._resolve_anthropic_auto_agent_native_lane_key


_resolve_anthropic_auto_agent_native_cooldown_lane_key = _aawm_lane_keys._resolve_anthropic_auto_agent_native_cooldown_lane_key


def _resolve_anthropic_auto_agent_session_key(
    request: Request,
    request_body: dict[str, Any],
    *,
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    metadata_session_id = metadata.get("session_id") if isinstance(metadata, dict) else None
    session_id = _clean_codex_auth_value(metadata_session_id)
    headers = _safe_get_request_headers(request)
    if session_id is None:
        session_id = (
            _get_codex_auto_agent_header(headers, "session_id")
            or _get_codex_auto_agent_header(headers, "session-id")
            or _get_codex_auto_agent_header(headers, "x-session-id")
        )
    if session_id is None:
        return None
    if alias_model == _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS:
        return f"{session_id}:{_resolve_anthropic_auto_agent_native_lane_key(request)}"
    return f"{alias_model}:{session_id}:" f"{_resolve_anthropic_auto_agent_native_lane_key(request)}"


_format_merged_alias_family_cooldown_state_source = _aawm_cooldown_state._format_merged_alias_family_cooldown_state_source


_get_anthropic_auto_agent_merged_codex_openai_cooldown_state = _aawm_cooldown_state._get_anthropic_auto_agent_merged_codex_openai_cooldown_state


_get_anthropic_auto_agent_active_cooldown_state = _aawm_cooldown_state._get_anthropic_auto_agent_active_cooldown_state


_get_anthropic_auto_agent_active_cooldown_seconds = _aawm_cooldown_state._get_anthropic_auto_agent_active_cooldown_seconds


_set_anthropic_auto_agent_cooldown = _aawm_cooldown_state._set_anthropic_auto_agent_cooldown


_get_anthropic_auto_agent_session_affinity = _aawm_cooldown_state._get_anthropic_auto_agent_session_affinity


_set_anthropic_auto_agent_session_affinity = _aawm_cooldown_state._set_anthropic_auto_agent_session_affinity


_find_anthropic_auto_agent_candidate = _aawm_selection._find_anthropic_auto_agent_candidate


_build_anthropic_auto_agent_candidate_states = _aawm_selection._build_anthropic_auto_agent_candidate_states


_raise_anthropic_auto_agent_in_flight_cooldown = _aawm_selection._raise_anthropic_auto_agent_in_flight_cooldown


_raise_anthropic_auto_agent_redispatch_required = _aawm_selection._raise_anthropic_auto_agent_redispatch_required


_select_anthropic_auto_agent_candidate = _aawm_selection._select_anthropic_auto_agent_candidate


_resolve_anthropic_openai_responses_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_openai_responses_adapter_model


_resolve_anthropic_xai_oauth_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_xai_oauth_adapter_model


_resolve_anthropic_grok_native_oauth_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_grok_native_oauth_adapter_model


_resolve_anthropic_openrouter_completion_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_openrouter_completion_adapter_model


_resolve_anthropic_nvidia_responses_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_nvidia_responses_adapter_model


_resolve_anthropic_openrouter_responses_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_openrouter_responses_adapter_model


_resolve_anthropic_google_completion_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_google_completion_adapter_model


_get_anthropic_adapter_google_auth_file_path = _aawm_google_oauth._get_anthropic_adapter_google_auth_file_path


_extract_google_oauth_client_values_from_bundle_text = (
    _aawm_google_oauth._extract_google_oauth_client_values_from_bundle_text
)


_add_google_cli_bundle_candidate_files = _aawm_google_oauth._add_google_cli_bundle_candidate_files


_iter_google_oauth_client_bundle_candidates = _aawm_google_oauth._iter_google_oauth_client_bundle_candidates


_load_google_oauth_client_values_from_local_gemini_cli_bundle = (
    _aawm_google_oauth._load_google_oauth_client_values_from_local_gemini_cli_bundle
)


_load_local_google_oauth_credentials = _aawm_google_oauth._load_local_google_oauth_credentials


_google_oauth_token_is_valid = _aawm_google_oauth._google_oauth_token_is_valid


_google_oauth_cached_token_is_valid = _aawm_google_oauth._google_oauth_cached_token_is_valid


_get_google_oauth_expiry_date = _aawm_google_oauth._get_google_oauth_expiry_date


_get_google_oauth_client_value = _aawm_google_oauth._get_google_oauth_client_value


_refresh_local_google_oauth_credentials = _aawm_google_oauth._refresh_local_google_oauth_credentials


_load_valid_local_google_oauth_access_token = _aawm_google_oauth._load_valid_local_google_oauth_access_token


def _extract_google_adapter_agent_name_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_adapter_agent_name_from_completion_messages(completion_messages)


def _extract_google_adapter_latest_user_prompt_text(completion_messages: list[dict[str, Any]]) -> Optional[str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_adapter_latest_user_prompt_text(completion_messages)


def _extract_google_adapter_latest_tool_result_fingerprint(completion_messages: list[dict[str, Any]]) -> Optional[str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_adapter_latest_tool_result_fingerprint(completion_messages)


def _resolve_google_adapter_session_id(
    request: Request,
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
) -> tuple[str, str]:
    direct_session_id = (
        _get_request_header_or_passthrough_alias(request, "session_id")
        or _safe_get_request_headers(request).get("x-claude-code-session-id")
        or _safe_get_request_headers(request).get("X-Claude-Code-Session-Id")
    )
    trace_id = (
        _get_request_header_or_passthrough_alias(request, "langfuse_trace_id")
        or _get_request_header_or_passthrough_alias(request, "langfuse_existing_trace_id")
        or _get_request_header_or_passthrough_alias(request, "trace_id")
    )
    trace_name = _get_request_header_or_passthrough_alias(request, "langfuse_trace_name")
    agent_name = _extract_google_adapter_agent_name_from_completion_messages(completion_messages)

    if isinstance(direct_session_id, str) and direct_session_id:
        seed = f"direct_session_id:{direct_session_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed)), "direct_session_id"

    identity_name = None
    identity_source = None
    if isinstance(trace_name, str) and trace_name:
        identity_name = trace_name
        identity_source = "trace_name"
    elif isinstance(agent_name, str) and agent_name:
        identity_name = agent_name
        identity_source = "agent_name"

    if identity_name:
        seed = f"{identity_source}:{identity_name}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed)), identity_source or "derived"

    if isinstance(trace_id, str) and trace_id:
        seed = f"trace_id:{trace_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed)), "trace_id"

    return str(uuid4()), "generated_uuid"


def _resolve_google_adapter_user_prompt_id(
    request: Request,
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
    session_id: str,
) -> str:
    trace_id = (
        _get_request_header_or_passthrough_alias(request, "langfuse_trace_id")
        or _get_request_header_or_passthrough_alias(request, "langfuse_existing_trace_id")
        or _get_request_header_or_passthrough_alias(request, "trace_id")
    )
    if isinstance(trace_id, str) and trace_id:
        seed = f"user_prompt_trace_id:{trace_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed))

    latest_tool_result = _extract_google_adapter_latest_tool_result_fingerprint(completion_messages)
    if isinstance(latest_tool_result, str) and latest_tool_result:
        seed = f"user_prompt_tool_result:{latest_tool_result}|" f"session_id:{session_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed))

    latest_user_prompt = _extract_google_adapter_latest_user_prompt_text(completion_messages)
    if isinstance(latest_user_prompt, str) and latest_user_prompt:
        prompt_hash = hashlib.sha1(latest_user_prompt.encode("utf-8")).hexdigest()[:16]
        seed = f"user_prompt_hash:{prompt_hash}|session_id:{session_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed))

    seed = f"user_prompt_session:{session_id}|model:{google_model}"
    return str(uuid5(NAMESPACE_URL, seed))


def _build_code_assist_adapter_native_headers(
    *,
    adapter_provider: str,
    access_token: str,
    model: Optional[str],
    accept: str,
) -> dict[str, str]:
    if adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER:
        headers = _build_antigravity_native_headers(access_token)
        headers["Accept"] = accept
        return headers
    return _build_google_adapter_native_headers(
        access_token=access_token,
        model=model,
        accept=accept,
    )


def _get_code_assist_adapter_target_base(adapter_provider: str) -> str:
    if adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER:
        return _get_antigravity_passthrough_target_base()
    return _get_anthropic_adapter_google_target_base()


async def _get_or_load_google_code_assist_project(
    access_token: str,
    *,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> str:
    return await _anthropic_google_process_cache._get_or_load_google_code_assist_project(
        access_token,
        runtime=_get_anthropic_google_process_cache_runtime(),
        adapter_provider=adapter_provider,
    )


_get_google_code_assist_prime_ttl_seconds = _google_env_policy._get_google_code_assist_prime_ttl_seconds


_get_google_code_assist_prime_cache_key = _google_env_policy._get_google_code_assist_prime_cache_key


_get_google_adapter_max_concurrent = _google_env_policy._get_google_adapter_max_concurrent


_get_google_adapter_shared_lane_key = _google_env_policy._get_google_adapter_shared_lane_key


_get_google_adapter_rate_limit_key = _google_env_policy._get_google_adapter_rate_limit_key


_get_google_adapter_rate_limit_key_from_kwargs = _google_env_policy._get_google_adapter_rate_limit_key_from_kwargs


def _get_google_adapter_semaphore(
    model: Optional[str] = None,
    *,
    access_token: Optional[str] = None,
    companion_project: Optional[str] = None,
    rate_limit_key: Optional[str] = None,
) -> asyncio.Semaphore:
    return _anthropic_google_process_cache._get_google_adapter_semaphore(
        model,
        runtime=_get_anthropic_google_process_cache_runtime(),
        access_token=access_token,
        companion_project=companion_project,
        rate_limit_key=rate_limit_key,
    )


_get_google_adapter_max_retries = _google_env_policy._get_google_adapter_max_retries


_coerce_non_negative_int = _google_env_policy._coerce_non_negative_int


_coerce_non_negative_float = _google_env_policy._coerce_non_negative_float


_get_google_adapter_post_tool_cooldown_seconds = _google_env_policy._get_google_adapter_post_tool_cooldown_seconds


_google_code_assist_unwrapped_chunk_contains_tool_call = _google_env_policy._google_code_assist_unwrapped_chunk_contains_tool_call


_get_google_adapter_max_output_tokens_cap = _google_env_policy._get_google_adapter_max_output_tokens_cap


_get_google_adapter_default_thinking_level = _google_env_policy._get_google_adapter_default_thinking_level


_get_google_adapter_max_contents_window = _google_env_policy._get_google_adapter_max_contents_window


_get_google_adapter_max_contents_text_chars = _google_env_policy._get_google_adapter_max_contents_text_chars


_estimate_google_content_text_chars = _aawm_persisted_output._estimate_google_content_text_chars


_google_content_has_text = _google_env_policy._google_content_has_text


_google_content_has_function_exchange = _google_context_window._google_content_has_function_exchange


_google_content_has_function_call = _google_context_window._google_content_has_function_call


def _google_content_function_call_ids(content_block: Any) -> set[str]:
    if not isinstance(content_block, dict):
        return set()
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return set()
    function_call_ids: set[str] = set()
    for part in parts:
        if not isinstance(part, dict):
            continue
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            function_call = part.get("function_call")
        if not isinstance(function_call, dict):
            continue
        function_call_id = function_call.get("id")
        if isinstance(function_call_id, str) and function_call_id.strip():
            function_call_ids.add(function_call_id.strip())
    return function_call_ids


def _google_content_function_response_ids(content_block: Any) -> set[str]:
    if not isinstance(content_block, dict):
        return set()
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return set()
    function_response_ids: set[str] = set()
    for part in parts:
        if not isinstance(part, dict):
            continue
        function_response = part.get("functionResponse")
        if not isinstance(function_response, dict):
            function_response = part.get("function_response")
        if not isinstance(function_response, dict):
            continue
        response_payload = function_response.get("response")
        nested_tool_use_id = response_payload.get("tool_use_id") if isinstance(response_payload, dict) else None
        for candidate in (function_response.get("id"), nested_tool_use_id):
            if isinstance(candidate, str) and candidate.strip():
                function_response_ids.add(candidate.strip())
    return function_response_ids


def _selected_google_contents_have_paired_function_responses(contents: list[Any], selected_indices: list[int]) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._selected_google_contents_have_paired_function_responses(
        contents, selected_indices
    )


def _selected_google_contents_have_complete_function_exchanges(
    contents: list[Any], selected_indices: list[int]
) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._selected_google_contents_have_complete_function_exchanges(
        contents, selected_indices
    )


def _find_prior_google_function_call_content_index(
    contents: list[Any],
    *,
    before_index: int,
    function_response_id: str,
) -> Optional[int]:
    for index in range(before_index - 1, -1, -1):
        if function_response_id in _google_content_function_call_ids(contents[index]):
            return index
    return None


def _add_required_google_function_call_pair_indices(contents: list[Any], selected_indices: list[int]) -> list[int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._add_required_google_function_call_pair_indices(contents, selected_indices)


def _trim_google_content_indices_to_window(
    contents: list[Any], selected_indices: list[int], *, protected_text_indices: set[int], max_window: int
) -> list[int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._trim_google_content_indices_to_window(
        contents, selected_indices, protected_text_indices=protected_text_indices, max_window=max_window
    )


_get_google_adapter_oversized_text_part_char_cap = _google_env_policy._get_google_adapter_oversized_text_part_char_cap


_get_google_adapter_pure_context_text_part_char_cap = _google_env_policy._get_google_adapter_pure_context_text_part_char_cap


_get_google_adapter_subagent_context_text_part_char_cap = _google_env_policy._get_google_adapter_subagent_context_text_part_char_cap


_get_google_adapter_followup_subagent_context_text_part_char_cap = _google_env_policy._get_google_adapter_followup_subagent_context_text_part_char_cap


_get_google_adapter_followup_allowed_tool_names = _google_env_policy._get_google_adapter_followup_allowed_tool_names


def _request_block_has_google_function_response(request_block: dict[str, Any]) -> bool:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return False
    for item in contents:
        if not isinstance(item, dict):
            continue
        parts = item.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("functionResponse"), dict) or isinstance(part.get("function_response"), dict):
                return True
    return False


def _trim_google_adapter_followup_tools(request_block: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._trim_google_adapter_followup_tools(request_block)


def _is_google_function_call_allowed_predecessor(content_block: Any) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._is_google_function_call_allowed_predecessor(content_block)


def _merge_google_model_content_parts(first_content: dict[str, Any], second_content: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._merge_google_model_content_parts(first_content, second_content)


def _google_adapter_function_call_anchor_content() -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {"text": ("[Gemini adapter inserted a conversation boundary before " "a preserved historical tool call.]")}
        ],
    }


def _repair_google_adapter_function_call_turn_adjacency(request_block: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._repair_google_adapter_function_call_turn_adjacency(request_block)


def _split_google_adapter_inline_context_and_prompt(request_block: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._split_google_adapter_inline_context_and_prompt(request_block)


def _compact_google_adapter_oversized_text_part(
    part: Any, *, cap: int, pure_context_cap: int, head_keep: int, tail_keep: int, is_followup_request: bool
) -> tuple[Any, bool, dict[str, int]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_oversized_text_part(
        part,
        cap=cap,
        pure_context_cap=pure_context_cap,
        head_keep=head_keep,
        tail_keep=tail_keep,
        is_followup_request=is_followup_request,
    )


def _compact_google_adapter_oversized_text_parts(request_block: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._compact_google_adapter_oversized_text_parts(request_block)


_apply_google_adapter_contents_window_policy = _google_context_window._apply_google_adapter_contents_window_policy


def _apply_google_adapter_generation_config_policy(
    request_block: dict[str, Any], *, model: Optional[str]
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_adapter_generation_config_policy(request_block, model=model)


def _apply_google_adapter_request_shape_policy(payload: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_adapter_request_shape_policy(payload)


_extract_google_adapter_exception_status_code = _google_error_signals._extract_google_adapter_exception_status_code


_extract_google_adapter_exception_detail = _google_error_signals._extract_google_adapter_exception_detail


def _extract_adapter_upstream_headers(exc: Any) -> dict[str, Any]:
    upstream_headers = getattr(exc, "upstream_headers", None)
    if isinstance(upstream_headers, dict):
        return {
            str(header_name): header_value
            for header_name, header_value in upstream_headers.items()
            if header_value is not None
        }
    response = getattr(exc, "response", None)
    response_headers = getattr(response, "headers", None)
    if response_headers is None:
        return {}
    return {str(header_name): str(header_value) for header_name, header_value in response_headers.items()}


def _get_adapter_header_value(headers: dict[str, Any], header_name: str) -> Optional[str]:
    if not headers:
        return None
    for key, value in headers.items():
        if not isinstance(key, str):
            continue
        if key.lower() != header_name.lower():
            continue
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)
    return None


def _parse_retry_after_seconds_from_headers(headers: dict[str, Any]) -> Optional[float]:
    retry_after_value = _get_adapter_header_value(headers, "Retry-After")
    if retry_after_value is None:
        return None
    try:
        return max(0.0, float(retry_after_value))
    except Exception:
        return None


def _parse_rate_limit_reset_wait_seconds_from_headers(headers: dict[str, Any]) -> Optional[float]:
    reset_value = _get_adapter_header_value(headers, "X-RateLimit-Reset")
    if reset_value is None:
        return None
    try:
        reset_number = float(reset_value)
    except Exception:
        return None
    if reset_number > 1_000_000_000_000:
        reset_epoch_seconds = reset_number / 1000.0
    else:
        reset_epoch_seconds = reset_number
    return max(0.0, reset_epoch_seconds - time.time())


_parse_google_rate_limit_reset_seconds = _google_error_signals._parse_google_rate_limit_reset_seconds


def _extract_embedded_json_payload_candidates(detail: object) -> list[str]:
    """Shared exception-detail JSON/bytes extraction (RR-054 #59)."""
    if isinstance(detail, dict):
        try:
            return [json.dumps(detail)]
        except Exception:
            return [str(detail)]
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    else:
        detail_text = str(detail or "")
    candidates: list[str] = [detail_text]
    brace_start = detail_text.find("{")
    brace_end = detail_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(detail_text[brace_start : brace_end + 1])
    bracket_start = detail_text.find("[")
    bracket_end = detail_text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        candidates.append(detail_text[bracket_start : bracket_end + 1])
    bytes_literal_match = re.search(r'b([\'"]).*', detail_text, re.DOTALL)
    if bytes_literal_match is not None:
        try:
            literal_value = ast.literal_eval(bytes_literal_match.group(0))
            if isinstance(literal_value, bytes):
                candidates.append(literal_value.decode("utf-8", errors="ignore"))
            else:
                candidates.append(str(literal_value))
        except Exception:
            pass
    # openrouter-style ": b'...'" wrappers
    if ": b'" in detail_text or ': b"' in detail_text:
        tail = detail_text.split(": ", 1)[-1].strip()
        if (tail.startswith("b'") and tail.endswith("'")) or (tail.startswith('b"') and tail.endswith('"')):
            try:
                literal_value = ast.literal_eval(tail)
                if isinstance(literal_value, bytes):
                    candidates.append(literal_value.decode("utf-8", errors="ignore"))
                elif isinstance(literal_value, str):
                    candidates.append(literal_value)
            except Exception:
                pass
    return candidates


def _parse_json_payloads_from_text_candidates(
    candidates: list[str],
) -> list[object]:
    parsed_payloads: list[object] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        parsed_payloads.append(parsed)
    return parsed_payloads


_extract_google_adapter_error_payloads = _google_error_signals._extract_google_adapter_error_payloads


_extract_google_adapter_error_reason = _google_error_signals._extract_google_adapter_error_reason


_extract_google_adapter_error_payload_for_logging = _google_error_signals._extract_google_adapter_error_payload_for_logging


_record_google_adapter_error_for_logging = _google_error_signals._record_google_adapter_error_for_logging


_get_google_adapter_model_capacity_max_retries = _google_env_policy._get_google_adapter_model_capacity_max_retries


_get_google_adapter_capacity_backoff_seconds = _google_env_policy._get_google_adapter_capacity_backoff_seconds


_get_google_adapter_hidden_retry_budget_seconds = _google_env_policy._get_google_adapter_hidden_retry_budget_seconds


_get_google_adapter_transient_retry_max_attempts = _google_env_policy._get_google_adapter_transient_retry_max_attempts


_get_google_adapter_transient_backoff_seconds = _google_env_policy._get_google_adapter_transient_backoff_seconds


def _is_google_adapter_transient_retryable_failure(
    exc: Any, *, status_code: Optional[int], error_reason: Optional[str]
) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._is_google_adapter_transient_retryable_failure(
        exc, status_code=status_code, error_reason=error_reason
    )


_build_google_adapter_terminal_error_log_context = _google_error_signals._build_google_adapter_terminal_error_log_context


_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES = (
    _google_retry_runtime._GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES
)

_google_adapter_hidden_retry_kwargs_from_passthrough_kwargs = (
    _google_retry_runtime._google_adapter_hidden_retry_kwargs_from_passthrough_kwargs
)

_record_google_adapter_hidden_retry_metadata = (
    _google_retry_runtime._record_google_adapter_hidden_retry_metadata
)

_record_google_adapter_terminal_transient_failure_metadata = (
    _google_retry_runtime._record_google_adapter_terminal_transient_failure_metadata
)

_google_adapter_hidden_retry_metadata = (
    _google_retry_runtime._google_adapter_hidden_retry_metadata
)

_record_google_adapter_success_after_transient_retry = (
    _google_retry_runtime._record_google_adapter_success_after_transient_retry
)

_log_google_adapter_terminal_transient_failure = (
    _google_retry_runtime._log_google_adapter_terminal_transient_failure
)

_wait_for_google_adapter_cooldown_if_needed = (
    _google_retry_runtime._wait_for_google_adapter_cooldown_if_needed
)

_set_google_adapter_cooldown = _google_retry_runtime._set_google_adapter_cooldown

_handle_google_adapter_rate_limit_failure = (
    _google_retry_runtime._handle_google_adapter_rate_limit_failure
)

_handle_google_adapter_transient_failure = (
    _google_retry_runtime._handle_google_adapter_transient_failure
)

_perform_google_adapter_pass_through_request = (
    _google_retry_runtime._perform_google_adapter_pass_through_request
)

_google_retry_runtime.configure_google_retry_runtime(
    _google_retry_runtime.Runtime(
        process_cache_runtime=_get_anthropic_google_process_cache_runtime(),
        rate_limit=_alias_routing_state.google_rate_limit,
        get_rate_limit_key_from_kwargs=lambda kwargs: (
            _get_google_adapter_rate_limit_key_from_kwargs(kwargs)
        ),
        get_max_retries=lambda: _get_google_adapter_max_retries(),
        coerce_non_negative_int=lambda value, default: (
            _coerce_non_negative_int(value, default)
        ),
        coerce_non_negative_float=lambda value, default: (
            _coerce_non_negative_float(value, default)
        ),
        get_model_capacity_max_retries=lambda: (
            _get_google_adapter_model_capacity_max_retries()
        ),
        get_capacity_backoff_seconds=lambda attempt: (
            _get_google_adapter_capacity_backoff_seconds(attempt)
        ),
        get_hidden_retry_budget_seconds=lambda: (
            _get_google_adapter_hidden_retry_budget_seconds()
        ),
        get_transient_retry_max_attempts=lambda: (
            _get_google_adapter_transient_retry_max_attempts()
        ),
        get_transient_backoff_seconds=lambda attempt: (
            _get_google_adapter_transient_backoff_seconds(attempt)
        ),
        extract_exception_status_code=lambda exc: (
            _extract_google_adapter_exception_status_code(exc)
        ),
        extract_error_reason=lambda exc: (
            _extract_google_adapter_error_reason(exc)
        ),
        parse_rate_limit_reset_seconds=lambda exc: (
            _parse_google_rate_limit_reset_seconds(exc)
        ),
        is_transient_retryable_failure=lambda exc, **kwargs: (
            _is_google_adapter_transient_retryable_failure(exc, **kwargs)
        ),
        classify_hidden_retry_failure=lambda exc: (
            _classify_passthrough_hidden_retry_failure(exc)
        ),
        record_error_for_logging=lambda passthrough_kwargs, **kwargs: (
            _record_google_adapter_error_for_logging(passthrough_kwargs, **kwargs)
        ),
        record_hidden_retry_metadata=lambda kwargs, **kw: (
            _record_passthrough_hidden_retry_metadata(kwargs, **kw)
        ),
        build_terminal_error_log_context=lambda passthrough_kwargs, **kwargs: (
            _build_google_adapter_terminal_error_log_context(
                passthrough_kwargs, **kwargs
            )
        ),
        pass_through_request=lambda **kwargs: pass_through_request(**kwargs),
        bound_token_cache=lambda cache: _bound_google_adapter_token_cache(cache),
        sleep=lambda seconds: asyncio.sleep(seconds),
        log_debug=verbose_proxy_logger.debug,
        log_warning=verbose_proxy_logger.warning,
        log_error=verbose_proxy_logger.error,
        host_globals=globals(),
    )
)

# Wave 6C Phase 2: Google Code Assist extraction install
_google_codex_code_assist.install(globals())

_ANTHROPIC_OPENROUTER_RETRY_TRANSPORT_RUNTIME = _anthropic_openrouter_retry_transport.Runtime(
    rate_limit=_alias_routing_state.openrouter_rate_limit,
    failure_circuit_until_monotonic_by_key=(_openrouter_adapter_failure_circuit_until_monotonic_by_key),
    clean_secret_string=lambda value: _clean_secret_string(value),
    extract_embedded_json_payload_candidates=(_extract_embedded_json_payload_candidates),
    parse_json_payloads_from_text_candidates=lambda values: (_parse_json_payloads_from_text_candidates(list(values))),
    extract_upstream_headers=_extract_adapter_upstream_headers,
    parse_retry_after_seconds_from_headers=lambda headers: (_parse_retry_after_seconds_from_headers(dict(headers))),
    get_header_value=lambda headers, name: _get_adapter_header_value(
        dict(headers),
        name,
    ),
    parse_reset_wait_seconds_from_headers=lambda headers: (
        _parse_rate_limit_reset_wait_seconds_from_headers(dict(headers))
    ),
    raise_candidate_unavailable=(_raise_openrouter_auto_agent_candidate_unavailable),
    maybe_raise_alias_probe_cooldown=(_maybe_raise_openrouter_adapter_alias_probe_cooldown),
    get_completion_model=_get_openrouter_completion_adapter_upstream_model,
    pass_through_request=lambda **kwargs: pass_through_request(**kwargs),
    wait_for_cooldown=lambda *args, **kwargs: (_wait_for_openrouter_adapter_cooldown_if_needed(*args, **kwargs)),
    set_cooldown_callback=lambda *args, **kwargs: (_set_openrouter_adapter_cooldown(*args, **kwargs)),
    maybe_raise_failure_circuit_open_callback=lambda *args, **kwargs: (
        _maybe_raise_openrouter_adapter_failure_circuit_open(*args, **kwargs)
    ),
    open_failure_circuit_callback=lambda *args, **kwargs: (_openrouter_adapter_open_failure_circuit(*args, **kwargs)),
    clear_failure_circuit_callback=lambda model: (_clear_openrouter_adapter_failure_circuit(model)),
    log_debug=verbose_proxy_logger.debug,
    log_warning=verbose_proxy_logger.warning,
    getenv=lambda name: os.getenv(name),
    sleep=lambda seconds: asyncio.sleep(seconds),
    monotonic=lambda: time.monotonic(),
)


_get_openrouter_adapter_rate_limit_key = _wave6b_openrouter_runtime._get_openrouter_adapter_rate_limit_key


_is_openrouter_adapter_free_model = _wave6b_openrouter_runtime._is_openrouter_adapter_free_model


_get_openrouter_adapter_wait_keys = _wave6b_openrouter_runtime._get_openrouter_adapter_wait_keys


_extract_openrouter_adapter_exception_status_code = _wave6b_openrouter_runtime._extract_openrouter_adapter_exception_status_code


_extract_openrouter_adapter_error_payload = _wave6b_openrouter_runtime._extract_openrouter_adapter_error_payload


_extract_openrouter_adapter_provider_name = _wave6b_openrouter_runtime._extract_openrouter_adapter_provider_name


_extract_openrouter_adapter_retry_after_seconds = _wave6b_openrouter_runtime._extract_openrouter_adapter_retry_after_seconds


_extract_openrouter_adapter_raw_message = _wave6b_openrouter_runtime._extract_openrouter_adapter_raw_message


_is_openrouter_adapter_no_endpoint_candidate_error = _wave6b_openrouter_runtime._is_openrouter_adapter_no_endpoint_candidate_error


_maybe_raise_openrouter_adapter_alias_probe_no_endpoint_unavailable = _wave6b_openrouter_runtime._maybe_raise_openrouter_adapter_alias_probe_no_endpoint_unavailable


_is_openrouter_adapter_provider_raw_error = _wave6b_openrouter_runtime._is_openrouter_adapter_provider_raw_error


_extract_openrouter_adapter_error_headers = _wave6b_openrouter_runtime._extract_openrouter_adapter_error_headers


_get_openrouter_adapter_header_value = _wave6b_openrouter_runtime._get_openrouter_adapter_header_value


_extract_openrouter_adapter_reset_wait_seconds = _wave6b_openrouter_runtime._extract_openrouter_adapter_reset_wait_seconds


_is_openrouter_adapter_long_window_rate_limit = _wave6b_openrouter_runtime._is_openrouter_adapter_long_window_rate_limit


_get_openrouter_adapter_cooldown_keys = _wave6b_openrouter_runtime._get_openrouter_adapter_cooldown_keys


_get_openrouter_adapter_retry_wait_seconds = _wave6b_openrouter_runtime._get_openrouter_adapter_retry_wait_seconds


_get_openrouter_adapter_max_retries = _wave6b_openrouter_runtime._get_openrouter_adapter_max_retries


_get_openrouter_adapter_backoff_seconds = _wave6b_openrouter_runtime._get_openrouter_adapter_backoff_seconds


_get_openrouter_adapter_hidden_retry_budget_seconds = _wave6b_openrouter_runtime._get_openrouter_adapter_hidden_retry_budget_seconds


_get_openrouter_adapter_post_failure_cooldown_seconds = _wave6b_openrouter_runtime._get_openrouter_adapter_post_failure_cooldown_seconds


_maybe_raise_openrouter_adapter_failure_circuit_open = _wave6b_openrouter_runtime._maybe_raise_openrouter_adapter_failure_circuit_open


_openrouter_adapter_open_failure_circuit = _wave6b_openrouter_runtime._openrouter_adapter_open_failure_circuit


_clear_openrouter_adapter_failure_circuit = _wave6b_openrouter_runtime._clear_openrouter_adapter_failure_circuit


_get_openrouter_adapter_active_cooldown_seconds = _wave6b_openrouter_runtime._get_openrouter_adapter_active_cooldown_seconds

# Wave 5A: bind quota state and adapter helpers into openrouter_quota.
def _get_openrouter_free_daily_quota_cache() -> tuple[Optional[float], float]:
    return _alias_routing_state.get_openrouter_free_quota_cache()


def _set_openrouter_free_daily_quota_cache(value: tuple[Optional[float], float]) -> None:
    _alias_routing_state.set_openrouter_free_quota_cache(value)


async def _fetch_openrouter_quota_row_via_facade():
    """Indirection so monkeypatching ``_fetch_openrouter_free_daily_quota_row``
    on this module is visible to the openrouter_quota module."""
    return await _fetch_openrouter_free_daily_quota_row()


_aawm_openrouter_quota.configure_openrouter_quota_runtime(
    get_quota_cache=_get_openrouter_free_daily_quota_cache,
    set_quota_cache=_set_openrouter_free_daily_quota_cache,
    quota_lock=_openrouter_free_daily_quota_lock,
    get_dynamic_injection_pool=_get_aawm_dynamic_injection_pool,
    get_adapter_active_cooldown_seconds=_get_openrouter_adapter_active_cooldown_seconds,
    get_adapter_rate_limit_key=_get_openrouter_adapter_rate_limit_key,
    fetch_quota_row=_fetch_openrouter_quota_row_via_facade,
    get_free_daily_quota_exhausted_cooldown_seconds=(
        lambda: globals()[
            "_get_openrouter_free_daily_quota_exhausted_cooldown_seconds"
        ]()
    ),
)

# Wave 5B: bind the state manager into cooldown_state.
_aawm_cooldown_state.configure_cooldown_state_runtime(
    manager=_alias_routing_state,
)

# Wave 5B: bind god-module / cooldown_state dependencies into selection.
# Late-binding lambdas ensure the god-module names are resolved at call time.
_aawm_selection.configure_selection_runtime(
    get_codex_active_cooldown_state=lambda *a, **kw: _get_codex_auto_agent_active_cooldown_state(*a, **kw),
    get_anthropic_active_cooldown_state=lambda *a, **kw: _get_anthropic_auto_agent_active_cooldown_state(*a, **kw),
    get_anthropic_merged_codex_openai_cooldown_state=lambda *a, **kw: _get_anthropic_auto_agent_merged_codex_openai_cooldown_state(*a, **kw),
    set_codex_cooldown=lambda *a, **kw: _set_codex_auto_agent_cooldown(*a, **kw),
    set_anthropic_cooldown=lambda *a, **kw: _set_anthropic_auto_agent_cooldown(*a, **kw),
    get_codex_session_affinity=lambda *a, **kw: _get_codex_auto_agent_session_affinity(*a, **kw),
    get_anthropic_session_affinity=lambda *a, **kw: _get_anthropic_auto_agent_session_affinity(*a, **kw),
    resolve_google_lane_key=lambda *a, **kw: _resolve_codex_auto_agent_google_lane_key(*a, **kw),
    resolve_antigravity_lane_state=lambda *a, **kw: _resolve_codex_auto_agent_antigravity_lane_state(*a, **kw),
    get_openrouter_adapter_active_cooldown_seconds=lambda *a, **kw: _get_openrouter_adapter_active_cooldown_seconds(*a, **kw),
    google_adapter_rate_limit_lock=_google_adapter_rate_limit_lock,
    google_adapter_rate_limit_until_monotonic_by_key=_google_adapter_rate_limit_until_monotonic_by_key,
    normalize_codex_alias_model=lambda *a, **kw: _normalize_codex_auto_agent_alias_model(*a, **kw),
    extract_client_product_label=lambda *a, **kw: _extract_auto_agent_alias_client_product_label(*a, **kw),
    resolve_codex_session_key=lambda *a, **kw: _resolve_codex_auto_agent_session_key(*a, **kw),
    resolve_anthropic_session_key=lambda *a, **kw: _resolve_anthropic_auto_agent_session_key(*a, **kw),
    has_continuation_state=lambda *a, **kw: _codex_auto_agent_request_has_continuation_state(*a, **kw),
    get_anthropic_candidates_for_alias=lambda *a, **kw: _get_anthropic_auto_agent_candidates_for_alias(*a, **kw),
    is_grok_account_quota_candidate=lambda *a, **kw: _is_codex_auto_agent_grok_account_quota_candidate(*a, **kw),
    get_grok_account_quota_lane_cooldown_key=lambda *a, **kw: _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(*a, **kw),
    is_kimi_code_candidate=lambda *a, **kw: _is_kimi_code_auto_agent_candidate(*a, **kw),
    get_kimi_managed_account_cooldown_key=lambda *a, **kw: _get_kimi_code_managed_account_cooldown_key(*a, **kw),
)

# Wave 5C: bind host dependencies into error_signals.
# Late-binding lambdas ensure god-module names resolve at call time.
_aawm_error_signals.configure_error_signals_runtime(
    extract_google_adapter_exception_detail=lambda *a, **kw: _extract_google_adapter_exception_detail(*a, **kw),
    extract_google_adapter_error_payloads=lambda *a, **kw: _extract_google_adapter_error_payloads(*a, **kw),
    is_openrouter_adapter_provider_raw_error=lambda *a, **kw: _is_openrouter_adapter_provider_raw_error(*a, **kw),
    extract_google_adapter_exception_status_code=lambda *a, **kw: _extract_google_adapter_exception_status_code(*a, **kw),
    extract_adapter_upstream_headers=lambda *a, **kw: _extract_adapter_upstream_headers(*a, **kw),
    parse_retry_after_seconds_from_headers=lambda *a, **kw: _parse_retry_after_seconds_from_headers(*a, **kw),
    get_adapter_header_value=lambda *a, **kw: _get_adapter_header_value(*a, **kw),
    extract_openrouter_adapter_raw_message=lambda *a, **kw: _extract_openrouter_adapter_raw_message(*a, **kw),
    parse_json_payloads_from_text_candidates=lambda *a, **kw: _parse_json_payloads_from_text_candidates(*a, **kw),
    get_passthrough_handled_http_error_summary=lambda *a, **kw: _get_passthrough_handled_http_error_summary(*a, **kw),
    is_known_grok_build_usage_balance_exhausted_response=lambda *a, **kw: _is_known_grok_build_usage_balance_exhausted_response(*a, **kw),
    is_known_grok_personal_team_spending_limit_response=lambda *a, **kw: _is_known_grok_personal_team_spending_limit_response(*a, **kw),
    durable_cooldown_error_classes=_CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES,
    capacity_error_tokens=_CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS,
    rate_limit_error_tokens=_CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS,
    native_grok_backoff_base_seconds=_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS,
    native_grok_backoff_max_seconds=_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS,
    native_grok_backoff_jitter_seconds=_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS,
)

# Wave 5C: bind error_signals / selection / cooldown_state / durable / state
# dependencies into cooldown_apply.
_aawm_cooldown_apply.configure_cooldown_apply_runtime(
    get_candidate_cooldown_scope=lambda *a, **kw: _get_codex_auto_agent_candidate_cooldown_scope(*a, **kw),
    get_kimi_managed_account_cooldown_key=lambda *a, **kw: _get_kimi_code_managed_account_cooldown_key(*a, **kw),
    get_grok_account_quota_lane_cooldown_key=lambda *a, **kw: _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(*a, **kw),
    get_request_local_cooldown_key=lambda *a, **kw: _get_codex_auto_agent_request_local_cooldown_key(*a, **kw),
    set_request_local_cooldown=lambda *a, **kw: _set_codex_auto_agent_request_local_cooldown(*a, **kw),
    exclude_request_local_candidate=lambda *a, **kw: _exclude_codex_auto_agent_request_local_candidate(*a, **kw),
    set_codex_cooldown=lambda *a, **kw: _set_codex_auto_agent_cooldown(*a, **kw),
    set_anthropic_cooldown=lambda *a, **kw: _set_anthropic_auto_agent_cooldown(*a, **kw),
    write_durable_payload=lambda *a, **kw: _aawm_alias_durable.write_aawm_alias_routing_durable_payload(*a, **kw),
    read_pilot_gate=_read_pilot_cooldown_gate,
    state_manager=_alias_routing_state,
)

# Wave 5C: bind error_signals / classification / host dependencies into
# attempt_records.
_aawm_attempt_records.configure_attempt_records_runtime(
    extract_error_tokens=lambda *a, **kw: _extract_codex_auto_agent_error_tokens(*a, **kw),
    extract_error_type_and_code=lambda *a, **kw: _extract_codex_auto_agent_error_type_and_code(*a, **kw),
    parse_header_wait_seconds=lambda *a, **kw: _parse_codex_auto_agent_header_wait_seconds(*a, **kw),
    get_source_error_summary=lambda *a, **kw: _get_codex_auto_agent_source_error_summary(*a, **kw),
    build_kimi_telemetry=lambda *a, **kw: _build_safe_kimi_code_selection_telemetry(*a, **kw),
    extract_status_code=lambda *a, **kw: _extract_google_adapter_exception_status_code(*a, **kw),
    safe_set_parsed_body=lambda *a, **kw: _safe_set_request_parsed_body(*a, **kw),
    emit_route_event=lambda *a, **kw: _emit_auto_agent_alias_route_event(*a, **kw),
    build_audit_event=lambda *a, **kw: _build_auto_agent_alias_audit_event(*a, **kw),
    build_audit_events=lambda *a, **kw: _build_auto_agent_alias_audit_events(*a, **kw),
    persist_audit_only_events=lambda *a, **kw: _persist_auto_agent_alias_audit_only_events_best_effort(*a, **kw),
    verbose_json_enabled=lambda *a, **kw: _aawm_alias_route_verbose_json_enabled(*a, **kw),
    healthy_json_enabled=lambda *a, **kw: _aawm_alias_route_healthy_json_enabled(*a, **kw),
    merge_metadata=lambda *a, **kw: _merge_litellm_metadata(*a, **kw),
    normalize_tag_value=lambda *a, **kw: _normalize_low_cardinality_tag_value(*a, **kw),
    normalize_codex_alias_model=lambda *a, **kw: _normalize_codex_auto_agent_alias_model(*a, **kw),
    normalize_anthropic_alias_model=lambda *a, **kw: _normalize_anthropic_auto_agent_alias_model(*a, **kw),
    load_bundled_model_cost=lambda *a, **kw: cast(
        Callable[..., dict[str, Any]],
        globals()["_load_bundled_model_cost_map_for_codex_policy"],
    )(*a, **kw),
    get_model_info=lambda *a, **kw: litellm.get_model_info(*a, **kw),
    model_cost=litellm.model_cost,
    openai_provider_value=litellm.LlmProviders.OPENAI.value,
    classify_failure=lambda *a, **kw: _aawm_alias_classification.classify_failure(*a, **kw),
    read_pilot_gate_record=lambda *a, **kw: _read_pilot_cooldown_gate.record(*a, **kw),
)

_wait_for_openrouter_adapter_cooldown_if_needed = _wave6b_openrouter_runtime._wait_for_openrouter_adapter_cooldown_if_needed


_set_openrouter_adapter_cooldown = _wave6b_openrouter_runtime._set_openrouter_adapter_cooldown


_run_openrouter_adapter_retry_loop = _wave6b_openrouter_runtime._run_openrouter_adapter_retry_loop


_perform_openrouter_completion_adapter_operation = _wave6b_openrouter_runtime._perform_openrouter_completion_adapter_operation


_perform_openrouter_adapter_pass_through_request = _wave6b_openrouter_runtime._perform_openrouter_adapter_pass_through_request


async def _prime_google_code_assist_session(
    access_token: str,
    companion_project: str,
    *,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> Optional[dict[str, Any]]:
    return await _anthropic_google_process_cache._prime_google_code_assist_session(
        access_token,
        companion_project,
        runtime=_get_anthropic_google_process_cache_runtime(),
        adapter_provider=adapter_provider,
    )


_load_local_google_oauth_access_token = _aawm_google_oauth._load_local_google_oauth_access_token


def _get_anthropic_adapter_google_target_base() -> str:
    return os.getenv("CODE_ASSIST_ENDPOINT") or "https://cloudcode-pa.googleapis.com"


def _normalize_google_completion_adapter_model_name(model: str) -> str:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._normalize_google_completion_adapter_model_name(model)


def _sanitize_google_schema_array_items(schema_node: Any, *, _depth: int = 0, _seen: Optional[set[int]] = None) -> int:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._sanitize_google_schema_array_items(schema_node, _depth=_depth, _seen=_seen)




_GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH = _google_codex_code_assist._GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH




_extract_completion_message_text = _google_context_window._extract_completion_message_text


def _is_google_adapter_synthetic_tool_context_text(text: Any) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._is_google_adapter_synthetic_tool_context_text(text)


def _is_google_adapter_synthetic_tool_context_message(message: Any) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._is_google_adapter_synthetic_tool_context_message(message)


_get_google_adapter_fallback_context_char_cap = _google_env_policy._get_google_adapter_fallback_context_char_cap


def _inject_google_adapter_fallback_text_context(
    google_request_dict: dict[str, Any], completion_messages: list[dict[str, Any]]
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._inject_google_adapter_fallback_text_context(
        google_request_dict, completion_messages
    )


_get_google_adapter_system_prompt_policy = _google_env_policy._get_google_adapter_system_prompt_policy


def _get_codex_google_code_assist_tool_contract_policy() -> str:
    raw_value = _clean_codex_auth_value(os.getenv(_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV))
    if raw_value is None:
        return _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"0", "false", "disabled", "none", "off"}:
        return "off"
    if normalized_value in {"1", "true", "enabled", "on", "append"}:
        return "append"
    return _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT


def _extract_google_adapter_system_text_from_content(content: Any) -> Optional[str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_adapter_system_text_from_content(content)


def _replace_google_adapter_system_message_text(message: dict[str, Any], rewritten_text: str) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._replace_google_adapter_system_message_text(message, rewritten_text)


def _append_codex_google_code_assist_tool_contract_to_system_text(system_text: str) -> str:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._append_codex_google_code_assist_tool_contract_to_system_text(system_text)


def _apply_codex_google_code_assist_tool_contract_policy(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_codex_google_code_assist_tool_contract_policy(completion_kwargs)


def _is_google_adapter_claude_overhead_block(block: str) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._is_google_adapter_claude_overhead_block(block)


def _strip_google_adapter_claude_system_overhead(system_text: str) -> tuple[str, int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._strip_google_adapter_claude_system_overhead(system_text)


def _build_google_adapter_system_prompt_policy_text(
    *, original_text: str, policy_mode: str
) -> tuple[str, dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_google_adapter_system_prompt_policy_text(
        original_text=original_text, policy_mode=policy_mode
    )


def _apply_google_adapter_system_prompt_policy(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_adapter_system_prompt_policy(completion_kwargs)


def _normalize_codex_openai_chat_kwargs_for_google_code_assist(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._normalize_codex_openai_chat_kwargs_for_google_code_assist(completion_kwargs)




def _has_codex_google_code_assist_anthropic_tool_replay_blocks(messages: list[Any]) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._has_codex_google_code_assist_anthropic_tool_replay_blocks(messages)




def _normalize_codex_google_code_assist_anthropic_assistant_message(
    *, message: dict[str, Any], message_index: int
) -> tuple[dict[str, Any], int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._normalize_codex_google_code_assist_anthropic_assistant_message(
        message=message, message_index=message_index
    )




def _normalize_codex_google_code_assist_anthropic_user_message(
    *, message: dict[str, Any], message_index: int
) -> tuple[list[dict[str, Any]], int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._normalize_codex_google_code_assist_anthropic_user_message(
        message=message, message_index=message_index
    )


def _build_codex_google_code_assist_anthropic_replay_changes(
    *, repaired_count: int, converted_tool_use_count: int, converted_tool_result_count: int
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_codex_google_code_assist_anthropic_replay_changes(
        repaired_count=repaired_count,
        converted_tool_use_count=converted_tool_use_count,
        converted_tool_result_count=converted_tool_result_count,
    )




def _deterministic_codex_google_code_assist_tool_call_id(
    *, message_index: int, tool_call_index: int, tool_call: dict[str, Any]
) -> str:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._deterministic_codex_google_code_assist_tool_call_id(
        message_index=message_index, tool_call_index=tool_call_index, tool_call=tool_call
    )


def _next_codex_google_code_assist_tool_messages(
    messages: list[Any], *, message_index: int
) -> list[tuple[int, dict[str, Any]]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._next_codex_google_code_assist_tool_messages(messages, message_index=message_index)


def _paired_codex_google_code_assist_tool_message(
    next_tool_messages: list[tuple[int, dict[str, Any]]], *, tool_call_index: int
) -> tuple[int, dict[str, Any]] | None:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._paired_codex_google_code_assist_tool_message(
        next_tool_messages, tool_call_index=tool_call_index
    )


def _repair_codex_google_code_assist_tool_call_id(
    *,
    message_index: int,
    tool_call_index: int,
    tool_call: dict[str, Any],
    paired_tool_message: tuple[int, dict[str, Any]] | None,
    copy_message_at: Callable[[int], Optional[dict[str, Any]]],
) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._repair_codex_google_code_assist_tool_call_id(
        message_index=message_index,
        tool_call_index=tool_call_index,
        tool_call=tool_call,
        paired_tool_message=paired_tool_message,
        copy_message_at=copy_message_at,
    )




def _normalize_codex_google_code_assist_reasoning_effort(
    mappable_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._normalize_codex_google_code_assist_reasoning_effort(mappable_params)


def _normalize_google_code_assist_thinking_max_tokens(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._normalize_google_code_assist_thinking_max_tokens(completion_kwargs)




def _infer_single_codex_google_code_assist_function_tool_name(tools: Any) -> Optional[str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._infer_single_codex_google_code_assist_function_tool_name(tools)


def _is_codex_google_code_assist_empty_text_content(content: Any) -> bool:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._is_codex_google_code_assist_empty_text_content(content)


def _previous_codex_google_code_assist_assistant_index(messages: list[Any], *, before_index: int) -> Optional[int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._previous_codex_google_code_assist_assistant_index(
        messages, before_index=before_index
    )


def _previous_codex_google_code_assist_contiguous_assistant_index(
    messages: list[Any], *, before_index: int
) -> Optional[int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._previous_codex_google_code_assist_contiguous_assistant_index(
        messages, before_index=before_index
    )


def _previous_codex_google_code_assist_tool_call(
    messages: list[Any], *, before_index: int, tool_call_id: str
) -> Optional[dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._previous_codex_google_code_assist_tool_call(
        messages, before_index=before_index, tool_call_id=tool_call_id
    )




def _build_codex_google_code_assist_synthetic_tool_call(
    *, tool_call_id: str, function_name: str, function_arguments: str
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_codex_google_code_assist_synthetic_tool_call(
        tool_call_id=tool_call_id, function_name=function_name, function_arguments=function_arguments
    )


def _append_codex_google_code_assist_tool_call_to_assistant(
    *, assistant_message: dict[str, Any], synthetic_tool_call: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._append_codex_google_code_assist_tool_call_to_assistant(
        assistant_message=assistant_message, synthetic_tool_call=synthetic_tool_call
    )


def _build_codex_google_code_assist_tool_pair_repair_changes(
    *, repaired_count: int, inserted_count: int, blank_text_suppressed_count: int, repaired_names: set[str]
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_codex_google_code_assist_tool_pair_repair_changes(
        repaired_count=repaired_count,
        inserted_count=inserted_count,
        blank_text_suppressed_count=blank_text_suppressed_count,
        repaired_names=repaired_names,
    )




def _append_codex_google_code_assist_orphan_tool_result_context(
    *, messages: list[Any], index: int, context_text: str
) -> None:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._append_codex_google_code_assist_orphan_tool_result_context(
        messages=messages, index=index, context_text=context_text
    )


def _sanitize_codex_google_code_assist_orphan_tool_results(
    completion_kwargs: dict[str, Any], *, scope_key: Optional[str] = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._sanitize_codex_google_code_assist_orphan_tool_results(
        completion_kwargs, scope_key=scope_key
    )




_get_google_code_assist_native_tool_aliases = _google_env_policy._get_google_code_assist_native_tool_aliases


def _apply_google_code_assist_alias_to_function_block(
    function_block: dict[str, Any], *, aliases: dict[str, str], tool_name_mapping: dict[str, str]
) -> tuple[dict[str, Any], Optional[str]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_code_assist_alias_to_function_block(
        function_block, aliases=aliases, tool_name_mapping=tool_name_mapping
    )


def _apply_google_code_assist_alias_to_tool(
    tool: Any, *, aliases: dict[str, str], tool_name_mapping: dict[str, str]
) -> tuple[Any, Optional[str]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_code_assist_alias_to_tool(
        tool, aliases=aliases, tool_name_mapping=tool_name_mapping
    )


def _apply_google_code_assist_aliases_to_tool_calls(
    tool_calls: Any, *, aliases: dict[str, str], tool_name_mapping: dict[str, str]
) -> tuple[Any, set[str]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_code_assist_aliases_to_tool_calls(
        tool_calls, aliases=aliases, tool_name_mapping=tool_name_mapping
    )


def _apply_google_code_assist_aliases_to_message(
    message: Any, *, aliases: dict[str, str], tool_name_mapping: dict[str, str]
) -> tuple[Any, set[str]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_code_assist_aliases_to_message(
        message, aliases=aliases, tool_name_mapping=tool_name_mapping
    )


def _apply_google_code_assist_native_tool_aliases(
    completion_kwargs: dict[str, Any], tool_name_mapping: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._apply_google_code_assist_native_tool_aliases(completion_kwargs, tool_name_mapping)


_get_google_adapter_max_completion_messages_window = _google_env_policy._get_google_adapter_max_completion_messages_window


_completion_message_has_visible_text = _google_context_window._completion_message_has_visible_text


def _inject_google_adapter_tool_call_context_text(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._inject_google_adapter_tool_call_context_text(messages)


_estimate_completion_message_text_chars = _google_context_window._estimate_completion_message_text_chars


_completion_message_has_tool_result = _google_context_window._completion_message_has_tool_result


_completion_message_tool_call_ids = _google_context_window._completion_message_tool_call_ids


_completion_message_tool_result_ids = _google_context_window._completion_message_tool_result_ids


_trim_completion_message_tail_preserving_tool_pairs = _google_context_window._trim_completion_message_tail_preserving_tool_pairs


_get_google_adapter_preserved_task_state_char_cap = _google_env_policy._get_google_adapter_preserved_task_state_char_cap


def _extract_google_adapter_preserved_task_excerpt(text: str) -> str:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_adapter_preserved_task_excerpt(text)


def _build_google_adapter_preserved_task_state_message(
    messages: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_google_adapter_preserved_task_state_message(messages)


_apply_google_adapter_completion_message_window = _google_context_window._apply_google_adapter_completion_message_window




_google_code_assist_duplicate_tool_results_from_completion_messages = _google_context_window._google_code_assist_duplicate_tool_results_from_completion_messages




_google_code_assist_tool_results_from_completion_messages = _google_context_window._google_code_assist_tool_results_from_completion_messages




def _extract_google_code_assist_text_metrics(content_block: Any) -> tuple[int, int]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_code_assist_text_metrics(content_block)


def _summarize_google_code_assist_content_preview_entry(content_entry: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._summarize_google_code_assist_content_preview_entry(content_entry)


def _summarize_google_code_assist_request_contents_shape(
    request_block: dict[str, Any], summary: dict[str, Any]
) -> None:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._summarize_google_code_assist_request_contents_shape(request_block, summary)


def _summarize_google_code_assist_generation_config_shape(
    request_block: dict[str, Any], summary: dict[str, Any]
) -> None:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._summarize_google_code_assist_generation_config_shape(request_block, summary)


def _extract_google_code_assist_function_names(request_block: Any) -> list[str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._extract_google_code_assist_function_names(request_block)


def _summarize_google_code_assist_request_shape(payload: Any) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._summarize_google_code_assist_request_shape(payload)




def _build_responses_response_from_adapter_response(response_obj: Any) -> Response:
    return Response(
        content=_serialize_responses_adapter_response(response_obj),
        media_type="application/json",
    )






def _wrap_streaming_response_with_release_callback(
    response: StreamingResponse,
    release_callback: Any,
) -> StreamingResponse:
    released = False

    def _release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            release_callback()
        except Exception:
            verbose_proxy_logger.exception("Failed to release adapted streaming response guard callback")

    original_iterator = getattr(response, "body_iterator", None)
    if original_iterator is None:
        _release_once()
        return response

    async def _wrapped_iterator():
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            _release_once()

    response.body_iterator = _wrapped_iterator()
    return response


_get_openrouter_api_key = _wave6b_openrouter_runtime._get_openrouter_api_key


_get_anthropic_adapter_openrouter_api_key = _wave6b_openrouter_runtime._get_anthropic_adapter_openrouter_api_key


_get_anthropic_adapter_nvidia_api_key = _wave6b_nvidia_runtime._get_anthropic_adapter_nvidia_api_key


_get_anthropic_adapter_nvidia_target_base = _wave6b_nvidia_runtime._get_anthropic_adapter_nvidia_target_base


_get_nvidia_adapter_max_retries = _wave6b_nvidia_runtime._get_nvidia_adapter_max_retries


_get_nvidia_adapter_request_timeout_seconds = _wave6b_nvidia_runtime._get_nvidia_adapter_request_timeout_seconds


_get_nvidia_adapter_inner_max_retries = _wave6b_nvidia_runtime._get_nvidia_adapter_inner_max_retries


_should_force_fake_stream_for_nvidia_adapter_model = _wave6b_nvidia_runtime._should_force_fake_stream_for_nvidia_adapter_model


_extract_nvidia_adapter_exception_status_code = _wave6b_nvidia_runtime._extract_nvidia_adapter_exception_status_code


_get_nvidia_adapter_retry_wait_seconds = _wave6b_nvidia_runtime._get_nvidia_adapter_retry_wait_seconds


_perform_nvidia_completion_adapter_operation = _wave6b_nvidia_runtime._perform_nvidia_completion_adapter_operation


_get_openrouter_target_base = _wave6b_openrouter_runtime._get_openrouter_target_base


_get_anthropic_adapter_openrouter_target_base = _wave6b_openrouter_runtime._get_anthropic_adapter_openrouter_target_base


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


def _antigravity_candidate_unavailable_detail(__exc, *__, **___):
    return _wave6b_common._antigravity_candidate_unavailable_detail(__exc, runtime=_wave6b_common_live_runtime())


def _raise_antigravity_auto_agent_candidate_unavailable(__exc, *__, **___):
    return _wave6b_common._raise_antigravity_auto_agent_candidate_unavailable(__exc, runtime=_wave6b_common_live_runtime())


_is_grok_unsupported_reasoning_parameter_detail = _wave6b_common._is_grok_unsupported_reasoning_parameter_detail


def _codex_native_openai_candidate_unavailable_detail(__exc, *__, **___):
    return _wave6b_common._codex_native_openai_candidate_unavailable_detail(__exc, runtime=_wave6b_common_live_runtime())


def _raise_codex_native_openai_auto_agent_candidate_unavailable(__exc, *__, **___):
    return _wave6b_common._raise_codex_native_openai_auto_agent_candidate_unavailable(__exc, runtime=_wave6b_common_live_runtime())


def _grok_native_candidate_unavailable_detail(__exc, *__, **___):
    return _wave6b_common._grok_native_candidate_unavailable_detail(__exc, runtime=_wave6b_common_live_runtime())


_xai_oauth_candidate_unavailable_detail = _wave6b_common._xai_oauth_candidate_unavailable_detail


_raise_xai_oauth_auto_agent_candidate_unavailable = _wave6b_common._raise_xai_oauth_auto_agent_candidate_unavailable


def _raise_grok_native_auto_agent_candidate_unavailable(__exc, *__, **___):
    return _wave6b_common._raise_grok_native_auto_agent_candidate_unavailable(__exc, runtime=_wave6b_common_live_runtime())



def _opencode_zen_candidate_unavailable_detail(__exc, *__, **___):
    return _wave6b_common._opencode_zen_candidate_unavailable_detail(__exc, runtime=_wave6b_common_live_runtime())


_raise_opencode_zen_auto_agent_candidate_unavailable = _wave6b_common._raise_opencode_zen_auto_agent_candidate_unavailable

# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


_openrouter_chat_message_function_call = _wave6b_openrouter_runtime._openrouter_chat_message_function_call


_openrouter_chat_message_has_valid_content_or_tool_calls = _wave6b_openrouter_runtime._openrouter_chat_message_has_valid_content_or_tool_calls


_copy_openrouter_message_value = _wave6b_openrouter_runtime._copy_openrouter_message_value


_serialize_openrouter_tool_call_arguments = _wave6b_openrouter_runtime._serialize_openrouter_tool_call_arguments


_normalize_openrouter_chat_message_tool_call_arguments = _wave6b_openrouter_runtime._normalize_openrouter_chat_message_tool_call_arguments


_sanitize_openrouter_completion_messages_for_chat_completion = _wave6b_openrouter_runtime._sanitize_openrouter_completion_messages_for_chat_completion


_apply_openrouter_completion_message_sanitization = _wave6b_openrouter_runtime._apply_openrouter_completion_message_sanitization


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


# Wave 6B: published by _wave6b_opencode_zen_runtime.install(globals())


_build_openrouter_default_headers = _wave6b_openrouter_runtime._build_openrouter_default_headers


def _get_claude_agent_spec_dir() -> Optional[Path]:
    for env_var in _CLAUDE_AGENT_SPEC_DIR_ENV_VARS:
        value = os.getenv(env_var)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return candidate

    for raw_path in _CLAUDE_AGENT_SPEC_DEFAULT_DIRS:
        candidate = Path(raw_path).expanduser()
        if candidate.is_dir():
            return candidate

    return None


def _extract_model_from_markdown_frontmatter(markdown_text: str) -> Optional[str]:
    if not markdown_text.startswith("---\n"):
        return None

    closing_index = markdown_text.find("\n---", 4)
    if closing_index == -1:
        return None

    frontmatter = markdown_text[4:closing_index]
    match = re.search(r"(?m)^model:\s*(?P<model>.+?)\s*$", frontmatter)
    if match is None:
        return None

    model_value = match.group("model").strip().strip('"').strip("'")
    return model_value or None


def _read_claude_agent_markdown(candidate_path: Path) -> Optional[str]:
    try:
        markdown_bytes = candidate_path.read_bytes()
    except OSError:
        return None

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return markdown_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    return markdown_bytes.decode("utf-8", errors="replace")


def _load_claude_agent_declared_model(agent_name: str) -> Optional[str]:
    normalized_agent_name = agent_name.strip()
    if not normalized_agent_name:
        return None

    if normalized_agent_name != Path(normalized_agent_name).name:
        return None

    agents_dir = _get_claude_agent_spec_dir()
    if agents_dir is None:
        return None

    candidate_path = agents_dir / f"{normalized_agent_name}.md"
    if not candidate_path.is_file():
        return None

    try:
        stat_result = candidate_path.stat()
    except OSError:
        return None

    cache_entry = _claude_agent_model_cache.get(candidate_path)
    cache_key = getattr(stat_result, "st_mtime_ns", None)
    if cache_entry is not None and cache_entry[0] == cache_key:
        return cache_entry[1]

    markdown_text = _read_claude_agent_markdown(candidate_path)
    if markdown_text is None:
        return None

    model_name = _extract_model_from_markdown_frontmatter(markdown_text)
    _claude_agent_model_cache[candidate_path] = (cache_key, model_name)
    return model_name


_aawm_alias_durable.configure_durable_runtime(
    clean_value=_clean_codex_auth_value,
    get_dual_cache_override=lambda: (
        globals()["_get_aawm_alias_routing_dual_cache"]()
        if globals().get("_get_aawm_alias_routing_dual_cache")
        is not _aawm_alias_durable.get_aawm_alias_routing_dual_cache
        else None
    ),
)


AntigravityOAuthTokenData = dict[str, object]
AntigravityPassthroughRequestBody = dict[str, object]
PassthroughLoggingMetadata = dict[str, object]


def _build_google_debug_header_summary(headers: dict[str, Any]) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_google_debug_header_summary(headers)


_get_google_adapter_native_user_agent = _google_env_policy._get_google_adapter_native_user_agent


_get_google_adapter_native_api_client_header = _google_env_policy._get_google_adapter_native_api_client_header


def _build_google_adapter_native_headers(*, access_token: str, model: Optional[str], accept: str) -> dict[str, str]:
    _anthropic_google_shaping.bind_runtime(globals())
    return _anthropic_google_shaping._build_google_adapter_native_headers(
        access_token=access_token, model=model, accept=accept
    )


def _write_json_file_atomic(
    path: Path,
    data: OAuthJsonData,
    *,
    failure_label: str,
) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        payload = json.dumps(data, indent=2) + "\n"
        tmp_path.write_text(payload, encoding="utf-8")
        try:
            current_mode = path.stat().st_mode & 0o777
            os.chmod(tmp_path, current_mode)
        except OSError:
            pass
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist refreshed {failure_label} auth data to {path}: {exc}",
        ) from exc


def _get_anthropic_adapter_openai_target_base(
    request: Request,
    *,
    prefer_chatgpt_codex_backend: bool = False,
) -> str:
    if prefer_chatgpt_codex_backend or _anthropic_adapter_request_uses_codex_native_auth(request):
        return os.getenv("CHATGPT_API_BASE") or CHATGPT_API_BASE
    return os.getenv("OPENAI_API_BASE") or "https://api.openai.com/"


def _add_codex_native_tool_alias_adapter_metadata(
    adapter_tags: list[str],
    adapter_extra_fields: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    adapter_tags.append("anthropic-openai-codex-native-tools")
    adapter_extra_fields["anthropic_adapter_codex_native_tool_aliases"] = True


_ANTHROPIC_PROVIDER_SHAPING_RUNTIME = _anthropic_provider_common.ShapingRuntime(
    normalize_function_tool_schemas=lambda body: (_normalize_openai_function_tool_schemas(body)),
    add_native_tool_metadata=lambda tags, fields, **kwargs: (
        _add_codex_native_tool_alias_adapter_metadata(tags, fields, **kwargs)
    ),
    apply_tool_description_patches=lambda body: (_apply_codex_tool_description_patches_to_request_body(body)),
    merge_metadata=lambda body, **kwargs: _merge_litellm_metadata(body, **kwargs),
    add_route_family_metadata=lambda body, family: (_add_route_family_logging_metadata(body, family)),
    build_span=lambda **kwargs: _build_langfuse_span_descriptor(**kwargs),
    apply_openai_parallel_policy=lambda body: (_apply_openai_adapter_parallel_instruction_policy(body)),
    apply_forced_responses_tool_choice=lambda source, translated: (
        _apply_forced_bash_tool_choice_for_responses_adapter(source, translated)
    ),
    apply_forced_completion_tool_choice=lambda body: (
        _maybe_force_explicit_bash_tool_choice_for_completion_adapter(body)
    ),
    log_debug=lambda message, *args: verbose_proxy_logger.debug(message, *args),
)


def _build_anthropic_responses_adapter_request_body(
    request_body: dict[str, Any],
    *,
    adapter_model: str,
    route_family: str = "anthropic_openai_responses_adapter",
    tag_prefix: str = "anthropic-openai-responses-adapter",
    span_name: str = "anthropic.openai_responses_adapter",
    target_endpoint: str = "/v1/responses",
    use_chatgpt_codex_defaults: bool = False,
) -> dict[str, Any]:
    return _anthropic_provider_common.build_responses_request_body(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,
        request_body,
        adapter_model=adapter_model,
        route_family=route_family,
        tag_prefix=tag_prefix,
        span_name=span_name,
        target_endpoint=target_endpoint,
        use_chatgpt_codex_defaults=use_chatgpt_codex_defaults,
    )


def _get_openai_adapter_function_tool_names(
    request_body: dict[str, Any],
) -> list[str]:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return []

    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _apply_responses_adapter_parallel_instruction_policy(
    request_body: dict[str, Any],
    *,
    tag_prefix: str,
    metadata_prefix: str,
    span_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if request_body.get("parallel_tool_calls") is not True:
        return request_body, {}

    function_tool_names = _get_openai_adapter_function_tool_names(request_body)
    if len(function_tool_names) < 2:
        return request_body, {}

    existing_instructions = request_body.get("instructions")
    if not isinstance(existing_instructions, str) or not existing_instructions.strip():
        return request_body, {}

    policy_block = _OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS
    # RR-054 #20: prepend the parallel-tool policy; never wipe the caller system prompt.
    if policy_block in existing_instructions:
        return request_body, {}

    rewritten_instructions = f"{policy_block}\n\n{existing_instructions}"
    updated_body = dict(request_body)
    updated_body["instructions"] = rewritten_instructions
    original_hash = hashlib.sha256(existing_instructions.encode("utf-8", errors="replace")).hexdigest()
    changes = {
        f"{metadata_prefix}_parallel_instruction_policy_applied": True,
        f"{metadata_prefix}_parallel_instruction_original_chars": len(existing_instructions),
        f"{metadata_prefix}_parallel_instruction_rewritten_chars": len(rewritten_instructions),
        f"{metadata_prefix}_parallel_instruction_original_hash": original_hash,
        f"{metadata_prefix}_parallel_instruction_tool_names": function_tool_names,
        f"{metadata_prefix}_parallel_instruction_mode": "prepend",
    }
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            f"{tag_prefix}-parallel-instruction-policy",
            *[
                f"{tag_prefix}-parallel-tool:{_normalize_low_cardinality_tag_value(tool_name) or 'unknown'}"
                for tool_name in function_tool_names
            ],
        ],
        extra_fields={
            **changes,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata={
                        "tool_names": function_tool_names,
                        "original_chars": len(existing_instructions),
                        "rewritten_chars": len(rewritten_instructions),
                        "mode": "prepend",
                    },
                )
            ],
        },
    )
    return updated_body, changes


def _apply_openai_adapter_parallel_instruction_policy(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _apply_responses_adapter_parallel_instruction_policy(
        request_body,
        tag_prefix="openai-adapter",
        metadata_prefix="openai_adapter",
        span_name="openai_adapter.parallel_instruction_policy",
    )


def _apply_openrouter_adapter_parallel_instruction_policy(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _apply_responses_adapter_parallel_instruction_policy(
        request_body,
        tag_prefix="openrouter-adapter",
        metadata_prefix="openrouter_adapter",
        span_name="openrouter_adapter.parallel_instruction_policy",
    )


def _drop_anthropic_grok_native_prior_function_call_replay(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, []

    # First pass: collect call_ids for prior function_call items to drop.
    drop_call_ids: set[str] = set()
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id.strip():
            drop_call_ids.add(call_id.strip())

    updated_input_items: list[Any] = []
    dropped_items: list[dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        cleaned_call_id = call_id.strip() if isinstance(call_id, str) and call_id.strip() else None
        # RR-054 #21: drop prior function_call items and any paired outputs so the
        # provider does not see orphaned function_call_output rows.
        if item_type == "function_call" or (
            item_type == "function_call_output" and cleaned_call_id is not None and cleaned_call_id in drop_call_ids
        ):
            metadata_item: dict[str, Any] = {
                "type": item_type,
                "index": index,
            }
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                metadata_item["name"] = name.strip()
            if cleaned_call_id is not None:
                metadata_item["call_id_hash"] = hashlib.sha256(
                    cleaned_call_id.encode("utf-8", errors="replace")
                ).hexdigest()[:12]
            dropped_items.append(metadata_item)
            continue
        updated_input_items.append(item)

    if not dropped_items:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    dropped_names = _dedupe_sorted_str_list(
        [item["name"] for item in dropped_items if isinstance(item.get("name"), str) and item["name"]]
    )
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "anthropic-grok-native-prior-function-call-replay-dropped",
        ],
        extra_fields={
            "anthropic_grok_native_prior_function_call_replay_dropped_count": len(dropped_items),
            "anthropic_grok_native_prior_function_call_replay_dropped_items": dropped_items,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.grok_native_prior_function_call_replay_dropped",
                    metadata={
                        "dropped_count": len(dropped_items),
                        "dropped_names": dropped_names,
                    },
                )
            ],
        },
    )
    return updated_body, dropped_items


def _build_anthropic_response_from_responses_response(
    response_body: dict[str, Any],
    *,
    reject_empty_success: bool = False,
    diagnostic_context: Optional[dict[str, Any]] = None,
    use_codex_native_tools: bool = False,
    retryable_failed_response: bool = False,
    failed_response_adapter_model: Optional[str] = None,
    failed_response_adapter: str = "anthropic_responses_adapter",
    failed_response_adapter_label: str = "Responses",
    malformed_intake_context: Optional[dict[str, Any]] = None,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
        LiteLLMAnthropicToResponsesAPIAdapter,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    if _is_failed_responses_body(response_body):
        _raise_responses_adapter_failed_response(
            response_body=response_body,
            adapter_model=failed_response_adapter_model or str(response_body.get("model") or "unknown-model"),
            adapter=failed_response_adapter,
            adapter_label=failed_response_adapter_label,
            retryable_alias_candidate=retryable_failed_response,
        )

    if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
        _raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body=response_body,
            adapter_model=failed_response_adapter_model or str(response_body.get("model") or "unknown-model"),
            adapter=failed_response_adapter,
            adapter_label=failed_response_adapter_label,
            intake_context=malformed_intake_context,
        )

    if reject_empty_success and _is_empty_success_responses_body(response_body):
        diagnostic = _build_empty_success_responses_diagnostic(
            response_body=response_body,
            diagnostic_context=diagnostic_context,
        )
        verbose_proxy_logger.warning(
            "OpenRouter Responses adapter returned empty successful response: %s",
            json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)[:8000],
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "OpenRouter Responses adapter returned empty successful response",
                "diagnostic": diagnostic,
            },
        )

    adapter = LiteLLMAnthropicToResponsesAPIAdapter()
    translated_response = adapter.translate_response(
        ResponsesAPIResponse(**response_body),
        use_codex_native_tools=use_codex_native_tools,
    )
    translated_response_any = cast(Any, translated_response)
    if hasattr(translated_response_any, "model_dump_json"):
        serialized_response = translated_response_any.model_dump_json(exclude_none=True)
    elif hasattr(translated_response_any, "json"):
        serialized_response = translated_response_any.json(exclude_none=True)
    else:
        serialized_response = json.dumps(translated_response)
    return Response(
        content=serialized_response,
        media_type="application/json",
    )


def _build_completion_adapter_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(request_body.get("metadata") or {})
    litellm_metadata = request_body.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        return metadata

    # Normal completion callbacks turn metadata.trace_* into Langfuse trace
    # fields. Keep provider-specific litellm_metadata intact, but mirror the
    # trace context into metadata so completion adapters match passthrough logs.
    for key in (
        "session_id",
        "trace_id",
        "existing_trace_id",
        "trace_name",
        "trace_user_id",
        "trace_environment",
    ):
        value = litellm_metadata.get(key)
        if value and (key in {"trace_name", "trace_user_id"} or not metadata.get(key)):
            metadata[key] = value
    for key in (
        "source_trace_name",
        "agent_name",
        "aawm_claude_agent_name",
        "tenant_id",
        "aawm_tenant_id",
        "aawm_claude_project",
    ):
        value = litellm_metadata.get(key)
        if value:
            metadata[key] = value
    for key in (
        "passthrough_route_family",
        "anthropic_adapter_model",
        "anthropic_adapter_original_model",
        "anthropic_adapter_target_endpoint",
        "langfuse_spans",
    ):
        value = litellm_metadata.get(key)
        if value:
            metadata[key] = value
    litellm_tags = litellm_metadata.get("tags")
    if isinstance(litellm_tags, list):
        existing_tags = metadata.get("tags")
        if not isinstance(existing_tags, list):
            existing_tags = []
        metadata["tags"] = [
            *existing_tags,
            *[tag for tag in litellm_tags if tag not in existing_tags],
        ]
    return metadata


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
        if schema_node.get("type") == "object" and not isinstance(schema_node.get("properties"), dict):
            schema_node["properties"] = {}
            fix_count += 1
        for value in schema_node.values():
            fix_count += _sanitize_openai_object_schema_properties(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            fix_count += _sanitize_openai_object_schema_properties(item)
    return fix_count


def _normalize_openai_function_tool_schemas(translated_body: dict[str, Any]) -> None:
    tools = translated_body.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        if "parameters" in tool:
            tool["parameters"] = _normalize_openai_function_tool_parameters(tool.get("parameters"))

        function_block = tool.get("function")
        if isinstance(function_block, dict):
            function_block["parameters"] = _normalize_openai_function_tool_parameters(function_block.get("parameters"))


def _copy_translated_anthropic_adapter_response_headers(
    *,
    translated_response: Response,
    upstream_response: Response,
) -> None:
    for header_name, header_value in upstream_response.headers.items():
        if header_name.lower() in {
            "content-length",
            "content-encoding",
            "transfer-encoding",
        }:
            continue
        translated_response.headers[header_name] = header_value


def _get_latest_adapter_user_prompt_text(request_body: dict[str, Any]) -> Optional[str]:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _prompt_explicitly_requests_bash_tool(prompt_text: Optional[str]) -> bool:
    if not isinstance(prompt_text, str) or not prompt_text:
        return False
    lowered_prompt = prompt_text.lower()
    return "bash tool" in lowered_prompt or "run the bash command" in lowered_prompt


def _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
    request_body: dict[str, Any],
    translated_body: dict[str, Any],
) -> dict[str, Any]:
    if translated_body.get("tool_choice") is not None:
        return {}

    tools = translated_body.get("tools")
    if not isinstance(tools, list):
        return {}

    latest_user_prompt = _get_latest_adapter_user_prompt_text(request_body)
    if not _prompt_explicitly_requests_bash_tool(latest_user_prompt):
        return {}

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        tool_name = tool.get("name")
        if tool_name in {"Bash", "run_shell_command"}:
            translated_body["tool_choice"] = {"type": "function", "name": tool_name}
            return {"forced_explicit_bash_tool_choice": tool_name}
    return {}


def _apply_forced_bash_tool_choice_for_responses_adapter(
    request_body: dict[str, Any],
    translated_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    forced_tool_choice_changes = _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
        request_body,
        translated_body,
    )
    if not forced_tool_choice_changes:
        return translated_body, {}
    return (
        _merge_litellm_metadata(
            translated_body,
            extra_fields=forced_tool_choice_changes,
        ),
        forced_tool_choice_changes,
    )


def _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    if request_body.get("tool_choice") is not None:
        return {}

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return {}

    latest_user_prompt = _get_latest_adapter_user_prompt_text(request_body)
    if not _prompt_explicitly_requests_bash_tool(latest_user_prompt):
        return {}

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name")
        if tool_name in {"Bash", "run_shell_command"}:
            request_body["tool_choice"] = {"type": "tool", "name": tool_name}
            return {"forced_explicit_bash_tool_choice": tool_name}
    return {}


def _responses_request_contains_mcp_tools(request_body: dict[str, Any]) -> bool:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "mcp":
            return True
    return False


def _coerce_mapping_to_namespace(
    value: Any,
    *,
    _depth: int = 0,
    _max_depth: int = _AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
) -> Any:
    # RR-054 #27: bound recursion so pathological SSE payloads cannot explode CPU/stack.
    if _depth > _max_depth:
        return value
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, dict):
        return SimpleNamespace(
            **{
                key: _coerce_mapping_to_namespace(val, _depth=_depth + 1, _max_depth=_max_depth)
                for key, val in value.items()
            }
        )
    if isinstance(value, list):
        return [_coerce_mapping_to_namespace(item, _depth=_depth + 1, _max_depth=_max_depth) for item in value]
    return value


















_GROK_COMPOSER_LITERAL_TOOL_LABEL_LINE_RE = re.compile(r"(?im)^Tool label:\s*(?P<name>[^\n]+)\s*$")
_GROK_COMPOSER_LITERAL_CORRELATION_REF_LINE_RE = re.compile(r"(?im)^Correlation ref:\s*(?P<call_id>[^\n]+)\s*$")
_GROK_COMPOSER_LITERAL_INPUT_PAYLOAD_LINE_RE = re.compile(r"(?im)^Input payload:\s*(?P<payload>.+?)\s*$")
_GROK_COMPOSER_LITERAL_TOOL_END_MARKER_RE = re.compile(
    r"^\s*(?:<\|tool_call_end\|>|<\|tool_calls_end\|>|<｜tool▁call▁end｜>|<｜tool▁calls▁end｜>)+\s*$"
)
_GROK_COMPOSER_LITERAL_TOOL_ARGUMENT_METADATA_KEYS = frozenset({"description"})
_GROK_COMPOSER_LITERAL_CONTEXT_NOTE_LINE_RE = re.compile(
    r"(?im)^\s*\[Context note - prior assistant step; not an executable tool invocation\]\s*$"
)
























































































































def _get_anthropic_adapter_access_log_target_label(
    target_url: Union[str, httpx.URL],
) -> str:
    parsed_url = urlparse(str(target_url))
    hostname = parsed_url.hostname or "unknown-target"
    path = parsed_url.path or "/"
    query = f"?{parsed_url.query}" if parsed_url.query else ""
    return f"{hostname}{path}{query}"


def _annotate_request_scope_for_adapted_access_log(request: Request, target_url: Union[str, httpx.URL]) -> None:
    """Record adapted target for access logs without mutating live query_string.

    RR-054 #39: cosmetic logging must not corrupt ASGI ``query_string`` / path
    observed by later middleware, guardrails, or exception handlers. The target
    label lives on private scope keys and is consumed by access-log builders.
    """
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return

    target_label = _get_anthropic_adapter_access_log_target_label(target_url)
    if scope.get("_aawm_adapted_access_log_target") == target_label:
        return
    scope["_aawm_adapted_access_log_target"] = target_label
    request_url = getattr(request, "url", None)
    if request_url is not None:
        original_path = getattr(request_url, "path", None) or scope.get("path", "")
        original_query = getattr(request_url, "query", None) or ""
    else:
        original_path = scope.get("path", "")
        raw_query_string = scope.get("query_string", b"")
        if isinstance(raw_query_string, bytes):
            original_query = raw_query_string.decode("utf-8", errors="replace")
        else:
            original_query = str(raw_query_string or "")
    if isinstance(original_query, bytes):
        original_query = original_query.decode("utf-8", errors="replace")
    display_query = f"{original_query} -> {target_label}" if original_query else f"adapted_to={target_label}"
    scope["_aawm_adapted_access_log_display_path"] = (
        f"{original_path}?{display_query}" if original_path else display_query
    )


def _get_proxy_shared_aiohttp_session() -> Optional[Any]:
    try:
        from litellm.proxy.proxy_server import shared_aiohttp_session
    except Exception:
        return None
    if shared_aiohttp_session is None:
        return None
    if getattr(shared_aiohttp_session, "closed", False):
        return None
    return shared_aiohttp_session


def _serialize_anthropic_adapter_response(response_obj: Any) -> str:
    if hasattr(response_obj, "model_dump_json"):
        return response_obj.model_dump_json(exclude_none=True)
    if hasattr(response_obj, "json"):
        return response_obj.json(exclude_none=True)
    return json.dumps(response_obj)


def _build_anthropic_response_from_completion_adapter_response(
    response_obj: Any,
) -> Response:
    return Response(
        content=_serialize_anthropic_adapter_response(response_obj),
        media_type="application/json",
    )






def _log_google_completion_adapter_debug(
    *,
    prepared_request_body: dict[str, Any],
    wrapped_request_body: dict[str, Any],
    google_model: str,
    adapter_headers: dict[str, str],
    sanitized_schema_fix_count: int,
    generation_policy_changes: dict[str, Any],
) -> None:
    if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") != "1":
        return

    try:
        debug_shape = _summarize_google_code_assist_request_shape(wrapped_request_body)
        request_payload = wrapped_request_body.get("request") if isinstance(wrapped_request_body, dict) else None
        function_names = _extract_google_code_assist_function_names(request_payload)
        litellm_metadata = (
            prepared_request_body.get("litellm_metadata") if isinstance(prepared_request_body, dict) else None
        )
        google_persisted_output_compacted_count = (
            litellm_metadata.get("google_adapter_persisted_output_compacted_count")
            if isinstance(litellm_metadata, dict)
            else None
        )
        completion_message_window_debug = (
            litellm_metadata.get("google_adapter_completion_message_window")
            if isinstance(litellm_metadata, dict)
            else None
        )
        verbose_proxy_logger.info(
            "Gemini adapter debug: model=%s upstream_headers=%s schema_fixes=%s google_persisted_output_compacted_count=%s completion_message_window=%s generation_policy_changes=%s body_shape=%s function_names=%s",
            google_model,
            _build_google_debug_header_summary(adapter_headers),
            sanitized_schema_fix_count,
            google_persisted_output_compacted_count,
            completion_message_window_debug,
            generation_policy_changes,
            debug_shape,
            function_names,
        )
    except Exception:
        verbose_proxy_logger.exception("Gemini adapter debug logging failed")


async def _prepare_anthropic_google_completion_adapter_request(
    *,
    request: Request,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> SimpleNamespace:
    _anthropic_google_shaping.bind_runtime(globals())
    return await _anthropic_google_shaping._prepare_anthropic_google_completion_adapter_request(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        adapter_provider=adapter_provider,
    )


def _release_google_adapter_semaphore_once(
    google_adapter_semaphore: Any,
    release_state: dict[str, bool],
    *,
    google_model: str,
) -> None:
    if release_state.get("released"):
        return
    release_state["released"] = True
    google_adapter_semaphore.release()
    if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
        verbose_proxy_logger.info(
            "Google adapter semaphore released for model=%s",
            google_model,
        )


async def _perform_anthropic_google_completion_adapter_request(
    *,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    adapter_request: SimpleNamespace,
    use_alias_candidate_probe: bool = False,
) -> Response:
    from litellm.litellm_core_utils.litellm_logging import Logging

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=str(adapter_request.annotated_target_url),
        headers=adapter_request.adapter_headers,
        credential_family="google",
        expected_target_family="google",
    )
    _annotate_request_scope_for_adapted_access_log(
        request,
        adapter_request.annotated_target_url,
    )

    google_adapter_semaphore = _get_google_adapter_semaphore(
        rate_limit_key=adapter_request.google_adapter_rate_limit_key
    )
    await google_adapter_semaphore.acquire()
    release_state = {"released": False}
    if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
        verbose_proxy_logger.info(
            "Google adapter semaphore acquired for model=%s stream=%s",
            adapter_request.google_model,
            adapter_request.is_stream,
        )

    stream_release_attached = False
    try:
        upstream_response = await _perform_google_adapter_pass_through_request(
            request=request,
            target=adapter_request.target_url,
            custom_headers=adapter_request.adapter_headers,
            user_api_key_dict=user_api_key_dict,
            custom_body=adapter_request.wrapped_request_body,
            forward_headers=False,
            query_params=adapter_request.target_query_params,
            stream=adapter_request.is_stream,
            custom_llm_provider=adapter_request.custom_llm_provider,
            egress_credential_family="google",
            expected_target_family="google",
            google_adapter_rate_limit_key=adapter_request.google_adapter_rate_limit_key,
            google_adapter_max_retries=0 if use_alias_candidate_probe else None,
            google_adapter_model_capacity_max_retries=(0 if use_alias_candidate_probe else None),
            google_adapter_hidden_retry_budget_seconds=(0 if use_alias_candidate_probe else None),
        )

        if not isinstance(upstream_response, StreamingResponse):
            raise HTTPException(
                status_code=502,
                detail="Google Code Assist adapter expected a streaming response.",
            )

        if adapter_request.client_requested_stream:
            streaming_response = _build_anthropic_streaming_response_from_google_code_assist_stream(
                response=upstream_response,
                adapter_model=adapter_request.google_model,
                tool_name_mapping=adapter_request.tool_name_mapping,
                gemini_optional_params=adapter_request.gemini_optional_params,
                rate_limit_key=adapter_request.google_adapter_rate_limit_key,
            )
            stream_release_attached = True
            return _wrap_streaming_response_with_release_callback(
                streaming_response,
                lambda: _release_google_adapter_semaphore_once(
                    google_adapter_semaphore,
                    release_state,
                    google_model=adapter_request.google_model,
                ),
            )

        logging_obj = Logging(
            model=adapter_request.google_model,
            messages=adapter_request.completion_messages,
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id=str(uuid4()),
            function_id="anthropic_google_completion_adapter",
        )
        logging_obj.optional_params = adapter_request.gemini_optional_params

        return await _collect_google_code_assist_response_from_stream(
            response=upstream_response,
            adapter_model=adapter_request.google_model,
            tool_name_mapping=adapter_request.tool_name_mapping,
            logging_obj=logging_obj,
        )
    finally:
        if not adapter_request.is_stream or not stream_release_attached:
            _release_google_adapter_semaphore_once(
                google_adapter_semaphore,
                release_state,
                google_model=adapter_request.google_model,
            )


async def _handle_anthropic_google_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
    use_alias_candidate_probe: bool = False,
) -> Response:
    try:
        adapter_request = await _prepare_anthropic_google_completion_adapter_request(
            request=request,
            prepared_request_body=prepared_request_body,
            adapter_model=adapter_model,
            adapter_provider=adapter_provider,
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
            and _antigravity_candidate_unavailable_detail(exc) is not None
        ):
            _raise_antigravity_auto_agent_candidate_unavailable(exc)
        raise
    return await _perform_anthropic_google_completion_adapter_request(
        request=request,
        user_api_key_dict=user_api_key_dict,
        adapter_request=adapter_request,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_codex_google_code_assist_adapter_request(
    *,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    adapter_request: SimpleNamespace,
    use_alias_candidate_probe: bool = False,
) -> Response:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=str(adapter_request.annotated_target_url),
        headers=adapter_request.adapter_headers,
        credential_family="google",
        expected_target_family="google",
    )
    _annotate_request_scope_for_adapted_access_log(
        request,
        adapter_request.annotated_target_url,
    )

    google_adapter_semaphore = _get_google_adapter_semaphore(
        rate_limit_key=adapter_request.google_adapter_rate_limit_key
    )
    await google_adapter_semaphore.acquire()
    release_state = {"released": False}
    stream_release_attached = False
    try:
        upstream_response = await _perform_google_adapter_pass_through_request(
            request=request,
            target=adapter_request.target_url,
            custom_headers=adapter_request.adapter_headers,
            user_api_key_dict=user_api_key_dict,
            custom_body=adapter_request.wrapped_request_body,
            forward_headers=False,
            query_params=adapter_request.target_query_params,
            stream=adapter_request.is_stream,
            custom_llm_provider=adapter_request.custom_llm_provider,
            egress_credential_family="google",
            expected_target_family="google",
            google_adapter_rate_limit_key=adapter_request.google_adapter_rate_limit_key,
            google_adapter_max_retries=0 if use_alias_candidate_probe else None,
            google_adapter_model_capacity_max_retries=(0 if use_alias_candidate_probe else None),
            google_adapter_hidden_retry_budget_seconds=(0 if use_alias_candidate_probe else None),
        )

        if not isinstance(upstream_response, StreamingResponse):
            raise HTTPException(
                status_code=502,
                detail="Google Code Assist adapter expected a streaming response.",
            )

        if adapter_request.client_requested_stream:
            streaming_response = _build_codex_streaming_response_from_google_code_assist_stream(
                response=upstream_response,
                adapter_request=adapter_request,
            )
            stream_release_attached = True
            return _wrap_streaming_response_with_release_callback(
                streaming_response,
                lambda: _release_google_adapter_semaphore_once(
                    google_adapter_semaphore,
                    release_state,
                    google_model=adapter_request.google_model,
                ),
            )

        logging_obj = Logging(
            model=adapter_request.google_model,
            messages=adapter_request.completion_messages,
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id=str(uuid4()),
            function_id="codex_google_code_assist_adapter",
        )
        logging_obj.optional_params = adapter_request.gemini_optional_params
        model_response = await _collect_google_code_assist_model_response_from_stream(
            response=upstream_response,
            adapter_model=adapter_request.google_model,
            logging_obj=logging_obj,
        )
        model_response = _restore_google_adapter_tool_call_names(
            model_response,
            adapter_request.tool_name_mapping,
        )
        if use_alias_candidate_probe and _is_codex_google_code_assist_empty_success_model_response(model_response):
            _raise_codex_auto_agent_empty_success_response(
                response_body={
                    "id": _mapping_or_attr_get(model_response, "id"),
                    "model": _mapping_or_attr_get(
                        model_response,
                        "model",
                        adapter_request.google_model,
                    ),
                    "status": "completed",
                    "output": [],
                    "usage": _model_response_usage_dict(_mapping_or_attr_get(model_response, "usage")),
                },
                adapter_model=adapter_request.google_model,
                adapter="codex_auto_agent_google_code_assist",
                adapter_label="Gemini Code Assist",
            )
        responses_api_response = (
            LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                chat_completion_response=model_response,
                request_input=adapter_request.codex_request_input,
                responses_api_request=adapter_request.responses_api_request,
            )
        )
        return _build_responses_response_from_adapter_response(responses_api_response)
    finally:
        if not adapter_request.is_stream or not stream_release_attached:
            _release_google_adapter_semaphore_once(
                google_adapter_semaphore,
                release_state,
                google_model=adapter_request.google_model,
            )


async def _handle_codex_google_code_assist_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
    use_alias_candidate_probe: bool = False,
) -> Response:
    try:
        adapter_request = await _prepare_codex_google_code_assist_adapter_request(
            request=request,
            prepared_request_body=prepared_request_body,
            adapter_model=adapter_model,
            adapter_provider=adapter_provider,
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
            and _antigravity_candidate_unavailable_detail(exc) is not None
        ):
            _raise_antigravity_auto_agent_candidate_unavailable(exc)
        raise
    return await _perform_codex_google_code_assist_adapter_request(
        request=request,
        user_api_key_dict=user_api_key_dict,
        adapter_request=adapter_request,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _resolve_anthropic_openai_responses_adapter_auth_context(
    request: Request,
) -> tuple[dict[str, Any], bool, bool, Optional[str]]:
    local_codex_headers = None
    has_client_auth = _anthropic_adapter_request_has_openai_client_auth(request)
    uses_codex_native_auth = _anthropic_adapter_request_uses_codex_native_auth(request)
    if not has_client_auth:
        local_codex_headers = await _load_local_codex_auth_headers(request)

    custom_headers: dict[str, Any] = {}
    forward_headers = _anthropic_adapter_should_forward_direct_auth_headers(request)
    if local_codex_headers is not None:
        custom_headers = local_codex_headers
        forward_headers = False
    elif not has_client_auth:
        openai_api_key = passthrough_endpoint_router.get_credentials(
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            region_name=None,
        )
        if openai_api_key is None:
            raise Exception(
                "Anthropic adapter requests for OpenAI/Codex models require forwarded OpenAI/Codex auth headers or 'OPENAI_API_KEY' in environment."
            )
        custom_headers = BaseOpenAIPassThroughHandler._assemble_headers(
            api_key=openai_api_key,
            request=request,
        )
        forward_headers = False

    use_chatgpt_codex_defaults = uses_codex_native_auth or local_codex_headers is not None
    egress_credential_family = "openai" if local_codex_headers is not None else None
    return (
        custom_headers,
        forward_headers,
        use_chatgpt_codex_defaults,
        egress_credential_family,
    )


_aawm_responses_finalize.configure_responses_finalize_runtime(
    _aawm_responses_finalize.ResponsesFinalizeRuntime(
        annotate_request=lambda *args, **kwargs: (_annotate_request_scope_for_adapted_access_log(*args, **kwargs)),
        validate_stream=lambda *args, **kwargs: (_validate_alias_candidate_responses_stream_if_needed(*args, **kwargs)),
        collect_stream=lambda *args, **kwargs: (_collect_responses_response_from_stream(*args, **kwargs)),
        build_response=lambda *args, **kwargs: (_build_anthropic_response_from_responses_response(*args, **kwargs)),
        copy_headers=lambda *args, **kwargs: (_copy_translated_anthropic_adapter_response_headers(*args, **kwargs)),
        build_streaming_response=lambda *args, **kwargs: (
            _build_anthropic_streaming_response_from_responses_stream(*args, **kwargs)
        ),
        decode_response_body=lambda *args, **kwargs: (_decode_http_response_body(*args, **kwargs)),
        build_malformed_context=lambda *args, **kwargs: (
            _build_malformed_intake_context_for_anthropic_responses_adapter(*args, **kwargs)
        ),
    )
)


async def _finalize_anthropic_responses_adapter_upstream_response(
    *,
    upstream_response: object,
    request: Request,
    translated_request_body: Payload,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    provider: str,
    target_url: object,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool,
    use_codex_native_tools: bool = False,
    unexpected_detail: str,
    response_builder_kwargs: Optional[Payload] = None,
    stream_builder_kwargs: Optional[Payload] = None,
    malformed_upstream_url: Optional[object] = None,
    skip_stream_probe_validation: bool = False,
) -> Response:
    """Thin compatibility wrapper around package-owned response finalization."""
    return await _aawm_responses_finalize.finalize_anthropic_responses_adapter_upstream_response(
        upstream_response=upstream_response,
        request=request,
        translated_request_body=translated_request_body,
        adapter_model=adapter_model,
        adapter=adapter,
        adapter_label=adapter_label,
        provider=provider,
        target_url=target_url,
        client_requested_stream=client_requested_stream,
        use_alias_candidate_probe=use_alias_candidate_probe,
        use_codex_native_tools=use_codex_native_tools,
        unexpected_detail=unexpected_detail,
        response_builder_kwargs=response_builder_kwargs,
        stream_builder_kwargs=stream_builder_kwargs,
        malformed_upstream_url=malformed_upstream_url,
        skip_stream_probe_validation=skip_stream_probe_validation,
    )


def _prepare_anthropic_completion_adapter_request_body(
    prepared_request_body: Payload,
    *,
    adapter_model: str,
    route_family: str,
    tag_prefix: str,
    span_name: str,
    target_endpoint_label: str,
    span_metadata_extra: Optional[Payload] = None,
) -> Payload:
    return _anthropic_provider_common.prepare_completion_request_body(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,
        prepared_request_body,
        adapter_model=adapter_model,
        route_family=route_family,
        tag_prefix=tag_prefix,
        span_name=span_name,
        target_endpoint_label=target_endpoint_label,
        span_metadata_extra=span_metadata_extra,
    )


def _apply_anthropic_responses_adapter_common_request_policies(
    prepared_request_body: Payload,
    translated_request_body: Payload,
    *,
    parallel_policy_log_label: str,
    forced_tool_choice_log_label: str,
) -> Payload:
    config = _aawm_adapter_config.AnthropicResponsesAdapterConfig(
        adapter="compat",
        adapter_label="compat",
        provider="openai",
        unexpected_detail="compat",
        parallel_policy_log_label=parallel_policy_log_label,
        forced_tool_choice_log_label=forced_tool_choice_log_label,
    )
    return _anthropic_provider_common.apply_responses_policies(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,
        prepared_request_body,
        translated_request_body,
        config=config,
    )


async def _finalize_anthropic_responses_adapter_from_config(
    *,
    config: "_aawm_adapter_config.AnthropicResponsesAdapterConfig",
    upstream_response: object,
    request: Request,
    translated_request_body: Payload,
    adapter_model: str,
    target_url: object,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool,
    use_codex_native_tools: Optional[bool] = None,
    malformed_upstream_url: Optional[object] = None,
) -> Response:
    """Config-driven Responses finalize entry (RR-054 #9)."""
    kwargs = _aawm_adapter_config.responses_finalize_kwargs(
        config,
        adapter_model=adapter_model,
        translated_request_body=translated_request_body,
    )
    if use_codex_native_tools is not None:
        kwargs["use_codex_native_tools"] = use_codex_native_tools
    if malformed_upstream_url is not None:
        kwargs["malformed_upstream_url"] = malformed_upstream_url
    return await _finalize_anthropic_responses_adapter_upstream_response(
        upstream_response=upstream_response,
        request=request,
        translated_request_body=translated_request_body,
        adapter_model=adapter_model,
        target_url=target_url,
        client_requested_stream=client_requested_stream,
        use_alias_candidate_probe=use_alias_candidate_probe,
        **kwargs,
    )


def _apply_anthropic_responses_adapter_policies_from_config(
    prepared_request_body: Payload,
    translated_request_body: Payload,
    *,
    config: "_aawm_adapter_config.AnthropicResponsesAdapterConfig",
) -> Payload:
    return _anthropic_provider_common.apply_responses_policies(
        _ANTHROPIC_PROVIDER_SHAPING_RUNTIME,
        prepared_request_body,
        translated_request_body,
        config=config,
    )


async def _perform_anthropic_responses_adapter_pass_through(
    *,
    config: "_aawm_adapter_config.AnthropicResponsesAdapterConfig",
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    translated_request_body: Payload,
    adapter_model: str,
    target_url: object,
    custom_headers: Payload,
    client_requested_stream: bool,
    use_alias_candidate_probe: bool = False,
    forward_headers: bool = False,
    allowed_forward_headers: Optional[list[str]] = None,
    allowed_pass_through_prefixed_headers: Optional[list[str]] = None,
    custom_llm_provider: Optional[str] = None,
    egress_credential_family: Optional[str] = None,
    expected_target_family: Optional[str] = None,
    retryable_upstream_status_codes: Optional[list[int]] = None,
    pass_through_fn: Optional[Callable[..., Awaitable[object]]] = None,
    use_codex_native_tools: Optional[bool] = None,
    malformed_upstream_url: Optional[object] = None,
    extra_pass_through_kwargs: Optional[Payload] = None,
) -> Response:
    """Shared Responses adapter pass-through + finalize driver (RR-054 #9)."""
    transport = pass_through_fn or pass_through_request
    retry_codes = retryable_upstream_status_codes
    if retry_codes is None:
        retry_codes = list(
            _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
            if use_alias_candidate_probe
            else _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT
        )
    pt_kwargs: dict[str, Any] = {
        "request": request,
        "target": str(target_url),
        "custom_headers": custom_headers,
        "user_api_key_dict": user_api_key_dict,
        "custom_body": translated_request_body,
        "forward_headers": forward_headers,
        "stream": bool(translated_request_body.get("stream")),
        "custom_llm_provider": custom_llm_provider or config.provider,
        "egress_credential_family": egress_credential_family or config.provider,
        "expected_target_family": expected_target_family or config.provider,
        "retryable_upstream_status_codes": retry_codes,
        "caller_managed_hidden_retry": use_alias_candidate_probe,
    }
    if allowed_forward_headers is not None:
        pt_kwargs["allowed_forward_headers"] = allowed_forward_headers
    if allowed_pass_through_prefixed_headers is not None:
        pt_kwargs["allowed_pass_through_prefixed_headers"] = allowed_pass_through_prefixed_headers
    if extra_pass_through_kwargs:
        pt_kwargs.update(extra_pass_through_kwargs)
    upstream_response = await transport(**pt_kwargs)
    return await _finalize_anthropic_responses_adapter_from_config(
        config=config,
        upstream_response=upstream_response,
        request=request,
        translated_request_body=translated_request_body,
        adapter_model=adapter_model,
        target_url=target_url,
        client_requested_stream=client_requested_stream,
        use_alias_candidate_probe=use_alias_candidate_probe,
        use_codex_native_tools=use_codex_native_tools,
        malformed_upstream_url=malformed_upstream_url,
    )


async def _perform_normalized_anthropic_completion_adapter_stream(
    *,
    handler: Any,
    handler_call_kwargs: dict[str, Any],
    handler_extra_kwargs: dict[str, Any],
    completion_stream_normalizer: Callable[[Any], Any],
) -> object:
    completion_kwargs, tool_name_mapping = handler._prepare_completion_kwargs(
        **handler_call_kwargs,
        extra_kwargs=handler_extra_kwargs,
    )
    raw_completion_stream = await litellm.acompletion(**completion_kwargs)
    normalized_completion_stream = completion_stream_normalizer(raw_completion_stream)
    return handler._transform_completion_response(
        normalized_completion_stream,
        model=handler_call_kwargs["model"],
        stream=True,
        tool_name_mapping=tool_name_mapping,
    )


def _finalize_anthropic_completion_adapter_response(
    *,
    completion_response: object,
    stream_flag: bool,
    fake_stream: bool,
    rollup_kwargs: dict[str, Any],
    adapter_label: str,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    if stream_flag:
        if fake_stream:
            if not _is_anthropic_messages_response(completion_response):
                raise TypeError("Fake Anthropic streaming requires a non-streaming response")
            response_stream = FakeAnthropicMessagesStreamIterator(completion_response)
        else:
            response_stream = completion_response
        streaming_response = _build_anthropic_streaming_response_from_completion_adapter_stream(
            response_stream,
        )
        return _record_adapted_completed_route_rollup_after_stream(
            streaming_response,
            rollup_kwargs,
            adapter_label=adapter_label,
        )

    response = _build_anthropic_response_from_completion_adapter_response(
        completion_response,
    )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label=adapter_label,
    )
    return response


async def _perform_anthropic_completion_adapter_messages_call(
    *,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    target_url: Union[str, httpx.URL],
    api_key: str,
    api_base: str,
    custom_llm_provider: Optional[str] = None,
    client_requested_stream: Optional[bool] = None,
    model_for_upstream: Optional[str] = None,
    stream_override: Optional[bool] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    operation_wrapper: Optional[Callable[[Callable[[], Awaitable[object]]], Awaitable[object]]] = None,
    fake_stream: bool = False,
    extra_handler_kwargs: Optional[Payload] = None,
    completion_stream_normalizer: Optional[Callable[[Any], Any]] = None,
) -> Response:
    """Shared completion-adapter messages handler + response branch (RR-054 #9)."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    stream_flag = (
        bool(prepared_request_body.get("stream")) if client_requested_stream is None else bool(client_requested_stream)
    )
    upstream_stream = stream_flag if stream_override is None else bool(stream_override)
    requested_model = prepared_request_body.get("model")
    model_name = model_for_upstream or (requested_model if isinstance(requested_model, str) else adapter_model)
    handler_extra_kwargs: dict[str, Any] = {
        "custom_llm_provider": custom_llm_provider or config.custom_llm_provider,
        "api_key": api_key,
        "api_base": api_base,
        "litellm_metadata": prepared_request_body.get("litellm_metadata") or {},
        "proxy_server_request": {
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
    }
    if timeout is not None:
        handler_extra_kwargs["timeout"] = timeout
    if max_retries is not None:
        handler_extra_kwargs["max_retries"] = max_retries
    if extra_handler_kwargs:
        handler_extra_kwargs.update(extra_handler_kwargs)

    raw_max_tokens = prepared_request_body.get("max_tokens")
    max_tokens = raw_max_tokens if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool) else 1024
    raw_messages = prepared_request_body.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    raw_stop_sequences = prepared_request_body.get("stop_sequences")
    stop_sequences = (
        [item for item in raw_stop_sequences if isinstance(item, str)] if isinstance(raw_stop_sequences, list) else None
    )
    raw_system = prepared_request_body.get("system")
    system = raw_system if isinstance(raw_system, str) else None
    raw_temperature = prepared_request_body.get("temperature")
    temperature = (
        float(raw_temperature)
        if isinstance(raw_temperature, (int, float)) and not isinstance(raw_temperature, bool)
        else None
    )
    raw_thinking = prepared_request_body.get("thinking")
    thinking = raw_thinking if isinstance(raw_thinking, dict) else None
    raw_tool_choice = prepared_request_body.get("tool_choice")
    tool_choice = raw_tool_choice if isinstance(raw_tool_choice, dict) else None
    raw_tools = prepared_request_body.get("tools")
    tools = raw_tools if isinstance(raw_tools, list) else None
    raw_top_k = prepared_request_body.get("top_k")
    top_k = raw_top_k if isinstance(raw_top_k, int) and not isinstance(raw_top_k, bool) else None
    raw_top_p = prepared_request_body.get("top_p")
    top_p = float(raw_top_p) if isinstance(raw_top_p, (int, float)) and not isinstance(raw_top_p, bool) else None
    raw_output_format = prepared_request_body.get("output_format")
    output_format = raw_output_format if isinstance(raw_output_format, dict) else None
    raw_output_config = prepared_request_body.get("output_config")
    output_config = raw_output_config if isinstance(raw_output_config, dict) else None
    handler_call_kwargs = {
        "max_tokens": max_tokens,
        "messages": messages,
        "model": model_name,
        "metadata": _build_completion_adapter_metadata(prepared_request_body),
        "stop_sequences": stop_sequences,
        "stream": upstream_stream,
        "system": system,
        "temperature": temperature,
        "thinking": thinking,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_k": top_k,
        "top_p": top_p,
        "output_format": output_format,
        "output_config": output_config,
    }

    async def _operation() -> object:
        if upstream_stream and completion_stream_normalizer is not None:
            return await _perform_normalized_anthropic_completion_adapter_stream(
                handler=LiteLLMMessagesToCompletionTransformationHandler,
                handler_call_kwargs=handler_call_kwargs,
                handler_extra_kwargs=handler_extra_kwargs,
                completion_stream_normalizer=completion_stream_normalizer,
            )
        return await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            **handler_call_kwargs,
            **handler_extra_kwargs,
        )

    litellm_metadata = prepared_request_body.get("litellm_metadata")
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata if isinstance(litellm_metadata, dict) else {})
    _annotate_request_scope_for_adapted_access_log(request, target_url)
    _emit_adapted_route_access_log(
        request=request,
        target_url=str(target_url),
        request_body=prepared_request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label=config.adapter_label,
    )
    if operation_wrapper is not None:
        completion_response = await operation_wrapper(_operation)
    else:
        completion_response = await _operation()

    return _finalize_anthropic_completion_adapter_response(
        completion_response=completion_response,
        stream_flag=stream_flag,
        fake_stream=fake_stream,
        rollup_kwargs=rollup_kwargs,
        adapter_label=config.adapter_label,
    )


def _is_anthropic_messages_response(
    value: object,
) -> TypeGuard[AnthropicMessagesResponse]:
    return isinstance(value, dict)


_ANTHROPIC_OPENAI_PROVIDER_RUNTIME = _anthropic_openai_provider.Runtime(
    resolve_auth_context=lambda request: (_resolve_anthropic_openai_responses_adapter_auth_context(request)),
    compact_context=lambda body, **kwargs: (
        _compact_openai_adapter_claude_context_in_anthropic_request_body(body, **kwargs)
    ),
    log_debug=lambda message, *args: verbose_proxy_logger.debug(message, *args),
    build_request_body=lambda body, **kwargs: (_build_anthropic_responses_adapter_request_body(body, **kwargs)),
    apply_policies=lambda source, translated, **kwargs: (
        _apply_anthropic_responses_adapter_policies_from_config(source, translated, **kwargs)
    ),
    add_breakout_metadata=lambda body: (_add_codex_request_breakout_logging_metadata(body)),
    contains_mcp_tools=lambda body: _responses_request_contains_mcp_tools(body),
    get_target_base=lambda request, **kwargs: (_get_anthropic_adapter_openai_target_base(request, **kwargs)),
    normalize_endpoint=lambda **kwargs: (BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(**kwargs)),
    join_url=lambda *args: BaseOpenAIPassThroughHandler._join_url_paths(*args),
    url_factory=httpx.URL,
    provider=litellm.LlmProviders.OPENAI.value,
    forward_header_allowlist=tuple(_ANTHROPIC_ADAPTER_OPENAI_FORWARD_HEADER_ALLOWLIST),
    xpass_header_allowlist=tuple(_ANTHROPIC_ADAPTER_OPENAI_XPASS_HEADER_ALLOWLIST),
)

_ANTHROPIC_XAI_PROVIDER_RUNTIME = _anthropic_xai_provider.Runtime(
    build_responses_body=lambda body, **kwargs: (_build_anthropic_responses_adapter_request_body(body, **kwargs)),
    apply_responses_policies=lambda source, translated, **kwargs: (
        _apply_anthropic_responses_adapter_policies_from_config(source, translated, **kwargs)
    ),
    drop_unsupported_params=lambda body: (_drop_unsupported_codex_request_params_from_request_body(body)),
    prepare_passthrough_request=_prepare_oa_xai_passthrough_request,
    unavailable_detail=lambda exc: _xai_oauth_candidate_unavailable_detail(exc),
    raise_candidate_unavailable=lambda detail: (
        _raise_xai_oauth_auto_agent_candidate_unavailable(cast(Exception, detail))
    ),
    to_native_model=lambda model: _to_xai_native_passthrough_model(model),
    normalize_endpoint=lambda **kwargs: (BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(**kwargs)),
    join_url=lambda *args: BaseOpenAIPassThroughHandler._join_url_paths(*args),
    url_factory=httpx.URL,
    assemble_headers=lambda **kwargs: (BaseOpenAIPassThroughHandler._assemble_headers(**kwargs)),
    prepare_completion_body=lambda body, **kwargs: (_prepare_anthropic_completion_adapter_request_body(body, **kwargs)),
    validate_egress=lambda **kwargs: (HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)),
    provider=litellm.LlmProviders.XAI.value,
    provider_target=litellm.LlmProviders.XAI,
)

_ANTHROPIC_GROK_PROVIDER_RUNTIME = _anthropic_grok_provider.Runtime(
    build_request_body=lambda body, **kwargs: (_build_anthropic_responses_adapter_request_body(body, **kwargs)),
    apply_policies=lambda source, translated, **kwargs: (
        _apply_anthropic_responses_adapter_policies_from_config(source, translated, **kwargs)
    ),
    drop_unsupported_params=lambda body: (_drop_unsupported_codex_request_params_from_request_body(body)),
    drop_prior_replay=lambda body: (_drop_anthropic_grok_native_prior_function_call_replay(body)),
    prepare_passthrough_request=lambda body, **kwargs: (_prepare_grok_native_oauth_passthrough_request(body, **kwargs)),
    unavailable_detail=lambda exc: _grok_native_candidate_unavailable_detail(exc),
    raise_candidate_unavailable=lambda exc: (_raise_grok_native_auto_agent_candidate_unavailable(exc)),
    join_url=lambda **kwargs: _join_grok_passthrough_url(**kwargs),
    provider=litellm.LlmProviders.XAI.value,
)

_ANTHROPIC_NVIDIA_PROVIDER_RUNTIME = _anthropic_nvidia_provider.Runtime(
    should_force_fake_stream=lambda model: (_should_force_fake_stream_for_nvidia_adapter_model(model)),
    prepare_request_body=lambda body, **kwargs: (_prepare_anthropic_completion_adapter_request_body(body, **kwargs)),
    get_api_key=lambda: _get_anthropic_adapter_nvidia_api_key(),
    get_target_base=lambda: _get_anthropic_adapter_nvidia_target_base(),
    normalize_endpoint=lambda **kwargs: (BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(**kwargs)),
    join_url=lambda *args: BaseOpenAIPassThroughHandler._join_url_paths(*args),
    url_factory=httpx.URL,
    validate_egress=lambda **kwargs: (HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)),
    perform_operation=lambda **kwargs: (_perform_nvidia_completion_adapter_operation(**kwargs)),
    get_timeout_seconds=lambda model: (_get_nvidia_adapter_request_timeout_seconds(model)),
    get_inner_max_retries=lambda: _get_nvidia_adapter_inner_max_retries(),
    provider=litellm.LlmProviders.NVIDIA_NIM.value,
    provider_target=litellm.LlmProviders.NVIDIA_NIM,
)

_ANTHROPIC_OPENROUTER_PROVIDER_RUNTIME = _anthropic_openrouter_provider.Runtime(
    compact_context=lambda body, **kwargs: (
        _compact_openai_adapter_claude_context_in_anthropic_request_body(body, **kwargs)
    ),
    log_debug=lambda message, *args: verbose_proxy_logger.debug(message, *args),
    build_responses_body=lambda body, **kwargs: (_build_anthropic_responses_adapter_request_body(body, **kwargs)),
    apply_parallel_policy=lambda body: (_apply_openrouter_adapter_parallel_instruction_policy(body)),
    apply_forced_tool_choice=lambda source, translated: (
        _apply_forced_bash_tool_choice_for_responses_adapter(source, translated)
    ),
    contains_mcp_tools=lambda body: _responses_request_contains_mcp_tools(body),
    get_api_key=lambda: _get_anthropic_adapter_openrouter_api_key(),
    raise_candidate_unavailable=lambda detail: (_raise_openrouter_auto_agent_candidate_unavailable(str(detail))),
    get_target_base=lambda: _get_anthropic_adapter_openrouter_target_base(),
    normalize_endpoint=lambda **kwargs: (BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(**kwargs)),
    join_url=lambda *args: BaseOpenAIPassThroughHandler._join_url_paths(*args),
    url_factory=httpx.URL,
    assemble_headers=lambda **kwargs: (BaseOpenAIPassThroughHandler._assemble_headers(**kwargs)),
    build_default_headers=lambda: _build_openrouter_default_headers(),
    perform_responses_request=lambda **kwargs: (_perform_openrouter_adapter_pass_through_request(**kwargs)),
    get_completion_model=lambda model: (_get_openrouter_completion_adapter_upstream_model(model)),
    prepare_completion_body=lambda body, **kwargs: (_prepare_anthropic_completion_adapter_request_body(body, **kwargs)),
    validate_egress=lambda **kwargs: (HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)),
    perform_completion_operation=lambda **kwargs: (_perform_openrouter_completion_adapter_operation(**kwargs)),
    provider=litellm.LlmProviders.OPENROUTER.value,
    provider_target=litellm.LlmProviders.OPENROUTER.value,
)

_ANTHROPIC_OPENCODE_ZEN_PROVIDER_RUNTIME = _anthropic_opencode_zen_provider.Runtime(
    build_responses_body=lambda body, **kwargs: (_build_anthropic_responses_adapter_request_body(body, **kwargs)),
    add_logging_metadata=lambda body, **kwargs: (_add_opencode_zen_logging_metadata(body, **kwargs)),
    apply_parallel_policy=lambda body: (_apply_openrouter_adapter_parallel_instruction_policy(body)),
    apply_forced_tool_choice=lambda source, translated: (
        _apply_forced_bash_tool_choice_for_responses_adapter(source, translated)
    ),
    log_debug=lambda message, *args: verbose_proxy_logger.debug(message, *args),
    contains_mcp_tools=lambda body: _responses_request_contains_mcp_tools(body),
    get_target_base=lambda: _get_opencode_zen_target_base(),
    join_url=lambda **kwargs: _join_opencode_zen_passthrough_url(**kwargs),
    build_headers=lambda request, **kwargs: (_build_opencode_zen_headers(request, **kwargs)),
    unavailable_detail=lambda exc: (_opencode_zen_candidate_unavailable_detail(exc)),
    raise_candidate_unavailable=lambda exc: (_raise_opencode_zen_auto_agent_candidate_unavailable(exc)),
    url_factory=httpx.URL,
    prepare_completion_body=lambda body, **kwargs: (_prepare_anthropic_completion_adapter_request_body(body, **kwargs)),
    load_api_key=lambda **kwargs: (_load_opencode_zen_api_key_for_candidate(**kwargs)),
    assemble_headers=lambda **kwargs: (BaseOpenAIPassThroughHandler._assemble_headers(**kwargs)),
    validate_egress=lambda **kwargs: (HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)),
    provider=_OPENCODE_ZEN_PROVIDER,
    completion_provider=litellm.LlmProviders.OPENAI.value,
)


async def _prepare_anthropic_openai_responses_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.ResponsesAdapterRoutePlan":
    return await _anthropic_openai_provider.prepare_responses_route(
        runtime=_ANTHROPIC_OPENAI_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_openai_responses_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response
    return await _aawm_adapter_driver.run_responses_adapter_route(
        prepare=_prepare_anthropic_openai_responses_adapter_route,
        perform=_perform_anthropic_responses_adapter_pass_through,
        request=request,
        user_api_key_dict=user_api_key_dict,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_xai_oauth_responses_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.ResponsesAdapterRoutePlan":
    return await _anthropic_xai_provider.prepare_responses_route(
        runtime=_ANTHROPIC_XAI_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_xai_oauth_responses_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response
    return await _aawm_adapter_driver.run_responses_adapter_route(
        prepare=_prepare_anthropic_xai_oauth_responses_adapter_route,
        perform=_perform_anthropic_responses_adapter_pass_through,
        request=request,
        user_api_key_dict=user_api_key_dict,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_grok_native_oauth_responses_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.ResponsesAdapterRoutePlan":
    return await _anthropic_grok_provider.prepare_responses_route(
        runtime=_ANTHROPIC_GROK_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_grok_native_oauth_responses_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response
    return await _aawm_adapter_driver.run_responses_adapter_route(
        prepare=_prepare_anthropic_grok_native_oauth_responses_adapter_route,
        perform=_perform_anthropic_responses_adapter_pass_through,
        request=request,
        user_api_key_dict=user_api_key_dict,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_xai_oauth_completion_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    return await _anthropic_xai_provider.prepare_completion_route(
        runtime=_ANTHROPIC_XAI_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_xai_oauth_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    return await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_anthropic_xai_oauth_completion_adapter_route,
        perform=_perform_anthropic_completion_adapter_messages_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=False,
    )


async def _prepare_anthropic_nvidia_completion_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    return await _anthropic_nvidia_provider.prepare_completion_route(
        runtime=_ANTHROPIC_NVIDIA_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_nvidia_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    return await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_anthropic_nvidia_completion_adapter_route,
        perform=_perform_anthropic_completion_adapter_messages_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=False,
    )


async def _prepare_anthropic_openrouter_completion_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    return await _anthropic_openrouter_provider.prepare_completion_route(
        runtime=_ANTHROPIC_OPENROUTER_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_openrouter_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    return await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_anthropic_openrouter_completion_adapter_route,
        perform=_perform_anthropic_completion_adapter_messages_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_openrouter_responses_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.ResponsesAdapterRoutePlan":
    return await _anthropic_openrouter_provider.prepare_responses_route(
        runtime=_ANTHROPIC_OPENROUTER_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_openrouter_responses_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response
    return await _aawm_adapter_driver.run_responses_adapter_route(
        prepare=_prepare_anthropic_openrouter_responses_adapter_route,
        perform=_perform_anthropic_responses_adapter_pass_through,
        request=request,
        user_api_key_dict=user_api_key_dict,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_opencode_zen_responses_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.ResponsesAdapterRoutePlan":
    return await _anthropic_opencode_zen_provider.prepare_responses_route(
        runtime=_ANTHROPIC_OPENCODE_ZEN_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_opencode_zen_responses_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response
    return await _aawm_adapter_driver.run_responses_adapter_route(
        prepare=_prepare_anthropic_opencode_zen_responses_adapter_route,
        perform=_perform_anthropic_responses_adapter_pass_through,
        request=request,
        user_api_key_dict=user_api_key_dict,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_opencode_zen_completion_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    return await _anthropic_opencode_zen_provider.prepare_completion_route(
        runtime=_ANTHROPIC_OPENCODE_ZEN_PROVIDER_RUNTIME,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_kimi_chat_completions_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    return await _kimi_code_adapters.prepare_anthropic_kimi_chat_completions_adapter_route(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_kimi_chat_completions_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    return await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_anthropic_kimi_chat_completions_adapter_route,
        perform=_perform_anthropic_completion_adapter_messages_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _prepare_anthropic_alibaba_token_plan_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    return await _alibaba_token_plan_adapters.prepare_anthropic_alibaba_token_plan_adapter_route(
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_alibaba_token_plan_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    return await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_anthropic_alibaba_token_plan_adapter_route,
        perform=_perform_anthropic_completion_adapter_messages_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_opencode_zen_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    return await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_anthropic_opencode_zen_completion_adapter_route,
        perform=_perform_anthropic_completion_adapter_messages_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _handle_anthropic_opencode_zen_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    if adapter_model in _OPENCODE_ZEN_ANTHROPIC_COMPLETION_MODELS:
        return await _handle_anthropic_opencode_zen_completion_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )
    return await _handle_anthropic_opencode_zen_responses_adapter_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    for secret_name in secret_names:
        value = _clean_secret_string(get_secret_str(secret_name))
        if value:
            return value
    return None


# RR-054 #1: wire Google OAuth package runtime deps after helpers exist.
_aawm_google_oauth.configure_google_oauth_runtime(
    clean_value=_clean_codex_auth_value,
    get_first_secret_value=_get_first_secret_value,
    invalidate_google_lane_cache=_invalidate_codex_auto_agent_google_lane_cache,
)


def _normalize_aawm_sslmode(value: Optional[str]) -> Optional[str]:
    cleaned = _clean_secret_string(value)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return "require"
    if lowered in {"0", "false", "no", "off"}:
        return "disable"
    return cleaned


_get_google_adapter_persisted_output_char_cap = _google_env_policy._get_google_adapter_persisted_output_char_cap


_get_google_adapter_auxiliary_context_char_cap = _google_env_policy._get_google_adapter_auxiliary_context_char_cap


_get_google_adapter_followup_persisted_output_char_cap = _google_env_policy._get_google_adapter_followup_persisted_output_char_cap


_get_google_adapter_followup_auxiliary_context_char_cap = _google_env_policy._get_google_adapter_followup_auxiliary_context_char_cap


def _is_anthropic_web_search_tool(value: dict[str, Any]) -> bool:
    tool_type = value.get("type")
    tool_name = value.get("name")
    return (isinstance(tool_type, str) and tool_type.startswith("web_search")) or tool_name == "web_search"


def _sanitize_anthropic_web_search_empty_domain_lists_in_value(
    value: Any,
) -> tuple[Any, int]:
    if isinstance(value, dict):
        updated_dict: dict[str, Any] = {}
        changed = False
        sanitized_count = 0
        is_web_search_tool = _is_anthropic_web_search_tool(value)
        for key, child in value.items():
            if is_web_search_tool and key in {"allowed_domains", "blocked_domains"} and child == []:
                updated_dict[key] = None
                changed = True
                sanitized_count += 1
                continue
            (
                updated_child,
                child_count,
            ) = _sanitize_anthropic_web_search_empty_domain_lists_in_value(child)
            updated_dict[key] = updated_child
            sanitized_count += child_count
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), sanitized_count

    if isinstance(value, list):
        updated_list: list[Any] = []
        changed = False
        sanitized_count = 0
        for child in value:
            (
                updated_child,
                child_count,
            ) = _sanitize_anthropic_web_search_empty_domain_lists_in_value(child)
            updated_list.append(updated_child)
            sanitized_count += child_count
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), sanitized_count

    return value, 0


def _sanitize_anthropic_web_search_empty_domain_lists(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    (
        updated_value,
        sanitized_count,
    ) = _sanitize_anthropic_web_search_empty_domain_lists_in_value(request_body)
    if not sanitized_count or not isinstance(updated_value, dict):
        return request_body, 0
    updated_body = _merge_litellm_metadata(
        updated_value,
        tags_to_add=["claude-web-search-domain-filter-sanitized"],
        extra_fields={
            "claude_web_search_domain_filter_sanitized_count": sanitized_count,
        },
    )
    return updated_body, sanitized_count


_rewrite_grok_native_unsupported_input_items_from_request_body = _wave6b_xai_request_prep._rewrite_grok_native_unsupported_input_items_from_request_body


_rewrite_grok_native_unsupported_input_items_in_place = _wave6b_xai_request_prep._rewrite_grok_native_unsupported_input_items_in_place


def _should_preserve_openai_client_auth(request: Request, endpoint: str) -> bool:
    """
    Preserve inbound client auth only for OpenAI Responses and model-list
    passthrough traffic.

    This keeps Codex-style already-authenticated requests as close to native
    behavior as possible while leaving the existing server-authenticated
    passthrough behavior intact for other OpenAI endpoints.
    """
    if _is_openai_responses_endpoint(endpoint):
        return _request_has_openai_client_auth(request)
    if _is_openai_models_endpoint(endpoint):
        return _request_has_openai_client_auth(request) and _request_uses_codex_native_auth(request)
    return False


def _get_openai_passthrough_target_base(request: Request, endpoint: str) -> str:
    if _should_preserve_openai_client_auth(request=request, endpoint=endpoint):
        if _request_uses_codex_native_auth(request):
            return os.getenv("CHATGPT_API_BASE") or CHATGPT_API_BASE
    return os.getenv("OPENAI_API_BASE") or "https://api.openai.com/"


def _is_gemini_code_assist_endpoint(endpoint: str) -> bool:
    normalized_endpoint = endpoint.lstrip("/")
    return normalized_endpoint.startswith("v1internal:")


_GEMINI_CODE_ASSIST_ENDPOINT_ACTION_RE = re.compile(r"^v1internal:[A-Za-z][A-Za-z0-9_]*$")


def _normalize_gemini_code_assist_endpoint_path(endpoint: str) -> str:
    """RR-054 #34: strict path for Code Assist v1internal:action routes.

    httpx.URL(...).path cannot be used because the colon is intentional action
    syntax (not a scheme). Reject anything outside the known shape so query/path
    smuggling cannot ride the OAuth-forwarding lane.
    """
    candidate = endpoint.split("?", 1)[0].strip()
    if not candidate:
        raise HTTPException(
            status_code=400,
            detail="Invalid Gemini Code Assist endpoint path",
        )
    if not candidate.startswith("/"):
        candidate = "/" + candidate
    # Disallow multi-segment smuggling while preserving the single action path.
    body = candidate.lstrip("/")
    if "/" in body or "\\" in body or "://" in body or ".." in body:
        raise HTTPException(
            status_code=400,
            detail="Invalid Gemini Code Assist endpoint path",
        )
    if not _GEMINI_CODE_ASSIST_ENDPOINT_ACTION_RE.fullmatch(body):
        raise HTTPException(
            status_code=400,
            detail="Invalid Gemini Code Assist endpoint action",
        )
    return candidate


def _get_gemini_passthrough_target_base(
    endpoint: str,
    has_google_oauth_bearer: bool,
) -> str:
    if has_google_oauth_bearer and _is_gemini_code_assist_endpoint(endpoint):
        return os.getenv("CODE_ASSIST_ENDPOINT") or "https://cloudcode-pa.googleapis.com"

    return os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"


_iter_antigravity_auth_file_path_candidates = _aawm_antigravity_oauth._iter_antigravity_auth_file_path_candidates


_get_antigravity_auth_file_path = _aawm_antigravity_oauth._get_antigravity_auth_file_path


_load_antigravity_oauth_token_data_from_path = _aawm_antigravity_oauth._load_antigravity_oauth_token_data_from_path


_load_local_antigravity_oauth_token_data = _aawm_antigravity_oauth._load_local_antigravity_oauth_token_data


_parse_antigravity_token_expiry = _aawm_antigravity_oauth._parse_antigravity_token_expiry


_antigravity_access_token_is_valid = _aawm_antigravity_oauth._antigravity_access_token_is_valid


_antigravity_access_token_is_unexpired = _aawm_antigravity_oauth._antigravity_access_token_is_unexpired


_antigravity_oauth_cached_token_is_valid = _aawm_antigravity_oauth._antigravity_oauth_cached_token_is_valid


_get_antigravity_oauth_expiry_date = _aawm_antigravity_oauth._get_antigravity_oauth_expiry_date


def _iter_antigravity_cli_binary_candidates(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._iter_antigravity_cli_binary_candidates(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


_extract_antigravity_oauth_client_values_from_cli_text = (
    _aawm_antigravity_oauth._extract_antigravity_oauth_client_values_from_cli_text
)


_add_antigravity_oauth_client_candidate = _aawm_antigravity_oauth._add_antigravity_oauth_client_candidate


_extract_antigravity_oauth_client_value_candidates_from_cli_text = (
    _aawm_antigravity_oauth._extract_antigravity_oauth_client_value_candidates_from_cli_text
)


_load_antigravity_oauth_client_values_from_local_cli_binary = (
    _aawm_antigravity_oauth._load_antigravity_oauth_client_values_from_local_cli_binary
)


_load_antigravity_oauth_client_value_candidates_from_local_cli_binary = (
    _aawm_antigravity_oauth._load_antigravity_oauth_client_value_candidates_from_local_cli_binary
)


_get_antigravity_oauth_client_value_from_token_data = (
    _aawm_antigravity_oauth._get_antigravity_oauth_client_value_from_token_data
)


_get_antigravity_oauth_client_value_candidates = _aawm_antigravity_oauth._get_antigravity_oauth_client_value_candidates


# RR-054 #1: wire Antigravity OAuth package runtime deps after helpers exist.
_aawm_antigravity_oauth.configure_antigravity_oauth_runtime(
    clean_value=_clean_codex_auth_value,
    get_first_secret_value=_get_first_secret_value,
    invalidate_lane_cache=_invalidate_codex_auto_agent_antigravity_lane_cache,
    write_json_atomic=_write_json_file_atomic,
    iter_cli_binaries=_iter_antigravity_cli_binary_candidates,
    oauth_error_code=_get_oauth_token_error_code,
    format_refresh_failure=_format_oauth_refresh_failure_detail,
)


_write_antigravity_oauth_token_data_atomic = _aawm_antigravity_oauth._write_antigravity_oauth_token_data_atomic


_get_antigravity_cli_refresh_home = _aawm_antigravity_oauth._get_antigravity_cli_refresh_home


_get_antigravity_cli_refresh_timeout_seconds = _aawm_antigravity_oauth._get_antigravity_cli_refresh_timeout_seconds


_refresh_local_antigravity_oauth_token_data_via_cli = (
    _aawm_antigravity_oauth._refresh_local_antigravity_oauth_token_data_via_cli
)


_refresh_local_antigravity_oauth_token_data = _aawm_antigravity_oauth._refresh_local_antigravity_oauth_token_data


_load_valid_local_antigravity_access_token = _aawm_antigravity_oauth._load_valid_local_antigravity_access_token


def _get_antigravity_passthrough_target_base(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._get_antigravity_passthrough_target_base(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


def _get_antigravity_client_header(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._get_antigravity_client_header(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


@lru_cache(maxsize=1)
def _get_anthropic_antigravity_runtime():
    return _wave6b_antigravity_runtime._get_anthropic_antigravity_runtime(runtime=_wave6b_antigravity_live_runtime())


def _build_antigravity_native_headers(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._build_antigravity_native_headers(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


_request_has_google_oauth_bearer = _wave6b_antigravity_runtime._request_has_google_oauth_bearer


def _get_antigravity_litellm_auth_header(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._get_antigravity_litellm_auth_header(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


def _prepare_antigravity_request_body_for_passthrough(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._prepare_antigravity_request_body_for_passthrough(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


def _get_antigravity_request_project(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._get_antigravity_request_project(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


def _get_antigravity_passthrough_logging_metadata(*__args, **__kwargs):
    return _wave6b_antigravity_runtime._get_antigravity_passthrough_logging_metadata(*__args, runtime=_wave6b_antigravity_live_runtime(), **__kwargs)


_normalize_antigravity_endpoint_for_target = _wave6b_antigravity_runtime._normalize_antigravity_endpoint_for_target


_join_antigravity_passthrough_url = _wave6b_antigravity_runtime._join_antigravity_passthrough_url


_is_antigravity_streaming_endpoint = _wave6b_antigravity_runtime._is_antigravity_streaming_endpoint


def _get_grok_passthrough_target_base() -> str:
    return (
        os.getenv("GROK_CLI_CHAT_PROXY_UPSTREAM_BASE_URL")
        or os.getenv("XAI_CLI_CHAT_PROXY_BASE_URL")
        or _GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL
    )


_normalize_grok_endpoint_for_target = _grok_side_channel._normalize_grok_endpoint_for_target


def _join_grok_passthrough_url(base_target_url: str, endpoint: str) -> str:
    return BaseOpenAIPassThroughHandler._join_url_paths(
        base_url=httpx.URL(base_target_url),
        path=_normalize_grok_endpoint_for_target(
            endpoint=endpoint,
            base_target_url=base_target_url,
        ),
        custom_llm_provider=litellm.LlmProviders.XAI,
    )


def _get_case_insensitive_header(headers: dict[str, Any], header_name: str) -> Optional[str]:
    wanted = header_name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted and value is not None:
            value_str = str(value).strip()
            if value_str:
                return value_str
    return None


def _format_litellm_passthrough_api_key(api_key: Optional[str]) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        return ""
    cleaned = api_key.strip()
    if cleaned.lower().startswith("bearer "):
        return cleaned
    return f"Bearer {cleaned}"


def _get_grok_litellm_auth_header(request: Request) -> str:
    header_key = request.headers.get("x-litellm-api-key")
    if header_key:
        return _format_litellm_passthrough_api_key(header_key)

    query_key = request.query_params.get("key")
    if query_key:
        return _format_litellm_passthrough_api_key(query_key)

    # RR-054 #48: normalize Authorization the same way as explicit key sources.
    authorization = request.headers.get("Authorization") or request.headers.get("authorization")
    if authorization:
        return _format_litellm_passthrough_api_key(authorization)
    return ""


def _prepare_grok_logging_body_for_passthrough(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    headers = _safe_get_request_headers(request)
    header_model_override = _get_case_insensitive_header(
        headers,
        "x-grok-model-override",
    )
    body_model_override = normalize_grok_native_oauth_model(request_body.get("model"))
    model_override = header_model_override or body_model_override
    session_id = _get_grok_native_oauth_session_id(
        request=request,
        request_body=request_body,
    )

    extra_fields: dict[str, Any] = {
        "client_name": "grok-build",
        "grok_cli_chat_proxy": True,
        "passthrough_route_family": "grok_cli_chat_proxy",
        "xai_cli_chat_proxy": True,
    }
    tags_to_add = ["grok-build", "route:grok_cli_chat_proxy"]
    if model_override:
        normalized_model_override = normalize_grok_native_oauth_model(model_override) or model_override
        extra_fields["grok_model_override"] = normalized_model_override
        extra_fields["model_group"] = normalized_model_override
        tags_to_add.append(f"grok-model:{normalized_model_override}")
    if session_id:
        extra_fields["session_id"] = session_id

    updated_body = copy.deepcopy(request_body)
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )
    return _prepare_request_body_for_passthrough_observability(
        request=request,
        request_body=updated_body,
    )


def _prepare_grok_request_body_for_passthrough(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    prepared_body = _prepare_grok_logging_body_for_passthrough(
        request=request,
        request_body=request_body,
    )
    (
        prepared_body,
        _grok_unsupported_request_params,
    ) = _drop_unsupported_codex_request_params_from_request_body(prepared_body)
    (
        prepared_body,
        _grok_unsupported_input_items,
    ) = _drop_unsupported_codex_input_items_from_request_body(prepared_body)
    _sanitize_grok_native_function_call_arguments_in_place(prepared_body)
    _rewrite_grok_native_unsupported_input_items_in_place(prepared_body)
    (
        prepared_body,
        _removed_tool_choice,
    ) = _drop_tool_choice_without_tools_from_request_body(prepared_body)
    return prepared_body


def _get_grok_passthrough_logging_metadata(request: Request) -> dict[str, Any]:
    logging_body = _prepare_grok_logging_body_for_passthrough(
        request=request,
        request_body={},
    )
    litellm_metadata = logging_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        return dict(litellm_metadata)
    return {}


def _is_grok_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").lower()
    return not content_type or "application/json" in content_type or content_type.endswith("+json")


def _is_grok_storage_endpoint(endpoint: str) -> bool:
    endpoint_path = httpx.URL(endpoint).path
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    if endpoint_path.startswith("/v1/"):
        endpoint_path = endpoint_path[len("/v1") :]
    return endpoint_path == "/storage" or endpoint_path.startswith("/storage/")


def _is_grok_coding_data_retention_endpoint(endpoint: str) -> bool:
    endpoint_path = _normalize_grok_endpoint_path(endpoint)
    return endpoint_path == "/privacy/coding-data-retention"


_normalize_grok_endpoint_path = _grok_side_channel._normalize_grok_endpoint_path


_get_grok_side_channel_endpoint_type = _grok_side_channel._get_grok_side_channel_endpoint_type


_get_grok_session_side_channel_endpoint_type = _grok_side_channel._get_grok_session_side_channel_endpoint_type


_get_grok_side_channel_endpoint_path_template = _grok_side_channel._get_grok_side_channel_endpoint_path_template


_get_grok_session_side_channel_endpoint_path_template = _grok_side_channel._get_grok_session_side_channel_endpoint_path_template


_json_shape_type_name = _grok_side_channel._json_shape_type_name


_extract_redacted_grok_json_request_shape = _grok_side_channel._extract_redacted_grok_json_request_shape


_stable_grok_side_channel_body_digest = _grok_side_channel._stable_grok_side_channel_body_digest


_build_grok_side_channel_request_shape_metadata = _grok_side_channel._build_grok_side_channel_request_shape_metadata


_merge_grok_side_channel_shape_into_passthrough_logging_metadata = _grok_side_channel._merge_grok_side_channel_shape_into_passthrough_logging_metadata


_get_grok_side_channel_retryable_status_codes = _grok_side_channel._get_grok_side_channel_retryable_status_codes


def _log_grok_forward_header_compare(
    *,
    endpoint: str,
    request: Request,
) -> None:
    incoming_headers = {str(header_name).lower() for header_name in _safe_get_request_headers(request).keys()}
    allowed_headers = {header.lower() for header in _GROK_CLI_FORWARD_HEADER_ALLOWLIST}
    forwarded_headers = sorted(incoming_headers & allowed_headers)
    stripped_headers = sorted(
        header
        for header in incoming_headers - allowed_headers
        if header not in _GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE and not header.startswith("x-pass-")
    )

    if not stripped_headers and os.getenv("AAWM_GROK_ROUTE_DEBUG") != "1":
        return

    verbose_proxy_logger.warning(
        "Grok passthrough header compare: endpoint=%s forwarded=%s stripped=%s",
        endpoint,
        forwarded_headers,
        stripped_headers,
    )


def create_request_copy(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": _safe_get_request_headers(request).copy(),
        "cookies": request.cookies,
        "query_params": dict(request.query_params),
    }


def is_passthrough_request_using_router_model(request_body: dict, llm_router: Optional[litellm.Router]) -> bool:
    """
    Returns True if the model is in the llm_router model names
    """
    try:
        model = request_body.get("model")
        return is_known_model(model, llm_router)
    except Exception:
        return False


def is_passthrough_request_streaming(request_body: dict) -> bool:
    """
    Returns True if the request is streaming
    """
    return request_body.get("stream", False)


async def llm_passthrough_factory_proxy_route(
    custom_llm_provider: str,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Factory function for creating pass-through endpoints for LLM providers.
    """
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    provider_config = ProviderConfigManager.get_provider_model_info(
        provider=LlmProviders(custom_llm_provider),
        model=None,
    )
    if provider_config is None:
        raise HTTPException(status_code=404, detail=f"Provider {custom_llm_provider} not found")

    base_target_url = provider_config.get_api_base()

    if base_target_url is None:
        raise HTTPException(status_code=404, detail=f"Provider {custom_llm_provider} api base not found")

    if _is_gemini_code_assist_endpoint(endpoint):
        encoded_endpoint = _normalize_gemini_code_assist_endpoint_path(endpoint)
    else:
        encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    # Join paths correctly by removing trailing/leading slashes as needed
    if not base_url.path or base_url.path == "/":
        # If base URL has no path, just use the new path
        updated_url = base_url.copy_with(path=encoded_endpoint)
    else:
        # Otherwise, combine the paths
        base_path = base_url.path.rstrip("/")
        clean_path = encoded_endpoint.lstrip("/")
        full_path = f"{base_path}/{clean_path}"
        updated_url = base_url.copy_with(path=full_path)

    # Add or update query parameters
    provider_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider=custom_llm_provider,
        region_name=None,
    )

    auth_headers = provider_config.validate_environment(
        headers={},
        model="",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=provider_api_key,
        api_base=base_target_url,
    )

    ## check for streaming
    is_streaming_request = False
    # anthropic is streaming when 'stream' = True is in the body
    if request.method == "POST":
        if "multipart/form-data" not in request.headers.get("content-type", ""):
            _request_body = await request.json()
        else:
            _request_body = await get_form_data(request)

        if _request_body.get("stream"):
            is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=auth_headers,
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/gemini/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Google AI Studio Pass-through", "pass-through"],
)
async def gemini_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/google_ai_studio)
    """
    ## CHECK FOR LITELLM API KEY IN THE QUERY PARAMS - ?..key=LITELLM_API_KEY
    google_ai_studio_api_key = request.query_params.get("key") or request.headers.get("x-goog-api-key")
    # RR-054 #41: never synthesize "Bearer None" when the client omitted both key sources.
    auth_api_key: Optional[str]
    if isinstance(google_ai_studio_api_key, str) and google_ai_studio_api_key.strip():
        auth_api_key = f"Bearer {google_ai_studio_api_key.strip()}"
    else:
        auth_api_key = request.headers.get("Authorization") or request.headers.get("authorization")
    if not isinstance(auth_api_key, str) or not auth_api_key.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing Google AI Studio pass-through API key.",
        )
    user_api_key_dict = await user_api_key_auth(request=request, api_key=auth_api_key)

    _auth_header = request.headers.get("authorization", "")
    _is_google_oauth = _auth_header.startswith("Bearer ya29.")

    base_target_url = _get_gemini_passthrough_target_base(
        endpoint=endpoint,
        has_google_oauth_bearer=_is_google_oauth,
    )
    if _is_gemini_code_assist_endpoint(endpoint):
        encoded_endpoint = _normalize_gemini_code_assist_endpoint_path(endpoint)
    else:
        encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    merged_params = dict(request.query_params)
    if _is_google_oauth:
        # Remove the 'key' param if the client sent one; Google OAuth auth
        # does not use API key query params.
        merged_params.pop("key", None)
    else:
        gemini_api_key: Optional[str] = passthrough_endpoint_router.get_credentials(
            custom_llm_provider="gemini",
            region_name=None,
        )
        if gemini_api_key is None:
            raise Exception(
                "Required 'GEMINI_API_KEY'/'GOOGLE_API_KEY' in environment to make pass-through calls to Google AI Studio."
            )
        # Merge query parameters, giving precedence to those in updated_url
        merged_params.update({"key": gemini_api_key})

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    if request.method == "POST":
        request_body = await get_request_body(request)
        if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1" and _is_google_oauth:
            debug_headers = _build_google_debug_header_summary(dict(request.headers))
            debug_body_summary = _summarize_google_code_assist_request_shape(request_body)
            request_block = (
                request_body.get("request")
                if isinstance(request_body, dict) and isinstance(request_body.get("request"), dict)
                else request_body
            )
            function_names = _extract_google_code_assist_function_names(request_block)
            verbose_proxy_logger.info(
                "Gemini passthrough debug: endpoint=%s headers=%s body_shape=%s function_names=%s",
                endpoint,
                debug_headers,
                debug_body_summary,
                function_names,
            )
        prepared_request_body = _add_gemini_request_breakout_logging_metadata(request_body)
        gemini_route_family = _get_gemini_passthrough_route_family(endpoint)
        if gemini_route_family is not None:
            prepared_request_body = _add_route_family_logging_metadata(
                prepared_request_body,
                gemini_route_family,
            )
        prepared_request_body = _prepare_request_body_for_passthrough_observability(
            request=request,
            request_body=prepared_request_body,
        )
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_llm_provider="gemini",
        _forward_headers=_is_google_oauth,
        is_streaming_request=is_streaming_request,
        query_params=merged_params,
        egress_credential_family="google" if _is_google_oauth else None,
        expected_target_family="google",
        allowed_forward_headers=(list(_GEMINI_OAUTH_FORWARD_HEADER_ALLOWLIST) if _is_google_oauth else None),
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/opencode/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["OpenCode Zen Pass-through", "pass-through"],
)
async def opencode_zen_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Native OpenCode Zen pass-through.

    OpenCode stores a provider-scoped API credential at
    `~/.local/share/opencode/auth.json`. LiteLLM auth should be supplied
    separately with `x-litellm-api-key` or a `key` query parameter.
    """
    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=_get_antigravity_litellm_auth_header(request),
    )

    target_url = _join_opencode_zen_passthrough_url(
        base_target_url=_get_opencode_zen_target_base(),
        endpoint=endpoint,
    )
    query_params = {key: value for key, value in dict(request.query_params).items() if str(key).lower() != "key"}

    custom_body: Optional[dict[str, Any]] = None
    stream = False
    passthrough_logging_metadata: dict[str, Any] = {
        "client_name": "opencode-zen",
        "opencode_zen": True,
        "passthrough_route_family": "opencode_zen",
        "tags": ["route:opencode_zen", "opencode-zen"],
    }
    if request.method in {"POST", "PUT", "PATCH"}:
        request_body = await get_request_body(request)
        if isinstance(request_body, dict):
            custom_body = _add_opencode_zen_logging_metadata(
                request_body,
                route_family="opencode_zen",
                tag_prefix="opencode-zen",
                requested_model=request_body.get("model"),
                client_name="opencode-zen",
            )
            custom_body = _prepare_request_body_for_passthrough_observability(
                request=request,
                request_body=custom_body,
            )
            if custom_body is not request_body:
                _safe_set_request_parsed_body(request, custom_body)
            stream = bool(custom_body.get("stream"))
            custom_metadata = custom_body.get("litellm_metadata")
            if isinstance(custom_metadata, dict):
                passthrough_logging_metadata = dict(custom_metadata)

    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))
    return await pass_through_request(
        request=request,
        target=target_url,
        custom_headers=await _build_opencode_zen_headers(request),
        user_api_key_dict=user_api_key_dict,
        custom_body=custom_body,
        forward_headers=False,
        query_params=query_params,
        stream=stream,
        custom_llm_provider=_OPENCODE_ZEN_PROVIDER,
        egress_credential_family="opencode",
        expected_target_family="opencode",
        allowed_forward_headers=[],
        allowed_pass_through_prefixed_headers=[],
        passthrough_logging_metadata=passthrough_logging_metadata,
    )


@router.api_route(
    "/antigravity/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Antigravity Code Assist Pass-through", "pass-through"],
)
async def antigravity_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Native Antigravity CLI pass-through for Google Code Assist.

    Antigravity uses its own Google OAuth credential and Code Assist client
    headers. LiteLLM auth should be supplied separately with
    `x-litellm-api-key` or a `key` query parameter when preserving an inbound
    Google OAuth Authorization header.
    """
    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=_get_antigravity_litellm_auth_header(request),
    )

    has_google_oauth_bearer = _request_has_google_oauth_bearer(request)
    local_antigravity_access_token: Optional[str] = None
    custom_headers: dict[str, str]
    if has_google_oauth_bearer:
        custom_headers = {}
    else:
        local_antigravity_access_token = await _load_valid_local_antigravity_access_token()
        custom_headers = _build_antigravity_native_headers(local_antigravity_access_token)

    target_url = _join_antigravity_passthrough_url(
        base_target_url=_get_antigravity_passthrough_target_base(),
        endpoint=endpoint,
    )
    query_params = {key: value for key, value in dict(request.query_params).items() if str(key).lower() != "key"}

    custom_body: Optional[dict[str, Any]] = None
    passthrough_logging_metadata = _get_antigravity_passthrough_logging_metadata(request)
    if request.method in {"POST", "PUT", "PATCH"}:
        request_body = await get_request_body(request)
        if isinstance(request_body, dict):
            custom_body = _prepare_antigravity_request_body_for_passthrough(
                request=request,
                request_body=request_body,
            )
            request_project = _get_antigravity_request_project(request_body)
            if local_antigravity_access_token is not None and request_project is not None:
                google_quota_observation = await _prime_google_code_assist_session(
                    local_antigravity_access_token,
                    request_project,
                    adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
                )
                if google_quota_observation:
                    litellm_metadata = custom_body.setdefault(
                        "litellm_metadata",
                        {},
                    )
                    if isinstance(litellm_metadata, dict):
                        litellm_metadata["google_retrieve_user_quota"] = google_quota_observation
            if custom_body is not request_body:
                _safe_set_request_parsed_body(request, custom_body)
            custom_metadata = custom_body.get("litellm_metadata")
            if isinstance(custom_metadata, dict):
                passthrough_logging_metadata = dict(custom_metadata)

    return await pass_through_request(
        request=request,
        target=target_url,
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=custom_body,
        forward_headers=has_google_oauth_bearer,
        query_params=query_params,
        stream=_is_antigravity_streaming_endpoint(endpoint, request),
        custom_llm_provider="antigravity",
        egress_credential_family="google",
        expected_target_family="google",
        allowed_forward_headers=(list(_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST) if has_google_oauth_bearer else None),
        passthrough_logging_metadata=passthrough_logging_metadata,
    )


@router.api_route(
    "/grok/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Grok Build Pass-through", "pass-through"],
)
async def grok_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Native Grok Build pass-through for the xAI CLI chat proxy.

    Grok Build keeps its own OIDC Authorization header and xAI routing headers.
    LiteLLM auth should be supplied separately with `x-litellm-api-key` or a
    `key` query parameter so the upstream Authorization header can remain intact.
    """
    is_storage_endpoint = _is_grok_storage_endpoint(endpoint)
    is_coding_data_retention_endpoint = _is_grok_coding_data_retention_endpoint(endpoint)
    raw_body_passthrough = request.method in {"POST", "PUT", "PATCH"} and (
        is_storage_endpoint or is_coding_data_retention_endpoint or not _is_grok_json_request(request)
    )
    if raw_body_passthrough:
        _safe_set_request_parsed_body(request, {})

    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=_get_grok_litellm_auth_header(request),
    )

    if is_storage_endpoint:
        return {
            "ok": True,
            "suppressed": True,
            "endpoint": "grok_storage",
        }

    if is_coding_data_retention_endpoint:
        return {
            "ok": True,
            "suppressed": True,
            "endpoint": "grok_coding_data_retention",
        }

    base_target_url = _get_grok_passthrough_target_base()
    target_url = _join_grok_passthrough_url(
        base_target_url=base_target_url,
        endpoint=endpoint,
    )

    _log_grok_forward_header_compare(endpoint=endpoint, request=request)

    custom_body: Optional[dict[str, Any]] = None
    custom_headers: dict[str, str] = {}
    passthrough_logging_metadata = _get_grok_passthrough_logging_metadata(request)
    upstream_request_body_for_shape: Any = None
    upstream_raw_body_for_shape: Optional[bytes] = None
    if request.method in {"POST", "PUT", "PATCH"}:
        if not raw_body_passthrough:
            request_body = await get_request_body(request)
            upstream_request_body_for_shape = request_body
            if isinstance(request_body, dict):
                custom_body = _prepare_grok_request_body_for_passthrough(
                    request=request,
                    request_body=request_body,
                )
                if custom_body is not request_body:
                    _safe_set_request_parsed_body(request, custom_body)
                custom_metadata = custom_body.get("litellm_metadata")
                if isinstance(custom_metadata, dict):
                    passthrough_logging_metadata = dict(custom_metadata)
                    grok_model_override = normalize_grok_native_oauth_model(custom_metadata.get("grok_model_override"))
                    if grok_model_override is not None and not _get_case_insensitive_header(
                        _safe_get_request_headers(request),
                        "x-grok-model-override",
                    ):
                        custom_headers["x-grok-model-override"] = grok_model_override
        elif _get_grok_side_channel_endpoint_type(endpoint) is not None:
            upstream_raw_body_for_shape = await request.body()

    side_channel_shape_metadata = _build_grok_side_channel_request_shape_metadata(
        endpoint=endpoint,
        request=request,
        parsed_body=upstream_request_body_for_shape,
        raw_body=upstream_raw_body_for_shape,
    )
    if side_channel_shape_metadata:
        passthrough_logging_metadata = _merge_grok_side_channel_shape_into_passthrough_logging_metadata(
            passthrough_logging_metadata,
            shape_metadata=side_channel_shape_metadata,
        )
        side_channel_shape_log = (
            verbose_proxy_logger.info if os.getenv("AAWM_GROK_ROUTE_DEBUG") == "1" else verbose_proxy_logger.debug
        )
        side_channel_shape_log(
            "Grok passthrough side-channel request shape: endpoint_type=%s body_byte_length=%s body_sha256=%s json_container_type=%s top_level_key_types=%s",
            side_channel_shape_metadata.get("grok_side_channel_endpoint_type"),
            side_channel_shape_metadata.get("grok_side_channel_request_body_byte_length"),
            side_channel_shape_metadata.get("grok_side_channel_request_body_sha256"),
            side_channel_shape_metadata.get("grok_side_channel_request_json_container_type"),
            side_channel_shape_metadata.get("grok_side_channel_request_top_level_key_types"),
        )

    query_params = {key: value for key, value in dict(request.query_params).items() if str(key).lower() != "key"}
    grok_side_channel_retryable_status_codes = _get_grok_side_channel_retryable_status_codes(endpoint)

    return await pass_through_request(
        request=request,
        target=target_url,
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=custom_body,
        forward_headers=True,
        query_params=query_params,
        stream="stream" in str(target_url),
        custom_llm_provider=litellm.LlmProviders.XAI.value,
        egress_credential_family="xai",
        expected_target_family="xai",
        allowed_forward_headers=list(_GROK_CLI_FORWARD_HEADER_ALLOWLIST),
        raw_body_passthrough=raw_body_passthrough,
        passthrough_logging_metadata=passthrough_logging_metadata,
        retryable_upstream_status_codes=grok_side_channel_retryable_status_codes,
        caller_managed_hidden_retry=bool(grok_side_channel_retryable_status_codes),
    )


@router.api_route(
    "/cohere/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Cohere Pass-through", "pass-through"],
)
async def cohere_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/cohere)
    """
    base_target_url = os.getenv("COHERE_API_BASE") or "https://api.cohere.com"
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    cohere_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="cohere",
        region_name=None,
    )

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "Bearer {}".format(cohere_api_key)},
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/vllm/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["VLLM Pass-through", "pass-through"],
)
async def vllm_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/vllm)
    """
    from litellm.proxy.proxy_server import llm_router

    request_body = await get_request_body(request)
    is_router_model = is_passthrough_request_using_router_model(request_body, llm_router)
    is_streaming_request = is_passthrough_request_streaming(request_body)
    if is_router_model and llm_router:
        result = cast(
            httpx.Response,
            await llm_router.allm_passthrough_route(
                model=request_body.get("model"),
                method=request.method,
                endpoint=endpoint,
                request_query_params=request.query_params,
                request_headers=_safe_get_request_headers(request),
                stream=request_body.get("stream", False),
                content=None,
                data=None,
                files=None,
                json=(request_body if request.headers.get("content-type") == "application/json" else None),
                params=None,
                headers=None,
                cookies=None,
            ),
        )

        if is_streaming_request:
            return StreamingResponse(
                content=result.aiter_bytes(),
                status_code=result.status_code,
                headers=HttpPassThroughEndpointHelpers.get_response_headers(
                    headers=result.headers,
                    custom_headers=None,
                ),
            )

        content = await result.aread()
        return Response(
            content=content,
            status_code=result.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                headers=result.headers,
                custom_headers=None,
            ),
        )

    return await llm_passthrough_factory_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        custom_llm_provider="vllm",
    )


@router.api_route(
    "/mistral/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Mistral Pass-through", "pass-through"],
)
async def mistral_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/mistral)
    """
    base_target_url = os.getenv("MISTRAL_API_BASE") or "https://api.mistral.ai"
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    mistral_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="mistral",
        region_name=None,
    )

    ## check for streaming
    is_streaming_request = False
    # anthropic is streaming when 'stream' = True is in the body
    if request.method == "POST":
        _request_body = await request.json()
        if _request_body.get("stream"):
            is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "Bearer {}".format(mistral_api_key)},
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/milvus/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Milvus Pass-through", "pass-through"],
)
async def milvus_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Enable using Milvus `/vectors` endpoint as a pass-through endpoint.
    """

    provider_config = ProviderConfigManager.get_provider_vector_stores_config(provider=LlmProviders.MILVUS)
    if not provider_config:
        raise HTTPException(
            status_code=500,
            detail="Unable to find Milvus vector store config.",
        )

    # check if managed vector store index is used
    request_body = await get_request_body(request)

    # check collectionName
    collection_name = cast(Optional[str], request_body.get("collectionName"))
    extra_headers = {}
    base_target_url: Optional[str] = None
    if not collection_name:
        raise HTTPException(
            status_code=400,
            detail=f"Collection name is required. Got {request_body}",
        )

    if not litellm.vector_store_index_registry or not litellm.vector_store_registry:
        raise HTTPException(
            status_code=500,
            detail="Unable to find Milvus vector store index registry or vector store registry.",
        )

    # check if vector store index
    is_vector_store_index = litellm.vector_store_index_registry.is_vector_store_index(
        vector_store_index_name=collection_name
    )

    if not is_vector_store_index:
        raise HTTPException(
            status_code=400,
            detail=f"Collection {collection_name} is not a litellm managed vector store index. Only litellm managed vector store indexes are supported.",
        )

    is_allowed_to_call_vector_store_endpoint(
        index_name=collection_name,
        provider=LlmProviders.MILVUS,
        request=request,
        user_api_key_dict=user_api_key_dict,
    )
    # get the vector store name from index registry

    index_object = (
        (litellm.vector_store_index_registry.get_vector_store_index_by_name(vector_store_index_name=collection_name))
        if litellm.vector_store_index_registry is not None
        else None
    )
    if index_object is None:
        raise Exception(f"Vector store index not found for {collection_name}")

    vector_store_name = index_object.litellm_params.vector_store_name
    vector_store_index = index_object.litellm_params.vector_store_index

    request_body["collectionName"] = vector_store_index

    # Update the request object with the modified collection name
    _safe_set_request_parsed_body(request, request_body)

    vector_store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry_by_name(
        vector_store_name=vector_store_name
    )
    if vector_store is None:
        raise Exception(f"Vector store not found for {vector_store_name}")
    litellm_params = vector_store.get("litellm_params") or {}
    auth_credentials = provider_config.get_auth_credentials(litellm_params=litellm_params)

    extra_headers = auth_credentials.get("headers") or {}

    litellm_params = vector_store.get("litellm_params") or {}

    base_target_url = provider_config.get_complete_url(
        api_base=litellm_params.get("api_base"), litellm_params=litellm_params
    )

    if base_target_url is None:
        raise Exception(f"api_base not found in vector store configuration for {vector_store_name}")

    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)
    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=extra_headers,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


async def is_streaming_request_fn(request: Request) -> bool:
    if request.method == "POST":
        content_type = request.headers.get("content-type", None)
        if content_type and "multipart/form-data" in content_type:
            _request_body = await get_form_data(request)
        else:
            _request_body = await _read_request_body(request)
        if _request_body.get("stream"):
            return True
    return False


async def _dispatch_auto_agent_alias_candidate_request(
    *,
    candidate: Payload,
    provider_handlers: Mapping[str, Callable[[], Awaitable[Response]]],
    default_handler: Callable[[], Awaitable[Response]],
    route_family_handlers: Optional[Mapping[str, Mapping[str, Callable[[], Awaitable[Response]]]]] = None,
) -> Response:
    """Table-driven provider/route_family candidate dispatch (RR-054 #10).

    Anthropic and Codex families keep different handler callables, but share one
    dispatch shape so provider branching does not re-grow divergent control flow.
    """
    provider = str(candidate.get("provider") or "")
    route_family = str(candidate.get("route_family") or "")
    if route_family_handlers and provider in route_family_handlers:
        family_map = route_family_handlers[provider]
        handler = family_map.get(route_family) or family_map.get("*")
        if handler is not None:
            return await handler()
    handler = provider_handlers.get(provider)
    if handler is not None:
        return await handler()
    return await default_handler()


async def _perform_anthropic_auto_agent_alias_candidate_request(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    candidate: dict[str, Any],
    candidate_body: dict[str, Any],
    target_url: str,
    custom_headers: dict[str, Any],
) -> Response:
    adapter_model = candidate["model"]

    async def _openai() -> Response:
        return await _handle_anthropic_openai_responses_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _antigravity() -> Response:
        return await _handle_anthropic_google_completion_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
            use_alias_candidate_probe=True,
        )

    async def _google() -> Response:
        return await _handle_anthropic_google_completion_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_completion() -> Response:
        return await _handle_anthropic_openrouter_completion_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_responses() -> Response:
        return await _handle_anthropic_openrouter_responses_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _xai_oauth() -> Response:
        return await _handle_anthropic_xai_oauth_responses_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _grok_native() -> Response:
        return await _handle_anthropic_grok_native_oauth_responses_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _opencode() -> Response:
        return await _handle_anthropic_opencode_zen_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _kimi_code() -> Response:
        return await _handle_anthropic_kimi_chat_completions_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _alibaba_token_plan() -> Response:
        return await _handle_anthropic_alibaba_token_plan_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _native() -> Response:
        native_candidate_body = candidate_body
        native_custom_headers = custom_headers
        blocked_pass_through_prefixed_headers: Optional[list[str]] = None
        (
            native_candidate_body,
            _normalized_native_model_alias,
        ) = _normalize_anthropic_native_passthrough_model_alias(native_candidate_body)
        (
            native_candidate_body,
            native_custom_headers,
            normalized_context_1m_model,
        ) = _prepare_anthropic_context_1m_native_passthrough(
            request=request,
            request_body=native_candidate_body,
            custom_headers=native_custom_headers,
        )
        if normalized_context_1m_model:
            blocked_pass_through_prefixed_headers = [_ANTHROPIC_BETA_HEADER_NAME]
        _safe_set_request_parsed_body(request, native_candidate_body)
        return await _perform_anthropic_native_passthrough_request(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            target_url=target_url,
            custom_headers=native_custom_headers,
            blocked_pass_through_prefixed_headers=blocked_pass_through_prefixed_headers,
        )

    return await _dispatch_auto_agent_alias_candidate_request(
        candidate=candidate,
        provider_handlers={
            _CODEX_AUTO_AGENT_NATIVE_PROVIDER: _openai,
            _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER: _antigravity,
            _CODEX_AUTO_AGENT_GOOGLE_PROVIDER: _google,
            _CODEX_AUTO_AGENT_OPENCODE_PROVIDER: _opencode,
            _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER: _kimi_code,
            _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER: _alibaba_token_plan,
        },
        route_family_handlers={
            _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER: {
                "anthropic_openrouter_completion_adapter": _openrouter_completion,
                "*": _openrouter_responses,
            },
            _CODEX_AUTO_AGENT_XAI_PROVIDER: {
                "anthropic_xai_oauth_responses_adapter": _xai_oauth,
                "*": _grok_native,
            },
        },
        default_handler=_native,
    )


_AutoAgentAliasSelectionFn = Callable[..., Awaitable[dict[str, Any]]]
_AutoAgentAliasMetadataFn = Callable[..., dict[str, Any]]


async def _handle_auto_agent_alias_route(  # noqa: PLR0915
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: Payload,
    max_candidate_attempts: int,
    select_candidate_fn: _AutoAgentAliasSelectionFn,
    add_alias_metadata_fn: _AutoAgentAliasMetadataFn,
    perform_candidate_request_fn: Callable[..., Awaitable[Response]],
    get_active_cooldown_state_fn: Callable[[str], Awaitable[tuple[float, str]]],
    set_session_affinity_fn: Callable[..., Awaitable[object]],
    apply_cooldown_fn: Callable[..., Awaitable[str]],
    raise_redispatch_required_fn: Callable[..., None],
    attempts_metadata_key: str,
    skipped_candidates_metadata_key: str,
    no_candidate_detail: str,
    log_label: str,
) -> Response:
    """Shared Anthropic/Codex auto-agent alias candidate loop (RR-054 #10).

    Thin façade that adapts the legacy per-call seam callables into the typed
    :class:`AliasRouteServices` bundle and delegates to
    :func:`aawm_alias_routing.candidate_loop.handle_alias_route`, which owns the
    R3-1 widened-lock single-flight publication. The production wrappers
    (``_handle_codex_auto_agent_alias_route`` /
    ``_handle_anthropic_auto_agent_alias_route``) build the services directly;
    this façade keeps the legacy seam contract for the RR-054 single-flight
    tests. Process-local publication uses the same synchronous family-memory
    writer as production. The legacy async applicator is isolated in the
    post-release persistence callback and never enters the typed synchronous
    publisher contract.
    """
    legacy_request: Optional[Request] = None
    legacy_candidate: dict[str, Any] = {}
    legacy_lane_key: Optional[str] = None
    legacy_selected_cooldown_key = ""
    legacy_cooldown_seconds = 0.0
    legacy_error_class: Optional[str] = None
    legacy_grok_account_quota_exhausted = False
    legacy_kimi_failure_metadata: Optional[dict[str, Any]] = None
    legacy_is_read_pilot_lane = False
    family_state = (
        _alias_routing_state.anthropic
        if alias_family == "anthropic_auto_agent"
        else _alias_routing_state.codex
    )

    def _legacy_resolve_publication(
        *,
        request: Optional[Request],
        candidate: dict[str, Any],
        lane_key: Optional[str],
        selected_cooldown_key: str,
        cooldown_seconds: float,
        error_class: Optional[str],
        grok_account_quota_exhausted: bool = False,
        kimi_failure_metadata: Optional[dict[str, Any]] = None,
        is_read_pilot_lane: bool = False,
    ) -> _aawm_alias_interfaces.CooldownPublicationPlan:
        nonlocal legacy_request
        nonlocal legacy_candidate
        nonlocal legacy_lane_key
        nonlocal legacy_selected_cooldown_key
        nonlocal legacy_cooldown_seconds
        nonlocal legacy_error_class
        nonlocal legacy_grok_account_quota_exhausted
        nonlocal legacy_kimi_failure_metadata
        nonlocal legacy_is_read_pilot_lane
        legacy_request = request
        legacy_candidate = candidate
        legacy_lane_key = lane_key
        legacy_selected_cooldown_key = selected_cooldown_key
        legacy_cooldown_seconds = cooldown_seconds
        legacy_error_class = error_class
        legacy_grok_account_quota_exhausted = grok_account_quota_exhausted
        legacy_kimi_failure_metadata = kimi_failure_metadata
        legacy_is_read_pilot_lane = is_read_pilot_lane
        return _resolve_auto_agent_cooldown_publication_plan(
            request=request,
            candidate=candidate,
            lane_key=lane_key,
            selected_cooldown_key=selected_cooldown_key,
            cooldown_seconds=cooldown_seconds,
            error_class=error_class,
            grok_account_quota_exhausted=grok_account_quota_exhausted,
            kimi_failure_metadata=kimi_failure_metadata,
            is_read_pilot_lane=is_read_pilot_lane,
        )

    def _legacy_publish_memory(*, keys: Sequence[str], seconds: float) -> None:
        for key in keys:
            family_state.set_cooldown_memory(key, seconds)

    async def _legacy_persist(*, keys: Sequence[str], seconds: float) -> None:
        if legacy_request is None:
            raise RuntimeError("legacy cooldown resolver did not capture a request")
        await apply_cooldown_fn(
            request=legacy_request,
            candidate=legacy_candidate,
            lane_key=legacy_lane_key,
            selected_cooldown_key=legacy_selected_cooldown_key,
            cooldown_seconds=legacy_cooldown_seconds,
            error_class=legacy_error_class,
            grok_account_quota_exhausted=legacy_grok_account_quota_exhausted,
            kimi_failure_metadata=legacy_kimi_failure_metadata,
            is_read_pilot_lane=legacy_is_read_pilot_lane,
        )

    async def _legacy_get_active_cooldown_state(
        cooldown_key: str,
    ) -> tuple[float, str]:
        memory_seconds = family_state.get_memory_cooldown_remaining(cooldown_key)
        if memory_seconds > 0:
            return memory_seconds, "memory"
        return await get_active_cooldown_state_fn(cooldown_key)

    # The legacy seam callables are type-erased (``Callable[..., ...]``); cast
    # them to the typed protocols at this bridge boundary. The production
    # wrappers pass conforming functions directly and need no cast.
    services = _aawm_alias_interfaces.AliasRouteServices(
        select_candidate_fn=cast(_aawm_alias_interfaces.SelectCandidateFn, select_candidate_fn),
        perform_candidate_request_fn=cast(
            _aawm_alias_interfaces.PerformCandidateRequestFn, perform_candidate_request_fn
        ),
        resolve_cooldown_publication_fn=_legacy_resolve_publication,
        publish_cooldown_memory_fn=_legacy_publish_memory,
        persist_cooldown_fn=_legacy_persist,
        set_session_affinity_fn=cast(_aawm_alias_interfaces.SetSessionAffinityFn, set_session_affinity_fn),
        add_alias_metadata_fn=add_alias_metadata_fn,
        raise_redispatch_fn=raise_redispatch_required_fn,
    )
    return await _aawm_alias_candidate_loop.handle_alias_route(
        services,
        alias_family=alias_family,
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=max_candidate_attempts,
        get_active_cooldown_state_fn=_legacy_get_active_cooldown_state,
        attempts_metadata_key=attempts_metadata_key,
        skipped_candidates_metadata_key=skipped_candidates_metadata_key,
        no_candidate_detail=no_candidate_detail,
        log_label=log_label,
    )


async def _handle_anthropic_auto_agent_alias_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    target_url: str,
    custom_headers: dict[str, Any],
) -> Response:
    alias_model = (
        _normalize_anthropic_auto_agent_alias_model(prepared_request_body.get("model"))
        or _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
    )

    async def _perform_candidate_request(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await _perform_anthropic_auto_agent_alias_candidate_request(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            candidate=candidate,
            candidate_body=candidate_body,
            target_url=target_url,
            custom_headers=custom_headers,
        )

    services = _aawm_alias_interfaces.AliasRouteServices(
        select_candidate_fn=_select_anthropic_auto_agent_candidate,
        perform_candidate_request_fn=_perform_candidate_request,
        resolve_cooldown_publication_fn=_resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=_publish_anthropic_cooldown_memory,
        persist_cooldown_fn=_persist_anthropic_cooldown_durable,
        set_session_affinity_fn=_set_anthropic_auto_agent_session_affinity,
        add_alias_metadata_fn=_add_anthropic_auto_agent_alias_metadata,
        raise_redispatch_fn=_raise_anthropic_auto_agent_redispatch_required,
    )
    return await _aawm_alias_candidate_loop.handle_alias_route(
        services,
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=len(_get_anthropic_auto_agent_candidates_for_alias(alias_model)),
        get_active_cooldown_state_fn=_get_anthropic_auto_agent_active_cooldown_state,
        attempts_metadata_key="anthropic_auto_agent_attempts",
        skipped_candidates_metadata_key="anthropic_auto_agent_skipped_candidates",
        no_candidate_detail="No Anthropic auto-agent alias candidates were available.",
        log_label="Anthropic",
    )


async def _perform_anthropic_native_passthrough_request(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    target_url: str,
    custom_headers: dict[str, Any],
    blocked_pass_through_prefixed_headers: Optional[list[str]] = None,
) -> Response:
    is_streaming_request = await is_streaming_request_fn(request)
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=target_url,
        custom_headers=custom_headers,
        _forward_headers=True,
        is_streaming_request=is_streaming_request,
        blocked_pass_through_prefixed_headers=blocked_pass_through_prefixed_headers,
    )
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )
    return received_value


_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX = "[1m]"
_ANTHROPIC_CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"
_ANTHROPIC_BETA_HEADER_NAME = "anthropic-beta"
_ANTHROPIC_BETA_XPASS_HEADER_NAME = f"x-pass-{_ANTHROPIC_BETA_HEADER_NAME}"
_ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME = "anthropic-dangerous-direct-browser-access"
_ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES = {
    "opus": "claude-opus-4-6",
    "opus-4-6": "claude-opus-4-6",
    "opus-4-8": "claude-opus-4-8",
    "fable-5": "claude-fable-5",
    "claude-fable-5": "claude-fable-5",
    "sonnet": "claude-sonnet-4-20250514",
    "sonnet-4-6": "claude-sonnet-4-6",
    "sonnet-4-20250514": "claude-sonnet-4-20250514",
    "sonnet-5": "claude-sonnet-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "haiku-4-5": "claude-haiku-4-5",
    "haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}


def _get_header_value_case_insensitive(
    headers: Any,
    header_name: str,
) -> Optional[str]:
    header_value = headers.get(header_name)
    if header_value is not None:
        return str(header_value)

    lowered_header_name = header_name.lower()
    for candidate_name, candidate_value in headers.items():
        if str(candidate_name).lower() == lowered_header_name:
            return str(candidate_value)
    return None


def _append_anthropic_beta_header_value(
    headers: dict[str, Any],
    beta_value: str,
) -> dict[str, Any]:
    existing_header_name = next(
        (header_name for header_name in headers if str(header_name).lower() == _ANTHROPIC_BETA_HEADER_NAME),
        None,
    )
    existing_beta = headers.pop(existing_header_name) if existing_header_name is not None else None
    if existing_beta is None:
        headers[_ANTHROPIC_BETA_HEADER_NAME] = beta_value
        return headers

    existing_values = [value.strip() for value in str(existing_beta).split(",") if value.strip()]
    if beta_value not in existing_values:
        existing_values.append(beta_value)
    headers[_ANTHROPIC_BETA_HEADER_NAME] = ", ".join(existing_values)
    return headers


def _prepare_anthropic_oauth_native_passthrough_headers(
    *,
    request: Request,
    custom_headers: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    auth_header = _get_header_value_case_insensitive(request.headers, "authorization")
    if not is_anthropic_oauth_key(auth_header):
        return custom_headers, False

    updated_headers = dict(custom_headers)
    request_beta = _get_header_value_case_insensitive(
        request.headers,
        _ANTHROPIC_BETA_HEADER_NAME,
    )
    if request_beta:
        for beta_value in str(request_beta).split(","):
            stripped_beta_value = beta_value.strip()
            if stripped_beta_value:
                _append_anthropic_beta_header_value(
                    updated_headers,
                    stripped_beta_value,
                )
    _append_anthropic_beta_header_value(
        updated_headers,
        ANTHROPIC_OAUTH_BETA_HEADER,
    )
    updated_headers[_ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME] = "true"
    return updated_headers, True


def _normalize_anthropic_native_passthrough_model_alias(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    model = request_body.get("model")
    if not isinstance(model, str):
        return request_body, False

    stripped_model = model.strip()
    if not stripped_model:
        return request_body, False

    suffix = ""
    alias_model = stripped_model
    if stripped_model.lower().endswith(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX):
        suffix = stripped_model[-len(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX) :]
        alias_model = stripped_model[: -len(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX)].strip()

    normalized_model = _ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES.get(alias_model.lower())
    if normalized_model is None:
        return request_body, False

    provider_model = f"{normalized_model}{suffix}"
    if provider_model == stripped_model:
        return request_body, False

    updated_body = dict(request_body)
    updated_body["model"] = provider_model
    metadata = updated_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.setdefault("inbound_model_alias", stripped_model)
    metadata.setdefault("requested_model_alias", stripped_model)
    metadata.setdefault("model_alias_label", stripped_model)
    metadata["anthropic_native_passthrough_model_alias"] = stripped_model
    metadata["anthropic_native_passthrough_normalized_model"] = provider_model
    updated_body["litellm_metadata"] = metadata
    return updated_body, True


def _prepare_anthropic_context_1m_native_passthrough(
    *,
    request: Request,
    request_body: dict[str, Any],
    custom_headers: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    model = request_body.get("model")
    if not isinstance(model, str):
        return request_body, custom_headers, False

    stripped_model = model.strip()
    if not stripped_model.lower().endswith(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX):
        return request_body, custom_headers, False

    base_model = stripped_model[: -len(_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX)].strip()
    if not base_model:
        return request_body, custom_headers, False

    updated_body = dict(request_body)
    updated_body["model"] = base_model

    metadata = updated_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.setdefault("inbound_model_alias", stripped_model)
    metadata.setdefault("requested_model_alias", stripped_model)
    metadata.setdefault("model_alias_label", stripped_model)
    metadata.setdefault("anthropic_native_passthrough_model_alias", stripped_model)
    metadata["anthropic_native_passthrough_normalized_model"] = base_model
    updated_body["litellm_metadata"] = metadata

    updated_headers = dict(custom_headers)
    for beta_header_name in (
        _ANTHROPIC_BETA_HEADER_NAME,
        _ANTHROPIC_BETA_XPASS_HEADER_NAME,
    ):
        request_beta = _get_header_value_case_insensitive(
            request.headers,
            beta_header_name,
        )
        if isinstance(request_beta, str) and request_beta.strip():
            _append_anthropic_beta_header_value(updated_headers, request_beta)
    _append_anthropic_beta_header_value(
        updated_headers,
        _ANTHROPIC_CONTEXT_1M_BETA_HEADER,
    )
    return updated_body, updated_headers, True


@router.api_route(
    "/anthropic",
    methods=["HEAD"],
    include_in_schema=False,
)
@router.api_route(
    "/anthropic/",
    methods=["HEAD"],
    include_in_schema=False,
)
async def anthropic_base_probe() -> Response:
    """Claude CLI reachability probe for a configured Anthropic base URL."""
    return Response(
        status_code=200,
        headers={"Cache-Control": "no-store"},
    )


@router.api_route(
    "/anthropic/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Anthropic Pass-through", "pass-through"],
)
async def anthropic_proxy_route(  # noqa: PLR0915
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    [Docs](https://docs.litellm.ai/docs/pass_through/anthropic_completion)
    """
    base_target_url = os.getenv("ANTHROPIC_API_BASE") or "https://api.anthropic.com"
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    anthropic_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="anthropic",
        region_name=None,
    )

    custom_headers = {}
    if "authorization" not in request.headers and "x-api-key" not in request.headers and anthropic_api_key is not None:
        custom_headers["x-api-key"] = "{}".format(anthropic_api_key)
    (
        custom_headers,
        normalized_oauth_native_passthrough_headers,
    ) = _prepare_anthropic_oauth_native_passthrough_headers(
        request=request,
        custom_headers=custom_headers,
    )

    blocked_pass_through_prefixed_headers: Optional[list[str]] = None
    if request.method == "POST":
        request_body = await get_request_body(request)
        (
            prepared_request_body,
            expanded_count,
            hooks,
            billing_header_fields,
        ) = await _prepare_anthropic_request_body_for_passthrough(request, request_body)
        if prepared_request_body is not request_body:
            _safe_set_request_parsed_body(request, prepared_request_body)
            verbose_proxy_logger.debug(
                "Prepared Anthropic passthrough request body; expanded_persisted_output=%s hooks=%s billing_header_keys=%s",
                expanded_count,
                sorted(hooks),
                sorted(billing_header_fields),
            )

        anthropic_auto_agent_alias = _resolve_anthropic_auto_agent_alias_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if anthropic_auto_agent_alias is not None:
            (
                prepared_request_body,
                _anthropic_read_guidance_changes,
            ) = _apply_aawm_read_agent_guidance_to_request_body(
                prepared_request_body,
                alias_model=anthropic_auto_agent_alias,
                target_field="system",
            )
            return await _handle_anthropic_auto_agent_alias_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                target_url=str(updated_url),
                custom_headers=custom_headers,
            )

        xai_oauth_adapter_model = _resolve_anthropic_xai_oauth_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if xai_oauth_adapter_model is not None:
            if _is_oa_xai_responses_model(xai_oauth_adapter_model):
                return await _handle_anthropic_xai_oauth_responses_adapter_route(
                    endpoint=endpoint,
                    request=request,
                    fastapi_response=fastapi_response,
                    user_api_key_dict=user_api_key_dict,
                    prepared_request_body=prepared_request_body,
                    adapter_model=xai_oauth_adapter_model,
                )
            return await _handle_anthropic_xai_oauth_completion_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=xai_oauth_adapter_model,
            )

        grok_native_oauth_adapter_model = _resolve_anthropic_grok_native_oauth_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if grok_native_oauth_adapter_model is not None:
            return await _handle_anthropic_grok_native_oauth_responses_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=grok_native_oauth_adapter_model,
            )

        adapter_model = _resolve_anthropic_openai_responses_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if adapter_model is not None:
            return await _handle_anthropic_openai_responses_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=adapter_model,
            )

        antigravity_adapter_model = _resolve_anthropic_antigravity_code_assist_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if antigravity_adapter_model is not None:
            return await _handle_anthropic_google_completion_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=antigravity_adapter_model,
                adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
            )

        opencode_zen_adapter_model = _resolve_anthropic_opencode_zen_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if opencode_zen_adapter_model is not None:
            return await _handle_anthropic_opencode_zen_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=opencode_zen_adapter_model,
            )

        kimi_code_adapter_model = _resolve_anthropic_kimi_chat_completions_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if kimi_code_adapter_model is not None:
            return await _handle_anthropic_kimi_chat_completions_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=kimi_code_adapter_model,
            )

        alibaba_token_plan_adapter_model = _resolve_anthropic_alibaba_token_plan_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if alibaba_token_plan_adapter_model is not None:
            return await _handle_anthropic_alibaba_token_plan_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=alibaba_token_plan_adapter_model,
            )

        google_adapter_model = _resolve_anthropic_google_completion_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if google_adapter_model is not None:
            return await _handle_anthropic_google_completion_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=google_adapter_model,
            )

        nvidia_adapter_model = _resolve_anthropic_nvidia_responses_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if nvidia_adapter_model is not None:
            return await _handle_anthropic_nvidia_completion_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=nvidia_adapter_model,
            )

        openrouter_completion_adapter_model = _resolve_anthropic_openrouter_completion_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if openrouter_completion_adapter_model is not None:
            return await _handle_anthropic_openrouter_completion_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=openrouter_completion_adapter_model,
            )

        openrouter_adapter_model = _resolve_anthropic_openrouter_responses_adapter_model(
            prepared_request_body,
            endpoint=encoded_endpoint,
        )
        if openrouter_adapter_model is not None:
            return await _handle_anthropic_openrouter_responses_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=prepared_request_body,
                adapter_model=openrouter_adapter_model,
            )

        (
            prepared_request_body,
            normalized_native_model_alias,
        ) = _normalize_anthropic_native_passthrough_model_alias(prepared_request_body)
        if normalized_native_model_alias or normalized_oauth_native_passthrough_headers:
            _safe_set_request_parsed_body(request, prepared_request_body)

        (
            prepared_request_body,
            custom_headers,
            normalized_context_1m_model,
        ) = _prepare_anthropic_context_1m_native_passthrough(
            request=request,
            request_body=prepared_request_body,
            custom_headers=custom_headers,
        )
        if normalized_context_1m_model:
            blocked_pass_through_prefixed_headers = [_ANTHROPIC_BETA_HEADER_NAME]
            _safe_set_request_parsed_body(request, prepared_request_body)

    return await _perform_anthropic_native_passthrough_request(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        target_url=str(updated_url),
        custom_headers=custom_headers,
        blocked_pass_through_prefixed_headers=blocked_pass_through_prefixed_headers,
    )


# Bedrock endpoint actions - consolidated list used for model extraction and streaming detection
BEDROCK_ENDPOINT_ACTIONS = {
    "invoke",
    "invoke-with-response-stream",
    "converse",
    "converse-stream",
    "count_tokens",
    "count-tokens",
}

BEDROCK_STREAMING_ACTIONS = {"invoke-with-response-stream", "converse-stream"}


def _extract_model_from_bedrock_endpoint(endpoint: str) -> str:
    """
    Extract model name from Bedrock endpoint path.

    Handles model names with slashes (e.g., aws/anthropic/bedrock-claude-3-5-sonnet-v1)
    by finding the action in the endpoint and extracting everything between "model" and the action.

    Args:
        endpoint: The endpoint path (e.g., "/model/aws/anthropic/model-name/invoke" or "v2/model/model-name/invoke")

    Returns:
        The extracted model name (e.g., "aws/anthropic/model-name" or "model-name")

    Raises:
        ValueError: If model cannot be extracted from endpoint
    """
    try:
        endpoint_parts = endpoint.split("/")

        if "application-inference-profile" in endpoint:
            # Format: model/application-inference-profile/{profile-id}/{action}
            return "/".join(endpoint_parts[1:3])

        # Format: model/{modelId}/{action} or v2/model/{modelId}/{action}
        # Find the index of "model" in the endpoint parts
        model_index = None
        for idx, part in enumerate(endpoint_parts):
            if part == "model":
                model_index = idx
                break

        # If "model" keyword not found, try to extract model from the endpoint
        # by finding the action and taking everything before it
        if model_index is None:
            # Find the index of the action in the endpoint parts
            action_index = None
            for idx, part in enumerate(endpoint_parts):
                if part in BEDROCK_ENDPOINT_ACTIONS:
                    action_index = idx
                    break

            if action_index is not None and action_index > 1:
                # Join all parts before the action (excluding empty strings)
                model_parts = [p for p in endpoint_parts[1:action_index] if p]
                if model_parts:
                    return "/".join(model_parts)

            raise ValueError(
                f"'model' keyword not found and unable to extract model from endpoint. Expected format: /model/{{modelId}}/{{action}}. Got: {endpoint}"
            )

        # Find the index of the action in the endpoint parts
        action_index = None
        for idx, part in enumerate(endpoint_parts):
            if part in BEDROCK_ENDPOINT_ACTIONS:
                action_index = idx
                break

        if action_index is not None and action_index > model_index + 1:
            # Join all parts between "model" and the action (excluding "model" itself)
            return "/".join(endpoint_parts[model_index + 1 : action_index])

        # Fallback to taking everything after "model" if no action found
        model_parts = [p for p in endpoint_parts[model_index + 1 :] if p]
        if model_parts:
            return "/".join(model_parts)

        raise ValueError(
            f"No model ID found after 'model' keyword. Expected format: /model/{{modelId}}/{{action}}. Got: {endpoint}"
        )

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        raise ValueError(
            f"Model missing from endpoint. Expected format: /model/{{modelId}}/{{action}}. Got: {endpoint}"
        ) from e


async def handle_bedrock_passthrough_router_model(
    model: str,
    endpoint: str,
    request: Request,
    request_body: dict,
    llm_router: litellm.Router,
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj,
    general_settings: dict,
    proxy_config,
    select_data_generator,
    user_model: Optional[str],
    user_temperature: Optional[float],
    user_request_timeout: Optional[float],
    user_max_tokens: Optional[int],
    user_api_base: Optional[str],
    version: Optional[str],
) -> Union[Response, StreamingResponse]:
    """
    Handle Bedrock passthrough for router models (models defined in config.yaml).

    Uses the same common processing path as non-router models to ensure
    metadata and hooks are properly initialized.

    Args:
        model: The router model name (e.g., "aws/anthropic/bedrock-claude-3-5-sonnet-v1")
        endpoint: The Bedrock endpoint path (e.g., "/model/{modelId}/invoke")
        request: The FastAPI request object
        request_body: The parsed request body
        llm_router: The LiteLLM router instance
        user_api_key_dict: The user API key authentication dictionary
        (additional args for common processing)

    Returns:
        Response or StreamingResponse depending on endpoint type
    """
    from fastapi import Response as FastAPIResponse

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    # Detect streaming based on endpoint
    is_streaming = any(action in endpoint for action in BEDROCK_STREAMING_ACTIONS)

    verbose_proxy_logger.debug(
        f"Bedrock router passthrough: model='{model}', endpoint='{endpoint}', streaming={is_streaming}"
    )

    # Use the common processing path (same as non-router models)
    # This ensures all metadata, hooks, and logging are properly initialized
    data: Dict[str, Any] = {}
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)

    data["model"] = model
    data["method"] = request.method
    data["endpoint"] = endpoint
    data["data"] = request_body
    data["custom_llm_provider"] = "bedrock"

    # Use the common passthrough processing to handle metadata and hooks
    # This also handles all response formatting (streaming/non-streaming) and exceptions
    try:
        result = await base_llm_response_processor.base_passthrough_process_llm_request(
            request=request,
            fastapi_response=FastAPIResponse(),
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        return result
    except Exception as e:
        # Use common exception handling
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


async def handle_bedrock_count_tokens(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle AWS Bedrock CountTokens API requests.

    This function processes count_tokens endpoints like:
    - /v1/messages/count_tokens
    - /v1/messages/count-tokens
    """
    from litellm.llms.bedrock.common_utils import BedrockError
    from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
    from litellm.proxy.proxy_server import llm_router

    try:
        # Initialize the handler
        handler = BedrockCountTokensHandler()

        # Extract model from request body
        model = request_body.get("model")
        if not model:
            raise HTTPException(status_code=400, detail={"error": "Model is required in request body"})

        # Get model parameters from router
        litellm_params = {"user_api_key_dict": user_api_key_dict}
        resolved_model = model  # Default fallback

        if llm_router:
            deployments = llm_router.get_model_list(model_name=model)
            if deployments and len(deployments) > 0:
                # Get the first matching deployment
                deployment = deployments[0]
                model_litellm_params = deployment.get("litellm_params", {})

                # Get the resolved model ID from the configuration
                if "model" in model_litellm_params:
                    resolved_model = model_litellm_params["model"]

                # Copy all litellm_params - BaseAWSLLM will handle AWS credential discovery
                for key, value in model_litellm_params.items():
                    if key != "user_api_key_dict":  # Don't overwrite user_api_key_dict
                        litellm_params[key] = value  # type: ignore

        verbose_proxy_logger.debug(f"Count tokens litellm_params: {litellm_params}")
        verbose_proxy_logger.debug(f"Resolved model: {resolved_model}")

        # Handle the count tokens request
        result = await handler.handle_count_tokens_request(
            request_data=request_body,
            litellm_params=litellm_params,
            resolved_model=resolved_model,
        )

        return result

    except BedrockError as e:
        # Convert BedrockError to HTTPException for FastAPI
        verbose_proxy_logger.error(f"BedrockError in handle_bedrock_count_tokens: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail={"error": e.message})
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error in handle_bedrock_count_tokens: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"CountTokens processing error: {str(e)}"})


async def bedrock_llm_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Handles Bedrock LLM API calls.

    Supports both direct Bedrock models and router models from config.yaml.

    Endpoints:
    - /model/{modelId}/invoke
    - /model/{modelId}/invoke-with-response-stream
    - /model/{modelId}/converse
    - /model/{modelId}/converse-stream
    - /model/application-inference-profile/{profileId}/{action}
    """
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    request_body = await _read_request_body(request=request)

    # Special handling for count_tokens endpoints
    if "count_tokens" in endpoint or "count-tokens" in endpoint:
        return await handle_bedrock_count_tokens(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            request_body=request_body,
        )

    # Extract model from endpoint path using helper
    try:
        model = _extract_model_from_bedrock_endpoint(endpoint=endpoint)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)},
        )

    # Check if this is a router model (from config.yaml)
    is_router_model = is_passthrough_request_using_router_model(request_body={"model": model}, llm_router=llm_router)

    # If router model, use dedicated router passthrough handler
    # This uses the same common processing path as non-router models
    if is_router_model and llm_router:
        return await handle_bedrock_passthrough_router_model(
            model=model,
            endpoint=endpoint,
            request=request,
            request_body=request_body,
            llm_router=llm_router,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

    # Fall back to existing implementation for direct Bedrock models
    verbose_proxy_logger.debug(f"Bedrock passthrough: Using direct Bedrock model '{model}' for endpoint '{endpoint}'")

    data: Dict[str, Any] = {}
    base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)

    data["method"] = request.method
    data["endpoint"] = endpoint
    data["data"] = request_body
    data["custom_llm_provider"] = "bedrock"

    try:
        result = await base_llm_response_processor.base_passthrough_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        return result
    except Exception as e:
        raise await base_llm_response_processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )


@router.api_route(
    "/bedrock/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Bedrock Pass-through", "pass-through"],
)
async def bedrock_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    This is the v1 passthrough for Bedrock.
    V2 is handled by the `/bedrock/v2` endpoint.
    [Docs](https://docs.litellm.ai/docs/pass_through/bedrock)
    """
    create_request_copy(request)

    try:
        from botocore.auth import SigV4Auth  # type: ignore[import-untyped]
        from botocore.awsrequest import AWSRequest  # type: ignore[import-untyped]
        from botocore.credentials import Credentials  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

    aws_region_name = litellm.utils.get_secret(secret_name="AWS_REGION_NAME")
    if _is_bedrock_agent_runtime_route(endpoint=endpoint):  # handle bedrock agents
        base_target_url = f"https://bedrock-agent-runtime.{aws_region_name}.amazonaws.com"
    else:
        return await bedrock_llm_proxy_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    from litellm.llms.bedrock.chat import BedrockConverseLLM

    bedrock_llm = BedrockConverseLLM()
    credentials: Credentials = bedrock_llm.get_credentials()  # type: ignore
    sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
    headers = {"Content-Type": "application/json"}
    # Assuming the body contains JSON data, parse it
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": e})
    _request = AWSRequest(method="POST", url=str(updated_url), data=json.dumps(data), headers=headers)
    sigv4.add_auth(_request)
    prepped = _request.prepare()

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(prepped.url),
        custom_headers=prepped.headers,  # type: ignore
        is_streaming_request=is_streaming_request,
        _forward_headers=True,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
        custom_body=data,  # type: ignore
    )

    return received_value


def _resolve_vertex_model_from_router(
    model_id: str,
    llm_router: Optional[litellm.Router],
    encoded_endpoint: str,
    endpoint: str,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Resolve Vertex AI model configuration from router.

    Args:
        model_id: The model ID extracted from the URL (e.g., "gcp/google/gemini-2.5-flash")
        llm_router: The LiteLLM router instance
        encoded_endpoint: The encoded endpoint path
        endpoint: The original endpoint path
        vertex_project: Current vertex project (may be from URL)
        vertex_location: Current vertex location (may be from URL)

    Returns:
        Tuple of (encoded_endpoint, endpoint, vertex_project, vertex_location)
        with resolved values from router config
    """
    if not llm_router:
        return encoded_endpoint, endpoint, vertex_project, vertex_location

    try:
        deployment = llm_router.get_available_deployment_for_pass_through(model=model_id)
        if not deployment:
            return encoded_endpoint, endpoint, vertex_project, vertex_location

        litellm_params = deployment.get("litellm_params", {})

        # Always override with router config values (they take precedence over URL values)
        config_vertex_project = litellm_params.get("vertex_project")
        config_vertex_location = litellm_params.get("vertex_location")
        if config_vertex_project:
            vertex_project = config_vertex_project
        if config_vertex_location:
            vertex_location = config_vertex_location

        # Get the actual Vertex AI model name by stripping the provider prefix
        # e.g., "vertex_ai/gemini-2.0-flash-exp" -> "gemini-2.0-flash-exp"
        model_from_config = litellm_params.get("model", "")
        if model_from_config:
            # get_llm_provider returns (model, custom_llm_provider, dynamic_api_key, api_base)
            # For "vertex_ai/gemini-2.0-flash-exp" it returns:
            # model="gemini-2.0-flash-exp", custom_llm_provider="vertex_ai"
            actual_model, custom_llm_provider, _, _ = get_llm_provider(model=model_from_config)

            # Log only non-sensitive information (model names and provider), never API keys or secrets.
            safe_actual_model = actual_model
            safe_custom_llm_provider = custom_llm_provider
            verbose_proxy_logger.debug(
                "get_llm_provider returned: actual_model=%s, custom_llm_provider=%s, model_id=%s",
                safe_actual_model,
                safe_custom_llm_provider,
                model_id,
            )

            if actual_model and model_id != actual_model:
                verbose_proxy_logger.debug(
                    "Resolved router model '%s' to '%s' (provider=%s) with project=%s, location=%s",
                    model_id,
                    actual_model,
                    custom_llm_provider,
                    vertex_project,
                    vertex_location,
                )
                encoded_endpoint = encoded_endpoint.replace(model_id, actual_model)
                endpoint = endpoint.replace(model_id, actual_model)

    except Exception as e:
        verbose_proxy_logger.debug(f"Error resolving vertex model from router for model {model_id}: {e}")

    return encoded_endpoint, endpoint, vertex_project, vertex_location


def _is_bedrock_agent_runtime_route(endpoint: str) -> bool:
    """
    Return True, if the endpoint should be routed to the `bedrock-agent-runtime` endpoint.
    """
    for _route in BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES:
        if _route in endpoint:
            return True
    return False


@router.api_route(
    "/assemblyai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["AssemblyAI Pass-through", "pass-through"],
)
@router.api_route(
    "/eu.assemblyai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["AssemblyAI EU Pass-through", "pass-through"],
)
async def assemblyai_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler import (
        AssemblyAIPassthroughLoggingHandler,
    )

    """
    [Docs](https://api.assemblyai.com)
    """
    # Set base URL based on the route
    assembly_region = AssemblyAIPassthroughLoggingHandler._get_assembly_region_from_url(url=str(request.url))
    base_target_url = AssemblyAIPassthroughLoggingHandler._get_assembly_base_url_from_region(region=assembly_region)
    encoded_endpoint = httpx.URL(endpoint).path
    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    assemblyai_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="assemblyai",
        region_name=assembly_region,
    )

    ## check for streaming
    is_streaming_request = False
    # assemblyai is streaming when 'stream' = True is in the body
    if request.method == "POST":
        _request_body = await request.json()
        if _request_body.get("stream"):
            is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": "{}".format(assemblyai_api_key)},
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
    )

    return received_value


@router.api_route(
    "/azure_ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Azure AI Pass-through", "pass-through"],
)
@router.api_route(
    "/azure/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Azure Pass-through", "pass-through"],
)
async def azure_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Call any azure endpoint using the proxy.

    Just use `{PROXY_BASE_URL}/azure/{endpoint:path}`

    Checks if the deployment id in the url is a litellm model name. If so, it will route using the llm_router.allm_passthrough_route.
    """
    from litellm.proxy.proxy_server import llm_router

    parts = endpoint.split(
        "/"
    )  # azure model is in the url - e.g. https://{endpoint}/openai/deployments/{deployment-id}/completions?api-version=2024-10-21

    if len(parts) > 1 and llm_router:
        for part in parts:
            # check if LLM MODEL
            is_router_model = is_passthrough_request_using_router_model(
                request_body={"model": part}, llm_router=llm_router
            )
            # check if vector store index
            is_vector_store_index = (
                (litellm.vector_store_index_registry.is_vector_store_index(vector_store_index_name=part))
                if litellm.vector_store_index_registry is not None
                else False
            )

            if is_router_model:
                request_body = await get_request_body(request)
                is_streaming_request = is_passthrough_request_streaming(request_body)
                result = await llm_router.allm_passthrough_route(
                    model=part,
                    method=request.method,
                    endpoint=endpoint,
                    request_query_params=request.query_params,
                    request_headers=_safe_get_request_headers(request),
                    stream=request_body.get("stream", False),
                    content=None,
                    data=None,
                    files=None,
                    json=(request_body if request.headers.get("content-type") == "application/json" else None),
                    params=None,
                    headers=None,
                    cookies=None,
                )

                if is_streaming_request:
                    # Check if result is an async generator (from _async_streaming)
                    import inspect

                    if inspect.isasyncgen(result):
                        # Result is already an async generator, use it directly
                        return StreamingResponse(
                            content=result,
                            status_code=200,
                            headers={"content-type": "text/event-stream"},
                        )
                    else:
                        # Result is an httpx.Response, use aiter_bytes()
                        result = cast(httpx.Response, result)
                        return StreamingResponse(
                            content=result.aiter_bytes(),
                            status_code=result.status_code,
                            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                                headers=result.headers,
                                custom_headers=None,
                            ),
                        )

                # Non-streaming response
                result = cast(httpx.Response, result)
                content = await result.aread()
                return Response(
                    content=content,
                    status_code=result.status_code,
                    headers=HttpPassThroughEndpointHelpers.get_response_headers(
                        headers=result.headers,
                        custom_headers=None,
                    ),
                )
            elif is_vector_store_index:
                # get the api key from the provider config
                provider_config = ProviderConfigManager.get_provider_vector_stores_config(
                    provider=litellm.LlmProviders.AZURE_AI
                )
                if provider_config is None:
                    raise Exception("Provider config not found for Azure AI")
                # get the index from registry
                if litellm.vector_store_registry is None:
                    raise Exception("Vector store registry not found")

                is_allowed_to_call_vector_store_endpoint(
                    index_name=part,
                    provider=litellm.LlmProviders.AZURE_AI,
                    request=request,
                    user_api_key_dict=user_api_key_dict,
                )
                # get the vector store name from index registry
                index_object = (
                    (litellm.vector_store_index_registry.get_vector_store_index_by_name(vector_store_index_name=part))
                    if litellm.vector_store_index_registry is not None
                    else None
                )
                if index_object is None:
                    raise Exception(f"Vector store index not found for {part}")

                vector_store_name = index_object.litellm_params.vector_store_name

                vector_store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry_by_name(
                    vector_store_name=vector_store_name
                )
                if vector_store is None:
                    raise Exception(f"Vector store not found for {vector_store_name}")
                litellm_params = vector_store.get("litellm_params") or {}
                auth_credentials = provider_config.get_auth_credentials(litellm_params=litellm_params)

                extra_headers = auth_credentials.get("headers") or {}

                base_target_url = litellm_params.get("api_base")
                if base_target_url is None:
                    raise Exception(f"API base not found for {part}")
                return await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                    endpoint=endpoint,
                    request=request,
                    fastapi_response=fastapi_response,
                    user_api_key_dict=user_api_key_dict,
                    base_target_url=base_target_url,
                    api_key=None,
                    custom_llm_provider=litellm.LlmProviders.AZURE_AI,
                    extra_headers=cast(dict, extra_headers),
                )

    base_target_url = get_secret_str(secret_name="AZURE_API_BASE")
    if base_target_url is None:
        raise Exception("Required 'AZURE_API_BASE' in environment to make pass-through calls to Azure.")
    # Add or update query parameters
    azure_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider=litellm.LlmProviders.AZURE.value,
        region_name=None,
    )
    if azure_api_key is None:
        raise Exception("Required 'AZURE_API_KEY' in environment to make pass-through calls to Azure.")

    return await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        base_target_url=base_target_url,
        api_key=azure_api_key,
        custom_llm_provider=litellm.LlmProviders.AZURE,
    )


from abc import ABC, abstractmethod


class BaseVertexAIPassThroughHandler(ABC):
    @staticmethod
    @abstractmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        pass

    @staticmethod
    @abstractmethod
    def update_base_target_url_with_credential_location(base_target_url: str, vertex_location: Optional[str]) -> str:
        pass


class VertexAIDiscoveryPassThroughHandler(BaseVertexAIPassThroughHandler):
    @staticmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        return "https://discoveryengine.googleapis.com/"

    @staticmethod
    def update_base_target_url_with_credential_location(base_target_url: str, vertex_location: Optional[str]) -> str:
        return base_target_url


class VertexAIPassThroughHandler(BaseVertexAIPassThroughHandler):
    @staticmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        return get_vertex_base_url(vertex_location)

    @staticmethod
    def update_base_target_url_with_credential_location(base_target_url: str, vertex_location: Optional[str]) -> str:
        return get_vertex_base_url(vertex_location)


def get_vertex_base_url(vertex_location: Optional[str]) -> str:
    """
    Returns the base URL for Vertex AI based on the provided location.
    """
    if vertex_location == "global":
        return "https://aiplatform.googleapis.com/"
    return f"https://{vertex_location}-aiplatform.googleapis.com/"


def get_vertex_ai_allowed_incoming_headers(request: Request) -> dict:
    """
    Extract only the allowed headers from incoming request for Vertex AI pass-through.

    Uses an allowlist approach for security - only forwards headers we explicitly trust.
    This prevents accidentally forwarding sensitive headers like the LiteLLM auth token.

    Args:
        request: The FastAPI request object

    Returns:
        dict: Headers dictionary with only allowed headers
    """
    incoming_headers = _safe_get_request_headers(request)
    headers = {}
    for header_name in ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS:
        if header_name in incoming_headers:
            headers[header_name] = incoming_headers[header_name]
    return headers


def get_vertex_pass_through_handler(
    call_type: Literal["discovery", "aiplatform"],
) -> BaseVertexAIPassThroughHandler:
    if call_type == "discovery":
        return VertexAIDiscoveryPassThroughHandler()
    elif call_type == "aiplatform":
        return VertexAIPassThroughHandler()
    else:
        raise ValueError(f"Invalid call type: {call_type}")


def _override_vertex_params_from_router_credentials(
    router_credentials: Optional[Any],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Override vertex_project and vertex_location with values from router_credentials if available.

    Args:
        router_credentials: Optional vector store credentials from registry (LiteLLM_ManagedVectorStore)
        vertex_project: Current vertex project ID (from URL)
        vertex_location: Current vertex location (from URL)

    Returns:
        Tuple of (vertex_project, vertex_location) with overridden values if applicable
    """
    if router_credentials is None:
        return vertex_project, vertex_location

    verbose_proxy_logger.debug("Using vector store credentials to override vertex project and location")

    litellm_params = router_credentials.get("litellm_params", {})
    if not litellm_params:
        verbose_proxy_logger.warning("Vector store credentials found but litellm_params is empty")
        return vertex_project, vertex_location

    # Extract vertex_project and vertex_location from litellm_params
    vector_store_project = litellm_params.get("vertex_project")
    vector_store_location = litellm_params.get("vertex_location")

    if vector_store_project:
        verbose_proxy_logger.debug(
            "Overriding vertex_project from URL (%s) with vector store value: %s",
            vertex_project,
            vector_store_project,
        )
        vertex_project = vector_store_project
    else:
        verbose_proxy_logger.warning("Vector store credentials found but missing vertex_project in litellm_params")

    if vector_store_location:
        verbose_proxy_logger.debug(
            "Overriding vertex_location from URL (%s) with vector store value: %s",
            vertex_location,
            vector_store_location,
        )
        vertex_location = vector_store_location
    else:
        verbose_proxy_logger.warning("Vector store credentials found but missing vertex_location in litellm_params")

    return vertex_project, vertex_location


async def _prepare_vertex_auth_headers(
    request: Request,
    vertex_credentials: Optional[Any],
    router_credentials: Optional[Any],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    base_target_url: Optional[str],
    get_vertex_pass_through_handler: BaseVertexAIPassThroughHandler,
) -> Tuple[dict, Optional[str], bool, Optional[str], Optional[str]]:
    """
    Prepare authentication headers for Vertex AI pass-through requests.

    Args:
        request: FastAPI request object
        vertex_credentials: Vertex AI credentials from config
        router_credentials: Optional vector store credentials from registry
        vertex_project: Vertex project ID
        vertex_location: Vertex location
        base_target_url: Base URL for the Vertex AI service
        get_vertex_pass_through_handler: Handler for the specific Vertex AI service

    Returns:
        Tuple containing:
            - headers: dict - Authentication headers to use
            - base_target_url: Optional[str] - Updated base target URL
            - headers_passed_through: bool - Whether headers were passed through from request
            - vertex_project: Optional[str] - Updated vertex project ID
            - vertex_location: Optional[str] - Updated vertex location
    """
    vertex_llm_base = VertexBase()
    headers_passed_through = False

    # Use headers from the incoming request if no vertex credentials are found
    if (vertex_credentials is None or vertex_credentials.vertex_project is None) and router_credentials is None:
        headers = _safe_get_request_headers(request).copy()
        headers_passed_through = True
        verbose_proxy_logger.debug("default_vertex_config  not set, incoming request headers %s", headers)
        headers.pop("content-length", None)
        headers.pop("host", None)
    else:
        if router_credentials is not None:
            vertex_credentials_str = None
        elif vertex_credentials is not None:
            # Use credentials from vertex_credentials
            # When vertex_credentials are provided (including default credentials),
            # use their project/location values if available
            if vertex_credentials.vertex_project is not None:
                vertex_project = vertex_credentials.vertex_project
            if vertex_credentials.vertex_location is not None:
                vertex_location = vertex_credentials.vertex_location
            vertex_credentials_str = vertex_credentials.vertex_credentials
        else:
            raise ValueError("No vertex credentials found")

        _auth_header, vertex_project = await vertex_llm_base._ensure_access_token_async(
            credentials=vertex_credentials_str,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = vertex_llm_base._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials_str,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base="",
        )

        # Use allowlist approach - only forward specific safe headers
        headers = get_vertex_ai_allowed_incoming_headers(request)
        # Add the Authorization header with vendor credentials
        headers["Authorization"] = f"Bearer {auth_header}"

        if base_target_url is not None:
            base_target_url = get_vertex_pass_through_handler.update_base_target_url_with_credential_location(
                base_target_url, vertex_location
            )

    return (
        headers,
        base_target_url,
        headers_passed_through,
        vertex_project,
        vertex_location,
    )


async def _base_vertex_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    get_vertex_pass_through_handler: BaseVertexAIPassThroughHandler,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
    router_credentials: Optional[Any] = None,
):
    """
    Base function for Vertex AI passthrough routes.
    Handles common logic for all Vertex AI services.

    Default base_target_url is `https://{vertex_location}-aiplatform.googleapis.com/`

    Args:
        endpoint: The endpoint path
        request: FastAPI request object
        fastapi_response: FastAPI response object
        get_vertex_pass_through_handler: Handler for the specific Vertex AI service
        user_api_key_dict: User API key authentication dict
        router_credentials: Optional vector store credentials from registry (LiteLLM_ManagedVectorStore)
    """
    from litellm.llms.vertex_ai.common_utils import (
        construct_target_url,
        get_vertex_location_from_url,
        get_vertex_model_id_from_url,
        get_vertex_project_id_from_url,
    )
    from litellm.proxy.proxy_server import llm_router

    encoded_endpoint = httpx.URL(endpoint).path
    verbose_proxy_logger.debug("requested endpoint %s", endpoint)
    headers: dict = {}
    # RR-054 #49: single auth call; user_api_key_auth raises rather than returning None
    # for invalid keys, so the previous re-check was dead/duplicate work.
    api_key_to_use = get_litellm_virtual_key(request=request)
    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=api_key_to_use,
    )

    vertex_project: Optional[str] = get_vertex_project_id_from_url(endpoint)
    vertex_location: Optional[str] = get_vertex_location_from_url(endpoint)

    # Override with vector store credentials if available
    vertex_project, vertex_location = _override_vertex_params_from_router_credentials(
        router_credentials=router_credentials,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

    # Check if model is in router config - always do this to resolve custom model names
    model_id = get_vertex_model_id_from_url(endpoint)
    if model_id:
        if llm_router:
            # Resolve model configuration from router
            (
                encoded_endpoint,
                endpoint,
                vertex_project,
                vertex_location,
            ) = _resolve_vertex_model_from_router(
                model_id=model_id,
                llm_router=llm_router,
                encoded_endpoint=encoded_endpoint,
                endpoint=endpoint,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
            )

    vertex_credentials = passthrough_endpoint_router.get_vertex_credentials(
        project_id=vertex_project,
        location=vertex_location,
    )

    base_target_url = get_vertex_pass_through_handler.get_default_base_target_url(vertex_location)

    # Prepare authentication headers
    (
        headers,
        base_target_url,
        headers_passed_through,
        vertex_project,
        vertex_location,
    ) = await _prepare_vertex_auth_headers(  # type: ignore
        request=request,
        vertex_credentials=vertex_credentials,
        router_credentials=router_credentials,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        base_target_url=base_target_url,
        get_vertex_pass_through_handler=get_vertex_pass_through_handler,
    )

    if base_target_url is None:
        base_target_url = get_vertex_base_url(vertex_location)

    request_route = encoded_endpoint
    verbose_proxy_logger.debug("request_route %s", request_route)

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    updated_url = construct_target_url(
        base_url=base_target_url,
        requested_route=encoded_endpoint,
        vertex_location=vertex_location,
        vertex_project=vertex_project,
    )

    verbose_proxy_logger.debug("updated url %s", updated_url)

    ## check for streaming
    target = str(updated_url)
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True
        target += "?alt=sse"

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=target,
        custom_headers=headers,
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path

    try:
        received_value = await endpoint_func(
            request,
            fastapi_response,
            user_api_key_dict,
        )
    except ProxyException as e:
        if headers_passed_through:
            e.message = f"No credentials found on proxy for project_name={vertex_project} + location={vertex_location}, check `/model/info` for allowed project + region combinations with `use_in_pass_through: true`. Headers were passed through directly but request failed with error: {e.message}"
        raise e

    return received_value


@router.api_route(
    "/vertex_ai/discovery/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
)
async def vertex_discovery_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Call any vertex discovery endpoint using the proxy.

    Just use `{PROXY_BASE_URL}/vertex_ai/discovery/{endpoint:path}`

    Target url: `https://discoveryengine.googleapis.com`
    """
    import re

    from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

    # Extract vector store ID from endpoint if present (e.g., dataStores/test-litellm-app_1761094730750)
    vector_store_credentials: Optional[LiteLLM_ManagedVectorStore] = None
    vector_store_id_match = re.search(r"dataStores/([^/]+)", endpoint)

    if vector_store_id_match:
        vector_store_id = vector_store_id_match.group(1)
        verbose_proxy_logger.debug("Extracted vector store ID from endpoint: %s", vector_store_id)

        # Retrieve vector store credentials from the registry
        vector_store_credentials = passthrough_endpoint_router.get_vector_store_credentials(
            vector_store_id=vector_store_id
        )

        if vector_store_credentials:
            verbose_proxy_logger.debug("Found vector store credentials for ID: %s", vector_store_id)
        else:
            verbose_proxy_logger.warning(
                "Vector store ID %s found in endpoint but no credentials found in registry",
                vector_store_id,
            )

    discovery_handler = get_vertex_pass_through_handler(call_type="discovery")
    return await _base_vertex_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        get_vertex_pass_through_handler=discovery_handler,
        router_credentials=vector_store_credentials,
    )


@router.api_route(
    "/vertex-ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
    include_in_schema=False,
)
@router.api_route(
    "/vertex_ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
)
async def vertex_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Call LiteLLM proxy via Vertex AI SDK.

    [Docs](https://docs.litellm.ai/docs/pass_through/vertex_ai)
    """
    ai_platform_handler = get_vertex_pass_through_handler(call_type="aiplatform")

    return await _base_vertex_proxy_route(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        get_vertex_pass_through_handler=ai_platform_handler,
        user_api_key_dict=user_api_key_dict,
    )


@router.api_route(
    "/openai_passthrough/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["OpenAI Pass-through", "pass-through"],
)
@router.api_route(
    "/openai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["OpenAI Pass-through", "pass-through"],
)
async def openai_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Pass-through endpoint for OpenAI API calls.

    Available on both routes:
    - /openai/{endpoint:path} - Standard OpenAI passthrough route
    - /openai_passthrough/{endpoint:path} - Dedicated passthrough route (recommended for Responses API)

    Use /openai_passthrough/* when you need guaranteed passthrough to OpenAI without conflicts
    with LiteLLM's native implementations (e.g., for the Responses API at /v1/responses).

    Examples:
        Standard route:
        - /openai/v1/chat/completions
        - /openai/v1/assistants
        - /openai/v1/threads

        Dedicated passthrough (for Responses API):
        - /openai_passthrough/v1/responses
        - /openai_passthrough/v1/responses/{response_id}
        - /openai_passthrough/v1/responses/{response_id}/input_items

    [Docs](https://docs.litellm.ai/docs/pass_through/openai_passthrough)
    """
    request_body: dict[str, Any] = {}
    is_oa_xai_request = False
    is_grok_native_oauth_request = False
    if request.method == "POST":
        request_body = await get_request_body(request)
        is_oa_xai_request = _is_oa_xai_request_body(request_body)
        is_grok_native_oauth_request = _is_openai_responses_endpoint(endpoint) and _is_grok_native_oauth_request_body(
            request_body
        )

    base_target_url = _get_openai_passthrough_target_base(
        request=request,
        endpoint=endpoint,
    )
    preserve_client_auth = _should_preserve_openai_client_auth(
        request=request,
        endpoint=endpoint,
    )
    openai_api_key: Optional[str] = None
    forward_headers = False
    if is_oa_xai_request:
        base_target_url = os.getenv("LITELLM_XAI_OAUTH_API_BASE") or XAI_API_BASE
    elif is_grok_native_oauth_request:
        base_target_url = _get_grok_passthrough_target_base()
    elif preserve_client_auth:
        forward_headers = True
    else:
        openai_api_key = passthrough_endpoint_router.get_credentials(
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            region_name=None,
        )
        if openai_api_key is None:
            raise Exception("Required 'OPENAI_API_KEY' in environment to make pass-through calls to OpenAI.")

    return await BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
        endpoint=endpoint,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        base_target_url=base_target_url,
        api_key=openai_api_key,
        custom_llm_provider=litellm.LlmProviders.OPENAI,
        forward_headers=forward_headers,
    )


async def _perform_codex_auto_agent_native_openai_request(
    *,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
    request_body: dict[str, Any],
) -> Response:
    is_streaming_request = "stream" in str(target_url)
    try:
        return await pass_through_request(
            request=request,
            target=target_url,
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=api_key,
                request=request,
            ),
            user_api_key_dict=user_api_key_dict,
            forward_headers=forward_headers,
            stream=is_streaming_request,
            custom_body=request_body,
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            egress_credential_family="openai" if forward_headers else None,
            expected_target_family="openai",
            # RR-054 #24
            retryable_upstream_status_codes=list(_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT),
            caller_managed_hidden_retry=False,
        )
    except Exception as exc:
        if _codex_native_openai_candidate_unavailable_detail(exc) is not None:
            _raise_codex_native_openai_auto_agent_candidate_unavailable(exc)
        raise


async def _perform_codex_auto_agent_grok_native_responses_request(
    *,
    endpoint: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, Any],
) -> Response:
    (
        adapted_request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    try:
        grok_context = await BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
            endpoint=endpoint,
            request=request,
            request_body=adapted_request_body,
            extra_headers={},
        )
    except Exception as exc:
        if _grok_native_candidate_unavailable_detail(exc) is not None:
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise
    if grok_context is None:
        _raise_grok_native_auto_agent_candidate_unavailable(
            Exception("Grok native Codex auto-agent candidate requires a managed " "Grok OIDC credential.")
        )
    assert grok_context is not None
    _, grok_headers, grok_prepared_body, updated_url = grok_context
    try:
        response = await pass_through_request(
            request=request,
            target=updated_url,
            custom_headers=grok_headers,
            user_api_key_dict=user_api_key_dict,
            forward_headers=False,
            stream=bool(grok_prepared_body.get("stream")),
            custom_body=grok_prepared_body,
            custom_llm_provider=litellm.LlmProviders.XAI.value,
            egress_credential_family="xai",
            expected_target_family="xai",
            retryable_upstream_status_codes=[
                429,
                *_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES,
            ],
            caller_managed_hidden_retry=True,
        )
    except Exception as exc:
        if _grok_native_candidate_unavailable_detail(exc) is not None:
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise
    return await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=str(grok_prepared_body.get("model") or request_body.get("model") or "unknown-model"),
        adapter="codex_auto_agent_grok_native_responses",
        adapter_label="Grok native",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            request_body,
            adapter="codex_auto_agent_grok_native_responses",
            upstream_url=str(updated_url),
            provider="grok",
        ),
        request_body=request_body,
    )


async def _perform_codex_auto_agent_oa_xai_responses_request(
    *,
    endpoint: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, Any],
) -> Response:
    (
        adapted_request_body,
        _adapted_custom_tools,
    ) = _adapt_codex_custom_tools_to_functions_from_request_body(request_body)
    try:
        oa_xai_context = await BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
            endpoint=endpoint,
            request_body=adapted_request_body,
        )
    except Exception as exc:
        if _xai_oauth_candidate_unavailable_detail(exc) is not None:
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    if oa_xai_context is None:
        _raise_xai_oauth_auto_agent_candidate_unavailable(
            Exception("Codex auto-agent xAI OAuth candidate requires a managed xAI " "OAuth credential.")
        )
    assert oa_xai_context is not None
    _, oa_xai_api_key, oa_xai_prepared_body, updated_url = oa_xai_context
    try:
        response = await pass_through_request(
            request=request,
            target=updated_url,
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=oa_xai_api_key,
                request=request,
            ),
            user_api_key_dict=user_api_key_dict,
            forward_headers=False,
            stream=bool(oa_xai_prepared_body.get("stream")),
            custom_body=oa_xai_prepared_body,
            custom_llm_provider=litellm.LlmProviders.XAI.value,
            egress_credential_family="xai",
            expected_target_family="xai",
            retryable_upstream_status_codes=[
                429,
                *_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES,
            ],
            caller_managed_hidden_retry=True,
        )
    except Exception as exc:
        if _xai_oauth_candidate_unavailable_detail(exc) is not None:
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    return await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=str(oa_xai_prepared_body.get("model") or request_body.get("model") or "unknown-model"),
        adapter="codex_auto_agent_xai_oauth_responses",
        adapter_label="xAI OAuth",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            request_body,
            adapter="codex_auto_agent_xai_oauth_responses",
            upstream_url=str(updated_url),
            provider="xai",
        ),
        request_body=request_body,
    )


async def _validate_codex_auto_agent_openrouter_responses_stream(
    response: StreamingResponse,
    *,
    adapter_model: str,
    intake_context: Optional[dict[str, Any]] = None,
) -> StreamingResponse:
    event_summaries: list[dict[str, Any]] = []
    peek = await _aawm_alias_streaming.peek_streaming_response(
        response,
        max_chunks=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,
        max_bytes=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,
    )
    if not peek.exhausted:
        return peek.response
    try:
        response_body = await _collect_responses_response_from_stream(
            peek.response,
            event_summaries=event_summaries,
        )
    except HTTPException as exc:
        if (
            exc.status_code == 502
            and str(exc.detail) == "OpenAI Responses stream completed without a response payload."
        ):
            _raise_codex_auto_agent_empty_success_response(
                response_body={
                    "model": adapter_model,
                    "status": "completed",
                    "output": [],
                },
                adapter_model=adapter_model,
                stream_event_summaries=event_summaries,
            )
        raise
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            stream_event_summaries=event_summaries,
        )
    if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
        _raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_responses",
            adapter_label="OpenRouter",
            intake_context=intake_context,
            stream_event_summaries=event_summaries,
        )
    if _is_failed_responses_body(response_body):
        _raise_codex_auto_agent_failed_responses_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_responses",
            adapter_label="OpenRouter",
            stream_event_summaries=event_summaries,
        )

    async def _replay_iterator() -> Any:
        for raw_chunk in peek.buffered_chunks:
            yield raw_chunk

    return StreamingResponse(
        _replay_iterator(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


async def _perform_codex_auto_agent_openrouter_responses_request(
    *,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    endpoint: str,
    adapter_model: str,
    request_body: dict[str, Any],
    use_alias_candidate_probe: bool = False,
) -> Response:
    openrouter_api_key = _get_openrouter_api_key()
    if openrouter_api_key is None:
        exc = ProxyException(
            message=(
                "OpenRouter Codex auto-agent candidate requires " "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
            ),
            type="rate_limit_error",
            param="model",
            code=429,
        )
        setattr(
            exc,
            "detail",
            {
                "error": {
                    "message": exc.message,
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                }
            },
        )
        raise exc

    target_base_url = _get_openrouter_target_base()
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint=endpoint,
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.OPENROUTER.value,
    )
    custom_headers: dict[str, Any] = BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=openrouter_api_key,
        request=request,
    )
    custom_headers.update(_build_openrouter_default_headers())
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    response = await _perform_openrouter_adapter_pass_through_request(
        adapter_model=adapter_model,
        log_warnings=not use_alias_candidate_probe,
        use_alias_candidate_probe=use_alias_candidate_probe,
        request=request,
        target=str(target_url),
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=request_body,
        forward_headers=False,
        allowed_forward_headers=[],
        allowed_pass_through_prefixed_headers=[],
        stream=bool(request_body.get("stream")),
        custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
        egress_credential_family="openrouter",
        expected_target_family="openrouter",
    )
    if isinstance(response, StreamingResponse):
        return await _validate_codex_auto_agent_openrouter_responses_stream(
            response,
            adapter_model=adapter_model,
            intake_context=_build_malformed_tool_call_intake_context(
                request,
                request_body,
                adapter="codex_auto_agent_openrouter_responses",
                upstream_url=str(target_url),
                provider="openrouter",
            ),
        )
    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
        try:
            response_body = json.loads(_decode_http_response_body(response.body))
        except Exception:
            return response
        if isinstance(response_body, dict) and _is_codex_auto_agent_empty_success_responses_body(response_body):
            _raise_codex_auto_agent_empty_success_response(
                response_body=response_body,
                adapter_model=adapter_model,
            )
        if isinstance(response_body, dict) and _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
            _raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter="codex_auto_agent_openrouter_responses",
                adapter_label="OpenRouter",
                intake_context=_build_malformed_tool_call_intake_context(
                    request,
                    request_body,
                    adapter="codex_auto_agent_openrouter_responses",
                    upstream_url=str(target_url),
                    provider="openrouter",
                ),
            )
        if isinstance(response_body, dict) and _is_failed_responses_body(response_body):
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter="codex_auto_agent_openrouter_responses",
                adapter_label="OpenRouter",
            )
    return response


async def _prepare_codex_kimi_chat_completions_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    prepared_request_body = _kimi_code_adapters.normalize_kimi_code_custom_tool_outputs(prepared_request_body)
    adapted_request_body, _adapted_custom_tools = _adapt_codex_custom_tools_to_functions_from_request_body(
        prepared_request_body
    )
    adapted_request_body, _adapted_namespace_tools = _adapt_codex_namespace_tools_to_functions_from_request_body(
        adapted_request_body
    )
    (
        adapted_request_body,
        _codex_tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(adapted_request_body)
    adapted_request_body, _unsupported_hosted_tools = _drop_unsupported_codex_hosted_tools_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _unsupported_input_items = _drop_unsupported_codex_input_items_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _removed_tool_choice = _drop_tool_choice_without_tools_from_request_body(adapted_request_body)
    return await _kimi_code_adapters.prepare_codex_kimi_chat_completions_adapter_route(
        request=request,
        prepared_request_body=adapted_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_codex_kimi_chat_completions_adapter_call(
    *,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    target_url: Union[str, httpx.URL],
    api_key: str,
    api_base: str,
    client_requested_stream: bool,
    completion_kwargs: Payload,
    request_input: Any,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    litellm_metadata: Payload,
    upstream_model: str,
) -> Response:
    """Execute Kimi chat completions and reuse the standard Responses wrapper."""
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = config
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(str(target_url)))
    completion_response = await litellm.acompletion(
        **completion_kwargs,
        api_key=api_key,
        api_base=api_base,
        litellm_metadata=litellm_metadata,
        proxy_server_request={
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
        shared_session=_get_proxy_shared_aiohttp_session(),
    )
    if client_requested_stream:
        return StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=upstream_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.KIMI_CODE.value,
                    litellm_metadata=litellm_metadata,
                )
            ),
            media_type="text/event-stream",
        )
    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _handle_codex_kimi_chat_completions_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    rollup_kwargs: dict[str, Any] = {}

    async def _prepare_and_emit_route_log(
        **kwargs: Any,
    ) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
        plan = await _prepare_codex_kimi_chat_completions_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(_build_adapted_route_rollup_kwargs(metadata if isinstance(metadata, dict) else {}))
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Kimi Code",
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_kimi_chat_completions_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_kimi_chat_completions_adapter",
        adapter_label="Kimi Code",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            prepared_request_body,
            adapter="codex_kimi_chat_completions_adapter",
            provider="kimi_code",
        ),
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="Kimi Code",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Kimi Code",
    )
    return validated_response


async def _prepare_codex_alibaba_token_plan_adapter_route(
    *,
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
    prepared_request_body = _alibaba_token_plan_adapters.normalize_alibaba_token_plan_custom_tool_outputs(
        prepared_request_body
    )
    adapted_request_body, _adapted_custom_tools = _adapt_codex_custom_tools_to_functions_from_request_body(
        prepared_request_body
    )
    adapted_request_body, _adapted_namespace_tools = _adapt_codex_namespace_tools_to_functions_from_request_body(
        adapted_request_body
    )
    (
        adapted_request_body,
        _codex_tool_description_patch_events,
    ) = _apply_codex_tool_description_patches_to_request_body(adapted_request_body)
    adapted_request_body, _unsupported_hosted_tools = _drop_unsupported_codex_hosted_tools_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _unsupported_input_items = _drop_unsupported_codex_input_items_from_request_body(
        adapted_request_body
    )
    adapted_request_body, _removed_tool_choice = _drop_tool_choice_without_tools_from_request_body(adapted_request_body)
    return await _alibaba_token_plan_adapters.prepare_codex_alibaba_token_plan_adapter_route(
        request=request,
        prepared_request_body=adapted_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )


async def _perform_codex_alibaba_token_plan_adapter_call(
    *,
    config: "_aawm_adapter_config.AnthropicCompletionAdapterConfig",
    request: Request,
    prepared_request_body: Payload,
    adapter_model: str,
    target_url: Union[str, httpx.URL],
    api_key: str,
    api_base: str,
    client_requested_stream: bool,
    completion_kwargs: Payload,
    request_input: Any,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    litellm_metadata: Payload,
    upstream_model: str,
) -> Response:
    """Execute Token Plan chat completions through the standard Responses wrapper."""
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = config, adapter_model
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(str(target_url)))
    completion_response = await litellm.acompletion(
        **completion_kwargs,
        api_key=api_key,
        api_base=api_base,
        litellm_metadata=litellm_metadata,
        proxy_server_request={
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
        shared_session=_get_proxy_shared_aiohttp_session(),
    )
    if client_requested_stream:
        return StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=upstream_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value,
                    litellm_metadata=litellm_metadata,
                )
            ),
            media_type="text/event-stream",
        )
    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _handle_codex_alibaba_token_plan_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    _ = endpoint, fastapi_response, user_api_key_dict
    rollup_kwargs: dict[str, Any] = {}

    async def _prepare_and_emit_route_log(
        **kwargs: Any,
    ) -> "_aawm_adapter_driver.CompletionAdapterRoutePlan":
        plan = await _prepare_codex_alibaba_token_plan_adapter_route(**kwargs)
        metadata = plan.perform_kwargs.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = plan.prepared_request_body.get("litellm_metadata")
        rollup_kwargs.update(_build_adapted_route_rollup_kwargs(metadata if isinstance(metadata, dict) else {}))
        _annotate_request_scope_for_adapted_access_log(request, plan.target_url)
        _emit_adapted_route_access_log(
            request=request,
            target_url=str(plan.target_url),
            request_body=plan.prepared_request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="Alibaba Token Plan",
        )
        return plan

    response = await _aawm_adapter_driver.run_completion_adapter_route(
        prepare=_prepare_and_emit_route_log,
        perform=_perform_codex_alibaba_token_plan_adapter_call,
        request=request,
        prepared_request_body=prepared_request_body,
        adapter_model=adapter_model,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    validated_response = await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter="codex_alibaba_token_plan_chat_completions_adapter",
        adapter_label="Alibaba Token Plan",
        intake_context=_build_malformed_tool_call_intake_context(
            request,
            prepared_request_body,
            adapter="codex_alibaba_token_plan_chat_completions_adapter",
            provider="alibaba_token_plan",
        ),
        request_body=prepared_request_body,
    )
    if isinstance(validated_response, StreamingResponse):
        return _record_adapted_completed_route_rollup_after_stream(
            validated_response,
            rollup_kwargs,
            adapter_label="Alibaba Token Plan",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="Alibaba Token Plan",
    )
    return validated_response


async def _handle_codex_opencode_zen_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    use_alias_candidate_probe: bool = False,
) -> Response:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    _ = fastapi_response
    normalized_request = await _anthropic_opencode_zen_normalization.normalize_codex_request(
        _get_anthropic_opencode_zen_normalization_runtime(),
        prepared_request_body,
        adapter_model=adapter_model,
    )
    request_body = normalized_request.request_body
    request_input = normalized_request.request_input
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        normalized_request.responses_api_request,
    )
    litellm_metadata = normalized_request.litellm_metadata
    completion_kwargs = normalized_request.completion_kwargs

    target_base_url = _get_opencode_zen_target_base()
    target_url = _join_opencode_zen_passthrough_url(
        base_target_url=target_base_url,
        endpoint="/v1/chat/completions",
    )
    api_key = await _load_opencode_zen_api_key_for_candidate(
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    custom_headers = BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=api_key,
        request=request,
    )
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers=custom_headers,
        credential_family="opencode",
        expected_target_family="opencode",
    )
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata)
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="OpenCode Zen",
    )
    try:
        completion_response = await litellm.acompletion(
            **completion_kwargs,
            api_key=api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            litellm_metadata=litellm_metadata,
            proxy_server_request={
                "headers": dict(request.headers),
                "body": request_body,
            },
            shared_session=_get_proxy_shared_aiohttp_session(),
        )
    except Exception as exc:
        if use_alias_candidate_probe and _opencode_zen_candidate_unavailable_detail(exc) is not None:
            _raise_opencode_zen_auto_agent_candidate_unavailable(exc)
        raise
    if bool(request_body.get("stream")):
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        return StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=adapter_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.OPENAI.value,
                    litellm_metadata=litellm_metadata,
                ),
                on_complete=lambda: _record_adapted_completed_route_rollup_turn(
                    rollup_kwargs,
                    adapter_label="OpenCode Zen",
                ),
            ),
            media_type="text/event-stream",
        )

    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    response_body = json.loads(_serialize_responses_adapter_response(responses_api_response))
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_opencode_zen_completion_adapter",
            adapter_label="OpenCode Zen chat-completions",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="OpenCode Zen",
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _perform_codex_auto_agent_openrouter_completion_request(
    *,
    request: Request,
    adapter_model: str,
    request_body: dict[str, Any],
    use_alias_candidate_probe: bool = False,
) -> Response:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    openrouter_api_key = _get_openrouter_api_key()
    if openrouter_api_key is None:
        exc = ProxyException(
            message=(
                "OpenRouter Codex auto-agent candidate requires " "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
            ),
            type="rate_limit_error",
            param="model",
            code=429,
        )
        setattr(
            exc,
            "detail",
            {
                "error": {
                    "message": exc.message,
                    "code": "aawm_codex_auto_agent_candidate_unavailable",
                }
            },
        )
        raise exc

    requested_model = request_body.get("model")
    upstream_adapter_model = _get_openrouter_completion_adapter_upstream_model(adapter_model) or adapter_model
    route_family = "codex_openrouter_completion_adapter"
    request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(request_body, route_family),
        tags_to_add=[
            "codex-openrouter-completion-adapter",
            f"codex-adapter-model:{adapter_model}",
            "codex-adapter-target:openrouter:/v1/chat/completions",
        ],
        extra_fields={
            "codex_adapter_model": adapter_model,
            "codex_adapter_original_model": requested_model,
            "codex_adapter_target_endpoint": "openrouter:/v1/chat/completions",
            "codex_adapter_input_shape": "openai_responses",
            "codex_adapter_output_shape": "openai_responses",
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="codex.openrouter_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": adapter_model,
                        "stream": bool(request_body.get("stream")),
                    },
                )
            ],
        },
    )
    request_input = request_body.get("input") or ""
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {key: value for key, value in request_body.items() if key not in {"input", "model", "litellm_metadata"}},
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=upstream_adapter_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    completion_kwargs["metadata"] = litellm_metadata
    (
        request_body,
        completion_kwargs,
        litellm_metadata,
    ) = _apply_openrouter_completion_message_sanitization(
        request_body=request_body,
        completion_kwargs=completion_kwargs,
        litellm_metadata=litellm_metadata,
        span_name="codex_openrouter.chat_message_shape_sanitized",
        tag="openrouter-chat-message-shape-sanitized",
    )

    target_base_url = _get_openrouter_target_base()
    target_url = f"{target_base_url.rstrip('/')}/v1/chat/completions"
    validation_headers = {
        **_build_openrouter_default_headers(),
        "Authorization": f"Bearer {openrouter_api_key}",
    }
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=target_url,
        headers=validation_headers,
        credential_family="openrouter",
        expected_target_family="openrouter",
    )
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))
    rollup_kwargs = _build_adapted_route_rollup_kwargs(litellm_metadata)
    _emit_adapted_route_access_log(
        request=request,
        target_url=target_url,
        request_body=request_body,
        rollup_kwargs=rollup_kwargs,
        adapter_label="OpenRouter chat-completions",
    )

    completion_response = await _perform_openrouter_completion_adapter_operation(
        adapter_model=upstream_adapter_model,
        operation=lambda: litellm.acompletion(
            **completion_kwargs,
            api_key=openrouter_api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            headers=_build_openrouter_default_headers(),
            litellm_metadata=litellm_metadata,
            proxy_server_request={
                "headers": dict(request.headers),
                "body": request_body,
            },
            shared_session=_get_proxy_shared_aiohttp_session(),
        ),
        log_warnings=not use_alias_candidate_probe,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    if bool(request_body.get("stream")):
        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        return StreamingResponse(
            _responses_sse_from_iterator(
                LiteLLMCompletionStreamingIterator(
                    model=upstream_adapter_model,
                    litellm_custom_stream_wrapper=completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                    custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
                    litellm_metadata=litellm_metadata,
                ),
                on_complete=lambda: _record_adapted_completed_route_rollup_turn(
                    rollup_kwargs,
                    adapter_label="OpenRouter chat-completions",
                ),
            ),
            media_type="text/event-stream",
        )

    responses_api_response = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    )
    response_body = json.loads(_serialize_responses_adapter_response(responses_api_response))
    if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
        _raise_codex_auto_agent_malformed_tool_call_text_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_completion_adapter",
            adapter_label="OpenRouter chat-completions",
            intake_context=_build_malformed_tool_call_intake_context(
                request,
                request_body,
                adapter="codex_auto_agent_openrouter_completion_adapter",
                upstream_url=target_url,
                provider="openrouter",
            ),
        )
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_completion_adapter",
            adapter_label="OpenRouter chat-completions",
        )
    _record_adapted_completed_route_rollup_turn(
        rollup_kwargs,
        adapter_label="OpenRouter chat-completions",
    )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _perform_codex_auto_agent_alias_candidate_request(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    candidate: dict[str, Any],
    candidate_body: dict[str, Any],
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
) -> Response:
    adapter_model = candidate["model"]

    async def _google() -> Response:
        return await _handle_codex_google_code_assist_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _antigravity() -> Response:
        return await _handle_codex_google_code_assist_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_completion() -> Response:
        return await _perform_codex_auto_agent_openrouter_completion_request(
            request=request,
            adapter_model=adapter_model,
            request_body=candidate_body,
            use_alias_candidate_probe=True,
        )

    async def _openrouter_responses() -> Response:
        return await _perform_codex_auto_agent_openrouter_responses_request(
            endpoint=endpoint,
            request=request,
            user_api_key_dict=user_api_key_dict,
            adapter_model=adapter_model,
            request_body=candidate_body,
            use_alias_candidate_probe=True,
        )

    async def _xai_oauth() -> Response:
        return await _perform_codex_auto_agent_oa_xai_responses_request(
            endpoint=endpoint,
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_body=candidate_body,
        )

    async def _grok_native() -> Response:
        return await _perform_codex_auto_agent_grok_native_responses_request(
            endpoint=endpoint,
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_body=candidate_body,
        )

    async def _opencode() -> Response:
        return await _handle_codex_opencode_zen_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _kimi_code() -> Response:
        return await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _alibaba_token_plan() -> Response:
        return await _handle_codex_alibaba_token_plan_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=adapter_model,
            use_alias_candidate_probe=True,
        )

    async def _native() -> Response:
        return await _perform_codex_auto_agent_native_openai_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            target_url=target_url,
            api_key=api_key,
            forward_headers=forward_headers,
            request_body=candidate_body,
        )

    return await _dispatch_auto_agent_alias_candidate_request(
        candidate=candidate,
        provider_handlers={
            _CODEX_AUTO_AGENT_GOOGLE_PROVIDER: _google,
            _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER: _antigravity,
            _CODEX_AUTO_AGENT_OPENCODE_PROVIDER: _opencode,
            _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER: _kimi_code,
            _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER: _alibaba_token_plan,
        },
        route_family_handlers={
            _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER: {
                "codex_openrouter_completion_adapter": _openrouter_completion,
                "*": _openrouter_responses,
            },
            _CODEX_AUTO_AGENT_XAI_PROVIDER: {
                "codex_xai_oauth_responses_adapter": _xai_oauth,
                "*": _grok_native,
            },
        },
        default_handler=_native,
    )


async def _handle_codex_auto_agent_alias_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    target_url: str,
    api_key: Optional[str],
    forward_headers: bool,
) -> Response:
    alias_model = (
        _normalize_codex_auto_agent_alias_model(prepared_request_body.get("model")) or _CODEX_AUTO_AGENT_MODEL_ALIAS
    )
    client_product_label = _extract_auto_agent_alias_client_product_label(request, prepared_request_body)

    async def _perform_candidate_request(
        *,
        candidate: dict[str, Any],
        candidate_body: dict[str, Any],
    ) -> Response:
        return await _perform_codex_auto_agent_alias_candidate_request(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            candidate=candidate,
            candidate_body=candidate_body,
            target_url=target_url,
            api_key=api_key,
            forward_headers=forward_headers,
        )

    services = _aawm_alias_interfaces.AliasRouteServices(
        select_candidate_fn=_select_codex_auto_agent_candidate,
        perform_candidate_request_fn=_perform_candidate_request,
        resolve_cooldown_publication_fn=_resolve_auto_agent_cooldown_publication_plan,
        publish_cooldown_memory_fn=_publish_codex_cooldown_memory,
        persist_cooldown_fn=_persist_codex_cooldown_durable,
        set_session_affinity_fn=_set_codex_auto_agent_session_affinity,
        add_alias_metadata_fn=_add_codex_auto_agent_alias_metadata,
        raise_redispatch_fn=_raise_codex_auto_agent_redispatch_required,
    )
    return await _aawm_alias_candidate_loop.handle_alias_route(
        services,
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        request=request,
        prepared_request_body=prepared_request_body,
        max_candidate_attempts=len(
            _resolve_aawm_alias_selection_enumeration(
                request,
                alias_model,
                client_product_label=client_product_label,
            ).candidates
        ),
        get_active_cooldown_state_fn=_get_codex_auto_agent_active_cooldown_state,
        attempts_metadata_key="codex_auto_agent_attempts",
        skipped_candidates_metadata_key="codex_auto_agent_skipped_candidates",
        no_candidate_detail="No Codex auto-agent alias candidates were available.",
        log_label="Codex",
    )


class BaseOpenAIPassThroughHandler:
    @staticmethod
    async def _prepare_openai_oa_xai_context(
        *,
        endpoint: str,
        request_body: dict[str, Any],
    ) -> Optional[tuple[str, str, dict[str, Any], str]]:
        (
            prepared_oa_xai,
            oa_xai_api_base,
            oa_xai_api_key,
        ) = await _prepare_oa_xai_passthrough_request(
            request_body,
            sanitize_responses_request=_is_openai_responses_endpoint(endpoint),
        )
        if not prepared_oa_xai:
            return None
        if oa_xai_api_base is None or oa_xai_api_key is None:
            raise Exception("OpenAI passthrough requests for xAI OAuth models require a managed xAI OAuth credential.")

        request_body["model"] = _to_xai_native_passthrough_model(request_body.get("model"))
        openai_route_family = _get_openai_passthrough_route_family(endpoint)
        encoded_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            endpoint=endpoint,
            base_target_url=oa_xai_api_base,
        )
        updated_url = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url=httpx.URL(oa_xai_api_base),
            path=encoded_endpoint,
            custom_llm_provider=litellm.LlmProviders.XAI,
        )
        prepared_request_body = _merge_litellm_metadata(
            request_body,
            tags_to_add=[
                f"openai-passthrough-route:{openai_route_family}",
            ],
            extra_fields={
                "openai_passthrough_route_family": openai_route_family,
            },
        )
        return (
            oa_xai_api_base,
            oa_xai_api_key,
            prepared_request_body,
            updated_url,
        )

    @staticmethod
    async def _prepare_openai_grok_native_oauth_context(
        *,
        endpoint: str,
        request: Request,
        request_body: dict[str, Any],
        extra_headers: Optional[dict],
    ) -> Optional[tuple[str, dict[str, Any], dict[str, Any], str]]:
        (
            prepared_grok_native,
            grok_target_base_url,
            grok_headers,
            grok_prepared_body,
        ) = await _prepare_grok_native_oauth_passthrough_request(
            request_body,
            request=request,
            tags_to_add=[
                "openai-grok-native-responses-adapter",
            ],
            extra_fields={
                "openai_passthrough_route_family": (_get_openai_passthrough_route_family(endpoint)),
                "grok_native_entrypoint": "openai_responses",
            },
        )
        if not prepared_grok_native:
            return None
        if grok_target_base_url is None:
            raise Exception("OpenAI passthrough requests for Grok native OAuth models require a Grok target base URL.")

        merged_headers = {
            **(extra_headers or {}),
            **grok_headers,
        }
        updated_url = _join_grok_passthrough_url(
            base_target_url=grok_target_base_url,
            endpoint="/v1/responses",
        )
        return (
            grok_target_base_url,
            merged_headers,
            grok_prepared_body,
            updated_url,
        )

    @staticmethod
    async def _base_openai_pass_through_handler(  # noqa: PLR0915
        endpoint: str,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        base_target_url: str,
        api_key: Optional[str],
        custom_llm_provider: litellm.LlmProviders,
        extra_headers: Optional[dict] = None,
        forward_headers: bool = False,
    ):
        encoded_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            endpoint=endpoint,
            base_target_url=base_target_url,
        )

        # Construct the full target URL by properly joining the base URL and endpoint path
        base_url = httpx.URL(base_target_url)
        updated_url = BaseOpenAIPassThroughHandler._join_url_paths(
            base_url=base_url,
            path=encoded_endpoint,
            custom_llm_provider=custom_llm_provider,
        )
        egress_credential_family: Optional[str] = None
        expected_target_family: Optional[str] = None
        endpoint_custom_body: Optional[dict[str, Any]] = None

        if request.method == "POST":
            request_body = await get_request_body(request)
            prepared_request_body = request_body
            body_was_prepared = False
            is_codex_responses_request = _request_uses_codex_native_auth(request) and _is_openai_responses_endpoint(
                endpoint
            )
            if (
                _resolve_codex_auto_agent_alias_model(
                    prepared_request_body,
                    endpoint=endpoint,
                )
                is not None
            ):
                is_codex_responses_request = True
            if is_codex_responses_request:
                prepared_request_body = _add_route_family_logging_metadata(
                    prepared_request_body,
                    "codex_responses",
                )
                (
                    prepared_request_body,
                    _codex_tool_description_patch_events,
                ) = _apply_codex_tool_description_patches_to_request_body(prepared_request_body)
                (
                    prepared_request_body,
                    _codex_unsupported_hosted_tools,
                ) = _drop_unsupported_codex_hosted_tools_from_request_body(prepared_request_body)
                (
                    prepared_request_body,
                    _codex_unsupported_request_params,
                ) = _drop_unsupported_codex_request_params_from_request_body(prepared_request_body)
                (
                    prepared_request_body,
                    _codex_unsupported_input_items,
                ) = _drop_unsupported_codex_input_items_from_request_body(prepared_request_body)
                if _is_oa_xai_request_body(prepared_request_body) or _is_grok_native_oauth_request_body(
                    prepared_request_body
                ):
                    (
                        prepared_request_body,
                        _codex_removed_empty_tool_choice,
                    ) = _drop_tool_choice_without_tools_from_request_body(prepared_request_body)
                prepared_request_body = _add_codex_request_breakout_logging_metadata(prepared_request_body)
            oa_xai_context = await BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                endpoint=endpoint,
                request_body=prepared_request_body,
            )
            if oa_xai_context is not None:
                body_was_prepared = True
                (
                    base_target_url,
                    api_key,
                    prepared_request_body,
                    updated_url,
                ) = oa_xai_context
                custom_llm_provider = litellm.LlmProviders.XAI
                forward_headers = False
                egress_credential_family = "xai"
                expected_target_family = "xai"
            elif _is_openai_responses_endpoint(endpoint):
                grok_native_context = await BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
                    endpoint=endpoint,
                    request=request,
                    request_body=prepared_request_body,
                    extra_headers=extra_headers,
                )
                if grok_native_context is not None:
                    body_was_prepared = True
                    (
                        base_target_url,
                        extra_headers,
                        prepared_request_body,
                        updated_url,
                    ) = grok_native_context
                    api_key = None
                    custom_llm_provider = litellm.LlmProviders.XAI
                    forward_headers = False
                    egress_credential_family = "xai"
                    expected_target_family = "xai"
                elif is_codex_responses_request:
                    codex_auto_agent_alias = _resolve_codex_auto_agent_alias_model(
                        prepared_request_body,
                        endpoint=endpoint,
                    )
                    if codex_auto_agent_alias is not None:
                        (
                            prepared_request_body,
                            _codex_auto_agent_guidance_changes,
                        ) = _apply_codex_auto_agent_prevention_guidance_to_request_body(prepared_request_body)
                        (
                            prepared_request_body,
                            _codex_read_guidance_changes,
                        ) = _apply_aawm_read_agent_guidance_to_request_body(
                            prepared_request_body,
                            alias_model=codex_auto_agent_alias,
                            target_field="instructions",
                        )
                        prepared_request_body = _prepare_request_body_for_passthrough_observability(
                            request=request,
                            request_body=prepared_request_body,
                        )
                        if prepared_request_body is not request_body:
                            _safe_set_request_parsed_body(request, prepared_request_body)
                        return await _handle_codex_auto_agent_alias_route(
                            endpoint=endpoint,
                            request=request,
                            fastapi_response=fastapi_response,
                            user_api_key_dict=user_api_key_dict,
                            prepared_request_body=prepared_request_body,
                            target_url=str(updated_url),
                            api_key=api_key,
                            forward_headers=forward_headers,
                        )
                    opencode_zen_adapter_model = _resolve_codex_opencode_zen_adapter_model(
                        prepared_request_body,
                        endpoint=endpoint,
                    )
                    if opencode_zen_adapter_model is not None:
                        prepared_request_body = _prepare_request_body_for_passthrough_observability(
                            request=request,
                            request_body=prepared_request_body,
                        )
                        if prepared_request_body is not request_body:
                            _safe_set_request_parsed_body(request, prepared_request_body)
                        return await _handle_codex_opencode_zen_adapter_route(
                            endpoint=endpoint,
                            request=request,
                            fastapi_response=fastapi_response,
                            user_api_key_dict=user_api_key_dict,
                            prepared_request_body=prepared_request_body,
                            adapter_model=opencode_zen_adapter_model,
                        )
                    kimi_code_adapter_model = _resolve_codex_kimi_chat_completions_adapter_model(
                        prepared_request_body,
                        endpoint=endpoint,
                    )
                    if kimi_code_adapter_model is not None:
                        prepared_request_body = _prepare_request_body_for_passthrough_observability(
                            request=request,
                            request_body=prepared_request_body,
                        )
                        if prepared_request_body is not request_body:
                            _safe_set_request_parsed_body(request, prepared_request_body)
                        return await _handle_codex_kimi_chat_completions_adapter_route(
                            endpoint=endpoint,
                            request=request,
                            fastapi_response=fastapi_response,
                            user_api_key_dict=user_api_key_dict,
                            prepared_request_body=prepared_request_body,
                            adapter_model=kimi_code_adapter_model,
                        )
                    alibaba_token_plan_adapter_model = _resolve_codex_alibaba_token_plan_adapter_model(
                        prepared_request_body,
                        endpoint=endpoint,
                    )
                    if alibaba_token_plan_adapter_model is not None:
                        prepared_request_body = _prepare_request_body_for_passthrough_observability(
                            request=request,
                            request_body=prepared_request_body,
                        )
                        if prepared_request_body is not request_body:
                            _safe_set_request_parsed_body(request, prepared_request_body)
                        return await _handle_codex_alibaba_token_plan_adapter_route(
                            endpoint=endpoint,
                            request=request,
                            fastapi_response=fastapi_response,
                            user_api_key_dict=user_api_key_dict,
                            prepared_request_body=prepared_request_body,
                            adapter_model=alibaba_token_plan_adapter_model,
                        )
                    antigravity_adapter_model = _resolve_codex_antigravity_code_assist_adapter_model(
                        prepared_request_body,
                        endpoint=endpoint,
                    )
                    if antigravity_adapter_model is not None:
                        prepared_request_body = _prepare_request_body_for_passthrough_observability(
                            request=request,
                            request_body=prepared_request_body,
                        )
                        if prepared_request_body is not request_body:
                            _safe_set_request_parsed_body(request, prepared_request_body)
                        return await _handle_codex_google_code_assist_adapter_route(
                            endpoint=endpoint,
                            request=request,
                            fastapi_response=fastapi_response,
                            user_api_key_dict=user_api_key_dict,
                            prepared_request_body=prepared_request_body,
                            adapter_model=antigravity_adapter_model,
                            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
                        )

                    google_adapter_model = _resolve_codex_google_code_assist_adapter_model(
                        prepared_request_body,
                        endpoint=endpoint,
                    )
                    if google_adapter_model is not None:
                        prepared_request_body = _prepare_request_body_for_passthrough_observability(
                            request=request,
                            request_body=prepared_request_body,
                        )
                        if prepared_request_body is not request_body:
                            _safe_set_request_parsed_body(request, prepared_request_body)
                        return await _handle_codex_google_code_assist_adapter_route(
                            endpoint=endpoint,
                            request=request,
                            fastapi_response=fastapi_response,
                            user_api_key_dict=user_api_key_dict,
                            prepared_request_body=prepared_request_body,
                            adapter_model=google_adapter_model,
                        )
                    direct_model = prepared_request_body.get("model")
                    if isinstance(direct_model, str) and direct_model:
                        (
                            prepared_request_body,
                            _direct_reasoning_effort_metadata,
                        ) = _normalize_codex_reasoning_effort_for_resolved_route(
                            prepared_request_body,
                            resolved_route={
                                "provider": litellm.LlmProviders.OPENAI.value,
                                "model": direct_model,
                                "route_family": "codex_responses",
                            },
                        )
            else:
                prepared_request_body = _add_route_family_logging_metadata(
                    prepared_request_body,
                    _get_openai_passthrough_route_family(endpoint),
                )
            prepared_request_body = _prepare_request_body_for_passthrough_observability(
                request=request,
                request_body=prepared_request_body,
            )
            if body_was_prepared or prepared_request_body is not request_body:
                _safe_set_request_parsed_body(request, prepared_request_body)
                endpoint_custom_body = prepared_request_body

        ## check for streaming
        is_streaming_request = "stream" in str(updated_url)

        ## CREATE PASS-THROUGH
        endpoint_func = create_pass_through_route(
            endpoint=endpoint,
            target=str(updated_url),
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=api_key, request=request, extra_headers=extra_headers
            ),
            _forward_headers=forward_headers,
            is_streaming_request=is_streaming_request,  # type: ignore
            custom_llm_provider=custom_llm_provider.value
            if isinstance(custom_llm_provider, litellm.LlmProviders)
            else custom_llm_provider,
            egress_credential_family=egress_credential_family,
            expected_target_family=expected_target_family,
        )  # dynamically construct pass-through endpoint based on incoming path
        return await endpoint_func(
            request,
            fastapi_response,
            user_api_key_dict,
            custom_body=endpoint_custom_body,
        )

    @staticmethod
    def _append_openai_beta_header(headers: dict, request: Request) -> dict:
        """
        Appends the OpenAI-Beta header to the headers if the request is an OpenAI Assistants API request
        """
        if RouteChecks._is_assistants_api_request(request) is True and "OpenAI-Beta" not in headers:
            headers["OpenAI-Beta"] = "assistants=v2"
        return headers

    @staticmethod
    def _assemble_headers(api_key: Optional[str], request: Request, extra_headers: Optional[dict] = None) -> dict:
        base_headers = {}
        if api_key is not None:
            base_headers = {
                "authorization": "Bearer {}".format(api_key),
                "api-key": "{}".format(api_key),
            }
        if extra_headers is not None:
            base_headers.update(extra_headers)
        return BaseOpenAIPassThroughHandler._append_openai_beta_header(
            headers=base_headers,
            request=request,
        )

    @staticmethod
    def _join_url_paths(
        base_url: httpx.URL,
        path: str,
        custom_llm_provider: Union[litellm.LlmProviders, str],
    ) -> str:
        """
        Properly joins a base URL with a path, preserving any existing path in the base URL.
        """
        # Join paths correctly by removing trailing/leading slashes as needed
        if not base_url.path or base_url.path == "/":
            # If base URL has no path, just use the new path
            joined_path_str = str(base_url.copy_with(path=path))
        else:
            # Otherwise, combine the paths
            base_path = base_url.path.rstrip("/")
            clean_path = path.lstrip("/")
            full_path = f"{base_path}/{clean_path}"
            joined_path_str = str(base_url.copy_with(path=full_path))

        # Apply OpenAI-specific path handling for both branches
        if custom_llm_provider == litellm.LlmProviders.OPENAI and "/v1/" not in joined_path_str:
            # Insert v1 after api.openai.com for OpenAI requests
            joined_path_str = joined_path_str.replace("api.openai.com/", "api.openai.com/v1/")

        return joined_path_str

    @staticmethod
    def _normalize_endpoint_for_target(endpoint: str, base_target_url: str) -> str:
        normalized_endpoint = httpx.URL(endpoint).path
        if not normalized_endpoint.startswith("/"):
            normalized_endpoint = "/" + normalized_endpoint

        base_url = httpx.URL(base_target_url)
        if (
            base_url.host
            and "chatgpt.com" in base_url.host
            and base_url.path.rstrip("/") == "/backend-api/codex"
            and normalized_endpoint.startswith("/v1/")
        ):
            return normalized_endpoint[len("/v1") :]
        if base_url.path.rstrip("/") == "/v1" and normalized_endpoint.startswith("/v1/"):
            return normalized_endpoint[len("/v1") :]
        return normalized_endpoint


@router.api_route(
    "/cursor/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Cursor Pass-through", "pass-through"],
)
async def cursor_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Pass-through endpoint for the Cursor Cloud Agents API.

    Supports all Cursor Cloud Agents endpoints:
    - GET    /v0/agents         — List agents
    - POST   /v0/agents         — Launch an agent
    - GET    /v0/agents/{id}    — Agent status
    - GET    /v0/agents/{id}/conversation — Agent conversation
    - POST   /v0/agents/{id}/followup    — Add follow-up
    - POST   /v0/agents/{id}/stop        — Stop an agent
    - DELETE /v0/agents/{id}    — Delete an agent
    - GET    /v0/me             — API key info
    - GET    /v0/models         — List models
    - GET    /v0/repositories   — List GitHub repositories

    Uses Basic Authentication (base64-encoded `API_KEY:`).

    Credential lookup order:
    1. passthrough_endpoint_router (config.yaml deployments with use_in_pass_through)
    2. litellm.credential_list (credentials added via UI)
    3. CURSOR_API_KEY environment variable
    """
    import base64

    base_target_url = os.getenv("CURSOR_API_BASE") or "https://api.cursor.com"

    cursor_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="cursor",
        region_name=None,
    )

    if cursor_api_key is None:
        for credential in litellm.credential_list:
            if credential.credential_info and credential.credential_info.get("custom_llm_provider") == "cursor":
                cursor_api_key = credential.credential_values.get("api_key")
                credential_api_base = credential.credential_values.get("api_base")
                if credential_api_base:
                    base_target_url = credential_api_base
                break

    if cursor_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Cursor API key not found. Add Cursor credentials via the UI (Models + Endpoints → LLM Credentials) or set CURSOR_API_KEY environment variable.",
        )

    encoded_endpoint = httpx.URL(endpoint).path

    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    auth_value = base64.b64encode(f"{cursor_api_key}:".encode("utf-8")).decode("ascii")

    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": f"Basic {auth_value}"},
        custom_llm_provider="cursor",
        caller_managed_hidden_retry=True,
    )
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


async def vertex_ai_live_websocket_passthrough(
    websocket: WebSocket,
    model: Optional[str] = None,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
):
    """
    Vertex AI Live API WebSocket Pass-through Function

    This function provides WebSocket passthrough functionality for Vertex AI Live API,
    allowing real-time communication with Google's Live API service.

    Registration must keep authentication outside this function. In proxy_server.py
    register a thin wrapper that depends on ``user_api_key_auth_websocket`` and
    forwards the resolved ``user_api_key_dict`` here. Do not register this function
    directly with ``app.websocket(...)`` without that auth dependency.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    # RR-054 #16: fail closed before accept/credential exchange if auth was omitted.
    if user_api_key_dict is None:
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.close(code=1008, reason="Authentication required")
        elif websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1008, reason="Authentication required")
        raise ValueError("user_api_key_dict is required for WebSocket passthrough")

    # RR-054 #16: do not accept the client websocket until Vertex credentials are ready.
    # Accepting first forces clients into an open socket that can still fail mid-handshake
    # after a credential exchange using server-held secrets.
    incoming_headers = dict(websocket.headers)
    vertex_credentials_config = passthrough_endpoint_router.get_vertex_credentials(
        project_id=vertex_project,
        location=vertex_location,
    )

    if vertex_credentials_config is None:
        # Attempt to load defaults from environment/config if not already initialised
        passthrough_endpoint_router.set_default_vertex_config()
        vertex_credentials_config = passthrough_endpoint_router.get_vertex_credentials(
            project_id=vertex_project,
            location=vertex_location,
        )

    resolved_project = vertex_project
    resolved_location: Optional[str] = vertex_location
    credentials_value: Optional[str] = None

    if vertex_credentials_config is not None:
        resolved_project = resolved_project or vertex_credentials_config.vertex_project
        temp_location = resolved_location or vertex_credentials_config.vertex_location
        # Ensure resolved_location is a string
        if isinstance(temp_location, dict):
            resolved_location = str(temp_location)
        elif temp_location is not None:
            resolved_location = str(temp_location)
        else:
            resolved_location = None
        credentials_value = (
            str(vertex_credentials_config.vertex_credentials)
            if vertex_credentials_config.vertex_credentials is not None
            else None
        )

    try:
        resolved_location = resolved_location or (vertex_llm_base.get_default_vertex_location())
        if model:
            resolved_location = vertex_llm_base.get_vertex_region(
                vertex_region=resolved_location,
                model=model,
            )

        (
            access_token,
            resolved_project,
        ) = await vertex_llm_base._ensure_access_token_async(
            credentials=credentials_value,
            project_id=resolved_project,
            custom_llm_provider="vertex_ai_beta",
        )
    except Exception as e:
        verbose_proxy_logger.exception("Failed to prepare Vertex AI credentials for live passthrough")
        # Log the authentication failure using proxy_logging_obj
        if proxy_logging_obj and user_api_key_dict:
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data={},
            )
        # Not accepted yet: reject the upgrade if still connecting.
        if websocket.client_state == WebSocketState.CONNECTING:
            await websocket.close(code=1011, reason="Vertex AI authentication failed")
        elif websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason="Vertex AI authentication failed")
        return

    await websocket.accept()

    host_location = resolved_location or vertex_llm_base.get_default_vertex_location()
    host = "aiplatform.googleapis.com" if host_location == "global" else f"{host_location}-aiplatform.googleapis.com"
    service_url = f"wss://{host}/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

    upstream_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    if resolved_project:
        upstream_headers["x-goog-user-project"] = resolved_project

    # Forward any custom x-goog-* headers provided by the caller if we haven't overridden them
    for header_name, header_value in incoming_headers.items():
        lower_header = header_name.lower()
        if lower_header.startswith("x-goog-") and header_name not in upstream_headers:
            upstream_headers[header_name] = header_value

    # Use the new WebSocket passthrough pattern (auth already validated above).
    return await websocket_passthrough_request(
        websocket=websocket,
        target=service_url,
        custom_headers=upstream_headers,
        user_api_key_dict=user_api_key_dict,
        forward_headers=False,
        endpoint="/vertex_ai/live",
        accept_websocket=False,
    )


def create_vertex_ai_live_websocket_endpoint():
    """
    Create a Vertex AI Live WebSocket endpoint using the new passthrough pattern.

    This demonstrates how to use the create_websocket_passthrough_route function
    for a provider-specific WebSocket endpoint.
    """
    # This would be used like:
    # endpoint_func = create_vertex_ai_live_websocket_endpoint()
    # app.websocket("/vertex_ai/live")(endpoint_func)

    # For now, we'll keep the existing implementation since it has
    # provider-specific logic for Vertex AI credentials and headers
    return vertex_ai_live_websocket_passthrough


def create_generic_websocket_passthrough_endpoint(
    provider: str,
    target_url: str,
    custom_headers: Optional[dict] = None,
    forward_headers: bool = False,
    cost_per_request: Optional[float] = None,
):
    """
    Create a generic WebSocket passthrough endpoint for any provider.

    This demonstrates the new WebSocket passthrough pattern that's similar to
    the HTTP create_pass_through_route function.

    Args:
        provider: The provider name (e.g., "anthropic", "cohere")
        target_url: The target WebSocket URL
        custom_headers: Custom headers to include
        forward_headers: Whether to forward incoming headers

    Returns:
        A WebSocket endpoint function that can be registered with app.websocket()

    Example usage:
        # Create a WebSocket endpoint for Anthropic
        anthropic_ws_func = create_generic_websocket_passthrough_endpoint(
            provider="anthropic",
            target_url="wss://api.anthropic.com/v1/ws",
            custom_headers={"x-api-key": "your-api-key"},
            forward_headers=True
        )

        # Register it in proxy_server.py
        app.websocket("/anthropic/ws")(anthropic_ws_func)
    """
    return create_websocket_passthrough_route(
        endpoint=f"/{provider}/ws",
        target=target_url,
        custom_headers=custom_headers,
        _forward_headers=forward_headers,
        cost_per_request=cost_per_request,
    )


# ---------------------------------------------------------------------------
# Wave 6B runtime configuration and facade installation
# ---------------------------------------------------------------------------

def _wave6b_common_live_runtime() -> _wave6b_common.Runtime:
    """Build a common Runtime with live host-global lookup."""
    return _wave6b_common.Runtime(
        extract_status_code=lambda exc: _extract_google_adapter_exception_status_code(exc),
        extract_detail=lambda exc: _extract_google_adapter_exception_detail(exc),
    )


def _wave6b_antigravity_live_runtime() -> _wave6b_antigravity_runtime.Runtime:
    """Build an Antigravity Runtime with live host-global lookup."""
    return _wave6b_antigravity_runtime.Runtime(
        clean_value=lambda v: _clean_codex_auth_value(v),
        merge_metadata=lambda *a, **kw: _merge_litellm_metadata(*a, **kw),
        prepare_observability=lambda **kw: _prepare_request_body_for_passthrough_observability(**kw),
        split_provider_prefix=lambda v: _split_anthropic_adapter_provider_prefix(v),
        format_api_key=lambda v: _format_litellm_passthrough_api_key(v),
        oauth_error_code=lambda r: _get_oauth_token_error_code(r),
        allowed_models=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS,
    )


# -- OpenRouter runtime configuration --
_wave6b_openrouter_runtime.configure_openrouter_runtime(
    _wave6b_openrouter_runtime.Runtime(
        retry_transport_runtime=_ANTHROPIC_OPENROUTER_RETRY_TRANSPORT_RUNTIME,
        clean_secret_string=lambda v: _clean_secret_string(v),
        get_first_secret_value=lambda names: _get_first_secret_value(names),
        getenv=lambda name: os.getenv(name),
        get_secret_str=lambda name: get_secret_str(name),
        sanitize_opencode_zen_completion_messages=lambda kw: _sanitize_opencode_zen_completion_messages_for_chat_completion(kw),
        chat_message_role=lambda msg: _opencode_zen_chat_message_role(msg),
        chat_message_tool_call_ids=lambda msg: _opencode_zen_chat_message_tool_call_ids(msg),
        chat_message_tool_result_id=lambda msg: _opencode_zen_chat_message_tool_result_id(msg),
        is_empty_text_content=lambda c: _is_codex_google_code_assist_empty_text_content(c),
        merge_litellm_metadata=lambda *a, **kw: _merge_litellm_metadata(*a, **kw),
        build_langfuse_span_descriptor=lambda *a, **kw: _build_langfuse_span_descriptor(*a, **kw),
    )
)

# -- NVIDIA runtime configuration --
_wave6b_nvidia_runtime.configure_nvidia_runtime(
    _wave6b_nvidia_runtime.NvidiaRuntimeDependencies(
        get_first_secret_value=lambda names: _get_first_secret_value(names),
        clean_secret_string=lambda v: _clean_secret_string(v),
        clean_auth_value=lambda v: _clean_codex_auth_value(v),
        get_env=lambda name: os.getenv(name),
        sleep=lambda seconds: asyncio.sleep(seconds),
        log_debug=verbose_proxy_logger.debug,
        log_warning=verbose_proxy_logger.warning,
    )
)

# -- xAI request-prep runtime configuration --
_wave6b_xai_request_prep.configure_xai_request_prep_runtime(
    _wave6b_xai_request_prep.build_default_xai_request_prep_runtime(
        get_model_metadata_entry=lambda m: _get_model_metadata_entry(m),
        get_openai_tool_type=lambda t: _get_openai_tool_type(t),
        normalize_low_cardinality_tag_value=lambda v: _normalize_low_cardinality_tag_value(v),
        dedupe_sorted_str_list=lambda lst: _dedupe_sorted_str_list(lst),
        merge_litellm_metadata=lambda *a, **kw: _merge_litellm_metadata(*a, **kw),
        build_langfuse_span_descriptor=lambda *a, **kw: _build_langfuse_span_descriptor(*a, **kw),
        drop_unsupported_codex_hosted_tools_from_request_body=lambda b: _drop_unsupported_codex_hosted_tools_from_request_body(b),
        drop_unsupported_codex_request_params_from_request_body=lambda b: _drop_unsupported_codex_request_params_from_request_body(b),
        drop_unsupported_codex_input_items_from_request_body=lambda b: _drop_unsupported_codex_input_items_from_request_body(b),
        drop_tool_choice_without_tools_from_request_body=lambda b: _drop_tool_choice_without_tools_from_request_body(b),
        replace_request_body_in_place=lambda orig, new: _replace_request_body_in_place(orig, new),
        safe_get_request_headers=lambda req: _safe_get_request_headers(req),
        get_case_insensitive_header=lambda hdrs, name: _get_case_insensitive_header(hdrs, name),
        get_rewrite_input_item_types_for_model=lambda m: _get_rewrite_input_item_types_for_model(m),
        get_grok_passthrough_target_base=lambda: _get_grok_passthrough_target_base(),
        get_grok_native_oauth_access_token=lambda: get_grok_native_oauth_access_token(),
    )
)

# -- OpenCode Zen install (configures runtime + publishes same-object facades) --
_wave6b_opencode_zen_runtime.install(globals())


# Wave 4 runtime injection -- FunctionType rebind for live host-global lookup
_aawm_lane_keys.install(globals())
_aawm_selection.install(globals())
_aawm_cooldown_state.install(globals())
_aawm_error_signals.install(globals())
_aawm_cooldown_apply.install(globals())
_aawm_attempt_records.install(globals())
_aawm_selection._attach_aawm_alias_routing_state_sources = (
    _aawm_cooldown_state._attach_aawm_alias_routing_state_sources
)
_aawm_adapter_model_resolution.install(globals())
_aawm_adapter_runtime.install(globals())
_google_env_policy.install(globals())
_google_context_window.install(globals())
_google_error_signals.install(globals())
_grok_side_channel.install(globals())

# Wave 5D runtime injection -- FunctionType rebind for live host-global lookup
_aawm_audit_context.install(globals())
_aawm_audit_build.install(globals())
_aawm_audit_persist.install(globals())
_aawm_audit_events.install(globals())

# Bind Wave 5D runtimes after installation so module fixtures restore the
# canonical rebound facades instead of pre-install wrappers.
_aawm_audit_context.configure_audit_context_runtime(
    clean_secret_string=_clean_secret_string,
    extract_metadata_value=_extract_auto_agent_alias_metadata_value,
    extract_client_product_label=_extract_auto_agent_alias_client_product_label,
    resolve_host_attribution=_resolve_auto_agent_alias_route_host_attribution,
    extract_session_id=_extract_auto_agent_alias_session_id,
    build_rollup_group_header_label=_build_auto_agent_alias_rollup_group_header_label,
    has_continuation_state=_codex_auto_agent_request_has_continuation_state,
)
_aawm_audit_build.configure_audit_build_runtime(
    get_request_context=_get_auto_agent_alias_request_context,
    attach_terminal_context_fields=_attach_auto_agent_alias_terminal_context_fields,
    format_timestamp=_format_auto_agent_alias_timestamp,
    extract_metadata_value=_extract_auto_agent_alias_metadata_value,
    extract_incoming_endpoint=_extract_auto_agent_alias_incoming_endpoint,
    resolve_outgoing_target=_resolve_auto_agent_alias_route_rollup_outgoing_target,
    to_int=_auto_agent_alias_int,
    cooldown_until=_auto_agent_alias_cooldown_until,
)
_aawm_audit_persist.configure_audit_persist_runtime(
    record_route_status_rollup=_record_auto_agent_alias_route_status_rollup,
    verbose_json_enabled=_aawm_alias_route_verbose_json_enabled,
    healthy_json_enabled=_aawm_alias_route_healthy_json_enabled,
)
_aawm_audit_events.configure_audit_events_runtime(
    get_request_context=_get_auto_agent_alias_request_context,
    attach_terminal_context_fields=_attach_auto_agent_alias_terminal_context_fields,
    format_timestamp=_format_auto_agent_alias_timestamp,
    extract_metadata_value=_extract_auto_agent_alias_metadata_value,
    extract_incoming_endpoint=_extract_auto_agent_alias_incoming_endpoint,
    resolve_codex_session_key=_resolve_codex_auto_agent_session_key,
    resolve_anthropic_session_key=_resolve_anthropic_auto_agent_session_key,
    emit_route_event=_emit_auto_agent_alias_route_event,
    build_audit_events=_build_auto_agent_alias_audit_events,
    persist_audit_only_events=_persist_auto_agent_alias_audit_only_events_best_effort,
)


# ---------------------------------------------------------------------------
# Wave 6D observability-metadata same-object facades
# ---------------------------------------------------------------------------
_merge_litellm_metadata = _aawm_observability_metadata._merge_litellm_metadata
_format_langfuse_span_timestamp = _aawm_observability_metadata._format_langfuse_span_timestamp
_build_langfuse_span_descriptor = _aawm_observability_metadata._build_langfuse_span_descriptor
_normalize_low_cardinality_tag_value = _aawm_observability_metadata._normalize_low_cardinality_tag_value
_dedupe_sorted_str_list = _aawm_observability_metadata._dedupe_sorted_str_list
_iter_anthropic_text_fragments = _aawm_observability_metadata._iter_anthropic_text_fragments
_extract_claude_agent_and_tenant_from_request_body = _aawm_observability_metadata._extract_claude_agent_and_tenant_from_request_body
_add_claude_child_agent_observability_metadata = _aawm_observability_metadata._add_claude_child_agent_observability_metadata
_detect_claude_post_rewrite_context_files = _aawm_observability_metadata._detect_claude_post_rewrite_context_files
_add_claude_post_rewrite_context_file_logging_metadata = _aawm_observability_metadata._add_claude_post_rewrite_context_file_logging_metadata
_get_nested_str_value = _aawm_observability_metadata._get_nested_str_value
_extract_passthrough_session_id = _aawm_observability_metadata._extract_passthrough_session_id
_normalize_passthrough_repository = _aawm_observability_metadata._normalize_passthrough_repository
_extract_passthrough_repository_from_text = _aawm_observability_metadata._extract_passthrough_repository_from_text
_walk_request_value_with_budget = _aawm_observability_metadata._walk_request_value_with_budget
_extract_passthrough_repository_from_body_text = _aawm_observability_metadata._extract_passthrough_repository_from_body_text
_extract_passthrough_repository = _aawm_observability_metadata._extract_passthrough_repository
_get_passthrough_trace_environment = _aawm_observability_metadata._get_passthrough_trace_environment
_add_passthrough_trace_context_metadata = _aawm_observability_metadata._add_passthrough_trace_context_metadata
_truncate_tool_definition_string = _aawm_observability_metadata._truncate_tool_definition_string
_redact_tool_definition_string = _aawm_observability_metadata._redact_tool_definition_string
_sanitize_tool_definition_value = _aawm_observability_metadata._sanitize_tool_definition_value
_tool_definition_name = _aawm_observability_metadata._tool_definition_name
_tool_definition_description = _aawm_observability_metadata._tool_definition_description
_tool_definition_parameters = _aawm_observability_metadata._tool_definition_parameters
_build_tool_definition_snapshot_entry = _aawm_observability_metadata._build_tool_definition_snapshot_entry
_tool_definition_snapshot_hash = _aawm_observability_metadata._tool_definition_snapshot_hash
_build_passthrough_tool_definition_metadata = _aawm_observability_metadata._build_passthrough_tool_definition_metadata
_add_passthrough_tool_definition_metadata = _aawm_observability_metadata._add_passthrough_tool_definition_metadata
_prepare_request_body_for_passthrough_observability = _aawm_observability_metadata._prepare_request_body_for_passthrough_observability
_extract_openai_passthrough_tool_choice = _aawm_observability_metadata._extract_openai_passthrough_tool_choice
_extract_claude_request_breakout_fields = _aawm_observability_metadata._extract_claude_request_breakout_fields
_add_claude_request_breakout_logging_metadata = _aawm_observability_metadata._add_claude_request_breakout_logging_metadata
_extract_gemini_request_breakout_fields = _aawm_observability_metadata._extract_gemini_request_breakout_fields
_add_gemini_request_breakout_logging_metadata = _aawm_observability_metadata._add_gemini_request_breakout_logging_metadata
_extract_codex_request_breakout_fields = _aawm_observability_metadata._extract_codex_request_breakout_fields
_add_codex_request_breakout_logging_metadata = _aawm_observability_metadata._add_codex_request_breakout_logging_metadata
_parse_anthropic_billing_header_text = _aawm_observability_metadata._parse_anthropic_billing_header_text
_extract_anthropic_billing_header_fields = _aawm_observability_metadata._extract_anthropic_billing_header_fields
_extract_anthropic_billing_header_fields_from_request_body = _aawm_observability_metadata._extract_anthropic_billing_header_fields_from_request_body
_add_anthropic_billing_header_logging_metadata = _aawm_observability_metadata._add_anthropic_billing_header_logging_metadata
_add_claude_persisted_output_logging_metadata = _aawm_observability_metadata._add_claude_persisted_output_logging_metadata
_add_route_family_logging_metadata = _aawm_observability_metadata._add_route_family_logging_metadata
_ANTHROPIC_BILLING_HEADER_PREFIX = _aawm_observability_metadata._ANTHROPIC_BILLING_HEADER_PREFIX
_AAWM_TOOL_DEFINITION_CAPTURE_VERSION = _aawm_observability_metadata._AAWM_TOOL_DEFINITION_CAPTURE_VERSION
_AAWM_TOOL_DEFINITION_MAX_TOOLS = _aawm_observability_metadata._AAWM_TOOL_DEFINITION_MAX_TOOLS
_PASSTHROUGH_SESSION_ID_HEADER_NAMES = _aawm_observability_metadata._PASSTHROUGH_SESSION_ID_HEADER_NAMES
_PASSTHROUGH_REPOSITORY_HEADER_NAMES = _aawm_observability_metadata._PASSTHROUGH_REPOSITORY_HEADER_NAMES
_PASSTHROUGH_REPOSITORY_BODY_KEYS = _aawm_observability_metadata._PASSTHROUGH_REPOSITORY_BODY_KEYS
_PASSTHROUGH_REPOSITORY_TEXT_PATTERNS = _aawm_observability_metadata._PASSTHROUGH_REPOSITORY_TEXT_PATTERNS
_PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES = _aawm_observability_metadata._PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES
_PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES = _aawm_observability_metadata._PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES
_append_codex_auto_agent_prevention_guidance_to_instructions = _aawm_alias_guidance._append_codex_auto_agent_prevention_guidance_to_instructions
_is_aawm_read_agent_alias_model = _aawm_alias_guidance._is_aawm_read_agent_alias_model
_append_aawm_read_agent_guidance_to_text = _aawm_alias_guidance._append_aawm_read_agent_guidance_to_text
_append_aawm_read_agent_guidance_to_anthropic_system = _aawm_alias_guidance._append_aawm_read_agent_guidance_to_anthropic_system
_apply_aawm_read_agent_guidance_to_request_body = _aawm_alias_guidance._apply_aawm_read_agent_guidance_to_request_body
_apply_codex_auto_agent_prevention_guidance_to_request_body = _aawm_alias_guidance._apply_codex_auto_agent_prevention_guidance_to_request_body


# ---------------------------------------------------------------------------
# Wave 6D request-policy runtime configuration and facade installation
# ---------------------------------------------------------------------------

# 1. Observability metadata: bind host callbacks for tenant, headers, env.
_aawm_observability_metadata.configure_observability_metadata_runtime(
    get_explicit_tenant_id=_get_aawm_tenant_header,
    get_request_headers=_safe_get_request_headers,
    get_env=os.getenv,
)

# 2. Persisted-output logging callback: publish into host globals so rebound
#    persisted-output functions resolve the callback at call time.
_persisted_output_logging_callback = (
    _aawm_observability_metadata._add_claude_persisted_output_logging_metadata
)

# 3. Persisted-output: bind runtime deps, configure callback, install facades.
_aawm_persisted_output.bind_runtime(globals())
_aawm_persisted_output.configure_persisted_output_logging_callback(
    _persisted_output_logging_callback
)
_aawm_persisted_output.install(globals())

# 4. Alias guidance: bind canonical observability merge/span callbacks.
_aawm_alias_guidance.configure_alias_guidance_runtime(
    callbacks=_aawm_alias_guidance.AliasGuidanceCallbacks(
        merge_litellm_metadata=_merge_litellm_metadata,
        build_langfuse_span_descriptor=_build_langfuse_span_descriptor,
    ),
)



# ---------------------------------------------------------------------------
# Wave 6E codex-tool-policy same-object facades
# ---------------------------------------------------------------------------

# Build the shared CodexToolPolicyCallbacks with live host-global lookups.
_CODEX_TOOL_POLICY_CALLBACKS = _aawm_codex_tool_policy.CodexToolPolicyCallbacks(
    normalize_tag_value=_normalize_low_cardinality_tag_value,
    dedupe_sorted=_dedupe_sorted_str_list,
    merge_metadata=_merge_litellm_metadata,
    build_span=_build_langfuse_span_descriptor,
    get_model_cost_map=lambda: litellm.model_cost,
    normalize_grok_native_oauth_model=normalize_grok_native_oauth_model,
    is_oa_xai_model=is_oa_xai_model,
    resolve_oa_xai_upstream_model=resolve_oa_xai_upstream_model,
    normalize_kimi_model_name=_normalize_kimi_code_chat_completions_adapter_model_name,
    normalize_kimi_custom_tool_outputs=lambda b: _kimi_code_adapters.normalize_kimi_code_custom_tool_outputs(b),
    grok_normalization=_anthropic_grok_normalization,
    grok_normalization_runtime=_get_anthropic_grok_normalization_runtime(),
    request_body_walk_max_depth=_AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
)

# -- Pure functions (same-object identity) --
_get_openai_tool_name = _aawm_codex_tool_policy.get_openai_tool_name
_get_openai_tool_type = _aawm_codex_tool_policy.get_openai_tool_type
_patch_codex_spawn_agent_description_text = _aawm_codex_tool_policy.patch_codex_spawn_agent_description_text
_patch_codex_spawn_agent_payload_parameters = _aawm_codex_tool_policy.patch_codex_spawn_agent_payload_parameters
_load_bundled_model_cost_map_for_codex_policy = _aawm_codex_tool_policy.load_bundled_model_cost_map_for_codex_policy
_adapted_custom_tool_function_schema = _aawm_codex_tool_policy.adapted_custom_tool_function_schema
_request_has_openai_tool_definitions = _aawm_codex_tool_policy.request_has_openai_tool_definitions
_apply_spawn_agent_parameter_patches = _aawm_codex_tool_policy._apply_spawn_agent_parameter_patches
_lookup_model_info_field = _aawm_codex_tool_policy._lookup_model_info_field

# -- Functions binding normalize_tag_value --

def _patch_codex_spawn_agent_tool_description(tool, *, tool_index):
    return _aawm_codex_tool_policy.patch_codex_spawn_agent_tool_description(
        tool, tool_index=tool_index, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _get_codex_core_tool_guidance(tool_name):
    return _aawm_codex_tool_policy.get_codex_core_tool_guidance(
        tool_name, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _append_codex_core_tool_guidance_to_description(description, *, guidance):
    return _aawm_codex_tool_policy.append_codex_core_tool_guidance_to_description(
        description, guidance=guidance,
    )

def _patch_codex_multi_agent_tool_search_description(tool, *, tool_index):
    return _aawm_codex_tool_policy.patch_codex_multi_agent_tool_search_description(
        tool, tool_index=tool_index, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _patch_codex_core_tool_description(tool, *, tool_index):
    return _aawm_codex_tool_policy.patch_codex_core_tool_description(
        tool, tool_index=tool_index, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapt_codex_custom_tool_definitions(tools, *, adapter_names):
    return _aawm_codex_tool_policy.adapt_codex_custom_tool_definitions(
        tools, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapted_custom_tool_call_ids(input_items, *, adapter_names):
    return _aawm_codex_tool_policy.adapted_custom_tool_call_ids(
        input_items, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapt_codex_custom_tool_input_items(input_items, *, adapter_names):
    return _aawm_codex_tool_policy.adapt_codex_custom_tool_input_items(
        input_items, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapt_codex_custom_tool_choice(tool_choice, *, adapter_names):
    return _aawm_codex_tool_policy.adapt_codex_custom_tool_choice(
        tool_choice, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapt_codex_namespace_tool_definitions(tools, *, adapter_names):
    return _aawm_codex_tool_policy.adapt_codex_namespace_tool_definitions(
        tools, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapt_codex_namespace_input_items(input_items, *, adapter_names):
    return _aawm_codex_tool_policy.adapt_codex_namespace_input_items(
        input_items, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _adapt_codex_namespace_tool_choice(tool_choice, *, adapter_names):
    return _aawm_codex_tool_policy.adapt_codex_namespace_tool_choice(
        tool_choice, adapter_names=adapter_names, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

def _openai_tool_choice_references_tool_type(tool_choice, tool_types):
    return _aawm_codex_tool_policy.openai_tool_choice_references_tool_type(
        tool_choice, tool_types, normalize_tag_value=_normalize_low_cardinality_tag_value,
    )

# -- Functions binding CodexToolPolicyCallbacks --
def _get_codex_tool_policy_model_cost_candidates(model):
    return _aawm_codex_tool_policy.get_codex_tool_policy_model_cost_candidates(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _get_unsupported_hosted_tool_types_for_model(model):
    return _aawm_codex_tool_policy.get_unsupported_hosted_tool_types_for_model(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _get_unsupported_request_param_names_for_model(model):
    return _aawm_codex_tool_policy.get_unsupported_request_param_names_for_model(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _get_unsupported_input_item_types_for_model(model):
    return _aawm_codex_tool_policy.get_unsupported_input_item_types_for_model(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _get_rewrite_input_item_types_for_model(model):
    return _aawm_codex_tool_policy.get_rewrite_input_item_types_for_model(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _get_custom_tool_function_adapter_names_for_model(model):
    return _aawm_codex_tool_policy.get_custom_tool_function_adapter_names_for_model(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _get_namespace_tool_function_adapter_names_for_model(model):
    return _aawm_codex_tool_policy.get_namespace_tool_function_adapter_names_for_model(
        model, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_codex_custom_tool_function_adapter_logging_metadata(
    request_body, *, adapted_tools, adapted_input_items, adapted_tool_choice,
):
    return _aawm_codex_tool_policy.add_codex_custom_tool_function_adapter_logging_metadata(
        request_body,
        adapted_tools=adapted_tools,
        adapted_input_items=adapted_input_items,
        adapted_tool_choice=adapted_tool_choice,
        callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _adapt_codex_custom_tools_to_functions_from_request_body(request_body):
    return _aawm_codex_tool_policy.adapt_codex_custom_tools_to_functions_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_codex_namespace_tool_function_adapter_logging_metadata(
    request_body, *, adapted_tools, adapted_input_items, adapted_tool_choice, skipped_tools,
):
    return _aawm_codex_tool_policy.add_codex_namespace_tool_function_adapter_logging_metadata(
        request_body,
        adapted_tools=adapted_tools,
        adapted_input_items=adapted_input_items,
        adapted_tool_choice=adapted_tool_choice,
        skipped_tools=skipped_tools,
        callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _adapt_codex_namespace_tools_to_functions_from_request_body(request_body):
    return _aawm_codex_tool_policy.adapt_codex_namespace_tools_to_functions_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_codex_unsupported_hosted_tool_logging_metadata(
    request_body, *, removed_tools, removed_tool_choice,
):
    return _aawm_codex_tool_policy.add_codex_unsupported_hosted_tool_logging_metadata(
        request_body,
        removed_tools=removed_tools,
        removed_tool_choice=removed_tool_choice,
        callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_tool_choice_without_tools_logging_metadata(request_body, *, removed_tool_choice):
    return _aawm_codex_tool_policy.add_tool_choice_without_tools_logging_metadata(
        request_body, removed_tool_choice=removed_tool_choice, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _drop_tool_choice_without_tools_from_request_body(request_body):
    return _aawm_codex_tool_policy.drop_tool_choice_without_tools_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_codex_unsupported_request_param_logging_metadata(request_body, *, removed_params):
    return _aawm_codex_tool_policy.add_codex_unsupported_request_param_logging_metadata(
        request_body, removed_params=removed_params, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _drop_unsupported_codex_request_params_from_request_body(request_body):
    return _aawm_codex_tool_policy.drop_unsupported_codex_request_params_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_codex_unsupported_input_item_logging_metadata(request_body, *, removed_items):
    return _aawm_codex_tool_policy.add_codex_unsupported_input_item_logging_metadata(
        request_body, removed_items=removed_items, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _drop_unsupported_codex_input_items_from_request_body(request_body):
    return _aawm_codex_tool_policy.drop_unsupported_codex_input_items_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _drop_unsupported_codex_hosted_tools_from_request_body(request_body):
    return _aawm_codex_tool_policy.drop_unsupported_codex_hosted_tools_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_codex_tool_description_patch_logging_metadata(request_body, patch_events):
    return _aawm_codex_tool_policy.add_codex_tool_description_patch_logging_metadata(
        request_body, patch_events, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _apply_codex_tool_description_patches_to_request_body(request_body):
    return _aawm_codex_tool_policy.apply_codex_tool_description_patches_to_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _stringify_grok_native_input_item_value(value):
    return _aawm_codex_tool_policy.stringify_grok_native_input_item_value(
        value, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _format_grok_native_function_call_input_message(item, *, include_correlation_ref=False):
    return _aawm_codex_tool_policy.format_grok_native_function_call_input_message(
        item, include_correlation_ref=include_correlation_ref, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _format_grok_native_function_call_output_input_message(item, *, include_correlation_ref=False):
    return _aawm_codex_tool_policy.format_grok_native_function_call_output_input_message(
        item, include_correlation_ref=include_correlation_ref, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _rewrite_grok_native_input_item_for_model_input(item, *, item_type, include_correlation_ref=False):
    return _aawm_codex_tool_policy.rewrite_grok_native_input_item_for_model_input(
        item, item_type=item_type, include_correlation_ref=include_correlation_ref,
        callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _is_anthropic_grok_native_responses_adapter_body(request_body):
    return _aawm_codex_tool_policy.is_anthropic_grok_native_responses_adapter_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _add_grok_native_input_item_rewrite_logging_metadata(request_body, *, rewritten_items):
    return _aawm_codex_tool_policy.add_grok_native_input_item_rewrite_logging_metadata(
        request_body, rewritten_items=rewritten_items, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _rewrite_grok_native_unsupported_input_items_from_request_body(request_body):
    return _aawm_codex_tool_policy.rewrite_grok_native_unsupported_input_items_from_request_body(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )

def _rewrite_grok_native_unsupported_input_items_in_place(request_body):
    return _aawm_codex_tool_policy.rewrite_grok_native_unsupported_input_items_in_place(
        request_body, callbacks=_CODEX_TOOL_POLICY_CALLBACKS,
    )


# ---------------------------------------------------------------------------
# Wave 6E claude-prompt-replacement same-object facades
# ---------------------------------------------------------------------------
_parse_claude_code_version = _aawm_claude_prompt_replacement._parse_claude_code_version
_resolve_claude_auto_memory_template_path = _aawm_claude_prompt_replacement._resolve_claude_auto_memory_template_path
_load_claude_context_replacement_template = _aawm_claude_prompt_replacement._load_claude_context_replacement_template
_load_claude_prompt_patch_manifest = _aawm_claude_prompt_replacement._load_claude_prompt_patch_manifest
_extract_markdown_section = _aawm_claude_prompt_replacement._extract_markdown_section
_render_claude_auto_memory_replacement = _aawm_claude_prompt_replacement._render_claude_auto_memory_replacement
_replace_claude_auto_memory_section_in_text = _aawm_claude_prompt_replacement._replace_claude_auto_memory_section_in_text
_replace_claude_system_prompt_override_in_value = _aawm_claude_prompt_replacement._replace_claude_system_prompt_override_in_value
_add_claude_system_prompt_override_logging_metadata = _aawm_claude_prompt_replacement._add_claude_system_prompt_override_logging_metadata
_replace_claude_system_prompt_in_anthropic_request_body = _aawm_claude_prompt_replacement._replace_claude_system_prompt_in_anthropic_request_body
_apply_claude_prompt_patches_in_text = _aawm_claude_prompt_replacement._apply_claude_prompt_patches_in_text
_replace_claude_prompt_patches_in_value = _aawm_claude_prompt_replacement._replace_claude_prompt_patches_in_value
_add_claude_prompt_patch_logging_metadata = _aawm_claude_prompt_replacement._add_claude_prompt_patch_logging_metadata
_apply_claude_prompt_patches_to_anthropic_request_body = _aawm_claude_prompt_replacement._apply_claude_prompt_patches_to_anthropic_request_body


# ---------------------------------------------------------------------------
# Wave 6E anthropic-body-prep same-object facades
# ---------------------------------------------------------------------------
_get_openai_adapter_claude_context_char_cap = _aawm_anthropic_body_prep._get_openai_adapter_claude_context_char_cap
_detect_openai_adapter_claude_context_markers = _aawm_anthropic_body_prep._detect_openai_adapter_claude_context_markers
_select_openai_adapter_context_summary_lines = _aawm_anthropic_body_prep._select_openai_adapter_context_summary_lines
_build_openai_adapter_compacted_claude_context_block = _aawm_anthropic_body_prep._build_openai_adapter_compacted_claude_context_block
_compact_openai_adapter_claude_context_text = _aawm_anthropic_body_prep._compact_openai_adapter_claude_context_text
_compact_openai_adapter_claude_context_value = _aawm_anthropic_body_prep._compact_openai_adapter_claude_context_value
_add_openai_adapter_claude_context_compaction_logging_metadata = _aawm_anthropic_body_prep._add_openai_adapter_claude_context_compaction_logging_metadata
_compact_openai_adapter_claude_context_in_anthropic_request_body = _aawm_anthropic_body_prep._compact_openai_adapter_claude_context_in_anthropic_request_body
_validate_anthropic_tool_blocks_for_passthrough = _aawm_anthropic_body_prep._validate_anthropic_tool_blocks_for_passthrough
_repair_anthropic_tool_use_ids_for_passthrough = _aawm_anthropic_body_prep._repair_anthropic_tool_use_ids_for_passthrough
_prepare_anthropic_request_body_for_passthrough = _aawm_anthropic_body_prep._prepare_anthropic_request_body_for_passthrough


# ---------------------------------------------------------------------------
# Wave 6E anthropic-body-prep runtime configuration
# ---------------------------------------------------------------------------
_aawm_anthropic_body_prep.configure_anthropic_body_prep_runtime(
    expand_persisted_output=_expand_claude_persisted_output_in_anthropic_request_body,
    extract_billing_header_fields=_extract_anthropic_billing_header_fields_from_request_body,
    apply_control_plane_rewrites=_aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body,
    expand_dynamic_directives=_aawm_expand_aawm_dynamic_directives_in_anthropic_request_body,
    add_post_rewrite_context_file_metadata=_aawm_add_claude_post_rewrite_context_file_logging_metadata,
    sanitize_web_search_domain_lists=_sanitize_anthropic_web_search_empty_domain_lists,
    add_billing_header_logging_metadata=_add_anthropic_billing_header_logging_metadata,
    add_route_family_logging_metadata=_add_route_family_logging_metadata,
    add_request_breakout_logging_metadata=_add_claude_request_breakout_logging_metadata,
    prepare_observability=_prepare_request_body_for_passthrough_observability,
    get_tenant_header=_get_aawm_tenant_header,
)


# Wave 5B: backward-compat module __getattr__ for manager-owned quota cache.
# Tests and external code may still reference the old module-level name.
_W5B_MANAGER_OWNED_ATTRS = {
    "_openrouter_free_daily_quota_cache": lambda: _alias_routing_state.get_openrouter_free_quota_cache(),
}


def __getattr__(name: str):
    factory = _W5B_MANAGER_OWNED_ATTRS.get(name)
    if factory is not None:
        return factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
