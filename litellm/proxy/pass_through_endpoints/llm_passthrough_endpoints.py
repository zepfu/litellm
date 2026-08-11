"""
What is this?

Provider-specific Pass-Through Endpoints

Use litellm with Anthropic SDK, Vertex AI SDK, Cohere SDK, etc.
"""

import asyncio  # noqa: F401 - live compatibility binding for provider runtimes
import codecs  # noqa: F401 - compatibility binding for extracted Wave 6A facades
import copy
import hashlib  # noqa: F401 - live host global for installed lane-key functions
import importlib
import json
import os
import posixpath
import random  # noqa: F401 - compatibility binding for extracted Wave 5C facades
import re
import time
from datetime import datetime, timedelta, timezone
from inspect import isawaitable  # noqa: F401 - Wave 6A facade host binding
from pathlib import Path
from functools import lru_cache, partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    Security,
    WebSocket,
    status as fastapi_status,
)
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState
from typing_extensions import TypeGuard  # noqa: F401 - Wave 6F facade host binding

globals()["status"] = fastapi_status

import litellm
from litellm import get_llm_provider
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    ALLOWED_VERTEX_AI_PASSTHROUGH_HEADERS,
    BEDROCK_AGENT_RUNTIME_PASS_THROUGH_ROUTES,
    XAI_API_BASE,
)
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,  # noqa: F401 - Wave 6A host binding
    is_malformed_grok_literal_tool_label_transcript_text,  # noqa: F401 - Wave 6A host binding
)
from litellm.integrations.aawm_agent_identity.constants import (
    _AAWM_LITELLM_ENVIRONMENT_ENV_VARS,
)
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_shape,  # noqa: F401 - pending shape-capture install wiring
)
from litellm.proxy.aawm_runtime_error_logging import (
    schedule_persist_malformed_tool_call_detection,  # noqa: F401 - Wave 6A host binding
)
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_API_BASE,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,  # noqa: F401 - Wave 6F facade host binding
)
from litellm.llms.xai.oauth import (
    is_oa_xai_model,
    get_grok_native_oauth_access_token,  # noqa: F401  # compatibility host-global
    normalize_grok_native_oauth_model,
    resolve_oa_xai_upstream_model,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._types import *
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
    _classify_passthrough_hidden_retry_failure,  # noqa: F401 - pending retry-classifier install wiring
    _get_passthrough_handled_http_error_summary,
    _get_passthrough_hidden_retry_wait_seconds,  # noqa: F401  # consumed by rebound env_policy functions
    _is_known_grok_build_usage_balance_exhausted_response,
    _is_known_grok_personal_team_spending_limit_response,
    _record_passthrough_hidden_retry_metadata,  # noqa: F401 - pending retry-metadata install wiring
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    adapter as _anthropic_grok_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    composer_repair as _anthropic_grok_composer_repair,  # noqa: F401 - Wave 6A host binding
)
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    normalization as _anthropic_grok_normalization,  # noqa: F401 - runtime globals() binding
)
from litellm.llms.anthropic.experimental_pass_through.providers import (
    common as _anthropic_provider_common,
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
    normalization as _anthropic_opencode_zen_normalization,  # noqa: F401 - Wave 6F facade host binding
)
from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
    adapter as _anthropic_openrouter_provider,
)
from litellm.llms.anthropic.experimental_pass_through.providers.xai import (
    adapter as _anthropic_xai_provider,
)
from litellm.proxy.aawm_route_logging import (
    aresolve_aawm_route_host_attribution,  # noqa: F401 - rollup install host binding
    build_aawm_route_rollup_group_header_label,  # noqa: F401 - rollup install host binding
    emit_aawm_route_access_log,  # noqa: F401 - Wave 6F facade host binding
    emit_aawm_route_status_event,  # noqa: F401 - rollup install host binding
    record_aawm_route_rollup,  # noqa: F401 - rollup install host binding
    record_aawm_route_rollup_turn,  # noqa: F401 - Wave 6F facade host binding
    resolve_aawm_route_host_attribution,  # noqa: F401 - rollup install host binding
)
from litellm.proxy.pass_through_endpoints import (
    aawm_context_query as _aawm_context_query,
)

_AAWM_CLAUDE_CONTROL_PLANE_MODULE = (
    "litellm.proxy.pass_through_endpoints.aawm_claude_control_plane"
)
_AAWM_CLAUDE_PROMPT_REPLACEMENT_MODULE = (
    "litellm.proxy.pass_through_endpoints.aawm_request_policy."
    "claude_prompt_replacement"
)
_AAWM_CLAUDE_CONTROL_PLANE_UNAVAILABLE = (
    "AAWM Claude control plane is unavailable"
)
_aawm_claude_control_plane_initialization_status: dict[str, Any] = {
    "state": "not_initialized",
    "mode": "unavailable",
    "ready": False,
    "reason": "not_initialized",
    "error_type": None,
}


def get_aawm_claude_control_plane_initialization_status() -> dict[str, Any]:
    return dict(_aawm_claude_control_plane_initialization_status)


def is_aawm_claude_control_plane_ready() -> bool:
    return bool(_aawm_claude_control_plane_initialization_status.get("ready"))


def _aawm_claude_control_plane_degraded_add_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    return request_body


async def _aawm_claude_control_plane_degraded_apply_rewrites(
    request_body: dict[str, Any],
    billing_header_fields: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return request_body, [], []


async def _aawm_claude_control_plane_degraded_expand_context(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return request_body, []


def _raise_aawm_claude_control_plane_unavailable() -> None:
    raise RuntimeError(_AAWM_CLAUDE_CONTROL_PLANE_UNAVAILABLE)


def _aawm_claude_control_plane_failed_add_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    _raise_aawm_claude_control_plane_unavailable()


async def _aawm_claude_control_plane_failed_apply_rewrites(
    request_body: dict[str, Any],
    billing_header_fields: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _raise_aawm_claude_control_plane_unavailable()


async def _aawm_claude_control_plane_failed_expand_context(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _raise_aawm_claude_control_plane_unavailable()


_aawm_add_claude_post_rewrite_context_file_logging_metadata = (
    _aawm_claude_control_plane_failed_add_metadata
)
_aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body = (
    _aawm_claude_control_plane_failed_apply_rewrites
)
_aawm_expand_aawm_dynamic_directives_in_anthropic_request_body = (
    _aawm_claude_control_plane_failed_expand_context
)

# Provider-neutral ownership with historical host compatibility exports.
_build_aawm_dynamic_injection_dsn = (
    _aawm_context_query._build_aawm_dynamic_injection_dsn
)
_call_aawm_get_agent_memories = (
    _aawm_context_query._call_aawm_get_agent_memories
)
_get_aawm_dynamic_injection_application_name = (
    _aawm_context_query._get_aawm_dynamic_injection_application_name
)
_get_aawm_dynamic_injection_pool = (
    _aawm_context_query._get_aawm_dynamic_injection_pool
)
_get_aawm_dynamic_injection_server_settings = (
    _aawm_context_query._get_aawm_dynamic_injection_server_settings
)
_initialize_aawm_dynamic_injection_connection = (
    _aawm_context_query._initialize_aawm_dynamic_injection_connection
)
close_aawm_dynamic_injection_pool = (
    _aawm_context_query.close_aawm_dynamic_injection_pool
)
_get_aawm_callback_pool = _aawm_context_query._get_aawm_callback_pool


def _set_aawm_claude_control_plane_initialization_status(
    *,
    state: str,
    mode: str,
    ready: bool,
    reason: Optional[str],
    error_type: Optional[str] = None,
) -> None:
    global _aawm_claude_control_plane_initialization_status
    _aawm_claude_control_plane_initialization_status = {
        "state": state,
        "mode": mode,
        "ready": ready,
        "reason": reason,
        "error_type": error_type,
    }


def _initialize_aawm_claude_control_plane(
    import_module: Optional[Callable[[str], Any]] = None,
) -> None:
    global _aawm_add_claude_post_rewrite_context_file_logging_metadata
    global _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body
    global _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body

    importer = import_module or importlib.import_module
    try:
        control_plane = importer(_AAWM_CLAUDE_CONTROL_PLANE_MODULE)
        prompt_replacement = importer(_AAWM_CLAUDE_PROMPT_REPLACEMENT_MODULE)
        context_services = _aawm_context_query.build_context_query_services(
            get_agent_memories=lambda **kwargs: (
                _call_aawm_get_agent_memories(**kwargs)
            ),
            get_context=lambda **kwargs: control_plane._call_aawm_context_grab(
                **kwargs
            ),
            get_reference_identifiers=lambda **kwargs: (
                control_plane._call_aawm_reference_identifier_list(**kwargs)
            ),
        )
        services = control_plane.build_claude_control_plane_services(
            prompt=prompt_replacement.build_claude_prompt_replacement_services(),
            context_query=context_services,
            now_utc=lambda: datetime.now(timezone.utc),
            merge_metadata=_merge_litellm_metadata,
            build_span=_build_langfuse_span_descriptor,
            format_span_timestamp=_format_langfuse_span_timestamp,
            add_context_file_metadata=(
                _aawm_observability_metadata._add_claude_post_rewrite_context_file_logging_metadata
            ),
        )
        rewriter = control_plane.compose_claude_control_plane(services)
    except ModuleNotFoundError as exc:
        if exc.name == _AAWM_CLAUDE_CONTROL_PLANE_MODULE:
            _aawm_add_claude_post_rewrite_context_file_logging_metadata = (
                _aawm_claude_control_plane_degraded_add_metadata
            )
            _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body = (
                _aawm_claude_control_plane_degraded_apply_rewrites
            )
            _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body = (
                _aawm_claude_control_plane_degraded_expand_context
            )
            _set_aawm_claude_control_plane_initialization_status(
                state="degraded",
                mode="optional",
                ready=True,
                reason="optional_module_absent",
            )
            return
        _aawm_add_claude_post_rewrite_context_file_logging_metadata = (
            _aawm_claude_control_plane_failed_add_metadata
        )
        _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body = (
            _aawm_claude_control_plane_failed_apply_rewrites
        )
        _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body = (
            _aawm_claude_control_plane_failed_expand_context
        )
        _set_aawm_claude_control_plane_initialization_status(
            state="failed",
            mode="unavailable",
            ready=False,
            reason="initialization_failed",
            error_type=exc.__class__.__name__,
        )
        return
    except Exception as exc:
        _aawm_add_claude_post_rewrite_context_file_logging_metadata = (
            _aawm_claude_control_plane_failed_add_metadata
        )
        _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body = (
            _aawm_claude_control_plane_failed_apply_rewrites
        )
        _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body = (
            _aawm_claude_control_plane_failed_expand_context
        )
        _set_aawm_claude_control_plane_initialization_status(
            state="failed",
            mode="unavailable",
            ready=False,
            reason="initialization_failed",
            error_type=exc.__class__.__name__,
        )
        return

    _aawm_add_claude_post_rewrite_context_file_logging_metadata = (
        rewriter.add_post_rewrite_context_file_metadata
    )
    _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body = (
        rewriter.apply_rewrites
    )
    _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body = (
        rewriter.expand_dynamic_context
    )
    _set_aawm_claude_control_plane_initialization_status(
        state="active",
        mode="enabled",
        ready=True,
        reason=None,
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
    ResponsesAPIOptionalRequestParams,  # noqa: F401 - Wave 6F facade host binding
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_aawm_context_query.configure_context_query_runtime(
    _aawm_context_query.ContextQueryRuntime(get_secret_str=get_secret_str)
)

from .passthrough_endpoint_router import passthrough_endpoint_router

from .aawm_alias_routing import adapter_config as _aawm_adapter_config  # noqa: F401 - Wave 6F facade host binding
from .aawm_alias_routing import adapter_driver as _aawm_adapter_driver
from .aawm_alias_routing import classification as _aawm_alias_classification
from .aawm_alias_routing import memory as _aawm_alias_memory
from .aawm_alias_routing import provider_shaping as _aawm_provider_shaping  # noqa: F401 - runtime globals() binding
from .aawm_alias_routing import responses_finalize as _aawm_responses_finalize
from .aawm_alias_routing import retry as _aawm_alias_retry  # noqa: F401
from .aawm_alias_routing import streaming as _aawm_alias_streaming  # noqa: F401 - Wave 6F facade host binding
from .aawm_alias_routing import candidate_loop as _aawm_alias_candidate_loop
from .aawm_alias_routing import durable as _aawm_alias_durable

# Wave 4 pure-leaf extraction imports
from litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen import constants as _opencode_zen_constants
from .aawm_alias_routing import lane_keys as _aawm_lane_keys
from .aawm_alias_routing import request_metadata as _aawm_request_metadata
from .aawm_alias_routing import runtime_memory as _aawm_runtime_memory
from .aawm_adapter_runtime import alias_candidate_dispatch as _aawm_alias_candidate_dispatch
from .aawm_adapter_runtime import anthropic_auto_agent_route as _aawm_anthropic_auto_agent_route
from .aawm_adapter_runtime import codex_auto_agent_route as _aawm_codex_auto_agent_route
from .aawm_adapter_runtime import model_resolution as _aawm_adapter_model_resolution
from . import aawm_adapter_runtime as _aawm_adapter_runtime
from .aawm_adapter_runtime import anthropic_adapter_calls as _aawm_anthropic_adapter_calls  # noqa: F401 - Wave 6F integration import
from .aawm_adapter_runtime import anthropic_dispatch as _aawm_anthropic_dispatch
from .aawm_adapter_runtime import codex_candidate_calls as _aawm_codex_candidate_calls  # noqa: F401 - Wave 6F integration import
from .aawm_adapter_runtime import codex_dispatch as _aawm_codex_dispatch  # noqa: F401 - Wave 6F integration import
from .aawm_request_policy import alias_guidance as _aawm_alias_guidance
from .aawm_request_policy import observability_metadata as _aawm_observability_metadata  # noqa: F401 - install(globals()) pending wiring
from .aawm_request_policy import persisted_output as _aawm_persisted_output  # noqa: F401 - install(globals()) pending wiring
from .aawm_request_policy import codex_tool_policy as _aawm_codex_tool_policy
from .aawm_request_policy import anthropic_body_prep as _aawm_anthropic_body_prep
from litellm.llms.anthropic.experimental_pass_through.providers.grok import side_channel as _grok_side_channel

_prepare_anthropic_request_body_for_passthrough = (
    _aawm_anthropic_body_prep._prepare_anthropic_request_body_for_passthrough
)

# Wave 6B extracted provider modules
from litellm.proxy.pass_through_endpoints.providers import common as _wave6b_common
from litellm.proxy.pass_through_endpoints.providers.openrouter import runtime as _wave6b_openrouter_runtime
from litellm.proxy.pass_through_endpoints.providers.nvidia import runtime as _wave6b_nvidia_runtime
from litellm.proxy.pass_through_endpoints.providers.opencode_zen import runtime as _wave6b_opencode_zen_runtime  # noqa: F401 - install(globals()) pending wiring
from litellm.proxy.pass_through_endpoints.providers.xai import request_prep as _wave6b_xai_request_prep
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


# Process-local maps/locks owned by aawm_alias_routing.state.
_codex_auto_agent_cooldown_until_monotonic_by_key = _alias_routing_state.codex.cooldown_until_monotonic_by_key
_codex_auto_agent_session_affinity_by_key = _alias_routing_state.codex.session_affinity_by_key
_codex_auto_agent_lock = _alias_routing_state.codex.lock
_anthropic_auto_agent_cooldown_until_monotonic_by_key = _alias_routing_state.anthropic.cooldown_until_monotonic_by_key
_anthropic_auto_agent_session_affinity_by_key = _alias_routing_state.anthropic.session_affinity_by_key
_anthropic_auto_agent_lock = _alias_routing_state.anthropic.lock
_codex_auto_agent_lane_state_cache_lock = _alias_routing_state.lane_state_cache_lock
_openrouter_adapter_rate_limit_until_monotonic_by_key = (
    _alias_routing_state.openrouter_rate_limit.until_monotonic_by_key
)
_openrouter_adapter_failure_circuit_until_monotonic_by_key = (
    _alias_routing_state.openrouter_failure_circuit.until_monotonic_by_key
)
_codex_auto_agent_cooldown_negative_until_monotonic_by_key = (
    _alias_routing_state.codex.cooldown_negative_until_monotonic_by_key
)
_anthropic_auto_agent_cooldown_negative_until_monotonic_by_key = (
    _alias_routing_state.anthropic.cooldown_negative_until_monotonic_by_key
)
_aawm_alias_routing_log_until_monotonic_by_key = _alias_routing_state.log_until_monotonic_by_key

_aawm_runtime_memory.configure_runtime_memory(
    runtime=_aawm_runtime_memory.RuntimeMemoryRuntime(
        log_until_map=_aawm_alias_routing_log_until_monotonic_by_key,
        max_size=_AAWM_ALIAS_ROUTING_MEMORY_STATE_MAX_SIZE,
    )
)
_should_log_aawm_alias_routing_event = (
    _aawm_runtime_memory.should_log_aawm_alias_routing_event
)
_replace_request_body_in_place = _aawm_runtime_memory.replace_request_body_in_place
_bound_aawm_alias_routing_memory_map = (
    _aawm_runtime_memory.bound_aawm_alias_routing_memory_map
)
_hydrate_aawm_alias_routing_cooldown_memory = (
    _aawm_runtime_memory.hydrate_aawm_alias_routing_cooldown_memory
)
_hydrate_aawm_alias_routing_affinity_memory = (
    _aawm_runtime_memory.hydrate_aawm_alias_routing_affinity_memory
)

# Retained generic provider/lane/cooldown/allowlist policy aliases.
from .aawm_alias_routing import policy as _aawm_alias_policy_compat
_aawm_alias_policy_compat.install_policy_compat_aliases(globals())
































































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
get_active_routing_snapshot = _aawm_snapshot_select.get_active_routing_snapshot
set_active_routing_snapshot = _aawm_snapshot_select.set_active_routing_snapshot
_lookup_active_snapshot_canonical_alias = (
    _aawm_snapshot_select._lookup_active_snapshot_canonical_alias
)
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
_select_snapshot_candidates = _aawm_snapshot_select._select_snapshot_candidates
_derive_round_robin_commit_token = _aawm_snapshot_select._derive_round_robin_commit_token
_get_aawm_alias_selection_context = _aawm_snapshot_select._get_aawm_alias_selection_context
_resolve_aawm_alias_selection_enumeration = _aawm_snapshot_select._resolve_aawm_alias_selection_enumeration

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
_bind_codex_oauth_candidate_to_request = (
    _aawm_codex_oauth._bind_codex_oauth_candidate_to_request
)
_get_bound_codex_oauth_candidate_identity = (
    _aawm_codex_oauth._get_bound_codex_oauth_candidate_identity
)
_load_bound_codex_oauth_auth = _aawm_codex_oauth._load_bound_codex_oauth_auth
_codex_oauth_responses_target_url = (
    _aawm_codex_oauth._codex_oauth_responses_target_url
)
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

# Wave 5B: failure evidence, cursor, and quota cache are manager-owned.
_codex_failure_evidence_gate = _alias_routing_state.codex_failure_evidence_gate
_round_robin_cursor_by_alias = _alias_routing_state.round_robin_cursor

# Bind the round-robin cursor into snapshot selection.
_aawm_snapshot_select.configure_snapshot_runtime(
    round_robin_cursor=_round_robin_cursor_by_alias,
)


def reset_module_singletons() -> None:
    """Clear legacy god-module singleton state (test-support).

    Clear the manager-owned Codex failure-evidence gates and round-robin cursor
    plus the active routing snapshot.
    """
    _codex_failure_evidence_gate.clear_for_tests()
    _round_robin_cursor_by_alias.clear()
    set_active_routing_snapshot(None)


def reset_alias_routing_state_for_tests() -> None:
    """Reset ALL process-local alias-routing state (test-support only).

    Clears manager-owned state via ``alias_routing_state.reset_for_tests()``
    and the active snapshot.
    """
    _alias_routing_state.reset_for_tests()
    set_active_routing_snapshot(None)


# RR-054 durable alias-routing helpers (package-owned).
_get_aawm_alias_routing_state_namespace = _aawm_alias_durable.get_aawm_alias_routing_state_namespace
_build_aawm_alias_routing_durable_cache_key = _aawm_alias_durable.build_aawm_alias_routing_durable_cache_key
_get_aawm_alias_routing_dual_cache = _aawm_alias_durable.get_aawm_alias_routing_dual_cache
_parse_aawm_alias_routing_durable_expiry = _aawm_alias_durable.parse_aawm_alias_routing_durable_expiry
_read_aawm_alias_routing_durable_payload = _aawm_alias_durable.read_aawm_alias_routing_durable_payload
_write_aawm_alias_routing_durable_payload = _aawm_alias_durable.write_aawm_alias_routing_durable_payload
_AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX = _aawm_alias_durable.AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX
_AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT = _aawm_alias_durable.AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT
_is_aawm_alias_routing_retryable_redis_error = _aawm_alias_durable.is_retryable_redis_error
_get_aawm_alias_routing_durable_write_retry_backoff_seconds = _aawm_alias_durable.get_durable_write_retry_backoff_seconds

# --- restored missing constants from HEAD (ordered) ---

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

# Singleton imported from .passthrough_endpoint_router (line 205)


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


# CFG-004: cooldown clear endpoint (thin authenticated route delegate).
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_clear import (
    handle_cooldown_clear as _aawm_cooldown_clear_handler,
)
from litellm.proxy.auth.user_api_key_auth import (
    api_key_header as _cfg004_api_key_header,
)


async def _cfg004_cooldown_clear_auth_dependency(
    request: Request,
    api_key: str = Security(_cfg004_api_key_header),
) -> UserAPIKeyAuth:
    """Finding 4: route-specific auth dependency with CFG-004 audit.

    Invokes the existing ``user_api_key_auth`` implementation.  On failure,
    emits exactly one minimal sanitized CFG-004 audit event (no headers,
    token, body, or target strings), then re-raises the same fail-closed
    HTTP error.  On success, returns UserAPIKeyAuth so the handler can
    still enforce PROXY_ADMIN + exact master-key.
    """
    import logging as _logging

    _audit_logger = _logging.getLogger("LiteLLMProxy")
    try:
        result = await user_api_key_auth(request=request, api_key=api_key)
        return result
    except Exception:
        # Emit exactly one minimal sanitized audit event.
        try:
            from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
                get_aawm_alias_routing_state_namespace as _get_ns,
            )
            _ns = _get_ns()
        except Exception:
            _ns = "unavailable"
        import os as _os
        _env = (
            _os.getenv("AAWM_LITELLM_ENVIRONMENT", "").strip()
            or _os.getenv("LITELLM_ENVIRONMENT", "").strip()
            or "unknown"
        )
        _audit_logger.info(
            "aawm_cooldown_clear_audit %s",
            {
                "event": "aawm_cooldown_clear_auth_failure",
                "target_description": "target_unavailable",
                "family": "",
                "ingress": "unavailable",
                "candidates": [],
                "result": "error",
                "error_code": "auth_dependency_failure",
                "prior_state_source": "",
                "bounded_remaining_ttl_seconds": 0.0,
                "environment": _env,
                "namespace": _ns,
            },
        )
        raise


@router.post(
    "/aawm/alias-routing/cooldowns/clear",
    tags=["AAWM Alias Routing"],
)
async def _aawm_alias_routing_cooldown_clear_endpoint(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(_cfg004_cooldown_clear_auth_dependency),
) -> dict[str, Any]:
    """Clear alias-routing cooldown state for a target (CFG-004).

    Requires PROXY_ADMIN role and master-key authentication.
    Manipulates state only; never sends provider traffic.
    Finding 4: auth dependency emits audit on failure; handler does not
    duplicate dependency audits.
    """
    return await _aawm_cooldown_clear_handler(request, user_api_key_dict)


@router.post(
    "/aawm/alias-routing/cooldowns/acceptance",
    tags=["AAWM Alias Routing"],
)
async def _aawm_alias_routing_cooldown_acceptance_endpoint(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(_cfg004_cooldown_clear_auth_dependency),
) -> dict[str, Any]:
    """Dev-only acceptance harness for CFG-004 criterion 11.

    Operations: prepare, inspect, restore.
    Gated on exact environment, enable flag, run_id, namespace, topology,
    PROXY_ADMIN role, and master-key authentication.
    """
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import (
        handle_cooldown_acceptance,
    )

    return await handle_cooldown_acceptance(request, user_api_key_dict)



# CFG-004 criterion 11: fail-closed auth bypass for the disposable
# /openai_passthrough Responses acceptance route.  Bypasses LiteLLM key
# auth ONLY when the complete disposable acceptance runtime contract is
# satisfied; otherwise delegates to user_api_key_auth unchanged.
_CFG004_ACCEPTANCE_PUBLIC_ROUTE = "/openai_passthrough/*"


def _cfg004_acceptance_auth_bypass_active(request: Request, endpoint: str) -> bool:
    """Return True only when ALL CFG-004 acceptance gates pass for the
    /openai_passthrough Responses route.  Fail closed on any ambiguity."""
    import re as _re

    # Gate: exact /openai_passthrough route family (not /openai).
    request_path = getattr(getattr(request, "url", None), "path", "") or ""
    if not request_path.startswith("/openai_passthrough/"):
        return False

    # Gate: Responses endpoint only.
    if not _is_openai_responses_endpoint(endpoint):
        return False

    # Gate: litellm-dev environment.
    if os.getenv("AAWM_LITELLM_ENVIRONMENT", "").strip() != "litellm-dev":
        return False

    # Gate: AAWM_CFG004_ACCEPTANCE_ENABLED=1.
    if os.getenv("AAWM_CFG004_ACCEPTANCE_ENABLED", "").strip() != "1":
        return False

    # Gate: single-worker topology (complete-runtime contract).
    if os.getenv("AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER", "").strip() != "1":
        return False

    # Gate: valid 32-hex run_id.
    run_id = os.getenv("AAWM_CFG004_ACCEPTANCE_RUN_ID", "").strip()
    if not _re.fullmatch(r"[0-9a-f]{32}", run_id):
        return False

    # Gate: matching namespace.
    try:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.durable import (
            get_aawm_alias_routing_state_namespace as _get_ns,
        )
        if _get_ns() != f"aawm-routing-dev-cfg004-{run_id}":
            return False
    except Exception:
        return False

    # Gate: disposable config's exact public-route declaration.
    try:
        from litellm.proxy.proxy_server import general_settings as _gs
        if not isinstance(_gs, dict):
            return False
        _pr = _gs.get("public_routes")
        if not isinstance(_pr, list) or _CFG004_ACCEPTANCE_PUBLIC_ROUTE not in _pr:
            return False
    except Exception:
        return False

    return True


async def _cfg004_openai_passthrough_auth_dependency(
    request: Request,
    endpoint: str,
    api_key: str = Security(_cfg004_api_key_header),
) -> UserAPIKeyAuth:
    """CFG-004 acceptance: bypass LiteLLM key auth under complete disposable
    gates; otherwise delegate to user_api_key_auth unchanged."""
    if _cfg004_acceptance_auth_bypass_active(request, endpoint):
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
    return await user_api_key_auth(request=request, api_key=api_key)



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
@lru_cache(maxsize=1)
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


def _request_has_openai_client_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    return bool(
        headers.get("authorization") or headers.get("Authorization") or headers.get("api-key") or headers.get("Api-Key")
    )


def _join_gemini_base_and_endpoint_path(base_url: httpx.URL, endpoint_path: str) -> str:
    """
    Combine the path component of ``base_url`` with ``endpoint_path``.

    Preserves any path prefix configured on the base URL and resolves
    ``..`` segments in the endpoint so the result stays within the base
    path. A trailing slash on ``endpoint_path`` is preserved.
    """
    trailing_slash = endpoint_path.endswith("/")
    base_path = base_url.path or ""
    if not base_path or base_path == "/":
        normalized_endpoint = posixpath.normpath("/" + endpoint_path.lstrip("/"))
        if trailing_slash and normalized_endpoint != "/":
            normalized_endpoint += "/"
        return normalized_endpoint

    base_path = base_path.rstrip("/")
    clean_endpoint = endpoint_path.lstrip("/")
    combined = posixpath.normpath(base_path + "/" + clean_endpoint)
    # If normalization climbs out of the base path, fall back to base.
    if combined != base_path and not combined.startswith(base_path + "/"):
        return base_path + "/"
    if trailing_slash and not combined.endswith("/"):
        combined += "/"
    return combined


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


_resolve_codex_opencode_zen_adapter_model = _aawm_adapter_model_resolution._resolve_codex_opencode_zen_adapter_model


_resolve_codex_kimi_chat_completions_adapter_model = _aawm_adapter_model_resolution._resolve_codex_kimi_chat_completions_adapter_model


_resolve_codex_alibaba_token_plan_adapter_model = _aawm_adapter_model_resolution._resolve_codex_alibaba_token_plan_adapter_model


_resolve_anthropic_opencode_zen_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_opencode_zen_adapter_model


_resolve_anthropic_kimi_chat_completions_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_kimi_chat_completions_adapter_model


_resolve_anthropic_alibaba_token_plan_adapter_model = _aawm_adapter_model_resolution._resolve_anthropic_alibaba_token_plan_adapter_model


_resolve_codex_auto_agent_alias_model = _aawm_adapter_model_resolution._resolve_codex_auto_agent_alias_model


_get_codex_auto_agent_header = _aawm_lane_keys._get_codex_auto_agent_header


_hash_codex_auto_agent_lane_value = _aawm_lane_keys._hash_codex_auto_agent_lane_value


_extract_passthrough_session_id = (
    _aawm_observability_metadata._extract_passthrough_session_id
)
_aawm_request_metadata.configure_request_metadata_runtime(
    extract_passthrough_session_id=_extract_passthrough_session_id,
    get_codex_auto_agent_header=_get_codex_auto_agent_header,
)
_aawm_request_metadata.install(globals())


_resolve_codex_auto_agent_openai_lane_key = _aawm_lane_keys._resolve_codex_auto_agent_openai_lane_key


_resolve_codex_auto_agent_openai_cooldown_lane_key = _aawm_lane_keys._resolve_codex_auto_agent_openai_cooldown_lane_key


_get_codex_auto_agent_lane_state_cache_ttl_seconds = _aawm_lane_keys._get_codex_auto_agent_lane_state_cache_ttl_seconds


def _invalidate_codex_auto_agent_lane_state_caches() -> None:
    pass


def _resolve_codex_auto_agent_session_key(
    request: Request,
    request_body: dict[str, Any],
    *,
    alias_model: str,
) -> Optional[str]:
    if not alias_model or alias_model.strip() != alias_model:
        raise ValueError("alias_model must be an explicit canonical alias")
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


_aawm_audit_context.configure_audit_context_runtime(
    clean_secret_string=lambda *a, **kw: _clean_secret_string(*a, **kw),
    extract_metadata_value=lambda *a, **kw: _extract_auto_agent_alias_metadata_value(
        *a, **kw
    ),
    extract_client_product_label=lambda *a, **kw: (
        _extract_auto_agent_alias_client_product_label(*a, **kw)
    ),
    resolve_host_attribution=lambda *a, **kw: (
        _resolve_auto_agent_alias_route_host_attribution(*a, **kw)
    ),
    extract_session_id=lambda *a, **kw: _extract_auto_agent_alias_session_id(
        *a, **kw
    ),
    build_rollup_group_header_label=lambda *a, **kw: (
        _build_auto_agent_alias_rollup_group_header_label(*a, **kw)
    ),
    has_continuation_state=lambda *a, **kw: (
        _codex_auto_agent_request_has_continuation_state(*a, **kw)
    ),
)
_aawm_audit_context.install(globals())
uuid4 = _aawm_audit_context.uuid4

_aawm_audit_build.configure_audit_build_runtime(
    get_request_context=lambda *a, **kw: _get_auto_agent_alias_request_context(
        *a, **kw
    ),
    attach_terminal_context_fields=lambda *a, **kw: (
        _attach_auto_agent_alias_terminal_context_fields(*a, **kw)
    ),
    format_timestamp=_format_auto_agent_alias_timestamp,
    extract_metadata_value=lambda *a, **kw: _extract_auto_agent_alias_metadata_value(
        *a, **kw
    ),
    extract_incoming_endpoint=lambda *a, **kw: (
        _extract_auto_agent_alias_incoming_endpoint(*a, **kw)
    ),
    resolve_outgoing_target=lambda *a, **kw: (
        _resolve_auto_agent_alias_route_rollup_outgoing_target(*a, **kw)
    ),
    to_int=_auto_agent_alias_int,
    cooldown_until=_auto_agent_alias_cooldown_until,
)
_aawm_audit_build.install(globals())

_aawm_audit_persist.configure_audit_persist_runtime(
    record_route_status_rollup=lambda *a, **kw: (
        _record_auto_agent_alias_route_status_rollup(*a, **kw)
    ),
    verbose_json_enabled=lambda: _aawm_alias_route_verbose_json_enabled(),
    healthy_json_enabled=lambda: _aawm_alias_route_healthy_json_enabled(),
)
_aawm_audit_persist.install(globals())

_aawm_audit_events.configure_audit_events_runtime(
    get_request_context=lambda *a, **kw: _get_auto_agent_alias_request_context(
        *a, **kw
    ),
    attach_terminal_context_fields=lambda *a, **kw: (
        _attach_auto_agent_alias_terminal_context_fields(*a, **kw)
    ),
    format_timestamp=_format_auto_agent_alias_timestamp,
    extract_metadata_value=lambda *a, **kw: _extract_auto_agent_alias_metadata_value(
        *a, **kw
    ),
    extract_incoming_endpoint=lambda *a, **kw: (
        _extract_auto_agent_alias_incoming_endpoint(*a, **kw)
    ),
    resolve_codex_session_key=lambda *a, **kw: (
        _resolve_codex_auto_agent_session_key(*a, **kw)
    ),
    resolve_anthropic_session_key=lambda *a, **kw: (
        _resolve_anthropic_auto_agent_session_key(*a, **kw)
    ),
    emit_route_event=lambda *a, **kw: _emit_auto_agent_alias_route_event(*a, **kw),
    build_audit_events=lambda *a, **kw: _build_auto_agent_alias_audit_events(
        *a, **kw
    ),
    persist_audit_only_events=lambda *a, **kw: (
        _persist_auto_agent_alias_audit_only_events_best_effort(*a, **kw)
    ),
)
_aawm_audit_events.install(globals())


# ---------------------------------------------------------------------------
# Wave 7: Rollup functions owned by aawm_alias_routing/rollup.py
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import rollup as _aawm_rollup

_aawm_rollup.install(globals())


# ---------------------------------------------------------------------------
# Wave 5D compatibility constants not published by owner install APIs.
# ---------------------------------------------------------------------------

verbose_aawm_route_logger = _aawm_audit_persist.verbose_aawm_route_logger

_AUTO_AGENT_ROLE_DECLARATION_RE = _aawm_audit_context._AUTO_AGENT_ROLE_DECLARATION_RE
_AUTO_AGENT_KNOWN_ROLE_NAMES = _aawm_audit_context._AUTO_AGENT_KNOWN_ROLE_NAMES
_AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES = _aawm_audit_context._AUTO_AGENT_PRIOR_TOOL_ITEM_TYPES
_AUTO_AGENT_FILE_EDIT_TOOL_NAMES = _aawm_audit_context._AUTO_AGENT_FILE_EDIT_TOOL_NAMES
_AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY = _aawm_audit_context._AUTO_AGENT_REQUEST_CALL_ID_STATE_KEY
_AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY = _aawm_audit_context._AUTO_AGENT_REQUEST_CONTEXT_STATE_KEY

_AutoAgentAliasRequestContext = _aawm_audit_context._AutoAgentAliasRequestContext
_KIMI_CODE_MANAGED_ACCOUNT_COOLDOWN_MODEL = _aawm_error_signals._KIMI_CODE_MANAGED_ACCOUNT_COOLDOWN_MODEL
_KIMI_CODE_SAFE_FAILURE_KINDS = _aawm_error_signals._KIMI_CODE_SAFE_FAILURE_KINDS
_KIMI_CODE_SAFE_FAILURE_SCOPES = _aawm_error_signals._KIMI_CODE_SAFE_FAILURE_SCOPES
_KIMI_CODE_SAFE_METADATA_GATES = _aawm_error_signals._KIMI_CODE_SAFE_METADATA_GATES
_KIMI_CODE_SAFE_RESET_REASONS = _aawm_error_signals._KIMI_CODE_SAFE_RESET_REASONS
_KIMI_CODE_SAFE_UPSTREAM_IDS = _aawm_error_signals._KIMI_CODE_SAFE_UPSTREAM_IDS

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


def _resolve_anthropic_auto_agent_alias_model(
    request_body: dict[str, Any],
    endpoint: str,
    *,
    request: Request,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    return _lookup_active_snapshot_canonical_alias(
        request_body.get("model"),
        request=request,
    )


_resolve_anthropic_auto_agent_native_lane_key = _aawm_lane_keys._resolve_anthropic_auto_agent_native_lane_key


_resolve_anthropic_auto_agent_native_cooldown_lane_key = _aawm_lane_keys._resolve_anthropic_auto_agent_native_cooldown_lane_key


def _resolve_anthropic_auto_agent_session_key(
    request: Request,
    request_body: dict[str, Any],
    *,
    alias_model: str,
) -> Optional[str]:
    if not alias_model or alias_model.strip() != alias_model:
        raise ValueError("alias_model must be an explicit canonical alias")
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


def _get_openrouter_free_daily_quota_cache() -> tuple[Optional[float], float]:
    return _alias_routing_state.get_openrouter_free_quota_cache()


def _set_openrouter_free_daily_quota_cache(value: tuple[Optional[float], float]) -> None:
    _alias_routing_state.set_openrouter_free_quota_cache(value)


async def _fetch_openrouter_quota_row_via_facade():
    """Indirection so monkeypatching ``_fetch_openrouter_free_daily_quota_row``
    on this module is visible to the openrouter_quota module."""
    return await _fetch_openrouter_free_daily_quota_row()


_get_openrouter_adapter_active_cooldown_seconds = _wave6b_openrouter_runtime._get_openrouter_adapter_active_cooldown_seconds
_get_openrouter_adapter_rate_limit_key = _wave6b_openrouter_runtime._get_openrouter_adapter_rate_limit_key

_aawm_openrouter_quota.configure_openrouter_quota_runtime(
    get_quota_cache=_get_openrouter_free_daily_quota_cache,
    set_quota_cache=_set_openrouter_free_daily_quota_cache,
    quota_lock=_openrouter_free_daily_quota_lock,
    get_dynamic_injection_pool=_get_aawm_callback_pool,
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
_aawm_cooldown_state.install(globals())

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
    get_openrouter_adapter_active_cooldown_seconds=lambda *a, **kw: _get_openrouter_adapter_active_cooldown_seconds(*a, **kw),
    extract_client_product_label=lambda *a, **kw: _extract_auto_agent_alias_client_product_label(*a, **kw),
    resolve_codex_session_key=lambda *a, **kw: _resolve_codex_auto_agent_session_key(*a, **kw),
    resolve_anthropic_session_key=lambda *a, **kw: _resolve_anthropic_auto_agent_session_key(*a, **kw),
    has_continuation_state=lambda *a, **kw: _codex_auto_agent_request_has_continuation_state(*a, **kw),
    is_grok_account_quota_candidate=lambda *a, **kw: _is_codex_auto_agent_grok_account_quota_candidate(*a, **kw),
    get_grok_account_quota_lane_cooldown_key=lambda *a, **kw: _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(*a, **kw),
    is_kimi_code_candidate=lambda *a, **kw: _is_kimi_code_auto_agent_candidate(*a, **kw),
    get_kimi_managed_account_cooldown_key=lambda *a, **kw: _get_kimi_code_managed_account_cooldown_key(*a, **kw),
    get_codex_quota_observation_pool=lambda: _get_aawm_callback_pool(),
    get_codex_quota_observation_environment=lambda: _get_first_secret_value(
        _AAWM_LITELLM_ENVIRONMENT_ENV_VARS
    ),
)
_aawm_selection.install(globals())

# Wave 5C: bind retained host dependencies into error_signals. Generic
# exception, OpenRouter error-shape, header, and JSON helpers are owner-local
# and are published into this module by install().
_aawm_error_signals.configure_error_signals_runtime(
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
_aawm_error_signals.install(globals())

# Wave 6B: configure the provider-neutral OpenRouter runtime and publish its
# historical host compatibility surface after generic error/header helpers.
_wave6b_openrouter_runtime.install(globals())

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
    codex_failure_evidence_gate=_codex_failure_evidence_gate,
    state_manager=_alias_routing_state,
)
_aawm_cooldown_apply.install(globals())

# Wave 5C: bind error_signals / classification / host dependencies into
# attempt_records.
_aawm_attempt_records.configure_attempt_records_runtime(
    extract_error_tokens=lambda *a, **kw: _extract_codex_auto_agent_error_tokens(*a, **kw),
    extract_error_type_and_code=lambda *a, **kw: _extract_codex_auto_agent_error_type_and_code(*a, **kw),
    parse_header_wait_seconds=lambda *a, **kw: _parse_codex_auto_agent_header_wait_seconds(*a, **kw),
    get_source_error_summary=lambda *a, **kw: _get_codex_auto_agent_source_error_summary(*a, **kw),
    build_kimi_telemetry=lambda *a, **kw: _build_safe_kimi_code_selection_telemetry(*a, **kw),
    extract_status_code=lambda *a, **kw: _extract_adapter_exception_status_code(*a, **kw),
    safe_set_parsed_body=lambda *a, **kw: _safe_set_request_parsed_body(*a, **kw),
    emit_route_event=lambda *a, **kw: _emit_auto_agent_alias_route_event(*a, **kw),
    build_audit_event=lambda *a, **kw: _build_auto_agent_alias_audit_event(*a, **kw),
    build_audit_events=lambda *a, **kw: _build_auto_agent_alias_audit_events(*a, **kw),
    persist_audit_only_events=lambda *a, **kw: _persist_auto_agent_alias_audit_only_events_best_effort(*a, **kw),
    verbose_json_enabled=lambda *a, **kw: _aawm_alias_route_verbose_json_enabled(*a, **kw),
    healthy_json_enabled=lambda *a, **kw: _aawm_alias_route_healthy_json_enabled(*a, **kw),
    merge_metadata=lambda *a, **kw: _merge_litellm_metadata(*a, **kw),
    normalize_tag_value=lambda *a, **kw: _normalize_low_cardinality_tag_value(*a, **kw),
    load_bundled_model_cost=lambda *a, **kw: cast(
        Callable[..., dict[str, Any]],
        _aawm_codex_tool_policy.load_bundled_model_cost_map_for_codex_policy,
    )(*a, **kw),
    get_model_info=lambda *a, **kw: litellm.get_model_info(*a, **kw),
    model_cost=litellm.model_cost,
    openai_provider_value=litellm.LlmProviders.OPENAI.value,
    classify_failure=lambda *a, **kw: _aawm_alias_classification.classify_failure(*a, **kw),
    codex_failure_evidence_gate_record=lambda *a, **kw: (
        _codex_failure_evidence_gate.record(*a, **kw)
    ),
)
_aawm_attempt_records.install(globals())

_wait_for_openrouter_adapter_cooldown_if_needed = _wave6b_openrouter_runtime._wait_for_openrouter_adapter_cooldown_if_needed


_set_openrouter_adapter_cooldown = _wave6b_openrouter_runtime._set_openrouter_adapter_cooldown


_run_openrouter_adapter_retry_loop = _wave6b_openrouter_runtime._run_openrouter_adapter_retry_loop


_perform_openrouter_completion_adapter_operation = _wave6b_openrouter_runtime._perform_openrouter_completion_adapter_operation


_perform_openrouter_adapter_pass_through_request = _wave6b_openrouter_runtime._perform_openrouter_adapter_pass_through_request



# Wave 7: response_utils owner install
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import response_utils as _aawm_response_utils
_aawm_response_utils.install(globals())


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


# Wave 7: claude_agent_spec owner install
from litellm.proxy.pass_through_endpoints.aawm_request_policy import claude_agent_spec as _aawm_claude_agent_spec
_aawm_claude_agent_spec.install(globals())


_aawm_alias_durable.configure_durable_runtime(
    clean_value=_clean_codex_auth_value,
    get_dual_cache_override=lambda: (
        globals()["_get_aawm_alias_routing_dual_cache"]()
        if globals().get("_get_aawm_alias_routing_dual_cache")
        is not _aawm_alias_durable.get_aawm_alias_routing_dual_cache
        else None
    ),
)


PassthroughLoggingMetadata = dict[str, object]


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


_ANTHROPIC_OPENAI_PROVIDER_RUNTIME = _anthropic_openai_provider.Runtime(
    resolve_auth_context=lambda request: (_resolve_anthropic_openai_responses_adapter_auth_context(request)),
    compact_context=lambda body, **kwargs: (
        _aawm_anthropic_body_prep._compact_openai_adapter_claude_context_in_anthropic_request_body(
            body, **kwargs
        )
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
        _aawm_anthropic_body_prep._compact_openai_adapter_claude_context_in_anthropic_request_body(
            body, **kwargs
        )
    ),
    log_debug=lambda message, *args: verbose_proxy_logger.debug(message, *args),
    build_responses_body=lambda body, **kwargs: (
        _build_anthropic_responses_adapter_request_body(body, **kwargs)
    ),
    apply_parallel_policy=lambda body: (
        _apply_openrouter_adapter_parallel_instruction_policy(body)
    ),
    apply_forced_tool_choice=lambda source, translated: (
        _apply_forced_bash_tool_choice_for_responses_adapter(source, translated)
    ),
    contains_mcp_tools=lambda body: _responses_request_contains_mcp_tools(body),
    get_api_key=lambda: _get_anthropic_adapter_openrouter_api_key(),
    raise_candidate_unavailable=lambda detail: (
        _raise_openrouter_auto_agent_candidate_unavailable(str(detail))
    ),
    get_target_base=lambda: _get_anthropic_adapter_openrouter_target_base(),
    normalize_endpoint=lambda **kwargs: (
        BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(**kwargs)
    ),
    join_url=lambda *args: BaseOpenAIPassThroughHandler._join_url_paths(*args),
    url_factory=httpx.URL,
    assemble_headers=lambda **kwargs: (
        BaseOpenAIPassThroughHandler._assemble_headers(**kwargs)
    ),
    build_default_headers=lambda: _build_openrouter_default_headers(),
    perform_responses_request=lambda **kwargs: (
        _perform_openrouter_adapter_pass_through_request(**kwargs)
    ),
    get_completion_model=lambda model: (
        _get_openrouter_completion_adapter_upstream_model(model)
    ),
    prepare_completion_body=lambda body, **kwargs: (
        _prepare_anthropic_completion_adapter_request_body(body, **kwargs)
    ),
    validate_egress=lambda **kwargs: (
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)
    ),
    perform_completion_operation=lambda **kwargs: (
        _perform_openrouter_completion_adapter_operation(**kwargs)
    ),
    provider=litellm.LlmProviders.OPENROUTER.value,
    provider_target=litellm.LlmProviders.OPENROUTER.value,
)


_ANTHROPIC_OPENCODE_ZEN_PROVIDER_RUNTIME = _anthropic_opencode_zen_provider.Runtime(
    build_responses_body=lambda body, **kwargs: (
        _build_anthropic_responses_adapter_request_body(body, **kwargs)
    ),
    add_logging_metadata=lambda body, **kwargs: (
        _add_opencode_zen_logging_metadata(body, **kwargs)
    ),
    apply_parallel_policy=lambda body: (
        _apply_openrouter_adapter_parallel_instruction_policy(body)
    ),
    apply_forced_tool_choice=lambda source, translated: (
        _apply_forced_bash_tool_choice_for_responses_adapter(source, translated)
    ),
    log_debug=lambda message, *args: verbose_proxy_logger.debug(message, *args),
    contains_mcp_tools=lambda body: _responses_request_contains_mcp_tools(body),
    get_target_base=lambda: _get_opencode_zen_target_base(),
    join_url=lambda **kwargs: _join_opencode_zen_passthrough_url(**kwargs),
    build_headers=lambda request, **kwargs: (
        _build_opencode_zen_headers(request, **kwargs)
    ),
    unavailable_detail=lambda exc: _opencode_zen_candidate_unavailable_detail(exc),
    raise_candidate_unavailable=lambda exc: (
        _raise_opencode_zen_auto_agent_candidate_unavailable(exc)
    ),
    url_factory=httpx.URL,
    prepare_completion_body=lambda body, **kwargs: (
        _prepare_anthropic_completion_adapter_request_body(body, **kwargs)
    ),
    load_api_key=lambda **kwargs: (
        _load_opencode_zen_api_key_for_candidate(**kwargs)
    ),
    assemble_headers=lambda **kwargs: (
        BaseOpenAIPassThroughHandler._assemble_headers(**kwargs)
    ),
    validate_egress=lambda **kwargs: (
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(**kwargs)
    ),
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


_wave6b_opencode_zen_runtime.install(globals())


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    for secret_name in secret_names:
        value = _clean_secret_string(get_secret_str(secret_name))
        if value:
            return value
    return None




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

    base_target_url = (
        os.getenv("GEMINI_API_BASE")
        or "https://generativelanguage.googleapis.com"
    )
    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(
        path=_join_gemini_base_and_endpoint_path(base_url, encoded_endpoint)
    )

    # Add or update query parameters
    gemini_api_key: Optional[str] = passthrough_endpoint_router.get_credentials(
        custom_llm_provider="gemini",
        region_name=None,
    )
    if gemini_api_key is None:
        raise Exception(
            "Required 'GEMINI_API_KEY'/'GOOGLE_API_KEY' in environment to make pass-through calls to Google AI Studio."
        )
    # Merge query parameters, giving precedence to those in updated_url
    merged_params = dict(request.query_params)
    merged_params.update({"key": gemini_api_key})

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_llm_provider="gemini",
        is_streaming_request=is_streaming_request,
        query_params=merged_params,
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
        api_key=_get_grok_litellm_auth_header(request),
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
    "/grok/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Grok Pass-through", "pass-through"],
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



_AutoAgentAliasSelectionFn = Callable[..., Awaitable[dict[str, Any]]]
_AutoAgentAliasMetadataFn = Callable[..., dict[str, Any]]


_ANTHROPIC_AUTO_AGENT_ROUTE_RUNTIME = (
    _aawm_anthropic_auto_agent_route.AnthropicAutoAgentRouteRuntime(
        handle_alias_route=lambda *args, **kwargs: (
            _aawm_alias_candidate_loop.handle_alias_route(*args, **kwargs)
        ),
        resolve_cooldown_publication=lambda *args, **kwargs: (
            _resolve_auto_agent_cooldown_publication_plan(*args, **kwargs)
        ),
        anthropic_family_state=_alias_routing_state.anthropic,
        codex_family_state=_alias_routing_state.codex,
        perform_candidate_request=lambda *args, **kwargs: (
            _perform_anthropic_auto_agent_alias_candidate_request(*args, **kwargs)
        ),
        select_candidate=lambda *args, **kwargs: (
            _select_anthropic_auto_agent_candidate(*args, **kwargs)
        ),
        publish_cooldown_memory=lambda *args, **kwargs: (
            _publish_anthropic_cooldown_memory(*args, **kwargs)
        ),
        persist_cooldown_durable=lambda *args, **kwargs: (
            _persist_anthropic_cooldown_durable(*args, **kwargs)
        ),
        set_session_affinity=lambda *args, **kwargs: (
            _set_anthropic_auto_agent_session_affinity(*args, **kwargs)
        ),
        add_alias_metadata=lambda *args, **kwargs: (
            _add_anthropic_auto_agent_alias_metadata(*args, **kwargs)
        ),
        raise_redispatch_required=lambda *args, **kwargs: (
            _raise_anthropic_auto_agent_redispatch_required(*args, **kwargs)
        ),
        extract_client_product_label=lambda *args, **kwargs: (
            _extract_auto_agent_alias_client_product_label(*args, **kwargs)
        ),
        resolve_selection_enumeration=lambda *args, **kwargs: (
            _resolve_aawm_alias_selection_enumeration(*args, **kwargs)
        ),
        get_active_cooldown_state=lambda *args, **kwargs: (
            _get_anthropic_auto_agent_active_cooldown_state(*args, **kwargs)
        ),
    )
)
_handle_auto_agent_alias_route = partial(
    _aawm_anthropic_auto_agent_route.handle_auto_agent_alias_route,
    _ANTHROPIC_AUTO_AGENT_ROUTE_RUNTIME,
)
_handle_anthropic_auto_agent_alias_route = partial(
    _aawm_anthropic_auto_agent_route.handle_anthropic_auto_agent_alias_route,
    _ANTHROPIC_AUTO_AGENT_ROUTE_RUNTIME,
)

# Wave 7: anthropic_native owner install
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import anthropic_native as _aawm_anthropic_native


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


_aawm_anthropic_native.install(
    globals(),
    runtime=_aawm_anthropic_native.AnthropicNativeRuntime(
        is_streaming_request_fn=lambda request: is_streaming_request_fn(request),
        create_pass_through_route=lambda *args, **kwargs: create_pass_through_route(
            *args, **kwargs
        ),
    ),
)

_aawm_alias_candidate_dispatch.install(
    globals(),
    runtime=_aawm_alias_candidate_dispatch.AliasCandidateDispatchRuntime(
        handle_openai_responses=lambda **kwargs: (
            _handle_anthropic_openai_responses_adapter_route(**kwargs)
        ),
        handle_openrouter_completion=lambda **kwargs: (
            _handle_anthropic_openrouter_completion_adapter_route(**kwargs)
        ),
        handle_openrouter_responses=lambda **kwargs: (
            _handle_anthropic_openrouter_responses_adapter_route(**kwargs)
        ),
        handle_xai_oauth_responses=lambda **kwargs: (
            _handle_anthropic_xai_oauth_responses_adapter_route(**kwargs)
        ),
        handle_grok_native_oauth_responses=lambda **kwargs: (
            _handle_anthropic_grok_native_oauth_responses_adapter_route(**kwargs)
        ),
        handle_opencode_zen=lambda **kwargs: (
            _handle_anthropic_opencode_zen_adapter_route(**kwargs)
        ),
        handle_kimi_chat_completions=lambda **kwargs: (
            _handle_anthropic_kimi_chat_completions_adapter_route(**kwargs)
        ),
        handle_alibaba_token_plan=lambda **kwargs: (
            _handle_anthropic_alibaba_token_plan_adapter_route(**kwargs)
        ),
        normalize_native_model_alias=lambda body: (
            _normalize_anthropic_native_passthrough_model_alias(body)
        ),
        prepare_context_1m_native=lambda **kwargs: (
            _prepare_anthropic_context_1m_native_passthrough(**kwargs)
        ),
        safe_set_request_parsed_body=lambda request, body: (
            _safe_set_request_parsed_body(request, body)
        ),
        perform_native_passthrough=lambda **kwargs: (
            _perform_anthropic_native_passthrough_request(**kwargs)
        ),
        provider_native=_CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        provider_openrouter=_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        provider_xai=_CODEX_AUTO_AGENT_XAI_PROVIDER,
        provider_opencode=_CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        provider_kimi=_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
        provider_alibaba=_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
        anthropic_beta_header_name=_ANTHROPIC_BETA_HEADER_NAME,
    ),
)


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
            request=request,
        )
        if anthropic_auto_agent_alias is not None:
            (
                prepared_request_body,
                _anthropic_read_guidance_changes,
            ) = _apply_aawm_read_agent_guidance_to_request_body(
                prepared_request_body,
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
                canonical_alias=anthropic_auto_agent_alias,
            )

        adapter_response = await try_dispatch_anthropic_adapter(
            endpoint=encoded_endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=prepared_request_body,
        )
        if adapter_response is not None:
            return adapter_response

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

    # D1-612: native Anthropic path shares the global session-owner guard.
    # Reservation is finalized in pass_through_request immediately before
    # upstream send; pre-check owned mismatches here so adapter fallthrough
    # cannot select a different owner first.
    import sys as _sys

    _sa = _sys.modules.get(
        "litellm.proxy.pass_through_endpoints.aawm_alias_routing.session_affinity"
    )
    if _sa is None:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
            session_affinity as _sa,
        )
    if request.method == "POST":
        try:
            _body = prepared_request_body if isinstance(prepared_request_body, dict) else {}
        except Exception:  # noqa: BLE001
            _body = {}
        _requested_model = _body.get("model") if isinstance(_body, dict) else None
        _attrs = _sa.build_session_owner_attributes(
            provider="anthropic",
            model=_requested_model,
            route_family="anthropic_native",
            endpoint_contract="anthropic_messages",
            state_format="anthropic",
            ingress="anthropic_passthrough",
            requested_model=_requested_model,
        )
        await _sa.ensure_session_owner_guard_for_request(
            request=request,
            request_body=_body if isinstance(_body, dict) else {},
            requested_attributes=_attrs,
            alias_model=str(_requested_model) if _requested_model is not None else None,
            failure_phase="session_owner_anthropic_native_pre_egress",
        )
        _lease = _sa.get_request_session_owner_lease(request)
        if _lease is not None and isinstance(_body, dict):
            _meta = _body.get("litellm_metadata")
            if not isinstance(_meta, dict):
                _meta = {}
                _body["litellm_metadata"] = _meta
            _sa.attach_session_owner_metadata(
                _meta,
                provenance=_sa.build_session_owner_provenance(
                    session_identity=_lease.session_identity,
                    decision=_lease.decision or "unknown",
                    owner_id=_lease.owner_id,
                ),
            )
            _safe_set_request_parsed_body(request, _body)

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
    user_api_key_dict: UserAPIKeyAuth = Depends(_cfg004_openai_passthrough_auth_dependency),
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
    is_codex_auto_agent_alias_request = False
    if request.method == "POST":
        request_body = await get_request_body(request)
        is_oa_xai_request = _is_oa_xai_request_body(request_body)
        is_grok_native_oauth_request = _is_openai_responses_endpoint(endpoint) and _is_grok_native_oauth_request_body(
            request_body
        )
        is_codex_auto_agent_alias_request = (
            _resolve_codex_auto_agent_alias_model(
                request_body,
                endpoint,
                request=request,
            )
            is not None
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
    elif is_codex_auto_agent_alias_request:
        base_target_url = os.getenv("CHATGPT_API_BASE") or CHATGPT_API_BASE
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






























_CODEX_AUTO_AGENT_ROUTE_RUNTIME = (
    _aawm_codex_auto_agent_route.CodexAutoAgentRouteRuntime(
        extract_client_product_label_fn=lambda *args, **kwargs: (
            _extract_auto_agent_alias_client_product_label(*args, **kwargs)
        ),
        perform_candidate_request_fn=lambda *args, **kwargs: (
            _perform_codex_auto_agent_alias_candidate_request(*args, **kwargs)
        ),
        select_candidate_fn=lambda *args, **kwargs: (
            _select_codex_auto_agent_candidate(*args, **kwargs)
        ),
        resolve_cooldown_publication_fn=lambda *args, **kwargs: (
            _resolve_auto_agent_cooldown_publication_plan(*args, **kwargs)
        ),
        publish_cooldown_memory_fn=lambda *args, **kwargs: (
            _publish_codex_cooldown_memory(*args, **kwargs)
        ),
        persist_cooldown_fn=lambda *args, **kwargs: (
            _persist_codex_cooldown_durable(*args, **kwargs)
        ),
        set_session_affinity_fn=lambda *args, **kwargs: (
            _set_codex_auto_agent_session_affinity(*args, **kwargs)
        ),
        add_alias_metadata_fn=lambda *args, **kwargs: (
            _add_codex_auto_agent_alias_metadata(*args, **kwargs)
        ),
        raise_redispatch_fn=lambda *args, **kwargs: (
            _raise_codex_auto_agent_redispatch_required(*args, **kwargs)
        ),
        get_active_cooldown_state_fn=lambda *args, **kwargs: (
            _get_codex_auto_agent_active_cooldown_state(*args, **kwargs)
        ),
        resolve_selection_enumeration_fn=lambda *args, **kwargs: (
            _resolve_aawm_alias_selection_enumeration(*args, **kwargs)
        ),
    )
)
_handle_codex_auto_agent_alias_route = partial(
    _aawm_codex_auto_agent_route.handle_codex_auto_agent_alias_route,
    _CODEX_AUTO_AGENT_ROUTE_RUNTIME,
)


# ---------------------------------------------------------------------------
# Wave 7: BaseOpenAIPassThroughHandler owned by
# aawm_adapter_runtime/openai_passthrough_handler.py. The class is imported
# and re-exported here; its DI runtime is installed lazily after all host
# callbacks are published (see install block near module tail).
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_passthrough_handler import (  # noqa: E402
    BaseOpenAIPassThroughHandler,
)


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
        extract_status_code=lambda exc: _extract_adapter_exception_status_code(exc),
        extract_detail=lambda exc: _extract_adapter_exception_detail(exc),
    )


_wave6b_nvidia_runtime.configure_nvidia_runtime(
    _wave6b_nvidia_runtime.NvidiaRuntimeDependencies(
        get_first_secret_value=lambda names: _get_first_secret_value(names),
        clean_secret_string=lambda value: _clean_secret_string(value),
        clean_auth_value=lambda value: _clean_codex_auth_value(value),
        get_env=lambda name: os.getenv(name),
        sleep=lambda seconds: asyncio.sleep(seconds),
        log_debug=verbose_proxy_logger.debug,
        log_warning=verbose_proxy_logger.warning,
    )
)

_wave6b_xai_request_prep.configure_xai_request_prep_runtime(
    _wave6b_xai_request_prep.build_default_xai_request_prep_runtime(
        get_model_metadata_entry=lambda model: _get_model_metadata_entry(model),
        get_openai_tool_type=lambda tool: (
            _aawm_codex_tool_policy.get_openai_tool_type(tool)
        ),
        normalize_low_cardinality_tag_value=lambda value: (
            _normalize_low_cardinality_tag_value(value)
        ),
        dedupe_sorted_str_list=lambda values: _dedupe_sorted_str_list(values),
        merge_litellm_metadata=lambda *args, **kwargs: (
            _merge_litellm_metadata(*args, **kwargs)
        ),
        build_langfuse_span_descriptor=lambda *args, **kwargs: (
            _build_langfuse_span_descriptor(*args, **kwargs)
        ),
        drop_unsupported_codex_hosted_tools_from_request_body=lambda body: (
            _drop_unsupported_codex_hosted_tools_from_request_body(body)
        ),
        drop_unsupported_codex_request_params_from_request_body=lambda body: (
            _drop_unsupported_codex_request_params_from_request_body(body)
        ),
        drop_unsupported_codex_input_items_from_request_body=lambda body: (
            _drop_unsupported_codex_input_items_from_request_body(body)
        ),
        drop_tool_choice_without_tools_from_request_body=lambda body: (
            _drop_tool_choice_without_tools_from_request_body(body)
        ),
        replace_request_body_in_place=lambda original, replacement: (
            _replace_request_body_in_place(original, replacement)
        ),
        safe_get_request_headers=lambda request: _safe_get_request_headers(
            request
        ),
        get_case_insensitive_header=lambda headers, name: (
            _get_case_insensitive_header(headers, name)
        ),
        get_rewrite_input_item_types_for_model=lambda model: (
            _get_rewrite_input_item_types_for_model(model)
        ),
        get_grok_passthrough_target_base=lambda: _get_grok_passthrough_target_base(),
        get_grok_native_oauth_access_token=lambda: (
            get_grok_native_oauth_access_token()
        ),
    )
)



# ---------------------------------------------------------------------------
# Wave 7: observability-metadata owner install (publishes shared primitives)
# ---------------------------------------------------------------------------
_aawm_observability_metadata.configure_observability_metadata_runtime(
    get_explicit_tenant_id=_get_aawm_tenant_header,
    get_request_headers=_safe_get_request_headers,
    get_env=os.getenv,
)
_aawm_observability_metadata.install(globals())
_initialize_aawm_claude_control_plane()

# ---------------------------------------------------------------------------
# Wave 7: codex-tool-policy owner install (replaces 42 inline wrappers)
# ---------------------------------------------------------------------------
_aawm_codex_tool_policy.configure_and_install_codex_tool_policy(
    globals(),
    _aawm_codex_tool_policy.CodexToolPolicyHostDeps(
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
        grok_normalization_runtime=_anthropic_grok_normalization.Runtime(
            normalize_tag=_normalize_low_cardinality_tag_value,
            dedupe_sorted=_dedupe_sorted_str_list,
            merge_metadata=_merge_litellm_metadata,
            build_span=_build_langfuse_span_descriptor,
            get_rewrite_input_item_types=lambda *a, **kw: globals()[
                "_get_rewrite_input_item_types_for_model"
            ](*a, **kw),
        ),
        request_body_walk_max_depth=_AAWM_REQUEST_BODY_WALK_MAX_DEPTH,
    ),
)

# ---------------------------------------------------------------------------
# Wave 7: OpenAI pass-through handler runtime installation
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    openai_passthrough_handler as _aawm_openai_passthrough_handler,
)

_aawm_openai_passthrough_handler.install_runtime(
    _aawm_openai_passthrough_handler.build_runtime_from_host()
)



# ---------------------------------------------------------------------------
# Wave 6E anthropic-body-prep runtime configuration
# ---------------------------------------------------------------------------
_aawm_persisted_output.install(globals())

_aawm_alias_guidance.configure_alias_guidance_runtime(
    callbacks=_aawm_alias_guidance.AliasGuidanceCallbacks(
        merge_litellm_metadata=_merge_litellm_metadata,
        build_langfuse_span_descriptor=_build_langfuse_span_descriptor,
    ),
)
_append_codex_auto_agent_prevention_guidance_to_instructions = (
    _aawm_alias_guidance._append_codex_auto_agent_prevention_guidance_to_instructions
)
_append_aawm_read_agent_guidance_to_text = (
    _aawm_alias_guidance._append_aawm_read_agent_guidance_to_text
)
_append_aawm_read_agent_guidance_to_anthropic_system = (
    _aawm_alias_guidance._append_aawm_read_agent_guidance_to_anthropic_system
)
_apply_aawm_read_agent_guidance_to_request_body = (
    _aawm_alias_guidance._apply_aawm_read_agent_guidance_to_request_body
)
_apply_codex_auto_agent_prevention_guidance_to_request_body = (
    _aawm_alias_guidance._apply_codex_auto_agent_prevention_guidance_to_request_body
)

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


# ---------------------------------------------------------------------------
# Wave 6F adapter-call facades and dispatch runtime
# ---------------------------------------------------------------------------
_aawm_lane_keys.install(globals())
_aawm_adapter_model_resolution.install(globals())
_aawm_adapter_runtime.install(globals())
_aawm_adapter_runtime.install_wave6f(globals())

_aawm_responses_finalize.configure_responses_finalize_runtime(
    _aawm_responses_finalize.ResponsesFinalizeRuntime(
        annotate_request=lambda *args, **kwargs: globals()[
            "_annotate_request_scope_for_adapted_access_log"
        ](*args, **kwargs),
        validate_stream=lambda *args, **kwargs: globals()[
            "_validate_alias_candidate_responses_stream_if_needed"
        ](*args, **kwargs),
        collect_stream=lambda *args, **kwargs: globals()[
            "_collect_responses_response_from_stream"
        ](*args, **kwargs),
        build_response=lambda *args, **kwargs: globals()[
            "_build_anthropic_response_from_responses_response"
        ](*args, **kwargs),
        copy_headers=lambda *args, **kwargs: globals()[
            "_copy_translated_anthropic_adapter_response_headers"
        ](*args, **kwargs),
        build_streaming_response=lambda *args, **kwargs: globals()[
            "_build_anthropic_streaming_response_from_responses_stream"
        ](*args, **kwargs),
        decode_response_body=lambda *args, **kwargs: globals()[
            "_decode_http_response_body"
        ](*args, **kwargs),
        build_malformed_context=lambda *args, **kwargs: globals()[
            "_build_malformed_intake_context_for_anthropic_responses_adapter"
        ](*args, **kwargs),
    )
)

_ANTHROPIC_DISPATCH_RUNTIME = _aawm_anthropic_dispatch.AnthropicDispatchRuntime(
    resolve_xai_oauth=lambda body, endpoint: _resolve_anthropic_xai_oauth_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_grok_native_oauth=lambda body, endpoint: _resolve_anthropic_grok_native_oauth_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_openai_responses=lambda body, endpoint: _resolve_anthropic_openai_responses_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_opencode_zen=lambda body, endpoint: _resolve_anthropic_opencode_zen_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_kimi=lambda body, endpoint: _resolve_anthropic_kimi_chat_completions_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_alibaba=lambda body, endpoint: _resolve_anthropic_alibaba_token_plan_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_nvidia=lambda body, endpoint: _resolve_anthropic_nvidia_responses_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_openrouter_completion=lambda body, endpoint: _resolve_anthropic_openrouter_completion_adapter_model(
        body, endpoint=endpoint
    ),
    resolve_openrouter_responses=lambda body, endpoint: _resolve_anthropic_openrouter_responses_adapter_model(
        body, endpoint=endpoint
    ),
    handle_xai_oauth_responses=lambda **kwargs: _handle_anthropic_xai_oauth_responses_adapter_route(
        **kwargs
    ),
    handle_xai_oauth_completion=lambda **kwargs: _handle_anthropic_xai_oauth_completion_adapter_route(
        **kwargs
    ),
    handle_grok_native_oauth_responses=lambda **kwargs: _handle_anthropic_grok_native_oauth_responses_adapter_route(
        **kwargs
    ),
    handle_openai_responses=lambda **kwargs: _handle_anthropic_openai_responses_adapter_route(
        **kwargs
    ),
    handle_opencode_zen=lambda **kwargs: _handle_anthropic_opencode_zen_adapter_route(
        **kwargs
    ),
    handle_kimi=lambda **kwargs: _handle_anthropic_kimi_chat_completions_adapter_route(
        **kwargs
    ),
    handle_alibaba=lambda **kwargs: _handle_anthropic_alibaba_token_plan_adapter_route(
        **kwargs
    ),
    handle_nvidia=lambda **kwargs: _handle_anthropic_nvidia_completion_adapter_route(
        **kwargs
    ),
    handle_openrouter_completion=lambda **kwargs: _handle_anthropic_openrouter_completion_adapter_route(
        **kwargs
    ),
    handle_openrouter_responses=lambda **kwargs: _handle_anthropic_openrouter_responses_adapter_route(
        **kwargs
    ),
    is_oa_xai_responses_model=lambda model: _is_oa_xai_responses_model(model),
)


try_dispatch_anthropic_adapter = partial(
    _aawm_anthropic_dispatch.try_dispatch_anthropic_adapter,
    _ANTHROPIC_DISPATCH_RUNTIME,
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
