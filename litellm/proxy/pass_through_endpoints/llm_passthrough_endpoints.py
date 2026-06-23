"""
What is this?

Provider-specific Pass-Through Endpoints

Use litellm with Anthropic SDK, Vertex AI SDK, Cohere SDK, etc.
"""

import ast
import asyncio
import base64
import codecs
import contextlib
import copy
import glob
import importlib
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import lru_cache
from importlib.resources import files
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional, Tuple, Union, cast
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

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
from litellm.integrations.aawm_passthrough_shape_capture import (
    capture_passthrough_shape,
)
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_API_BASE,
    get_chatgpt_default_headers,
)
from litellm.llms.xai.oauth import (
    build_grok_native_oauth_metadata,
    get_grok_native_oauth_access_token,
    is_grok_native_oauth_model,
    is_oa_xai_model,
    normalize_grok_native_oauth_model,
    prepare_oa_xai_request,
    resolve_oa_xai_upstream_model,
)
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
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
    PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS,
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES,
    create_pass_through_route,
    create_websocket_passthrough_route,
    pass_through_request,
    websocket_passthrough_request,
    _classify_passthrough_hidden_retry_failure,
    _get_passthrough_hidden_retry_wait_seconds,
    _record_passthrough_hidden_retry_metadata,
)
from litellm.proxy.pass_through_endpoints.google_code_assist_quota import (
    sanitize_google_code_assist_quota_for_logging as _sanitize_google_code_assist_quota_for_logging,
)
from litellm.proxy.aawm_route_logging import (
    build_aawm_route_rollup_group_header_label,
    emit_aawm_route_status_event,
    record_aawm_route_rollup,
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
from litellm.proxy.utils import is_known_model
from litellm.proxy.vector_store_endpoints.utils import (
    is_allowed_to_call_vector_store_endpoint,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIOptionalRequestParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from .passthrough_endpoint_router import PassthroughEndpointRouter


vertex_llm_base = VertexBase()
router = APIRouter()
default_vertex_config = None

passthrough_endpoint_router = PassthroughEndpointRouter()


def _decode_http_response_body(body: Any) -> str:
    return bytes(body).decode("utf-8")


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
_ANTIGRAVITY_AUTH_FILE_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_AUTH_FILE",
    "ANTIGRAVITY_OAUTH_TOKEN_FILE",
)
_ANTIGRAVITY_MANAGED_AUTH_FILE_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE",
)
_ANTIGRAVITY_SEED_AUTH_FILE_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_SEED_AUTH_FILE",
)
_ANTIGRAVITY_DEFAULT_AUTH_PATHS = (
    "/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token",
    "~/.gemini/antigravity-cli/antigravity-oauth-token",
)
_ANTIGRAVITY_OAUTH_CLIENT_ID_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_ID",
    "ANTIGRAVITY_OAUTH_CLIENT_ID",
)
_ANTIGRAVITY_OAUTH_CLIENT_SECRET_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_OAUTH_CLIENT_SECRET",
    "ANTIGRAVITY_OAUTH_CLIENT_SECRET",
)
_ANTIGRAVITY_CLI_BINARY_PATH_ENV_VARS = (
    "LITELLM_ANTIGRAVITY_CLI_PATH",
    "ANTIGRAVITY_CLI_PATH",
)
_ANTIGRAVITY_DEFAULT_CLI_BINARY_PATHS = (
    "/home/zepfu/.local/bin/agy",
    "~/.local/bin/agy",
)
_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "authorization",
        "content-type",
        "user-agent",
        "x-goog-api-client",
        "x-goog-fieldmask",
        "x-goog-request-params",
        "x-goog-request-reason",
    }
)

_OPENCODE_ZEN_DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
_OPENCODE_ZEN_PROVIDER = "opencode_zen"
_OPENCODE_ZEN_AUTH_FILE_ENV_VARS = (
    "LITELLM_OPENCODE_AUTH_FILE",
    "OPENCODE_AUTH_FILE",
)
_OPENCODE_ZEN_API_KEY_ENV_VARS = (
    "LITELLM_OPENCODE_API_KEY",
    "OPENCODE_API_KEY",
)
_OPENCODE_ZEN_DEFAULT_AUTH_PATHS = (
    "/home/zepfu/.local/share/opencode/auth.json",
    "~/.local/share/opencode/auth.json",
)
_OPENCODE_ZEN_FREE_MODELS = frozenset(
    {
        "big-pickle",
        "mini-v2.5",
        "north-mini-code",
        "nemotron-3-ultra",
        "deepseek-v4-flash",
    }
)
_OPENCODE_ZEN_ANTHROPIC_COMPLETION_MODELS = frozenset({"big-pickle"})

_GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL = "https://cli-chat-proxy.grok.com"

_GROK_CLI_FORWARD_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "accept-encoding",
        "authorization",
        "content-type",
        "grok-shell-timestamp",
        "user-agent",
        "x-email",
        "x-grok-agent-id",
        "x-grok-client-identifier",
        "x-grok-client-version",
        "x-grok-conv-id",
        "x-grok-model-override",
        "x-grok-req-id",
        "x-grok-session-id",
        "x-grok-turn-idx",
        "x-grok-user-id",
        "x-request-id",
        "x-teamid",
        "x-userid",
        "x-xai-token-auth",
    }
)

_GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE = frozenset(
    {
        "content-length",
        "host",
        "traceparent",
        "tracestate",
        "x-litellm-api-key",
    }
)

_CLAUDE_PERSISTED_OUTPUT_PATTERN = re.compile(
    r"\A<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context: <persisted-output>\n"
    r"Output too large \([^)]+\)\. Full output saved to: (?P<path>/[^\n]+)\n\n"
    r"Preview \(first 2KB\):\n"
    r"(?P<preview>.*)"
    r"\n</persisted-output>\n</system-reminder>\n?\Z",
    re.DOTALL,
)
_CLAUDE_EXPANDED_PERSISTED_OUTPUT_PATTERN = re.compile(
    r"\A<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context: <persisted-output>\n"
    r"(?P<content>.*)"
    r"\n</persisted-output>\n</system-reminder>\n?\Z",
    re.DOTALL,
)
_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN = re.compile(
    r"<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context: <persisted-output>\n"
    r"Output too large \([^)]+\)\. Full output saved to: (?P<path>/[^\n]+)\n\n"
    r"Preview \(first 2KB\):\n"
    r"(?P<preview>.*?)"
    r"\n</persisted-output>\n</system-reminder>\n?",
    re.DOTALL,
)
_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN = re.compile(
    r"<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context: <persisted-output>\n"
    r"(?P<content>.*?)"
    r"\n</persisted-output>\n</system-reminder>\n?",
    re.DOTALL,
)
_CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN = re.compile(
    r"<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context:(?P<body>.*?)"
    r"</system-reminder>\n?",
    re.DOTALL,
)
_ANTHROPIC_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME = "google_anthropic_system_prompt_policy"
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION = "2026-04-27.v2"
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV = (
    "AAWM_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY"
)
_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT = "replace_compact"
_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT = """You are a non-interactive CLI software engineering agent.

Work in this cycle: understand, plan briefly, implement, verify, finalize.
Use the provided tools to inspect and modify the workspace when the task
requires it.

Tool usage:
- Prefer search tools before broad file reads.
- Batch independent search/read calls in parallel when possible.
- Use write/edit tools to complete requested artifacts or code changes.
- If a tool is unavailable or blocked, recover with another available tool.
- Do not remain in read-only exploration when the user requested an
  implementation or artifact.
- Final responses must include visible assistant text. Never end a completed
  task with only thoughts or reasoning. After tool results, write the requested
  final answer in normal text.

Follow the preserved project, workspace, safety, and operator instructions
below."""
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME = (
    "codex_google_code_assist_tool_contract_policy"
)
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION = "2026-05-12.v1"
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV = (
    "AAWM_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY"
)
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT = "append"
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT = """Codex tool contract:
- Tool results are observations only. Never copy a previous tool result, terminal transcript, "Chunk ID", "Wall time", "Process exited", or "Output:" text into the arguments for a later tool call.
- For every function call, construct arguments from the declared tool schema. If the tool requires `cmd`, the arguments must contain a non-empty `cmd` string. Do not use `output`, `content`, or raw terminal transcript text as a substitute.
- After a tool result, continue the assigned task. Use the latest user task and requested output shape as authoritative.
- If a previous tool call failed because required arguments were missing, either retry once with schema-valid arguments or stop and explain the blocker in the final answer.
- Final answers must address the assigned task directly. Do not return generic descriptions of files unless the user asked for a file overview."""
_CODEX_AUTO_AGENT_MODEL_ALIAS = "aawm-codex-agent-auto"
_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME = (
    "codex_auto_agent_prevention_guidance"
)
_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION = "2026-06-01.v1"
_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT = """Codex auto-agent completion contract:
- Always produce a non-empty final answer after completing or stopping the task; do not end a successful request with only reasoning, tool calls, or no visible assistant text.
- Do not return internal planning text as the final answer. Complete the requested work, or state the exact blocker and the next concrete step.
- If a required tool is unavailable or blocked, state the exact observed tool/platform error and continue with bounded evidence from available context; do not claim tools or filesystem are unavailable unless a tool/platform error proves it.
- If the user requested code or artifact changes, either make the scoped change or explicitly say no files were modified and why. Do not answer with a generic explanation of the function or file when implementation or verification was requested.
- If verification could not be run, name the command or check that was not run and why."""
_AAWM_READ_AGENT_GUIDANCE_POLICY_NAME = "aawm_read_agent_guidance"
_AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION = "2026-06-06.v1"
_AAWM_READ_AGENT_GUIDANCE_PROMPT = """AAWM read-only agent contract:
- Treat the delegated task as exploration, audit, review, or investigation unless the prompt explicitly authorizes file edits for this worker.
- Do not edit files, create files, apply patches, or run commands that modify the worktree.
- If a fix is needed, describe the patch only. Do not claim the patch was implemented unless the prompt explicitly authorized edits and the files were actually changed.
- If the delegated prompt requires the exact final phrase `No files were modified.`, include that phrase truthfully in the final answer.
- Return findings, evidence, coverage gaps, and recommended next steps. Do not return implementation summaries for read-only work."""
_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS = 6 * 60 * 60
_CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS = 30.0
_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS = 3 * 60 * 60.0
_CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS = (
    _CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
_CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = (
    _CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
_CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS = (
    _CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS
)
_CODEX_AUTO_AGENT_AUTH_DEGRADED_COOLDOWN_SECONDS = 5 * 60.0
_CODEX_AUTO_AGENT_AUTH_DEGRADED_LOG_INTERVAL_SECONDS = 60.0
_CODEX_AUTO_AGENT_ANTIGRAVITY_AUTH_DEGRADED_LANE_KEY = (
    "antigravity:auth_degraded"
)
_ANTHROPIC_AUTO_AGENT_NO_TOOL_COMPATIBLE_RETRY_AFTER_SECONDS = 5 * 60
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
_CODEX_AUTO_AGENT_NATIVE_PROVIDER = "openai"
_CODEX_AUTO_AGENT_GOOGLE_PROVIDER = "google_code_assist"
_CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER = "antigravity"
_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER = "openrouter"
_CODEX_AUTO_AGENT_XAI_PROVIDER = "xai"
_CODEX_AUTO_AGENT_OPENCODE_PROVIDER = _OPENCODE_ZEN_PROVIDER
_CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY = "openrouter"
_CODEX_AUTO_AGENT_XAI_LANE_KEY = "xai_grok_native"
_CODEX_AUTO_AGENT_OPENCODE_LANE_KEY = _OPENCODE_ZEN_PROVIDER
_codex_auto_agent_antigravity_auth_degraded_log_until_monotonic = 0.0
_CODEX_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "deepseek/deepseek-v4-flash",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.4-mini",
        "route_family": "codex_responses",
        "last_resort": True,
    },
)
_CODEX_AAWM_READ_ALIAS = "aawm-read"
_CODEX_AAWM_SOTA_ALIAS = "aawm-sota"
_CODEX_AAWM_CODE_ALIAS = "aawm-code"
_CODEX_AAWM_LOW_ALIAS = "aawm-low"
_CODEX_AAWM_SOTA_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.5",
        "route_family": "codex_responses",
        "last_resort": False,
    },
)
_CODEX_AAWM_CODE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "codex_responses",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-composer-2.5-fast",
        "route_family": "codex_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "oa_xai/grok-build",
        "route_family": "codex_xai_oauth_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.5",
        "route_family": "codex_responses",
        "last_resort": True,
        "default_reasoning_effort": "medium",
    },
)
_CODEX_AAWM_LOW_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/cohere/north-mini-code:free",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/owl-alpha",
        "route_family": "codex_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "deepseek-v4-flash",
        "route_family": "codex_opencode_zen_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "big-pickle",
        "route_family": "codex_opencode_zen_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.4-mini",
        "route_family": "codex_responses",
        "last_resort": True,
    },
)
_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[str, tuple[dict[str, Any], ...]] = {
    _CODEX_AUTO_AGENT_MODEL_ALIAS: _CODEX_AUTO_AGENT_CANDIDATES,
    _CODEX_AAWM_READ_ALIAS: _CODEX_AUTO_AGENT_CANDIDATES,
    _CODEX_AAWM_SOTA_ALIAS: _CODEX_AAWM_SOTA_CANDIDATES,
    _CODEX_AAWM_CODE_ALIAS: _CODEX_AAWM_CODE_CANDIDATES,
    _CODEX_AAWM_LOW_ALIAS: _CODEX_AAWM_LOW_CANDIDATES,
}
_ANTHROPIC_AUTO_AGENT_MODEL_ALIAS = "aawm-anthropic-agent-auto"
_ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER = "anthropic"
_ANTHROPIC_AUTO_AGENT_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_AUTO_AGENT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "anthropic_openai_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "deepseek/deepseek-v4-flash",
        "route_family": "anthropic_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": _ANTHROPIC_AUTO_AGENT_HAIKU_MODEL,
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
_ANTHROPIC_AAWM_READ_ALIAS = "aawm-read-anthropic"
_ANTHROPIC_AAWM_SOTA_ALIAS = "aawm-sota-anthropic"
_ANTHROPIC_AAWM_CODE_ALIAS = "aawm-code-anthropic"
_ANTHROPIC_AAWM_LOW_ALIAS = "aawm-low-anthropic"
_ANTHROPIC_AAWM_SOTA_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-opus-4-8",
        "route_family": "anthropic_messages",
        "last_resort": False,
    },
)
_ANTHROPIC_AAWM_CODE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "gpt-5.3-codex-spark",
        "route_family": "anthropic_openai_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "grok-composer-2.5-fast",
        "route_family": "anthropic_grok_native_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_XAI_PROVIDER,
        "model": "oa_xai/grok-build",
        "route_family": "anthropic_xai_oauth_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": "claude-sonnet-4-6",
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
_ANTHROPIC_AAWM_LOW_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/cohere/north-mini-code:free",
        "route_family": "anthropic_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER,
        "model": "openrouter/owl-alpha",
        "route_family": "anthropic_openrouter_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "deepseek-v4-flash",
        "route_family": "anthropic_opencode_zen_responses_adapter",
        "last_resort": False,
    },
    {
        "provider": _CODEX_AUTO_AGENT_OPENCODE_PROVIDER,
        "model": "big-pickle",
        "route_family": "anthropic_opencode_zen_completion_adapter",
        "last_resort": False,
    },
    {
        "provider": _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER,
        "model": _ANTHROPIC_AUTO_AGENT_HAIKU_MODEL,
        "route_family": "anthropic_messages",
        "last_resort": True,
    },
)
_ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS: dict[
    str, tuple[dict[str, Any], ...]
] = {
    _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS: _ANTHROPIC_AUTO_AGENT_CANDIDATES,
    _ANTHROPIC_AAWM_READ_ALIAS: _ANTHROPIC_AUTO_AGENT_CANDIDATES,
    _ANTHROPIC_AAWM_SOTA_ALIAS: _ANTHROPIC_AAWM_SOTA_CANDIDATES,
    _ANTHROPIC_AAWM_CODE_ALIAS: _ANTHROPIC_AAWM_CODE_CANDIDATES,
    _ANTHROPIC_AAWM_LOW_ALIAS: _ANTHROPIC_AAWM_LOW_CANDIDATES,
}


def _get_codex_auto_agent_candidates_for_alias(
    alias_model: str,
) -> tuple[dict[str, Any], ...]:
    candidates = _CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(
        alias_model,
        _CODEX_AUTO_AGENT_CANDIDATES,
    )
    return candidates


def _get_anthropic_auto_agent_candidates_for_alias(
    alias_model: str,
) -> tuple[dict[str, Any], ...]:
    candidates = _ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(
        alias_model,
        _ANTHROPIC_AUTO_AGENT_CANDIDATES,
    )
    return candidates


_codex_auto_agent_cooldown_until_monotonic_by_key: dict[str, float] = {}
_codex_auto_agent_session_affinity_by_key: dict[str, dict[str, Any]] = {}
_codex_auto_agent_lock = asyncio.Lock()
_anthropic_auto_agent_cooldown_until_monotonic_by_key: dict[str, float] = {}
_anthropic_auto_agent_session_affinity_by_key: dict[str, dict[str, Any]] = {}
_anthropic_auto_agent_lock = asyncio.Lock()
_codex_auto_agent_google_lane_key_until_monotonic_by_key: dict[str, float] = {}
_codex_auto_agent_google_lane_key_by_key: dict[str, str] = {}
_codex_auto_agent_antigravity_lane_key_until_monotonic_by_key: dict[str, float] = {}
_codex_auto_agent_antigravity_lane_key_by_key: dict[str, str] = {}
_codex_auto_agent_lane_state_cache_lock = asyncio.Lock()
_AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT = "aawm-routing-v1"
_AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX = "aawm:alias-routing"


def _get_aawm_alias_routing_state_namespace() -> str:
    raw = _clean_codex_auth_value(
        os.getenv("AAWM_ALIAS_ROUTING_STATE_NAMESPACE")
    )
    if raw is not None:
        return raw
    return _AAWM_ALIAS_ROUTING_STATE_NAMESPACE_DEFAULT


def _build_aawm_alias_routing_durable_cache_key(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> str:
    namespace = _get_aawm_alias_routing_state_namespace()
    normalized_family = alias_family.strip().lower()
    normalized_kind = state_kind.strip().lower()
    return (
        f"{_AAWM_ALIAS_ROUTING_STATE_KEY_PREFIX}:{namespace}:"
        f"{normalized_family}:{normalized_kind}:{state_key}"
    )


def _get_aawm_alias_routing_dual_cache() -> Optional[Any]:
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj
    except Exception:
        return None
    if proxy_logging_obj is None:
        return None
    internal_usage_cache = getattr(proxy_logging_obj, "internal_usage_cache", None)
    if internal_usage_cache is None:
        return None
    dual_cache = getattr(internal_usage_cache, "dual_cache", None)
    if dual_cache is None or getattr(dual_cache, "redis_cache", None) is None:
        return None
    return dual_cache


def _parse_aawm_alias_routing_durable_expiry(
    payload: Any,
) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    expires_at = payload.get("expires_at_epoch")
    if not isinstance(expires_at, (int, float)):
        return None
    if float(expires_at) <= time.time():
        return None
    return float(expires_at)


async def _read_aawm_alias_routing_durable_payload(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
) -> Optional[dict[str, Any]]:
    dual_cache = _get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None
    cache_key = _build_aawm_alias_routing_durable_cache_key(
        alias_family=alias_family,
        state_kind=state_kind,
        state_key=state_key,
    )
    try:
        payload = await dual_cache.async_get_cache(key=cache_key)
    except Exception:
        verbose_proxy_logger.warning(
            "AAWM alias routing durable read failed for family=%s kind=%s",
            alias_family,
            state_kind,
            exc_info=True,
        )
        return None
    if not isinstance(payload, dict):
        return None
    if _parse_aawm_alias_routing_durable_expiry(payload) is None:
        return None
    return dict(payload)


async def _write_aawm_alias_routing_durable_payload(
    *,
    alias_family: str,
    state_kind: str,
    state_key: str,
    payload: dict[str, Any],
    ttl_seconds: float,
) -> bool:
    dual_cache = _get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return False
    cache_key = _build_aawm_alias_routing_durable_cache_key(
        alias_family=alias_family,
        state_kind=state_kind,
        state_key=state_key,
    )
    durable_payload = dict(payload)
    durable_payload["expires_at_epoch"] = (
        time.time() + max(0.0, float(ttl_seconds))
    )
    try:
        await dual_cache.async_set_cache(
            key=cache_key,
            value=durable_payload,
            ttl=max(1.0, float(ttl_seconds)),
        )
        return True
    except Exception:
        verbose_proxy_logger.warning(
            "AAWM alias routing durable write failed for family=%s kind=%s",
            alias_family,
            state_kind,
            exc_info=True,
        )
        return False


def _hydrate_aawm_alias_routing_cooldown_memory(
    *,
    memory_map: dict[str, float],
    cooldown_key: str,
    expires_at_epoch: float,
) -> None:
    remaining = max(0.0, float(expires_at_epoch) - time.time())
    if remaining <= 0:
        return
    until = time.monotonic() + remaining
    current_until = memory_map.get(cooldown_key, 0.0)
    if until > current_until:
        memory_map[cooldown_key] = until


def _hydrate_aawm_alias_routing_affinity_memory(
    *,
    memory_map: dict[str, dict[str, Any]],
    session_key: str,
    payload: dict[str, Any],
    expires_at_epoch: float,
) -> dict[str, Any]:
    remaining = max(0.0, float(expires_at_epoch) - time.time())
    if remaining <= 0:
        return {}
    affinity = {
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "route_family": payload.get("route_family"),
        "last_resort": bool(payload.get("last_resort")),
        "expires_at_monotonic": time.monotonic() + remaining,
    }
    memory_map[session_key] = affinity
    return dict(affinity)

_GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING = (
    "# Preserved Project And Safety Instructions"
)
_GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING = (
    "# Original Claude System Instructions"
)
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
_OPENAI_ADAPTER_SYSTEM_REMINDER_INLINE_PATTERN = re.compile(
    r"<system-reminder>\n.*?</system-reminder>\n?",
    re.DOTALL,
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
_ANTHROPIC_RESPONSES_ADAPTER_ENDPOINTS = frozenset(
    {"/messages", "/v1/messages"}
)
_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.3-codex-spark",
    }
)
_ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "deepseek-ai/deepseek-v3.1-terminus",
        "deepseek-ai/deepseek-v3.2",
        "minimaxai/minimax-m2.7",
        "mistralai/devstral-2-123b-instruct-2512",
        "z-ai/glm4.7",
    }
)
_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "openrouter/free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "minimax/minimax-m2.5:free",
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
        "gpt-oss-20b:free",
        "gpt-oss-120b:free",
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen3.6-flash",
        "qwen/qwen3-coder:free",
    }
)
_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "cohere/north-mini-code:free",
        "deepseek/deepseek-v4-flash:free",
        "openrouter/elephant-alpha",
        "inclusionai/ling-2.6-flash",
        "owl-alpha",
    }
)
_ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    "gemini-3.1",
    "gemini-3-flash-preview",
)
_CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    "gemini-3.1",
    "gemini-3-flash-preview",
)
_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER = "antigravity"
_ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "chat_20706",
        "chat_23310",
        "claude-opus-4-6-thinking",
        "claude-sonnet-4-6",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-thinking",
        "gemini-2.5-pro",
        "gemini-3-flash",
        "gemini-3-flash-agent",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-high",
        "gemini-3.1-pro-low",
        "gemini-3.5-flash-extra-low",
        "gemini-3.5-flash-low",
        "gemini-pro-agent",
        "gpt-oss-120b-medium",
        "tab_flash_lite_preview",
        "tab_jump_flash_lite_preview",
    }
)
_CODEX_GOOGLE_CODE_ASSIST_DEFAULT_MAX_TOKENS = 8192
_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE = 2048
_codex_google_code_assist_tool_call_name_cache: dict[str, str] = {}
_codex_google_code_assist_tool_call_arguments_cache: dict[str, str] = {}
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
_ANTHROPIC_ADAPTER_NVIDIA_RETRYABLE_STATUS_CODES = frozenset(
    {408, 429, 500, 502, 503, 504}
)
_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES = sorted(
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES - {429}
)
_ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS = (
    "LITELLM_CODEX_AUTH_FILE",
    "CHATGPT_AUTH_FILE",
)
_ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS = (
    "LITELLM_CODEX_TOKEN_DIR",
    "CHATGPT_TOKEN_DIR",
)
_ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS = (
    "/home/zepfu/.codex/auth.json",
    "~/.codex/auth.json",
    "~/.config/litellm/chatgpt/auth.json",
)
_ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS = (
    "LITELLM_GEMINI_AUTH_FILE",
    "GEMINI_OAUTH_CREDS_FILE",
)
_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS = (
    "/home/zepfu/.gemini/oauth_creds.json",
    "~/.gemini/oauth_creds.json",
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS = (
    "LITELLM_GEMINI_OAUTH_CLIENT_ID",
    "GEMINI_OAUTH_CLIENT_ID",
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS = (
    "LITELLM_GEMINI_OAUTH_CLIENT_SECRET",
    "GEMINI_OAUTH_CLIENT_SECRET",
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS = (
    "LITELLM_GEMINI_CLI_BUNDLE_PATH",
    "GEMINI_CLI_BUNDLE_PATH",
)
_ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS = (
    "/home/zepfu/.nvm/versions/node/*/lib/node_modules/@google/gemini-cli/bundle",
    "~/.nvm/versions/node/*/lib/node_modules/@google/gemini-cli/bundle",
)
_ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN = re.compile(
    r'OAUTH_CLIENT_ID\s*=\s*"(?P<value>[^"]+)"'
)
_ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN = re.compile(
    r'OAUTH_CLIENT_SECRET\s*=\s*"(?P<value>[^"]+)"'
)
_ANTIGRAVITY_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN = re.compile(
    r"(?P<value>[0-9]+-[A-Za-z0-9_-]+\.apps\.googleusercontent\.com)"
)
_ANTIGRAVITY_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN = re.compile(
    r"(?P<value>GOCSPX-[A-Za-z0-9_-]+)"
)
_CLAUDE_AGENT_SPEC_DIR_ENV_VARS = (
    "LITELLM_CLAUDE_AGENTS_DIR",
    "CLAUDE_AGENTS_DIR",
)
_CLAUDE_AGENT_SPEC_DEFAULT_DIRS = (
    "/home/zepfu/.claude/agents",
    "~/.claude/agents",
)
_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR = (
    Path(__file__).resolve().parents[3] / "context-replacement" / "claude-code"
)
_CLAUDE_AUTO_MEMORY_TEMPLATE_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR / "2.1.110" / "auto-memory-replacement.md"
)
_CLAUDE_PROMPT_PATCH_MANIFEST_PATH = (
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR
    / "prompt-patches"
    / "roman01la-2026-04-02.json"
)
_CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION = (2, 1, 110)
_CLAUDE_AUTO_MEMORY_SECTION_PATTERN = re.compile(
    r"(?ms)^# auto memory\n.*?(?=^# Environment\b|\Z)"
)
_CLAUDE_TYPES_XML_BLOCK_PATTERN = re.compile(r"<types>\n.*?\n</types>", re.DOTALL)
_CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")
_CLAUDE_CC_VERSION_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
_AAWM_DYNAMIC_DIRECTIVE_PATTERN = re.compile(
    r"<!--\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<html_attrs>.*?)\s*-->"
    r"|@@@\s*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<at_attrs>.*?)\s*@@@"
    r"|^[ \t]*AAWM(?=[ \t]+(?:p|proc)=)\s+(?P<line_attrs>[^\r\n]+?)\s*$",
    re.DOTALL | re.MULTILINE,
)
_AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_-]*)='
    r'(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\'|(?P<bare>[^\s]+))'
)
_CLAUDE_AGENT_TENANT_PATTERN = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project\b"
)
_CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS: tuple[tuple[str, str], ...] = (
    ("MEMORY.md", "memory-md"),
    ("CLAUDE.md", "claude-md"),
)
_AAWM_AGENT_MEMORY_PROC_NAME = "get_agent_memories"
_AAWM_DYNAMIC_PROC_ALIASES = {"get_agent_memory": _AAWM_AGENT_MEMORY_PROC_NAME}
_AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS: dict[str, tuple[str, ...]] = {
    _AAWM_AGENT_MEMORY_PROC_NAME: ("agent", "tenant"),
}
_AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE = (
    "## AAWM Injection Status\n\n"
    'AAWM "{proc_name}" failed for this session.\n'
    "Alert the user or session orchestrator.\n"
)
_AAWM_NO_MEMORIES_TEMPLATE = (
    "# Memory Injection\n"
    "You have saved no memories as of yet.\n"
)
_AAWM_DB_HOST_ENV_VARS = (
    "AAWM_DB_HOST",
    "AAWM_POSTGRES_SERVER",
    "POSTGRES_SERVER",
    "PGHOST",
)
_AAWM_DB_PORT_ENV_VARS = (
    "AAWM_DB_PORT",
    "AAWM_POSTGRES_PORT",
    "POSTGRES_PORT",
    "PGPORT",
)
_AAWM_DB_USER_ENV_VARS = (
    "AAWM_DB_USER",
    "AAWM_POSTGRES_USER",
    "POSTGRES_USER",
    "PGUSER",
)
_AAWM_DB_PASSWORD_ENV_VARS = (
    "AAWM_DB_PASSWORD",
    "AAWM_DB_PWD",
    "AAWM_POSTGRES_PASSWORD",
    "AAWM_POSTGRES_PWD",
    "POSTGRES_PASSWORD",
    "POSTGRES_PWD",
    "PGPASSWORD",
)
_AAWM_DB_NAME_ENV_VARS = (
    "AAWM_DB_NAME",
    "AAWM_POSTGRES_DATABASE",
    "POSTGRES_DATABASE",
    "PGDATABASE",
)
_AAWM_DB_SSLMODE_ENV_VARS = (
    "AAWM_DB_SSLMODE",
    "AAWM_POSTGRES_SSLMODE",
    "POSTGRES_SSLMODE",
    "PGSSLMODE",
)
_AAWM_DB_SSL_BOOL_ENV_VARS = (
    "AAWM_DB_SSL",
    "AAWM_POSTGRES_SSL",
    "POSTGRES_SSL",
)
_AAWM_DB_URL_ENV_VARS = (
    "AAWM_DB_URL",
    "AAWM_DATABASE_URL",
    "AAWM_POSTGRES_URL",
)
_AAWM_DB_APPLICATION_NAME_ENV_VARS = (
    "AAWM_DYNAMIC_INJECTION_DB_APPLICATION_NAME",
    "AAWM_DB_APPLICATION_NAME",
    "AAWM_POSTGRES_APPLICATION_NAME",
    "PGAPPNAME",
)
_AAWM_DYNAMIC_INJECTION_APPLICATION_NAME = "aawm-litellm-dynamic-injection"
_aawm_dynamic_injection_pool: Optional[Any] = None
_aawm_dynamic_injection_pool_lock = asyncio.Lock()
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}
_claude_agent_model_cache: dict[Path, tuple[Optional[int], Optional[str]]] = {}
_google_oauth_access_token_cache: dict[str, tuple[str, int]] = {}
_google_oauth_access_token_lock = asyncio.Lock()
_antigravity_oauth_access_token_cache: dict[str, tuple[str, int]] = {}
_antigravity_oauth_access_token_lock = asyncio.Lock()
_google_code_assist_project_cache: dict[str, str] = {}
_google_code_assist_project_lock = asyncio.Lock()
_google_code_assist_prime_until_monotonic_by_key: dict[str, float] = {}
_google_code_assist_prime_quota_by_key: dict[str, dict[str, Any]] = {}
_google_code_assist_prime_lock = asyncio.Lock()
_google_adapter_semaphores: dict[tuple[str, int], asyncio.Semaphore] = {}
_google_adapter_rate_limit_lock = asyncio.Lock()
_google_adapter_rate_limit_until_monotonic_by_key: dict[str, float] = {}
_google_adapter_user_prompt_turn_lock = asyncio.Lock()
_google_adapter_user_prompt_turn_counters: dict[str, int] = {}
_openrouter_adapter_rate_limit_lock = asyncio.Lock()
_openrouter_adapter_rate_limit_until_monotonic_by_key: dict[str, float] = {}
_openrouter_adapter_failure_circuit_until_monotonic_by_key: dict[str, float] = {}
_CODEX_SPAWN_AGENT_TOOL_NAME = "spawn_agent"
_CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE = "tool_search"
_CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID = "spawn-agent-fanout-policy"
_CODEX_SPAWN_AGENT_PAYLOAD_SCHEMA_PATCH_ID = "spawn-agent-payload-schema"
_CODEX_CORE_TOOL_GUIDANCE_PATCH_PREFIX = "core-tool-guidance"
_CODEX_UNSUPPORTED_HOSTED_TOOLS_MODEL_INFO_FIELD = "unsupported_hosted_tools"
_CODEX_UNSUPPORTED_REQUEST_PARAMS_MODEL_INFO_FIELD = "unsupported_request_params"
_CODEX_UNSUPPORTED_INPUT_ITEM_TYPES_MODEL_INFO_FIELD = "unsupported_input_item_types"
_CODEX_SPAWN_AGENT_FANOUT_POLICY = (
    "Use subagents to parallelize independent work while keeping one local owner "
    "on the critical path. Follow the current operator and project instructions "
    "that authorize fanout; do not treat generic depth or investigation wording "
    "as permission to launch unrelated autonomous fanout. Do not duplicate the "
    "same task across agents.\n\n"
    "For read-only or exploration workers, call multi_agent_v1.spawn_agent with "
    'lower-case payload fields: model="aawm-codex-agent-auto", '
    "fork_context=false unless context sharing is explicitly needed, and message "
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
    "model": {
        "type": "string",
        "description": (
            "Optional lower-case model override accepted by the orchestrator. "
            "Use aawm-codex-agent-auto for read-only/exploration workers; use "
            "the selected coding model for coding workers."
        ),
    },
    "fork_context": {
        "type": "boolean",
        "description": (
            "Whether to fork the current conversation context into the worker. "
            "Use false for isolated read-only audits unless context sharing is "
            "explicitly required."
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
    "model",
    "fork_context",
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


def _get_openai_passthrough_route_family(endpoint: str) -> str:
    normalized_path = httpx.URL(endpoint).path.rstrip("/")
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    if _is_openai_responses_endpoint(endpoint):
        return "openai_responses"
    if normalized_path in {"/chat/completions", "/v1/chat/completions"}:
        return "openai_chat_completions"
    return "openai_passthrough"


def _is_oa_xai_request_body(request_body: dict[str, Any]) -> bool:
    return is_oa_xai_model(request_body.get("model"))


def _is_grok_native_oauth_request_body(request_body: dict[str, Any]) -> bool:
    return is_grok_native_oauth_model(request_body.get("model"))


@lru_cache(maxsize=1)
def _load_local_model_metadata() -> dict[str, Any]:
    model_metadata_path = (
        Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
    )
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


def _is_oa_xai_responses_model(model: Any) -> bool:
    if not is_oa_xai_model(model):
        return False

    candidate_models = [model]
    try:
        candidate_models.append(resolve_oa_xai_upstream_model(cast(str, model)))
    except Exception:
        pass

    for candidate_model in candidate_models:
        model_info = _get_model_metadata_entry(candidate_model)
        if isinstance(model_info, dict) and model_info.get("mode") == "responses":
            return True
    return False


def _to_xai_native_passthrough_model(model: Any) -> Any:
    if isinstance(model, str) and model.startswith("xai/"):
        return model[len("xai/") :]
    return model


def _xai_responses_sanitized_tool_changes(
    original_tools: Any,
    sanitized_tools: Any,
) -> list[dict[str, Any]]:
    if not isinstance(original_tools, list) or not isinstance(sanitized_tools, list):
        return []

    tool_changes: list[dict[str, Any]] = []
    for index, original_tool in enumerate(original_tools):
        sanitized_tool = (
            sanitized_tools[index] if index < len(sanitized_tools) else None
        )
        if original_tool == sanitized_tool:
            continue

        change: dict[str, Any] = {"index": index}
        if isinstance(original_tool, dict):
            tool_type = _get_openai_tool_type(original_tool)
            if tool_type:
                change["type"] = tool_type
            if isinstance(sanitized_tool, dict):
                removed_fields = [
                    key for key in original_tool.keys() if key not in sanitized_tool
                ]
                if removed_fields:
                    change["removed_fields"] = sorted(removed_fields)
        elif isinstance(original_tool, str):
            change["type"] = original_tool

        tool_changes.append(change)
    return tool_changes


def _sanitize_xai_responses_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    sanitized_body = XAIResponsesAPIConfig().map_openai_params(
        cast(ResponsesAPIOptionalRequestParams, request_body),
        model=str(request_body.get("model") or ""),
        drop_params=True,
    )
    removed_params = [
        key
        for key in request_body.keys()
        if key not in sanitized_body and key != "litellm_metadata"
    ]
    tool_changes = _xai_responses_sanitized_tool_changes(
        request_body.get("tools"),
        sanitized_body.get("tools"),
    )
    decoded_previous_response_id = False
    previous_response_id = sanitized_body.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        decoded = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
            previous_response_id
        )
        if decoded != previous_response_id:
            sanitized_body = dict(sanitized_body)
            sanitized_body["previous_response_id"] = decoded
            decoded_previous_response_id = True

    if not removed_params and not tool_changes and not decoded_previous_response_id:
        return request_body, [], []

    tool_types = _dedupe_sorted_str_list(
        [
            tool_change["type"]
            for tool_change in tool_changes
            if isinstance(tool_change.get("type"), str)
        ]
    )
    normalized_removed_params = _dedupe_sorted_str_list(
        [
            normalized
            for param in removed_params
            if (normalized := _normalize_low_cardinality_tag_value(param))
        ]
    )
    updated_body = _merge_litellm_metadata(
        sanitized_body,
        tags_to_add=[
            "xai-responses-request-sanitized",
            *(
                ["xai-responses-previous-response-id-decoded"]
                if decoded_previous_response_id
                else []
            ),
            *(
                f"xai-responses-removed-param:{param}"
                for param in normalized_removed_params
            ),
            *(f"xai-responses-sanitized-tool:{tool}" for tool in tool_types),
        ],
        extra_fields={
            "xai_responses_request_sanitized": True,
            "xai_responses_sanitized_removed_params": normalized_removed_params,
            "xai_responses_sanitized_tool_count": len(tool_changes),
            "xai_responses_sanitized_tool_types": tool_types,
            "xai_responses_sanitized_tools": tool_changes,
            "xai_responses_previous_response_id_decoded": (
                decoded_previous_response_id
            ),
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="xai.responses_request_sanitized",
                    metadata={
                        "removed_params": normalized_removed_params,
                        "tool_count": len(tool_changes),
                        "tool_types": tool_types,
                        "previous_response_id_decoded": (
                            decoded_previous_response_id
                        ),
                    },
                )
            ],
        },
    )
    return updated_body, removed_params, tool_changes


def _sanitize_xai_responses_request_body_in_place(
    request_body: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    updated_body, removed_params, tool_changes = _sanitize_xai_responses_request_body(
        request_body
    )
    if updated_body is not request_body:
        request_body.clear()
        request_body.update(updated_body)
    return removed_params, tool_changes


async def _prepare_oa_xai_passthrough_request(
    request_body: dict[str, Any],
    *,
    sanitize_responses_request: bool = False,
) -> tuple[bool, Optional[str], Optional[str]]:
    if is_oa_xai_model(request_body.get("model")) and not isinstance(
        request_body.get("litellm_metadata"), dict
    ):
        request_body["litellm_metadata"] = {}
    prepared = await prepare_oa_xai_request(request_body)
    if not prepared:
        return False, None, None

    if sanitize_responses_request:
        updated_body, _xai_unsupported_hosted_tools = (
            _drop_unsupported_codex_hosted_tools_from_request_body(request_body)
        )
        if updated_body is not request_body:
            request_body.clear()
            request_body.update(updated_body)
        updated_body, _xai_unsupported_request_params = (
            _drop_unsupported_codex_request_params_from_request_body(request_body)
        )
        if updated_body is not request_body:
            request_body.clear()
            request_body.update(updated_body)
        updated_body, _xai_unsupported_input_items = (
            _drop_unsupported_codex_input_items_from_request_body(request_body)
        )
        if updated_body is not request_body:
            request_body.clear()
            request_body.update(updated_body)
        _sanitize_xai_responses_request_body_in_place(request_body)
        updated_body, _removed_tool_choice = (
            _drop_tool_choice_without_tools_from_request_body(request_body)
        )
        if updated_body is not request_body:
            request_body.clear()
            request_body.update(updated_body)

    api_base = request_body.pop("api_base", None)
    api_key = request_body.pop("api_key", None)
    request_body.pop("custom_llm_provider", None)
    return (
        True,
        api_base if isinstance(api_base, str) and api_base.strip() else None,
        api_key if isinstance(api_key, str) and api_key.strip() else None,
    )


def _get_grok_native_oauth_client_version() -> str:
    return (
        get_secret_str("LITELLM_XAI_GROK_CLIENT_VERSION")
        or get_secret_str("GROK_CLIENT_VERSION")
        or "0.1.210"
    )


def _get_grok_native_oauth_session_id(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()

    for header_name in (
        "x-grok-session-id",
        "session_id",
        "x-session-id",
        "x-grok-conv-id",
    ):
        header_value = _get_case_insensitive_header(
            _safe_get_request_headers(request),
            header_name,
        )
        if header_value:
            return header_value
    return None


def _get_grok_native_oauth_request_id(request: Request) -> str:
    for header_name in ("x-grok-req-id", "x-request-id", "request_id"):
        header_value = _get_case_insensitive_header(
            _safe_get_request_headers(request),
            header_name,
        )
        if header_value:
            return header_value
    return str(uuid4())


def _build_grok_native_oauth_headers(
    *,
    access_token: str,
    model: str,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    client_version = _get_grok_native_oauth_client_version()
    request_id = _get_grok_native_oauth_request_id(request)
    headers: dict[str, Any] = {
        "accept": "application/json",
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "user-agent": (
            get_secret_str("LITELLM_XAI_GROK_USER_AGENT")
            or f"grok/{client_version}"
        ),
        "x-grok-client-identifier": (
            get_secret_str("LITELLM_XAI_GROK_CLIENT_IDENTIFIER") or "grok-cli"
        ),
        "x-grok-client-version": client_version,
        "x-grok-model-override": model,
        "x-grok-req-id": request_id,
        "x-request-id": request_id,
        "x-xai-token-auth": (
            get_secret_str("LITELLM_XAI_GROK_XAI_TOKEN_AUTH") or "xai-grok-cli"
        ),
    }
    session_id = _get_grok_native_oauth_session_id(
        request=request,
        request_body=request_body,
    )
    if session_id:
        headers["x-grok-session-id"] = session_id
    return headers


def _add_grok_native_oauth_metadata(
    request_body: dict[str, Any],
    *,
    model: str,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = build_grok_native_oauth_metadata(model)
    metadata_tags = metadata.pop("tags", [])
    existing_litellm_metadata = request_body.get("litellm_metadata")
    preserved_route_family: Optional[str] = None
    if isinstance(existing_litellm_metadata, dict):
        for route_family_key in ("passthrough_route_family", "route_family"):
            route_family_value = existing_litellm_metadata.get(route_family_key)
            if isinstance(route_family_value, str) and route_family_value.strip():
                preserved_route_family = route_family_value.strip()
                break

    merged_extra_fields = {
        **metadata,
        **(extra_fields or {}),
    }
    if preserved_route_family:
        merged_extra_fields.setdefault(
            "source_passthrough_route_family", preserved_route_family
        )
        merged_extra_fields.setdefault("source_route_family", preserved_route_family)
        merged_extra_fields["grok_cli_chat_proxy_used"] = True
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=[
            *(metadata_tags if isinstance(metadata_tags, list) else []),
            *(tags_to_add or []),
        ],
        extra_fields=merged_extra_fields,
    )


async def _prepare_grok_native_oauth_passthrough_request(
    request_body: dict[str, Any],
    *,
    request: Request,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> tuple[bool, Optional[str], dict[str, Any], dict[str, Any]]:
    model = normalize_grok_native_oauth_model(request_body.get("model"))
    if model is None:
        return False, None, {}, request_body

    prepared_body = dict(request_body)
    prepared_body["model"] = model
    prepared_body = _add_grok_native_oauth_metadata(
        prepared_body,
        model=model,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )
    prepared_body, _grok_unsupported_hosted_tools = (
        _drop_unsupported_codex_hosted_tools_from_request_body(prepared_body)
    )
    prepared_body, _grok_unsupported_request_params = (
        _drop_unsupported_codex_request_params_from_request_body(prepared_body)
    )
    prepared_body, _grok_unsupported_input_items = (
        _drop_unsupported_codex_input_items_from_request_body(prepared_body)
    )
    _sanitize_xai_responses_request_body_in_place(prepared_body)
    prepared_body, _removed_tool_choice = (
        _drop_tool_choice_without_tools_from_request_body(prepared_body)
    )
    access_token = await get_grok_native_oauth_access_token()
    headers = _build_grok_native_oauth_headers(
        access_token=access_token,
        model=model,
        request=request,
        request_body=prepared_body,
    )
    return True, _get_grok_passthrough_target_base(), headers, prepared_body


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
        headers.get("authorization")
        or headers.get("Authorization")
        or headers.get("api-key")
        or headers.get("Api-Key")
    )


def _get_request_header_or_passthrough_alias(
    request: Request, header_name: str
) -> Optional[str]:
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


def _normalize_anthropic_adapter_model_name(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized_model = model.strip()
    return normalized_model or None


def _split_anthropic_adapter_provider_prefix(model: Any) -> tuple[Optional[str], Optional[str]]:
    normalized_model = _normalize_anthropic_adapter_model_name(model)
    if normalized_model is None:
        return None, None
    if "/" not in normalized_model:
        return None, normalized_model

    prefix, remainder = normalized_model.split("/", 1)
    provider = {
        "agy": "antigravity",
        "chatgpt": "openai",
        "gemini": "google",
        "google-antigravity": "antigravity",
        "nvidia_nim": "nvidia",
        "opencode": _OPENCODE_ZEN_PROVIDER,
        "opencode-zen": _OPENCODE_ZEN_PROVIDER,
        "zen": _OPENCODE_ZEN_PROVIDER,
    }.get(
        prefix,
        prefix
        if prefix
        in (
            "openai",
            "google",
            "openrouter",
            "nvidia",
            "antigravity",
            _OPENCODE_ZEN_PROVIDER,
        )
        else None,
    )
    if provider is None:
        return None, normalized_model
    return provider, remainder.strip()


def _get_anthropic_adapter_model_candidates(request_body: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    requested_model = _normalize_anthropic_adapter_model_name(request_body.get("model"))
    if requested_model is not None:
        candidates.append(requested_model)

    agent_name, _tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if not agent_name:
        return candidates

    agent_model = _normalize_anthropic_adapter_model_name(
        _load_claude_agent_declared_model(agent_name)
    )
    if agent_model is not None:
        candidates.append(agent_model)
    return candidates


def _has_anthropic_responses_adapter_endpoint(endpoint: str) -> bool:
    normalized_endpoint = endpoint.strip()
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = f"/{normalized_endpoint}"
    return normalized_endpoint in _ANTHROPIC_RESPONSES_ADAPTER_ENDPOINTS


def _normalize_anthropic_openai_responses_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in (None, "openai") or candidate is None:
        return None
    if candidate in _ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS:
        return candidate
    return None


def _normalize_anthropic_nvidia_responses_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in (None, "nvidia") or candidate is None:
        return None

    requested_model = model.strip() if isinstance(model, str) else ""
    has_explicit_nvidia_prefix = requested_model.startswith("nvidia/")
    normalized_candidate = candidate.strip()
    nvidia_model_aliases = {
        "minimax/minimax-m2.7": "minimaxai/minimax-m2.7",
    }
    normalized_candidate = nvidia_model_aliases.get(
        normalized_candidate, normalized_candidate
    )
    is_openrouter_namespace_model = (
        requested_model in _ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS
    )
    if has_explicit_nvidia_prefix and not is_openrouter_namespace_model:
        return normalized_candidate or None
    if normalized_candidate in _ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS:
        return normalized_candidate
    return None


def _normalize_anthropic_openrouter_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    normalized_candidate = (
        candidate
        if explicit_provider == "openrouter"
        else _normalize_anthropic_adapter_model_name(model)
    )
    if normalized_candidate is None:
        return None

    openrouter_model_aliases = {
        "free": "openrouter/free",
        "elephant-alpha": "openrouter/elephant-alpha",
        "meta-llama/llama-3.3-70b-instructfree": (
            "meta-llama/llama-3.3-70b-instruct:free"
        ),
    }
    normalized_candidate = openrouter_model_aliases.get(
        normalized_candidate, normalized_candidate
    )
    return normalized_candidate or None


def _get_openrouter_completion_adapter_upstream_model(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider == "openrouter" and candidate is not None:
        candidate = candidate.strip()
        return candidate or None
    return _normalize_anthropic_adapter_model_name(model)


def _raise_openrouter_auto_agent_candidate_unavailable(message: str) -> None:
    exc = ProxyException(
        message=message,
        type="rate_limit_error",
        param="model",
        code=429,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise exc


def _normalize_opencode_zen_adapter_model_name(model: Any) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider != _OPENCODE_ZEN_PROVIDER or candidate is None:
        return None
    normalized_candidate = candidate.strip()
    if normalized_candidate in _OPENCODE_ZEN_FREE_MODELS:
        return normalized_candidate
    return None


def _normalize_anthropic_google_completion_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in (None, "google") or candidate is None:
        return None
    normalized_candidate = _normalize_google_completion_adapter_model_name(candidate)
    if normalized_candidate.startswith(
        _ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES
    ):
        return normalized_candidate
    return None


def _normalize_antigravity_code_assist_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in {
        "antigravity",
        "agy",
        "google-antigravity",
    } or candidate is None:
        return None
    normalized_candidate = candidate.strip()
    if normalized_candidate in _ANTIGRAVITY_CODE_ASSIST_ADAPTER_ALLOWED_MODELS:
        return normalized_candidate
    return None


def _normalize_codex_google_code_assist_adapter_model_name(
    model: Any,
) -> Optional[str]:
    if not isinstance(model, str):
        return None
    candidate = model.strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if lowered.startswith("openrouter/"):
        return None
    for prefix in ("google-code-assist/", "code-assist/"):
        if lowered.startswith(prefix):
            candidate = candidate.split("/", 1)[1]
            lowered = candidate.lower()
            break
    if lowered.startswith("codex-gemini-"):
        candidate = candidate[len("codex-") :]

    explicit_provider, split_candidate = _split_anthropic_adapter_provider_prefix(
        candidate
    )
    if explicit_provider not in (None, "google") or split_candidate is None:
        return None
    normalized_candidate = _normalize_google_completion_adapter_model_name(
        split_candidate
    )
    if normalized_candidate.startswith(
        _CODEX_GOOGLE_CODE_ASSIST_ADAPTER_ALLOWED_MODEL_PREFIXES
    ):
        return normalized_candidate
    return None


def _resolve_codex_opencode_zen_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_opencode_zen_adapter_model_name(request_body.get("model"))


def _resolve_anthropic_opencode_zen_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_opencode_zen_adapter_model_name(candidate)
        if normalized_model is not None:
            return normalized_model
    return None


def _resolve_anthropic_antigravity_code_assist_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    _ = endpoint
    return _normalize_antigravity_code_assist_adapter_model_name(
        request_body.get("model")
    )


def _resolve_codex_google_code_assist_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_codex_google_code_assist_adapter_model_name(
        request_body.get("model")
    )


def _resolve_codex_antigravity_code_assist_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_antigravity_code_assist_adapter_model_name(
        request_body.get("model")
    )


def _normalize_codex_auto_agent_alias_model(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    for alias in _CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS:
        if normalized == alias.lower():
            return alias
    return None


def _is_codex_auto_agent_alias_model(model: Any) -> bool:
    return _normalize_codex_auto_agent_alias_model(model) is not None


def _resolve_codex_auto_agent_alias_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_codex_auto_agent_alias_model(request_body.get("model"))


def _get_codex_auto_agent_header(
    headers: dict[str, Any], header_name: str
) -> Optional[str]:
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != header_name.lower():
            continue
        cleaned = _clean_codex_auth_value(value)
        if cleaned is not None:
            return cleaned
    return None


def _hash_codex_auto_agent_lane_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _resolve_codex_auto_agent_openai_lane_key(
    request: Request,
    *,
    include_session_fallback: bool = True,
) -> str:
    headers = _safe_get_request_headers(request)
    account_id = _get_codex_auto_agent_header(headers, "chatgpt-account-id")
    if account_id is not None:
        return f"chatgpt-account:{account_id}"
    authorization = _get_codex_auto_agent_header(headers, "authorization")
    if authorization is not None:
        return f"auth:{_hash_codex_auto_agent_lane_value(authorization)}"
    if include_session_fallback:
        session_header = (
            _get_codex_auto_agent_header(headers, "session_id")
            or _get_codex_auto_agent_header(headers, "session-id")
        )
        if session_header is not None:
            return f"session:{session_header}"
    return "__default__"


def _resolve_codex_auto_agent_openai_cooldown_lane_key(request: Request) -> str:
    return _resolve_codex_auto_agent_openai_lane_key(
        request,
        include_session_fallback=False,
    )


def _get_codex_auto_agent_lane_state_cache_ttl_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS")
    )
    if raw_value is None:
        return _CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS
    try:
        parsed = float(raw_value)
    except Exception:
        return _CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS
    return max(0.0, parsed)


def _get_codex_auto_agent_google_lane_cache_key() -> str:
    auth_path = _get_anthropic_adapter_google_auth_file_path()
    if auth_path is not None:
        return f"google-auth:{auth_path.expanduser()}"
    return "google-auth:__default__"


def _get_codex_auto_agent_antigravity_lane_cache_key() -> str:
    auth_path = _get_antigravity_auth_file_path()
    if auth_path is not None:
        return f"antigravity-auth:{auth_path.expanduser()}"
    return "antigravity-auth:__default__"


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
    cache_key = _get_codex_auto_agent_google_lane_cache_key()
    ttl_seconds = _get_codex_auto_agent_lane_state_cache_ttl_seconds()
    if ttl_seconds > 0:
        async with _codex_auto_agent_lane_state_cache_lock:
            cached_until = (
                _codex_auto_agent_google_lane_key_until_monotonic_by_key.get(
                    cache_key, 0.0
                )
            )
            if cached_until > time.monotonic():
                cached_lane_key = _codex_auto_agent_google_lane_key_by_key.get(
                    cache_key
                )
                if isinstance(cached_lane_key, str) and cached_lane_key:
                    return cached_lane_key

    try:
        google_access_token = await _load_valid_local_google_oauth_access_token()
        google_project = await _get_or_load_google_code_assist_project(
            google_access_token
        )
        lane_key = _get_google_adapter_rate_limit_key(
            None,
            access_token=google_access_token,
            companion_project=google_project,
        )
    except Exception:
        _invalidate_codex_auto_agent_google_lane_cache()
        verbose_proxy_logger.warning(
            "Codex auto-agent alias could not resolve Google Code Assist lane; using default lane",
            exc_info=True,
        )
        return "__default__"

    if ttl_seconds > 0:
        async with _codex_auto_agent_lane_state_cache_lock:
            _codex_auto_agent_google_lane_key_by_key[cache_key] = lane_key
            _codex_auto_agent_google_lane_key_until_monotonic_by_key[cache_key] = (
                time.monotonic() + ttl_seconds
            )
    return lane_key


def _is_codex_auto_agent_antigravity_auth_degraded_exception(exc: Any) -> bool:
    if not isinstance(exc, HTTPException):
        return False
    detail_text = str(getattr(exc, "detail", ""))
    return "Antigravity OAuth" in detail_text and (
        "expired or invalid" in detail_text
        or "does not contain" in detail_text
        or "sidecar owns Antigravity auth refresh" in detail_text
    )


def _log_codex_auto_agent_antigravity_auth_degraded(exc: HTTPException) -> None:
    global _codex_auto_agent_antigravity_auth_degraded_log_until_monotonic

    now = time.monotonic()
    if now < _codex_auto_agent_antigravity_auth_degraded_log_until_monotonic:
        return
    _codex_auto_agent_antigravity_auth_degraded_log_until_monotonic = (
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
            cached_until = (
                _codex_auto_agent_antigravity_lane_key_until_monotonic_by_key.get(
                    cache_key, 0.0
                )
            )
            if cached_until > time.monotonic():
                cached_lane_key = _codex_auto_agent_antigravity_lane_key_by_key.get(
                    cache_key
                )
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
            _codex_auto_agent_antigravity_lane_key_until_monotonic_by_key[cache_key] = (
                time.monotonic() + ttl_seconds
            )
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
    metadata_session_id = (
        metadata.get("session_id") if isinstance(metadata, dict) else None
    )
    session_id = _clean_codex_auth_value(metadata_session_id)
    headers = _safe_get_request_headers(request)
    if session_id is None:
        session_id = (
            _get_codex_auto_agent_header(headers, "session_id")
            or _get_codex_auto_agent_header(headers, "session-id")
        )
    if session_id is None:
        return None
    if alias_model == _CODEX_AUTO_AGENT_MODEL_ALIAS:
        return f"{session_id}:{_resolve_codex_auto_agent_openai_lane_key(request)}"
    return (
        f"{alias_model}:{session_id}:"
        f"{_resolve_codex_auto_agent_openai_lane_key(request)}"
    )


def _codex_auto_agent_candidate_key(
    candidate: dict[str, Any],
    lane_key: str,
) -> str:
    return "{}:{}:{}".format(
        candidate["provider"],
        candidate["model"],
        lane_key or "__default__",
    )


def _codex_auto_agent_candidate_public_shape(
    candidate: dict[str, Any],
    *,
    lane_key: Optional[str] = None,
    cooldown_seconds: Optional[float] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    shaped = {
        "provider": candidate["provider"],
        "model": candidate["model"],
        "route_family": candidate["route_family"],
        "last_resort": bool(candidate.get("last_resort")),
    }
    if lane_key is not None:
        shaped["lane_key"] = lane_key
    if cooldown_seconds is not None:
        shaped["cooldown_seconds"] = round(float(cooldown_seconds), 3)
    if reason is not None:
        shaped["reason"] = reason
    return shaped


def _auto_agent_alias_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        name = _normalize_auto_agent_alias_client_product(
            metadata.get("client_name")
        )
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
        value = _normalize_auto_agent_alias_client_product(
            _get_codex_auto_agent_header(headers, header_name)
        )
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
    if event_type == "no_candidate_available":
        return "Exhausted"
    if "auth_degraded" in candidate_status or "auth_degraded" in selection_reason:
        return "Degraded"
    if event.get("redispatch_required") or "cooldown" in candidate_status:
        return "Cooling Down"
    if failure_class in {"rate_limited", "capacity_exhausted", "transient_error"}:
        return "Cooling Down"
    if event.get("error_status_code") or failure_class:
        return "Failed"
    return None


def _auto_agent_alias_route_status_message(event: dict[str, Any]) -> str:
    parts: list[str] = []
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

    group_header_label = _clean_codex_auth_value(event.get("rollup_group_header_label"))
    incoming_endpoint = _clean_codex_auth_value(event.get("incoming_endpoint"))
    outgoing_target = _clean_codex_auth_value(event.get("outgoing_target")) or (
        _clean_codex_auth_value(event.get("route_family")) or "candidate_selection"
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
        )


def _emit_auto_agent_alias_route_event(
    event: dict[str, Any],
    *,
    level: str = "info",
) -> None:
    _record_auto_agent_alias_route_status_rollup(event)
    if not _should_emit_auto_agent_alias_route_event(event, level=level):
        return
    log_payload = {"event": "aawm_alias_route", **event}
    message = "AAWM_ALIAS_ROUTE: {}".format(
        json.dumps(log_payload, sort_keys=True, default=str, separators=(",", ":"))
    )
    if level == "warning":
        verbose_proxy_logger.warning(message)
    else:
        verbose_proxy_logger.info(message)


def _should_emit_auto_agent_alias_route_event(
    event: dict[str, Any],
    *,
    level: str = "info",
) -> bool:
    if level == "warning":
        return True

    if os.getenv("AAWM_ALIAS_ROUTE_LOG_HEALTHY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True

    if event.get("failure_class") or event.get("error_status_code"):
        return True
    if event.get("redispatch_required") or event.get("redispatch_threshold_crossed"):
        return True

    event_type = str(event.get("event_type") or "")
    candidate_status = str(event.get("candidate_status") or "")
    if event_type in {
        "candidate_attempt_started",
        "candidate_selected",
    }:
        return False
    if candidate_status in {"started", "selected"}:
        return False
    if event.get("selection_reason") == "session_affinity":
        return False
    return True


def _emit_auto_agent_alias_no_candidate_event(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    exc: HTTPException,
) -> None:
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    candidates = detail.get("candidates") if isinstance(detail, dict) else None
    repository = _extract_auto_agent_alias_metadata_value(
        request_body,
        "repository",
        "repo",
        "repo_name",
        "repository_name",
    )
    client_product_label = _extract_auto_agent_alias_client_product_label(
        request,
        request_body,
    )
    _emit_auto_agent_alias_route_event(
        {
            "observed_at": _format_auto_agent_alias_timestamp(
                datetime.now(timezone.utc)
            ),
            "alias_family": alias_family,
            "alias_model": alias_model,
            "session_id": _extract_auto_agent_alias_session_id(
                request, request_body
            ),
            "agent_id": _extract_auto_agent_alias_metadata_value(
                request_body,
                "agent_id",
                "aawm_agent_id",
                "codex_agent_id",
                "claude_agent_id",
            ),
            "repository": repository,
            "client_product_label": client_product_label,
            "rollup_group_header_label": build_aawm_route_rollup_group_header_label(
                repository=repository,
                client_product_label=client_product_label,
            ),
            "incoming_endpoint": _extract_auto_agent_alias_incoming_endpoint(request),
            "outgoing_target": "candidate_selection",
            "event_type": "no_candidate_available",
            "candidate_status": "all_candidates_unavailable",
            "failure_phase": "candidate_selection",
            "attempted_provider_call": False,
            "error_status_code": exc.status_code,
            "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
            "candidates": candidates if isinstance(candidates, list) else None,
        },
        level="warning",
    )


def _build_auto_agent_alias_audit_event(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    selection: dict[str, Any],
    candidate: dict[str, Any],
    event_type: str,
    candidate_status: str,
    attempt_number: Optional[int] = None,
    selected: bool = False,
    skipped: bool = False,
    selection_reason: Optional[str] = None,
    lane_key: Optional[str] = None,
    cooldown_key: Optional[str] = None,
    cooldown_seconds: Optional[Any] = None,
    cooldown_scope: Optional[str] = None,
    failure_class: Optional[str] = None,
    error_status_code: Optional[Any] = None,
    error_type: Optional[str] = None,
    error_code: Optional[Any] = None,
    error_tokens: Optional[Any] = None,
    retry_after_seconds: Optional[Any] = None,
    failure_phase: Optional[str] = None,
    attempted_provider_call: Optional[bool] = None,
    redispatch_required: bool = False,
) -> dict[str, Any]:
    normalized_cooldown_seconds = _auto_agent_alias_float(cooldown_seconds)
    if lane_key is None:
        lane_key = selection.get("lane_key")
    if cooldown_key is None and lane_key is not None:
        cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key)
    repository = _extract_auto_agent_alias_metadata_value(
        request_body,
        "repository",
        "repo",
        "repo_name",
        "repository_name",
    )
    client_product_label = _extract_auto_agent_alias_client_product_label(
        request,
        request_body,
    )
    event: dict[str, Any] = {
        "observed_at": _format_auto_agent_alias_timestamp(datetime.now(timezone.utc)),
        "alias_family": alias_family,
        "alias_model": alias_model,
        "session_id": _extract_auto_agent_alias_session_id(request, request_body),
        "agent_id": _extract_auto_agent_alias_metadata_value(
            request_body,
            "agent_id",
            "aawm_agent_id",
            "codex_agent_id",
            "claude_agent_id",
        ),
        "repository": repository,
        "client_product_label": client_product_label,
        "rollup_group_header_label": build_aawm_route_rollup_group_header_label(
            repository=repository,
            client_product_label=client_product_label,
        ),
        "incoming_endpoint": _extract_auto_agent_alias_incoming_endpoint(request),
        "outgoing_target": candidate.get("route_family"),
        "session_key": selection.get("session_key"),
        "provider": candidate.get("provider"),
        "model": candidate.get("model"),
        "route_family": candidate.get("route_family"),
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "attempt_number": attempt_number,
        "event_type": event_type,
        "selection_reason": selection_reason,
        "candidate_status": candidate_status,
        "failure_class": failure_class,
        "error_status_code": _auto_agent_alias_int(error_status_code),
        "error_type": error_type,
        "error_code": str(error_code) if error_code is not None else None,
        "retry_after_seconds": _auto_agent_alias_float(retry_after_seconds),
        "failure_phase": failure_phase,
        "attempted_provider_call": attempted_provider_call,
        "cooldown_scope": cooldown_scope,
        "cooldown_seconds": (
            round(normalized_cooldown_seconds, 3)
            if normalized_cooldown_seconds is not None
            else None
        ),
        "cooldown_until": _auto_agent_alias_cooldown_until(
            normalized_cooldown_seconds
        ),
        "selected": selected,
        "skipped": skipped,
        "last_resort": bool(candidate.get("last_resort")),
        "in_flight_session": bool(selection.get("in_flight_session")),
        "redispatch_required": redispatch_required,
        "redispatch_threshold_crossed": False,
    }
    if isinstance(error_tokens, list):
        event["error_tokens"] = error_tokens
    elif isinstance(error_tokens, set):
        event["error_tokens"] = sorted(error_tokens)
    return {key: value for key, value in event.items() if value is not None}


def _build_auto_agent_alias_audit_events(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    skipped_candidates = selection.get("skipped")
    if isinstance(skipped_candidates, list):
        for skipped_candidate in skipped_candidates:
            if not isinstance(skipped_candidate, dict):
                continue
            reason = str(skipped_candidate.get("reason") or "cooldown")
            event_type = (
                "candidate_skipped_provider_degraded"
                if reason == "auth_degraded"
                else "candidate_skipped_cooldown"
            )
            events.append(
                _build_auto_agent_alias_audit_event(
                    alias_family=alias_family,
                    alias_model=alias_model,
                    request=request,
                    request_body=request_body,
                    selection=selection,
                    candidate=skipped_candidate,
                    event_type=event_type,
                    candidate_status=f"skipped_{reason}",
                    selected=False,
                    skipped=True,
                    selection_reason=reason,
                    lane_key=skipped_candidate.get("lane_key"),
                    cooldown_seconds=skipped_candidate.get("cooldown_seconds"),
                    failure_phase=skipped_candidate.get("failure_phase"),
                    attempted_provider_call=skipped_candidate.get(
                        "attempted_provider_call"
                    ),
                )
            )

    audit_attempts = attempts
    if not audit_attempts and isinstance(selection.get("candidate"), dict):
        audit_attempts = [
            _codex_auto_agent_candidate_public_shape(
                selection["candidate"],
                lane_key=selection.get("lane_key"),
                reason=selection.get("selection_reason"),
            )
        ]

    for index, attempt in enumerate(audit_attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        status = str(attempt.get("status") or "").strip()
        failure_class = attempt.get("error_class")
        redispatch_required = status == "terminal_in_flight_cooldown_set"
        if redispatch_required:
            event_type = "redispatch_required"
        elif failure_class or status == "cooldown_set":
            event_type = "candidate_retryable_failure"
        else:
            event_type = "candidate_selected"
        events.append(
            _build_auto_agent_alias_audit_event(
                alias_family=alias_family,
                alias_model=alias_model,
                request=request,
                request_body=request_body,
                selection=selection,
                candidate=attempt,
                event_type=event_type,
                candidate_status=status or "selected",
                attempt_number=index,
                selected=True,
                skipped=False,
                selection_reason=attempt.get("reason")
                or selection.get("selection_reason"),
                lane_key=attempt.get("lane_key") or selection.get("lane_key"),
                cooldown_key=selection.get("cooldown_key")
                if index == len(audit_attempts)
                else None,
                cooldown_seconds=attempt.get("cooldown_seconds"),
                cooldown_scope=attempt.get("cooldown_scope"),
                failure_class=failure_class,
                error_status_code=attempt.get("error_status_code"),
                error_type=attempt.get("error_type"),
                error_code=attempt.get("error_code"),
                error_tokens=attempt.get("error_tokens"),
                retry_after_seconds=attempt.get("retry_after_seconds"),
                failure_phase=attempt.get("failure_phase"),
                attempted_provider_call=attempt.get("attempted_provider_call"),
                redispatch_required=redispatch_required,
            )
        )
    return events


def _codex_auto_agent_request_has_continuation_state(
    value: Any,
    _seen: Optional[set[int]] = None,
) -> bool:
    if isinstance(value, (dict, list)):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return False
        _seen.add(value_id)

    if isinstance(value, dict):
        for key in (
            "previous_response_id",
            "call_id",
            "tool_call_id",
            "item_id",
        ):
            if value.get(key):
                return True
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type in {
            "function_call",
            "function_call_output",
            "mcp_call",
            "mcp_approval_request",
            "mcp_approval_response",
            "reasoning",
            "tool_use",
            "tool_result",
        }:
            return True
        if value.get("role") == "tool":
            return True
        if value.get("tool_calls"):
            return True
        return any(
            _codex_auto_agent_request_has_continuation_state(child, _seen)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _codex_auto_agent_request_has_continuation_state(item, _seen)
            for item in value
        )
    return False


def _raise_codex_auto_agent_in_flight_cooldown(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_session_affinity_cooldown",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Codex auto-agent alias target is cooling down for an in-flight "
                    "session; provider switching is disabled for stateful agent "
                    "continuations. Redispatch a fresh agent attempt to re-run the "
                    "auto selector."
                ),
                "type": "rate_limit_error",
                "code": "aawm_codex_auto_agent_in_flight_provider_cooling_down",
            },
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(int(max(1.0, cooldown_seconds)))},
    )


def _raise_codex_auto_agent_redispatch_required(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
) -> None:
    retry_after_seconds = int(max(1.0, cooldown_seconds))
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_retryable_provider_exhaustion",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Codex auto-agent alias target hit retryable provider exhaustion "
                    "for an in-flight session. Do not continue this child agent. "
                    f"Redispatch a fresh subagent using model {alias_model} "
                    "so the auto selector can choose the next available candidate."
                ),
                "type": "rate_limit_error",
                "code": "aawm_codex_auto_agent_redispatch_required",
            },
            "redispatch_model": alias_model,
            "redispatch_reason": "in_flight_retryable_provider_exhaustion",
            "selected_provider": candidate.get("provider"),
            "selected_model": candidate.get("model"),
            "selected_route_family": candidate.get("route_family"),
            "cooldown_seconds": round(float(cooldown_seconds), 3),
            "retry_after_seconds": retry_after_seconds,
            "error_tokens": sorted(error_tokens),
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(retry_after_seconds)},
    )


async def _get_codex_auto_agent_active_cooldown_state(
    cooldown_key: str,
) -> tuple[float, str]:
    async with _codex_auto_agent_lock:
        now = time.monotonic()
        until = _codex_auto_agent_cooldown_until_monotonic_by_key.get(
            cooldown_key, 0.0
        )
        if until > now:
            return max(0.0, until - now), "memory"
        _codex_auto_agent_cooldown_until_monotonic_by_key.pop(
            cooldown_key, None
        )
    dual_cache = _get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return 0.0, "local_fallback"
    durable_payload = await _read_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="cooldown",
        state_key=cooldown_key,
    )
    if durable_payload is None:
        return 0.0, "local_fallback"
    expires_at_epoch = _parse_aawm_alias_routing_durable_expiry(durable_payload)
    if expires_at_epoch is None:
        return 0.0, "local_fallback"
    async with _codex_auto_agent_lock:
        _hydrate_aawm_alias_routing_cooldown_memory(
            memory_map=_codex_auto_agent_cooldown_until_monotonic_by_key,
            cooldown_key=cooldown_key,
            expires_at_epoch=expires_at_epoch,
        )
        until = _codex_auto_agent_cooldown_until_monotonic_by_key.get(
            cooldown_key, 0.0
        )
        return max(0.0, until - time.monotonic()), "durable_cache"


async def _get_codex_auto_agent_active_cooldown_seconds(
    cooldown_key: str,
) -> float:
    seconds, _ = await _get_codex_auto_agent_active_cooldown_state(cooldown_key)
    return seconds


async def _set_codex_auto_agent_cooldown(
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    ttl_seconds = max(0.0, float(cooldown_seconds))
    async with _codex_auto_agent_lock:
        until = time.monotonic() + ttl_seconds
        current_until = _codex_auto_agent_cooldown_until_monotonic_by_key.get(
            cooldown_key, 0.0
        )
        if until > current_until:
            _codex_auto_agent_cooldown_until_monotonic_by_key[cooldown_key] = until
    if ttl_seconds <= 0:
        return
    await _write_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="cooldown",
        state_key=cooldown_key,
        payload={"cooldown_key": cooldown_key},
        ttl_seconds=ttl_seconds,
    )


async def _get_codex_auto_agent_session_affinity(
    session_key: Optional[str],
) -> Optional[dict[str, Any]]:
    if session_key is None:
        return None
    async with _codex_auto_agent_lock:
        affinity = _codex_auto_agent_session_affinity_by_key.get(session_key)
        if isinstance(affinity, dict):
            expires_at = affinity.get("expires_at_monotonic", 0.0)
            if isinstance(expires_at, (int, float)) and expires_at > time.monotonic():
                hydrated = dict(affinity)
                hydrated["affinity_state_source"] = affinity.get(
                    "affinity_state_source", "memory"
                )
                return hydrated
            _codex_auto_agent_session_affinity_by_key.pop(session_key, None)
    dual_cache = _get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None
    durable_payload = await _read_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_key,
    )
    if durable_payload is None:
        return None
    expires_at_epoch = _parse_aawm_alias_routing_durable_expiry(durable_payload)
    if expires_at_epoch is None:
        return None
    async with _codex_auto_agent_lock:
        affinity = _hydrate_aawm_alias_routing_affinity_memory(
            memory_map=_codex_auto_agent_session_affinity_by_key,
            session_key=session_key,
            payload=durable_payload,
            expires_at_epoch=expires_at_epoch,
        )
        if not affinity:
            return None
        affinity["affinity_state_source"] = "durable_cache"
        return dict(affinity)


async def _set_codex_auto_agent_session_affinity(
    session_key: Optional[str],
    candidate: dict[str, Any],
) -> None:
    if session_key is None:
        return
    async with _codex_auto_agent_lock:
        _codex_auto_agent_session_affinity_by_key[session_key] = {
            "provider": candidate["provider"],
            "model": candidate["model"],
            "route_family": candidate["route_family"],
            "last_resort": bool(candidate.get("last_resort")),
            "expires_at_monotonic": (
                time.monotonic() + _CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS
            ),
            "affinity_state_source": "memory",
        }
    await _write_aawm_alias_routing_durable_payload(
        alias_family="codex",
        state_kind="affinity",
        state_key=session_key,
        payload={
            "provider": candidate["provider"],
            "model": candidate["model"],
            "route_family": candidate["route_family"],
            "last_resort": bool(candidate.get("last_resort")),
        },
        ttl_seconds=_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS,
    )


def _find_codex_auto_agent_candidate(
    provider: Any,
    model: Any,
    *,
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
) -> Optional[dict[str, Any]]:
    for candidate in _get_codex_auto_agent_candidates_for_alias(alias_model):
        if candidate["provider"] == provider and candidate["model"] == model:
            return dict(candidate)
    return None


def _build_auto_agent_skipped_candidates_from_states(
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for state in states:
        if state["cooldown_seconds"] <= 0:
            continue
        shaped = _codex_auto_agent_candidate_public_shape(
            state["candidate"],
            lane_key=state["lane_key"],
            cooldown_seconds=state["cooldown_seconds"],
            reason=state.get("skip_reason") or "cooldown",
        )
        for field in ("failure_phase", "attempted_provider_call"):
            if field in state:
                shaped[field] = state[field]
        skipped.append(shaped)
    return skipped


async def _apply_codex_auto_agent_forced_candidate_cooldown(
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    await _set_codex_auto_agent_cooldown(cooldown_key, cooldown_seconds)


async def _apply_anthropic_auto_agent_forced_candidate_cooldown(
    *,
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    await _set_anthropic_auto_agent_cooldown(cooldown_key, cooldown_seconds)


async def _build_codex_auto_agent_candidate_state(
    request: Request,
    *,
    candidate_template: dict[str, Any],
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
    openai_lane_key: Optional[str] = None,
    google_lane_key: Optional[str] = None,
    antigravity_lane_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    candidate = dict(candidate_template)
    if openai_lane_key is None:
        openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    forced_cooldown_seconds: Optional[float] = None
    skip_reason: Optional[str] = None
    cooldown_state_source_override: Optional[str] = None
    failure_phase: Optional[str] = None
    attempted_provider_call: Optional[bool] = None
    if candidate["provider"] == _CODEX_AUTO_AGENT_GOOGLE_PROVIDER:
        if google_lane_key is None:
            google_lane_key = await _resolve_codex_auto_agent_google_lane_key()
        lane_key = google_lane_key
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER:
        if antigravity_lane_state is None:
            antigravity_lane_state = (
                await _resolve_codex_auto_agent_antigravity_lane_state()
            )
        lane_key = str(antigravity_lane_state["lane_key"])
        forced_cooldown_seconds = _auto_agent_alias_float(
            antigravity_lane_state.get("forced_cooldown_seconds")
        )
        skip_reason = cast(Optional[str], antigravity_lane_state.get("skip_reason"))
        cooldown_state_source_override = cast(
            Optional[str], antigravity_lane_state.get("cooldown_state_source")
        )
        failure_phase = cast(Optional[str], antigravity_lane_state.get("failure_phase"))
        attempted_provider_call = cast(
            Optional[bool], antigravity_lane_state.get("attempted_provider_call")
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_XAI_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
    else:
        lane_key = openai_lane_key
    cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key)
    cooldown_seconds, cooldown_state_source = (
        await _get_codex_auto_agent_active_cooldown_state(cooldown_key)
    )
    if (
        forced_cooldown_seconds is not None
        and forced_cooldown_seconds > cooldown_seconds
    ):
        await _apply_codex_auto_agent_forced_candidate_cooldown(
            cooldown_key=cooldown_key,
            cooldown_seconds=forced_cooldown_seconds,
        )
        cooldown_seconds = forced_cooldown_seconds
        cooldown_state_source = (
            cooldown_state_source_override or "forced_candidate_cooldown"
        )
    state = {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_state_source": cooldown_state_source,
    }
    if skip_reason is not None:
        state["skip_reason"] = skip_reason
    if failure_phase is not None:
        state["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        state["attempted_provider_call"] = attempted_provider_call
    return state


async def _build_anthropic_auto_agent_candidate_state(
    request: Request,
    *,
    candidate_template: dict[str, Any],
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
    openai_lane_key: Optional[str] = None,
    anthropic_lane_key: Optional[str] = None,
    google_lane_key: Optional[str] = None,
    antigravity_lane_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    candidate = dict(candidate_template)
    if openai_lane_key is None:
        openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    if anthropic_lane_key is None:
        anthropic_lane_key = _resolve_anthropic_auto_agent_native_cooldown_lane_key(
            request
        )
    forced_cooldown_seconds: Optional[float] = None
    skip_reason: Optional[str] = None
    cooldown_state_source_override: Optional[str] = None
    failure_phase: Optional[str] = None
    attempted_provider_call: Optional[bool] = None
    if candidate["provider"] == _CODEX_AUTO_AGENT_GOOGLE_PROVIDER:
        if google_lane_key is None:
            google_lane_key = await _resolve_codex_auto_agent_google_lane_key()
        lane_key = google_lane_key
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER:
        if antigravity_lane_state is None:
            antigravity_lane_state = (
                await _resolve_codex_auto_agent_antigravity_lane_state()
            )
        lane_key = str(antigravity_lane_state["lane_key"])
        forced_cooldown_seconds = _auto_agent_alias_float(
            antigravity_lane_state.get("forced_cooldown_seconds")
        )
        skip_reason = cast(Optional[str], antigravity_lane_state.get("skip_reason"))
        cooldown_state_source_override = cast(
            Optional[str], antigravity_lane_state.get("cooldown_state_source")
        )
        failure_phase = cast(Optional[str], antigravity_lane_state.get("failure_phase"))
        attempted_provider_call = cast(
            Optional[bool], antigravity_lane_state.get("attempted_provider_call")
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENROUTER_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_XAI_LANE_KEY
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        lane_key = _CODEX_AUTO_AGENT_OPENCODE_LANE_KEY
    elif candidate["provider"] == _ANTHROPIC_AUTO_AGENT_NATIVE_PROVIDER:
        lane_key = anthropic_lane_key
    else:
        lane_key = openai_lane_key
    cooldown_key = _codex_auto_agent_candidate_key(candidate, lane_key)
    cooldown_seconds, cooldown_state_source = (
        await _get_anthropic_auto_agent_active_cooldown_state(cooldown_key)
    )
    if (
        forced_cooldown_seconds is not None
        and forced_cooldown_seconds > cooldown_seconds
    ):
        await _apply_anthropic_auto_agent_forced_candidate_cooldown(
            cooldown_key=cooldown_key,
            cooldown_seconds=forced_cooldown_seconds,
        )
        cooldown_seconds = forced_cooldown_seconds
        cooldown_state_source = (
            cooldown_state_source_override or "forced_candidate_cooldown"
        )
    state = {
        "candidate": candidate,
        "lane_key": lane_key,
        "cooldown_key": cooldown_key,
        "cooldown_seconds": cooldown_seconds,
        "cooldown_state_source": cooldown_state_source,
    }
    if skip_reason is not None:
        state["skip_reason"] = skip_reason
    if failure_phase is not None:
        state["failure_phase"] = failure_phase
    if attempted_provider_call is not None:
        state["attempted_provider_call"] = attempted_provider_call
    return state


async def _build_codex_auto_agent_candidate_states(
    request: Request,
    *,
    alias_model: str = _CODEX_AUTO_AGENT_MODEL_ALIAS,
) -> list[dict[str, Any]]:
    openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    google_lane_key: Optional[str] = None
    antigravity_lane_state: Optional[dict[str, Any]] = None
    states: list[dict[str, Any]] = []
    for candidate_template in _get_codex_auto_agent_candidates_for_alias(
        alias_model
    ):
        if (
            candidate_template["provider"] == _CODEX_AUTO_AGENT_GOOGLE_PROVIDER
            and google_lane_key is None
        ):
            google_lane_key = await _resolve_codex_auto_agent_google_lane_key()
        if (
            candidate_template["provider"] == _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER
            and antigravity_lane_state is None
        ):
            antigravity_lane_state = (
                await _resolve_codex_auto_agent_antigravity_lane_state()
            )
        states.append(
            await _build_codex_auto_agent_candidate_state(
                request,
                candidate_template=candidate_template,
                alias_model=alias_model,
                openai_lane_key=openai_lane_key,
                google_lane_key=google_lane_key,
                antigravity_lane_state=antigravity_lane_state,
            )
        )
    return states




def _attach_aawm_alias_routing_state_sources(
    selection: dict[str, Any],
    *,
    affinity: Optional[dict[str, Any]] = None,
    selected_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    enriched = dict(selection)
    if affinity is not None:
        enriched["affinity_state_source"] = affinity.get(
            "affinity_state_source", "local_fallback"
        )
    if selected_state is not None:
        enriched["cooldown_state_source"] = selected_state.get(
            "cooldown_state_source", "local_fallback"
        )
    return enriched

async def _select_codex_auto_agent_candidate(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    alias_model = (
        _normalize_codex_auto_agent_alias_model(request_body.get("model"))
        or _CODEX_AUTO_AGENT_MODEL_ALIAS
    )
    session_key = _resolve_codex_auto_agent_session_key(
        request,
        request_body,
        alias_model=alias_model,
    )
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(
        request_body
    )

    affinity = await _get_codex_auto_agent_session_affinity(session_key)
    if affinity is not None and not has_continuation_state:
        affinity = None
    if affinity is not None and has_continuation_state:
        affinity_candidate = _find_codex_auto_agent_candidate(
            affinity.get("provider"),
            affinity.get("model"),
            alias_model=alias_model,
        )
        if affinity_candidate is not None:
            affinity_state = await _build_codex_auto_agent_candidate_state(
                request,
                candidate_template=affinity_candidate,
                alias_model=alias_model,
            )
            if affinity_state["cooldown_seconds"] > 0:
                _raise_codex_auto_agent_in_flight_cooldown(
                    candidate=affinity_candidate,
                    lane_key=affinity_state.get("lane_key"),
                    cooldown_seconds=affinity_state["cooldown_seconds"],
                )
            return _attach_aawm_alias_routing_state_sources(
                {
                    **affinity_state,
                    "alias_model": alias_model,
                    "session_key": session_key,
                    "selection_reason": "session_affinity",
                    "skipped": [],
                    "in_flight_session": has_continuation_state,
                },
                affinity=affinity,
                selected_state=affinity_state,
            )

    states = await _build_codex_auto_agent_candidate_states(
        request,
        alias_model=alias_model,
    )
    skipped = _build_auto_agent_skipped_candidates_from_states(states)

    if affinity is not None:
        affinity_candidate = _find_codex_auto_agent_candidate(
            affinity.get("provider"),
            affinity.get("model"),
            alias_model=alias_model,
        )
        if affinity_candidate is not None:
            affinity_state = next(
                (
                    state
                    for state in states
                    if state["candidate"]["provider"] == affinity_candidate["provider"]
                    and state["candidate"]["model"] == affinity_candidate["model"]
                ),
                None,
            )
            if affinity_state is not None:
                if affinity_state["cooldown_seconds"] > 0:
                    if has_continuation_state:
                        _raise_codex_auto_agent_in_flight_cooldown(
                            candidate=affinity_candidate,
                            lane_key=affinity_state.get("lane_key"),
                            cooldown_seconds=affinity_state["cooldown_seconds"],
                        )
                    skipped.append(
                        _codex_auto_agent_candidate_public_shape(
                            affinity_candidate,
                            lane_key=affinity_state.get("lane_key"),
                            cooldown_seconds=affinity_state["cooldown_seconds"],
                            reason="session_affinity_cooldown",
                        )
                    )
                else:
                    return _attach_aawm_alias_routing_state_sources(
                        {
                            **affinity_state,
                            "alias_model": alias_model,
                            "session_key": session_key,
                            "selection_reason": "session_affinity",
                            "skipped": skipped,
                            "in_flight_session": has_continuation_state,
                        },
                        affinity=affinity,
                        selected_state=affinity_state,
                    )
            preferred_available = any(
                not state["candidate"].get("last_resort")
                and state["cooldown_seconds"] <= 0
                for state in states
            )
            if (
                affinity_state is not None
                and affinity_state["cooldown_seconds"] <= 0
                and (
                    not affinity_candidate.get("last_resort")
                    or not preferred_available
                )
            ):
                return _attach_aawm_alias_routing_state_sources(
                    {
                        **affinity_state,
                        "alias_model": alias_model,
                        "session_key": session_key,
                        "selection_reason": "session_affinity",
                        "skipped": skipped,
                    },
                    affinity=affinity,
                    selected_state=affinity_state,
                )

    for state in states:
        if state["candidate"].get("last_resort") or state["cooldown_seconds"] > 0:
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "first_available",
                "skipped": skipped,
            },
            selected_state=state,
        )

    for state in states:
        if not state["candidate"].get("last_resort") or state["cooldown_seconds"] > 0:
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "last_resort",
                "skipped": skipped,
            },
            selected_state=state,
        )

    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "All Codex auto-agent alias candidates are currently cooled down."
                ),
                "type": "rate_limit_error",
                "code": "aawm_codex_auto_agent_all_candidates_cooling_down",
            },
            "candidates": skipped,
        },
    )


def _codex_auto_agent_error_text(exc: Any) -> str:
    detail = _extract_google_adapter_exception_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    else:
        detail_text = str(detail)
    return " ".join(
        str(part)
        for part in (
            getattr(exc, "message", None),
            getattr(exc, "code", None),
            detail_text,
            str(exc),
        )
        if part is not None
    )


def _add_codex_auto_agent_text_error_tokens(
    tokens: set[str],
    text_lower: str,
) -> None:
    if "usage_limit_reached" in text_lower:
        tokens.add("usage_limit_reached")
    if "resource_exhausted" in text_lower or "resource exhausted" in text_lower:
        tokens.add("RESOURCE_EXHAUSTED")
    if (
        "model_capacity_exhausted" in text_lower
        or "model capacity exhausted" in text_lower
    ):
        tokens.add("MODEL_CAPACITY_EXHAUSTED")
    if (
        "currently experiencing high demand" in text_lower
        or "experiencing high demand" in text_lower
    ):
        tokens.add("HIGH_DEMAND")
    if "selected model is at capacity" in text_lower or (
        "model is at capacity" in text_lower
        and "try a different model" in text_lower
    ):
        tokens.add("MODEL_AT_CAPACITY")
    if "model is overloaded" in text_lower or "overloaded_error" in text_lower:
        tokens.add("MODEL_OVERLOADED")
    if "busy upstream" in text_lower or (
        "upstream" in text_lower and "busy" in text_lower
    ):
        tokens.add("UPSTREAM_BUSY")
    if "rate_limit_exceeded" in text_lower or "rate limit" in text_lower:
        tokens.add("RATE_LIMIT_EXCEEDED")
    if "too many requests" in text_lower:
        tokens.add("429")
        tokens.add("RATE_LIMIT_EXCEEDED")
    if "aawm_codex_auto_agent_candidate_unavailable" in text_lower:
        tokens.add("aawm_codex_auto_agent_candidate_unavailable")
    if "aawm_auto_agent_failed_responses_payload" in text_lower:
        tokens.add("aawm_auto_agent_failed_responses_payload")
    if (
        "error from provider (deepseek)" in text_lower
        and "assistant message with 'tool_calls' must be followed by tool messages"
        in text_lower
    ) or "insufficient tool messages following tool_calls message" in text_lower:
        tokens.add("DEEPSEEK_TOOL_MESSAGE_MISMATCH")
    if (
        "invalid message provided" in text_lower
        and "must have non-empty content or tool calls" in text_lower
    ):
        tokens.add("OPENROUTER_INVALID_CHAT_MESSAGE")


def _extract_codex_auto_agent_error_tokens(exc: Any) -> set[str]:
    tokens: set[str] = set()
    for parsed in _extract_google_adapter_error_payloads(exc):
        error_blocks: list[dict[str, Any]] = []
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                error_blocks.append(error)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("error"), dict):
                    error_blocks.append(item["error"])
        for error in error_blocks:
            for key in ("code", "status", "type"):
                value = error.get(key)
                if isinstance(value, str) and value:
                    tokens.add(value)
                elif isinstance(value, int):
                    tokens.add(str(value))
            details = error.get("details")
            if isinstance(details, list):
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    reason = detail.get("reason")
                    if isinstance(reason, str) and reason:
                        tokens.add(reason)
            message = error.get("message")
            if isinstance(message, str) and message:
                lowered = message.lower()
                if "usage_limit_reached" in lowered:
                    tokens.add("usage_limit_reached")
                if "resource_exhausted" in lowered:
                    tokens.add("RESOURCE_EXHAUSTED")
                if "model_capacity_exhausted" in lowered:
                    tokens.add("MODEL_CAPACITY_EXHAUSTED")
    text_lower = _codex_auto_agent_error_text(exc).lower()
    _add_codex_auto_agent_text_error_tokens(tokens, text_lower)
    if _is_openrouter_adapter_provider_raw_error(exc):
        tokens.add("OPENROUTER_PROVIDER_RAW_ERROR")
    return tokens


def _classify_codex_auto_agent_retryable_exhaustion(
    exc: Any,
) -> Optional[str]:
    status_code = _extract_google_adapter_exception_status_code(exc)
    tokens = _extract_codex_auto_agent_error_tokens(exc)
    if "aawm_codex_auto_agent_candidate_unavailable" in tokens:
        return "candidate_unavailable"
    if "usage_limit_reached" in tokens:
        return "usage_limit_reached"
    if tokens & _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS:
        return "capacity_exhausted"
    if "DEEPSEEK_TOOL_MESSAGE_MISMATCH" in tokens:
        return "provider_format_rejected"
    if "OPENROUTER_INVALID_CHAT_MESSAGE" in tokens:
        return "provider_format_rejected"
    if "OPENROUTER_PROVIDER_RAW_ERROR" in tokens:
        return "provider_terminal_error"
    if "aawm_auto_agent_failed_responses_payload" in tokens:
        return "provider_terminal_error"
    if tokens & _CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS:
        return "rate_limited"
    if status_code == 429:
        return "rate_limited"
    if status_code in {500, 502, 503, 529}:
        return "upstream_overloaded"
    if status_code == 504:
        return "upstream_timeout"
    return None


def _is_codex_auto_agent_retryable_exhaustion(exc: Any) -> bool:
    return _classify_codex_auto_agent_retryable_exhaustion(exc) is not None


def _parse_codex_auto_agent_header_wait_seconds(exc: Any) -> Optional[float]:
    headers = _extract_adapter_upstream_headers(exc)
    retry_after = _parse_retry_after_seconds_from_headers(headers)
    if retry_after is not None:
        return max(1.0, retry_after)

    wait_candidates: list[float] = []
    for header_name in (
        "X-RateLimit-Reset",
        "x-ratelimit-reset",
        "x-codex-primary-reset-at",
        "x-codex-secondary-reset-at",
        "x-codex-bengalfox-primary-reset-at",
        "x-codex-bengalfox-secondary-reset-at",
    ):
        reset_value = _get_adapter_header_value(headers, header_name)
        if reset_value is None:
            continue
        try:
            reset_number = float(reset_value)
        except Exception:
            continue
        if reset_number > 1_000_000_000_000:
            reset_epoch_seconds = reset_number / 1000.0
        else:
            reset_epoch_seconds = reset_number
        wait_candidates.append(max(1.0, reset_epoch_seconds - time.time()))
    if not wait_candidates:
        return None
    return min(wait_candidates)


def _get_codex_auto_agent_cooldown_seconds(exc: Any) -> float:
    header_wait = _parse_codex_auto_agent_header_wait_seconds(exc)
    if header_wait is not None:
        return max(_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS, header_wait)

    error_class = _classify_codex_auto_agent_retryable_exhaustion(exc)
    tokens = _extract_codex_auto_agent_error_tokens(exc)
    if (
        error_class in {"capacity_exhausted", "upstream_overloaded"}
        or tokens & _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS
    ):
        return _CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS
    if "usage_limit_reached" in tokens:
        return _CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS
    if "RESOURCE_EXHAUSTED" in tokens or "RATE_LIMIT_EXCEEDED" in tokens:
        return _CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    if _extract_google_adapter_exception_status_code(exc) in {429, 503, 529}:
        return _CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    return _CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS


def _iter_codex_auto_agent_error_blocks(exc: Any) -> list[dict[str, Any]]:
    payloads: list[Any] = []
    detail = getattr(exc, "detail", None)
    if detail is not None:
        payloads.append(detail)
    payloads.extend(_extract_google_adapter_error_payloads(exc))

    error_blocks: list[dict[str, Any]] = []
    for parsed in payloads:
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                error_blocks.append(error)
        elif isinstance(parsed, list):
            error_blocks.extend(
                item["error"]
                for item in parsed
                if isinstance(item, dict) and isinstance(item.get("error"), dict)
            )
    return error_blocks


def _extract_codex_auto_agent_error_type_and_code(
    exc: Any,
) -> tuple[Optional[str], Optional[Any]]:
    fallback_error_type = _clean_codex_auth_value(getattr(exc, "type", None))
    fallback_error_code: Optional[Any] = getattr(exc, "code", None)
    error_type: Optional[str] = None
    error_code: Optional[Any] = None
    for error in _iter_codex_auto_agent_error_blocks(exc):
        if error_type is None:
            error_type = _clean_codex_auth_value(
                error.get("type") or error.get("status")
            )
        if error_code is None:
            error_code = error.get("code") or error.get("status")
        if error_type is not None and error_code is not None:
            return error_type, error_code
    return error_type or fallback_error_type, error_code or fallback_error_code


async def _set_codex_auto_agent_candidate_cooldowns(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    selected_cooldown_key: str,
    cooldown_seconds: float,
    exc: Any,
) -> str:
    await _set_codex_auto_agent_cooldown(
        selected_cooldown_key,
        cooldown_seconds,
    )
    return "candidate"


def _update_codex_auto_agent_retryable_attempt_record(
    *,
    attempt_record: dict[str, Any],
    exc: Any,
    error_class: str,
    cooldown_seconds: float,
    cooldown_scope: Optional[str] = None,
) -> set[str]:
    error_tokens = _extract_codex_auto_agent_error_tokens(exc)
    error_status_code = _extract_google_adapter_exception_status_code(exc)
    error_type, error_code = _extract_codex_auto_agent_error_type_and_code(exc)
    retry_after_seconds = _parse_codex_auto_agent_header_wait_seconds(exc)
    update: dict[str, Any] = {
        "status": "cooldown_set",
        "cooldown_seconds": round(float(cooldown_seconds), 3),
        "error_class": error_class,
        "error_tokens": sorted(error_tokens),
        "failure_phase": "provider_attempt",
        "attempted_provider_call": True,
    }
    if cooldown_scope is not None:
        update["cooldown_scope"] = cooldown_scope
    if error_status_code is not None:
        update["error_status_code"] = error_status_code
    if error_type is not None:
        update["error_type"] = error_type
    if error_code is not None:
        update["error_code"] = str(error_code)
    if retry_after_seconds is not None:
        update["retry_after_seconds"] = round(float(retry_after_seconds), 3)
    attempt_record.update(update)
    return error_tokens


def _record_auto_agent_alias_attempt_started(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    candidate_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    _safe_set_request_parsed_body(request, candidate_body)
    return candidate_body


def _record_auto_agent_alias_attempt_failure(
    *,
    alias_family: str,
    alias_model: str,
    request: Request,
    prepared_request_body: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempt_record: dict[str, Any],
    error_class: str,
    add_alias_metadata_fn: Callable[..., dict[str, Any]],
    redispatch_required: bool = False,
) -> dict[str, Any]:
    failure_body = add_alias_metadata_fn(
        prepared_request_body,
        request=request,
        selection=selection,
        attempts=attempts,
    )
    _safe_set_request_parsed_body(request, failure_body)
    _emit_auto_agent_alias_route_event(
        _build_auto_agent_alias_audit_event(
            alias_family=alias_family,
            alias_model=alias_model,
            request=request,
            request_body=prepared_request_body,
            selection=selection,
            candidate=attempt_record,
            event_type="redispatch_required"
            if redispatch_required
            else "candidate_retryable_failure",
            candidate_status=attempt_record.get("status") or "cooldown_set",
            attempt_number=len(attempts),
            selected=True,
            selection_reason=selection.get("selection_reason"),
            lane_key=selection.get("lane_key"),
            cooldown_key=selection.get("cooldown_key"),
            cooldown_seconds=attempt_record.get("cooldown_seconds"),
            cooldown_scope=attempt_record.get("cooldown_scope"),
            failure_class=error_class,
            error_status_code=attempt_record.get("error_status_code"),
            error_type=attempt_record.get("error_type"),
            error_code=attempt_record.get("error_code"),
            error_tokens=attempt_record.get("error_tokens"),
            retry_after_seconds=attempt_record.get("retry_after_seconds"),
            failure_phase=attempt_record.get("failure_phase"),
            attempted_provider_call=attempt_record.get("attempted_provider_call"),
            redispatch_required=redispatch_required,
        ),
        level="warning",
    )
    return failure_body


def _add_codex_auto_agent_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = selection["candidate"]
    alias_model = (
        selection.get("alias_model")
        or _normalize_codex_auto_agent_alias_model(request_body.get("model"))
        or _CODEX_AUTO_AGENT_MODEL_ALIAS
    )
    target_model = candidate["model"]
    updated_body = copy.deepcopy(request_body)
    updated_body["model"] = target_model
    default_reasoning_effort = _normalize_low_cardinality_tag_value(
        candidate.get("default_reasoning_effort")
    )
    default_reasoning_applied = False
    if default_reasoning_effort and "reasoning_effort" not in updated_body:
        reasoning = updated_body.get("reasoning")
        if not isinstance(reasoning, dict):
            updated_body["reasoning"] = {"effort": default_reasoning_effort}
            default_reasoning_applied = True
        elif not reasoning.get("effort"):
            updated_body["reasoning"] = {
                **reasoning,
                "effort": default_reasoning_effort,
            }
            default_reasoning_applied = True
    skipped = selection.get("skipped") or []
    audit_events = _build_auto_agent_alias_audit_events(
        alias_family="codex_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=request_body,
        selection=selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "codex-auto-agent-alias",
            f"codex-auto-agent-selected:{target_model}",
            f"codex-auto-agent-route:{candidate['route_family']}",
            f"model-alias:{alias_model}",
            *(
                ["codex-auto-agent-last-resort"]
                if candidate.get("last_resort")
                else []
            ),
            *(
                [f"codex-auto-agent-default-effort:{default_reasoning_effort}"]
                if default_reasoning_applied
                else []
            ),
            f"codex-auto-agent-alias:{alias_model}",
        ],
        extra_fields={
            "model_alias_label": alias_model,
            "requested_model_alias": alias_model,
            "codex_auto_agent_alias": alias_model,
            "codex_auto_agent_selected_provider": candidate["provider"],
            "codex_auto_agent_selected_model": target_model,
            "codex_auto_agent_selected_route_family": candidate["route_family"],
            "codex_auto_agent_selected_last_resort": bool(
                candidate.get("last_resort")
            ),
            **(
                {
                    "codex_auto_agent_default_reasoning_effort": (
                        default_reasoning_effort
                    ),
                    "codex_reasoning_effort": default_reasoning_effort,
                }
                if default_reasoning_applied
                else {}
            ),
            "codex_auto_agent_selection_reason": selection.get("selection_reason"),
            "codex_auto_agent_affinity_state_source": selection.get(
                "affinity_state_source"
            ),
            "codex_auto_agent_cooldown_state_source": selection.get(
                "cooldown_state_source"
            ),
            "codex_auto_agent_lane_key": selection.get("lane_key"),
            "codex_auto_agent_attempts": attempts,
            "codex_auto_agent_skipped_candidates": skipped,
            "codex_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )

def _normalize_anthropic_auto_agent_alias_model(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized = model.strip().lower()
    for alias in _ANTHROPIC_AUTO_AGENT_CANDIDATES_BY_ALIAS:
        if normalized == alias.lower():
            return alias
    return None


def _is_anthropic_auto_agent_alias_model(model: Any) -> bool:
    return _normalize_anthropic_auto_agent_alias_model(model) is not None


def _resolve_anthropic_auto_agent_alias_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    return _normalize_anthropic_auto_agent_alias_model(request_body.get("model"))


def _resolve_anthropic_auto_agent_native_lane_key(
    request: Request,
    *,
    include_session_fallback: bool = True,
) -> str:
    headers = _safe_get_request_headers(request)
    for header_name in ("x-api-key", "authorization"):
        header_value = _get_codex_auto_agent_header(headers, header_name)
        if header_value is not None:
            return f"{header_name}:{_hash_codex_auto_agent_lane_value(header_value)}"
    if include_session_fallback:
        session_header = (
            _get_codex_auto_agent_header(headers, "session_id")
            or _get_codex_auto_agent_header(headers, "session-id")
            or _get_codex_auto_agent_header(headers, "x-session-id")
        )
        if session_header is not None:
            return f"session:{session_header}"
    return "__default__"


def _resolve_anthropic_auto_agent_native_cooldown_lane_key(request: Request) -> str:
    return _resolve_anthropic_auto_agent_native_lane_key(
        request,
        include_session_fallback=False,
    )


def _resolve_anthropic_auto_agent_session_key(
    request: Request,
    request_body: dict[str, Any],
    *,
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    metadata_session_id = (
        metadata.get("session_id") if isinstance(metadata, dict) else None
    )
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
    return (
        f"{alias_model}:{session_id}:"
        f"{_resolve_anthropic_auto_agent_native_lane_key(request)}"
    )


async def _get_anthropic_auto_agent_active_cooldown_state(
    cooldown_key: str,
) -> tuple[float, str]:
    async with _anthropic_auto_agent_lock:
        now = time.monotonic()
        until = _anthropic_auto_agent_cooldown_until_monotonic_by_key.get(
            cooldown_key, 0.0
        )
        if until > now:
            return max(0.0, until - now), "memory"
        _anthropic_auto_agent_cooldown_until_monotonic_by_key.pop(
            cooldown_key, None
        )
    dual_cache = _get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return 0.0, "local_fallback"
    durable_payload = await _read_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="cooldown",
        state_key=cooldown_key,
    )
    if durable_payload is None:
        return 0.0, "local_fallback"
    expires_at_epoch = _parse_aawm_alias_routing_durable_expiry(durable_payload)
    if expires_at_epoch is None:
        return 0.0, "local_fallback"
    async with _anthropic_auto_agent_lock:
        _hydrate_aawm_alias_routing_cooldown_memory(
            memory_map=_anthropic_auto_agent_cooldown_until_monotonic_by_key,
            cooldown_key=cooldown_key,
            expires_at_epoch=expires_at_epoch,
        )
        until = _anthropic_auto_agent_cooldown_until_monotonic_by_key.get(
            cooldown_key, 0.0
        )
        return max(0.0, until - time.monotonic()), "durable_cache"


async def _get_anthropic_auto_agent_active_cooldown_seconds(
    cooldown_key: str,
) -> float:
    seconds, _ = await _get_anthropic_auto_agent_active_cooldown_state(cooldown_key)
    return seconds


async def _set_anthropic_auto_agent_cooldown(
    cooldown_key: str,
    cooldown_seconds: float,
) -> None:
    ttl_seconds = max(0.0, float(cooldown_seconds))
    async with _anthropic_auto_agent_lock:
        until = time.monotonic() + ttl_seconds
        current_until = _anthropic_auto_agent_cooldown_until_monotonic_by_key.get(
            cooldown_key, 0.0
        )
        if until > current_until:
            _anthropic_auto_agent_cooldown_until_monotonic_by_key[cooldown_key] = until
    if ttl_seconds <= 0:
        return
    await _write_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="cooldown",
        state_key=cooldown_key,
        payload={"cooldown_key": cooldown_key},
        ttl_seconds=ttl_seconds,
    )


async def _get_anthropic_auto_agent_session_affinity(
    session_key: Optional[str],
) -> Optional[dict[str, Any]]:
    if session_key is None:
        return None
    async with _anthropic_auto_agent_lock:
        affinity = _anthropic_auto_agent_session_affinity_by_key.get(session_key)
        if isinstance(affinity, dict):
            expires_at = affinity.get("expires_at_monotonic", 0.0)
            if isinstance(expires_at, (int, float)) and expires_at > time.monotonic():
                hydrated = dict(affinity)
                hydrated["affinity_state_source"] = affinity.get(
                    "affinity_state_source", "memory"
                )
                return hydrated
            _anthropic_auto_agent_session_affinity_by_key.pop(session_key, None)
    dual_cache = _get_aawm_alias_routing_dual_cache()
    if dual_cache is None:
        return None
    durable_payload = await _read_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=session_key,
    )
    if durable_payload is None:
        return None
    expires_at_epoch = _parse_aawm_alias_routing_durable_expiry(durable_payload)
    if expires_at_epoch is None:
        return None
    async with _anthropic_auto_agent_lock:
        affinity = _hydrate_aawm_alias_routing_affinity_memory(
            memory_map=_anthropic_auto_agent_session_affinity_by_key,
            session_key=session_key,
            payload=durable_payload,
            expires_at_epoch=expires_at_epoch,
        )
        if not affinity:
            return None
        affinity["affinity_state_source"] = "durable_cache"
        return dict(affinity)


async def _set_anthropic_auto_agent_session_affinity(
    session_key: Optional[str],
    candidate: dict[str, Any],
) -> None:
    if session_key is None:
        return
    async with _anthropic_auto_agent_lock:
        _anthropic_auto_agent_session_affinity_by_key[session_key] = {
            "provider": candidate["provider"],
            "model": candidate["model"],
            "route_family": candidate["route_family"],
            "last_resort": bool(candidate.get("last_resort")),
            "expires_at_monotonic": (
                time.monotonic() + _CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS
            ),
            "affinity_state_source": "memory",
        }
    await _write_aawm_alias_routing_durable_payload(
        alias_family="anthropic",
        state_kind="affinity",
        state_key=session_key,
        payload={
            "provider": candidate["provider"],
            "model": candidate["model"],
            "route_family": candidate["route_family"],
            "last_resort": bool(candidate.get("last_resort")),
        },
        ttl_seconds=_CODEX_AUTO_AGENT_SESSION_AFFINITY_TTL_SECONDS,
    )


def _find_anthropic_auto_agent_candidate(
    provider: Any,
    model: Any,
    *,
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
) -> Optional[dict[str, Any]]:
    for candidate in _get_anthropic_auto_agent_candidates_for_alias(alias_model):
        if candidate["provider"] == provider and candidate["model"] == model:
            return dict(candidate)
    return None


async def _build_anthropic_auto_agent_candidate_states(
    request: Request,
    *,
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
) -> list[dict[str, Any]]:
    openai_lane_key = _resolve_codex_auto_agent_openai_cooldown_lane_key(request)
    anthropic_lane_key = _resolve_anthropic_auto_agent_native_cooldown_lane_key(
        request
    )
    google_lane_key: Optional[str] = None
    antigravity_lane_state: Optional[dict[str, Any]] = None
    states: list[dict[str, Any]] = []
    for candidate_template in _get_anthropic_auto_agent_candidates_for_alias(
        alias_model
    ):
        if (
            candidate_template["provider"] == _CODEX_AUTO_AGENT_GOOGLE_PROVIDER
            and google_lane_key is None
        ):
            google_lane_key = await _resolve_codex_auto_agent_google_lane_key()
        if (
            candidate_template["provider"] == _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER
            and antigravity_lane_state is None
        ):
            antigravity_lane_state = (
                await _resolve_codex_auto_agent_antigravity_lane_state()
            )
        states.append(
            await _build_anthropic_auto_agent_candidate_state(
                request,
                candidate_template=candidate_template,
                alias_model=alias_model,
                openai_lane_key=openai_lane_key,
                anthropic_lane_key=anthropic_lane_key,
                google_lane_key=google_lane_key,
                antigravity_lane_state=antigravity_lane_state,
            )
        )
    return states


def _raise_anthropic_auto_agent_in_flight_cooldown(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
) -> None:
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_session_affinity_cooldown",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Anthropic auto-agent alias target is cooling down for an "
                    "in-flight session; provider switching is disabled for "
                    "stateful Claude continuations. Redispatch a fresh agent "
                    "attempt to re-run the auto selector."
                ),
                "type": "rate_limit_error",
                "code": "aawm_anthropic_auto_agent_in_flight_provider_cooling_down",
            },
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(int(max(1.0, cooldown_seconds)))},
    )


def _raise_anthropic_auto_agent_redispatch_required(
    *,
    candidate: dict[str, Any],
    lane_key: Optional[str],
    cooldown_seconds: float,
    error_tokens: set[str],
    alias_model: str = _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS,
) -> None:
    retry_after_seconds = int(max(1.0, cooldown_seconds))
    shaped_candidate = _codex_auto_agent_candidate_public_shape(
        candidate,
        lane_key=lane_key,
        cooldown_seconds=cooldown_seconds,
        reason="in_flight_retryable_provider_exhaustion",
    )
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "Anthropic auto-agent alias target hit retryable provider "
                    "exhaustion for an in-flight session. Do not continue this "
                    f"child agent. Redispatch a fresh subagent using model {alias_model} "
                    "so the auto selector can choose the next available candidate."
                ),
                "type": "rate_limit_error",
                "code": "aawm_anthropic_auto_agent_redispatch_required",
            },
            "redispatch_model": alias_model,
            "redispatch_reason": "in_flight_retryable_provider_exhaustion",
            "selected_provider": candidate.get("provider"),
            "selected_model": candidate.get("model"),
            "selected_route_family": candidate.get("route_family"),
            "cooldown_seconds": round(float(cooldown_seconds), 3),
            "retry_after_seconds": retry_after_seconds,
            "error_tokens": sorted(error_tokens),
            "candidate": shaped_candidate,
        },
        headers={"Retry-After": str(retry_after_seconds)},
    )


async def _select_anthropic_auto_agent_candidate(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    alias_model = (
        _normalize_anthropic_auto_agent_alias_model(request_body.get("model"))
        or _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
    )
    session_key = _resolve_anthropic_auto_agent_session_key(
        request,
        request_body,
        alias_model=alias_model,
    )
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(
        request_body
    )

    affinity = await _get_anthropic_auto_agent_session_affinity(session_key)
    if affinity is not None and not has_continuation_state:
        affinity = None
    if affinity is not None and has_continuation_state:
        affinity_candidate = _find_anthropic_auto_agent_candidate(
            affinity.get("provider"),
            affinity.get("model"),
            alias_model=alias_model,
        )
        if affinity_candidate is not None:
            affinity_state = await _build_anthropic_auto_agent_candidate_state(
                request,
                candidate_template=affinity_candidate,
                alias_model=alias_model,
            )
            if affinity_state["cooldown_seconds"] > 0:
                _raise_anthropic_auto_agent_in_flight_cooldown(
                    candidate=affinity_candidate,
                    lane_key=affinity_state.get("lane_key"),
                    cooldown_seconds=affinity_state["cooldown_seconds"],
                )
            return _attach_aawm_alias_routing_state_sources(
                {
                    **affinity_state,
                    "alias_model": alias_model,
                    "session_key": session_key,
                    "selection_reason": "session_affinity",
                    "skipped": [],
                    "in_flight_session": has_continuation_state,
                },
                affinity=affinity,
                selected_state=affinity_state,
            )

    states = await _build_anthropic_auto_agent_candidate_states(
        request,
        alias_model=alias_model,
    )
    skipped = _build_auto_agent_skipped_candidates_from_states(states)

    if affinity is not None:
        affinity_candidate = _find_anthropic_auto_agent_candidate(
            affinity.get("provider"),
            affinity.get("model"),
            alias_model=alias_model,
        )
        if affinity_candidate is not None:
            affinity_state = next(
                (
                    state
                    for state in states
                    if state["candidate"]["provider"] == affinity_candidate["provider"]
                    and state["candidate"]["model"] == affinity_candidate["model"]
                ),
                None,
            )
            if affinity_state is not None:
                if affinity_state["cooldown_seconds"] > 0:
                    if has_continuation_state:
                        _raise_anthropic_auto_agent_in_flight_cooldown(
                            candidate=affinity_candidate,
                            lane_key=affinity_state.get("lane_key"),
                            cooldown_seconds=affinity_state["cooldown_seconds"],
                        )
                    skipped.append(
                        _codex_auto_agent_candidate_public_shape(
                            affinity_candidate,
                            lane_key=affinity_state.get("lane_key"),
                            cooldown_seconds=affinity_state["cooldown_seconds"],
                            reason="session_affinity_cooldown",
                        )
                    )
                else:
                    return _attach_aawm_alias_routing_state_sources(
                        {
                            **affinity_state,
                            "alias_model": alias_model,
                            "session_key": session_key,
                            "selection_reason": "session_affinity",
                            "skipped": skipped,
                            "in_flight_session": has_continuation_state,
                        },
                        affinity=affinity,
                        selected_state=affinity_state,
                    )

    for state in states:
        if state["candidate"].get("last_resort") or state["cooldown_seconds"] > 0:
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "first_available",
                "skipped": skipped,
                "in_flight_session": has_continuation_state,
            },
            selected_state=state,
        )

    for state in states:
        if not state["candidate"].get("last_resort") or state["cooldown_seconds"] > 0:
            continue
        return _attach_aawm_alias_routing_state_sources(
            {
                **state,
                "alias_model": alias_model,
                "session_key": session_key,
                "selection_reason": "last_resort",
                "skipped": skipped,
                "in_flight_session": has_continuation_state,
            },
            selected_state=state,
        )

    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "message": (
                    "All Anthropic auto-agent alias candidates are currently cooled down."
                ),
                "type": "rate_limit_error",
                "code": "aawm_anthropic_auto_agent_all_candidates_cooling_down",
            },
            "candidates": skipped,
        },
    )


def _add_anthropic_auto_agent_alias_metadata(
    request_body: dict[str, Any],
    *,
    request: Request,
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = selection["candidate"]
    alias_model = (
        selection.get("alias_model")
        or _normalize_anthropic_auto_agent_alias_model(request_body.get("model"))
        or _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
    )
    target_model = candidate["model"]
    updated_body = copy.deepcopy(request_body)
    updated_body["model"] = target_model
    skipped = selection.get("skipped") or []
    audit_events = _build_auto_agent_alias_audit_events(
        alias_family="anthropic_auto_agent",
        alias_model=alias_model,
        request=request,
        request_body=request_body,
        selection=selection,
        attempts=attempts,
    )
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "anthropic-auto-agent-alias",
            f"anthropic-auto-agent-selected:{target_model}",
            f"anthropic-auto-agent-route:{candidate['route_family']}",
            f"model-alias:{alias_model}",
            *(
                ["anthropic-auto-agent-last-resort"]
                if candidate.get("last_resort")
                else []
            ),
            f"anthropic-auto-agent-alias:{alias_model}",
        ],
        extra_fields={
            "model_alias_label": alias_model,
            "requested_model_alias": alias_model,
            "anthropic_auto_agent_alias": alias_model,
            "anthropic_auto_agent_selected_provider": candidate["provider"],
            "anthropic_auto_agent_selected_model": target_model,
            "anthropic_auto_agent_selected_route_family": candidate["route_family"],
            "anthropic_auto_agent_selected_last_resort": bool(
                candidate.get("last_resort")
            ),
            "anthropic_auto_agent_selection_reason": selection.get(
                "selection_reason"
            ),
            "anthropic_auto_agent_affinity_state_source": selection.get(
                "affinity_state_source"
            ),
            "anthropic_auto_agent_cooldown_state_source": selection.get(
                "cooldown_state_source"
            ),
            "anthropic_auto_agent_lane_key": selection.get("lane_key"),
            "anthropic_auto_agent_attempts": attempts,
            "anthropic_auto_agent_skipped_candidates": skipped,
            "anthropic_auto_agent_audit_events": audit_events,
            "aawm_alias_routing_audit_events": audit_events,
        },
    )


def _resolve_anthropic_openai_responses_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_openai_responses_adapter_model_name(
            candidate
        )
        if normalized_model is not None:
            return normalized_model
    return None


def _resolve_anthropic_xai_oauth_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        if is_oa_xai_model(candidate):
            return candidate
    return None


def _resolve_anthropic_grok_native_oauth_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = normalize_grok_native_oauth_model(candidate)
        if normalized_model is not None:
            return normalized_model
    return None


def _resolve_anthropic_openrouter_completion_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_openrouter_adapter_model_name(candidate)
        if normalized_model in _ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS:
            return normalized_model
    return None


def _resolve_anthropic_nvidia_responses_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_nvidia_responses_adapter_model_name(
            candidate
        )
        if normalized_model is not None:
            return normalized_model
    return None


def _resolve_anthropic_openrouter_responses_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_openrouter_adapter_model_name(candidate)
        if normalized_model in _ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS:
            return normalized_model
        explicit_provider, _ = _split_anthropic_adapter_provider_prefix(candidate)
        if explicit_provider == "openrouter" and normalized_model is not None:
            return normalized_model
    return None


def _resolve_anthropic_google_completion_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_google_completion_adapter_model_name(
            candidate
        )
        if normalized_model is not None:
            return normalized_model
    return None


def _get_anthropic_adapter_google_auth_file_path() -> Optional[Path]:
    for env_name in _ANTHROPIC_ADAPTER_GEMINI_AUTH_FILE_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if path.exists():
            return path

    for candidate_str in _ANTHROPIC_ADAPTER_GEMINI_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.exists():
            return candidate

    return None


def _extract_google_oauth_client_values_from_bundle_text(
    bundle_text: str,
) -> tuple[Optional[str], Optional[str]]:
    client_id_match = _ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_ID_PATTERN.search(
        bundle_text
    )
    client_secret_match = (
        _ANTHROPIC_ADAPTER_GEMINI_CLI_OAUTH_CLIENT_SECRET_PATTERN.search(bundle_text)
    )
    client_id = (
        _clean_codex_auth_value(client_id_match.group("value"))
        if client_id_match is not None
        else None
    )
    client_secret = (
        _clean_codex_auth_value(client_secret_match.group("value"))
        if client_secret_match is not None
        else None
    )
    return client_id, client_secret


def _add_google_cli_bundle_candidate_files(
    raw_path: Path, candidate_files: list[Path], seen_paths: set[str]
) -> None:
    path = raw_path.expanduser()
    if not path.exists():
        return

    if path.is_file():
        resolved = str(path.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            candidate_files.append(path)
        return

    bundle_dir = path
    if (bundle_dir / "bundle").is_dir():
        bundle_dir = bundle_dir / "bundle"
    elif (
        bundle_dir / "lib" / "node_modules" / "@google" / "gemini-cli" / "bundle"
    ).is_dir():
        bundle_dir = (
            bundle_dir / "lib" / "node_modules" / "@google" / "gemini-cli" / "bundle"
        )

    chunk_files = sorted(bundle_dir.glob("chunk-*.js"))
    gemini_bundle = bundle_dir / "gemini.js"
    ordered_files = chunk_files + ([gemini_bundle] if gemini_bundle.is_file() else [])
    for candidate in ordered_files:
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_files.append(candidate)


def _iter_google_oauth_client_bundle_candidates() -> list[Path]:
    candidate_files: list[Path] = []
    seen_paths: set[str] = set()

    for env_name in _ANTHROPIC_ADAPTER_GEMINI_CLI_BUNDLE_PATH_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if raw_value:
            _add_google_cli_bundle_candidate_files(
                Path(raw_value), candidate_files, seen_paths
            )

    for bundle_glob in _ANTHROPIC_ADAPTER_GEMINI_DEFAULT_CLI_BUNDLE_GLOBS:
        for matched_path in sorted(glob.glob(os.path.expanduser(bundle_glob)), reverse=True):
            _add_google_cli_bundle_candidate_files(
                Path(matched_path), candidate_files, seen_paths
            )

    return candidate_files


def _load_google_oauth_client_values_from_local_gemini_cli_bundle(
) -> tuple[Optional[str], Optional[str]]:
    for candidate in _iter_google_oauth_client_bundle_candidates():
        try:
            bundle_text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        client_id, client_secret = _extract_google_oauth_client_values_from_bundle_text(
            bundle_text
        )
        if client_id and client_secret:
            return client_id, client_secret

    return None, None


async def _load_local_google_oauth_credentials() -> tuple[dict[str, Any], Path]:
    auth_path = _get_anthropic_adapter_google_auth_file_path()
    if auth_path is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Anthropic adapter requests for Gemini models require local Google OAuth creds at "
                "'~/.gemini/oauth_creds.json' or 'LITELLM_GEMINI_AUTH_FILE'."
            ),
        )

    try:
        auth_data = json.loads(auth_path.read_text())
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Gemini OAuth credentials from {auth_path}: {exc}",
        ) from exc

    if not isinstance(auth_data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Gemini OAuth credentials at {auth_path} are not a JSON object.",
        )

    return auth_data, auth_path


def _google_oauth_token_is_valid(auth_data: dict[str, Any]) -> bool:
    access_token = _clean_codex_auth_value(auth_data.get("access_token"))
    expiry_date = auth_data.get("expiry_date")
    if access_token is None:
        return False
    if not isinstance(expiry_date, (int, float)):
        return False
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return int(expiry_date) > now_ms + 60_000


def _google_oauth_cached_token_is_valid(cached_token: tuple[str, int]) -> bool:
    _access_token, expiry_date = cached_token
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return expiry_date > now_ms + 60_000


def _get_google_oauth_expiry_date(auth_data: dict[str, Any]) -> Optional[int]:
    expiry_date = auth_data.get("expiry_date")
    if isinstance(expiry_date, (int, float)):
        return int(expiry_date)
    return None


def _get_google_oauth_client_value(
    auth_data: dict[str, Any],
    candidate_keys: tuple[str, ...],
    env_var_names: tuple[str, ...],
) -> Optional[str]:
    for key in candidate_keys:
        value = _clean_codex_auth_value(auth_data.get(key))
        if value is not None:
            return value
    return _get_first_secret_value(env_var_names)


async def _refresh_local_google_oauth_credentials(
    auth_data: dict[str, Any],
) -> dict[str, Any]:
    refresh_token = _clean_codex_auth_value(auth_data.get("refresh_token"))
    if refresh_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini OAuth credentials do not contain a refresh_token. "
                "Re-authenticate Gemini CLI before using Gemini Anthropic adapter models."
            ),
        )

    client_id = _get_google_oauth_client_value(
        auth_data,
        ("client_id", "clientId"),
        _ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_ID_ENV_VARS,
    )
    client_secret = _get_google_oauth_client_value(
        auth_data,
        ("client_secret", "clientSecret"),
        _ANTHROPIC_ADAPTER_GEMINI_OAUTH_CLIENT_SECRET_ENV_VARS,
    )
    if client_id is None or client_secret is None:
        (
            bundle_client_id,
            bundle_client_secret,
        ) = _load_google_oauth_client_values_from_local_gemini_cli_bundle()
        client_id = client_id or bundle_client_id
        client_secret = client_secret or bundle_client_secret
    if client_id is None or client_secret is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini OAuth credentials do not contain client_id/client_secret and no fallback env vars or Gemini CLI bundle were found. "
                "Re-authenticate Gemini CLI or configure Gemini OAuth client env vars before using Gemini Anthropic adapter models."
            ),
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to refresh Gemini OAuth access token for Anthropic adapter models: "
                f"{response.text}"
            ),
        )

    refreshed = response.json()
    expires_in = refreshed.get("expires_in")
    expiry_date: Optional[int] = None
    if isinstance(expires_in, (int, float)):
        expiry_date = int(datetime.now(timezone.utc).timestamp() * 1000) + int(expires_in * 1000)

    updated_auth_data = dict(auth_data)
    updated_auth_data.update(refreshed)
    updated_auth_data["refresh_token"] = refresh_token
    if expiry_date is not None:
        updated_auth_data["expiry_date"] = expiry_date

    _invalidate_codex_auto_agent_google_lane_cache()
    return updated_auth_data


async def _load_valid_local_google_oauth_access_token() -> str:
    auth_data, _auth_path = await _load_local_google_oauth_credentials()
    cache_key = str(_auth_path.expanduser())
    cached_token = _google_oauth_access_token_cache.get(cache_key)
    if cached_token is not None and _google_oauth_cached_token_is_valid(cached_token):
        return cached_token[0]

    async with _google_oauth_access_token_lock:
        cached_token = _google_oauth_access_token_cache.get(cache_key)
        if cached_token is not None and _google_oauth_cached_token_is_valid(
            cached_token
        ):
            return cached_token[0]

        auth_data, _auth_path = await _load_local_google_oauth_credentials()
        if not _google_oauth_token_is_valid(auth_data):
            auth_data = await _refresh_local_google_oauth_credentials(auth_data)

        access_token = _clean_codex_auth_value(auth_data.get("access_token"))
        if access_token is None:
            raise HTTPException(
                status_code=500,
                detail="Gemini OAuth credentials did not yield a valid access_token.",
            )
        expiry_date = _get_google_oauth_expiry_date(auth_data)
        if expiry_date is not None:
            _google_oauth_access_token_cache[cache_key] = (access_token, expiry_date)
        _invalidate_codex_auto_agent_google_lane_cache()
        return access_token


def _extract_google_adapter_agent_name_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    for message in completion_messages:
        content = message.get('content')
        if not isinstance(content, str) or not content:
            continue
        match = _CLAUDE_AGENT_TENANT_PATTERN.search(content)
        if match:
            agent_name = match.group('agent').strip()
            if agent_name:
                return agent_name
    return None


def _extract_google_adapter_latest_user_prompt_text(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    for message in reversed(completion_messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _extract_completion_message_text(message).strip()
        if not text:
            continue
        reminder_matches = list(
            re.finditer(r"<system-reminder>.*?</system-reminder>\n*", text, re.DOTALL)
        )
        if reminder_matches:
            trailing_text = text[reminder_matches[-1].end() :].strip()
            if trailing_text:
                return trailing_text
            continue
        return text
    return None


def _extract_google_adapter_latest_tool_result_fingerprint(
    completion_messages: list[dict[str, Any]],
) -> Optional[str]:
    for message in reversed(completion_messages):
        if not isinstance(message, dict) or message.get("role") not in {
            "tool",
            "function",
        }:
            continue
        tool_call_id = message.get("tool_call_id") or message.get("name") or ""
        text = _extract_completion_message_text(message).strip()
        if not tool_call_id and not text:
            continue
        result_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        return f"{tool_call_id}:{result_hash}"
    return None


def _resolve_google_adapter_session_id(
    request: Request,
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
) -> tuple[str, str]:
    direct_session_id = (
        _get_request_header_or_passthrough_alias(request, 'session_id')
        or _safe_get_request_headers(request).get('x-claude-code-session-id')
        or _safe_get_request_headers(request).get('X-Claude-Code-Session-Id')
    )
    trace_id = (
        _get_request_header_or_passthrough_alias(request, 'langfuse_trace_id')
        or _get_request_header_or_passthrough_alias(request, 'langfuse_existing_trace_id')
        or _get_request_header_or_passthrough_alias(request, 'trace_id')
    )
    trace_name = _get_request_header_or_passthrough_alias(request, 'langfuse_trace_name')
    agent_name = _extract_google_adapter_agent_name_from_completion_messages(completion_messages)

    if isinstance(direct_session_id, str) and direct_session_id:
        seed = f"direct_session_id:{direct_session_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed)), 'direct_session_id'

    identity_name = None
    identity_source = None
    if isinstance(trace_name, str) and trace_name:
        identity_name = trace_name
        identity_source = 'trace_name'
    elif isinstance(agent_name, str) and agent_name:
        identity_name = agent_name
        identity_source = 'agent_name'

    if identity_name:
        seed = f"{identity_source}:{identity_name}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed)), identity_source or 'derived'

    if isinstance(trace_id, str) and trace_id:
        seed = f"trace_id:{trace_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed)), 'trace_id'

    return str(uuid4()), 'generated_uuid'


def _resolve_google_adapter_user_prompt_id(
    request: Request,
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
    session_id: str,
) -> str:
    trace_id = (
        _get_request_header_or_passthrough_alias(request, "langfuse_trace_id")
        or _get_request_header_or_passthrough_alias(
            request, "langfuse_existing_trace_id"
        )
        or _get_request_header_or_passthrough_alias(request, "trace_id")
    )
    if isinstance(trace_id, str) and trace_id:
        seed = f"user_prompt_trace_id:{trace_id}|model:{google_model}"
        return str(uuid5(NAMESPACE_URL, seed))

    latest_tool_result = _extract_google_adapter_latest_tool_result_fingerprint(
        completion_messages
    )
    if isinstance(latest_tool_result, str) and latest_tool_result:
        seed = (
            f"user_prompt_tool_result:{latest_tool_result}|"
            f"session_id:{session_id}|model:{google_model}"
        )
        return str(uuid5(NAMESPACE_URL, seed))

    latest_user_prompt = _extract_google_adapter_latest_user_prompt_text(
        completion_messages
    )
    if isinstance(latest_user_prompt, str) and latest_user_prompt:
        prompt_hash = hashlib.sha1(latest_user_prompt.encode("utf-8")).hexdigest()[:16]
        seed = (
            f"user_prompt_hash:{prompt_hash}|session_id:{session_id}|model:{google_model}"
        )
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
    target_base = _get_code_assist_adapter_target_base(adapter_provider)
    cache_key = hashlib.sha256(
        f"{adapter_provider}:{target_base}:{access_token}".encode("utf-8")
    ).hexdigest()
    cached_project = _google_code_assist_project_cache.get(cache_key)
    if isinstance(cached_project, str) and cached_project:
        return cached_project

    async with _google_code_assist_project_lock:
        cached_project = _google_code_assist_project_cache.get(cache_key)
        if isinstance(cached_project, str) and cached_project:
            return cached_project

        load_url = f"{target_base.rstrip('/')}/v1internal:loadCodeAssist"
        request_body = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }
        headers = _build_code_assist_adapter_native_headers(
            adapter_provider=adapter_provider,
            access_token=access_token,
            model=None,
            accept="application/json",
        )
        HttpPassThroughEndpointHelpers.validate_outgoing_egress(
            url=load_url,
            headers=headers,
            credential_family="google",
            expected_target_family="google",
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(load_url, headers=headers, json=request_body)

        try:
            response_body = response.json()
        except Exception:
            response_body = None
        capture_passthrough_shape(
            mode="google_code_assist_loadCodeAssist",
            provider=adapter_provider,
            url_route=load_url,
            request_body=request_body,
            response=response,
            response_body=response_body,
            response_content=response.content,
            extra_metadata={
                "direct_google_code_assist_preflight": True,
                "code_assist_adapter_provider": adapter_provider,
            },
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to load Google Code Assist project for Anthropic adapter models: "
                    f"{response.text}"
                ),
            )

        if not isinstance(response_body, dict):
            response_body = response.json()
        project = _clean_codex_auth_value(response_body.get("cloudaicompanionProject"))
        if project is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Google Code Assist bootstrap did not return a cloudaicompanionProject."
                ),
            )

        _google_code_assist_project_cache[cache_key] = project
        return project


def _get_google_code_assist_prime_ttl_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_CODE_ASSIST_PRIME_TTL_SECONDS")
    )
    if raw_value is None:
        # Current Gemini CLI caches Code Assist user/project setup for 30s.
        # Match that default instead of re-priming on every adapted request.
        return 30.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 30.0
    return max(0.0, parsed)


def _get_google_code_assist_prime_cache_key(
    access_token: str,
    companion_project: str,
) -> str:
    token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:12]
    return f"{token_hash}:{companion_project}"


def _get_google_adapter_max_concurrent() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_CONCURRENT"))
    if raw_value is None:
        return 1
    try:
        parsed = int(raw_value)
    except Exception:
        return 1
    return max(1, parsed)


def _get_google_adapter_shared_lane_key(
    *,
    access_token: Optional[str],
    companion_project: Optional[str],
) -> Optional[str]:
    # Gemini CLI's Code Assist envelope matches our request shape, but its
    # actual traffic is serialized on the shared account/project lane instead of
    # being split by model id. Mirror that here to avoid fanout-only 429s.
    cleaned_access_token = _clean_codex_auth_value(access_token)
    cleaned_companion_project = _clean_codex_auth_value(companion_project)
    if cleaned_access_token is None or cleaned_companion_project is None:
        return None
    return _get_google_code_assist_prime_cache_key(
        cleaned_access_token,
        cleaned_companion_project,
    )


def _get_google_adapter_rate_limit_key(
    model: Optional[str],
    *,
    access_token: Optional[str] = None,
    companion_project: Optional[str] = None,
) -> str:
    shared_lane_key = _get_google_adapter_shared_lane_key(
        access_token=access_token,
        companion_project=companion_project,
    )
    if shared_lane_key is not None:
        return shared_lane_key
    normalized = _clean_codex_auth_value(model)
    if normalized is None:
        return "__default__"
    return normalized


def _get_google_adapter_rate_limit_key_from_kwargs(kwargs: dict[str, Any]) -> str:
    explicit_rate_limit_key = _clean_codex_auth_value(
        cast(Optional[str], kwargs.get("google_adapter_rate_limit_key"))
    )
    if explicit_rate_limit_key is not None:
        return explicit_rate_limit_key
    custom_body = kwargs.get("custom_body")
    model = custom_body.get("model") if isinstance(custom_body, dict) else None
    project = custom_body.get("project") if isinstance(custom_body, dict) else None
    access_token = cast(Optional[str], kwargs.get("google_access_token"))
    return _get_google_adapter_rate_limit_key(
        cast(Optional[str], model),
        access_token=access_token,
        companion_project=cast(Optional[str], project),
    )


def _get_google_adapter_semaphore(
    model: Optional[str] = None,
    *,
    access_token: Optional[str] = None,
    companion_project: Optional[str] = None,
    rate_limit_key: Optional[str] = None,
) -> asyncio.Semaphore:
    max_concurrent = _get_google_adapter_max_concurrent()
    resolved_rate_limit_key = _clean_codex_auth_value(rate_limit_key) or _get_google_adapter_rate_limit_key(
        model,
        access_token=access_token,
        companion_project=companion_project,
    )
    semaphore_key = (resolved_rate_limit_key, max_concurrent)
    semaphore = _google_adapter_semaphores.get(semaphore_key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max_concurrent)
        _google_adapter_semaphores[semaphore_key] = semaphore
    return semaphore


def _get_google_adapter_max_retries() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_RETRIES"))
    if raw_value is None:
        return 1
    try:
        parsed = int(raw_value)
    except Exception:
        return 1
    return max(0, parsed)


def _coerce_non_negative_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(0, parsed)


def _coerce_non_negative_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(0.0, parsed)


def _get_google_adapter_post_tool_cooldown_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_POST_TOOL_COOLDOWN_SECONDS")
    )
    if raw_value is None:
        return 0.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 0.0
    return max(0.0, parsed)


def _google_code_assist_unwrapped_chunk_contains_tool_call(
    unwrapped: dict[str, Any],
) -> bool:
    candidates = unwrapped.get("candidates")
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("functionCall"), dict):
                return True
    return False


def _get_google_adapter_max_output_tokens_cap() -> Optional[int]:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_MAX_OUTPUT_TOKENS_CAP")
    )
    if raw_value is None:
        return 8192
    try:
        parsed = int(raw_value)
    except Exception:
        return 8192
    if parsed <= 0:
        return None
    return parsed


def _get_google_adapter_default_thinking_level(model: Optional[str]) -> Optional[str]:
    disabled = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_DISABLE_DEFAULT_THINKING_CONFIG")
    )
    if isinstance(disabled, str) and disabled.lower() in {"1", "true", "yes", "on"}:
        return None

    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_DEFAULT_THINKING_LEVEL")
    )
    if raw_value:
        return raw_value

    normalized_model = (model or "").lower()
    if "flash-lite" in normalized_model:
        return "minimal"
    return "low"


def _get_google_adapter_max_contents_window() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_CONTENTS_WINDOW"))
    if raw_value is None:
        return 24
    try:
        parsed = int(raw_value)
    except Exception:
        return 24
    return max(2, parsed)


def _get_google_adapter_max_contents_text_chars() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_MAX_CONTENTS_TEXT_CHARS"))
    if raw_value is None:
        return 12000
    try:
        parsed = int(raw_value)
    except Exception:
        return 12000
    return max(1000, parsed)


def _estimate_google_content_text_chars(content_block: Any) -> int:
    if not isinstance(content_block, dict):
        return 0
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return 0
    total = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            total += len(text)
    return total


def _google_content_has_text(content_block: Any) -> bool:
    return _estimate_google_content_text_chars(content_block) > 0


def _google_content_has_function_exchange(content_block: Any) -> bool:
    if not isinstance(content_block, dict):
        return False
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("functionCall"), dict) or isinstance(
            part.get("function_call"), dict
        ):
            return True
        if isinstance(part.get("functionResponse"), dict) or isinstance(
            part.get("function_response"), dict
        ):
            return True
    return False


def _google_content_has_function_call(content_block: Any) -> bool:
    if not isinstance(content_block, dict):
        return False
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("functionCall"), dict) or isinstance(
            part.get("function_call"), dict
        ):
            return True
    return False


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
        nested_tool_use_id = (
            response_payload.get("tool_use_id")
            if isinstance(response_payload, dict)
            else None
        )
        for candidate in (function_response.get("id"), nested_tool_use_id):
            if isinstance(candidate, str) and candidate.strip():
                function_response_ids.add(candidate.strip())
    return function_response_ids


def _selected_google_contents_have_paired_function_responses(
    contents: list[Any],
    selected_indices: list[int],
) -> bool:
    seen_function_call_ids: set[str] = set()
    for index in selected_indices:
        content = contents[index]
        response_ids = _google_content_function_response_ids(content)
        if response_ids and not response_ids.issubset(seen_function_call_ids):
            return False
        seen_function_call_ids.update(_google_content_function_call_ids(content))
    return True


def _selected_google_contents_have_complete_function_exchanges(
    contents: list[Any],
    selected_indices: list[int],
) -> bool:
    seen_function_call_ids: set[str] = set()
    pending_function_call_ids: set[str] = set()
    for index in selected_indices:
        content = contents[index]
        response_ids = _google_content_function_response_ids(content)
        if response_ids and not response_ids.issubset(seen_function_call_ids):
            return False
        pending_function_call_ids.difference_update(response_ids)

        function_call_ids = _google_content_function_call_ids(content)
        seen_function_call_ids.update(function_call_ids)
        pending_function_call_ids.update(function_call_ids)
    return not pending_function_call_ids


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


def _add_required_google_function_call_pair_indices(
    contents: list[Any],
    selected_indices: list[int],
) -> list[int]:
    selected_index_set = set(selected_indices)
    for index in list(selected_indices):
        for function_response_id in _google_content_function_response_ids(
            contents[index]
        ):
            if any(
                function_response_id in _google_content_function_call_ids(
                    contents[prior_index]
                )
                for prior_index in selected_indices
                if prior_index < index
            ):
                continue
            paired_index = _find_prior_google_function_call_content_index(
                contents,
                before_index=index,
                function_response_id=function_response_id,
            )
            if paired_index is not None:
                selected_index_set.add(paired_index)
    return sorted(selected_index_set)


def _trim_google_content_indices_to_window(
    contents: list[Any],
    selected_indices: list[int],
    *,
    protected_text_indices: set[int],
    max_window: int,
) -> list[int]:
    selected_indices = list(selected_indices)
    while len(selected_indices) > max_window:
        removed = False
        for position, index in enumerate(selected_indices):
            if index in protected_text_indices:
                continue
            trial_indices = (
                selected_indices[:position] + selected_indices[position + 1 :]
            )
            if _selected_google_contents_have_complete_function_exchanges(
                contents,
                trial_indices,
            ):
                selected_indices = trial_indices
                removed = True
                break
        if removed:
            continue

        removable_position = next(
            (
                position
                for position, index in enumerate(selected_indices)
                if index not in protected_text_indices
            ),
            0,
        )
        selected_indices.pop(removable_position)

    while not _selected_google_contents_have_complete_function_exchanges(
        contents,
        selected_indices,
    ):
        for position, index in enumerate(selected_indices):
            response_ids = _google_content_function_response_ids(contents[index])
            prior_call_ids: set[str] = set()
            for prior_index in selected_indices[:position]:
                prior_call_ids.update(
                    _google_content_function_call_ids(contents[prior_index])
                )
            if not response_ids.issubset(prior_call_ids):
                selected_indices.pop(position)
                break

            function_call_ids = _google_content_function_call_ids(contents[index])
            later_response_ids: set[str] = set()
            for later_index in selected_indices[position + 1 :]:
                later_response_ids.update(
                    _google_content_function_response_ids(contents[later_index])
                )
            if function_call_ids and not function_call_ids.issubset(
                later_response_ids
            ):
                selected_indices.pop(position)
                break
        else:
            break
    return selected_indices


def _get_google_adapter_oversized_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_OVERSIZED_TEXT_PART_CHAR_CAP"))
    if raw_value is None:
        return 6000
    try:
        parsed = int(raw_value)
    except Exception:
        return 6000
    return max(1500, parsed)



def _get_google_adapter_pure_context_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_PURE_CONTEXT_TEXT_PART_CHAR_CAP")
    )
    if raw_value is None:
        return 6000
    try:
        parsed = int(raw_value)
    except Exception:
        return 6000
    return max(512, parsed)


def _get_google_adapter_subagent_context_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_SUBAGENT_CONTEXT_TEXT_PART_CHAR_CAP")
    )
    if raw_value is None:
        return 2000
    try:
        parsed = int(raw_value)
    except Exception:
        return 2000
    return max(512, parsed)


def _get_google_adapter_followup_subagent_context_text_part_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_SUBAGENT_CONTEXT_TEXT_PART_CHAR_CAP")
    )
    if raw_value is None:
        return 1200
    try:
        parsed = int(raw_value)
    except Exception:
        return 1200
    return max(256, parsed)


def _get_google_adapter_followup_allowed_tool_names() -> set[str]:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_ALLOWED_TOOL_NAMES")
    )
    if raw_value:
        allowed_tool_names = {
            item.strip()
            for item in raw_value.split(",")
            if isinstance(item, str) and item.strip()
        }
    else:
        allowed_tool_names = {
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Bash",
            "WebSearch",
            "WebFetch",
        }

    aliases = _get_google_code_assist_native_tool_aliases()
    expanded_tool_names = set(allowed_tool_names)
    for tool_name in list(allowed_tool_names):
        alias_name = aliases.get(tool_name)
        if isinstance(alias_name, str) and alias_name:
            expanded_tool_names.add(alias_name)

    return expanded_tool_names


def _get_openai_adapter_claude_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP")
    )
    if raw_value is None:
        return 1200
    try:
        parsed = int(raw_value)
    except Exception:
        return 1200
    return max(256, parsed)


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
            if isinstance(part.get("functionResponse"), dict) or isinstance(
                part.get("function_response"), dict
            ):
                return True
    return False


def _trim_google_adapter_followup_tools(request_block: dict[str, Any]) -> dict[str, Any]:
    if not _request_block_has_google_function_response(request_block):
        return {}

    allowed_tool_names = _get_google_adapter_followup_allowed_tool_names()
    tools = request_block.get("tools")
    if not isinstance(tools, list) or not allowed_tool_names:
        return {}

    original_decl_count = 0
    trimmed_decl_count = 0
    any_trimmed = False
    updated_tools: list[Any] = []

    for tool_entry in tools:
        if not isinstance(tool_entry, dict):
            updated_tools.append(tool_entry)
            continue
        key = None
        decls = tool_entry.get("functionDeclarations")
        if isinstance(decls, list):
            key = "functionDeclarations"
        else:
            decls = tool_entry.get("function_declarations")
            if isinstance(decls, list):
                key = "function_declarations"
        if key is None or not isinstance(decls, list):
            updated_tools.append(tool_entry)
            continue
        original_decl_count += len(decls)
        filtered_decls = []
        for decl in decls:
            if not isinstance(decl, dict):
                continue
            name = decl.get("name")
            if isinstance(name, str) and name in allowed_tool_names:
                filtered_decls.append(decl)
        trimmed_decl_count += len(filtered_decls)
        if len(filtered_decls) != len(decls):
            any_trimmed = True
        if filtered_decls:
            copied_entry = dict(tool_entry)
            copied_entry[key] = filtered_decls
            updated_tools.append(copied_entry)

    if not any_trimmed:
        return {}

    request_block["tools"] = updated_tools
    return {
        "trimmed_followup_function_declarations_from": original_decl_count,
        "trimmed_followup_function_declarations_to": trimmed_decl_count,
    }


def _is_google_function_call_allowed_predecessor(content_block: Any) -> bool:
    if not isinstance(content_block, dict):
        return False
    if content_block.get("role") == "user":
        return True
    return bool(_google_content_function_response_ids(content_block))


def _merge_google_model_content_parts(
    first_content: dict[str, Any],
    second_content: dict[str, Any],
) -> dict[str, Any]:
    first_parts = first_content.get("parts")
    second_parts = second_content.get("parts")
    merged = dict(first_content)
    merged["parts"] = [
        *(first_parts if isinstance(first_parts, list) else []),
        *(second_parts if isinstance(second_parts, list) else []),
    ]
    return merged


def _google_adapter_function_call_anchor_content() -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {
                "text": (
                    "[Gemini adapter inserted a conversation boundary before "
                    "a preserved historical tool call.]"
                )
            }
        ],
    }


def _repair_google_adapter_function_call_turn_adjacency(
    request_block: dict[str, Any],
) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return {}

    updated_contents: list[Any] = []
    merged_model_turn_count = 0
    inserted_anchor_count = 0
    changed = False

    for content in contents:
        updated_contents.append(content)
        if (
            not isinstance(content, dict)
            or content.get("role") != "model"
            or not _google_content_has_function_call(content)
        ):
            continue

        while (
            len(updated_contents) >= 2
            and isinstance(updated_contents[-1], dict)
            and isinstance(updated_contents[-2], dict)
            and updated_contents[-1].get("role") == "model"
            and updated_contents[-2].get("role") == "model"
        ):
            updated_contents[-2] = _merge_google_model_content_parts(
                updated_contents[-2],
                updated_contents[-1],
            )
            updated_contents.pop()
            merged_model_turn_count += 1
            changed = True

        current_index = len(updated_contents) - 1
        predecessor = (
            updated_contents[current_index - 1] if current_index > 0 else None
        )
        if not _is_google_function_call_allowed_predecessor(predecessor):
            updated_contents.insert(
                current_index,
                _google_adapter_function_call_anchor_content(),
            )
            inserted_anchor_count += 1
            changed = True

    if not changed:
        return {}

    request_block["contents"] = updated_contents
    changes: dict[str, Any] = {}
    if merged_model_turn_count:
        changes[
            "repaired_function_call_adjacency_merged_model_turn_count"
        ] = merged_model_turn_count
    if inserted_anchor_count:
        changes[
            "repaired_function_call_adjacency_inserted_user_anchor_count"
        ] = inserted_anchor_count
    return changes


def _split_google_adapter_inline_context_and_prompt(request_block: dict[str, Any]) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return {}

    updated_contents: list[Any] = []
    split_count = 0
    split_prompt_chars = 0

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        if content.get("role") != "user":
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if not isinstance(parts, list) or len(parts) != 1:
            updated_contents.append(content)
            continue
        part = parts[0]
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            updated_contents.append(content)
            continue
        text_value = part["text"]
        stripped_text = text_value.lstrip()
        if not stripped_text.startswith("<system-reminder>"):
            updated_contents.append(content)
            continue

        reminder_matches = list(re.finditer(r"<system-reminder>.*?</system-reminder>\n*", text_value, re.DOTALL))
        if not reminder_matches:
            updated_contents.append(content)
            continue
        trailing_text = text_value[reminder_matches[-1].end():].strip()
        if not trailing_text:
            updated_contents.append(content)
            continue

        split_count += 1
        split_prompt_chars += len(trailing_text)
        context_text = text_value[:reminder_matches[-1].end()].rstrip() + "\n"
        updated_contents.append({"role": "user", "parts": [{"text": context_text}]})
        updated_contents.append({"role": "user", "parts": [{"text": trailing_text}]})

    if split_count == 0:
        return {}

    request_block["contents"] = updated_contents
    return {
        "split_inline_context_prompt_count": split_count,
        "split_inline_context_prompt_chars": split_prompt_chars,
    }


def _compact_google_adapter_oversized_text_part(
    part: Any,
    *,
    cap: int,
    pure_context_cap: int,
    head_keep: int,
    tail_keep: int,
    is_followup_request: bool,
) -> tuple[Any, bool, dict[str, int]]:
    stats = {
        "original_text_chars": 0,
        "compacted_text_chars": 0,
        "compacted_count": 0,
        "pure_context_compacted_count": 0,
        "subagent_context_compacted_count": 0,
    }
    if not isinstance(part, dict) or not isinstance(part.get("text"), str):
        return part, False, stats

    text_value = part["text"]
    stats["original_text_chars"] = len(text_value)
    stripped_text = text_value.strip()
    reminder_matches = list(
        re.finditer(r"<system-reminder>.*?</system-reminder>\n*", text_value, re.DOTALL)
    )
    trailing_text = text_value[reminder_matches[-1].end():].strip() if reminder_matches else None
    is_reminder_only_context = bool(reminder_matches) and stripped_text.startswith("<system-reminder>") and not trailing_text
    is_subagent_context = (
        "SubagentStart hook additional context:" in text_value
        or "SubAgentStart hook additional context:" in text_value
    )
    reminder_only_context_cap = pure_context_cap if is_followup_request else cap
    if is_subagent_context:
        reminder_only_context_cap = (
            _get_google_adapter_followup_subagent_context_text_part_char_cap()
            if is_followup_request
            else _get_google_adapter_subagent_context_text_part_char_cap()
        )

    if is_reminder_only_context and len(text_value) > reminder_only_context_cap:
        updated_part = dict(part)
        updated_part["text"] = text_value[:reminder_only_context_cap].rstrip()
        stats["compacted_text_chars"] = len(updated_part["text"])
        stats["compacted_count"] = 1
        stats["pure_context_compacted_count"] = 1
        stats["subagent_context_compacted_count"] = int(is_subagent_context)
        return updated_part, True, stats

    if len(text_value) <= cap:
        stats["compacted_text_chars"] = len(text_value)
        return part, False, stats

    prefix = text_value[:head_keep].rstrip()
    suffix = text_value[-tail_keep:].lstrip()
    compacted_text = (
        f"{prefix}\n\n"
        f"[Gemini adapter compacted oversized user text from {len(text_value)} chars to preserve head/tail context.]\n\n"
        f"{suffix}"
    )
    updated_part = dict(part)
    updated_part["text"] = compacted_text
    stats["compacted_text_chars"] = len(compacted_text)
    stats["compacted_count"] = 1
    return updated_part, True, stats


def _compact_google_adapter_oversized_text_parts(request_block: dict[str, Any]) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return {}

    cap = _get_google_adapter_oversized_text_part_char_cap()
    pure_context_cap = _get_google_adapter_pure_context_text_part_char_cap()
    head_keep = max(512, cap // 3)
    tail_keep = max(1024, cap - head_keep - 64)
    updated_contents: list[Any] = []
    compacted_count = 0
    original_text_chars = 0
    compacted_text_chars = 0
    pure_context_compacted_count = 0
    subagent_context_compacted_count = 0
    is_followup_request = len(contents) > 2

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if content.get("role") != "user" or not isinstance(parts, list):
            updated_contents.append(content)
            continue

        updated_parts: list[Any] = []
        part_changed = False
        for part in parts:
            updated_part, changed, stats = _compact_google_adapter_oversized_text_part(
                part,
                cap=cap,
                pure_context_cap=pure_context_cap,
                head_keep=head_keep,
                tail_keep=tail_keep,
                is_followup_request=is_followup_request,
            )
            original_text_chars += stats["original_text_chars"]
            compacted_text_chars += stats["compacted_text_chars"]
            compacted_count += stats["compacted_count"]
            pure_context_compacted_count += stats["pure_context_compacted_count"]
            subagent_context_compacted_count += stats["subagent_context_compacted_count"]
            part_changed = part_changed or changed
            updated_parts.append(updated_part)

        if part_changed:
            if updated_parts:
                updated_content = dict(content)
                updated_content["parts"] = updated_parts
                updated_contents.append(updated_content)
            else:
                continue
        else:
            updated_contents.append(content)

    if compacted_count == 0:
        return {}

    request_block["contents"] = updated_contents
    changes = {
        "compacted_oversized_text_parts_count": compacted_count,
        "compacted_oversized_text_parts_cap": cap,
        "compacted_oversized_text_parts_before_chars": original_text_chars,
        "compacted_oversized_text_parts_after_chars": compacted_text_chars,
    }
    if pure_context_compacted_count > 0:
        changes["retained_followup_reminder_only_context_count"] = pure_context_compacted_count
        changes["compacted_pure_context_text_parts_count"] = pure_context_compacted_count
        changes["compacted_pure_context_text_parts_cap"] = (
            _get_google_adapter_followup_subagent_context_text_part_char_cap()
            if is_followup_request and subagent_context_compacted_count == pure_context_compacted_count
            else pure_context_cap
        )
    if subagent_context_compacted_count > 0:
        changes["subagent_context_text_parts_compacted_count"] = subagent_context_compacted_count
        changes["subagent_context_text_parts_cap"] = (
            _get_google_adapter_followup_subagent_context_text_part_char_cap()
            if is_followup_request
            else _get_google_adapter_subagent_context_text_part_char_cap()
        )
    return changes


def _apply_google_adapter_contents_window_policy(request_block: dict[str, Any]) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list) or len(contents) <= 2:
        return {}
    session_id = request_block.get("session_id")
    if not isinstance(session_id, str) or len(session_id) == 0:
        return {}

    original_count = len(contents)
    original_text_chars = sum(_estimate_google_content_text_chars(item) for item in contents)
    max_window = _get_google_adapter_max_contents_window()
    max_text_chars = _get_google_adapter_max_contents_text_chars()

    selected_indices = list(range(max(0, original_count - max_window), original_count))
    text_indices = [idx for idx, item in enumerate(contents) if _google_content_has_text(item)]
    protected_text_indices = text_indices[-2:]
    protected_text_index_set = set(protected_text_indices)
    selected_indices = _add_required_google_function_call_pair_indices(
        contents,
        sorted(set(protected_text_indices + selected_indices)),
    )
    protected_indices = sorted(
        index
        for index in selected_indices
        if _google_content_has_function_exchange(contents[index])
        or index in protected_text_index_set
    )
    if protected_indices:
        selected_indices = _trim_google_content_indices_to_window(
            contents,
            selected_indices,
            protected_text_indices=protected_text_index_set,
            max_window=max_window,
        )

    trimmed_contents = [contents[idx] for idx in selected_indices]
    protected_positions = {
        pos for pos, idx in enumerate(selected_indices) if idx in set(protected_indices)
    }
    trimmed_text_chars = sum(_estimate_google_content_text_chars(item) for item in trimmed_contents)
    while len(trimmed_contents) > 2 and trimmed_text_chars > max_text_chars:
        removable_pos = next(
            (pos for pos in range(len(trimmed_contents)) if pos not in protected_positions),
            None,
        )
        if removable_pos is None:
            break
        removed = trimmed_contents.pop(removable_pos)
        trimmed_text_chars -= _estimate_google_content_text_chars(removed)
        protected_positions = {
            pos - 1 if pos > removable_pos else pos
            for pos in protected_positions
            if pos != removable_pos
        }

    if len(trimmed_contents) == original_count and trimmed_text_chars == original_text_chars:
        return {}

    request_block["contents"] = trimmed_contents
    return {
        "trimmed_contents_from_count": original_count,
        "trimmed_contents_to_count": len(trimmed_contents),
        "trimmed_contents_from_text_chars": original_text_chars,
        "trimmed_contents_to_text_chars": trimmed_text_chars,
        "trimmed_contents_max_window": max_window,
        "trimmed_contents_max_text_chars": max_text_chars,
        "trimmed_contents_preserved_text_entries": len(protected_text_indices),
        "trimmed_contents_preserved_function_exchange_entries": len(
            [idx for idx in selected_indices if _google_content_has_function_exchange(contents[idx])]
        ),
    }


def _apply_google_adapter_generation_config_policy(
    request_block: dict[str, Any],
    *,
    model: Optional[str],
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    generation_config = request_block.get("generationConfig")
    if not isinstance(generation_config, dict):
        generation_config = {}
        request_block["generationConfig"] = generation_config

    if not isinstance(generation_config.get("thinkingConfig"), dict):
        default_thinking_level = _get_google_adapter_default_thinking_level(model)
        if default_thinking_level:
            generation_config["thinkingConfig"] = {
                "includeThoughts": False,
                "thinkingLevel": default_thinking_level,
            }
            changes["injected_default_thinking_config"] = True
            changes["injected_default_thinking_level"] = default_thinking_level

    max_output_tokens = generation_config.get("max_output_tokens")
    cap = _get_google_adapter_max_output_tokens_cap()
    thinking_config = generation_config.get("thinkingConfig")
    thinking_budget = (
        thinking_config.get("thinkingBudget")
        if isinstance(thinking_config, dict)
        else None
    )
    should_preserve_max_output_for_thinking = (
        isinstance(max_output_tokens, int)
        and not isinstance(max_output_tokens, bool)
        and isinstance(thinking_budget, int)
        and not isinstance(thinking_budget, bool)
        and thinking_budget > 0
        and max_output_tokens > thinking_budget
    )
    if (
        isinstance(max_output_tokens, int)
        and cap is not None
        and max_output_tokens > cap
        and should_preserve_max_output_for_thinking
    ):
        changes["preserved_oversized_max_output_tokens_for_thinking_budget"] = (
            max_output_tokens
        )
        changes["preserved_oversized_thinking_budget"] = thinking_budget
        changes["preserved_oversized_max_output_tokens_cap"] = cap
    elif isinstance(max_output_tokens, int) and cap is not None and max_output_tokens > cap:
        generation_config.pop("max_output_tokens", None)
        changes["removed_oversized_max_output_tokens_from"] = max_output_tokens
        changes["removed_oversized_max_output_tokens_cap"] = cap

    temperature = generation_config.get("temperature")
    if isinstance(temperature, (int, float)) and float(temperature) == 1.0:
        generation_config.pop("temperature", None)
        changes["removed_default_temperature"] = True

    if not generation_config:
        request_block.pop("generationConfig", None)
        changes["removed_empty_generation_config"] = True

    return changes


def _apply_google_adapter_request_shape_policy(payload: dict[str, Any]) -> dict[str, Any]:
    request_block = payload.get("request") if isinstance(payload.get("request"), dict) else None
    if not isinstance(request_block, dict):
        return {}

    changes: dict[str, Any] = {}
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    split_changes = _split_google_adapter_inline_context_and_prompt(request_block)
    if split_changes:
        changes.update(split_changes)
    followup_content_changes = _compact_google_adapter_followup_request_contents(request_block)
    if followup_content_changes:
        changes.update(followup_content_changes)
    followup_tool_changes = _trim_google_adapter_followup_tools(request_block)
    if followup_tool_changes:
        changes.update(followup_tool_changes)
    oversized_text_changes = _compact_google_adapter_oversized_text_parts(request_block)
    if oversized_text_changes:
        changes.update(oversized_text_changes)
    content_window_changes = _apply_google_adapter_contents_window_policy(request_block)
    if content_window_changes:
        changes.update(content_window_changes)
    function_call_adjacency_changes = (
        _repair_google_adapter_function_call_turn_adjacency(request_block)
    )
    if function_call_adjacency_changes:
        changes.update(function_call_adjacency_changes)
    generation_config_changes = _apply_google_adapter_generation_config_policy(
        request_block,
        model=model,
    )
    if generation_config_changes:
        changes.update(generation_config_changes)

    return changes


def _extract_google_adapter_exception_status_code(exc: Any) -> Optional[int]:
    for attr_name in ("status_code", "code"):
        raw_status = getattr(exc, attr_name, None)
        if isinstance(raw_status, int):
            return raw_status
        if isinstance(raw_status, str):
            try:
                return int(raw_status)
            except Exception:
                pass
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _extract_google_adapter_exception_detail(exc: Any) -> Any:
    for attr_name in ("detail", "message"):
        detail = getattr(exc, attr_name, None)
        if detail is not None:
            return detail
    response = getattr(exc, "response", None)
    if response is not None:
        response_content = getattr(response, "content", None)
        if response_content:
            return response_content
        response_text = getattr(response, "text", None)
        if response_text:
            return response_text
    return str(exc)


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
    return {
        str(header_name): str(header_value)
        for header_name, header_value in response_headers.items()
    }


def _get_adapter_header_value(
    headers: dict[str, Any], header_name: str
) -> Optional[str]:
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


def _parse_retry_after_seconds_from_headers(
    headers: dict[str, Any]
) -> Optional[float]:
    retry_after_value = _get_adapter_header_value(headers, "Retry-After")
    if retry_after_value is None:
        return None
    try:
        return max(0.0, float(retry_after_value))
    except Exception:
        return None


def _parse_rate_limit_reset_wait_seconds_from_headers(
    headers: dict[str, Any]
) -> Optional[float]:
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


def _parse_google_rate_limit_reset_seconds(exc: Any) -> float:
    upstream_headers = _extract_adapter_upstream_headers(exc)
    retry_after_seconds = _parse_retry_after_seconds_from_headers(upstream_headers)
    if retry_after_seconds is not None:
        return max(1.0, retry_after_seconds)
    reset_wait_seconds = _parse_rate_limit_reset_wait_seconds_from_headers(
        upstream_headers
    )
    if reset_wait_seconds is not None:
        return max(1.0, reset_wait_seconds)
    detail = _extract_google_adapter_exception_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    else:
        detail_text = str(detail)
    match = re.search(r"reset after\s+(\d+)s", detail_text)
    if match is None:
        return 5.0
    try:
        return max(1.0, float(match.group(1)))
    except Exception:
        return 5.0


def _extract_google_adapter_error_payloads(exc: Any) -> list[Any]:
    detail = _extract_google_adapter_exception_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    else:
        detail_text = str(detail)

    candidate_payloads: list[str] = [detail_text]
    brace_start = detail_text.find("{")
    brace_end = detail_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate_payloads.append(detail_text[brace_start : brace_end + 1])

    bracket_start = detail_text.find("[")
    bracket_end = detail_text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        candidate_payloads.append(detail_text[bracket_start : bracket_end + 1])

    bytes_literal_match = re.search(r'b([\'"]).*\1', detail_text, re.DOTALL)
    if bytes_literal_match is not None:
        try:
            literal_value = ast.literal_eval(bytes_literal_match.group(0))
            if isinstance(literal_value, bytes):
                candidate_payloads.append(
                    literal_value.decode("utf-8", errors="ignore")
                )
            else:
                candidate_payloads.append(str(literal_value))
        except Exception:
            pass

    parsed_payloads: list[Any] = []
    for candidate in candidate_payloads:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        parsed_payloads.append(parsed)
    return parsed_payloads


def _extract_google_adapter_error_reason(exc: Any) -> Optional[str]:
    for parsed in _extract_google_adapter_error_payloads(exc):
        error_blocks: list[dict[str, Any]] = []
        if isinstance(parsed, dict):
            error_block = parsed.get("error")
            if isinstance(error_block, dict):
                error_blocks.append(error_block)
        elif isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                error_block = item.get("error")
                if isinstance(error_block, dict):
                    error_blocks.append(error_block)
        for error_block in error_blocks:
            details = error_block.get("details")
            if not isinstance(details, list):
                continue
            for item in details:
                if not isinstance(item, dict):
                    continue
                reason = item.get("reason")
                if isinstance(reason, str) and reason:
                    return reason
    return None


def _extract_google_adapter_error_payload_for_logging(exc: Any) -> dict[str, Any]:
    for parsed in _extract_google_adapter_error_payloads(exc):
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            return dict(parsed)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("error"), dict):
                    return dict(item)
            return {"payload": parsed}
    return {}


def _record_google_adapter_error_for_logging(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    error_reason: Optional[str],
    attempt: int,
    wait_seconds: float,
) -> None:
    custom_body = passthrough_kwargs.get("custom_body")
    if not isinstance(custom_body, dict):
        return
    metadata = custom_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        custom_body["litellm_metadata"] = metadata

    payload = _extract_google_adapter_error_payload_for_logging(exc)
    if not isinstance(payload.get("error"), dict):
        detail = _extract_google_adapter_exception_detail(exc)
        if isinstance(detail, bytes):
            detail_text = detail.decode("utf-8", errors="ignore")
        else:
            detail_text = str(detail)
        synthesized_error: dict[str, Any] = {
            "code": status_code,
            "message": detail_text[:1000],
        }
        if status_code == 429:
            synthesized_error["status"] = "RESOURCE_EXHAUSTED"
        if error_reason:
            synthesized_error["details"] = [{"reason": error_reason}]
        payload["error"] = synthesized_error

    payload["source"] = "google_generate_content_error"
    payload["adapter_attempt"] = attempt
    payload["adapter_wait_seconds"] = wait_seconds
    payload["adapter_error_reason"] = error_reason
    metadata["google_generate_content_error"] = payload
    metadata["google_generate_content_error_count"] = (
        int(metadata.get("google_generate_content_error_count") or 0) + 1
    )


def _get_google_adapter_model_capacity_max_retries() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_MAX_RETRIES")
    )
    if raw_value is None:
        return 3
    try:
        parsed = int(raw_value)
    except Exception:
        return 3
    return max(0, parsed)


def _get_google_adapter_capacity_backoff_seconds(attempt: int) -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_MODEL_CAPACITY_BACKOFF_SECONDS")
    )
    if raw_value:
        try:
            values = [max(1.0, float(item.strip())) for item in raw_value.split(",") if item.strip()]
        except Exception:
            values = []
        if values:
            index = min(max(1, attempt) - 1, len(values) - 1)
            return values[index]
    schedule = (5.0, 15.0, 30.0, 60.0)
    index = min(max(1, attempt) - 1, len(schedule) - 1)
    return schedule[index]




def _get_google_adapter_hidden_retry_budget_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS")
    )
    if raw_value is None:
        return 0.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 0.0
    return max(0.0, parsed)


_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES = frozenset(
    PASSTHROUGH_PRE_FIRST_BYTE_RETRYABLE_STATUS_CODES
)


def _get_google_adapter_transient_retry_max_attempts() -> int:
    return len(PASSTHROUGH_PRE_FIRST_BYTE_RETRY_BACKOFF_SECONDS) + 1


def _get_google_adapter_transient_backoff_seconds(attempt: int) -> float:
    return _get_passthrough_hidden_retry_wait_seconds(max(0, attempt - 1))


def _is_google_adapter_transient_retryable_failure(
    exc: Any,
    *,
    status_code: Optional[int],
    error_reason: Optional[str],
) -> bool:
    if status_code == 429 or error_reason in {
        "MODEL_CAPACITY_EXHAUSTED",
        "RATE_LIMIT_EXCEEDED",
    }:
        return False
    if status_code in _GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES:
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    _status_code, failure_class, _failure_classification = (
        _classify_passthrough_hidden_retry_failure(exc)
    )
    return failure_class in {
        "upstream_connectivity_failure",
        "transport_dns_failure",
    }


def _google_adapter_hidden_retry_kwargs_from_passthrough_kwargs(
    passthrough_kwargs: dict[str, Any],
) -> dict[str, Any]:
    custom_body = passthrough_kwargs.get("custom_body")
    if not isinstance(custom_body, dict):
        return {}
    metadata = custom_body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        custom_body["litellm_metadata"] = metadata
    return {"litellm_params": {"metadata": metadata}}


def _record_google_adapter_hidden_retry_metadata(
    passthrough_kwargs: dict[str, Any],
    *,
    attempt_number: int,
    max_attempts: int,
    status_code: Optional[int],
    failure_class: str,
    wait_seconds: float,
    final_outcome: Optional[str] = None,
    failure_classification: Optional[str] = None,
) -> None:
    kwargs = _google_adapter_hidden_retry_kwargs_from_passthrough_kwargs(
        passthrough_kwargs
    )
    if not kwargs:
        return
    _record_passthrough_hidden_retry_metadata(
        kwargs,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        status_code=status_code,
        failure_class=failure_class,
        wait_seconds=wait_seconds,
        final_outcome=final_outcome,
        failure_classification=failure_classification,
    )


def _record_google_adapter_terminal_transient_failure_metadata(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    attempt: int,
    max_attempts: int,
    status_code: Optional[int],
    error_reason: Optional[str],
    failure_class: str,
    failure_classification: Optional[str],
) -> None:
    _record_google_adapter_error_for_logging(
        passthrough_kwargs,
        exc=exc,
        status_code=status_code,
        error_reason=error_reason,
        attempt=attempt,
        wait_seconds=0.0,
    )
    _record_google_adapter_hidden_retry_metadata(
        passthrough_kwargs,
        attempt_number=attempt,
        max_attempts=max_attempts,
        status_code=status_code,
        failure_class=failure_class,
        wait_seconds=0.0,
        final_outcome=(
            "failed_after_retry" if attempt > 1 else "failed_without_retry"
        ),
        failure_classification=failure_classification,
    )


def _google_adapter_hidden_retry_metadata(
    passthrough_kwargs: dict[str, Any],
) -> dict[str, Any]:
    custom_body = passthrough_kwargs.get("custom_body")
    if not isinstance(custom_body, dict):
        return {}
    metadata = custom_body.get("litellm_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _record_google_adapter_success_after_transient_retry(
    passthrough_kwargs: dict[str, Any],
    *,
    attempt: int,
    max_attempts: int,
) -> None:
    metadata = _google_adapter_hidden_retry_metadata(passthrough_kwargs)
    if not metadata.get("aawm_passthrough_hidden_retry_count"):
        return
    if metadata.get("aawm_passthrough_hidden_retry_final_outcome"):
        return
    _record_google_adapter_hidden_retry_metadata(
        passthrough_kwargs,
        attempt_number=attempt,
        max_attempts=max_attempts,
        status_code=None,
        failure_class="success",
        wait_seconds=0.0,
        final_outcome="success_after_retry",
    )


def _build_google_adapter_terminal_error_log_context(
    passthrough_kwargs: dict[str, Any],
    *,
    status_code: Optional[int],
    failure_classification: Optional[str],
) -> dict[str, Any]:
    metadata = _google_adapter_hidden_retry_metadata(passthrough_kwargs)
    request = passthrough_kwargs.get("request")
    endpoint = None
    if request is not None:
        try:
            endpoint = HttpPassThroughEndpointHelpers._get_passthrough_request_url_path(
                request
            )
        except Exception:
            endpoint = None
    custom_body = passthrough_kwargs.get("custom_body")
    model = None
    if isinstance(custom_body, dict):
        model = custom_body.get("model")
    return {
        "source": "google_code_assist_adapter",
        "endpoint": endpoint,
        "upstream_url": passthrough_kwargs.get("target"),
        "provider": passthrough_kwargs.get("custom_llm_provider")
        or metadata.get("custom_llm_provider"),
        "model": model or metadata.get("model_group"),
        "model_alias": metadata.get("requested_model_alias"),
        "route_family": metadata.get("passthrough_route_family"),
        "status_code": status_code,
        "failure_kind": (
            "transient_provider_connectivity"
            if failure_classification
            in {"transport_dns_failure", "upstream_connectivity_failure"}
            else "expected_upstream_capacity_or_internal"
        ),
        "hidden_retry_final_outcome": metadata.get(
            "aawm_passthrough_hidden_retry_final_outcome"
        ),
        "hidden_retry_failure_classification": failure_classification,
        "hidden_retry_count": metadata.get("aawm_passthrough_hidden_retry_count"),
        "trace_id": metadata.get("trace_id"),
    }


def _log_google_adapter_terminal_transient_failure(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    failure_classification: Optional[str],
) -> None:
    metadata = _google_adapter_hidden_retry_metadata(passthrough_kwargs)
    verbose_proxy_logger.error(
        (
            "Google adapter exhausted hidden retries for transient upstream "
            "failure status=%s error=%s final_outcome=%s retry_count=%s"
        ),
        status_code,
        str(exc),
        metadata.get("aawm_passthrough_hidden_retry_final_outcome"),
        metadata.get("aawm_passthrough_hidden_retry_count"),
        extra=_build_google_adapter_terminal_error_log_context(
            passthrough_kwargs,
            status_code=status_code,
            failure_classification=failure_classification,
        ),
        exc_info=True,
    )


async def _wait_for_google_adapter_cooldown_if_needed(rate_limit_key: str) -> None:
    async with _google_adapter_rate_limit_lock:
        now = time.monotonic()
        wait_seconds = _google_adapter_rate_limit_until_monotonic_by_key.get(rate_limit_key, 0.0) - now
    if wait_seconds > 0:
        verbose_proxy_logger.warning(
            "Google adapter cooldown active for %s; sleeping %.1fs before upstream request",
            rate_limit_key,
            wait_seconds,
        )
        await asyncio.sleep(wait_seconds)


async def _set_google_adapter_cooldown(rate_limit_key: str, wait_seconds: float) -> None:
    async with _google_adapter_rate_limit_lock:
        until = time.monotonic() + max(0.0, wait_seconds)
        current_until = _google_adapter_rate_limit_until_monotonic_by_key.get(rate_limit_key, 0.0)
        if until > current_until:
            _google_adapter_rate_limit_until_monotonic_by_key[rate_limit_key] = until


async def _handle_google_adapter_rate_limit_failure(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    error_reason: Optional[str],
    attempt: int,
    retry_limit: int,
    wait_seconds: float,
    rate_limit_key: str,
    accumulated_hidden_wait_seconds: float,
    hidden_retry_budget_seconds: float,
    is_capacity_retry: bool,
) -> float:
    _record_google_adapter_error_for_logging(
        passthrough_kwargs,
        exc=exc,
        status_code=status_code,
        error_reason=error_reason,
        attempt=attempt,
        wait_seconds=wait_seconds,
    )
    projected_hidden_wait_seconds = accumulated_hidden_wait_seconds + wait_seconds
    within_hidden_budget = (
        hidden_retry_budget_seconds > 0
        and projected_hidden_wait_seconds <= hidden_retry_budget_seconds
    )
    if attempt >= retry_limit and not within_hidden_budget:
        verbose_proxy_logger.warning(
            "Google adapter upstream attempt %s failed with %s (%s, reason=%s) and will not be retried",
            attempt,
            status_code,
            exc.__class__.__name__,
            error_reason,
        )
        raise exc
    if attempt >= retry_limit and within_hidden_budget:
        verbose_proxy_logger.warning(
            "Google adapter keeping 429 hidden from client for %s; hidden retry wait %.1fs/%.1fs (reason=%s)",
            rate_limit_key,
            projected_hidden_wait_seconds,
            hidden_retry_budget_seconds,
            error_reason,
        )
    if is_capacity_retry:
        verbose_proxy_logger.warning(
            "Google adapter upstream attempt %s hit 429 (%s, reason=%s); exponential backoff %.1fs",
            attempt,
            exc.__class__.__name__,
            error_reason,
            wait_seconds,
        )
    else:
        verbose_proxy_logger.warning(
            "Google adapter upstream attempt %s hit 429 (%s, reason=%s); parsed reset %.1fs",
            attempt,
            exc.__class__.__name__,
            error_reason,
            wait_seconds,
        )
    await _set_google_adapter_cooldown(rate_limit_key, wait_seconds + 1.0)
    return projected_hidden_wait_seconds


async def _handle_google_adapter_transient_failure(
    passthrough_kwargs: dict[str, Any],
    *,
    exc: Any,
    status_code: Optional[int],
    error_reason: Optional[str],
    attempt: int,
    transient_retry_max_attempts: int,
    failure_class: str,
    failure_classification: Optional[str],
) -> None:
    transient_wait_seconds = _get_google_adapter_transient_backoff_seconds(attempt)
    if attempt >= transient_retry_max_attempts:
        _record_google_adapter_terminal_transient_failure_metadata(
            passthrough_kwargs,
            exc=exc,
            attempt=attempt,
            max_attempts=transient_retry_max_attempts,
            status_code=status_code,
            error_reason=error_reason,
            failure_class=failure_class,
            failure_classification=failure_classification,
        )
        verbose_proxy_logger.warning(
            "Google adapter upstream attempt %s failed with transient %s (%s, reason=%s) and will not be retried",
            attempt,
            status_code,
            exc.__class__.__name__,
            error_reason,
        )
        _log_google_adapter_terminal_transient_failure(
            passthrough_kwargs,
            exc=exc,
            status_code=status_code,
            failure_classification=failure_classification,
        )
        raise exc
    verbose_proxy_logger.warning(
        "Google adapter upstream attempt %s hit transient %s (%s, reason=%s); hidden retry wait %.1fs",
        attempt,
        status_code,
        exc.__class__.__name__,
        error_reason,
        transient_wait_seconds,
    )
    _record_google_adapter_hidden_retry_metadata(
        passthrough_kwargs,
        attempt_number=attempt,
        max_attempts=transient_retry_max_attempts,
        status_code=status_code,
        failure_class=failure_class,
        wait_seconds=transient_wait_seconds,
        failure_classification=failure_classification,
    )
    await asyncio.sleep(transient_wait_seconds)


async def _perform_google_adapter_pass_through_request(**kwargs: Any) -> Response:
    passthrough_kwargs = dict(kwargs)
    max_retries = _coerce_non_negative_int(
        passthrough_kwargs.pop("google_adapter_max_retries", None),
        _get_google_adapter_max_retries(),
    )
    total_attempts = max_retries + 1
    capacity_total_attempts = (
        _coerce_non_negative_int(
            passthrough_kwargs.pop("google_adapter_model_capacity_max_retries", None),
            _get_google_adapter_model_capacity_max_retries(),
        )
        + 1
    )
    hidden_retry_budget_seconds = _coerce_non_negative_float(
        passthrough_kwargs.pop("google_adapter_hidden_retry_budget_seconds", None),
        _get_google_adapter_hidden_retry_budget_seconds(),
    )
    accumulated_hidden_wait_seconds = 0.0
    rate_limit_key = _get_google_adapter_rate_limit_key_from_kwargs(kwargs)
    transient_retry_max_attempts = _get_google_adapter_transient_retry_max_attempts()
    passthrough_kwargs.pop("google_access_token", None)
    passthrough_kwargs.pop("google_adapter_rate_limit_key", None)
    attempt = 0
    while True:
        attempt += 1
        verbose_proxy_logger.debug(
            "Google adapter upstream attempt %s/%s",
            attempt,
            max(total_attempts, capacity_total_attempts, transient_retry_max_attempts),
        )
        await _wait_for_google_adapter_cooldown_if_needed(rate_limit_key)
        try:
            passthrough_kwargs["retryable_upstream_status_codes"] = sorted(
                {429, *_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES}
            )
            passthrough_kwargs["caller_managed_hidden_retry"] = True
            response = await pass_through_request(**passthrough_kwargs)
            _record_google_adapter_success_after_transient_retry(
                passthrough_kwargs,
                attempt=attempt,
                max_attempts=transient_retry_max_attempts,
            )
            return response
        except Exception as exc:
            status_code = _extract_google_adapter_exception_status_code(exc)
            error_reason = _extract_google_adapter_error_reason(exc)
            is_capacity_retry = error_reason == "MODEL_CAPACITY_EXHAUSTED"
            is_rate_limit_retry = status_code == 429 or error_reason in {
                "MODEL_CAPACITY_EXHAUSTED",
                "RATE_LIMIT_EXCEEDED",
            }
            is_transient_retry = _is_google_adapter_transient_retryable_failure(
                exc,
                status_code=status_code,
                error_reason=error_reason,
            )
            failure_class = exc.__class__.__name__
            failure_classification: Optional[str] = None
            if is_transient_retry:
                _status_code, failure_class, failure_classification = (
                    _classify_passthrough_hidden_retry_failure(exc)
                )
                if _status_code is not None and status_code is None:
                    status_code = _status_code
            retry_limit = capacity_total_attempts if is_capacity_retry else total_attempts
            if is_capacity_retry:
                wait_seconds = _get_google_adapter_capacity_backoff_seconds(attempt)
            else:
                wait_seconds = _parse_google_rate_limit_reset_seconds(exc)
            if is_rate_limit_retry:
                accumulated_hidden_wait_seconds = await _handle_google_adapter_rate_limit_failure(
                    passthrough_kwargs,
                    exc=exc,
                    status_code=status_code,
                    error_reason=error_reason,
                    attempt=attempt,
                    retry_limit=retry_limit,
                    wait_seconds=wait_seconds,
                    rate_limit_key=rate_limit_key,
                    accumulated_hidden_wait_seconds=accumulated_hidden_wait_seconds,
                    hidden_retry_budget_seconds=hidden_retry_budget_seconds,
                    is_capacity_retry=is_capacity_retry,
                )
                continue
            if is_transient_retry:
                await _handle_google_adapter_transient_failure(
                    passthrough_kwargs,
                    exc=exc,
                    status_code=status_code,
                    error_reason=error_reason,
                    attempt=attempt,
                    transient_retry_max_attempts=transient_retry_max_attempts,
                    failure_class=failure_class,
                    failure_classification=failure_classification,
                )
                continue
            verbose_proxy_logger.warning(
                "Google adapter upstream attempt %s failed with %s (%s, reason=%s) and will not be retried",
                attempt,
                status_code,
                exc.__class__.__name__,
                error_reason,
            )
            raise


def _get_openrouter_adapter_rate_limit_key(model: Optional[str]) -> str:
    cleaned_model = _clean_secret_string(model)
    return cleaned_model or "__default__"


def _is_openrouter_adapter_free_model(model: Optional[str]) -> bool:
    cleaned_model = _clean_secret_string(model)
    if not cleaned_model:
        return False
    return (
        cleaned_model == "openrouter/elephant-alpha"
        or cleaned_model == "openrouter/free"
        or cleaned_model.endswith(":free")
    )


def _get_openrouter_adapter_wait_keys(model: Optional[str]) -> str:
    return _get_openrouter_adapter_rate_limit_key(model)


def _extract_openrouter_adapter_exception_status_code(exc: Any) -> Optional[int]:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    text_value = str(exc)
    if "429" in text_value:
        return 429
    return None


def _extract_openrouter_adapter_error_payload(exc: Any) -> Optional[dict[str, Any]]:
    candidates = [getattr(exc, "detail", None), getattr(exc, "message", None), str(exc)]
    for candidate in candidates:
        if candidate is None:
            continue
        payload_texts: list[str] = []
        if isinstance(candidate, bytes):
            payload_texts.append(candidate.decode("utf-8", errors="ignore"))
        elif isinstance(candidate, str):
            payload_text = candidate.strip()
            if ": b'" in payload_text or ': b"' in payload_text:
                payload_text = payload_text.split(": ", 1)[-1].strip()
            if (payload_text.startswith("b'") and payload_text.endswith("'")) or (
                payload_text.startswith('b"') and payload_text.endswith('"')
            ):
                try:
                    literal_value = ast.literal_eval(payload_text)
                except Exception:
                    literal_value = None
                if isinstance(literal_value, bytes):
                    payload_text = literal_value.decode("utf-8", errors="ignore")
                elif isinstance(literal_value, str):
                    payload_text = literal_value
            payload_texts.append(payload_text)
            brace_start = payload_text.find("{")
            brace_end = payload_text.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                payload_texts.append(payload_text[brace_start : brace_end + 1])
        elif isinstance(candidate, dict):
            return candidate
        for payload_text in payload_texts:
            if not payload_text:
                continue
            try:
                parsed = json.loads(payload_text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _extract_openrouter_adapter_provider_name(exc: Any) -> Optional[str]:
    payload = _extract_openrouter_adapter_error_payload(exc)
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("error", {}).get("metadata")
    if isinstance(metadata, dict):
        provider_name = metadata.get("provider_name")
        if isinstance(provider_name, str) and provider_name:
            return provider_name
    return None


def _extract_openrouter_adapter_retry_after_seconds(exc: Any) -> Optional[float]:
    payload = _extract_openrouter_adapter_error_payload(exc)
    if isinstance(payload, dict):
        metadata = payload.get("error", {}).get("metadata")
        if isinstance(metadata, dict):
            retry_after_value = metadata.get("retry_after_seconds")
            try:
                if retry_after_value is not None:
                    return max(0.0, float(retry_after_value))
            except Exception:
                return None
    return _parse_retry_after_seconds_from_headers(
        _extract_openrouter_adapter_error_headers(exc)
    )


def _extract_openrouter_adapter_raw_message(exc: Any) -> Optional[str]:
    payload = _extract_openrouter_adapter_error_payload(exc)
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("error", {}).get("metadata")
    if isinstance(metadata, dict):
        raw_message = metadata.get("raw")
        if isinstance(raw_message, str) and raw_message:
            return raw_message
    error_message = payload.get("error", {}).get("message")
    if isinstance(error_message, str) and error_message:
        return error_message
    return None


def _is_openrouter_adapter_provider_raw_error(exc: Any) -> bool:
    payload = _extract_openrouter_adapter_error_payload(exc)
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return False
    raw_message = metadata.get("raw")
    provider_name = metadata.get("provider_name")
    error_message = error.get("message")
    return (
        isinstance(provider_name, str)
        and bool(provider_name.strip())
        and isinstance(raw_message, str)
        and raw_message.strip().upper() == "ERROR"
        and isinstance(error_message, str)
        and "provider" in error_message.lower()
    )


def _extract_openrouter_adapter_error_headers(exc: Any) -> dict[str, Any]:
    merged_headers = _extract_adapter_upstream_headers(exc)
    payload = _extract_openrouter_adapter_error_payload(exc)
    if not isinstance(payload, dict):
        return merged_headers
    metadata = payload.get("error", {}).get("metadata")
    if not isinstance(metadata, dict):
        return merged_headers
    headers = metadata.get("headers")
    if not isinstance(headers, dict):
        return merged_headers
    merged_headers.update(headers)
    return merged_headers


def _get_openrouter_adapter_header_value(
    headers: dict[str, Any], header_name: str
) -> Optional[str]:
    return _get_adapter_header_value(headers, header_name)


def _extract_openrouter_adapter_reset_wait_seconds(exc: Any) -> Optional[float]:
    headers = _extract_openrouter_adapter_error_headers(exc)
    return _parse_rate_limit_reset_wait_seconds_from_headers(headers)


def _is_openrouter_adapter_long_window_rate_limit(
    exc: Any,
    *,
    hidden_retry_budget_seconds: float,
) -> bool:
    threshold_seconds = max(hidden_retry_budget_seconds, 30.0)
    retry_after_seconds = _extract_openrouter_adapter_retry_after_seconds(exc)
    if retry_after_seconds is not None:
        return retry_after_seconds > threshold_seconds
    headers = _extract_openrouter_adapter_error_headers(exc)
    remaining_value = _get_openrouter_adapter_header_value(
        headers, "X-RateLimit-Remaining"
    )
    if remaining_value not in {"0", "0.0"}:
        return False
    reset_wait_seconds = _extract_openrouter_adapter_reset_wait_seconds(exc)
    if reset_wait_seconds is None:
        return False
    return reset_wait_seconds > threshold_seconds


def _get_openrouter_adapter_cooldown_keys(
    *, model: Optional[str], exc: Any
) -> str:
    return _get_openrouter_adapter_rate_limit_key(model)


def _get_openrouter_adapter_retry_wait_seconds(exc: Any, attempt: int) -> float:
    wait_seconds = _get_openrouter_adapter_backoff_seconds(attempt)
    retry_after_seconds = _extract_openrouter_adapter_retry_after_seconds(exc)
    if retry_after_seconds is not None:
        retry_after_backoff_seconds = min(max(retry_after_seconds + 1.0, 1.0), 60.0)
        return max(wait_seconds, retry_after_backoff_seconds)
    headers = _extract_openrouter_adapter_error_headers(exc)
    remaining_value = _get_openrouter_adapter_header_value(
        headers, "X-RateLimit-Remaining"
    )
    reset_wait_seconds = _extract_openrouter_adapter_reset_wait_seconds(exc)
    if remaining_value in {"0", "0.0"} and reset_wait_seconds is not None:
        reset_backoff_seconds = min(max(reset_wait_seconds + 1.0, 1.0), 60.0)
        return max(wait_seconds, reset_backoff_seconds)
    return wait_seconds


def _get_openrouter_adapter_max_retries() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_OPENROUTER_ADAPTER_MAX_RETRIES")
    )
    if raw_value is None:
        return 3
    try:
        parsed = int(raw_value)
    except Exception:
        return 3
    return max(0, parsed)


def _get_openrouter_adapter_backoff_seconds(attempt: int) -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_OPENROUTER_ADAPTER_BACKOFF_SECONDS")
    )
    if raw_value:
        try:
            values = [max(1.0, float(item.strip())) for item in raw_value.split(",") if item.strip()]
        except Exception:
            values = []
        if values:
            index = min(max(1, attempt) - 1, len(values) - 1)
            return values[index]
    schedule = (2.0, 10.0, 20.0, 30.0)
    index = min(max(1, attempt) - 1, len(schedule) - 1)
    return schedule[index]


def _get_openrouter_adapter_hidden_retry_budget_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_OPENROUTER_ADAPTER_HIDDEN_RETRY_BUDGET_SECONDS")
    )
    if raw_value is None:
        return 0.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 0.0
    return max(0.0, parsed)


def _get_openrouter_adapter_post_failure_cooldown_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_OPENROUTER_ADAPTER_POST_FAILURE_COOLDOWN_SECONDS")
    )
    if raw_value is None:
        return 60.0
    try:
        parsed = float(raw_value)
    except Exception:
        return 60.0
    return max(0.0, parsed)


async def _maybe_raise_openrouter_adapter_failure_circuit_open(
    adapter_model: Optional[str],
) -> None:
    rate_limit_key = _get_openrouter_adapter_rate_limit_key(adapter_model)
    async with _openrouter_adapter_rate_limit_lock:
        now = time.monotonic()
        wait_seconds = (
            _openrouter_adapter_failure_circuit_until_monotonic_by_key.get(rate_limit_key, 0.0)
            - now
        )
    if wait_seconds > 0:
        rounded_wait = max(1, int(wait_seconds))
        verbose_proxy_logger.warning(
            "OpenRouter adapter failure circuit open for %s; failing fast for %ss",
            rate_limit_key,
            rounded_wait,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"OpenRouter model {rate_limit_key} is temporarily cooling down after repeated provider 429s. "
                f"Retry after ~{rounded_wait}s."
            ),
        )


async def _openrouter_adapter_open_failure_circuit(
    adapter_model: Optional[str],
    *,
    exc: Any,
) -> None:
    rate_limit_key = _get_openrouter_adapter_rate_limit_key(adapter_model)
    cooldown_seconds = _get_openrouter_adapter_post_failure_cooldown_seconds()
    retry_after_seconds = _extract_openrouter_adapter_retry_after_seconds(exc)
    reset_wait_seconds = _extract_openrouter_adapter_reset_wait_seconds(exc)
    for candidate in (retry_after_seconds, reset_wait_seconds):
        if candidate is not None:
            cooldown_seconds = max(cooldown_seconds, candidate)
    cooldown_seconds = min(max(cooldown_seconds, 0.0), 300.0)
    async with _openrouter_adapter_rate_limit_lock:
        until = time.monotonic() + cooldown_seconds
        current_until = _openrouter_adapter_failure_circuit_until_monotonic_by_key.get(
            rate_limit_key, 0.0
        )
        if until > current_until:
            _openrouter_adapter_failure_circuit_until_monotonic_by_key[rate_limit_key] = until


def _clear_openrouter_adapter_failure_circuit(adapter_model: Optional[str]) -> None:
    rate_limit_key = _get_openrouter_adapter_rate_limit_key(adapter_model)
    _openrouter_adapter_failure_circuit_until_monotonic_by_key.pop(rate_limit_key, None)


async def _wait_for_openrouter_adapter_cooldown_if_needed(
    rate_limit_keys: Union[str, list[str], tuple[str, ...]]
) -> None:
    if isinstance(rate_limit_keys, str):
        normalized_keys = [rate_limit_keys]
    else:
        normalized_keys = [key for key in rate_limit_keys if isinstance(key, str) and key]
    if not normalized_keys:
        normalized_keys = ["__default__"]
    async with _openrouter_adapter_rate_limit_lock:
        now = time.monotonic()
        wait_seconds = max(
            (
                _openrouter_adapter_rate_limit_until_monotonic_by_key.get(key, 0.0) - now
                for key in normalized_keys
            ),
            default=0.0,
        )
    if wait_seconds > 0:
        verbose_proxy_logger.warning(
            "OpenRouter adapter cooldown active for %s; sleeping %.1fs before upstream request",
            ", ".join(normalized_keys),
            wait_seconds,
        )
        await asyncio.sleep(wait_seconds)


async def _set_openrouter_adapter_cooldown(
    rate_limit_keys: Union[str, list[str], tuple[str, ...]], wait_seconds: float
) -> None:
    if isinstance(rate_limit_keys, str):
        normalized_keys = [rate_limit_keys]
    else:
        normalized_keys = [key for key in rate_limit_keys if isinstance(key, str) and key]
    if not normalized_keys:
        normalized_keys = ["__default__"]
    async with _openrouter_adapter_rate_limit_lock:
        until = time.monotonic() + max(0.0, wait_seconds)
        for key in normalized_keys:
            current_until = _openrouter_adapter_rate_limit_until_monotonic_by_key.get(key, 0.0)
            if until > current_until:
                _openrouter_adapter_rate_limit_until_monotonic_by_key[key] = until


async def _perform_openrouter_completion_adapter_operation(
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    max_retries = _get_openrouter_adapter_max_retries()
    total_attempts = max_retries + 1
    hidden_retry_budget_seconds = _get_openrouter_adapter_hidden_retry_budget_seconds()
    accumulated_hidden_wait_seconds = 0.0
    wait_keys = _get_openrouter_adapter_wait_keys(adapter_model)
    await _maybe_raise_openrouter_adapter_failure_circuit_open(adapter_model)
    attempt = 0
    while True:
        attempt += 1
        verbose_proxy_logger.debug(
            "OpenRouter completion adapter upstream attempt %s/%s for model=%s",
            attempt,
            total_attempts,
            adapter_model,
        )
        await _wait_for_openrouter_adapter_cooldown_if_needed(wait_keys)
        try:
            result = await operation()
            _clear_openrouter_adapter_failure_circuit(adapter_model)
            return result
        except Exception as exc:
            status_code = _extract_openrouter_adapter_exception_status_code(exc)
            provider_name = _extract_openrouter_adapter_provider_name(exc)
            raw_message = _extract_openrouter_adapter_raw_message(exc)
            reset_wait_seconds = _extract_openrouter_adapter_reset_wait_seconds(exc)
            is_long_window_rate_limit = _is_openrouter_adapter_long_window_rate_limit(
                exc,
                hidden_retry_budget_seconds=hidden_retry_budget_seconds,
            )
            wait_seconds = _get_openrouter_adapter_retry_wait_seconds(exc, attempt)
            projected_hidden_wait_seconds = accumulated_hidden_wait_seconds + wait_seconds
            within_hidden_budget = (
                hidden_retry_budget_seconds > 0
                and projected_hidden_wait_seconds <= hidden_retry_budget_seconds
            )
            if status_code == 429 and is_long_window_rate_limit:
                cooldown_seconds = min(max(reset_wait_seconds or 0.0, 30.0), 300.0)
                verbose_proxy_logger.warning(
                    "OpenRouter completion adapter upstream attempt %s hit long-window 429 (%s, provider=%s, raw=%s, reset_wait=%.1fs) and will not be hidden-retried",
                    attempt,
                    exc.__class__.__name__,
                    provider_name,
                    raw_message,
                    reset_wait_seconds or 0.0,
                )
                await _set_openrouter_adapter_cooldown(
                    _get_openrouter_adapter_cooldown_keys(model=adapter_model, exc=exc),
                    cooldown_seconds,
                )
                await _openrouter_adapter_open_failure_circuit(adapter_model, exc=exc)
                raise
            if status_code != 429 or (attempt >= total_attempts and not within_hidden_budget):
                verbose_proxy_logger.warning(
                    "OpenRouter completion adapter upstream attempt %s failed with %s (%s, provider=%s, raw=%s) and will not be retried",
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    provider_name,
                    raw_message,
                )
                if status_code == 429:
                    await _openrouter_adapter_open_failure_circuit(adapter_model, exc=exc)
                raise
            if attempt >= total_attempts and within_hidden_budget:
                verbose_proxy_logger.warning(
                    "OpenRouter completion adapter keeping 429 hidden from client for model=%s; hidden retry wait %.1fs/%.1fs",
                    adapter_model,
                    projected_hidden_wait_seconds,
                    hidden_retry_budget_seconds,
                )
            verbose_proxy_logger.warning(
                "OpenRouter completion adapter upstream attempt %s hit 429 (%s, provider=%s, raw=%s); backoff %.1fs",
                attempt,
                exc.__class__.__name__,
                provider_name,
                raw_message,
                wait_seconds,
            )
            accumulated_hidden_wait_seconds = projected_hidden_wait_seconds
            await _set_openrouter_adapter_cooldown(
                _get_openrouter_adapter_cooldown_keys(model=adapter_model, exc=exc),
                wait_seconds,
            )


async def _perform_openrouter_adapter_pass_through_request(
    *,
    adapter_model: Optional[str],
    **kwargs: Any,
) -> Response:
    max_retries = _get_openrouter_adapter_max_retries()
    total_attempts = max_retries + 1
    hidden_retry_budget_seconds = _get_openrouter_adapter_hidden_retry_budget_seconds()
    accumulated_hidden_wait_seconds = 0.0
    model_rate_limit_key = _get_openrouter_adapter_rate_limit_key(adapter_model)
    wait_keys = _get_openrouter_adapter_wait_keys(adapter_model)
    await _maybe_raise_openrouter_adapter_failure_circuit_open(adapter_model)
    attempt = 0
    while True:
        attempt += 1
        verbose_proxy_logger.debug(
            "OpenRouter adapter upstream attempt %s/%s for model=%s",
            attempt,
            total_attempts,
            model_rate_limit_key,
        )
        await _wait_for_openrouter_adapter_cooldown_if_needed(wait_keys)
        try:
            result = await pass_through_request(
                **kwargs,
                retryable_upstream_status_codes=[429, 500, 502, 503, 504],
                caller_managed_hidden_retry=True,
            )
            _clear_openrouter_adapter_failure_circuit(adapter_model)
            return result
        except Exception as exc:
            status_code = _extract_openrouter_adapter_exception_status_code(exc)
            provider_name = _extract_openrouter_adapter_provider_name(exc)
            raw_message = _extract_openrouter_adapter_raw_message(exc)
            reset_wait_seconds = _extract_openrouter_adapter_reset_wait_seconds(exc)
            is_long_window_rate_limit = _is_openrouter_adapter_long_window_rate_limit(
                exc,
                hidden_retry_budget_seconds=hidden_retry_budget_seconds,
            )
            wait_seconds = _get_openrouter_adapter_retry_wait_seconds(exc, attempt)
            projected_hidden_wait_seconds = accumulated_hidden_wait_seconds + wait_seconds
            within_hidden_budget = (
                hidden_retry_budget_seconds > 0
                and projected_hidden_wait_seconds <= hidden_retry_budget_seconds
            )
            if status_code == 429 and is_long_window_rate_limit:
                cooldown_seconds = min(max(reset_wait_seconds or 0.0, 30.0), 300.0)
                verbose_proxy_logger.warning(
                    "OpenRouter adapter upstream attempt %s hit long-window 429 (%s, provider=%s, raw=%s, reset_wait=%.1fs) and will not be hidden-retried",
                    attempt,
                    exc.__class__.__name__,
                    provider_name,
                    raw_message,
                    reset_wait_seconds or 0.0,
                )
                await _set_openrouter_adapter_cooldown(
                    _get_openrouter_adapter_cooldown_keys(model=adapter_model, exc=exc),
                    cooldown_seconds,
                )
                await _openrouter_adapter_open_failure_circuit(adapter_model, exc=exc)
                raise
            if status_code != 429 or (attempt >= total_attempts and not within_hidden_budget):
                verbose_proxy_logger.warning(
                    "OpenRouter adapter upstream attempt %s failed with %s (%s, provider=%s, raw=%s) and will not be retried",
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    provider_name,
                    raw_message,
                )
                if status_code == 429:
                    await _openrouter_adapter_open_failure_circuit(adapter_model, exc=exc)
                raise
            if attempt >= total_attempts and within_hidden_budget:
                verbose_proxy_logger.warning(
                    "OpenRouter adapter keeping 429 hidden from client for model=%s; hidden retry wait %.1fs/%.1fs",
                    adapter_model,
                    projected_hidden_wait_seconds,
                    hidden_retry_budget_seconds,
                )
            verbose_proxy_logger.warning(
                "OpenRouter adapter upstream attempt %s hit 429 (%s, provider=%s, raw=%s); backoff %.1fs",
                attempt,
                exc.__class__.__name__,
                provider_name,
                raw_message,
                wait_seconds,
            )
            accumulated_hidden_wait_seconds = projected_hidden_wait_seconds
            await _set_openrouter_adapter_cooldown(
                _get_openrouter_adapter_cooldown_keys(model=adapter_model, exc=exc),
                wait_seconds,
            )


async def _prime_google_code_assist_session(
    access_token: str,
    companion_project: str,
    *,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> Optional[dict[str, Any]]:
    ttl_seconds = _get_google_code_assist_prime_ttl_seconds()
    prime_access_token_key = (
        f"{adapter_provider}:{access_token}"
        if adapter_provider != litellm.LlmProviders.GEMINI.value
        else access_token
    )
    cache_key = _get_google_code_assist_prime_cache_key(
        prime_access_token_key,
        companion_project,
    )
    async with _google_code_assist_prime_lock:
        if ttl_seconds > 0:
            cached_until = _google_code_assist_prime_until_monotonic_by_key.get(
                cache_key, 0.0
            )
            if cached_until > time.monotonic():
                if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
                    verbose_proxy_logger.info(
                        "Google adapter prime cache hit for project=%s",
                        companion_project,
                    )
                return _google_code_assist_prime_quota_by_key.get(cache_key)

        target_base = _get_code_assist_adapter_target_base(adapter_provider).rstrip("/")
        headers = _build_code_assist_adapter_native_headers(
            adapter_provider=adapter_provider,
            access_token=access_token,
            model=None,
            accept="application/json",
        )
        metadata = {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": companion_project,
        }
        preflight_requests = (
            (
                f"{target_base}/v1internal:retrieveUserQuota",
                {"project": companion_project},
            ),
            (
                f"{target_base}/v1internal:fetchAdminControls",
                {"project": companion_project},
            ),
            (
                f"{target_base}/v1internal:listExperiments",
                {"project": companion_project, "metadata": metadata},
            ),
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            sanitized_quota_response: Optional[dict[str, Any]] = None
            for url, body in preflight_requests:
                HttpPassThroughEndpointHelpers.validate_outgoing_egress(
                    url=url,
                    headers=headers,
                    credential_family="google",
                    expected_target_family="google",
                )
                try:
                    response = await client.post(url, headers=headers, json=body)
                except Exception:
                    continue
                try:
                    response_body = response.json()
                except Exception:
                    response_body = None
                capture_passthrough_shape(
                    mode="google_code_assist_preflight",
                    provider=adapter_provider,
                    url_route=url,
                    request_body=body,
                    response=response,
                    response_body=response_body,
                    response_content=response.content,
                    extra_metadata={
                        "direct_google_code_assist_preflight": True,
                        "code_assist_adapter_provider": adapter_provider,
                        "preflight_endpoint": url.rsplit(":", 1)[-1],
                    },
                )
                if "retrieveUserQuota" not in url:
                    continue
                quota_source = (
                    "antigravity_retrieve_user_quota"
                    if adapter_provider == "antigravity"
                    else "google_retrieve_user_quota"
                )
                sanitized_quota_response = _sanitize_google_code_assist_quota_for_logging(
                    response_body,
                    source=quota_source,
                )
        if ttl_seconds > 0:
            _google_code_assist_prime_until_monotonic_by_key[cache_key] = (
                time.monotonic() + ttl_seconds
            )
        if sanitized_quota_response:
            _google_code_assist_prime_quota_by_key[cache_key] = sanitized_quota_response
        return sanitized_quota_response



def _load_local_google_oauth_access_token() -> Optional[str]:
    auth_path = _get_anthropic_adapter_google_auth_file_path()
    if auth_path is None:
        return None

    try:
        auth_data = json.loads(auth_path.read_text())
    except Exception:
        return None

    access_token = _clean_codex_auth_value(auth_data.get("access_token"))
    if access_token is None:
        return None
    return access_token


def _get_anthropic_adapter_google_target_base() -> str:
    return os.getenv("CODE_ASSIST_ENDPOINT") or "https://cloudcode-pa.googleapis.com"


def _normalize_google_completion_adapter_model_name(model: str) -> str:
    normalized_model = model.strip()
    if normalized_model.startswith(("gemini/", "google/")):
        normalized_model = normalized_model.split("/", 1)[1]

    # Claude-facing agent configs use stable shorthand names or newer naming
    # variants; Google Code Assist currently serves the corresponding model ids.
    google_model_aliases = {
        "gemini-3.1": "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    }
    return google_model_aliases.get(normalized_model, normalized_model)


def _sanitize_google_schema_array_items(schema_node: Any) -> int:
    fix_count = 0
    if isinstance(schema_node, dict):
        if schema_node.get("type") == "array":
            items = schema_node.get("items")
            if not isinstance(items, dict) or not items.get("type"):
                schema_node["items"] = {"type": "string"}
                fix_count += 1
        for value in schema_node.values():
            fix_count += _sanitize_google_schema_array_items(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            fix_count += _sanitize_google_schema_array_items(item)
    return fix_count


def _merge_google_code_assist_schema_annotations(
    source: dict[str, Any],
    target: dict[str, Any],
) -> None:
    for key in ("description", "title", "default"):
        if key in source and key not in target:
            target[key] = copy.deepcopy(source[key])


def _simplify_google_code_assist_union_schema(schema_node: dict[str, Any]) -> int:  # noqa: PLR0915
    fix_count = 0
    for union_key in ("anyOf", "oneOf", "allOf"):
        variants = schema_node.get(union_key)
        if not isinstance(variants, list):
            continue
        dict_variants = [variant for variant in variants if isinstance(variant, dict)]
        if not dict_variants:
            schema_node.pop(union_key, None)
            fix_count += 1
            continue

        nullable = any(variant.get("type") == "null" for variant in dict_variants)
        non_null_variants = [
            variant for variant in dict_variants if variant.get("type") != "null"
        ]
        if len(non_null_variants) == 1:
            replacement = copy.deepcopy(non_null_variants[0])
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        string_variant = next(
            (
                variant
                for variant in non_null_variants
                if variant.get("type") == "string"
            ),
            None,
        )
        if string_variant is not None:
            replacement = {
                key: copy.deepcopy(value)
                for key, value in string_variant.items()
                if key in {"type", "description", "title", "enum", "default"}
            }
            replacement.setdefault("type", "string")
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        object_variants = [
            variant
            for variant in non_null_variants
            if variant.get("type") == "object"
            and isinstance(variant.get("properties"), dict)
        ]
        if object_variants:
            merged_properties: dict[str, Any] = {}
            for variant in object_variants:
                merged_properties.update(copy.deepcopy(variant.get("properties") or {}))
            replacement = {
                "type": "object",
                "properties": merged_properties,
            }
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        typed_variant = next(
            (
                variant
                for variant in non_null_variants
                if isinstance(variant.get("type"), str)
            ),
            None,
        )
        if typed_variant is not None:
            replacement = copy.deepcopy(typed_variant)
            _merge_google_code_assist_schema_annotations(schema_node, replacement)
            if nullable:
                replacement["nullable"] = True
            schema_node.clear()
            schema_node.update(replacement)
            fix_count += 1
            continue

        schema_node.pop(union_key, None)
        schema_node.setdefault("type", "object")
        schema_node.setdefault("properties", {})
        fix_count += 1
    return fix_count


def _sanitize_google_code_assist_union_schemas(schema_node: Any) -> int:
    fix_count = 0
    if isinstance(schema_node, dict):
        fix_count += _simplify_google_code_assist_union_schema(schema_node)
        for value in list(schema_node.values()):
            fix_count += _sanitize_google_code_assist_union_schemas(value)
    elif isinstance(schema_node, list):
        for item in schema_node:
            fix_count += _sanitize_google_code_assist_union_schemas(item)
    return fix_count


def _sanitize_google_code_assist_tool_schema(schema_node: Any) -> int:
    fix_count = 0
    if not isinstance(schema_node, dict):
        return fix_count

    fix_count += _sanitize_google_code_assist_union_schemas(schema_node)
    if schema_node.get("type") is None:
        schema_node["type"] = "object"
        fix_count += 1
    if schema_node.get("type") == "object" and not isinstance(
        schema_node.get("properties"), dict
    ):
        schema_node["properties"] = {}
        fix_count += 1

    fix_count += _sanitize_google_schema_array_items(schema_node)
    fix_count += _sanitize_openai_object_schema_properties(schema_node)
    return fix_count


def _extract_completion_message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts)


def _is_google_adapter_synthetic_tool_context_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return bool(
        _GOOGLE_ADAPTER_SYNTHETIC_TOOL_CONTEXT_PATTERN.fullmatch(text.strip())
    )


def _is_google_adapter_synthetic_tool_context_message(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) == 0:
        return False
    return _is_google_adapter_synthetic_tool_context_text(
        _extract_completion_message_text(message)
    )


def _get_google_adapter_fallback_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_FALLBACK_CONTEXT_CHAR_CAP")
    )
    if raw_value is None:
        return 2000
    try:
        parsed = int(raw_value)
    except Exception:
        return 2000
    return max(256, parsed)


def _inject_google_adapter_fallback_text_context(
    google_request_dict: dict[str, Any], completion_messages: list[dict[str, Any]]
) -> dict[str, Any]:
    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}
    if any(_google_content_has_text(content) for content in contents):
        return {}

    text_snippets: list[str] = []
    for message in reversed(completion_messages):
        if _is_google_adapter_synthetic_tool_context_message(message):
            continue
        text = _extract_completion_message_text(message).strip()
        if text:
            text_snippets.append(text)
        if len(text_snippets) >= 2:
            break
    if not text_snippets:
        return {}

    text_snippets.reverse()
    fallback_text = "\n\n".join(text_snippets)
    cap = _get_google_adapter_fallback_context_char_cap()
    if len(fallback_text) > cap:
        fallback_text = fallback_text[-cap:].lstrip()

    google_request_dict["contents"] = [
        {
            "role": "user",
            "parts": [{"text": fallback_text}],
        },
        *contents,
    ]
    return {
        "inserted_fallback_text_context": True,
        "inserted_fallback_text_context_chars": len(fallback_text),
        "inserted_fallback_text_context_sources": len(text_snippets),
    }


def _get_google_adapter_system_prompt_policy() -> str:
    raw_value = _clean_codex_auth_value(
        os.getenv(_GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_ENV)
    )
    if raw_value is None:
        return _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"0", "false", "disabled", "none", "off"}:
        return "off"
    if normalized_value in {"append", "replace_compact"}:
        return normalized_value
    return _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_DEFAULT


def _get_codex_google_code_assist_tool_contract_policy() -> str:
    raw_value = _clean_codex_auth_value(
        os.getenv(_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_ENV)
    )
    if raw_value is None:
        return _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"0", "false", "disabled", "none", "off"}:
        return "off"
    if normalized_value in {"1", "true", "enabled", "on", "append"}:
        return "append"
    return _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_DEFAULT


def _extract_google_adapter_system_text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {None, "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def _replace_google_adapter_system_message_text(
    message: dict[str, Any],
    rewritten_text: str,
) -> dict[str, Any]:
    updated_message = dict(message)
    content = updated_message.get("content")
    if isinstance(content, list):
        first_text_index: Optional[int] = None
        updated_content: list[Any] = []
        for index, part in enumerate(content):
            if (
                first_text_index is None
                and isinstance(part, dict)
                and part.get("type") in {None, "text"}
                and isinstance(part.get("text"), str)
            ):
                first_text_index = index
                updated_part = dict(part)
                updated_part["text"] = rewritten_text
                updated_content.append(updated_part)
                continue
            if (
                first_text_index is not None
                and isinstance(part, dict)
                and part.get("type") in {None, "text"}
                and isinstance(part.get("text"), str)
            ):
                continue
            updated_content.append(part)
        if first_text_index is not None:
            updated_message["content"] = updated_content
            return updated_message
    updated_message["content"] = rewritten_text
    return updated_message


def _append_codex_google_code_assist_tool_contract_to_system_text(
    system_text: str,
) -> str:
    stripped_system_text = system_text.strip()
    if _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT in stripped_system_text:
        return stripped_system_text
    if not stripped_system_text:
        return _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT
    return (
        f"{stripped_system_text}\n\n"
        f"{_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT}"
    )


def _apply_codex_google_code_assist_tool_contract_policy(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_mode = _get_codex_google_code_assist_tool_contract_policy()
    metadata = dict(completion_kwargs.get("metadata") or {})
    policy_metadata: dict[str, Any] = {
        "codex_google_code_assist_tool_contract_policy_name": (
            _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_NAME
        ),
        "codex_google_code_assist_tool_contract_policy": policy_mode,
        "codex_google_code_assist_tool_contract_policy_version": (
            _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_POLICY_VERSION
        ),
        "codex_google_code_assist_tool_contract_policy_applied": (
            policy_mode != "off"
        ),
    }
    if policy_mode != "off":
        policy_metadata[
            "codex_google_code_assist_tool_contract_prompt_chars"
        ] = len(_CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT)

    metadata.update(policy_metadata)
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        tags = []
    metadata["tags"] = list(
        dict.fromkeys(
            [
                *tags,
                "codex-google-code-assist-tool-contract-policy",
                f"codex-google-code-assist-tool-contract-policy:{policy_mode}",
            ]
        )
    )

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["metadata"] = metadata
    if policy_mode == "off":
        return updated_kwargs, policy_metadata

    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return updated_kwargs, policy_metadata

    updated_messages = list(messages)
    system_message_index: Optional[int] = None
    system_text: Optional[str] = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        candidate_text = _extract_google_adapter_system_text_from_content(
            message.get("content")
        )
        if isinstance(candidate_text, str):
            system_message_index = index
            system_text = candidate_text
            break

    if system_message_index is None or system_text is None:
        updated_messages.insert(
            0,
            {
                "role": "system",
                "content": _CODEX_GOOGLE_CODE_ASSIST_TOOL_CONTRACT_PROMPT,
            },
        )
    else:
        updated_messages[system_message_index] = (
            _replace_google_adapter_system_message_text(
                cast(dict[str, Any], updated_messages[system_message_index]),
                _append_codex_google_code_assist_tool_contract_to_system_text(
                    system_text
                ),
            )
        )
    updated_kwargs["messages"] = updated_messages
    return updated_kwargs, policy_metadata


def _is_google_adapter_claude_overhead_block(block: str) -> bool:
    stripped_block = block.strip()
    if not stripped_block:
        return False

    lowered_block = stripped_block.lower()
    if lowered_block.startswith(_ANTHROPIC_BILLING_HEADER_PREFIX):
        return True
    if any(marker in lowered_block for marker in _GOOGLE_ADAPTER_CLAUDE_OVERHEAD_MARKERS):
        return True
    if "claude code" in lowered_block and any(
        marker in lowered_block
        for marker in (
            "slash command",
            "task management",
            "todowrite",
            "tool use policy",
        )
    ):
        return True
    return False


def _strip_google_adapter_claude_system_overhead(
    system_text: str,
) -> tuple[str, int]:
    preserved_blocks: list[str] = []
    removed_chars = 0
    for block in re.split(r"\n{2,}", system_text):
        stripped_block = block.strip()
        if not stripped_block:
            continue
        if _is_google_adapter_claude_overhead_block(stripped_block):
            removed_chars += len(block)
            continue
        preserved_blocks.append(stripped_block)
    return "\n\n".join(preserved_blocks).strip(), removed_chars


def _build_google_adapter_system_prompt_policy_text(
    *,
    original_text: str,
    policy_mode: str,
) -> tuple[str, dict[str, Any]]:
    normalized_original_text = original_text.strip()
    if policy_mode == "off":
        rewritten_text = original_text
        preserved_text = original_text
        removed_chars = 0
    elif policy_mode == "append":
        preserved_text = normalized_original_text
        removed_chars = 0
        rewritten_text = (
            f"{_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT}\n\n"
            f"{_GOOGLE_ADAPTER_ORIGINAL_SYSTEM_PROMPT_HEADING}\n\n"
            f"{preserved_text}"
        ).strip()
    else:
        preserved_text, removed_chars = _strip_google_adapter_claude_system_overhead(
            normalized_original_text
        )
        if preserved_text:
            rewritten_text = (
                f"{_GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT}\n\n"
                f"{_GOOGLE_ADAPTER_PRESERVED_SYSTEM_PROMPT_HEADING}\n\n"
                f"{preserved_text}"
            )
        else:
            rewritten_text = _GOOGLE_ADAPTER_COMPACT_SYSTEM_PROMPT

    metadata = {
        "google_adapter_system_prompt_policy_name": _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_NAME,
        "google_adapter_system_prompt_policy": policy_mode,
        "google_adapter_system_prompt_policy_version": _GOOGLE_ADAPTER_SYSTEM_PROMPT_POLICY_VERSION,
        "google_adapter_system_prompt_original_chars": len(original_text),
        "google_adapter_system_prompt_rewritten_chars": len(rewritten_text),
        "google_adapter_system_prompt_removed_claude_overhead_chars": removed_chars,
        "google_adapter_system_prompt_preserved_instruction_chars": len(
            preserved_text
        ),
        "google_adapter_system_prompt_policy_applied": policy_mode != "off",
    }
    return rewritten_text, metadata


def _apply_google_adapter_system_prompt_policy(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    system_message_index: Optional[int] = None
    system_text: Optional[str] = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        candidate_text = _extract_google_adapter_system_text_from_content(
            message.get("content")
        )
        if isinstance(candidate_text, str):
            system_message_index = index
            system_text = candidate_text
            break
    if system_message_index is None or system_text is None:
        return completion_kwargs, {}

    policy_mode = _get_google_adapter_system_prompt_policy()
    rewritten_text, policy_metadata = _build_google_adapter_system_prompt_policy_text(
        original_text=system_text,
        policy_mode=policy_mode,
    )

    updated_kwargs = dict(completion_kwargs)
    updated_messages = list(messages)
    if policy_mode != "off":
        updated_messages[system_message_index] = _replace_google_adapter_system_message_text(
            cast(dict[str, Any], updated_messages[system_message_index]),
            rewritten_text,
        )
        updated_kwargs["messages"] = updated_messages

    metadata = updated_kwargs.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.update(policy_metadata)
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        tags = []
    metadata["tags"] = list(
        dict.fromkeys(
            [
                *tags,
                "google-adapter-system-prompt-policy",
                f"google-adapter-system-prompt-policy:{policy_mode}",
            ]
        )
    )
    updated_kwargs["metadata"] = metadata
    return updated_kwargs, policy_metadata


def _normalize_codex_openai_chat_kwargs_for_google_code_assist(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    developer_message_count = 0
    normalized_messages: list[Any] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "developer":
            updated_message = dict(message)
            updated_message["role"] = "system"
            normalized_messages.append(updated_message)
            developer_message_count += 1
        else:
            normalized_messages.append(message)

    if developer_message_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = normalized_messages
    return updated_kwargs, {
        "google_adapter_codex_developer_messages_as_system_count": (
            developer_message_count
        )
    }


def _is_anthropic_tool_use_content_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_use"


def _is_anthropic_tool_result_content_block(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    block_type = block.get("type")
    return block_type == "tool_result" or (
        isinstance(block_type, str) and block_type.endswith("_tool_result")
    )


def _has_codex_google_code_assist_anthropic_tool_replay_blocks(
    messages: list[Any],
) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            _is_anthropic_tool_use_content_block(block)
            or _is_anthropic_tool_result_content_block(block)
            for block in content
        ):
            return True
    return False


def _codex_google_code_assist_tool_result_content_to_openai_content(
    content: Any,
) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _codex_google_code_assist_anthropic_tool_use_to_openai_tool_call(
    *,
    block: dict[str, Any],
    message_index: int,
    content_index: int,
) -> dict[str, Any]:
    tool_use_id = block.get("id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid Anthropic tool_use block at "
                f"messages.{message_index}.content.{content_index}: "
                "missing required non-empty string tool_use.id"
            ),
        )
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    return {
        "id": tool_use_id.strip(),
        "type": "function",
        "function": {
            "name": str(block.get("name") or ""),
            "arguments": json.dumps(tool_input, ensure_ascii=False),
        },
    }


def _normalize_codex_google_code_assist_anthropic_assistant_message(
    *,
    message: dict[str, Any],
    message_index: int,
) -> tuple[dict[str, Any], int]:
    content = message.get("content")
    if not isinstance(content, list):
        return message, 0

    updated_message = dict(message)
    existing_tool_calls = updated_message.get("tool_calls")
    tool_calls = list(existing_tool_calls) if isinstance(existing_tool_calls, list) else []
    text_parts: list[dict[str, Any]] = []
    converted_tool_use_count = 0
    for content_index, block in enumerate(content):
        if _is_anthropic_tool_use_content_block(block):
            tool_calls.append(
                _codex_google_code_assist_anthropic_tool_use_to_openai_tool_call(
                    block=cast(dict[str, Any], block),
                    message_index=message_index,
                    content_index=content_index,
                )
            )
            converted_tool_use_count += 1
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(
                {
                    "type": "text",
                    "text": str(block.get("text") or ""),
                }
            )

    updated_message["tool_calls"] = tool_calls
    if text_parts:
        updated_message["content"] = (
            text_parts[0]["text"] if len(text_parts) == 1 else text_parts
        )
    else:
        updated_message["content"] = None
    return updated_message, converted_tool_use_count


def _codex_google_code_assist_anthropic_tool_result_to_openai_tool_message(
    *,
    block: dict[str, Any],
    message_index: int,
    content_index: int,
) -> dict[str, Any]:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid Anthropic tool_result block at "
                f"messages.{message_index}.content.{content_index}: "
                "missing required non-empty string tool_result.tool_use_id"
            ),
        )
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_use_id.strip(),
        "content": _codex_google_code_assist_tool_result_content_to_openai_content(
            block.get("content")
        ),
    }
    cache_control = block.get("cache_control")
    if cache_control is not None:
        tool_message["cache_control"] = cache_control
    return tool_message


def _normalize_codex_google_code_assist_anthropic_user_message(
    *,
    message: dict[str, Any],
    message_index: int,
) -> tuple[list[dict[str, Any]], int]:
    content = message.get("content")
    if not isinstance(content, list):
        return [message], 0

    remaining_user_content: list[Any] = []
    normalized_messages: list[dict[str, Any]] = []
    converted_tool_result_count = 0
    for content_index, block in enumerate(content):
        if not _is_anthropic_tool_result_content_block(block):
            remaining_user_content.append(block)
            continue
        normalized_messages.append(
            _codex_google_code_assist_anthropic_tool_result_to_openai_tool_message(
                block=cast(dict[str, Any], block),
                message_index=message_index,
                content_index=content_index,
            )
        )
        converted_tool_result_count += 1

    if remaining_user_content:
        updated_message = dict(message)
        updated_message["content"] = remaining_user_content
        normalized_messages.append(updated_message)
    return normalized_messages, converted_tool_result_count


def _build_codex_google_code_assist_anthropic_replay_changes(
    *,
    repaired_count: int,
    converted_tool_use_count: int,
    converted_tool_result_count: int,
) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    if repaired_count:
        changes["google_adapter_codex_repaired_anthropic_tool_replay_id_count"] = (
            repaired_count
        )
    if converted_tool_use_count:
        changes["google_adapter_codex_converted_anthropic_tool_use_count"] = (
            converted_tool_use_count
        )
    if converted_tool_result_count:
        changes["google_adapter_codex_converted_anthropic_tool_result_count"] = (
            converted_tool_result_count
        )
    return changes


def _normalize_codex_google_code_assist_anthropic_tool_replay(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list) or not _has_codex_google_code_assist_anthropic_tool_replay_blocks(
        messages
    ):
        return completion_kwargs, {}

    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    repaired_messages, repaired_count = (
        LiteLLMAnthropicMessagesAdapter.repair_missing_anthropic_tool_use_ids(
            messages
        )
    )

    normalized_messages: list[Any] = []
    converted_tool_use_count = 0
    converted_tool_result_count = 0

    for message_index, message in enumerate(repaired_messages):
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue

        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            normalized_messages.append(message)
            continue

        if role == "assistant" and any(
            _is_anthropic_tool_use_content_block(block) for block in content
        ):
            updated_message, message_tool_use_count = (
                _normalize_codex_google_code_assist_anthropic_assistant_message(
                    message=message,
                    message_index=message_index,
                )
            )
            converted_tool_use_count += message_tool_use_count
            normalized_messages.append(updated_message)
            continue

        if role == "user" and any(
            _is_anthropic_tool_result_content_block(block) for block in content
        ):
            new_messages, message_tool_result_count = (
                _normalize_codex_google_code_assist_anthropic_user_message(
                    message=message,
                    message_index=message_index,
                )
            )
            normalized_messages.extend(new_messages)
            converted_tool_result_count += message_tool_result_count
            continue

        normalized_messages.append(message)

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = normalized_messages
    return updated_kwargs, _build_codex_google_code_assist_anthropic_replay_changes(
        repaired_count=repaired_count,
        converted_tool_use_count=converted_tool_use_count,
        converted_tool_result_count=converted_tool_result_count,
    )


def _deterministic_codex_google_code_assist_tool_call_id(
    *,
    message_index: int,
    tool_call_index: int,
    tool_call: dict[str, Any],
) -> str:
    try:
        seed_payload = json.dumps(tool_call, sort_keys=True, default=str)
    except Exception:
        seed_payload = str(tool_call)
    seed = "|".join(
        (
            "codex-google-code-assist-tool-call-id",
            str(message_index),
            str(tool_call_index),
            seed_payload,
        )
    )
    return f"call_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:28]}"


def _next_codex_google_code_assist_tool_messages(
    messages: list[Any],
    *,
    message_index: int,
) -> list[tuple[int, dict[str, Any]]]:
    next_tool_messages: list[tuple[int, dict[str, Any]]] = []
    for next_index in range(message_index + 1, len(messages)):
        next_message = messages[next_index]
        if not isinstance(next_message, dict):
            continue
        if next_message.get("role") == "assistant":
            break
        if next_message.get("role") == "tool":
            next_tool_messages.append((next_index, next_message))
    return next_tool_messages


def _paired_codex_google_code_assist_tool_message(
    next_tool_messages: list[tuple[int, dict[str, Any]]],
    *,
    tool_call_index: int,
) -> tuple[int, dict[str, Any]] | None:
    if tool_call_index < len(next_tool_messages):
        return next_tool_messages[tool_call_index]
    return None


def _repair_codex_google_code_assist_tool_call_id(
    *,
    message_index: int,
    tool_call_index: int,
    tool_call: dict[str, Any],
    paired_tool_message: tuple[int, dict[str, Any]] | None,
    copy_message_at: Callable[[int], Optional[dict[str, Any]]],
) -> bool:
    existing_id = tool_call.get("id")
    if isinstance(existing_id, str) and existing_id.strip():
        return False

    paired_tool_call_id = (
        paired_tool_message[1].get("tool_call_id")
        if paired_tool_message is not None
        else None
    )
    repaired_id = (
        paired_tool_call_id.strip()
        if isinstance(paired_tool_call_id, str) and paired_tool_call_id.strip()
        else _deterministic_codex_google_code_assist_tool_call_id(
            message_index=message_index,
            tool_call_index=tool_call_index,
            tool_call=tool_call,
        )
    )

    assistant_copy = copy_message_at(message_index)
    if assistant_copy is None:
        return False
    copied_tool_calls = assistant_copy.get("tool_calls")
    if not isinstance(copied_tool_calls, list):
        return False
    copied_tool_call = copied_tool_calls[tool_call_index]
    if not isinstance(copied_tool_call, dict):
        return False
    copied_tool_call["id"] = repaired_id

    if paired_tool_message is not None and not (
        isinstance(paired_tool_call_id, str) and paired_tool_call_id.strip()
    ):
        tool_message_copy = copy_message_at(paired_tool_message[0])
        if tool_message_copy is not None:
            tool_message_copy["tool_call_id"] = repaired_id

    return True


def _repair_codex_google_code_assist_openai_tool_call_ids(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages = list(messages)
    copied_messages: set[int] = set()
    repaired_count = 0

    def copy_message_at(index: int) -> Optional[dict[str, Any]]:
        message = updated_messages[index]
        if not isinstance(message, dict):
            return None
        if index not in copied_messages:
            message = dict(message)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                message["tool_calls"] = [
                    dict(tool_call) if isinstance(tool_call, dict) else tool_call
                    for tool_call in tool_calls
                ]
            updated_messages[index] = message
            copied_messages.add(index)
        return cast(dict[str, Any], message)

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        next_tool_messages = _next_codex_google_code_assist_tool_messages(
            messages,
            message_index=message_index,
        )

        for tool_call_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            repaired = _repair_codex_google_code_assist_tool_call_id(
                message_index=message_index,
                tool_call_index=tool_call_index,
                tool_call=tool_call,
                paired_tool_message=_paired_codex_google_code_assist_tool_message(
                    next_tool_messages,
                    tool_call_index=tool_call_index,
                ),
                copy_message_at=copy_message_at,
            )
            if repaired:
                repaired_count += 1

    if repaired_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    return updated_kwargs, {
        "google_adapter_codex_repaired_openai_tool_call_id_count": repaired_count
    }


def _normalize_codex_google_code_assist_reasoning_effort(
    mappable_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reasoning_effort = mappable_params.get("reasoning_effort")
    if reasoning_effort != "xhigh":
        return mappable_params, {}
    updated_params = dict(mappable_params)
    updated_params["reasoning_effort"] = "high"
    return updated_params, {
        "google_adapter_codex_reasoning_effort_normalized_from": "xhigh",
        "google_adapter_codex_reasoning_effort_normalized_to": "high",
    }


def _normalize_google_code_assist_thinking_max_tokens(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thinking = completion_kwargs.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return completion_kwargs, {}

    budget_tokens = thinking.get("budget_tokens")
    max_tokens = completion_kwargs.get("max_tokens")
    if (
        not isinstance(budget_tokens, int)
        or isinstance(budget_tokens, bool)
        or budget_tokens <= 0
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens > budget_tokens
    ):
        return completion_kwargs, {}

    normalized_max_tokens = budget_tokens + 1024
    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["max_tokens"] = normalized_max_tokens
    return updated_kwargs, {
        "google_adapter_thinking_max_tokens_normalized": True,
        "google_adapter_thinking_budget_tokens": budget_tokens,
        "google_adapter_thinking_original_max_tokens": max_tokens,
        "google_adapter_thinking_normalized_max_tokens": normalized_max_tokens,
    }


def _remember_codex_google_code_assist_tool_call_name(
    tool_call_id: Any,
    function_name: Any,
    function_arguments: Any = None,
) -> None:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return
    if not isinstance(function_name, str) or not function_name:
        function_name = (
            _codex_google_code_assist_tool_call_name_cache.get(tool_call_id)
            or (
                _codex_google_code_assist_tool_call_name_cache.get(
                    tool_call_id.split("__thought__", 1)[0]
                )
                if "__thought__" in tool_call_id
                else None
            )
        )
        if not isinstance(function_name, str) or not function_name:
            return
    if (
        len(_codex_google_code_assist_tool_call_name_cache)
        >= _CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE
    ):
        try:
            oldest_key = next(iter(_codex_google_code_assist_tool_call_name_cache))
            _codex_google_code_assist_tool_call_name_cache.pop(oldest_key, None)
            _codex_google_code_assist_tool_call_arguments_cache.pop(oldest_key, None)
        except StopIteration:
            pass
    _codex_google_code_assist_tool_call_name_cache[tool_call_id] = function_name
    normalized_arguments = _normalize_codex_google_code_assist_tool_call_arguments(
        function_arguments
    )
    if normalized_arguments is None:
        return
    existing_arguments = _codex_google_code_assist_tool_call_arguments_cache.get(
        tool_call_id, ""
    )
    if not existing_arguments:
        _codex_google_code_assist_tool_call_arguments_cache[
            tool_call_id
        ] = normalized_arguments
    elif normalized_arguments.startswith(existing_arguments):
        _codex_google_code_assist_tool_call_arguments_cache[
            tool_call_id
        ] = normalized_arguments
    elif not existing_arguments.endswith(normalized_arguments):
        _codex_google_code_assist_tool_call_arguments_cache[
            tool_call_id
        ] = f"{existing_arguments}{normalized_arguments}"


def _normalize_codex_google_code_assist_tool_call_arguments(
    function_arguments: Any,
) -> Optional[str]:
    if function_arguments is None:
        return None
    if isinstance(function_arguments, str):
        return function_arguments
    if isinstance(function_arguments, (dict, list)):
        try:
            return json.dumps(function_arguments, separators=(",", ":"))
        except Exception:
            return None
    return None


def _lookup_codex_google_code_assist_tool_call_name(tool_call_id: Any) -> Optional[str]:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    cached_name = _codex_google_code_assist_tool_call_name_cache.get(tool_call_id)
    if cached_name:
        return cached_name
    if "__thought__" in tool_call_id:
        return _codex_google_code_assist_tool_call_name_cache.get(
            tool_call_id.split("__thought__", 1)[0]
        )
    return None


def _lookup_codex_google_code_assist_tool_call_arguments(
    tool_call_id: Any,
) -> Optional[str]:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return None
    cached_arguments = _codex_google_code_assist_tool_call_arguments_cache.get(
        tool_call_id
    )
    if cached_arguments is not None:
        return cached_arguments
    if "__thought__" in tool_call_id:
        return _codex_google_code_assist_tool_call_arguments_cache.get(
            tool_call_id.split("__thought__", 1)[0]
        )
    return None


def _infer_single_codex_google_code_assist_function_tool_name(
    tools: Any,
) -> Optional[str]:
    if not isinstance(tools, list):
        return None
    function_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
        else:
            name = tool.get("name")
        if isinstance(name, str) and name:
            function_names.append(name)
    if len(function_names) == 1:
        return function_names[0]
    return None


def _is_codex_google_code_assist_empty_text_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if not isinstance(content, list):
        return False
    if not content:
        return True
    for part in content:
        if isinstance(part, str):
            if part.strip():
                return False
            continue
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type not in (None, "text", "output_text"):
            return False
        text = part.get("text")
        if not isinstance(text, str) or text.strip():
            return False
    return True


def _previous_codex_google_code_assist_assistant_index(
    messages: list[Any],
    *,
    before_index: int,
) -> Optional[int]:
    for candidate_index in range(before_index - 1, -1, -1):
        candidate = messages[candidate_index]
        if isinstance(candidate, dict) and candidate.get("role") == "assistant":
            return candidate_index
    return None


def _previous_codex_google_code_assist_contiguous_assistant_index(
    messages: list[Any],
    *,
    before_index: int,
) -> Optional[int]:
    previous_assistant_index = _previous_codex_google_code_assist_assistant_index(
        messages,
        before_index=before_index,
    )
    if previous_assistant_index is None:
        return None
    for candidate_index in range(before_index - 1, previous_assistant_index, -1):
        candidate = messages[candidate_index]
        if not isinstance(candidate, dict) or candidate.get("role") != "tool":
            return None
    return previous_assistant_index


def _previous_codex_google_code_assist_tool_call(
    messages: list[Any],
    *,
    before_index: int,
    tool_call_id: str,
) -> Optional[dict[str, Any]]:
    previous_assistant_index = _previous_codex_google_code_assist_assistant_index(
        messages,
        before_index=before_index,
    )
    if previous_assistant_index is None:
        return None
    previous_assistant = messages[previous_assistant_index]
    if not isinstance(previous_assistant, dict):
        return None
    tool_calls = previous_assistant.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        if tool_call.get("id") == tool_call_id:
            return tool_call
    return None


def _codex_google_code_assist_tool_call_function_name(
    tool_call: Optional[dict[str, Any]],
) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _codex_google_code_assist_tool_call_function_arguments(
    tool_call: Optional[dict[str, Any]],
) -> Optional[str]:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    return _normalize_codex_google_code_assist_tool_call_arguments(
        function.get("arguments")
    )


def _build_codex_google_code_assist_synthetic_tool_call(
    *,
    tool_call_id: str,
    function_name: str,
    function_arguments: str,
) -> dict[str, Any]:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": function_arguments,
        },
    }


def _append_codex_google_code_assist_tool_call_to_assistant(
    *,
    assistant_message: dict[str, Any],
    synthetic_tool_call: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    updated_assistant = dict(assistant_message)
    tool_calls = updated_assistant.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    blank_text_suppressed = False
    if _is_codex_google_code_assist_empty_text_content(
        updated_assistant.get("content")
    ):
        updated_assistant.pop("content", None)
        blank_text_suppressed = True
    updated_assistant["tool_calls"] = [
        *tool_calls,
        synthetic_tool_call,
    ]
    return updated_assistant, blank_text_suppressed


def _build_codex_google_code_assist_tool_pair_repair_changes(
    *,
    repaired_count: int,
    inserted_count: int,
    blank_text_suppressed_count: int,
    repaired_names: set[str],
) -> dict[str, Any]:
    changes: dict[str, Any] = {
        "google_adapter_codex_repaired_missing_tool_call_names": sorted(
            repaired_names
        ),
    }
    if repaired_count:
        changes["google_adapter_codex_repaired_missing_tool_call_count"] = (
            repaired_count
        )
    if inserted_count:
        changes["google_adapter_codex_inserted_missing_tool_call_count"] = (
            inserted_count
        )
    if blank_text_suppressed_count:
        changes[
            "google_adapter_codex_repaired_blank_tool_call_text_suppressed_count"
        ] = blank_text_suppressed_count
    return changes


def _codex_google_code_assist_tool_result_message_content(
    message: dict[str, Any],
) -> str:
    return str(
        _codex_google_code_assist_tool_result_content_to_openai_content(
            message.get("content")
        )
        or ""
    ).strip()


def _codex_google_code_assist_orphan_tool_result_context_text(
    *,
    tool_call_id: str,
    content: str,
) -> str:
    normalized_content = content.strip()
    if not normalized_content:
        return (
            "Previous tool result context (unmapped tool call "
            f"{tool_call_id}): no output was recorded."
        )
    return (
        "Previous tool result context (unmapped tool call "
        f"{tool_call_id}):\n{normalized_content}"
    )


def _codex_google_code_assist_display_tool_call_id(tool_call_id: str) -> str:
    return tool_call_id.split("__thought__", 1)[0]


def _append_codex_google_code_assist_orphan_tool_result_context(
    *,
    messages: list[Any],
    index: int,
    context_text: str,
) -> None:
    if index > 0:
        previous_message = messages[index - 1]
        if (
            isinstance(previous_message, dict)
            and previous_message.get("role") == "user"
            and not _completion_message_has_tool_result(previous_message)
        ):
            updated_previous = dict(previous_message)
            previous_content = updated_previous.get("content")
            if isinstance(previous_content, str):
                if previous_content.strip():
                    updated_previous["content"] = (
                        f"{previous_content.rstrip()}\n\n{context_text}"
                    )
                else:
                    updated_previous["content"] = context_text
            elif previous_content is None:
                updated_previous["content"] = context_text
            else:
                updated_previous["content"] = context_text
            messages[index - 1] = updated_previous
            return

    messages.insert(
        index,
        {
            "role": "user",
            "content": context_text,
        },
    )


def _sanitize_codex_google_code_assist_orphan_tool_results(  # noqa: PLR0915
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages = list(messages)
    converted_count = 0
    removed_blank_assistant_count = 0
    converted_tool_call_ids: list[str] = []
    processed_tool_call_ids: set[str] = set()
    fallback_tool_name = _infer_single_codex_google_code_assist_function_tool_name(
        completion_kwargs.get("tools")
    )

    index = 0
    while index < len(updated_messages):
        message = updated_messages[index]
        if not isinstance(message, dict) or message.get("role") != "tool":
            index += 1
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            index += 1
            continue

        previous_assistant_index = _previous_codex_google_code_assist_assistant_index(
            updated_messages,
            before_index=index,
        )
        previous_assistant = (
            updated_messages[previous_assistant_index]
            if previous_assistant_index is not None
            else None
        )
        existing_tool_call_ids = (
            _completion_message_tool_call_ids(previous_assistant)
            if isinstance(previous_assistant, dict)
            else set()
        )
        if tool_call_id in existing_tool_call_ids:
            index += 1
            continue

        function_name = (
            _lookup_codex_google_code_assist_tool_call_name(tool_call_id)
            or fallback_tool_name
        )
        if function_name:
            index += 1
            continue

        normalized_tool_call_id = _codex_google_code_assist_display_tool_call_id(
            tool_call_id
        )
        if normalized_tool_call_id in processed_tool_call_ids:
            updated_messages.pop(index)
            converted_count += 1
            converted_tool_call_ids.append(normalized_tool_call_id)
            continue

        context_text = _codex_google_code_assist_orphan_tool_result_context_text(
            tool_call_id=normalized_tool_call_id,
            content=_codex_google_code_assist_tool_result_message_content(message),
        )
        updated_messages.pop(index)
        converted_count += 1
        converted_tool_call_ids.append(normalized_tool_call_id)
        processed_tool_call_ids.add(normalized_tool_call_id)

        if (
            previous_assistant_index is not None
            and isinstance(previous_assistant, dict)
            and previous_assistant.get("role") == "assistant"
            and not _completion_message_tool_call_ids(previous_assistant)
            and _is_codex_google_code_assist_empty_text_content(
                previous_assistant.get("content")
            )
        ):
            if previous_assistant_index < index:
                index -= 1
            updated_messages.pop(previous_assistant_index)
            removed_blank_assistant_count += 1

        _append_codex_google_code_assist_orphan_tool_result_context(
            messages=updated_messages,
            index=index,
            context_text=context_text,
        )
        index += 1

    if converted_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    changes: dict[str, Any] = {
        "google_adapter_codex_converted_orphan_tool_result_count": converted_count,
        "google_adapter_codex_converted_orphan_tool_result_ids": sorted(
            converted_tool_call_ids
        ),
    }
    if removed_blank_assistant_count:
        changes[
            "google_adapter_codex_removed_blank_assistant_before_orphan_tool_result_count"
        ] = removed_blank_assistant_count
    return updated_kwargs, changes


def _ensure_codex_google_code_assist_tool_results_have_calls(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages = list(messages)
    repaired_count = 0
    inserted_count = 0
    blank_text_suppressed_count = 0
    repaired_names: set[str] = set()
    fallback_tool_name = _infer_single_codex_google_code_assist_function_tool_name(
        completion_kwargs.get("tools")
    )

    index = 0
    while index < len(updated_messages):
        message = updated_messages[index]
        if not isinstance(message, dict) or message.get("role") != "tool":
            index += 1
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            index += 1
            continue
        previous_assistant_index = (
            _previous_codex_google_code_assist_contiguous_assistant_index(
                updated_messages,
                before_index=index,
            )
        )
        previous_tool_call = _previous_codex_google_code_assist_tool_call(
            updated_messages,
            before_index=index,
            tool_call_id=tool_call_id,
        )

        function_name = (
            _lookup_codex_google_code_assist_tool_call_name(tool_call_id)
            or fallback_tool_name
            or _codex_google_code_assist_tool_call_function_name(previous_tool_call)
        )
        if not function_name:
            index += 1
            continue
        function_arguments = (
            _lookup_codex_google_code_assist_tool_call_arguments(tool_call_id)
            or _codex_google_code_assist_tool_call_function_arguments(
                previous_tool_call
            )
            or "{}"
        )
        synthetic_tool_call = _build_codex_google_code_assist_synthetic_tool_call(
            tool_call_id=tool_call_id,
            function_name=function_name,
            function_arguments=function_arguments,
        )

        if previous_assistant_index is None:
            updated_messages.insert(
                index,
                {
                    "role": "assistant",
                    "tool_calls": [synthetic_tool_call],
                },
            )
            inserted_count += 1
            repaired_names.add(function_name)
            index += 2
            continue

        previous_assistant = updated_messages[previous_assistant_index]
        if not isinstance(previous_assistant, dict):
            index += 1
            continue
        existing_tool_call_ids = _completion_message_tool_call_ids(previous_assistant)
        if tool_call_id in existing_tool_call_ids:
            index += 1
            continue

        updated_assistant, blank_text_suppressed = (
            _append_codex_google_code_assist_tool_call_to_assistant(
                assistant_message=previous_assistant,
                synthetic_tool_call=synthetic_tool_call,
            )
        )
        if blank_text_suppressed:
            blank_text_suppressed_count += 1
        updated_messages[previous_assistant_index] = updated_assistant
        repaired_count += 1
        repaired_names.add(function_name)
        index += 1

    if repaired_count == 0 and inserted_count == 0:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    return updated_kwargs, _build_codex_google_code_assist_tool_pair_repair_changes(
        repaired_count=repaired_count,
        inserted_count=inserted_count,
        blank_text_suppressed_count=blank_text_suppressed_count,
        repaired_names=repaired_names,
    )


async def _build_google_code_assist_request_from_completion_kwargs(  # noqa: PLR0915
    *,
    completion_kwargs: dict[str, Any],
    adapter_model: str,
    project: str,
    request: Request,
    completion_kwargs_are_openai_chat: bool = False,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body

    google_model = _normalize_google_completion_adapter_model_name(adapter_model)
    if completion_kwargs_are_openai_chat:
        completion_kwargs = dict(completion_kwargs)
        (
            completion_kwargs,
            openai_chat_shape_changes,
        ) = _normalize_codex_openai_chat_kwargs_for_google_code_assist(
            completion_kwargs
        )
        (
            completion_kwargs,
            codex_anthropic_tool_replay_changes,
        ) = _normalize_codex_google_code_assist_anthropic_tool_replay(
            completion_kwargs
        )
        (
            completion_kwargs,
            codex_openai_tool_call_id_changes,
        ) = _repair_codex_google_code_assist_openai_tool_call_ids(
            completion_kwargs
        )
        (
            completion_kwargs,
            codex_orphan_tool_result_changes,
        ) = _sanitize_codex_google_code_assist_orphan_tool_results(
            completion_kwargs
        )
        (
            completion_kwargs,
            codex_tool_pair_changes,
        ) = _ensure_codex_google_code_assist_tool_results_have_calls(
            completion_kwargs
        )
        tool_name_mapping: dict[str, str] = {}
        anthropic_native_tool_replay_changes: dict[str, Any] = {}
    else:
        from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        (
            completion_kwargs,
            anthropic_native_tool_use_id_repaired_count,
        ) = _repair_anthropic_tool_use_ids_for_passthrough(completion_kwargs)
        _validate_anthropic_tool_blocks_for_passthrough(completion_kwargs)
        anthropic_native_tool_replay_changes = {}
        if anthropic_native_tool_use_id_repaired_count:
            anthropic_native_tool_replay_changes[
                "google_adapter_repaired_anthropic_native_tool_use_id_count"
            ] = anthropic_native_tool_use_id_repaired_count

        completion_kwargs, tool_name_mapping = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=completion_kwargs["max_tokens"],
                messages=completion_kwargs.get("messages") or [],
                model=google_model,
                metadata=completion_kwargs.get("metadata"),
                stop_sequences=completion_kwargs.get("stop_sequences"),
                stream=completion_kwargs.get("stream"),
                system=completion_kwargs.get("system"),
                temperature=completion_kwargs.get("temperature"),
                thinking=completion_kwargs.get("thinking"),
                tool_choice=completion_kwargs.get("tool_choice"),
                tools=completion_kwargs.get("tools"),
                top_k=completion_kwargs.get("top_k"),
                top_p=completion_kwargs.get("top_p"),
                output_format=completion_kwargs.get("output_format"),
                output_config=completion_kwargs.get("output_config"),
                extra_kwargs={
                    "custom_llm_provider": litellm.LlmProviders.GEMINI.value,
                    "metadata": completion_kwargs.get("metadata"),
                    "parallel_tool_calls": completion_kwargs.get("parallel_tool_calls"),
                    "response_format": completion_kwargs.get("response_format"),
                    "reasoning_effort": completion_kwargs.get("reasoning_effort"),
                    "frequency_penalty": completion_kwargs.get("frequency_penalty"),
                    "presence_penalty": completion_kwargs.get("presence_penalty"),
                    "seed": completion_kwargs.get("seed"),
                    "n": completion_kwargs.get("n"),
                },
            )
        )
        openai_chat_shape_changes = {}
        codex_tool_pair_changes = {}
        codex_orphan_tool_result_changes = {}
        codex_anthropic_tool_replay_changes = {}
        codex_openai_tool_call_id_changes = {}
    completion_kwargs, thinking_max_tokens_changes = _normalize_google_code_assist_thinking_max_tokens(
        completion_kwargs
    )
    completion_kwargs, system_prompt_policy_changes = _apply_google_adapter_system_prompt_policy(
        completion_kwargs
    )
    codex_tool_contract_policy_changes: dict[str, Any] = {}
    if completion_kwargs_are_openai_chat:
        (
            completion_kwargs,
            codex_tool_contract_policy_changes,
        ) = _apply_codex_google_code_assist_tool_contract_policy(
            completion_kwargs
        )
    completion_messages = list(completion_kwargs.get("messages") or [])
    (
        completion_messages,
        completion_message_window_changes,
    ) = _apply_google_adapter_completion_message_window(completion_messages)
    (
        completion_messages,
        tool_call_context_changes,
    ) = _inject_google_adapter_tool_call_context_text(completion_messages)
    if tool_call_context_changes:
        completion_message_window_changes = {
            **completion_message_window_changes,
            **tool_call_context_changes,
        }
    completion_kwargs["messages"] = completion_messages
    completion_kwargs, native_tool_alias_changes = _apply_google_code_assist_native_tool_aliases(
        completion_kwargs,
        tool_name_mapping,
    )
    completion_messages = list(completion_kwargs.get("messages") or [])

    mappable_params = {
        key: value
        for key, value in completion_kwargs.items()
        if key
        not in {
            "model",
            "messages",
            "metadata",
            "stream",
            "stream_options",
            "litellm_logging_obj",
            "custom_llm_provider",
            "api_key",
            "api_base",
            "user",
        }
        and value is not None
    }
    (
        mappable_params,
        reasoning_effort_policy_changes,
    ) = _normalize_codex_google_code_assist_reasoning_effort(mappable_params)
    gemini_optional_params = litellm.GoogleAIStudioGeminiConfig().map_openai_params(
        non_default_params=mappable_params,
        optional_params={},
        model=google_model,
        drop_params=False,
    )
    litellm_params: dict[str, Any] = {}
    metadata = completion_kwargs.get("metadata")
    if isinstance(metadata, dict):
        litellm_params["metadata"] = metadata

    google_request = _transform_request_body(
        messages=completion_messages,
        model=google_model,
        optional_params=gemini_optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )
    google_request_dict = _normalize_google_code_assist_httpx_payload(dict(google_request))
    claude_tool_response_id_changes = (
        _annotate_google_code_assist_claude_tool_response_ids(
            google_request_dict,
            completion_messages,
            google_model=google_model,
        )
    )
    claude_tool_pair_changes = (
        _insert_google_code_assist_missing_claude_function_call_pairs(
            google_request_dict,
            google_model=google_model,
        )
    )
    duplicate_tool_response_changes = (
        _annotate_google_code_assist_duplicate_tool_responses(
            google_request_dict,
            completion_messages,
        )
    )
    fallback_context_changes = _inject_google_adapter_fallback_text_context(
        google_request_dict,
        completion_messages,
    )
    system_instruction = google_request_dict.pop("system_instruction", None)
    if system_instruction is not None:
        google_request_dict["systemInstruction"] = system_instruction
    session_id, session_id_source = _resolve_google_adapter_session_id(
        request,
        completion_messages,
        google_model=google_model,
    )
    google_request_dict["session_id"] = session_id
    user_prompt_id = _resolve_google_adapter_user_prompt_id(
        request,
        completion_messages,
        google_model=google_model,
        session_id=session_id,
    )

    wrapped_request = {
        "model": google_model,
        "project": project,
        "user_prompt_id": user_prompt_id,
        "request": google_request_dict,
    }
    session_id_hash = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:8]
    if isinstance(metadata, dict) and metadata:
        wrapped_request["litellm_metadata"] = dict(metadata)
    litellm_metadata = wrapped_request.setdefault("litellm_metadata", {})
    litellm_metadata.setdefault("session_id", session_id)
    litellm_metadata["google_adapter_session_id"] = session_id
    litellm_metadata["google_adapter_session_id_source"] = session_id_source
    litellm_metadata["google_adapter_session_id_hash"] = session_id_hash
    if fallback_context_changes:
        completion_message_window_changes = {
            **completion_message_window_changes,
            **fallback_context_changes,
        }
    completion_message_window_changes = {
        **completion_message_window_changes,
        **openai_chat_shape_changes,
        **codex_anthropic_tool_replay_changes,
        **codex_openai_tool_call_id_changes,
        **codex_orphan_tool_result_changes,
        **codex_tool_pair_changes,
        **anthropic_native_tool_replay_changes,
        **reasoning_effort_policy_changes,
        **thinking_max_tokens_changes,
        **native_tool_alias_changes,
        **claude_tool_response_id_changes,
        **claude_tool_pair_changes,
        **duplicate_tool_response_changes,
        **system_prompt_policy_changes,
        **codex_tool_contract_policy_changes,
        "google_adapter_session_id_source": session_id_source,
        "google_adapter_session_id_hash": session_id_hash,
    }
    return wrapped_request, tool_name_mapping, completion_messages, gemini_optional_params, litellm_params, completion_message_window_changes


def _drop_codex_google_code_assist_non_function_tools(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body, []

    kept_tools: list[Any] = []
    dropped_tool_types: list[str] = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            kept_tools.append(tool)
            continue
        if isinstance(tool, dict):
            dropped_tool_types.append(str(tool.get("type") or "unknown"))
        else:
            dropped_tool_types.append(type(tool).__name__)

    if not dropped_tool_types:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["tools"] = kept_tools
    tool_choice = updated_body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") != "function":
        updated_body.pop("tool_choice", None)
    elif isinstance(tool_choice, str) and tool_choice not in {
        "auto",
        "none",
        "required",
    }:
        updated_body.pop("tool_choice", None)

    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=["codex-google-code-assist-tools-sanitized"],
        extra_fields={
            "codex_google_code_assist_dropped_response_tool_types": _dedupe_sorted_str_list(
                dropped_tool_types
            ),
        },
    ), dropped_tool_types


def _build_codex_google_code_assist_completion_kwargs(
    prepared_request_body: dict[str, Any],
    *,
    adapter_model: str,
) -> tuple[dict[str, Any], Any, ResponsesAPIOptionalRequestParams]:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    request_input = prepared_request_body.get("input") or ""
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {
            key: value
            for key, value in prepared_request_body.items()
            if key not in {"input", "model", "litellm_metadata"}
        },
    )
    litellm_metadata = dict(prepared_request_body.get("litellm_metadata") or {})
    completion_kwargs = cast(
        dict[str, Any],
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model=adapter_model,
            input=request_input,
            responses_api_request=responses_api_request,
            custom_llm_provider=litellm.LlmProviders.GEMINI.value,
            stream=bool(prepared_request_body.get("stream")),
            metadata=litellm_metadata,
        ),
    )
    completion_kwargs["metadata"] = litellm_metadata
    if not completion_kwargs.get("max_tokens"):
        completion_kwargs["max_tokens"] = _CODEX_GOOGLE_CODE_ASSIST_DEFAULT_MAX_TOKENS
    return completion_kwargs, request_input, responses_api_request


async def _prepare_codex_google_code_assist_adapter_request(
    *,
    request: Request,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
    adapter_provider: str = litellm.LlmProviders.GEMINI.value,
) -> SimpleNamespace:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    google_access_token = (
        await _load_valid_local_antigravity_access_token()
        if adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
        else await _load_valid_local_google_oauth_access_token()
    )
    google_project = await _get_or_load_google_code_assist_project(
        google_access_token,
        adapter_provider=adapter_provider,
    )
    google_quota_observation = await _prime_google_code_assist_session(
        google_access_token,
        google_project,
        adapter_provider=adapter_provider,
    )

    is_antigravity_adapter = (
        adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
    )
    route_family = (
        "codex_antigravity_code_assist_adapter"
        if is_antigravity_adapter
        else "codex_google_code_assist_adapter"
    )
    adapter_tag = (
        "codex-antigravity-code-assist-adapter"
        if is_antigravity_adapter
        else "codex-google-code-assist-adapter"
    )
    target_provider_label = "antigravity" if is_antigravity_adapter else "google"
    requested_model = prepared_request_body.get("model")
    google_target_base = _get_code_assist_adapter_target_base(adapter_provider)
    google_model = _normalize_google_completion_adapter_model_name(adapter_model)
    google_adapter_rate_limit_key = _get_google_adapter_rate_limit_key(
        google_model,
        access_token=google_access_token,
        companion_project=google_project,
    )
    if is_antigravity_adapter:
        google_adapter_rate_limit_key = f"antigravity:{google_adapter_rate_limit_key}"
    client_requested_stream = bool(prepared_request_body.get("stream"))
    target_endpoint_label = "/v1internal:streamGenerateContent"
    target_query_params = {"alt": "sse"}
    target_url = f"{google_target_base.rstrip('/')}{target_endpoint_label}"
    annotated_target_url = httpx.URL(target_url).copy_with(params=target_query_params)

    prepared_request_body, _dropped_tool_types = (
        _drop_codex_google_code_assist_non_function_tools(prepared_request_body)
    )
    prepared_request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(prepared_request_body, route_family),
        tags_to_add=[
            adapter_tag,
            f"codex-adapter-model:{google_model}",
            f"codex-adapter-target:{target_provider_label}:{target_endpoint_label}",
        ],
        extra_fields={
            "codex_adapter_model": google_model,
            "codex_adapter_original_model": requested_model,
            "codex_adapter_provider": adapter_provider,
            "codex_adapter_target_endpoint": (
                f"{target_provider_label}:{target_endpoint_label}"
            ),
            "codex_adapter_input_shape": "openai_responses",
            "codex_adapter_output_shape": "openai_responses",
            **(
                {"antigravity_code_assist": True}
                if is_antigravity_adapter
                else {}
            ),
            **(
                {"google_retrieve_user_quota": google_quota_observation}
                if google_quota_observation
                else {}
            ),
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=(
                        "codex.antigravity_code_assist_adapter"
                        if is_antigravity_adapter
                        else "codex.google_code_assist_adapter"
                    ),
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": google_model,
                        "adapter_provider": adapter_provider,
                        "stream": client_requested_stream,
                        "upstream_stream": True,
                    },
                )
            ],
        },
    )
    (
        completion_kwargs,
        codex_request_input,
        responses_api_request,
    ) = _build_codex_google_code_assist_completion_kwargs(
        prepared_request_body,
        adapter_model=google_model,
    )
    previous_response_id = responses_api_request.get("previous_response_id")
    if previous_response_id:
        completion_kwargs = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=str(previous_response_id),
            litellm_completion_request=completion_kwargs,
        )

    wrapped_request_body, tool_name_mapping, completion_messages, gemini_optional_params, litellm_params, completion_message_window_changes = await _build_google_code_assist_request_from_completion_kwargs(
        completion_kwargs=completion_kwargs,
        adapter_model=google_model,
        project=google_project,
        request=request,
        completion_kwargs_are_openai_chat=True,
    )
    if isinstance(prepared_request_body.get("litellm_metadata"), dict):
        # pass_through_request strips LiteLLM params before the HTTP send; keep
        # adapter metadata here so logging survives without reaching Code Assist.
        wrapped_request_body["litellm_metadata"] = {
            **dict(wrapped_request_body.get("litellm_metadata") or {}),
            **dict(prepared_request_body["litellm_metadata"]),
        }

    generation_policy_changes = _apply_google_adapter_request_shape_policy(
        wrapped_request_body
    )
    adapter_headers = _build_code_assist_adapter_native_headers(
        adapter_provider=adapter_provider,
        access_token=google_access_token,
        model=google_model,
        accept="*/*",
    )
    if isinstance(wrapped_request_body.get("litellm_metadata"), dict):
        if completion_message_window_changes:
            wrapped_request_body["litellm_metadata"][
                "google_adapter_completion_message_window"
            ] = completion_message_window_changes
        if generation_policy_changes:
            wrapped_request_body["litellm_metadata"][
                "google_adapter_request_shape_policy"
            ] = generation_policy_changes

    sanitized_schema_fix_count = _sanitize_google_code_assist_request_schemas(
        wrapped_request_body
    )
    _log_google_completion_adapter_debug(
        prepared_request_body=prepared_request_body,
        wrapped_request_body=wrapped_request_body,
        google_model=google_model,
        adapter_headers=adapter_headers,
        sanitized_schema_fix_count=sanitized_schema_fix_count,
        generation_policy_changes=generation_policy_changes,
    )

    return SimpleNamespace(
        adapter_headers=adapter_headers,
        annotated_target_url=annotated_target_url,
        client_requested_stream=client_requested_stream,
        codex_request_input=codex_request_input,
        completion_messages=completion_messages,
        gemini_optional_params=gemini_optional_params,
        google_adapter_rate_limit_key=google_adapter_rate_limit_key,
        google_model=google_model,
        is_stream=True,
        litellm_metadata=dict(wrapped_request_body.get("litellm_metadata") or {}),
        litellm_params=litellm_params,
        custom_llm_provider=adapter_provider,
        responses_api_request=responses_api_request,
        target_query_params=target_query_params,
        target_url=target_url,
        tool_name_mapping=tool_name_mapping,
        wrapped_request_body=wrapped_request_body,
    )


def _get_google_code_assist_native_tool_aliases() -> dict[str, str]:
    return {
        "Bash": "run_shell_command",
        "Read": "read_file",
        "Write": "write_file",
        "Edit": "replace",
        "Glob": "glob",
        "Grep": "grep_search",
        "WebFetch": "web_fetch",
        "WebSearch": "google_web_search",
    }


def _apply_google_code_assist_alias_to_function_block(
    function_block: dict[str, Any],
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[dict[str, Any], Optional[str]]:
    original_name = function_block.get("name")
    if not isinstance(original_name, str) or not original_name:
        return function_block, None

    alias_name = aliases.get(original_name)
    if not isinstance(alias_name, str) or not alias_name:
        return function_block, None

    updated_function = dict(function_block)
    updated_function["name"] = alias_name
    tool_name_mapping[alias_name] = tool_name_mapping.get(original_name, original_name)
    return updated_function, alias_name


def _apply_google_code_assist_alias_to_tool(
    tool: Any,
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[Any, Optional[str]]:
    if not isinstance(tool, dict):
        return tool, None
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        return tool, None

    updated_function, alias_name = _apply_google_code_assist_alias_to_function_block(
        dict(tool["function"]),
        aliases=aliases,
        tool_name_mapping=tool_name_mapping,
    )
    if alias_name is None:
        return tool, None

    updated_tool = dict(tool)
    updated_tool["function"] = updated_function
    return updated_tool, alias_name


def _apply_google_code_assist_aliases_to_tool_calls(
    tool_calls: Any,
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[Any, set[str]]:
    if not isinstance(tool_calls, list):
        return tool_calls, set()

    updated_tool_calls: list[Any] = []
    aliased_names: set[str] = set()
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            updated_tool_calls.append(tool_call)
            continue
        function_block = tool_call.get("function")
        if not isinstance(function_block, dict):
            updated_tool_calls.append(tool_call)
            continue

        updated_function, alias_name = _apply_google_code_assist_alias_to_function_block(
            function_block,
            aliases=aliases,
            tool_name_mapping=tool_name_mapping,
        )
        if alias_name is None:
            updated_tool_calls.append(tool_call)
            continue

        updated_tool_call = dict(tool_call)
        updated_tool_call["function"] = updated_function
        updated_tool_calls.append(updated_tool_call)
        aliased_names.add(alias_name)

    return updated_tool_calls, aliased_names


def _apply_google_code_assist_aliases_to_message(
    message: Any,
    *,
    aliases: dict[str, str],
    tool_name_mapping: dict[str, str],
) -> tuple[Any, set[str]]:
    if not isinstance(message, dict):
        return message, set()
    updated_tool_calls, aliased_names = _apply_google_code_assist_aliases_to_tool_calls(
        message.get("tool_calls"),
        aliases=aliases,
        tool_name_mapping=tool_name_mapping,
    )
    if not aliased_names:
        return message, set()

    updated_message = dict(message)
    updated_message["tool_calls"] = updated_tool_calls
    return updated_message, aliased_names


def _apply_google_code_assist_native_tool_aliases(
    completion_kwargs: dict[str, Any],
    tool_name_mapping: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = _get_google_code_assist_native_tool_aliases()
    alias_count = 0
    aliased_names: set[str] = set()

    tools = completion_kwargs.get("tools")
    if isinstance(tools, list):
        updated_tools = []
        for tool in tools:
            updated_tool, alias_name = _apply_google_code_assist_alias_to_tool(
                tool,
                aliases=aliases,
                tool_name_mapping=tool_name_mapping,
            )
            updated_tools.append(updated_tool)
            if alias_name is not None:
                alias_count += 1
                aliased_names.add(alias_name)
        completion_kwargs["tools"] = updated_tools

    messages = completion_kwargs.get("messages")
    if isinstance(messages, list):
        updated_messages = []
        for message in messages:
            updated_message, message_aliases = _apply_google_code_assist_aliases_to_message(
                message,
                aliases=aliases,
                tool_name_mapping=tool_name_mapping,
            )
            updated_messages.append(updated_message)
            aliased_names.update(message_aliases)
        completion_kwargs["messages"] = updated_messages

    tool_choice = completion_kwargs.get("tool_choice")
    if isinstance(tool_choice, dict):
        updated_tool_choice = dict(tool_choice)
        function_block = updated_tool_choice.get("function")
        if isinstance(function_block, dict):
            updated_function, alias_name = _apply_google_code_assist_alias_to_function_block(
                function_block,
                aliases=aliases,
                tool_name_mapping=tool_name_mapping,
            )
            if alias_name is not None:
                updated_tool_choice["function"] = updated_function
                completion_kwargs["tool_choice"] = updated_tool_choice
                aliased_names.add(alias_name)
                alias_count += 1

    changes: dict[str, Any] = {}
    if alias_count > 0 or aliased_names:
        changes["google_native_tool_alias_count"] = alias_count
        changes["google_native_tool_aliases"] = sorted(aliased_names)
    return completion_kwargs, changes


def _get_google_adapter_max_completion_messages_window() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_MAX_COMPLETION_MESSAGES_WINDOW")
    )
    if raw_value is None:
        return 12
    try:
        parsed = int(raw_value)
    except Exception:
        return 12
    return max(2, parsed)


def _completion_message_has_visible_text(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return True
    return False


def _inject_google_adapter_tool_call_context_text(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated_messages: list[dict[str, Any]] = []
    suppressed_count = 0

    for message in messages:
        if not isinstance(message, dict):
            updated_messages.append(message)
            continue
        if message.get("role") != "assistant":
            updated_messages.append(message)
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) == 0:
            updated_messages.append(message)
            continue
        if _completion_message_has_visible_text(message):
            if _is_google_adapter_synthetic_tool_context_message(message):
                updated_message = dict(message)
                updated_message["content"] = ""
                updated_messages.append(updated_message)
                suppressed_count += 1
                continue
            updated_messages.append(message)
            continue

        updated_messages.append(message)

    if suppressed_count == 0:
        return messages, {}
    return updated_messages, {
        "google_adapter_suppressed_tool_call_context_text_count": suppressed_count,
    }


def _estimate_completion_message_text_chars(message: Any) -> int:
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                total += len(text)
        return total
    return 0


def _completion_message_has_tool_result(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") == "tool":
        return True
    if isinstance(message.get("tool_call_id"), str):
        return True
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_result":
                return True
            if isinstance(part.get("tool_result"), dict):
                return True
    return False


def _completion_message_tool_call_ids(message: Any) -> set[str]:
    if not isinstance(message, dict):
        return set()
    tool_call_ids: set[str] = set()
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_call_ids.add(tool_call_id)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                tool_call_id = part.get("id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    tool_call_ids.add(tool_call_id)
    return tool_call_ids


def _completion_message_tool_result_ids(message: Any) -> set[str]:
    if not isinstance(message, dict):
        return set()
    tool_result_ids: set[str] = set()
    tool_call_id = message.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        tool_result_ids.add(tool_call_id)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_tool_use_id = part.get("tool_use_id")
            if isinstance(part_tool_use_id, str) and part_tool_use_id:
                tool_result_ids.add(part_tool_use_id)
            tool_result = part.get("tool_result")
            if isinstance(tool_result, dict):
                nested_tool_use_id = tool_result.get("tool_use_id")
                if isinstance(nested_tool_use_id, str) and nested_tool_use_id:
                    tool_result_ids.add(nested_tool_use_id)
    return tool_result_ids


def _trim_completion_message_tail_preserving_tool_pairs(
    messages: list[dict[str, Any]],
    tail_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tail_budget <= 0:
        return [], {}

    tail_start = max(0, len(messages) - tail_budget)
    boundary_adjustments = 0
    while tail_start < len(messages) and _completion_message_has_tool_result(
        messages[tail_start]
    ):
        tail_start += 1
        boundary_adjustments += 1

    while tail_start < len(messages):
        seen_tool_call_ids: set[str] = set()
        orphan_index: Optional[int] = None
        for index, message in enumerate(messages[tail_start:]):
            seen_tool_call_ids.update(_completion_message_tool_call_ids(message))
            tool_result_ids = _completion_message_tool_result_ids(message)
            if tool_result_ids and not tool_result_ids.issubset(seen_tool_call_ids):
                orphan_index = tail_start + index
                break
        if orphan_index is None:
            break
        tail_start = orphan_index + 1
        boundary_adjustments += 1
        while tail_start < len(messages) and _completion_message_has_tool_result(
            messages[tail_start]
        ):
            tail_start += 1
            boundary_adjustments += 1

    changes: dict[str, Any] = {}
    if boundary_adjustments:
        changes["trimmed_completion_messages_tool_pair_boundary_adjustments"] = (
            boundary_adjustments
        )
    return messages[tail_start:], changes


def _get_google_adapter_preserved_task_state_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_PRESERVED_TASK_STATE_CHAR_CAP")
    )
    if raw_value is None:
        return 6000
    try:
        parsed = int(raw_value)
    except Exception:
        return 6000
    return max(512, parsed)


def _extract_google_adapter_preserved_task_excerpt(text: str) -> str:
    text_value = text.strip()
    reminder_matches = list(
        re.finditer(r"<system-reminder>.*?</system-reminder>\n*", text_value, re.DOTALL)
    )
    if reminder_matches:
        trailing_text = text_value[reminder_matches[-1].end():].strip()
        if trailing_text:
            text_value = trailing_text

    cap = _get_google_adapter_preserved_task_state_char_cap()
    if len(text_value) <= cap:
        return text_value
    return text_value[-cap:].lstrip()


def _build_google_adapter_preserved_task_state_message(
    messages: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    task_markers = (
        "Run this numbered script",
        "numbered script",
        "next and only valid tool call",
        "A final response immediately after Bash is invalid",
        "After WebFetch",
    )
    fallback: tuple[int, str] | None = None
    selected: tuple[int, str] | None = None

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") == "tool":
            continue
        if _completion_message_has_tool_result(message):
            continue
        if _is_google_adapter_synthetic_tool_context_message(message):
            continue
        text = _extract_completion_message_text(message).strip()
        if not text:
            continue
        if fallback is None:
            fallback = (index, text)
        if any(marker in text for marker in task_markers):
            selected = (index, text)
            break

    if selected is None:
        selected = fallback
    if selected is None:
        return None, {}

    source_index, source_text = selected
    excerpt = _extract_google_adapter_preserved_task_excerpt(source_text)
    if not excerpt:
        return None, {}

    preserved_text = (
        "<system-reminder>\n"
        "Gemini adapter preserved active child-agent task state from trimmed conversation. "
        "Continue to follow this original task and its next-tool obligations.\n\n"
        "Original task excerpt:\n"
        f"{excerpt}\n"
        "</system-reminder>"
    )
    return {
        "role": "user",
        "content": preserved_text,
    }, {
        "preserved_active_task_state": True,
        "preserved_active_task_state_chars": len(excerpt),
        "preserved_active_task_state_source_index": source_index,
    }


def _apply_google_adapter_completion_message_window(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(messages) <= 2:
        return messages, {}
    max_window = _get_google_adapter_max_completion_messages_window()
    original_count = len(messages)
    original_text_chars = sum(_estimate_completion_message_text_chars(message) for message in messages)
    trimmed_messages = list(messages[-max_window:])
    preserved_task_message: Optional[dict[str, Any]] = None
    preserved_task_changes: dict[str, Any] = {}
    if (
        max_window >= 3
        and any(_completion_message_has_tool_result(message) for message in messages)
    ):
        first_retained_index = max(0, original_count - max_window)
        initial_text_index = next(
            (
                index
                for index, message in enumerate(messages)
                if _extract_completion_message_text(message).strip()
            ),
            None,
        )
        if initial_text_index is not None and initial_text_index < first_retained_index:
            preserved_task_message, preserved_task_changes = (
                _build_google_adapter_preserved_task_state_message(messages)
            )
            if preserved_task_message is not None:
                retained_tail, tail_boundary_changes = (
                    _trim_completion_message_tail_preserving_tool_pairs(
                        messages, max_window - 1
                    )
                )
                trimmed_messages = [
                    preserved_task_message,
                    *retained_tail,
                ]
                preserved_task_changes = {
                    **preserved_task_changes,
                    **tail_boundary_changes,
                }
    trimmed_text_chars = sum(_estimate_completion_message_text_chars(message) for message in trimmed_messages)
    if len(trimmed_messages) == original_count:
        return messages, {}
    return trimmed_messages, {
        "trimmed_completion_messages_from_count": original_count,
        "trimmed_completion_messages_to_count": len(trimmed_messages),
        "trimmed_completion_messages_from_text_chars": original_text_chars,
        "trimmed_completion_messages_to_text_chars": trimmed_text_chars,
        "trimmed_completion_messages_max_window": max_window,
        **preserved_task_changes,
    }


def _normalize_google_code_assist_httpx_payload(value: Any) -> Any:
    key_mapping = {
        "function_call": "functionCall",
        "function_response": "functionResponse",
        "inline_data": "inlineData",
        "file_data": "fileData",
        "mime_type": "mimeType",
        "file_uri": "fileUri",
        "media_resolution": "mediaResolution",
        "function_declarations": "functionDeclarations",
        "allowed_function_names": "allowedFunctionNames",
    }
    if isinstance(value, list):
        return [_normalize_google_code_assist_httpx_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = key_mapping.get(key, key) if isinstance(key, str) else str(key)
        normalized[normalized_key] = _normalize_google_code_assist_httpx_payload(item)
    return normalized


def _google_code_assist_duplicate_tool_results_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    duplicate_tool_results: list[tuple[str, str]] = []
    pending_tool_calls_by_id: dict[str, str] = {}
    duplicate_tool_call_names: set[str] = set()

    for message in completion_messages:
        role = message.get("role")
        if role == "assistant":
            pending_tool_calls_by_id.clear()
            duplicate_tool_call_names.clear()
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            tool_call_name_counts: dict[str, int] = {}
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not isinstance(function, dict):
                    continue
                function_name = function.get("name")
                if not isinstance(function_name, str) or not function_name:
                    continue
                pending_tool_calls_by_id[tool_call_id] = function_name
                tool_call_name_counts[function_name] = (
                    tool_call_name_counts.get(function_name, 0) + 1
                )
            duplicate_tool_call_names = {
                name for name, count in tool_call_name_counts.items() if count > 1
            }
            continue

        if role != "tool":
            pending_tool_calls_by_id.clear()
            duplicate_tool_call_names.clear()
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            continue
        function_name = pending_tool_calls_by_id.get(tool_call_id)
        if function_name in duplicate_tool_call_names:
            duplicate_tool_results.append((function_name, tool_call_id))

    return duplicate_tool_results


def _annotate_google_code_assist_duplicate_tool_response_parts(
    contents: list[Any],
    duplicate_tool_results: list[tuple[str, str]],
    *,
    annotate_function_response_id: bool = False,
) -> int:
    annotated_count = 0
    pending_index = 0
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if pending_index >= len(duplicate_tool_results):
                break
            if not isinstance(part, dict):
                continue
            function_response = part.get("functionResponse")
            if not isinstance(function_response, dict):
                continue
            function_name, tool_call_id = duplicate_tool_results[pending_index]
            if function_response.get("name") != function_name:
                continue
            response_payload = function_response.get("response")
            if not isinstance(response_payload, dict):
                response_payload = {}
                function_response["response"] = response_payload
            if annotate_function_response_id:
                function_response.setdefault("id", tool_call_id)
            response_payload.setdefault("tool_use_id", tool_call_id)
            annotated_count += 1
            pending_index += 1
    return annotated_count


def _annotate_google_code_assist_duplicate_tool_responses(
    google_request_dict: dict[str, Any],
    completion_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve Claude tool_use ids when Gemini has same-name parallel tool results."""
    duplicate_tool_results = (
        _google_code_assist_duplicate_tool_results_from_completion_messages(
            completion_messages
        )
    )
    if not duplicate_tool_results:
        return {}

    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}

    annotated_count = _annotate_google_code_assist_duplicate_tool_response_parts(
        contents,
        duplicate_tool_results,
    )
    if annotated_count == 0:
        return {}
    return {
        "google_adapter_annotated_duplicate_tool_response_count": annotated_count,
    }


def _google_code_assist_tool_results_from_completion_messages(
    completion_messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    tool_results: list[tuple[str, str]] = []
    pending_tool_calls_by_id: dict[str, str] = {}

    for message in completion_messages:
        role = message.get("role")
        if role == "assistant":
            pending_tool_calls_by_id.clear()
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not isinstance(function, dict):
                    continue
                function_name = function.get("name")
                if isinstance(function_name, str) and function_name:
                    pending_tool_calls_by_id[tool_call_id] = function_name
            continue

        if role != "tool":
            pending_tool_calls_by_id.clear()
            continue

        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            continue
        function_name = pending_tool_calls_by_id.get(tool_call_id)
        if isinstance(function_name, str) and function_name:
            tool_results.append((function_name, tool_call_id))

    return tool_results


def _annotate_google_code_assist_claude_tool_response_ids(
    google_request_dict: dict[str, Any],
    completion_messages: list[dict[str, Any]],
    *,
    google_model: str,
) -> dict[str, Any]:
    if "claude" not in google_model.lower():
        return {}

    tool_results = _google_code_assist_tool_results_from_completion_messages(
        completion_messages
    )
    if not tool_results:
        return {}

    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}

    annotated_count = _annotate_google_code_assist_duplicate_tool_response_parts(
        contents,
        tool_results,
        annotate_function_response_id=True,
    )
    if annotated_count == 0:
        return {}
    return {
        "google_adapter_annotated_claude_tool_response_id_count": annotated_count,
    }


def _google_code_assist_function_response_id(
    function_response: dict[str, Any],
) -> Optional[str]:
    response_payload = function_response.get("response")
    response_tool_use_id = (
        response_payload.get("tool_use_id")
        if isinstance(response_payload, dict)
        else None
    )
    for candidate in (function_response.get("id"), response_tool_use_id):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _google_code_assist_function_call_args_for_id(tool_call_id: str) -> dict[str, Any]:
    cached_arguments = _lookup_codex_google_code_assist_tool_call_arguments(tool_call_id)
    if not isinstance(cached_arguments, str) or not cached_arguments.strip():
        return {}
    try:
        parsed_arguments = json.loads(cached_arguments)
    except Exception:
        return {}
    return parsed_arguments if isinstance(parsed_arguments, dict) else {}


def _insert_google_code_assist_missing_claude_function_call_pairs(
    google_request_dict: dict[str, Any],
    *,
    google_model: str,
) -> dict[str, Any]:
    if "claude" not in google_model.lower():
        return {}

    contents = google_request_dict.get("contents")
    if not isinstance(contents, list):
        return {}

    updated_contents: list[Any] = []
    seen_function_call_ids: set[str] = set()
    inserted_count = 0

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            updated_contents.append(content)
            continue

        missing_function_call_parts: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                function_call_id = function_call.get("id")
                if isinstance(function_call_id, str) and function_call_id.strip():
                    seen_function_call_ids.add(function_call_id.strip())
                continue

            function_response = part.get("functionResponse")
            if not isinstance(function_response, dict):
                continue
            tool_call_id = _google_code_assist_function_response_id(function_response)
            function_name = function_response.get("name")
            if (
                not isinstance(tool_call_id, str)
                or not isinstance(function_name, str)
                or not function_name
                or tool_call_id in seen_function_call_ids
            ):
                continue
            missing_function_call_parts.append(
                {
                    "functionCall": {
                        "name": function_name,
                        "args": _google_code_assist_function_call_args_for_id(
                            tool_call_id
                        ),
                        "id": tool_call_id,
                    }
                }
            )
            seen_function_call_ids.add(tool_call_id)
            inserted_count += 1

        if missing_function_call_parts:
            updated_contents.append(
                {"role": "model", "parts": missing_function_call_parts}
            )
        updated_contents.append(content)

    if inserted_count == 0:
        return {}

    google_request_dict["contents"] = updated_contents
    return {
        "google_adapter_inserted_claude_function_call_pair_count": inserted_count
    }


def _extract_google_code_assist_text_metrics(content_block: Any) -> tuple[int, int]:
    part_count = 0
    char_count = 0
    if not isinstance(content_block, dict):
        return part_count, char_count
    parts = content_block.get("parts")
    if not isinstance(parts, list):
        return part_count, char_count
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_count += 1
        text = part.get("text")
        if isinstance(text, str):
            char_count += len(text)
    return part_count, char_count


def _summarize_google_code_assist_content_preview_entry(
    content_entry: dict[str, Any],
) -> dict[str, Any]:
    role = content_entry.get("role")
    parts = content_entry.get("parts")
    part_kinds = []
    text_preview = None
    preview_parts, preview_chars = _extract_google_code_assist_text_metrics(content_entry)
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            keys = [
                key
                for key in ("text", "functionCall", "functionResponse", "thought")
                if key in part
            ]
            if keys:
                part_kinds.extend(keys)
            text_value = part.get("text")
            if text_preview is None and isinstance(text_value, str):
                text_preview = text_value[:120].replace("\n", "\\n")
            function_response = part.get("functionResponse")
            if isinstance(function_response, dict):
                response_payload = function_response.get("response")
                if isinstance(response_payload, dict):
                    response_keys = sorted(response_payload.keys())
                    part_kinds.append(
                        f"functionResponseKeys:{','.join(response_keys)}"
                    )
                    content_value = response_payload.get("content")
                    if text_preview is None and isinstance(content_value, str):
                        text_preview = content_value[:120].replace("\n", "\\n")
    return {
        "role": role,
        "part_count": preview_parts,
        "text_chars": preview_chars,
        "part_kinds": part_kinds,
        "text_preview": text_preview,
    }


def _summarize_google_code_assist_request_contents_shape(
    request_block: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    contents = request_block.get("contents")
    if not isinstance(contents, list):
        return

    summary["contents_count"] = len(contents)
    content_part_count = 0
    content_text_chars = 0
    text_entry_count = 0
    preview_entries = []
    for content_entry in contents:
        parts, chars = _extract_google_code_assist_text_metrics(content_entry)
        content_part_count += parts
        content_text_chars += chars
        if chars > 0:
            text_entry_count += 1
    for content_entry in contents[-4:]:
        if isinstance(content_entry, dict):
            preview_entries.append(
                _summarize_google_code_assist_content_preview_entry(content_entry)
            )
    summary["contents_part_count"] = content_part_count
    summary["contents_text_chars"] = content_text_chars
    summary["contents_text_entry_count"] = text_entry_count
    summary["contents_tail_preview"] = preview_entries


def _summarize_google_code_assist_generation_config_shape(
    request_block: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    generation_config = request_block.get("generationConfig")
    if not isinstance(generation_config, dict):
        return

    summary["generation_config_keys"] = sorted(generation_config.keys())
    generation_config_summary = {}
    for key in ("max_output_tokens", "temperature", "top_p", "candidate_count"):
        if key in generation_config:
            generation_config_summary[key] = generation_config.get(key)
    thinking_config = generation_config.get("thinkingConfig")
    if isinstance(thinking_config, dict):
        generation_config_summary["thinking_config_keys"] = sorted(thinking_config.keys())
        if "thinkingBudgetTokens" in thinking_config:
            generation_config_summary["thinking_budget_tokens"] = thinking_config.get("thinkingBudgetTokens")
    if generation_config_summary:
        summary["generation_config_values"] = generation_config_summary


def _extract_google_code_assist_function_names(request_block: Any) -> list[str]:
    request_tools = request_block.get("tools") if isinstance(request_block, dict) else None
    function_names: list[str] = []
    if not isinstance(request_tools, list):
        return function_names

    for tool_entry in request_tools:
        if not isinstance(tool_entry, dict):
            continue
        decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
        if not isinstance(decls, list):
            continue
        for declaration in decls:
            if not isinstance(declaration, dict):
                continue
            name = declaration.get("name")
            if isinstance(name, str):
                function_names.append(name)
    return function_names


def _summarize_google_code_assist_request_shape(payload: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not isinstance(payload, dict):
        summary["payload_type"] = type(payload).__name__
        return summary

    summary["top_level_keys"] = sorted(payload.keys())
    for key in ("model", "project", "user_prompt_id", "session_id"):
        if key in payload:
            summary[key] = payload.get(key)

    request_block = payload.get("request") if isinstance(payload.get("request"), dict) else payload
    if isinstance(request_block, dict):
        summary["request_keys"] = sorted(request_block.keys())
        _summarize_google_code_assist_request_contents_shape(request_block, summary)
        tools = request_block.get("tools")
        if isinstance(tools, list):
            summary["tools_count"] = len(tools)
            function_declaration_count = 0
            for tool_entry in tools:
                if isinstance(tool_entry, dict):
                    decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
                    if isinstance(decls, list):
                        function_declaration_count += len(decls)
            summary["function_declaration_count"] = function_declaration_count
        session_id = request_block.get("session_id")
        if isinstance(session_id, str) and session_id:
            summary["session_id_hash"] = hashlib.sha1(session_id.encode('utf-8')).hexdigest()[:8]
        _summarize_google_code_assist_generation_config_shape(request_block, summary)
        tool_config = request_block.get("toolConfig")
        if isinstance(tool_config, dict):
            summary["tool_config_keys"] = sorted(tool_config.keys())
        system_instruction = request_block.get("systemInstruction")
        if isinstance(system_instruction, dict):
            summary["has_system_instruction"] = True
            system_parts, system_chars = _extract_google_code_assist_text_metrics(
                system_instruction
            )
            summary["system_instruction_part_count"] = system_parts
            summary["system_instruction_text_chars"] = system_chars
    return summary


def _unwrap_google_code_assist_response_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    response_payload = payload.get("response")
    if not isinstance(response_payload, dict):
        return None
    unwrapped = dict(response_payload)
    trace_id = payload.get("traceId")
    if isinstance(trace_id, str) and trace_id and "responseId" not in unwrapped:
        unwrapped["responseId"] = trace_id
    return unwrapped


async def _translate_google_code_assist_response_to_anthropic(
    *,
    response: Response,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    completion_messages: list[dict[str, Any]],
    gemini_optional_params: dict[str, Any],
    litellm_params: dict[str, Any],
    logging_obj: Any,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.main import _get_encoding
    from litellm.utils import ModelResponse

    try:
        outer_payload = json.loads(_decode_http_response_body(response.body))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Google Code Assist adapter returned invalid JSON: {exc}",
        ) from exc

    unwrapped_payload = _unwrap_google_code_assist_response_payload(outer_payload)
    if unwrapped_payload is None:
        raise HTTPException(
            status_code=502,
            detail="Google Code Assist adapter response did not contain a `response` payload.",
        )

    raw_response = httpx.Response(
        status_code=response.status_code,
        headers=dict(response.headers),
        content=json.dumps(unwrapped_payload).encode("utf-8"),
    )
    model_response = VertexGeminiConfig().transform_response(
        model=_normalize_google_completion_adapter_model_name(adapter_model),
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data=unwrapped_payload,
        messages=cast(list[AllMessageValues], completion_messages),
        optional_params=gemini_optional_params,
        litellm_params=litellm_params,
        encoding=_get_encoding(),
        api_key="",
    )
    anthropic_response = AnthropicAdapter().translate_completion_output_params(
        model_response,
        tool_name_mapping=tool_name_mapping,
    )
    return _build_anthropic_response_from_completion_adapter_response(
        anthropic_response
    )


async def _iterate_google_code_assist_unwrapped_stream(
    body_iterator: Any,
    *,
    adapter_model: Optional[str] = None,
    rate_limit_key: Optional[str] = None,
) -> Any:
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

    debug_logged = False
    post_tool_cooldown_armed = False

    async def _iter_event_block_lines(event_block: str):
        nonlocal debug_logged, post_tool_cooldown_armed
        for line in event_block.splitlines():
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
            if not isinstance(parsed_chunk, dict):
                continue
            unwrapped = _unwrap_google_code_assist_response_payload(parsed_chunk)
            if unwrapped is None:
                continue
            if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1" and not debug_logged:
                try:
                    first_candidate = None
                    candidates = unwrapped.get("candidates") if isinstance(unwrapped, dict) else None
                    if isinstance(candidates, list) and candidates:
                        first_candidate = candidates[0]
                    verbose_proxy_logger.info(
                        "Gemini adapter stream debug: first_unwrapped_keys=%s first_candidate=%s",
                        sorted(unwrapped.keys()) if isinstance(unwrapped, dict) else type(unwrapped).__name__,
                        first_candidate,
                    )
                    debug_logged = True
                except Exception:
                    verbose_proxy_logger.exception("Gemini adapter stream debug logging failed")
            if (
                not post_tool_cooldown_armed
                and _google_code_assist_unwrapped_chunk_contains_tool_call(unwrapped)
            ):
                cooldown_seconds = _get_google_adapter_post_tool_cooldown_seconds()
                if cooldown_seconds > 0:
                    await _set_google_adapter_cooldown(
                        _clean_codex_auth_value(rate_limit_key)
                        or _get_google_adapter_rate_limit_key(adapter_model),
                        cooldown_seconds,
                    )
                    post_tool_cooldown_armed = True
                    verbose_proxy_logger.debug(
                        "Google adapter post-tool cooldown armed for %.1fs",
                        cooldown_seconds,
                    )
            yield f"data: {json.dumps(unwrapped)}\n\n"

    buffer = ""
    async for raw_chunk in body_iterator:
        if isinstance(raw_chunk, bytes):
            buffer += raw_chunk.decode("utf-8")
        else:
            buffer += str(raw_chunk)

        while "\n\n" in buffer:
            event_block, buffer = buffer.split("\n\n", 1)
            async for emitted_chunk in _iter_event_block_lines(event_block):
                yield emitted_chunk

    if buffer.strip():
        async for emitted_chunk in _iter_event_block_lines(buffer):
            yield emitted_chunk


def _build_anthropic_streaming_response_from_google_code_assist_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    gemini_optional_params: dict[str, Any],
    rate_limit_key: Optional[str] = None,
) -> StreamingResponse:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    logging_obj: Any = SimpleNamespace(
        optional_params=gemini_optional_params,
        post_call=lambda **_: None,
    )
    completion_stream = ModelResponseIterator(
        streaming_response=_iterate_google_code_assist_unwrapped_stream(
            response.body_iterator,
            adapter_model=adapter_model,
            rate_limit_key=rate_limit_key,
        ),
        sync_stream=False,
        logging_obj=logging_obj,
    )
    anthropic_stream = AnthropicAdapter().translate_completion_output_params_streaming(
        completion_stream,
        model=_normalize_google_completion_adapter_model_name(adapter_model),
        tool_name_mapping=tool_name_mapping,
    )
    return _build_anthropic_streaming_response_from_completion_adapter_stream(
        anthropic_stream
    )


def _restore_google_adapter_tool_call_names(
    response_obj: Any,
    tool_name_mapping: dict[str, str],
) -> Any:
    choices = getattr(response_obj, "choices", None)
    if not isinstance(choices, list):
        return response_obj
    for choice in choices:
        for message_attr in ("message", "delta"):
            message = getattr(choice, message_attr, None)
            if message is None:
                continue
            tool_calls = getattr(message, "tool_calls", None)
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                function = (
                    tool_call.get("function")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "function", None)
                )
                if function is None:
                    continue
                current_name = (
                    function.get("name")
                    if isinstance(function, dict)
                    else getattr(function, "name", None)
                )
                function_arguments = (
                    function.get("arguments")
                    if isinstance(function, dict)
                    else getattr(function, "arguments", None)
                )
                original_name = tool_name_mapping.get(str(current_name or ""))
                final_name = original_name or current_name
                tool_call_id = (
                    tool_call.get("id")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "id", None)
                )
                _remember_codex_google_code_assist_tool_call_name(
                    tool_call_id,
                    final_name,
                    function_arguments,
                )
                if not original_name:
                    continue
                if isinstance(function, dict):
                    function["name"] = original_name
                else:
                    setattr(function, "name", original_name)
    return response_obj


async def _restore_google_adapter_tool_call_names_stream(
    completion_stream: Any,
    tool_name_mapping: dict[str, str],
) -> Any:
    async for chunk in completion_stream:
        yield _restore_google_adapter_tool_call_names(chunk, tool_name_mapping)


async def _collect_google_code_assist_model_response_from_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    logging_obj: Any,
) -> Any:
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
        GeminiPassthroughLoggingHandler,
    )

    all_chunks: list[str] = []
    body_iterator = response.body_iterator
    try:
        async for raw_chunk in body_iterator:
            if isinstance(raw_chunk, bytes):
                all_chunks.append(raw_chunk.decode("utf-8", errors="replace"))
            else:
                all_chunks.append(str(raw_chunk))
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if callable(aclose):
            await aclose()

    model_response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=all_chunks,
        litellm_logging_obj=logging_obj,
        model=_normalize_google_completion_adapter_model_name(adapter_model),
        url_route="/v1internal:streamGenerateContent",
    )
    if model_response is None:
        raise HTTPException(
            status_code=502,
            detail="Google Code Assist streaming adapter could not build a complete response.",
        )
    return model_response


async def _collect_google_code_assist_response_from_stream(
    *,
    response: StreamingResponse,
    adapter_model: str,
    tool_name_mapping: dict[str, str],
    logging_obj: Any,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        AnthropicAdapter,
    )

    model_response = await _collect_google_code_assist_model_response_from_stream(
        response=response,
        adapter_model=adapter_model,
        logging_obj=logging_obj,
    )

    anthropic_response = AnthropicAdapter().translate_completion_output_params(
        model_response,
        tool_name_mapping=tool_name_mapping,
    )
    return _build_anthropic_response_from_completion_adapter_response(
        anthropic_response
    )


def _serialize_responses_adapter_response(response_obj: Any) -> str:
    if hasattr(response_obj, "model_dump_json"):
        return response_obj.model_dump_json(exclude_none=True)
    if hasattr(response_obj, "json"):
        return response_obj.json(exclude_none=True)
    return json.dumps(response_obj)


def _build_responses_response_from_adapter_response(response_obj: Any) -> Response:
    return Response(
        content=_serialize_responses_adapter_response(response_obj),
        media_type="application/json",
    )


async def _responses_sse_from_iterator(responses_iterator: Any) -> Any:
    async for event in responses_iterator:
        event_type = getattr(event, "type", None)
        serialized = _serialize_responses_adapter_response(event)
        if isinstance(event_type, str) and event_type:
            yield f"event: {event_type}\ndata: {serialized}\n\n"
        else:
            yield f"data: {serialized}\n\n"
    yield "data: [DONE]\n\n"


def _build_codex_streaming_response_from_google_code_assist_stream(
    *,
    response: StreamingResponse,
    adapter_request: SimpleNamespace,
) -> StreamingResponse:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )

    logging_obj = Logging(
        model=adapter_request.google_model,
        messages=adapter_request.completion_messages,
        stream=True,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id=str(uuid4()),
        function_id="codex_google_code_assist_adapter",
    )
    logging_obj.optional_params = adapter_request.gemini_optional_params
    completion_stream = ModelResponseIterator(
        streaming_response=_iterate_google_code_assist_unwrapped_stream(
            response.body_iterator,
            adapter_model=adapter_request.google_model,
            rate_limit_key=adapter_request.google_adapter_rate_limit_key,
        ),
        sync_stream=False,
        logging_obj=logging_obj,
    )
    completion_stream = _restore_google_adapter_tool_call_names_stream(
        completion_stream,
        adapter_request.tool_name_mapping,
    )
    streamwrapper = CustomStreamWrapper(
        completion_stream=completion_stream,
        model=adapter_request.google_model,
        custom_llm_provider=litellm.LlmProviders.GEMINI.value,
        logging_obj=logging_obj,
    )
    responses_iterator = LiteLLMCompletionStreamingIterator(
        model=adapter_request.google_model,
        litellm_custom_stream_wrapper=streamwrapper,
        request_input=adapter_request.codex_request_input,
        responses_api_request=adapter_request.responses_api_request,
        custom_llm_provider=litellm.LlmProviders.GEMINI.value,
        litellm_metadata=adapter_request.litellm_metadata,
    )
    return StreamingResponse(
        _responses_sse_from_iterator(responses_iterator),
        media_type="text/event-stream",
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
            verbose_proxy_logger.exception(
                "Failed to release adapted streaming response guard callback"
            )

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


def _get_openrouter_api_key() -> Optional[str]:
    return _get_first_secret_value(_ANTHROPIC_ADAPTER_OPENROUTER_API_KEY_ENV_VARS)


def _get_anthropic_adapter_openrouter_api_key() -> Optional[str]:
    return _get_openrouter_api_key()


def _get_anthropic_adapter_nvidia_api_key() -> Optional[str]:
    return _get_first_secret_value(_ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS)


def _get_anthropic_adapter_nvidia_target_base() -> str:
    cleaned = (
        _clean_secret_string(os.getenv("NVIDIA_NIM_API_BASE"))
        or _clean_secret_string(os.getenv("AAWM_NVIDIA_API_BASE"))
        or "https://integrate.api.nvidia.com/v1"
    )
    cleaned = cleaned.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_nvidia_adapter_max_retries() -> int:
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_NVIDIA_ADAPTER_MAX_RETRIES"))
    if raw_value is None:
        return 1
    try:
        parsed = int(raw_value)
    except Exception:
        return 1
    return max(0, parsed)


def _get_nvidia_adapter_request_timeout_seconds(
    adapter_model: Optional[str] = None,
) -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_NVIDIA_ADAPTER_REQUEST_TIMEOUT_SECONDS")
    )
    if raw_value is None:
        if _should_force_fake_stream_for_nvidia_adapter_model(adapter_model):
            return 240.0
        return 120.0
    try:
        parsed = float(raw_value)
    except Exception:
        if _should_force_fake_stream_for_nvidia_adapter_model(adapter_model):
            return 240.0
        return 120.0
    return max(5.0, parsed)


def _get_nvidia_adapter_inner_max_retries() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_NVIDIA_ADAPTER_INNER_MAX_RETRIES")
    )
    if raw_value is None:
        return 0
    try:
        parsed = int(raw_value)
    except Exception:
        return 0
    return max(0, parsed)


def _should_force_fake_stream_for_nvidia_adapter_model(
    adapter_model: Optional[str],
) -> bool:
    configured_models = _clean_codex_auth_value(
        os.getenv("AAWM_NVIDIA_ADAPTER_FORCE_FAKE_STREAM_MODELS")
    )
    if configured_models is None:
        normalized_models = {"minimaxai/minimax-m2.7"}
    else:
        normalized_models = {
            item.strip() for item in configured_models.split(",") if item.strip()
        }
    return bool(adapter_model and adapter_model in normalized_models)


def _extract_nvidia_adapter_exception_status_code(exc: Any) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue

    text_value = str(exc)
    if "Timeout Error" in text_value or exc.__class__.__name__.lower() == "timeout":
        return 504

    match = re.search(r"\b(408|429|500|502|503|504)\b", text_value)
    if match is not None:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _get_nvidia_adapter_retry_wait_seconds(attempt: int) -> float:
    return min(float(2 ** max(0, attempt - 1)), 8.0)


async def _perform_nvidia_completion_adapter_operation(
    *,
    adapter_model: Optional[str],
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    max_retries = _get_nvidia_adapter_max_retries()
    total_attempts = max_retries + 1
    attempt = 0
    while True:
        attempt += 1
        verbose_proxy_logger.debug(
            "NVIDIA completion adapter upstream attempt %s/%s for model=%s",
            attempt,
            total_attempts,
            adapter_model,
        )
        try:
            return await operation()
        except Exception as exc:
            status_code = _extract_nvidia_adapter_exception_status_code(exc)
            raw_message = str(exc)
            if (
                status_code not in _ANTHROPIC_ADAPTER_NVIDIA_RETRYABLE_STATUS_CODES
                or attempt >= total_attempts
            ):
                verbose_proxy_logger.warning(
                    "NVIDIA completion adapter upstream attempt %s failed with %s (%s, raw=%s) and will not be retried",
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    raw_message,
                )
                raise HTTPException(
                    status_code=status_code or 502,
                    detail=raw_message,
                )
            wait_seconds = _get_nvidia_adapter_retry_wait_seconds(attempt)
            verbose_proxy_logger.warning(
                "NVIDIA completion adapter upstream attempt %s hit %s (%s, raw=%s); backoff %.1fs",
                attempt,
                status_code,
                exc.__class__.__name__,
                raw_message,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)


def _get_openrouter_target_base() -> str:
    cleaned = (
        _clean_secret_string(os.getenv("OPENROUTER_API_BASE"))
        or "https://openrouter.ai/api"
    ).rstrip("/")
    if cleaned.endswith("/api/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_anthropic_adapter_openrouter_target_base() -> str:
    return _get_openrouter_target_base()


def _get_opencode_zen_target_base() -> str:
    cleaned = (
        _clean_secret_string(get_secret_str("OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(get_secret_str("AAWM_OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(os.getenv("OPENCODE_ZEN_API_BASE"))
        or _clean_secret_string(os.getenv("AAWM_OPENCODE_ZEN_API_BASE"))
        or _OPENCODE_ZEN_DEFAULT_BASE_URL
    ).rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


def _get_opencode_zen_auth_file_path() -> Optional[Path]:
    for env_name in _OPENCODE_ZEN_AUTH_FILE_ENV_VARS:
        value = _clean_secret_string(os.getenv(env_name))
        if value:
            candidate = Path(value).expanduser()
            if candidate.is_file():
                return candidate

    for candidate_str in _OPENCODE_ZEN_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.is_file():
            return candidate
    return None


async def _load_local_opencode_zen_api_key() -> str:
    explicit_key = _get_first_secret_value(_OPENCODE_ZEN_API_KEY_ENV_VARS)
    if explicit_key is not None:
        return explicit_key

    auth_path = _get_opencode_zen_auth_file_path()
    if auth_path is None:
        raise FileNotFoundError(
            "OpenCode Zen auth file not found. Expected "
            "'~/.local/share/opencode/auth.json' or set 'LITELLM_OPENCODE_AUTH_FILE'."
        )

    try:
        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to read OpenCode Zen auth file at {auth_path}") from exc

    provider_auth = auth_data.get("opencode") if isinstance(auth_data, dict) else None
    api_key = (
        _clean_secret_string(provider_auth.get("key"))
        if isinstance(provider_auth, dict)
        else None
    )
    auth_type = (
        _clean_secret_string(provider_auth.get("type"))
        if isinstance(provider_auth, dict)
        else None
    )
    if api_key is None or auth_type not in {None, "api"}:
        raise ValueError(
            "OpenCode Zen auth file must contain provider 'opencode' with API-key auth."
        )
    return api_key


def _raise_opencode_zen_auto_agent_candidate_unavailable(exc: Exception) -> None:
    proxy_exc = ProxyException(
        message=(
            "OpenCode Zen auto-agent candidate requires a valid OpenCode API-key "
            f"credential: {exc}"
        ),
        type="rate_limit_error",
        param="model",
        code=429,
    )
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": proxy_exc.message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise proxy_exc from exc


def _opencode_zen_candidate_unavailable_detail(exc: Exception) -> Optional[str]:
    status_code = _extract_google_adapter_exception_status_code(exc)
    detail = _extract_google_adapter_exception_detail(exc)
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="ignore")
    else:
        detail_text = str(detail or exc)
    detail_text = " ".join(
        str(part)
        for part in (
            getattr(exc, "message", None),
            getattr(exc, "code", None),
            detail_text,
            str(exc),
        )
        if part is not None
    )
    normalized = detail_text.lower()
    if any(
        marker in normalized
        for marker in (
            "freeusagelimiterror",
            "free usage limit",
            "creditserror",
            "no payment method",
            "add a payment method",
            "billing",
            "payment required",
        )
    ):
        return detail_text
    if "not supported for format openai" in normalized:
        return detail_text
    if status_code in {401, 402, 403} and any(
        marker in normalized
        for marker in (
            "authentication",
            "authorization",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "api-key",
            "api key",
            "credential",
            "opencode",
        )
    ):
        return detail_text
    return None


def _antigravity_candidate_unavailable_detail(exc: Exception) -> Optional[str]:
    if not isinstance(exc, HTTPException):
        return None
    detail = getattr(exc, "detail", None)
    if isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    else:
        detail_text = str(detail or exc)
    normalized = detail_text.lower()
    if "agy cli" in normalized and "auth refresh" in normalized:
        return detail_text
    if "antigravity oauth" in normalized or "antigravity cli" in normalized:
        return detail_text
    if "antigravity" not in normalized:
        return None
    if not any(
        marker in normalized
        for marker in (
            "auth provider",
            "authentication",
            "authorization",
            "credential",
            "credentials",
            "log in",
            "login",
            "not logged in",
            "not logged into",
            "oauth",
            "token source",
        )
    ):
        return None
    return detail_text


def _raise_antigravity_auto_agent_candidate_unavailable(exc: Exception) -> None:
    detail = _antigravity_candidate_unavailable_detail(exc) or str(exc)
    proxy_exc = ProxyException(
        message=(
            "Antigravity auto-agent candidate requires a valid Antigravity OAuth "
            f"credential: {detail}"
        ),
        type="rate_limit_error",
        param="model",
        code=429,
    )
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": proxy_exc.message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise proxy_exc from exc


def _is_grok_unsupported_reasoning_parameter_detail(normalized_detail: str) -> bool:
    if "grok" not in normalized_detail:
        return False
    if not any(
        marker in normalized_detail
        for marker in (
            "reasoningeffort",
            "reasoning_effort",
            "output_config.effort",
            "reasoning",
        )
    ):
        return False
    return any(
        marker in normalized_detail
        for marker in (
            "does not support parameter",
            "unsupported parameter",
            "invalid-argument",
            "invalid argument",
        )
    )


def _grok_native_candidate_unavailable_detail(exc: Exception) -> Optional[str]:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    elif detail is not None:
        detail_text = str(detail)
    else:
        detail_text = str(exc)
    normalized = detail_text.lower()
    if _is_grok_unsupported_reasoning_parameter_detail(normalized):
        return detail_text
    if "could not decode the compaction blob" in normalized:
        return detail_text
    if (
        "xai oauth credential" not in normalized
        and "grok oidc credential" not in normalized
        and "grok native" not in normalized
    ):
        return None
    return detail_text


def _xai_oauth_candidate_unavailable_detail(exc: Exception) -> Optional[str]:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, (dict, list)):
        detail_text = json.dumps(detail, sort_keys=True, default=str)
    elif detail is not None:
        detail_text = str(detail)
    else:
        detail_text = str(exc)
    normalized = detail_text.lower()
    if _is_grok_unsupported_reasoning_parameter_detail(normalized):
        return detail_text
    if "could not decode the compaction blob" in normalized:
        return detail_text
    if not any(
        marker in normalized
        for marker in (
            "xai oauth credential",
            "xai oauth-managed",
            "managed xai oauth",
            "litellm_xai_oauth_auth_file",
        )
    ):
        return None
    return detail_text


def _raise_xai_oauth_auto_agent_candidate_unavailable(exc: Exception) -> None:
    detail = _xai_oauth_candidate_unavailable_detail(exc) or str(exc)
    proxy_exc = ProxyException(
        message=(
            "xAI OAuth auto-agent candidate requires a valid managed xAI OAuth "
            f"credential: {detail}"
        ),
        type="rate_limit_error",
        param="model",
        code=429,
    )
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": proxy_exc.message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise proxy_exc from exc


def _raise_grok_native_auto_agent_candidate_unavailable(exc: Exception) -> None:
    detail = _grok_native_candidate_unavailable_detail(exc) or str(exc)
    proxy_exc = ProxyException(
        message=(
            "Grok native auto-agent candidate requires a valid managed xAI/Grok "
            f"credential: {detail}"
        ),
        type="rate_limit_error",
        param="model",
        code=429,
    )
    setattr(
        proxy_exc,
        "detail",
        {
            "error": {
                "message": proxy_exc.message,
                "code": "aawm_codex_auto_agent_candidate_unavailable",
            }
        },
    )
    raise proxy_exc from exc


async def _load_opencode_zen_api_key_for_candidate(
    *,
    use_alias_candidate_probe: bool = False,
) -> str:
    try:
        return await _load_local_opencode_zen_api_key()
    except (FileNotFoundError, ValueError) as exc:
        if use_alias_candidate_probe:
            _raise_opencode_zen_auto_agent_candidate_unavailable(exc)
        raise


async def _build_opencode_zen_headers(
    request: Request,
    *,
    use_alias_candidate_probe: bool = False,
) -> dict[str, str]:
    api_key = await _load_opencode_zen_api_key_for_candidate(
        use_alias_candidate_probe=use_alias_candidate_probe,
    )
    return BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=api_key,
        request=request,
    )


def _add_opencode_zen_logging_metadata(
    request_body: dict[str, Any],
    *,
    route_family: str,
    tag_prefix: str,
    requested_model: Any,
    adapter_model: Optional[str] = None,
    input_shape: Optional[str] = None,
    output_shape: Optional[str] = None,
    client_name: Optional[str] = None,
) -> dict[str, Any]:
    extra_fields: dict[str, Any] = {
        "opencode_zen": True,
        "opencode_zen_requested_model": requested_model,
    }
    if client_name is not None:
        extra_fields["client_name"] = client_name
    if adapter_model is not None:
        extra_fields["opencode_zen_adapter_model"] = adapter_model
    if input_shape is not None:
        extra_fields["codex_adapter_input_shape"] = input_shape
    if output_shape is not None:
        extra_fields["codex_adapter_output_shape"] = output_shape

    tags = [tag_prefix, "opencode-zen"]
    if adapter_model is not None:
        tags.append(f"opencode-zen-model:{adapter_model}")

    return _merge_litellm_metadata(
        _add_route_family_logging_metadata(request_body, route_family),
        tags_to_add=tags,
        extra_fields=extra_fields,
    )


def _get_opencode_zen_responses_tool_name(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    name = _clean_secret_string(tool.get("name"))
    if name:
        return name
    function = tool.get("function")
    if isinstance(function, dict):
        return _clean_secret_string(function.get("name"))
    return None


def _ordered_unique_str_values(values: list[Optional[str]]) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _strip_opencode_zen_unsupported_responses_tools(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body

    supported_tools: list[Any] = []
    removed_tool_types: list[Optional[str]] = []
    removed_tool_names: list[Optional[str]] = []
    for tool in tools:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if tool_type == "function":
            supported_tools.append(tool)
            continue
        removed_tool_types.append(str(tool_type) if tool_type is not None else "unknown")
        removed_tool_names.append(_get_opencode_zen_responses_tool_name(tool))

    removed_count = len(tools) - len(supported_tools)
    if removed_count <= 0:
        return request_body

    updated_body = dict(request_body)
    if supported_tools:
        updated_body["tools"] = supported_tools
    else:
        updated_body.pop("tools", None)

    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=["opencode-zen-unsupported-tools-stripped"],
        extra_fields={
            "opencode_zen_removed_unsupported_tool_count": removed_count,
            "opencode_zen_removed_unsupported_tool_types": _ordered_unique_str_values(
                removed_tool_types
            ),
            "opencode_zen_removed_unsupported_tool_names": _ordered_unique_str_values(
                removed_tool_names
            ),
        },
    )


def _opencode_zen_chat_message_role(message: Any) -> Optional[str]:
    role = (
        message.get("role")
        if isinstance(message, dict)
        else getattr(message, "role", None)
    )
    return role if isinstance(role, str) else None


def _opencode_zen_chat_tool_call_id(tool_call: Any) -> Optional[str]:
    tool_call_id = (
        tool_call.get("id")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "id", None)
    )
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def _opencode_zen_chat_message_tool_call_ids(message: Any) -> list[str]:
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(tool_calls, list):
        return []

    tool_call_ids: list[str] = []
    for tool_call in tool_calls:
        tool_call_id = _opencode_zen_chat_tool_call_id(tool_call)
        if tool_call_id is not None:
            tool_call_ids.append(tool_call_id)
    return tool_call_ids


def _opencode_zen_chat_message_tool_result_id(message: Any) -> Optional[str]:
    tool_call_id = (
        message.get("tool_call_id")
        if isinstance(message, dict)
        else getattr(message, "tool_call_id", None)
    )
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def _collect_opencode_zen_following_tool_block(
    messages: list[Any],
    start_index: int,
) -> tuple[list[Any], list[Optional[str]], int]:
    tool_block: list[Any] = []
    tool_block_ids: list[Optional[str]] = []
    next_index = start_index
    while next_index < len(messages):
        next_message = messages[next_index]
        if _opencode_zen_chat_message_role(next_message) != "tool":
            break
        tool_block.append(next_message)
        tool_block_ids.append(_opencode_zen_chat_message_tool_result_id(next_message))
        next_index += 1
    return tool_block, tool_block_ids, next_index


def _sanitize_opencode_zen_completion_messages_for_chat_completion(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, {}

    updated_messages: list[Any] = []
    removed_assistant_count = 0
    removed_orphan_tool_count = 0
    removed_partial_tool_count = 0
    removed_extra_tool_count = 0

    index = 0
    while index < len(messages):
        message = messages[index]
        role = _opencode_zen_chat_message_role(message)

        if role == "tool":
            removed_orphan_tool_count += 1
            index += 1
            continue

        if role != "assistant":
            updated_messages.append(message)
            index += 1
            continue

        required_tool_call_ids = _opencode_zen_chat_message_tool_call_ids(message)
        if not required_tool_call_ids:
            updated_messages.append(message)
            index += 1
            continue

        required_tool_call_id_set = set(required_tool_call_ids)
        tool_block, tool_block_ids, next_index = (
            _collect_opencode_zen_following_tool_block(messages, index + 1)
        )

        present_tool_call_ids = {
            tool_call_id
            for tool_call_id in tool_block_ids
            if tool_call_id is not None
        }
        if not required_tool_call_id_set.issubset(present_tool_call_ids):
            removed_assistant_count += 1
            removed_partial_tool_count += len(tool_block)
            index = next_index
            continue

        updated_messages.append(message)
        retained_tool_call_ids: set[str] = set()
        for tool_message, tool_call_id in zip(tool_block, tool_block_ids):
            if (
                tool_call_id is None
                or tool_call_id not in required_tool_call_id_set
                or tool_call_id in retained_tool_call_ids
            ):
                removed_extra_tool_count += 1
                continue
            updated_messages.append(tool_message)
            retained_tool_call_ids.add(tool_call_id)
        index = next_index

    if (
        removed_assistant_count == 0
        and removed_orphan_tool_count == 0
        and removed_partial_tool_count == 0
        and removed_extra_tool_count == 0
    ):
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    changes: dict[str, Any] = {
        "opencode_zen_chat_tool_adjacency_sanitized": True,
        "opencode_zen_chat_tool_adjacency_removed_assistant_count": (
            removed_assistant_count
        ),
        "opencode_zen_chat_tool_adjacency_removed_orphan_tool_count": (
            removed_orphan_tool_count
        ),
        "opencode_zen_chat_tool_adjacency_removed_partial_tool_count": (
            removed_partial_tool_count
        ),
        "opencode_zen_chat_tool_adjacency_removed_extra_tool_count": (
            removed_extra_tool_count
        ),
        "opencode_zen_chat_tool_adjacency_messages_from_count": len(messages),
        "opencode_zen_chat_tool_adjacency_messages_to_count": len(
            updated_messages
        ),
    }
    return updated_kwargs, changes


def _openrouter_chat_message_function_call(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("function_call")
    return getattr(message, "function_call", None)


def _openrouter_chat_message_has_valid_content_or_tool_calls(message: Any) -> bool:
    role = _opencode_zen_chat_message_role(message)
    if role == "tool":
        return _opencode_zen_chat_message_tool_result_id(message) is not None

    if _opencode_zen_chat_message_tool_call_ids(message):
        return True
    if _openrouter_chat_message_function_call(message):
        return True

    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return not _is_codex_google_code_assist_empty_text_content(content)


def _sanitize_openrouter_completion_messages_for_chat_completion(
    completion_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_kwargs, adjacency_changes = (
        _sanitize_opencode_zen_completion_messages_for_chat_completion(
            completion_kwargs
        )
    )

    messages = completion_kwargs.get("messages")
    if not isinstance(messages, list):
        return completion_kwargs, adjacency_changes

    updated_messages: list[Any] = []
    removed_empty_message_count = 0
    for message in messages:
        if _openrouter_chat_message_has_valid_content_or_tool_calls(message):
            updated_messages.append(message)
            continue
        removed_empty_message_count += 1

    if removed_empty_message_count == 0 and not adjacency_changes:
        return completion_kwargs, {}

    updated_kwargs = dict(completion_kwargs)
    updated_kwargs["messages"] = updated_messages
    changes: dict[str, Any] = {
        "openrouter_chat_message_shape_sanitized": True,
        "openrouter_chat_message_shape_messages_from_count": len(messages),
        "openrouter_chat_message_shape_messages_to_count": len(updated_messages),
        "openrouter_chat_message_shape_removed_empty_message_count": (
            removed_empty_message_count
        ),
    }
    if adjacency_changes:
        changes.update(adjacency_changes)
        changes["openrouter_chat_tool_adjacency_sanitized"] = True
    return updated_kwargs, changes


def _apply_openrouter_completion_message_sanitization(
    *,
    request_body: dict[str, Any],
    completion_kwargs: dict[str, Any],
    litellm_metadata: dict[str, Any],
    span_name: str,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    completion_kwargs, sanitization_changes = (
        _sanitize_openrouter_completion_messages_for_chat_completion(
            completion_kwargs
        )
    )
    if not sanitization_changes:
        return request_body, completion_kwargs, litellm_metadata

    metadata_body = _merge_litellm_metadata(
        {"litellm_metadata": litellm_metadata},
        tags_to_add=[tag],
        extra_fields={
            **sanitization_changes,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata=sanitization_changes,
                )
            ],
        },
    )
    litellm_metadata = dict(metadata_body.get("litellm_metadata") or {})
    request_body = dict(request_body)
    request_body["litellm_metadata"] = litellm_metadata
    completion_kwargs = dict(completion_kwargs)
    completion_kwargs["metadata"] = litellm_metadata
    return request_body, completion_kwargs, litellm_metadata


def _opencode_zen_responses_sse_event(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _opencode_zen_response_payload_for_stream(
    *,
    response_id: str,
    model: str,
    status: str,
    output: Optional[list[dict[str, Any]]] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
    }
    if output is not None:
        payload["output"] = output
    if usage is not None:
        payload["usage"] = usage
    return payload


def _opencode_zen_message_item_for_stream(
    *,
    message_id: str,
    status: str,
    output_text: str = "",
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if status == "completed":
        content.append(
            {
                "type": "output_text",
                "text": output_text,
                "annotations": [],
            }
        )
    return {
        "id": message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def _opencode_zen_completed_response_for_stream(
    *,
    response_event: dict[str, Any],
    response_id: str,
    model: str,
    message_id: Optional[str],
    output_text: str,
) -> dict[str, Any]:
    response_payload = response_event.get("response")
    response_dict = dict(response_payload) if isinstance(response_payload, dict) else {}
    response_dict.setdefault("id", response_id)
    response_dict.setdefault("object", "response")
    response_dict.setdefault("created_at", int(time.time()))
    response_dict.setdefault("status", "completed")
    response_dict.setdefault("model", model)
    output = response_dict.get("output")
    if (
        message_id is not None
        and isinstance(output_text, str)
        and output_text
        and not (
            isinstance(output, list)
            and any(_responses_output_item_has_meaningful_content(item) for item in output)
        )
    ):
        response_dict["output"] = [
            _opencode_zen_message_item_for_stream(
                message_id=message_id,
                status="completed",
                output_text=output_text,
            )
        ]
    return response_dict


async def _normalize_opencode_zen_responses_stream_for_codex(
    response: StreamingResponse,
    *,
    adapter_model: str,
) -> Any:
    response_id: Optional[str] = None
    message_id: Optional[str] = None
    response_created_sent = False
    message_started = False
    output_text_parts: list[str] = []

    async for event in _iterate_responses_sse_events(response.body_iterator):
        event_dict = _coerce_namespace_to_mapping(event)
        if not isinstance(event_dict, dict):
            continue
        event_type = event_dict.get("type")
        if not isinstance(event_type, str) or not event_type:
            continue

        if event_type == "response.output_text.delta":
            raw_response_payload = event_dict.get("response")
            response_payload = (
                raw_response_payload if isinstance(raw_response_payload, dict) else {}
            )
            response_id = (
                _clean_secret_string(event_dict.get("response_id"))
                or _clean_secret_string(event_dict.get("id"))
                or _clean_secret_string(response_payload.get("id"))
                or response_id
                or f"resp_{uuid4().hex}"
            )
            response_model = (
                _clean_secret_string(response_payload.get("model"))
                or _clean_secret_string(event_dict.get("model"))
                or adapter_model
            )
            if not response_created_sent:
                yield _opencode_zen_responses_sse_event(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": _opencode_zen_response_payload_for_stream(
                            response_id=response_id,
                            model=response_model,
                            status="in_progress",
                            output=[],
                        ),
                    },
                )
                response_created_sent = True

            message_id = (
                _clean_secret_string(event_dict.get("item_id"))
                or message_id
                or f"msg_{uuid4().hex[:24]}"
            )
            if not message_started:
                yield _opencode_zen_responses_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": _opencode_zen_message_item_for_stream(
                            message_id=message_id,
                            status="in_progress",
                        ),
                    },
                )
                yield _opencode_zen_responses_sse_event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                )
                message_started = True

            delta = event_dict.get("delta")
            if isinstance(delta, str):
                output_text_parts.append(delta)
            event_dict["item_id"] = message_id
            event_dict.setdefault("output_index", 0)
            event_dict.setdefault("content_index", 0)
            yield _opencode_zen_responses_sse_event(event_type, event_dict)
            continue

        if event_type == "response.completed":
            raw_response_payload = event_dict.get("response")
            response_payload = (
                raw_response_payload if isinstance(raw_response_payload, dict) else {}
            )
            response_id = (
                _clean_secret_string(response_payload.get("id"))
                or _clean_secret_string(event_dict.get("id"))
                or response_id
                or f"resp_{uuid4().hex}"
            )
            response_model = (
                _clean_secret_string(response_payload.get("model"))
                or _clean_secret_string(event_dict.get("model"))
                or adapter_model
            )
            if not response_created_sent:
                yield _opencode_zen_responses_sse_event(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": _opencode_zen_response_payload_for_stream(
                            response_id=response_id,
                            model=response_model,
                            status="in_progress",
                            output=[],
                        ),
                    },
                )
                response_created_sent = True

            output_text = "".join(output_text_parts)
            if message_started and message_id is not None:
                yield _opencode_zen_responses_sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": output_text,
                    },
                )
                yield _opencode_zen_responses_sse_event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": output_text,
                            "annotations": [],
                        },
                    },
                )
                yield _opencode_zen_responses_sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": _opencode_zen_message_item_for_stream(
                            message_id=message_id,
                            status="completed",
                            output_text=output_text,
                        ),
                    },
                )

            event_dict["response"] = _opencode_zen_completed_response_for_stream(
                response_event=event_dict,
                response_id=response_id,
                model=response_model,
                message_id=message_id,
                output_text=output_text,
            )
            yield _opencode_zen_responses_sse_event(event_type, event_dict)
            continue

        yield _opencode_zen_responses_sse_event(event_type, event_dict)

    yield "data: [DONE]\n\n"


def _build_codex_opencode_zen_streaming_response(
    response: StreamingResponse,
    *,
    adapter_model: str,
) -> StreamingResponse:
    return StreamingResponse(
        _normalize_opencode_zen_responses_stream_for_codex(
            response,
            adapter_model=adapter_model,
        ),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type="text/event-stream",
    )


def _join_opencode_zen_passthrough_url(base_target_url: str, endpoint: str) -> str:
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint=endpoint,
        base_target_url=base_target_url,
    )
    return str(
        BaseOpenAIPassThroughHandler._join_url_paths(
            httpx.URL(base_target_url),
            normalized_endpoint,
            _OPENCODE_ZEN_PROVIDER,
        )
    )


def _build_openrouter_default_headers() -> dict[str, str]:
    headers = {
        "HTTP-Referer": _clean_secret_string(get_secret_str("OR_SITE_URL")) or "https://litellm.ai",
        "X-Title": _clean_secret_string(get_secret_str("OR_APP_NAME")) or "liteLLM",
    }
    return headers


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

    model_value = match.group("model").strip().strip('\"').strip("'")
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


def _anthropic_adapter_request_has_openai_client_auth(request: Request) -> bool:
    # On the Anthropic route, direct Authorization headers are typically Anthropic auth
    # from Claude clients, not OpenAI/Codex credentials. Treat direct auth as OpenAI
    # client auth only when the request also carries Codex-native request markers.
    if (
        _get_request_header_or_passthrough_alias(request, "x-pass-authorization")
        or _get_request_header_or_passthrough_alias(request, "x-pass-api-key")
    ):
        return True

    if _anthropic_adapter_request_uses_codex_native_auth(request):
        return bool(
            _get_request_header_or_passthrough_alias(request, "authorization")
            or _get_request_header_or_passthrough_alias(request, "api-key")
        )

    return False


def _anthropic_adapter_request_uses_codex_native_auth(request: Request) -> bool:
    chatgpt_account_id = _get_request_header_or_passthrough_alias(
        request, "ChatGPT-Account-Id"
    )
    originator = _get_request_header_or_passthrough_alias(request, "originator")
    user_agent = _get_request_header_or_passthrough_alias(request, "user-agent")
    session_id = _get_request_header_or_passthrough_alias(request, "session_id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


def _anthropic_adapter_should_forward_direct_auth_headers(request: Request) -> bool:
    return _anthropic_adapter_request_has_openai_client_auth(request)


def _clean_codex_auth_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


CodexAuthData = dict[str, object]
CodexTokenData = dict[str, object]
OAuthJsonData = dict[str, object]
AntigravityOAuthTokenData = dict[str, object]
AntigravityPassthroughRequestBody = dict[str, object]
PassthroughLoggingMetadata = dict[str, object]


def _build_google_debug_header_summary(headers: dict[str, Any]) -> dict[str, Any]:
    interesting_keys = (
        "authorization",
        "user-agent",
        "x-goog-api-client",
        "x-client-info",
        "x-goog-user-project",
        "origin",
        "referer",
        "accept",
    )
    summary: dict[str, Any] = {}
    for key in interesting_keys:
        value = headers.get(key) or headers.get(key.title())
        if not isinstance(value, str) or not value:
            continue
        if key == "authorization":
            summary[key] = value[:12]
        else:
            summary[key] = value
    return summary


def _get_google_adapter_native_user_agent(model: Optional[str]) -> str:
    configured = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_NATIVE_USER_AGENT"))
    if configured:
        return configured
    model_name = model or "gemini-3-flash-preview"
    return f"GeminiCLI/0.38.2/{model_name} (linux; x64; terminal) google-api-nodejs-client/9.15.1"


def _get_google_adapter_native_api_client_header() -> str:
    configured = _clean_codex_auth_value(os.getenv("AAWM_GOOGLE_ADAPTER_NATIVE_X_GOOG_API_CLIENT"))
    if configured:
        return configured
    return "gl-node/24.13.1"


def _build_google_adapter_native_headers(*, access_token: str, model: Optional[str], accept: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": _get_google_adapter_native_user_agent(model),
        "x-goog-api-client": _get_google_adapter_native_api_client_header(),
        "Accept": accept,
    }


def _get_anthropic_adapter_codex_auth_file_path() -> Optional[Path]:
    for env_name in _ANTHROPIC_ADAPTER_CODEX_AUTH_FILE_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if path.exists():
            return path

    token_dir: Optional[Path] = None
    for env_name in _ANTHROPIC_ADAPTER_CODEX_TOKEN_DIR_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            token_dir = candidate
            break
    if token_dir is not None:
        candidate = token_dir / "auth.json"
        if candidate.exists():
            return candidate

    for candidate_str in _ANTHROPIC_ADAPTER_CODEX_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if candidate.exists():
            return candidate

    return None


def _decode_jwt_claims_without_validation(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
    except Exception:
        return {}


def _extract_codex_account_id_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    claims = _decode_jwt_claims_without_validation(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _get_codex_auth_token_data(auth_data: CodexAuthData) -> CodexTokenData:
    token_data = auth_data.get("tokens")
    if isinstance(token_data, dict):
        return dict(token_data)
    return auth_data


def _get_codex_auth_token_expiry(access_token: str) -> Optional[int]:
    claims = _decode_jwt_claims_without_validation(access_token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return int(exp)
    return None


def _codex_auth_access_token_is_valid(token_data: CodexTokenData) -> bool:
    access_token = _clean_codex_auth_value(token_data.get("access_token"))
    if access_token is None:
        return False
    expires_at = token_data.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = _get_codex_auth_token_expiry(access_token)
    if not isinstance(expires_at, (int, float)):
        return True
    return time.time() < float(expires_at) - 60


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


async def _load_codex_auth_data_from_path(auth_path: Path) -> Optional[CodexAuthData]:
    try:
        auth_data = json.loads(auth_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(auth_data, dict):
        return None
    return auth_data


async def _load_local_codex_auth_headers(request: Request) -> Optional[dict[str, str]]:
    auth_path = _get_anthropic_adapter_codex_auth_file_path()
    if auth_path is None:
        return None

    auth_data = await _load_codex_auth_data_from_path(auth_path)
    if auth_data is None:
        return None

    token_data = _get_codex_auth_token_data(auth_data)
    access_token = _clean_codex_auth_value(token_data.get("access_token"))
    if access_token is None:
        return None
    if not _codex_auth_access_token_is_valid(token_data):
        raise HTTPException(
            status_code=500,
            detail=(
                "Codex OAuth access token is expired or invalid. The "
                "provider-status sidecar owns Codex auth refresh; confirm the "
                "sidecar can write the configured auth file and refresh "
                f"{auth_path}."
            ),
        )

    account_id = _clean_codex_auth_value(token_data.get("account_id")) or _extract_codex_account_id_from_token(
        _clean_codex_auth_value(token_data.get("id_token")) or access_token
    )

    headers = _safe_get_request_headers(request)
    session_id = (
        _get_request_header_or_passthrough_alias(request, "session_id")
        or headers.get("x-claude-code-session-id")
        or headers.get("X-Claude-Code-Session-Id")
    )

    return get_chatgpt_default_headers(
        access_token=access_token,
        account_id=account_id,
        session_id=session_id,
    )


def _get_anthropic_adapter_openai_target_base(
    request: Request,
    *,
    prefer_chatgpt_codex_backend: bool = False,
) -> str:
    if prefer_chatgpt_codex_backend or _anthropic_adapter_request_uses_codex_native_auth(
        request
    ):
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
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
        LiteLLMAnthropicToResponsesAPIAdapter,
    )
    from litellm.types.llms.anthropic import AnthropicMessagesRequest

    adapter = LiteLLMAnthropicToResponsesAPIAdapter()
    minimal_adapter_instructions = "You are a helpful assistant."
    request_fields = {
        "model": adapter_model,
        "messages": request_body.get("messages") or [],
        "max_tokens": request_body.get("max_tokens"),
    }
    for field_name in (
        "context_management",
        "mcp_servers",
        "metadata",
        "output_config",
        "output_format",
        "stop_sequences",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_p",
    ):
        if field_name in request_body:
            request_fields[field_name] = request_body[field_name]

    anthropic_request = cast(
        AnthropicMessagesRequest,
        {k: v for k, v in request_fields.items() if v is not None},
    )
    translated_body = adapter.translate_request(
        anthropic_request,
        custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        use_codex_native_tools=use_chatgpt_codex_defaults,
    )
    _normalize_openai_function_tool_schemas(translated_body)

    if request_body.get("stream") is True:
        translated_body["stream"] = True

    if use_chatgpt_codex_defaults:
        translated_body.pop("user", None)
        existing_instructions = translated_body.get("instructions")
        if not isinstance(existing_instructions, str) or not existing_instructions.strip():
            translated_body["instructions"] = minimal_adapter_instructions
        translated_body.setdefault("store", False)
        translated_body["stream"] = True
        include = list(translated_body.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        translated_body["include"] = include
        for unsupported_field in ("max_output_tokens", "temperature", "top_p"):
            translated_body.pop(unsupported_field, None)

    from litellm.llms.anthropic.experimental_pass_through.adapters.observability import (
        derive_prompt_cache_key,
        normalize_reasoning_effort_for_provider,
        request_contains_cache_control,
    )

    normalized_effort = normalize_reasoning_effort_for_provider(
        thinking=request_body.get("thinking"),
        output_config=request_body.get("output_config"),
        reasoning_effort=request_body.get("reasoning_effort"),
        model=adapter_model,
        custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        native_provider="openai",
        native_field="reasoning.effort",
    )
    adapter_tags: list[str] = []
    adapter_extra_fields: dict[str, Any] = {}
    _add_codex_native_tool_alias_adapter_metadata(
        adapter_tags,
        adapter_extra_fields,
        enabled=use_chatgpt_codex_defaults,
    )
    if normalized_effort is not None:
        adapter_tags.extend(normalized_effort.tags())
        adapter_extra_fields.update(normalized_effort.metadata())

    cache_requested = request_contains_cache_control(request_body)
    if not cache_requested:
        translated_body.pop("prompt_cache_key", None)

    if cache_requested:
        prompt_cache_key = request_body.get("prompt_cache_key")
        if not isinstance(prompt_cache_key, str) or not prompt_cache_key.strip():
            prompt_cache_key = derive_prompt_cache_key(request_body)
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            if len(prompt_cache_key) > 64:
                prompt_cache_key = derive_prompt_cache_key(
                    {"prompt_cache_key": prompt_cache_key},
                    prefix="anthropic-cache-key",
                )
            translated_body["prompt_cache_key"] = prompt_cache_key
            adapter_extra_fields["openai_prompt_cache_key_present"] = True
            adapter_extra_fields["anthropic_adapter_cache_control_present"] = True

    existing_litellm_metadata = request_body.get("litellm_metadata")
    if isinstance(existing_litellm_metadata, dict):
        translated_body["litellm_metadata"] = {
            **existing_litellm_metadata,
            **dict(translated_body.get("litellm_metadata") or {}),
        }
    translated_body, _codex_tool_description_patch_events = (
        _apply_codex_tool_description_patches_to_request_body(translated_body)
    )

    span_metadata = {
        "requested_model": request_body.get("model"),
        "adapter_model": adapter_model,
        "stream": bool(request_body.get("stream")),
    }
    return _merge_litellm_metadata(
        _add_route_family_logging_metadata(
            translated_body,
            route_family,
        ),
        tags_to_add=[
            tag_prefix,
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint}",
            *adapter_tags,
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": request_body.get("model"),
            "anthropic_adapter_target_endpoint": target_endpoint,
            **adapter_extra_fields,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata=span_metadata,
                )
            ],
        },
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

    replacement = _OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS
    if existing_instructions == replacement:
        return request_body, {}

    updated_body = dict(request_body)
    updated_body["instructions"] = replacement
    original_hash = hashlib.sha256(
        existing_instructions.encode("utf-8", errors="replace")
    ).hexdigest()
    changes = {
        f"{metadata_prefix}_parallel_instruction_policy_applied": True,
        f"{metadata_prefix}_parallel_instruction_original_chars": len(
            existing_instructions
        ),
        f"{metadata_prefix}_parallel_instruction_rewritten_chars": len(replacement),
        f"{metadata_prefix}_parallel_instruction_original_hash": original_hash,
        f"{metadata_prefix}_parallel_instruction_tool_names": function_tool_names,
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
                        "rewritten_chars": len(replacement),
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
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
        LiteLLMAnthropicToResponsesAPIAdapter,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    if _is_failed_responses_body(response_body):
        _raise_responses_adapter_failed_response(
            response_body=response_body,
            adapter_model=failed_response_adapter_model
            or str(response_body.get("model") or "unknown-model"),
            adapter=failed_response_adapter,
            adapter_label=failed_response_adapter_label,
            retryable_alias_candidate=retryable_failed_response,
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


def _normalize_openai_function_tool_schemas(translated_body: dict[str, Any]) -> None:
    tools = translated_body.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        if "parameters" in tool:
            tool["parameters"] = _normalize_openai_function_tool_parameters(
                tool.get("parameters")
            )

        function_block = tool.get("function")
        if isinstance(function_block, dict):
            function_block["parameters"] = _normalize_openai_function_tool_parameters(
                function_block.get("parameters")
            )


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
    forced_tool_choice_changes = (
        _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
            request_body,
            translated_body,
        )
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


def _coerce_mapping_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _coerce_mapping_to_namespace(val) for key, val in value.items()}
        )
    if isinstance(value, list):
        return [_coerce_mapping_to_namespace(item) for item in value]
    return value


async def _iterate_responses_sse_events(
    body_iterator: Any,
) -> Any:
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator

    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    async for raw_chunk in body_iterator:
        if isinstance(raw_chunk, bytes):
            buffer += decoder.decode(raw_chunk)
        else:
            buffer += str(raw_chunk)

        while "\n\n" in buffer:
            event_block, buffer = buffer.split("\n\n", 1)
            for line in event_block.splitlines():
                parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
                if parsed_chunk is not None:
                    yield _coerce_mapping_to_namespace(parsed_chunk)

    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        for line in buffer.splitlines():
            parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(line)
            if parsed_chunk is not None:
                yield _coerce_mapping_to_namespace(parsed_chunk)


def _coerce_namespace_to_mapping(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {
            key: _coerce_namespace_to_mapping(val)
            for key, val in vars(value).items()
        }
    if isinstance(value, list):
        return [_coerce_namespace_to_mapping(item) for item in value]
    return value


def _responses_event_text_key(event: Any) -> str:
    item_id = getattr(event, "item_id", None) or (
        event.get("item_id") if isinstance(event, dict) else None
    )
    if isinstance(item_id, str) and item_id:
        return item_id
    output_index = getattr(event, "output_index", None) or (
        event.get("output_index") if isinstance(event, dict) else None
    )
    if isinstance(output_index, int):
        return f"output:{output_index}"
    return "output:0"


def _responses_stream_event_summary(event: Any) -> dict[str, Any]:
    event_type = getattr(event, "type", None) or (
        event.get("type") if isinstance(event, dict) else None
    )
    summary: dict[str, Any] = {"type": event_type}
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = getattr(event, "item", None) or (
            event.get("item") if isinstance(event, dict) else None
        )
        if item is not None:
            summary["item_type"] = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            summary["item_id"] = getattr(item, "id", None) or (
                item.get("id") if isinstance(item, dict) else None
            )
            summary["item_name"] = getattr(item, "name", None) or (
                item.get("name") if isinstance(item, dict) else None
            )
        return summary
    if event_type in {
        "response.output_text.delta",
        "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.reasoning_summary_text.delta",
    }:
        summary["item_id"] = getattr(event, "item_id", None) or (
            event.get("item_id") if isinstance(event, dict) else None
        )
        text = getattr(event, "delta", None) or (
            event.get("delta") if isinstance(event, dict) else None
        )
        if text is None:
            text = getattr(event, "arguments", None) or (
                event.get("arguments") if isinstance(event, dict) else None
            )
        if text is None:
            text = getattr(event, "text", None) or (
                event.get("text") if isinstance(event, dict) else None
            )
        if isinstance(text, str):
            summary["text_len"] = len(text)
            summary["text_preview"] = text[:200]
        return summary
    if event_type in {
        "response.completed",
        "response.failed",
        "response.incomplete",
    }:
        response_payload = getattr(event, "response", None) or (
            event.get("response") if isinstance(event, dict) else None
        )
        response_dict = _coerce_namespace_to_mapping(response_payload)
        if isinstance(response_dict, dict):
            output = response_dict.get("output") or []
            usage = response_dict.get("usage") or {}
            summary.update(
                {
                    "response_id": response_dict.get("id"),
                    "response_status": response_dict.get("status"),
                    "response_model": response_dict.get("model"),
                    "output_count": len(output) if isinstance(output, list) else 0,
                    "output_types": [
                        item.get("type")
                        for item in output[:20]
                        if isinstance(item, dict)
                    ]
                    if isinstance(output, list)
                    else [],
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0)
                        if isinstance(usage, dict)
                        else 0,
                        "output_tokens": usage.get("output_tokens", 0)
                        if isinstance(usage, dict)
                        else 0,
                    },
                }
            )
    return summary


def _responses_output_item_has_meaningful_content(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    if item_type in {"function_call", "mcp_call"}:
        return True
    if item_type == "message":
        content = item.get("content") or []
        if not isinstance(content, list):
            return False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
        return False
    if item_type == "reasoning":
        summary = item.get("summary") or []
        if not isinstance(summary, list):
            return False
        for part in summary:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
        return False
    return False


def _is_empty_success_responses_body(response_body: dict[str, Any]) -> bool:
    status = response_body.get("status")
    if status not in {None, "completed"}:
        return False
    output = response_body.get("output") or []
    if not isinstance(output, list):
        return False
    if any(_responses_output_item_has_meaningful_content(item) for item in output):
        return False
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return False
    return True


def _is_failed_responses_body(response_body: dict[str, Any]) -> bool:
    return (
        response_body.get("status") == "failed"
        or response_body.get("error") is not None
    )


async def _validate_alias_candidate_responses_stream_if_needed(
    response: Response,
    *,
    enabled: bool,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
) -> Response:
    if not enabled or not isinstance(response, StreamingResponse):
        return response
    return await _validate_codex_auto_agent_responses_payload(
        response,
        adapter_model=adapter_model,
        adapter=adapter,
        adapter_label=adapter_label,
    )


def _is_codex_auto_agent_empty_success_responses_body(
    response_body: dict[str, Any],
) -> bool:
    if not _is_empty_success_responses_body(response_body):
        return False
    usage = response_body.get("usage") or {}
    if not isinstance(usage, dict):
        return False
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        return False
    try:
        return int(output_tokens) <= 1
    except Exception:
        return False


def _mapping_or_attr_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _usage_has_no_more_than_one_output_token(usage: Any) -> bool:
    if usage is None:
        return True
    saw_output_field = False
    for field in ("completion_tokens", "output_tokens", "output"):
        token_count = _coerce_optional_int(_mapping_or_attr_get(usage, field))
        if token_count is None:
            continue
        saw_output_field = True
        if token_count > 1:
            return False
    if saw_output_field:
        return True
    total_tokens = _coerce_optional_int(_mapping_or_attr_get(usage, "total_tokens"))
    if total_tokens == 0:
        return True
    return False


def _model_response_usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    result: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "output_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def _is_codex_google_code_assist_empty_success_model_response(
    model_response: Any,
) -> bool:
    choices = _mapping_or_attr_get(model_response, "choices") or []
    if not isinstance(choices, list):
        return False
    if not choices:
        return _usage_has_no_more_than_one_output_token(
            _mapping_or_attr_get(model_response, "usage")
        )

    saw_message = False
    for choice in choices:
        message = _mapping_or_attr_get(choice, "message")
        if message is None:
            continue
        saw_message = True
        if not _is_codex_google_code_assist_empty_text_content(
            _mapping_or_attr_get(message, "content")
        ):
            return False
        if _mapping_or_attr_get(message, "tool_calls"):
            return False
        if _mapping_or_attr_get(message, "function_call"):
            return False

    if not saw_message:
        return _usage_has_no_more_than_one_output_token(
            _mapping_or_attr_get(model_response, "usage")
        )
    return _usage_has_no_more_than_one_output_token(
        _mapping_or_attr_get(model_response, "usage")
    )


def _raise_codex_auto_agent_empty_success_response(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str = "codex_auto_agent_openrouter_responses",
    adapter_label: str = "OpenRouter",
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_empty_success_responses_diagnostic(
        response_body=response_body,
        diagnostic_context={
            "adapter": adapter,
            "adapter_model": adapter_model,
            **(
                {"stream_events": stream_event_summaries}
                if stream_event_summaries is not None
                else {}
            ),
        },
    )
    exc = ProxyException(
        message=(
            f"Codex auto-agent {adapter_label} candidate returned an empty successful "
            "Responses payload."
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
                "code": "aawm_codex_auto_agent_empty_success",
                "status": "RATE_LIMIT_EXCEEDED",
                "type": "rate_limit_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _build_failed_responses_diagnostic(
    *,
    response_body: dict[str, Any],
    adapter: str,
    adapter_model: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    output = response_body.get("output") or []
    diagnostic: dict[str, Any] = {
        "adapter": adapter,
        "adapter_model": adapter_model,
        "response_id": response_body.get("id"),
        "status": response_body.get("status"),
        "model": response_body.get("model"),
        "error": response_body.get("error"),
        "incomplete_details": response_body.get("incomplete_details"),
        "output_count": len(output) if isinstance(output, list) else 0,
        "output_types": [
            item.get("type") for item in output[:20] if isinstance(item, dict)
        ]
        if isinstance(output, list)
        else [],
    }
    if stream_event_summaries is not None:
        diagnostic["stream_events"] = stream_event_summaries
    return diagnostic


def _raise_codex_auto_agent_failed_responses_payload(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    exc = ProxyException(
        message=(
            f"Auto-agent {adapter_label} candidate returned a failed Responses "
            "payload."
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
                "code": "aawm_auto_agent_failed_responses_payload",
                "status": "RESPONSES_STATUS_FAILED",
                "type": "rate_limit_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _raise_responses_adapter_failed_response(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    retryable_alias_candidate: bool = False,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    if retryable_alias_candidate:
        _raise_codex_auto_agent_failed_responses_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            stream_event_summaries=stream_event_summaries,
        )

    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": f"{adapter_label} Responses adapter returned a failed response.",
            "diagnostic": diagnostic,
        },
    )


async def _validate_codex_auto_agent_responses_payload(
    response: Response,
    *,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
) -> Response:
    if isinstance(response, StreamingResponse):
        buffered_chunks: list[Any] = []
        event_summaries: list[dict[str, Any]] = []

        async def _recording_iterator() -> Any:
            async for raw_chunk in response.body_iterator:
                buffered_chunks.append(raw_chunk)
                yield raw_chunk

        recording_response = StreamingResponse(
            _recording_iterator(),
            headers=dict(response.headers),
            status_code=response.status_code,
            media_type=response.media_type or "text/event-stream",
        )
        response_body = await _collect_responses_response_from_stream(
            recording_response,
            event_summaries=event_summaries,
        )
        if _is_failed_responses_body(response_body):
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                stream_event_summaries=event_summaries,
            )

        async def _replay_iterator() -> Any:
            for raw_chunk in buffered_chunks:
                yield raw_chunk

        return StreamingResponse(
            _replay_iterator(),
            headers=dict(response.headers),
            status_code=response.status_code,
            media_type=response.media_type or "text/event-stream",
        )

    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
        try:
            response_body = json.loads(_decode_http_response_body(response.body))
        except Exception:
            return response
        if isinstance(response_body, dict) and _is_failed_responses_body(response_body):
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
            )
    return response


def _responses_output_stream_key(
    *,
    item: Optional[dict[str, Any]] = None,
    output_index: Any = None,
    item_id: Any = None,
    fallback_index: Optional[int] = None,
) -> str:
    if isinstance(item, dict):
        for key in ("call_id", "id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(item_id, str) and item_id.strip():
        return item_id.strip()
    if isinstance(output_index, int):
        return f"output:{output_index}"
    if fallback_index is not None:
        return f"fallback:{fallback_index}"
    return "fallback:0"


def _merge_responses_output_lists(
    completed_output: Optional[list[dict[str, Any]]],
    streamed_output: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    merged_by_key: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for output_list in (streamed_output or [], completed_output or []):
        for item in output_list:
            if not isinstance(item, dict):
                continue
            key = _responses_output_stream_key(
                item=item,
                fallback_index=len(ordered_keys),
            )
            if key not in ordered_keys:
                ordered_keys.append(key)
            existing = merged_by_key.get(key, {})
            merged_item = {**existing, **item}
            if "arguments" in existing and "arguments" not in item:
                merged_item["arguments"] = existing["arguments"]
            merged_by_key[key] = merged_item

    return [merged_by_key[key] for key in ordered_keys if key in merged_by_key]


def _responses_output_has_message_text(output: Any) -> bool:
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") in {"output_text", "text"}
                and isinstance(part.get("text"), str)
                and part["text"]
            ):
                return True
    return False


def _build_collected_responses_text_output_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_adapter_0",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _record_collected_responses_output_item_event(
    *,
    event: Any,
    output_items: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    key_aliases: dict[str, str],
    key_by_output_index: dict[int, str],
) -> None:
    item = _coerce_namespace_to_mapping(getattr(event, "item", None))
    if not isinstance(item, dict):
        return

    output_index = getattr(event, "output_index", None)
    raw_key = _responses_output_stream_key(
        item=item,
        output_index=output_index,
        fallback_index=len(ordered_keys),
    )
    if isinstance(output_index, int) and output_index in key_by_output_index:
        key = key_by_output_index[output_index]
    else:
        key = key_aliases.get(raw_key, raw_key)
    if key not in ordered_keys:
        ordered_keys.append(key)

    existing = output_items.get(key, {})
    merged_item = {**existing, **item}
    if "arguments" in existing and "arguments" not in item:
        merged_item["arguments"] = existing["arguments"]
    output_items[key] = merged_item

    if isinstance(output_index, int):
        key_by_output_index[output_index] = key
    for alias in (raw_key, item.get("id"), item.get("call_id")):
        if isinstance(alias, str) and alias.strip():
            key_aliases[alias.strip()] = key


def _record_collected_responses_arguments_event(
    *,
    event: Any,
    event_type: str,
    output_items: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    key_aliases: dict[str, str],
    key_by_output_index: dict[int, str],
) -> None:
    item_id = getattr(event, "item_id", None)
    output_index = getattr(event, "output_index", None)
    raw_key = _responses_output_stream_key(
        output_index=output_index,
        item_id=item_id,
        fallback_index=len(ordered_keys),
    )
    if isinstance(output_index, int) and output_index in key_by_output_index:
        key = key_by_output_index[output_index]
    else:
        key = key_aliases.get(raw_key, raw_key)
    if key not in ordered_keys:
        ordered_keys.append(key)

    existing = output_items.get(key, {})
    if not existing:
        item_type = "mcp_call" if "mcp_call" in event_type else "function_call"
        existing = {"type": item_type, "id": item_id}
        if item_type == "function_call" and isinstance(item_id, str) and item_id:
            existing["call_id"] = item_id

    value = getattr(event, "arguments", None)
    if not isinstance(value, str):
        value = getattr(event, "delta", None)
    if isinstance(value, str):
        if event_type.endswith(".delta"):
            existing["arguments"] = f"{existing.get('arguments', '')}{value}"
        else:
            existing["arguments"] = value

    output_items[key] = existing
    if isinstance(output_index, int):
        key_by_output_index[output_index] = key
    if isinstance(item_id, str) and item_id.strip():
        key_aliases[item_id.strip()] = key


def _finalize_collected_responses_stream_response(
    *,
    response_dict: dict[str, Any],
    output_text_parts: list[str],
    output_items: dict[str, dict[str, Any]],
    ordered_keys: list[str],
) -> dict[str, Any]:
    streamed_output = [
        output_items[key]
        for key in ordered_keys
        if key in output_items
    ]
    completed_output = response_dict.get("output")
    if (
        output_text_parts
        and not _responses_output_has_message_text(streamed_output)
        and not _responses_output_has_message_text(completed_output)
    ):
        streamed_output.append(
            _build_collected_responses_text_output_item("".join(output_text_parts))
        )
    if streamed_output:
        response_dict["output"] = _merge_responses_output_lists(
            completed_output if isinstance(completed_output, list) else [],
            streamed_output,
        )
    elif not response_dict.get("output") and output_text_parts:
        response_dict["output"] = [
            _build_collected_responses_text_output_item("".join(output_text_parts))
        ]
    return response_dict


def _build_empty_success_responses_diagnostic(
    *,
    response_body: dict[str, Any],
    diagnostic_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    output = response_body.get("output") or []
    usage = response_body.get("usage") or {}
    diagnostic = {
        "id": response_body.get("id"),
        "status": response_body.get("status"),
        "model": response_body.get("model"),
        "output_count": len(output) if isinstance(output, list) else 0,
        "output_types": [
            item.get("type") for item in output[:20] if isinstance(item, dict)
        ]
        if isinstance(output, list)
        else [],
        "usage": usage if isinstance(usage, dict) else {},
        "error": response_body.get("error"),
        "incomplete_details": response_body.get("incomplete_details"),
    }
    if diagnostic_context:
        diagnostic["context"] = diagnostic_context
    return diagnostic


async def _collect_responses_response_from_stream(
    response: StreamingResponse,
    *,
    event_summaries: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    output_text_parts: list[str] = []
    text_done_keys_seen: set[str] = set()
    output_items: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    key_aliases: dict[str, str] = {}
    key_by_output_index: dict[int, str] = {}
    completed_response_dict: Optional[dict[str, Any]] = None
    event_iterator = _iterate_responses_sse_events(response.body_iterator)
    try:
        async for event in event_iterator:
            event_type = getattr(event, "type", None)
            if event_summaries is not None and len(event_summaries) < 50:
                event_summaries.append(_responses_stream_event_summary(event))
            if event_type in {"response.output_item.added", "response.output_item.done"}:
                _record_collected_responses_output_item_event(
                    event=event,
                    output_items=output_items,
                    ordered_keys=ordered_keys,
                    key_aliases=key_aliases,
                    key_by_output_index=key_by_output_index,
                )
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str):
                    output_text_parts.append(delta)
                    text_done_keys_seen.add(_responses_event_text_key(event))
            if event_type == "response.output_text.done":
                text = getattr(event, "text", None)
                text_key = _responses_event_text_key(event)
                if (
                    isinstance(text, str)
                    and text
                    and text_key not in text_done_keys_seen
                ):
                    output_text_parts.append(text)
                    text_done_keys_seen.add(text_key)
            if event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
            }:
                _record_collected_responses_arguments_event(
                    event=event,
                    event_type=event_type,
                    output_items=output_items,
                    ordered_keys=ordered_keys,
                    key_aliases=key_aliases,
                    key_by_output_index=key_by_output_index,
                )
            if event_type == "response.completed":
                response_payload = getattr(event, "response", None)
                if response_payload is None:
                    continue
                response_dict = _coerce_namespace_to_mapping(response_payload)
                if isinstance(response_dict, dict):
                    completed_response_dict = response_dict
    finally:
        if completed_response_dict is None:
            await event_iterator.aclose()
            body_iterator = getattr(response, "body_iterator", None)
            aclose = getattr(body_iterator, "aclose", None)
            if callable(aclose):
                await aclose()
    if completed_response_dict is not None:
        return _finalize_collected_responses_stream_response(
            response_dict=completed_response_dict,
            output_text_parts=output_text_parts,
            output_items=output_items,
            ordered_keys=ordered_keys,
        )
    raise HTTPException(
        status_code=502,
        detail="OpenAI Responses stream completed without a response payload.",
    )


def _build_anthropic_streaming_response_from_responses_stream(
    response: StreamingResponse,
    *,
    model: str,
    request_body: Optional[dict[str, Any]] = None,
    reject_empty_success: bool = False,
    use_codex_native_tools: bool = False,
) -> StreamingResponse:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
        AnthropicResponsesStreamWrapper,
    )

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_iterate_responses_sse_events(response.body_iterator),
        model=model,
        request_body=request_body,
        reject_empty_success=reject_empty_success,
        use_codex_native_tools=use_codex_native_tools,
    )
    return StreamingResponse(
        wrapper.async_anthropic_sse_wrapper(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type="text/event-stream",
    )


def _get_anthropic_adapter_access_log_target_label(
    target_url: Union[str, httpx.URL],
) -> str:
    parsed_url = urlparse(str(target_url))
    hostname = parsed_url.hostname or "unknown-target"
    path = parsed_url.path or "/"
    query = f"?{parsed_url.query}" if parsed_url.query else ""
    return f"{hostname}{path}{query}"


def _annotate_request_scope_for_adapted_access_log(
    request: Request, target_url: Union[str, httpx.URL]
) -> None:
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return

    target_label = _get_anthropic_adapter_access_log_target_label(target_url)
    existing_path = scope.get("path")
    existing_query_string = scope.get("query_string", b"")
    if isinstance(existing_path, str) and isinstance(existing_query_string, bytes):
        if f" -> {target_label}".encode("utf-8") in existing_query_string:
            return
    if isinstance(existing_path, str) and f" -> {target_label}" in existing_path:
        return

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

    annotated_query = (
        f"{original_query} -> {target_label}"
        if original_query
        else f"adapted_to={target_label}"
    )
    scope["path"] = str(original_path)
    scope["query_string"] = annotated_query.encode("utf-8", errors="replace")


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


def _build_anthropic_streaming_response_from_completion_adapter_stream(
    response_stream: Any,
) -> StreamingResponse:
    return StreamingResponse(
        response_stream,
        media_type="text/event-stream",
    )


def _sanitize_google_code_assist_request_schemas(wrapped_request_body: Any) -> int:
    sanitized_schema_fix_count = 0
    request_payload = (
        wrapped_request_body.get("request")
        if isinstance(wrapped_request_body, dict)
        else None
    )
    request_tools = request_payload.get("tools") if isinstance(request_payload, dict) else None
    if not isinstance(request_tools, list):
        return sanitized_schema_fix_count

    for tool_entry in request_tools:
        if not isinstance(tool_entry, dict):
            continue
        decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
        if not isinstance(decls, list):
            continue
        for declaration in decls:
            if not isinstance(declaration, dict):
                continue
            parameters = declaration.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
                declaration["parameters"] = parameters
                sanitized_schema_fix_count += 1
            sanitized_schema_fix_count += _sanitize_google_code_assist_tool_schema(
                parameters
            )
    return sanitized_schema_fix_count


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
        request_payload = (
            wrapped_request_body.get("request")
            if isinstance(wrapped_request_body, dict)
            else None
        )
        function_names = _extract_google_code_assist_function_names(request_payload)
        litellm_metadata = (
            prepared_request_body.get("litellm_metadata")
            if isinstance(prepared_request_body, dict)
            else None
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
    google_access_token = (
        await _load_valid_local_antigravity_access_token()
        if adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
        else await _load_valid_local_google_oauth_access_token()
    )
    google_project = await _get_or_load_google_code_assist_project(
        google_access_token,
        adapter_provider=adapter_provider,
    )
    google_quota_observation = await _prime_google_code_assist_session(
        google_access_token,
        google_project,
        adapter_provider=adapter_provider,
    )

    is_antigravity_adapter = (
        adapter_provider == _ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER
    )
    route_family = (
        "anthropic_antigravity_completion_adapter"
        if is_antigravity_adapter
        else "anthropic_google_completion_adapter"
    )
    adapter_tag = (
        "anthropic-antigravity-completion-adapter"
        if is_antigravity_adapter
        else "anthropic-google-completion-adapter"
    )
    target_provider_label = "antigravity" if is_antigravity_adapter else "google"
    requested_model = prepared_request_body.get("model")
    google_target_base = _get_code_assist_adapter_target_base(adapter_provider)
    google_model = _normalize_google_completion_adapter_model_name(adapter_model)
    google_adapter_rate_limit_key = _get_google_adapter_rate_limit_key(
        google_model,
        access_token=google_access_token,
        companion_project=google_project,
    )
    if is_antigravity_adapter:
        google_adapter_rate_limit_key = f"antigravity:{google_adapter_rate_limit_key}"
    client_requested_stream = bool(prepared_request_body.get("stream"))
    is_stream = True
    target_endpoint_label = "/v1internal:streamGenerateContent"
    target_query_params = {"alt": "sse"}
    target_url = f"{google_target_base.rstrip('/')}{target_endpoint_label}"
    annotated_target_url = (
        httpx.URL(target_url).copy_with(params=target_query_params)
        if target_query_params
        else httpx.URL(target_url)
    )

    (
        prepared_request_body,
        google_persisted_output_compacted_count,
        google_persisted_output_hooks,
        google_persisted_output_metadata,
    ) = _compact_google_adapter_persisted_output_in_anthropic_request_body(
        prepared_request_body
    )

    prepared_request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(prepared_request_body, route_family),
        tags_to_add=[
            adapter_tag,
            f"anthropic-adapter-model:{google_model}",
            f"anthropic-adapter-target:{target_provider_label}:{target_endpoint_label}",
            *([
                "google-adapter-persisted-output-compacted",
                *[
                    f"google-adapter-persisted-output-hook:{hook}"
                    for hook in sorted(google_persisted_output_hooks)
                    if hook
                ],
            ] if google_persisted_output_compacted_count else []),
        ],
        extra_fields={
            "anthropic_adapter_model": google_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_provider": adapter_provider,
            "anthropic_adapter_target_endpoint": (
                f"{target_provider_label}:{target_endpoint_label}"
            ),
            **(
                {"antigravity_code_assist": True}
                if is_antigravity_adapter
                else {}
            ),
            "google_adapter_persisted_output_compacted": bool(
                google_persisted_output_compacted_count
            ),
            "google_adapter_persisted_output_compacted_count": google_persisted_output_compacted_count,
            "google_adapter_persisted_output_hooks": sorted(google_persisted_output_hooks),
            "google_adapter_persisted_output_metadata": google_persisted_output_metadata,
            **(
                {"google_retrieve_user_quota": google_quota_observation}
                if google_quota_observation
                else {}
            ),
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=(
                        "anthropic.antigravity_completion_adapter"
                        if is_antigravity_adapter
                        else "anthropic.google_completion_adapter"
                    ),
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": google_model,
                        "adapter_provider": adapter_provider,
                        "stream": client_requested_stream,
                        "upstream_stream": True,
                        "persisted_output_compacted_count": google_persisted_output_compacted_count,
                    },
                )
            ],
        },
    )

    wrapped_request_body, tool_name_mapping, completion_messages, gemini_optional_params, litellm_params, completion_message_window_changes = await _build_google_code_assist_request_from_completion_kwargs(
        completion_kwargs=prepared_request_body,
        adapter_model=google_model,
        project=google_project,
        request=request,
    )
    if isinstance(prepared_request_body.get("litellm_metadata"), dict):
        wrapped_request_body["litellm_metadata"] = {
            **dict(wrapped_request_body.get("litellm_metadata") or {}),
            **dict(prepared_request_body["litellm_metadata"]),
        }

    generation_policy_changes = _apply_google_adapter_request_shape_policy(wrapped_request_body)

    adapter_headers = _build_code_assist_adapter_native_headers(
        adapter_provider=adapter_provider,
        access_token=google_access_token,
        model=google_model,
        accept="*/*",
    )
    if isinstance(wrapped_request_body.get("litellm_metadata"), dict):
        if completion_message_window_changes:
            wrapped_request_body["litellm_metadata"]["google_adapter_completion_message_window"] = completion_message_window_changes
        if generation_policy_changes:
            wrapped_request_body["litellm_metadata"]["google_adapter_request_shape_policy"] = generation_policy_changes

    sanitized_schema_fix_count = _sanitize_google_code_assist_request_schemas(
        wrapped_request_body
    )
    _log_google_completion_adapter_debug(
        prepared_request_body=prepared_request_body,
        wrapped_request_body=wrapped_request_body,
        google_model=google_model,
        adapter_headers=adapter_headers,
        sanitized_schema_fix_count=sanitized_schema_fix_count,
        generation_policy_changes=generation_policy_changes,
    )

    return SimpleNamespace(
        adapter_headers=adapter_headers,
        annotated_target_url=annotated_target_url,
        client_requested_stream=client_requested_stream,
        completion_messages=completion_messages,
        gemini_optional_params=gemini_optional_params,
        google_adapter_rate_limit_key=google_adapter_rate_limit_key,
        google_model=google_model,
        is_stream=is_stream,
        litellm_params=litellm_params,
        custom_llm_provider=adapter_provider,
        target_query_params=target_query_params,
        target_url=target_url,
        tool_name_mapping=tool_name_mapping,
        wrapped_request_body=wrapped_request_body,
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
            google_adapter_model_capacity_max_retries=(
                0 if use_alias_candidate_probe else None
            ),
            google_adapter_hidden_retry_budget_seconds=(
                0 if use_alias_candidate_probe else None
            ),
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
            google_adapter_model_capacity_max_retries=(
                0 if use_alias_candidate_probe else None
            ),
            google_adapter_hidden_retry_budget_seconds=(
                0 if use_alias_candidate_probe else None
            ),
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
        if (
            use_alias_candidate_probe
            and _is_codex_google_code_assist_empty_success_model_response(
                model_response
            )
        ):
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
                    "usage": _model_response_usage_dict(
                        _mapping_or_attr_get(model_response, "usage")
                    ),
                },
                adapter_model=adapter_request.google_model,
                adapter="codex_auto_agent_google_code_assist",
                adapter_label="Gemini Code Assist",
            )
        responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=model_response,
            request_input=adapter_request.codex_request_input,
            responses_api_request=adapter_request.responses_api_request,
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

    use_chatgpt_codex_defaults = (
        uses_codex_native_auth or local_codex_headers is not None
    )
    egress_credential_family = (
        "openai" if local_codex_headers is not None else None
    )
    return (
        custom_headers,
        forward_headers,
        use_chatgpt_codex_defaults,
        egress_credential_family,
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
    client_requested_stream = bool(prepared_request_body.get("stream"))
    (
        custom_headers,
        forward_headers,
        use_chatgpt_codex_defaults,
        egress_credential_family,
    ) = await _resolve_anthropic_openai_responses_adapter_auth_context(request)
    (
        prepared_request_body,
        openai_context_compacted_count,
        openai_context_compacted_markers,
        _openai_context_compaction_metadata,
    ) = _compact_openai_adapter_claude_context_in_anthropic_request_body(
        prepared_request_body
    )
    if openai_context_compacted_count > 0:
        verbose_proxy_logger.debug(
            "Compacted Claude Code context for OpenAI Responses adapter; count=%s markers=%s",
            openai_context_compacted_count,
            sorted(openai_context_compacted_markers),
        )
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        use_chatgpt_codex_defaults=use_chatgpt_codex_defaults,
    )
    (
        translated_request_body,
        openai_parallel_instruction_policy_changes,
    ) = _apply_openai_adapter_parallel_instruction_policy(translated_request_body)
    if openai_parallel_instruction_policy_changes:
        verbose_proxy_logger.debug(
            "Applied OpenAI adapter parallel instruction policy; tools=%s original_chars=%s rewritten_chars=%s",
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_tool_names"
            ),
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_original_chars"
            ),
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_rewritten_chars"
            ),
        )
    translated_request_body, forced_tool_choice_changes = (
        _apply_forced_bash_tool_choice_for_responses_adapter(
            prepared_request_body,
            translated_request_body,
        )
    )
    if forced_tool_choice_changes:
        verbose_proxy_logger.debug(
            "Applied OpenAI adapter explicit Bash tool choice: %s",
            forced_tool_choice_changes.get("forced_explicit_bash_tool_choice"),
        )
    if use_chatgpt_codex_defaults:
        translated_request_body = _add_codex_request_breakout_logging_metadata(
            translated_request_body
        )
    if _responses_request_contains_mcp_tools(translated_request_body):
        raise HTTPException(
            status_code=400,
            detail=(
                "Anthropic adapter does not currently support raw MCP server/toolset "
                "requests (`mcp_servers` / `mcp_toolset`). Use Claude Code-exposed tools "
                "such as `mcp__...` or call the native OpenAI Responses API directly."
            ),
        )

    adapter_provider = litellm.LlmProviders.OPENAI.value
    target_base_url = _get_anthropic_adapter_openai_target_base(
        request,
        prefer_chatgpt_codex_backend=use_chatgpt_codex_defaults,
    )
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint="/v1/responses",
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.OPENAI.value,
    )

    upstream_response = await pass_through_request(
        request=request,
        target=str(target_url),
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=translated_request_body,
        forward_headers=forward_headers,
        allowed_forward_headers=list(_ANTHROPIC_ADAPTER_OPENAI_FORWARD_HEADER_ALLOWLIST),
        allowed_pass_through_prefixed_headers=list(_ANTHROPIC_ADAPTER_OPENAI_XPASS_HEADER_ALLOWLIST),
        stream=bool(translated_request_body.get("stream")),
        custom_llm_provider=adapter_provider,
        egress_credential_family=egress_credential_family,
        expected_target_family="openai",
        retryable_upstream_status_codes=(
            [429] + _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
            if use_alias_candidate_probe
            else [429]
        ),
        caller_managed_hidden_retry=use_alias_candidate_probe,
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    if isinstance(upstream_response, StreamingResponse):
        upstream_response = await _validate_alias_candidate_responses_stream_if_needed(
            upstream_response,
            enabled=use_alias_candidate_probe,
            adapter_model=adapter_model,
            adapter="anthropic_openai_responses_adapter",
            adapter_label="OpenAI",
        )
        if not client_requested_stream:
            response_body = await _collect_responses_response_from_stream(
                upstream_response
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body,
                use_codex_native_tools=use_chatgpt_codex_defaults,
                retryable_failed_response=use_alias_candidate_probe,
                failed_response_adapter_model=adapter_model,
                failed_response_adapter="anthropic_openai_responses_adapter",
                failed_response_adapter_label="OpenAI",
            )
            _copy_translated_anthropic_adapter_response_headers(
                translated_response=translated_response,
                upstream_response=upstream_response,
            )
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
            request_body=translated_request_body,
            use_codex_native_tools=use_chatgpt_codex_defaults,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from OpenAI Responses passthrough.",
        )

    response_body = json.loads(_decode_http_response_body(upstream_response.body))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body,
        use_codex_native_tools=use_chatgpt_codex_defaults,
        retryable_failed_response=use_alias_candidate_probe,
        failed_response_adapter_model=adapter_model,
        failed_response_adapter="anthropic_openai_responses_adapter",
        failed_response_adapter_label="OpenAI",
    )
    _copy_translated_anthropic_adapter_response_headers(
        translated_response=translated_response,
        upstream_response=upstream_response,
    )
    translated_response.status_code = upstream_response.status_code
    return translated_response

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
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_xai_oauth_responses_adapter",
        tag_prefix="anthropic-xai-oauth-responses-adapter",
        span_name="anthropic.xai_oauth_responses_adapter",
        target_endpoint="xai:/v1/responses",
    )
    (
        translated_request_body,
        openai_parallel_instruction_policy_changes,
    ) = _apply_openai_adapter_parallel_instruction_policy(translated_request_body)
    if openai_parallel_instruction_policy_changes:
        verbose_proxy_logger.debug(
            "Applied xAI OAuth responses adapter parallel instruction policy; tools=%s original_chars=%s rewritten_chars=%s",
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_tool_names"
            ),
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_original_chars"
            ),
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_rewritten_chars"
            ),
        )
    translated_request_body, forced_tool_choice_changes = (
        _apply_forced_bash_tool_choice_for_responses_adapter(
            prepared_request_body,
            translated_request_body,
        )
    )
    if forced_tool_choice_changes:
        verbose_proxy_logger.debug(
            "Applied xAI OAuth adapter explicit Bash tool choice: %s",
            forced_tool_choice_changes.get("forced_explicit_bash_tool_choice"),
        )
    translated_request_body, _xai_unsupported_request_params = (
        _drop_unsupported_codex_request_params_from_request_body(
            translated_request_body
        )
    )

    try:
        prepared_oa_xai, target_base_url, xai_api_key = (
            await _prepare_oa_xai_passthrough_request(
                translated_request_body,
                sanitize_responses_request=True,
            )
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and _xai_oauth_candidate_unavailable_detail(exc) is not None
        ):
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    if not prepared_oa_xai or target_base_url is None or xai_api_key is None:
        missing_credential_error = Exception(
            "Anthropic adapter requests for xAI OAuth models require a managed xAI OAuth credential."
        )
        if use_alias_candidate_probe:
            _raise_xai_oauth_auto_agent_candidate_unavailable(
                missing_credential_error
            )
        raise missing_credential_error
    translated_request_body["model"] = _to_xai_native_passthrough_model(
        translated_request_body.get("model")
    )

    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint="/v1/responses",
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.XAI,
    )
    custom_headers = BaseOpenAIPassThroughHandler._assemble_headers(
        api_key=xai_api_key,
        request=request,
    )

    try:
        upstream_response = await pass_through_request(
            request=request,
            target=str(target_url),
            custom_headers=custom_headers,
            user_api_key_dict=user_api_key_dict,
            custom_body=translated_request_body,
            forward_headers=False,
            stream=bool(translated_request_body.get("stream")),
            custom_llm_provider=litellm.LlmProviders.XAI.value,
            egress_credential_family="xai",
            expected_target_family="xai",
            retryable_upstream_status_codes=(
                _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
                if use_alias_candidate_probe
                else None
            ),
            caller_managed_hidden_retry=use_alias_candidate_probe,
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and _xai_oauth_candidate_unavailable_detail(exc) is not None
        ):
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    if isinstance(upstream_response, StreamingResponse):
        upstream_response = await _validate_alias_candidate_responses_stream_if_needed(
            upstream_response,
            enabled=use_alias_candidate_probe,
            adapter_model=adapter_model,
            adapter="anthropic_xai_oauth_responses_adapter",
            adapter_label="xAI OAuth",
        )
        if not client_requested_stream:
            response_body = await _collect_responses_response_from_stream(
                upstream_response
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body,
                retryable_failed_response=use_alias_candidate_probe,
                failed_response_adapter_model=adapter_model,
                failed_response_adapter="anthropic_xai_oauth_responses_adapter",
                failed_response_adapter_label="xAI OAuth",
            )
            _copy_translated_anthropic_adapter_response_headers(
                translated_response=translated_response,
                upstream_response=upstream_response,
            )
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
            request_body=translated_request_body,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from xAI Responses passthrough.",
        )

    response_body = json.loads(_decode_http_response_body(upstream_response.body))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body,
        retryable_failed_response=use_alias_candidate_probe,
        failed_response_adapter_model=adapter_model,
        failed_response_adapter="anthropic_xai_oauth_responses_adapter",
        failed_response_adapter_label="xAI OAuth",
    )
    _copy_translated_anthropic_adapter_response_headers(
        translated_response=translated_response,
        upstream_response=upstream_response,
    )
    translated_response.status_code = upstream_response.status_code
    return translated_response


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
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_grok_native_responses_adapter",
        tag_prefix="anthropic-grok-native-responses-adapter",
        span_name="anthropic.grok_native_responses_adapter",
        target_endpoint="xai:/v1/responses",
    )
    (
        translated_request_body,
        openai_parallel_instruction_policy_changes,
    ) = _apply_openai_adapter_parallel_instruction_policy(translated_request_body)
    if openai_parallel_instruction_policy_changes:
        verbose_proxy_logger.debug(
            "Applied Grok native responses adapter parallel instruction policy; tools=%s original_chars=%s rewritten_chars=%s",
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_tool_names"
            ),
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_original_chars"
            ),
            openai_parallel_instruction_policy_changes.get(
                "openai_adapter_parallel_instruction_rewritten_chars"
            ),
        )
    translated_request_body, forced_tool_choice_changes = (
        _apply_forced_bash_tool_choice_for_responses_adapter(
            prepared_request_body,
            translated_request_body,
        )
    )
    if forced_tool_choice_changes:
        verbose_proxy_logger.debug(
            "Applied Grok native adapter explicit Bash tool choice: %s",
            forced_tool_choice_changes.get("forced_explicit_bash_tool_choice"),
        )
    translated_request_body, _grok_unsupported_request_params = (
        _drop_unsupported_codex_request_params_from_request_body(
            translated_request_body
        )
    )

    try:
        (
            prepared_grok_native,
            target_base_url,
            grok_headers,
            translated_request_body,
        ) = await _prepare_grok_native_oauth_passthrough_request(
            translated_request_body,
            request=request,
            tags_to_add=[
                "anthropic-grok-native-responses-adapter-entrypoint",
            ],
            extra_fields={
                "anthropic_grok_native_requested_model": prepared_request_body.get("model"),
                "anthropic_grok_native_adapter_model": adapter_model,
                "anthropic_adapter_target_endpoint": "xai:/v1/responses",
                "grok_native_entrypoint": "anthropic_messages",
                "passthrough_route_family": "anthropic_grok_native_responses_adapter",
                "route_family": "anthropic_grok_native_responses_adapter",
            },
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and _grok_native_candidate_unavailable_detail(exc) is not None
        ):
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise
    if not prepared_grok_native or target_base_url is None:
        if use_alias_candidate_probe:
            _raise_grok_native_auto_agent_candidate_unavailable(
                Exception(
                    "Anthropic adapter requests for Grok native OAuth models "
                    "require a Grok OIDC credential."
                )
            )
        raise Exception(
            "Anthropic adapter requests for Grok native OAuth models require a Grok OIDC credential."
        )

    target_url = _join_grok_passthrough_url(
        base_target_url=target_base_url,
        endpoint="/v1/responses",
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    try:
        upstream_response = await pass_through_request(
            request=request,
            target=str(target_url),
            custom_headers=grok_headers,
            user_api_key_dict=user_api_key_dict,
            custom_body=translated_request_body,
            forward_headers=False,
            stream=bool(translated_request_body.get("stream")),
            custom_llm_provider=litellm.LlmProviders.XAI.value,
            egress_credential_family="xai",
            expected_target_family="xai",
            retryable_upstream_status_codes=(
                _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
                if use_alias_candidate_probe
                else None
            ),
            caller_managed_hidden_retry=use_alias_candidate_probe,
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and _grok_native_candidate_unavailable_detail(exc) is not None
        ):
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise

    if isinstance(upstream_response, StreamingResponse):
        upstream_response = await _validate_alias_candidate_responses_stream_if_needed(
            upstream_response,
            enabled=use_alias_candidate_probe,
            adapter_model=adapter_model,
            adapter="anthropic_grok_native_responses_adapter",
            adapter_label="Grok native",
        )
        if not client_requested_stream:
            response_body = await _collect_responses_response_from_stream(
                upstream_response
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body,
                retryable_failed_response=use_alias_candidate_probe,
                failed_response_adapter_model=adapter_model,
                failed_response_adapter="anthropic_grok_native_responses_adapter",
                failed_response_adapter_label="Grok native",
            )
            _copy_translated_anthropic_adapter_response_headers(
                translated_response=translated_response,
                upstream_response=upstream_response,
            )
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
            request_body=translated_request_body,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from Grok native Responses passthrough.",
        )

    response_body = json.loads(_decode_http_response_body(upstream_response.body))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body,
        retryable_failed_response=use_alias_candidate_probe,
        failed_response_adapter_model=adapter_model,
        failed_response_adapter="anthropic_grok_native_responses_adapter",
        failed_response_adapter_label="Grok native",
    )
    _copy_translated_anthropic_adapter_response_headers(
        translated_response=translated_response,
        upstream_response=upstream_response,
    )
    translated_response.status_code = upstream_response.status_code
    return translated_response


async def _handle_anthropic_xai_oauth_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    client_requested_stream = bool(prepared_request_body.get("stream"))
    requested_model = prepared_request_body.get("model")
    route_family = "anthropic_xai_oauth_completion_adapter"
    target_endpoint_label = "xai:/v1/chat/completions"

    prepared_request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(prepared_request_body, route_family),
        tags_to_add=[
            "anthropic-xai-oauth-completion-adapter",
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint_label}",
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_target_endpoint": target_endpoint_label,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.xai_oauth_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": adapter_model,
                        "stream": client_requested_stream,
                    },
                )
            ],
        },
    )
    forced_tool_choice_changes = _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
        prepared_request_body,
    )
    if forced_tool_choice_changes:
        prepared_request_body = _merge_litellm_metadata(
            prepared_request_body,
            extra_fields=forced_tool_choice_changes,
        )

    prepared_oa_xai, target_base_url, xai_api_key = (
        await _prepare_oa_xai_passthrough_request(prepared_request_body)
    )
    if not prepared_oa_xai or target_base_url is None or xai_api_key is None:
        raise Exception(
            "Anthropic adapter requests for xAI OAuth models require a managed xAI OAuth credential."
        )

    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint="/v1/chat/completions",
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.XAI,
    )
    validation_headers = {"Authorization": f"Bearer {xai_api_key}"}
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=str(target_url),
        headers=validation_headers,
        credential_family="xai",
        expected_target_family="xai",
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    completion_response = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=int(prepared_request_body.get("max_tokens") or 1024),
        messages=prepared_request_body.get("messages") or [],
        model=cast(str, prepared_request_body.get("model")),
        metadata=_build_completion_adapter_metadata(prepared_request_body),
        stop_sequences=prepared_request_body.get("stop_sequences"),
        stream=client_requested_stream,
        system=prepared_request_body.get("system"),
        temperature=prepared_request_body.get("temperature"),
        thinking=prepared_request_body.get("thinking"),
        tool_choice=prepared_request_body.get("tool_choice"),
        tools=prepared_request_body.get("tools"),
        top_k=prepared_request_body.get("top_k"),
        top_p=prepared_request_body.get("top_p"),
        output_format=prepared_request_body.get("output_format"),
        output_config=prepared_request_body.get("output_config"),
        custom_llm_provider=litellm.LlmProviders.XAI.value,
        api_key=xai_api_key,
        api_base=target_base_url,
        litellm_metadata=prepared_request_body.get("litellm_metadata") or {},
        proxy_server_request={
            "headers": dict(request.headers),
            "body": prepared_request_body,
        },
    )

    if client_requested_stream:
        return _build_anthropic_streaming_response_from_completion_adapter_stream(
            completion_response,
        )
    return _build_anthropic_response_from_completion_adapter_response(
        completion_response,
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
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    client_requested_stream = bool(prepared_request_body.get("stream"))
    use_fake_stream = client_requested_stream and _should_force_fake_stream_for_nvidia_adapter_model(
        adapter_model
    )
    upstream_stream = client_requested_stream and not use_fake_stream
    requested_model = prepared_request_body.get("model")
    route_family = "anthropic_nvidia_completion_adapter"
    target_endpoint_label = "nvidia:/v1/chat/completions"

    prepared_request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(prepared_request_body, route_family),
        tags_to_add=[
            "anthropic-nvidia-completion-adapter",
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint_label}",
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_target_endpoint": target_endpoint_label,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.nvidia_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": adapter_model,
                        "stream": client_requested_stream,
                        "upstream_stream": upstream_stream,
                        "fake_stream": use_fake_stream,
                    },
                )
            ],
        },
    )
    forced_tool_choice_changes = _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
        prepared_request_body,
    )
    if forced_tool_choice_changes:
        prepared_request_body = _merge_litellm_metadata(
            prepared_request_body,
            extra_fields=forced_tool_choice_changes,
        )

    nvidia_api_key = _get_anthropic_adapter_nvidia_api_key()
    if nvidia_api_key is None:
        raise Exception(
            "Anthropic adapter requests for NVIDIA models require 'AAWM_NVIDIA_API_KEY', "
            "'NVIDIA_NIM_API_KEY', or 'NVIDIA_API_KEY' in environment."
        )

    target_base_url = _get_anthropic_adapter_nvidia_target_base()
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint="/v1/chat/completions",
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.NVIDIA_NIM,
    )
    validation_headers = {"Authorization": f"Bearer {nvidia_api_key}"}
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=str(target_url),
        headers=validation_headers,
        credential_family="nvidia",
        expected_target_family="nvidia",
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    completion_response = await _perform_nvidia_completion_adapter_operation(
        adapter_model=adapter_model,
        operation=lambda: LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=int(prepared_request_body.get("max_tokens") or 1024),
            messages=prepared_request_body.get("messages") or [],
            model=adapter_model,
            metadata=_build_completion_adapter_metadata(prepared_request_body),
            stop_sequences=prepared_request_body.get("stop_sequences"),
            stream=upstream_stream,
            system=prepared_request_body.get("system"),
            temperature=prepared_request_body.get("temperature"),
            thinking=prepared_request_body.get("thinking"),
            tool_choice=prepared_request_body.get("tool_choice"),
            tools=prepared_request_body.get("tools"),
            top_k=prepared_request_body.get("top_k"),
            top_p=prepared_request_body.get("top_p"),
            output_format=prepared_request_body.get("output_format"),
            output_config=prepared_request_body.get("output_config"),
            custom_llm_provider=litellm.LlmProviders.NVIDIA_NIM.value,
            api_key=nvidia_api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            timeout=_get_nvidia_adapter_request_timeout_seconds(adapter_model),
            max_retries=_get_nvidia_adapter_inner_max_retries(),
            litellm_metadata=prepared_request_body.get("litellm_metadata") or {},
            proxy_server_request={
                "headers": dict(request.headers),
                "body": prepared_request_body,
            },
        ),
    )
    if client_requested_stream:
        if use_fake_stream:
            return _build_anthropic_streaming_response_from_completion_adapter_stream(
                FakeAnthropicMessagesStreamIterator(completion_response),
            )
        return _build_anthropic_streaming_response_from_completion_adapter_stream(
            completion_response,
        )
    return _build_anthropic_response_from_completion_adapter_response(
        completion_response,
    )


async def _handle_anthropic_openrouter_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    client_requested_stream = bool(prepared_request_body.get("stream"))
    requested_model = prepared_request_body.get("model")
    route_family = "anthropic_openrouter_completion_adapter"
    target_endpoint_label = "openrouter:/v1/chat/completions"
    upstream_adapter_model = (
        _get_openrouter_completion_adapter_upstream_model(adapter_model)
        or adapter_model
    )

    prepared_request_body = _merge_litellm_metadata(
        _add_route_family_logging_metadata(prepared_request_body, route_family),
        tags_to_add=[
            "anthropic-openrouter-completion-adapter",
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint_label}",
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_target_endpoint": target_endpoint_label,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.openrouter_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": adapter_model,
                        "stream": client_requested_stream,
                    },
                )
            ],
        },
    )
    forced_tool_choice_changes = _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
        prepared_request_body,
    )
    if forced_tool_choice_changes:
        prepared_request_body = _merge_litellm_metadata(
            prepared_request_body,
            extra_fields=forced_tool_choice_changes,
        )

    openrouter_api_key = _get_anthropic_adapter_openrouter_api_key()
    if openrouter_api_key is None:
        _raise_openrouter_auto_agent_candidate_unavailable(
            "Anthropic adapter requests for OpenRouter models require 'AAWM_OPENROUTER_API_KEY' or 'OPENROUTER_API_KEY' in environment."
        )

    target_base_url = _get_anthropic_adapter_openrouter_target_base()
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint="/v1/chat/completions",
        base_target_url=target_base_url,
    )
    target_url = BaseOpenAIPassThroughHandler._join_url_paths(
        httpx.URL(target_base_url),
        normalized_endpoint,
        litellm.LlmProviders.OPENROUTER.value,
    )
    validation_headers = {
        **_build_openrouter_default_headers(),
        "Authorization": f"Bearer {openrouter_api_key}",
    }
    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=str(target_url),
        headers=validation_headers,
        credential_family="openrouter",
        expected_target_family="openrouter",
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    completion_response = await _perform_openrouter_completion_adapter_operation(
        adapter_model=upstream_adapter_model,
        operation=lambda: LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=int(prepared_request_body.get("max_tokens") or 1024),
            messages=prepared_request_body.get("messages") or [],
            model=upstream_adapter_model,
            metadata=_build_completion_adapter_metadata(prepared_request_body),
            stop_sequences=prepared_request_body.get("stop_sequences"),
            stream=client_requested_stream,
            system=prepared_request_body.get("system"),
            temperature=prepared_request_body.get("temperature"),
            thinking=prepared_request_body.get("thinking"),
            tool_choice=prepared_request_body.get("tool_choice"),
            tools=prepared_request_body.get("tools"),
            top_k=prepared_request_body.get("top_k"),
            top_p=prepared_request_body.get("top_p"),
            output_format=prepared_request_body.get("output_format"),
            output_config=prepared_request_body.get("output_config"),
            custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
            api_key=openrouter_api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            headers=_build_openrouter_default_headers(),
            litellm_metadata=prepared_request_body.get("litellm_metadata") or {},
            proxy_server_request={
                "headers": dict(request.headers),
                "body": prepared_request_body,
            },
        ),
    )

    if client_requested_stream:
        return _build_anthropic_streaming_response_from_completion_adapter_stream(
            completion_response,
        )
    return _build_anthropic_response_from_completion_adapter_response(
        completion_response,
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
    client_requested_stream = bool(prepared_request_body.get("stream"))
    (
        prepared_request_body,
        openrouter_context_compacted_count,
        openrouter_context_compacted_markers,
        _openrouter_context_compaction_metadata,
    ) = _compact_openai_adapter_claude_context_in_anthropic_request_body(
        prepared_request_body,
        tag_prefix="openrouter-adapter",
        metadata_prefix="openrouter_adapter",
        span_name="openrouter_adapter.claude_context_compaction",
    )
    if openrouter_context_compacted_count > 0:
        verbose_proxy_logger.debug(
            "Compacted Claude Code context for OpenRouter Responses adapter; count=%s markers=%s",
            openrouter_context_compacted_count,
            sorted(openrouter_context_compacted_markers),
        )
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_openrouter_responses_adapter",
        tag_prefix="anthropic-openrouter-responses-adapter",
        span_name="anthropic.openrouter_responses_adapter",
        target_endpoint="openrouter:/v1/responses",
    )
    (
        translated_request_body,
        openrouter_parallel_instruction_policy_changes,
    ) = _apply_openrouter_adapter_parallel_instruction_policy(translated_request_body)
    if openrouter_parallel_instruction_policy_changes:
        verbose_proxy_logger.debug(
            "Applied OpenRouter adapter parallel instruction policy; tools=%s",
            openrouter_parallel_instruction_policy_changes.get(
                "openrouter_adapter_parallel_instruction_tool_names"
            ),
        )
    translated_request_body, forced_tool_choice_changes = (
        _apply_forced_bash_tool_choice_for_responses_adapter(
            prepared_request_body,
            translated_request_body,
        )
    )
    if _responses_request_contains_mcp_tools(translated_request_body):
        raise HTTPException(
            status_code=400,
            detail=(
                "Anthropic adapter does not currently support raw MCP server/toolset "
                "requests (`mcp_servers` / `mcp_toolset`). Use Claude Code-exposed tools "
                "such as `mcp__...` or call the native OpenAI Responses API directly."
            ),
        )

    openrouter_api_key = _get_anthropic_adapter_openrouter_api_key()
    if openrouter_api_key is None:
        _raise_openrouter_auto_agent_candidate_unavailable(
            "Anthropic adapter requests for OpenRouter models require 'AAWM_OPENROUTER_API_KEY' or 'OPENROUTER_API_KEY' in environment."
        )

    adapter_provider = litellm.LlmProviders.OPENROUTER.value
    target_base_url = _get_anthropic_adapter_openrouter_target_base()
    normalized_endpoint = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
        endpoint="/v1/responses",
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

    upstream_response = await _perform_openrouter_adapter_pass_through_request(
        adapter_model=adapter_model,
        request=request,
        target=str(target_url),
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=translated_request_body,
        forward_headers=False,
        allowed_forward_headers=[],
        allowed_pass_through_prefixed_headers=[],
        stream=bool(translated_request_body.get("stream")),
        custom_llm_provider=adapter_provider,
        egress_credential_family="openrouter",
        expected_target_family="openrouter",
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    if isinstance(upstream_response, StreamingResponse):
        upstream_response = await _validate_alias_candidate_responses_stream_if_needed(
            upstream_response,
            enabled=use_alias_candidate_probe,
            adapter_model=adapter_model,
            adapter="anthropic_openrouter_responses_adapter",
            adapter_label="OpenRouter",
        )
        if not client_requested_stream:
            response_event_summaries: list[dict[str, Any]] = []
            response_body = await _collect_responses_response_from_stream(
                upstream_response,
                event_summaries=response_event_summaries,
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body,
                reject_empty_success=True,
                diagnostic_context={
                    "adapter": "openrouter_responses",
                    "adapter_model": adapter_model,
                    "stream_events": response_event_summaries,
                    "request_model": translated_request_body.get("model"),
                    "request_stream": translated_request_body.get("stream"),
                },
                retryable_failed_response=use_alias_candidate_probe,
                failed_response_adapter_model=adapter_model,
                failed_response_adapter="anthropic_openrouter_responses_adapter",
                failed_response_adapter_label="OpenRouter",
            )
            _copy_translated_anthropic_adapter_response_headers(
                translated_response=translated_response,
                upstream_response=upstream_response,
            )
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
            request_body=translated_request_body,
            reject_empty_success=True,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from OpenRouter Responses passthrough.",
        )

    response_body = json.loads(_decode_http_response_body(upstream_response.body))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body,
        reject_empty_success=True,
        diagnostic_context={
            "adapter": "openrouter_responses",
            "adapter_model": adapter_model,
            "request_model": translated_request_body.get("model"),
            "request_stream": translated_request_body.get("stream"),
        },
        retryable_failed_response=use_alias_candidate_probe,
        failed_response_adapter_model=adapter_model,
        failed_response_adapter="anthropic_openrouter_responses_adapter",
        failed_response_adapter_label="OpenRouter",
    )
    _copy_translated_anthropic_adapter_response_headers(
        translated_response=translated_response,
        upstream_response=upstream_response,
    )
    translated_response.status_code = upstream_response.status_code
    return translated_response


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
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_opencode_zen_responses_adapter",
        tag_prefix="anthropic-opencode-zen-responses-adapter",
        span_name="anthropic.opencode_zen_responses_adapter",
        target_endpoint="opencode_zen:/v1/responses",
    )
    translated_request_body = _add_opencode_zen_logging_metadata(
        translated_request_body,
        route_family="anthropic_opencode_zen_responses_adapter",
        tag_prefix="anthropic-opencode-zen-responses-adapter",
        requested_model=prepared_request_body.get("model"),
        adapter_model=adapter_model,
        input_shape="anthropic_messages",
        output_shape="anthropic_messages",
    )
    (
        translated_request_body,
        openrouter_parallel_instruction_policy_changes,
    ) = _apply_openrouter_adapter_parallel_instruction_policy(translated_request_body)
    if openrouter_parallel_instruction_policy_changes:
        verbose_proxy_logger.debug(
            "Applied OpenCode Zen adapter parallel instruction policy; tools=%s",
            openrouter_parallel_instruction_policy_changes.get(
                "openrouter_adapter_parallel_instruction_tool_names"
            ),
        )
    translated_request_body, forced_tool_choice_changes = (
        _apply_forced_bash_tool_choice_for_responses_adapter(
            prepared_request_body,
            translated_request_body,
        )
    )
    if forced_tool_choice_changes:
        verbose_proxy_logger.debug(
            "Applied OpenCode Zen adapter explicit Bash tool choice: %s",
            forced_tool_choice_changes.get("forced_explicit_bash_tool_choice"),
        )
    if _responses_request_contains_mcp_tools(translated_request_body):
        raise HTTPException(
            status_code=400,
            detail=(
                "Anthropic adapter does not currently support raw MCP server/toolset "
                "requests (`mcp_servers` / `mcp_toolset`). Use Claude Code-exposed tools "
                "such as `mcp__...` or call the native OpenAI Responses API directly."
            ),
        )

    target_base_url = _get_opencode_zen_target_base()
    target_url = _join_opencode_zen_passthrough_url(
        base_target_url=target_base_url,
        endpoint="/v1/responses",
    )
    custom_headers = await _build_opencode_zen_headers(
        request,
        use_alias_candidate_probe=use_alias_candidate_probe,
    )

    try:
        upstream_response = await pass_through_request(
            request=request,
            target=target_url,
            custom_headers=custom_headers,
            user_api_key_dict=user_api_key_dict,
            custom_body=translated_request_body,
            forward_headers=False,
            allowed_forward_headers=[],
            allowed_pass_through_prefixed_headers=[],
            stream=bool(translated_request_body.get("stream")),
            custom_llm_provider=_OPENCODE_ZEN_PROVIDER,
            egress_credential_family="opencode",
            expected_target_family="opencode",
            retryable_upstream_status_codes=(
                _AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
                if use_alias_candidate_probe
                else None
            ),
            caller_managed_hidden_retry=use_alias_candidate_probe,
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and _opencode_zen_candidate_unavailable_detail(exc) is not None
        ):
            _raise_opencode_zen_auto_agent_candidate_unavailable(exc)
        raise
    _annotate_request_scope_for_adapted_access_log(request, httpx.URL(target_url))

    if isinstance(upstream_response, StreamingResponse):
        if not client_requested_stream:
            response_body = await _collect_responses_response_from_stream(
                upstream_response
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body,
                reject_empty_success=True,
                diagnostic_context={
                    "adapter": "opencode_zen_responses",
                    "adapter_model": adapter_model,
                    "request_model": translated_request_body.get("model"),
                    "request_stream": translated_request_body.get("stream"),
                },
            )
            _copy_translated_anthropic_adapter_response_headers(
                translated_response=translated_response,
                upstream_response=upstream_response,
            )
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
            request_body=translated_request_body,
            reject_empty_success=True,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from OpenCode Zen Responses passthrough.",
        )

    response_body = json.loads(_decode_http_response_body(upstream_response.body))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body,
        reject_empty_success=True,
        diagnostic_context={
            "adapter": "opencode_zen_responses",
            "adapter_model": adapter_model,
            "request_model": translated_request_body.get("model"),
            "request_stream": translated_request_body.get("stream"),
        },
    )
    _copy_translated_anthropic_adapter_response_headers(
        translated_response=translated_response,
        upstream_response=upstream_response,
    )
    translated_response.status_code = upstream_response.status_code
    return translated_response


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
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    _ = endpoint, fastapi_response, user_api_key_dict
    client_requested_stream = bool(prepared_request_body.get("stream"))
    requested_model = prepared_request_body.get("model")
    route_family = "anthropic_opencode_zen_completion_adapter"
    target_endpoint_label = "opencode_zen:/v1/chat/completions"

    prepared_request_body = _add_opencode_zen_logging_metadata(
        prepared_request_body,
        route_family=route_family,
        tag_prefix="anthropic-opencode-zen-completion-adapter",
        requested_model=requested_model,
        adapter_model=adapter_model,
        input_shape="anthropic_messages",
        output_shape="anthropic_messages",
    )
    prepared_request_body = _merge_litellm_metadata(
        prepared_request_body,
        tags_to_add=[
            f"anthropic-adapter-model:{adapter_model}",
            f"anthropic-adapter-target:{target_endpoint_label}",
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": requested_model,
            "anthropic_adapter_target_endpoint": target_endpoint_label,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.opencode_zen_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": adapter_model,
                        "stream": client_requested_stream,
                    },
                )
            ],
        },
    )
    forced_tool_choice_changes = _maybe_force_explicit_bash_tool_choice_for_completion_adapter(
        prepared_request_body,
    )
    if forced_tool_choice_changes:
        prepared_request_body = _merge_litellm_metadata(
            prepared_request_body,
            extra_fields=forced_tool_choice_changes,
        )

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

    try:
        completion_response = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=int(prepared_request_body.get("max_tokens") or 1024),
            messages=prepared_request_body.get("messages") or [],
            model=adapter_model,
            metadata=_build_completion_adapter_metadata(prepared_request_body),
            stop_sequences=prepared_request_body.get("stop_sequences"),
            stream=client_requested_stream,
            system=prepared_request_body.get("system"),
            temperature=prepared_request_body.get("temperature"),
            thinking=prepared_request_body.get("thinking"),
            tool_choice=prepared_request_body.get("tool_choice"),
            tools=prepared_request_body.get("tools"),
            top_k=prepared_request_body.get("top_k"),
            top_p=prepared_request_body.get("top_p"),
            output_format=prepared_request_body.get("output_format"),
            output_config=prepared_request_body.get("output_config"),
            custom_llm_provider=litellm.LlmProviders.OPENAI.value,
            api_key=api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            litellm_metadata=prepared_request_body.get("litellm_metadata") or {},
            proxy_server_request={
                "headers": dict(request.headers),
                "body": prepared_request_body,
            },
        )
    except Exception as exc:
        if (
            use_alias_candidate_probe
            and _opencode_zen_candidate_unavailable_detail(exc) is not None
        ):
            _raise_opencode_zen_auto_agent_candidate_unavailable(exc)
        raise

    if client_requested_stream:
        return _build_anthropic_streaming_response_from_completion_adapter_stream(
            completion_response,
        )
    return _build_anthropic_response_from_completion_adapter_response(
        completion_response,
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


def _is_claude_persisted_output_expansion_enabled() -> bool:
    value = os.getenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _get_claude_persisted_output_root() -> Path:
    return Path(
        os.getenv(
            "LITELLM_CLAUDE_PERSISTED_OUTPUT_ROOT", "/home/zepfu/.claude/projects"
        )
    ).expanduser()


def _resolve_claude_persisted_output_path(path_str: str) -> Optional[Path]:
    try:
        root = _get_claude_persisted_output_root().resolve(strict=True)
        candidate = Path(path_str).expanduser().resolve(strict=True)
    except Exception:
        return None

    if not candidate.is_file():
        return None
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if "tool-results" not in candidate.parts:
        return None
    if not candidate.name.endswith("-additionalContext.txt"):
        return None
    return candidate


def _build_claude_persisted_output_source_metadata(
    *, resolved_path: Path, file_text: str
) -> dict[str, Any]:
    file_bytes = file_text.encode("utf-8")
    return {
        "path": str(resolved_path),
        "basename": resolved_path.name,
        "content_hash": hashlib.sha256(file_bytes).hexdigest(),
        "bytes": len(file_bytes),
    }


def _get_google_adapter_persisted_output_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_PERSISTED_OUTPUT_CHAR_CAP")
    )
    if raw_value is None:
        return 2000
    try:
        parsed = int(raw_value)
    except Exception:
        return 2000
    return max(256, parsed)


def _get_google_adapter_auxiliary_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_AUXILIARY_CONTEXT_CHAR_CAP")
    )
    if raw_value is None:
        return 4000
    try:
        parsed = int(raw_value)
    except Exception:
        return 4000
    return max(512, parsed)


def _get_google_adapter_followup_persisted_output_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_PERSISTED_OUTPUT_CHAR_CAP")
    )
    if raw_value is None:
        return 512
    try:
        parsed = int(raw_value)
    except Exception:
        return 512
    return max(128, parsed)


def _get_google_adapter_followup_auxiliary_context_char_cap() -> int:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_GOOGLE_ADAPTER_FOLLOWUP_AUXILIARY_CONTEXT_CHAR_CAP")
    )
    if raw_value is None:
        return 1024
    try:
        parsed = int(raw_value)
    except Exception:
        return 1024
    return max(256, parsed)


def _compact_google_adapter_persisted_output_preview_and_expanded_text(
    text: str,
    *,
    cap: int,
) -> tuple[str, int, set[str], list[dict[str, Any]]]:
    updated_text = text
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []

    preview_matches = list(_CLAUDE_PERSISTED_OUTPUT_INLINE_PATTERN.finditer(text))
    for match in reversed(preview_matches):
        hook = match.group("hook")
        resolved_path = match.group("path")
        compacted_block = (
            "<system-reminder>\n"
            f"{hook} hook additional context: <persisted-output>\n"
            f"[Gemini adapter compacted persisted-output preview. Full output saved to: {resolved_path}]\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )
        updated_text = (
            updated_text[: match.start()]
            + compacted_block
            + updated_text[match.end() :]
        )
        compacted_count += 1
        hooks.add(hook.lower())
        metadata_items.append(
            {
                "original_chars": len(match.group(0)),
                "kept_chars": len(compacted_block),
                "mode": "preview_block_cap",
            }
        )

    matches = list(_CLAUDE_EXPANDED_PERSISTED_OUTPUT_INLINE_PATTERN.finditer(updated_text))
    for match in reversed(matches):
        content = match.group("content")
        if len(content) <= cap:
            continue
        hook = match.group("hook")
        summary_lines = [line.strip() for line in content.splitlines() if line.strip()][:3]
        summary_text = "\n".join(summary_lines)
        truncated = summary_text[:cap].rstrip()
        compacted_block = (
            "<system-reminder>\n"
            f"{hook} hook additional context: <persisted-output>\n"
            f"{truncated}\n\n"
            f"[Gemini adapter compacted persisted-output from {len(content)} chars to {len(truncated)} chars. Refer to current prompt and tools for full context.]\n"
            "</persisted-output>\n"
            "</system-reminder>\n"
        )
        updated_text = (
            updated_text[: match.start()]
            + compacted_block
            + updated_text[match.end() :]
        )
        compacted_count += 1
        hooks.add(hook.lower())
        metadata_items.append(
            {
                "original_chars": len(content),
                "kept_chars": len(truncated),
            }
        )

    return updated_text, compacted_count, hooks, metadata_items


def _compact_expanded_claude_persisted_output_text_for_google_adapter(
    text: str,
    *,
    persisted_output_char_cap: Optional[int] = None,
    auxiliary_context_char_cap: Optional[int] = None,
) -> Tuple[str, int, set[str], list[dict[str, Any]]]:
    cap = persisted_output_char_cap or _get_google_adapter_persisted_output_char_cap()
    updated_text = text
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    (
        updated_text,
        compacted_count,
        hooks,
        metadata_items,
    ) = _compact_google_adapter_persisted_output_preview_and_expanded_text(
        updated_text,
        cap=cap,
    )

    auxiliary_cap = auxiliary_context_char_cap or _get_google_adapter_auxiliary_context_char_cap()
    auxiliary_matches = list(
        _CLAUDE_EXPANDED_AUXILIARY_CONTEXT_INLINE_PATTERN.finditer(updated_text)
    )
    for match in reversed(auxiliary_matches):
        auxiliary_block = match.group(0)
        if len(auxiliary_block) <= auxiliary_cap:
            continue
        hook = match.group("hook")
        body = match.group("body").lstrip("\n")
        summary_lines = [line.strip() for line in body.splitlines() if line.strip()][:4]
        summary_text = "\n".join(summary_lines).strip()
        if not summary_text:
            summary_text = "[Additional context omitted for Gemini adapter compaction.]"
        truncated = summary_text[:auxiliary_cap].rstrip()
        compacted_block = (
            "<system-reminder>\n"
            f"{hook} hook additional context:\n"
            f"{truncated}\n\n"
            f"[Gemini adapter compacted auxiliary context block from {len(auxiliary_block)} chars to {len(truncated)} chars. Refer to the current prompt, tools, and recent messages for full context.]\n"
            "</system-reminder>\n"
        )
        updated_text = (
            updated_text[: match.start()]
            + compacted_block
            + updated_text[match.end() :]
        )
        compacted_count += 1
        hooks.add(hook.lower())
        metadata_items.append(
            {
                "original_chars": len(auxiliary_block),
                "kept_chars": len(truncated),
                "mode": "auxiliary_context_block_cap",
            }
        )

    marker = "hook additional context:"
    stripped_updated_text = updated_text.strip()
    if (
        marker in updated_text
        and len(updated_text) > auxiliary_cap
        and stripped_updated_text.startswith("<system-reminder>")
        and stripped_updated_text.endswith("</system-reminder>")
        and stripped_updated_text.count("<system-reminder>") == 1
    ):
        hook_match = re.search(
            r"(SubagentStart|SubAgentStart|SessionStart) hook additional context:",
            updated_text,
        )
        fallback_hook = hook_match.group(1).lower() if hook_match else None
        truncated_text = updated_text[:auxiliary_cap].rstrip()
        updated_text = (
            f"{truncated_text}\n\n"
            f"[Gemini adapter compacted oversized additional context from {len(text)} chars to {len(truncated_text)} chars.]"
        )
        compacted_count += 1
        if fallback_hook:
            hooks.add(fallback_hook)
        metadata_items.append(
            {
                "original_chars": len(text),
                "kept_chars": len(truncated_text),
                "mode": "fallback_text_cap",
            }
        )

    metadata_items.reverse()
    return updated_text, compacted_count, hooks, metadata_items


def _compact_google_adapter_text_part_sequence(
    parts: list[Any],
) -> Tuple[list[Any], int, set[str], list[dict[str, Any]], bool]:
    updated_parts: list[Any] = []
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    changed = False
    index = 0

    while index < len(parts):
        item = parts[index]
        if not (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            updated_parts.append(item)
            index += 1
            continue

        group: list[dict[str, Any]] = []
        while index < len(parts):
            candidate = parts[index]
            if not (
                isinstance(candidate, dict)
                and candidate.get("type") == "text"
                and isinstance(candidate.get("text"), str)
            ):
                break
            group.append(candidate)
            index += 1

        combined_text = "".join(str(part.get("text") or "") for part in group)
        (
            compacted_text,
            child_count,
            child_hooks,
            child_metadata,
        ) = _compact_expanded_claude_persisted_output_text_for_google_adapter(
            combined_text
        )
        if child_count > 0 or len(group) > 1:
            replacement = dict(group[0])
            replacement["text"] = compacted_text
            updated_parts.append(replacement)
            compacted_count += child_count
            hooks.update(child_hooks)
            metadata_items.extend(child_metadata)
            changed = True
        else:
            updated_parts.extend(group)

    return updated_parts, compacted_count, hooks, metadata_items, changed


def _compact_google_adapter_followup_request_contents(
    request_block: dict[str, Any],
) -> dict[str, Any]:
    contents = request_block.get("contents")
    if not isinstance(contents, list) or len(contents) <= 1:
        return {}

    followup_persisted_cap = _get_google_adapter_followup_persisted_output_char_cap()
    followup_auxiliary_cap = _get_google_adapter_followup_auxiliary_context_char_cap()
    original_text_chars = sum(_estimate_google_content_text_chars(item) for item in contents)
    updated_contents: list[Any] = []
    compacted_count = 0
    hooks: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    changed = False

    for content in contents:
        if not isinstance(content, dict):
            updated_contents.append(content)
            continue
        parts = content.get("parts")
        if content.get("role") != "user" or not isinstance(parts, list):
            updated_contents.append(content)
            continue

        updated_parts: list[Any] = []
        part_changed = False
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                (
                    compacted_text,
                    child_count,
                    child_hooks,
                    child_metadata,
                ) = _compact_expanded_claude_persisted_output_text_for_google_adapter(
                    part["text"],
                    persisted_output_char_cap=followup_persisted_cap,
                    auxiliary_context_char_cap=followup_auxiliary_cap,
                )
                compacted_count += child_count
                hooks.update(child_hooks)
                metadata_items.extend(child_metadata)
                if compacted_text != part["text"]:
                    updated_part = dict(part)
                    updated_part["text"] = compacted_text
                    updated_parts.append(updated_part)
                    part_changed = True
                    changed = True
                else:
                    updated_parts.append(part)
            else:
                updated_parts.append(part)

        if part_changed:
            updated_content = dict(content)
            updated_content["parts"] = updated_parts
            updated_contents.append(updated_content)
        else:
            updated_contents.append(content)

    if not changed:
        return {}

    request_block["contents"] = updated_contents
    compacted_text_chars = sum(_estimate_google_content_text_chars(item) for item in updated_contents)
    changes: dict[str, Any] = {
        "followup_persisted_output_compacted_count": compacted_count,
        "followup_persisted_output_text_chars_before": original_text_chars,
        "followup_persisted_output_text_chars_after": compacted_text_chars,
        "followup_persisted_output_char_cap": followup_persisted_cap,
        "followup_auxiliary_context_char_cap": followup_auxiliary_cap,
    }
    if hooks:
        changes["followup_persisted_output_hooks"] = sorted(hooks)
    if metadata_items:
        changes["followup_persisted_output_compaction"] = metadata_items
    return changes


def _compact_google_adapter_persisted_output_value(
    value: Any,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            (
                compacted_text,
                compacted_count,
                compacted_hooks,
                compact_metadata,
            ) = _compact_expanded_claude_persisted_output_text_for_google_adapter(
                value["text"]
            )
            if compacted_count > 0:
                updated_value = dict(value)
                updated_value["text"] = compacted_text
                return (
                    updated_value,
                    compacted_count,
                    compacted_hooks,
                    compact_metadata,
                )
            return value, 0, set(), []

        updated_dict: dict[str, Any] = {}
        compacted_count = 0
        hooks: set[str] = set()
        metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_count, child_hooks, child_metadata = (
                _compact_google_adapter_persisted_output_value(child)
            )
            updated_dict[key] = updated_child
            compacted_count += child_count
            hooks.update(child_hooks)
            metadata_items.extend(child_metadata)
            changed = changed or updated_child is not child
        if changed:
            return updated_dict, compacted_count, hooks, metadata_items
        return value, compacted_count, hooks, metadata_items

    if isinstance(value, list):
        updated_list: list[Any] = value
        compacted_count = 0
        list_hooks: set[str] = set()
        list_metadata_items: list[dict[str, Any]] = []
        changed = False

        if any(
            isinstance(child, dict)
            and child.get("type") == "text"
            and isinstance(child.get("text"), str)
            for child in value
        ):
            (
                updated_list,
                sequence_count,
                sequence_hooks,
                sequence_metadata,
                sequence_changed,
            ) = _compact_google_adapter_text_part_sequence(value)
            compacted_count += sequence_count
            list_hooks.update(sequence_hooks)
            list_metadata_items.extend(sequence_metadata)
            changed = changed or sequence_changed

        recursively_updated_list = []
        for child in updated_list:
            updated_child, child_count, child_hooks, child_metadata = (
                _compact_google_adapter_persisted_output_value(child)
            )
            recursively_updated_list.append(updated_child)
            compacted_count += child_count
            list_hooks.update(child_hooks)
            list_metadata_items.extend(child_metadata)
            changed = changed or updated_child is not child
        if changed:
            return (
                recursively_updated_list,
                compacted_count,
                list_hooks,
                list_metadata_items,
            )
        return value, compacted_count, list_hooks, list_metadata_items

    return value, 0, set(), []


def _compact_google_adapter_persisted_output_in_anthropic_request_body(
    request_body: dict[str, Any],
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    updated_body, compacted_count, hooks, metadata_items = (
        _compact_google_adapter_persisted_output_value(request_body)
    )
    if not isinstance(updated_body, dict):
        return request_body, 0, set(), []
    return updated_body, compacted_count, hooks, metadata_items


def _detect_openai_adapter_claude_context_markers(text: str) -> set[str]:
    markers: set[str] = set()
    for marker_text, marker_name in _OPENAI_ADAPTER_CONTEXT_MARKERS:
        if marker_text in text:
            markers.add(marker_name)
    return markers


def _select_openai_adapter_context_summary_lines(text: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        include_line = (
            line.startswith("SubagentStart hook additional context:")
            or line.startswith("SubAgentStart hook additional context:")
            or line.startswith("#")
            or line.startswith("Contents of ")
            or line.startswith("You are '")
            or line.startswith("Codebase and user instructions")
            or line.startswith("IMPORTANT:")
        )
        if not include_line:
            continue
        if line in seen:
            continue
        selected.append(line)
        seen.add(line)
        if len(selected) >= 10:
            break
    if selected:
        return selected
    return [line.strip() for line in text.splitlines() if line.strip()][:4]


def _build_openai_adapter_compacted_claude_context_block(
    *,
    original_block: str,
    markers: set[str],
    cap: int,
) -> str:
    marker_text = ", ".join(sorted(markers)) or "unknown"
    heading = (
        "[OpenAI adapter compacted Claude Code context block "
        f"from {len(original_block)} chars. Markers: {marker_text}. "
        "The current child task, tool schemas, and latest user instructions remain authoritative.]"
    )
    summary_budget = max(0, cap - len(heading) - 64)
    summary_text = "\n".join(
        _select_openai_adapter_context_summary_lines(original_block)
    ).strip()
    if len(summary_text) > summary_budget:
        summary_text = summary_text[:summary_budget].rstrip()
    if summary_text:
        body = f"{heading}\n{summary_text}"
    else:
        body = heading
    return f"<system-reminder>\n{body}\n</system-reminder>\n"


def _compact_openai_adapter_claude_context_text(
    text: str,
    *,
    cap: Optional[int] = None,
) -> Tuple[str, int, set[str], list[dict[str, Any]]]:
    effective_cap = cap or _get_openai_adapter_claude_context_char_cap()
    updated_text = text
    compacted_count = 0
    combined_markers: set[str] = set()
    metadata_items: list[dict[str, Any]] = []

    matches = list(_OPENAI_ADAPTER_SYSTEM_REMINDER_INLINE_PATTERN.finditer(text))
    for match in reversed(matches):
        reminder_block = match.group(0)
        markers = _detect_openai_adapter_claude_context_markers(reminder_block)
        if not markers or len(reminder_block) <= effective_cap:
            continue

        compacted_block = _build_openai_adapter_compacted_claude_context_block(
            original_block=reminder_block,
            markers=markers,
            cap=effective_cap,
        )
        updated_text = (
            updated_text[: match.start()]
            + compacted_block
            + updated_text[match.end() :]
        )
        compacted_count += 1
        combined_markers.update(markers)
        metadata_items.append(
            {
                "markers": sorted(markers),
                "original_chars": len(reminder_block),
                "kept_chars": len(compacted_block),
                "mode": "system_reminder_context_cap",
            }
        )

    metadata_items.reverse()
    return updated_text, compacted_count, combined_markers, metadata_items


def _compact_openai_adapter_claude_context_value(
    value: Any,
    *,
    cap: Optional[int] = None,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, str):
        return _compact_openai_adapter_claude_context_text(value, cap=cap)

    if isinstance(value, dict):
        updated_dict: dict[str, Any] = {}
        compacted_count = 0
        markers: set[str] = set()
        metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_count, child_markers, child_metadata = (
                _compact_openai_adapter_claude_context_value(child, cap=cap)
            )
            updated_dict[key] = updated_child
            compacted_count += child_count
            markers.update(child_markers)
            metadata_items.extend(child_metadata)
            changed = changed or updated_child != child
        if changed:
            return updated_dict, compacted_count, markers, metadata_items
        return value, compacted_count, markers, metadata_items

    if isinstance(value, list):
        updated_list: list[Any] = []
        compacted_count = 0
        list_markers: set[str] = set()
        list_metadata_items: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_count, child_markers, child_metadata = (
                _compact_openai_adapter_claude_context_value(child, cap=cap)
            )
            updated_list.append(updated_child)
            compacted_count += child_count
            list_markers.update(child_markers)
            list_metadata_items.extend(child_metadata)
            changed = changed or updated_child != child
        if changed:
            return updated_list, compacted_count, list_markers, list_metadata_items
        return value, compacted_count, list_markers, list_metadata_items

    return value, 0, set(), []


def _add_openai_adapter_claude_context_compaction_logging_metadata(
    request_body: dict[str, Any],
    *,
    compacted_count: int,
    markers: set[str],
    metadata_items: list[dict[str, Any]],
    span_started_at: datetime,
    tag_prefix: str = "openai-adapter",
    metadata_prefix: str = "openai_adapter",
    span_name: str = "openai_adapter.claude_context_compaction",
) -> dict[str, Any]:
    original_chars = sum(
        item.get("original_chars", 0)
        for item in metadata_items
        if isinstance(item.get("original_chars"), int)
    )
    compacted_chars = sum(
        item.get("kept_chars", 0)
        for item in metadata_items
        if isinstance(item.get("kept_chars"), int)
    )
    sorted_markers = sorted(markers)
    tags = [
        f"{tag_prefix}-claude-context-compacted",
        *[
            f"{tag_prefix}-claude-context:{marker}"
            for marker in sorted_markers
        ],
    ]
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags,
        extra_fields={
            f"{metadata_prefix}_claude_context_compacted": True,
            f"{metadata_prefix}_claude_context_compacted_count": compacted_count,
            f"{metadata_prefix}_claude_context_markers": sorted_markers,
            f"{metadata_prefix}_claude_context_original_chars": original_chars,
            f"{metadata_prefix}_claude_context_compacted_chars": compacted_chars,
            f"{metadata_prefix}_claude_context_saved_chars": max(
                0, original_chars - compacted_chars
            ),
            f"{metadata_prefix}_claude_context_compaction_events": metadata_items,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata={
                        "compacted_count": compacted_count,
                        "markers": sorted_markers,
                        "original_chars": original_chars,
                        "compacted_chars": compacted_chars,
                        "saved_chars": max(0, original_chars - compacted_chars),
                    },
                    start_time=span_started_at,
                    end_time=datetime.now(timezone.utc),
                )
            ],
        },
    )


def _compact_openai_adapter_claude_context_in_anthropic_request_body(
    request_body: dict[str, Any],
    *,
    tag_prefix: str = "openai-adapter",
    metadata_prefix: str = "openai_adapter",
    span_name: str = "openai_adapter.claude_context_compaction",
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    compacted_count = 0
    markers: set[str] = set()
    metadata_items: list[dict[str, Any]] = []
    changed = False

    for top_level_key in ("system", "messages"):
        if top_level_key not in request_body:
            continue
        updated_value, value_count, value_markers, value_metadata = (
            _compact_openai_adapter_claude_context_value(request_body[top_level_key])
        )
        if value_count > 0:
            updated_body[top_level_key] = updated_value
            compacted_count += value_count
            markers.update(value_markers)
            metadata_items.extend(value_metadata)
            changed = True

    if not changed:
        return request_body, 0, set(), []

    updated_body = _add_openai_adapter_claude_context_compaction_logging_metadata(
        updated_body,
        compacted_count=compacted_count,
        markers=markers,
        metadata_items=metadata_items,
        span_started_at=span_started_at,
        tag_prefix=tag_prefix,
        metadata_prefix=metadata_prefix,
        span_name=span_name,
    )
    return updated_body, compacted_count, markers, metadata_items


def _expand_claude_persisted_output_text(
    text: str,
) -> Tuple[str, bool, Optional[str], Optional[dict[str, Any]]]:
    if not _is_claude_persisted_output_expansion_enabled():
        return text, False, None, None

    match = _CLAUDE_PERSISTED_OUTPUT_PATTERN.match(text)
    if match is None:
        return text, False, None, None

    resolved_path = _resolve_claude_persisted_output_path(match.group("path"))
    if resolved_path is None:
        return text, False, None, None

    try:
        file_text = resolved_path.read_text(encoding="utf-8", errors="replace").rstrip(
            "\n"
        )
    except Exception:
        return text, False, None, None

    hook = match.group("hook")
    expanded = (
        "<system-reminder>\n"
        f"{hook} hook additional context: <persisted-output>\n"
        f"{file_text}\n"
        "</persisted-output>\n"
        "</system-reminder>\n"
    )
    return (
        expanded,
        True,
        hook.lower(),
        _build_claude_persisted_output_source_metadata(
            resolved_path=resolved_path,
            file_text=file_text,
        ),
    )


def _expand_claude_persisted_output_value(
    value: Any,
) -> Tuple[Any, int, set[str], list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            (
                expanded_text,
                was_expanded,
                hook_name,
                source_metadata,
            ) = _expand_claude_persisted_output_text(value["text"])
            if was_expanded:
                updated_value = dict(value)
                updated_value["text"] = expanded_text
                return (
                    updated_value,
                    1,
                    {hook_name} if hook_name else set(),
                    [source_metadata] if source_metadata else [],
                )
            return value, 0, set(), []

        updated_dict: dict[str, Any] = {}
        expanded_count = 0
        hooks: set[str] = set()
        source_metadata_items: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            (
                updated_child,
                child_expanded_count,
                child_hooks,
                child_source_metadata_items,
            ) = _expand_claude_persisted_output_value(child)
            updated_dict[key] = updated_child
            expanded_count += child_expanded_count
            hooks.update(child_hooks)
            source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                changed = True
        return (
            updated_dict if changed else value,
            expanded_count,
            hooks,
            source_metadata_items,
        )

    if isinstance(value, list):
        updated_list = []
        expanded_count = 0
        list_hooks: set[str] = set()
        list_source_metadata_items: list[dict[str, Any]] = []
        changed = False
        for child in value:
            (
                updated_child,
                child_expanded_count,
                child_hooks,
                child_source_metadata_items,
            ) = _expand_claude_persisted_output_value(child)
            updated_list.append(updated_child)
            expanded_count += child_expanded_count
            list_hooks.update(child_hooks)
            list_source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                changed = True
        return (
            updated_list if changed else value,
            expanded_count,
            list_hooks,
            list_source_metadata_items,
        )

    return value, 0, set(), []


def _merge_litellm_metadata(
    request_body: dict[str, Any],
    *,
    tags_to_add: Optional[list[str]] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    existing_tags = litellm_metadata.get("tags") or []
    if not isinstance(existing_tags, list):
        existing_tags = []

    merged_tags = list(existing_tags)
    for tag in tags_to_add or []:
        if tag not in merged_tags:
            merged_tags.append(tag)

    litellm_metadata["tags"] = merged_tags
    if extra_fields:
        existing_spans = litellm_metadata.get("langfuse_spans")
        incoming_spans = extra_fields.get("langfuse_spans")
        if isinstance(existing_spans, list) and isinstance(incoming_spans, list):
            merged_extra_fields = dict(extra_fields)
            merged_extra_fields["langfuse_spans"] = list(existing_spans) + list(
                incoming_spans
            )
            litellm_metadata.update(merged_extra_fields)
        else:
            litellm_metadata.update(extra_fields)

    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body


def _format_langfuse_span_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_langfuse_span_descriptor(
    *,
    name: str,
    metadata: Optional[dict[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"name": name}
    if input_data is not None:
        descriptor["input"] = input_data
    if output_data is not None:
        descriptor["output"] = output_data
    if metadata:
        descriptor["metadata"] = metadata
    if start_time is not None:
        descriptor["start_time"] = _format_langfuse_span_timestamp(start_time)
    if end_time is not None:
        descriptor["end_time"] = _format_langfuse_span_timestamp(end_time)
    return descriptor


def _get_nested_str_value(source: Any, path: tuple[str, ...]) -> Optional[str]:
    current = source
    for key in path:
        if isinstance(current, str):
            stripped_current = current.strip()
            if not stripped_current:
                return None
            try:
                current = json.loads(stripped_current)
            except json.JSONDecodeError:
                return None
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, str) and current.strip():
        return current.strip()
    return None


def _extract_passthrough_session_id(
    request: Request, request_body: Optional[dict[str, Any]] = None
) -> Optional[str]:
    if isinstance(request_body, dict):
        for path in (
            ("session_id",),
            ("request", "session_id"),
            ("metadata", "session_id"),
            ("metadata", "user_id", "session_id"),
        ):
            value = _get_nested_str_value(request_body, path)
            if value:
                return value

    headers = _safe_get_request_headers(request)
    for header_name in _PASSTHROUGH_SESSION_ID_HEADER_NAMES:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_passthrough_repository(value: str) -> Optional[str]:
    cleaned = value.strip().strip("`'\"")
    if not cleaned:
        return None
    if cleaned.startswith("git@") and ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    elif "://" in cleaned:
        parsed = urlparse(cleaned)
        path = parsed.path.strip("/")
        netloc = parsed.netloc.split("@", 1)[-1]
        if parsed.scheme == "file" and path:
            cleaned = path.rstrip("/").rsplit("/", 1)[-1]
        elif netloc.lower().endswith("github.com") and path:
            cleaned = path
        else:
            cleaned = f"{netloc}/{path}".strip("/")
    elif cleaned.startswith("/"):
        cleaned = cleaned.rstrip("/").rsplit("/", 1)[-1]
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip("/")
    if not cleaned:
        return None

    normalized = cleaned.lower()
    if normalized.endswith(" (memory)"):
        normalized = normalized[: -len(" (memory)")]
    if (
        normalized in _PASSTHROUGH_REPOSITORY_PLACEHOLDER_VALUES
        or _PASSTHROUGH_REPOSITORY_TRANSCRIPT_ARTIFACT_RE.fullmatch(normalized)
        or normalized in _PASSTHROUGH_REPOSITORY_AGENT_ROLE_VALUES
        or _PASSTHROUGH_REPOSITORY_AGENT_ID_RE.fullmatch(normalized)
        or _PASSTHROUGH_REPOSITORY_WAVE_AGENT_RE.fullmatch(normalized)
    ):
        return None

    return cleaned


def _extract_passthrough_repository_from_text(value: str) -> Optional[str]:
    for pattern in _PASSTHROUGH_REPOSITORY_TEXT_PATTERNS:
        matches = list(pattern.finditer(value))
        for match in reversed(matches):
            repository = _normalize_passthrough_repository(match.group("path"))
            if repository:
                return repository
    return None


def _extract_passthrough_repository_from_body_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _extract_passthrough_repository_from_text(value)
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PASSTHROUGH_REPOSITORY_BODY_KEYS and isinstance(child, str):
                repository = _normalize_passthrough_repository(child)
                if repository:
                    return repository
            repository = _extract_passthrough_repository_from_body_text(child)
            if repository:
                return repository
    if isinstance(value, list):
        for child in reversed(value):
            repository = _extract_passthrough_repository_from_body_text(child)
            if repository:
                return repository
    return None


def _extract_passthrough_repository(
    request: Request, request_body: Optional[dict[str, Any]] = None
) -> Optional[str]:
    if isinstance(request_body, dict):
        for path in (
            ("repository",),
            ("repo",),
            ("workspace_root",),
            ("workspaceRoot",),
            ("project_root",),
            ("projectRoot",),
            ("root_path",),
            ("rootPath",),
            ("working_directory",),
            ("workingDirectory",),
            ("cwd_path",),
            ("cwdPath",),
            ("cwd_uri",),
            ("cwdUri",),
            ("metadata", "repository"),
            ("metadata", "repo"),
            ("metadata", "workspace_root"),
            ("metadata", "workspaceRoot"),
            ("litellm_metadata", "repository"),
            ("request", "repository"),
            ("request", "workspace_root"),
            ("request", "workspaceRoot"),
            ("request", "project_root"),
            ("request", "projectRoot"),
            ("request", "root_path"),
            ("request", "rootPath"),
            ("request", "working_directory"),
            ("request", "workingDirectory"),
            ("request", "cwd_path"),
            ("request", "cwdPath"),
            ("request", "cwd_uri"),
            ("request", "cwdUri"),
            ("request", "metadata", "repository"),
            ("request", "metadata", "workspace_root"),
            ("request", "metadata", "workspaceRoot"),
        ):
            value = _get_nested_str_value(request_body, path)
            if value:
                return _normalize_passthrough_repository(value)
        repository = _extract_passthrough_repository_from_body_text(request_body)
        if repository:
            return repository

    headers = _safe_get_request_headers(request)
    for header_name in _PASSTHROUGH_REPOSITORY_HEADER_NAMES:
        value = headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return _normalize_passthrough_repository(value)
    return None


def _get_passthrough_trace_environment() -> Optional[str]:
    for env_var in (
        "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
        "LANGFUSE_TRACING_ENVIRONMENT",
    ):
        value = os.getenv(env_var)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _add_passthrough_trace_context_metadata(
    request_body: dict[str, Any],
    *,
    session_id: Optional[str],
    trace_environment: Optional[str],
    repository: Optional[str] = None,
) -> dict[str, Any]:
    updated_body = dict(request_body)
    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    changed = False

    if session_id and not litellm_metadata.get("session_id"):
        litellm_metadata["session_id"] = session_id
        changed = True

    if trace_environment:
        existing_trace_environment = litellm_metadata.get("trace_environment")
        if existing_trace_environment != trace_environment:
            if existing_trace_environment and not litellm_metadata.get(
                "source_trace_environment"
            ):
                litellm_metadata["source_trace_environment"] = existing_trace_environment
            litellm_metadata["trace_environment"] = trace_environment
            changed = True

    if repository and not litellm_metadata.get("repository"):
        litellm_metadata["repository"] = repository
        changed = True

    if not changed:
        return request_body

    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body


_AAWM_TOOL_DEFINITION_CAPTURE_VERSION = "v1"
_AAWM_TOOL_DEFINITION_MAX_TOOLS = 64
_AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS = 128
_AAWM_TOOL_DEFINITION_MAX_STRING_CHARS = 4096
_AAWM_TOOL_DEFINITION_MAX_DEPTH = 20
_AAWM_TOOL_DEFINITION_REDACTED = "redacted-by-litellm"
_AAWM_TOOL_DEFINITION_SECRET_KEY_RE = re.compile(
    r"("
    r"authorization|api[_-]?key|bearer|credential|password|secret|^token$"
    r"|access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token"
    r")",
    re.IGNORECASE,
)
_AAWM_TOOL_DEFINITION_SECRET_VALUE_RES = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{6,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9._-]{8,}\b", re.IGNORECASE),
)


def _truncate_tool_definition_string(value: str) -> tuple[str, bool]:
    if len(value) <= _AAWM_TOOL_DEFINITION_MAX_STRING_CHARS:
        return value, False
    return value[:_AAWM_TOOL_DEFINITION_MAX_STRING_CHARS], True


def _redact_tool_definition_string(value: str) -> str:
    redacted = value
    for pattern in _AAWM_TOOL_DEFINITION_SECRET_VALUE_RES:
        redacted = pattern.sub(_AAWM_TOOL_DEFINITION_REDACTED, redacted)
    return redacted


def _sanitize_tool_definition_value(
    value: Any,
    *,
    depth: int = 0,
    key_hint: Optional[str] = None,
) -> tuple[Any, bool]:
    if depth > _AAWM_TOOL_DEFINITION_MAX_DEPTH:
        return {"__truncated__": "max_depth"}, True
    if key_hint and _AAWM_TOOL_DEFINITION_SECRET_KEY_RE.search(key_hint):
        return _AAWM_TOOL_DEFINITION_REDACTED, False
    if isinstance(value, str):
        return _truncate_tool_definition_string(_redact_tool_definition_string(value))
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if isinstance(value, list):
        truncated = len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS
        sanitized_items: list[Any] = []
        for item in value[:_AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS]:
            sanitized_item, item_truncated = _sanitize_tool_definition_value(
                item,
                depth=depth + 1,
                key_hint=key_hint,
            )
            truncated = truncated or item_truncated
            sanitized_items.append(sanitized_item)
        if len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS:
            sanitized_items.append(
                {"__truncated_items__": len(value) - _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS}
            )
        return sanitized_items, truncated
    if isinstance(value, dict):
        truncated = len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS
        sanitized_dict: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS:
                break
            sanitized_item, item_truncated = _sanitize_tool_definition_value(
                item,
                depth=depth + 1,
                key_hint=str(key),
            )
            truncated = truncated or item_truncated
            sanitized_dict[str(key)] = sanitized_item
        if len(value) > _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS:
            sanitized_dict["__truncated_keys__"] = (
                len(value) - _AAWM_TOOL_DEFINITION_MAX_CONTAINER_ITEMS
            )
        return sanitized_dict, truncated
    return str(value), False


def _tool_definition_name(tool: dict[str, Any]) -> Optional[str]:
    function_definition = tool.get("function")
    for candidate in (
        tool.get("name"),
        function_definition.get("name")
        if isinstance(function_definition, dict)
        else None,
        tool.get("tool_name"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _tool_definition_description(tool: dict[str, Any]) -> Optional[str]:
    function_definition = tool.get("function")
    for candidate in (
        tool.get("description"),
        function_definition.get("description")
        if isinstance(function_definition, dict)
        else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _tool_definition_parameters(tool: dict[str, Any]) -> Any:
    function_definition = tool.get("function")
    if isinstance(function_definition, dict) and "parameters" in function_definition:
        return function_definition.get("parameters")
    for key in ("parameters", "input_schema", "schema", "json_schema"):
        if key in tool:
            return tool.get(key)
    return None


def _build_tool_definition_snapshot_entry(
    *,
    source: str,
    index: int,
    tool: Any,
) -> tuple[Optional[dict[str, Any]], bool]:
    if not isinstance(tool, dict):
        return None, False

    sanitized_definition, definition_truncated = _sanitize_tool_definition_value(tool)
    sanitized_parameters, parameters_truncated = _sanitize_tool_definition_value(
        _tool_definition_parameters(tool)
    )
    description, description_truncated = _truncate_tool_definition_string(
        _redact_tool_definition_string(_tool_definition_description(tool) or "")
    )
    entry = {
        "source": source,
        "index": index,
        "type": tool.get("type"),
        "name": _tool_definition_name(tool),
        "description": description,
        "parameters": sanitized_parameters,
        "definition": sanitized_definition,
    }
    return entry, bool(
        definition_truncated or parameters_truncated or description_truncated
    )


def _tool_definition_snapshot_hash(snapshot: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_passthrough_tool_definition_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tool_sources: tuple[tuple[str, Any], ...] = (
        ("tools", request_body.get("tools")),
        ("functions", request_body.get("functions")),
    )
    snapshot: list[dict[str, Any]] = []
    available_count = 0
    truncated = False
    source_names: list[str] = []

    for source, tools in tool_sources:
        if not isinstance(tools, list):
            continue
        source_names.append(source)
        available_count += len(tools)
        for index, tool in enumerate(tools):
            if len(snapshot) >= _AAWM_TOOL_DEFINITION_MAX_TOOLS:
                truncated = True
                break
            entry, entry_truncated = _build_tool_definition_snapshot_entry(
                source=source,
                index=index,
                tool=tool,
            )
            if entry is None:
                continue
            snapshot.append(entry)
            truncated = truncated or entry_truncated

    if not snapshot:
        return {}

    names = [
        entry["name"]
        for entry in snapshot
        if isinstance(entry.get("name"), str) and entry.get("name")
    ]
    tool_types = [
        entry["type"]
        for entry in snapshot
        if isinstance(entry.get("type"), str) and entry.get("type")
    ]
    return {
        "aawm_tool_definition_capture_version": (
            _AAWM_TOOL_DEFINITION_CAPTURE_VERSION
        ),
        "aawm_tool_definition_capture_source": "passthrough_request_body",
        "aawm_tool_definition_count": available_count,
        "aawm_tool_definition_captured_count": len(snapshot),
        "aawm_tool_definition_sources": source_names,
        "aawm_tool_definition_names": names,
        "aawm_tool_definition_types": tool_types,
        "aawm_tool_definition_snapshot": snapshot,
        "aawm_tool_definition_snapshot_hash": (
            _tool_definition_snapshot_hash(snapshot)
        ),
        "aawm_tool_definition_snapshot_truncated": truncated
        or available_count > len(snapshot),
        "aawm_tool_definition_snapshot_storage": (
            "session_history_tool_definition_snapshots"
        ),
        "aawm_tool_definition_snapshot_storage_key": (
            "session_id,aawm_tool_definition_snapshot_hash"
        ),
    }


def _add_passthrough_tool_definition_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tool_definition_metadata = _build_passthrough_tool_definition_metadata(request_body)
    if not tool_definition_metadata:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        extra_fields=tool_definition_metadata,
    )


def _prepare_request_body_for_passthrough_observability(
    request: Request, request_body: dict[str, Any]
) -> dict[str, Any]:
    session_id = _extract_passthrough_session_id(
        request=request, request_body=request_body
    )
    repository = _extract_passthrough_repository(
        request=request, request_body=request_body
    )
    trace_environment = _get_passthrough_trace_environment()
    prepared_body = _add_passthrough_trace_context_metadata(
        request_body,
        session_id=session_id,
        trace_environment=trace_environment,
        repository=repository,
    )
    return _add_passthrough_tool_definition_metadata(prepared_body)


def _add_route_family_logging_metadata(
    request_body: dict[str, Any], route_family: str
) -> dict[str, Any]:
    normalized_route_family = _normalize_low_cardinality_tag_value(route_family)
    if not normalized_route_family:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=[f"route:{normalized_route_family}"],
        extra_fields={"passthrough_route_family": normalized_route_family},
    )


def _append_codex_auto_agent_prevention_guidance_to_instructions(
    instructions: Optional[str],
) -> str:
    existing_instructions = instructions.strip() if isinstance(instructions, str) else ""
    if _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT in existing_instructions:
        return existing_instructions
    if not existing_instructions:
        return _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT
    return (
        f"{existing_instructions}\n\n"
        f"{_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT}"
    )


def _is_aawm_read_agent_alias_model(alias_model: Any) -> bool:
    if not isinstance(alias_model, str):
        return False
    return alias_model in {
        _CODEX_AAWM_READ_ALIAS,
        _ANTHROPIC_AAWM_READ_ALIAS,
    }


def _append_aawm_read_agent_guidance_to_text(value: Optional[str]) -> str:
    existing_value = value.strip() if isinstance(value, str) else ""
    if _AAWM_READ_AGENT_GUIDANCE_PROMPT in existing_value:
        return existing_value
    if not existing_value:
        return _AAWM_READ_AGENT_GUIDANCE_PROMPT
    return f"{existing_value}\n\n{_AAWM_READ_AGENT_GUIDANCE_PROMPT}"


def _append_aawm_read_agent_guidance_to_anthropic_system(
    system_value: Any,
) -> tuple[Any, bool, int]:
    if system_value is None or isinstance(system_value, str):
        original_chars = len(system_value) if isinstance(system_value, str) else 0
        updated_system = _append_aawm_read_agent_guidance_to_text(system_value)
        return updated_system, updated_system != system_value, original_chars

    if not isinstance(system_value, list):
        return system_value, False, 0

    original_chars = 0
    for item in system_value:
        text_value: Optional[str] = None
        if isinstance(item, str):
            text_value = item
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text_value = item["text"]
        if text_value is None:
            continue
        original_chars += len(text_value)
        if _AAWM_READ_AGENT_GUIDANCE_PROMPT in text_value:
            return system_value, False, original_chars

    return (
        [
            *system_value,
            {"type": "text", "text": _AAWM_READ_AGENT_GUIDANCE_PROMPT},
        ],
        True,
        original_chars,
    )


def _apply_aawm_read_agent_guidance_to_request_body(
    request_body: dict[str, Any],
    *,
    alias_model: Any,
    target_field: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_aawm_read_agent_alias_model(alias_model):
        return request_body, {}

    updated_body = dict(request_body)
    original_chars = 0
    if target_field == "instructions":
        existing_instructions = request_body.get("instructions")
        if existing_instructions is not None and not isinstance(
            existing_instructions, str
        ):
            return request_body, {}
        updated_value = _append_aawm_read_agent_guidance_to_text(
            existing_instructions
        )
        if updated_value == existing_instructions:
            return request_body, {}
        updated_body["instructions"] = updated_value
        original_chars = (
            len(existing_instructions) if isinstance(existing_instructions, str) else 0
        )
    elif target_field == "system":
        updated_system, changed, original_chars = (
            _append_aawm_read_agent_guidance_to_anthropic_system(
                request_body.get("system")
            )
        )
        if not changed:
            return request_body, {}
        updated_body["system"] = updated_system
    else:
        return request_body, {}

    guidance_metadata = {
        "aawm_read_agent_guidance_policy_name": (
            _AAWM_READ_AGENT_GUIDANCE_POLICY_NAME
        ),
        "aawm_read_agent_guidance_policy_version": (
            _AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION
        ),
        "aawm_read_agent_guidance_applied": True,
        "aawm_read_agent_guidance_alias": alias_model,
        "aawm_read_agent_guidance_target_field": target_field,
        "aawm_read_agent_guidance_original_chars": original_chars,
        "aawm_read_agent_guidance_prompt_chars": len(
            _AAWM_READ_AGENT_GUIDANCE_PROMPT
        ),
    }
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "aawm-read-agent-guidance",
            (
                "aawm-read-agent-guidance:"
                f"{_AAWM_READ_AGENT_GUIDANCE_POLICY_VERSION}"
            ),
            f"aawm-read-agent-guidance-alias:{alias_model}",
        ],
        extra_fields={
            **guidance_metadata,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="aawm.read_agent_guidance",
                    metadata=guidance_metadata,
                )
            ],
        },
    )
    return updated_body, guidance_metadata


def _apply_codex_auto_agent_prevention_guidance_to_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing_instructions = request_body.get("instructions")
    if existing_instructions is not None and not isinstance(existing_instructions, str):
        return request_body, {}

    updated_instructions = (
        _append_codex_auto_agent_prevention_guidance_to_instructions(
            existing_instructions
        )
    )
    if updated_instructions == existing_instructions:
        return request_body, {}

    updated_body = dict(request_body)
    updated_body["instructions"] = updated_instructions
    original_chars = (
        len(existing_instructions) if isinstance(existing_instructions, str) else 0
    )
    guidance_metadata = {
        "codex_auto_agent_prevention_guidance_policy_name": (
            _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME
        ),
        "codex_auto_agent_prevention_guidance_policy_version": (
            _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION
        ),
        "codex_auto_agent_prevention_guidance_applied": True,
        "codex_auto_agent_prevention_guidance_original_instruction_chars": (
            original_chars
        ),
        "codex_auto_agent_prevention_guidance_prompt_chars": len(
            _CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT
        ),
    }
    updated_body = _merge_litellm_metadata(
        updated_body,
        tags_to_add=[
            "codex-auto-agent-prevention-guidance",
            (
                "codex-auto-agent-prevention-guidance:"
                f"{_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION}"
            ),
        ],
        extra_fields={
            **guidance_metadata,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="codex.auto_agent_prevention_guidance",
                    metadata=guidance_metadata,
                )
            ],
        },
    )
    return updated_body, guidance_metadata


def _normalize_low_cardinality_tag_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned or None
    return None


def _dedupe_sorted_str_list(values: list[str]) -> list[str]:
    return sorted({value for value in values if isinstance(value, str) and value})


def _extract_claude_request_breakout_fields(
    request_body: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    tags_to_add: list[str] = []
    extra_fields: dict[str, Any] = {}

    thinking = request_body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = _normalize_low_cardinality_tag_value(thinking.get("type"))
        if thinking_type:
            tags_to_add.extend(
                [f"claude-thinking-type:{thinking_type}", f"thinking-type:{thinking_type}"]
            )
            extra_fields["claude_thinking_type"] = thinking_type

    output_config = request_body.get("output_config")
    if isinstance(output_config, dict):
        effort = _normalize_low_cardinality_tag_value(output_config.get("effort"))
        if effort:
            tags_to_add.extend([f"claude-effort:{effort}", f"effort:{effort}"])
            extra_fields["claude_effort"] = effort

    context_management = request_body.get("context_management")
    context_edits = []
    if isinstance(context_management, dict):
        edits = context_management.get("edits")
        if isinstance(edits, list):
            context_edits = [edit for edit in edits if isinstance(edit, dict)]

    edit_types: list[str] = []
    keep_values: list[str] = []
    for edit in context_edits:
        edit_type = _normalize_low_cardinality_tag_value(edit.get("type"))
        if edit_type:
            edit_types.append(edit_type)
            tags_to_add.append(f"claude-context-edit:{edit_type}")
        keep_value = _normalize_low_cardinality_tag_value(edit.get("keep"))
        if keep_value:
            keep_values.append(keep_value)
            tags_to_add.append(f"claude-context-keep:{keep_value}")

    if context_edits:
        extra_fields["claude_context_edit_count"] = len(context_edits)
    if edit_types:
        extra_fields["claude_context_edit_types"] = _dedupe_sorted_str_list(edit_types)
    if keep_values:
        extra_fields["claude_context_keep_values"] = _dedupe_sorted_str_list(
            keep_values
        )

    account_uuid = _get_nested_str_value(request_body, ("metadata", "user_id", "account_uuid"))
    if account_uuid:
        extra_fields["claude_account_uuid"] = account_uuid
    device_id = _get_nested_str_value(request_body, ("metadata", "user_id", "device_id"))
    if device_id:
        extra_fields["claude_device_id"] = device_id

    return tags_to_add, extra_fields


def _add_claude_request_breakout_logging_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tags_to_add, extra_fields = _extract_claude_request_breakout_fields(request_body)
    if not tags_to_add and not extra_fields:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _is_anthropic_web_search_tool(value: dict[str, Any]) -> bool:
    tool_type = value.get("type")
    tool_name = value.get("name")
    return (
        isinstance(tool_type, str)
        and tool_type.startswith("web_search")
    ) or tool_name == "web_search"


def _sanitize_anthropic_web_search_empty_domain_lists_in_value(
    value: Any,
) -> tuple[Any, int]:
    if isinstance(value, dict):
        updated_dict: dict[str, Any] = {}
        changed = False
        sanitized_count = 0
        is_web_search_tool = _is_anthropic_web_search_tool(value)
        for key, child in value.items():
            if (
                is_web_search_tool
                and key in {"allowed_domains", "blocked_domains"}
                and child == []
            ):
                updated_dict[key] = None
                changed = True
                sanitized_count += 1
                continue
            updated_child, child_count = (
                _sanitize_anthropic_web_search_empty_domain_lists_in_value(child)
            )
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
            updated_child, child_count = (
                _sanitize_anthropic_web_search_empty_domain_lists_in_value(child)
            )
            updated_list.append(updated_child)
            sanitized_count += child_count
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), sanitized_count

    return value, 0


def _sanitize_anthropic_web_search_empty_domain_lists(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    updated_value, sanitized_count = (
        _sanitize_anthropic_web_search_empty_domain_lists_in_value(request_body)
    )
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


def _extract_gemini_request_breakout_fields(
    request_body: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    tags_to_add: list[str] = []
    extra_fields: dict[str, Any] = {}

    generation_config = request_body.get("generationConfig")
    if not isinstance(generation_config, dict):
        request_block = request_body.get("request")
        if isinstance(request_block, dict):
            nested_generation_config = request_block.get("generationConfig")
            if isinstance(nested_generation_config, dict):
                generation_config = nested_generation_config

    if isinstance(generation_config, dict):
        thinking_config = generation_config.get("thinkingConfig")
        if isinstance(thinking_config, dict):
            tags_to_add.append("gemini-thinking-config-present")
            extra_fields["gemini_thinking_config_present"] = True

            include_thoughts = thinking_config.get("includeThoughts")
            if isinstance(include_thoughts, bool):
                include_thoughts_tag = "true" if include_thoughts else "false"
                tags_to_add.extend(
                    [
                        f"gemini-include-thoughts:{include_thoughts_tag}",
                        f"include-thoughts:{include_thoughts_tag}",
                    ]
                )
                extra_fields["gemini_include_thoughts"] = include_thoughts

            thinking_level = thinking_config.get("thinkingLevel")
            normalized_thinking_level = _normalize_low_cardinality_tag_value(
                thinking_level
            )
            if normalized_thinking_level:
                tags_to_add.extend(
                    [
                        f"gemini-thinking-level:{normalized_thinking_level}",
                        f"thinking-level:{normalized_thinking_level}",
                    ]
                )
                extra_fields["gemini_thinking_level"] = normalized_thinking_level

            thinking_budget = thinking_config.get("thinkingBudget")
            if isinstance(thinking_budget, (int, float)) and thinking_budget > 0:
                tags_to_add.append("gemini-thinking-budget-configured")
                extra_fields["gemini_thinking_budget"] = thinking_budget

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        request_block = request_body.get("request")
        if isinstance(request_block, dict):
            nested_tools = request_block.get("tools")
            if isinstance(nested_tools, list):
                tools = nested_tools

    if isinstance(tools, list) and tools:
        tags_to_add.append("gemini-tools-present")
        extra_fields["gemini_tools_present"] = True
        extra_fields["gemini_tool_count"] = len(tools)

    for key in ("user_prompt_id", "project"):
        value = request_body.get(key)
        if not value and isinstance(request_body.get("request"), dict):
            value = request_body["request"].get(key)
        if isinstance(value, str) and value.strip():
            extra_fields[f"gemini_{key}"] = value.strip()

    return tags_to_add, extra_fields


def _add_gemini_request_breakout_logging_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tags_to_add, extra_fields = _extract_gemini_request_breakout_fields(request_body)
    if not tags_to_add and not extra_fields:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _patch_codex_spawn_agent_description_text(description: str) -> tuple[str, int]:
    updated_description = description
    replacement_count = 0
    for pattern in _CODEX_SPAWN_AGENT_RESTRICTIVE_DESCRIPTION_PATTERNS:
        updated_description, count = pattern.subn(
            _CODEX_SPAWN_AGENT_FANOUT_POLICY,
            updated_description,
            count=1,
        )
        replacement_count += count
    return updated_description, replacement_count


def _get_codex_core_tool_guidance(tool_name: Optional[str]) -> Optional[str]:
    normalized_tool_name = _normalize_low_cardinality_tag_value(tool_name)
    if not normalized_tool_name:
        return None
    return _CODEX_CORE_TOOL_GUIDANCE_BY_NAME.get(normalized_tool_name)


def _append_codex_core_tool_guidance_to_description(
    description: Any,
    *,
    guidance: str,
) -> tuple[str, bool]:
    existing_description = description if isinstance(description, str) else ""
    if guidance in existing_description:
        return existing_description, False
    if not existing_description.strip():
        return guidance, True
    return f"{existing_description.rstrip()}\n\n{guidance}", True


def _patch_codex_spawn_agent_payload_parameters(
    parameters: Any,
) -> tuple[Any, list[str]]:
    if parameters is None:
        updated_parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
        }
    elif isinstance(parameters, dict):
        updated_parameters = copy.deepcopy(parameters)
        if "type" not in updated_parameters:
            updated_parameters["type"] = "object"
    else:
        return parameters, []

    properties = updated_parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    else:
        properties = dict(properties)

    added_fields: list[str] = []
    for field_name in _CODEX_SPAWN_AGENT_PAYLOAD_FIELD_ORDER:
        if field_name in properties:
            continue
        properties[field_name] = copy.deepcopy(
            _CODEX_SPAWN_AGENT_PAYLOAD_FIELD_SCHEMAS[field_name]
        )
        added_fields.append(field_name)

    if not added_fields:
        return parameters, []

    updated_parameters["properties"] = properties
    return updated_parameters, added_fields


def _get_openai_tool_name(tool: dict[str, Any]) -> Optional[str]:
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = tool.get("function")
    if isinstance(function, dict):
        function_name = function.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name.strip()
    return None


def _get_openai_tool_type(tool: dict[str, Any]) -> Optional[str]:
    tool_type = tool.get("type")
    if isinstance(tool_type, str) and tool_type.strip():
        return tool_type.strip()
    return None


@lru_cache(maxsize=1)
def _load_bundled_model_cost_map_for_codex_tool_policy() -> dict[str, Any]:
    try:
        content = files("litellm").joinpath(
            "bundled_model_prices_and_context_window_fallback.json"
        ).read_text(encoding="utf-8")
        loaded = json.loads(content)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return {}
    return {}


def _get_codex_tool_policy_model_cost_candidates(model: Any) -> list[str]:
    if not isinstance(model, str) or not model.strip():
        return []

    model_name = model.strip()
    split_model_name = model_name.split("/", 1)[1] if "/" in model_name else model_name
    candidates = [
        model_name,
        model_name.lower(),
        split_model_name,
        split_model_name.lower(),
        f"chatgpt/{split_model_name}",
        f"chatgpt/{split_model_name.lower()}",
        f"openai/{split_model_name}",
        f"openai/{split_model_name.lower()}",
    ]
    grok_native_model = normalize_grok_native_oauth_model(model_name)
    if grok_native_model is not None:
        candidates.extend(
            [
                f"xai/{grok_native_model}",
                f"xai/{grok_native_model.lower()}",
            ]
        )
    if is_oa_xai_model(model_name):
        try:
            xai_oauth_upstream_model = resolve_oa_xai_upstream_model(model_name)
            candidates.extend(
                [
                    xai_oauth_upstream_model,
                    xai_oauth_upstream_model.lower(),
                ]
            )
        except Exception:
            pass

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _get_unsupported_hosted_tool_types_for_model(model: Any) -> set[str]:
    candidate_model_cost_keys = _get_codex_tool_policy_model_cost_candidates(model)
    if not candidate_model_cost_keys:
        return set()

    model_cost_sources = [
        litellm.model_cost,
        _load_bundled_model_cost_map_for_codex_tool_policy(),
    ]
    for model_cost in model_cost_sources:
        for key in candidate_model_cost_keys:
            model_info = model_cost.get(key)
            if not isinstance(model_info, dict):
                continue

            unsupported_tools = model_info.get(
                _CODEX_UNSUPPORTED_HOSTED_TOOLS_MODEL_INFO_FIELD
            )
            if not isinstance(unsupported_tools, list):
                continue

            return {
                normalized
                for value in unsupported_tools
                if (normalized := _normalize_low_cardinality_tag_value(value))
            }

    return set()


def _get_unsupported_request_param_names_for_model(model: Any) -> set[str]:
    candidate_model_cost_keys = _get_codex_tool_policy_model_cost_candidates(model)
    if not candidate_model_cost_keys:
        return set()

    model_cost_sources = [
        litellm.model_cost,
        _load_bundled_model_cost_map_for_codex_tool_policy(),
    ]
    for model_cost in model_cost_sources:
        for key in candidate_model_cost_keys:
            model_info = model_cost.get(key)
            if not isinstance(model_info, dict):
                continue

            unsupported_params = model_info.get(
                _CODEX_UNSUPPORTED_REQUEST_PARAMS_MODEL_INFO_FIELD
            )
            if not isinstance(unsupported_params, list):
                continue

            return {
                normalized
                for value in unsupported_params
                if (normalized := _normalize_low_cardinality_tag_value(value))
            }

    return set()


def _get_unsupported_input_item_types_for_model(model: Any) -> set[str]:
    candidate_model_cost_keys = _get_codex_tool_policy_model_cost_candidates(model)
    if not candidate_model_cost_keys:
        return set()

    model_cost_sources = [
        litellm.model_cost,
        _load_bundled_model_cost_map_for_codex_tool_policy(),
    ]
    for model_cost in model_cost_sources:
        for key in candidate_model_cost_keys:
            model_info = model_cost.get(key)
            if not isinstance(model_info, dict):
                continue

            unsupported_input_item_types = model_info.get(
                _CODEX_UNSUPPORTED_INPUT_ITEM_TYPES_MODEL_INFO_FIELD
            )
            if not isinstance(unsupported_input_item_types, list):
                continue

            return {
                normalized
                for value in unsupported_input_item_types
                if (normalized := _normalize_low_cardinality_tag_value(value))
            }

    return set()


def _openai_tool_choice_references_tool_type(
    tool_choice: Any,
    tool_types: set[str],
) -> bool:
    if not tool_types:
        return False

    candidates: list[Any] = []
    if isinstance(tool_choice, str):
        candidates.append(tool_choice)
    elif isinstance(tool_choice, dict):
        candidates.extend([tool_choice.get("type"), tool_choice.get("name")])
        function = tool_choice.get("function")
        if isinstance(function, dict):
            candidates.append(function.get("name"))

    for candidate in candidates:
        normalized = _normalize_low_cardinality_tag_value(candidate)
        if normalized in tool_types:
            return True
    return False


def _add_codex_unsupported_hosted_tool_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_tools: list[dict[str, Any]],
    removed_tool_choice: Optional[Any],
) -> dict[str, Any]:
    removed_tool_types = _dedupe_sorted_str_list(
        [
            tool["type"]
            for tool in removed_tools
            if isinstance(tool.get("type"), str) and tool["type"]
        ]
    )
    span_metadata: dict[str, Any] = {
        "removed_count": len(removed_tools),
        "removed_tool_types": removed_tool_types,
    }
    if removed_tool_choice is not None:
        span_metadata["removed_tool_choice"] = removed_tool_choice

    tags_to_add = ["codex-unsupported-hosted-tool-removed"]
    tags_to_add.extend(
        f"codex-unsupported-hosted-tool:{tool_type}"
        for tool_type in removed_tool_types
    )
    if removed_tool_choice is not None:
        tags_to_add.append("codex-unsupported-hosted-tool-choice-removed")

    extra_fields: dict[str, Any] = {
        "codex_unsupported_hosted_tool_removed_count": len(removed_tools),
        "codex_unsupported_hosted_tool_types_removed": removed_tool_types,
        "codex_unsupported_hosted_tools_removed": removed_tools,
        "langfuse_spans": [
            _build_langfuse_span_descriptor(
                name="codex.unsupported_hosted_tool_removed",
                metadata=span_metadata,
            )
        ],
    }
    if removed_tool_choice is not None:
        extra_fields["codex_unsupported_hosted_tool_choice_removed"] = (
            removed_tool_choice
        )

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _request_has_openai_tool_definitions(request_body: dict[str, Any]) -> bool:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return False

    for tool in tools:
        if isinstance(tool, dict) and _get_openai_tool_type(tool):
            return True
    return False


def _add_tool_choice_without_tools_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_tool_choice: Any,
) -> dict[str, Any]:
    span_metadata: dict[str, Any] = {
        "removed_tool_choice": removed_tool_choice,
        "reason": "missing_tools",
    }
    extracted_tool_choice = _extract_openai_passthrough_tool_choice(removed_tool_choice)
    if extracted_tool_choice:
        span_metadata["tool_choice"] = extracted_tool_choice

    tags_to_add = ["xai-tool-choice-without-tools-removed"]
    if extracted_tool_choice:
        tags_to_add.append(
            f"xai-tool-choice-without-tools:{extracted_tool_choice}"
        )

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "xai_tool_choice_without_tools_removed": removed_tool_choice,
            "xai_tool_choice_without_tools_removed_reason": "missing_tools",
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="xai.tool_choice_without_tools_removed",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _drop_tool_choice_without_tools_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], Optional[Any]]:
    if "tool_choice" not in request_body:
        return request_body, None

    if _request_has_openai_tool_definitions(request_body):
        return request_body, None

    updated_body = dict(request_body)
    removed_tool_choice = updated_body.pop("tool_choice", None)
    updated_body = _add_tool_choice_without_tools_logging_metadata(
        updated_body,
        removed_tool_choice=removed_tool_choice,
    )
    return updated_body, removed_tool_choice


def _add_codex_unsupported_request_param_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_params: list[str],
) -> dict[str, Any]:
    normalized_params = _dedupe_sorted_str_list(
        [
            normalized
            for param in removed_params
            if (normalized := _normalize_low_cardinality_tag_value(param))
        ]
    )
    span_metadata: dict[str, Any] = {
        "removed_count": len(removed_params),
        "removed_params": normalized_params,
    }
    tags_to_add = ["codex-unsupported-request-param-removed"]
    tags_to_add.extend(
        f"codex-unsupported-request-param:{param}" for param in normalized_params
    )
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_unsupported_request_param_removed_count": len(removed_params),
            "codex_unsupported_request_params_removed": normalized_params,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="codex.unsupported_request_param_removed",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _drop_unsupported_codex_request_params_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    unsupported_params = _get_unsupported_request_param_names_for_model(
        request_body.get("model")
    )
    if not unsupported_params:
        return request_body, []

    def _drop_from_value(
        value: Any,
        *,
        path: tuple[str, ...] = (),
    ) -> tuple[Any, list[str], bool]:
        if isinstance(value, dict):
            updated_dict: dict[str, Any] = {}
            removed: list[str] = []
            changed = False
            for key, child_value in value.items():
                normalized_key = _normalize_low_cardinality_tag_value(key)
                normalized_path = (
                    ".".join([*path, normalized_key])
                    if normalized_key is not None
                    else None
                )
                if normalized_key in unsupported_params or (
                    normalized_path in unsupported_params
                ):
                    removed.append(
                        normalized_path
                        if normalized_key not in unsupported_params
                        and normalized_path in unsupported_params
                        else key
                    )
                    changed = True
                    continue
                updated_child, child_removed, child_changed = _drop_from_value(
                    child_value,
                    path=(
                        (*path, normalized_key)
                        if normalized_key is not None
                        else path
                    ),
                )
                updated_dict[key] = updated_child
                removed.extend(child_removed)
                changed = changed or child_changed
            return (updated_dict if changed else value), removed, changed

        if isinstance(value, list):
            updated_list: list[Any] = []
            list_removed: list[str] = []
            changed = False
            for item in value:
                updated_item, item_removed, item_changed = _drop_from_value(
                    item,
                    path=path,
                )
                updated_list.append(updated_item)
                list_removed.extend(item_removed)
                changed = changed or item_changed
            return (updated_list if changed else value), list_removed, changed

        return value, [], False

    updated_value, removed_params, changed = _drop_from_value(request_body)
    if not removed_params:
        return request_body, []

    updated_body = (
        updated_value
        if changed and isinstance(updated_value, dict)
        else dict(request_body)
    )

    updated_body = _add_codex_unsupported_request_param_logging_metadata(
        updated_body,
        removed_params=removed_params,
    )
    return updated_body, removed_params


def _add_codex_unsupported_input_item_logging_metadata(
    request_body: dict[str, Any],
    *,
    removed_items: list[dict[str, Any]],
) -> dict[str, Any]:
    removed_item_types = _dedupe_sorted_str_list(
        [
            item["type"]
            for item in removed_items
            if isinstance(item.get("type"), str) and item["type"]
        ]
    )
    span_metadata: dict[str, Any] = {
        "removed_count": len(removed_items),
        "removed_item_types": removed_item_types,
    }

    tags_to_add = ["codex-unsupported-input-item-removed"]
    tags_to_add.extend(
        f"codex-unsupported-input-item:{item_type}"
        for item_type in removed_item_types
    )

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_unsupported_input_item_removed_count": len(removed_items),
            "codex_unsupported_input_item_types_removed": removed_item_types,
            "codex_unsupported_input_items_removed": removed_items,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="codex.unsupported_input_item_removed",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _drop_unsupported_codex_input_items_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unsupported_input_item_types = _get_unsupported_input_item_types_for_model(
        request_body.get("model")
    )
    if not unsupported_input_item_types:
        return request_body, []

    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return request_body, []

    updated_input_items: list[Any] = []
    removed_items: list[dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if not isinstance(item, dict):
            updated_input_items.append(item)
            continue

        item_type = _normalize_low_cardinality_tag_value(item.get("type"))
        if item_type in unsupported_input_item_types:
            removed_item: dict[str, Any] = {
                "type": item_type,
                "index": index,
            }
            if item_type == "reasoning" and isinstance(
                item.get("encrypted_content"), str
            ):
                removed_item["encrypted_content"] = True
            removed_items.append(removed_item)
            continue

        updated_input_items.append(item)

    if not removed_items:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["input"] = updated_input_items
    updated_body = _add_codex_unsupported_input_item_logging_metadata(
        updated_body,
        removed_items=removed_items,
    )
    return updated_body, removed_items


def _drop_unsupported_codex_hosted_tools_from_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unsupported_tool_types = _get_unsupported_hosted_tool_types_for_model(
        request_body.get("model")
    )
    if not unsupported_tool_types:
        return request_body, []

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body, []

    updated_tools: list[Any] = []
    removed_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue

        tool_type = _normalize_low_cardinality_tag_value(_get_openai_tool_type(tool))
        if tool_type in unsupported_tool_types:
            removed_tool: dict[str, Any] = {
                "type": tool_type,
                "index": index,
            }
            tool_name = _get_openai_tool_name(tool)
            if tool_name:
                removed_tool["name"] = tool_name
            removed_tools.append(removed_tool)
            continue

        updated_tools.append(tool)

    if not removed_tools:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["tools"] = updated_tools

    removed_tool_types = {
        tool["type"]
        for tool in removed_tools
        if isinstance(tool.get("type"), str) and tool["type"]
    }
    removed_tool_choice = None
    if _openai_tool_choice_references_tool_type(
        updated_body.get("tool_choice"),
        removed_tool_types,
    ):
        removed_tool_choice = updated_body.pop("tool_choice", None)

    updated_body = _add_codex_unsupported_hosted_tool_logging_metadata(
        updated_body,
        removed_tools=removed_tools,
        removed_tool_choice=removed_tool_choice,
    )
    return updated_body, removed_tools


def _patch_codex_spawn_agent_tool_description(
    tool: dict[str, Any],
    *,
    tool_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _get_openai_tool_name(tool) != _CODEX_SPAWN_AGENT_TOOL_NAME:
        return tool, []

    updated_tool = tool
    patch_events: list[dict[str, Any]] = []
    description_targets: list[tuple[dict[str, Any], str, str]] = [
        (tool, "description", f"tools.{tool_index}.description")
    ]
    function = tool.get("function")
    if isinstance(function, dict):
        description_targets.append(
            (
                function,
                "description",
                f"tools.{tool_index}.function.description",
            )
        )

    for container, key, path in description_targets:
        description = container.get(key)
        if not isinstance(description, str):
            continue

        updated_description, replacement_count = (
            _patch_codex_spawn_agent_description_text(description)
        )
        if replacement_count == 0 or updated_description == description:
            continue

        if updated_tool is tool:
            updated_tool = dict(tool)

        if container is tool:
            updated_tool[key] = updated_description
        else:
            updated_function = dict(container)
            updated_function[key] = updated_description
            updated_tool["function"] = updated_function

        patch_events.append(
            {
                "id": _CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID,
                "status": "applied",
                "tool_name": _CODEX_SPAWN_AGENT_TOOL_NAME,
                "path": path,
                "occurrences": replacement_count,
            }
        )

    parameter_targets: list[tuple[str, str]] = []
    function = updated_tool.get("function")
    if isinstance(function, dict):
        parameter_targets.append(
            ("function", f"tools.{tool_index}.function.parameters")
        )
    if "parameters" in updated_tool or not parameter_targets:
        parameter_targets.append(("tool", f"tools.{tool_index}.parameters"))

    for target_kind, path in parameter_targets:
        if target_kind == "function":
            current_function = updated_tool.get("function")
            if not isinstance(current_function, dict):
                continue
            parameters = current_function.get("parameters")
        else:
            parameters = updated_tool.get("parameters")

        updated_parameters, added_fields = (
            _patch_codex_spawn_agent_payload_parameters(parameters)
        )
        if not added_fields or updated_parameters is parameters:
            continue

        if updated_tool is tool:
            updated_tool = dict(tool)

        if target_kind == "function":
            current_function = updated_tool.get("function")
            if not isinstance(current_function, dict):
                continue
            updated_function = dict(current_function)
            updated_function["parameters"] = updated_parameters
            updated_tool["function"] = updated_function
        else:
            updated_tool["parameters"] = updated_parameters

        patch_events.append(
            {
                "id": _CODEX_SPAWN_AGENT_PAYLOAD_SCHEMA_PATCH_ID,
                "status": "applied",
                "tool_name": _CODEX_SPAWN_AGENT_TOOL_NAME,
                "path": path,
                "fields_added": added_fields,
                "occurrences": 0,
            }
        )

    return updated_tool, patch_events


def _patch_codex_multi_agent_tool_search_description(
    tool: dict[str, Any],
    *,
    tool_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _normalize_low_cardinality_tag_value(tool.get("type")) != (
        _CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE
    ):
        return tool, []

    description = tool.get("description")
    if not isinstance(description, str):
        return tool, []
    if _CODEX_SPAWN_AGENT_FANOUT_POLICY in description:
        return tool, []
    if (
        "Multi-agent tools" not in description
        and "Spawn and manage sub-agents" not in description
    ):
        return tool, []

    updated_tool = dict(tool)
    updated_tool["description"] = (
        f"{description.rstrip()}\n\n{_CODEX_SPAWN_AGENT_FANOUT_POLICY}"
    )
    return updated_tool, [
        {
            "id": _CODEX_SPAWN_AGENT_FANOUT_POLICY_PATCH_ID,
            "status": "applied",
            "tool_name": _CODEX_MULTI_AGENT_TOOL_SEARCH_TYPE,
            "path": f"tools.{tool_index}.description",
            "occurrences": 0,
            "guidance_chars": len(_CODEX_SPAWN_AGENT_FANOUT_POLICY),
        }
    ]


def _patch_codex_core_tool_description(
    tool: dict[str, Any],
    *,
    tool_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tool_name = _get_openai_tool_name(tool)
    guidance = _get_codex_core_tool_guidance(tool_name)
    if guidance is None:
        return tool, []

    updated_tool = tool
    patch_events: list[dict[str, Any]] = []
    description_targets: list[tuple[dict[str, Any], str, str]] = []
    function = tool.get("function")
    if isinstance(function, dict):
        description_targets.append(
            (
                function,
                "description",
                f"tools.{tool_index}.function.description",
            )
        )
    if "description" in tool or not description_targets:
        description_targets.append(
            (tool, "description", f"tools.{tool_index}.description")
        )

    for container, key, path in description_targets:
        updated_description, changed = _append_codex_core_tool_guidance_to_description(
            container.get(key),
            guidance=guidance,
        )
        if not changed:
            continue

        if updated_tool is tool:
            updated_tool = dict(tool)

        if container is tool:
            updated_tool[key] = updated_description
        else:
            updated_function = dict(container)
            updated_function[key] = updated_description
            updated_tool["function"] = updated_function

        normalized_tool_name = (
            _normalize_low_cardinality_tag_value(tool_name) or "unknown"
        )
        patch_events.append(
            {
                "id": f"{_CODEX_CORE_TOOL_GUIDANCE_PATCH_PREFIX}-{normalized_tool_name}",
                "status": "applied",
                "tool_name": tool_name,
                "path": path,
                "occurrences": 0,
                "guidance_chars": len(guidance),
            }
        )

    return updated_tool, patch_events


def _add_codex_tool_description_patch_logging_metadata(
    request_body: dict[str, Any],
    patch_events: list[dict[str, Any]],
) -> dict[str, Any]:
    patch_ids = _dedupe_sorted_str_list(
        [
            event["id"]
            for event in patch_events
            if isinstance(event.get("id"), str) and event["id"]
        ]
    )
    replacement_count = sum(
        event["occurrences"]
        for event in patch_events
        if isinstance(event.get("occurrences"), int)
    )
    span_metadata: dict[str, Any] = {
        "patch_count": len(patch_events),
        "replacement_count": replacement_count,
    }
    if patch_ids:
        span_metadata["patch_ids"] = patch_ids

    tags_to_add = ["codex-tool-description-patch"]
    tags_to_add.extend(
        f"codex-tool-description-patch:{patch_id}" for patch_id in patch_ids
    )

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "codex_tool_description_patch_count": len(patch_events),
            "codex_tool_description_patch_replacement_count": replacement_count,
            "codex_tool_description_patch_ids": patch_ids,
            "codex_tool_description_patch_events": patch_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="codex.tool_description_patch",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _apply_codex_tool_description_patches_to_request_body(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return request_body, []

    updated_tools: list[Any] = []
    patch_events: list[dict[str, Any]] = []
    changed = False
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            updated_tools.append(tool)
            continue
        updated_tool, tool_patch_events = _patch_codex_spawn_agent_tool_description(
            tool,
            tool_index=index,
        )
        updated_tool, tool_search_patch_events = (
            _patch_codex_multi_agent_tool_search_description(
                updated_tool,
                tool_index=index,
            )
        )
        updated_tool, core_tool_patch_events = _patch_codex_core_tool_description(
            updated_tool,
            tool_index=index,
        )
        updated_tools.append(updated_tool)
        patch_events.extend(tool_patch_events)
        patch_events.extend(tool_search_patch_events)
        patch_events.extend(core_tool_patch_events)
        if updated_tool is not tool:
            changed = True

    if not changed or not patch_events:
        return request_body, []

    updated_body = dict(request_body)
    updated_body["tools"] = updated_tools
    updated_body = _add_codex_tool_description_patch_logging_metadata(
        updated_body,
        patch_events,
    )
    return updated_body, patch_events


def _extract_openai_passthrough_tool_choice(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _normalize_low_cardinality_tag_value(value)
    if isinstance(value, dict):
        for key in ("type", "name"):
            normalized = _normalize_low_cardinality_tag_value(value.get(key))
            if normalized:
                return normalized
    return None


def _extract_codex_request_breakout_fields(
    request_body: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    tags_to_add: list[str] = []
    extra_fields: dict[str, Any] = {}

    reasoning = request_body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _normalize_low_cardinality_tag_value(reasoning.get("effort"))
        if effort:
            tags_to_add.extend([f"codex-effort:{effort}", f"effort:{effort}"])
            extra_fields["codex_reasoning_effort"] = effort

    tool_choice = _extract_openai_passthrough_tool_choice(request_body.get("tool_choice"))
    if tool_choice:
        tags_to_add.append(f"codex-tool-choice:{tool_choice}")
        extra_fields["codex_tool_choice"] = tool_choice

    parallel_tool_calls = request_body.get("parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool):
        tags_to_add.append(
            f"codex-parallel-tools:{'true' if parallel_tool_calls else 'false'}"
        )
        extra_fields["codex_parallel_tool_calls"] = parallel_tool_calls

    include = request_body.get("include")
    normalized_includes: list[str] = []
    if isinstance(include, list):
        for value in include:
            normalized = _normalize_low_cardinality_tag_value(value)
            if normalized:
                normalized_includes.append(normalized)
                tags_to_add.append(f"codex-include:{normalized}")
    if normalized_includes:
        extra_fields["codex_include"] = _dedupe_sorted_str_list(normalized_includes)

    prompt_cache_key = request_body.get("prompt_cache_key")
    if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
        extra_fields["codex_prompt_cache_key_present"] = True

    return tags_to_add, extra_fields


def _add_codex_request_breakout_logging_metadata(
    request_body: dict[str, Any],
) -> dict[str, Any]:
    tags_to_add, extra_fields = _extract_codex_request_breakout_fields(request_body)
    if not tags_to_add and not extra_fields:
        return request_body
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _add_claude_persisted_output_logging_metadata(
    request_body: dict[str, Any],
    expanded_count: int,
    hooks: set[str],
    source_metadata_items: list[dict[str, Any]],
) -> dict[str, Any]:
    span_metadata: dict[str, Any] = {
        "expanded_count": expanded_count,
        "hook_count": len(hooks),
    }
    if hooks:
        span_metadata["hooks"] = sorted(hooks)
    if source_metadata_items:
        span_metadata["source_count"] = len(source_metadata_items)
        span_metadata["source_paths"] = [
            item["path"]
            for item in source_metadata_items
            if isinstance(item.get("path"), str)
        ]
        span_metadata["source_content_hashes"] = [
            item["content_hash"]
            for item in source_metadata_items
            if isinstance(item.get("content_hash"), str)
        ]
        span_metadata["source_bytes"] = [
            item["bytes"]
            for item in source_metadata_items
            if isinstance(item.get("bytes"), int)
        ]
    tags_to_add = ["claude-persisted-output-expanded"]
    tags_to_add.extend(
        f"claude-persisted-output-hook:{hook}" for hook in sorted(hooks) if hook
    )
    extra_fields: dict[str, Any] = {
        "claude_persisted_output_expanded": True,
        "claude_persisted_output_expanded_count": expanded_count,
        "langfuse_spans": [
            _build_langfuse_span_descriptor(
                name="claude.persisted_output_expand",
                metadata=span_metadata,
            )
        ],
    }
    if hooks:
        extra_fields["claude_persisted_output_hooks"] = sorted(hooks)
    if source_metadata_items:
        extra_fields["claude_persisted_output_source_paths"] = [
            item["path"]
            for item in source_metadata_items
            if isinstance(item.get("path"), str)
        ]
        extra_fields["claude_persisted_output_source_basenames"] = [
            item["basename"]
            for item in source_metadata_items
            if isinstance(item.get("basename"), str)
        ]
        extra_fields["claude_persisted_output_source_content_hashes"] = [
            item["content_hash"]
            for item in source_metadata_items
            if isinstance(item.get("content_hash"), str)
        ]
        extra_fields["claude_persisted_output_source_bytes"] = [
            item["bytes"]
            for item in source_metadata_items
            if isinstance(item.get("bytes"), int)
        ]
    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _parse_anthropic_billing_header_text(text: str) -> dict[str, str]:
    parsed_fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line.lower().startswith(_ANTHROPIC_BILLING_HEADER_PREFIX):
            continue
        raw_header_value = stripped_line.split(":", 1)[1].strip()
        for segment in raw_header_value.split(";"):
            cleaned_segment = segment.strip()
            if not cleaned_segment or "=" not in cleaned_segment:
                continue
            key, value = cleaned_segment.split("=", 1)
            cleaned_key = key.strip()
            cleaned_value = value.strip()
            if cleaned_key and cleaned_value:
                parsed_fields[cleaned_key] = cleaned_value
    return parsed_fields


def _extract_anthropic_billing_header_fields(value: Any) -> dict[str, str]:
    parsed_fields: dict[str, str] = {}

    if isinstance(value, str):
        return _parse_anthropic_billing_header_text(value)

    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            parsed_fields.update(_parse_anthropic_billing_header_text(value["text"]))
        for child in value.values():
            parsed_fields.update(_extract_anthropic_billing_header_fields(child))
        return parsed_fields

    if isinstance(value, list):
        for child in value:
            parsed_fields.update(_extract_anthropic_billing_header_fields(child))

    return parsed_fields


def _extract_anthropic_billing_header_fields_from_request_body(
    request_body: dict[str, Any]
) -> dict[str, str]:
    return _extract_anthropic_billing_header_fields(request_body.get("system"))


def _parse_claude_code_version(cc_version: Optional[str]) -> Optional[tuple[int, int, int]]:
    if not cc_version:
        return None

    match = _CLAUDE_CC_VERSION_PATTERN.match(cc_version.strip())
    if match is None:
        return None

    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _resolve_claude_auto_memory_template_path(cc_version: Optional[str]) -> Optional[Path]:
    parsed_version = _parse_claude_code_version(cc_version)
    if parsed_version is None:
        return None

    major, minor, patch = parsed_version
    min_major, min_minor, min_patch = _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION

    if (major, minor) != (min_major, min_minor):
        return None
    if patch < min_patch:
        return None

    return _CLAUDE_AUTO_MEMORY_TEMPLATE_PATH


def _load_claude_context_replacement_template(template_path: Path) -> str:
    cached_template = _claude_context_replacement_template_cache.get(template_path)
    if cached_template is not None:
        return cached_template

    template_text = template_path.read_text(encoding="utf-8").strip()
    if not template_text:
        raise ValueError(f"Claude context replacement template is empty: {template_path}")

    cached_template = template_text + "\n"
    _claude_context_replacement_template_cache[template_path] = cached_template
    return cached_template


def _load_claude_prompt_patch_manifest(template_path: Path) -> dict[str, Any]:
    cached_manifest = _claude_prompt_patch_manifest_cache.get(template_path)
    if cached_manifest is not None:
        return cached_manifest

    manifest = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid Claude prompt patch manifest: {template_path}")

    patches = manifest.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError(f"Claude prompt patch manifest has no patches: {template_path}")

    normalized_patches: list[dict[str, str]] = []
    for patch_descriptor in patches:
        if not isinstance(patch_descriptor, dict):
            raise ValueError(
                f"Invalid Claude prompt patch descriptor in {template_path}"
            )
        patch_id = patch_descriptor.get("id")
        before_text = patch_descriptor.get("before")
        after_text = patch_descriptor.get("after")
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError(
                f"Claude prompt patch manifest is missing patch id in {template_path}"
            )
        if not isinstance(before_text, str) or not before_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing before text for {patch_id}"
            )
        if not isinstance(after_text, str) or not after_text:
            raise ValueError(
                f"Claude prompt patch manifest is missing after text for {patch_id}"
            )
        normalized_patches.append(
            {
                "id": patch_id,
                "before": before_text,
                "after": after_text,
            }
        )

    normalized_manifest = {
        "source": manifest.get("source"),
        "patches": normalized_patches,
    }
    _claude_prompt_patch_manifest_cache[template_path] = normalized_manifest
    return normalized_manifest


def _extract_markdown_section(markdown_text: str, heading: str) -> str:
    section_pattern = re.compile(
        rf"(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)"
    )
    match = section_pattern.search(markdown_text)
    if match is None:
        raise ValueError(f"Missing Claude auto-memory section: {heading}")
    return match.group(0).rstrip()


def _render_claude_auto_memory_replacement(
    auto_memory_section: str, cc_version: str
) -> tuple[str, Path]:
    template_path = _resolve_claude_auto_memory_template_path(cc_version)
    if template_path is None:
        raise ValueError(f"Unsupported Claude Code version for auto-memory override: {cc_version}")

    template_text = _load_claude_context_replacement_template(template_path)
    rendered_text = template_text
    if "{{TYPES_XML_BLOCK}}" in rendered_text:
        types_match = _CLAUDE_TYPES_XML_BLOCK_PATTERN.search(auto_memory_section)
        if types_match is None:
            raise ValueError("Missing Claude auto-memory <types> block")
        rendered_text = rendered_text.replace(
            "{{TYPES_XML_BLOCK}}", types_match.group(0).rstrip()
        )

    section_placeholders = {
        "{{WHAT_NOT_TO_SAVE_SECTION}}": "What NOT to save in memory",
        "{{BEFORE_RECOMMENDING_SECTION}}": "Before recommending from memory",
        "{{MEMORY_AND_PERSISTENCE_SECTION}}": "Memory and other forms of persistence",
    }
    for placeholder, heading in section_placeholders.items():
        if placeholder in rendered_text:
            rendered_text = rendered_text.replace(
                placeholder, _extract_markdown_section(auto_memory_section, heading)
            )

    unresolved_placeholders = _CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN.findall(
        rendered_text
    )
    if unresolved_placeholders:
        raise ValueError(
            "Unresolved Claude context replacement placeholders: "
            + ", ".join(sorted(unresolved_placeholders))
        )

    return rendered_text.rstrip() + "\n", template_path


def _replace_claude_auto_memory_section_in_text(
    text: str, cc_version: str
) -> tuple[str, Optional[dict[str, Any]]]:
    if "# auto memory" not in text:
        return text, None

    section_match = _CLAUDE_AUTO_MEMORY_SECTION_PATTERN.search(text)
    if section_match is None:
        return text, None

    replacement_text, template_path = _render_claude_auto_memory_replacement(
        section_match.group(0),
        cc_version,
    )
    replacement_event: dict[str, Any] = {
        "id": "auto-memory",
        "status": "resolved",
        "cc_version": cc_version,
        "template_path": str(template_path.relative_to(Path(__file__).resolve().parents[3])),
        "output_chars": len(replacement_text),
    }
    return (
        text[: section_match.start()] + replacement_text + text[section_match.end() :],
        replacement_event,
    )


def _replace_claude_system_prompt_override_in_value(
    value: Any, cc_version: str
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            if "# auto memory" not in value["text"]:
                return value, []
            try:
                updated_text, event = _replace_claude_auto_memory_section_in_text(
                    value["text"], cc_version
                )
            except Exception as exc:
                return value, [
                    {
                        "id": "auto-memory",
                        "status": "failed",
                        "cc_version": cc_version,
                        "error": exc.__class__.__name__,
                    }
                ]

            if event is None:
                return value, []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, [event]

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        list_combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            list_combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), list_combined_events

    return value, []


def _add_claude_system_prompt_override_logging_metadata(
    request_body: dict[str, Any], override_events: list[dict[str, Any]]
) -> dict[str, Any]:
    override_ids = sorted(
        {
            event["id"]
            for event in override_events
            if isinstance(event.get("id"), str) and event["id"]
        }
    )
    failure_ids = sorted(
        {
            event["id"]
            for event in override_events
            if event.get("status") == "failed"
            and isinstance(event.get("id"), str)
            and event["id"]
        }
    )
    statuses = [
        event["status"]
        for event in override_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in override_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    template_paths = sorted(
        {
            event["template_path"]
            for event in override_events
            if isinstance(event.get("template_path"), str) and event["template_path"]
        }
    )

    tags_to_add = ["claude-system-prompt-override"]
    tags_to_add.extend(
        f"claude-system-prompt-override:{override_id}" for override_id in override_ids
    )
    if failure_ids:
        tags_to_add.append("claude-system-prompt-override-failed")

    span_metadata: dict[str, Any] = {
        "override_count": len(override_events),
        "failure_count": len(failure_ids),
    }
    if override_ids:
        span_metadata["override_ids"] = override_ids
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_system_prompt_override_count": len(override_events),
            "claude_system_prompt_override_ids": override_ids,
            "claude_system_prompt_override_failure_ids": failure_ids,
            "claude_system_prompt_override_statuses": statuses,
            "claude_system_prompt_override_cc_versions": cc_versions,
            "claude_system_prompt_override_template_paths": template_paths,
            "claude_system_prompt_override_events": override_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="claude.system_prompt_override",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _replace_claude_system_prompt_in_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cc_version = billing_header_fields.get("cc_version")
    if not isinstance(cc_version, str) or not cc_version:
        return request_body, []
    template_path = _resolve_claude_auto_memory_template_path(cc_version)
    if template_path is None or "system" not in request_body:
        return request_body, []

    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    updated_system, override_events = _replace_claude_system_prompt_override_in_value(
        request_body["system"],
        cc_version,
    )
    if not override_events:
        return request_body, []

    updated_body["system"] = updated_system
    updated_body = _add_claude_system_prompt_override_logging_metadata(
        updated_body,
        override_events,
    )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if (
                    isinstance(span_descriptor, dict)
                    and span_descriptor.get("name") == "claude.system_prompt_override"
                ):
                    span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, override_events


def _apply_claude_prompt_patches_in_text(
    text: str, cc_version: str
) -> tuple[str, list[dict[str, Any]]]:
    manifest_path = _CLAUDE_PROMPT_PATCH_MANIFEST_PATH
    manifest = _load_claude_prompt_patch_manifest(manifest_path)
    updated_text = text
    patch_events: list[dict[str, Any]] = []
    relative_manifest_path = str(
        manifest_path.relative_to(Path(__file__).resolve().parents[3])
    )

    for patch_descriptor in manifest["patches"]:
        before_text = patch_descriptor["before"]
        if before_text not in updated_text:
            continue

        after_text = patch_descriptor["after"]
        occurrences = updated_text.count(before_text)
        updated_text = updated_text.replace(before_text, after_text)
        patch_events.append(
            {
                "id": patch_descriptor["id"],
                "status": "resolved",
                "cc_version": cc_version,
                "manifest_path": relative_manifest_path,
                "occurrences": occurrences,
            }
        )

    return updated_text, patch_events


def _replace_claude_prompt_patches_in_value(
    value: Any, cc_version: str
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            try:
                updated_text, patch_events = _apply_claude_prompt_patches_in_text(
                    value["text"], cc_version
                )
            except Exception as exc:
                return value, [
                    {
                        "id": "manifest-load",
                        "status": "failed",
                        "cc_version": cc_version,
                        "error": exc.__class__.__name__,
                    }
                ]
            if not patch_events:
                return value, []
            updated_value = dict(value)
            updated_value["text"] = updated_text
            return updated_value, patch_events

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        list_combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            list_combined_events.extend(child_events)
            if updated_child is not child:
                changed = True
        return (updated_list if changed else value), list_combined_events

    return value, []


def _add_claude_prompt_patch_logging_metadata(
    request_body: dict[str, Any], patch_events: list[dict[str, Any]]
) -> dict[str, Any]:
    patch_ids = sorted(
        {
            event["id"]
            for event in patch_events
            if isinstance(event.get("id"), str) and event["id"]
        }
    )
    failure_ids = sorted(
        {
            event["id"]
            for event in patch_events
            if event.get("status") == "failed"
            and isinstance(event.get("id"), str)
            and event["id"]
        }
    )
    statuses = [
        event["status"]
        for event in patch_events
        if isinstance(event.get("status"), str) and event["status"]
    ]
    cc_versions = sorted(
        {
            event["cc_version"]
            for event in patch_events
            if isinstance(event.get("cc_version"), str) and event["cc_version"]
        }
    )
    manifest_paths = sorted(
        {
            event["manifest_path"]
            for event in patch_events
            if isinstance(event.get("manifest_path"), str) and event["manifest_path"]
        }
    )
    total_occurrences = sum(
        event["occurrences"]
        for event in patch_events
        if isinstance(event.get("occurrences"), int)
    )

    tags_to_add = ["claude-prompt-patch"]
    tags_to_add.extend(f"claude-prompt-patch:{patch_id}" for patch_id in patch_ids)
    if failure_ids:
        tags_to_add.append("claude-prompt-patch-failed")

    span_metadata: dict[str, Any] = {
        "patch_count": len(patch_events),
        "replacement_count": total_occurrences,
        "failure_count": len(failure_ids),
    }
    if patch_ids:
        span_metadata["patch_ids"] = patch_ids
    if cc_versions:
        span_metadata["cc_versions"] = cc_versions

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_prompt_patch_count": len(patch_events),
            "claude_prompt_patch_replacement_count": total_occurrences,
            "claude_prompt_patch_ids": patch_ids,
            "claude_prompt_patch_failure_ids": failure_ids,
            "claude_prompt_patch_statuses": statuses,
            "claude_prompt_patch_cc_versions": cc_versions,
            "claude_prompt_patch_manifest_paths": manifest_paths,
            "claude_prompt_patch_events": patch_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="claude.prompt_patch",
                    metadata=span_metadata,
                )
            ],
        },
    )


def _apply_claude_prompt_patches_to_anthropic_request_body(
    request_body: dict[str, Any], billing_header_fields: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cc_version = billing_header_fields.get("cc_version")
    if not cc_version:
        return request_body, []

    span_started_at = datetime.now(timezone.utc)
    updated_body, patch_events = _replace_claude_prompt_patches_in_value(
        request_body,
        cc_version,
    )
    if not patch_events:
        return request_body, []

    if not isinstance(updated_body, dict):
        return request_body, []

    updated_body = _add_claude_prompt_patch_logging_metadata(
        updated_body,
        patch_events,
    )

    litellm_metadata = updated_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        langfuse_spans = litellm_metadata.get("langfuse_spans")
        if isinstance(langfuse_spans, list):
            for span_descriptor in langfuse_spans:
                if (
                    isinstance(span_descriptor, dict)
                    and span_descriptor.get("name") == "claude.prompt_patch"
                ):
                    span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                        span_started_at
                    )
                    span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                        datetime.now(timezone.utc)
                    )
    return updated_body, patch_events


def _add_anthropic_billing_header_logging_metadata(
    request_body: dict[str, Any],
    billing_header_fields: dict[str, str],
) -> dict[str, Any]:
    tags_to_add = ["anthropic-billing-header"]
    for key in sorted(billing_header_fields):
        value = billing_header_fields[key]
        tags_to_add.append(f"anthropic-billing-header-key:{key}")
        tags_to_add.append(f"anthropic-billing-header:{key}={value}")

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "anthropic_billing_header_present": True,
            "anthropic_billing_header_keys": sorted(billing_header_fields),
            "anthropic_billing_header_fields": dict(billing_header_fields),
        },
    )


def _expand_claude_persisted_output_in_anthropic_request_body(
    request_body: dict[str, Any]
) -> Tuple[dict[str, Any], int, set[str], list[dict[str, Any]]]:
    span_started_at = datetime.now(timezone.utc)
    (
        updated_body,
        expanded_count,
        hooks,
        source_metadata_items,
    ) = _expand_claude_persisted_output_value(
        request_body
    )
    if isinstance(updated_body, dict):
        if expanded_count > 0:
            updated_body = _add_claude_persisted_output_logging_metadata(
                updated_body,
                expanded_count,
                hooks,
                source_metadata_items,
            )
            litellm_metadata = updated_body.get("litellm_metadata")
            if isinstance(litellm_metadata, dict):
                langfuse_spans = litellm_metadata.get("langfuse_spans")
                if isinstance(langfuse_spans, list):
                    for span_descriptor in langfuse_spans:
                        if (
                            isinstance(span_descriptor, dict)
                            and span_descriptor.get("name")
                            == "claude.persisted_output_expand"
                        ):
                            span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                                span_started_at
                            )
                            span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                                datetime.now(timezone.utc)
                            )
        return updated_body, expanded_count, hooks, source_metadata_items
    return request_body, 0, set(), []


def _parse_aawm_directive_attributes(attrs_text: str) -> dict[str, str]:
    parsed_attrs: dict[str, str] = {}
    for match in _AAWM_DYNAMIC_DIRECTIVE_ATTR_PATTERN.finditer(attrs_text):
        value = (
            match.group("double")
            or match.group("single")
            or match.group("bare")
            or ""
        ).strip()
        if value:
            parsed_attrs[match.group("key")] = value
    return parsed_attrs


def _get_aawm_directive_attrs_text(match: re.Match[str]) -> str:
    return (
        (match.group("html_attrs") or match.group("at_attrs") or match.group("line_attrs") or "")
    ).strip()


def _iter_anthropic_text_fragments(value: Any):
    if isinstance(value, str):
        yield value
        return

    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            yield value["text"]
            return
        for child in value.values():
            yield from _iter_anthropic_text_fragments(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from _iter_anthropic_text_fragments(child)


def _extract_claude_agent_and_tenant_from_request_body(
    request_body: dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
    for top_level_key in ("messages", "system"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            match = _CLAUDE_AGENT_TENANT_PATTERN.search(fragment)
            if match is None:
                continue
            agent = match.group("agent").strip()
            tenant = match.group("tenant").strip()
            if agent and tenant:
                return agent, tenant
    return None, None


def _build_aawm_context_for_anthropic_request(
    request_body: dict[str, Any]
) -> dict[str, str]:
    context: dict[str, str] = {}
    agent, tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if agent:
        context["agent"] = agent
    if tenant:
        context["tenant"] = tenant
    return context


def _add_claude_child_agent_observability_metadata(
    request_body: dict[str, Any],
    *,
    explicit_tenant_id: Optional[str] = None,
) -> dict[str, Any]:
    agent, tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if not agent and not tenant:
        return request_body

    extra_fields: dict[str, Any] = {}
    tags_to_add: list[str] = []
    litellm_metadata = request_body.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        litellm_metadata = {}

    if agent:
        extra_fields["agent_name"] = agent
        extra_fields["aawm_claude_agent_name"] = agent
        normalized_agent = _normalize_low_cardinality_tag_value(agent) or "unknown"
        tags_to_add.append(f"claude-agent:{normalized_agent}")

        existing_trace_name = litellm_metadata.get("trace_name")
        child_trace_name = f"claude-code.{agent}"
        if existing_trace_name != child_trace_name:
            if existing_trace_name and not litellm_metadata.get("source_trace_name"):
                extra_fields["source_trace_name"] = existing_trace_name
            extra_fields["trace_name"] = child_trace_name

    if tenant:
        tenant_for_identity = explicit_tenant_id or tenant
        extra_fields["tenant_id"] = tenant_for_identity
        extra_fields["aawm_tenant_id"] = tenant_for_identity
        extra_fields["aawm_claude_project"] = tenant
        existing_trace_user_id = litellm_metadata.get("trace_user_id")
        if existing_trace_user_id != tenant_for_identity:
            if existing_trace_user_id and not litellm_metadata.get(
                "source_trace_user_id"
            ):
                extra_fields["source_trace_user_id"] = existing_trace_user_id
            extra_fields["trace_user_id"] = tenant_for_identity
        tags_to_add.append(
            f"claude-project:{_normalize_low_cardinality_tag_value(tenant) or 'unknown'}"
        )

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )


def _detect_claude_post_rewrite_context_files(
    request_body: dict[str, Any]
) -> list[str]:
    present_files: list[str] = []
    seen_files: set[str] = set()

    for top_level_key in ("system", "messages"):
        for fragment in _iter_anthropic_text_fragments(request_body.get(top_level_key)):
            for marker, _tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
                if marker in seen_files:
                    continue
                if marker in fragment:
                    present_files.append(marker)
                    seen_files.add(marker)

    return present_files


def _add_claude_post_rewrite_context_file_logging_metadata(
    request_body: dict[str, Any]
) -> dict[str, Any]:
    present_files = _detect_claude_post_rewrite_context_files(request_body)
    if not present_files:
        return request_body

    tags_to_add = ["claude-post-rewrite-context-file-present"]
    for marker, tag_suffix in _CLAUDE_POST_REWRITE_CONTEXT_FILE_MARKERS:
        if marker in present_files:
            tags_to_add.append(f"claude-post-rewrite-context-file:{tag_suffix}")

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "claude_post_rewrite_context_files_present": present_files,
            "claude_post_rewrite_context_file_count": len(present_files),
        },
    )


def _build_aawm_dynamic_injection_failure_text(proc_name: str) -> str:
    return _AAWM_DYNAMIC_INJECTION_FAILURE_TEMPLATE.format(
        proc_name=proc_name or "unknown"
    )


def _append_aawm_dynamic_injection_dsn_query_params(
    dsn: str,
    params: dict[str, Optional[str]],
) -> str:
    parsed = urlsplit(dsn)
    if not parsed.scheme:
        return dsn

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    existing_keys = {key for key, _value in query_items}
    for key, value in params.items():
        cleaned_value = _clean_secret_string(value)
        if cleaned_value and key not in existing_keys:
            query_items.append((key, cleaned_value))
            existing_keys.add(key)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def _get_aawm_dynamic_injection_application_name() -> str:
    return (
        _get_first_secret_value(_AAWM_DB_APPLICATION_NAME_ENV_VARS)
        or _AAWM_DYNAMIC_INJECTION_APPLICATION_NAME
    )


def _get_aawm_dynamic_injection_server_settings() -> dict[str, str]:
    return {"application_name": _get_aawm_dynamic_injection_application_name()}


async def _initialize_aawm_dynamic_injection_connection(conn: Any) -> None:
    await conn.execute(
        "select set_config($1, $2, false)",
        "application_name",
        _get_aawm_dynamic_injection_application_name(),
    )


def _build_aawm_dynamic_injection_dsn() -> Optional[str]:
    host = _get_first_secret_value(_AAWM_DB_HOST_ENV_VARS)
    port = _get_first_secret_value(_AAWM_DB_PORT_ENV_VARS)
    user = _get_first_secret_value(_AAWM_DB_USER_ENV_VARS)
    password = _get_first_secret_value(_AAWM_DB_PASSWORD_ENV_VARS)
    database = _get_first_secret_value(_AAWM_DB_NAME_ENV_VARS)
    sslmode = _normalize_aawm_sslmode(
        _get_first_secret_value(_AAWM_DB_SSLMODE_ENV_VARS)
        or _get_first_secret_value(_AAWM_DB_SSL_BOOL_ENV_VARS)
    )

    has_component_config = any((host, port, user, password, database, sslmode))
    if has_component_config:
        if not host or not user or not database:
            return None

        credentials = quote(user, safe="")
        if password:
            credentials += f":{quote(password, safe='')}"
        dsn = (
            f"postgresql://{credentials}@{host}:{port or '5432'}/"
            f"{quote(database, safe='')}"
        )
        if sslmode:
            dsn += f"?{urlencode({'sslmode': sslmode})}"
        return _append_aawm_dynamic_injection_dsn_query_params(
            dsn,
            {"application_name": _get_aawm_dynamic_injection_application_name()},
        )

    url_dsn = _get_first_secret_value(_AAWM_DB_URL_ENV_VARS)
    if not url_dsn:
        return None
    return _append_aawm_dynamic_injection_dsn_query_params(
        url_dsn,
        {"application_name": _get_aawm_dynamic_injection_application_name()},
    )


async def _get_aawm_dynamic_injection_pool() -> Any:
    global _aawm_dynamic_injection_pool

    if _aawm_dynamic_injection_pool is not None:
        return _aawm_dynamic_injection_pool

    async with _aawm_dynamic_injection_pool_lock:
        if _aawm_dynamic_injection_pool is not None:
            return _aawm_dynamic_injection_pool

        dsn = _build_aawm_dynamic_injection_dsn()
        if not dsn:
            raise RuntimeError("AAWM dynamic injection database configuration is missing")

        try:
            asyncpg = importlib.import_module("asyncpg")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AAWM dynamic injection requires asyncpg to be installed"
            ) from exc

        _aawm_dynamic_injection_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=4,
            command_timeout=10,
            statement_cache_size=0,
            server_settings=_get_aawm_dynamic_injection_server_settings(),
            init=_initialize_aawm_dynamic_injection_connection,
        )
        return _aawm_dynamic_injection_pool


async def _call_aawm_get_agent_memories(
    *, agent_name: str, tenant_id: str
) -> Optional[str]:
    pool = await _get_aawm_dynamic_injection_pool()
    result = await pool.fetchval(
        "SELECT get_agent_memories($1, $2)",
        agent_name,
        tenant_id,
    )
    if isinstance(result, str):
        stripped_result = result.strip()
        if stripped_result:
            return stripped_result
    return None


def _resolve_aawm_dynamic_context_fields(
    proc_name: str, directive_attrs: dict[str, str]
) -> tuple[str, ...]:
    raw_ctx = directive_attrs.get("ctx")
    if raw_ctx:
        ctx_fields = tuple(
            field.strip() for field in raw_ctx.split(",") if field.strip()
        )
    else:
        ctx_fields = _AAWM_DYNAMIC_PROC_DEFAULT_CTX_FIELDS.get(proc_name, ())

    if not ctx_fields:
        raise ValueError("No AAWM context fields were provided")

    return tuple(dict.fromkeys(ctx_fields))


def _select_aawm_dynamic_context(
    *, context_fields: tuple[str, ...], available_context: dict[str, str]
) -> dict[str, str]:
    selected_context: dict[str, str] = {}
    for field_name in context_fields:
        if field_name not in {"agent", "tenant"}:
            raise ValueError(f"Unsupported AAWM context field: {field_name}")

        field_value = available_context.get(field_name)
        if not field_value:
            raise ValueError(f"Missing AAWM context field: {field_name}")
        selected_context[field_name] = field_value
    return selected_context


async def _resolve_aawm_dynamic_directive(
    directive_attrs: dict[str, str],
    available_context: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    raw_proc_name = directive_attrs.get("p") or directive_attrs.get("proc") or "unknown"
    proc_name = _AAWM_DYNAMIC_PROC_ALIASES.get(raw_proc_name, raw_proc_name)
    context_fields = _resolve_aawm_dynamic_context_fields(proc_name, directive_attrs)
    selected_context = _select_aawm_dynamic_context(
        context_fields=context_fields,
        available_context=available_context,
    )

    event: dict[str, Any] = {
        "proc": proc_name,
        "context_keys": list(context_fields),
    }
    version = directive_attrs.get("v") or directive_attrs.get("version")
    if version:
        event["version"] = version
    scope = directive_attrs.get("s") or directive_attrs.get("scope")
    if scope:
        event["scope"] = scope

    if proc_name == _AAWM_AGENT_MEMORY_PROC_NAME:
        injected_text = await _call_aawm_get_agent_memories(
            agent_name=selected_context["agent"],
            tenant_id=selected_context["tenant"],
        )
        if injected_text is None:
            event["status"] = "empty"
            return _AAWM_NO_MEMORIES_TEMPLATE, event

        event["status"] = "resolved"
        event["output_chars"] = len(injected_text)
        return injected_text, event

    raise ValueError(f"Unsupported AAWM proc: {proc_name}")


async def _expand_aawm_dynamic_directives_in_text(
    text: str, available_context: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    matches = list(_AAWM_DYNAMIC_DIRECTIVE_PATTERN.finditer(text))
    if not matches:
        return text, []

    rebuilt_parts: list[str] = []
    injection_events: list[dict[str, Any]] = []
    cursor = 0

    for match in matches:
        rebuilt_parts.append(text[cursor:match.start()])
        directive_attrs = _parse_aawm_directive_attributes(
            _get_aawm_directive_attrs_text(match)
        )
        proc_name = (
            directive_attrs.get("p")
            or directive_attrs.get("proc")
            or "unknown"
        )
        try:
            replacement_text, event = await _resolve_aawm_dynamic_directive(
                directive_attrs,
                available_context,
            )
        except Exception as exc:
            normalized_proc_name = _AAWM_DYNAMIC_PROC_ALIASES.get(proc_name, proc_name)
            replacement_text = _build_aawm_dynamic_injection_failure_text(
                normalized_proc_name
            )
            requested_context_fields = []
            raw_ctx = directive_attrs.get("ctx")
            if raw_ctx:
                requested_context_fields = [
                    field.strip() for field in raw_ctx.split(",") if field.strip()
                ]
            event = {
                "proc": normalized_proc_name,
                "status": "failed",
                "error": exc.__class__.__name__,
                "context_keys": requested_context_fields or list(available_context.keys()),
            }
        rebuilt_parts.append(replacement_text)
        injection_events.append(event)
        cursor = match.end()

    rebuilt_parts.append(text[cursor:])
    return "".join(rebuilt_parts), injection_events


async def _expand_aawm_dynamic_directives_in_value(
    value: Any,
    available_context: dict[str, str],
) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            updated_text, injection_events = await _expand_aawm_dynamic_directives_in_text(
                value["text"],
                available_context,
            )
            if injection_events:
                updated_value = dict(value)
                updated_value["text"] = updated_text
                return updated_value, injection_events
            return value, []

        updated_dict: dict[str, Any] = {}
        combined_events: list[dict[str, Any]] = []
        changed = False
        for key, child in value.items():
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
            )
            updated_dict[key] = updated_child
            combined_events.extend(child_events)
            if updated_child is not child:
                return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        list_combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
            )
            updated_list.append(updated_child)
            list_combined_events.extend(child_events)
            if updated_child is not child:
                return (updated_list if changed else value), list_combined_events

    return value, []


def _add_aawm_dynamic_injection_logging_metadata(
    request_body: dict[str, Any], injection_events: list[dict[str, Any]]
) -> dict[str, Any]:
    proc_names = sorted(
        {
            event["proc"]
            for event in injection_events
            if isinstance(event.get("proc"), str) and event["proc"]
        }
    )
    failure_procs = sorted(
        {
            event["proc"]
            for event in injection_events
            if event.get("status") == "failed"
            and isinstance(event.get("proc"), str)
            and event["proc"]
        }
    )
    context_keys = sorted(
        {
            context_key
            for event in injection_events
            for context_key in event.get("context_keys", [])
            if isinstance(context_key, str) and context_key
        }
    )
    status_values = [
        event["status"]
        for event in injection_events
        if isinstance(event.get("status"), str) and event["status"]
    ]

    tags_to_add = ["aawm-dynamic-injection"]
    tags_to_add.extend(f"aawm-proc:{proc_name}" for proc_name in proc_names)
    if failure_procs:
        tags_to_add.append("aawm-dynamic-injection-failed")

    span_metadata: dict[str, Any] = {
        "injection_count": len(injection_events),
        "failure_count": len(failure_procs),
    }
    if proc_names:
        span_metadata["procs"] = proc_names
    if context_keys:
        span_metadata["context_keys"] = context_keys

    return _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields={
            "aawm_dynamic_injection_count": len(injection_events),
            "aawm_dynamic_injection_procs": proc_names,
            "aawm_dynamic_injection_failure_procs": failure_procs,
            "aawm_dynamic_injection_context_keys": context_keys,
            "aawm_dynamic_injection_statuses": status_values,
            "aawm_dynamic_injection_events": injection_events,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="aawm.dynamic_injection",
                    metadata=span_metadata,
                )
            ],
        },
    )


async def _expand_aawm_dynamic_directives_in_anthropic_request_body(
    request_body: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available_context = _build_aawm_context_for_anthropic_request(request_body)
    span_started_at = datetime.now(timezone.utc)
    updated_body = dict(request_body)
    injection_events: list[dict[str, Any]] = []
    changed = False

    for top_level_key in ("system", "messages"):
        if top_level_key not in request_body:
            continue
        updated_value, value_events = await _expand_aawm_dynamic_directives_in_value(
            request_body[top_level_key],
            available_context,
        )
        if value_events:
            updated_body[top_level_key] = updated_value
            injection_events.extend(value_events)
    
    if not injection_events:
        return request_body, []

    updated_body = _add_aawm_dynamic_injection_logging_metadata(
        updated_body,
        injection_events,
    )
    if changed:
        litellm_metadata = updated_body.get("litellm_metadata")
        if isinstance(litellm_metadata, dict):
            langfuse_spans = litellm_metadata.get("langfuse_spans")
            if isinstance(langfuse_spans, list):
                for span_descriptor in langfuse_spans:
                    if (
                        isinstance(span_descriptor, dict)
                        and span_descriptor.get("name") == "aawm.dynamic_injection"
                    ):
                        span_descriptor["start_time"] = _format_langfuse_span_timestamp(
                            span_started_at
                        )
                        span_descriptor["end_time"] = _format_langfuse_span_timestamp(
                            datetime.now(timezone.utc)
                        )
    return updated_body, injection_events


def _validate_anthropic_tool_blocks_for_passthrough(
    request_body: dict[str, Any],
) -> None:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_use_id = block.get("id")
                if not isinstance(tool_use_id, str) or not tool_use_id.strip():
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Invalid Anthropic tool_use block at "
                            f"messages.{message_index}.content.{content_index}: "
                            "missing required non-empty string tool_use.id"
                        ),
                    )
                continue
            if block_type != "tool_result" and not (
                isinstance(block_type, str) and block_type.endswith("_tool_result")
            ):
                continue
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id.strip():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid Anthropic tool_result block at "
                        f"messages.{message_index}.content.{content_index}: "
                        "missing required non-empty string "
                        f"tool_result.tool_use_id for block type {block_type!r}"
                    ),
                )


def _repair_anthropic_tool_use_ids_for_passthrough(
    request_body: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return request_body, 0

    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    repaired_messages, repaired_count = (
        LiteLLMAnthropicMessagesAdapter.repair_missing_anthropic_tool_use_ids(messages)
    )
    if repaired_count == 0:
        return request_body, 0

    updated_body = dict(request_body)
    updated_body["messages"] = repaired_messages
    return _merge_litellm_metadata(
        updated_body,
        tags_to_add=["anthropic-tool-use-id-repaired"],
        extra_fields={"anthropic_tool_use_id_repaired_count": repaired_count},
    ), repaired_count


async def _prepare_anthropic_request_body_for_passthrough(
    request: Request, request_body: dict[str, Any]
) -> Tuple[dict[str, Any], int, set[str], dict[str, str]]:
    updated_body, expanded_count, hooks, _source_metadata_items = (
        _expand_claude_persisted_output_in_anthropic_request_body(request_body)
    )
    billing_header_fields = _extract_anthropic_billing_header_fields_from_request_body(
        updated_body
    )
    (
        updated_body,
        _claude_system_prompt_override_events,
        _claude_prompt_patch_events,
    ) = await _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
        updated_body,
        billing_header_fields,
    )
    updated_body, _aawm_injection_events = (
        await _aawm_expand_aawm_dynamic_directives_in_anthropic_request_body(
            updated_body
        )
    )
    updated_body = _aawm_add_claude_post_rewrite_context_file_logging_metadata(
        updated_body
    )
    updated_body, _web_search_domain_filter_sanitized_count = (
        _sanitize_anthropic_web_search_empty_domain_lists(updated_body)
    )
    updated_body = _add_claude_child_agent_observability_metadata(
        updated_body,
        explicit_tenant_id=_get_aawm_tenant_header(request),
    )
    if billing_header_fields:
        updated_body = _add_anthropic_billing_header_logging_metadata(
            updated_body,
            billing_header_fields,
        )
    updated_body = _add_route_family_logging_metadata(
        updated_body, "anthropic_messages"
    )
    updated_body = _add_claude_request_breakout_logging_metadata(updated_body)
    updated_body = _prepare_request_body_for_passthrough_observability(
        request=request,
        request_body=updated_body,
    )
    updated_body, _repaired_tool_use_id_count = (
        _repair_anthropic_tool_use_ids_for_passthrough(updated_body)
    )
    _validate_anthropic_tool_blocks_for_passthrough(updated_body)
    return updated_body, expanded_count, hooks, billing_header_fields


def _request_uses_codex_native_auth(request: Request) -> bool:
    headers = _safe_get_request_headers(request)
    chatgpt_account_id = headers.get("chatgpt-account-id") or headers.get(
        "ChatGPT-Account-Id"
    )
    originator = headers.get("originator") or headers.get("Originator")
    user_agent = headers.get("user-agent") or headers.get("User-Agent")
    session_id = headers.get("session_id") or headers.get("Session_Id")

    if isinstance(chatgpt_account_id, str) and len(chatgpt_account_id) > 0:
        return True
    if isinstance(originator, str) and "codex" in originator.lower():
        return True
    return bool(
        isinstance(user_agent, str)
        and "codex" in user_agent.lower()
        and isinstance(session_id, str)
        and len(session_id) > 0
    )


def _should_preserve_openai_client_auth(request: Request, endpoint: str) -> bool:
    """
    Preserve inbound client auth only for OpenAI Responses passthrough traffic.

    This keeps Codex-style already-authenticated requests as close to native
    behavior as possible while leaving the existing server-authenticated
    passthrough behavior intact for other OpenAI endpoints.
    """
    return _is_openai_responses_endpoint(endpoint) and _request_has_openai_client_auth(
        request
    )


def _get_openai_passthrough_target_base(request: Request, endpoint: str) -> str:
    if _should_preserve_openai_client_auth(request=request, endpoint=endpoint):
        if _request_uses_codex_native_auth(request):
            return os.getenv("CHATGPT_API_BASE") or CHATGPT_API_BASE
    return os.getenv("OPENAI_API_BASE") or "https://api.openai.com/"


def _is_gemini_code_assist_endpoint(endpoint: str) -> bool:
    normalized_endpoint = endpoint.lstrip("/")
    return normalized_endpoint.startswith("v1internal:")


def _get_gemini_passthrough_target_base(
    endpoint: str,
    has_google_oauth_bearer: bool,
) -> str:
    if has_google_oauth_bearer and _is_gemini_code_assist_endpoint(endpoint):
        return os.getenv("CODE_ASSIST_ENDPOINT") or "https://cloudcode-pa.googleapis.com"

    return os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"


def _iter_antigravity_auth_file_path_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen_paths: set[str] = set()
    for env_name in (
        *_ANTIGRAVITY_MANAGED_AUTH_FILE_ENV_VARS,
        *_ANTIGRAVITY_AUTH_FILE_ENV_VARS,
        *_ANTIGRAVITY_SEED_AUTH_FILE_ENV_VARS,
    ):
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidates.append(path)

    for candidate_str in _ANTIGRAVITY_DEFAULT_AUTH_PATHS:
        candidate = Path(candidate_str).expanduser()
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidates.append(candidate)
    return candidates


def _get_antigravity_auth_file_path() -> Optional[Path]:
    candidates = _iter_antigravity_auth_file_path_candidates()
    if not candidates:
        return None
    return candidates[0]



async def _load_antigravity_oauth_token_data_from_path(
    auth_path: Path,
) -> AntigravityOAuthTokenData:
    try:
        token_data = json.loads(auth_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Antigravity OAuth token data from {auth_path}: {exc}",
        ) from exc

    if not isinstance(token_data, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Antigravity OAuth token data at {auth_path} is not a JSON object.",
        )

    return token_data


async def _load_local_antigravity_oauth_token_data() -> tuple[AntigravityOAuthTokenData, Path]:
    candidates = _iter_antigravity_auth_file_path_candidates()
    if not candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity passthrough requires local OAuth token data at "
                "'~/.gemini/antigravity-cli/antigravity-oauth-token', "
                "'LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE', "
                "'LITELLM_ANTIGRAVITY_SEED_AUTH_FILE', or "
                "'LITELLM_ANTIGRAVITY_AUTH_FILE'."
            ),
        )

    first_loaded: Optional[tuple[AntigravityOAuthTokenData, Path]] = None
    first_error: Optional[HTTPException] = None
    for auth_path in candidates:
        try:
            token_data = await _load_antigravity_oauth_token_data_from_path(auth_path)
        except HTTPException as exc:
            if first_error is None:
                first_error = exc
            continue
        if first_loaded is None:
            first_loaded = (token_data, auth_path)
        if _antigravity_access_token_is_valid(token_data):
            return token_data, auth_path

    if first_loaded is not None:
        return first_loaded
    if first_error is not None:
        raise first_error
    raise HTTPException(
        status_code=500,
        detail="Antigravity OAuth token candidate paths could not be loaded.",
    )


def _parse_antigravity_token_expiry(expiry: Any) -> Optional[datetime]:
    if not isinstance(expiry, str) or not expiry.strip():
        return None
    cleaned = expiry.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _antigravity_access_token_is_valid(token_data: dict[str, Any]) -> bool:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        return False
    access_token = _clean_codex_auth_value(token_block.get("access_token"))
    if access_token is None:
        return False
    expiry = _parse_antigravity_token_expiry(token_block.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc) + timedelta(seconds=60)


def _antigravity_access_token_is_unexpired(token_data: dict[str, Any]) -> bool:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        return False
    access_token = _clean_codex_auth_value(token_block.get("access_token"))
    if access_token is None:
        return False
    expiry = _parse_antigravity_token_expiry(token_block.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc)


def _antigravity_oauth_cached_token_is_valid(cached_token: tuple[str, int]) -> bool:
    _access_token, expiry_date = cached_token
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return expiry_date > now_ms + 60_000


def _get_antigravity_oauth_expiry_date(token_data: dict[str, Any]) -> Optional[int]:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        return None
    expiry = _parse_antigravity_token_expiry(token_block.get("expiry"))
    if expiry is None:
        return None
    return int(expiry.timestamp() * 1000)


def _iter_antigravity_cli_binary_candidates() -> list[Path]:
    candidate_files: list[Path] = []
    seen_paths: set[str] = set()
    for env_name in _ANTIGRAVITY_CLI_BINARY_PATH_ENV_VARS:
        raw_value = _clean_codex_auth_value(os.getenv(env_name))
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                candidate_files.append(candidate)

    for candidate_str in _ANTIGRAVITY_DEFAULT_CLI_BINARY_PATHS:
        candidate = Path(candidate_str).expanduser()
        if not candidate.is_file():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_files.append(candidate)
    return candidate_files


def _extract_antigravity_oauth_client_values_from_cli_text(
    cli_text: str,
) -> tuple[Optional[str], Optional[str]]:
    candidates = _extract_antigravity_oauth_client_value_candidates_from_cli_text(
        cli_text
    )
    if not candidates:
        return None, None
    return candidates[0]


def _add_antigravity_oauth_client_candidate(
    candidates: list[tuple[str, str]],
    seen: set[tuple[str, str]],
    client_id: Optional[str],
    client_secret: Optional[str],
) -> None:
    if not client_id or not client_secret:
        return
    candidate = (client_id, client_secret)
    if candidate in seen:
        return
    seen.add(candidate)
    candidates.append(candidate)


def _extract_antigravity_oauth_client_value_candidates_from_cli_text(
    cli_text: str,
) -> list[tuple[str, str]]:
    client_secret_matches = list(
        _ANTIGRAVITY_CLI_OAUTH_CLIENT_SECRET_VALUE_PATTERN.finditer(cli_text)
    )
    client_id_matches = list(
        _ANTIGRAVITY_CLI_OAUTH_CLIENT_ID_VALUE_PATTERN.finditer(cli_text)
    )
    if not client_secret_matches or not client_id_matches:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for client_secret_match in client_secret_matches:
        client_secret = _clean_codex_auth_value(
            client_secret_match.group("value")
        )
        for client_id_match in sorted(
            client_id_matches,
            key=lambda match: abs(match.start() - client_secret_match.start()),
        ):
            _add_antigravity_oauth_client_candidate(
                candidates,
                seen,
                _clean_codex_auth_value(client_id_match.group("value")),
                client_secret,
            )
    return candidates


def _load_antigravity_oauth_client_values_from_local_cli_binary(
) -> tuple[Optional[str], Optional[str]]:
    candidates = _load_antigravity_oauth_client_value_candidates_from_local_cli_binary()
    if not candidates:
        return None, None
    return candidates[0]


def _load_antigravity_oauth_client_value_candidates_from_local_cli_binary(
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in _iter_antigravity_cli_binary_candidates():
        try:
            cli_text = candidate.read_bytes().decode("latin1", errors="ignore")
        except OSError:
            continue
        client_value_candidates = (
            _extract_antigravity_oauth_client_value_candidates_from_cli_text(cli_text)
        )
        for client_id, client_secret in client_value_candidates:
            _add_antigravity_oauth_client_candidate(
                candidates,
                seen,
                client_id,
                client_secret,
            )
    return candidates


def _get_antigravity_oauth_client_value_from_token_data(
    token_data: AntigravityOAuthTokenData,
    candidate_keys: tuple[str, ...],
) -> Optional[str]:
    for key in candidate_keys:
        value = _clean_codex_auth_value(token_data.get(key))
        if value is not None:
            return value
    return None


def _get_antigravity_oauth_client_value_candidates(
    token_data: AntigravityOAuthTokenData,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    env_client_id = _get_first_secret_value(_ANTIGRAVITY_OAUTH_CLIENT_ID_ENV_VARS)
    env_client_secret = _get_first_secret_value(
        _ANTIGRAVITY_OAUTH_CLIENT_SECRET_ENV_VARS
    )
    token_client_id = _get_antigravity_oauth_client_value_from_token_data(
        token_data,
        ("client_id", "clientId"),
    )
    token_client_secret = _get_antigravity_oauth_client_value_from_token_data(
        token_data,
        ("client_secret", "clientSecret"),
    )
    _add_antigravity_oauth_client_candidate(
        candidates,
        seen,
        env_client_id,
        env_client_secret,
    )
    _add_antigravity_oauth_client_candidate(
        candidates,
        seen,
        token_client_id,
        token_client_secret,
    )
    for client_id, client_secret in (
        _load_antigravity_oauth_client_value_candidates_from_local_cli_binary()
    ):
        _add_antigravity_oauth_client_candidate(
            candidates,
            seen,
            client_id,
            client_secret,
        )
    return candidates


def _get_oauth_token_error_code(response: httpx.Response) -> Optional[str]:
    try:
        response_body = response.json()
    except ValueError:
        return None
    if not isinstance(response_body, dict):
        return None
    return _clean_codex_auth_value(response_body.get("error"))


def _format_oauth_refresh_failure_detail(
    *,
    provider_label: str,
    response: httpx.Response,
) -> str:
    error_code = _get_oauth_token_error_code(response)
    suffix = (
        f"status={response.status_code}, error={error_code}"
        if error_code
        else f"status={response.status_code}"
    )
    return (
        f"Failed to refresh {provider_label} OAuth access token ({suffix}). "
        f"Re-authenticate {provider_label} CLI or configure valid OAuth client "
        "environment overrides."
    )


def _write_antigravity_oauth_token_data_atomic(
    auth_path: Path,
    token_data: AntigravityOAuthTokenData,
) -> None:
    _write_json_file_atomic(
        auth_path,
        token_data,
        failure_label="Antigravity OAuth token",
    )


def _get_antigravity_cli_refresh_home(auth_path: Path) -> Optional[Path]:
    parts = auth_path.expanduser().parts
    if len(parts) < 4:
        return None
    if parts[-3:] != (
        ".gemini",
        "antigravity-cli",
        "antigravity-oauth-token",
    ):
        return None
    return auth_path.expanduser().parents[2]


def _get_antigravity_cli_refresh_timeout_seconds() -> float:
    raw_value = _clean_codex_auth_value(
        os.getenv("AAWM_ANTIGRAVITY_CLI_REFRESH_TIMEOUT_SECONDS")
    )
    if raw_value is None:
        return 30.0
    try:
        parsed = float(raw_value)
    except ValueError:
        return 30.0
    return max(parsed, 1.0)


async def _refresh_local_antigravity_oauth_token_data_via_cli(
    auth_path: Path,
    original_token_data: Optional[AntigravityOAuthTokenData] = None,
) -> AntigravityOAuthTokenData:
    refresh_home = _get_antigravity_cli_refresh_home(auth_path)
    cli_candidates = _iter_antigravity_cli_binary_candidates()
    if refresh_home is None or not cli_candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth direct refresh failed and AGY CLI silent "
                "refresh is unavailable for this auth-file path."
            ),
        )

    log_path = Path(os.getenv("TMPDIR") or "/tmp") / (
        f"litellm-antigravity-refresh-{os.getpid()}-{time.monotonic_ns()}.log"
    )
    env = dict(os.environ)
    env["HOME"] = str(refresh_home)
    try:
        process = await asyncio.create_subprocess_exec(
            str(cli_candidates[0]),
            "--log-file",
            str(log_path),
            "models",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        await asyncio.wait_for(
            process.communicate(),
            timeout=_get_antigravity_cli_refresh_timeout_seconds(),
        )
    except asyncio.TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(OSError, RuntimeError):
            await process.wait()
        raise HTTPException(
            status_code=500,
            detail="AGY CLI silent auth refresh timed out.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AGY CLI silent auth refresh failed ({type(exc).__name__}).",
        ) from exc
    finally:
        with contextlib.suppress(OSError):
            log_path.unlink()

    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(
                "AGY CLI silent auth refresh failed. Re-authenticate "
                "Antigravity CLI before using Antigravity passthrough."
            ),
        )

    refreshed_token_data = await _load_antigravity_oauth_token_data_from_path(auth_path)
    if not _antigravity_access_token_is_valid(refreshed_token_data):
        if (
            original_token_data is not None
            and refreshed_token_data == original_token_data
            and _antigravity_access_token_is_unexpired(refreshed_token_data)
        ):
            return refreshed_token_data
        raise HTTPException(
            status_code=500,
            detail="AGY CLI silent auth refresh did not produce a valid token.",
        )
    return refreshed_token_data


async def _refresh_local_antigravity_oauth_token_data(
    token_data: AntigravityOAuthTokenData,
    auth_path: Optional[Path] = None,
) -> AntigravityOAuthTokenData:
    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth token data does not contain a token object."
            ),
        )
    refresh_token = _clean_codex_auth_value(token_block.get("refresh_token"))
    if refresh_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth token data does not contain a refresh_token. "
                "Re-authenticate Antigravity CLI before using Antigravity passthrough."
            ),
        )

    client_candidates = _get_antigravity_oauth_client_value_candidates(token_data)
    if not client_candidates:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth token data does not contain client_id/client_secret "
                "and no fallback env vars or Antigravity CLI binary values were found."
            ),
        )

    response: Optional[httpx.Response] = None
    async with httpx.AsyncClient(timeout=30.0) as client:
        for client_id, client_secret in client_candidates:
            response = await client.post(
                _ANTHROPIC_ADAPTER_GEMINI_OAUTH_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status_code == 200:
                break
            error_code = _get_oauth_token_error_code(response)
            if error_code not in {
                "invalid_client",
                "invalid_grant",
                "unauthorized_client",
            }:
                break

    if response is None or response.status_code != 200:
        if (
            response is not None
            and auth_path is not None
            and _get_oauth_token_error_code(response)
            in {"invalid_client", "invalid_grant", "unauthorized_client"}
        ):
            return await _refresh_local_antigravity_oauth_token_data_via_cli(
                auth_path,
                token_data,
            )
        raise HTTPException(
            status_code=500,
            detail=_format_oauth_refresh_failure_detail(
                provider_label="Antigravity",
                response=response,
            )
            if response is not None
            else "Failed to refresh Antigravity OAuth access token.",
        )

    refreshed = response.json()
    refreshed_access_token = _clean_codex_auth_value(refreshed.get("access_token"))
    if refreshed_access_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Antigravity OAuth refresh response did not contain an access_token."
            ),
        )
    expires_in = refreshed.get("expires_in")
    if not isinstance(expires_in, (int, float)):
        raise HTTPException(
            status_code=500,
            detail="Antigravity OAuth refresh response did not contain expires_in.",
        )
    expiry = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

    updated_token_block = dict(token_block)
    updated_token_block.update(refreshed)
    updated_token_block["access_token"] = refreshed_access_token
    updated_token_block["refresh_token"] = refresh_token
    updated_token_block["expiry"] = expiry.isoformat()

    updated_token_data = dict(token_data)
    updated_token_data["token"] = updated_token_block
    _invalidate_codex_auto_agent_antigravity_lane_cache()
    return updated_token_data


async def _load_valid_local_antigravity_access_token() -> str:
    token_data, auth_path = await _load_local_antigravity_oauth_token_data()
    cache_key = str(auth_path.expanduser())
    cached_token = _antigravity_oauth_access_token_cache.get(cache_key)
    if cached_token is not None and _antigravity_oauth_cached_token_is_valid(
        cached_token
    ):
        return cached_token[0]

    async with _antigravity_oauth_access_token_lock:
        cached_token = _antigravity_oauth_access_token_cache.get(cache_key)
        if cached_token is not None and _antigravity_oauth_cached_token_is_valid(
            cached_token
        ):
            return cached_token[0]

        token_data, auth_path = await _load_local_antigravity_oauth_token_data()
        if not _antigravity_access_token_is_valid(token_data):
            raise HTTPException(
                status_code=500,
                detail=(
                    "Antigravity OAuth token is expired or invalid. The "
                    "provider-status sidecar owns Antigravity auth refresh; "
                    "confirm the sidecar can write the configured token file "
                    f"and refresh {auth_path}."
                ),
            )

    token_block = token_data.get("token")
    if not isinstance(token_block, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Antigravity OAuth token data at {auth_path} does not contain "
                "a token object."
            ),
        )
    access_token = _clean_codex_auth_value(token_block.get("access_token"))
    if access_token is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Antigravity OAuth token data at {auth_path} does not contain "
                "an access_token."
            ),
        )
    expiry_date = _get_antigravity_oauth_expiry_date(token_data)
    if expiry_date is not None:
        _antigravity_oauth_access_token_cache[cache_key] = (access_token, expiry_date)
    _invalidate_codex_auto_agent_antigravity_lane_cache()
    return access_token


def _get_antigravity_passthrough_target_base() -> str:
    return (
        os.getenv("ANTIGRAVITY_CODE_ASSIST_ENDPOINT")
        or os.getenv("ANTIGRAVITY_CLI_CODE_ASSIST_ENDPOINT")
        or _ANTIGRAVITY_CODE_ASSIST_DEFAULT_BASE_URL
    )


def _get_antigravity_client_header() -> str:
    return (
        _clean_codex_auth_value(os.getenv("AAWM_ANTIGRAVITY_CLIENT_HEADER"))
        or _ANTIGRAVITY_CLIENT_HEADER_DEFAULT
    )


def _build_antigravity_native_headers(access_token: str) -> dict[str, str]:
    client_header = _get_antigravity_client_header()
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": client_header,
        "x-goog-api-client": client_header,
        "Accept": "application/json",
    }


def _request_has_google_oauth_bearer(request: Request) -> bool:
    authorization = request.headers.get("authorization", "").strip()
    return authorization.lower().startswith("bearer ya29.")


def _get_antigravity_litellm_auth_header(request: Request) -> str:
    header_key = request.headers.get("x-litellm-api-key")
    if header_key:
        return _format_litellm_passthrough_api_key(header_key)

    query_key = request.query_params.get("key")
    if query_key:
        return _format_litellm_passthrough_api_key(query_key)

    return request.headers.get("authorization", "")


def _prepare_antigravity_request_body_for_passthrough(
    *,
    request: Request,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    updated_body = _merge_litellm_metadata(
        request_body,
        tags_to_add=["antigravity-code-assist", "route:antigravity_code_assist"],
        extra_fields={
            "client_name": "antigravity-cli",
            "antigravity_code_assist": True,
            "passthrough_route_family": "antigravity_code_assist",
        },
    )
    return _prepare_request_body_for_passthrough_observability(
        request=request,
        request_body=updated_body,
    )


def _get_antigravity_request_project(
    request_body: AntigravityPassthroughRequestBody,
) -> Optional[str]:
    return _clean_codex_auth_value(request_body.get("project"))


def _get_antigravity_passthrough_logging_metadata(
    request: Request,
) -> PassthroughLoggingMetadata:
    logging_body = _prepare_antigravity_request_body_for_passthrough(
        request=request,
        request_body={},
    )
    litellm_metadata = logging_body.get("litellm_metadata")
    if isinstance(litellm_metadata, dict):
        return dict(litellm_metadata)
    return {}


def _normalize_antigravity_endpoint_for_target(endpoint: str) -> str:
    normalized_endpoint = endpoint.split("?", 1)[0].lstrip("/")
    if not normalized_endpoint:
        return "/"
    return f"/{normalized_endpoint}"


def _join_antigravity_passthrough_url(base_target_url: str, endpoint: str) -> str:
    endpoint_path = _normalize_antigravity_endpoint_for_target(endpoint)
    base_url = httpx.URL(base_target_url)
    base_path = base_url.path.rstrip("/")
    if base_path:
        endpoint_path = f"{base_path}/{endpoint_path.lstrip('/')}"
    return str(base_url.copy_with(path=endpoint_path))


def _is_antigravity_streaming_endpoint(endpoint: str, request: Request) -> bool:
    normalized_endpoint = endpoint.lstrip("/")
    return "streamGenerateContent" in normalized_endpoint or (
        str(request.query_params.get("alt", "")).lower() == "sse"
    )


def _get_grok_passthrough_target_base() -> str:
    return (
        os.getenv("GROK_CLI_CHAT_PROXY_UPSTREAM_BASE_URL")
        or os.getenv("XAI_CLI_CHAT_PROXY_BASE_URL")
        or _GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL
    )


def _normalize_grok_endpoint_for_target(endpoint: str, base_target_url: str) -> str:
    normalized_endpoint = httpx.URL(endpoint).path
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = "/" + normalized_endpoint

    base_url = httpx.URL(base_target_url)
    base_path = base_url.path.rstrip("/")
    if base_path.endswith("/v1") and normalized_endpoint.startswith("/v1/"):
        normalized_endpoint = normalized_endpoint[len("/v1") :]
    return normalized_endpoint


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

    return request.headers.get("Authorization", "")


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
        normalized_model_override = (
            normalize_grok_native_oauth_model(model_override) or model_override
        )
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
    prepared_body, _grok_unsupported_request_params = (
        _drop_unsupported_codex_request_params_from_request_body(prepared_body)
    )
    prepared_body, _grok_unsupported_input_items = (
        _drop_unsupported_codex_input_items_from_request_body(prepared_body)
    )
    prepared_body, _removed_tool_choice = (
        _drop_tool_choice_without_tools_from_request_body(prepared_body)
    )
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
    return (
        not content_type
        or "application/json" in content_type
        or content_type.endswith("+json")
    )


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


def _normalize_grok_endpoint_path(endpoint: str) -> str:
    endpoint_path = httpx.URL(endpoint).path
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    if endpoint_path.startswith("/v1/"):
        endpoint_path = endpoint_path[len("/v1") :]
    return endpoint_path


def _get_grok_side_channel_endpoint_type(endpoint: str) -> Optional[str]:
    endpoint_path = _normalize_grok_endpoint_path(endpoint)
    if endpoint_path == "/sessions/register":
        return "sessions_register"
    if endpoint_path.startswith("/sessions/") and endpoint_path.endswith(
        "/replicas/update"
    ):
        return "sessions_replicas_update"
    if endpoint_path.startswith("/sessions/") and endpoint_path.endswith(
        "/signals"
    ):
        return "sessions_signals"
    if endpoint_path.startswith("/sessions/") and endpoint_path.endswith(
        "/turn-deltas"
    ):
        return "sessions_turn_deltas"
    if endpoint_path == "/traces":
        return "traces"
    return None


def _get_grok_session_side_channel_endpoint_type(endpoint: str) -> Optional[str]:
    return _get_grok_side_channel_endpoint_type(endpoint)


def _get_grok_side_channel_endpoint_path_template(
    endpoint_type: str,
) -> Optional[str]:
    if endpoint_type == "sessions_register":
        return "/sessions/register"
    if endpoint_type == "sessions_replicas_update":
        return "/sessions/{session_id}/replicas/update"
    if endpoint_type == "sessions_signals":
        return "/sessions/{session_id}/signals"
    if endpoint_type == "sessions_turn_deltas":
        return "/sessions/{session_id}/turn-deltas"
    if endpoint_type == "traces":
        return "/traces"
    return None


def _get_grok_session_side_channel_endpoint_path_template(
    endpoint_type: str,
) -> Optional[str]:
    return _get_grok_side_channel_endpoint_path_template(endpoint_type)


def _json_shape_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _extract_redacted_grok_json_request_shape(parsed_body: Any) -> dict[str, Any]:
    if isinstance(parsed_body, dict):
        top_level_key_types = {
            str(key): _json_shape_type_name(parsed_body.get(key))
            for key in sorted(parsed_body.keys(), key=str)
            if str(key) != "litellm_metadata"
        }
        return {
            "json_container_type": "object",
            "top_level_key_types": top_level_key_types,
        }
    if isinstance(parsed_body, list):
        return {
            "json_container_type": "array",
            "array_length": len(parsed_body),
        }
    if parsed_body is None:
        return {"json_container_type": "null"}
    return {"json_container_type": _json_shape_type_name(parsed_body)}


def _stable_grok_side_channel_body_digest(
    *,
    parsed_body: Any = None,
    raw_body: Optional[bytes] = None,
) -> tuple[int, str, str]:
    if raw_body is not None:
        body_bytes = raw_body
        digest_source = "raw_body"
    elif isinstance(parsed_body, dict):
        upstream_body = {
            key: value
            for key, value in parsed_body.items()
            if str(key) != "litellm_metadata"
        }
        body_bytes = json.dumps(
            upstream_body,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest_source = "canonical_json_without_litellm_metadata"
    elif isinstance(parsed_body, list):
        body_bytes = json.dumps(
            parsed_body,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest_source = "canonical_json"
    else:
        body_bytes = b""
        digest_source = "empty_body"

    return len(body_bytes), hashlib.sha256(body_bytes).hexdigest(), digest_source


def _build_grok_side_channel_request_shape_metadata(
    *,
    endpoint: str,
    request: Request,
    parsed_body: Any = None,
    raw_body: Optional[bytes] = None,
) -> Optional[dict[str, Any]]:
    endpoint_type = _get_grok_side_channel_endpoint_type(endpoint)
    if endpoint_type is None:
        return None

    content_type = request.headers.get("content-type")
    body_byte_length, body_sha256, digest_source = (
        _stable_grok_side_channel_body_digest(
            parsed_body=parsed_body,
            raw_body=raw_body,
        )
    )
    json_shape = _extract_redacted_grok_json_request_shape(parsed_body)

    metadata: dict[str, Any] = {
        "grok_side_channel": True,
        "grok_side_channel_endpoint_type": endpoint_type,
        "grok_side_channel_endpoint_path_template": (
            _get_grok_side_channel_endpoint_path_template(endpoint_type)
        ),
        "grok_side_channel_request_content_type": content_type,
        "grok_side_channel_request_body_byte_length": body_byte_length,
        "grok_side_channel_request_body_sha256": body_sha256,
        "grok_side_channel_request_body_digest_source": digest_source,
        "grok_side_channel_request_json_container_type": json_shape.get(
            "json_container_type"
        ),
    }
    if "top_level_key_types" in json_shape:
        metadata["grok_side_channel_request_top_level_key_types"] = json_shape[
            "top_level_key_types"
        ]
    if "array_length" in json_shape:
        metadata["grok_side_channel_request_array_length"] = json_shape["array_length"]

    return metadata


def _merge_grok_side_channel_shape_into_passthrough_logging_metadata(
    passthrough_logging_metadata: dict[str, Any],
    *,
    shape_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not shape_metadata:
        return passthrough_logging_metadata
    merged = dict(passthrough_logging_metadata)
    merged.update(shape_metadata)
    tags = list(merged.get("tags") or [])
    if "grok-side-channel" not in tags:
        tags.append("grok-side-channel")
    merged["tags"] = tags
    return merged


def _get_grok_side_channel_retryable_status_codes(endpoint: str) -> list[int]:
    is_session_side_channel = (
        _get_grok_side_channel_endpoint_type(endpoint) is not None
    )
    if not is_session_side_channel:
        return []

    return [500, 502, 503, 504]


def _log_grok_forward_header_compare(
    *,
    endpoint: str,
    request: Request,
) -> None:
    incoming_headers = {
        str(header_name).lower()
        for header_name in _safe_get_request_headers(request).keys()
    }
    allowed_headers = {header.lower() for header in _GROK_CLI_FORWARD_HEADER_ALLOWLIST}
    forwarded_headers = sorted(incoming_headers & allowed_headers)
    stripped_headers = sorted(
        header
        for header in incoming_headers - allowed_headers
        if header not in _GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE
        and not header.startswith("x-pass-")
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


def is_passthrough_request_using_router_model(
    request_body: dict, llm_router: Optional[litellm.Router]
) -> bool:
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
        raise HTTPException(
            status_code=404, detail=f"Provider {custom_llm_provider} not found"
        )

    base_target_url = provider_config.get_api_base()

    if base_target_url is None:
        raise HTTPException(
            status_code=404, detail=f"Provider {custom_llm_provider} api base not found"
        )

    if _is_gemini_code_assist_endpoint(endpoint):
        encoded_endpoint = endpoint.split("?", 1)[0]
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
    google_ai_studio_api_key = request.query_params.get("key") or request.headers.get(
        "x-goog-api-key"
    )

    user_api_key_dict = await user_api_key_auth(
        request=request, api_key=f"Bearer {google_ai_studio_api_key}"
    )

    _auth_header = request.headers.get("authorization", "")
    _is_google_oauth = _auth_header.startswith("Bearer ya29.")

    base_target_url = _get_gemini_passthrough_target_base(
        endpoint=endpoint,
        has_google_oauth_bearer=_is_google_oauth,
    )
    if _is_gemini_code_assist_endpoint(endpoint):
        encoded_endpoint = endpoint.split("?", 1)[0]
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
                if isinstance(request_body, dict)
                and isinstance(request_body.get("request"), dict)
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
        prepared_request_body = _add_gemini_request_breakout_logging_metadata(
            request_body
        )
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
        allowed_forward_headers=(
            list(_GEMINI_OAUTH_FORWARD_HEADER_ALLOWLIST)
            if _is_google_oauth
            else None
        ),
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
    query_params = {
        key: value
        for key, value in dict(request.query_params).items()
        if str(key).lower() != "key"
    }

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
        local_antigravity_access_token = (
            await _load_valid_local_antigravity_access_token()
        )
        custom_headers = _build_antigravity_native_headers(local_antigravity_access_token)

    target_url = _join_antigravity_passthrough_url(
        base_target_url=_get_antigravity_passthrough_target_base(),
        endpoint=endpoint,
    )
    query_params = {
        key: value
        for key, value in dict(request.query_params).items()
        if str(key).lower() != "key"
    }

    custom_body: Optional[dict[str, Any]] = None
    passthrough_logging_metadata = _get_antigravity_passthrough_logging_metadata(
        request
    )
    if request.method in {"POST", "PUT", "PATCH"}:
        request_body = await get_request_body(request)
        if isinstance(request_body, dict):
            custom_body = _prepare_antigravity_request_body_for_passthrough(
                request=request,
                request_body=request_body,
            )
            request_project = _get_antigravity_request_project(request_body)
            if (
                local_antigravity_access_token is not None
                and request_project is not None
            ):
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
                        litellm_metadata["google_retrieve_user_quota"] = (
                            google_quota_observation
                        )
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
        allowed_forward_headers=(
            list(_ANTIGRAVITY_FORWARD_HEADER_ALLOWLIST)
            if has_google_oauth_bearer
            else None
        ),
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
    is_coding_data_retention_endpoint = (
        _is_grok_coding_data_retention_endpoint(endpoint)
    )
    raw_body_passthrough = request.method in {"POST", "PUT", "PATCH"} and (
        is_storage_endpoint
        or is_coding_data_retention_endpoint
        or not _is_grok_json_request(request)
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
                    grok_model_override = normalize_grok_native_oauth_model(
                        custom_metadata.get("grok_model_override")
                    )
                    if (
                        grok_model_override is not None
                        and not _get_case_insensitive_header(
                            _safe_get_request_headers(request),
                            "x-grok-model-override",
                        )
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
        passthrough_logging_metadata = (
            _merge_grok_side_channel_shape_into_passthrough_logging_metadata(
                passthrough_logging_metadata,
                shape_metadata=side_channel_shape_metadata,
            )
        )
        side_channel_shape_log = (
            verbose_proxy_logger.info
            if os.getenv("AAWM_GROK_ROUTE_DEBUG") == "1"
            else verbose_proxy_logger.debug
        )
        side_channel_shape_log(
            "Grok passthrough side-channel request shape: endpoint_type=%s body_byte_length=%s body_sha256=%s json_container_type=%s top_level_key_types=%s",
            side_channel_shape_metadata.get("grok_side_channel_endpoint_type"),
            side_channel_shape_metadata.get(
                "grok_side_channel_request_body_byte_length"
            ),
            side_channel_shape_metadata.get("grok_side_channel_request_body_sha256"),
            side_channel_shape_metadata.get(
                "grok_side_channel_request_json_container_type"
            ),
            side_channel_shape_metadata.get(
                "grok_side_channel_request_top_level_key_types"
            ),
        )

    query_params = {
        key: value
        for key, value in dict(request.query_params).items()
        if str(key).lower() != "key"
    }
    grok_side_channel_retryable_status_codes = (
        _get_grok_side_channel_retryable_status_codes(endpoint)
    )

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
    is_router_model = is_passthrough_request_using_router_model(
        request_body, llm_router
    )
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
                json=(
                    request_body
                    if request.headers.get("content-type") == "application/json"
                    else None
                ),
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

    provider_config = ProviderConfigManager.get_provider_vector_stores_config(
        provider=LlmProviders.MILVUS
    )
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
        (
            litellm.vector_store_index_registry.get_vector_store_index_by_name(
                vector_store_index_name=collection_name
            )
        )
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
    auth_credentials = provider_config.get_auth_credentials(
        litellm_params=litellm_params
    )

    extra_headers = auth_credentials.get("headers") or {}

    litellm_params = vector_store.get("litellm_params") or {}

    base_target_url = provider_config.get_complete_url(
        api_base=litellm_params.get("api_base"), litellm_params=litellm_params
    )

    if base_target_url is None:
        raise Exception(
            f"api_base not found in vector store configuration for {vector_store_name}"
        )

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
    if candidate["provider"] == _CODEX_AUTO_AGENT_NATIVE_PROVIDER:
        response = await _handle_anthropic_openai_responses_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            use_alias_candidate_probe=True,
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER:
        response = await _handle_anthropic_google_completion_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
            use_alias_candidate_probe=True,
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_GOOGLE_PROVIDER:
        response = await _handle_anthropic_google_completion_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            use_alias_candidate_probe=True,
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        if candidate.get("route_family") == "anthropic_openrouter_completion_adapter":
            response = await _handle_anthropic_openrouter_completion_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=candidate_body,
                adapter_model=candidate["model"],
            )
        else:
            response = await _handle_anthropic_openrouter_responses_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=candidate_body,
                adapter_model=candidate["model"],
                use_alias_candidate_probe=True,
            )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        if candidate.get("route_family") == "anthropic_xai_oauth_responses_adapter":
            response = await _handle_anthropic_xai_oauth_responses_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=candidate_body,
                adapter_model=candidate["model"],
                use_alias_candidate_probe=True,
            )
        else:
            response = await _handle_anthropic_grok_native_oauth_responses_adapter_route(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                prepared_request_body=candidate_body,
                adapter_model=candidate["model"],
                use_alias_candidate_probe=True,
            )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        response = await _handle_anthropic_opencode_zen_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            use_alias_candidate_probe=True,
        )
    else:
        _safe_set_request_parsed_body(request, candidate_body)
        response = await _perform_anthropic_native_passthrough_request(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            target_url=target_url,
            custom_headers=custom_headers,
        )
    return response


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
    attempts: list[dict[str, Any]] = []
    last_retryable_exc: Optional[Exception] = None
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(
        prepared_request_body
    )
    alias_model = (
        _normalize_anthropic_auto_agent_alias_model(prepared_request_body.get("model"))
        or _ANTHROPIC_AUTO_AGENT_MODEL_ALIAS
    )

    for _attempt_number in range(
        len(_get_anthropic_auto_agent_candidates_for_alias(alias_model))
    ):
        try:
            selection = await _select_anthropic_auto_agent_candidate(
                request=request,
                request_body=prepared_request_body,
            )
        except HTTPException as exc:
            if exc.status_code == 429:
                _emit_auto_agent_alias_no_candidate_event(
                    alias_family="anthropic_auto_agent",
                    alias_model=alias_model,
                    request=request,
                    request_body=prepared_request_body,
                    exc=exc,
                )
            raise
        candidate = selection["candidate"]
        attempt_record = _codex_auto_agent_candidate_public_shape(
            candidate,
            lane_key=selection.get("lane_key"),
            reason=selection.get("selection_reason"),
        )
        attempts.append(attempt_record)
        candidate_body = _record_auto_agent_alias_attempt_started(
            alias_family="anthropic_auto_agent",
            alias_model=alias_model,
            request=request,
            prepared_request_body=prepared_request_body,
            selection=selection,
            attempts=attempts,
            attempt_record=attempt_record,
            add_alias_metadata_fn=_add_anthropic_auto_agent_alias_metadata,
        )

        try:
            response = await _perform_anthropic_auto_agent_alias_candidate_request(
                endpoint=endpoint,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                candidate=candidate,
                candidate_body=candidate_body,
                target_url=target_url,
                custom_headers=custom_headers,
            )
            await _set_anthropic_auto_agent_session_affinity(
                selection.get("session_key"),
                candidate,
            )
            return response
        except Exception as exc:
            error_class = _classify_codex_auto_agent_retryable_exhaustion(exc)
            if error_class is None:
                raise
            last_retryable_exc = exc
            cooldown_seconds = _get_codex_auto_agent_cooldown_seconds(exc)
            await _set_anthropic_auto_agent_cooldown(
                selection["cooldown_key"],
                cooldown_seconds,
            )
            error_tokens = _update_codex_auto_agent_retryable_attempt_record(
                attempt_record=attempt_record,
                exc=exc,
                error_class=error_class,
                cooldown_seconds=cooldown_seconds,
                cooldown_scope="candidate",
            )
            if has_continuation_state:
                attempt_record["status"] = "terminal_in_flight_cooldown_set"
                _record_auto_agent_alias_attempt_failure(
                    alias_family="anthropic_auto_agent",
                    alias_model=alias_model,
                    request=request,
                    prepared_request_body=prepared_request_body,
                    selection=selection,
                    attempts=attempts,
                    attempt_record=attempt_record,
                    error_class=error_class,
                    add_alias_metadata_fn=_add_anthropic_auto_agent_alias_metadata,
                    redispatch_required=True,
                )
                verbose_proxy_logger.warning(
                    "Anthropic auto-agent alias %s target %s/%s hit %s "
                    "for an in-flight session on attempt %s; signaling redispatch",
                    alias_model,
                    candidate["provider"],
                    candidate["model"],
                    error_class,
                    len(attempts),
                )
                _raise_anthropic_auto_agent_redispatch_required(
                    candidate=candidate,
                    lane_key=selection.get("lane_key"),
                    cooldown_seconds=cooldown_seconds,
                    error_tokens=error_tokens,
                    alias_model=alias_model,
                )
            _record_auto_agent_alias_attempt_failure(
                alias_family="anthropic_auto_agent",
                alias_model=alias_model,
                request=request,
                prepared_request_body=prepared_request_body,
                selection=selection,
                attempts=attempts,
                attempt_record=attempt_record,
                error_class=error_class,
                add_alias_metadata_fn=_add_anthropic_auto_agent_alias_metadata,
            )
            verbose_proxy_logger.warning(
                "Anthropic auto-agent alias %s target %s/%s hit %s on attempt %s; "
                "cooldown %.1fs tokens=%s",
                alias_model,
                candidate["provider"],
                candidate["model"],
                error_class,
                len(attempts),
                cooldown_seconds,
                sorted(error_tokens),
            )
            continue

    if last_retryable_exc is not None:
        raise last_retryable_exc
    raise HTTPException(
        status_code=429,
        detail="No Anthropic auto-agent alias candidates were available.",
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
        (
            header_name
            for header_name in headers
            if str(header_name).lower() == _ANTHROPIC_BETA_HEADER_NAME
        ),
        None,
    )
    existing_beta = (
        headers.pop(existing_header_name)
        if existing_header_name is not None
        else None
    )
    if existing_beta is None:
        headers[_ANTHROPIC_BETA_HEADER_NAME] = beta_value
        return headers

    existing_values = [
        value.strip()
        for value in str(existing_beta).split(",")
        if value.strip()
    ]
    if beta_value not in existing_values:
        existing_values.append(beta_value)
    headers[_ANTHROPIC_BETA_HEADER_NAME] = ", ".join(existing_values)
    return headers


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
    "/anthropic/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Anthropic Pass-through", "pass-through"],
)

async def anthropic_proxy_route(
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
    if (
        "authorization" not in request.headers
        and "x-api-key" not in request.headers
        and anthropic_api_key is not None
    ):
        custom_headers["x-api-key"] = "{}".format(anthropic_api_key)

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
            prepared_request_body, _anthropic_read_guidance_changes = (
                _apply_aawm_read_agent_guidance_to_request_body(
                    prepared_request_body,
                    alias_model=anthropic_auto_agent_alias,
                    target_field="system",
                )
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
            custom_headers,
            normalized_context_1m_model,
        ) = (
            _prepare_anthropic_context_1m_native_passthrough(
                request=request,
                request_body=prepared_request_body,
                custom_headers=custom_headers,
            )
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
            raise HTTPException(
                status_code=400, detail={"error": "Model is required in request body"}
            )

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
        verbose_proxy_logger.error(
            f"BedrockError in handle_bedrock_count_tokens: {str(e)}"
        )
        raise HTTPException(status_code=e.status_code, detail={"error": e.message})
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error in handle_bedrock_count_tokens: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": f"CountTokens processing error: {str(e)}"}
        )


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
    is_router_model = is_passthrough_request_using_router_model(
        request_body={"model": model}, llm_router=llm_router
    )

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
    verbose_proxy_logger.debug(
        f"Bedrock passthrough: Using direct Bedrock model '{model}' for endpoint '{endpoint}'"
    )

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
        base_target_url = (
            f"https://bedrock-agent-runtime.{aws_region_name}.amazonaws.com"
        )
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
    _request = AWSRequest(
        method="POST", url=str(updated_url), data=json.dumps(data), headers=headers
    )
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
        deployment = llm_router.get_available_deployment_for_pass_through(
            model=model_id
        )
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
            actual_model, custom_llm_provider, _, _ = get_llm_provider(
                model=model_from_config
            )

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
        verbose_proxy_logger.debug(
            f"Error resolving vertex model from router for model {model_id}: {e}"
        )

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
    assembly_region = AssemblyAIPassthroughLoggingHandler._get_assembly_region_from_url(
        url=str(request.url)
    )
    base_target_url = (
        AssemblyAIPassthroughLoggingHandler._get_assembly_base_url_from_region(
            region=assembly_region
        )
    )
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
                (
                    litellm.vector_store_index_registry.is_vector_store_index(
                        vector_store_index_name=part
                    )
                )
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
                    json=(
                        request_body
                        if request.headers.get("content-type") == "application/json"
                        else None
                    ),
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
                provider_config = (
                    ProviderConfigManager.get_provider_vector_stores_config(
                        provider=litellm.LlmProviders.AZURE_AI
                    )
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
                    (
                        litellm.vector_store_index_registry.get_vector_store_index_by_name(
                            vector_store_index_name=part
                        )
                    )
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
                auth_credentials = provider_config.get_auth_credentials(
                    litellm_params=litellm_params
                )

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
        raise Exception(
            "Required 'AZURE_API_BASE' in environment to make pass-through calls to Azure."
        )
    # Add or update query parameters
    azure_api_key = passthrough_endpoint_router.get_credentials(
        custom_llm_provider=litellm.LlmProviders.AZURE.value,
        region_name=None,
    )
    if azure_api_key is None:
        raise Exception(
            "Required 'AZURE_API_KEY' in environment to make pass-through calls to Azure."
        )

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
    def update_base_target_url_with_credential_location(
        base_target_url: str, vertex_location: Optional[str]
    ) -> str:
        pass


class VertexAIDiscoveryPassThroughHandler(BaseVertexAIPassThroughHandler):
    @staticmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        return "https://discoveryengine.googleapis.com/"

    @staticmethod
    def update_base_target_url_with_credential_location(
        base_target_url: str, vertex_location: Optional[str]
    ) -> str:
        return base_target_url


class VertexAIPassThroughHandler(BaseVertexAIPassThroughHandler):
    @staticmethod
    def get_default_base_target_url(vertex_location: Optional[str]) -> str:
        return get_vertex_base_url(vertex_location)

    @staticmethod
    def update_base_target_url_with_credential_location(
        base_target_url: str, vertex_location: Optional[str]
    ) -> str:
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

    verbose_proxy_logger.debug(
        "Using vector store credentials to override vertex project and location"
    )

    litellm_params = router_credentials.get("litellm_params", {})
    if not litellm_params:
        verbose_proxy_logger.warning(
            "Vector store credentials found but litellm_params is empty"
        )
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
        verbose_proxy_logger.warning(
            "Vector store credentials found but missing vertex_project in litellm_params"
        )

    if vector_store_location:
        verbose_proxy_logger.debug(
            "Overriding vertex_location from URL (%s) with vector store value: %s",
            vertex_location,
            vector_store_location,
        )
        vertex_location = vector_store_location
    else:
        verbose_proxy_logger.warning(
            "Vector store credentials found but missing vertex_location in litellm_params"
        )

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
    if (
        vertex_credentials is None or vertex_credentials.vertex_project is None
    ) and router_credentials is None:
        headers = _safe_get_request_headers(request).copy()
        headers_passed_through = True
        verbose_proxy_logger.debug(
            "default_vertex_config  not set, incoming request headers %s", headers
        )
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
    api_key_to_use = get_litellm_virtual_key(request=request)
    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=api_key_to_use,
    )

    if user_api_key_dict is None:
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

    base_target_url = get_vertex_pass_through_handler.get_default_base_target_url(
        vertex_location
    )

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
        verbose_proxy_logger.debug(
            "Extracted vector store ID from endpoint: %s", vector_store_id
        )

        # Retrieve vector store credentials from the registry
        vector_store_credentials = (
            passthrough_endpoint_router.get_vector_store_credentials(
                vector_store_id=vector_store_id
            )
        )

        if vector_store_credentials:
            verbose_proxy_logger.debug(
                "Found vector store credentials for ID: %s", vector_store_id
            )
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
        is_grok_native_oauth_request = (
            _is_openai_responses_endpoint(endpoint)
            and _is_grok_native_oauth_request_body(request_body)
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
            raise Exception(
                "Required 'OPENAI_API_KEY' in environment to make pass-through calls to OpenAI."
            )

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
        retryable_upstream_status_codes=[429],
        caller_managed_hidden_retry=False,
    )


async def _perform_codex_auto_agent_grok_native_responses_request(
    *,
    endpoint: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, Any],
) -> Response:
    try:
        grok_context = await BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
            endpoint=endpoint,
            request=request,
            request_body=request_body,
            extra_headers={},
        )
    except Exception as exc:
        if _grok_native_candidate_unavailable_detail(exc) is not None:
            _raise_grok_native_auto_agent_candidate_unavailable(exc)
        raise
    if grok_context is None:
        _raise_grok_native_auto_agent_candidate_unavailable(
            Exception(
                "Grok native Codex auto-agent candidate requires a managed "
                "Grok OIDC credential."
            )
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
    )


async def _perform_codex_auto_agent_oa_xai_responses_request(
    *,
    endpoint: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, Any],
) -> Response:
    try:
        oa_xai_context = (
            await BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                endpoint=endpoint,
                request_body=request_body,
            )
        )
    except Exception as exc:
        if _xai_oauth_candidate_unavailable_detail(exc) is not None:
            _raise_xai_oauth_auto_agent_candidate_unavailable(exc)
        raise
    if oa_xai_context is None:
        _raise_xai_oauth_auto_agent_candidate_unavailable(
            Exception(
                "Codex auto-agent xAI OAuth candidate requires a managed xAI "
                "OAuth credential."
            )
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
    )


async def _validate_codex_auto_agent_openrouter_responses_stream(
    response: StreamingResponse,
    *,
    adapter_model: str,
) -> StreamingResponse:
    buffered_chunks: list[Any] = []
    event_summaries: list[dict[str, Any]] = []

    async def _recording_iterator() -> Any:
        async for raw_chunk in response.body_iterator:
            buffered_chunks.append(raw_chunk)
            yield raw_chunk

    recording_response = StreamingResponse(
        _recording_iterator(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )
    try:
        response_body = await _collect_responses_response_from_stream(
            recording_response,
            event_summaries=event_summaries,
        )
    except HTTPException as exc:
        if (
            exc.status_code == 502
            and str(exc.detail)
            == "OpenAI Responses stream completed without a response payload."
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
    if _is_failed_responses_body(response_body):
        _raise_codex_auto_agent_failed_responses_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_responses",
            adapter_label="OpenRouter",
            stream_event_summaries=event_summaries,
        )

    async def _replay_iterator() -> Any:
        for raw_chunk in buffered_chunks:
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
) -> Response:
    openrouter_api_key = _get_openrouter_api_key()
    if openrouter_api_key is None:
        exc = ProxyException(
            message=(
                "OpenRouter Codex auto-agent candidate requires "
                "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
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
        )
    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
        try:
            response_body = json.loads(_decode_http_response_body(response.body))
        except Exception:
            return response
        if (
            isinstance(response_body, dict)
            and _is_codex_auto_agent_empty_success_responses_body(response_body)
        ):
            _raise_codex_auto_agent_empty_success_response(
                response_body=response_body,
                adapter_model=adapter_model,
            )
        if isinstance(response_body, dict) and _is_failed_responses_body(response_body):
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter="codex_auto_agent_openrouter_responses",
                adapter_label="OpenRouter",
            )
    return response


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
    requested_model = prepared_request_body.get("model")
    request_body = copy.deepcopy(prepared_request_body)
    request_body["model"] = adapter_model
    removed_format = request_body.pop("format", None)
    request_body = _strip_opencode_zen_unsupported_responses_tools(request_body)
    request_body = _add_opencode_zen_logging_metadata(
        request_body,
        route_family="codex_opencode_zen_adapter",
        tag_prefix="codex-opencode-zen-adapter",
        requested_model=requested_model,
        adapter_model=adapter_model,
        input_shape="openai_responses",
        output_shape="openai_responses",
    )
    target_endpoint = "opencode_zen:/v1/chat/completions"
    tags_to_add = [
        f"codex-adapter-model:{adapter_model}",
        f"codex-adapter-target:{target_endpoint}",
    ]
    extra_fields: dict[str, Any] = {
        "codex_adapter_model": adapter_model,
        "codex_adapter_original_model": requested_model,
        "codex_adapter_provider": _OPENCODE_ZEN_PROVIDER,
        "codex_adapter_target_endpoint": target_endpoint,
    }
    if removed_format is not None:
        tags_to_add.append("opencode-zen-unsupported-format-stripped")
        extra_fields["opencode_zen_removed_unsupported_format"] = removed_format
    request_body = _merge_litellm_metadata(
        request_body,
        tags_to_add=tags_to_add,
        extra_fields=extra_fields,
    )
    request_input = request_body.get("input") or ""
    responses_api_request = cast(
        ResponsesAPIOptionalRequestParams,
        {
            key: value
            for key, value in request_body.items()
            if key not in {"input", "model", "litellm_metadata"}
        },
    )
    litellm_metadata = dict(request_body.get("litellm_metadata") or {})
    completion_kwargs = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=adapter_model,
        input=request_input,
        responses_api_request=responses_api_request,
        custom_llm_provider=litellm.LlmProviders.OPENAI.value,
        stream=bool(request_body.get("stream")),
        metadata=litellm_metadata,
    )
    completion_kwargs["metadata"] = litellm_metadata
    previous_response_id = responses_api_request.get("previous_response_id")
    if previous_response_id:
        completion_kwargs = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id=str(previous_response_id),
            litellm_completion_request=completion_kwargs,
        )
    completion_kwargs, chat_message_sanitization_changes = (
        _sanitize_opencode_zen_completion_messages_for_chat_completion(
            completion_kwargs
        )
    )
    if chat_message_sanitization_changes:
        metadata_body = _merge_litellm_metadata(
            {"litellm_metadata": litellm_metadata},
            tags_to_add=["opencode-zen-chat-tool-adjacency-sanitized"],
            extra_fields={
                **chat_message_sanitization_changes,
                "langfuse_spans": [
                    _build_langfuse_span_descriptor(
                        name="opencode_zen.chat_tool_adjacency_sanitized",
                        metadata=chat_message_sanitization_changes,
                    )
                ],
            },
        )
        litellm_metadata = dict(metadata_body.get("litellm_metadata") or {})
        request_body["litellm_metadata"] = litellm_metadata
        completion_kwargs["metadata"] = litellm_metadata

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
        if (
            use_alias_candidate_probe
            and _opencode_zen_candidate_unavailable_detail(exc) is not None
        ):
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
                )
            ),
            media_type="text/event-stream",
        )

    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=completion_response,
        request_input=request_input,
        responses_api_request=responses_api_request,
    )
    response_body = json.loads(
        _serialize_responses_adapter_response(responses_api_response)
    )
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_opencode_zen_completion_adapter",
            adapter_label="OpenCode Zen chat-completions",
        )
    return _build_responses_response_from_adapter_response(responses_api_response)


async def _perform_codex_auto_agent_openrouter_completion_request(
    *,
    request: Request,
    adapter_model: str,
    request_body: dict[str, Any],
) -> Response:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    openrouter_api_key = _get_openrouter_api_key()
    if openrouter_api_key is None:
        exc = ProxyException(
            message=(
                "OpenRouter Codex auto-agent candidate requires "
                "AAWM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
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
    upstream_adapter_model = (
        _get_openrouter_completion_adapter_upstream_model(adapter_model)
        or adapter_model
    )
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
        {
            key: value
            for key, value in request_body.items()
            if key not in {"input", "model", "litellm_metadata"}
        },
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
    request_body, completion_kwargs, litellm_metadata = (
        _apply_openrouter_completion_message_sanitization(
            request_body=request_body,
            completion_kwargs=completion_kwargs,
            litellm_metadata=litellm_metadata,
            span_name="codex_openrouter.chat_message_shape_sanitized",
            tag="openrouter-chat-message-shape-sanitized",
        )
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
                )
            ),
            media_type="text/event-stream",
        )

    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        chat_completion_response=completion_response,
        request_input=request_input,
        responses_api_request=responses_api_request,
    )
    response_body = json.loads(
        _serialize_responses_adapter_response(responses_api_response)
    )
    if _is_codex_auto_agent_empty_success_responses_body(response_body):
        _raise_codex_auto_agent_empty_success_response(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter="codex_auto_agent_openrouter_completion_adapter",
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
    if candidate["provider"] == _CODEX_AUTO_AGENT_GOOGLE_PROVIDER:
        response = await _handle_codex_google_code_assist_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            use_alias_candidate_probe=True,
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_ANTIGRAVITY_PROVIDER:
        response = await _handle_codex_google_code_assist_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            adapter_provider=_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER,
            use_alias_candidate_probe=True,
        )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENROUTER_PROVIDER:
        if candidate.get("route_family") == "codex_openrouter_completion_adapter":
            response = await _perform_codex_auto_agent_openrouter_completion_request(
                request=request,
                adapter_model=candidate["model"],
                request_body=candidate_body,
            )
        else:
            response = await _perform_codex_auto_agent_openrouter_responses_request(
                endpoint=endpoint,
                request=request,
                user_api_key_dict=user_api_key_dict,
                adapter_model=candidate["model"],
                request_body=candidate_body,
            )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_XAI_PROVIDER:
        if candidate.get("route_family") == "codex_xai_oauth_responses_adapter":
            response = await _perform_codex_auto_agent_oa_xai_responses_request(
                endpoint=endpoint,
                request=request,
                user_api_key_dict=user_api_key_dict,
                request_body=candidate_body,
            )
        else:
            response = await _perform_codex_auto_agent_grok_native_responses_request(
                endpoint=endpoint,
                request=request,
                user_api_key_dict=user_api_key_dict,
                request_body=candidate_body,
            )
    elif candidate["provider"] == _CODEX_AUTO_AGENT_OPENCODE_PROVIDER:
        response = await _handle_codex_opencode_zen_adapter_route(
            endpoint=endpoint,
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            prepared_request_body=candidate_body,
            adapter_model=candidate["model"],
            use_alias_candidate_probe=True,
        )
    else:
        response = await _perform_codex_auto_agent_native_openai_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            target_url=target_url,
            api_key=api_key,
            forward_headers=forward_headers,
            request_body=candidate_body,
        )
    return response


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
    attempts: list[dict[str, Any]] = []
    last_retryable_exc: Optional[Exception] = None
    has_continuation_state = _codex_auto_agent_request_has_continuation_state(
        prepared_request_body
    )
    alias_model = (
        _normalize_codex_auto_agent_alias_model(prepared_request_body.get("model"))
        or _CODEX_AUTO_AGENT_MODEL_ALIAS
    )

    for _attempt_number in range(
        len(_get_codex_auto_agent_candidates_for_alias(alias_model))
    ):
        try:
            selection = await _select_codex_auto_agent_candidate(
                request=request,
                request_body=prepared_request_body,
            )
        except HTTPException as exc:
            if exc.status_code == 429:
                _emit_auto_agent_alias_no_candidate_event(
                    alias_family="codex_auto_agent",
                    alias_model=alias_model,
                    request=request,
                    request_body=prepared_request_body,
                    exc=exc,
                )
            raise
        candidate = selection["candidate"]
        attempt_record = _codex_auto_agent_candidate_public_shape(
            candidate,
            lane_key=selection.get("lane_key"),
            reason=selection.get("selection_reason"),
        )
        attempts.append(attempt_record)
        candidate_body = _record_auto_agent_alias_attempt_started(
            alias_family="codex_auto_agent",
            alias_model=alias_model,
            request=request,
            prepared_request_body=prepared_request_body,
            selection=selection,
            attempts=attempts,
            attempt_record=attempt_record,
            add_alias_metadata_fn=_add_codex_auto_agent_alias_metadata,
        )

        try:
            response = await _perform_codex_auto_agent_alias_candidate_request(
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
            await _set_codex_auto_agent_session_affinity(
                selection.get("session_key"),
                candidate,
            )
            return response
        except Exception as exc:
            error_class = _classify_codex_auto_agent_retryable_exhaustion(exc)
            if error_class is None:
                raise
            last_retryable_exc = exc
            cooldown_seconds = _get_codex_auto_agent_cooldown_seconds(exc)
            cooldown_scope = await _set_codex_auto_agent_candidate_cooldowns(
                candidate=candidate,
                lane_key=selection.get("lane_key"),
                selected_cooldown_key=selection["cooldown_key"],
                cooldown_seconds=cooldown_seconds,
                exc=exc,
            )
            error_tokens = _update_codex_auto_agent_retryable_attempt_record(
                attempt_record=attempt_record,
                exc=exc,
                error_class=error_class,
                cooldown_seconds=cooldown_seconds,
                cooldown_scope=cooldown_scope,
            )
            if has_continuation_state:
                attempt_record["status"] = "terminal_in_flight_cooldown_set"
                _record_auto_agent_alias_attempt_failure(
                    alias_family="codex_auto_agent",
                    alias_model=alias_model,
                    request=request,
                    prepared_request_body=prepared_request_body,
                    selection=selection,
                    attempts=attempts,
                    attempt_record=attempt_record,
                    error_class=error_class,
                    add_alias_metadata_fn=_add_codex_auto_agent_alias_metadata,
                    redispatch_required=True,
                )
                verbose_proxy_logger.warning(
                    "Codex auto-agent alias %s target %s/%s hit %s "
                    "for an in-flight session on attempt %s; signaling redispatch",
                    alias_model,
                    candidate["provider"],
                    candidate["model"],
                    error_class,
                    len(attempts),
                )
                _raise_codex_auto_agent_redispatch_required(
                    candidate=candidate,
                    lane_key=selection.get("lane_key"),
                    cooldown_seconds=cooldown_seconds,
                    error_tokens=error_tokens,
                    alias_model=alias_model,
                )
            _record_auto_agent_alias_attempt_failure(
                alias_family="codex_auto_agent",
                alias_model=alias_model,
                request=request,
                prepared_request_body=prepared_request_body,
                selection=selection,
                attempts=attempts,
                attempt_record=attempt_record,
                error_class=error_class,
                add_alias_metadata_fn=_add_codex_auto_agent_alias_metadata,
            )
            verbose_proxy_logger.warning(
                "Codex auto-agent alias %s target %s/%s hit %s on attempt %s; "
                "cooldown %.1fs scope=%s tokens=%s",
                alias_model,
                candidate["provider"],
                candidate["model"],
                error_class,
                len(attempts),
                cooldown_seconds,
                cooldown_scope,
                sorted(error_tokens),
            )
            continue

    if last_retryable_exc is not None:
        raise last_retryable_exc
    raise HTTPException(
        status_code=429,
        detail="No Codex auto-agent alias candidates were available.",
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
            raise Exception(
                "OpenAI passthrough requests for xAI OAuth models require a managed xAI OAuth credential."
            )

        request_body["model"] = _to_xai_native_passthrough_model(
            request_body.get("model")
        )
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
                "openai_passthrough_route_family": (
                    _get_openai_passthrough_route_family(endpoint)
                ),
                "grok_native_entrypoint": "openai_responses",
            },
        )
        if not prepared_grok_native:
            return None
        if grok_target_base_url is None:
            raise Exception(
                "OpenAI passthrough requests for Grok native OAuth models require a Grok target base URL."
            )

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
            is_codex_responses_request = (
                _request_uses_codex_native_auth(request)
                and _is_openai_responses_endpoint(endpoint)
            )
            if is_codex_responses_request:
                prepared_request_body = _add_route_family_logging_metadata(
                    prepared_request_body,
                    "codex_responses",
                )
                prepared_request_body, _codex_tool_description_patch_events = (
                    _apply_codex_tool_description_patches_to_request_body(
                        prepared_request_body
                    )
                )
                prepared_request_body, _codex_unsupported_hosted_tools = (
                    _drop_unsupported_codex_hosted_tools_from_request_body(
                        prepared_request_body
                    )
                )
                prepared_request_body, _codex_unsupported_request_params = (
                    _drop_unsupported_codex_request_params_from_request_body(
                        prepared_request_body
                    )
                )
                prepared_request_body, _codex_unsupported_input_items = (
                    _drop_unsupported_codex_input_items_from_request_body(
                        prepared_request_body
                    )
                )
                if _is_oa_xai_request_body(
                    prepared_request_body
                ) or _is_grok_native_oauth_request_body(prepared_request_body):
                    prepared_request_body, _codex_removed_empty_tool_choice = (
                        _drop_tool_choice_without_tools_from_request_body(
                            prepared_request_body
                        )
                    )
                prepared_request_body = _add_codex_request_breakout_logging_metadata(
                    prepared_request_body
                )
            oa_xai_context = await (
                BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                    endpoint=endpoint,
                    request_body=prepared_request_body,
                )
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
                grok_native_context = await (
                    BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
                        endpoint=endpoint,
                        request=request,
                        request_body=prepared_request_body,
                        extra_headers=extra_headers,
                    )
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
                        prepared_request_body, _codex_auto_agent_guidance_changes = (
                            _apply_codex_auto_agent_prevention_guidance_to_request_body(
                                prepared_request_body
                            )
                        )
                        prepared_request_body, _codex_read_guidance_changes = (
                            _apply_aawm_read_agent_guidance_to_request_body(
                                prepared_request_body,
                                alias_model=codex_auto_agent_alias,
                                target_field="instructions",
                            )
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
        if (
            RouteChecks._is_assistants_api_request(request) is True
            and "OpenAI-Beta" not in headers
        ):
            headers["OpenAI-Beta"] = "assistants=v2"
        return headers

    @staticmethod
    def _assemble_headers(
        api_key: Optional[str], request: Request, extra_headers: Optional[dict] = None
    ) -> dict:
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
        if (
            custom_llm_provider == litellm.LlmProviders.OPENAI
            and "/v1/" not in joined_path_str
        ):
            # Insert v1 after api.openai.com for OpenAI requests
            joined_path_str = joined_path_str.replace(
                "api.openai.com/", "api.openai.com/v1/"
            )

        return joined_path_str

    @staticmethod
    def _normalize_endpoint_for_target(
        endpoint: str, base_target_url: str
    ) -> str:
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
        if base_url.path.rstrip("/") == "/v1" and normalized_endpoint.startswith(
            "/v1/"
        ):
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
            if (
                credential.credential_info
                and credential.credential_info.get("custom_llm_provider") == "cursor"
            ):
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

    Note: This function should be registered in proxy_server.py using:
    app.websocket("/vertex_ai/live")(vertex_ai_live_websocket_passthrough)
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    _ = user_api_key_dict  # passthrough route already authenticated; avoid lint warnings

    await websocket.accept()

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
        resolved_location = resolved_location or (
            vertex_llm_base.get_default_vertex_location()
        )
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
        verbose_proxy_logger.exception(
            "Failed to prepare Vertex AI credentials for live passthrough"
        )
        # Log the authentication failure using proxy_logging_obj
        if proxy_logging_obj and user_api_key_dict:
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data={},
            )
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason="Vertex AI authentication failed")
        return

    host_location = resolved_location or vertex_llm_base.get_default_vertex_location()
    host = (
        "aiplatform.googleapis.com"
        if host_location == "global"
        else f"{host_location}-aiplatform.googleapis.com"
    )
    service_url = (
        f"wss://{host}/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )

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

    # Use the new WebSocket passthrough pattern
    if user_api_key_dict is None:
        raise ValueError("user_api_key_dict is required for WebSocket passthrough")

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
