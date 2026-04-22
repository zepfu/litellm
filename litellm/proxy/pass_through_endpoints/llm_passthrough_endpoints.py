"""
What is this?

Provider-specific Pass-Through Endpoints

Use litellm with Anthropic SDK, Vertex AI SDK, Cohere SDK, etc.
"""

import ast
import asyncio
import base64
import codecs
import glob
import importlib
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Tuple, Union, cast
from urllib.parse import quote, urlencode, urlparse

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
)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
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
    create_pass_through_route,
    create_websocket_passthrough_route,
    pass_through_request,
    websocket_passthrough_request,
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

    def _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
        request_body: dict[str, Any],
        billing_header_fields: dict[str, str] | None = None,
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
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from .passthrough_endpoint_router import PassthroughEndpointRouter

if TYPE_CHECKING:
    import asyncpg

vertex_llm_base = VertexBase()
router = APIRouter()
default_vertex_config = None

passthrough_endpoint_router = PassthroughEndpointRouter()

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
_PASSTHROUGH_SESSION_ID_HEADER_NAMES = (
    "session_id",
    "Session_Id",
    "x-session-id",
    "X-Session-Id",
)
_PASS_THROUGH_HEADER_PREFIX = "x-pass-"
_ANTHROPIC_RESPONSES_ADAPTER_ENDPOINTS = frozenset(
    {"/messages", "/v1/messages"}
)
_ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex-spark",
    }
)
_ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "openrouter/free",
        "inclusionai/ling-2.6-flash:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "minimax/minimax-m2.5:free",
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
        "gpt-oss-20b:free",
        "gpt-oss-120b:free",
        "qwen/qwen3-coder:free",
    }
)
_ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS = frozenset(
    {
        "openrouter/elephant-alpha",
    }
)
_ANTHROPIC_GOOGLE_COMPLETION_ADAPTER_ALLOWED_MODEL_PREFIXES = (
    "gemini-3.1",
    "gemini-3-flash-preview",
)
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
_aawm_dynamic_injection_pool: Optional[Any] = None
_aawm_dynamic_injection_pool_lock = asyncio.Lock()
_claude_context_replacement_template_cache: dict[Path, str] = {}
_claude_prompt_patch_manifest_cache: dict[Path, dict[str, Any]] = {}
_claude_agent_model_cache: dict[Path, tuple[Optional[int], Optional[str]]] = {}
_google_code_assist_project_cache: dict[str, str] = {}
_google_code_assist_prime_until_monotonic_by_key: dict[str, float] = {}
_google_code_assist_prime_lock = asyncio.Lock()
_google_adapter_semaphores: dict[tuple[str, int], asyncio.Semaphore] = {}
_google_adapter_rate_limit_lock = asyncio.Lock()
_google_adapter_rate_limit_until_monotonic_by_key: dict[str, float] = {}
_google_adapter_user_prompt_turn_lock = asyncio.Lock()
_google_adapter_user_prompt_turn_counters: dict[str, int] = {}
_openrouter_adapter_rate_limit_lock = asyncio.Lock()
_openrouter_adapter_rate_limit_until_monotonic_by_key: dict[str, float] = {}
_openrouter_adapter_failure_circuit_until_monotonic_by_key: dict[str, float] = {}


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
        "chatgpt": "openai",
        "gemini": "google",
    }.get(prefix, prefix if prefix in ("openai", "google", "openrouter") else None)
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
        "ling-2-6-flash": "inclusionai/ling-2.6-flash:free",
        "meta-llama/llama-3.3-70b-instructfree": (
            "meta-llama/llama-3.3-70b-instruct:free"
        ),
    }
    normalized_candidate = openrouter_model_aliases.get(
        normalized_candidate, normalized_candidate
    )
    return normalized_candidate or None


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
    except Exception as exc:
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


def _get_google_oauth_client_value(
    auth_data: dict[str, Any], candidate_keys: tuple[str, ...], env_var_names: tuple[str, ...]
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

    return updated_auth_data


async def _load_valid_local_google_oauth_access_token() -> str:
    auth_data, _auth_path = await _load_local_google_oauth_credentials()
    if not _google_oauth_token_is_valid(auth_data):
        auth_data = await _refresh_local_google_oauth_credentials(auth_data)

    access_token = _clean_codex_auth_value(auth_data.get("access_token"))
    if access_token is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini OAuth credentials did not yield a valid access_token.",
        )
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


async def _get_or_load_google_code_assist_project(
    access_token: str,
) -> str:
    cache_key = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
    cached_project = _google_code_assist_project_cache.get(cache_key)
    if isinstance(cached_project, str) and cached_project:
        return cached_project

    target_base = _get_anthropic_adapter_google_target_base()
    load_url = f"{target_base.rstrip('/')}/v1internal:loadCodeAssist"
    request_body = {
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        },
    }
    headers = _build_google_adapter_native_headers(
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

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load Google Code Assist project for Anthropic adapter models: "
                f"{response.text}"
            ),
        )

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
        allowed_tool_names = {"Read", "Write", "Edit", "Glob", "Grep", "Bash"}

    aliases = _get_google_code_assist_native_tool_aliases()
    expanded_tool_names = set(allowed_tool_names)
    for tool_name in list(allowed_tool_names):
        alias_name = aliases.get(tool_name)
        if isinstance(alias_name, str) and alias_name:
            expanded_tool_names.add(alias_name)

    return expanded_tool_names


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
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                updated_parts.append(part)
                continue
            text_value = part["text"]
            original_text_chars += len(text_value)
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
                compacted_count += 1
                pure_context_compacted_count += 1
                if is_subagent_context:
                    subagent_context_compacted_count += 1
                part_changed = True
                updated_part = dict(part)
                updated_part["text"] = text_value[:reminder_only_context_cap].rstrip()
                compacted_text_chars += len(updated_part["text"])
                updated_parts.append(updated_part)
                continue
            if len(text_value) <= cap:
                compacted_text_chars += len(text_value)
                updated_parts.append(part)
                continue
            compacted_count += 1
            part_changed = True
            prefix = text_value[:head_keep].rstrip()
            suffix = text_value[-tail_keep:].lstrip()
            compacted_text = (
                f"{prefix}\n\n"
                f"[Gemini adapter compacted oversized user text from {len(text_value)} chars to preserve head/tail context.]\n\n"
                f"{suffix}"
            )
            compacted_text_chars += len(compacted_text)
            updated_part = dict(part)
            updated_part["text"] = compacted_text
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
    if protected_text_indices:
        selected_indices = sorted(set(protected_text_indices + selected_indices))
        if len(selected_indices) > max_window:
            selected_indices = selected_indices[-max_window:]
            if not set(protected_text_indices).issubset(set(selected_indices)):
                tail_count = max(0, max_window - len(protected_text_indices))
                selected_indices = sorted(set(protected_text_indices + selected_indices[-tail_count:]))
                if len(selected_indices) > max_window:
                    selected_indices = selected_indices[-max_window:]

    trimmed_contents = [contents[idx] for idx in selected_indices]
    protected_positions = {
        pos for pos, idx in enumerate(selected_indices) if idx in set(protected_text_indices)
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
    }


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
    if isinstance(max_output_tokens, int) and cap is not None and max_output_tokens > cap:
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


def _parse_google_rate_limit_reset_seconds(exc: Any) -> float:
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


def _extract_google_adapter_error_reason(exc: Any) -> Optional[str]:
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

    for candidate in candidate_payloads:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
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


async def _perform_google_adapter_pass_through_request(**kwargs: Any) -> Response:
    passthrough_kwargs = dict(kwargs)
    max_retries = _get_google_adapter_max_retries()
    total_attempts = max_retries + 1
    capacity_total_attempts = _get_google_adapter_model_capacity_max_retries() + 1
    hidden_retry_budget_seconds = _get_google_adapter_hidden_retry_budget_seconds()
    accumulated_hidden_wait_seconds = 0.0
    rate_limit_key = _get_google_adapter_rate_limit_key_from_kwargs(kwargs)
    passthrough_kwargs.pop("google_access_token", None)
    passthrough_kwargs.pop("google_adapter_rate_limit_key", None)
    attempt = 0
    while True:
        attempt += 1
        verbose_proxy_logger.warning(
            "Google adapter upstream attempt %s/%s",
            attempt,
            max(total_attempts, capacity_total_attempts),
        )
        await _wait_for_google_adapter_cooldown_if_needed(rate_limit_key)
        try:
            return await pass_through_request(**passthrough_kwargs)
        except Exception as exc:
            status_code = _extract_google_adapter_exception_status_code(exc)
            error_reason = _extract_google_adapter_error_reason(exc)
            is_capacity_retry = error_reason == "MODEL_CAPACITY_EXHAUSTED"
            retry_limit = capacity_total_attempts if is_capacity_retry else total_attempts
            if is_capacity_retry:
                wait_seconds = _get_google_adapter_capacity_backoff_seconds(attempt)
            else:
                wait_seconds = _parse_google_rate_limit_reset_seconds(exc)
            projected_hidden_wait_seconds = accumulated_hidden_wait_seconds + wait_seconds
            within_hidden_budget = (
                hidden_retry_budget_seconds > 0
                and projected_hidden_wait_seconds <= hidden_retry_budget_seconds
            )
            if status_code != 429 or (attempt >= retry_limit and not within_hidden_budget):
                verbose_proxy_logger.warning(
                    "Google adapter upstream attempt %s failed with %s (%s, reason=%s) and will not be retried",
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    error_reason,
                )
                raise
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
            accumulated_hidden_wait_seconds = projected_hidden_wait_seconds
            await _set_google_adapter_cooldown(rate_limit_key, wait_seconds + 1.0)


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
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("error", {}).get("metadata")
    if not isinstance(metadata, dict):
        return None
    retry_after_value = metadata.get("retry_after_seconds")
    try:
        if retry_after_value is not None:
            return max(0.0, float(retry_after_value))
    except Exception:
        return None
    return None


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


def _extract_openrouter_adapter_error_headers(exc: Any) -> dict[str, Any]:
    payload = _extract_openrouter_adapter_error_payload(exc)
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("error", {}).get("metadata")
    if not isinstance(metadata, dict):
        return {}
    headers = metadata.get("headers")
    if not isinstance(headers, dict):
        return {}
    return headers


def _get_openrouter_adapter_header_value(
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


def _extract_openrouter_adapter_reset_wait_seconds(exc: Any) -> Optional[float]:
    headers = _extract_openrouter_adapter_error_headers(exc)
    reset_value = _get_openrouter_adapter_header_value(headers, "X-RateLimit-Reset")
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
        verbose_proxy_logger.warning(
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
        verbose_proxy_logger.warning(
            "OpenRouter adapter upstream attempt %s/%s for model=%s",
            attempt,
            total_attempts,
            model_rate_limit_key,
        )
        await _wait_for_openrouter_adapter_cooldown_if_needed(wait_keys)
        try:
            result = await pass_through_request(**kwargs)
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
) -> None:
    ttl_seconds = _get_google_code_assist_prime_ttl_seconds()
    cache_key = _get_google_code_assist_prime_cache_key(
        access_token,
        companion_project,
    )
    if ttl_seconds > 0:
        async with _google_code_assist_prime_lock:
            cached_until = _google_code_assist_prime_until_monotonic_by_key.get(
                cache_key, 0.0
            )
        if cached_until > time.monotonic():
            if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
                verbose_proxy_logger.info(
                    "Google adapter prime cache hit for project=%s",
                    companion_project,
                )
            return

    target_base = _get_anthropic_adapter_google_target_base().rstrip("/")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    metadata = {
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
        "duetProject": companion_project,
    }
    preflight_requests = (
        (f"{target_base}/v1internal:retrieveUserQuota", {"project": companion_project}),
        (f"{target_base}/v1internal:fetchAdminControls", {"project": companion_project}),
        (f"{target_base}/v1internal:listExperiments", {"project": companion_project, "metadata": metadata}),
    )

    async with httpx.AsyncClient(timeout=20.0) as client:
        for url, body in preflight_requests:
            HttpPassThroughEndpointHelpers.validate_outgoing_egress(
                url=url,
                headers=headers,
                credential_family="google",
                expected_target_family="google",
            )
            try:
                await client.post(url, headers=headers, json=body)
            except Exception:
                continue
    if ttl_seconds > 0:
        async with _google_code_assist_prime_lock:
            _google_code_assist_prime_until_monotonic_by_key[cache_key] = (
                time.monotonic() + ttl_seconds
            )



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


async def _build_google_code_assist_request_from_completion_kwargs(
    *,
    completion_kwargs: dict[str, Any],
    adapter_model: str,
    project: str,
    request: Request,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )
    from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body

    google_model = _normalize_google_completion_adapter_model_name(adapter_model)
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
            extra_kwargs={
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
    if fallback_context_changes:
        completion_message_window_changes = {
            **completion_message_window_changes,
            **fallback_context_changes,
        }
    completion_message_window_changes = {
        **completion_message_window_changes,
        **native_tool_alias_changes,
        'google_adapter_session_id_source': session_id_source,
        'google_adapter_session_id_hash': hashlib.sha1(session_id.encode('utf-8')).hexdigest()[:8],
    }
    return wrapped_request, tool_name_mapping, completion_messages, gemini_optional_params, litellm_params, completion_message_window_changes


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
            if not isinstance(tool, dict):
                updated_tools.append(tool)
                continue
            if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                updated_tools.append(tool)
                continue
            function_block = dict(tool["function"])
            original_name = function_block.get("name")
            alias_name = aliases.get(original_name)
            if isinstance(alias_name, str) and alias_name:
                function_block["name"] = alias_name
                updated_tool = dict(tool)
                updated_tool["function"] = function_block
                updated_tools.append(updated_tool)
                tool_name_mapping[alias_name] = tool_name_mapping.get(original_name, original_name)
                alias_count += 1
                aliased_names.add(alias_name)
            else:
                updated_tools.append(tool)
        completion_kwargs["tools"] = updated_tools

    messages = completion_kwargs.get("messages")
    if isinstance(messages, list):
        updated_messages = []
        for message in messages:
            if not isinstance(message, dict):
                updated_messages.append(message)
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                updated_messages.append(message)
                continue
            updated_tool_calls = []
            message_changed = False
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    updated_tool_calls.append(tool_call)
                    continue
                function_block = tool_call.get("function")
                if not isinstance(function_block, dict):
                    updated_tool_calls.append(tool_call)
                    continue
                original_name = function_block.get("name")
                alias_name = aliases.get(original_name)
                if isinstance(alias_name, str) and alias_name:
                    updated_function = dict(function_block)
                    updated_function["name"] = alias_name
                    updated_tool_call = dict(tool_call)
                    updated_tool_call["function"] = updated_function
                    updated_tool_calls.append(updated_tool_call)
                    tool_name_mapping[alias_name] = tool_name_mapping.get(original_name, original_name)
                    message_changed = True
                    aliased_names.add(alias_name)
                else:
                    updated_tool_calls.append(tool_call)
            if message_changed:
                updated_message = dict(message)
                updated_message["tool_calls"] = updated_tool_calls
                updated_messages.append(updated_message)
            else:
                updated_messages.append(message)
        completion_kwargs["messages"] = updated_messages

    tool_choice = completion_kwargs.get("tool_choice")
    if isinstance(tool_choice, dict):
        updated_tool_choice = dict(tool_choice)
        function_block = updated_tool_choice.get("function")
        if isinstance(function_block, dict):
            original_name = function_block.get("name")
            alias_name = aliases.get(original_name)
            if isinstance(alias_name, str) and alias_name:
                updated_function = dict(function_block)
                updated_function["name"] = alias_name
                updated_tool_choice["function"] = updated_function
                completion_kwargs["tool_choice"] = updated_tool_choice
                tool_name_mapping[alias_name] = tool_name_mapping.get(original_name, original_name)
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
    injected_count = 0

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
            updated_messages.append(message)
            continue

        tool_names: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_block = tool_call.get("function")
            if not isinstance(function_block, dict):
                continue
            tool_name = function_block.get("name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        if tool_names:
            if len(tool_names) == 1:
                synthesized_text = f"Calling tool {tool_names[0]}."
            else:
                synthesized_text = f"Calling tools: {', '.join(tool_names)}."
        else:
            synthesized_text = "Calling tool."

        updated_message = dict(message)
        updated_message["content"] = synthesized_text
        updated_messages.append(updated_message)
        injected_count += 1

    if injected_count == 0:
        return messages, {}
    return updated_messages, {
        "google_adapter_injected_tool_call_context_count": injected_count,
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


def _apply_google_adapter_completion_message_window(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(messages) <= 2:
        return messages, {}
    max_window = _get_google_adapter_max_completion_messages_window()
    original_count = len(messages)
    original_text_chars = sum(_estimate_completion_message_text_chars(message) for message in messages)
    trimmed_messages = list(messages[-max_window:])
    trimmed_text_chars = sum(_estimate_completion_message_text_chars(message) for message in trimmed_messages)
    if len(trimmed_messages) == original_count:
        return messages, {}
    return trimmed_messages, {
        "trimmed_completion_messages_from_count": original_count,
        "trimmed_completion_messages_to_count": len(trimmed_messages),
        "trimmed_completion_messages_from_text_chars": original_text_chars,
        "trimmed_completion_messages_to_text_chars": trimmed_text_chars,
        "trimmed_completion_messages_max_window": max_window,
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
    }
    if isinstance(value, list):
        return [_normalize_google_code_assist_httpx_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = key_mapping.get(key, key)
        normalized[normalized_key] = _normalize_google_code_assist_httpx_payload(item)
    return normalized


def _summarize_google_code_assist_request_shape(payload: Any) -> dict[str, Any]:
    def _extract_text_metrics(content_block: Any) -> tuple[int, int]:
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
        contents = request_block.get("contents")
        if isinstance(contents, list):
            summary["contents_count"] = len(contents)
            content_part_count = 0
            content_text_chars = 0
            text_entry_count = 0
            preview_entries = []
            for content_entry in contents:
                parts, chars = _extract_text_metrics(content_entry)
                content_part_count += parts
                content_text_chars += chars
                if chars > 0:
                    text_entry_count += 1
            for content_entry in contents[-4:]:
                if not isinstance(content_entry, dict):
                    continue
                role = content_entry.get("role")
                parts = content_entry.get("parts")
                part_kinds = []
                text_preview = None
                preview_parts, preview_chars = _extract_text_metrics(content_entry)
                if isinstance(parts, list):
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        keys = [key for key in ("text", "functionCall", "functionResponse", "thought") if key in part]
                        if keys:
                            part_kinds.extend(keys)
                        if text_preview is None and isinstance(part.get("text"), str):
                            text_preview = part.get("text")[:120].replace("\n", "\\n")
                        function_response = part.get("functionResponse")
                        if isinstance(function_response, dict):
                            response_payload = function_response.get("response")
                            if isinstance(response_payload, dict):
                                response_keys = sorted(response_payload.keys())
                                part_kinds.append(
                                    f"functionResponseKeys:{','.join(response_keys)}"
                                )
                                if text_preview is None and isinstance(response_payload.get("content"), str):
                                    text_preview = response_payload.get("content")[:120].replace("\n", "\\n")
                preview_entries.append({
                    "role": role,
                    "part_count": preview_parts,
                    "text_chars": preview_chars,
                    "part_kinds": part_kinds,
                    "text_preview": text_preview,
                })
            summary["contents_part_count"] = content_part_count
            summary["contents_text_chars"] = content_text_chars
            summary["contents_text_entry_count"] = text_entry_count
            summary["contents_tail_preview"] = preview_entries
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
        generation_config = request_block.get("generationConfig")
        if isinstance(generation_config, dict):
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
        tool_config = request_block.get("toolConfig")
        if isinstance(tool_config, dict):
            summary["tool_config_keys"] = sorted(tool_config.keys())
        system_instruction = request_block.get("systemInstruction")
        if isinstance(system_instruction, dict):
            summary["has_system_instruction"] = True
            system_parts, system_chars = _extract_text_metrics(system_instruction)
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
        outer_payload = json.loads(response.body.decode("utf-8"))
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
        messages=completion_messages,
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
                    verbose_proxy_logger.info(
                        "Google adapter post-tool cooldown armed for %.1fs",
                        cooldown_seconds,
                    )
            yield f"data: {json.dumps(unwrapped)}\n\n"

    debug_logged = False
    post_tool_cooldown_armed = False
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

    logging_obj = SimpleNamespace(optional_params=gemini_optional_params, post_call=lambda **_: None)
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

    anthropic_response = AnthropicAdapter().translate_completion_output_params(
        model_response,
        tool_name_mapping=tool_name_mapping,
    )
    return _build_anthropic_response_from_completion_adapter_response(
        anthropic_response
    )


def _wrap_streaming_response_with_release_callback(
    response: StreamingResponse,
    release_callback: Any,
) -> StreamingResponse:
    original_iterator = response.body_iterator
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

    async def _wrapped_iterator():
        try:
            async for chunk in original_iterator:
                yield chunk
        finally:
            _release_once()

    response.body_iterator = _wrapped_iterator()
    return response


def _get_anthropic_adapter_openrouter_api_key() -> Optional[str]:
    return _get_first_secret_value(_ANTHROPIC_ADAPTER_OPENROUTER_API_KEY_ENV_VARS)


def _get_anthropic_adapter_openrouter_target_base() -> str:
    cleaned = _clean_secret_string(os.getenv("OPENROUTER_API_BASE")) or "https://openrouter.ai/api"
    if cleaned.endswith("/api/v1"):
        return cleaned[: -len("/v1")]
    return cleaned


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


def _load_local_codex_auth_headers(request: Request) -> Optional[dict[str, str]]:
    auth_path = _get_anthropic_adapter_codex_auth_file_path()
    if auth_path is None:
        return None

    try:
        auth_data = json.loads(auth_path.read_text())
    except Exception:
        return None

    token_data = auth_data.get("tokens")
    if not isinstance(token_data, dict):
        token_data = auth_data

    access_token = _clean_codex_auth_value(token_data.get("access_token"))
    if access_token is None:
        return None

    account_id = _clean_codex_auth_value(token_data.get("account_id")) or _extract_codex_account_id_from_token(
        _clean_codex_auth_value(token_data.get("id_token")) or access_token
    )

    from litellm.llms.chatgpt.common_utils import get_chatgpt_default_headers

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

    anthropic_request = AnthropicMessagesRequest(
        **{k: v for k, v in request_fields.items() if v is not None}
    )
    translation_provider = litellm.LlmProviders.OPENAI.value
    translated_body = adapter.translate_request(
        anthropic_request,
        custom_llm_provider=translation_provider,
    )

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

    existing_litellm_metadata = request_body.get("litellm_metadata")
    if isinstance(existing_litellm_metadata, dict):
        translated_body["litellm_metadata"] = dict(existing_litellm_metadata)

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
        ],
        extra_fields={
            "anthropic_adapter_model": adapter_model,
            "anthropic_adapter_original_model": request_body.get("model"),
            "anthropic_adapter_target_endpoint": target_endpoint,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name=span_name,
                    metadata=span_metadata,
                )
            ],
        },
    )


def _build_anthropic_response_from_responses_response(
    response_body: dict[str, Any],
) -> Response:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
        LiteLLMAnthropicToResponsesAPIAdapter,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse

    adapter = LiteLLMAnthropicToResponsesAPIAdapter()
    translated_response = adapter.translate_response(ResponsesAPIResponse(**response_body))
    if hasattr(translated_response, "model_dump_json"):
        serialized_response = translated_response.model_dump_json(exclude_none=True)
    elif hasattr(translated_response, "json"):
        serialized_response = translated_response.json(exclude_none=True)
    else:
        serialized_response = json.dumps(translated_response)
    return Response(
        content=serialized_response,
        media_type="application/json",
    )


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


async def _collect_responses_response_from_stream(
    response: StreamingResponse,
) -> dict[str, Any]:
    output_text_parts: list[str] = []
    event_iterator = _iterate_responses_sse_events(response.body_iterator)
    try:
        async for event in event_iterator:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str):
                    output_text_parts.append(delta)
            if event_type == "response.completed":
                response_payload = getattr(event, "response", None)
                if response_payload is None:
                    continue
                response_dict = _coerce_namespace_to_mapping(response_payload)
                if isinstance(response_dict, dict):
                    if not response_dict.get("output") and output_text_parts:
                        response_dict["output"] = [
                            {
                                "type": "message",
                                "id": "msg_adapter_0",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "".join(output_text_parts),
                                        "annotations": [],
                                    }
                                ],
                            }
                        ]
                    return response_dict
    finally:
        await event_iterator.aclose()
        body_iterator = getattr(response, "body_iterator", None)
        aclose = getattr(body_iterator, "aclose", None)
        if callable(aclose):
            await aclose()
    raise HTTPException(
        status_code=502,
        detail="OpenAI Responses stream completed without a response payload.",
    )


def _build_anthropic_streaming_response_from_responses_stream(
    response: StreamingResponse,
    *,
    model: str,
) -> StreamingResponse:
    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
        AnthropicResponsesStreamWrapper,
    )

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_iterate_responses_sse_events(response.body_iterator),
        model=model,
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


async def _handle_anthropic_google_completion_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
) -> Response:
    from litellm.litellm_core_utils.litellm_logging import Logging

    google_access_token = await _load_valid_local_google_oauth_access_token()
    google_project = await _get_or_load_google_code_assist_project(google_access_token)
    await _prime_google_code_assist_session(google_access_token, google_project)

    route_family = "anthropic_google_completion_adapter"
    requested_model = prepared_request_body.get("model")
    google_target_base = _get_anthropic_adapter_google_target_base()
    google_model = _normalize_google_completion_adapter_model_name(adapter_model)
    google_adapter_rate_limit_key = _get_google_adapter_rate_limit_key(
        google_model,
        access_token=google_access_token,
        companion_project=google_project,
    )
    client_requested_stream = bool(prepared_request_body.get("stream"))
    is_stream = True
    target_endpoint_label = "/v1internal:streamGenerateContent"
    target_query_params = {"alt": "sse"}
    target_url = f"{google_target_base.rstrip('/')}{target_endpoint_label}"
    annotated_target_url = httpx.URL(target_url).copy_with(params=target_query_params) if target_query_params else httpx.URL(target_url)

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
            "anthropic-google-completion-adapter",
            f"anthropic-adapter-model:{google_model}",
            f"anthropic-adapter-target:google:{target_endpoint_label}",
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
            "anthropic_adapter_target_endpoint": f"google:{target_endpoint_label}",
            "google_adapter_persisted_output_compacted": bool(
                google_persisted_output_compacted_count
            ),
            "google_adapter_persisted_output_compacted_count": google_persisted_output_compacted_count,
            "google_adapter_persisted_output_hooks": sorted(google_persisted_output_hooks),
            "google_adapter_persisted_output_metadata": google_persisted_output_metadata,
            "langfuse_spans": [
                _build_langfuse_span_descriptor(
                    name="anthropic.google_completion_adapter",
                    metadata={
                        "requested_model": requested_model,
                        "adapter_model": google_model,
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
        wrapped_request_body["litellm_metadata"] = dict(prepared_request_body["litellm_metadata"])

    generation_policy_changes = _apply_google_adapter_request_shape_policy(wrapped_request_body)

    adapter_headers = _build_google_adapter_native_headers(
        access_token=google_access_token,
        model=google_model,
        accept="*/*",
    )
    if isinstance(wrapped_request_body.get("litellm_metadata"), dict):
        if completion_message_window_changes:
            wrapped_request_body["litellm_metadata"]["google_adapter_completion_message_window"] = completion_message_window_changes
        if generation_policy_changes:
            wrapped_request_body["litellm_metadata"]["google_adapter_request_shape_policy"] = generation_policy_changes

    sanitized_schema_fix_count = 0
    request_payload = wrapped_request_body.get("request") if isinstance(wrapped_request_body, dict) else None
    request_tools = request_payload.get("tools") if isinstance(request_payload, dict) else None
    if isinstance(request_tools, list):
        for tool_entry in request_tools:
            if not isinstance(tool_entry, dict):
                continue
            decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
            if isinstance(decls, list):
                for declaration in decls:
                    if isinstance(declaration, dict):
                        sanitized_schema_fix_count += _sanitize_google_schema_array_items(declaration.get("parameters"))

    if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
        try:
            debug_shape = _summarize_google_code_assist_request_shape(wrapped_request_body)
            request_payload = wrapped_request_body.get("request") if isinstance(wrapped_request_body, dict) else None
            request_tools = request_payload.get("tools") if isinstance(request_payload, dict) else None
            function_names: list[str] = []
            if isinstance(request_tools, list):
                for tool_entry in request_tools:
                    if not isinstance(tool_entry, dict):
                        continue
                    decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
                    if isinstance(decls, list):
                        for declaration in decls:
                            if isinstance(declaration, dict):
                                name = declaration.get("name")
                                if isinstance(name, str):
                                    function_names.append(name)
            litellm_metadata = prepared_request_body.get("litellm_metadata") if isinstance(prepared_request_body, dict) else None
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

    HttpPassThroughEndpointHelpers.validate_outgoing_egress(
        url=str(annotated_target_url),
        headers=adapter_headers,
        credential_family="google",
        expected_target_family="google",
    )
    _annotate_request_scope_for_adapted_access_log(request, annotated_target_url)

    google_adapter_semaphore = _get_google_adapter_semaphore(
        rate_limit_key=google_adapter_rate_limit_key
    )
    await google_adapter_semaphore.acquire()
    semaphore_released = False

    def _release_google_adapter_semaphore() -> None:
        nonlocal semaphore_released
        if semaphore_released:
            return
        semaphore_released = True
        google_adapter_semaphore.release()
        if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
            verbose_proxy_logger.info(
                "Google adapter semaphore released for model=%s",
                google_model,
            )

    if os.getenv("AAWM_GEMINI_ROUTE_DEBUG") == "1":
        verbose_proxy_logger.info(
            "Google adapter semaphore acquired for model=%s stream=%s",
            google_model,
            is_stream,
        )

    stream_release_attached = False
    try:
        upstream_response = await _perform_google_adapter_pass_through_request(
            request=request,
            target=target_url,
            custom_headers=adapter_headers,
            user_api_key_dict=user_api_key_dict,
            custom_body=wrapped_request_body,
            forward_headers=False,
            query_params=target_query_params,
            stream=is_stream,
            custom_llm_provider=litellm.LlmProviders.GEMINI.value,
            egress_credential_family="google",
            expected_target_family="google",
            google_adapter_rate_limit_key=google_adapter_rate_limit_key,
        )

        if not isinstance(upstream_response, StreamingResponse):
            raise HTTPException(
                status_code=502,
                detail="Google Code Assist adapter expected a streaming response.",
            )

        if client_requested_stream:
            streaming_response = _build_anthropic_streaming_response_from_google_code_assist_stream(
                response=upstream_response,
                adapter_model=google_model,
                tool_name_mapping=tool_name_mapping,
                gemini_optional_params=gemini_optional_params,
                rate_limit_key=google_adapter_rate_limit_key,
            )
            stream_release_attached = True
            return _wrap_streaming_response_with_release_callback(
                streaming_response,
                _release_google_adapter_semaphore,
            )

        logging_obj = Logging(
            model=google_model,
            messages=completion_messages,
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id=str(uuid4()),
            function_id="anthropic_google_completion_adapter",
        )
        logging_obj.optional_params = gemini_optional_params

        return await _collect_google_code_assist_response_from_stream(
            response=upstream_response,
            adapter_model=google_model,
            tool_name_mapping=tool_name_mapping,
            logging_obj=logging_obj,
        )
    finally:
        if not is_stream or not stream_release_attached:
            _release_google_adapter_semaphore()


async def _handle_anthropic_openai_responses_adapter_route(
    *,
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    prepared_request_body: dict[str, Any],
    adapter_model: str,
) -> Response:
    client_requested_stream = bool(prepared_request_body.get("stream"))
    local_codex_headers = None
    has_client_auth = _anthropic_adapter_request_has_openai_client_auth(request)
    uses_codex_native_auth = _anthropic_adapter_request_uses_codex_native_auth(request)
    if not has_client_auth:
        local_codex_headers = _load_local_codex_auth_headers(request)

    use_chatgpt_codex_defaults = uses_codex_native_auth or local_codex_headers is not None
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        use_chatgpt_codex_defaults=use_chatgpt_codex_defaults,
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
    forward_headers = _anthropic_adapter_should_forward_direct_auth_headers(request)
    custom_headers: dict[str, Any] = {}

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
        egress_credential_family="openai" if local_codex_headers is not None else None,
        expected_target_family="openai",
    )
    _annotate_request_scope_for_adapted_access_log(request, target_url)

    if isinstance(upstream_response, StreamingResponse):
        if not client_requested_stream:
            response_body = await _collect_responses_response_from_stream(
                upstream_response
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body
            )
            translated_response.headers.update(dict(upstream_response.headers))
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from OpenAI Responses passthrough.",
        )

    response_body = json.loads(upstream_response.body.decode("utf-8"))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body
    )
    translated_response.headers.update(dict(upstream_response.headers))
    translated_response.status_code = upstream_response.status_code
    return translated_response



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
        raise Exception(
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
        adapter_model=adapter_model,
        operation=lambda: LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=int(prepared_request_body.get("max_tokens") or 1024),
            messages=prepared_request_body.get("messages") or [],
            model=adapter_model,
            metadata=prepared_request_body.get("metadata") or {},
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
            custom_llm_provider=litellm.LlmProviders.OPENROUTER.value,
            api_key=openrouter_api_key,
            api_base=f"{target_base_url.rstrip('/')}/v1",
            headers=_build_openrouter_default_headers(),
            litellm_metadata=prepared_request_body.get("litellm_metadata") or {},
            proxy_server_request={
                "headers": dict(request.headers),
                "body": prepared_request_body,
            },
            standard_callback_dynamic_params={},
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
) -> Response:
    client_requested_stream = bool(prepared_request_body.get("stream"))
    translated_request_body = _build_anthropic_responses_adapter_request_body(
        prepared_request_body,
        adapter_model=adapter_model,
        route_family="anthropic_openrouter_responses_adapter",
        tag_prefix="anthropic-openrouter-responses-adapter",
        span_name="anthropic.openrouter_responses_adapter",
        target_endpoint="openrouter:/v1/responses",
    )
    forced_tool_choice_changes = (
        _maybe_force_explicit_bash_tool_choice_for_responses_adapter(
            prepared_request_body,
            translated_request_body,
        )
    )
    if forced_tool_choice_changes:
        translated_request_body = _merge_litellm_metadata(
            translated_request_body,
            extra_fields=forced_tool_choice_changes,
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
        raise Exception(
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
        if not client_requested_stream:
            response_body = await _collect_responses_response_from_stream(
                upstream_response
            )
            translated_response = _build_anthropic_response_from_responses_response(
                response_body
            )
            translated_response.headers.update(dict(upstream_response.headers))
            translated_response.status_code = upstream_response.status_code
            return translated_response
        return _build_anthropic_streaming_response_from_responses_stream(
            upstream_response,
            model=adapter_model,
        )

    if not isinstance(upstream_response, Response):
        raise HTTPException(
            status_code=502,
            detail="Unexpected upstream response type from OpenRouter Responses passthrough.",
        )

    response_body = json.loads(upstream_response.body.decode("utf-8"))
    translated_response = _build_anthropic_response_from_responses_response(
        response_body
    )
    translated_response.headers.update(dict(upstream_response.headers))
    translated_response.status_code = upstream_response.status_code
    return translated_response


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
        hooks: set[str] = set()
        metadata_items: list[dict[str, Any]] = []
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
            hooks.update(sequence_hooks)
            metadata_items.extend(sequence_metadata)
            changed = changed or sequence_changed

        recursively_updated_list = []
        for child in updated_list:
            updated_child, child_count, child_hooks, child_metadata = (
                _compact_google_adapter_persisted_output_value(child)
            )
            recursively_updated_list.append(updated_child)
            compacted_count += child_count
            hooks.update(child_hooks)
            metadata_items.extend(child_metadata)
            changed = changed or updated_child is not child
        if changed:
            return recursively_updated_list, compacted_count, hooks, metadata_items
        return value, compacted_count, hooks, metadata_items

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
                return (
            updated_dict if changed else value,
            expanded_count,
            hooks,
            source_metadata_items,
        )

    if isinstance(value, list):
        updated_list = []
        expanded_count = 0
        hooks: set[str] = set()
        source_metadata_items: list[dict[str, Any]] = []
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
            hooks.update(child_hooks)
            source_metadata_items.extend(child_source_metadata_items)
            if updated_child is not child:
                return (
            updated_list if changed else value,
            expanded_count,
            hooks,
            source_metadata_items,
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
) -> dict[str, Any]:
    updated_body = dict(request_body)
    litellm_metadata = dict(updated_body.get("litellm_metadata") or {})
    changed = False

    if session_id and not litellm_metadata.get("session_id"):
        litellm_metadata["session_id"] = session_id

    if trace_environment and not litellm_metadata.get("trace_environment"):
        litellm_metadata["trace_environment"] = trace_environment

    if not changed:
        return request_body

    updated_body["litellm_metadata"] = litellm_metadata
    return updated_body


def _prepare_request_body_for_passthrough_observability(
    request: Request, request_body: dict[str, Any]
) -> dict[str, Any]:
    session_id = _extract_passthrough_session_id(
        request=request, request_body=request_body
    )
    trace_environment = _get_passthrough_trace_environment()
    return _add_passthrough_trace_context_metadata(
        request_body,
        session_id=session_id,
        trace_environment=trace_environment,
    )


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
    types_match = _CLAUDE_TYPES_XML_BLOCK_PATTERN.search(auto_memory_section)
    if types_match is None:
        raise ValueError("Missing Claude auto-memory <types> block")

    rendered_text = template_text
    rendered_text = rendered_text.replace(
        "{{TYPES_XML_BLOCK}}", types_match.group(0).rstrip()
    )
    rendered_text = rendered_text.replace(
        "{{WHAT_NOT_TO_SAVE_SECTION}}",
        _extract_markdown_section(auto_memory_section, "What NOT to save in memory"),
    )
    rendered_text = rendered_text.replace(
        "{{BEFORE_RECOMMENDING_SECTION}}",
        _extract_markdown_section(auto_memory_section, "Before recommending from memory"),
    )
    rendered_text = rendered_text.replace(
        "{{MEMORY_AND_PERSISTENCE_SECTION}}",
        _extract_markdown_section(
            auto_memory_section, "Memory and other forms of persistence"
        ),
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
                return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_system_prompt_override_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                return (updated_list if changed else value), combined_events

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
                return (updated_dict if changed else value), combined_events

    if isinstance(value, list):
        updated_list = []
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = _replace_claude_prompt_patches_in_value(
                child,
                cc_version,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                return (updated_list if changed else value), combined_events

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
        return dsn

    return _get_first_secret_value(_AAWM_DB_URL_ENV_VARS)


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
        combined_events: list[dict[str, Any]] = []
        changed = False
        for child in value:
            updated_child, child_events = await _expand_aawm_dynamic_directives_in_value(
                child,
                available_context,
            )
            updated_list.append(updated_child)
            combined_events.extend(child_events)
            if updated_child is not child:
                return (updated_list if changed else value), combined_events

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
    ) = _aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
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
            request_block = request_body.get("request") if isinstance(request_body, dict) and isinstance(request_body.get("request"), dict) else request_body
            request_tools = request_block.get("tools") if isinstance(request_block, dict) else None
            function_names: list[str] = []
            if isinstance(request_tools, list):
                for tool_entry in request_tools:
                    if not isinstance(tool_entry, dict):
                        continue
                    decls = tool_entry.get("functionDeclarations") or tool_entry.get("function_declarations")
                    if isinstance(decls, list):
                        for declaration in decls:
                            if isinstance(declaration, dict):
                                name = declaration.get("name")
                                if isinstance(name, str):
                                    function_names.append(name)
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
        gemini_route_family = (
            "gemini_stream_generate_content"
            if "streamgeneratecontent" in endpoint.lower()
            else "gemini_generate_content"
        )
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
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


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

    ## check for streaming
    is_streaming_request = await is_streaming_request_fn(request)

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=custom_headers,
        _forward_headers=True,
        is_streaming_request=is_streaming_request,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value


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
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials
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
    if preserve_client_auth:
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


class BaseOpenAIPassThroughHandler:
    @staticmethod
    async def _base_openai_pass_through_handler(
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

        if request.method == "POST":
            request_body = await get_request_body(request)
            prepared_request_body = request_body
            if _request_uses_codex_native_auth(request) and _is_openai_responses_endpoint(
                endpoint
            ):
                prepared_request_body = _add_route_family_logging_metadata(
                    prepared_request_body,
                    "codex_responses",
                )
                prepared_request_body = _add_codex_request_breakout_logging_metadata(
                    prepared_request_body
                )
            prepared_request_body = _prepare_request_body_for_passthrough_observability(
                request=request,
                request_body=prepared_request_body,
            )
            if prepared_request_body is not request_body:
                _safe_set_request_parsed_body(request, prepared_request_body)

        ## check for streaming
        is_streaming_request = False
        if "stream" in str(updated_url):
            is_streaming_request = True

        ## CREATE PASS-THROUGH
        endpoint_func = create_pass_through_route(
            endpoint=endpoint,
            target=str(updated_url),
            custom_headers=BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=api_key, request=request, extra_headers=extra_headers
            ),
            _forward_headers=forward_headers,
            is_streaming_request=is_streaming_request,  # type: ignore
        )  # dynamically construct pass-through endpoint based on incoming path
        received_value = await endpoint_func(
            request,
            fastapi_response,
            user_api_key_dict,
        )

        return received_value

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
        base_url: httpx.URL, path: str, custom_llm_provider: litellm.LlmProviders
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
