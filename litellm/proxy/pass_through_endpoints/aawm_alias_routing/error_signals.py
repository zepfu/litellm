"""Error-signal extraction, classification, and retry planning for alias routing.

Wave 5C extraction from ``llm_passthrough_endpoints.py``.  Behavior-preserving
relocation only; no logic changes.

Host dependencies (exception detail/status/header extraction, JSON parsing,
handled-error summaries, provider-limit classifiers) are injected via
:func:`configure_error_signals_runtime`.  Shared constants that remain
authoritative in the god module are likewise injected.  Direct imports from
sibling package modules (``lane_keys``, ``policy``, ``codex_oauth``) and the
Grok side-channel module are used where those modules own the symbols.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import time
from typing import Any, Callable, Optional

import httpx

import litellm
from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    side_channel as _grok_side_channel,
)

from .codex_oauth import _clean_codex_auth_value
from . import classification as _classification
from . import failure_actions as _failure_actions
from .lane_keys import (
    _codex_auto_agent_candidate_key,
    _CODEX_AUTO_AGENT_MALFORMED_TOOL_CALL_COOLDOWN_SECONDS,
    _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS,
    _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS_ENV,
    _CODEX_AUTO_AGENT_SPARK_DURABLE_COOLDOWN_SECONDS,
    _CODEX_AUTO_AGENT_SPARK_MODEL,
    _CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES,
)
from .policy import (
    CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER as _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER,
    CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS as _CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS,
    CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS as _CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS,
    CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS as _CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS as _CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS,
    CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS as _CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS,
    CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY as _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
    CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER as _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
    CODEX_AUTO_AGENT_XAI_PROVIDER as _CODEX_AUTO_AGENT_XAI_PROVIDER,
)
from .types import Payload

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_NativeGrokContinuationRetryMetadata = dict[str, Any]

# ---------------------------------------------------------------------------
# Grok side-channel constants (authoritative source is the side_channel module)
# ---------------------------------------------------------------------------
_CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS = (
    _grok_side_channel._CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS
)
_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN = (
    _grok_side_channel._CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN
)
_CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN = (
    _grok_side_channel._CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN
)
_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL = (
    _grok_side_channel._CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL
)

# ---------------------------------------------------------------------------
# Kimi safe-metadata allowlist constants (owned by this module)
# ---------------------------------------------------------------------------
_KIMI_CODE_SAFE_FAILURE_KINDS = frozenset(
    {
        "refresh_required_auth",
        "quota",
        "provider_capacity",
        "transient",
        "malformed",
        "unsupported_model",
        "unsupported_effort",
        "unsupported_capability",
        "unknown",
    }
)
_KIMI_CODE_SAFE_FAILURE_SCOPES = frozenset({"managed_account", "candidate", "telemetry", "none"})
_KIMI_CODE_SAFE_METADATA_GATES = frozenset({"none", "model_id", "think_effort", "capability"})
_KIMI_CODE_SAFE_RESET_REASONS = frozenset(
    {
        "refresh_required",
        "quota_exhausted",
        "provider_capacity",
        "transient_upstream_failure",
        "malformed_provider_response",
        "unsupported_model",
        "unsupported_effort",
        "unsupported_capability",
        "unclassified_failure",
    }
)
_KIMI_CODE_SAFE_UPSTREAM_IDS = frozenset({"k3", "kimi-for-coding", "kimi-for-coding-highspeed"})
_KIMI_CODE_MANAGED_ACCOUNT_COOLDOWN_MODEL = "__managed_account__"

# ---------------------------------------------------------------------------
# Owner-local header / JSON helpers (D1-591)
#
# Exact-semantics copies of the six god-module helpers that error_signals
# previously received as host callbacks.  Defining them here lets a later
# integrator remove ~103 lines from llm_passthrough_endpoints.py without a
# god-module import or mixed host/owner lookup.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Provider-neutral exception helpers and retained OpenRouter error shape
# ---------------------------------------------------------------------------


def _extract_adapter_exception_detail(exc: Any) -> Any:
    detail = getattr(exc, "detail", None)
    if detail is not None:
        return detail
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if response_text is not None:
        return response_text
    return getattr(exc, "message", None)


def _extract_adapter_error_payloads(exc: Any) -> list[Any]:
    parsed_payloads: list[Any] = []
    for candidate in (
        getattr(exc, "detail", None),
        getattr(exc, "message", None),
        str(exc),
    ):
        if isinstance(candidate, (dict, list)):
            parsed_payloads.append(candidate)
            continue
        parsed_payloads.extend(
            _parse_json_payloads_from_text_candidates(
                _extract_embedded_json_payload_candidates(candidate)
            )
        )
    return parsed_payloads


def _extract_adapter_exception_status_code(exc: Any) -> Optional[int]:
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for attr in ("status_code", "code"):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
            try:
                if value is not None:
                    return int(value)
            except Exception:
                continue
    if "429" in str(exc):
        return 429
    return None


class _OpenRouterErrorShapeRuntime:
    def extract_embedded_json_payload_candidates(
        self,
        detail: object,
    ) -> list[str]:
        return _extract_embedded_json_payload_candidates(detail)

    def parse_json_payloads_from_text_candidates(
        self,
        candidates: list[str],
    ) -> list[object]:
        return _parse_json_payloads_from_text_candidates(candidates)

    def extract_upstream_headers(self, exc: object) -> dict[str, Any]:
        return _extract_adapter_upstream_headers(exc)

    def parse_retry_after_seconds_from_headers(
        self,
        headers: dict[str, Any],
    ) -> Optional[float]:
        return _parse_retry_after_seconds_from_headers(headers)

    def get_header_value(
        self,
        headers: dict[str, Any],
        header_name: str,
    ) -> Optional[str]:
        return _get_adapter_header_value(headers, header_name)

    def parse_reset_wait_seconds_from_headers(
        self,
        headers: dict[str, Any],
    ) -> Optional[float]:
        return _parse_rate_limit_reset_wait_seconds_from_headers(headers)


_OPENROUTER_ERROR_SHAPE_RUNTIME = _OpenRouterErrorShapeRuntime()


def _extract_openrouter_adapter_raw_message(exc: Any) -> Optional[str]:
    from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
        error_shape as _openrouter_error_shape,
    )

    return _openrouter_error_shape.extract_raw_message(
        _OPENROUTER_ERROR_SHAPE_RUNTIME,
        exc,
    )


def _is_openrouter_adapter_provider_raw_error(exc: Any) -> bool:
    from litellm.llms.anthropic.experimental_pass_through.providers.openrouter import (
        error_shape as _openrouter_error_shape,
    )

    return _openrouter_error_shape.is_provider_raw_error(
        _OPENROUTER_ERROR_SHAPE_RUNTIME,
        exc,
    )


# ---------------------------------------------------------------------------
# Injected runtime seams (god-module / host dependencies)
# ---------------------------------------------------------------------------
_get_passthrough_handled_http_error_summary: Optional[Callable[..., str]] = None
_is_known_grok_build_usage_balance_exhausted_response: Optional[Callable[..., bool]] = None
_is_known_grok_personal_team_spending_limit_response: Optional[Callable[..., bool]] = None

# Shared constant seams (authoritative in god module)
_CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES: Optional[frozenset[str]] = None
_CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS: Optional[frozenset[str]] = None
_CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS: Optional[frozenset[str]] = None
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS: float = 0.05
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS: float = 1.0
_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS: float = 0.05


# Reference to host_globals set by install(); configure updates it too.
_host_globals_ref: dict | None = None


def configure_error_signals_runtime(  # noqa: PLR0915
    *,
    get_passthrough_handled_http_error_summary: Callable[..., str],
    is_known_grok_build_usage_balance_exhausted_response: Callable[..., bool],
    is_known_grok_personal_team_spending_limit_response: Callable[..., bool],
    durable_cooldown_error_classes: frozenset[str],
    capacity_error_tokens: frozenset[str],
    rate_limit_error_tokens: frozenset[str],
    native_grok_backoff_base_seconds: float = 0.05,
    native_grok_backoff_max_seconds: float = 1.0,
    native_grok_backoff_jitter_seconds: float = 0.05,


) -> None:
    """Bind host dependencies for error-signal extraction and classification."""
    global _get_passthrough_handled_http_error_summary
    global _is_known_grok_build_usage_balance_exhausted_response
    global _is_known_grok_personal_team_spending_limit_response
    global _CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES
    global _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS
    global _CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS
    global _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS
    global _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS
    global _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS

    _get_passthrough_handled_http_error_summary = get_passthrough_handled_http_error_summary
    _is_known_grok_build_usage_balance_exhausted_response = (
        is_known_grok_build_usage_balance_exhausted_response
    )
    _is_known_grok_personal_team_spending_limit_response = (
        is_known_grok_personal_team_spending_limit_response
    )
    _CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES = durable_cooldown_error_classes
    _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS = capacity_error_tokens
    _CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS = rate_limit_error_tokens
    _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS = (
        native_grok_backoff_base_seconds
    )
    _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS = (
        native_grok_backoff_max_seconds
    )
    _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS = (
        native_grok_backoff_jitter_seconds
    )
    # If install() has been called, also update host_globals so rebound
    # functions see the new seam values.
    if _host_globals_ref is not None:
        _mod = globals()
        _host_globals_ref["_get_passthrough_handled_http_error_summary"] = _mod["_get_passthrough_handled_http_error_summary"]
        _host_globals_ref["_is_known_grok_build_usage_balance_exhausted_response"] = _mod["_is_known_grok_build_usage_balance_exhausted_response"]
        _host_globals_ref["_is_known_grok_personal_team_spending_limit_response"] = _mod["_is_known_grok_personal_team_spending_limit_response"]
        _host_globals_ref["_CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES"] = _mod["_CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES"]
        _host_globals_ref["_CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS"] = _mod["_CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS"]
        _host_globals_ref["_CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS"] = _mod["_CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS"]
        _host_globals_ref["_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS"] = _mod["_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS"]
        _host_globals_ref["_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS"] = _mod["_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS"]
        _host_globals_ref["_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS"] = _mod["_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS"]


# ---------------------------------------------------------------------------
# Error text and token extraction
# ---------------------------------------------------------------------------


def _codex_auto_agent_error_text(exc: Any) -> str:
    detail = _extract_adapter_exception_detail(exc)
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
    if "grok build usage balance exhausted" in text_lower:
        tokens.add(_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN)
    if "personal-team-blocked:spending-limit" in text_lower:
        tokens.add(_CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN)
    if (
        "usage_limit_reached" in text_lower
        or "usage limit" in text_lower
        or "weekly limit" in text_lower
        or "quota exceeded" in text_lower
        or "quota exhausted" in text_lower
        or "quota limit" in text_lower
    ):
        tokens.add("usage_limit_reached")
    if "resource_exhausted" in text_lower or "resource exhausted" in text_lower:
        tokens.add("RESOURCE_EXHAUSTED")
    if "model_capacity_exhausted" in text_lower or "model capacity exhausted" in text_lower:
        tokens.add("MODEL_CAPACITY_EXHAUSTED")
    if "currently experiencing high demand" in text_lower or "experiencing high demand" in text_lower:
        tokens.add("HIGH_DEMAND")
    if "selected model is at capacity" in text_lower or (
        "model is at capacity" in text_lower and "try a different model" in text_lower
    ):
        tokens.add("MODEL_AT_CAPACITY")
    if "model is overloaded" in text_lower or "overloaded_error" in text_lower:
        tokens.add("MODEL_OVERLOADED")
    if "busy upstream" in text_lower or ("upstream" in text_lower and "busy" in text_lower):
        tokens.add("UPSTREAM_BUSY")
    if "rate_limit_exceeded" in text_lower or "rate limit" in text_lower:
        tokens.add("RATE_LIMIT_EXCEEDED")
    if "too many requests" in text_lower:
        tokens.add("429")
        tokens.add("RATE_LIMIT_EXCEEDED")
    if "aawm_codex_auto_agent_candidate_unavailable" in text_lower:
        tokens.add("aawm_codex_auto_agent_candidate_unavailable")
    if "not supported when using codex with a chatgpt account" in text_lower and (
        "model is not supported" in text_lower or " is not supported" in text_lower
    ):
        tokens.add("aawm_codex_auto_agent_candidate_unavailable")
    if "grok-4.5" in text_lower and any(
        marker in text_lower
        for marker in (
            "model not found",
            "model does not exist",
            "model is not available",
            "unknown model",
            "unsupported model",
        )
    ):
        tokens.add("aawm_codex_auto_agent_candidate_unavailable")
    if "aawm_auto_agent_failed_responses_payload" in text_lower:
        tokens.add("aawm_auto_agent_failed_responses_payload")
    if "aawm_auto_agent_malformed_tool_call_text" in text_lower:
        tokens.add("aawm_auto_agent_malformed_tool_call_text")
    if (
        "permission-denied" in text_lower
        and "content violates usage guidelines" in text_lower
        and "safety_check_type_cyber" in text_lower
    ):
        tokens.add("safety_policy_denied")
    if (
        "error from provider (deepseek)" in text_lower
        and "assistant message with 'tool_calls' must be followed by tool messages" in text_lower
    ) or "insufficient tool messages following tool_calls message" in text_lower:
        tokens.add("DEEPSEEK_TOOL_MESSAGE_MISMATCH")
    if "invalid message provided" in text_lower and "must have non-empty content or tool calls" in text_lower:
        tokens.add("OPENROUTER_INVALID_CHAT_MESSAGE")
    if "invalid tool call provided" in text_lower and "tool arguments must be a stringified json object" in text_lower:
        tokens.add("OPENROUTER_INVALID_TOOL_CALL_ARGUMENTS")


def _extract_codex_auto_agent_error_tokens(exc: Any) -> set[str]:
    tokens: set[str] = set()
    for parsed in _extract_adapter_error_payloads(exc):
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


# ---------------------------------------------------------------------------
# Error class predicates
# ---------------------------------------------------------------------------


def _is_codex_auto_agent_durable_cooldown_error_class(
    error_class: Optional[str],
) -> bool:
    assert _CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES is not None
    return error_class in _CODEX_AUTO_AGENT_DURABLE_COOLDOWN_ERROR_CLASSES


def _is_codex_auto_agent_spark_candidate(candidate: Optional[dict[str, Any]]) -> bool:
    if not isinstance(candidate, dict):
        return False
    return str(candidate.get("model") or "") == _CODEX_AUTO_AGENT_SPARK_MODEL


def _is_codex_auto_agent_grok_4_5_candidate(
    candidate: Optional[dict[str, Any]],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    if candidate.get("provider") != _CODEX_AUTO_AGENT_XAI_PROVIDER:
        return False
    model = str(candidate.get("model") or "")
    if model in {"oa_xai/grok-4.5", "grok-4.5", "xai/grok-4.5"}:
        return True
    route_family = str(candidate.get("route_family") or "")
    return route_family in {
        "codex_xai_oauth_responses_adapter",
        "codex_grok_native_responses_adapter",
        "anthropic_grok_native_responses_adapter",
    } and model.endswith("grok-4.5")


def _is_codex_auto_agent_native_grok_4_5_candidate(
    candidate: Optional[dict[str, Any]],
) -> bool:
    if not _is_codex_auto_agent_grok_4_5_candidate(candidate):
        return False
    route_family = str((candidate or {}).get("route_family") or "")
    return route_family in {
        "codex_grok_native_responses_adapter",
        "anthropic_grok_native_responses_adapter",
    }


def _is_codex_auto_agent_xai_candidate(
    candidate: Optional[dict[str, Any]],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    return candidate.get("provider") == _CODEX_AUTO_AGENT_XAI_PROVIDER


# ---------------------------------------------------------------------------
# Kimi code helpers
# ---------------------------------------------------------------------------


def _is_kimi_code_auto_agent_candidate(candidate: Optional[dict[str, Any]]) -> bool:
    return isinstance(candidate, dict) and candidate.get("provider") == _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER


def _get_kimi_code_managed_account_cooldown_key() -> str:
    # Kimi managed-account keys are deliberately UN-tagged (no epoch_tag):
    # they represent account-level quota state that is independent of the
    # routing config generation.  Tagging them would incorrectly reset
    # account-level cooldowns on every config refresh.
    return _codex_auto_agent_candidate_key(
        {
            "provider": _CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER,
            "model": _KIMI_CODE_MANAGED_ACCOUNT_COOLDOWN_MODEL,
        },
        _CODEX_AUTO_AGENT_KIMI_CODE_LANE_KEY,
    )


def _get_safe_kimi_code_probe_failure_metadata(
    exc: Any,
    *,
    candidate: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return only adapter-classifier allowlisted Kimi probe metadata."""

    if not _is_kimi_code_auto_agent_candidate(candidate):
        return None
    raw_metadata = getattr(exc, "kimi_code_probe_failure_metadata", None)
    if not isinstance(raw_metadata, dict):
        return None
    kind = raw_metadata.get("kind")
    scope = raw_metadata.get("scope")
    upstream_id = raw_metadata.get("upstream_id")
    metadata_gate = raw_metadata.get("metadata_gate")
    reset_reason = raw_metadata.get("reset_reason")
    status_code = raw_metadata.get("status_code")
    trace_id = raw_metadata.get("trace_id")
    if (
        kind not in _KIMI_CODE_SAFE_FAILURE_KINDS
        or scope not in _KIMI_CODE_SAFE_FAILURE_SCOPES
        or upstream_id not in _KIMI_CODE_SAFE_UPSTREAM_IDS
        or metadata_gate not in _KIMI_CODE_SAFE_METADATA_GATES
        or reset_reason not in _KIMI_CODE_SAFE_RESET_REASONS
    ):
        return None
    if status_code is not None and (
        not isinstance(status_code, int) or isinstance(status_code, bool) or status_code < 100 or status_code > 599
    ):
        return None
    if trace_id is not None and (
        not isinstance(trace_id, str)
        or len(trace_id) > 256
        or not trace_id
        or any(not (character.isalnum() or character in "._-") for character in trace_id)
    ):
        return None
    return {
        "kind": kind,
        "scope": scope,
        "upstream_id": upstream_id,
        "metadata_gate": metadata_gate,
        "status_code": status_code,
        "trace_id": trace_id,
        "reset_reason": reset_reason,
    }


def _classify_kimi_code_auto_agent_probe_failure(
    metadata: Optional[dict[str, Any]],
) -> Optional[str]:
    if metadata is None:
        return None
    scope = metadata["scope"]
    if scope == "managed_account":
        return "kimi_code_managed_account"
    if scope == "candidate":
        return "kimi_code_candidate_failure"
    return "kimi_code_no_cooldown"


def _build_safe_kimi_code_selection_telemetry(
    *,
    alias_model: str,
    candidate: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build an alias-attempt record without copying provider payload details."""

    return {
        "alias": alias_model,
        "candidate": candidate["model"],
        "upstream_id": metadata["upstream_id"],
        "metadata_gate": metadata["metadata_gate"],
        "scope": metadata["scope"],
        "reset_reason": metadata["reset_reason"],
        "kind": metadata["kind"],
        "status_code": metadata["status_code"],
        "trace_id": metadata["trace_id"],
    }


# ---------------------------------------------------------------------------
# Transient internal error classification
# ---------------------------------------------------------------------------


def _is_codex_auto_agent_transient_internal_error_class(
    error_class: Optional[str],
) -> bool:
    # Classifier emits upstream_transient_internal only; do not accept a dead alias.
    return error_class == "upstream_transient_internal"


# ---------------------------------------------------------------------------
# Native Grok continuation transient retry planning
# ---------------------------------------------------------------------------


def _get_codex_auto_agent_native_grok_continuation_transient_max_attempts() -> int:
    """Request-scoped total provider attempts for native Grok continuation retries.

    Independent of alias candidate-pool length and preserved across outer
    candidate-selection re-entry within the same request. Default 8 so bursts
    of 3-6 terminal-event 502s can recover without switching providers.
    """
    raw_value = _clean_codex_auth_value(
        os.getenv(_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS_ENV)
    )
    if raw_value is None:
        return _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS
    try:
        parsed = int(raw_value)
    except Exception:
        return _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS
    # Conservative clamp: never drop below 6 (live work pool size) and
    # never allow unbounded same-candidate storms.
    return max(6, min(16, parsed))


def _get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds(
    failed_attempt: int,
) -> float:
    """Short exponential backoff with bounded jitter; each delay capped near 1s."""
    attempt_index = max(1, int(failed_attempt))
    base_seconds = min(
        _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS,
        _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_BASE_SECONDS * (2 ** (attempt_index - 1)),
    )
    jitter_cap = min(
        _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_JITTER_SECONDS,
        base_seconds,
    )
    jitter_seconds = random.uniform(0.0, jitter_cap) if jitter_cap > 0 else 0.0
    return min(
        _CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_BACKOFF_MAX_SECONDS,
        base_seconds + jitter_seconds,
    )


def _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
    *,
    is_native_grok_4_5_candidate: bool,
    has_continuation_state: bool,
    error_class: Optional[str],
    cooldown_scope: Optional[str],
) -> bool:
    """Whether this failure may consume the native Grok continuation transient budget.

    Deliberately excludes ``candidate_unavailable`` even when native Grok 4.5 uses
    cooldown scope ``none`` for that class. Generic probe/credential unavailability
    must not enter the same-candidate 502-style retry budget; only bare transient
    internal failures (``upstream_transient_internal``) are eligible.
    """
    # Keep this allow-list tight: do not treat candidate_unavailable as eligible.
    return bool(
        has_continuation_state
        and cooldown_scope == "none"
        and is_native_grok_4_5_candidate
        and _is_codex_auto_agent_transient_internal_error_class(error_class)
    )


def _build_codex_auto_agent_native_grok_continuation_retry_metadata(
    *,
    status: str,
    provider_attempt: int,
    max_attempts: int,
    provider: Optional[str],
    model: Optional[str],
    route_family: Optional[str],
    backoff_seconds: Optional[float] = None,
) -> _NativeGrokContinuationRetryMetadata:
    metadata: _NativeGrokContinuationRetryMetadata = {
        "status": status,
        "provider_attempt": int(provider_attempt),
        "max_attempts": int(max_attempts),
        "provider": provider,
        "model": model,
        "route_family": route_family,
    }
    if backoff_seconds is not None:
        metadata["backoff_seconds"] = round(float(backoff_seconds), 3)
    return metadata


def _plan_codex_auto_agent_native_grok_continuation_transient_retry(
    *,
    is_native_grok_4_5_candidate: bool,
    has_continuation_state: bool,
    error_class: Optional[str],
    cooldown_scope: Optional[str],
    provider_attempt: int,
    provider: Optional[str],
    model: Optional[str],
    route_family: Optional[str],
    max_attempts: Optional[int] = None,
) -> tuple[
    bool,
    Optional[float],
    Optional[_NativeGrokContinuationRetryMetadata],
]:
    """Annotate attempt metadata and decide whether to retry the same candidate.

    ``provider_attempt`` must be the request-scoped total of eligible native Grok
    continuation transient provider attempts so far (not reset on outer
    candidate-selection re-entry). Backoff is only returned when a same-candidate
    retry is scheduled; callers must not sleep after the final failed attempt.
    """
    if not _is_codex_auto_agent_native_grok_continuation_transient_retry_eligible(
        is_native_grok_4_5_candidate=is_native_grok_4_5_candidate,
        has_continuation_state=has_continuation_state,
        error_class=error_class,
        cooldown_scope=cooldown_scope,
    ):
        return False, None, None

    resolved_max_attempts = (
        int(max_attempts)
        if max_attempts is not None
        else _get_codex_auto_agent_native_grok_continuation_transient_max_attempts()
    )
    if provider_attempt < resolved_max_attempts:
        backoff_seconds = _get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds(provider_attempt)
        metadata = _build_codex_auto_agent_native_grok_continuation_retry_metadata(
            status="scheduled_same_candidate_retry",
            provider_attempt=provider_attempt,
            max_attempts=resolved_max_attempts,
            provider=provider,
            model=model,
            route_family=route_family,
            backoff_seconds=backoff_seconds,
        )
        return True, backoff_seconds, metadata

    metadata = _build_codex_auto_agent_native_grok_continuation_retry_metadata(
        status="same_candidate_retry_exhausted",
        provider_attempt=provider_attempt,
        max_attempts=resolved_max_attempts,
        provider=provider,
        model=model,
        route_family=route_family,
    )
    return False, None, metadata


# ---------------------------------------------------------------------------
# Cooldown scope derivation
# ---------------------------------------------------------------------------


def _get_codex_auto_agent_cooldown_scope(error_class: Optional[str]) -> str:
    if _is_codex_auto_agent_durable_cooldown_error_class(error_class):
        return "candidate"
    return "request_local"


def _get_codex_auto_agent_candidate_cooldown_scope(
    error_class: Optional[str],
    *,
    candidate: Optional[dict[str, Any]] = None,
    kimi_failure_metadata: Optional[dict[str, Any]] = None,
) -> str:
    if _is_kimi_code_auto_agent_candidate(candidate):
        if (
            error_class == "kimi_code_managed_account"
            and kimi_failure_metadata is not None
            and kimi_failure_metadata.get("scope") == "managed_account"
        ):
            return "managed_account"
        if (
            error_class == "kimi_code_candidate_failure"
            and kimi_failure_metadata is not None
            and kimi_failure_metadata.get("scope") == "candidate"
        ):
            return "candidate"
        if error_class == "kimi_code_no_cooldown":
            return "none"
    if error_class == "safety_policy_denied":
        return "request_local"
    # Native Grok 4.5 is live. Broad candidate-unavailable probes can still
    # happen on transient/request-shape blips, so do not evict the native
    # candidate from routing. Other xAI alias candidates (Composer, Grok Build,
    # managed OAuth Grok 4.5, etc.) stay request-local so missing/refreshing
    # credentials cannot leave multi-hour Redis candidate cooldowns.
    # Explicit rate-limit / capacity / quota classes still use candidate scope.
    if error_class == "candidate_unavailable" and _is_codex_auto_agent_native_grok_4_5_candidate(candidate):
        return "none"
    if error_class == "candidate_unavailable" and _is_codex_auto_agent_xai_candidate(candidate):
        return "request_local"
    # Native Grok 4.5 malformed tool-call text remains rejected and can still
    # redispatch in-flight, but must not write a durable candidate cooldown.
    # Composer / Grok Build / non-native candidates keep durable candidate
    # cooldowns for this class.
    if error_class == "malformed_tool_call_text" and _is_codex_auto_agent_native_grok_4_5_candidate(candidate):
        return "request_local"
    if _is_codex_auto_agent_native_grok_4_5_candidate(
        candidate
    ) and _is_codex_auto_agent_transient_internal_error_class(error_class):
        return "none"
    if _is_codex_auto_agent_spark_candidate(candidate) and _is_codex_auto_agent_transient_internal_error_class(
        error_class
    ):
        return "candidate"
    return _get_codex_auto_agent_cooldown_scope(error_class)


# ---------------------------------------------------------------------------
# Grok account quota exhaustion
# ---------------------------------------------------------------------------


def _is_codex_auto_agent_grok_build_usage_balance_exhausted(exc: Any) -> bool:
    assert _is_known_grok_build_usage_balance_exhausted_response is not None
    status_code = _extract_adapter_exception_status_code(exc)
    return _is_known_grok_build_usage_balance_exhausted_response(
        url=httpx.URL(_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL),
        custom_llm_provider=litellm.LlmProviders.XAI.value,
        status_code=status_code,
        exc=exc,
    )


def _is_codex_auto_agent_grok_personal_team_spending_limit(exc: Any) -> bool:
    assert _is_known_grok_personal_team_spending_limit_response is not None
    status_code = _extract_adapter_exception_status_code(exc)
    return _is_known_grok_personal_team_spending_limit_response(
        url=httpx.URL(_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL),
        custom_llm_provider=litellm.LlmProviders.XAI.value,
        status_code=status_code,
        exc=exc,
    )


def _is_codex_auto_agent_grok_account_quota_candidate(
    candidate: Optional[dict[str, Any]],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    if candidate.get("provider") != _CODEX_AUTO_AGENT_XAI_PROVIDER:
        return False
    route_family = str(candidate.get("route_family") or "")
    return route_family in {
        "codex_grok_native_responses_adapter",
        "codex_xai_oauth_responses_adapter",
        "anthropic_grok_native_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
    }


def _get_codex_auto_agent_grok_account_quota_lane_cooldown_key(
    candidate: Payload,
    lane_key: Optional[str],
) -> Optional[str]:
    if not lane_key or not _is_codex_auto_agent_grok_account_quota_candidate(candidate):
        return None
    return f"{candidate.get('provider')}:__account_quota__:{lane_key}"


def _is_codex_auto_agent_grok_account_quota_exhaustion(
    exc: Any,
    *,
    candidate: Optional[dict[str, Any]] = None,
) -> bool:
    if not (
        _is_codex_auto_agent_grok_build_usage_balance_exhausted(exc)
        or _is_codex_auto_agent_grok_personal_team_spending_limit(exc)
    ):
        return False
    if candidate is None:
        return True
    return _is_codex_auto_agent_grok_account_quota_candidate(candidate)


# ---------------------------------------------------------------------------
# Retryable exhaustion classification
# ---------------------------------------------------------------------------


_ALIBABA_TOKEN_PLAN_UNSUPPORTED_MODEL_ERROR_CODES = frozenset(
    {
        "ModelNotFound",
        "ModelNotSupported",
        "UnsupportedModel",
        "model_not_found",
        "model_not_supported",
        "InvalidParameter.Model",
    }
)

_ALIBABA_TOKEN_PLAN_UNSUPPORTED_MODEL_MESSAGE_MARKERS = (
    "model not exist",
    "model does not exist",
    "model is not supported",
    "model not supported",
    "unsupported model",
    "has been withdrawn",
)


def _is_alibaba_token_plan_unsupported_model_response(
    exc: Any,
    *,
    candidate: Optional[dict[str, Any]] = None,
) -> bool:
    """Detect a structured Alibaba Token Plan unsupported/withdrawn-model rejection.

    Both conditions are mandatory:

    1. Trusted Alibaba provider attribution: the failed *candidate* must be
       the Alibaba Token Plan provider. A local exception or another
       provider's ``ModelNotFound``-style error never matches.
    2. A structured error payload: a structured error block carrying a known
       Alibaba model-admission error code, or a structured error message
       that names the model as unsupported/unknown/withdrawn. Free-form
       local exception text (e.g. a local ``ValueError``) never matches.

    Generic capacity, auth, or request-shape errors never match here, so
    programming and configuration errors are not hidden behind candidate
    classification.
    """
    if not isinstance(candidate, dict):
        return False
    if candidate.get("provider") != _CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER:
        return False

    error_blocks = _iter_codex_auto_agent_error_blocks(exc)
    if not error_blocks:
        return False

    _error_type, error_code = _extract_codex_auto_agent_error_type_and_code(exc)
    if isinstance(error_code, str) and error_code in _ALIBABA_TOKEN_PLAN_UNSUPPORTED_MODEL_ERROR_CODES:
        return True
    for error in error_blocks:
        message = error.get("message")
        if not isinstance(message, str):
            continue
        message_lower = message.lower()
        if "model" not in message_lower:
            continue
        if any(
            marker in message_lower
            for marker in _ALIBABA_TOKEN_PLAN_UNSUPPORTED_MODEL_MESSAGE_MARKERS
        ):
            return True
    return False


def _classify_codex_auto_agent_retryable_exhaustion(
    exc: Any,
    *,
    candidate: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    assert _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS is not None
    assert _CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS is not None
    status_code = _extract_adapter_exception_status_code(exc)
    tokens = _extract_codex_auto_agent_error_tokens(exc)
    if _is_codex_auto_agent_grok_account_quota_exhaustion(exc):
        return "capacity_exhausted"
    if _is_alibaba_token_plan_unsupported_model_response(exc, candidate=candidate):
        return "candidate_unavailable"
    if "usage_limit_reached" in tokens:
        return "usage_limit_reached"
    if tokens & _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS:
        return "capacity_exhausted"
    if tokens & _CODEX_AUTO_AGENT_RATE_LIMIT_ERROR_TOKENS:
        return "rate_limited"
    if "aawm_codex_auto_agent_candidate_unavailable" in tokens:
        return "candidate_unavailable"
    if "DEEPSEEK_TOOL_MESSAGE_MISMATCH" in tokens:
        return "provider_format_rejected"
    if "OPENROUTER_INVALID_CHAT_MESSAGE" in tokens:
        return "provider_format_rejected"
    if "OPENROUTER_INVALID_TOOL_CALL_ARGUMENTS" in tokens:
        return "provider_format_rejected"
    if "OPENROUTER_PROVIDER_RAW_ERROR" in tokens:
        return "provider_terminal_error"
    if "aawm_auto_agent_failed_responses_payload" in tokens:
        return "provider_terminal_error"
    if "aawm_auto_agent_malformed_tool_call_text" in tokens:
        return "malformed_tool_call_text"
    if "safety_policy_denied" in tokens:
        return "safety_policy_denied"
    if status_code == 429:
        return "rate_limited"
    if status_code in _CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES:
        return "upstream_transient_internal"
    if status_code == 504:
        return "upstream_timeout"
    return None


def _is_codex_auto_agent_retryable_exhaustion(exc: Any) -> bool:
    return _classify_codex_auto_agent_retryable_exhaustion(exc) is not None


# ---------------------------------------------------------------------------
# Header wait / cooldown seconds
# ---------------------------------------------------------------------------


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


def _get_codex_auto_agent_cooldown_seconds(
    exc: Any,
    *,
    candidate: Optional[dict[str, Any]] = None,
) -> float:
    assert _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS is not None
    header_wait = _parse_codex_auto_agent_header_wait_seconds(exc)
    error_class = _classify_codex_auto_agent_retryable_exhaustion(
        exc, candidate=candidate
    )
    tokens = _extract_codex_auto_agent_error_tokens(exc)
    if header_wait is not None:
        resolved = max(_CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS, header_wait)
    elif (
        error_class in {"capacity_exhausted", "upstream_overloaded"} or tokens & _CODEX_AUTO_AGENT_CAPACITY_ERROR_TOKENS
    ):
        resolved = _CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS
    elif _is_codex_auto_agent_transient_internal_error_class(error_class):
        resolved = _CODEX_AUTO_AGENT_DEFAULT_TRANSIENT_COOLDOWN_SECONDS
    elif "usage_limit_reached" in tokens:
        resolved = _CODEX_AUTO_AGENT_DEFAULT_USAGE_LIMIT_COOLDOWN_SECONDS
    elif error_class == "malformed_tool_call_text":
        resolved = _CODEX_AUTO_AGENT_MALFORMED_TOOL_CALL_COOLDOWN_SECONDS
    elif "RESOURCE_EXHAUSTED" in tokens or "RATE_LIMIT_EXCEEDED" in tokens:
        resolved = _CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    elif _extract_adapter_exception_status_code(exc) in {429, 503, 529}:
        resolved = _CODEX_AUTO_AGENT_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    else:
        resolved = _CODEX_AUTO_AGENT_DEFAULT_CAPACITY_COOLDOWN_SECONDS

    if _is_codex_auto_agent_spark_candidate(candidate) and _is_codex_auto_agent_durable_cooldown_error_class(
        error_class
    ):
        return _CODEX_AUTO_AGENT_SPARK_DURABLE_COOLDOWN_SECONDS
    if _is_codex_auto_agent_grok_account_quota_exhaustion(
        exc,
        candidate=candidate,
    ) and _is_codex_auto_agent_durable_cooldown_error_class(error_class):
        return _CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS
    return resolved


# ---------------------------------------------------------------------------
# Error block iteration and type/code extraction
# ---------------------------------------------------------------------------


def _iter_codex_auto_agent_error_blocks(exc: Any) -> list[dict[str, Any]]:
    payloads: list[Any] = []
    detail = getattr(exc, "detail", None)
    if detail is not None:
        payloads.append(detail)
    payloads.extend(
        payload
        for payload in _extract_adapter_error_payloads(exc)
        if payload is not detail
    )

    error_blocks: list[dict[str, Any]] = []
    for parsed in payloads:
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                error_blocks.append(error)
        elif isinstance(parsed, list):
            error_blocks.extend(
                item["error"] for item in parsed if isinstance(item, dict) and isinstance(item.get("error"), dict)
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
            error_type = _clean_codex_auth_value(error.get("type") or error.get("status"))
        if error_code is None:
            error_code = error.get("code") or error.get("status")
        if error_type is not None and error_code is not None:
            return error_type, error_code
    return error_type or fallback_error_type, error_code or fallback_error_code


# ---------------------------------------------------------------------------
# Source error summary
# ---------------------------------------------------------------------------


def _get_codex_auto_agent_source_error_summary(
    exc: Any,
    *,
    status_code: Optional[int],
) -> str:
    assert _get_passthrough_handled_http_error_summary is not None
    from fastapi import HTTPException
    from starlette import status as http_status

    raw_message = _extract_openrouter_adapter_raw_message(exc)
    if isinstance(raw_message, str) and raw_message:
        for parsed in _parse_json_payloads_from_text_candidates([raw_message]):
            if not isinstance(parsed, dict):
                continue
            message = parsed.get("message")
            if not isinstance(message, str):
                error = parsed.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
            if isinstance(message, str) and message:
                return _get_passthrough_handled_http_error_summary(
                    HTTPException(
                        status_code=status_code or http_status.HTTP_502_BAD_GATEWAY,
                        detail=message,
                    ),
                    status_code=status_code,
                )
    return _get_passthrough_handled_http_error_summary(
        exc,
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# D1-586 shadow failure-action decisions (observational only)
# ---------------------------------------------------------------------------


def build_shadow_failure_action_decision(
    *,
    status_code: Optional[int] = None,
    message: str = "",
    retry_after_seconds: Optional[float] = None,
    provider: Optional[str] = None,
    current_error_class: Optional[str] = None,
    current_cooldown_scope: Optional[str] = None,
    current_status: Optional[str] = None,
    policy: Optional[_failure_actions.FailureActionPolicy] = None,
) -> _failure_actions.ShadowFailureActionDecision:
    """Classify then map to a shadow action without enforcing policy.

    Classification remains in :mod:`classification`; this helper only attaches
    the configurable action vocabulary decision for observability. Enforcement
    stays disabled (:data:`FAILURE_ACTION_ENFORCEMENT_ENABLED` is False).
    """
    event = _classification.classify_failure(
        status_code=status_code,
        provider=provider,
        message=message or "",
        retry_after_seconds=retry_after_seconds,
    )
    return _failure_actions.decide_shadow_failure_action(
        event,
        policy=policy,
        current_error_class=current_error_class,
        current_cooldown_scope=current_cooldown_scope,
        current_status=current_status,
    )


def build_shadow_failure_action_decision_from_exc(
    exc: Any,
    *,
    candidate: Optional[dict[str, Any]] = None,
    current_error_class: Optional[str] = None,
    current_cooldown_scope: Optional[str] = None,
    current_status: Optional[str] = None,
    policy: Optional[_failure_actions.FailureActionPolicy] = None,
) -> _failure_actions.ShadowFailureActionDecision:
    """Shadow decision for a raised exception using existing signal extractors.

    Uses status/summary/retry-after extractors already owned by this module so
    the candidate loop can stamp the same sanitized comparison fields without
    re-implementing classification inputs.
    """
    del candidate  # reserved for future provider-neutral context; unused (no hardcoding)
    status_code = _extract_adapter_exception_status_code(exc)
    source_error = _get_codex_auto_agent_source_error_summary(
        exc,
        status_code=status_code,
    )
    retry_after_seconds = _parse_codex_auto_agent_header_wait_seconds(exc)
    return build_shadow_failure_action_decision(
        status_code=status_code,
        message=str(source_error or ""),
        retry_after_seconds=retry_after_seconds,
        provider=None,
        current_error_class=current_error_class,
        current_cooldown_scope=current_cooldown_scope,
        current_status=current_status,
        policy=policy,
    )

# ---------------------------------------------------------------------------
# God-module facade installation (Wave 5C)
# ---------------------------------------------------------------------------

_HOST_FUNCTION_NAMES = (
    "_codex_auto_agent_error_text",
    "_add_codex_auto_agent_text_error_tokens",
    "_extract_codex_auto_agent_error_tokens",
    "_is_codex_auto_agent_durable_cooldown_error_class",
    "_is_codex_auto_agent_spark_candidate",
    "_is_codex_auto_agent_grok_4_5_candidate",
    "_is_codex_auto_agent_native_grok_4_5_candidate",
    "_is_codex_auto_agent_xai_candidate",
    "_is_kimi_code_auto_agent_candidate",
    "_get_kimi_code_managed_account_cooldown_key",
    "_get_safe_kimi_code_probe_failure_metadata",
    "_classify_kimi_code_auto_agent_probe_failure",
    "_build_safe_kimi_code_selection_telemetry",
    "_is_codex_auto_agent_transient_internal_error_class",
    "_get_codex_auto_agent_native_grok_continuation_transient_max_attempts",
    "_get_codex_auto_agent_native_grok_continuation_transient_backoff_seconds",
    "_is_codex_auto_agent_native_grok_continuation_transient_retry_eligible",
    "_build_codex_auto_agent_native_grok_continuation_retry_metadata",
    "_plan_codex_auto_agent_native_grok_continuation_transient_retry",
    "_get_codex_auto_agent_cooldown_scope",
    "_get_codex_auto_agent_candidate_cooldown_scope",
    "_is_codex_auto_agent_grok_build_usage_balance_exhausted",
    "_is_codex_auto_agent_grok_personal_team_spending_limit",
    "_is_codex_auto_agent_grok_account_quota_candidate",
    "_get_codex_auto_agent_grok_account_quota_lane_cooldown_key",
    "_is_codex_auto_agent_grok_account_quota_exhaustion",
    "_classify_codex_auto_agent_retryable_exhaustion",
    "_is_codex_auto_agent_retryable_exhaustion",
    "_parse_codex_auto_agent_header_wait_seconds",
    "_get_codex_auto_agent_cooldown_seconds",
    "_iter_codex_auto_agent_error_blocks",
    "_extract_codex_auto_agent_error_type_and_code",
    "_get_codex_auto_agent_source_error_summary",
    "build_shadow_failure_action_decision",
    "build_shadow_failure_action_decision_from_exc",
    "_extract_adapter_exception_detail",
    "_extract_adapter_error_payloads",
    "_extract_adapter_exception_status_code",
    "_extract_openrouter_adapter_raw_message",
    "_is_openrouter_adapter_provider_raw_error",
    "_extract_adapter_upstream_headers",
    "_get_adapter_header_value",
    "_parse_retry_after_seconds_from_headers",
    "_parse_rate_limit_reset_wait_seconds_from_headers",
    "_extract_embedded_json_payload_candidates",
    "_parse_json_payloads_from_text_candidates",
)


def install(host_globals: dict) -> None:
    """Publish same-object god-module facades for the moved functions.

    Functions retain this module's globals. Provider-neutral exception helpers,
    OpenRouter error-shape helpers, and header/JSON helpers stay owner-local and
    are published directly; host-owned dependencies remain late-bound through
    the callbacks configured by
    :func:`configure_error_signals_runtime`.
    """
    global _host_globals_ref
    _host_globals_ref = host_globals
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        host_globals[_name] = _mod[_name]
