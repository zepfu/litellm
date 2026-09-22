"""Wave 4 extraction: lane_keys pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Optional

from fastapi import Request

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global constants
    _CODEX_AUTO_AGENT_XAI_LANE_KEY: str
    _CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY: str

    # Host-global functions
    def _clean_codex_auth_value(value: Any) -> Optional[str]: ...
    def _safe_get_request_headers(request: Any) -> dict: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_get_codex_auto_agent_header",
    "_hash_codex_auto_agent_lane_value",
    "_resolve_codex_auto_agent_openai_lane_key",
    "_resolve_codex_auto_agent_openai_cooldown_lane_key",
    "_get_codex_auto_agent_lane_state_cache_ttl_seconds",
    "_codex_auto_agent_candidate_key",
    "_resolve_codex_auto_agent_xai_lane_key",
    "_resolve_anthropic_auto_agent_native_lane_key",
    "_resolve_anthropic_auto_agent_native_cooldown_lane_key",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound


# ── Restored constants ──────────────────────────────────────────────

_CLAUDE_PERSISTED_OUTPUT_PATTERN = re.compile(
    r"\A<system-reminder>\n"
    r"(?P<hook>SubagentStart|SubAgentStart|SessionStart) hook additional context: <persisted-output>\n"
    r"Output too large \([^)]+\)\. Full output saved to: (?P<path>/[^\n]+)\n\n"
    r"Preview \(first 2KB\):\n"
    r"(?P<preview>.*)"
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

_CODEX_REASONING_EFFORT_TIERS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

_CODEX_REASONING_EFFORT_TIER_INDEX = {effort: index for index, effort in enumerate(_CODEX_REASONING_EFFORT_TIERS)}

_CODEX_AUTO_AGENT_REASONING_EFFORT_AUDIT_FIELDS = (
    "reasoning_effort_requested",
    "reasoning_effort_config_value",
    "reasoning_effort_config_source",
    "reasoning_effort_source",
    "reasoning_effort_native_provider",
    "reasoning_effort_native_value",
    "reasoning_effort_native_field",
    "reasoning_effort_supported_ceiling",
    "reasoning_effort_resolved_model",
    "reasoning_effort_resolved_provider",
    "reasoning_effort_candidate_attempt",
    "reasoning_effort_mapping_reason",
    "reasoning_effort_clamped_from",
    "reasoning_effort_clamp_reason",
)

_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_NAME = "codex_auto_agent_prevention_guidance"

_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_POLICY_VERSION = "2026-07-21.v2"

_CODEX_AUTO_AGENT_PREVENTION_GUIDANCE_PROMPT = """Codex auto-agent completion contract:
- Always produce a non-empty final answer after completing or stopping the task; do not end a successful request with only reasoning, tool calls, or no visible assistant text.
- Do not return internal planning text as the final answer. Complete the requested work, or state the exact blocker and the next concrete step.
- If a required tool is unavailable or blocked, state the exact observed tool/platform error and continue with bounded evidence from available context; do not claim tools or filesystem are unavailable unless a tool/platform error proves it.
- If the user requested code or artifact changes, either make the scoped change or explicitly say no files were modified and why. Do not answer with a generic explanation of the function or file when implementation or verification was requested.
- If verification could not be run, name the command or check that was not run and why.
- For a coding or file-edit task, a design summary, plan, or statement that edits are about to begin is not a valid final answer. Do not stop until the edit tool has returned and the requested checks have run, or an explicit blocker has been proven.
- Never claim `apply_patch` failed, aborted, or cannot edit a linked `/tmp` worktree unless the client returned an explicit tool error. Absolute paths to writable linked worktrees are supported. If no tool result is visible, retry the tool call instead of switching editing methods or finalizing.
- Preserve the caller's editing contract. Do not replace `apply_patch` with Python, `sed`, or another file-mutation mechanism when the task or repository requires `apply_patch`.
- A successful coding-task final answer must name the changed paths and requested verification results."""

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

_CODEX_AUTO_AGENT_MALFORMED_TOOL_CALL_COOLDOWN_SECONDS = 30.0 * 60.0

_CODEX_AUTO_AGENT_SPARK_MODEL = "gpt-5.3-codex-spark"

_CODEX_AUTO_AGENT_SPARK_DURABLE_COOLDOWN_SECONDS = 300.0

_CODEX_AUTO_AGENT_TRANSIENT_UPSTREAM_STATUS_CODES = frozenset({500, 502, 503, 529})

_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS = 8

_CODEX_AUTO_AGENT_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS_ENV = (
    "AAWM_NATIVE_GROK_CONTINUATION_TRANSIENT_MAX_ATTEMPTS"
)

# ── Extracted functions ─────────────────────────────────────────────

def _get_codex_auto_agent_header(headers: dict[str, Any], header_name: str) -> Optional[str]:
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
        session_header = _get_codex_auto_agent_header(headers, "session_id") or _get_codex_auto_agent_header(
            headers, "session-id"
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
    raw_value = _clean_codex_auth_value(os.getenv("AAWM_CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS"))
    if raw_value is None:
        return _CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS
    try:
        parsed = float(raw_value)
    except Exception:
        return _CODEX_AUTO_AGENT_LANE_STATE_CACHE_TTL_SECONDS
    return max(0.0, parsed)

def _codex_auto_agent_candidate_key(
    candidate: dict[str, Any],
    lane_key: str,
    *,
    cooldown_identity_tag: Optional[str] = None,
) -> str:
    """Build the canonical cooldown/evidence/probe state key for a candidate.

    CFG-019: when ``cooldown_identity_tag`` is set (snapshot-resolved aliases
    only) it is the stable per-candidate semantic identity tag
    ``alias:<canonical-alias>:{provider}:{model}:{route_family}`` and the key
    is prefixed with ``h{cooldown_identity_tag}:``. Scoping the tag to the
    owning alias plus the candidate's provider/model/resolved route semantics
    means an unrelated alias or config change preserves the cooldown, while a
    genuine semantic change to that candidate invalidates only its own
    identity. The global ``snapshot.config_hash`` is deliberately NOT part of
    cooldown identity.

    Static/legacy routes and Kimi managed-account keys pass no
    ``cooldown_identity_tag`` and keep bare keys (account-global by design).

    Affinity is explicitly excluded from this invalidation rule -- session
    pins survive config changes as long as the candidate remains compatible.
    """
    base = "{}:{}:{}".format(
        candidate["provider"],
        candidate["model"],
        lane_key or "__default__",
    )
    if cooldown_identity_tag:
        return "h{}:{}".format(cooldown_identity_tag, base)
    return base

_OPENROUTER_CREDENTIAL_LANE_RE = re.compile(r"\Aopenrouter:credential:[0-9a-f]{12}\Z")


def resolve_openrouter_credential_lane_key() -> str:
    """Derive a nonreversible OpenRouter credential lane from the same
    effective credential as egress; keep missing or unconfigured keys local."""
    from ..providers.openrouter.runtime import _get_openrouter_api_key

    try:
        credential = _get_openrouter_api_key()
    except RuntimeError:
        return "openrouter"
    if not credential:
        return "openrouter"
    return "openrouter:credential:" + _hash_codex_auto_agent_lane_value(credential)


def openrouter_credit_lane_cooldown_key(candidate: dict[str, Any], lane_key: Optional[str]) -> Optional[str]:
    # Never publish billing state under the legacy shared/default lane.
    if candidate.get("provider") != "openrouter" or not isinstance(lane_key, str):
        return None
    if re.fullmatch(r"openrouter:credential:[0-9a-f]{12}", lane_key) is None:
        return None
    return "openrouter:__credit_exhausted__:" + lane_key


def openrouter_account_lane_cooldown_key(
    candidate: dict[str, Any],
    lane_key: Optional[str],
) -> Optional[str]:
    """Credential-wide account key. Never publish under the shared default lane."""
    if candidate.get("provider") != "openrouter" or not isinstance(lane_key, str):
        return None
    if _OPENROUTER_CREDENTIAL_LANE_RE.fullmatch(lane_key) is None:
        return None
    return "openrouter:__account__:" + lane_key


_COHERE_UNSCOPED_CREDENTIAL_SENTINEL = "cohere"
_COHERE_CREDENTIAL_LANE_RE = re.compile(r"\Acohere:credential:[0-9a-f]{12}\Z")


def resolve_cohere_credential_lane_sentinel(
    credential: Optional[str] = None,
) -> str:
    """Return a one-way Cohere credential sentinel.

    An omitted credential resolves the canonical runtime key. An explicit
    credential is fingerprinted and discarded. Missing, blank, or unreadable
    credentials stay on an unscoped sentinel that cannot publish a shared
    cooldown. The result never contains credential material.
    """

    resolved = credential
    if credential is None:
        # Deferred: cohere runtime imports proxy types, and this module is
        # loaded before the proxy finishes importing.
        from ..providers.cohere.runtime import _get_cohere_api_key

        try:
            resolved = _get_cohere_api_key()
        except Exception:
            return _COHERE_UNSCOPED_CREDENTIAL_SENTINEL
    if not isinstance(resolved, str) or resolved == "":
        return _COHERE_UNSCOPED_CREDENTIAL_SENTINEL
    fingerprint = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return "cohere:credential:" + fingerprint


_COHERE_ATTEMPT_CREDENTIAL_SENTINEL_ATTR = "_aawm_cohere_attempt_credential_sentinel"


def clear_cohere_attempt_credential_sentinel(request: Any = None) -> None:
    """Drop a previous attempt's fingerprint before this attempt loads a key.

    A missing-key failure must not publish cooldown for the prior attempt.
    The current exception's stamped fingerprint is independent of this value.
    """

    state = getattr(request, "state", None)
    if state is None:
        return
    try:
        setattr(state, _COHERE_ATTEMPT_CREDENTIAL_SENTINEL_ATTR, None)
    except (AttributeError, TypeError):
        return


def capture_cohere_attempt_credential_sentinel(
    credential: str,
    *,
    request: Any = None,
) -> str:
    """Fingerprint the key this attempt selected and retain only the sentinel."""

    sentinel = resolve_cohere_credential_lane_sentinel(credential)
    _store_cohere_attempt_credential_sentinel(
        getattr(request, "state", None),
        sentinel,
    )
    return sentinel


def stamp_cohere_attempt_credential_sentinel(exc: BaseException, sentinel: str) -> None:
    """Bind a captured sentinel to the exception from that attempt."""

    _store_cohere_attempt_credential_sentinel(exc, sentinel)


def read_cohere_attempt_credential_sentinel(
    exc: Any = None,
    request: Any = None,
) -> Optional[str]:
    """Return the sentinel captured for this attempt.

    Publication must not re-read the canonical key. A missing or unscoped
    capture returns None so a later key is not cooled in its place.
    """

    for source in (exc, getattr(request, "state", None)):
        value = getattr(source, _COHERE_ATTEMPT_CREDENTIAL_SENTINEL_ATTR, None)
        if isinstance(value, str) and _COHERE_CREDENTIAL_LANE_RE.fullmatch(value):
            return value
    return None


def _store_cohere_attempt_credential_sentinel(target: Any, sentinel: str) -> None:
    if target is None or _COHERE_CREDENTIAL_LANE_RE.fullmatch(sentinel) is None:
        return
    try:
        setattr(target, _COHERE_ATTEMPT_CREDENTIAL_SENTINEL_ATTR, sentinel)
    except (AttributeError, TypeError):
        return


def cohere_credential_lane_cooldown_key(
    candidate: dict[str, Any],
    sentinel: Optional[str],
) -> Optional[str]:
    """Shared Cohere cooldown key for one hashed credential sentinel.

    Same-key siblings use this key. A different fingerprint, an unscoped
    sentinel, or any other provider returns None so those candidates stay
    independent. The key contains only the truncated digest.
    """

    if candidate.get("provider") != "cohere" or not isinstance(sentinel, str):
        return None
    if _COHERE_CREDENTIAL_LANE_RE.fullmatch(sentinel) is None:
        return None
    return "cohere:__credential__:" + sentinel


def _resolve_codex_auto_agent_xai_lane_key(candidate: dict[str, Any]) -> str:
    route_family = str(candidate.get("route_family") or "")
    if route_family in {
        "codex_xai_oauth_responses_adapter",
        "anthropic_xai_oauth_responses_adapter",
        "codex_auto_agent_xai_oauth_responses",
    }:
        account_lane = candidate.get("xai_oauth_lane_key")
        if isinstance(account_lane, str) and account_lane:
            return account_lane
        return _CODEX_AUTO_AGENT_XAI_OAUTH_LANE_KEY  # noqa: F821
    return _CODEX_AUTO_AGENT_XAI_LANE_KEY  # noqa: F821

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
