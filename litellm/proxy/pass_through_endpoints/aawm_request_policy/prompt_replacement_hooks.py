"""Prompt-replacement early request-hook integration (CFG-073).

Applies shared prompt-replacement rules before routing/conversion at three
insertion points:

1. Standard proxy pre-call (common_request_processing.py)
2. OpenAI passthrough ingress (openai_passthrough_handler.py)
3. Anthropic / Gemini native ingress (llm_passthrough_endpoints.py)

Only the insertion points and identity extraction are owned here; the
engine, manifest loading, cache, rule matchers, and Claude compatibility
wrapper belong to the engine owner.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_agent_identity import _parse_client_identity_from_user_agent
from litellm.proxy.pass_through_endpoints.aawm_request_policy.prompt_replacement_engine import (
    ReplacementRule,
    _DEFAULT_MANIFEST_CACHE,
    ReplacementResult,
    apply_rules_to_anthropic_request,
    apply_rules_to_chat_completions_messages,
    apply_rules_to_gemini_request,
    apply_rules_to_responses_request,
    build_replacement_metadata,
    load_manifests,
)
from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import (
    _merge_litellm_metadata,
)

# ---------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MANIFEST_DIR = _REPO_ROOT / "context-replacement" / "shared"

# Installed-package fallback: wheel data lands under the same parent
# package as the existing Claude control-plane data directory.
_PACKAGED_MANIFEST_DIR: Optional[Path] = None
_claude_cp_root = Path(__file__).resolve().parent.parent / "aawm_claude_control_plane_data"
_packaged_candidate = _claude_cp_root / "context-replacement" / "shared"
if _packaged_candidate.is_dir():
    _PACKAGED_MANIFEST_DIR = _packaged_candidate
del _claude_cp_root, _packaged_candidate

# Shared cache reused across all callers (engine, claude_prompt_replacement,
# and our hooks).  Owned by the engine owner; we reference it here to avoid
# a duplicate module-level cache.
_shared_rule_cache = _DEFAULT_MANIFEST_CACHE


def _discover_manifest_paths() -> list[Path]:
    """Discover prompt-replacement manifest files.

    Checks ``AAWM_PROMPT_REPLACEMENT_MANIFEST_DIR`` env var first, then the
    default ``context-replacement/shared/`` directory.
    """
    env_dir = os.getenv("AAWM_PROMPT_REPLACEMENT_MANIFEST_DIR")
    if env_dir:
        candidate = Path(env_dir)
        if candidate.is_dir():
            return sorted(candidate.glob("*.json"))
    if _DEFAULT_MANIFEST_DIR.is_dir():
        return sorted(_DEFAULT_MANIFEST_DIR.glob("*.json"))
    if _PACKAGED_MANIFEST_DIR is not None and _PACKAGED_MANIFEST_DIR.is_dir():
        return sorted(_PACKAGED_MANIFEST_DIR.glob("*.json"))
    return []


def _get_rules() -> tuple[list[ReplacementRule], list[str]]:
    """Return (rules, logical_paths) from discovered manifests.

    Thin wrapper around the engine-owned ``load_manifests`` with a shared
    cache.  Returns ``([], [])`` when no manifests are found.  Let
    ``load_manifest`` / ``load_manifests`` raise on malformed manifests so
    failures are observable.
    """
    paths = _discover_manifest_paths()
    if not paths:
        return [], []

    rules = load_manifests(paths, cache=_shared_rule_cache)
    logical_paths = [str(p) for p in paths]
    return rules, logical_paths


# ---------------------------------------------------------------------------
# Identity extraction helpers
# ---------------------------------------------------------------------------

_SOURCE_TUI_HEADER_NAMES = (
    "x-aawm-client-name",
    "x-litellm-client-name",
    "x-client-name",
)
_CC_VERSION_HEADER_NAMES = (
    "x-aawm-client-version",
    "x-litellm-client-version",
    "x-client-version",
)


def _normalize_tui_family(label: Optional[str]) -> Optional[str]:
    """Normalize client product label to a stable TUI family."""
    if not label:
        return None
    product = label.split("/", 1)[0].strip().lower()
    product = product.replace("-", "").replace("_", "").replace(" ", "")

    if product in ("codex", "codexcli", "codextui", "codexclirs"):
        return "codex"
    if product in ("claude", "claudecode"):
        return "claude"
    if product in ("grok", "grokbuild"):
        return "grok"
    if product in ("qwen", "qwenchat", "qwencode", "qwencodecli"):
        return "qwen"
    if product in ("kimi", "kimichat", "kimicode", "kimicodecli"):
        return "kimi"
    if product in ("ohmypi", "omp", "ompla"):
        return "ohmypi"
    return label.split("/", 1)[0].strip().lower()


def _extract_source_tui(
    request: Any,
    data: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Extract normalized TUI family from request headers and body metadata."""
    if isinstance(data, dict):
        metadata = data.get("litellm_metadata") or data.get("metadata")
        if isinstance(metadata, dict):
            name = metadata.get("client_name")
            if isinstance(name, str) and name.strip():
                tui = _normalize_tui_family(name.strip())
                if tui:
                    return tui

    headers = _safe_get_headers(request)
    for header_name in _SOURCE_TUI_HEADER_NAMES:
        name = headers.get(header_name)
        if isinstance(name, str) and name.strip():
            tui = _normalize_tui_family(name.strip())
            if tui:
                return tui

    name, _ = _parse_client_identity_from_user_agent(headers.get("user-agent"))
    return _normalize_tui_family(name)


def _extract_cc_version(
    request: Any,
    data: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Extract client version from request headers and body metadata."""
    if isinstance(data, dict):
        metadata = data.get("litellm_metadata") or data.get("metadata")
        if isinstance(metadata, dict):
            version = metadata.get("client_version") or metadata.get("cc_version")
            if isinstance(version, str) and version.strip():
                return version.strip()

    headers = _safe_get_headers(request)
    for header_name in _CC_VERSION_HEADER_NAMES:
        version = headers.get(header_name)
        if isinstance(version, str) and version.strip():
            return version.strip()

    _, version = _parse_client_identity_from_user_agent(headers.get("user-agent"))
    return version


def _safe_get_headers(request: Any) -> dict[str, str]:
    """Extract headers from a Starlette Request, Headers, or dict.

    Starlette ``request.headers`` is a ``Headers`` (MutableMapping), not a
    plain ``dict``.  Convert it so identity and filter lookups work.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return {}
    if isinstance(headers, dict):
        return headers
    # Starlette Headers, werkzeug EnvironHeaders, etc.
    try:
        return dict(headers)
    except (TypeError, ValueError, AttributeError):
        return {}


# ---------------------------------------------------------------------------
# Metadata merge (logging-only; never leaks into provider JSON)
# ---------------------------------------------------------------------------

def _merge_replacement_metadata(
    body: dict[str, Any],
    result: ReplacementResult,
    *,
    manifest_paths: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Merge prompt-replacement metadata into the body's litellm_metadata.

    This metadata lives under ``litellm_metadata`` (stripped before egress
    by the existing observability contract); it never reaches the provider.
    """
    if not result.matched and not result.failures:
        return body

    meta = build_replacement_metadata(result, manifest_paths=manifest_paths)
    tags = ["prompt-replacement"]
    for rule_id in result.matched_ids:
        tags.append(f"prompt-replacement-rule:{rule_id}")
    if result.failures:
        tags.append("prompt-replacement-rule-failed")

    return _merge_litellm_metadata(
        body,
        tags_to_add=tags,
        extra_fields={"prompt_replacement": meta},
    )


# ---------------------------------------------------------------------------
# Per-insertion-point hook functions
# ---------------------------------------------------------------------------

# Recognised standard route types for prompt replacement.
_STANDARD_ROUTE_TYPES = frozenset({
    "acompletion",
    "aresponses",
    "anthropic_messages",
    "agenerate_content",
    "agenerate_content_stream",
})


def apply_prompt_replacement_to_standard_request(
    data: dict[str, Any],
    *,
    route_type: str,
    request: Any,
) -> dict[str, Any]:
    """Apply prompt-replacement rules inside ``common_processing_pre_call_logic``.

    Called after call-id / identity setup and before the pre-call hook.
    Only touches Chat, Responses, Anthropic, and Gemini route types.
    """
    if route_type not in _STANDARD_ROUTE_TYPES:
        return data

    try:
        rules, manifest_paths = _get_rules()
    except Exception:
        verbose_proxy_logger.exception(
            "prompt_replacement: manifest load failed in standard hook; "
            "passing request through unchanged.  Fix the manifest files "
            "at the configured manifest directory or remove the "
            "AAWM_PROMPT_REPLACEMENT_MANIFEST_DIR override."
        )
        return _merge_replacement_metadata(
            data, ReplacementResult(failures=[{"id": "manifest-load", "status": "failed", "error": "manifest-load-failed"}]),
        )

    if not rules:
        return data

    source_tui = _extract_source_tui(request, data)
    cc_version = _extract_cc_version(request, data)

    if route_type == "acompletion":
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data
        updated_messages, result = apply_rules_to_chat_completions_messages(
            messages, rules, source_tui=source_tui, cc_version=cc_version,
        )
        if not result.matched and not result.failures:
            return data
        data = dict(data)
        data["messages"] = updated_messages
        return _merge_replacement_metadata(data, result, manifest_paths=manifest_paths)

    if route_type == "aresponses":
        updated_body, result = apply_rules_to_responses_request(
            data, rules, source_tui=source_tui, cc_version=cc_version,
        )
        if not result.matched and not result.failures:
            return data
        return _merge_replacement_metadata(
            updated_body, result, manifest_paths=manifest_paths,
        )

    if route_type == "anthropic_messages":
        updated_body, result = apply_rules_to_anthropic_request(
            data, rules, source_tui=source_tui, cc_version=cc_version,
        )
        if not result.matched and not result.failures:
            return data
        return _merge_replacement_metadata(
            updated_body, result, manifest_paths=manifest_paths,
        )

    if route_type in ("agenerate_content", "agenerate_content_stream"):
        updated_body, result = apply_rules_to_gemini_request(
            data, rules, source_tui=source_tui, cc_version=cc_version,
        )
        if not result.matched and not result.failures:
            return data
        return _merge_replacement_metadata(
            updated_body, result, manifest_paths=manifest_paths,
        )

    return data


def apply_prompt_replacement_to_passthrough_body(
    body: dict[str, Any],
    *,
    request: Any,
    adapter: str,
    cc_version: Optional[str] = None,
) -> dict[str, Any]:
    """Apply prompt-replacement rules inside a passthrough handler.

    Called after body parse and before alias/routing decisions.
    *adapter* selects the API-surface adapter; *cc_version* overrides the
    header-extracted version (e.g. from Anthropic billing header).
    """
    try:
        rules, manifest_paths = _get_rules()
    except Exception:
        verbose_proxy_logger.exception(
            "prompt_replacement: manifest load failed in passthrough hook; "
            "passing request through unchanged.  Fix the manifest files "
            "at the configured manifest directory or remove the "
            "AAWM_PROMPT_REPLACEMENT_MANIFEST_DIR override."
        )
        return _merge_replacement_metadata(
            body, ReplacementResult(failures=[{"id": "manifest-load", "status": "failed", "error": "manifest-load-failed"}]),
        )

    if not rules:
        return body

    source_tui = _extract_source_tui(request, body)
    version = cc_version or _extract_cc_version(request, body)

    if adapter == "chat":
        messages = body.get("messages")
        if not isinstance(messages, list):
            return body
        updated_messages, result = apply_rules_to_chat_completions_messages(
            messages, rules, source_tui=source_tui, cc_version=version,
        )
        if not result.matched and not result.failures:
            return body
        body = dict(body)
        body["messages"] = updated_messages
        return _merge_replacement_metadata(body, result, manifest_paths=manifest_paths)

    if adapter == "responses":
        updated_body, result = apply_rules_to_responses_request(
            body, rules, source_tui=source_tui, cc_version=version,
        )
        if not result.matched and not result.failures:
            return body
        return _merge_replacement_metadata(
            updated_body, result, manifest_paths=manifest_paths,
        )

    if adapter == "anthropic":
        updated_body, result = apply_rules_to_anthropic_request(
            body, rules, source_tui=source_tui, cc_version=version,
        )
        if not result.matched and not result.failures:
            return body
        return _merge_replacement_metadata(
            updated_body, result, manifest_paths=manifest_paths,
        )

    if adapter == "gemini":
        updated_body, result = apply_rules_to_gemini_request(
            body, rules, source_tui=source_tui, cc_version=version,
        )
        if not result.matched and not result.failures:
            return body
        nested = updated_body.get("custom_body")
        if isinstance(nested, dict):
            updated_body["custom_body"] = _merge_replacement_metadata(
                nested, result, manifest_paths=manifest_paths,
            )
            return updated_body
        return _merge_replacement_metadata(
            updated_body, result, manifest_paths=manifest_paths,
        )

    return body


# ---------------------------------------------------------------------------
# Owned symbols
# ---------------------------------------------------------------------------

OWNED_SYMBOLS: tuple[str, ...] = (
    "apply_prompt_replacement_to_standard_request",
    "apply_prompt_replacement_to_passthrough_body",
)
