"""Request and host metadata helpers for auto-agent alias routing.

Wave 7 extraction from ``llm_passthrough_endpoints.py``. Behavior-preserving
relocation only; no logic changes.

The owner module imports its auth, request, URL parsing, and host-attribution
dependencies from their owning modules. Because installed functions are rebound
to the host namespace, :func:`install` also publishes those dependencies into
that namespace. Session extraction and lane-key header lookup remain injected
through :func:`configure_request_metadata_runtime`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from urllib.parse import parse_qsl, urlencode, urlparse

from fastapi import Request

from litellm.proxy.aawm_route_logging import (
    _select_aawm_route_host_attribution_for_request,
    aresolve_aawm_route_host_attribution,
    resolve_aawm_route_host_attribution,
    _set_aawm_route_host_attribution_request_state,
)
from litellm.proxy.common_utils.http_parsing_utils import (
    _safe_get_request_headers,
)

from .codex_oauth import _clean_codex_auth_value

# ---------------------------------------------------------------------------
# Injected runtime seams (cross-module)
# ---------------------------------------------------------------------------

_extract_passthrough_session_id: Optional[
    Callable[..., Optional[str]]
] = None

_get_codex_auto_agent_header: Optional[
    Callable[[dict, str], Optional[str]]
] = None

_HOST_FUNCTION_NAMES = (
    "_extract_auto_agent_alias_session_id",
    "_extract_auto_agent_alias_metadata_value",
    "_normalize_auto_agent_alias_client_product",
    "_extract_auto_agent_alias_client_product_label",
    "_normalize_tui_family",
    "_extract_auto_agent_alias_incoming_endpoint",
    "_resolve_auto_agent_alias_route_host_attribution",
    "_aresolve_auto_agent_alias_route_host_attribution",
)

_host_globals: Optional[dict] = None


def configure_request_metadata_runtime(
    *,
    extract_passthrough_session_id: Callable[..., Optional[str]],
    get_codex_auto_agent_header: Callable[[dict, str], Optional[str]],
) -> None:
    """Inject dependencies before or after installation into host globals."""
    global _extract_passthrough_session_id, _get_codex_auto_agent_header

    _extract_passthrough_session_id = extract_passthrough_session_id
    _get_codex_auto_agent_header = get_codex_auto_agent_header

    if _host_globals is not None:
        _host_globals[
            "_extract_passthrough_session_id"
        ] = extract_passthrough_session_id
        if _host_globals.get("_get_codex_auto_agent_header") is None:
            _host_globals[
                "_get_codex_auto_agent_header"
            ] = get_codex_auto_agent_header
        _host_globals.update(
            {name: globals()[name] for name in _HOST_FUNCTION_NAMES}
        )


# ---------------------------------------------------------------------------
# Frozen functions (baseline llm_passthrough_endpoints.py)
# ---------------------------------------------------------------------------


def _extract_auto_agent_alias_session_id(
    request: Request,
    request_body: dict[str, Any],
) -> Optional[str]:
    metadata = request_body.get("litellm_metadata")
    if isinstance(metadata, dict):
        session_id = _clean_codex_auth_value(metadata.get("session_id"))
        if session_id is not None:
            return session_id
    if _extract_passthrough_session_id is None:
        raise RuntimeError(
            "request_metadata runtime not configured: "
            "missing extract_passthrough_session_id"
        )
    passthrough_session_id = _extract_passthrough_session_id(request, request_body)
    if passthrough_session_id is not None:
        return passthrough_session_id
    headers = _safe_get_request_headers(request)
    if _get_codex_auto_agent_header is None:
        raise RuntimeError(
            "request_metadata runtime not configured: "
            "missing get_codex_auto_agent_header"
        )
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
    elif normalized_name in {"qwen", "qwen-code", "qwen-code-cli"}:
        name = "Qwen"
    elif normalized_name in {"kimi", "kimi-code", "kimi-code-cli"}:
        name = "Kimi"
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
            if value and not (
                key in {"client_user_agent", "user_agent"}
                and _normalize_tui_family(value) == "qwen"
            ):
                return value
        name = _normalize_auto_agent_alias_client_product(metadata.get("client_name"))
        version = _clean_codex_auth_value(metadata.get("client_version"))
        if name and version and "/" not in name:
            return f"{name}/{version}"
        if name:
            return name
    headers = _safe_get_request_headers(request)
    if _get_codex_auto_agent_header is None:
        raise RuntimeError(
            "request_metadata runtime not configured: "
            "missing get_codex_auto_agent_header"
        )
    for header_name in (
        "x-aawm-client",
        "x-litellm-client",
        "x-client-name-version",
        "user-agent",
    ):
        value = _normalize_auto_agent_alias_client_product(
            _get_codex_auto_agent_header(headers, header_name)
        )
        if value and not (
            header_name == "user-agent"
            and _normalize_tui_family(value) == "qwen"
        ):
            return value
    return None



def _normalize_tui_family(client_product_label: Optional[str]) -> str:
    """Normalize client product label to stable TUI family (CFG-007).

    Returns one of: codex, claude, grok, qwen, kimi, or unknown.
    Versions are stripped; only the product family matters for dispatch.
    """
    if not client_product_label:
        return "unknown"

    product = client_product_label.split("/", 1)[0].strip().lower()
    product = product.replace("-", "").replace("_", "")

    if product in ("codex", "codexcli", "codextui"):
        return "codex"
    if product in ("claude", "claudecode"):
        return "claude"
    if product in ("grok", "grokbuild"):
        return "grok"
    if product in ("qwen", "qwenchat", "qwencode", "qwencodecli"):
        return "qwen"
    if product in ("kimi", "kimichat", "kimicode", "kimicodecli"):
        return "kimi"
    return "unknown"


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


def _resolve_auto_agent_alias_route_host_attribution(
    request: Request,
) -> dict[str, Optional[str]]:
    """Non-blocking host attribution for sync audit/event builders (RR-054 #4).

    Reverse-DNS is never performed inline on the event loop: the shared resolver
    defaults to ``allow_blocking_lookup=False`` and schedules background enrichment.
    Async request paths that need a full lookup should await
    ``_aresolve_auto_agent_alias_route_host_attribution``.
    """
    cached = _select_aawm_route_host_attribution_for_request(
        request=request,
        metadata={},
    )
    if cached is not None:
        return cached
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
    cached = _select_aawm_route_host_attribution_for_request(
        request=request,
        metadata={},
    )
    if cached is not None:
        return cached
    try:
        attribution = await aresolve_aawm_route_host_attribution(
            request,
            allow_blocking_lookup=True,
        )
        _set_aawm_route_host_attribution_request_state(request, attribution)
        return attribution
    except Exception:
        return {
            "client_ip": None,
            "client_ip_source": None,
            "host_name": None,
            "host_name_source": None,
        }


# ---------------------------------------------------------------------------
# Host-global install (serial integration)
# ---------------------------------------------------------------------------

from types import FunctionType as _FunctionType  # noqa: E402


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    Imported runtime dependencies and both injected seam names are also
    published so the host need not pre-populate request-metadata globals.
    """
    global _host_globals
    _host_globals = host_globals
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = _FunctionType(
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

    # Rebound functions resolve imported dependencies through host_globals.
    for _dependency_name, _dependency in (
        ("_clean_codex_auth_value", _clean_codex_auth_value),
        ("_safe_get_request_headers", _safe_get_request_headers),
        ("parse_qsl", parse_qsl),
        ("urlencode", urlencode),
        ("urlparse", urlparse),
        (
            "_select_aawm_route_host_attribution_for_request",
            _select_aawm_route_host_attribution_for_request,
        ),
        (
            "resolve_aawm_route_host_attribution",
            resolve_aawm_route_host_attribution,
        ),
        (
            "aresolve_aawm_route_host_attribution",
            aresolve_aawm_route_host_attribution,
        ),
        (
            "_set_aawm_route_host_attribution_request_state",
            _set_aawm_route_host_attribution_request_state,
        ),
        ("_normalize_tui_family", _normalize_tui_family),
    ):
        host_globals.setdefault(_dependency_name, _dependency)

    # Always seed seam names. Explicit None values make unconfigured calls
    # reach the RuntimeError guards instead of failing with NameError.
    for _sk, _sv in (
        ("_extract_passthrough_session_id", _extract_passthrough_session_id),
        ("_get_codex_auto_agent_header", _get_codex_auto_agent_header),
    ):
        host_globals.setdefault(_sk, _sv)
