"""Wave 4 extraction: google_error_signals pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import re
import time  # noqa: F401
from typing import Any, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global modules
    _anthropic_google_shaping: Any

    # Host-global functions
    def _extract_adapter_upstream_headers(exc: Any) -> dict: ...
    def _parse_retry_after_seconds_from_headers(headers: dict) -> Optional[float]: ...
    def _parse_rate_limit_reset_wait_seconds_from_headers(headers: dict) -> Optional[float]: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_extract_google_adapter_exception_status_code",
    "_extract_google_adapter_exception_detail",
    "_parse_google_rate_limit_reset_seconds",
    "_extract_google_adapter_error_payloads",
    "_extract_google_adapter_error_reason",
    "_extract_google_adapter_error_payload_for_logging",
    "_record_google_adapter_error_for_logging",
    "_build_google_adapter_terminal_error_log_context",
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


# ── Extracted functions ─────────────────────────────────────────────

def _extract_google_adapter_exception_status_code(exc: Any) -> Optional[int]:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._extract_google_adapter_exception_status_code(exc)  # noqa: F821

def _extract_google_adapter_exception_detail(exc: Any) -> Any:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._extract_google_adapter_exception_detail(exc)  # noqa: F821

def _parse_google_rate_limit_reset_seconds(exc: Any) -> float:
    upstream_headers = _extract_adapter_upstream_headers(exc)
    retry_after_seconds = _parse_retry_after_seconds_from_headers(upstream_headers)
    if retry_after_seconds is not None:
        return max(1.0, retry_after_seconds)
    reset_wait_seconds = _parse_rate_limit_reset_wait_seconds_from_headers(upstream_headers)
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
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._extract_google_adapter_error_payloads(exc)  # noqa: F821

def _extract_google_adapter_error_reason(exc: Any) -> Optional[str]:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._extract_google_adapter_error_reason(exc)  # noqa: F821

def _extract_google_adapter_error_payload_for_logging(exc: Any) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._extract_google_adapter_error_payload_for_logging(exc)  # noqa: F821

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
    metadata["google_generate_content_error_count"] = int(metadata.get("google_generate_content_error_count") or 0) + 1

def _build_google_adapter_terminal_error_log_context(
    passthrough_kwargs: dict[str, Any], *, status_code: Optional[int], failure_classification: Optional[str]
) -> dict[str, Any]:
    _anthropic_google_shaping.bind_runtime(globals())  # noqa: F821
    return _anthropic_google_shaping._build_google_adapter_terminal_error_log_context(  # noqa: F821
        passthrough_kwargs, status_code=status_code, failure_classification=failure_classification
    )
