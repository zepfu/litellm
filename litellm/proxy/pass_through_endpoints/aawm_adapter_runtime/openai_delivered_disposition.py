"""Read the request-scoped final delivered wire disposition."""

from __future__ import annotations

from typing import Any, Dict, Optional

_TRACE_STATE_KEY = "_aawm_openai_responses_wire_trace"
_COMMITMENT_STATE_KEY = "_aawm_openai_responses_wire_commitment"
_SUCCESSFUL_DISPOSITIONS = {"completed"}
_FAILED_DISPOSITIONS = {"failed", "incomplete", "cancelled", "disconnected"}
_TRANSFER_PHASES = {
    "completed": "completed",
    "failed": "failed",
    "incomplete": "failed",
    "cancelled": "cancelled",
    "disconnected": "disconnected",
}


def _request_from_kwargs(kwargs: Any) -> Any:
    if not isinstance(kwargs, dict):
        return None
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return None
    proxy_request = litellm_params.get("proxy_server_request")
    if not isinstance(proxy_request, dict):
        return None
    return proxy_request.get("_request")


def _request_state(request: Any) -> Any:
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        getattr(state, _COMMITMENT_STATE_KEY, None)
    except Exception:
        return None
    return state


def get_delivered_wire_disposition(
    kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Return a copy of the delivered wire snapshot when a disposition exists."""

    request = _request_from_kwargs(kwargs)
    state = _request_state(request)
    if state is None:
        return None
    commitment = getattr(state, _COMMITMENT_STATE_KEY, None)
    if not isinstance(commitment, dict):
        trace = getattr(state, _TRACE_STATE_KEY, None)
        snapshot = getattr(trace, "snapshot", None)
        if callable(snapshot):
            try:
                commitment = snapshot()
            except Exception:
                return None
    if not isinstance(commitment, dict):
        return None
    snapshot = dict(commitment)
    disposition = str(snapshot.get("disposition") or "").strip().lower()
    snapshot["delivered_disposition"] = disposition or None
    snapshot["is_success"] = disposition in _SUCCESSFUL_DISPOSITIONS or None
    snapshot["is_failure"] = disposition in _FAILED_DISPOSITIONS or None
    snapshot["transfer_phase"] = _TRANSFER_PHASES.get(disposition)
    return snapshot
