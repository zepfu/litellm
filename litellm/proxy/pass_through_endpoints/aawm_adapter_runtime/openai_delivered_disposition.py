"""Read the immutable delivered OpenAI Responses disposition."""

from __future__ import annotations

from typing import Any, Dict, Optional

_DELIVERED_STATE_KEY = "_aawm_openai_responses_delivered_snapshot"
_SUCCESSFUL_DISPOSITIONS = {"completed"}
_FAILED_DISPOSITIONS = {
    "failed",
    "incomplete",
    "cancelled",
    "disconnected",
}
_TRANSFER_PHASES = {
    "completed": "completed",
    "failed": "failed",
    "incomplete": "failed",
    "cancelled": "cancelled",
    "disconnected": "disconnected",
}


def annotate_delivered_wire_failure(
    exception: Exception,
    *,
    delivered_disposition: str,
) -> Exception:
    """Mark failure logging as retaining provider response evidence."""

    normalized_disposition = str(delivered_disposition or "").strip().lower()
    setattr(exception, "_aawm_preserve_response_evidence", True)
    setattr(
        exception,
        "delivered_disposition",
        normalized_disposition or "unknown",
    )
    return exception


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


def get_delivered_wire_disposition(
    kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Return only the post-ASGI immutable delivered snapshot.

    Provisional commitment snapshots and the mutable live trace are
    intentionally ignored. Consumers must wait for the response wrapper to
    publish this snapshot after finalization.
    """

    request = _request_from_kwargs(kwargs)
    state = getattr(request, "state", None)
    if state is None:
        return None
    try:
        snapshot = getattr(state, _DELIVERED_STATE_KEY, None)
    except Exception:
        return None
    if not isinstance(snapshot, dict):
        return None
    if (
        snapshot.get("finalized") is not True
        or snapshot.get("asgi_delivery_complete") is not True
    ):
        return None

    disposition = str(snapshot.get("disposition") or "").strip().lower()
    if disposition not in _SUCCESSFUL_DISPOSITIONS | _FAILED_DISPOSITIONS:
        return None

    delivered_snapshot = dict(snapshot)
    delivered_snapshot["delivered_disposition"] = disposition
    delivered_snapshot["is_success"] = disposition in _SUCCESSFUL_DISPOSITIONS
    delivered_snapshot["is_failure"] = disposition in _FAILED_DISPOSITIONS
    delivered_snapshot["transfer_phase"] = _TRANSFER_PHASES[disposition]
    return delivered_snapshot

