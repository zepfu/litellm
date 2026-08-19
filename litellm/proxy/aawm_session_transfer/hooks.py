"""Best-effort transfer-registry hooks for pass-through and adapter streams."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from litellm.proxy.aawm_session_transfer.identity import extract_transfer_identity
from litellm.proxy.aawm_session_transfer.registry import (
    safe_finalize,
    safe_mark_phase,
    safe_record_chunks,
    safe_upsert,
)


def build_transfer_identity(
    *,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    logging_obj: Any = None,
    kwargs: Optional[Mapping[str, Any]] = None,
    litellm_call_id: Optional[str] = None,
    url_route: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    stream_path: str = "unknown",
) -> dict[str, Any]:
    return extract_transfer_identity(
        request=request,
        request_body=request_body,
        logging_obj=logging_obj,
        kwargs=kwargs,
        litellm_call_id=litellm_call_id,
        url_route=url_route,
        custom_llm_provider=custom_llm_provider,
        stream_path=stream_path,
    )


async def publish_transfer_phase(
    identity: Mapping[str, Any],
    phase: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    await safe_mark_phase(identity, phase, extra)


async def publish_transfer_chunks(
    identity: Mapping[str, Any],
    **kwargs: Any,
) -> None:
    await safe_record_chunks(identity, **kwargs)


async def publish_transfer_terminal(
    identity: Mapping[str, Any],
    phase: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    await safe_finalize(identity, phase, extra)


async def publish_adapter_transfer_event(
    *,
    request: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    logging_obj: Any = None,
    kwargs: Optional[Mapping[str, Any]] = None,
    litellm_call_id: Optional[str] = None,
    url_route: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    phase: str = "request_received",
    extra: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Public hook for remaining adapter StreamingResponse paths.

    First landing instruments the central pass-through stream handler. Adapter
    authors should call this helper from their own generators rather than
    inventing a second registry.
    """
    identity = build_transfer_identity(
        request=request,
        request_body=request_body,
        logging_obj=logging_obj,
        kwargs=kwargs,
        litellm_call_id=litellm_call_id,
        url_route=url_route,
        custom_llm_provider=custom_llm_provider,
        stream_path="adapter",
    )
    if extra:
        await safe_upsert(identity, extra, force=True)
    return await safe_mark_phase(identity, phase, extra)
