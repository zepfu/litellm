"""Retain bounded outcomes for Langfuse v2's actual SDK ingestion attempts."""

from __future__ import annotations

import hashlib
import logging
import re
from threading import local
from typing import Any, Callable

import httpx

_logger = logging.getLogger("langfuse")
_event_id = re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}")


def _batch_evidence(batch: list[dict[str, Any]]) -> dict[str, Any] | None:
    event_ids = [
        event_id
        for event in batch
        if isinstance(event, dict) and isinstance(event_id := event.get("id"), str) and _event_id.fullmatch(event_id)
    ]
    if not event_ids:
        return None
    return {
        "langfuse_upload_batch_id": hashlib.sha256(",".join(sorted(event_ids)).encode("ascii")).hexdigest(),
        "langfuse_upload_batch_size": len(batch),
        "langfuse_upload_event_ids": event_ids[:5],
    }


def _observe_terminal_batch(
    upload_batch: Callable[[list[dict[str, Any]]], None], worker: local
) -> Callable[[list[dict[str, Any]]], None]:
    def observed_upload(batch: list[dict[str, Any]]) -> None:
        worker.batch_post_entered = False
        try:
            return upload_batch(batch)
        except Exception as error:
            try:
                evidence = _batch_evidence(batch) or {"langfuse_upload_batch_size": len(batch)}
                _logger.error(
                    "Langfuse SDK ingestion ended with a terminal batch failure",
                    extra={
                        **evidence,
                        "langfuse_upload_outcome": "terminal_sdk_error",
                        "langfuse_upload_exception_type": type(error).__name__[:96],
                        "langfuse_upload_post_entered": worker.batch_post_entered,
                    },
                )
            except Exception:
                # Keep the SDK's terminal exception and queue acknowledgement.
                pass
            raise
        finally:
            worker.batch_post_entered = False

    return observed_upload


def observe_langfuse_uploads(sdk: Any) -> None:
    """Wrap only this SDK instance's ingestion call and preserve its retry policy."""
    # The v2 SDK starts consumers at construction and supplies this callable
    # from their shared, per-instance client. Attach before enqueueing events.
    try:
        from langfuse.request import APIError, APIErrors

        manager = sdk.task_manager
        batch_post = manager._client.batch_post
    except (AttributeError, ImportError):
        _logger.warning("Langfuse SDK ingestion evidence is unavailable for this client")
        return
    worker = local()

    def observed_batch_post(*args: Any, **kwargs: Any) -> httpx.Response:
        worker.batch_post_entered = True
        try:
            response = batch_post(*args, **kwargs)
        except Exception as error:
            try:
                evidence = _batch_evidence(kwargs["batch"])
                if evidence is not None:
                    statuses = set()
                    failures = error.errors if isinstance(error, APIErrors) else [error]
                    for failure in failures:
                        if isinstance(failure, APIError):
                            raw_status = str(failure.status)
                            if re.fullmatch(r"[1-5][0-9]{2}", raw_status):
                                statuses.add(int(raw_status))
                    if isinstance(error, httpx.TimeoutException):
                        outcome = "timeout"
                    elif isinstance(error, httpx.TransportError):
                        outcome = "transport_error"
                    elif isinstance(error, APIErrors):
                        outcome = "batch_item_rejected"
                    elif isinstance(error, APIError):
                        outcome = "http_rejected"
                    else:
                        outcome = "sdk_error"
                    worker.failed_batch = evidence["langfuse_upload_batch_id"]
                    _logger.error(
                        "Langfuse ingestion attempt failed for batch %s (%s)",
                        worker.failed_batch,
                        outcome,
                        extra={
                            **evidence,
                            "langfuse_upload_outcome": outcome,
                            "langfuse_upload_statuses": sorted(statuses),
                        },
                    )
            except Exception:
                # An evidence failure must preserve the SDK's original error.
                pass
            raise

        failed_batch = getattr(worker, "failed_batch", None)
        if failed_batch is not None:
            worker.failed_batch = None
            try:
                evidence = _batch_evidence(kwargs["batch"])
                if evidence is not None and evidence["langfuse_upload_batch_id"] == failed_batch:
                    _logger.warning(
                        "Langfuse ingestion retry accepted for batch %s (HTTP %d)",
                        failed_batch,
                        response.status_code,
                        extra={
                            **evidence,
                            "langfuse_upload_outcome": "accepted_after_failure",
                            "langfuse_upload_statuses": [response.status_code],
                        },
                    )
            except Exception:
                # Evidence emission cannot change an accepted upload.
                pass
        return response

    manager._client.batch_post = observed_batch_post

    for consumer in getattr(manager, "_ingestion_consumers", ()):
        upload_batch = getattr(consumer, "_upload_batch", None)
        if callable(upload_batch):
            consumer._upload_batch = _observe_terminal_batch(upload_batch, worker)
