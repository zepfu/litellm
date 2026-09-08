"""Request-scoped actual-send accounting for managed xAI OAuth requests.

The counter is observational.  It does not impose a retry budget and it never
uses bearer material or caller-supplied identity.  Producers call it only
after their request-specific validation has passed and immediately before the
concrete transport send.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from starlette.requests import Request

MANAGED_XAI_SEND_REQUEST_KWARG = "_aawm_managed_xai_send_request"
MANAGED_XAI_SEND_COUNTER_STATE_KEY = "aawm_xai_oauth_actual_send_counter"
MANAGED_XAI_SEND_COUNTER_SNAPSHOT_STATE_KEY = (
    "aawm_xai_oauth_actual_send_counter_snapshot"
)
MANAGED_XAI_SEND_ORDINAL_STATE_KEY = "aawm_xai_oauth_actual_send_ordinal"
MANAGED_XAI_MAX_SEND_RECORDS = 32
_MAX_CONTEXT_VALUE_LENGTH = 256


def _safe_context_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    if not normalized:
        return None
    return normalized[:_MAX_CONTEXT_VALUE_LENGTH]


def _target_fingerprint(target: Any) -> Optional[str]:
    normalized = _safe_context_value(target)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ManagedXaiActualSendRecord:
    """Immutable, credential-free evidence for one concrete xAI send."""

    ordinal: int
    account_hash: str
    lane_key: str
    target_fingerprint: Optional[str]
    route_family: Optional[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "account_hash": self.account_hash,
            "lane_key": self.lane_key,
            "target_fingerprint": self.target_fingerprint,
            "route_family": self.route_family,
        }


class ManagedXaiActualSendCounter:
    """Monotonic request counter with a bounded immutable-record window."""

    def __init__(self, *, max_records: int = MANAGED_XAI_MAX_SEND_RECORDS) -> None:
        self.max_records = max(1, int(max_records))
        self._actual_send_count = 0
        self._records: deque[ManagedXaiActualSendRecord] = deque(
            maxlen=self.max_records
        )

    @property
    def actual_send_count(self) -> int:
        return self._actual_send_count

    @property
    def send_count(self) -> int:
        """Short alias used by request-local consumers."""

        return self._actual_send_count

    @property
    def records(self) -> tuple[ManagedXaiActualSendRecord, ...]:
        return tuple(self._records)

    @property
    def last_record(self) -> Optional[ManagedXaiActualSendRecord]:
        return self._records[-1] if self._records else None

    def record(
        self,
        *,
        account_hash: str,
        lane_key: str,
        target: Any = None,
        route_family: Any = None,
    ) -> ManagedXaiActualSendRecord:
        self._actual_send_count += 1
        record = ManagedXaiActualSendRecord(
            ordinal=self._actual_send_count,
            account_hash=account_hash,
            lane_key=lane_key,
            target_fingerprint=_target_fingerprint(target),
            route_family=_safe_context_value(route_family),
        )
        self._records.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        last_record = self.last_record
        return {
            "actual_send_count": self._actual_send_count,
            "next_send_ordinal": self._actual_send_count + 1,
            "last_send_ordinal": (
                last_record.ordinal if last_record is not None else None
            ),
            "max_records": self.max_records,
            "records": [record.to_metadata() for record in self._records],
        }


def _request_state(request: Any) -> Any:
    return getattr(request, "state", None)


def get_managed_xai_send_request(value: Any) -> Optional[Request]:
    """Resolve the private request handle without accepting metadata identity."""

    request = None
    if isinstance(value, Mapping):
        request = value.get(MANAGED_XAI_SEND_REQUEST_KWARG)
    else:
        get_fn = getattr(value, "get", None)
        if callable(get_fn):
            request = get_fn(MANAGED_XAI_SEND_REQUEST_KWARG)
        if request is None and _request_state(value) is not None:
            request = value
    if _request_state(request) is None:
        return None
    return request


def get_managed_xai_actual_send_counter(
    request: Any,
) -> Optional[ManagedXaiActualSendCounter]:
    state = _request_state(request)
    counter = (
        getattr(state, MANAGED_XAI_SEND_COUNTER_STATE_KEY, None)
        if state is not None
        else None
    )
    return (
        counter
        if isinstance(counter, ManagedXaiActualSendCounter)
        else None
    )


def get_or_create_managed_xai_actual_send_counter(
    request: Any,
) -> Optional[ManagedXaiActualSendCounter]:
    state = _request_state(request)
    if state is None:
        return None
    counter = get_managed_xai_actual_send_counter(request)
    if counter is None:
        counter = ManagedXaiActualSendCounter()
        setattr(state, MANAGED_XAI_SEND_COUNTER_STATE_KEY, counter)
    return counter


def get_managed_xai_actual_send_counter_snapshot(
    request: Any,
) -> Optional[dict[str, Any]]:
    counter = get_managed_xai_actual_send_counter(request)
    return counter.snapshot() if counter is not None else None


def _validated_xai_account_identity(
    request: Any,
) -> Optional[tuple[str, str]]:
    try:
        from litellm.proxy.pass_through_endpoints.aawm_alias_routing.xai_oauth import (
            get_bound_xai_oauth_selected_account,
        )

        selected = get_bound_xai_oauth_selected_account(request)
    except Exception:
        return None
    if selected is None:
        return None
    account_hash = _safe_context_value(getattr(selected, "account_hash", None))
    lane_key = _safe_context_value(getattr(selected, "lane_key", None))
    if account_hash is None or lane_key is None:
        return None
    return account_hash, lane_key


def record_managed_xai_actual_send(
    request: Any,
    *,
    target: Any = None,
    route_family: Any = None,
) -> Optional[ManagedXaiActualSendRecord]:
    """Record one send using only the currently validated server binding."""

    identity = _validated_xai_account_identity(request)
    if identity is None:
        return None
    counter = get_or_create_managed_xai_actual_send_counter(request)
    if counter is None:
        return None
    account_hash, lane_key = identity
    record = counter.record(
        account_hash=account_hash,
        lane_key=lane_key,
        target=target,
        route_family=route_family,
    )
    state = _request_state(request)
    if state is not None:
        setattr(
            state,
            MANAGED_XAI_SEND_COUNTER_SNAPSHOT_STATE_KEY,
            counter.snapshot(),
        )
        setattr(state, MANAGED_XAI_SEND_ORDINAL_STATE_KEY, record.ordinal)
    return record


__all__ = [
    "MANAGED_XAI_MAX_SEND_RECORDS",
    "MANAGED_XAI_SEND_COUNTER_STATE_KEY",
    "MANAGED_XAI_SEND_COUNTER_SNAPSHOT_STATE_KEY",
    "MANAGED_XAI_SEND_ORDINAL_STATE_KEY",
    "MANAGED_XAI_SEND_REQUEST_KWARG",
    "ManagedXaiActualSendCounter",
    "ManagedXaiActualSendRecord",
    "get_managed_xai_actual_send_counter",
    "get_managed_xai_actual_send_counter_snapshot",
    "get_or_create_managed_xai_actual_send_counter",
    "get_managed_xai_send_request",
    "record_managed_xai_actual_send",
]
