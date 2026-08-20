"""Safe failure classification for Z.AI Coding Plan provider calls.

This module only classifies provider failures for later consumers. It does not
perform retries, candidate routing, cooldowns, logging, or credential refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

import httpx

from .chat.transformation import ZAI_CODING_PLAN_MODEL_IDS


class ZAICodingPlanFailureKind(str, Enum):
    """Normalized Coding Plan failure categories."""

    AUTH = "auth"
    QUOTA = "quota"
    RATE = "rate"
    CAPACITY = "capacity"
    VALIDATION = "validation"
    ROUTING = "routing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ZAICodingPlanFailureMetadata:
    """Safe, normalized metadata for a Coding Plan failure."""

    kind: ZAICodingPlanFailureKind
    upstream_id: Optional[str]
    status_code: Optional[int]
    reset_reason: str

    def to_safe_metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "upstream_id": self.upstream_id,
            "status_code": self.status_code,
            "reset_reason": self.reset_reason,
        }


_CODE_CLASSIFICATION: dict[int, tuple[ZAICodingPlanFailureKind, str]] = {
    1000: (ZAICodingPlanFailureKind.AUTH, "credentials_rejected"),
    1001: (ZAICodingPlanFailureKind.AUTH, "missing_authorization"),
    1113: (ZAICodingPlanFailureKind.ROUTING, "wrong_base_or_non_coding_key"),
    1211: (ZAICodingPlanFailureKind.VALIDATION, "unknown_model"),
    1302: (ZAICodingPlanFailureKind.RATE, "request_rate_limit"),
    1308: (ZAICodingPlanFailureKind.QUOTA, "time_window_exhausted"),
    1309: (ZAICodingPlanFailureKind.QUOTA, "subscription_expired"),
    1310: (ZAICodingPlanFailureKind.QUOTA, "plan_window_exhausted"),
    1311: (ZAICodingPlanFailureKind.VALIDATION, "model_not_in_subscription"),
    1313: (ZAICodingPlanFailureKind.QUOTA, "fair_use_restriction"),
    1316: (ZAICodingPlanFailureKind.QUOTA, "five_hour_window_exhausted"),
    1317: (ZAICodingPlanFailureKind.QUOTA, "seven_day_window_exhausted"),
}


def classify_zai_coding_plan_failure(
    *,
    status_code: object,
    error_code: object = None,
    message: object = None,
    upstream_id: object = None,
) -> ZAICodingPlanFailureMetadata:
    """Classify a Coding Plan error without retaining raw provider detail."""

    _ = message
    normalized_code = _safe_error_code(error_code)
    if normalized_code in _CODE_CLASSIFICATION:
        kind, reset_reason = _CODE_CLASSIFICATION[normalized_code]
    else:
        kind, reset_reason = ZAICodingPlanFailureKind.UNKNOWN, "unclassified_failure"
    return ZAICodingPlanFailureMetadata(
        kind=kind,
        upstream_id=_safe_upstream_id(upstream_id),
        status_code=_safe_status_code(status_code),
        reset_reason=reset_reason,
    )


def classify_zai_coding_plan_http_failure(
    response: httpx.Response,
    *,
    upstream_id: object = None,
) -> ZAICodingPlanFailureMetadata:
    """Classify one HTTP response while discarding its untrusted payload."""

    error_code, message = _extract_openai_compatible_error(response)
    return classify_zai_coding_plan_failure(
        status_code=response.status_code,
        error_code=error_code,
        message=message,
        upstream_id=upstream_id,
    )


def _safe_status_code(value: object) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _safe_error_code(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _safe_upstream_id(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    model_id = value.removeprefix("zai_coding_plan/").strip()
    return model_id if model_id in ZAI_CODING_PLAN_MODEL_IDS else None


def _extract_openai_compatible_error(response: httpx.Response) -> tuple[object, object]:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return None, None
    if not isinstance(payload, Mapping):
        return None, None
    error = payload.get("error", payload)
    if not isinstance(error, Mapping):
        return None, None
    return error.get("code", error.get("type")), error.get("message")
