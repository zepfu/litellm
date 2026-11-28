"""Client-safe terminal error shapes for managed Codex Responses traffic."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

from fastapi.responses import JSONResponse

_SAFE_ACCOUNT_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_CODEX_QUOTA_CODES = frozenset({"usage_limit_reached", "insufficient_quota", "quota_exceeded"})
_CODEX_AUTH_CODES = frozenset(
    {
        "token_invalidated",
        "token_revoked",
        "authentication_error",
        "authorization_error",
        "credential_rejected",
    }
)
_CODEX_THROTTLE_CODES = frozenset({"rate_limit_error", "rate_limit_exceeded", "rate_limited", "too_many_requests"})


def is_chatgpt_codex_responses_upstream(url: Any) -> bool:
    try:
        parsed = urlparse(str(url))
    except (TypeError, ValueError):
        return False
    return (
        str(parsed.hostname or "").lower() == "chatgpt.com"
        and "/backend-api/codex/" in str(parsed.path or "").lower()
        and str(parsed.path or "").lower().rstrip("/").endswith("/responses")
    )


def _future_epoch(value: Any, *, now_epoch: float) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) <= now_epoch:
            return None
        return int(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        return _future_epoch(numeric, now_epoch=now_epoch)
    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")  # noqa: FURB162 - Python 3.9 compatibility
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)  # noqa: UP017 - Python 3.9 compatibility
    timestamp = parsed.timestamp()
    if not math.isfinite(timestamp) or timestamp <= now_epoch:
        return None
    return int(timestamp)


def _future_epoch_from_seconds(value: Any, *, now_epoch: float) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return int(now_epoch + seconds)


def codex_verified_reset_at(
    *sources: Any,
    now_epoch: float | None = None,
) -> int | None:
    """Return the earliest explicit future reset found in trusted error data."""
    now = time.time() if now_epoch is None else float(now_epoch)
    reset_values: list[int] = []
    absolute_keys = ("resets_at", "reset_at", "next_reset_at", "provider_resets_at")
    relative_keys = (
        "resets_in_seconds",
        "reset_hint_seconds",
        "provider_resets_in_seconds",
    )
    pending = list(sources)
    seen: set[int] = set()
    while pending:
        source = pending.pop()
        if not isinstance(source, Mapping) or id(source) in seen:
            continue
        seen.add(id(source))
        for key in absolute_keys:
            value = source.get(key)
            reset_at = _future_epoch(value, now_epoch=now)
            if reset_at is not None:
                reset_values.append(reset_at)
        for key in relative_keys:
            reset_at = _future_epoch_from_seconds(source.get(key), now_epoch=now)
            if reset_at is not None:
                reset_values.append(reset_at)
        for key in ("error", "quota", "terminal_reset", "quota_windows", "exhausted_windows"):
            value = source.get(key)
            if isinstance(value, Mapping):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(value[:16])
    return min(reset_values) if reset_values else None


def _format_reset(reset_at: int | None) -> str | None:
    if reset_at is None:
        return None
    return datetime.fromtimestamp(
        reset_at,
        timezone.utc,  # noqa: UP017 - Python 3.9 compatibility
    ).strftime("%Y-%m-%d %H:%M UTC")


def codex_usage_limit_error(
    *reset_sources: Any,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Build the stock Codex usage-limit envelope without provider text."""
    reset_at = codex_verified_reset_at(*reset_sources, now_epoch=now_epoch)
    message = "The managed Codex account usage limit has been reached."
    if reset_at is not None:
        message += f" A quota window resets at {_format_reset(reset_at)}."
    payload: dict[str, Any] = {
        "message": message,
        "type": "usage_limit_reached",
        "code": "usage_limit_reached",
    }
    if reset_at is not None:
        payload["resets_at"] = reset_at
    return payload


def codex_sse_usage_limit_error(
    error: Mapping[str, Any],
    *,
    now_epoch: float | None = None,
) -> dict[str, Any] | None:
    """Sanitize an explicit Codex SSE quota event for stock-client retry handling."""
    code = str(error.get("code") or "").strip().lower()
    error_type = str(error.get("type") or "").strip().lower()
    if code not in _CODEX_QUOTA_CODES and error_type not in _CODEX_QUOTA_CODES:
        return None
    sanitized = codex_usage_limit_error(error, now_epoch=now_epoch)
    sanitized["type"] = "insufficient_quota"
    sanitized["code"] = "insufficient_quota"
    return sanitized


def _safe_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    return label if _SAFE_ACCOUNT_LABEL.fullmatch(label) else None


def _first_present(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def codex_account_failure(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project one managed Codex attempt onto a safe account label and cause."""
    provider = str(_first_present(record, ("provider", "selected_provider")) or "").strip().lower()
    route = str(_first_present(record, ("route_family", "selected_route_family")) or "").strip().lower()
    identity_marker = (
        _first_present(
            record,
            (
                "account_label",
                "codex_oauth_account_label",
                "codex_auto_agent_selected_account_label",
                "aawm_selected_account_label",
            ),
        )
        is not None
        or record.get("aawm_selected_account_provider") == "codex_oauth"
    )
    if (
        provider not in {"openai", "codex", "codex_oauth"}
        or route
        not in {
            "",
            "codex_responses",
        }
        or not identity_marker
    ):
        return None

    raw_class = (
        str(
            _first_present(
                record,
                ("error_class", "failure_class", "error_type", "error_code", "outcome"),
            )
            or ""
        )
        .strip()
        .lower()
    )
    reason = (
        str(
            _first_present(
                record,
                ("reason", "skip_reason", "admission_decision", "terminal_reason"),
            )
            or ""
        )
        .strip()
        .lower()
    )
    status = _first_present(record, ("upstream_status_code", "error_status_code", "status_code"))
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None

    labels = (
        "sign-in required"
        if raw_class in _CODEX_AUTH_CODES
        or reason in _CODEX_AUTH_CODES
        or (status_code == 401 and raw_class in {"provider_terminal_error", ""})
        else "credential unavailable"
        if raw_class in {"credential_unavailable", "auth_degraded"}
        or reason in {"credential_unavailable", "auth_degraded"}
        else "quota exhausted"
        if raw_class in _CODEX_QUOTA_CODES or reason in {"quota_exhausted", "confirmed_exhausted"}
        else "temporarily throttled"
        if raw_class in _CODEX_THROTTLE_CODES or reason in {"rate_limited", "throttled"}
        else "account unavailable"
        if reason in {"cooldown", "candidate_unavailable", "account_unavailable"}
        else None
    )
    if labels is None:
        return None
    failure: dict[str, Any] = {"cause": labels}
    safe_label = _safe_label(
        _first_present(
            record,
            (
                "account_label",
                "codex_oauth_account_label",
                "codex_auto_agent_selected_account_label",
                "aawm_selected_account_label",
            ),
        )
    )
    if safe_label is not None:
        failure["label"] = safe_label
    reset_at = codex_verified_reset_at(record)
    if reset_at is not None:
        failure["resets_at"] = reset_at
    return failure


def is_managed_codex_oauth_account_record(record: Mapping[str, Any]) -> bool:
    """Return whether a record identifies an account on the Codex OAuth route."""
    provider = str(_first_present(record, ("provider", "selected_provider")) or "").strip().lower()
    route = str(_first_present(record, ("route_family", "selected_route_family")) or "").strip().lower()
    identity_marker = (
        _first_present(
            record,
            (
                "account_label",
                "codex_oauth_account_label",
                "codex_auto_agent_selected_account_label",
                "aawm_selected_account_label",
            ),
        )
        is not None
        or record.get("aawm_selected_account_provider") == "codex_oauth"
    )
    return provider in {"openai", "codex", "codex_oauth"} and route == "codex_responses" and identity_marker


def collect_codex_account_failures(
    *sources: Any,
) -> list[dict[str, Any]]:
    """Collect per-account causes while keeping only allowlisted public fields."""
    failures: dict[str, dict[str, Any]] = {}
    unlabeled: list[dict[str, Any]] = []
    for source in sources:
        rows = source if isinstance(source, list) else [source]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            failure = codex_account_failure(row)
            if failure is None:
                continue
            label = failure.get("label")
            if isinstance(label, str):
                failures[label] = failure
            else:
                unlabeled.append(failure)
    return [*failures.values(), *unlabeled]


def _exception_status(exc: Any) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(exc, "code", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _exception_detail(exc: Any) -> Mapping[str, Any]:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, Mapping):
        return detail
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", errors="replace")
    if isinstance(detail, str):
        try:
            parsed = json.loads(detail)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _error_mapping(detail: Mapping[str, Any]) -> Mapping[str, Any]:
    error = detail.get("error")
    return error if isinstance(error, Mapping) else {}


def _is_confirmed_quota(detail: Mapping[str, Any], failures: list[dict[str, Any]]) -> bool:
    error = _error_mapping(detail)
    code = str(error.get("code") or detail.get("failover_disposition") or "").lower()
    error_type = str(error.get("type") or "").lower()
    return (
        code in _CODEX_QUOTA_CODES
        or error_type in _CODEX_QUOTA_CODES
        or detail.get("failover_disposition") == "usage_limit_reached"
        or bool(failures)
        and all(item.get("cause") == "quota exhausted" for item in failures)
    )


def _apply_additional_terminal_reset(
    failures: list[dict[str, Any]],
    terminal_detail: Mapping[str, Any],
    *,
    now_epoch: float,
) -> None:
    reset_at = codex_verified_reset_at(terminal_detail, now_epoch=now_epoch)
    if reset_at is None:
        return
    for failure in failures:
        if (
            failure.get("cause") == "quota exhausted"
            and not isinstance(failure.get("resets_at"), int)
        ):
            failure["resets_at"] = reset_at


def _retry_after_seconds(exc: Any, *, now_epoch: float) -> int | None:
    response = getattr(exc, "response", None)
    for headers in (
        getattr(exc, "upstream_headers", None),
        getattr(exc, "headers", None),
        getattr(response, "headers", None),
    ):
        if not isinstance(headers, Mapping):
            continue
        raw = next(
            (
                value
                for key, value in headers.items()
                if str(key).lower() == "retry-after"
            ),
            None,
        )
        if raw is None:
            continue
        try:
            seconds = float(str(raw).strip())
        except (TypeError, ValueError):
            try:
                reset = parsedate_to_datetime(str(raw))
            except (TypeError, ValueError, OverflowError):
                continue
            if reset.tzinfo is None:
                reset = reset.replace(tzinfo=timezone.utc)  # noqa: UP017 - Python 3.9 compatibility
            seconds = reset.timestamp() - now_epoch
        if math.isfinite(seconds) and seconds > 0:
            return max(1, int(seconds))
    return None


def codex_terminal_json_response(
    exc: Any,
    *,
    account_failure_records: Iterable[Any] = (),
    codex_quota_route_confirmed: bool = False,
    codex_terminal_route_confirmed: bool = False,
    additional_terminal_detail: Mapping[str, Any] | None = None,
    now_epoch: float | None = None,
) -> JSONResponse | None:
    """Shape only recognized managed Codex terminal failures for stock Codex."""
    now = time.time() if now_epoch is None else float(now_epoch)
    status_code = _exception_status(exc)
    detail = _exception_detail(exc)
    terminal_detail = (
        additional_terminal_detail
        if isinstance(additional_terminal_detail, Mapping)
        else {}
    )
    account_records = list(account_failure_records)
    failures = collect_codex_account_failures(
        account_records,
        detail.get("terminal_account_failures"),
        detail.get("skipped_candidates"),
        detail.get("candidates"),
        detail.get("attempts"),
        terminal_detail.get("terminal_account_failures"),
        terminal_detail.get("skipped_candidates"),
        terminal_detail.get("candidates"),
        terminal_detail.get("attempts"),
    )
    _apply_additional_terminal_reset(failures, terminal_detail, now_epoch=now)
    causes = {str(item.get("cause")) for item in failures}
    has_auth_failure = bool(causes & {"sign-in required", "credential unavailable", "account unavailable"})
    is_mixed = has_auth_failure and len(causes) > 1

    if (
        (is_mixed or has_auth_failure)
        and codex_terminal_route_confirmed
        and status_code in {401, 403, 429, 503}
    ):
        message_parts = []
        for failure in failures:
            label = failure.get("label")
            prefix = f"{label}: " if isinstance(label, str) else "Managed account: "
            cause = str(failure.get("cause") or "account unavailable")
            reset_at = failure.get("resets_at")
            reset_text = _format_reset(reset_at if isinstance(reset_at, int) else None)
            if cause == "sign-in required":
                cause_text = "sign-in is required; sign in again or refresh this account's OAuth credentials"
            elif cause == "credential unavailable":
                cause_text = "OAuth credentials are unavailable; sign in again"
            elif cause == "quota exhausted":
                cause_text = "usage limit reached" + (f" until its quota reset at {reset_text}" if reset_text else "")
            elif cause == "temporarily throttled":
                cause_text = "temporarily throttled"
            else:
                cause_text = "currently unavailable"
            message_parts.append(prefix + cause_text)
        if not message_parts:
            message_parts.append(
                "The managed Codex OAuth account is unavailable. Sign in again or refresh its credentials, then retry."
            )
        error: dict[str, Any] = {
            "message": "Managed Codex OAuth could not serve this request: " + "; ".join(message_parts) + ".",
            "type": "service_unavailable",
            "code": "aawm_codex_oauth_account_unavailable",
            "accounts": failures,
        }
        return JSONResponse(status_code=503, content={"error": error})

    if status_code == 429 and codex_quota_route_confirmed and _is_confirmed_quota(detail, failures):
        error = codex_usage_limit_error(
            detail,
            terminal_detail,
            *failures,
            *[item for item in account_records if isinstance(item, Mapping)],
            now_epoch=now,
        )
        headers: dict[str, str] = {}
        retry_after = _retry_after_seconds(exc, now_epoch=now)
        reset_at = error.get("resets_at")
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        elif isinstance(reset_at, int):
            headers["Retry-After"] = str(max(1, reset_at - int(now)))
        return JSONResponse(
            status_code=429,
            content={"error": error},
            headers=headers or None,
        )

    error = _error_mapping(detail)
    error_code = str(error.get("code") or "").strip().lower()
    error_type = str(error.get("type") or "").strip().lower()
    if (
        status_code == 429
        and codex_terminal_route_confirmed
        and (error_code in _CODEX_THROTTLE_CODES or error_type in _CODEX_THROTTLE_CODES)
    ):
        retry_after = _retry_after_seconds(exc, now_epoch=now)
        if retry_after is None:
            return None
        response_error = {
            "message": "The Codex service is temporarily throttling requests.",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
        return JSONResponse(
            status_code=429,
            content={"error": response_error},
            headers={"Retry-After": str(retry_after)},
        )
    return None
