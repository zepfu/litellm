"""OpenRouter provider error parsing and rate-limit classification."""

from __future__ import annotations

import re
from typing import Callable, Iterable, Mapping, Optional, Protocol

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload

_HTTP_STATUS_MIN = 100
_HTTP_STATUS_MAX = 599

# Full-string OpenRouter/HTTP status line. Digits are accepted only when a
# status label or OpenRouter attribution names them as an HTTP status.
_OPENROUTER_STATUS_LINE = re.compile(
    r"""
    \A
    (?:
        (?:openrouter
           (?:[\s_-]+(?:completion|adapter|request|upstream|error|returned|failed))*
           [\s:.\-]+)?
        (?:client|server)\s+error\s+'
        (?P<httpx_status>[1-5]\d{2})
        \s+[A-Za-z][\w\- ]*'
        \s+for\s+url\s+\S+
      |
        (?:openrouter
           (?:[\s_-]+(?:completion|adapter|request|upstream|error|returned|failed))*
           [\s:.\-]+)?
        (?:
            HTTP(?:/\d+\.\d+)?[\s/]+
          | status(?:[\s_-]*code)?[\s:=]+
          | error[\s_-]*code[\s:=]+
        )
        (?P<labeled_status>[1-5]\d{2})
        (?:\s+[A-Za-z][\w\- ]*)?
      |
        openrouter
        (?:[\s_-]+(?:completion|adapter|request|upstream|error|returned|failed))*
        [\s:.\-]+
        (?P<attributed_status>[1-5]\d{2})
        (?:\s+[A-Za-z][\w\- ]*)?
    )
    \Z
    """,
    re.IGNORECASE | re.VERBOSE,
)


class ErrorShapeRuntime(Protocol):
    """Callbacks required to normalize OpenRouter provider errors."""

    @property
    def extract_embedded_json_payload_candidates(
        self,
    ) -> Callable[[object], Iterable[str]]: ...

    @property
    def parse_json_payloads_from_text_candidates(
        self,
    ) -> Callable[[Iterable[str]], Iterable[object]]: ...

    @property
    def extract_upstream_headers(
        self,
    ) -> Callable[[object], Mapping[str, object]]: ...

    @property
    def parse_retry_after_seconds_from_headers(
        self,
    ) -> Callable[[Mapping[str, object]], Optional[float]]: ...

    @property
    def get_header_value(
        self,
    ) -> Callable[[Mapping[str, object], str], Optional[str]]: ...

    @property
    def parse_reset_wait_seconds_from_headers(
        self,
    ) -> Callable[[Mapping[str, object]], Optional[float]]: ...


def _mapping(value: object) -> Optional[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return None
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
    }


def _payload(value: object) -> Optional[Payload]:
    if not isinstance(value, dict):
        return None
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
    }


def _error_metadata(payload: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    error = _mapping(payload.get("error"))
    if error is None:
        return None
    return _mapping(error.get("metadata"))


def _coerce_non_negative_float(value: object) -> Optional[float]:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _http_status_code(value: object) -> Optional[int]:
    """Return a proven HTTP status. Reject bools and unlabeled numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        status = value
    elif isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[1-5]\d{2}", stripped) is None:
            return None
        status = int(stripped)
    else:
        return None
    if _HTTP_STATUS_MIN <= status <= _HTTP_STATUS_MAX:
        return status
    return None


def _status_from_structured_fields(exc: object) -> Optional[int]:
    sources = (exc, getattr(exc, "response", None))
    for source in sources:
        if source is None:
            continue
        for attr in ("status_code", "code"):
            status = _http_status_code(getattr(source, attr, None))
            if status is not None:
                return status
    return None


def _status_from_payload(payload: Mapping[str, object]) -> Optional[int]:
    candidates: list[object] = []
    error = _mapping(payload.get("error"))
    if error is not None:
        candidates.extend(
            (error.get("status_code"), error.get("status"), error.get("code"))
        )
        metadata = _mapping(error.get("metadata"))
        if metadata is not None:
            candidates.extend(
                (metadata.get("status_code"), metadata.get("status"))
            )
    candidates.extend(
        (payload.get("status_code"), payload.get("status"), payload.get("code"))
    )
    for value in candidates:
        status = _http_status_code(value)
        if status is not None:
            return status
    return None


def _status_from_openrouter_status_line(text: object) -> Optional[int]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    first_line = stripped.splitlines()[0]
    if first_line not in candidates:
        candidates.append(first_line)
    for candidate in candidates:
        compact = " ".join(candidate.split())
        if not compact:
            continue
        match = _OPENROUTER_STATUS_LINE.fullmatch(compact)
        if match is None:
            continue
        groups = match.groupdict()
        for key in ("httpx_status", "labeled_status", "attributed_status"):
            raw = groups.get(key)
            if raw:
                return int(raw)
    return None


def extract_exception_status_code(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[int]:
    """Parse an HTTP status from structured OpenRouter error fields.

    Text is accepted only through a bounded OpenRouter-attributed status-line
    grammar. Bare digits in model names, request IDs, or messages are not
    HTTP statuses.
    """
    structured = _status_from_structured_fields(exc)
    if structured is not None:
        return structured
    payload = extract_error_payload(runtime, exc)
    if payload is not None:
        payload_status = _status_from_payload(payload)
        if payload_status is not None:
            return payload_status
    for candidate in (
        str(exc),
        getattr(exc, "message", None),
        getattr(exc, "detail", None),
    ):
        line_status = _status_from_openrouter_status_line(candidate)
        if line_status is not None:
            return line_status
    return None


def extract_error_payload(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[Payload]:
    candidates = [
        getattr(exc, "detail", None),
        getattr(exc, "message", None),
        str(exc),
    ]
    for candidate in candidates:
        payload = _payload(candidate)
        if payload is not None:
            return payload
        embedded = runtime.extract_embedded_json_payload_candidates(candidate)
        for parsed in runtime.parse_json_payloads_from_text_candidates(embedded):
            payload = _payload(parsed)
            if payload is not None:
                return payload
            if isinstance(parsed, list):
                for item in parsed:
                    payload = _payload(item)
                    if payload is not None:
                        return payload
    return None


def extract_provider_name(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[str]:
    payload = extract_error_payload(runtime, exc)
    if payload is None:
        return None
    metadata = _error_metadata(payload)
    if metadata is None:
        return None
    provider_name = metadata.get("provider_name")
    if isinstance(provider_name, str) and provider_name:
        return provider_name
    return None


def extract_retry_after_seconds(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[float]:
    payload = extract_error_payload(runtime, exc)
    if payload is not None:
        metadata = _error_metadata(payload)
        if metadata is not None:
            retry_after = _coerce_non_negative_float(
                metadata.get("retry_after_seconds")
            )
            if retry_after is not None:
                return retry_after
    return runtime.parse_retry_after_seconds_from_headers(
        extract_error_headers(runtime, exc)
    )


def extract_raw_message(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[str]:
    payload = extract_error_payload(runtime, exc)
    if payload is None:
        return None
    metadata = _error_metadata(payload)
    if metadata is not None:
        raw_message = metadata.get("raw")
        if isinstance(raw_message, str) and raw_message:
            return raw_message
    error = _mapping(payload.get("error"))
    if error is None:
        return None
    error_message = error.get("message")
    if isinstance(error_message, str) and error_message:
        return error_message
    return None


def is_no_endpoint_candidate_error(
    runtime: ErrorShapeRuntime,
    exc: object,
    *,
    status_code: Optional[int] = None,
    raw_message: Optional[str] = None,
) -> bool:
    """Return whether OpenRouter reports no upstream endpoint for a model."""
    if status_code is None:
        status_code = extract_exception_status_code(runtime, exc)
    if status_code != 404:
        return False
    if raw_message is None:
        raw_message = extract_raw_message(runtime, exc)
    haystacks: list[str] = []
    if isinstance(raw_message, str) and raw_message:
        haystacks.append(raw_message)
    haystacks.append(str(exc))
    message_attr = getattr(exc, "message", None)
    if isinstance(message_attr, str) and message_attr:
        haystacks.append(message_attr)
    return "no endpoints found" in " ".join(haystacks).lower()


def is_retired_ox_alpha_candidate_error(
    runtime: ErrorShapeRuntime,
    exc: object,
    *,
    model: Optional[str],
    status_code: Optional[int] = None,
    raw_message: Optional[str] = None,
) -> bool:
    """Return whether OpenRouter reports the retired ox-alpha test model."""
    if status_code is None:
        status_code = extract_exception_status_code(runtime, exc)
    if status_code != 404:
        return False
    if model not in {"stealth/ox-alpha", "openrouter/stealth/ox-alpha"}:
        return False
    if raw_message is None:
        raw_message = extract_raw_message(runtime, exc)
    combined_text = " ".join(
        str(part)
        for part in (
            raw_message,
            getattr(exc, "message", None),
            getattr(exc, "detail", None),
            str(exc),
        )
        if part is not None
    ).casefold()
    normalized_text = " ".join(combined_text.split())
    observed_withdrawal_message = (
        "thank you for participating in the stealth ox alpha testing period. "
        "this model was zai's glm-5.3 flash."
    )
    return observed_withdrawal_message in normalized_text


def is_provider_raw_error(runtime: ErrorShapeRuntime, exc: object) -> bool:
    payload = extract_error_payload(runtime, exc)
    if payload is None:
        return False
    error = _mapping(payload.get("error"))
    if error is None:
        return False
    metadata = _mapping(error.get("metadata"))
    if metadata is None:
        return False
    raw_message = metadata.get("raw")
    provider_name = metadata.get("provider_name")
    error_message = error.get("message")
    return (
        isinstance(provider_name, str)
        and bool(provider_name.strip())
        and isinstance(raw_message, str)
        and raw_message.strip().upper() == "ERROR"
        and isinstance(error_message, str)
        and "provider" in error_message.lower()
    )


def extract_error_headers(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Payload:
    merged_headers = dict(runtime.extract_upstream_headers(exc))
    payload = extract_error_payload(runtime, exc)
    if payload is None:
        return merged_headers
    metadata = _error_metadata(payload)
    if metadata is None:
        return merged_headers
    headers = _mapping(metadata.get("headers"))
    if headers is not None:
        merged_headers.update(headers)
    return merged_headers


def get_header_value(
    runtime: ErrorShapeRuntime,
    headers: Mapping[str, object],
    header_name: str,
) -> Optional[str]:
    return runtime.get_header_value(headers, header_name)


def extract_reset_wait_seconds(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[float]:
    headers = extract_error_headers(runtime, exc)
    return runtime.parse_reset_wait_seconds_from_headers(headers)


def is_long_window_rate_limit(
    runtime: ErrorShapeRuntime,
    exc: object,
    *,
    hidden_retry_budget_seconds: float,
) -> bool:
    threshold_seconds = max(hidden_retry_budget_seconds, 30.0)
    retry_after_seconds = extract_retry_after_seconds(runtime, exc)
    if retry_after_seconds is not None:
        return retry_after_seconds > threshold_seconds
    headers = extract_error_headers(runtime, exc)
    remaining_value = get_header_value(runtime, headers, "X-RateLimit-Remaining")
    if remaining_value not in {"0", "0.0"}:
        return False
    reset_wait_seconds = extract_reset_wait_seconds(runtime, exc)
    if reset_wait_seconds is None:
        return False
    return reset_wait_seconds > threshold_seconds


__all__ = [
    "ErrorShapeRuntime",
    "extract_error_headers",
    "extract_error_payload",
    "extract_exception_status_code",
    "extract_provider_name",
    "extract_raw_message",
    "extract_reset_wait_seconds",
    "extract_retry_after_seconds",
    "get_header_value",
    "is_long_window_rate_limit",
    "is_no_endpoint_candidate_error",
    "is_provider_raw_error",
]
