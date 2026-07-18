"""OpenRouter provider error parsing and rate-limit classification."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Protocol

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.types import Payload


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


def extract_exception_status_code(
    runtime: ErrorShapeRuntime,
    exc: object,
) -> Optional[int]:
    _ = runtime
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    if "429" in str(exc):
        return 429
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
