"""Cohere cancellation outcome helpers.

Kept free of provider client imports so HTTP handlers can close a cancelled
upstream response without an import cycle.
"""

import asyncio
from inspect import isawaitable
from typing import Any, Optional
from urllib.parse import urlparse

import anyio

COHERE_CANCELLATION_FAILURE_CLASS = "cancellation"
_COHERE_PROVIDERS = frozenset({"cohere", "cohere_chat"})
_COHERE_API_HOSTS = frozenset({"api.cohere.com", "api.cohere.ai"})
_UPSTREAM_CLOSED_ONCE = "_aawm_upstream_closed_once"


def is_cohere_provider_name(provider: Optional[str]) -> bool:
    return str(provider or "").strip().lower() in _COHERE_PROVIDERS


def is_cohere_api_host(url: Any) -> bool:
    try:
        hostname = str(urlparse(str(url or "")).hostname or "").lower()
    except (TypeError, ValueError):
        return False
    return hostname in _COHERE_API_HOSTS


def cohere_exception_status_code(exc: BaseException) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def is_cohere_request_cancellation(exc: Any) -> bool:
    """Return whether a Cohere call ended by provider 499 or local cancellation."""

    if not isinstance(exc, BaseException):
        return False
    if getattr(exc, "_aawm_cohere_cancellation", False) is True:
        return True
    provider = getattr(exc, "llm_provider", None) or getattr(exc, "custom_llm_provider", None)
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return is_cohere_provider_name(provider)
    if cohere_exception_status_code(exc) != 499:
        return False
    if type(exc).__name__ == "CohereError" or is_cohere_provider_name(provider):
        return True
    response = getattr(exc, "response", None)
    return is_cohere_api_host(getattr(response, "url", None)) or is_cohere_api_host(getattr(exc, "request_url", None))


def mark_cohere_cancellation(
    exc: BaseException,
    *,
    provider_returned: bool,
) -> None:
    setattr(exc, "_aawm_cohere_cancellation", True)
    if provider_returned:
        setattr(exc, "_aawm_provider_returned", True)


def attach_cohere_upstream_response(
    provider: Optional[str],
    stream: Any,
    response: Any,
) -> None:
    if stream is None or response is None or not is_cohere_provider_name(provider):
        return
    setattr(stream, "_aawm_raw_response", response)


def close_upstream_response_once(response: Any) -> None:
    if response is None or getattr(response, _UPSTREAM_CLOSED_ONCE, False) is True:
        return
    close = getattr(response, "close", None)
    if not callable(close):
        return
    setattr(response, _UPSTREAM_CLOSED_ONCE, True)
    try:
        close()
    except OSError:
        return


async def aclose_upstream_response_once(response: Any) -> None:
    if response is None or getattr(response, _UPSTREAM_CLOSED_ONCE, False) is True:
        return
    setattr(response, _UPSTREAM_CLOSED_ONCE, True)
    with anyio.CancelScope(shield=True):
        try:
            aclose = getattr(response, "aclose", None)
            if callable(aclose):
                result = aclose()
                if isawaitable(result):
                    await result
                return
            close = getattr(response, "close", None)
            if not callable(close):
                return
            result = close()
            if isawaitable(result):
                await result
        except OSError:
            return


def close_cohere_cancelled_response(response: Any) -> None:
    if getattr(response, "status_code", None) != 499:
        return
    if not is_cohere_api_host(getattr(response, "url", None)):
        return
    close_upstream_response_once(response)


async def aclose_cohere_cancelled_response(response: Any) -> None:
    if getattr(response, "status_code", None) != 499:
        return
    if not is_cohere_api_host(getattr(response, "url", None)):
        return
    await aclose_upstream_response_once(response)
