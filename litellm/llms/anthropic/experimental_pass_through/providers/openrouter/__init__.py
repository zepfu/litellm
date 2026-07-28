"""OpenRouter Anthropic pass-through provider preparation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import retry_transport
    from .adapter import Runtime, prepare_completion_route, prepare_responses_route

__all__ = [
    "Runtime",
    "prepare_completion_route",
    "prepare_responses_route",
    "retry_transport",
]


def __getattr__(name: str) -> Any:
    if name == "retry_transport":
        value = import_module(f"{__name__}.retry_transport")
    elif name in {"Runtime", "prepare_completion_route", "prepare_responses_route"}:
        value = getattr(import_module(f"{__name__}.adapter"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
