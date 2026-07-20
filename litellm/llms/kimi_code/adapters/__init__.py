"""Ingress adapters for the managed Kimi Code chat-completions transport."""

from .adapter import (
    KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL,
    normalize_kimi_code_chat_completions_adapter_model_name,
    prepare_anthropic_kimi_chat_completions_adapter_route,
    prepare_codex_kimi_chat_completions_adapter_route,
)

__all__ = [
    "KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL",
    "normalize_kimi_code_chat_completions_adapter_model_name",
    "prepare_anthropic_kimi_chat_completions_adapter_route",
    "prepare_codex_kimi_chat_completions_adapter_route",
]
