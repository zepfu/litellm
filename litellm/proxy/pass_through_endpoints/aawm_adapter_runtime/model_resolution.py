"""Wave 4 extraction: model_resolution pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from fastapi import Request

    # Host-global modules (bound via install())
    _alibaba_token_plan_adapters: Any

    # Host-global constants
    _OPENCODE_ZEN_PROVIDER: str
    _OPENCODE_ZEN_FREE_MODELS: frozenset
    _OPENCODE_GO_PROVIDER: str
    _OPENCODE_GO_FREE_MODELS: frozenset
    _ANTHROPIC_RESPONSES_ADAPTER_ENDPOINTS: frozenset
    _ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS: frozenset
    _ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS: frozenset
    _ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS: frozenset
    _ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS: frozenset

    # Host-global functions
    def _extract_claude_agent_and_tenant_from_request_body(request_body: dict) -> tuple: ...
    def _load_claude_agent_declared_model(agent_name: str) -> Optional[str]: ...
    def _is_openai_responses_endpoint(endpoint: str) -> bool: ...
    def is_oa_xai_model(model: str) -> bool: ...
    def normalize_grok_native_oauth_model(model: Any) -> Optional[str]: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_normalize_anthropic_adapter_model_name",
    "_split_anthropic_adapter_provider_prefix",
    "_get_anthropic_adapter_model_candidates",
    "_has_anthropic_responses_adapter_endpoint",
    "_normalize_anthropic_openai_responses_adapter_model_name",
    "_normalize_anthropic_nvidia_responses_adapter_model_name",
    "_normalize_anthropic_openrouter_adapter_model_name",
    "_get_openrouter_completion_adapter_upstream_model",
    "_normalize_opencode_zen_adapter_model_name",
    "_normalize_opencode_go_adapter_model_name",
    "_normalize_kimi_code_chat_completions_adapter_model_name",
    "_normalize_alibaba_token_plan_adapter_model_name",
    "_normalize_zai_coding_plan_adapter_model_name",
    "_resolve_codex_opencode_zen_adapter_model",
    "_resolve_codex_opencode_go_adapter_model",
    "_resolve_codex_nous_chat_completions_adapter_model",
    "_resolve_codex_kimi_chat_completions_adapter_model",
    "_resolve_codex_alibaba_token_plan_adapter_model",
    "_resolve_codex_zai_coding_plan_adapter_model",
    "_resolve_anthropic_opencode_zen_adapter_model",
    "_resolve_anthropic_kimi_chat_completions_adapter_model",
    "_resolve_anthropic_alibaba_token_plan_adapter_model",
    "_resolve_codex_auto_agent_alias_model",
    "_resolve_anthropic_openai_responses_adapter_model",
    "_resolve_anthropic_xai_oauth_adapter_model",
    "_resolve_anthropic_grok_native_oauth_adapter_model",
    "_resolve_anthropic_openrouter_completion_adapter_model",
    "_resolve_anthropic_nvidia_responses_adapter_model",
    "_resolve_anthropic_openrouter_responses_adapter_model",
)


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _obj = _mod[_name]
        _rebound = FunctionType(
            _obj.__code__,
            host_globals,
            _obj.__name__,
            _obj.__defaults__,
            _obj.__closure__,
        )
        _rebound.__kwdefaults__ = _obj.__kwdefaults__
        _rebound.__annotations__ = _obj.__annotations__
        _rebound.__doc__ = _obj.__doc__
        _rebound.__module__ = _obj.__module__
        _rebound.__qualname__ = _obj.__qualname__
        if _obj.__dict__:
            _rebound.__dict__.update(_obj.__dict__)
        _mod[_name] = _rebound
        host_globals[_name] = _rebound


# ── Extracted functions ─────────────────────────────────────────────

def _normalize_anthropic_adapter_model_name(model: Any) -> Optional[str]:
    if not isinstance(model, str):
        return None
    normalized_model = model.strip()
    return normalized_model or None

def _split_anthropic_adapter_provider_prefix(
    model: Any,
) -> tuple[Optional[str], Optional[str]]:
    normalized_model = _normalize_anthropic_adapter_model_name(model)
    if normalized_model is None:
        return None, None
    if "/" not in normalized_model:
        return None, normalized_model

    prefix, remainder = normalized_model.split("/", 1)
    provider = {
        "chatgpt": "openai",
        "nvidia_nim": "nvidia",
        "opencode": _OPENCODE_ZEN_PROVIDER,  # noqa: F821
        "opencode-zen": _OPENCODE_ZEN_PROVIDER,  # noqa: F821
        "zen": _OPENCODE_ZEN_PROVIDER,  # noqa: F821
        "opencode-go": _OPENCODE_GO_PROVIDER,  # noqa: F821
        "opencode_go": _OPENCODE_GO_PROVIDER,  # noqa: F821
        "nous": "nous",
    }.get(
        prefix,
        prefix
        if prefix
        in (
            "openai",
            "openrouter",
            "nvidia",
            "nous",
            _OPENCODE_ZEN_PROVIDER,  # noqa: F821
            _OPENCODE_GO_PROVIDER,  # noqa: F821
        )
        else None,
    )
    if provider is None:
        return None, normalized_model
    return provider, remainder.strip()

def _get_anthropic_adapter_model_candidates(request_body: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    requested_model = _normalize_anthropic_adapter_model_name(request_body.get("model"))
    if requested_model is not None:
        candidates.append(requested_model)

    agent_name, _tenant = _extract_claude_agent_and_tenant_from_request_body(request_body)
    if not agent_name:
        return candidates

    agent_model = _normalize_anthropic_adapter_model_name(_load_claude_agent_declared_model(agent_name))
    if agent_model is not None:
        candidates.append(agent_model)
    return candidates

def _has_anthropic_responses_adapter_endpoint(endpoint: str) -> bool:
    normalized_endpoint = endpoint.strip()
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = f"/{normalized_endpoint}"
    return normalized_endpoint in _ANTHROPIC_RESPONSES_ADAPTER_ENDPOINTS  # noqa: F821

def _normalize_anthropic_openai_responses_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in (None, "openai") or candidate is None:
        return None
    if candidate in _ANTHROPIC_OPENAI_RESPONSES_ADAPTER_ALLOWED_MODELS:  # noqa: F821
        return candidate
    return None

def _normalize_anthropic_nvidia_responses_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider not in (None, "nvidia") or candidate is None:
        return None

    requested_model = model.strip() if isinstance(model, str) else ""
    has_explicit_nvidia_prefix = requested_model.startswith("nvidia/")
    normalized_candidate = candidate.strip()
    nvidia_model_aliases = {
        "minimax/minimax-m2.7": "minimaxai/minimax-m2.7",
    }
    normalized_candidate = nvidia_model_aliases.get(normalized_candidate, normalized_candidate)
    is_openrouter_namespace_model = requested_model in _ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS  # noqa: F821
    if has_explicit_nvidia_prefix and not is_openrouter_namespace_model:
        return normalized_candidate or None
    if normalized_candidate in _ANTHROPIC_NVIDIA_RESPONSES_ADAPTER_ALLOWED_MODELS:  # noqa: F821
        return normalized_candidate
    return None

def _normalize_anthropic_openrouter_adapter_model_name(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider == "nous":
        return None
    normalized_candidate = (
        candidate if explicit_provider == "openrouter" else _normalize_anthropic_adapter_model_name(model)
    )
    if normalized_candidate is None:
        return None

    openrouter_model_aliases = {
        "free": "openrouter/free",
        "elephant-alpha": "openrouter/elephant-alpha",
        "meta-llama/llama-3.3-70b-instructfree": ("meta-llama/llama-3.3-70b-instruct:free"),
    }
    normalized_candidate = openrouter_model_aliases.get(normalized_candidate, normalized_candidate)
    return normalized_candidate or None

def _get_openrouter_completion_adapter_upstream_model(
    model: Any,
) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider == "openrouter" and candidate is not None:
        candidate = candidate.strip()
        return candidate or None
    return _normalize_anthropic_adapter_model_name(model)

def _normalize_opencode_zen_adapter_model_name(model: Any) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider != _OPENCODE_ZEN_PROVIDER or candidate is None:  # noqa: F821
        return None
    normalized_candidate = candidate.strip()
    if normalized_candidate in _OPENCODE_ZEN_FREE_MODELS:  # noqa: F821
        return normalized_candidate
    return None


def _normalize_opencode_go_adapter_model_name(model: Any) -> Optional[str]:
    explicit_provider, candidate = _split_anthropic_adapter_provider_prefix(model)
    if explicit_provider != _OPENCODE_GO_PROVIDER or candidate is None:  # noqa: F821
        return None
    normalized_candidate = candidate.strip()
    if normalized_candidate in _OPENCODE_GO_FREE_MODELS:  # noqa: F821
        return normalized_candidate
    return None

def _normalize_kimi_code_chat_completions_adapter_model_name(
    model: Any,
) -> Optional[str]:
    # Binding-safe: install() rebinds this function into host_globals, so a
    # module-imported helper name would disappear from the visible namespace.
    # Resolve the policy helper at call time instead (host-owned dependencies
    # stay late-bound through host_globals).
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        normalize_kimi_code_chat_completions_adapter_model_name as _normalize_kimi_code_adapter_model_name,
    )

    return _normalize_kimi_code_adapter_model_name(model)

def _normalize_alibaba_token_plan_adapter_model_name(
    model: Any,
) -> Optional[str]:
    return _alibaba_token_plan_adapters.normalize_alibaba_token_plan_adapter_model_name(  # noqa: F821
        model,
    )


def _normalize_zai_coding_plan_adapter_model_name(
    model: Any,
) -> Optional[str]:
    # Binding-safe: install() rebinds this function into host_globals, so a
    # module-imported helper name would disappear from the visible namespace.
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        normalize_zai_coding_plan_adapter_model_name as _normalize_zai_coding_plan_model_name,
    )

    return _normalize_zai_coding_plan_model_name(model)


def _resolve_codex_opencode_zen_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_opencode_zen_adapter_model_name(request_body.get("model"))


def _resolve_codex_opencode_go_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_opencode_go_adapter_model_name(request_body.get("model"))


def _resolve_codex_nous_chat_completions_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    model = request_body.get("model")
    if not isinstance(model, str):
        return None
    normalized = model.strip()
    nous_provider = globals().get("_NOUS_PROVIDER", "nous")
    prefix = f"{nous_provider}/"
    if not normalized.startswith(prefix):
        return None
    remainder = normalized[len(prefix) :].strip()
    return remainder or None

def _resolve_codex_kimi_chat_completions_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_kimi_code_chat_completions_adapter_model_name(request_body.get("model"))

def _resolve_codex_alibaba_token_plan_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_alibaba_token_plan_adapter_model_name(request_body.get("model"))


def _resolve_codex_zai_coding_plan_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    return _normalize_zai_coding_plan_adapter_model_name(request_body.get("model"))

def _resolve_anthropic_opencode_zen_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_opencode_zen_adapter_model_name(candidate)
        if normalized_model is not None:
            return normalized_model
    return None

def _resolve_anthropic_kimi_chat_completions_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_kimi_code_chat_completions_adapter_model_name(candidate)
        if normalized_model is not None:
            return normalized_model
    return None

def _resolve_anthropic_alibaba_token_plan_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_alibaba_token_plan_adapter_model_name(candidate)
        if normalized_model is not None:
            return normalized_model
    return None


def _resolve_codex_auto_agent_alias_model(
    request_body: dict[str, Any],
    endpoint: str,
    *,
    request: "Request",
) -> Optional[str]:
    if not _is_openai_responses_endpoint(endpoint):
        return None
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
        _lookup_active_snapshot_canonical_alias,
    )

    requested_model = request_body.get("model")
    if (
        isinstance(requested_model, str)
        and requested_model.strip().casefold() == "chatgpt/codex-auto-review"
    ):
        requested_model = "codex-auto-review"

    return _lookup_active_snapshot_canonical_alias(
        requested_model,
        request=request,
    )

def _resolve_anthropic_openai_responses_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_openai_responses_adapter_model_name(candidate)
        if normalized_model is not None:
            return normalized_model
    return None

def _resolve_anthropic_xai_oauth_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        if is_oa_xai_model(candidate):
            return candidate
    return None

def _resolve_anthropic_grok_native_oauth_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = normalize_grok_native_oauth_model(candidate)
        if normalized_model is not None:
            return normalized_model
    return None

def _resolve_anthropic_openrouter_completion_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_openrouter_adapter_model_name(candidate)
        if normalized_model in _ANTHROPIC_OPENROUTER_COMPLETION_ADAPTER_ALLOWED_MODELS:  # noqa: F821
            return normalized_model
    return None

def _resolve_anthropic_nvidia_responses_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_nvidia_responses_adapter_model_name(candidate)
        if normalized_model is not None:
            return normalized_model
    return None

def _resolve_anthropic_openrouter_responses_adapter_model(
    request_body: dict[str, Any],
    endpoint: str,
) -> Optional[str]:
    if not _has_anthropic_responses_adapter_endpoint(endpoint):
        return None
    for candidate in _get_anthropic_adapter_model_candidates(request_body):
        normalized_model = _normalize_anthropic_openrouter_adapter_model_name(candidate)
        if normalized_model in _ANTHROPIC_OPENROUTER_RESPONSES_ADAPTER_ALLOWED_MODELS:  # noqa: F821
            return normalized_model
        explicit_provider, _ = _split_anthropic_adapter_provider_prefix(candidate)
        if explicit_provider == "openrouter" and normalized_model is not None:
            return normalized_model
    return None
