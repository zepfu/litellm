"""Named output-guard policies and request selectors (CFG-025).

Thresholds live in ``output_guards.yaml``. Selectors match resolved provider /
route-family identity on OpenAI passthrough Responses — never a model-name
allowlist. Adding another provider later is another selector row.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

DEFAULT_OUTPUT_GUARDS_YAML = Path(__file__).with_name("output_guards.yaml")

OPENAI_PASSTHROUGH_RESPONSES_INGRESS = "openai_passthrough_responses"

XAI_OUTPUT_GUARD_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {
        "codex_xai_oauth_responses_adapter",
        "codex_auto_agent_xai_oauth_responses",
        "codex_auto_agent_grok_native_responses",
    }
)


@dataclass(frozen=True)
class OutputGuardPolicy:
    """Numeric thresholds for one named visible-text repetition policy."""

    name: str
    inspect: str = "visible_output_text"
    window_words: int = 400
    min_words: int = 80
    min_ngram: int = 8
    min_repeats: int = 6
    min_coverage: float = 0.72
    growth_words: int = 120
    max_novelty: float = 0.18
    max_ngram_edit_distance: int = 2


@dataclass(frozen=True)
class OutputGuardSelector:
    ingress: str
    provider: str
    policy: str


@dataclass(frozen=True)
class OutputGuardConfig:
    policies: Mapping[str, OutputGuardPolicy]
    selectors: tuple[OutputGuardSelector, ...]


@dataclass(frozen=True)
class OutputGuardRequestContext:
    ingress_path: str
    method: str = "POST"
    custom_llm_provider: Optional[str] = None
    egress_credential_family: Optional[str] = None
    route_family: Optional[str] = None
    resolved_model: Optional[str] = None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_positive_int(name: str, value: Any) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _require_unit_float(name: str, value: Any) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _parse_policy(name: str, payload: Mapping[str, Any]) -> OutputGuardPolicy:
    return OutputGuardPolicy(
        name=name,
        inspect=str(payload.get("inspect") or "visible_output_text"),
        window_words=_require_positive_int(
            "window_words", payload.get("window_words", 400)
        ),
        min_words=_require_positive_int("min_words", payload.get("min_words", 80)),
        min_ngram=_require_positive_int("min_ngram", payload.get("min_ngram", 8)),
        min_repeats=_require_positive_int(
            "min_repeats", payload.get("min_repeats", 6)
        ),
        min_coverage=_require_unit_float(
            "min_coverage", payload.get("min_coverage", 0.72)
        ),
        growth_words=_require_positive_int(
            "growth_words", payload.get("growth_words", 120)
        ),
        max_novelty=_require_unit_float(
            "max_novelty", payload.get("max_novelty", 0.18)
        ),
        max_ngram_edit_distance=int(payload.get("max_ngram_edit_distance", 2)),
    )


def load_output_guard_config(
    path: Optional[Path] = None,
) -> OutputGuardConfig:
    """Load named policies and selectors from YAML."""
    yaml_path = Path(path) if path is not None else DEFAULT_OUTPUT_GUARDS_YAML
    raw_text = yaml_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw_text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("output_guards.yaml root must be a mapping")
    root = loaded.get("output_guards", loaded)
    if not isinstance(root, dict):
        raise ValueError("output_guards document must be a mapping")

    raw_policies = root.get("policies") or {}
    if not isinstance(raw_policies, dict):
        raise ValueError("output_guards.policies must be a mapping")
    policies = {
        name: _parse_policy(name, payload if isinstance(payload, dict) else {})
        for name, payload in raw_policies.items()
    }

    selectors: list[OutputGuardSelector] = []
    for row in root.get("selectors") or []:
        if not isinstance(row, dict):
            continue
        match = row.get("match") or {}
        if not isinstance(match, dict):
            match = {}
        policy_name = str(row.get("policy") or "").strip()
        if not policy_name:
            continue
        selectors.append(
            OutputGuardSelector(
                ingress=str(match.get("ingress") or "").strip(),
                provider=str(match.get("provider") or "").strip().lower(),
                policy=policy_name,
            )
        )
    return OutputGuardConfig(policies=policies, selectors=tuple(selectors))


@lru_cache(maxsize=1)
def default_output_guard_config() -> OutputGuardConfig:
    return load_output_guard_config(DEFAULT_OUTPUT_GUARDS_YAML)


def is_openai_passthrough_responses_ingress(
    ingress_path: str,
    method: str = "POST",
) -> bool:
    """Return whether this request is POST OpenAI-passthrough Responses."""
    if str(method or "POST").upper() not in {"", "POST"}:
        return False
    normalized = str(ingress_path or "").split("?", 1)[0].lower().rstrip("/")
    if "openai_passthrough" not in normalized:
        return False
    if "chat/completions" in normalized:
        return False
    return (
        normalized.endswith("/responses")
        or "/responses/" in normalized
        or normalized.endswith("/openai_passthrough/responses")
    )


def is_resolved_xai_identity(context: OutputGuardRequestContext) -> bool:
    """Resolved xAI identity: provider, credential family, or route family."""
    provider = str(context.custom_llm_provider or "").strip().lower()
    if provider == "xai":
        return True
    credential_family = str(context.egress_credential_family or "").strip().lower()
    if credential_family == "xai":
        return True
    route_family = str(context.route_family or "").strip()
    return route_family in XAI_OUTPUT_GUARD_ROUTE_FAMILIES


def _selector_matches(
    selector: OutputGuardSelector,
    context: OutputGuardRequestContext,
) -> bool:
    if selector.ingress != OPENAI_PASSTHROUGH_RESPONSES_INGRESS:
        return False
    if not is_openai_passthrough_responses_ingress(
        context.ingress_path,
        context.method,
    ):
        return False
    if selector.provider == "xai":
        return is_resolved_xai_identity(context)
    provider = str(context.custom_llm_provider or "").strip().lower()
    return bool(selector.provider) and provider == selector.provider


def resolve_output_guard_policy(
    context: OutputGuardRequestContext,
    *,
    config: Optional[OutputGuardConfig] = None,
) -> Optional[OutputGuardPolicy]:
    """Return the named policy for ``context``, or None when no selector matches."""
    active = config if config is not None else default_output_guard_config()
    for selector in active.selectors:
        if not _selector_matches(selector, context):
            continue
        policy = active.policies.get(selector.policy)
        if policy is not None:
            return policy
    return None


def output_guard_context_from_passthrough(
    *,
    ingress_path: str,
    method: str = "POST",
    custom_llm_provider: Any = None,
    egress_credential_family: Optional[str] = None,
    route_family: Optional[str] = None,
    resolved_model: Any = None,
    request_body: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> OutputGuardRequestContext:
    """Build a selector context from passthrough request fields."""
    provider = custom_llm_provider
    if hasattr(provider, "value"):
        provider = getattr(provider, "value")
    metadata: dict[str, Any] = {}
    body_model = None
    if isinstance(request_body, Mapping):
        body_model = request_body.get("model")
        litellm_metadata = request_body.get("litellm_metadata")
        if isinstance(litellm_metadata, Mapping):
            metadata.update(dict(litellm_metadata))
    if isinstance(extra_metadata, Mapping):
        metadata.update(dict(extra_metadata))
    resolved_route_family = (
        _as_optional_str(route_family)
        or _as_optional_str(metadata.get("passthrough_route_family"))
        or _as_optional_str(metadata.get("openai_passthrough_route_family"))
        or _as_optional_str(metadata.get("route_family"))
        or _as_optional_str(egress_credential_family)
    )
    model = resolved_model if resolved_model is not None else body_model
    return OutputGuardRequestContext(
        ingress_path=str(ingress_path or ""),
        method=str(method or "POST"),
        custom_llm_provider=_as_optional_str(provider),
        egress_credential_family=_as_optional_str(egress_credential_family),
        route_family=resolved_route_family,
        resolved_model=_as_optional_str(model),
    )


__all__ = [
    "DEFAULT_OUTPUT_GUARDS_YAML",
    "OPENAI_PASSTHROUGH_RESPONSES_INGRESS",
    "OutputGuardConfig",
    "OutputGuardPolicy",
    "OutputGuardRequestContext",
    "OutputGuardSelector",
    "XAI_OUTPUT_GUARD_ROUTE_FAMILIES",
    "default_output_guard_config",
    "is_openai_passthrough_responses_ingress",
    "is_resolved_xai_identity",
    "load_output_guard_config",
    "output_guard_context_from_passthrough",
    "resolve_output_guard_policy",
]
