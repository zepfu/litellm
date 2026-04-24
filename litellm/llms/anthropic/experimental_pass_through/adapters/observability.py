import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from litellm.utils import supports_reasoning, supports_xhigh_reasoning_effort


_EFFORT_BY_BUDGET = (
    (10000, "high"),
    (5000, "medium"),
    (2000, "low"),
)


@dataclass
class AdapterReasoningEffort:
    requested_value: str
    source: str
    native_provider: str
    native_value: Optional[str]
    native_field: Optional[str]
    clamped_from: Optional[str] = None
    clamp_reason: Optional[str] = None

    def metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "reasoning_effort_requested": self.requested_value,
            "reasoning_effort_source": self.source,
            "reasoning_effort_native_provider": self.native_provider,
        }
        if self.native_value:
            metadata["reasoning_effort_native_value"] = self.native_value
            metadata[f"{self.native_provider}_reasoning_effort"] = self.native_value
        if self.native_field:
            metadata["reasoning_effort_native_field"] = self.native_field
        if self.clamped_from:
            metadata["reasoning_effort_clamped_from"] = self.clamped_from
        if self.clamp_reason:
            metadata["reasoning_effort_clamp_reason"] = self.clamp_reason
        return metadata

    def tags(self) -> list[str]:
        value = self.native_value or self.requested_value
        tags = [f"effort:{value}"]
        if self.native_value:
            tags.append(f"{self.native_provider}-effort:{self.native_value}")
        if self.clamped_from or self.clamp_reason:
            tags.append("reasoning-effort-clamped")
        return tags


def _normalize_effort_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        return normalized
    return None


def _budget_to_effort(budget_tokens: Any) -> str:
    try:
        budget = int(budget_tokens)
    except Exception:
        budget = 0
    for threshold, effort in _EFFORT_BY_BUDGET:
        if budget >= threshold:
            return effort
    return "minimal"


def _extract_requested_effort(
    *,
    thinking: Any = None,
    output_config: Any = None,
    reasoning_effort: Any = None,
) -> tuple[Optional[str], Optional[str]]:
    if isinstance(output_config, dict):
        output_effort = _normalize_effort_value(output_config.get("effort"))
        if output_effort:
            return output_effort, "output_config.effort"

    direct_effort = _normalize_effort_value(reasoning_effort)
    if direct_effort:
        return direct_effort, "reasoning_effort"

    if not isinstance(thinking, dict):
        return None, None

    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        return "none", "thinking.type"
    if thinking_type == "adaptive":
        return "medium", "thinking.type"
    if thinking_type == "enabled":
        return _budget_to_effort(thinking.get("budget_tokens")), "thinking.budget_tokens"
    return None, None


def normalize_reasoning_effort_for_provider(
    *,
    thinking: Any = None,
    output_config: Any = None,
    reasoning_effort: Any = None,
    model: Optional[str],
    custom_llm_provider: Optional[str],
    native_provider: str,
    native_field: str = "reasoning_effort",
    require_capability: bool = False,
) -> Optional[AdapterReasoningEffort]:
    requested, source = _extract_requested_effort(
        thinking=thinking,
        output_config=output_config,
        reasoning_effort=reasoning_effort,
    )
    if not requested or not source:
        return None

    provider = native_provider.strip().lower()
    native_value = requested
    clamped_from: Optional[str] = None
    clamp_reason: Optional[str] = None

    if require_capability and model and not supports_reasoning(
        model=model, custom_llm_provider=custom_llm_provider
    ):
        return AdapterReasoningEffort(
            requested_value=requested,
            source=source,
            native_provider=provider,
            native_value=None,
            native_field=None,
            clamped_from=requested,
            clamp_reason=f"{provider}_reasoning_unsupported",
        )

    if requested in {"max", "xhigh"}:
        if provider == "openai" and model and supports_xhigh_reasoning_effort(
            model=model, custom_llm_provider=custom_llm_provider
        ):
            native_value = "xhigh"
        elif provider == "openrouter" and model and supports_xhigh_reasoning_effort(
            model=model, custom_llm_provider=custom_llm_provider
        ):
            native_value = "xhigh"
        else:
            native_value = "high"
            clamped_from = requested
            clamp_reason = f"{provider}_max_effort_clamped_to_high"
    elif provider in {"gemini", "nvidia"} and requested == "minimal":
        native_value = "low"
        clamped_from = requested
        clamp_reason = f"{provider}_minimal_effort_clamped_to_low"
    elif requested == "none":
        return AdapterReasoningEffort(
            requested_value=requested,
            source=source,
            native_provider=provider,
            native_value=None,
            native_field=None,
        )

    return AdapterReasoningEffort(
        requested_value=requested,
        source=source,
        native_provider=provider,
        native_value=native_value,
        native_field=native_field,
        clamped_from=clamped_from,
        clamp_reason=clamp_reason,
    )


def request_contains_cache_control(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("cache_control") is not None or payload.get("cacheControl") is not None:
            return True
        return any(request_contains_cache_control(value) for value in payload.values())
    if isinstance(payload, list):
        return any(request_contains_cache_control(item) for item in payload)
    return False


def _collect_cache_control_material(payload: Any, collector: list[Any]) -> None:
    if isinstance(payload, dict):
        cache_control = payload.get("cache_control")
        if cache_control is None:
            cache_control = payload.get("cacheControl")
        if cache_control is not None:
            material = dict(payload)
            material.pop("cache_control", None)
            material.pop("cacheControl", None)
            collector.append({"cache_control": cache_control, "content": material})
        for value in payload.values():
            _collect_cache_control_material(value, collector)
    elif isinstance(payload, list):
        for item in payload:
            _collect_cache_control_material(item, collector)


def derive_prompt_cache_key(payload: Any, *, prefix: str = "anthropic-cache") -> Optional[str]:
    cache_material: list[Any] = []
    _collect_cache_control_material(payload, cache_material)
    if not cache_material:
        return None
    encoded = json.dumps(cache_material, sort_keys=True, default=str, separators=(",", ":"))
    cleaned_prefix = prefix.strip()[:23] or "cache"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:40]
    return f"{cleaned_prefix}-{digest}"[:64]


def provider_cache_intent_metadata(
    *,
    provider: str,
    attempted: bool,
    native_supported: bool,
    miss_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if not attempted:
        return {}

    provider_key = provider.strip().lower()
    status = "miss" if not native_supported else None
    reason = miss_reason
    if status == "miss" and reason is None:
        reason = f"{provider_key}_no_native_prompt_cache"

    metadata: Dict[str, Any] = {
        "usage_provider_cache_attempted": True,
        "usage_provider_cache_source": "anthropic_adapter.cache_control",
        f"{provider_key}_provider_cache_attempted": True,
        f"{provider_key}_provider_cache_source": "anthropic_adapter.cache_control",
    }
    if status is not None:
        metadata.update(
            {
                "usage_provider_cache_status": status,
                "usage_provider_cache_miss": True,
                "usage_provider_cache_miss_reason": reason,
                f"{provider_key}_provider_cache_status": status,
                f"{provider_key}_provider_cache_miss": True,
                f"{provider_key}_provider_cache_miss_reason": reason,
            }
        )
    return metadata
