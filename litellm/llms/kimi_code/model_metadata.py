"""Conservative parsing and admission helpers for Kimi Code `/models` metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Dict, Final, FrozenSet, Optional, Tuple


K3_MODEL_ID: Final[str] = "k3"
K2_7_MODEL_IDS: Final[FrozenSet[str]] = frozenset({"kimi-for-coding", "kimi-for-coding-highspeed"})
MANAGED_KIMI_CODE_MODEL_IDS: Final[FrozenSet[str]] = frozenset({K3_MODEL_ID, *K2_7_MODEL_IDS})

_EXPLICIT_CAPABILITY_FIELDS: Final[FrozenSet[str]] = frozenset(
    {"supports_reasoning", "supports_image_in", "supports_video_in"}
)


@dataclass(frozen=True)
class KimiCodeThinkEfforts:
    """The explicitly advertised thinking-effort controls for one model."""

    support: bool
    valid_efforts: Tuple[str, ...]
    default_effort: str


@dataclass(frozen=True)
class KimiCodeModelMetadata:
    """The subset of Kimi Code model metadata explicitly present in `/models`."""

    model_id: str
    context_length: Optional[int]
    max_output_tokens: Optional[int]
    supports_reasoning: Optional[bool]
    supports_image_in: Optional[bool]
    supports_video_in: Optional[bool]
    supports_thinking_type: Optional[str]
    think_efforts: Optional[KimiCodeThinkEfforts]


def is_managed_kimi_code_model_id(model_id: object) -> bool:
    """Return whether `model_id` is one of the exact managed Kimi Code IDs."""

    return isinstance(model_id, str) and model_id in MANAGED_KIMI_CODE_MODEL_IDS


def is_k3_model_id(model_id: object) -> bool:
    """Return whether `model_id` is exactly the managed K3 ID."""

    return model_id == K3_MODEL_ID


def is_k2_7_model_id(model_id: object) -> bool:
    """Return whether `model_id` is exactly one of the managed K2.7 IDs."""

    return isinstance(model_id, str) and model_id in K2_7_MODEL_IDS


def parse_kimi_code_models_payload(
    payload: object,
) -> Dict[str, KimiCodeModelMetadata]:
    """
    Parse the managed records from a Kimi Code `/models` payload.

    Fields not observed in the payload contract are intentionally ignored.
    """

    if not isinstance(payload, Mapping):
        return {}

    models = payload.get("data")
    if not isinstance(models, list):
        return {}

    metadata_by_id: Dict[str, KimiCodeModelMetadata] = {}
    for model in models:
        metadata = _parse_managed_model_metadata(model)
        if metadata is not None:
            metadata_by_id[metadata.model_id] = metadata
    return metadata_by_id


def get_kimi_code_model_metadata(payload: object, model_id: object) -> Optional[KimiCodeModelMetadata]:
    """Return metadata for an exact managed ID from one `/models` payload."""

    if not isinstance(model_id, str) or not is_managed_kimi_code_model_id(model_id):
        return None
    return parse_kimi_code_models_payload(payload).get(model_id)


def supports_k3_think_effort(metadata: Optional[KimiCodeModelMetadata], effort: object) -> bool:
    """Return whether an explicitly advertised K3 effort is admitted."""

    if metadata is None or not is_k3_model_id(metadata.model_id):
        return False

    think_efforts = metadata.think_efforts
    return (
        isinstance(effort, str)
        and think_efforts is not None
        and think_efforts.support
        and effort in think_efforts.valid_efforts
    )


def get_k3_default_think_effort(
    metadata: Optional[KimiCodeModelMetadata],
) -> Optional[str]:
    """Return K3's explicitly advertised default effort when it is admissible."""

    if metadata is None or not is_k3_model_id(metadata.model_id):
        return None

    think_efforts = metadata.think_efforts
    if (
        think_efforts is None
        or not think_efforts.support
        or think_efforts.default_effort not in think_efforts.valid_efforts
    ):
        return None
    return think_efforts.default_effort


def is_always_thinking_eligible(
    metadata: Optional[KimiCodeModelMetadata],
) -> bool:
    """Map the explicit `only` thinking marker to always-thinking eligibility."""

    return metadata is not None and metadata.supports_thinking_type == "only"


def supports_explicit_capabilities(
    metadata: Optional[KimiCodeModelMetadata],
    required_capabilities: Iterable[str],
) -> bool:
    """
    Admit a model only when every requested capability is explicitly true.

    Missing, false, malformed, or unknown capability names fail closed.
    """

    if metadata is None:
        return False

    for capability in required_capabilities:
        if capability not in _EXPLICIT_CAPABILITY_FIELDS:
            return False
        if getattr(metadata, capability) is not True:
            return False
    return True


def _parse_managed_model_metadata(
    model: object,
) -> Optional[KimiCodeModelMetadata]:
    if not isinstance(model, Mapping):
        return None

    model_id = model.get("id")
    if not isinstance(model_id, str) or not is_managed_kimi_code_model_id(model_id):
        return None

    return KimiCodeModelMetadata(
        model_id=model_id,
        context_length=_parse_optional_integer(model.get("context_length")),
        max_output_tokens=_parse_optional_integer(model.get("max_output_tokens")),
        supports_reasoning=_parse_optional_boolean(model.get("supports_reasoning")),
        supports_image_in=_parse_optional_boolean(model.get("supports_image_in")),
        supports_video_in=_parse_optional_boolean(model.get("supports_video_in")),
        supports_thinking_type=_parse_optional_string(model.get("supports_thinking_type")),
        think_efforts=_parse_think_efforts(model.get("think_efforts")),
    )


def _parse_optional_integer(value: object) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _parse_optional_boolean(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _parse_optional_string(value: object) -> Optional[str]:
    if isinstance(value, str):
        return value
    return None


def _parse_think_efforts(value: object) -> Optional[KimiCodeThinkEfforts]:
    if not isinstance(value, Mapping):
        return None

    support = value.get("support")
    valid_efforts = value.get("valid_efforts")
    default_effort = value.get("default_effort")
    if (
        not isinstance(support, bool)
        or not isinstance(valid_efforts, list)
        or not all(isinstance(effort, str) for effort in valid_efforts)
        or not isinstance(default_effort, str)
    ):
        return None

    return KimiCodeThinkEfforts(
        support=support,
        valid_efforts=tuple(valid_efforts),
        default_effort=default_effort,
    )
