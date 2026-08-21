"""Immutable OpenAI passthrough text-watermark configuration (CFG-026)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .text_nodes import _normalize_endpoint

WatermarkMode = Literal["off", "detect", "sanitize", "enforce"]
UnicodePolicyName = Literal["conservative", "aggressive"]
StreamPolicyName = Literal[
    "audit_only",
    "safe_subset",
    "buffer_text_item",
    "buffer_response",
]
UnremovablePolicyName = Literal["allow", "reject"]
WatermarkEndpointName = Literal["responses", "chat_completions"]


class TextWatermarkDirectionsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: bool = True
    response: bool = True


class TextWatermarkUnicodeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    policy: UnicodePolicyName = "conservative"
    normalize_spaces: bool = True
    nfkc: bool = False
    detect_confusables: bool = False


class TextWatermarkRemovalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    stream_policy: StreamPolicyName = "audit_only"
    on_unremovable: UnremovablePolicyName = "allow"

    @field_validator("on_unremovable", mode="before")
    @classmethod
    def _coerce_block_alias(cls, value: Any) -> Any:
        if value == "block":
            return "reject"
        return value


class TextWatermarkLimitsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_text_bytes_per_direction: int = 1_048_576
    max_text_nodes_per_direction: int = 256
    max_reported_paths: int = 32
    max_reported_hits_per_path: int = 16


class StatisticalDetectorSettings(BaseModel):
    """Disabled-by-default statistical detector descriptor.

    Keys and tokenizers are stored for later CFG work. Runtime evaluation
    never loads torch/transformers and never treats an empty registry as
    ``clean``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type: str
    enabled: bool = False
    tokenizer: Optional[str] = None
    key_secret_ref: Optional[str] = None
    threshold: Optional[float] = None
    minimum_tokens: Optional[int] = None


class OpenAIPassthroughTextWatermarkSettings(BaseModel):
    """Shipped default: ``mode=off``, ``removal.enabled=false``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: WatermarkMode = "off"
    directions: TextWatermarkDirectionsSettings = Field(
        default_factory=TextWatermarkDirectionsSettings
    )
    endpoints: tuple[WatermarkEndpointName, ...] = ("responses", "chat_completions")
    unicode: TextWatermarkUnicodeSettings = Field(
        default_factory=TextWatermarkUnicodeSettings
    )
    removal: TextWatermarkRemovalSettings = Field(
        default_factory=TextWatermarkRemovalSettings
    )
    statistical_detectors: tuple[StatisticalDetectorSettings, ...] = ()
    limits: TextWatermarkLimitsSettings = Field(
        default_factory=TextWatermarkLimitsSettings
    )

    def allows_endpoint(self, endpoint: str) -> bool:
        """True when the normalized endpoint is in ``endpoints``."""

        return _normalize_endpoint(endpoint) in self.endpoints

    @model_validator(mode="after")
    def _require_explicit_removal_and_buffer_for_enforce(
        self,
    ) -> "OpenAIPassthroughTextWatermarkSettings":
        if self.mode in {"sanitize", "enforce"} and not self.removal.enabled:
            raise ValueError(
                "mode 'sanitize'/'enforce' requires removal.enabled=true"
            )
        if self.mode == "enforce" and self.removal.stream_policy != "buffer_response":
            raise ValueError(
                "mode 'enforce' requires removal.stream_policy='buffer_response' "
                "for streamed output"
            )
        return self


def _payload_to_mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, OpenAIPassthroughTextWatermarkSettings):
        return payload.model_dump()
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError(
        "openai_passthrough_text_watermark config must be a mapping or settings object"
    )


def load_text_watermark_config(
    payload: Optional[Union[Mapping[str, Any], BaseModel]] = None,
) -> OpenAIPassthroughTextWatermarkSettings:
    """Validate and freeze a watermark policy. ``None`` loads shipped defaults."""

    if isinstance(payload, OpenAIPassthroughTextWatermarkSettings):
        return payload
    return OpenAIPassthroughTextWatermarkSettings.model_validate(
        _payload_to_mapping(payload)
    )


def get_runtime_text_watermark_config() -> OpenAIPassthroughTextWatermarkSettings:
    """Read live proxy settings; missing or invalid config stays ``mode=off``."""

    try:
        from litellm.proxy.proxy_server import general_settings
    except Exception:
        return load_text_watermark_config(None)

    payload: Any = None
    try:
        if isinstance(general_settings, Mapping):
            payload = general_settings.get("openai_passthrough_text_watermark")
        else:
            payload = getattr(
                general_settings, "openai_passthrough_text_watermark", None
            )
        return load_text_watermark_config(payload)
    except Exception:
        return load_text_watermark_config(None)
