from datetime import datetime
from typing import Any, Optional, Union

from typing_extensions import TypedDict


class LangfuseLoggingConfig(TypedDict):
    langfuse_secret: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: Optional[str]


class LangfuseUsageDetails(TypedDict):
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]


class LangfuseCostDetails(TypedDict, total=False):
    total: float


class LangfuseSpanDescriptor(TypedDict, total=False):
    name: str
    input: Any
    output: Any
    metadata: dict[str, Any]
    start_time: Union[str, datetime]
    end_time: Union[str, datetime]


__all__ = [
    "LangfuseLoggingConfig",
    "LangfuseUsageDetails",
    "LangfuseCostDetails",
    "LangfuseSpanDescriptor",
]
