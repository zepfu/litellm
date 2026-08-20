"""Z.AI Coding Plan provider support."""

from .chat.transformation import (
    ZAI_CODING_PLAN_API_BASE,
    ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
    ZAI_CODING_PLAN_MODEL_IDS,
    ZAI_CODING_PLAN_USER_AGENT_PREFIX,
    ZAICodingPlanApiBaseError,
    ZAICodingPlanAuthenticationError,
    ZAICodingPlanChatConfig,
)
from .failure_classification import (
    ZAICodingPlanFailureKind,
    ZAICodingPlanFailureMetadata,
    classify_zai_coding_plan_failure,
    classify_zai_coding_plan_http_failure,
)

__all__ = [
    "ZAI_CODING_PLAN_API_BASE",
    "ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL",
    "ZAI_CODING_PLAN_MODEL_IDS",
    "ZAI_CODING_PLAN_USER_AGENT_PREFIX",
    "ZAICodingPlanApiBaseError",
    "ZAICodingPlanAuthenticationError",
    "ZAICodingPlanChatConfig",
    "ZAICodingPlanFailureKind",
    "ZAICodingPlanFailureMetadata",
    "classify_zai_coding_plan_failure",
    "classify_zai_coding_plan_http_failure",
]
