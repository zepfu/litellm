"""Chat-completions support for Z.AI Coding Plan."""

from .transformation import (
    ZAI_CODING_PLAN_API_BASE,
    ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
    ZAI_CODING_PLAN_MODEL_IDS,
    ZAI_CODING_PLAN_USER_AGENT_PREFIX,
    ZAICodingPlanApiBaseError,
    ZAICodingPlanAuthenticationError,
    ZAICodingPlanChatConfig,
)

__all__ = [
    "ZAI_CODING_PLAN_API_BASE",
    "ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL",
    "ZAI_CODING_PLAN_MODEL_IDS",
    "ZAI_CODING_PLAN_USER_AGENT_PREFIX",
    "ZAICodingPlanApiBaseError",
    "ZAICodingPlanAuthenticationError",
    "ZAICodingPlanChatConfig",
]
