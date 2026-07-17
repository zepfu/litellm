from typing import TYPE_CHECKING

from .dynamoai import DynamoAIGuardrails

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    """
    Initialize a DynamoAI guardrail and register it as a LiteLLM callback.

    Discovery via guardrail_registry.discover_guardrail_initializers requires either
    this function or guardrail_initializer_registry to be exported from this package.
    """
    import litellm

    _dynamoai_callback = DynamoAIGuardrails(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
        # Extra DynamoAI fields are allowed on LitellmParams (extra="allow")
        model_id=getattr(litellm_params, "model_id", None) or "",
        policy_ids=getattr(litellm_params, "policy_ids", None) or [],
    )
    litellm.logging_callback_manager.add_litellm_callback(_dynamoai_callback)

    return _dynamoai_callback


# String key matches litellm_params.guardrail / directory name used by dynamic discovery.
# Prefer a plain string so registration works even if SupportedGuardrailIntegrations has
# not yet been extended with DYNAMOAI in this fork.
guardrail_initializer_registry = {
    "dynamoai": initialize_guardrail,
}


guardrail_class_registry = {
    "dynamoai": DynamoAIGuardrails,
}


__all__ = [
    "DynamoAIGuardrails",
    "initialize_guardrail",
    "guardrail_initializer_registry",
    "guardrail_class_registry",
]
