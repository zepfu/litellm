"""AAWM alias-routing package (RR-054 #1/#9/#11/#12/#35).

Layers:
- ``policy``: static candidate tables, allowlists, cooldown defaults
- ``state``: process-local cooldown/affinity/lane/oauth maps + locks
- ``memory`` / ``retry``: shared map bounding and cooldown wait primitives
- ``error_signals``: failure extraction, classification, cooldown scope, retry planning
- ``cooldown_apply``: immutable publication plans and cooldown application
- ``attempt_records``: attempt mutation, evidence, reasoning normalization, metadata
- ``adapter_config``: config descriptors for Anthropic adapter routes
- ``oauth_token_cache`` / ``google_oauth``: Google OAuth token cache + I/O
- ``task_state``: structured/configurable task-state preservation contract
- ``durable``: durable key, max-expiry, read/write, and DualCache selection

The Redis connection manager remains in
``litellm.proxy.aawm_alias_routing_redis``. The ``durable`` module is imported
explicitly rather than re-exported from this package root.
"""

from __future__ import annotations

from . import (
    adapter_config,
    adapter_driver,
    attempt_records,
    audit_build,
    audit_context,
    audit_events,
    audit_persist,
    cooldown_apply,
    cooldown_state,
    error_signals,
    google_oauth,
    memory,
    oauth_token_cache,
    policy,
    provider_shaping,
    responses_finalize,
    retry,
    selection,
    state,
    streaming,
    task_state,
)
from .state import AliasRoutingStateManager, alias_routing_state

__all__ = [
    "adapter_config",
    "adapter_driver",
    "alias_routing_state",
    "AliasRoutingStateManager",
    "attempt_records",
    "audit_build",
    "audit_context",
    "audit_events",
    "audit_persist",
    "cooldown_apply",
    "cooldown_state",
    "error_signals",
    "google_oauth",
    "memory",
    "oauth_token_cache",
    "policy",
    "provider_shaping",
    "responses_finalize",
    "retry",
    "selection",
    "state",
    "streaming",
    "task_state",
]
