"""AAWM alias-routing package (RR-054 #1/#9/#11/#12/#35).

Layers:
- ``policy``: static candidate tables, allowlists, cooldown defaults
- ``state``: process-local cooldown/affinity/lane/oauth maps + locks
- ``memory`` / ``retry``: shared map bounding and cooldown wait primitives
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
    google_oauth,
    memory,
    oauth_token_cache,
    policy,
    provider_shaping,
    responses_finalize,
    retry,
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
    "google_oauth",
    "memory",
    "oauth_token_cache",
    "policy",
    "provider_shaping",
    "responses_finalize",
    "retry",
    "state",
    "streaming",
    "task_state",
]
