"""Runtime memory ownership for AAWM alias routing (Wave 7 / D1-591).

Extracts the five god-module functions that manage process-local mutable
memory maps (log suppression, cooldown hydration, affinity hydration,
in-place body replacement, and FIFO bounding).  All mutable state is
injected explicitly via :class:`RuntimeMemoryRuntime` so this module
never imports the pass-through god module.

Behavior-preserving relocation only; no logic changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

from . import memory as _memory
from .types import Payload

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_SUPPRESSION_WINDOW_SECONDS: float = 30.0

# ---------------------------------------------------------------------------
# Injected runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeMemoryRuntime:
    """Dependencies for runtime-memory functions.

    Attributes:
        log_until_map: Mutable mapping of log-key -> monotonic-until timestamp
            used by the log-suppression gate.  Injected so the owner does not
            create or own the process-global dict.
        max_size: FIFO bound applied to all memory maps.
    """

    log_until_map: MutableMapping[str, float]
    max_size: int = _memory.DEFAULT_MEMORY_STATE_MAX_SIZE


_runtime: Optional[RuntimeMemoryRuntime] = None


def configure_runtime_memory(*, runtime: RuntimeMemoryRuntime) -> None:
    """Bind the runtime-memory runtime.  Must be called before use."""
    global _runtime
    _runtime = runtime


def _require_runtime() -> RuntimeMemoryRuntime:
    if _runtime is None:
        raise RuntimeError(
            "runtime_memory runtime not configured; "
            "call configure_runtime_memory() first"
        )
    return _runtime


# ---------------------------------------------------------------------------
# Owned functions
# ---------------------------------------------------------------------------


def should_log_aawm_alias_routing_event(
    log_key: str,
    *,
    suppression_window: float = DEFAULT_LOG_SUPPRESSION_WINDOW_SECONDS,
) -> bool:
    """Return True and refresh the suppression window if the key is not suppressed.

    Exact semantics of the god-module ``_should_log_aawm_alias_routing_event``:
    - If ``now < until`` for the key, return False (still suppressed).
    - Otherwise set ``until = now + suppression_window``, bound the map, and
      return True.
    """
    runtime = _require_runtime()
    now = time.monotonic()
    until = runtime.log_until_map.get(log_key, 0.0)
    if now < until:
        return False
    runtime.log_until_map[log_key] = now + suppression_window
    _memory.bound_memory_map(runtime.log_until_map, max_size=runtime.max_size)
    return True


def replace_request_body_in_place(
    request_body: Payload,
    updated_body: Payload,
) -> None:
    """Replace *request_body* contents with *updated_body* in place.

    No-op when both references are the same object.
    """
    if updated_body is request_body:
        return
    request_body.clear()
    request_body.update(updated_body)


def bound_aawm_alias_routing_memory_map(
    cache: MutableMapping[Any, Any],
    *,
    max_size: Optional[int] = None,
) -> None:
    """FIFO-trim *cache* to *max_size* entries.

    When *max_size* is ``None`` the configured runtime's ``max_size`` is used.
    """
    if max_size is None:
        max_size = _require_runtime().max_size
    _memory.bound_memory_map(cache, max_size=max_size)


def hydrate_aawm_alias_routing_cooldown_memory(
    *,
    memory_map: MutableMapping[str, float],
    cooldown_key: str,
    expires_at_epoch: float,
    max_size: Optional[int] = None,
) -> None:
    """Hydrate a cooldown entry from an epoch expiry into monotonic space.

    Delegates to :func:`memory.hydrate_cooldown_memory` with the configured
    (or explicit) *max_size*.
    """
    if max_size is None:
        max_size = _require_runtime().max_size
    _memory.hydrate_cooldown_memory(
        memory_map=memory_map,
        cooldown_key=cooldown_key,
        expires_at_epoch=expires_at_epoch,
        max_size=max_size,
    )


def hydrate_aawm_alias_routing_affinity_memory(
    *,
    memory_map: MutableMapping[str, Payload],
    session_key: str,
    payload: Payload,
    expires_at_epoch: float,
    max_size: Optional[int] = None,
) -> Payload:
    """Hydrate an affinity entry from an epoch expiry into monotonic space.

    Delegates to :func:`memory.hydrate_affinity_memory` with the configured
    (or explicit) *max_size*.  Returns the effective affinity dict.
    """
    if max_size is None:
        max_size = _require_runtime().max_size
    return _memory.hydrate_affinity_memory(
        memory_map=memory_map,
        session_key=session_key,
        payload=payload,
        expires_at_epoch=expires_at_epoch,
        max_size=max_size,
    )
