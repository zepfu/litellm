"""Open error-class vocabulary + ``FailureEvent`` (Wave 1: D1-583/D1-584 seam).

This is a pure module (no I/O, no provider imports) that both the config-driven
alias framework (D1-583) and the provider/TUI failure-classification effort
(D1-584) build on:

- ``FailureClassRegistry`` is an *open* registry (dict-backed), not a frozen
  ``Enum`` — new failure-class names may be registered/looked up at runtime
  without raising, so later taxonomy growth (e.g. Wave 6's error-CSV
  coverage pass) never forces a schema change here.
- ``FailureEvent`` is the frozen, sanitized event shape produced by the
  classification adapter (Wave 2) and consumed by the cooldown-evidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

Origin = Literal["upstream", "client", "unknown"]
Confidence = Literal["structured", "marker", "unknown"]
Scope = Literal["provider", "account", "model", "lane", "alias"]

_VALID_ORIGINS: frozenset[str] = frozenset({"upstream", "client", "unknown"})
_VALID_CONFIDENCES: frozenset[str] = frozenset({"structured", "marker", "unknown"})
_VALID_SCOPES: frozenset[str] = frozenset({"provider", "account", "model", "lane", "alias"})

# The ~12 structured seed classes. Growth beyond this seed set happens via
# FailureClassRegistry.register() at runtime -- this tuple is a starting
# point, not an exhaustive closed set.
SEED_FAILURE_CLASSES: tuple[str, ...] = (
    "rate_limit",
    "capacity",
    "usage_limit",
    "transient",
    "auth",
    "quota_exhausted",
    "model_unavailable",
    "provider_5xx",
    "provider_4xx_other",
    "serialization",
    "client_cancelled",
    "unknown",
)


class FailureClassRegistry:
    """Open, dict-backed registry of failure-class names.

    Deliberately not a frozen ``Enum``: unknown class names may be registered
    and looked up at runtime without raising, so later taxonomy growth never
    forces a schema change to this module.
    """

    def __init__(self) -> None:
        self._names: set[str] = set()

    def register(self, class_name: str) -> None:
        """Register ``class_name``. Idempotent; never raises for new names."""
        self._names.add(class_name)

    def contains(self, class_name: str) -> bool:
        """Return whether ``class_name`` has been registered."""
        return class_name in self._names

    @classmethod
    def with_seed_classes(cls) -> "FailureClassRegistry":
        """Return a registry pre-seeded with the ~12 structured seed classes."""
        registry = cls()
        for class_name in SEED_FAILURE_CLASSES:
            registry.register(class_name)
        return registry


@dataclass(frozen=True)
class FailureEvent:
    """A single classified failure signal.

    Carries only sanitized, taxonomy-level ``evidence`` -- never raw payloads
    or secrets. Frozen so a classified event cannot be mutated after
    creation.
    """

    class_name: str
    origin: Origin
    confidence: Confidence
    provider: Optional[str]
    scope: Scope
    retryable: Optional[bool]
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.origin not in _VALID_ORIGINS:
            raise ValueError(f"invalid FailureEvent.origin: {self.origin!r}")
        if self.confidence not in _VALID_CONFIDENCES:
            raise ValueError(f"invalid FailureEvent.confidence: {self.confidence!r}")
        if self.scope not in _VALID_SCOPES:
            raise ValueError(f"invalid FailureEvent.scope: {self.scope!r}")


def is_coolable(event: FailureEvent) -> bool:
    """Only ``origin == "upstream"`` events are eligible to cool a candidate.

    ``client`` (e.g. caller-cancelled requests) and ``unknown`` origin
    failures must never advance a candidate toward cooldown.
    """
    return event.origin == "upstream"
