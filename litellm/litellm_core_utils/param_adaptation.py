"""Shared parameter adaptation helpers.

Provides bounded, value-free adaptation metadata for provider parameter
filtering.  Designed for reuse across providers (D1-556).

Adaptation records contain names, actions, and reasons only -- never
parameter values.  They are request-local and must not enter provider
bodies or global/class state.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RECORDS: int = 32
MAX_PARAM_NAME_LENGTH: int = 64
PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY = "provider_parameter_adaptations"
PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY = (
    "provider_parameter_adaptations_truncated_count"
)

# ---------------------------------------------------------------------------
# Typed literals
# ---------------------------------------------------------------------------

AdaptationAction = Literal["dropped", "rejected", "renamed"]
AdaptationReason = Literal[
    "unsupported_param", "extra_body_policy", "invalid_type", "provider_rename"
]

# Runtime-validation tuples (mirror the Literal aliases).
VALID_ACTIONS: Tuple[str, ...] = ("dropped", "rejected", "renamed")
VALID_REASONS: Tuple[str, ...] = (
    "unsupported_param",
    "extra_body_policy",
    "invalid_type",
    "provider_rename",
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptationRecord:
    """A single value-free adaptation event.

    Attributes:
        name: Parameter name (truncated to max_name_length by the collector).
        action: One of ``VALID_ACTIONS``.
        reason: One of ``VALID_REASONS``.

    Raises:
        ValueError: If *action* or *reason* is not in the allowed set.
    """

    name: str
    action: AdaptationAction
    reason: AdaptationReason

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {VALID_ACTIONS}, got {self.action!r}"
            )
        if self.reason not in VALID_REASONS:
            raise ValueError(
                f"reason must be one of {VALID_REASONS}, got {self.reason!r}"
            )


class AdaptationCollector:
    """Request-local bounded collector for adaptation records.

    Caps total records at *max_records* (default 32).  Once the cap is
    reached, further additions are silently counted in *truncated_count*
    but no record is stored.  Parameter names are truncated to
    *max_name_length* (default 64) characters.

    Records are deduplicated by (name, action, reason).  This object must
    never hold parameter values or enter global state.

    Raises:
        ValueError: If *max_records* or *max_name_length* is not a
            positive integer (bool is rejected).
    """

    __slots__ = (
        "_records",
        "_seen",
        "_truncated_count",
        "_max_records",
        "_max_name_len",
    )

    def __init__(
        self,
        max_records: int = MAX_RECORDS,
        max_name_length: int = MAX_PARAM_NAME_LENGTH,
    ) -> None:
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records < 1
        ):
            raise ValueError(
                f"max_records must be a positive integer, got {max_records!r}"
            )
        if (
            isinstance(max_name_length, bool)
            or not isinstance(max_name_length, int)
            or max_name_length < 1
        ):
            raise ValueError(
                f"max_name_length must be a positive integer, got {max_name_length!r}"
            )
        self._records: List[AdaptationRecord] = []
        self._seen: set = set()
        self._truncated_count: int = 0
        self._max_records = max_records
        self._max_name_len = max_name_length

    # -- mutators -----------------------------------------------------------

    def add(
        self, name: str, action: AdaptationAction, reason: AdaptationReason
    ) -> None:
        """Record one adaptation event (value-free).

        Raises:
            ValueError: If *action* or *reason* is not in the allowed set.
        """
        if action not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {VALID_ACTIONS}, got {action!r}")
        if reason not in VALID_REASONS:
            raise ValueError(f"reason must be one of {VALID_REASONS}, got {reason!r}")
        safe_name = str(name)[: self._max_name_len]
        key = (safe_name, action, reason)
        if key in self._seen:
            return
        if len(self._records) >= self._max_records:
            self._truncated_count += 1
            return
        self._seen.add(key)
        self._records.append(
            AdaptationRecord(name=safe_name, action=action, reason=reason)
        )

    def add_many(
        self, names: Sequence[str], action: AdaptationAction, reason: AdaptationReason
    ) -> None:
        """Record adaptation events for multiple parameter names."""
        for n in names:
            self.add(n, action, reason)

    # -- accessors ----------------------------------------------------------

    @property
    def records(self) -> List[AdaptationRecord]:
        """Return a copy of stored records."""
        return list(self._records)

    @property
    def truncated_count(self) -> int:
        """Number of events dropped due to the record cap."""
        return self._truncated_count

    def to_metadata(self) -> dict:
        """Serialize records as deterministic, value-free request metadata."""
        records: List[Dict[str, str]] = [
            {
                "name": record.name,
                "action": record.action,
                "reason": record.reason,
            }
            for record in sorted(
                self._records,
                key=lambda record: (record.name, record.action, record.reason),
            )
        ]
        return {
            PROVIDER_PARAMETER_ADAPTATIONS_METADATA_KEY: records,
            PROVIDER_PARAMETER_ADAPTATIONS_TRUNCATED_COUNT_METADATA_KEY: self._truncated_count,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"AdaptationCollector(records={len(self._records)}, "
            f"truncated={self._truncated_count})"
        )
