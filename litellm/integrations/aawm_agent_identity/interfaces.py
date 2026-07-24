"""Typed seams for the AAWM identity observation pipeline (Wave A3A).

This module introduces the *typed specification* of the rate-limit /
provider-error observation contract that the extracted bands
(``rate_limit_base.py``, ``rate_limit_providers.py``, ``provider_errors.py``)
implement. It is deliberately dependency-light and imports nothing from the
identity host package, so it can be imported freely without risking the
circular-import guard the package relies on.

Behavior note (Wave A3A is behavior-preserving): the extractors still produce
and return plain ``dict`` observations this wave -- the record-dict
input/output contract and the DB payload tuple shapes are UNCHANGED (the
golden parity tests in ``test_rate_limit_observations_unit.py`` pin the exact
dict shape and the byte-identical 22-tuple). :class:`RateLimitObservation` is
therefore a *specification* of that dict shape, not a runtime replacement:
it is declared ``eq=False`` so a dataclass instance is never accidentally
``==`` to the dict the extractors emit (identity equality only), and it is not
yet constructed on the hot path. A later wave may migrate the extractors to
build it directly once the record boundary moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class CallbackEnvelope(Dict[str, Any]):
    """The per-call ``kwargs`` envelope threaded through the callback hooks.

    A typed alias for the mutable mapping LiteLLM hands to
    ``logging_hook`` / ``async_logging_hook`` and the session-history record
    builders. It carries ``model``, ``custom_llm_provider``, ``litellm_call_id``,
    ``litellm_params`` (with nested ``metadata``), ``standard_logging_object``,
    and the pass-through payloads the rate-limit / provider-error extractors
    probe. Subclassing ``Dict`` keeps it a drop-in mapping while documenting the
    contract; it adds no behavior.
    """


class IdentityResolution(Dict[str, Any]):
    """Resolved identity context for a single call.

    A typed alias for the identity fields the context builder resolves
    (``tenant_id``, ``repository``, ``session_id``, ``trace_id``,
    ``account_hash``, ``environment``, ``client_name`` / ``client_version`` /
    ``client_user_agent``). Documented as a mapping alias for the same reason
    as :class:`CallbackEnvelope`.
    """


@dataclass(frozen=True, eq=False)
class RateLimitObservation:
    """Typed specification of a finalized rate-limit observation record.

    Field-for-field mirror of the dict the extractors produce after
    ``_build_rate_limit_context`` + ``_finalize_rate_limit_observation``. The
    union covers the shared context fields, the fields ``_finalize`` derives
    (``quota_period``, ``inferred_window_start_at``, ``limit_key``, ``status``),
    and the per-provider quota fields (``quota_type``, ``remaining_pct``,
    ``quota_limit`` / ``quota_used`` / ``quota_remaining``, billing-period
    bounds, ``exhaustion_kind``).

    ``eq=False`` is load-bearing: the Wave A3A extractors still return plain
    dicts and the golden tests assert ``observations == [golden_dict]``. With
    identity-only equality, a dataclass instance can never compare equal to a
    dict, so introducing this type cannot silently satisfy a dict-equality
    assertion.
    """

    # --- shared context (from _build_rate_limit_context) ---
    observed_at: Optional[datetime] = None
    provider: Optional[str] = None
    client_family: Optional[str] = None
    account_hash: Optional[str] = None
    environment: Optional[str] = None
    tenant_id: Optional[str] = None
    repository: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    litellm_call_id: Optional[str] = None
    route_family: Optional[str] = None
    request_model: Optional[str] = None
    response_model: Optional[str] = None
    model: Optional[str] = None
    model_family: Optional[str] = None
    model_tier: Optional[str] = None
    client_name: Optional[str] = None
    client_version: Optional[str] = None
    client_user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- limit identification ---
    source: Optional[str] = None
    limit_id: Optional[str] = None
    limit_name: Optional[str] = None
    limit_scope: Optional[str] = None
    limit_key: Optional[str] = None

    # --- window / period (derived by _finalize) ---
    window_minutes: Optional[int] = None
    quota_period: Optional[str] = None
    inferred_window_start_at: Optional[datetime] = None
    provider_resets_at: Optional[datetime] = None
    reset_hint_seconds: Optional[int] = None

    # --- usage math ---
    used_percentage: Optional[float] = None
    remaining_requests: Optional[int] = None
    used_requests: Optional[int] = None
    total_requests: Optional[int] = None

    # --- per-provider quota fields ---
    quota_type: Optional[str] = None
    remaining_pct: Optional[float] = None
    quota_limit: Optional[float] = None
    quota_used: Optional[float] = None
    quota_remaining: Optional[float] = None
    billing_period_start_at: Optional[datetime] = None
    billing_period_end_at: Optional[datetime] = None

    # --- exhaustion / status ---
    exhausted: bool = False
    exhaustion_kind: Optional[str] = None
    status: Optional[str] = None

    # --- provenance ---
    raw_provider_fields: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObservationExtractor(Protocol):
    """Protocol for a rate-limit observation extractor.

    Each provider extractor maps a callback envelope + upstream result + the
    observation anchor timestamp to a list of observation records. Explicit
    keyword parameters (no ``**kwargs`` catch-all) pin the call contract so a
    signature drift fails type-checking instead of silently mis-binding.

    The return type is ``List[Dict[str, Any]]`` (not ``List[RateLimitObservation]``)
    because Wave A3A keeps the record-dict contract unchanged; the dataclass is
    the forward-looking specification of that dict shape.
    """

    def __call__(
        self,
        kwargs: CallbackEnvelope,
        result: Any,
        observed_at: Any,
    ) -> List[Dict[str, Any]]:
        ...
