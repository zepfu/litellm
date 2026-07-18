"""Canonical ordered identity-selection helpers for session_history scripts.

Repair/backfill scripts historically each inlined slightly different "first
non-empty source wins" cascades for repository/project identity. This module
owns the shared selection contract so priority order and first-match behavior
cannot drift across those scripts.

Policy-specific sources (which fields count, normalization rules, known-repo
filters) remain caller-owned. This module only provides:

- ``select_first_identity`` — first non-empty candidate from ordered sources
- ``iter_identity_candidates`` — lazy enumeration of (source, value) pairs
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

IdentitySource = Tuple[str, Callable[[], Any]]
IdentityCandidate = Tuple[str, T]


def _default_is_present(value: Any) -> bool:
    """True when *value* is a usable identity result.

    ``None`` and empty/whitespace-only strings are absent. Other values
    (including ``0`` / ``False``) are treated as present so callers that
    intentionally return non-string markers are not filtered out.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def iter_identity_candidates(
    sources: Sequence[IdentitySource],
    *,
    normalize: Optional[Callable[[Any], Any]] = None,
    is_present: Optional[Callable[[Any], bool]] = None,
) -> Iterator[IdentityCandidate[Any]]:
    """Yield ``(source_name, normalized_value)`` for each ordered source.

    *sources* is an ordered sequence of ``(source_name, extractor)`` pairs.
    Each extractor is a zero-arg callable so expensive lookups stay lazy.
    Optional *normalize* runs on every raw extractor result before presence
    checks. *is_present* defaults to non-``None`` and non-blank strings.
    """
    presence = is_present or _default_is_present
    for source_name, extractor in sources:
        raw = extractor()
        value = normalize(raw) if normalize is not None else raw
        if not presence(value):
            continue
        yield source_name, value


def select_first_identity(
    sources: Sequence[IdentitySource],
    *,
    normalize: Optional[Callable[[Any], Any]] = None,
    is_present: Optional[Callable[[Any], bool]] = None,
) -> Optional[IdentityCandidate[Any]]:
    """Return the first present ``(source_name, value)`` from *sources*.

    Lower index in *sources* is higher priority. Returns ``None`` when no
    source yields a present value after optional normalization.
    """
    for candidate in iter_identity_candidates(
        sources,
        normalize=normalize,
        is_present=is_present,
    ):
        return candidate
    return None


def select_first_identity_value(
    sources: Sequence[IdentitySource],
    *,
    normalize: Optional[Callable[[Any], Any]] = None,
    is_present: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Return only the first present identity value (or ``None``)."""
    selected = select_first_identity(
        sources,
        normalize=normalize,
        is_present=is_present,
    )
    if selected is None:
        return None
    return selected[1]


__all__ = [
    "IdentityCandidate",
    "IdentitySource",
    "iter_identity_candidates",
    "select_first_identity",
    "select_first_identity_value",
]
