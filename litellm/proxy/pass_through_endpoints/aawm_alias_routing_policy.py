"""Compatibility re-export of ``aawm_alias_routing.policy`` (RR-054 #1/#11)."""

from __future__ import annotations

import sys

from .aawm_alias_routing import policy as _policy

_PUBLIC_NAMES = tuple(
    name for name in vars(_policy) if not name.startswith("_")
)
_COMPATIBILITY_MODULE = sys.modules[__name__]
for _name in _PUBLIC_NAMES:
    setattr(_COMPATIBILITY_MODULE, _name, getattr(_policy, _name))

__all__ = [*_PUBLIC_NAMES]
