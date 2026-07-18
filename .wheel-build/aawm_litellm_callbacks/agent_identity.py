"""Checkout loader for aawm_litellm_callbacks.agent_identity (RR-003).

This file is intentionally *not* a maintained full source copy of
``litellm.integrations.aawm_agent_identity``.

- In a source checkout, import the canonical integration module and re-export
  its symbols so local ``aawm_litellm_callbacks.agent_identity`` imports stay
  aligned without dual maintenance (including underscore helpers that
  ``import *`` would omit).
- When this package is built into the ``aawm-litellm-callbacks`` wheel, hatch
  force-includes the canonical file as ``aawm_litellm_callbacks/agent_identity.py``
  (see ``.wheel-build/pyproject.toml``), so the installed wheel module is the
  full implementation and does not depend on this loader.

Do not paste the full integration body back into this path.
"""

from __future__ import annotations

from litellm.integrations import aawm_agent_identity as _canonical
from litellm.integrations.aawm_agent_identity import (  # noqa: F401
    AawmAgentIdentity,
    aawm_agent_identity_instance,
)

# Re-export the full canonical namespace, including private helpers that dual
# import probes and tests resolve from either package path.
for _name, _value in vars(_canonical).items():
    if _name.startswith("__") and _name not in {
        "__doc__",
        "__all__",
        "__annotations__",
    }:
        continue
    globals()[_name] = _value

del _name, _value, _canonical

__all__ = ["AawmAgentIdentity", "aawm_agent_identity_instance"]
