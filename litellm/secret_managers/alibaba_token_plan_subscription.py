"""Shared stdlib-only Alibaba Token Plan subscription identity."""

from __future__ import annotations

import hashlib
from typing import Optional

ALIBABA_TOKEN_PLAN_SUBSCRIPTION_IDENTITY_SOURCE = "instance_code_sha256"


def alibaba_token_plan_subscription_identity(instance_code: str) -> Optional[str]:
    """Hash a non-secret Token Plan instance code.

    The raw instance code is not returned. API keys, RAM secrets, and bearer
    tokens are not accepted as identity material.
    """

    if not isinstance(instance_code, str):
        return None
    normalized = instance_code.strip()
    if not normalized or any(ord(character) < 32 for character in normalized):
        return None
    material = f"alibaba-token-plan|instanceCode={normalized}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()
