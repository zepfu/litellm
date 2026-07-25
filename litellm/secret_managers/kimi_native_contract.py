"""Canonical Kimi Code OAuth-native contract descriptor.

Pure-stdlib module that can be copied into the provider-status image
without importing the full LiteLLM package.

The contract is read from a configured JSON file on every request so
atomic replacement is restart-free and naturally reaches every worker.

Deployment gate
---------------
Set ``LITELLM_KIMI_NATIVE_CONTRACT_PATH`` to the descriptor file path
and ``LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED=true`` to fail closed when
the descriptor is absent or invalid.  Without the required flag the
resolver returns ``None`` and callers fall back to the built-in
constants (honest fallback that does not claim native parity).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
KIMI_NATIVE_CONTRACT_PATH_ENV = "LITELLM_KIMI_NATIVE_CONTRACT_PATH"
KIMI_NATIVE_CONTRACT_REQUIRED_ENV = "LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KIMI_NATIVE_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_NATIVE_SCHEMA_VERSION = 2
KIMI_NATIVE_CONTRACT_MAX_BYTES = 65_536  # 64 KiB

_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "client_name",
        "client_version",
        "base_url",
        "user_agent",
        "issued_at",
        "expires_at",
        "digest",
        "x_msh_platform",
        "x_msh_version",
        "x_msh_device_name",
        "x_msh_device_model",
        "x_msh_os_version",
        "x_msh_device_id",
    }
)
_ISSUED_AT_FUTURE_SKEW_SECONDS = 300  # 5 min
_ENDPOINT_PATHS = {
    "models": "models",
    "usages": "usages",
    "chat_completions": "chat/completions",
}
_X_MSH_DESCRIPTOR_FIELDS = (
    "x_msh_platform",
    "x_msh_version",
    "x_msh_device_name",
    "x_msh_device_model",
    "x_msh_os_version",
    "x_msh_device_id",
)
_ASCII_PRINTABLE_RE = re.compile(r"\A[\x20-\x7e]+\Z")
_KIMI_CLIENT_NAME = "kimi-code"
_KIMI_USER_AGENT_PREFIX = "kimi-code-cli/"
_KIMI_X_MSH_PLATFORM = "kimi_code_cli"


class KimiNativeContractError(Exception):
    """Raised when the contract descriptor is missing, stale, or malformed."""


@dataclasses.dataclass(frozen=True)
class KimiNativeContract:
    """Validated, immutable snapshot of the native contract descriptor."""

    schema_version: int
    client_name: str
    client_version: str
    base_url: str
    user_agent: str
    issued_at: float
    expires_at: float
    digest: str
    x_msh_platform: str
    x_msh_version: str
    x_msh_device_name: str
    x_msh_device_model: str
    x_msh_os_version: str
    x_msh_device_id: str


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def compute_canonical_digest(payload: Dict) -> str:
    """Deterministic SHA-256 over canonical non-secret fields.

    The digest covers every field except ``digest`` itself, serialized
    as compact JSON with sorted keys.
    """
    canonical = {k: v for k, v in payload.items() if k != "digest"}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def _parse_timestamp(value: object) -> float:
    """Parse an ISO-8601 string or epoch-seconds number."""
    if isinstance(value, bool):
        raise KimiNativeContractError("timestamp must not be a boolean")
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts / 1000.0 if ts > 10_000_000_000 else ts
    if isinstance(value, str):
        # ISO-8601
        try:
            normalized = (
                value.replace("Z", "+00:00") if value.endswith("Z") else value
            )
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
        # Epoch string
        try:
            ts = float(value)
            return ts / 1000.0 if ts > 10_000_000_000 else ts
        except ValueError:
            pass
    raise KimiNativeContractError(
        f"unparseable timestamp: {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_and_build(payload: Dict, *, now: float) -> KimiNativeContract:
    """Strict schema validation and construction."""
    unknown = set(payload.keys()) - _REQUIRED_FIELDS
    if unknown:
        raise KimiNativeContractError(
            f"contract contains unknown fields: {sorted(unknown)}"
        )
    missing = _REQUIRED_FIELDS - set(payload.keys())
    if missing:
        raise KimiNativeContractError(
            f"contract is missing required fields: {sorted(missing)}"
        )

    schema_version = payload["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise KimiNativeContractError("schema_version must be an integer")
    if schema_version != KIMI_NATIVE_SCHEMA_VERSION:
        raise KimiNativeContractError(
            f"unsupported schema_version {schema_version}; "
            f"expected {KIMI_NATIVE_SCHEMA_VERSION}"
        )

    for field in ("client_name", "client_version", "user_agent"):
        val = payload[field]
        if not isinstance(val, str) or not val.strip():
            raise KimiNativeContractError(
            f"{field} must be a non-empty string"
            )

    for field in _X_MSH_DESCRIPTOR_FIELDS:
        val = payload[field]
        if not isinstance(val, str) or not val:
            raise KimiNativeContractError(
                f"{field} must be a non-empty string"
            )
        if not _ASCII_PRINTABLE_RE.match(val):
            raise KimiNativeContractError(
                f"{field} must contain only printable ASCII characters"
            )

    # -- dynamic coherence (version-independent) ---------------------------
    client_version = payload["client_version"]
    if payload["client_name"] != _KIMI_CLIENT_NAME:
        raise KimiNativeContractError(
            f"client_name must be exactly {_KIMI_CLIENT_NAME!r}"
        )
    expected_ua = f"{_KIMI_USER_AGENT_PREFIX}{client_version}"
    if payload["user_agent"] != expected_ua:
        raise KimiNativeContractError(
            f"user_agent must be exactly {expected_ua!r}"
        )
    if payload["x_msh_platform"] != _KIMI_X_MSH_PLATFORM:
        raise KimiNativeContractError(
            f"x_msh_platform must be exactly {_KIMI_X_MSH_PLATFORM!r}"
        )
    if payload["x_msh_version"] != client_version:
        raise KimiNativeContractError(
            "x_msh_version must equal client_version "
            f"({client_version!r})"
        )
    try:
        device_id = payload["x_msh_device_id"]
        if str(_uuid.UUID(device_id)) != device_id:
            raise ValueError
    except (ValueError, AttributeError):
        raise KimiNativeContractError(
            "x_msh_device_id must be a canonical lowercase hyphenated UUID"
        ) from None

    base_url = payload["base_url"]
    if not isinstance(base_url, str) or base_url != KIMI_NATIVE_BASE_URL:
        raise KimiNativeContractError(
            f"base_url must be exactly {KIMI_NATIVE_BASE_URL!r}"
        )

    issued_at = _parse_timestamp(payload["issued_at"])
    expires_at = _parse_timestamp(payload["expires_at"])
    if expires_at <= now:
        raise KimiNativeContractError("contract descriptor is expired")
    if issued_at > now + _ISSUED_AT_FUTURE_SKEW_SECONDS:
        raise KimiNativeContractError("contract issued_at is in the future")

    digest = payload["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise KimiNativeContractError(
            "digest must be a sha256: prefixed string"
        )
    if digest != compute_canonical_digest(payload):
        raise KimiNativeContractError(
            "digest mismatch: contract may have been tampered with"
        )

    return KimiNativeContract(
        schema_version=schema_version,
        client_name=payload["client_name"],
        client_version=payload["client_version"],
        base_url=base_url,
        user_agent=payload["user_agent"],
        issued_at=issued_at,
        expires_at=expires_at,
        digest=digest,
        x_msh_platform=payload["x_msh_platform"],
        x_msh_version=payload["x_msh_version"],
        x_msh_device_name=payload["x_msh_device_name"],
        x_msh_device_model=payload["x_msh_device_model"],
        x_msh_os_version=payload["x_msh_os_version"],
        x_msh_device_id=payload["x_msh_device_id"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_contract(
    path: Optional[str] = None,
    *,
    required: Optional[bool] = None,
    now: Optional[float] = None,
) -> Optional[KimiNativeContract]:
    """Resolve the native contract from the configured descriptor file.

    Returns ``None`` when the descriptor is absent or invalid and not
    required.  Raises :class:`KimiNativeContractError` when *required*
    and the descriptor is absent or invalid (fail closed).
    """
    if path is None:
        path = os.environ.get(KIMI_NATIVE_CONTRACT_PATH_ENV)
    if required is None:
        raw = os.environ.get(
            KIMI_NATIVE_CONTRACT_REQUIRED_ENV, ""
        ).strip().lower()
        required = raw in ("1", "true", "yes")
    if now is None:
        now = time.time()

    if not path:
        if required:
            raise KimiNativeContractError(
                "native contract is required but no descriptor path "
                "is configured"
            )
        return None

    # -- file-level checks --------------------------------------------------
    try:
        st = os.stat(path)
    except OSError:
        if required:
            raise KimiNativeContractError(
                "native contract is required but descriptor is "
                f"missing: {path}"
            )
        return None

    if not stat.S_ISREG(st.st_mode):
        if required:
            raise KimiNativeContractError(
                f"native contract descriptor is not a regular file: {path}"
            )
        return None

    if st.st_size > KIMI_NATIVE_CONTRACT_MAX_BYTES:
        if required:
            raise KimiNativeContractError(
                f"native contract descriptor exceeds "
                f"{KIMI_NATIVE_CONTRACT_MAX_BYTES} bytes"
            )
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    except OSError:
        if required:
            raise KimiNativeContractError(
                f"native contract descriptor is unreadable: {path}"
            )
        return None

    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        if required:
            raise KimiNativeContractError(
                "native contract descriptor is not valid JSON"
            )
        return None

    if not isinstance(payload, dict):
        if required:
            raise KimiNativeContractError(
                "native contract descriptor root must be a JSON object"
            )
        return None

    try:
        return _validate_and_build(payload, now=now)
    except KimiNativeContractError:
        if required:
            raise
        return None


def resolve_endpoint_url(
    contract: Optional[KimiNativeContract],
    usage: str,
) -> str:
    """Resolve the exact endpoint URL for a given *usage*.

    Supported usages: ``"models"``, ``"usages"``, and
    ``"chat_completions"``.
    """
    endpoint_path = _ENDPOINT_PATHS.get(usage)
    if endpoint_path is None:
        raise ValueError(f"unknown contract usage: {usage!r}")
    base = (
        contract.base_url if contract is not None else KIMI_NATIVE_BASE_URL
    )
    return f"{base}/{endpoint_path}"


def build_outbound_headers(
    contract: Optional[KimiNativeContract],
    access_token: Optional[str] = None,
    *,
    json_body: bool = False,
    accept_json: bool = False,
    fallback_user_agent: str = "litellm/unknown",
) -> Dict[str, str]:
    """Build outbound headers from a trusted *access_token*.

    Emits only ``Authorization``, the descriptor ``User-Agent``, and
    ``Content-Type`` when *json_body* is ``True``.  No caller headers
    enter this builder.

    When a contract descriptor is present the six ``X-Msh-*`` identity
    headers are emitted from descriptor-controlled values.  When
    *accept_json* is ``True`` an ``Accept: application/json`` header is
    emitted (models/usages GET parity).
    """
    user_agent = (
        contract.user_agent
        if contract is not None
        else fallback_user_agent
    )
    headers: Dict[str, str] = {"User-Agent": user_agent}
    if contract is not None:
        headers["X-Msh-Platform"] = contract.x_msh_platform
        headers["X-Msh-Version"] = contract.x_msh_version
        headers["X-Msh-Device-Name"] = contract.x_msh_device_name
        headers["X-Msh-Device-Model"] = contract.x_msh_device_model
        headers["X-Msh-Os-Version"] = contract.x_msh_os_version
        headers["X-Msh-Device-Id"] = contract.x_msh_device_id
    if accept_json:
        headers["Accept"] = "application/json"
    if access_token is not None:
        if not isinstance(access_token, str) or not access_token.strip():
            raise KimiNativeContractError(
                "access_token must be a non-empty string"
            )
        headers["Authorization"] = f"Bearer {access_token}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers
