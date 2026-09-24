"""Canonical Kimi Code OAuth-native contract descriptor.

Pure-stdlib module that can be copied into the provider-status image
without importing the full LiteLLM package.

The contract is read from a configured JSON file on every request so
atomic replacement is restart-free and naturally reaches every worker.

Deployment gate
---------------
Set ``LITELLM_KIMI_NATIVE_CONTRACT_PATH`` to the descriptor file path
and ``LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED=true`` to require a resolved
identity for Kimi routes.  Required mode uses only a current descriptor
or a bounded last-known-good descriptor.  It never fabricates a native
identity.

MS-050 required-mode states:

* ``current`` -- the file just read is structurally valid and unexpired;
* ``lkg`` -- a structurally valid descriptor whose ``expires_at`` has
  passed, but ``now`` is still within
  ``expires_at + LITELLM_KIMI_NATIVE_CONTRACT_LKG_WINDOW_SECONDS``
  (default 3600).  The same bounded snapshot is reused when a later read
  is missing, malformed, digest-invalid, or future-dated;
* ``unavailable`` -- no current or in-window descriptor exists, including
  a descriptor past the LKG window.  Required mode raises
  :class:`KimiNativeContractError` with status 503 and a sanitized
  message that contains no descriptor body, digest, or identity.

Without the required flag the resolver returns ``None`` for unavailable
cases and callers fall back to built-in constants (honest fallback that
does not claim native parity).  Bounded fail-closed behavior can reduce
availability during publisher outages.

This module is strictly read-only: it never writes the descriptor, the
OAuth credential, or any other file, and it never logs descriptor
contents or credentials.  Stale-source telemetry is emitted as one
sanitized warning per classification transition, naming only the
classification and the descriptor path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import stat
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
KIMI_NATIVE_CONTRACT_PATH_ENV = "LITELLM_KIMI_NATIVE_CONTRACT_PATH"
KIMI_NATIVE_CONTRACT_REQUIRED_ENV = "LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED"
KIMI_NATIVE_CONTRACT_LKG_WINDOW_ENV = "LITELLM_KIMI_NATIVE_CONTRACT_LKG_WINDOW_SECONDS"

_logger = logging.getLogger("litellm.secret_managers.kimi_native_contract")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KIMI_NATIVE_BASE_URL = "https://api.kimi.com/coding/v1"
KIMI_NATIVE_SCHEMA_VERSION = 2
KIMI_NATIVE_CONTRACT_MAX_BYTES = 65_536  # 64 KiB

# Required-mode states.  ``current`` and ``lkg`` are published identities.
# ``unavailable`` is not an identity.
KIMI_NATIVE_CONTRACT_SOURCE_CURRENT = "current"
KIMI_NATIVE_CONTRACT_SOURCE_LKG = "lkg"
KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE = "unavailable"
KIMI_NATIVE_CONTRACT_LKG_WINDOW_SECONDS = 3600
KIMI_NATIVE_CONTRACT_UNAVAILABLE_STATUS = 503
KIMI_NATIVE_CONTRACT_UNAVAILABLE_DETAIL = "Kimi native contract is unavailable."
# Historical source labels kept for callers that still import them.
# Required mode does not emit builtin or unbounded stale identities.
KIMI_NATIVE_CONTRACT_SOURCE_DESCRIPTOR = "descriptor"
KIMI_NATIVE_CONTRACT_SOURCE_STALE = "stale"
KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN = "builtin"
_LKG_WINDOW_RE = re.compile(r"\A(?:0|[1-9][0-9]*)\Z")

# Conservative built-in identity floor for the installed Kimi Code client.
# This is a lower bound on the installed client contract, not the claimed
# current version; it claims no native parity beyond the pinned device
# identity below.
KIMI_NATIVE_BUILTIN_CLIENT_VERSION = "0.29.1"
KIMI_NATIVE_BUILTIN_DEVICE_ID = "3dfef765-cf3b-471f-b6bc-d78bba1c4b59"

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

# Sanitized stale/builtin-source telemetry: one warning per source
# classification transition per process.  Messages name only the
# classification and the descriptor path, never descriptor contents or
# credential material.
_SOURCE_TELEMETRY_STATE: Dict[str, object] = {
    "descriptor": False,
    "stale": False,
    "builtin": False,
    "current": False,
    "lkg": False,
    "unavailable": False,
    "path": None,
}
_LKG_BY_PATH: Dict[str, KimiNativeContract] = {}


def _reset_source_telemetry_state() -> None:
    """Test seam: reset per-process source telemetry and the LKG snapshot."""
    _SOURCE_TELEMETRY_STATE["descriptor"] = False
    _SOURCE_TELEMETRY_STATE["stale"] = False
    _SOURCE_TELEMETRY_STATE["builtin"] = False
    _SOURCE_TELEMETRY_STATE["current"] = False
    _SOURCE_TELEMETRY_STATE["lkg"] = False
    _SOURCE_TELEMETRY_STATE["unavailable"] = False
    _SOURCE_TELEMETRY_STATE["path"] = None
    _LKG_BY_PATH.clear()


def _record_contract_source(source: str, path: Optional[str]) -> None:
    """Emit sanitized transition telemetry for LKG and unavailable resolution.

    Only the source classification and descriptor path are logged; the
    descriptor body and credential material are never logged.
    """
    if source == KIMI_NATIVE_CONTRACT_SOURCE_CURRENT:
        _SOURCE_TELEMETRY_STATE["current"] = True
        _SOURCE_TELEMETRY_STATE["path"] = path
        return
    if source == KIMI_NATIVE_CONTRACT_SOURCE_DESCRIPTOR:
        _SOURCE_TELEMETRY_STATE["descriptor"] = True
        _SOURCE_TELEMETRY_STATE["path"] = path
        return
    if (
        _SOURCE_TELEMETRY_STATE.get(source)
        and _SOURCE_TELEMETRY_STATE.get("path") == path
    ):
        return
    _SOURCE_TELEMETRY_STATE[source] = True
    _SOURCE_TELEMETRY_STATE["path"] = path
    if source == KIMI_NATIVE_CONTRACT_SOURCE_LKG:
        _logger.warning(
            "Kimi native contract source=lkg: descriptor at %s is inside "
            "the bounded last-known-good window.",
            path if path else "<unset>",
        )
    elif source == KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE:
        _logger.warning(
            "Kimi native contract source=unavailable: descriptor at %s "
            "has no current or in-window identity.",
            path if path else "<unset>",
        )


class KimiNativeContractError(Exception):
    """Raised when required mode has no current or in-window descriptor.

    ``status_code`` is 503.  The message is sanitized and contains no
    descriptor body, digest, path, or identity fields.
    """

    status_code = KIMI_NATIVE_CONTRACT_UNAVAILABLE_STATUS

    def __init__(self, message: str = KIMI_NATIVE_CONTRACT_UNAVAILABLE_DETAIL) -> None:
        super().__init__(message)
        self.status_code = KIMI_NATIVE_CONTRACT_UNAVAILABLE_STATUS
        self.source = KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE


@dataclasses.dataclass(frozen=True)
class KimiNativeContract:
    """Validated, immutable snapshot of the native contract identity.

    ``source`` classifies where the identity came from:

    * ``current`` -- the descriptor file just read is valid and unexpired;
    * ``lkg`` -- a published descriptor inside the bounded LKG window;
    * ``unavailable`` is not stored on a contract.  Required mode raises
      instead of fabricating an identity.
    """

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
    source: str = KIMI_NATIVE_CONTRACT_SOURCE_DESCRIPTOR


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def compute_canonical_digest(payload: Dict) -> str:
    """Deterministic SHA-256 over canonical non-secret fields.

    The digest covers every field except ``digest`` itself, serialized
    as compact JSON with sorted keys.
    """
    canonical = {k: v for k, v in payload.items() if k != "digest"}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
            normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
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
    raise KimiNativeContractError(f"unparseable timestamp: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_and_build(
    payload: Dict, *, now: float
) -> Tuple[KimiNativeContract, bool]:
    """Strict schema validation and construction.

    Returns ``(contract, expired)``.  Expiry no longer rejects the
    descriptor: it only downgrades the source classification so callers
    keep operating on the older claimed client identity with stale-source
    telemetry.
    """
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
            raise KimiNativeContractError(f"{field} must be a non-empty string")

    for field in _X_MSH_DESCRIPTOR_FIELDS:
        val = payload[field]
        if not isinstance(val, str) or not val:
            raise KimiNativeContractError(f"{field} must be a non-empty string")
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
        raise KimiNativeContractError(f"user_agent must be exactly {expected_ua!r}")
    if payload["x_msh_platform"] != _KIMI_X_MSH_PLATFORM:
        raise KimiNativeContractError(
            f"x_msh_platform must be exactly {_KIMI_X_MSH_PLATFORM!r}"
        )
    if payload["x_msh_version"] != client_version:
        raise KimiNativeContractError(
            "x_msh_version must equal client_version " f"({client_version!r})"
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
    expired = expires_at <= now
    if issued_at > now + _ISSUED_AT_FUTURE_SKEW_SECONDS:
        raise KimiNativeContractError("contract issued_at is in the future")

    digest = payload["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise KimiNativeContractError("digest must be a sha256: prefixed string")
    if digest != compute_canonical_digest(payload):
        raise KimiNativeContractError(
            "digest mismatch: contract may have been tampered with"
        )

    return (
        KimiNativeContract(
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
            source=(
                KIMI_NATIVE_CONTRACT_SOURCE_STALE
                if expired
                else KIMI_NATIVE_CONTRACT_SOURCE_DESCRIPTOR
            ),
        ),
        expired,
    )


# ---------------------------------------------------------------------------
# Conservative installed-client identity (MS-035)
# ---------------------------------------------------------------------------


def _derive_builtin_client_version() -> str:
    """Best-effort read-only lookup of the installed Kimi CLI version.

    Falls back to :data:`KIMI_NATIVE_BUILTIN_CLIENT_VERSION` on any
    failure.  Never raises, writes, or contacts the network.
    """
    kimi_home = os.environ.get("KIMI_CODE_HOME", "~/.kimi-code")
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603,S607
            [os.path.join(os.path.expanduser(kimi_home), "bin", "kimi"), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        version = result.stdout.strip().splitlines()
        if len(version) == 1 and re.fullmatch(r"\d+\.\d+\.\d+", version[0]):
            return version[0]
    except Exception:
        pass
    return KIMI_NATIVE_BUILTIN_CLIENT_VERSION


def _derive_builtin_device_id() -> str:
    """Best-effort read-only lookup of the installed Kimi CLI device ID.

    Falls back to :data:`KIMI_NATIVE_BUILTIN_DEVICE_ID` on any failure.
    Never raises or writes.
    """
    kimi_home = os.environ.get("KIMI_CODE_HOME", "~/.kimi-code")
    try:
        raw = (
            open(
                os.path.join(os.path.expanduser(kimi_home), "device_id"),
                "r",
                encoding="utf-8",
            )
            .read()
            .strip()
        )
        if str(_uuid.UUID(raw)) == raw:
            return raw
    except Exception:
        pass
    return KIMI_NATIVE_BUILTIN_DEVICE_ID


def _build_builtin_contract() -> KimiNativeContract:
    """Conservative installed-client identity for descriptor-less operation."""
    version = _derive_builtin_client_version()
    return KimiNativeContract(
        schema_version=KIMI_NATIVE_SCHEMA_VERSION,
        client_name=_KIMI_CLIENT_NAME,
        client_version=version,
        base_url=KIMI_NATIVE_BASE_URL,
        user_agent=f"{_KIMI_USER_AGENT_PREFIX}{version}",
        issued_at=0.0,
        expires_at=0.0,
        digest="",
        x_msh_platform=_KIMI_X_MSH_PLATFORM,
        x_msh_version=version,
        x_msh_device_name="aawm-service-node",
        x_msh_device_model="aawm-managed",
        x_msh_os_version="linux-6.x",
        x_msh_device_id=_derive_builtin_device_id(),
        source=KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KimiNativeContractDecision:
    """Exact required-mode result for one descriptor read.

    ``source`` and ``state`` are ``current``, ``lkg``, or ``unavailable``.
    ``target`` is the canonical chat-completions URL.  ``headers`` carries
    descriptor identity only for ``current`` and ``lkg``; unavailable
    results have no identity headers.  ``status`` is 200 while a published
    identity is in use and 503 when required mode has nothing to serve.
    """

    state: str
    source: str
    target: str
    headers: Dict[str, str]
    status: int
    contract: Optional[KimiNativeContract] = None


def _resolve_lkg_window(explicit: Optional[int]) -> int:
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
            raise KimiNativeContractError()
        return explicit
    raw = os.environ.get(KIMI_NATIVE_CONTRACT_LKG_WINDOW_ENV)
    if raw is None or raw.strip() == "":
        return KIMI_NATIVE_CONTRACT_LKG_WINDOW_SECONDS
    if _LKG_WINDOW_RE.fullmatch(raw.strip()) is None:
        raise KimiNativeContractError()
    return int(raw)


def _lkg_key(path: Optional[str]) -> str:
    return path or ""


def _classify_published(
    contract: KimiNativeContract, *, now: float, window: int
) -> str:
    if contract.expires_at > now:
        return KIMI_NATIVE_CONTRACT_SOURCE_CURRENT
    if now <= contract.expires_at + window:
        return KIMI_NATIVE_CONTRACT_SOURCE_LKG
    return KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE


def _remember_lkg(path: Optional[str], contract: KimiNativeContract) -> None:
    _LKG_BY_PATH[_lkg_key(path)] = contract


def _recall_lkg(
    path: Optional[str], *, now: float, window: int
) -> Optional[KimiNativeContract]:
    remembered = _LKG_BY_PATH.get(_lkg_key(path))
    if remembered is None:
        return None
    if now <= remembered.expires_at + window:
        return dataclasses.replace(remembered, source=KIMI_NATIVE_CONTRACT_SOURCE_LKG)
    _LKG_BY_PATH.pop(_lkg_key(path), None)
    return None


def _read_descriptor_payload(path: str) -> Optional[Dict]:
    """Return a JSON object, or ``None`` when the file cannot be used.

    Structural rejection here is not an identity.  Callers may still
    serve a bounded in-memory LKG snapshot.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_size > KIMI_NATIVE_CONTRACT_MAX_BYTES:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _decision_for(
    state: str, contract: Optional[KimiNativeContract]
) -> KimiNativeContractDecision:
    target = resolve_endpoint_url(contract, "chat_completions")
    if contract is None or state == KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE:
        return KimiNativeContractDecision(
            state=KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE,
            source=KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE,
            target=target,
            headers={},
            status=KIMI_NATIVE_CONTRACT_UNAVAILABLE_STATUS,
            contract=None,
        )
    stamped = dataclasses.replace(contract, source=state)
    return KimiNativeContractDecision(
        state=state,
        source=state,
        target=target,
        headers=build_outbound_headers(stamped),
        status=200,
        contract=stamped,
    )


def resolve_managed_contract_decision(
    path: Optional[str] = None,
    *,
    required: Optional[bool] = None,
    now: Optional[float] = None,
    lkg_window_seconds: Optional[int] = None,
) -> KimiNativeContractDecision:
    """Classify one descriptor read as current, bounded LKG, or unavailable.

    Does not raise for a missing or invalid descriptor.  Required-mode
    callers that need the sanitized 503 use :func:`resolve_contract`.
    """
    if path is None:
        path = os.environ.get(KIMI_NATIVE_CONTRACT_PATH_ENV)
    if required is None:
        raw = os.environ.get(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, "").strip().lower()
        required = raw in ("1", "true", "yes")
    if now is None:
        now = time.time()
    window = _resolve_lkg_window(lkg_window_seconds)
    _ = required

    published: Optional[KimiNativeContract] = None
    if path:
        payload = _read_descriptor_payload(path)
        if payload is not None:
            try:
                published, _expired = _validate_and_build(payload, now=now)
            except KimiNativeContractError:
                published = None

    if published is not None:
        state = _classify_published(published, now=now, window=window)
        if state == KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE:
            # The publication is past its own window. Keep a remembered
            # snapshot only while that snapshot's deadline is still open.
            recalled = _recall_lkg(path, now=now, window=window)
            if recalled is not None:
                return _decision_for(KIMI_NATIVE_CONTRACT_SOURCE_LKG, recalled)
            return _decision_for(KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE, None)
        stamped = dataclasses.replace(published, source=state)
        _remember_lkg(path, stamped)
        return _decision_for(state, stamped)

    recalled = _recall_lkg(path, now=now, window=window)
    if recalled is not None:
        return _decision_for(KIMI_NATIVE_CONTRACT_SOURCE_LKG, recalled)
    return _decision_for(KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE, None)


def resolve_contract(
    path: Optional[str] = None,
    *,
    required: Optional[bool] = None,
    now: Optional[float] = None,
    lkg_window_seconds: Optional[int] = None,
) -> Optional[KimiNativeContract]:
    """Resolve a current or bounded last-known-good native contract.

    Required mode never fabricates identity.  A current descriptor or an
    in-window LKG descriptor is returned.  Missing, malformed,
    digest-invalid, future, and beyond-LKG reads with no in-window
    snapshot raise :class:`KimiNativeContractError` (status 503, sanitized
    message).  Without the required flag those cases return ``None``.
    """
    if path is None:
        path = os.environ.get(KIMI_NATIVE_CONTRACT_PATH_ENV)
    if required is None:
        raw = os.environ.get(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, "").strip().lower()
        required = raw in ("1", "true", "yes")
    try:
        decision = resolve_managed_contract_decision(
            path,
            required=required,
            now=now,
            lkg_window_seconds=lkg_window_seconds,
        )
    except KimiNativeContractError:
        if required:
            _record_contract_source(KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE, path)
            raise
        return None
    if decision.contract is None:
        if required:
            _record_contract_source(KIMI_NATIVE_CONTRACT_SOURCE_UNAVAILABLE, path)
            raise KimiNativeContractError()
        return None
    _record_contract_source(decision.contract.source, path)
    return decision.contract


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
    base = contract.base_url if contract is not None else KIMI_NATIVE_BASE_URL
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
    user_agent = contract.user_agent if contract is not None else fallback_user_agent
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
            raise KimiNativeContractError("access_token must be a non-empty string")
        headers["Authorization"] = f"Bearer {access_token}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers
