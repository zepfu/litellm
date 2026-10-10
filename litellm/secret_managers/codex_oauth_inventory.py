"""Explicit, ordered Codex OAuth credential inventory.

The inventory contains only operator-selected labels, paths, routing metadata,
and pinned non-secret account hashes. Credential files are opened read-only,
without following a final symlink, and are never discovered by directory scan.
"""

from __future__ import annotations

import base64
import asyncio
import errno
import hashlib
import json
import math
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

CODEX_OAUTH_INVENTORY_ENV = "LITELLM_CODEX_OAUTH_INVENTORY"
CODEX_OAUTH_INVENTORY_SCHEMA_VERSION = 1
CODEX_OAUTH_ACCOUNT_HASH_LENGTH = 12
CODEX_OAUTH_AUTH_FILE_MODE = 0o600
CODEX_OAUTH_AUTH_FILE_MAX_BYTES = 1_048_576
CODEX_OAUTH_SNAPSHOT_CACHE_MAX_ENTRIES = 128
CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV = "AAWM_CODEX_OAUTH_ACCOUNT_ENABLE_FILE"
CODEX_OAUTH_INVENTORY_GENERATION_LENGTH = hashlib.sha256().digest_size * 2
_TRUE_ENABLE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENABLE_VALUES = frozenset({"0", "false", "no", "off"})

_SAFE_LABEL_RE = re.compile(r"\A[a-z][a-z0-9._-]{0,63}\Z")
_ACCOUNT_HASH_RE = re.compile(rf"\A[0-9a-f]{{{CODEX_OAUTH_ACCOUNT_HASH_LENGTH}}}\Z")
_INVENTORY_GENERATION_RE = re.compile(rf"\A[0-9a-f]{{{CODEX_OAUTH_INVENTORY_GENERATION_LENGTH}}}\Z")
_EMAIL_LOCAL_PART_RE = re.compile(r"\A[A-Za-z0-9._%+-]+\Z")
_EMAIL_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_EMAIL_DOMAIN_LABEL_RE = re.compile(
    r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z",
    re.IGNORECASE,
)
CODEX_OAUTH_REDACTED_ACCOUNT_DISPLAY = "OpenAI-account(redacted)"
_ROOT_FIELDS = frozenset({"schema_version", "routing", "accounts"})
_ROOT_REQUIRED_FIELDS = frozenset({"schema_version", "accounts"})
_ROUTING_FIELDS = frozenset(
    {
        "credential_affinity",
        "strategy",
    }
)
_ACCOUNT_FIELDS = frozenset(
    {
        "label",
        "auth_path",
        "lock_path",
        "priority",
        "weight",
        "enabled",
        "models",
        "expected_account_hash",
    }
)
_PATH_GLOB_CHARACTERS = frozenset("*?[")


class CodexOAuthInventoryError(ValueError):
    """Raised when the explicit inventory is absent or invalid."""


class CodexOAuthCredentialError(CodexOAuthInventoryError):
    """Raised when one configured credential is not safe or usable."""


class CodexOAuthIdentityMismatchError(CodexOAuthCredentialError):
    """Raised when a credential no longer matches its pinned account hash."""

    def __init__(
        self,
        *,
        label: str,
        expected_account_hash: str,
        actual_account_hash: str,
    ) -> None:
        self.label = label
        self.expected_account_hash = expected_account_hash
        self.actual_account_hash = actual_account_hash
        super().__init__(
            f"Codex OAuth credential identity mismatch for account '{label}' "
            f"(expected_hash={expected_account_hash}, "
            f"actual_hash={actual_account_hash})."
        )


@dataclass(frozen=True)
class CodexOAuthCredentialRecord:
    """Validated operator configuration for one independently owned account."""

    label: str
    auth_path: Path = field(repr=False)
    lock_path: Path = field(repr=False)
    priority: int
    weight: float
    enabled: bool
    models: tuple[str, ...]
    expected_account_hash: str
    declaration_order: int = field(repr=False, compare=False)

    def is_model_eligible(self, model: Optional[str]) -> bool:
        if not self.enabled:
            return False
        if model is None:
            return True
        cleaned_model = model.strip()
        if not cleaned_model:
            return False
        return "*" in self.models or cleaned_model in self.models


@dataclass(frozen=True)
class CodexOAuthRoutingPolicy:
    """Optional account-pool routing policy."""

    credential_affinity: str = "pinned"
    strategy: str = "priority"

    @property
    def accounts_are_interchangeable(self) -> bool:
        return self.credential_affinity == "interchangeable"


@dataclass(frozen=True)
class CodexOAuthInventory:
    """Immutable inventory with deterministic priority/declaration ordering."""

    records: tuple[CodexOAuthCredentialRecord, ...]
    routing: CodexOAuthRoutingPolicy = field(default_factory=CodexOAuthRoutingPolicy)

    def ordered_records(
        self,
        *,
        enabled_only: bool = False,
        model: Optional[str] = None,
    ) -> tuple[CodexOAuthCredentialRecord, ...]:
        records = self.records
        if enabled_only or model is not None:
            records = tuple(
                record for record in records if record.enabled and (model is None or record.is_model_eligible(model))
            )
        return tuple(
            sorted(
                records,
                key=lambda record: (record.priority, record.declaration_order),
            )
        )

    def select_record(
        self,
        *,
        label: Optional[str] = None,
        model: Optional[str] = None,
    ) -> CodexOAuthCredentialRecord:
        if label is not None:
            selected = next(
                (record for record in self.records if record.label == label),
                None,
            )
            if selected is None:
                raise CodexOAuthInventoryError("Selected Codex OAuth account label is not configured.")
            if not selected.enabled:
                raise CodexOAuthInventoryError(f"Codex OAuth account '{selected.label}' is disabled.")
            if model is not None and not selected.is_model_eligible(model):
                raise CodexOAuthInventoryError(
                    f"Codex OAuth account '{selected.label}' is not eligible " "for the selected model."
                )
            return selected

        eligible = self.ordered_records(enabled_only=True, model=model)
        if not eligible:
            raise CodexOAuthInventoryError("No enabled Codex OAuth account is eligible for selection.")
        return eligible[0]


@dataclass(frozen=True)
class CodexOAuthCredentialSnapshot:
    """One identity-validated, immutable token snapshot with secrets hidden."""

    record: CodexOAuthCredentialRecord
    account_hash: str
    expires_at: Optional[float]
    access_token: str = field(repr=False)
    account_id: str = field(repr=False)
    account_display: Optional[str] = None
    issued_at: Optional[float] = None
    issued_lifetime_seconds: Optional[float] = None
    raw_token_data: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw_auth_data: Mapping[str, Any] = field(default_factory=dict, repr=False)
    file_identity: Optional[tuple[int, ...]] = field(default=None, repr=False)
    issued_lifetime_source: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("raw_token_data", "raw_auth_data"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise CodexOAuthCredentialError(f"Codex OAuth snapshot {name} must be a mapping.")
            object.__setattr__(self, name, _freeze_auth_value(value))

    def copy_auth_data(self) -> dict[str, Any]:
        """Return a mutable writer copy without changing the factual snapshot."""
        return _copy_auth_value(self.raw_auth_data)


def _freeze_auth_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_auth_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_auth_value(item) for item in value)
    return value


def _copy_auth_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_auth_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_auth_value(item) for item in value]
    return value


_SNAPSHOT_CACHE: dict[CodexOAuthCredentialRecord, CodexOAuthCredentialSnapshot] = {}
_SNAPSHOT_CACHE_LOCK = threading.Lock()


def read_codex_oauth_snapshot_sync(
    record: CodexOAuthCredentialRecord,
    *,
    previous_snapshot: Optional[CodexOAuthCredentialSnapshot] = None,
) -> CodexOAuthCredentialSnapshot:
    """Validate the opened generation, reusing its immutable facts when unchanged."""
    if not record.enabled:
        raise CodexOAuthCredentialError(f"Codex OAuth account '{record.label}' is disabled.")
    with _SNAPSHOT_CACHE_LOCK:
        descriptor = _open_private_auth_file(record.auth_path, label=record.label)
        try:
            identity = _private_auth_file_identity(descriptor, label=record.label)
            for cached in (previous_snapshot, _SNAPSHOT_CACHE.get(record)):
                if cached is not None and cached.record == record and cached.file_identity == identity:
                    return cached
            payload = _read_private_auth_descriptor(descriptor, label=record.label, identity=identity)
        finally:
            os.close(descriptor)
        snapshot = build_codex_oauth_credential_snapshot(
            record=record,
            token_data=get_codex_oauth_token_data(payload, label=record.label),
            auth_data=payload,
            file_identity=identity,
        )
        _SNAPSHOT_CACHE[record] = snapshot
        while len(_SNAPSHOT_CACHE) > CODEX_OAUTH_SNAPSHOT_CACHE_MAX_ENTRIES:
            _SNAPSHOT_CACHE.pop(next(iter(_SNAPSHOT_CACHE)))
        return snapshot


async def read_codex_oauth_snapshot(
    record: CodexOAuthCredentialRecord,
) -> CodexOAuthCredentialSnapshot:
    """Read credential facts without blocking the async request thread."""
    return await asyncio.to_thread(read_codex_oauth_snapshot_sync, record)


def codex_oauth_account_identity_hash(account_id: Any) -> str:
    """Return the existing stable short SHA-256 account identity."""
    cleaned = _clean_string(account_id)
    if cleaned is None:
        raise CodexOAuthCredentialError("Codex OAuth credential account identity is unavailable.")
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:CODEX_OAUTH_ACCOUNT_HASH_LENGTH]


def codex_oauth_inventory_generation_digest(
    inventory: CodexOAuthInventory,
) -> str:
    """Return a stable non-secret identity for the inventory contents.

    The input is the validated configuration, including credential path
    topology, but not credential contents or identities read from those files.
    Record declaration order is retained because it is the validated fallback
    ordering for equal priorities.
    """
    payload = {
        "schema_version": CODEX_OAUTH_INVENTORY_SCHEMA_VERSION,
        "routing": {
            "credential_affinity": inventory.routing.credential_affinity,
            "strategy": inventory.routing.strategy,
        },
        "accounts": [
            {
                "label": record.label,
                "auth_path": record.auth_path.as_posix(),
                "lock_path": record.lock_path.as_posix(),
                "priority": record.priority,
                "weight": record.weight,
                "enabled": record.enabled,
                "models": list(record.models),
                "expected_account_hash": record.expected_account_hash,
            }
            for record in inventory.records
        ],
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def is_codex_oauth_inventory_generation(value: Any) -> bool:
    """Return whether *value* has the canonical inventory digest shape."""
    return isinstance(value, str) and _INVENTORY_GENERATION_RE.fullmatch(value) is not None


def codex_oauth_masked_account_display(email: Any) -> Optional[str]:
    """Return a bounded display form for an untrusted email claim.

    Long local parts expose exactly four leading and four trailing
    characters. Short local parts expose only their first character, so no
    complete local part is returned. Invalid values have no display.
    """
    if not isinstance(email, str) or email != email.strip():
        return None
    cleaned = email.strip()
    if not cleaned or _EMAIL_CONTROL_RE.search(cleaned):
        return None
    if len(cleaned) > 254 or cleaned.count("@") != 1:
        return None
    local_part, domain = cleaned.split("@", 1)
    if (
        not local_part
        or local_part.startswith(".")
        or local_part.endswith(".")
        or ".." in local_part
        or _EMAIL_LOCAL_PART_RE.fullmatch(local_part) is None
    ):
        return None
    domain_labels = domain.split(".")
    if (
        len(domain) > 253
        or len(domain_labels) < 2
        or any(_EMAIL_DOMAIN_LABEL_RE.fullmatch(label) is None for label in domain_labels)
    ):
        return None
    if len(local_part) >= 10:
        return f"{local_part[:4]}*{local_part[-4:]}@{domain}"
    if len(local_part) >= 3:
        return f"{local_part[0]}*{local_part[-1]}@{domain}"
    if len(local_part) == 2:
        return f"{local_part[0]}*@{domain}"
    return f"*@{domain}"


def load_codex_oauth_inventory(
    raw_inventory: Optional[str] = None,
) -> CodexOAuthInventory:
    """Parse the explicit JSON inventory from an argument or environment."""
    raw_value = raw_inventory if raw_inventory is not None else os.getenv(CODEX_OAUTH_INVENTORY_ENV)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise CodexOAuthInventoryError(f"Codex OAuth inventory is not configured in " f"{CODEX_OAUTH_INVENTORY_ENV}.")
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        raise CodexOAuthInventoryError("Codex OAuth inventory is not valid JSON.") from None
    return parse_codex_oauth_inventory(payload)


async def load_codex_oauth_inventory_async(
    raw_inventory: Optional[str] = None,
) -> CodexOAuthInventory:
    """Parse inventory and overlay without blocking the async request thread."""
    if raw_inventory is None:
        return await asyncio.to_thread(load_codex_oauth_inventory)
    return await asyncio.to_thread(load_codex_oauth_inventory, raw_inventory)


def parse_codex_oauth_inventory(payload: Any) -> CodexOAuthInventory:
    """Validate a versioned inventory payload and return immutable records."""
    if not isinstance(payload, dict):
        raise CodexOAuthInventoryError("Codex OAuth inventory must contain a JSON object.")
    actual_fields = set(payload)
    missing_fields = _ROOT_REQUIRED_FIELDS - actual_fields
    unknown_fields = actual_fields - _ROOT_FIELDS
    if missing_fields:
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory is missing required fields: " f"{sorted(missing_fields)}."
        )
    if unknown_fields:
        raise CodexOAuthInventoryError("Codex OAuth inventory contains unknown fields: " f"{sorted(unknown_fields)}.")

    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != CODEX_OAUTH_INVENTORY_SCHEMA_VERSION
    ):
        raise CodexOAuthInventoryError("Codex OAuth inventory has an unsupported schema_version.")

    accounts = payload.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        raise CodexOAuthInventoryError("Codex OAuth inventory accounts must be a non-empty array.")

    routing = _parse_routing_policy(payload.get("routing"))
    overlay = load_codex_oauth_account_enable_overlay()
    records: list[CodexOAuthCredentialRecord] = []
    seen_labels: set[str] = set()
    seen_paths: dict[str, str] = {}
    seen_account_hashes: dict[str, str] = {}
    for declaration_order, account in enumerate(accounts):
        record = _parse_account_record(
            account,
            declaration_order=declaration_order,
            overlay=overlay,
        )
        if record.label in seen_labels:
            raise CodexOAuthInventoryError(f"Duplicate Codex OAuth account label '{record.label}'.")
        seen_labels.add(record.label)

        for path in (record.auth_path, record.lock_path):
            path_key = os.path.normcase(os.fspath(path))
            prior_label = seen_paths.get(path_key)
            if prior_label is not None:
                raise CodexOAuthInventoryError(
                    f"Codex OAuth accounts '{prior_label}' and " f"'{record.label}' reuse a configured path."
                )
            seen_paths[path_key] = record.label

        prior_identity_label = seen_account_hashes.get(record.expected_account_hash)
        if prior_identity_label is not None:
            raise CodexOAuthInventoryError(
                f"Codex OAuth accounts '{prior_identity_label}' and " f"'{record.label}' pin the same account identity."
            )
        seen_account_hashes[record.expected_account_hash] = record.label
        records.append(record)

    return CodexOAuthInventory(records=tuple(records), routing=routing)


def _parse_routing_policy(value: Any) -> CodexOAuthRoutingPolicy:
    if value is None:
        return CodexOAuthRoutingPolicy()
    if not isinstance(value, dict):
        raise CodexOAuthInventoryError("Codex OAuth inventory routing must contain a JSON object.")
    _validate_exact_fields(
        value,
        expected=_ROUTING_FIELDS,
        subject="Codex OAuth inventory routing",
    )
    credential_affinity = value.get("credential_affinity", "pinned")
    if credential_affinity not in {"pinned", "interchangeable"}:
        raise CodexOAuthInventoryError("Codex OAuth routing credential_affinity is unsupported.")
    strategy = value.get("strategy", "priority")
    if strategy != "priority":
        raise CodexOAuthInventoryError("Codex OAuth routing strategy is unsupported.")
    return CodexOAuthRoutingPolicy(
        credential_affinity=credential_affinity,
        strategy=strategy,
    )


def load_codex_oauth_credential(
    record: CodexOAuthCredentialRecord,
) -> CodexOAuthCredentialSnapshot:
    """Read and validate exactly one selected credential without writing."""
    return read_codex_oauth_snapshot_sync(record)


def build_codex_oauth_credential_snapshot(
    *,
    record: CodexOAuthCredentialRecord,
    token_data: Mapping[str, Any],
    auth_data: Optional[Mapping[str, Any]] = None,
    file_identity: Optional[tuple[int, ...]] = None,
) -> CodexOAuthCredentialSnapshot:
    """Validate a token mapping and construct its immutable factual snapshot.

    ``file_identity`` marks a direct descriptor-backed read and is the only
    way to create a cache-eligible snapshot. Refresh callers can use the raw
    token mapping after an in-memory response without reopening the file.
    """
    if not record.enabled:
        raise CodexOAuthCredentialError(f"Codex OAuth account '{record.label}' is disabled.")
    access_token = _clean_string(token_data.get("access_token"))
    if access_token is None:
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{record.label}' is missing access_token.")
    account_id, account_hash, account_display = validate_codex_oauth_account_identity(
        record,
        token_data,
    )
    issued_at = get_codex_oauth_token_issued_at(token_data)
    expires_at = get_codex_oauth_token_expiry(token_data)
    issued_lifetime, issued_lifetime_source = get_codex_oauth_token_issued_lifetime(
        expires_in=token_data.get("expires_in"),
        access_token=access_token,
        expires_at=expires_at,
        issued_at=token_data.get("issued_at"),
        obtained_at=token_data.get("obtained_at"),
        refreshed_at=token_data.get("refreshed_at"),
    )
    return CodexOAuthCredentialSnapshot(
        record=record,
        account_hash=account_hash,
        expires_at=expires_at,
        access_token=access_token,
        account_id=account_id,
        account_display=account_display or CODEX_OAUTH_REDACTED_ACCOUNT_DISPLAY,
        issued_at=issued_at,
        issued_lifetime_seconds=issued_lifetime,
        issued_lifetime_source=issued_lifetime_source,
        raw_token_data=token_data,
        raw_auth_data=auth_data if auth_data is not None else {"tokens": token_data},
        file_identity=file_identity,
    )


def get_codex_oauth_token_data(
    auth_data: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Return a strict nested or legacy-flat token mapping."""
    if "tokens" not in auth_data:
        return auth_data
    token_data = auth_data.get("tokens")
    if isinstance(token_data, Mapping):
        return token_data
    raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' has an invalid tokens object.")


def validate_codex_oauth_account_identity(
    record: CodexOAuthCredentialRecord,
    token_data: Mapping[str, Any],
) -> tuple[str, str, Optional[str]]:
    """Validate internal identity coherence and the configured account pin."""
    account_ids: list[str] = []
    configured_account_id = _clean_string(token_data.get("account_id"))
    if configured_account_id is not None:
        account_ids.append(configured_account_id)

    for token_field in ("id_token", "access_token"):
        token = _clean_string(token_data.get(token_field))
        if token is None:
            continue
        token_account_id = _extract_account_id_from_token(token)
        if token_account_id is not None:
            account_ids.append(token_account_id)

    unique_account_ids = set(account_ids)
    if not unique_account_ids:
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{record.label}' has no account identity.")
    if len(unique_account_ids) != 1:
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' contains mismatched " "account identities."
        )

    account_id = unique_account_ids.pop()
    account_hash = codex_oauth_account_identity_hash(account_id)
    if account_hash != record.expected_account_hash:
        raise CodexOAuthIdentityMismatchError(
            label=record.label,
            expected_account_hash=record.expected_account_hash,
            actual_account_hash=account_hash,
        )
    # D1-676: the validated top-level ID-token `email` claim is the sole
    # display authority; nested and access-token email claims stay redacted.
    id_token = _clean_string(token_data.get("id_token"))
    if id_token is not None:
        id_token_claims = _decode_jwt_claims_without_validation(id_token)
        top_level_email = codex_oauth_masked_account_display(id_token_claims.get("email"))
        if top_level_email is not None:
            return account_id, account_hash, top_level_email
    return account_id, account_hash, None


def get_codex_oauth_token_expiry(
    token_data: Mapping[str, Any],
) -> Optional[float]:
    expires_at = _parse_timestamp_seconds(token_data.get("expires_at"))
    if expires_at is not None:
        return expires_at

    access_token = _clean_string(token_data.get("access_token"))
    if access_token is not None:
        exp = _parse_finite_seconds(_decode_jwt_claims_without_validation(access_token).get("exp"))
        if exp is not None:
            return exp
    issued_at = _first_timestamp_seconds(
        token_data.get("issued_at"),
        token_data.get("obtained_at"),
        token_data.get("refreshed_at"),
    )
    lifetime = _parse_finite_seconds(token_data.get("expires_in"))
    if issued_at is not None and lifetime is not None and lifetime > 0:
        expiry = issued_at + lifetime
        if math.isfinite(expiry):
            return expiry
    return None


def get_codex_oauth_token_issued_at(
    token_data: Mapping[str, Any],
) -> Optional[float]:
    issued_at = _first_timestamp_seconds(
        token_data.get("issued_at"),
        token_data.get("obtained_at"),
        token_data.get("refreshed_at"),
    )
    if issued_at is not None:
        return issued_at
    access_token = _clean_string(token_data.get("access_token"))
    if access_token is None:
        return None
    jwt_claims = _get_jwt_time_claims(access_token)
    if jwt_claims is not None:
        return jwt_claims[0]
    return None


def get_codex_oauth_token_issued_lifetime(
    *,
    expires_in: Any = None,
    access_token: Optional[str] = None,
    expires_at: Any = None,
    issued_at: Any = None,
    obtained_at: Any = None,
    refreshed_at: Any = None,
) -> tuple[Optional[float], str]:
    """Derive issued lifetime using the shared refresh-parser precedence."""
    provider_lifetime = _parse_finite_seconds(expires_in)
    if provider_lifetime is not None and provider_lifetime > 0:
        return provider_lifetime, "expires_in"

    jwt_claims = _get_jwt_time_claims(access_token)
    if jwt_claims is not None:
        jwt_issued_at, jwt_expires_at = jwt_claims
        return jwt_expires_at - jwt_issued_at, "jwt"

    persisted_issued_at = _first_timestamp_seconds(
        issued_at,
        obtained_at,
        refreshed_at,
    )
    persisted_expires_at = _parse_timestamp_seconds(expires_at)
    if (
        persisted_issued_at is not None
        and persisted_expires_at is not None
        and persisted_expires_at > persisted_issued_at
    ):
        return persisted_expires_at - persisted_issued_at, "persisted_timestamp"
    return None, "fallback"


def _parse_timestamp_seconds(value: Any) -> Optional[float]:
    numeric = _parse_finite_seconds(value)
    if numeric is not None:
        return numeric
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        timestamp = value.astimezone(timezone.utc).timestamp()
        return timestamp if math.isfinite(timestamp) else None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    timestamp = parsed.astimezone(timezone.utc).timestamp()
    return timestamp if math.isfinite(timestamp) else None


def _parse_finite_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _first_timestamp_seconds(*values: Any) -> Optional[float]:
    for value in values:
        timestamp = _parse_timestamp_seconds(value)
        if timestamp is not None:
            return timestamp
    return None


def _get_jwt_time_claims(
    access_token: Optional[str],
) -> Optional[Tuple[float, float]]:
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    if len(access_token.split(".")) != 3:
        return None
    claims = _decode_jwt_claims_without_validation(access_token)
    issued_at = _parse_finite_seconds(claims.get("iat"))
    expires_at = _parse_finite_seconds(claims.get("exp"))
    if issued_at is None or expires_at is None or expires_at <= issued_at:
        return None
    return issued_at, expires_at


def _parse_account_record(
    payload: Any,
    *,
    declaration_order: int,
    overlay: Optional[Mapping[str, bool]] = None,
) -> CodexOAuthCredentialRecord:
    if not isinstance(payload, dict):
        raise CodexOAuthInventoryError("Each Codex OAuth inventory account must be a JSON object.")
    _validate_exact_fields(
        payload,
        expected=_ACCOUNT_FIELDS,
        subject="Codex OAuth inventory account",
    )

    label_value = payload.get("label")
    if not isinstance(label_value, str) or _SAFE_LABEL_RE.fullmatch(label_value) is None:
        raise CodexOAuthInventoryError("Codex OAuth account labels must be lowercase safe identifiers.")
    label = label_value

    auth_path = _parse_explicit_path(
        payload.get("auth_path"),
        label=label,
        field_name="auth_path",
    )
    lock_path = _parse_explicit_path(
        payload.get("lock_path"),
        label=label,
        field_name="lock_path",
    )
    if auth_path == lock_path:
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' must use distinct auth and lock paths.")

    priority = payload.get("priority")
    if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' priority must be a non-negative integer.")

    weight_value = payload.get("weight")
    if (
        not isinstance(weight_value, (int, float))
        or isinstance(weight_value, bool)
        or not math.isfinite(float(weight_value))
        or float(weight_value) <= 0
    ):
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' weight must be a positive number.")
    weight = float(weight_value)

    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' enabled must be a boolean.")
    enabled = resolve_codex_oauth_account_enabled(
        label,
        enabled,
        overlay=overlay,
    )

    models_value = payload.get("models")
    if not isinstance(models_value, list) or not models_value:
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' models must be a non-empty array.")
    models: list[str] = []
    for model in models_value:
        if not isinstance(model, str) or not model.strip():
            raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' models must contain " "non-empty strings.")
        cleaned_model = model.strip()
        if any(character in cleaned_model for character in "?["):
            raise CodexOAuthInventoryError(
                f"Codex OAuth account '{label}' model eligibility does not " "support glob patterns."
            )
        if "*" in cleaned_model and cleaned_model != "*":
            raise CodexOAuthInventoryError(
                f"Codex OAuth account '{label}' model eligibility supports "
                "only exact names or the explicit '*' value."
            )
        if cleaned_model in models:
            raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' has duplicate model eligibility.")
        models.append(cleaned_model)
    if "*" in models and len(models) != 1:
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' cannot combine '*' with exact models.")

    expected_account_hash = payload.get("expected_account_hash")
    if not isinstance(expected_account_hash, str) or _ACCOUNT_HASH_RE.fullmatch(expected_account_hash) is None:
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' expected_account_hash must be a "
            f"{CODEX_OAUTH_ACCOUNT_HASH_LENGTH}-character lowercase SHA-256 prefix."
        )

    return CodexOAuthCredentialRecord(
        label=label,
        auth_path=auth_path,
        lock_path=lock_path,
        priority=priority,
        weight=weight,
        enabled=enabled,
        models=tuple(models),
        expected_account_hash=expected_account_hash,
        declaration_order=declaration_order,
    )


def _parse_explicit_path(
    value: Any,
    *,
    label: str,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' {field_name} must be a non-empty path.")
    raw_path = value.strip()
    if any(character in raw_path for character in _PATH_GLOB_CHARACTERS):
        raise CodexOAuthInventoryError(f"Codex OAuth account '{label}' {field_name} must name one explicit path.")
    expanded_path = Path(raw_path).expanduser()
    if not expanded_path.is_absolute():
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' {field_name} must be absolute " "after user expansion."
        )
    return Path(os.path.abspath(os.fspath(expanded_path)))


def _validate_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    subject: str,
) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise CodexOAuthInventoryError(f"{subject} is missing required fields: {sorted(missing)}.")
    if unknown:
        raise CodexOAuthInventoryError(f"{subject} contains unknown fields: {sorted(unknown)}.")


def read_codex_oauth_auth_payload(
    auth_path: Path,
    *,
    label: str = "legacy",
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Read a bounded private payload using the same policy as managed snapshots."""
    file_descriptor = _open_private_auth_file(auth_path, label=label)
    try:
        identity = _private_auth_file_identity(file_descriptor, label=label)
        payload = _read_private_auth_descriptor(file_descriptor, label=label, identity=identity)
    finally:
        os.close(file_descriptor)
    return payload, identity


def _private_auth_file_identity(descriptor: int, *, label: str) -> tuple[int, ...]:
    file_stat = os.fstat(descriptor)
    if not stat.S_ISREG(file_stat.st_mode):
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' is not a regular file.")
    if stat.S_IMODE(file_stat.st_mode) != CODEX_OAUTH_AUTH_FILE_MODE:
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{label}' has invalid permissions; " "expected mode 0600."
        )
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
        file_stat.st_mode,
        file_stat.st_uid,
        file_stat.st_gid,
    )


def _read_private_auth_descriptor(descriptor: int, *, label: str, identity: tuple[int, ...]) -> dict[str, Any]:
    raw_bytes = _read_bounded_bytes(descriptor, label=label)
    if _private_auth_file_identity(descriptor, label=label) != identity:
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' changed during its read.")
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' is malformed.") from None
    if not isinstance(payload, dict):
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' must contain a JSON object.")
    return payload


def _open_private_auth_file(auth_path: Path, *, label: str) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CodexOAuthCredentialError("Secure Codex OAuth credential reads are unsupported on this platform.")
    flags = os.O_RDONLY | os.O_NONBLOCK | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    try:
        return os.open(auth_path, flags)
    except FileNotFoundError:
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' is missing.") from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            reason = "is a symlink or unsafe path"
        elif exc.errno in (errno.EACCES, errno.EPERM):
            reason = "is not readable"
        else:
            reason = "could not be opened"
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' {reason}.") from None


def _read_bounded_bytes(file_descriptor: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    bytes_read = 0
    while bytes_read <= CODEX_OAUTH_AUTH_FILE_MAX_BYTES:
        try:
            chunk = os.read(
                file_descriptor,
                min(
                    8192,
                    CODEX_OAUTH_AUTH_FILE_MAX_BYTES + 1 - bytes_read,
                ),
            )
        except OSError:
            raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' could not be read.") from None
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
    if bytes_read > CODEX_OAUTH_AUTH_FILE_MAX_BYTES:
        raise CodexOAuthCredentialError(f"Codex OAuth credential '{label}' is too large.")
    return b"".join(chunks)


def _extract_account_id_from_token(token: str) -> Optional[str]:
    claims = _decode_jwt_claims_without_validation(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if not isinstance(auth_claims, dict):
        return None
    return _clean_string(auth_claims.get("chatgpt_account_id"))


def _decode_jwt_claims_without_validation(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _clean_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _codex_oauth_account_enabled_env_name(label: str) -> str:
    normalized = re.sub(r"[.\-]", "_", label.strip()).upper()
    return f"AAWM_CODEX_OAUTH_{normalized}_ENABLED"


def _parse_codex_oauth_account_enabled_flag(
    value: Any,
    *,
    source: str,
) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value.strip():
        raise CodexOAuthInventoryError(f"Codex OAuth account enable override from {source} must be a boolean.")
    normalized = value.strip().lower()
    if normalized in _TRUE_ENABLE_VALUES:
        return True
    if normalized in _FALSE_ENABLE_VALUES:
        return False
    raise CodexOAuthInventoryError(f"Codex OAuth account enable override from {source} is invalid.")


def load_codex_oauth_account_enable_overlay(
    overlay_path: Optional[str] = None,
) -> dict[str, bool]:
    """Read the optional testing overlay of label -> enabled flags."""
    raw_path = overlay_path if overlay_path is not None else os.getenv(CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    path = Path(raw_path.strip())
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, json.JSONDecodeError):
        raise CodexOAuthInventoryError("Codex OAuth account enable overlay is not valid JSON.") from None
    if not isinstance(payload, dict):
        raise CodexOAuthInventoryError("Codex OAuth account enable overlay must be a JSON object.")
    overlay: dict[str, bool] = {}
    for label, value in payload.items():
        if not isinstance(label, str) or _SAFE_LABEL_RE.fullmatch(label) is None:
            continue
        overlay[label] = _parse_codex_oauth_account_enabled_flag(
            value,
            source=f"overlay:{label}",
        )
    return overlay


def resolve_codex_oauth_account_enabled(
    label: str,
    configured_enabled: bool,
    *,
    overlay: Optional[Mapping[str, bool]] = None,
) -> bool:
    """Resolve effective account enablement: overlay, then env, then JSON."""
    mapping = overlay if overlay is not None else load_codex_oauth_account_enable_overlay()
    if label in mapping:
        return bool(mapping[label])
    env_name = _codex_oauth_account_enabled_env_name(label)
    env_value = os.getenv(env_name)
    if env_value is not None and str(env_value).strip():
        return _parse_codex_oauth_account_enabled_flag(
            env_value,
            source=env_name,
        )
    return bool(configured_enabled)


__all__ = [
    "CODEX_OAUTH_REDACTED_ACCOUNT_DISPLAY",
    "CODEX_OAUTH_ACCOUNT_HASH_LENGTH",
    "CODEX_OAUTH_AUTH_FILE_MAX_BYTES",
    "CODEX_OAUTH_AUTH_FILE_MODE",
    "CODEX_OAUTH_INVENTORY_ENV",
    "CODEX_OAUTH_INVENTORY_GENERATION_LENGTH",
    "CODEX_OAUTH_INVENTORY_SCHEMA_VERSION",
    "CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV",
    "CodexOAuthCredentialError",
    "CodexOAuthCredentialRecord",
    "CodexOAuthCredentialSnapshot",
    "CodexOAuthIdentityMismatchError",
    "CodexOAuthInventory",
    "CodexOAuthInventoryError",
    "CodexOAuthRoutingPolicy",
    "codex_oauth_account_identity_hash",
    "codex_oauth_masked_account_display",
    "codex_oauth_inventory_generation_digest",
    "is_codex_oauth_inventory_generation",
    "get_codex_oauth_token_data",
    "get_codex_oauth_token_expiry",
    "get_codex_oauth_token_issued_at",
    "get_codex_oauth_token_issued_lifetime",
    "build_codex_oauth_credential_snapshot",
    "load_codex_oauth_credential",
    "read_codex_oauth_snapshot",
    "read_codex_oauth_snapshot_sync",
    "read_codex_oauth_auth_payload",
    "load_codex_oauth_inventory",
    "load_codex_oauth_inventory_async",
    "load_codex_oauth_account_enable_overlay",
    "parse_codex_oauth_inventory",
    "resolve_codex_oauth_account_enabled",
    "validate_codex_oauth_account_identity",
]
