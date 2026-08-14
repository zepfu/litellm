"""Explicit, ordered Codex OAuth credential inventory.

The inventory contains only operator-selected labels, paths, routing metadata,
and pinned non-secret account hashes. Credential files are opened read-only,
without following a final symlink, and are never discovered by directory scan.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

CODEX_OAUTH_INVENTORY_ENV = "LITELLM_CODEX_OAUTH_INVENTORY"
CODEX_OAUTH_INVENTORY_SCHEMA_VERSION = 1
CODEX_OAUTH_ACCOUNT_HASH_LENGTH = 12
CODEX_OAUTH_AUTH_FILE_MODE = 0o600
CODEX_OAUTH_AUTH_FILE_MAX_BYTES = 1_048_576

_SAFE_LABEL_RE = re.compile(r"\A[a-z][a-z0-9._-]{0,63}\Z")
_ACCOUNT_HASH_RE = re.compile(
    rf"\A[0-9a-f]{{{CODEX_OAUTH_ACCOUNT_HASH_LENGTH}}}\Z"
)
_ROOT_FIELDS = frozenset({"schema_version", "routing", "accounts"})
_ROOT_REQUIRED_FIELDS = frozenset({"schema_version", "accounts"})
_ROUTING_FIELDS = frozenset(
    {
        "credential_affinity",
        "strategy",
        "balance_band_percentage_points",
        "within_band_strategy",
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
    """Optional account-pool routing policy with backward-compatible defaults."""

    credential_affinity: str = "pinned"
    strategy: str = "priority"
    balance_band_percentage_points: float = 10.0
    within_band_strategy: str = "priority"

    @property
    def accounts_are_interchangeable(self) -> bool:
        return self.credential_affinity == "interchangeable"


@dataclass(frozen=True)
class CodexOAuthInventory:
    """Immutable inventory with deterministic priority/declaration ordering."""

    records: tuple[CodexOAuthCredentialRecord, ...]
    routing: CodexOAuthRoutingPolicy = field(
        default_factory=CodexOAuthRoutingPolicy
    )

    def ordered_records(
        self,
        *,
        enabled_only: bool = False,
        model: Optional[str] = None,
    ) -> tuple[CodexOAuthCredentialRecord, ...]:
        records = self.records
        if enabled_only or model is not None:
            records = tuple(
                record
                for record in records
                if record.enabled
                and (model is None or record.is_model_eligible(model))
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
                raise CodexOAuthInventoryError(
                    "Selected Codex OAuth account label is not configured."
                )
            if not selected.enabled:
                raise CodexOAuthInventoryError(
                    f"Codex OAuth account '{selected.label}' is disabled."
                )
            if model is not None and not selected.is_model_eligible(model):
                raise CodexOAuthInventoryError(
                    f"Codex OAuth account '{selected.label}' is not eligible "
                    "for the selected model."
                )
            return selected

        eligible = self.ordered_records(enabled_only=True, model=model)
        if not eligible:
            raise CodexOAuthInventoryError(
                "No enabled Codex OAuth account is eligible for selection."
            )
        return eligible[0]


@dataclass(frozen=True)
class CodexOAuthCredentialSnapshot:
    """One atomically read credential snapshot with secret fields hidden."""

    record: CodexOAuthCredentialRecord
    account_hash: str
    expires_at: Optional[float]
    access_token: str = field(repr=False)
    account_id: str = field(repr=False)


def codex_oauth_account_identity_hash(account_id: Any) -> str:
    """Return the existing stable short SHA-256 account identity."""
    cleaned = _clean_string(account_id)
    if cleaned is None:
        raise CodexOAuthCredentialError(
            "Codex OAuth credential account identity is unavailable."
        )
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[
        :CODEX_OAUTH_ACCOUNT_HASH_LENGTH
    ]


def load_codex_oauth_inventory(
    raw_inventory: Optional[str] = None,
) -> CodexOAuthInventory:
    """Parse the explicit JSON inventory from an argument or environment."""
    raw_value = (
        raw_inventory
        if raw_inventory is not None
        else os.getenv(CODEX_OAUTH_INVENTORY_ENV)
    )
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise CodexOAuthInventoryError(
            f"Codex OAuth inventory is not configured in "
            f"{CODEX_OAUTH_INVENTORY_ENV}."
        )
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory is not valid JSON."
        ) from None
    return parse_codex_oauth_inventory(payload)


def parse_codex_oauth_inventory(payload: Any) -> CodexOAuthInventory:
    """Validate a versioned inventory payload and return immutable records."""
    if not isinstance(payload, dict):
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory must contain a JSON object."
        )
    actual_fields = set(payload)
    missing_fields = _ROOT_REQUIRED_FIELDS - actual_fields
    unknown_fields = actual_fields - _ROOT_FIELDS
    if missing_fields:
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory is missing required fields: "
            f"{sorted(missing_fields)}."
        )
    if unknown_fields:
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory contains unknown fields: "
            f"{sorted(unknown_fields)}."
        )

    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != CODEX_OAUTH_INVENTORY_SCHEMA_VERSION
    ):
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory has an unsupported schema_version."
        )

    accounts = payload.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory accounts must be a non-empty array."
        )

    routing = _parse_routing_policy(payload.get("routing"))
    records: list[CodexOAuthCredentialRecord] = []
    seen_labels: set[str] = set()
    seen_paths: dict[str, str] = {}
    seen_account_hashes: dict[str, str] = {}
    for declaration_order, account in enumerate(accounts):
        record = _parse_account_record(account, declaration_order=declaration_order)
        if record.label in seen_labels:
            raise CodexOAuthInventoryError(
                f"Duplicate Codex OAuth account label '{record.label}'."
            )
        seen_labels.add(record.label)

        for path in (record.auth_path, record.lock_path):
            path_key = os.path.normcase(os.fspath(path))
            prior_label = seen_paths.get(path_key)
            if prior_label is not None:
                raise CodexOAuthInventoryError(
                    f"Codex OAuth accounts '{prior_label}' and "
                    f"'{record.label}' reuse a configured path."
                )
            seen_paths[path_key] = record.label

        prior_identity_label = seen_account_hashes.get(
            record.expected_account_hash
        )
        if prior_identity_label is not None:
            raise CodexOAuthInventoryError(
                f"Codex OAuth accounts '{prior_identity_label}' and "
                f"'{record.label}' pin the same account identity."
            )
        seen_account_hashes[record.expected_account_hash] = record.label
        records.append(record)

    return CodexOAuthInventory(records=tuple(records), routing=routing)


def _parse_routing_policy(value: Any) -> CodexOAuthRoutingPolicy:
    if value is None:
        return CodexOAuthRoutingPolicy()
    if not isinstance(value, dict):
        raise CodexOAuthInventoryError(
            "Codex OAuth inventory routing must contain a JSON object."
        )
    _validate_exact_fields(
        value,
        expected=_ROUTING_FIELDS,
        subject="Codex OAuth inventory routing",
    )
    credential_affinity = value.get("credential_affinity", "pinned")
    if credential_affinity not in {"pinned", "interchangeable"}:
        raise CodexOAuthInventoryError(
            "Codex OAuth routing credential_affinity is unsupported."
        )
    strategy = value.get("strategy", "priority")
    if strategy not in {"priority", "dual_quota_balance"}:
        raise CodexOAuthInventoryError(
            "Codex OAuth routing strategy is unsupported."
        )
    within_band_strategy = value.get("within_band_strategy", "priority")
    if within_band_strategy not in {"priority", "weighted_round_robin"}:
        raise CodexOAuthInventoryError(
            "Codex OAuth routing within_band_strategy is unsupported."
        )
    raw_band = value.get("balance_band_percentage_points", 10.0)
    if (
        not isinstance(raw_band, (int, float))
        or isinstance(raw_band, bool)
        or not math.isfinite(float(raw_band))
        or not 0 < float(raw_band) <= 100
    ):
        raise CodexOAuthInventoryError(
            "Codex OAuth routing balance_band_percentage_points must be "
            "greater than 0 and at most 100."
        )
    return CodexOAuthRoutingPolicy(
        credential_affinity=credential_affinity,
        strategy=strategy,
        balance_band_percentage_points=float(raw_band),
        within_band_strategy=within_band_strategy,
    )


def load_codex_oauth_credential(
    record: CodexOAuthCredentialRecord,
) -> CodexOAuthCredentialSnapshot:
    """Read and validate exactly one selected credential without writing."""
    if not record.enabled:
        raise CodexOAuthCredentialError(
            f"Codex OAuth account '{record.label}' is disabled."
        )
    payload = _read_private_auth_payload(record)
    token_data = get_codex_oauth_token_data(payload, label=record.label)
    access_token = _clean_string(token_data.get("access_token"))
    if access_token is None:
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' is missing access_token."
        )
    account_id, account_hash = validate_codex_oauth_account_identity(
        record,
        token_data,
    )
    return CodexOAuthCredentialSnapshot(
        record=record,
        account_hash=account_hash,
        expires_at=get_codex_oauth_token_expiry(token_data),
        access_token=access_token,
        account_id=account_id,
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
    if isinstance(token_data, dict):
        return token_data
    raise CodexOAuthCredentialError(
        f"Codex OAuth credential '{label}' has an invalid tokens object."
    )


def validate_codex_oauth_account_identity(
    record: CodexOAuthCredentialRecord,
    token_data: Mapping[str, Any],
) -> tuple[str, str]:
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
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' has no account identity."
        )
    if len(unique_account_ids) != 1:
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' contains mismatched "
            "account identities."
        )

    account_id = unique_account_ids.pop()
    account_hash = codex_oauth_account_identity_hash(account_id)
    if account_hash != record.expected_account_hash:
        raise CodexOAuthIdentityMismatchError(
            label=record.label,
            expected_account_hash=record.expected_account_hash,
            actual_account_hash=account_hash,
        )
    return account_id, account_hash


def get_codex_oauth_token_expiry(
    token_data: Mapping[str, Any],
) -> Optional[float]:
    expires_at = token_data.get("expires_at")
    if isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool):
        return float(expires_at)
    if isinstance(expires_at, str) and expires_at.strip():
        try:
            return float(expires_at.strip())
        except ValueError:
            pass

    access_token = _clean_string(token_data.get("access_token"))
    if access_token is None:
        return None
    exp = _decode_jwt_claims_without_validation(access_token).get("exp")
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        return float(exp)
    return None


def _parse_account_record(
    payload: Any,
    *,
    declaration_order: int,
) -> CodexOAuthCredentialRecord:
    if not isinstance(payload, dict):
        raise CodexOAuthInventoryError(
            "Each Codex OAuth inventory account must be a JSON object."
        )
    _validate_exact_fields(
        payload,
        expected=_ACCOUNT_FIELDS,
        subject="Codex OAuth inventory account",
    )

    label_value = payload.get("label")
    if not isinstance(label_value, str) or _SAFE_LABEL_RE.fullmatch(label_value) is None:
        raise CodexOAuthInventoryError(
            "Codex OAuth account labels must be lowercase safe identifiers."
        )
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
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' must use distinct auth and lock paths."
        )

    priority = payload.get("priority")
    if (
        not isinstance(priority, int)
        or isinstance(priority, bool)
        or priority < 0
    ):
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' priority must be a non-negative integer."
        )

    weight_value = payload.get("weight")
    if (
        not isinstance(weight_value, (int, float))
        or isinstance(weight_value, bool)
        or not math.isfinite(float(weight_value))
        or float(weight_value) <= 0
    ):
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' weight must be a positive number."
        )
    weight = float(weight_value)

    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' enabled must be a boolean."
        )

    models_value = payload.get("models")
    if not isinstance(models_value, list) or not models_value:
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' models must be a non-empty array."
        )
    models: list[str] = []
    for model in models_value:
        if not isinstance(model, str) or not model.strip():
            raise CodexOAuthInventoryError(
                f"Codex OAuth account '{label}' models must contain "
                "non-empty strings."
            )
        cleaned_model = model.strip()
        if any(character in cleaned_model for character in "?["):
            raise CodexOAuthInventoryError(
                f"Codex OAuth account '{label}' model eligibility does not "
                "support glob patterns."
            )
        if "*" in cleaned_model and cleaned_model != "*":
            raise CodexOAuthInventoryError(
                f"Codex OAuth account '{label}' model eligibility supports "
                "only exact names or the explicit '*' value."
            )
        if cleaned_model in models:
            raise CodexOAuthInventoryError(
                f"Codex OAuth account '{label}' has duplicate model eligibility."
            )
        models.append(cleaned_model)
    if "*" in models and len(models) != 1:
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' cannot combine '*' with exact models."
        )

    expected_account_hash = payload.get("expected_account_hash")
    if (
        not isinstance(expected_account_hash, str)
        or _ACCOUNT_HASH_RE.fullmatch(expected_account_hash) is None
    ):
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
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' {field_name} must be a non-empty path."
        )
    raw_path = value.strip()
    if any(character in raw_path for character in _PATH_GLOB_CHARACTERS):
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' {field_name} must name one explicit path."
        )
    expanded_path = Path(raw_path).expanduser()
    if not expanded_path.is_absolute():
        raise CodexOAuthInventoryError(
            f"Codex OAuth account '{label}' {field_name} must be absolute "
            "after user expansion."
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
        raise CodexOAuthInventoryError(
            f"{subject} is missing required fields: {sorted(missing)}."
        )
    if unknown:
        raise CodexOAuthInventoryError(
            f"{subject} contains unknown fields: {sorted(unknown)}."
        )


def _read_private_auth_payload(
    record: CodexOAuthCredentialRecord,
) -> Mapping[str, Any]:
    file_descriptor = _open_private_auth_file(record)
    try:
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise CodexOAuthCredentialError(
                f"Codex OAuth credential '{record.label}' is not a regular file."
            )
        if stat.S_IMODE(file_stat.st_mode) != CODEX_OAUTH_AUTH_FILE_MODE:
            raise CodexOAuthCredentialError(
                f"Codex OAuth credential '{record.label}' has invalid permissions; "
                "expected mode 0600."
            )
        raw_bytes = _read_bounded_bytes(file_descriptor, label=record.label)
    finally:
        os.close(file_descriptor)

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' is malformed."
        ) from None
    if not isinstance(payload, dict):
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' must contain a JSON object."
        )
    return payload


def _open_private_auth_file(record: CodexOAuthCredentialRecord) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CodexOAuthCredentialError(
            "Secure Codex OAuth credential reads are unsupported on this platform."
        )
    flags = os.O_RDONLY | nofollow
    cloexec = getattr(os, "O_CLOEXEC", None)
    if cloexec is not None:
        flags |= cloexec
    try:
        return os.open(record.auth_path, flags)
    except FileNotFoundError:
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' is missing."
        ) from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            reason = "is a symlink or unsafe path"
        elif exc.errno in (errno.EACCES, errno.EPERM):
            reason = "is not readable"
        else:
            reason = "could not be opened"
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{record.label}' {reason}."
        ) from None


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
            raise CodexOAuthCredentialError(
                f"Codex OAuth credential '{label}' could not be read."
            ) from None
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
    if bytes_read > CODEX_OAUTH_AUTH_FILE_MAX_BYTES:
        raise CodexOAuthCredentialError(
            f"Codex OAuth credential '{label}' is too large."
        )
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
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        )
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _clean_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


__all__ = [
    "CODEX_OAUTH_ACCOUNT_HASH_LENGTH",
    "CODEX_OAUTH_AUTH_FILE_MAX_BYTES",
    "CODEX_OAUTH_AUTH_FILE_MODE",
    "CODEX_OAUTH_INVENTORY_ENV",
    "CODEX_OAUTH_INVENTORY_SCHEMA_VERSION",
    "CodexOAuthCredentialError",
    "CodexOAuthCredentialRecord",
    "CodexOAuthCredentialSnapshot",
    "CodexOAuthIdentityMismatchError",
    "CodexOAuthInventory",
    "CodexOAuthInventoryError",
    "CodexOAuthRoutingPolicy",
    "codex_oauth_account_identity_hash",
    "get_codex_oauth_token_data",
    "get_codex_oauth_token_expiry",
    "load_codex_oauth_credential",
    "load_codex_oauth_inventory",
    "parse_codex_oauth_inventory",
    "validate_codex_oauth_account_identity",
]
