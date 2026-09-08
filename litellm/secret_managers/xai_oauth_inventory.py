"""Explicit managed xAI OAuth account inventory.

The inventory is server configuration only. It names exact credential
file/scope records and pins the stable nonsecret identity expected from each
record; request paths never discover credentials by scanning directories.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from litellm.secret_managers.main import get_secret_str
from litellm.secret_managers.xai_oauth_credentials import (
    resolve_xai_oauth_auth_path,
    resolve_xai_oauth_scope,
)

XAI_OAUTH_INVENTORY_ENV = "LITELLM_XAI_OAUTH_INVENTORY"
XAI_OAUTH_INVENTORY_SCHEMA_VERSION = 1
XAI_OAUTH_INVENTORY_MAX_ACCOUNTS = 32

_SAFE_LABEL_RE = re.compile(r"\A[a-z][a-z0-9._-]{0,63}\Z")
_ACCOUNT_IDENTITY_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_PATH_GLOB_CHARACTERS = frozenset("*?[")
_ROOT_FIELDS = frozenset({"schema_version", "accounts"})
_ACCOUNT_FIELDS = frozenset(
    {
        "label",
        "auth_path",
        "scope",
        "priority",
        "enabled",
        "expected_account_identity",
    }
)


class XaiOAuthInventoryError(ValueError):
    """Raised when managed xAI OAuth inventory is absent or invalid."""


class XaiOAuthIdentityMismatchError(XaiOAuthInventoryError):
    """Raised when a configured account does not match its identity pin."""


@dataclass(frozen=True)
class XaiOAuthAccountRecord:
    """One configured credential file and exact scope record."""

    label: str
    auth_path: Path = field(repr=False)
    scope: str = field(repr=False)
    priority: int
    enabled: bool
    expected_account_identity: Optional[str]
    declaration_order: int = field(repr=False, compare=False)
    legacy: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True)
class XaiOAuthInventory:
    """Immutable ordered managed xAI OAuth records."""

    records: tuple[XaiOAuthAccountRecord, ...]

    def ordered_records(
        self,
        *,
        enabled_only: bool = False,
    ) -> tuple[XaiOAuthAccountRecord, ...]:
        records = self.records
        if enabled_only:
            records = tuple(record for record in records if record.enabled)
        return tuple(
            sorted(
                records,
                key=lambda record: (record.priority, record.declaration_order),
            )
        )

    def select_record(self, *, label: Optional[str] = None) -> XaiOAuthAccountRecord:
        if label is not None:
            record = next(
                (candidate for candidate in self.records if candidate.label == label),
                None,
            )
            if record is None:
                raise XaiOAuthInventoryError(
                    "Selected xAI OAuth account label is not configured."
                )
            if not record.enabled:
                raise XaiOAuthInventoryError(
                    f"xAI OAuth account '{record.label}' is disabled."
                )
            return record

        records = self.ordered_records(enabled_only=True)
        if not records:
            raise XaiOAuthInventoryError(
                "No enabled xAI OAuth account is eligible for selection."
            )
        return records[0]


def load_xai_oauth_inventory(
    raw_inventory: Optional[str] = None,
) -> XaiOAuthInventory:
    """Load the strict versioned inventory from an argument or environment."""

    raw_value = (
        raw_inventory
        if raw_inventory is not None
        else os.getenv(XAI_OAUTH_INVENTORY_ENV)
    )
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise XaiOAuthInventoryError(
            f"xAI OAuth inventory is not configured in {XAI_OAUTH_INVENTORY_ENV}."
        )
    try:
        payload = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        raise XaiOAuthInventoryError(
            "xAI OAuth inventory is not valid JSON."
        ) from None
    return parse_xai_oauth_inventory(payload)


def parse_xai_oauth_inventory(payload: Any) -> XaiOAuthInventory:
    """Validate a versioned inventory and return immutable account records."""

    if not isinstance(payload, dict):
        raise XaiOAuthInventoryError(
            "xAI OAuth inventory must contain a JSON object."
        )
    _validate_exact_fields(
        payload,
        expected=_ROOT_FIELDS,
        subject="xAI OAuth inventory",
    )
    if payload.get("schema_version") != XAI_OAUTH_INVENTORY_SCHEMA_VERSION:
        raise XaiOAuthInventoryError(
            "xAI OAuth inventory has an unsupported schema_version."
        )
    accounts = payload.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        raise XaiOAuthInventoryError(
            "xAI OAuth inventory accounts must be a non-empty array."
        )
    if len(accounts) > XAI_OAUTH_INVENTORY_MAX_ACCOUNTS:
        raise XaiOAuthInventoryError(
            "xAI OAuth inventory exceeds the maximum configured accounts."
        )

    records: list[XaiOAuthAccountRecord] = []
    seen_labels: set[str] = set()
    seen_account_identities: dict[str, str] = {}
    seen_bindings: dict[tuple[str, str], str] = {}
    for declaration_order, account in enumerate(accounts):
        record = _parse_account_record(
            account,
            declaration_order=declaration_order,
        )
        if record.label in seen_labels:
            raise XaiOAuthInventoryError(
                f"Duplicate xAI OAuth account label '{record.label}'."
            )
        seen_labels.add(record.label)

        binding_key = (os.path.normcase(os.fspath(record.auth_path)), record.scope)
        prior_binding = seen_bindings.get(binding_key)
        if prior_binding is not None:
            raise XaiOAuthInventoryError(
                f"xAI OAuth accounts '{prior_binding}' and '{record.label}' "
                "reuse one auth-path/scope record."
            )
        seen_bindings[binding_key] = record.label

        assert record.expected_account_identity is not None
        prior_identity = seen_account_identities.get(
            record.expected_account_identity
        )
        if prior_identity is not None:
            raise XaiOAuthInventoryError(
                f"xAI OAuth accounts '{prior_identity}' and '{record.label}' "
                "pin the same account identity."
            )
        seen_account_identities[record.expected_account_identity] = record.label
        records.append(record)

    return XaiOAuthInventory(records=tuple(records))


def legacy_xai_oauth_account_record() -> XaiOAuthAccountRecord:
    """Return the existing single-file configuration as one server-owned lane."""

    path_resolution = resolve_xai_oauth_auth_path(value_getter=get_secret_str)
    scope_resolution = resolve_xai_oauth_scope(value_getter=get_secret_str)
    return XaiOAuthAccountRecord(
        label="default",
        auth_path=Path(
            os.path.abspath(os.fspath(path_resolution.path.expanduser()))
        ),
        scope=scope_resolution.scope,
        priority=0,
        enabled=True,
        expected_account_identity=None,
        declaration_order=0,
        legacy=True,
    )


def xai_oauth_record_identity(record: XaiOAuthAccountRecord) -> str:
    """Return one opaque stable identity for the configured account record."""

    payload = {
        "account_identity": record.expected_account_identity,
        "auth_path": os.path.normcase(os.fspath(record.auth_path)),
        "label": record.label,
        "scope": record.scope,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def xai_oauth_scope_identity(record: XaiOAuthAccountRecord) -> str:
    """Return a separate opaque exact-scope identity for observation metadata."""

    digest = hashlib.sha256(record.scope.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _parse_account_record(
    payload: Any,
    *,
    declaration_order: int,
) -> XaiOAuthAccountRecord:
    if not isinstance(payload, dict):
        raise XaiOAuthInventoryError(
            "Each xAI OAuth inventory account must be a JSON object."
        )
    _validate_exact_fields(
        payload,
        expected=_ACCOUNT_FIELDS,
        subject="xAI OAuth inventory account",
    )

    label = payload.get("label")
    if not isinstance(label, str) or _SAFE_LABEL_RE.fullmatch(label) is None:
        raise XaiOAuthInventoryError(
            "xAI OAuth account labels must be lowercase safe identifiers."
        )
    auth_path = _parse_explicit_path(payload.get("auth_path"), label=label)

    scope = payload.get("scope")
    if (
        not isinstance(scope, str)
        or not scope.strip()
        or len(scope.strip()) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in scope)
    ):
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' scope must be one bounded exact record selector."
        )
    scope = scope.strip()

    priority = payload.get("priority")
    if (
        not isinstance(priority, int)
        or isinstance(priority, bool)
        or priority < 0
    ):
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' priority must be a non-negative integer."
        )

    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' enabled must be a boolean."
        )

    expected_account_identity = payload.get("expected_account_identity")
    if (
        not isinstance(expected_account_identity, str)
        or _ACCOUNT_IDENTITY_RE.fullmatch(expected_account_identity) is None
    ):
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' expected_account_identity must be a "
            "sha256-prefixed stable nonsecret identity."
        )

    return XaiOAuthAccountRecord(
        label=label,
        auth_path=auth_path,
        scope=scope,
        priority=priority,
        enabled=enabled,
        expected_account_identity=expected_account_identity,
        declaration_order=declaration_order,
    )


def _parse_explicit_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' auth_path must be a non-empty path."
        )
    raw_path = value.strip()
    if any(character in raw_path for character in _PATH_GLOB_CHARACTERS):
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' auth_path must name one explicit path."
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise XaiOAuthInventoryError(
            f"xAI OAuth account '{label}' auth_path must be absolute after user expansion."
        )
    return Path(os.path.abspath(os.fspath(path)))


def _validate_exact_fields(
    payload: dict[str, Any],
    *,
    expected: frozenset[str],
    subject: str,
) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise XaiOAuthInventoryError(
            f"{subject} is missing required fields: {sorted(missing)}."
        )
    if unknown:
        raise XaiOAuthInventoryError(
            f"{subject} contains unknown fields: {sorted(unknown)}."
        )


__all__ = [
    "XAI_OAUTH_INVENTORY_ENV",
    "XAI_OAUTH_INVENTORY_MAX_ACCOUNTS",
    "XAI_OAUTH_INVENTORY_SCHEMA_VERSION",
    "XaiOAuthAccountRecord",
    "XaiOAuthIdentityMismatchError",
    "XaiOAuthInventory",
    "XaiOAuthInventoryError",
    "legacy_xai_oauth_account_record",
    "load_xai_oauth_inventory",
    "parse_xai_oauth_inventory",
    "xai_oauth_record_identity",
    "xai_oauth_scope_identity",
]
