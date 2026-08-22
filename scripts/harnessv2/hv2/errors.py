"""Harness v2 error types."""

from __future__ import annotations


class HarnessError(Exception):
    """Operator-facing failure. Default exit 2 (usage / safety)."""

    exit_code = 2


class ProtectedTargetError(HarnessError):
    """Refused aawm-litellm, litellm-dev, or host ports 4000/4001."""

    exit_code = 2


class ConfigError(HarnessError):
    """Invalid YAML/JSON or unresolved placeholder."""

    exit_code = 2


class PlanError(HarnessError):
    """CLI + config cannot form a runnable plan."""

    exit_code = 2


class InstanceError(HarnessError):
    """Docker inspect / published-port resolution failed."""

    exit_code = 2


class DriverError(HarnessError):
    """TUI driver cannot run (missing binary, forbid_flags, not implemented)."""

    exit_code = 2


class CheckError(HarnessError):
    """A named check type failed."""

    exit_code = 1


# Alias used by instance/CLI call sites.
ProtectedInstanceError = ProtectedTargetError
