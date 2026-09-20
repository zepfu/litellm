"""OpenCode Go local preflight: no-call outcomes before transport commitment.

Local target, credential, header, and egress-validation failures must not look
like provider attempts. A provider-returned 401 stays an attempted call.
"""

from __future__ import annotations

from typing import Never, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException
from litellm.secret_managers.credential_error_sanitizer import (
    sanitize_credential_error_message,
)


OPENCODE_GO_PREFLIGHT_PHASE = "candidate_preflight"
OPENCODE_GO_PREFLIGHT_INELIGIBILITY_REASON = "preflight_skipped"
OPENCODE_GO_PREFLIGHT_ERROR_CODE = "aawm_codex_auto_agent_candidate_ineligible"
OPENCODE_GO_PREFLIGHT_REASONS = frozenset(
    {
        "missing_credential",
        "unreadable_credential",
        "malformed_credential",
        "wrong_type_credential",
        "wrong_entry_credential",
        "invalid_target",
        "invalid_headers",
        "egress_validation",
    }
)
OPENCODE_GO_PREFLIGHT_CALL_MODES = frozenset({"alias", "direct"})
_PROVIDER_RETURNED_AUTH_STATUS_CODES = frozenset({401, 403})


def _normalized_exception_text(exc: BaseException) -> str:
    return " ".join(str(exc).lower().split())


def is_opencode_go_provider_returned_auth_failure(exc: BaseException) -> bool:
    """True when the failure is a real provider authentication response."""

    if getattr(exc, "attempted_provider_call", None) is True:
        return True
    if getattr(exc, "_aawm_provider_returned", False) is True:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) in _PROVIDER_RETURNED_AUTH_STATUS_CODES
    except (TypeError, ValueError):
        return False


def classify_opencode_go_credential_preflight_reason(exc: BaseException) -> str:
    """Map local credential defects onto one bounded preflight reason."""

    text = _normalized_exception_text(exc)
    if (
        isinstance(exc, FileNotFoundError)
        or "not found" in text
        or "missing or not a regular file" in text
    ):
        return "missing_credential"
    if "not readable" in text:
        return "unreadable_credential"
    if "valid json" in text:
        return "malformed_credential"
    if "api-key auth type" in text or (
        "auth type" in text and "api-key" in text
    ):
        return "wrong_type_credential"
    if "api-key auth" in text:
        return "wrong_entry_credential"
    if isinstance(exc, (ValueError, OSError, TypeError)):
        return "wrong_entry_credential"
    return "missing_credential"


class OpencodeGoPreflightError(ProxyException):
    """Local Go preflight ineligibility raised before any provider I/O."""

    def __init__(
        self,
        *,
        reason: str,
        call_mode: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        bounded_reason = (
            reason if reason in OPENCODE_GO_PREFLIGHT_REASONS else "missing_credential"
        )
        bounded_call_mode = call_mode if call_mode in OPENCODE_GO_PREFLIGHT_CALL_MODES else "direct"
        cause_text = (
            sanitize_credential_error_message(str(cause), limit=256)
            if cause is not None
            else ""
        )
        prefix = (
            "OpenCode Go auto-agent candidate failed local preflight"
            if bounded_call_mode == "alias"
            else "Direct OpenCode Go route failed local preflight"
        )
        message = f"{prefix} ({bounded_reason})."
        if cause_text:
            message = f"{message} {cause_text}"
        super().__init__(
            message=message,
            type="invalid_request_error",
            param="model",
            code=400,
        )
        setattr(self, "status_code", 400)
        setattr(self, "candidate_status", "ineligible")
        setattr(self, "ineligibility_reason", OPENCODE_GO_PREFLIGHT_INELIGIBILITY_REASON)
        setattr(self, "failure_phase", OPENCODE_GO_PREFLIGHT_PHASE)
        setattr(self, "attempted_provider_call", False)
        setattr(self, "preflight_reason", bounded_reason)
        setattr(self, "opencode_go_call_mode", bounded_call_mode)
        setattr(
            self,
            "detail",
            {
                "error": {
                    "message": message,
                    "code": OPENCODE_GO_PREFLIGHT_ERROR_CODE,
                },
                "failure_phase": OPENCODE_GO_PREFLIGHT_PHASE,
                "attempted_provider_call": False,
                "preflight_reason": bounded_reason,
                "opencode_go_call_mode": bounded_call_mode,
            },
        )


def raise_opencode_go_preflight(
    exc: BaseException,
    *,
    reason: str,
    use_alias_candidate_probe: bool,
) -> Never:
    """Raise a no-call Go preflight error, never relabeling provider auth."""

    if isinstance(exc, OpencodeGoPreflightError):
        raise exc
    if is_opencode_go_provider_returned_auth_failure(exc):
        raise exc
    call_mode = "alias" if use_alias_candidate_probe else "direct"
    verbose_proxy_logger.debug(
        "OpenCode Go local preflight rejected the candidate: reason=%s call_mode=%s",
        reason if reason in OPENCODE_GO_PREFLIGHT_REASONS else "missing_credential",
        call_mode,
    )
    raise OpencodeGoPreflightError(
        reason=reason,
        call_mode=call_mode,
        cause=exc,
    ) from exc
