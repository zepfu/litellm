import httpx
import pytest

from litellm.llms.kimi_code import (
    KimiCodeFailureKind,
    KimiCodeFailureScope,
    KimiCodeManagedEndpoint,
    KimiCodeMetadataGate,
    classify_kimi_code_failure,
    classify_kimi_code_http_failure,
)


@pytest.mark.parametrize(
    (
        "status_code",
        "error_code",
        "message",
        "expected_kind",
        "expected_scope",
        "expected_gate",
        "expected_reason",
    ),
    [
        (
            401,
            "invalid_token",
            "The managed credential is expired.",
            KimiCodeFailureKind.REFRESH_REQUIRED_AUTH,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            KimiCodeMetadataGate.NONE,
            "refresh_required",
        ),
        (
            429,
            "insufficient_quota",
            "Managed quota exhausted.",
            KimiCodeFailureKind.QUOTA,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            KimiCodeMetadataGate.NONE,
            "quota_exhausted",
        ),
        (
            529,
            "server_busy",
            "Provider capacity is unavailable.",
            KimiCodeFailureKind.PROVIDER_CAPACITY,
            KimiCodeFailureScope.MANAGED_ACCOUNT,
            KimiCodeMetadataGate.NONE,
            "provider_capacity",
        ),
        (
            503,
            "timeout",
            "Please try again.",
            KimiCodeFailureKind.TRANSIENT,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.NONE,
            "transient_upstream_failure",
        ),
        (
            400,
            "model_not_found",
            "The selected model is unavailable.",
            KimiCodeFailureKind.UNSUPPORTED_MODEL,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.MODEL_ID,
            "unsupported_model",
        ),
        (
            400,
            "unsupported_reasoning_effort",
            "The requested reasoning effort is not supported.",
            KimiCodeFailureKind.UNSUPPORTED_EFFORT,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.THINK_EFFORT,
            "unsupported_effort",
        ),
        (
            400,
            "unsupported_capability",
            "The requested capability is not supported.",
            KimiCodeFailureKind.UNSUPPORTED_CAPABILITY,
            KimiCodeFailureScope.CANDIDATE,
            KimiCodeMetadataGate.CAPABILITY,
            "unsupported_capability",
        ),
    ],
)
def test_should_classify_managed_account_and_candidate_failures(
    status_code: int,
    error_code: str,
    message: str,
    expected_kind: KimiCodeFailureKind,
    expected_scope: KimiCodeFailureScope,
    expected_gate: KimiCodeMetadataGate,
    expected_reason: str,
):
    failure = classify_kimi_code_failure(
        status_code=status_code,
        error_code=error_code,
        message=message,
        upstream_id="kimi_code/k3",
    )

    assert failure.kind == expected_kind
    assert failure.scope == expected_scope
    assert failure.metadata_gate == expected_gate
    assert failure.reset_reason == expected_reason
    assert failure.upstream_id == "k3"
    assert failure.is_account_scoped is (expected_scope == KimiCodeFailureScope.MANAGED_ACCOUNT)
    assert failure.is_candidate_scoped is (expected_scope == KimiCodeFailureScope.CANDIDATE)


def test_should_classify_malformed_usages_as_telemetry_degradation():
    failure = classify_kimi_code_failure(
        status_code=422,
        error_code="malformed_response",
        upstream_id="k3",
        endpoint=KimiCodeManagedEndpoint.USAGES,
    )

    assert failure.kind == KimiCodeFailureKind.MALFORMED
    assert failure.scope == KimiCodeFailureScope.TELEMETRY
    assert not failure.is_account_scoped
    assert not failure.is_candidate_scoped
    assert failure.reset_reason == "malformed_provider_response"


def test_should_allowlist_safe_metadata_and_redact_untrusted_http_details():
    secret = "Bearer live-secret-value"
    response = httpx.Response(
        status_code=429,
        headers={
            "Authorization": secret,
            "X-Msh-Device-Id": "identifying-device-value",
            "X-Trace-Id": "kimi-trace_123",
        },
        json={
            "error": {
                "code": "insufficient_quota",
                "message": (f"{secret}; X-Msh-Device-Id=identifying-device-value; " "access_token=payload-secret"),
            }
        },
    )

    failure = classify_kimi_code_http_failure(response, upstream_id="k3")
    safe_metadata = failure.to_safe_metadata()

    assert safe_metadata == {
        "kind": "quota",
        "scope": "managed_account",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": 429,
        "trace_id": "kimi-trace_123",
        "reset_reason": "quota_exhausted",
    }
    assert secret not in repr(failure)
    assert "identifying-device-value" not in repr(failure)
    assert "payload-secret" not in repr(failure)
    assert "authorization" not in safe_metadata
    assert "x-msh-device-id" not in safe_metadata


def test_should_reject_unknown_upstream_ids_and_unsafe_trace_ids():
    failure = classify_kimi_code_failure(
        status_code=500,
        error_code="internal_error",
        upstream_id="untrusted-model-id",
        headers={"X-Trace-Id": "trace id with spaces"},
    )

    assert failure.kind == KimiCodeFailureKind.TRANSIENT
    assert failure.upstream_id is None
    assert failure.trace_id is None


def test_should_discard_malformed_usages_payload_content():
    response = httpx.Response(
        status_code=200,
        json={"access_token": "payload-secret", "refresh_token": "another-secret"},
    )

    failure = classify_kimi_code_http_failure(
        response,
        upstream_id="k3",
        endpoint=KimiCodeManagedEndpoint.USAGES,
    )

    assert failure.kind == KimiCodeFailureKind.MALFORMED
    assert failure.scope == KimiCodeFailureScope.TELEMETRY
    assert "payload-secret" not in repr(failure)
    assert "another-secret" not in repr(failure)
