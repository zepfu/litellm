"""Failure classification for Z.AI Coding Plan business codes."""

from __future__ import annotations

import httpx
import pytest

from litellm.llms.zai_coding_plan import (
    ZAICodingPlanFailureKind,
    classify_zai_coding_plan_failure,
    classify_zai_coding_plan_http_failure,
)


@pytest.mark.parametrize(
    ("error_code", "expected_kind", "expected_reason"),
    (
        (1000, ZAICodingPlanFailureKind.AUTH, "credentials_rejected"),
        (1001, ZAICodingPlanFailureKind.AUTH, "missing_authorization"),
        (
            1113,
            ZAICodingPlanFailureKind.ROUTING,
            "wrong_base_or_non_coding_key",
        ),
        (1211, ZAICodingPlanFailureKind.VALIDATION, "unknown_model"),
        (1302, ZAICodingPlanFailureKind.RATE, "request_rate_limit"),
        (1308, ZAICodingPlanFailureKind.QUOTA, "time_window_exhausted"),
        (1309, ZAICodingPlanFailureKind.QUOTA, "subscription_expired"),
        (1310, ZAICodingPlanFailureKind.QUOTA, "plan_window_exhausted"),
        (1311, ZAICodingPlanFailureKind.VALIDATION, "model_not_in_subscription"),
        (1313, ZAICodingPlanFailureKind.QUOTA, "fair_use_restriction"),
        (1316, ZAICodingPlanFailureKind.QUOTA, "five_hour_window_exhausted"),
        (1317, ZAICodingPlanFailureKind.QUOTA, "seven_day_window_exhausted"),
    ),
)
def test_should_map_coding_plan_business_codes_onto_shared_classes(
    error_code: int,
    expected_kind: ZAICodingPlanFailureKind,
    expected_reason: str,
) -> None:
    failure = classify_zai_coding_plan_failure(
        status_code=429,
        error_code=error_code,
        message="provider detail",
        upstream_id="zai_coding_plan/glm-5.3",
    )

    assert failure.kind == expected_kind
    assert failure.reset_reason == expected_reason
    assert failure.upstream_id == "glm-5.3"
    assert "recharge" not in failure.reset_reason
    assert "ordinary" not in failure.reset_reason


def test_should_treat_1113_on_coding_requests_as_routing_defect_not_ordinary_balance() -> None:
    failure = classify_zai_coding_plan_failure(
        status_code=429,
        error_code="1113",
        message="Insufficient balance",
        upstream_id="glm-5.3",
    )

    assert failure.kind == ZAICodingPlanFailureKind.ROUTING
    assert failure.reset_reason == "wrong_base_or_non_coding_key"
    safe = failure.to_safe_metadata()
    assert "recharge" not in str(safe).lower()
    assert "ordinary" not in str(safe).lower()
    assert "Insufficient balance" not in str(safe)


def test_should_redact_untrusted_http_details_from_safe_metadata() -> None:
    secret = "Bearer live-secret-value"
    response = httpx.Response(
        status_code=429,
        headers={
            "Authorization": secret,
            "X-Msh-Device-Id": "identifying-device-value",
        },
        json={
            "error": {
                "code": 1113,
                "message": f"{secret}; recharge ordinary balance",
            }
        },
    )

    failure = classify_zai_coding_plan_http_failure(response, upstream_id="glm-5.3")
    safe_metadata = failure.to_safe_metadata()

    assert safe_metadata["kind"] == "routing"
    assert safe_metadata["reset_reason"] == "wrong_base_or_non_coding_key"
    assert secret not in repr(failure)
    assert "identifying-device-value" not in repr(failure)
    assert "recharge ordinary balance" not in str(safe_metadata)
