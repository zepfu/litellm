import json

import pytest

from litellm.integrations.aawm_review_decisions import (
    AAWM_REVIEW_MAX_CLASSIFICATION_CHARS,
    AAWM_REVIEW_MAX_CONTENT_ITEMS,
    AAWM_REVIEW_MAX_OUTPUT_ITEMS,
    AAWM_REVIEW_MAX_OUTPUT_TEXT_CHARS,
    AAWM_REVIEW_MAX_RATIONALE_CHARS,
    AawmReviewDecisionParseFailure,
    parse_aawm_review_decision,
)


def _review_response(
    output_text: str,
    *,
    status: str = "completed",
    role: str = "assistant",
) -> dict:
    return {
        "status": status,
        "output": [
            {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                    }
                ],
            }
        ],
    }


def test_parse_aawm_review_decision_accepts_exact_sanitized_decision() -> None:
    raw_secret = "sk-secret123"
    response = _review_response(
        json.dumps(
            {
                "outcome": "deny",
                "rationale": f"first\nsecond \x1b[31mred\x1b[0m {raw_secret}",
            }
        )
    )

    result = parse_aawm_review_decision(response)

    assert result.ok
    assert result.decision is not None
    assert result.decision.outcome == "deny"
    assert result.decision.rationale == "first second red [REDACTED]"
    assert raw_secret not in repr(result)
    assert not hasattr(result, "response")


def test_parse_aawm_review_decision_accepts_real_full_reviewer_shape() -> None:
    result = parse_aawm_review_decision(
        _review_response(
            json.dumps(
                {
                    "risk_level": "medium",
                    "user_authorization": "high",
                    "outcome": "deny",
                    "rationale": "Unsafe mutation",
                }
            )
        )
    )

    assert result.ok
    assert result.decision is not None
    assert result.decision.outcome == "deny"
    assert result.decision.rationale == "Unsafe mutation"
    assert not hasattr(result.decision, "risk_level")
    assert not hasattr(result.decision, "user_authorization")


@pytest.mark.parametrize(
    ("response", "expected_failure"),
    [
        (
            _review_response('{"outcome":"allow"}', status="in_progress"),
            AawmReviewDecisionParseFailure.RESPONSE_NOT_COMPLETED,
        ),
        (
            _review_response('{"outcome":"allow"'),
            AawmReviewDecisionParseFailure.JSON_MALFORMED,
        ),
        (
            _review_response('{"outcome":"allow"} trailing'),
            AawmReviewDecisionParseFailure.JSON_TRAILING_CONTENT,
        ),
        (
            _review_response('{"outcome":"allow","outcome":"deny"}'),
            AawmReviewDecisionParseFailure.DUPLICATE_OUTCOME,
        ),
        (
            _review_response('{"outcome":"ALLOW"}'),
            AawmReviewDecisionParseFailure.OUTCOME_INVALID,
        ),
        (
            _review_response('{"outcome":"allow","extra":true}'),
            AawmReviewDecisionParseFailure.UNEXPECTED_FIELD,
        ),
        (
            _review_response(
                '{"risk_level":"severe","user_authorization":"high",'
                '"outcome":"allow"}'
            ),
            AawmReviewDecisionParseFailure.RISK_LEVEL_INVALID,
        ),
        (
            _review_response(
                '{"risk_level":"low","user_authorization":"explicit",'
                '"outcome":"allow"}'
            ),
            AawmReviewDecisionParseFailure.USER_AUTHORIZATION_INVALID,
        ),
        (
            _review_response(
                json.dumps(
                    {
                        "risk_level": "r"
                        * (AAWM_REVIEW_MAX_CLASSIFICATION_CHARS + 1),
                        "outcome": "allow",
                    }
                )
            ),
            AawmReviewDecisionParseFailure.RISK_LEVEL_INVALID,
        ),
        (
            _review_response('{"outcome":"allow"}', role="user"),
            AawmReviewDecisionParseFailure.NON_ASSISTANT_OUTPUT_TEXT,
        ),
    ],
)
def test_parse_aawm_review_decision_rejects_non_exact_forms(
    response: dict,
    expected_failure: AawmReviewDecisionParseFailure,
) -> None:
    result = parse_aawm_review_decision(response)

    assert not result.ok
    assert result.decision is None
    assert result.failure == expected_failure


def test_parse_aawm_review_decision_rejects_multiple_output_texts() -> None:
    response = _review_response('{"outcome":"allow"}')
    response["output"][0]["content"].append(
        {"type": "output_text", "text": '{"outcome":"deny"}'}
    )

    result = parse_aawm_review_decision(response)

    assert result.failure == AawmReviewDecisionParseFailure.OUTPUT_TEXT_MULTIPLE


@pytest.mark.parametrize(
    ("response", "expected_failure"),
    [
        (
            {
                "status": "completed",
                "output": [{}] * (AAWM_REVIEW_MAX_OUTPUT_ITEMS + 1),
            },
            AawmReviewDecisionParseFailure.OUTPUT_TOO_LARGE,
        ),
        (
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{}] * (AAWM_REVIEW_MAX_CONTENT_ITEMS + 1),
                    }
                ],
            },
            AawmReviewDecisionParseFailure.CONTENT_TOO_LARGE,
        ),
        (
            _review_response("x" * (AAWM_REVIEW_MAX_OUTPUT_TEXT_CHARS + 1)),
            AawmReviewDecisionParseFailure.OUTPUT_TEXT_TOO_LARGE,
        ),
        (
            _review_response(
                json.dumps(
                    {
                        "outcome": "allow",
                        "rationale": "r"
                        * (AAWM_REVIEW_MAX_RATIONALE_CHARS + 1),
                    }
                )
            ),
            AawmReviewDecisionParseFailure.RATIONALE_TOO_LARGE,
        ),
    ],
)
def test_parse_aawm_review_decision_enforces_bounds(
    response: dict,
    expected_failure: AawmReviewDecisionParseFailure,
) -> None:
    result = parse_aawm_review_decision(response)

    assert result.decision is None
    assert result.failure == expected_failure


def test_parse_aawm_review_decision_accepts_bounded_rationale_without_one() -> None:
    with_rationale = parse_aawm_review_decision(
        _review_response(
            json.dumps(
                {
                    "outcome": "allow",
                    "rationale": "r" * AAWM_REVIEW_MAX_RATIONALE_CHARS,
                }
            )
        )
    )
    without_rationale = parse_aawm_review_decision(
        _review_response('{"outcome":"deny"}')
    )

    assert with_rationale.ok
    assert with_rationale.decision is not None
    assert len(with_rationale.decision.rationale or "") == AAWM_REVIEW_MAX_RATIONALE_CHARS
    assert without_rationale.ok
    assert without_rationale.decision is not None
    assert without_rationale.decision.rationale is None
