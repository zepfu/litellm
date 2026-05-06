from typing import Any, Optional


GOOGLE_CODE_ASSIST_QUOTA_LOG_KEYS = {
    "modelId",
    "tokenType",
    "remainingFraction",
    "quotaId",
    "quotaName",
    "quotaPeriod",
    "period",
    "remainingRequests",
    "remaining_requests",
    "requestsRemaining",
    "usedRequests",
    "used_requests",
    "requestsUsed",
    "totalRequests",
    "total_requests",
    "requestLimit",
    "dailyLimit",
    "usagePercentage",
    "usedPercentage",
    "used_percentage",
    "resetsAt",
    "resets_at",
    "resetAt",
    "resetTime",
    "model",
}


def sanitize_google_code_assist_quota_for_logging(
    value: Any,
) -> Optional[dict[str, Any]]:
    if isinstance(value, list):
        sanitized_items = [
            item
            for item in (
                sanitize_google_code_assist_quota_for_logging(item) for item in value
            )
            if item
        ]
        return {"items": sanitized_items} if sanitized_items else None
    if not isinstance(value, dict):
        return None

    sanitized: dict[str, Any] = {}
    for key in GOOGLE_CODE_ASSIST_QUOTA_LOG_KEYS:
        if key not in value:
            continue
        field_value = value[key]
        if field_value is None or isinstance(field_value, (str, int, float, bool)):
            sanitized[key] = field_value
    for nested_key in (
        "buckets",
        "quota",
        "quotas",
        "userQuota",
        "modelQuota",
        "modelQuotas",
    ):
        nested_value = value.get(nested_key)
        nested_sanitized = sanitize_google_code_assist_quota_for_logging(nested_value)
        if nested_sanitized:
            sanitized[nested_key] = nested_sanitized
    if sanitized:
        sanitized["source"] = "google_retrieve_user_quota"
    return sanitized or None
