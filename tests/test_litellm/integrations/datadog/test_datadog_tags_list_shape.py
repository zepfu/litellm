"""
RR-012: get_datadog_tags must return List[str] of individual tags.
"""

import os
from unittest.mock import patch

from litellm.integrations.datadog.datadog_handler import get_datadog_tags
from litellm.types.utils import StandardLoggingMetadata, StandardLoggingPayload


def test_get_datadog_tags_returns_list_of_individual_tags():
    with patch.dict(
        os.environ,
        {
            "DD_ENV": "test-env",
            "DD_SERVICE": "test-service",
            "DD_VERSION": "1.0.0",
            "HOSTNAME": "test-host",
            "POD_NAME": "test-pod",
        },
    ):
        tags = get_datadog_tags()
        assert isinstance(tags, list)
        assert all(isinstance(t, str) for t in tags)
        assert "env:test-env" in tags
        assert "service:test-service" in tags
        # Must NOT be a single comma-joined element
        assert not any("," in t for t in tags)


def test_get_datadog_tags_includes_request_and_team_tags_as_list_items():
    with patch.dict(
        os.environ,
        {
            "DD_ENV": "test-env",
            "DD_SERVICE": "test-service",
            "DD_VERSION": "1.0.0",
            "HOSTNAME": "test-host",
            "POD_NAME": "test-pod",
        },
    ):
        payload = StandardLoggingPayload(
            metadata=StandardLoggingMetadata(user_api_key_team_alias="t1"),
            request_tags=["alpha", "beta"],
        )
        tags = get_datadog_tags(payload)
        assert "request_tag:alpha" in tags
        assert "request_tag:beta" in tags
        assert "team:t1" in tags
