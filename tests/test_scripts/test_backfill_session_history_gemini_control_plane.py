import os
import subprocess
import sys
from pathlib import Path

import scripts.backfill_session_history as backfill_session_history


def test_should_default_backfill_repairs_to_local_model_cost_map() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env.pop("LITELLM_MODEL_COST_MAP_URL", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import scripts.backfill_session_history; "
                "import litellm; "
                "print(litellm.model_cost['gpt-5.5']['output_cost_per_token'])"
            ),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "2.5e-05"


def test_should_price_chatgpt_rows_without_auth_provider(monkeypatch) -> None:
    import litellm

    calls = []

    def fake_cost_per_token(**kwargs):
        calls.append(kwargs)
        return 0.1, 0.2

    monkeypatch.setattr(litellm, "cost_per_token", fake_cost_per_token)

    cost = backfill_session_history._recalculate_session_history_response_cost(
        {
            "provider": "chatgpt",
            "model": "chatgpt/gpt-5.5",
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 0,
        }
    )

    assert cost == 0.1 + 0.2
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["custom_llm_provider"] == "openai"


def test_should_recalculate_anthropic_cache_creation_as_one_hour() -> None:
    cost = backfill_session_history._recalculate_session_history_response_cost(
        {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "input_tokens": 9143,
            "output_tokens": 7,
            "cache_read_input_tokens": 8787,
            "cache_creation_input_tokens": 307,
        }
    )

    assert round(cost or 0, 10) == round(0.0078835, 10)


def test_should_recalculate_rows_where_cache_tokens_exceed_input_tokens() -> None:
    cost = backfill_session_history._recalculate_session_history_response_cost(
        {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "input_tokens": 376,
            "output_tokens": 152,
            "cache_read_input_tokens": 478666,
            "cache_creation_input_tokens": 375,
        }
    )

    assert cost is not None
    assert cost > 0


def test_should_extract_gemini_control_plane_methods_from_routes() -> None:
    for method in (
        "loadCodeAssist",
        "listExperiments",
        "retrieveUserQuota",
        "fetchAdminControls",
    ):
        assert (
            backfill_session_history._extract_gemini_control_plane_method(
                f"/gemini/v1internal:{method}"
            )
            == method
        )


def test_should_match_gemini_control_plane_row_with_method_route() -> None:
    is_match, method = (
        backfill_session_history._is_gemini_control_plane_session_history_row(
            {
                "provider": "gemini",
                "model": "unknown",
                "call_type": "/gemini/v1internal:loadCodeAssist",
                "metadata": {"client_name": "gemini-cli"},
            }
        )
    )

    assert is_match is True
    assert method == "loadCodeAssist"


def test_should_match_gemini_control_plane_row_with_json_metadata_route() -> None:
    is_match, method = (
        backfill_session_history._is_gemini_control_plane_session_history_row(
            {
                "provider": None,
                "model": "unknown",
                "call_type": None,
                "metadata": (
                    '{"custom_llm_provider":"gemini",'
                    '"user_api_key_request_route":"/v1internal:listExperiments"}'
                ),
            }
        )
    )

    assert is_match is True
    assert method == "listExperiments"


def test_should_not_match_normal_gemini_model_row() -> None:
    is_match, method = (
        backfill_session_history._is_gemini_control_plane_session_history_row(
            {
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "call_type": "/gemini/v1internal:generateContent",
                "metadata": {
                    "custom_llm_provider": "gemini",
                    "request_tags": ["route:gemini_generate_content"],
                },
            }
        )
    )

    assert is_match is False
    assert method is None


def test_should_not_match_unknown_zero_token_generate_row_without_method() -> None:
    is_match, method = (
        backfill_session_history._is_gemini_control_plane_session_history_row(
            {
                "provider": "gemini",
                "model": "unknown",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "client_name": "gemini-cli",
                "metadata": {
                    "trace_name": "native_gemini_passthrough_stream_generate_content",
                    "request_tags": ["route:gemini_stream_generate_content"],
                },
            }
        )
    )

    assert is_match is False
    assert method is None


def test_should_not_match_non_gemini_row_with_unrelated_metadata_mention() -> None:
    is_match, method = (
        backfill_session_history._is_gemini_control_plane_session_history_row(
            {
                "provider": "openai",
                "model": "gpt-5",
                "call_type": "/v1/responses",
                "metadata": {"note": "previously saw loadCodeAssist during startup"},
            }
        )
    )

    assert is_match is False
    assert method is None
