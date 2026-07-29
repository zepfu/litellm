"""D1-574/MS-033 focused regression tests.

Covers:
  - Internal header consumption in _handle_codex_opencode_zen_adapter_route
    (body precedence, alias-probe ignore, invalid/empty 400, strict default)
  - Run-unique session injection and preservation (_inject_codex_session_id)
  - Separate thread_id extraction (_extract_command_thread_id)
  - _extract_command_session_id rejects thread fields
  - Exact Langfuse session lookup (_poll_langfuse_session_traces)
  - Named config header and session-history route filtering
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RA_SCRIPT = _REPO_ROOT / "scripts" / "local-ci" / "run_acceptance.py"
_ADAPTER_SCRIPT = (
    _REPO_ROOT / "scripts" / "local-ci" / "run_anthropic_adapter_acceptance.py"
)
_CONFIG_PATH = (
    _REPO_ROOT / "scripts" / "local-ci" / "anthropic_adapter_config.json"
)


def _load_ra():
    name = "run_acceptance_d1574"
    spec = importlib.util.spec_from_file_location(name, _RA_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_adapter():
    name = "run_anthropic_adapter_acceptance_d1574"
    spec = importlib.util.spec_from_file_location(name, _ADAPTER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ra():
    return _load_ra()


# ── Helpers ────────────────────────────────────────────────────────


def _make_request(headers: dict[str, str] | None = None) -> Request:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "query_string": b"",
    }
    return Request(scope)


def _make_normalized_request(body: dict[str, Any]):
    ns = MagicMock()
    ns.request_body = body
    ns.request_input = []
    ns.responses_api_request = {}
    return ns


# ── Header consumption tests ───────────────────────────────────────


def _install_stubs() -> MagicMock:
    """Bind the install()-gated names on the module and return mock norm."""
    mock_norm = MagicMock()
    mock_norm.normalize_codex_request = AsyncMock(
        side_effect=RuntimeError("stop-after-normalization")
    )
    codex_candidate_calls._anthropic_opencode_zen_normalization = mock_norm
    codex_candidate_calls._get_anthropic_opencode_zen_normalization_runtime = (
        lambda: MagicMock()
    )
    codex_candidate_calls.ProxyException = ProxyException
    return mock_norm


class TestHeaderConsumption:
    """x-aawm-opencode-zen-unsupported-tools-mode header behavior."""

    @pytest.mark.asyncio
    async def test_valid_drop_header_injects_metadata(self):
        mock_norm = _install_stubs()
        body: dict[str, Any] = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        try:
            await codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                adapter_model="opencode_zen/big-pickle",
                use_alias_candidate_probe=False,
            )
        except RuntimeError:
            pass
        call_body = mock_norm.normalize_codex_request.call_args[0][1]
        assert (
            call_body["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "drop"
        )

    @pytest.mark.asyncio
    async def test_body_metadata_wins_over_header(self):
        mock_norm = _install_stubs()
        body: dict[str, Any] = {
            "model": "opencode_zen/big-pickle",
            "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "strict"},
        }
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        try:
            await codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                adapter_model="opencode_zen/big-pickle",
                use_alias_candidate_probe=False,
            )
        except RuntimeError:
            pass
        call_body = mock_norm.normalize_codex_request.call_args[0][1]
        assert (
            call_body["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "strict"
        )

    @pytest.mark.asyncio
    async def test_alias_probe_ignores_header(self):
        mock_norm = _install_stubs()
        body: dict[str, Any] = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        try:
            await codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                adapter_model="opencode_zen/big-pickle",
                use_alias_candidate_probe=True,
            )
        except RuntimeError:
            pass
        call_body = mock_norm.normalize_codex_request.call_args[0][1]
        assert "litellm_metadata" not in call_body or (
            "opencode_zen_unsupported_tools_mode"
            not in call_body.get("litellm_metadata", {})
        )

    @pytest.mark.asyncio
    async def test_invalid_header_value_returns_400(self):
        _install_stubs()
        body: dict[str, Any] = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "yolo"}
        )
        with pytest.raises(ProxyException) as exc_info:
            await codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                adapter_model="opencode_zen/big-pickle",
                use_alias_candidate_probe=False,
            )
        assert str(exc_info.value.code) == "400"

    @pytest.mark.asyncio
    async def test_empty_header_value_returns_400(self):
        _install_stubs()
        body: dict[str, Any] = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "  "}
        )
        with pytest.raises(ProxyException) as exc_info:
            await codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                adapter_model="opencode_zen/big-pickle",
                use_alias_candidate_probe=False,
            )
        assert str(exc_info.value.code) == "400"

    @pytest.mark.asyncio
    async def test_header_absent_preserves_strict_default(self):
        mock_norm = _install_stubs()
        body: dict[str, Any] = {"model": "opencode_zen/big-pickle"}
        request = _make_request({})
        try:
            await codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                endpoint="/v1/responses",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=body,
                adapter_model="opencode_zen/big-pickle",
                use_alias_candidate_probe=False,
            )
        except RuntimeError:
            pass
        call_body = mock_norm.normalize_codex_request.call_args[0][1]
        assert "litellm_metadata" not in call_body or (
            "opencode_zen_unsupported_tools_mode"
            not in call_body.get("litellm_metadata", {})
        )


# ── Session injection tests ────────────────────────────────────────


class TestSessionInjection:
    """_inject_codex_session_id run-unique session behavior."""

    def test_injects_run_unique_session(self, ra):
        cmd = ["codex", "exec", "-p", "litellm", "--json", "hello"]
        new_cmd, session = ra._inject_codex_session_id(cmd)
        assert session.startswith("acceptance-")
        assert len(session) > len("acceptance-")
        # Must contain the -c override
        joined = " ".join(new_cmd)
        assert "model_providers.litellm.http_headers.session_id=" in joined
        assert session in joined

    def test_preserves_preexisting_session(self, ra):
        cmd = [
            "codex", "exec", "-p", "myprofile",
            "-c",
            'model_providers.myprofile.http_headers.session_id="my-fixed-id"',
            "--json",
            "hello",
        ]
        new_cmd, session = ra._inject_codex_session_id(cmd)
        assert session == "my-fixed-id"
        assert new_cmd == cmd  # unchanged

    def test_two_calls_produce_unique_sessions(self, ra):
        cmd = ["codex", "exec", "-p", "litellm", "--json", "hi"]
        _, s1 = ra._inject_codex_session_id(cmd)
        _, s2 = ra._inject_codex_session_id(cmd)
        assert s1 != s2


# ── Thread extraction tests ────────────────────────────────────────


class TestThreadExtraction:
    """_extract_command_thread_id and session_id separation."""

    def test_extracts_thread_id_from_thread_started(self, ra):
        stdout = json.dumps(
            {"type": "thread.started", "thread_id": "thr-abc123"}
        )
        assert ra._extract_command_thread_id(stdout) == "thr-abc123"

    def test_extracts_thread_id_camel_case(self, ra):
        stdout = json.dumps(
            {"type": "thread.started", "threadId": "thr-xyz"}
        )
        assert ra._extract_command_thread_id(stdout) == "thr-xyz"

    def test_ignores_non_thread_started_type(self, ra):
        stdout = json.dumps(
            {"type": "message.created", "thread_id": "thr-nope"}
        )
        assert ra._extract_command_thread_id(stdout) is None

    def test_session_id_extractor_rejects_thread_fields(self, ra):
        """_extract_command_session_id must NOT accept thread_id/threadId."""
        stdout = json.dumps(
            {"type": "thread.started", "thread_id": "thr-only"}
        )
        assert ra._extract_command_session_id(stdout) is None

    def test_session_id_extractor_accepts_session_fields(self, ra):
        stdout = json.dumps({"session_id": "sess-123"})
        assert ra._extract_command_session_id(stdout) == "sess-123"


# ── Langfuse session lookup tests ──────────────────────────────────


class TestLangfuseSessionLookup:
    """_poll_langfuse_session_traces uses exact session_id."""

    def test_passes_session_id_to_query(self, ra):
        import datetime as dt

        captured_params: dict[str, Any] = {}

        def fake_all_traces(**kwargs):
            captured_params.update(kwargs)
            return [{"id": "t1", "name": "test"}]

        with patch.object(
            ra, "_recent_langfuse_all_traces", side_effect=fake_all_traces
        ):
            traces, err = ra._poll_langfuse_session_traces(
                query_url="http://localhost:3000",
                public_key="pk",
                secret_key="sk",
                user_id=None,
                start_time=dt.datetime.now(dt.timezone.utc),
                session_id="acceptance-deadbeef1234",
                timeout_seconds=1,
            )
        assert captured_params["session_id"] == "acceptance-deadbeef1234"
        assert len(traces) == 1
        assert err is None


# ── Config-level assertions ────────────────────────────────────────


class TestNamedConfig:
    """anthropic_adapter_config.json D1-574 case structure."""

    @pytest.fixture(scope="class")
    def config(self):
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_harness_case_has_internal_header(self, config):
        case = config["cases"][
            "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"
        ]
        joined = " ".join(case["command"])
        assert "x-aawm-opencode-zen-unsupported-tools-mode" in joined
        assert '"drop"' in joined

    def test_session_history_requires_opencode_zen_route(self, config):
        case = config["cases"][
            "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"
        ]
        shv = case["session_history_validation"]
        rows = shv["expected_rows"]
        assert len(rows) == 1
        row = rows[0]
        assert row["required_equals"]["provider"] == "opencode_zen"
        assert row["required_equals"]["model"] == "big-pickle"
        assert (
            row["metadata_required_equals"]["passthrough_route_family"]
            == "codex_opencode_zen_adapter"
        )

    def test_session_history_rejects_unrelated_openai_row(self, config):
        """Real matcher rejects unrelated OpenAI row and accepts OpenCode Zen row."""
        adapter = _load_adapter()
        case = config["cases"][
            "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"
        ]
        shv = case["session_history_validation"]
        expected_rows = shv["expected_rows"]

        # Unrelated OpenAI row must be rejected
        unrelated_openai_row = {
            "provider": "openai",
            "model": "gpt-5.6-terra",
            "metadata": {"passthrough_route_family": "codex_responses"},
        }
        matched, failures = adapter._match_session_history_expected_rows(
            family="test",
            records=[unrelated_openai_row],
            expected_rows=expected_rows,
        )
        assert len(matched) == 0
        assert len(failures) > 0

        # Correct OpenCode Zen row must match
        correct_row = {
            "provider": "opencode_zen",
            "model": "big-pickle",
            "metadata": {"passthrough_route_family": "codex_opencode_zen_adapter"},
        }
        matched, failures = adapter._match_session_history_expected_rows(
            family="test",
            records=[correct_row],
            expected_rows=expected_rows,
        )
        assert len(matched) == 1
        assert len(failures) == 0


# ── Artifact-level thread_id recording ─────────────────────────────


class TestValidateCodexThreadArtifact:
    """_validate_codex records command_thread_id separately from session_id."""

    def test_baseline_validate_codex_records_thread_id(self, ra):
        stdout = (
            json.dumps({"type": "thread.started", "thread_id": "thr-artifact"})
            + "\n"
            + json.dumps({"session_id": "sess-artifact"})
        )
        run_result = {
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_original_chars": len(stdout),
            "stderr_original_chars": 0,
            "output_max_chars": 200_000,
            "response_excerpt": "ok",
        }
        config = {
            "command": ["codex", "exec", "-p", "litellm", "--json", "hi"],
            "timeout_seconds": 10,
        }
        with (
            patch.object(ra, "_run_command", return_value=run_result),
            patch.object(
                ra, "_poll_langfuse_session_traces", return_value=([], None)
            ),
            patch.object(
                ra,
                "_validate_generation_observations",
                return_value=([], [], []),
            ),
            patch.object(
                ra, "_validate_trace_enrichment", return_value=({}, [], [])
            ),
            patch.object(
                ra, "_validate_trace_context", return_value=({}, [])
            ),
            patch.object(
                ra, "_validate_generation_metadata", return_value=({}, [])
            ),
            patch.object(
                ra, "_validate_span_observations", return_value=([], [], [])
            ),
        ):
            result = ra._validate_codex(
                config,
                query_url="http://lf",
                public_key="pk",
                secret_key="sk",
            )
        assert result["langfuse"]["command_thread_id"] == "thr-artifact"
        assert result["langfuse"]["command_session_id"] == "sess-artifact"
        assert (
            result["langfuse"]["command_thread_id"]
            != result["langfuse"]["command_session_id"]
        )
