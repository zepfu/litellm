"""D1-574/MS-033 focused regression tests.

Covers:
  - Direct tools-mode resolution in _handle_codex_opencode_zen_adapter_route
    (body precedence, alias-probe strict, invalid/empty 400, drop default)
  - Run-unique session injection and preservation (_inject_codex_session_id)
  - Separate thread_id extraction (_extract_command_thread_id)
  - _extract_command_session_id rejects thread fields
  - Exact Langfuse session lookup (_poll_langfuse_session_traces)
  - Named config header and session-history route filtering
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

import litellm
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
    codex_dispatch,
)
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
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
    """Bind install()-gated names in the active handler host globals."""
    mock_norm = MagicMock()
    mock_norm.normalize_codex_request = AsyncMock(
        side_effect=RuntimeError("stop-after-normalization")
    )
    host_globals = (
        codex_candidate_calls._handle_codex_opencode_zen_adapter_route.__globals__
    )
    host_globals["_anthropic_opencode_zen_normalization"] = mock_norm
    host_globals["_get_anthropic_opencode_zen_normalization_runtime"] = (
        lambda: MagicMock()
    )
    host_globals["ProxyException"] = ProxyException
    return mock_norm


@pytest.fixture(autouse=True)
def _restore_direct_handler_host_stubs():
    """Keep active production-host stubs isolated to each focused test."""
    host_globals = (
        codex_candidate_calls._handle_codex_opencode_zen_adapter_route.__globals__
    )
    names = (
        "_anthropic_opencode_zen_normalization",
        "_get_anthropic_opencode_zen_normalization_runtime",
        "ProxyException",
    )
    missing = object()
    originals = {name: host_globals.get(name, missing) for name in names}
    yield
    for name, original in originals.items():
        if original is missing:
            host_globals.pop(name, None)
        else:
            host_globals[name] = original


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
    async def test_header_absent_injects_drop_default(self):
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
        assert (
            call_body["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "drop"
        )

    @pytest.mark.asyncio
    async def test_explicit_body_drop_wins_without_header(self):
        mock_norm = _install_stubs()
        body: dict[str, Any] = {
            "model": "opencode_zen/big-pickle",
            "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "drop"},
        }
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
        assert (
            call_body["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "drop"
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

    def test_harness_case_has_no_control_header(self, config):
        case = config["cases"][
            "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"
        ]
        joined = " ".join(case["command"])
        assert "x-aawm-opencode-zen-unsupported-tools-mode" not in joined

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


# ── Pre-alias header translation tests (codex_dispatch) ────────────


class TestPreAliasHeaderTranslation:
    """_prepare_opencode_zen_direct_tools_mode pre-alias behavior."""

    def test_valid_drop_header_translates_to_metadata(self):
        body = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model="big-pickle"
        )
        assert (
            result["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "drop"
        )

    def test_body_metadata_wins_over_header(self):
        body = {
            "model": "opencode_zen/big-pickle",
            "litellm_metadata": {"opencode_zen_unsupported_tools_mode": "strict"},
        }
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model="big-pickle"
        )
        assert (
            result["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "strict"
        )

    def test_invalid_header_value_returns_400(self):
        body = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "yolo"}
        )
        with pytest.raises(ProxyException) as exc_info:
            codex_dispatch._prepare_opencode_zen_direct_tools_mode(
                request, body, direct_adapter_model="big-pickle"
            )
        assert str(exc_info.value.code) == "400"

    def test_empty_header_value_returns_400(self):
        body = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "  "}
        )
        with pytest.raises(ProxyException) as exc_info:
            codex_dispatch._prepare_opencode_zen_direct_tools_mode(
                request, body, direct_adapter_model="big-pickle"
            )
        assert str(exc_info.value.code) == "400"

    def test_absent_header_injects_drop_default(self):
        body = {"model": "opencode_zen/big-pickle"}
        request = _make_request({})
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model="big-pickle"
        )
        assert (
            result["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "drop"
        )

    def test_unrelated_model_no_effect(self):
        body = {"model": "gpt-5.6-terra"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model=None
        )
        assert result is body
        assert "litellm_metadata" not in result

    def test_alias_model_no_effect(self):
        """Auto-agent alias models are not direct OpenCode models."""
        body = {"model": "codex-auto-agent"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model=None
        )
        assert result is body

    def test_unsupported_opencode_prefix_valid_header_no_effect(self):
        body = {"model": "opencode_zen/not-supported"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model=None
        )
        assert result is body
        assert "litellm_metadata" not in result

    def test_unsupported_opencode_prefix_invalid_header_no_effect(self):
        body = {"model": "opencode_zen/not-supported"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "invalid"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model=None
        )
        assert result is body
        assert "litellm_metadata" not in result

    def test_does_not_mutate_original_body(self):
        body = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        result = codex_dispatch._prepare_opencode_zen_direct_tools_mode(
            request, body, direct_adapter_model="big-pickle"
        )
        assert result is not body
        assert "litellm_metadata" not in body

    def test_probe_behavior_unchanged(self):
        """Alias candidate handler still ignores headers (probe mode)."""
        mock_norm = _install_stubs()
        body = {"model": "opencode_zen/big-pickle"}
        request = _make_request(
            {"x-aawm-opencode-zen-unsupported-tools-mode": "drop"}
        )
        import asyncio

        try:
            asyncio.run(
                codex_candidate_calls._handle_codex_opencode_zen_adapter_route(
                    endpoint="/v1/responses",
                    request=request,
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    prepared_request_body=body,
                    adapter_model="opencode_zen/big-pickle",
                    use_alias_candidate_probe=True,
                )
            )
        except RuntimeError as exc:
            if "stop-after-normalization" not in str(exc):
                raise
        call_body = mock_norm.normalize_codex_request.call_args[0][1]
        assert "litellm_metadata" not in call_body or (
            "opencode_zen_unsupported_tools_mode"
            not in call_body.get("litellm_metadata", {})
        )


# ── try_dispatch integration-shaped tests ─────────────────────────


class TestTryDispatchDropIntegration:
    """try_dispatch_codex_request injects drop metadata for direct OpenCode."""

    @pytest.mark.asyncio
    async def test_direct_dispatch_injects_drop_metadata(self):
        captured: dict[str, Any] = {}

        async def fake_handler(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        dispatch_globals = codex_dispatch.try_dispatch_codex_request.__globals__
        with patch.dict(
            dispatch_globals,
            {
                "_resolve_codex_opencode_zen_adapter_model": MagicMock(
                    return_value="big-pickle"
                ),
                "_resolve_codex_auto_agent_alias_model": MagicMock(
                    return_value=None
                ),
                "_prepare_request_body_for_passthrough_observability": (
                    lambda *, request, request_body: request_body
                ),
                "_safe_set_request_parsed_body": MagicMock(),
                "_handle_codex_opencode_zen_adapter_route": fake_handler,
            },
        ):
            body: dict[str, Any] = {"model": "opencode_zen/big-pickle"}
            result = await codex_dispatch.try_dispatch_codex_request(
                endpoint="/v1/responses",
                request=_make_request({}),
                request_body=body,
                prepared_request_body=body,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                target_url="http://localhost",
                api_key=None,
                forward_headers=False,
            )

        assert result is not None
        dispatched_body = captured["prepared_request_body"]
        assert (
            dispatched_body["litellm_metadata"][
                "opencode_zen_unsupported_tools_mode"
            ]
            == "drop"
        )
        # Original body must not be mutated
        assert "litellm_metadata" not in body


class TestInstalledHostDirectHandler:
    """Installed host handler defaults direct OpenCode before normalization."""

    @pytest.mark.parametrize(
        (
            "raw_headers",
            "authenticated_end_user_id",
            "expected_trace_name",
            "expected_trace_user_id",
        ),
        [
            (
                {
                    "langfuse_trace_name": "  normalized-name  ",
                    "langfuse_trace_user_id": "raw-untrusted-user",
                    "x-litellm-end-user-id": "  normalized-user  ",
                },
                "  normalized-user  ",
                "normalized-name",
                "normalized-user",
            ),
            (
                {
                    "Langfuse-Trace-Name": "  hyphen-name  ",
                    "Langfuse-Trace-User-Id": "  hyphen-user  ",
                    "x-litellm-end-user-id": "  authenticated-user  ",
                },
                "  authenticated-user  ",
                "hyphen-name",
                "authenticated-user",
            ),
            (
                {
                    "langfuse_trace_name": "  ",
                    "langfuse_trace_user_id": "\t",
                    "x-litellm-end-user-id": "\t",
                },
                "\t",
                "orchestrator",
                "existing-user",
            ),
            (
                {
                    "langfuse_trace_name": "n" * 513,
                    "langfuse_trace_user_id": "u" * 513,
                    "x-litellm-end-user-id": "u" * 513,
                },
                "u" * 513,
                "orchestrator",
                "existing-user",
            ),
            (
                {
                    "langfuse_trace_name": "trusted-name",
                    "langfuse_trace_user_id": "raw-untrusted-user",
                },
                None,
                "trusted-name",
                "existing-user",
            ),
            (
                {
                    "langfuse_trace_name": "trusted-name",
                    "x-litellm-end-user-id": "header-user",
                },
                "different-auth-user",
                "trusted-name",
                "existing-user",
            ),
        ],
    )
    def test_direct_callback_headers_cannot_override_bounded_trace_identity(
        self,
        raw_headers: dict[str, str],
        authenticated_end_user_id: str | None,
        expected_trace_name: str,
        expected_trace_user_id: str,
    ):
        request = SimpleNamespace(
            headers={
                **raw_headers,
                "langfuse_environment": "dev",
                "x-correlation-id": "correlation-1",
            }
        )
        original_headers = dict(request.headers)
        original_body = {
            "litellm_metadata": {
                "session_id": "session-1",
                "tenant_id": "tenant-1",
                "repository": "org/repo",
                "trace_name": "orchestrator",
                "trace_user_id": "existing-user",
            }
        }
        (
            prepared_body,
            accepted_trace_user_id,
        ) = (
            codex_candidate_calls._prepare_opencode_zen_direct_observability_metadata(
                request,
                original_body,
                False,
                SimpleNamespace(end_user_id=authenticated_end_user_id),
            )
        )
        normalized_metadata = prepared_body["litellm_metadata"]
        assert accepted_trace_user_id == (
            expected_trace_user_id
            if expected_trace_user_id != "existing-user"
            else None
        )

        callback_headers = (
            codex_candidate_calls._opencode_zen_callback_headers(request)
        )
        assert request.headers == original_headers
        assert original_body["litellm_metadata"] == {
            "session_id": "session-1",
            "tenant_id": "tenant-1",
            "repository": "org/repo",
            "trace_name": "orchestrator",
            "trace_user_id": "existing-user",
        }
        assert callback_headers["x-correlation-id"] == "correlation-1"
        assert callback_headers["langfuse_environment"] == "dev"
        assert all(
            name.lower().replace("-", "_")
            not in {"langfuse_trace_name", "langfuse_trace_user_id"}
            for name in callback_headers
        )

        callback_metadata = LangFuseLogger.add_metadata_from_header(
            {"proxy_server_request": {"headers": callback_headers}},
            dict(normalized_metadata),
        )
        assert callback_metadata["trace_name"] == expected_trace_name
        assert callback_metadata["trace_user_id"] == expected_trace_user_id
        assert callback_metadata["environment"] == "dev"
        assert callback_metadata["session_id"] == "session-1"
        assert callback_metadata["tenant_id"] == "tenant-1"
        assert callback_metadata["repository"] == "org/repo"
        if expected_trace_name != "orchestrator":
            assert callback_metadata["source_trace_name"] == "orchestrator"
        else:
            assert "source_trace_name" not in callback_metadata
        if expected_trace_user_id != "existing-user":
            assert (
                callback_metadata["source_trace_user_id"]
                == "existing-user"
            )
        else:
            assert "source_trace_user_id" not in callback_metadata

    def test_alias_observability_headers_are_unchanged(self):
        request = _make_request(
            {
                "langfuse_trace_name": "alias-name",
                "langfuse_trace_user_id": "alias-user",
            }
        )
        body = {
            "litellm_metadata": {
                "trace_name": "orchestrator",
                "trace_user_id": "existing-user",
            }
        }

        result, accepted_trace_user_id = (
            codex_candidate_calls._prepare_opencode_zen_direct_observability_metadata(
                request,
                body,
                True,
            )
        )

        assert result is body
        assert accepted_trace_user_id is None

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"langfuse_trace_user_id": "  "},
            {"Langfuse-Trace-User-Id": "u" * 513},
            {"langfuse_trace_user_id": "raw-untrusted-user"},
        ],
    )
    def test_untrusted_header_does_not_select_body_trace_user_for_promotion(
        self,
        headers: dict[str, str],
    ):
        request = _make_request(headers)
        body = {
            "litellm_metadata": {
                "trace_user_id": "body-controlled-user",
            }
        }

        prepared_body, accepted_trace_user_id = (
            codex_candidate_calls._prepare_opencode_zen_direct_observability_metadata(
                request,
                body,
                False,
                SimpleNamespace(end_user_id=None),
            )
        )

        assert prepared_body is body
        assert accepted_trace_user_id is None
        assert "user_api_key_end_user_id" not in body["litellm_metadata"]

    @pytest.mark.parametrize(
        "body",
        [
            {"user": "body-user"},
            {"litellm_metadata": {"user": "body-user"}},
            {"metadata": {"user_id": "body-user"}},
            {"safety_identifier": "body-user"},
        ],
    )
    def test_body_derived_auth_end_user_is_not_promoted(
        self,
        body: dict[str, Any],
    ):
        authenticated_end_user_id = get_end_user_id_from_request_body(
            body,
            {},
        )
        assert authenticated_end_user_id == "body-user"
        prepared_body = {
            **body,
            "litellm_metadata": {
                **dict(body.get("litellm_metadata") or {}),
                "trace_user_id": "existing-user",
            },
        }

        result, accepted_trace_user_id = (
            codex_candidate_calls._prepare_opencode_zen_direct_observability_metadata(
                _make_request({}),
                prepared_body,
                False,
                SimpleNamespace(end_user_id=authenticated_end_user_id),
            )
        )

        assert result is prepared_body
        assert accepted_trace_user_id is None
        assert result["litellm_metadata"]["trace_user_id"] == "existing-user"
        assert "source_trace_user_id" not in result["litellm_metadata"]

    def test_direct_and_kimi_logging_select_same_success_callbacks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        configured_callbacks = ["aawm_agent_identity", "langfuse"]
        monkeypatch.setattr(
            litellm,
            "_async_success_callback",
            configured_callbacks,
        )
        metadata = {
            "trace_name": "d1-574",
            "trace_user_id": "d1-574-user",
        }
        base_kwargs = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": metadata,
            "litellm_metadata": metadata,
            "proxy_server_request": {"headers": {}, "body": {}},
            "stream": True,
        }

        def make_logging(model: str, call_id: str) -> Logging:
            return Logging(
                model=model,
                messages=base_kwargs["messages"],
                stream=True,
                call_type="acompletion",
                start_time=datetime.datetime.now(),
                litellm_call_id=call_id,
                function_id="",
                kwargs={**base_kwargs, "model": model},
            )

        def selected_callbacks(logging_obj: Logging) -> list[Any]:
            litellm_params = logging_obj.model_call_details["litellm_params"]
            callbacks = logging_obj.get_combined_callback_list(
                dynamic_success_callbacks=(
                    logging_obj.dynamic_async_success_callbacks
                ),
                global_callbacks=litellm._async_success_callback,
            )
            return [
                callback
                for callback in callbacks
                if logging_obj.should_run_callback(
                    callback=callback,
                    litellm_params=litellm_params,
                    event_hook="async_success_handler",
                )
            ]

        direct_logging = make_logging("openai/big-pickle", "direct-call")
        kimi_logging = make_logging("openai/kimi-k2.5", "kimi-call")

        assert (
            direct_logging.model_call_details["litellm_params"].get(
                "litellm_disabled_callbacks"
            )
            is None
        )
        assert (
            kimi_logging.model_call_details["litellm_params"].get(
                "litellm_disabled_callbacks"
            )
            is None
        )
        assert selected_callbacks(direct_logging) == configured_callbacks
        assert selected_callbacks(kimi_logging) == configured_callbacks

    @pytest.mark.asyncio
    async def test_installed_direct_handler_injects_drop_before_normalization(self):
        mock_norm = MagicMock()
        mock_norm.normalize_codex_request = AsyncMock(
            side_effect=RuntimeError("stop-after-normalization")
        )
        host_globals = dict(codex_candidate_calls.__dict__)
        host_globals["litellm"] = litellm
        codex_candidate_calls.install(host_globals)
        host_globals["_anthropic_opencode_zen_normalization"] = mock_norm
        host_globals["_get_anthropic_opencode_zen_normalization_runtime"] = (
            lambda: MagicMock()
        )

        with pytest.raises(RuntimeError, match="stop-after-normalization"):
            await host_globals["_handle_codex_opencode_zen_adapter_route"](
                endpoint="/v1/responses",
                request=_make_request({}),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body={
                    "model": "opencode_zen/big-pickle"
                },
                adapter_model="big-pickle",
                use_alias_candidate_probe=False,
            )

        call_body = mock_norm.normalize_codex_request.call_args[0][1]
        assert (
            call_body["litellm_metadata"]["opencode_zen_unsupported_tools_mode"]
            == "drop"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("client_user", ["litellm", "explicit-client-user"])
    async def test_installed_direct_handler_promotes_bounded_trace_metadata(  # noqa: PLR0915
        self,
        monkeypatch: pytest.MonkeyPatch,
        client_user: str,
    ):
        trace_user_id = "d1-574-harness-user"
        trace_name = (
            "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"
        )
        litellm_metadata = {
            "opencode_zen_unsupported_tools_mode": "drop",
            "passthrough_route_family": "codex_opencode_zen_adapter",
            "session_id": "acceptance-d1574-observability",
            "tenant_id": "tenant-d1574",
            "repository": "litellm",
            "trace_name": "orchestrator",
            "trace_user_id": "litellm",
        }

        def make_normalized_request(prepared_body: dict[str, Any]):
            normalized_metadata = dict(
                prepared_body.get("litellm_metadata") or {}
            )
            normalized_user = prepared_body.get("user")
            return SimpleNamespace(
                request_body={
                    "model": "big-pickle",
                    "litellm_metadata": normalized_metadata,
                    "stream": True,
                    "user": normalized_user,
                },
                request_input="Reply with exactly two words: opencode pickle",
                responses_api_request={
                    "stream": True,
                    "user": normalized_user,
                },
                litellm_metadata=normalized_metadata,
                completion_kwargs={
                    "messages": [
                        {"role": "user", "content": "opencode pickle"}
                    ],
                    "metadata": normalized_metadata,
                    "model": "big-pickle",
                    "stream": True,
                    "user": normalized_user,
                    "custom_llm_provider": "openai",
                },
                requested_model="opencode_zen/big-pickle",
                client_requested_stream=True,
            )

        mock_norm = MagicMock()
        mock_norm.normalize_codex_request = AsyncMock(
            side_effect=lambda _runtime, prepared_body, **_kwargs: (
                make_normalized_request(prepared_body)
            )
        )
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-after-acompletion")

        host_globals = dict(codex_candidate_calls.__dict__)
        host_globals["httpx"] = httpx
        host_globals["litellm"] = litellm
        host_globals["ResponsesAPIOptionalRequestParams"] = dict
        codex_candidate_calls.install(host_globals)
        host_globals["_anthropic_opencode_zen_normalization"] = mock_norm
        host_globals["_get_anthropic_opencode_zen_normalization_runtime"] = (
            lambda: MagicMock()
        )
        host_globals["_load_opencode_zen_api_key_for_candidate"] = AsyncMock(
            return_value="test-opencode-key"
        )
        host_globals["_get_opencode_zen_target_base"] = (
            lambda: "https://opencode.ai/zen"
        )
        host_globals["_join_opencode_zen_passthrough_url"] = (
            lambda *, base_target_url, endpoint: (
                f"{base_target_url.rstrip('/')}{endpoint}"
            )
        )
        host_globals["BaseOpenAIPassThroughHandler"] = SimpleNamespace(
            _assemble_headers=lambda **_kwargs: {}
        )
        host_globals["HttpPassThroughEndpointHelpers"] = SimpleNamespace(
            validate_outgoing_egress=lambda **_kwargs: None
        )
        host_globals["_annotate_request_scope_for_adapted_access_log"] = (
            lambda *_args, **_kwargs: None
        )
        host_globals["_build_adapted_route_rollup_kwargs"] = (
            lambda _metadata: {}
        )
        host_globals["_emit_adapted_route_access_log"] = (
            lambda **_kwargs: None
        )
        host_globals["_maybe_raise_opencode_zen_direct_rate_limit"] = (
            lambda _exc: None
        )
        host_globals["_opencode_zen_candidate_unavailable_detail"] = (
            lambda _exc: None
        )
        host_globals["_get_proxy_shared_aiohttp_session"] = lambda: MagicMock()
        monkeypatch.setattr(
            litellm,
            "acompletion",
            fake_acompletion,
        )

        direct_request = _make_request(
            {
                "langfuse_trace_name": f" {trace_name} ",
                "x-litellm-end-user-id": trace_user_id,
                "x-correlation-id": "correlation-1",
            }
        )
        direct_headers_before = list(direct_request.headers.raw)
        direct_request_body = {
            "model": "opencode_zen/big-pickle",
            "litellm_metadata": dict(litellm_metadata),
            "stream": True,
            "user": client_user,
        }
        authenticated_end_user_id = get_end_user_id_from_request_body(
            direct_request_body,
            dict(direct_request.headers),
        )
        assert authenticated_end_user_id == trace_user_id
        with pytest.raises(RuntimeError, match="stop-after-acompletion"):
            await host_globals["_handle_codex_opencode_zen_adapter_route"](
                endpoint="/v1/responses",
                request=direct_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=SimpleNamespace(
                    end_user_id=authenticated_end_user_id
                ),
                prepared_request_body=direct_request_body,
                adapter_model="big-pickle",
                use_alias_candidate_probe=False,
            )

        direct_captured = dict(captured)
        assert list(direct_request.headers.raw) == direct_headers_before
        callback_headers = direct_captured["proxy_server_request"]["headers"]
        assert callback_headers["x-correlation-id"] == "correlation-1"
        assert callback_headers["x-litellm-end-user-id"] == trace_user_id
        assert "langfuse_trace_name" not in callback_headers
        assert "langfuse_trace_user_id" not in callback_headers
        assert (
            direct_captured["proxy_server_request"]["body"]["user"]
            == client_user
        )
        assert direct_captured["metadata"]["trace_name"] == trace_name
        assert direct_captured["metadata"]["trace_user_id"] == trace_user_id
        assert direct_captured["metadata"]["source_trace_name"] == "orchestrator"
        assert direct_captured["metadata"]["source_trace_user_id"] == "litellm"
        assert (
            direct_captured["metadata"]["session_id"]
            == "acceptance-d1574-observability"
        )
        assert direct_captured["metadata"]["tenant_id"] == "tenant-d1574"
        assert direct_captured["metadata"]["repository"] == "litellm"
        assert (
            direct_captured["metadata"]["user_api_key_end_user_id"]
            == trace_user_id
        )
        assert (
            direct_captured["litellm_metadata"]["user_api_key_end_user_id"]
            == trace_user_id
        )
        assert (
            direct_captured["metadata"]
            is direct_captured["litellm_metadata"]
        )
        assert direct_captured["user"] == client_user

        langfuse_callback = object.__new__(LangfusePromptManagement)
        fake_langfuse_logger = MagicMock()
        monkeypatch.setattr(
            LangFuseHandler,
            "get_langfuse_logger_for_request",
            MagicMock(return_value=fake_langfuse_logger),
        )
        monkeypatch.setattr(
            litellm,
            "_async_success_callback",
            [langfuse_callback],
        )
        monkeypatch.setattr(litellm, "callbacks", [])
        monkeypatch.setattr(litellm, "success_callback", [])
        logging_obj, _ = litellm.utils.function_setup(
            original_function="acompletion",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.datetime.now(),
            litellm_call_id="d1-574-direct-callback",
            model=direct_captured["model"],
            messages=direct_captured["messages"],
            stream=True,
            custom_llm_provider=direct_captured["custom_llm_provider"],
            metadata=direct_captured["metadata"],
            litellm_metadata=direct_captured["litellm_metadata"],
            proxy_server_request=direct_captured["proxy_server_request"],
            user=direct_captured["user"],
        )
        selected_callbacks = logging_obj.get_combined_callback_list(
            dynamic_success_callbacks=(
                logging_obj.dynamic_async_success_callbacks
            ),
            global_callbacks=litellm._async_success_callback,
        )
        assert selected_callbacks == [langfuse_callback]

        response = ModelResponse(
            id="d1-574-response",
            model="big-pickle",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "opencode pickle",
                    },
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )
        callback_started = datetime.datetime.now()
        await logging_obj.async_success_handler(
            result=response,
            start_time=callback_started,
            end_time=callback_started,
            cache_hit=False,
        )
        fake_langfuse_logger.log_event_on_langfuse.assert_called_once()
        callback_kwargs = (
            fake_langfuse_logger.log_event_on_langfuse.call_args.kwargs[
                "kwargs"
            ]
        )
        assert (
            callback_kwargs["standard_logging_object"]["metadata"][
                "user_api_key_end_user_id"
            ]
            == trace_user_id
        )

        captured.clear()
        alias_request = _make_request(
            {"langfuse_trace_user_id": "raw-alias-user"}
        )
        with pytest.raises(RuntimeError, match="stop-after-acompletion"):
            await host_globals["_handle_codex_opencode_zen_adapter_route"](
                endpoint="/v1/responses",
                request=alias_request,
                fastapi_response=MagicMock(),
                user_api_key_dict=SimpleNamespace(
                    end_user_id=trace_user_id
                ),
                prepared_request_body={
                    "model": "opencode_zen/big-pickle",
                    "litellm_metadata": dict(litellm_metadata),
                    "stream": True,
                    "user": client_user,
                },
                adapter_model="big-pickle",
                use_alias_candidate_probe=True,
            )

        assert captured["metadata"]["trace_user_id"] == "litellm"
        assert "user_api_key_end_user_id" not in captured["metadata"]
        assert captured["user"] == client_user
        assert (
            captured["proxy_server_request"]["headers"][
                "langfuse_trace_user_id"
            ]
            == "raw-alias-user"
        )

        captured.clear()
        kimi_completion_metadata = {"trace_user_id": trace_user_id}
        with pytest.raises(RuntimeError, match="stop-after-acompletion"):
            await host_globals[
                "_perform_codex_kimi_chat_completions_adapter_call"
            ](
                config=MagicMock(),
                request=_make_request({}),
                prepared_request_body={
                    "model": "kimi-test",
                    "user": "litellm",
                },
                adapter_model="kimi-test",
                target_url="https://example.invalid/v1/chat/completions",
                api_key="test-key",
                api_base="https://example.invalid/v1",
                client_requested_stream=False,
                completion_kwargs={
                    "messages": [],
                    "metadata": dict(kimi_completion_metadata),
                    "model": "kimi-test",
                    "stream": False,
                },
                request_input=[],
                responses_api_request={},
                litellm_metadata=dict(kimi_completion_metadata),
                upstream_model="kimi-test",
            )

        assert captured["metadata"] == kimi_completion_metadata
        assert "user_api_key_end_user_id" not in captured["metadata"]
        assert "user" not in captured


# ── turn.failed sanitizer tests ────────────────────────────────────


class TestTurnFailedSanitizer:
    """_sanitize_turn_failed_output parser behavior."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return _load_adapter()

    def test_400_code_extracted(self, adapter):
        inner = json.dumps({"error": {"code": 400}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert result["api_error_status"] == 400
        assert result["status_code"] == 400

    def test_429_code_extracted(self, adapter):
        inner = json.dumps({"error": {"code": 429}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert result["api_error_status"] == 429
        assert result["status_code"] == 429

    def test_string_code_400(self, adapter):
        inner = json.dumps({"error": {"code": "400"}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["api_error_status"] == 400

    def test_string_code_429(self, adapter):
        inner = json.dumps({"error": {"code": "429"}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["api_error_status"] == 429

    def test_malformed_json_returns_bare_error(self, adapter):
        obj = {
            "type": "turn.failed",
            "error": {"message": "not json at all"},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert "api_error_status" not in result

    def test_out_of_range_code_rejected(self, adapter):
        inner = json.dumps({"error": {"code": 500}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert "api_error_status" not in result

    def test_top_level_integer_code_rejected(self, adapter):
        inner = json.dumps({"code": 429})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert "api_error_status" not in result

    def test_second_layer_top_level_string_code_rejected(self, adapter):
        layer2 = json.dumps({"code": "400"})
        layer1 = json.dumps({"error": {"message": layer2}})
        obj = {
            "type": "turn.failed",
            "error": {"message": layer1},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert "api_error_status" not in result

    def test_bool_code_rejected(self, adapter):
        inner = json.dumps({"error": {"code": True}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert "api_error_status" not in result

    def test_float_code_rejected(self, adapter):
        inner = json.dumps({"error": {"code": 429.0}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert "api_error_status" not in result

    def test_secret_bearing_message_rejected(self, adapter):
        inner = json.dumps({"error": {"code": 429, "detail": "sk-abc123"}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert "api_error_status" not in result
        # Must not copy raw content
        assert "sk-abc123" not in json.dumps(result)

    @pytest.mark.parametrize(
        "encoded_secret",
        [
            '"api\\u002dkey":"secret-value"',
            '"api\\u005fkey":"secret-value"',
            '"identifier":"person\\u0040example.com"',
        ],
    )
    def test_parsed_layer_secret_bearing_message_rejected(
        self, adapter, encoded_secret
    ):
        inner = f'{{"error":{{"code":429}},{encoded_secret}}}'
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result == {"type": "turn.failed", "is_error": True}
        serialized = json.dumps(result)
        assert "secret-value" not in serialized
        assert "person@example.com" not in serialized

    def test_oversized_message_rejected(self, adapter):
        inner = json.dumps({"error": {"code": 429}})
        padded = inner + "x" * 5000
        obj = {
            "type": "turn.failed",
            "error": {"message": padded},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert "api_error_status" not in result

    def test_non_turn_failed_passthrough(self, adapter):
        obj = {"type": "result", "is_error": False}
        result = adapter._sanitize_turn_failed_output(obj)
        assert result is obj

    def test_two_layer_nested_json(self, adapter):
        layer2 = json.dumps({"error": {"code": 429}})
        layer1 = json.dumps({"error": {"message": layer2}})
        obj = {
            "type": "turn.failed",
            "error": {"message": layer1},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["api_error_status"] == 429
        assert result["status_code"] == 429

    def test_no_raw_message_copied(self, adapter):
        inner = json.dumps({"error": {"code": 400}})
        obj = {
            "type": "turn.failed",
            "error": {"message": inner},
        }
        result = adapter._sanitize_turn_failed_output(obj)
        assert "message" not in result
        assert "error" not in result

    def test_missing_error_key(self, adapter):
        obj = {"type": "turn.failed"}
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert "api_error_status" not in result

    def test_non_dict_error(self, adapter):
        obj = {"type": "turn.failed", "error": "string-error"}
        result = adapter._sanitize_turn_failed_output(obj)
        assert result["is_error"] is True
        assert "api_error_status" not in result


# ── Config retention/removal tests ─────────────────────────────────


class TestConfigRetentionRemoval:
    """D1-574 named case config: obsolete check removed, rest preserved."""

    @pytest.fixture(scope="class")
    def config(self):
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def case(self, config):
        return config["cases"][
            "native_openai_passthrough_responses_codex_opencode_zen_big_pickle"
        ]

    def test_command_json_checks_requires_terminal_error(self, case):
        assert case["command_json_checks"]["required_equals"]["is_error"] is True

    def test_expected_429_preserved(self, case):
        assert case["expected_api_error_status"] == 429

    def test_secret_checks_preserved(self, case):
        stdout_checks = case["command_stdout_text_checks"]
        forbidden = stdout_checks["forbidden_substrings"]
        assert "sk-" in forbidden
        assert "Bearer " in forbidden
        assert "api_key" in forbidden

    def test_provider_observations_preserved(self, case):
        obs = case["provider_error_observations_validation"]
        rows = obs["expected_rows"]
        assert len(rows) >= 1
        assert rows[0]["required_equals"]["provider"] == "opencode_zen"
        assert rows[0]["required_equals"]["status_code"] == 429

    def test_session_history_preserved(self, case):
        shv = case["session_history_validation"]
        rows = shv["expected_rows"]
        assert len(rows) == 1
        assert rows[0]["required_equals"]["model"] == "big-pickle"

    def test_no_control_header_in_command(self, case):
        joined = " ".join(case["command"])
        assert "x-aawm-opencode-zen-unsupported-tools-mode" not in joined




# ── D1-574 known-free cost lifecycle tests ────────────────────────


class TestKnownFreeCostLifecycle:
    """D1-574 direct OpenCode cost timing, streaming, and isolation."""

    _ISOLATED_HOST_NAMES = (
        "_anthropic_opencode_zen_normalization",
        "_get_anthropic_opencode_zen_normalization_runtime",
        "_load_opencode_zen_api_key_for_candidate",
        "_get_opencode_zen_target_base",
        "_join_opencode_zen_passthrough_url",
        "BaseOpenAIPassThroughHandler",
        "HttpPassThroughEndpointHelpers",
        "_annotate_request_scope_for_adapted_access_log",
        "_build_adapted_route_rollup_kwargs",
        "_emit_adapted_route_access_log",
        "_get_proxy_shared_aiohttp_session",
        "_opencode_zen_callback_headers",
        "_record_adapted_completed_route_rollup_turn",
        "_maybe_raise_opencode_zen_direct_rate_limit",
        "_opencode_zen_candidate_unavailable_detail",
        "_serialize_responses_adapter_response",
        "_is_codex_auto_agent_empty_success_responses_body",
        "_build_responses_response_from_adapter_response",
        "_responses_sse_from_iterator",
        "_aawm_alias_streaming",
        "_opencode_zen_direct_stream_terminal_error",
        "_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES",
        "ResponsesAPIOptionalRequestParams",
        "ProxyException",
        "StreamingResponse",
        "httpx",
        "litellm",
    )

    @staticmethod
    def _install_fake_langfuse(monkeypatch) -> MagicMock:
        fake_langfuse = MagicMock()
        monkeypatch.setattr(
            LangFuseHandler,
            "get_langfuse_logger_for_request",
            MagicMock(return_value=fake_langfuse),
        )
        monkeypatch.setattr(
            litellm,
            "_async_success_callback",
            [object.__new__(LangfusePromptManagement)],
        )
        monkeypatch.setattr(litellm, "callbacks", [])
        monkeypatch.setattr(litellm, "success_callback", [])
        return fake_langfuse

    @staticmethod
    def _model_response(model: str) -> ModelResponse:
        return ModelResponse(
            id=f"d1-574-{model}",
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "cost lifecycle"},
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    @staticmethod
    def _langfuse_callback_kwargs(
        fake_langfuse: MagicMock,
    ) -> list[dict[str, Any]]:
        return [
            call.kwargs["kwargs"]
            for call in fake_langfuse.log_event_on_langfuse.call_args_list
            if call.kwargs.get("kwargs") is not None
        ]

    @staticmethod
    def _logging_for_call_kwargs(call_kwargs: dict[str, Any]) -> Logging:
        setup_kwargs = dict(call_kwargs)
        setup_kwargs.setdefault("litellm_call_id", "d1-574-normal-cost")
        logging_obj, _ = litellm.utils.function_setup(
            original_function="acompletion",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.datetime.now(),
            **setup_kwargs,
        )
        return logging_obj

    def _isolated_route_host(
        self,
        monkeypatch,
        *,
        adapter_model: str,
        stream: bool,
        fake_acompletion,
    ) -> tuple[dict[str, Any], dict[str, Any], list[ModelResponseStream]]:
        captured: dict[str, Any] = {}
        streamed_chunks: list[ModelResponseStream] = []
        litellm_metadata = {"session_id": f"d1-574-{adapter_model}"}

        async def capturing_acompletion(**kwargs):
            captured.update(kwargs)
            return await fake_acompletion(**kwargs)

        normalized = MagicMock()
        normalized.request_body = {
            "model": f"opencode_zen/{adapter_model}",
            "stream": stream,
        }
        normalized.request_input = []
        normalized.responses_api_request = {}
        normalized.litellm_metadata = litellm_metadata
        normalized.completion_kwargs = {
            "model": adapter_model,
            "messages": [{"role": "user", "content": "test"}],
            "custom_llm_provider": "openai",
            "stream": stream,
            "metadata": litellm_metadata,
        }
        mock_norm = MagicMock()
        mock_norm.normalize_codex_request = AsyncMock(return_value=normalized)

        async def passthrough_sse(iterator, on_complete=None, **_kwargs):
            async for chunk in iterator.litellm_custom_stream_wrapper:
                streamed_chunks.append(chunk)
                yield b"data: {}\n\n"
            if on_complete is not None:
                on_complete()

        async def no_peek(response, **_kwargs):
            return SimpleNamespace(response=response)

        host_globals = dict(codex_candidate_calls.__dict__)
        host_globals.update(
            {
                "httpx": httpx,
                "litellm": litellm,
                "StreamingResponse": StreamingResponse,
                "ResponsesAPIOptionalRequestParams": dict,
                "ProxyException": ProxyException,
            }
        )
        codex_candidate_calls.install(host_globals)
        host_globals.update(
            {
                "_anthropic_opencode_zen_normalization": mock_norm,
                "_get_anthropic_opencode_zen_normalization_runtime": (
                    lambda: MagicMock()
                ),
                "_load_opencode_zen_api_key_for_candidate": AsyncMock(
                    return_value="test-key"
                ),
                "_get_opencode_zen_target_base": (
                    lambda: "https://opencode.ai/zen"
                ),
                "_join_opencode_zen_passthrough_url": (
                    lambda *, base_target_url, endpoint: (
                        f"{base_target_url.rstrip('/')}{endpoint}"
                    )
                ),
                "BaseOpenAIPassThroughHandler": SimpleNamespace(
                    _assemble_headers=lambda **_kwargs: {}
                ),
                "HttpPassThroughEndpointHelpers": SimpleNamespace(
                    validate_outgoing_egress=lambda **_kwargs: None
                ),
                "_annotate_request_scope_for_adapted_access_log": (
                    lambda *_args, **_kwargs: None
                ),
                "_build_adapted_route_rollup_kwargs": lambda _metadata: {},
                "_emit_adapted_route_access_log": lambda **_kwargs: None,
                "_get_proxy_shared_aiohttp_session": lambda: MagicMock(),
                "_opencode_zen_callback_headers": lambda _request: {},
                "_record_adapted_completed_route_rollup_turn": (
                    lambda *_args, **_kwargs: None
                ),
                "_maybe_raise_opencode_zen_direct_rate_limit": lambda _exc: None,
                "_opencode_zen_candidate_unavailable_detail": lambda _exc: None,
                "_serialize_responses_adapter_response": (
                    lambda _response: json.dumps({"id": "test-response"})
                ),
                "_is_codex_auto_agent_empty_success_responses_body": (
                    lambda _body: False
                ),
                "_build_responses_response_from_adapter_response": (
                    lambda _response: MagicMock()
                ),
                "_responses_sse_from_iterator": passthrough_sse,
                "_aawm_alias_streaming": SimpleNamespace(
                    peek_streaming_response=no_peek
                ),
                "_opencode_zen_direct_stream_terminal_error": lambda _exc: None,
                "_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES": 1024,
            }
        )
        monkeypatch.setattr(litellm, "acompletion", capturing_acompletion)
        return host_globals, captured, streamed_chunks

    async def _call_route(
        self,
        host_globals: dict[str, Any],
        *,
        adapter_model: str,
        use_alias_candidate_probe: bool = False,
    ):
        return await host_globals["_handle_codex_opencode_zen_adapter_route"](
            endpoint="/v1/responses",
            request=_make_request({}),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": f"opencode_zen/{adapter_model}",
            },
            adapter_model=adapter_model,
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    def test_isolated_route_host_does_not_pollute_installed_globals(
        self, monkeypatch
    ):
        installed_globals = (
            codex_candidate_calls._handle_codex_opencode_zen_adapter_route.__globals__
        )
        missing = object()
        before = {
            name: installed_globals.get(name, missing)
            for name in self._ISOLATED_HOST_NAMES
        }

        async def fake_acompletion(**_kwargs):
            return self._model_response("big-pickle")

        isolated_globals, _, _ = self._isolated_route_host(
            monkeypatch,
            adapter_model="big-pickle",
            stream=False,
            fake_acompletion=fake_acompletion,
        )

        assert isolated_globals is not installed_globals
        for name, original in before.items():
            if original is missing:
                assert name not in installed_globals
            else:
                assert installed_globals[name] is original

    @pytest.mark.asyncio
    async def test_known_free_nonstream_cost_is_callback_visible_before_return(
        self, monkeypatch
    ):
        fake_langfuse = self._install_fake_langfuse(monkeypatch)
        response = self._model_response("big-pickle")

        async def fake_acompletion(**kwargs):
            logging_obj = kwargs["litellm_logging_obj"]
            assert logging_obj.model_call_details["response_cost"] == 0.0
            now = datetime.datetime.now()
            await logging_obj.async_success_handler(
                result=response,
                start_time=now,
                end_time=now,
                cache_hit=False,
            )
            return response

        host_globals, captured, _ = self._isolated_route_host(
            monkeypatch,
            adapter_model="big-pickle",
            stream=False,
            fake_acompletion=fake_acompletion,
        )
        await self._call_route(host_globals, adapter_model="big-pickle")

        callback_kwargs = self._langfuse_callback_kwargs(fake_langfuse)
        assert len(callback_kwargs) == 1
        assert callback_kwargs[0]["response_cost"] == 0.0
        assert captured["litellm_logging_obj"].model_call_details[
            "response_cost"
        ] == 0.0
        assert response._hidden_params["response_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_unknown_nonstream_cost_remains_null(self, monkeypatch):
        fake_langfuse = self._install_fake_langfuse(monkeypatch)
        monkeypatch.setattr(
            litellm,
            "response_cost_calculator",
            lambda **_kwargs: None,
        )
        response = self._model_response("not-free")

        async def fake_acompletion(**kwargs):
            assert "litellm_logging_obj" not in kwargs
            logging_obj = self._logging_for_call_kwargs(kwargs)
            now = datetime.datetime.now()
            await logging_obj.async_success_handler(
                result=response,
                start_time=now,
                end_time=now,
                cache_hit=False,
            )
            return response

        host_globals, captured, _ = self._isolated_route_host(
            monkeypatch,
            adapter_model="not-free",
            stream=False,
            fake_acompletion=fake_acompletion,
        )
        await self._call_route(host_globals, adapter_model="not-free")

        callback_kwargs = self._langfuse_callback_kwargs(fake_langfuse)
        assert len(callback_kwargs) == 1
        assert callback_kwargs[0]["response_cost"] is None
        assert "litellm_logging_obj" not in captured
        assert "response_cost" not in response._hidden_params

    @pytest.mark.asyncio
    async def test_known_free_stream_preserves_zero_in_chunks_and_callback(
        self, monkeypatch
    ):
        fake_langfuse = self._install_fake_langfuse(monkeypatch)

        async def fake_acompletion(**kwargs):
            logging_obj = kwargs["litellm_logging_obj"]
            assert logging_obj.model_call_details["response_cost"] == 0.0

            async def completion_stream():
                yield ModelResponseStream(
                    id="d1-574-stream",
                    model="big-pickle",
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(role="assistant", content="free"),
                            finish_reason=None,
                        )
                    ],
                )

            return litellm.CustomStreamWrapper(
                completion_stream=completion_stream(),
                model="big-pickle",
                logging_obj=logging_obj,
                custom_llm_provider="openai",
            )

        host_globals, _, streamed_chunks = self._isolated_route_host(
            monkeypatch,
            adapter_model="big-pickle",
            stream=True,
            fake_acompletion=fake_acompletion,
        )
        streaming_response = await self._call_route(
            host_globals,
            adapter_model="big-pickle",
        )
        _ = [chunk async for chunk in streaming_response.body_iterator]

        for _attempt in range(50):
            if fake_langfuse.log_event_on_langfuse.call_count:
                break
            await asyncio.sleep(0.01)

        assert streamed_chunks
        assert all(
            chunk._hidden_params.get("response_cost") == 0.0
            for chunk in streamed_chunks
        )
        callback_kwargs = self._langfuse_callback_kwargs(fake_langfuse)
        assert callback_kwargs
        assert all(
            callback_kwargs_item["response_cost"] == 0.0
            for callback_kwargs_item in callback_kwargs
        )

    @pytest.mark.asyncio
    async def test_known_free_alias_probe_remains_unpriced(self, monkeypatch):
        response = self._model_response("big-pickle")

        async def fake_acompletion(**kwargs):
            assert "litellm_logging_obj" not in kwargs
            return response

        host_globals, captured, _ = self._isolated_route_host(
            monkeypatch,
            adapter_model="big-pickle",
            stream=False,
            fake_acompletion=fake_acompletion,
        )
        await self._call_route(
            host_globals,
            adapter_model="big-pickle",
            use_alias_candidate_probe=True,
        )

        assert "litellm_logging_obj" not in captured
        assert "response_cost" not in response._hidden_params
