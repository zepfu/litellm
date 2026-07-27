"""Wave 6A Author D ownership and behavior tests: payload_validation extraction.

Enforces the behavior-preserving extraction contract from
``llm_passthrough_endpoints.py`` (the "god module") into
``aawm_adapter_runtime/payload_validation.py``.

Covers:
- AST symbol inventory + ordering of the moved detector/raise/validate band.
- Dependency isolation: no god-module import at module scope, no wildcard
  imports, host-global seams resolved via ``install()`` rebinding.
- Production object-identity facade contract with the god module.
- Fixed behavior for the pure detectors and the exception taxonomy.
- Replay/buffer bound constants (5,000 chunks / 8 MiB) preserved.

Write scope: this file only.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from litellm.proxy._types import ProxyException
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    payload_validation as pv,
    sse as sse_mod,
)

GOD_PATH = Path(lpe.__file__).resolve()
TARGET_PATH = Path(pv.__file__).resolve()

# ---------------------------------------------------------------------------
# Symbol inventory (ordered as required by the ownership scope).
# ---------------------------------------------------------------------------
EXPECTED_SYMBOLS_ORDER: tuple[str, ...] = (
    "_is_codex_auto_agent_malformed_tool_call_text_output",
    "_validate_alias_candidate_responses_stream_if_needed",
    "_build_malformed_intake_context_for_anthropic_responses_adapter",
    "_is_codex_auto_agent_empty_success_responses_body",
    "_coerce_optional_int",
    "_usage_has_no_more_than_one_output_token",
    "_model_response_usage_dict",
    "_is_codex_google_code_assist_empty_success_model_response",
    "_raise_codex_auto_agent_empty_success_response",
    "_build_failed_responses_diagnostic",
    "_raise_codex_auto_agent_malformed_tool_call_text_payload",
    "_raise_codex_auto_agent_failed_responses_payload",
    "_raise_responses_adapter_failed_response",
    "_validate_codex_auto_agent_responses_payload",
)
EXPECTED_SYMBOLS: frozenset[str] = frozenset(EXPECTED_SYMBOLS_ORDER)

# Explicitly excluded: belongs to stream_collect, not payload_validation.
EXCLUDED_SYMBOL = "_build_empty_success_responses_diagnostic"
HOST_GLOBAL_SEAMS = {
    "_mapping_or_attr_get",
    "_decode_http_response_body",
    "_responses_sse_from_repaired_response_body",
    "_build_empty_success_responses_diagnostic",
}
def _module_function_defs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

# ---------------------------------------------------------------------------
# AST / structural ownership
# ---------------------------------------------------------------------------
class TestPayloadValidationAST:
    def test_all_expected_symbols_defined(self):
        defined = set(_module_function_defs(TARGET_PATH))
        missing = EXPECTED_SYMBOLS - defined
        assert not missing, f"payload_validation missing symbols: {sorted(missing)}"

    def test_excluded_symbol_not_defined_here(self):
        defined = set(_module_function_defs(TARGET_PATH))
        assert EXCLUDED_SYMBOL not in defined, (
            f"{EXCLUDED_SYMBOL} belongs to stream_collect, must not be defined "
            "in payload_validation"
        )

    def test_host_global_seams_not_defined_here(self):
        defined = set(_module_function_defs(TARGET_PATH))
        assert not HOST_GLOBAL_SEAMS & defined

    def test_host_function_names_matches_inventory(self):
        assert set(pv._HOST_FUNCTION_NAMES) == EXPECTED_SYMBOLS
        # install() must cover exactly the owned inventory.
        assert len(pv._HOST_FUNCTION_NAMES) == len(EXPECTED_SYMBOLS)

    def test_owned_symbol_order_and_god_module_cutover(self):
        target_defs = _module_function_defs(TARGET_PATH)
        target_order = [s for s in target_defs if s in EXPECTED_SYMBOLS]
        assert target_order == list(EXPECTED_SYMBOLS_ORDER)
        assert not EXPECTED_SYMBOLS & set(_module_function_defs(GOD_PATH))

    def test_replay_bound_constants_referenced(self):
        src = TARGET_PATH.read_text(encoding="utf-8")
        assert "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS" in src
        assert "_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES" in src

    def test_god_module_replay_bounds_values(self):
        assert lpe._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS == 5000
        assert lpe._AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES == 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Dependency isolation
# ---------------------------------------------------------------------------
class TestPayloadValidationDependencyIsolation:
    def test_no_god_module_import_at_module_scope(self):
        tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "llm_passthrough_endpoints" in alias.name:
                        violations.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "llm_passthrough_endpoints" in mod:
                    violations.append(mod)
        assert not violations, (
            f"payload_validation imports the god module at module scope: {violations}"
        )

    def test_no_wildcard_imports(self):
        tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(a.name == "*" for a in node.names), (
                    "payload_validation must not use wildcard imports"
                )

    def test_host_global_seams_are_type_checking_or_runtime_bound(self):
        """Host-global functions must not be imported at runtime.

        They are declared under TYPE_CHECKING and resolved via install().
        """
        tree = ast.parse(TARGET_PATH.read_text(encoding="utf-8"))
        runtime_imported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    runtime_imported.add(alias.asname or alias.name)
        host_seams = {
            "_build_malformed_tool_call_intake_context",
            "_is_empty_success_responses_body",
            "_is_failed_responses_body",
            "_should_log_aawm_alias_routing_event",
            "_mapping_or_attr_get",
            "_decode_http_response_body",
            "_collect_responses_response_from_stream",
            "_responses_sse_from_repaired_response_body",
            "_build_empty_success_responses_diagnostic",
            "_anthropic_google_shaping",
            "_aawm_alias_streaming",
        }
        leaked = host_seams & runtime_imported
        assert not leaked, f"host-global seams imported at runtime: {sorted(leaked)}"


# ---------------------------------------------------------------------------
# Production facade / object identity
# ---------------------------------------------------------------------------
class TestPayloadValidationInstallContract:
    def test_production_import_publishes_rebound_objects_to_god(self):
        for name in EXPECTED_SYMBOLS_ORDER:
            god_obj = getattr(lpe, name, None)
            pv_obj = getattr(pv, name, None)
            assert god_obj is not None, f"{name} not published to god module"
            assert pv_obj is not None, f"{name} not present in payload_validation"
            assert god_obj is pv_obj, f"{name} identity mismatch"

    def test_production_functions_use_god_namespace(self):
        rebound = pv._usage_has_no_more_than_one_output_token
        assert rebound.__globals__ is vars(lpe)


# ---------------------------------------------------------------------------
# Behavioral parity: pure detectors
# ---------------------------------------------------------------------------
class TestPureDetectors:
    def test_mapping_or_attr_get_dict(self):
        # _mapping_or_attr_get is owned by aawm_adapter_runtime.sse, not
        # payload_validation; call through the extracted owner directly.
        assert sse_mod._mapping_or_attr_get({"a": 1}, "a") == 1
        assert sse_mod._mapping_or_attr_get({"a": 1}, "b", 9) == 9

    def test_mapping_or_attr_get_object(self):
        obj = SimpleNamespace(a=5)
        assert sse_mod._mapping_or_attr_get(obj, "a") == 5
        assert sse_mod._mapping_or_attr_get(obj, "b", 7) == 7

    def test_coerce_optional_int(self):
        assert pv._coerce_optional_int(None) is None
        assert pv._coerce_optional_int("3") == 3
        assert pv._coerce_optional_int(4.0) == 4
        assert pv._coerce_optional_int("nope") is None

    @pytest.mark.parametrize(
        "usage,expected",
        [
            (None, True),
            ({}, False),
            ({"completion_tokens": 0}, True),
            ({"completion_tokens": 1}, True),
            ({"completion_tokens": 2}, False),
            ({"output_tokens": 1}, True),
            ({"output_tokens": 5}, False),
            ({"total_tokens": 0}, True),
            ({"total_tokens": 3}, False),
            ({"prompt_tokens": 10}, False),
        ],
    )
    def test_usage_has_no_more_than_one_output_token(self, usage, expected):
        assert pv._usage_has_no_more_than_one_output_token(usage) is expected

    def test_model_response_usage_dict_none(self):
        assert pv._model_response_usage_dict(None) == {}

    def test_model_response_usage_dict_from_dict(self):
        src = {"prompt_tokens": 2, "completion_tokens": 3}
        out = pv._model_response_usage_dict(src)
        assert out == src
        assert out is not src  # copied

    def test_model_response_usage_dict_model_dump(self):
        class Usage:
            def model_dump(self, exclude_none=True):
                return {"total_tokens": 9}

        assert pv._model_response_usage_dict(Usage()) == {"total_tokens": 9}

    def test_model_response_usage_dict_attr_fallback(self):
        class Usage:
            prompt_tokens = 1
            completion_tokens = None
            total_tokens = 4
            output_tokens = 2

        out = pv._model_response_usage_dict(Usage())
        assert out == {"prompt_tokens": 1, "total_tokens": 4, "output_tokens": 2}


class TestMalformedToolCallTextDetector:
    def test_non_list_output_false(self):
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output({"output": "x"}) is False
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output({}) is False

    def test_composer_call_function_name_true(self):
        body = {"output": [{"type": "function_call", "name": " Composer_Call "}]}
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output(body) is True

    def test_mcp_call_composer_name_true(self):
        body = {"output": [{"type": "mcp_call", "name": "composer_call"}]}
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output(body) is True

    def test_benign_function_call_false(self):
        body = {"output": [{"type": "function_call", "name": "read_file"}]}
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output(body) is False

    def test_message_string_content_malformed_marker(self, monkeypatch):
        fn_ns = pv._is_codex_auto_agent_malformed_tool_call_text_output.__globals__
        monkeypatch.setitem(fn_ns, "is_malformed_composer_call_literal_text", lambda t: "BAD" in t)
        monkeypatch.setitem(
            fn_ns, "is_malformed_grok_literal_tool_label_transcript_text", lambda t: False
        )
        body = {"output": [{"type": "message", "content": "this is BAD text"}]}
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output(body) is True

    def test_message_part_list_malformed_marker(self, monkeypatch):
        fn_ns = pv._is_codex_auto_agent_malformed_tool_call_text_output.__globals__
        monkeypatch.setitem(fn_ns, "is_malformed_composer_call_literal_text", lambda t: False)
        monkeypatch.setitem(
            fn_ns, "is_malformed_grok_literal_tool_label_transcript_text", lambda t: "GROK" in t
        )
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "GROK marker"}],
                }
            ]
        }
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output(body) is True

    def test_message_benign_content_false(self, monkeypatch):
        fn_ns = pv._is_codex_auto_agent_malformed_tool_call_text_output.__globals__
        monkeypatch.setitem(fn_ns, "is_malformed_composer_call_literal_text", lambda t: False)
        monkeypatch.setitem(
            fn_ns, "is_malformed_grok_literal_tool_label_transcript_text", lambda t: False
        )
        body = {"output": [{"type": "message", "content": "hello world"}]}
        assert pv._is_codex_auto_agent_malformed_tool_call_text_output(body) is False


# ---------------------------------------------------------------------------
# Behavioral parity: exception taxonomy
# ---------------------------------------------------------------------------
class TestExceptionTaxonomy:
    def _body(self) -> dict:
        return {
            "id": "resp_1",
            "status": "failed",
            "model": "m",
            "error": {"code": "x"},
            "output": [{"type": "message"}, {"type": "function_call"}],
        }

    def test_build_failed_responses_diagnostic_shape(self):
        diag = pv._build_failed_responses_diagnostic(
            response_body=self._body(),
            adapter="adp",
            adapter_model="m",
            stream_event_summaries=[{"e": 1}],
        )
        assert diag["adapter"] == "adp"
        assert diag["adapter_model"] == "m"
        assert diag["response_id"] == "resp_1"
        assert diag["status"] == "failed"
        assert diag["output_count"] == 2
        assert diag["output_types"] == ["message", "function_call"]
        assert diag["stream_events"] == [{"e": 1}]

    def test_build_failed_responses_diagnostic_no_stream_key_when_none(self):
        diag = pv._build_failed_responses_diagnostic(
            response_body=self._body(), adapter="a", adapter_model="m"
        )
        assert "stream_events" not in diag

    def test_raise_failed_responses_payload_taxonomy(self):
        with pytest.raises(ProxyException) as excinfo:
            pv._raise_codex_auto_agent_failed_responses_payload(
                response_body=self._body(),
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
            )
        exc = excinfo.value
        assert exc.type == "upstream_error"
        assert exc.code == "502"
        detail = exc.detail
        assert detail["error"]["code"] == "aawm_auto_agent_failed_responses_payload"
        assert detail["error"]["status"] == "RESPONSES_STATUS_FAILED"
        assert detail["error"]["type"] == "upstream_error"
        assert detail["diagnostic"]["adapter"] == "adp"

    def test_raise_malformed_tool_call_text_taxonomy(self, monkeypatch):
        calls = {}
        fn_ns = pv._raise_codex_auto_agent_malformed_tool_call_text_payload.__globals__
        monkeypatch.setitem(
            fn_ns,
            "schedule_persist_malformed_tool_call_detection",
            lambda **kw: calls.update(kw),
        )
        with pytest.raises(ProxyException) as excinfo:
            pv._raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body=self._body(),
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
                intake_context={"session_id": "s"},
            )
        exc = excinfo.value
        assert exc.type == "invalid_request_error"
        assert exc.code == "502"
        assert exc.detail["error"]["code"] == "aawm_auto_agent_malformed_tool_call_text"
        assert exc.detail["error"]["status"] == "RESPONSES_MALFORMED_TOOL_CALL"
        assert calls["adapter"] == "adp"
        assert calls["intake_context"] == {"session_id": "s"}

    def test_raise_malformed_intake_failure_is_swallowed(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("disk down")

        fn_ns = pv._raise_codex_auto_agent_malformed_tool_call_text_payload.__globals__
        monkeypatch.setitem(fn_ns, "schedule_persist_malformed_tool_call_detection", boom)
        # Must still raise the ProxyException, not the intake error.
        with pytest.raises(ProxyException):
            pv._raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body=self._body(),
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
            )

    def test_raise_responses_adapter_failed_non_retryable_http(self):
        with pytest.raises(HTTPException) as excinfo:
            pv._raise_responses_adapter_failed_response(
                response_body=self._body(),
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
            )
        assert excinfo.value.status_code == 502
        assert "diagnostic" in excinfo.value.detail

    def test_raise_responses_adapter_failed_retryable_uses_proxy_exception(self):
        with pytest.raises(ProxyException) as excinfo:
            pv._raise_responses_adapter_failed_response(
                response_body=self._body(),
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
                retryable_alias_candidate=True,
            )
        assert excinfo.value.detail["error"]["code"] == (
            "aawm_auto_agent_failed_responses_payload"
        )

    def test_raise_empty_success_response_taxonomy(self, monkeypatch):
        fn_ns = pv._raise_codex_auto_agent_empty_success_response.__globals__
        monkeypatch.setitem(
            fn_ns,
            "_build_empty_success_responses_diagnostic",
            lambda *, response_body, diagnostic_context: {"ctx": diagnostic_context},
        )
        with pytest.raises(ProxyException) as excinfo:
            pv._raise_codex_auto_agent_empty_success_response(
                response_body=self._body(),
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
                stream_event_summaries=[{"e": 1}],
            )
        exc = excinfo.value
        assert exc.type == "upstream_error"
        assert exc.detail["error"]["code"] == "aawm_codex_auto_agent_empty_success"
        assert exc.detail["error"]["status"] == "EMPTY_SUCCESS_RESPONSE"
        ctx = exc.detail["diagnostic"]["ctx"]
        assert ctx["adapter"] == "adp"
        assert ctx["stream_events"] == [{"e": 1}]


# ---------------------------------------------------------------------------
# Empty-success body detector through the production host seam
# ---------------------------------------------------------------------------
class TestEmptySuccessBodyDetector:
    def test_empty_success_responses_body(self):
        body = {"status": "completed", "output": [], "usage": {"output_tokens": 1}}
        assert pv._is_codex_auto_agent_empty_success_responses_body(body) is True

    def test_empty_success_requires_output_tokens(self):
        body = {"status": "completed", "output": [], "usage": {}}
        assert pv._is_codex_auto_agent_empty_success_responses_body(body) is False

    def test_empty_success_more_than_one_token_false(self):
        body = {"status": "completed", "output": [], "usage": {"output_tokens": 5}}
        assert pv._is_codex_auto_agent_empty_success_responses_body(body) is False


# ---------------------------------------------------------------------------
# Behavioral parity: stream validation disabled passthrough
# ---------------------------------------------------------------------------
class TestValidateAliasCandidateStream:
    def test_disabled_returns_same_response(self):
        async def run():
            resp = StreamingResponse(iter([b"data: {}\n\n"]))
            out = await pv._validate_alias_candidate_responses_stream_if_needed(
                resp,
                enabled=False,
                adapter_model="m",
                adapter="adp",
                adapter_label="Label",
            )
            return out is resp

        assert asyncio.run(run()) is True


# ---------------------------------------------------------------------------
# Signature contracts preserved
# ---------------------------------------------------------------------------
class TestSignatureContracts:
    @pytest.mark.parametrize("name", sorted(EXPECTED_SYMBOLS))
    def test_facades_preserve_declared_names(self, name):
        assert getattr(pv, name).__name__ == name

    def test_async_flags_match_contract(self):
        async_symbols = {
            "_validate_alias_candidate_responses_stream_if_needed",
            "_validate_codex_auto_agent_responses_payload",
        }
        for name in EXPECTED_SYMBOLS_ORDER:
            assert inspect.iscoroutinefunction(getattr(pv, name)) is (
                name in async_symbols
            ), f"{name} async-ness diverged"
