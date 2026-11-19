import os
import sys
from unittest.mock import MagicMock

import pytest
from litellm.exceptions import UnsupportedParamsError

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import (
    COHERE_V2_NAMESPACE_FUNCTION_MAP_KEY,
    CohereV2ChatConfig,
    translate_cohere_v2_responses_input,
    translate_cohere_v2_responses_request,
    translate_cohere_v2_responses_tools,
    translate_cohere_v2_tool_choice,
)


class TestCohereTransform:
    def setup_method(self):
        self.config = CohereChatConfig()
        self.model = "command-r-plus-latest"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_cohere_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}


class TestCohereV2NamespaceCollisionTranslation:
    def _tools(self) -> list[dict]:
        return [
            {
                "type": "namespace",
                "name": "mcp__codex_apps__github",
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
            {
                "type": "namespace",
                "name": "mcp__codex_apps__gmail",
                "tools": [
                    {
                        "type": "function",
                        "name": "_get_profile",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
            {
                "type": "function",
                "name": "_get_profile",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def test_wire_names_preserve_basenames_and_bare_identity(self) -> None:
        translated, namespace_by_name = translate_cohere_v2_responses_tools(
            self._tools()
        )
        assert translated[0]["name"] == "mcp__codex_apps__github___get_profile"
        assert translated[1]["name"] == "mcp__codex_apps__gmail___get_profile"
        assert translated[2] == self._tools()[2]
        assert namespace_by_name == {
            ("_get_profile", "mcp__codex_apps__github"): "mcp__codex_apps__github",
            ("_get_profile", "mcp__codex_apps__gmail"): "mcp__codex_apps__gmail",
            ("_get_profile", None): None,
        }

    def test_direct_chat_keeps_legacy_names_and_rejects_wire_collisions(self) -> None:
        config = CohereV2ChatConfig()
        single_namespace = [
            {
                "type": "namespace",
                "name": "ops",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        ]
        body = config.transform_request(
            model="command-r-plus-latest",
            messages=[],
            optional_params={"tools": single_namespace},
            litellm_params={},
            headers={},
        )
        assert body["tools"][0]["function"]["name"] == "run"

        ambiguous_namespaces = [
            single_namespace[0],
            {
                "type": "namespace",
                "name": "admin",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        ]
        with pytest.raises(UnsupportedParamsError, match="reuses function name"):
            config.transform_request(
                model="command-r-plus-latest",
                messages=[],
                optional_params={"tools": ambiguous_namespaces},
                litellm_params={},
                headers={},
            )

    def test_flat_namespace_tool_and_history_use_the_same_wire_name(self) -> None:
        request = {
            "tools": [
                {
                    "type": "function",
                    "namespace": "ops",
                    "name": "run",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "input": [
                {
                    "type": "function_call",
                    "name": "run",
                    "namespace": "ops",
                    "call_id": "call_run",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "namespace": "ops",
                    "call_id": "call_run",
                    "output": "ok",
                },
            ],
        }

        translated, namespace_by_name = translate_cohere_v2_responses_request(request)

        assert translated["tools"][0]["name"] == "ops__run"
        assert translated["tools"][0]["name"] == translated["input"][0]["name"]
        assert translated["input"][0]["call_id"] == "call_run"
        assert "namespace" not in translated["input"][0]
        assert translated["input"][1] == {
            "type": "function_call_output",
            "call_id": "call_run",
            "output": "ok",
        }
        assert namespace_by_name == {("run", "ops"): "ops"}

    def test_rejects_actual_wire_name_collision(self) -> None:
        tools = [
            {
                "type": "function",
                "name": "ops__run",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "namespace",
                "name": "ops",
                "tools": [
                    {
                        "type": "function",
                        "name": "run",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        ]
        with pytest.raises(UnsupportedParamsError, match="reuses function name"):
            translate_cohere_v2_responses_tools(tools)

    def test_rejects_ambiguous_basename_tool_choice(self) -> None:
        _, namespace_by_name = translate_cohere_v2_responses_tools(self._tools())
        choice = {
            "type": "function",
            "namespace": "mcp__codex_apps__github",
            "name": "_get_profile",
        }
        with pytest.raises(Exception, match="not an advertised translated function"):
            translate_cohere_v2_tool_choice(choice, namespace_by_name)

    def test_matches_qualified_tool_choice_then_rejects_namespace_selection(self) -> None:
        _, namespace_by_name = translate_cohere_v2_responses_tools(self._tools())
        choice = {
            "type": "function",
            "namespace": "mcp__codex_apps__github",
            "name": "mcp__codex_apps__github._get_profile",
        }
        with pytest.raises(Exception, match="namespace-specific named selection"):
            translate_cohere_v2_tool_choice(choice, namespace_by_name)

    def test_responses_input_matches_identity_and_qualifies_call_name(self) -> None:
        _, namespace_by_name = translate_cohere_v2_responses_tools(self._tools())
        original_items = [
            {
                "type": "function_call",
                "name": "_get_profile",
                "namespace": "mcp__codex_apps__github",
                "call_id": "call_github",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_github",
                "namespace": "mcp__codex_apps__github",
                "output": "ok",
            },
        ]
        translated = translate_cohere_v2_responses_input(
            original_items,
            namespace_by_name,
        )
        assert translated[0]["name"] == "mcp__codex_apps__github___get_profile"
        assert translated[0]["call_id"] == "call_github"
        assert translated[0]["arguments"] == "{}"
        assert "namespace" not in translated[0]
        assert translated[1] == {
            "type": "function_call_output",
            "call_id": "call_github",
            "output": "ok",
        }
        assert original_items[0]["name"] == "_get_profile"

    def test_request_metadata_is_identity_keyed_and_unchanged_without_translation(self) -> None:
        request_body = {"tools": [], "input": []}
        translated_request, namespace_by_name = translate_cohere_v2_responses_request(request_body)
        assert translated_request is request_body
        assert namespace_by_name == {}
        assert COHERE_V2_NAMESPACE_FUNCTION_MAP_KEY not in translated_request.get("litellm_metadata", {})
