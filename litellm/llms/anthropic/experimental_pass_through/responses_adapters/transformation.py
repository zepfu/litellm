"""
Transformation layer: Anthropic /v1/messages <-> OpenAI Responses API.

This module owns all *request/response body* conversions for the direct
v1/messages -> Responses API path used for OpenAI and Azure models.

Streaming Anthropic SSE reconstruction (``event: content_block_*``) lives in
``responses_adapters/streaming_iterator.py`` and the parallel Chat Completions
tree under ``adapters/streaming_iterator.py``. This transformation module is
not an SSE emitter; shared streaming-emitter consolidation belongs there, not
here. Shared request helpers already come from ``adapters.observability`` and
``adapters.transformation`` (e.g. ``derive_prompt_cache_key``,
``sanitize_anthropic_tool_use_input``).
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Union, cast

from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthopicMessagesAssistantMessageParam,
    AnthropicFinishReason,
    AnthropicMcpServerTool,
    AnthropicMcpToolset,
    AnthropicMessagesRequest,
    AnthropicMessagesToolChoice,
    AnthropicMessagesUserMessageParam,
    AnthropicResponseContentBlockMcpToolResult,
    AnthropicResponseContentBlockMcpToolUse,
    AnthropicResponseContentBlockText,
    AnthropicResponseContentBlockThinking,
    AnthropicResponseContentBlockToolUse,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.utils import supports_reasoning_summary

from ..adapters.observability import (
    derive_prompt_cache_key,
    normalize_reasoning_effort_for_provider,
    provider_cache_intent_metadata,
    request_contains_cache_control,
)
from ..adapters.transformation import (
    sanitize_anthropic_tool_use_id,
    sanitize_anthropic_tool_use_input,
)


class AnthropicResponsesProviderError(RuntimeError):
    pass


def _response_attr(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    value = getattr(response, key, default)
    if type(value).__module__.startswith("unittest.mock"):
        return default
    return value


def summarize_failed_responses_response(response: Any) -> Dict[str, Any]:
    output = _response_attr(response, "output", []) or []
    return {
        "response_id": _response_attr(response, "id"),
        "status": _response_attr(response, "status"),
        "model": _response_attr(response, "model"),
        "error": _response_attr(response, "error"),
        "incomplete_details": _response_attr(response, "incomplete_details"),
        "output_count": len(output) if isinstance(output, list) else 0,
        "output_types": [_response_attr(item, "type") for item in output[:20]]
        if isinstance(output, list)
        else [],
    }


def is_failed_responses_response(response: Any) -> bool:
    return (
        _response_attr(response, "status") == "failed"
        or _response_attr(response, "error") is not None
    )


def raise_for_failed_responses_response(
    response: Any, *, context: str = "Responses API"
) -> None:
    if not is_failed_responses_response(response):
        return
    diagnostic = summarize_failed_responses_response(response)
    raise AnthropicResponsesProviderError(
        f"{context} returned a failed response: "
        + json.dumps(diagnostic, default=str, ensure_ascii=False, sort_keys=True)
    )


def _bound_prompt_cache_key(value: str, *, prefix: str = "anthropic-cache") -> str:
    """Return an OpenAI-safe prompt_cache_key (<= 64 chars).

    Caller-supplied keys may exceed OpenAI's length limit. Re-deriving via
    ``derive_prompt_cache_key({"prompt_cache_key": ...})`` is incorrect because
    that helper only hashes stable system/tools roots (RR-026/RR-032). Hash the
    raw key string instead when truncation is required.
    """
    cleaned = value.strip()
    if len(cleaned) <= 64:
        return cleaned
    cleaned_prefix = prefix.strip()[:23] or "cache"
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:40]
    return f"{cleaned_prefix}-{digest}"[:64]


def _resolve_responses_prompt_cache_key(anthropic_request: Any) -> Optional[str]:
    """Derive a *stable* OpenAI prompt_cache_key for Responses adaptation.

    Preference order:
    1. Explicit request ``prompt_cache_key`` (bounded).
    2. Shared ``derive_prompt_cache_key`` over system/tools only — never
       message/turn content — so multi-turn Claude Code cache breakpoints do
       not churn the key every request.

    Returns None when only volatile (message-level) cache_control is present.
    Omitting the key is preferred over a per-turn volatile key, which can
    shard cache affinity worse than OpenAI's default prefix routing.
    """
    if isinstance(anthropic_request, dict):
        explicit = anthropic_request.get("prompt_cache_key")
    else:
        explicit = getattr(anthropic_request, "get", lambda *_a, **_k: None)(
            "prompt_cache_key"
        )
        if explicit is None and hasattr(anthropic_request, "prompt_cache_key"):
            explicit = getattr(anthropic_request, "prompt_cache_key", None)
    if isinstance(explicit, str) and explicit.strip():
        return _bound_prompt_cache_key(explicit)
    derived = derive_prompt_cache_key(anthropic_request)
    if isinstance(derived, str) and derived.strip():
        return _bound_prompt_cache_key(derived)
    return None


class LiteLLMAnthropicToResponsesAPIAdapter:
    """
    Converts Anthropic /v1/messages requests to OpenAI Responses API format and
    converts Responses API responses back to Anthropic format.
    """

    # ------------------------------------------------------------------ #
    # Request translation: Anthropic -> Responses API                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_anthropic_image_source_to_url(source: dict) -> Optional[str]:
        """Convert Anthropic image source to a URL string."""
        source_type = source.get("type")
        if source_type == "base64":
            media_type = source.get("media_type", "image/jpeg")
            data = source.get("data", "")
            return f"data:{media_type};base64,{data}" if data else None
        elif source_type == "url":
            return source.get("url")
        return None

    @staticmethod
    def _anthropic_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(filter(None, parts))
        return str(content)

    @staticmethod
    def _responses_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(json.dumps(item))
                else:
                    parts.append(str(item))
            return "\n".join(filter(None, parts))
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    @staticmethod
    def _anthropic_thinking_block_to_reasoning_item(
        block: Dict[str, Any],
        reasoning_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert Anthropic thinking blocks into hidden Responses reasoning items.

        The conservative choice is to preserve the hidden block as a reasoning
        item rather than surface it as assistant `output_text`.
        """
        block_type = block.get("type")
        if block_type not in ("thinking", "redacted_thinking"):
            return None

        thinking_text = block.get("thinking", "")
        encrypted_content = (
            block.get("data", "") if block_type == "redacted_thinking" else ""
        )

        if not thinking_text and not encrypted_content:
            return None

        reasoning_item: Dict[str, Any] = {
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [{"type": "summary_text", "text": ""}],
        }

        if thinking_text:
            reasoning_item["content"] = [
                {"type": "reasoning_text", "text": thinking_text}
            ]
        if encrypted_content:
            reasoning_item["encrypted_content"] = encrypted_content

        return reasoning_item

    @staticmethod
    def _responses_reasoning_id_from_anthropic_block(
        block: Dict[str, Any],
    ) -> Optional[str]:
        for key in ("id", "signature"):
            value = block.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value.startswith("rs"):
                return value
        return None

    @staticmethod
    def _deserialize_tool_input(arguments: Any) -> Dict[str, Any]:
        if arguments in (None, ""):
            return {}
        if isinstance(arguments, dict):
            return arguments
        if hasattr(arguments, "model_dump"):
            dumped = arguments.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _codex_native_exec_command_parameters() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "login": {
                    "type": "boolean",
                    "description": "Run the command through a login shell.",
                },
                "tty": {
                    "type": "boolean",
                    "description": "Allocate a TTY for the command.",
                },
                "yield_time_ms": {
                    "type": "integer",
                    "description": "Milliseconds to wait before yielding output.",
                },
                "max_output_tokens": {
                    "type": "integer",
                    "description": "Maximum output tokens to return.",
                },
            },
            "required": ["cmd"],
            "additionalProperties": False,
        }

    @staticmethod
    def _codex_native_tool_name_to_responses(
        tool_name: Any,
        *,
        use_codex_native_tools: bool,
    ) -> Any:
        if use_codex_native_tools and tool_name == "Bash":
            return "exec_command"
        return tool_name

    @staticmethod
    def codex_native_tool_name_from_responses(
        tool_name: Any,
        *,
        use_codex_native_tools: bool,
    ) -> Any:
        if use_codex_native_tools and tool_name == "exec_command":
            return "Bash"
        return tool_name

    @staticmethod
    def _codex_native_tool_input_to_responses(
        tool_name: Any,
        tool_input: Any,
        *,
        use_codex_native_tools: bool,
    ) -> Any:
        if not use_codex_native_tools or tool_name != "Bash":
            return tool_input
        if not isinstance(tool_input, dict):
            return tool_input

        mapped: Dict[str, Any] = {}
        command = tool_input.get("command")
        if command is not None:
            mapped["cmd"] = command
        for key in ("login", "tty", "yield_time_ms", "max_output_tokens"):
            if key in tool_input:
                mapped[key] = tool_input[key]
        return mapped

    @staticmethod
    def codex_native_tool_input_from_responses(
        tool_name: Any,
        tool_input: Any,
        *,
        use_codex_native_tools: bool,
    ) -> Any:
        if not use_codex_native_tools or tool_name != "exec_command":
            return tool_input
        if not isinstance(tool_input, dict):
            return tool_input

        mapped: Dict[str, Any] = {}
        command = tool_input.get("cmd")
        if command is not None:
            mapped["command"] = command
        for key, value in tool_input.items():
            if key in {"cmd", "login", "tty", "yield_time_ms", "max_output_tokens"}:
                continue
            mapped[key] = value
        return mapped

    @classmethod
    def codex_native_tool_input_json_from_responses(
        cls,
        tool_name: Any,
        arguments_json: str,
        *,
        use_codex_native_tools: bool,
    ) -> str:
        if not use_codex_native_tools or tool_name != "exec_command":
            return arguments_json
        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError:
            return arguments_json
        mapped = cls.codex_native_tool_input_from_responses(
            tool_name,
            parsed,
            use_codex_native_tools=use_codex_native_tools,
        )
        if mapped == parsed:
            return arguments_json
        return json.dumps(mapped, separators=(",", ":"))

    def _responses_function_call_to_anthropic_tool_use(
        self,
        *,
        upstream_tool_name: Any,
        arguments: Any,
        call_id: Any,
        fallback_id: Any,
        use_codex_native_tools: bool,
    ) -> Dict[str, Any]:
        anthropic_tool_name = self.codex_native_tool_name_from_responses(
            upstream_tool_name,
            use_codex_native_tools=use_codex_native_tools,
        )
        tool_input = self.codex_native_tool_input_from_responses(
            upstream_tool_name,
            self._deserialize_tool_input(arguments),
            use_codex_native_tools=use_codex_native_tools,
        )
        input_data = sanitize_anthropic_tool_use_input(
            tool_name=anthropic_tool_name,
            tool_input=tool_input,
        )
        return AnthropicResponseContentBlockToolUse(
            type="tool_use",
            id=sanitize_anthropic_tool_use_id(call_id or fallback_id or ""),
            name=anthropic_tool_name,
            input=input_data,
        ).model_dump()

    @classmethod
    def _codex_native_tool_schema_to_responses(
        cls,
        tool_name: Any,
        schema: Any,
        *,
        use_codex_native_tools: bool,
    ) -> Any:
        if not use_codex_native_tools or tool_name != "Bash":
            return schema
        return cls._codex_native_exec_command_parameters()

    @staticmethod
    def _extract_allowed_tools_from_mcp_toolset(
        toolset: AnthropicMcpToolset,
    ) -> Optional[List[str]]:
        default_config = toolset.get("default_config")
        configs = toolset.get("configs")
        if not isinstance(default_config, dict):
            default_config = {}
        if not isinstance(configs, dict):
            configs = {}

        default_enabled = default_config.get("enabled", True)
        if default_enabled is False:
            enabled_tools = []
            for tool_name, config in configs.items():
                if not isinstance(config, dict) or config.get("enabled", True) is True:
                    enabled_tools.append(tool_name)
            return enabled_tools

        # OpenAI MCP tools only expose an allowlist. Anthropic's denylist and
        # per-tool defer_loading semantics have no direct equivalent here.
        return None

    def translate_mcp_servers_to_responses_tools(
        self,
        mcp_servers: List[AnthropicMcpServerTool],
        tools: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        toolsets_by_server: Dict[str, AnthropicMcpToolset] = {}

        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "mcp_toolset":
                    server_name = tool.get("mcp_server_name")
                    if isinstance(server_name, str) and server_name:
                        toolsets_by_server[server_name] = cast(
                            AnthropicMcpToolset, tool
                        )

        for server in mcp_servers:
            server_name = server.get("name")
            server_url = server.get("url")
            if not isinstance(server_name, str) or not isinstance(server_url, str):
                continue

            responses_tool: Dict[str, Any] = {
                "type": "mcp",
                "server_label": server_name,
                "server_url": server_url,
                "require_approval": "never",
            }

            authorization_token = server.get("authorization_token")
            if isinstance(authorization_token, str) and authorization_token:
                responses_tool["authorization"] = authorization_token

            allowed_tools: Optional[List[str]] = None
            toolset = toolsets_by_server.get(server_name)
            if toolset is not None:
                allowed_tools = self._extract_allowed_tools_from_mcp_toolset(toolset)
            else:
                tool_configuration = server.get("tool_configuration")
                if isinstance(tool_configuration, dict):
                    configured_allowed = tool_configuration.get("allowed_tools")
                    if isinstance(configured_allowed, list):
                        allowed_tools = [
                            tool_name
                            for tool_name in configured_allowed
                            if isinstance(tool_name, str)
                        ]

            if allowed_tools:
                responses_tool["allowed_tools"] = allowed_tools

            result.append(responses_tool)

        return result

    def translate_messages_to_responses_input(  # noqa: PLR0915
        self,
        messages: List[
            Union[
                AnthropicMessagesUserMessageParam,
                AnthopicMessagesAssistantMessageParam,
            ]
        ],
        *,
        use_codex_native_tools: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convert Anthropic messages list to Responses API `input` items.

        Mapping:
          user text          -> message(role=user, input_text)
          user image         -> message(role=user, input_image)
          user tool_result   -> function_call_output
          assistant text     -> message(role=assistant, output_text)
          assistant tool_use -> function_call
        """
        input_items: List[Dict[str, Any]] = []
        reasoning_item_index = 0

        for m in messages:
            role = m["role"]
            content = m.get("content")

            if role == "user":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    user_parts: List[Dict[str, Any]] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            user_parts.append(
                                {"type": "input_text", "text": block.get("text", "")}
                            )
                        elif btype == "image":
                            url = self._translate_anthropic_image_source_to_url(
                                block.get("source", {})
                            )
                            if url:
                                user_parts.append(
                                    {"type": "input_image", "image_url": url}
                                )
                        elif btype == "tool_result":
                            tool_use_id = block.get("tool_use_id", "")
                            output_text = self._anthropic_content_to_text(
                                block.get("content")
                            )
                            # tool_result is a top-level item, not inside the message
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_use_id,
                                    "output": output_text,
                                }
                            )
                    if user_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": user_parts,
                            }
                        )

            elif role == "assistant":
                if isinstance(content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                elif isinstance(content, list):
                    asst_parts: List[Dict[str, Any]] = []

                    def flush_assistant_parts() -> None:
                        if not asst_parts:
                            return
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": list(asst_parts),
                            }
                        )
                        asst_parts.clear()

                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            asst_parts.append(
                                {"type": "output_text", "text": block.get("text", "")}
                            )
                        elif btype == "tool_use":
                            flush_assistant_parts()
                            tool_name = block.get("name", "")
                            responses_tool_name = (
                                self._codex_native_tool_name_to_responses(
                                    tool_name,
                                    use_codex_native_tools=use_codex_native_tools,
                                )
                            )
                            tool_input = self._codex_native_tool_input_to_responses(
                                tool_name,
                                block.get("input", {}),
                                use_codex_native_tools=use_codex_native_tools,
                            )
                            # tool_use becomes a top-level function_call item
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": block.get("id", ""),
                                    "name": responses_tool_name,
                                    "arguments": json.dumps(tool_input),
                                }
                            )
                        elif btype == "mcp_tool_result":
                            output_text = self._anthropic_content_to_text(
                                block.get("content")
                            )
                            if output_text:
                                asst_parts.append(
                                    {"type": "output_text", "text": output_text}
                                )
                        elif btype in ("thinking", "redacted_thinking"):
                            reasoning_id = (
                                self._responses_reasoning_id_from_anthropic_block(block)
                            )
                            if reasoning_id is None:
                                continue
                            reasoning_item = (
                                self._anthropic_thinking_block_to_reasoning_item(
                                    block,
                                    reasoning_id=reasoning_id,
                                )
                            )
                            if reasoning_item is not None:
                                flush_assistant_parts()
                                reasoning_item_index += 1
                                input_items.append(reasoning_item)
                    flush_assistant_parts()

        return input_items

    def translate_tools_to_responses_api(
        self,
        tools: List[AllAnthropicToolsValues],
        *,
        use_codex_native_tools: bool = False,
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic tool definitions to Responses API function tools."""
        result: List[Dict[str, Any]] = []
        for tool in tools:
            tool_dict = cast(Dict[str, Any], tool)
            tool_type = tool_dict.get("type", "")
            tool_name = tool_dict.get("name", "")
            # web_search tool
            if (
                isinstance(tool_type, str) and tool_type.startswith("web_search")
            ) or tool_name == "web_search":
                result.append({"type": "web_search_preview"})
                continue
            responses_tool_name = self._codex_native_tool_name_to_responses(
                tool_name,
                use_codex_native_tools=use_codex_native_tools,
            )
            func_tool: Dict[str, Any] = {
                "type": "function",
                "name": responses_tool_name,
            }
            if "description" in tool_dict:
                func_tool["description"] = tool_dict["description"]
            if "input_schema" in tool_dict:
                func_tool["parameters"] = self._codex_native_tool_schema_to_responses(
                    tool_name,
                    tool_dict["input_schema"],
                    use_codex_native_tools=use_codex_native_tools,
                )
            result.append(func_tool)
        return result

    @classmethod
    def translate_tool_choice_to_responses_api(
        cls,
        tool_choice: AnthropicMessagesToolChoice,
        *,
        use_codex_native_tools: bool = False,
    ) -> Any:
        """Convert Anthropic tool_choice to Responses API tool_choice."""
        tc_type = tool_choice.get("type")
        if tc_type == "any":
            return "required"
        elif tc_type == "none":
            return "none"
        elif tc_type == "auto":
            return "auto"
        elif tc_type == "tool":
            return {
                "type": "function",
                "name": cls._codex_native_tool_name_to_responses(
                    tool_choice.get("name", ""),
                    use_codex_native_tools=use_codex_native_tools,
                ),
            }
        return "auto"

    @staticmethod
    def translate_parallel_tool_calls_to_responses_api(
        tool_choice: Optional[AnthropicMessagesToolChoice],
        translated_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        """
        Convert Anthropic parallel tool-use semantics to Responses.

        Only emit the parameter when there are translated function tools on the
        request. Built-in Responses tools do not support parallel function-calling
        semantics in the same way, so leaving the parameter unset is the safer
        best-effort translation for mixed/non-function tool sets.
        """
        if not translated_tools:
            return None

        if any(tool.get("type") != "function" for tool in translated_tools):
            return None

        if tool_choice and tool_choice.get("type") == "none":
            return None

        if tool_choice and tool_choice.get("disable_parallel_tool_use"):
            return False

        return True

    @staticmethod
    def translate_context_management_to_responses_api(
        context_management: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert Anthropic context_management dict to OpenAI Responses API array format.

        Anthropic format: {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}
        OpenAI format:    [{"type": "compaction", "compact_threshold": 150000}]
        """
        if not isinstance(context_management, dict):
            return None

        edits = context_management.get("edits", [])
        if not isinstance(edits, list):
            return None

        result: List[Dict[str, Any]] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112":
                entry: Dict[str, Any] = {"type": "compaction"}
                trigger = edit.get("trigger")
                if isinstance(trigger, dict) and trigger.get("value") is not None:
                    entry["compact_threshold"] = int(trigger["value"])
                result.append(entry)

        return result if result else None

    @staticmethod
    def translate_thinking_to_reasoning(
        thinking: Any,
        output_config: Optional[Dict[str, Any]] = None,
        *,
        model: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert Anthropic thinking/output_config params to Responses API reasoning.

        thinking.budget_tokens maps to reasoning effort:
          >= 10000 -> high, >= 5000 -> medium, >= 2000 -> low, < 2000 -> minimal

        thinking.type='adaptive' does not expose a fixed budget. We approximate it
        as medium effort unless output_config.effort supplies a more precise value.

        output_config.effort is Anthropic's newer control surface. We map it to the
        closest supported Responses effort. Anthropic's 'max' / 'xhigh' maps to
        OpenAI 'xhigh' only when model metadata says the target supports it.
        """
        normalized_effort = normalize_reasoning_effort_for_provider(
            thinking=thinking,
            output_config=output_config,
            model=model,
            custom_llm_provider=custom_llm_provider,
            native_provider="openai",
            native_field="reasoning.effort",
        )
        if normalized_effort is None or normalized_effort.native_value is None:
            return None
        reasoning: Dict[str, Any] = {"effort": normalized_effort.native_value}
        if (
            isinstance(thinking, dict)
            and thinking.get("type") in ("enabled", "adaptive")
            and isinstance(model, str)
            and model
            and supports_reasoning_summary(
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        ):
            reasoning["summary"] = "detailed"
        return reasoning

    def translate_request(  # noqa: PLR0915
        self,
        anthropic_request: AnthropicMessagesRequest,
        *,
        custom_llm_provider: Optional[str] = None,
        use_codex_native_tools: bool = False,
    ) -> Dict[str, Any]:
        """
        Translate a full Anthropic /v1/messages request dict to
        litellm.responses() / litellm.aresponses() kwargs.
        """
        model: str = anthropic_request["model"]
        messages_list = cast(
            List[
                Union[
                    AnthropicMessagesUserMessageParam,
                    AnthopicMessagesAssistantMessageParam,
                ]
            ],
            anthropic_request["messages"],
        )

        responses_kwargs: Dict[str, Any] = {
            "model": model,
            "input": self.translate_messages_to_responses_input(
                messages_list,
                use_codex_native_tools=use_codex_native_tools,
            ),
        }

        # system -> instructions
        system = anthropic_request.get("system")
        if system:
            if isinstance(system, str):
                responses_kwargs["instructions"] = system
            elif isinstance(system, list):
                text_parts = [
                    b.get("text", "")
                    for b in system
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                responses_kwargs["instructions"] = "\n".join(filter(None, text_parts))

        # max_tokens -> max_output_tokens
        max_tokens = anthropic_request.get("max_tokens")
        if max_tokens:
            responses_kwargs["max_output_tokens"] = max_tokens

        # temperature / top_p passed through
        if "temperature" in anthropic_request:
            responses_kwargs["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            responses_kwargs["top_p"] = anthropic_request["top_p"]

        output_config = anthropic_request.get("output_config")

        # tools
        translated_tools: Optional[List[Dict[str, Any]]] = None
        tools = anthropic_request.get("tools")
        mcp_servers = anthropic_request.get("mcp_servers")
        translated_tools_list: List[Dict[str, Any]] = []
        if tools:
            standard_tools = [
                tool
                for tool in tools
                if not (isinstance(tool, dict) and tool.get("type") == "mcp_toolset")
            ]
            translated_tools_list.extend(
                self.translate_tools_to_responses_api(
                    cast(List[AllAnthropicToolsValues], standard_tools),
                    use_codex_native_tools=use_codex_native_tools,
                )
            )
        if mcp_servers:
            translated_tools_list.extend(
                self.translate_mcp_servers_to_responses_tools(
                    cast(List[AnthropicMcpServerTool], mcp_servers),
                    tools=cast(Optional[List[Any]], tools),
                )
            )
        if translated_tools_list:
            translated_tools = translated_tools_list
            responses_kwargs["tools"] = translated_tools

        # tool_choice
        tool_choice = anthropic_request.get("tool_choice")
        if tool_choice:
            translated_tool_choice = self.translate_tool_choice_to_responses_api(
                cast(AnthropicMessagesToolChoice, tool_choice),
                use_codex_native_tools=use_codex_native_tools,
            )
            tool_choice_type = cast(AnthropicMessagesToolChoice, tool_choice).get(
                "type"
            )
            if not (tool_choice_type == "none" and not translated_tools):
                responses_kwargs["tool_choice"] = translated_tool_choice

        parallel_tool_calls = self.translate_parallel_tool_calls_to_responses_api(
            cast(Optional[AnthropicMessagesToolChoice], tool_choice),
            translated_tools=translated_tools,
        )
        if parallel_tool_calls is not None:
            responses_kwargs["parallel_tool_calls"] = parallel_tool_calls

        # thinking -> reasoning
        thinking = anthropic_request.get("thinking")
        reasoning = self.translate_thinking_to_reasoning(
            thinking,
            output_config if isinstance(output_config, dict) else None,
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
        if reasoning:
            responses_kwargs["reasoning"] = reasoning

        # output_format / output_config.format -> text format
        # output_format: {"type": "json_schema", "schema": {...}}
        # output_config: {"format": {"type": "json_schema", "schema": {...}}}
        output_format: Any = anthropic_request.get("output_format")
        if not isinstance(output_format, dict) and isinstance(output_config, dict):
            output_format = output_config.get("format")  # type: ignore[assignment]
        if (
            isinstance(output_format, dict)
            and output_format.get("type") == "json_schema"
        ):
            schema = output_format.get("schema")
            if schema:
                responses_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_output",
                        "schema": schema,
                        "strict": True,
                    }
                }

        # context_management: Anthropic dict -> OpenAI array
        context_management = anthropic_request.get("context_management")
        if isinstance(context_management, dict):
            openai_cm = self.translate_context_management_to_responses_api(
                context_management
            )
            if openai_cm is not None:
                responses_kwargs["context_management"] = openai_cm

        # Anthropic cache_control -> OpenAI prompt_cache_key + cache intent.
        # Key derivation is shared (system/tools only) so moving per-turn
        # breakpoints do not invalidate the key (RR-032 #1).
        cache_requested = request_contains_cache_control(anthropic_request)
        prompt_cache_key = _resolve_responses_prompt_cache_key(anthropic_request)
        if cache_requested or prompt_cache_key is not None:
            litellm_metadata = dict(responses_kwargs.get("litellm_metadata") or {})
            litellm_metadata.update(
                provider_cache_intent_metadata(
                    provider="openai",
                    attempted=True,
                    native_supported=True,
                )
            )
            litellm_metadata[
                "anthropic_adapter_cache_control_present"
            ] = cache_requested
            if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
                responses_kwargs["prompt_cache_key"] = prompt_cache_key
                litellm_metadata["openai_prompt_cache_key_present"] = True
                litellm_metadata["openai_prompt_cache_key_omitted_reason"] = None
            else:
                # Volatile-only cache_control (messages) — omit key rather than
                # emit a per-turn value that defeats cache affinity.
                litellm_metadata["openai_prompt_cache_key_present"] = False
                litellm_metadata[
                    "openai_prompt_cache_key_omitted_reason"
                ] = "no_stable_system_or_tools_cache_surface"
            responses_kwargs["litellm_metadata"] = litellm_metadata

        # metadata user_id -> user
        metadata = anthropic_request.get("metadata")
        if isinstance(metadata, dict) and "user_id" in metadata:
            responses_kwargs["user"] = str(metadata["user_id"])[:64]

        return responses_kwargs

    # ------------------------------------------------------------------ #
    # Response translation: Responses API -> Anthropic                    #
    # ------------------------------------------------------------------ #

    def translate_response(
        self,
        response: ResponsesAPIResponse,
        *,
        use_codex_native_tools: bool = False,
    ) -> AnthropicMessagesResponse:
        """
        Translate an OpenAI ResponsesAPIResponse to AnthropicMessagesResponse.
        """
        raise_for_failed_responses_response(
            response, context="Anthropic Responses adapter"
        )

        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseReasoningItem,
        )

        from litellm.types.llms.openai import ResponseAPIUsage

        content: List[Dict[str, Any]] = []
        stop_reason: AnthropicFinishReason = "end_turn"

        for item in response.output:
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )

            if item_type == "mcp_call":
                item_id = getattr(item, "id", None) or (
                    item.get("id") if isinstance(item, dict) else None
                )
                name = getattr(item, "name", None) or (
                    item.get("name") if isinstance(item, dict) else None
                )
                server_label = getattr(item, "server_label", None) or (
                    item.get("server_label") if isinstance(item, dict) else None
                )
                arguments = getattr(item, "arguments", None) or (
                    item.get("arguments") if isinstance(item, dict) else None
                )
                input_data = self._deserialize_tool_input(arguments)
                call_id = str(item_id or "")
                content.append(
                    AnthropicResponseContentBlockMcpToolUse(
                        type="mcp_tool_use",
                        id=call_id,
                        name=str(name or ""),
                        server_name=str(server_label or ""),
                        input=input_data,
                    ).model_dump()
                )
                output = getattr(item, "output", None) or (
                    item.get("output") if isinstance(item, dict) else None
                )
                error = getattr(item, "error", None) or (
                    item.get("error") if isinstance(item, dict) else None
                )
                result_text = self._responses_content_to_text(error or output)
                content.append(
                    AnthropicResponseContentBlockMcpToolResult(
                        type="mcp_tool_result",
                        tool_use_id=call_id,
                        is_error=error is not None,
                        content=[{"type": "text", "text": result_text}],
                    ).model_dump()
                )
                continue

            if item_type == "mcp_list_tools":
                continue

            if isinstance(item, ResponseReasoningItem):
                for summary in item.summary:
                    text = getattr(summary, "text", "")
                    if text:
                        content.append(
                            AnthropicResponseContentBlockThinking(
                                type="thinking",
                                thinking=text,
                                signature=None,
                            ).model_dump()
                        )

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        content.append(
                            AnthropicResponseContentBlockText(
                                type="text", text=getattr(part, "text", "")
                            ).model_dump()
                        )

            elif isinstance(item, ResponseFunctionToolCall):
                content.append(
                    self._responses_function_call_to_anthropic_tool_use(
                        upstream_tool_name=item.name,
                        arguments=item.arguments,
                        call_id=item.call_id,
                        fallback_id=item.id,
                        use_codex_native_tools=use_codex_native_tools,
                    )
                )
                stop_reason = "tool_use"
            elif isinstance(item, dict):
                if item_type == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            content.append(
                                AnthropicResponseContentBlockText(
                                    type="text", text=part.get("text", "")
                                ).model_dump()
                            )
                elif item_type == "function_call":
                    content.append(
                        self._responses_function_call_to_anthropic_tool_use(
                            upstream_tool_name=item.get("name", ""),
                            arguments=item.get("arguments"),
                            call_id=item.get("call_id"),
                            fallback_id=item.get("id"),
                            use_codex_native_tools=use_codex_native_tools,
                        )
                    )
                    stop_reason = "tool_use"

        # status -> stop_reason override
        if response.status == "incomplete":
            stop_reason = "max_tokens"

        # usage
        raw_usage: Optional[ResponseAPIUsage] = response.usage
        input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)

        anthropic_usage = AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return AnthropicMessagesResponse(
            id=response.id,
            type="message",
            role="assistant",
            model=response.model or "unknown-model",
            stop_sequence=None,
            usage=anthropic_usage,  # type: ignore
            content=content,  # type: ignore
            stop_reason=stop_reason,
        )
