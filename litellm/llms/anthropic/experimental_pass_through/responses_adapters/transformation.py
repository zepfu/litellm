"""
Transformation layer: Anthropic /v1/messages <-> OpenAI Responses API.

This module owns all format conversions for the direct v1/messages -> Responses API
path used for OpenAI and Azure models.
"""

import json
from typing import Any, Dict, List, Optional, Union, cast

from litellm.utils import supports_reasoning_summary
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
                        toolsets_by_server[server_name] = cast(AnthropicMcpToolset, tool)

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
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            asst_parts.append(
                                {"type": "output_text", "text": block.get("text", "")}
                            )
                        elif btype == "tool_use":
                            # tool_use becomes a top-level function_call item
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "call_id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
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
                        elif btype == "thinking":
                            thinking_text = block.get("thinking", "")
                            if thinking_text:
                                asst_parts.append(
                                    {"type": "output_text", "text": thinking_text}
                                )
                    if asst_parts:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": asst_parts,
                            }
                        )

        return input_items

    def translate_tools_to_responses_api(
        self,
        tools: List[AllAnthropicToolsValues],
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
            func_tool: Dict[str, Any] = {"type": "function", "name": tool_name}
            if "description" in tool_dict:
                func_tool["description"] = tool_dict["description"]
            if "input_schema" in tool_dict:
                func_tool["parameters"] = tool_dict["input_schema"]
            result.append(func_tool)
        return result

    @staticmethod
    def translate_tool_choice_to_responses_api(
        tool_choice: AnthropicMessagesToolChoice,
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
            return {"type": "function", "name": tool_choice.get("name", "")}
        return "auto"

    @staticmethod
    def translate_parallel_tool_calls_to_responses_api(
        tool_choice: AnthropicMessagesToolChoice,
        translated_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        """
        Convert Anthropic disable_parallel_tool_use to Responses parallel_tool_calls.

        Only emit the parameter when there are translated function tools on the
        request. Built-in Responses tools do not support parallel function-calling
        semantics in the same way, so leaving the parameter unset is the safer
        best-effort translation for mixed/non-function tool sets.
        """
        if not tool_choice.get("disable_parallel_tool_use"):
            return None

        if not translated_tools:
            return None

        if any(tool.get("type") != "function" for tool in translated_tools):
            return None

        return False

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
        closest supported Responses effort. Anthropic's 'max' does not have a stable
        cross-model equivalent on the OpenAI side, so we clamp it to 'high' rather
        than assume all adapted targets support 'xhigh'.
        """
        output_effort: Optional[str] = None
        if isinstance(output_config, dict):
            raw_output_effort = output_config.get("effort")
            if isinstance(raw_output_effort, str):
                output_effort = {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "max": "high",
                }.get(raw_output_effort.lower())

        if not isinstance(thinking, dict):
            if output_effort:
                return {"effort": output_effort}
            return None

        thinking_type = thinking.get("type")
        if thinking_type == "disabled":
            return None

        effort = output_effort
        if effort is None and thinking_type == "enabled":
            budget = thinking.get("budget_tokens", 0)
            if budget >= 10000:
                effort = "high"
            elif budget >= 5000:
                effort = "medium"
            elif budget >= 2000:
                effort = "low"
            else:
                effort = "minimal"
        elif effort is None and thinking_type == "adaptive":
            effort = "medium"

        if effort is None:
            return None

        reasoning: Dict[str, Any] = {"effort": effort}
        if (
            thinking_type in ("enabled", "adaptive")
            and isinstance(model, str)
            and model
            and supports_reasoning_summary(
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        ):
            reasoning["summary"] = "detailed"
        return reasoning

    def translate_request(
        self,
        anthropic_request: AnthropicMessagesRequest,
        *,
        custom_llm_provider: Optional[str] = None,
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
            "input": self.translate_messages_to_responses_input(messages_list),
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
                    cast(List[AllAnthropicToolsValues], standard_tools)
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
                cast(AnthropicMessagesToolChoice, tool_choice)
            )
            tool_choice_type = cast(AnthropicMessagesToolChoice, tool_choice).get("type")
            if not (tool_choice_type == "none" and not translated_tools):
                responses_kwargs["tool_choice"] = translated_tool_choice
            parallel_tool_calls = (
                self.translate_parallel_tool_calls_to_responses_api(
                    cast(AnthropicMessagesToolChoice, tool_choice),
                    translated_tools=translated_tools,
                )
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
    ) -> AnthropicMessagesResponse:
        """
        Translate an OpenAI ResponsesAPIResponse to AnthropicMessagesResponse.
        """
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
                input_data = self._deserialize_tool_input(item.arguments)
                content.append(
                    AnthropicResponseContentBlockToolUse(
                        type="tool_use",
                        id=item.call_id or item.id or "",
                        name=item.name,
                        input=input_data,
                    ).model_dump()
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
                    input_data = self._deserialize_tool_input(item.get("arguments"))
                    content.append(
                        AnthropicResponseContentBlockToolUse(
                            type="tool_use",
                            id=item.get("call_id") or item.get("id", ""),
                            name=item.get("name", ""),
                            input=input_data,
                        ).model_dump()
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
