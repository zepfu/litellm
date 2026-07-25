"""Tool activity detection, classification, extraction, summarization, and sensitive-config handling.

Behavior-preserving Wave A4B extraction from the identity package
``__init__``. Function bodies resolve free names through the identity
host namespace after :func:`install` rebinds ``__globals__`` (record.py
contract), so module-level imports of identity helpers are intentionally
absent here."""

import json
import re
import shlex
from functools import lru_cache
from types import FunctionType as _FunctionType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:

    def _extract_provider_specific_fields(message: Any) -> Dict[str, Any]: ...

    def _extract_responses_completed_payload_from_passthrough_fallback_text(
        response_text: Any,
    ) -> Optional[Dict[str, Any]]: ...

    def _maybe_get(obj: Any, key: str, default: Any = None) -> Any: ...

    def _maybe_get_path(obj: Any, *keys: str, default: Any = None) -> Any: ...

    def _safe_int(value: Any) -> Optional[int]: ...

_TOOL_ACTIVITY_READ_NAMES = {
    "read",
    "view",
    "cat",
    "grep",
    "glob",
    "ls",
    "listdir",
    "list_files",
    "search",
    "fetch",
    "webfetch",
    "web_fetch",
    "notebookread",
}

_TOOL_ACTIVITY_MODIFY_NAMES = {
    "write",
    "edit",
    "replace",
    "replacement",
    "multiedit",
    "apply_patch",
    "applypatch",
    "notebookedit",
    "notebookwrite",
}

_TOOL_ACTIVITY_COMMAND_NAMES = {
    "bash",
    "shell",
    "terminal",
    "run",
    "exec",
    "exec_command",
    "browser_run_code",
}

_TOOL_ACTIVITY_SKIP_PATH_KEYS = {
    "content",
    "old_str",
    "new_str",
    "replacement",
    "patch",
    "command",
    "cmd",
    "description",
    "thinking",
    "reason",
}

_APPLY_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)

_APPLY_PATCH_MOVE_TO_RE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)

_GIT_COMMAND_RE = re.compile(r"(?<!\S)git\b(?P<args>[^;&|]*)")

_GIT_GLOBAL_OPTIONS_WITH_VALUES = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--config-env",
}

_TOOL_ACTIVITY_COMMAND_TEXT_KEYS = (
    "command",
    "cmd",
    "raw_text",
    "input",
    "script",
    "shell",
    "bash",
    "code",
    "text",
)

_TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS = {
    "description",
    "reason",
    "thinking",
    "title",
    "summary",
}

_SENSITIVE_CONFIG_CHANGE_FIELDS = (
    "changed_pre_commit_config",
    "changed_env_file",
    "changed_pyproject_toml",
    "changed_gitignore",
)

_SENSITIVE_CONFIG_ENV_REDACTION = "[redacted_sensitive_config_file_content]"

_SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS = {
    "bash",
    "cmd",
    "code",
    "command",
    "content",
    "input",
    "new_str",
    "old_str",
    "patch",
    "raw_text",
    "replacement",
    "script",
    "shell",
    "text",
    "value",
}

_SENSITIVE_CONFIG_ENV_COMMAND_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])\.env[A-Za-z0-9._-]*(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)

_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES: Dict[str, str] = {
    "apply_patch_call": "apply_patch",
    "custom_tool_call": "custom_tool_call",
    "computer_call": "computer_call",
    "local_shell_call": "local_shell_call",
    "mcp_call": "mcp_call",
    "web_search_call": "web_search_call",
    "file_search_call": "file_search_call",
    "image_generation_call": "image_generation_call",
}

_RESPONSE_OUTPUT_TOOL_ITEM_TYPES = set(_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES) | {"function_call"}


def _dedupe_strings(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        stripped = str(value).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result


def _normalize_changed_file_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().strip("'\"").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        return None
    return normalized


def _changed_file_basename(value: Any) -> Optional[str]:
    normalized = _normalize_changed_file_path(value)
    if normalized is None:
        return None
    return normalized.rstrip("/").rsplit("/", 1)[-1]


def _sensitive_config_change_flags_from_paths(paths: List[str]) -> Dict[str, bool]:
    flags = {field: False for field in _SENSITIVE_CONFIG_CHANGE_FIELDS}
    for path in _dedupe_strings(paths):
        basename = _changed_file_basename(path)
        if not basename:
            continue
        basename_lower = basename.lower()
        if basename_lower in {".pre-commit-config.yaml", ".pre-commit-config.yml"}:
            flags["changed_pre_commit_config"] = True
        if basename_lower.startswith(".env"):
            flags["changed_env_file"] = True
        if basename_lower == "pyproject.toml":
            flags["changed_pyproject_toml"] = True
        if basename_lower == ".gitignore":
            flags["changed_gitignore"] = True
    return flags


def _text_mentions_env_file(value: Any) -> bool:
    return isinstance(value, str) and bool(_SENSITIVE_CONFIG_ENV_COMMAND_RE.search(value))


def _redact_sensitive_config_argument_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, nested_value in value.items():
            key_lower = str(key).lower()
            if key_lower in _SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS:
                redacted[key] = _SENSITIVE_CONFIG_ENV_REDACTION
            else:
                redacted[key] = _redact_sensitive_config_argument_value(nested_value)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_config_argument_value(item) for item in value]
    return value


def _sanitize_tool_activity_arguments_for_sensitive_config(
    arguments: Any,
    *,
    file_paths_modified: List[str],
    command_text: Optional[str] = None,
) -> Any:
    flags = _sensitive_config_change_flags_from_paths(file_paths_modified)
    if not flags["changed_env_file"] and not _text_mentions_env_file(command_text):
        return arguments
    if isinstance(arguments, str):
        return _SENSITIVE_CONFIG_ENV_REDACTION
    return _redact_sensitive_config_argument_value(arguments)


def _normalize_sensitive_config_change_state_on_record(record: Dict[str, Any]) -> None:
    modified_paths: List[str] = []
    tool_activity = record.get("tool_activity")
    if not isinstance(tool_activity, list):
        return
    if isinstance(tool_activity, list):
        for item in tool_activity:
            if not isinstance(item, dict):
                continue
            modified_paths.extend(value for value in (item.get("file_paths_modified") or []) if isinstance(value, str))

    flags = _sensitive_config_change_flags_from_paths(modified_paths)
    for field, derived_value in flags.items():
        record[field] = bool(record.get(field)) or derived_value


def _parse_tool_arguments(arguments: Any) -> Any:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, (dict, list)):
        return arguments
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw_text": stripped}
    return {"value": arguments}


def _is_empty_claude_read_pages_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _sanitize_tool_activity_arguments(tool_name: str, arguments: Any) -> Any:
    if tool_name != "Read" or not isinstance(arguments, dict):
        return arguments
    if "pages" not in arguments:
        return arguments
    if not _is_empty_claude_read_pages_value(arguments.get("pages")):
        return arguments

    sanitized_arguments = dict(arguments)
    sanitized_arguments.pop("pages", None)
    return sanitized_arguments


def _extract_paths_from_patch_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    paths = _APPLY_PATCH_FILE_RE.findall(text) + _APPLY_PATCH_MOVE_TO_RE.findall(text)
    return _dedupe_strings(paths)


def _extract_file_paths_from_tool_arguments(arguments: Any) -> List[str]:
    parsed_arguments = _parse_tool_arguments(arguments)
    if isinstance(parsed_arguments, str):
        return []
    return _dedupe_strings(_collect_file_paths_from_value(parsed_arguments))


def _extract_command_text_from_tool_arguments(arguments: Any) -> Optional[str]:
    parsed_arguments = _parse_tool_arguments(arguments)
    command_text = _find_command_text_in_value(parsed_arguments)
    if command_text is not None:
        return command_text
    if isinstance(parsed_arguments, str) and parsed_arguments.strip():
        return parsed_arguments.strip()
    return None


def _count_git_subcommand(command_text: str, subcommand: str) -> int:
    count = 0
    for match in _GIT_COMMAND_RE.finditer(command_text):
        command = f"git{match.group('args') or ''}"
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in _GIT_GLOBAL_OPTIONS_WITH_VALUES:
                index += 2
                continue
            if any(token.startswith(f"{option}=") for option in _GIT_GLOBAL_OPTIONS_WITH_VALUES):
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                continue
            if token == subcommand:
                count += 1
            break
    return count


def _collect_file_paths_from_value(value: Any) -> List[str]:
    collected: List[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            collected.append(stripped)
    elif isinstance(value, list):
        for item in value:
            collected.extend(_collect_file_paths_from_value(item))
    elif isinstance(value, dict):
        for nested_key, nested_value in list(value.items()):
            nested_key_lower = str(nested_key).lower()
            if nested_key_lower in _TOOL_ACTIVITY_SKIP_PATH_KEYS:
                continue
            if any(token in nested_key_lower for token in ("path", "file")):
                collected.extend(_collect_file_paths_from_value(nested_value))
    return collected


def _find_command_text_in_value(value: Any, *, depth: int = 0) -> Optional[str]:
    if depth > 4:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            command_text = _find_command_text_in_value(item, depth=depth + 1)
            if command_text is not None:
                return command_text
        return None
    if not isinstance(value, dict):
        return None

    for key in _TOOL_ACTIVITY_COMMAND_TEXT_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for key, nested_value in list(value.items()):
        if str(key).lower() in _TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS:
            continue
        command_text = _find_command_text_in_value(nested_value, depth=depth + 1)
        if command_text is not None:
            return command_text
    return None


def _classify_tool_kind(tool_name: str) -> str:
    normalized_name = (tool_name or "").strip().lower()
    if normalized_name.startswith("mcp__"):
        return "mcp"
    if normalized_name in _TOOL_ACTIVITY_COMMAND_NAMES or any(
        token in normalized_name for token in ("bash", "shell", "terminal")
    ):
        return "command"
    if normalized_name in _TOOL_ACTIVITY_MODIFY_NAMES or any(
        token in normalized_name for token in ("write", "edit", "patch")
    ):
        return "modify"
    if normalized_name in _TOOL_ACTIVITY_READ_NAMES or any(
        token in normalized_name for token in ("read", "view", "grep", "glob", "search", "fetch")
    ):
        return "read"
    return "other"


def _build_tool_activity_entry(
    *,
    tool_index: int,
    tool_name: str,
    arguments: Any,
    tool_call_id: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    parsed_arguments = _parse_tool_arguments(arguments)
    parsed_arguments = _sanitize_tool_activity_arguments(tool_name, parsed_arguments)
    tool_kind = _classify_tool_kind(tool_name)
    file_paths_read: List[str] = []
    file_paths_modified: List[str] = []
    command_text: Optional[str] = None

    if tool_kind == "read":
        file_paths_read = _extract_file_paths_from_tool_arguments(parsed_arguments)
    elif tool_kind == "modify":
        file_paths_modified = _extract_file_paths_from_tool_arguments(parsed_arguments)
        if tool_name.strip().lower() in {"apply_patch", "applypatch"}:
            patch_text = _extract_command_text_from_tool_arguments(parsed_arguments)
            if patch_text:
                file_paths_modified = _dedupe_strings(file_paths_modified + _extract_paths_from_patch_text(patch_text))
    elif tool_kind == "command":
        command_text = _extract_command_text_from_tool_arguments(parsed_arguments)

    if command_text is None and tool_name.strip().lower() in {"apply_patch", "applypatch"}:
        command_text = _extract_command_text_from_tool_arguments(parsed_arguments)

    git_commit_count = 0
    git_push_count = 0
    if isinstance(command_text, str) and command_text:
        git_commit_count = _count_git_subcommand(command_text, "commit")
        git_push_count = _count_git_subcommand(command_text, "push")

    sensitive_config_flags = _sensitive_config_change_flags_from_paths(file_paths_modified)
    stored_arguments = _sanitize_tool_activity_arguments_for_sensitive_config(
        parsed_arguments,
        file_paths_modified=file_paths_modified,
        command_text=command_text,
    )
    if (
        sensitive_config_flags["changed_env_file"] or _text_mentions_env_file(command_text)
    ) and command_text is not None:
        command_text = _SENSITIVE_CONFIG_ENV_REDACTION

    return {
        "tool_index": tool_index,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_kind": tool_kind,
        "file_paths_read": _dedupe_strings(file_paths_read),
        "file_paths_modified": _dedupe_strings(file_paths_modified),
        "git_commit_count": git_commit_count,
        "git_push_count": git_push_count,
        "command_text": command_text,
        "arguments": stored_arguments,
        "metadata": {"source": source} if source else {},
    }


def _extract_tool_activity_from_message(message: Any) -> List[Dict[str, Any]]:
    activity: List[Dict[str, Any]] = []
    raw_tool_calls = _maybe_get(message, "tool_calls")
    if isinstance(raw_tool_calls, list):
        for index, tool_call in enumerate(raw_tool_calls):
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(tool_call, "name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            activity.append(
                _build_tool_activity_entry(
                    tool_index=index,
                    tool_name=tool_name.strip(),
                    arguments=_maybe_get(function_obj, "arguments"),
                    tool_call_id=_maybe_get(tool_call, "id"),
                    source="message.tool_calls",
                )
            )
        return activity

    content = _maybe_get(message, "content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict):
                block_type = block.get("type")
                tool_name = block.get("name")
                arguments = block.get("input") or block.get("arguments")
                tool_call_id = block.get("id")
            else:
                block_type = getattr(block, "type", None)
                tool_name = getattr(block, "name", None)
                arguments = getattr(block, "input", None) or getattr(block, "arguments", None)
                tool_call_id = getattr(block, "id", None)
            if block_type not in {"tool_use", "function_call"}:
                continue
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            activity.append(
                _build_tool_activity_entry(
                    tool_index=index,
                    tool_name=tool_name.strip(),
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    source="message.content",
                )
            )
        if activity:
            return activity

    provider_specific_fields = _extract_provider_specific_fields(message)
    provider_tool_calls = provider_specific_fields.get("tool_calls")
    if isinstance(provider_tool_calls, list):
        for index, tool_call in enumerate(provider_tool_calls):
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(tool_call, "name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            activity.append(
                _build_tool_activity_entry(
                    tool_index=index,
                    tool_name=tool_name.strip(),
                    arguments=_maybe_get(function_obj, "arguments"),
                    tool_call_id=_maybe_get(tool_call, "id"),
                    source="provider_specific_fields.tool_calls",
                )
            )

    return activity


def _extract_response_output_items(result: Any, standard_logging_object: Optional[Dict[str, Any]] = None) -> List[Any]:
    candidate_sources: List[Any] = [result]
    if isinstance(standard_logging_object, dict):
        candidate_sources.append(standard_logging_object.get("response"))

    for source in candidate_sources:
        if isinstance(source, list):
            return source

        output_items = _maybe_get(source, "output")
        if isinstance(output_items, list):
            return output_items

        output_items = _maybe_get_path(source, "_hidden_params", "responses_output")
        if isinstance(output_items, list):
            return output_items

        completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
            _maybe_get(source, "response")
        )
        if isinstance(completed_payload, dict):
            output_items = _maybe_get(_maybe_get(completed_payload, "response"), "output")
            if isinstance(output_items, list):
                return output_items

    return []


def _resolve_response_output_tool_name(item: Any) -> Optional[str]:
    tool_name = _maybe_get(item, "name")
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()

    item_type = _maybe_get(item, "type")
    if not isinstance(item_type, str) or not item_type.strip():
        return None

    fallback_name = _RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES.get(item_type)
    if isinstance(fallback_name, str) and fallback_name.strip():
        return fallback_name.strip()

    return None


def _extract_response_output_tool_activity(
    result: Any, standard_logging_object: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    output_items = _extract_response_output_items(result, standard_logging_object)
    if not output_items:
        return []

    activity: List[Dict[str, Any]] = []
    for index, item in enumerate(output_items):
        item_type = _maybe_get(item, "type")
        if item_type not in _RESPONSE_OUTPUT_TOOL_ITEM_TYPES:
            continue
        tool_name = _resolve_response_output_tool_name(item)
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        arguments = _maybe_get(item, "arguments")
        if arguments is None and item_type in {"apply_patch_call", "custom_tool_call"}:
            arguments = _maybe_get(item, "patch") or _maybe_get(item, "input")
        activity.append(
            _build_tool_activity_entry(
                tool_index=index,
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=_maybe_get(item, "call_id") or _maybe_get(item, "id"),
                source="responses.output",
            )
        )

    return activity


def _summarize_tool_activity(tool_activity: List[Dict[str, Any]]) -> Dict[str, int]:
    read_paths: List[str] = []
    modified_paths: List[str] = []
    git_commit_count = 0
    git_push_count = 0
    for item in tool_activity:
        read_paths.extend(value for value in (item.get("file_paths_read") or []) if isinstance(value, str))
        modified_paths.extend(value for value in (item.get("file_paths_modified") or []) if isinstance(value, str))
        git_commit_count += _safe_int(item.get("git_commit_count")) or 0
        git_push_count += _safe_int(item.get("git_push_count")) or 0
    return {
        "file_read_count": len(_dedupe_strings(read_paths)),
        "file_modified_count": len(_dedupe_strings(modified_paths)),
        **_sensitive_config_change_flags_from_paths(modified_paths),
        "git_commit_count": git_commit_count,
        "git_push_count": git_push_count,
    }


def _extract_tool_call_info(message: Any) -> Tuple[int, List[str]]:
    raw_tool_calls = _maybe_get(message, "tool_calls")
    if isinstance(raw_tool_calls, list):
        tool_names: List[str] = []
        for tool_call in raw_tool_calls:
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(tool_call, "name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return len(raw_tool_calls), tool_names

    content = _maybe_get(message, "content")
    if isinstance(content, list):
        tool_names = []
        tool_call_count = 0
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
            else:
                block_type = getattr(block, "type", None)
            if block_type not in {"tool_use", "function_call"}:
                continue
            tool_call_count += 1
            tool_name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        if tool_call_count:
            return tool_call_count, tool_names

    provider_specific_fields = _extract_provider_specific_fields(message)
    provider_tool_calls = provider_specific_fields.get("tool_calls")
    if isinstance(provider_tool_calls, list):
        tool_names = []
        for tool_call in provider_tool_calls:
            tool_name = _maybe_get(_maybe_get(tool_call, "function"), "name") or _maybe_get(tool_call, "name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return len(provider_tool_calls), tool_names

    return 0, []


def _extract_response_output_tool_call_info(
    result: Any, standard_logging_object: Optional[Dict[str, Any]] = None
) -> Tuple[int, List[str]]:
    output_items = _extract_response_output_items(result, standard_logging_object)
    if not output_items:
        return 0, []

    tool_call_count = 0
    tool_names: List[str] = []
    for item in output_items:
        item_type = _maybe_get(item, "type")
        if item_type not in _RESPONSE_OUTPUT_TOOL_ITEM_TYPES:
            continue
        tool_call_count += 1
        tool_name = _resolve_response_output_tool_name(item)
        if isinstance(tool_name, str) and tool_name.strip():
            tool_names.append(tool_name)

    return tool_call_count, tool_names


_HOST_FUNCTION_NAMES = (
    "_dedupe_strings",
    "_normalize_changed_file_path",
    "_changed_file_basename",
    "_sensitive_config_change_flags_from_paths",
    "_text_mentions_env_file",
    "_redact_sensitive_config_argument_value",
    "_sanitize_tool_activity_arguments_for_sensitive_config",
    "_normalize_sensitive_config_change_state_on_record",
    "_parse_tool_arguments",
    "_is_empty_claude_read_pages_value",
    "_sanitize_tool_activity_arguments",
    "_extract_paths_from_patch_text",
    "_extract_file_paths_from_tool_arguments",
    "_extract_command_text_from_tool_arguments",
    "_count_git_subcommand",
    "_collect_file_paths_from_value",
    "_find_command_text_in_value",
    "_classify_tool_kind",
    "_build_tool_activity_entry",
    "_extract_tool_activity_from_message",
    "_extract_response_output_items",
    "_resolve_response_output_tool_name",
    "_extract_response_output_tool_activity",
    "_summarize_tool_activity",
    "_extract_tool_call_info",
    "_extract_response_output_tool_call_info",
)


def _rebind_to_host_globals(fn, host_globals):
    rebound = _FunctionType(
        fn.__code__,
        host_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = getattr(fn, "__annotations__", {})
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    rebound.__doc__ = fn.__doc__
    return rebound


def _rebind_installable_callable(value, host_globals):
    if isinstance(value, _FunctionType):
        return _rebind_to_host_globals(value, host_globals)

    wrapped = getattr(value, "__wrapped__", None)
    cache_parameters = getattr(value, "cache_parameters", None)
    if not isinstance(wrapped, _FunctionType) or not callable(cache_parameters):
        return value

    parameters = cache_parameters()
    if not isinstance(parameters, dict) or not {"maxsize", "typed"} <= parameters.keys():
        return value

    rebound_wrapped = _rebind_to_host_globals(wrapped, host_globals)
    rebound = lru_cache(
        maxsize=parameters["maxsize"],
        typed=bool(parameters["typed"]),
    )(rebound_wrapped)
    for attribute, attribute_value in getattr(value, "__dict__", {}).items():
        if attribute != "__wrapped__":
            setattr(rebound, attribute, attribute_value)
    return rebound


def install(host_globals):
    """Publish this module's helpers onto the identity host namespace.

    Plain functions are rebound so their ``__globals__`` is the identity
    package dict (record.py contract) -- free-name lookups then resolve
    through the identity namespace and monkeypatches on it stay effective.
    """
    mod = globals()
    for _name in _HOST_FUNCTION_NAMES:
        _original = mod[_name]
        _installed = _rebind_installable_callable(_original, host_globals)
        mod[_name] = _installed
        host_globals[_name] = _installed
