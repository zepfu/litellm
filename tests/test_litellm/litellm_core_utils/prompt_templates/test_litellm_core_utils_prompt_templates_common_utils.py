import json
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _attempt_json_repair,
    _quote_bare_object_keys,
    add_system_prompt_to_messages,
    get_format_from_file_id,
    handle_any_messages_to_chat_completion_str_messages_conversion,
    parse_tool_call_arguments,
    split_concatenated_json_objects,
    update_messages_with_model_file_ids,
)


def test_get_format_from_file_id():
    unified_file_id = (
        "litellm_proxy:application/pdf;unified_id,cbbe3534-8bf8-4386-af00-f5f6b7e370bf"
    )

    format = get_format_from_file_id(unified_file_id)

    assert format == "application/pdf"


def test_update_messages_with_model_file_ids():
    file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg"
    model_id = "my_model_id"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file_id,
                    },
                },
            ],
        },
    ]

    model_file_id_mapping = {file_id: {"my_model_id": "provider_file_id"}}

    updated_messages = update_messages_with_model_file_ids(
        messages, model_id, model_file_id_mapping
    )

    assert updated_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "provider_file_id",
                        "format": "application/pdf",
                    },
                },
            ],
        }
    ]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list():
    # Test with list of messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_list_infinite_loop():
    # Test that list handling doesn't cause infinite recursion
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    # This should complete without stack overflow
    result = handle_any_messages_to_chat_completion_str_messages_conversion(messages)
    assert len(result) == 2
    assert result[0] == messages[0]
    assert result[1] == messages[1]


def test_handle_any_messages_to_chat_completion_str_messages_conversion_dict():
    # Test with single dictionary message
    message = {"role": "user", "content": "Hello"}
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_handle_any_messages_to_chat_completion_str_messages_conversion_str():
    # Test with string message
    message = "Hello"
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == message


def test_handle_any_messages_to_chat_completion_str_messages_conversion_other():
    # Test with non-string/dict/list type
    message = 123
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == "123"


def test_handle_any_messages_to_chat_completion_str_messages_conversion_complex():
    # Test with complex nested structure
    message = {
        "role": "user",
        "content": {"text": "Hello", "metadata": {"timestamp": "2024-01-01"}},
    }
    result = handle_any_messages_to_chat_completion_str_messages_conversion(message)
    assert len(result) == 1
    assert result[0]["input"] == json.dumps(message)


def test_add_system_prompt_to_messages_prepend():
    """Adds system prompt at beginning when no system message exists."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = add_system_prompt_to_messages(messages, "You are a helpful assistant.")
    assert result == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_add_system_prompt_to_messages_empty_prompt_unchanged():
    """Returns messages unchanged when system_prompt is empty."""
    messages = [{"role": "user", "content": "Hello"}]
    assert add_system_prompt_to_messages(messages, "") == messages
    assert add_system_prompt_to_messages(messages, None) == messages


def test_add_system_prompt_to_messages_merge_with_first_system():
    """Merges new prompt into first system message when merge_with_first_system=True."""
    messages = [
        {"role": "system", "content": "Existing system prompt."},
        {"role": "user", "content": "Hello"},
    ]
    result = add_system_prompt_to_messages(
        messages, "You are helpful.", merge_with_first_system=True
    )
    assert result == [
        {"role": "system", "content": "You are helpful.\n\nExisting system prompt."},
        {"role": "user", "content": "Hello"},
    ]


def test_add_system_prompt_to_messages_merge_with_first_system_adds_new_when_no_system():
    """When merge_with_first_system=True but no system message, adds new one at start."""
    messages = [{"role": "user", "content": "Hello"}]
    result = add_system_prompt_to_messages(
        messages, "You are helpful.", merge_with_first_system=True
    )
    assert result == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


def test_add_system_prompt_to_messages_empty_list():
    """Adds system prompt to empty messages list."""
    result = add_system_prompt_to_messages([], "You are helpful.")
    assert result == [{"role": "system", "content": "You are helpful."}]


def test_convert_prefix_message_to_non_prefix_messages():
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        convert_prefix_message_to_non_prefix_messages,
    )

    messages = [
        {"role": "assistant", "content": "value", "prefix": True},
    ]
    result = convert_prefix_message_to_non_prefix_messages(messages)
    assert result == [
        {
            "role": "system",
            "content": "You are a helpful assistant. You are given a message and you need to respond to it. You are also given a generated content. You need to respond to the message in continuation of the generated content. Do not repeat the same content. Your response should be in continuation of this text: ",
        },
        {"role": "assistant", "content": "value"},
    ]


# ── split_concatenated_json_objects tests ──


def test_split_concatenated_json_single_object():
    """A single valid JSON object is returned as a one-element list."""
    result = split_concatenated_json_objects('{"location": "Boston"}')
    assert result == [{"location": "Boston"}]


def test_split_concatenated_json_multiple_objects():
    """
    Multiple JSON objects concatenated without separators are split correctly.
    This is the exact pattern from issue #20543 where Bedrock Claude Sonnet 4.5
    returns concatenated JSON in a single tool call arguments string.
    """
    raw = (
        '{"command": ["curl", "-i", "http://localhost:9009"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}'
        '{"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}'
    )
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"command": ["curl", "-i", "http://localhost:9009"]}
    assert result[1] == {"command": ["curl", "-i", "http://localhost:9009/robots.txt"]}
    assert result[2] == {"command": ["curl", "-i", "http://localhost:9009/sitemap.xml"]}


def test_split_concatenated_json_with_whitespace():
    """Objects separated by whitespace are handled correctly."""
    raw = '{"a": 1}  {"b": 2}\n{"c": 3}'
    result = split_concatenated_json_objects(raw)
    assert len(result) == 3
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}
    assert result[2] == {"c": 3}


def test_split_concatenated_json_empty_string():
    """Empty or whitespace-only strings return an empty list."""
    assert split_concatenated_json_objects("") == []
    assert split_concatenated_json_objects("   ") == []


def test_split_concatenated_json_non_dict_value():
    """Non-dict JSON values (e.g. arrays, strings) are replaced with {}."""
    result = split_concatenated_json_objects('[1, 2, 3]')
    assert result == [{}]


def test_split_concatenated_json_invalid_raises():
    """Completely invalid JSON raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        split_concatenated_json_objects("not json at all")



# ============ RR-020: string-literal/object-position aware key-quote repair ============


def test_rr020_missing_key_quote_repair_is_string_literal_aware():
    """Comma/word/colon sequences inside strings must not be rewritten as keys."""
    valid = '{"msg": "urgent, action: required", "note": "See config, timeout: 30"}'
    assert _quote_bare_object_keys(valid) == valid
    assert _attempt_json_repair(valid) is None
    assert parse_tool_call_arguments(valid) == {
        "msg": "urgent, action: required",
        "note": "See config, timeout: 30",
    }

    braces_in_string = '{"cmd": "echo {a:1, b:2}, done"}'
    assert _quote_bare_object_keys(braces_in_string) == braces_in_string
    assert _attempt_json_repair(braces_in_string) is None
    assert parse_tool_call_arguments(braces_in_string) == {
        "cmd": "echo {a:1, b:2}, done"
    }


def test_rr020_preserves_escaped_quotes_and_backslashes_inside_strings():
    """Escaped quotes / backslashes must not desync string tracking."""
    escaped_quote = r'{"text": "say \"hi\", action: now", "path": "C:\\tmp"}'
    assert _quote_bare_object_keys(escaped_quote) == escaped_quote
    assert _attempt_json_repair(escaped_quote) is None
    assert parse_tool_call_arguments(escaped_quote) == {
        "text": 'say "hi", action: now',
        "path": "C:\\tmp",
    }

    # Odd number of backslashes before a quote keeps the quote escaped.
    odd_escape = r'{"a": "x\\\"y, z: w"}'
    assert _quote_bare_object_keys(odd_escape) == odd_escape
    assert _attempt_json_repair(odd_escape) is None
    assert parse_tool_call_arguments(odd_escape) == {"a": 'x\\"y, z: w'}

    # Bare key after a string that contains lookalike comma/colon patterns.
    mixed = r'{"msg": "path\\, action: keep", bare: 1}'
    assert _attempt_json_repair(mixed) == {"msg": "path\\, action: keep", "bare": 1}


def test_rr020_braces_and_commas_inside_strings_do_not_change_structure():
    nested_braces = '{"x": "use {foo: bar, baz: {q:1}}"}'
    assert _quote_bare_object_keys(nested_braces) == nested_braces
    assert _attempt_json_repair(nested_braces) is None
    assert parse_tool_call_arguments(nested_braces) == {
        "x": "use {foo: bar, baz: {q:1}}"
    }

    # Unmatched { / [ only inside a string must not trigger bracket repair.
    openers_in_string = '{"x": "open { and [ here"}'
    assert _attempt_json_repair(openers_in_string) is None
    assert parse_tool_call_arguments(openers_in_string) == {
        "x": "open { and [ here"
    }


def test_rr020_repairs_bare_and_half_quoted_object_keys():
    assert _attempt_json_repair('{command": "date -u"}') == {"command": "date -u"}
    assert _attempt_json_repair('{command: "date -u", retries: 2}') == {
        "command": "date -u",
        "retries": 2,
    }
    assert _attempt_json_repair('{"ok": 1, bare: true, empty: null}') == {
        "ok": 1,
        "bare": True,
        "empty": None,
    }


def test_rr020_nested_objects_and_arrays_with_bare_keys():
    nested = '{a: {b: [{c: 1}, {d: "x, y: z"}]}}'
    assert _attempt_json_repair(nested) == {
        "a": {"b": [{"c": 1}, {"d": "x, y: z"}]}
    }
    truncated = (
        '{meta: {note: "urgent, action: required"}, '
        'command: "x", count: 1'
    )
    assert _attempt_json_repair(truncated) == {
        "meta": {"note": "urgent, action: required"},
        "command": "x",
        "count": 1,
    }
    assert _attempt_json_repair('{"items": [{id: 1}, {id: 2}') == {
        "items": [{"id": 1}, {"id": 2}]
    }


def test_rr020_array_commas_are_not_object_key_positions():
    """Comma separators inside arrays must not be treated as object-key sites."""
    false_array_keys = "[1, two: 3]"
    assert _quote_bare_object_keys(false_array_keys) == false_array_keys
    assert _attempt_json_repair(false_array_keys) is None

    # Bare keys inside objects that are array elements still repair.
    assert _attempt_json_repair("[{a: 1}, {b: 2}]") == [{"a": 1}, {"b": 2}]


def test_rr020_fail_closed_for_unrepairable_malformed_input():
    """If quoting bare keys is insufficient, return None (do not surface corruption)."""
    assert _attempt_json_repair("{a: 1, b}") is None
    assert _attempt_json_repair("{command: date}") is None
    assert _attempt_json_repair("{'command': 'x'}") is None
    assert _attempt_json_repair('{1: "x"}') is None
    assert _attempt_json_repair('{"a":1}{"b":2}') is None
    assert _attempt_json_repair('{a: [b, c]}') is None
    assert _attempt_json_repair('{"key": "incomplete value') is None


def test_rr020_already_valid_json_is_identity_no_repair():
    """Valid JSON must not be rewritten; _attempt_json_repair returns None."""
    samples = [
        "{}",
        "[]",
        '{"a": 1}',
        '{"a": [{"b": {"c": 1}}]}',
        '{"msg": "urgent, action: required"}',
        r'{"t": "say \"hi\", action: now"}',
        '{"cmd": "echo {a:1, b:2}"}',
    ]
    for raw in samples:
        assert _quote_bare_object_keys(raw) == raw
        assert _attempt_json_repair(raw) is None
        # parse_tool_call_arguments still returns the normal loads() result
        assert parse_tool_call_arguments(raw) == json.loads(raw)
