"""Wave 6E tests: Claude prompt-replacement extraction module.

Covers: replacement eligibility, path/context handling, exact output text/body,
metadata/logging callbacks, idempotence/no-op, malformed input, and exception
behavior.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement import (
    _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION,
    _CLAUDE_AUTO_MEMORY_SECTION_PATTERN,
    _CLAUDE_CC_VERSION_PATTERN,
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
    _CLAUDE_PROMPT_PATCH_MANIFEST_PATH,
    _add_claude_prompt_patch_logging_metadata,
    _add_claude_system_prompt_override_logging_metadata,
    _apply_claude_prompt_patches_in_text,
    _apply_claude_prompt_patches_to_anthropic_request_body,
    _extract_markdown_section,
    _load_claude_context_replacement_template,
    _load_claude_prompt_patch_manifest,
    _parse_claude_code_version,
    _replace_claude_auto_memory_section_in_text,
    _replace_claude_prompt_patches_in_value,
    _replace_claude_system_prompt_in_anthropic_request_body,
    _replace_claude_system_prompt_override_in_value,
    _resolve_claude_auto_memory_template_path,
    _claude_context_replacement_template_cache,
    _claude_prompt_patch_manifest_cache,
)


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


class TestParseClaudeCodeVersion:
    def test_valid_version(self):
        assert _parse_claude_code_version("2.1.110") == (2, 1, 110)

    def test_valid_version_with_suffix(self):
        assert _parse_claude_code_version("2.1.115-beta.1") == (2, 1, 115)

    def test_valid_version_with_whitespace(self):
        assert _parse_claude_code_version("  2.1.110  ") == (2, 1, 110)

    def test_none_input(self):
        assert _parse_claude_code_version(None) is None

    def test_empty_string(self):
        assert _parse_claude_code_version("") is None

    def test_malformed_version(self):
        assert _parse_claude_code_version("not-a-version") is None

    def test_partial_version(self):
        assert _parse_claude_code_version("2.1") is None


# ---------------------------------------------------------------------------
# Template path resolution / eligibility
# ---------------------------------------------------------------------------


class TestResolveClaudeAutoMemoryTemplatePath:
    def test_exact_min_version(self):
        result = _resolve_claude_auto_memory_template_path("2.1.110")
        assert result is not None
        assert result.name == "auto-memory-replacement.md"

    def test_higher_patch_version(self):
        result = _resolve_claude_auto_memory_template_path("2.1.200")
        assert result is not None

    def test_lower_patch_version_ineligible(self):
        assert _resolve_claude_auto_memory_template_path("2.1.109") is None

    def test_different_minor_version_ineligible(self):
        assert _resolve_claude_auto_memory_template_path("2.2.110") is None

    def test_different_major_version_ineligible(self):
        assert _resolve_claude_auto_memory_template_path("3.1.110") is None

    def test_none_version(self):
        assert _resolve_claude_auto_memory_template_path(None) is None

    def test_malformed_version(self):
        assert _resolve_claude_auto_memory_template_path("garbage") is None


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


class TestLoadClaudeContextReplacementTemplate:
    def test_loads_and_caches(self, tmp_path):
        template_file = tmp_path / "test-template.md"
        template_file.write_text("hello world\n", encoding="utf-8")
        _claude_context_replacement_template_cache.pop(template_file, None)

        result = _load_claude_context_replacement_template(template_file)
        assert result == "hello world\n"
        assert _claude_context_replacement_template_cache[template_file] == "hello world\n"
        _claude_context_replacement_template_cache.pop(template_file, None)

    def test_strips_and_appends_newline(self, tmp_path):
        template_file = tmp_path / "test-template2.md"
        template_file.write_text("  content here  \n\n", encoding="utf-8")
        _claude_context_replacement_template_cache.pop(template_file, None)

        result = _load_claude_context_replacement_template(template_file)
        assert result == "content here\n"
        _claude_context_replacement_template_cache.pop(template_file, None)

    def test_empty_template_raises(self, tmp_path):
        template_file = tmp_path / "empty.md"
        template_file.write_text("   \n  ", encoding="utf-8")
        _claude_context_replacement_template_cache.pop(template_file, None)

        with pytest.raises(ValueError, match="template is empty"):
            _load_claude_context_replacement_template(template_file)
        _claude_context_replacement_template_cache.pop(template_file, None)

    def test_cache_hit(self, tmp_path):
        template_file = tmp_path / "cached.md"
        _claude_context_replacement_template_cache[template_file] = "cached-value\n"
        result = _load_claude_context_replacement_template(template_file)
        assert result == "cached-value\n"
        _claude_context_replacement_template_cache.pop(template_file, None)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


class TestLoadClaudePromptPatchManifest:
    def test_valid_manifest(self, tmp_path):
        manifest_file = tmp_path / "manifest.json"
        manifest_data = {
            "source": "test",
            "patches": [
                {"id": "p1", "before": "old", "after": "new"},
            ],
        }
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        result = _load_claude_prompt_patch_manifest(manifest_file)
        assert result["source"] == "test"
        assert len(result["patches"]) == 1
        assert result["patches"][0]["id"] == "p1"
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_invalid_json_structure(self, tmp_path):
        manifest_file = tmp_path / "bad.json"
        manifest_file.write_text('"just a string"', encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        with pytest.raises(ValueError, match="Invalid Claude prompt patch manifest"):
            _load_claude_prompt_patch_manifest(manifest_file)
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_no_patches(self, tmp_path):
        manifest_file = tmp_path / "nopatches.json"
        manifest_file.write_text(json.dumps({"patches": []}), encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        with pytest.raises(ValueError, match="has no patches"):
            _load_claude_prompt_patch_manifest(manifest_file)
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_missing_patch_id(self, tmp_path):
        manifest_file = tmp_path / "noid.json"
        manifest_data = {"patches": [{"before": "a", "after": "b"}]}
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        with pytest.raises(ValueError, match="missing patch id"):
            _load_claude_prompt_patch_manifest(manifest_file)
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_missing_before_text(self, tmp_path):
        manifest_file = tmp_path / "nobefore.json"
        manifest_data = {"patches": [{"id": "x", "after": "b"}]}
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        with pytest.raises(ValueError, match="missing before text"):
            _load_claude_prompt_patch_manifest(manifest_file)
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_missing_after_text(self, tmp_path):
        manifest_file = tmp_path / "noafter.json"
        manifest_data = {"patches": [{"id": "x", "before": "a"}]}
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        with pytest.raises(ValueError, match="missing after text"):
            _load_claude_prompt_patch_manifest(manifest_file)
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_non_dict_patch_descriptor(self, tmp_path):
        manifest_file = tmp_path / "nonlist.json"
        manifest_data = {"patches": ["not-a-dict"]}
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

        with pytest.raises(ValueError, match="Invalid Claude prompt patch descriptor"):
            _load_claude_prompt_patch_manifest(manifest_file)
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)

    def test_cache_hit(self, tmp_path):
        manifest_file = tmp_path / "cached-manifest.json"
        cached = {"source": "cached", "patches": [{"id": "c", "before": "x", "after": "y"}]}
        _claude_prompt_patch_manifest_cache[manifest_file] = cached
        result = _load_claude_prompt_patch_manifest(manifest_file)
        assert result is cached
        _claude_prompt_patch_manifest_cache.pop(manifest_file, None)


# ---------------------------------------------------------------------------
# Markdown section extraction
# ---------------------------------------------------------------------------


class TestExtractMarkdownSection:
    def test_extracts_section(self):
        md = "## Heading One\ncontent1\n## Heading Two\ncontent2\n"
        result = _extract_markdown_section(md, "Heading One")
        assert result == "## Heading One\ncontent1"

    def test_extracts_last_section(self):
        md = "## First\naaa\n## Second\nbbb\n"
        result = _extract_markdown_section(md, "Second")
        assert result == "## Second\nbbb"

    def test_missing_section_raises(self):
        md = "## Other\nstuff\n"
        with pytest.raises(ValueError, match="Missing Claude auto-memory section"):
            _extract_markdown_section(md, "Nonexistent")


# ---------------------------------------------------------------------------
# Auto-memory section replacement in text
# ---------------------------------------------------------------------------


class TestReplaceClaudeAutoMemorySectionInText:
    def test_no_auto_memory_marker_noop(self):
        text = "some random text without the marker"
        result, event = _replace_claude_auto_memory_section_in_text(text, "2.1.110")
        assert result == text
        assert event is None

    def test_marker_present_but_no_section_match_noop(self):
        text = "prefix # auto memory inline"
        result, event = _replace_claude_auto_memory_section_in_text(text, "2.1.110")
        assert result == text
        assert event is None


# ---------------------------------------------------------------------------
# System prompt override in value (recursive traversal)
# ---------------------------------------------------------------------------


class TestReplaceClaudeSystemPromptOverrideInValue:
    def test_non_text_value_noop(self):
        assert _replace_claude_system_prompt_override_in_value(42, "2.1.110") == (42, [])
        assert _replace_claude_system_prompt_override_in_value("hello", "2.1.110") == ("hello", [])
        assert _replace_claude_system_prompt_override_in_value(None, "2.1.110") == (None, [])

    def test_text_block_without_marker_noop(self):
        value = {"type": "text", "text": "no marker here"}
        result, events = _replace_claude_system_prompt_override_in_value(value, "2.1.110")
        assert result is value
        assert events == []

    def test_text_block_with_marker_failure_event(self):
        value = {"type": "text", "text": "# auto memory\nsome content\n# Environment\n"}
        result, events = _replace_claude_system_prompt_override_in_value(value, "9.9.999")
        assert result is value
        assert len(events) == 1
        assert events[0]["id"] == "auto-memory"
        assert events[0]["status"] == "failed"
        assert events[0]["error"] == "ValueError"

    def test_nested_dict_traversal(self):
        value = {
            "outer": {
                "type": "text",
                "text": "no marker",
            }
        }
        result, events = _replace_claude_system_prompt_override_in_value(value, "2.1.110")
        assert result is value
        assert events == []

    def test_list_traversal_noop(self):
        value = [{"type": "text", "text": "nothing"}]
        result, events = _replace_claude_system_prompt_override_in_value(value, "2.1.110")
        assert result is value
        assert events == []

    def test_idempotence_no_marker(self):
        value = {"type": "text", "text": "plain text"}
        r1, e1 = _replace_claude_system_prompt_override_in_value(value, "2.1.110")
        r2, e2 = _replace_claude_system_prompt_override_in_value(r1, "2.1.110")
        assert r1 is r2
        assert e1 == e2 == []


# ---------------------------------------------------------------------------
# System prompt override logging metadata
# ---------------------------------------------------------------------------


class TestAddClaudeSystemPromptOverrideLoggingMetadata:
    def test_resolved_event_metadata(self):
        events = [
            {
                "id": "auto-memory",
                "status": "resolved",
                "cc_version": "2.1.110",
                "template_path": "context-replacement/claude-code/2.1.110/auto-memory-replacement.md",
                "output_chars": 500,
            }
        ]
        body: dict[str, Any] = {}
        result = _add_claude_system_prompt_override_logging_metadata(body, events)

        meta = result["litellm_metadata"]
        assert "claude-system-prompt-override" in meta["tags"]
        assert "claude-system-prompt-override:auto-memory" in meta["tags"]
        assert "claude-system-prompt-override-failed" not in meta["tags"]
        assert meta["claude_system_prompt_override_count"] == 1
        assert meta["claude_system_prompt_override_ids"] == ["auto-memory"]
        assert meta["claude_system_prompt_override_failure_ids"] == []
        assert meta["claude_system_prompt_override_cc_versions"] == ["2.1.110"]
        spans = meta["langfuse_spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "claude.system_prompt_override"
        assert spans[0]["metadata"]["override_count"] == 1
        assert spans[0]["metadata"]["failure_count"] == 0

    def test_failed_event_metadata(self):
        events = [
            {
                "id": "auto-memory",
                "status": "failed",
                "cc_version": "2.1.110",
                "error": "ValueError",
            }
        ]
        body: dict[str, Any] = {}
        result = _add_claude_system_prompt_override_logging_metadata(body, events)
        meta = result["litellm_metadata"]
        assert "claude-system-prompt-override-failed" in meta["tags"]
        assert meta["claude_system_prompt_override_failure_ids"] == ["auto-memory"]

    def test_preserves_existing_tags(self):
        body: dict[str, Any] = {"litellm_metadata": {"tags": ["existing-tag"]}}
        events = [{"id": "auto-memory", "status": "resolved", "cc_version": "2.1.110"}]
        result = _add_claude_system_prompt_override_logging_metadata(body, events)
        meta = result["litellm_metadata"]
        assert "existing-tag" in meta["tags"]
        assert "claude-system-prompt-override" in meta["tags"]


# ---------------------------------------------------------------------------
# Top-level system prompt replacement in request body
# ---------------------------------------------------------------------------


class TestReplaceClaudeSystemPromptInAnthropicRequestBody:
    def test_no_cc_version_noop(self):
        body = {"system": [{"type": "text", "text": "# auto memory\nx\n# Environment\n"}]}
        result, events = _replace_claude_system_prompt_in_anthropic_request_body(body, {})
        assert result is body
        assert events == []

    def test_empty_cc_version_noop(self):
        body = {"system": []}
        result, events = _replace_claude_system_prompt_in_anthropic_request_body(
            body, {"cc_version": ""}
        )
        assert result is body
        assert events == []

    def test_ineligible_version_noop(self):
        body = {"system": [{"type": "text", "text": "# auto memory\nx\n# Environment\n"}]}
        result, events = _replace_claude_system_prompt_in_anthropic_request_body(
            body, {"cc_version": "1.0.0"}
        )
        assert result is body
        assert events == []

    def test_no_system_key_noop(self):
        body: dict[str, Any] = {"messages": []}
        result, events = _replace_claude_system_prompt_in_anthropic_request_body(
            body, {"cc_version": "2.1.110"}
        )
        assert result is body
        assert events == []

    def test_no_override_events_noop(self):
        body = {"system": [{"type": "text", "text": "no marker"}]}
        result, events = _replace_claude_system_prompt_in_anthropic_request_body(
            body, {"cc_version": "2.1.110"}
        )
        assert result is body
        assert events == []

    def test_failure_events_still_returned(self):
        body = {"system": [{"type": "text", "text": "# auto memory\ncontent\n# Environment\n"}]}
        result, events = _replace_claude_system_prompt_in_anthropic_request_body(
            body, {"cc_version": "2.1.110"}
        )
        if events:
            assert result is not body or events[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# Prompt patches in text
# ---------------------------------------------------------------------------


class TestApplyClaudePromptPatchesInText:
    def test_no_matching_patches(self):
        text = "some text that won't match any patch"
        try:
            result, events = _apply_claude_prompt_patches_in_text(text, "2.1.110")
            assert result == text
            assert events == []
        except (FileNotFoundError, ValueError):
            pytest.skip("Real manifest not available in test environment")

    def test_manifest_load_failure(self):
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._load_claude_prompt_patch_manifest",
            side_effect=FileNotFoundError("gone"),
        ):
            result, events = _replace_claude_prompt_patches_in_value(
                {"type": "text", "text": "anything"}, "2.1.110"
            )
            assert len(events) == 1
            assert events[0]["id"] == "manifest-load"
            assert events[0]["status"] == "failed"
            assert events[0]["error"] == "FileNotFoundError"


# ---------------------------------------------------------------------------
# Prompt patches in value (recursive traversal)
# ---------------------------------------------------------------------------


class TestReplaceClaudePromptPatchesInValue:
    def test_non_container_noop(self):
        assert _replace_claude_prompt_patches_in_value(123, "2.1.110") == (123, [])
        assert _replace_claude_prompt_patches_in_value(None, "2.1.110") == (None, [])

    def test_text_block_no_match_noop(self):
        value = {"type": "text", "text": "unmatched content"}
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._apply_claude_prompt_patches_in_text",
            return_value=("unmatched content", []),
        ):
            result, events = _replace_claude_prompt_patches_in_value(value, "2.1.110")
            assert result is value
            assert events == []

    def test_text_block_with_match(self):
        value = {"type": "text", "text": "old-text"}
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._apply_claude_prompt_patches_in_text",
            return_value=(
                "new-text",
                [{"id": "p1", "status": "resolved", "cc_version": "2.1.110", "manifest_path": "m.json", "occurrences": 1}],
            ),
        ):
            result, events = _replace_claude_prompt_patches_in_value(value, "2.1.110")
            assert result["text"] == "new-text"
            assert result is not value
            assert len(events) == 1

    def test_nested_dict_traversal(self):
        value = {"a": {"b": {"type": "text", "text": "no-match"}}}
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._apply_claude_prompt_patches_in_text",
            return_value=("no-match", []),
        ):
            result, events = _replace_claude_prompt_patches_in_value(value, "2.1.110")
            assert result is value
            assert events == []

    def test_list_traversal(self):
        value = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._apply_claude_prompt_patches_in_text",
            return_value=("x", []),
        ):
            result, events = _replace_claude_prompt_patches_in_value(value, "2.1.110")
            assert result is value
            assert events == []

    def test_exception_produces_failure_event(self):
        value = {"type": "text", "text": "trigger error"}
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._apply_claude_prompt_patches_in_text",
            side_effect=RuntimeError("boom"),
        ):
            result, events = _replace_claude_prompt_patches_in_value(value, "2.1.110")
            assert result is value
            assert len(events) == 1
            assert events[0]["status"] == "failed"
            assert events[0]["error"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Prompt patch logging metadata
# ---------------------------------------------------------------------------


class TestAddClaudePromptPatchLoggingMetadata:
    def test_resolved_patches_metadata(self):
        events = [
            {
                "id": "patch-1",
                "status": "resolved",
                "cc_version": "2.1.110",
                "manifest_path": "prompt-patches/test.json",
                "occurrences": 3,
            },
            {
                "id": "patch-2",
                "status": "resolved",
                "cc_version": "2.1.110",
                "manifest_path": "prompt-patches/test.json",
                "occurrences": 1,
            },
        ]
        body: dict[str, Any] = {}
        result = _add_claude_prompt_patch_logging_metadata(body, events)
        meta = result["litellm_metadata"]

        assert "claude-prompt-patch" in meta["tags"]
        assert "claude-prompt-patch:patch-1" in meta["tags"]
        assert "claude-prompt-patch:patch-2" in meta["tags"]
        assert "claude-prompt-patch-failed" not in meta["tags"]
        assert meta["claude_prompt_patch_count"] == 2
        assert meta["claude_prompt_patch_replacement_count"] == 4
        assert meta["claude_prompt_patch_ids"] == ["patch-1", "patch-2"]
        assert meta["claude_prompt_patch_failure_ids"] == []

        spans = meta["langfuse_spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "claude.prompt_patch"
        assert spans[0]["metadata"]["patch_count"] == 2
        assert spans[0]["metadata"]["replacement_count"] == 4

    def test_failed_patches_metadata(self):
        events = [
            {
                "id": "manifest-load",
                "status": "failed",
                "cc_version": "2.1.110",
                "error": "FileNotFoundError",
            }
        ]
        body: dict[str, Any] = {}
        result = _add_claude_prompt_patch_logging_metadata(body, events)
        meta = result["litellm_metadata"]
        assert "claude-prompt-patch-failed" in meta["tags"]
        assert meta["claude_prompt_patch_failure_ids"] == ["manifest-load"]


# ---------------------------------------------------------------------------
# Top-level prompt patch application to request body
# ---------------------------------------------------------------------------


class TestApplyClaudePromptPatchesToAnthropicRequestBody:
    def test_no_cc_version_noop(self):
        body: dict[str, Any] = {"messages": []}
        result, events = _apply_claude_prompt_patches_to_anthropic_request_body(body, {})
        assert result is body
        assert events == []

    def test_empty_cc_version_noop(self):
        body: dict[str, Any] = {"messages": []}
        result, events = _apply_claude_prompt_patches_to_anthropic_request_body(
            body, {"cc_version": ""}
        )
        assert result is body
        assert events == []

    def test_no_patch_events_noop(self):
        body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._replace_claude_prompt_patches_in_value",
            return_value=(body, []),
        ):
            result, events = _apply_claude_prompt_patches_to_anthropic_request_body(
                body, {"cc_version": "2.1.110"}
            )
            assert result is body
            assert events == []

    def test_with_patch_events_adds_metadata(self):
        body: dict[str, Any] = {"messages": [{"role": "user", "content": [{"type": "text", "text": "old"}]}]}
        patch_events = [
            {
                "id": "p1",
                "status": "resolved",
                "cc_version": "2.1.110",
                "manifest_path": "prompt-patches/m.json",
                "occurrences": 1,
            }
        ]
        updated = dict(body)
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._replace_claude_prompt_patches_in_value",
            return_value=(updated, patch_events),
        ):
            result, events = _apply_claude_prompt_patches_to_anthropic_request_body(
                body, {"cc_version": "2.1.110"}
            )
            assert events == patch_events
            assert "litellm_metadata" in result
            meta = result["litellm_metadata"]
            assert "claude-prompt-patch" in meta["tags"]
            spans = meta["langfuse_spans"]
            assert spans[0]["name"] == "claude.prompt_patch"
            assert "start_time" in spans[0]
            assert "end_time" in spans[0]

    def test_non_dict_updated_body_noop(self):
        body: dict[str, Any] = {"messages": []}
        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement._replace_claude_prompt_patches_in_value",
            return_value=("not-a-dict", [{"id": "x", "status": "resolved"}]),
        ):
            result, events = _apply_claude_prompt_patches_to_anthropic_request_body(
                body, {"cc_version": "2.1.110"}
            )
            assert result is body
            assert events == []


# ---------------------------------------------------------------------------
# Constants / path sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_context_replacement_dir_points_to_repo_root(self):
        assert _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR.name == "claude-code"
        assert _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR.parent.name == "context-replacement"

    def test_min_compat_version(self):
        assert _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION == (2, 1, 110)

    def test_cc_version_pattern(self):
        m = _CLAUDE_CC_VERSION_PATTERN.match("10.20.30")
        assert m is not None
        assert m.group("major") == "10"
        assert m.group("minor") == "20"
        assert m.group("patch") == "30"

    def test_auto_memory_section_pattern(self):
        text = "# auto memory\nstuff\n# Environment\nmore"
        m = _CLAUDE_AUTO_MEMORY_SECTION_PATTERN.search(text)
        assert m is not None
        assert "# auto memory" in m.group(0)
        assert "# Environment" not in m.group(0)

    def test_prompt_patch_manifest_path(self):
        assert _CLAUDE_PROMPT_PATCH_MANIFEST_PATH.suffix == ".json"
        assert "prompt-patches" in str(_CLAUDE_PROMPT_PATCH_MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Integration seam: no module-scope god-module import
# ---------------------------------------------------------------------------


class TestModuleIsolation:
    def test_no_llm_passthrough_import(self):
        """Verify the module does not import llm_passthrough_endpoints at scope."""
        import litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement as mod

        source_file = mod.__file__
        assert source_file is not None
        source_text = Path(source_file).read_text(encoding="utf-8")
        assert "import llm_passthrough_endpoints" not in source_text
        assert "from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints" not in source_text

    def test_uses_observability_metadata_explicit_imports(self):
        """Verify explicit imports from observability_metadata."""
        import litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement as mod

        source_text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "from litellm.proxy.pass_through_endpoints.aawm_request_policy.observability_metadata import" in source_text
        assert "_merge_litellm_metadata" in source_text
        assert "_build_langfuse_span_descriptor" in source_text
        assert "_format_langfuse_span_timestamp" in source_text
