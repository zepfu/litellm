"""CFG-073 legacy Claude prompt-replacement wrapper.

Preserves the exact ``ClaudePromptReplacementServices`` contract and all
symbols that ``aawm_claude_control_plane.py`` imports from
``claude_prompt_replacement.py``.  This module re-exports those symbols
unchanged so no Claude control-plane caller is affected.

New shared engine functionality lives in ``prompt_replacement_engine.py``.
"""

from __future__ import annotations

from litellm.proxy.pass_through_endpoints.aawm_request_policy.claude_prompt_replacement import (  # noqa: F401
    _CLAUDE_AUTO_MEMORY_MIN_COMPAT_VERSION,
    _CLAUDE_AUTO_MEMORY_SECTION_PATTERN,
    _CLAUDE_AUTO_MEMORY_TEMPLATE_LOGICAL_PATH,
    _CLAUDE_AUTO_MEMORY_TEMPLATE_PATH,
    _CLAUDE_CC_VERSION_PATTERN,
    _CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
    _CLAUDE_CONTEXT_REPLACEMENT_PLACEHOLDER_PATTERN,
    _CLAUDE_MEMORY_SECTION_PATTERN,
    _CLAUDE_PROMPT_PATCH_MANIFEST_LOGICAL_PATH,
    _CLAUDE_PROMPT_PATCH_MANIFEST_PATH,
    _CLAUDE_REPORT_FILE_INSTRUCTION_PATTERN,
    _CLAUDE_REPORT_FILE_PATCH_ID,
    _CLAUDE_TYPES_XML_BLOCK_PATTERN,
    _PACKAGED_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
    _REPO_CLAUDE_CODE_CONTEXT_REPLACEMENT_DIR,
    ClaudePromptReplacementServices,
    _add_claude_prompt_patch_logging_metadata,
    _add_claude_system_prompt_override_logging_metadata,
    _apply_claude_prompt_patch_manifest_to_text,
    _apply_claude_prompt_patches_in_text,
    _apply_claude_prompt_patches_to_anthropic_request_body,
    _candidate_context_replacement_dirs,
    _claude_context_replacement_template_cache,
    _claude_prompt_patch_manifest_cache,
    _extract_markdown_section,
    _load_claude_context_replacement_template,
    _load_claude_prompt_patch_manifest,
    _parse_claude_code_version,
    _render_claude_auto_memory_replacement,
    _replace_claude_auto_memory_section_in_text,
    _replace_claude_prompt_patches_in_value,
    _replace_claude_system_prompt_in_anthropic_request_body,
    _replace_claude_system_prompt_override_in_value,
    _resolve_claude_auto_memory_template_path,
    _resolve_claude_prompt_patch_manifest_path,
    _resolve_context_replacement_file,
    build_claude_prompt_replacement_services,
)
