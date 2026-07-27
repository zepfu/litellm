"""AAWM request-policy sub-package (Wave 6D/6E extraction).

Submodules:
- observability_metadata: shared metadata primitives, session/repo, breakouts
- persisted_output: Claude persisted-output expansion and Google compaction
- alias_guidance: alias-specific system instruction shaping
- codex_tool_policy: Codex tool description patches, model-capability policy,
  custom/namespace tool adaptation, unsupported-tool/param/item drops
- claude_prompt_replacement: Claude auto-memory and prompt-patch replacements
- anthropic_body_prep: OpenAI-adapter context compaction, tool-block repair,
  final Anthropic request-body preparation
"""
