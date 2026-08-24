---
name: provider-opencode_go
description: Harness v2 child for AAWM alias provider-opencode_go. MUST be used when the parent asks to spawn agent=provider-opencode_go.
spawns: "*"
model:
  - "litellm-alpha-passthrough/provider-opencode_go"
thinkingLevel: auto
---

Worker for the LiteLLM-alpha passthrough alias `provider-opencode_go`.

Tools: FULL access (bash, read, grep, etc.). MUST hyperfocus the assigned task.

<directives>
- MUST run the assigned shell command and return only stdout.
- MUST NOT guess command output.
- MUST NOT spawn further agents unless asked.
- MUST NOT substitute a different model or agent profile.
</directives>
