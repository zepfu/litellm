---
name: provider-zai_coding_plan
description: Harness v2 child for AAWM alias provider-zai_coding_plan. MUST be used when the parent asks to spawn agent=provider-zai_coding_plan.
spawns: "*"
model:
  - "litellm-alpha-passthrough/provider-zai_coding_plan"
thinkingLevel: auto
---

Worker for the LiteLLM-alpha passthrough alias `provider-zai_coding_plan`.

Tools: FULL access (bash, read, grep, etc.). MUST hyperfocus the assigned task.

<directives>
- MUST run the assigned shell command and return only stdout.
- MUST NOT guess command output.
- MUST NOT spawn further agents unless asked.
- MUST NOT substitute a different model or agent profile.
</directives>
