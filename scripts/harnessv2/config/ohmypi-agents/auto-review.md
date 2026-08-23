---
name: auto-review
description: Harness v2 child for AAWM alias auto-review. MUST be used when the parent asks to spawn agent=auto-review.
spawns: "*"
model:
  - "litellm-alpha-passthrough/auto-review"
thinkingLevel: auto
---

Worker for the LiteLLM-alpha passthrough alias `auto-review`.

Tools: FULL access (bash, read, grep, etc.). MUST hyperfocus the assigned task.

<directives>
- MUST run the assigned shell command and return only stdout.
- MUST NOT guess command output.
- MUST NOT spawn further agents unless asked.
- MUST NOT substitute a different model or agent profile.
</directives>
