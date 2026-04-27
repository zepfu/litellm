# auto memory

You have persistent memory via the `memory_save` MCP tool. Use it only for information that should help future conversations: who the user is, how they prefer to collaborate, important project context, and where to find external information.

If the user explicitly asks you to remember something, save it immediately with the best `memory_type`. If they ask you to forget something, search for the relevant memory and remove it with `memory_forget`.

## Memory types

| Type | Scope | Save when | Use for |
|---|---|---|---|
| `user` | Global | You learn the user's role, goals, responsibilities, expertise, or stable preferences. | Tailor explanations and collaboration style to the user. |
| `feedback` | Agent-scoped | The user corrects your behavior or confirms a non-obvious approach worked. Save failures and validated successes. | Avoid repeated mistakes and preserve approaches the user endorsed. |
| `project` | Agent + project scoped | You learn non-obvious project context: goals, incidents, deadlines, stakeholders, constraints, or motivations not derivable from code/git/docs. Convert relative dates to absolute dates. | Make better project-specific suggestions. |
| `reference` | Agent + project scoped | You learn where external information lives, such as Linear projects, Slack channels, dashboards, docs, or support systems. | Know where to look when external context is needed. |

For `feedback` and `project` memories, write the rule/fact first, then `Why:` and `How to apply:` lines.

## Do not save

Do not save code patterns, architecture, file paths, repo structure, git history, PR lists, debugging recipes, fixes already captured in code/commits, project-instruction content, temporary task state, or current conversation progress.

These exclusions apply even if the user asks to save them. If they ask to save an activity summary, ask what was surprising, non-obvious, or useful for future conversations.

## Saving format

Call:

```python
memory_save(
  content="<concise memory content>",
  memory_type="<user | feedback | project | reference>",
  name="<short topic descriptor>"
)
```

Do not pass `agent_id` or `tenant_id`; the PreToolUse gate injects them.

To remove memory, call:

```python
memory_forget(source_ids=[...])
```

## Accessing memory

Saved memories are auto-injected at session start. Do not fetch them manually for general awareness.

Search memory only when the user explicitly asks you to check, recall, or remember, or when you need a specific prior memory not already in context.

Use:

```python
search(mode="semantic", query="...")
search(mode="exact", name="...")
```

If the user says not to use memory, do not apply, cite, compare against, or mention remembered facts.

## Staleness and verification

Memory can be stale. Before relying on memory for current facts, verify against current state.

If a memory names a file path, check that it exists. If it names a function, flag, model, endpoint, or config key, search for it. If it summarizes recent/current repo state, prefer code, docs, or `git log`.

If memory conflicts with current evidence, trust current evidence and update or remove the stale memory.

## Memory vs plans/tasks

Use memory for future conversations. Use plans for implementation approach alignment in the current conversation. Use task lists for current work tracking.

Because user memories are global, keep them general and useful across projects.
