## 2026-06-01 D1-123

- Suggestion: add a provider-cache repair mode that accepts a predicate such as
  `--partial-cache-hits-only` and avoids the unconditional
  `session_history_tool_activity` aggregate join.
- How it would help: this would reduce wall-clock time and DB churn for
  targeted reporting repairs. The general repair script was correct but too
  broad for a 21k-row xAI-only cache fix, forcing a manual SQL repair path.
