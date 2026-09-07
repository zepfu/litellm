# ChatGPT ordinary Chat usage collector

Standalone read-only collector for ordinary Chat history metadata. It reconstructs generation attempts, stores them in a local SQLite ledger, and reports activity by model family, calendar day, elapsed lookback, and explicitly defined quota windows.

This is an **independent usage ledger**, not an official quota meter. Unknown remaining values render as `Unknown`, never `0`. Work, Codex, API, image, voice, Deep Research, and Agent Mode counters stay out of Chat totals.

## Quick start

```bash
python -m scripts.chatgpt_chat_usage_capture --config scripts/chatgpt_chat_usage_capture/config.example.yaml init
python -m scripts.chatgpt_chat_usage_capture --config scripts/chatgpt_chat_usage_capture/config.yaml --fixture-root scripts/chatgpt_chat_usage_capture/tests/fixtures inspect-capabilities
python -m scripts.chatgpt_chat_usage_capture --config scripts/chatgpt_chat_usage_capture/config.yaml --fixture-root scripts/chatgpt_chat_usage_capture/tests/fixtures refresh
python -m scripts.chatgpt_chat_usage_capture --config scripts/chatgpt_chat_usage_capture/config.yaml report --since 7d
python -m scripts.chatgpt_chat_usage_capture --config scripts/chatgpt_chat_usage_capture/config.yaml dashboard
```

`config.yaml` is the seed fixture-oriented policy (`pro200-*-chat` buckets). `config.example.yaml` matches the build-spec bucket IDs (`astra_weekly`, `sol_daily`, `pro_combined_daily`). Accounting is policy-driven; both shapes are valid.

## Commands

Lookback reports (`--since 7d`, `--since 24h`) are elapsed-time activity windows. They are not quota weeks. `--window` / `windows set-explicit` define actual quota bounds with evidence and reason. `quota record-observation` stores a remaining snapshot and **does not** set a window start.

Local API binds to loopback. When `application.local_api_auth` is `required` (default), send `Authorization: Bearer <token>` or `X-API-Token`. Browser POSTs from non-loopback origins are rejected.

See `docs/` for architecture, counting, reset windows, the operating runbook, and known limitations.
