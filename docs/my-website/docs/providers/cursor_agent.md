# Cursor Agent CLI (`cursor_agent`)

Direct LiteLLM provider for the **Cursor Agent CLI** Connect route. This is
**not** [Cursor Cloud Agents](/docs/pass_through/cursor).

| Property | Details |
|----------|---------|
| Provider | `cursor_agent` |
| Turn host | `https://agentn.global.api5.cursor.sh` |
| Turn RPC | `POST /agent.v1.AgentService/Run` (HTTP/2 Connect) |
| Dashboard / auth host | `https://api2.cursor.sh` |
| Credential | `CURSOR_AUTH_TOKEN` or `CURSOR_API_KEY` |
| Not used | `CURSOR_CLI_KEY`, Cloud Agents `cursor`, `CURSOR_API_BASE` |

Cloud Agents stay on `custom_llm_provider="cursor"` and
`LITELLM_PROXY_BASE_URL/cursor`. Do not send Agent CLI turns through
`/cursor` or `api.cursor.com`. This route is not `openai_like`.

## API key

The raw API key is **not** the request credential. LiteLLM sends
`Authorization: Bearer <accessToken>`.

```python
import os

os.environ["CURSOR_API_KEY"] = ""          # raw key; sent as Bearer until exchanged
# or
os.environ["CURSOR_AUTH_TOKEN"] = ""       # already an access token
```

`CURSOR_AUTH_TOKEN` is preferred over `CURSOR_API_KEY`. `CURSOR_CLI_KEY` is
ignored.

## Sample usage

```python
from litellm import completion
import os

os.environ["CURSOR_API_KEY"] = ""
response = completion(
    model="cursor_agent/composer-2.5",
    messages=[{"role": "user", "content": "hello from litellm"}],
)
print(response)
```

Supported first slugs: `composer-2.5`, `cursor-grok-4.6-high`. Pricing and
alias YAML are not part of this provider landing.

## Monthly usage

Account-scoped monthly included spend is read from Dashboard Connect
`POST /aiserver.v1.DashboardService/GetCurrentPeriodUsage` on
`https://api2.cursor.sh`. That RPC is not Cloud Agents `GET /v0/me` and
is not the `agentn` turn.

When the provider-status sidecar is explicitly enabled
(`AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED=1` or
`--cursor-agent-usage-poll-enabled`), it maps:

| Observation field | Dashboard field |
|---|---|
| `quota_used` | `planUsage.includedSpend` (USD cents) |
| `quota_limit` | `planUsage.limit` |
| `quota_remaining` | `planUsage.remaining` |
| `quota_period` | `monthly` |

The included fraction is `includedSpend / limit`. Do not treat
`totalPercentUsed` / `autoPercentUsed` / `apiPercentUsed` as
`totalSpend / limit`. Account identity is hashed. Failed refreshes keep
the last valid `public.rate_limit_observations` row.

Weekly Cursor Grok Bot used/limit/reset is still unknown. There is no
weekly `quota_key`. Do not treat xAI Grok Build weekly credits or BugBot
license RPCs as Grok Bot. Set
`AAWM_CURSOR_AGENT_GROK_BOT_USAGE_SOURCE` only when a verified weekly
source exists; until then the checkpoint stays unknown.

The poller is disabled by default so LiteLLM does not send live
dashboard traffic.

## What this is not

- Cloud Agents `/v0/agents` on `https://api.cursor.com`
- OpenAI `/v1/chat/completions`
- HTTP/1.1 `RunSSE` + `BidiAppend`
- A `cursor-agent` / `agent` subprocess
- Cloud Agents `GET /v0/me` usage
