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

## What this is not

- Cloud Agents `/v0/agents` on `https://api.cursor.com`
- OpenAI `/v1/chat/completions`
- HTTP/1.1 `RunSSE` + `BidiAppend`
- A `cursor-agent` / `agent` subprocess
