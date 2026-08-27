# LiteLLM Alpha

`litellm-alpha` is the testing-only LiteLLM proxy for candidate work in this
checkout. It listens on port `4011`.

Never route production traffic, persistent clients, scheduled jobs, or normal
development traffic to this service. Do not use alpha results as production
deployment evidence. The container intentionally runs code that may be
incomplete, unreviewed, or broken.

## Compose Ownership

Alpha is managed exclusively by `docker-compose.alpha.yml`, which defines only
the `litellm-alpha` service. It manages only alpha: it never includes,
extends, depends on, lists, starts, stops, or recreates `litellm-dev`,
`provider-status-observations`, or the alias-routing Redis service. All
`docker compose` commands for alpha use
`docker compose -f docker-compose.alpha.yml ...`.

The file declares its own Compose project, `name: litellm-alpha`, separate
from the dev project, so its state and naming never collide with
`docker-compose.dev.yml`.

The file defines no Redis, database, or other helper services and declares no
`depends_on`. It preserves alpha's existing connectivity by attaching to two
networks that must already exist:

- `litellm_default` — the dev Compose project's default network, providing
  name resolution for the dev alias-routing Redis container
  (`aawm-alias-routing-redis`).
- `aawm-infrastructure_default` — the shared infrastructure network
  (PgBouncer and friends).

Before starting alpha, make sure those external networks and their dependency
services are already up (normally via `docker-compose.dev.yml` / the
infrastructure stack). Alpha uses its own alias-routing Redis namespace and
does not share candidate cooldown or affinity keys with `litellm-dev`.

## Runtime Contract

- Container: `litellm-alpha`
- Image: `litellm-alpha:local`
- Compose file: `docker-compose.alpha.yml` (alpha only)
- Local endpoint: `http://127.0.0.1:4011`
- Tailscale endpoint: `http://100.109.19.233:4011`
- Config: `/app/litellm-dev-config.yaml`, bind-mounted from this repository
- Source: `/app`, bind-mounted read-only from this repository
- Cursor GUI auth directory: `/home/zepfu/.config/cursor`, bind-mounted
  read-only at the same path; the directory mount keeps sidecar atomic
  auth-file replacement visible without recreating alpha
- Cursor auth path variable:
  `LITELLM_CURSOR_AGENT_AUTH_FILE=/home/zepfu/.config/cursor/auth.json`
- Proxy environment label:
  `AAWM_LITELLM_ENVIRONMENT=litellm-alpha`
- Alias-routing Redis namespace:
  `AAWM_ALIAS_ROUTING_STATE_NAMESPACE=aawm-routing-alpha-v1`
- Database application names:
  `AAWM_SESSION_HISTORY_DB_APPLICATION_NAME=aawm-litellm-alpha-session-history`,
  `AAWM_DYNAMIC_INJECTION_DB_APPLICATION_NAME=aawm-litellm-alpha-dynamic-injection`,
  and `PGAPPNAME=aawm-litellm-alpha-runtime`
- Error-log label: `LITELLM_AAWM_ERROR_LOG_ENV=alpha`
- Langfuse trace label: `LITELLM_LANGFUSE_TRACE_ENVIRONMENT=alpha`
- Session-history spool:
  `/app/.analysis/runtime/litellm-alpha/session_history`

The Cursor Agent provider reads the GUI auth file inside the mounted
directory. A fresh
`accessToken` is used directly; an `apiKey`/`api_key` is exchanged with
`https://api2.cursor.sh/auth/exchange_user_api_key`, and the returned
`accessToken` is used as the Agent bearer credential. Only the auth-file path
is supplied through Compose/environment; the auth JSON contents and raw API
key are never placed there.

The image contains the Python dependencies and an editable LiteLLM install.
At runtime, the repository is mounted over `/app`, and `PYTHONPATH=/app`
ensures imports resolve from the live checkout.

Alpha never mounts or scans the development session-history spool at
`/mnt/e/litellm/session_history`. Its durable queue fallback lives under the
repository's ignored `.analysis/runtime/litellm-alpha/` tree, so alpha cannot
claim, replay, quarantine, or delete `litellm-dev` spool records.

`watchfiles` restarts the LiteLLM process when files change under:

- `litellm/`
- `enterprise/`
- `litellm-dev-config.yaml`
- `model_prices_and_context_window.json`
- `context-replacement/`

Python and watched configuration changes therefore become active without
rebuilding or recreating the container. The process briefly becomes
unavailable on port `4011` while it restarts.

Changes to dependencies, `Dockerfile.alpha`, `requirements.txt`, or packaging
metadata require an image rebuild.

## Start

Use the same environment preparation required by `litellm-dev`, including the
two expected Codex OAuth account hashes. Live testing of the current temporary
`read` and `basic` aliases requires `ZAI_KEY`. Live `auto-review` /
`codex-auto-review` testing requires `ZAI_KEY` and `AAWM_OPENROUTER_API_KEY`;
alpha receives only the provider credentials present when its container is
created.

```bash
docker compose -f docker-compose.alpha.yml build litellm-alpha
docker compose -f docker-compose.alpha.yml up -d litellm-alpha
```

## Verify

```bash
docker compose -f docker-compose.alpha.yml ps litellm-alpha
docker compose -f docker-compose.alpha.yml logs --tail=100 litellm-alpha
curl --fail http://127.0.0.1:4011/health/liveliness
docker exec litellm-alpha python -c \
  'import litellm; print(litellm.__file__)'
```

The import path must resolve under `/app/litellm`.

When a watched file changes, the logs should show the existing LiteLLM process
stop and a new process start. Wait for `/health/liveliness` before running the
next live test.

## Current Alpha Acceptance

On August 27, 2026, current alpha live acceptance succeeded as follows:

- `read` and `basic` succeeded through `zai_coding_plan/glm-5.3-flash`.
- `work` succeeded through `cursor_agent/cursor-grok-4.6-high` without xAI
  fallback in the temporary Cursor-first work graph. This August 27, 2026
  result is historical and is not evidence for the current shared CFG-035/038
  graph.
- `expert` succeeded through `gpt-5.6-terra`.
- `sota-openai` (`gpt` role) succeeded through `gpt-5.6-sol`.

This is alpha acceptance only, not `litellm-dev`/production acceptance or
production evidence.

## Rebuild

```bash
docker compose -f docker-compose.alpha.yml build litellm-alpha
docker compose -f docker-compose.alpha.yml up -d --force-recreate litellm-alpha
```

Rebuild only when the image dependency layer or Dockerfile changed. Ordinary
source and watched configuration edits do not require this.

## Stop

Only the alpha service can be affected by this file, so `stop` and `down` here
never touch `litellm-dev` or any other service.

```bash
docker compose -f docker-compose.alpha.yml stop litellm-alpha
docker compose -f docker-compose.alpha.yml rm -f litellm-alpha
```

## Testing Boundary

Use alpha for short-lived live checks while candidate code is actively being
edited. Keep each test attributable to the current checkout state and recheck
health after every automatic restart.

Alpha shares the development proxy's read-only provider credentials and
development database connections for parity. Treat all calls as real provider
and development-data operations. It is isolated by port, container name,
environment label, process application names, and alias-routing Redis
namespace, but it is not a sandbox for destructive database or provider tests.
The Cursor auth-file wiring is testing-only. The August 27, 2026 `work`
acceptance is historical for the temporary Cursor-first work graph; it is not
evidence for the current shared CFG-035/038 graph or
`litellm-dev`/production acceptance.
