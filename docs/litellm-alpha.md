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
- Config: `/app/litellm-alpha-config.yaml`, bind-mounted from this
  repository (`litellm-alpha-config.yaml`). Do not load
  `litellm-dev-config.yaml` in the alpha process; that file remains the
  `:4001` config.
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
- `litellm-alpha-config.yaml`
- `model_prices_and_context_window.json`
- `context-replacement/`

Python and watched configuration changes therefore become active without
rebuilding or recreating the container. The process briefly becomes
unavailable on port `4011` while it restarts.

Changes to dependencies, `Dockerfile.alpha`, `requirements.txt`, or packaging
metadata require an image rebuild.

## Routing Redis Persistence

The shared alias-routing Redis is owned by the dev Compose file and is not part
of the alpha-only Compose project. It uses AOF-only persistence. The exact
command `["redis-server", "--save", "", "--appendonly", "yes"]` uses an empty
save schedule to explicitly disable RDB.

State is stored in the existing `aawm_alias_routing_redis_data` named volume.
The isolated routing namespaces are `aawm-routing-dev-v1` for dev,
`aawm-routing-alpha-v1` for alpha, and `aawm-routing-prod-v1` for production.
Both Compose consumers retain their established `AAWM_ALIAS_ROUTING_REDIS_DB`
default of `0`; production selects its runtime-configured database when its own
deployment supplies that variable. Keys survive Redis restarts and container
recreation only within their configured namespace and database. Recreating the
shared Redis service requires explicit operator approval because it can disrupt
all consumers of that volume.

## Start

Use the same environment preparation required by `litellm-dev`, including the
two expected Codex OAuth account hashes. Live testing of the current temporary
role aliases requires the credentials for the candidates being exercised.
Live `auto-review` / `codex-auto-review` testing requires `ZAI_KEY` and
`AAWM_OPENROUTER_API_KEY`; alpha receives only the provider credentials present
when its container is created.

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

## Current Alpha Routing

The current canonical alias YAML exposes these alpha test paths:

- `basic` orders native Cohere `cohere/north-mini-code-1-0`, then OpenRouter
  Cohere `openrouter/cohere/north-mini-code:free`, then OpenCode Zen
  `big-pickle`, then `basic-other`. The helper adds a nightly Alibaba
  promotion, Z.AI Coding Plan, Cursor Composer, and mutually exclusive
  Luna/Haiku tails.
- `work` references `work-other`, then keeps its Claude-only native Sonnet
  leaves and OpenAI `gpt-5.6-luna`. `work-other` orders the nightly DeepSeek
  promotion, Z.AI Coding Plan, Moonshot, then xAI.
- `expert` references `expert-other` before OpenAI `gpt-5.6-terra`;
  `expert-other` orders nightly Alibaba Qwen Max, Cursor Grok, then native xAI.
- `auto-review` references `auto-review-other`, then Luna and OpenRouter
  DeepSeek. `codex-auto-review` references that same public graph.
- `sota-openai` uses OpenAI `gpt-5.6-sol`.
- `sota-xai` orders Cursor Agent
  `cursor_agent/cursor-grok-4.6-high`, native xAI/OIDC
  `xai/grok-4.6`, then managed xAI/OAuth `oa_xai/grok-4.6`.
  `UserMessageAction.requestContext` is official schema field 2. For a fresh
  clean alias dispatch, LiteLLM intentionally sends it as an empty message
  (`requestContext: {}`). The official CLI normally computes and populates
  `RequestContext` from local context; it does not itself prove an explicit
  `{}` send. AgentRunRequest field 12 is never sent, and the direct provider
  route remains available.

These are alpha-only testing paths for the current checkout. A passing alpha
call is not `litellm-dev` or production acceptance, deployment evidence, or
authorization to promote candidates or configuration to either environment.

For OpenRouter free candidates, the literal
`free-models-per-day-high-balance` provider error is classified as
OpenRouter-only `usage_limit_reached`, with exact candidate-scoped cooldown
and failover. It is separate from direct-provider quota state. Fresh
replay-safe provider-attributed transient `408`, `500`, `502`, `503`, `504`,
and `529` failures remain finite and request-local; they do not establish a
durable provider-wide cooldown, and stateful requests preserve affinity.

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
The Cursor auth-file wiring is testing-only. Alpha results remain confined to
the alpha service and current checkout; they do not promote candidates or
configuration to `litellm-dev` or production.

## Muse Code (alpha only)

Muse Code may use alpha as `--base-url`. Production (`:4000`) and
development (`:4001`) are out of scope for this work. Do not point Muse
at `musel` or `muselt` to exercise the alpha route.

### `musela`

Host launchers (Fish, Bash, and `~/.local/bin/musela`) pin Muse
`endpoint_transport` to alpha and do **not** pass `--base-url`. Muse
1.1.1 withholds the Meta bearer on a `--base-url` flag unless the same
origin is pinned in settings with `auth = "bearer"`; a settings pin
does not vouch for `--base-url`.

```bash
musela
# XDG_CONFIG_HOME=~/.config/aawm-musela muse
# settings.endpoint_transport.base_url =
#   http://litellm-dev.tailf1878c.ts.net:4011
# settings.endpoint_transport.auth = bearer
```

Override with `AAWM_MUSE_LITELLM_ALPHA_URL`. The value must be an origin
only (`http://host:4011`), not `.../v1` and not `.../muse-code`. Muse
then calls `GET /muse-code/models` and `POST /responses` on that origin.
Native `muse` (no overlay) stays on Meta's front door. `musel` /
`muselt` still pass `--base-url` and are not this route.

Confirm the function in a fresh shell:

```bash
bash -lc 'type musela'
fish -c 'type musela'
```

Default client model: `muse-spark-1.3-contributor` (host
`~/.config/muse/settings.json`). Alpha does not synthesize a catalog or
remap those ids. `GET /muse-code/models` is
`GET https://api.meta.ai/muse-code/models`; Muse-shaped
`POST /responses` is `POST https://api.meta.ai/v1/responses`. Meta
keeps `muse-spark-*` ids. Do not map them onto Z.AI, OpenRouter, TAP,
or any other unrelated provider.

### Credentials

`musela` is Meta OAuth pass-through. Muse already authenticates with
Meta; alpha forwards the inbound `Authorization: Bearer` plus
`x-client-id` and, on model calls, `x-tbh-session-id` /
`x-meta-ai-gateway-session-id` / `traceparent`. Missing or malformed
Bearer fails closed. Do not treat the Meta token as a LiteLLM virtual
key, and do not put Meta OAuth tokens or API keys in Compose,
Langfuse, or `session_history`. Host Muse login stays in
`~/.config/muse/auth.json` and is not an alpha bind-mount.

Enable the facade only on alpha:

```yaml
# docker-compose.alpha.yml environment (alpha only)
AAWM_MUSE_CODE_FACADE_ENABLED: "1"
```

Leave that variable unset on `litellm-dev` and production. When unset,
`GET /muse-code/models` remains FastAPI `{"detail":"Not Found"}`.

### Known limitations

- Alpha-only. `musel` / `muselt` are not this route.
- Muse `--base-url` is origin-only. A base that already ends in `/v1`
  is a different join and is not the `musela` contract.
- Tool execution (shell, filesystem, MCP) stays in the Muse client.
  LiteLLM must not run those tools. Namespace tool bodies are forwarded
  unmodified.
- Approvals are local Muse/MSP, not Meta HTTP.
- Harness v2 Muse TUI source may exist under `scripts/harnessv2/`;
  running it still needs an explicit operator request.
- Body limits and stream timeouts remain alpha-config knobs and must
  not be added to `litellm-dev-config.yaml`. Do not add Muse spark
  aliases to `litellm-alpha-config.yaml`.

### Logs

- Container:
  `docker compose -f docker-compose.alpha.yml logs litellm-alpha`
- Error JSONL: repository `.analysis/alpha-error.jsonl`
  (`LITELLM_AAWM_ERROR_LOG_ENV=alpha`)
- Langfuse: `LITELLM_LANGFUSE_TRACE_ENVIRONMENT=alpha`
- Session-history spool:
  `/app/.analysis/runtime/litellm-alpha/session_history`
- Route family for Muse catalog/call attribution: `muse_code` (no
  credentials in tags)
- Rollup effort is the request `reasoning.effort` token as sent (or
  top-level `reasoning_effort`). Catalog rows stay `:none`. Muse TUI
  default `high` renders `:high`. Muse TUI `max` is `ultra` on the
  wire and renders `:ultra`, not `:none`.

### Rollback / removal

Do not use `docker-compose.dev.yml` or production Compose for this
rollback. Alpha-only:

1. Remove `AAWM_MUSE_CODE_FACADE_ENABLED` (and any later
   `AAWM_MUSE_*` / `LITELLM_MUSE_*` keys) from
   `docker-compose.alpha.yml`.
2. Do not add Muse spark `model_list` rows to undo this route. If any
   were added to `litellm-alpha-config.yaml`, remove them there only.
   Do not edit `litellm-dev-config.yaml` to “undo” Muse.
3. Recreate **only** `litellm-alpha`:

   ```bash
   docker compose -f docker-compose.alpha.yml up -d --force-recreate litellm-alpha
   ```

4. Confirm `curl -sS http://127.0.0.1:4011/muse-code/models` is 404
   `{"detail":"Not Found"}` and that `:4000` / `:4001` were not
   restarted.
5. Source rollback is the reverse of the Muse facade commit (router
   include + `muse_code_gateway` module). Alias YAML under
   `litellm/proxy/aawm_alias_config/` must not have received Muse
   entries; if it did, revert that as a defect.

Host `musela` can remain; it only pins alpha `endpoint_transport` in
`~/.config/aawm-musela/`. Removing the launcher is optional and is not
required to disable the alpha route.
