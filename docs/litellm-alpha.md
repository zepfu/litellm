# LiteLLM Alpha

`litellm-alpha` is the testing-only LiteLLM proxy for candidate work in this
checkout. It listens on port `4011`.

Never route production traffic, persistent clients, scheduled jobs, or normal
development traffic to this service. Do not use alpha results as production
deployment evidence. The container intentionally runs code that may be
incomplete, unreviewed, or broken.

## Runtime Contract

- Container: `litellm-alpha`
- Image: `litellm-alpha:local`
- Local endpoint: `http://127.0.0.1:4011`
- Tailscale endpoint: `http://100.109.19.233:4011`
- Config: `/app/litellm-dev-config.yaml`, bind-mounted from this repository
- Source: `/app`, bind-mounted read-only from this repository
- Environment label: `litellm-alpha`
- Alias-routing namespace: `aawm-routing-alpha-v1`

The image contains the Python dependencies and an editable LiteLLM install.
At runtime, the repository is mounted over `/app`, and `PYTHONPATH=/app`
ensures imports resolve from the live checkout.

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
two expected Codex OAuth account hashes.

```bash
docker compose -f docker-compose.dev.yml build litellm-alpha
docker compose -f docker-compose.dev.yml up -d litellm-alpha
```

Starting alpha may also start the shared development alias-routing Redis
service when it is not already running. Alpha uses its own Redis namespace and
does not share candidate cooldown or affinity keys with `litellm-dev`.

## Verify

```bash
docker compose -f docker-compose.dev.yml ps litellm-alpha
docker compose -f docker-compose.dev.yml logs --tail=100 litellm-alpha
curl --fail http://127.0.0.1:4011/health/liveliness
docker exec litellm-alpha python -c \
  'import litellm; print(litellm.__file__)'
```

The import path must resolve under `/app/litellm`.

When a watched file changes, the logs should show the existing LiteLLM process
stop and a new process start. Wait for `/health/liveliness` before running the
next live test.

## Rebuild

```bash
docker compose -f docker-compose.dev.yml build litellm-alpha
docker compose -f docker-compose.dev.yml up -d --force-recreate litellm-alpha
```

Rebuild only when the image dependency layer or Dockerfile changed. Ordinary
source and watched configuration edits do not require this.

## Stop

Stop or remove only the alpha service. Do not use `docker compose down`, because
the Compose project also owns shared development services.

```bash
docker compose -f docker-compose.dev.yml stop litellm-alpha
docker compose -f docker-compose.dev.yml rm -f litellm-alpha
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
