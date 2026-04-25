# Production Release Runbook

This runbook is the durable release process for moving the AAWM LiteLLM fork,
overlay wheels, model config, and acceptance harness into the production-style
container on `:4000`.

## Runtime Split

Use the target ports consistently:

- `:4001` is `litellm-dev` and is the development / pre-promotion validation
  runtime.
- `:4000` is `aawm-litellm` and is the production-style runtime.

The Anthropic adapter harness target profile must match the selected runtime:

- `--target dev` validates `http://127.0.0.1:4001/anthropic`, Docker container
  `litellm-dev`, and Langfuse trace environment `dev`.
- `--target prod` validates `http://127.0.0.1:4000/anthropic`, Docker container
  `aawm-litellm`, and Langfuse trace environment `prod`.

Do not validate prod with ad hoc URL overrides unless the override is part of a
one-off incident investigation. Normal release validation should use the named
target profile so port, container, and Langfuse environment checks stay tied
together.

## Preconditions

Before cutting or promoting a release:

- Confirm the LiteLLM working tree only contains intentional release changes.
  Do not stage unrelated local edits.
- Confirm `develop`, `main`, and the remote branches are intentionally
  converged before tagging. The image publisher requires fork image tags to
  point at current `main`.
- Confirm `litellm-dev` on `:4001` is healthy.
- Confirm `aawm-litellm` on `:4000` is the prod target being promoted.
- Confirm `.env` / production env files contain the provider credentials needed
  for the lanes being validated.
- Confirm `PATCHES.md`, `TODO.md`, `COMPLETED.md`, `WHEEL.md`, and
  `TEST_HARNESS.md` reflect the current release state when behavior changes.

## Pre-Promotion Validation On Dev

Validate the change on `:4001` before touching prod.

1. Run focused dev cases for the changed area.

   ```bash
   ./.venv/bin/python -m dotenv run -- \
     ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
       --target dev \
       --cases claude_adapter_gpt54 \
       --write-artifact /tmp/litellm-dev-focused.json
   ```

2. Run the default dev adapter harness after focused cases pass.

   ```bash
   ./.venv/bin/python -m dotenv run -- \
     ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
       --target dev \
       --write-artifact /tmp/litellm-dev-harness.json
   ```

3. Inspect overlapping `litellm-dev` logs. Treat these as release blockers:

   ```bash
   docker logs --tail 500 litellm-dev
   ```

   Block on async task exceptions, ASGI exceptions, `KeyError: choices`, stale
   `Content-Length` / `h11` protocol failures, unhandled passthrough 429/5xx
   tracebacks, missing session-history rows, missing runtime/client identity,
   missing cost, missing reasoning attribution, or missing tool activity for
   cases that should emit it.

4. Before a prod cutover, run a production-style preflight against the exact
   image / installed wheel path on `:4001` when packaging, overlay wheels, model
   config, or callback behavior changed. This is the hardening step that catches
   dev-source versus prod-wheel drift before `:4000` is touched.

## Build And Publish The Fork Image

The base fork image answers: "which LiteLLM fork release are we running?"

1. Bump the fork version in `pyproject.toml` when core fork behavior changed.
2. Confirm `PATCHES.md` records the current patch number and release behavior.
3. Commit the release changes to `main`; keep `develop` converged.
4. Create an annotated fork image tag from current `main`.

   ```bash
   git tag -a v<upstream>-aawm.<n> -m "Release v<upstream>-aawm.<n>"
   git push origin main
   git push origin v<upstream>-aawm.<n>
   ```

5. Verify `.github/workflows/aawm-publish.yml` succeeds. It publishes
   `ghcr.io/zepfu/litellm:<upstream>-aawm.<n>` and updates `latest`.
6. If an artifact autobump moves `main` after the tag, cut a new image tag from
   the new `main` head. Do not force-move an already-published release tag.

## Overlay Artifacts

The overlay artifacts answer: "which AAWM non-core behavior set are we layering
on top?"

Main pushes auto-bump independently shipped artifacts when their inputs change:

- callback wheel: `cb-v*`
- control-plane wheel: `cp-v*`
- harness archive: `h-v*`
- model config archive: `cfg-v*`

Before rebuilding infrastructure, verify the GitHub Releases and assets exist.
Remote tags alone are not enough: the infrastructure Dockerfile selects the
latest non-draft GitHub Release assets, not the latest git tags. If an autobump
workflow creates tags but the release workflows do not publish assets, the
Docker build will silently keep resolving the previous released overlay.

```bash
gh release view cb-v<version> --repo zepfu/litellm
gh release view cfg-v<version> --repo zepfu/litellm
gh release view h-v<version> --repo zepfu/litellm
```

If a tag exists but the release is missing, publish or rerun the artifact
release before building prod. After publishing a missing overlay asset, rebuild
with cache busting so Docker re-resolves the release API instead of reusing a
layer that installed stale artifacts.

Normal infrastructure rebuilds should pin the base fork image but float the
latest versioned callback, control-plane, harness, and model-config artifacts.
Pin overlay artifact versions only for incident response, rollback testing, or
when reproducing a historical build exactly.

## Promote Infrastructure

Promotion happens in `/home/zepfu/projects/aawm-infrastructure`.

1. Update the prod LiteLLM base image pin in the infrastructure files.

   Expected locations:

   - `Dockerfile.litellm`
   - `docker-compose.litellm.yml`

2. Build the prod image.

   ```bash
   docker compose -f docker-compose.litellm.yml build --pull --no-cache litellm
   ```

   Inspect the built image before restarting prod:

   ```bash
   docker run --rm --entrypoint python3 aawm-litellm:latest -c \
     "import importlib.metadata as m; print(m.version('litellm')); print(m.version('aawm-litellm-callbacks')); print(m.version('aawm-litellm-control-plane'))"
   ```

   Direct OpenAI passthrough routes require `OPENAI_API_KEY` in the container
   environment. In AAWM infra, map it from `AAWM_OPENAI_API_KEY`; having only
   `AAWM_OPENAI_API_KEY` is not enough for `/openai_passthrough/*`.

3. Start the prod container.

   ```bash
   docker compose -f docker-compose.litellm.yml up -d litellm
   ```

4. Confirm readiness and version on `:4000`.

   ```bash
   curl -sS http://127.0.0.1:4000/health/readiness
   curl -sS http://127.0.0.1:4000/health
   ```

5. Inspect startup logs.

   ```bash
   docker logs --tail 500 aawm-litellm
   ```

Do not use `:latest` as the prod base image pin. Production infrastructure
should pin an exact fork image such as `ghcr.io/zepfu/litellm:1.82.3-aawm.34`.

## Prod Validation

Run focused prod checks first when a changed provider lane needs explicit
coverage.

```bash
./.venv/bin/python -m dotenv run -- \
  ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
    --target prod \
    --cases claude_adapter_openrouter_ling_26_flash,claude_adapter_peeromega_fanout \
    --write-artifact /tmp/litellm-prod-focused.json
```

Then run the default prod adapter harness.

```bash
./.venv/bin/python -m dotenv run -- \
  ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
    --target prod \
    --write-artifact /tmp/litellm-prod-harness-<version>.json
```

Acceptance requires:

- zero default-suite failures
- zero default-suite warnings unless a documented warning-only case was
  intentionally selected
- Langfuse traces under `environment=prod`
- session tags tied to the initiating parent session
- `public.session_history` rows with provider, model, cost, runtime identity,
  client identity, reasoning attribution, and provider-cache telemetry
- `public.session_history_tool_activity` rows for cases that invoke tools or
  dispatch child agents
- no overlapping prod log blockers

Always inspect overlapping prod logs after the harness:

```bash
docker logs --tail 1000 aawm-litellm
```

For native passthrough changes, run the opt-in native shard in the existing
harness. This includes the real Claude, Codex, and Gemini CLIs and should not
be replaced with skipped or synthetic key checks.

```bash
./.venv/bin/python -m dotenv run -- \
  ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
    --target prod \
    --cases native_anthropic_passthrough_claude,native_openai_passthrough_chat,native_openai_passthrough_responses,native_openai_passthrough_responses_codex,native_gemini_passthrough_generate_content,native_gemini_passthrough_stream_generate_content \
    --write-artifact /tmp/litellm-prod-native.json
```

For `/anthropic` effort/cache changes, run the opt-in shards that are
default-excluded to keep the default suite stable:

```bash
./.venv/bin/python -m dotenv run -- \
  ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
    --target prod \
    --cases claude_adapter_openai_output_config_effort,claude_adapter_openai_prompt_cache_two_pass,claude_adapter_gemini_output_config_effort,claude_adapter_gemini_output_config_minimal_effort,claude_adapter_gemini_output_config_max_effort,claude_adapter_gemini_output_config_minimal_effort_cache,claude_adapter_gemini_output_config_max_effort_cache \
    --write-artifact /tmp/litellm-prod-effort-cache-openai-gemini.json

./.venv/bin/python -m dotenv run -- \
  ./.venv/bin/python scripts/local-ci/run_anthropic_adapter_acceptance.py \
    --target prod \
    --cases claude_adapter_openrouter_output_config_effort_cache,claude_adapter_openrouter_output_config_max_effort,claude_adapter_openrouter_output_config_max_effort_cache,claude_adapter_openrouter_output_config_none_effort,claude_adapter_openrouter_output_config_none_effort_cache,claude_adapter_openrouter_no_effort_cache_control \
    --write-artifact /tmp/litellm-prod-effort-cache-openrouter.json
```

Harness cases should use generated per-run session ids unless a stable session
id is essential to the test. Static session ids can collide with older
`public.session_history` rows and produce false environment mismatches during
prod validation.

## Optional Provider Lanes

These cases remain available but are not part of the default promotion suite
unless explicitly requested:

- OpenRouter GPT-OSS edge models:
  `claude_adapter_gpt_oss_20b`, `claude_adapter_gpt_oss_120b`
- OpenRouter Gemma edge models:
  `claude_adapter_gemma_31b`, `claude_adapter_gemma_26b_a4b`
- NVIDIA opt-in checks:
  `claude_adapter_nvidia_deepseek_v32`, `claude_adapter_nvidia_glm47`,
  `claude_adapter_nvidia_minimax_m27`

When selected, `claude_adapter_gpt_oss_120b` may soft-fail only for the exact
OpenRouter provider-unavailable signature:

- HTTP `503`
- `provider=OpenInference`
- `raw=no healthy upstream`

Other timeouts, command failures, adapter exceptions, traceback patterns, cost
gaps, and session-history gaps remain hard failures.

## Finalization

After prod validation passes:

- Commit and push LiteLLM release/docs/version updates to `main` and `develop`.
- Commit and push the infrastructure image pin update to the infrastructure
  repo's `main` and `develop` branches.
- Update `COMPLETED.md` with the promoted image version, overlay artifact
  versions, harness artifact path, focused cases, default prod harness result,
  and any warning-only opt-in outcomes.
- Keep unresolved follow-up hardening in `TODO.md`.

## Rollback

If prod validation fails after promotion:

1. Revert the infrastructure image pin to the last known-good fork image.
2. Rebuild and restart `aawm-litellm`.
3. Confirm readiness on `:4000`.
4. Run a focused smoke case and inspect prod logs.
5. Record the failed image, failing cases, logs, and rollback result in
   `TODO.md` / `COMPLETED.md`.

Do not force-move published release tags for rollback. Cut a new fixed release
or repin infrastructure to a previous known-good image.
