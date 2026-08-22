# Harness v2

YAML/JSON-first LiteLLM acceptance harness. Ohmypi is the only v1 TUI.
Claude is out of scope. Codex, Grok, and OpenCode are stubs.

**Operator document (as-built runner + intended testing process):**
[`TESTING.md`](TESTING.md). Legacy Claude/Codex harnesses stay in
`TEST_HARNESS.md` and `scripts/local-ci/`.

Do **not** aim this runner at `aawm-litellm` (`:4000`) or `litellm-dev`
(`:4001`). The documented runtime is `litellm-alpha`. Host port comes
from `docker inspect`, not a CLI flag (today that is `127.0.0.1:4011`).

Legacy `scripts/local-ci/` stays frozen. This tree does not import it.

## Usage

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --test platform \
  --write-artifact /tmp/hv2-platform.json
```

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui ohmypi \
  --test model \
  --model work \
  --dry-run
```

`--instance`, `--container`, and `--target` are the same flag: a Docker
container name, or a YAML alias (`alpha` → `litellm-alpha`). The host
port is **not** a flag. The harness reads `NetworkSettings.Ports` from
`docker inspect` and prefers `127.0.0.1`. Names `aawm-litellm` /
`litellm-dev` and host ports `4000` / `4001` are refused before inspect.

`--dry-run` prints the resolved plan and exits 0 (no TUI, no HTTP, no
docker logs of protected containers). `--overlay PATH` deep-merges extra
YAML/JSON onto the checked-in config.

Lists of aliases, log forbiddens, endpoints, Redis ceilings, and Ohmypi
argv live under `config/`. Python changes only when a new *kind of step*
is invented.

## Kinds (`--test`)

| Kind | TUI | What it does |
|---|---|---|
| `platform` | forbidden | Health, custom endpoints, error JSONL, Redis prefix SCAN, docker logs |
| `catalog` | optional | CFG-023/024 HTTP catalog; Ohmypi picker if `--tui ohmypi` |
| `model` | required | Interactive Ohmypi PONG (or clean provider 404) per model/group |
| `orchestration` | required | Parent alias spawns `basic`/`work`/`expert`/`sota` |

`--test model` and `--test orchestration` launch a dedicated interactive
Ohmypi tmux session on socket `tmux37` (`hv2-ohmypi-<model>-<pid>`) with
`--model litellm-alpha-passthrough/<alias>`. They never use `omp -p` /
`--print` and they do not reuse leftover `omp-alpha-test` panes.

Every non-dry-run also appends a durable JSONL log under
`.analysis/harnessv2/` (`run_start`, one `step` line per check with
`pass`/`fail` and bounded failure detail, then `run_end`). The start
line records `git rev-parse HEAD` of the checkout under test. At the
end the harness re-reads HEAD: if the commit or dirty state changed
mid-run it adds a **strong warning** and does **not** invalidate `ok`
for that reason alone.

`Traceback (most recent call last)`, `Exception in ASGI application`,
and leftover native uvicorn access lines (`INFO: … "METHOD /path HTTP/…"`)
are hard stop signals, except health-like paths in
`checks.logs.leftover_uvicorn.allow_paths`. This is not a replaced-path
allowlist: Ohmypi discovery probes such as `GET /model_group/info` and
`GET /openai_passthrough/{,v1/,v2/}model/info` halt if they still print
native uvicorn. Those probes must be suppressed in-process (health
filter for native `/model_group/info` 200, AAWM route replacement for
the `/openai_passthrough` 404s) and must not be added to `allow_paths`.
Startup lines (`Started server process`,
`Application startup complete`, `Uvicorn running on`) are not access
lines and are ignored. The run fails and remaining steps are skipped.
`--test catalog --tui ohmypi` requires the picker selector
`litellm-alpha-passthrough/<alias>` (a bare alias substring is not
enough). `--test orchestration` pastes `agent=basic` / `work` /
`expert` / `sota` (no `model=` spawn field), stages those profiles
into `{cwd}/.omp/agents`, and fails unless Ohmypi `task` / `hub`
completions exist for all four children (`hub` jobs, `async-result`
`<task-result>`, idle hub peers, or nested child `yield`). Nested
`bash` `date` while the parent is still waiting is not a pass. A spawn
announcement is not a pass. `※ recap:` is wait-complete only;
recap-only is not a pass. Live `--test orchestration` on
`litellm-alpha` with parent `sota-openai` now has a green artifact
with hub/`yield` child evidence for `basic`/`work`/`expert`/`sota`
(not recap-only). Leftover
uvicorn ACCESS except `/health*` is 0 on alpha as of 2026-08-22
(including native `GET /v2/model/info` 500). Empty `docker logs
--since` windows are not leftover-uvicorn pass.

## Unit tests

```text
./.venv/bin/pytest tests/test_litellm/scripts/test_harnessv2.py -q --tb=short
```
