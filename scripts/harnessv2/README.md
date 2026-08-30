# Harness v2

YAML/JSON-first LiteLLM acceptance harness. Implemented TUIs are Ohmypi
and Codex. Claude is out of scope. Grok and OpenCode are stubs.

Current Harness v2 closeout excludes Anthropic/Claude provider, model,
alias, TUI, test, and acceptance work. Do not select or run it.
Historical Anthropic/Claude mentions are legacy/non-goal.

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

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui codex \
  --test model \
  --dry-run
```

`--instance`, `--container`, and `--target` are the same flag: a Docker
container name, or a YAML alias (`alpha` → `litellm-alpha`). The host
port is **not** a flag. The harness reads `NetworkSettings.Ports` from
`docker inspect` and prefers `127.0.0.1`. Names `aawm-litellm` /
`litellm-dev` and host ports `4000` / `4001` are refused before inspect.

`--dry-run` prints the resolved plan and exits 0 (no TUI, no HTTP, no
docker logs of protected containers). `--overlay PATH` deep-merges extra
YAML/JSON onto the checked-in config. Overlays may add protected
containers/ports but cannot remove the immutable `aawm-litellm`,
`litellm-dev`, `4000`, `4001`.

Lists of aliases, log forbiddens, endpoints, Redis ceilings, and Ohmypi
argv live under `config/`. Python changes only when a new *kind of step*
is invented.

## Kinds (`--test`)

Baseline walk is `platform` → `catalog` → `orchestration`. Independent
`--test model` stays available (`--model` / `--model all`) but is **not**
a baseline full-suite step.

| Kind | TUI | What it does |
|---|---|---|
| `platform` | forbidden | Health, custom endpoints, error JSONL, Redis prefix SCAN, docker logs |
| `catalog` | optional | CFG-023/024 HTTP catalog; Ohmypi picker if `--tui ohmypi`; Codex skips live picker |
| `model` | required | Independent per-alias TUI turn (not baseline). Ohmypi: idle exact PONG or provider 404. Codex: tool-bearing child `date`/`pwd` stdout plus `hv2-codex-child` on `basic`; local `/root/hv2_child_*` or `/root/hv2_codex_child*` chrome is not a pass |
| `orchestration` | required | Parent alias spawns children. Ohmypi default is the thirteen mixed aliases; `--orchestration-children provider_coverage` spawns the provider-pinned aliases. Codex defaults to parent `basic` and requires explicit children |

`--test model` and `--test orchestration` launch a dedicated interactive
tmux session on socket `tmux37`. Ohmypi uses `hv2-ohmypi-<model>-<pid>`
with `--model litellm-alpha-passthrough/<alias>`. Codex uses
`hv2-codex-<model>-<pid>` and `codex --cd … --model <alias>` plus `-c`
identity header overrides. They never use `-p` / `--print` / `codex exec`
and they do not reuse leftover operator panes (`omp-alpha-test` or
`codex`). Codex 0.149 submit waits until the latest `model:` header
leaves `loading` (footer `{alias} default` is not selected), then uses
YAML `submit_keys` (`C-m`) after `submit_delay_seconds` (default `1.0`);
Enter is a composer newline.
Leave dedicated `hv2-*` sessions open after `_step_tui_model` and
`_step_tui_orchestration`. Do not close them at the end of those
kinds. Operator inspects leftovers after a claimed pass:

```text
tmux -L tmux37 ls
tmux -L tmux37 attach -t <hv2-ohmypi-…>
```

Leftover count: platform 0, catalog 0 (HTTP; optional picker does not
leave a dedicated inspect session as a baseline leftover), optional
`--test model` / `--model all` = one dedicated session per compiled
alias (including `provider-*` except `provider-anthropic` and all
`claude-*` aliases, which current closeout must not select or run),
orchestration = 1 parent session.
Baseline walk leftover is the orch parent session, not the full
`--model all` leftover set.

Ohmypi `--model all` expands compiled aliases, including OMP-facing
`auto-review`, Codex-client compatibility `codex-auto-review`, and the
`provider-<id>` aliases except `provider-anthropic` and all `claude-*`
aliases. Those remain catalog/history facts, not current closeout
selection/run targets. Codex `--test model` defaults to `basic` only.
Do not treat Ohmypi
`--model all` as the Codex OC-003 surface. Codex model/orchestration is
tool-bearing (child `date`/`pwd`); it is not Ohmypi `--no-tools` PONG.
`--tui grok` and `--tui opencode` remain stubs. `--tui claude` stays out
of scope and is excluded from current closeout (do not select or run;
historical mentions are legacy/non-goal).

Provider-pinned orchestration is a separate group. It is not the
baseline mixed-alias walk:

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui ohmypi \
  --test orchestration \
  --orchestration-parent sota-openai \
  --orchestration-children provider_coverage \
  --dry-run
```

That plan must stay tools-on. Each child is a `provider-<id>` Ohmypi
`agent=` profile except `provider-anthropic` and all `claude-*` aliases,
which current closeout `provider_coverage` must not select or run.
Credential, quota, tool-contract, and provider errors fail that
provider; they are not converted into a mixed-alias pass. Never
target `aawm-litellm` or `litellm-dev`.

For `--test model` the driver waits until the TUI returns idle, then
passes only on a standalone exact `PONG` reply or an explicit provider
404 needle (`No endpoints found for`, `404 Not Found`, `status code
404`, `Error: 404`). `※ recap:` is only a wait signal; generic
`Error:`, recap-only, and non-idle panes fail. Recap is wait-only; a
standalone exact `PONG` line after a prompt echo newer than the
pre-send pane (not the prompt echo, not a leftover session-dir
`PONG`, and not a restored complete echo+PONG turn already on the
pane before send) completes the model wait even when recap never
paints. Each alias uses a fresh `--session-dir` under
`/tmp/omp-alpha-sessions/hv2-<alias>`. Launch splash (`Welcome
back` / `Recent sessions`) remaining in scrollback is not non-idle.

Ohmypi `task` children inherit tmux env, not parent `--config`. Overlay
row is identity first:

```text
PI_CONFIG_FILES=<session_dir>/hv2-ohmypi-identity.yml:$HOME/.omp/agent/litellm-alpha.yml
```

Identity first so task children inherit `x-aawm-client*` headers; parent
still gets `--config`. Do not revert this to operator-only
`PI_CONFIG_FILES`.

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
`expert` / `basic-other` / `work-other` / `expert-other` / `sota` /
`sota-xai` / `sota-alibaba` / `sota-moonshot` / `sota-zai` /
`auto-review` / `auto-review-other` (no `model=` spawn field; spawn name is
`auto-review`, not `codex-auto-review`). Not orch children:
`sota-deepseek`, `codex-auto-review`. Each child's first
directive is exact `PONG`, then `date`, then a follow-up parallel
`pwd` / `uname -s` / `echo omp-alpha-fanout`. Profiles are staged into
`{cwd}/.omp/agents`. The run fails unless Ohmypi `task` / `hub`
completions exist for all thirteen children (`hub` jobs, `async-result`
`<task-result>`, idle hub peers, or nested child `yield`). Nested
`bash` `date` while the parent is still waiting is not a pass. A spawn
announcement is not a pass. `※ recap:` is wait-complete only;
recap-only is not a pass. Leftover uvicorn ACCESS except `/health*` is
0 on alpha as of 2026-08-22 (including native `GET /v2/model/info`
500). Empty `docker logs --since` windows are not leftover-uvicorn
pass. Post-TUI `docker_logs` polls up to
`checks.logs.rollup.settle_seconds` (default 180s) when the AAWM
route-rollup header is the only miss, including a 0-byte window
after ACCESS replacement. Bind-mount alpha can emit the header after
the pane goes idle (60s rollup flush plus leftover-session grouping).
Leftover uvicorn still fails immediately.

## Closeout checkpoint

Fail-fast continuation for the current closeout: record the last passed
node and the exact failed node. After a fix, rely on existing focused
evidence for the corrected node, then run only the first unverified
successor and later nodes/files. Never rerun a passed prefix or the
full gate unless that evidence is invalidated or the operator
explicitly requests it. At the next failure: stop, preserve the
checkpoint, fix, and continue there.

## Unit tests

```text
./.venv/bin/pytest tests/test_litellm/scripts/test_harnessv2.py -q --tb=short
```
