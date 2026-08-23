# Harness v2 — as built and testing process

Date: 2026-08-23

This is the operator document for the YAML/JSON-first LiteLLM acceptance
harness in `scripts/harnessv2/`. It describes what is built, what it is
allowed to touch, how a live run is supposed to proceed, and what still
blocks a full wrap-up.

Quick CLI cheat sheet: `scripts/harnessv2/README.md`.
Policy split vs the legacy Claude/Codex harnesses: `TEST_HARNESS.md`.
Implementation plan / traceback ids: `.analysis/202608/harnessv2-tui-agnostic-plan.md`.

This tree does **not** replace `scripts/local-ci/`. Do not import it. Do
not run `scripts/local-ci/` as part of a v2 run.

---

## 1. What this harness is

Harness v2 is an Ohmypi-only acceptance runner aimed at the live-repo
bind-mount instance `litellm-alpha` (host port from `docker inspect`;
today `127.0.0.1:4011`).

It exists because the legacy products in `scripts/local-ci/` are not
TUI-agnostic:

| Product | Entry | Clients | Default target |
|---|---|---|---|
| Baseline local acceptance | `run_acceptance.py` / `run_acceptance.sh` | Codex (`codex exec -p …`) and Claude (`claude -p …`) | `:4000` / `aawm-litellm` |
| Anthropic-route adapter suite | `run_anthropic_adapter_acceptance.py` | Real Claude CLI (plus Codex exec / opt-in Grok) | `:4001` / `litellm-dev` |

v2 is a **new** tree. YAML/JSON is the source of truth. Python changes
only when a new *kind of step* is invented. Adding a model, prompt,
forbidden log string, HTTP probe, or Ohmypi argv token is a YAML edit.

v1 TUI is Ohmypi (`omp` 17.3.8 via `ompla`). Claude is out of scope, not
a stub. Codex, Grok, and OpenCode are stubs (`enabled: false`).

Entry point:

```text
python scripts/harnessv2/run.py --instance litellm-alpha --test <kind>
```

`--instance`, `--container`, and `--target` are the same flag: a Docker
container name, or a YAML alias (`alpha` → `litellm-alpha`). The host
port is **not** a flag.

---

## 2. Runtime firewall

Every Docker and HTTP helper goes through `hv2/docker_guard.py`. There
is no `HV2_ALLOW_PROD` escape hatch.

**Never:**

- `aawm-litellm` (`:4000`)
- `litellm-dev` (`:4001`)
- `docker compose down`
- mutate prod/dev configs to make an alpha test pass
- send Anthropic/Claude upstream models through Codex / ChatGPT OAuth
  (TOS boundary; fail closed)

**Allowed:**

- `litellm-alpha` (bind-mount `.:/app:ro`, watchfiles, Redis namespace
  `aawm-routing-alpha-v1`)
- if alpha needs a process recycle: `docker restart litellm-alpha`
  (never compose down)

`targets.yaml` lists `dev` / `prod` aliases so operators see the names.
Those aliases are `enabled: false` and the protected-container list
refuses them before inspect.

Host port resolution: `docker inspect` → `NetworkSettings.Ports`, prefer
`127.0.0.1`. Names `aawm-litellm` / `litellm-dev` and host ports `4000` /
`4001` are refused before inspect.

Redis: read-only `INFO memory` + prefix SCAN of
`aawm-routing-alpha-v1`. The harness never `FLUSH*` and never writes
keys. Ceilings live in `targets.yaml`.

---

## 3. Tree (as built)

```text
scripts/harnessv2/
  run.py                  # CLI entry; thin interpreter over YAML
  README.md               # short usage
  TESTING.md              # this document
  config/
    harness.yaml          # includes + timeouts + artifact schema
    targets.yaml          # instance firewall, Redis, inspect env keys
    tuis.yaml             # Ohmypi driver contract; stubs; Claude out
    models.yaml           # compiled aliases, groups, skip prefixes
    kinds.yaml            # platform / catalog / model / orchestration
    checks.yaml           # health, HTTP suite, leftover uvicorn, JSONL, Ohmypi rollup identity
    prompts.yaml
    prompts/pong.txt
    prompts/orchestration.txt
  hv2/
    cli.py                # argparse only; no docker/HTTP
    plan.py               # fail-closed RunPlan before any inspect
    instance.py           # container name → inspect → base_url
    docker_guard.py       # protected names/ports
    kinds/runner.py       # walk YAML steps; halt on logging regression
    drivers/ohmypi.py     # interactive tmux; never -p/--print
    drivers/stub.py
    checks/               # health, HTTP, logs, redis, error JSONL
    artifact.py           # JSON artifact + durable JSONL + SHA stamp
    pane.py               # pane needles; ignore needles in sent prompt
    envscrub.py           # child env allow/deny
  fixtures/logs/          # leftover uvicorn / ASGI / clean rollup / Ohmypi identity
tests/test_litellm/scripts/test_harnessv2.py
```

Durable run logs: `.analysis/harnessv2/*.jsonl`.

---

## 4. Config is the source of truth

Root: `config/harness.yaml` (`schema_version: 1`). It includes the YAML
files above. `--overlay PATH` deep-merges extra YAML/JSON onto that
tree. Overlays may add protected containers/ports but cannot remove the
immutable `aawm-litellm`, `litellm-dev`, `4000`, `4001`.

Timeouts (defaults):

| Key | Default | Used for |
|---|---|---|
| `docker_seconds` | 30 | `docker inspect` / `docker logs` |
| `http_seconds` | 15 | health and HTTP suite |
| `tui_seconds` | 420 | Ohmypi reply wait |

Identity: `AAWM_HARNESS_USER_ID` (default `harnessv2`). Injected as
`x-litellm-end-user-id` / `x-aawm-client: harnessv2` on harness HTTP.
Ohmypi child env must not inherit Claude Code / Anthropic / Langfuse /
DB / `LITELLM_MASTER_KEY` secrets (`checks.yaml` `child_env`).

### Compiled aliases (`config/models.yaml`)

`--model all` expands `compiled_aliases` only, including both
OMP-facing `auto-review` and Codex-client compatibility
`codex-auto-review`:

- `basic`, `work`, `work-other`, `expert`, `sota`
- `sota-openai`, `sota-xai`, `sota-alibaba`, `sota-moonshot`,
  `sota-deepseek`, `sota-zai`
- `auto-review`, `codex-auto-review`

Skip prefixes: `aawm-`, `claude-`. Absent catalog ids
(`aawm-sota-zai`, …) are recorded so a picker must not treat them as
present.

Groups:

- `all` → compiled aliases
- `all-sota` → the six `sota-*` parents
- `orchestration_children` (nine) → `basic`, `work`, `expert`, `sota`,
  `sota-xai`, `sota-alibaba`, `sota-moonshot`, `sota-zai`, `auto-review`.
  Spawn name is `auto-review`, not `codex-auto-review`. Not orchestration
  children: `work-other`, `sota-deepseek`, `codex-auto-review`.
- `catalog_picker` → `work`, `sota-zai`

Ohmypi session `--model` is `litellm-alpha-passthrough/<alias>`, not a
bare alias and not `litellm-alpha/<alias>` (that lane is completions).

---

## 5. Kinds (`--test`) and intended process

Baseline walk is `platform` → `catalog` → `orchestration`. Do **not**
skip the logging gate. Independent `--test model` stays available
(`--model` / `--model all`) but is **not** a baseline full-suite step.

| Kind | TUI | What it proves |
|---|---|---|
| `platform` | forbidden | Health, custom HTTP, error JSONL, Redis prefix SCAN, docker logs |
| `catalog` | optional | CFG-023/024 HTTP catalog; Ohmypi picker if `--tui ohmypi` |
| `model` | required | Independent per-alias Ohmypi turn (not baseline). Waits for idle; standalone exact PONG or explicit provider 404 |
| `orchestration` | required | Parent alias spawns the nine orchestration children through Ohmypi `task` |

`--dry-run` prints the resolved plan and exits 0. No TUI, no HTTP, no
docker logs of protected containers. Use it to confirm instance/kind/
model expansion before a live run.

### 5.1 `platform` (no TUI)

Steps: `health` → `http_suite` → `error_jsonl` → `redis_scan` →
`docker_logs`.

Health: `GET /health/liveliness` expect 200.

HTTP suite (`checks.yaml`):

- `GET /openai_passthrough/v1/models` and `/openai_passthrough/models`
  → 200, compiled alias catalog (CFG-023)
- `GET /internal/aawm/session-transfer-status` without identity →
  400/401/403/404, never 500
- `GET /kimi/v1/models` → 200 or 401
- `GET /grok/v1` and `/grok/v1/models` → 200/401/403(/404)
- `POST /v1/chat/completions` `model=work` → miss is pass (400/401/404/422),
  200/500 is fail

Redis: SCAN prefix `aawm-routing-alpha-v1` only. Warn/fail ceilings in
`targets.yaml`. Never flush.

Docker logs: see §7. Rollup is **not** required on platform.

### 5.2 `catalog`

Steps: `health` → `catalog_http` → optional `tui_catalog` →
`docker_logs` → `error_jsonl`.

HTTP catalog is the CFG-023 GET. Ohmypi picker (`omp models --json` /
`omp models find`) is optional and only runs when `--tui ohmypi`. A
find hit must contain the full selector (`litellm-alpha-passthrough/work`);
a bare alias substring such as `work` on another provider is not enough.

Catalog listing does not generate route-rollup traffic. Leftover native
uvicorn ACCESS except `/health*` is still a halt.

### 5.3 `model` (interactive Ohmypi)

Steps: `health` → `tui_model` → `docker_logs` → `error_jsonl`.

For each selected alias the driver:

1. Creates a **dedicated** tmux session on socket `tmux37`, name
   `hv2-ohmypi-<model>-<pid>`. Never reuse leftover `omp-alpha-test`.
2. Launches `omp --session-dir … --model litellm-alpha-passthrough/<alias>`.
   `--no-tools` for model kind.
3. Refuses argv containing `-p` / `--print`.
4. Waits for ready needles (`π`), then pastes `prompts/pong.txt`
   (`Reply with exactly the word PONG.`).
5. Waits until the pane returns idle. After idle, pass requires a
   standalone exact `PONG` reply **or** an explicit provider 404
   needle (`No endpoints found for`, `404 Not Found`, `status code
   404`, `Error: 404`). `※ recap:` is only a wait signal; generic
   `Error:`, recap-only, and non-idle panes fail. Recap is wait-only;
   a standalone exact `PONG` line **after the latest prompt echo**
   (not the prompt echo itself) completes the model wait even when
   recap never paints. A leftover session-dir `PONG` restored by
   `capture-pane -S -200` before that echo is not this alias's live
   reply. A restored complete echo+PONG turn already on the pane
   before send is also not this alias's live reply: require a new
   prompt echo after send, then `PONG` after that echo. Each alias
   uses a fresh `--session-dir` under
   `/tmp/omp-alpha-sessions/hv2-<alias>` so Ohmypi cannot restore the
   prior PONG conversation. Launch splash (`Welcome back` /
   `Recent sessions`) remaining in scrollback is not non-idle. Needles that are substrings of the
   **sent prompt** are ignored (H-6), except a standalone pane line
   equal to the needle (for `PONG`, that line must follow the latest
   prompt echo).
6. Leaves the dedicated `hv2-ohmypi-<alias>-<pid>` session open after
   `_step_tui_model`. Do not close it before the next alias or at the
   end of the kind. Operator inspects leftovers after a claimed pass:

   ```text
   tmux -L tmux37 ls
   tmux -L tmux37 attach -t <hv2-ohmypi-…>
   ```

   Optional `--test model` / `--model all` leftover count is one
   dedicated session per alias (12 if all compiled aliases). This kind
   is not a baseline full-suite step; baseline leftover is the orch
   parent session, not 12.

`--model` is repeatable / comma-separated. Default is group `all`.
`--test model` stays available independently; it is not part of the
baseline `platform` → `catalog` → `orchestration` walk.

Post-TUI `docker_logs` **requires** an AAWM route-rollup header
(`YYYYMMDD HH:MM:SS … /path`) plus the Ohmypi identity gate in §7.
A recap without a real Responses POST + rollup is not pass evidence.

### 5.4 `orchestration` (interactive Ohmypi)

Steps: `health` → `tui_orchestration` → `docker_logs` → `error_jsonl`.

Post-TUI `docker_logs` uses the same `require_rollup` + Ohmypi identity
gate as model kind (§7). `child_evidence.ok` is not docker_logs pass.

Default parent: `sota-openai` (group `all-sota`). Default children are
the nine `orchestration_children`: `basic`, `work`, `expert`, `sota`,
`sota-xai`, `sota-alibaba`, `sota-moonshot`, `sota-zai`, `auto-review`.
Spawn name is `auto-review`, not `codex-auto-review`. Not orch children:
`work-other`, `sota-deepseek`, `codex-auto-review`.

Launches the parent with tools enabled. Pastes
`prompts/orchestration.txt` with `{parent}` substituted. Each child's
**first** directive is exact `PONG`, then `date`, then a follow-up
parallel `pwd` / `uname -s` / `echo omp-alpha-fanout`. Wait needle is
`※ recap:` — **not** `omp-alpha-fanout`, which is inside the prompt and
would false-complete (H-6). Recap is wait-only, never pass evidence.

Spawn contract (H-7, **closed in code**): Ohmypi `task` has no
`model=` field. The parent must spawn `agent=basic`, `agent=work`,
`agent=expert`, `agent=sota`, `agent=sota-xai`, `agent=sota-alibaba`,
`agent=sota-moonshot`, `agent=sota-zai`, and `agent=auto-review`. Those
names are harness profiles staged from `config/ohmypi-agents/` into
`{cwd}/.omp/agents` (`/tmp/omp-alpha-workspace/.omp/agents` by
default). They are **not** the built-in Ohmypi names (`scout`,
`designer`, `reviewer`, …) and **not** LiteLLM catalog ids. Session
`--model` stays `litellm-alpha-passthrough/<parent>`.

`※ recap:` is wait-complete only. Ohmypi 17.4 often finishes the parent
turn with the nine-child PONG / date list and idle `hub` peers, without
a recap glyph. Missing recap is **not** a fail when `child_evidence.ok`
is true.

The runner then calls `child_spawn_evidence()` against Ohmypi session
JSONL, including nested `session_dir/<parent-id>/*.jsonl` child
transcripts. Pass requires successful child completions for all nine
orchestration children. Ohmypi 17.4 delivers those as:

- `hub` job rows (`details.jobs[].resolvedModel` /
  `<task-result agent="…" status="completed">`)
- `customType=async-result` notices (a child can finish this way and
  then drop out of later `hub wait` snapshots)
- idle hub peers (`SotaDate [sota · sub · idle]`)
- nested child `session_init` / `model_change` plus a successful
  `yield` (nested `bash` `date` while the parent is still waiting is
  not enough)

A `Spawned N background agents using basic, work, expert, sota, …` line
is spawn intent, not a completed child result. Recap-only, `Unknown
agent`, failed preflight, or empty hub/task completions is fail. The
wait loop does not stop on recap while any of the nine children is still
missing. Leave the dedicated tmux session open after
`_step_tui_orchestration`; do not close it at the end of the kind.
SIGHUP during `hub wait` is not a pass. Operator inspects leftovers
after a claimed pass:

```text
tmux -L tmux37 ls
tmux -L tmux37 attach -t <hv2-ohmypi-…>
```

Baseline leftover count: platform 0, catalog 0 (HTTP; optional picker
does not leave a dedicated inspect session as a baseline leftover),
orchestration = 1 parent session. Baseline walk leftover is that orch
parent session, not 4–5 and not 12.

Stage child profiles into the Ohmypi **workspace** `.omp/agents`
(`{cwd}` expanded by the driver, default `/tmp/omp-alpha-workspace/.omp/agents`).
A literal `{cwd}/.omp/agents` directory in the checkout is a staging bug:
Ohmypi will not see `sota` and preflight will report `Unknown agent "sota"`.

YAML `evidence_order` is recorded for later DB/route proof:

1. `session_history_tool_activity`
2. `aawm_child_routes`
3. `tui_transcript`

`session_history` polling is currently `enabled: false` in
`checks.yaml`. Do not claim DB child evidence until that is wired.
The live gate is Ohmypi `task` results, not Langfuse / Prisma.

---

## 6. Ohmypi driver contract

From `config/tuis.yaml`:

| Item | Value |
|---|---|
| Binary | `omp` (wrapper `ompla`) |
| Min version | 17.3.8 |
| Overlay | Identity first, then operator: `PI_CONFIG_FILES=<session_dir>/hv2-ohmypi-identity.yml:$HOME/.omp/agent/litellm-alpha.yml`. Ohmypi `task` children inherit tmux env, not parent `--config`. Identity first so those children inherit `x-aawm-client*` headers; parent still gets `--config`. Do not revert this to operator-only `PI_CONFIG_FILES`. That overlay is what stamps rollup as `litellm#Ohmypi[<version>]@<host>`. |
| CWD | `/tmp/omp-alpha-workspace` |
| Session dir | `/tmp/omp-alpha-sessions/hv2-<alias>` (identity overlay stays in `/tmp/omp-alpha-sessions`) |
| tmux socket | `tmux37` |
| Alias lane | `litellm-alpha-passthrough` |
| Completions lane | `litellm-alpha` |
| Forbid | `-p`, `--print` |
| Model tools | off |
| Orchestration tools | on |
| Wait needle | `※ recap:` (wait signal only, never pass evidence) |
| Model pass needles | standalone exact `PONG` after latest prompt echo; `No endpoints found for` / `404 Not Found` / `status code 404` / `Error: 404` |
| Orchestration pass | `child_evidence.ok` (hub `<task-result>` / nested `yield`) |

Child env allow-prefixes include `PI_` / `OMP_` / `CODEX_` / `OPENAI_` /
`XAI_`. Deny Langfuse, DB, master key, Anthropic/Claude inheritance.

Do not send-keys into an existing operator `omp-alpha-test` pane and
call that a harness pass.

**Smoother path (dedicated session):** every `--test model` /
`--test orchestration` row creates `hv2-ohmypi-<alias>-<pid>` on
socket `tmux37`, with identity-first
`PI_CONFIG_FILES=<session_dir>/hv2-ohmypi-identity.yml:$HOME/.omp/agent/litellm-alpha.yml`
and alias lane `litellm-alpha-passthrough/<alias>`. Never `omp -p` /
`--print`. Never reuse leftover `omp-alpha-test`. Leave dedicated
`hv2-ohmypi-*` sessions open after `_step_tui_model` and
`_step_tui_orchestration`. Do not close them at the end of those kinds.
Operator inspects leftovers after a claimed pass (`tmux -L tmux37 ls`,
then `attach -t <hv2-ohmypi-…>`). Baseline leftover is the orch parent
session; optional `--model all` would leave 12.

Long alias ids can render as `AAWM alias / model <id>` instead of the
full selector or `AAWM alias <id>`. That truncated chrome is selected
evidence. `ensure_session` waits for selected needles again after MCP
because Ohmypi can paint the alias chrome after connect. Do not invent
live greens from truncated chrome alone.

Unclassified alias-loop provider errors (OpenRouter 422, missing Z.AI
Coding Plan key) must become `ProxyException` / classified cooldown —
not uvicorn `Exception in ASGI application` plus a full traceback. Do
not add those strings to leftover-uvicorn `allow_paths`.

---

## 7. Halt protocol (logging regression)

After every kind, `docker_logs` scans alpha logs since run start.

**Hard stop** (run `ok: false`, remaining steps skipped as
`halted_on_logging_regression`):

- `Traceback (most recent call last)` that is not an expected signature
- `Exception in ASGI application`
- leftover native uvicorn ACCESS:
  `INFO: … "GET|POST|PUT|PATCH|DELETE|HEAD /path HTTP/…"`

Leftover uvicorn is **inverted**. `allow_paths` is the only set of
paths that may still print native uvicorn. Today that is `/health*`
only:

- `/health`
- `/health/`
- `/health/liveliness`
- `/health/readiness`
- `/health/services`

Everything else is a halt, including Ohmypi discovery probes:

- native `GET /model_group/info`, `/model/info`, `/v1/model/info`,
  `/v2/model/info`
- passthrough `GET /openai_passthrough/{,v1/,v2/}model/info` and
  `/openai_passthrough/model_group/info`
- catalog `GET /openai_passthrough/v1/models` and `/models`
- alias `POST /openai_passthrough/v1/responses`

`replaced_route_paths` in YAML is documentation only. Scanning does
**not** use it as an allowlist. Do **not** add discovery probes to
`allow_paths` to make a run green.

Startup lines (`Started server process`, `Application startup complete`,
`Uvicorn running on`) are not ACCESS lines and are ignored.

Ohmypi TUI kinds (`model`, `orchestration`) set `require_rollup` and
`tui=ohmypi`. Pass then also requires at least one rollup header
matching `litellm#Ohmypi[<version>]@<host>` (example:
`20260823 17:36:58 litellm#Ohmypi[17.4.2]@thoth /openai_passthrough/v1/responses`).
That stamp comes from the identity overlay in §6. Platform does not
require rollup.

Unlabeled Ohmypi still fails even when a labeled Ohmypi row is in the
same window. Concurrent labeled Ohmypi does not excuse these:

- `Oh@host` (`Oh@thoth`)
- `Bun[...]@host` (`Bun[1.3.14]@thoth`)
- `litellm@host /v1/chat/completions` (`litellm@thoth` — this repo
  missing `#Ohmypi[ver]`)

Concurrent other-workspace headers on the shared alpha instance are
**not** unlabeled Ohmypi. Example:
`aawm-infrastructure@thoth /openai_passthrough/responses`. When Ohmypi
identity is present, those known-repo `@host` rows are ignored. The
allowlist is `checks.yaml` `logs.rollup.concurrent_workspace_repos`
and must **not** include `litellm`.

Concurrent Codex-client `litellm@thoth /openai_passthrough/responses`
followed by `- codex-auto-review:low` is also **not** unlabeled Ohmypi.
Marker list: `checks.yaml` `logs.rollup.concurrent_codex_client_markers`
(includes `codex-auto-review`). OMP spawn name remains `auto-review`;
`codex-auto-review` is Codex-client compatibility. Bare `litellm@thoth`
without a following `codex-auto-review` model line still fails.

If the window has only concurrent rows (other-workspace or Codex-client)
and no `litellm#Ohmypi[...]@...`, still fail.

Identity miss fails `docker_logs` (`ok: false`). It is not
leftover-uvicorn halt (`halted_on_logging_regression`).

Fixtures: `ohmypi_identity_ok.txt`, `ohmypi_identity_miss.txt`,
`ohmypi_identity_ok_with_concurrent_aawm_infrastructure.txt`,
`ohmypi_identity_ok_with_concurrent_codex_auto_review.txt`.

Expected traceback signature (warning, not fail): `model=work` with
`There are no healthy deployments` on the generic `/v1/chat/completions`
miss probe.

Soft-fail signatures (warning, not `ok` flip) live under
`checks.yaml` `soft_fail` (example: OpenRouter `owl-alpha` 404
`No endpoints found`).

On halt:

1. Stop remaining `--test model` / `--test orchestration`.
2. Source-fix in the shared checkout (`litellm-alpha` bind-mounts it).
3. Let watchfiles reload, or `docker restart litellm-alpha`.
4. Re-prove leftover uvicorn ACCESS = 0 except `/health*` on a
   harness-owned window (HTTP probes are enough for leftover ACCESS;
   do not resume Ohmypi TUI until that count is 0).
5. Resume the same harness kind.

---

## 8. Artifacts

Every non-dry-run:

1. Optional JSON artifact (`--write-artifact PATH`), schema
   `harnessv2.artifact.v1`. Secrets in `artifact.redact_headers` are
   replaced with `<redacted>`.
2. Durable JSONL under `.analysis/harnessv2/`:

```text
{stamp}-{kind}-{container}-{git12}.jsonl
```

Events:

- `run_start` — kind, tui, instance `base_url` / host port, git stamp
  (`commit`, `branch`, `dirty`)
- `step` — `pass` / `fail`, bounded detail
- `run_end` — `ok`, `halted`, warnings, git stamp

If HEAD or dirty state changed mid-run the harness adds a **strong
warning** and does **not** flip `ok` for that reason alone.

`--dry-run` may still write a JSON artifact of the plan; it does not
append durable JSONL of a live run.

---

## 9. How to run

From the repo root, use the project venv if you need pytest; `run.py`
is plain Python 3.

### Unit tests (no Docker, no alpha)

```text
./.venv/bin/pytest tests/test_litellm/scripts/test_harnessv2.py -q --tb=short
```

Covers argparse, YAML load, docker_guard refuse of `:4000`/`:4001`,
leftover-uvicorn invert, Ohmypi forbid `-p`, Ohmypi rollup identity
(`require_rollup` + `tui=ohmypi`, including concurrent
`aawm-infrastructure@thoth` and concurrent Codex-client
`litellm@thoth` + `codex-auto-review`), dry-run plans, H-6 prompt
substring needles.

### Dry-run a kind

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --test platform \
  --dry-run
```

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui ohmypi \
  --test model \
  --model work \
  --dry-run
```

### Live platform

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --test platform \
  --write-artifact /tmp/hv2-platform.json
```

### Live catalog (HTTP only)

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --test catalog \
  --write-artifact /tmp/hv2-catalog.json
```

Add `--tui ohmypi` only when you also want the picker.

### Live model (interactive Ohmypi)

Independent of the baseline walk (`platform` → `catalog` →
`orchestration`). Leftover uvicorn ACCESS except `/health*` is **0**
on a harness-owned alpha window as of 2026-08-22 (see §11). Remaining
compiled aliases still need live Ohmypi `--test model`. Leave dedicated
`hv2-ohmypi-*` sessions open after `_step_tui_model` for inspect
(`tmux -L tmux37 ls`). `--model all` leftover is one session per
compiled alias (12), not a baseline leftover.

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui ohmypi \
  --test model \
  --model work \
  --write-artifact /tmp/hv2-model-work.json
```

Repeat `--model` / comma-separated ids, or omit it to expand `all`
(includes `auto-review` and `codex-auto-review`).

### Live orchestration

Same leftover-uvicorn gate **and** the Ohmypi identity rollup gate
(§7). Recap is wait-only, never pass evidence: `tui_orchestration`
fails unless `child_evidence.ok` is true for the nine orchestration
children (`basic`, `work`, `expert`, `sota`, `sota-xai`,
`sota-alibaba`, `sota-moonshot`, `sota-zai`, `auto-review`). Spawn
name is `auto-review`, not `codex-auto-review`. Historical recap-only
`ok: true` artifacts are stale relative to the current gate. Leave
the dedicated parent session open after `_step_tui_orchestration`.
Baseline leftover is that one orch parent session. `child_evidence.ok`
without green post-TUI `docker_logs` is not a full orch pass.

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui ohmypi \
  --test orchestration \
  --orchestration-parent sota-openai \
  --write-artifact /tmp/hv2-orch-sota-openai.json
```

### Overlay

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --test platform \
  --overlay /tmp/hv2-overlay.yaml
```

---

## 10. Intended testing process (operator loop)

1. Confirm `litellm-alpha` is up. Do not touch `:4000` / `:4001`.
2. Run unit tests in `tests/test_litellm/scripts/test_harnessv2.py`.
3. Dry-run the kind you are about to execute.
4. `--test platform`. If leftover uvicorn / Traceback / ASGI: **halt**.
   Empty `docker logs --since` is **not** leftover-uvicorn pass: send
   the real HTTP probes, then scan that window.
5. `--test catalog`. HTTP CFG-023 is required for the baseline walk.
   Add `--tui ohmypi` only when you also want the picker
   (`litellm-alpha-passthrough/work` and the other catalog sample).
   HTTP-only catalog leftover is 0; optional picker does not leave a
   dedicated inspect session as a baseline leftover. Same halt rule.
6. HTTP leftover-uvicorn proof for Ohmypi discovery probes (no TUI
   required for this gate). This gate **passed** on 2026-08-22
   (`CURSOR=2026-08-22T02:44:06Z`; leftover ACCESS except `/health*`
   = 0, including native `/v2/model/info` 500):
   - `GET /health/liveliness` 200, leftover ACCESS allowed
   - `GET /openai_passthrough/v1/models` 200, leftover ACCESS = 0
   - `GET /openai_passthrough/{,v1/,v2/}model/info` 404, leftover ACCESS = 0
   - native `GET /model_group/info`, `/model/info`, `/v1/model/info`,
     `/v2/model/info`: leftover ACCESS = 0 even if the HTTP status is
     not 200
7. Optional independent `--test model` / `--model all` is **not** a
   baseline full-suite step. If you run it, leave dedicated
   `hv2-ohmypi-*` sessions open (12 if all compiled aliases). On
   traceback / leftover uvicorn: halt, source-fix on the shared
   checkout, `docker restart litellm-alpha` only if watchfiles did not
   reload, resume the **same** row.
8. Baseline next: `--test orchestration --orchestration-parent
   sota-openai` after leftover uvicorn is gone. Gate spawn on
   `child_evidence.ok` for the nine orchestration children (`basic`,
   `work`, `expert`, `sota`, `sota-xai`, `sota-alibaba`,
   `sota-moonshot`, `sota-zai`, `auto-review`; spawn name
   `auto-review`, not `codex-auto-review`). Recap is wait-only, never
   pass evidence. Post-TUI `docker_logs` still needs
   `litellm#Ohmypi[<version>]@<host>` (concurrent
   `aawm-infrastructure@thoth` and concurrent Codex-client
   `litellm@thoth /openai_passthrough/responses` + `codex-auto-review`
   on the shared alpha instance are not unlabeled Ohmypi; see §7).
   Leave the dedicated parent session open; inspect with
   `tmux -L tmux37 ls` / `attach -t <hv2-ohmypi-…>`. Baseline leftover
   is that one orch parent session, not 4–5 and not 12.
9. Keep `TEST_HARNESS.md` / this document / the plan file in sync when
   halt signatures or spawn contract change.

Alpha is testing-only. Uncommitted halt-fixes in the shared checkout
are visible to the bind-mount immediately after watchfiles reload.

---

## 11. Current status (2026-08-23)

Proven on `litellm-alpha` (do not re-claim without a new artifact):

| Claim | Status |
|---|---|
| `--test platform` | passed (`/tmp/hv2-platform.json`, `ok: true`) |
| CFG-023 catalog GET leftover uvicorn | 0 leftover ACCESS |
| T-5 leftover uvicorn (discovery probes) | closed: post-cursor leftover ACCESS except `/health*` is **0**, including native `GET /v2/model/info` **500**. Worker reloaded via watchfiles after `_logging.py` mtime `2026-08-22T02:36:57Z`. Cursor `2026-08-22T02:44:06Z`. Ohmypi TUI was not used for this proof. |
| `--test model` `work` / `basic` / `expert` / `sota` / `work-other` | passed: real Responses POST + rollup, `halted: false` |
| `--test orchestration --orchestration-parent sota-openai` | Live 2026-08-23 nine-child orch on `cceab88cd3`: TUI `child_evidence.ok` for all nine children; leftover session `hv2-ohmypi-sota-openai-3839403` left open. `docker_logs` failed on concurrent `aawm-infrastructure@thoth` until the §7 concurrent-workspace filter. Live 2026-08-23 orch retry3 on `e397f7b12e`: `docker_logs` failed on concurrent `litellm@thoth /openai_passthrough/responses` + `codex-auto-review` until the §7 concurrent Codex-client filter. That retry3 TUI also missed `basic` / `work` / `expert` / `sota` (Ohmypi no-yield / null yield) — a separate spawn flake, not this identity gate. Do **not** treat those live artifacts as a full orch pass until TUI nine-child `child_evidence.ok` and `docker_logs` are both green. Four-child artifact (`/tmp/grok-goal-4ce5b5ad827f/implementer/hv2-orch.json`) is historical. Recap-only `ok: true` and the 2026-08-22T06:17 premature-close (`hv2-orch-premature-close.json`, nested `date` without `yield`) are not spawn proof. Current TUI gate is `child_evidence.ok` for the nine orchestration children; recap is wait-only. |
| T-1 unhashable `type` list / ASGI | closed |
| T-4 `UnicodeDecodeError` truncated UTF-8 peek | closed (incremental decoder `errors="ignore"`) |
| H-6 wait needle | closed (`※ recap:`; ignore needles contained in the sent prompt) |
| H-4 leftover-uvicorn invert | in tree (any ACCESS except `/health*` is halt) |
| H-7 Ohmypi spawn | **closed in code**: `orchestration.txt` uses `agent=`; profiles staged into `{cwd}/.omp/agents`; recap-only / `Unknown agent` fail |

**Not closed:**

| Gap | Notes |
|---|---|
| Remaining `--test model` | optional, not baseline. Compiled aliases through `sota-zai` completed on a halted `all` row; `codex-auto-review` resumed after truncated-chrome + ASGI wrap (`hv2-model.json`). `--model all` now also expands OMP-facing `auto-review`. Sequential `--model` rows that together cover `all` are valid. Do not treat the halted `all` JSONL as a full pass. |
| Nine-child orch `docker_logs` | Live 2026-08-23 on `cceab88cd3` had TUI `child_evidence.ok` for all nine children and leftover `hv2-ohmypi-sota-openai-3839403`. `docker_logs` failed on concurrent `aawm-infrastructure@thoth` until the §7 workspace filter. Live 2026-08-23 retry3 on `e397f7b12e` failed `docker_logs` on concurrent `litellm@thoth /openai_passthrough/responses` + `codex-auto-review` until the §7 Codex-client filter. Do **not** treat those artifacts as a full orch pass until TUI and `docker_logs` are both green. Identity miss is a `docker_logs` fail, not leftover-uvicorn halt. |
| Nine-child orch TUI spawn | Live 2026-08-23 retry3 on `e397f7b12e` missed `basic` / `work` / `expert` / `sota` (Ohmypi no-yield / null yield). Separate from the §7 identity gate. |
| Native `/v2/model/info` HTTP 500 body | `DB not connected` when `prisma_client is None`. Separate from leftover ACCESS. T-5 logging-halt does **not** require this body to become 200. |
| Anthropic/Claude TUI | deferred (account canceled); still out of scope for v2 |

---

## 12. Related files

| File | Role |
|---|---|
| `scripts/harnessv2/README.md` | Short CLI usage |
| `scripts/harnessv2/TESTING.md` | This document |
| `TEST_HARNESS.md` | Legacy Claude/Codex harnesses + pointer here |
| `scripts/local-ci/README.md` | Frozen legacy bundle; not v2 |
| `.analysis/202608/harnessv2-tui-agnostic-plan.md` | Build plan, T-\* / H-\* ids |
| `docs/aawm-alias-config.md` | CFG-023 catalog + access-log replacement |
| `docs/litellm-alpha.md` | Alpha instance notes |
