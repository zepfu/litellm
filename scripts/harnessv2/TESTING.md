# Harness v2 — as built and testing process

Date: 2026-08-22

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
    checks.yaml           # health, HTTP suite, leftover uvicorn, JSONL
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
  fixtures/logs/          # leftover uvicorn / ASGI / clean rollup
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

`--model all` expands `compiled_aliases` only:

- `basic`, `work`, `work-other`, `expert`, `sota`
- `sota-openai`, `sota-xai`, `sota-alibaba`, `sota-moonshot`,
  `sota-deepseek`, `sota-zai`
- `codex-auto-review`

Skip prefixes: `aawm-`, `claude-`. Absent catalog ids
(`aawm-sota-zai`, …) are recorded so a picker must not treat them as
present.

Groups:

- `all` → compiled aliases
- `all-sota` → the six `sota-*` parents
- `orchestration_children` → `basic`, `work`, `expert`, `sota`
- `catalog_picker` → `work`, `sota-zai`

Ohmypi session `--model` is `litellm-alpha-passthrough/<alias>`, not a
bare alias and not `litellm-alpha/<alias>` (that lane is completions).

---

## 5. Kinds (`--test`) and intended process

Walk kinds in this order. Do **not** skip the logging gate.

| Kind | TUI | What it proves |
|---|---|---|
| `platform` | forbidden | Health, custom HTTP, error JSONL, Redis prefix SCAN, docker logs |
| `catalog` | optional | CFG-023/024 HTTP catalog; Ohmypi picker if `--tui ohmypi` |
| `model` | required | Waits for idle; standalone exact PONG or explicit provider 404 per alias |
| `orchestration` | required | Parent alias spawns children through Ohmypi `task` |

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
   `Error:`, recap-only, and non-idle panes fail. Needles that are
   substrings of the **sent prompt** are ignored (H-6).
6. Closes that session before the next model.

`--model` is repeatable / comma-separated. Default is group `all`.

Post-TUI `docker_logs` **requires** an AAWM route-rollup header
(`YYYYMMDD HH:MM:SS … /path`). A recap without a real Responses POST +
rollup is not pass evidence.

### 5.4 `orchestration` (interactive Ohmypi)

Steps: `health` → `tui_orchestration` → `docker_logs` → `error_jsonl`.

Default parent: `sota-openai` (group `all-sota`). Default children:
`basic`, `work`, `expert`, `sota`.

Launches the parent with tools enabled. Pastes
`prompts/orchestration.txt` with `{parent}` substituted. Wait needle is
`※ recap:` — **not** `omp-alpha-fanout`, which is inside the prompt and
would false-complete (H-6).

Spawn contract (H-7, **closed in code**): Ohmypi `task` has no
`model=` field. The parent must spawn `agent=basic`, `agent=work`,
`agent=expert`, and `agent=sota`. Those names are harness profiles
staged from `config/ohmypi-agents/` into `{cwd}/.omp/agents`
(`/tmp/omp-alpha-workspace/.omp/agents` by default). They are **not**
the built-in Ohmypi names (`scout`, `designer`, `reviewer`, …) and
**not** LiteLLM catalog ids. Session `--model` stays
`litellm-alpha-passthrough/<parent>`.

`※ recap:` is wait-complete only. Ohmypi 17.4 often finishes the parent
turn with the four-child date list and idle `hub` peers, without a recap
glyph. Missing recap is **not** a fail when `child_evidence.ok` is true.

The runner then calls `child_spawn_evidence()` against Ohmypi session
JSONL, including nested `session_dir/<parent-id>/*.jsonl` child
transcripts. Pass requires successful child completions for `basic` /
`work` / `expert` / `sota`. Ohmypi 17.4 delivers those as:

- `hub` job rows (`details.jobs[].resolvedModel` /
  `<task-result agent="…" status="completed">`)
- `customType=async-result` notices (a child can finish this way and
  then drop out of later `hub wait` snapshots)
- idle hub peers (`SotaDate [sota · sub · idle]`)
- nested child `session_init` / `model_change` plus a successful
  `yield` (nested `bash` `date` while the parent is still waiting is
  not enough)

A `Spawned N background agents using basic, work, expert, sota` line is
spawn intent, not a completed child result. Recap-only, `Unknown agent`,
failed preflight, or empty hub/task completions is fail. The wait loop
does not stop on recap while any of the four children is still missing.
Do not close the dedicated tmux session until those completions exist;
SIGHUP during `hub wait` is not a pass.

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
| Overlay | `PI_CONFIG_FILES=$HOME/.omp/agent/litellm-alpha.yml` |
| CWD | `/tmp/omp-alpha-workspace` |
| Session dir | `/tmp/omp-alpha-sessions` |
| tmux socket | `tmux37` |
| Alias lane | `litellm-alpha-passthrough` |
| Completions lane | `litellm-alpha` |
| Forbid | `-p`, `--print` |
| Model tools | off |
| Orchestration tools | on |
| Wait needle | `※ recap:` (wait signal only, never pass evidence) |
| Model pass needles | standalone exact `PONG`; `No endpoints found for` / `404 Not Found` / `status code 404` / `Error: 404` |
| Orchestration pass | `child_evidence.ok` (hub `<task-result>` / nested `yield`) |

Child env allow-prefixes include `PI_` / `OMP_` / `CODEX_` / `OPENAI_` /
`XAI_`. Deny Langfuse, DB, master key, Anthropic/Claude inheritance.

Do not send-keys into an existing operator `omp-alpha-test` pane and
call that a harness pass.

**Smoother path (dedicated session):** every `--test model` /
`--test orchestration` row creates `hv2-ohmypi-<alias>-<pid>` on
socket `tmux37`, with `PI_CONFIG_FILES=$HOME/.omp/agent/litellm-alpha.yml`
and alias lane `litellm-alpha-passthrough/<alias>`. Never `omp -p` /
`--print`. Never reuse leftover `omp-alpha-test`. Close the dedicated
session before the next alias.

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
leftover-uvicorn invert, Ohmypi forbid `-p`, dry-run plans, H-6 prompt
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

Leftover uvicorn ACCESS except `/health*` is **0** on a harness-owned
alpha window as of 2026-08-22 (see §11). Remaining compiled aliases
still need live Ohmypi `--test model`.

```text
python scripts/harnessv2/run.py \
  --instance litellm-alpha \
  --tui ohmypi \
  --test model \
  --model work \
  --write-artifact /tmp/hv2-model-work.json
```

Repeat `--model` / comma-separated ids, or omit it to expand `all`.

### Live orchestration

Same leftover-uvicorn gate. Recap is not enough: `tui_orchestration`
fails unless `child_evidence.ok` is true for `basic` / `work` /
`expert` / `sota`. Historical recap-only `ok: true` artifacts are
stale relative to the current gate.

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
5. `--test catalog --tui ohmypi`. HTTP CFG-023 is required **and**
   `tui_catalog` must find selector `litellm-alpha-passthrough/work`
   (and the other catalog sample). HTTP-only is not enough. Same halt
   rule.
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
7. Only then `--test model` per compiled alias through Ohmypi TUI
   (default expands `all`). On traceback / leftover uvicorn: halt,
   source-fix on the shared checkout, `docker restart litellm-alpha`
   only if watchfiles did not reload, resume the **same** row.
8. `--test orchestration --orchestration-parent sota-openai` only
   after leftover uvicorn is gone. Gate spawn on `child_evidence.ok`
   (successful Ohmypi `task` results for `basic`/`work`/`expert`/`sota`),
   not recap-only.
9. Keep `TEST_HARNESS.md` / this document / the plan file in sync when
   halt signatures or spawn contract change.

Alpha is testing-only. Uncommitted halt-fixes in the shared checkout
are visible to the bind-mount immediately after watchfiles reload.

---

## 11. Current status (2026-08-22)

Proven on `litellm-alpha` (do not re-claim without a new artifact):

| Claim | Status |
|---|---|
| `--test platform` | passed (`/tmp/hv2-platform.json`, `ok: true`) |
| CFG-023 catalog GET leftover uvicorn | 0 leftover ACCESS |
| T-5 leftover uvicorn (discovery probes) | closed: post-cursor leftover ACCESS except `/health*` is **0**, including native `GET /v2/model/info` **500**. Worker reloaded via watchfiles after `_logging.py` mtime `2026-08-22T02:36:57Z`. Cursor `2026-08-22T02:44:06Z`. Ohmypi TUI was not used for this proof. |
| `--test model` `work` / `basic` / `expert` / `sota` / `work-other` | passed: real Responses POST + rollup, `halted: false` |
| `--test orchestration --orchestration-parent sota-openai` | passed (`/tmp/grok-goal-4ce5b5ad827f/implementer/hv2-orch.json`, `ok: true`, `halted: false`). Child evidence: hub `<task-result>` for `work`/`expert`/`sota` plus nested `yield` for `basic` (session `2026-08-22T06-21-45-763Z_01a02821-e7a3-7000-871c-04f719688caf`). Historical recap-only `ok: true` and the 2026-08-22T06:17 premature-close (`hv2-orch-premature-close.json`, nested `date` without `yield`) are not spawn proof. |
| T-1 unhashable `type` list / ASGI | closed |
| T-4 `UnicodeDecodeError` truncated UTF-8 peek | closed (incremental decoder `errors="ignore"`) |
| H-6 wait needle | closed (`※ recap:`; ignore needles contained in the sent prompt) |
| H-4 leftover-uvicorn invert | in tree (any ACCESS except `/health*` is halt) |
| H-7 Ohmypi spawn | **closed in code**: `orchestration.txt` uses `agent=`; profiles staged into `{cwd}/.omp/agents`; recap-only / `Unknown agent` fail |

**Not closed:**

| Gap | Notes |
|---|---|
| Remaining `--test model` | compiled aliases through `sota-zai` completed on a halted `all` row; `codex-auto-review` resumed after truncated-chrome + ASGI wrap (`hv2-model.json`). Sequential `--model` rows that together cover `all` are valid. Do not treat the halted `all` JSONL as a full pass. |
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
