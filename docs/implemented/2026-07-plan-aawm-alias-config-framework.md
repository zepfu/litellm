# AAWM Config-Driven Alias Framework + Failure-Handling (D1-583 / D1-584) — Implementation Plan

**Date:** 2026-07-22
**Author:** orchestrator (self-authored /spec)
**Subject:** Config-driven AAWM alias routing with an isolated `read` pilot (D1-583) plus a maintainable provider/TUI failure-classification framework (D1-584), built as one cohesive effort sharing an open error-class vocabulary seam.
**Scope:** `litellm/proxy/pass_through_endpoints/aawm_alias_routing/` (state, retry, durable, new compiler/schema/vocabulary/classification modules), `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` (selection engine, refresh route), a new YAML config file + fixtures under `.analysis/error-archive/`, and tests under `tests/test_litellm/`.
**Status:** PROMOTED (2026-07-22)

---

## Executive Summary

Today every AAWM alias (`aawm-read`, `aawm-low`, `aawm-sota*`, `aawm-code`, `aawm-orchestration`, and their `-anthropic` variants) is a **hard-coded Python tuple** in `policy.py` — ordered candidate lists with a `last_resort: bool`, consumed by a large selection engine inside the 27,418-line `llm_passthrough_endpoints.py` (`policy.py:71-509`, consumed at `llm_passthrough_endpoints.py:584-595`, selected at `:5497-5546`, dispatched at `:22886`). There is **no numeric priority, no YAML, no hot-reload, and no unified failure taxonomy** — failure→cooldown mapping is scattered across provider-specific classifiers (e.g. `_classify_kimi_code_auto_agent_probe_failure` at `:5862`) that each set a flat cooldown (e.g. `CODEX_AUTO_AGENT_DEFAULT_COOLDOWN_SECONDS = 3h`, `policy.py:14`).

This plan introduces, as one program built in dependency order:

1. **An open error-class vocabulary + typed `FailureEvent`** (the shared seam). 584 owns and grows it; 583 keys cooldown policy on it. Class names are an **open registry**, not a closed enum, so later taxonomy growth never forces a 583 schema change.
2. **Confidence-tiered N-of-M cooldown evidence** with a three-valued `origin` (`upstream`/`client`/`unknown`); only `upstream` cools, `unknown` never cools, replacing today's single-event flat-3h behavior.
3. **A typed, validated YAML schema + compiler** producing an **immutable routing snapshot** with first-class numeric priority (descending; ties → declared distribution strategy → declaration order; `0` reserved = last resort), proportional routing weights, a per-model `tui_attached` flag, schedule windows, and a deferred `config_epoch`.
4. **Selector integration for the `read` pilot only** — the new `read` alias (go-forward replacement for `aawm-read`/`aawm-read-anthropic`, additive until operator-approved cutover) is resolved from the compiled snapshot; all other aliases keep their hard-coded tables untouched.
5. **An unauthenticated LAN refresh endpoint** that validates→compiles→atomically swaps the snapshot, fail-closed preserving last-known-good.
6. **error-archive-seeded classification fixtures** with the unified error CSV as a coverage checklist, plus a **shadow parity harness** proving the pilot reproduces today's `read`/`low` lane selection.

**No database change and no new routing-decision recording.** Per operator, routing decisions are NOT recorded beyond what `session_history` already persists as-routed (`inbound_model_alias`, `provider`, `model` — `sql.py:20-22`). The pilot adds no config-hash/version persistence, no telemetry, no DDL, no migration, no DB Foundation gate. (`config_hash`/`config_version` remain in-memory properties of the compiled snapshot for identity and the refresh-endpoint response only — never persisted.)

Out of scope (explicitly deferred, not dispatched): migrating the other six alias families to config; full error-policy enforcement beyond cooldown; multi-worker config consensus (deployment is single-worker); cutover of `aawm-read` → `read`.

---

## Rollout Order

```
Wave 1: Vocabulary + FailureEvent        — new pure module (no deps)        ─┐ parallel
Wave 3: YAML schema + compiler + snapshot — new modules + read.yaml (no god-module edits) ─┘
  │                                          │
Wave 2: Classification + cooldown evidence  │  — wraps provider classifiers, N-of-M, origin (edits god-module)
  │  (depends on Wave 1)                     │
  └────────────────┬───────────────────────┘
                   │
Wave 4: Selector integration (read pilot)  — edits god-module; depends on Waves 1,2,3
  │
Wave 5: Refresh endpoint                   — edits god-module; depends on Wave 3
  │
Wave 6: Fixtures + shadow-parity + smoke   — depends on Waves 2 + 4
  │
Wave 7: Backlog reconciliation + promotion — orchestrator-inline, at close-out (depends on all PASS)
```

**Dependencies:**
- Wave 2 depends on Wave 1 (`FailureEvent`/vocabulary types).
- Wave 4 depends on Waves 1, 2, 3.
- Wave 5 depends on Wave 3 only.
- Wave 6 depends on Waves 2 and 4.
- Wave 7 depends on every prior wave passing QA.

**Same-file serialization:** Waves 2, 4, 5 all edit `llm_passthrough_endpoints.py`. They MUST be serialized against each other. Waves 1 and 3 add new modules (no god-module edits) and run in parallel with each other and with Wave 2's new-module portion.

**Validation (litellm-native, not aawm gate):** engineers validate with `make test-unit` (or `poetry run pytest tests/test_litellm/<target> -q`) plus `ruff`/`mypy`/`black` on changed files. `run_gate_check` and pytest-classifier are NOT used in this fork. Landing uses the existing AAWM worktree → rebase-onto-`develop` → stage → land flow already used in this repo.

**Maximum concurrent agents: 2** (Engineer A on Waves 1+2's new-module work in parallel with Engineer B on Wave 3; the god-module edits then serialize).

---

## Implementation Waves

<!-- SPECIFICATION ONLY — grouped by surface area, ordered by dependency. -->

### Wave 1: Error-class vocabulary + `FailureEvent` (the shared 583/584 seam)

**Depends on:** (none)
**Scope:** new module `litellm/proxy/pass_through_endpoints/aawm_alias_routing/failure_vocabulary.py`

#### Impact Analysis
`N/A — net-new functionality, no existing behavior modified.` (New module; nothing imports it until Waves 2/3/4.)

#### Test Spec (tester's input)
**Test files:**
  - `tests/test_litellm/proxy/aawm_alias_routing/test_failure_vocabulary.py` — unit
**Test cases (must fail before implementation):**
  - `test_vocabulary_is_open_registry_not_enum` — an unknown class name can be registered/looked up without raising; the registry accepts new names at runtime (not a frozen `Enum`).
  - `test_seed_classes_present` — the ~12 structured seed classes exist (e.g. `rate_limit`, `capacity`, `usage_limit`, `transient`, `auth`, `quota_exhausted`, `model_unavailable`, `provider_5xx`, `provider_4xx_other`, `serialization`, `client_cancelled`, `unknown`).
  - `test_failure_event_fields` — `FailureEvent` carries `class_name: str`, `origin: Literal["upstream","client","unknown"]`, `confidence: Literal["structured","marker","unknown"]`, `provider: Optional[str]`, `scope: Literal["provider","account","model","lane","alias"]`, `retryable: Optional[bool]`, `evidence: Mapping[str,str]` (sanitized, no payloads/secrets).
  - `test_only_upstream_is_coolable` — `is_coolable(event)` returns True only when `origin == "upstream"`; `client` and `unknown` return False.
**Assertions:** field names/types exact; `is_coolable` truth table exact.

#### Source Spec (engineer's input)
**Source files:**
  - `failure_vocabulary.py` — the open `FailureClassRegistry` (dict-backed, seedable, accepts new names), the frozen `FailureEvent` dataclass, the `origin`/`confidence`/`scope` literals, and `is_coolable()`. Seed with the ~12 structured classes. No I/O, no provider imports.

---

### Wave 2: Failure classification adapter + N-of-M cooldown evidence

**Depends on:** Wave 1
**Scope:** new module `aawm_alias_routing/classification.py`; modifications to `state.py` (evidence counters) and the cooldown-setting seam in `llm_passthrough_endpoints.py` / `retry.py`.

#### Impact Analysis
**Type:** modification
**Affected symbols:** provider classifiers such as `_classify_kimi_code_auto_agent_probe_failure` (`llm_passthrough_endpoints.py:5862`), the failure recorder `_record_auto_agent_alias_attempt_failure` (`:6728`), and cooldown setters `set_cooldown_memory`/`set_monotonic_cooldown_map` (`state.py:70-83`, `retry.py:98-109`).
**Callers/importers:** the dispatch loop `_dispatch_auto_agent_alias_candidate_request` (`:22886`) and OpenRouter durable quota cooldown (`:2730`). Existing classifiers keep their behavior; the adapter **wraps** their output into a `FailureEvent` without changing their return contracts.
**Grep verification (run at execution):** `grep -rn "set_cooldown_memory\|set_monotonic_cooldown_map\|_record_auto_agent_alias_attempt_failure" litellm/proxy/pass_through_endpoints/` — enumerate all cooldown-write sites; route the `read`-pilot lane through the evidence gate, leaving other lanes' current behavior behind a lane check.
**Behavior boundary:** classification (produce `FailureEvent`) stays separate from policy (decide cool/scope/duration).

#### Test Spec (tester's input)
**Test files:**
  - `tests/test_litellm/proxy/aawm_alias_routing/test_classification.py` — unit
  - `tests/test_litellm/proxy/aawm_alias_routing/test_cooldown_evidence.py` — unit
**Test cases (must fail before implementation):**
  - `test_structured_code_maps_to_upstream_event` — 429/quota/auth → correct class, `origin="upstream"`, `confidence="structured"`.
  - `test_freetext_marker_low_confidence` — message-only match → `confidence="marker"`.
  - `test_client_cancelled_is_client_origin` — an `asyncio.CancelledError`-shaped signal → `client_cancelled`, `origin="client"`, `is_coolable() is False`.
  - `test_unknown_defaults_never_cools` — unrecognized failure → `class_name="unknown"`, `origin="unknown"`, not coolable.
  - `test_structured_cools_on_single_event` — N=1 for `confidence="structured"`.
  - `test_marker_requires_n_of_m` — no cool on 1 marker-only event; cools at the configured N-within-window (default 3-in-60s).
  - `test_cooldown_scope_narrowest` — auth → account/credential scope; model-not-found → model scope; provider 5xx storm → provider scope.
  - `test_duration_signal_derived_then_backoff` — uses Retry-After/quota-reset when present; else capped exponential backoff (`exponential_backoff_seconds`, `retry.py:176`) instead of flat 3h.
  - `test_half_open_probe_recovery` — after expiry a single trial is allowed; success restores, failure re-cools with backoff.
**Assertions:** exact origin/confidence/scope; exact N-of-M gating; duration source precedence.

#### Source Spec (engineer's input)
**Source files:**
  - `classification.py` — adapter mapping provider classifier outputs + exception shapes → `FailureEvent`; a `CooldownEvidenceGate` (confidence-tiered N-of-M, narrowest-scope, signal-derived-else-backoff duration capped, half-open recovery).
  - `state.py` — bounded per-key evidence counters (sliding window) alongside existing cooldown maps (`AliasFamilyState`, `:29`).
  - `llm_passthrough_endpoints.py` — route the `read`-pilot lane's cooldown writes through the evidence gate; add the three-valued `origin` at the failure-record site (`:6728`). Non-pilot lanes retain current behavior behind a lane check.

---

### Wave 3: YAML schema + compiler + immutable snapshot

**Depends on:** (none for authoring; references Wave 1 vocabulary)
**Scope:** new modules `aawm_alias_routing/config_schema.py`, `config_compiler.py`, `config_snapshot.py`; new config file `litellm/proxy/aawm_alias_config/read.yaml`.

#### Impact Analysis
`N/A — net-new functionality, no existing behavior modified.` (New modules + YAML; the hard-coded `policy.py` tables are untouched. Wave 4 wires the snapshot in, only for `read`.)

#### Test Spec (tester's input)
**Test files:**
  - `tests/test_litellm/proxy/aawm_alias_routing/test_config_schema.py` — unit
  - `tests/test_litellm/proxy/aawm_alias_routing/test_config_compiler.py` — unit
**Test cases (must fail before implementation):**
  - `test_rejects_unknown_keys_and_malformed` — rejects unknown fields, missing required fields, non-typed values.
  - `test_rejects_arbitrary_behavior` — a candidate whose `route_family`/`provider` is not a registered code behavior is rejected at compile (references-only, no eval).
  - `test_numeric_priority_required_and_typed` — each candidate requires an integer `priority`; non-int rejected.
  - `test_priority_descending_with_zero_last_resort` — compiled ordering descending; `priority: 0` placed last regardless of declaration order.
  - `test_tie_break_distribution_then_declaration_order` — equal non-zero priorities resolve by declared strategy (`proportional`/`round_robin`); with no strategy, stable declaration order.
  - `test_proportional_weights_normalized` — proportional weights normalize; account/credential binding NOT expressed in YAML.
  - `test_tui_attached_flag_compiles` — a candidate may declare `tui_attached: <client>`; compiler records it for per-model gating.
  - `test_schedule_windows_utc_only` — windows UTC; overlaps resolve deterministically; non-UTC rejected.
  - `test_inheritance_resolves` — typed inheritance (defaults → alias → candidate) merges without duplicate-definition ambiguity.
  - `test_config_epoch_and_hash_present` — snapshot carries `config_epoch`, content `config_hash`, `config_version`.
  - `test_snapshot_is_immutable` — compiled snapshot is frozen (mutation raises).
  - `test_error_class_refs_open_vocabulary` — error rules may reference class names not yet in the seed registry without failing compile.
**Assertions:** exact rejection, ordering, tie semantics, immutability, open-vocabulary tolerance.

#### Source Spec (engineer's input)
**Source files:**
  - `config_schema.py` — typed schema (pydantic v2) for alias/candidate/schedule/error-rule with inheritance; validation rejecting malformed/ambiguous/arbitrary-behavior YAML.
  - `config_compiler.py` — compile validated YAML → immutable `RoutingSnapshot` (numeric priority, normalized weights, `tui_attached`, schedule windows, open error-class refs), with `config_hash`, `config_version`, `config_epoch`.
  - `config_snapshot.py` — the frozen snapshot type + process-local holder with an atomic swap primitive (used by Wave 5).
  - `litellm/proxy/aawm_alias_config/read.yaml` — the `read` pilot mirroring current `CODEX_AAWM_LOW_CANDIDATES` order (`policy.py:227-277`): `cohere/north-mini-code:free`, `owl-alpha`, `deepseek-v4-flash`, `big-pickle`, `gpt-5.6-luna`, `qwen3.6-flash`, `kimi-for-coding`, then `gpt-5.4-mini` at `priority: 0`; plus `alibaba_token_plan/qwen3.8-max-preview` highest during its promo window. Priorities assigned descending to match that order (operator to confirm exact integers + promo-window boundaries).

---

### Wave 4: Selector integration for the `read` pilot

**Depends on:** Waves 1, 2, 3
**Scope:** `llm_passthrough_endpoints.py` (selection engine), reading the compiled snapshot for the `read` alias and recording routing outcome via existing columns.

#### Impact Analysis
**Type:** modification (scoped to the new `read` alias only)
**Affected symbols:** `_get_codex_auto_agent_candidates_for_alias` (`:584`), `_get_anthropic_auto_agent_candidates_for_alias` (`:595`), the selection block (`:5497-5546`, `last_resort`/affinity/`selection_reason`), and the existing session-history write path.
**Callers/importers:** `_dispatch_auto_agent_alias_candidate_request` (`:22886`) is the single dispatch path; the candidate getters are the single lookup path. For `read` these consult the snapshot; **all other aliases fall through to the existing `CANDIDATES_BY_ALIAS` tables unchanged** (`policy.py:292-305`, `:498-509`).
**Grep verification (run at execution):** `grep -rn "_get_codex_auto_agent_candidates_for_alias\|_get_anthropic_auto_agent_candidates_for_alias\|CANDIDATES_BY_ALIAS" litellm/` — confirm the snapshot branch is gated on `read` and every other caller is untouched.
**TUI gate correctness:** per-model — `_extract_auto_agent_alias_client_product_label` (`:2839`) already normalizes to `Claude/x.y`. On undetermined TUI, exclude only `tui_attached` candidates; the chain continues to non-TUI candidates so the alias still resolves.
**Persistence:** none added. Routing decisions are NOT recorded per operator — `session_history` keeps its existing as-routed behavior unchanged (`inbound_model_alias`/`provider`/`model`). The selector adds no session_history write path and no config-hash/version persistence.

#### Test Spec (tester's input)
**Test files:**
  - `tests/test_litellm/proxy/aawm_alias_routing/test_read_pilot_selection.py` — unit
**Test cases (must fail before implementation):**
  - `test_read_alias_uses_snapshot` — `read` resolves candidates from the compiled snapshot, not `CODEX_AAWM_LOW_CANDIDATES`.
  - `test_other_aliases_unchanged` — `aawm-read`, `aawm-low`, `aawm-sota` still resolve from the hard-coded tables.
  - `test_priority_descending_selection` — higher priority first; `priority:0` only when all others cooled/ineligible.
  - `test_proportional_tie_distribution` — equal-priority candidates split by weight over many selections within tolerance.
  - `test_tui_attached_excluded_on_unknown_tui` — with no client-product label, a `tui_attached: Claude` candidate is skipped and a non-TUI candidate is selected; alias still resolves.
  - `test_tui_attached_selected_when_identified` — with `Claude/x.y` present, the `tui_attached: Claude` candidate is eligible.
  - `test_schedule_window_close_stops_new_affinity` — after a window closes, no NEW affinity to the out-of-window model; an existing affinity-pinned session continues on it (`AliasFamilyState.get_affinity_memory`, `state.py:102`).
  - `test_no_new_routing_decision_recording` — selecting a `read` candidate adds no new session_history write path and no config-hash/version persistence; existing as-routed recording is unchanged.
**Assertions:** exact selection order, TUI gating truth table, affinity-vs-schedule rule, no new recording path added.

#### Source Spec (engineer's input)
**Source files:**
  - `llm_passthrough_endpoints.py` — for `read`, resolve from the snapshot; apply numeric-priority ordering + proportional tie distribution + per-model TUI gate + schedule-window/affinity rule. Add NO new session_history/recording logic — routing decisions are not recorded per operator. Reuse existing candidate-scoped cooldown/affinity state (`state.py`); thread `config_epoch` so a later swap invalidates stale keys.

---

### Wave 5: Operator refresh endpoint (unauthenticated, fail-closed)

**Depends on:** Wave 3
**Scope:** new route in `llm_passthrough_endpoints.py` (registered on `router`, `:1103`); atomic swap via `config_snapshot.py`.

#### Impact Analysis
`N/A — net-new functionality, no existing behavior modified.` (Adds one POST route; does not alter existing routes. Endpoint is intentionally open to any LAN client per operator decision — see Risks R-3.)

#### Test Spec (tester's input)
**Test files:**
  - `tests/test_litellm/proxy/aawm_alias_routing/test_refresh_endpoint.py` — API (FastAPI TestClient)
**Test cases (must fail before implementation):**
  - `test_valid_refresh_compiles_and_activates` — valid YAML compiles + atomically activates; response returns attempted+active `config_hash`/`config_version` and `changed=true`.
  - `test_noop_refresh_reports_no_change` — identical re-post → `changed=false`, same active hash.
  - `test_invalid_refresh_fails_closed` — malformed config rejected; previously active snapshot remains active (last-known-good); secret-safe validation errors.
  - `test_in_flight_uses_prior_snapshot` — a selection begun before the swap uses the prior immutable snapshot; a selection after uses the new one.
  - `test_no_auth_required` — accepts an unauthenticated LAN request.
  - `test_response_omits_secrets` — response/audit never include credentials or raw config secrets.
**Assertions:** activation atomicity, fail-closed last-known-good, no-op detection, secret-safe diagnostics.

#### Source Spec (engineer's input)
**Source files:**
  - `llm_passthrough_endpoints.py` — add `POST /aawm/alias-config/refresh` (no auth dependency): read the configured YAML source, validate+compile via Wave 3, atomically swap the active snapshot; on failure preserve last-known-good and return validation errors; emit attempted/active `config_hash`+`config_version`, changed-vs-no-op, and activation timestamp (secret-safe).

---

### Wave 6: Fixtures, shadow-parity harness, and smoke

**Depends on:** Waves 2, 4
**Scope:** classification fixtures seeded from `.analysis/error-archive/`, the unified error CSV as coverage checklist, a shadow harness proving pilot parity, and smoke tests.

#### Impact Analysis
`N/A — net-new tests/fixtures, no source behavior modified.`

#### Test Spec (tester's input)
**Test files:**
  - `tests/test_litellm/proxy/aawm_alias_routing/test_failure_fixtures.py` — unit
  - `tests/test_litellm/proxy/aawm_alias_routing/test_read_pilot_shadow_parity.py` — unit
  - `tests/test_litellm/proxy/aawm_alias_routing/test_smoke_alias_config_framework.py` — smoke (import + compile)
**Test cases (must fail before implementation):**
  - `test_archive_incidents_classify` — real incidents from `.analysis/error-archive/*.jsonl` classify to expected `FailureEvent`s (fixtures seeded from observed wire data, not the CSV).
  - `test_csv_coverage_checklist` — every structured `Machine Code / Type / Event` in `agentic_tui_error_code_catalog_unified_2026-07-20.csv` whose `Layer` is provider-reachable maps to a registered class OR is an explicit known gap; TUI-layer rows asserted out-of-scope (never coolable).
  - `test_shadow_parity_read_vs_low` — snapshot-driven `read` selection reproduces current `CODEX_AAWM_LOW_CANDIDATES` ordering/eligibility (shared-state contact points documented, not asserted-isolated; tolerance documented).
  - `test_module_imports` (smoke) — `from litellm.proxy.pass_through_endpoints.aawm_alias_routing import config_compiler, failure_vocabulary, classification` loads.
  - `test_read_yaml_compiles` (smoke) — `read.yaml` compiles to a valid snapshot with a stable `config_hash`.
**Assertions:** archive incidents classify correctly; CSV coverage accounted; parity within tolerance.

#### Source Spec (engineer's input)
`N/A — test/fixture-only wave; no application source changes.` Fixtures derive from `.analysis/error-archive/` incident JSONL; the CSV is a checklist input, not the fixture source of truth.

---

### Wave 7: Backlog reconciliation + plan promotion (close-out bookkeeping)

**Depends on:** every prior wave (QA PASS)
**Surface area:** Backlog / documentation (`.analysis/todo.md`, `.analysis/todo.deferred.md`, `.analysis/completed-YYYY-MM-DD.md`, `docs/implemented/`)
**Type:** Non-testable — orchestrator-inline bookkeeping (no agent dispatch, no worktree)

#### Impact Analysis
**Type:** modification (documentation/backlog files only; no source or tests)
**Affected files:** `.analysis/todo.md` (remove `D1-583` block), `.analysis/todo.deferred.md` (remove `D1-584` block), `.analysis/completed-YYYY-MM-DD.md` (dated for completion date; create if absent), `docs/implemented/2026-07-aawm-alias-config-framework.md`.
**Grep verification (run at execution):** `grep -n "D1-583" .analysis/todo.md` and `grep -n "D1-584" .analysis/todo.deferred.md` must each return 0 after this wave; the same IDs appear exactly once in the dated `completed-*.md`.
**Ordering safety:** orchestrator-inline, non-worktree. Copy-before-delete: (1) append completed-file entries, confirm on disk non-empty, (2) only then remove todo blocks. Never delete a todo block before its completed entry is verified. Do not dispatch a worktree agent (`.analysis/` is symlinked into worktrees — deletions there destroy originals).

#### Test Spec
`N/A — close-out bookkeeping wave. No new behavior to test; correctness is the grep verification above.`

#### Source Spec (orchestrator-inline — no agent dispatch)
1. **Write closeout entries** into `.analysis/completed-YYYY-MM-DD.md` — one `## D1-583 - <title>` and one `## D1-584 - <title>` block matching the established completed-file schema exactly: `Created on:` / `Initiated on:` / `Completed on:`, `Linked completed item:` (583↔584 + this plan), `Outcome:` (open vocabulary + `FailureEvent`; N-of-M cooldown with three-valued origin; YAML schema/compiler/immutable snapshot; `read` pilot selector integration additive/others-untouched; unauthenticated fail-closed refresh endpoint; no DB change), `Implementation and release:` (commit SHAs on `main`/`develop`, any tag), `Verification:` (QA verdicts per wave, smoke PASS, shadow-parity result — gate check intentionally skipped per operator), `Changed tracked paths:`.
2. **Record the deferred remainder as NEW backlog items** in `.analysis/todo.md`: (a) migrate the remaining six alias families to config; (b) full error-policy enforcement beyond cooldown (retry/failover actions by class); (c) `aawm-read`/`aawm-read-anthropic` → `read` cutover after operator-approved validation; (d) multi-worker config consensus if the deployment scales past one worker. Each references completed D1-583/D1-584.
3. **Remove** the `D1-583` block from `todo.md` and the `D1-584` block from `todo.deferred.md` — only after step 1's entries are verified on disk.
4. **Promote** the plan: copy to `docs/implemented/2026-07-aawm-alias-config-framework.md`, set `Status: IMPLEMENTED (YYYY-MM-DD)`, verify the copy exists non-empty, commit.

#### Acceptance Criteria
- [ ] `D1-583` absent from `todo.md`; `D1-584` absent from `todo.deferred.md`.
- [ ] Both IDs present exactly once in the dated `completed-*.md` with full sections.
- [ ] Deferred-remainder follow-up items created in `todo.md`.
- [ ] Plan promoted to `docs/implemented/` and committed.

---

## Schema Verification

`N/A — no database change. Routing-outcome observability uses existing session_history columns (inbound_model_alias, provider, model — sql.py:20-22); config hash/version ride the existing routing metadata/log path. No DDL, no migration, no DB Foundation gate.`

---

## Risks and Mitigations

| # | Risk | Mitigation |
|---|------|-----------|
| R-1 | Selection engine lives in a 27,418-line god-module; edits risk collateral behavior. | Scope all selection changes behind a `read`-alias branch; every other alias falls through to untouched tables. QA greps all caller sites (Wave 4 Impact Analysis). Waves 2/4/5 serialized on the file. |
| R-2 | Shadow "parity" cannot be perfectly isolated — the pilot shares candidate-scoped cooldown keys, OpenRouter free daily quota (process-global, `policy.py:26`), and the production OpenAI account for last-resort `gpt-5.4-mini`. | Reframed per operator: prove no production selection-logic/config change + reproduce ordering, NOT state isolation. Document shared-state contact points; use a documented tolerance. |
| R-3 | Unauthenticated refresh endpoint is a config-mutation surface reachable by any LAN client. | Operator-accepted for this single-operator, no-auth instance. Safety is fail-closed + last-known-good + secret-safe diagnostics; the endpoint can only replace config with a valid compile. |
| R-4 | Error CSV is docs-derived (343 rows, ~99 normalized classes, ~30% free-text-only) — not wire-observed. | Fixtures seed from `.analysis/error-archive/` observed incidents; CSV is a coverage checklist only. Taxonomy collapses to ~20 (code already runs ~12). Open registry absorbs later growth. |
| R-5 | Client-abort vs upstream-failure disambiguation currently holds only because `CancelledError` is a `BaseException` slipping past `except Exception`. | Wave 2 adds an explicit three-valued `origin`; `unknown` never cools; the invariant is asserted by tests. |
| R-6 | Multi-worker atomic swap is unsolved in general. | Deployment is single-worker; intra-process atomic snapshot swap only; instance-wide consensus explicitly deferred. |
| R-7 | `mcppg` MCP tool intermittently fails checkout ("Catalog not yet loaded" / "checkout timed out") — this is the tool, not the DB (DB is up). | Not a plan blocker: no DB change and gate check skipped. If live SQL inspection is needed, restart the `mcppg` MCP server (per memory ac6fe6bc). |

---

## Close-Out Checklist

- [ ] QA is MANDATORY for every wave. No exceptions.
- [ ] QA dispatched and PASS for every wave (inline under h4)
- [ ] Gate check — **SKIPPED per operator.** Per-wave engineer validation via `make test-unit`/`poetry run pytest <target>` + `ruff`/`mypy` is the substitute.
- [ ] Smoke test PASS (`tests/test_litellm/proxy/aawm_alias_routing/test_smoke_alias_config_framework.py` via `poetry run pytest`)
- [ ] Ops validation (POST refresh with valid + invalid YAML; confirm fail-closed + idempotent no-op on 2nd identical POST)
- [ ] Operator nudges captured in retrospective (real-time, not batched)
- [ ] Lessons learned / Hindsight (≥5 items)
- [ ] Tool errors documented
- [ ] Suggested persona/template adjustments
- [ ] **Wave 7 backlog reconciliation:** `D1-583` moved out of `todo.md` and `D1-584` out of `todo.deferred.md` into the dated `completed-YYYY-MM-DD.md` with full closeout notes matching the established format — copy-before-delete, verified on disk
- [ ] Deferred-remainder follow-up items created in `todo.md`
- [ ] Plan promoted to `docs/implemented/2026-07-aawm-alias-config-framework.md`

---

## Smoke Test Procedure

Smoke tests are pytest functions in `tests/test_litellm/proxy/aawm_alias_routing/test_smoke_alias_config_framework.py`, run via `poetry run pytest <path> -q` (NOT `run_gate_check`, NOT bare `python -c`).

Required smoke assertions:
- `test_module_imports()` — `from litellm.proxy.pass_through_endpoints.aawm_alias_routing import config_compiler, failure_vocabulary, classification` loads.
- `test_read_yaml_compiles()` — compiles `read.yaml` to a valid immutable snapshot with a stable `config_hash`.
- `test_refresh_endpoint_registered()` — `POST /aawm/alias-config/refresh` is present on the app router.

---

## Confidence Notes (Pre-Execution)

| Wave | Pre-Execution | Post-Execution | Notes |
|------|--------------|----------------|-------|
| 1 Vocabulary/FailureEvent | HIGH | | Pure module, no deps. |
| 2 Classification/cooldown | MEDIUM | | Touches the god-module cooldown seam; provider classifiers wrapped without changing contracts. |
| 3 Schema/compiler | HIGH | | Net-new modules; well-specified semantics. |
| 4 Selector integration | LOW | | Highest risk — deep in the 27k-line selection engine; TUI gate + proportional distribution + schedule/affinity interplay. |
| 5 Refresh endpoint | MEDIUM | | Atomicity + fail-closed crux; single-worker simplifies. |
| 6 Fixtures/shadow/smoke | MEDIUM | | Parity tolerance + CSV coverage curation. |
| 7 Backlog reconciliation | HIGH | | Orchestrator-inline bookkeeping. |

---

## Dispatch Plan

### Keepalive Cron
- Job ID: `825ddd50` (hourly at :13, session-only). Do NOT cancel per skill policy.

### Wave 0: Infrastructure Health Check

| Check | Command | Expected | Actual (2026-07-22) |
|-------|---------|----------|--------|
| CWD | `pwd` | `/home/zepfu/projects/litellm` | ✅ |
| Branch | `git branch --show-current` | working branch | `main`; `develop`/`origin/develop` exist — worktree agents rebase onto `origin/develop` per dispatch prompt |
| Worktrees | `ls .claude/worktrees/` | empty | ✅ empty |
| Gate baseline | — | — | ⏭️ SKIPPED per operator |
| Dev DB | — | — | ✅ up (mcppg MCP tool was flaky, not the DB); no DB change required anyway |
| MCP tasks | `list_tasks()` | none stale | ✅ none |

**Wave 0 verdict: CLEAR — proceed.** Native `TaskCreate` tool not surfaced this session; wave progress is tracked in the Dispatch Log below instead of native task rows.

### Dispatch Log

**Dispatch note (litellm fork):** agent-frontmatter worktree auto-isolation does NOT fire in this session — subagents land in the main tree unless `isolation: "worktree"` is passed **explicitly** on the Agent call. All dispatches here must set it explicitly. (First tester attempt `tester-w1-5` blocked with no worktree; re-dispatched as `tester-w1-5b` with explicit isolation.)

| Wave | Phase | Agent | Target files | Worktree | Result | Notes |
|------|-------|-------|-------------|----------|--------|-------|
| 1–5 | test | tester (team, ×2) | — | none | SUPERSEDED | team dispatches — no worktree isolation; stopped |
| 1–5 | test | tester (non-team) | 8 files in `tests/test_litellm/proxy/aawm_alias_routing/` | agent-a499c90d3515dc161 | LANDED | Red confirmed (7 collection errors); commit `8489c0c210` → merge `07f991137f`. Non-team dispatch got frontmatter worktree isolation ✅ |

**Branch/land fix:** the `land` tool requires the main checkout on `develop`; this fork's main checkout was on `main`. Switched main checkout to `develop` (ff'd to `origin/develop` `8e20d6fe46`) — stays on develop for the duration of execution; promote develop→main at close-out.

| 1+2 | impl | engineer A (non-team) | `failure_vocabulary.py`, `classification.py`, `state.py` (NO god-module) | agent-a97880be2e31d3627 | LANDED | 20 tests pass, ruff clean, no new mypy; commit `0279dd89a4` → merge `e11b27ca4b`. **Deviation:** god-module cooldown-seam wiring deferred to Wave 4 (read-pilot lane doesn't exist until then — correct sequencing). |
| 3 | impl | engineer B1 (non-team) | `config_schema.py`, `config_compiler.py`, `config_snapshot.py`, `read.yaml` | agent-a2d8fe137045dc96e | LANDED | 21 tests pass, ruff/mypy/black clean; commit `879f54d5e6` → merge `fdc0631c6e`. Priorities 80..20, qwen promo=1000, gpt-5.4-mini=0 |
| 4+5 | impl | engineer B2 (non-team) | selector + refresh route | agent-aef807bd82ba804ae (auto-cleaned) | FAILED | provider exhaustion (429, `aawm_anthropic_auto_agent_redispatch_required`) after ~48min investigation, 0 file edits. Split into B2a (Wave 4) + B2b (Wave 5); investigation map handed to redispatch. |
| 4 | impl | engineer B2a (non-team) | selector + cooldown-gate wiring in `llm_passthrough_endpoints.py` | agent-a757fc1426adbc2f0 | LANDED | 8/8 read-pilot + 49/49 suite pass, ruff clean, no new mypy; commit `34a4690280` → merge `ef0aad1017`. Gated on `read` alias only; other aliases untouched. Redispatch w/ map+budget succeeded. |
| 5 | impl | engineer B2b (non-team) | refresh endpoint route in `llm_passthrough_endpoints.py` | agent-a1709fd96d874b4df | LANDED | 6/6 refresh + 55/55 suite pass, ruff clean, no new mypy; commit `08f0a67273` → merge `ef80aa7c29`. Fail-closed last-known-good, no-op detection, secret-safe. |
| 6 | test | tester (non-team) | `test_failure_fixtures.py`, `test_read_pilot_shadow_parity.py`, `test_smoke_alias_config_framework.py` | agent-a491390384529eb22 | LANDED | 70 passed (full dir), ruff clean; commit `d484d13ff6` → merge `4876f4ce80`. No real bugs; 48 unmapped CSV classes = documented known-gaps (open-vocab). |
| 1–6 | qa | qa (non-team, read-only) | full change set review | — | **PASS** | 70/70 new suite; scope contained to `read`; 0 new mypy. Regression 1158 passed / 1 failed = PROVEN pre-existing cross-test contamination (`test_rr054_package_contracts.py` leaks durable-runtime globals into `test_moonshot_alias_routing.py`; fails identically on pre-wave tree `07f991137f~1`). Full verdict in § QA Report. |
| fix | impl | engineer (non-team) | `tests/.../test_rr054_package_contracts.py` durable-globals cleanup | agent-ad8e5cb090b526127 | LANDED | save/restore `durable._clean_value`/`_dual_cache_override`; before=1 failed, after=50 passed both orders; commit `b078a98b34` → merge `5966e7cef1`. Suite fully green. |

**Deviation note (Wave 2→4):** the `CooldownEvidenceGate` + classification landed standalone; wiring it into the `read`-pilot lane's cooldown writes in `llm_passthrough_endpoints.py` moves to Wave 4 (Engineer B2), since the read-pilot selection path is created there. Added to B2 scope.

### Total Estimated Effort

| Category | Planned Dispatches | Notes |
|----------|-------------------|-------|
| Tester | 1 (+1 for Wave 6 fixtures) | Writes ALL failing tests |
| Engineer | 3 (by token budget + same-file serialization) | A: Waves 1+2 (vocabulary + classification/cooldown); B1: Wave 3 (compiler + read.yaml); B2: Waves 4+5 (selector + endpoint, serialized after A on the god-module) |
| QA | 1 | Reviews ALL changes together |
| Orchestrator-inline | 1 | Wave 7 backlog reconciliation + promotion |
| **Total waves** | **7** | |
| **Max concurrent agents** | **2** | Engineer A (Waves 1+2 new-module work) ∥ Engineer B1 (Wave 3) |

### Token Estimate

| Dispatch | Target files | Est. tokens | Rationale |
|----------|-------------|-------------|-----------|
| Tester | all `tests/test_litellm/proxy/aawm_alias_routing/*.py` + smoke | ~110k | ~40 test functions; reads policy.py/state.py/retry.py/selection block |
| Engineer A (Waves 1+2) | `failure_vocabulary.py`, `classification.py`, `state.py`, cooldown seam in `llm_passthrough_endpoints.py` | ~115k | New modules + wrap classifiers + evidence gate; god-module context |
| Engineer B1 (Wave 3) | `config_schema.py`, `config_compiler.py`, `config_snapshot.py`, `read.yaml` | ~80k | Compiler + config authoring; no god-module edits |
| Engineer B2 (Waves 4+5) | selection block + refresh route in `llm_passthrough_endpoints.py` | ~95k | Deep selection-engine integration + route; serialized after A |
| QA | (read-only) | ~40k | Review all changes |

### Wave dispatch assignments

- **Wave 1** (tester → Engineer A → QA): `failure_vocabulary.py`.
- **Wave 2** (tester → Engineer A → QA): `classification.py`, `state.py`, cooldown seam.
- **Wave 3** (tester → Engineer B1 → QA): compiler modules + `read.yaml`. Parallel with Engineer A.
- **Wave 4** (tester → Engineer B2 → QA): selector integration. Serialized after Waves 1,2,3 land (god-module + deps).
- **Wave 5** (tester → Engineer B2 → QA): refresh endpoint. After Wave 3; serialized on god-module after Wave 4.
- **Wave 6** (tester → QA): fixtures + shadow parity + smoke. After Waves 2,4.
- **Wave 7** (orchestrator-inline): backlog reconciliation + promote.

**Engineer dispatch prompt requirements (litellm-native):** first action `git fetch origin develop && git rebase origin/develop`; make ALL changes; validate with `poetry run pytest tests/test_litellm/proxy/aawm_alias_routing/<targets> -q` + `ruff check` + `mypy` on changed files (NOT `run_gate_check`, NOT bare `python -c`); call `stage` ONCE; `land`. Two agents targeting `llm_passthrough_endpoints.py` never run concurrently.

**Two-Strike Escalation:** for Engineer B2 (selection-engine graft), a second failure escalates to a principal for targeted diagnosis before a third dispatch (Agent Failure Recovery Protocol).

## Operator Nudges

1. **Design decisions pre-settled** — numeric priority, proportional-not-account YAML, session-history-outcomes-only, three-valued origin, per-model TUI fail-closed, schedule/affinity rule, open-LAN refresh endpoint, N-of-M evidence model — encoded as fixed constraints.
2. **Author the spec in-session** — operator directed the plan be authored directly, not offloaded.
3. **No DB change; DB is up; skip gate check** (2026-07-22) — I mis-read an `mcppg` MCP tool checkout timeout as "DB down" and had planned a session_history migration + DB Foundation gate. Operator corrected: the DB is up (tool was flaky), no schema change is required, and the gate check can be skipped. Plan revised: DB waves removed, validation is litellm-native (`make test-unit`/`poetry pytest` + ruff/mypy), gate check dropped from close-out.
4. **No routing-decision recording** (2026-07-22) — after removing the DB columns I still tried to persist config hash/version via existing metadata. Operator clarified they had already stated routing decisions are not to be recorded beyond `session_history`'s existing as-routed fields. Removed all routing-outcome persistence from Wave 4; the selector adds zero recording logic. `config_hash`/`config_version` remain in-memory only (snapshot identity + refresh response).

## Tool Errors and Infrastructure Failures

| Error | Frequency | Context | Resolution |
|-------|-----------|---------|------------|
| `mcppg` "Catalog not yet loaded" / "Read connection checkout timed out after 10.0s" | 3x during planning | `pg_columns` / `pg_query('SELECT 1')` on `session_history` | The `mcppg` MCP tool, not the DB (DB is up). Fell back to DDL source; no DB change needed so no live verification required. If needed later, restart the mcppg MCP server (memory ac6fe6bc). |

---

## Coverage Table (Phase 3)

| Ask | Satisfied by |
|-----|-------------|
| Config-driven alias definitions (typed YAML, inheritance) | Wave 3 |
| Numeric priority, descending, ties→distribution→declaration, 0=last resort | Waves 3 (compile) + 4 (select) |
| Proportional routing weights in YAML; account binding in code | Wave 3 (schema) + 4 (distribution) |
| `read` pilot, additive, other aliases untouched | Wave 4 (scoped branch) |
| Per-model TUI fail-closed on unknown TUI | Wave 4 |
| Schedule windows (UTC) + affinity continuation rule | Waves 3 + 4 |
| Immutable snapshot + `config_epoch` | Wave 3 |
| Refresh endpoint, no auth, fail-closed, atomic, secret-safe | Wave 5 |
| Routing decisions NOT recorded (per operator); no DB change | N/A — `session_history` unchanged |
| Open error-class vocabulary + `FailureEvent` | Wave 1 |
| Three-valued origin; only upstream cools; unknown never cools | Waves 1 + 2 |
| Confidence-tiered N-of-M cooldown, scope, duration, half-open | Wave 2 |
| CSV → coverage checklist; fixtures from error-archive | Wave 6 |
| Shadow parity (no production logic change) | Wave 6 |
| Multi-worker consensus | Out of scope — single-worker (R-6) |
| Migrate other 6 alias families; full error-policy enforcement; `aawm-read`→`read` cutover | Out of scope — deferred (Wave 7 spins into follow-up items) |

## Alternatives Considered (Phase 3)

1. **Two separate plans (583 and 584).** Rejected: they share the error-class vocabulary seam, the `read` pilot, and cooldown state; sequencing them apart risks 583 reaching its error-policy need with no taxonomy. One plan with vocabulary as Wave 1 removes the seam risk.
2. **Replace all alias tables with config at once.** Rejected: the selection engine is a 27k-line god-module and the operator mandated a non-destructive pilot; the `read`-only branch proves the approach first.
3. **Persist routing-decision records (new columns or existing metadata) to session_history.** Rejected per operator: routing decisions are not recorded at all beyond `session_history`'s existing as-routed fields — no new columns, no metadata capture, no DDL.

## Self-Critique (Phase 3)

- **The weakest part of this spec is:** Wave 6's shadow-parity assertion. Because the pilot shares cooldown keys and OpenRouter's process-global free quota with the live `aawm-low` lane, a strict parity test can flap on ambient production state. It needs a documented tolerance and a way to neutralize ambient cooldown state, or it produces false failures.
- **The biggest assumption I made is:** that `read.yaml`'s candidate order should mirror `CODEX_AAWM_LOW_CANDIDATES` (`policy.py:227-277`) plus the promo-gated qwen, with priorities assigned descending to match. The order matches the operator's stated list, but the exact priority integers and promo-window boundaries are inferred and want confirmation.
- **The thing most likely to need revision after the first execution attempt is:** Wave 4. Grafting numeric priority + proportional distribution + per-model TUI gating + schedule/affinity onto the existing `last_resort`-tuple selection block (`:5497-5546`) inside a 27k-line module is the highest-uncertainty work; expect a principal diagnosis and at least one re-scope of how the snapshot branch grafts on.

## QA Report — D1-583/D1-584

**Overall verdict: PASS** (QA run 2026-07-22, HEAD `4876f4ce80` on `develop`, read-only verification; one pre-existing, wave-independent test-isolation flake documented under item 7 — not a regression from this change set).

### 1. All new tests pass — PASS
- `./.venv/bin/pytest tests/test_litellm/proxy/aawm_alias_routing/ -q` → **70 passed, 0 failed, 0 errors** (exit 0, 12.26s). Evidence: `/tmp/qa_alias.txt`.
- All 10 test files present and collected (`test_classification.py`, `test_config_compiler.py`, `test_config_schema.py`, `test_cooldown_evidence.py`, `test_failure_fixtures.py`, `test_failure_vocabulary.py`, `test_read_pilot_selection.py`, `test_read_pilot_shadow_parity.py`, `test_refresh_endpoint.py`, `test_smoke_alias_config_framework.py`).

### 2. Tests assert real values (not vacuous mocks) — PASS
Spot-checked; all four files assert exact values a wrong implementation would fail:
- `test_read_pilot_selection.py:72-74` asserts snapshot-only model appears AND ordering differs from `CODEX_AAWM_LOW_CANDIDATES`; `:90-92` exact priority ordering (`ordered[0].model == "openrouter/snapshot-only-model"`, `ordered[-1].priority == 0`); `:127-130` statistical weight split 0.25/0.75 over 4000 seeded trials (±0.05); `:173-174` exact TUI product-label match/mismatch (`"Claude/1.2"` eligible, `"Codex/1.0"` not).
- `test_cooldown_evidence.py:49-54` asserts `is_coolable(...) is False` AND `decision.should_cool is False` for both client-origin (`asyncio.CancelledError`) and unknown-origin events; `:63` single structured 429 cools immediately despite `marker_n=5`; `:70-74` sliding-window expiry with explicit `now_monotonic` values.
- `test_config_compiler.py:63-66` exact compiled candidate order by model name; `:91-93` normalized weights sum to 1.0 with exact per-model values; `:47-48` frozen-snapshot mutation raises; `:54` unknown top-level key rejects compile; `:181` unregistered provider rejects compile.
- `test_refresh_endpoint.py:60-66` exact response contract (`changed is True`, attempted==active hash); `:79` no-op re-post `changed is False` with same hash; `:91-99` invalid YAML → 400/422, raw YAML not echoed, last-known-good hash still active via `config_snapshot.get_active_snapshot()`; `:115-117` in-flight prior-snapshot immutability across swap.
- config_hash stability: `config_hash = sha256(raw_yaml)` (`config_compiler.py:98`); the no-op refresh test (`test_refresh_endpoint.py:69-80`) pins hash stability across identical re-posts.

### 3. Scope containment: `read`-only gating — PASS (critical check)
- Getter: `llm_passthrough_endpoints.py:674-682` — `_get_codex_auto_agent_candidates_for_alias` branches to `_select_read_pilot_snapshot_candidates()` ONLY when `alias_model == _READ_PILOT_ALIAS_NAME` (`"read"`, defined `:561`); all other aliases fall through to `_CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS.get(...)` unchanged. The Anthropic getter (`:685-691`) is entirely untouched by the snapshot path.
- Failure recorder: `llm_passthrough_endpoints.py:6519-6523` — `_record_read_pilot_cooldown_evidence` invoked only under `if alias_model == _READ_PILOT_ALIAS_NAME`; the rest of `_record_auto_agent_alias_attempt_failure` is the pre-existing audit-event path for all aliases.
- `policy.py:292-305` `CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS` contains no `"read"` key — `aawm-read`, `aawm-low`, `aawm-sota*`, `aawm-code`, `aawm-orchestration` all still resolve from the hard-coded tables, enforced by `test_read_pilot_selection.py::test_other_aliases_unchanged` (`:77-83`, exact table-identity assertions).
- Graceful degradation: with no active snapshot (or no `read` alias in it), `_select_read_pilot_snapshot_candidates` (`:645-651`) falls back to the hard-coded table — no raise.

### 4. No new routing-decision recording — PASS
- `test_no_new_routing_decision_recording` (`test_read_pilot_selection.py:229-251`) is a real assertion: it inspects `llm_passthrough_endpoints` source around every `inbound_model_alias` (session_history-write marker) region and asserts `config_hash`/`config_version` never appear there, plus asserts no `_record_read_pilot_routing_decision` / `_persist_routing_snapshot_selection` symbols exist.
- Source confirms: the read-pilot lane adds only in-memory evidence bookkeeping (`_record_read_pilot_cooldown_evidence`, `:6479-6504` — writes to `attempt_record["origin"]` and the process-local gate; explicit docstring: "additional, read-pilot-only evidence bookkeeping"). `config_snapshot.py:66-69` documents epoch/hash/version as "in-memory-only ... Neither is persisted".

### 5. Cooldown gate + three-valued origin — PASS
- Read-pilot failures route through `CooldownEvidenceGate`: module-level `_read_pilot_cooldown_gate` (`llm_passthrough_endpoints.py:562`), fed via `classify_failure(...)` → `gate.record(...)` in `_record_read_pilot_cooldown_evidence` (`:6494-6503`); `attempt_record["origin"] = event.origin` records the three-valued origin.
- `failure_vocabulary.py:101-107`: `is_coolable` returns `event.origin == "upstream"` only. `classification.py:275-276`: gate short-circuits `should_cool=False` for non-coolable events, so `client`/`unknown` never cool and never advance evidence. `FailureEvent.__post_init__` (`failure_vocabulary.py:92-98`) validates origin/confidence/scope against frozen sets.
- Enforced by `test_cooldown_evidence.py::test_only_upstream_origin_events_advance_evidence_toward_cooling` and `test_failure_fixtures.py::test_client_cancelled_asyncio_cancelled_error_is_never_coolable`.

### 6. Refresh endpoint — PASS
- `POST /aawm/alias-config/refresh` (`llm_passthrough_endpoints.py:1167-1224`): no auth dependency on the route; compile failure raises HTTP 400 with a generic message + last-known-good hash only (never echoes YAML/secrets — verified by `test_refresh_endpoint.py::test_invalid_refresh_fails_closed` and `::test_response_omits_secrets`); previous snapshot preserved on failure (last-known-good read happens before any swap; swap only on successful compile); no-op re-post keeps the exact same snapshot object (`:1211-1216`) for in-flight reader stability. Atomic swap is lock-guarded in `RoutingSnapshotHolder.swap` (`config_snapshot.py:94-99`).
- Fail-closed at compile: `config_schema.py` uses `extra="forbid"` on all models and validates provider/route_family against `REGISTERED_PROVIDERS`/`REGISTERED_ROUTE_FAMILIES` (`:71-77`), matching `test_config_compiler.py::test_rejects_arbitrary_behavior_at_compile`.

### 7. Regression on other aliases — PASS (with one pre-existing environmental flake, NOT a wave regression)
- Ran the 7 files referencing `_get_codex_auto_agent_candidates_for_alias` / `CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS` (`test_alibaba_alias_routing.py`, `test_moonshot_alias_routing.py`, `test_llm_pass_through_endpoints.py`, `test_rr054_{llm_passthrough_residuals,package_contracts,package_ownership,structural_extraction}.py`): **1158 passed, 1 failed** — reproducible across two runs (`/tmp/qa_alias_regr.txt`, `/tmp/qa_alias_regr2.txt`).
- The single failure, `test_moonshot_alias_routing.py::test_should_persist_one_kimi_managed_account_lane_and_continue_to_grok` (`assert 0.0 > 0` on `highspeed_state["cooldown_seconds"]`, line 313), **passes solo** (7 passed) and passes when paired with `test_llm_pass_through_endpoints.py` (989 passed). Bisection isolated the contaminator to `test_rr054_package_contracts.py::test_rr054_durable_cache_key_and_expiry_parse`, which calls `durable.configure_durable_runtime(clean_value=...)` (`test_rr054_package_contracts.py:309`) and thereby clears the module-global `_dual_cache_override` for the rest of the pytest session, changing how the moonshot test's `_FakeDurableAliasCache` patch interacts with durable-state reads.
- **Wave-independence proof:** the identical 2-test pair fails the same way on a pristine pre-wave tree (`git archive 07f991137f~1` extracted to scratchpad → `1 failed, 7 passed`). Both test files predate the waves (`878fe31359`, 2026-07-18 < first wave merge 2026-07-22) and the waves changed neither file nor `durable.py`/`memory.py`. This is a pre-existing test-isolation bug in the RR-054 contract test, out of scope for D1-583/D1-584. Recommend a separate ticket: `test_rr054_durable_cache_key_and_expiry_parse` should save/restore the durable runtime globals (or use monkeypatch) instead of calling `configure_durable_runtime` without cleanup.

### 8. Lint / types — PASS
- `ruff check` on all 5 new modules + `state.py` + `llm_passthrough_endpoints.py` + all 10 test files: **clean (no output, exit 0)**.
- `mypy llm_passthrough_endpoints.py` (fresh cache): **15 errors** — identical to the documented 15-error baseline (2 stale-policy `attr-defined` at `:238`, 12 pre-existing `assignment`/`arg-type` at `:15388/:15513/:24540/:24701`). **0 new errors**; none reference the wave modules or helpers.

### Findings (non-blocking)
1. **[Pre-existing, dispatch recommended]** RR-054 contract test leaks durable-runtime global state (see item 7). Not caused by this change set; fails identically pre-wave.
2. **[Observation]** `_select_proportional_snapshot_candidate` and `_is_tui_attached_candidate_eligible`/`_is_snapshot_candidate_in_schedule_window` are wired into `_select_read_pilot_snapshot_candidates` for eligibility/ordering, but the proportional tie-break helper is exercised only via its pure-function tests — end-to-end proportional selection inside a live `read` request is deferred, consistent with the pilot scope.
3. **[Observation]** `read.yaml`'s promo window (`2026-07-01..2026-08-01`) is currently OPEN as of QA date (2026-07-22), so live `read` traffic with an activated snapshot would rank `alibaba_token_plan/qwen3.8-max-preview` first — this is intentional per the shadow-parity test's documented tolerance (`test_read_pilot_shadow_parity.py:119-132`), noting it here for operator awareness.

## Hindsight (self-generated, 2026-07-22)

Execution completed Waves 1–6 in ~6h; QA PASS. ≥5 hindsight items from actual execution evidence:

1. **Team vs non-team dispatch broke worktree isolation (2 wasted dispatches).** First two testers were dispatched as *named* team agents; the team form does not receive frontmatter worktree isolation, so they landed in the main tree and refused to write. Operator caught it. **Lesson (repo-specific):** in this litellm fork session, source-modifying subagents MUST be dispatched non-team (no `name`) so frontmatter isolation provisions the worktree; explicit `isolation:"worktree"` on a team dispatch is not the sanctioned pattern.
2. **Land requires the main checkout on `develop`; this fork defaults to `main`.** The first land preflight-failed ("Main repo is on 'main', expected 'develop'"). **Lesson:** switch the main checkout to `develop` (ff to origin) before dispatching any worktree agents in this fork; promote develop→main at the very end.
3. **DB over-reach from a misread tool error.** I planned a `session_history` migration + DB Foundation gate + Phase 2.5 live-migration gate, having read an `mcppg` MCP "checkout timed out" as "DB down." The DB was up (flaky tool), AND no schema change was needed — existing columns + the operator's "don't record routing decisions" rule covered it. **Lesson:** probe the DB with a real query before concluding it's down; don't reach for new columns when existing telemetry + an explicit operator constraint already cover the need. Removing the DB spine cut the plan from 9 to 7 waves.
4. **The combined Waves 4+5 dispatch was too big → 48 min of investigation, 0 code, then a throttle kill.** Operator caught it. **Lesson:** split god-module integration into single-wave dispatches, hand over the concrete integration map (functions + line numbers) from the failed agent's transcript, and impose a hard read-budget ("read the test + these functions, then STOP reading and WRITE"). The split Wave 4 redispatch landed cleanly with the map+budget.
5. **Provider exhaustion is a redispatch, not a hold.** Engineer B2 died on a 429 (`aawm_anthropic_auto_agent_redispatch_required`); a fresh dispatch auto-routed around the throttled candidate. **Lesson:** on provider exhaustion, redispatch immediately — do not treat the per-candidate cooldown as a blanket pause.
6. **`TaskCreate` premise was wrong twice.** I first called it "removed" (it's the current default, gated off this session by the tool allowlist), then relayed research claiming the *orchestrator* is gated by an agent `.md` allowlist — but the orchestrator doesn't load from an agent file (operator corrected both). **Lesson:** tool absence is a launch/allowlist gate to verify, not a product removal; the orchestrator's toolset is distinct from subagent agent-file allowlists.
7. **QA's open-vocabulary validation paid off.** The 48 unmapped CSV error classes surfaced as *documented known-gaps* rather than failures — exactly what the open-registry design intended — so a real taxonomy-coverage gap is tracked without blocking the pilot.

### Outstanding at close-out (tracked, not silently dropped)
- **Pre-existing test contamination** (`test_rr054_package_contracts.py` leaking durable-runtime globals into `test_moonshot_alias_routing.py`) — proven wave-independent; fix or ticket per operator.
- **Live `:4001` dev validation** — real `read` sessions + refresh endpoint against the running dev proxy; requires deploying `develop` to `litellm-dev`. This is the plan's "development validation" acceptance criterion and is not yet done.
- **Deferred-remainder backlog items** (for Wave 7): migrate the other six alias families to config; full error-policy enforcement (retry/failover by class); `aawm-read`→`read` cutover after validation; multi-worker config consensus if scaled; expand `classify_failure` taxonomy toward the 48 known-gap classes.
