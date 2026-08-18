# AAWM alias-routing configuration (CFG-002)

Deterministic, directory-based alias-routing configuration for the AAWM
passthrough proxy. One immutable routing snapshot is built at startup from
YAML files in a canonical directory; the snapshot gates readiness and drives
all alias candidate selection for the worker's lifetime. The compiled YAML
snapshot is the sole AAWM alias and candidate authority: there are no built-in
candidate tables and no startup or no-snapshot fallback.

## Canonical directory

```
litellm/proxy/aawm_alias_config/
```

Relative to the repository root (and `/app/litellm/proxy/aawm_alias_config/`
inside the container). The path is resolved from
`litellm/proxy/pass_through_endpoints/aawm_alias_routing/config_startup.py`
at import time; no environment variable override exists.

## Immutable inventory capture

At startup the loader scans the canonical directory into a frozen internal
`_ConfigInventory`. Each regular file is captured as **immutable raw bytes**
plus a content digest, size, and filesystem identity metadata (`st_dev`,
`st_ino`, `st_mtime_ns`, `st_ctime_ns`); each directory records its relative
name, identity metadata, and sorted child names.

The scanner uses **descriptor-anchored, component-by-component `O_NOFOLLOW`
traversal** with pre/post `fstat` consistency checks:

- The root is opened with `O_DIRECTORY | O_NOFOLLOW`; each nested directory
  is opened relative to its parent fd with `O_DIRECTORY | O_NOFOLLOW`; each
  file is opened relative to its parent fd with `O_NOFOLLOW | O_NONBLOCK`.
- Symlinks are detected with `lstat` (via the parent directory fd) and
  rejected before any open, so a symlinked root or ancestor cannot escape the
  canonical tree.
- Each file's `fstat` is taken before and after the bounded read; a change in
  device, inode, mtime, or size during the read fails closed.

Compilation (`_compile_inventory`) operates **exclusively on the captured
bytes** -- never on live filesystem paths. No subset, nondeterministic,
absolute, empty, or `..` inventory input can be injected: relative names are
validated to be non-absolute and free of `..` components before compilation.

## Discovery contract

Discovery is **fail-closed**: any violation marks the alias-config startup
as failed rather than silently skipping a file. The process continues
running; readiness returns 503 and all alias routing fails closed (empty
candidate sets).

- The directory root must not be a symlink.
- Symlinks at any depth are rejected regardless of suffix.
- Non-regular files (FIFOs, sockets, devices) are rejected.
- Directories whose name carries a YAML-like suffix (any case) are rejected.
- Unreadable nested directories are rejected.
- Only **lowercase** `.yaml` / `.yml` regular files are accepted.
  Case-variant suffixes (`.YAML`, `.Yml`) are rejected.
- `__init__.py` is Python package infrastructure and is silently ignored;
  it is not an alias config file.
- Any other regular file is rejected (fail closed).
- Files are ordered by POSIX-sorted relative path for deterministic merge.

## Startup lifecycle

1. `activate_alias_config_directory()` runs once per worker during proxy
   startup, before the lifespan yields (before readiness).
2. The directory is scanned into an immutable `_ConfigInventory` (raw bytes
   captured, identity metadata recorded).
3. The captured bytes are parsed, schema-validated
   (`config_schema.RoutingConfigDocument`), checked for cross-file duplicate
   aliases (case-insensitive), and merged into a single document with
   unambiguous defaults, then compiled into one **immutable**
   `RoutingSnapshot` via `config_compiler.compile_yaml`.
4. The complete tree is scanned **again immediately before activation** and
   the two inventories are compared for exact equality. Any drift
   (add/remove/edit/replace-file/replace-root/replace-nested-directory)
   raises `InventoryDriftError` and fails closed.
5. On exact match, the snapshot is atomically installed as the process-local
   active routing snapshot.

Any failure at steps 2-5 clears any prior snapshot, marks startup as failed,
and blocks readiness. There is no fallback to a partial snapshot, built-in
candidate table, or other non-YAML authority after a failed directory load.

### Drift detection is not a filesystem transaction

The scan-compile-rescan-swap sequence detects **bounded-load** changes that
land between the first scan and the final revalidation. It is **not** a
filesystem transaction. Changes that land **after** the final revalidation
are not observed by this activation and load on the **next restart** (or the
next successful refresh). The read-only container mount (below) is the
primary guarantee against in-container mutation; the rescan is a bounded
defense against host-side races on the bind-mounted source.

## Readiness and failure behavior

- `/health/readiness` requires alias-config startup `state=active`. Both a
  **failed** startup and a **not_loaded** startup (never attempted) return
  **503**. The response body includes a sanitized `aawm_alias_config` status
  object.
- When startup succeeds, the same status object is embedded in the 200
  readiness response for observability.
- A failed startup also prevents alias routing from degrading to any
  non-snapshot candidate source: all alias candidate getters return empty
  tuples, so no traffic is dispatched without an active compiled snapshot.

## Secret-safe status and logging

The startup status object exposes only: `state`, `config_hash`,
`config_version`, `config_epoch`, relative file names, alias names, alias
count, and `activation_result`. On failure it exposes only the error **class
name**.

Success logging includes the relative source file names, active aliases, the
full semantic config hash/version, and the activation result.

Raw validation errors, YAML content, secrets, and absolute paths never
appear in status responses or structured log output.

## Alias model and ingress route families

Each config-defined alias is one logical routing identity (e.g. `basic`).
A single alias carries per-candidate `route_family` and
`anthropic_route_family` projections so that Codex ingress and Anthropic
Messages ingress each see only their own route-family view. No cross-provider
fallback is introduced by the alias model.

Every alias compiled from YAML is an ordinary exact-name route and a valid
`alias_reference` or dispatch target. There is no public/internal or
visibility routing distinction in YAML or Python. Codex and Claude TUI
selection catalogs are separate client model-definition lists; they are not
generated from every snapshot alias and do not invent a denylist.

## Maintained `basic` alias behavior (CFG-008)

The maintained alias name is `basic`. Every request uses this common candidate
order:

1. OpenRouter Cohere North Mini Code free
2. OpenRouter Owl Alpha
3. OpenCode Zen deepseek-v4-flash
4. OpenCode Zen big-pickle
5. Alibaba Token Plan deepseek-v4-flash-0731
6. Alibaba Token Plan qwen3.6-flash

The managed `sota-xai` candidate has no candidate-level reasoning override:
caller `reasoning.effort=xhigh` is sent unchanged to `oa_xai/grok-4.6`, with
requested/native effort metadata and the provider-native field/provider
recorded. Its route rollup is `grok-4.6(sota-xai):xhigh`.

For this managed `oa_xai`/`sota-xai` flow, Grok 4.6 collaboration namespace
children are flattened into provider-bound function tools and sanitized before
egress. Provider function calls and results are restored to stock Codex
collaboration tool calls and results with continuation IDs preserved. Native
Grok remains a separate, unchanged flow.

Managed `oa_xai` Responses preparation applies the existing catalog/model-
metadata-driven `rewrite_input_item_types` continuation rewrite before xAI
request sanitization. The rewrite runs only when model metadata requests those
input-item types; it does not use hardcoded model-name checks. This describes
managed `oa_xai` behavior only and does not claim runtime or container
acceptance.

The final candidate is mutually exclusive by originating TUI:

- Claude-origin requests use native Anthropic Haiku only.
- Codex, non-Claude, missing, and unknown origins use OpenAI
  `gpt-5.6-luna` only.

Luna's YAML `reasoning_effort: low` is authoritative through the generic
CFG-006 pipeline and replaces caller-provided reasoning. Haiku remains
Anthropic-native; Luna uses the intended OpenAI/Codex OAuth route. The removed
candidates are the `qwen3.8-max-preview` promo, `kimi-for-coding`, and
`gpt-5.4-mini`.

Cost metadata for these shared candidates is route-specific. OpenRouter North
may carry a third-party hosted reference baseline, while Owl Alpha is
unpriced. OpenCode Big Pickle and its free DeepSeek route remain unpriced, and
Alibaba DeepSeek V4 Flash and Qwen 3.6 are reference-priced only under the
subscription `actual_invoice_cost_known=false` contract, using the
international Model Studio direct list rate rather than claiming Token Plan
subscription invoice economics. Reference totals are provenance metadata and
never standard spend or `response_cost_usd`; Luna remains the last-resort
actual routed fallback.

## Maintained `work` and `work-other` alias behavior

The `work` alias is compiled from `work.yaml`. Candidate order is:

1. OpenAI `gpt-5.3-codex-spark`
2. Nested `alias_reference: work-other`
3. Claude-origin only: native Anthropic Sonnet tail
4. OpenAI `gpt-5.6-luna` last resort

`work-other` is an ordinary configured alias compiled from `work-other.yaml`.
It is a valid exact-name route and a valid `alias_reference` target. It remains
absent from Codex and Claude TUI selection only because those clients' explicit
model-definition inclusion lists omit it, not because YAML or Python marks it
internal.

## Maintained `expert` alias behavior (CFG-013)

The `expert` alias is compiled from `expert.yaml` in the canonical directory.
It has exactly two candidates:

1. **Claude-origin requests only** (`tui_attached: Claude`): native Anthropic
   `claude-opus-5`, highest priority. Canonical Opus 5 is inherently a
   1M-context model, so there is no `claude-opus-5[1m]` selector; a second
   selector would duplicate the same upstream model.
2. **Universal last resort** (`priority: 0`): OpenAI/Codex `gpt-5.6-terra`.
   Terra carries no `tui_excluded` gate, so it is the direct/default candidate
   for Codex, non-Claude, missing, unknown, and otherwise unconfigured
   origins, and it remains available to Claude as the fallback after an Opus
   failure.

Both candidates carry authoritative `reasoning_effort: max`. This value
replaces caller-provided reasoning through the shared CFG-006 candidate
pipeline; there is no expert-specific reasoning precedence.

The provider-native credential boundary applies: Opus uses
`anthropic_messages` on both ingress projections and must egress exclusively
through Anthropic-native credentials. The Codex ingress excludes
`anthropic_messages` candidates, so Opus never routes through Codex/OpenAI
credentials; Terra uses `codex_responses` (projected to
`anthropic_openai_responses_adapter` on Anthropic ingress) and keeps its
OpenAI/Codex credential domain on both ingresses.

## Provider-native credential boundary

Anthropic/Claude model traffic must egress exclusively through
Anthropic-native provider routes with credentials accepted for that native
route. If the native Anthropic route or credential is unavailable, the
system fails closed with an explicit routing/authentication error.

Cross-provider egress (Codex OAuth, ChatGPT backend-api, OpenAI adapters)
for a resolved Anthropic/Claude upstream model is prohibited and treated as
a terms-of-service violation. This rule applies to normal routing, fallbacks,
retries, cooldown recovery, probes, and acceptance harnesses.

## Restart, refresh, and recovery semantics

- **Routing snapshot lifetime**: once activated, the routing snapshot lasts
  until a **successful refresh** replaces it or the worker **restarts**. A
  failed refresh preserves the last-known-good snapshot.
- **Restart**: the startup loader re-scans and compiles the complete canonical
  directory. A successful empty-body/default refresh performs the same full
  re-scan and compile, so host-side file additions, removals, and edits can
  take effect without a restart. Recovery from a failed startup still requires
  a worker restart.
- **Invalid hot refresh**: if a runtime refresh attempt encounters invalid
  config, the active last-known-good snapshot is preserved. No restart is
  needed; routing continues on the prior valid definition.
- **Semantically unchanged refresh**: reports `unchanged` and preserves the
  same active definition (same `config_hash`, same `config_epoch`).
- **Recovery after failure**: requires a config fix followed by a worker
  restart. A failed startup clears the snapshot; there is no automatic retry.

## Read-only container mount

`docker-compose.dev.yml` bind-mounts the canonical directory **read-only**
(`./litellm/proxy/aawm_alias_config:/app/litellm/proxy/aawm_alias_config:ro`).
This blocks **container-side** writes to the config tree. It does **not**
prevent host-side edits to the bind-mounted source. A successful
empty-body/default refresh re-scans and compiles the complete canonical
directory, including those host-side additions, removals, and edits; otherwise
they load on the next restart.

## CFG-017 source identity and parity contract

For dev parity, `litellm-dev` must load alias-config only from the canonical
`aawm_alias_config` directory in the current checked-out workspace. The
compose mount source (`./litellm/proxy/aawm_alias_config`) must therefore be the
same canonicalized source for every startup and refresh attempt.

Contract-level parity checks for operators are:

1. `docker-compose.dev.yml` contains exactly the mount
   `./litellm/proxy/aawm_alias_config:/app/litellm/proxy/aawm_alias_config:ro`.
2. The mount source resolves to the current repo checkout path
   (`$compose_dir/litellm/proxy/aawm_alias_config`) and no alternate source
   path is used.
3. The preflight inventory path used by startup is the mounted destination
   `/app/litellm/proxy/aawm_alias_config` (no other alias-config source is
   consulted).

The later runtime preflight should compare the host source mount identity
(`realpath` of `compose_dir/litellm/proxy/aawm_alias_config`) against the
expected checkout path and compare startup inventory identity metadata from the
two-pass scan (`root_st_dev`, `root_st_ino`, `root_st_mtime_ns`,
`root_st_ctime_ns`) to guarantee the mounted source did not change before
activation.

## Production parity

`litellm-dev` is the implementation and acceptance environment. Production
parity is enforced during the release process; see [PROD_RELEASE.md](../PROD_RELEASE.md).
The dev compose mount and startup wiring here are the reference
implementation.

## Cooldown-clear operator contract (CFG-004)

The implemented operator ingress for clearing alias-routing cooldown state is:

```
POST /aawm/alias-routing/cooldowns/clear
```

This is the only supported clear path. There is no GET, wildcard/global clear,
or supported Redis/DB side channel.

### Authentication and topology

The request must pass both the authenticated proxy path and explicit checks:

- the authenticated token has `PROXY_ADMIN`;
- the token is the configured LiteLLM master key, compared by safe hash;
- `AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER=1` is set literally.

Missing or malformed delegated auth, a missing master key, a non-master token,
or a closed topology gate fails closed. The gate is single-worker only; this
endpoint does not claim multi-worker support. `docker-compose.dev.yml` wires
the flag for the single-worker `litellm-dev` environment and labels that
environment `litellm-dev`.

### Strict request and resolution

The JSON object accepts only string fields `ingress`, `alias`, `provider`, and
`model`. `ingress` is required and must be `codex` or `anthropic`. The target
is exclusive: use `alias`, or use both `provider` and `model`. Combined,
partial, empty, extra, or non-string fields are rejected. Raw keys, hashes,
namespaces, patterns, global-clear fields, and internal identifiers are
forbidden.

Resolution uses only the active routing snapshot. Unknown targets are
rejected. Exact provider/model matches resolving to multiple distinct
identities or route-family projections are ambiguous and rejected. Codex
uses `route_family`; Anthropic uses `anthropic_route_family` and fails closed
if that projection is unavailable.

### Clear semantics and response

After snapshot resolution, each identity is reserved before local or durable
inspection. The local lane index and authoritative durable identity membership
are then unioned, in-flight publication intents are drained with a bounded
wait, and prior state is inspected. Mutation uses sorted per-key barriers,
the family mutation lock, then sorted probe locks. Durable state is removed by
an atomic compare-and-clear transaction. Local cooldown-derived state and
lane-index entries are invalidated, followed by strict durable, identity,
local-state, and local-index postcondition checks.

Responses include `result`, `family`, `target_description`, `ingress`,
resolved provider/model/route-family `candidates`, `keys_cleared`,
`members_removed`, `affinity_preserved`, `prior_state_source`,
`bounded_remaining_ttl_seconds`, `environment`, `namespace`, and
`timestamp_utc`. `result` is `cleared` or idempotent `not_active`; the latter
is returned only after authoritative absence proof. Missing cache, Redis
errors, transaction uncertainty, membership drift, publication-drain timeout,
or failed postconditions fail closed. Each HTTP exit emits one sanitized audit
event with target, candidates, ingress, result, prior source/TTL,
environment, namespace, and bounded error code. Secrets, credentials, raw
keys, hashes, and traceback locals are excluded.

There is no dry-run parameter or inspection-only endpoint. Clearing stale
state is safe after quota replenishment but does not replenish provider quota,
budget, or rate limits; an upstream that remains exhausted can recreate
cooldown state. The endpoint performs no provider traffic and does not restart
workers.

### Operator procedure

There is no dry-run parameter or inspection-only cooldown endpoint. The clear
request mutates local state and durable state; do not use it as a probe.

Perform this non-mutating preflight first:

1. Confirm the target environment is the single-worker `litellm-dev` proxy and
   that `AAWM_ALIAS_ROUTING_COOLDOWN_CLEAR_SINGLE_WORKER=1` is configured.
2. Check the implemented readiness surface:

```bash
curl -sS http://127.0.0.1:4001/health/readiness
```

Require HTTP 200 and inspect `aawm_alias_config.state=active`, the expected
`environment` outside the response if your deployment labels it, and the
snapshot's `config_hash`, `config_version`, and `aliases`. The requested alias
must be listed there. Readiness does not expose cooldown membership or TTL.
There is no supported remote Redis inspection command or raw-key/hash
procedure for CFG-004.
3. For an exact provider/model request, inspect the same active configuration
   source that produced the snapshot:

```bash
rg -n -C 2 'alias:|provider:|model:|route_family:|anthropic_route_family:' \
  litellm/proxy/aawm_alias_config/
```

Use only the provider and model values in that schema, and use the matching
`route_family` for `codex` or `anthropic_route_family` for `anthropic`. Do not
infer a target from a Redis key. Confirm the upstream quota issue has been
addressed before clearing. The source inspection cannot prove current
cooldown membership or TTL; those are available only in the clear operation's
sanitized result.
4. After the preflight evidence is recorded, send the master key with one
   strict request:

```bash
curl -sS http://127.0.0.1:4001/aawm/alias-routing/cooldowns/clear \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"alias":"basic","ingress":"codex"}'
```

An exact target uses the actual snapshot identity:

```json
{"provider":"openai","model":"gpt-5.6-luna","ingress":"codex"}
```

5. Require `cleared` or `not_active`, inspect the returned candidates, source,
   TTL, environment, and namespace, and correlate the sanitized audit event.
   Do not treat a 409/503 as a successful clear.

## Failure-action shadow mode (D1-586)

Alias routing classifies failures into the open `FailureEvent` vocabulary and,
separately, maps each class to a configurable action (`observe`, `retry_same`,
`failover`, `cooldown`, `terminal`, `redispatch`). The first delivery emits only
a deterministic **shadow** decision on retryable attempt records as
`shadow_failure_action` so operators can compare current cooldown/retry behavior
with the mapped action. Enforcement of class-keyed retry/failover remains
**disabled**; client and unknown-origin events stay non-cooling and
non-retryable regardless of the mapped action.

## Multi-worker refresh consensus

Explicitly **out of scope**. Each worker performs its own deterministic
directory load at startup. There is no cross-worker coordination, distributed
lock, or consensus protocol for config activation.

## Snapshot-only alias authority

AAWM alias names and candidate sets come only from the active compiled YAML
snapshot. There is no built-in candidate table and no startup or no-snapshot
fallback. Missing, failed, or inactive config fails closed with empty candidate
sets and blocked readiness.

A name that is absent from the snapshot is an ordinary unknown model identity
and follows the same generic snapshot lookup failure as every other absent
name. No identity-specific recognition, rejection, redirect, or compatibility
path applies.

## Cohere source identity (COHERE-001)

The AAWM direct Codex Cohere alias route is a **direct, native provider lane**,
not routed through OpenRouter or any other proxy egress. It uses the Cohere API
with `https://api.cohere.com/v2/chat` as the default endpoint. The effective
endpoint may be validated as native Chat V2 on `api.cohere.com` or
`api.cohere.ai`. The provider/lane identity is `cohere` / `cohere_native`.
Codex-shaped ingress selects the `codex_cohere_chat_completions_adapter` route
family. This is a Codex adapter route to Cohere's native API, not an OpenRouter
route.

### Provider identity

- The direct Codex Cohere alias route must originate from the Cohere-native
  provider route. Cross-provider routes (e.g. through OpenRouter, Codex OAuth
  adapters, or ChatGPT backend-api) are not direct Cohere alias traffic.
- Proven selection/attempt metadata is limited to the alias, provider, model,
  route, and lane. The route-rollup endpoint label may also identify the
  endpoint. Aliases that include Cohere candidates set the exact `model` string
  and `provider: cohere` in YAML; the selected route family is
  `codex_cohere_chat_completions_adapter` and the lane is `cohere_native`.
- Direct Cohere is separate from OpenRouter. An OpenRouter candidate such as
  `openrouter/cohere/north-mini-code:free` remains an OpenRouter request and must not be
  recorded or interpreted as direct Cohere-native traffic.

For COHERE-002 usage observations, the source identity is limited to accepted
direct Codex Cohere terminal HTTP 200 `/v2/chat` calls with
`provider=cohere` and `lane=cohere_native`. Calls are counted once by stable
`litellm_call_id` and recorded as `source=locally_counted`; OpenRouter remains
separate. Anthropic adapter integration and Anthropic testing or acceptance are
not part of this Cohere contract. Migration, deployment, and authenticated
acceptance remain separately authorized.

### Dev compatibility key

- Canonical runtime credential environment variable: **`COHERE_API_KEY`**.
- The legacy **`COHERE_KEY`** is retained only as a dev-time compatibility
  fallback. It does not appear in production deployments. Do not introduce new
  references to `COHERE_KEY`; prefer `COHERE_API_KEY` consistently across new
  code, configs, and documentation.
