# AAWM alias-routing configuration (CFG-002)

Deterministic, directory-based alias-routing configuration for the AAWM
passthrough proxy. One immutable routing snapshot is built at startup from
YAML files in a canonical directory; the snapshot gates readiness and drives
all alias candidate selection for the worker's lifetime.

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
and blocks readiness. There is no fallback to a partial snapshot or the
static candidate table after a failed directory load.

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
- A failed startup also prevents alias routing from degrading to the legacy
  static candidate table: all alias candidate getters return empty tuples,
  so no traffic is dispatched on a broken config.

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

Each config-defined alias is one logical routing identity (e.g. `read`).
A single alias carries per-candidate `route_family` and
`anthropic_route_family` projections so that Codex ingress and Anthropic
Messages ingress each see only their own route-family view. No cross-provider
fallback is introduced by the alias model.

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
- **Restart**: file additions or removals in the canonical directory take
  effect on the next worker restart (or container recreation). The startup
  loader re-scans the full directory inventory.
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
prevent host-side edits to the bind-mounted source; those are bounded by the
scan-rescan drift check at activation and otherwise load on the next restart.

## Production parity

Production deployment parity (equivalent read-only mounting, startup ordering,
and readiness gating in the production image/orchestrator) remains an
**external blocker** tracked outside this repository. The dev compose mount
and startup wiring here are the reference implementation.

## Multi-worker refresh consensus

Explicitly **out of scope**. Each worker performs its own deterministic
directory load at startup. There is no cross-worker coordination, distributed
lock, or consensus protocol for config activation.

## Legacy aliases

Aliases registered in the static candidate table
(`CODEX_AUTO_AGENT_CANDIDATES_BY_ALIAS`) that are not redefined in the YAML
config continue to resolve from the static table when a snapshot is active.
Arbitrary normalized aliases that appear in neither the config nor the static
table fail closed (empty candidate set) rather than receiving a generic
fallback.
