# AAWM OAuth Credential Maintenance

## MS-030/MS-031: managed Kimi contract boundary

The authoritative native client for this contract is the WSL installation at
`/home/zepfu/.kimi-code/bin/kimi`, version `0.29.1`. Older Thoth state is not
authoritative. `MS-032`, the Thoth daily updater, is paused by operator
decision and is not a prerequisite for this contract. Do not resume that
updater or publish a Thoth-derived descriptor without separate authorization.

All AAWM Moonshot traffic is OAuth-only through `kimi_code/*` and the Kimi Code
`/coding/v1` upstream; no AAWM route uses a Moonshot API key. Generic upstream
LiteLLM `moonshot/*` support remains unchanged caller-supplied API-key
functionality. It is not converted to OAuth, governed by this descriptor, or
selected as an AAWM route or fallback.

Managed chat, the Codex and Claude adapters, the raw gateway, and usage
polling consume one sanitized native contract descriptor. It contains the
exact endpoint `https://api.kimi.com/coding/v1`, native client name/version,
the native `User-Agent` and header names, service-owned non-personal identity
values, issued/expiry timestamps, and an integrity digest. It contains no
bearer or refresh token and no workstation hostname, hardware model, kernel
release, personal device identifier, account, session, or caller identity.

Managed model allowlisting remains in existing `kimi_code` provider
metadata/config and is not part of this descriptor. Native usage/chat
correlation remains polling and persistence behavior outside the descriptor.

The descriptor is authoritative and caller-spoof rejection is mandatory.
Validate and publish replacements atomically so readers see a new snapshot
without a restart. The host descriptor directory is configured by
`AAWM_KIMI_NATIVE_DESCRIPTOR_DIR` and mounted read-only at
`/app/kimi-descriptor`; consumers read
`/app/kimi-descriptor/kimi-native-contract.json`. The image contains the
resolver, not a captured descriptor or OAuth credential.

The standalone image default
`LITELLM_KIMI_NATIVE_CONTRACT_REQUIRED=false` is compatibility mode and does
not establish exact-parity acceptance. Managed dev Compose defaults the gate
to `true`, as must any fail-closed managed deployment. In required mode, a
missing, stale, malformed, digest-invalid, endpoint-mismatched,
version-incoherent, or hostile descriptor fails closed. Do not use a guessed
contract, alternate endpoint, Moonshot API key, generic `moonshot/*` route, or
another provider to recover.

This document is the operator-facing maintenance guide for AAWM OAuth and OIDC
credential files used by LiteLLM and the provider-status sidecar. It covers the
shared publish path introduced and hardened under RR-065, RR-074, RR-075, and
RR-092.

Related deeper context:

- `docs/aawm-session-history.md` (per-provider route ownership and telemetry)
- `docs/aawm-provider-status-observations.md` (sidecar probe and auth observation loop)

## Scope

| Family | Writer | Typical consumer | Default portable auth path |
| --- | --- | --- | --- |
| Codex / ChatGPT OAuth | Provider-status sidecar, using `scripts/codex_oauth_refresh.py` once per enabled inventory record | LiteLLM Codex adapter routes | Explicit `LITELLM_CODEX_OAUTH_INVENTORY`; managed dev enrolls `~/.codex/oauth.account1.json` and `~/.codex/oauth.account2.json` |
| Managed xAI OAuth (`oa_xai/*`) | `scripts/xai_oauth_refresh.py` (sidecar) | LiteLLM managed xAI OAuth routes | `~/.litellm/xai/oauth-auth.json` |
| Grok native OIDC | `scripts/grok_oidc_refresh.py` (sidecar) | LiteLLM Grok native routes | Caller-supplied configured path |
| Kimi Code CLI OAuth (`kimi_code`) | Existing Kimi Code CLI grant; sidecar refresh only when enabled | Configured LiteLLM Kimi Code consumers | `~/.kimi-code/credentials/kimi-code.json` |

LiteLLM is a **read-only consumer** of these files during request handling. It
selects a still-valid access token (or fails the candidate with a clear
refresh-required message). It must not refresh, seed, or rewrite these
credentials on the request path.

Kimi Code uses the existing host Kimi CLI credential in place. It is not a
LiteLLM-owned second grant. A configured managed `kimi_code` route consumes the
same credential read-only; possessing the file or naming an alias does not
enable routing or transport by itself.

## Portable default paths

Refresh scripts with built-in auth-file defaults and the in-package xAI OAuth
helpers use **portable `~`-relative defaults** expanded with
`Path.expanduser()` at use sites. Grok OIDC requires the caller to provide its
credential path. Defaults must not hardcode a specific operator home directory.

| Credential | Auth file default | Lock file default |
| --- | --- | --- |
| Codex standalone one-file refresh primitive | `~/.codex/auth.json` | `~/.codex/auth.json.lock` |
| Managed xAI OAuth | `~/.litellm/xai/oauth-auth.json` | `~/.litellm/xai/oauth-auth.json.lock` |
| Grok OIDC | Caller-supplied configured path | same directory, `.lock` sibling when configured |
| Kimi Code CLI OAuth | `~/.kimi-code/credentials/kimi-code.json` | `~/.kimi-code/oauth/kimi-code` (native `proper-lockfile` creates the transient `kimi-code.lock` directory) |

Override paths with the normal env vars for the family in use (for example
`AAWM_CODEX_AUTH_FILE` / `AAWM_CODEX_LOCK_FILE` only for the standalone Codex
one-file primitive, `AAWM_XAI_OAUTH_AUTH_FILE` /
`LITELLM_XAI_OAUTH_AUTH_FILE`,
`AAWM_KIMI_OAUTH_AUTH_FILE` / `LITELLM_KIMI_OAUTH_AUTH_FILE`,
`LITELLM_XAI_GROK_AUTH_FILE`,
variants). Compose may bind the expanded host path into containers; the script
defaults themselves remain `~`-relative so other operators and hosts work
without patching source.

`LITELLM_CODEX_AUTH_FILE` is no longer a managed proxy enrollment surface.
Managed Codex proxy consumers require the explicit inventory below and do not
fall back to `~/.codex/auth.json`, directory scans, backup files, path globs, or
`api.openai.com`.

## Codex ordered account inventory (OPENAI-001)

`LITELLM_CODEX_OAUTH_INVENTORY` is a versioned JSON object whose `accounts`
array explicitly enrolls each credential. Managed development Compose supplies
the same inventory to the proxy and provider-status sidecar:

| Label | Auth path | Independent lock path | Priority | Weight | Enabled | Models |
| --- | --- | --- | --- | --- | --- | --- |
| `account1` | `/home/zepfu/.codex/oauth.account1.json` | `/home/zepfu/.codex/oauth.account1.json.lock` | `10` | `1.0` | `AAWM_CODEX_OAUTH_ACCOUNT1_ENABLED` (default `true`) | `["*"]` |
| `account2` | `/home/zepfu/.codex/oauth.account2.json` | `/home/zepfu/.codex/oauth.account2.json.lock` | `20` | `1.0` | `AAWM_CODEX_OAUTH_ACCOUNT2_ENABLED` (default `true`) | `["*"]` |

Selection order is priority first and declaration order second. Labels are
stable, non-secret operator names; they are not upstream account identities.
Configured labels differ from hashed upstream identities and must not be used
interchangeably.
Each record separately pins a 12-character lowercase SHA-256 prefix derived
from the raw upstream ChatGPT account ID. The required values are supplied
outside tracked configuration:

```bash
export AAWM_CODEX_OAUTH_ACCOUNT1_EXPECTED_HASH='<12 lowercase hex>'
export AAWM_CODEX_OAUTH_ACCOUNT2_EXPECTED_HASH='<12 lowercase hex>'
```

Generate each value from an operator-known raw account ID without echoing the
input:

```bash
./.venv/bin/python -c 'import getpass, hashlib; value=getpass.getpass("Raw ChatGPT account ID (hidden): ").strip(); assert value; print(hashlib.sha256(value.encode()).hexdigest()[:12])'
```

Do not put raw account IDs, token fields, credential contents, or invented hash
values in tracked config. Duplicate labels, auth paths, lock paths, or expected
account hashes fail closed. A file whose actual account identity no longer
matches its record's expected hash also fails closed without redefining the
label.

### Enrollment and removal

1. Publish each credential at its configured auth path as a regular file with
   mode `0600`; do not use a symlink.
2. Give each record its own sibling lock path. Never share a lock between
   records.
3. Generate the expected hash from the known upstream account ID using hidden
   input, then supply both required hash env values.
4. Render the Compose config and confirm both records, their order, paths,
   hashes, enable flags, and model eligibility before recreating consumers.
5. To stage removal, set only that record's `AAWM_CODEX_OAUTH_ACCOUNT*_ENABLED`
   value to `false` and redeploy. Remove the record from the explicit inventory
   before archiving its file. Unlisted files and backups are never enrolled.

The managed proxy mounts `/home/zepfu/.codex` read-only. The provider-status
sidecar mounts the same parent directory read-write so its lock and atomic
`os.replace` publication remain visible to readers. Keep directory mounts
rather than individual file mounts; bind-mounting one file can hide later
inode replacements. Any production proxy/sidecar deployment must preserve the
same read-only consumer and single-writer directory boundary.

### Rotation and rollback

- Normal token refresh stays under the record's own lock and publishes back to
  the same auth path. The stable label and expected account hash do not change.
- For an intentional upstream identity replacement, disable that label first,
  publish the replacement to that label's path, update only its externally
  supplied expected hash, verify the rendered inventory, and then re-enable it.
- Roll back by restoring the prior same-label credential and its prior expected
  hash together. Never swap `account1` and `account2` files or hashes to recover
  from a failed rotation.
- An identity mismatch is a deployment or rotation error, not a reason to
  accept the new identity automatically.

The provider-status loop schedules refresh, passive health, and native
reset-credit/quota polling independently for every enabled inventory record.
Each label has separate timers and its configured lock path. A failure for one
record does not suppress another, and successful skipped/no-op refreshes remain
usable. Aggregate health is degraded while at least one record remains usable
and terminal when none do. Sidecar events and observations use only the
configured label and expected safe hash; they do not emit raw paths, account
IDs, or tokens. Account-aware request routing remains OPENAI-004 scope.

## Shared atomic 0600 publication

All credential writers share the same private publish pipeline under
`litellm/secret_managers/`:

| Module | Responsibility |
| --- | --- |
| `credential_file_lock.py` | Nonblocking advisory `fcntl` flock; lock unavailability or contention fails closed |
| `credential_file_metadata.py` | Snapshot / resolve / apply uid, gid, mode |
| `credential_file_write.py` | Private temp create + atomic publish |
| `credential_error_sanitizer.py` | Secret-value redaction for error summaries |

Preferred one-shot write path is `write_and_publish_private_text()`:

1. Refuse if the final credential path is a symlink.
2. Create an exclusive same-directory temp with unpredictable name
   (`O_EXCL`, `O_NOFOLLOW` when available).
3. Create the temp with private mode at open time (**no umask window**); default
   mode is `0600`.
4. Apply ownership/mode metadata to the temp without following links.
5. Atomically `os.replace` the temp onto the final path.
6. Best-effort unlink the temp on any failure.

Group/other permission bits are always clamped back to private `0600`. A prior
file that was left group- or world-readable is corrected on the next successful
publish or metadata repair cycle rather than perpetuated.

Matching lock files sit beside the auth files. Writers acquire the advisory
lock with `LOCK_EX | LOCK_NB` and hold it for read/refresh/write, including
metadata repair on skipped refresh cycles where the script supports that
(Codex and Grok repair paths). Missing locking support, lock-file open failure,
or contention raises `CredentialFileLockError`; the writer performs no
unlocked credential mutation and does not wait indefinitely. The Codex,
managed xAI, and Grok refresh APIs catch that acquisition error at their
existing result boundary and return a sanitized failed refresh summary, so
sidecar and standalone callers retain the normal structured failure contract.

## Metadata env overrides

Publication preserves existing file ownership and private mode unless optional
env overrides are set. Each family uses the same shape:

| Purpose | Codex | Managed xAI | Grok OIDC | Kimi Code CLI |
| --- | --- | --- | --- | --- | --- |
| UID | `AAWM_CODEX_AUTH_FILE_UID` | `AAWM_XAI_OAUTH_AUTH_FILE_UID` | `AAWM_GROK_OIDC_AUTH_FILE_UID` | `AAWM_KIMI_OAUTH_AUTH_FILE_UID` |
| GID | `AAWM_CODEX_AUTH_FILE_GID` | `AAWM_XAI_OAUTH_AUTH_FILE_GID` | `AAWM_GROK_OIDC_AUTH_FILE_GID` | `AAWM_KIMI_OAUTH_AUTH_FILE_GID` |
| Mode | `AAWM_CODEX_AUTH_FILE_MODE` | `AAWM_XAI_OAUTH_AUTH_FILE_MODE` | `AAWM_GROK_OIDC_AUTH_FILE_MODE` | `AAWM_KIMI_OAUTH_AUTH_FILE_MODE` |

Rules:

- Values are optional non-negative integers (`0o600`-style literals accepted).
- Mode overrides that include group/other bits are rejected and fall back to
  `0600`.
- When overrides are unset, the writer snapshots the current file (via `lstat`)
  and re-applies that ownership/mode after refresh.
- Dev compose commonly sets host-user uid/gid and `0o600` so a previous
  container-owned `nobody:nogroup` credential is repaired on the next sidecar
  cycle without giving the LiteLLM container write access.

## Symlink refusal

Credential paths must be regular files, not symlinks. Shared helpers refuse
symlink targets at every sensitive step:

- snapshot (`lstat`, optional hard refuse)
- exclusive temp create (`O_NOFOLLOW` when available)
- metadata apply (`chown`/`chmod` without following links; `lchmod` preferred)
- final publish before and after `os.replace`

A symlink final path raises `CredentialPathIsSymlinkError` (for example
`Refusing symlink credential target: …`). Operators must point env vars at the
real credential file, not at a link that could redirect writes into an
unexpected location.

## Redacted errors (500 characters)

Refresh summaries, sidecar logs, and `provider_auth_observations` rows use
shared value redaction via `sanitize_credential_error_message()` with a default
**500-character** limit.

Behavior:

- Redacts secret *values* for known fields (`access_token`, `refresh_token`,
  `id_token`, `client_secret`, `key`), not merely the field-name labels.
- Handles bare `key=value` / `key: value`, quoted values, JSON forms, and
  query/form boundaries.
- Optionally redacts scoped `Authorization: Bearer …` credentials.
- Truncates the sanitized text to at most 500 characters (`...` suffix when
  truncated).

Rows and summaries must never include access tokens, refresh tokens, raw
auth-file contents, or raw auth-file path material beyond what operators already
configured.

## Kimi Code CLI credential ownership

The shared Kimi Code CLI credential and native lock target are:

```text
~/.kimi-code/credentials/kimi-code.json
~/.kimi-code/oauth/kimi-code
```

Use that same existing JSON in place. Do not copy it into a LiteLLM directory,
symlink it, or create another grant. The Kimi CLI's native `proper-lockfile`
lock for the `oauth/kimi-code` target is the transient sibling directory
`oauth/kimi-code.lock`; do not bind-mount that transient directory directly.

Dev compose has a strict writer/consumer split:

- `litellm-dev` bind-mounts only
  `~/.kimi-code/credentials/kimi-code.json` read-only. A configured Kimi Code
  request worker consumes that shared file and picks up later replacements on
  subsequent requests without a container restart.
- `provider-status-observations` receives read-write access only to
  `~/.kimi-code/credentials` and `~/.kimi-code/oauth`. The latter is required
  for the native `oauth/kimi-code` lock target and its `kimi-code.lock`
  directory; the sidecar does not receive the broader `~/.kimi-code` tree.
- The standalone sidecar CLI keeps scheduled Kimi refresh disabled unless
  `AAWM_KIMI_OAUTH_REFRESH_ENABLED=1` is set. Development Compose enables it
  by default and checks every 300 seconds because Kimi Code access tokens are
  short-lived. It refreshes the existing CLI grant in place under the native
  lock; it does not create or copy a credential. Override
  `AAWM_KIMI_OAUTH_AUTH_FILE`, `AAWM_KIMI_OAUTH_LOCK_FILE`, interval, timeout,
  or uid/gid/mode only when the deployment requires different host paths.

The compose contract controls credential ownership and hot-reload visibility;
it does not enable a Kimi route by itself. Managed-route behavior, exact model
IDs, and `/models` capability gating are documented in
[`moonshot.md`](my-website/docs/providers/moonshot.md#managed-kimi-code-oauth-aawm).
Kimi Code `0.29.1` derives a separate OAuth storage slot for a custom
`KIMI_CODE_BASE_URL`, so the CLI cannot currently use the local gateway while
retaining this exact default credential file. Do not work around that client
limitation by copying or symlinking the credential or enrolling another grant.
Any production-equivalent deployment must preserve the same read-only worker
and single-writer sidecar contract, but production mutation remains a separate
operator-authorized rollout.

## Historical credential records

Older deployment notes may mention Antigravity credential files or refresh
helpers. Those references are retained only to interpret historical records;
they do not describe a current LiteLLM route, package, credential, or
maintenance contract.

## No container restart required

Credential refresh is **file-based hot reload**:

- Writers replace the auth JSON (or token file) in place under lock.
- LiteLLM mounts the host credential directories **read-only** and re-reads them
  when selecting a candidate or building provider headers.
- Successful sidecar or manual refresh does **not** require restarting the
  LiteLLM proxy container, the provider-status sidecar, or the host CLI for the
  new token to become visible to subsequent requests.

Restart only when changing compose mounts, env path overrides, or process-level
configuration that is not re-read from disk. Token rotation alone is not a
restart event.

## Operator checklist

1. Keep defaults or env overrides on portable `~` / expanded host paths; avoid
   committing operator-specific absolute homes.
2. Confirm auth files are regular files (`ls -l`; no `l` symlink bit on the
   final path).
3. Confirm mode is private (`0600`) after refresh; set uid/gid/mode env overrides
   if a prior container UID owns the file.
4. Run the family refresh (sidecar cycle or manual script) and inspect the
   summary: `refreshed` / `skipped` / redacted `error_message` only.
5. Verify LiteLLM continues serving without restart once the file is updated.
6. For `--once`, require exit `0` for enabled Grok OIDC, every enabled Codex
   record, and managed xAI OAuth refresh. Optional telemetry degradation is
   reported separately and does not mask required refresh failures.

## Implementation map

| Area | Location |
| --- | --- |
| Shared lock | `litellm/secret_managers/credential_file_lock.py` |
| Shared metadata | `litellm/secret_managers/credential_file_metadata.py` |
| Shared atomic write | `litellm/secret_managers/credential_file_write.py` |
| Shared error sanitizer | `litellm/secret_managers/credential_error_sanitizer.py` |
| Codex ordered inventory | `litellm/secret_managers/codex_oauth_inventory.py` |
| Codex refresh | `scripts/codex_oauth_refresh.py` |
| Managed xAI refresh | `scripts/xai_oauth_refresh.py` |
| Grok OIDC refresh | `scripts/grok_oidc_refresh.py` |
| Managed Kimi Code refresh | `scripts/kimi_oauth_refresh.py` |
| Sidecar loop | `scripts/run_provider_status_observations_loop.py` |
