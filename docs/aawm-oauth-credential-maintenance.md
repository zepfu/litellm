# AAWM OAuth Credential Maintenance

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
| Codex / ChatGPT OAuth | `scripts/codex_oauth_refresh.py` (sidecar) | LiteLLM Codex adapter routes | `~/.codex/auth.json` |
| Managed xAI OAuth (`oa_xai/*`) | `scripts/xai_oauth_refresh.py` (sidecar) | LiteLLM managed xAI OAuth routes | `~/.litellm/xai/oauth-auth.json` |
| Grok native OIDC | `scripts/grok_oidc_refresh.py` (sidecar) | LiteLLM Grok native routes | Caller-supplied configured path |
| Kimi Code CLI OAuth (`kimi_code`) | Existing Kimi Code CLI grant; sidecar refresh only when enabled | Configured LiteLLM Kimi Code consumers | `~/.kimi-code/credentials/kimi-code.json` |
| Antigravity OAuth | `scripts/antigravity_oauth_refresh.py` (manual / non-sidecar) | LiteLLM Antigravity routes | `~/.gemini/antigravity-cli/antigravity-oauth-token` |

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
| Codex | `~/.codex/auth.json` | `~/.codex/auth.json.lock` |
| Managed xAI OAuth | `~/.litellm/xai/oauth-auth.json` | `~/.litellm/xai/oauth-auth.json.lock` |
| Grok OIDC | Caller-supplied configured path | same directory, `.lock` sibling when configured |
| Kimi Code CLI OAuth | `~/.kimi-code/credentials/kimi-code.json` | `~/.kimi-code/oauth/kimi-code` (native `proper-lockfile` creates the transient `kimi-code.lock` directory) |
| Antigravity | `~/.gemini/antigravity-cli/antigravity-oauth-token` | `~/.gemini/antigravity-cli/antigravity-oauth-token.lock` |
| Antigravity CLI binary candidates | `~/.local/bin/agy` | n/a |

Override paths with the normal env vars for the family in use (for example
`AAWM_CODEX_AUTH_FILE` / `LITELLM_CODEX_AUTH_FILE`,
`AAWM_XAI_OAUTH_AUTH_FILE` / `LITELLM_XAI_OAUTH_AUTH_FILE`,
`AAWM_KIMI_OAUTH_AUTH_FILE` / `LITELLM_KIMI_OAUTH_AUTH_FILE`,
`LITELLM_XAI_GROK_AUTH_FILE`,
`AAWM_ANTIGRAVITY_AUTH_FILE` / `LITELLM_ANTIGRAVITY_MANAGED_AUTH_FILE` and seed
variants). Compose may bind the expanded host path into containers; the script
defaults themselves remain `~`-relative so other operators and hosts work
without patching source.

## Shared atomic 0600 publication

All credential writers share the same private publish pipeline under
`litellm/secret_managers/`:

| Module | Responsibility |
| --- | --- |
| `credential_file_lock.py` | Advisory `fcntl` flock; warns (never silent) if lock is unavailable |
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

Matching lock files sit beside the auth files. Writers hold the advisory lock
for read/refresh/write, including metadata repair on skipped refresh cycles
where the script supports that (Codex and Grok repair paths).

## Metadata env overrides

Publication preserves existing file ownership and private mode unless optional
env overrides are set. Each family uses the same shape:

| Purpose | Codex | Managed xAI | Grok OIDC | Kimi Code CLI | Antigravity |
| --- | --- | --- | --- | --- | --- |
| UID | `AAWM_CODEX_AUTH_FILE_UID` | `AAWM_XAI_OAUTH_AUTH_FILE_UID` | `AAWM_GROK_OIDC_AUTH_FILE_UID` | `AAWM_KIMI_OAUTH_AUTH_FILE_UID` | `AAWM_ANTIGRAVITY_AUTH_FILE_UID` |
| GID | `AAWM_CODEX_AUTH_FILE_GID` | `AAWM_XAI_OAUTH_AUTH_FILE_GID` | `AAWM_GROK_OIDC_AUTH_FILE_GID` | `AAWM_KIMI_OAUTH_AUTH_FILE_GID` | `AAWM_ANTIGRAVITY_AUTH_FILE_GID` |
| Mode | `AAWM_CODEX_AUTH_FILE_MODE` | `AAWM_XAI_OAUTH_AUTH_FILE_MODE` | `AAWM_GROK_OIDC_AUTH_FILE_MODE` | `AAWM_KIMI_OAUTH_AUTH_FILE_MODE` | `AAWM_ANTIGRAVITY_AUTH_FILE_MODE` |

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
- Scheduled refresh is disabled by default:
  `AAWM_KIMI_OAUTH_REFRESH_ENABLED=0`. Set it to `1` only when refresh of the
  existing CLI grant is explicitly intended, and set any needed
  `AAWM_KIMI_OAUTH_AUTH_FILE`, `AAWM_KIMI_OAUTH_LOCK_FILE`, interval, timeout,
  and optional uid/gid/mode overrides.

The compose contract controls credential ownership and hot-reload visibility;
it does not enable a Kimi route by itself. Managed-route behavior, exact model
IDs, and `/models` capability gating are documented in
[`moonshot.md`](my-website/docs/providers/moonshot.md#managed-kimi-code-oauth-aawm).
Any production-equivalent deployment must preserve the same read-only worker
and single-writer sidecar contract, but production mutation remains a separate
operator-authorized rollout.

## Antigravity staged HOME distinction

Antigravity is different from the Codex / xAI / Grok sidecar-owned families:

| Concern | Behavior |
| --- | --- |
| Scheduled writer | Provider-status sidecar does **not** own Antigravity OAuth refresh or Antigravity auth telemetry. |
| Manual / external writer | `scripts/antigravity_oauth_refresh.py` |
| LiteLLM role | Read-only consumer of managed and/or seed token files |
| Alias degradation | Stale/missing Antigravity auth is `auth_degraded` candidate skip, not a request-path refresh |

When direct OAuth refresh fails and the script falls back to the `agy` CLI, it
stages an isolated private HOME tree:

1. Create an unpredictable private temp directory (`tempfile.mkdtemp`, mode
   `0700`).
2. Stage seed credentials under
   `<staged_home>/.gemini/antigravity-cli/antigravity-oauth-token` at mode
   `0600` using the same private write helpers (no umask window, symlink
   refusal).
3. Set `HOME` for the CLI subprocess to that staged home so the CLI never
   mutates the operator's real home layout during fallback.
4. Prefer the staged home for `--log-file` output; otherwise use another private
   `0700` temp parent.
5. Unconditionally remove the staged tree in `finally` on every exit path
   (success, timeout, non-zero CLI exit, OSError).

The staged HOME is temporary publish/isolation state only. The durable
credential that LiteLLM reads remains the configured Antigravity auth/seed path
after the script publishes or leaves a valid token in place.

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
6. For Antigravity, treat CLI fallback staged HOME as ephemeral and confirm
   cleanup left no `litellm-antigravity-cli-home-*` directories behind.

## Implementation map

| Area | Location |
| --- | --- |
| Shared lock | `litellm/secret_managers/credential_file_lock.py` |
| Shared metadata | `litellm/secret_managers/credential_file_metadata.py` |
| Shared atomic write | `litellm/secret_managers/credential_file_write.py` |
| Shared error sanitizer | `litellm/secret_managers/credential_error_sanitizer.py` |
| Codex refresh | `scripts/codex_oauth_refresh.py` |
| Managed xAI refresh | `scripts/xai_oauth_refresh.py` |
| Grok OIDC refresh | `scripts/grok_oidc_refresh.py` |
| Managed Kimi Code refresh | `scripts/kimi_oauth_refresh.py` |
| Antigravity refresh | `scripts/antigravity_oauth_refresh.py` |
| Sidecar loop | `scripts/run_provider_status_observations_loop.py` |
