# Stage-2A operating runbook

Stage 2A is a read-only history acquisition lane. It uses the same
GET-only adapter for fixture and live operation and writes only sanitized local
identity/checkpoint state.

## First run

1. Generate a config with `usage-capture init`.
2. Set `expected_provider_user_id`, `expected_workspace_id`, and
   `quota_owner_id` from the intended account.
3. Set `browser.profile_path` to a dedicated user-owned profile. Never point
   it at a normal browser profile or copy its cookies/storage.
4. Run `bootstrap`, then `inspect-capabilities`.
5. Run `backfill`. The default range is the preceding 14 elapsed days.

For offline acceptance, use `browser.adapter: "fixture_history"` and pass
`--fixture-root`. Fixture mode never contacts ChatGPT.

## Collection modes

```bash
node bin/usage-capture.mjs backfill --config ./config.json
node bin/usage-capture.mjs backfill --config ./config.json \
  --since 30d --until 2026-09-08T00:00:00Z
node bin/usage-capture.mjs refresh --config ./config.json
node bin/usage-capture.mjs reconcile --config ./config.json \
  --since 2026-09-01T00:00:00Z --until 2026-09-08T00:00:00Z
```

`backfill` accepts an elapsed duration or an ISO-8601 start. `--until` is
exclusive. `reconcile` requires an explicit start and does not substitute the
incremental watermark for the requested range.

## Checkpoints and revisits

History state is stored under
`<state-directory>/history/<encoded-account-id>.json` with restrictive local
permissions. Each active/archived scope has its own continuation, page budget,
pagination interpretation, range, warnings, and last complete discovery start.

An incomplete detail or message traversal creates an outstanding revisit. A
later refresh processes the revisit even when the index update timestamp did
not change. A complete detail traversal removes it. A page budget,
non-advancing/repeated cursor, contradictory total, unknown schema, provider
error, or authentication pause leaves the relevant work partial and does not
advance that scope's completion watermark.

## Coverage interpretation

- `complete` means the declared active/archived index path and acquired detail
  pages completed for the available provider response.
- `partial` means a budget, schema, cursor, missing timestamp, provider error,
  project/branch gap, or incomplete revisit remains.
- `unknown` means the adapter could not validate the page controls.
- Active and archived copies of one conversation are deduplicated while both
  scopes remain visible.
- A discovered Project ID proves only coverage for that observed Project
  population. `has_versions` is branch/version evidence, not proof of all
  branches.

Do not interpret local acquisition as official quota usage. Stage 2A does not
reconstruct attempts, write SQLite records, report remaining quota, schedule
recurring work, or mutate provider state.

## Recovery

- `auth_required` or identity mismatch: stop and repair the dedicated browser
  session/configured binding; do not rotate accounts or copy credentials.
- `429`: treat it as transport throttling. Do not use it as a quota signal and
  do not use legacy detail fallback.
- Partial index/detail: rerun `refresh`; the durable continuation or revisit
  queue is the resume point.
- Need an older or exact range: rerun `backfill` or `reconcile` with explicit
  `--since`/`--until`; the watermark will not override that range.
- Schema drift: preserve the prior checkpoint and inspect the explicit
  pagination/coverage warnings before changing fixtures or adapter logic.
