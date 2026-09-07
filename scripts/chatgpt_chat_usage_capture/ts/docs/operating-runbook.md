# Stage-2B operating runbook

Stage 2B operates on a local SQLite ledger. It does not traverse browser
history, schedule collection, submit provider requests, or expose an API/UI.

## Initialize

```bash
cd scripts/chatgpt_chat_usage_capture/ts
npm ci
npm run build
node bin/usage-capture.mjs init --config ./config.json
```

Set the verified account identity fields before any future browser collection.
The default database path is `./state/usage.sqlite`; it can be changed with
`application.database_path`.

## Record and review mappings

The TypeScript API records a reviewed mapping version through
`Ledger.saveModelMapping`. The CLI review view reads only retained ledger
attempts:

```bash
node bin/usage-capture.mjs models \
  --config ./config.json \
  --database ./state/usage.sqlite \
  --mapping-version initial-unmapped
```

Do not treat a suggestion as an approved billing or quota classification.
Review exact raw slug, mode, and reasoning-effort evidence before recording a
new approved version.

## Report

```bash
node bin/usage-capture.mjs report \
  --config ./config.json \
  --account personal-primary \
  --database ./state/usage.sqlite \
  --last-hours 24
```

This is observed raw-model activity. It does not mean remaining quota or
provider-charged messages.

## Rebuild

Preview deterministic reaggregation without changing the ledger:

```bash
node bin/usage-capture.mjs rebuild \
  --config ./config.json \
  --account personal-primary \
  --database ./state/usage.sqlite \
  --mapping-version mapping-v2
```

Apply only after reviewing the preview:

```bash
node bin/usage-capture.mjs rebuild \
  --config ./config.json \
  --account personal-primary \
  --database ./state/usage.sqlite \
  --mapping-version mapping-v2 \
  --apply
```

Rebuilds use stored message records and raw model evidence. They do not issue
website requests.

## Backup and restore

Stop writers before copying the database, or use SQLite's online backup
facility. Preserve the `-wal` and `-shm` files while a live writer exists.
Restoring a copy must be followed by `npm run build` and an offline report or
rebuild preview before it is used for any later collection.

Never back up the dedicated browser profile with ordinary report artifacts.
Browser state contains authentication material and is outside the ledger
backup boundary.

## Coverage and revisions

Inspect `coverage_gaps`, observation revisions, message revisions, attempt
revisions, and aggregate revisions before treating a number as complete for a
declared scope. A partial or unrecognized detail page is not an empty usage
result.

## Deferred operations

Browser discovery traversal, recurring scheduling, reset-window accounting,
provider quota observations, local API/UI, and exports are later lanes. Do not
work around these gaps by treating a raw activity count as an official quota
balance.
