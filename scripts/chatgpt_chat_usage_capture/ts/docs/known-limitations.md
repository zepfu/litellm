# Known limitations

- Stage 2A stops at sanitized history discovery/acquisition and JSON
  checkpoints. There is no SQLite ledger/schema, accounting, attempt
  reconstruction, quota-window evaluation, scheduler, local API, dashboard,
  export, or model report.
- A complete active and archived index scan is complete only for the records
  exposed by those private endpoints during that scan. Temporary, deleted,
  inaccessible, or otherwise unenumerated conversations remain outside the
  observed coverage.
- Project coverage remains `unknown` unless Project identifiers are actually
  observed. `has_versions` exposes branch/version metadata or active-branch
  evidence; it does not prove that all historical branches are visible.
- JSON checkpoints are an interim Stage-2A durability boundary. They support
  restartable discovery and incomplete-detail revisits but do not provide the
  later ledger's transactional observation, message, or attempt history.
- A page budget, repeated cursor, contradictory total, unknown schema, or
  provider error produces partial coverage and keeps the relevant watermark
  from advancing. Previously valid checkpoints are retained.
- Explicit backfill and reconciliation ranges do not use an incremental
  watermark. Implicit refresh uses a 48-hour discovery overlap; this overlap
  does not discard older stored activity.
- Legacy detail is used only after an approved capability and a modern `404` or
  `405`. Missing conversations and endpoint capability failures cannot always
  be distinguished from a private provider response.
- The capability record is a versioned adapter declaration plus evidence
  observed during collection. It is not an official provider capability
  guarantee.
- Live operation requires a dedicated persistent Playwright profile and a
  locally installed Chromium browser. Automated credential acquisition is not
  supported.
- Fixture-backed commands are offline contract acceptance only; they do not
  prove live endpoint availability, authentication, or provider schema
  compatibility.
- Interactive login is intentionally opt-in and human-driven. The package
  never accepts credentials on the command line or writes them to config,
  logs, fixtures, or state.
- Provider route and response schemas may change. Unknown shapes and
  contradictory pagination are marked explicitly; the adapter does not invent
  complete coverage.
- Quota metadata and official remaining values are not collected in Stage 2A.
- The fixture transport is deterministic synthetic support, not a provider
  emulator and not a live endpoint proof.
- The Python implementation remains the reference for later accounting and
  scheduling behavior. This TypeScript package does not duplicate those
  stages.
