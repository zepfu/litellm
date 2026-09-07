# Known limitations

- Stage 1 stops at bootstrap and capability inspection. There is no ledger,
  accounting, attempt reconstruction, quota-window evaluation, scheduler,
  local API, dashboard, export, or model report.
- `inspect-capabilities` reads only the first active and archived index pages.
  The adapter reports continuation and coverage signals, but Stage 1 does not
  perform a complete history traversal.
- The capability record is a versioned adapter declaration plus the evidence
  observed during the command. It is not an official provider capability
  guarantee.
- Live operation requires a dedicated persistent Playwright profile and a
  locally installed Chromium browser. Automated credential acquisition is not
  supported.
- Interactive login is intentionally opt-in and human-driven. The package
  never accepts credentials on the command line or writes them to config,
  logs, fixtures, or state.
- Provider route and response schemas may change. Unknown shapes are marked
  `unrecognized` or `partial`; the adapter does not invent complete coverage.
- Project coverage and quota metadata are reported as unknown or not collected
  in Stage 1. No official remaining-quota value is inferred.
- The fixture transport is deterministic test support, not a provider emulator
  and not a live endpoint proof.
- The Python implementation remains the reference for later accounting and
  scheduling behavior. This TypeScript package does not duplicate those
  stages.
