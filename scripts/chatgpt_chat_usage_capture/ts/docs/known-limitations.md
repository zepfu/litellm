# Known limitations

- Stage 2B does not perform browser discovery traversal. Existing adapted
  conversation pages must be supplied by a later collection lane before they
  can be ingested.
- The ledger stores metadata and relationship evidence only. Prompt/answer
  content, tool arguments/results, credentials, cookies, and browser storage
  are not retained.
- `inspect-capabilities` reads only the first active and archived index pages.
  The adapter reports continuation and coverage signals, but this lane does not
  perform a complete history traversal.
- The capability record is a versioned adapter declaration plus the evidence
  observed during the command; it is not an official provider capability
  guarantee.
- Attempt reconstruction is deterministic over retained message evidence, but
  an upstream schema that hides a branch, generation, request, or terminal
  status can leave an attempt provisional, unresolved, or time-ambiguous.
- A shared, copied, or imported conversation is retained as evidence and
  explicitly excluded from ordinary Chat raw-model counts. The exclusion is
  based on observed origin metadata and cannot identify an origin that the
  provider does not expose.
- Raw-model reports count observed activity in a half-open interval and retain
  unknown/ambiguous time, model mismatches, and unmapped labels. They do not
  calculate remaining quota, reset periods, billing, or provider-side usage.
- Model families are reviewed local classifications. Reclassification changes
  retained projections and mapping history; it does not establish provider
  billing or entitlement semantics.
- Live operation requires a dedicated persistent Playwright profile and a
  locally installed Chromium browser. Automated credential acquisition is not
  supported.
- Fixture-backed `inspect-capabilities` is offline contract acceptance only; it
  does not prove live endpoint availability, authentication, or provider schema
  compatibility.
- Interactive login is intentionally opt-in and human-driven. The package
  never accepts credentials on the command line or writes them to config,
  logs, fixtures, or state.
- Provider route and response schemas may change. Unknown shapes are marked
  `unrecognized` or `partial`; the adapter and ledger record coverage gaps
  rather than inventing complete coverage.
- Project coverage, quota metadata, and reset windows are not collected in
  Stage 2B. No official remaining-quota value is inferred.
- The fixture transport is deterministic synthetic support, not a provider
  emulator or a live endpoint proof.
- The Python implementation remains outside this lane. This TypeScript package
  does not import or modify it.
