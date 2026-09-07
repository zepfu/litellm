# Offline report exports

`src/reporting/export.ts` renders an existing `RawModelReport` as JSON, CSV,
or Markdown. It is a pure formatter: it does not open SQLite, fetch history,
call a provider, calculate quota, or wire a CLI/API/UI surface.

## Input contract

Every export requires explicit context:

- `evaluatedAt` is the report evaluation instant.
- `timezone` is the display timezone.
- `freshness` records `fresh`, `stale`, or `unknown`, with optional observation
  and age fields.
- `coverage` records `complete`, `partial`, or `unknown`, with optional source
  and observation fields.
- `provenance` may preserve report revision/fingerprint, policy, and
  bucket-window evidence without accepting arbitrary nested payloads.

An optional `QuotaEstimate` is projected separately from raw-model activity.
Requested-model counts, completed recorded-final counts, mismatches, possible
activity, uncertainty, coverage gaps, working quota values, window IDs, and
policy identifiers remain distinct.

```ts
import {
  renderCsv,
  renderJson,
  renderMarkdown,
} from "./src/reporting/export.js";

const input = {
  report,
  quotaEstimate,
  context: {
    evaluatedAt: "2026-09-07T12:00:00.000Z",
    timezone: "America/New_York",
    freshness: { status: "fresh", observedAt: "2026-09-07T11:59:00.000Z" },
    coverage: { status: "partial", source: "history-ledger" },
  },
};

renderJson(input);
renderCsv(input);
renderMarkdown(input);
```

JSON keeps unknown numeric values as `null`. CSV and Markdown display null and
unknown values as `Unknown`; neither format converts unknown data to zero.
Negative unclamped remainders and their discrepancy qualification are rendered
separately from the presentation-safe remaining estimate.

## Observation qualification

`QuotaEstimate.serverReportedRemaining` only says that a value occupied the
quota engine's server-observation field. It does not identify the source.
Provide `quotaObservationContextByBucket` when source and qualification are
known:

```ts
quotaObservationContextByBucket: {
  "bucket-1": {
    source: "provider",
    qualification: "provider_verified",
    scoped: true,
    provenance: "provider-window-observation",
  },
}
```

Confidence A is assigned only when all three values are explicit:
`provider`, `provider_verified`, and `scoped: true`. Operator, manual, and
absent-source observations retain their source/qualification state and are
never upgraded to provider-verified/A. Working quota estimates are always
labeled `Working estimate; not an official remaining quota` and use confidence
C. Reconstructed raw-model counts use confidence B. Markdown includes
`Relevance: 100% to requested interval`.

CSV cells are quoted and escaped according to CSV rules. Textual cells that
begin with `=`, `+`, `-`, or `@` receive a leading apostrophe so spreadsheet
software does not interpret them as formulas. Markdown escapes untrusted table
values and removes raw line breaks. Only the typed allowlist in
`ReportExportDocument` is serialized; coverage-gap details are reduced to
`coverage` and `warnings`, and credentials, message content, titles, and
arbitrary nested payloads are excluded.
