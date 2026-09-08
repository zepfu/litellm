# Pure Counting Module Provenance

Current `develop` contains only the Stage 1 collection path. The following
modules restore the previously reviewed TypeScript counting implementation
without its SQLite runtime, standalone CLI, dashboard, local API, service, or
tests. Runtime adapter compatibility is retained because the Stage 1 adapter
and sanitizer were superseded by the same preserved Stage 2 tree.

| Module | Preserved source | Role |
| --- | --- | --- |
| `src/contracts/records.ts` | `fb51bf0287` | Shared normalized record and pagination contracts. |
| `src/adapters/chatgpt/adapter.ts` | `fb51bf0287` | Compatible pure request/response projection including pagination-state evidence. |
| `src/security/sanitizer.ts` | `fb51bf0287` | Privacy and bounded-projection dependency used by the adapter. |
| `src/contracts/history.ts` | `fb51bf0287` | Collection/request contracts consumed by reconstruction flows. |
| `src/ledger/identity.ts` | `fb51bf0287` | Stable scope/attempt identity and canonical JSON helpers. |
| `src/ledger/types.ts` | `fb51bf0287` | Ledger scope, mapping, and reconstructed-attempt types. |
| `src/normalize/model-mapping.ts` | `fb51bf0287` | Versioned model mapping resolution. |
| `src/normalize/reconstruct.ts` | `920c18cc5a`; identical hardened body at `6c1eae83ae` | Pure generation graph reconstruction and evidence linkage. |
| `src/accounting/quota.ts` | `fb51bf0287` | Corrected pure quota estimation semantics. |
| `src/accounting/windows.ts` | `fb51bf0287` | Pure reset-window resolution and interval membership. |

`6c1eae83ae` hardened reconstruction behavior in its own tree, while its
`src/normalize/reconstruct.ts` blob is byte-identical to `920c18cc5a`. The
assignment therefore records both provenance sources and restores one
authoritative body.

The new `src/counting/index.ts` is the package-internal export surface for a
future bounded worker. It deliberately does not define a transport protocol or
connect a live runtime. No `ledger/store.ts`, SQLite package dependency,
scheduler store, old standalone CLI, dashboard/API entrypoint, obsolete browser
launch, or test source was restored.
