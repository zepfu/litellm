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
| `src/normalize/reconstruct.ts` | `920c18cc5a`, hardened at `6c1eae83ae`, with local D1-752 reconstruction fixes | Pure generation graph reconstruction and evidence linkage. |
| `src/accounting/quota.ts` | `fb51bf0287` | Corrected pure quota estimation semantics. |
| `src/accounting/windows.ts` | `fb51bf0287` | Pure reset-window resolution and interval membership. |

`6c1eae83ae` hardened reconstruction behavior in its own tree. The current
local fixes normalize connected evidence components, avoid backward prompt
attribution, and preserve imported/copied metadata. Generation IDs take
precedence over request aliases, including when one generation carries multiple
request IDs. Request-only fallback uses the normalized branch/component and
does not combine disconnected components that reuse a request ID.

Fragment linkage cannot absorb a completed original response into a later
generation. Conflicting generation links remain unresolved. Prompt evidence is
selected separately from grouping: graph precedence or unambiguous sibling
timestamps can identify an original request-only response without passing its
model/time to a later regeneration. Missing/tied sibling timestamps, ambiguous
explicit prompt IDs, and reused requests across sibling generation anchors do
not create prompt attribution.

The grouping pass collects all forward user views and orphan evidence before
building attempts. Identical generation IDs therefore retain final/model
evidence from disconnected components and multiple user views. Conflicting
user associations withhold prompt-derived fields but retain all contextual
surface/origin evidence.

Generation and request candidates are resolved against immutable owner sets.
Request-free bridges with multiple owners, multiple user/branch roots, and
unsupported continuation-versus-regeneration relations remain unresolved.
Unresolved identity does not erase observed generation activity. Request
fallback uses actual residual connectivity as well as the owner and request
tuple; request equality cannot connect disjoint evidence. Prompt ownership is
resolved once across all identity classes, including earlier unresolved or
provisional evidence, before projecting the requested model and submission time.
Synthesized prompt aliases use the existing tuple-hashing helper so their
unambiguous identity also satisfies the storage token contract.

Request components stop at terminal response boundaries, so a reused request
cannot combine a completed original and its later response. Forward attachment
to a request owner requires continuation/fragment evidence; unknown status,
channel, and turn state leave that relationship unresolved.

Terminal observations preserve completion independently of final-node selection.
With multiple terminal nodes, complete valid timestamp ordering must identify
a unique latest node. Tied or missing timestamps retain all plausible terminal
observations, expose
`terminal_selection_ambiguous`, and project only model evidence on which those
observations agree. Conflicting recorded/resolved models additionally expose
`terminal_model_conflict`; message IDs never decide the final model.

The new `src/counting/index.ts` is the package-internal export surface for a
future bounded worker. It deliberately does not define a transport protocol or
connect a live runtime. No `ledger/store.ts`, SQLite package dependency,
scheduler store, old standalone CLI, dashboard/API entrypoint, obsolete browser
launch, or test source was restored.
