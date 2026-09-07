# Privacy and threat model

## Assets

- browser cookies, session tokens, local storage, and other authentication
  state inside the dedicated Playwright profile;
- provider user, workspace, and quota-owner identifiers;
- conversation and message relationship identifiers;
- model, mode, reasoning, status, timestamp, and generation metadata;
- local config and bootstrap state files.

## Controls

### Profile isolation

Live collection requires a non-default dedicated profile path. The bootstrap
flow does not create or touch a missing profile unless interactive login was
explicitly requested. Profile and state directories are created with mode
`0700`; the persisted bootstrap file is created with mode `0600` where the
filesystem permits it.

### Egress restriction

The adapter and both transports share a boundary that permits only `GET` and
checks exact route shapes and safe conversation ID tokens before issuing a
request. They never submit prompts, change conversations, archive or delete
data, call the init route, or export browser state.

### Identity binding

The account is ready only when all configured provider user, workspace, and
quota-owner values are observed and equal. Partial or guessed identity is
blocked.

### Data minimization

Content keys, titles, prompts, answers, credentials, cookies, raw headers,
storage, and email addresses are excluded from normalized projections.
Metadata is typed and allowlisted. Persisted identity is checked with
`assertNoSecrets` immediately before writing.

### Test isolation

The fixture transport, offline CLI path, and all committed fixtures are
synthetic. Tests and fixture CLI acceptance do not read a user browser profile
or make network calls.

## Threats and residual risk

| Threat | Stage-1 response | Residual risk |
| --- | --- | --- |
| Accidental mutating provider call | `GET`-only method and exact route allowlist | A future change could weaken the boundary; route tests must remain part of the gate |
| Wrong account attribution | Required three-field identity match | Provider schema changes can make identity unavailable until the adapter is updated |
| Credential or content persistence | Allowlist projection and secret scan | Upstream response data still exists in memory during the request |
| Shared/default profile exposure | Dedicated-profile checks and explicit login | Filesystem ownership and OS-level access are outside the package |
| HTML login or auth challenge treated as data | HTML, `401`, and `403` become auth-required | Live provider behavior can change and require new detection |
| Provider throttling mistaken for quota | `429` becomes `RateLimitedError` | No retry scheduler or quota accounting exists in Stage 1 |
| Schema drift | Coverage and warnings are returned instead of inventing completeness | The current adapter cannot interpret unknown future shapes |

This is a local privacy boundary, not a security boundary against a
compromised host, compromised Node process, malicious dependencies, or a user
who grants access to the profile directory.
