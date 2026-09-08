# Cursor Agent CLI (`cursor_agent`)

Direct LiteLLM provider for the **Cursor Agent CLI** Connect route. This is
**not** [Cursor Cloud Agents](/docs/pass_through/cursor).

| Property | Details |
|----------|---------|
| Provider | `cursor_agent` |
| Turn host | `https://agentn.global.api5.cursor.sh` |
| Turn RPC | `POST /agent.v1.AgentService/Run` (HTTP/2 Connect) |
| Dashboard / auth host | `https://api2.cursor.sh` |
| Credential | `CURSOR_AUTH_TOKEN` or `CURSOR_API_KEY` |
| Not used | `CURSOR_CLI_KEY`, Cloud Agents `cursor`, `CURSOR_API_BASE` |

Cloud Agents stay on `custom_llm_provider="cursor"` and
`LITELLM_PROXY_BASE_URL/cursor`. Do not send Agent CLI turns through
`/cursor` or `api.cursor.com`. This route is not `openai_like`.

## API key

The raw API key is **not** the request credential. LiteLLM sends
`Authorization: Bearer <accessToken>`.

```python
import os

os.environ["CURSOR_API_KEY"] = ""          # raw key; sent as Bearer until exchanged
# or
os.environ["CURSOR_AUTH_TOKEN"] = ""       # already an access token
```

`CURSOR_AUTH_TOKEN` is preferred over `CURSOR_API_KEY`. `CURSOR_CLI_KEY` is
ignored.

## Sample usage

```python
from litellm import completion
import os

os.environ["CURSOR_API_KEY"] = ""
response = completion(
    model="cursor_agent/composer-2.5",
    messages=[{"role": "user", "content": "hello from litellm"}],
)
print(response)
```

Supported first slugs: `composer-2.5`, `cursor-grok-4.6-high`. Catalog keys
are `cursor_agent/composer-2.5` and `cursor_agent/cursor-grok-4.6-high`.

Composer 2.5 standard is not Fast (`composer-2.5-fast`) and is not xAI
`grok-composer-2.5-fast`. Cursor Grok 4.6 is not `oa_xai/grok-4.6` or
`xai/grok-4.6`. Alias YAML uses `provider: cursor_agent` with
`codex_cursor_agent_aiserver_adapter` /
`anthropic_cursor_agent_aiserver_adapter`. Cloud Agents `cursor` stays
untouched.

Public Cursor list rates (2026-08-19, `https://cursor.com/docs/models-and-pricing`)
are stored as reference-only catalog rows with
`actual_invoice_cost_known=false`:

| Catalog key | Input / M | Cache read / M | Output / M |
|---|---:|---:|---:|
| `cursor_agent/composer-2.5` | $0.50 | $0.20 | $2.50 |
| `cursor_agent/cursor-grok-4.6-high` | $2.00 | $0.50 | $6.00 |

These are not Cursor Models pool / subscription invoice economics. Do not
bake the temporary Grok 4.6 launch discount into the cost map. Alias
dispatch for these families uses official schema field 2,
`UserMessageAction.requestContext`. For a fresh clean dispatch, LiteLLM
intentionally sends that field as an empty message (`requestContext: {}`).
The official CLI normally computes and populates `RequestContext` from local
context; it does not itself prove an explicit `{}` send. AgentRunRequest field
12 is never sent. The direct `cursor_agent` provider route remains available.

## Monthly usage

Account-scoped monthly included spend is read from Dashboard Connect
`POST /aiserver.v1.DashboardService/GetCurrentPeriodUsage` on
`https://api2.cursor.sh`. That RPC is not Cloud Agents `GET /v0/me` and
is not the `agentn` turn.

When the provider-status sidecar is explicitly enabled
(`AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED=1` or
`--cursor-agent-usage-poll-enabled`), it maps:

| Observation field | Dashboard field |
|---|---|
| `quota_used` | `planUsage.includedSpend` (USD cents) |
| `quota_limit` | `planUsage.limit` |
| `quota_remaining` | `planUsage.remaining` |
| `quota_period` | `monthly` |

The included fraction is `includedSpend / limit`. Do not treat
`totalPercentUsed` / `autoPercentUsed` / `apiPercentUsed` as
`totalSpend / limit`. Account identity is hashed. Failed refreshes keep
the last valid `public.rate_limit_observations` row.

Weekly Cursor Grok Bot used/limit/reset is still unknown. There is no
weekly `quota_key`. Do not treat xAI Grok Build weekly credits or BugBot
license RPCs as Grok Bot. Set
`AAWM_CURSOR_AGENT_GROK_BOT_USAGE_SOURCE` only when a verified weekly
source exists; until then the checkpoint stays unknown.

The poller is disabled by default so LiteLLM does not send live
dashboard traffic.

The provider-status sidecar can separately maintain the GUI auth file at
`/home/zepfu/.config/cursor/auth.json` using the verified API-key exchange
endpoint. LiteLLM proxies consume the containing directory read-only, while the
sidecar writes a complete replacement atomically under a filesystem lock. The
sidecar does not execute or depend on the Cursor CLI. If no exchangeable
`apiKey` is available, it fails closed rather than continuing with an expired
or unusable access token. Optional usage polling remains disabled and is not
auth-refresh evidence.

## Stock Codex child agents

The Codex adapter advertises configured `collaboration` namespace tools as child
functions with their input schemas, not as a callable namespace container.
JSON and SSE responses restore their namespace identity for fresh runs and
retained-session continuations; replay state preserves the original tool
definitions. Bounded full-history replay preserves a function call's optional
namespace as a nonempty string without surrounding whitespace, alongside its
original name, arguments, and call ID.

For stock `collaboration.wait_agent` and its supported namespace/tool aliases,
Cursor's build/restoration copies narrow `timeout_ms` from `number` to `integer`
to match the client's integer parser without changing the advertised names.
The native Grok and managed xAI Codex fallback routes apply the same correction.
Finite integral protobuf values are returned as JSON integers; fractional values
and unrelated numeric arguments are not coerced. Canonical replay tool
definitions remain unchanged.

Stock Codex child-agent requests arrive from Cursor as ExecServerMessage field
28 (`SubagentArgs`). LiteLLM bridges the portable fields to the advertised
`spawn_agent` tool:

| SubagentArgs field | spawn_agent argument |
|---|---|
| `subagent_type` | `agent_type` |
| `model_id` | `model` |
| `prompt` | `message` |
| `readonly` | `readonly` |

When Cursor advertises `spawn_agent`, its input schema must be an object with
`properties`. The bridged properties must use the canonical names above, declare
the matching scalar types, and accept the requested values. Required properties
that cannot be represented, ambiguous schema definitions, invalid schema
payloads, and explicit `readonly=true` values that the schema cannot represent
are rejected before child dispatch. An explicit `readonly=false` is omitted
when the advertised schema has no `readonly` property. Without an advertised
schema, LiteLLM uses the canonical argument names.

Unsupported subagent fields and schema failures are deterministic candidate
ineligibility (`aawm_codex_auto_agent_candidate_ineligible`) with no candidate
cooldown. LiteLLM does not forward unsupported child-agent context such as
credentials, resume/fork state, selected context, parent state, environment,
or model parameters. Protocol framing and transport failures remain separate
upstream errors.

Before Codex ownership or provider-route admission, LiteLLM normalizes only
recognized V2 collaboration message schemas. The stock client is asked to
place the exact assignment in a versioned plaintext frame inside the `message`
string; LiteLLM validates that frame, supports `NEW_TASK`, `MESSAGE`, and
`FINAL_ANSWER` envelopes, checks their author/recipient identities, and then
makes readable assignment text visible in the existing `agent_message`.
Opaque, unknown, malformed, mixed, or stale single-part payloads fail closed
before provider dispatch. Direct and native xAI parent routes use the same
framing instruction. Regenerate and resend the assignment after correction;
retries or profile changes cannot repair an existing ciphertext blob. Other
envelope shapes, encrypted reasoning, and provider-owned continuation state
remain subject to the existing ownership guards.

For native OpenAI Responses egress, reserved `collaboration.*` function names
are replaced only for the recognized V2 identities with deterministic,
request-local `aawm_cfg047_v1_...` wire aliases. The inverse preserves the exact
function name and namespace in JSON and SSE responses. Alias collisions with an
unrelated advertised function fail closed; foreign-provider tool names and
ordinary encrypted state are not rewritten. A source request that already
contains an unreadable assignment still requires regeneration rather than
retrying the old ciphertext.

At the Cursor boundary, stock `agent_message` items with an author, recipient,
and entirely plaintext `input_text` content become user messages in the derived
chat history. The canonical Responses items are not rewritten. Mixed opaque
content, malformed parts, and unsupported fields fail closed before egress.

## Continuations and ownership

Ordinary external-tool continuations retain the Cursor session assignment while
the provider-owned session is live. Generic MCP/function calls (operation field
11) return their actual output through Cursor's `McpResult` on the same open
Run. Passive tool-progress notifications do not authorize another execution.
Each bounded HTTP/2 read is fully consumed before pausing at a tool call, with
later Connect frames retained in order and data acknowledged once. Received
terminal or abort evidence prevents continuation result writes. Per-continuation
transport evidence distinguishes result bytes handed to the writer from
subsequent provider data; it contains no tool payloads or credentials.

For stock full-history requests without `previous_response_id`, lookup requires
the existing guarded owner identity, unchanged assignment and tools, and exact
pending call IDs and qualified namespace/name identities. The trusted history
prefix may contain earlier completed call/result pairs; only the newly
completed pending outputs are sent back. Historical outputs are not executed
again. A mismatched or ambiguous live session fails closed. Claimed and consumed
generations remain non-replayable within the process-local registry's existing
600-second/256-entry retention bounds. This is not durable deduplication across
worker replacement or eviction.

Missing retained state and recoverable failure of one live transport do not
enter shared cooldown evidence or publication. Socket EOF after complete frames
but before an accepted terminal is a transport failure; malformed framing and
protocol messages do not qualify for this recovery. Failed live continuations carry
separate invocation, result-write and provider-progress evidence; unknown write
status is counted conservatively, not reported as no egress. Full-history
recovery retains the existing single trailing call/output-pair grammar.
Output-only recovery uses a separate snapshot of the stored history with no
live session pointer. A consumed generation does not authorize another recovery
attempt merely because its socket has closed.

LiteLLM only permits provider-neutral
fallback after reconstructing a complete, replay-safe request with the
original assignment and completed tool history; partial incremental bodies,
opaque Cursor state, unresolved tool calls, nested Cursor identifiers, and
ownerless provider state fail closed before egress. A valid replay may then
traverse native `xai` and managed `oa_xai` candidates without migrating the
owned Cursor session.

Replay validation preserves function namespaces even when the installed OpenAI
SDK predates namespace tools. The compatibility path validates the namespace
shape and each function schema rather than restricting namespace or function
names to a fixed catalog. Unknown fields, malformed children, and duplicate
function names fail closed; accepted replay retains the original tool definitions.

For Responses streams, session ownership is promoted only after the validator
has observed a structurally valid terminal response with `status=completed` and
the response stream has reached its terminal lifecycle. Malformed, incomplete,
failed, or prematurely closed responses release the pending reservation and do
not establish durable affinity.

When validation continues lazily after its bounded stream peek, it also rejects
literal tool-call text before forwarding a successful terminal event. Already
forwarded text is not replayed as executable calls; buffered-response repair
remains separate. Rejection retains the existing malformed-tool-call error and
cleanup path without promoting the pending owner.

## What this is not

- Cloud Agents `/v0/agents` on `https://api.cursor.com`
- OpenAI `/v1/chat/completions`
- HTTP/1.1 `RunSSE` + `BidiAppend`
- A `cursor-agent` / `agent` subprocess
- Cloud Agents `GET /v0/me` usage
