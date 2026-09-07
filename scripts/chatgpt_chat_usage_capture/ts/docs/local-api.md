# Local API

The local API is a small HTTP server that exposes the collector's read and
write surface to the dashboard and CLI. It is a thin security and routing
layer over the existing collector domain services; it does **not** talk to
ChatGPT or any other provider, and it does **not** expose an arbitrary
fetch endpoint.

## Binding and authentication

- The server binds to `127.0.0.1` by default and uses an ephemeral port
  unless one is configured.
- Binding to a non-loopback interface requires an explicit
  `allowRemoteBind: true` opt-in; otherwise the server refuses to start.
- Every `/api/v1/*` route requires a bearer token:
  `Authorization: Bearer <token>`. The token is never rendered in HTML,
  never logged, and never returned in a response body.
- Browser clients must also send `X-Usage-Capture-Token: <token>` on POST
  requests for CSRF protection. Non-browser clients may omit `Origin`.

## Host, Origin, and CSRF controls

- `Host` must be loopback (or an explicitly configured allowed host).
- `Origin` must be a loopback origin matching the bound port (or an
  explicitly configured allowed origin).
- Mutating requests without a valid CSRF header are rejected with `403`.

## Routes

### Health

| Route | Purpose |
|---|---|
| `GET /health/live` | Process is running. Returns quickly. |
| `GET /health/ready` | Database/config are usable. |

Health endpoints are unauthenticated and return minimal JSON. If the
corresponding `checkLive`/`checkReady` service is not injected, they return
a minimal default payload.

### Read routes

| Route | Query parameters |
|---|---|
| `GET /api/v1/accounts` | none |
| `GET /api/v1/status` | `account` |
| `GET /api/v1/usage` | `account`, `from`, `to`, `group_by` |
| `GET /api/v1/attempts` | `account`, `cursor`, `limit` |
| `GET /api/v1/quota-windows` | `account` |
| `GET /api/v1/quota-observations` | `account` |
| `GET /api/v1/coverage` | `account`, `from`, `to` |
| `GET /api/v1/runs` | `account` |

All query values are validated strings. `from`/`to` must be valid RFC 3339
timestamps. `limit` is clamped to a hard cap of 1000; the default is 200.

### Write routes

| Route | Body |
|---|---|
| `POST /api/v1/refresh` | `{ "account"?: string, "idempotency_key"?: string }` |
| `POST /api/v1/window-definitions` | `{ "account"?: string, "window"?: object, "idempotency_key"?: string }` |
| `POST /api/v1/manual-observations` | `{ "account"?: string, "observation"?: object, "idempotency_key"?: string }` |
| `POST /api/v1/rebuild-preview` | `{ "account"?: string, "idempotency_key"?: string }` |
| `POST /api/v1/rebuild-apply` | `{ "account"?: string, "preview_id"?: string, "idempotency_key"?: string }` |
| `POST /api/v1/schedule` | Reserved for scheduler stage; returns `501 not_implemented`. |

Write routes accept a bounded JSON object (default 16 KiB, max nesting depth
16, max 256 keys). If `idempotency_key` is provided, the key and a
deterministic request snapshot are reserved before the service callback runs.
Concurrent requests with the same key and body coalesce onto the original
callback; reusing the same key with a different body returns `409 conflict`,
including while the original callback is still running. The response is
stored as an immutable snapshot, so later callback mutations do not change a
replay.

The idempotency cache is bounded (default 1000 entries) and evicts completed
entries oldest-first. In-flight reservations are never evicted. If all
configured slots are in flight, a new keyed write returns `503 unavailable`.
If a callback fails, its reservation is removed and concurrent waiters receive
the same failure response; a later retry may claim the key again.

## Service injection and availability

The server constructor accepts a `LocalApiServices` object. Each route maps
to one typed callback. When a callback is not injected, the route returns an
explicit `503 unavailable` with `service_unavailable_no_injected_handler`.
This is intentional: the API does not fabricate payloads.

```ts
import { createLocalApiServer } from "./api/index.js";

const token = process.env.USAGE_CAPTURE_API_TOKEN;
if (token === undefined) {
  throw new Error("USAGE_CAPTURE_API_TOKEN is required");
}

const server = await createLocalApiServer({
  authToken: token,
  services: {
    listAccounts: async () => ({ accounts: [] }),
    getStatus: async ({ account }) => ({ account, status: "unknown" }),
  },
});

console.log(server.address());
await server.close();
```

## Clean shutdown

`close()` stops accepting new connections and closes idle ones. Call it
during process shutdown to avoid leaking sockets.
