import { describe, expect, it, afterEach } from "vitest";
import { Socket } from "node:net";

import {
  createLocalApiServer,
  type LocalApiServerHandle,
  type LocalApiServices,
} from "../../src/api/index.js";
import { IdempotencyCache } from "../../src/api/idempotency.js";
import { HTTP_STATUS, UNAVAILABLE_REASON } from "../../src/api/types.js";

const AUTH_TOKEN = "test-token-0123456789abcdef";

interface StartedServer {
  server: LocalApiServerHandle;
  baseUrl: string;
}

async function startServer(
  services: LocalApiServices,
  options: Partial<Parameters<typeof createLocalApiServer>[0]> = {},
): Promise<StartedServer> {
  const server = await createLocalApiServer({
    authToken: AUTH_TOKEN,
    services,
    ...options,
  });
  const address = server.address();
  return {
    server,
    baseUrl: `http://${address.host}:${address.port}`,
  };
}

async function apiFetch(
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("Host", new URL(url).host);
  if (!headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${AUTH_TOKEN}`);
  }
  return fetch(url, { ...init, headers });
}

function postJson(
  url: string,
  body: Record<string, unknown>,
): Promise<Response> {
  return apiFetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Usage-Capture-Token": AUTH_TOKEN,
    },
    body: JSON.stringify(body),
  });
}

/**
 * Send a raw HTTP/1.1 request over a socket so headers Node's fetch
 * normalizes (notably Host) reach the server verbatim.
 */
function rawRequest(
  port: number,
  requestText: string,
): Promise<{ status: number; body: string }> {
  return new Promise((resolvePromise, rejectPromise) => {
    const socket = new Socket();
    let received = "";
    socket.connect(port, "127.0.0.1", () => {
      socket.write(requestText);
    });
    socket.on("data", (chunk) => {
      received += chunk.toString("utf8");
    });
    socket.on("end", () => {
      const headerEnd = received.indexOf("\r\n\r\n");
      const statusLine = received.slice(0, received.indexOf("\r\n"));
      const status = Number.parseInt(statusLine.split(" ")[1] ?? "0", 10);
      resolvePromise({
        status,
        body: headerEnd >= 0 ? received.slice(headerEnd + 4) : "",
      });
    });
    socket.on("error", rejectPromise);
  });
}

describe("local API server", () => {
  const servers: LocalApiServerHandle[] = [];

  afterEach(async () => {
    await Promise.all(servers.splice(0).map((server) => server.close()));
  });

  it("serves authenticated read routes with injected services", async () => {
    const { server, baseUrl } = await startServer({
      listAccounts: async () => ({ accounts: ["personal-primary"] }),
      getStatus: async ({ account }) => ({ account, status: "fresh" }),
      getUsage: async ({ account, from, to, groupBy }) => ({
        account,
        from,
        to,
        groupBy,
        total: 42,
      }),
      getAttempts: async ({ account, cursor, limit }) => ({
        account,
        cursor,
        limit,
        items: [],
      }),
      getQuotaWindows: async ({ account }) => ({ account, windows: [] }),
      getQuotaObservations: async ({ account }) => ({
        account,
        observations: [],
      }),
      getCoverage: async ({ account, from, to }) => ({
        account,
        from,
        to,
        coverage: "partial",
      }),
      getRuns: async ({ account }) => ({ account, runs: [] }),
    });
    servers.push(server);

    const accounts = await apiFetch(`${baseUrl}/api/v1/accounts`);
    expect(accounts.status).toBe(HTTP_STATUS.ok);
    await expect(accounts.json()).resolves.toEqual({
      accounts: ["personal-primary"],
    });

    const status = await apiFetch(
      `${baseUrl}/api/v1/status?account=personal-primary`,
    );
    expect(status.status).toBe(HTTP_STATUS.ok);
    await expect(status.json()).resolves.toEqual({
      account: "personal-primary",
      status: "fresh",
    });

    const usage = await apiFetch(
      `${baseUrl}/api/v1/usage?account=personal-primary&from=2026-09-01T00:00:00Z&to=2026-09-07T00:00:00Z&group_by=model`,
    );
    expect(usage.status).toBe(HTTP_STATUS.ok);
    await expect(usage.json()).resolves.toMatchObject({ total: 42 });

    const attempts = await apiFetch(
      `${baseUrl}/api/v1/attempts?account=personal-primary&cursor=abc123&limit=50`,
    );
    expect(attempts.status).toBe(HTTP_STATUS.ok);
    await expect(attempts.json()).resolves.toMatchObject({
      cursor: "abc123",
      limit: 50,
    });
  });

  it("rejects unauthenticated requests without exposing the token", async () => {
    const { server, baseUrl } = await startServer({
      listAccounts: async () => ({ accounts: [] }),
    });
    servers.push(server);

    const response = await fetch(`${baseUrl}/api/v1/accounts`);
    expect(response.status).toBe(HTTP_STATUS.unauthorized);
    const body = (await response.json()) as { error: { code: string } };
    expect(body.error.code).toBe("unauthorized");
    expect(JSON.stringify(body)).not.toContain(AUTH_TOKEN);
  });

  it("rejects requests with a disallowed Host header", async () => {
    const { server, baseUrl } = await startServer({
      listAccounts: async () => ({ accounts: [] }),
    });
    servers.push(server);
    const { port } = server.address();

    const response = await rawRequest(
      port,
      [
        "GET /api/v1/accounts HTTP/1.1",
        "Host: evil.example.com",
        `Authorization: Bearer ${AUTH_TOKEN}`,
        "Connection: close",
        "",
        "",
      ].join("\r\n"),
    );
    expect(response.status).toBe(HTTP_STATUS.forbidden);
    expect(response.body).not.toContain(AUTH_TOKEN);
    expect(baseUrl).toContain(String(port));
  });

  it("rejects requests with a disallowed Origin header", async () => {
    const { server, baseUrl } = await startServer({
      listAccounts: async () => ({ accounts: [] }),
    });
    servers.push(server);

    const response = await apiFetch(`${baseUrl}/api/v1/accounts`, {
      headers: { Origin: "https://evil.example.com" },
    });
    expect(response.status).toBe(HTTP_STATUS.forbidden);
  });

  it("rejects POST without the CSRF header", async () => {
    const { server, baseUrl } = await startServer({
      requestRefresh: async ({ account }) => ({ account, accepted: true }),
    });
    servers.push(server);

    const response = await apiFetch(`${baseUrl}/api/v1/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ account: "personal-primary" }),
    });
    expect(response.status).toBe(HTTP_STATUS.forbidden);
  });

  it("serves POST routes with the CSRF header and idempotency key", async () => {
    let callCount = 0;
    const { server, baseUrl } = await startServer({
      requestRefresh: async ({ account, idempotencyKey }) => {
        callCount += 1;
        return { account, idempotencyKey, accepted: true, callCount };
      },
    });
    servers.push(server);

    const init: RequestInit = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Usage-Capture-Token": AUTH_TOKEN,
      },
      body: JSON.stringify({
        account: "personal-primary",
        idempotency_key: "refresh-001",
      }),
    };

    const first = await apiFetch(`${baseUrl}/api/v1/refresh`, init);
    expect(first.status).toBe(HTTP_STATUS.accepted);
    await expect(first.json()).resolves.toMatchObject({
      account: "personal-primary",
      idempotencyKey: "refresh-001",
      callCount: 1,
    });

    const second = await apiFetch(`${baseUrl}/api/v1/refresh`, init);
    expect(second.status).toBe(HTTP_STATUS.accepted);
    await expect(second.json()).resolves.toMatchObject({
      account: "personal-primary",
      idempotencyKey: "refresh-001",
      callCount: 1,
    });
    expect(callCount).toBe(1);
  });

  it("rejects idempotency-key replay with a different body", async () => {
    const { server, baseUrl } = await startServer({
      requestRefresh: async ({ account }) => ({ account, accepted: true }),
    });
    servers.push(server);

    const headers = {
      "Content-Type": "application/json",
      "X-Usage-Capture-Token": AUTH_TOKEN,
    };

    const first = await apiFetch(`${baseUrl}/api/v1/refresh`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        account: "personal-primary",
        idempotency_key: "refresh-002",
      }),
    });
    expect(first.status).toBe(HTTP_STATUS.accepted);

    const second = await apiFetch(`${baseUrl}/api/v1/refresh`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        account: "other-account",
        idempotency_key: "refresh-002",
      }),
    });
    expect(second.status).toBe(HTTP_STATUS.conflict);
  });

  it("coalesces concurrent same-key requests and invokes the service once", async () => {
    let callCount = 0;
    let release!: () => void;
    const callbackGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let markStarted!: () => void;
    const callbackStarted = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const { server, baseUrl } = await startServer({
      requestRefresh: async ({ account }) => {
        callCount += 1;
        markStarted();
        await callbackGate;
        return { account, accepted: true, callCount };
      },
    });
    servers.push(server);

    const body = {
      account: "personal-primary",
      idempotency_key: "refresh-concurrent",
    };
    const firstPromise = postJson(`${baseUrl}/api/v1/refresh`, body);
    await callbackStarted;
    const secondPromise = postJson(`${baseUrl}/api/v1/refresh`, body);
    await new Promise<void>((resolve) => setImmediate(resolve));
    release();

    const [first, second] = await Promise.all([firstPromise, secondPromise]);
    expect(first.status).toBe(HTTP_STATUS.accepted);
    expect(second.status).toBe(HTTP_STATUS.accepted);
    await expect(first.json()).resolves.toEqual({
      account: "personal-primary",
      accepted: true,
      callCount: 1,
    });
    await expect(second.json()).resolves.toEqual({
      account: "personal-primary",
      accepted: true,
      callCount: 1,
    });
    expect(callCount).toBe(1);
  });

  it("rejects a changed body while the original key is still in flight", async () => {
    let callCount = 0;
    let release!: () => void;
    const callbackGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let markStarted!: () => void;
    const callbackStarted = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const { server, baseUrl } = await startServer({
      requestRefresh: async ({ account }) => {
        callCount += 1;
        if (callCount === 1) {
          markStarted();
          await callbackGate;
        }
        return { account, accepted: true, callCount };
      },
    });
    servers.push(server);

    const firstPromise = postJson(`${baseUrl}/api/v1/refresh`, {
      account: "personal-primary",
      idempotency_key: "refresh-in-flight-conflict",
    });
    await callbackStarted;

    const changed = await postJson(`${baseUrl}/api/v1/refresh`, {
      account: "other-account",
      idempotency_key: "refresh-in-flight-conflict",
    });
    expect(changed.status).toBe(HTTP_STATUS.conflict);
    expect(callCount).toBe(1);

    release();
    const first = await firstPromise;
    expect(first.status).toBe(HTTP_STATUS.accepted);
    expect(callCount).toBe(1);
  });

  it("fingerprints the request before a callback mutates nested input", async () => {
    let callCount = 0;
    const { server, baseUrl } = await startServer({
      recordManualObservation: async (input) => {
        callCount += 1;
        const observation = input.observation as
          | { nested?: { value?: string } }
          | undefined;
        if (observation?.nested !== undefined) {
          observation.nested.value = "callback-mutated";
        }
        return { accepted: true, callCount };
      },
    });
    servers.push(server);

    const body = {
      account: "personal-primary",
      observation: { nested: { value: "original" } },
      idempotency_key: "observation-mutation",
    };
    const first = await postJson(
      `${baseUrl}/api/v1/manual-observations`,
      body,
    );
    expect(first.status).toBe(HTTP_STATUS.accepted);

    const second = await postJson(
      `${baseUrl}/api/v1/manual-observations`,
      body,
    );
    expect(second.status).toBe(HTTP_STATUS.accepted);
    await expect(second.json()).resolves.toEqual({
      accepted: true,
      callCount: 1,
    });
    expect(callCount).toBe(1);
  });

  it("replays an immutable response snapshot after the callback result changes", async () => {
    let callCount = 0;
    const callbackResult = { nested: { value: "original" } };
    const { server, baseUrl } = await startServer({
      requestRefresh: async () => {
        callCount += 1;
        return callbackResult;
      },
    });
    servers.push(server);

    const body = {
      account: "personal-primary",
      idempotency_key: "response-snapshot",
    };
    const first = await postJson(`${baseUrl}/api/v1/refresh`, body);
    expect(first.status).toBe(HTTP_STATUS.accepted);
    await expect(first.json()).resolves.toEqual({
      nested: { value: "original" },
    });

    callbackResult.nested.value = "changed-after-callback";
    const second = await postJson(`${baseUrl}/api/v1/refresh`, body);
    expect(second.status).toBe(HTTP_STATUS.accepted);
    await expect(second.json()).resolves.toEqual({
      nested: { value: "original" },
    });
    expect(callCount).toBe(1);
  });

  it("releases a failed reservation so a later HTTP retry can claim the key", async () => {
    let callCount = 0;
    const { server, baseUrl } = await startServer({
      requestRefresh: async ({ account }) => {
        callCount += 1;
        if (callCount === 1) {
          throw new Error("callback failed");
        }
        return { account, accepted: true, callCount };
      },
    });
    servers.push(server);

    const body = {
      account: "personal-primary",
      idempotency_key: "failure-retry",
    };
    const first = await postJson(`${baseUrl}/api/v1/refresh`, body);
    expect(first.status).toBe(HTTP_STATUS.unavailable);
    expect(callCount).toBe(1);

    const retry = await postJson(`${baseUrl}/api/v1/refresh`, body);
    expect(retry.status).toBe(HTTP_STATUS.accepted);
    await expect(retry.json()).resolves.toEqual({
      account: "personal-primary",
      accepted: true,
      callCount: 2,
    });
    expect(callCount).toBe(2);
  });

  it("rejects pending waiters on failure without wedging the key", async () => {
    const cache = new IdempotencyCache(1);
    const owner = cache.reserve("refresh", "key", "fingerprint");
    const waiter = cache.reserve("refresh", "key", "fingerprint");
    const failure = new Error("callback failed");

    if (owner.kind !== "new" || waiter.kind !== "pending") {
      throw new Error("expected a new reservation and a pending waiter");
    }

    owner.fail(failure);
    await expect(waiter.promise).rejects.toBe(failure);

    const retry = cache.reserve("refresh", "key", "fingerprint");
    expect(retry.kind).toBe("new");
  });

  it("keeps active reservations within the configured cache bound", async () => {
    let callCount = 0;
    let release!: () => void;
    const callbackGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let markStarted!: () => void;
    const callbackStarted = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const { server, baseUrl } = await startServer(
      {
        requestRefresh: async ({ account }) => {
          callCount += 1;
          if (callCount === 1) {
            markStarted();
            await callbackGate;
          }
          return { account, accepted: true, callCount };
        },
      },
      { maxIdempotencyEntries: 1 },
    );
    servers.push(server);

    const firstPromise = postJson(`${baseUrl}/api/v1/refresh`, {
      account: "personal-primary",
      idempotency_key: "bounded-1",
    });
    await callbackStarted;

    const blocked = await postJson(`${baseUrl}/api/v1/refresh`, {
      account: "personal-primary",
      idempotency_key: "bounded-2",
    });
    expect(blocked.status).toBe(HTTP_STATUS.unavailable);
    expect(callCount).toBe(1);

    release();
    const first = await firstPromise;
    expect(first.status).toBe(HTTP_STATUS.accepted);

    const second = await postJson(`${baseUrl}/api/v1/refresh`, {
      account: "personal-primary",
      idempotency_key: "bounded-2",
    });
    expect(second.status).toBe(HTTP_STATUS.accepted);
    expect(callCount).toBe(2);

    const firstReplay = await postJson(`${baseUrl}/api/v1/refresh`, {
      account: "personal-primary",
      idempotency_key: "bounded-1",
    });
    expect(firstReplay.status).toBe(HTTP_STATUS.accepted);
    expect(callCount).toBe(3);
  });

  it("returns explicit 503 when a service is not injected", async () => {
    const { server, baseUrl } = await startServer({});
    servers.push(server);

    const response = await apiFetch(`${baseUrl}/api/v1/accounts`);
    expect(response.status).toBe(HTTP_STATUS.unavailable);
    const body = (await response.json()) as { error: { message: string } };
    expect(body.error.message).toBe(UNAVAILABLE_REASON);
  });

  it("returns 501 for reserved routes owned by other stages", async () => {
    const { server, baseUrl } = await startServer({});
    servers.push(server);

    const response = await apiFetch(`${baseUrl}/api/v1/schedule`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Usage-Capture-Token": AUTH_TOKEN,
      },
      body: JSON.stringify({ account: "personal-primary" }),
    });
    expect(response.status).toBe(HTTP_STATUS.notImplemented);
  });

  it("validates query parameters and rejects invalid timestamps", async () => {
    const { server, baseUrl } = await startServer({
      getUsage: async () => ({}),
    });
    servers.push(server);

    const response = await apiFetch(
      `${baseUrl}/api/v1/usage?account=personal-primary&from=not-a-date`,
    );
    expect(response.status).toBe(HTTP_STATUS.badRequest);
  });

  it("clamps pagination limits to the hard cap", async () => {
    const { server, baseUrl } = await startServer({
      getAttempts: async ({ limit }) => ({ limit }),
    });
    servers.push(server);

    const response = await apiFetch(
      `${baseUrl}/api/v1/attempts?account=personal-primary&limit=9999`,
    );
    expect(response.status).toBe(HTTP_STATUS.badRequest);
  });

  it("serves health endpoints without authentication", async () => {
    const { server, baseUrl } = await startServer({});
    servers.push(server);

    const live = await fetch(`${baseUrl}/health/live`);
    expect(live.status).toBe(HTTP_STATUS.ok);
    await expect(live.json()).resolves.toEqual({ status: "live" });

    const ready = await fetch(`${baseUrl}/health/ready`);
    expect(ready.status).toBe(HTTP_STATUS.ok);
    await expect(ready.json()).resolves.toEqual({ status: "ready" });
  });

  it("refuses to bind a non-loopback host without explicit opt-in", async () => {
    await expect(
      createLocalApiServer({
        authToken: AUTH_TOKEN,
        host: "0.0.0.0",
        services: {},
      }),
    ).rejects.toThrow(/allowRemoteBind/);
  });

  it("closes cleanly and releases the port", async () => {
    const { server, baseUrl } = await startServer({
      listAccounts: async () => ({ accounts: [] }),
    });
    const address = server.address();
    await server.close();

    const reopened = await createLocalApiServer({
      authToken: AUTH_TOKEN,
      host: address.host,
      port: address.port,
      services: {},
    });
    servers.push(reopened);

    const response = await apiFetch(`${baseUrl}/health/live`);
    expect(response.status).toBe(HTTP_STATUS.ok);
  });
});
