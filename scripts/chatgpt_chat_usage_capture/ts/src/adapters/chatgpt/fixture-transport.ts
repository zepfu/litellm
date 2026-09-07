/**
 * Fixture-backed transport for deterministic, offline Stage-1 verification.
 * Serves reviewed synthetic pages from a directory of JSON fixtures.
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

import {
  AdapterError,
  isAllowedPath,
  ALLOWED_METHODS,
  SESSION_ROUTE,
  raiseIfAuthenticationRequired,
  raiseIfRateLimited,
} from "./adapter.js";
import type { HistoryTransport } from "./adapter.js";

interface FixtureManifestRoute {
  method?: string;
  path: string;
  params?: Record<string, unknown>;
  fixture: string;
}

interface FixtureManifest {
  routes: FixtureManifestRoute[];
}

export class FixtureTransport implements HistoryTransport {
  readonly requests: Array<{
    method: string;
    path: string;
    params: Record<string, unknown>;
  }> = [];

  constructor(private readonly root: string) {}

  async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> {
    const normalizedMethod = method.toUpperCase();
    if (!ALLOWED_METHODS.has(normalizedMethod)) {
      throw new AdapterError(`method not allowlisted: ${normalizedMethod} ${path}`);
    }
    if (!isAllowedPath(path)) {
      throw new AdapterError(`path not allowlisted: ${path}`);
    }
    this.requests.push({ method: normalizedMethod, path, params: { ...params } });
    const payload = loadFixture(this.root, normalizedMethod, path, params);
    raiseIfAuthenticationRequired(payload, path);
    if (String(payload.content_type ?? "").toLowerCase().startsWith("text/html")) {
      throw new AdapterError(`HTML login page for ${path}`);
    }
    raiseIfRateLimited(payload, path);
    return payload;
  }
}

function loadFixture(
  root: string,
  method: string,
  path: string,
  params: Record<string, unknown>,
): Record<string, unknown> {
  const manifestPath = resolve(root, "manifest.json");
  if (existsSync(manifestPath)) {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as FixtureManifest;
    for (const entry of manifest.routes ?? []) {
      if ((entry.method ?? "GET").toUpperCase() !== method) {
        continue;
      }
      if (entry.path !== path) {
        continue;
      }
      if (!paramsMatch(entry.params ?? {}, params)) {
        continue;
      }
      const payload = JSON.parse(
        readFileSync(resolve(root, entry.fixture), "utf8"),
      ) as Record<string, unknown>;
      payload.http_status = payload.http_status ?? 200;
      return payload;
    }
  }

  const archived = String(params.is_archived ?? "").toLowerCase();
  const offset = params.offset;
  const before = params.before;
  const candidates: string[] = [];
  if (path.includes("conversations") && path.split("/").length === 3) {
    const label = archived === "true" ? "archived" : "active";
    candidates.push(`conversations-${label}-offset-${offset ?? 0}.json`);
  }
  if (path.endsWith("/messages")) {
    const conversationId = path.split("/").at(-2) ?? "";
    const suffix = typeof before === "string" && before ? before : "latest";
    candidates.push(`messages-${conversationId}-${suffix}.json`);
  }
  if (path.startsWith("/backend-api/conversations/") && !path.endsWith("/messages")) {
    const conversationId = path.split("/").at(-1) ?? "";
    candidates.push(`conversation-${conversationId}.json`);
  }
  if (path.startsWith("/backend-api/conversation/") && !path.endsWith("/messages")) {
    const conversationId = path.split("/").at(-1) ?? "";
    candidates.push(`legacy-conversation-${conversationId}.json`);
  }
  if (path === SESSION_ROUTE) {
    candidates.push("session.json");
  }

  for (const candidate of candidates) {
    const filePath = resolve(root, candidate);
    if (existsSync(filePath)) {
      const payload = JSON.parse(readFileSync(filePath, "utf8")) as Record<string, unknown>;
      payload.http_status = payload.http_status ?? 200;
      return payload;
    }
  }
  throw new AdapterError(`no fixture for ${method} ${path} ${JSON.stringify(params)}`);
}

function paramsMatch(
  expected: Record<string, unknown>,
  actual: Record<string, unknown>,
): boolean {
  for (const [key, value] of Object.entries(expected)) {
    if (String(actual[key]) !== String(value)) {
      return false;
    }
  }
  return true;
}
