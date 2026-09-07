/**
 * Browser boundary for the ChatGPT Chat usage collector.
 *
 * Owns the Playwright persistent context bound to a dedicated, user-owned
 * browser profile. All application-issued requests are GET-only against an
 * explicit route allowlist. Interactive login is opt-in only. Browser state
 * (cookies, tokens, storage) never leaves the profile directory and is never
 * copied into persistence, logs, fixtures, or exports.
 */

import { existsSync, mkdirSync, chmodSync, readdirSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { chromium, type BrowserContext, type APIRequestContext } from "playwright";

import {
  type HistoryTransport,
  AdapterError,
  assertAllowedRequest,
} from "../adapters/chatgpt/adapter.js";

export const CHATGPT_ORIGIN = "https://chatgpt.com";

export const LIVE_GATE =
  "live Playwright collection requires Playwright and an existing dedicated browser profile. " +
  "Tokens and cookies remain in that profile and are never copied into persistence.";

export class LiveBrowserUnavailable extends AdapterError {
  constructor(message: string) {
    super(message);
    this.name = "LiveBrowserUnavailable";
  }
}

export interface BrowserConfig {
  adapter: "playwright_persistent_context" | "fixture_history";
  profilePath: string;
  headless: boolean;
  allowInteractiveLogin: boolean;
  requestTimeoutSeconds: number;
}

export function dedicatedProfileReady(profilePath: string): boolean {
  if (!profilePath) {
    return false;
  }
  let resolved: string;
  try {
    resolved = resolve(profilePath.replace(/^~(?=$|\/)/, homedir()));
  } catch {
    return false;
  }
  if (resolved === resolve(homedir())) {
    return false;
  }
  const name = resolved.split("/").at(-1)?.toLowerCase() ?? "";
  if (["browser", "default", "chrome", "chromium"].includes(name)) {
    return false;
  }
  return existsSync(resolved) && statSync(resolved).isDirectory();
}

export function liveBrowserGate(config: BrowserConfig): string | null {
  if (!dedicatedProfileReady(config.profilePath)) {
    if (!config.allowInteractiveLogin) {
      return "interactive_login_disabled";
    }
    return "dedicated_profile_missing";
  }
  return null;
}

export class PlaywrightTransport implements HistoryTransport {
  readonly requests: Array<{
    method: string;
    path: string;
    params: Record<string, unknown>;
  }> = [];

  private context: BrowserContext | null = null;
  private requestContext: APIRequestContext | null = null;

  constructor(private readonly config: BrowserConfig) {}

  async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
  ): Promise<Record<string, unknown>> {
    assertAllowedRequest(method, path);
    const normalizedMethod = method.toUpperCase();
    const query: Record<string, string> = {};
    for (const [key, value] of Object.entries(params)) {
      query[key] = String(value);
    }
    this.requests.push({ method: normalizedMethod, path, params: { ...params } });
    const requestContext = await this.ensureRequestContext();
    const url = new URL(`${CHATGPT_ORIGIN}${path}`);
    for (const [key, value] of Object.entries(query)) {
      url.searchParams.set(key, value);
    }
    let response;
    try {
      response = await requestContext.get(url.toString(), {
        timeout: this.config.requestTimeoutSeconds * 1000,
      });
    } catch (error) {
      throw new AdapterError(
        `browser GET failed for ${path}`,
      );
    }
    return adaptResponse(response);
  }

  async close(): Promise<void> {
    this.requestContext = null;
    const context = this.context;
    this.context = null;
    if (context) {
      await context.close().catch(() => undefined);
    }
  }

  private async ensureRequestContext(): Promise<APIRequestContext> {
    if (this.requestContext) {
      return this.requestContext;
    }
    const reason = liveBrowserGate(this.config);
    if (reason !== null) {
      throw new LiveBrowserUnavailable(`${LIVE_GATE} (${reason})`);
    }
    try {
      this.context = await chromium.launchPersistentContext(
        resolve(this.config.profilePath.replace(/^~(?=$|\/)/, homedir())),
        {
          headless: this.config.headless,
          acceptDownloads: false,
        },
      );
      this.requestContext = this.context.request;
    } catch (error) {
      await this.close();
      throw new LiveBrowserUnavailable(`${LIVE_GATE} (launch_failed)`);
    }
    return this.requestContext;
  }
}

async function adaptResponse(response: {
  status(): number;
  headers(): Record<string, string>;
  json(): Promise<unknown>;
}): Promise<Record<string, unknown>> {
  const status = response.status();
  const headers = response.headers();
  const contentType = String(headers["content-type"] ?? "").split(";")[0]?.trim().toLowerCase() ?? "";
  const retryAfter = headers["retry-after"] ?? null;
  let payload: Record<string, unknown> = {};
  try {
    const parsed = await response.json();
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      payload = parsed as Record<string, unknown>;
    }
  } catch {
    payload = {};
  }
  return {
    ...payload,
    http_status: status,
    content_type: contentType,
    retry_after: retryAfter,
  };
}

export function createProfileDirectory(profilePath: string): string {
  const resolved = resolve(profilePath.replace(/^~(?=$|\/)/, homedir()));
  mkdirSync(resolved, { recursive: true, mode: 0o700 });
  try {
    chmodSync(resolved, 0o700);
  } catch {
    // Best-effort hardening; the directory remains user-owned.
  }
  return resolved;
}

export function ensureRestrictivePermissions(profilePath: string): void {
  const resolved = resolve(profilePath.replace(/^~(?=$|\/)/, homedir()));
  if (!existsSync(resolved)) {
    return;
  }
  for (const entry of readdirSync(resolved, { withFileTypes: true })) {
    const entryPath = resolve(resolved, entry.name);
    if (entry.isDirectory()) {
      try {
        chmodSync(entryPath, 0o700);
      } catch {
        // Ignore individual permission failures.
      }
    }
  }
}
