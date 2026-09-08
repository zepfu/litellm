/**
 * Browser boundary for the ChatGPT Chat usage collector.
 *
 * Owns either a Playwright persistent context bound to a dedicated,
 * user-owned browser profile or one bounded attach-only CDP history
 * observation. All application-issued requests are GET-only against an
 * explicit route allowlist. Interactive login is opt-in only. Browser state
 * (cookies, tokens, storage) never leaves the profile directory and is never
 * copied into persistence, logs, fixtures, or exports.
 */

import { existsSync, mkdirSync, chmodSync, readdirSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { chromium, type BrowserContext, type APIRequestContext } from "playwright";

import { createHash } from "node:crypto";
import { MODERN_INDEX } from "../adapters/chatgpt/adapter.js";
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

export class HistoryObservationBoundaryError extends AdapterError {
  constructor(message: string) {
    super(message);
    this.name = "HistoryObservationBoundaryError";
  }
}

export interface BrowserConfig {
  adapter: "playwright_persistent_context" | "fixture_history";
  profilePath: string;
  headless: boolean;
  allowInteractiveLogin: boolean;
  requestTimeoutSeconds: number;
}

export interface NativeHistoryObservationOptions {
  cdpEndpoint: string;
  pageTargetId: string;
  expectedAccountHash: string;
  lifetimeMs: number;
  maxResponseBytes: number;
}

export interface NativeHistoryObservation {
  accountIdentityVerified: boolean;
  accountIdentitySource: "native_request_header" | null;
  requestResponseCorrelated: boolean;
  requestMethod: string | null;
  pageTargetIdMatched: boolean;
  httpStatus: number | null;
  retryAfterSeconds: number | null;
  browserChallenge: boolean;
  requestCount: number;
  responseBytes: number | null;
  modelFieldPresence: {
    requestedModel: boolean | null;
    recordedFinalModel: boolean | null;
  };
  warnings: string[];
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

function headerValue(headers: unknown, name: string): string | null {
  if (!headers || typeof headers !== "object" || Array.isArray(headers)) {
    return null;
  }
  for (const [key, value] of Object.entries(headers as Record<string, unknown>)) {
    if (key.toLowerCase() === name && typeof value === "string") {
      return value;
    }
  }
  return null;
}

function canonicalAccountHash(value: string): string | null {
  const normalized = value.trim();
  if (!normalized || normalized === "default") {
    return null;
  }
  return createHash("sha256").update(normalized, "utf8").digest("hex").slice(0, 12);
}

function retryAfterSeconds(headers: Record<string, string>): number | null {
  const value = headerValue(headers, "retry-after");
  if (value === null) {
    return null;
  }
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

async function responseModelFieldPresence(
  response: import("playwright").APIResponse,
): Promise<{ requestedModel: boolean | null; recordedFinalModel: boolean | null }> {
  let parsed: unknown;
  try {
    parsed = await response.json();
  } catch {
    return { requestedModel: null, recordedFinalModel: null };
  }
  let requested: boolean | null = null;
  let recorded: boolean | null = null;
  let objectCount = 0;
  const inspect = (value: unknown): void => {
    if (value === null || typeof value !== "object" || objectCount >= 512) {
      return;
    }
    objectCount += 1;
    if (Array.isArray(value)) {
      value.slice(0, 64).forEach(inspect);
      return;
    }
    const record = value as Record<string, unknown>;
    const requestedValue = record.requested_model ?? record.requestedModel;
    const recordedValue = record.model_slug ?? record.recorded_model ?? record.default_model_slug;
    if (requested !== true && (typeof requestedValue === "string" || requestedValue === null)) {
      requested = typeof requestedValue === "string" ? true : requested;
    }
    if (recorded !== true && (typeof recordedValue === "string" || recordedValue === null)) {
      recorded = typeof recordedValue === "string" ? true : recorded;
    }
  };
  inspect(parsed);
  return { requestedModel: requested, recordedFinalModel: recorded };
}

/**
 * Perform one GET-only native history index observation in a fresh page owned
 * by the caller's attached browser context. Authentication, correlation, and
 * size checks fail closed; the attached browser and its unrelated pages are
 * never closed.
 */
export async function observeNativeHistoryIndex(
  config: BrowserConfig,
  options: NativeHistoryObservationOptions,
): Promise<NativeHistoryObservation> {
  const accountHash = canonicalAccountHash(options.expectedAccountHash);
  if (!/^[0-9a-f]{12}$/.test(options.expectedAccountHash) || accountHash !== options.expectedAccountHash) {
    throw new HistoryObservationBoundaryError(
      "history observation requires the canonical inventory account hash",
    );
  }
  if (config.adapter !== "playwright_persistent_context") {
    throw new HistoryObservationBoundaryError(
      "history observation requires the live browser adapter",
    );
  }
  if (!Number.isFinite(options.lifetimeMs) || options.lifetimeMs <= 0 || options.lifetimeMs > 30_000) {
    throw new HistoryObservationBoundaryError("history observation lifetime is invalid");
  }
  if (!Number.isInteger(options.maxResponseBytes) || options.maxResponseBytes <= 0 || options.maxResponseBytes > 2_097_152) {
    throw new HistoryObservationBoundaryError("history observation response budget is invalid");
  }

  const { chromium } = await import("playwright");
  const browser = await chromium.connectOverCDP(options.cdpEndpoint, {
    timeout: Math.min(options.lifetimeMs, 10_000),
  });
  let page: import("playwright").Page | null = null;
  const observations: string[] = [];
  let requestCount = 0;
  const startedAt = Date.now();
  let pageTargetIdMatched = false;

  try {
    const context = browser.contexts().find((candidate) =>
      candidate.pages().some((candidatePage) => candidatePage.url().startsWith(CHATGPT_ORIGIN)),
    );
    if (!context) {
      throw new HistoryObservationBoundaryError("attached browser has no context");
    }
    if (
      !context.pages().some((candidatePage) =>
        candidatePage.url().startsWith(CHATGPT_ORIGIN),
      )
    ) {
      throw new HistoryObservationBoundaryError(
        "attached browser has no existing ChatGPT page binding",
      );
    }
    page = await context.newPage();
    page.setDefaultTimeout(Math.min(options.lifetimeMs, 15_000));
    const pageCdpSession = await context.newCDPSession(page);
    let ownTargetId: string | null = null;
    try {
      const target = pageCdpSession.send("Target.getTargetInfo") as { targetInfo?: { targetId?: unknown } };
      const targetId = target.targetInfo?.targetId;
      ownTargetId = typeof targetId === "string" ? targetId : null;
    } catch {
      ownTargetId = null;
    } finally {
      await pageCdpSession.detach().catch(() => undefined);
    }
    if (ownTargetId !== options.pageTargetId) {
      throw new HistoryObservationBoundaryError(
        "owned page target id did not match the configured CDP binding",
      );
    }
    pageTargetIdMatched = true;

    const requestHeaderPromise = new Promise<string | null>((resolvePromise, rejectPromise) => {
      const timer = setTimeout(() => resolvePromise(null), Math.max(1, options.lifetimeMs - (Date.now() - startedAt)));
      page!.once("request", (request: { url(): string; headers(): Record<string, string> }) => {
        requestCount += 1;
        if (new URL(request.url()).pathname !== MODERN_INDEX) {
          return;
        }
        const account = headerValue(request.headers(), "chatgpt-account-id");
        const hash = account === null ? null : canonicalAccountHash(account);
        clearTimeout(timer);
        if (hash === options.expectedAccountHash) {
          resolvePromise(account);
        } else {
          rejectPromise(new HistoryObservationBoundaryError("history account identity mismatch"));
        }
      });
    });

    const indexUrl = new URL(`${CHATGPT_ORIGIN}${MODERN_INDEX}`);
    indexUrl.searchParams.set("offset", "0");
    indexUrl.searchParams.set("limit", "1");
    indexUrl.searchParams.set("order", "updated");
    indexUrl.searchParams.set("is_archived", "false");
    const response = await Promise.race([
      page.request.get(indexUrl.toString(), {
        headers: { accept: "application/json" },
        timeout: Math.max(1, options.lifetimeMs - (Date.now() - startedAt)),
        maxRedirects: 0,
      }),
      new Promise<never>((_, rejectPromise) =>
        setTimeout(
          () => rejectPromise(new HistoryObservationBoundaryError("history observation deadline expired")),
          Math.max(1, options.lifetimeMs - (Date.now() - startedAt)),
        ),
      ),
    ]);
    const accountFromRequest = await requestHeaderPromise;
    if (accountFromRequest === null) {
      throw new HistoryObservationBoundaryError("history request identity evidence is missing");
    }
    if (response.status() === 401 || response.status() === 403) {
      throw new HistoryObservationBoundaryError("history observation requires authentication");
    }
    if (response.status() === 429) {
      throw new HistoryObservationBoundaryError("history observation is rate limited");
    }
    if (headerValue(response.headers(), "cf-mitigated") === "challenge") {
      throw new HistoryObservationBoundaryError("history observation encountered a browser challenge");
    }
    if (!response.ok() || [204, 206, 304].includes(response.status())) {
      throw new HistoryObservationBoundaryError("history observation returned an invalid status");
    }
    const bodyBuffer = await response.body();
    const contentLength = Number(headerValue(response.headers(), "content-length") ?? bodyBuffer.byteLength);
    const responseBytes = Number.isFinite(contentLength)
      ? Math.max(bodyBuffer.byteLength, contentLength)
      : bodyBuffer.byteLength;
    if (bodyBuffer.byteLength > options.maxResponseBytes || responseBytes > options.maxResponseBytes) {
      throw new HistoryObservationBoundaryError("history observation exceeded its response budget");
    }
    if (String(headerValue(response.headers(), "content-type") ?? "").split(";")[0]?.trim().toLowerCase() !== "application/json") {
      throw new HistoryObservationBoundaryError("history observation returned non-JSON content");
    }
    const modelFields = await responseModelFieldPresence(response);
    return {
      accountIdentityVerified: true,
      accountIdentitySource: "native_request_header",
      requestResponseCorrelated: true,
      requestMethod: "GET",
      pageTargetIdMatched,
      httpStatus: response.status(),
      retryAfterSeconds: retryAfterSeconds(response.headers()),
      browserChallenge: false,
      requestCount,
      responseBytes,
      modelFieldPresence: modelFields,
      warnings: observations,
    };
  } catch (error) {
    if (error instanceof HistoryObservationBoundaryError) {
      throw error;
    }
    throw new HistoryObservationBoundaryError("history observation boundary failed");
  } finally {
    if (page) {
      await page.close().catch(() => undefined);
    }
    await browser.close().catch(() => undefined);
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
