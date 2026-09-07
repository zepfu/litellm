/**
 * Browser boundary for the ChatGPT Chat usage collector.
 *
 * Owns the Playwright persistent context bound to a dedicated, user-owned
 * browser profile. All application-issued requests are GET-only against an
 * explicit route allowlist. Interactive login is opt-in only. Browser state
 * (cookies, tokens, storage) never leaves the profile directory and is never
 * copied into persistence, logs, fixtures, or exports.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  readdirSync,
  realpathSync,
  statSync,
} from "node:fs";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { homedir } from "node:os";
import { chromium, type BrowserContext, type APIRequestContext } from "playwright";

import {
  type HistoryTransport,
  type HistoryTransportRequestOptions,
  AdapterError,
  assertAllowedRequest,
} from "../adapters/chatgpt/adapter.js";

export const CHATGPT_ORIGIN = "https://chatgpt.com";
export const DEFAULT_MAX_RESPONSE_BYTES = 32 * 1024 * 1024;
export const MAX_RESPONSE_BYTES = DEFAULT_MAX_RESPONSE_BYTES;

export const LIVE_GATE =
  "live Playwright collection requires Playwright and an existing dedicated browser profile. " +
  "Tokens and cookies remain in that profile and are never copied into persistence.";

const DISALLOWED_PROFILE_NAMES = new Set([
  "browser",
  "chrome",
  "chromium",
  "default",
  "default profile",
  "edge",
  "google-chrome",
  "guest profile",
  "microsoft-edge",
  "profile",
  "system profile",
  "user data",
  "user-data",
]);

const DEFAULT_BROWSER_PROFILE_ROOTS = [
  resolve(homedir(), ".config", "google-chrome"),
  resolve(homedir(), ".config", "chromium"),
  resolve(homedir(), ".config", "microsoft-edge"),
  resolve(homedir(), ".config", "BraveSoftware", "Brave-Browser"),
  resolve(homedir(), "Library", "Application Support", "Google", "Chrome"),
  resolve(homedir(), "Library", "Application Support", "Chromium"),
  resolve(homedir(), "Library", "Application Support", "Microsoft Edge"),
  resolve(homedir(), "AppData", "Local", "Google", "Chrome", "User Data"),
  resolve(homedir(), "AppData", "Local", "Chromium", "User Data"),
  resolve(homedir(), "AppData", "Local", "Microsoft", "Edge", "User Data"),
] as const;

const DISALLOWED_PROFILE_PATHS = new Set([
  resolve("/"),
  resolve("/tmp"),
  resolve("/var/tmp"),
]);

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
  maxResponseBytes?: number;
}

export function resolveDedicatedProfilePath(profilePath: string): string | null {
  if (
    typeof profilePath !== "string" ||
    profilePath.length === 0 ||
    profilePath !== profilePath.trim() ||
    profilePath.includes("\0")
  ) {
    return null;
  }
  let resolved: string;
  try {
    resolved = resolve(profilePath.replace(/^~(?=$|\/)/, homedir()));
  } catch {
    return null;
  }
  const canonical = canonicalizePath(resolved);
  if (canonical === null || !isDedicatedCanonicalProfilePath(canonical)) {
    return null;
  }
  return canonical;
}

export function dedicatedProfileReady(profilePath: string): boolean {
  const resolved = resolveDedicatedProfilePath(profilePath);
  if (resolved === null) {
    return false;
  }
  try {
    return existsSync(resolved) && statSync(resolved).isDirectory();
  } catch {
    return false;
  }
}

export function liveBrowserGate(config: BrowserConfig): string | null {
  const resolved = resolveDedicatedProfilePath(config.profilePath);
  if (resolved === null) {
    return "invalid_profile_path";
  }
  if (dedicatedProfileReady(resolved)) {
    return null;
  }
  if (!config.allowInteractiveLogin) {
    return "interactive_login_disabled";
  }
  return "dedicated_profile_missing";
}

export class PlaywrightTransport implements HistoryTransport {
  readonly requests: Array<{
    method: string;
    path: string;
    params: Record<string, unknown>;
  }> = [];

  private context: BrowserContext | null = null;
  private requestContext: APIRequestContext | null = null;
  private activeRequestController: AbortController | null = null;

  constructor(private readonly config: BrowserConfig) {}

  async request(
    method: string,
    path: string,
    params: Record<string, unknown> = {},
    options: HistoryTransportRequestOptions = {},
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
    const requestController = new AbortController();
    const onAbort = (): void => {
      requestController.abort();
    };
    if (options.signal?.aborted) {
      requestController.abort();
    } else {
      options.signal?.addEventListener("abort", onAbort, { once: true });
    }
    this.activeRequestController = requestController;
    try {
      response = await requestContext.get(url.toString(), {
        timeout: this.config.requestTimeoutSeconds * 1000,
        maxRedirects: 0,
        signal: requestController.signal,
      });
    } catch (error) {
      throw new AdapterError(
        `browser GET failed for ${path}`,
      );
    } finally {
      options.signal?.removeEventListener("abort", onAbort);
      if (this.activeRequestController === requestController) {
        this.activeRequestController = null;
      }
    }
    if (response.status() >= 300 && response.status() < 400) {
      throw new AdapterError(`browser redirect rejected for ${path}`);
    }
    return adaptResponse(response, this.config.maxResponseBytes);
  }

  async close(): Promise<void> {
    await this.cancel();
    this.requestContext = null;
    const context = this.context;
    this.context = null;
    if (context) {
      await context.close().catch(() => undefined);
    }
  }

  async cancel(): Promise<void> {
    this.activeRequestController?.abort();
  }

  private async ensureRequestContext(): Promise<APIRequestContext> {
    if (this.requestContext) {
      return this.requestContext;
    }
    const reason = liveBrowserGate(this.config);
    if (reason !== null) {
      throw new LiveBrowserUnavailable(`${LIVE_GATE} (${reason})`);
    }
    const profilePath = resolveDedicatedProfilePath(this.config.profilePath);
    if (profilePath === null || !dedicatedProfileReady(profilePath)) {
      throw new LiveBrowserUnavailable(`${LIVE_GATE} (invalid_profile_path)`);
    }
    try {
      this.context = await chromium.launchPersistentContext(
        profilePath,
        {
          headless: this.config.headless,
          acceptDownloads: false,
          chromiumSandbox: true,
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

export async function adaptResponse(response: {
  status(): number;
  headers(): Record<string, string>;
  body(): Promise<Uint8Array>;
}, maxResponseBytes = MAX_RESPONSE_BYTES): Promise<Record<string, unknown>> {
  const status = response.status();
  const headers = response.headers();
  const contentType = String(headers["content-type"] ?? "").split(";")[0]?.trim().toLowerCase() ?? "";
  const retryAfter = headers["retry-after"] ?? null;
  let payload: Record<string, unknown> = {};
  try {
    const declaredLength = responseByteLength(headers);
    if (declaredLength !== null && declaredLength > maxResponseBytes) {
      throw responseTooLargeError(declaredLength, maxResponseBytes);
    }
    const body = await response.body();
    if (body.byteLength > maxResponseBytes) {
      throw responseTooLargeError(body.byteLength, maxResponseBytes);
    }
    const parsed = JSON.parse(Buffer.from(body).toString("utf8")) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      payload = parsed as Record<string, unknown>;
    }
  } catch (error) {
    if (error instanceof AdapterError) {
      throw error;
    }
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
  const resolved = resolveDedicatedProfilePath(profilePath);
  if (resolved === null) {
    throw new LiveBrowserUnavailable(`${LIVE_GATE} (invalid_profile_path)`);
  }
  mkdirSync(resolved, { recursive: true, mode: 0o700 });
  try {
    chmodSync(resolved, 0o700);
  } catch {
    // Best-effort hardening; the directory remains user-owned.
  }
  return resolved;
}

export function ensureRestrictivePermissions(profilePath: string): void {
  const resolved = resolveDedicatedProfilePath(profilePath);
  if (resolved === null || !existsSync(resolved)) {
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

function canonicalizePath(path: string): string | null {
  let existingPath = path;
  const missingParts: string[] = [];
  try {
    while (!existsSync(existingPath)) {
      const parent = dirname(existingPath);
      if (parent === existingPath) {
        return null;
      }
      missingParts.unshift(basename(existingPath));
      existingPath = parent;
    }
    let canonical = realpathSync(existingPath);
    for (const part of missingParts) {
      canonical = resolve(canonical, part);
    }
    return canonical;
  } catch {
    return null;
  }
}

function isDedicatedCanonicalProfilePath(profilePath: string): boolean {
  const canonical = resolve(profilePath);
  if (
    canonical === resolve(homedir()) ||
    DISALLOWED_PROFILE_PATHS.has(canonical) ||
    DEFAULT_BROWSER_PROFILE_ROOTS.some((root) => isPathWithin(root, canonical))
  ) {
    return false;
  }
  const name = basename(canonical).toLowerCase();
  return (
    !DISALLOWED_PROFILE_NAMES.has(name) &&
    !/^profile \d+$/.test(name)
  );
}

function isPathWithin(parent: string, candidate: string): boolean {
  const childPath = relative(resolve(parent), resolve(candidate));
  return (
    childPath === "" ||
    (childPath !== ".." &&
      !childPath.startsWith(`..${sep}`) &&
      !isAbsolute(childPath))
  );
}

function responseByteLength(headers: Record<string, string>): number | null {
  const value = Object.entries(headers).find(
    ([key]) => key.toLowerCase() === "content-length",
  )?.[1];
  if (value === undefined || value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null;
}

function responseTooLargeError(
  actualBytes: number,
  maxResponseBytes: number,
): AdapterError {
  return new AdapterError(
    `browser response exceeds ${maxResponseBytes} byte limit (${actualBytes} bytes)`,
  );
}
