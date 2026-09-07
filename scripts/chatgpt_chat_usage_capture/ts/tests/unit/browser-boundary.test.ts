import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const playwrightMocks = vi.hoisted(() => ({
  launchPersistentContext: vi.fn(),
}));

vi.mock("playwright", () => ({
  chromium: {
    launchPersistentContext: playwrightMocks.launchPersistentContext,
  },
}));

import {
  adaptResponse,
  CHATGPT_ORIGIN,
  createProfileDirectory,
  dedicatedProfileReady,
  LiveBrowserUnavailable,
  MAX_RESPONSE_BYTES,
  PlaywrightTransport,
} from "../../src/browser/session.js";
import { SESSION_ROUTE } from "../../src/adapters/chatgpt/adapter.js";

const MIB = 1024 * 1024;

function makeConfig(profilePath: string) {
  return {
    adapter: "playwright_persistent_context" as const,
    profilePath,
    headless: true,
    allowInteractiveLogin: false,
    requestTimeoutSeconds: 1,
    maxResponseBytes: MAX_RESPONSE_BYTES,
  };
}

function makeResponse(
  status: number,
  body: Uint8Array,
  headers: Record<string, string> = {},
) {
  const readBody = vi.fn().mockResolvedValue(body);
  return {
    response: {
      status: () => status,
      headers: () => headers,
      body: readBody,
    },
    readBody,
  };
}

function installContext(response: unknown) {
  const get = vi.fn().mockResolvedValue(response);
  const context = {
    request: { get },
    close: vi.fn().mockResolvedValue(undefined),
  };
  playwrightMocks.launchPersistentContext.mockResolvedValue(context);
  return { context, get };
}

describe("browser boundary", () => {
  let tempRoot: string;

  beforeEach(() => {
    tempRoot = mkdtempSync(join(tmpdir(), "chatgpt-browser-boundary-"));
    playwrightMocks.launchPersistentContext.mockReset();
  });

  afterEach(() => {
    rmSync(tempRoot, { recursive: true, force: true });
  });

  it("should reject invalid and canonical non-dedicated profile paths", () => {
    const dedicated = join(tempRoot, "personal-primary");
    const defaultProfile = join(tempRoot, "Default");
    const defaultProfileLink = join(tempRoot, "dedicated-link");
    const profileFile = join(tempRoot, "profile-file");
    mkdirSync(dedicated);
    mkdirSync(defaultProfile);
    symlinkSync(defaultProfile, defaultProfileLink, "dir");
    writeFileSync(profileFile, "not a directory");

    expect(dedicatedProfileReady(dedicated)).toBe(true);
    expect(dedicatedProfileReady(defaultProfile)).toBe(false);
    expect(dedicatedProfileReady(defaultProfileLink)).toBe(false);
    expect(dedicatedProfileReady(profileFile)).toBe(false);
    expect(dedicatedProfileReady("/")).toBe(false);
    expect(() => createProfileDirectory(join(tempRoot, "Profile 1"))).toThrow(
      "invalid_profile_path",
    );
  });

  it("should require the Chromium sandbox and disable API redirects", async () => {
    const profilePath = join(tempRoot, "personal-primary");
    mkdirSync(profilePath);
    const body = Buffer.from(JSON.stringify({ ok: true }));
    const { response } = makeResponse(200, body, {
      "content-type": "application/json",
      "content-length": String(body.byteLength),
    });
    const { get } = installContext(response);
    const transport = new PlaywrightTransport(makeConfig(profilePath));

    await expect(transport.request("GET", SESSION_ROUTE)).resolves.toMatchObject({
      ok: true,
    });
    expect(get).toHaveBeenCalledWith(`${CHATGPT_ORIGIN}${SESSION_ROUTE}`, {
      timeout: 1000,
      maxRedirects: 0,
    });
    expect(playwrightMocks.launchPersistentContext).toHaveBeenCalledWith(
      profilePath,
      expect.objectContaining({
        acceptDownloads: false,
        chromiumSandbox: true,
      }),
    );
  });

  it("should use the 32 MiB default and accept a declared 16 MiB response without allocating it", async () => {
    const profilePath = join(tempRoot, "personal-primary");
    mkdirSync(profilePath);
    const body = Buffer.from(JSON.stringify({ ok: true }));
    const { response } = makeResponse(200, body, {
      "content-type": "application/json",
      "content-length": String(16 * MIB),
    });
    installContext(response);
    const transport = new PlaywrightTransport(makeConfig(profilePath));

    expect(MAX_RESPONSE_BYTES).toBe(32 * MIB);
    await expect(transport.request("GET", SESSION_ROUTE)).resolves.toMatchObject({
      ok: true,
    });
  });

  it("should pass an overridden response ceiling to the production transport", async () => {
    const profilePath = join(tempRoot, "personal-primary");
    mkdirSync(profilePath);
    const { response, readBody } = makeResponse(
      200,
      Buffer.from("{}"),
      { "content-length": String(16 * MIB) },
    );
    installContext(response);
    const transport = new PlaywrightTransport({
      ...makeConfig(profilePath),
      maxResponseBytes: 8 * MIB,
    });

    await expect(transport.request("GET", SESSION_ROUTE)).rejects.toThrow(
      `${8 * MIB} byte limit`,
    );
    expect(readBody).not.toHaveBeenCalled();
  });

  it("should fail closed on an HTTP redirect without parsing its body", async () => {
    const profilePath = join(tempRoot, "personal-primary");
    mkdirSync(profilePath);
    const { response, readBody } = makeResponse(
      302,
      Buffer.from("redirect"),
      { location: "https://example.invalid/" },
    );
    installContext(response);
    const transport = new PlaywrightTransport(makeConfig(profilePath));

    await expect(transport.request("GET", SESSION_ROUTE)).rejects.toThrow(
      "browser redirect rejected",
    );
    expect(readBody).not.toHaveBeenCalled();
  });

  it("should reject a declared response larger than the byte ceiling before reading it", async () => {
    const { response, readBody } = makeResponse(
      200,
      Buffer.from("{}"),
      {
        "content-length": String(MAX_RESPONSE_BYTES + 1),
      },
    );

    await expect(adaptResponse(response)).rejects.toThrow(
      `${MAX_RESPONSE_BYTES} byte limit`,
    );
    expect(readBody).not.toHaveBeenCalled();
  });

  it("should reject an actual response larger than the byte ceiling before JSON parsing", async () => {
    const body = { byteLength: MAX_RESPONSE_BYTES + 1 } as Uint8Array;
    const { response } = makeResponse(200, body);

    await expect(adaptResponse(response)).rejects.toThrow(
      `${MAX_RESPONSE_BYTES} byte limit`,
    );
  });

  it("should fail closed when the sandboxed Chromium launch is unavailable", async () => {
    const profilePath = join(tempRoot, "personal-primary");
    mkdirSync(profilePath);
    playwrightMocks.launchPersistentContext.mockRejectedValue(
      new Error("sandbox unavailable"),
    );
    const transport = new PlaywrightTransport(makeConfig(profilePath));

    await expect(transport.request("GET", SESSION_ROUTE)).rejects.toBeInstanceOf(
      LiveBrowserUnavailable,
    );
    expect(playwrightMocks.launchPersistentContext).toHaveBeenCalledWith(
      profilePath,
      expect.objectContaining({ chromiumSandbox: true }),
    );
  });
});
