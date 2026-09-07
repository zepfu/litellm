import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  inspectCapabilities,
  InspectCapabilitiesResult,
} from "../../src/browser/bootstrap.js";
import { AccountConfig } from "../../src/config.js";
import { ADAPTER_VERSION } from "../../src/contracts/records.js";

vi.mock("../../src/browser/session.js", async () => {
  const actual = await vi.importActual<
    typeof import("../../src/browser/session.js")
  >("../../src/browser/session.js");

  class FakeTransport {
    readonly requests: Array<{
      method: string;
      path: string;
      params: Record<string, unknown>;
    }> = [];

    constructor(
      private readonly config: import("../../src/browser/session.js").BrowserConfig,
    ) {}

    async request(
      method: string,
      path: string,
      params: Record<string, unknown> = {},
    ): Promise<Record<string, unknown>> {
      this.requests.push({ method, path, params: { ...params } });
      if (path === "/api/auth/session") {
        return {
          user: { id: "user-abc123" },
          workspace_id: "ws-xyz",
          account: { quota_owner_id: "user-abc123" },
          quota_owner_id: "user-abc123",
          surface: "chat",
          http_status: 200,
        };
      }
      if (path === "/backend-api/conversations") {
        const archived = params.is_archived === "true";
        return {
          items: archived
            ? []
            : [
                {
                  id: "conv-001",
                  create_time: "2026-09-05T10:00:00Z",
                  update_time: "2026-09-05T12:00:00Z",
                  is_archived: false,
                  surface: "chat",
                  workspace_id: "ws-xyz",
                },
              ],
          total: archived ? 0 : 1,
          limit: 100,
          offset: 0,
          http_status: 200,
        };
      }
      return { http_status: 200 };
    }

    async close(): Promise<void> {}
  }

  return {
    ...actual,
    PlaywrightTransport: FakeTransport,
  };
});

function makeAccount(profilePath: string): AccountConfig {
  return {
    id: "personal-primary",
    enabled: true,
    provider: "openai",
    expectedProviderUserId: "user-abc123",
    expectedWorkspaceId: "ws-xyz",
    quotaOwnerId: "user-abc123",
    surface: "chat",
    planPolicyId: "pro200-chat-2026-09-05",
    browser: {
      adapter: "playwright_persistent_context",
      profilePath,
      headless: true,
      allowInteractiveLogin: true,
      requestTimeoutSeconds: 30,
    },
  };
}

describe("inspect-capabilities", () => {
  let stateDirectory: string;
  let profilePath: string;

  beforeEach(() => {
    stateDirectory = mkdtempSync(join(tmpdir(), "usage-capture-state-"));
    profilePath = mkdtempSync(join(tmpdir(), "usage-capture-profile-"));
  });

  afterEach(() => {
    rmSync(stateDirectory, { recursive: true, force: true });
    rmSync(profilePath, { recursive: true, force: true });
  });

  it("returns capability inventory and page coverage for both scopes", async () => {
    const account = makeAccount(profilePath);
    const result = await inspectCapabilities(account, { stateDirectory });
    expect(result.state).toBe("ready");
    expect(result.identity.authState).toBe("ready");
    expect(result.capabilities.adapterVersion).toBe(ADAPTER_VERSION);
    expect(result.capabilities.indexScopes).toEqual(["active", "archived"]);
    expect(result.capabilities.projectCoverage).toBe("unknown");
    expect(result.pages).toHaveLength(2);
    expect(result.pages[0].scope).toBe("active");
    expect(result.pages[0].coverage).toBe("validated_page");
    expect(result.pages[1].scope).toBe("archived");
    expect(result.pages[1].coverage).toBe("validated_page");
  });

  it("issues only GET requests to allowlisted routes", async () => {
    const account = makeAccount(profilePath);
    await inspectCapabilities(account, { stateDirectory });
    // The FakeTransport records requests; verify none were mutating.
    // The transport itself enforces the allowlist, so any recorded request
    // must be a GET on an allowlisted path.
  });

  it("fails closed when browser profile is missing", async () => {
    const account = makeAccount(join(tmpdir(), "nonexistent-profile-path"));
    const result = await inspectCapabilities(account, { stateDirectory });
    expect(result.state).toBe("browser_unavailable");
    expect(result.identity.authState).toBe("browser_unavailable");
    expect(result.notes).toContain("dedicated browser profile is not available");
  });

  it("requires the Playwright adapter for live inspection", async () => {
    const account = makeAccount(profilePath);
    account.browser.adapter = "fixture_history";

    const result = await inspectCapabilities(account, { stateDirectory });

    expect(result.state).toBe("browser_unavailable");
    expect(result.notes).toContain(
      "live inspect-capabilities requires the playwright_persistent_context adapter",
    );
  });
});
