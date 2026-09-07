import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { bootstrapAccount } from "../../src/browser/bootstrap.js";
import { AccountConfig } from "../../src/config.js";
import { ADAPTER_VERSION } from "../../src/contracts/records.js";
import { assertNoSecrets } from "../../src/security/sanitizer.js";

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

describe("no secret or content persistence", () => {
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

  it("persists only sanitized identity metadata", async () => {
    const account = makeAccount(profilePath);
    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });
    expect(result.state).toBe("ready");

    const stateFile = join(stateDirectory, "bootstrap", "personal-primary.json");
    const persisted = JSON.parse(readFileSync(stateFile, "utf8"));

    expect(persisted.accountId).toBe("personal-primary");
    expect(persisted.state).toBe("ready");
    expect(persisted.adapterVersion).toBe(ADAPTER_VERSION);
    expect(persisted.identity.providerUserId).toBe("user-abc123");
    expect(persisted.identity.workspaceId).toBe("ws-xyz");
    expect(persisted.identity.quotaOwnerId).toBe("user-abc123");
    expect(persisted.identity.surface).toBe("chat");

    expect(persisted.identity.email).toBeUndefined();
    expect(persisted.identity.cookie).toBeUndefined();
    expect(persisted.identity.token).toBeUndefined();
    expect(persisted.identity.authorization).toBeUndefined();

    expect(() => assertNoSecrets(persisted)).not.toThrow();
  });

  it("never stores raw headers, cookies, tokens, or browser storage", async () => {
    const account = makeAccount(profilePath);
    await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });

    const stateFile = join(stateDirectory, "bootstrap", "personal-primary.json");
    const persisted = readFileSync(stateFile, "utf8");

    expect(persisted).not.toContain("bearer");
    expect(persisted).not.toContain("cookie");
    expect(persisted).not.toContain("set-cookie");
    expect(persisted).not.toContain("authorization");
    expect(persisted).not.toContain("session_token");
    expect(persisted).not.toContain("access_token");
    expect(persisted).not.toContain("refresh_token");
    expect(persisted).not.toContain("eyJ");
    expect(persisted).not.toContain("email");
  });
});
