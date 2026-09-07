import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  bootstrapAccount,
  BootstrapResult,
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
        return {
          items: [],
          total: 0,
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

describe("bootstrap state transitions", () => {
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

  it("transitions to ready when identity matches and auth is valid", async () => {
    const account = makeAccount(profilePath);
    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });
    expect(result.state).toBe("ready");
    expect(result.identity.authState).toBe("ready");
    expect(result.liveVerification).toBe("passed");
    expect(result.interactiveLoginUsed).toBe(false);
    expect(result.notes).toEqual([]);
    expect(result.capabilities.adapterVersion).toBe(ADAPTER_VERSION);

    const stateFile = join(stateDirectory, "bootstrap", "personal-primary.json");
    expect(existsSync(stateFile)).toBe(true);
    const persisted = JSON.parse(readFileSync(stateFile, "utf8"));
    expect(persisted.state).toBe("ready");
    expect(persisted.identity.authState).toBe("ready");
  });

  it("transitions to identity_mismatch when observed identity differs", async () => {
    const account = makeAccount(profilePath);
    account.expectedProviderUserId = "user-def456";
    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });
    expect(result.state).toBe("identity_mismatch");
    expect(result.identity.authState).toBe("identity_mismatch");
    expect(result.identity.identityErrors).toContain("provider_user_id_mismatch");
    expect(result.liveVerification).toBe("blocked");
  });

  it("transitions to unconfigured when expected identity is missing", async () => {
    const account = makeAccount(profilePath);
    account.expectedProviderUserId = null;
    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });
    expect(result.state).toBe("unconfigured");
    expect(result.identity.authState).toBe("unconfigured");
    expect(result.identity.identityErrors).toContain(
      "missing_expected_provider_user_id",
    );
  });

  it("does not use interactive login unless explicitly invoked", async () => {
    const account = makeAccount(profilePath);
    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });
    expect(result.interactiveLoginUsed).toBe(false);
    expect(result.state).toBe("ready");
  });

  it("fails closed without creating a missing profile when login is not requested", async () => {
    const missingProfile = join(stateDirectory, "missing-profile");
    const account = makeAccount(missingProfile);
    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });

    expect(result.state).toBe("auth_required");
    expect(result.interactiveLoginUsed).toBe(false);
    expect(existsSync(missingProfile)).toBe(false);
  });

  it("keeps bootstrap state filenames inside the state directory", async () => {
    const account = makeAccount(profilePath);
    account.id = "../bootstrap-escape";

    await expect(
      bootstrapAccount(account, {
        interactiveLogin: false,
        stateDirectory,
      }),
    ).rejects.toThrow("bootstrap state filename must remain within the state directory");

    expect(existsSync(resolve(stateDirectory, "..", "bootstrap-escape.json"))).toBe(
      false,
    );
  });

  it("requires the Playwright adapter for live bootstrap", async () => {
    const account = makeAccount(profilePath);
    account.browser.adapter = "fixture_history";

    const result = await bootstrapAccount(account, {
      interactiveLogin: false,
      stateDirectory,
    });

    expect(result.state).toBe("browser_unavailable");
    expect(result.liveVerification).toBe("not_attempted");
    expect(result.notes).toContain(
      "bootstrap requires the playwright_persistent_context adapter",
    );
  });

});
