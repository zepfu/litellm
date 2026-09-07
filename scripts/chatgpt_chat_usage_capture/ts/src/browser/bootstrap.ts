/**
 * Bootstrap and capability inspection for the TypeScript collector.
 *
 * Bootstrap binds an account to a dedicated browser profile and verifies
 * identity, workspace, and quota owner before declaring the account ready.
 * Interactive login is explicit and opt-in only; no credentials, cookies,
 * tokens, raw headers, browser storage, or message content are persisted.
 */

import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import type { AccountConfig } from "../config.js";
import { ADAPTER_VERSION } from "../contracts/records.js";
import type { CapabilityRecord, IdentityRecord } from "../contracts/records.js";
import { assertNoSecrets } from "../security/sanitizer.js";
import {
  type BrowserConfig,
  LiveBrowserUnavailable,
  PlaywrightTransport,
  createProfileDirectory,
  dedicatedProfileReady,
  ensureRestrictivePermissions,
  liveBrowserGate,
} from "./session.js";
import {
  AuthenticationRequiredError,
  ChatGPTHistoryAdapter,
} from "../adapters/chatgpt/adapter.js";
import { FixtureTransport } from "../adapters/chatgpt/fixture-transport.js";
import { emptyCapabilities } from "../normalize/identity.js";

export type BootstrapState =
  | "unconfigured"
  | "ready"
  | "auth_required"
  | "identity_mismatch"
  | "browser_unavailable"
  | "schema_unavailable"
  | "paused";

export interface BootstrapResult {
  state: BootstrapState;
  accountId: string;
  profilePath: string;
  identity: IdentityRecord;
  capabilities: CapabilityRecord;
  interactiveLoginUsed: boolean;
  liveVerification: "passed" | "blocked" | "not_attempted";
  notes: string[];
}

export interface InspectCapabilitiesResult {
  accountId: string;
  identity: IdentityRecord;
  capabilities: CapabilityRecord;
  pages: Array<{
    scope: "active" | "archived";
    coverage: string;
    itemCount: number;
    warnings: string[];
  }>;
  state: BootstrapState;
  notes: string[];
}

export async function bootstrapAccount(
  account: AccountConfig,
  options: {
    interactiveLogin?: boolean;
    stateDirectory: string;
  },
): Promise<BootstrapResult> {
  const notes: string[] = [];
  const interactiveLogin = options.interactiveLogin === true;
  const profilePath = account.browser.profilePath;
  let interactiveLoginUsed = false;

  if (account.browser.adapter !== "playwright_persistent_context") {
    return {
      state: "browser_unavailable",
      accountId: account.id,
      profilePath,
      identity: emptyIdentity("browser_unavailable"),
      capabilities: emptyCapabilities(),
      interactiveLoginUsed,
      liveVerification: "not_attempted",
      notes: ["bootstrap requires the playwright_persistent_context adapter"],
    };
  }

  if (!profilePath) {
    return {
      state: "browser_unavailable",
      accountId: account.id,
      profilePath,
      identity: emptyIdentity("browser_unavailable"),
      capabilities: emptyCapabilities(),
      interactiveLoginUsed,
      liveVerification: "not_attempted",
      notes: ["no browser profile path configured"],
    };
  }

  const gate = liveBrowserGate(account.browser);
  if (gate !== null) {
    if (gate === "dedicated_profile_missing" && interactiveLogin) {
      createProfileDirectory(profilePath);
      ensureRestrictivePermissions(profilePath);
      notes.push("dedicated profile will be created by interactive login");
      await performInteractiveLogin(account.browser);
      interactiveLoginUsed = true;
    } else {
      const state = gate === "dedicated_profile_missing"
        ? "auth_required"
        : "browser_unavailable";
      const identity = emptyIdentity(state);
      persistBootstrapState(account.id, state, identity, options.stateDirectory);
      return {
        state,
        accountId: account.id,
        profilePath,
        identity,
        capabilities: emptyCapabilities(),
        interactiveLoginUsed,
        liveVerification: "not_attempted",
        notes: [
          gate === "dedicated_profile_missing"
            ? "dedicated browser profile does not exist yet; rerun bootstrap with --interactive-login to create it"
            : `live browser is unavailable (${gate})`,
        ],
      };
    }
  } else {
    ensureRestrictivePermissions(profilePath);
  }

  let adapter = new ChatGPTHistoryAdapter(
    new PlaywrightTransport(account.browser),
    expectedIdentity(account),
  );

  try {
    let identity = await adapter.inspectSessionIdentity();
    if (
      identity.authState === "auth_required" &&
      interactiveLogin &&
      !interactiveLoginUsed
    ) {
      await adapter.close();
      notes.push("opening dedicated browser for interactive login");
      await performInteractiveLogin(account.browser);
      interactiveLoginUsed = true;
      adapter = new ChatGPTHistoryAdapter(
        new PlaywrightTransport(account.browser),
        expectedIdentity(account),
      );
      identity = await adapter.inspectSessionIdentity();
    }

    const state = stateForIdentity(identity);
    assertNoSecrets(identity);
    persistBootstrapState(account.id, state, identity, options.stateDirectory);
    return {
      state,
      accountId: account.id,
      profilePath,
      identity,
      capabilities: adapter.capabilities,
      interactiveLoginUsed,
      liveVerification: state === "ready" ? "passed" : "blocked",
      notes,
    };
  } catch (error) {
    if (error instanceof LiveBrowserUnavailable) {
      return {
        state: "browser_unavailable",
        accountId: account.id,
        profilePath,
        identity: emptyIdentity("browser_unavailable"),
        capabilities: adapter.capabilities,
        interactiveLoginUsed,
        liveVerification: "blocked",
        notes: [...notes, error.message],
      };
    }
    throw error;
  } finally {
    await adapter.close();
  }
}

export async function inspectCapabilities(
  account: AccountConfig,
  options: { stateDirectory: string },
): Promise<InspectCapabilitiesResult> {
  const profilePath = account.browser.profilePath;

  if (account.browser.adapter !== "playwright_persistent_context") {
    return unavailableInspectResult(
      account,
      "live inspect-capabilities requires the playwright_persistent_context adapter",
    );
  }

  if (!profilePath || !dedicatedProfileReady(profilePath)) {
    return unavailableInspectResult(
      account,
      "dedicated browser profile is not available",
    );
  }

  const adapter = new ChatGPTHistoryAdapter(
    new PlaywrightTransport(account.browser),
    expectedIdentity(account),
  );
  return inspectCapabilitiesWithAdapter(account, adapter, options);
}

export async function inspectFixtureCapabilities(
  account: AccountConfig,
  options: { fixtureRoot: string; stateDirectory: string },
): Promise<InspectCapabilitiesResult> {
  if (account.browser.adapter !== "fixture_history") {
    return unavailableInspectResult(
      account,
      "fixture inspect-capabilities requires the fixture_history adapter",
    );
  }

  const adapter = new ChatGPTHistoryAdapter(
    new FixtureTransport(options.fixtureRoot),
    expectedIdentity(account),
  );
  return inspectCapabilitiesWithAdapter(account, adapter, options);
}

async function inspectCapabilitiesWithAdapter(
  account: AccountConfig,
  adapter: ChatGPTHistoryAdapter,
  options: { stateDirectory: string },
): Promise<InspectCapabilitiesResult> {
  const notes: string[] = [];
  try {
    const identity = await adapter.inspectSessionIdentity();
    const state = stateForIdentity(identity);
    assertNoSecrets(identity);
    if (state !== "ready") {
      persistBootstrapState(account.id, state, identity, options.stateDirectory);
      return {
        accountId: account.id,
        identity,
        capabilities: adapter.capabilities,
        pages: [],
        state,
        notes: ["capability inspection requires a verified ready identity"],
      };
    }

    const pages: InspectCapabilitiesResult["pages"] = [];
    for (const scope of ["active", "archived"] as const) {
      const page = await adapter.listConversations({
        archived: scope === "archived",
        offset: 0,
        limit: 100,
      });
      pages.push({
        scope,
        coverage: page.coverage,
        itemCount: page.items.length,
        warnings: page.warnings,
      });
    }

    assertNoSecrets(identity);
    persistBootstrapState(account.id, state, identity, options.stateDirectory);
    return {
      accountId: account.id,
      identity,
      capabilities: adapter.capabilities,
      pages,
      state,
      notes,
    };
  } catch (error) {
    if (error instanceof LiveBrowserUnavailable) {
      return {
        accountId: account.id,
        identity: emptyIdentity("browser_unavailable"),
        capabilities: adapter.capabilities,
        pages: [],
        state: "browser_unavailable",
        notes: [...notes, error.message],
      };
    }
    if (error instanceof AuthenticationRequiredError) {
      const identity = emptyIdentity("auth_required");
      persistBootstrapState(account.id, "auth_required", identity, options.stateDirectory);
      return {
        accountId: account.id,
        identity,
        capabilities: adapter.capabilities,
        pages: [],
        state: "auth_required",
        notes: [...notes, error.message],
      };
    }
    throw error;
  } finally {
    await adapter.close();
  }
}

function unavailableInspectResult(
  account: AccountConfig,
  note: string,
): InspectCapabilitiesResult {
  return {
    accountId: account.id,
    identity: emptyIdentity("browser_unavailable"),
    capabilities: emptyCapabilities(),
    pages: [],
    state: "browser_unavailable",
    notes: [note],
  };
}

function expectedIdentity(account: AccountConfig) {
  return {
    providerUserId: account.expectedProviderUserId,
    workspaceId: account.expectedWorkspaceId,
    quotaOwnerId: account.quotaOwnerId,
  };
}

function stateForIdentity(identity: IdentityRecord): BootstrapState {
  switch (identity.authState) {
    case "ready":
      return "ready";
    case "unconfigured":
      return "unconfigured";
    case "auth_required":
      return "auth_required";
    case "identity_mismatch":
      return "identity_mismatch";
    case "browser_unavailable":
      return "browser_unavailable";
    case "schema_unavailable":
      return "schema_unavailable";
    case "paused":
      return "paused";
  }
  return "paused";
}

async function performInteractiveLogin(config: BrowserConfig): Promise<void> {
  const resolved = resolve(
    config.profilePath.replace(/^~(?=$|\/)/, process.env.HOME ?? "/"),
  );
  const { chromium } = await import("playwright");
  const context = await chromium.launchPersistentContext(resolved, {
    headless: false,
    acceptDownloads: false,
  });
  try {
    const page = await context.newPage();
    await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded" });
    console.log(
      "Interactive login window opened in the dedicated browser profile. " +
        "Sign in, then press Enter here to continue verification.",
    );
    await waitForEnter();
  } finally {
    await context.close();
  }
}

function waitForEnter(): Promise<void> {
  return new Promise((resolvePromise) => {
    process.stdin.resume();
    process.stdin.once("data", () => {
      process.stdin.pause();
      resolvePromise();
    });
  });
}

function persistBootstrapState(
  accountId: string,
  state: BootstrapState,
  identity: IdentityRecord,
  stateDirectory: string,
): void {
  const dir = resolve(stateDirectory, "bootstrap");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const payload = {
    accountId,
    state,
    identity,
    adapterVersion: ADAPTER_VERSION,
    recordedAt: new Date().toISOString(),
  };
  assertNoSecrets(payload);
  writeFileSync(
    resolve(dir, `${accountId}.json`),
    JSON.stringify(payload, null, 2) + "\n",
    { encoding: "utf8", mode: 0o600 },
  );
}

function emptyIdentity(authState: IdentityRecord["authState"]): IdentityRecord {
  return {
    providerUserId: null,
    workspaceId: null,
    quotaOwnerId: null,
    surface: "unknown",
    authState,
    identityErrors: [],
  };
}
