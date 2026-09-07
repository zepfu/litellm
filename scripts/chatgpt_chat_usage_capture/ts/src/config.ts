/**
 * Stage-1 configuration loader. Only the subset of the spec configuration
 * required for bootstrap and inspect-capabilities is modeled here; scheduler,
 * accounting, and ledger configuration belong to later stages.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";

import type { BrowserConfig } from "./browser/session.js";

export interface AccountConfig {
  id: string;
  enabled: boolean;
  provider: string;
  expectedProviderUserId: string | null;
  expectedWorkspaceId: string | null;
  quotaOwnerId: string | null;
  surface: string;
  planPolicyId: string;
  browser: BrowserConfig;
}

export interface ApplicationConfig {
  reportTimezone: string;
  stateDirectory: string;
}

export interface Stage1Config {
  schemaVersion: number;
  application: ApplicationConfig;
  accounts: AccountConfig[];
}

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

const DEFAULT_TIMEZONE = "America/New_York";
const DEFAULT_STATE_DIRECTORY = "./state";

export function loadConfig(path: string): Stage1Config {
  const resolvedPath = resolve(path);
  if (!existsSync(resolvedPath)) {
    throw new ConfigError(`config not found: ${resolvedPath}`);
  }
  let raw: Record<string, unknown>;
  try {
    const parsed = JSON.parse(readFileSync(resolvedPath, "utf8")) as unknown;
    if (!isRecord(parsed)) {
      throw new ConfigError("config root must be an object");
    }
    raw = parsed;
  } catch (error) {
    if (error instanceof ConfigError) {
      throw error;
    }
    throw new ConfigError(
      `invalid JSON config: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (raw.schema_version !== 1) {
    throw new ConfigError(`unsupported schema_version: ${raw.schema_version}`);
  }
  const applicationRaw = (raw.application as Record<string, unknown> | undefined) ?? {};
  const accountsRaw = Array.isArray(raw.accounts) ? raw.accounts : [];
  const accounts: AccountConfig[] = accountsRaw.map((entry, index) => {
    if (!isRecord(entry)) {
      throw new ConfigError(`accounts[${index}] must be an object`);
    }
    const account = entry;
    const browserRaw = (account.browser as Record<string, unknown> | undefined) ?? {};
    if (
      browserRaw.adapter !== "playwright_persistent_context" &&
      browserRaw.adapter !== "fixture_history"
    ) {
      throw new ConfigError(
        `accounts[${index}].browser.adapter must be playwright_persistent_context or fixture_history`,
      );
    }
    const id = optionalString(account.id);
    if (id === null) {
      throw new ConfigError(`accounts[${index}].id must be a non-empty string`);
    }
    return {
      id,
      enabled: account.enabled !== false,
      provider: String(account.provider ?? "openai"),
      expectedProviderUserId: optionalString(account.expected_provider_user_id),
      expectedWorkspaceId: optionalString(account.expected_workspace_id),
      quotaOwnerId: optionalString(account.quota_owner_id),
      surface: String(account.surface ?? "chat"),
      planPolicyId: String(account.plan_policy_id ?? ""),
      browser: {
        adapter: browserRaw.adapter,
        profilePath: String(browserRaw.profile_path ?? ""),
        headless: browserRaw.headless === true,
        allowInteractiveLogin: browserRaw.allow_interactive_login === true,
        requestTimeoutSeconds: positiveNumber(
          (account.collection as Record<string, unknown> | undefined)
            ?.request_timeout_seconds,
        ),
      },
    };
  });
  return {
    schemaVersion: 1,
    application: {
      reportTimezone: String(applicationRaw.report_timezone ?? DEFAULT_TIMEZONE),
      stateDirectory: String(applicationRaw.state_directory ?? DEFAULT_STATE_DIRECTORY),
    },
    accounts,
  };
}

export function defaultConfig(): Stage1Config {
  return {
    schemaVersion: 1,
    application: {
      reportTimezone: "America/New_York",
      stateDirectory: DEFAULT_STATE_DIRECTORY,
    },
    accounts: [
      {
        id: "personal-primary",
        enabled: true,
        provider: "openai",
        expectedProviderUserId: null,
        expectedWorkspaceId: null,
        quotaOwnerId: null,
        surface: "chat",
        planPolicyId: "pro200-chat-2026-09-05",
        browser: {
          adapter: "playwright_persistent_context",
          profilePath: "./state/browser/personal-primary",
          headless: false,
          allowInteractiveLogin: true,
          requestTimeoutSeconds: 30,
        },
      },
    ],
  };
}

export function saveConfig(config: Stage1Config, path: string): void {
  const resolvedPath = resolve(path);
  mkdirSync(dirname(resolvedPath), { recursive: true });
  const raw = {
    schema_version: config.schemaVersion,
    application: {
      report_timezone: config.application.reportTimezone,
      state_directory: config.application.stateDirectory,
    },
    accounts: config.accounts.map((account) => ({
      id: account.id,
      enabled: account.enabled,
      provider: account.provider,
      expected_provider_user_id: account.expectedProviderUserId,
      expected_workspace_id: account.expectedWorkspaceId,
      quota_owner_id: account.quotaOwnerId,
      surface: account.surface,
      plan_policy_id: account.planPolicyId,
      browser: {
        adapter: account.browser.adapter,
        profile_path: account.browser.profilePath,
        headless: account.browser.headless,
        allow_interactive_login: account.browser.allowInteractiveLogin,
      },
      collection: {
        request_timeout_seconds: account.browser.requestTimeoutSeconds,
      },
    })),
  };
  writeFileSync(resolvedPath, JSON.stringify(raw, null, 2) + "\n", {
    encoding: "utf8",
    mode: 0o600,
  });
  try {
    chmodSync(resolvedPath, 0o600);
  } catch {
    // Best-effort hardening for the identity-binding config.
  }
}

function optionalString(value: unknown): string | null {
  if (value === null || value === undefined || value === "" || value === false) {
    return null;
  }
  if (value === true) {
    return "true";
  }
  return String(value);
}

function positiveNumber(value: unknown): number {
  const number = Number(value ?? 30);
  return Number.isFinite(number) && number > 0 ? number : 30;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
