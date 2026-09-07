/**
 * Shared configuration for history collection and the Stage-2 local ledger.
 * Scheduler, reset-window accounting, provider quota accounting, and API
 * configuration belong to later stages.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";

import { MAX_RESPONSE_BYTES, type BrowserConfig } from "./browser/session.js";
import {
  DEFAULT_BACKFILL_DAYS,
  DEFAULT_INDEX_PAGE_SIZE,
  DEFAULT_MAX_INDEX_PAGES,
  DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION,
  DEFAULT_OVERLAP_MS,
} from "./contracts/history.js";

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
  collection?: CollectionConfig;
}

export interface CollectionConfig {
  requestTimeoutSeconds: number;
  initialBackfillDays: number;
  overlapMs: number;
  indexPageSize: number;
  maxIndexPagesPerScope: number;
  maxPagesPerConversationPerRun: number;
  maxResponseBytes: number;
}

export interface ApplicationConfig {
  reportTimezone: string;
  stateDirectory: string;
  databasePath: string;
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
const DEFAULT_DATABASE_PATH = "./state/usage.sqlite";
const DEFAULT_REQUEST_TIMEOUT_SECONDS = 30;
const DAY_MS = 24 * 60 * 60 * 1000;
const COLLECTION_KEYS = new Set([
  "request_timeout_seconds",
  "initial_backfill_duration",
  "overlap_duration",
  "index_page_size",
  "max_index_pages_per_scope",
  "max_pages_per_conversation_per_run",
  "max_message_pages_per_conversation",
  "max_response_bytes",
]);

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
  const applicationRaw = recordOrEmpty(raw.application, "application");
  if (raw.accounts !== undefined && !Array.isArray(raw.accounts)) {
    throw new ConfigError("accounts must be an array");
  }
  const accountsRaw = Array.isArray(raw.accounts) ? raw.accounts : [];
  const accounts: AccountConfig[] = accountsRaw.map((entry, index) => {
    if (!isRecord(entry)) {
      throw new ConfigError(`accounts[${index}] must be an object`);
    }
    const account = entry;
    const browserRaw = recordOrEmpty(account.browser, `accounts[${index}].browser`);
    const collectionRaw = recordOrEmpty(
      account.collection,
      `accounts[${index}].collection`,
    );
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
    const collection = parseCollectionConfig(
      collectionRaw,
      `accounts[${index}].collection`,
    );
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
        requestTimeoutSeconds: collection.requestTimeoutSeconds,
        maxResponseBytes: collection.maxResponseBytes,
      },
      collection,
    };
  });
  return {
    schemaVersion: 1,
    application: {
      reportTimezone: String(applicationRaw.report_timezone ?? DEFAULT_TIMEZONE),
      stateDirectory: String(applicationRaw.state_directory ?? DEFAULT_STATE_DIRECTORY),
      databasePath: String(
        applicationRaw.database_path ??
          join(String(applicationRaw.state_directory ?? DEFAULT_STATE_DIRECTORY), "usage.sqlite"),
      ),
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
      databasePath: DEFAULT_DATABASE_PATH,
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
          maxResponseBytes: MAX_RESPONSE_BYTES,
        },
        collection: defaultCollectionConfig(),
      },
    ],
  };
}

export function defaultCollectionConfig(
  requestTimeoutSeconds = DEFAULT_REQUEST_TIMEOUT_SECONDS,
  maxResponseBytes = MAX_RESPONSE_BYTES,
): CollectionConfig {
  return {
    requestTimeoutSeconds,
    initialBackfillDays: DEFAULT_BACKFILL_DAYS,
    overlapMs: DEFAULT_OVERLAP_MS,
    indexPageSize: DEFAULT_INDEX_PAGE_SIZE,
    maxIndexPagesPerScope: DEFAULT_MAX_INDEX_PAGES,
    maxPagesPerConversationPerRun: DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION,
    maxResponseBytes,
  };
}

export function resolveDatabasePath(
  config: Stage1Config,
  overrides: { databasePath: string | null; stateDirectory: string | null },
): string {
  return resolve(
    overrides.databasePath ??
      (overrides.stateDirectory
        ? join(overrides.stateDirectory, "usage.sqlite")
        : config.application.databasePath),
  );
}

export function saveConfig(config: Stage1Config, path: string): void {
  const resolvedPath = resolve(path);
  mkdirSync(dirname(resolvedPath), { recursive: true });
  const raw = {
    schema_version: config.schemaVersion,
    application: {
      report_timezone: config.application.reportTimezone,
      state_directory: config.application.stateDirectory,
      database_path: config.application.databasePath,
    },
    accounts: config.accounts.map((account) => {
      const collection =
        account.collection ??
        defaultCollectionConfig(
          account.browser.requestTimeoutSeconds,
          account.browser.maxResponseBytes,
        );
      return {
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
          request_timeout_seconds: collection.requestTimeoutSeconds,
          initial_backfill_duration: formatDuration(collection.initialBackfillDays * DAY_MS),
          overlap_duration: formatDuration(collection.overlapMs),
          index_page_size: collection.indexPageSize,
          max_index_pages_per_scope: collection.maxIndexPagesPerScope,
          max_pages_per_conversation_per_run:
            collection.maxPagesPerConversationPerRun,
          max_response_bytes: collection.maxResponseBytes,
        },
      };
    }),
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

function parseCollectionConfig(
  raw: Record<string, unknown>,
  path: string,
): CollectionConfig {
  for (const key of Object.keys(raw)) {
    if (!COLLECTION_KEYS.has(key)) {
      throw new ConfigError(
        `${path}.${key} is an unsupported Stage-2 collection control`,
      );
    }
  }

  const maxPagesPerConversation = resolveAliasedValue(
    raw,
    "max_pages_per_conversation_per_run",
    "max_message_pages_per_conversation",
    path,
  );
  const requestTimeoutSeconds = positiveNumber(
    raw.request_timeout_seconds,
    `${path}.request_timeout_seconds`,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
  );
  const initialBackfillMs = parseDurationMilliseconds(
    raw.initial_backfill_duration,
    `${path}.initial_backfill_duration`,
    DEFAULT_BACKFILL_DAYS * DAY_MS,
  );
  const overlapMs = parseDurationMilliseconds(
    raw.overlap_duration,
    `${path}.overlap_duration`,
    DEFAULT_OVERLAP_MS,
  );
  const maxResponseBytes = positiveInteger(
    raw.max_response_bytes,
    `${path}.max_response_bytes`,
    MAX_RESPONSE_BYTES,
  );

  return {
    requestTimeoutSeconds,
    initialBackfillDays: initialBackfillMs / DAY_MS,
    overlapMs,
    indexPageSize: positiveInteger(
      raw.index_page_size,
      `${path}.index_page_size`,
      DEFAULT_INDEX_PAGE_SIZE,
    ),
    maxIndexPagesPerScope: positiveInteger(
      raw.max_index_pages_per_scope,
      `${path}.max_index_pages_per_scope`,
      DEFAULT_MAX_INDEX_PAGES,
    ),
    maxPagesPerConversationPerRun: positiveInteger(
      maxPagesPerConversation,
      `${path}.max_pages_per_conversation_per_run`,
      DEFAULT_MAX_MESSAGE_PAGES_PER_CONVERSATION,
    ),
    maxResponseBytes,
  };
}

function resolveAliasedValue(
  raw: Record<string, unknown>,
  primary: string,
  alias: string,
  path: string,
): unknown {
  if (raw[primary] !== undefined && raw[alias] !== undefined) {
    throw new ConfigError(`${path} cannot set both ${primary} and ${alias}`);
  }
  return raw[primary] ?? raw[alias];
}

function positiveNumber(value: unknown, path: string, fallback: number): number {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new ConfigError(`${path} must be a positive finite number`);
  }
  return value;
}

function positiveInteger(value: unknown, path: string, fallback: number): number {
  if (value === undefined) {
    return fallback;
  }
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value <= 0
  ) {
    throw new ConfigError(`${path} must be a positive integer`);
  }
  return value;
}

function parseDurationMilliseconds(
  value: unknown,
  path: string,
  fallback: number,
): number {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "string") {
    throw new ConfigError(`${path} must be an ISO-8601 duration`);
  }
  const text = value.trim();
  const shortMatch = /^(\d+(?:\.\d+)?)([dhm])$/i.exec(text);
  if (shortMatch) {
    const amount = Number(shortMatch[1]);
    const unit = shortMatch[2]!.toLowerCase();
    const multiplier =
      unit === "d" ? DAY_MS : unit === "h" ? 60 * 60 * 1000 : 60 * 1000;
    const milliseconds = amount * multiplier;
    if (Number.isFinite(milliseconds) && milliseconds > 0) {
      return milliseconds;
    }
  }

  const match =
    /^P(?:(\d+(?:\.\d+)?)D)?(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?$/i.exec(
      text,
    );
  if (!match || !match.slice(1).some((part) => part !== undefined)) {
    throw new ConfigError(`${path} must be a positive ISO-8601 duration`);
  }
  const milliseconds =
    Number(match[1] ?? 0) * DAY_MS +
    Number(match[2] ?? 0) * 60 * 60 * 1000 +
    Number(match[3] ?? 0) * 60 * 1000 +
    Number(match[4] ?? 0) * 1000;
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) {
    throw new ConfigError(`${path} must be a positive ISO-8601 duration`);
  }
  return milliseconds;
}

function formatDuration(milliseconds: number): string {
  if (milliseconds % DAY_MS === 0) {
    return `P${milliseconds / DAY_MS}D`;
  }
  if (milliseconds % (60 * 60 * 1000) === 0) {
    return `PT${milliseconds / (60 * 60 * 1000)}H`;
  }
  if (milliseconds % (60 * 1000) === 0) {
    return `PT${milliseconds / (60 * 1000)}M`;
  }
  return `PT${milliseconds / 1000}S`;
}

function recordOrEmpty(
  value: unknown,
  path: string,
): Record<string, unknown> {
  if (value === undefined) {
    return {};
  }
  if (!isRecord(value)) {
    throw new ConfigError(`${path} must be an object`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
