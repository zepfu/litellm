/**
 * TypeScript Stage-2 collector CLI entry point.
 *
 * History collection and offline ledger commands share one config and database.
 * Scheduling, reset accounting, API, and UI remain deferred. No command submits
 * prompts, exports credentials, or bypasses the GET-only route allowlist.
 */

import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  loadConfig, saveConfig, defaultConfig, ConfigError, resolveDatabasePath,
} from "../config.js";
import {
  bootstrapAccount,
  inspectCapabilities,
  inspectFixtureCapabilities,
} from "../browser/bootstrap.js";
import {
  ChatGPTHistoryAdapter,
  type HistoryTransport,
} from "../adapters/chatgpt/adapter.js";
import { FixtureTransport } from "../adapters/chatgpt/fixture-transport.js";
import { PlaywrightTransport } from "../browser/session.js";
import { buildRawModelReport } from "../accounting/raw-model.js";
import { reaggregate } from "../accounting/reaggregate.js";
import { ADAPTER_VERSION } from "../contracts/records.js";
import { Ledger } from "../ledger/store.js";
import type { BootstrapResult, InspectCapabilitiesResult } from "../browser/bootstrap.js";
import type {
  HistoryCollectionMode,
  HistoryCollectionResult,
  HistoryRange,
} from "../contracts/history.js";
import {
  assertAccountBinding,
  collectIntoLedger,
  configuredScope,
  selectMapping,
} from "../history/ingest.js";
import {
  defaultBackfillRange,
  makeRange,
  parseDuration,
  parseInstant,
} from "../history/range.js";

interface CliArgs {
  command: string;
  configPath: string;
  accountId: string | null;
  interactiveLogin: boolean;
  stateDirectory: string | null;
  fixtureRoot: string | null;
  since: string | null;
  until: string | null;
  databasePath: string | null;
  lastHours: number | null;
  mappingVersion: string | null;
  apply: boolean;
}

const STAGE1_COMMANDS = new Set(["init", "bootstrap", "inspect-capabilities"]);
const STAGE2_HISTORY_COMMANDS = new Set(["backfill", "refresh", "reconcile"]);
const DEFERRED_COMMANDS = new Set([
  "run",
  "status",
  "schedule",
  "windows",
  "quota",
  "export",
  "dashboard",
]);

export async function run(argv: string[]): Promise<number> {
  let args: CliArgs;
  try {
    args = parseArgs(argv);
  } catch (error) {
    console.error((error as Error).message);
    return 2;
  }

  if (STAGE2_HISTORY_COMMANDS.has(args.command)) {
    try {
      return await runHistoryCollection(args);
    } catch (error) {
      console.error(`usage-capture: ${(error as Error).message}`);
      return 1;
    }
  }
  if (!STAGE1_COMMANDS.has(args.command)) {
    if (["report", "rebuild", "models"].includes(args.command)) {
      try {
        return await execute(args);
      } catch (error) {
        console.error(`usage-capture: ${(error as Error).message}`);
        return 1;
      }
    }
    if (DEFERRED_COMMANDS.has(args.command)) {
      console.error(
        `usage-capture: '${args.command}' is deferred beyond Stage 2. ` +
          "Scheduling, reset accounting, API, and UI are not implemented.",
      );
      return 2;
    }
    console.error(`usage-capture: unknown command '${args.command}'`);
    return 2;
  }

  try {
    return await execute(args);
  } catch (error) {
    console.error(`usage-capture: ${(error as Error).message}`);
    return 1;
  }
}

async function execute(args: CliArgs): Promise<number> {
  if (args.command === "init") {
    return runInit(args);
  }
  if (args.command === "bootstrap") {
    return runBootstrap(args);
  }
  if (args.command === "inspect-capabilities") {
    return runInspectCapabilities(args);
  }
  if (args.command === "report") {
    return runReport(args);
  }
  if (args.command === "rebuild") {
    return runRebuild(args);
  }
  return runModels(args);
}

function runInit(args: CliArgs): number {
  const configPath = resolve(args.configPath);
  if (existsSync(configPath)) {
    console.log(`usage-capture init: config already exists at ${configPath}`);
    return 0;
  }
  const config = defaultConfig();
  saveConfig(config, configPath);
  console.log(
    `usage-capture init: wrote starter config to ${configPath}. ` +
      "Bind expected_provider_user_id / expected_workspace_id / quota_owner_id before bootstrap.",
  );
  return 0;
}

async function runBootstrap(args: CliArgs): Promise<number> {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture bootstrap: account '${args.accountId}' not found`);
    return 2;
  }
  if (account.browser.adapter !== "playwright_persistent_context") {
    console.error(
      "usage-capture bootstrap: browser adapter is not playwright_persistent_context",
    );
    return 2;
  }

  const result = await bootstrapAccount(account, {
    interactiveLogin: args.interactiveLogin,
    stateDirectory: resolve(args.stateDirectory ?? config.application.stateDirectory),
  });
  printBootstrapResult(result);
  return result.state === "ready" ? 0 : 1;
}

async function runInspectCapabilities(args: CliArgs): Promise<number> {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture inspect-capabilities: account '${args.accountId}' not found`);
    return 2;
  }
  const stateDirectory = resolve(
    args.stateDirectory ?? config.application.stateDirectory,
  );

  if (account.browser.adapter === "fixture_history") {
    if (!args.fixtureRoot) {
      console.error(
        "usage-capture inspect-capabilities: fixture_history requires --fixture-root",
      );
      return 2;
    }
    const result = await inspectFixtureCapabilities(account, {
      fixtureRoot: resolve(args.fixtureRoot),
      stateDirectory,
    });
    printInspectCapabilitiesResult(result);
    return result.state === "ready" ? 0 : 1;
  }

  if (args.fixtureRoot) {
    console.error(
      "usage-capture inspect-capabilities: --fixture-root requires browser.adapter fixture_history",
    );
    return 2;
  }

  if (account.browser.adapter !== "playwright_persistent_context") {
    console.error(
      "usage-capture inspect-capabilities: browser adapter is not playwright_persistent_context",
    );
    return 2;
  }

  const result = await inspectCapabilities(account, {
    stateDirectory,
  });
  printInspectCapabilitiesResult(result);
  return result.state === "ready" ? 0 : 1;
}

async function runHistoryCollection(args: CliArgs): Promise<number> {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture ${args.command}: account '${args.accountId}' not found`);
    return 2;
  }
  if (account.browser.adapter === "fixture_history" && !args.fixtureRoot) {
    console.error(
      `usage-capture ${args.command}: fixture_history requires --fixture-root`,
    );
    return 2;
  }
  if (account.browser.adapter === "playwright_persistent_context" && args.fixtureRoot) {
    console.error(
      `usage-capture ${args.command}: --fixture-root requires browser.adapter fixture_history`,
    );
    return 2;
  }

  const mode: HistoryCollectionMode =
    args.command === "backfill"
      ? "backfill"
      : args.command === "reconcile"
        ? "reconciliation"
        : "incremental";
  let range: HistoryRange | undefined;
  try {
    range = collectionRange(args, mode);
  } catch (error) {
    console.error(`usage-capture ${args.command}: ${(error as Error).message}`);
    return 2;
  }

  const transport: HistoryTransport =
    account.browser.adapter === "fixture_history"
      ? new FixtureTransport(resolve(args.fixtureRoot!))
      : new PlaywrightTransport(account.browser);
  const adapter = new ChatGPTHistoryAdapter(transport, {
    providerUserId: account.expectedProviderUserId,
    workspaceId: account.expectedWorkspaceId,
    quotaOwnerId: account.quotaOwnerId,
  });
  let ledger: Ledger | undefined;
  try {
    ledger = new Ledger(resolveDatabasePath(config, args));
    const request = range ? { mode, range } : { mode };
    const result = await collectIntoLedger(adapter, ledger, account, request, args.mappingVersion);
    printCollectionResult(result);
    return result.status === "blocked" ? 1 : 0;
  } finally {
    try {
      await adapter.close();
    } finally {
      ledger?.close();
    }
  }
}

function collectionRange(
  args: CliArgs,
  mode: HistoryCollectionMode,
): HistoryRange | undefined {
  if (mode === "incremental" && !args.since && !args.until) {
    return undefined;
  }
  if (mode === "reconciliation" && !args.since) {
    throw new Error("reconcile requires --since and an explicit range");
  }
  const now = new Date();
  const end = args.until ? parseInstant(args.until) : now;
  if (!args.since) {
    return defaultBackfillRange(end);
  }
  const sinceIsDuration = /^\d+(?:\.\d+)?[dhm]$/i.test(args.since.trim());
  if (!sinceIsDuration) {
    return makeRange(parseInstant(args.since), end);
  }
  const duration = parseDuration(args.since);
  return makeRange(new Date(end.getTime() - duration), end);
}

function runReport(args: CliArgs): number {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture report: account '${args.accountId}' not found`);
    return 2;
  }
  const ledger = new Ledger(resolveDatabasePath(config, args));
  try {
    assertAccountBinding(ledger, configuredScope(account));
    const scope = ledger.accountScope(account.id);
    console.log(
      JSON.stringify(
        buildRawModelReport(ledger, scope, {
          durationMs: (args.lastHours ?? 24) * 60 * 60 * 1000,
          ...(args.until ? { now: parseInstant(args.until).toISOString() } : {}),
        }),
        null,
        2,
      ),
    );
    return 0;
  } finally {
    ledger.close();
  }
}

function runRebuild(args: CliArgs): number {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture rebuild: account '${args.accountId}' not found`);
    return 2;
  }
  const ledger = new Ledger(resolveDatabasePath(config, args));
  try {
    assertAccountBinding(ledger, configuredScope(account));
    const scope = ledger.accountScope(account.id);
    const mapping = selectMapping(ledger, args.mappingVersion);
    const result = reaggregate(ledger, scope, mapping, {
      apply: args.apply,
      ...(args.until ? { evaluatedAt: parseInstant(args.until).toISOString() } : {}),
    });
    console.log(
      JSON.stringify({ mode: args.apply ? "apply" : "preview", ...result }, null, 2),
    );
    return 0;
  } finally {
    ledger.close();
  }
}

function runModels(args: CliArgs): number {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture models: account '${args.accountId}' not found`);
    return 2;
  }
  const ledger = new Ledger(resolveDatabasePath(config, args));
  try {
    assertAccountBinding(ledger, configuredScope(account));
    const scope = ledger.accountScope(account.id);
    const mapping = selectMapping(ledger, args.mappingVersion);
    console.log(
      JSON.stringify(
        {
          mapping_version: mapping.version,
          review_status: mapping.reviewStatus,
          suggestions: ledger.modelMappingSuggestions(scope, mapping),
        },
        null,
        2,
      ),
    );
    return 0;
  } finally {
    ledger.close();
  }
}

function printBootstrapResult(result: BootstrapResult): void {
  console.log(JSON.stringify(result, null, 2));
}

function printInspectCapabilitiesResult(result: InspectCapabilitiesResult): void {
  console.log(JSON.stringify(result, null, 2));
}

function printCollectionResult(result: HistoryCollectionResult): void {
  console.log(
    JSON.stringify(
      {
        ...result,
        conversations: result.conversations.map((conversation) => ({
          summary: conversation.summary,
          scopes: conversation.scopes,
          coverage: conversation.coverage,
          messageCount: conversation.messages.length,
          revisit: conversation.revisit,
          warnings: conversation.warnings,
        })),
      },
      null,
      2,
    ),
  );
}

function selectAccount<T extends { id: string; enabled: boolean }>(
  accounts: T[],
  accountId: string | null,
): T | null {
  if (accountId) {
    return accounts.find((account) => account.id === accountId) ?? null;
  }
  return accounts.find((account) => account.enabled) ?? accounts[0] ?? null;
}

function parseArgs(argv: string[]): CliArgs {
  const command = argv[0] ?? "";
  let configPath = "./config.json";
  let accountId: string | null = null;
  let interactiveLogin = false;
  let stateDirectory: string | null = null;
  let fixtureRoot: string | null = null;
  let since: string | null = null;
  let until: string | null = null;
  let databasePath: string | null = null;
  let lastHours: number | null = null;
  let mappingVersion: string | null = null;
  let apply = false;

  for (let index = 1; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === "--config" && argv[index + 1]) {
      configPath = argv[index + 1]!;
      index += 1;
    } else if (flag === "--account" && argv[index + 1]) {
      accountId = argv[index + 1]!;
      index += 1;
    } else if (flag === "--interactive-login") {
      interactiveLogin = true;
    } else if (flag === "--state-directory" && argv[index + 1]) {
      stateDirectory = argv[index + 1]!;
      index += 1;
    } else if (flag === "--fixture-root" && argv[index + 1]) {
      fixtureRoot = argv[index + 1]!;
      index += 1;
    } else if (flag === "--since" && argv[index + 1]) {
      since = argv[index + 1]!;
      index += 1;
    } else if (flag === "--until" && argv[index + 1]) {
      until = argv[index + 1]!;
      index += 1;
    } else if (flag === "--database" && argv[index + 1]) {
      databasePath = argv[index + 1]!;
      index += 1;
    } else if (flag === "--last-hours" && argv[index + 1]) {
      const value = Number(argv[index + 1]);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error("--last-hours must be a positive number");
      }
      lastHours = value;
      index += 1;
    } else if (flag === "--mapping-version" && argv[index + 1]) {
      mappingVersion = argv[index + 1]!;
      index += 1;
    } else if (flag === "--apply") {
      apply = true;
    } else if (flag === "--help" || flag === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`unknown flag: ${flag}`);
    }
  }

  if (!command) {
    printHelp();
    process.exit(2);
  }

  if (fixtureRoot && command !== "inspect-capabilities") {
    if (!STAGE2_HISTORY_COMMANDS.has(command)) {
      throw new Error(
        "--fixture-root is only supported by inspect-capabilities or history collection",
      );
    }
  }

  return {
    command,
    configPath,
    accountId,
    interactiveLogin,
    stateDirectory,
    fixtureRoot,
    since,
    until,
    databasePath,
    lastHours,
    mappingVersion,
    apply,
  };
}

function printHelp(): void {
  console.log(
    `usage-capture (Stage 2, adapter ${ADAPTER_VERSION})

Bootstrap and capability commands:
  init --config <path>
      Write a starter JSON config (schema_version 1) for the dedicated profile.

  bootstrap --config <path> [--account <id>] [--state-directory <path>]
      [--interactive-login]
      Verify the dedicated browser profile and bind identity, workspace, and
      quota owner. Interactive login is explicit and opt-in only.

  inspect-capabilities --config <path> [--account <id>] [--state-directory <path>]
      [--fixture-root <path>]
      Read-only capability inspection of active and archived history indexes.
      Use --fixture-root only with a fixture_history account for offline
      acceptance; live inspection requires playwright_persistent_context.

Stage-2 history commands:
  backfill --config <path> [--account <id>] [--fixture-root <path>]
      [--since <14d|ISO-8601>] [--until <ISO-8601>]
      Acquire active/archived history into SQLite. The default range is 14 days.

  refresh --config <path> [--account <id>] [--fixture-root <path>]
      Run incremental discovery with the durable per-scope watermark and a
      48-hour overlap.

  reconcile --config <path> [--account <id>] [--fixture-root <path>]
      --since <ISO-8601> [--until <ISO-8601>]
      Re-read the explicitly requested history range.

Stage-2 offline ledger commands:
  report --config <path> [--account <id>] [--database <path>] [--last-hours <n>]
      [--until <ISO-8601>]
      Report observed raw-model activity from SQLite over the last N hours.

  models --config <path> [--account <id>] [--database <path>]
      Show raw-model mapping review suggestions from retained attempts.

  rebuild --config <path> [--account <id>] [--database <path>]
      [--mapping-version <version>] [--until <ISO-8601>] [--apply]
      Preview or apply deterministic reaggregation without website requests.

All history and ledger commands share application.database_path. --database
overrides it; --state-directory uses <path>/usage.sqlite unless --database is
also supplied. Reports and rebuilds never issue website requests.

Scheduling, reset accounting, API, UI, and exports remain deferred. Collection
is GET-only; no prompt submission, credential export, or provider mutation is
supported.`,
  );
}

export { ADAPTER_VERSION, ConfigError };

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  run(process.argv.slice(2))
    .then((code) => {
      process.exit(code);
    })
    .catch((error) => {
      console.error(`usage-capture: ${(error as Error).message}`);
      process.exit(1);
    });
}
