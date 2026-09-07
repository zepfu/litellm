/**
 * TypeScript Stage-2B CLI entry point.
 *
 * Browser commands remain read-only. Ledger commands operate only on retained
 * local evidence; deferred browser traversal, scheduling, reset accounting,
 * API, and UI commands fail closed. No command submits prompts, exports
 * credentials, or bypasses the GET-only route allowlist.
 */

import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { loadConfig, saveConfig, defaultConfig, ConfigError } from "../config.js";
import {
  bootstrapAccount,
  inspectCapabilities,
  inspectFixtureCapabilities,
} from "../browser/bootstrap.js";
import { buildRawModelReport } from "../accounting/raw-model.js";
import { reaggregate } from "../accounting/reaggregate.js";
import { ADAPTER_VERSION } from "../contracts/records.js";
import { Ledger } from "../ledger/store.js";
import type { BootstrapResult, InspectCapabilitiesResult } from "../browser/bootstrap.js";

interface CliArgs {
  command: string;
  configPath: string;
  accountId: string | null;
  interactiveLogin: boolean;
  stateDirectory: string | null;
  fixtureRoot: string | null;
  databasePath: string | null;
  lastHours: number | null;
  mappingVersion: string | null;
  apply: boolean;
}

const STAGE1_COMMANDS = new Set(["init", "bootstrap", "inspect-capabilities"]);
const STAGE2_PLUS_COMMANDS = new Set([
  "backfill",
  "refresh",
  "run",
  "status",
  "report",
  "schedule",
  "windows",
  "quota",
  "rebuild",
  "export",
  "dashboard",
  "models",
]);

export async function run(argv: string[]): Promise<number> {
  let args: CliArgs;
  try {
    args = parseArgs(argv);
  } catch (error) {
    console.error((error as Error).message);
    return 2;
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
    if (STAGE2_PLUS_COMMANDS.has(args.command)) {
      console.error(
        `usage-capture: '${args.command}' is deferred beyond Stage 2B. ` +
          "Browser traversal, scheduling, reset accounting, API, and UI are not implemented.",
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

function runReport(args: CliArgs): number {
  const config = loadConfig(args.configPath);
  const account = selectAccount(config.accounts, args.accountId);
  if (!account) {
    console.error(`usage-capture report: account '${args.accountId}' not found`);
    return 2;
  }
  const ledger = new Ledger(
    resolve(args.databasePath ?? config.application.databasePath),
  );
  try {
    const scope = ledger.accountScope(account.id);
    console.log(
      JSON.stringify(
        buildRawModelReport(ledger, scope, {
          durationMs: (args.lastHours ?? 24) * 60 * 60 * 1000,
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
  const ledger = new Ledger(
    resolve(args.databasePath ?? config.application.databasePath),
  );
  try {
    const scope = ledger.accountScope(account.id);
    const mapping = selectMapping(ledger, args.mappingVersion);
    const result = reaggregate(ledger, scope, mapping, { apply: args.apply });
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
  const ledger = new Ledger(
    resolve(args.databasePath ?? config.application.databasePath),
  );
  try {
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

function selectMapping(
  ledger: Ledger,
  version: string | null,
) {
  if (version) {
    return ledger.modelMapping(version);
  }
  const mapping = ledger.modelMappings().at(-1);
  if (!mapping) {
    throw new Error(
      "no model mapping version is recorded; use --mapping-version after recording one",
    );
  }
  return mapping;
}

function printBootstrapResult(result: BootstrapResult): void {
  console.log(JSON.stringify(result, null, 2));
}

function printInspectCapabilitiesResult(result: InspectCapabilitiesResult): void {
  console.log(JSON.stringify(result, null, 2));
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
    throw new Error("--fixture-root is only supported by inspect-capabilities");
  }

  return {
    command,
    configPath,
    accountId,
    interactiveLogin,
    stateDirectory,
    fixtureRoot,
    databasePath,
    lastHours,
    mappingVersion,
    apply,
  };
}

function printHelp(): void {
  console.log(
    `usage-capture (Stage 2B, adapter ${ADAPTER_VERSION})

Stage-1 commands:
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

Stage-2B offline ledger commands:
  report --config <path> [--account <id>] [--database <path>] [--last-hours <n>]
      Report observed raw-model activity from SQLite over the last N hours.

  models --config <path> [--account <id>] [--database <path>]
      Show raw-model mapping review suggestions from retained attempts.

  rebuild --config <path> [--account <id>] [--database <path>]
      [--mapping-version <version>] [--apply]
      Preview or apply deterministic reaggregation without website requests.

Deferred commands (browser traversal, scheduling, reset accounting, API, and
UI) fail closed. No prompt submission, credential export, or provider mutation
is supported.`,
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
