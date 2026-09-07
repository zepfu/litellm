/**
 * Stage-1 CLI entry point.
 *
 * Implemented commands: init, bootstrap, inspect-capabilities. All other
 * Stage-2+ commands fail closed with an explicit "not implemented in Stage 1"
 * error. No command submits prompts, exports credentials, or bypasses the
 * GET-only route allowlist.
 */

import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { loadConfig, saveConfig, defaultConfig, ConfigError } from "../config.js";
import {
  bootstrapAccount,
  inspectCapabilities,
} from "../browser/bootstrap.js";
import { ADAPTER_VERSION } from "../contracts/records.js";
import type { BootstrapResult, InspectCapabilitiesResult } from "../browser/bootstrap.js";

interface CliArgs {
  command: string;
  configPath: string;
  accountId: string | null;
  interactiveLogin: boolean;
  stateDirectory: string | null;
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
    if (STAGE2_PLUS_COMMANDS.has(args.command)) {
      console.error(
        `usage-capture: '${args.command}' is not implemented in Stage 1. ` +
          "Stage 1 covers bootstrap, inspect-capabilities, and init only.",
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
  return runInspectCapabilities(args);
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
  if (account.browser.adapter !== "playwright_persistent_context") {
    console.error(
      "usage-capture inspect-capabilities: browser adapter is not playwright_persistent_context",
    );
    return 2;
  }

  const result = await inspectCapabilities(account, {
    stateDirectory: resolve(args.stateDirectory ?? config.application.stateDirectory),
  });
  printInspectCapabilitiesResult(result);
  return result.state === "ready" ? 0 : 1;
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

  return { command, configPath, accountId, interactiveLogin, stateDirectory };
}

function printHelp(): void {
  console.log(
    `usage-capture (Stage 1, adapter ${ADAPTER_VERSION})

Stage-1 commands:
  init --config <path>
      Write a starter JSON config (schema_version 1) for the dedicated profile.

  bootstrap --config <path> [--account <id>] [--state-directory <path>]
      [--interactive-login]
      Verify the dedicated browser profile and bind identity, workspace, and
      quota owner. Interactive login is explicit and opt-in only.

  inspect-capabilities --config <path> [--account <id>] [--state-directory <path>]
      Read-only capability inspection of active and archived history indexes.

All other commands (backfill, refresh, report, schedule, windows, quota,
rebuild, export, dashboard, models) are Stage-2+ and fail closed with an
explicit error. No prompt submission, credential export, or provider mutation
is supported in Stage 1.`,
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
