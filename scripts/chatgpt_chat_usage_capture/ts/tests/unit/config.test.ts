import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { defaultConfig, loadConfig, resolveDatabasePath, saveConfig } from "../../src/config.js";

describe("Stage-2 config contract", () => {
  const temporaryDirectories: string[] = [];

  afterEach(() => {
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("round-trips the generated JSON config through the loader", () => {
    const directory = mkdtempSync(join(tmpdir(), "usage-capture-config-"));
    temporaryDirectories.push(directory);
    const path = join(directory, "config.json");

    saveConfig(defaultConfig(), path);

    const raw = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
    expect(raw.schema_version).toBe(1);
    expect(raw.accounts).toBeInstanceOf(Array);
    expect((raw.accounts as Array<Record<string, unknown>>)[0]?.expected_provider_user_id)
      .toBeNull();

    const loaded = loadConfig(path);
    expect(loaded.schemaVersion).toBe(1);
    expect(loaded.accounts[0]?.browser.adapter).toBe("playwright_persistent_context");
    expect(loaded.accounts[0]?.planPolicyId).toBe("pro200-chat-2026-09-05");
    expect(loaded.application.databasePath).toBe("./state/usage.sqlite");
  });

  it("uses one database path with consistent command-line override precedence", () => {
    const config = defaultConfig();
    config.application.databasePath = "./custom/ledger.sqlite";
    expect(resolveDatabasePath(config, { databasePath: null, stateDirectory: null }))
      .toBe(resolve("./custom/ledger.sqlite"));
    expect(resolveDatabasePath(config, { databasePath: null, stateDirectory: "./isolated" }))
      .toBe(resolve("./isolated/usage.sqlite"));
    expect(resolveDatabasePath(config, { databasePath: "./explicit.sqlite", stateDirectory: "./isolated" }))
      .toBe(resolve("./explicit.sqlite"));
  });
});
