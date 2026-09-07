import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { run } from "../../src/cli/main.js";
import { defaultConfig, loadConfig, saveConfig } from "../../src/config.js";

describe("CLI", () => {
  let stateDirectory: string;

  beforeEach(() => {
    stateDirectory = mkdtempSync(join(tmpdir(), "usage-capture-state-"));
  });

  afterEach(() => {
    rmSync(stateDirectory, { recursive: true, force: true });
  });

  it("rejects Stage-2+ commands with an explicit error", async () => {
    for (const command of ["backfill", "refresh", "schedule", "windows", "quota"]) {
      const exitCode = await run([command, "--config", "unused.yaml"]);
      expect(exitCode).toBe(2);
    }
  });

  it("rejects unknown commands", async () => {
    const exitCode = await run(["unknown-command"]);
    expect(exitCode).toBe(2);
  });

  it("init writes a starter config", async () => {
    const configPath = join(stateDirectory, "config.json");
    const exitCode = await run(["init", "--config", configPath]);
    expect(exitCode).toBe(0);
    expect(JSON.parse(readFileSync(configPath, "utf8")).schema_version).toBe(1);
    expect(loadConfig(configPath).accounts).toHaveLength(1);
  });

  it("runs fixture-backed inspect-capabilities offline", async () => {
    const config = defaultConfig();
    const account = config.accounts[0]!;
    account.id = "fixture-primary";
    account.expectedProviderUserId = "user-abc123";
    account.expectedWorkspaceId = "ws-xyz";
    account.quotaOwnerId = "user-abc123";
    account.browser.adapter = "fixture_history";
    account.browser.profilePath = "";
    config.application.stateDirectory = stateDirectory;

    const configPath = join(stateDirectory, "fixture-config.json");
    saveConfig(config, configPath);
    const fixtureRoot = new URL("../fixtures/v1/", import.meta.url).pathname;

    const exitCode = await run([
      "inspect-capabilities",
      "--config",
      configPath,
      "--fixture-root",
      fixtureRoot,
      "--state-directory",
      stateDirectory,
    ]);

    expect(exitCode).toBe(0);
    const persisted = JSON.parse(
      readFileSync(join(stateDirectory, "bootstrap", "fixture-primary.json"), "utf8"),
    );
    expect(persisted.state).toBe("ready");
    expect(persisted.identity.providerUserId).toBe("user-abc123");
  });
});
