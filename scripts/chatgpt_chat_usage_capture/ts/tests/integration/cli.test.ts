import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { run } from "../../src/cli/main.js";
import { loadConfig } from "../../src/config.js";

describe("CLI", () => {
  let stateDirectory: string;

  beforeEach(() => {
    stateDirectory = mkdtempSync(join(tmpdir(), "usage-capture-state-"));
  });

  afterEach(() => {
    rmSync(stateDirectory, { recursive: true, force: true });
  });

  it("rejects Stage-2+ commands with an explicit error", async () => {
    for (const command of ["backfill", "refresh", "report", "schedule", "rebuild"]) {
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
});
