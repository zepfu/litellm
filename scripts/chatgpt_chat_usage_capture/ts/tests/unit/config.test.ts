import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
    expect(loaded.accounts[0]?.collection).toMatchObject({
      requestTimeoutSeconds: 30,
      initialBackfillDays: 14,
      overlapMs: 48 * 60 * 60 * 1000,
      indexPageSize: 100,
      maxIndexPagesPerScope: 500,
      maxPagesPerConversationPerRun: 100,
    });
  });

  it("preserves supported collection bounds and rejects unsupported controls", () => {
    const directory = mkdtempSync(join(tmpdir(), "usage-capture-config-"));
    temporaryDirectories.push(directory);
    const path = join(directory, "config.json");
    const config = defaultConfig();
    config.accounts[0]!.collection = {
      requestTimeoutSeconds: 45,
      initialBackfillDays: 7,
      overlapMs: 12 * 60 * 60 * 1000,
      indexPageSize: 50,
      maxIndexPagesPerScope: 9,
      maxPagesPerConversationPerRun: 8,
    };

    saveConfig(config, path);
    const written = JSON.parse(readFileSync(path, "utf8")) as {
      accounts: Array<{ collection: Record<string, unknown> }>;
    };
    expect(written.accounts[0]?.collection).toMatchObject({
      request_timeout_seconds: 45,
      initial_backfill_duration: "P7D",
      overlap_duration: "PT12H",
      index_page_size: 50,
      max_index_pages_per_scope: 9,
      max_pages_per_conversation_per_run: 8,
    });
    expect(loadConfig(path).accounts[0]?.collection).toMatchObject(
      config.accounts[0]!.collection,
    );

    written.accounts[0]!.collection = { max_response_bytes: 1024 };
    writeFileSync(path, JSON.stringify(written), "utf8");
    expect(() => loadConfig(path)).toThrow(
      "accounts[0].collection.max_response_bytes is an unsupported Stage-2 collection control",
    );

    written.accounts[0]!.collection = { max_pages_per_conversation_per_run: 0 };
    writeFileSync(path, JSON.stringify(written), "utf8");
    expect(() => loadConfig(path)).toThrow(
      "accounts[0].collection.max_pages_per_conversation_per_run must be a positive integer",
    );
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
