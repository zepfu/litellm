import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../src/browser/bootstrap.js", async () => {
  const actual = await vi.importActual<
    typeof import("../../src/browser/bootstrap.js")
  >("../../src/browser/bootstrap.js");
  return {
    ...actual,
    bootstrapAccount: vi.fn(),
  };
});

import { run } from "../../src/cli/main.js";
import { bootstrapAccount } from "../../src/browser/bootstrap.js";
import { defaultConfig, loadConfig, saveConfig } from "../../src/config.js";
import { emptyCapabilities } from "../../src/normalize/identity.js";
import { Ledger } from "../../src/ledger/store.js";
import {
  SqliteCheckpointStore,
} from "../../src/history/checkpoints.js";
import { configuredScope } from "../../src/history/ingest.js";

describe("CLI", () => {
  let stateDirectory: string;

  beforeEach(() => {
    vi.clearAllMocks();
    stateDirectory = mkdtempSync(join(tmpdir(), "usage-capture-state-"));
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    rmSync(stateDirectory, { recursive: true, force: true });
  });

  it("rejects deferred commands with an explicit error", async () => {
    for (const command of ["run", "schedule", "windows", "quota"]) {
      const exitCode = await run([command, "--config", "unused.yaml"]);
      expect(exitCode).toBe(2);
    }
  });

  it("rejects unknown commands", async () => {
    const exitCode = await run(["unknown-command"]);
    expect(exitCode).toBe(2);
  });

  it("supports top-level help without requiring a config or command", async () => {
    const output = vi.spyOn(console, "log").mockImplementation(() => {});

    expect(await run(["--help"])).toBe(0);
    expect(output.mock.calls.flat().join("\n")).toContain("usage-capture");
  });

  it("refuses collection when no account is enabled", async () => {
    const config = defaultConfig();
    const account = config.accounts[0]!;
    account.enabled = false;
    account.browser.adapter = "fixture_history";
    account.browser.profilePath = "";
    config.application.stateDirectory = stateDirectory;
    const configPath = join(stateDirectory, "disabled-config.json");
    saveConfig(config, configPath);
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const fixtureRoot = new URL("../fixtures/v1/", import.meta.url).pathname;

    expect(await run([
      "backfill",
      "--config",
      configPath,
      "--fixture-root",
      fixtureRoot,
    ])).toBe(2);
    expect(error).toHaveBeenCalledWith(
      "usage-capture backfill: no enabled account is configured",
    );
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

  it("clears an authentication pause only after interactive bootstrap verification", async () => {
    const config = defaultConfig();
    const account = config.accounts[0]!;
    account.id = "fixture-primary";
    account.expectedProviderUserId = "user-abc123";
    account.expectedWorkspaceId = "ws-xyz";
    account.quotaOwnerId = "user-abc123";
    config.application.stateDirectory = stateDirectory;
    config.application.databasePath = join(stateDirectory, "ledger.sqlite");
    const configPath = join(stateDirectory, "config.json");
    saveConfig(config, configPath);

    const ledger = new Ledger(config.application.databasePath);
    const scope = configuredScope(account);
    ledger.upsertAccount(scope, { authState: "paused" });
    const store = new SqliteCheckpointStore(ledger, scope);
    store.saveAccountState({
      status: "paused",
      reason: "authentication",
      pausedAt: "2026-09-07T12:00:00.000Z",
      cooldownUntil: null,
      lastError: "auth_required",
    });
    ledger.transaction(() => store.persist());
    ledger.close();

    vi.mocked(bootstrapAccount).mockResolvedValueOnce({
      state: "ready",
      accountId: account.id,
      profilePath: account.browser.profilePath,
      identity: {
        providerUserId: "user-abc123",
        workspaceId: "ws-xyz",
        quotaOwnerId: "user-abc123",
        surface: "chat",
        authState: "ready",
        identityErrors: [],
      },
      capabilities: emptyCapabilities(),
      interactiveLoginUsed: true,
      liveVerification: "passed",
      notes: [],
    });
    const output = vi.spyOn(console, "log").mockImplementation(() => {});

    expect(
      await run([
        "bootstrap",
        "--config",
        configPath,
        "--interactive-login",
      ]),
    ).toBe(0);

    const reopened = new Ledger(config.application.databasePath);
    try {
      const restored = new SqliteCheckpointStore(reopened, scope);
      expect(restored.loadAccountState()).toEqual({
        status: "ready",
        reason: null,
        pausedAt: null,
        cooldownUntil: null,
        lastError: null,
      });
      expect(reopened.listAccounts()[0]?.authState).toBe("ready");
      expect(output.mock.calls.at(-1)?.[0]).toContain(
        "cleared persisted authentication pause after verified interactive recovery",
      );
    } finally {
      reopened.close();
    }
  });

  it("collects, replays, reports, and rebuilds through one configured SQLite database", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-09-07T12:00:00.000Z"));
    const output = vi.spyOn(console, "log").mockImplementation(() => {});
    const config = defaultConfig();
    const account = config.accounts[0]!;
    account.id = "fixture-primary";
    account.expectedProviderUserId = "user-abc123";
    account.expectedWorkspaceId = "ws-xyz";
    account.quotaOwnerId = "user-abc123";
    account.browser.adapter = "fixture_history";
    account.browser.profilePath = "";
    config.application.stateDirectory = stateDirectory;
    config.application.databasePath = join(stateDirectory, "ledger.sqlite");

    const configPath = join(stateDirectory, "fixture-config.json");
    saveConfig(config, configPath);
    const fixtureRoot = new URL("../fixtures/v1/", import.meta.url).pathname;

    const collectArgs = ["--config", configPath, "--fixture-root", fixtureRoot];
    const localArgs = ["--config", configPath];
    const end = "2026-09-08T00:00:00.000Z";
    const lastResult = () => JSON.parse(String(output.mock.calls.at(-1)?.[0]));
    expect(await run(["backfill", ...collectArgs, "--until", end])).toBe(0);
    const first = lastResult();
    expect(first.ledger).toMatchObject({
      committed: true,
      databasePath: config.application.databasePath,
      observationsInserted: 2,
      messagesInserted: 5,
      attemptsInserted: 3,
    });
    expect(first.range).toEqual({
      start: "2026-08-25T00:00:00.000Z",
      end,
    });
    expect(first.conversations.map((item: { messageCount: number }) => item.messageCount))
      .toEqual([3, 2]);
    expect(await run(["backfill", ...collectArgs, "--until", end])).toBe(0);
    expect(lastResult().ledger).toMatchObject({
      observationsInserted: 0,
      messagesInserted: 0,
      messagesDeduplicated: 5,
      attemptsInserted: 0,
      attemptsUpdated: 0,
      attemptsDeduplicated: 3,
    });

    expect(await run(["report", ...localArgs, "--last-hours", "168", "--until", end]))
      .toBe(0);
    const report = lastResult();
    expect(report.includedAttempts).toBe(3);
    // Provisional branches cannot inherit the prompt's requested-model evidence.
    expect(report.observedAttemptsByRequestedModel).toEqual({});
    expect(report.completedAnswersByRecordedFinalModel).toEqual({});
    expect(report.possibleCompletedAnswersByRecordedFinalModel)
      .toEqual({ "gpt-5.6-astra-pro": 2 });
    expect(report.modelMismatches).toBe(0);
    expect(report.coverageGaps.some(
      (gap: { reason: string }) => gap.reason === "incomplete_history_coverage",
    )).toBe(true);

    expect(await run(["report", ...localArgs, "--since", "24h", "--until", end]))
      .toBe(0);
    expect(lastResult()).toMatchObject({
      durationMs: 24 * 60 * 60 * 1000,
      start: "2026-09-07T00:00:00.000Z",
      end,
    });

    // Unknown-time provisional branches remain possible, never definite,
    // regardless of the conversation's newer update time.
    expect(await run(["report", ...localArgs, "--last-hours", "24", "--until", end])).toBe(0);
    expect(lastResult().includedAttempts).toBe(2);
    expect(lastResult().possibleAttemptIds).toHaveLength(2);
    expect(lastResult().observedAttemptsByRequestedModel).toEqual({});
    expect(lastResult().completedAnswersByRecordedFinalModel).toEqual({});
    expect(await run(["models", ...localArgs])).toBe(0);
    expect(lastResult()).toMatchObject({
      mapping_version: "initial-unmapped",
      review_status: "draft",
    });
    expect(lastResult().suggestions[0]?.suggestedFamily).toBeNull();

    const ledger = new Ledger(config.application.databasePath);
    try {
      const scope = ledger.accountScope("fixture-primary");
      const attempts = ledger.listAttempts(scope);
      const evidenceCounts = () => Object.fromEntries(
        ["observations", "message_revisions", "attempt_revisions", "attempt_aliases"]
          .map((table) => [table, ledger.db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get()]),
      );
      const priorCounts = evidenceCounts();
      const rebuildArgs = ["rebuild", ...localArgs, "--until", end];
      expect(await run(rebuildArgs)).toBe(0);
      const preview = lastResult();
      expect(preview).toMatchObject({ mode: "preview", rebuiltAttempts: 0 });
      expect(preview.report.attemptIds).toEqual(report.attemptIds);
      expect(ledger.aggregateRevisions(scope)).toHaveLength(0);
      expect(ledger.listAttempts(scope)).toEqual(attempts);
      expect(evidenceCounts()).toEqual(priorCounts);
      expect(await run([...rebuildArgs, "--apply"])).toBe(0);
      expect(lastResult().revisionId).toBe(preview.revisionId);
      expect(ledger.aggregateRevisions(scope)).toHaveLength(1);
      expect(ledger.listAttempts(scope)).toEqual(attempts);

      expect(await run(["refresh", ...collectArgs])).toBe(0);
      expect(lastResult().mode).toBe("incremental");
      expect(lastResult().scopes[0]?.candidateCutoff).toBe("2026-08-24T12:00:00.000Z");
      expect(await run([
        "reconcile", ...collectArgs, "--since", "2026-09-01T00:00:00.000Z", "--until", end,
      ])).toBe(0);
      expect(lastResult().conversations).toHaveLength(2);
      expect(ledger.listAttempts(scope)).toEqual(attempts);
      expect(evidenceCounts()).toEqual(priorCounts);
      const store = new SqliteCheckpointStore(ledger, scope);
      expect(store.loadDiscovery("active")?.status).toBe("complete");
      expect(store.loadDiscovery("archived")?.status).toBe("complete");
    } finally {
      ledger.close();
    }
  });
});
