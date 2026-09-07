import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { resolve } from "node:path";

import type {
  DiscoveryCheckpoint,
  HistoryCheckpointStore,
  HistoryScope,
  RevisitEntry,
} from "../contracts/history.js";
import { HISTORY_STATE_VERSION } from "../contracts/history.js";

interface PersistedHistoryState {
  stateVersion: typeof HISTORY_STATE_VERSION;
  accountId: string;
  checkpoints: Partial<Record<HistoryScope, DiscoveryCheckpoint>>;
  revisits: RevisitEntry[];
}

/**
 * File-backed Stage-2A state. It contains only sanitized IDs, timestamps,
 * pagination hints, and coverage status; browser state and response bodies
 * never enter this file.
 */
export class JsonCheckpointStore implements HistoryCheckpointStore {
  private state: PersistedHistoryState | null = null;
  private readonly path: string;

  constructor(
    private readonly stateDirectory: string,
    private readonly accountId: string,
  ) {
    this.path = resolve(
      stateDirectory,
      "history",
      `${encodeURIComponent(accountId)}.json`,
    );
  }

  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null {
    return this.load().checkpoints[scope] ?? null;
  }

  saveDiscovery(checkpoint: DiscoveryCheckpoint): void {
    const state = this.load();
    state.checkpoints[checkpoint.scope] = clone(checkpoint);
    this.persist(state);
  }

  listRevisits(): RevisitEntry[] {
    return this.load()
      .revisits.filter((entry) => entry.status === "pending")
      .map(clone)
      .sort((left, right) =>
        left.nextEligibleAt.localeCompare(right.nextEligibleAt) ||
        left.conversationId.localeCompare(right.conversationId),
      );
  }

  upsertRevisit(entry: RevisitEntry): void {
    const state = this.load();
    const index = state.revisits.findIndex(
      (candidate) =>
        candidate.accountId === entry.accountId &&
        candidate.conversationId === entry.conversationId,
    );
    if (index < 0) {
      state.revisits.push(clone(entry));
    } else {
      state.revisits[index] = clone(entry);
    }
    this.persist(state);
  }

  completeRevisit(accountId: string, conversationId: string): void {
    const state = this.load();
    state.revisits = state.revisits.filter(
      (entry) =>
        entry.accountId !== accountId ||
        entry.conversationId !== conversationId,
    );
    this.persist(state);
  }

  private load(): PersistedHistoryState {
    if (this.state) {
      return this.state;
    }
    if (!existsSync(this.path)) {
      this.state = emptyState(this.accountId);
      return this.state;
    }
    const parsed = JSON.parse(readFileSync(this.path, "utf8")) as unknown;
    if (!isPersistedState(parsed) || parsed.accountId !== this.accountId) {
      throw new Error("history checkpoint state is invalid");
    }
    this.state = parsed;
    return parsed;
  }

  private persist(state: PersistedHistoryState): void {
    const directory = resolve(this.stateDirectory, "history");
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    const temporaryPath = `${this.path}.${process.pid}.${Date.now()}.tmp`;
    writeFileSync(temporaryPath, JSON.stringify(state, null, 2) + "\n", {
      encoding: "utf8",
      mode: 0o600,
    });
    try {
      chmodSync(temporaryPath, 0o600);
    } catch {
      // Best-effort local permission hardening.
    }
    renameSync(temporaryPath, this.path);
    this.state = state;
  }
}

export class MemoryCheckpointStore implements HistoryCheckpointStore {
  private readonly checkpoints = new Map<HistoryScope, DiscoveryCheckpoint>();
  private readonly revisits = new Map<string, RevisitEntry>();

  loadDiscovery(scope: HistoryScope): DiscoveryCheckpoint | null {
    const checkpoint = this.checkpoints.get(scope);
    return checkpoint ? clone(checkpoint) : null;
  }

  saveDiscovery(checkpoint: DiscoveryCheckpoint): void {
    this.checkpoints.set(checkpoint.scope, clone(checkpoint));
  }

  listRevisits(): RevisitEntry[] {
    return [...this.revisits.values()]
      .filter((entry) => entry.status === "pending")
      .map(clone)
      .sort((left, right) =>
        left.nextEligibleAt.localeCompare(right.nextEligibleAt) ||
        left.conversationId.localeCompare(right.conversationId),
      );
  }

  upsertRevisit(entry: RevisitEntry): void {
    this.revisits.set(entry.conversationId, clone(entry));
  }

  completeRevisit(accountId: string, conversationId: string): void {
    const entry = this.revisits.get(conversationId);
    if (entry?.accountId === accountId) {
      this.revisits.delete(conversationId);
    }
  }
}

function emptyState(accountId: string): PersistedHistoryState {
  return {
    stateVersion: HISTORY_STATE_VERSION,
    accountId,
    checkpoints: {},
    revisits: [],
  };
}

function isPersistedState(value: unknown): value is PersistedHistoryState {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    candidate.stateVersion === HISTORY_STATE_VERSION &&
    typeof candidate.accountId === "string" &&
    candidate.checkpoints !== null &&
    typeof candidate.checkpoints === "object" &&
    !Array.isArray(candidate.checkpoints) &&
    Array.isArray(candidate.revisits)
  );
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
