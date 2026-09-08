import * as scheduleTransitions from "./transitions.js";
import {
  SchedulerTransitionError,
  validateEpoch,
  validateScope,
} from "./types.js";
import type {
  CompleteTriggerRequest,
  RefreshRequestResult,
  ScheduleMutationResult,
  ScheduleOptions,
  ScheduleScope,
  ScheduleState,
  TriggerClaim,
} from "./types.js";

export interface BridgeTransport {
  request(
    operation: "loadState" | "compareAndSetState",
    payload: unknown,
  ): Promise<BridgeResult>;
}

export interface BridgeEnvelope {
  readonly runId: string;
  readonly bindingGeneration: number;
  readonly leaseFencingToken: number;
}

export interface BridgeResult {
  readonly ok: true;
  readonly result: {
    readonly state: ScheduleState | null;
    readonly stateVersion: number;
  };
}

export class ScheduleBridgeStore {
  constructor(
    private readonly transport: BridgeTransport,
    private readonly envelope: BridgeEnvelope,
  ) {}

  async ensureSchedule(
    scope: ScheduleScope,
    options: ScheduleOptions,
    at: number,
  ): Promise<ScheduleState> {
    const loaded = await this.load(scope);
    if (loaded.state) {
      return loaded.state;
    }
    const next = scheduleTransitions.createScheduleState(
      scope,
      options,
      { at },
      loaded.stateVersion + 1,
    );
    const stored = await this.cas(scope, loaded.stateVersion, next, at);
    return stored ?? next;
  }

  async requestRefresh(scope: ScheduleScope, at: number): Promise<RefreshRequestResult> {
    return this.mutate(scope, at, (state) => {
      const result = scheduleTransitions.requestRefresh(state, { at });
      return { result, nextState: result.state };
    });
  }

  async claimTrigger(scope: ScheduleScope, at: number): Promise<TriggerClaim | null> {
    const loaded = await this.load(scope);
    const state = requireState(scope, loaded.state);
    const claimed = scheduleTransitions.claimTrigger(
      state,
      { at },
      this.envelope.leaseFencingToken,
    );
    if (!claimed) {
      return null;
    }
    const stored = await this.cas(scope, loaded.stateVersion, claimed.state, at);
    return stored ? { ...claimed, state: stored } : claimed;
  }

  async completeTrigger(
    scope: ScheduleScope,
    at: number,
    request: CompleteTriggerRequest,
  ): Promise<ScheduleMutationResult> {
    return this.mutate(scope, at, (state) => {
      const result = scheduleTransitions.completeTrigger(
        state,
        { at },
        this.envelope.leaseFencingToken,
        request,
      );
      return { result, nextState: result.state };
    });
  }

  private async load(scope: ScheduleScope): Promise<BridgeResult["result"]> {
    validateScope(scope);
    const response = await this.transport.request("loadState", { kind: "schedule", scope });
    return response.result;
  }

  private async cas(
    scope: ScheduleScope,
    expectedStateVersion: number,
    state: ScheduleState | null,
    at: number,
  ): Promise<ScheduleState | null> {
    validateEpoch(at, "at");
    const response = await this.transport.request("compareAndSetState", {
      kind: "schedule",
      scope,
      expectedStateVersion,
      state,
      fencingToken: this.envelope.leaseFencingToken,
      bindingGeneration: this.envelope.bindingGeneration,
    });
    return response.result.state;
  }

  private async mutate<T extends RefreshRequestResult | ScheduleMutationResult>(
    scope: ScheduleScope,
    at: number,
    mutate: (state: ScheduleState | null) => {
      result: T;
      nextState: ScheduleState | null;
    },
  ): Promise<T> {
    const loaded = await this.load(scope);
    const mutation = mutate(loaded.state);
    const stored = await this.cas(scope, loaded.stateVersion, mutation.nextState, at);
    if (stored && "state" in mutation.result && mutation.result.state) {
      return { ...mutation.result, state: stored } as T;
    }
    return mutation.result;
  }
}

function requireState(scope: ScheduleScope, state: ScheduleState | null): ScheduleState {
  validateScope(scope);
  if (!state) {
    throw new SchedulerTransitionError("invalid_schedule_state", "schedule state is missing");
  }
  return state;
}
