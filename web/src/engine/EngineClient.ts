import type {
  BotProfile,
  HintPolicy,
  ReplayRecord,
  WorkerRequest,
  WorkerResponse,
} from "../../../engine/wasm/js/src/protocol";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";

type Pending = {
  resolve: (response: WorkerResponse) => void;
  reject: (error: Error) => void;
};

type WithoutRequestId<T> = T extends WorkerRequest
  ? Omit<T, "requestId">
  : never;
type ClientRequest = WithoutRequestId<WorkerRequest>;

const READY_TIMEOUT_MS = 30_000;

export interface TransitionResult {
  snapshot: UiSnapshot;
  animationHints: Extract<
    WorkerResponse,
    { type: "transition" }
  >["animationHints"];
}

export interface ProgressResult {
  snapshot: UiSnapshot;
  animationHints: Extract<
    WorkerResponse,
    { type: "progress" }
  >["animationHints"];
}

export class EngineClient {
  readonly ready: Promise<number>;

  private readonly worker: Worker;
  private readonly pending = new Map<string, Pending>();
  private readonly progressListeners = new Set<
    (transition: ProgressResult) => void
  >();
  private nextRequest = 0;
  private resolveReady!: (version: number) => void;
  private rejectReady!: (error: Error) => void;
  private readyTimer: ReturnType<typeof setTimeout> | undefined;
  private disposeTimer: ReturnType<typeof setTimeout> | undefined;
  private readySettled = false;

  constructor() {
    this.ready = new Promise<number>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.worker = new Worker(new URL("./engine.worker.ts", import.meta.url), {
      type: "module",
      name: "bloodflow-engine",
    });
    this.worker.addEventListener("message", this.onMessage);
    this.worker.addEventListener("error", (event) => {
      const error = new Error(
        event.message || "The engine worker stopped unexpectedly.",
      );
      this.fail(error);
    });
    this.readyTimer = setTimeout(() => {
      const error = new Error(
        `The engine worker did not initialize within ${READY_TIMEOUT_MS / 1000} seconds.`,
      );
      this.worker.terminate();
      this.fail(error);
    }, READY_TIMEOUT_MS);
  }

  dispose(): void {
    if (this.disposeTimer !== undefined) return;
    // React StrictMode runs an effect cleanup followed immediately by a new
    // setup in development. Defer teardown one task so that transient cleanup
    // can be cancelled without killing the shared worker.
    this.disposeTimer = setTimeout(() => {
      this.disposeTimer = undefined;
      this.worker.removeEventListener("message", this.onMessage);
      this.worker.terminate();
      this.fail(new Error("The engine client was disposed."));
    }, 0);
  }

  cancelDispose(): void {
    if (this.disposeTimer !== undefined) {
      clearTimeout(this.disposeTimer);
      this.disposeTimer = undefined;
    }
  }

  subscribeProgress(
    listener: (transition: ProgressResult) => void,
  ): () => void {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  async newGame(
    seed: string,
    humanSeat: number,
    botProfiles: readonly [BotProfile, BotProfile, BotProfile, BotProfile],
  ): Promise<UiSnapshot> {
    await this.ready;
    const response = await this.request({
      type: "new_game",
      seed,
      humanSeat,
      botProfiles,
    });
    return this.expectSnapshot(response);
  }

  async submit(actionId: number): Promise<TransitionResult> {
    const response = await this.request({ type: "submit", actionId });
    if (response.type !== "transition") {
      throw new Error(`Expected transition response, got ${response.type}.`);
    }
    return {
      snapshot: response.snapshot,
      animationHints: response.animationHints,
    };
  }

  async hint(policy: HintPolicy): Promise<number | null> {
    const response = await this.request({ type: "request_hint", policy });
    if (response.type !== "hint") {
      throw new Error(`Expected hint response, got ${response.type}.`);
    }
    return response.actionId;
  }

  async pause(): Promise<UiSnapshot> {
    return this.expectSnapshot(await this.request({ type: "pause" }));
  }

  async resume(): Promise<UiSnapshot> {
    return this.expectSnapshot(await this.request({ type: "resume" }));
  }

  async exportReplay(): Promise<ReplayRecord> {
    const response = await this.request({ type: "export_replay" });
    if (response.type !== "replay") {
      throw new Error(`Expected replay response, got ${response.type}.`);
    }
    return response.replay;
  }

  async loadReplay(replay: ReplayRecord): Promise<UiSnapshot> {
    return this.expectSnapshot(
      await this.request({ type: "load_replay", replay }),
    );
  }

  private readonly onMessage = (event: MessageEvent<WorkerResponse>): void => {
    const response = event.data;
    if (response.requestId === "boot") {
      if (response.type === "ready") {
        if (!this.readySettled) {
          this.readySettled = true;
          this.clearReadyTimer();
          this.resolveReady(response.engineRulesVersion);
        }
      } else if (response.type === "error") {
        this.fail(new Error(response.message));
      }
      return;
    }
    if (response.type === "progress") {
      const transition: ProgressResult = {
        snapshot: response.snapshot,
        animationHints: response.animationHints,
      };
      for (const listener of this.progressListeners) listener(transition);
      return;
    }
    const request = this.pending.get(response.requestId);
    if (request == null) {
      return;
    }
    this.pending.delete(response.requestId);
    if (response.type === "error") {
      request.reject(new Error(response.message));
    } else {
      request.resolve(response);
    }
  };

  private request(request: ClientRequest): Promise<WorkerResponse> {
    const requestId = String(++this.nextRequest);
    return new Promise<WorkerResponse>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      this.worker.postMessage({ ...request, requestId });
    });
  }

  private clearReadyTimer(): void {
    if (this.readyTimer !== undefined) {
      clearTimeout(this.readyTimer);
      this.readyTimer = undefined;
    }
  }

  private fail(error: Error): void {
    if (!this.readySettled) {
      this.readySettled = true;
      this.clearReadyTimer();
      this.rejectReady(error);
    }
    for (const request of this.pending.values()) {
      request.reject(error);
    }
    this.pending.clear();
  }

  private expectSnapshot(response: WorkerResponse): UiSnapshot {
    if (response.type !== "snapshot") {
      throw new Error(`Expected snapshot response, got ${response.type}.`);
    }
    return response.snapshot;
  }
}
