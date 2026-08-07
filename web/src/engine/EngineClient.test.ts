import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  EngineCapabilities,
  WorkerResponse,
} from "../../../engine/wasm/js/src/protocol";
import { EngineClient } from "./EngineClient";

class FakeWorker {
  static latest: FakeWorker;

  private readonly listeners = new Map<
    string,
    Set<EventListenerOrEventListenerObject>
  >();

  constructor() {
    FakeWorker.latest = this;
  }

  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
  ): void {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
  ): void {
    this.listeners.get(type)?.delete(listener);
  }

  postMessage(): void {}

  terminate(): void {}

  emit(response: WorkerResponse): void {
    const event = { data: response } as MessageEvent<WorkerResponse>;
    for (const listener of this.listeners.get("message") ?? []) {
      if (typeof listener === "function") listener(event);
      else listener.handleEvent(event);
    }
  }
}

describe("EngineClient capabilities", () => {
  beforeEach(() => {
    vi.stubGlobal("Worker", FakeWorker as unknown as typeof Worker);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("becomes ready before optional NN detection completes", async () => {
    const client = new EngineClient();
    const updates: EngineCapabilities[] = [];
    const unsubscribe = client.subscribeCapabilities((capabilities) =>
      updates.push(capabilities),
    );

    FakeWorker.latest.emit({
      type: "ready",
      requestId: "boot",
      engineRulesVersion: 10,
      nnAvailable: false,
    });

    await expect(client.ready).resolves.toEqual({
      engineRulesVersion: 10,
      nnAvailable: false,
    });
    expect(updates).toEqual([{ engineRulesVersion: 10, nnAvailable: false }]);

    FakeWorker.latest.emit({
      type: "capabilities",
      requestId: "capabilities",
      engineRulesVersion: 10,
      nnAvailable: true,
    });
    expect(updates).toEqual([
      { engineRulesVersion: 10, nnAvailable: false },
      { engineRulesVersion: 10, nnAvailable: true },
    ]);

    unsubscribe();
    client.dispose();
  });
});
