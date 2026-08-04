import assert from "node:assert/strict";
import { before, test } from "node:test";
import { PHASE_FINISHED } from "../../pkg/bloodflow_mahjong_wasm.js";
import {
  cloneBytes,
  loadWasm,
  playUntil,
  publicFingerprint,
  replayActions,
  simpleRulePolicy,
  type WasmModule,
} from "./harness.ts";

let wasm: WasmModule;

before(async () => {
  wasm = await loadWasm();
});

function terminalFingerprint(game: {
  scores(): Int32Array;
  rankings(): Uint8Array;
  terminationReason: number | undefined;
  phase: number;
}): string {
  return [
    game.phase,
    cloneBytes(game.scores()).join(","),
    cloneBytes(game.rankings()).join(","),
    game.terminationReason ?? -1,
  ].join("|");
}

test("same seed and action sequence replays to the same terminal state", () => {
  for (const seed of [42n, 7n, 99n]) {
    const live = new wasm.Game(seed);
    const { actions } = playUntil(live, simpleRulePolicy);
    assert.equal(live.phase, PHASE_FINISHED());

    const replay = replayActions(wasm, seed, actions);
    assert.equal(terminalFingerprint(replay), terminalFingerprint(live));
    assert.equal(publicFingerprint(replay), publicFingerprint(live));
    live.free();
    replay.free();
  }
});

test("different seeds produce independent action streams", () => {
  const a = new wasm.Game(1n);
  const b = new wasm.Game(2n);
  const actionsA = playUntil(a, simpleRulePolicy).actions;
  const actionsB = playUntil(b, simpleRulePolicy).actions;
  assert.notEqual(actionsA.join(","), actionsB.join(","));
  a.free();
  b.free();
});
