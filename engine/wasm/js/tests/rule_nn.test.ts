import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { before, test } from "node:test";
import { PHASE_FINISHED } from "../../pkg/bloodflow_mahjong_wasm.js";
import {
  MODEL_PATH,
  assertActionLegal,
  loadWasm,
  playUntil,
  type WasmModule,
  type WasmRuleNn,
} from "./harness.ts";

let wasm: WasmModule;
let modelBytes: Uint8Array | undefined;

before(async () => {
  wasm = await loadWasm();
  try {
    modelBytes = new Uint8Array(await readFile(MODEL_PATH));
  } catch {
    modelBytes = undefined;
  }
});

test("RuleNn loads the bundled model and returns a legal action", async (t) => {
  if (modelBytes == null || modelBytes.byteLength === 0) {
    t.skip(`model not found at ${MODEL_PATH}`);
    return;
  }

  const policy: WasmRuleNn = new wasm.RuleNn(modelBytes);
  const game = new wasm.Game(42n);
  const action = policy.action(game);
  assert.notEqual(action, undefined);
  assertActionLegal(game, action!);
  game.free();
  policy.free();
});

test("RuleNn can finish one seeded game", async (t) => {
  if (modelBytes == null || modelBytes.byteLength === 0) {
    t.skip(`model not found at ${MODEL_PATH}`);
    return;
  }

  const policy: WasmRuleNn = new wasm.RuleNn(modelBytes);
  const game = new wasm.Game(42n);
  const { steps } = playUntil(game, (g) => policy.action(g), { maxSteps: 5_000 });
  assert.ok(steps > 0);
  assert.equal(game.phase, PHASE_FINISHED());
  assert.equal(policy.action(game), undefined);
  game.free();
  policy.free();
});
