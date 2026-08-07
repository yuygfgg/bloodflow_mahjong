import assert from "node:assert/strict";
import { before, test } from "node:test";
import { PHASE_TURN } from "../../pkg/bloodflow_mahjong_wasm.js";
import {
  assertActionLegal,
  assertRejects,
  loadWasm,
  playUntil,
  simpleRulePolicy,
  type WasmModule,
} from "./harness.ts";

let wasm: WasmModule;

before(async () => {
  wasm = await loadWasm();
});

test("RuleEvConfig validates its search budget", async () => {
  const standard = wasm.RuleEvConfig.standard();
  assert.equal(standard.searchDepth, 1);
  assert.equal(standard.defense, true);
  const fast = wasm.RuleEvConfig.fast();
  assert.equal(fast.searchDepth, 0);
  assert.equal(fast.defense, true);
  await assertRejects(() => new wasm.RuleEvConfig(4, true), "search_depth");

  standard.free();
  fast.free();
});

test("rule-ev actions stay legal", () => {
  const game = new wasm.Game(43n);
  const initialAction = game.ruleEvAction();
  assert.notEqual(initialAction, undefined);
  assertActionLegal(game, initialAction!);

  playUntil(game, simpleRulePolicy, {
    maxSteps: 256,
    stopWhen: (g) => g.phase === PHASE_TURN(),
  });

  const fast = wasm.RuleEvConfig.fast();
  const action = game.ruleEvActionWithConfig(fast);
  assert.notEqual(action, undefined);
  assertActionLegal(game, action!);

  fast.free();
  game.free();
});

test("rule-ev can finish a short seeded game", () => {
  const game = new wasm.Game(101n);
  const config = wasm.RuleEvConfig.fast();
  const { steps } = playUntil(
    game,
    (g) => g.ruleEvActionWithConfig(config),
    { maxSteps: 2_000 },
  );
  assert.ok(steps > 0);
  assert.equal(game.phase, 5);
  config.free();
  game.free();
});
