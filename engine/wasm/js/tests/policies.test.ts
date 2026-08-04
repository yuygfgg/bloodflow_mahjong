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

test("RuleEvConfig and RulePlannerConfig validate budgets", async () => {
  const standard = wasm.RuleEvConfig.standard();
  assert.equal(standard.searchDepth, 1);
  assert.equal(standard.defense, true);
  const fast = wasm.RuleEvConfig.fast();
  assert.equal(fast.searchDepth, 0);
  assert.equal(fast.defense, true);
  await assertRejects(() => new wasm.RuleEvConfig(4, true), "search_depth");

  const planner = wasm.RulePlannerConfig.defaultConfig();
  assert.deepEqual(
    [
      planner.handChanges,
      planner.drawHorizon,
      planner.candidateStates,
      planner.beliefWorlds,
      planner.responseWorlds,
      planner.searchIterations,
    ],
    [0, 1, 1, 64, 0, 64],
  );

  const invalid: Array<[string, () => unknown]> = [
    ["hand_changes", () => new wasm.RulePlannerConfig(3, 1, 1, 64, 0, 64)],
    ["draw_horizon", () => new wasm.RulePlannerConfig(0, 33, 1, 64, 0, 64)],
    ["candidate_states", () => new wasm.RulePlannerConfig(0, 1, 0, 64, 0, 64)],
    ["belief_worlds", () => new wasm.RulePlannerConfig(0, 1, 1, 257, 0, 64)],
    ["response_worlds", () => new wasm.RulePlannerConfig(0, 1, 1, 64, 257, 64)],
    ["search_iterations", () => new wasm.RulePlannerConfig(0, 1, 1, 64, 0, 4097)],
  ];
  for (const [name, build] of invalid) {
    await assertRejects(build, name);
  }

  standard.free();
  fast.free();
  planner.free();
});

test("rule-ev and minimal planner actions stay legal", () => {
  const game = new wasm.Game(43n);
  const minimalPlanner = new wasm.RulePlannerConfig(0, 0, 1, 64, 0, 0);

  for (const action of [
    game.ruleEvAction(),
    game.rulePlannerActionWithConfig(minimalPlanner),
  ]) {
    assert.notEqual(action, undefined);
    assertActionLegal(game, action!);
  }

  playUntil(game, simpleRulePolicy, {
    maxSteps: 256,
    stopWhen: (g) => g.phase === PHASE_TURN(),
  });

  const fast = wasm.RuleEvConfig.fast();
  const actions = [
    game.ruleEvActionWithConfig(fast),
    game.rulePlannerActionWithConfig(minimalPlanner),
  ];
  for (const action of actions) {
    assert.notEqual(action, undefined);
    assertActionLegal(game, action!);
  }

  fast.free();
  minimalPlanner.free();
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
