import assert from "node:assert/strict";
import { before, test } from "node:test";
import {
  ACTION_PASS,
  ACTION_SPACE_SIZE,
  PHASE_EXCHANGE,
  PHASE_FINISHED,
  PHASE_TURN,
  SHANTEN_COMPLETE,
  SHANTEN_MAX,
  STEP_RECORD_WIDTH,
  TILE_KIND_COUNT,
} from "../../pkg/bloodflow_mahjong_wasm.js";
import {
  assertActionLegal,
  assertRejects,
  cloneBytes,
  createEventBuffers,
  createObservationBuffers,
  createOracleBuffer,
  createStepRecord,
  createTileHistogram,
  firstLegalAction,
  loadWasm,
  playUntil,
  publicFingerprint,
  simpleRulePolicy,
  sumHistogram,
  type WasmModule,
} from "./harness.ts";

let wasm: WasmModule;

before(async () => {
  wasm = await loadWasm();
});

test("Game starts in exchange and fills basic state buffers", () => {
  const game = new wasm.Game(42n);
  assert.equal(game.phase, PHASE_EXCHANGE());
  assert.ok(game.decision != null);
  assert.deepEqual(Array.from(game.decision!), [0, PHASE_EXCHANGE()]);

  const action = game.simpleRuleAction();
  assert.notEqual(action, undefined);
  assertActionLegal(game, action!);

  const hand = createTileHistogram();
  game.concealedInto(0, hand);
  assert.equal(sumHistogram(hand), 14);

  const analysis = game.handAnalysis(0);
  assert.equal(analysis.length, 2);
  assert.ok(analysis[0]! >= SHANTEN_COMPLETE() && analysis[0]! <= SHANTEN_MAX());
  assert.equal((analysis[1]! >>> 0) >> TILE_KIND_COUNT(), 0);

  const record = game.stepId(firstLegalAction(game));
  assert.equal(record.length, STEP_RECORD_WIDTH());
  game.free();
});

test("withExchangeDirection and reset keep a usable environment", () => {
  const GameCtor = wasm.Game as unknown as {
    new (seed: bigint): import("./harness.ts").WasmGame;
    withExchangeDirection(seed: bigint, direction: number): import("./harness.ts").WasmGame;
  };
  const game = GameCtor.withExchangeDirection(7n, 2);
  assert.equal(game.exchangeDirection, 2);
  game.reset(9n);
  assert.equal(game.phase, PHASE_EXCHANGE());
  assert.ok(game.simpleRuleAction() != null);
  game.free();
});

test("caller-owned step and observation buffers validate length", async () => {
  const game = new wasm.Game(11n);
  const action = firstLegalAction(game);
  const good = createStepRecord();
  const before = publicFingerprint(game);

  await assertRejects(() => game.stepInto(action, new Int32Array(3)), "output length");
  assert.equal(publicFingerprint(game), before, "invalid stepInto must not mutate");

  game.stepInto(action, good);
  assert.equal(good.length, STEP_RECORD_WIDTH());

  const obs = createObservationBuffers();
  await assertRejects(
    () =>
      game.observeInto(
        0,
        new Uint8Array(10),
        obs.melds,
        obs.river,
        obs.meta,
      ),
    "tile_obs",
  );
  game.observeInto(0, obs.tileObs, obs.melds, obs.river, obs.meta);
  assert.equal(obs.meta[0], game.phase);

  await assertRejects(() => game.handAnalysis(4), "seat");
  await assertRejects(() => game.concealedInto(0, new Uint8Array(3)), "output length");
  game.free();
});

test("illegal actions are rejected atomically", async () => {
  const game = new wasm.Game(13n);
  const before = publicFingerprint(game);
  const illegal = ACTION_PASS(); // not legal in exchange
  assert.notEqual(illegal, firstLegalAction(game));
  await assertRejects(() => game.stepId(illegal));
  assert.equal(publicFingerprint(game), before);
  await assertRejects(() => game.stepId(ACTION_SPACE_SIZE()), "out of range");
  assert.equal(publicFingerprint(game), before);
  game.free();
});

test("event history and step deltas use flat caller buffers", () => {
  const game = new wasm.Game(17n);
  const events = createEventBuffers();
  const initial = game.eventsInto(0, events.history);
  assert.ok(initial >= 1);

  const action = firstLegalAction(game);
  game.stepId(action);
  const stepCount = game.stepEventsInto(0, events.step);
  assert.ok(stepCount >= 1);
  assert.ok((events.step[0] ?? -1) >= 0);
  game.free();
});

test("event buffer capacity and alignment are enforced without mutation", async () => {
  const game = new wasm.Game(19n);
  const before = publicFingerprint(game);
  await assertRejects(() => game.eventsInto(0, new Int32Array(7)), "multiple");
  await assertRejects(() => game.eventsInto(0, new Int32Array(0)), "capacity");
  assert.equal(publicFingerprint(game), before);
  game.free();
});

test("observation hides another player's draw tile", () => {
  const game = new wasm.Game(127n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 256,
    stopWhen: (g) => g.currentDraw != null,
  });
  assert.ok(game.currentDraw != null);
  const drawer = game.currentDraw![0]!;
  const tile = game.currentDraw![1]!;
  const other = (drawer + 1) % 4;

  const drawerObs = createObservationBuffers();
  const otherObs = createObservationBuffers();
  game.observeInto(drawer, drawerObs.tileObs, drawerObs.melds, drawerObs.river, drawerObs.meta);
  game.observeInto(other, otherObs.tileObs, otherObs.melds, otherObs.river, otherObs.meta);

  assert.equal(drawerObs.meta[5], tile);
  assert.equal(otherObs.meta[5], -1);
  assert.equal(drawerObs.meta[1], drawer);
  assert.equal(otherObs.meta[1], drawer);
  game.free();
});

test("information-set resampling preserves actor observation", () => {
  const game = new wasm.Game(29n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 64,
    stopWhen: (g) => g.phase === PHASE_TURN(),
  });
  const actor = game.decision![0]!;
  const original = createObservationBuffers();
  game.observeInto(actor, original.tileObs, original.melds, original.river, original.meta);

  const sampled = game.resampleInformationSet(1001n);
  const resampled = createObservationBuffers();
  sampled.observeInto(actor, resampled.tileObs, resampled.melds, resampled.river, resampled.meta);

  assert.deepEqual(cloneBytes(resampled.tileObs), cloneBytes(original.tileObs));
  assert.deepEqual(cloneBytes(resampled.melds), cloneBytes(original.melds));
  assert.deepEqual(cloneBytes(resampled.river), cloneBytes(original.river));
  assert.deepEqual(cloneBytes(resampled.meta), cloneBytes(original.meta));
  assert.deepEqual(
    Array.from(sampled.legalActionMask, (v) => v.toString()),
    Array.from(game.legalActionMask, (v) => v.toString()),
  );

  const oracle = createOracleBuffer();
  const sampledOracle = createOracleBuffer();
  game.oracleTileCountsInto(oracle);
  sampled.oracleTileCountsInto(sampledOracle);
  // Histograms of known public tiles need not match plane-by-plane, but total
  // tile mass remains 108.
  assert.equal(sumHistogram(oracle), 108);
  assert.equal(sumHistogram(sampledOracle), 108);

  sampled.free();
  game.free();
});

test("finished games report rankings and no further actions", () => {
  const game = new wasm.Game(42n);
  const { steps } = playUntil(game, simpleRulePolicy);
  assert.ok(steps > 0);
  assert.equal(game.phase, PHASE_FINISHED());
  assert.equal(game.decision, undefined);
  assert.equal(game.simpleRuleAction(), undefined);
  assert.equal(game.ruleEvAction(), undefined);
  assert.deepEqual(Array.from(game.rankings()).sort(), [0, 1, 2, 3]);
  assert.equal(game.terminationReason != null, true);
  game.free();
});
