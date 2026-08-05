import assert from "node:assert/strict";
import { before, test } from "node:test";
import {
  ENGINE_RULES_VERSION,
  PHASE_EXCHANGE,
  PHASE_FINISHED,
  PHASE_TURN,
  PLAYER_COUNT,
  TERMINATION_WALL_EXHAUSTED,
  TILE_KIND_COUNT,
} from "../../pkg/bloodflow_mahjong_wasm.js";
import {
  UI_SNAPSHOT_SCHEMA_VERSION,
  buildUiSnapshot,
  createObservationBuffers,
} from "../src/snapshot.ts";
import {
  loadWasm,
  playUntil,
  simpleRulePolicy,
  type WasmModule,
} from "./harness.ts";

let wasm: WasmModule;

before(async () => {
  wasm = await loadWasm();
});

test("buildUiSnapshot is viewer-relative and schema-complete", () => {
  const game = new wasm.Game(42n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 32,
    stopWhen: (g) => g.phase !== PHASE_EXCHANGE(),
  });

  const buffers = createObservationBuffers();
  for (const viewer of [0, 1, 2, 3]) {
    const snapshot = buildUiSnapshot(game, {
      viewer,
      engineRulesVersion: ENGINE_RULES_VERSION(),
      buffers,
    });

    assert.equal(snapshot.schemaVersion, UI_SNAPSHOT_SCHEMA_VERSION);
    assert.equal(snapshot.engineRulesVersion, ENGINE_RULES_VERSION());
    assert.equal(snapshot.ownHand.length, TILE_KIND_COUNT());
    assert.equal(snapshot.unlockedHand.length, TILE_KIND_COUNT());
    assert.equal(snapshot.winBase.length, TILE_KIND_COUNT());
    const directWinBase = new Uint8Array(TILE_KIND_COUNT());
    game.winBaseInto(viewer, directWinBase);
    assert.deepEqual(snapshot.winBase, directWinBase);
    assert.equal(snapshot.lockedTiles.length, PLAYER_COUNT());
    assert.equal(snapshot.scores.length, PLAYER_COUNT());
    assert.equal(snapshot.winCounts.length, PLAYER_COUNT());
    assert.equal(snapshot.lastWins.length, PLAYER_COUNT());
    assert.equal(snapshot.wallSettlement, null);
    assert.equal(snapshot.legalActionMask.length, 2);
    assert.equal(typeof snapshot.legalActionMask[0], "string");
    assert.doesNotThrow(() => JSON.stringify(snapshot));
    assert.ok(Array.isArray(snapshot.legalActionIds));
    assert.equal(snapshot.dealer, (game.dealer - viewer) & 3);

    if (snapshot.decisionActor != null && game.decision != null) {
      assert.equal(snapshot.decisionActor, (game.decision[0]! - viewer) & 3);
    }

    // Opponent concealed planes are not exposed as full hands.
    assert.equal(snapshot.handCounts.length, 4);
    for (let relative = 1; relative < 4; relative += 1) {
      assert.ok(snapshot.handCounts[relative]! >= 0);
    }
  }
  game.free();
});

test("terminal snapshot includes rankings and termination reason", () => {
  const game = new wasm.Game(42n);
  playUntil(game, simpleRulePolicy);
  const snapshot = buildUiSnapshot(game, {
    viewer: 2,
    engineRulesVersion: ENGINE_RULES_VERSION(),
  });
  assert.equal(snapshot.phase, PHASE_FINISHED());
  assert.ok(snapshot.rankings != null);
  assert.equal(snapshot.rankings!.length, 4);
  assert.deepEqual([...snapshot.rankings!].sort(), [0, 1, 2, 3]);
  assert.equal(snapshot.terminationReason, game.terminationReason ?? null);
  assert.equal(snapshot.decisionActor, null);
  assert.equal(
    snapshot.wallSettlement != null,
    snapshot.terminationReason === TERMINATION_WALL_EXHAUSTED(),
  );
  if (snapshot.wallSettlement != null) {
    assert.equal(snapshot.wallSettlement.hands.length, PLAYER_COUNT());
    assert.equal(snapshot.wallSettlement.flowerPig.length, PLAYER_COUNT());
    assert.equal(snapshot.wallSettlement.ready.length, PLAYER_COUNT());
    assert.equal(
      snapshot.wallSettlement.maxShapeMultipliers.length,
      PLAYER_COUNT(),
    );
    for (const hand of snapshot.wallSettlement.hands) {
      assert.equal(hand.length, TILE_KIND_COUNT());
    }
  }
  game.free();
});

test("exchange selection count and suit track the staged tiles", () => {
  const game = new wasm.Game(2024n);
  const buffers = createObservationBuffers();
  assert.equal(game.phase, PHASE_EXCHANGE());

  const sum = (plane: Uint8Array): number => plane.reduce((a, b) => a + b, 0);
  const observedCounts = new Set<number>();
  let steps = 0;

  while (game.phase === PHASE_EXCHANGE() && steps < 64) {
    const actor = game.decision![0]!;
    const before = buildUiSnapshot(game, {
      viewer: actor,
      engineRulesVersion: ENGINE_RULES_VERSION(),
      buffers,
    });
    const picks = before.legalActionIds.filter((id) => id < TILE_KIND_COUNT());
    assert.ok(picks.length > 0, "exchange phase must offer tile picks");

    // meta[10]/meta[11] must agree with the selection plane itself.
    assert.equal(before.exchangeSelectedCount, sum(before.exchangeSelection));
    if (before.exchangeSelectedCount === 0) {
      assert.equal(before.exchangeSelectionSuit, -1);
    } else {
      assert.ok(before.exchangeSelectionSuit >= 0);
      for (const pick of picks) {
        assert.equal(
          Math.floor(pick / 9),
          before.exchangeSelectionSuit,
          "picks must stay inside the staged suit",
        );
      }
      for (let tile = 0; tile < TILE_KIND_COUNT(); tile += 1) {
        if ((before.exchangeSelection[tile] ?? 0) > 0) {
          assert.equal(Math.floor(tile / 9), before.exchangeSelectionSuit);
        }
      }
    }

    game.stepId(picks[0]!);
    steps += 1;
    observedCounts.add(
      sum(
        buildUiSnapshot(game, {
          viewer: actor,
          engineRulesVersion: ENGINE_RULES_VERSION(),
          buffers,
        }).exchangeSelection,
      ),
    );
  }

  assert.ok(
    observedCounts.has(1) && observedCounts.has(2),
    `expected partial selections, saw ${[...observedCounts].join(",")}`,
  );
  game.free();
});

test("snapshot draw tile stays hidden for non-drawer viewers", () => {
  const game = new wasm.Game(127n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 256,
    stopWhen: (g) => g.currentDraw != null,
  });
  assert.ok(game.currentDraw != null);
  const drawer = game.currentDraw![0]!;
  const tile = game.currentDraw![1]!;
  const other = (drawer + 1) % 4;

  const drawerSnap = buildUiSnapshot(game, {
    viewer: drawer,
    engineRulesVersion: ENGINE_RULES_VERSION(),
  });
  const otherSnap = buildUiSnapshot(game, {
    viewer: other,
    engineRulesVersion: ENGINE_RULES_VERSION(),
  });
  assert.equal(drawerSnap.drawTile, tile);
  assert.equal(otherSnap.drawTile, -1);
  game.free();
});

test("snapshot exposes legal actions only to the decision actor", () => {
  const game = new wasm.Game(127n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 64,
    stopWhen: (current) => current.phase === PHASE_TURN(),
  });
  assert.equal(game.phase, PHASE_TURN());
  const actor = game.decision![0]!;
  const observer = (actor + 1) % PLAYER_COUNT();

  const actorSnapshot = buildUiSnapshot(game, {
    viewer: actor,
    engineRulesVersion: ENGINE_RULES_VERSION(),
  });
  const observerSnapshot = buildUiSnapshot(game, {
    viewer: observer,
    engineRulesVersion: ENGINE_RULES_VERSION(),
  });

  assert.equal(actorSnapshot.decisionActor, 0);
  assert.ok(actorSnapshot.legalActionIds.length > 0);
  assert.notEqual(observerSnapshot.decisionActor, 0);
  assert.deepEqual(observerSnapshot.legalActionMask, ["0", "0"]);
  assert.deepEqual(observerSnapshot.legalActionIds, []);
  game.free();
});

test("snapshot separates the viewer win base from winning references", () => {
  const game = new wasm.Game(42n);
  playUntil(game, simpleRulePolicy, {
    maxSteps: 10_000,
    stopWhen: (g) => [0, 1, 2, 3].some((seat) => g.hasWon(seat)),
  });
  const winner = [0, 1, 2, 3].find((seat) => game.hasWon(seat));
  assert.notEqual(winner, undefined);

  const snapshot = buildUiSnapshot(game, {
    viewer: winner!,
    engineRulesVersion: ENGINE_RULES_VERSION(),
  });
  const sum = (histogram: Uint8Array): number =>
    histogram.reduce((total, count) => total + count, 0);
  const baseCount = sum(snapshot.winBase);
  const lockedCount = sum(snapshot.lockedTiles[0]);

  assert.ok(baseCount > 0, "a completed win must establish a stable base");
  assert.ok(
    lockedCount > baseCount,
    "at least one winning reference must sit outside the stable base",
  );
  for (let tile = 0; tile < TILE_KIND_COUNT(); tile += 1) {
    assert.ok(snapshot.winBase[tile]! <= snapshot.lockedTiles[0][tile]!);
  }
  game.free();
});
