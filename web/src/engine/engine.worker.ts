/// <reference lib="webworker" />

import modelUrl from "../../../model/latest.onnx?url";
import wasmUrl from "../../../engine/wasm/pkg/bloodflow_mahjong_wasm_bg.wasm?url";
import init, {
  ENGINE_RULES_VERSION,
  EVENT_KIND_DISCARD,
  EVENT_KIND_DRAW,
  EVENT_KIND_EXCHANGE_COMPLETE,
  EVENT_KIND_GAME_END,
  EVENT_KIND_HU,
  EVENT_KIND_MELD,
  EVENT_KIND_MISSING_REVEALED,
  EVENT_KIND_PAYMENT,
  EVENT_KIND_SETTLEMENT_STAGE,
  Game,
  PHASE_FINISHED,
  RuleNn,
  initPanicHook,
} from "../../../engine/wasm/pkg/bloodflow_mahjong_wasm.js";
import {
  REPLAY_PROTOCOL_VERSION,
  type AnimationHint,
  type BotProfile,
  type ReplayRecord,
  type WorkerRequest,
  type WorkerResponse,
} from "../../../engine/wasm/js/src/protocol";
import {
  buildUiSnapshot,
  createObservationBuffers,
} from "../../../engine/wasm/js/src/snapshot";
import type {
  EventRecord,
  ObservationBuffers,
  UiSnapshot,
} from "../../../engine/wasm/js/src/types";
import { scoreChangingSettlementStageIndexes } from "../game/settlement";

const scope = self as DedicatedWorkerGlobalScope;
const MAX_SEED = (1n << 64n) - 1n;
const BOT_ACTION_DELAY_MS = 2_000;

let game: Game | undefined;
let nn: RuleNn | undefined;
let nnLoad: Promise<RuleNn> | undefined;
let humanSeat = 0;
let seed = "";
let botProfiles: [BotProfile, BotProfile, BotProfile, BotProfile] = [
  "rule-fast",
  "rule-fast",
  "rule-fast",
  "rule-fast",
];
let actions: number[] = [];
let paused = false;
let buffers: ObservationBuffers | undefined;

function post(message: WorkerResponse): void {
  scope.postMessage(message);
}

function parseSeed(value: number | string): bigint {
  const text = String(value).trim();
  if (!/^[0-9]+$/.test(text)) {
    throw new Error("Seed must be an unsigned decimal integer.");
  }
  const parsed = BigInt(text);
  if (parsed > MAX_SEED) {
    throw new Error("Seed must fit in an unsigned 64-bit integer.");
  }
  return parsed;
}

function requireGame(): Game {
  if (game == null) {
    throw new Error("Start a game before sending this request.");
  }
  return game;
}

function snapshot(includeEventHistory = true): UiSnapshot {
  return buildUiSnapshot(requireGame(), {
    viewer: humanSeat,
    engineRulesVersion: ENGINE_RULES_VERSION(),
    buffers,
    includeEventHistory,
  });
}

function actionFor(profile: BotProfile): number | undefined {
  const current = requireGame();
  switch (profile) {
    case "rule-fast":
      return current.simpleRuleAction();
    case "rule-ev":
      return current.ruleEvAction();
    case "rule-nn":
      throw new Error("rule-nn is not loaded.");
  }
}

/**
 * Load the optional NN policy only when a game actually needs it. Model
 * construction performs synchronous WASM work, so doing it during worker boot
 * prevents the worker from reporting that the rules engine is ready.
 */
async function loadNn(): Promise<RuleNn> {
  if (nn != null) return nn;
  if (nnLoad != null) return nnLoad;

  nnLoad = (async () => {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`NN model request failed with HTTP ${response.status}.`);
    }
    const loaded = new RuleNn(new Uint8Array(await response.arrayBuffer()));
    nn = loaded;
    return loaded;
  })().catch((error: unknown) => {
    nnLoad = undefined;
    throw error;
  });
  return nnLoad;
}

async function chooseAction(profile: BotProfile): Promise<number | undefined> {
  const current = requireGame();
  if (profile === "rule-nn") {
    return (await loadNn()).action(current);
  }
  return actionFor(profile);
}

function hintsFromEvents(events: readonly EventRecord[]): AnimationHint[] {
  const hints: AnimationHint[] = [];
  const visibleSettlementStages = scoreChangingSettlementStageIndexes(events);
  for (const [index, event] of events.entries()) {
    const [kind, actor, target, tile, flags, value] = event;
    if (kind === EVENT_KIND_DRAW()) {
      hints.push({
        kind: "draw",
        seatRelative: actor,
        tile,
        replacement: (flags & 1) !== 0,
      });
    } else if (kind === EVENT_KIND_DISCARD()) {
      hints.push({ kind: "discard", seatRelative: actor, tile });
    } else if (kind === EVENT_KIND_MELD()) {
      hints.push({ kind: "meld", seatRelative: actor, tile, meldKind: flags });
    } else if (kind === EVENT_KIND_HU()) {
      hints.push({
        kind: "hu",
        seatRelative: actor,
        sourceRelative: target,
        tile,
        multiplier: value,
      });
    } else if (kind === EVENT_KIND_PAYMENT()) {
      hints.push({
        kind: "payment",
        payerRelative: actor,
        payeeRelative: target,
        amount: value,
      });
    } else if (kind === EVENT_KIND_SETTLEMENT_STAGE()) {
      if (visibleSettlementStages.has(index)) {
        hints.push({ kind: "settlement_stage", stage: flags });
      }
    } else if (kind === EVENT_KIND_EXCHANGE_COMPLETE()) {
      hints.push({ kind: "exchange_complete", direction: flags });
    } else if (kind === EVENT_KIND_MISSING_REVEALED()) {
      if (!hints.some((hint) => hint.kind === "missing_revealed")) {
        hints.push({ kind: "missing_revealed" });
      }
    } else if (kind === EVENT_KIND_GAME_END()) {
      hints.push({ kind: "game_end" });
    }
  }
  return hints;
}

function step(actionId: number): EventRecord[] {
  const current = requireGame();
  current.stepId(actionId);
  actions.push(actionId);
  return [...snapshot(false).stepEvents];
}

async function yieldToMessages(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function waitForBotTurn(): Promise<void> {
  await new Promise<void>((resolve) =>
    setTimeout(resolve, BOT_ACTION_DELAY_MS),
  );
}

async function advanceBots(
  events: EventRecord[],
  progressRequestId?: string,
): Promise<number> {
  const current = requireGame();
  let steps = 0;
  while (!paused && current.phase !== PHASE_FINISHED()) {
    const decision = current.decision;
    if (decision == null || decision[0] === humanSeat) {
      return steps;
    }
    const actor = decision[0]!;
    if (progressRequestId != null) await waitForBotTurn();
    const action = await chooseAction(botProfiles[actor]!);
    if (action == null) {
      throw new Error(
        `${botProfiles[actor]} did not return an action for seat ${actor}.`,
      );
    }
    const stepEvents = step(action);
    events.push(...stepEvents);
    if (progressRequestId != null) {
      post({
        type: "progress",
        requestId: progressRequestId,
        snapshot: snapshot(),
        animationHints: hintsFromEvents(stepEvents),
      });
    }
    steps += 1;
    if (steps > 10_000) {
      throw new Error("Bot advancement exceeded the game step limit.");
    }
    await yieldToMessages();
  }
  return steps;
}

async function startGame(
  request: Extract<WorkerRequest, { type: "new_game" }>,
): Promise<void> {
  const parsedSeed = parseSeed(request.seed);
  if (request.humanSeat < 0 || request.humanSeat > 3) {
    throw new Error("Human seat must be in 0..3.");
  }
  game?.free();
  game = new Game(parsedSeed);
  humanSeat = request.humanSeat;
  seed = parsedSeed.toString();
  botProfiles = [...request.botProfiles];
  actions = [];
  paused = false;
  buffers = createObservationBuffers();

  const events: EventRecord[] = [];
  await advanceBots(events);
  post({
    type: "snapshot",
    requestId: request.requestId,
    snapshot: snapshot(),
  });
}

async function submit(
  request: Extract<WorkerRequest, { type: "submit" }>,
): Promise<void> {
  const before = requireGame();
  if (paused) {
    throw new Error("Resume the game before submitting an action.");
  }
  if (before.decision?.[0] !== humanSeat) {
    throw new Error("The current decision does not belong to the human seat.");
  }
  const events = step(request.actionId);
  const afterHuman = requireGame().decision;
  const hasBotFollowUp = afterHuman != null && afterHuman[0] !== humanSeat;
  if (hasBotFollowUp) {
    post({
      type: "progress",
      requestId: request.requestId,
      snapshot: snapshot(),
      animationHints: hintsFromEvents(events),
    });
  }
  await advanceBots(events, hasBotFollowUp ? request.requestId : undefined);
  const next = snapshot();
  post({
    type: "transition",
    requestId: request.requestId,
    actionId: request.actionId,
    snapshot: { ...next, stepEvents: events },
    stepEvents: events,
    animationHints: hasBotFollowUp ? [] : hintsFromEvents(events),
  });
}

function replayRecord(): ReplayRecord {
  return {
    protocolVersion: REPLAY_PROTOCOL_VERSION,
    engineRulesVersion: ENGINE_RULES_VERSION(),
    seed,
    humanSeat,
    botProfiles,
    actions,
  };
}

async function loadReplay(
  request: Extract<WorkerRequest, { type: "load_replay" }>,
): Promise<void> {
  const replay = request.replay;
  if (replay.protocolVersion !== REPLAY_PROTOCOL_VERSION) {
    throw new Error(`Unsupported replay protocol ${replay.protocolVersion}.`);
  }
  if (replay.engineRulesVersion !== ENGINE_RULES_VERSION()) {
    throw new Error(
      `Replay uses engine rules ${replay.engineRulesVersion}; this build uses ${ENGINE_RULES_VERSION()}.`,
    );
  }
  const parsedSeed = parseSeed(replay.seed);
  if (replay.humanSeat < 0 || replay.humanSeat > 3) {
    throw new Error("Replay human seat must be in 0..3.");
  }

  const loaded = new Game(parsedSeed);
  for (const action of replay.actions) {
    loaded.stepId(action);
  }
  game?.free();
  game = loaded;
  humanSeat = replay.humanSeat;
  seed = parsedSeed.toString();
  botProfiles = [...replay.botProfiles];
  actions = [...replay.actions];
  paused = false;
  buffers = createObservationBuffers();
  post({
    type: "snapshot",
    requestId: request.requestId,
    snapshot: snapshot(),
  });
}

async function handle(request: WorkerRequest): Promise<void> {
  switch (request.type) {
    case "new_game":
      await startGame(request);
      return;
    case "submit":
      await submit(request);
      return;
    case "request_hint": {
      const actionId = (await chooseAction(request.policy)) ?? null;
      post({
        type: "hint",
        requestId: request.requestId,
        actionId,
        policy: request.policy,
      });
      return;
    }
    case "pause":
      paused = true;
      post({
        type: "snapshot",
        requestId: request.requestId,
        snapshot: snapshot(),
      });
      return;
    case "resume": {
      paused = false;
      const events: EventRecord[] = [];
      await advanceBots(events, request.requestId);
      post({
        type: "snapshot",
        requestId: request.requestId,
        snapshot: snapshot(),
      });
      return;
    }
    case "export_replay":
      post({
        type: "replay",
        requestId: request.requestId,
        replay: replayRecord(),
      });
      return;
    case "load_replay":
      await loadReplay(request);
  }
}

scope.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  void handle(request).catch((error: unknown) => {
    post({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  });
});

void (async () => {
  try {
    await init({ module_or_path: wasmUrl });
    initPanicHook();
    post({
      type: "ready",
      requestId: "boot",
      engineRulesVersion: ENGINE_RULES_VERSION(),
    });
  } catch (error) {
    post({
      type: "error",
      requestId: "boot",
      message: error instanceof Error ? error.message : String(error),
    });
  }
})();
