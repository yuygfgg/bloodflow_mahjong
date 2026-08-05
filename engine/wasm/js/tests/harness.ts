/**
 * Shared WASM test harness.
 *
 * Loads the WASM package once, exposes typed helpers, and keeps tests
 * free of ad-hoc path and buffer boilerplate.
 */
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  ACTION_SPACE_SIZE,
  EVENT_HISTORY_CAPACITY,
  EVENT_RECORD_WIDTH,
  META_OBSERVATION_WIDTH,
  MELD_OBSERVATION_WIDTH,
  ORACLE_TILE_COUNT_PLANES,
  PHASE_FINISHED,
  RIVER_OBSERVATION_WIDTH,
  STEP_RECORD_WIDTH,
  TILE_KIND_COUNT,
  TILE_OBSERVATION_WIDTH,
} from "../../pkg/bloodflow_mahjong_wasm.js";
import { isLegalAction, readMaskWords } from "../src/legal.ts";

const here = dirname(fileURLToPath(import.meta.url));
export const PKG_DIR = resolve(here, "../../pkg");
export const REPO_ROOT = resolve(here, "../../../..");
export const MODEL_PATH = resolve(REPO_ROOT, "model/latest.onnx");

/** Minimal surface used by the harness and tests. */
export interface WasmGame {
  free(): void;
  phase: number;
  decision: Uint8Array | undefined;
  legalActionMask: BigUint64Array;
  dealer: number;
  exchangeDirection: number;
  wallRemaining: number;
  eventCount: number;
  eventDropped: bigint;
  terminationReason: number | undefined;
  currentDraw: Uint8Array | undefined;
  reset(seed: bigint): void;
  stepId(action: number): Int32Array;
  stepInto(action: number, output: Int32Array): void;
  simpleRuleAction(): number | undefined;
  ruleEvAction(): number | undefined;
  ruleEvActionWithConfig(config: WasmRuleEvConfig): number | undefined;
  rulePlannerAction(): number | undefined;
  rulePlannerActionWithConfig(config: WasmRulePlannerConfig): number | undefined;
  scores(): Int32Array;
  missingSuits(): Int8Array;
  rankings(): Uint8Array;
  maxWinMultipliers(): Uint32Array;
  hasWon(seat: number): boolean;
  handAnalysis(seat: number): Int32Array;
  concealedInto(seat: number, output: Uint8Array): void;
  lockedInto(seat: number, output: Uint8Array): void;
  winBaseInto(seat: number, output: Uint8Array): void;
  exchangeSelectionInto(seat: number, output: Uint8Array): void;
  melds(seat: number): Uint8Array;
  discards(): Uint8Array;
  observeInto(
    viewer: number,
    tileObs: Uint8Array,
    melds: Uint8Array,
    river: Uint8Array,
    meta: Int32Array,
  ): void;
  eventsInto(viewer: number, output: Int32Array): number;
  stepEventsInto(viewer: number, output: Int32Array): number;
  playerUiStatsInto(viewer: number, output: Int32Array): void;
  wallSettlementInto(viewer: number, meta: Int32Array, hands: Uint8Array): boolean;
  oracleTileCountsInto(output: Uint8Array): void;
  resampleInformationSet(seed: bigint): WasmGame;
}

export interface WasmRuleEvConfig {
  free(): void;
  readonly searchDepth: number;
  readonly defense: boolean;
}

export interface WasmRulePlannerConfig {
  free(): void;
  readonly handChanges: number;
  readonly drawHorizon: number;
  readonly candidateStates: number;
  readonly beliefWorlds: number;
  readonly responseWorlds: number;
  readonly searchIterations: number;
}

export interface WasmRuleNn {
  free(): void;
  action(game: WasmGame): number | undefined;
}

export interface WasmModule {
  default(input?: { module_or_path: BufferSource | string | URL }): Promise<unknown>;
  initPanicHook(): void;
  Game: new (seed: bigint) => WasmGame;
  RuleEvConfig: {
    new (searchDepth: number, defense: boolean): WasmRuleEvConfig;
    fast(): WasmRuleEvConfig;
    standard(): WasmRuleEvConfig;
  };
  RulePlannerConfig: {
    new (
      handChanges: number,
      drawHorizon: number,
      candidateStates: number,
      beliefWorlds: number,
      responseWorlds: number,
      searchIterations: number,
    ): WasmRulePlannerConfig;
    defaultConfig(): WasmRulePlannerConfig;
  };
  RuleNn: new (onnx: Uint8Array) => WasmRuleNn;
  [name: string]: unknown;
}

export interface ObservationBuffers {
  tileObs: Uint8Array;
  winBase: Uint8Array;
  melds: Uint8Array;
  river: Uint8Array;
  meta: Int32Array;
}

export interface EventBuffers {
  history: Int32Array;
  step: Int32Array;
}

export interface PlayResult {
  actions: number[];
  steps: number;
}

export type ActionPolicy = (game: WasmGame) => number | undefined;

let wasmLoad: Promise<WasmModule> | undefined;

/** Load and initialize the WASM package once per process. */
export async function loadWasm(): Promise<WasmModule> {
  if (wasmLoad == null) {
    wasmLoad = (async () => {
      const jsPath = resolve(PKG_DIR, "bloodflow_mahjong_wasm.js");
      const wasmPath = resolve(PKG_DIR, "bloodflow_mahjong_wasm_bg.wasm");
      const mod = (await import(pathToFileURL(jsPath).href)) as WasmModule;
      const bytes = await readFile(wasmPath);
      await mod.default({ module_or_path: bytes });
      mod.initPanicHook();
      return mod;
    })();
  }
  return wasmLoad;
}

/** Read a numeric WASM export `NAME()` and fail if missing. */
export function wasmConst(wasm: WasmModule, name: string): number {
  const fn = wasm[name];
  if (typeof fn !== "function") {
    throw new Error(`missing WASM constant export: ${name}`);
  }
  return (fn as () => number)();
}

export function createObservationBuffers(): ObservationBuffers {
  return {
    tileObs: new Uint8Array(TILE_OBSERVATION_WIDTH()),
    winBase: new Uint8Array(TILE_KIND_COUNT()),
    melds: new Uint8Array(MELD_OBSERVATION_WIDTH()),
    river: new Uint8Array(RIVER_OBSERVATION_WIDTH()),
    meta: new Int32Array(META_OBSERVATION_WIDTH()),
  };
}

export function createEventBuffers(capacity = EVENT_HISTORY_CAPACITY()): EventBuffers {
  return {
    history: new Int32Array(capacity * EVENT_RECORD_WIDTH()),
    step: new Int32Array(capacity * EVENT_RECORD_WIDTH()),
  };
}

export function createStepRecord(): Int32Array {
  return new Int32Array(STEP_RECORD_WIDTH());
}

export function createTileHistogram(): Uint8Array {
  return new Uint8Array(TILE_KIND_COUNT());
}

export function createOracleBuffer(): Uint8Array {
  return new Uint8Array(ORACLE_TILE_COUNT_PLANES() * TILE_KIND_COUNT());
}

/** Lowest set action id in the legal mask. */
export function firstLegalAction(game: WasmGame): number {
  const [low, high] = readMaskWords(game.legalActionMask);
  for (let action = 0; action < ACTION_SPACE_SIZE(); action += 1) {
    if (isLegalAction(action, low, high)) {
      return action;
    }
  }
  throw new Error("non-terminal environment has no legal action");
}

export function assertActionLegal(game: WasmGame, action: number): void {
  const [low, high] = readMaskWords(game.legalActionMask);
  if (!isLegalAction(action, low, high)) {
    throw new Error(`action ${action} is not legal under mask [${low}, ${high}]`);
  }
}

/** Compact public state used to prove illegal actions do not mutate. */
export function publicFingerprint(game: WasmGame): string {
  const decision = game.decision == null ? "none" : Array.from(game.decision).join(":");
  const mask = Array.from(game.legalActionMask, (word) => word.toString()).join(":");
  const scores = Array.from(game.scores()).join(",");
  return [
    game.phase,
    decision,
    mask,
    scores,
    game.eventCount,
    game.eventDropped.toString(),
    game.wallRemaining,
    game.dealer,
    game.exchangeDirection,
    game.terminationReason ?? -1,
  ].join("|");
}

export function playUntil(
  game: WasmGame,
  policy: ActionPolicy,
  options: { maxSteps?: number; stopWhen?: (game: WasmGame) => boolean } = {},
): PlayResult {
  const maxSteps = options.maxSteps ?? 10_000;
  const stopWhen = options.stopWhen ?? ((g) => g.phase === PHASE_FINISHED());
  const actions: number[] = [];
  let steps = 0;

  while (!stopWhen(game)) {
    const action = policy(game);
    if (action == null) {
      throw new Error(`policy returned no action at step ${steps}, phase=${game.phase}`);
    }
    assertActionLegal(game, action);
    game.stepId(action);
    actions.push(action);
    steps += 1;
    if (steps > maxSteps) {
      throw new Error(`playUntil exceeded maxSteps=${maxSteps}`);
    }
  }

  return { actions, steps };
}

export function simpleRulePolicy(game: WasmGame): number | undefined {
  return game.simpleRuleAction();
}

export function replayActions(wasm: WasmModule, seed: bigint, actions: readonly number[]): WasmGame {
  const game = new wasm.Game(seed);
  for (const action of actions) {
    game.stepId(action);
  }
  return game;
}

export function sumHistogram(values: Uint8Array): number {
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total;
}

export function cloneBytes(values: Uint8Array | Int32Array | Int8Array | Uint32Array): number[] {
  return Array.from(values);
}

/** Reject helper that works with both Error and WASM string throws. */
export function errorMessage(error: unknown): string {
  if (typeof error === "string") {
    return error;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export async function assertRejects(
  body: () => unknown | Promise<unknown>,
  match?: RegExp | string,
): Promise<void> {
  let threw = false;
  try {
    await body();
  } catch (error) {
    threw = true;
    if (match != null) {
      const message = errorMessage(error);
      const ok = typeof match === "string" ? message.includes(match) : match.test(message);
      if (!ok) {
        throw new Error(`rejected with unexpected message: ${message}`);
      }
    }
  }
  if (!threw) {
    throw new Error("expected function to reject");
  }
}
