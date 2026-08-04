/**
 * Worker message protocol between the main thread and the engine worker.
 *
 * The worker owns the WASM Game instance. The main thread never mutates rules
 * state directly.
 */

import type { UiSnapshot } from "./types.ts";

export type BotProfile = "rule-fast" | "rule-ev" | "rule-nn";

export type HintPolicy = BotProfile;

export type WorkerRequest =
  | {
      type: "new_game";
      requestId: string;
      seed: number | string;
      humanSeat: number;
      botProfiles: readonly [BotProfile, BotProfile, BotProfile, BotProfile];
    }
  | {
      type: "submit";
      requestId: string;
      actionId: number;
    }
  | {
      type: "request_hint";
      requestId: string;
      policy: HintPolicy;
    }
  | {
      type: "pause";
      requestId: string;
    }
  | {
      type: "resume";
      requestId: string;
    }
  | {
      type: "export_replay";
      requestId: string;
    }
  | {
      type: "load_replay";
      requestId: string;
      engineRulesVersion: number;
      seed: number | string;
      actions: readonly number[];
      humanSeat: number;
    };

export type AnimationHint =
  | { kind: "draw"; seatRelative: number; tile: number; replacement: boolean }
  | { kind: "discard"; seatRelative: number; tile: number }
  | { kind: "meld"; seatRelative: number; tile: number; meldKind: number }
  | { kind: "hu"; seatRelative: number; tile: number; multiplier: number }
  | { kind: "payment"; payerRelative: number; payeeRelative: number; amount: number }
  | { kind: "exchange_complete"; direction: number }
  | { kind: "missing_revealed" }
  | { kind: "game_end" };

export type WorkerResponse =
  | {
      type: "ready";
      requestId: string;
      engineRulesVersion: number;
    }
  | {
      type: "snapshot";
      requestId: string;
      snapshot: UiSnapshot;
    }
  | {
      type: "transition";
      requestId: string;
      actionId: number;
      snapshot: UiSnapshot;
      stepEvents: UiSnapshot["stepEvents"];
      animationHints: readonly AnimationHint[];
    }
  | {
      type: "hint";
      requestId: string;
      actionId: number | null;
      policy: HintPolicy;
    }
  | {
      type: "replay";
      requestId: string;
      engineRulesVersion: number;
      seed: string;
      actions: readonly number[];
    }
  | {
      type: "error";
      requestId: string;
      message: string;
    };
