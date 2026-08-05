import { describe, expect, it } from "vitest";
import type { BotProfile } from "../../../engine/wasm/js/src/protocol";
import {
  createNextGameConfig,
  loadStoredGameConfig,
  storeGameConfig,
  type GameConfig,
} from "./config";

const BOT_PROFILES: readonly [BotProfile, BotProfile, BotProfile, BotProfile] =
  ["rule-fast", "rule-ev", "rule-nn", "rule-fast"];

describe("game config", () => {
  it("ignores a stored seed while restoring seat and bot profiles", () => {
    const config = loadStoredGameConfig(
      {
        getItem: () =>
          JSON.stringify({
            seed: "42",
            humanSeat: 2,
            botProfiles: BOT_PROFILES,
          }),
      },
      "1234",
    );

    expect(config).toEqual({
      seed: "1234",
      humanSeat: 2,
      botProfiles: BOT_PROFILES,
    });
  });

  it("keeps the seat and bot profiles when starting a new round", () => {
    const current: GameConfig = {
      seed: "42",
      humanSeat: 3,
      botProfiles: BOT_PROFILES,
    };
    const values = [42, 43];
    const next = createNextGameConfig(current, (target) => {
      target.set([0, values.shift()!]);
    });

    expect(next).toEqual({
      seed: "43",
      humanSeat: 3,
      botProfiles: BOT_PROFILES,
    });
  });

  it("stores only reusable player and bot settings", () => {
    let stored = "";
    storeGameConfig(
      { seed: "42", humanSeat: 1, botProfiles: BOT_PROFILES },
      {
        setItem: (_key, value) => {
          stored = value;
        },
      },
    );

    expect(JSON.parse(stored)).toEqual({
      humanSeat: 1,
      botProfiles: BOT_PROFILES,
    });
  });

  it("falls back when storage is unavailable", () => {
    expect(
      loadStoredGameConfig(
        {
          getItem: () => {
            throw new Error("unavailable");
          },
        },
        "1234",
      ),
    ).toEqual({
      seed: "1234",
      humanSeat: 0,
      botProfiles: ["rule-fast", "rule-fast", "rule-fast", "rule-fast"],
    });

    expect(() =>
      storeGameConfig(
        { seed: "42", humanSeat: 1, botProfiles: BOT_PROFILES },
        {
          setItem: () => {
            throw new Error("unavailable");
          },
        },
      ),
    ).not.toThrow();
  });
});
