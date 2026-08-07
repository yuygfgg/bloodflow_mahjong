import { describe, expect, it } from "vitest";
import type { BotProfile } from "../../../engine/wasm/js/src/protocol";
import {
  BOT_PROFILES,
  areConfiguredBotsAvailable,
  createNextGameConfig,
  isBotProfileAvailable,
  loadStoredGameConfig,
  storeGameConfig,
  type GameConfig,
} from "./config";

const TEST_BOT_PROFILES: readonly [
  BotProfile,
  BotProfile,
  BotProfile,
  BotProfile,
] = ["rule-fast", "rule-ev", "rule-fast", "rule-ev"];

describe("game config", () => {
  it("ignores a stored seed while restoring seat and bot profiles", () => {
    const config = loadStoredGameConfig(
      {
        getItem: () =>
          JSON.stringify({
            seed: "42",
            humanSeat: 2,
            botProfiles: TEST_BOT_PROFILES,
          }),
      },
      "1234",
    );

    expect(config).toEqual({
      seed: "1234",
      humanSeat: 2,
      botProfiles: TEST_BOT_PROFILES,
    });
  });

  it("keeps the seat and bot profiles when starting a new round", () => {
    const current: GameConfig = {
      seed: "42",
      humanSeat: 3,
      botProfiles: TEST_BOT_PROFILES,
    };
    const values = [42, 43];
    const next = createNextGameConfig(current, (target) => {
      target.set([0, values.shift()!]);
    });

    expect(next).toEqual({
      seed: "43",
      humanSeat: 3,
      botProfiles: TEST_BOT_PROFILES,
    });
  });

  it("stores only reusable player and bot settings", () => {
    let stored = "";
    storeGameConfig(
      { seed: "42", humanSeat: 1, botProfiles: TEST_BOT_PROFILES },
      {
        setItem: (_key, value) => {
          stored = value;
        },
      },
    );

    expect(JSON.parse(stored)).toEqual({
      humanSeat: 1,
      botProfiles: TEST_BOT_PROFILES,
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
        { seed: "42", humanSeat: 1, botProfiles: TEST_BOT_PROFILES },
        {
          setItem: () => {
            throw new Error("unavailable");
          },
        },
      ),
    ).not.toThrow();
  });

  it("restores a neural profile before runtime capability detection", () => {
    expect(
      loadStoredGameConfig(
        {
          getItem: () =>
            JSON.stringify({
              humanSeat: 0,
              botProfiles: ["rule-nn", "rule-fast", "rule-fast", "rule-fast"],
            }),
        },
        "1234",
      ),
    ).toEqual({
      seed: "1234",
      humanSeat: 0,
      botProfiles: ["rule-nn", "rule-fast", "rule-fast", "rule-fast"],
    });
  });

  it("keeps every supported profile in the persistent config domain", () => {
    expect(BOT_PROFILES).toEqual(["rule-fast", "rule-ev", "rule-nn"]);
    expect(isBotProfileAvailable("rule-fast", false)).toBe(true);
    expect(isBotProfileAvailable("rule-ev", false)).toBe(true);
    expect(isBotProfileAvailable("rule-nn", false)).toBe(false);
    expect(isBotProfileAvailable("rule-nn", true)).toBe(true);
  });

  it("requires a validated model only for seats controlled by rule-nn", () => {
    const config: GameConfig = {
      seed: "42",
      humanSeat: 0,
      botProfiles: ["rule-nn", "rule-ev", "rule-nn", "rule-fast"],
    };

    expect(areConfiguredBotsAvailable(config, false)).toBe(false);
    expect(areConfiguredBotsAvailable(config, true)).toBe(true);

    expect(
      areConfiguredBotsAvailable(
        {
          ...config,
          botProfiles: ["rule-nn", "rule-ev", "rule-fast", "rule-fast"],
        },
        false,
      ),
    ).toBe(true);
  });
});
