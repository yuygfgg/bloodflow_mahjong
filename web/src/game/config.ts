import type { BotProfile } from "../../../engine/wasm/js/src/protocol";
import { createRandomSeed, type RandomFill } from "./seed";

const STORAGE_KEY = "bloodflow-game-config";
const SESSION_DEFAULT_SEED = createRandomSeed();
const DEFAULT_BOT_PROFILES = [
  "rule-fast",
  "rule-fast",
  "rule-fast",
  "rule-fast",
] as const;
export const BOT_PROFILES = ["rule-fast", "rule-ev", "rule-nn"] as const;

export interface GameConfig {
  seed: string;
  humanSeat: number;
  botProfiles: readonly [BotProfile, BotProfile, BotProfile, BotProfile];
}

interface ConfigReader {
  getItem(key: string): string | null;
}

interface ConfigWriter {
  setItem(key: string, value: string): void;
}

function defaultConfig(seed: string): GameConfig {
  return {
    seed,
    humanSeat: 0,
    botProfiles: DEFAULT_BOT_PROFILES,
  };
}

function isBotProfile(value: unknown): value is BotProfile {
  return BOT_PROFILES.some((profile) => profile === value);
}

export function isBotProfileAvailable(
  profile: BotProfile,
  nnAvailable: boolean,
): boolean {
  return profile !== "rule-nn" || nnAvailable;
}

export function areConfiguredBotsAvailable(
  config: GameConfig,
  nnAvailable: boolean,
): boolean {
  return config.botProfiles.every(
    (profile, seat) =>
      seat === config.humanSeat || isBotProfileAvailable(profile, nnAvailable),
  );
}

export function loadStoredGameConfig(
  storage: ConfigReader | undefined = undefined,
  defaultSeed = SESSION_DEFAULT_SEED,
): GameConfig {
  const fallback = defaultConfig(defaultSeed);
  try {
    const stored = (storage ?? globalThis.localStorage)?.getItem(STORAGE_KEY);
    if (stored == null) return fallback;
    const parsed = JSON.parse(stored) as Partial<GameConfig>;
    const { humanSeat, botProfiles } = parsed;
    if (
      typeof humanSeat !== "number" ||
      !Number.isInteger(humanSeat) ||
      humanSeat < 0 ||
      humanSeat > 3 ||
      !Array.isArray(botProfiles) ||
      botProfiles.length !== 4 ||
      botProfiles.some((profile) => !isBotProfile(profile))
    ) {
      return fallback;
    }
    return {
      seed: defaultSeed,
      humanSeat,
      botProfiles: botProfiles as GameConfig["botProfiles"],
    };
  } catch {
    return fallback;
  }
}

export function storeGameConfig(
  config: GameConfig,
  storage: ConfigWriter | undefined = undefined,
): void {
  try {
    (storage ?? globalThis.localStorage)?.setItem(
      STORAGE_KEY,
      JSON.stringify({
        humanSeat: config.humanSeat,
        botProfiles: config.botProfiles,
      }),
    );
  } catch {
    // Persistent storage is optional. The current session keeps the config.
  }
}

export function createNextGameConfig(
  current: GameConfig,
  fill?: RandomFill,
): GameConfig {
  return {
    ...current,
    seed: createRandomSeed(current.seed, fill),
  };
}
