import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  BotProfile,
  ReplayRecord,
} from "../../../engine/wasm/js/src/protocol";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";
import { EngineClient, type TransitionResult } from "../engine/EngineClient";
import {
  createNextGameConfig,
  loadStoredGameConfig,
  storeGameConfig,
  type GameConfig,
} from "./config";

export type { GameConfig } from "./config";

export type GameStatus = "booting" | "ready" | "playing" | "error";
export type EngineActivity = "starting" | "turn" | "hint" | "pause" | "replay";

export interface AnimationBatch {
  id: number;
  hints: TransitionResult["animationHints"];
}

export interface UseGameEngineResult {
  status: GameStatus;
  error: string | null;
  snapshot: UiSnapshot | null;
  config: GameConfig;
  busy: boolean;
  activity: EngineActivity | null;
  paused: boolean;
  hintAction: number | null;
  animationBatch: AnimationBatch | null;
  setConfig: (config: GameConfig) => void;
  start: (config?: GameConfig) => Promise<boolean>;
  submit: (actionId: number) => Promise<boolean>;
  requestHint: (policy: BotProfile) => Promise<void>;
  togglePause: () => Promise<void>;
  restart: () => Promise<void>;
  playAgain: () => Promise<boolean>;
  exportReplay: () => Promise<ReplayRecord>;
  loadReplay: (replay: ReplayRecord) => Promise<void>;
  clearError: () => void;
}

export function useGameEngine(): UseGameEngineResult {
  const client = useMemo(() => new EngineClient(), []);
  const [status, setStatus] = useState<GameStatus>("booting");
  const [error, setError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<UiSnapshot | null>(null);
  const [config, setConfigState] = useState<GameConfig>(loadStoredGameConfig);
  const [activity, setActivity] = useState<EngineActivity | null>(null);
  const [paused, setPaused] = useState(false);
  const [hintAction, setHintAction] = useState<number | null>(null);
  const [animationBatch, setAnimationBatch] = useState<AnimationBatch | null>(
    null,
  );
  const animationId = useRef(0);
  const activeConfig = useRef<GameConfig>(config);
  const startInFlight = useRef(false);

  useEffect(() => {
    let mounted = true;
    client.cancelDispose();
    void client.ready
      .then(() => {
        if (mounted) setStatus("ready");
      })
      .catch((reason: unknown) => {
        if (mounted) {
          setStatus("error");
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => {
      mounted = false;
      client.dispose();
    };
  }, [client]);

  useEffect(() => {
    return client.subscribeProgress((transition) => {
      setSnapshot(transition.snapshot);
      setAnimationBatch({
        id: ++animationId.current,
        hints: transition.animationHints,
      });
      setHintAction(null);
    });
  }, [client]);

  const setConfig = useCallback((next: GameConfig) => {
    activeConfig.current = next;
    setConfigState(next);
    storeGameConfig(next);
  }, []);

  const start = useCallback(
    async (requested?: GameConfig): Promise<boolean> => {
      if (startInFlight.current) return false;
      startInFlight.current = true;
      const next = requested ?? activeConfig.current;
      setConfig(next);
      setActivity("starting");
      setError(null);
      try {
        const nextSnapshot = await client.newGame(
          next.seed,
          next.humanSeat,
          next.botProfiles,
        );
        setSnapshot(nextSnapshot);
        setPaused(false);
        setHintAction(null);
        setAnimationBatch(null);
        setStatus("playing");
        return true;
      } catch (reason: unknown) {
        setStatus("error");
        setError(reason instanceof Error ? reason.message : String(reason));
        return false;
      } finally {
        startInFlight.current = false;
        setActivity(null);
      }
    },
    [client, setConfig],
  );

  const submit = useCallback(
    async (actionId: number): Promise<boolean> => {
      setActivity("turn");
      setError(null);
      try {
        const transition = await client.submit(actionId);
        setSnapshot(transition.snapshot);
        setAnimationBatch({
          id: ++animationId.current,
          hints: transition.animationHints,
        });
        setHintAction(null);
        return true;
      } catch (reason: unknown) {
        setError(reason instanceof Error ? reason.message : String(reason));
        return false;
      } finally {
        setActivity(null);
      }
    },
    [client],
  );

  const requestHint = useCallback(
    async (policy: BotProfile) => {
      setActivity("hint");
      try {
        setHintAction(await client.hint(policy));
      } catch (reason: unknown) {
        setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        setActivity(null);
      }
    },
    [client],
  );

  const togglePause = useCallback(async () => {
    setActivity("pause");
    try {
      const nextSnapshot = paused
        ? await client.resume()
        : await client.pause();
      setSnapshot(nextSnapshot);
      setPaused((value) => !value);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setActivity(null);
    }
  }, [client, paused]);

  const restart = useCallback(async (): Promise<void> => {
    await start(activeConfig.current);
  }, [start]);

  const playAgain = useCallback((): Promise<boolean> => {
    const current = activeConfig.current;
    return start(createNextGameConfig(current));
  }, [start]);

  const exportReplay = useCallback(() => client.exportReplay(), [client]);

  const loadReplay = useCallback(
    async (replay: ReplayRecord) => {
      setActivity("replay");
      setError(null);
      try {
        const nextSnapshot = await client.loadReplay(replay);
        const nextConfig: GameConfig = {
          seed: replay.seed,
          humanSeat: replay.humanSeat,
          botProfiles: replay.botProfiles,
        };
        setConfig(nextConfig);
        setSnapshot(nextSnapshot);
        setPaused(false);
        setAnimationBatch(null);
        setHintAction(null);
        setStatus("playing");
      } catch (reason: unknown) {
        setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        setActivity(null);
      }
    },
    [client, setConfig],
  );

  const clearError = useCallback(() => setError(null), []);

  return {
    status,
    error,
    snapshot,
    config,
    busy: activity != null,
    activity,
    paused,
    hintAction,
    animationBatch,
    setConfig,
    start,
    submit,
    requestHint,
    togglePause,
    restart,
    playAgain,
    exportReplay,
    loadReplay,
    clearError,
  };
}
