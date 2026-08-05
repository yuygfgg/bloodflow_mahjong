import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
} from "react";
import type {
  AnimationHint,
  BotProfile,
  ReplayRecord,
} from "../../engine/wasm/js/src/protocol";
import {
  ActionBar,
  CenterStatus,
  SeatHuds,
  canSelectExchangeTile,
} from "./components/GameHud";
import {
  ErrorDialog,
  PauseOverlay,
  ReplayDialog,
  SettingsDialog,
  SettlementOverlay,
  StartScreen,
  TopControls,
  WinNotice,
} from "./components/Overlays";
import { TableCanvas } from "./components/TableCanvas";
import { GameAudio } from "./game/audio";
import {
  Action,
  Phase,
  discardTileAction,
  tileSuit,
  isActionLegal,
  tileLabel,
} from "./game/domain";
import type { ExchangeSelection } from "./game/domain";
import { useGameEngine, type EngineActivity } from "./game/useGameEngine";

interface Preferences {
  muted: boolean;
  reducedMotion: boolean;
  highContrast: boolean;
}

interface EventCue {
  id: number;
  text: string;
  payment?: Extract<AnimationHint, { kind: "payment" }>;
}

function loadPreferences(): Preferences {
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  try {
    const stored = localStorage.getItem("bloodflow-ui-preferences");
    return stored == null
      ? { muted: false, reducedMotion, highContrast: false }
      : {
          ...JSON.parse(stored),
          reducedMotion: JSON.parse(stored).reducedMotion ?? reducedMotion,
        };
  } catch {
    return { muted: false, reducedMotion, highContrast: false };
  }
}

export function App() {
  const game = useGameEngine();
  const audio = useMemo(() => new GameAudio(), []);
  const [preferences, setPreferences] = useState<Preferences>(loadPreferences);
  const [setupOpen, setSetupOpen] = useState(true);
  const [sceneReady, setSceneReady] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [replayOpen, setReplayOpen] = useState(false);
  const [pendingExchangeSelections, setPendingExchangeSelections] = useState<
    ExchangeSelection[]
  >([]);
  const [pendingMissingSuit, setPendingMissingSuit] = useState<number | null>(
    null,
  );
  const [eventCue, setEventCue] = useState<EventCue | null>(null);
  const [replay, setReplay] = useState<ReplayRecord | null>(null);
  const [hintPolicy, setHintPolicy] = useState<BotProfile>("rule-ev");

  audio.muted = preferences.muted;

  const updatePreferences = useCallback((next: Preferences) => {
    setPreferences(next);
    localStorage.setItem("bloodflow-ui-preferences", JSON.stringify(next));
  }, []);

  const resetRoundUi = useCallback(() => {
    setPendingExchangeSelections([]);
    setPendingMissingSuit(null);
    setEventCue(null);
  }, []);

  useEffect(() => {
    setPendingExchangeSelections([]);
    setPendingMissingSuit(null);
  }, [game.snapshot?.phase]);

  useEffect(() => {
    const batch = game.animationBatch;
    if (batch == null) return;
    const timers: number[] = [];
    let delay = 0;
    for (const hint of batch.hints) {
      const cue = cueForHint(hint);
      if (cue == null) continue;
      const cueDelay = delay;
      timers.push(
        window.setTimeout(
          () => {
            playHint(audio, hint);
            setEventCue({
              id: batch.id * 1000 + cueDelay,
              text: cue,
              payment: hint.kind === "payment" ? hint : undefined,
            });
          },
          preferences.reducedMotion ? 0 : cueDelay,
        ),
      );
      delay += durationForHint(hint);
    }
    timers.push(
      window.setTimeout(
        () => setEventCue(null),
        preferences.reducedMotion ? 250 : delay + 700,
      ),
    );
    return () => timers.forEach(window.clearTimeout);
  }, [audio, game.animationBatch, preferences.reducedMotion]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.key !== "Escape" ||
        game.snapshot == null ||
        settingsOpen ||
        replayOpen
      )
        return;
      event.preventDefault();
      void game.togglePause();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [game, replayOpen, settingsOpen]);

  const start = async () => {
    audio.unlock();
    if (await game.start()) {
      resetRoundUi();
      setSetupOpen(false);
    }
  };

  const playAgain = async () => {
    audio.unlock();
    setEventCue(null);
    if (await game.playAgain()) {
      resetRoundUi();
      setSetupOpen(false);
    }
  };

  const submit = useCallback(
    (actionId: number) => {
      audio.unlock();
      audio.play("click");
      if (
        actionId >= Action.ChooseMissingOffset &&
        actionId < Action.DiscardOffset
      ) {
        setPendingMissingSuit(null);
      }
      void game.submit(actionId);
    },
    [audio, game],
  );

  const toggleExchangeTile = useCallback(
    (tile: number, sourceKey: string) => {
      const snapshot = game.snapshot;
      if (snapshot == null || game.busy || snapshot.decisionActor !== 0) return;
      audio.unlock();
      audio.play("tile");

      const existing = pendingExchangeSelections.findIndex(
        (selection) => selection.sourceKey === sourceKey,
      );
      if (existing >= 0) {
        setPendingExchangeSelections((current) =>
          current.filter((_, index) => index !== existing),
        );
        return;
      }

      const totalSelected =
        snapshot.exchangeSelectedCount + pendingExchangeSelections.length;
      if (totalSelected >= 3 || !canSelectExchangeTile(snapshot, tile)) return;

      const stagedSuit =
        pendingExchangeSelections.length > 0
          ? tileSuit(pendingExchangeSelections[0]!.tile)
          : snapshot.exchangeSelectionSuit;
      if (stagedSuit >= 0 && tileSuit(tile) !== stagedSuit) return;

      setPendingExchangeSelections((current) => [
        ...current,
        { tile, sourceKey },
      ]);
    },
    [audio, game.busy, game.snapshot, pendingExchangeSelections],
  );

  const confirmExchange = useCallback(async () => {
    const snapshot = game.snapshot;
    const selections = pendingExchangeSelections;
    if (
      snapshot == null ||
      game.busy ||
      snapshot.decisionActor !== 0 ||
      snapshot.exchangeSelectedCount + selections.length !== 3 ||
      selections.length === 0
    ) {
      return;
    }
    setPendingExchangeSelections([]);
    for (const selection of selections) {
      const accepted = await game.submit(
        Action.ExchangeOffset + selection.tile,
      );
      if (!accepted) return;
    }
  }, [game, pendingExchangeSelections]);

  const tileClick = useCallback(
    (tile: number, sourceKey = `hand:${tile}`) => {
      const snapshot = game.snapshot;
      if (snapshot == null || game.busy || snapshot.decisionActor !== 0) return;
      if (snapshot.phase === Phase.Exchange) {
        toggleExchangeTile(tile, sourceKey);
      } else if (snapshot.phase === Phase.Turn) {
        audio.unlock();
        audio.play("tile");
        const action = discardTileAction(tile);
        if (isActionLegal(snapshot, action)) submit(action);
      }
    },
    [audio, game.snapshot, game.busy, submit, toggleExchangeTile],
  );

  const pause = () => {
    void game.togglePause();
  };

  const restart = () => {
    setReplayOpen(false);
    setSettingsOpen(false);
    void game.restart();
  };

  const openReplay = async () => {
    try {
      setReplay(await game.exportReplay());
    } catch {
      setReplay(null);
    }
    setReplayOpen(true);
  };

  const loadReplay = (next: ReplayRecord) => {
    setReplayOpen(false);
    void game.loadReplay(next);
  };

  const snapshot = game.snapshot;
  const showStart = snapshot == null || setupOpen;
  const handleSceneReady = useCallback(() => setSceneReady(true), []);

  return (
    <main
      className={`app-shell ${preferences.highContrast ? "high-contrast" : ""}`}
      aria-busy={game.busy}
      onPointerDown={() => audio.unlock()}
    >
      <TableCanvas
        snapshot={snapshot}
        hintAction={game.hintAction}
        pendingExchangeSelectionKeys={pendingExchangeSelections.map(
          (selection) => selection.sourceKey,
        )}
        animationHints={game.animationBatch?.hints ?? []}
        reducedMotion={preferences.reducedMotion}
        onTileClick={tileClick}
        onReady={handleSceneReady}
      />

      {snapshot != null && (
        <div className="game-ui">
          <SeatHuds
            snapshot={snapshot}
            config={game.config}
            pendingMissingSuit={pendingMissingSuit}
          />
          <CenterStatus snapshot={snapshot} />
          <TopControls
            paused={game.paused}
            busy={game.busy}
            muted={preferences.muted}
            onPause={pause}
            onRestart={restart}
            onReplay={() => void openReplay()}
            onSettings={() => setSettingsOpen(true)}
            onMute={() =>
              updatePreferences({ ...preferences, muted: !preferences.muted })
            }
          />
          <ActionBar
            snapshot={snapshot}
            busy={game.busy}
            pendingExchangeSelections={pendingExchangeSelections}
            pendingMissingSuit={pendingMissingSuit}
            hintPolicy={hintPolicy}
            onExchangeTile={toggleExchangeTile}
            onConfirmExchange={() => void confirmExchange()}
            onPendingMissing={setPendingMissingSuit}
            onSubmit={submit}
            onHintPolicyChange={setHintPolicy}
            onHint={(policy) => void game.requestHint(policy)}
          />
          <WinNotice snapshot={snapshot} />
          <SettlementOverlay
            snapshot={snapshot}
            busy={game.busy}
            onExportReplay={game.exportReplay}
            onPlayAgain={() => void playAgain()}
          />
        </div>
      )}

      <div className="game-overlay">
        {eventCue != null && <EventCueView cue={eventCue} />}
        {(game.busy || !sceneReady) && (
          <div className="loading-indicator" role="status">
            {!sceneReady ? "正在布置牌桌" : engineActivityLabel(game.activity)}
          </div>
        )}
      </div>

      {showStart && (
        <StartScreen
          status={game.status}
          busy={game.busy}
          config={game.config}
          error={game.error}
          onConfig={game.setConfig}
          onStart={() => void start()}
          onDismissError={game.clearError}
        />
      )}
      {!showStart && game.paused && !settingsOpen && !replayOpen && (
        <PauseOverlay
          busy={game.busy}
          onClose={pause}
          onRestart={restart}
          onReplay={() => void openReplay()}
        />
      )}
      {settingsOpen && (
        <SettingsDialog
          muted={preferences.muted}
          reducedMotion={preferences.reducedMotion}
          highContrast={preferences.highContrast}
          onMuted={(muted) => updatePreferences({ ...preferences, muted })}
          onReducedMotion={(reducedMotion) =>
            updatePreferences({ ...preferences, reducedMotion })
          }
          onHighContrast={(highContrast) =>
            updatePreferences({ ...preferences, highContrast })
          }
          onClose={() => setSettingsOpen(false)}
        />
      )}
      {replayOpen && (
        <ReplayDialog
          replay={replay}
          onExport={game.exportReplay}
          onLoad={loadReplay}
          onClose={() => setReplayOpen(false)}
        />
      )}
      {!showStart && game.error != null && (
        <ErrorDialog error={game.error} onClose={game.clearError} />
      )}
    </main>
  );
}

function engineActivityLabel(activity: EngineActivity | null): string {
  switch (activity) {
    case "starting":
      return "正在开局";
    case "hint":
      return "正在计算提示";
    case "pause":
      return "正在同步对局";
    case "replay":
      return "正在载入回放";
    case "turn":
    default:
      return "对手思考中";
  }
}

function cueForHint(hint: AnimationHint): string | null {
  switch (hint.kind) {
    case "draw":
      return hint.replacement ? "杠后补牌" : "摸牌";
    case "discard":
      return `${tileLabel(hint.tile)}`;
    case "meld":
      return ["碰", "直杠", "碰杠", "暗杠"][hint.meldKind] ?? "副露";
    case "hu":
      return `和牌 · ${hint.multiplier} 倍`;
    case "payment":
      return `${hint.amount.toLocaleString("zh-CN")} 点`;
    case "exchange_complete":
      return "换牌完成";
    case "missing_revealed":
      return "定缺公开";
    case "settlement_stage":
      return hint.stage === 0 ? "查花猪" : "查大叫";
    case "game_end":
      return "本局结束";
  }
}

function durationForHint(hint: AnimationHint): number {
  switch (hint.kind) {
    case "draw":
      return 350;
    case "discard":
      return 450;
    case "meld":
      return 500;
    case "hu":
      return 1000;
    case "payment":
      return 500;
    default:
      return 350;
  }
}

function playHint(audio: GameAudio, hint: AnimationHint): void {
  switch (hint.kind) {
    case "draw":
      audio.play("draw");
      break;
    case "discard":
      audio.play("discard");
      break;
    case "meld":
      audio.play("slide");
      break;
    case "hu":
    case "missing_revealed":
      audio.play("reveal");
      break;
    case "payment":
      audio.play("score_count");
      break;
    case "exchange_complete":
      audio.play("flip");
      break;
    default:
      break;
  }
}

function EventCueView({ cue }: { cue: EventCue }) {
  const payment = cue.payment;
  const positions = [
    [50, 86],
    [88, 48],
    [50, 12],
    [12, 48],
  ];
  const style =
    payment == null
      ? undefined
      : ({
          "--from-x": `${positions[payment.payerRelative]?.[0] ?? 50}%`,
          "--from-y": `${positions[payment.payerRelative]?.[1] ?? 50}%`,
          "--to-x": `${positions[payment.payeeRelative]?.[0] ?? 50}%`,
          "--to-y": `${positions[payment.payeeRelative]?.[1] ?? 50}%`,
        } as CSSProperties);
  return payment == null ? (
    <div className="event-cue" key={cue.id}>
      {cue.text}
    </div>
  ) : (
    <div className="payment-token" style={style} key={cue.id}>
      +{cue.text}
    </div>
  );
}
