import {
  Copy,
  Download,
  FileDown,
  FileUp,
  Pause,
  Play,
  RotateCcw,
  Settings,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { useState, type ChangeEvent, type ReactNode } from "react";
import type {
  BotProfile,
  ReplayRecord,
} from "../../../engine/wasm/js/src/protocol";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";
import type { GameConfig } from "../game/useGameEngine";
import {
  Phase,
  SEAT_NAMES,
  WIND_LABELS,
  patternLabels,
  replayFromIdentifier,
  replayIdentifier,
  tileLabel,
} from "../game/domain";
import { settlementStageLabel } from "../game/settlement";

const PROFILES: BotProfile[] = ["rule-fast", "rule-ev", "rule-nn"];
const PROFILE_LABEL: Record<BotProfile, string> = {
  "rule-fast": "极速规则",
  "rule-ev": "估值搜索",
  "rule-nn": "神经网络",
};

function saveReplayFile(replay: ReplayRecord): void {
  const contents = JSON.stringify(replay, null, 2);
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `bloodflow-replay-${replay.seed}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

interface StartScreenProps {
  status: "booting" | "ready" | "playing" | "error";
  busy: boolean;
  config: GameConfig;
  error: string | null;
  onConfig: (config: GameConfig) => void;
  onStart: () => void;
  onDismissError: () => void;
}

export function StartScreen({
  status,
  busy,
  config,
  error,
  onConfig,
  onStart,
  onDismissError,
}: StartScreenProps) {
  const updateBot = (absoluteSeat: number, value: BotProfile) => {
    const botProfiles = [...config.botProfiles] as [
      BotProfile,
      BotProfile,
      BotProfile,
      BotProfile,
    ];
    botProfiles[absoluteSeat] = value;
    onConfig({ ...config, botProfiles });
  };
  const updateHumanSeat = (seat: number) => {
    const botProfiles = [...config.botProfiles] as [
      BotProfile,
      BotProfile,
      BotProfile,
      BotProfile,
    ];
    botProfiles[seat] = "rule-fast";
    onConfig({ ...config, humanSeat: seat, botProfiles });
  };
  return (
    <div className="start-layer">
      <div className="start-panel" role="dialog" aria-labelledby="start-title">
        <div className="brand-lockup">
          <span className="brand-mark">血</span>
          <div>
            <h1 id="start-title">血流成河</h1>
          </div>
        </div>
        <div className="start-fields">
          <label>
            <span>种子</span>
            <input
              inputMode="numeric"
              pattern="[0-9]*"
              value={config.seed}
              onChange={(event) =>
                onConfig({ ...config, seed: event.target.value })
              }
            />
          </label>
        </div>
        <fieldset className="seat-picker">
          <legend>玩家座位</legend>
          <div className="seat-picker-options">
            {WIND_LABELS.map((wind, seat) => (
              <button
                className={
                  seat === config.humanSeat
                    ? "seat-choice selected"
                    : "seat-choice"
                }
                type="button"
                aria-pressed={seat === config.humanSeat}
                key={wind}
                onClick={() => updateHumanSeat(seat)}
              >
                {wind}
              </button>
            ))}
          </div>
        </fieldset>
        <div className="bot-grid">
          {[1, 2, 3].map((relativeSeat) => {
            const seat = (config.humanSeat + relativeSeat) & 3;
            return (
              <label key={seat}>
                <span>
                  {SEAT_NAMES[relativeSeat]} · {WIND_LABELS[seat]} Bot
                </span>
                <select
                  value={config.botProfiles[seat]}
                  onChange={(event) =>
                    updateBot(seat, event.target.value as BotProfile)
                  }
                >
                  {PROFILES.map((profile) => (
                    <option value={profile} key={profile}>
                      {PROFILE_LABEL[profile]}
                    </option>
                  ))}
                </select>
              </label>
            );
          })}
        </div>
        <button
          className="menu-button start-button"
          type="button"
          disabled={busy || status === "booting"}
          onClick={onStart}
        >
          <Play aria-hidden="true" />
          {status === "booting" ? "正在加载引擎" : "开始对局"}
        </button>
        <small className="license-note">
          AGPL-3.0-only · OpenRiichi GPL-3.0-or-later assets
        </small>
      </div>
      {error != null && <ErrorDialog error={error} onClose={onDismissError} />}
    </div>
  );
}

export function TopControls({
  paused,
  busy,
  muted,
  onPause,
  onRestart,
  onReplay,
  onSettings,
  onMute,
}: {
  paused: boolean;
  busy: boolean;
  muted: boolean;
  onPause: () => void;
  onRestart: () => void;
  onReplay: () => void;
  onSettings: () => void;
  onMute: () => void;
}) {
  return (
    <nav className="top-controls" aria-label="对局控制">
      <IconButton
        label={paused ? "继续" : "暂停"}
        onClick={onPause}
        disabled={busy}
      >
        {paused ? <Play aria-hidden="true" /> : <Pause aria-hidden="true" />}
      </IconButton>
      <IconButton label="重新开始" onClick={onRestart} disabled={busy}>
        <RotateCcw aria-hidden="true" />
      </IconButton>
      <IconButton label="回放" onClick={onReplay} disabled={busy}>
        <FileDown aria-hidden="true" />
      </IconButton>
      <IconButton label={muted ? "打开音效" : "静音"} onClick={onMute}>
        {muted ? (
          <VolumeX aria-hidden="true" />
        ) : (
          <Volume2 aria-hidden="true" />
        )}
      </IconButton>
      <IconButton label="设置" onClick={onSettings}>
        <Settings aria-hidden="true" />
      </IconButton>
    </nav>
  );
}

export function PauseOverlay({
  busy,
  onClose,
  onRestart,
  onReplay,
}: {
  busy: boolean;
  onClose: () => void;
  onRestart: () => void;
  onReplay: () => void;
}) {
  return (
    <div
      className="pause-layer"
      role="dialog"
      aria-modal="true"
      aria-labelledby="pause-title"
    >
      <div className="pause-menu">
        <h2 id="pause-title">暂停</h2>
        <button
          className="menu-button"
          type="button"
          disabled={busy}
          onClick={onClose}
        >
          <Play aria-hidden="true" />
          继续
        </button>
        <button
          className="menu-button"
          type="button"
          disabled={busy}
          onClick={onRestart}
        >
          <RotateCcw aria-hidden="true" />
          重新开始
        </button>
        <button
          className="menu-button"
          type="button"
          disabled={busy}
          onClick={onReplay}
        >
          <FileDown aria-hidden="true" />
          回放
        </button>
      </div>
    </div>
  );
}

export function SettingsDialog({
  muted,
  reducedMotion,
  highContrast,
  onMuted,
  onReducedMotion,
  onHighContrast,
  onClose,
}: {
  muted: boolean;
  reducedMotion: boolean;
  highContrast: boolean;
  onMuted: (value: boolean) => void;
  onReducedMotion: (value: boolean) => void;
  onHighContrast: (value: boolean) => void;
  onClose: () => void;
}) {
  return (
    <Modal title="设置" onClose={onClose}>
      <div className="settings-list">
        <Toggle
          label="音效"
          checked={!muted}
          onChange={(event) => onMuted(!event.target.checked)}
        />
        <Toggle
          label="减少动画"
          checked={reducedMotion}
          onChange={(event) => onReducedMotion(event.target.checked)}
        />
        <Toggle
          label="高对比度"
          checked={highContrast}
          onChange={(event) => onHighContrast(event.target.checked)}
        />
      </div>
      <p className="settings-license">资源与规则许可见 LICENSE。</p>
    </Modal>
  );
}

export function ReplayDialog({
  replay,
  onExport,
  onLoad,
  onClose,
}: {
  replay: ReplayRecord | null;
  onExport: () => Promise<ReplayRecord>;
  onLoad: (replay: ReplayRecord) => void;
  onClose: () => void;
}) {
  const [text, setText] = useState(
    replay == null ? "" : JSON.stringify(replay, null, 2),
  );
  const [identifier, setIdentifier] = useState("");
  const [message, setMessage] = useState("");

  const exportReplay = async () => {
    const next = await onExport();
    const json = JSON.stringify(next, null, 2);
    setText(json);
    setIdentifier(replayIdentifier(JSON.stringify(next)));
    setMessage("已生成");
  };

  const load = () => {
    try {
      let parsed: ReplayRecord;
      try {
        parsed = JSON.parse(text) as ReplayRecord;
      } catch {
        parsed = JSON.parse(replayFromIdentifier(text)) as ReplayRecord;
      }
      onLoad(parsed);
      setMessage("已载入");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "回放格式错误");
    }
  };

  return (
    <Modal title="回放" onClose={onClose} wide>
      <div className="replay-actions">
        <button
          className="menu-button"
          type="button"
          onClick={() => void exportReplay()}
        >
          <Download aria-hidden="true" />
          导出当前动作
        </button>
        <button className="menu-button" type="button" onClick={load}>
          <FileUp aria-hidden="true" />
          载入 JSON
        </button>
        {identifier && (
          <button
            className="menu-button"
            type="button"
            onClick={() => void navigator.clipboard?.writeText(identifier)}
          >
            <Copy aria-hidden="true" />
            复制回放 ID
          </button>
        )}
      </div>
      <textarea
        className="replay-text"
        value={text}
        onChange={(event) => setText(event.target.value)}
        spellCheck={false}
        aria-label="回放 JSON"
      />
      {message && <p className="dialog-message">{message}</p>}
    </Modal>
  );
}

export function WinNotice({ snapshot }: { snapshot: UiSnapshot }) {
  const winEvent = [...snapshot.stepEvents]
    .reverse()
    .find((event) => event[0] === 8);
  const winner = winEvent?.[1] ?? -1;
  const win = winner >= 0 && winner < 4 ? snapshot.lastWins[winner] : null;
  if (win == null || winEvent == null) return null;
  const labels = patternLabels(win.patterns);
  const eventLabels = labels.filter((label) =>
    ["抢杠胡", "杠上炮", "杠上开花", "海底捞月", "天胡", "地胡"].includes(
      label,
    ),
  );
  const shapeLabels = labels.filter((label) => !eventLabels.includes(label));
  return (
    <div className="win-notice" role="status">
      <strong>和牌</strong>
      <span>{tileLabel(winEvent[3])}</span>
      <span>
        {shapeLabels.join(" · ") || "平胡"} × {win.shapeMultiplier}
      </span>
      {eventLabels.length > 0 && <span>{eventLabels.join(" · ")}</span>}
      <b>{win.multiplier} 倍</b>
    </div>
  );
}

export function SettlementOverlay({
  snapshot,
  busy,
  onExportReplay,
  onPlayAgain,
}: {
  snapshot: UiSnapshot;
  busy: boolean;
  onExportReplay: () => Promise<ReplayRecord>;
  onPlayAgain: () => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState("");

  const downloadReplay = async () => {
    setDownloading(true);
    setDownloadError("");
    try {
      saveReplayFile(await onExportReplay());
    } catch (error) {
      setDownloadError(error instanceof Error ? error.message : "回放下载失败");
    } finally {
      setDownloading(false);
    }
  };

  if (snapshot.phase !== Phase.Finished) return null;
  const settlement = snapshot.wallSettlement;
  return (
    <div
      className="settlement-layer"
      role="dialog"
      aria-live="polite"
      aria-modal="true"
      aria-labelledby="settlement-title"
      aria-busy={busy || downloading}
    >
      <div className="settlement-panel">
        <span className="settlement-stage" id="settlement-title">
          {settlement == null
            ? "结算"
            : settlementStageLabel(snapshot.stepEvents)}
        </span>
        {settlement != null && (
          <div className="settlement-grid">
            {[0, 1, 2, 3].map((seat) => (
              <div
                className={`settlement-seat ${settlement.flowerPig[seat] ? "pig" : ""}`}
                key={seat}
              >
                <strong>{SEAT_NAMES[seat]}</strong>
                <span>
                  {settlement.flowerPig[seat]
                    ? "花猪"
                    : settlement.ready[seat]
                      ? "听牌"
                      : "未听"}
                </span>
                <small>{settlement.maxShapeMultipliers[seat]} 倍</small>
              </div>
            ))}
          </div>
        )}
        <div className="final-scores">
          {snapshot.scores.map((score, seat) => (
            <span key={seat}>
              {SEAT_NAMES[seat]} {score.toLocaleString("zh-CN")}
            </span>
          ))}
        </div>
        {snapshot.rankings != null && (
          <p>
            排名：
            {snapshot.rankings.map((seat) => SEAT_NAMES[seat]).join(" · ")}
          </p>
        )}
        <div className="settlement-actions">
          <button
            className="menu-button"
            type="button"
            disabled={busy || downloading}
            onClick={() => void downloadReplay()}
          >
            <Download aria-hidden="true" />
            {downloading ? "正在下载" : "下载回放"}
          </button>
          <button
            className="menu-button"
            type="button"
            disabled={busy || downloading}
            onClick={() => {
              setDownloadError("");
              onPlayAgain();
            }}
          >
            <RotateCcw aria-hidden="true" />
            {busy ? "正在开局" : "再来一局"}
          </button>
        </div>
        {downloadError && (
          <p className="settlement-download-error" role="alert">
            {downloadError}
          </p>
        )}
      </div>
    </div>
  );
}

export function ErrorDialog({
  error,
  onClose,
}: {
  error: string;
  onClose: () => void;
}) {
  return (
    <div className="error-dialog" role="alertdialog" aria-modal="true">
      <strong>引擎错误</strong>
      <p>{error}</p>
      <button className="menu-button" type="button" onClick={onClose}>
        <X aria-hidden="true" />
        关闭
      </button>
    </div>
  );
}

function IconButton({
  label,
  children,
  onClick,
  disabled = false,
}: {
  label: string;
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      className="icon-button"
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className="toggle-track" aria-hidden="true">
        <span />
      </span>
    </label>
  );
}

function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  return (
    <div
      className="modal-layer"
      role="dialog"
      aria-modal="true"
      aria-labelledby={`${title}-title`}
    >
      <div className={`modal-panel ${wide ? "wide" : ""}`}>
        <div className="modal-heading">
          <h2 id={`${title}-title`}>{title}</h2>
          <button
            className="icon-button"
            type="button"
            onClick={onClose}
            aria-label="关闭"
            title="关闭"
          >
            <X aria-hidden="true" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
