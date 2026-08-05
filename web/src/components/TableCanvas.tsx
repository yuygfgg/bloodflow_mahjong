import { useEffect, useRef } from "react";
import type { AnimationHint } from "../../../engine/wasm/js/src/protocol";
import type { UiSnapshot } from "../../../engine/wasm/js/src/types";
import { TableRenderer } from "../scene/TableRenderer";

interface TableCanvasProps {
  snapshot: UiSnapshot | null;
  hintAction: number | null;
  pendingExchangeSelectionKeys: readonly string[];
  animationHints: readonly AnimationHint[];
  reducedMotion: boolean;
  onTileClick: (tile: number, sourceKey?: string) => void;
  onReady: () => void;
}

export function TableCanvas({
  snapshot,
  hintAction,
  pendingExchangeSelectionKeys,
  animationHints,
  reducedMotion,
  onTileClick,
  onReady,
}: TableCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onTileClickRef = useRef(onTileClick);
  const reducedMotionRef = useRef(reducedMotion);
  onTileClickRef.current = onTileClick;
  reducedMotionRef.current = reducedMotion;

  // React owns the canvas. Three.js owns the scene graph below it.
  const rendererRef = useRef<TableRenderer | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas == null) return;
    const renderer = new TableRenderer(canvas, {
      reducedMotion: reducedMotionRef.current,
      onReady,
      onTileClick: (tile, sourceKey) => onTileClickRef.current(tile, sourceKey),
    });
    rendererRef.current = renderer;
    return () => {
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [onReady]);

  useEffect(() => {
    rendererRef.current?.setSnapshot(
      snapshot,
      hintAction,
      pendingExchangeSelectionKeys,
      animationHints,
    );
  }, [animationHints, hintAction, pendingExchangeSelectionKeys, snapshot]);

  useEffect(() => {
    rendererRef.current?.setReducedMotion(reducedMotion);
  }, [reducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      className="table-canvas"
      aria-label="麻将桌"
    />
  );
}
