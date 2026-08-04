/**
 * Client-facing TypeScript surface.
 *
 * Engine constants and low-level Game types come from the WASM module, the
 * same way the Python extension exposes module attributes. This package only
 * adds client-shaped helpers on top.
 */
export * from "../../pkg/bloodflow_mahjong_wasm.js";
export { default as init } from "../../pkg/bloodflow_mahjong_wasm.js";

export * from "./legal.ts";
export * from "./protocol.ts";
export * from "./snapshot.ts";
export type {
  EventRecord,
  GameLike,
  MeldView,
  ObservationBuffers,
  RiverEntry,
  UiSnapshot,
} from "./types.ts";
