#!/usr/bin/env bash
# Build the WASM cdylib and generate JS/TS glue under pkg/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENGINE_ROOT="$(cd "$ROOT/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$ENGINE_ROOT/target}"
OUT_DIR="${1:-$ROOT/pkg}"
PROFILE="${PROFILE:-release}"

cd "$ENGINE_ROOT"

echo "Building bloodflow-mahjong-wasm (${PROFILE}) for wasm32-unknown-unknown..."
cargo build -p bloodflow-mahjong-wasm --target wasm32-unknown-unknown --"${PROFILE}"

WASM_PATH="$TARGET_DIR/wasm32-unknown-unknown/${PROFILE}/bloodflow_mahjong_wasm.wasm"
if [[ ! -f "$WASM_PATH" ]]; then
  echo "missing wasm artifact: $WASM_PATH" >&2
  exit 1
fi

echo "Generating JS bindings into ${OUT_DIR}..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen \
  --target web \
  --typescript \
  --out-dir "$OUT_DIR" \
  --out-name bloodflow_mahjong_wasm \
  "$WASM_PATH"

echo "Done. Artifact size: $(du -h "$OUT_DIR/bloodflow_mahjong_wasm_bg.wasm" | cut -f1)"
