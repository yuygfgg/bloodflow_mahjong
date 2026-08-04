import { ACTION_SPACE_SIZE } from "../../pkg/bloodflow_mahjong_wasm.js";

/** Expand the two-word legal mask into sorted action ids. */
export function legalActionIdsFromMask(low: bigint, high: bigint): number[] {
  const actionSpaceSize = ACTION_SPACE_SIZE();
  const ids: number[] = [];
  for (let bit = 0; bit < 64; bit += 1) {
    if ((low >> BigInt(bit)) & 1n) {
      ids.push(bit);
    }
  }
  for (let bit = 0; bit < actionSpaceSize - 64; bit += 1) {
    if ((high >> BigInt(bit)) & 1n) {
      ids.push(64 + bit);
    }
  }
  return ids;
}

/** Read `[low, high]` from a WASM mask return value. */
export function readMaskWords(
  mask: ArrayLike<bigint | number> | undefined,
): [bigint, bigint] {
  if (mask == null) {
    return [0n, 0n];
  }
  const low = BigInt(mask[0] ?? 0);
  const high = BigInt(mask[1] ?? 0);
  return [low, high];
}

export function isLegalAction(actionId: number, low: bigint, high: bigint): boolean {
  if (actionId < 0 || actionId >= ACTION_SPACE_SIZE()) {
    return false;
  }
  if (actionId < 64) {
    return Boolean((low >> BigInt(actionId)) & 1n);
  }
  return Boolean((high >> BigInt(actionId - 64)) & 1n);
}
