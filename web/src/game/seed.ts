const MAX_U64 = (1n << 64n) - 1n;

export type RandomFill = (target: Uint32Array) => void;

const fillRandomValues: RandomFill = (target) => {
  globalThis.crypto.getRandomValues(target);
};

function randomU64(fill: RandomFill): bigint {
  const words = new Uint32Array(2);
  fill(words);
  return (BigInt(words[0]!) << 32n) | BigInt(words[1]!);
}

function parseU64(value: string | undefined): bigint | null {
  if (value == null || !/^\d+$/.test(value)) return null;
  const parsed = BigInt(value);
  return parsed <= MAX_U64 ? parsed : null;
}

export function createRandomSeed(
  excluded?: string,
  fill: RandomFill = fillRandomValues,
): string {
  const excludedValue = parseU64(excluded);
  let value: bigint;
  do {
    value = randomU64(fill);
  } while (value === excludedValue);
  return value.toString();
}
