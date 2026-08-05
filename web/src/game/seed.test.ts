import { describe, expect, it } from "vitest";
import { createRandomSeed } from "./seed";

const MAX_U64 = (1n << 64n) - 1n;

describe("game seeds", () => {
  it("covers the full unsigned 64-bit range", () => {
    const low = createRandomSeed(undefined, (target) => target.fill(0));
    const high = createRandomSeed(undefined, (target) =>
      target.fill(0xffffffff),
    );

    expect(low).toBe("0");
    expect(high).toBe(MAX_U64.toString());
  });

  it("generates a different seed for a new round", () => {
    const values = [7, 8];
    const seed = createRandomSeed("7", (target) => {
      target.set([0, values.shift()!]);
    });

    expect(seed).toBe("8");
    expect(values).toHaveLength(0);
  });
});
