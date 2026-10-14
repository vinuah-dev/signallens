import { describe, expect, it } from "vitest";
import { bytes, num, pct, score } from "./format";
import { niceTicks } from "../charts/core";

describe("format", () => {
  it("formats percentages and missing values", () => {
    expect(pct(0.9449)).toBe("94%");
    expect(pct(0.9449, 1)).toBe("94.5%");
    expect(pct(null)).toBe("—");
    expect(pct(Number.NaN)).toBe("—");
  });
  it("formats numbers at a human scale", () => {
    expect(num(1234567)).toBe("1,234,567");
    expect(num(3.14159)).toBe("3.142");
    expect(num(0.00012)).toBe("1.20e-4");
    expect(score(0.96491)).toBe("0.965");
  });
  it("formats byte sizes", () => {
    expect(bytes(512)).toBe("512 B");
    expect(bytes(4500)).toBe("4.4 KB");
    expect(bytes(25 * 1024 * 1024)).toBe("25.0 MB");
  });
  it("produces nice axis ticks", () => {
    expect(niceTicks(0, 1, 4)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(niceTicks(3, 3)).toEqual([3]);
  });
});
