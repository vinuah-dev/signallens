import { describe, expect, it } from "vitest";
import { checkFile, checkNumber } from "./validate";

const limits = { max_upload_bytes: 1024, extensions: [".csv", ".tsv", ".txt"] };
const file = (name: string, size: number) => ({ name, size }) as File;
const spec = { min: 0, max: 10, integer: false };

describe("checkFile", () => {
  it("accepts a csv within limits", () => expect(checkFile(file("data.CSV", 10), limits)).toBeNull());
  it("rejects other formats with an Excel hint", () => {
    expect(checkFile(file("book.xlsx", 10), limits)).toMatch(/Save as/);
    expect(checkFile(file("data.json", 10), limits)).toMatch(/aren’t supported/);
  });
  it("rejects empty and oversized files", () => {
    expect(checkFile(file("a.csv", 0), limits)).toMatch(/empty/);
    expect(checkFile(file("a.csv", 4096), limits)).toMatch(/limit/);
  });
});

describe("checkNumber", () => {
  it("treats blank as empty (missing)", () => expect(checkNumber("  ", spec)).toEqual({ state: "empty" }));
  it("parses plain, signed, decimal, exponent and thousands forms", () => {
    expect(checkNumber("4.5", spec)).toEqual({ state: "ok", value: 4.5 });
    expect(checkNumber("-.5", { ...spec, min: -1 })).toEqual({ state: "ok", value: -0.5 });
    expect(checkNumber("1e1", spec)).toEqual({ state: "ok", value: 10 });
    expect(checkNumber("1,000", { ...spec, max: 2000 })).toEqual({ state: "ok", value: 1000 });
  });
  it("rejects text", () => {
    expect(checkNumber("abc", spec).state).toBe("error");
    expect(checkNumber("4.5.6", spec).state).toBe("error");
    expect(checkNumber("12kg", spec).state).toBe("error");
  });
  it("flags out-of-range values but still accepts them", () => {
    const near = checkNumber("11", spec);
    const far = checkNumber("100", spec);
    expect(near.state === "ok" && near.warning).toMatch(/Outside training range/);
    expect(far.state === "ok" && far.warning).toMatch(/Far outside/);
  });
  it("notes fractions for integer-only features", () => {
    const r = checkNumber("2.5", { ...spec, integer: true });
    expect(r.state === "ok" && r.warning).toMatch(/whole numbers/);
  });
});
