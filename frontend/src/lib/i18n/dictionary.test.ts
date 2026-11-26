import { describe, expect, it } from "vitest";
import { dictionary } from "./dictionary";
import knowledge from "../../data/waste_knowledge.json";
import { ITEM_TO_CATEGORY } from "../taxonomy";
import { CATEGORIES } from "../types";

/** Collects "a.b.c" key paths so EN and HI can be compared structurally. */
function paths(obj: unknown, prefix = ""): string[] {
  if (obj === null || typeof obj !== "object") return [prefix];
  if (Array.isArray(obj)) return [`${prefix}[${obj.length}]`, ...obj.flatMap((v, i) => paths(v, `${prefix}[${i}]`))];
  return Object.entries(obj).flatMap(([k, v]) => paths(v, prefix ? `${prefix}.${k}` : k));
}

describe("i18n dictionary", () => {
  it("Hindi has exactly the same keys (and list lengths) as English", () => {
    expect(paths(dictionary.hi).sort()).toEqual(paths(dictionary.en).sort());
  });
  it("has no empty strings", () => {
    for (const lang of ["en", "hi"] as const) {
      const walk = (v: unknown): void => {
        if (typeof v === "string") expect(v.trim().length).toBeGreaterThan(0);
        else if (v && typeof v === "object") Object.values(v).forEach(walk);
      };
      walk(dictionary[lang]);
    }
  });
  it("names every model category in both languages", () => {
    for (const c of CATEGORIES) {
      expect(dictionary.en.categories[c]).toBeTruthy();
      expect(dictionary.hi.categories[c]).toBeTruthy();
    }
  });
});

describe("knowledge base", () => {
  it("covers every category and item of the model taxonomy", () => {
    expect(Object.keys(knowledge.categories).sort()).toEqual([...CATEGORIES].sort());
    expect(Object.keys(knowledge.items).sort()).toEqual(Object.keys(ITEM_TO_CATEGORY).sort());
    for (const c of Object.values(ITEM_TO_CATEGORY)) expect(CATEGORIES).toContain(c);
  });
});

describe("knowledge base guidance lists", () => {
  const LIST_FIELDS = ["segregation", "disposal", "recycling", "safety", "common_mistakes", "environment"] as const;
  it("every bullet field is a non-empty list of full sentences in EN and HI with matching lengths", () => {
    for (const [c, k] of Object.entries(knowledge.categories as unknown as Record<string, Record<string, { en: unknown; hi: unknown }>>)) {
      for (const f of LIST_FIELDS) {
        const v = k[f];
        expect(Array.isArray(v.en), `${c}.${f}.en`).toBe(true);
        expect(Array.isArray(v.hi), `${c}.${f}.hi`).toBe(true);
        const en = v.en as string[], hi = v.hi as string[];
        expect(en.length, `${c}.${f}`).toBeGreaterThan(0);
        expect(hi.length, `${c}.${f}`).toBe(en.length);
        for (const s of [...en, ...hi]) expect(s.length, `${c}.${f}: "${s}"`).toBeGreaterThan(15);
      }
    }
  });
});
