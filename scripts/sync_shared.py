#!/usr/bin/env python3
"""Copy the canonical shared knowledge base into the backend and frontend.

    python scripts/sync_shared.py          # copy
    python scripts/sync_shared.py --check  # exit 1 if any copy is out of date (CI)

`shared/waste_knowledge.json` is the single source of truth for category names,
guidance and colours (English + Hindi). Backend and frontend are deployed
separately (Render/Railway vs Vercel), so each needs its own copy in-tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "shared" / "waste_knowledge.json"
TARGETS = [
    ROOT / "backend" / "app" / "data" / "waste_knowledge.json",
    ROOT / "frontend" / "src" / "data" / "waste_knowledge.json",
]
LIST_FIELDS = ("segregation", "disposal", "recycling", "safety", "common_mistakes", "environment")
CATEGORIES = ["organic", "paper_cardboard", "plastic", "metal", "glass", "e_waste", "hazardous"]


def validate(data: dict) -> None:
    cats = data["categories"]
    missing = [c for c in CATEGORIES if c not in cats]
    if missing:
        raise SystemExit(f"knowledge base missing categories: {missing}")
    for c in CATEGORIES:
        for field in ("name", "short", "description", "disposal", "safety"):
            value = cats[c].get(field)
            if not isinstance(value, dict) or not {"en", "hi"} <= set(value):
                raise SystemExit(f"{c}.{field} must have en and hi translations")
        # Bullet-list fields must be lists of strings in both languages. A plain string here once
        # rendered "Why it matters" one character per bullet (list("abc") == ["a", "b", "c"]).
        for field in LIST_FIELDS:
            value = cats[c].get(field)
            if not isinstance(value, dict) or not {"en", "hi"} <= set(value):
                raise SystemExit(f"{c}.{field} must have en and hi translations")
            for lang in ("en", "hi"):
                items = value[lang]
                if not isinstance(items, list) or not items or not all(isinstance(x, str) and x.strip() for x in items):
                    raise SystemExit(f"{c}.{field}.{lang} must be a non-empty list of strings")
            if len(value["en"]) != len(value["hi"]):
                raise SystemExit(f"{c}.{field}: en and hi must have the same number of items")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    raw = SOURCE.read_text(encoding="utf-8")
    validate(json.loads(raw))
    stale = []
    for t in TARGETS:
        if args.check:
            if not t.exists() or t.read_text(encoding="utf-8") != raw:
                stale.append(str(t.relative_to(ROOT)))
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(raw, encoding="utf-8")
            print(f"synced {t.relative_to(ROOT)}")
    if stale:
        print("out of date: " + ", ".join(stale) + "  (run python scripts/sync_shared.py)")
        sys.exit(1)


if __name__ == "__main__":
    main()
