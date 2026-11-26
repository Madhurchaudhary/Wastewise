"""Dataset v1.1: the v1 evaluation splits, unchanged, plus a larger training split.

    python -m ml.src.expand_train --raw-dir /path/to/raw --out-dir /path/to/processed_v11

Why
---
Dataset v1 capped every category at 1,100 images, which kept CPU training time low. As a result
only ~4.8k of the ~24k usable raw images were used for training. v1.1 adds more of the unused
images **to the training split only**. Validation, test and OOD stay identical to v1 (same files,
same uids), so v1 and v1.1 models are compared on exactly the same held-out images.

Leakage protection (same similarity measure as prepare_data.py)
--------------------------------------------------------------
* every v1 image is rebuilt from its source_path and must match the manifest MD5;
* candidates whose MD5 already appears in v1, or appears with conflicting labels, are dropped;
* candidates that are near-duplicates (thumbnail correlation >= --group-threshold with a similar
  colour layout) of ANY v1 validation/test/OOD image are dropped. v1 put whole near-duplicate
  groups into one split, and this rule keeps that guarantee;
* candidates that are near-copies (>= --collapse-threshold) of a v1 training image, or of another
  candidate, are collapsed (they add no information);
* candidates near-duplicate to an image with a different category are dropped (label noise).

Then each category's training split is topped up to --train-per-category images. The extra
images are spread over the category's source folders as evenly as possible (allocate_quota).

Outputs: ``out_dir/images/...``, ``out_dir/manifest.csv`` and
``ml/datasets/manifests/manifest_v1_1.csv`` (one row per image, v1 rows first).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import md5_file
from .prepare_data import (IMG_EXTS, UnionFind, _scan_file, _write_processed, allocate_quota,
                           near_duplicate_pairs)
from .taxonomy import CATEGORIES, SOURCE_FOLDERS

REPO_ML = Path(__file__).resolve().parents[1]
MAX_COLOR_DIFF = 20.0


def _max_sim_to(cand_t, cand_c, ref_t, ref_c, threshold: float, chunk: int = 1024) -> np.ndarray:
    """For each candidate: True if any reference image is a near-duplicate (sim >= threshold and similar colours)."""
    hit = np.zeros(len(cand_t), dtype=bool)
    for s in range(0, len(cand_t), chunk):
        sim = cand_t[s:s + chunk] @ ref_t.T
        ii, jj = np.nonzero(sim >= threshold)
        for a, b in zip(ii, jj):
            if not hit[s + a] and float(np.abs(cand_c[s + a] - ref_c[b]).mean()) < MAX_COLOR_DIFF:
                hit[s + a] = True
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.environ.get("WASTEWISE_RAW_DIR", str(REPO_ML / "datasets/raw")))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--v1-manifest", default=str(REPO_ML / "datasets/manifests/manifest_v1.csv"))
    ap.add_argument("--train-per-category", type=int, default=1800)
    ap.add_argument("--group-threshold", type=float, default=0.97)
    ap.add_argument("--collapse-threshold", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    t0 = time.time()

    v1 = pd.read_csv(args.v1_manifest, dtype={"uid": str, "dhash": str})
    # ---------- 1. rebuild v1 exactly ----------
    srcs = [str(raw / p) for p in v1["source_path"]]
    missing = [p for p in srcs if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} v1 source files missing, e.g. {missing[0]} - run ml/scripts/fetch_datasets.sh")
    with ProcessPoolExecutor(args.workers) as ex:
        md5s = list(ex.map(md5_file, srcs, chunksize=64))
    bad = int((np.array(md5s) != v1["md5"].to_numpy()).sum())
    if bad:
        raise SystemExit(f"{bad} v1 source files do not match the manifest MD5 - wrong dataset version")
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_write_processed, [(s, str(out / f)) for s, f in zip(srcs, v1["file"])], chunksize=64))
    print(f"[v1] rebuilt {len(v1)} images, all MD5s match  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 2. scan all in-distribution raw images ----------
    rows = []
    for sf in SOURCE_FOLDERS:
        if sf.role != "train":
            continue
        folder = raw / sf.rel_path
        for p in sorted(q for q in folder.iterdir() if q.suffix.lower() in IMG_EXTS):
            rows.append({"path": str(p), "dataset": sf.dataset, "stratum": sf.rel_path,
                         "category": sf.category, "item": sf.item})
    with ProcessPoolExecutor(args.workers) as ex:
        scanned = list(ex.map(_scan_file, [(r["path"],) for r in rows], chunksize=64))
    df = pd.DataFrame([{**r, **s} for r, s in zip(rows, scanned) if s["error"] is None])
    print(f"[scan] {len(df)} readable in-distribution raw images  ({time.time()-t0:.0f}s)", flush=True)

    # signatures of v1 images (scan their source files the same way)
    v1_paths = set(srcs)
    sig = df.set_index("path")
    v1_extra = [p for p in srcs if p not in sig.index]   # v1 OOD images live outside the train folders
    if v1_extra:
        with ProcessPoolExecutor(args.workers) as ex:
            extra = list(ex.map(_scan_file, [(p,) for p in v1_extra], chunksize=64))
        sig = pd.concat([sig, pd.DataFrame(extra).set_index("path")[["thumb", "color", "md5"]]])
    v1_split = dict(zip(srcs, v1["split"]))
    v1_cat = dict(zip(srcs, v1["category"]))

    # ---------- 3. candidate filtering ----------
    label_by_md5 = df.groupby("md5")["category"].nunique()
    conflicting_md5 = set(label_by_md5[label_by_md5 > 1].index)
    v1_md5 = set(v1["md5"])
    cand = df[~df["path"].isin(v1_paths) & ~df["md5"].isin(v1_md5) & ~df["md5"].isin(conflicting_md5)]
    cand = cand.drop_duplicates("md5").reset_index(drop=True)
    stats = {"candidates_after_exact_dedupe": int(len(cand))}

    ct, cc = np.stack(cand["thumb"].to_list()), np.stack(cand["color"].to_list())
    held = [p for p in srcs if v1_split[p] != "train"]
    ht = np.stack([sig.loc[p, "thumb"] for p in held])
    hc = np.stack([sig.loc[p, "color"] for p in held])
    near_heldout = _max_sim_to(ct, cc, ht, hc, args.group_threshold)
    stats["dropped_near_duplicate_of_val_test_ood"] = int(near_heldout.sum())

    tr = [p for p in srcs if v1_split[p] == "train"]
    tt = np.stack([sig.loc[p, "thumb"] for p in tr])
    tc = np.stack([sig.loc[p, "color"] for p in tr])
    near_train_copy = _max_sim_to(ct, cc, tt, tc, args.collapse_threshold)
    stats["dropped_near_copy_of_v1_train"] = int((near_train_copy & ~near_heldout).sum())

    # label conflicts against v1 train images of another category
    conflict = np.zeros(len(cand), dtype=bool)
    tr_cat = np.array([v1_cat[p] for p in tr])
    for c in CATEGORIES:
        m_c = (cand["category"] == c).to_numpy()
        other = tr_cat != c
        if m_c.any() and other.any():
            conflict[m_c] = _max_sim_to(ct[m_c], cc[m_c], tt[other], tc[other], args.group_threshold)
    keep = ~(near_heldout | near_train_copy | conflict)
    stats["dropped_label_conflict_with_v1_train"] = int((conflict & ~near_heldout & ~near_train_copy).sum())
    cand = cand[keep].reset_index(drop=True)
    ct, cc = ct[keep], cc[keep]

    # near-duplicates among candidates: drop conflicting groups, collapse near-copies
    uf_g, uf_d = UnionFind(len(cand)), UnionFind(len(cand))
    for a, b, s in near_duplicate_pairs(ct, cc, args.group_threshold):
        uf_g.union(a, b)
        if s >= args.collapse_threshold:
            uf_d.union(a, b)
    cand["cgroup"] = [uf_g.find(i) for i in range(len(cand))]
    cand["cdup"] = [uf_d.find(i) for i in range(len(cand))]
    nun = cand.groupby("cgroup")["category"].nunique()
    bad_groups = set(nun[nun > 1].index)
    stats["dropped_label_conflict_among_candidates"] = int(cand["cgroup"].isin(bad_groups).sum())
    cand = cand[~cand["cgroup"].isin(bad_groups)]
    before = len(cand)
    cand = cand.drop_duplicates("cdup").reset_index(drop=True)
    stats["collapsed_near_copies_among_candidates"] = int(before - len(cand))
    stats["eligible_extra_images"] = int(len(cand))
    print(f"[filter] {json.dumps(stats)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 4. top up each category's training split ----------
    parts, added = [], {}
    v1_train_counts = v1[v1["split"] == "train"].groupby("category").size().to_dict()
    for c in CATEGORIES:
        need = max(0, args.train_per_category - int(v1_train_counts.get(c, 0)))
        cdf = cand[cand["category"] == c]
        quota = allocate_quota(cdf.groupby("stratum").size().to_dict(), need)
        for stratum, q in quota.items():
            if q:
                parts.append(cdf[cdf["stratum"] == stratum].sample(n=q, random_state=args.seed))
        added[c] = int(sum(quota.values()))
    extra = pd.concat(parts).reset_index(drop=True) if parts else cand.iloc[:0]
    extra = extra.assign(split="train", uid=[f"x{i:05d}" for i in range(len(extra))])
    extra["file"] = [f"images/train/{r.category}/{r.uid}.jpg" for r in extra.itertuples()]
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_write_processed, [(r.path, str(out / r.file)) for r in extra.itertuples()], chunksize=64))

    extra_rows = extra.assign(source_path=[os.path.relpath(p, raw) for p in extra["path"]],
                              dhash=[f"{int(x):016x}" for x in extra["dhash"]], group=-1)[v1.columns]
    manifest = pd.concat([v1, extra_rows]).reset_index(drop=True)
    manifest.to_csv(out / "manifest.csv", index=False)
    manifest.to_csv(REPO_ML / "datasets/manifests/manifest_v1_1.csv", index=False)

    report = {
        "dataset_version": "v1.1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": "v1 (validation/test/OOD unchanged)",
        "params": {k: v for k, v in vars(args).items() if k not in ("raw_dir", "out_dir", "v1_manifest", "workers")},
        "filtering": stats,
        "added_train_per_category": added,
        "train_per_category": manifest[manifest["split"] == "train"].groupby("category").size()
                                      .reindex(CATEGORIES).fillna(0).astype(int).to_dict(),
        "split_totals": manifest.groupby("split").size().to_dict(),
        "added_per_dataset": extra.groupby("dataset").size().to_dict(),
    }
    json.dump(report, open(REPO_ML / "reports/dataset_report_v1_1.json", "w"), indent=2)
    print(json.dumps(report, indent=1))
    print(f"[done] {len(manifest)} rows  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
