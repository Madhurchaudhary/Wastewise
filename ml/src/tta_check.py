"""Does horizontal-flip test-time augmentation help? Decided on the VALIDATION split only.

    python -m ml.src.tta_check --checkpoint ml/models/experiments/<exp>/best_model.pth

Prints validation accuracy / macro-F1 with and without flip-averaged logits. The test split is not
read, so the result can be used to choose the inference mode without touching test data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .augmentation import eval_transforms
from .data_loader import WasteDataset, load_manifest
from .evaluate import load_checkpoint

REPO_ML = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("WASTEWISE_DATA_DIR", str(REPO_ML / "datasets/processed")))
    args = ap.parse_args()
    model, _ = load_checkpoint(args.checkpoint)
    ds = WasteDataset(args.data_dir, load_manifest(args.data_dir), "validation", eval_transforms())
    plain, flip, ys = [], [], []
    for x, y, _ in DataLoader(ds, batch_size=64, num_workers=2):
        a = model(x)[0]
        b = model(torch.flip(x, dims=[3]))[0]
        plain.append(a)
        flip.append((a + b) / 2)
        ys.append(y)
    y = torch.cat(ys).numpy()
    out = {}
    for name, z in (("plain", torch.cat(plain)), ("hflip_tta", torch.cat(flip))):
        p = z.argmax(1).numpy()
        out[name] = {"val_acc": round(accuracy_score(y, p), 4), "val_f1_macro": round(f1_score(y, p, average="macro"), 4)}
    print(json.dumps({"checkpoint": args.checkpoint, **out}, indent=1))


if __name__ == "__main__":
    main()
