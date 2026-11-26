"""Configurable training pipeline.

    python -m ml.src.train --config ml/configs/mobilenet_v3.json

Two-phase schedule for transfer models:
  phase 1 ("warmup")   : backbone frozen, only neck + heads train (lr_head)
  phase 2 ("finetune") : backbone stages >= unfreeze_from train too (lr_finetune, cosine decay)
The custom CNN trains end-to-end in a single phase.

``init_checkpoint`` (optional) warm-starts from an earlier WasteWise checkpoint instead of the
ImageNet weights, e.g. to continue training the v1 model on the larger v1.1 training split.

Early stopping and checkpoint selection use validation macro-F1 on the
category task. The test split is never touched here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .data_loader import IGNORE_INDEX, class_weights, make_loaders
from .model import WasteClassifier, count_parameters
from .preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE
from .taxonomy import CATEGORIES, ITEM_TO_CATEGORY, ITEMS

REPO_ML = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "experiment_id": None,
    "arch": "mobilenet_v3_large",
    "dataset_version": "v1",
    "seed": 42,
    "batch_size": 32,
    "num_workers": 2,
    "image_size": IMAGE_SIZE,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "label_smoothing": 0.05,
    "item_loss_weight": 0.3,
    "use_class_weights": True,
    "pretrained": True,
    "warmup_epochs": 3,
    "lr_head": 1e-3,
    "finetune_epochs": 12,
    "lr_finetune": 3e-4,
    "unfreeze_from": 10,
    "scheduler": "cosine",
    "early_stopping_patience": 4,
    "max_minutes": None,
    "init_checkpoint": None,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, device, crit_cat, crit_item, item_w, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, n, correct = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.set_grad_enabled(train):
        for x, yc, yi in loader:
            x, yc, yi = x.to(device), yc.to(device), yi.to(device)
            lc, li = model(x)
            loss = crit_cat(lc, yc)
            if (yi != IGNORE_INDEX).any():
                loss = loss + item_w * crit_item(li, yi)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            tot_loss += loss.item() * bs
            n += bs
            pred = lc.argmax(1)
            correct += (pred == yc).sum().item()
            y_true.append(yc.cpu())
            y_pred.append(pred.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return {"loss": tot_loss / n, "acc": correct / n,
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("WASTEWISE_DATA_DIR", str(REPO_ML / "datasets/processed")))
    ap.add_argument("--out-dir", default=str(REPO_ML / "models/experiments"))
    ap.add_argument("--pretrained-dir", default=str(REPO_ML / "models/pretrained"))
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(json.load(open(args.config)))
    cfg["experiment_id"] = cfg["experiment_id"] or f"{cfg['arch']}-{time.strftime('%Y%m%d-%H%M')}"
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"[train] {cfg['experiment_id']} on {device}")

    loaders, train_ds = make_loaders(args.data_dir, cfg["batch_size"], cfg["num_workers"], cfg["image_size"])
    warm = cfg.get("init_checkpoint")
    model = WasteClassifier(cfg["arch"], len(CATEGORIES), len(ITEMS), cfg["dropout"],
                            args.pretrained_dir if cfg["pretrained"] and not warm else None).to(device)
    if warm:
        init = torch.load(REPO_ML.parent / warm, map_location="cpu", weights_only=False)
        if init["arch"] != cfg["arch"] or list(init["categories"]) != CATEGORIES or list(init["items"]) != ITEMS:
            raise SystemExit(f"{warm} does not match arch/categories/items of this config")
        model.load_state_dict(init["state_dict"])
        print(f"[train] warm start from {warm} (val_f1={init.get('val_f1_macro')})")

    cw = class_weights(train_ds.category_counts()).to(device) if cfg["use_class_weights"] else None
    crit_cat = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg["label_smoothing"])
    crit_item = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=cfg["label_smoothing"])
    print(f"[train] class weights: {dict(zip(CATEGORIES, [round(float(w), 3) for w in cw]))}" if cw is not None else "")

    out_dir = Path(args.out_dir) / cfg["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = {"f1_macro": -1.0, "epoch": -1}
    patience = 0
    t_start = time.time()
    phases = []
    if cfg["arch"] == "custom_cnn":
        phases.append(("full", cfg["finetune_epochs"], cfg["lr_finetune"], None))
    else:
        if cfg["warmup_epochs"] > 0:
            phases.append(("warmup", cfg["warmup_epochs"], cfg["lr_head"], "freeze"))
        phases.append(("finetune", cfg["finetune_epochs"], cfg["lr_finetune"], cfg["unfreeze_from"]))

    epoch = 0
    stop = False
    for phase, n_epochs, lr, freeze in phases:
        if stop:
            break
        if freeze == "freeze":
            model.freeze_backbone(None)
        elif isinstance(freeze, int):
            model.freeze_backbone(freeze)
        else:
            model.unfreeze_all()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        if cfg["scheduler"] == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, n_epochs), eta_min=lr * 0.02)
        else:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=1)
        n_trainable = sum(p.numel() for p in params)
        print(f"[train] phase={phase} epochs={n_epochs} lr={lr} trainable_params={n_trainable:,}")
        patience = 0  # reset early-stopping patience at phase change
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            cur_lr = opt.param_groups[0]["lr"]
            tr = run_epoch(model, loaders["train"], device, crit_cat, crit_item, cfg["item_loss_weight"], opt)
            va = run_epoch(model, loaders["validation"], device, crit_cat, crit_item, cfg["item_loss_weight"])
            if cfg["scheduler"] == "cosine":
                sched.step()
            else:
                sched.step(va["f1_macro"])
            rec = {"epoch": epoch, "phase": phase, "lr": cur_lr, "train_loss": tr["loss"], "train_acc": tr["acc"],
                   "val_loss": va["loss"], "val_acc": va["acc"], "val_f1_macro": va["f1_macro"],
                   "seconds": round(time.time() - t0, 1)}
            history.append(rec)
            print(f"[epoch {epoch:02d}] {phase} lr={cur_lr:.2e} train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
                  f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} val_f1={va['f1_macro']:.4f} ({rec['seconds']}s)",
                  flush=True)
            if va["f1_macro"] > best["f1_macro"]:
                best = {"f1_macro": va["f1_macro"], "acc": va["acc"], "epoch": epoch,
                        "train_acc": tr["acc"]}
                patience = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "arch": cfg["arch"],
                    "categories": CATEGORIES,
                    "items": ITEMS,
                    "item_to_category": ITEM_TO_CATEGORY,
                    "preprocessing": {"image_size": cfg["image_size"], "resize_size": RESIZE_SIZE,
                                      "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                                      "interpolation": "bilinear", "color": "RGB"},
                    "config": cfg,
                    "best_epoch": epoch,
                    "val_f1_macro": va["f1_macro"],
                    "val_acc": va["acc"],
                }, out_dir / "best_model.pth")
            else:
                patience += 1
            json.dump({"config": cfg, "history": history, "best": best}, open(out_dir / "training_history.json", "w"),
                      indent=2)
            if patience >= cfg["early_stopping_patience"]:
                print(f"[train] early stopping at epoch {epoch}")
                stop = True
                break
            if cfg["max_minutes"] and (time.time() - t_start) / 60 > cfg["max_minutes"]:
                print("[train] time budget reached")
                stop = True
                break

    summary = {"experiment_id": cfg["experiment_id"], "arch": cfg["arch"], "best": best,
               "epochs_run": epoch, "training_minutes": round((time.time() - t_start) / 60, 1),
               "params": count_parameters(model), "device": str(device),
               "checkpoint": str(out_dir / "best_model.pth")}
    json.dump(summary, open(out_dir / "train_summary.json", "w"), indent=2)
    print("[train] done", json.dumps(summary))


if __name__ == "__main__":
    main()
