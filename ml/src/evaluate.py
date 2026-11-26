"""Evaluate a trained checkpoint.

    python -m ml.src.evaluate --experiment exp02_mobilenet_v3_large

Produces ``ml/reports/<experiment>/``:
  evaluation.json        all metrics (validation + test + OOD + latency + calibration)
  confusion_matrix.png   test confusion matrix (counts + row-normalised)
  selective_curve.png    coverage vs accuracy as the confidence threshold varies
  misclassified.jpg      grid of test errors (error analysis)

Protocol
--------
* Temperature scaling and the confidence threshold are fitted on the
  **validation** split only.
* The **test** split is used once, for reporting. Model selection
  (``select_model.py``) uses validation macro-F1, never test metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .augmentation import eval_transforms  # noqa: E402
from .data_loader import WasteDataset, load_manifest  # noqa: E402
from .metrics import (classification_metrics, expected_calibration_error, fit_temperature,  # noqa: E402
                      selective_curve)
from .model import WasteClassifier, count_parameters  # noqa: E402
from .preprocessing import load_rgb  # noqa: E402
from .taxonomy import CATEGORIES, ITEMS  # noqa: E402

REPO_ML = Path(__file__).resolve().parents[1]
TARGET_SELECTIVE_ACC = 0.95   # accepted predictions should be >= 95% correct on validation
THRESHOLD_BOUNDS = (0.50, 0.90)
DEFAULT_THRESHOLD = 0.65


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WasteClassifier(ckpt["arch"], len(ckpt["categories"]), len(ckpt["items"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_logits(model, loader):
    cl, il, yc, yi = [], [], [], []
    for x, c, i in loader:
        a, b = model(x)
        cl.append(a)
        il.append(b)
        yc.append(c)
        yi.append(i)
    return torch.cat(cl), torch.cat(il), torch.cat(yc), torch.cat(yi)


def choose_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, list[dict]]:
    grid = np.round(np.arange(0.30, 0.991, 0.01), 2)
    curve = selective_curve(probs, y, grid)
    lo, hi = THRESHOLD_BOUNDS
    for row in curve:
        if lo <= row["threshold"] <= hi and row["selective_accuracy"] is not None \
                and row["selective_accuracy"] >= TARGET_SELECTIVE_ACC:
            return row["threshold"], curve
    return DEFAULT_THRESHOLD, curve


@torch.no_grad()
def measure_latency(model, runs: int = 40) -> dict:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    x1 = torch.randn(1, 3, 224, 224)
    xb = torch.randn(16, 3, 224, 224)
    for _ in range(5):
        model(x1)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x1)
        times.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    for _ in range(3):
        model(xb)
    thr = 48 / (time.perf_counter() - t)
    return {"single_image_ms_median": float(np.median(times)), "single_image_ms_p95": float(np.percentile(times, 95)),
            "batch16_images_per_s": float(thr), "device": "cpu", "threads": torch.get_num_threads()}


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, data, ttl, fmt in ((axes[0], cm, "counts", "d"), (axes[1], cmn, "row-normalised (recall)", ".2f")):
        im = ax.imshow(data, cmap="Greens")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(ttl)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                        color="white" if (v > data.max() * 0.6) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_selective(curve: list[dict], ood_conf: np.ndarray, threshold: float, path: Path) -> None:
    th = [r["threshold"] for r in curve]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(th, [r["coverage"] for r in curve], label="validation coverage")
    ax.plot(th, [r["selective_accuracy"] or np.nan for r in curve], label="validation accuracy of accepted")
    ax.plot(th, [(ood_conf < t).mean() for t in th], label="OOD rejection rate", linestyle="--")
    ax.axvline(threshold, color="k", linewidth=0.8)
    ax.set_xlabel("confidence threshold (temperature-scaled max softmax)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title(f"Selective prediction (chosen threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def misclassified_grid(data_dir: str, df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, conf: np.ndarray,
                       path: Path, n: int = 24) -> list[dict]:
    wrong = np.nonzero(y != pred)[0]
    # prioritise hazardous / e-waste errors (safety-relevant), then most confident errors
    pri = sorted(wrong, key=lambda i: (CATEGORIES[y[i]] not in ("hazardous", "e_waste")
                                       and CATEGORIES[pred[i]] not in ("hazardous", "e_waste"), -conf[i]))
    pri = pri[:n]
    W = 160
    cols = 6
    rows = max(1, (len(pri) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * W, rows * (W + 28)), "white")
    d = ImageDraw.Draw(sheet)
    out = []
    for k, i in enumerate(pri):
        r = df.iloc[i]
        im = load_rgb(os.path.join(data_dir, r["file"]))
        im.thumbnail((W, W))
        x, yy = (k % cols) * W, (k // cols) * (W + 28)
        sheet.paste(im, (x, yy))
        d.text((x + 2, yy + W + 1), f"T:{CATEGORIES[y[i]][:9]}", fill="black")
        d.text((x + 2, yy + W + 13), f"P:{CATEGORIES[pred[i]][:9]} {conf[i]:.2f}", fill="red")
        out.append({"uid": str(r["uid"]), "true": CATEGORIES[y[i]], "pred": CATEGORIES[pred[i]],
                    "confidence": float(conf[i]), "dataset": r["dataset"], "source_path": r["source_path"]})
    sheet.save(path, quality=88)
    return out


def append_experiment_log(row: dict) -> None:
    path = REPO_ML / "reports/experiment_log.csv"
    cols = ["experiment_id", "date", "model_architecture", "dataset_version", "epochs", "batch_size",
            "learning_rate", "optimizer", "training_accuracy", "validation_accuracy", "test_accuracy",
            "precision_macro", "recall_macro", "f1_macro", "model_size_mb", "inference_latency_ms",
            "training_minutes", "notes"]
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["experiment_id"] != row["experiment_id"]]
    rows.append({c: row.get(c, "") for c in cols})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("WASTEWISE_DATA_DIR", str(REPO_ML / "datasets/processed")))
    ap.add_argument("--models-dir", default=str(REPO_ML / "models/experiments"))
    ap.add_argument("--threshold-override", type=float, default=None,
                    help="deployment operating threshold to use instead of the validation rule's value")
    ap.add_argument("--override-reason", default=None, help="required with --threshold-override; stored in the report")
    args = ap.parse_args()
    if args.threshold_override is not None and not args.override_reason:
        raise SystemExit("--threshold-override needs --override-reason")

    exp_dir = Path(args.models_dir) / args.experiment
    ckpt_path = exp_dir / "best_model.pth"
    model, ckpt = load_checkpoint(str(ckpt_path))
    out = REPO_ML / "reports" / args.experiment
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.data_dir)
    tf = eval_transforms()
    sets = {s: WasteDataset(args.data_dir, manifest, s, tf) for s in ("validation", "test", "ood")}
    loaders = {s: DataLoader(d, batch_size=64, num_workers=2) for s, d in sets.items()}
    logits = {s: collect_logits(model, ld) for s, ld in loaders.items()}

    # ---- calibration on validation ----
    v_cl, _, v_y, _ = logits["validation"]
    T = fit_temperature(v_cl, v_y)
    sm = lambda z, t=1.0: torch.softmax(z / t, 1).numpy()  # noqa: E731
    v_raw, v_cal = sm(v_cl), sm(v_cl, T)
    threshold, curve = choose_threshold(v_cal, v_y.numpy())
    rule_threshold = threshold
    if args.threshold_override is not None:
        threshold = round(float(args.threshold_override), 2)

    # ---- test ----
    t_cl, t_il, t_y, t_yi = logits["test"]
    t_raw, t_cal = sm(t_cl), sm(t_cl, T)
    y = t_y.numpy()
    test_m = classification_metrics(y, t_cal, CATEGORIES)
    val_m = classification_metrics(v_y.numpy(), v_cal, CATEGORIES)
    conf = t_cal.max(1)
    pred = t_cal.argmax(1)
    accepted = conf >= threshold

    # item head (only for test images with a specific item label)
    yi = t_yi.numpy()
    mask = yi >= 0
    item_pred = t_il.argmax(1).numpy()
    item_acc = float((item_pred[mask] == yi[mask]).mean()) if mask.any() else None

    # per-source domain breakdown
    tdf = sets["test"].df
    domain = (tdf.assign(correct=(pred == y), sub=tdf["stratum"].str.extract(r"/(default|real_world)$")[0].fillna(""))
              .assign(domain=lambda d: d["dataset"] + d["sub"].map(lambda s: f":{s}" if s else ""))
              .groupby("domain")["correct"].agg(["mean", "size"]).round(4))

    # ---- OOD ----
    o_cl = logits["ood"][0]
    o_cal = sm(o_cl, T)
    o_conf = o_cal.max(1)
    odf = sets["ood"].df
    ood_by_source = (odf.assign(rejected=o_conf < threshold).groupby("stratum")["rejected"].mean().round(3).to_dict())

    cm = np.array(test_m["confusion_matrix"])
    plot_confusion(cm, CATEGORIES, out / "confusion_matrix.png",
                   f"{ckpt['arch']} - test set (n={len(y)}), argmax predictions")
    plot_selective(curve, o_conf, threshold, out / "selective_curve.png")
    errors = misclassified_grid(args.data_dir, tdf, y, pred, conf, out / "misclassified.jpg")

    # confusion pairs (error analysis)
    pairs = []
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": CATEGORIES[i], "pred": CATEGORIES[j], "count": int(cm[i, j])})
    pairs.sort(key=lambda r: -r["count"])
    safety = {}
    for c in ("hazardous", "e_waste"):
        ci = CATEGORIES.index(c)
        safety[c] = {
            "recall": test_m["per_class"][c]["recall"],
            "precision": test_m["per_class"][c]["precision"],
            "support": int(test_m["per_class"][c]["support"]),
            "missed_as": {CATEGORIES[j]: int(cm[ci, j]) for j in range(len(CATEGORIES)) if j != ci and cm[ci, j]},
            "false_alarms_from": {CATEGORIES[i]: int(cm[i, ci]) for i in range(len(CATEGORIES)) if i != ci and cm[i, ci]},
        }

    latency = measure_latency(model)
    size_mb = os.path.getsize(ckpt_path) / 1e6
    hist = json.load(open(exp_dir / "training_history.json"))
    tsum = json.load(open(exp_dir / "train_summary.json")) if (exp_dir / "train_summary.json").exists() else {}

    result = {
        "experiment_id": args.experiment,
        "arch": ckpt["arch"],
        "dataset_version": ckpt["config"].get("dataset_version"),
        "best_epoch": ckpt["best_epoch"],
        "params": count_parameters(model),
        "checkpoint_size_mb": round(size_mb, 2),
        "temperature": T,
        "confidence_threshold": threshold,
        "threshold_rule": f"smallest threshold in [{THRESHOLD_BOUNDS[0]}, {THRESHOLD_BOUNDS[1]}] whose validation "
                          f"selective accuracy >= {TARGET_SELECTIVE_ACC} (fallback {DEFAULT_THRESHOLD})",
        "threshold_from_rule": rule_threshold,
        "threshold_override": None if args.threshold_override is None else
        {"value": threshold, "reason": args.override_reason},
        "validation": {k: val_m[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                             "f1_weighted", "roc_auc_ovr_macro")},
        "test": {k: v for k, v in test_m.items()},
        "test_item_head_accuracy": item_acc,
        "test_item_labelled": int(mask.sum()),
        "test_selective": {"threshold": threshold, "coverage": float(accepted.mean()),
                           "accuracy_on_accepted": float((pred[accepted] == y[accepted]).mean()) if accepted.any() else None,
                           "accuracy_on_rejected": float((pred[~accepted] == y[~accepted]).mean()) if (~accepted).any() else None},
        "calibration": {"val_ece_before": expected_calibration_error(v_raw, v_y.numpy()),
                        "val_ece_after": expected_calibration_error(v_cal, v_y.numpy()),
                        "test_ece_before": expected_calibration_error(t_raw, y),
                        "test_ece_after": expected_calibration_error(t_cal, y)},
        "ood": {"n": int(len(o_conf)), "rejection_rate": float((o_conf < threshold).mean()),
                "mean_confidence": float(o_conf.mean()), "rejection_by_source": ood_by_source},
        "domain_accuracy_test": domain.to_dict(orient="index"),
        "top_confusions": pairs[:12],
        "safety_classes": safety,
        "misclassified_examples": errors,
        "latency_pytorch_cpu": latency,
        "training": {"epochs_run": tsum.get("epochs_run", len(hist["history"])),
                     "training_minutes": tsum.get("training_minutes"),
                     "best_train_acc": hist["best"].get("train_acc")},
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(result, open(out / "evaluation.json", "w"), indent=2)

    cfg = ckpt["config"]
    append_experiment_log({
        "experiment_id": args.experiment, "date": time.strftime("%Y-%m-%d"), "model_architecture": ckpt["arch"],
        "dataset_version": cfg.get("dataset_version"), "epochs": result["training"]["epochs_run"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["lr_finetune"] if ckpt["arch"] == "custom_cnn" else f"{cfg['lr_head']}/{cfg['lr_finetune']}",
        "optimizer": cfg["optimizer"], "training_accuracy": round(result["training"]["best_train_acc"] or 0, 4),
        "validation_accuracy": round(val_m["accuracy"], 4), "test_accuracy": round(test_m["accuracy"], 4),
        "precision_macro": round(test_m["precision_macro"], 4), "recall_macro": round(test_m["recall_macro"], 4),
        "f1_macro": round(test_m["f1_macro"], 4), "model_size_mb": round(size_mb, 2),
        "inference_latency_ms": round(latency["single_image_ms_median"], 1),
        "training_minutes": result["training"]["training_minutes"],
        "notes": f"best_epoch={ckpt['best_epoch']}; T={T:.2f}; threshold={threshold:.2f}; "
                 f"coverage={result['test_selective']['coverage']:.3f}; OOD rejection={result['ood']['rejection_rate']:.3f}",
    })
    print(json.dumps({"experiment": args.experiment, "val_f1": val_m["f1_macro"], "test_acc": test_m["accuracy"],
                      "test_f1": test_m["f1_macro"], "T": T, "threshold": threshold,
                      "coverage": result["test_selective"], "ood": result["ood"]["rejection_rate"],
                      "latency_ms": latency["single_image_ms_median"]}, indent=1))


if __name__ == "__main__":
    main()
