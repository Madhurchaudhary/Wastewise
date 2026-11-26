"""Compare evaluated experiments, select the deployment model and export it.

    python -m ml.src.select_model

Selection rule: highest **validation** macro-F1 (the test split is not used
for selection). Ties within 0.5 percentage points are broken by lower CPU
latency, which favours the lightweight model for deployment.

Outputs
  ml/reports/model_comparison.{json,md}, ml/reports/training_curves.png
  ml/reports/{classification_report.json, confusion_matrix.png, training_history.json}  (selected model)
  ml/models/best_model.pth                         PyTorch checkpoint (+ temperature/threshold)
  backend/model_artifacts/wastewise.onnx          ONNX graph: image -> (features, category_logits, item_logits)
  backend/model_artifacts/gradcam_head.npz        head weights for exact numpy Grad-CAM
  backend/model_artifacts/model_card.json         metadata consumed by the API
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .evaluate import load_checkpoint  # noqa: E402
from .taxonomy import DATASETS  # noqa: E402

REPO_ML = Path(__file__).resolve().parents[1]
REPO = REPO_ML.parent
ARTIFACTS = REPO / "backend/model_artifacts"
ITEM_DISPLAY_THRESHOLD = 0.6
MODEL_NAME = "wastewise-classifier"


class ExportWrapper(nn.Module):
    """Expose the last conv feature map alongside the logits (needed for Grad-CAM)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = self.m.backbone(x)
        h = self.m.neck(self.m.pool(feats))
        return feats, self.m.category_head(h), self.m.item_head(h)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(model, path: Path) -> None:
    wrapper = ExportWrapper(model).eval()
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(wrapper, (x,), str(path), input_names=["image"],
                      output_names=["features", "category_logits", "item_logits"],
                      dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"},
                                    "category_logits": {0: "batch"}, "item_logits": {0: "batch"}},
                      opset_version=17, dynamo=False)


def onnx_latency(path: Path, runs: int = 40) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(5):
        sess.run(None, {"image": x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"image": x})
        ts.append((time.perf_counter() - t) * 1000)
    return {"single_image_ms_median": float(np.median(ts)), "single_image_ms_p95": float(np.percentile(ts, 95))}


def plot_curves(evals: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for exp, _ in evals.items():
        hist = json.load(open(REPO_ML / "models/experiments" / exp / "training_history.json"))["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=f"{exp} train")
        axes[0].plot(ep, [h["val_loss"] for h in hist], "--", label=f"{exp} val")
        axes[1].plot(ep, [h["val_f1_macro"] for h in hist], label=exp)
        axes[2].plot(ep, [h["lr"] for h in hist], label=exp)
    axes[0].set_title("loss")
    axes[1].set_title("validation macro-F1")
    axes[2].set_title("learning rate")
    axes[2].set_yscale("log")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    rep = REPO_ML / "reports"
    exps = args.experiments or sorted(p.parent.name for p in rep.glob("*/evaluation.json"))
    evals = {e: json.load(open(rep / e / "evaluation.json")) for e in exps}
    if not evals:
        raise SystemExit("no evaluated experiments found")

    best_f1 = max(e["validation"]["f1_macro"] for e in evals.values())
    contenders = [k for k, e in evals.items() if best_f1 - e["validation"]["f1_macro"] <= 0.005]
    selected = min(contenders, key=lambda k: evals[k]["latency_pytorch_cpu"]["single_image_ms_median"])
    sel = evals[selected]
    print(f"[select] contenders={contenders} -> {selected}")

    rows = []
    for k, e in evals.items():
        rows.append({
            "experiment": k, "architecture": e["arch"], "params_m": round(e["params"] / 1e6, 2),
            "size_mb": e["checkpoint_size_mb"], "epochs": e["training"]["epochs_run"],
            "train_minutes": e["training"]["training_minutes"],
            "val_acc": round(e["validation"]["accuracy"], 4), "val_f1_macro": round(e["validation"]["f1_macro"], 4),
            "test_acc": round(e["test"]["accuracy"], 4), "test_precision_macro": round(e["test"]["precision_macro"], 4),
            "test_recall_macro": round(e["test"]["recall_macro"], 4), "test_f1_macro": round(e["test"]["f1_macro"], 4),
            "test_f1_weighted": round(e["test"]["f1_weighted"], 4),
            "test_roc_auc": round(e["test"]["roc_auc_ovr_macro"] or 0, 4),
            "hazardous_recall": round(e["safety_classes"]["hazardous"]["recall"], 4),
            "e_waste_recall": round(e["safety_classes"]["e_waste"]["recall"], 4),
            "threshold": e["confidence_threshold"], "coverage": round(e["test_selective"]["coverage"], 4),
            "acc_on_accepted": round(e["test_selective"]["accuracy_on_accepted"] or 0, 4),
            "ood_rejection": round(e["ood"]["rejection_rate"], 4),
            "latency_ms_cpu": round(e["latency_pytorch_cpu"]["single_image_ms_median"], 1),
            "selected": k == selected,
        })
    json.dump({"selection_rule": "max validation macro-F1; ties (<=0.5pp) -> lowest CPU latency",
               "selected": selected, "models": rows}, open(rep / "model_comparison.json", "w"), indent=2)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (rep / "model_comparison.md").write_text(
        "# Model comparison\n\nSelection rule: highest validation macro-F1; ties within 0.5 pp broken by lower CPU latency.\n"
        "Test metrics are reported for information only and were not used for selection.\n\n" + "\n".join(md) + "\n")
    plot_curves(evals, rep / "training_curves.png")

    # ---- spec-mandated top-level reports for the selected model ----
    shutil.copy(rep / selected / "confusion_matrix.png", rep / "confusion_matrix.png")
    json.dump({"model": selected, **sel["test"]["per_class"],
               "accuracy": sel["test"]["accuracy"], "f1_macro": sel["test"]["f1_macro"],
               "f1_weighted": sel["test"]["f1_weighted"]}, open(rep / "classification_report.json", "w"), indent=2)
    shutil.copy(REPO_ML / "models/experiments" / selected / "training_history.json", rep / "training_history.json")

    # ---- deployment checkpoint ----
    model, ckpt = load_checkpoint(str(REPO_ML / "models/experiments" / selected / "best_model.pth"))
    version = f"{args.version}-{sel['arch']}"
    ckpt.update({"temperature": sel["temperature"], "confidence_threshold": sel["confidence_threshold"],
                 "model_name": MODEL_NAME, "model_version": version, "experiment_id": selected})
    torch.save(ckpt, REPO_ML / "models/best_model.pth")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    onnx_path = ARTIFACTS / "wastewise.onnx"
    export_onnx(model, onnx_path)
    np.savez(ARTIFACTS / "gradcam_head.npz",
             w1=model.neck[1].weight.detach().numpy(), b1=model.neck[1].bias.detach().numpy(),
             w2=model.category_head.weight.detach().numpy(), b2=model.category_head.bias.detach().numpy())
    ort_lat = onnx_latency(onnx_path)

    card = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "architecture": sel["arch"],
        "experiment_id": selected,
        "dataset_version": sel["dataset_version"],
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": "onnxruntime (CPU)",
        "onnx_sha256": sha256(onnx_path),
        "gradcam_head_sha256": sha256(ARTIFACTS / "gradcam_head.npz"),
        "categories": ckpt["categories"],
        "items": ckpt["items"],
        "item_to_category": ckpt["item_to_category"],
        "preprocessing": ckpt["preprocessing"],
        "temperature": sel["temperature"],
        "confidence_threshold": sel["confidence_threshold"],
        "confidence_threshold_from_validation_rule": sel.get("threshold_from_rule", sel["confidence_threshold"]),
        "confidence_threshold_override": sel.get("threshold_override"),
        "item_display_threshold": ITEM_DISPLAY_THRESHOLD,
        "metrics": {
            "test_accuracy": sel["test"]["accuracy"], "test_precision_macro": sel["test"]["precision_macro"],
            "test_recall_macro": sel["test"]["recall_macro"], "test_f1_macro": sel["test"]["f1_macro"],
            "test_f1_weighted": sel["test"]["f1_weighted"], "test_roc_auc_ovr_macro": sel["test"]["roc_auc_ovr_macro"],
            "validation_f1_macro": sel["validation"]["f1_macro"],
            "test_coverage_at_threshold": sel["test_selective"]["coverage"],
            "test_accuracy_on_accepted": sel["test_selective"]["accuracy_on_accepted"],
            "ood_rejection_rate": sel["ood"]["rejection_rate"],
            "test_ece_uncalibrated": sel["calibration"]["test_ece_before"],
            "test_ece_calibrated": sel["calibration"]["test_ece_after"],
            "per_class_recall": {c: sel["test"]["per_class"][c]["recall"] for c in ckpt["categories"]},
            "test_images": int(sum(sel["test"]["per_class"][c]["support"] for c in ckpt["categories"])),
        },
        "latency": {"pytorch_cpu_ms": sel["latency_pytorch_cpu"]["single_image_ms_median"],
                    "onnxruntime_cpu_ms": ort_lat["single_image_ms_median"],
                    "measured_on": "2 vCPU sandbox (no GPU)"},
        "training_data": [{"name": v["name"], "url": v["url"], "license": v["license"]} for v in DATASETS.values()],
        "limitations": [
            "Hazardous class is trained only on aerosol-can images (285 unique). Paint tins, chemical bottles and other hazardous items are NOT represented and may be misclassified.",
            "E-waste is represented by batteries and 8 device types (phones, laptops, keyboards, mice, TVs, cameras, microwaves, smartwatches); chargers, cables and circuit boards are not in the training data.",
            "Most training images show a single, centred object, often on a clean background. Cluttered scenes and mixed waste reduce reliability.",
            "The model assigns ONE label per photo. With several items in the frame it usually labels the most prominent one, often with high confidence, and does not warn that other items were ignored.",
            "Confidence scores are temperature-scaled softmax probabilities, not guarantees. A confident prediction can still be wrong.",
            "Images outside the 7 categories (textiles, shoes, mixed trash) are only partially rejected by the confidence threshold.",
        ],
    }
    json.dump(card, open(ARTIFACTS / "model_card.json", "w"), indent=2)
    print(json.dumps({"selected": selected, "version": version, "onnx_latency_ms": ort_lat}, indent=1))


if __name__ == "__main__":
    main()
