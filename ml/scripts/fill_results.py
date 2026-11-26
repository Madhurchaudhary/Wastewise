"""Write measured results into README.md and docs/ml_documentation.md.

    python ml/scripts/fill_results.py

Reads only generated artefacts (ml/reports/model_comparison.json,
ml/reports/<exp>/evaluation.json, backend/model_artifacts/model_card.json) and
replaces the text between the RESULTS markers. Nothing is typed in by hand, so
the docs cannot drift from the evaluation output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REP = REPO / "ml/reports"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f} %"


def comparison_table(cmp: dict) -> str:
    head = ("| Experiment | Params | Epochs | Train min | Val macro-F1 | Test acc | Test macro-F1 | "
            "ROC-AUC | Hazardous recall | Coverage @ thr | Acc on accepted | OOD rejected | CPU ms |")
    rows = [head, "|" + "---|" * 13]
    for r in cmp["models"]:
        name = f"**{r['experiment']}** (selected)" if r["selected"] else r["experiment"]
        rows.append(f"| {name} | {r['params_m']} M | {r['epochs']} | {r['train_minutes']:.0f} | {r['val_f1_macro']:.4f} | "
                    f"{r['test_acc']:.4f} | {r['test_f1_macro']:.4f} | {r['test_roc_auc']:.4f} | {r['hazardous_recall']:.3f} | "
                    f"{pct(r['coverage'])} | {pct(r['acc_on_accepted'])} | {pct(r['ood_rejection'])} | {r['latency_ms_cpu']:.0f} |")
    for r in cmp["models"]:
        hist = REPO / "ml/models/experiments" / r["experiment"] / "training_history.json"
        cfg = json.load(open(hist))["config"] if hist.exists() else {}
        if cfg.get("init_checkpoint"):
            rows.append(f"\n_{r['experiment']}: warm-started from the previously deployed model and fine-tuned on dataset "
                        f"{cfg.get('dataset_version')} (same validation/test/OOD splits, larger training split). Its "
                        f"{r['train_minutes']:.0f} training minutes come on top of the starting model's own training._")
    for r in cmp["models"]:
        if r["coverage"] < 0.5:
            rows.append(f"\n_{r['experiment']}: its high OOD-rejection figure is not a strength. The model is unconfident on almost "
                        f"everything (only {pct(r['coverage'])} of in-scope test images pass its threshold of {r['threshold']})._")
    return "\n".join(rows)


def selected_details(e: dict, card: dict) -> str:
    cats = card["categories"]
    pc = e["test"]["per_class"]
    lines = ["| Category | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in cats:
        if c in pc and isinstance(pc[c], dict):
            lines.append(f"| {c} | {pc[c]['precision']:.3f} | {pc[c]['recall']:.3f} | {pc[c]['f1-score']:.3f} | {int(pc[c]['support'])} |")
    cal = e["calibration"]
    sel = e["test_selective"]
    conf = ", ".join(f"{t['true']}→{t['pred']} ({t['count']})" for t in e["top_confusions"][:5])
    dom = ", ".join(f"{k} {v['mean']:.3f} (n={v['size']})" for k, v in e["domain_accuracy_test"].items())
    return "\n".join([
        f"**Selected model: `{e['experiment_id']}`** ({e['arch']}, model version `{card['model_version']}`).",
        "",
        f"- **Test set** ({card['metrics']['test_images']} images): accuracy {e['test']['accuracy']:.4f}, "
        f"macro-F1 {e['test']['f1_macro']:.4f}, weighted F1 {e['test']['f1_weighted']:.4f}, "
        f"macro ROC-AUC {e['test']['roc_auc_ovr_macro']:.4f}.",
        f"- **Calibration:** temperature {e['temperature']:.4f}; test ECE {cal['test_ece_before']:.4f} → {cal['test_ece_after']:.4f}.",
        f"- **Rejection:** threshold {e['confidence_threshold']:.2f} accepts {pct(sel['coverage'])} of test images, "
        f"with {pct(sel['accuracy_on_accepted'])} accuracy on those. "
        f"{pct(e['ood']['rejection_rate'])} of {e['ood']['n']} out-of-scope probe images are rejected.",
        *([f"- **Threshold policy:** the validation rule gave {e['threshold_from_rule']:.2f}; the deployed threshold is "
           f"{e['confidence_threshold']:.2f}. Reason: {e['threshold_override']['reason']}"] if e.get("threshold_override") else []),
        f"- **Top confusions:** {conf}.",
        f"- **Accuracy by source dataset:** {dom}.",
        f"- **Latency (2 vCPU, no GPU):** ONNX Runtime {card['latency']['onnxruntime_cpu_ms']:.0f} ms, "
        f"PyTorch {card['latency']['pytorch_cpu_ms']:.0f} ms median per image.",
        "",
        *lines,
    ])


def replace(path: Path, start: str, end: str, body: str) -> None:
    s = path.read_text()
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pat.search(s):
        raise SystemExit(f"markers not found in {path}")
    path.write_text(pat.sub(lambda _: f"{start}\n{body}\n{end}", s))
    print("updated", path.relative_to(REPO))


def main() -> None:
    cmp = json.load(open(REP / "model_comparison.json"))
    e = json.load(open(REP / cmp["selected"] / "evaluation.json"))
    card = json.load(open(REPO / "backend/model_artifacts/model_card.json"))
    table = comparison_table(cmp)
    note = ("Selection uses validation macro-F1 only; test numbers are reported for information. "
            "All figures are produced by `ml/src/evaluate.py` and `ml/src/select_model.py`. "
            "See `ml/reports/` for confusion matrices, selective-prediction curves and misclassified examples.")
    replace(REPO / "README.md", "<!-- README-RESULTS:START -->", "<!-- README-RESULTS:END -->",
            f"{table}\n\n{selected_details(e, card)}\n\n{note}")
    replace(REPO / "docs/ml_documentation.md", "<!-- RESULTS:START -->", "<!-- RESULTS:END -->",
            f"{table}\n\n![training curves](../ml/reports/training_curves.png)\n\n{selected_details(e, card)}\n\n"
            f"![confusion matrix](../ml/reports/{cmp['selected']}/confusion_matrix.png)\n\n{note}")


if __name__ == "__main__":
    main()
