"""
Evaluation Suite for the 6-Class Skin Disease Classifier

Computes per-class Precision / Recall / F1, Macro-F1, Top-3 accuracy,
Expected Calibration Error and the confusion matrix, then writes machine- and
human-readable reports. Optionally performs a Fitzpatrick skin-type subgroup
fairness audit (safety priority: suspicious_lesion recall must stay high).

Usage:
    python src/eval.py --checkpoint models/best_model.pt \
        --test_dir data/test --output_dir reports \
        --metadata_csv data/metadata.csv --temperature backend/models/temperature.json
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model import build_model
from dataset import SkinDiseaseDataset, CLASS_NAMES, IDX_TO_CLASS
from calibrate import collect_logits, apply_temperature, compute_ece, load_temperature

SUSPICIOUS_INDEX = CLASS_NAMES.index("suspicious_lesion")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Fraction of samples whose true class is within the top-k predictions."""
    if len(labels) == 0:
        return 0.0
    k = min(k, probabilities.shape[1])
    topk = np.argsort(probabilities, axis=1)[:, -k:]
    hits = np.any(topk == labels.reshape(-1, 1), axis=1)
    return float(np.mean(hits))


def compute_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Aggregate the full clinical metric suite from calibrated probabilities."""
    predictions = np.argmax(probabilities, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    cm = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(labels, predictions, average="weighted", zero_division=0)), 4),
        "top3_accuracy": round(top_k_accuracy(probabilities, labels, k=3), 4),
        "ece": round(float(compute_ece(probabilities, labels, n_bins=n_bins)), 4),
        "suspicious_lesion_recall": round(float(recall[SUSPICIOUS_INDEX]), 4),
        "per_class": per_class,
        "num_samples": int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def confusion_matrix_to_csv(cm: np.ndarray, path: str) -> None:
    """Write the confusion matrix as a labelled CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = ",".join(["actual_predicted"] + CLASS_NAMES)
    lines = [header]
    for i, name in enumerate(CLASS_NAMES):
        row = [name] + [str(int(v)) for v in cm[i]]
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def maybe_plot_confusion(cm: np.ndarray, path: str) -> bool:
    """Save a confusion-matrix heatmap if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def save_reports(
    metrics: Dict[str, Any],
    cm: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
    subgroup: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """Write JSON metrics, CSV confusion matrix, and a text classification report."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    payload = dict(metrics)
    payload["confusion_matrix"] = cm.tolist()
    if subgroup is not None:
        payload["subgroup_fairness"] = subgroup
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    paths["metrics"] = metrics_path

    cm_csv = os.path.join(output_dir, "confusion_matrix.csv")
    confusion_matrix_to_csv(cm, cm_csv)
    paths["confusion_matrix_csv"] = cm_csv

    report_txt = os.path.join(output_dir, "classification_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(labels, predictions, target_names=CLASS_NAMES, zero_division=0))
    paths["classification_report"] = report_txt

    cm_png = os.path.join(output_dir, "confusion_matrix.png")
    if maybe_plot_confusion(cm, cm_png):
        paths["confusion_matrix_png"] = cm_png

    return paths


# ---------------------------------------------------------------------------
# Fitzpatrick subgroup fairness audit
# ---------------------------------------------------------------------------
def _fitzpatrick_bucket(fst: int) -> str:
    if fst <= 2:
        return "FST_I_II"
    if fst <= 4:
        return "FST_III_IV"
    return "FST_V_VI"


def subgroup_audit(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_fst: List[Optional[int]]
) -> Dict[str, Any]:
    """Report suspicious_lesion recall and overall metrics per Fitzpatrick bucket."""
    predictions = np.argmax(probabilities, axis=1)
    buckets: Dict[str, Dict[str, Any]] = {}

    for bucket in ("FST_I_II", "FST_III_IV", "FST_V_VI"):
        idx = [i for i, fst in enumerate(sample_fst)
               if fst is not None and _fitzpatrick_bucket(int(fst)) == bucket]
        if not idx:
            continue
        idx_arr = np.array(idx)
        b_labels = labels[idx_arr]
        b_preds = predictions[idx_arr]

        susp_mask = b_labels == SUSPICIOUS_INDEX
        susp_recall = float(np.mean(b_preds[susp_mask] == SUSPICIOUS_INDEX)) if susp_mask.any() else None

        buckets[bucket] = {
            "num_samples": int(len(idx)),
            "accuracy": round(float(accuracy_score(b_labels, b_preds)), 4),
            "macro_f1": round(float(f1_score(b_labels, b_preds, average="macro", zero_division=0)), 4),
            "suspicious_lesion_recall": (round(susp_recall, 4) if susp_recall is not None else None),
            "suspicious_support": int(susp_mask.sum()),
        }
    return buckets


def load_fst_map(metadata_csv: str) -> Dict[str, int]:
    """Map image filename -> Fitzpatrick skin type from a metadata CSV."""
    import csv as _csv
    fst_map: Dict[str, int] = {}
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        fields = reader.fieldnames or []
        fst_col = next((c for c in ("fitzpatrick", "fitzpatrick_skin_type", "fst", "skin_type") if c in fields), None)
        name_col = next((c for c in ("image", "filename", "img", "path", "image_id") if c in fields), None)
        if fst_col is None or name_col is None:
            return {}
        for row in reader:
            raw = str(row.get(fst_col, "")).strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                continue
            fst_map[os.path.basename(str(row[name_col]))] = int(digits[0])
    return fst_map


def build_test_loader(test_dir: str, batch_size: int) -> Tuple[DataLoader, SkinDiseaseDataset]:
    dataset = SkinDiseaseDataset(data_dir=test_dir, is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate the 6-class skin classifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained .pt checkpoint")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Test images folder")
    parser.add_argument("--output_dir", type=str, default="reports", help="Report output folder")
    parser.add_argument("--temperature", type=str,
                        default=os.path.join("backend", "models", "temperature.json"),
                        help="Calibration JSON (optional)")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Optional CSV with filename + fitzpatrick for subgroup audit")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[Eval] Device: {device}")

    if not os.path.isdir(args.test_dir):
        print(f"[Eval] Test dir '{args.test_dir}' not found. Nothing to evaluate.")
        return

    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=(args.checkpoint is None),
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    loader, dataset = build_test_loader(args.test_dir, args.batch_size)
    logits, labels = collect_logits(model, loader, device)
    if len(labels) == 0:
        print("[Eval] No samples found; aborting.")
        return

    temperature = load_temperature(args.temperature) or 1.0
    print(f"[Eval] Applying temperature T={temperature}")
    probabilities = apply_temperature(logits, temperature)

    metrics = compute_metrics(probabilities, labels, n_bins=args.n_bins)
    predictions = np.argmax(probabilities, axis=1)
    cm = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))

    subgroup = None
    if args.metadata_csv and os.path.exists(args.metadata_csv):
        fst_map = load_fst_map(args.metadata_csv)
        if fst_map:
            sample_fst = [fst_map.get(os.path.basename(path)) for path, _ in dataset.samples]
            subgroup = subgroup_audit(probabilities, labels, sample_fst)
            print(f"[Eval] Subgroup audit: {json.dumps(subgroup, indent=2)}")

    paths = save_reports(metrics, cm, labels, predictions, args.output_dir, subgroup=subgroup)

    print("\n===== Evaluation Summary =====")
    print(f"  Accuracy        : {metrics['accuracy']}")
    print(f"  Macro-F1        : {metrics['macro_f1']}")
    print(f"  Top-3 Accuracy  : {metrics['top3_accuracy']}")
    print(f"  ECE             : {metrics['ece']}")
    print(f"  Suspicious Rec. : {metrics['suspicious_lesion_recall']}")
    print(f"  Samples         : {metrics['num_samples']}")
    print(f"  Reports written : {paths}")


if __name__ == "__main__":
    main()