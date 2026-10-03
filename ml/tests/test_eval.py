"""
Tests for ml/src/eval.py (metrics, confusion matrix, subgroup audit).
Deterministic synthetic data; no checkpoint required.
"""

import os
import sys
import json
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import (  # noqa: E402
    CLASS_NAMES,
    compute_metrics,
    top_k_accuracy,
    subgroup_audit,
    save_reports,
)
from sklearn.metrics import confusion_matrix  # noqa: E402

SUSP = CLASS_NAMES.index("suspicious_lesion")


def _onehot_probs(labels, confidence=0.9):
    """Probabilities that put `confidence` on the true label (perfectly correct)."""
    n, c = len(labels), len(CLASS_NAMES)
    probs = np.full((n, c), (1.0 - confidence) / (c - 1))
    probs[np.arange(n), labels] = confidence
    return probs


def test_top_k_accuracy():
    # Correct label always within top-3
    probs = np.array([
        [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],  # top3 = {2,1,3}
    ])
    labels = np.array([3])
    assert top_k_accuracy(probs, labels, k=3) == 1.0

    labels = np.array([5])  # class 5 has lowest prob -> miss
    assert top_k_accuracy(probs, labels, k=3) == 0.0
    print("[PASS] top-k accuracy")


def test_compute_metrics_structure():
    labels = np.array([0, 1, 2, 3, 4, SUSP, SUSP, 0, 1, 5])
    probs = _onehot_probs(labels)

    metrics = compute_metrics(probs, labels)

    for key in ("accuracy", "macro_f1", "top3_accuracy", "ece",
                "suspicious_lesion_recall", "per_class", "num_samples"):
        assert key in metrics, f"Missing metric: {key}"
    assert metrics["accuracy"] == 1.0
    assert metrics["suspicious_lesion_recall"] == 1.0
    assert set(metrics["per_class"].keys()) == set(CLASS_NAMES)
    assert metrics["num_samples"] == 10
    print(f"[PASS] compute_metrics acc={metrics['accuracy']} macro_f1={metrics['macro_f1']} "
          f"susp_recall={metrics['suspicious_lesion_recall']}")


def test_suspicious_recall_penalised_on_miss():
    # suspicious_lesion true but predicted eczema -> recall 0
    labels = np.array([SUSP, 0, 1])
    probs = _onehot_probs(np.array([0, 0, 1]))  # deliberately wrong on first
    metrics = compute_metrics(probs, labels)
    assert metrics["suspicious_lesion_recall"] == 0.0, "Missed suspicious lesion must drop recall"
    print("[PASS] suspicious recall penalised on false negative")


def test_subgroup_audit():
    labels = np.array([SUSP, SUSP, 0, SUSP, 1, SUSP])
    probs = _onehot_probs(labels)
    fst = [1, 2, 3, 5, 4, 6]  # I-II, III-IV, V-VI

    audit = subgroup_audit(probs, labels, fst)
    assert "FST_I_II" in audit and "FST_V_VI" in audit
    for bucket in audit.values():
        assert "accuracy" in bucket and "suspicious_lesion_recall" in bucket
    assert audit["FST_V_VI"]["suspicious_lesion_recall"] == 1.0
    print(f"[PASS] subgroup audit -> {json.dumps(audit)}")


def test_save_reports_writes_files():
    labels = np.array([0, 1, 2, 3, 4, SUSP])
    probs = _onehot_probs(labels)
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    metrics = compute_metrics(probs, labels)

    with tempfile.TemporaryDirectory() as tmp:
        paths = save_reports(metrics, cm, labels, preds, tmp)
        assert os.path.exists(paths["metrics"])
        assert os.path.exists(paths["confusion_matrix_csv"])
        assert os.path.exists(paths["classification_report"])
        with open(paths["metrics"], "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert len(payload["confusion_matrix"]) == len(CLASS_NAMES)
    print("[PASS] save_reports writes metrics/CSV/report")


if __name__ == "__main__":
    test_top_k_accuracy()
    test_compute_metrics_structure()
    test_suspicious_recall_penalised_on_miss()
    test_subgroup_audit()
    test_save_reports_writes_files()
    print("\nAll eval tests passed.")