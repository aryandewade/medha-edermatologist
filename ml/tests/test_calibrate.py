"""
Tests for ml/src/calibrate.py (temperature scaling + ECE).

Deterministic: uses synthetic logits, so no trained checkpoint is required.
"""

import os
import sys
import json
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calibrate import (  # noqa: E402
    softmax,
    compute_ece,
    calibrate_from_logits,
    optimize_temperature,
    apply_temperature,
    save_temperature,
    load_temperature,
)

C = 6


def _overconfident_logits(seed: int = 0, n: int = 800):
    """High-magnitude logits where only ~50% are actually correct."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, C, n)
    pred = labels.copy()
    wrong = rng.random(n) < 0.5
    offsets = rng.integers(1, C, int(wrong.sum()))
    pred[wrong] = (labels[wrong] + offsets) % C

    logits = np.full((n, C), -1.0)
    logits[np.arange(n), pred] = 5.0
    return logits, labels


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0), "Softmax rows must sum to 1"
    print("[PASS] softmax normalisation")


def test_ece_zero_for_perfectly_calibrated():
    # Confidence == accuracy in every bin => ECE ~ 0
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 0])  # accuracy in the 0.9 bin = 2/3... build carefully
    # Construct exact calibration: bin [0.8,0.9] with acc 0.9 is impractical; use
    # identical confidence and matching accuracy via repeated blocks.
    probs = np.tile(np.array([[0.7, 0.3]]), (10, 1))
    labels = np.array([0] * 7 + [1] * 3)  # accuracy 0.7 == confidence 0.7
    ece = compute_ece(probs, labels, n_bins=10)
    assert ece < 1e-6, f"ECE should be ~0 for calibrated probs, got {ece}"
    print("[PASS] ECE ~ 0 for calibrated probabilities")


def test_temperature_scaling_reduces_ece():
    logits, labels = _overconfident_logits()

    ece_before = compute_ece(apply_temperature(logits, 1.0), labels)
    temperature = optimize_temperature(logits, labels)
    ece_after = compute_ece(apply_temperature(logits, temperature), labels)

    print(f"      T={temperature:.3f}  ECE before={ece_before:.4f}  after={ece_after:.4f}")
    assert temperature > 1.0, "Overconfident logits require T > 1 to soften"
    assert ece_after < ece_before, "Calibration must reduce ECE"
    print("[PASS] temperature scaling reduces ECE")


def test_calibrate_from_logits_empty():
    T, before, after = calibrate_from_logits(np.zeros((0, C)), np.zeros((0,), dtype=int))
    assert T == 1.0 and before == 0.0 and after == 0.0
    print("[PASS] empty input handled")


def test_save_and_load_temperature_schema():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "temperature.json")
        save_temperature(path, 1.25, 0.124, 0.038)

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Interface-2 contract keys
        for key in ("temperature", "ece_before", "ece_after", "calibrated_at"):
            assert key in payload, f"Missing required key: {key}"
        assert payload["temperature"] == 1.25
        assert load_temperature(path) == 1.25
        print(f"[PASS] temperature.json schema OK -> {payload}")


if __name__ == "__main__":
    test_softmax_rows_sum_to_one()
    test_ece_zero_for_perfectly_calibrated()
    test_temperature_scaling_reduces_ece()
    test_calibrate_from_logits_empty()
    test_save_and_load_temperature_schema()
    print("\nAll calibrate tests passed.")