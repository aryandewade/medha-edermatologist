"""
Post-hoc Temperature Scaling Calibration for the 6-Class Skin Classifier

Deep networks are systematically overconfident. We fit a single scalar
temperature T > 0 on the held-out validation logits so that
    P_i = softmax(z_i / T)
minimises Negative Log-Likelihood (NLL), then report the Expected Calibration
Error (ECE) before and after.

Coordination contract (interface 2): writes backend/models/temperature.json
    {
      "temperature": 1.25,
      "ece_before": 0.124,
      "ece_after": 0.038,
      "calibrated_at": "2026-10-03"
    }
"""

import os
import json
import argparse
from datetime import date
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar

from model import build_model
from dataset import SkinDiseaseDataset, CLASS_NAMES


# ---------------------------------------------------------------------------
# Core probability / calibration maths
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by 1/T and return calibrated probabilities."""
    return softmax(logits / max(float(temperature), 1e-4))


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of the ground-truth class at temperature T."""
    probs = apply_temperature(logits, temperature)
    correct = probs[np.arange(len(labels)), labels]
    correct = np.clip(correct, 1e-12, 1.0)
    return float(-np.mean(np.log(correct)))


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error with equal-width confidence bins.

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Half-open bins (lo, hi]; the final bin includes confidence == 1.0.
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        avg_conf = float(np.mean(confidences[in_bin]))
        avg_acc = float(np.mean(accuracies[in_bin]))
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: Tuple[float, float] = (0.05, 10.0)
) -> float:
    """Find the scalar T minimising NLL via bounded scalar minimisation."""
    result = minimize_scalar(
        lambda t: _nll(logits, labels, t),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success or not np.isfinite(result.x):
        return 1.0
    return float(result.x)


# ---------------------------------------------------------------------------
# Logit collection & persistence
# ---------------------------------------------------------------------------
def build_loader(val_dir: str, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    """Non-augmenting validation loader over class-subfolder structure."""
    dataset = SkinDiseaseDataset(data_dir=val_dir, is_train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a dataset and return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(targets.numpy())

    if not all_logits:
        return np.zeros((0, len(CLASS_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    logits = np.concatenate(all_logits, axis=0).astype(np.float64)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    return logits, labels


def save_temperature(
    output_path: str,
    temperature: float,
    ece_before: float,
    ece_after: float,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Write the calibration artifact in the frozen interface-2 JSON schema."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "temperature": round(float(temperature), 4),
        "ece_before": round(float(ece_before), 4),
        "ece_after": round(float(ece_after), 4),
        "calibrated_at": date.today().isoformat(),
    }
    if extra:
        payload.update(extra)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Calibrate] Saved temperature artifact to {output_path}")


def load_temperature(path: str) -> Optional[float]:
    """Read a stored temperature value, or None if unavailable."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return float(json.load(f)["temperature"])
    except (ValueError, KeyError, OSError) as exc:
        print(f"[Calibrate] Could not read temperature from {path}: {exc}")
    return None


def calibrate_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, float]:
    """
    Fit T and return (temperature, ece_before, ece_after). Deterministic and
    dependency-light so it can be unit-tested without a trained network.
    """
    if len(labels) == 0:
        return 1.0, 0.0, 0.0
    ece_before = compute_ece(apply_temperature(logits, 1.0), labels, n_bins=n_bins)
    temperature = optimize_temperature(logits, labels)
    ece_after = compute_ece(apply_temperature(logits, temperature), labels, n_bins=n_bins)
    return temperature, ece_before, ece_after


def main():
    parser = argparse.ArgumentParser(description="Temperature-scale the 6-class classifier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Validation images folder")
    parser.add_argument("--output", type=str,
                        default=os.path.join("backend", "models", "temperature.json"),
                        help="Output JSON path (interface-2 contract)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bins", type=int, default=10, help="ECE confidence bins")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[Calibrate] Device: {device}")

    if args.checkpoint is None:
        print("[Calibrate] WARNING: no --checkpoint supplied; using ImageNet-pretrained "
              "backbone with an untrained head. Results are placeholders only.")

    model = build_model(
        num_classes=len(CLASS_NAMES),
        pretrained=(args.checkpoint is None),
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if not os.path.isdir(args.val_dir):
        print(f"[Calibrate] Validation dir '{args.val_dir}' not found. "
              f"Writing identity calibration (T=1.0).")
        save_temperature(args.output, 1.0, 0.0, 0.0, extra={"note": "no_validation_data"})
        return

    loader = build_loader(args.val_dir, batch_size=args.batch_size)
    logits, labels = collect_logits(model, loader, device)
    print(f"[Calibrate] Collected {len(labels)} validation samples.")

    temperature, ece_before, ece_after = calibrate_from_logits(logits, labels, n_bins=args.n_bins)
    print(f"[Calibrate] Optimal T = {temperature:.4f}")
    print(f"[Calibrate] ECE before = {ece_before:.4f} -> after = {ece_after:.4f}")

    save_temperature(
        args.output, temperature, ece_before, ece_after,
        extra={"num_samples": int(len(labels)), "class_names": CLASS_NAMES}
    )


if __name__ == "__main__":
    main()