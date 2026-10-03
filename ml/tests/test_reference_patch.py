"""
Tests for ml/src/reference_patch.py

Runnable standalone (python ml/tests/test_reference_patch.py) or via pytest.
Synthetic captures model a smartphone spacer frame: skin fill, a central lesion,
and a neutral 18% gray patch mounted on the margin.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reference_patch import (  # noqa: E402
    detect_and_calibrate_patch,
    find_reference_patch,
)


def _make_spacer_capture(with_patch: bool = True, cast: bool = True) -> np.ndarray:
    """Build a synthetic BGR spacer capture with optional gray patch/cast."""
    size = 400
    img = np.full((size, size, 3), (170, 175, 215), dtype=np.uint8)  # skin (BGR)

    # Central dark lesion (excluded from the peripheral annulus).
    cv2.circle(img, (size // 2, size // 2), 42, (95, 105, 160), -1)

    if with_patch:
        # Neutral 18% gray card on the left margin.
        img[170:220, 25:75] = (120, 120, 120)

    if cast:
        # Warm tungsten-like cast: suppress blue, boost red.
        f = img.astype(np.float32)
        f[:, :, 0] *= 0.82   # B
        f[:, :, 2] *= 1.06   # R
        img = np.clip(f, 0, 255).astype(np.uint8)
    return img


def test_patch_detected_and_calibrated():
    img = _make_spacer_capture(with_patch=True, cast=True)

    corrected, found, metrics = detect_and_calibrate_patch(img)

    assert found is True, "Reference patch should be detected in margin"
    assert metrics["method"] in ("reference_patch", "reference_patch_weak")

    x, y, w, h = metrics["patch_coords"]
    assert 20 <= x <= 40 and 165 <= y <= 225, f"Patch box off-target: {metrics['patch_coords']}"

    # After calibration the patch channels must be near-neutral.
    patch_mean = np.array(metrics["patch_mean_bgr"], dtype=np.float32)
    assert metrics["gray_deviation"] < 0.06, f"Residual cast too high: {metrics['gray_deviation']}"

    # Corrected image must differ from the (cast) input.
    assert not np.array_equal(corrected, img), "Calibration should modify a cast image"
    assert corrected.shape == img.shape and corrected.dtype == np.uint8
    print(f"[PASS] patch found at {metrics['patch_coords']}, "
          f"gains={metrics['gains_bgr']}, dev={metrics['gray_deviation']}")


def test_no_patch_returns_unchanged():
    img = _make_spacer_capture(with_patch=False, cast=True)

    corrected, found, metrics = detect_and_calibrate_patch(img)

    assert found is False, "No gray patch should be reported as not found"
    assert metrics["method"] == "none"
    assert metrics["patch_coords"] == [-1, -1, 0, 0]
    assert np.array_equal(corrected, img), "Frame must be returned unchanged"
    print("[PASS] no-patch frame returned unchanged")


def test_find_reference_patch_returns_none_on_blank():
    blank = np.full((300, 300, 3), 250, dtype=np.uint8)  # all white, no patch
    assert find_reference_patch(blank) is None
    print("[PASS] blank image yields no patch")


def test_handles_invalid_input():
    corrected, found, metrics = detect_and_calibrate_patch(None)
    assert found is False
    assert corrected is None
    print("[PASS] invalid input handled gracefully")


if __name__ == "__main__":
    test_patch_detected_and_calibrated()
    test_no_patch_returns_unchanged()
    test_find_reference_patch_returns_none_on_blank()
    test_handles_invalid_input()
    print("\nAll reference_patch tests passed.")