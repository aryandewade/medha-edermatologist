"""
Integration test: full inference pipeline including reference-patch colour
constancy (Task 2.2) and temperature calibration loading (Task 2.3).

Uses the ImageNet-pretrained backbone (weights are cached locally), so the
class probabilities are arbitrary — we assert on pipeline structure/contract,
not on clinical correctness.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import MobileSpacerInferenceEngine, CLASS_NAMES  # noqa: E402


def _make_capture() -> np.ndarray:
    size = 400
    img = np.full((size, size, 3), (170, 175, 215), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), 42, (95, 105, 160), -1)
    img[170:220, 25:75] = (120, 120, 120)  # 18% gray patch on margin
    f = img.astype(np.float32)
    f[:, :, 0] *= 0.82
    f[:, :, 2] *= 1.06
    return np.clip(f, 0, 255).astype(np.uint8)


def test_full_pipeline_contract():
    engine = MobileSpacerInferenceEngine(device="cpu")
    result = engine.predict_image(_make_capture())

    # --- Response contract ---
    for key in ("prediction", "top3", "probabilities", "quality", "advice", "latency_ms", "roi_preview"):
        assert key in result, f"Missing result key: {key}"

    probs = result["probabilities"]
    assert set(probs.keys()) == set(CLASS_NAMES), "Probability vector must span all 6 classes"
    assert abs(sum(probs.values()) - 1.0) < 1e-3, "Probabilities must sum to ~1"

    assert len(result["top3"]) == 3, "Top-3 must contain 3 entries"
    assert result["advice"]["level"] in ("ok", "uncertain", "urgent")

    # --- Task 2.2 integration: real reference-patch telemetry ---
    quality = result["quality"]
    assert "reference_patch_found" in quality, "quality must carry patch status"
    assert quality["reference_patch_found"] is True, "Synthetic patch should be detected"
    assert quality["colour_cast"] in ("corrected_neutral", "corrected_weak", "corrected_gray_world")

    # --- Task 2.3 integration: temperature applied ---
    assert hasattr(engine, "temperature") and engine.temperature > 0

    print(f"[PASS] prediction={result['prediction']['class_id']} "
          f"conf={result['prediction']['confidence']} patch={quality['reference_patch_found']} "
          f"latency={result['latency_ms']}ms")


if __name__ == "__main__":
    test_full_pipeline_contract()
    print("\nInference pipeline integration test passed.")