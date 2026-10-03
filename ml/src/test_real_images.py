"""
Test E-Dermatologist V2 Inference Pipeline on Real Clinical Images
Runs inference on actual test images across all 6 dermatological classes.
"""

import sys
from pathlib import Path
import cv2

# Add ml/src to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import MobileSpacerInferenceEngine, CLASS_NAMES


def test_real_samples():
    print("=" * 70)
    print("INITIALIZING E-DERMATOLOGIST INFERENCE ENGINE (V2 RESIDUAL MODEL)")
    print("=" * 70)

    engine = MobileSpacerInferenceEngine(
        checkpoint_path="backend/models/best_model.pt",
        temperature_path="backend/models/temperature.json",
        device="cpu"
    )

    test_root = Path("ml/data/test")
    if not test_root.exists():
        print(f"Error: {test_root} not found.")
        return

    print("\n" + "=" * 70)
    print("RUNNING INFERENCE ON REAL CLINICAL PHOTOGRAPHS")
    print("=" * 70)

    correct = 0
    total = 0

    for class_name in CLASS_NAMES:
        class_folder = test_root / class_name
        if not class_folder.exists():
            continue

        images = [f for f in class_folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not images:
            continue

        # Test first 2 images from each class
        sample_images = images[:2]
        for img_path in sample_images:
            total += 1
            bgr_image = cv2.imread(str(img_path))
            if bgr_image is None:
                continue

            result = engine.predict_image(bgr_image)
            pred_class = result["prediction"]["class_id"]
            pred_name = result["prediction"]["label"]
            confidence = result["prediction"]["confidence"] * 100.0
            advice = result["advice"]
            quality = result["quality"]
            top3 = result["top3"]
            latency = result["latency_ms"]

            is_match = (pred_class == class_name)
            if is_match:
                correct += 1

            badge = "MATCH" if is_match else "CONFUSED"

            print(f"\n[{badge}] Ground Truth: {class_name.upper()} | File: {img_path.name}")
            print(f"  --> Top Prediction: {pred_name} ({confidence:.1f}%) [Level: {advice['level'].upper()}]")
            top3_str = ", ".join([f"{item['label']}: {item['probability']*100:.1f}%" for item in top3])
            print(f"  --> Differential (Top 3): {top3_str}")
            print(f"  --> Clinical Advice: {advice['title']} - {advice['message']}")
            ref_patch = quality.get("reference_patch_found", False)
            print(f"  --> Image Quality Gate: Passed={quality['passed']} (Blur: {quality['blur_score']}, Glare: {quality['glare_ratio']*100:.1f}%), Ref Patch={ref_patch}")
            print(f"  --> Processing Latency: {latency:.1f} ms")

    acc = (correct / total) * 100.0 if total > 0 else 0
    print("\n" + "=" * 70)
    print(f"REAL IMAGE SCREENING TEST COMPLETE: {correct}/{total} ({acc:.1f}% Top-1 Accuracy)")
    print("=" * 70)


if __name__ == "__main__":
    test_real_samples()
