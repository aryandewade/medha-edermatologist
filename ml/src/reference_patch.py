"""
Reference Patch Detection & Colour Calibration (Mobile Spacer Profile)

Locates the calibrated 18% neutral-gray reference patch mounted on the spacer's
contact rim and uses it to perform high-precision white balance so that skin
colour is preserved regardless of the ambient clinic illuminant.

The detector searches the peripheral annulus of the circular spacer field of
view (the lesion occupies the centre) for a neutral, mid-luminance, compact
region. Once located, per-channel gains are computed relative to the green
channel and applied across the whole frame.

Public interface (coordination contract - do not rename):
    detect_and_calibrate_patch(image_bgr) -> (corrected_bgr, patch_found, metrics)
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tuning constants (18% neutral gray card characteristics)
# ---------------------------------------------------------------------------
# An 18% reflectance gray card renders at roughly sRGB 100-150 under normal
# exposure. We accept a generous band to survive mild under/over-exposure.
GRAY_VALUE_MIN = 55
GRAY_VALUE_MAX = 205
# Absolute ceiling on candidate saturation. A gray card under even a strong
# illuminant cast stays below this; skin sits well above it.
MAX_SATURATION = 110
# A candidate must be clearly *more neutral* than the surrounding skin: its mean
# saturation must fall below this fraction of the annulus baseline saturation.
RELATIVE_NEUTRALITY_FACTOR = 0.85
# ...or be absolutely this neutral (weak-cast / good-illumination captures).
# 70/255 HSV saturation allows a moderate tungsten/daylight white-balance cast
# on a true gray card while still excluding skin.
ABSOLUTE_NEUTRAL_SATURATION = 70
# Patch must occupy a sane fraction of the field of view.
MIN_PATCH_AREA_RATIO = 0.0015
MAX_PATCH_AREA_RATIO = 0.09
# Only look in the outer annulus (lesion sits centrally).
ANNULUS_INNER_FRAC = 0.55
ANNULUS_OUTER_FRAC = 1.02
# Guard rails so a spurious detection cannot destroy the image.
MIN_GAIN = 0.6
MAX_GAIN = 1.8
# A patch whose channels differ by more than this after normalisation is
# considered a poor calibration (still returned, but flagged).
NEUTRAL_SPREAD_TOLERANCE = 0.06


def _peripheral_annulus_mask(height: int, width: int) -> np.ndarray:
    """Binary mask covering the outer annulus of the circular spacer field."""
    cy, cx = height / 2.0, width / 2.0
    max_r = 0.5 * min(height, width)
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inner = ANNULUS_INNER_FRAC * max_r
    outer = ANNULUS_OUTER_FRAC * max_r
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[(dist >= inner) & (dist <= outer)] = 255
    return mask


def _candidate_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Broad mask of mid-luminance, reasonably desaturated pixels. Final neutrality
    ranking is applied per-blob in find_reference_patch().
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    candidate = (saturation <= MAX_SATURATION) & (value >= GRAY_VALUE_MIN) & (value <= GRAY_VALUE_MAX)
    return (candidate.astype(np.uint8)) * 255


def _baseline_annulus_saturation(image_bgr: np.ndarray, annulus: np.ndarray) -> float:
    """
    Median HSV saturation of the annulus (dominated by skin). Used as the
    adaptive reference a gray patch must beat to be considered neutral.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Include all reasonably lit annulus pixels (skin may legitimately sit above
    # the gray card's exposure window). Only skip the near-black spacer ring.
    valid = (annulus > 0) & (value > 20)
    pixels = saturation[valid]
    if pixels.size == 0:
        return float(np.median(saturation))
    return float(np.median(pixels))


def find_reference_patch(
    image_bgr: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate the 18% gray reference patch on the spacer margin.

    Returns:
        (x, y, w, h) bounding box of the detected patch, or None if not found.
    """
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        return None

    h, w = image_bgr.shape[:2]
    annulus = _peripheral_annulus_mask(h, w)
    candidates = _candidate_mask(image_bgr)

    blob_mask = cv2.bitwise_and(annulus, candidates)
    # Clean speckle then close the patch into a single blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Adaptive neutrality reference derived from the surrounding skin.
    baseline_sat = _baseline_annulus_saturation(image_bgr, annulus)
    relative_limit = RELATIVE_NEUTRALITY_FACTOR * baseline_sat
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    total_area = float(h * w)
    best_box: Optional[Tuple[int, int, int, int]] = None
    best_sat = 1e9
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        ratio = area / total_area
        if ratio < MIN_PATCH_AREA_RATIO or ratio > MAX_PATCH_AREA_RATIO:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 5 or bh < 5:
            continue

        # A physical flat patch is roughly square/rectangular and compact.
        aspect = bw / float(bh)
        if aspect < 0.4 or aspect > 2.5:
            continue
        rectangularity = area / float(bw * bh + 1e-6)
        if rectangularity < 0.55:
            continue

        # Neutrality gate: must be absolutely neutral OR clearly more neutral
        # than the surrounding skin baseline.
        region_sat = saturation[y:y + bh, x:x + bw]
        blob_sat = float(np.mean(region_sat[region_sat > 0])) if region_sat.size else 255.0
        is_neutral = (blob_sat < ABSOLUTE_NEUTRAL_SATURATION) or (blob_sat < relative_limit and blob_sat < MAX_SATURATION)
        if not is_neutral:
            continue

        # Rank primarily by neutrality (lowest saturation), then by size.
        if blob_sat < best_sat - 1e-6 or (abs(blob_sat - best_sat) <= 1e-6 and area > best_area):
            best_sat = blob_sat
            best_area = area
            best_box = (int(x), int(y), int(bw), int(bh))

    return best_box


def _sample_region_mean(image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Mean BGR of the central portion of a bounding box (avoids edge bleed)."""
    x, y, w, h = box
    # Shrink 20% inward to avoid the dark patch border / spacer shadow.
    inset_x = max(1, int(w * 0.2))
    inset_y = max(1, int(h * 0.2))
    x0, y0 = x + inset_x, y + inset_y
    x1 = min(image_bgr.shape[1], x + w - inset_x)
    y1 = min(image_bgr.shape[0], y + h - inset_y)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = x, y, x + w, y + h
    region = image_bgr[y0:y1, x0:x1]
    return region.reshape(-1, 3).mean(axis=0).astype(np.float32)


def white_balance_from_patch(
    image_bgr: np.ndarray,
    patch_box: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply per-channel gains so the sampled reference patch becomes neutral.

    Returns:
        (corrected_bgr, gains_bgr, patch_mean_bgr)
    """
    patch_mean = _sample_region_mean(image_bgr, patch_box)  # order: [B, G, R]
    b, g, r = float(patch_mean[0]), float(patch_mean[1]), float(patch_mean[2])
    g = max(g, 1e-3)

    gain_b = g / max(b, 1e-3)
    gain_g = 1.0
    gain_r = g / max(r, 1e-3)

    # Guard rails to avoid catastrophic amplification on bad detections.
    gain_b = float(np.clip(gain_b, MIN_GAIN, MAX_GAIN))
    gain_r = float(np.clip(gain_r, MIN_GAIN, MAX_GAIN))
    gains = np.array([gain_b, gain_g, gain_r], dtype=np.float32)

    corrected = image_bgr.astype(np.float32) * gains.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected, gains, patch_mean


def _neutrality_deviation(patch_mean_bgr: np.ndarray) -> float:
    """
    Normalised spread of the (post-calibration) patch channels.

    0.0 == perfectly neutral gray, higher == stronger residual colour cast.
    """
    mean = float(np.mean(patch_mean_bgr))
    if mean < 1e-3:
        return 1.0
    return float((np.max(patch_mean_bgr) - np.min(patch_mean_bgr)) / mean)


def detect_and_calibrate_patch(
    image_bgr: np.ndarray
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Locate the 18% gray reference patch on the spacer margin and white-balance
    the frame against it.

    Args:
        image_bgr: OpenCV BGR uint8 image captured through the spacer.

    Returns:
        corrected_bgr: Colour-normalised BGR image (unchanged if no patch found).
        patch_found: True if an 18% gray patch was located on the spacer margin.
        metrics: {
            "gray_deviation": float,        # residual colour cast (0 = neutral)
            "patch_coords": [x, y, w, h],    # detected box ([-1,-1,0,0] if none)
            "gains_bgr": [gB, gG, gR],       # applied white-balance gains
            "patch_mean_bgr": [B, G, R],     # sampled patch colour
            "method": str                    # "reference_patch" | "reference_patch_weak" | "none"
        }
    """
    default_metrics: Dict[str, Any] = {
        "gray_deviation": 1.0,
        "patch_coords": [-1, -1, 0, 0],
        "gains_bgr": [1.0, 1.0, 1.0],
        "patch_mean_bgr": [0.0, 0.0, 0.0],
        "method": "none",
    }

    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        return image_bgr, False, default_metrics

    patch_box = find_reference_patch(image_bgr)
    if patch_box is None:
        return image_bgr, False, default_metrics

    corrected, gains, patch_mean = white_balance_from_patch(image_bgr, patch_box)

    # Re-sample the calibrated image to report the achieved neutrality.
    calibrated_patch_mean = _sample_region_mean(corrected, patch_box)
    deviation = _neutrality_deviation(calibrated_patch_mean)

    metrics: Dict[str, Any] = {
        "gray_deviation": round(deviation, 4),
        "patch_coords": [patch_box[0], patch_box[1], patch_box[2], patch_box[3]],
        "gains_bgr": [round(float(g), 4) for g in gains],
        "patch_mean_bgr": [round(float(v), 2) for v in patch_mean],
        "method": "reference_patch",
    }

    if deviation > NEUTRAL_SPREAD_TOLERANCE:
        # Detection was marginal - fall back to the raw frame so we never emit a
        # worse image than we received.
        metrics["method"] = "reference_patch_weak"
        return image_bgr, True, metrics

    return corrected, True, metrics


def count_patch_matches(image_bgr: np.ndarray) -> int:
    """Debug helper: number of candidate neutral blobs found in the annulus."""
    h, w = image_bgr.shape[:2]
    annulus = _peripheral_annulus_mask(h, w)
    candidates = _candidate_mask(image_bgr)
    blob_mask = cv2.bitwise_and(annulus, candidates)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect & calibrate the 18% gray reference patch")
    parser.add_argument("--image", type=str, required=True, help="Path to input spacer capture")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[ReferencePatch] Could not read image: {args.image}")

    corrected, found, metrics = detect_and_calibrate_patch(img)
    print(f"[ReferencePatch] patch_found={found}")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    out_path = args.image.rsplit(".", 1)[0] + "_wb.jpg"
    cv2.imwrite(out_path, corrected)
    print(f"[ReferencePatch] Saved calibrated image to {out_path}")