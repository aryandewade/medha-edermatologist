"""
Digital Image Processing Pipeline for Smartphone + 3D Spacer Skin Screening

Modules:
  1. Quality Gate: Laplacian blur check, illumination dynamic range, glare ratio.
  2. Colour Constancy: Shades-of-Gray (Minkowski p=6) and reference-patch normalization.
  3. Lesion Localization: Mask spacer rim, skin segmentation (YCbCr), and lesion ROI crop.
  4. Standardization: 224x224 bicubic resampling and ImageNet normalization.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch

from reference_patch import detect_and_calibrate_patch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Quality thresholds for mobile 3D spacer captures
DEFAULT_QUALITY_THRESHOLDS = {
    "blur_min_laplacian_var": 55.0,
    "brightness_min": 40.0,
    "brightness_max": 225.0,
    "glare_max_ratio": 0.12,
}


def check_image_quality(
    image: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Evaluates image quality of mobile spacer capture before model evaluation.
    """
    cfg = thresholds or DEFAULT_QUALITY_THRESHOLDS
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 1. Blur evaluation (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = laplacian_var >= cfg["blur_min_laplacian_var"]

    # 2. Illumination evaluation
    mean_brightness = float(np.mean(gray))
    is_well_lit = cfg["brightness_min"] <= mean_brightness <= cfg["brightness_max"]

    # 3. Specular glare ratio (overexposed white hot spots)
    glare_pixels = np.sum(gray >= 245)
    total_pixels = gray.size
    glare_ratio = float(glare_pixels / max(total_pixels, 1))
    glare_ok = glare_ratio <= cfg["glare_max_ratio"]

    passed = is_sharp and is_well_lit and glare_ok
    issues = []
    if not is_sharp:
        issues.append(f"Image too blurry (score {laplacian_var:.1f} < {cfg['blur_min_laplacian_var']:.1f}). Hold phone steady.")
    if mean_brightness < cfg["brightness_min"]:
        issues.append(f"Too dark ({mean_brightness:.1f} < {cfg['brightness_min']:.1f}). Increase clinic room lighting.")
    elif mean_brightness > cfg["brightness_max"]:
        issues.append(f"Overexposed ({mean_brightness:.1f} > {cfg['brightness_max']:.1f}). Reduce direct glare.")
    if not glare_ok:
        issues.append(f"Excessive specular reflection ({glare_ratio*100:.1f}%). Tilt spacer slightly.")

    return {
        "passed": passed,
        "blur_score": round(laplacian_var, 2),
        "brightness": round(mean_brightness, 2),
        "glare_ratio": round(glare_ratio, 4),
        "status": "good" if passed else "warning",
        "issues": issues
    }


def apply_shades_of_gray(img_rgb: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Applies the Shades-of-Gray colour constancy algorithm (Minkowski p-norm)
    to neutralize illumination color casts caused by varied room bulbs.
    """
    img_float = img_rgb.astype(np.float32)
    # Compute Minkowski p-norm along spatial dimensions for each channel
    e_r = np.power(np.mean(np.power(img_float[:, :, 0], p)), 1.0 / p)
    e_g = np.power(np.mean(np.power(img_float[:, :, 1], p)), 1.0 / p)
    e_b = np.power(np.mean(np.power(img_float[:, :, 2], p)), 1.0 / p)

    illuminant = np.array([e_r, e_g, e_b], dtype=np.float32)
    norm = np.linalg.norm(illuminant)
    if norm < 1e-4:
        return img_rgb

    illuminant /= norm
    # Target neutral grey: [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    target = 1.0 / np.sqrt(3.0)
    gains = target / np.maximum(illuminant, 1e-4)

    corrected = img_float * gains
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def localize_lesion_roi(
    img_rgb: np.ndarray,
    mask_spacer_rim: bool = True
) -> np.ndarray:
    """
    Localizes the primary skin lesion, eliminates dark peripheral spacer boundaries,
    and crops a symmetric square bounding region preserving morphological aspect ratio.
    """
    h, w = img_rgb.shape[:2]
    min_dim = min(h, w)

    # 1. Mask dark circular spacer rim if present
    if mask_spacer_rim:
        center_x, center_y = w // 2, h // 2
        radius = int(min_dim * 0.44)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    else:
        mask = np.ones((h, w), dtype=np.uint8) * 255

    # 2. Segment skin in YCrCb color space
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    combined_mask = cv2.bitwise_and(skin_mask, mask)

    # 3. Detect lesion via Otsu thresholding on green channel (high contrast for erythema)
    green = img_rgb[:, :, 1]
    masked_green = cv2.bitwise_and(green, green, mask=combined_mask)
    
    # Threshold for darker/erythematous regions
    _, lesion_mask = cv2.threshold(masked_green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lesion_mask = cv2.bitwise_and(lesion_mask, combined_mask)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If a prominent lesion contour is found, crop around it
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > (min_dim * min_dim * 0.02):  # Lesion occupies > 2% of area
            x, y, bw, bh = cv2.boundingRect(c)
            # Expand to square with 25% margin
            box_side = int(max(bw, bh) * 1.25)
            center_bx, center_by = x + bw // 2, y + bh // 2
            
            x1 = max(0, center_bx - box_side // 2)
            y1 = max(0, center_by - box_side // 2)
            x2 = min(w, x1 + box_side)
            y2 = min(h, y1 + box_side)
            
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 50 and crop.shape[1] > 50:
                return crop

    # Default fallback: Center square crop of the clear aperture
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return img_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]


def preprocess_frame_ex(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_colour_constancy: bool = True,
    is_bgr: bool = True,
    use_reference_patch: bool = True
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """
    Full end-to-end preprocessing pipeline for mobile spacer photos.

    Colour constancy prefers the calibrated 18% gray reference patch (Task 2.2)
    and gracefully falls back to Shades-of-Gray when no patch is detected.

    Returns:
        tensor: Standardized PyTorch tensor (1, 3, 224, 224)
        roi_preview: Color-corrected, localized RGB crop (224, 224, 3)
        meta: {"reference_patch_found": bool, "colour_cast": str, "patch_metrics": dict}
    """
    # 1. Convert to RGB (retain a BGR view for the reference-patch detector)
    if is_bgr and len(frame.shape) == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bgr = frame.copy()
    else:
        rgb = frame.copy()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame

    patch_found = False
    colour_cast = "normal"
    patch_metrics: Dict[str, Any] = {
        "gray_deviation": 1.0,
        "patch_coords": [-1, -1, 0, 0],
        "gains_bgr": [1.0, 1.0, 1.0],
        "patch_mean_bgr": [0.0, 0.0, 0.0],
        "method": "none",
    }

    # 2. Colour Constancy: reference patch first, Shades-of-Gray as fallback.
    if apply_colour_constancy:
        if use_reference_patch and len(bgr.shape) == 3:
            corrected_bgr, patch_found, patch_metrics = detect_and_calibrate_patch(bgr)
            if patch_found:
                rgb_corrected = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
                colour_cast = "corrected_neutral" if patch_metrics["method"] == "reference_patch" else "corrected_weak"
        if not patch_found:
            rgb_corrected = apply_shades_of_gray(rgb, p=6)
            colour_cast = "corrected_gray_world"
    else:
        rgb_corrected = rgb
        colour_cast = "uncorrected"

    # 3. Lesion Localization & Spacer Rim Removal
    roi = localize_lesion_roi(rgb_corrected, mask_spacer_rim=True)

    # 4. Resample to 224 x 224 using bicubic interpolation
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)

    # 5. ImageNet Standardization
    img_float = resized.astype(np.float32) / 255.0
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # 6. Permute to (1, 3, 224, 224) PyTorch Tensor
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

    meta: Dict[str, Any] = {
        "reference_patch_found": bool(patch_found),
        "colour_cast": colour_cast,
        "patch_metrics": patch_metrics,
    }
    return tensor, resized, meta


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_colour_constancy: bool = True,
    is_bgr: bool = True,
    use_reference_patch: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Backward-compatible wrapper around preprocess_frame_ex().

    Returns:
        tensor: Standardized PyTorch tensor (1, 3, 224, 224)
        roi_preview: Color-corrected, localized RGB crop (224, 224, 3)
    """
    tensor, roi_preview, _ = preprocess_frame_ex(
        frame,
        target_size=target_size,
        apply_colour_constancy=apply_colour_constancy,
        is_bgr=is_bgr,
        use_reference_patch=use_reference_patch,
    )
    return tensor, roi_preview
