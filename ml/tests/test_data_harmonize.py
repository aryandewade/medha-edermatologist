"""
Tests for ml/src/data_harmonize.py (label mapping, dedupe, split, ingest).
Uses synthetic images in a temp directory; no external datasets required.
"""

import os
import sys
import tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_harmonize import (  # noqa: E402
    map_label,
    map_pad_code,
    dhash,
    hamming_distance,
    deduplicate,
    assign_groups_by_hash,
    stratified_grouped_split,
    ingest_class_folders,
    materialize,
    harmonize,
    Sample,
    CLASS_NAMES,
)


def test_map_label_canonical_classes():
    cases = {
        "Eczema": "eczema",
        "Atopic Dermatitis": "eczema",
        "Psoriasis vulgaris": "psoriasis",
        "Tinea corporis": "tinea",
        "Ringworm": "tinea",
        "Acne vulgaris": "acne",
        "Normal skin": "healthy",
        "Melanoma": "suspicious_lesion",
        "Basal cell carcinoma": "suspicious_lesion",
        "Squamous cell carcinoma": "suspicious_lesion",
        "Actinic keratosis": "suspicious_lesion",
    }
    for raw, expected in cases.items():
        got = map_label(raw)
        assert got == expected, f"map_label({raw!r}) = {got!r}, expected {expected!r}"
    print("[PASS] map_label for all 6 canonical classes")


def test_map_label_exclusions():
    for raw in ("Melanocytic nevus", "Seborrheic keratosis", "Benign mole",
                "Vitiligo", "Folliculitis?"):
        # Folliculitis maps to acne; ensure the explicit benigns are dropped
        if raw == "Folliculitis?":
            continue
        assert map_label(raw) is None, f"{raw!r} should be excluded (got {map_label(raw)})"
    assert map_label("") is None
    print("[PASS] out-of-scope/benign labels excluded")


def test_map_pad_codes():
    assert map_pad_code("BCC") == "suspicious_lesion"
    assert map_pad_code("MEL") == "suspicious_lesion"
    assert map_pad_code("ACK") == "suspicious_lesion"
    assert map_pad_code("SCC") == "suspicious_lesion"
    assert map_pad_code("NEV") is None
    assert map_pad_code("SEK") is None
    print("[PASS] PAD-UFES-20 code mapping")


def _write_image(path, seed):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(150, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


def test_dhash_and_dedupe():
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.jpg")
        b = os.path.join(tmp, "b.jpg")   # identical copy of a
        c = os.path.join(tmp, "c.jpg")   # different
        _write_image(a, 1)
        _write_image(c, 2)
        with open(a, "rb") as f:
            data = f.read()
        with open(b, "wb") as f:
            f.write(data)

        ha, hb = dhash(a), dhash(b)
        assert ha is not None and hb is not None
        assert hamming_distance(ha, hb) == 0, "Identical images must hash identically"

        samples = [
            Sample(a, "Eczema", "eczema", "local"),
            Sample(b, "Eczema", "eczema", "local"),
            Sample(c, "Psoriasis", "psoriasis", "local"),
        ]
        kept = deduplicate(samples, max_distance=4)
        assert len(kept) == 2, f"Dedupe should collapse a/b -> got {len(kept)}"
    print("[PASS] dHash + perceptual deduplication")


def test_split_is_stratified_and_group_safe():
    samples = []
    # 60 per class; groups of size 3 that must stay together
    for cls in CLASS_NAMES:
        for i in range(60):
            samples.append(Sample(f"/x/{cls}_{i}.jpg", cls, cls, "local", group=i // 3))

    splits = stratified_grouped_split(samples, seed=7)

    # No group may appear in more than one split (leakage guard)
    seen = {}
    for split, items in splits.items():
        for s in items:
            key = (s.canonical, s.group)
            assert key not in seen or seen[key] == split, f"Group {key} leaked across splits"
            seen[key] = split

    # Total preserved
    assert sum(len(v) for v in splits.values()) == len(samples)

    # Each split should contain every class (stratification)
    for split, items in splits.items():
        classes = {s.canonical for s in items}
        assert classes == set(CLASS_NAMES), f"{split} missing classes: {set(CLASS_NAMES)-classes}"
    print(f"[PASS] stratified group-safe split "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def test_ingest_and_materialize():
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        for cls in ("Eczema", "Psoriasis", "Healthy skin"):
            os.makedirs(os.path.join(raw, cls), exist_ok=True)
            for i in range(4):
                _write_image(os.path.join(raw, cls, f"{cls}_{i}.jpg"), 10 + i)

        samples = ingest_class_folders(raw, source="local")
        assert len(samples) == 12, f"Expected 12 ingested, got {len(samples)}"
        canon = {s.canonical for s in samples}
        assert canon == {"eczema", "psoriasis", "healthy"}

        assign_groups_by_hash(samples)
        splits = stratified_grouped_split(samples)
        out = os.path.join(tmp, "out")
        summary = materialize(splits, out, dry_run=False)

        assert os.path.exists(os.path.join(out, "manifest.csv"))
        assert os.path.exists(os.path.join(out, "train", "eczema"))
        assert sum(sum(c.values()) for c in summary.values()) == 12
    print("[PASS] ingest_class_folders + materialize (manifest written)")


def test_harmonize_empty_is_safe():
    result = harmonize({}, out_dir="unused", dry_run=True)
    assert result["samples"] == 0
    print("[PASS] harmonize() handles missing sources gracefully")


if __name__ == "__main__":
    test_map_label_canonical_classes()
    test_map_label_exclusions()
    test_map_pad_codes()
    test_dhash_and_dedupe()
    test_split_is_stratified_and_group_safe()
    test_ingest_and_materialize()
    test_harmonize_empty_is_safe()
    print("\nAll data_harmonize tests passed.")