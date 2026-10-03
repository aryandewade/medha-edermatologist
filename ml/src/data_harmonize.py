"""
Dataset Harmonization Pipeline (Mobile Spacer Profile)

Ingests heterogeneous clinical/smartphone datasets (Fitzpatrick17k, SCIN,
PAD-UFES-20, DermNet, SD-198 and custom folders), maps thousands of raw disease
tags into the 6 canonical classes, removes near-duplicates via perceptual
hashing, and produces a patient/group-aware stratified train/val/test split.

Canonical classes (configs/classes.yaml):
    eczema, psoriasis, tinea, acne, healthy, suspicious_lesion

Usage:
    python src/data_harmonize.py \
        --fitzpatrick_dir raw/fitzpatrick17k \
        --scin_dir raw/scin \
        --pad_dir raw/pad-ufes-20 \
        --out_dir data --dedupe --dry_run
"""

import os
import csv
import json
import shutil
import argparse
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

CLASS_NAMES = ["eczema", "psoriasis", "tinea", "acne", "healthy", "suspicious_lesion"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
# Ordering matters: the first group whose keyword is found wins, so the
# high-risk suspicious class is evaluated before look-alike inflammatory terms.
# "actinic keratosis" must be caught by `suspicious_lesion`, not by a generic
# "keratosis" match elsewhere.
LABEL_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("suspicious_lesion", [
        "melanoma", "basal cell", "bcc", "squamous cell", "scc", "carcinoma",
        "actinic keratosis", "actinic", "keratoacanthoma", "bowen",
        "malignant", "neoplasm", "neoplasia", "dysplastic", "atypical",
        "cancer", "skincancer", "skin cancer",
    ]),
    ("eczema", [
        "eczema", "atopic", "dermatitis", "dyshidrotic", "lichen simplex",
    ]),
    ("psoriasis", ["psoriasis", "psoriatic"]),
    ("tinea", [
        "tinea", "ringworm", "dermatophyt", "dermatophyte", "fungal", "fungus",
        "candid", "candida",
    ]),
    ("acne", ["acne", "comedo", "comedonal", "folliculitis", "follicular"]),
    ("psoriasis", ["parapsoriasis"]),
    ("healthy", ["healthy", "normal skin", "normal", "benign skin", "no disease",
                 "nil", "clear skin"]),
]

# Conditions explicitly OUT of scope / benign look-alikes to discard.
EXCLUDE_KEYWORDS: List[str] = [
    "nevus", "nevi", "naevus", "mole", "seborrheic", "lentigo", "freckle",
    "melanocytic nevus", "benign", "wart", "verruca", "vitiligo", "hemangioma",
    "haemangioma", "cyst", "scar", "keloid", "urticaria", "hives", "rosacea",
    "melasma", "alopecia", "onychomycosis", "herpes", "molluscum",
]


def normalize_label(raw: str) -> str:
    """Lower-case and collapse separators for robust keyword matching."""
    if raw is None:
        return ""
    text = str(raw).strip().lower()
    for ch in ("_", "-", "/", "(", ")", "[", "]", ":", ";"):
        text = text.replace(ch, " ")
    return " ".join(text.split())


def map_label(raw: str) -> Optional[str]:
    """
    Map a raw source disease tag to one of the 6 canonical classes.

    Returns None for out-of-scope / benign-look-alike conditions.
    """
    text = normalize_label(raw)
    if not text:
        return None

    for token in EXCLUDE_KEYWORDS:
        if token in text:
            return None

    for canonical, keywords in LABEL_KEYWORDS:
        for kw in keywords:
            if kw in text:
                return canonical
    return None


# PAD-UFES-20 explicit diagnostic codes (authoritative short codes).
PAD_UFES_CODES: Dict[str, Optional[str]] = {
    "bcc": "suspicious_lesion",
    "scc": "suspicious_lesion",
    "mel": "suspicious_lesion",
    "ack": "suspicious_lesion",
    "nev": None,
    "sek": None,
    "bkl": None,
}


def map_pad_code(raw: str) -> Optional[str]:
    """Resolve a PAD-UFES-20 diagnostic code (BCC/SCC/MEL/ACK/NEV/SEK/BKL)."""
    code = normalize_label(raw)
    if code in PAD_UFES_CODES:
        return PAD_UFES_CODES[code]
    return map_label(raw)


# ---------------------------------------------------------------------------
# Sample container & perceptual deduplication
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    path: str
    raw_label: str
    canonical: str
    source: str
    group: int = -1          # near-duplicate cluster id (set during dedupe)
    image_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def dhash(image_path: str, hash_size: int = 8) -> Optional[int]:
    """Difference hash returned as an integer for ultra-fast bit_count() comparisons."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    except (OSError, ValueError):
        return None
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    raw_bytes = np.packbits(diff.flatten()).tobytes()
    return int.from_bytes(raw_bytes, "big")


def deduplicate(
    samples: List[Sample],
    max_distance: int = 4,
    hash_size: int = 8
) -> List[Sample]:
    """
    Greedy perceptual-hash clustering using fast POPCNT/bit_count. Assigns a shared `group` id to
    near-identical images (Hamming distance <= max_distance) so that duplicates
    never straddle train/test splits. Returns the deduplicated sample list
    (first occurrence of each cluster is kept).
    """
    clusters: List[Tuple[Sample, int]] = []
    kept: List[Sample] = []

    for i, sample in enumerate(samples):
        if (i + 1) % 1000 == 0:
            print(f"[Harmonize] Deduplicating progress: {i + 1}/{len(samples)}...", flush=True)
        h = dhash(sample.path, hash_size=hash_size)
        if h is None:
            sample.group = -1
            kept.append(sample)
            continue
        sample.image_hash = hex(h)[2:].zfill(16)

        matched_group = -1
        for rep_sample, rep_hash in clusters:
            if (h ^ rep_hash).bit_count() <= max_distance:
                matched_group = rep_sample.group
                break

        if matched_group == -1:
            sample.group = len(clusters)
            clusters.append((sample, h))
            kept.append(sample)
        # duplicates (matched_group != -1) are dropped

    return kept


def assign_groups_by_hash(samples: List[Sample], hash_size: int = 8, max_distance: int = 4) -> None:
    """
    Assign cluster ids to every sample (in place) WITHOUT dropping duplicates.
    Used when we want grouping info but keep all images.
    """
    clusters: List[Tuple[int, int]] = []
    for sample in samples:
        h = dhash(sample.path, hash_size=hash_size)
        if h is None:
            sample.group = -1
            continue
        group = -1
        for gid, rep_hash in clusters:
            if (h ^ rep_hash).bit_count() <= max_distance:
                group = gid
                break
        if group == -1:
            group = len(clusters)
            clusters.append((group, h))
        sample.group = group


# ---------------------------------------------------------------------------
# Source ingestion (CSV-driven with class-folder fallback)
# ---------------------------------------------------------------------------
def _build_image_index(roots: List[str]) -> Dict[str, str]:
    """Map image stem (and basename) -> absolute path across the given roots."""
    index: Dict[str, str] = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    full = os.path.join(dirpath, name)
                    stem = os.path.splitext(name)[0]
                    index.setdefault(stem, full)
                    index.setdefault(name, full)
    return index


def _find_csv(root: str, preferred: List[str]) -> Optional[str]:
    for name in preferred:
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(".csv"):
            return os.path.join(root, name)
    return None


def _pick_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def ingest_csv_source(
    root: str,
    source: str,
    csv_names: List[str],
    image_cols: List[str],
    label_cols: List[str],
    label_mapper=map_label,
) -> List[Sample]:
    """Parse a metadata CSV and resolve each row to an image + canonical class."""
    samples: List[Sample] = []
    if not root or not os.path.isdir(root):
        return samples

    csv_path = _find_csv(root, csv_names)
    if csv_path is None:
        return samples

    index = _build_image_index([root])
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        img_col = _pick_column(fields, image_cols)
        lbl_col = _pick_column(fields, label_cols)
        if img_col is None or lbl_col is None:
            print(f"[Harmonize] {source}: could not resolve columns in {csv_path} "
                  f"(fields={fields[:8]})")
            return samples

        for row in reader:
            raw_name = str(row.get(img_col, "")).strip()
            raw_label = str(row.get(lbl_col, "")).strip()
            if not raw_name:
                continue
            stem = os.path.splitext(os.path.basename(raw_name))[0]
            path = index.get(stem) or index.get(os.path.basename(raw_name))
            if not path:
                continue
            canonical = label_mapper(raw_label)
            if canonical is None:
                continue
            samples.append(Sample(path=path, raw_label=raw_label,
                                  canonical=canonical, source=source))
    return samples


def ingest_fitzpatrick17k(root: str) -> List[Sample]:
    return ingest_csv_source(
        root, "fitzpatrick17k",
        csv_names=["fitzpatrick17k.csv", "metadata.csv"],
        image_cols=["md5hash", "md5", "image", "filename", "img"],
        label_cols=["label", "disease", "diagnosis"],
        label_mapper=map_label,
    )


def ingest_scin(root: str) -> List[Sample]:
    return ingest_csv_source(
        root, "scin",
        csv_names=["SCIN_dataset.csv", "scin.csv", "metadata.csv"],
        image_cols=["image", "filename", "image_id", "path"],
        label_cols=["disease", "label", "diagnosis", "condition"],
        label_mapper=map_label,
    )


def ingest_pad_ufes(root: str) -> List[Sample]:
    return ingest_csv_source(
        root, "pad-ufes-20",
        csv_names=["metadata.csv", "pad_ufes.csv"],
        image_cols=["img_id", "image", "filename"],
        label_cols=["diagnostic", "label", "diagnosis"],
        label_mapper=map_pad_code,
    )


def ingest_class_folders(root: str, source: str = "local") -> List[Sample]:
    """
    Fallback: ingest a pre-organized folder tree whose subfolders are raw labels
    or canonical class names.
    """
    samples: List[Sample] = []
    if not root or not os.path.isdir(root):
        return samples

    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        canonical = map_label(entry)
        if canonical is None and entry.lower() in CLASS_NAMES:
            canonical = entry.lower()
        if canonical is None:
            continue
        for dirpath, _, filenames in os.walk(sub):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    samples.append(Sample(path=os.path.join(dirpath, name),
                                          raw_label=entry, canonical=canonical,
                                          source=source))
    return samples


# ---------------------------------------------------------------------------
# Split & materialization
# ---------------------------------------------------------------------------
def class_counts(samples: List[Sample]) -> Dict[str, int]:
    """Per-canonical-class sample counts."""
    counts = {name: 0 for name in CLASS_NAMES}
    for s in samples:
        counts[s.canonical] = counts.get(s.canonical, 0) + 1
    return counts


def stratified_grouped_split(
    samples: List[Sample],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42
) -> Dict[str, List[Sample]]:
    """
    Stratified split by canonical class that keeps near-duplicate groups
    (identical `group` ids) together, preventing train/test leakage.
    """
    rng = random.Random(seed)
    by_class: Dict[str, List[Sample]] = {name: [] for name in CLASS_NAMES}
    for s in samples:
        by_class.setdefault(s.canonical, []).append(s)

    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    train_r, val_r, _ = ratios

    for cls_samples in by_class.values():
        if not cls_samples:
            continue
        groups: Dict[int, List[Sample]] = {}
        for s in cls_samples:
            groups.setdefault(s.group, []).append(s)

        gids = list(groups.keys())
        rng.shuffle(gids)

        n = len(cls_samples)
        n_train = int(n * train_r)
        n_val = int(n * (train_r + val_r))

        cursor = 0
        for gid in gids:
            members = groups[gid]
            if cursor < n_train:
                splits["train"].extend(members)
            elif cursor < n_val:
                splits["val"].extend(members)
            else:
                splits["test"].extend(members)
            cursor += len(members)

    return splits


def materialize(
    splits: Dict[str, List[Sample]],
    out_dir: str,
    dry_run: bool = False,
    write_manifest: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Copy each sample into out_dir/<split>/<canonical>/<filename> and optionally
    write a manifest CSV recording provenance.
    """
    summary: Dict[str, Dict[str, int]] = {}
    manifest_rows: List[Dict[str, Any]] = []

    for split, samples in splits.items():
        counts = {name: 0 for name in CLASS_NAMES}
        for s in samples:
            counts[s.canonical] = counts.get(s.canonical, 0) + 1
            dest_dir = os.path.join(out_dir, split, s.canonical)
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, os.path.basename(s.path))
                if not os.path.exists(dest):
                    try:
                        shutil.copy2(s.path, dest)
                    except OSError as exc:
                        print(f"[Harmonize] copy failed {s.path}: {exc}")
            manifest_rows.append({
                "split": split, "canonical": s.canonical,
                "source": s.source, "group": s.group,
                "original_path": s.path, "raw_label": s.raw_label,
            })
        summary[split] = counts

    if write_manifest and not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, "manifest.csv")
        fieldnames = (list(manifest_rows[0].keys()) if manifest_rows else
                      ["split", "canonical", "source", "group", "original_path", "raw_label"])
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"[Harmonize] Wrote manifest to {manifest_path}")

    return summary


def harmonize(
    sources: Dict[str, Optional[str]],
    out_dir: str = "data",
    dedupe: bool = True,
    seed: int = 42,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    End-to-end pipeline: ingest all sources -> map -> dedupe -> split -> materialize.
    """
    all_samples: List[Sample] = []

    if sources.get("fitzpatrick_dir"):
        got = ingest_fitzpatrick17k(sources["fitzpatrick_dir"])
        print(f"[Harmonize] Fitzpatrick17k: {len(got)} mapped images")
        all_samples.extend(got)

    if sources.get("scin_dir"):
        got = ingest_scin(sources["scin_dir"])
        print(f"[Harmonize] SCIN: {len(got)} mapped images")
        all_samples.extend(got)

    if sources.get("pad_dir"):
        got = ingest_pad_ufes(sources["pad_dir"])
        print(f"[Harmonize] PAD-UFES-20: {len(got)} mapped images")
        all_samples.extend(got)

    if sources.get("local_dir"):
        got = ingest_class_folders(sources["local_dir"], source="local")
        print(f"[Harmonize] Local folders: {len(got)} mapped images")
        all_samples.extend(got)

    if not all_samples:
        print("[Harmonize] No source images found. Provide valid --*_dir paths.")
        return {"samples": 0, "splits": {}}

    before = len(all_samples)
    if dedupe:
        all_samples = deduplicate(all_samples)
        print(f"[Harmonize] Deduplicated: {before} -> {len(all_samples)} images")
    else:
        assign_groups_by_hash(all_samples)

    print(f"[Harmonize] Class distribution: {class_counts(all_samples)}")

    splits = stratified_grouped_split(all_samples, seed=seed)
    summary = materialize(splits, out_dir, dry_run=dry_run)

    print("\n===== Harmonization Summary =====")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {summary.get(split, {})}")
    print(f"  total unique: {len(all_samples)}")

    return {"samples": len(all_samples), "splits": summary}


def main():
    parser = argparse.ArgumentParser(description="Harmonize clinical skin datasets into 6 classes")
    parser.add_argument("--fitzpatrick_dir", type=str, default=None)
    parser.add_argument("--scin_dir", type=str, default=None)
    parser.add_argument("--pad_dir", type=str, default=None)
    parser.add_argument("--local_dir", type=str, default=None,
                        help="Pre-organized folder tree (subfolders = class labels)")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable perceptual deduplication")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Report only; write nothing")
    args = parser.parse_args()

    harmonize(
        sources={
            "fitzpatrick_dir": args.fitzpatrick_dir,
            "scin_dir": args.scin_dir,
            "pad_dir": args.pad_dir,
            "local_dir": args.local_dir,
        },
        out_dir=args.out_dir,
        dedupe=not args.no_dedupe,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()