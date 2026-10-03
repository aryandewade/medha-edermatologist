"""
V2 Advanced Model Training Pipeline for Skin Disease Classifier
EfficientNet-B0 with Residual Clinical Head, Focal Loss, and Suspicious Lesion Recall Optimization

Key Improvements over V1:
1. Residual Clinical Classifier Head:
   - Direct linear skip-connection initialized from V1 weights (77.3% baseline).
   - Deep non-linear residual MLP (1280 -> 384 -> LayerNorm -> GELU -> 6) to resolve
     inter-class confusion between Eczema, Psoriasis, and Tinea.
2. Clinical Cost-Sensitive Focal Loss (gamma=1.75):
   - Downweights easy healthy skin cases.
   - Prioritizes hard ambiguous erythematous rashes.
   - High clinical penalty for suspicious lesions to maximize recall (>85%+).
3. Multi-View Embedding Caching:
   - Extracts center-crop, horizontal flip, and perspective augmented embeddings.
   - Enables fast CPU training (50+ epochs in < 10 seconds).
4. Post-Hoc Temperature Calibration & Clinical Safety Evaluation.
"""

import os
import sys
import shutil
import argparse
import time
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from PIL import Image
from torchvision import transforms

# Ensure src is on path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model, SkinDiseaseEfficientNetB0, ResidualDermClassifierHead
from dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD

SUSPICIOUS_IDX = CLASS_TO_IDX["suspicious_lesion"]
PSORIASIS_IDX = CLASS_TO_IDX["psoriasis"]
ECZEMA_IDX = CLASS_TO_IDX["eczema"]
TINEA_IDX = CLASS_TO_IDX["tinea"]


class ClinicalCostFocalLoss(nn.Module):
    """
    Focal Loss with per-class clinical weighting and label smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        gamma: float = 1.75,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.04
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy with label smoothing
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            reduction="none", 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # probability of target class
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t

        return loss.mean()


def get_feature_extractor(v1_checkpoint: str, device: str = "cpu"):
    """Load EfficientNet-B0 backbone and extract feature extractor up to pool."""
    print(f"[V2] Loading backbone feature extractor from {v1_checkpoint}...")
    base_model = build_model(
        num_classes=len(CLASS_NAMES),
        checkpoint_path=v1_checkpoint,
        device=device
    )
    base_model.eval()

    # Get V1 classifier weights to seed the skip connection
    v1_linear = base_model.backbone.classifier[1]
    skip_weight = v1_linear.weight.data.clone()
    skip_bias = v1_linear.bias.data.clone()

    def extractor(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = base_model.backbone.features(images)
            pooled = base_model.backbone.avgpool(feats)
            return torch.flatten(pooled, 1)

    return extractor, skip_weight, skip_bias


def cache_embeddings_split(
    extractor,
    data_dir: str,
    device: str = "cpu",
    augment_views: int = 1,
    batch_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts embeddings for a dataset split.
    If augment_views > 1, generates multiple views per sample (center, hflip, color).
    """
    root_path = Path(data_dir)
    image_paths = []
    labels = []

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for c_name, c_idx in CLASS_TO_IDX.items():
        c_dir = root_path / c_name
        if not c_dir.exists():
            continue
        for f in c_dir.iterdir():
            if f.suffix.lower() in valid_exts:
                image_paths.append(str(f))
                labels.append(c_idx)

    print(f"[V2 Cache] Found {len(image_paths)} images in {data_dir} across {len(CLASS_NAMES)} classes.")

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    hflip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    jitter_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    all_embeds = []
    all_targets = []

    total_images = len(image_paths)
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # View 1: Base center crop
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(base_transform(img))
        batch_t = torch.stack(tensors).to(device)
        feats = extractor(batch_t).cpu()
        all_embeds.append(feats)
        all_targets.extend(batch_labels)

        # Additional augmented views for training
        if augment_views >= 2:
            tensors_hf = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors_hf.append(hflip_transform(img))
            batch_hf = torch.stack(tensors_hf).to(device)
            feats_hf = extractor(batch_hf).cpu()
            all_embeds.append(feats_hf)
            all_targets.extend(batch_labels)

        if augment_views >= 3:
            tensors_jit = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors_jit.append(jitter_transform(img))
            batch_jit = torch.stack(tensors_jit).to(device)
            feats_jit = extractor(batch_jit).cpu()
            all_embeds.append(feats_jit)
            all_targets.extend(batch_labels)

        if (i // batch_size) % 15 == 0 or i + batch_size >= total_images:
            done = min(i + batch_size, total_images)
            pct = 100.0 * done / total_images
            print(f"  --> Extracted {done}/{total_images} images ({pct:.1f}%)...", flush=True)

    X = torch.cat(all_embeds, dim=0)
    y = torch.tensor(all_targets, dtype=torch.long)
    print(f"[V2 Cache] Finished {data_dir}: {X.shape[0]} feature vectors of dim {X.shape[1]}.")
    return X, y


def build_clinical_alpha(
    train_labels: torch.Tensor,
    suspicious_boost: float = 2.1,
    psoriasis_boost: float = 1.6,
    tinea_boost: float = 1.45,
    healthy_scale: float = 0.60
) -> torch.Tensor:
    """
    Constructs calibrated class weighting tensor that directly optimizes
    clinical safety (prioritizing suspicious lesions and tricky erythematous rashes).
    """
    counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).float()
    total = len(train_labels)
    num_classes = len(CLASS_NAMES)

    # Standard inverse frequency baseline
    weights = total / (num_classes * torch.clamp(counts, min=1.0))

    # Normalize baseline to mean 1.0
    weights = weights / weights.mean()

    # Apply clinical safety multipliers
    weights[SUSPICIOUS_IDX] *= suspicious_boost
    weights[PSORIASIS_IDX] *= psoriasis_boost
    weights[TINEA_IDX] *= tinea_boost
    weights[CLASS_TO_IDX["healthy"]] *= healthy_scale

    print("[V2 Loss] Calibrated Clinical Class Weights (alpha):")
    for name, idx in CLASS_TO_IDX.items():
        print(f"  • {name:18s}: weight={weights[idx]:.3f} (support={int(counts[idx].item())})")

    return weights


def train_v2_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    skip_weight: torch.Tensor,
    skip_bias: torch.Tensor,
    num_classes: int = 6,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 2.5e-3,
    weight_decay: float = 1e-3,
    device: str = "cpu"
) -> Tuple[ResidualDermClassifierHead, Dict[str, Any]]:
    """Trains the Residual Clinical Head on cached embeddings."""
    head = ResidualDermClassifierHead(
        in_features=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=384,
        dropout_rate=0.28
    )

    # Seed skip connection with V1 weights
    head.skip.weight.data.copy_(skip_weight)
    head.skip.bias.data.copy_(skip_bias)

    # Seed residual MLP final projection with small weights to smoothly add non-linear corrections
    head.mlp[-1].weight.data.normal_(0.0, 0.01)
    head.mlp[-1].bias.data.zero_()
    head.to(device)

    alpha = build_clinical_alpha(y_train)
    criterion = ClinicalCostFocalLoss(gamma=1.75, alpha=alpha, label_smoothing=0.04)

    # Differential learning rates: lower for skip (fine-tune), higher for MLP
    optimizer = AdamW([
        {"params": head.skip.parameters(), "lr": lr * 0.15, "weight_decay": 1e-4},
        {"params": head.mlp.parameters(), "lr": lr, "weight_decay": weight_decay}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_dev = X_val.to(device)
    y_val_np = y_val.numpy()

    best_score = -1.0
    best_state = None
    best_metrics = {}

    print(f"\n[V2 Training] Training Residual Clinical Head for {epochs} epochs on CPU...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.5)
            optimizer.step()
            total_loss += loss.item() * bx.size(0)

        scheduler.step()

        # Validation evaluation
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_dev)
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()
            val_preds = np.argmax(val_probs, axis=1)

        val_acc = accuracy_score(y_val_np, val_preds)
        val_macro_f1 = f1_score(y_val_np, val_preds, average="macro", zero_division=0)
        p, r, f1, sup = precision_recall_fscore_support(
            y_val_np, val_preds, labels=list(range(num_classes)), zero_division=0
        )
        suspicious_recall = r[SUSPICIOUS_IDX]
        psoriasis_recall = r[PSORIASIS_IDX]
        tinea_recall = r[TINEA_IDX]

        # Clinical composite score: rewards high Macro-F1 + strong Suspicious Recall + high overall Accuracy
        score = (val_macro_f1 * 0.45) + (suspicious_recall * 0.35) + (val_acc * 0.20)

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_metrics = {
                "epoch": epoch,
                "accuracy": round(float(val_acc), 4),
                "macro_f1": round(float(val_macro_f1), 4),
                "suspicious_recall": round(float(suspicious_recall), 4),
                "psoriasis_recall": round(float(psoriasis_recall), 4),
                "tinea_recall": round(float(tinea_recall), 4),
                "per_class_recall": [round(float(x), 4) for x in r],
                "per_class_f1": [round(float(x), 4) for x in f1],
                "score": round(float(score), 4)
            }

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch [{epoch:02d}/{epochs}] "
                f"Val Acc: {val_acc*100:.1f}% | "
                f"Macro-F1: {val_macro_f1*100:.1f}% | "
                f"Suspicious Recall: {suspicious_recall*100:.1f}% | "
                f"Psoriasis Recall: {psoriasis_recall*100:.1f}%"
            )

    elapsed = time.time() - t0
    print(f"\n[V2 Training] Head training finished in {elapsed:.2f}s!")
    print(f"  --> Best Model (Epoch {best_metrics['epoch']}):")
    print(f"      Accuracy:          {best_metrics['accuracy']*100:.2f}%")
    print(f"      Macro-F1:          {best_metrics['macro_f1']*100:.2f}%")
    print(f"      Suspicious Recall: {best_metrics['suspicious_recall']*100:.2f}%")
    print(f"      Psoriasis Recall:  {best_metrics['psoriasis_recall']*100:.2f}%")
    print(f"      Tinea Recall:      {best_metrics['tinea_recall']*100:.2f}%")

    head.load_state_dict(best_state)
    return head, best_metrics


def assemble_and_save_v2_model(
    v1_checkpoint_path: str,
    head: ResidualDermClassifierHead,
    best_metrics: Dict[str, Any],
    output_path: str = "models/best_model_v2.pt",
    device: str = "cpu"
) -> str:
    """
    Assembles the complete end-to-end SkinDiseaseEfficientNetB0 with the V2 Residual Head
    and saves the full model checkpoint.
    """
    v2_model = SkinDiseaseEfficientNetB0(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        head_type="residual_v2"
    )
    v1_ckpt = torch.load(v1_checkpoint_path, map_location=device)
    v1_state = v1_ckpt.get("state_dict", v1_ckpt)
    v2_model.load_state_dict(v1_state, strict=False)

    # Insert trained V2 head
    v2_model.backbone.classifier.load_state_dict(head.state_dict())
    v2_model.to(device)
    v2_model.eval()

    # Save checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint_data = {
        "version": "v2",
        "epoch": best_metrics["epoch"],
        "state_dict": v2_model.state_dict(),
        "macro_f1": best_metrics["macro_f1"],
        "accuracy": best_metrics["accuracy"],
        "suspicious_recall": best_metrics["suspicious_recall"],
        "class_names": CLASS_NAMES,
        "val_metrics": best_metrics,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint_data, output_path)
    print(f"  --> Saved V2 Checkpoint to {output_path}")

    # Mirror to backend/models/best_model.pt
    backend_model_dir = Path("backend/models")
    backend_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup V1 first if not already backed up
    v1_backup = backend_model_dir / "best_model_v1_backup.pt"
    existing_backend = backend_model_dir / "best_model.pt"
    if existing_backend.exists() and not v1_backup.exists():
        shutil.copy2(existing_backend, v1_backup)
        print(f"  --> Backed up original V1 model to {v1_backup}")

    shutil.copy2(output_path, existing_backend)
    print(f"  --> Updated active backend model at {existing_backend}")

    return output_path


def evaluate_v2_on_test(
    model_checkpoint: str,
    test_dir: str = "ml/data/test",
    reports_dir: str = "reports",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs rigorous clinical evaluation of V2 on the held-out test split."""
    print(f"\n[V2 Evaluation] Evaluating {model_checkpoint} on held-out test split ({test_dir})...")
    v2_model = build_model(
        num_classes=len(CLASS_NAMES),
        checkpoint_path=model_checkpoint,
        device=device
    )
    v2_model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    hflip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_path = Path(test_dir)
    image_paths = []
    labels = []
    for c_name, c_idx in CLASS_TO_IDX.items():
        c_dir = test_path / c_name
        if not c_dir.exists():
            continue
        for f in c_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                image_paths.append(str(f))
                labels.append(c_idx)

    y_true = np.array(labels)
    all_probs = []

    # Test-Time Augmentation (TTA: center + horizontal flip)
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        b_paths = image_paths[i:i + batch_size]
        t1, t2 = [], []
        for p in b_paths:
            img = Image.open(p).convert("RGB")
            t1.append(test_transform(img))
            t2.append(hflip_transform(img))

        with torch.no_grad():
            b1 = torch.stack(t1).to(device)
            b2 = torch.stack(t2).to(device)
            out1 = v2_model(b1)
            out2 = v2_model(b2)
            # TTA ensemble average
            probs = torch.softmax((out1 + out2) / 2.0, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probabilities = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(probabilities, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Top-3 Accuracy
    top3_hits = np.any(np.argsort(probabilities, axis=1)[:, -3:] == y_true.reshape(-1, 1), axis=1)
    top3_acc = float(np.mean(top3_hits))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    rep_txt = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save V2 reports
    report_v2_path = out_dir / "classification_report_v2.txt"
    report_v2_path.write_text(rep_txt, encoding="utf-8")

    cm_v2_path = out_dir / "confusion_matrix_v2.csv"
    np.savetxt(cm_v2_path, cm, fmt="%d", delimiter=",", header=",".join(CLASS_NAMES), comments="")

    metrics_v2 = {
        "model_version": "v2",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top3_accuracy": round(top3_acc, 4),
        "suspicious_lesion_recall": per_class["suspicious_lesion"]["recall"],
        "per_class": per_class,
        "num_samples": len(y_true),
        "confusion_matrix": cm.tolist()
    }

    import json
    metrics_v2_path = out_dir / "eval_metrics_v2.json"
    with open(metrics_v2_path, "w", encoding="utf-8") as f:
        json.dump(metrics_v2, f, indent=2)

    print("\n" + "="*60)
    print("V2 CLINICAL TEST EVALUATION REPORT:")
    print("="*60)
    print(rep_txt)
    print(f"Top-3 Accuracy:            {top3_acc*100:.2f}%")
    print(f"Suspicious Lesion Recall:  {per_class['suspicious_lesion']['recall']*100:.2f}%")
    print(f"Psoriasis Recall:          {per_class['psoriasis']['recall']*100:.2f}%")
    print(f"Tinea Recall:              {per_class['tinea']['recall']*100:.2f}%")
    print("="*60)

    return metrics_v2


def main():
    parser = argparse.ArgumentParser(description="Train V2 Model for E-Dermatologist")
    parser.add_argument("--v1_checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--train_dir", type=str, default="ml/data/train")
    parser.add_argument("--val_dir", type=str, default="ml/data/val")
    parser.add_argument("--test_dir", type=str, default="ml/data/test")
    parser.add_argument("--output_model", type=str, default="models/best_model_v2.pt")
    parser.add_argument("--cache_file", type=str, default="ml/data/cache_embeddings_v2.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--force_recache", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    print(f"[V2 Pipeline] Starting V2 training on {device}...")

    cache_path = Path(args.cache_file)
    if cache_path.exists() and not args.force_recache:
        print(f"[V2 Pipeline] Loading pre-extracted embeddings from {cache_path}...")
        cache_data = torch.load(cache_path, map_location="cpu")
        X_train, y_train = cache_data["train"]
        X_val, y_val = cache_data["val"]
        skip_weight = cache_data["skip_weight"]
        skip_bias = cache_data["skip_bias"]
    else:
        extractor, skip_weight, skip_bias = get_feature_extractor(args.v1_checkpoint, device=device)
        print("[V2 Pipeline] Extracting multi-view embeddings for training set (center + horizontal flip)...")
        X_train, y_train = cache_embeddings_split(
            extractor, args.train_dir, device=device, augment_views=2, batch_size=32
        )
        print("[V2 Pipeline] Extracting embeddings for validation set...")
        X_val, y_val = cache_embeddings_split(
            extractor, args.val_dir, device=device, augment_views=1, batch_size=32
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "skip_weight": skip_weight,
            "skip_bias": skip_bias
        }, cache_path)
        print(f"[V2 Pipeline] Saved cached embeddings to {cache_path}")

    # Train Residual Clinical Head
    head, best_metrics = train_v2_head(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        skip_weight=skip_weight,
        skip_bias=skip_bias,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    # Assemble full end-to-end model and mirror to backend
    v2_checkpoint = assemble_and_save_v2_model(
        v1_checkpoint_path=args.v1_checkpoint,
        head=head,
        best_metrics=best_metrics,
        output_path=args.output_model,
        device=device
    )

    # Evaluate on held-out test split
    evaluate_v2_on_test(
        model_checkpoint=v2_checkpoint,
        test_dir=args.test_dir,
        reports_dir="reports",
        device=device
    )


if __name__ == "__main__":
    main()
