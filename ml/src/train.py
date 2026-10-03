"""
Training and Fine-Tuning Pipeline for EfficientNet-B0 Skin Disease Classifier

Two-Stage Transfer Learning:
  Stage 1 (Warmup): Frozen backbone, train classifier head (2-3 epochs).
  Stage 2 (Fine-tuning): Unfreeze backbone with differential learning rates, 
                         cosine annealing schedule, label smoothing, and class weights.
"""

import os
import shutil
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, classification_report

from model import build_model, SkinDiseaseEfficientNetB0
from dataset import create_dataloaders, CLASS_NAMES


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    total_batches = len(dataloader)
    for b_idx, (images, targets) in enumerate(dataloader, 1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        if b_idx % 25 == 0 or b_idx == total_batches:
            print(f"    Batch [{b_idx}/{total_batches}] - Loss: {loss.item():.4f}", flush=True)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model on validation split."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=-1)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    total = max(len(dataloader.dataset), 1)
    val_loss = running_loss / total
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=CLASS_NAMES, 
        output_dict=True,
        zero_division=0
    )

    return {
        "loss": val_loss,
        "accuracy": val_acc,
        "macro_f1": val_macro_f1,
        "report": report
    }


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Skin Disease Classifier")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Path to training images folder")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Path to validation images folder")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Head warmup epochs")
    parser.add_argument("--finetune_epochs", type=int, default=20, help="Full fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr_head", type=float, default=3e-4, help="Learning rate for head")
    parser.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate for backbone")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[Training] Using compute device: {device}")

    # 1. Load Data
    train_loader, val_loader, class_weights = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size
    )
    class_weights = class_weights.to(device)

    # 2. Build Model
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True, device=args.device)

    # Criterion with class weighting and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    best_macro_f1 = 0.0
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    # ==========================================
    # STAGE 1: Classifier Warmup (Frozen Backbone)
    # ==========================================
    print(f"\n--- STAGE 1: Head Warmup ({args.warmup_epochs} epochs) ---")
    model.freeze_backbone()
    optimizer = AdamW(model.backbone.classifier.parameters(), lr=args.lr_head, weight_decay=1e-2)

    for epoch in range(1, args.warmup_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Warmup Epoch [{epoch}/{args.warmup_epochs}] - Train Loss: {loss:.4f} Acc: {acc:.3f} | Val Loss: {val_metrics['loss']:.4f} Macro-F1: {val_metrics['macro_f1']:.3f}")

    # ==========================================
    # STAGE 2: End-to-End Fine-Tuning
    # ==========================================
    print(f"\n--- STAGE 2: End-to-End Fine-Tuning ({args.finetune_epochs} epochs) ---")
    model.unfreeze_backbone()

    # Differential learning rate: lower for backbone, higher for head
    param_groups = [
        {"params": model.backbone.features.parameters(), "lr": args.lr_backbone},
        {"params": model.backbone.classifier.parameters(), "lr": args.lr_head}
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.finetune_epochs, eta_min=1e-6)

    for epoch in range(1, args.finetune_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_f1 = val_metrics["macro_f1"]
        print(f"Epoch [{epoch}/{args.finetune_epochs}] - Train Loss: {loss:.4f} Acc: {acc:.3f} | Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} Macro-F1: {current_f1:.3f}")

        # Checkpoint if best Macro-F1 achieved
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "macro_f1": best_macro_f1,
                "class_names": CLASS_NAMES,
                "val_report": val_metrics["report"]
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Macro-F1: {best_macro_f1:.4f})")

            # Mirror to backend/models/best_model.pt
            backend_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "models")
            os.makedirs(backend_models_dir, exist_ok=True)
            shutil.copy2(best_checkpoint_path, os.path.join(backend_models_dir, "best_model.pt"))
            print(f"  --> Mirrored checkpoint to backend/models/best_model.pt")

    print(f"\nTraining Complete. Best Validation Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
