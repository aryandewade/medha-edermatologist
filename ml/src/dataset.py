"""
PyTorch Dataset and DataLoaders for 6-Class Mobile Skin Disease Classification
Supported Classes: Eczema, Psoriasis, Tinea, Acne, Healthy, Suspicious Lesion
"""

import os
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


CLASS_NAMES = [
    "eczema", 
    "psoriasis", 
    "tinea", 
    "acne", 
    "healthy", 
    "suspicious_lesion"
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = True, image_size: int = 224) -> transforms.Compose:
    """
    Returns image transformation pipeline for mobile smartphone photography.
    Applies illumination shifts, rotation, and affine perspective variations.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class SkinDiseaseDataset(Dataset):
    """
    PyTorch Dataset loading images from class subfolders:
       data_dir/
         eczema/
         psoriasis/
         tinea/
         acne/
         healthy/
         suspicious_lesion/
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[str, int]]] = None,
        data_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
        image_size: int = 224
    ):
        self.transform = transform or get_transforms(is_train=is_train, image_size=image_size)
        self.samples: List[Tuple[str, int]] = []

        if samples is not None:
            self.samples = samples
        elif data_dir is not None:
            self._load_from_dir(data_dir)

    def _load_from_dir(self, root_dir: str):
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        root_path = Path(root_dir)
        
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = root_path / class_name
            if not class_folder.exists():
                continue
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_file), class_idx))

        print(f"[Dataset] Loaded {len(self.samples)} images from {root_dir} across {len(CLASS_NAMES)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label

    def get_class_counts(self) -> Dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        counts = self.get_class_counts()
        total = max(len(self.samples), 1)
        num_classes = len(CLASS_NAMES)
        weights = []
        for name in CLASS_NAMES:
            c = max(counts[name], 1)
            w = total / (num_classes * c)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    train_dataset = SkinDiseaseDataset(data_dir=train_dir, is_train=True, image_size=image_size)
    val_dataset = SkinDiseaseDataset(data_dir=val_dir, is_train=False, image_size=image_size)

    class_weights = train_dataset.get_class_weights()
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )
    return train_loader, val_loader, class_weights
