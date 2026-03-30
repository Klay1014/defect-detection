"""MVTec AD Dataset — PyTorch Dataset with OpenCV preprocessing."""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── OpenCV preprocessing pipeline ────────────────────────────────────────────

def opencv_preprocess(image_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    """OpenCV-based preprocessing: resize + denoise + normalize.

    Args:
        image_bgr: Raw BGR image from cv2.imread.
        size: Target spatial size.

    Returns:
        RGB float32 image, shape (size, size, 3), range [0, 1].
    """
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# ── Albumentations augmentation pipeline ────────────────────────────────────

def get_augmentation_pipeline() -> A.Compose:
    """Data augmentation for training robustness.

    Conservative augmentations suitable for anomaly detection:
    - We don't want augmentations that create false "defects"
    - Focus on geometric + lighting variation
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.3),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """MVTec AD dataset for one category.

    Directory structure expected:
        data/{category}/train/good/       — normal training images
        data/{category}/test/good/        — normal test images
        data/{category}/test/{defect}/    — defective test images
        data/{category}/ground_truth/{defect}/  — pixel-level masks
    """

    def __init__(
        self,
        data_root: str,
        category: str,
        split: str = "train",
        size: int = 256,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.category = category
        self.split = split
        self.size = size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.augmentation = get_augmentation_pipeline() if augment else None

        self.images: List[str] = []
        self.labels: List[int] = []      # 0 = normal, 1 = anomaly
        self.mask_paths: List[Optional[str]] = []

        cat_dir = self.data_root / category

        if split == "train":
            good_dir = cat_dir / "train" / "good"
            for img_name in sorted(os.listdir(good_dir)):
                if img_name.endswith(".png"):
                    self.images.append(str(good_dir / img_name))
                    self.labels.append(0)
                    self.mask_paths.append(None)
        else:
            test_dir = cat_dir / "test"
            gt_dir = cat_dir / "ground_truth"
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = test_dir / defect_type
                if not defect_dir.is_dir():
                    continue
                is_good = defect_type == "good"
                for img_name in sorted(os.listdir(defect_dir)):
                    if not img_name.endswith(".png"):
                        continue
                    self.images.append(str(defect_dir / img_name))
                    self.labels.append(0 if is_good else 1)
                    if is_good:
                        self.mask_paths.append(None)
                    else:
                        mask_name = img_name.replace(".png", "_mask.png")
                        self.mask_paths.append(
                            str(gt_dir / defect_type / mask_name)
                        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Returns (image_tensor, label, mask_tensor)."""
        # Load with OpenCV
        img_bgr = cv2.imread(self.images[idx])
        img = opencv_preprocess(img_bgr, self.size)

        # Apply augmentation (training only)
        if self.augmentation is not None:
            augmented = self.augmentation(image=(img * 255).astype(np.uint8))
            img = augmented["image"].astype(np.float32) / 255.0

        img_tensor = self.transform(img)

        label = self.labels[idx]

        # Load mask (for test set anomalies)
        if self.mask_paths[idx] is not None:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.size, self.size),
                              interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.tensor(mask / 255.0, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(self.size, self.size, dtype=torch.float32)

        return img_tensor, label, mask_tensor


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = MVTecDataset("data", "bottle", split="train")
    print(f"Train: {len(ds)} images")
    img, label, mask = ds[0]
    print(f"  shape={img.shape}, label={label}, mask_sum={mask.sum():.0f}")

    ds_test = MVTecDataset("data", "bottle", split="test")
    print(f"Test:  {len(ds_test)} images")
    # Find first anomaly
    for i in range(len(ds_test)):
        img, label, mask = ds_test[i]
        if label == 1:
            print(f"  Anomaly #{i}: shape={img.shape}, mask_sum={mask.sum():.0f}")
            break
