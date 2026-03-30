"""PatchCore — SOTA unsupervised anomaly detection via patch-level memory bank.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
Key idea: Build a memory bank of normal patch features, then score test patches
by nearest-neighbor distance. No training required — only feature extraction.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from scipy.ndimage import gaussian_filter
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from tqdm import tqdm

from dataset import MVTecDataset


# ── Feature Extractor ───────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """Extract layer2 (128-ch) + layer3 (256-ch) from pretrained ResNet18.

    Concatenates to 384-dim patch-level features at 32x32 spatial resolution.
    Multi-scale features capture both fine texture and coarse structure.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2   # -> (128, 32, 32)
        self.layer3 = backbone.layer3   # -> (256, 16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        feat2 = self.layer2(x)                          # (B, 128, 32, 32)
        feat3 = self.layer3(feat2)                       # (B, 256, 16, 16)
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[2:],
            mode="bilinear", align_corners=False,
        )                                                # (B, 256, 32, 32)
        return torch.cat([feat2, feat3_up], dim=1)       # (B, 384, 32, 32)


# ── PatchCore Algorithm ─────────────────────────────────────────────────────

@torch.no_grad()
def run_patchcore(
    category: str,
    data_root: str = str(PROJECT_ROOT / "data"),
    size: int = 256,
    subsample_ratio: float = 0.1,
    device: str = "auto",
) -> dict:
    """Run PatchCore: build memory bank from train set, evaluate on test set.

    Args:
        category: MVTec AD category (e.g., "bottle", "carpet", "hazelnut").
        data_root: Path to dataset root.
        size: Image resize dimension.
        subsample_ratio: Fraction of patches to keep in memory bank (0.01-0.1).
        device: Compute device.

    Returns:
        Dictionary with AUROC metrics, anomaly maps, and labels.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    extractor = FeatureExtractor().eval().to(device)

    # ── Build memory bank from normal training patches ──────────────────
    train_ds = MVTecDataset(data_root, category, split="train", size=size)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)

    train_feats = []
    for imgs, _, _ in tqdm(train_loader, desc=f"[{category}] Extracting train features"):
        feats = extractor(imgs.to(device))
        train_feats.append(feats.cpu())
    train_feats = torch.cat(train_feats)  # (N, 384, H, W)
    N, C, H, W = train_feats.shape

    # Reshape to patch-level: (N*H*W, 384)
    memory_bank = train_feats.permute(0, 2, 3, 1).reshape(-1, C)

    # Random coreset subsampling to reduce memory
    n_keep = max(int(memory_bank.shape[0] * subsample_ratio), 2000)
    idx = torch.randperm(memory_bank.shape[0])[:n_keep]
    memory_bank = memory_bank[idx].to(device)
    print(f"[{category}] Memory bank: {memory_bank.shape[0]:,} patches "
          f"(from {N * H * W:,}, kept {subsample_ratio:.0%})")

    # ── Score test images ───────────────────────────────────────────────
    test_ds = MVTecDataset(data_root, category, split="test", size=size)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_maps, all_labels, all_masks = [], [], []

    for imgs, lbls, masks in tqdm(test_loader, desc=f"[{category}] Scoring test images"):
        feats = extractor(imgs.to(device))  # (B, 384, H, W)
        B = feats.shape[0]

        for i in range(B):
            # Per-patch nearest-neighbor distance
            patches = feats[i].permute(1, 2, 0).reshape(-1, C)  # (H*W, 384)
            dists = torch.cdist(patches, memory_bank)             # (H*W, M)
            min_dists = dists.min(dim=1).values.cpu().numpy().reshape(H, W)

            # Upsample to original resolution + Gaussian smoothing
            score_map = np.array(
                Image.fromarray(min_dists).resize((size, size), Image.BILINEAR)
            )
            score_map = gaussian_filter(score_map, sigma=4)

            all_maps.append(score_map)
            all_labels.append(lbls[i].item())
            all_masks.append(masks[i].numpy())

    # ── Compute metrics ─────────────────────────────────────────────────
    labels = np.array(all_labels)
    img_scores = np.array([m.max() for m in all_maps])

    # Image-level AUROC
    img_auroc = roc_auc_score(labels, img_scores)
    fpr, tpr, thresholds = roc_curve(labels, img_scores)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    img_f1 = f1_score(labels, (img_scores >= best_threshold).astype(int))

    # Pixel-level AUROC
    px_scores = np.concatenate([
        m.flatten() for m, g in zip(all_maps, all_masks) if g.max() > 0
    ])
    px_labels = np.concatenate([
        (g > 0.5).astype(int).flatten() for g in all_masks if g.max() > 0
    ])
    pix_auroc = roc_auc_score(px_labels, px_scores)

    print(f"[{category}] Image AUROC: {img_auroc:.4f} | F1: {img_f1:.4f} | "
          f"Pixel AUROC: {pix_auroc:.4f}")

    return {
        "category": category,
        "image_auroc": img_auroc,
        "image_f1": img_f1,
        "pixel_auroc": pix_auroc,
        "threshold": best_threshold,
        "anomaly_maps": all_maps,
        "labels": labels,
        "gt_masks": all_masks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PatchCore anomaly detection")
    parser.add_argument("--category", type=str, default="bottle",
                        help="MVTec AD category")
    parser.add_argument("--data-root", type=str,
                        default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--subsample", type=float, default=0.1,
                        help="Memory bank subsample ratio (default: 0.1)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    result = run_patchcore(
        category=args.category,
        data_root=args.data_root,
        subsample_ratio=args.subsample,
        device=args.device,
    )

    print(f"\n{'='*50}")
    print(f"PatchCore Results — {args.category}")
    print(f"{'='*50}")
    print(f"  Image AUROC:  {result['image_auroc']:.4f}")
    print(f"  Image F1:     {result['image_f1']:.4f}")
    print(f"  Pixel AUROC:  {result['pixel_auroc']:.4f}")
