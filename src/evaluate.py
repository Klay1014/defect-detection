"""Anomaly detection evaluation — Image-level & Pixel-level AUROC."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from tqdm import tqdm

from dataset import MVTecDataset
from model import AnomalyAutoencoder


def compute_anomaly_maps(
    model: AnomalyAutoencoder,
    loader: DataLoader,
    device: str,
) -> tuple:
    """Compute pixel-wise reconstruction error for all test images.

    Returns:
        anomaly_maps: list of (H, W) anomaly score arrays
        labels: list of image-level labels (0=normal, 1=anomaly)
        gt_masks: list of (H, W) ground truth masks
    """
    model.eval()
    anomaly_maps = []
    labels = []
    gt_masks = []

    with torch.no_grad():
        for imgs, lbls, masks in tqdm(loader, desc="Computing anomaly maps"):
            imgs = imgs.to(device)
            recon = model(imgs)

            # Pixel-wise MSE as anomaly score
            error = (imgs - recon) ** 2
            # Average across RGB channels -> (B, H, W)
            error_map = error.mean(dim=1).cpu().numpy()

            for i in range(len(lbls)):
                anomaly_maps.append(error_map[i])
                labels.append(lbls[i].item())
                gt_masks.append(masks[i].numpy())

    return anomaly_maps, labels, gt_masks


def evaluate(
    data_root: str = str(PROJECT_ROOT / "data"),
    category: str = "bottle",
    checkpoint: str = None,
    size: int = 256,
    device: str = "auto",
):
    # Device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Load model
    model = AnomalyAutoencoder(pretrained=False).to(device)
    if checkpoint is None:
        checkpoint = str(PROJECT_ROOT / "checkpoints" / f"{category}_best.pt")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})")

    # Test data
    test_ds = MVTecDataset(data_root, category, split="test", size=size)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} images")

    # Compute anomaly maps
    anomaly_maps, labels, gt_masks = compute_anomaly_maps(model, test_loader, device)

    # ── Image-level AUROC ────────────────────────────────────────────────
    # Image anomaly score = max pixel error in the anomaly map
    image_scores = np.array([amap.max() for amap in anomaly_maps])
    labels = np.array(labels)

    img_auroc = roc_auc_score(labels, image_scores)

    # Optimal threshold via Youden's J statistic
    fpr, tpr, thresholds = roc_curve(labels, image_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    preds = (image_scores >= best_threshold).astype(int)
    img_f1 = f1_score(labels, preds)

    print(f"\n{'='*50}")
    print(f"Image-level Results ({category})")
    print(f"{'='*50}")
    print(f"  AUROC:     {img_auroc:.4f}")
    print(f"  F1:        {img_f1:.4f}")
    print(f"  Threshold: {best_threshold:.6f} (Youden's J)")
    print(f"  Normal:    {(labels == 0).sum()} | Anomaly: {(labels == 1).sum()}")

    # ── Pixel-level AUROC ────────────────────────────────────────────────
    # Only compute for images that have ground truth masks
    pixel_scores_all = []
    pixel_labels_all = []

    for amap, gt in zip(anomaly_maps, gt_masks):
        if gt.max() > 0:  # has ground truth mask
            pixel_scores_all.append(amap.flatten())
            pixel_labels_all.append((gt > 0.5).astype(int).flatten())

    if pixel_scores_all:
        pixel_scores_all = np.concatenate(pixel_scores_all)
        pixel_labels_all = np.concatenate(pixel_labels_all)
        pix_auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
        print(f"\nPixel-level Results ({category})")
        print(f"{'='*50}")
        print(f"  AUROC:     {pix_auroc:.4f}")
        print(f"  Pixels:    {len(pixel_scores_all):,}")
    else:
        pix_auroc = None
        print("\n  No pixel-level ground truth available.")

    return {
        "category": category,
        "image_auroc": img_auroc,
        "image_f1": img_f1,
        "pixel_auroc": pix_auroc,
        "threshold": best_threshold,
        "anomaly_maps": anomaly_maps,
        "labels": labels,
        "gt_masks": gt_masks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate(
        data_root=args.data_root,
        category=args.category,
        checkpoint=args.checkpoint,
        device=args.device,
    )
