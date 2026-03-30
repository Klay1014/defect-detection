"""Heatmap visualization — overlay anomaly maps on original images."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import MVTecDataset
from model import AnomalyAutoencoder


def generate_heatmaps(
    data_root: str = str(PROJECT_ROOT / "data"),
    category: str = "bottle",
    checkpoint: str = None,
    output_dir: str = "results/heatmaps",
    n_samples: int = 8,
    size: int = 256,
    device: str = "auto",
):
    """Generate side-by-side visualization: Original | Heatmap | GT Mask."""

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
    model.eval()

    # Test data
    test_ds = MVTecDataset(data_root, category, split="test", size=size)

    # Collect anomaly samples
    anomaly_indices = [i for i in range(len(test_ds)) if test_ds.labels[i] == 1]
    normal_indices = [i for i in range(len(test_ds)) if test_ds.labels[i] == 0]

    # Pick samples: half anomaly, half normal
    n_anom = min(n_samples // 2, len(anomaly_indices))
    n_norm = min(n_samples - n_anom, len(normal_indices))
    indices = anomaly_indices[:n_anom] + normal_indices[:n_norm]

    out_path = Path(output_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img_tensor, label, gt_mask = test_ds[idx]

        # Reconstruct
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            recon = model(inp).cpu().squeeze(0)

        # Anomaly map
        error = ((img_tensor - recon) ** 2).mean(dim=0).numpy()

        # Denormalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = img_tensor.permute(1, 2, 0).numpy() * std + mean
        img_display = np.clip(img_display, 0, 1)

        recon_display = recon.permute(1, 2, 0).numpy() * std + mean
        recon_display = np.clip(recon_display, 0, 1)

        # Col 0: Original
        axes[row, 0].imshow(img_display)
        tag = "ANOMALY" if label == 1 else "NORMAL"
        axes[row, 0].set_title(f"Original ({tag})", fontsize=11)
        axes[row, 0].axis("off")

        # Col 1: Reconstruction
        axes[row, 1].imshow(recon_display)
        axes[row, 1].set_title("Reconstruction", fontsize=11)
        axes[row, 1].axis("off")

        # Col 2: Anomaly heatmap overlaid
        axes[row, 2].imshow(img_display)
        heatmap = axes[row, 2].imshow(error, cmap="jet", alpha=0.5,
                                       vmin=0, vmax=np.percentile(error, 99))
        axes[row, 2].set_title(f"Anomaly Map (max={error.max():.4f})", fontsize=11)
        axes[row, 2].axis("off")

        # Col 3: Ground truth mask
        axes[row, 3].imshow(gt_mask.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("Ground Truth", fontsize=11)
        axes[row, 3].axis("off")

    plt.suptitle(f"MVTec AD — {category}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = out_path / f"{category}_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anomaly heatmaps")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", type=str, default="results/heatmaps")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    generate_heatmaps(
        data_root=args.data_root,
        category=args.category,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device=args.device,
    )
