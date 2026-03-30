"""Training script for anomaly detection autoencoder."""

import argparse
import time
import sys
from pathlib import Path

# Resolve project root so script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset
from model import AnomalyAutoencoder


def train(
    data_root: str = str(PROJECT_ROOT / "data"),
    category: str = "bottle",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    size: int = 256,
    device: str = "auto",
    save_dir: str = str(PROJECT_ROOT / "checkpoints"),
    augment: bool = False,
):
    # Device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Data — only normal images for training
    train_ds = MVTecDataset(data_root, category, split="train", size=size, augment=augment)
    if augment:
        print("Data augmentation: ON (albumentations)")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    print(f"Training on {len(train_ds)} normal images ({category})")

    # Model
    model = AnomalyAutoencoder(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # For reconstruction target: we need unnormalized images
    # The model outputs [0, 1] via Sigmoid, so we build a second dataset
    # that returns normalized input + raw [0,1] target
    from torchvision import transforms
    target_transform = transforms.ToTensor()  # just [0,1], no normalize
    target_ds = MVTecDataset(
        data_root, category, split="train", size=size,
        transform=target_transform,
    )
    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Since we need both normalized input and raw target in same order,
    # we'll handle it inline
    best_loss = float("inf")
    Path(save_dir).mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for imgs, labels, masks in train_loader:
            imgs = imgs.to(device)

            # Reconstruct
            recon = model(imgs)

            # For loss: compare reconstruction against normalized input
            # (model learns to reconstruct the normalized representation)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(save_dir) / f"{category}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "category": category,
            }, save_path)

    # Save final
    save_path = Path(save_dir) / f"{category}_final.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "category": category,
    }, save_path)
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (albumentations)")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        category=args.category,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        augment=args.augment,
    )
