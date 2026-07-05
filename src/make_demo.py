"""Generate the README demo figure: AE vs PatchCore, side by side.

For a few representative bottle test images (normal / broken / contamination),
render: original | AE heatmap+verdict | PatchCore heatmap+verdict. The
contamination row visually shows the AE *missing* a real defect that PatchCore
catches — the core "why method choice matters" story.

Output: results/demo_comparison.png

Usage: python src/make_demo.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from model import AnomalyAutoencoder
from dataset import opencv_preprocess
from patchcore import FeatureExtractor, patchcore_anomaly_map

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
TF = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
CAT = "bottle"
CK = PROJECT_ROOT / "checkpoints"

# Models
ae = AnomalyAutoencoder(pretrained=False).to(DEVICE).eval()
ae.load_state_dict(torch.load(CK / f"{CAT}_best.pt", map_location=DEVICE,
                              weights_only=True)["model_state_dict"])
ae_th = torch.load(CK / f"{CAT}_ae_calib.pt", weights_only=False)["threshold"]
pcd = torch.load(CK / f"{CAT}_patchcore.pt", weights_only=False)
extractor = FeatureExtractor().to(DEVICE).eval()
bank = pcd["memory_bank"].to(DEVICE)
pc_th = pcd["threshold"]


@torch.no_grad()
def maps(path):
    bgr = cv2.imread(str(path))
    rgb = opencv_preprocess(bgr, 256)
    x = TF(rgb).unsqueeze(0).to(DEVICE)
    # AE
    recon = ae(x)
    ae_map = ((x - recon) ** 2).mean(1).squeeze(0).cpu().numpy()
    # PatchCore (shared helper)
    pc_map = patchcore_anomaly_map(extractor, bank, x, 256)
    return rgb, ae_map, pc_map


def overlay(rgb, m):
    norm = (m / (m.max() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return 0.55 * rgb + 0.45 * heat


rows = [
    ("data/bottle/test/good/000.png", "Normal (ground truth: OK)"),
    ("data/bottle/test/broken_large/000.png", "Broken (ground truth: defect)"),
    ("data/bottle/test/contamination/000.png", "Contamination (ground truth: defect)"),
]

fig, axes = plt.subplots(len(rows), 3, figsize=(10, 10.5))
for r, (path, title) in enumerate(rows):
    rgb, ae_map, pc_map = maps(PROJECT_ROOT / path)
    gt_defect = "good" not in path
    for c, (img, name, score, th) in enumerate([
        (rgb, "Input", None, None),
        (overlay(rgb, ae_map), "Autoencoder (baseline)", ae_map.max(), ae_th),
        (overlay(rgb, pc_map), "PatchCore (SOTA)", pc_map.max(), pc_th),
    ]):
        ax = axes[r, c]
        ax.imshow(np.clip(img, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(title, fontsize=10)
            ax.set_title("Input", fontsize=11)
        else:
            pred = score >= th
            correct = (pred == gt_defect)
            verdict = "ANOMALY" if pred else "normal"
            color = "#2e7d32" if correct else "#c62828"
            mark = "✓" if correct else "✗ MISS" if gt_defect else "✗ FALSE"
            ax.set_title(f"{name}\n{verdict}  {mark}", fontsize=10, color=color)

fig.suptitle("MVTec AD · bottle — Autoencoder vs PatchCore (calibrated, leakage-aware)",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = PROJECT_ROOT / "results" / "demo_comparison.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved → {out}")
