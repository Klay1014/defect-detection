"""Calibrate the Autoencoder baseline's decision threshold (leakage-aware).

Mirrors build_patchcore.py's calibration: the threshold is set on a held-out
slice of NORMAL training images only, then escape/overkill is reported on the
test set. Lets the API serve a fair AE-vs-PatchCore comparison.

Output: checkpoints/{category}_ae_calib.pt  { threshold, calib_scores, meta }

Usage: python src/build_ae_threshold.py --category bottle
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import MVTecDataset
from model import AnomalyAutoencoder


def _device(d):
    if d != "auto":
        return d
    return "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _score(model, img, device):
    x = img.unsqueeze(0).to(device)
    r = model(x)
    return float(((x - r) ** 2).mean(dim=1).squeeze(0).cpu().numpy().max())


def build(category, data_root, calib_frac, target_overkill, device, seed=0):
    device = _device(device)
    rng = np.random.default_rng(seed)
    model = AnomalyAutoencoder(pretrained=False).to(device)
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / f"{category}_best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_ds = MVTecDataset(data_root, category, split="train", size=256)
    n = len(train_ds)
    # same split convention as build_patchcore: first n_calib of the permutation
    calib_idx = rng.permutation(n)[:max(int(n * calib_frac), 10)]
    calib_scores = np.array([_score(model, train_ds[i][0], device) for i in calib_idx])
    threshold = float(np.quantile(calib_scores, 1.0 - target_overkill))

    test_ds = MVTecDataset(data_root, category, split="test", size=256)
    scores, labels = [], []
    for i in range(len(test_ds)):
        img, lbl, _ = test_ds[i]
        scores.append(_score(model, img, device)); labels.append(lbl)
    scores, labels = np.array(scores), np.array(labels)
    auroc = roc_auc_score(labels, scores)
    escape = float((scores[labels == 1] < threshold).mean())
    overkill = float((scores[labels == 0] >= threshold).mean())
    print(f"[{category}] AE  threshold={threshold:.3f}  AUROC={auroc:.4f}  "
          f"escape={escape:.1%}  overkill={overkill:.1%}")

    out = PROJECT_ROOT / "checkpoints" / f"{category}_ae_calib.pt"
    torch.save({"threshold": threshold, "calib_scores": calib_scores,
                "meta": {"category": category, "image_auroc": auroc,
                         "escape": escape, "overkill": overkill,
                         "target_overkill": target_overkill}}, str(out))
    print(f"[{category}] saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--target-overkill", type=float, default=0.02)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    build(a.category, a.data_root, a.calib_frac, a.target_overkill, a.device)
