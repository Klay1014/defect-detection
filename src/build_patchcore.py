"""Build and persist a PatchCore memory bank + leakage-aware threshold.

Unlike `patchcore.py` (which rebuilds everything per run for evaluation), this
script builds the memory bank ONCE and saves it for fast serving. Crucially, the
decision threshold is calibrated on a *held-out slice of normal training images*
— never on the test set — so the reported escape/overkill numbers are honest.

Output: checkpoints/{category}_patchcore.pt with
    memory_bank   : (M, 384) float32 coreset of normal patches
    threshold     : calibrated decision threshold (overkill-controlled)
    feat_hw       : feature map spatial size (for reshaping at inference)
    calib_scores  : normal-only calibration image scores (for operating-point UI)
    meta          : dict of build/eval stats

Usage:
    python src/build_patchcore.py --category bottle
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import MVTecDataset
from patchcore import FeatureExtractor, patchcore_anomaly_map


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def _extract(extractor, img_tensor, device):
    """Return (H*W, C) patch features for a single (3,H,W) image tensor."""
    feats = extractor(img_tensor.unsqueeze(0).to(device))  # (1, C, H, W)
    _, C, H, W = feats.shape
    return feats[0].permute(1, 2, 0).reshape(-1, C), (H, W)


def _score_map(extractor, img_tensor, memory_bank, size, device):
    """Anomaly map (size,size) via the shared scoring helper."""
    return patchcore_anomaly_map(extractor, memory_bank, img_tensor.to(device), size)


def build(category: str, data_root: str, subsample: float, calib_frac: float,
          target_overkill: float, device: str, size: int = 256, seed: int = 0):
    device = _pick_device(device)
    rng = np.random.default_rng(seed)
    extractor = FeatureExtractor().eval().to(device)

    # ── Split normal train images: memory bank vs calibration ───────────────
    train_ds = MVTecDataset(data_root, category, split="train", size=size)
    n = len(train_ds)
    perm = rng.permutation(n)
    n_calib = max(int(n * calib_frac), 10)
    calib_idx, bank_idx = perm[:n_calib], perm[n_calib:]
    print(f"[{category}] {n} normal train imgs → bank {len(bank_idx)} / calib {len(calib_idx)} "
          f"(device={device})")

    # ── Build memory bank from bank split ───────────────────────────────────
    feats, feat_hw = [], None
    for i in bank_idx:
        img, _, _ = train_ds[i]
        p, feat_hw = _extract(extractor, img, device)
        feats.append(p.cpu())
    bank = torch.cat(feats)                                   # (Nb, C)
    n_keep = max(int(bank.shape[0] * subsample), 2000)
    keep = torch.from_numpy(rng.permutation(bank.shape[0])[:n_keep])
    memory_bank = bank[keep].to(device)
    print(f"[{category}] memory bank: {memory_bank.shape[0]:,} patches "
          f"(coreset {subsample:.0%} of {bank.shape[0]:,})")

    # ── Calibrate threshold on held-out NORMAL images only (no leakage) ─────
    calib_scores = np.array([
        _score_map(extractor, train_ds[i][0], memory_bank, size, device).max()
        for i in calib_idx
    ])
    # threshold = quantile that lets through `target_overkill` of normals
    threshold = float(np.quantile(calib_scores, 1.0 - target_overkill))
    print(f"[{category}] calib normal score: μ={calib_scores.mean():.3f} "
          f"max={calib_scores.max():.3f} → threshold={threshold:.3f} "
          f"(target overkill {target_overkill:.0%})")

    # ── Honest evaluation on the test set ───────────────────────────────────
    test_ds = MVTecDataset(data_root, category, split="test", size=size)
    scores, labels = [], []
    for i in range(len(test_ds)):
        img, lbl, _ = test_ds[i]
        scores.append(_score_map(extractor, img, memory_bank, size, device).max())
        labels.append(lbl)
    scores, labels = np.array(scores), np.array(labels)
    auroc = roc_auc_score(labels, scores)
    defect, good = scores[labels == 1], scores[labels == 0]
    escape = float((defect < threshold).mean())
    overkill = float((good >= threshold).mean())
    print(f"[{category}] TEST  AUROC={auroc:.4f}  escape={escape:.1%}  overkill={overkill:.1%}")

    out = PROJECT_ROOT / "checkpoints" / f"{category}_patchcore.pt"
    torch.save({
        "memory_bank": memory_bank.cpu(),
        "threshold": threshold,
        "feat_hw": feat_hw,
        "size": size,
        "calib_scores": calib_scores,
        "meta": {
            "category": category, "subsample": subsample,
            "image_auroc": auroc, "escape": escape, "overkill": overkill,
            "target_overkill": target_overkill,
        },
    }, str(out))
    print(f"[{category}] saved → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--subsample", type=float, default=0.1)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--target-overkill", type=float, default=0.02)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    build(args.category, args.data_root, args.subsample,
          args.calib_frac, args.target_overkill, args.device)
