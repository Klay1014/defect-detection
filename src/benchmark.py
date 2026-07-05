"""Inference latency / throughput benchmark — AE vs PatchCore.

Reports per-image latency (ms) and throughput (FPS) for both methods on the
available device(s). Relevant because AOI is an inline, real-time task where
throughput is a hard product constraint, not a nice-to-have.

Usage: python src/benchmark.py --category bottle --iters 50
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torchvision import transforms

from model import AnomalyAutoencoder
from patchcore import FeatureExtractor, patchcore_anomaly_map

TF = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench(category, iters, device):
    x = TF(np.random.rand(256, 256, 3).astype(np.float32)).unsqueeze(0).to(device)

    ae = AnomalyAutoencoder(pretrained=False).to(device).eval()
    ae.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / f"{category}_best.pt",
                                  map_location=device, weights_only=True)["model_state_dict"])

    pc = torch.load(PROJECT_ROOT / "checkpoints" / f"{category}_patchcore.pt",
                    map_location=device, weights_only=False)
    extractor = FeatureExtractor().to(device).eval()
    bank = pc["memory_bank"].to(device)

    def run_ae():
        r = ae(x); _ = ((x - r) ** 2).mean(1).squeeze(0).cpu().numpy().max()

    def run_pc():
        _ = patchcore_anomaly_map(extractor, bank, x, 256).max()

    results = {}
    for name, fn in [("Autoencoder", run_ae), ("PatchCore", run_pc)]:
        for _ in range(5):
            fn()                       # warm-up
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(device)
        ms = (time.perf_counter() - t0) / iters * 1000
        results[name] = ms
    return results, device


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    print(f"\nLatency / throughput (batch=1, 256x256, {args.iters} iters)\n")
    print(f"{'Method':<14}{'Device':<8}{'Latency (ms)':>14}{'Throughput (FPS)':>18}")
    print("-" * 54)
    for dev in devices:
        res, _ = bench(args.category, args.iters, dev)
        for method, ms in res.items():
            print(f"{method:<14}{dev:<8}{ms:>14.1f}{1000/ms:>18.1f}")
