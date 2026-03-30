"""Export PyTorch model to ONNX format + inference benchmark.

Demonstrates model deployment pipeline: PyTorch → ONNX → ONNXRuntime.
ONNX enables deployment on edge devices, C++ inference, TensorRT, etc.
"""

import argparse
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from model import AnomalyAutoencoder


def export_to_onnx(
    checkpoint: str,
    output_path: str = None,
    size: int = 256,
    device: str = "cpu",
):
    """Export trained autoencoder to ONNX format.

    Args:
        checkpoint: Path to .pt checkpoint file.
        output_path: Where to save .onnx file. Defaults to same dir as checkpoint.
        size: Input image size.
        device: Device for export tracing.
    """
    # Load model
    model = AnomalyAutoencoder(pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    category = ckpt.get("category", "unknown")
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded: {checkpoint} (category={category}, epoch={epoch})")

    # Output path
    if output_path is None:
        output_path = str(Path(checkpoint).with_suffix(".onnx"))

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, size, size, device=device)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["image"],
        output_names=["reconstruction"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
        },
    )

    onnx_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported: {output_path} ({onnx_size_mb:.1f} MB)")

    return output_path


def benchmark(checkpoint: str, onnx_path: str, size: int = 256, n_runs: int = 50):
    """Compare PyTorch vs ONNX Runtime inference speed.

    Args:
        checkpoint: Path to .pt checkpoint.
        onnx_path: Path to .onnx model.
        size: Input image size.
        n_runs: Number of inference runs for timing.
    """
    device = "cpu"  # Fair comparison on CPU

    # ── PyTorch inference ────────────────────────────────────────────
    model = AnomalyAutoencoder(pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, size, size, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)

    # Timed runs
    torch_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        torch_times.append(time.perf_counter() - start)

    torch_avg = np.mean(torch_times) * 1000  # ms

    # ── ONNX Runtime inference ───────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime")
        print(f"\nPyTorch avg latency: {torch_avg:.1f} ms (CPU, {n_runs} runs)")
        return

    session = ort.InferenceSession(onnx_path)
    dummy_np = dummy.numpy()

    # Warmup
    for _ in range(5):
        session.run(None, {"image": dummy_np})

    # Timed runs
    ort_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"image": dummy_np})
        ort_times.append(time.perf_counter() - start)

    ort_avg = np.mean(ort_times) * 1000  # ms

    # ── Report ───────────────────────────────────────────────────────
    speedup = torch_avg / ort_avg

    print(f"\n{'='*50}")
    print(f"Inference Benchmark (CPU, {n_runs} runs)")
    print(f"{'='*50}")
    print(f"  PyTorch:      {torch_avg:7.1f} ms")
    print(f"  ONNX Runtime: {ort_avg:7.1f} ms")
    print(f"  Speedup:      {speedup:.2f}x")
    print(f"  Model size:   {Path(onnx_path).stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX + benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: same as checkpoint)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run PyTorch vs ONNX speed benchmark")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of benchmark runs")
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.checkpoint, args.output)

    if args.benchmark:
        benchmark(args.checkpoint, onnx_path, n_runs=args.n_runs)
