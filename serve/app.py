"""FastAPI inference server for anomaly detection.

Endpoints:
    POST /predict   — Upload image → get anomaly score + heatmap
    GET  /health    — Health check

Usage:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

import io
import sys
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import AnomalyAutoencoder
from dataset import opencv_preprocess
from torchvision import transforms

# ── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Defect Detection API",
    description="Anomaly detection for manufacturing quality inspection",
    version="1.0.0",
)

# Global model (loaded once at startup)
MODEL = None
DEVICE = None
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CATEGORY = "bottle"


def load_model(category: str = DEFAULT_CATEGORY):
    """Load model checkpoint into global state."""
    global MODEL, DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    MODEL = AnomalyAutoencoder(pretrained=False).to(DEVICE)
    ckpt_path = CHECKPOINT_DIR / f"{category}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    print(f"Model loaded: {category} (epoch {ckpt['epoch']}, device={DEVICE})")


@app.on_event("startup")
async def startup():
    """Load default model on server start."""
    try:
        load_model(DEFAULT_CATEGORY)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Start the server after training a model.")


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file (PNG/JPG)"),
    threshold: float = 0.5,
):
    """Predict anomaly score for uploaded image.

    Returns:
        anomaly_score: Maximum reconstruction error (higher = more anomalous)
        is_anomaly: Boolean based on threshold
        heatmap_base64: Base64-encoded anomaly heatmap (PNG)
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess
    img_rgb = opencv_preprocess(img_bgr, size=256)
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        recon = MODEL(img_tensor)

    # Anomaly map: pixel-wise MSE
    error = ((img_tensor - recon) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
    anomaly_score = float(error.max())
    is_anomaly = anomaly_score >= threshold

    # Generate heatmap as base64 PNG
    heatmap_norm = (error / (error.max() + 1e-8) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", heatmap_color)
    heatmap_b64 = base64.b64encode(buf).decode("utf-8")

    return JSONResponse({
        "anomaly_score": round(anomaly_score, 6),
        "is_anomaly": is_anomaly,
        "threshold": threshold,
        "heatmap_base64": heatmap_b64,
        "image_size": list(img_bgr.shape[:2]),
    })


@app.post("/load-model")
async def load_model_endpoint(category: str = "bottle"):
    """Switch to a different category's model."""
    try:
        load_model(category)
        return {"status": "ok", "category": category}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
