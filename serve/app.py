"""FastAPI inference server — Autoencoder baseline vs PatchCore (SOTA), side by side.

Each /predict call scores the uploaded image with BOTH methods and returns their
calibrated verdicts together, so the quality gap is visible in one response.
Thresholds are calibrated without touching the test set; PatchCore additionally
holds its calibration normals out of the memory bank.

Endpoints:
    GET  /health    — health check (which methods/category loaded)
    POST /predict   — upload image → AE + PatchCore scores, verdicts, heatmaps
    POST /load-model?category=bottle — switch category (loads both methods)

Run:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import AnomalyAutoencoder                          # noqa: E402
from dataset import opencv_preprocess                         # noqa: E402
from patchcore import FeatureExtractor, patchcore_anomaly_map  # noqa: E402
from torchvision import transforms                            # noqa: E402

app = FastAPI(
    title="Defect Detection API — AE vs PatchCore",
    description="Unsupervised anomaly detection for manufacturing inspection",
    version="2.0.0",
)

CKPT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CATEGORY = "bottle"
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Global state, populated by load_category()
STATE = {"category": None, "device": None, "ae": None, "patch": None}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Required checkpoint files per category (all three must exist to serve it).
_REQUIRED = ("{c}_best.pt", "{c}_ae_calib.pt", "{c}_patchcore.pt")


def available_categories():
    """Categories that have every required checkpoint present — the whitelist."""
    cats = []
    for p in sorted(CKPT_DIR.glob("*_patchcore.pt")):
        c = p.name[:-len("_patchcore.pt")]
        if all((CKPT_DIR / f.format(c=c)).exists() for f in _REQUIRED):
            cats.append(c)
    return cats


def load_category(category: str = DEFAULT_CATEGORY):
    """Load a category's models, then swap into global STATE atomically.

    Everything is loaded into locals first; STATE is only mutated once all
    artifacts are present and parsed, so a missing/invalid checkpoint can never
    leave the server in a half-loaded state. `category` is validated against the
    checkpoint whitelist to avoid path injection.
    """
    if category not in available_categories():
        raise FileNotFoundError(
            f"Category '{category}' not available. Have: {available_categories()}")

    device = _device()

    # ── Autoencoder + calibrated threshold (load into locals) ───────────────
    ae_model = AnomalyAutoencoder(pretrained=False).to(device)
    ae_ckpt = torch.load(CKPT_DIR / f"{category}_best.pt",
                         map_location=device, weights_only=True)
    ae_model.load_state_dict(ae_ckpt["model_state_dict"])
    ae_model.eval()
    ae_cal = torch.load(CKPT_DIR / f"{category}_ae_calib.pt",
                        map_location="cpu", weights_only=False)
    new_ae = {"model": ae_model, "threshold": ae_cal["threshold"],
              "meta": ae_cal["meta"]}

    # ── PatchCore memory bank + calibrated threshold (load into locals) ─────
    pc = torch.load(CKPT_DIR / f"{category}_patchcore.pt",
                    map_location="cpu", weights_only=False)
    new_patch = {
        "extractor": FeatureExtractor().eval().to(device),
        "memory_bank": pc["memory_bank"].to(device),
        "threshold": pc["threshold"],
        "feat_hw": pc["feat_hw"],
        "size": pc["size"],
        "meta": pc["meta"],
    }

    # ── Atomic swap: only mutate STATE once everything succeeded ─────────────
    STATE.update({"category": category, "device": device,
                  "ae": new_ae, "patch": new_patch})
    print(f"Loaded '{category}' on {device}: AE + PatchCore ready")


@app.on_event("startup")
async def _startup():
    try:
        load_category(DEFAULT_CATEGORY)
    except FileNotFoundError as e:
        print(f"Warning: {e}\nRun build_patchcore.py / build_ae_threshold.py first.")


# ── Scoring helpers ──────────────────────────────────────────────────────────

def _heatmap_b64(error_map: np.ndarray) -> str:
    norm = (error_map / (error_map.max() + 1e-8) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", color)
    return base64.b64encode(buf).decode("utf-8")


def _loaded_payload(state):
    methods_loaded = {
        "autoencoder": state["ae"] is not None,
        "patchcore": state["patch"] is not None,
    }
    loaded = all(methods_loaded.values())
    return {
        "status": "healthy" if loaded else "unavailable",
        "category": state["category"],
        "device": state["device"],
        "methods_loaded": methods_loaded,
    }


@torch.no_grad()
def _score_ae(img_tensor, ae_state):
    recon = ae_state["model"](img_tensor)
    err = ((img_tensor - recon) ** 2).mean(dim=1).squeeze(0).cpu().numpy()
    th = ae_state["threshold"]
    m = ae_state["meta"]
    return {
        "method": "autoencoder", "label": "Autoencoder (baseline)",
        "anomaly_score": round(float(err.max()), 4), "threshold": round(th, 4),
        "is_anomaly": bool(err.max() >= th),
        "val_image_auroc": round(m["image_auroc"], 4),
        "val_escape_rate": round(m["escape"], 4),
        "val_overkill_rate": round(m["overkill"], 4),
        "heatmap_base64": _heatmap_b64(err),
    }


@torch.no_grad()
def _score_patchcore(img_tensor, patch_state):
    p = patch_state
    m = patchcore_anomaly_map(p["extractor"], p["memory_bank"], img_tensor, p["size"])
    th = p["threshold"]
    meta = p["meta"]
    return {
        "method": "patchcore", "label": "PatchCore (SOTA)",
        "anomaly_score": round(float(m.max()), 4), "threshold": round(th, 4),
        "is_anomaly": bool(m.max() >= th),
        "val_image_auroc": round(meta["image_auroc"], 4),
        "val_escape_rate": round(meta["escape"], 4),
        "val_overkill_rate": round(meta["overkill"], 4),
        "heatmap_base64": _heatmap_b64(m),
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    state = STATE.copy()
    payload = _loaded_payload(state)
    status_code = 200 if payload["status"] == "healthy" else 503
    return JSONResponse(payload, status_code=status_code)


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Image (PNG/JPG)")):
    """Score the image with BOTH methods and return their calibrated verdicts."""
    # Snapshot the model state once so a concurrent /load-model request cannot
    # mix an AE from one category with a PatchCore bank from another.
    state = STATE.copy()
    ae_state = state["ae"]
    patch_state = state["patch"]
    if ae_state is None or patch_state is None:
        raise HTTPException(503, "Models not loaded")

    contents = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Invalid image file")

    img_rgb = opencv_preprocess(img_bgr, size=256)
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(state["device"])

    ae = _score_ae(img_tensor, ae_state)
    pc = _score_patchcore(img_tensor, patch_state)
    return JSONResponse({
        "category": state["category"],
        "image_size": list(img_bgr.shape[:2]),
        "methods": {"autoencoder": ae, "patchcore": pc},
        "methods_agree": ae["is_anomaly"] == pc["is_anomaly"],
    })


@app.post("/load-model")
async def load_model_endpoint(category: str = DEFAULT_CATEGORY):
    try:
        load_category(category)
        return {"status": "ok", "category": category}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
