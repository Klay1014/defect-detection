# Industrial Defect Detection — Unsupervised Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![ONNX](https://img.shields.io/badge/ONNX-exportable-orange)

End-to-end manufacturing defect detection system using unsupervised anomaly detection on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark. Trains only on **normal (defect-free) images** — no labeled defect data required.

## Highlights

- **Two methods compared**: Autoencoder baseline vs. PatchCore (SOTA), achieving **97%+ pixel-level AUROC**
- **Production-ready**: FastAPI serving + Docker deployment + ONNX export
- **Full preprocessing pipeline**: OpenCV + albumentations data augmentation

---

## Architecture

```mermaid
graph TD
    A[(MVTec AD<br/>15 categories<br/>5000+ images)] --> B[OpenCV Preprocess<br/>resize 256 · blur · normalize]
    B --> C{Method}

    subgraph Autoencoder ["Method A: Autoencoder"]
        D[ResNet18 Encoder<br/>pretrained · 512×8×8] --> E[Decoder<br/>TransposeConv ×5 · 3×256×256]
        E --> F[MSE Loss<br/>reconstruct normal images]
    end

    subgraph PatchCore ["Method B: PatchCore — SOTA"]
        G[ResNet18 layer2+3<br/>384-dim patch features] --> H[Memory Bank<br/>normal patches · 10% coreset]
        H --> I[Nearest Neighbor<br/>distance per patch]
    end

    C -->|train 100 epochs| D
    C -->|no training needed| G

    F --> J[Anomaly Map<br/>pixel-wise error]
    I --> J

    J --> K[Evaluation<br/>Image AUROC · Pixel AUROC · F1]
    J --> L[FastAPI + Docker<br/>POST /predict]
    J --> M[ONNX Export<br/>edge deployment]

    style A fill:#c8e6c9,stroke:#2e7d32,color:#000
    style B fill:#c8e6c9,stroke:#2e7d32,color:#000
    style D fill:#bbdefb,stroke:#1565c0,color:#000
    style E fill:#bbdefb,stroke:#1565c0,color:#000
    style F fill:#bbdefb,stroke:#1565c0,color:#000
    style G fill:#b3e5fc,stroke:#0277bd,color:#000
    style H fill:#b3e5fc,stroke:#0277bd,color:#000
    style I fill:#b3e5fc,stroke:#0277bd,color:#000
    style J fill:#fff9c4,stroke:#f9a825,color:#000
    style L fill:#ffe0b2,stroke:#e65100,color:#000
    style M fill:#ffe0b2,stroke:#e65100,color:#000
```

The system compares two unsupervised approaches: a **reconstruction-based Autoencoder** (learns to reconstruct normal images; defects cause high error) and **PatchCore** (builds a memory bank of normal patch features; defects are far from any known normal patch).

---

## Results

Evaluated on 3 MVTec AD categories (trained on Kaggle T4 GPU):

| Category | Autoencoder |  | PatchCore |  |
|----------|:-----------:|:-----------:|:---------:|:-----------:|
|          | Image AUROC | Pixel AUROC | Image AUROC | Pixel AUROC |
| Bottle   | 0.7992 | 0.3228 | **0.9984** | **0.9770** |
| Carpet   | 0.6364 | 0.6023 | **0.8535** | **0.9760** |
| Hazelnut | 0.9846 | 0.2338 | **0.9968** | **0.9801** |
| **Avg**  | 0.8067 | 0.3863 | **0.9496** | **0.9777** |

**Key insight**: PatchCore dramatically improves pixel-level localization (**0.39 → 0.98** AUROC) because it scores each patch independently via nearest-neighbor distance, rather than relying on global reconstruction quality.

> Run `python src/visualize.py --category bottle` to generate heatmap images in `results/heatmaps/`.

---

## Inference Pipeline

```mermaid
graph LR
    A[Upload Image<br/>PNG / JPG] --> B[cv2.imdecode]
    B --> C[OpenCV Preprocess<br/>resize · blur · BGR→RGB]
    C --> D[ImageNet Normalize<br/>ToTensor]
    D --> E[Autoencoder<br/>forward pass]
    E --> F[Pixel-wise MSE<br/>reconstruction error]
    F --> G["Anomaly Score<br/>max(error)"]
    F --> H[Heatmap<br/>JET colormap → base64]
    G --> I[JSON Response<br/>score + heatmap + is_anomaly]
    H --> I

    style A fill:#e8f5e9,stroke:#2e7d32,color:#000
    style E fill:#bbdefb,stroke:#1565c0,color:#000
    style I fill:#ffe0b2,stroke:#e65100,color:#000
```

**API endpoints** (FastAPI + Docker):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (model loaded? device?) |
| `/predict` | POST | Upload image → anomaly score + heatmap |
| `/load-model` | POST | Switch to different category model |

```bash
# Local
uvicorn serve.app:app --host 0.0.0.0 --port 8000

# Docker
docker-compose up --build

# Test
curl -X POST http://localhost:8000/predict -F "file=@test_image.png"
```

---

## Methods

### Autoencoder (Baseline)

- **Encoder**: Pretrained ResNet18 conv layers → 512-d features at 8×8
- **Decoder**: 5 transposed conv layers → reconstruct 256×256 RGB
- **Anomaly score**: Max pixel-wise MSE; threshold via Youden's J statistic

### PatchCore (SOTA)

- **Features**: ResNet18 layer2 (128-ch) + layer3 (256-ch) → 384-dim patches
- **Memory bank**: Normal training patches, 10% random coreset subsampling
- **Scoring**: Nearest-neighbor L2 distance → upsample + Gaussian smooth (σ=4)

> Reference: *Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022*

---

## Module Map

```mermaid
graph BT
    DS[dataset.py<br/>OpenCV + augmentation]
    MD[model.py<br/>ResNet18 Autoencoder]

    TR[train.py] --> DS
    TR --> MD
    EV[evaluate.py] --> DS
    EV --> MD
    VZ[visualize.py] --> DS
    VZ --> MD
    PC[patchcore.py<br/>own FeatureExtractor] --> DS
    EX[export_onnx.py] --> MD
    AP[serve/app.py] --> DS
    AP --> MD

    style DS fill:#c8e6c9,stroke:#2e7d32,color:#000
    style MD fill:#c8e6c9,stroke:#2e7d32,color:#000
    style TR fill:#bbdefb,stroke:#1565c0,color:#000
    style EV fill:#bbdefb,stroke:#1565c0,color:#000
    style VZ fill:#bbdefb,stroke:#1565c0,color:#000
    style PC fill:#b3e5fc,stroke:#0277bd,color:#000
    style EX fill:#ffe0b2,stroke:#e65100,color:#000
    style AP fill:#ffe0b2,stroke:#e65100,color:#000
```

`dataset.py` and `model.py` are the foundation modules with no local imports. `patchcore.py` has its own `FeatureExtractor` (extracts intermediate ResNet layers) rather than using the autoencoder from `model.py`.

---

## Project Structure

```
defect-detection/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── data/                          # MVTec AD dataset (gitignored)
├── checkpoints/                   # Trained models (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Data exploration + OpenCV showcase
│   └── kaggle_train.ipynb        # Full training notebook (Kaggle GPU)
├── src/
│   ├── dataset.py                # PyTorch Dataset + OpenCV + augmentation
│   ├── model.py                  # ResNet18 Autoencoder
│   ├── patchcore.py              # PatchCore (SOTA method)
│   ├── train.py                  # Training script
│   ├── evaluate.py               # AUROC/F1 evaluation
│   ├── visualize.py              # Anomaly heatmap generation
│   └── export_onnx.py           # ONNX export + benchmark
├── serve/
│   ├── app.py                    # FastAPI inference API
│   └── Dockerfile                # Multi-stage Docker build
└── results/
    └── heatmaps/                 # Generated visualizations
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download MVTec AD → extract to data/
tar xf mvtec_anomaly_detection.tar.xz -C data/

# 3. Train autoencoder (with optional augmentation)
python src/train.py --category bottle --epochs 100 --augment

# 4. Evaluate
python src/evaluate.py --category bottle

# 5. Run PatchCore
python src/patchcore.py --category bottle

# 6. Generate heatmaps
python src/visualize.py --category bottle

# 7. Export to ONNX + benchmark
python src/export_onnx.py --checkpoint checkpoints/bottle_best.pt --benchmark

# 8. Serve API (or: docker-compose up --build)
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, torchvision |
| Image Processing | OpenCV |
| Data Augmentation | albumentations |
| Evaluation | scikit-learn (AUROC, F1, ROC) |
| Serving | FastAPI, uvicorn |
| Deployment | Docker, ONNX Runtime |
| Visualization | matplotlib |

## License

This project uses the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) for academic/research purposes.
