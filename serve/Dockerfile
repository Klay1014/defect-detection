# Multi-stage Dockerfile for anomaly detection API
# Stage 1: Dependencies only (cached layer)
# Stage 2: Application code

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake torchvision's ResNet18 ImageNet weights into the image so PatchCore's
# FeatureExtractor starts with no network access (offline-safe deployment).
ENV TORCH_HOME=/opt/torch
RUN python -c "import torchvision.models as m; m.resnet18(weights=m.ResNet18_Weights.DEFAULT)"

# ── Application stage ───────────────────────────────────────────────────────
FROM base AS app

WORKDIR /app

# Copy source code
COPY src/ src/
COPY serve/ serve/
COPY checkpoints/ checkpoints/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run server
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
