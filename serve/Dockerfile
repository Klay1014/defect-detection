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
