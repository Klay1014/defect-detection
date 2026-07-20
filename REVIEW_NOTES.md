# Review Notes — serving + calibration + benchmark change set

Context for a code reviewer (e.g. Codex). This change set makes the deployment
actually run and turns the README's claims into measured, leakage-aware numbers.

## Review resolution (2026-06-28)

Addressing the review findings:

- **P1 README stale** → ✅ rewritten: bottle headline table (AUROC + escape/overkill
  + latency), dual-method inference diagram, demo image, build steps in Quick Start,
  multi-category table relabelled as earlier raw runs.
- **P1 Docker not self-contained** → ✅ `serve/Dockerfile` now bakes the torchvision
  ResNet18 weights into the image (`TORCH_HOME=/opt/torch` + pre-download) → offline
  startup safe.
- **P1 AE "held-out calibration" wording** → ✅ corrected in README *Evaluation notes*:
  PatchCore = true held-out (disjoint from memory bank); AE = calibrated on
  training-seen normals (no test leakage, not held out from AE fit).
- **P2 AE training normalisation mismatch** → ✅ documented as an intentional weak
  baseline in README (not retrained — keeps the checkpoint stable).
- **P2 `/load-model` non-atomic** → ✅ `load_category` now loads into locals and swaps
  `STATE` once; added a checkpoint whitelist (`available_categories`) that rejects
  unknown/path-injection categories (verified: `../etc/passwd` → 404).
- **P2 dependency pinning** → ✅ pinned `fastapi==0.104.1 / starlette==0.27.0 /
  uvicorn==0.24.0 / httpx==0.27.2`; verified `TestClient` works and `/predict` returns
  200 with both methods.

- **Open item: duplicated PatchCore scoring** → ✅ factored into
  `patchcore.patchcore_anomaly_map()`; `serve/app.py`, `build_patchcore.py`,
  `make_demo.py`, `benchmark.py` now all call it (rebuild verified bit-identical
  metrics). Root scratch notebooks moved into `notebooks/` (still gitignored).

## Review resolution (2026-07-15)

- **Healthcheck false-positive** → ✅ `/health` now returns 503 with
  `status=unavailable` unless both AE and PatchCore are loaded, so Docker's
  healthcheck fails when `/predict` would fail.
- **Concurrent `/load-model` race** → ✅ `/predict` snapshots the current category,
  device, AE state and PatchCore state once per request before scoring.
- **Docker context too large** → ✅ added `.dockerignore`; `data/`, notebooks,
  caches and unused checkpoints are excluded while the bottle serving checkpoints
  remain included.
- **AE calibration wording** → ✅ README headline and Methods now say normal-only
  calibration for AE, not held-out from AE fitting.
- **Review notes stale items** → ✅ known-issues list below now matches the current
  state.

Remaining open: single-seed calibration, fine-texture failure modes (`grid` /
`tile`), and dual-method serving beyond `bottle` (PatchCore exists for 9 classes,
but AE checkpoints/calibrations are only available for bottle).

## What changed

| File | Status | Purpose |
|------|--------|---------|
| `serve/app.py` | modified | Was serving only the Autoencoder with a hardcoded `threshold=0.5` (flagged *everything* as anomalous). Rewritten to v2: loads **both** AE + PatchCore and `/predict` returns both verdicts side by side, using calibrated thresholds. |
| `src/build_patchcore.py` | new | Builds + persists a PatchCore memory bank to `checkpoints/{cat}_patchcore.pt`. Threshold calibrated on a **held-out slice of normal training images** (never the test set). |
| `src/build_ae_threshold.py` | new | Normal-only, no-test-leakage calibration for the AE baseline → `checkpoints/{cat}_ae_calib.pt`. The AE has already been fit on `train/good`, so these normals are not held out from AE training. |
| `src/benchmark.py` | new | Per-image latency (ms) + throughput (FPS) for both methods, CPU + MPS. |
| `src/make_demo.py` | new | Renders `results/demo_comparison.png` (normal / broken / contamination × input / AE / PatchCore). |

Measured on `bottle` (leakage-free calibration, target overkill 2%):

| Method | image AUROC | escape | overkill | latency (MPS) |
|---|---|---|---|---|
| Autoencoder (baseline) | 0.859 | 38% | 5% | 13 ms / 78 FPS |
| PatchCore (SOTA) | 1.000 | 1.6% | 0% | 34 ms / 30 FPS |

## Known issues / please scrutinise

These are deliberate trade-offs or known gaps — not blind spots:

1. **`torch.load(..., weights_only=False)`** on the `*_patchcore.pt` / `*_ae_calib.pt`
   files (they store a dict with numpy arrays + a tensor). Fine for our own local
   artifacts; flag if you'd prefer a safer serialization (e.g. `safetensors` +
   separate JSON for scalars).
2. **Single random calibration split (seed 0), small calib set (~41 imgs for
   bottle).** No
   cross-validation / confidence interval on the escape/overkill numbers. Is a
   single split defensible for a portfolio claim, or should I report mean±sd over
   a few seeds?
3. **AE anomaly score = max pixel MSE**, which is noise-sensitive and normalised
   per-image for the heatmap (not comparable across images). This is the baseline,
   so weak behaviour is expected/intended — but confirm the framing is honest.
4. **Dual-method serving is currently bottle-only.** PatchCore checkpoints exist
   for 9 categories, but `/predict` requires AE + AE calibration + PatchCore, and
   only bottle currently has all three.
5. **Fine regular textures still need a stronger PatchCore variant.** `grid` and
   `tile` have poor calibrated AUROC/escape with ResNet18 layer2/3 at 256 px.

## How to reproduce

```bash
pip install -r requirements.txt          # adds torchvision
python src/build_patchcore.py   --category bottle
python src/build_ae_threshold.py --category bottle
python src/benchmark.py --category bottle
python src/make_demo.py
uvicorn serve.app:app --port 8000        # POST /predict -F file=@<img.png>
```
