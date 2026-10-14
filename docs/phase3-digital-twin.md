# Phase 3: digital twin and calibration

The twin has two modes:

1. **Analytical baseline** — a portable scaling estimate based on default model parameter profiles, input/output lengths, batch size, approximate attention work, and decoder KV-cache memory. Its broad ±50% interval is an explicit pre-calibration assumption, not a performance claim.
2. **Calibrated latency predictor** — a Ridge regression trained only from persisted `measured` records. Synthetic records cannot enter its training data.

Calibration requires at least eight measured records covering at least three input lengths. It trains on lower input-length regimes and holds out longer input lengths, avoiding a random split across near-identical repeats. It reports MAE, RMSE, MAPE (when defined), R² (when defined), and residual-based heuristic intervals.

Peak-memory calibration is intentionally deferred: the calibrated latency response keeps the analytical memory estimate rather than misrepresenting it as learned.

## API

- `POST /api/v1/twin/predict` — returns analytical or calibrated prediction plus assumptions and interval.
- `POST /api/v1/calibration/train` — fits/evaluates using available measured records.
- `GET /api/v1/calibration/report` — returns current training and held-out evaluation status.


## Update: first-principles baseline, physics calibration, and the chat loop

- **Baseline** (`analytical_prediction`): prefill is compute-bound (`2 × params × input tokens / FLOP/s`) and every generated token streams the weights from memory (`weight bytes / bandwidth`; without a KV cache each step re-reads the growing sequence). It uses each model's real parameter count (`KNOWN_PARAMS_M`, then a size in the model name, then the architecture default). The assumed speeds (800 GFLOP/s, 50 GB/s) are assumptions, chosen from rough early timings on the development machine, so the baseline is better than a blind guess and the measured learning gain is understated.
- **Calibrated predictor**: `time = intercept + a·prefill + b·decode` per architecture with `a, b ≥ 0` (non-negative least squares on measured runs only). About five parameters, so a few dozen runs are enough, and `a`/`b` read as "how far the assumed machine speed was off" (`corrections()`). An architecture with no measurements falls back to the formula. A generic Ridge regression over lengths and model identity was tried first and overfit small data; it remains in the ablation for comparison.
- **Chat integration** (`app/twin_link.py`): every warm decoder-only and encoder-decoder answer in Chat is saved as a measured run (`benchmark_version = chat-turn-v1`) with its real token counts and time. Cold runs (model just loaded) and encoder-only extraction are excluded. Before learning from a turn, the twin predicts it (`predicted_seconds` on each answer card); then it re-trains. `GET /twin/estimate` gives the expected time shown before sending; `GET /twin/status` backs the Twin tab.
- **First real result** (22 runs from 12 chat questions on an M-series CPU, held out by input length, 4 test runs): typical error 2.06 s (formula) → 1.86 s (physics-calibrated); MAPE 19.6% → 13.4%; generic Ridge 1.92 s. Small test, dominated by one 19.8 s long-prompt run, so treat it as indicative. The encoder-decoder corrections were large (prefill ×10, decode ×3.4) because flan-t5 runs beam search, which the formula ignores; the decoder's were near 1 (×0.72, ×1.09).
