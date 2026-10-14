# Phase 6 — Research evaluation

`POST /api/v1/research/report` (or the dashboard's *Research report* panel) writes:

- `artifacts/manifests/manifest-<time>.json`: code SHA-256 over `backend/app/**/*.py`, hardware/OS, package versions, measured-record count and hashes, models/devices/precisions used, saved study ids, RAG index state.
- `artifacts/reports/report-<time>.md`: tables computed only from saved measured records, studies, the latest calibration, the ablation, and the latest RAG evaluation. Missing experiments are written as "Not run". Synthetic records are never used.

`POST /api/v1/research/ablation` compares, on the calibration's sequence-length held-out split: the analytical baseline, an intercept-only model, the full Ridge model, and Ridge with each feature removed in turn. It also reports empirical coverage of the ±1.96σ interval.

## Reproducing a report

1. Run the studies (scaling, cache, multi-turn), then `POST /calibration/train`, index the corpus and run the RAG evaluation.
2. Generate the report. Compare the manifest hashes and package versions to confirm the same code and data.

Measured numbers will differ across machines; the manifest records what is needed to explain the difference. Records from benchmark versions before `phase4-local-v2` are excluded from the report's measurement table because earlier engines mis-implemented cache-off decoding and batching.
