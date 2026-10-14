# Incremental delivery plan

## Phase 1 — Foundation (complete)

- FastAPI API, SQLite repository, workload compatibility validation, deterministic synthetic demo records, React dashboard, and tests.
- Synthetic data is marked in storage and the UI. Calibration is always reported as untrained.

## Phase 2 — Real inference (complete)

- Hugging Face adapters for the three Transformer families, environment probes, warmups, repeated timing, and measured-only records.

## Phase 3 — Digital twin and calibration (complete)

- Separate analytical simulator, architecture-aware features, held-out split by workload regime, calibrated models, uncertainty, and validation UI.

## Phase 4 — Advanced inference (complete)

- Multi-turn/caching experiments and scaling studies, documenting actual runtime cache semantics. Delivered: scaling sweeps, KV-cache on/off comparison, and multi-turn recompute vs KV-reuse study (see phase4-advanced-inference.md).

## Phase 5 — Local RAG (complete)

- Local ingestion, embeddings, FAISS retrieval, citations, and independent retrieval evaluation.

## Phase 6 — Research evaluation (complete)

- Reproducible manifests, measurements, calibration ablation, limitations, and a report populated only from completed experiments.
