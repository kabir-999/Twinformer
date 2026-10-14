# TransformerTwin

TransformerTwin is a local-first digital twin for Transformer inference. It predicts, measures, compares, and eventually calibrates inference performance without paid APIs. The repository is being delivered in phases.

## Current status: Phase 6 plus the Compare tab

The runnable system provides:

- **Compare tab**: a passage from your documents run through an encoder, a decoder and an encoder-decoder model, with real attention maps, per-step token probabilities, tokenization and timing side by side ([notes](docs/compare-architectures.md)).
- A FastAPI backend with validated architecture-aware workloads and SQLite persistence.
- Deterministic **synthetic demonstration** records, visibly labelled in both API and dashboard.
- A React/TypeScript dashboard with overview, architecture explorer, a functional simulator, scaling preview, and experiment history.
- Local encoder-only, decoder-only, and encoder-decoder adapter paths with warmup/repetitions, timing, memory, and runtime metadata.
- Measured records are persisted only after successful local execution; model downloads require per-request opt-in.
- An assumption-bound analytical simulator and a measured-record-only calibrated latency predictor with a sequence-length held-out evaluation.

- Measured scaling sweeps (input/output tokens, batch size), KV-cache on/off comparison, and a multi-turn study comparing history recompute against KV-cache reuse.

- **Chat** (the whole UI): pick any subset of your files (folders have select-all), ask, and get answers from an encoder-only, a decoder-only and an encoder-decoder model, each in a collapsible card, ending with a "Best model" line and its reason. Follow-up questions are rewritten to stand alone. An optional *Auto-pick best fit* mode routes sentiment to an encoder classifier, French/German/Romanian translation to the encoder-decoder, and summaries and open questions to the decoder, and shows the reason. Chats are saved to History. Earlier text on this page about separate Compare/Measure/Predict/Report tabs describes backend APIs that remain, but those screens are no longer in the UI.
- (earlier description) **Ask your documents**: one question answered by an encoder-only, a decoder-only and an encoder-decoder model from the same retrieved passages, then compared by an automatic judge or (opt-in) a Groq-hosted LLM that explains which answer is best ([notes](docs/phase5-local-rag.md)).
- Local RAG over your `~/Documents/MBA` and `~/Documents/Sem 7` folders (PDF/PPTX/DOCX): MiniLM embeddings, exact FAISS retrieval, page/slide citations, optional locally written answers and an externally-labelled retrieval evaluation.
- Research evaluation: calibration ablation, reproducibility manifests, and a report generated only from completed experiments.

It does **not** run concurrent scheduling or OCR scanned PDFs.

## Run locally

Requires Python 3.10+ and Node 20+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the displayed Vite URL (normally `http://localhost:5173`). Top bar: the app name (home) and **+ New chat** always start a fresh chat. Three tabs: **Chat** (choose files, ask, compare the three architectures, ask follow-ups) **History** (reopen or delete earlier chats) and **Twin** (the digital twin: predicts how long each model takes on this computer, learns from every chat answer, and shows prediction vs reality). Put `GROQ_API_KEY=...` in `.env` for the Groq judge; it is read live, no restart needed. The backend docs are at `http://localhost:8000/docs`.

## Verify

```bash
PYTHONPATH=backend pytest backend/tests -q
cd frontend && npm run build
```

## API highlights

- `GET /health`
- `GET /api/v1/overview`
- `GET /api/v1/records`
- `POST /api/v1/simulation/synthetic` creates a labelled synthetic simulation.
- `POST /api/v1/benchmarks` executes a real local benchmark and persists it only if it completes.
- `POST /api/v1/twin/predict` returns an analytical or calibrated twin prediction with assumptions and uncertainty.
- `POST /api/v1/calibration/train` trains and evaluates from saved measured records.
- `POST /api/v1/rag/answers` answers a question three ways and judges them.
- `POST /api/v1/compare` runs one text through all three architectures.
- `POST /api/v1/rag/index`, `/rag/query`, `/rag/evaluate` for local RAG.
- `POST /api/v1/research/ablation`, `/research/manifest`, `/research/report` for Phase 6.
- `POST /api/v1/studies/scaling`, `/studies/cache`, `/studies/multi-turn` run measured studies; `GET /api/v1/studies?kind=` lists saved ones.

See [the phase plan](docs/phase-plan.md), [architecture](docs/architecture.md), and [report template](artifacts/reports/research-report-template.md).

For the benchmark metric definitions and runtime limitations, read [the Phase 2 notes](docs/phase2-real-inference.md).
For calibration conditions and evaluation behavior, read [the Phase 3 notes](docs/phase3-digital-twin.md).
For cache semantics and study methodology, read [the Phase 4 notes](docs/phase4-advanced-inference.md).
For RAG see [the Phase 5 notes](docs/phase5-local-rag.md); for the report and manifest see [the Phase 6 notes](docs/phase6-research-evaluation.md).
