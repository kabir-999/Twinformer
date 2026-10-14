# TransformerTwin architecture

```mermaid
flowchart LR
  UI[React dashboard] --> API[FastAPI]
  API --> W[Workload validator]
  API --> S[(SQLite benchmark records)]
  W --> SIM[Digital-twin simulator]
  W --> BENCH[Real benchmark engine · Phase 2]
  BENCH --> S
  SIM --> S
  S --> CAL[Calibration + validation · Phase 3]
  DOCS[Local documents] --> RAG[Local RAG · Phase 5]
  S --> RAG
```

Phase 1 only enables `SIM`, using deterministic synthetic demonstration records. The database source column and dashboard distinguish `synthetic` from `measured`; the benchmark API intentionally returns 501 until actual local inference is implemented.
