# Phase 4 — Advanced inference

All studies are measured on the local machine and persisted only if every underlying run succeeds. Scaling and cache studies also save their individual runs as ordinary `measured` benchmark records (version `phase4-local-v2`), so they feed calibration.

## Cache semantics (as actually implemented)

- `cache_enabled=true`: prefill once, then each decode step feeds a single new token with `past_key_values`.
- `cache_enabled=false`: each decode step re-feeds the **entire** sequence with `use_cache=False`. For encoder-decoder models the encoder output is still reused; only the decoder is recomputed.
- Correction: before Phase 4 the cache-off path fed only the last token with no context, which is not a true recompute. Records created by `phase2-local-v1` with `cache_enabled=false` under-report latency and should not be compared with Phase 4 cache-off results.

## Studies

- **Scaling** (`POST /api/v1/studies/scaling`): sweeps input tokens, output tokens, or batch size (2–12 unique values). Reports a log-log exponent when ≥3 points exist; it describes the sampled range only. Note the engine tokenizes a repeated word, so actual token counts are stored per record.
- **Cache comparison** (`/studies/cache`): same workload with cache on and off; speedup = off / on latency. Requires ≥2 output tokens.
- **Multi-turn** (`/studies/multi-turn`, decoder-only): the same greedy conversation run two ways — `recompute_history` re-processes all prior tokens each turn, `reuse_kv_cache` keeps `past_key_values` across turns and processes only the unprocessed last token plus new tokens. One run per mode after a warmup; no repetition statistics and no memory measurement.

## Limitations

- Batch size is now materialised as `batch_size` identical prompts (previously it only scaled the reported tokens/s). Earlier records with `batch_size>1` measured a single sequence.
- Concurrent scheduling is not implemented.
- Results are specific to the device, model, and software versions in `runtime_metadata`.
