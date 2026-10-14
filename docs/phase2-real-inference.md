# Phase 2: real local inference

`POST /api/v1/benchmarks` uses Hugging Face Transformers locally. It selects the adapter from the requested architecture:

| Architecture | Adapter | Default model |
|---|---|---|
| Encoder-only | `AutoModel` | `distilbert/distilbert-base-uncased` |
| Decoder-only | `AutoModelForCausalLM` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Encoder-decoder | `AutoModelForSeq2SeqLM` | `google/flan-t5-small` |

Downloads are disabled by default (`allow_model_download: false`). Cache a compatible model in Hugging Face's local cache or explicitly opt in to its download from the dashboard/API. Review each model card and license before enabling a download.

Each completed record stores the median of the requested repetitions, population standard deviation of end-to-end latency, actual token counts, and runtime metadata. Timing uses `perf_counter`; CUDA operations are synchronized before timing boundaries.

## Current limits

- Phase 2 supports a single in-flight request only. It rejects `concurrent_requests > 1` rather than reporting an invented concurrency metric.
- CPU memory is current process RSS sampled around a request; it is not a reliable request-only peak. CUDA memory is PyTorch's peak allocated memory and excludes driver/runtime allocations.
- The first uncached load reports loading time; later in-process cache hits report `0` loading time and `model_cached: true`.
- Decoder timings use greedy, manual token stepping. They characterize inference mechanics, not text quality or production-serving behavior.
- `int8` loading and queueing/batching policies remain future work.
