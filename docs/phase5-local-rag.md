# Phase 5 — Local RAG

- **Corpus**: by default `~/Documents/MBA` and `~/Documents/Sem 7` (override with `TRANSFORMER_TWIN_SOURCES`, paths separated by `:`), plus `.md`/`.txt` files uploaded to `data/documents` via `POST /api/v1/rag/documents` (names restricted to `[A-Za-z0-9._-]`). Supported: PDF, PPTX, DOCX, MD, TXT. Extracted text is cached in `data/rag/text`. Unreadable files and PDFs with no text layer are skipped and listed in the index status; nothing is OCR'd.
- **Locations**: citations carry the page (PDF), slide (PPTX) or heading (DOCX) the passage starts in.
- **Indexing** runs in a background thread (`POST /rag/index`, poll `GET /rag/index`); the status reports when files changed since the last build.
- **Chunking**: word windows (default 120 words, 30 overlap). Each chunk stores character offsets into its source file, so every citation can be checked against the document.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` loaded with plain `transformers` (mean pooling, L2-normalised). Loaded from the local cache only unless `allow_model_download=true`.
- **Index**: exact `faiss.IndexFlatIP` (cosine on normalised vectors), persisted to `data/rag/`. Falls back to exact NumPy search if FAISS is not installed.
- **Query**: `POST /api/v1/rag/query` returns ranked passages with document, chunk id, offsets and cosine score. Retrieval returns passages only; `generate: true` additionally asks a local chat model to write an answer from the top passages. Written answers from small models can be wrong (a 0.5B model misquoted a figure in testing), so the passages are the source of truth. Scores are not calibrated probabilities.
- **Evaluation**: `POST /api/v1/rag/evaluate` takes caller-supplied cases (`question`, `document`, `must_contain`). A hit is a retrieved chunk from the labelled document that contains the labelled phrase; labels never come from the retriever. Reports hit-rate@k, MRR and median retrieval latency, and saves the result for the Phase 6 report. The bundled `data/rag/eval_cases.json` (8 cases over the Sem 7 slides and CAT syllabus; on the 17-file corpus: hit@1 0.44, hit@3 0.89) was written by the project author, so treat it as a smoke-level benchmark.

## Platform note

On macOS, importing `faiss` before `torch` segfaults (duplicate OpenMP runtimes). The index code imports `torch` first; keep that order if you modify it.

## Three-way answers and the judge (Documents tab)

`POST /api/v1/rag/answers` retrieves passages, then answers the question with one model from each family, all from the same passages:

- **Encoder-only** (MiniLM): an encoder cannot write text, so it does extractive answering: it embeds every sentence of the passages and returns the two closest to the question, with `[n]` source markers. Long unpunctuated runs (slides, lists) are wrapped at ~260 characters.
- **Decoder-only** (prefers an instruct model, e.g. Qwen2.5-0.5B-Instruct): writes a short answer with the chat template.
- **Encoder-decoder** (prefers flan-t5-base/large): reads the context with the encoder, decoder writes the answer (beam search, minimum 10 tokens so it does not answer with one word).

**Retrieval** is hybrid: the embedding top-50 is fused with a BM25 keyword ranking by reciprocal rank fusion. On the 8 labelled cases this moved hit@1 0.38 to 0.50, hit@3 0.88 to 1.00 and MRR 0.60 to 0.71 (small set; indicative only).

**Judging** (`Which was best?`):

1. *Automatic checks* (always): share of the answer's content words found in the sources, question-to-answer similarity, repeated 3-word phrases. Scores within 5 points are reported as a tie. These measure support and relevance, not truth: a fluent answer built from the wrong passage still scores well, and the encoder-only answer is 100% supported by construction.
2. *Groq judge* (opt-in, `use_groq`): a large model compares the answers against the sources. Names are hidden and order is shuffled to reduce bias; it returns correctness/grounding/completeness scores and reasoning. This sends the question, passages and answers to Groq. The key comes from `GROQ_API_KEY` (environment or `.env`) or is pasted in the UI, where it is kept in memory only.

## Chat endpoints

- `POST /api/v1/chat` `{chat_id?, question, documents[], mode: "all"|"auto", top_k}`: retrieves only from `documents` (empty = all files), rewrites follow-ups into standalone questions (Groq if a key exists, otherwise `"<previous question> — <follow-up>"`), runs the three models (or the routed one), judges, and saves the chat.
- `GET /api/v1/chats`, `GET /api/v1/chats/{id}`, `DELETE /api/v1/chats/{id}`: history.
- Routing (`mode: "auto"`) is rule-based so the reason shown is the real reason: sentiment → encoder-only classifier (DistilBERT SST-2); translation → encoder-decoder (below); summaries and open questions → decoder-only; "where / which slide / find" lookups → encoder-only extraction.
- **Positions, not topics.** "first slide", "last page", "slide 7", "page 12" are read directly from the file (`app/rag/locator.py`), for every question type. Semantic search is skipped because a position is not a topic. If the unit does not exist in the selected files, the closest passages are used and a note says so.
- **Translation** (any mode, `app/rag/translate.py`): the located slide/page (or best passages) is translated, not summarised. The encoder-decoder uses a dedicated translation model, preferring NLLB-200-distilled-600M (one model for 21 languages including Hindi, Tamil, Telugu, Bengali, Marathi) and falling back to a cached per-language MarianMT model (`opus-mt-en-xx`). The decoder-only model is asked to translate too, for comparison. Encoder-only models cannot generate text and sit out. Text is split into lines/sentences/bullets, long ALL-CAPS words are normalised, and pieces are translated in batches, because gluing slide fragments into one run produced garbage. With a Groq key, a judge compares the two translations; without one, the app says translation quality cannot be checked automatically.
- Missing models are fetched only when "Allow downloads" is ticked (NLLB is about 2.5 GB).
- The Groq key is read from `.env` on every request (never stored by the app or sent to the browser).
- Platform note: `app/__init__.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` because torch and faiss each bundle an OpenMP runtime and FAISS search aborts on macOS without it.
