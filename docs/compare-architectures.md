# Compare tab: one document passage, three Transformer designs

In the UI the input always comes from your indexed folders (`~/Documents/MBA`, `~/Documents/Sem 7`): ask a question, pick one of the retrieved passages, and all three models work on that passage (answer the question from it, or summarise it). The API also accepts arbitrary `text`.

`POST /api/v1/compare` runs the same text through an encoder-only, a decoder-only and an encoder-decoder model and returns, per model:

- tokens (each tokenizer splits the text differently), parameter count, weight memory
- timings: tokenise, read the input (encoder forward / decoder prefill), write the output (generation)
- output: an embedding for the encoder; greedy-decoded text for the other two
- real attention weights (last layer, mean over heads): bidirectional self-attention, causal self-attention, and cross-attention
- the top-5 next-token probabilities at each generation step
- for the encoder, the nearest passages from your indexed documents when it is the same model as the search index

Default models (all small, run on CPU): `all-MiniLM-L6-v2` (22.7M), `Qwen2.5-0.5B-Instruct` (494M), `flan-t5-base` (248M); the app falls back to any cached model of the right type. Any locally cached model of the right type can be selected through the API.

## What the comparison does and does not show

- Models differ in size and training data, not just architecture. A 464M decoder is slower than a 77M encoder-decoder largely because it is 6× bigger. The UI says so.
- Each family gets the prompt style it was trained on (e.g. `Summarize: …` for T5, `Question: … Answer:` for the base decoder).
- Attention maps show where a model looks, not a full explanation of its output.
- Models load with eager attention so weights can be read; timings are therefore not comparable with the SDPA-based benchmark tab.
- Generation timing is a single run after a warm-up; encode/prefill timing is the median of 3.
- Inputs are cut at 384 tokens. Heatmaps show the 32 most-attended tokens (kept in original order, labelled with their positions), so long passages stay readable.
- The encoder card also shows the cosine similarity between your question and the passage.
- flan-t5-small is weak: it answers "What is the capital of France?" in French-looking gibberish (`l'ondre`). That is the checkpoint, not the pipeline (verified with plain `generate`).
