from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from app.rag.extract import SUPPORTED, extract_segments
from app.schemas.workloads import (
    Citation, RagAnswer, RagEvalCaseResult, RagEvalReport, RagEvalRequest, RagIndexRequest, RagIndexStatus, RagSource,
)

Embedder = Callable[[list[str]], np.ndarray]  # returns L2-normalised float32 vectors


class RagUnavailable(RuntimeError):
    """Raised when the local embedding model/index is unavailable."""


@dataclass
class Chunk:
    id: str
    document: str
    char_start: int
    char_end: int
    text: str
    location: str = ""


def chunk_text(document: str, text: str, chunk_words: int, overlap_words: int, locate: Callable[[int], str] | None = None) -> list[Chunk]:
    """Word-window chunks; char offsets point into the document text so citations can be verified."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    step = chunk_words - overlap_words
    chunks = []
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + chunk_words]
        if not window:
            break
        s, e = window[0][0], window[-1][1]
        location = ""
        if locate:
            first, last = locate(s), locate(e - 1)
            location = first if first == last else f"{first} – {last}"
        chunks.append(Chunk(f"{document}#{i}", document, s, e, text[s:e], location))
        if start + chunk_words >= len(words):
            break
    return chunks


def transformers_embedder(model_id: str, allow_download: bool) -> Embedder:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RagUnavailable("Local embedding dependencies are absent. Install backend/requirements.txt.") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=not allow_download)
        model = AutoModel.from_pretrained(model_id, local_files_only=not allow_download).eval()
    except Exception as exc:
        raise RagUnavailable(f"Could not load embedding model {model_id}. Cache it locally or set allow_model_download=true. Original error: {type(exc).__name__}: {exc}") from exc

    def embed(texts: list[str]) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                enc = tokenizer(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt")
                hidden = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pooling
                out.append(torch.nn.functional.normalize(vec, dim=-1).numpy())
        return np.concatenate(out).astype("float32")
    return embed


class RagService:
    def __init__(self, documents_dir: str, index_dir: str, embedder_factory: Callable[[str, bool], Embedder] = transformers_embedder,
                 sources: dict[str, Path] | None = None) -> None:
        self.documents_dir, self.index_dir = Path(documents_dir), Path(index_dir)
        self.sources = {label: Path(path) for label, path in (sources or {}).items()}
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "text").mkdir(parents=True, exist_ok=True)
        self._factory = embedder_factory
        self._embedder: Embedder | None = None
        self._embedder_key: tuple[str, bool] | None = None
        self._index = None
        self._vectors: np.ndarray | None = None
        self._bm25: _BM25 | None = None
        self._chunks: list[Chunk] = []
        self._meta: dict = {}
        self._lock = threading.Lock()
        self._progress: dict = {"state": "idle", "stage": None, "done": 0, "total": 0, "error": None, "skipped": []}
        self._load()

    # --- documents -------------------------------------------------------------------
    def add_document(self, name: str, text: str) -> None:
        (self.documents_dir / name).write_text(text, encoding="utf-8")

    def _files(self) -> dict[str, Path]:
        """Document name -> path. Files in documents_dir keep plain names; source folders are prefixed with their label."""
        found: dict[str, Path] = {}
        roots = [("", self.documents_dir, False)] + [(label, path, True) for label, path in self.sources.items()]
        for label, root, recursive in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*") if recursive else root.glob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED and not path.name.startswith(("~$", ".")):
                    found[f"{label}/{path.relative_to(root).as_posix()}" if label else path.name] = path
        return found

    def list_documents(self) -> list[str]:
        return list(self._files())

    def _fingerprint(self) -> str:
        return hashlib.sha256("|".join(f"{n}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for n, p in self._files().items()).encode()).hexdigest()

    def _document_text(self, name: str, path: Path) -> tuple[str, list[tuple[int, str]]]:
        """Extracted text plus [(start_offset, location)] markers; cached on disk keyed by file size/mtime."""
        cache = self.index_dir / "text" / (hashlib.sha1(name.encode()).hexdigest() + ".json")
        stat = path.stat()
        if cache.exists():
            data = json.loads(cache.read_text())
            if data["size"] == stat.st_size and data["mtime"] == int(stat.st_mtime):
                return data["text"], [tuple(m) for m in data["markers"]]
        parts, markers, offset = [], [], 0
        for label, text in extract_segments(path):
            markers.append((offset, label))
            parts.append(text)
            offset += len(text) + 2
        text = "\n\n".join(parts)
        cache.write_text(json.dumps({"size": stat.st_size, "mtime": int(stat.st_mtime), "text": text, "markers": markers}))
        return text, markers

    def segments(self, name: str) -> list[tuple[str, str]]:
        """The document split into its natural units: [("slide 1", text), ("slide 2", text), ...] or pages."""
        text, markers = self._document_text(name, self._files()[name])
        out = []
        for i, (start, label) in enumerate(markers):
            end = markers[i + 1][0] - 2 if i + 1 < len(markers) else len(text)
            out.append((label, text[start:end].strip()))
        return out

    def document_text(self, name: str) -> str:
        return self._document_text(name, self._files()[name])[0]

    # --- index -----------------------------------------------------------------------
    def _embed(self, texts: list[str], model_id: str, allow_download: bool) -> np.ndarray:
        key = (model_id, allow_download)
        if self._embedder is None or self._embedder_key != key:
            self._embedder, self._embedder_key = self._factory(model_id, allow_download), key
        return self._embedder(texts)

    def _set(self, **values) -> None:
        self._progress.update(values)

    def build_index(self, request: RagIndexRequest) -> RagIndexStatus:
        self._set(state="building", stage="Reading documents", done=0, total=0, error=None, skipped=[])
        try:
            files = self._files()
            chunks, skipped = [], []
            for n, (name, path) in enumerate(files.items()):
                self._set(stage=f"Reading {name}", done=n, total=len(files))
                try:
                    text, markers = self._document_text(name, path)
                except Exception as exc:  # unreadable or corrupt file: skip it, say so
                    skipped.append(f"{name}: {type(exc).__name__}")
                    continue
                if not text.strip():
                    skipped.append(f"{name}: no extractable text (scanned images?)")
                    continue
                starts = [m[0] for m in markers]

                def locate(offset: int, starts=starts, markers=markers) -> str:
                    lo, hi = 0, len(starts) - 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        lo, hi = (mid, hi) if starts[mid] <= offset else (lo, mid - 1)
                    return markers[lo][1]
                chunks += chunk_text(name, text, request.chunk_words, request.overlap_words, locate)
            if not chunks:
                raise RagUnavailable("No document text to index. Add documents or check the source folders.")
            order = sorted(range(len(chunks)), key=lambda i: len(chunks[i].text))  # similar lengths per batch = less padding
            vectors = np.zeros((len(chunks), 384), dtype="float32")
            for done in range(0, len(order), 128):
                self._set(stage="Embedding passages", done=done, total=len(chunks))
                ids = order[done:done + 128]
                batch = self._embed([chunks[i].text for i in ids], request.embedding_model, request.allow_model_download)
                if vectors.shape[1] != batch.shape[1]:
                    vectors = np.zeros((len(chunks), batch.shape[1]), dtype="float32")
                vectors[ids] = batch
            index, backend = self._make_index(vectors)
            meta = {"embedding_model": request.embedding_model, "chunk_words": request.chunk_words, "overlap_words": request.overlap_words,
                    "built_at": datetime.now(timezone.utc).isoformat(), "document_count": len(files) - len(skipped), "chunk_count": len(chunks),
                    "index_backend": backend, "fingerprint": self._fingerprint()}
            with self._lock:
                self._index, self._chunks, self._meta = index, chunks, meta
                self._vectors, self._bm25 = vectors, _BM25([c.text for c in chunks])
            np.save(self.index_dir / "vectors.npy", vectors)
            (self.index_dir / "chunks.json").write_text(json.dumps([c.__dict__ for c in chunks]))
            (self.index_dir / "meta.json").write_text(json.dumps(meta))
            self._set(state="ready", stage="Done", done=len(chunks), total=len(chunks), skipped=skipped)
        except Exception as exc:
            self._set(state="error", error=str(exc))
            raise
        return self.status()

    def start_build(self, request: RagIndexRequest) -> RagIndexStatus:
        """Builds in a background thread so the UI can show progress."""
        if self._progress["state"] == "building":
            return self.status()
        self._set(state="building", stage="Starting", done=0, total=0, error=None)

        def work() -> None:
            try:
                self.build_index(request)
            except Exception:
                pass  # recorded in progress["error"]
        threading.Thread(target=work, daemon=True).start()
        return self.status()

    @staticmethod
    def _make_index(vectors: np.ndarray):
        try:
            import torch  # noqa: F401  # see app/__init__.py: torch + faiss need KMP_DUPLICATE_LIB_OK on macOS
        except ImportError:
            pass
        try:
            import faiss
        except ImportError:
            return _NumpyIndex(vectors), "numpy_exact_inner_product"
        index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; cosine because vectors are normalised
        index.add(vectors)
        return index, "faiss.IndexFlatIP"

    def _load(self) -> None:
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            return
        self._meta = json.loads(meta_path.read_text())
        self._chunks = [Chunk(**c) for c in json.loads((self.index_dir / "chunks.json").read_text())]
        self._vectors = np.load(self.index_dir / "vectors.npy")
        self._index, _ = self._make_index(self._vectors)
        self._bm25 = _BM25([c.text for c in self._chunks])
        self._progress["state"] = "ready"

    def status(self) -> RagIndexStatus:
        files = self._files()
        sources = [RagSource(label=label, path=str(path), file_count=sum(n.startswith(label + "/") for n in files)) for label, path in self.sources.items()]
        local = sum("/" not in n for n in files)
        if local:
            sources.insert(0, RagSource(label="Uploaded", path=str(self.documents_dir), file_count=local))
        base = dict(state=self._progress["state"], stage=self._progress["stage"], done=self._progress["done"], total=self._progress["total"],
                    error=self._progress["error"], skipped=self._progress["skipped"], sources=sources)
        if self._index is None:
            return RagIndexStatus(indexed=False, document_count=len(files), **base)
        changed = self._meta.get("fingerprint") != self._fingerprint()
        return RagIndexStatus(indexed=True, documents_changed=changed, **base,
                              **{k: self._meta.get(k) for k in ("document_count", "chunk_count", "embedding_model", "index_backend", "chunk_words", "overlap_words", "built_at")})

    # --- retrieval -------------------------------------------------------------------
    def indexed_documents(self) -> list[str]:
        return sorted({c.document for c in self._chunks})

    def retrieve(self, question: str, top_k: int, allow_download: bool = False, hybrid: bool = True, documents: set[str] | None = None) -> list[Citation]:
        """Dense (embedding) search, optionally fused with BM25 keyword search by reciprocal rank fusion.
        `documents` restricts the search to those files only."""
        with self._lock:
            index, chunks, meta, vectors, bm25 = self._index, self._chunks, self._meta, self._vectors, self._bm25
        if index is None:
            raise RagUnavailable("No index has been built yet. Click “Index documents” first.")
        query = self._embed([question], meta["embedding_model"], allow_download)
        if documents:
            allowed = [i for i, c in enumerate(chunks) if c.document in documents]
            if not allowed:
                raise RagUnavailable("None of the selected files are in the index. Re-index, or pick other files.")
            allowed_set = set(allowed)
            pool = min(len(allowed), max(50, top_k * 10))
            scores = vectors[allowed] @ query[0]
            dense = [allowed[j] for j in np.argsort(-scores)[:pool]]
        else:
            allowed_set = None
            pool = min(len(chunks), max(50, top_k * 10))
            _, dense_ids = index.search(query, pool)
            dense = [int(i) for i in dense_ids[0] if i >= 0]
        if hybrid and bm25 is not None:
            fused: dict[int, float] = {}
            for ranking in (dense, bm25.search(question, pool, allowed_set)):
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
            chosen = sorted(fused, key=lambda i: -fused[i])[:top_k]
        else:
            chosen = dense[:top_k]
        cosines = vectors[chosen] @ query[0]
        return [Citation(rank=r + 1, document=chunks[i].document, chunk_id=chunks[i].id, char_start=chunks[i].char_start,
                         char_end=chunks[i].char_end, location=chunks[i].location, score=round(float(cosines[r]), 4), text=chunks[i].text)
                for r, i in enumerate(chosen)]

    def answer(self, question: str, top_k: int) -> RagAnswer:
        return RagAnswer(question=question, citations=self.retrieve(question, top_k), notes=[
            "These are passages found in your files, shown word for word. Similarity scores only rank the passages; they are not a measure of correctness.",
        ])

    # --- evaluation ------------------------------------------------------------------
    def evaluate(self, request: RagEvalRequest, hybrid: bool = True) -> RagEvalReport:
        """Labels come from the caller (document + phrase that must appear), never from the retriever's own scores."""
        max_k = max(request.k_values)
        results, latencies = [], []
        for case in request.cases:
            first = None
            started = time.perf_counter()
            retrieved = self.retrieve(case.question, max_k, hybrid=hybrid)
            latencies.append((time.perf_counter() - started) * 1000)
            for c in retrieved:
                if c.document == case.document and case.must_contain.lower() in c.text.lower():
                    first = c.rank
                    break
            results.append(RagEvalCaseResult(question=case.question, first_relevant_rank=first))
        n = len(results)
        hits = {str(k): round(sum(r.first_relevant_rank is not None and r.first_relevant_rank <= k for r in results) / n, 4) for k in sorted(set(request.k_values))}
        mrr = round(sum(1 / r.first_relevant_rank for r in results if r.first_relevant_rank) / n, 4)
        report = RagEvalReport(case_count=n, retrieval_latency_ms_median=round(statistics.median(latencies), 3), evaluated_at=datetime.now(timezone.utc).isoformat(),
                               chunk_words=self._meta.get("chunk_words"), overlap_words=self._meta.get("overlap_words"),
                               hit_rate_at_k=hits, mrr=mrr, per_case=results, embedding_model=self._meta["embedding_model"], limitations=[
            "A chunk counts as relevant only if it comes from the labelled document and contains the labelled phrase; correct passages phrased differently count as misses.",
            "Results depend on chunk size/overlap and on the labelled cases; small case sets give wide uncertainty.",
            "Retrieval quality only; answer faithfulness is not evaluated.",
        ])
        (self.index_dir / "last_eval.json").write_text(report.model_dump_json())
        return report

    def last_evaluation(self) -> RagEvalReport | None:
        path = self.index_dir / "last_eval.json"
        return RagEvalReport.model_validate_json(path.read_text()) if path.exists() else None


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of and or to in on for is are was were be by with as at from that this it its what which who how why when where do does did can".split())


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class _BM25:
    """Small in-memory BM25 so exact keywords (e.g. "quant", "DVFS") can rescue questions the embeddings blur."""
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths = np.zeros(len(texts), dtype="float32")
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for term in _terms(text):
                counts[term] = counts.get(term, 0) + 1
            self.lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((i, tf))
        self.avg = float(self.lengths.mean()) if len(texts) else 1.0
        self.n = len(texts)

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[int]:
        scores: dict[int, float] = {}
        for term in set(_terms(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = np.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                if allowed is not None and i not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                scores[i] = scores.get(i, 0.0) + float(idf) * tf * (self.k1 + 1) / norm
        return sorted(scores, key=lambda i: -scores[i])[:k]


class _NumpyIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query: np.ndarray, k: int):
        scores = query @ self.vectors.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids
