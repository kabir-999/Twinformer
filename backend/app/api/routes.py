from __future__ import annotations

import json
import os
from pathlib import Path
import platform
from fastapi import APIRouter, HTTPException, Response

from app.advanced.service import run_cache_comparison, run_multi_turn_study, run_scaling_study
from app.rag.answer import generate_answer
from app.models.local import list_local_models
from app.benchmark.engine import BenchmarkUnavailable, engine
from app.calibration.service import calibration_service
from app.schemas.workloads import RagAnswer, RagDocumentIn, RagEvalReport, RagEvalRequest, RagIndexRequest, RagIndexStatus, RagQuery, BenchmarkRecord, BenchmarkRequest, CacheComparison, CacheComparisonRequest, CalibrationReport, MultiTurnRequest, MultiTurnStudy, Overview, ScalingStudy, ScalingStudyRequest, TwinPrediction, WorkloadRequest
from app.storage.repository import BenchmarkRepository
from app.rag.service import RagService, RagUnavailable
from app.research.service import AblationReport, build_manifest, render_report, run_ablation, save_manifest
from app.compare.service import CompareRequest, CompareResult, CompareService, CompareUnavailable, RelatedPassage
from app.chat.service import Chat, ChatRequest, ChatSummary, new_chat, run_turn
from datetime import datetime, timezone
from app import twin_link
from app.secrets import get_groq_key
from app.rag.qa import clean_text, AnswersRequest, ThreeWayAnswer, answer_three_ways
from app.synthetic.generator import synthetic_record

router = APIRouter(prefix="/api/v1")
DATA_ROOT = os.getenv("TRANSFORMER_TWIN_DATA", "data")
def _source_folders() -> dict[str, Path]:
    """Document folders to index. Override with TRANSFORMER_TWIN_SOURCES (paths separated by ':')."""
    configured = os.getenv("TRANSFORMER_TWIN_SOURCES")
    paths = [Path(p).expanduser() for p in configured.split(os.pathsep) if p] if configured else [Path.home() / "Documents" / "MBA", Path.home() / "Documents" / "Sem 7"]
    return {p.name: p for p in paths if p.is_dir()}


rag_service = RagService(f"{DATA_ROOT}/documents", f"{DATA_ROOT}/rag", sources=_source_folders())
repo = BenchmarkRepository(os.getenv("TRANSFORMER_TWIN_DB", "data/benchmarks/transformer_twin.sqlite3"))


@router.get("/overview", response_model=Overview)
def overview() -> Overview:
    records = repo.list()
    calibration_status = calibration_service.last_report.status
    return Overview(
        record_count=len(records), synthetic_record_count=sum(r.source == "synthetic" for r in records),
        measured_record_count=sum(r.source == "measured" for r in records), calibration_status="trained" if calibration_status == "trained" else "untrained",
        device_summary=f"{platform.system()} {platform.machine()} (Phase 4: benchmarks, digital twin and cache/scaling studies enabled)",
        supported_defaults={"encoder_only": "distilbert/distilbert-base-uncased", "decoder_only": "Qwen/Qwen2.5-0.5B-Instruct", "encoder_decoder": "google/flan-t5-small"},
    )


@router.get("/records", response_model=list[BenchmarkRecord])
def records() -> list[BenchmarkRecord]:
    return repo.list()


@router.post("/simulation/synthetic", response_model=BenchmarkRecord, status_code=201)
def create_synthetic_simulation(workload: WorkloadRequest) -> BenchmarkRecord:
    record = synthetic_record(workload, len(repo.list()))
    repo.add(record)
    return record


@router.post("/twin/predict", response_model=TwinPrediction)
def predict(workload: WorkloadRequest) -> TwinPrediction:
    return calibration_service.predict(workload)


@router.post("/calibration/train", response_model=CalibrationReport)
def train_calibration() -> CalibrationReport:
    return calibration_service.train(repo.list())


@router.get("/calibration/report", response_model=CalibrationReport)
def calibration_report() -> CalibrationReport:
    return calibration_service.last_report


@router.post("/benchmarks", response_model=BenchmarkRecord, status_code=201)
def benchmark(request: BenchmarkRequest) -> BenchmarkRecord:
    try:
        record = engine.run(request)
    except BenchmarkUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    repo.add(record)
    return record


def _persist_study(compute):
    """Runs a study; records and the study are saved only if every run succeeded."""
    try:
        study, records = compute()
    except BenchmarkUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    for record in records:
        repo.add(record)
    repo.add_study(study)
    return study


@router.post("/studies/scaling", response_model=ScalingStudy, status_code=201)
def scaling_study(request: ScalingStudyRequest) -> ScalingStudy:
    return _persist_study(lambda: run_scaling_study(request, engine.run))


@router.post("/studies/cache", response_model=CacheComparison, status_code=201)
def cache_study(request: CacheComparisonRequest) -> CacheComparison:
    return _persist_study(lambda: run_cache_comparison(request, engine.run))


@router.post("/studies/multi-turn", response_model=MultiTurnStudy, status_code=201)
def multi_turn_study(request: MultiTurnRequest) -> MultiTurnStudy:
    return _persist_study(lambda: (run_multi_turn_study(request), []))


@router.get("/studies")
def studies(kind: str | None = None) -> list[dict]:
    return repo.list_studies(kind)


def _rag(call):
    try:
        return call()
    except RagUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/rag/documents", response_model=list[str])
def rag_documents() -> list[str]:
    """Files that are searchable (indexed); falls back to all discovered files before the first index build."""
    return rag_service.indexed_documents() or rag_service.list_documents()


@router.post("/rag/documents", status_code=201)
def rag_add_document(document: RagDocumentIn) -> dict[str, str]:
    rag_service.add_document(document.name, document.text)
    return {"name": document.name, "note": "Saved. Rebuild the index for it to be searchable."}


@router.post("/rag/index", response_model=RagIndexStatus)
def rag_build_index(request: RagIndexRequest, wait: bool = False) -> RagIndexStatus:
    """Starts indexing in the background (poll GET /rag/index); wait=true blocks until done."""
    return _rag(lambda: rag_service.build_index(request) if wait else rag_service.start_build(request))


@router.get("/rag/index", response_model=RagIndexStatus)
def rag_status() -> RagIndexStatus:
    return rag_service.status()


@router.post("/rag/query", response_model=RagAnswer)
def rag_query(query: RagQuery) -> RagAnswer:
    answer = _rag(lambda: rag_service.answer(query.question, query.top_k))
    for citation in answer.citations:
        citation.text = clean_text(citation.text)
    if query.generate:
        try:
            answer.answer, answer.answer_model, answer.answer_seconds = generate_answer(query.question, answer.citations, query.model_id, query.allow_model_download)
        except BenchmarkUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        answer.notes.append("The written answer comes from a small local model and can be wrong. Check it against the passages.")
    return answer


@router.post("/rag/evaluate", response_model=RagEvalReport)
def rag_evaluate(request: RagEvalRequest) -> RagEvalReport:
    return _rag(lambda: rag_service.evaluate(request))


@router.get("/rag/eval-cases")
def rag_eval_cases() -> list[dict]:
    path = Path(DATA_ROOT) / "rag" / "eval_cases.json"
    return json.loads(path.read_text()) if path.exists() else []


ARTIFACTS_ROOT = Path(os.getenv("TRANSFORMER_TWIN_ARTIFACTS", "artifacts"))


@router.post("/research/ablation", response_model=AblationReport)
def research_ablation() -> AblationReport:
    return run_ablation(repo.list())


@router.post("/research/manifest")
def research_manifest() -> dict:
    manifest = build_manifest(repo.list(), repo.list_studies(), rag_service.last_evaluation(), rag_service.status().model_dump())
    manifest["saved_to"] = str(save_manifest(manifest, ARTIFACTS_ROOT / "manifests"))
    return manifest


@router.post("/research/report")
def research_report() -> dict[str, str]:
    records, studies, rag_eval = repo.list(), repo.list_studies(), rag_service.last_evaluation()
    manifest = build_manifest(records, studies, rag_eval, rag_service.status().model_dump())
    manifest_path = save_manifest(manifest, ARTIFACTS_ROOT / "manifests")
    if calibration_service.last_report.status == "untrained":
        calibration_service.train(records)  # reports are computed from saved data, not from whatever happened to be trained earlier
    markdown = render_report(records, manifest, calibration_service.last_report, run_ablation(records), studies, rag_eval, str(manifest_path))
    report_path = ARTIFACTS_ROOT / "reports" / f"report-{manifest['created_at'][:19].replace(':', '')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown)
    return {"path": str(report_path), "manifest": str(manifest_path), "markdown": markdown}


@router.get("/models/local")
def local_models() -> list[dict[str, str]]:
    return list_local_models()


def _related_passages(text: str, k: int) -> list[RelatedPassage]:
    try:
        return [RelatedPassage(document=c.document, location=c.location, text=c.text, score=c.score) for c in rag_service.retrieve(text, k)]
    except RagUnavailable:
        return []


compare_service = CompareService(_related_passages, lambda: rag_service.status().embedding_model)


@router.get("/compare/models")
def compare_models() -> dict:
    return compare_service.available_models()


@router.post("/compare", response_model=CompareResult)
def compare(request: CompareRequest) -> CompareResult:
    try:
        result = compare_service.run(request)
    except CompareUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    repo.add_study(result)
    return result


@router.get("/judge/status")
def judge_status() -> dict[str, bool]:
    return {"groq_configured": bool(get_groq_key())}


@router.post("/rag/answers", response_model=ThreeWayAnswer)
def rag_answers(request: AnswersRequest) -> ThreeWayAnswer:
    return _rag(lambda: answer_three_ways(request, rag_service, compare_service, get_groq_key()))


@router.post("/chat", response_model=Chat)
def chat_turn(request: ChatRequest) -> Chat:
    """Adds one question to a chat (creating it if needed), answers it over the selected files, and saves the conversation."""
    chat = None
    if request.chat_id:
        stored = repo.get_chat(request.chat_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="That chat no longer exists.")
        chat = Chat.model_validate(stored)
    if chat is None:
        chat = new_chat(request)
    else:
        chat.documents = request.documents
    turn = _rag(lambda: run_turn(request, chat, rag_service, compare_service, get_groq_key()))
    twin_link.annotate_predictions(turn.result.answers, calibration_service)          # what the twin expected...
    for record in twin_link.records_from_answers(turn.result.answers):                # ...then it learns from what really happened
        repo.add(record)
    calibration_service.train(repo.list())
    chat.turns.append(turn)
    chat.updated_at = datetime.now(timezone.utc).isoformat()
    repo.save_chat(chat)
    return chat


@router.get("/chats", response_model=list[ChatSummary])
def chats() -> list[ChatSummary]:
    return [ChatSummary(id=c["id"], title=c["title"], updated_at=c["updated_at"], turn_count=len(c["turns"]), document_count=len(c["documents"])) for c in repo.list_chats()]


@router.get("/chats/{chat_id}", response_model=Chat)
def get_chat(chat_id: str) -> Chat:
    stored = repo.get_chat(chat_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return Chat.model_validate(stored)


@router.delete("/chats/{chat_id}", status_code=204, response_class=Response)
def delete_chat(chat_id: str):
    if not repo.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return Response(status_code=204)


@router.get("/twin/estimate")
def twin_estimate(top_k: int = 4) -> list[dict]:
    return twin_link.estimate(calibration_service, compare_service.available_models()["defaults"], max(1, min(top_k, 8)))


@router.get("/twin/status")
def twin_status() -> dict:
    return twin_link.status(repo.list(), calibration_service)
