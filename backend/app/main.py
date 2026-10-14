from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import calibration_service, repo, router
from app.synthetic.generator import demo_records


@asynccontextmanager
async def lifespan(_: FastAPI):
    for record in demo_records():
        repo.add(record)
    calibration_service.train(repo.list())   # the twin starts from everything measured so far
    yield


app = FastAPI(title="TransformerTwin", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "6"}
