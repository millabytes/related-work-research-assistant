from fastapi import FastAPI
from fastapi.responses import JSONResponse

from related_work_assistant.api.evaluations import router as evaluations_router
from related_work_assistant.api.papers import router as papers_router
from related_work_assistant.core.config import Settings
from related_work_assistant.database.health import check_database
from related_work_assistant.database.session import make_engine

app = FastAPI(title="Related Work Assistant", version="0.1.0")
app.include_router(papers_router)
app.include_router(evaluations_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    engine = None
    try:
        engine = make_engine(Settings())
        if check_database(engine):
            return JSONResponse({"status": "ready"})
    except Exception:
        pass
    finally:
        if engine is not None:
            engine.dispose()
    return JSONResponse({"status": "unavailable"}, status_code=503)
