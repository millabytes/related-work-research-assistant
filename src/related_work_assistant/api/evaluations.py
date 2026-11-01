"""Read-only API for explicitly generated, versioned evaluation artifacts."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from related_work_assistant.evaluation.generation.storage import (
    latest_evidence_for_paper,
    list_runs,
    load_latest,
)

router = APIRouter(prefix="/evaluations/generation", tags=["generation-evaluation"])


@router.get("/benchmark/latest")
def latest_benchmark() -> dict:
    result = load_latest("evaluation_a")
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation A has not been run")
    return result


@router.get("/evidence/fixed/latest")
def latest_fixed_evidence() -> dict:
    result = next(
        (
            run
            for run in list_runs("evaluation_b", limit=100)
            if run.get("config", {}).get("fixed_set") is True
        ),
        None,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation B has not been run")
    return result


@router.get("/evidence/{paper_id}/latest")
def latest_paper_evidence(paper_id: UUID) -> dict:
    result = latest_evidence_for_paper(str(paper_id))
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Evaluation B has not been run for paper {paper_id}"
        )
    return result


@router.get("/runs")
def evaluation_runs(
    kind: Annotated[str, Query(pattern="^(a|b)$")] = "a",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    return list_runs(f"evaluation_{kind}", limit=limit)
