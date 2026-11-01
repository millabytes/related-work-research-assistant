from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import Engine

from related_work_assistant.api.dependencies import get_engine
from related_work_assistant.api.schemas import (
    EvidenceResponse,
    PaperListResponse,
    PaperRecommendationResponse,
    PDFRenderRequest,
    RelatedWorkRequest,
)
from related_work_assistant.core.config import Settings
from related_work_assistant.generation.service import enqueue, related_work_status
from related_work_assistant.pdf_viewer import (
    PDFNotFoundError,
    paper_pdf_info,
    render_pdf_page,
)
from related_work_assistant.retrieval.chunks import (
    ChunkEmbeddingNotFoundError,
    retrieve_chunk_pairs,
)
from related_work_assistant.retrieval.evidence import retrieve_evidence
from related_work_assistant.retrieval.papers import (
    InvalidPaperFilterError,
    PaperEmbeddingNotFoundError,
    PaperNotFoundError,
    list_papers,
    recommend_papers,
)
from related_work_assistant.retrieval.reranking import (
    finalize_bge_pairs,
    get_chunk_reranker,
    register_reranker_version,
    rerank_chunk_pairs,
)

router = APIRouter(prefix="/papers", tags=["papers"])
Database = Annotated[Engine, Depends(get_engine)]


@router.get("/{paper_id}/related-work")
def get_related_work(
    paper_id: UUID,
    engine: Database,
    model: Annotated[str | None, Query(pattern="^(primera|qwen|qwen_lora|gemma)$")] = None,
) -> dict[str, object]:
    """Read cached generation results, active status, errors and version history."""
    try:
        return related_work_status(engine, paper_id, model)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{paper_id}/related-work/generate", status_code=status.HTTP_202_ACCEPTED)
def generate_related_work(
    paper_id: UUID, request: RelatedWorkRequest, engine: Database
) -> dict[str, object]:
    """Enqueue missing PRIMERA/Qwen/Qwen-LoRA results; never infer in the request."""
    try:
        jobs = enqueue(engine, Settings(), paper_id, request.models, force=False)
        return {"paper_id": paper_id, "job_ids": jobs, "cached_or_active": not jobs}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{paper_id}/related-work/regenerate", status_code=status.HTTP_202_ACCEPTED)
def regenerate_related_work(
    paper_id: UUID, request: RelatedWorkRequest, engine: Database
) -> dict[str, object]:
    """Enqueue a new history-preserving version for selected generators."""
    try:
        jobs = enqueue(engine, Settings(), paper_id, request.models, force=True)
        return {"paper_id": paper_id, "job_ids": jobs, "already_active": not jobs}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=list[PaperListResponse])
def get_papers(
    engine: Database,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    has_embedding: bool | None = None,
) -> list[PaperListResponse]:
    """List selectable papers and whether each has a SPECTER2 retrieval vector."""
    return [
        PaperListResponse.model_validate(item, from_attributes=True)
        for item in list_papers(engine, limit=limit, offset=offset, has_embedding=has_embedding)
    ]


@router.get("/{paper_id}/recommendations", response_model=list[PaperRecommendationResponse])
def get_recommendations(
    paper_id: UUID,
    engine: Database,
    top_k: Annotated[int, Query(ge=1, le=100)] = 5,
    year_from: Annotated[int | None, Query(ge=1800, le=2100)] = None,
    year_to: Annotated[int | None, Query(ge=1800, le=2100)] = None,
    venue: Annotated[str | None, Query(max_length=300)] = None,
    chunk_pairs_per_paper: Annotated[int, Query(ge=0, le=10)] = 3,
    chunk_candidates_per_paper: Annotated[int, Query(ge=1, le=100)] = 20,
    rerank: bool = True,
) -> list[PaperRecommendationResponse]:
    """Return SPECTER2-nearest papers and their most relevant BGE chunk pairs."""
    try:
        recommendations = recommend_papers(
            engine,
            paper_id,
            top_k=top_k,
            year_from=year_from,
            year_to=year_to,
            venue=venue,
        )
        candidate_count = 0
        if chunk_pairs_per_paper:
            candidate_count = (
                max(chunk_candidates_per_paper, chunk_pairs_per_paper)
                if rerank
                else chunk_pairs_per_paper
            )
        chunk_candidates = retrieve_chunk_pairs(
            engine,
            paper_id,
            [recommendation.paper_id for recommendation in recommendations],
            pairs_per_paper=candidate_count,
        )
        if rerank and chunk_pairs_per_paper and any(chunk_candidates.values()):
            reranker = get_chunk_reranker()
            register_reranker_version(engine, reranker.descriptor)
            chunk_pairs = rerank_chunk_pairs(
                chunk_candidates,
                reranker,
                final_pairs_per_paper=chunk_pairs_per_paper,
            )
        else:
            chunk_pairs = finalize_bge_pairs(chunk_candidates, chunk_pairs_per_paper)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PaperEmbeddingNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidPaperFilterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChunkEmbeddingNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return [
        PaperRecommendationResponse.model_validate(
            {
                **recommendation.__dict__,
                "chunk_pairs": chunk_pairs[recommendation.paper_id],
            },
            from_attributes=True,
        )
        for recommendation in recommendations
    ]


@router.get(
    "/{paper_id}/recommendations/{recommended_paper_id}/evidence",
    response_model=EvidenceResponse,
)
def get_sentence_evidence(
    paper_id: UUID,
    recommended_paper_id: UUID,
    engine: Database,
    chunk_candidates: Annotated[int, Query(ge=3, le=100)] = 20,
    chunk_pairs: Annotated[int, Query(ge=1, le=10)] = 3,
    sentence_candidates: Annotated[int, Query(ge=3, le=100)] = 20,
    sentence_pairs: Annotated[int, Query(ge=1, le=10)] = 3,
) -> EvidenceResponse:
    """Build mapped sentence evidence for one selected recommendation."""
    try:
        bundle = retrieve_evidence(
            engine,
            paper_id,
            recommended_paper_id,
            chunk_candidate_count=chunk_candidates,
            chunk_pair_count=chunk_pairs,
            sentence_candidate_count=sentence_candidates,
            sentence_pair_count=sentence_pairs,
        )
    except ChunkEmbeddingNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EvidenceResponse.model_validate(bundle, from_attributes=True)


@router.get("/{paper_id}/pdf")
def get_paper_pdf(paper_id: UUID, engine: Database) -> FileResponse:
    """Return the untouched original PDF associated with a paper."""
    try:
        info = paper_pdf_info(engine, Settings(), paper_id)
    except PDFNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(info.path, media_type="application/pdf", filename=info.filename)


@router.post("/{paper_id}/pdf/pages/{page_number}/render")
def render_paper_pdf_page(
    paper_id: UUID,
    page_number: int,
    request: PDFRenderRequest,
    engine: Database,
) -> Response:
    """Render one page with temporary highlights; never modify the source PDF."""
    try:
        info = paper_pdf_info(engine, Settings(), paper_id)
        image = render_pdf_page(
            info,
            page_number,
            [highlight.model_dump() for highlight in request.highlights],
            zoom=request.zoom,
        )
    except PDFNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        image,
        media_type="image/png",
        headers={"X-PDF-Page": str(page_number), "X-PDF-Page-Count": str(info.page_count)},
    )
