from uuid import UUID

from pydantic import BaseModel, Field


class RelatedWorkRequest(BaseModel):
    models: list[str] | None = None


class PaperListResponse(BaseModel):
    paper_id: UUID
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    has_specter2_embedding: bool


class ChunkPassageResponse(BaseModel):
    chunk_id: UUID
    text: str
    section_heading: str | None
    section_type: str | None
    page_start: int | None
    page_end: int | None
    source_mapping: list[dict[str, object]]


class ChunkPairResponse(BaseModel):
    query_chunk: ChunkPassageResponse
    recommended_chunk: ChunkPassageResponse
    similarity_score: float
    reranking_score: float | None
    rank: int


class PaperRecommendationResponse(BaseModel):
    paper_id: UUID
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    similarity_score: float
    rank: int
    chunk_pairs: list[ChunkPairResponse]


class SentenceRegionResponse(BaseModel):
    page: int
    bbox: tuple[float, float, float, float]
    coordinate_system: str


class SentenceEvidenceResponse(BaseModel):
    sentence_id: str | None
    chunk_id: UUID
    text: str
    section_heading: str | None
    section_type: str | None
    page_start: int
    page_end: int
    coordinates: list[SentenceRegionResponse]
    mapping_status: str
    text_similarity: float | None


class SentencePairResponse(BaseModel):
    query_sentence: SentenceEvidenceResponse
    recommended_sentence: SentenceEvidenceResponse
    bge_similarity_score: float
    reranking_score: float | None
    parent_chunk_pair_rank: int
    rank: int


class EvidenceResponse(BaseModel):
    bge_model_version_id: UUID
    bge_checkpoint_revision: str
    reranker_model_version_id: UUID
    bge_chunk_candidates: list[ChunkPairResponse]
    reranked_chunk_pairs: list[ChunkPairResponse]
    bge_sentence_candidates: list[SentencePairResponse]
    sentence_pairs: list[SentencePairResponse]
    warnings: list[str]


class PDFHighlight(BaseModel):
    bbox: tuple[float, float, float, float]
    color: str = Field(default="#FFD54F", pattern=r"^#[0-9A-Fa-f]{6}$")


class PDFRenderRequest(BaseModel):
    highlights: list[PDFHighlight] = Field(default_factory=list, max_length=100)
    zoom: float = Field(default=1.6, ge=0.75, le=3.0)
