from uuid import uuid4

import numpy as np

from related_work_assistant.retrieval.chunks import ChunkPair, ChunkPassage
from related_work_assistant.retrieval.sentences import (
    extract_mapped_sentences,
    rerank_sentence_pairs,
    sentence_candidates,
)


def passage(text: str, status: str = "mapped") -> ChunkPassage:
    return ChunkPassage(
        chunk_id=uuid4(),
        text=text,
        section_heading="Methods",
        section_type="methods",
        page_start=2,
        page_end=2,
        source_mapping=[
            {
                "grobid_sentence_id": "s1",
                "grobid_text": text,
                "mapping_status": status,
                "text_similarity": 0.98,
                "pymupdf_regions": [
                    {
                        "page": 2,
                        "pymupdf_bbox": [10.0, 20.0, 200.0, 32.0],
                        "coordinate_system": "pdf_points_top_left_unrotated",
                    }
                ],
            }
        ],
    )


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.array([1.0, 0.0]) if "latent" in text else np.array([0.0, 1.0])
            vectors.append(vector)
        return np.stack(vectors)


class FakeScorer:
    def score(self, text_pairs):
        return np.arange(len(text_pairs), dtype=np.float32)


def test_only_reliable_original_mappings_become_sentence_evidence():
    mapped = extract_mapped_sentences(passage("Original GROBID sentence about latent factors."))
    assert len(mapped) == 1
    assert mapped[0].text == "Original GROBID sentence about latent factors."
    assert mapped[0].coordinates[0].bbox == (10.0, 20.0, 200.0, 32.0)
    assert extract_mapped_sentences(passage("Weak mapping.", status="weak")) == []


def test_bge_sentence_candidates_and_cross_encoder_reranking():
    chunk_pairs = [
        ChunkPair(
            passage("A latent factor sentence."),
            passage("Another latent representation sentence."),
            similarity_score=0.9,
            rank=1,
        ),
        ChunkPair(
            passage("An evaluation sentence."),
            passage("A benchmark evaluation sentence."),
            similarity_score=0.8,
            rank=2,
        ),
    ]
    candidates = sentence_candidates(chunk_pairs, FakeEncoder(), candidate_count=2)
    assert len(candidates) == 2
    assert all(pair.bge_similarity_score == 1.0 for pair in candidates)
    reranked = rerank_sentence_pairs(candidates, FakeScorer(), final_count=2)
    assert [pair.reranking_score for pair in reranked] == [1.0, 0.0]
    assert [pair.rank for pair in reranked] == [1, 2]
