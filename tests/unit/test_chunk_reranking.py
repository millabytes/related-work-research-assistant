from uuid import uuid4

import numpy as np

from related_work_assistant.retrieval.chunks import ChunkPair, ChunkPassage
from related_work_assistant.retrieval.reranking import (
    finalize_bge_pairs,
    rerank_chunk_pairs,
)


def passage(text):
    return ChunkPassage(
        chunk_id=uuid4(),
        text=text,
        section_heading="Section",
        section_type="other",
        page_start=1,
        page_end=1,
        source_mapping=[],
    )


class FakeReranker:
    def score(self, text_pairs):
        assert text_pairs == [("query one", "candidate one"), ("query two", "candidate two")]
        return np.array([-2.0, 4.0], dtype=np.float32)


def test_reranking_reorders_candidates_and_preserves_bge_scores():
    paper_id = uuid4()
    candidates = {
        paper_id: [
            ChunkPair(passage("query one"), passage("candidate one"), 0.99, 1),
            ChunkPair(passage("query two"), passage("candidate two"), 0.80, 2),
        ]
    }
    result = rerank_chunk_pairs(
        candidates, FakeReranker(), final_pairs_per_paper=2
    )[paper_id]
    assert [pair.query_chunk.text for pair in result] == ["query two", "query one"]
    assert [pair.reranking_score for pair in result] == [4.0, -2.0]
    assert [pair.similarity_score for pair in result] == [0.80, 0.99]
    assert [pair.rank for pair in result] == [1, 2]


def test_disabled_reranking_keeps_bge_order_and_null_score():
    paper_id = uuid4()
    candidates = {
        paper_id: [
            ChunkPair(passage("first"), passage("first candidate"), 0.9, 1),
            ChunkPair(passage("second"), passage("second candidate"), 0.8, 2),
        ]
    }
    result = finalize_bge_pairs(candidates, 1)[paper_id]
    assert len(result) == 1
    assert result[0].query_chunk.text == "first"
    assert result[0].reranking_score is None
