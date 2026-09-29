"""Unit tests for retrieval evaluation module.

All tests use synthetic documents, chunks, and mocked retrievers to verify
supporting-fact mapping, boundary overlaps, multiple facts, and recall metrics.
"""

from unittest.mock import MagicMock

import pytest
from app.evaluation.evaluator import (
    calculate_recall_metrics,
    evaluate_question,
    evaluate_retrieval,
    map_supporting_facts_to_chunks,
)


@pytest.fixture
def sample_corpus():
    """Create a minimal synthetic corpus with documents and overlapping chunks."""
    documents_by_title = {
        "Article Alpha": {
            "document_id": "doc_01",
            "title": "Article Alpha",
            "text": "Alpha sentence zero. Alpha sentence one.",
            "sentences": [
                {"sent_id": 0, "text": "Alpha sentence zero.", "character_start": 0, "character_end": 20},
                {"sent_id": 1, "text": "Alpha sentence one.", "character_start": 21, "character_end": 40},
            ],
        },
        "Article Beta": {
            "document_id": "doc_02",
            "title": "Article Beta",
            "text": "Beta sentence zero covers boundaries across multiple chunks.",
            "sentences": [
                {"sent_id": 0, "text": "Beta sentence zero covers boundaries across multiple chunks.", "character_start": 0, "character_end": 60},
            ],
        },
    }

    chunks_by_title = {
        "Article Alpha": [
            {"chunk_id": "chunk_a0", "title": "Article Alpha", "character_start": 0, "character_end": 20},
            {"chunk_id": "chunk_a1", "title": "Article Alpha", "character_start": 15, "character_end": 40},
        ],
        "Article Beta": [
            # Two chunks overlapping the same single sentence [0, 60]
            {"chunk_id": "chunk_b0", "title": "Article Beta", "character_start": 0, "character_end": 35},
            {"chunk_id": "chunk_b1", "title": "Article Beta", "character_start": 30, "character_end": 60},
        ],
    }

    return documents_by_title, chunks_by_title


def test_supporting_sentence_to_chunk_mapping(sample_corpus):
    """Verify single supporting fact maps to the correct chunk."""
    docs, chunks = sample_corpus
    sf = [{"title": "Article Alpha", "sent_id": 0}]

    gold_chunks = map_supporting_facts_to_chunks(sf, docs, chunks)
    # Sentence 0 is [0, 20].
    # chunk_a0 is [0, 20] -> overlap (0, 20)
    # chunk_a1 is [15, 40] -> max(15, 0) < min(40, 20) -> 15 < 20 -> overlap!
    assert "chunk_a0" in gold_chunks
    assert "chunk_a1" in gold_chunks


def test_sentence_overlapping_chunk_boundaries(sample_corpus):
    """Verify a sentence spanning a chunk boundary is mapped to both chunks."""
    docs, chunks = sample_corpus
    # Beta sentence 0 is [0, 60]
    sf = [{"title": "Article Beta", "sent_id": 0}]

    gold_chunks = map_supporting_facts_to_chunks(sf, docs, chunks)
    assert gold_chunks == ["chunk_b0", "chunk_b1"]


def test_multiple_supporting_facts(sample_corpus):
    """Verify multiple supporting facts from different documents map to union of chunks."""
    docs, chunks = sample_corpus
    sf = [
        {"title": "Article Alpha", "sent_id": 1},  # [21, 40] -> chunk_a1 only (chunk_a0 ends at 20)
        {"title": "Article Beta", "sent_id": 0},   # [0, 60] -> chunk_b0, chunk_b1
    ]

    gold_chunks = map_supporting_facts_to_chunks(sf, docs, chunks)
    assert "chunk_a1" in gold_chunks
    assert "chunk_a0" not in gold_chunks
    assert "chunk_b0" in gold_chunks
    assert "chunk_b1" in gold_chunks


def test_missing_supporting_chunk_handling(sample_corpus):
    """Verify missing document or invalid sentence ID does not crash and returns empty list."""
    docs, chunks = sample_corpus
    sf = [
        {"title": "Non-Existent Article", "sent_id": 0},
        {"title": "Article Alpha", "sent_id": 999},  # Invalid sentence ID
    ]

    gold_chunks = map_supporting_facts_to_chunks(sf, docs, chunks)
    assert gold_chunks == []


def test_recall_at_k_calculation():
    """Verify Recall@K metric accurately counts proportion of any-hit questions."""
    top_k_list = [1, 3, 5]

    eval_records = [
        # Q1: Hit at 1, 3, 5
        {"hit": {"1": True, "3": True, "5": True}, "all_evidence_hit": {"1": True, "3": True, "5": True}},
        # Q2: Hit at 3, 5 (Miss at 1)
        {"hit": {"1": False, "3": True, "5": True}, "all_evidence_hit": {"1": False, "3": False, "5": True}},
        # Q3: Hit at 5 (Miss at 1, 3)
        {"hit": {"1": False, "3": False, "5": True}, "all_evidence_hit": {"1": False, "3": False, "5": False}},
        # Q4: Miss at all K
        {"hit": {"1": False, "3": False, "5": False}, "all_evidence_hit": {"1": False, "3": False, "5": False}},
    ]

    metrics = calculate_recall_metrics(eval_records, top_k_list)

    assert metrics["Recall@1"] == 0.25   # 1 / 4
    assert metrics["Recall@3"] == 0.50   # 2 / 4
    assert metrics["Recall@5"] == 0.75   # 3 / 4


def test_all_evidence_recall_at_k_calculation():
    """Verify Evidence Recall@K requires ALL gold chunks to be present in top-K."""
    top_k_list = [1, 3]

    eval_records = [
        # Q1: All evidence present at 1 and 3
        {"hit": {"1": True, "3": True}, "all_evidence_hit": {"1": True, "3": True}},
        # Q2: Partial evidence present at 1 and 3 (any-hit=True, all-hit=False)
        {"hit": {"1": True, "3": True}, "all_evidence_hit": {"1": False, "3": False}},
        # Q3: No evidence present
        {"hit": {"1": False, "3": False}, "all_evidence_hit": {"1": False, "3": False}},
    ]

    metrics = calculate_recall_metrics(eval_records, top_k_list)

    assert metrics["Recall@1"] == round(2 / 3, 4)
    assert metrics["Evidence Recall@1"] == round(1 / 3, 4)
    assert metrics["Recall@3"] == round(2 / 3, 4)
    assert metrics["Evidence Recall@3"] == round(1 / 3, 4)


def test_evaluate_question_with_mock_retriever(sample_corpus):
    """Verify evaluate_question builds hit and all-evidence maps correctly."""
    docs, chunks = sample_corpus

    mock_retriever = MagicMock()
    # Mock retrieval returns chunk_a1 first, then chunk_b0
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "chunk_a1"},
        {"chunk_id": "chunk_b0"},
    ]

    # Question where gold chunks are chunk_a1 and chunk_b0
    q_record = {
        "question_id": "q001",
        "question": "Sample Question?",
        "supporting_facts": [
            {"title": "Article Alpha", "sent_id": 1},  # maps to chunk_a1
            {"title": "Article Beta", "sent_id": 0},   # maps to chunk_b0, chunk_b1
        ],
    }

    res = evaluate_question(
        question_record=q_record,
        retriever=mock_retriever,
        documents_by_title=docs,
        chunks_by_title=chunks,
        top_k_list=[1, 2],
    )

    # Top 1 retrieved: ['chunk_a1']
    # Gold: ['chunk_a1', 'chunk_b0', 'chunk_b1']
    assert res["hit"]["1"] is True          # chunk_a1 is in gold
    assert res["all_evidence_hit"]["1"] is False  # chunk_b0 and chunk_b1 are missing

    # Top 2 retrieved: ['chunk_a1', 'chunk_b0']
    assert res["hit"]["2"] is True
    # chunk_b1 is still missing, so all_evidence is False
    assert res["all_evidence_hit"]["2"] is False


def test_top_k_larger_than_corpus(sample_corpus):
    """Verify evaluation runs smoothly when top_k is larger than available chunks."""
    docs, chunks = sample_corpus

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "chunk_a0"},
        {"chunk_id": "chunk_a1"},
    ]

    q_record = {
        "question_id": "q002",
        "question": "Where is it?",
        "supporting_facts": [{"title": "Article Alpha", "sent_id": 0}],
    }

    res = evaluate_question(
        question_record=q_record,
        retriever=mock_retriever,
        documents_by_title=docs,
        chunks_by_title=chunks,
        top_k_list=[10, 50],
    )

    assert res["hit"]["10"] is True
    assert res["hit"]["50"] is True


def test_empty_evaluation_input():
    """Verify evaluate_retrieval handles empty question list gracefully."""
    mock_retriever = MagicMock()
    results = evaluate_retrieval(
        questions=[],
        retriever=mock_retriever,
        documents=[],
        chunks=[],
        top_k_list=[1, 5],
    )

    assert results["summary"]["num_questions"] == 0
    assert results["summary"]["metrics"]["Recall@1"] == 0.0
    assert results["summary"]["metrics"]["Recall@5"] == 0.0
    assert len(results["per_question_results"]) == 0
