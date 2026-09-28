"""Unit tests for dense vector retrieval module.

All tests use synthetic vectors and mocked models to avoid loading external weights.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from app.retrieval.retriever import DenseRetriever


@pytest.fixture
def sample_corpus_data():
    """Create a small synthetic corpus with 3 normalized 3D embeddings."""
    # Unit vectors along orthogonal and intermediate axes
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],                          # Index 0: chunk_001
            [0.0, 1.0, 0.0],                          # Index 1: chunk_002
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0], # Index 2: chunk_003
        ],
        dtype=np.float32,
    )

    metadata = [
        {"embedding_index": 0, "chunk_id": "c01", "document_id": "d01", "title": "First Article"},
        {"embedding_index": 1, "chunk_id": "c02", "document_id": "d02", "title": "Second Article"},
        {"embedding_index": 2, "chunk_id": "c03", "document_id": "d03", "title": "Third Article"},
    ]

    chunks = [
        {"chunk_id": "c01", "document_id": "d01", "title": "First Article", "text": "First chunk text about apples."},
        {"chunk_id": "c02", "document_id": "d02", "title": "Second Article", "text": "Second chunk text about bananas."},
        {"chunk_id": "c03", "document_id": "d03", "title": "Third Article", "text": "Third chunk text about fruit salads."},
    ]

    return embeddings, metadata, chunks


def test_dot_product_scoring_and_descending_ranking(sample_corpus_data):
    """Verify dot product scoring and strict descending order ranking."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)

    # Query vector pointing directly along X-axis: [1.0, 0.0, 0.0]
    # Expected dot products:
    # c01: 1.0 * 1.0 = 1.0
    # c03: 1.0 * (1/sqrt(2)) = ~0.7071
    # c02: 1.0 * 0.0 = 0.0
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.retrieve_by_vector(query_vec, top_k=3)

    assert len(results) == 3

    # Rank 1: c01
    assert results[0]["rank"] == 1
    assert results[0]["chunk_id"] == "c01"
    assert np.isclose(results[0]["score"], 1.0, atol=1e-4)

    # Rank 2: c03
    assert results[1]["rank"] == 2
    assert results[1]["chunk_id"] == "c03"
    assert np.isclose(results[1]["score"], 0.7071, atol=1e-4)

    # Rank 3: c02
    assert results[2]["rank"] == 3
    assert results[2]["chunk_id"] == "c02"
    assert np.isclose(results[2]["score"], 0.0, atol=1e-4)


def test_top_k_selection(sample_corpus_data):
    """Verify top_k parameter restricts the number of returned chunks."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)

    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = retriever.retrieve_by_vector(query_vec, top_k=2)

    assert len(results) == 2
    assert [r["chunk_id"] for r in results] == ["c01", "c03"]


def test_top_k_larger_than_corpus(sample_corpus_data):
    """Verify top_k larger than total corpus size returns all chunks without error."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)

    query_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = retriever.retrieve_by_vector(query_vec, top_k=100)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "c02"  # Exact match for [0, 1, 0]


def test_metadata_and_chunk_text_mapping(sample_corpus_data):
    """Verify retrieved chunks correctly fetch full text and provenance from chunks.json."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)

    query_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = retriever.retrieve_by_vector(query_vec, top_k=1)

    top = results[0]
    assert top["chunk_id"] == "c02"
    assert top["document_id"] == "d02"
    assert top["title"] == "Second Article"
    assert top["text"] == "Second chunk text about bananas."
    assert "score" in top
    assert "rank" in top


def test_invalid_top_k(sample_corpus_data):
    """Verify non-positive top_k raises ValueError."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve_by_vector(query_vec, top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve_by_vector(query_vec, top_k=-5)


def test_embedding_dimension_mismatch(sample_corpus_data):
    """Verify passing a query vector of mismatched dimension raises ValueError."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)

    # 4D vector when corpus is 3D
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="does not match corpus embedding dimension"):
        retriever.retrieve_by_vector(query_vec, top_k=2)


def test_validation_embeddings_vs_metadata_length():
    """Verify error when embedding rows do not match metadata record count."""
    embeddings = np.ones((4, 3), dtype=np.float32)
    metadata = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]  # Only 2 records
    chunks = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]

    with pytest.raises(ValueError, match="Embedding rows .* does not match metadata count"):
        DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)


def test_validation_missing_chunk_id():
    """Verify error when metadata references a chunk_id not present in chunks."""
    embeddings = np.ones((2, 3), dtype=np.float32)
    metadata = [{"chunk_id": "c1"}, {"chunk_id": "non_existent_chunk"}]
    chunks = [{"chunk_id": "c1"}]

    with pytest.raises(ValueError, match="metadata chunk IDs missing from chunks.json"):
        DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)


def test_retrieve_with_mocked_model(sample_corpus_data):
    """Verify end-to-end retrieve() method with a mocked SentenceTransformer model."""
    embeddings, metadata, chunks = sample_corpus_data
    mock_model = MagicMock()
    # Mock model encoding returns [1, 0, 0]
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    retriever = DenseRetriever(
        embeddings=embeddings, metadata=metadata, chunks=chunks, model=mock_model
    )

    results = retriever.retrieve("Any test query", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c01"
    assert results[0]["score"] == 1.0


def test_empty_query_raises_value_error(sample_corpus_data):
    """Verify empty query string raises ValueError."""
    embeddings, metadata, chunks = sample_corpus_data
    retriever = DenseRetriever(embeddings=embeddings, metadata=metadata, chunks=chunks)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("", top_k=5)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("   ", top_k=5)
