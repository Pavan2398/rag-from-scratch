"""Unit tests for embeddings generation, normalization, and persistence.

All tests use mocked SentenceTransformer models to avoid downloading real weights.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from app.embeddings.embedder import (
    build_embedding_metadata,
    encode_query,
    generate_embeddings,
    load_embeddings,
    normalize_vectors,
    run_embedding_pipeline,
    save_embeddings,
)


def test_normalize_vectors_2d():
    """Verify that 2D matrix rows are normalized to unit Euclidean length (L2 norm = 1)."""
    # Create arbitrary non-normalized vectors
    vecs = np.array(
        [
            [3.0, 4.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    normed = normalize_vectors(vecs)

    # Check shape is preserved
    assert normed.shape == vecs.shape

    # Check each row's norm is exactly 1.0
    row_norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(row_norms, [1.0, 1.0, 1.0], rtol=1e-5)

    # For unit vectors, dot product with self must equal 1.0
    for i in range(3):
        assert np.isclose(np.dot(normed[i], normed[i]), 1.0, atol=1e-5)


def test_normalize_vectors_zero_vector():
    """Verify that an all-zero vector is handled safely without division by zero."""
    zero_vec = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normed = normalize_vectors(zero_vec)
    assert normed.shape == (1, 3)
    np.testing.assert_allclose(normed, [[0.0, 0.0, 0.0]])


def test_normalize_vectors_1d():
    """Verify 1D vector normalization."""
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = normalize_vectors(vec)
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-5)
    np.testing.assert_allclose(normed, [0.6, 0.8], atol=1e-5)


def test_generate_embeddings_with_mock():
    """Verify generate_embeddings calls model.encode with batching and normalizes output."""
    mock_model = MagicMock()
    embedding_dim = 384
    num_texts = 5
    dummy_output = np.random.randn(num_texts, embedding_dim).astype(np.float32)

    mock_model.encode.return_value = dummy_output

    texts = [f"Sample text chunk {i}" for i in range(num_texts)]
    embeddings = generate_embeddings(
        texts=texts,
        model=mock_model,
        batch_size=2,
        normalize=True,
        show_progress_bar=False,
    )

    mock_model.encode.assert_called_once_with(
        texts,
        batch_size=2,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    assert embeddings.shape == (num_texts, embedding_dim)
    row_norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(num_texts), rtol=1e-5)


def test_generate_embeddings_empty_texts():
    """Verify empty text list returns an empty array with shape (0, 384)."""
    mock_model = MagicMock()
    embeddings = generate_embeddings([], model=mock_model)
    assert embeddings.shape == (0, 384)
    mock_model.encode.assert_not_called()


def test_encode_query_with_mock():
    """Verify query encoding produces a 1D unit vector."""
    mock_model = MagicMock()
    dummy_vec = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    mock_model.encode.return_value = dummy_vec

    q_vec = encode_query("What is the capital?", model=mock_model, normalize=True)
    assert q_vec.shape == (3,)
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-5)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        encode_query("   ", model=mock_model)


def test_build_embedding_metadata():
    """Verify embedding metadata links chunk provenance and maintains exact index sequence."""
    chunks = [
        {
            "chunk_id": "doc_0001_c00",
            "document_id": "doc_0001",
            "title": "Alpha Title",
            "character_start": 0,
            "character_end": 450,
            "sentence_ids": [0, 1],
        },
        {
            "chunk_id": "doc_0001_c01",
            "document_id": "doc_0001",
            "title": "Alpha Title",
            "character_start": 400,
            "character_end": 800,
            "sentence_ids": [1, 2],
        },
    ]

    meta = build_embedding_metadata(chunks)
    assert len(meta) == 2

    assert meta[0]["embedding_index"] == 0
    assert meta[0]["chunk_id"] == "doc_0001_c00"
    assert meta[0]["document_id"] == "doc_0001"
    assert meta[0]["title"] == "Alpha Title"
    assert meta[0]["character_start"] == 0
    assert meta[0]["character_end"] == 450
    assert meta[0]["sentence_ids"] == [0, 1]

    assert meta[1]["embedding_index"] == 1
    assert meta[1]["chunk_id"] == "doc_0001_c01"


def test_save_and_load_embeddings(tmp_path: Path):
    """Verify embeddings and metadata persist and reload with complete numerical fidelity."""
    matrix_file = tmp_path / "embeddings.npy"
    metadata_file = tmp_path / "embedding_metadata.json"

    original_matrix = np.random.randn(10, 384).astype(np.float32)
    original_meta = [
        {"embedding_index": i, "chunk_id": f"c_{i}", "document_id": f"d_{i}", "title": f"T_{i}"}
        for i in range(10)
    ]

    save_embeddings(original_matrix, original_meta, matrix_file, metadata_file)
    assert matrix_file.exists()
    assert metadata_file.exists()

    loaded_matrix, loaded_meta = load_embeddings(matrix_file, metadata_file)
    np.testing.assert_array_equal(original_matrix, loaded_matrix)
    assert original_meta == loaded_meta


def test_run_embedding_pipeline_skips_when_cached(tmp_path: Path):
    """Verify that existing embeddings are reused and not recomputed unless force=True."""
    config_file = tmp_path / "test_config.yaml"
    chunks_file = tmp_path / "chunks.json"
    matrix_file = tmp_path / "embeddings.npy"
    metadata_file = tmp_path / "embedding_metadata.json"

    # Save mock existing data
    existing_matrix = np.ones((5, 384), dtype=np.float32)
    existing_meta = [{"embedding_index": i} for i in range(5)]
    np.save(matrix_file, existing_matrix)
    import json
    with open(metadata_file, "w") as f:
        json.dump(existing_meta, f)
    with open(chunks_file, "w") as f:
        json.dump([{"text": "dummy"} for _ in range(5)], f)

    test_config = {
        "embeddings": {"model_name": "mock-model", "batch_size": 16, "normalize": True},
        "paths": {
            "chunks_file": str(chunks_file),
            "embeddings_matrix_file": str(matrix_file),
            "embeddings_metadata_file": str(metadata_file),
        },
    }
    import yaml
    with open(config_file, "w") as f:
        yaml.dump(test_config, f)

    with patch("app.embeddings.embedder.load_embedder_model") as mock_load:
        result = run_embedding_pipeline(config_path=str(config_file), force=False)
        assert result["status"] == "cached"
        assert result["number_of_chunks"] == 5
        mock_load.assert_not_called()
