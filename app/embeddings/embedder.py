"""Embeddings Generation and Persistence Module.

Generates dense vector representations for document chunks using Sentence Transformers.
Embeddings are L2-normalized so that cosine similarity can be computed via dot product:
    cos(u, v) = (u . v) / (||u|| * ||v||) = u . v   (when ||u|| = ||v|| = 1)

Embeddings are saved to disk as a NumPy array (embeddings.npy) alongside
a metadata mapping file (embedding_metadata.json) linking each embedding
row index back to its chunk_id, document_id, and Wikipedia title.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization of an embedding matrix.

    Divides each vector by its Euclidean length (norm).
    For unit-length vectors, dot product is identical to cosine similarity:
        cosine_similarity(a, b) = np.dot(a, b)

    Args:
        vectors: NumPy array of shape (N, D).

    Returns:
        NumPy array of shape (N, D) where each row has unit norm (length = 1.0).
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against division by zero for any edge-case all-zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def load_embedder_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Load a SentenceTransformer model by name.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        SentenceTransformer model instance.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.error(
            "sentence-transformers is not installed. Please install via: pip install sentence-transformers"
        )
        raise e

    logger.info(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(
    texts: List[str],
    model: Any,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode a list of texts into dense vectors in batches.

    Args:
        texts: List of text strings to embed.
        model: SentenceTransformer model instance (or mock supporting .encode).
        batch_size: Number of texts to process in each forward pass.
        normalize: If True, L2-normalize the resulting vectors.
        show_progress_bar: Whether to display a progress indicator.

    Returns:
        NumPy float32 matrix of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    logger.info(
        f"Encoding {len(texts)} texts in batches of {batch_size} (normalize={normalize})..."
    )

    # Use model's batch encoding
    raw_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    if normalize:
        embeddings = normalize_vectors(embeddings)

    return embeddings


def encode_query(
    query: str,
    model: Any,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a single query string into a 1D unit vector.

    Args:
        query: User search query.
        model: SentenceTransformer model instance.
        normalize: If True, normalize to unit length.

    Returns:
        1D NumPy array of shape (embedding_dim,).
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")

    raw = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    vec = np.asarray(raw[0], dtype=np.float32)
    if normalize:
        vec = normalize_vectors(vec)
    return vec


def build_embedding_metadata(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build metadata records linking each embedding row index to its chunk provenance.

    Args:
        chunks: List of chunk dictionaries loaded from data/chunks/chunks.json.

    Returns:
        List of metadata records containing embedding_index, chunk_id, document_id, title,
        character_start, character_end, and sentence_ids.
    """
    metadata_list: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        record = {
            "embedding_index": idx,
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": chunk.get("document_id", ""),
            "title": chunk.get("title", ""),
            "character_start": chunk.get("character_start", 0),
            "character_end": chunk.get("character_end", 0),
            "sentence_ids": chunk.get("sentence_ids", []),
        }
        metadata_list.append(record)
    return metadata_list


def save_embeddings(
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    matrix_path: Path,
    metadata_path: Path,
) -> None:
    """Save embeddings matrix and metadata records to disk.

    Args:
        embeddings: NumPy array of shape (N, D).
        metadata: List of N metadata records.
        matrix_path: Target .npy file path.
        metadata_path: Target .json file path.
    """
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving embeddings matrix of shape {embeddings.shape} to: {matrix_path}")
    np.save(matrix_path, embeddings)

    logger.info(f"Saving {len(metadata)} embedding metadata records to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_embeddings(
    matrix_path: Path,
    metadata_path: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load previously saved embeddings and metadata from disk.

    Args:
        matrix_path: Path to the .npy embeddings matrix.
        metadata_path: Path to the .json metadata mapping.

    Returns:
        Tuple of (embeddings_matrix, metadata_list).
    """
    if not matrix_path.exists():
        raise FileNotFoundError(f"Embeddings matrix not found at: {matrix_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Embeddings metadata not found at: {metadata_path}")

    embeddings = np.load(matrix_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embeddings, metadata


def run_embedding_pipeline(
    config_path: str = "config/config.yaml",
    batch_size: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the complete chunk embedding and persistence pipeline.

    Args:
        config_path: Path to config.yaml.
        batch_size: Optional override for batch encoding size.
        force: If True, re-compute even if embeddings already exist.

    Returns:
        Dictionary containing summary metrics and file paths.
    """
    config = load_config(config_path)
    emb_cfg = config.get("embeddings", {})
    paths_cfg = config.get("paths", {})

    model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    configured_batch_size = (
        batch_size if batch_size is not None else emb_cfg.get("batch_size", 32)
    )
    normalize = emb_cfg.get("normalize", True)

    chunks_file = Path(paths_cfg.get("chunks_file", "data/chunks/chunks.json"))
    matrix_file = Path(paths_cfg.get("embeddings_matrix_file", "data/embeddings/embeddings.npy"))
    metadata_file = Path(
        paths_cfg.get("embeddings_metadata_file", "data/embeddings/embedding_metadata.json")
    )

    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found at: {chunks_file}. "
            "Please run chunking first: python -m app.cli.chunk"
        )

    # Check if embeddings already exist
    if matrix_file.exists() and metadata_file.exists() and not force:
        logger.info(
            f"Embeddings already exist at {matrix_file} and {metadata_file}. "
            "Skipping computation (use --force to recompute)."
        )
        existing_matrix = np.load(matrix_file, mmap_mode="r")
        with open(metadata_file, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        return {
            "status": "cached",
            "number_of_chunks": len(existing_meta),
            "model_name": model_name,
            "embedding_dimension": existing_matrix.shape[1],
            "batch_size": configured_batch_size,
            "matrix_path": str(matrix_file),
            "metadata_path": str(metadata_file),
            "embedding_shape": existing_matrix.shape,
            "elapsed_seconds": 0.0,
        }

    logger.info(f"Loading chunks from: {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    num_chunks = len(chunks)
    logger.info(f"Loaded {num_chunks} chunks.")

    # Extract text from each chunk while preserving exact sequence order
    texts = [chunk["text"] for chunk in chunks]

    # Load model and compute embeddings
    model = load_embedder_model(model_name)

    start_time = time.time()
    embeddings = generate_embeddings(
        texts=texts,
        model=model,
        batch_size=configured_batch_size,
        normalize=normalize,
        show_progress_bar=True,
    )
    elapsed_time = time.time() - start_time

    # Build metadata linking rows to chunks
    metadata = build_embedding_metadata(chunks)

    # Save to disk
    save_embeddings(embeddings, metadata, matrix_file, metadata_file)

    result = {
        "status": "computed",
        "number_of_chunks": num_chunks,
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_size": configured_batch_size,
        "matrix_path": str(matrix_file),
        "metadata_path": str(metadata_file),
        "embedding_shape": embeddings.shape,
        "elapsed_seconds": round(elapsed_time, 2),
    }

    logger.info(
        f"Embedding completed in {elapsed_time:.2f}s | Shape: {embeddings.shape} | Dim: {embeddings.shape[1]}"
    )
    return result
