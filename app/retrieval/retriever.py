"""Dense Vector Retrieval Module for RAG From Scratch — Phase 1.

Implements vector similarity retrieval using pure NumPy (no vector databases).
Computes cosine similarity between normalized query embeddings and precomputed
chunk embeddings via dot product:
    scores = embeddings @ query_embedding

Ranks chunks in descending order of similarity and returns top-K results
with chunk provenance and full text fetched from chunks.json.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from app.embeddings.embedder import encode_query, load_embedder_model

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DenseRetriever:
    """NumPy-based dense vector retriever for precomputed document chunk embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        model: Any = None,
    ):
        """Initialize the DenseRetriever with embeddings, metadata, and chunk text.

        Args:
            embeddings: Float32 NumPy array of shape (N, D) containing L2-normalized embeddings.
            metadata: List of N metadata dicts linking row indices to chunk provenance.
            chunks: List of chunk dicts from chunks.json containing full text.
            model: SentenceTransformer model instance (or mock) for query encoding.
        """
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.metadata = metadata
        self.model = model

        # Index chunks by chunk_id for fast O(1) text lookup
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {
            c["chunk_id"]: c for c in chunks if "chunk_id" in c
        }

        # Validate data consistency
        self._validate_integrity()

    def _validate_integrity(self) -> None:
        """Validate alignment between embeddings, metadata, and chunks."""
        num_embeddings = self.embeddings.shape[0]
        num_metadata = len(self.metadata)

        if num_embeddings != num_metadata:
            raise ValueError(
                f"Embedding rows ({num_embeddings}) does not match metadata count ({num_metadata})."
            )

        # Verify all metadata chunk_ids exist in chunks lookup
        missing_ids = [
            m.get("chunk_id")
            for m in self.metadata
            if m.get("chunk_id") not in self.chunk_lookup
        ]
        if missing_ids:
            sample_missing = missing_ids[:5]
            raise ValueError(
                f"Found {len(missing_ids)} metadata chunk IDs missing from chunks.json. "
                f"Sample missing IDs: {sample_missing}"
            )

    @classmethod
    def from_config(
        cls,
        config_path: str = "config/config.yaml",
        load_model: bool = True,
    ) -> "DenseRetriever":
        """Instantiate a DenseRetriever from config paths and saved artifacts.

        Args:
            config_path: Path to config.yaml.
            load_model: Whether to load the SentenceTransformer model immediately.

        Returns:
            Configured DenseRetriever instance.
        """
        config = load_config(config_path)
        paths_cfg = config.get("paths", {})
        emb_cfg = config.get("embeddings", {})

        matrix_path = Path(
            paths_cfg.get("embeddings_matrix_file", "data/embeddings/embeddings.npy")
        )
        metadata_path = Path(
            paths_cfg.get("embeddings_metadata_file", "data/embeddings/embedding_metadata.json")
        )
        chunks_path = Path(paths_cfg.get("chunks_file", "data/chunks/chunks.json"))

        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Embeddings matrix not found at: {matrix_path}. "
                "Please run embedding generation first: python -m app.cli.embed"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Embedding metadata not found at: {metadata_path}. "
                "Please run embedding generation first: python -m app.cli.embed"
            )
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found at: {chunks_path}. "
                "Please run chunking first: python -m app.cli.chunk"
            )

        logger.info(f"Loading embeddings matrix from: {matrix_path}")
        embeddings = np.load(matrix_path)

        logger.info(f"Loading embedding metadata from: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.info(f"Loading chunks text from: {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        model = None
        if load_model:
            model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            model = load_embedder_model(model_name)

        return cls(embeddings=embeddings, metadata=metadata, chunks=chunks, model=model)

    def retrieve_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K chunks using an already-encoded 1D query vector.

        Uses explicit NumPy dot product between pre-normalized chunk embeddings
        and normalized query vector.

        Args:
            query_vector: 1D NumPy array of shape (embedding_dim,).
            top_k: Number of highest-similarity chunks to return.

        Returns:
            List of result dictionaries sorted by descending score.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        query_vec = np.asarray(query_vector, dtype=np.float32).flatten()
        embedding_dim = self.embeddings.shape[1]

        if query_vec.shape[0] != embedding_dim:
            raise ValueError(
                f"Query vector dimension ({query_vec.shape[0]}) does not match "
                f"corpus embedding dimension ({embedding_dim})."
            )

        # Cosine similarity via dot product (both query and corpus embeddings are unit normalized)
        # scores has shape (N,)
        scores = self.embeddings @ query_vec

        total_chunks = len(scores)
        k = min(top_k, total_chunks)

        # Sort indices by descending score: highest similarity first
        sorted_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            meta = self.metadata[idx]
            chunk_id = meta["chunk_id"]
            chunk = self.chunk_lookup[chunk_id]
            score_val = float(scores[idx])

            record = {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id", chunk.get("document_id", "")),
                "title": meta.get("title", chunk.get("title", "")),
                "text": chunk.get("text", ""),
                "score": round(score_val, 4),
                "character_start": meta.get("character_start", chunk.get("character_start", 0)),
                "character_end": meta.get("character_end", chunk.get("character_end", 0)),
                "sentence_ids": meta.get("sentence_ids", chunk.get("sentence_ids", [])),
            }
            results.append(record)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K chunks for a natural language text query.

        Args:
            query: User search query string.
            top_k: Number of highest-similarity chunks to return.

        Returns:
            List of result dictionaries sorted by descending score.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if self.model is None:
            raise RuntimeError(
                "No SentenceTransformer model loaded in DenseRetriever. "
                "Initialize with a model or pass a query vector to retrieve_by_vector()."
            )

        query_vec = encode_query(query=query, model=self.model, normalize=True)
        return self.retrieve_by_vector(query_vector=query_vec, top_k=top_k)
