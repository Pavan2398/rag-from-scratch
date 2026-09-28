"""Document Chunking Module for RAG From Scratch — Phase 1.

Implements character-based sliding-window chunking over the processed documents corpus.
Preserves chunk IDs, parent document IDs, Wikipedia titles, exact character boundary
offsets (character_start, character_end), and sentence ID mappings for supporting-fact
evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Tuple[int, int, str]]:
    """Split a string into overlapping character chunks using a sliding window.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of tuples: (character_start, character_end, chunk_text)
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    text_len = len(text)
    if text_len == 0:
        return []

    # If document text is smaller than or equal to chunk_size, return a single chunk
    if text_len <= chunk_size:
        return [(0, text_len, text)]

    stride = chunk_size - chunk_overlap
    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append((start, end, text[start:end]))
        # If this chunk reached or passed the end of the text, stop
        if end >= text_len:
            break
        start += stride

    return chunks


def find_overlapping_sentence_ids(
    chunk_start: int,
    chunk_end: int,
    sentences: List[Dict[str, Any]],
) -> List[int]:
    """Identify all sentence IDs from the document that overlap with this chunk.

    Two half-open intervals [A_start, A_end) and [B_start, B_end) overlap iff:
        max(A_start, B_start) < min(A_end, B_end)

    Args:
        chunk_start: Start character index of the chunk.
        chunk_end: End character index of the chunk.
        sentences: List of sentence metadata dictionaries with character_start,
                   character_end, and sent_id.

    Returns:
        List of integer sentence IDs overlapping this chunk.
    """
    overlapping_ids: List[int] = []
    for s in sentences:
        s_start = s.get("character_start", 0)
        s_end = s.get("character_end", 0)
        # Check interval overlap
        if max(chunk_start, s_start) < min(chunk_end, s_end):
            overlapping_ids.append(int(s.get("sent_id", 0)))
    return overlapping_ids


def chunk_document(
    doc: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Chunk a single document record from data/processed/documents.json.

    Args:
        doc: Document dictionary containing document_id, title, text, sentences.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of chunk dictionaries with character boundaries and metadata.
    """
    doc_id = doc.get("document_id", "doc_unknown")
    title = doc.get("title", "")
    text = doc.get("text", "")
    sentences = doc.get("sentences", [])
    source_dataset = doc.get("source_dataset", "hotpotqa")

    text_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks: List[Dict[str, Any]] = []

    for idx, (start, end, chunk_str) in enumerate(text_chunks):
        chunk_id = f"{doc_id}_c{idx:02d}"
        overlapping_sentences = find_overlapping_sentence_ids(start, end, sentences)

        chunk_record = {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "title": title,
            "text": chunk_str,
            "character_start": start,
            "character_end": end,
            "sentence_ids": overlapping_sentences,
            "metadata": {
                "title": title,
                "document_id": doc_id,
                "character_start": start,
                "character_end": end,
                "sentence_ids": overlapping_sentences,
                "source_dataset": source_dataset,
            },
        }
        doc_chunks.append(chunk_record)

    return doc_chunks


def chunk_corpus(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Chunk an entire corpus of documents.

    Args:
        documents: List of document dictionaries.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        Aggregated list of all chunk dictionaries across all documents.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks


def run_chunking(
    config_path: str = "config/config.yaml",
    chunk_size: int = None,
    chunk_overlap: int = None,
    force: bool = False,
) -> Path:
    """Load processed documents, perform chunking, and persist chunks to disk.

    Args:
        config_path: Path to configuration YAML.
        chunk_size: Optional override for chunk_size.
        chunk_overlap: Optional override for chunk_overlap.
        force: If True, overwrite existing chunks file.

    Returns:
        Path to the saved chunks JSON file.
    """
    config = load_config(config_path)
    chunking_cfg = config.get("chunking", {})
    paths_cfg = config.get("paths", {})

    size = chunk_size if chunk_size is not None else chunking_cfg.get("chunk_size", 500)
    overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else chunking_cfg.get("chunk_overlap", 50)
    )

    documents_file = Path(paths_cfg.get("documents_file", "data/processed/documents.json"))
    chunks_file = Path(paths_cfg.get("chunks_file", "data/chunks/chunks.json"))

    if not documents_file.exists():
        raise FileNotFoundError(
            f"Processed documents file not found at: {documents_file}. "
            "Please run preprocessing first: python -m app.data.preprocess"
        )

    if chunks_file.exists() and not force:
        logger.info(
            f"Chunks file already exists at {chunks_file}. Use force=True to re-chunk."
        )
        return chunks_file

    logger.info(f"Loading documents from {documents_file}...")
    with open(documents_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info(
        f"Chunking {len(documents)} documents (chunk_size={size}, chunk_overlap={overlap})..."
    )
    chunks = chunk_corpus(documents, chunk_size=size, chunk_overlap=overlap)

    # Ensure output directory exists
    chunks_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents.")
    logger.info(f"Saving chunks to {chunks_file}...")
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Chunking completed successfully.")
    return chunks_file
