"""Unit tests for document chunking module."""

import pytest
from app.chunking.chunker import (
    chunk_corpus,
    chunk_document,
    chunk_text,
    find_overlapping_sentence_ids,
)


def test_chunk_text_empty():
    """Verify that an empty string produces an empty list of chunks."""
    assert chunk_text("", chunk_size=500, chunk_overlap=50) == []


def test_chunk_text_shorter_than_chunk_size():
    """Verify that text shorter than chunk_size produces exactly one chunk."""
    text = "Short text example."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    start, end, chunk_str = chunks[0]
    assert start == 0
    assert end == len(text)
    assert chunk_str == text


def test_chunk_text_exact_chunk_size():
    """Verify that text exactly matching chunk_size produces exactly one chunk."""
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == (0, 500, text)


def test_chunk_text_sliding_window_and_overlap():
    """Verify sliding window character offsets and overlap when text exceeds chunk_size."""
    # Length 600 with chunk_size=500 and chunk_overlap=50
    # Chunk 1: [0, 500]
    # Stride = 500 - 50 = 450
    # Chunk 2: [450, 600]
    text = "".join(f"[{i:04d}]" for i in range(100))  # 600 characters
    assert len(text) == 600

    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2

    c1_start, c1_end, c1_text = chunks[0]
    c2_start, c2_end, c2_text = chunks[1]

    assert c1_start == 0
    assert c1_end == 500
    assert c1_text == text[0:500]

    assert c2_start == 450
    assert c2_end == 600
    assert c2_text == text[450:600]

    # Verify overlap region is identical
    overlap_len = c1_end - c2_start
    assert overlap_len == 50
    assert text[c2_start:c1_end] == c1_text[450:500] == c2_text[0:50]


def test_chunk_text_invalid_parameters():
    """Verify validation of chunk_size and chunk_overlap parameters."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("Hello", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_text("Hello", chunk_size=100, chunk_overlap=-10)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_text("Hello", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_text("Hello", chunk_size=100, chunk_overlap=150)


def test_find_overlapping_sentence_ids():
    """Verify that sentence intervals overlapping the chunk interval are correctly detected."""
    sentences = [
        {"sent_id": 0, "character_start": 0, "character_end": 100},
        {"sent_id": 1, "character_start": 101, "character_end": 250},
        {"sent_id": 2, "character_start": 251, "character_end": 450},
        {"sent_id": 3, "character_start": 451, "character_end": 600},
    ]

    # Chunk covers 0 to 200 -> overlaps sent_id 0 and 1
    assert find_overlapping_sentence_ids(0, 200, sentences) == [0, 1]

    # Chunk covers 200 to 500 -> overlaps sent_id 1, 2, and 3
    assert find_overlapping_sentence_ids(200, 500, sentences) == [1, 2, 3]

    # Chunk covers 500 to 600 -> overlaps only sent_id 3
    assert find_overlapping_sentence_ids(500, 600, sentences) == [3]


def test_chunk_document_preserves_metadata_and_boundaries():
    """Verify that chunk_document preserves all required schema fields and accurate slices."""
    sentences = [
        "First sentence of the document.",
        "Second sentence is slightly longer and contains more details.",
        "Third sentence finishes the first block of text.",
        "Fourth sentence begins a new block of text that extends the document further.",
    ]
    # Replicate preprocess sentence boundary calculation
    sent_meta = []
    curr = 0
    for idx, s in enumerate(sentences):
        sent_meta.append({
            "sent_id": idx,
            "text": s,
            "character_start": curr,
            "character_end": curr + len(s),
        })
        curr += len(s) + 1
    full_text = " ".join(sentences)

    doc = {
        "document_id": "doc_0042",
        "title": "Quantum Computing",
        "text": full_text,
        "sentences": sent_meta,
        "source_question_ids": ["q_1001"],
        "source_dataset": "hotpotqa",
    }

    # Use small chunk_size to force multiple chunks
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1

    for idx, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"doc_0042_c{idx:02d}"
        assert chunk["document_id"] == "doc_0042"
        assert chunk["title"] == "Quantum Computing"
        assert chunk["text"] == full_text[chunk["character_start"]:chunk["character_end"]]
        assert "sentence_ids" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["title"] == "Quantum Computing"
        assert chunk["metadata"]["document_id"] == "doc_0042"


def test_chunk_corpus():
    """Verify chunk_corpus processes a list of documents and aggregates all chunks."""
    docs = [
        {
            "document_id": "doc_0001",
            "title": "Doc 1",
            "text": "This is doc 1 text.",
            "sentences": [{"sent_id": 0, "text": "This is doc 1 text.", "character_start": 0, "character_end": 19}],
            "source_dataset": "hotpotqa",
        },
        {
            "document_id": "doc_0002",
            "title": "Doc 2",
            "text": "This is doc 2 text.",
            "sentences": [{"sent_id": 0, "text": "This is doc 2 text.", "character_start": 0, "character_end": 19}],
            "source_dataset": "hotpotqa",
        },
    ]

    all_chunks = chunk_corpus(docs, chunk_size=500, chunk_overlap=50)
    assert len(all_chunks) == 2
    assert all_chunks[0]["document_id"] == "doc_0001"
    assert all_chunks[1]["document_id"] == "doc_0002"
