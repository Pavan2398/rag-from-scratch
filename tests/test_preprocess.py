"""Unit tests for HotpotQA preprocessing and sentence character boundary tracking."""

import pytest
from app.data.preprocess import build_document_record


def test_build_document_record_sentence_offsets():
    """Verify that sentence character offsets are calculated accurately."""
    title = "Test Article"
    sentences = [
        "First sentence of article.",
        "Second sentence is slightly longer and descriptive.",
        "Third sentence concludes the paragraph.",
    ]
    doc = build_document_record(
        doc_index=1,
        title=title,
        sentences_list=sentences,
        question_id="q_test_123",
    )

    assert doc["document_id"] == "doc_0001"
    assert doc["title"] == title
    assert doc["source_question_ids"] == ["q_test_123"]
    assert doc["source_dataset"] == "hotpotqa"

    full_text = doc["text"]
    assert full_text == "First sentence of article. Second sentence is slightly longer and descriptive. Third sentence concludes the paragraph."

    # Validate each sentence's character boundaries
    assert len(doc["sentences"]) == len(sentences)
    for sent_meta, original_sent in zip(doc["sentences"], sentences):
        start = sent_meta["character_start"]
        end = sent_meta["character_end"]
        # Slice into full document text must match original sentence exactly
        assert full_text[start:end] == original_sent
        assert sent_meta["text"] == original_sent


def test_build_document_record_empty_sentences():
    """Verify handling when an article has empty or single sentence."""
    title = "Single Sentence Article"
    sentences = ["Only one sentence."]
    doc = build_document_record(
        doc_index=42,
        title=title,
        sentences_list=sentences,
        question_id="q_test_single",
    )
    assert doc["document_id"] == "doc_0042"
    assert doc["text"] == "Only one sentence."
    assert len(doc["sentences"]) == 1
    assert doc["text"][doc["sentences"][0]["character_start"]:doc["sentences"][0]["character_end"]] == "Only one sentence."
