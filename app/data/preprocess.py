"""Preprocessing module for HotpotQA.

Transforms the raw HotpotQA subset into:
1. data/processed/documents.json:
   - A deduplicated document corpus indexed by Wikipedia title.
   - Preserves exact sentence boundaries with character_start and character_end offsets.
   - Pure document text (no metadata prepended/injected into the text).
2. data/processed/questions.json:
   - Evaluation questions containing question text, answer, difficulty,
     and gold supporting fact annotations (title + sent_id).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_document_record(
    doc_index: int,
    title: str,
    sentences_list: List[str],
    question_id: str,
) -> Dict[str, Any]:
    """Build a document record with sentence-level character boundary tracking.
    
    Sentences are joined with a single space (' ').
    Offsets are calculated such that document_text[start:end] exactly equals the sentence.
    """
    joined_text_parts: List[str] = []
    sentences_meta: List[Dict[str, Any]] = []
    current_offset = 0

    for sent_id, sentence_str in enumerate(sentences_list):
        sent_len = len(sentence_str)
        start_char = current_offset
        end_char = start_char + sent_len

        sentences_meta.append({
            "sent_id": sent_id,
            "text": sentence_str,
            "character_start": start_char,
            "character_end": end_char,
        })
        joined_text_parts.append(sentence_str)
        # Advance offset by sentence length + 1 character for the joining space
        current_offset = end_char + 1

    document_text = " ".join(joined_text_parts)

    return {
        "document_id": f"doc_{doc_index:04d}",
        "title": title,
        "text": document_text,
        "sentences": sentences_meta,
        "source_question_ids": [question_id],
        "source_dataset": "hotpotqa",
    }


def preprocess_hotpotqa(
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> Dict[str, Path]:
    """Process the raw HotpotQA subset into documents and evaluation questions.

    Args:
        config_path: Path to configuration YAML file.
        force: If True, re-process and overwrite existing processed files.

    Returns:
        Dictionary with paths to 'documents_file' and 'questions_file'.
    """
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})

    raw_file = Path(paths_cfg.get("raw_subset_file", "data/raw/hotpotqa_subset_500.json"))
    documents_file = Path(paths_cfg.get("documents_file", "data/processed/documents.json"))
    questions_file = Path(paths_cfg.get("questions_file", "data/processed/questions.json"))

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw HotpotQA subset not found at {raw_file}. "
            f"Run the dataset download script first: python -m app.data.download"
        )

    if documents_file.exists() and questions_file.exists() and not force:
        logger.info(
            f"Processed data already exists at {documents_file} and {questions_file}. "
            f"Use force=True to re-process."
        )
        return {"documents_file": documents_file, "questions_file": questions_file}

    # Ensure output directories exist
    documents_file.parent.mkdir(parents=True, exist_ok=True)
    questions_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw HotpotQA subset from {raw_file}...")
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_examples = json.load(f)

    logger.info(f"Loaded {len(raw_examples)} raw examples. Extracting documents and questions...")

    # Title -> Document Record mapping for deduplication
    documents_by_title: Dict[str, Dict[str, Any]] = {}
    questions_list: List[Dict[str, Any]] = []

    doc_counter = 0

    for example in raw_examples:
        q_id = example["id"]
        question_text = example["question"]
        answer_text = example.get("answer", "")
        q_type = example.get("type", "unknown")
        q_level = example.get("level", "unknown")

        # Normalize supporting facts: {"title": [...], "sent_id": [...]} or list of [title, sent_id]
        raw_sf = example.get("supporting_facts", {})
        parsed_supporting_facts: List[Dict[str, Any]] = []
        gold_titles: List[str] = []

        if isinstance(raw_sf, dict):
            titles = raw_sf.get("title", [])
            sent_ids = raw_sf.get("sent_id", [])
            for t, s in zip(titles, sent_ids):
                parsed_supporting_facts.append({"title": t, "sent_id": int(s)})
                if t not in gold_titles:
                    gold_titles.append(t)
        elif isinstance(raw_sf, list):
            for item in raw_sf:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    t, s = item[0], item[1]
                    parsed_supporting_facts.append({"title": t, "sent_id": int(s)})
                    if t not in gold_titles:
                        gold_titles.append(t)
                elif isinstance(item, dict):
                    t, s = item.get("title"), item.get("sent_id")
                    if t is not None and s is not None:
                        parsed_supporting_facts.append({"title": t, "sent_id": int(s)})
                        if t not in gold_titles:
                            gold_titles.append(t)

        questions_list.append({
            "question_id": q_id,
            "question": question_text,
            "answer": answer_text,
            "type": q_type,
            "level": q_level,
            "supporting_facts": parsed_supporting_facts,
            "gold_titles": gold_titles,
        })

        # Process context passages into deduplicated documents
        context = example.get("context", {})
        context_titles: List[str] = []
        context_sentences: List[List[str]] = []

        if isinstance(context, dict):
            context_titles = context.get("title", [])
            context_sentences = context.get("sentences", [])
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    context_titles.append(item[0])
                    context_sentences.append(item[1])

        for title, sentences_list in zip(context_titles, context_sentences):
            if title in documents_by_title:
                # Document already seen in a prior question: append question ID for provenance
                if q_id not in documents_by_title[title]["source_question_ids"]:
                    documents_by_title[title]["source_question_ids"].append(q_id)
                # If the new sentence list is longer or more complete, update the sentences
                if len(sentences_list) > len(documents_by_title[title]["sentences"]):
                    doc_id = documents_by_title[title]["document_id"]
                    source_qids = documents_by_title[title]["source_question_ids"]
                    updated_record = build_document_record(
                        int(doc_id.split("_")[1]),
                        title,
                        sentences_list,
                        q_id,
                    )
                    updated_record["source_question_ids"] = source_qids
                    documents_by_title[title] = updated_record
            else:
                doc_counter += 1
                doc_record = build_document_record(
                    doc_counter,
                    title,
                    sentences_list,
                    q_id,
                )
                documents_by_title[title] = doc_record

    documents_list = list(documents_by_title.values())

    logger.info(f"Extracted {len(documents_list)} unique Wikipedia documents from {len(raw_examples)} questions.")
    logger.info(f"Saving documents to {documents_file}...")
    with open(documents_file, "w", encoding="utf-8") as f:
        json.dump(documents_list, f, indent=2, ensure_ascii=False)

    logger.info(f"Saving questions to {questions_file}...")
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions_list, f, indent=2, ensure_ascii=False)

    logger.info("Preprocessing complete.")
    return {"documents_file": documents_file, "questions_file": questions_file}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw HotpotQA subset into corpus documents and questions."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if output files already exist.",
    )
    args = parser.parse_args()

    preprocess_hotpotqa(config_path=args.config, force=args.force)


if __name__ == "__main__":
    main()
