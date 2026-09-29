"""Retrieval Evaluation Module for RAG From Scratch — Phase 1.

Evaluates dense retrieval against HotpotQA supporting-fact ground truth.
Maps sentence-level supporting facts to document chunks using exact character
offset intersections:
    max(chunk_start, sentence_start) < min(chunk_end, sentence_end)

Computes:
- Recall@K: Proportion of questions where AT LEAST ONE gold chunk appears in top-K.
- Evidence Recall@K: Proportion of questions where ALL gold chunks appear in top-K.
- Average retrieval latency per question (in milliseconds).
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
import yaml

from app.retrieval.retriever import DenseRetriever

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_supporting_facts_to_chunks(
    supporting_facts: List[Dict[str, Any]],
    documents_by_title: Dict[str, Dict[str, Any]],
    chunks_by_title: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Map sentence-level supporting facts to covering chunk IDs.

    For each supporting fact (title, sent_id):
    1. Looks up the document by title to retrieve sentence character offsets.
    2. Finds all chunks whose character boundaries overlap with the sentence:
       max(chunk_start, sent_start) < min(chunk_end, sent_end)

    Args:
        supporting_facts: List of dicts with 'title' and 'sent_id'.
        documents_by_title: Mapping from Wikipedia title to document dict.
        chunks_by_title: Mapping from Wikipedia title to list of chunk dicts.

    Returns:
        Sorted list of unique chunk IDs covering the supporting facts.
    """
    gold_chunk_ids: Set[str] = set()

    for sf in supporting_facts:
        title = sf.get("title", "")
        sent_id = sf.get("sent_id", None)

        if sent_id is None:
            continue

        doc = documents_by_title.get(title)
        if not doc:
            logger.debug(f"Supporting fact document '{title}' not found in documents.")
            continue

        # Find matching sentence by sent_id
        target_sent = None
        for s in doc.get("sentences", []):
            if s.get("sent_id") == sent_id:
                target_sent = s
                break

        if not target_sent:
            logger.debug(
                f"Sentence ID {sent_id} not found in document '{title}'."
            )
            continue

        sent_start = target_sent.get("character_start", 0)
        sent_end = target_sent.get("character_end", 0)

        # Find chunks overlapping this sentence
        doc_chunks = chunks_by_title.get(title, [])
        for chunk in doc_chunks:
            c_start = chunk.get("character_start", 0)
            c_end = chunk.get("character_end", 0)

            # Half-open interval intersection test
            if max(c_start, sent_start) < min(c_end, sent_end):
                gold_chunk_ids.add(chunk["chunk_id"])

    return sorted(list(gold_chunk_ids))


def calculate_recall_metrics(
    eval_records: List[Dict[str, Any]],
    top_k_list: List[int],
) -> Dict[str, float]:
    """Calculate Recall@K and Evidence Recall@K across all evaluation records.

    Args:
        eval_records: List of per-question evaluation dictionaries.
        top_k_list: List of K thresholds (e.g. [1, 3, 5, 10]).

    Returns:
        Dictionary mapping metric names (e.g. 'Recall@1', 'Evidence Recall@1')
        to percentage values (0.0 to 1.0).
    """
    total_questions = len(eval_records)
    metrics: Dict[str, float] = {}

    if total_questions == 0:
        for k in top_k_list:
            metrics[f"Recall@{k}"] = 0.0
            metrics[f"Evidence Recall@{k}"] = 0.0
        return metrics

    for k in top_k_list:
        k_str = str(k)
        any_hits = sum(1 for r in eval_records if r["hit"].get(k_str, False))
        all_hits = sum(
            1 for r in eval_records if r["all_evidence_hit"].get(k_str, False)
        )

        metrics[f"Recall@{k}"] = round(any_hits / total_questions, 4)
        metrics[f"Evidence Recall@{k}"] = round(all_hits / total_questions, 4)

    return metrics


def evaluate_question(
    question_record: Dict[str, Any],
    retriever: DenseRetriever,
    documents_by_title: Dict[str, Dict[str, Any]],
    chunks_by_title: Dict[str, List[Dict[str, Any]]],
    top_k_list: List[int],
) -> Dict[str, Any]:
    """Evaluate retrieval for a single question record.

    Args:
        question_record: Dict with 'question_id', 'question', 'supporting_facts'.
        retriever: DenseRetriever instance.
        documents_by_title: Title -> document mapping.
        chunks_by_title: Title -> chunks list mapping.
        top_k_list: List of integer top-K ranks to evaluate.

    Returns:
        Structured evaluation dictionary for this question.
    """
    q_id = question_record.get("question_id", "")
    query_text = question_record.get("question", "")
    supporting_facts = question_record.get("supporting_facts", [])

    # Step 1: Map supporting facts to ground-truth chunk IDs
    gold_chunk_ids = map_supporting_facts_to_chunks(
        supporting_facts=supporting_facts,
        documents_by_title=documents_by_title,
        chunks_by_title=chunks_by_title,
    )
    gold_set = set(gold_chunk_ids)

    # Step 2: Retrieve chunks up to max(top_k_list)
    max_k = max(top_k_list) if top_k_list else 10
    start_time = time.perf_counter()
    retrieved_results = retriever.retrieve(query=query_text, top_k=max_k)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    retrieved_chunk_ids_full = [res["chunk_id"] for res in retrieved_results]

    hit_map: Dict[str, bool] = {}
    all_evidence_map: Dict[str, bool] = {}
    retrieved_by_k: Dict[str, List[str]] = {}

    for k in top_k_list:
        k_str = str(k)
        retrieved_k = retrieved_chunk_ids_full[:k]
        retrieved_set_k = set(retrieved_k)
        retrieved_by_k[f"top_{k}"] = retrieved_k

        # Hit@K: at least one gold chunk is present in top K
        hit_map[k_str] = bool(gold_set & retrieved_set_k) if gold_set else False

        # All-Evidence Hit@K: ALL gold chunks are present in top K
        all_evidence_map[k_str] = (
            gold_set.issubset(retrieved_set_k) if gold_set else False
        )

    return {
        "question_id": q_id,
        "question": query_text,
        "ground_truth_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": retrieved_by_k,
        "hit": hit_map,
        "all_evidence_hit": all_evidence_map,
        "latency_ms": round(latency_ms, 2),
    }


def evaluate_retrieval(
    questions: List[Dict[str, Any]],
    retriever: DenseRetriever,
    documents: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    top_k_list: Optional[List[int]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run retrieval evaluation across a corpus of evaluation questions.

    Args:
        questions: List of question records loaded from questions.json.
        retriever: DenseRetriever instance.
        documents: List of document records from documents.json.
        chunks: List of chunk records from chunks.json.
        top_k_list: K values for evaluation (default: [1, 3, 5, 10]).
        limit: Optional maximum number of questions to evaluate.

    Returns:
        Complete evaluation result dictionary with summary and per-question records.
    """
    if top_k_list is None:
        top_k_list = [1, 3, 5, 10]

    # Pre-index documents and chunks by title for O(1) lookups
    documents_by_title = {d["title"]: d for d in documents if "title" in d}
    chunks_by_title: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        title = c.get("title", "")
        chunks_by_title.setdefault(title, []).append(c)

    eval_subset = questions[:limit] if limit is not None else questions
    total_eval = len(eval_subset)

    logger.info(
        f"Starting retrieval evaluation on {total_eval} questions with top_k={top_k_list}..."
    )

    per_question_results: List[Dict[str, Any]] = []
    total_eval_start = time.time()

    for q in tqdm(eval_subset, desc="Evaluating Questions", unit="q"):
        res = evaluate_question(
            question_record=q,
            retriever=retriever,
            documents_by_title=documents_by_title,
            chunks_by_title=chunks_by_title,
            top_k_list=top_k_list,
        )
        per_question_results.append(res)

    total_eval_time = time.time() - total_eval_start

    # Compute metrics
    metrics = calculate_recall_metrics(per_question_results, top_k_list)
    avg_latency = (
        sum(r["latency_ms"] for r in per_question_results) / total_eval
        if total_eval > 0
        else 0.0
    )

    summary = {
        "num_questions": total_eval,
        "top_k_values": top_k_list,
        "metrics": metrics,
        "average_latency_ms": round(avg_latency, 2),
        "total_evaluation_time_seconds": round(total_eval_time, 2),
    }

    return {
        "summary": summary,
        "per_question_results": per_question_results,
    }


def format_evaluation_summary(results: Dict[str, Any]) -> str:
    """Format evaluation metrics into the requested human-readable summary string."""
    summary = results.get("summary", {})
    num_questions = summary.get("num_questions", 0)
    metrics = summary.get("metrics", {})
    avg_latency = summary.get("average_latency_ms", 0.0)
    top_k_values = summary.get("top_k_values", [1, 3, 5, 10])

    lines = [
        "",
        "=" * 60,
        " RETRIEVAL EVALUATION REPORT ",
        "=" * 60,
        f"Evaluation examples: {num_questions}",
        "",
    ]

    for k in top_k_values:
        val = metrics.get(f"Recall@{k}", 0.0) * 100.0
        lines.append(f"Recall@{k:<2}:         {val:6.2f}%")

    lines.append("")
    for k in top_k_values:
        val = metrics.get(f"Evidence Recall@{k}", 0.0) * 100.0
        lines.append(f"Evidence Recall@{k:<2}: {val:6.2f}%")

    lines.append("")
    lines.append(f"Average retrieval latency: {avg_latency:.2f} ms")
    lines.append(f"Total evaluation time:     {summary.get('total_evaluation_time_seconds', 0.0):.2f} s")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_evaluation_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save full evaluation results to disk as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving evaluation results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_evaluation_pipeline(
    config_path: str = "config/config.yaml",
    top_k_list: Optional[List[int]] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete retrieval evaluation pipeline from config and saved data.

    Args:
        config_path: Path to config.yaml.
        top_k_list: Optional override list of top-K values.
        limit: Optional limit on number of questions to evaluate.
        output_path: Optional custom output path for retrieval_results.json.

    Returns:
        Evaluation results dictionary.
    """
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    eval_cfg = config.get("evaluation", {})

    k_values = (
        top_k_list
        if top_k_list is not None
        else eval_cfg.get("top_k_values", [1, 3, 5, 10])
    )

    out_file = Path(
        output_path
        if output_path is not None
        else paths_cfg.get(
            "evaluation_results_file", "data/evaluation/retrieval_results.json"
        )
    )

    questions_path = Path(paths_cfg.get("questions_file", "data/processed/questions.json"))
    documents_path = Path(paths_cfg.get("documents_file", "data/processed/documents.json"))
    chunks_path = Path(paths_cfg.get("chunks_file", "data/chunks/chunks.json"))

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    logger.info("Loading questions, documents, and chunks...")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Initializing DenseRetriever...")
    retriever = DenseRetriever.from_config(config_path=config_path, load_model=True)

    results = evaluate_retrieval(
        questions=questions,
        retriever=retriever,
        documents=documents,
        chunks=chunks,
        top_k_list=k_values,
        limit=limit,
    )

    save_evaluation_results(results, out_file)
    return results
