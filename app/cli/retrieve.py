"""CLI command to run dense vector retrieval against document chunks.

Usage:
  python -m app.cli.retrieve --query "Who directed the movie?" --top-k 5
  python -m app.cli.retrieve (interactive mode)
"""

import argparse
import logging
import sys
import time

from app.retrieval.retriever import DenseRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def format_retrieval_output(query: str, results: list) -> str:
    """Format retrieval results cleanly for console display."""
    lines = [
        "",
        "=" * 70,
        f" QUERY: {query}",
        f" RETRIEVED TOP {len(results)} CHUNKS",
        "=" * 70,
    ]
    for res in results:
        lines.append(f"\n[Rank {res['rank']}] Similarity Score: {res['score']:.4f}")
        lines.append(f"Title:     {res['title']}")
        lines.append(f"Chunk ID:  {res['chunk_id']} (Doc ID: {res['document_id']})")
        lines.append(f"Offsets:   Chars {res['character_start']}..{res['character_end']} | Sentences: {res['sentence_ids']}")
        lines.append("Text:")
        lines.append(f"  {res['text']}")
        lines.append("-" * 70)
    return "\n".join(lines)


def run_interactive_mode(retriever: DenseRetriever, default_top_k: int) -> None:
    """Run interactive REPL loop for ad-hoc user queries."""
    print("\n" + "=" * 70)
    print(" INTERACTIVE RETRIEVAL SHELL (Type 'exit' or 'quit' to stop) ")
    print("=" * 70)
    while True:
        try:
            query = input("\nEnter query: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting interactive retrieval.")
                break

            start = time.time()
            results = retriever.retrieve(query=query, top_k=default_top_k)
            latency = (time.time() - start) * 1000

            print(format_retrieval_output(query, results))
            print(f"Retrieval latency: {latency:.2f} ms")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive retrieval.")
            break
        except Exception as e:
            print(f"Error during retrieval: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search document chunks using dense vector cosine similarity."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query string. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    try:
        retriever = DenseRetriever.from_config(config_path=args.config, load_model=True)

        if args.query:
            start = time.time()
            results = retriever.retrieve(query=args.query, top_k=args.top_k)
            latency = (time.time() - start) * 1000

            print(format_retrieval_output(args.query, results))
            print(f"Retrieval latency: {latency:.2f} ms\n")
        else:
            run_interactive_mode(retriever=retriever, default_top_k=args.top_k)

    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
