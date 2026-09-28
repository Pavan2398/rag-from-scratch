"""Main CLI entry point for RAG From Scratch — Phase 1: Document Retrieval Engine.

Usage:
  python main.py --help
  python -m app.data.download
  python -m app.data.preprocess
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG From Scratch — Phase 1: Document Retrieval Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download reproducible HotpotQA subset")
    download_parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess HotpotQA into corpus documents")
    preprocess_parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    preprocess_parser.add_argument("--force", action="store_true", help="Force re-processing")

    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Chunk processed documents into overlapping text chunks")
    chunk_parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    chunk_parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk_size")
    chunk_parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk_overlap")
    chunk_parser.add_argument("--force", action="store_true", help="Force re-chunking")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate dense vector embeddings for document chunks")
    embed_parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    embed_parser.add_argument("--batch-size", type=int, default=None, help="Batch size for encoding")
    embed_parser.add_argument("--force", action="store_true", help="Force re-generating embeddings")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve top-K chunks for a query")
    retrieve_parser.add_argument("--query", type=str, default=None, help="Search query string")
    retrieve_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return (default: 5)")
    retrieve_parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    if args.command == "download":
        from app.data.download import download_hotpotqa_subset
        download_hotpotqa_subset(config_path=args.config, force=args.force)
    elif args.command == "preprocess":
        from app.data.preprocess import preprocess_hotpotqa
        preprocess_hotpotqa(config_path=args.config, force=args.force)
    elif args.command == "chunk":
        from app.chunking.chunker import run_chunking
        run_chunking(
            config_path=args.config,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force,
        )
    elif args.command == "embed":
        from app.embeddings.embedder import run_embedding_pipeline
        run_embedding_pipeline(
            config_path=args.config,
            batch_size=args.batch_size,
            force=args.force,
        )
    elif args.command == "retrieve":
        from app.cli.retrieve import format_retrieval_output, run_interactive_mode
        from app.retrieval.retriever import DenseRetriever

        retriever = DenseRetriever.from_config(config_path=args.config, load_model=True)
        if args.query:
            results = retriever.retrieve(query=args.query, top_k=args.top_k)
            print(format_retrieval_output(args.query, results))
        else:
            run_interactive_mode(retriever=retriever, default_top_k=args.top_k)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
