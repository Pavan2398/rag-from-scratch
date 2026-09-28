"""CLI command to generate and save chunk embeddings.

Usage:
  python -m app.cli.embed
  python -m app.cli.embed --batch-size 64 --force
"""

import argparse
import logging
import sys
import time

from app.embeddings.embedder import run_embedding_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dense vector embeddings for document chunks."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputing embeddings even if embeddings.npy exists",
    )

    args = parser.parse_args()

    try:
        start = time.time()
        result = run_embedding_pipeline(
            config_path=args.config,
            batch_size=args.batch_size,
            force=args.force,
        )
        total_time = time.time() - start

        print("\n" + "=" * 60)
        print(" EMBEDDING GENERATION SUMMARY ")
        print("=" * 60)
        print(f" Status:               {result['status'].upper()}")
        print(f" Number of Chunks:     {result['number_of_chunks']}")
        print(f" Model Name:           {result['model_name']}")
        print(f" Embedding Dimension:  {result['embedding_dimension']}")
        print(f" Batch Size:           {result['batch_size']}")
        print(f" Matrix Output Path:   {result['matrix_path']}")
        print(f" Metadata Output Path: {result['metadata_path']}")
        print(f" Matrix Shape:         {result['embedding_shape']}")
        print(f" Encoding Time:        {result['elapsed_seconds']}s")
        print(f" Total Pipeline Time:  {total_time:.2f}s")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
