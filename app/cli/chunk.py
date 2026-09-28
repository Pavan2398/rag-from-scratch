"""CLI command to run document chunking.

Usage:
  python -m app.cli.chunk
  python -m app.cli.chunk --chunk-size 500 --chunk-overlap 50 --force
"""

import argparse
import logging
import sys

from app.chunking.chunker import run_chunking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk processed documents into overlapping text chunks."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk_size in characters (default from config: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk_overlap in characters (default from config: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-chunking even if chunks.json already exists",
    )

    args = parser.parse_args()

    try:
        output_file = run_chunking(
            config_path=args.config,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force,
        )
        print(f"\nChunking finished successfully! Chunks saved to: {output_file}")
    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
