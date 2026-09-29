"""CLI command to run retrieval evaluation against HotpotQA supporting facts.

Usage:
  python -m app.cli.evaluate
  python -m app.cli.evaluate --top-k 1 3 5 10 --limit 500
"""

import argparse
import logging
import sys

from app.evaluation.evaluator import format_evaluation_summary, run_evaluation_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval using HotpotQA supporting-fact ground truth."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=None,
        help="List of top-K values to evaluate (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (default: data/evaluation/retrieval_results.json)",
    )

    args = parser.parse_args()

    try:
        results = run_evaluation_pipeline(
            config_path=args.config,
            top_k_list=args.top_k,
            limit=args.limit,
            output_path=args.output,
        )

        # Print human-readable summary
        print(format_evaluation_summary(results))

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
