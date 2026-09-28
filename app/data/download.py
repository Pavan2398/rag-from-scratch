"""Download and extract a reproducible subset of the HotpotQA dataset.

This script fetches the HotpotQA dataset (distractor configuration),
shuffles it using a fixed random seed (default: 42) for reproducibility,
selects a subset of approximately 500 examples, and saves the raw data
to data/raw/hotpotqa_subset_500.json.
"""

import argparse
import json
import logging
import os
import random
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


def validate_example(example: Dict[str, Any], index: int) -> None:
    """Ensure that the HotpotQA example contains all expected fields."""
    required_fields = [
        "id",
        "question",
        "answer",
        "type",
        "level",
        "supporting_facts",
        "context",
    ]
    for field in required_fields:
        if field not in example:
            raise ValueError(
                f"Missing required field '{field}' in example at index {index} (ID: {example.get('id', 'unknown')})"
            )


def sample_with_reservoir(
    stream_iterable,
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample k elements from a stream using Algorithm R (Reservoir Sampling).
    
    This ensures uniform probability of selection across an arbitrary length
    stream without requiring the entire dataset to be loaded into memory.
    """
    logger.info(f"Using reservoir sampling to select {k} examples with seed={seed}...")
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []

    for idx, item in enumerate(stream_iterable):
        if idx < k:
            reservoir.append(dict(item))
        else:
            j = rng.randint(0, idx)
            if j < k:
                reservoir[j] = dict(item)

        if (idx + 1) % 5000 == 0:
            logger.info(f"Streamed {idx + 1} items so far...")

    if len(reservoir) < k:
        logger.warning(
            f"Stream ended early: requested {k} items, but only found {len(reservoir)}."
        )
    return reservoir


def download_hotpotqa_subset(
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> Path:
    """Download and extract a reproducible subset of HotpotQA.

    Args:
        config_path: Path to the YAML configuration file.
        force: If True, overwrite existing raw subset file.

    Returns:
        Path to the saved raw subset JSON file.
    """
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})

    dataset_name = dataset_cfg.get("name", "hotpotqa/hotpot_qa")
    config_name = dataset_cfg.get("config_name", "distractor")
    split = dataset_cfg.get("split", "train")
    subset_size = dataset_cfg.get("subset_size", 500)
    random_seed = dataset_cfg.get("random_seed", 42)
    use_streaming = dataset_cfg.get("streaming", False)

    output_path = Path(paths_cfg.get("raw_subset_file", "data/raw/hotpotqa_subset_500.json"))

    if output_path.exists() and not force:
        logger.info(
            f"Raw subset already exists at {output_path}. Use force=True to re-download."
        )
        return output_path

    # Ensure target directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import datasets here to keep module importable even before dependencies are installed
    try:
        from datasets import load_dataset
    except ImportError as e:
        logger.error(
            "The 'datasets' library is required to download HotpotQA. "
            "Please install requirements using: pip install -r requirements.txt"
        )
        raise e

    logger.info(
        f"Acquiring HotpotQA ({dataset_name}, config='{config_name}', split='{split}')..."
    )

    if use_streaming:
        # Reservoir sampling over stream
        dataset_stream = load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=True,
        )
        selected_examples = sample_with_reservoir(
            dataset_stream,
            k=subset_size,
            seed=random_seed,
        )
    else:
        # Standard approach: shuffle full split with seed and select the first subset_size
        logger.info(
            f"Loading full split, shuffling with seed={random_seed}, and selecting {subset_size} examples..."
        )
        dataset = load_dataset(dataset_name, config_name, split=split)
        shuffled = dataset.shuffle(seed=random_seed)
        selected = shuffled.select(range(min(subset_size, len(shuffled))))
        selected_examples = [dict(example) for example in selected]

    logger.info(f"Validating {len(selected_examples)} extracted examples...")
    for idx, ex in enumerate(selected_examples):
        validate_example(ex, idx)

    logger.info(f"Saving raw subset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_examples, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Successfully created reproducible HotpotQA subset ({len(selected_examples)} examples) at {output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract a reproducible subset of HotpotQA."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if subset file already exists.",
    )
    args = parser.parse_args()

    download_hotpotqa_subset(config_path=args.config, force=args.force)


if __name__ == "__main__":
    main()
