"""Unit tests for dataset download configuration and sampling logic."""

import pytest
from app.data.download import load_config, sample_with_reservoir, validate_example


def test_load_config():
    """Verify that config/config.yaml loads properly with expected keys."""
    config = load_config("config/config.yaml")
    assert "dataset" in config
    assert "chunking" in config
    assert "embeddings" in config
    assert "retrieval" in config
    assert "paths" in config

    assert config["dataset"]["subset_size"] == 500
    assert config["dataset"]["random_seed"] == 42
    assert config["chunking"]["chunk_size"] == 500
    assert config["chunking"]["chunk_overlap"] == 50


def test_sample_with_reservoir_reproducibility():
    """Verify that reservoir sampling is deterministic for a given seed."""
    stream_data = [{"id": f"item_{i}", "val": i} for i in range(1000)]
    k = 50
    seed = 42

    sample1 = sample_with_reservoir(stream_data, k=k, seed=seed)
    sample2 = sample_with_reservoir(stream_data, k=k, seed=seed)

    assert len(sample1) == k
    assert len(sample2) == k
    assert sample1 == sample2

    # Different seed should give different sample
    sample3 = sample_with_reservoir(stream_data, k=k, seed=999)
    assert sample1 != sample3


def test_validate_example():
    """Verify that validate_example raises error on missing fields."""
    valid_example = {
        "id": "123",
        "question": "Q?",
        "answer": "A",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {},
        "context": {},
    }
    # Should not raise
    validate_example(valid_example, 0)

    # Missing field
    invalid_example = dict(valid_example)
    del invalid_example["supporting_facts"]
    with pytest.raises(ValueError, match="Missing required field 'supporting_facts'"):
        validate_example(invalid_example, 0)
