"""Integration test for the full benchmark harness (generation + scoring).

Skipped unless the OmegAMP checkout/checkpoint/embeddings are present — see
tests/test_benchmark_generators.py for why.
"""

from __future__ import annotations

import pytest

from pharmamp.benchmark.generators import DEFAULT_CHECKPOINT, DEFAULT_EMBEDDINGS, OMEGAMP_ROOT

BENCHMARK_READY = OMEGAMP_ROOT.exists() and DEFAULT_CHECKPOINT.exists() and DEFAULT_EMBEDDINGS.exists()

pytestmark = pytest.mark.skipif(
    not BENCHMARK_READY,
    reason="OmegAMP checkout/checkpoint/embeddings not present locally",
)


def test_run_benchmark_covers_omegamp_and_both_baselines():
    from pharmamp.benchmark.run_benchmark import run

    results = run(num_samples=4, batch_size=4, seed=0)

    assert set(results.index) == {"OmegAMP", "Random", "Shuffled-natural"}
    expected_metrics = {
        "Count",
        "Uniqueness",
        "Novelty",
        "Length",
        "Diversity",
        "AggregationPropensity",
        "Solubility",
    }
    assert expected_metrics <= set(results.columns.get_level_values(0))
    assert (results.loc[:, ("Count", "value")] == 4).all()
