"""Fast unit tests for run_benchmark.py's pure-Python logic — no checkpoint
or embeddings needed, only the small git-tracked reference CSV inside the
vendored third_party/OmegAMP checkout. Kept separate from
test_run_benchmark.py so a fresh clone with just `git clone
.../OmegAMP.git third_party/OmegAMP` (no multi-GB downloads) still
exercises this logic in CI.
"""

from __future__ import annotations

import pytest

from pharmamp.benchmark.generators import OMEGAMP_ROOT

pytestmark = pytest.mark.skipif(
    not OMEGAMP_ROOT.exists(),
    reason="third_party/OmegAMP checkout not present locally",
)


def test_load_reference_amps_returns_only_amp_sequences():
    from pharmamp.benchmark.run_benchmark import load_reference_amps

    reference = load_reference_amps()
    assert len(reference) > 0
    assert all(isinstance(seq, str) and seq for seq in reference[:100])


def test_build_metrics_includes_all_expected_metrics():
    from pharmamp.benchmark.run_benchmark import build_metrics

    metrics = build_metrics(["ACDEFGHIK", "KLMNPQRST"])
    names = {metric.name for metric in metrics}
    assert names == {
        "Count",
        "Uniqueness",
        "Novelty",
        "Length",
        "Diversity",
        "AggregationPropensity",
        "Solubility",
    }
