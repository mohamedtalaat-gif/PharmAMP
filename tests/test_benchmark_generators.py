"""Integration test for the OmegAMP generator wrapper.

Skipped unless the pretrained checkpoint/embeddings and the vendored
third_party/OmegAMP checkout are present (see README.md's benchmark install
step) — those are multi-gigabyte downloads that shouldn't be a prerequisite
for the core `pharmamp.metrics` test suite.
"""

from __future__ import annotations

import pytest

from pharmamp.benchmark.generators import DEFAULT_CHECKPOINT, DEFAULT_EMBEDDINGS, OMEGAMP_ROOT
from pharmamp.metrics.descriptors import AMINO_ACIDS

BENCHMARK_READY = OMEGAMP_ROOT.exists() and DEFAULT_CHECKPOINT.exists() and DEFAULT_EMBEDDINGS.exists()

pytestmark = pytest.mark.skipif(
    not BENCHMARK_READY,
    reason="OmegAMP checkout/checkpoint/embeddings not present locally",
)


def test_omegamp_generates_valid_amino_acid_sequences():
    from pharmamp.benchmark import OmegAMPGenerator

    gen = OmegAMPGenerator()
    sequences = gen.generate(num_samples=4, batch_size=4, seed=0)

    assert len(sequences) == 4
    valid_amino_acids = set(AMINO_ACIDS)
    for seq in sequences:
        assert seq, "generated sequence is empty"
        assert set(seq) <= valid_amino_acids, f"unexpected characters in {seq!r}"
