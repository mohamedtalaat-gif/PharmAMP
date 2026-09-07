"""Benchmark harness: generate candidates, score them with `seqme`'s own
metrics and PharmAMP's developability metrics side by side.

Both metric families implement the same `seqme.core.base.Metric` interface,
so `seqme.evaluate` produces one table without any glue code between the two.
"""

from __future__ import annotations

import argparse

import pandas as pd
import seqme as sm
from seqme.metrics.count import Count
from seqme.metrics.diversity import Diversity
from seqme.metrics.length import Length
from seqme.metrics.novelty import Novelty
from seqme.metrics.uniqueness import Uniqueness

from pharmamp import AggregationPropensityMetric, SolubilityMetric
from pharmamp.benchmark.baselines import RandomAminoAcidGenerator, ShuffledReferenceGenerator
from pharmamp.benchmark.generators import OMEGAMP_ROOT, OmegAMPGenerator

REFERENCE_DATASET_CSV = OMEGAMP_ROOT / "data" / "generative-model-data" / "generative-model-dataset.csv"


def load_reference_amps() -> list[str]:
    """Known AMPs from OmegAMP's own training set, used as the novelty reference."""
    df = pd.read_csv(REFERENCE_DATASET_CSV)
    return df.loc[df["IsAMP"] == 1, "Sequence"].tolist()


def build_metrics(reference_amps: list[str]) -> list[sm.Metric]:
    return [
        Count(),
        Uniqueness(),
        Novelty(reference_amps),
        Length(),
        Diversity(),
        AggregationPropensityMetric(),
        SolubilityMetric(),
    ]


def run(num_samples: int = 100, batch_size: int = 32, seed: int | None = 0) -> pd.DataFrame:
    reference_amps = load_reference_amps()

    # Caveat for reading the resulting table: Shuffled-natural draws its
    # sequences from this same reference_amps list, and Novelty checks exact
    # string membership — shuffling a peptide's residues essentially never
    # reproduces the original string, so Shuffled-natural will score close to
    # 100% novel regardless of generation quality. That's expected, not a
    # sign the baseline is meaningfully novel; Uniqueness/Diversity, not
    # Novelty, are the informative columns for this particular baseline.
    generators = [
        OmegAMPGenerator(),
        RandomAminoAcidGenerator(reference_lengths=[len(s) for s in reference_amps]),
        ShuffledReferenceGenerator(reference_sequences=reference_amps),
    ]
    sequences = {
        gen.name: gen.generate(num_samples=num_samples, batch_size=batch_size, seed=seed) for gen in generators
    }

    metrics = build_metrics(reference_amps)
    return sm.evaluate(sequences, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    results = run(num_samples=args.num_samples, batch_size=args.batch_size, seed=args.seed)
    print(results)

    if args.output_csv:
        results.to_csv(args.output_csv)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
