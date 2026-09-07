"""Aggregation-propensity metric.

Scores how likely a peptide is to self-associate into amyloid-like aggregates,
following the general logic behind sequence-based aggregation predictors
(AGGRESCAN, TANGO, Zyggregator): a hydrophobic patch drives aggregation, and net
charge suppresses it by keeping the peptide soluble (Chiti & Dobson, 2006).

This is a first-pass linear heuristic over descriptors from the `peptides`
package, not a fitted model — see the README for calibration plans against
public aggregation datasets (WALTZ-DB, CPAD).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from seqme.core.base import Metric, MetricResult

from pharmamp.metrics.descriptors import check_non_empty, max_window_hydrophobicity, net_charge


class AggregationPropensityMetric(Metric):
    """Sequence-based aggregation-propensity score (lower is better)."""

    def __init__(
        self,
        window: int = 5,
        charge_weight: float = 5.0,
        scale: str = "KyteDoolittle",
        *,
        name: str = "AggregationPropensity",
    ):
        """
        Args:
            window: Width of the hydrophobic-patch sliding window.
            charge_weight: How strongly net charge suppresses the score, per unit of
                charge-per-residue.
            scale: Hydrophobicity scale passed to `peptides.Peptide.hydrophobicity`.
            name: Metric name.
        """
        self.window = window
        self.charge_weight = charge_weight
        self.scale = scale
        self._name = name

    def score(self, sequences: list[str]) -> list[float]:
        """Per-sequence aggregation-propensity scores."""
        check_non_empty(sequences)
        scores = []
        for seq in sequences:
            patch = max_window_hydrophobicity(seq, window=self.window, scale=self.scale)
            charge_penalty = self.charge_weight * abs(net_charge(seq)) / len(seq)
            scores.append(patch - charge_penalty)
        return scores

    def __call__(self, sequences: list[str]) -> MetricResult:
        scores = self.score(sequences)
        return MetricResult(value=float(np.mean(scores)), deviation=float(np.std(scores)))

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"
