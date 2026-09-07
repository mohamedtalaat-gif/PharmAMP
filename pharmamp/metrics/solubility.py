"""Aqueous-solubility metric.

Combines net charge, mean hydrophobicity, and amphipathicity (hydrophobic
moment) into a single solubility score, following the descriptor set used by
sequence-based solubility predictors such as CamSol (Sormanni et al., 2015):
charged, weakly hydrophobic, low-amphipathicity sequences are favoured.

Like `AggregationPropensityMetric`, this is a first-pass linear heuristic, not
a fitted model.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from seqme.core.base import Metric, MetricResult

from pharmamp.metrics.descriptors import (
    check_non_empty,
    hydrophobic_moment,
    mean_hydrophobicity,
    net_charge,
)


class SolubilityMetric(Metric):
    """Sequence-based aqueous-solubility score (higher is better)."""

    def __init__(
        self,
        charge_weight: float = 5.0,
        moment_weight: float = 0.5,
        scale: str = "KyteDoolittle",
        *,
        name: str = "Solubility",
    ):
        """
        Args:
            charge_weight: How strongly net charge contributes to solubility, per unit
                of charge-per-residue.
            moment_weight: How strongly amphipathicity (hydrophobic moment) penalises
                solubility.
            scale: Hydrophobicity scale passed to `peptides.Peptide.hydrophobicity`.
            name: Metric name.
        """
        self.charge_weight = charge_weight
        self.moment_weight = moment_weight
        self.scale = scale
        self._name = name

    def score(self, sequences: list[str]) -> list[float]:
        """Per-sequence solubility scores."""
        check_non_empty(sequences)
        scores = []
        for seq in sequences:
            charge_term = self.charge_weight * abs(net_charge(seq)) / len(seq)
            hydrophobicity_term = mean_hydrophobicity(seq, scale=self.scale)
            moment_term = self.moment_weight * hydrophobic_moment(seq)
            scores.append(charge_term - hydrophobicity_term - moment_term)
        return scores

    def __call__(self, sequences: list[str]) -> MetricResult:
        scores = self.score(sequences)
        return MetricResult(value=float(np.mean(scores)), deviation=float(np.std(scores)))

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
