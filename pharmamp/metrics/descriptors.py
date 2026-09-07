"""Shared physicochemical descriptor helpers, built on the `peptides` package.

These wrap standard, published descriptor scales (Kyte-Doolittle hydrophobicity,
Eisenberg hydrophobic moment, Henderson-Hasselbalch net charge) rather than
reimplementing them, so the numbers trace back to a maintained, citable source
instead of hand-copied tables.
"""

from __future__ import annotations

import peptides
from Bio.Data.IUPACData import protein_letters

AMINO_ACIDS = protein_letters  # canonical 20-letter alphabet, single source of truth


def check_non_empty(sequences: list[str]) -> None:
    """Raise a clear error before an empty sequence reaches a descriptor
    calculation, instead of a library-internal ZeroDivisionError or a
    silently propagated NaN."""
    if any(not seq for seq in sequences):
        raise ValueError("Cannot score an empty sequence.")


def net_charge(sequence: str, ph: float = 7.4) -> float:
    return peptides.Peptide(sequence).charge(pH=ph)


def mean_hydrophobicity(sequence: str, scale: str = "KyteDoolittle") -> float:
    values = peptides.Peptide(sequence).hydrophobicity(scale=scale)
    return sum(values) / len(values) if isinstance(values, (list, tuple)) else float(values)


def hydrophobic_moment(sequence: str, angle: int = 100) -> float:
    return peptides.Peptide(sequence).hydrophobic_moment(window=min(len(sequence), 11), angle=angle)


def max_window_hydrophobicity(sequence: str, window: int = 5, scale: str = "KyteDoolittle") -> float:
    """Peak hydrophobicity over sliding windows — the hydrophobic-patch signal used by
    amyloid/aggregation-propensity schemes (e.g. AGGRESCAN, TANGO) to flag aggregation-prone
    stretches, rather than relying on the whole-sequence average."""
    profile = peptides.Peptide(sequence).hydrophobicity_profile(window=window, scale=scale)
    if len(profile) == 0:  # sequence shorter than the window
        return mean_hydrophobicity(sequence, scale=scale)
    return max(profile)
