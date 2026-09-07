"""Non-lab baseline generators for the benchmark harness.

Neither of these is a trained model — they're null models, standard practice
in generative-sequence benchmarking: a real generator should clearly beat
them, or it isn't adding anything a distribution-matched coin flip couldn't.
Both share `OmegAMPGenerator`'s `.generate(num_samples, batch_size, seed)`
interface so `run_benchmark.py` can call all generators the same way.
"""

from __future__ import annotations

import random

from seqme.utils.sequences import shuffle_characters

from pharmamp.metrics.descriptors import AMINO_ACIDS


class RandomAminoAcidGenerator:
    """Uniform-random amino acids, length-matched to a reference sequence set.

    The weakest possible baseline: no composition or motif structure at all.
    """

    name = "Random"

    def __init__(self, reference_lengths: list[int]):
        if not reference_lengths:
            raise ValueError("reference_lengths must be non-empty")
        self.reference_lengths = reference_lengths

    def generate(self, num_samples: int = 32, batch_size: int | None = None, seed: int | None = None) -> list[str]:
        rng = random.Random(seed)
        return [
            "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.choice(self.reference_lengths)))
            for _ in range(num_samples)
        ]


class ShuffledReferenceGenerator:
    """Real reference AMPs with residues shuffled in place.

    Same length and amino-acid composition as a real peptide — unlike the
    random baseline — but no motif or positional structure. The stronger,
    more informative of the two null models.
    """

    name = "Shuffled-natural"

    def __init__(self, reference_sequences: list[str]):
        if not reference_sequences:
            raise ValueError("reference_sequences must be non-empty")
        self.reference_sequences = reference_sequences

    def generate(self, num_samples: int = 32, batch_size: int | None = None, seed: int | None = None) -> list[str]:
        rng = random.Random(seed)
        chosen = (
            rng.sample(self.reference_sequences, num_samples)
            if num_samples <= len(self.reference_sequences)
            else rng.choices(self.reference_sequences, k=num_samples)
        )
        return shuffle_characters(chosen, seed=seed)
