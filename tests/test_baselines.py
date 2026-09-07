from pharmamp.benchmark.baselines import RandomAminoAcidGenerator, ShuffledReferenceGenerator
from pharmamp.metrics.descriptors import AMINO_ACIDS

VALID_AMINO_ACIDS = set(AMINO_ACIDS)
REFERENCE = ["KWKSFIKKLTSVAKKVLTTALKALS", "GLLAKLLKKLLKKLLKK", "FLPLIGRVLSGIL"]


def test_random_generator_produces_valid_length_matched_sequences():
    gen = RandomAminoAcidGenerator(reference_lengths=[len(s) for s in REFERENCE])
    sequences = gen.generate(num_samples=10, seed=0)

    assert len(sequences) == 10
    reference_lengths = {len(s) for s in REFERENCE}
    for seq in sequences:
        assert set(seq) <= VALID_AMINO_ACIDS
        assert len(seq) in reference_lengths


def test_random_generator_is_deterministic_given_a_seed():
    gen = RandomAminoAcidGenerator(reference_lengths=[len(s) for s in REFERENCE])
    assert gen.generate(num_samples=5, seed=42) == gen.generate(num_samples=5, seed=42)


def test_shuffled_generator_preserves_length_and_composition():
    gen = ShuffledReferenceGenerator(reference_sequences=REFERENCE)
    sequences = gen.generate(num_samples=len(REFERENCE), seed=0)

    assert sorted(len(s) for s in sequences) == sorted(len(s) for s in REFERENCE)
    assert sorted("".join(sequences)) == sorted("".join(REFERENCE))


def test_shuffled_generator_can_oversample_with_replacement():
    gen = ShuffledReferenceGenerator(reference_sequences=REFERENCE)
    sequences = gen.generate(num_samples=20, seed=0)

    assert len(sequences) == 20
    # every output must be a shuffle of some real reference sequence — with
    # only 3 references and 20 requested, that's only possible with reuse
    reference_compositions = {tuple(sorted(s)) for s in REFERENCE}
    for seq in sequences:
        assert tuple(sorted(seq)) in reference_compositions
