"""Unit tests for app.py's pure logic functions, independent of Streamlit's
runtime. Previously untested — the only place this logic was exercised was
manually, by clicking through the live app.
"""

from __future__ import annotations

import pandas as pd

from app import flag_risk, score_sequences, validate_sequences

AGGREGATION_PRONE = "VQIVYK"
SOLUBLE_CONTROL = "KEKEKEKEKE"


def test_validate_sequences_normalizes_and_splits_valid_invalid():
    valid, invalid = validate_sequences(["  acdefghik  ", "ACDXFGHIK", "", "   ", "AC-DE"])

    assert valid == ["ACDEFGHIK"]
    assert invalid == ["ACDXFGHIK", "", "   ", "AC-DE"]


def test_flag_risk_marks_small_batches_as_not_applicable():
    df = score_sequences([AGGREGATION_PRONE])
    result = flag_risk(df)

    assert result["Developability flag"].iloc[0].startswith("N/A")


def test_flag_risk_flags_worst_quartile_in_a_larger_batch():
    sequences = [AGGREGATION_PRONE, SOLUBLE_CONTROL, "GLLAKLLKKLLKKLLKK", "FLPLIGRVLSGIL"]
    df = flag_risk(score_sequences(sequences))

    assert not (df["Developability flag"] == "OK").all()
    assert set(df["Developability flag"]) <= {"High aggregation risk", "Low solubility", "OK"}


def test_score_sequences_returns_expected_columns():
    df = score_sequences([AGGREGATION_PRONE])
    assert list(df.columns) == [
        "Sequence",
        "Length",
        "Net charge (pH 7.4)",
        "Mean hydrophobicity",
        "Aggregation propensity",
        "Solubility",
    ]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
