"""Streamlit tool for scoring peptide developability.

Takes sequences (pasted or as a FASTA upload) and reports aggregation-propensity
and solubility scores alongside basic physicochemical properties, so a batch of
AMP candidates can be triaged before anyone looks at wet-lab formulation work.
"""

from io import StringIO

import pandas as pd
import streamlit as st
from Bio import SeqIO

from pharmamp import AggregationPropensityMetric, SolubilityMetric
from pharmamp.metrics.descriptors import AMINO_ACIDS, mean_hydrophobicity, net_charge

VALID_AMINO_ACIDS = set(AMINO_ACIDS)


def read_sequences(fasta_file, text_input) -> list[str]:
    if fasta_file is not None:
        try:
            content = fasta_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.warning("Could not read that FASTA file as UTF-8 text.")
            return []
        return [str(record.seq) for record in SeqIO.parse(StringIO(content), "fasta")]
    if text_input:
        return [line.strip() for line in text_input.splitlines() if line.strip()]
    return []


def validate_sequences(sequences: list[str]) -> tuple[list[str], list[str]]:
    valid, invalid = [], []
    for seq in sequences:
        cleaned = seq.strip().upper()
        if cleaned and set(cleaned) <= VALID_AMINO_ACIDS:
            valid.append(cleaned)
        else:
            invalid.append(seq)
    return valid, invalid


def score_sequences(sequences: list[str]) -> pd.DataFrame:
    aggregation = AggregationPropensityMetric()
    solubility = SolubilityMetric()
    rows = []
    for seq, agg_score, sol_score in zip(
        sequences, aggregation.score(sequences), solubility.score(sequences), strict=True
    ):
        rows.append(
            {
                "Sequence": seq,
                "Length": len(seq),
                "Net charge (pH 7.4)": round(net_charge(seq), 2),
                "Mean hydrophobicity": round(mean_hydrophobicity(seq), 2),
                "Aggregation propensity": round(agg_score, 3),
                "Solubility": round(sol_score, 3),
            }
        )
    return pd.DataFrame(rows)


MIN_BATCH_FOR_RELATIVE_FLAGGING = 4  # below this, a quartile-of-the-batch flag isn't meaningful


def flag_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Flags each sequence relative to the rest of *this batch* — the worst
    quartile submitted together, not an absolute clinical threshold. With
    too few sequences for a quartile to mean anything, every flag would
    trivially compare a value to itself and read as always "OK", so those
    batches are marked as not applicable instead."""
    if len(df) < MIN_BATCH_FOR_RELATIVE_FLAGGING:
        df["Developability flag"] = f"N/A (need ≥{MIN_BATCH_FOR_RELATIVE_FLAGGING} sequences to compare)"
        return df

    agg_threshold = df["Aggregation propensity"].quantile(0.75)
    sol_threshold = df["Solubility"].quantile(0.25)
    df["Developability flag"] = [
        "High aggregation risk"
        if agg > agg_threshold
        else "Low solubility"
        if sol < sol_threshold
        else "OK"
        for agg, sol in zip(df["Aggregation propensity"], df["Solubility"], strict=True)
    ]
    return df


def app():
    st.title("PharmAMP: Peptide Developability Scoring")
    st.caption(
        "Scores candidate antimicrobial peptides for aggregation propensity and "
        "aqueous solubility — the pharmaceutical-developability side of AMP design "
        "that potency-focused generators don't cover."
    )

    col1, col2 = st.columns(2)
    with col1:
        fasta_file = st.file_uploader("Upload a .fasta file", type=["fasta", "fa"])
    with col2:
        text_input = st.text_area("Or paste sequences, one per line")

    if st.button("Score sequences"):
        sequences = read_sequences(fasta_file, text_input)
        if not sequences:
            st.warning("Upload a FASTA file or paste at least one sequence.")
            return

        valid, invalid = validate_sequences(sequences)
        if invalid:
            st.warning(f"Skipped {len(invalid)} sequence(s) with non-standard amino acids.")
        if not valid:
            st.error("No valid sequences to score.")
            return

        with st.spinner("Scoring..."):
            df = flag_risk(score_sequences(valid))

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results (CSV)",
            data=df.to_csv(index=False),
            file_name="pharmamp_developability_scores.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    st.set_page_config(page_title="PharmAMP", page_icon="🧪", layout="wide")
    app()
