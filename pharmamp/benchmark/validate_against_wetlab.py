"""Correlate PharmAMP's developability scores against real wet-lab safety data.

Data source: Szymczak, Der Torossian Torres, Soares et al., "Designing
antimicrobials with programmable mechanism and safety" (bioRxiv, 2026,
doi:10.64898/2026.09.01.747572), supplementary file media-2.xlsx — 217
OmegAMP-generated peptides with measured HC50 (hemolysis), CC50
(mammalian cytotoxicity), and BeStSel circular-dichroism secondary
structure. Not redistributed here (CC-BY-ND preprint) — download it from
the bioRxiv supplementary materials and point this script at it.

This is a statistical validation of the metrics against independent
measurements, not a fitted calibration — the metrics' coefficients are
unchanged by running this.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from pharmamp import AggregationPropensityMetric, SolubilityMetric

DEFAULT_SUPPLEMENT_PATH = Path(__file__).resolve().parents[2] / "data" / "wetlab-supplement" / "media-2.xlsx"


def _parse_censored(value: str) -> float:
    """Parse values like '>128' (no effect detected up to the highest tested
    concentration) as the threshold itself — a defensible simplification for
    a rank correlation, not a substitute for proper censored-data regression.
    """
    return float(str(value).strip().lstrip("><"))


def load_and_score(supplement_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(supplement_path, sheet_name="Peptide metadata")[["short_name", "sequence"]]
    hc50 = pd.read_excel(supplement_path, sheet_name="HC50")
    cc50 = pd.read_excel(supplement_path, sheet_name="CC50")
    bestsel = pd.read_excel(supplement_path, sheet_name="BeStSel")

    df = (
        meta.merge(hc50, on="short_name", how="inner")
        .merge(cc50, on="short_name", how="inner")
        .merge(bestsel, on="short_name", how="inner")
        .dropna(subset=["sequence"])
        .copy()
    )

    df["HC50_censored"] = df["HC50"].astype(str).str.startswith(">")
    df["CC50_censored"] = df["CC50"].astype(str).str.startswith(">")
    df["HC50_num"] = df["HC50"].apply(_parse_censored)
    df["CC50_num"] = df["CC50"].apply(_parse_censored)
    df["beta_sheet_h2o"] = df["f_beta_anti_H2O"].fillna(0) + df["f_beta_par_H2O"].fillna(0)

    aggregation = AggregationPropensityMetric()
    solubility = SolubilityMetric()
    df["agg_score"] = aggregation.score(df["sequence"].tolist())
    df["sol_score"] = solubility.score(df["sequence"].tolist())

    return df


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    # censored_col is None for BeStSel pairs — that axis has no right-censoring;
    # for HC50/CC50 it names the column disclosing what fraction of the
    # correlation is computed against a ">threshold" ceiling tie rather than
    # a real measured value (see _parse_censored's docstring).
    pairs = [
        ("agg_score", "HC50_num", "AggregationPropensity", "HC50 (hemolysis)", "HC50_censored"),
        ("agg_score", "CC50_num", "AggregationPropensity", "CC50 (cytotoxicity)", "CC50_censored"),
        ("agg_score", "beta_sheet_h2o", "AggregationPropensity", "BeStSel beta-sheet", None),
        ("sol_score", "HC50_num", "Solubility", "HC50 (hemolysis)", "HC50_censored"),
        ("sol_score", "CC50_num", "Solubility", "CC50 (cytotoxicity)", "CC50_censored"),
        ("sol_score", "beta_sheet_h2o", "Solubility", "BeStSel beta-sheet", None),
    ]
    rows = []
    for x_col, y_col, x_name, y_name, censored_col in pairs:
        rho, p = stats.spearmanr(df[x_col], df[y_col], nan_policy="omit")
        pct_censored = round(float(df[censored_col].mean() * 100), 1) if censored_col else None
        rows.append(
            {
                "metric": x_name,
                "against": y_name,
                "spearman_rho": rho,
                "p_value": p,
                "n": len(df),
                "pct_censored": pct_censored,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-path", type=Path, default=DEFAULT_SUPPLEMENT_PATH)
    args = parser.parse_args()

    if not args.supplement_path.exists():
        raise FileNotFoundError(
            f"{args.supplement_path} not found. Download media-2.xlsx from the bioRxiv "
            "supplementary materials for doi:10.64898/2026.09.01.747572 and place it there, "
            "or pass --supplement-path."
        )

    df = load_and_score(args.supplement_path)
    print(correlations(df).to_string(index=False))


if __name__ == "__main__":
    main()
