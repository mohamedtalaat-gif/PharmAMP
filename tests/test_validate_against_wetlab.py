"""Skipped unless the (non-redistributed) wet-lab supplement file is present
locally — see pharmamp/benchmark/validate_against_wetlab.py for the source
and download instructions.
"""

from __future__ import annotations

import pytest

from pharmamp.benchmark.validate_against_wetlab import DEFAULT_SUPPLEMENT_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_SUPPLEMENT_PATH.exists(),
    reason="wet-lab supplement (media-2.xlsx) not present locally",
)


def test_aggregation_and_solubility_correlate_with_measured_cytotoxicity():
    from pharmamp.benchmark.validate_against_wetlab import correlations, load_and_score

    df = load_and_score(DEFAULT_SUPPLEMENT_PATH)
    result = correlations(df).set_index(["metric", "against"])

    # Directional sanity check, not a pinned coefficient: AggregationPropensity
    # is "lower is better" and Solubility is "higher is better", so a real
    # correlation with real cytotoxicity (CC50) must run in opposite signs.
    agg_rho = result.loc[("AggregationPropensity", "CC50 (cytotoxicity)"), "spearman_rho"]
    sol_rho = result.loc[("Solubility", "CC50 (cytotoxicity)"), "spearman_rho"]
    assert agg_rho < 0
    assert sol_rho > 0
