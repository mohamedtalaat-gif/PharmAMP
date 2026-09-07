import pytest

from pharmamp import AggregationPropensityMetric, SolubilityMetric

AGGREGATION_PRONE = "VQIVYK"  # tau-derived hexapeptide, well-documented amyloid-forming fragment
SOLUBLE_CONTROL = "KEKEKEKEKE"  # alternating charge, low hydrophobicity


def test_aggregation_metric_returns_metric_result():
    metric = AggregationPropensityMetric()
    result = metric([AGGREGATION_PRONE, SOLUBLE_CONTROL])
    assert result.value is not None
    assert result.deviation is not None
    assert metric.name == "AggregationPropensity"
    assert metric.objective == "minimize"


def test_aggregation_metric_ranks_known_amyloid_fragment_higher():
    metric = AggregationPropensityMetric()
    prone_score, control_score = metric.score([AGGREGATION_PRONE, SOLUBLE_CONTROL])
    assert prone_score > control_score


def test_solubility_metric_ranks_charged_sequence_higher():
    metric = SolubilityMetric()
    prone_score, control_score = metric.score([AGGREGATION_PRONE, SOLUBLE_CONTROL])
    assert control_score > prone_score
    assert metric.objective == "maximize"


def test_aggregation_metric_rejects_empty_sequence():
    with pytest.raises(ValueError):
        AggregationPropensityMetric().score([AGGREGATION_PRONE, ""])


def test_solubility_metric_rejects_empty_sequence():
    with pytest.raises(ValueError):
        SolubilityMetric().score([AGGREGATION_PRONE, ""])
