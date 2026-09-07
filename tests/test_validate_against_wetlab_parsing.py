"""Unit tests for validate_against_wetlab.py's pure-Python parsing logic.

Decoupled from tests/test_validate_against_wetlab.py's full-pipeline test,
which needs the (non-redistributable) wet-lab supplement file and so can
never run in public CI. `_parse_censored` has no such dependency and should
be tested wherever the `validate` extra is installed, regardless of whether
that data file happens to be present.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")  # validate_against_wetlab.py imports scipy at module level

from pharmamp.benchmark.validate_against_wetlab import _parse_censored


def test_parse_censored_handles_plain_numeric_values():
    assert _parse_censored("1.98") == 1.98
    assert _parse_censored(3.3435) == 3.3435


def test_parse_censored_strips_right_censoring_marker():
    assert _parse_censored(">128") == 128.0
    assert _parse_censored(">64") == 64.0


def test_parse_censored_strips_left_censoring_marker():
    assert _parse_censored("<0.5") == 0.5
