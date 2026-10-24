"""The deploy gate's decision rule (src/gate.py): floor, comparison with the live model, tolerance."""
import pytest

from src.config import GATE_MIN_F1, F1_TOLERANCE
from src.gate import decide


def test_passes_when_it_ties_the_live_model():
    # Same data and seed give the same model: a tie must deploy, or nothing ever would
    passed, _ = decide(0.851, live_f1=0.851)
    assert passed


def test_passes_within_the_tolerance():
    passed, _ = decide(0.851 - F1_TOLERANCE / 2, live_f1=0.851)
    assert passed


def test_fails_when_worse_than_live_by_more_than_the_tolerance():
    passed, reason = decide(0.851 - 2 * F1_TOLERANCE, live_f1=0.851)
    assert not passed
    assert 'worse than the live model' in reason


def test_fails_below_the_floor_even_when_live_is_worse():
    # A tie with a bad live model must not pass
    passed, reason = decide(GATE_MIN_F1 - 0.01, live_f1=GATE_MIN_F1 - 0.02)
    assert not passed
    assert 'below the floor' in reason


@pytest.mark.parametrize('new_f1, expected', [
    pytest.param(GATE_MIN_F1 + 0.01, True, id='above the floor'),
    pytest.param(GATE_MIN_F1 - 0.01, False, id='below the floor'),
])
def test_without_a_live_model_only_the_floor_applies(new_f1, expected):
    passed, reason = decide(new_f1, live_f1=None)
    assert passed is expected
    if passed:
        assert 'no live model to compare' in reason
