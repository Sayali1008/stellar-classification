"""Feature engineering: the colors the model learns from, and the column selection step."""
import pandas as pd
import pytest

from src.config import FINAL_FEATURES
from src.features import add_colors, select_final_features

# Round numbers, so every color can be checked by eye
ONE_OBJECT = pd.DataFrame({'u': [22.0], 'g': [21.0], 'r': [20.5], 'i': [20.25], 'z': [20.0]})


def test_add_colors_computes_adjacent_filter_differences():
    colors = add_colors(ONE_OBJECT).iloc[0]
    assert colors['u_g'] == pytest.approx(1.0)
    assert colors['g_r'] == pytest.approx(0.5)
    assert colors['r_i'] == pytest.approx(0.25)
    assert colors['i_z'] == pytest.approx(0.25)


def test_add_colors_keeps_the_magnitudes_and_leaves_the_input_unchanged():
    original = ONE_OBJECT.copy()
    result = add_colors(ONE_OBJECT)
    assert list(result.columns) == ['u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z']
    # The pipeline passes the caller's DataFrame in; it must not gain columns as a side effect
    pd.testing.assert_frame_equal(ONE_OBJECT, original)


def test_select_final_features_returns_the_final_set_in_order():
    selected = select_final_features(add_colors(ONE_OBJECT))
    assert list(selected.columns) == FINAL_FEATURES


def test_select_final_features_accepts_another_feature_set():
    # The comparison scripts pass their own lists; the order given is the order returned
    selected = select_final_features(add_colors(ONE_OBJECT), features=['r', 'u_g'])
    assert list(selected.columns) == ['r', 'u_g']
