"""Tests for the grade predictor. Run: pytest -q"""
import pytest

import grade_model as gm


@pytest.fixture(scope="module")
def bundle():
    return gm.train(use_previous_grades=True)


def test_prediction_in_valid_range(bundle):
    g = gm.predict_grade(bundle, {"G1": 12, "G2": 13, "studytime": 3, "failures": 0})
    assert 0 <= g <= 20


def test_better_previous_grades_give_higher_prediction(bundle):
    low = gm.predict_grade(bundle, {"G1": 5, "G2": 5})
    high = gm.predict_grade(bundle, {"G1": 16, "G2": 17})
    assert high > low


def test_model_beats_baseline(bundle):
    assert bundle["r2"] > 0.3


def test_works_without_previous_grades():
    b = gm.train(use_previous_grades=False)
    assert "G1" not in b["raw_columns"]
    assert 0 <= gm.predict_grade(b, {"studytime": 2}) <= 20
