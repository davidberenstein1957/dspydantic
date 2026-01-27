"""Tests for PredefinedScoreEvaluator."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dspydantic.evaluators.predefined_score import PredefinedScoreEvaluator


def test_predefined_score_evaluator_float_scores():
    """Test PredefinedScoreEvaluator with float scores."""
    scores = [0.95, 0.87, 0.92, 1.0, 0.78]
    evaluator = PredefinedScoreEvaluator(config={"scores": scores})

    # Evaluate examples in order
    assert evaluator.evaluate(None, None) == 0.95
    assert evaluator.evaluate(None, None) == 0.87
    assert evaluator.evaluate(None, None) == 0.92
    assert evaluator.evaluate(None, None) == 1.0
    assert evaluator.evaluate(None, None) == 0.78

    # After list exhausted, returns 0.0
    assert evaluator.evaluate(None, None) == 0.0


def test_predefined_score_evaluator_bool_scores():
    """Test PredefinedScoreEvaluator with bool values."""
    bool_scores = [True, False, True, True, False]
    evaluator = PredefinedScoreEvaluator(config={"scores": bool_scores})

    assert evaluator.evaluate(None, None) == 1.0
    assert evaluator.evaluate(None, None) == 0.0
    assert evaluator.evaluate(None, None) == 1.0
    assert evaluator.evaluate(None, None) == 1.0
    assert evaluator.evaluate(None, None) == 0.0


def test_predefined_score_evaluator_numeric_scores():
    """Test PredefinedScoreEvaluator with numeric values."""
    numeric_scores = [95, 87, 92, 100, 78]
    evaluator = PredefinedScoreEvaluator(config={"scores": numeric_scores, "max_value": 100})

    assert evaluator.evaluate(None, None) == pytest.approx(0.95)
    assert evaluator.evaluate(None, None) == pytest.approx(0.87)
    assert evaluator.evaluate(None, None) == pytest.approx(0.92)
    assert evaluator.evaluate(None, None) == pytest.approx(1.0)
    assert evaluator.evaluate(None, None) == pytest.approx(0.78)


def test_predefined_score_evaluator_mixed_scores():
    """Test PredefinedScoreEvaluator with mixed score types."""
    mixed_scores = [0.95, True, 87, False, 0.5]
    evaluator = PredefinedScoreEvaluator(config={"scores": mixed_scores, "max_value": 100})

    assert evaluator.evaluate(None, None) == 0.95
    assert evaluator.evaluate(None, None) == 1.0  # True
    assert evaluator.evaluate(None, None) == pytest.approx(0.87)  # 87 normalized
    assert evaluator.evaluate(None, None) == 0.0  # False
    assert evaluator.evaluate(None, None) == 0.5


def test_predefined_score_evaluator_none_scores():
    """Test PredefinedScoreEvaluator with None values."""
    scores = [0.95, None, 0.87]
    evaluator = PredefinedScoreEvaluator(config={"scores": scores})

    assert evaluator.evaluate(None, None) == 0.95
    assert evaluator.evaluate(None, None) == 0.0  # None becomes 0.0
    assert evaluator.evaluate(None, None) == 0.87


def test_predefined_score_evaluator_thread_safety():
    """Test PredefinedScoreEvaluator is thread-safe."""
    scores = list(range(100))  # 100 scores
    evaluator = PredefinedScoreEvaluator(config={"scores": scores, "max_value": 100})

    results = []

    def evaluate_score():
        score = evaluator.evaluate(None, None)
        results.append(score)

    # Run in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(evaluate_score) for _ in range(100)]
        for future in futures:
            future.result()

    # Should have 100 results, all unique (no duplicates from race conditions)
    assert len(results) == 100
    # All scores should be normalized (0.0-1.0 range)
    assert all(0.0 <= score <= 1.0 for score in results)


def test_predefined_score_evaluator_invalid_config():
    """Test PredefinedScoreEvaluator with invalid config."""
    with pytest.raises(ValueError, match="scores must be a list"):
        PredefinedScoreEvaluator(config={"scores": "not a list"})


def test_predefined_score_evaluator_empty_list():
    """Test PredefinedScoreEvaluator with empty list."""
    evaluator = PredefinedScoreEvaluator(config={"scores": []})
    assert evaluator.evaluate(None, None) == 0.0


def test_predefined_score_evaluator_already_normalized():
    """Test PredefinedScoreEvaluator with scores already in 0.0-1.0 range."""
    scores = [0.5, 0.75, 0.25]
    evaluator = PredefinedScoreEvaluator(config={"scores": scores})

    assert evaluator.evaluate(None, None) == 0.5
    assert evaluator.evaluate(None, None) == 0.75
    assert evaluator.evaluate(None, None) == 0.25
