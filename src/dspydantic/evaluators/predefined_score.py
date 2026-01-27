"""PredefinedScoreEvaluator for using pre-computed scores/numbers/bool values as feedback."""

import threading
from typing import Any

from dspydantic.evaluators.config import BaseEvaluator, register_evaluator


class PredefinedScoreEvaluator:
    """Evaluator that uses pre-computed scores from a list.

    This evaluator pops scores from a provided list in order as examples are evaluated.
    Useful when you already have ground truth scores and don't want to recompute them.

    Supports:
    - Float scores (0.0-1.0): Used directly
    - Bool values: True → 1.0, False → 0.0
    - Numbers: Normalized to 0.0-1.0 range (assumes max is 100 if not specified)

    Thread-safe for parallel evaluation using thread-local storage.

    Examples:
        ```python
        # Float scores
        scores = [0.95, 0.87, 0.92, 1.0, 0.78]
        evaluator = PredefinedScoreEvaluator(config={"scores": scores})

        # Bool values
        bool_scores = [True, False, True, True]
        evaluator = PredefinedScoreEvaluator(config={"scores": bool_scores})

        # Numbers (normalized)
        numeric_scores = [95, 87, 92, 100]
        evaluator = PredefinedScoreEvaluator(config={"scores": numeric_scores, "max_value": 100})
        ```
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize PredefinedScoreEvaluator.

        Args:
            config: Configuration dictionary with:
                - "scores": List of scores (float, bool, or numbers)
                - "max_value": Optional max value for normalization (default: 100)
        """
        config = config or {}
        self.scores = config.get("scores", [])
        self.max_value = config.get("max_value", 100.0)

        if not isinstance(self.scores, list):
            raise ValueError("scores must be a list")

        # Thread-local storage for tracking which score to use
        self._local = threading.local()

    def _get_next_score(self) -> float:
        """Get the next score from the list, converting to float 0.0-1.0.

        Returns:
            Score between 0.0 and 1.0.
        """
        # Initialize thread-local index if not exists
        if not hasattr(self._local, "index"):
            self._local.index = 0

        # Check if we have more scores
        if self._local.index >= len(self.scores):
            return 0.0  # Default if list exhausted

        # Get score at current index
        score = self.scores[self._local.index]
        self._local.index += 1

        # Convert to float 0.0-1.0
        return self._normalize_score(score)

    def _normalize_score(self, score: Any) -> float:
        """Normalize a score to 0.0-1.0 range.

        Args:
            score: Score value (float, bool, or number).

        Returns:
            Normalized score between 0.0 and 1.0.
        """
        # Handle bool values
        if isinstance(score, bool):
            return 1.0 if score else 0.0

        # Handle None
        if score is None:
            return 0.0

        # Convert to float
        try:
            score_float = float(score)
        except (ValueError, TypeError):
            return 0.0

        # If already in 0.0-1.0 range, return as-is
        if 0.0 <= score_float <= 1.0:
            return score_float

        # Normalize assuming max_value
        normalized = score_float / self.max_value
        # Clamp to 0.0-1.0 range
        return max(0.0, min(1.0, normalized))

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using pre-defined score.

        This method ignores extracted/expected values and returns the next
        pre-defined score from the list.

        Args:
            extracted: The extracted value (ignored).
            expected: The expected value (ignored).
            input_data: Optional input data (ignored).
            field_path: Optional field path (ignored).

        Returns:
            Pre-defined score between 0.0 and 1.0.
        """
        return self._get_next_score()
