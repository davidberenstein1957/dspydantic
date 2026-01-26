"""Levenshtein distance evaluator for fuzzy string matching."""

from typing import Any


class LevenshteinEvaluator:
    """Evaluator that uses Levenshtein distance for fuzzy string matching.

    Config options:
        threshold (float): Minimum similarity threshold (0-1, default: 0.0)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize LevenshteinEvaluator.

        Args:
            config: Configuration dictionary with threshold option.
        """
        self.config = config
        self.threshold = config.get("threshold", 0.0)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using Levenshtein distance.

        Args:
            extracted: Extracted value.
            expected: Expected value.
            input_data: Optional input data (not used).
            field_path: Optional field path (not used).

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        extracted_str = str(extracted).strip()
        expected_str = str(expected).strip()

        if extracted_str == expected_str:
            return 1.0

        max_len = max(len(extracted_str), len(expected_str))
        if max_len == 0:
            return 1.0

        distance = self._levenshtein_distance(extracted_str, expected_str)
        similarity = 1.0 - (distance / max_len)

        return max(0.0, similarity) if similarity >= self.threshold else 0.0
