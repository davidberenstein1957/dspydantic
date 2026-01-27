"""String check evaluator for exact string matching."""

from typing import Any


class StringCheckEvaluator:
    """Evaluator that performs exact string matching.

    Config options:
        case_sensitive (bool): Whether comparison should be case-sensitive (default: True)
        strip_whitespace (bool): Whether to strip whitespace before comparison (default: True)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize StringCheckEvaluator.

        Args:
            config: Configuration dictionary with case_sensitive and strip_whitespace options.
        """
        self.config = config
        self.case_sensitive = config.get("case_sensitive", True)
        self.strip_whitespace = config.get("strip_whitespace", True)

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using exact string matching.

        Args:
            extracted: Extracted value.
            expected: Expected value.
            input_data: Optional input data (not used).
            field_path: Optional field path (not used).

        Returns:
            Score 1.0 if match, 0.0 otherwise.
        """
        extracted_str = str(extracted)
        expected_str = str(expected)

        if self.strip_whitespace:
            extracted_str = extracted_str.strip()
            expected_str = expected_str.strip()

        if not self.case_sensitive:
            extracted_str = extracted_str.lower()
            expected_str = expected_str.lower()

        return 1.0 if extracted_str == expected_str else 0.0
