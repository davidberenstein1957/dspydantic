"""Python code evaluator for custom callable evaluation."""

from typing import Any


class PythonCodeEvaluator:
    """Evaluator that uses a callable for custom evaluation logic.

    Config options:
        function (Callable): Callable/method to use for evaluation.
            Must accept (extracted, expected, input_data=None, field_path=None) and return float.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize PythonCodeEvaluator.

        Args:
            config: Configuration dictionary with 'function' key containing a callable.
        """
        self.config = config
        self.function = config.get("function")

        if self.function is None:
            raise ValueError("'function' must be provided for PythonCodeEvaluator")

        if not callable(self.function):
            raise ValueError("'function' must be a callable")

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using the provided callable.

        Args:
            extracted: Extracted value.
            expected: Expected value.
            input_data: Optional input data for context.
            field_path: Optional field path for context.

        Returns:
            Score between 0.0 and 1.0.
        """
        try:
            score = float(
                self.function(extracted, expected, input_data=input_data, field_path=field_path)
            )
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"Error executing Python code evaluator function: {e}") from e
