"""Python code evaluator for custom callable evaluation."""

from typing import Any


class PythonCodeEvaluator:
    """Evaluator that uses a callable for custom evaluation logic.

    Use this when built-in evaluators don't match your requirements, such as
    domain-specific validation rules or complex business logic.

    Args:
        config: Configuration dictionary with:
            - function (Callable): Function that takes (extracted, expected, input_data, field_path)
              and returns a float score between 0.0 and 1.0.

    Raises:
        ValueError: If 'function' is not provided or not callable.
        RuntimeError: If the function raises an exception during evaluation.

    Example:
        >>> def age_evaluator(extracted, expected, input_data=None, field_path=None):
        ...     if extracted == expected:
        ...         return 1.0
        ...     diff = abs(int(extracted) - int(expected))
        ...     return max(0.0, 1.0 - (diff / 10))
        >>> evaluator = PythonCodeEvaluator(config={"function": age_evaluator})
        >>> evaluator.evaluate(30, 30)
        1.0
        >>> evaluator.evaluate(28, 30)  # Off by 2 years
        0.8
        >>> evaluator.evaluate(20, 30)  # Off by 10 years
        0.0
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
