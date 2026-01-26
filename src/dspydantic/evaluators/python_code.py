"""Python code evaluator for custom code execution."""

from pathlib import Path
from typing import Any


class PythonCodeEvaluator:
    """Evaluator that executes custom Python code for evaluation.

    Config options:
        code (str): Python code string to execute
        code_file (str | Path): Path to Python file with evaluation function
        function_name (str): Name of function to call (default: "evaluate")
        sandbox (bool): Use sandboxed execution (default: True)
        allowed_modules (list[str]): Modules allowed in sandbox
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize PythonCodeEvaluator.

        Args:
            config: Configuration dictionary with code, code_file, function_name,
                sandbox, allowed_modules options.
        """
        self.config = config
        self.code = config.get("code")
        self.code_file = config.get("code_file")
        self.function_name = config.get("function_name", "evaluate")
        self.sandbox = config.get("sandbox", True)
        self.allowed_modules = config.get("allowed_modules", [])

        if not self.code and not self.code_file:
            raise ValueError("Either 'code' or 'code_file' must be provided for PythonCodeEvaluator")

        if self.code and self.code_file:
            raise ValueError("Cannot provide both 'code' and 'code_file'")

    def _load_code(self) -> str:
        """Load code from file or return provided code."""
        if self.code_file:
            path = Path(self.code_file)
            if not path.exists():
                raise FileNotFoundError(f"Code file not found: {self.code_file}")
            return path.read_text()
        return self.code or ""

    def evaluate(
        self,
        extracted: Any,
        expected: Any,
        input_data: dict[str, Any] | None = None,
        field_path: str | None = None,
    ) -> float:
        """Evaluate using custom Python code.

        Args:
            extracted: Extracted value.
            expected: Expected value.
            input_data: Optional input data for context.
            field_path: Optional field path for context.

        Returns:
            Score between 0.0 and 1.0.
        """
        code = self._load_code()

        # Prepare execution context
        context: dict[str, Any] = {
            "extracted": extracted,
            "expected": expected,
            "input_data": input_data or {},
            "field_path": field_path,
        }

        if self.sandbox:
            # Restricted execution environment
            restricted_builtins = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bool": bool,
                    "dict": dict,
                    "float": float,
                    "int": int,
                    "len": len,
                    "list": list,
                    "max": max,
                    "min": min,
                    "range": range,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "type": type,
                    "zip": zip,
                }
            }

            # Add allowed modules
            for module_name in self.allowed_modules:
                try:
                    restricted_builtins[module_name] = __import__(module_name)
                except ImportError:
                    pass

            context.update(restricted_builtins)
        else:
            # Full execution environment (use with caution)
            import builtins

            context["__builtins__"] = builtins

        # Execute code
        try:
            exec(code, context)

            # Call the evaluation function
            if self.function_name not in context:
                raise ValueError(
                    f"Function '{self.function_name}' not found in code. "
                    f"Available names: {list(context.keys())}"
                )

            evaluate_func = context[self.function_name]
            if not callable(evaluate_func):
                raise ValueError(f"'{self.function_name}' is not callable")

            score = float(evaluate_func(extracted, expected, input_data=input_data, field_path=field_path))
            return max(0.0, min(1.0, score))
        except Exception as e:
            raise RuntimeError(f"Error executing Python code evaluator: {e}") from e
