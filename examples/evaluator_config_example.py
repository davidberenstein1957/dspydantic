"""Example demonstrating evaluator configuration system.

This example shows how to use the new evaluator configuration system with:
- String output optimization
- Per-field evaluator configuration
- Custom evaluator classes
- Different evaluator types
"""

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class User(BaseModel):
    """User model for testing."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")
    description: str = Field(description="User's bio or description")
    rating: float = Field(description="Quality rating from 0-1")


def example_string_output() -> None:
    """Example: Optimize prompts for string output evaluation."""
    print("=" * 60)
    print("Example 1: String Output Optimization")
    print("=" * 60)

    examples = [
        Example(text="Good response", expected_output="excellent"),
        Example(text="Bad response", expected_output="poor"),
        Example(text="Average response", expected_output="average"),
    ]

    # Auto-creates OutputModel with single "output" field
    optimizer = PydanticOptimizer(
        model=None,  # Will auto-create OutputModel
        examples=examples,
        model_id="gpt-4o-mini",
        evaluator_config={
            "default": "text_similarity",  # Use semantic similarity for strings
            "field_overrides": {},
        },
        verbose=True,
    )

    result = optimizer.optimize()
    print(f"\nOptimized score: {result.optimized_score:.2%}")
    print(f"Baseline score: {result.baseline_score:.2%}")


def example_per_field_evaluators() -> None:
    """Example: Use different evaluators for different fields."""
    print("\n" + "=" * 60)
    print("Example 2: Per-Field Evaluator Configuration")
    print("=" * 60)

    examples = [
        Example(
            text="John Doe, 30 years old, Software engineer passionate about AI",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "description": "Software engineer passionate about AI",
                "rating": 0.85,
            },
        ),
        Example(
            text="Jane Smith, 25, Data scientist working on ML models",
            expected_output={
                "name": "Jane Smith",
                "age": 25,
                "description": "Data scientist working on ML models",
                "rating": 0.90,
            },
        ),
    ]

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        model_id="gpt-4o-mini",
        evaluator_config={
            "default": {
                "type": "exact",
                "config": {"case_sensitive": False},
            },
            "field_overrides": {
                "name": {
                    "type": "exact",
                    "config": {"case_sensitive": True},  # Names must match exactly
                },
                "description": {
                    "type": "text_similarity",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "threshold": 0.7,  # Semantic similarity for descriptions
                    },
                },
                "rating": {
                    "type": "score_model_grader",
                    "config": {
                        "criteria": "Rate the quality of this rating on a scale of 0-1",
                        "temperature": 0.0,
                    },
                },
            },
        },
        verbose=True,
    )

    result = optimizer.optimize()
    print(f"\nOptimized score: {result.optimized_score:.2%}")
    print(f"Baseline score: {result.baseline_score:.2%}")


def example_custom_evaluator() -> None:
    """Example: Use a custom evaluator class."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Evaluator Class")
    print("=" * 60)

    class ThresholdEvaluator:
        """Custom evaluator that checks if values are within a threshold."""

        def __init__(self, config: dict) -> None:
            self.threshold = config.get("threshold", 0.1)

        def evaluate(
            self,
            extracted: float,
            expected: float,
            input_data: dict | None = None,
            field_path: str | None = None,
        ) -> float:
            """Check if extracted value is within threshold of expected."""
            diff = abs(extracted - expected)
            return 1.0 if diff <= self.threshold else max(0.0, 1.0 - (diff / expected))

    examples = [
        Example(
            text="Rating: 0.85",
            expected_output={"rating": 0.85},
        ),
        Example(
            text="Rating: 0.90",
            expected_output={"rating": 0.90},
        ),
    ]

    class RatingModel(BaseModel):
        """Simple rating model."""

        rating: float = Field(description="Rating value")

    optimizer = PydanticOptimizer(
        model=RatingModel,
        examples=examples,
        model_id="gpt-4o-mini",
        evaluator_config={
            "default": {
                "class": ThresholdEvaluator,
                "config": {"threshold": 0.05},  # Allow 5% difference
            },
        },
        verbose=True,
    )

    result = optimizer.optimize()
    print(f"\nOptimized score: {result.optimized_score:.2%}")
    print(f"Baseline score: {result.baseline_score:.2%}")


def example_python_code_evaluator() -> None:
    """Example: Use Python code evaluator for custom logic."""
    print("\n" + "=" * 60)
    print("Example 4: Python Code Evaluator")
    print("=" * 60)

    # Custom evaluation code
    eval_code = """
def evaluate(extracted, expected, input_data=None, field_path=None):
    # Custom logic: check if extracted age is within 2 years of expected
    if field_path == "age":
        diff = abs(extracted - expected)
        if diff == 0:
            return 1.0
        elif diff <= 2:
            return 0.8
        else:
            return max(0.0, 1.0 - (diff / 10))
    # For other fields, use exact match
    return 1.0 if extracted == expected else 0.0
"""

    examples = [
        Example(
            text="John Doe, 30 years old",
            expected_output={"name": "John Doe", "age": 30},
        ),
        Example(
            text="Jane Smith, 25",
            expected_output={"name": "Jane Smith", "age": 25},
        ),
    ]

    class SimpleUser(BaseModel):
        """Simple user model."""

        name: str = Field(description="User name")
        age: int = Field(description="User age")

    optimizer = PydanticOptimizer(
        model=SimpleUser,
        examples=examples,
        model_id="gpt-4o-mini",
        evaluator_config={
            "default": "exact",
            "field_overrides": {
                "age": {
                    "type": "python_code",
                    "config": {
                        "code": eval_code,
                        "function_name": "evaluate",
                        "sandbox": True,
                    },
                },
            },
        },
        verbose=True,
    )

    result = optimizer.optimize()
    print(f"\nOptimized score: {result.optimized_score:.2%}")
    print(f"Baseline score: {result.baseline_score:.2%}")


def main() -> None:
    """Run all examples."""
    print("\nEvaluator Configuration Examples")
    print("=" * 60)

    # Note: These examples require API keys and will make actual API calls
    # Uncomment the examples you want to run:

    # example_string_output()
    # example_per_field_evaluators()
    # example_custom_evaluator()
    # example_python_code_evaluator()

    print("\n" + "=" * 60)
    print("Examples defined. Uncomment in main() to run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
