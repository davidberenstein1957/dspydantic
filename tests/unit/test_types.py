"""Tests for types module."""

from pydantic import BaseModel, Field

from dspydantic.types import Example, OptimizationResult


class User(BaseModel):
    """User model for testing."""

    name: str = Field(description="User name")
    age: int = Field(description="User age")


def test_example_creation() -> None:
    """Test creating an Example instance."""
    example = Example(
        text="John Doe, 30",
        expected_output={"name": "John Doe", "age": 30},
    )
    assert example.input_data == {"text": "John Doe, 30"}
    assert example.expected_output == {"name": "John Doe", "age": 30}


def test_example_with_pydantic_model() -> None:
    """Test creating an Example with a Pydantic model as expected_output."""
    user = User(name="John Doe", age=30)
    example = Example(
        text="John Doe, 30",
        expected_output=user,
    )
    assert isinstance(example.expected_output, BaseModel)
    assert example.expected_output.name == "John Doe"
    assert example.expected_output.age == 30


def test_optimization_result_creation() -> None:
    """Test creating an OptimizationResult instance."""
    result = OptimizationResult(
        optimized_descriptions={"name": "Optimized name", "age": "Optimized age"},
        optimized_system_prompt=None,
        optimized_instruction_prompt=None,
        metrics={"average_score": 0.85, "improvement": 0.1},
        baseline_score=0.75,
        optimized_score=0.85,
    )
    assert result.optimized_descriptions == {"name": "Optimized name", "age": "Optimized age"}
    assert result.baseline_score == 0.75
    assert result.optimized_score == 0.85
    assert result.metrics["average_score"] == 0.85
