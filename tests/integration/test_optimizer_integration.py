"""Integration tests for PydanticOptimizer with real DSPy setup."""

import os
from unittest.mock import MagicMock

import dspy
import pytest
from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class Address(BaseModel):
    """Address model for integration testing."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")


class User(BaseModel):
    """User model with nested address for integration testing."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")
    email: str = Field(description="Email address")
    address: Address = Field(description="User address")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping integration test",
)
def test_optimizer_with_nested_model_and_prompts() -> None:
    """Integration test: Run optimizer with nested model and prompts."""
    # Configure DSPy first
    api_key = os.getenv("OPENAI_API_KEY")
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=api_key)
    dspy.configure(lm=lm)

    examples = [
        Example(
            text="John Doe, 30 years old, john@example.com, 123 Main St, New York, 10001",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "zip_code": "10001",
                },
            },
        ),
        Example(
            text="Jane Smith, 25, jane@example.com, 456 Oak Ave, Los Angeles, 90001",
            expected_output={
                "name": "Jane Smith",
                "age": 25,
                "email": "jane@example.com",
                "address": {
                    "street": "456 Oak Ave",
                    "city": "Los Angeles",
                    "zip_code": "90001",
                },
            },
        ),
    ]

    evaluation_calls: list[tuple[str | None, str | None]] = []

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        """Mock evaluation function that tracks calls."""
        evaluation_calls.append((optimized_system_prompt, optimized_instruction_prompt))
        # Return a mock score
        return 0.85

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information from text",
        instruction_prompt="Parse the following text and extract structured data",
        optimizer="miprov2zeroshot",
        num_threads=1,
        verbose=False,
    )

    # Verify optimizer is set up correctly
    assert optimizer.system_prompt == "Extract user information from text"
    assert optimizer.instruction_prompt == "Parse the following text and extract structured data"
    assert len(optimizer.field_descriptions) > 0
    assert "name" in optimizer.field_descriptions
    assert "address.street" in optimizer.field_descriptions

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == 2
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping integration test",
)
def test_optimizer_metric_function_integration() -> None:
    """Integration test: Verify metric function works with real DSPy predictions."""
    examples = [
        Example(
            text="John Doe, 30",
            expected_output={"name": "John Doe", "age": 30},
        )
    ]

    captured_prompts: list[tuple[str | None, str | None]] = []

    def evaluate_fn(
        example: Example,
        optimized_descriptions: dict[str, str],
        optimized_system_prompt: str | None,
        optimized_instruction_prompt: str | None,
    ) -> float:
        captured_prompts.append((optimized_system_prompt, optimized_instruction_prompt))
        return 0.9

    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user info",
        instruction_prompt="Parse text",
    )

    # Create a mock LM for the metric function
    mock_lm = MagicMock(spec=dspy.LM)

    metric = optimizer._create_metric_function(mock_lm)

    # Create a real DSPy prediction
    prediction = dspy.Prediction(
        optimized_system_prompt="Optimized system prompt",
        optimized_instruction_prompt="Optimized instruction prompt",
    )

    example = dspy.Example(
        input_data={"text": "John Doe, 30"},
        expected_output={"name": "John Doe", "age": 30},
    )

    score = metric(example, prediction)

    assert score == 0.9
    assert len(captured_prompts) == 1
    assert captured_prompts[0][0] == "Optimized system prompt"
    assert captured_prompts[0][1] == "Optimized instruction prompt"

