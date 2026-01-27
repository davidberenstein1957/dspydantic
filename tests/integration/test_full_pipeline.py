"""Full pipeline integration test.

- Nested Pydantic models
- System prompt optimization
- Instruction prompt optimization
- Field description optimization

Run with: uv run pytest tests/integration/test_full_pipeline.py -v
"""

import os

import pytest
from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


@pytest.fixture
def examples():
    """Example inputs and expected outputs for the pipeline."""
    return [
        Example(
            text="John Doe, 30 years old, john@example.com, 123 Main St, New York, 10001",
            expected_output={
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com",
                "address": {"street": "123 Main St", "city": "New York", "zip_code": "10001"},
            },
        ),
        Example(
            text="Jane Smith, 25, jane@example.com, 456 Oak Ave, Los Angeles, 90001",
            expected_output={
                "name": "Jane Smith",
                "age": 25,
                "email": "jane@example.com",
                "address": {"street": "456 Oak Ave", "city": "Los Angeles", "zip_code": "90001"},
            },
        ),
        Example(
            text="Bob Johnson, 40, bob@example.com, 789 Pine Rd, Chicago, 60601",
            expected_output={
                "name": "Bob Johnson",
                "age": 40,
                "email": "bob@example.com",
                "address": {"street": "789 Pine Rd", "city": "Chicago", "zip_code": "60601"},
            },
        ),
    ]


class Address(BaseModel):
    """Address model."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP code")


class User(BaseModel):
    """User model with nested address."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age")
    email: str = Field(description="Email address")
    address: Address = Field(description="User address")


def evaluate_fn(
    example: Example,
    optimized_descriptions: dict[str, str],
    optimized_system_prompt: str | None,
    optimized_instruction_prompt: str | None,
) -> float:
    """Mock evaluation function for testing.

    In a real scenario, this would:
    1. Use the optimized descriptions and prompts with an LLM
    2. Extract data from example.input_data
    3. Compare with example.expected_output
    4. Return a score based on accuracy

    Args:
        example: The example with input_data and expected_output
        optimized_descriptions: Dictionary of optimized field descriptions
        optimized_system_prompt: Optimized system prompt (if provided)
        optimized_instruction_prompt: Optimized instruction prompt (if provided)

    Returns:
        Score between 0.0 and 1.0
    """
    # For this integration test, return a mock score
    # In production, you would actually call your LLM here
    return 0.85


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_full_pipeline(lm, examples) -> None:
    """Run full optimization pipeline with nested model and prompt optimization."""
    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information from unstructured text",
        instruction_prompt=(
            "Parse the following text and extract structured user data "
            "including name, age, email, and address"
        ),
        optimizer="miprov2zeroshot",
        num_threads=1,
        verbose=False,
    )

    assert "address.street" in optimizer.field_descriptions
    assert "address.city" in optimizer.field_descriptions
    assert "address.zip_code" in optimizer.field_descriptions

    dspy_examples = optimizer._prepare_dspy_examples()
    assert len(dspy_examples) == len(examples)
    assert hasattr(dspy_examples[0], "system_prompt")
    assert hasattr(dspy_examples[0], "instruction_prompt")
    assert dspy_examples[0].system_prompt == optimizer.system_prompt
    assert dspy_examples[0].instruction_prompt == optimizer.instruction_prompt

    result = optimizer.optimize()
    assert result.baseline_score >= 0
    assert result.optimized_score >= 0
    assert "improvement" in result.metrics

