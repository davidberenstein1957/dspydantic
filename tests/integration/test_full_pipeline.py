"""Full pipeline integration test script.

This script runs a complete optimization pipeline with:
- Nested Pydantic models
- System prompt optimization
- Instruction prompt optimization
- Field description optimization

Run with: uv run python tests/integration/test_full_pipeline.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import dspydantic
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


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


def main() -> None:
    """Run the full pipeline integration test."""
    print("=" * 70)
    print("Full Pipeline Integration Test")
    print("=" * 70)
    print("\nThis test runs a complete optimization pipeline with:")
    print("  - Nested Pydantic models (User with Address)")
    print("  - System prompt optimization")
    print("  - Instruction prompt optimization")
    print("  - Field description optimization")
    print()

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not set.")
        print("   This test requires an API key to run.")
        print("   Set it with: export OPENAI_API_KEY=your-key")
        print()
        print("   Skipping actual optimization (will test setup only)...")
        api_key = None
    else:
        print("✓ OPENAI_API_KEY found")
        print()

    # Prepare examples
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
        Example(
            text="Bob Johnson, 40, bob@example.com, 789 Pine Rd, Chicago, 60601",
            expected_output={
                "name": "Bob Johnson",
                "age": 40,
                "email": "bob@example.com",
                "address": {
                    "street": "789 Pine Rd",
                    "city": "Chicago",
                    "zip_code": "60601",
                },
            },
        ),
    ]

    print(f"Prepared {len(examples)} examples")
    print()

    # Create optimizer
    print("Creating optimizer...")
    optimizer = PydanticOptimizer(
        model=User,
        examples=examples,
        evaluate_fn=evaluate_fn,
        system_prompt="Extract user information from unstructured text",
        instruction_prompt="Parse the following text and extract structured user data including name, age, email, and address",
        model_id="gpt-4o",
        api_key=api_key,
        optimizer_type="miprov2zeroshot",
        num_threads=1,
        verbose=True,
    )

    print("✓ Optimizer created")
    print(f"  - Model: {optimizer.model.__name__}")
    print(f"  - System prompt: {optimizer.system_prompt[:50]}...")
    print(f"  - Instruction prompt: {optimizer.instruction_prompt[:50]}...")
    print(f"  - Fields to optimize: {len(optimizer.field_descriptions)}")
    print()

    # Verify field descriptions include nested fields
    print("Field descriptions:")
    for field_path, description in optimizer.field_descriptions.items():
        print(f"  - {field_path}: {description[:50]}...")
    print()

    # Verify nested fields are present
    assert "address.street" in optimizer.field_descriptions
    assert "address.city" in optimizer.field_descriptions
    assert "address.zip_code" in optimizer.field_descriptions
    print("✓ Nested field descriptions detected")
    print()

    # Verify examples are prepared correctly
    dspy_examples = optimizer._prepare_dspy_examples()
    print(f"✓ Prepared {len(dspy_examples)} DSPy examples")
    assert len(dspy_examples) == len(examples)

    # Verify prompts are included in examples
    example = dspy_examples[0]
    assert hasattr(example, "system_prompt")
    assert hasattr(example, "instruction_prompt")
    assert example.system_prompt == optimizer.system_prompt
    assert example.instruction_prompt == optimizer.instruction_prompt
    print("✓ Prompts included in examples")
    print()

    if api_key:
        print("Running optimization...")
        print("(This may take a few minutes)")
        print()

        try:
            result = optimizer.optimize()

            print()
            print("=" * 70)
            print("Optimization Results")
            print("=" * 70)
            print(f"Baseline score: {result.baseline_score:.2%}")
            print(f"Optimized score: {result.optimized_score:.2%}")
            print(f"Improvement: {result.metrics['improvement']:+.2%}")
            print()

            if result.optimized_system_prompt:
                print("Optimized System Prompt:")
                print(f"  {result.optimized_system_prompt}")
                print()

            if result.optimized_instruction_prompt:
                print("Optimized Instruction Prompt:")
                print(f"  {result.optimized_instruction_prompt}")
                print()

            print("Optimized Field Descriptions:")
            for field_path, description in result.optimized_descriptions.items():
                print(f"  {field_path}: {description}")
            print()

            print("=" * 70)
            print("✓ Integration test completed successfully!")
            print("=" * 70)

        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            raise
    else:
        print("=" * 70)
        print("✓ Setup verification completed successfully!")
        print("=" * 70)
        print("\nTo run the full optimization, set OPENAI_API_KEY and run again.")


if __name__ == "__main__":
    main()
