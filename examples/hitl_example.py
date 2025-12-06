"""Human-in-the-Loop (HITL) evaluation example.

This example demonstrates how to use the human-in-the-loop evaluation functions
('exact-hitl' and 'levenshtein-hitl') to review and edit extracted outputs
during optimization.

When using HITL evaluation:
- A GUI popup will appear for each evaluation
- You can review the input text and proposed output
- You can edit the output JSON if needed
- The score reflects whether changes were made:
  - exact-hitl: 0.0 if edited, 1.0 if not edited
  - levenshtein-hitl: Levenshtein similarity if edited, 1.0 if not edited

This example demonstrates HITL with text inputs.
"""

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer, create_optimized_model


class UserProfile(BaseModel):
    """User profile extraction model."""

    name: str = Field(description="Full name of the user")
    email: str = Field(description="Email address")
    age: int = Field(description="Age in years")
    city: str = Field(description="City of residence")


def main():
    """Run the HITL evaluation example with text inputs."""
    print("=" * 60)
    print("Human-in-the-Loop (HITL) Evaluation Example")
    print("=" * 60)
    print("\nThis example demonstrates HITL evaluation functions.")
    print("During optimization, a GUI popup will appear for each evaluation.")
    print("You can review and edit the proposed output before continuing.")
    print("This example uses TEXT examples.\n")

    # Create examples with text input
    print("Creating text examples...")
    text_examples = [
        Example(
            text="John Doe, 30 years old, lives in New York. Contact: john.doe@example.com",
            expected_output=UserProfile(
                name="John Doe",
                email="john.doe@example.com",
                age=30,
                city="New York",
            ),
        ),
        Example(
            text="Jane Smith is 25 and resides in San Francisco. Email: jane.smith@email.com",
            expected_output=UserProfile(
                name="Jane Smith",
                email="jane.smith@email.com",
                age=25,
                city="San Francisco",
            ),
        ),
        Example(
            text="Bob Johnson, age 35, email bob@test.com, located in Chicago",
            expected_output=UserProfile(
                name="Bob Johnson",
                email="bob@test.com",
                age=35,
                city="Chicago",
            ),
        ),
    ]

    # Example 1: Text examples with exact-hitl
    print("\n" + "=" * 60)
    print("Example 1: Text Examples with 'exact-hitl' evaluation")
    print("-" * 60)
    print("Score: 0.0 if you edit the output, 1.0 if you don't edit\n")

    optimizer_text = PydanticOptimizer(
        model=UserProfile,
        examples=text_examples,
        evaluate_fn="exact-hitl",
        model_id="gpt-4o-mini",
        verbose=True,
    )

    print("Starting optimization with text examples...")
    print("A popup will appear for each evaluation. Review and edit as needed.\n")

    result_text = optimizer_text.optimize()

    print("\n" + "=" * 60)
    print("Results (Text Examples - exact-hitl)")
    print("=" * 60)
    print(f"Baseline score: {result_text.baseline_score:.2%}")
    print(f"Optimized score: {result_text.optimized_score:.2%}")
    print(f"Improvement: {result_text.metrics['improvement']:+.2%}")

    # Example 2: Text examples with levenshtein-hitl
    print("\n\n" + "=" * 60)
    print("Example 2: Text Examples with 'levenshtein-hitl' evaluation")
    print("-" * 60)
    print("Score: Levenshtein similarity if you edit the output, 1.0 if you don't edit\n")

    optimizer_levenshtein = PydanticOptimizer(
        model=UserProfile,
        examples=text_examples,
        evaluate_fn="levenshtein-hitl",
        model_id="gpt-4o-mini",
        verbose=True,
    )

    print("Starting optimization with levenshtein-hitl evaluation...")
    print("A popup will appear for each evaluation. Review and edit as needed.\n")

    result_levenshtein = optimizer_levenshtein.optimize()

    print("\n" + "=" * 60)
    print("Results (levenshtein-hitl)")
    print("=" * 60)
    print(f"Baseline score: {result_levenshtein.baseline_score:.2%}")
    print(f"Optimized score: {result_levenshtein.optimized_score:.2%}")
    print(f"Improvement: {result_levenshtein.metrics['improvement']:+.2%}")

    print("\n" + "=" * 60)
    print("Optimized Descriptions")
    print("=" * 60)
    for field_path, description in result_text.optimized_descriptions.items():
        print(f"\n{field_path}:")
        print(f"  {description}")

    # Show how to use the optimized model
    print("\n" + "=" * 60)
    print("Using the Optimized Model")
    print("=" * 60)
    optimized_model = create_optimized_model(UserProfile, result_text.optimized_descriptions)
    print("\nCreated optimized model with improved field descriptions.")
    print(f"Model class name: {optimized_model.__name__}")
    print("You can now use this optimized model for better extraction accuracy.")


if __name__ == "__main__":
    main()
