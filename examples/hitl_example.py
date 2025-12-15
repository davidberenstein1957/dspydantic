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

This example demonstrates HITL with contact information extraction.
"""

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer, create_optimized_model


class ContactInfo(BaseModel):
    """Contact information extraction model."""

    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    company: str = Field(description="Company or organization name")
    title: str = Field(description="Job title or position")


def main():
    """Run the HITL evaluation example with contact information extraction."""
    print("=" * 60)
    print("Human-in-the-Loop (HITL) Evaluation Example")
    print("=" * 60)
    print("\nThis example demonstrates HITL evaluation functions.")
    print("During optimization, a GUI popup will appear for each evaluation.")
    print("You can review and edit the proposed output before continuing.")
    print("This example extracts contact information from business cards and emails.\n")

    # Create examples with contact information
    print("Creating contact information examples...")
    text_examples = [
        Example(
            text=(
                "Sarah Chen, Senior Software Engineer at Google. "
                "Email: sarah.chen@google.com, Phone: (650) 555-0123"
            ),
            expected_output=ContactInfo(
                name="Sarah Chen",
                email="sarah.chen@google.com",
                phone="(650) 555-0123",
                company="Google",
                title="Senior Software Engineer",
            ),
        ),
        Example(
            text=(
                "Michael Rodriguez - Product Manager at Microsoft. "
                "Contact: mrodriguez@microsoft.com, Tel: 206-555-0198"
            ),
            expected_output=ContactInfo(
                name="Michael Rodriguez",
                email="mrodriguez@microsoft.com",
                phone="206-555-0198",
                company="Microsoft",
                title="Product Manager",
            ),
        ),
        Example(
            text=(
                "Dr. Emily Watson, Chief Data Scientist, Amazon Web Services. "
                "Email: ewatson@aws.com, Phone: 206-555-0176"
            ),
            expected_output=ContactInfo(
                name="Dr. Emily Watson",
                email="ewatson@aws.com",
                phone="206-555-0176",
                company="Amazon Web Services",
                title="Chief Data Scientist",
            ),
        ),
    ]

    # Example 1: Contact examples with exact-hitl
    print("\n" + "=" * 60)
    print("Example 1: Contact Information with 'exact-hitl' evaluation")
    print("-" * 60)
    print("Score: 0.0 if you edit the output, 1.0 if you don't edit\n")

    optimizer_text = PydanticOptimizer(
        model=ContactInfo,
        examples=text_examples,
        evaluate_fn="exact-hitl",
        model_id="gpt-4o-mini",
        verbose=True,
    )

    print("Starting optimization with contact information examples...")
    print("A popup will appear for each evaluation. Review and edit as needed.\n")

    result_text = optimizer_text.optimize()

    print("\n" + "=" * 60)
    print("Results (Contact Information - exact-hitl)")
    print("=" * 60)
    print(f"Baseline score: {result_text.baseline_score:.2%}")
    print(f"Optimized score: {result_text.optimized_score:.2%}")
    print(f"Improvement: {result_text.metrics['improvement']:+.2%}")

    # Example 2: Contact examples with levenshtein-hitl
    print("\n\n" + "=" * 60)
    print("Example 2: Contact Information with 'levenshtein-hitl' evaluation")
    print("-" * 60)
    print("Score: Levenshtein similarity if you edit the output, 1.0 if you don't edit\n")

    optimizer_levenshtein = PydanticOptimizer(
        model=ContactInfo,
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
    optimized_model = create_optimized_model(ContactInfo, result_text.optimized_descriptions)
    print("\nCreated optimized model with improved field descriptions.")
    print(f"Model class name: {optimized_model.__name__}")
    print("You can now use this optimized model for better extraction accuracy.")


if __name__ == "__main__":
    main()
