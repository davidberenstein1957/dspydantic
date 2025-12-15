"""Multi-label classification example - Product aspects.

This example demonstrates how to optimize a Pydantic model for multi-label classification,
based on GLiNER2's multi-label classification tutorial example.
It classifies product reviews across multiple aspects: camera, performance, battery, display, and price.
"""

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class ProductAspects(BaseModel):
    """Multi-label product aspect classification model."""

    aspects: list[str] = Field(
        default_factory=list,
        description=(
            "Product aspects mentioned in the review. "
            "Possible values: camera, performance, battery, display, price"
        ),
    )


def main():
    """Run the multi-label product aspect classification optimization example."""
    print("=" * 60)
    print("Multi-label Classification Example - Product Aspects")
    print("=" * 60)
    print("\nThis example classifies product reviews across multiple aspects:")
    print("- camera")
    print("- performance")
    print("- battery")
    print("- display")
    print("- price")
    print("\nBased on GLiNER2's multi-label classification tutorial.\n")

    # Create examples based on GLiNER2 tutorial
    examples = [
        Example(
            text="Great camera quality, decent performance, but poor battery life.",
            expected_output=ProductAspects(
                aspects=["camera", "performance", "battery"]
            ),
        ),
        Example(
            text="Amazing display and excellent performance, though the price is high.",
            expected_output=ProductAspects(
                aspects=["display", "performance", "price"]
            ),
        ),
        Example(
            text="The camera is outstanding and the battery lasts all day.",
            expected_output=ProductAspects(
                aspects=["camera", "battery"]
            ),
        ),
        Example(
            text="Poor display quality and terrible performance. Not worth the price.",
            expected_output=ProductAspects(
                aspects=["display", "performance", "price"]
            ),
        ),
        Example(
            text="Excellent camera, great performance, long battery life, and beautiful display.",
            expected_output=ProductAspects(
                aspects=["camera", "performance", "battery", "display"]
            ),
        ),
        Example(
            text="The price is reasonable but the battery drains too quickly.",
            expected_output=ProductAspects(
                aspects=["price", "battery"]
            ),
        ),
        Example(
            text="Outstanding display and camera, but performance could be better.",
            expected_output=ProductAspects(
                aspects=["display", "camera", "performance"]
            ),
        ),
        Example(
            text="Good value for the price, decent camera and performance.",
            expected_output=ProductAspects(
                aspects=["price", "camera", "performance"]
            ),
        ),
    ]

    print(f"Created {len(examples)} examples\n")
    print("Sample examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        text_preview = example.input_data.get("text", "")
        print(f"  Text: {text_preview}")
        expected = example.expected_output
        if isinstance(expected, dict):
            aspects = expected.get("aspects", [])
            print(f"  Expected aspects: {aspects}")

    # Create optimizer
    optimizer = PydanticOptimizer(
        model=ProductAspects,
        examples=examples,
        model_id="gpt-4o-mini",
        verbose=True,
        optimizer="miprov2zeroshot",
        system_prompt=(
            "You are an expert product review analysis assistant specializing in "
            "multi-label classification. You can identify multiple product aspects "
            "mentioned in reviews including camera, performance, battery, display, and price. "
            "You understand that a single review can mention multiple aspects simultaneously."
        ),
        instruction_prompt=(
            "Identify all product aspects mentioned in the provided review text. "
            "The aspects can be: camera, performance, battery, display, or price. "
            "A review can mention multiple aspects. Return all mentioned aspects in a list."
        ),
    )

    # Optimize
    print("\n" + "=" * 60)
    print("Starting optimization...")
    print("=" * 60)
    result = optimizer.optimize()

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Baseline score: {result.baseline_score:.2%}")
    print(f"Optimized score: {result.optimized_score:.2%}")
    print(f"Improvement: {result.metrics['improvement']:+.2%}")
    print("\nOptimized system prompt:")
    print(f"  {result.optimized_system_prompt}")
    print("\nOptimized instruction prompt:")
    print(f"  {result.optimized_instruction_prompt}")
    print("\nOptimized field descriptions:")
    for field_path, description in result.optimized_descriptions.items():
        print(f"\n  {field_path}:")
        print(f"    {description}")


if __name__ == "__main__":
    main()
