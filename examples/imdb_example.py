"""Text classification example - Sentiment analysis.

This example demonstrates how to optimize a Pydantic model for sentiment classification,
based on GLiNER2's text classification tutorial example.
It classifies product reviews as positive, negative, or neutral.
"""

from typing import Literal

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class SentimentClassification(BaseModel):
    """Sentiment classification model for product reviews."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the review"
    )


def main():
    """Run the sentiment classification optimization example."""
    print("=" * 60)
    print("Text Classification Example - Sentiment Analysis")
    print("=" * 60)
    print("\nThis example classifies product reviews as positive, negative, or neutral.")
    print("Based on GLiNER2's text classification tutorial.\n")

    # Create examples based on GLiNER2 tutorial
    examples = [
        Example(
            text="This laptop has amazing performance but terrible battery life!",
            expected_output={"sentiment": "negative"},
        ),
        Example(
            text="Absolutely love this product! It exceeded all my expectations.",
            expected_output={"sentiment": "positive"},
        ),
        Example(
            text="The product arrived on time and works as described.",
            expected_output={"sentiment": "neutral"},
        ),
        Example(
            text="Great features, but the price is way too high for what you get.",
            expected_output={"sentiment": "negative"},
        ),
        Example(
            text="Outstanding quality and excellent customer service. Highly recommend!",
            expected_output={"sentiment": "positive"},
        ),
        Example(
            text="It's okay, nothing special. Does the job but nothing more.",
            expected_output={"sentiment": "neutral"},
        ),
        Example(
            text="Worst purchase I've ever made. Complete waste of money.",
            expected_output={"sentiment": "negative"},
        ),
        Example(
            text="Perfect! Exactly what I was looking for. Worth every penny.",
            expected_output={"sentiment": "positive"},
        ),
    ]

    print(f"Created {len(examples)} examples")
    print("\nSample examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        text_preview = example.input_data.get("text", "")
        print(f"  Review: {text_preview}")
        print(f"  Expected sentiment: {example.expected_output['sentiment']}")

    # Create optimizer with system and instruction prompts
    optimizer = PydanticOptimizer(
        model=SentimentClassification,
        examples=examples,
        model_id="gpt-4o-mini",
        verbose=True,
        optimizer="miprov2zeroshot",
        system_prompt=(
            "You are an expert sentiment analysis assistant specializing in product review "
            "classification. You understand nuanced language, sarcasm, and contextual cues "
            "that indicate positive, negative, or neutral sentiment in written reviews. "
            "You can accurately distinguish between genuine praise and criticism even when "
            "reviews contain mixed signals."
        ),
        instruction_prompt=(
            "Classify the sentiment of the following product review as either 'positive', "
            "'negative', or 'neutral'. Consider the overall tone, language used, and the "
            "reviewer's opinion when making your classification."
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
    print("\nOptimized descriptions:")
    for field_path, description in result.optimized_descriptions.items():
        print(f"  {field_path}: {description}")


if __name__ == "__main__":
    main()
