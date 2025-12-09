"""IMDB text classification example using stanfordnlp/imdb dataset.

This example demonstrates how to optimize a Pydantic model for sentiment classification
on the IMDB movie review dataset from HuggingFace. It showcases template functionality
by using instruction prompt templates with placeholders that are filled from example
text dictionaries.
"""

import random
from typing import Literal

from pydantic import BaseModel, Field

from dspydantic import Example, PydanticOptimizer


class SentimentClassification(BaseModel):
    """Sentiment classification model for movie reviews."""

    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the movie review, either positive or negative"
    )


def load_imdb_examples(num_examples: int = 10) -> list[Example]:
    """Load examples from the IMDB dataset.

    Ensures at least one example for each sentiment label (positive and negative).

    Args:
        num_examples: Number of examples to load (default: 10).

    Returns:
        List of Example objects with text and expected sentiment.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required. Install it with: uv pip install datasets"
        )

    # Load the IMDB dataset
    dataset = load_dataset("stanfordnlp/imdb", split="train")
    dataset_size = len(dataset)

    # Find indices for each label
    positive_indices: list[int] = []
    negative_indices: list[int] = []

    for idx in range(dataset_size):
        if dataset[idx]["label"] == 1:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)

    # Ensure at least one example for each label
    selected_indices: set[int] = set()
    if positive_indices:
        selected_indices.add(random.choice(positive_indices))
    if negative_indices:
        selected_indices.add(random.choice(negative_indices))

    # Fill remaining slots randomly
    remaining_needed = num_examples - len(selected_indices)
    if remaining_needed > 0:
        available_indices = set(range(dataset_size)) - selected_indices
        if available_indices:
            additional_indices = random.sample(
                list(available_indices), min(remaining_needed, len(available_indices))
            )
            selected_indices.update(additional_indices)

    # Convert to list and limit to num_examples
    selected_indices_list = list(selected_indices)[:num_examples]
    random.shuffle(selected_indices_list)

    # Build examples with template text dict
    examples = []
    for idx in selected_indices_list:
        item = dataset[idx]
        # Convert label (0=negative, 1=positive) to string
        sentiment = "positive" if item["label"] == 1 else "negative"
        review_text = item["text"]
        review_length = len(review_text.split())

        # Use text as dict for template formatting
        # The "review" key will be automatically extracted for input_data
        example = Example(
            text={
                "review": review_text,
                "review_length": str(review_length),
            },
            expected_output={"sentiment": sentiment},
        )
        examples.append(example)

    return examples


def main():
    """Run the IMDB text classification optimization example."""
    print("Loading IMDB dataset examples...")
    examples = load_imdb_examples(num_examples=10)

    print(f"Loaded {len(examples)} examples")
    print("\nSample examples:")
    for i, example in enumerate(examples[:3], 1):
        print(f"\nExample {i}:")
        text_dict = example.text_dict
        review_preview = text_dict.get("review", "")[:100] if text_dict else ""
        print(f"  Review preview: {review_preview}...")
        review_len = text_dict.get("review_length", "N/A")
        print(f"  Review length: {review_len} words")
        print(f"  Expected sentiment: {example.expected_output['sentiment']}")

    # Create optimizer with system and instruction prompts
    optimizer = PydanticOptimizer(
        model=SentimentClassification,
        examples=examples,
        model_id="gpt-4o-mini",  # Will use OPENAI_API_KEY from environment
        verbose=True,
        optimizer="bootstrapfewshot",
        system_prompt=(
            "You are an expert sentiment analysis assistant specializing in movie review "
            "classification. You understand nuanced language, sarcasm, and contextual cues "
            "that indicate positive or negative sentiment in written reviews. You can "
            "accurately distinguish between genuine praise and criticism even when reviews "
            "contain mixed signals."
        ),
        instruction_prompt="A review of a movie: {review}",
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

